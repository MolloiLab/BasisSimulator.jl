# Docs coverage: every exported name has a docstring, and the API reference's section map
# (docs/src/api_sections.jl, the same file the generated /api/ page uses) places every one of
# them. Dependency-free: the map is plain Julia data plus one resolver function.
#
# Standalone: julia --project=. -e 'using Test, BasisSimulator; const BS = BasisSimulator; include("test/docs.jl")'

const _API_MAP_FILE = joinpath(pkgdir(BS), "docs", "src", "api_sections.jl")
const _APIMap = Module(:_APIMap)
Base.include(_APIMap, _API_MAP_FILE)

const _EXPORTED = sort!(filter(n -> n !== nameof(BS), names(BS)))

# `(path relative to src/, line)` of every docstring of `n`.
function _doc_locations(n::Symbol)
    md = get(Base.Docs.meta(BS), Base.Docs.Binding(BS, n), nothing)
    md === nothing && return Tuple{String, Int}[]
    [_APIMap.api_doc_location(md.docs[s].data) for s in md.order]
end

@testset "every exported name has a docstring" begin
    undocumented = [n for n in _EXPORTED if !Base.Docs.hasdoc(BS, n)]
    @test undocumented == Symbol[]
end

@testset "API page section map" begin
    section_ids = Set(s.id for s in _APIMap.API_SECTIONS)
    @test "other" in section_ids

    # Every exported name lands in a real section, not the "Other" catch-all. This is exactly the
    # resolution the page performs, so a name added in a new, unmapped file fails here.
    unplaced = String[]
    for n in _EXPORTED
        locs = _doc_locations(n)
        path = isempty(locs) ? nothing : first(locs)[1]
        _APIMap.api_section(string(n), path) == "other" && push!(unplaced, "$n ($(something(path, "no docstring")))")
    end
    @test unplaced == String[]

    # Every source file holding an exported name's docstring is covered by the file map itself
    # (exact file or folder prefix), whatever the per-name overrides say — a new file must be
    # given a section deliberately.
    files = sort!(unique(p for n in _EXPORTED for (p, _) in _doc_locations(n) if p !== nothing))
    covered(p) = haskey(_APIMap.API_FILE_SECTION, p) ||
        any(k -> endswith(k, "/") && startswith(p, k), keys(_APIMap.API_FILE_SECTION))
    @test filter(!covered, files) == String[]

    # The map points only at sections that exist and at names / files that still exist.
    all_targets = [values(_APIMap.API_NAME_SECTION)..., values(_APIMap.API_PREFIX_SECTION)...,
                   values(_APIMap.API_FILE_SECTION)...]
    @test filter(!in(section_ids), all_targets) == String[]
    exported = Set(string.(_EXPORTED))
    @test filter(!in(exported), collect(keys(_APIMap.API_NAME_SECTION))) == String[]
    @test filter(!in(exported), _APIMap.API_LEADS) == String[]
    src = joinpath(pkgdir(BS), "src")
    srcfiles = [replace(relpath(joinpath(r, f), src), '\\' => '/') for (r, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")]
    stale = [k for k in keys(_APIMap.API_FILE_SECTION) if !any(f -> f == k || (endswith(k, "/") && startswith(f, k)), srcfiles)]
    @test stale == String[]
end
