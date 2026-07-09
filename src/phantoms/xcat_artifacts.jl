"""
    src/phantoms/xcat_artifacts.jl

Download + load the voxelized XCAT chest phantoms published by the XCIST project
(github.com/xcist/phantoms-voxelized, BSD-3-Clause) as ready-to-simulate
[`Phantom`](@ref)s.

Four phantoms are available (`:female_slab`, `:female_chest`, `:male_slab`,
`:male_chest`).  `download_xcat_*` fetches the phantom into Julia's standard
content-addressed artifact store (`~/.julia/artifacts/`) — downloading only if
it is not already there, verifying the download by SHA-256 — and returns the
directory.  `load_xcat_*` parses that directory into a labeled `Phantom` with an
editable material dictionary.

The on-disk XCIST format is one `int8` volume-fraction map per material, each
cropped to its own bounding box and positioned by an isocenter-relative offset
(see `_parse_xcist_voxelized`).  For the XCAT phantoms the fractions are binary
(one material per voxel), so the labeled conversion is exact.

Please cite the phantom source when you use these — see [`xcat_citation`](@ref).

BasisSim-original loader; phantom data © the XCIST authors (Segars XCAT), BSD-3.
"""

# =============================================================================
# Registry — the 4 XCAT phantoms in xcist/phantoms-voxelized
# =============================================================================

const _XCAT_BASE_URL = "https://raw.githubusercontent.com/xcist/phantoms-voxelized/HEAD"

struct XCATPhantomEntry
    zipname::String        # file in the phantoms-voxelized repo
    sha256::String         # SHA-256 of the .zip (download integrity)
    tree_hash::String      # git-tree-sha1 of the extracted tree (artifact store key)
    matrix::NTuple{3, Int} # nominal (nx, ny, nz) from the phantom name
    desc::String
end

const XCAT_REGISTRY = Dict{Symbol, XCATPhantomEntry}(
    :female_slab => XCATPhantomEntry(
        "Adult_Female_50percentile_Chest_Phantom_slab_400_1650x1050x1.zip",
        "99229613a794100bfa1d4fe22393596d904696cc3199d329dcdba9fcb62af363",
        "48a5b0b4de4f204b3b8f0ce626ac8a147a027393",
        (1650, 1050, 1), "Adult female 50th-percentile chest — single slab (slice 400)",
    ),
    :female_chest => XCATPhantomEntry(
        "Adult_Female_50percentile_Chest_Phantom_1650x1050x880.zip",
        "6db9aa8e3430a1d72595a9c4ee467c2360958e9ed782d4a22fbb52de4c317693",
        "8f1da0e052498c4de7de5b5973559f3cb8664d9d",
        (1650, 1050, 880), "Adult female 50th-percentile chest — full volume",
    ),
    :male_slab => XCATPhantomEntry(
        "Adult_Male_50percentile_Chest_Phantom_slab_400_1700x1050x1.zip",
        "c2bc6898daed9719ba9203a4e3ed6fa9ca51c7dcb22bca070be94264c7fa539d",
        "c40b8751aa72f236a0404192984de4578b56b066",
        (1700, 1050, 1), "Adult male 50th-percentile chest — single slab (slice 400)",
    ),
    :male_chest => XCATPhantomEntry(
        "Adult_Male_50percentile_Chest_Phantom_1700x1050x900.zip",
        "5a04b384fb126f1cd944908b6a3dcfbf6b43b0384039eb0c74cd9b53d1185b32",
        "ce69351073fc478cbcf2b44b503c2480c18b6e0d",
        (1700, 1050, 900), "Adult male 50th-percentile chest — full volume",
    ),
)

"""
    xcat_phantoms() -> Vector{Symbol}

List the XCAT phantoms available to [`download_xcat_phantom`](@ref) /
[`load_xcat_phantom`](@ref).
"""
xcat_phantoms() = sort!(collect(keys(XCAT_REGISTRY)))

# =============================================================================
# XCIST material name → XrayAttenuation material
# =============================================================================

"""
    xcat_default_materials() -> Dict{String, XA.Material}

The default mapping from XCIST/NCAT material names to XrayAttenuation materials
used by [`load_xcat_phantom`](@ref).  Copy, edit, and pass back via the
`materials` keyword to override (e.g. to dope the blood pool with iodine).
"""
xcat_default_materials() = Dict{String, XA.Material}(
    "ncat_water"     => XA.Materials.water,
    "ncat_muscle"    => XA.Materials.ncat_muscle,
    "ncat_lung"      => XA.Materials.ncat_lung,
    "ncat_dry_spine" => XA.Materials.ncat_spine,
    "ncat_dry_rib"   => XA.Materials.ncat_rib,
    "ncat_adipose"   => XA.Materials.ncat_fat,
    "ncat_blood"     => XA.Materials.ncat_blood,
    "ncat_heart"     => XA.Materials.ncat_heart,
    "ncat_cartilage" => XA.Materials.cartilage,
    "ncat_liver"     => XA.Materials.ncat_liver,
    "ncat_intestine" => XA.Materials.intestine,
    "ncat_spleen"    => XA.Materials.ncat_spleen,
)

# =============================================================================
# Citation
# =============================================================================

"""
    xcat_citation() -> String

Attribution for the XCAT voxelized phantoms.  Printed on first download.
"""
xcat_citation() = """
XCAT voxelized chest phantom — please cite:
  • W. P. Segars et al., "4D XCAT phantom for multimodality imaging research,"
    Medical Physics 37(9):4902–4915, 2010.
  • M. Wu et al., "XCIST — an open access x-ray/CT simulation toolkit,"
    Physics in Medicine & Biology 67, 2022 (PMC10151073).
  Source: github.com/xcist/phantoms-voxelized (BSD-3-Clause).
"""

# =============================================================================
# Download → artifact store
# =============================================================================

"""
    download_xcat_phantom(name::Symbol; path=nothing, quiet=false) -> String

Return a directory holding the XCIST voxelized files for XCAT phantom `name`
(one of [`xcat_phantoms`](@ref)).

Resolution order:
1. If `path` is given, it is returned unchanged (bring-your-own copy — point at
   the extracted phantom folder).
2. If the phantom is already in Julia's artifact store (`~/.julia/artifacts/`),
   its cached path is returned — no download.
3. Otherwise the `.zip` is downloaded from the XCIST repository (with a progress
   log and a one-time citation), verified by SHA-256, extracted, and installed
   into the artifact store keyed by its git-tree-sha1.

The full-volume phantoms are ~65 MB; the slabs are <100 KB.
"""
function download_xcat_phantom(name::Symbol; path::Union{Nothing, AbstractString} = nothing, quiet::Bool = false)
    path !== nothing && return String(path)
    haskey(XCAT_REGISTRY, name) ||
        error("unknown XCAT phantom :$name — available: $(xcat_phantoms())")
    entry = XCAT_REGISTRY[name]
    hash = Base.SHA1(entry.tree_hash)

    # (2) already installed?
    Artifacts.artifact_exists(hash) && return Artifacts.artifact_path(hash)

    # (3) download → verify → extract → install
    quiet || @info """Downloading XCAT phantom :$name ($(entry.desc)) into the Julia artifact store…

    $(xcat_citation())"""

    dest = Artifacts.artifact_path(hash)
    artifacts_dir = dirname(dest)
    mkpath(artifacts_dir)
    # stage on the SAME filesystem as the store so the final install is a rename
    staging = mktempdir(artifacts_dir)
    try
        zippath = joinpath(staging, entry.zipname)
        url = "$(_XCAT_BASE_URL)/$(entry.zipname)"
        Downloads.download(url, zippath; progress = _xcat_progress(name, quiet))
        quiet || print(stderr, "\n")

        got = open(io -> bytes2hex(SHA.sha256(io)), zippath)
        got == entry.sha256 ||
            error("SHA-256 mismatch for XCAT :$name\n  expected $(entry.sha256)\n  got      $got")

        extracted = joinpath(staging, "extracted")
        mkpath(extracted)
        run(pipeline(`$(p7zip_jll.p7zip()) x -y -o$(extracted) $(zippath)`; stdout = devnull))

        if Artifacts.artifact_exists(hash)          # a concurrent download won the race
            return Artifacts.artifact_path(hash)
        end
        mv(extracted, dest)                          # same-fs install
    finally
        rm(staging; recursive = true, force = true)
    end
    return dest
end

# Downloads.jl progress callback: percent logger to stderr, emitting only when
# the integer percentage changes (Downloads calls back very frequently).
function _xcat_progress(name::Symbol, quiet::Bool)
    quiet && return (total, now) -> nothing
    last = Ref(-1)
    return function (total, now)
        total == 0 && return
        pct = round(Int, 100 * now / total)
        pct == last[] && return
        last[] = pct
        print(stderr, "\r  :$name  $(lpad(pct, 3))%  ($(round(now / 1e6, digits = 1)) / $(round(total / 1e6, digits = 1)) MB)   ")
        flush(stderr)
    end
end

# Convenience wrappers -------------------------------------------------------
for (fn, sym) in (
        (:download_xcat_female_slab, :female_slab), (:download_xcat_female_chest, :female_chest),
        (:download_xcat_male_slab, :male_slab), (:download_xcat_male_chest, :male_chest),
    )
    @eval $fn(; kwargs...) = download_xcat_phantom($(QuoteNode(sym)); kwargs...)
end

# =============================================================================
# Parse the XCIST voxelized format → labeled Phantom
# =============================================================================

# Recursively find the single .json phantom header under `dir`.
function _find_xcist_json(dir::AbstractString)
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".json") && return joinpath(root, f)
    end
    error("no .json phantom header found under $dir")
end

_xcist_eltype(dt::AbstractString) =
    dt == "int8"    ? Int8    :
    dt == "uint8"   ? UInt8   :
    dt == "float32" ? Float32 :
    dt == "float64" ? Float64 :
    error("unsupported XCIST volumefractionmap_datatype \"$dt\"")

# Assemble the per-material volume-fraction maps under `dir` into a single
# labeled UInt8 mask (one label per voxel; label i = XCIST material i, 0 = air),
# returning the mask, the per-label material NAMES, and the voxel size (cm).
# Each material map is placed on a common grid by isocenter alignment
# (`global = local − offset`); binary XCAT maps make the label reduction exact.
function _assemble_xcist(
        dir::AbstractString;
        voxel_size_cm::Union{Nothing, NTuple{3, Real}} = nothing,
        downsample::Integer = 1,
    )
    jf = _find_xcist_json(dir)
    base = dirname(jf)
    hdr = JSON.parsefile(jf)

    nm = Int(hdr["n_materials"])
    names  = String.(hdr["mat_name"])
    files  = String.(hdr["volumefractionmap_filename"])
    dtypes = String.(hdr["volumefractionmap_datatype"])
    cols   = Int.(hdr["cols"]);   rows   = Int.(hdr["rows"]);   slices = Int.(hdr["slices"])
    xoff   = Float64.(hdr["x_offset"]); yoff = Float64.(hdr["y_offset"]); zoff = Float64.(hdr["z_offset"])
    xs     = Float64.(hdr["x_size"]);   ys   = Float64.(hdr["y_size"]);   zs   = Float64.(hdr["z_size"])

    # Common grid spanning the union of all bounding boxes, aligned by isocenter.
    gc0 = minimum(-xoff[i] for i in 1:nm)
    gr0 = minimum(-yoff[i] for i in 1:nm)
    gs0 = minimum(-zoff[i] for i in 1:nm)
    GC = round(Int, maximum(cols[i]   - 1 - xoff[i] for i in 1:nm) - gc0) + 1
    GR = round(Int, maximum(rows[i]   - 1 - yoff[i] for i in 1:nm) - gr0) + 1
    GS = round(Int, maximum(slices[i] - 1 - zoff[i] for i in 1:nm) - gs0) + 1

    # Build the (optionally downsampled) mask directly: iterate the OUTPUT grid
    # and sample each material's nearest-neighbor block center.  Fusing the
    # downsample into placement avoids materializing the full-resolution mask
    # (the full chest is ~1.4 billion voxels — assembling it then downsampling
    # is minutes; this is seconds).  With f = 1 this reduces to a plain place.
    f = max(1, Int(downsample))
    off = f ÷ 2
    dGC = max(1, GC ÷ f); dGR = max(1, GR ÷ f); dGS = max(1, GS ÷ f)
    mask = zeros(UInt8, dGC, dGR, dGS)
    for i in 1:nm
        T = _xcist_eltype(dtypes[i])
        A = reshape(read!(joinpath(base, files[i]), Vector{T}(undef, cols[i] * rows[i] * slices[i])),
            (cols[i], rows[i], slices[i]))                       # [col, row, slice], col fastest
        c0 = round(Int, -xoff[i] - gc0)
        r0 = round(Int, -yoff[i] - gr0)
        s0 = round(Int, -zoff[i] - gs0)
        ci, ri, si = cols[i], rows[i], slices[i]
        @inbounds for K in 1:dGS
            s = (K - 1) * f + off + 1 - s0
            (1 <= s <= si) || continue
            for J in 1:dGR
                r = (J - 1) * f + off + 1 - r0
                (1 <= r <= ri) || continue
                for I in 1:dGC
                    c = (I - 1) * f + off + 1 - c0
                    (1 <= c <= ci) && A[c, r, s] > 0 && (mask[I, J, K] = UInt8(i))
                end
            end
        end
    end

    # header is mm → cm; scale each axis by its full/downsampled size ratio so
    # the physical extent is preserved (degenerate axes stay unscaled).  An
    # explicit voxel_size_cm is taken as the final size.
    vs_cm = voxel_size_cm !== nothing ? Float64.(voxel_size_cm) :
        (xs[1] / 10, ys[1] / 10, zs[1] / 10) .* ((GC, GR, GS) ./ (dGC, dGR, dGS))
    return (mask = mask, names = names, voxel_size_cm = vs_cm)
end

# label → XA.Material dict (0 = air) from per-label names + an XCIST-name→material map.
function _xcist_materials_dict(names::Vector{String}; materials::Union{Nothing, AbstractDict} = nothing)
    matmap = materials === nothing ? xcat_default_materials() : materials
    mdict = Dict{Int, XA.Material}(0 => XA.Materials.air)
    for (i, nm) in enumerate(names)
        haskey(matmap, nm) ||
            error("no material mapping for XCIST material \"$nm\" — pass a `materials` dict covering it")
        mdict[i] = matmap[nm]
    end
    return mdict
end

"""
    load_xcat_phantom(name::Symbol; path=nothing, materials=nothing,
                      voxel_size_cm=nothing, downsample=1, quiet=false)
        -> (; mask, materials, label_names, voxel_size_cm)

Download (if needed) and load XCAT phantom `name`, returning the labeled pieces:
the CPU `mask` (`UInt8`, one label per voxel, `0` = air), the `materials` map
(`label → XA.Material`), the `label_names` (`label → XCIST name`), and the voxel
size (cm).  You build the `Phantom` yourself in one line — placing the mask on
your device first when you want a GPU simulation (`create_eict_workspace` picks
the compute backend from the mask's array type):

```julia
p = load_xcat_female_slab()                                     # auto-download → pieces
phantom = Phantom(p.mask, p.materials, p.voxel_size_cm)          # CPU
phantom = Phantom(to_gpu(p.mask), p.materials, p.voxel_size_cm)  # GPU

# customize materials up front, by XCIST name …
mats = xcat_default_materials(); mats["ncat_blood"] = XA.Materials.wholeblood
p = load_xcat_female_slab(; materials = mats, downsample = 6)
# … or edit the returned label-keyed dict before building:
p.materials[7] = XA.Materials.wholeblood
```

`path` / caching: see [`download_xcat_phantom`](@ref).  `downsample` is a
label-preserving nearest-neighbor integer factor (extent preserved).
"""
function load_xcat_phantom(
        name::Symbol;
        path::Union{Nothing, AbstractString} = nothing,
        materials::Union{Nothing, AbstractDict} = nothing,
        voxel_size_cm::Union{Nothing, NTuple{3, Real}} = nothing,
        downsample::Integer = 1,
        quiet::Bool = false,
    )
    dir = download_xcat_phantom(name; path = path, quiet = quiet)
    a = _assemble_xcist(dir; voxel_size_cm = voxel_size_cm, downsample = downsample)
    return (
        mask = a.mask,
        materials = _xcist_materials_dict(a.names; materials = materials),
        label_names = Dict(i => a.names[i] for i in eachindex(a.names)),
        voxel_size_cm = a.voxel_size_cm,
    )
end

for sym in (:female_slab, :female_chest, :male_slab, :male_chest)
    @eval $(Symbol(:load_xcat_, sym))(; kwargs...) = load_xcat_phantom($(QuoteNode(sym)); kwargs...)
end

# =============================================================================
# Exports
# =============================================================================

export xcat_phantoms, xcat_default_materials, xcat_citation,
    download_xcat_phantom, load_xcat_phantom,
    download_xcat_female_slab, download_xcat_female_chest,
    download_xcat_male_slab, download_xcat_male_chest,
    load_xcat_female_slab, load_xcat_female_chest,
    load_xcat_male_slab, load_xcat_male_chest
