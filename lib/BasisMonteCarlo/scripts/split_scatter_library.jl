# Split the scatter library BasisMonteCarlo writes (`write_scatter_library`, notebook 10: one
# header and one data file for every scanner) into one header + data file per scanner, in the same
# format, so that BasisSimulator ships each scanner's kernels as its own artifact (about 250 MB
# each, not 1.5 GB). Each file keeps its scanner's arrays and the shared `elements` table; the
# header is the original's with `scanners` and `arrays` narrowed to that scanner.
#
#     julia lib/BasisMonteCarlo/scripts/split_scatter_library.jl scatter_library_v2.toml out/
#     julia lib/BasisMonteCarlo/scripts/split_scatter_library.jl scatter_library_v2.toml out/ --archive
#
# `--archive` also zips each pair (`<stem>.zip`, the release asset) and prints the lines of
# BasisSimulator's `SCATTER_KERNEL_ARTIFACTS` to fill at release: the zip's SHA-256 and the
# extracted directory's git-tree-sha1. It needs p7zip_jll in the active environment.
# Plain Julia (TOML, SHA, Pkg): it never loads BasisMonteCarlo.

import TOML

"""The file stem of a scanner's entry: `"SOMATOM Force"` → `"somatom_force"`."""
entry_stem(name) = replace(lowercase(name), r"[^a-z0-9]+" => "_")

"""
    write_entry(stem, header, name, arrays) -> header path

Write `stem.toml` and `stem.bin`: the scanner `name`'s arrays (`Dict(kind => array)`) and the
shared `elements` table (`arrays["elements/scattering"]`), little-endian Float32 in column-major
order, under `header` with its `scanners` and `arrays` narrowed to that scanner.
"""
function write_entry(stem, header, name, arrays)
    out = deepcopy(header)
    out["data_file"] = basename(stem) * ".bin"
    out["scanners"] = [s for s in header["scanners"] if s["name"] == name]
    length(out["scanners"]) == 1 || error("the library has no single scanner \"$(name)\"")
    entries = Dict{String, Any}[]
    offset = 0
    open(stem * ".bin", "w") do io
        for (owner, kind) in vcat([(name, k) for k in sort!(collect(keys(arrays))) if k != "elements/scattering"], [("elements", "scattering")])
            A = owner == "elements" ? arrays["elements/scattering"] : arrays[kind]
            write(io, htol.(vec(Float32.(A))))
            push!(entries, Dict{String, Any}("scanner" => owner, "kind" => kind, "shape" => collect(size(A)), "offset_bytes" => offset))
            offset += sizeof(Float32) * length(A)
        end
    end
    out["arrays"] = entries
    open(io -> TOML.print(io, out), stem * ".toml", "w")
    return stem * ".toml"
end

"""
    read_arrays(header_path, owner) -> Dict(kind => Array{Float32})

The arrays of one scanner (or of `"elements"`) from a library's data file.
"""
function read_arrays(header_path, owner)
    header = TOML.parsefile(header_path)
    arrays = Dict{String, Array{Float32}}()
    open(joinpath(dirname(header_path), header["data_file"])) do io
        for item in header["arrays"]
            item["scanner"] == owner || continue
            A = Array{Float32}(undef, Tuple(item["shape"]))
            seek(io, item["offset_bytes"])
            read!(io, A)
            arrays[item["kind"]] = ltoh.(A)
        end
    end
    return arrays
end

"""
    split_scatter_library(header_path, out_dir) -> Vector of header paths

One header + data file per scanner of the library at `header_path`, in `out_dir`.
"""
function split_scatter_library(header_path, out_dir)
    mkpath(out_dir)
    header = TOML.parsefile(header_path)
    elements = read_arrays(header_path, "elements")["scattering"]
    return map(header["scanners"]) do s
        arrays = merge(read_arrays(header_path, s["name"]), Dict("elements/scattering" => elements))
        write_entry(joinpath(out_dir, entry_stem(s["name"])), header, s["name"], arrays)
    end
end

# The release assets: one zip per entry, its SHA-256 and the extracted tree's git-tree-sha1.
function archive_entries(headers)
    p7zip = Base.require(Base.PkgId(Base.UUID("3f19e933-33d8-53b3-aaab-bd5110c3b7a0"), "p7zip_jll"))
    Pkg = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
    SHA = Base.require(Base.PkgId(Base.UUID("ea8e919c-243c-51af-8825-aaa63cd721ce"), "SHA"))
    for header in headers
        stem = splitext(header)[1]
        zip = stem * ".zip"
        rm(zip; force = true)
        run(pipeline(`$(Base.invokelatest(p7zip.p7zip)) a -tzip $(zip) $(stem * ".toml") $(stem * ".bin")`; stdout = devnull))
        sha = open(io -> bytes2hex(Base.invokelatest(SHA.sha256, io)), zip)
        tree = mktempdir() do dir
            cp(stem * ".toml", joinpath(dir, basename(stem) * ".toml"))
            cp(stem * ".bin", joinpath(dir, basename(stem) * ".bin"))
            bytes2hex(Base.invokelatest(Pkg.GitTools.tree_hash, dir))
        end
        name = only(TOML.parsefile(header)["scanners"])["name"]
        println("\"$(name)\": file $(basename(stem)).toml, sha256 \"$(sha)\", tree_hash \"$(tree)\"")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("usage: split_scatter_library.jl library.toml out_dir [--archive]")
    headers = split_scatter_library(ARGS[1], ARGS[2])
    foreach(println, headers)
    "--archive" in ARGS && archive_entries(headers)
end
