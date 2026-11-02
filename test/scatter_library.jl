# A small synthetic scatter library in the library's own on-disk format (a TOML header and
# little-endian Float32 arrays, as BasisMonteCarlo's `write_scatter_library` writes it), loaded
# under two entries for every test that simulates scatter: `_SCATTER_TEST_ENTRY` (an energy
# integrator's, on the SOMATOM Force's geometry) and `"BS test counter"` (a photon counter's, on
# the NAEOTOM Alpha's). The real kernels are an artifact the tests never download.

import TOML

# A library in BasisMonteCarlo's format: one header and one data file, `entries` a vector of
# (scanner header Dict, Dict(kind => array)) and `elements` the shared scattering table.
function _write_scatter_library(path, entries, elements; axes)
    arrays = Dict{String, Any}[]
    offset = 0
    open(path * ".bin", "w") do io
        blocks = vcat(
            [(s["name"], kind, A) for (s, data) in entries for (kind, A) in data],
            [("elements", "scattering", elements)],
        )
        for (owner, kind, A) in blocks
            write(io, htol.(vec(Float32.(A))))
            push!(
                arrays,
                Dict{String, Any}("scanner" => owner, "kind" => kind, "shape" => collect(size(A)), "offset_bytes" => offset),
            )
            offset += sizeof(Float32) * length(A)
        end
    end
    header = Dict{String, Any}(
        "format" => "BasisSimulator scatter library",
        "format_version" => 2,
        "data_file" => basename(path) * ".bin",
        "element_type" => "Float32",
        "byte_order" => "little-endian",
        "array_order" => "column-major",
        "units" => "detected signal per incident photon",
        "axes" => axes,
        "water_mu_per_mm" => [BS.compute_μ_at_energy(BS.XA.Materials.water, Float64(E)) / 10 for E in 1:150],
        "scatter_angle_edges_deg" => [0.0, 2.5, 5.0, 10.0, 20.0, 40.0, 90.0, 180.0],
        "forward_rayleigh_deg" => 10.0,
        "path_energy_edges_keV" => collect(0.0:5.0:150.0),
        "element_Z" => collect(1:92),
        "scanners" => [s for (s, _) in entries],
        "arrays" => arrays,
    )
    open(io -> TOML.print(io, header), path * ".toml", "w")
    return path * ".toml"
end

# Smooth synthetic kernels: a bump across the fan (moved toward the ray's side with the impact,
# so the mirror image matters) and along z, growing with thickness and energy; a quarter of it
# Rayleigh, a tenth forward Rayleigh.
function _synthetic_scatter_library(dir; name = "BS test scanner", energy_weighted = true, sid = 595.0, sdd = 1085.6)
    axes = Dict{String, Any}(
        "thickness_mm" => [140.0, 300.0, 500.0],
        "aspect" => [0.6, 1.0, 2.5],
        "impact_fraction" => [0.0, 0.5, 0.99],
        "energy_keV" => [20.0, 70.0, 140.0],
    )
    cols = collect(-25.0:2.0:25.0)
    rows = collect(-55.0:10.0:55.0)
    centres(e) = (e[1:(end - 1)] .+ e[2:end]) ./ 2
    γ, z = centres(cols), centres(rows)
    T, r, u, E = axes["thickness_mm"], axes["aspect"], axes["impact_fraction"], axes["energy_keV"]
    S = [
        1.0e-4 * (t / 300) * sqrt(e / 70) * (1 + 0.2 * (a - 1)) * exp(-((g - 3 * v) / 12)^2 - (y / 40)^2)
            for g in γ, y in z, v in u, a in r, t in T, e in E
    ]
    scanner = Dict{String, Any}(
        "name" => name,
        "source_to_isocenter_mm" => sid,
        "source_to_detector_mm" => sdd,
        "fan_deg" => 50.0,
        "z_half_mm" => 55.0,
        "column_edges_deg" => cols,
        "row_edges_mm" => rows,
        "energy_weighted" => energy_weighted,
    )
    nc, nu, na, nT, nE = length(γ), length(u), length(r), length(T), length(E)
    data = Dict{String, Any}(
        "scatter" => S,
        "rayleigh" => 0.25 .* S,
        "forward" => 0.1 .* S,
        "primary" => ones(nu, na, nT, nE),
        "paths" => [t / 2 for _ in 1:2, _ in 1:30, _ in u, _ in r, t in T, _ in E],
        "offsets" => zeros(nc, nu, na, nT, nE),
        "spreads" => fill(2 / sqrt(12), nc, nu, na, nT, nE),
        "angles" => ones(2, 7, nE),
    )
    elements = [1.0 + 0.01 * Z + 0.1 * b for Z in 1:92, _ in 1:2, b in 1:7, _ in 1:150]
    mkpath(dir)
    return _write_scatter_library(joinpath(dir, replace(lowercase(name), " " => "_")), [(scanner, data)], elements; axes)
end

const _SCATTER_DIR = mktempdir()
const _SCATTER_TEST_ENTRY = "BS test scanner"
const _SCATTER_TEST_HEADER = _synthetic_scatter_library(_SCATTER_DIR)
const _SCATTER_TEST_COUNTING = _synthetic_scatter_library(_SCATTER_DIR; name = "BS test counter", energy_weighted = false, sid = 610.0, sdd = 1113.0)
BS.scatter_kernels(_SCATTER_TEST_ENTRY; path = _SCATTER_TEST_HEADER)
BS.scatter_kernels("BS test counter"; path = _SCATTER_TEST_COUNTING)
