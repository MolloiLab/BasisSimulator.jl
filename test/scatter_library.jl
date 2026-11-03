# A small synthetic scatter library in the library's own on-disk format (a TOML header and
# little-endian Float32 arrays, as BasisMonteCarlo's `write_scatter_library` writes it), loaded
# under three entries for every test that simulates scatter: `_SCATTER_TEST_ENTRY` (an energy
# integrator's, on the SOMATOM Force's geometry), `"BS test counter"` (a photon counter's, on the
# NAEOTOM Alpha's) and `_SCATTER_AS_MADE_ENTRY` (the first's kernels in a water that does not
# scatter, so that they keep their single scatter: the superposition without the swap). The real
# kernels are an artifact the tests never download.

import TOML

const _U = BS.Unitful

# A library in BasisMonteCarlo's format: one header and one data file, `entries` a vector of
# (scanner header Dict, Dict(kind => array)), `elements` the shared tables (Dict(kind => array)) and
# `fields` the header's other fields.
function _write_scatter_library(path, entries, elements; axes, fields)
    arrays = Dict{String, Any}[]
    offset = 0
    open(path * ".bin", "w") do io
        blocks = vcat(
            [(s["name"], kind, A) for (s, data) in entries for (kind, A) in data],
            [("elements", kind, elements[kind]) for kind in sort!(collect(keys(elements)))],
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
    header = merge(
        Dict{String, Any}(
            "format" => "BasisSimulator scatter library",
            "format_version" => 2,
            "data_file" => basename(path) * ".bin",
            "element_type" => "Float32",
            "byte_order" => "little-endian",
            "array_order" => "column-major",
            "units" => "detected signal per incident photon",
            "axes" => axes,
            "scatter_angle_edges_deg" => [0.0, 2.5, 5.0, 10.0, 20.0, 40.0, 90.0, 180.0],
            "forward_rayleigh_deg" => 10.0,
            "path_energy_edges_keV" => collect(0.0:5.0:150.0),
            "element_Z" => collect(1:92),
            "scanners" => [s for (s, _) in entries],
            "arrays" => arrays,
        ),
        fields,
    )
    open(io -> TOML.print(io, header), path * ".toml", "w")
    return path * ".toml"
end

# Water's attenuation (per mm) on the 1-keV grid, in all (`nothing`) or by one process, from
# BasisSimulator's tables
_water_per_mm(process) = [
    process === nothing ? BS.compute_μ_at_energy(BS.XA.Materials.water, Float64(E)) / 10 :
        _U.ustrip(_U.@u_str("cm^-1"), BS.XA.linear_attenuation_coeff(BS.XA.Materials.water, E * _U.@u_str("keV"), process)) / 10
        for E in 1:150
]

# Water's angular tables, per steradian, in `cells` equal cells of the cosine from −1: Klein–Nishina
# (Compton) and Thomson times a form factor falling with the momentum transfer (Rayleigh, peaked
# forward), each normalised over the sphere at every keV
function _synthetic_water_pdf(cells)
    μ = [-1 + (c - 0.5) * 2 / cells for c in 1:cells]
    pdf = zeros(cells, 150, 2)
    for E in 1:150
        ratio = 1 ./ (1 .+ E / 511 .* (1 .- μ))
        pdf[:, E, 1] .= ratio .^ 2 .* (ratio .+ 1 ./ ratio .- (1 .- μ .^ 2)) ./ 2
        pdf[:, E, 2] .= (1 .+ μ .^ 2) ./ 2 .* exp.(-(1 .- μ) .* (E / 12)^2)
        for p in 1:2
            pdf[:, E, p] ./= 2π * (2 / cells) * sum(pdf[:, E, p])
        end
    end
    return pdf
end

# Smooth synthetic kernels: a multiple scatter, a bump across the fan (toward the ray's side with the
# impact, and for an ellipse thicker than wide even through its axis, so that the mirror image
# matters there too; a ray through a round one's axis is symmetric) and along z, growing with
# thickness and energy, a quarter of it Rayleigh and a tenth forward Rayleigh; and each
# kernel's own single scatter, by the library's quadrature in its water ellipse, added to it (as the
# Monte Carlo's kernels hold both). `scatters = false` writes the same kernels with a water that does
# not scatter, so that loading them takes nothing out and no ray adds its own.
function _synthetic_scatter_library(
        dir;
        name = "BS test scanner",
        energy_weighted = true,
        sid = 595.0,
        sdd = 1085.6,
        scatters = true,
    )
    # the library's own spacing of impacts (a ray's single scatter is far from linear in it: an edge
    # ray's is twenty times the axis ray's), coarser thicknesses, aspects and energies
    axes = Dict{String, Any}(
        "thickness_mm" => [140.0, 220.0, 300.0, 380.0, 500.0],
        "aspect" => [0.6, 0.8, 1.0, 1.25, 1.6, 2.5],
        "impact_fraction" => [collect(0.0:0.05:0.95); 0.99],
        "energy_keV" => [20.0, 70.0, 140.0],
    )
    cols = collect(-25.0:2.0:25.0)
    rows = collect(-55.0:10.0:55.0)
    centres(e) = (e[1:(end - 1)] .+ e[2:end]) ./ 2
    γ, z = centres(cols), centres(rows)
    T, r, u, E = axes["thickness_mm"], axes["aspect"], axes["impact_fraction"], axes["energy_keV"]
    bump = [
        1.0e-4 * (t / 300) * sqrt(e / 70) * (1 + 0.2 * (a - 1)) * exp(-((g - 3 * v - 5 * max(0, 1 - a)) / 12)^2 - (y / 40)^2)
            for g in γ, y in z, v in u, a in r, t in T, e in E
    ]
    detector_mu = [5.0 * (60 / k)^2.7 + 0.3 for k in 1:150]
    scanner = Dict{String, Any}(
        "name" => name,
        "source_to_isocenter_mm" => sid,
        "source_to_detector_mm" => sdd,
        "fan_deg" => 50.0,
        "z_half_mm" => 55.0,
        "column_edges_deg" => cols,
        "row_edges_mm" => rows,
        "energy_weighted" => energy_weighted,
        "efficiency" => 1 .- exp.(-1.4 .* detector_mu),
        "detector_mu_per_mm" => detector_mu,
        "detector_depth_mm" => 1.4,
        "grid" => "both",
        "grid_height_mm" => 10.0,
        "grid_opening_mm" => [0.94, 1.01],
    )
    quadrature = Dict{String, Any}("depths" => 32, "points_per_column" => 4, "points_per_row" => 2, "columns_each_side" => 8)
    water = (
        mu = _water_per_mm(nothing),
        compton = _water_per_mm(BS.XA.IncoherentScattering()),
        rayleigh = _water_per_mm(BS.XA.CoherentScattering()),
    )
    water_pdf = _synthetic_water_pdf(400)

    # each kernel's own single scatter, as the library's reader computes it
    tables = BS._single_tables(
        Float64;
        cols,
        rows,
        sdd,
        energies = E,
        water,
        water_pdf = Float32.(water_pdf),
        detector = (efficiency = scanner["efficiency"], mu = detector_mu, depth = 1.4),
        energy_weighted,
        grid = :both,
        grid_height = 10.0,
        grid_opening = (0.94, 1.01),
        single = (depths = 32, points_per_column = 4, points_per_row = 2, columns_each_side = 8),
        forward_deg = 10.0,
    )
    nc, nr, nu, na, nT, nE = size(bump)
    single = zeros(nc, nr, nu, na, nT, nE, 3)
    for (iT, t) in enumerate(T), (ia, a) in enumerate(r), (iu, v) in enumerate(u)
        b = BS._kernel_half_width(t, a)
        angle = asind(BS._kernel_offset(t / 2, b, v) / sid)
        s = zeros(nr, nc, nE, 3)
        BS._single_scatter!(s, BS._WaterEllipse(t / 2, b, sid), angle, tables)
        single[:, :, iu, ia, iT, :, :] .= permutedims(s, (2, 1, 3, 4))
    end
    compton, wide, forward = (selectdim(single, 7, p) for p in 1:3)
    data = Dict{String, Any}(
        "scatter" => bump .+ compton .+ wide .+ forward,
        "rayleigh" => 0.25 .* bump .+ wide .+ forward,
        "forward" => 0.1 .* bump .+ forward,
        "primary" => ones(nu, na, nT, nE),
        # half the thickness, spread over the 5-keV bands from 20 to 140 keV, where the photons are
        "paths" => [5 <= band <= 28 ? t / 48 : 0.0 for _ in 1:2, band in 1:30, _ in u, _ in r, t in T, _ in E],
        "offsets" => zeros(nc, nu, na, nT, nE),
        "spreads" => fill(2 / sqrt(12), nc, nu, na, nT, nE),
        "angles" => ones(2, 7, nE),
    )
    elements = Dict(
        "scattering" => [1.0 + 0.01 * Z + 0.1 * b for Z in 1:92, _ in 1:2, b in 1:7, _ in 1:150],
        "water_pdf" => water_pdf,
    )
    fields = Dict{String, Any}(
        "water_mu_per_mm" => water.mu,
        "water_compton_per_mm" => scatters ? water.compton : zeros(150),
        "water_rayleigh_per_mm" => scatters ? water.rayleigh : zeros(150),
        "water_pdf_cells" => 400,
        "single_scatter" => quadrature,
    )
    mkpath(dir)
    stem = joinpath(dir, replace(lowercase(name), r"[^a-z0-9]+" => "_"))
    return (; header = _write_scatter_library(stem, [(scanner, data)], elements; axes, fields), bump, single)
end

const _SCATTER_DIR = mktempdir()
const _SCATTER_TEST_ENTRY = "BS test scanner"
const _SCATTER_AS_MADE_ENTRY = "BS test scanner, kernels as made"
const _SCATTER_TEST_LIBRARY = _synthetic_scatter_library(_SCATTER_DIR)
const _SCATTER_TEST_HEADER = _SCATTER_TEST_LIBRARY.header
const _SCATTER_TEST_COUNTING = _synthetic_scatter_library(_SCATTER_DIR; name = "BS test counter", energy_weighted = false, sid = 610.0, sdd = 1113.0).header
const _SCATTER_AS_MADE_HEADER = _synthetic_scatter_library(_SCATTER_DIR; name = _SCATTER_AS_MADE_ENTRY, scatters = false).header
BS.scatter_kernels(_SCATTER_TEST_ENTRY; path = _SCATTER_TEST_HEADER)
BS.scatter_kernels("BS test counter"; path = _SCATTER_TEST_COUNTING)
BS.scatter_kernels(_SCATTER_AS_MADE_ENTRY; path = _SCATTER_AS_MADE_HEADER)
