# =============================================================================
# The scatter library on disk, for BasisSimulator: plain arrays and a TOML header
# =============================================================================
# BasisSimulator never loads BasisMonteCarlo. So the library is written as one binary file of
# little-endian Float32 arrays, laid out one after another in Julia's (column-major) order, and a
# TOML header beside it that says what every array is: its name, shape, axes, units and byte
# offset, the scanners as modelled, the grids, the water attenuation, how it was made and what it
# passed. `read_scatter_library` reads it back with nothing but Base and TOML, so BasisSimulator
# can carry a copy of it unchanged.
# =============================================================================

"""
    write_scatter_library(path, library::ShapeScatterLibrary; scorecard = "", provenance = Dict())
        -> (; header, data)

Write `library` as `path * ".toml"` (the header) and `path * ".bin"` (the arrays). For every
scanner: `scatter`, `rayleigh` and `forward` (column, row, impact, aspect, thickness, energy) and
`primary` (impact, aspect, thickness, energy), all per incident photon; `paths` (process, energy
band, impact, aspect, thickness, energy), the mean length in mm the detected photons travelled in
each band; `offsets` and `spreads` (column, impact, aspect, thickness, energy), the mean and the
standard deviation in degrees of where in each column the scatter landed; and `angles` (process,
angle bin, energy), the profile of the scatter its detector sees. And once, `elements/scattering`
([`element_scattering_table`](@ref): element, process, angle bin, keV) and `elements/water_pdf`
(cosine cell, keV, process: water's chance per steradian of scattering at each cosine of the angle,
Compton then Rayleigh, Geant4's scattering functions and form factors on `water_pdf_cells` equal
cells of the cosine from −1). Little-endian Float32, column-major. The header records each array's
shape and byte offset, the axes, each scanner's geometry, bins, grid and detector (its efficiency
at every keV, and its attenuation and depth for the obliquity), Geant4's attenuation of water on a
1-keV grid in all and by Compton and Rayleigh scattering, the single-scatter quadrature
([`single_scatter`](@ref)), `provenance` and `scorecard`.
"""
function write_scatter_library(
    path::AbstractString,
    lib::ShapeScatterLibrary;
    scorecard::AbstractString = "",
    provenance = Dict{String, Any}(),
)
    header_path, data_path = path * ".toml", path * ".bin"
    arrays = Dict{String, Any}[]
    offset = 0
    open(data_path, "w") do io
        blocks = vcat(
            [
                (s.name, kind, A)
                for s in lib.scanners
                for (kind, A) in (
                    ("scatter", lib.scatter[s.name]),
                    ("rayleigh", lib.rayleigh[s.name]),
                    ("forward", lib.forward[s.name]),
                    ("primary", lib.primary[s.name]),
                    ("paths", lib.paths[s.name]),
                    ("offsets", lib.offsets[s.name]),
                    ("spreads", lib.spreads[s.name]),
                    ("angles", lib.angles[s.name]),
                )
            ],
            [
                ("elements", "scattering", Float32.(element_scattering_table(ELEMENT_Z))),
                (
                    "elements",
                    "water_pdf",
                    Float32.(cached_scattering_tables(DETECTOR_MATERIALS[:Water]).pdf),
                ),
            ],
        )
        for (owner, kind, A) in blocks
            write(io, htol.(vec(Float32.(A))))
            push!(
                arrays,
                Dict{String, Any}(
                    "scanner" => owner,
                    "kind" => kind,
                    "shape" => collect(size(A)),
                    "offset_bytes" => offset,
                ),
            )
            offset += sizeof(Float32) * length(A)
        end
    end
    header = Dict{String, Any}(
        "format" => "BasisSimulator scatter library",
        "format_version" => 2,
        "data_file" => basename(data_path),
        "element_type" => "Float32",
        "byte_order" => "little-endian",
        "array_order" => "column-major",
        "units" =>
            "detected signal per incident photon (keV per photon for energy-integrating detectors, counts per photon for photon-counting ones)",
        "axes" => Dict{String, Any}(
            "scatter" => [
                "column",
                "row",
                "impact_fraction",
                "aspect",
                "thickness_mm",
                "energy_keV",
            ],
            "primary" => ["impact_fraction", "aspect", "thickness_mm", "energy_keV"],
            "paths" => [
                "process",
                "path_energy_band",
                "impact_fraction",
                "aspect",
                "thickness_mm",
                "energy_keV",
            ],
            "angles" => ["process", "scatter_angle_bin", "energy_keV"],
            "offsets" => [
                "column",
                "impact_fraction",
                "aspect",
                "thickness_mm",
                "energy_keV",
            ],
            "spreads" => [
                "column",
                "impact_fraction",
                "aspect",
                "thickness_mm",
                "energy_keV",
            ],
            "thickness_mm" => lib.thicknesses_mm,
            "aspect" => lib.aspects,
            "impact_fraction" => lib.impact_fractions,
            "energy_keV" => lib.energies_keV,
        ),
        "geometry" =>
            "a pencil beam along the thickness of a water elliptical cylinder `thickness_mm` thick and `aspect × thickness_mm` wide (at most 700 mm), passing its axis at `impact_fraction` of the half-width the tilted ray sees (sqrt(b² cos²t + a² sin²t), t its tilt), tilted as a ray from a focal spot 600 mm from the axis; the scanner's focal spot on the ray's line at its source-to-isocentre distance from the cylinder's axis; columns in degrees across the fan from the central ray (+ toward the ray's side), rows in mm along z at the detector",
        "water_mu_per_mm" => lib.water_mu_per_mm,
        "water_compton_per_mm" => lib.water_compton_per_mm,
        "water_rayleigh_per_mm" => lib.water_rayleigh_per_mm,
        "water_pdf_cells" => cached_scattering_tables(DETECTOR_MATERIALS[:Water]).cells,
        "single_scatter" => Dict{String, Any}(
            "depths" => SINGLE_DEPTHS,
            "points_per_column" => SINGLE_POINTS[1],
            "points_per_row" => SINGLE_POINTS[2],
            "columns_each_side" => SINGLE_COLUMNS,
        ),
        "scatter_angle_edges_deg" => SCATTER_ANGLE_EDGES_DEG,
        "forward_rayleigh_deg" => FORWARD_RAYLEIGH_DEG,
        "path_energy_edges_keV" => PATH_ENERGY_EDGES_KEV,
        "element_Z" => collect(ELEMENT_Z),
        "scanners" => [
            Dict{String, Any}(
                "name" => s.name,
                "source_to_isocenter_mm" => s.source_to_isocenter_mm,
                "source_to_detector_mm" => s.source_to_detector_mm,
                "fan_deg" => s.fan_deg,
                "z_half_mm" => s.z_half_mm,
                "column_edges_deg" => column_edges(s),
                "row_edges_mm" => row_edges(s),
                "grid" => String(s.grid),
                "grid_height_mm" => s.grid_height_mm,
                "grid_opening_mm" => [isinf(x) ? -1.0 : x for x in s.grid_opening_mm],
                "energy_weighted" => s.energy_weighted,
                "efficiency" => s.efficiency,
                "detector_mu_per_mm" => s.mu_per_mm,
                "detector_depth_mm" => s.depth_mm,
            )
            for s in lib.scanners
        ],
        "arrays" => arrays,
        "provenance" => provenance,
        "scorecard" => scorecard,
    )
    open(io -> TOML.print(io, header), header_path, "w")
    return (header = header_path, data = data_path)
end

"""The elements whose scattering the library file tabulates (hydrogen to uranium)."""
const ELEMENT_Z = 1:92

"""
    read_scatter_library(header_path) -> Dict{String, Any}

Read a library [`write_scatter_library`](@ref) wrote, with nothing but Base and TOML: the header's
contents, with `"arrays"` replaced by `Dict(scanner => Dict(kind => Array{Float32}))`. This is the
reference reader for BasisSimulator, which keeps its own copy and never loads BasisMonteCarlo.
"""
function read_scatter_library(header_path::AbstractString)
    header = TOML.parsefile(header_path)
    data_path = joinpath(dirname(header_path), header["data_file"])
    arrays = Dict{String, Dict{String, Array{Float32}}}()
    open(data_path) do io
        for entry in header["arrays"]
            A = Array{Float32}(undef, Tuple(entry["shape"]))
            seek(io, entry["offset_bytes"])
            read!(io, A)
            A .= ltoh.(A)
            get!(
                arrays,
                entry["scanner"],
                Dict{String, Array{Float32}}(),
            )[entry["kind"]] = A
        end
    end
    header["arrays"] = arrays
    return header
end

"""
    element_scattering_table(Zs = 1:92) -> Array{Float64, 4}

Each element's mass cross-section for Compton and Rayleigh scattering (cm²/g, NIST XCOM's
incoherent and coherent), split over the scattering-angle bins of `SCATTER_ANGLE_EDGES_DEG` by
Geant4's scattering functions and form factors: `table[element, process, bin, keV]` (process 1
Compton, 2 Rayleigh; 1 to 150 keV). A material's split is its elements' summed by mass fraction,
so BasisSimulator can weigh any material's scattering by the angles a detector sees, from this
table alone.
"""
function element_scattering_table(Zs = 1:92)
    nbins = length(SCATTER_ANGLE_EDGES_DEG) - 1
    table = zeros(length(Zs), 2, nbins, 150)
    for (i, Z) in enumerate(Zs)
        symbol = element_symbol(Z)
        tables = scattering_tables(DetectorMaterial(
            symbol;
            density = 1.0,
            mass_fractions = Dict(symbol => 1.0),
        ))
        cells = tables.cells
        for E in 1:150, (process, xa) in (
                (1, XA.IncoherentScattering()),
                (2, XA.CoherentScattering()),
            )
            σ = ustrip(XA.mass_attenuation_coeff(XA.get_element_by_Z(Z), E * u"keV", xa))
            mass = zeros(nbins)
            for k in 1:cells
                θ = acosd(clamp(-1 + (k - 0.5) * 2 / cells, -1.0, 1.0))
                mass[clamp(
                    searchsortedlast(SCATTER_ANGLE_EDGES_DEG, θ),
                    1,
                    nbins,
                )] += tables.pdf[k, E, process]
            end
            table[i, process, :, E] .= σ .* mass ./ sum(mass)
        end
    end
    return table
end

"""
    material_scattering_factors(header, scanner, mass_fractions) -> (; compton, rayleigh)

For a library read by [`read_scatter_library`](@ref): how much more (or less) a material of
`mass_fractions` (`Dict(Z => fraction)`) scatters into the angles `scanner`'s detector sees than
its total cross-section says, against water's, by process, on a 1-keV grid. With nothing but the
header's `element_scattering` table and the scanner's `angles` profile (Base and TOML only): the
reference for BasisSimulator, which multiplies each material's Compton and Rayleigh attenuation by
these before taking a ray's ratios to water.
"""
function material_scattering_factors(header, scanner::AbstractString, mass_fractions)
    table = header["arrays"]["elements"]["scattering"]
    Zs = header["element_Z"]
    profile = header["arrays"][scanner]["angles"]
    energies = Float64.(header["axes"]["energy_keV"])
    nbins = size(table, 3)
    water = Dict(1 => 0.111894, 8 => 0.888106)
    split(fractions, process, E) = sum(
        w .* Float64.(table[findfirst(==(Z), Zs), process, :, E])
        for (Z, w) in fractions
    )
    factors = zeros(2, 150)
    for process in 1:2, E in 1:150
        # the profile's share of each bin at this energy, interpolated between the library's
        i = clamp(searchsortedlast(energies, Float64(E)), 1, length(energies))
        j = min(i + 1, length(energies))
        t = j == i ? 0.0 : clamp((E - energies[i]) / (energies[j] - energies[i]), 0.0, 1.0)
        f = (1 - t) .* Float64.(profile[process, :, i]) .+
            t .* Float64.(profile[process, :, j])
        f = sum(f) > 0 ? f ./ sum(f) : fill(1 / nbins, nbins)
        m, w = split(mass_fractions, process, E), split(water, process, E)
        sm, sw = m ./ sum(m), w ./ sum(w)
        factors[process, E] = sum(f[b] * (sw[b] > 0 ? sm[b] / sw[b] : 1.0) for b in 1:nbins)
    end
    return (compton = factors[1, :], rayleigh = factors[2, :])
end
