# =============================================================================
# A scanner's whole detector: the arc
# =============================================================================
# The frame: the focal spot at the origin, the central ray along +y, the fan in the x–y plane,
# z along the patient. The detector is a cylindrical shell of CdTe around the focal spot,
# `radius_mm` to its front face, covering the fan across and the rows along z. In front of it
# stands a focused anti-scatter grid: plates between groups of dexel columns, pointing at the
# focal spot, and walls between groups of rows, tilted toward it.
#
# A dexel is addressed by (column, row). Across the fan a dexel's position is the arc length
# s = radius × angle at the front face; along z it is z itself. Columns and rows are laid out as
# in a DexelReadout: groups of 4 × 6 dexels separated by the gaps under the grid's walls, the
# whole pattern centred on the central ray.
# =============================================================================

"""
    Spectrum(energies_keV, weights)

A photon spectrum: relative numbers of photons per bin, at bin centres `energies_keV` on a
uniform grid. [`simulate_spectrum`](@ref) draws each photon's energy from it.
"""
struct Spectrum
    energies_keV::Vector{Float64}
    weights::Vector{Float64}
end

mean_energy(s::Spectrum) = sum(s.energies_keV .* s.weights) / sum(s.weights)

"""
    ArcDetector(; columns, rows, radius_mm = 1113.0, thickness_mm = 1.6, dexel_mm = (0.275, 0.322),
                  group = (4, 6), gap_mm = (0.26, 0.25), septum_material = nothing,
                  grid_height_mm = 10.0, grid_walls = :both, grid_wall_mm = gap_mm,
                  material = DetectorMaterial(:CdTe), grid_material = DetectorMaterial(:W))

A scanner's whole detector: a cylindrical shell of `material`, `thickness_mm` deep, at
`radius_mm` from the focal spot, with `columns` × `rows` dexels of `dexel_mm` (across the fan,
along z) in groups of `group` separated by `gap_mm`.

- `septum_material`: `nothing` for a photon-counting crystal, which is continuous under the gaps
  (the dexels are only its electrode pattern); a material (a reflector) for a scintillator
  array, whose crystal is cut into dexels with the gaps filled by that material, the full depth.
- The anti-scatter grid: focused tungsten walls `grid_height_mm` tall (`0` for no grid) and
  `grid_wall_mm` thick (across, along z) standing over the gaps, either in both directions
  (`grid_walls = :both`, a 2-D grid) or only between columns (`:across`, a 1-D grid of plates).

The defaults are the NAEOTOM Alpha's (see notebook 03); `columns = 2752` is its detector A,
`1984` its detector B, `rows = 288` the Alpha and Alpha.Prime, `192` the Alpha.Pro.
"""
Base.@kwdef struct ArcDetector <: AbstractDetector
    columns::Int
    rows::Int
    radius_mm::Float64 = 1113.0
    thickness_mm::Float64 = 1.6
    dexel_mm::Tuple{Float64, Float64} = (0.275, 0.322)
    group::Tuple{Int, Int} = (4, 6)
    gap_mm::Tuple{Float64, Float64} = (0.26, 0.25)
    septum_material::Union{Nothing, DetectorMaterial} = nothing
    grid_height_mm::Float64 = 10.0
    grid_walls::Symbol = :both
    grid_wall_mm::Tuple{Float64, Float64} = gap_mm
    material::DetectorMaterial = DetectorMaterial(:CdTe)
    grid_material::DetectorMaterial = DetectorMaterial(:W)
end

material_key(m::Nothing) = nothing
material_key(m::DetectorMaterial) = m.name

function Base.:(==)(a::ArcDetector, b::ArcDetector)
    materials = (:material, :grid_material, :septum_material)
    same_materials = all(material_key(getfield(a, f)) == material_key(getfield(b, f)) for f in materials)
    return same_materials && all(
        getfield(a, f) == getfield(b, f) for f in fieldnames(ArcDetector) if !(f in materials)
    )
end

same_geometry(a::ArcDetector, b::ArcDetector) = a == b

"""Grid-cell pitch (across, along z) at the front face, in mm."""
cell_pitch(d::ArcDetector) = (
    d.group[1] * d.dexel_mm[1] + d.gap_mm[1],
    d.group[2] * d.dexel_mm[2] + d.gap_mm[2],
)

"""Number of grid cells (across, along z)."""
cell_count(d::ArcDetector) = (d.columns ÷ d.group[1], d.rows ÷ d.group[2])

"""Arc length across the fan and length along z of the detector's front face, in mm."""
function face_size_mm(d::ArcDetector)
    pitch, cells = cell_pitch(d), cell_count(d)
    return pitch[1] * cells[1], pitch[2] * cells[2]
end

"""Total fan angle the detector covers, in degrees."""
fan_angle_deg(d::ArcDetector) = rad2deg(face_size_mm(d)[1] / d.radius_mm)

"""
    focal_spot(detector::ArcDetector) -> FocalSpot

The beam that illuminates every dexel of `detector` uniformly from the focal spot.
"""
focal_spot(d::ArcDetector) = FocalSpot(;
    distance_mm = d.radius_mm,
    fan_deg = fan_angle_deg(d),
    z_half_mm = face_size_mm(d)[2] / 2,
)

world_half_mm(d::ArcDetector) = d.radius_mm + d.thickness_mm + 100

function build_detector!(world, d::ArcDetector)
    if d.septum_material === nothing
        build_continuous_arc!(world, d)
    else
        build_septate_arc!(world, d)
    end
    d.grid_height_mm > 0 && build_arc_grid!(world, d)
    return nothing
end

"""A photon-counting crystal: one continuous shell (the dexels are its electrode pattern)."""
function build_continuous_arc!(world, d::ArcDetector)
    s_half, z_half = face_size_mm(d) ./ 2
    half_angle = s_half / d.radius_mm

    # The crystal: a cylindrical shell segment centred on +y (Geant4 measures φ from +x)
    shell = G4Tubs(
        "Detector",
        d.radius_mm * mm,
        (d.radius_mm + d.thickness_mm) * mm,
        z_half * mm,
        π / 2 - half_angle,
        2 * half_angle,
    )
    crystal = G4LogicalVolume(shell, g4_material(d.material), "Detector")
    G4PVPlacement(nothing, G4ThreeVector(), crystal, "Detector", world, false, CRYSTAL_COPY)
    return nothing
end

"""
A scintillator array: a shell of reflector holding one ring of crystal per group of rows, each
cut across the fan by radial reflector septa between column groups. The septa are focused on
the focal spot, like the dexels; a step in one is not in the crystal.
"""
function build_septate_arc!(world, d::ArcDetector)
    R, t = d.radius_mm, d.thickness_mm
    pitch = cell_pitch(d)
    cells_across, cells_along = cell_count(d)
    s_half, z_half = face_size_mm(d) ./ 2
    reflector = g4_material(d.septum_material)

    # The reflector: the whole face, the full depth
    shell = G4LogicalVolume(
        G4Tubs(
            "Reflector",
            R * mm,
            (R + t) * mm,
            z_half * mm,
            π / 2 - s_half / R,
            2 * s_half / R,
        ),
        reflector,
        "Reflector",
    )
    G4PVPlacement(nothing, G4ThreeVector(), shell, "Reflector", world, false, SEPTUM_COPY)

    # One ring of crystal per group of rows, spanning the columns' active width
    ring_half_z = d.group[2] * d.dexel_mm[2] / 2
    active_half = s_half - d.gap_mm[1] / 2
    ring = G4LogicalVolume(
        G4Tubs(
            "Detector",
            R * mm,
            (R + t) * mm,
            ring_half_z * mm,
            π / 2 - active_half / R,
            2 * active_half / R,
        ),
        g4_material(d.material),
        "Detector",
    )
    for j in 0:(cells_along - 1)
        z_centre = (j + 0.5) * pitch[2] - z_half
        G4PVPlacement(
            nothing,
            G4ThreeVector(0, 0, z_centre * mm),
            ring,
            "Detector",
            shell,
            false,
            CRYSTAL_COPY,
        )
    end

    # Radial septa between column groups, inside every ring (1 µm short of each face)
    septum = G4LogicalVolume(
        G4Box("Septum", d.gap_mm[1] * mm / 2, (t / 2 - 1.0e-3) * mm, ring_half_z * mm),
        reflector,
        "Septum",
    )
    r_mid = R + t / 2
    for i in 1:(cells_across - 1)
        θ = (i * pitch[1] - s_half) / R
        rotation = G4RotationMatrix()
        rotateZ(rotation, -θ)
        position = G4ThreeVector(r_mid * sin(θ) * mm, r_mid * cos(θ) * mm, 0)
        G4PVPlacement(
            G4Transform3D(rotation, position),
            septum,
            "Septum",
            ring,
            false,
            SEPTUM_COPY,
        )
    end
    return nothing
end

"""
The focused grid: plates between column groups pointing at the focal spot, and (for a 2-D grid)
wall segments between row groups, each tilted toward the focal spot, between neighbouring plates.
"""
function build_arc_grid!(world, d::ArcDetector)
    tungsten = g4_material(d.grid_material)
    H, R = d.grid_height_mm, d.radius_mm
    pitch = cell_pitch(d)
    cells_across, cells_along = cell_count(d)
    s_half, z_half = face_size_mm(d) ./ 2
    r_mid = R - H / 2

    # Plates between column groups: at angle θ from +y, radial, the full height of the detector
    plate = G4LogicalVolume(
        G4Box("PlateX", d.grid_wall_mm[1] * mm / 2, H * mm / 2, z_half * mm),
        tungsten,
        "PlateX",
    )
    for i in 0:cells_across
        θ = (i * pitch[1] - s_half) / R
        rotation = G4RotationMatrix()
        rotateZ(rotation, -θ)
        position = G4ThreeVector(r_mid * sin(θ) * mm, r_mid * cos(θ) * mm, 0)
        G4PVPlacement(
            G4Transform3D(rotation, position),
            plate,
            "PlateX",
            world,
            false,
            COLLIMATOR_COPY,
        )
    end

    d.grid_walls === :across && return nothing

    # Walls between row groups: one segment per cell across, tilted toward the focal spot
    segment = G4LogicalVolume(
        G4Box(
            "WallZ",
            (pitch[1] - d.grid_wall_mm[1]) * mm / 2,
            H * mm / 2,
            d.grid_wall_mm[2] * mm / 2,
        ),
        tungsten,
        "WallZ",
    )
    for j in 0:cells_along, i in 0:(cells_across - 1)
        z0 = j * pitch[2] - z_half
        θ = ((i + 0.5) * pitch[1] - s_half) / R
        tilt = atan(z0 / R)
        rotation = G4RotationMatrix()
        rotateX(rotation, tilt)
        rotateZ(rotation, -θ)
        position = G4ThreeVector(
            r_mid * sin(θ) * mm,
            r_mid * cos(θ) * mm,
            z0 * r_mid / R * mm,
        )
        G4PVPlacement(
            G4Transform3D(rotation, position),
            segment,
            "WallZ",
            world,
            false,
            COLLIMATOR_COPY,
        )
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Spectra
# -----------------------------------------------------------------------------

function set_spectrum!(e::Engine, spectrum::Spectrum, beam::Beam, seed::Integer)
    gen = e.app.generator.data
    gen.beam = beam
    cdf = cumsum(spectrum.weights)
    gen.spectrum_keV = copy(spectrum.energies_keV)
    gen.spectrum_cdf = cdf ./ last(cdf)
    gen.spectrum_bin_keV = length(spectrum.energies_keV) > 1 ? spectrum.energies_keV[2] - spectrum.energies_keV[1] : 0.0

    h = hash((seed, :spectrum))
    ui("/random/setSeeds $(1 + h % 2147483646) $(1 + (h >> 32) % 2147483646)")
    Random.seed!(gen.rng, h)
    return e
end

"""
    simulate_spectrum(detector, spectrum; n = 100_000, beam, physics = PhysicsSettings(), seed = 1,
                      record_deposits = false, record_vertices = false) -> MonoenergeticRun

As [`simulate_monoenergetic`](@ref), with each photon's energy drawn from `spectrum` (uniformly
within its bin). The run's `energy_keV` is the spectrum's mean and `photon_energy_keV` holds
each photon's energy.
"""
function simulate_spectrum(
        detector::AbstractDetector,
        spectrum::Spectrum;
        n::Integer = 100_000,
        beam::Beam,
        physics::PhysicsSettings = PhysicsSettings(),
        seed::Integer = 1,
        record_deposits::Bool = false,
        record_vertices::Bool = false,
    )
    e = engine(detector, physics)
    set_spectrum!(e, spectrum, beam, seed)
    return run_photons(
        e,
        detector,
        mean_energy(spectrum),
        n,
        beam,
        physics,
        seed,
        record_deposits,
        record_vertices,
    )
end

# -----------------------------------------------------------------------------
# Reading out the whole detector
# -----------------------------------------------------------------------------

"""
    ArcReadout(; thresholds_keV, cloud_sigma_um = 30.0, electronic_noise_keV = 1.5, fano = 0.1,
                 pair_energy_eV = 4.43)

How an [`ArcDetector`](@ref) counts: every energy deposit spreads into a Gaussian charge cloud
of width `cloud_sigma_um`, each dexel collects its share, its energy is blurred by Fano and
electronic noise, and it counts once for every threshold its energy exceeds.
"""
Base.@kwdef struct ArcReadout
    thresholds_keV::Vector{Float64}
    cloud_sigma_um::Float64 = 30.0
    electronic_noise_keV::Float64 = 1.5
    fano::Float64 = 0.1
    pair_energy_eV::Float64 = 4.43
end

"""
    ArcCounts

The counts of every dexel of an [`ArcDetector`](@ref) from one run (`counts[column, row,
threshold]`), with the number of photons fired and the readout used.
"""
struct ArcCounts
    detector::ArcDetector
    readout::ArcReadout
    n::Int
    counts::Array{Int32, 3}
end

Base.:+(a::ArcCounts, b::ArcCounts) = ArcCounts(a.detector, a.readout, a.n + b.n, a.counts .+ b.counts)

"""Dexel index along one axis (1-based) and its collecting edges, for a position `u` on the face."""
function arc_dexel(u, p, G, g, n_cells)
    # Dexel m of the centred layout; with an even number of cells the centre falls on a gap
    offset = iseven(n_cells) ? (G * p + g) / 2 : 0.0
    v = u - offset
    pitch = G * p + g
    group = floor(Int, (v + pitch / 2) / pitch)
    local_u = v - group * pitch
    k = clamp(floor(Int, (local_u + G * p / 2) / p), 0, G - 1)
    m = group * G + k
    first_group = -(n_cells ÷ 2)      # groups run from -(n ÷ 2) on either layout
    index = m - first_group * G + 1
    return index, m, offset
end

"""
    arc_counts(run, readout) -> ArcCounts

Count every dexel of the run's [`ArcDetector`](@ref): the run must have been simulated with
`record_deposits = true`. Charge clouds are shared with the neighbouring dexel on each side.
"""
function arc_counts(run::MonoenergeticRun, readout::ArcReadout)
    d = run.detector
    d isa ArcDetector || throw(ArgumentError("arc_counts needs a run on an ArcDetector"))
    run.deposits === nothing && throw(ArgumentError("simulate with record_deposits = true"))

    R = d.radius_mm
    σ = readout.cloud_sigma_um / 1000
    w = readout.pair_energy_eV / 1000
    cells = cell_count(d)
    p, G, g = d.dexel_mm, d.group, d.gap_mm
    counts = zeros(Int32, d.columns, d.rows, length(readout.thresholds_keV))
    rng = Xoshiro(hash((run.seed, :readout)))

    # The energy each dexel collected from the current photon
    collected = Dict{Tuple{Int, Int}, Float64}()

    function flush!()
        for ((col, row), e) in collected
            spread = sqrt(readout.electronic_noise_keV^2 + readout.fano * w * e)
            measured = e + spread * randn(rng)
            for (t, threshold) in enumerate(readout.thresholds_keV)
                measured >= threshold && (counts[col, row, t] += 1)
            end
        end
        return empty!(collected)
    end

    dep = run.deposits
    current = dep.event[1]
    for k in eachindex(dep.event)
        if dep.event[k] != current
            flush!()
            current = dep.event[k]
        end
        x, y, z, e = dep.x_mm[k], dep.y_mm[k], dep.z_mm[k], dep.energy_keV[k]
        s = R * atan(x, y)          # arc length at the front face (charge drifts radially)

        col, mc, off_s = arc_dexel(s, p[1], G[1], g[1], cells[1])
        row, mr, off_z = arc_dexel(z, p[2], G[2], g[2], cells[2])

        # Share the cloud with the neighbours on each side
        for dc in -1:1, dr in -1:1
            c, r = col + dc, row + dr
            (1 <= c <= d.columns && 1 <= r <= d.rows) || continue
            sx = cloud_share(s - off_s, dexel_edges(mc + dc, p[1], G[1], g[1])..., σ)
            sz = cloud_share(z - off_z, dexel_edges(mr + dr, p[2], G[2], g[2])..., σ)
            share = sx * sz
            share > 1.0e-6 || continue
            collected[(c, r)] = get(collected, (c, r), 0.0) + e * share
        end
    end
    isempty(dep.event) || flush!()

    return ArcCounts(d, readout, run.n, counts)
end

# -----------------------------------------------------------------------------
# Reading out an energy-integrating detector
# -----------------------------------------------------------------------------

"""
    ArcEnergyReadout()

How an energy-integrating [`ArcDetector`](@ref) reads out: each dexel's signal is the energy left
in its own crystal (the reflector keeps the light in the dexel, and the photodiode integrates
it). [`arc_signal`](@ref) keeps, per dexel, the sum over photons of that energy and of its
square: the mean signal and the Swank factor follow from them.
"""
struct ArcEnergyReadout end

"""
    ArcSignal

The signal of every dexel of an [`ArcDetector`](@ref) from one run: `energy_keV[column, row]`
is the energy the dexel collected, summed over photons, and `energy2_keV2` the sum over photons
of its square; `n` photons were fired, `photon_keV` is their summed energy.
"""
struct ArcSignal
    detector::ArcDetector
    n::Int
    photon_keV::Float64
    energy_keV::Matrix{Float64}
    energy2_keV2::Matrix{Float64}
end

Base.:+(a::ArcSignal, b::ArcSignal) = ArcSignal(
    a.detector,
    a.n + b.n,
    a.photon_keV + b.photon_keV,
    a.energy_keV .+ b.energy_keV,
    a.energy2_keV2 .+ b.energy2_keV2,
)

"""
    arc_signal(run, ::ArcEnergyReadout) -> ArcSignal

Sum, for every dexel of the run's [`ArcDetector`](@ref), the energy it collected from each
photon (and its square). The run must have been simulated with `record_deposits = true`.
"""
function arc_signal(run::MonoenergeticRun, ::ArcEnergyReadout)
    d = run.detector
    d isa ArcDetector || throw(ArgumentError("arc_signal needs a run on an ArcDetector"))
    run.deposits === nothing && throw(ArgumentError("simulate with record_deposits = true"))

    R = d.radius_mm
    cells = cell_count(d)
    p, G, g = d.dexel_mm, d.group, d.gap_mm
    energy = zeros(d.columns, d.rows)
    energy2 = zeros(d.columns, d.rows)

    # The energy each dexel collected from the current photon
    collected = Dict{Tuple{Int, Int}, Float64}()

    function flush!()
        for ((col, row), e) in collected
            energy[col, row] += e
            energy2[col, row] += e^2
        end
        return empty!(collected)
    end

    dep = run.deposits
    current = isempty(dep.event) ? 0 : dep.event[1]
    for k in eachindex(dep.event)
        if dep.event[k] != current
            flush!()
            current = dep.event[k]
        end
        s = R * atan(dep.x_mm[k], dep.y_mm[k])
        col = first(arc_dexel(s, p[1], G[1], g[1], cells[1]))
        row = first(arc_dexel(dep.z_mm[k], p[2], G[2], g[2], cells[2]))
        (1 <= col <= d.columns && 1 <= row <= d.rows) || continue
        collected[(col, row)] = get(collected, (col, row), 0.0) + dep.energy_keV[k]
    end
    flush!()

    drawn = run.photon_energy_keV
    photon_keV = drawn === nothing || isempty(drawn) ? run.n * run.energy_keV : sum(Float64, drawn)
    return ArcSignal(d, run.n, photon_keV, energy, energy2)
end

"""Reduce a run where it ran: dexel signals for a DexelReadout, dexel counts for an ArcReadout."""
reduce_run(run, ::Nothing) = run

# What a readout needs the engine to record: every deposit (the detectors' readouts), every photon
# scattering (forced detection)
needs_deposits(::Nothing) = false
needs_deposits(readout) = true
needs_vertices(readout) = false

reduce_run(run, readout::DexelReadout) = dexel_response(run, readout)
reduce_run(run, readout::ArcReadout) = arc_counts(run, readout)
reduce_run(run, readout::ArcEnergyReadout) = arc_signal(run, readout)
reduce_run(run, readout::DexelEnergyReadout) = dexel_energy(run, readout)
