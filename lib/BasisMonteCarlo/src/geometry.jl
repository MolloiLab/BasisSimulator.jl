# =============================================================================
# Geometry and beam
# =============================================================================
# The coordinate frame every figure and every number uses:
#   z  the beam axis; photons travel toward +z
#   the detector's entrance face sits at z = 0, its back face at z = thickness
#   x, y  across the face, centred on the beam
# The world around the detector is vacuum, so nothing outside it scatters back in.
# =============================================================================

"""
    AbstractDetector

Anything Geant4 can build as the detector: a [`SlabDetector`](@ref) (one block, with an
optional grid) or an [`ArcDetector`](@ref) (a scanner's whole detector).
"""
abstract type AbstractDetector end

"""
    Collimator(; height_mm, wall_mm, pitch_x_mm, pitch_y_mm, walls = :both, gap_mm = 0.0,
               cells = 9, tilt_x_deg = 0.0, tilt_y_deg = 0.0, material = DetectorMaterial(:W))

An anti-scatter collimator in front of the crystal: thin walls of `material` (tungsten by
default), `height_mm` tall along the beam and `wall_mm` thick, standing on the dexel
boundaries so that each opening sits over one dexel (or one group of dexels: the wall pitch
`pitch_x_mm` × `pitch_y_mm` need not equal the dexel pitch).

- `walls = :x`: walls between columns only (planes parallel to y, the classic 1-D grid);
  `:y` between rows only; `:both` a 2-D grid of square-ish cells.
- `gap_mm`: space between the bottom of the walls and the crystal's front face.
- `cells`: the walls are built over `cells × cells` openings around the beam, enough for a
  beam that lands on the central few dexels.
- `tilt_x_deg`, `tilt_y_deg`: a focused grid's walls lean toward the focal spot, so at a
  dexel that sees the source at an angle they are tilted by that angle. Give the beam's angles
  here (`angle_x_deg`, `angle_y_deg` of the [`AreaBeam`](@ref)) and the walls stand parallel to
  the photons, each still meeting the crystal over its gap.
"""
Base.@kwdef struct Collimator
    height_mm::Float64
    wall_mm::Float64
    pitch_x_mm::Float64
    pitch_y_mm::Float64
    walls::Symbol = :both
    gap_mm::Float64 = 0.0
    cells::Int = 9
    tilt_x_deg::Float64 = 0.0
    tilt_y_deg::Float64 = 0.0
    material::DetectorMaterial = DetectorMaterial(:W)
end

function Base.:(==)(a::Collimator, b::Collimator)
    same_material = a.material.name == b.material.name && a.material.density == b.material.density
    return same_material && all(
        getfield(a, f) == getfield(b, f) for f in fieldnames(Collimator) if f !== :material
    )
end

"""
    Septa(; pitch_x_mm, pitch_y_mm, width_x_mm, width_y_mm, material, cells = 9)

Reflector septa inside a scintillator: thin walls of `material` through the crystal's full
depth, `width_x_mm` (between columns) and `width_y_mm` (between rows) wide, on a grid of
`pitch_x_mm` × `pitch_y_mm`, dividing the crystal into pixels (dexels). Each dexel's crystal is
centred on a grid point, the central one on the beam axis. Energy left in a septum is not
signal. Built over `cells × cells` dexels around the beam.
"""
Base.@kwdef struct Septa
    pitch_x_mm::Float64
    pitch_y_mm::Float64
    width_x_mm::Float64
    width_y_mm::Float64
    material::DetectorMaterial
    cells::Int = 9
end

function Base.:(==)(a::Septa, b::Septa)
    same_material = a.material.name == b.material.name && a.material.density == b.material.density
    return same_material && all(getfield(a, f) == getfield(b, f) for f in fieldnames(Septa) if f !== :material)
end

"""
    SlabDetector(material, thickness_mm; width_mm = 50.0, shape = :box, collimator = nothing,
                 septa = nothing)

A single homogeneous detector block, `thickness_mm` deep along the beam (z), in vacuum. The
entrance face is at z = 0 and the photons enter it head on. Across the beam it is a
`width_mm` square (`shape = :box`) or a disc of diameter `width_mm` (`shape = :disc`, the
geometry of most published pencil-beam benchmarks). A [`Collimator`](@ref), if given, stands
in front of the entrance face; [`Septa`](@ref), if given, divide the crystal into dexels.

`material` is a [`DetectorMaterial`](@ref) or a key of [`DETECTOR_MATERIALS`](@ref). The
default 50 mm face is wide enough that nothing escapes through the sides of a 1–3 mm slab, so
the slab behaves as the semi-infinite layer of the published benchmarks.
"""
struct SlabDetector <: AbstractDetector
    material::DetectorMaterial
    thickness_mm::Float64
    width_mm::Float64
    shape::Symbol
    collimator::Union{Nothing, Collimator}
    septa::Union{Nothing, Septa}
end

function SlabDetector(
        material::DetectorMaterial,
        thickness_mm::Real;
        width_mm::Real = 50.0,
        shape::Symbol = :box,
        collimator::Union{Nothing, Collimator} = nothing,
        septa::Union{Nothing, Septa} = nothing,
    )
    shape in (:box, :disc) || throw(ArgumentError("shape must be :box or :disc"))
    return SlabDetector(
        material,
        Float64(thickness_mm),
        Float64(width_mm),
        shape,
        collimator,
        septa,
    )
end

SlabDetector(key::Symbol, thickness_mm::Real; kw...) =
    SlabDetector(DetectorMaterial(key), thickness_mm; kw...)

function Base.show(io::IO, d::SlabDetector)
    across = d.shape === :disc ? "$(d.width_mm) mm diameter" : "$(d.width_mm) mm wide"
    grid = d.collimator === nothing ? "" : ", behind a collimator"
    return print(
        io,
        "SlabDetector(",
        d.material.name,
        ", ",
        d.thickness_mm,
        " mm thick, ",
        across,
        grid,
        ")",
    )
end

"""
    PencilBeam(; x_mm = 0.0, y_mm = 0.0, angle_x_deg = 0.0, angle_y_deg = 0.0, start_mm = 10.0)

An infinitely narrow monoenergetic beam that reaches the entrance face (z = 0) at (`x_mm`,
`y_mm`). It travels along +z, tilted by `angle_x_deg` in the x–z plane (across the fan) and
`angle_y_deg` in the y–z plane (along the rows: the cone angle). Each photon starts at
z = −`start_mm` (in vacuum, so the distance changes nothing but the drawings; it must clear
any collimator).
"""
Base.@kwdef struct PencilBeam
    x_mm::Float64 = 0.0
    y_mm::Float64 = 0.0
    angle_x_deg::Float64 = 0.0
    angle_y_deg::Float64 = 0.0
    start_mm::Float64 = 10.0
end

"""
    AreaBeam(; width_mm, height_mm, x_mm = 0.0, y_mm = 0.0, angle_x_deg = 0.0, angle_y_deg = 0.0,
               start_mm = 10.0)

A monoenergetic beam whose photons reach the entrance face (z = 0) uniformly over a `width_mm`
(x) by `height_mm` (y) rectangle centred on (`x_mm`, `y_mm`): one dexel uniformly
illuminated, for example. Tilts as for [`PencilBeam`](@ref). Each photon's landing point is
drawn from a generator seeded with the run's seed, so the same seed and energy give the same
photons.
"""
Base.@kwdef struct AreaBeam
    width_mm::Float64
    height_mm::Float64
    x_mm::Float64 = 0.0
    y_mm::Float64 = 0.0
    angle_x_deg::Float64 = 0.0
    angle_y_deg::Float64 = 0.0
    start_mm::Float64 = 10.0
end

"""
    FocalSpot(; distance_mm, fan_deg, z_half_mm)

Photons from a point focal spot at the origin, aimed at a cylindrical detector of radius
`distance_mm` around it (an [`ArcDetector`](@ref)): directions spread uniformly over the fan
(`fan_deg` across, centred on +y) and over the detector's height (`±z_half_mm` at the
detector), so every dexel receives the same mean number of photons. It is the uniform
illumination of an air scan, without the inverse-square fall-off toward the edges.
"""
Base.@kwdef struct FocalSpot
    distance_mm::Float64
    fan_deg::Float64
    z_half_mm::Float64
end

"""
    RayBeam(; origin_mm, direction)

Every photon from `origin_mm` along `direction`: one ray, as a pencil-beam scatter kernel needs
(the direction need not be normalised).
"""
Base.@kwdef struct RayBeam
    origin_mm::NTuple{3, Float64}
    direction::NTuple{3, Float64}
end

beam_ray(b::RayBeam, rng) = (b.origin_mm, b.direction ./ sqrt(sum(abs2, b.direction)))

const Beam = Union{PencilBeam, AreaBeam, FocalSpot, RayBeam}

"""Where a photon of beam `b` reaches the entrance face (z = 0)."""
beam_position(b::PencilBeam, rng) = (b.x_mm, b.y_mm)

function beam_position(b::AreaBeam, rng)
    x = b.x_mm + (rand(rng) - 0.5) * b.width_mm
    y = b.y_mm + (rand(rng) - 0.5) * b.height_mm
    return (x, y)
end

"""The unit direction of travel of a parallel beam `b`."""
function beam_direction(b::Union{PencilBeam, AreaBeam})
    v = (tand(b.angle_x_deg), tand(b.angle_y_deg), 1.0)
    return v ./ sqrt(sum(abs2, v))
end

"""Start point and direction of one photon: it starts at z = −start_mm on its way to its landing point."""
function beam_ray(b::Union{PencilBeam, AreaBeam}, rng)
    x, y = beam_position(b, rng)
    u = beam_direction(b)
    t = b.start_mm / u[3]
    return (x - t * u[1], y - t * u[2], -b.start_mm), u
end

"""A photon from the focal spot: uniform over the fan angle and over the detector's height."""
function beam_ray(b::FocalSpot, rng)
    θ = deg2rad((rand(rng) - 0.5) * b.fan_deg)
    z = (rand(rng) - 0.5) * 2 * b.z_half_mm
    v = (b.distance_mm * sin(θ), b.distance_mm * cos(θ), z)
    return (0.0, 0.0, 0.0), v ./ sqrt(sum(abs2, v))
end

# -----------------------------------------------------------------------------
# Geant4 side. One G4JLDetector type wraps whatever geometry is current, so the run manager
# can be re-initialised with a new geometry without a new application.
# -----------------------------------------------------------------------------

mutable struct G4Geometry <: G4JLDetector
    detector::AbstractDetector
end

# Copy numbers: the crystal is 1 (the only volume that counts as "the detector"); collimator
# walls are 2.
const CRYSTAL_COPY = 1
const COLLIMATOR_COPY = 2
const SEPTUM_COPY = 3

function build_world(g::G4Geometry)::CxxPtr{G4VPhysicalVolume}
    d = g.detector
    nist = G4NistManager!Instance()

    # The world: vacuum, comfortably larger than the detector
    vacuum = FindOrBuildMaterial(nist, "G4_Galactic")
    half = world_half_mm(d) * mm
    world = G4LogicalVolume(G4Box("World", half, half, half), vacuum, "World")
    world_pv = G4PVPlacement(nothing, G4ThreeVector(), world, "World", nothing, false, 0)

    build_detector!(world, d)
    return world_pv
end

world_half_mm(d::SlabDetector) = max(d.width_mm, d.thickness_mm) + 50

function build_detector!(world, d::SlabDetector)
    # The crystal, with its entrance face at z = 0
    solid = if d.shape === :disc
        G4Tubs("Detector", 0, d.width_mm * mm / 2, d.thickness_mm * mm / 2, 0, 2π)
    else
        G4Box("Detector", d.width_mm * mm / 2, d.width_mm * mm / 2, d.thickness_mm * mm / 2)
    end
    crystal = G4LogicalVolume(solid, g4_material(d.material), "Detector")
    G4PVPlacement(
        nothing,
        G4ThreeVector(0, 0, d.thickness_mm * mm / 2),
        crystal,
        "Detector",
        world,
        false,
        CRYSTAL_COPY,
    )

    d.collimator === nothing || build_collimator!(world, d.collimator)
    d.septa === nothing || build_septa!(crystal, d.septa, d.thickness_mm)
    return nothing
end

"""
Place reflector septa inside the crystal: full-length walls between columns, segments between
rows. They are daughters of the crystal, so a step inside one is not in the crystal.
"""
function build_septa!(crystal, s::Septa, thickness_mm)
    reflector = g4_material(s.material)
    h = s.cells ÷ 2
    span_x, span_y = s.cells * s.pitch_x_mm, s.cells * s.pitch_y_mm
    wall_x = G4LogicalVolume(
        G4Box("SeptumX", s.width_x_mm * mm / 2, span_y * mm / 2, thickness_mm * mm / 2),
        reflector,
        "SeptumX",
    )
    for i in (-h - 1):h
        G4PVPlacement(
            nothing,
            G4ThreeVector((i + 0.5) * s.pitch_x_mm * mm, 0, 0),
            wall_x,
            "SeptumX",
            crystal,
            false,
            SEPTUM_COPY,
        )
    end
    wall_y = G4LogicalVolume(
        G4Box(
            "SeptumY",
            (s.pitch_x_mm - s.width_x_mm) * mm / 2,
            s.width_y_mm * mm / 2,
            thickness_mm * mm / 2,
        ),
        reflector,
        "SeptumY",
    )
    for i in (-h):h, j in (-h - 1):h
        G4PVPlacement(
            nothing,
            G4ThreeVector(i * s.pitch_x_mm * mm, (j + 0.5) * s.pitch_y_mm * mm, 0),
            wall_y,
            "SeptumY",
            crystal,
            false,
            SEPTUM_COPY,
        )
    end
    return nothing
end

"""
Place the collimator walls in front of the crystal. Walls between columns run the full length
of the grid; walls between rows are cut into segments between them, so no two volumes overlap.
Tilted (focused) walls lean along the beam and still meet the crystal over their gaps.
"""
function build_collimator!(world, c::Collimator)
    tungsten = g4_material(c.material)
    h = c.cells ÷ 2
    z_centre = -(c.gap_mm + c.height_mm / 2)
    span_x = c.cells * c.pitch_x_mm
    span_y = c.cells * c.pitch_y_mm

    # A tilted wall's centre sits where the ray through its foot is, at the wall's mid-height
    shift_x = z_centre * tand(c.tilt_x_deg)
    shift_y = z_centre * tand(c.tilt_y_deg)

    function place!(logical, name, x, y, rotation)
        transform = G4Transform3D(rotation, G4ThreeVector(x * mm, y * mm, z_centre * mm))
        return G4PVPlacement(transform, logical, name, world, false, COLLIMATOR_COPY)
    end

    # Walls between columns: planes at x = (i + ½) · pitch_x, along the whole grid in y
    if c.walls in (:x, :both)
        wall = G4LogicalVolume(
            G4Box("WallX", c.wall_mm * mm / 2, span_y * mm / 2, c.height_mm * mm / 2),
            tungsten,
            "WallX",
        )
        for i in (-h - 1):h
            rotation = G4RotationMatrix()
            rotateY(rotation, deg2rad(c.tilt_x_deg))
            place!(wall, "WallX", (i + 0.5) * c.pitch_x_mm + shift_x, shift_y, rotation)
        end
    end

    # Walls between rows: planes at y = (j + ½) · pitch_y; between the column walls if there are any
    if c.walls in (:y, :both)
        length_x = c.walls === :both ? c.pitch_x_mm - c.wall_mm : span_x
        wall = G4LogicalVolume(
            G4Box("WallY", length_x * mm / 2, c.wall_mm * mm / 2, c.height_mm * mm / 2),
            tungsten,
            "WallY",
        )
        columns = c.walls === :both ? ((-h):h) : (0:0)
        for i in columns, j in (-h - 1):h
            rotation = G4RotationMatrix()
            rotateX(rotation, -deg2rad(c.tilt_y_deg))
            place!(
                wall,
                "WallY",
                i * c.pitch_x_mm + shift_x,
                (j + 0.5) * c.pitch_y_mm + shift_y,
                rotation,
            )
        end
    end

    return nothing
end

Geant4.getConstructor(::G4Geometry)::Function = build_world
