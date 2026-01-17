"""
    Geometry/Scanner.jl

CT scanner geometry definitions and pre-computed trajectory positions.

This file provides two complementary abstractions:

1. `Scanner{T}` - Generic scanner definition struct with all physical parameters.
   Accepts kwargs for flexible scanner configuration.

2. `CTGeometry` - Pre-computed source/detector positions for simulation.
   All positions computed at construction for Reactant/XLA compatibility.

# Workflow
```julia
# Define a scanner with physical parameters
scanner = Scanner(
    source_to_isocenter = 541.0,  # mm
    source_to_detector = 949.0,   # mm
    detector_rows = 64,
    detector_cols = 900
)

# Create computed geometry for simulation
geom = CTGeometry(scanner; n_angles=360, fov_cm=35.0)

# Or use legacy factory functions
geom = create_aquilion_one(n_angles=360, n_rows=64, fov_cm=35.0)
```
"""

# =============================================================================
# Scanner Definition Struct (Physical Parameters)
# =============================================================================

"""
    DetectorShape

Enumeration of detector array geometries.
"""
@enum DetectorShape begin
    CURVED_DETECTOR    # Third-generation CT: curved detector array
    FLAT_DETECTOR      # Flat panel detectors (CBCT, interventional)
end

"""
    Scanner{T<:AbstractFloat}

Generic CT scanner definition with all physical parameters.

This struct defines the physical scanner configuration. Use `CTGeometry` for
simulation with pre-computed trajectories. All distances in mm for consistency
with CatSim and medical imaging conventions.

# Geometry Parameters (Required)
- `source_to_isocenter::T`: Source-to-isocenter distance (mm), aka SID/SOD
- `source_to_detector::T`: Source-to-detector distance (mm), aka SDD

# Detector Parameters
- `detector_rows::Int`: Number of detector rows (z-direction)
- `detector_cols::Int`: Number of detector columns (fan direction)
- `detector_row_size::T`: Detector element size in z (mm) at isocenter
- `detector_col_size::T`: Detector element size in fan direction (mm) at isocenter
- `detector_shape::DetectorShape`: Detector geometry (:curved or :flat)
- `detector_row_offset::T`: Row offset from centered position (rows)
- `detector_col_offset::T`: Column offset (quarter-detector offset for aliasing)

# Source Parameters
- `focal_spot_width::T`: Focal spot width (mm)
- `focal_spot_length::T`: Focal spot length (mm)
- `target_angle::T`: Anode target angle (degrees)

# Gantry Parameters
- `gantry_rotation_time::T`: Gantry rotation time (seconds)
- `max_scan_fov::T`: Maximum scan field of view (mm)
- `gantry_aperture::T`: Gantry bore diameter (mm)

# Filter Parameters
- `flat_filter_material::Symbol`: Flat filter material (:aluminum, :copper, :titanium)
- `flat_filter_thickness::T`: Flat filter thickness (mm)

# Detection Parameters
- `detector_material::Symbol`: Detector scintillator/sensor material
- `detector_depth::T`: Detector sensor depth (mm)
- `fill_factor_row::T`: Active area fraction (row direction, 0-1)
- `fill_factor_col::T`: Active area fraction (column direction, 0-1)
- `detection_gain::T`: Conversion gain (electrons/keV)
- `electronic_noise::T`: Electronic noise std dev (electrons)

# Constructor
```julia
Scanner(;
    source_to_isocenter = 540.0,
    source_to_detector = 950.0,
    detector_rows = 64,
    detector_cols = 900,
    # ... other kwargs with defaults
)
```

# CatSim Parameter Mapping
| Scanner Field | CatSim Parameter |
|---------------|------------------|
| source_to_isocenter | scanner.sid |
| source_to_detector | scanner.sdd |
| detector_rows | scanner.detectorRowCount |
| detector_cols | scanner.detectorColCount |
| detector_row_size | scanner.detectorRowSize |
| detector_col_size | scanner.detectorColSize |
| detector_row_offset | scanner.detectorRowOffset |
| detector_col_offset | scanner.detectorColOffset |
| target_angle | scanner.targetAngle |
| focal_spot_width | scanner.focalspotWidth |
| focal_spot_length | scanner.focalspotLength |
| detector_material | scanner.detectorMaterial |
| detector_depth | scanner.detectorDepth |
| fill_factor_row | scanner.detectorRowFillFraction |
| fill_factor_col | scanner.detectorColFillFraction |
| detection_gain | scanner.detectionGain |
| electronic_noise | scanner.eNoise |

# References
- CatSim scanner configuration: cfg/Scanner_Default.cfg
- AAPM TG-233: CT Image Quality Standards
"""
struct Scanner{T<:AbstractFloat}
    # Geometry (required)
    source_to_isocenter::T      # mm (SID/SOD)
    source_to_detector::T       # mm (SDD)

    # Detector array
    detector_rows::Int          # number of rows
    detector_cols::Int          # number of columns
    detector_row_size::T        # mm at isocenter
    detector_col_size::T        # mm at isocenter
    detector_shape::DetectorShape
    detector_row_offset::T      # rows
    detector_col_offset::T      # columns (quarter-detector offset)

    # Source/focal spot
    focal_spot_width::T         # mm
    focal_spot_length::T        # mm
    target_angle::T             # degrees

    # Gantry
    gantry_rotation_time::T     # seconds
    max_scan_fov::T             # mm
    gantry_aperture::T          # mm

    # Filters
    flat_filter_material::Symbol
    flat_filter_thickness::T    # mm

    # Detection
    detector_material::Symbol
    detector_depth::T           # mm
    fill_factor_row::T          # 0-1
    fill_factor_col::T          # 0-1
    detection_gain::T           # electrons/keV
    electronic_noise::T         # electrons
end

"""
    Scanner(; kwargs...)

Construct a Scanner with configurable parameters via kwargs.

All distances are in mm. Default values match a generic research CT scanner
(similar to CatSim defaults).

# Keyword Arguments (with defaults)
- `source_to_isocenter::Real = 540.0`: Source-to-isocenter distance (mm)
- `source_to_detector::Real = 950.0`: Source-to-detector distance (mm)
- `detector_rows::Int = 64`: Number of detector rows
- `detector_cols::Int = 900`: Number of detector columns
- `detector_row_size::Real = 1.0`: Detector row pitch (mm)
- `detector_col_size::Real = 1.0`: Detector column pitch (mm)
- `detector_shape::DetectorShape = CURVED_DETECTOR`: Detector geometry
- `detector_row_offset::Real = 0.0`: Row offset (rows)
- `detector_col_offset::Real = 0.25`: Column offset for quarter-detector shift
- `focal_spot_width::Real = 1.0`: Focal spot width (mm)
- `focal_spot_length::Real = 1.0`: Focal spot length (mm)
- `target_angle::Real = 7.0`: Anode target angle (degrees)
- `gantry_rotation_time::Real = 0.5`: Rotation time (seconds)
- `max_scan_fov::Real = 500.0`: Maximum scan FOV (mm)
- `gantry_aperture::Real = 700.0`: Gantry bore diameter (mm)
- `flat_filter_material::Symbol = :aluminum`: Flat filter material
- `flat_filter_thickness::Real = 2.0`: Flat filter thickness (mm)
- `detector_material::Symbol = :lumex`: Detector scintillator
- `detector_depth::Real = 3.0`: Detector depth (mm)
- `fill_factor_row::Real = 0.9`: Row fill factor (0-1)
- `fill_factor_col::Real = 0.9`: Column fill factor (0-1)
- `detection_gain::Real = 15.0`: Detection gain (electrons/keV)
- `electronic_noise::Real = 5000.0`: Electronic noise (electrons)

# Example
```julia
# Generic research scanner (defaults)
scanner = Scanner()

# Custom scanner with specific geometry
scanner = Scanner(
    source_to_isocenter = 626.0,  # GE Revolution-like
    source_to_detector = 1097.0,
    detector_rows = 256,
    detector_cols = 832,
    detector_row_size = 0.625,
    target_angle = 10.0
)

# Flat panel scanner
scanner = Scanner(
    detector_shape = FLAT_DETECTOR,
    detector_rows = 512,
    detector_cols = 512,
    detector_row_size = 0.15,
    detector_col_size = 0.15
)
```
"""
function Scanner(;
    # Geometry (CatSim defaults)
    source_to_isocenter::Real = 540.0,
    source_to_detector::Real = 950.0,

    # Detector array
    detector_rows::Int = 64,
    detector_cols::Int = 900,
    detector_row_size::Real = 1.0,
    detector_col_size::Real = 1.0,
    detector_shape::DetectorShape = CURVED_DETECTOR,
    detector_row_offset::Real = 0.0,
    detector_col_offset::Real = 0.25,

    # Source/focal spot
    focal_spot_width::Real = 1.0,
    focal_spot_length::Real = 1.0,
    target_angle::Real = 7.0,

    # Gantry
    gantry_rotation_time::Real = 0.5,
    max_scan_fov::Real = 500.0,
    gantry_aperture::Real = 700.0,

    # Filters
    flat_filter_material::Symbol = :aluminum,
    flat_filter_thickness::Real = 2.0,

    # Detection
    detector_material::Symbol = :lumex,
    detector_depth::Real = 3.0,
    fill_factor_row::Real = 0.9,
    fill_factor_col::Real = 0.9,
    detection_gain::Real = 15.0,
    electronic_noise::Real = 5000.0
)
    T = Float64
    return Scanner{T}(
        T(source_to_isocenter),
        T(source_to_detector),
        detector_rows,
        detector_cols,
        T(detector_row_size),
        T(detector_col_size),
        detector_shape,
        T(detector_row_offset),
        T(detector_col_offset),
        T(focal_spot_width),
        T(focal_spot_length),
        T(target_angle),
        T(gantry_rotation_time),
        T(max_scan_fov),
        T(gantry_aperture),
        flat_filter_material,
        T(flat_filter_thickness),
        detector_material,
        T(detector_depth),
        T(fill_factor_row),
        T(fill_factor_col),
        T(detection_gain),
        T(electronic_noise)
    )
end

"""
    validate_scanner(scanner::Scanner) -> (valid::Bool, messages::Vector{String})

Validate scanner geometric consistency.

Checks:
1. Source geometry: SID < SDD (source must be between isocenter and detector)
2. Source geometry: Both distances must be positive
3. Detector array: Rows and columns must be positive integers
4. Detector geometry: Pixel sizes must be positive
5. Fill factors: Must be between 0 and 1
6. FOV consistency: Max FOV fits within detector coverage
7. Target angle: Must be positive and < 90 degrees

# Returns
- `valid::Bool`: True if all checks pass
- `messages::Vector{String}`: List of validation messages (errors and warnings)

# Example
```julia
scanner = Scanner(source_to_detector = 500.0)  # Invalid: SDD < default SID
valid, msgs = validate_scanner(scanner)
# valid = false
# msgs = ["ERROR: source_to_detector (500.0) must be > source_to_isocenter (540.0)"]
```
"""
function validate_scanner(scanner::Scanner{T}) where T
    messages = String[]
    valid = true

    # Geometry checks
    if scanner.source_to_isocenter <= 0
        push!(messages, "ERROR: source_to_isocenter must be positive (got $(scanner.source_to_isocenter))")
        valid = false
    end

    if scanner.source_to_detector <= 0
        push!(messages, "ERROR: source_to_detector must be positive (got $(scanner.source_to_detector))")
        valid = false
    end

    if scanner.source_to_detector <= scanner.source_to_isocenter
        push!(messages, "ERROR: source_to_detector ($(scanner.source_to_detector)) must be > source_to_isocenter ($(scanner.source_to_isocenter))")
        valid = false
    end

    # Detector array checks
    if scanner.detector_rows <= 0
        push!(messages, "ERROR: detector_rows must be positive (got $(scanner.detector_rows))")
        valid = false
    end

    if scanner.detector_cols <= 0
        push!(messages, "ERROR: detector_cols must be positive (got $(scanner.detector_cols))")
        valid = false
    end

    if scanner.detector_row_size <= 0
        push!(messages, "ERROR: detector_row_size must be positive (got $(scanner.detector_row_size))")
        valid = false
    end

    if scanner.detector_col_size <= 0
        push!(messages, "ERROR: detector_col_size must be positive (got $(scanner.detector_col_size))")
        valid = false
    end

    # Fill factor checks
    if !(0 < scanner.fill_factor_row <= 1)
        push!(messages, "ERROR: fill_factor_row must be in (0, 1] (got $(scanner.fill_factor_row))")
        valid = false
    end

    if !(0 < scanner.fill_factor_col <= 1)
        push!(messages, "ERROR: fill_factor_col must be in (0, 1] (got $(scanner.fill_factor_col))")
        valid = false
    end

    # Target angle check
    if !(0 < scanner.target_angle < 90)
        push!(messages, "ERROR: target_angle must be in (0, 90) degrees (got $(scanner.target_angle))")
        valid = false
    end

    # Gantry checks
    if scanner.gantry_rotation_time <= 0
        push!(messages, "ERROR: gantry_rotation_time must be positive (got $(scanner.gantry_rotation_time))")
        valid = false
    end

    # FOV consistency warning
    # Detector coverage at isocenter = detector_cols * detector_col_size * (SID/SDD)
    magnification = scanner.source_to_detector / scanner.source_to_isocenter
    detector_coverage_at_iso = scanner.detector_cols * scanner.detector_col_size / magnification
    if scanner.max_scan_fov > detector_coverage_at_iso
        push!(messages, "WARNING: max_scan_fov ($(scanner.max_scan_fov) mm) exceeds detector coverage at isocenter ($(round(detector_coverage_at_iso, digits=1)) mm)")
    end

    # Z coverage
    z_coverage = scanner.detector_rows * scanner.detector_row_size / magnification
    if z_coverage > 0
        push!(messages, "INFO: Z coverage at isocenter: $(round(z_coverage, digits=1)) mm")
    end

    return valid, messages
end

"""
    print_scanner_summary(scanner::Scanner)

Print a summary of scanner parameters.
"""
function print_scanner_summary(scanner::Scanner{T}) where T
    println("=" ^ 60)
    println("Scanner Configuration")
    println("=" ^ 60)
    println()
    println("GEOMETRY")
    println("-" ^ 40)
    println("  Source-to-Isocenter:  $(scanner.source_to_isocenter) mm")
    println("  Source-to-Detector:   $(scanner.source_to_detector) mm")
    magnification = scanner.source_to_detector / scanner.source_to_isocenter
    println("  Magnification:        $(round(magnification, digits=3))")
    println("  Max Scan FOV:         $(scanner.max_scan_fov) mm")
    println("  Gantry Aperture:      $(scanner.gantry_aperture) mm")
    println()
    println("DETECTOR ($(scanner.detector_shape))")
    println("-" ^ 40)
    println("  Array Size:           $(scanner.detector_cols) × $(scanner.detector_rows)")
    println("  Element Size:         $(scanner.detector_col_size) × $(scanner.detector_row_size) mm")
    println("  Offset (col, row):    $(scanner.detector_col_offset), $(scanner.detector_row_offset)")
    z_coverage = scanner.detector_rows * scanner.detector_row_size / magnification
    println("  Z Coverage (iso):     $(round(z_coverage, digits=1)) mm")
    println("  Material:             $(scanner.detector_material)")
    println("  Depth:                $(scanner.detector_depth) mm")
    println("  Fill Factor:          $(scanner.fill_factor_col) × $(scanner.fill_factor_row)")
    println()
    println("X-RAY SOURCE")
    println("-" ^ 40)
    println("  Focal Spot:           $(scanner.focal_spot_width) × $(scanner.focal_spot_length) mm")
    println("  Target Angle:         $(scanner.target_angle)°")
    println("  Flat Filter:          $(scanner.flat_filter_thickness) mm $(scanner.flat_filter_material)")
    println()
    println("ACQUISITION")
    println("-" ^ 40)
    println("  Rotation Time:        $(scanner.gantry_rotation_time) s")
    println("  Detection Gain:       $(scanner.detection_gain) e⁻/keV")
    println("  Electronic Noise:     $(scanner.electronic_noise) e⁻")
    println("=" ^ 60)
end

# =============================================================================
# CTGeometry - Pre-computed Trajectory Positions
# =============================================================================

"""
    CTGeometry

CT scanner geometry with pre-computed trajectories.

All source and detector positions are pre-computed at construction time
to enable Reactant/XLA compilation (no runtime trig).

# Fields
- `SAD::Float64`: Source-to-axis distance (cm)
- `SDD::Float64`: Source-to-detector distance (cm)
- `n_angles::Int`: Number of projection angles
- `n_rows::Int`: Detector rows (z direction)
- `n_cols::Int`: Detector columns (fan direction)
- `pixel_size::Float64`: Detector pixel size (cm) at isocenter
- `angles::Vector{Float64}`: Projection angles (radians)
- `source_positions::Matrix{Float64}`: [3, n_angles] source XYZ positions
- `detector_centers::Matrix{Float64}`: [3, n_angles] detector center XYZ
- `detector_u::Matrix{Float64}`: [3, n_angles] detector u-axis (column direction)
- `detector_v::Matrix{Float64}`: [3, n_angles] detector v-axis (row direction)
- `fov::NTuple{3, Float64}`: (fov_x, fov_y, fov_z) volume FOV in cm

# Coordinate System
- X: left-right (increasing right)
- Y: anterior-posterior (source starts at -SAD on Y-axis)
- Z: inferior-superior (increasing superior)
- Rotation around Z-axis, counter-clockwise when viewed from above
"""
struct CTGeometry
    SAD::Float64
    SDD::Float64
    n_angles::Int
    n_rows::Int
    n_cols::Int
    pixel_size::Float64
    angles::Vector{Float64}
    source_positions::Matrix{Float64}
    detector_centers::Matrix{Float64}
    detector_u::Matrix{Float64}
    detector_v::Matrix{Float64}
    fov::NTuple{3, Float64}  # (fov_x, fov_y, fov_z) in cm
end

"""
    CTGeometry(scanner::Scanner; n_angles=360, fov_cm=nothing, z_cm=nothing, n_rows=nothing, n_cols=nothing)

Create a CTGeometry from a Scanner definition.

This constructor converts the physical Scanner parameters into pre-computed
trajectory positions suitable for simulation.

# Arguments
- `scanner::Scanner`: Scanner definition with physical parameters (in mm)

# Keyword Arguments
- `n_angles::Int = 360`: Number of projection angles
- `fov_cm::Union{Float64,Nothing} = nothing`: XY FOV in cm. If nothing, uses scanner.max_scan_fov/10.
- `z_cm::Union{Float64,Nothing} = nothing`: Z FOV in cm. If nothing, computes from detector coverage.
- `n_rows::Union{Int,Nothing} = nothing`: Override detector rows. If nothing, uses scanner.detector_rows.
- `n_cols::Union{Int,Nothing} = nothing`: Override detector columns. If nothing, uses scanner.detector_cols.

# Returns
`CTGeometry` with pre-computed source/detector positions.

# Example
```julia
# Create scanner and geometry
scanner = Scanner(
    source_to_isocenter = 541.0,
    source_to_detector = 949.0,
    detector_rows = 64,
    detector_cols = 900
)
geom = CTGeometry(scanner; n_angles=360, fov_cm=35.0)

# Or with reduced detector for fast testing
geom_fast = CTGeometry(scanner; n_angles=90, n_rows=16, n_cols=128, fov_cm=35.0)
```
"""
function CTGeometry(scanner::Scanner{T};
    n_angles::Int = 360,
    fov_cm::Union{Float64,Nothing} = nothing,
    z_cm::Union{Float64,Nothing} = nothing,
    n_rows::Union{Int,Nothing} = nothing,
    n_cols::Union{Int,Nothing} = nothing
) where T

    # Use scanner values or overrides
    _n_rows = n_rows !== nothing ? n_rows : scanner.detector_rows
    _n_cols = n_cols !== nothing ? n_cols : scanner.detector_cols

    # Convert mm to cm (BasisSimulator internal unit)
    SAD = scanner.source_to_isocenter / 10.0
    SDD = scanner.source_to_detector / 10.0

    # Determine FOV and pixel size
    if fov_cm !== nothing
        fov_xy = fov_cm
        # Compute pixel size to cover the specified FOV with margin
        pixel_size = (fov_cm * 1.1) / _n_cols
    else
        # Use scanner's max FOV
        fov_xy = scanner.max_scan_fov / 10.0
        pixel_size = (fov_xy * 1.1) / _n_cols
    end

    # Z FOV
    if z_cm !== nothing
        fov_z = z_cm
    else
        # Compute from detector coverage at isocenter
        magnification = scanner.source_to_detector / scanner.source_to_isocenter
        z_coverage_mm = scanner.detector_rows * scanner.detector_row_size / magnification
        fov_z = z_coverage_mm / 10.0  # Convert to cm
    end

    # Generate angles (full 360° rotation)
    angles = collect(range(0.0, 2π - 2π/n_angles, length=n_angles))

    # Pre-compute all positions
    source_positions = Matrix{Float64}(undef, 3, n_angles)
    detector_centers = Matrix{Float64}(undef, 3, n_angles)
    detector_u = Matrix{Float64}(undef, 3, n_angles)
    detector_v = Matrix{Float64}(undef, 3, n_angles)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Source position: starts at (0, -SAD, 0), rotates around Z
        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = 0.0

        # Detector center: opposite side of source
        det_dist = SDD - SAD
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = 0.0

        # Detector u-axis (column direction)
        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        # Detector v-axis (row direction): always +Z
        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    fov = (fov_xy, fov_xy, fov_z)

    return CTGeometry(
        SAD, SDD, n_angles, _n_rows, _n_cols, pixel_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov
    )
end

"""
    create_aquilion_one(; n_angles=360, n_rows=64, n_cols=128, fov_cm=nothing, z_cm=nothing, sad=nothing, sdd=nothing)

Create CT scanner geometry (defaults to Canon Aquilion ONE specifications).

# Canon Aquilion ONE Specifications (defaults)
- SAD: 600 mm (source-to-axis distance)
- SDD: 1000 mm (source-to-detector distance)
- Detector: 320 rows × 896 columns (full)
- Pixel pitch: 0.5 mm at isocenter
- Z-coverage: 16 cm (320 × 0.5 mm)

# Arguments
- `n_angles::Int`: Number of projection angles (default 360)
- `n_rows::Int`: Detector rows, reduced for fast iteration (default 64)
- `n_cols::Int`: Detector columns, reduced for fast iteration (default 128)
- `fov_cm::Union{Float64,Nothing}`: XY field of view in cm. If nothing, compute from pixel size.
- `z_cm::Union{Float64,Nothing}`: Z field of view in cm. If nothing, compute from pixel size and n_rows.
- `sad::Union{Float64,Nothing}`: Source-to-axis distance in cm (default 60.0 cm / 600 mm)
- `sdd::Union{Float64,Nothing}`: Source-to-detector distance in cm (default 100.0 cm / 1000 mm)

# Returns
`CTGeometry` with pre-computed source/detector positions.
"""
function create_aquilion_one(;
    n_angles::Int=360,
    n_rows::Int=64,
    n_cols::Int=128,
    fov_cm::Union{Float64,Nothing}=nothing,
    z_cm::Union{Float64,Nothing}=nothing,
    sad::Union{Float64,Nothing}=nothing,
    sdd::Union{Float64,Nothing}=nothing
)
    # Canon Aquilion ONE specifications (defaults)
    SAD_mm = 600.0   # Source-to-axis distance (mm)
    SDD_mm = 1000.0  # Source-to-detector distance (mm)
    pixel_pitch_mm = 0.5  # At isocenter (mm)

    # Use custom SAD/SDD if provided (input in cm), otherwise use defaults
    SAD = sad !== nothing ? sad : SAD_mm / 10.0
    SDD = sdd !== nothing ? sdd : SDD_mm / 10.0

    # Determine pixel size and FOV
    if fov_cm === nothing
        pixel_size = pixel_pitch_mm / 10.0
        fov_xy = pixel_size * n_cols  # FOV from detector size
    else
        # Compute pixel size to cover the specified FOV
        # Add some margin (1.1x) to ensure full coverage
        pixel_size = (fov_cm * 1.1) / n_cols
        fov_xy = fov_cm
    end

    # Z FOV
    if z_cm === nothing
        fov_z = pixel_size * n_rows
    else
        fov_z = z_cm
    end

    # Generate angles (full 360° rotation)
    angles = collect(range(0.0, 2π - 2π/n_angles, length=n_angles))

    # Pre-compute all positions
    source_positions = Matrix{Float64}(undef, 3, n_angles)
    detector_centers = Matrix{Float64}(undef, 3, n_angles)
    detector_u = Matrix{Float64}(undef, 3, n_angles)
    detector_v = Matrix{Float64}(undef, 3, n_angles)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Source position: starts at (0, -SAD, 0), rotates around Z
        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = 0.0

        # Detector center: opposite side of source
        det_dist = SDD - SAD  # Distance from isocenter to detector
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = 0.0

        # Detector u-axis (column direction): perpendicular to source-detector line, in XY plane
        # Points in the direction of increasing column index
        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        # Detector v-axis (row direction): always points in +Z
        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    fov = (fov_xy, fov_xy, fov_z)

    return CTGeometry(
        SAD, SDD, n_angles, n_rows, n_cols, pixel_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov
    )
end

"""
    get_detector_pixel_position(geom::CTGeometry, angle_idx::Int, row::Int, col::Int)

Get the 3D position of a detector pixel center.

# Arguments
- `geom::CTGeometry`: Scanner geometry
- `angle_idx::Int`: Projection angle index (1-based)
- `row::Int`: Detector row (1-based, 1 = bottom)
- `col::Int`: Detector column (1-based, 1 = left when facing detector)

# Returns
`(x, y, z)` position of pixel center in cm.
"""
function get_detector_pixel_position(geom::CTGeometry, angle_idx::Int, row::Int, col::Int)
    # Pixel offsets from detector center (in detector coordinates)
    # u: column direction, v: row direction
    u_offset = (col - (geom.n_cols + 1) / 2) * geom.pixel_size * (geom.SDD / geom.SAD)
    v_offset = (row - (geom.n_rows + 1) / 2) * geom.pixel_size * (geom.SDD / geom.SAD)

    # Convert to world coordinates
    x = geom.detector_centers[1, angle_idx] + u_offset * geom.detector_u[1, angle_idx] + v_offset * geom.detector_v[1, angle_idx]
    y = geom.detector_centers[2, angle_idx] + u_offset * geom.detector_u[2, angle_idx] + v_offset * geom.detector_v[2, angle_idx]
    z = geom.detector_centers[3, angle_idx] + u_offset * geom.detector_u[3, angle_idx] + v_offset * geom.detector_v[3, angle_idx]

    return (x, y, z)
end

"""
    get_source_position(geom::CTGeometry, angle_idx::Int)

Get the source position for a given angle.

# Returns
`(x, y, z)` position of X-ray source in cm.
"""
function get_source_position(geom::CTGeometry, angle_idx::Int)
    return (
        geom.source_positions[1, angle_idx],
        geom.source_positions[2, angle_idx],
        geom.source_positions[3, angle_idx]
    )
end

# =============================================================================
# Exports
# =============================================================================

# Scanner definition
export Scanner, DetectorShape, CURVED_DETECTOR, FLAT_DETECTOR
export validate_scanner, print_scanner_summary

# CTGeometry (computed positions for simulation)
export CTGeometry, create_aquilion_one
export get_detector_pixel_position, get_source_position
