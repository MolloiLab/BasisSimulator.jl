"""
    Geometry/Scanner.jl

CT scanner geometry with pre-computed source/detector positions.

All positions are pre-computed at construction time for Reactant compatibility.
No trigonometric functions are called during the forward/back projection.
"""

# =============================================================================
# Scanner Geometry
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
end

"""
    create_aquilion_one(; n_angles=360, n_rows=64, n_cols=128, fov_cm=nothing, sad=nothing, sdd=nothing)

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
- `fov_cm::Union{Float64,Nothing}`: If specified, adjust pixel size to cover this FOV.
  If nothing, use real scanner pixel size (0.5mm).
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

    # Determine pixel size
    if fov_cm === nothing
        pixel_size = pixel_pitch_mm / 10.0
    else
        # Compute pixel size to cover the specified FOV
        # Add some margin (1.1x) to ensure full coverage
        pixel_size = (fov_cm * 1.1) / n_cols
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

    return CTGeometry(
        SAD, SDD, n_angles, n_rows, n_cols, pixel_size,
        angles, source_positions, detector_centers, detector_u, detector_v
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

export CTGeometry, create_aquilion_one
export get_detector_pixel_position, get_source_position
