# =============================================================================
# Helical CT Protocol Integration with Scanner Specifications
# =============================================================================
#
# This file extends HelicalRecon.jl to work with clinical scanner specifications.
# It must be loaded AFTER both Scanners.jl and HelicalRecon.jl.
#
# GE Revolution Apex Helical Specifications (FDA K133705, K213715):
# - Pitch values: 0.5, 0.531, 0.969, 0.992, 1.375, 1.531
# - Z-coverage: 160 mm (256 rows × 0.625 mm)
# - Max table speed: 437 mm/s (HyperDrive mode)
# - Min rotation time: 0.23 s
#
# =============================================================================

export create_helical_geometry_from_spec

"""
    create_helical_geometry_from_spec(spec::AbstractScannerSpec, protocol::HelicalProtocol; n_rows::Int=64, n_cols::Int=0, fov_cm::Float64=0.0)

Create a HelicalGeometry from a scanner specification and helical protocol.

This is the recommended way to create helical geometry for GE Revolution Apex
and other clinical scanners.

# Arguments
- `spec::AbstractScannerSpec`: Scanner specification (e.g., GERevolutionApex())
- `protocol::HelicalProtocol`: Helical scan protocol with pitch, rotation time, etc.

# Keyword Arguments
- `n_rows::Int`: Number of detector rows to simulate (default: 64 for speed)
- `n_cols::Int`: Number of detector columns (default: from spec)
- `fov_cm::Float64`: Field of view in cm (default: from spec)

# Returns
`HelicalGeometry` with all parameters derived from scanner spec and protocol.

# Pitch Definition (GE Convention)
Pitch = table_advance_per_rotation / beam_width
- pitch < 1: Overlapping coverage (better image quality, higher dose)
- pitch = 1: Adjacent rotations just touch
- pitch > 1: Gaps between rotations (faster scan, lower dose)

# GE Revolution Apex Available Pitches (AJR 2018)
- 0.5, 0.531: Cardiac/low pitch
- 0.969, 0.992: Standard body
- 1.375, 1.531: Fast scanning

# Example
```julia
spec = GERevolutionApex()
protocol = GEApexChestHelical()  # 120 kVp, 0.5s rotation, 0.992 pitch
helical_geom = create_helical_geometry_from_spec(spec, protocol)

# Or with custom protocol
protocol = HelicalProtocol(120, 400, 0.5, 0.992, 3.0, 984, 0.625)
helical_geom = create_helical_geometry_from_spec(spec, protocol; n_rows=128)
```

# References
- FDA 510(k) K133705, K213715
- AJR 2018 GE Revolution specifications
"""
function create_helical_geometry_from_spec(
    spec::AbstractScannerSpec,
    protocol::HelicalProtocol;
    n_rows::Int = 64,
    n_cols::Int = 0,
    fov_cm::Float64 = 0.0
)
    # Get scanner specifications
    det = detector(spec)
    geom_spec = geometry(spec)

    # Determine parameters from spec or kwargs
    _n_cols = n_cols > 0 ? n_cols : det.n_cols[]
    _fov_cm = fov_cm > 0.0 ? fov_cm : geom_spec.max_sfov_mm[] / 10.0

    # Convert mm to cm for BasisSimulator
    sid_cm = geom_spec.sid_mm[] / 10.0
    sdd_cm = geom_spec.sdd_mm[] / 10.0

    # Compute beam width (z-coverage at isocenter)
    magnification = sdd_cm / sid_cm
    beam_width_cm = n_rows * (det.row_size_mm[] / 10.0) / magnification

    # Protocol parameters
    pitch = protocol.pitch
    rotation_time = protocol.rotation_time_s
    n_rotations = protocol.n_rotations
    angles_per_rotation = protocol.n_angles_per_rotation

    # Compute helical parameters
    table_advance_per_rotation = pitch * beam_width_cm  # cm per rotation
    table_speed = table_advance_per_rotation / rotation_time  # cm/s

    # Total number of projection angles
    total_angles = round(Int, angles_per_rotation * n_rotations)

    # Compute pixel size to cover FOV
    pixel_size_cm = (_fov_cm * 1.1) / _n_cols

    # Generate angles for all rotations
    angles = Vector{Float64}(undef, total_angles)
    for i in 1:total_angles
        rotation_fraction = (i - 1) / angles_per_rotation
        θ = 2π * (rotation_fraction - floor(rotation_fraction))
        angles[i] = θ
    end

    # Pre-compute all source/detector positions with helical motion
    source_positions = Matrix{Float64}(undef, 3, total_angles)
    detector_centers = Matrix{Float64}(undef, 3, total_angles)
    detector_u = Matrix{Float64}(undef, 3, total_angles)
    detector_v = Matrix{Float64}(undef, 3, total_angles)

    # Compute z-positions for each angle
    z_start = -beam_width_cm / 2  # Start half a beam width before center
    z_positions = Vector{Float64}(undef, total_angles)

    for i in 1:total_angles
        rotation_fraction = (i - 1) / angles_per_rotation
        θ = angles[i]
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Table z position (linear motion)
        z_table = z_start + rotation_fraction * table_advance_per_rotation
        z_positions[i] = z_table

        # Source position with table z
        source_positions[1, i] = -sid_cm * sinθ
        source_positions[2, i] = -sid_cm * cosθ
        source_positions[3, i] = z_table

        # Detector center with table z
        det_dist = sdd_cm - sid_cm
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = z_table

        # Detector axes (unchanged for each rotation)
        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    # Compute FOV (including z range from helical travel)
    z_range = z_positions[end] - z_positions[1]
    fov_z = z_range + beam_width_cm
    fov = (_fov_cm, _fov_cm, fov_z)

    # Create base CTGeometry
    base_geom = CTGeometry(
        sid_cm, sdd_cm, total_angles, n_rows, _n_cols, pixel_size_cm,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov
    )

    return HelicalGeometry{Float64, Vector{Float64}}(
        base_geom,
        Float64(pitch),
        Float64(table_speed),
        Float64(rotation_time),
        Float64(n_rotations),
        Float64(z_start),
        z_positions,
        angles_per_rotation,
        Float64(beam_width_cm)
    )
end
