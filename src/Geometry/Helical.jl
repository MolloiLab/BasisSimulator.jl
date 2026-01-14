"""
    Geometry/Helical.jl

Helical (spiral) CT scanning mode.

This module extends CTGeometry to support helical scanning where the table
moves continuously during acquisition. The design mirrors physical scanner
protocol setup:

1. **Scan Mode**: `:axial` (step-and-shoot) or `:helical` (spiral)
2. **Pitch**: Table movement per rotation / detector collimation
3. **Table Position**: Start Z position

On a physical scanner, you'd set:
- Protocol → Helical/Axial
- Pitch factor (e.g., 0.5, 1.0, 1.5)
- Table start position
- Number of rotations

FDK Considerations:
- Axial (volume): Standard FDK cone-beam reconstruction works directly
- Helical: Requires interpolation to pseudo-axial data before FDK,
  or specialized helical reconstruction (Katsevich algorithm)

Note: Implementation designed for Reactant/XLA compatibility.
"""

# =============================================================================
# Scan Mode Enum
# =============================================================================

"""
    ScanMode

CT scan acquisition mode.

- `:axial`: Step-and-shoot acquisition (table stationary during rotation)
- `:helical`: Spiral acquisition (table moves continuously)
"""
@enum ScanMode begin
    AXIAL = 1
    HELICAL = 2
end

# =============================================================================
# Extended Geometry with Table Motion
# =============================================================================

"""
    create_scan_geometry(; scanner=:aquilion_one, mode=:axial, kwargs...)

Create scanner geometry with protocol-like parameters.

This is the recommended way to create geometries as it mirrors how you'd
set up a scan protocol on a physical scanner.

# Scanner Presets
- `:aquilion_one`: Canon Aquilion ONE (600mm SAD, 1000mm SDD)

# Scan Mode
- `:axial`: Table stationary, single or multiple axial rotations
- `:helical`: Table moves continuously during rotation

# Common Arguments
- `n_angles::Int`: Projections per rotation (default: 360)
- `n_rows::Int`: Detector rows (default: 64)
- `n_cols::Int`: Detector columns (default: 128)
- `fov_cm::Float64`: Field of view in cm (optional)

# Helical-Specific Arguments (only used when mode=:helical)
- `pitch::Float64`: Pitch factor (default: 1.0)
  - pitch = table_movement_per_rotation / detector_collimation
  - pitch < 1: overlapping coverage (better quality)
  - pitch = 1: adjacent rotations just touch
  - pitch > 1: gaps between rotations (faster scan)
- `n_rotations::Float64`: Number of gantry rotations (default: 1.0)
- `z_start::Float64`: Starting table Z position in cm (default: 0.0)

# Returns
`CTGeometry` with appropriate source/detector positions.
For helical mode, Z-positions account for continuous table motion.

# Examples
```julia
# Standard axial scan (like volume CT)
geom = create_scan_geometry(mode=:axial, n_angles=360)

# Helical scan with pitch 1.0
geom = create_scan_geometry(mode=:helical, pitch=1.0, n_rotations=3.0)

# Fast helical scan with pitch 1.5
geom = create_scan_geometry(mode=:helical, pitch=1.5, n_rotations=5.0)
```
"""
function create_scan_geometry(;
    scanner::Symbol=:aquilion_one,
    mode::Symbol=:axial,
    n_angles::Int=360,
    n_rows::Int=64,
    n_cols::Int=128,
    fov_cm::Union{Float64,Nothing}=nothing,
    # Helical parameters
    pitch::Float64=1.0,
    n_rotations::Float64=1.0,
    z_start::Float64=0.0
)
    # Get scanner specifications
    if scanner == :aquilion_one
        SAD = 60.0   # cm
        SDD = 100.0  # cm
        base_pixel_size = 0.05  # cm (0.5mm at isocenter)
    else
        error("Unknown scanner: $scanner. Supported: :aquilion_one")
    end

    # Determine pixel size
    if fov_cm === nothing
        pixel_size = base_pixel_size
    else
        pixel_size = (fov_cm * 1.1) / n_cols
    end

    if mode == :axial
        # Standard axial geometry
        return create_axial_geometry(
            SAD, SDD, n_angles, n_rows, n_cols, pixel_size
        )
    elseif mode == :helical
        # Helical geometry with table motion
        return create_helical_geometry_internal(
            SAD, SDD, n_angles, n_rows, n_cols, pixel_size,
            pitch, n_rotations, z_start
        )
    else
        error("Unknown scan mode: $mode. Use :axial or :helical")
    end
end

"""
    create_axial_geometry(SAD, SDD, n_angles, n_rows, n_cols, pixel_size)

Create standard axial (step-and-shoot) geometry.
"""
function create_axial_geometry(
    SAD::Float64, SDD::Float64,
    n_angles::Int, n_rows::Int, n_cols::Int, pixel_size::Float64
)
    angles = collect(range(0.0, 2π - 2π/n_angles, length=n_angles))

    source_positions = Matrix{Float64}(undef, 3, n_angles)
    detector_centers = Matrix{Float64}(undef, 3, n_angles)
    detector_u = Matrix{Float64}(undef, 3, n_angles)
    detector_v = Matrix{Float64}(undef, 3, n_angles)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = 0.0

        det_dist = SDD - SAD
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = 0.0

        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    # Default FOV for axial geometry (based on detector coverage)
    fov_xy = n_cols * pixel_size  # Approximate FOV from detector
    fov_z = n_rows * pixel_size
    fov = (fov_xy, fov_xy, fov_z)

    return CTGeometry(
        SAD, SDD, n_angles, n_rows, n_cols, pixel_size,
        angles, source_positions, detector_centers, detector_u, detector_v, fov
    )
end

"""
    create_helical_geometry_internal(SAD, SDD, angles_per_rot, n_rows, n_cols,
                                      pixel_size, pitch, n_rotations, z_start)

Create helical geometry with continuous table motion.

The Z-position of source and detector changes with each projection angle,
simulating table movement during acquisition.
"""
function create_helical_geometry_internal(
    SAD::Float64, SDD::Float64,
    angles_per_rotation::Int, n_rows::Int, n_cols::Int, pixel_size::Float64,
    pitch::Float64, n_rotations::Float64, z_start::Float64
)
    @assert pitch > 0 "Pitch must be positive"
    @assert n_rotations > 0 "Number of rotations must be positive"

    # Total number of projection angles
    total_angles = round(Int, angles_per_rotation * n_rotations)

    # Beam collimation (detector Z-coverage at isocenter)
    collimation = n_rows * pixel_size

    # Table movement per rotation
    table_per_rotation = pitch * collimation

    # Pre-allocate
    angles = Vector{Float64}(undef, total_angles)
    source_positions = Matrix{Float64}(undef, 3, total_angles)
    detector_centers = Matrix{Float64}(undef, 3, total_angles)
    detector_u = Matrix{Float64}(undef, 3, total_angles)
    detector_v = Matrix{Float64}(undef, 3, total_angles)

    for i in 1:total_angles
        # Gantry angle (wraps around each rotation)
        rotation_fraction = (i - 1) / angles_per_rotation
        θ = 2π * (rotation_fraction - floor(rotation_fraction))
        angles[i] = θ

        # Table Z position (increases linearly with angle)
        z_table = z_start + rotation_fraction * table_per_rotation

        cosθ = cos(θ)
        sinθ = sin(θ)

        # Source position with table Z
        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = z_table  # Table position

        # Detector center with table Z
        det_dist = SDD - SAD
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = z_table  # Table position

        # Detector axes (unchanged)
        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    # Default FOV for helical geometry (based on detector coverage and z range)
    fov_xy = n_cols * pixel_size  # Approximate FOV from detector
    z_range = abs(source_positions[3, end] - source_positions[3, 1])
    fov_z = z_range + n_rows * pixel_size  # Z coverage plus beam width
    fov = (fov_xy, fov_xy, fov_z)

    return CTGeometry(
        SAD, SDD, total_angles, n_rows, n_cols, pixel_size,
        angles, source_positions, detector_centers, detector_u, detector_v, fov
    )
end

# =============================================================================
# Helical Interpolation for Reconstruction
# =============================================================================

"""
    is_helical(geom::CTGeometry) -> Bool

Check if geometry represents a helical scan.

A helical scan is detected by checking if source Z-positions vary.
"""
function is_helical(geom::CTGeometry)
    if geom.n_angles < 2
        return false
    end
    # Check if Z positions vary (helical) vs constant (axial)
    z_first = geom.source_positions[3, 1]
    z_last = geom.source_positions[3, end]
    return abs(z_last - z_first) > 1e-6
end

"""
    get_helical_parameters(geom::CTGeometry) -> NamedTuple

Extract helical parameters from geometry.

Returns nothing if geometry is axial.
"""
function get_helical_parameters(geom::CTGeometry)
    if !is_helical(geom)
        return nothing
    end

    # Estimate parameters from geometry
    collimation = geom.n_rows * geom.pixel_size
    z_range = geom.source_positions[3, end] - geom.source_positions[3, 1]

    # Estimate angles per rotation (find first angle wrap)
    angles_per_rotation = geom.n_angles
    for i in 2:geom.n_angles
        if geom.angles[i] < geom.angles[i-1]
            angles_per_rotation = i - 1
            break
        end
    end

    n_rotations = geom.n_angles / angles_per_rotation
    table_per_rotation = z_range / (n_rotations - 1/angles_per_rotation)
    pitch = table_per_rotation / collimation

    return (
        pitch = pitch,
        n_rotations = n_rotations,
        collimation_cm = collimation,
        z_start = geom.source_positions[3, 1],
        z_end = geom.source_positions[3, end],
        table_travel_cm = z_range,
        angles_per_rotation = angles_per_rotation
    )
end

"""
    interpolate_helical_to_axial(sinogram, geom::CTGeometry, z_target::Float64) -> Array

Interpolate helical sinogram to pseudo-axial data at target Z position.

This is the key step for helical reconstruction: convert spiral data to
what the projections would look like if acquired axially at z_target.

# Arguments
- `sinogram`: Helical sinogram [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Helical geometry
- `z_target::Float64`: Target Z position for interpolation (cm)

# Returns
Interpolated sinogram [n_cols, n_rows, n_angles_per_rotation]

# Algorithm
Uses linear interpolation (180LI approach) between views at the same
gantry angle but different rotations (and thus different Z positions).

# Notes
- This enables using standard FDK on helical data
- Quality depends on pitch (lower pitch = better interpolation)
"""
function interpolate_helical_to_axial(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    z_target::Float64
) where T
    if !is_helical(geom)
        return sinogram  # Already axial
    end

    params = get_helical_parameters(geom)
    n_cols, n_rows, n_total = size(sinogram)
    n_per_rot = params.angles_per_rotation

    # Output: one rotation of pseudo-axial data
    result = zeros(T, n_cols, n_rows, n_per_rot)

    # Z positions for all views
    z_positions = [geom.source_positions[3, i] for i in 1:n_total]

    for angle_idx in 1:n_per_rot
        # Find all views at this gantry angle (across rotations)
        view_indices = angle_idx:n_per_rot:n_total

        if length(view_indices) < 2
            # Not enough data, just copy
            result[:, :, angle_idx] = sinogram[:, :, first(view_indices)]
            continue
        end

        # Z positions at this gantry angle
        z_at_angle = z_positions[view_indices]

        # Find bracketing views
        idx_below = findlast(z -> z <= z_target, z_at_angle)
        idx_above = findfirst(z -> z >= z_target, z_at_angle)

        if idx_below === nothing
            # z_target is below all data
            result[:, :, angle_idx] = sinogram[:, :, view_indices[1]]
        elseif idx_above === nothing
            # z_target is above all data
            result[:, :, angle_idx] = sinogram[:, :, view_indices[end]]
        elseif idx_below == idx_above
            # Exact match
            result[:, :, angle_idx] = sinogram[:, :, view_indices[idx_below]]
        else
            # Linear interpolation
            z_below = z_at_angle[idx_below]
            z_above = z_at_angle[idx_above]
            weight = (z_target - z_below) / (z_above - z_below)

            val_below = sinogram[:, :, view_indices[idx_below]]
            val_above = sinogram[:, :, view_indices[idx_above]]

            result[:, :, angle_idx] = T.((1 - weight) .* val_below .+ weight .* val_above)
        end
    end

    return result
end

"""
    reconstruct_helical_fdk(sinogram, geom::CTGeometry; z_positions=nothing) -> Array

Reconstruct helical scan using slice-by-slice FDK with interpolation.

# Arguments
- `sinogram`: Helical sinogram [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Helical geometry
- `z_positions`: Vector of Z positions to reconstruct (cm).
                 Default: evenly spaced within scan range.

# Returns
Stack of reconstructed slices [nx, ny, nz]

# Algorithm
1. For each target Z position:
   a. Interpolate helical data to pseudo-axial at that Z
   b. Run FDK reconstruction
   c. Extract central slice

# Notes
This is a simplified helical reconstruction. For production use,
consider more advanced algorithms (Katsevich, ASSR).
"""
function reconstruct_helical_fdk(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry;
    z_positions::Union{Vector{Float64},Nothing}=nothing,
    recon_size::Int=128
) where T
    params = get_helical_parameters(geom)
    if params === nothing
        error("Geometry is not helical. Use fdk_reconstruct for axial data.")
    end

    # Default Z positions: span the scan range
    if z_positions === nothing
        n_slices = round(Int, params.table_travel_cm / params.collimation_cm * 2)
        n_slices = max(n_slices, 1)
        z_positions = collect(range(
            params.z_start + params.collimation_cm/2,
            params.z_end - params.collimation_cm/2,
            length=n_slices
        ))
    end

    n_slices = length(z_positions)
    result = zeros(T, recon_size, recon_size, n_slices)

    # Create axial geometry for FDK (one rotation)
    axial_geom = create_axial_geometry(
        geom.SAD, geom.SDD,
        params.angles_per_rotation, geom.n_rows, geom.n_cols, geom.pixel_size
    )

    for (i, z) in enumerate(z_positions)
        # Interpolate to pseudo-axial at this Z
        pseudo_axial = interpolate_helical_to_axial(sinogram, geom, z)

        # Reconstruct using FDK
        vol = fdk_reconstruct(pseudo_axial, axial_geom; n_voxels=recon_size)

        # Extract central slice (or appropriate Z slice)
        central_slice = div(size(vol, 3), 2) + 1
        result[:, :, i] = vol[:, :, central_slice]
    end

    return result
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    estimate_helical_scan_time(geom::CTGeometry, rotation_time::Float64) -> Float64

Estimate total scan time for helical acquisition.

# Arguments
- `geom::CTGeometry`: Helical geometry
- `rotation_time::Float64`: Time per rotation in seconds

# Returns
Total scan time in seconds.
"""
function estimate_helical_scan_time(geom::CTGeometry, rotation_time::Float64)
    params = get_helical_parameters(geom)
    if params === nothing
        return rotation_time  # Axial
    end
    return params.n_rotations * rotation_time
end

"""
    compute_helical_z_coverage(geom::CTGeometry) -> Float64

Compute total Z-axis coverage of helical scan in cm.
"""
function compute_helical_z_coverage(geom::CTGeometry)
    params = get_helical_parameters(geom)
    if params === nothing
        return geom.n_rows * geom.pixel_size  # Axial: just collimation
    end

    # Coverage = table travel + collimation (beam width at start and end)
    return params.table_travel_cm + params.collimation_cm
end

# =============================================================================
# Exports
# =============================================================================

export ScanMode, AXIAL, HELICAL
export create_scan_geometry
export is_helical, get_helical_parameters
export interpolate_helical_to_axial, reconstruct_helical_fdk
export estimate_helical_scan_time, compute_helical_z_coverage
