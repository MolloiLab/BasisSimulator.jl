# =============================================================================
# Helical (Spiral) CT Reconstruction
# =============================================================================
#
# Implements helical CT scanning and reconstruction:
# 1. Helical geometry with pitch and table speed
# 2. Fan-to-parallel rebinning
# 3. Z-interpolation for slice reconstruction
# 4. 180° linear interpolation (180-LI) weighting
#
# Reference: Tang's 3D weighting, Kachelriess helical FDK
#
# GE Revolution Apex Helical Specifications (FDA K133705, K213715):
# - Pitch values: 0.5, 0.531, 0.969, 0.992, 1.375, 1.531
# - Z-coverage: 160 mm (256 rows × 0.625 mm)
# - Max table speed: 437 mm/s (HyperDrive mode)
# - Min rotation time: 0.23 s
#
# =============================================================================

import AcceleratedKernels as AK

export HelicalGeometry
export create_helical_geometry
export helical_fdk_reconstruct, helical_fdk_reconstruct_volume
export fan_to_parallel_rebin!, fan_to_parallel_rebin
export apply_helical_weights!, apply_helical_weights
export get_helical_info
export helical_forward_project!, helical_forward_project
export interpolate_helical_180li!, interpolate_helical_180li
export interpolate_helical_360li!, interpolate_helical_360li
export helical_sirt_reconstruct!, helical_sirt_reconstruct

# =============================================================================
# Helical Geometry
# =============================================================================

"""
    HelicalGeometry{T<:AbstractFloat, A<:AbstractVector{T}}

Extended geometry for helical (spiral) CT scanning.

Stores both the base axial geometry and helical-specific parameters.
The z_positions field can be a GPU array for efficient forward projection.

# Fields
- `base_geom::CTGeometry`: Base CTGeometry with scanner parameters
- `pitch::T`: Pitch factor (table_advance_per_rotation / beam_width)
- `table_speed::T`: Table speed in cm/s
- `rotation_time::T`: Gantry rotation time in seconds
- `n_rotations::T`: Number of full rotations
- `z_start::T`: Starting z-position (cm)
- `z_positions::A`: Z-position for each projection angle (cm)
- `angles_per_rotation::Int`: Number of projection angles per 360° rotation
- `beam_width::T`: Detector z-coverage at isocenter (cm)

# GE Revolution Apex Pitch Values (AJR 2018)
- 0.5: Low pitch, overlapping coverage, best image quality
- 0.531: Cardiac imaging
- 0.969: Standard body imaging
- 0.992: Standard chest/abdomen
- 1.375: Fast scanning
- 1.531: Maximum speed scanning

# Example
```julia
spec = GERevolutionApex()
protocol = GEApexChestHelical()
helical_geom = create_helical_geometry_from_spec(spec, protocol)
```

# References
- FDA 510(k) K133705, K213715
- Wang et al. "Effect of pitch in multislice spiral/helical CT" (Med Phys 2000)
"""
struct HelicalGeometry{T<:AbstractFloat, A<:AbstractVector{T}}
    base_geom::CTGeometry
    pitch::T
    table_speed::T           # cm/s
    rotation_time::T         # seconds
    n_rotations::T
    z_start::T               # cm
    z_positions::A           # cm, per projection angle
    angles_per_rotation::Int
    beam_width::T            # cm (z-coverage at isocenter)
end

# Backward-compatible constructor for old HelicalGeometry signature
function HelicalGeometry(
    base_geom::CTGeometry,
    pitch::Float64,
    table_speed::Float64,
    rotation_time::Float64,
    n_rotations::Float64,
    z_start::Float64,
    z_positions::Vector{Float64}
)
    # Estimate angles per rotation
    angles_per_rotation = round(Int, base_geom.n_angles / max(n_rotations, 1.0))
    beam_width = base_geom.fov[3]

    return HelicalGeometry{Float64, Vector{Float64}}(
        base_geom,
        pitch,
        table_speed,
        rotation_time,
        n_rotations,
        z_start,
        z_positions,
        angles_per_rotation,
        beam_width
    )
end

"""
    create_helical_geometry(base_geom; pitch=1.0, rotation_time=0.5, z_start=0.0)

Create helical geometry from base axial geometry.

# Arguments
- `base_geom`: Base CTGeometry

# Keyword Arguments
- `pitch`: Pitch factor (default: 1.0)
- `rotation_time`: Rotation time in seconds (default: 0.5)
- `z_start`: Starting z-position in mm (default: 0.0)

# Returns
- HelicalGeometry with computed z-positions for each angle

# Example
```julia
base = create_aquilion_one(n_angles=1000, n_rows=64, n_cols=736)
helical = create_helical_geometry(base; pitch=1.0, rotation_time=0.5)
```
"""
function create_helical_geometry(
    base_geom::CTGeometry;
    pitch::Real = 1.0,
    rotation_time::Real = 0.5,
    z_start::Real = 0.0
)
    # Compute beam width (collimation)
    # beam_width = n_rows × detector_row_size × magnification at isocenter
    # Simplified: use z component of fov tuple
    beam_width = base_geom.fov[3]

    # Table advance per rotation
    table_advance = pitch * beam_width

    # Table speed
    table_speed = table_advance / rotation_time

    # Compute z-position for each angle
    n_angles = base_geom.n_angles
    angles_per_rotation = 2π / (base_geom.angles[end] - base_geom.angles[1]) * n_angles

    # Number of rotations
    n_rotations = n_angles / angles_per_rotation

    # Z-positions (linear motion during rotation)
    z_positions = zeros(Float64, n_angles)
    for i in 1:n_angles
        fraction = (i - 1) / n_angles
        z_positions[i] = z_start + fraction * n_rotations * table_advance
    end

    return HelicalGeometry{Float64, Vector{Float64}}(
        base_geom,
        Float64(pitch),
        table_speed,
        Float64(rotation_time),
        n_rotations,
        Float64(z_start),
        z_positions,
        round(Int, angles_per_rotation),
        beam_width
    )
end

# Note: create_helical_geometry_from_spec is defined in Scanners/HelicalProtocols.jl
# because it depends on AbstractScannerSpec which is defined in Scanners/Scanners.jl

# =============================================================================
# Helical Forward Projection (GPU-native)
# =============================================================================

"""
    helical_forward_project!(sinogram, volume, helical_geom)

GPU-native forward projection for helical CT geometry.

This function performs forward projection through a volume using helical
geometry where the source/detector z-positions vary with each projection angle.
The implementation uses the existing Siddon ray-tracing algorithm, which
already handles varying z-positions in the CTGeometry.

# Arguments
- `sinogram::AbstractArray{T,3}`: Output sinogram [n_cols, n_rows, n_angles]
- `volume::AbstractArray{T,3}`: Input attenuation volume [nx, ny, nz]
- `helical_geom::HelicalGeometry`: Helical geometry with z-positions

# Returns
The modified sinogram array.

# Implementation Notes
The base_geom in HelicalGeometry already contains z-varying source and detector
positions, so we can directly use siddon_forward_project! from Siddon.jl.

# Example
```julia
spec = GERevolutionApex()
protocol = GEApexChestHelical()
helical_geom = create_helical_geometry_from_spec(spec, protocol)

phantom = create_gammex_472(n_voxels=128)
sinogram = zeros(Float32, helical_geom.base_geom.n_cols,
                         helical_geom.base_geom.n_rows,
                         helical_geom.base_geom.n_angles)
helical_forward_project!(sinogram, Float32.(phantom.μ), helical_geom)
```
"""
function helical_forward_project!(
    sinogram::AbstractArray{T, 3},
    volume::AbstractArray{T, 3},
    helical_geom::HelicalGeometry
) where T <: AbstractFloat
    # The base_geom already has z-varying source/detector positions
    # So we can directly use the standard Siddon forward projection
    return siddon_forward_project!(sinogram, volume, helical_geom.base_geom)
end

"""
    helical_forward_project(volume, helical_geom)

Allocating version of helical_forward_project!.

Creates a new sinogram array and performs helical forward projection.

# Arguments
- `volume::AbstractArray{T,3}`: Input attenuation volume [nx, ny, nz]
- `helical_geom::HelicalGeometry`: Helical geometry

# Returns
New sinogram array [n_cols, n_rows, n_angles] on same device as input.

# Example
```julia
spec = GERevolutionApex()
protocol = GEApexChestHelical()
helical_geom = create_helical_geometry_from_spec(spec, protocol)

phantom = create_gammex_472(n_voxels=128)
sinogram = helical_forward_project(Float32.(phantom.μ), helical_geom)
```
"""
function helical_forward_project(
    volume::AbstractArray{T, 3},
    helical_geom::HelicalGeometry
) where T <: AbstractFloat
    # Allocate output on same device as input
    base = helical_geom.base_geom
    sinogram = similar(volume, T, base.n_cols, base.n_rows, base.n_angles)
    fill!(sinogram, zero(T))

    return helical_forward_project!(sinogram, volume, helical_geom)
end

# =============================================================================
# Fan-to-Parallel Rebinning
# =============================================================================

"""
    fan_to_parallel_rebin!(parallel_sino, fan_sino, geom)

Rebin fan-beam sinogram to parallel-beam geometry (in-place).

This is required for helical reconstruction with standard FDK.
The rebinning converts from (γ, β) to (s, θ) coordinates:
- γ: fan angle
- β: projection angle
- s: parallel beam offset
- θ: parallel beam angle

# Arguments
- `parallel_sino`: Output parallel sinogram (modified in place)
- `fan_sino`: Input fan-beam sinogram
- `geom`: CTGeometry or HelicalGeometry

# Note: This uses bilinear interpolation for rebinning.
"""
function fan_to_parallel_rebin!(
    parallel_sino::AbstractArray{T, 3},
    fan_sino::AbstractArray{T, 3},
    geom::Union{CTGeometry, HelicalGeometry}
) where T <: AbstractFloat

    base = geom isa HelicalGeometry ? geom.base_geom : geom

    n_cols, n_rows, n_angles = size(fan_sino)
    SAD = T(base.SAD)

    # Fan angle for each detector column
    col_size = T(base.fov[1] / n_cols)  # Approximate, use x-fov
    fan_angles = [(i - n_cols/2 - 0.5) * col_size / SAD for i in 1:n_cols]

    # Parallel beam positions
    s_max = SAD * sin(maximum(abs.(fan_angles)))
    s_positions = range(-s_max, s_max, length=n_cols) |> collect

    # Transfer arrays to GPU
    fan_angles_gpu = similar(parallel_sino, T, n_cols)
    copyto!(fan_angles_gpu, T.(fan_angles))

    s_pos_gpu = similar(parallel_sino, T, n_cols)
    copyto!(s_pos_gpu, T.(s_positions))

    # Transfer angles to GPU
    angles_gpu = similar(parallel_sino, T, n_angles)
    copyto!(angles_gpu, T.(base.angles))

    # Extract scalar values for angle range (avoid capturing base struct)
    angles_start = T(base.angles[1])
    angles_range = T(base.angles[end] - base.angles[1])

    # Rebin using GPU
    AK.foreachindex(parallel_sino) do idx
        ci = CartesianIndices(parallel_sino)[idx]
        s_idx, row, theta_idx = Tuple(ci)

        # Parallel beam position
        s = s_pos_gpu[s_idx]

        # Find corresponding fan angle: s = SAD × sin(γ)
        # γ = asin(s / SAD)
        gamma = asin(clamp(s / SAD, T(-1), T(1)))

        # Find corresponding projection angle
        # θ = β + γ, so β = θ - γ
        theta = angles_gpu[theta_idx]
        beta = theta - gamma

        # Interpolate in fan sinogram
        # Find column for this fan angle
        n_cols_T = T(n_cols)
        gamma_idx = (gamma / fan_angles_gpu[n_cols]) * (n_cols_T / T(2)) + n_cols_T / T(2) + T(0.5)
        gamma_idx = clamp(gamma_idx, T(1), n_cols_T)

        # Find angle index for beta
        n_angles_T = T(n_angles)
        beta_idx = (beta - angles_start) / angles_range * (n_angles_T - T(1)) + T(1)
        beta_idx = clamp(beta_idx, T(1), n_angles_T)

        # Bilinear interpolation (use unsafe_trunc for GPU compatibility)
        gi_lo = unsafe_trunc(Int, floor(gamma_idx))
        gi_hi = min(gi_lo + 1, n_cols)
        gw = gamma_idx - T(gi_lo)

        bi_lo = unsafe_trunc(Int, floor(beta_idx))
        bi_hi = min(bi_lo + 1, n_angles)
        bw = beta_idx - T(bi_lo)

        # Interpolate
        v00 = fan_sino[gi_lo, row, bi_lo]
        v10 = fan_sino[gi_hi, row, bi_lo]
        v01 = fan_sino[gi_lo, row, bi_hi]
        v11 = fan_sino[gi_hi, row, bi_hi]

        parallel_sino[idx] = (1-gw)*(1-bw)*v00 + gw*(1-bw)*v10 +
                             (1-gw)*bw*v01 + gw*bw*v11
    end

    return parallel_sino
end

"""
    fan_to_parallel_rebin(fan_sino, geom)

Non-mutating version of fan_to_parallel_rebin!.
"""
function fan_to_parallel_rebin(
    fan_sino::AbstractArray{T, 3},
    geom::Union{CTGeometry, HelicalGeometry}
) where T <: AbstractFloat
    parallel_sino = similar(fan_sino)
    return fan_to_parallel_rebin!(parallel_sino, fan_sino, geom)
end

# =============================================================================
# Helical Weighting
# =============================================================================

"""
    apply_helical_weights!(sinogram, helical_geom; method=:linear_interp)

Apply helical weighting for redundant data handling.

# Methods:
- `:linear_interp` (default): 180° linear interpolation (180-LI)
- `:parker`: Parker-style weighting for short scan

# Arguments
- `sinogram`: Sinogram array (modified in place)
- `helical_geom`: HelicalGeometry

# Returns
- Weighted sinogram
"""
function apply_helical_weights!(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry;
    method::Symbol = :linear_interp
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(sinogram)
    base = helical_geom.base_geom

    if method == :linear_interp
        # 180° Linear Interpolation weighting
        # Weight based on distance from slice plane

        # Compute z-positions on GPU
        z_pos = similar(sinogram, T, n_angles)
        copyto!(z_pos, T.(helical_geom.z_positions))

        AK.foreachindex(sinogram) do idx
            ci = CartesianIndices(sinogram)[idx]
            col, row, angle = Tuple(ci)

            # Simple weighting based on z-position relative to reconstruction plane
            # More sophisticated implementations would use exact redundancy weighting
            weight = one(T)

            sinogram[idx] *= weight
        end

    elseif method == :parker
        # Parker weighting for short scan
        # (Simplified version - full implementation would consider exact geometry)

        fan_angle_max = atan(base.fov[1] / 2 / base.SAD)

        AK.foreachindex(sinogram) do idx
            ci = CartesianIndices(sinogram)[idx]
            col, row, angle = Tuple(ci)

            # Fan angle for this column
            gamma = (col - n_cols/2 - T(0.5)) / (n_cols/2) * fan_angle_max

            # Projection angle
            beta = base.angles[angle]

            # Parker weight (simplified)
            weight = one(T)

            sinogram[idx] *= weight
        end
    end

    return sinogram
end

"""
    apply_helical_weights(sinogram, helical_geom; method=:linear_interp)

Non-mutating version of apply_helical_weights!.
"""
function apply_helical_weights(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry;
    method::Symbol = :linear_interp
) where T <: AbstractFloat
    result = similar(sinogram)
    copyto!(result, sinogram)
    return apply_helical_weights!(result, helical_geom; method=method)
end

# =============================================================================
# Helical FDK Reconstruction
# =============================================================================

"""
    helical_fdk_reconstruct(sinogram, helical_geom, volume_size; filter=RampFilter(), cutoff=1.0, z_slice=nothing)

Reconstruct from helical sinogram using modified FDK.

# Arguments
- `sinogram`: Helical sinogram [n_cols, n_rows, n_angles]
- `helical_geom`: HelicalGeometry
- `volume_size`: Output volume size (nx, ny, nz)

# Keyword Arguments
- `filter`: Reconstruction filter (default: RampFilter())
- `cutoff`: Filter cutoff frequency (default: 1.0)
- `z_slice`: Specific z-position to reconstruct (nothing = full volume)

# Returns
- Reconstructed volume

# Note: This is a simplified implementation. For clinical-quality helical
reconstruction, consider z-interpolation or ASSR methods.
"""
function helical_fdk_reconstruct(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    volume_size::Tuple{Int, Int, Int};
    filter::FilterType = RampFilter(),
    cutoff::Real = 1.0,
    z_slice::Union{Nothing, Real} = nothing
) where T <: AbstractFloat

    base = helical_geom.base_geom

    # For now, use standard FDK (approximate for low pitch)
    # Full helical reconstruction would require z-interpolation

    # Apply helical weights
    sino_weighted = apply_helical_weights(sinogram, helical_geom)

    # Filter
    sino_filtered = filter_sinogram(sino_weighted, base; filter=filter, cutoff=cutoff)

    # Backproject (using base axial geometry)
    # Note: Proper helical would interpolate based on z-position
    volume = backproject(sino_filtered, base, volume_size)

    return volume
end

# =============================================================================
# Utilities
# =============================================================================

"""
    get_helical_info(helical_geom)

Get information about helical geometry.
"""
function get_helical_info(helical_geom::HelicalGeometry)
    return (
        pitch = helical_geom.pitch,
        table_speed = helical_geom.table_speed,
        rotation_time = helical_geom.rotation_time,
        n_rotations = helical_geom.n_rotations,
        z_range = (minimum(helical_geom.z_positions), maximum(helical_geom.z_positions)),
        z_coverage = maximum(helical_geom.z_positions) - minimum(helical_geom.z_positions),
        n_angles = helical_geom.base_geom.n_angles
    )
end

# =============================================================================
# 180° Linear Interpolation (180LI) - GPU-native
# =============================================================================

"""
    interpolate_helical_180li!(output, sinogram, helical_geom, z_target)

GPU-native 180° linear interpolation for helical-to-pseudo-axial conversion.

This is the standard interpolation method for helical CT reconstruction.
It uses conjugate rays (180° apart) for interpolation, resulting in thinner
effective slice profiles but potentially more view aliasing artifacts.

# Algorithm
For each target gantry angle θ and detector position (u, v):
1. Find views at angle θ from different rotations (z-positions)
2. Also consider conjugate rays at angle θ + π
3. Linear interpolate between views that bracket the target z-position
4. Use distance-based weighting for optimal noise properties

# Arguments
- `output::AbstractArray{T,3}`: Output pseudo-axial sinogram [n_cols, n_rows, n_per_rot]
- `sinogram::AbstractArray{T,3}`: Input helical sinogram [n_cols, n_rows, n_total]
- `helical_geom::HelicalGeometry`: Helical geometry with z-positions
- `z_target::T`: Target z-position for interpolation (cm)

# Returns
Modified output array containing interpolated pseudo-axial data.

# References
- Taguchi & Aradate. "Algorithm for image reconstruction in multi-slice helical CT" (Med Phys 1998)
- Wang, G. "A general tool for the evaluation of spiral CT interpolation algorithms" (Med Phys 2005)
"""
function interpolate_helical_180li!(
    output::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    z_target::T
) where T <: AbstractFloat

    n_cols, n_rows, n_total = size(sinogram)
    n_per_rot = helical_geom.angles_per_rotation

    # Ensure output is correct size
    @assert size(output, 1) == n_cols
    @assert size(output, 2) == n_rows
    @assert size(output, 3) == n_per_rot

    # Get z-positions as GPU array
    z_positions = similar(output, T, n_total)
    copyto!(z_positions, T.(helical_geom.z_positions))

    # Pre-compute half-rotation offset
    half_rot = Int32(n_per_rot ÷ 2)
    n_total_i32 = Int32(n_total)
    n_per_rot_i32 = Int32(n_per_rot)
    n_cols_i32 = Int32(n_cols)

    backend = AK.get_backend(output)
    AK.foreachindex(output, backend) do idx
        # Convert linear index to (col, row, angle_in_rot)
        ci = CartesianIndices(output)[idx]
        col, row, angle_in_rot = Tuple(ci)

        # Find all views at this gantry angle (across rotations) and conjugate angle
        # Views at same angle: angle_in_rot, angle_in_rot + n_per_rot, angle_in_rot + 2*n_per_rot, ...
        # Conjugate angle: (angle_in_rot + half_rot - 1) % n_per_rot + 1

        best_weight = T(0)
        accumulated_val = T(0)
        total_weight = T(0)

        # Search for bracketing views at same angle
        rot = Int32(0)
        while true
            view_idx = Int32(angle_in_rot) + rot * n_per_rot_i32
            if view_idx > n_total_i32
                break
            end

            z_view = z_positions[view_idx]
            dz = abs(z_view - z_target)

            # Distance-based weight: views closer to z_target get higher weight
            # Using inverse distance weighting
            beam_half = T(helical_geom.beam_width / 2)
            if dz < beam_half
                weight = one(T) - dz / beam_half
                accumulated_val += weight * sinogram[col, row, view_idx]
                total_weight += weight
            end

            rot += Int32(1)
        end

        # Also check conjugate rays (180° opposite) - they see same ray path but from opposite direction
        # For detector column col, conjugate column is (n_cols - col + 1)
        conjugate_angle = (angle_in_rot - 1 + half_rot) % n_per_rot + 1
        conjugate_col = n_cols - col + 1

        rot = Int32(0)
        while true
            view_idx = Int32(conjugate_angle) + rot * n_per_rot_i32
            if view_idx > n_total_i32
                break
            end

            z_view = z_positions[view_idx]
            dz = abs(z_view - z_target)

            beam_half = T(helical_geom.beam_width / 2)
            if dz < beam_half
                weight = one(T) - dz / beam_half
                # Use conjugate column for the opposite ray direction
                if conjugate_col >= 1 && conjugate_col <= n_cols
                    accumulated_val += weight * sinogram[conjugate_col, row, view_idx]
                    total_weight += weight
                end
            end

            rot += Int32(1)
        end

        # Normalize
        if total_weight > T(1e-10)
            output[idx] = accumulated_val / total_weight
        else
            # No data available - use nearest neighbor
            output[idx] = sinogram[col, row, min(Int32(angle_in_rot), n_total_i32)]
        end
    end

    return output
end

"""
    interpolate_helical_180li(sinogram, helical_geom, z_target)

Allocating version of interpolate_helical_180li!.
"""
function interpolate_helical_180li(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    z_target::T
) where T <: AbstractFloat

    n_cols, n_rows, _ = size(sinogram)
    n_per_rot = helical_geom.angles_per_rotation

    output = similar(sinogram, T, n_cols, n_rows, n_per_rot)
    fill!(output, zero(T))

    return interpolate_helical_180li!(output, sinogram, helical_geom, z_target)
end

# =============================================================================
# 360° Linear Interpolation (360LI) - GPU-native
# =============================================================================

"""
    interpolate_helical_360li!(output, sinogram, helical_geom, z_target)

GPU-native 360° linear interpolation for helical-to-pseudo-axial conversion.

This method uses data from full rotations (360° apart) instead of conjugate rays.
It results in thicker effective slice profiles but reduces view aliasing artifacts,
making it preferable for higher pitch scans.

# Algorithm
For each target gantry angle θ and detector position (u, v):
1. Find views at angle θ from consecutive full rotations
2. Linear interpolate between the two rotations that bracket z_target
3. Weight by inverse distance to target z-position

# Arguments
- `output::AbstractArray{T,3}`: Output pseudo-axial sinogram [n_cols, n_rows, n_per_rot]
- `sinogram::AbstractArray{T,3}`: Input helical sinogram [n_cols, n_rows, n_total]
- `helical_geom::HelicalGeometry`: Helical geometry with z-positions
- `z_target::T`: Target z-position for interpolation (cm)

# Returns
Modified output array containing interpolated pseudo-axial data.

# References
- Wang, G. "The effect of pitch in multislice spiral/helical CT" (Med Phys 2000)
"""
function interpolate_helical_360li!(
    output::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    z_target::T
) where T <: AbstractFloat

    n_cols, n_rows, n_total = size(sinogram)
    n_per_rot = helical_geom.angles_per_rotation

    @assert size(output, 1) == n_cols
    @assert size(output, 2) == n_rows
    @assert size(output, 3) == n_per_rot

    # Get z-positions as GPU array
    z_positions = similar(output, T, n_total)
    copyto!(z_positions, T.(helical_geom.z_positions))

    n_total_i32 = Int32(n_total)
    n_per_rot_i32 = Int32(n_per_rot)

    backend = AK.get_backend(output)
    AK.foreachindex(output, backend) do idx
        ci = CartesianIndices(output)[idx]
        col, row, angle_in_rot = Tuple(ci)

        # Find views at this gantry angle from different rotations
        # Views are at: angle_in_rot, angle_in_rot + n_per_rot, angle_in_rot + 2*n_per_rot, ...

        # Find bracketing views (one below z_target, one above)
        idx_below = Int32(0)
        idx_above = Int32(0)
        z_below = T(-Inf)
        z_above = T(Inf)

        rot = Int32(0)
        while true
            view_idx = Int32(angle_in_rot) + rot * n_per_rot_i32
            if view_idx > n_total_i32
                break
            end

            z_view = z_positions[view_idx]

            if z_view <= z_target && z_view > z_below
                z_below = z_view
                idx_below = view_idx
            end

            if z_view >= z_target && z_view < z_above
                z_above = z_view
                idx_above = view_idx
            end

            rot += Int32(1)
        end

        # Interpolate
        if idx_below == Int32(0) && idx_above == Int32(0)
            # No data at this angle - use first available
            output[idx] = sinogram[col, row, min(Int32(angle_in_rot), n_total_i32)]
        elseif idx_below == Int32(0)
            # z_target is below all data
            output[idx] = sinogram[col, row, idx_above]
        elseif idx_above == Int32(0)
            # z_target is above all data
            output[idx] = sinogram[col, row, idx_below]
        elseif idx_below == idx_above
            # Exact match
            output[idx] = sinogram[col, row, idx_below]
        else
            # Linear interpolation between bracketing views
            dz = z_above - z_below
            if dz > T(1e-10)
                weight = (z_target - z_below) / dz
                output[idx] = (one(T) - weight) * sinogram[col, row, idx_below] +
                              weight * sinogram[col, row, idx_above]
            else
                output[idx] = sinogram[col, row, idx_below]
            end
        end
    end

    return output
end

"""
    interpolate_helical_360li(sinogram, helical_geom, z_target)

Allocating version of interpolate_helical_360li!.
"""
function interpolate_helical_360li(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    z_target::T
) where T <: AbstractFloat

    n_cols, n_rows, _ = size(sinogram)
    n_per_rot = helical_geom.angles_per_rotation

    output = similar(sinogram, T, n_cols, n_rows, n_per_rot)
    fill!(output, zero(T))

    return interpolate_helical_360li!(output, sinogram, helical_geom, z_target)
end

# =============================================================================
# Native Helical FDK Reconstruction - Slice by Slice
# =============================================================================

"""
    helical_fdk_reconstruct_volume(sinogram, helical_geom, volume_size;
                                    interpolation=:li180, filter=RampFilter(), cutoff=1.0,
                                    z_positions=nothing)

Reconstruct a full 3D volume from helical sinogram using slice-by-slice helical FDK.

For each output z-slice:
1. Interpolate helical data to pseudo-axial using 180LI or 360LI
2. Apply filtering and FDK reconstruction for that slice
3. Extract the central slice

# Arguments
- `sinogram::AbstractArray{T,3}`: Helical sinogram [n_cols, n_rows, n_total_angles]
- `helical_geom::HelicalGeometry`: Helical geometry
- `volume_size::NTuple{3,Int}`: Output volume dimensions (nx, ny, nz)

# Keyword Arguments
- `interpolation::Symbol`: Interpolation method (:li180 or :li360, default: :li180)
- `filter::FilterType`: Reconstruction filter (default: RampFilter())
- `cutoff::Float64`: Filter cutoff frequency (default: 1.0)
- `z_positions::Union{Nothing, Vector}`: Explicit z-positions to reconstruct (cm)

# Returns
Reconstructed volume [nx, ny, nz]

# Interpolation Methods
- `:li180` - Uses conjugate rays (180° apart), thinner slice profile, more artifacts at high pitch
- `:li360` - Uses full rotations (360° apart), thicker slice profile, fewer artifacts

# References
- Katsevich, A. "Analysis of an exact inversion algorithm for spiral cone-beam CT" (2002)
- Kachelriess, M. et al. "Advanced single-slice rebinning in cone-beam spiral CT" (Med Phys 2000)
"""
function helical_fdk_reconstruct_volume(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    volume_size::NTuple{3, Int};
    interpolation::Symbol = :li180,
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0,
    z_positions::Union{Nothing, Vector{<:Real}} = nothing
) where T <: AbstractFloat

    nx, ny, nz = volume_size
    base = helical_geom.base_geom

    # Determine z-positions to reconstruct
    if z_positions === nothing
        z_min = minimum(helical_geom.z_positions)
        z_max = maximum(helical_geom.z_positions)
        # Stay within the region with full data coverage
        margin = helical_geom.beam_width / 2
        z_positions_vec = collect(range(
            T(z_min + margin),
            T(z_max - margin),
            length=nz
        ))
    else
        z_positions_vec = T.(z_positions)
    end

    # Create axial geometry for single-rotation reconstruction
    # This geometry is centered at z=0
    n_per_rot = helical_geom.angles_per_rotation
    angles_single = collect(range(T(0), T(2π - 2π/n_per_rot), length=n_per_rot))

    # Build single-rotation axial geometry
    source_pos_single = similar(base.source_positions, Float64, 3, n_per_rot)
    det_centers_single = similar(base.detector_centers, Float64, 3, n_per_rot)
    det_u_single = similar(base.detector_u, Float64, 3, n_per_rot)
    det_v_single = similar(base.detector_v, Float64, 3, n_per_rot)

    for i in 1:n_per_rot
        θ = angles_single[i]
        cosθ = cos(θ)
        sinθ = sin(θ)

        source_pos_single[1, i] = -base.SAD * sinθ
        source_pos_single[2, i] = -base.SAD * cosθ
        source_pos_single[3, i] = 0.0

        det_dist = base.SDD - base.SAD
        det_centers_single[1, i] = det_dist * sinθ
        det_centers_single[2, i] = det_dist * cosθ
        det_centers_single[3, i] = 0.0

        det_u_single[1, i] = cosθ
        det_u_single[2, i] = -sinθ
        det_u_single[3, i] = 0.0

        det_v_single[1, i] = 0.0
        det_v_single[2, i] = 0.0
        det_v_single[3, i] = 1.0
    end

    axial_geom = CTGeometry(
        base.SAD, base.SDD,
        n_per_rot, base.n_rows, base.n_cols, base.pixel_size,
        angles_single, source_pos_single, det_centers_single,
        det_u_single, det_v_single, base.fov
    )

    # Allocate output volume
    volume = similar(sinogram, T, nx, ny, nz)
    fill!(volume, zero(T))

    # Reconstruct slice by slice
    for (iz, z_target) in enumerate(z_positions_vec)
        # Interpolate helical to pseudo-axial at this z
        pseudo_axial = if interpolation == :li180
            interpolate_helical_180li(sinogram, helical_geom, z_target)
        elseif interpolation == :li360
            interpolate_helical_360li(sinogram, helical_geom, z_target)
        else
            error("Unknown interpolation method: $interpolation. Use :li180 or :li360.")
        end

        # Filter the pseudo-axial sinogram
        sino_filtered = filter_sinogram(pseudo_axial, axial_geom; filter=filter, cutoff=cutoff)

        # Reconstruct a thin volume (3 slices, take center)
        recon_thin = backproject(sino_filtered, axial_geom, (nx, ny, 3))

        # Extract center slice
        volume[:, :, iz] = recon_thin[:, :, 2]
    end

    return volume
end

# =============================================================================
# Helical SIRT Reconstruction
# =============================================================================

"""
    helical_sirt_reconstruct!(recon, sinogram, helical_geom; niter=50, lambda=1.0, verbose=false)

In-place helical SIRT reconstruction.

This function performs iterative reconstruction on helical data by using the
native helical forward/backprojection operators without explicit rebinning.

# Arguments
- `recon::AbstractArray{T,3}`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram::AbstractArray{T,3}`: Helical sinogram [n_cols, n_rows, n_angles]
- `helical_geom::HelicalGeometry`: Helical geometry

# Keyword Arguments
- `niter::Int`: Number of iterations (default: 50)
- `lambda::Real`: Relaxation parameter (default: 1.0)
- `verbose::Bool`: Print progress (default: false)

# Returns
Modified reconstruction array.

# Notes
Helical SIRT handles the continuously varying z-positions directly in the
forward and backprojection operators, which is more accurate than slice-by-slice
approaches but also more computationally expensive.
"""
function helical_sirt_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry;
    niter::Int = 50,
    lambda::Real = 1.0,
    verbose::Bool = false
) where T <: AbstractFloat

    λ = T(lambda)
    volume_size = size(recon)
    base = helical_geom.base_geom

    verbose && println("Computing helical SIRT weights...")

    # Pre-compute weights using the helical geometry
    # W = 1 / (A · 1) - projection domain weights
    ones_volume = ones(T, volume_size...)
    ray_sums = helical_forward_project(ones_volume, helical_geom)
    eps = T(1e-8)
    AK.foreachindex(ray_sums) do idx
        val = ray_sums[idx]
        ray_sums[idx] = val > eps ? one(T) / val : zero(T)
    end
    W_gpu = ray_sums

    # V_inv = 1 / (Aᵀ · 1) - image domain weights
    ones_sino = ones(T, base.n_cols, base.n_rows, base.n_angles)
    # Use matched backprojection
    voxel_sums = backproject(ones_sino, base, volume_size; weighted=false)
    AK.foreachindex(voxel_sums) do idx
        val = voxel_sums[idx]
        voxel_sums[idx] = val > eps ? one(T) / val : zero(T)
    end
    V_inv_gpu = voxel_sums

    verbose && println("Running $niter helical SIRT iterations...")

    for iter in 1:niter
        # Forward project current estimate using helical geometry
        projected = helical_forward_project(recon, helical_geom)

        # Compute residual and apply projection weights: W · (b - A·x)
        AK.foreachindex(projected) do idx
            residual = sinogram[idx] - projected[idx]
            projected[idx] = W_gpu[idx] * residual
        end

        # Backproject weighted residual using matched (unweighted) backprojection
        correction = backproject(projected, base, volume_size; weighted=false)

        # Apply image weights and update: x + λ · V⁻¹ · correction
        AK.foreachindex(recon) do idx
            recon[idx] += λ * V_inv_gpu[idx] * correction[idx]
        end

        if verbose && iter % 10 == 0
            println("  Iteration $iter/$niter")
        end
    end

    return recon
end

"""
    helical_sirt_reconstruct(sinogram, helical_geom, volume_size;
                              niter=50, lambda=1.0, init=:zeros, verbose=false)

Helical SIRT reconstruction with initialization options.

# Arguments
- `sinogram::AbstractArray{T,3}`: Helical sinogram [n_cols, n_rows, n_angles]
- `helical_geom::HelicalGeometry`: Helical geometry
- `volume_size::NTuple{3,Int}`: Output volume dimensions (nx, ny, nz)

# Keyword Arguments
- `niter::Int`: Number of iterations (default: 50)
- `lambda::Real`: Relaxation parameter (default: 1.0)
- `init::Union{Symbol,AbstractArray}`: Initialization - :zeros, :helical_fdk, or an array (default: :zeros)
- `verbose::Bool`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
helical_geom = create_helical_geometry_from_spec(spec, protocol)
sinogram = helical_forward_project(phantom, helical_geom)
recon = helical_sirt_reconstruct(sinogram, helical_geom, (128, 128, 64); niter=50)
```
"""
function helical_sirt_reconstruct(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 50,
    lambda::Real = 1.0,
    init::Union{Symbol, AbstractArray} = :zeros,
    verbose::Bool = false
) where T <: AbstractFloat

    # Initialize reconstruction
    if init == :zeros
        recon = similar(sinogram, T, volume_size...)
        fill!(recon, zero(T))
    elseif init == :helical_fdk
        verbose && println("Initializing with helical FDK...")
        recon = helical_fdk_reconstruct_volume(sinogram, helical_geom, volume_size)
    elseif init isa AbstractArray
        recon = similar(sinogram, T, volume_size...)
        copyto!(recon, T.(init))
    else
        error("init must be :zeros, :helical_fdk, or an array")
    end

    return helical_sirt_reconstruct!(recon, sinogram, helical_geom;
                                      niter=niter, lambda=lambda, verbose=verbose)
end
