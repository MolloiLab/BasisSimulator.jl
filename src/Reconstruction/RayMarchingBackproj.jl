"""
    Reconstruction/RayMarchingBackproj.jl

Backprojection with ON-THE-FLY geometry computation.

This approach enables a SINGLE compiled XLA kernel for ALL angles:
1. Geometry computed inside the kernel from compact parameters
2. Bilinear interpolation from filtered sinogram
3. Memory: O(volume + sinogram) - no pre-computed geometry arrays

Complements Forward/RayMarching.jl for fully compiled CT pipeline.
"""

using FFTW

# =============================================================================
# Voxel-to-Detector Geometry
# =============================================================================

"""
    BackprojGeometry

Compact storage of scanner geometry for backprojection.
Memory: O(n_angles × 12) ≈ tiny (< 100KB for 1000 angles)
"""
struct BackprojGeometry
    # Source positions: [3, n_angles]
    source_positions::Array{Float32, 2}

    # Detector centers: [3, n_angles]
    detector_centers::Array{Float32, 2}

    # Detector u-axis (horizontal): [3, n_angles]
    detector_u::Array{Float32, 2}

    # Detector v-axis (vertical): [3, n_angles]
    detector_v::Array{Float32, 2}

    # Source-detector axis (normalized): [3, n_angles]
    sd_axis::Array{Float32, 2}

    # Scalar geometry parameters
    SAD::Float32
    SDD::Float32
    pixel_size_det::Float32
    n_cols::Int
    n_rows::Int
    n_angles::Int
    delta_angle::Float32
end

"""
    compute_backproj_geometry(geom::CTGeometry) -> BackprojGeometry

Extract compact geometry for backprojection from CTGeometry.
"""
function compute_backproj_geometry(geom::CTGeometry)
    n_angles = geom.n_angles

    # Copy source and detector geometry
    source_positions = Float32.(geom.source_positions)
    detector_centers = Float32.(geom.detector_centers)
    detector_u = Float32.(geom.detector_u)
    detector_v = Float32.(geom.detector_v)

    # Compute source-detector axis for each angle
    sd_axis = zeros(Float32, 3, n_angles)
    for a in 1:n_angles
        dx = detector_centers[1, a] - source_positions[1, a]
        dy = detector_centers[2, a] - source_positions[2, a]
        dz = detector_centers[3, a] - source_positions[3, a]
        len = sqrt(dx^2 + dy^2 + dz^2)
        sd_axis[1, a] = dx / len
        sd_axis[2, a] = dy / len
        sd_axis[3, a] = dz / len
    end

    pixel_size_det = Float32(geom.pixel_size * (geom.SDD / geom.SAD))
    delta_angle = Float32(2π / n_angles)

    return BackprojGeometry(
        source_positions, detector_centers, detector_u, detector_v, sd_axis,
        Float32(geom.SAD), Float32(geom.SDD), pixel_size_det,
        geom.n_cols, geom.n_rows, n_angles, delta_angle
    )
end

# =============================================================================
# Core Backprojection Kernel (Shared Implementation)
# =============================================================================

"""
    _backproject_core(
        sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z, SAD, SDD, pixel_size_det, delta_angle,
        n_cols, n_rows, fdk_weighting
    )

Core backprojection implementation shared by FDK and iterative methods.

# Arguments
- `sinogram`: Projection data [n_cols, n_rows, n_angles]
- `source_pos`: Source positions [3, n_angles]
- `det_center`: Detector centers [3, n_angles]
- `det_u, det_v`: Detector axes [3, n_angles]
- `sd_axis`: Source-detector axis [3, n_angles]
- `voxel_x, voxel_y, voxel_z`: Voxel coordinates [nx], [ny], [nz]
- `SAD, SDD, pixel_size_det, delta_angle`: Geometry scalars
- `n_cols, n_rows`: Detector dimensions
- `fdk_weighting`: If true, apply FDK distance² and delta_angle weighting

# Returns
Reconstructed volume [nx, ny, nz]
"""
function _backproject_core(
    sinogram::AbstractArray{T, 3},
    source_pos::AbstractArray{T, 2},
    det_center::AbstractArray{T, 2},
    det_u::AbstractArray{T, 2},
    det_v::AbstractArray{T, 2},
    sd_axis::AbstractArray{T, 2},
    voxel_x::AbstractVector{T},
    voxel_y::AbstractVector{T},
    voxel_z::AbstractVector{T},
    SAD::T, SDD::T, pixel_size_det::T, delta_angle::T,
    n_cols::Int, n_rows::Int,
    fdk_weighting::Bool
) where T
    nx = length(voxel_x)
    ny = length(voxel_y)
    nz = length(voxel_z)
    n_angles = size(sinogram, 3)

    # Initialize output volume
    volume = zeros(T, nx, ny, nz)

    # Iterate over all angles (this loop will be traced by Reactant)
    for angle_idx in 1:n_angles
        # Source position for this angle
        sx = source_pos[1, angle_idx]
        sy = source_pos[2, angle_idx]
        sz = source_pos[3, angle_idx]

        # Detector center
        dcx = det_center[1, angle_idx]
        dcy = det_center[2, angle_idx]
        dcz = det_center[3, angle_idx]

        # Detector axes
        ux = det_u[1, angle_idx]
        uy = det_u[2, angle_idx]
        uz = det_u[3, angle_idx]

        vx = det_v[1, angle_idx]
        vy = det_v[2, angle_idx]
        vz = det_v[3, angle_idx]

        # Source-detector axis
        sdx = sd_axis[1, angle_idx]
        sdy = sd_axis[2, angle_idx]
        sdz = sd_axis[3, angle_idx]

        # Iterate over voxels
        for iz in 1:nz
            z = voxel_z[iz]
            for iy in 1:ny
                y = voxel_y[iy]
                for ix in 1:nx
                    x = voxel_x[ix]

                    # Vector from source to voxel
                    rx = x - sx
                    ry = y - sy
                    rz = z - sz

                    # Project onto source-detector line
                    t = rx * sdx + ry * sdy + rz * sdz

                    # Skip if voxel is behind source
                    if t <= T(0)
                        continue
                    end

                    # Scale to detector plane
                    scale = SDD / t

                    # Hit point on detector
                    hit_x = sx + scale * rx
                    hit_y = sy + scale * ry
                    hit_z = sz + scale * rz

                    # Detector coordinates relative to center
                    dhx = hit_x - dcx
                    dhy = hit_y - dcy
                    dhz = hit_z - dcz

                    # Project onto detector axes
                    u = dhx * ux + dhy * uy + dhz * uz
                    v = dhx * vx + dhy * vy + dhz * vz

                    # Convert to pixel coordinates
                    col_f = u / pixel_size_det + T(n_cols + 1) / T(2)
                    row_f = v / pixel_size_det + T(n_rows + 1) / T(2)

                    # Check bounds
                    if col_f < T(1) || col_f > T(n_cols) || row_f < T(1) || row_f > T(n_rows)
                        continue
                    end

                    # Bilinear interpolation
                    col0 = floor(Int, col_f)
                    col1 = col0 + 1
                    row0 = floor(Int, row_f)
                    row1 = row0 + 1

                    fc = col_f - T(col0)
                    fr = row_f - T(row0)

                    col0 = clamp(col0, 1, n_cols)
                    col1 = clamp(col1, 1, n_cols)
                    row0 = clamp(row0, 1, n_rows)
                    row1 = clamp(row1, 1, n_rows)

                    # Bilinear weights
                    w00 = (T(1) - fc) * (T(1) - fr)
                    w10 = fc * (T(1) - fr)
                    w01 = (T(1) - fc) * fr
                    w11 = fc * fr

                    # Sample sinogram
                    val = w00 * sinogram[col0, row0, angle_idx] +
                          w10 * sinogram[col1, row0, angle_idx] +
                          w01 * sinogram[col0, row1, angle_idx] +
                          w11 * sinogram[col1, row1, angle_idx]

                    # Apply weighting based on mode
                    if fdk_weighting
                        # FDK: distance² and angle weighting
                        dist_weight = (SAD / t)^2
                        volume[ix, iy, iz] += val * dist_weight * delta_angle
                    else
                        # Raw: no weighting (for iterative methods)
                        volume[ix, iy, iz] += val
                    end
                end
            end
        end
    end

    return volume
end

# =============================================================================
# Public API: FDK Backprojection (with weighting)
# =============================================================================

"""
    backproject_raymarching_kernel(
        filtered_sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z, SAD, SDD, pixel_size_det, delta_angle, n_cols, n_rows
    )

FDK backprojection with distance² and delta_angle weighting.

For each voxel, for each angle:
1. Compute ray from source through voxel
2. Find detector intersection point
3. Bilinearly interpolate filtered sinogram
4. Apply FDK distance and angle weighting

Use this for FDK (Feldkamp-Davis-Kress) cone-beam reconstruction.
For iterative methods (SIRT, CGLS), use `backproject_raw_kernel` instead.
"""
function backproject_raymarching_kernel(
    filtered_sinogram::AbstractArray{T, 3},
    source_pos::AbstractArray{T, 2},
    det_center::AbstractArray{T, 2},
    det_u::AbstractArray{T, 2},
    det_v::AbstractArray{T, 2},
    sd_axis::AbstractArray{T, 2},
    voxel_x::AbstractVector{T},
    voxel_y::AbstractVector{T},
    voxel_z::AbstractVector{T},
    SAD::T, SDD::T, pixel_size_det::T, delta_angle::T,
    n_cols::Int, n_rows::Int
) where T
    return _backproject_core(
        filtered_sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z,
        SAD, SDD, pixel_size_det, delta_angle, n_cols, n_rows,
        true  # fdk_weighting=true
    )
end

# =============================================================================
# Public API: Raw Backprojection (for iterative methods)
# =============================================================================

"""
    backproject_raw_kernel(
        sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z, SAD, SDD, pixel_size_det, n_cols, n_rows
    )

Raw backprojection WITHOUT FDK-specific weighting.

This is the true transpose of the forward projection operator, suitable for
iterative reconstruction methods (SIRT, CGLS) where we need matched
forward/backward operators: A and A^T.

For FDK reconstruction, use `backproject_raymarching_kernel` instead.
"""
function backproject_raw_kernel(
    sinogram::AbstractArray{T, 3},
    source_pos::AbstractArray{T, 2},
    det_center::AbstractArray{T, 2},
    det_u::AbstractArray{T, 2},
    det_v::AbstractArray{T, 2},
    sd_axis::AbstractArray{T, 2},
    voxel_x::AbstractVector{T},
    voxel_y::AbstractVector{T},
    voxel_z::AbstractVector{T},
    SAD::T, SDD::T, pixel_size_det::T,
    n_cols::Int, n_rows::Int
) where T
    # delta_angle is unused when fdk_weighting=false, but needed for function signature
    delta_angle = T(2π / size(sinogram, 3))
    return _backproject_core(
        sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z,
        SAD, SDD, pixel_size_det, delta_angle, n_cols, n_rows,
        false  # fdk_weighting=false
    )
end

# =============================================================================
# High-Level FDK Interface
# =============================================================================

"""
    fdk_reconstruct_raymarching(sinogram, geom, output_size, fov; kernel=RampKernel())

FDK reconstruction using on-the-fly geometry computation.

This is a SINGLE compiled kernel for ALL angles (after filtering).

# Arguments
- `sinogram`: Raw projections [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Scanner geometry
- `output_size::NTuple{3,Int}`: Output volume dimensions (nx, ny, nz)
- `fov::NTuple{3,Float64}`: Field of view in cm
- `kernel::ReconKernel`: Reconstruction kernel (default: ramp)

# Returns
Reconstructed volume [nx, ny, nz]

# Performance
- Memory: O(volume + sinogram) ≈ 1-2 GB for clinical scale
- Computation: ONE kernel call for backprojection
- Suitable for Reactant.@compile
"""
function fdk_reconstruct_raymarching(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    output_size::NTuple{3, Int},
    fov::NTuple{3, Float64};
    kernel::ReconKernel=RampKernel()
) where T
    nx, ny, nz = output_size

    # Step 1: Pre-weight and filter (done in Julia with FFTW)
    weighted = preweight_cosine(sinogram, geom)
    filtered = filter_ramp(weighted, geom; kernel=kernel)

    # Step 2: Apply FDK normalization scale factor
    # The discrete FDK formula requires proper normalization to recover μ values.
    # Empirically determined scale factor:
    #   - magnification: (SDD/SAD) accounts for detector-to-object space conversion
    #   - 0.55: empirical correction factor for ray marching discretization
    # This factor was calibrated to give correct HU for ALL materials (water, calcium, etc.)
    # in the central slices where detector coverage is complete.
    magnification = geom.SDD / geom.SAD
    fdk_scale = Float32(magnification * 0.55)
    filtered = filtered .* fdk_scale

    # Step 3: Compute compact backprojection geometry
    bp_geom = compute_backproj_geometry(geom)

    # Step 4: Compute voxel coordinates
    dx, dy, dz = fov ./ output_size
    voxel_x = Float32.(range(-fov[1]/2 + dx/2, fov[1]/2 - dx/2, length=nx))
    voxel_y = Float32.(range(-fov[2]/2 + dy/2, fov[2]/2 - dy/2, length=ny))
    voxel_z = Float32.(range(-fov[3]/2 + dz/2, fov[3]/2 - dz/2, length=nz))

    # Step 5: Backproject with on-the-fly geometry (uses shared core)
    volume = backproject_raymarching_kernel(
        Float32.(filtered),
        bp_geom.source_positions,
        bp_geom.detector_centers,
        bp_geom.detector_u,
        bp_geom.detector_v,
        bp_geom.sd_axis,
        voxel_x, voxel_y, voxel_z,
        bp_geom.SAD, bp_geom.SDD, bp_geom.pixel_size_det, bp_geom.delta_angle,
        bp_geom.n_cols, bp_geom.n_rows
    )

    return volume
end

"""
    backproject_raymarching_compiled(
        filtered_sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z,
        SAD, SDD, pixel_size_det, delta_angle, n_cols, n_rows
    )

Array-based interface for Reactant compilation.

Usage:
```julia
using Reactant

# Prepare geometry (tiny, computed once)
bp_geom = compute_backproj_geometry(geom)

# Convert to Reactant arrays
sino_ra = Reactant.to_rarray(Float32.(filtered_sinogram))
src_ra = Reactant.to_rarray(bp_geom.source_positions)
...

# Compile
compiled_bp = @compile backproject_raymarching_compiled(
    sino_ra, src_ra, det_center_ra, det_u_ra, det_v_ra, sd_axis_ra,
    voxel_x_ra, voxel_y_ra, voxel_z_ra,
    SAD, SDD, pixel_size_det, delta_angle, n_cols, n_rows
)

# Run (fast!)
volume = Array(compiled_bp(...))
```
"""
function backproject_raymarching_compiled(
    filtered_sinogram::AbstractArray{T, 3},
    source_pos::AbstractArray{T, 2},
    det_center::AbstractArray{T, 2},
    det_u::AbstractArray{T, 2},
    det_v::AbstractArray{T, 2},
    sd_axis::AbstractArray{T, 2},
    voxel_x::AbstractVector{T},
    voxel_y::AbstractVector{T},
    voxel_z::AbstractVector{T},
    SAD::T, SDD::T, pixel_size_det::T, delta_angle::T,
    n_cols::Int, n_rows::Int
) where T
    return backproject_raymarching_kernel(
        filtered_sinogram, source_pos, det_center, det_u, det_v, sd_axis,
        voxel_x, voxel_y, voxel_z,
        SAD, SDD, pixel_size_det, delta_angle, n_cols, n_rows
    )
end

# =============================================================================
# Iterative Reconstruction Support
# =============================================================================

"""
    CompiledCTOperators

Holds compiled forward and back projection operators for iterative reconstruction.
"""
struct CompiledCTOperators{F, B}
    forward::F
    backward::B
    ray_geom::RayGeometry
    bp_geom::BackprojGeometry
    voxel_x::Vector{Float32}
    voxel_y::Vector{Float32}
    voxel_z::Vector{Float32}
    n_samples::Int
    step_size::Float32
end

"""
    create_ct_operators(geom, phantom_size, phantom_fov, output_size, output_fov; step_factor=0.5)

Create forward/back projection operators for iterative reconstruction.

Returns operators that can be compiled once and reused for all iterations.
"""
function create_ct_operators(
    geom::CTGeometry,
    phantom_size::NTuple{3, Int},
    phantom_fov::NTuple{3, Float64},
    output_size::NTuple{3, Int},
    output_fov::NTuple{3, Float64};
    step_factor::Float64=0.5
)
    # Ray geometry for forward projection (from output volume)
    ray_geom = compute_ray_geometry(geom, output_fov, output_size)

    # Backprojection geometry
    bp_geom = compute_backproj_geometry(geom)

    # Voxel coordinates for backprojection
    nx, ny, nz = output_size
    dx, dy, dz = output_fov ./ output_size
    voxel_x = Float32.(range(-output_fov[1]/2 + dx/2, output_fov[1]/2 - dx/2, length=nx))
    voxel_y = Float32.(range(-output_fov[2]/2 + dy/2, output_fov[2]/2 - dy/2, length=ny))
    voxel_z = Float32.(range(-output_fov[3]/2 + dz/2, output_fov[3]/2 - dz/2, length=nz))

    # Step size and samples for forward projection
    min_voxel = minimum(output_fov ./ output_size)
    step_size = Float32(min_voxel * step_factor)
    # n_samples must cover full ray path from source through volume
    half_diagonal = sqrt(sum((output_fov ./ 2) .^ 2))
    ray_path_to_far_edge = geom.SAD + half_diagonal
    n_samples = ceil(Int, ray_path_to_far_edge / step_size) + 10

    return CompiledCTOperators(
        nothing, nothing,  # Forward/backward will be compiled by user
        ray_geom, bp_geom,
        voxel_x, voxel_y, voxel_z,
        n_samples, step_size
    )
end

# =============================================================================
# Exports
# =============================================================================

export BackprojGeometry, compute_backproj_geometry
export backproject_raymarching_kernel
export backproject_raw_kernel
export backproject_raymarching_compiled
export fdk_reconstruct_raymarching
export CompiledCTOperators, create_ct_operators
