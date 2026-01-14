# =============================================================================
# Siddon Forward Projection (TIGRE-style, AcceleratedKernels.jl)
# =============================================================================
#
# Direct port of TIGRE's Siddon algorithm using AcceleratedKernels.jl
# for backend-agnostic GPU/CPU execution.
#
# Reference:
#   - Siddon (1985) "Fast calculation of the exact radiological path"
#   - TIGRE: CERN/TIGRE/Common/CUDA/Siddon_projection.cu
#
# Key advantage of AcceleratedKernels.jl:
#   - No @kernel macros needed - just normal Julia code in typed functions
#   - AK.foreachindex automatically parallelizes
#   - Works on Metal (Apple), CUDA (NVIDIA), ROCm (AMD), or CPU
#
# =============================================================================

import AcceleratedKernels as AK

export siddon_forward_project!, siddon_forward_project

# =============================================================================
# Single Ray Trace (inlined into the main loop)
# =============================================================================

"""
    siddon_trace_ray(...)

Trace a single ray through the volume using Siddon's algorithm.
Port of TIGRE's ray tracing logic from Siddon_projection.cu

Returns line integral (path-length weighted sum of attenuation)

Note: Uses Int32 for dimensions to ensure GPU compatibility.
"""
@inline function siddon_trace_ray(
    volume::AbstractArray{T, 3},
    src_x::T, src_y::T, src_z::T,
    det_x::T, det_y::T, det_z::T,
    vol_min_x::T, vol_min_y::T, vol_min_z::T,
    vol_max_x::T, vol_max_y::T, vol_max_z::T,
    voxel_size_x::T, voxel_size_y::T, voxel_size_z::T,
    nx::Int32, ny::Int32, nz::Int32
) where T

    # Ray direction (not normalized - we use parametric form)
    ray_x = det_x - src_x
    ray_y = det_y - src_y
    ray_z = det_z - src_z

    # Ray length for proper path length scaling
    ray_length = sqrt(ray_x^2 + ray_y^2 + ray_z^2)

    # Avoid division by zero
    eps = T(1e-10)
    ray_x = abs(ray_x) < eps ? (ray_x >= zero(T) ? eps : -eps) : ray_x
    ray_y = abs(ray_y) < eps ? (ray_y >= zero(T) ? eps : -eps) : ray_y
    ray_z = abs(ray_z) < eps ? (ray_z >= zero(T) ? eps : -eps) : ray_z

    # Compute parametric t where ray intersects volume boundaries
    # Following TIGRE: t = (boundary - source) / ray
    t_x_min = (vol_min_x - src_x) / ray_x
    t_x_max = (vol_max_x - src_x) / ray_x
    t_y_min = (vol_min_y - src_y) / ray_y
    t_y_max = (vol_max_y - src_y) / ray_y
    t_z_min = (vol_min_z - src_z) / ray_z
    t_z_max = (vol_max_z - src_z) / ray_z

    # Sort to get entry/exit (TIGRE style)
    if t_x_min > t_x_max
        t_x_min, t_x_max = t_x_max, t_x_min
    end
    if t_y_min > t_y_max
        t_y_min, t_y_max = t_y_max, t_y_min
    end
    if t_z_min > t_z_max
        t_z_min, t_z_max = t_z_max, t_z_min
    end

    # Ray enters at max of entry t, exits at min of exit t
    t_enter = max(t_x_min, t_y_min, t_z_min)
    t_exit = min(t_x_max, t_y_max, t_z_max)

    # Check if ray misses volume
    if t_enter >= t_exit || t_exit <= zero(T)
        return zero(T)
    end

    # Clamp to positive t
    t_enter = max(t_enter, zero(T))

    # Entry point
    entry_x = src_x + t_enter * ray_x
    entry_y = src_y + t_enter * ray_y
    entry_z = src_z + t_enter * ray_z

    # Initial voxel indices (0-based for computation, Int32 for GPU)
    ix = unsafe_trunc(Int32, floor((entry_x - vol_min_x) / voxel_size_x))
    iy = unsafe_trunc(Int32, floor((entry_y - vol_min_y) / voxel_size_y))
    iz = unsafe_trunc(Int32, floor((entry_z - vol_min_z) / voxel_size_z))

    # Clamp to valid range
    ix = clamp(ix, Int32(0), nx - Int32(1))
    iy = clamp(iy, Int32(0), ny - Int32(1))
    iz = clamp(iz, Int32(0), nz - Int32(1))

    # Step direction (TIGRE: iu, ju, ku)
    step_x = ray_x >= zero(T) ? Int32(1) : Int32(-1)
    step_y = ray_y >= zero(T) ? Int32(1) : Int32(-1)
    step_z = ray_z >= zero(T) ? Int32(1) : Int32(-1)

    # Delta t to cross one voxel (TIGRE: axu, ayu, azu)
    dt_x = abs(voxel_size_x / ray_x)
    dt_y = abs(voxel_size_y / ray_y)
    dt_z = abs(voxel_size_z / ray_z)

    # t to next boundary
    if ray_x >= zero(T)
        t_next_x = t_enter + (vol_min_x + T(ix + Int32(1)) * voxel_size_x - entry_x) / ray_x
    else
        t_next_x = t_enter + (vol_min_x + T(ix) * voxel_size_x - entry_x) / ray_x
    end

    if ray_y >= zero(T)
        t_next_y = t_enter + (vol_min_y + T(iy + Int32(1)) * voxel_size_y - entry_y) / ray_y
    else
        t_next_y = t_enter + (vol_min_y + T(iy) * voxel_size_y - entry_y) / ray_y
    end

    if ray_z >= zero(T)
        t_next_z = t_enter + (vol_min_z + T(iz + Int32(1)) * voxel_size_z - entry_z) / ray_z
    else
        t_next_z = t_enter + (vol_min_z + T(iz) * voxel_size_z - entry_z) / ray_z
    end

    # Traverse voxels and accumulate (TIGRE main loop)
    t_current = t_enter
    line_integral = zero(T)

    # Maximum iterations to prevent infinite loops (use Int32)
    max_iter = nx + ny + nz + Int32(10)
    iter = Int32(0)

    while t_current < t_exit && iter < max_iter
        iter += Int32(1)

        # Check bounds
        if ix < Int32(0) || ix >= nx || iy < Int32(0) || iy >= ny || iz < Int32(0) || iz >= nz
            break
        end

        # Next boundary crossing
        t_next = min(t_next_x, t_next_y, t_next_z, t_exit)

        # Path length through this voxel
        path_length = (t_next - t_current) * ray_length

        if path_length > eps
            # Get voxel value (1-based indexing)
            voxel_val = volume[ix + Int32(1), iy + Int32(1), iz + Int32(1)]
            line_integral += voxel_val * path_length
        end

        # Step to next voxel (TIGRE style comparison)
        if t_next_x <= t_next_y && t_next_x <= t_next_z
            ix += step_x
            t_next_x += dt_x
        elseif t_next_y <= t_next_z
            iy += step_y
            t_next_y += dt_y
        else
            iz += step_z
            t_next_z += dt_z
        end

        t_current = t_next
    end

    return line_integral
end

# =============================================================================
# High-Level Interface using AcceleratedKernels.jl
# =============================================================================

"""
    siddon_forward_project!(sinogram, volume, geom)

In-place Siddon forward projection using AcceleratedKernels.jl.

Automatically runs on GPU (Metal/CUDA/ROCm) or CPU based on array type.

# Arguments
- `sinogram`: Output array [n_cols, n_rows, n_angles] (modified in place)
- `volume`: Input volume [nx, ny, nz]
- `geom`: CTGeometry with scanner parameters

# Returns
The modified sinogram array
"""
function siddon_forward_project!(
    sinogram::AbstractArray{T, 3},
    volume::AbstractArray{T, 3},
    geom::CTGeometry
) where T <: AbstractFloat

    # Get dimensions as Int32 for GPU compatibility
    nx = Int32(size(volume, 1))
    ny = Int32(size(volume, 2))
    nz = Int32(size(volume, 3))
    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Pre-compute volume parameters (typed constants for GPU)
    vol_min_x = T(-geom.fov[1] / 2)
    vol_min_y = T(-geom.fov[2] / 2)
    vol_min_z = T(-geom.fov[3] / 2)
    vol_max_x = T(geom.fov[1] / 2)
    vol_max_y = T(geom.fov[2] / 2)
    vol_max_z = T(geom.fov[3] / 2)
    voxel_size_x = T(geom.fov[1]) / T(nx)
    voxel_size_y = T(geom.fov[2]) / T(ny)
    voxel_size_z = T(geom.fov[3]) / T(nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)

    # Pre-compute detector center offset for GPU
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Extract geometry and convert to same array type as volume (GPU compatibility)
    source_positions = similar(volume, T, size(geom.source_positions)...)
    copyto!(source_positions, T.(geom.source_positions))
    detector_centers = similar(volume, T, size(geom.detector_centers)...)
    copyto!(detector_centers, T.(geom.detector_centers))
    detector_u = similar(volume, T, size(geom.detector_u)...)
    copyto!(detector_u, T.(geom.detector_u))
    detector_v = similar(volume, T, size(geom.detector_v)...)
    copyto!(detector_v, T.(geom.detector_v))

    # Use AcceleratedKernels.jl to parallelize over all rays
    AK.foreachindex(sinogram) do idx
        # Convert linear index to (col, row, angle) using integer arithmetic
        # Use Int32 throughout to avoid boxing
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)
        angle = (idx_0 ÷ n_rows) + Int32(1)

        # Source position for this angle
        src_x = source_positions[1, angle]
        src_y = source_positions[2, angle]
        src_z = source_positions[3, angle]

        # Detector center and orientation
        dcx = detector_centers[1, angle]
        dcy = detector_centers[2, angle]
        dcz = detector_centers[3, angle]

        dux = detector_u[1, angle]
        duy = detector_u[2, angle]
        duz = detector_u[3, angle]

        dvx = detector_v[1, angle]
        dvy = detector_v[2, angle]
        dvz = detector_v[3, angle]

        # Compute detector pixel position
        u_offset = (T(col) - col_center) * pixel_size * magnification
        v_offset = (T(row) - row_center) * pixel_size * magnification

        det_x = dcx + u_offset * dux + v_offset * dvx
        det_y = dcy + u_offset * duy + v_offset * dvy
        det_z = dcz + u_offset * duz + v_offset * dvz

        # Trace ray and store result
        sinogram[idx] = siddon_trace_ray(
            volume,
            src_x, src_y, src_z,
            det_x, det_y, det_z,
            vol_min_x, vol_min_y, vol_min_z,
            vol_max_x, vol_max_y, vol_max_z,
            voxel_size_x, voxel_size_y, voxel_size_z,
            nx, ny, nz
        )
    end

    return sinogram
end

"""
    siddon_forward_project(volume, geom)

Allocating version of Siddon forward projection.

# Arguments
- `volume`: Input volume [nx, ny, nz]
- `geom`: CTGeometry with scanner parameters

# Returns
New sinogram array [n_cols, n_rows, n_angles]
"""
function siddon_forward_project(
    volume::AbstractArray{T, 3},
    geom::CTGeometry
) where T <: AbstractFloat

    # Allocate output on same device as input
    sinogram = similar(volume, T, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, zero(T))

    return siddon_forward_project!(sinogram, volume, geom)
end
