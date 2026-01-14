# =============================================================================
# Voxel-Driven Backprojection (TIGRE-style, KernelAbstractions.jl)
# =============================================================================
#
# Direct port of TIGRE's voxel backprojection algorithm using KernelAbstractions.jl
# for backend-agnostic GPU/CPU execution.
#
# Reference:
#   - TIGRE: CERN/TIGRE/Common/CUDA/voxel_backprojection.cu
#   - Feldkamp, Davis, Kress (1984) for FDK weights
#
# =============================================================================

using KernelAbstractions

export backproject!, backproject

# =============================================================================
# GPU Kernel for Voxel Backprojection
# =============================================================================

@kernel function backproject_kernel!(
    volume, @Const(sinogram),
    @Const(source_positions), @Const(detector_centers),
    @Const(detector_u), @Const(detector_v),
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size_x, voxel_size_y, voxel_size_z,
    nx::Int32, ny::Int32, nz::Int32,
    n_cols::Int32, n_rows::Int32, n_angles::Int32,
    col_center, row_center,
    pixel_size, magnification, SAD, pi_over_angles
)
    idx = @index(Global)
    T = eltype(volume)

    # Convert linear index to (ix, iy, iz) using integer arithmetic
    idx_0 = idx - Int32(1)
    ix = (idx_0 % nx) + Int32(1)
    idx_0 = idx_0 ÷ nx
    iy = (idx_0 % ny) + Int32(1)
    iz = (idx_0 ÷ ny) + Int32(1)

    # Voxel center in world coordinates
    half = T(0.5)
    voxel_x = vol_min_x + (T(ix) - half) * voxel_size_x
    voxel_y = vol_min_y + (T(iy) - half) * voxel_size_y
    voxel_z = vol_min_z + (T(iz) - half) * voxel_size_z

    # Accumulator for backprojection
    acc = zero(T)
    SAD_sq = SAD * SAD
    pixel_mag = pixel_size * magnification

    # Loop over all angles
    for angle in Int32(1):n_angles
        # Source position
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

        # Vector from source to voxel
        sv_x = voxel_x - src_x
        sv_y = voxel_y - src_y
        sv_z = voxel_z - src_z

        # Vector from source to detector center
        sd_x = dcx - src_x
        sd_y = dcy - src_y
        sd_z = dcz - src_z

        # Distance squared from source to detector center
        sd_len_sq = sd_x^2 + sd_y^2 + sd_z^2

        # Parameter t where ray from source through voxel intersects detector plane
        sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z
        t = sd_len_sq / sv_dot_sd

        # Projected point on detector plane
        proj_x = src_x + t * sv_x
        proj_y = src_y + t * sv_y
        proj_z = src_z + t * sv_z

        # Vector from detector center to projected point
        dp_x = proj_x - dcx
        dp_y = proj_y - dcy
        dp_z = proj_z - dcz

        # Detector coordinates (u, v)
        u = (dp_x * dux + dp_y * duy + dp_z * duz) / pixel_mag
        v = (dp_x * dvx + dp_y * dvy + dp_z * dvz) / pixel_mag

        # Convert to pixel indices (centered)
        col_f = u + col_center
        row_f = v + row_center

        # Check if within detector bounds
        if col_f >= one(T) && col_f <= T(n_cols) && row_f >= one(T) && row_f <= T(n_rows)
            # Bilinear interpolation indices
            col_lo = unsafe_trunc(Int32, col_f)
            col_hi = col_lo + Int32(1)
            row_lo = unsafe_trunc(Int32, row_f)
            row_hi = row_lo + Int32(1)

            # Interpolation weights
            w_col = col_f - T(col_lo)
            w_row = row_f - T(row_lo)

            # Clamp indices to valid range
            col_lo = clamp(col_lo, Int32(1), n_cols)
            col_hi = clamp(col_hi, Int32(1), n_cols)
            row_lo = clamp(row_lo, Int32(1), n_rows)
            row_hi = clamp(row_hi, Int32(1), n_rows)

            # Bilinear interpolation
            val = (one(T) - w_col) * (one(T) - w_row) * sinogram[col_lo, row_lo, angle] +
                  w_col * (one(T) - w_row) * sinogram[col_hi, row_lo, angle] +
                  (one(T) - w_col) * w_row * sinogram[col_lo, row_hi, angle] +
                  w_col * w_row * sinogram[col_hi, row_hi, angle]

            # FDK distance weighting
            dist_sq = sv_x^2 + sv_y^2 + sv_z^2
            weight = SAD_sq / dist_sq

            acc += val * weight
        end
    end

    # Scale by angle step
    volume[idx] = acc * pi_over_angles
end

# =============================================================================
# High-Level Interface
# =============================================================================

"""
    backproject!(volume, sinogram, geom)

In-place FDK backprojection using KernelAbstractions.jl.

Automatically runs on GPU (Metal/CUDA/ROCm) or CPU based on array type.

# Arguments
- `volume`: Output volume [nx, ny, nz] (modified in place)
- `sinogram`: Filtered sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters

# Returns
The modified volume array
"""
function backproject!(
    volume::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry
) where T <: AbstractFloat

    nx, ny, nz = size(volume)
    n_cols, n_rows, n_angles = size(sinogram)

    # Volume parameters (typed constants)
    vol_min_x = T(-geom.fov[1] / 2)
    vol_min_y = T(-geom.fov[2] / 2)
    vol_min_z = T(-geom.fov[3] / 2)

    voxel_size_x = T(geom.fov[1] / nx)
    voxel_size_y = T(geom.fov[2] / ny)
    voxel_size_z = T(geom.fov[3] / nz)

    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    SAD = T(geom.SAD)

    # Pre-compute constants
    col_center = T((n_cols + 1) / 2)
    row_center = T((n_rows + 1) / 2)
    pi_over_angles = T(π) / T(n_angles)

    # Copy geometry arrays to same device as sinogram
    source_positions = similar(sinogram, T, size(geom.source_positions)...)
    copyto!(source_positions, T.(geom.source_positions))
    detector_centers = similar(sinogram, T, size(geom.detector_centers)...)
    copyto!(detector_centers, T.(geom.detector_centers))
    detector_u = similar(sinogram, T, size(geom.detector_u)...)
    copyto!(detector_u, T.(geom.detector_u))
    detector_v = similar(sinogram, T, size(geom.detector_v)...)
    copyto!(detector_v, T.(geom.detector_v))

    # Get backend from array type
    backend = KernelAbstractions.get_backend(volume)

    # Launch kernel
    kernel! = backproject_kernel!(backend)
    kernel!(
        volume, sinogram,
        source_positions, detector_centers,
        detector_u, detector_v,
        vol_min_x, vol_min_y, vol_min_z,
        voxel_size_x, voxel_size_y, voxel_size_z,
        Int32(nx), Int32(ny), Int32(nz),
        Int32(n_cols), Int32(n_rows), Int32(n_angles),
        col_center, row_center,
        pixel_size, magnification, SAD, pi_over_angles;
        ndrange=length(volume)
    )
    KernelAbstractions.synchronize(backend)

    return volume
end

"""
    backproject(sinogram, geom, volume_size)

Allocating version of FDK backprojection.

# Arguments
- `sinogram`: Filtered sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Returns
New volume array [nx, ny, nz]
"""
function backproject(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int}
) where T <: AbstractFloat

    volume = similar(sinogram, T, volume_size...)
    fill!(volume, zero(T))

    return backproject!(volume, sinogram, geom)
end
