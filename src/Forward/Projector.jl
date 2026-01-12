"""
    Forward/Projector.jl

Ray-driven forward projector for cone-beam CT.

Designed for Reactant/XLA compatibility:
- No runtime trigonometric functions (uses pre-computed geometry)
- No dynamic control flow (continue/break)
- Generic array types for TracedRArray support
- Uses @allowscalar for array indexing (future: vectorize fully)
"""

using Reactant
using Reactant: @allowscalar

# =============================================================================
# Forward Projection
# =============================================================================

"""
    forward_project(phantom::Phantom, geom::CTGeometry; n_steps::Int=256)

Compute cone-beam projections (sinogram) from a phantom volume.

# Arguments
- `phantom::Phantom`: Input phantom with μ values (cm⁻¹)
- `geom::CTGeometry`: Scanner geometry with pre-computed positions
- `n_steps::Int`: Number of steps along each ray (default 256)

# Returns
`Array{Float32,3}`: Sinogram of shape [n_cols, n_rows, n_angles]
- Values are line integrals of μ (dimensionless, cm⁻¹ × cm)

# Algorithm
Ray-driven projection: for each detector pixel, trace a ray from source
through volume, accumulating μ values using trilinear interpolation.
"""
function forward_project(phantom::Phantom, geom::CTGeometry; n_steps::Int=256)
    # Output sinogram: [n_cols, n_rows, n_angles]
    sinogram = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)

    # Ray casting through volume
    forward_project!(sinogram, phantom.μ, phantom.voxel_size, phantom.fov, geom, n_steps)

    return sinogram
end

"""
    forward_project!(sinogram, volume, voxel_size, fov, geom, n_steps)

In-place forward projection (Reactant-compilable core).

Uses dense ray marching without early termination for XLA compatibility.
Rays that miss the volume contribute zero.
"""
function forward_project!(
    sinogram,
    volume,
    voxel_size::NTuple{3,Float64},
    fov::NTuple{3,Float64},
    geom::CTGeometry,
    n_steps::Int
)
    nx, ny, nz = size(volume)

    # Volume bounds (centered at origin)
    x_min, x_max = -fov[1]/2, fov[1]/2
    y_min, y_max = -fov[2]/2, fov[2]/2
    z_min, z_max = -fov[3]/2, fov[3]/2

    # Detector pixel size at detector plane
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    # Pre-extract geometry arrays for Reactant
    source_pos = geom.source_positions
    det_centers = geom.detector_centers
    det_u = geom.detector_u
    det_v = geom.detector_v

    for angle_idx in 1:geom.n_angles
        # Source position for this angle
        sx = source_pos[1, angle_idx]
        sy = source_pos[2, angle_idx]
        sz = source_pos[3, angle_idx]

        # Detector center and axes
        dcx = det_centers[1, angle_idx]
        dcy = det_centers[2, angle_idx]
        dcz = det_centers[3, angle_idx]

        ux = det_u[1, angle_idx]
        uy = det_u[2, angle_idx]
        uz = det_u[3, angle_idx]

        vx = det_v[1, angle_idx]
        vy = det_v[2, angle_idx]
        vz = det_v[3, angle_idx]

        for row in 1:geom.n_rows
            for col in 1:geom.n_cols
                # Detector pixel position
                u_offset = (col - (geom.n_cols + 1) / 2) * pixel_size_det
                v_offset = (row - (geom.n_rows + 1) / 2) * pixel_size_det

                dx = dcx + u_offset * ux + v_offset * vx
                dy = dcy + u_offset * uy + v_offset * vy
                dz = dcz + u_offset * uz + v_offset * vz

                # Ray direction (source to detector)
                ray_dx = dx - sx
                ray_dy = dy - sy
                ray_dz = dz - sz
                ray_len = sqrt(ray_dx^2 + ray_dy^2 + ray_dz^2)

                # Normalize ray direction
                ray_dx = ray_dx / ray_len
                ray_dy = ray_dy / ray_len
                ray_dz = ray_dz / ray_len

                # Find intersection with volume bounding box
                t_min, t_max = ray_box_intersection_xla(
                    sx, sy, sz, ray_dx, ray_dy, ray_dz,
                    x_min, x_max, y_min, y_max, z_min, z_max
                )

                # Compute step size (will be 0 or negative if no intersection)
                # Use max to ensure non-negative range
                t_range = max(t_max - t_min, 0.0)
                step_size = t_range / n_steps

                # Accumulate line integral (0 if ray misses volume)
                line_integral = zero(eltype(volume))

                for step in 0:n_steps-1
                    t = t_min + (step + 0.5) * step_size

                    # Sample point
                    px = sx + t * ray_dx
                    py = sy + t * ray_dy
                    pz = sz + t * ray_dz

                    # Sample volume using trilinear interpolation
                    μ_val = sample_volume_trilinear_xla(
                        volume, px, py, pz,
                        x_min, y_min, z_min,
                        voxel_size[1], voxel_size[2], voxel_size[3],
                        nx, ny, nz
                    )

                    line_integral = line_integral + μ_val * step_size
                end

                sinogram[col, row, angle_idx] = line_integral
            end
        end
    end

    return sinogram
end

"""
    ray_box_intersection_xla(ox, oy, oz, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max)

Compute ray-box intersection using slab method (XLA-compatible, no branches).

Returns (t_min, t_max) where t parameterizes the ray as O + t*D.
If t_max <= t_min, ray doesn't intersect box.
"""
function ray_box_intersection_xla(
    ox, oy, oz,  # Ray origin
    dx, dy, dz,  # Ray direction (normalized)
    x_min, x_max, y_min, y_max, z_min, z_max  # Box bounds
)
    # Use small epsilon to avoid division by zero
    eps = 1e-10

    # Safe inverse (add epsilon to denominator)
    inv_dx = 1.0 / (dx + eps * sign(abs(dx) < eps))
    inv_dy = 1.0 / (dy + eps * sign(abs(dy) < eps))
    inv_dz = 1.0 / (dz + eps * sign(abs(dz) < eps))

    # Compute t values for each slab
    t1_x = (x_min - ox) * inv_dx
    t2_x = (x_max - ox) * inv_dx
    t1_y = (y_min - oy) * inv_dy
    t2_y = (y_max - oy) * inv_dy
    t1_z = (z_min - oz) * inv_dz
    t2_z = (z_max - oz) * inv_dz

    # Get min/max for each axis (handles negative directions)
    t_x_min = min(t1_x, t2_x)
    t_x_max = max(t1_x, t2_x)
    t_y_min = min(t1_y, t2_y)
    t_y_max = max(t1_y, t2_y)
    t_z_min = min(t1_z, t2_z)
    t_z_max = max(t1_z, t2_z)

    # Find intersection of all slabs
    t_min = max(t_x_min, t_y_min, t_z_min, 0.0)  # Clamp to positive
    t_max = min(t_x_max, t_y_max, t_z_max)

    return t_min, t_max
end

"""
    sample_volume_trilinear_xla(volume, x, y, z, x_min, y_min, z_min, dx, dy, dz, nx, ny, nz)

Sample a 3D volume at continuous coordinates using trilinear interpolation.
XLA-compatible version without floor(Int, ...) - uses trunc instead.
"""
function sample_volume_trilinear_xla(
    volume,
    x, y, z,           # World coordinates
    x_min, y_min, z_min,  # Volume origin
    dx, dy, dz,        # Voxel sizes
    nx, ny, nz         # Volume dimensions
)
    # Convert to voxel coordinates (0-indexed continuous)
    vx = (x - x_min) / dx - 0.5
    vy = (y - y_min) / dy - 0.5
    vz = (z - z_min) / dz - 0.5

    # Get integer voxel indices using trunc (XLA-compatible)
    # trunc returns Float, we need to be careful with indexing
    ix_f = trunc(vx)
    iy_f = trunc(vy)
    iz_f = trunc(vz)

    # Fractional parts
    fx = vx - ix_f
    fy = vy - iy_f
    fz = vz - iz_f

    # Convert to 1-based indices and clamp
    # Use arithmetic clamping for XLA compatibility
    ix0 = clamp_index(ix_f + 1.0, nx)
    ix1 = clamp_index(ix_f + 2.0, nx)
    iy0 = clamp_index(iy_f + 1.0, ny)
    iy1 = clamp_index(iy_f + 2.0, ny)
    iz0 = clamp_index(iz_f + 1.0, nz)
    iz1 = clamp_index(iz_f + 2.0, nz)

    # Trilinear interpolation with safe indexing
    c000 = safe_getindex(volume, ix0, iy0, iz0, nx, ny, nz)
    c100 = safe_getindex(volume, ix1, iy0, iz0, nx, ny, nz)
    c010 = safe_getindex(volume, ix0, iy1, iz0, nx, ny, nz)
    c110 = safe_getindex(volume, ix1, iy1, iz0, nx, ny, nz)
    c001 = safe_getindex(volume, ix0, iy0, iz1, nx, ny, nz)
    c101 = safe_getindex(volume, ix1, iy0, iz1, nx, ny, nz)
    c011 = safe_getindex(volume, ix0, iy1, iz1, nx, ny, nz)
    c111 = safe_getindex(volume, ix1, iy1, iz1, nx, ny, nz)

    # Interpolate along x
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    # Interpolate along y
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    # Interpolate along z
    return c0 * (1 - fz) + c1 * fz
end

"""
    clamp_index(idx_f, n)

Clamp a floating-point index to valid range [1, n] and convert to Int.
"""
@inline function clamp_index(idx_f, n)
    clamped = max(1.0, min(Float64(n), idx_f))
    return unsafe_trunc(Int, clamped)
end

"""
    safe_getindex(volume, ix, iy, iz, nx, ny, nz)

Safely get value from volume, returning 0 for out-of-bounds indices.
"""
@inline function safe_getindex(volume, ix, iy, iz, nx, ny, nz)
    # Check bounds (using arithmetic to avoid branches)
    in_bounds = (1 <= ix <= nx) && (1 <= iy <= ny) && (1 <= iz <= nz)
    if in_bounds
        return @inbounds volume[ix, iy, iz]
    else
        return zero(eltype(volume))
    end
end

# =============================================================================
# Exports
# =============================================================================

export forward_project, forward_project!
