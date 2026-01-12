"""
    Forward/Projector.jl

Ray-driven forward projector for cone-beam CT.

Designed for Reactant/XLA compatibility:
- No runtime trigonometric functions (uses pre-computed geometry)
- Simple loops amenable to XLA compilation
- Pre-allocated output arrays
"""

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
"""
function forward_project!(
    sinogram::Array{Float32,3},
    volume::Array{Float32,3},
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

    for angle_idx in 1:geom.n_angles
        # Source position for this angle
        sx = geom.source_positions[1, angle_idx]
        sy = geom.source_positions[2, angle_idx]
        sz = geom.source_positions[3, angle_idx]

        # Detector center and axes
        dcx = geom.detector_centers[1, angle_idx]
        dcy = geom.detector_centers[2, angle_idx]
        dcz = geom.detector_centers[3, angle_idx]

        ux = geom.detector_u[1, angle_idx]
        uy = geom.detector_u[2, angle_idx]
        uz = geom.detector_u[3, angle_idx]

        vx = geom.detector_v[1, angle_idx]
        vy = geom.detector_v[2, angle_idx]
        vz = geom.detector_v[3, angle_idx]

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
                ray_dx /= ray_len
                ray_dy /= ray_len
                ray_dz /= ray_len

                # Find intersection with volume bounding box
                t_min, t_max = ray_box_intersection(
                    sx, sy, sz, ray_dx, ray_dy, ray_dz,
                    x_min, x_max, y_min, y_max, z_min, z_max
                )

                if t_max <= t_min
                    continue  # Ray doesn't intersect volume
                end

                # Step size for ray marching
                step_size = (t_max - t_min) / n_steps

                # Accumulate line integral
                line_integral = 0.0f0

                for step in 0:n_steps-1
                    t = t_min + (step + 0.5) * step_size

                    # Sample point
                    px = sx + t * ray_dx
                    py = sy + t * ray_dy
                    pz = sz + t * ray_dz

                    # Sample volume using trilinear interpolation
                    μ_val = sample_volume_trilinear(
                        volume, px, py, pz,
                        x_min, y_min, z_min,
                        voxel_size[1], voxel_size[2], voxel_size[3],
                        nx, ny, nz
                    )

                    line_integral += μ_val * Float32(step_size)
                end

                sinogram[col, row, angle_idx] = line_integral
            end
        end
    end

    return sinogram
end

"""
    ray_box_intersection(ox, oy, oz, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max)

Compute ray-box intersection using slab method.

Returns (t_min, t_max) where t parameterizes the ray as O + t*D.
If t_max <= t_min, ray doesn't intersect box.
"""
function ray_box_intersection(
    ox, oy, oz,  # Ray origin
    dx, dy, dz,  # Ray direction (normalized)
    x_min, x_max, y_min, y_max, z_min, z_max  # Box bounds
)
    # Avoid division by zero
    inv_dx = dx != 0 ? 1.0 / dx : 1e10 * sign(dx + 1e-10)
    inv_dy = dy != 0 ? 1.0 / dy : 1e10 * sign(dy + 1e-10)
    inv_dz = dz != 0 ? 1.0 / dz : 1e10 * sign(dz + 1e-10)

    # X slab
    if inv_dx >= 0
        t_x_min = (x_min - ox) * inv_dx
        t_x_max = (x_max - ox) * inv_dx
    else
        t_x_min = (x_max - ox) * inv_dx
        t_x_max = (x_min - ox) * inv_dx
    end

    # Y slab
    if inv_dy >= 0
        t_y_min = (y_min - oy) * inv_dy
        t_y_max = (y_max - oy) * inv_dy
    else
        t_y_min = (y_max - oy) * inv_dy
        t_y_max = (y_min - oy) * inv_dy
    end

    # Z slab
    if inv_dz >= 0
        t_z_min = (z_min - oz) * inv_dz
        t_z_max = (z_max - oz) * inv_dz
    else
        t_z_min = (z_max - oz) * inv_dz
        t_z_max = (z_min - oz) * inv_dz
    end

    # Find intersection of all slabs
    t_min = max(t_x_min, t_y_min, t_z_min, 0.0)  # Clamp to positive
    t_max = min(t_x_max, t_y_max, t_z_max)

    return t_min, t_max
end

"""
    sample_volume_trilinear(volume, x, y, z, x_min, y_min, z_min, dx, dy, dz, nx, ny, nz)

Sample a 3D volume at continuous coordinates using trilinear interpolation.
"""
function sample_volume_trilinear(
    volume::Array{Float32,3},
    x, y, z,           # World coordinates
    x_min, y_min, z_min,  # Volume origin
    dx, dy, dz,        # Voxel sizes
    nx, ny, nz         # Volume dimensions
)
    # Convert to voxel coordinates (0-indexed continuous)
    vx = (x - x_min) / dx - 0.5
    vy = (y - y_min) / dy - 0.5
    vz = (z - z_min) / dz - 0.5

    # Get integer voxel indices
    ix = floor(Int, vx)
    iy = floor(Int, vy)
    iz = floor(Int, vz)

    # Fractional parts
    fx = vx - ix
    fy = vy - iy
    fz = vz - iz

    # Clamp to valid range
    ix0 = clamp(ix + 1, 1, nx)
    ix1 = clamp(ix + 2, 1, nx)
    iy0 = clamp(iy + 1, 1, ny)
    iy1 = clamp(iy + 2, 1, ny)
    iz0 = clamp(iz + 1, 1, nz)
    iz1 = clamp(iz + 2, 1, nz)

    # Trilinear interpolation
    c000 = volume[ix0, iy0, iz0]
    c100 = volume[ix1, iy0, iz0]
    c010 = volume[ix0, iy1, iz0]
    c110 = volume[ix1, iy1, iz0]
    c001 = volume[ix0, iy0, iz1]
    c101 = volume[ix1, iy0, iz1]
    c011 = volume[ix0, iy1, iz1]
    c111 = volume[ix1, iy1, iz1]

    # Interpolate along x
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    # Interpolate along y
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    # Interpolate along z
    return Float32(c0 * (1 - fz) + c1 * fz)
end

# =============================================================================
# Exports
# =============================================================================

export forward_project, forward_project!
