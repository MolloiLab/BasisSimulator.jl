"""
    Forward/Projector.jl

Ray-driven forward projector for cone-beam CT.

Fully vectorized for Reactant/XLA compatibility:
- Pre-computed ray geometry and sample indices (not traced)
- Only volume sampling and accumulation is traced
- No scalar array indexing in traced code
"""

# =============================================================================
# Pre-computed Projection Geometry
# =============================================================================

"""
    ProjectionGeometry

Pre-computed geometry for forward projection.
All indices are computed once and reused for multiple projections.
"""
struct ProjectionGeometry
    # Sample positions along each ray: [n_cols, n_rows, n_angles, n_steps, 3]
    sample_positions::Array{Float64, 5}

    # Linear indices into flattened volume: [n_cols, n_rows, n_angles, n_steps]
    linear_indices::Array{Int, 4}

    # Step sizes for integration: [n_cols, n_rows, n_angles]
    step_sizes::Array{Float64, 3}

    # Volume dimensions
    nx::Int
    ny::Int
    nz::Int
end

"""
    precompute_projection_geometry(geom::CTGeometry, fov, voxel_size, volume_size, n_steps)

Pre-compute all ray geometry and volume indices for forward projection.
This is done once and reused for multiple projections with the same geometry.
"""
function precompute_projection_geometry(
    geom::CTGeometry,
    fov::NTuple{3,Float64},
    voxel_size::NTuple{3,Float64},
    volume_size::NTuple{3,Int},
    n_steps::Int
)
    nx, ny, nz = volume_size
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles

    # Volume bounds
    x_min, x_max = -fov[1]/2, fov[1]/2
    y_min, y_max = -fov[2]/2, fov[2]/2
    z_min, z_max = -fov[3]/2, fov[3]/2

    # Detector pixel size at detector plane
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    # Pre-allocate arrays
    sample_positions = zeros(Float64, n_cols, n_rows, n_angles, n_steps, 3)
    step_sizes = zeros(Float64, n_cols, n_rows, n_angles)

    # Step fractions along ray
    step_fractions = collect((0:n_steps-1) .+ 0.5) ./ n_steps

    for angle_idx in 1:n_angles
        # Source position
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

        for row in 1:n_rows
            for col in 1:n_cols
                # Detector pixel position
                u_offset = (col - (n_cols + 1) / 2) * pixel_size_det
                v_offset = (row - (n_rows + 1) / 2) * pixel_size_det

                dx = dcx + u_offset * ux + v_offset * vx
                dy = dcy + u_offset * uy + v_offset * vy
                dz = dcz + u_offset * uz + v_offset * vz

                # Ray direction (source to detector)
                ray_dx = dx - sx
                ray_dy = dy - sy
                ray_dz = dz - sz
                ray_len = sqrt(ray_dx^2 + ray_dy^2 + ray_dz^2)

                # Normalize
                ray_dx /= ray_len
                ray_dy /= ray_len
                ray_dz /= ray_len

                # Compute ray-box intersection
                t_min, t_max = ray_box_intersection(
                    sx, sy, sz, ray_dx, ray_dy, ray_dz,
                    x_min, x_max, y_min, y_max, z_min, z_max
                )

                ray_length = max(t_max - t_min, 0.0)
                step_sizes[col, row, angle_idx] = ray_length / n_steps

                # Compute sample positions along ray
                for s in 1:n_steps
                    t = t_min + step_fractions[s] * ray_length

                    sample_positions[col, row, angle_idx, s, 1] = sx + t * ray_dx
                    sample_positions[col, row, angle_idx, s, 2] = sy + t * ray_dy
                    sample_positions[col, row, angle_idx, s, 3] = sz + t * ray_dz
                end
            end
        end
    end

    # Convert sample positions to linear indices
    linear_indices = compute_linear_indices(
        sample_positions, fov, voxel_size, nx, ny, nz
    )

    return ProjectionGeometry(
        sample_positions,
        linear_indices,
        step_sizes,
        nx, ny, nz
    )
end

"""
    ray_box_intersection(ox, oy, oz, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max)

Compute ray-box intersection using slab method.
"""
function ray_box_intersection(
    ox, oy, oz, dx, dy, dz,
    x_min, x_max, y_min, y_max, z_min, z_max
)
    eps = 1e-10

    inv_dx = 1.0 / (abs(dx) > eps ? dx : eps * sign(dx + eps))
    inv_dy = 1.0 / (abs(dy) > eps ? dy : eps * sign(dy + eps))
    inv_dz = 1.0 / (abs(dz) > eps ? dz : eps * sign(dz + eps))

    t1_x = (x_min - ox) * inv_dx
    t2_x = (x_max - ox) * inv_dx
    t1_y = (y_min - oy) * inv_dy
    t2_y = (y_max - oy) * inv_dy
    t1_z = (z_min - oz) * inv_dz
    t2_z = (z_max - oz) * inv_dz

    t_x_min = min(t1_x, t2_x)
    t_x_max = max(t1_x, t2_x)
    t_y_min = min(t1_y, t2_y)
    t_y_max = max(t1_y, t2_y)
    t_z_min = min(t1_z, t2_z)
    t_z_max = max(t1_z, t2_z)

    t_min = max(t_x_min, t_y_min, t_z_min, 0.0)
    t_max = min(t_x_max, t_y_max, t_z_max)

    return t_min, t_max
end

"""
    compute_linear_indices(positions, fov, voxel_size, nx, ny, nz)

Convert world positions to linear indices into flattened volume.
"""
function compute_linear_indices(
    positions::Array{Float64, 5},
    fov::NTuple{3,Float64},
    voxel_size::NTuple{3,Float64},
    nx, ny, nz
)
    n_cols, n_rows, n_angles, n_steps, _ = size(positions)

    # World origin
    x_origin = -fov[1] / 2
    y_origin = -fov[2] / 2
    z_origin = -fov[3] / 2

    linear_indices = zeros(Int, n_cols, n_rows, n_angles, n_steps)

    for s in 1:n_steps
        for angle_idx in 1:n_angles
            for row in 1:n_rows
                for col in 1:n_cols
                    x = positions[col, row, angle_idx, s, 1]
                    y = positions[col, row, angle_idx, s, 2]
                    z = positions[col, row, angle_idx, s, 3]

                    # Convert to voxel indices
                    vx = (x - x_origin) / voxel_size[1]
                    vy = (y - y_origin) / voxel_size[2]
                    vz = (z - z_origin) / voxel_size[3]

                    ix = clamp(floor(Int, vx) + 1, 1, nx)
                    iy = clamp(floor(Int, vy) + 1, 1, ny)
                    iz = clamp(floor(Int, vz) + 1, 1, nz)

                    # Linear index (column-major)
                    linear_indices[col, row, angle_idx, s] = ix + (iy - 1) * nx + (iz - 1) * nx * ny
                end
            end
        end
    end

    return linear_indices
end

# =============================================================================
# Forward Projection (XLA-compatible)
# =============================================================================

"""
    forward_project(phantom::Phantom, geom::CTGeometry; n_steps::Int=256)

Compute cone-beam projections (sinogram) from a phantom volume.
"""
function forward_project(phantom::Phantom, geom::CTGeometry; n_steps::Int=256)
    # Pre-compute geometry (not traced)
    proj_geom = precompute_projection_geometry(
        geom, phantom.fov, phantom.voxel_size, size(phantom.μ), n_steps
    )

    # Project volume (this part can be traced)
    sinogram = project_volume(phantom.μ, proj_geom)

    return sinogram
end

"""
    project_volume(volume, proj_geom::ProjectionGeometry)

Project a volume using pre-computed geometry.
This is the XLA-compilable core - only uses pre-computed indices.
"""
function project_volume(volume, proj_geom::ProjectionGeometry)
    T = eltype(volume)

    # Flatten volume for linear indexing
    volume_flat = vec(volume)

    # Gather samples using pre-computed indices
    # linear_indices: [n_cols, n_rows, n_angles, n_steps]
    samples = volume_flat[proj_geom.linear_indices]

    # Integrate: multiply by step size and sum along ray dimension
    # step_sizes: [n_cols, n_rows, n_angles]
    n_cols, n_rows, n_angles, n_steps = size(proj_geom.linear_indices)
    step_sizes_T = T.(proj_geom.step_sizes)
    step_sizes_expanded = reshape(step_sizes_T, n_cols, n_rows, n_angles, 1)

    # Sum along step dimension
    sinogram = dropdims(sum(samples .* step_sizes_expanded, dims=4), dims=4)

    return sinogram
end

"""
    forward_project!(sinogram, volume, voxel_size, fov, geom, n_steps)

In-place forward projection for backward compatibility.
"""
function forward_project!(
    sinogram,
    volume,
    voxel_size::NTuple{3,Float64},
    fov::NTuple{3,Float64},
    geom::CTGeometry,
    n_steps::Int
)
    # Pre-compute geometry
    proj_geom = precompute_projection_geometry(
        geom, fov, voxel_size, size(volume), n_steps
    )

    # Project and store in-place
    result = project_volume(volume, proj_geom)
    sinogram .= result
    return sinogram
end

# =============================================================================
# Exports
# =============================================================================

export forward_project, forward_project!
export ProjectionGeometry, precompute_projection_geometry, project_volume
