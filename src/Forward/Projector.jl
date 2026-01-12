"""
    Forward/Projector.jl

Ray-driven forward projector for cone-beam CT.

Supports two sampling methods:
1. Uniform sampling: Fixed number of evenly-spaced samples along each ray
2. Siddon's method: Exact voxel boundary intersections with path-length weighting

Both methods are fully vectorized for Reactant/XLA compatibility:
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
All indices and weights are computed once and reused for multiple projections.

Supports both uniform sampling and Siddon's exact method through the same interface.
"""
struct ProjectionGeometry
    # Linear indices into flattened volume: [n_cols, n_rows, n_angles, n_samples]
    linear_indices::Array{Int, 4}

    # Per-sample weights (path lengths): [n_cols, n_rows, n_angles, n_samples]
    # For uniform sampling: all equal to step_size
    # For Siddon: intersection lengths through each voxel
    sample_weights::Array{Float64, 4}

    # Volume dimensions
    nx::Int
    ny::Int
    nz::Int

    # Sampling method used
    method::Symbol  # :uniform or :siddon
end

# =============================================================================
# Siddon's Exact Voxel Traversal
# =============================================================================

"""
    precompute_siddon_geometry(geom::CTGeometry, fov, voxel_size, volume_size)

Pre-compute projection geometry using Siddon's exact voxel traversal algorithm.

Siddon's method computes exact intersection points where each ray crosses voxel
boundaries, then samples at the midpoint of each intersection segment. This is
more accurate than uniform sampling, especially for coarse volumes.

Reference: Siddon, R.L. (1985). "Fast calculation of the exact radiological path
for a three-dimensional CT array." Medical Physics, 12(2), 252-255.
"""
function precompute_siddon_geometry(
    geom::CTGeometry,
    fov::NTuple{3,Float64},
    voxel_size::NTuple{3,Float64},
    volume_size::NTuple{3,Int}
)
    nx, ny, nz = volume_size
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles

    # Maximum number of intersections per ray (worst case: diagonal through volume)
    # Each ray can cross at most (nx+1) + (ny+1) + (nz+1) planes
    # This gives at most nx + ny + nz voxel segments
    max_samples = nx + ny + nz

    # Volume bounds
    x_min, x_max = -fov[1]/2, fov[1]/2
    y_min, y_max = -fov[2]/2, fov[2]/2
    z_min, z_max = -fov[3]/2, fov[3]/2

    # Voxel boundary planes
    x_planes = collect(range(x_min, x_max, length=nx+1))
    y_planes = collect(range(y_min, y_max, length=ny+1))
    z_planes = collect(range(z_min, z_max, length=nz+1))

    # Detector pixel size at detector plane
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    # Pre-allocate output arrays (padded to max_samples)
    linear_indices = ones(Int, n_cols, n_rows, n_angles, max_samples)  # Default to index 1
    sample_weights = zeros(Float64, n_cols, n_rows, n_angles, max_samples)  # Default to 0 weight

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

                # Ray direction (source to detector, not normalized)
                ray_dx = dx - sx
                ray_dy = dy - sy
                ray_dz = dz - sz

                # Compute Siddon intersections
                alphas, indices, weights = compute_siddon_intersections(
                    sx, sy, sz, ray_dx, ray_dy, ray_dz,
                    x_planes, y_planes, z_planes,
                    x_min, y_min, z_min,
                    voxel_size, nx, ny, nz
                )

                # Store in padded arrays
                n_valid = length(indices)
                for s in 1:min(n_valid, max_samples)
                    linear_indices[col, row, angle_idx, s] = indices[s]
                    sample_weights[col, row, angle_idx, s] = weights[s]
                end
            end
        end
    end

    return ProjectionGeometry(
        linear_indices,
        sample_weights,
        nx, ny, nz,
        :siddon
    )
end

"""
    compute_siddon_intersections(sx, sy, sz, dx, dy, dz, x_planes, y_planes, z_planes, ...)

Compute exact voxel intersections for a single ray using Siddon's algorithm.

Returns:
- alphas: Sorted parametric intersection values
- indices: Linear indices of traversed voxels
- weights: Path length through each voxel (in cm)
"""
function compute_siddon_intersections(
    sx, sy, sz,           # Source position
    dx, dy, dz,           # Ray direction (not normalized)
    x_planes, y_planes, z_planes,  # Voxel boundary planes
    x_min, y_min, z_min,  # Volume origin
    voxel_size,           # Voxel dimensions
    nx, ny, nz            # Volume size
)
    eps = 1e-10
    ray_length = sqrt(dx^2 + dy^2 + dz^2)

    # Compute parametric intersections with all planes
    # α such that: source + α * direction = plane
    alphas = Float64[]

    # X-planes
    if abs(dx) > eps
        for xp in x_planes
            α = (xp - sx) / dx
            if 0 < α < 1  # Only within ray segment
                push!(alphas, α)
            end
        end
    end

    # Y-planes
    if abs(dy) > eps
        for yp in y_planes
            α = (yp - sy) / dy
            if 0 < α < 1
                push!(alphas, α)
            end
        end
    end

    # Z-planes
    if abs(dz) > eps
        for zp in z_planes
            α = (zp - sz) / dz
            if 0 < α < 1
                push!(alphas, α)
            end
        end
    end

    # Add ray entry and exit points
    α_min, α_max = compute_alpha_bounds(sx, sy, sz, dx, dy, dz,
        x_planes[1], x_planes[end], y_planes[1], y_planes[end],
        z_planes[1], z_planes[end])

    if α_max <= α_min
        # Ray misses volume
        return Float64[], Int[], Float64[]
    end

    # Filter alphas to valid range and add bounds
    alphas = filter(α -> α_min <= α <= α_max, alphas)
    push!(alphas, α_min)
    push!(alphas, α_max)

    # Sort and remove duplicates
    sort!(alphas)
    unique!(alphas)

    # Compute midpoints and path lengths
    n_segments = length(alphas) - 1
    if n_segments <= 0
        return Float64[], Int[], Float64[]
    end

    indices = Int[]
    weights = Float64[]

    for i in 1:n_segments
        α1 = alphas[i]
        α2 = alphas[i+1]

        # Midpoint of segment
        α_mid = (α1 + α2) / 2
        x_mid = sx + α_mid * dx
        y_mid = sy + α_mid * dy
        z_mid = sz + α_mid * dz

        # Convert to voxel indices
        ix = floor(Int, (x_mid - x_min) / voxel_size[1]) + 1
        iy = floor(Int, (y_mid - y_min) / voxel_size[2]) + 1
        iz = floor(Int, (z_mid - z_min) / voxel_size[3]) + 1

        # Check bounds
        if 1 <= ix <= nx && 1 <= iy <= ny && 1 <= iz <= nz
            # Linear index (column-major)
            lin_idx = ix + (iy - 1) * nx + (iz - 1) * nx * ny

            # Path length through this segment (in world units = cm)
            path_length = (α2 - α1) * ray_length

            push!(indices, lin_idx)
            push!(weights, path_length)
        end
    end

    return alphas, indices, weights
end

"""
    compute_alpha_bounds(sx, sy, sz, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max)

Compute the parametric bounds where ray enters and exits the volume.
"""
function compute_alpha_bounds(
    sx, sy, sz, dx, dy, dz,
    x_min, x_max, y_min, y_max, z_min, z_max
)
    eps = 1e-10

    α_x_min, α_x_max = -Inf, Inf
    α_y_min, α_y_max = -Inf, Inf
    α_z_min, α_z_max = -Inf, Inf

    if abs(dx) > eps
        α1 = (x_min - sx) / dx
        α2 = (x_max - sx) / dx
        α_x_min = min(α1, α2)
        α_x_max = max(α1, α2)
    elseif sx < x_min || sx > x_max
        return 1.0, 0.0  # Ray parallel to X and outside
    end

    if abs(dy) > eps
        α1 = (y_min - sy) / dy
        α2 = (y_max - sy) / dy
        α_y_min = min(α1, α2)
        α_y_max = max(α1, α2)
    elseif sy < y_min || sy > y_max
        return 1.0, 0.0  # Ray parallel to Y and outside
    end

    if abs(dz) > eps
        α1 = (z_min - sz) / dz
        α2 = (z_max - sz) / dz
        α_z_min = min(α1, α2)
        α_z_max = max(α1, α2)
    elseif sz < z_min || sz > z_max
        return 1.0, 0.0  # Ray parallel to Z and outside
    end

    α_min = max(α_x_min, α_y_min, α_z_min, 0.0)
    α_max = min(α_x_max, α_y_max, α_z_max, 1.0)

    return α_min, α_max
end

# =============================================================================
# Uniform Sampling (Original Method)
# =============================================================================

"""
    precompute_uniform_geometry(geom::CTGeometry, fov, voxel_size, volume_size, n_steps)

Pre-compute projection geometry using uniform sampling along rays.

This is simpler and faster to pre-compute than Siddon, but less accurate
for coarse volumes. Use `n_steps` ≥ 2 * max(nx, ny, nz) for good accuracy.
"""
function precompute_uniform_geometry(
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
    linear_indices = ones(Int, n_cols, n_rows, n_angles, n_steps)
    sample_weights = zeros(Float64, n_cols, n_rows, n_angles, n_steps)

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
                step_size = ray_length / n_steps

                # Compute sample positions and weights
                for s in 1:n_steps
                    t = t_min + step_fractions[s] * ray_length

                    x = sx + t * ray_dx
                    y = sy + t * ray_dy
                    z = sz + t * ray_dz

                    # Convert to voxel indices
                    vx_idx = (x - (-fov[1]/2)) / voxel_size[1]
                    vy_idx = (y - (-fov[2]/2)) / voxel_size[2]
                    vz_idx = (z - (-fov[3]/2)) / voxel_size[3]

                    ix = clamp(floor(Int, vx_idx) + 1, 1, nx)
                    iy = clamp(floor(Int, vy_idx) + 1, 1, ny)
                    iz = clamp(floor(Int, vz_idx) + 1, 1, nz)

                    linear_indices[col, row, angle_idx, s] = ix + (iy - 1) * nx + (iz - 1) * nx * ny
                    sample_weights[col, row, angle_idx, s] = step_size
                end
            end
        end
    end

    return ProjectionGeometry(
        linear_indices,
        sample_weights,
        nx, ny, nz,
        :uniform
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

# =============================================================================
# Unified Pre-computation Interface
# =============================================================================

"""
    precompute_projection_geometry(geom, fov, voxel_size, volume_size; method=:siddon, n_steps=nothing)

Pre-compute projection geometry for forward projection.

# Arguments
- `geom::CTGeometry`: Scanner geometry
- `fov::NTuple{3,Float64}`: Field of view (cm)
- `voxel_size::NTuple{3,Float64}`: Voxel dimensions (cm)
- `volume_size::NTuple{3,Int}`: Volume dimensions (voxels)
- `method::Symbol`: Sampling method, `:siddon` (default) or `:uniform`
- `n_steps::Union{Int,Nothing}`: Number of steps for uniform sampling (ignored for Siddon)

# Returns
`ProjectionGeometry` with pre-computed indices and weights.
"""
function precompute_projection_geometry(
    geom::CTGeometry,
    fov::NTuple{3,Float64},
    voxel_size::NTuple{3,Float64},
    volume_size::NTuple{3,Int};
    method::Symbol=:siddon,
    n_steps::Union{Int,Nothing}=nothing
)
    if method == :siddon
        return precompute_siddon_geometry(geom, fov, voxel_size, volume_size)
    elseif method == :uniform
        if n_steps === nothing
            # Default: 2x the maximum dimension for good accuracy
            n_steps = 2 * maximum(volume_size)
        end
        return precompute_uniform_geometry(geom, fov, voxel_size, volume_size, n_steps)
    else
        error("Unknown projection method: $method. Use :siddon or :uniform")
    end
end

# Backward-compatible signature (positional n_steps implies uniform)
function precompute_projection_geometry(
    geom::CTGeometry,
    fov::NTuple{3,Float64},
    voxel_size::NTuple{3,Float64},
    volume_size::NTuple{3,Int},
    n_steps::Int
)
    return precompute_uniform_geometry(geom, fov, voxel_size, volume_size, n_steps)
end

# =============================================================================
# Forward Projection (XLA-compatible)
# =============================================================================

"""
    project_volume(volume, proj_geom::ProjectionGeometry)

Project a volume using pre-computed geometry.
This is the XLA-compilable core - only uses pre-computed indices and weights.

Works identically for both Siddon and uniform sampling methods.
"""
function project_volume(volume, proj_geom::ProjectionGeometry)
    T = eltype(volume)

    # Flatten volume for linear indexing
    volume_flat = vec(volume)

    # Gather samples using pre-computed indices
    # linear_indices: [n_cols, n_rows, n_angles, n_samples]
    samples = volume_flat[proj_geom.linear_indices]

    # Multiply by pre-computed weights and sum
    # sample_weights: [n_cols, n_rows, n_angles, n_samples]
    weights_T = T.(proj_geom.sample_weights)

    # Weighted sum along sample dimension
    sinogram = dropdims(sum(samples .* weights_T, dims=4), dims=4)

    return sinogram
end

"""
    forward_project(phantom::Phantom, geom::CTGeometry; method=:siddon, n_steps=nothing)

Compute cone-beam projections (sinogram) from a phantom volume.

# Arguments
- `phantom::Phantom`: Input phantom with μ values
- `geom::CTGeometry`: Scanner geometry
- `method::Symbol`: Sampling method, `:siddon` (default) or `:uniform`
- `n_steps::Union{Int,Nothing}`: Number of steps for uniform sampling

# Returns
Sinogram array of shape [n_cols, n_rows, n_angles]
"""
function forward_project(
    phantom::Phantom,
    geom::CTGeometry;
    method::Symbol=:siddon,
    n_steps::Union{Int,Nothing}=nothing
)
    # Pre-compute geometry (not traced)
    proj_geom = precompute_projection_geometry(
        geom, phantom.fov, phantom.voxel_size, size(phantom.μ);
        method=method, n_steps=n_steps
    )

    # Project volume (this part can be traced)
    sinogram = project_volume(phantom.μ, proj_geom)

    return sinogram
end

# Backward-compatible signature
function forward_project(phantom::Phantom, geom::CTGeometry, n_steps::Int)
    return forward_project(phantom, geom; method=:uniform, n_steps=n_steps)
end

"""
    forward_project!(sinogram, volume, voxel_size, fov, geom; method=:siddon, n_steps=nothing)

In-place forward projection.
"""
function forward_project!(
    sinogram,
    volume,
    voxel_size::NTuple{3,Float64},
    fov::NTuple{3,Float64},
    geom::CTGeometry;
    method::Symbol=:siddon,
    n_steps::Union{Int,Nothing}=nothing
)
    # Pre-compute geometry
    proj_geom = precompute_projection_geometry(
        geom, fov, voxel_size, size(volume);
        method=method, n_steps=n_steps
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
export precompute_siddon_geometry, precompute_uniform_geometry
