"""
    Forward/Projector.jl

Ray-driven forward projector for cone-beam CT using Siddon's exact voxel traversal.

Siddon's algorithm computes exact intersection points where each ray crosses voxel
boundaries, then samples at the midpoint of each intersection segment. This provides
accurate line integrals with proper path-length weighting.

Reference: Siddon, R.L. (1985). "Fast calculation of the exact radiological path
for a three-dimensional CT array." Medical Physics, 12(2), 252-255.

Architecture for Reactant/XLA compatibility:
- Pre-computed ray geometry and sample indices (not traced)
- Only volume sampling and accumulation is traced
- No scalar array indexing in traced code
"""

# =============================================================================
# Pre-computed Projection Geometry
# =============================================================================

"""
    ProjectionGeometry

Pre-computed geometry for forward projection using Siddon's method.
All indices and weights are computed once and reused for multiple projections.
"""
struct ProjectionGeometry
    # Linear indices into flattened volume: [n_cols, n_rows, n_angles, n_samples]
    linear_indices::Array{Int, 4}

    # Per-sample weights (path lengths in cm): [n_cols, n_rows, n_angles, n_samples]
    sample_weights::Array{Float64, 4}

    # Volume dimensions
    nx::Int
    ny::Int
    nz::Int
end

# =============================================================================
# Siddon's Exact Voxel Traversal
# =============================================================================

"""
    precompute_projection_geometry(geom, fov, voxel_size, volume_size)

Pre-compute projection geometry using Siddon's exact voxel traversal algorithm.

# Arguments
- `geom::CTGeometry`: Scanner geometry
- `fov::NTuple{3,Float64}`: Field of view (cm)
- `voxel_size::NTuple{3,Float64}`: Voxel dimensions (cm)
- `volume_size::NTuple{3,Int}`: Volume dimensions (voxels)

# Returns
`ProjectionGeometry` with pre-computed indices and path-length weights.
"""
function precompute_projection_geometry(
    geom::CTGeometry,
    fov::NTuple{3,Float64},
    voxel_size::NTuple{3,Float64},
    volume_size::NTuple{3,Int}
)
    nx, ny, nz = volume_size
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles

    # Maximum samples per ray (worst case: diagonal through volume)
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
    linear_indices = ones(Int, n_cols, n_rows, n_angles, max_samples)
    sample_weights = zeros(Float64, n_cols, n_rows, n_angles, max_samples)

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
                indices, weights = compute_siddon_ray(
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

    return ProjectionGeometry(linear_indices, sample_weights, nx, ny, nz)
end

"""
    compute_siddon_ray(sx, sy, sz, dx, dy, dz, ...) -> (indices, weights)

Compute exact voxel intersections for a single ray using Siddon's algorithm.

Returns linear indices and path lengths (cm) for each traversed voxel.
"""
function compute_siddon_ray(
    sx, sy, sz,           # Source position
    dx, dy, dz,           # Ray direction (not normalized)
    x_planes, y_planes, z_planes,
    x_min, y_min, z_min,
    voxel_size,
    nx, ny, nz
)
    eps = 1e-10
    ray_length = sqrt(dx^2 + dy^2 + dz^2)

    # Compute parametric intersections with all planes
    alphas = Float64[]

    if abs(dx) > eps
        for xp in x_planes
            α = (xp - sx) / dx
            if 0 < α < 1
                push!(alphas, α)
            end
        end
    end

    if abs(dy) > eps
        for yp in y_planes
            α = (yp - sy) / dy
            if 0 < α < 1
                push!(alphas, α)
            end
        end
    end

    if abs(dz) > eps
        for zp in z_planes
            α = (zp - sz) / dz
            if 0 < α < 1
                push!(alphas, α)
            end
        end
    end

    # Compute ray entry/exit bounds
    α_min, α_max = compute_alpha_bounds(
        sx, sy, sz, dx, dy, dz,
        x_planes[1], x_planes[end],
        y_planes[1], y_planes[end],
        z_planes[1], z_planes[end]
    )

    if α_max <= α_min
        return Int[], Float64[]  # Ray misses volume
    end

    # Filter and add bounds
    alphas = filter(α -> α_min <= α <= α_max, alphas)
    push!(alphas, α_min)
    push!(alphas, α_max)
    sort!(alphas)
    unique!(alphas)

    n_segments = length(alphas) - 1
    if n_segments <= 0
        return Int[], Float64[]
    end

    indices = Int[]
    weights = Float64[]

    for i in 1:n_segments
        α1, α2 = alphas[i], alphas[i+1]

        # Midpoint of segment
        α_mid = (α1 + α2) / 2
        x_mid = sx + α_mid * dx
        y_mid = sy + α_mid * dy
        z_mid = sz + α_mid * dz

        # Convert to voxel indices
        ix = floor(Int, (x_mid - x_min) / voxel_size[1]) + 1
        iy = floor(Int, (y_mid - y_min) / voxel_size[2]) + 1
        iz = floor(Int, (z_mid - z_min) / voxel_size[3]) + 1

        if 1 <= ix <= nx && 1 <= iy <= ny && 1 <= iz <= nz
            lin_idx = ix + (iy - 1) * nx + (iz - 1) * nx * ny
            path_length = (α2 - α1) * ray_length

            push!(indices, lin_idx)
            push!(weights, path_length)
        end
    end

    return indices, weights
end

"""
    compute_alpha_bounds(sx, sy, sz, dx, dy, dz, ...) -> (α_min, α_max)

Compute parametric bounds where ray enters and exits the volume.
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
        α_x_min, α_x_max = minmax(α1, α2)
    elseif sx < x_min || sx > x_max
        return 1.0, 0.0
    end

    if abs(dy) > eps
        α1 = (y_min - sy) / dy
        α2 = (y_max - sy) / dy
        α_y_min, α_y_max = minmax(α1, α2)
    elseif sy < y_min || sy > y_max
        return 1.0, 0.0
    end

    if abs(dz) > eps
        α1 = (z_min - sz) / dz
        α2 = (z_max - sz) / dz
        α_z_min, α_z_max = minmax(α1, α2)
    elseif sz < z_min || sz > z_max
        return 1.0, 0.0
    end

    α_min = max(α_x_min, α_y_min, α_z_min, 0.0)
    α_max = min(α_x_max, α_y_max, α_z_max, 1.0)

    return α_min, α_max
end

# =============================================================================
# Forward Projection (XLA-compatible)
# =============================================================================

"""
    project_volume(volume, proj_geom::ProjectionGeometry)

Project a volume using pre-computed Siddon geometry.
This is the XLA-compilable core - only uses pre-computed indices and weights.
"""
function project_volume(volume, proj_geom::ProjectionGeometry)
    T = eltype(volume)

    # Flatten volume for linear indexing
    volume_flat = vec(volume)

    # Gather samples using pre-computed indices
    samples = volume_flat[proj_geom.linear_indices]

    # Multiply by path-length weights and sum
    weights_T = T.(proj_geom.sample_weights)

    # Weighted sum along sample dimension
    sinogram = dropdims(sum(samples .* weights_T, dims=4), dims=4)

    return sinogram
end

"""
    forward_project(phantom::Phantom, geom::CTGeometry)

Compute cone-beam projections using Siddon's exact voxel traversal.

# Arguments
- `phantom::Phantom`: Input phantom with μ values
- `geom::CTGeometry`: Scanner geometry

# Returns
Sinogram array of shape [n_cols, n_rows, n_angles]
"""
function forward_project(phantom::Phantom, geom::CTGeometry)
    proj_geom = precompute_projection_geometry(
        geom, phantom.fov, phantom.voxel_size, size(phantom.μ)
    )
    return project_volume(phantom.μ, proj_geom)
end

# =============================================================================
# Exports
# =============================================================================

export forward_project
export ProjectionGeometry, precompute_projection_geometry, project_volume
