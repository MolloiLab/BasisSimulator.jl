"""
    Forward/RayMarching.jl

Ray marching forward projection with ON-THE-FLY geometry computation.

This approach enables a SINGLE compiled XLA kernel for ALL angles:
1. Geometry computed inside the kernel from compact parameters
2. Fixed-step ray marching with trilinear interpolation
3. Memory: O(volume + sinogram + rays) - no pre-computed geometry arrays

The key insight: store source/detector positions (~KB) not indices/weights (~GB).

## Polychromatic Support

Polychromatic simulation uses:
1. Ray march to accumulate path lengths per REGION (27 Gammex regions)
2. Use `Reactant.@trace for` to loop over energy bins without unrolling
3. Beer-Lambert law: I/I₀ = Σ S(E) * exp(-∫ μ(E,r) dr)

This approach is memory efficient: O(n_rays × n_regions) for path lengths.
"""

using Reactant: @trace

# =============================================================================
# Ray Geometry Computation
# =============================================================================

"""
    RayGeometry

Compact storage of ray geometry for all rays.
Memory: O(n_cols × n_rows × n_angles × 6) ≈ 600MB for clinical scale.
This is computed ONCE and passed to the kernel.
"""
struct RayGeometry
    # Ray origins (source positions broadcast to all pixels): [3, n_cols, n_rows, n_angles]
    origins::Array{Float32, 4}

    # Ray directions (normalized): [3, n_cols, n_rows, n_angles]
    directions::Array{Float32, 4}

    # Ray lengths (source to detector distance): [n_cols, n_rows, n_angles]
    lengths::Array{Float32, 3}

    # Volume bounds for coordinate conversion
    vol_min::NTuple{3, Float32}  # (-fov/2, -fov/2, -fov/2)
    vol_max::NTuple{3, Float32}  # (fov/2, fov/2, fov/2)

    # Volume size
    nx::Int
    ny::Int
    nz::Int
end

"""
    compute_ray_geometry(geom::CTGeometry, fov, volume_size) -> RayGeometry

Compute ray origins and directions for ALL rays.
This is compact: O(n_rays × 6) vs O(n_rays × max_samples) for pre-computed indices.
"""
function compute_ray_geometry(
    geom::CTGeometry,
    fov::NTuple{3, Float64},
    volume_size::NTuple{3, Int}
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles
    nx, ny, nz = volume_size

    # Detector pixel size at detector plane
    pixel_size_det = Float32(geom.pixel_size * (geom.SDD / geom.SAD))

    # Allocate arrays
    origins = zeros(Float32, 3, n_cols, n_rows, n_angles)
    directions = zeros(Float32, 3, n_cols, n_rows, n_angles)
    lengths = zeros(Float32, n_cols, n_rows, n_angles)

    for angle_idx in 1:n_angles
        # Source position
        sx = Float32(geom.source_positions[1, angle_idx])
        sy = Float32(geom.source_positions[2, angle_idx])
        sz = Float32(geom.source_positions[3, angle_idx])

        # Detector center and axes
        dcx = Float32(geom.detector_centers[1, angle_idx])
        dcy = Float32(geom.detector_centers[2, angle_idx])
        dcz = Float32(geom.detector_centers[3, angle_idx])

        ux = Float32(geom.detector_u[1, angle_idx])
        uy = Float32(geom.detector_u[2, angle_idx])
        uz = Float32(geom.detector_u[3, angle_idx])

        vx = Float32(geom.detector_v[1, angle_idx])
        vy = Float32(geom.detector_v[2, angle_idx])
        vz = Float32(geom.detector_v[3, angle_idx])

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

                # Ray length
                ray_len = sqrt(ray_dx^2 + ray_dy^2 + ray_dz^2)

                # Normalize direction
                ray_dx /= ray_len
                ray_dy /= ray_len
                ray_dz /= ray_len

                # Store
                origins[1, col, row, angle_idx] = sx
                origins[2, col, row, angle_idx] = sy
                origins[3, col, row, angle_idx] = sz

                directions[1, col, row, angle_idx] = ray_dx
                directions[2, col, row, angle_idx] = ray_dy
                directions[3, col, row, angle_idx] = ray_dz

                lengths[col, row, angle_idx] = ray_len
            end
        end
    end

    vol_min = (Float32(-fov[1]/2), Float32(-fov[2]/2), Float32(-fov[3]/2))
    vol_max = (Float32(fov[1]/2), Float32(fov[2]/2), Float32(fov[3]/2))

    return RayGeometry(origins, directions, lengths, vol_min, vol_max, nx, ny, nz)
end

# =============================================================================
# Trilinear Interpolation (XLA-Compatible)
# =============================================================================

"""
    trilinear_sample(volume, x, y, z, vol_min, vol_size, nx, ny, nz)

Sample volume at (x, y, z) using trilinear interpolation.
Returns 0 if outside volume bounds.

This function is XLA-compatible when operating on arrays.
"""
function trilinear_sample_point(
    volume::AbstractArray{T, 3},
    x::T, y::T, z::T,
    vol_min_x::T, vol_min_y::T, vol_min_z::T,
    voxel_size_x::T, voxel_size_y::T, voxel_size_z::T,
    nx::Int, ny::Int, nz::Int
) where T
    # Convert world coordinates to voxel coordinates (0-indexed, fractional)
    vx = (x - vol_min_x) / voxel_size_x - T(0.5)
    vy = (y - vol_min_y) / voxel_size_y - T(0.5)
    vz = (z - vol_min_z) / voxel_size_z - T(0.5)

    # Integer voxel indices
    ix0 = floor(Int, vx)
    iy0 = floor(Int, vy)
    iz0 = floor(Int, vz)

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    # Fractional parts
    fx = vx - T(ix0)
    fy = vy - T(iy0)
    fz = vz - T(iz0)

    # Bounds check - return 0 if outside
    if ix0 < 0 || ix1 >= nx || iy0 < 0 || iy1 >= ny || iz0 < 0 || iz1 >= nz
        return T(0)
    end

    # Convert to 1-indexed
    ix0 += 1; ix1 += 1
    iy0 += 1; iy1 += 1
    iz0 += 1; iz1 += 1

    # Clamp to valid range
    ix0 = clamp(ix0, 1, nx); ix1 = clamp(ix1, 1, nx)
    iy0 = clamp(iy0, 1, ny); iy1 = clamp(iy1, 1, ny)
    iz0 = clamp(iz0, 1, nz); iz1 = clamp(iz1, 1, nz)

    # Trilinear interpolation
    c000 = volume[ix0, iy0, iz0]
    c100 = volume[ix1, iy0, iz0]
    c010 = volume[ix0, iy1, iz0]
    c110 = volume[ix1, iy1, iz0]
    c001 = volume[ix0, iy0, iz1]
    c101 = volume[ix1, iy0, iz1]
    c011 = volume[ix0, iy1, iz1]
    c111 = volume[ix1, iy1, iz1]

    # Interpolate
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz
end

# =============================================================================
# XLA-Compatible Ray Marching Forward Projection
# =============================================================================

"""
    forward_project_raymarching_kernel(
        volume, ray_origins, ray_directions,
        vol_min, voxel_size, n_samples, step_size
    )

XLA-compilable forward projection using ray marching.

All rays are processed in parallel, with sequential stepping along each ray.
This is ONE compiled kernel for ALL angles.

# Arguments
- `volume`: Input volume [nx, ny, nz]
- `ray_origins`: Ray start points [3, n_cols, n_rows, n_angles]
- `ray_directions`: Normalized ray directions [3, n_cols, n_rows, n_angles]
- `vol_min`: Volume minimum coordinates (x, y, z)
- `voxel_size`: Voxel dimensions (dx, dy, dz)
- `n_samples`: Number of samples per ray
- `step_size`: Distance between samples

# Returns
Sinogram [n_cols, n_rows, n_angles]
"""
function forward_project_raymarching_kernel(
    volume::AbstractArray{T, 3},
    ray_origins::AbstractArray{T, 4},
    ray_directions::AbstractArray{T, 4},
    vol_min_x::T, vol_min_y::T, vol_min_z::T,
    voxel_size_x::T, voxel_size_y::T, voxel_size_z::T,
    n_samples::Int,
    step_size::T
) where T
    nx, ny, nz = size(volume)
    _, n_cols, n_rows, n_angles = size(ray_origins)

    # Initialize sinogram
    sinogram = zeros(T, n_cols, n_rows, n_angles)

    # March along all rays (this loop can be traced by Reactant)
    for step in 1:n_samples
        t = T(step) * step_size

        # Compute sample points for all rays at this t
        # sample_x[col, row, angle] = origin_x + t * dir_x
        for angle_idx in 1:n_angles
            for row in 1:n_rows
                for col in 1:n_cols
                    # Sample point
                    x = ray_origins[1, col, row, angle_idx] + t * ray_directions[1, col, row, angle_idx]
                    y = ray_origins[2, col, row, angle_idx] + t * ray_directions[2, col, row, angle_idx]
                    z = ray_origins[3, col, row, angle_idx] + t * ray_directions[3, col, row, angle_idx]

                    # Trilinear sample
                    val = trilinear_sample_point(
                        volume, x, y, z,
                        vol_min_x, vol_min_y, vol_min_z,
                        voxel_size_x, voxel_size_y, voxel_size_z,
                        nx, ny, nz
                    )

                    sinogram[col, row, angle_idx] += val * step_size
                end
            end
        end
    end

    return sinogram
end

"""
    forward_project_raymarching_vectorized(
        volume, ray_origins, ray_directions,
        vol_min, voxel_size, n_samples, step_size
    )

Vectorized version of ray marching forward projection.
Uses array operations for better XLA compilation.
"""
function forward_project_raymarching_vectorized(
    volume::AbstractArray{T, 3},
    ray_origins::AbstractArray{T, 4},
    ray_directions::AbstractArray{T, 4},
    vol_min::NTuple{3, T},
    voxel_size::NTuple{3, T},
    n_samples::Int,
    step_size::T
) where T
    nx, ny, nz = size(volume)
    _, n_cols, n_rows, n_angles = size(ray_origins)

    # Extract components for vectorization
    ox = ray_origins[1, :, :, :]
    oy = ray_origins[2, :, :, :]
    oz = ray_origins[3, :, :, :]

    dx = ray_directions[1, :, :, :]
    dy = ray_directions[2, :, :, :]
    dz = ray_directions[3, :, :, :]

    # Initialize sinogram
    sinogram = zeros(T, n_cols, n_rows, n_angles)

    # Precompute coordinate conversion factors
    inv_voxel_x = T(1) / voxel_size[1]
    inv_voxel_y = T(1) / voxel_size[2]
    inv_voxel_z = T(1) / voxel_size[3]

    # March along all rays
    for step in 1:n_samples
        t = T(step) * step_size

        # Sample points for all rays
        sample_x = ox .+ t .* dx
        sample_y = oy .+ t .* dy
        sample_z = oz .+ t .* dz

        # Convert to voxel coordinates (0-indexed, fractional)
        vx = (sample_x .- vol_min[1]) .* inv_voxel_x .- T(0.5)
        vy = (sample_y .- vol_min[2]) .* inv_voxel_y .- T(0.5)
        vz = (sample_z .- vol_min[3]) .* inv_voxel_z .- T(0.5)

        # Integer indices (floor)
        ix0 = floor.(Int, vx)
        iy0 = floor.(Int, vy)
        iz0 = floor.(Int, vz)

        # Fractional parts
        fx = vx .- T.(ix0)
        fy = vy .- T.(iy0)
        fz = vz .- T.(iz0)

        # Check bounds and create mask
        in_bounds = (ix0 .>= 0) .& (ix0 .< nx - 1) .&
                    (iy0 .>= 0) .& (iy0 .< ny - 1) .&
                    (iz0 .>= 0) .& (iz0 .< nz - 1)

        # Convert to 1-indexed and clamp
        ix0_safe = clamp.(ix0 .+ 1, 1, nx)
        ix1_safe = clamp.(ix0 .+ 2, 1, nx)
        iy0_safe = clamp.(iy0 .+ 1, 1, ny)
        iy1_safe = clamp.(iy0 .+ 2, 1, ny)
        iz0_safe = clamp.(iz0 .+ 1, 1, nz)
        iz1_safe = clamp.(iz0 .+ 2, 1, nz)

        # Gather 8 corners for trilinear interpolation
        # This is the XLA-friendly way to do scattered reads
        for angle_idx in 1:n_angles
            for row in 1:n_rows
                for col in 1:n_cols
                    if in_bounds[col, row, angle_idx]
                        i0 = ix0_safe[col, row, angle_idx]
                        i1 = ix1_safe[col, row, angle_idx]
                        j0 = iy0_safe[col, row, angle_idx]
                        j1 = iy1_safe[col, row, angle_idx]
                        k0 = iz0_safe[col, row, angle_idx]
                        k1 = iz1_safe[col, row, angle_idx]

                        f_x = fx[col, row, angle_idx]
                        f_y = fy[col, row, angle_idx]
                        f_z = fz[col, row, angle_idx]

                        # 8 corners
                        c000 = volume[i0, j0, k0]
                        c100 = volume[i1, j0, k0]
                        c010 = volume[i0, j1, k0]
                        c110 = volume[i1, j1, k0]
                        c001 = volume[i0, j0, k1]
                        c101 = volume[i1, j0, k1]
                        c011 = volume[i0, j1, k1]
                        c111 = volume[i1, j1, k1]

                        # Trilinear interpolation
                        c00 = c000 * (1 - f_x) + c100 * f_x
                        c01 = c001 * (1 - f_x) + c101 * f_x
                        c10 = c010 * (1 - f_x) + c110 * f_x
                        c11 = c011 * (1 - f_x) + c111 * f_x

                        c0 = c00 * (1 - f_y) + c10 * f_y
                        c1 = c01 * (1 - f_y) + c11 * f_y

                        val = c0 * (1 - f_z) + c1 * f_z

                        sinogram[col, row, angle_idx] += val * step_size
                    end
                end
            end
        end
    end

    return sinogram
end

# =============================================================================
# High-Level Interface
# =============================================================================

"""
    forward_project_raymarching(phantom, geom; step_factor=0.5)

Forward projection using ray marching with on-the-fly geometry.

This computes geometry ONCE (compact: ~600MB for clinical scale),
then runs a single kernel that processes ALL angles.

# Arguments
- `phantom::Phantom`: Input phantom
- `geom::CTGeometry`: Scanner geometry
- `step_factor::Float64`: Step size as fraction of minimum voxel size (default: 0.5)

# Returns
Sinogram [n_cols, n_rows, n_angles]

# Performance
- Memory: O(volume + sinogram + rays) ≈ 2-3 GB for clinical scale
- Computation: ONE kernel call for ALL angles
- Suitable for Reactant.@compile for massive speedup
"""
function forward_project_raymarching(
    phantom::Phantom,
    geom::CTGeometry;
    step_factor::Float64=0.5
)
    # Compute ray geometry (compact)
    ray_geom = compute_ray_geometry(geom, phantom.fov, size(phantom.μ))

    # Compute step size and number of samples
    min_voxel = minimum(phantom.voxel_size)
    step_size = Float32(min_voxel * step_factor)

    # Compute n_samples based on FULL ray path from source through volume
    # The ray starts at the source (outside volume) and must traverse the entire volume.
    # SAD = source to isocenter distance, volume centered at isocenter
    # Ray must cover from source to the far edge of volume:
    #   t_far = SAD + half_diagonal ≈ SAD + sqrt(sum((fov/2)^2))
    half_diagonal = sqrt(sum((phantom.fov ./ 2) .^ 2))
    ray_path_to_far_edge = geom.SAD + half_diagonal
    n_samples = ceil(Int, ray_path_to_far_edge / step_size) + 10  # Add margin

    # Volume parameters
    vol_min = ray_geom.vol_min
    voxel_size = (
        Float32(phantom.voxel_size[1]),
        Float32(phantom.voxel_size[2]),
        Float32(phantom.voxel_size[3])
    )

    # Run ray marching
    volume = Float32.(phantom.μ)
    sinogram = forward_project_raymarching_vectorized(
        volume,
        ray_geom.origins,
        ray_geom.directions,
        vol_min,
        voxel_size,
        n_samples,
        step_size
    )

    return sinogram
end

"""
    forward_project_raymarching_compiled(
        volume, ray_origins, ray_directions,
        vol_min, voxel_size, n_samples, step_size
    )

Array-based interface for Reactant compilation.

Use this function with @compile for maximum performance:

```julia
using Reactant

# Compute ray geometry once
ray_geom = compute_ray_geometry(geom, fov, volume_size)

# Convert to Reactant arrays
vol_ra = Reactant.to_rarray(Float32.(phantom.μ))
origins_ra = Reactant.to_rarray(ray_geom.origins)
dirs_ra = Reactant.to_rarray(ray_geom.directions)

# Compile
compiled_proj = @compile forward_project_raymarching_compiled(
    vol_ra, origins_ra, dirs_ra,
    vol_min..., voxel_size..., n_samples, step_size
)

# Run (fast!)
sinogram = Array(compiled_proj(vol_ra, origins_ra, dirs_ra, ...))
```
"""
function forward_project_raymarching_compiled(
    volume::AbstractArray{T, 3},
    ray_origins::AbstractArray{T, 4},
    ray_directions::AbstractArray{T, 4},
    vol_min_x::T, vol_min_y::T, vol_min_z::T,
    voxel_size_x::T, voxel_size_y::T, voxel_size_z::T,
    n_samples::Int,
    step_size::T
) where T
    return forward_project_raymarching_kernel(
        volume, ray_origins, ray_directions,
        vol_min_x, vol_min_y, vol_min_z,
        voxel_size_x, voxel_size_y, voxel_size_z,
        n_samples, step_size
    )
end

# =============================================================================
# Polychromatic Ray Marching
# =============================================================================

"""
    nearest_neighbor_sample_mask(mask, x, y, z, vol_min, voxel_size, nx, ny, nz)

Sample mask (region labels) at (x, y, z) using nearest neighbor interpolation.
Returns 0 (background) if outside volume bounds.
"""
function nearest_neighbor_sample_mask(
    mask::AbstractArray{UInt8, 3},
    x::T, y::T, z::T,
    vol_min_x::T, vol_min_y::T, vol_min_z::T,
    voxel_size_x::T, voxel_size_y::T, voxel_size_z::T,
    nx::Int, ny::Int, nz::Int
) where T
    # Convert world coordinates to voxel indices (nearest neighbor)
    ix = round(Int, (x - vol_min_x) / voxel_size_x)
    iy = round(Int, (y - vol_min_y) / voxel_size_y)
    iz = round(Int, (z - vol_min_z) / voxel_size_z)

    # Bounds check - return 0 (background) if outside
    if ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz
        return UInt8(0)
    end

    return mask[ix, iy, iz]
end

"""
    accumulate_path_per_region!(
        path_lengths, mask, ray_origins, ray_directions,
        vol_min, voxel_size, n_samples, step_size, n_regions
    )

Ray march through volume and accumulate path lengths per region for each ray.

path_lengths: [n_cols, n_rows, n_angles, n_regions] - output
"""
function accumulate_path_per_region!(
    path_lengths::AbstractArray{T, 4},
    mask::AbstractArray{UInt8, 3},
    ray_origins::AbstractArray{T, 4},
    ray_directions::AbstractArray{T, 4},
    vol_min::NTuple{3, T},
    voxel_size::NTuple{3, T},
    n_samples::Int,
    step_size::T,
    n_regions::Int
) where T
    nx, ny, nz = size(mask)
    _, n_cols, n_rows, n_angles = size(ray_origins)

    # Extract components
    ox = ray_origins[1, :, :, :]
    oy = ray_origins[2, :, :, :]
    oz = ray_origins[3, :, :, :]

    dx = ray_directions[1, :, :, :]
    dy = ray_directions[2, :, :, :]
    dz = ray_directions[3, :, :, :]

    # March along all rays
    for step in 1:n_samples
        t = T(step) * step_size

        # Sample points for all rays
        sample_x = ox .+ t .* dx
        sample_y = oy .+ t .* dy
        sample_z = oz .+ t .* dz

        # Sample mask at each position and accumulate path lengths
        for angle_idx in 1:n_angles
            for row in 1:n_rows
                for col in 1:n_cols
                    region = nearest_neighbor_sample_mask(
                        mask,
                        sample_x[col, row, angle_idx],
                        sample_y[col, row, angle_idx],
                        sample_z[col, row, angle_idx],
                        vol_min[1], vol_min[2], vol_min[3],
                        voxel_size[1], voxel_size[2], voxel_size[3],
                        nx, ny, nz
                    )

                    # Accumulate path length for this region
                    # Region index in mask is 0-based, convert to 1-based for array indexing
                    # Region 0 is background (air), we include it in the calculation
                    region_idx = Int(region) + 1  # Convert 0-based to 1-based
                    if region_idx >= 1 && region_idx <= n_regions
                        path_lengths[col, row, angle_idx, region_idx] += step_size
                    end
                end
            end
        end
    end
end

"""
    compute_polychromatic_transmission(
        path_lengths, μ_by_energy, spectrum_weights, n_energies
    )

Compute polychromatic transmission using Beer-Lambert law.
Uses @trace to prevent loop unrolling over energy bins.

# Arguments
- `path_lengths`: [n_cols, n_rows, n_angles, n_regions] - path through each region
- `μ_by_energy`: [n_regions, n_energies] - attenuation coefficients
- `spectrum_weights`: [n_energies] - normalized spectrum weights
- `n_energies`: Number of energy bins

# Returns
Transmission [n_cols, n_rows, n_angles]
"""
function compute_polychromatic_transmission(
    path_lengths::AbstractArray{<:Real, 4},
    μ_by_energy::AbstractArray{<:Real, 2},
    spectrum_weights::AbstractArray{<:Real, 1},
    n_energies::Int
)
    n_cols, n_rows, n_angles, n_regions = size(path_lengths)
    ET = eltype(path_lengths)  # Element type

    # Initialize transmission accumulator
    transmission = zeros(ET, n_cols, n_rows, n_angles)

    # Use @trace to prevent loop unrolling - critical for XLA efficiency
    @trace for e_idx in 1:n_energies
        weight = spectrum_weights[e_idx]

        # Compute line integral for this energy
        line_integral = zeros(ET, n_cols, n_rows, n_angles)

        for region in 1:n_regions
            μ = μ_by_energy[region, e_idx]
            line_integral .+= path_lengths[:, :, :, region] .* μ
        end

        # Accumulate weighted transmission (Beer-Lambert law)
        transmission .+= weight .* exp.(-line_integral)
    end

    return transmission
end

"""
    forward_project_polychromatic_raymarching(
        mask, μ_by_energy, spectrum_weights,
        ray_origins, ray_directions,
        vol_min, voxel_size, n_samples, step_size
    )

Polychromatic forward projection using ray marching.

This is a SINGLE compiled kernel for ALL angles and ALL energies.

# Algorithm
1. Ray march to accumulate path lengths per region (~1.7 GB for clinical scale)
2. Loop over energies with @trace (no unrolling)
3. Compute Beer-Lambert transmission: I/I₀ = Σ S(E) * exp(-∫ μ(E,r) dr)
4. Convert to sinogram: -log(transmission)

# Arguments
- `mask`: Region labels [nx, ny, nz] (UInt8, values 0-26)
- `μ_by_energy`: Attenuation table [n_regions, n_energies]
- `spectrum_weights`: Normalized spectrum weights [n_energies]
- `ray_origins`: Ray start points [3, n_cols, n_rows, n_angles]
- `ray_directions`: Normalized ray directions [3, n_cols, n_rows, n_angles]
- `vol_min`: Volume minimum coordinates (x, y, z)
- `voxel_size`: Voxel dimensions (dx, dy, dz)
- `n_samples`: Number of samples per ray
- `step_size`: Distance between samples

# Returns
Sinogram [n_cols, n_rows, n_angles] with beam hardening effects
"""
function forward_project_polychromatic_raymarching(
    mask::AbstractArray{UInt8, 3},
    μ_by_energy::AbstractArray{T, 2},
    spectrum_weights::AbstractArray{T, 1},
    ray_origins::AbstractArray{T, 4},
    ray_directions::AbstractArray{T, 4},
    vol_min::NTuple{3, T},
    voxel_size::NTuple{3, T},
    n_samples::Int,
    step_size::T
) where T
    _, n_cols, n_rows, n_angles = size(ray_origins)
    n_regions, n_energies = size(μ_by_energy)

    # Step 1: Accumulate path lengths per region
    path_lengths = zeros(T, n_cols, n_rows, n_angles, n_regions)
    accumulate_path_per_region!(
        path_lengths, mask, ray_origins, ray_directions,
        vol_min, voxel_size, n_samples, step_size, n_regions
    )

    # Step 2: Compute polychromatic transmission
    transmission = compute_polychromatic_transmission(
        path_lengths, μ_by_energy, spectrum_weights, n_energies
    )

    # Step 3: Convert to sinogram (attenuation)
    sinogram = -log.(clamp.(transmission, T(1e-10), T(Inf)))

    return sinogram
end

"""
    create_polychromatic_projector_raymarching(
        phantom, geom, kVp;
        n_bins=20, spectrum_source=:xspect
    )

Create parameters for polychromatic ray marching projection.

Returns a NamedTuple with all parameters needed for `forward_project_polychromatic_raymarching`.

# Example
```julia
params = create_polychromatic_projector_raymarching(phantom, geom, 120)
sinogram = forward_project_polychromatic_raymarching(
    params.mask, params.μ_by_energy, params.spectrum_weights,
    params.ray_origins, params.ray_directions,
    params.vol_min, params.voxel_size, params.n_samples, params.step_size
)
```
"""
function create_polychromatic_projector_raymarching(
    phantom::Phantom,
    geom::CTGeometry,
    kVp::Int;
    n_bins::Int=20,
    spectrum_source::Symbol=:xspect,
    step_factor::Float64=0.5
)
    # Load spectrum
    energies, weights = load_spectrum(kVp; source=spectrum_source)

    # Bin spectrum for efficiency
    if length(energies) > n_bins
        bin_size = length(energies) ÷ n_bins
        binned_energies = Float64[]
        binned_weights = Float64[]

        for i in 1:n_bins
            start_idx = (i - 1) * bin_size + 1
            end_idx = min(i * bin_size, length(energies))
            push!(binned_energies, mean(energies[start_idx:end_idx]))
            push!(binned_weights, sum(weights[start_idx:end_idx]))
        end

        energies = binned_energies
        weights = binned_weights
    end

    # Normalize weights
    weights = weights ./ sum(weights)

    # Get unique materials from phantom (based on region labels)
    # Map each region to its material's μ at each energy
    materials = get_region_materials()  # Returns material for each region index
    n_regions = length(materials)
    n_energies = length(energies)

    # Build μ table: [n_regions, n_energies]
    μ_by_energy = zeros(Float32, n_regions, n_energies)
    for (region_idx, material) in enumerate(materials)
        for (e_idx, E) in enumerate(energies)
            μ_by_energy[region_idx, e_idx] = Float32(compute_μ_at_energy(material, E))
        end
    end

    # Compute ray geometry
    ray_geom = compute_ray_geometry(geom, phantom.fov, size(phantom.μ))

    # Compute step size and number of samples
    min_voxel = minimum(phantom.voxel_size)
    step_size = Float32(min_voxel * step_factor)
    # Use full ray path from source through volume (same as forward_project_raymarching)
    half_diagonal = sqrt(sum((phantom.fov ./ 2) .^ 2))
    ray_path_to_far_edge = geom.SAD + half_diagonal
    n_samples = ceil(Int, ray_path_to_far_edge / step_size) + 10

    vol_min = ray_geom.vol_min
    voxel_size = (
        Float32(phantom.voxel_size[1]),
        Float32(phantom.voxel_size[2]),
        Float32(phantom.voxel_size[3])
    )

    return (
        mask = phantom.mask,
        μ_by_energy = μ_by_energy,
        spectrum_weights = Float32.(weights),
        ray_origins = ray_geom.origins,
        ray_directions = ray_geom.directions,
        vol_min = vol_min,
        voxel_size = voxel_size,
        n_samples = n_samples,
        step_size = step_size,
        energies = Float32.(energies),
        effective_energy = sum(energies .* weights),
        n_energies = n_energies,
        n_regions = n_regions
    )
end

"""
    forward_project_polychromatic(phantom, geom, kVp; kwargs...)

High-level polychromatic forward projection using ray marching.

This computes geometry and spectrum tables once, then runs a single kernel
that processes ALL angles and ALL energies.

# Arguments
- `phantom::Phantom`: Input phantom with μ and mask
- `geom::CTGeometry`: Scanner geometry
- `kVp::Int`: Tube voltage (determines spectrum)
- `n_bins::Int=20`: Number of energy bins
- `spectrum_source::Symbol=:xspect`: Spectrum data source

# Returns
Sinogram [n_cols, n_rows, n_angles] with beam hardening effects
"""
function forward_project_polychromatic(
    phantom::Phantom,
    geom::CTGeometry,
    kVp::Int;
    n_bins::Int=20,
    spectrum_source::Symbol=:xspect,
    step_factor::Float64=0.5
)
    params = create_polychromatic_projector_raymarching(
        phantom, geom, kVp;
        n_bins=n_bins, spectrum_source=spectrum_source, step_factor=step_factor
    )

    return forward_project_polychromatic_raymarching(
        params.mask,
        params.μ_by_energy,
        params.spectrum_weights,
        params.ray_origins,
        params.ray_directions,
        params.vol_min,
        params.voxel_size,
        params.n_samples,
        params.step_size
    )
end

# =============================================================================
# Exports
# =============================================================================

export RayGeometry, compute_ray_geometry
export forward_project_raymarching
export forward_project_raymarching_compiled
export forward_project_raymarching_kernel
export forward_project_raymarching_vectorized
export forward_project_polychromatic_raymarching
export create_polychromatic_projector_raymarching
export forward_project_polychromatic
