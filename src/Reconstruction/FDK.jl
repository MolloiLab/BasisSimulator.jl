"""
    Reconstruction/FDK.jl

Feldkamp-Davis-Kress (FDK) cone-beam CT reconstruction.

Two implementations:
1. **Legacy**: Loop-based with FFTW (faster for single runs)
2. **Differentiable**: XLA-compatible with pre-computed geometry (for Reactant/autodiff)

Standard FDK algorithm:
1. Pre-weight sinogram (cosine weighting)
2. Filter each row with ramp filter
3. Backproject into volume
"""

using FFTW

# =============================================================================
# Pre-computed Backprojection Geometry (for XLA compatibility)
# =============================================================================

"""
    BackprojectionGeometry

Pre-computed geometry for FDK backprojection using gather operations.
All indices and weights are computed once and reused for multiple reconstructions.

This enables XLA compilation - the backprojection kernel only uses array operations.
"""
struct BackprojectionGeometry
    # For each voxel, which 4 sinogram pixels to sample (bilinear corners)
    # Shape: [4, nx, ny, nz, n_angles]
    linear_indices::Array{Int, 5}

    # Bilinear interpolation weights for each corner
    # Shape: [4, nx, ny, nz, n_angles]
    bilinear_weights::Array{Float32, 5}

    # Distance weighting for each voxel-angle pair
    # Shape: [nx, ny, nz, n_angles]
    distance_weights::Array{Float32, 4}

    # Volume and sinogram dimensions
    nx::Int
    ny::Int
    nz::Int
    n_cols::Int
    n_rows::Int
    n_angles::Int
end

"""
    precompute_backprojection_geometry(geom, output_size, fov; tang_order=0)

Pre-compute backprojection geometry for XLA-compatible FDK.

# Arguments
- `geom::CTGeometry`: Scanner geometry
- `output_size::NTuple{3,Int}`: Output volume size (nx, ny, nz)
- `fov::NTuple{3,Float64}`: Field of view (cm)
- `tang_order::Int`: Tang 3D weighting order (default: 0)

# Returns
`BackprojectionGeometry` with pre-computed indices and weights.
"""
function precompute_backprojection_geometry(
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64};
    tang_order::Int=0
)
    nx, ny, nz = output_size
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles

    # Voxel sizes
    dx = fov[1] / nx
    dy = fov[2] / ny
    dz = fov[3] / nz

    # Coordinate ranges (centered at origin)
    x_coords = range(-fov[1]/2 + dx/2, fov[1]/2 - dx/2, length=nx)
    y_coords = range(-fov[2]/2 + dy/2, fov[2]/2 - dy/2, length=ny)
    z_coords = range(-fov[3]/2 + dz/2, fov[3]/2 - dz/2, length=nz)

    # Detector pixel size at detector plane
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    # Angular step for scaling
    delta_angle = Float32(2π / n_angles)

    # Pre-allocate output arrays
    # 4 corners for bilinear interpolation
    linear_indices = ones(Int, 4, nx, ny, nz, n_angles)
    bilinear_weights = zeros(Float32, 4, nx, ny, nz, n_angles)
    distance_weights = zeros(Float32, nx, ny, nz, n_angles)

    for angle_idx in 1:n_angles
        # Source position
        sx = geom.source_positions[1, angle_idx]
        sy = geom.source_positions[2, angle_idx]
        sz = geom.source_positions[3, angle_idx]

        # Detector center
        dcx = geom.detector_centers[1, angle_idx]
        dcy = geom.detector_centers[2, angle_idx]
        dcz = geom.detector_centers[3, angle_idx]

        # Detector axes
        ux = geom.detector_u[1, angle_idx]
        uy = geom.detector_u[2, angle_idx]
        uz = geom.detector_u[3, angle_idx]

        vx = geom.detector_v[1, angle_idx]
        vy = geom.detector_v[2, angle_idx]
        vz = geom.detector_v[3, angle_idx]

        # Source-detector axis (normalized)
        sd_x = dcx - sx
        sd_y = dcy - sy
        sd_z = dcz - sz
        sd_len = sqrt(sd_x^2 + sd_y^2 + sd_z^2)
        sd_x /= sd_len
        sd_y /= sd_len
        sd_z /= sd_len

        for iz in 1:nz
            z = z_coords[iz]

            for iy in 1:ny
                y = y_coords[iy]

                for ix in 1:nx
                    x = x_coords[ix]

                    # Vector from source to voxel
                    rx = x - sx
                    ry = y - sy
                    rz = z - sz

                    # Project voxel onto source-detector line
                    t = rx * sd_x + ry * sd_y + rz * sd_z

                    # Skip if behind source
                    if t <= 0
                        continue
                    end

                    # Scale factor: where ray through voxel hits detector plane
                    scale = geom.SDD / t

                    # Find intersection point on detector
                    hit_x = sx + scale * rx
                    hit_y = sy + scale * ry
                    hit_z = sz + scale * rz

                    # Convert to detector coordinates (u, v)
                    dhx = hit_x - dcx
                    dhy = hit_y - dcy
                    dhz = hit_z - dcz

                    u = dhx * ux + dhy * uy + dhz * uz
                    v = dhx * vx + dhy * vy + dhz * vz

                    # Convert to detector pixel indices (continuous)
                    col_f = u / pixel_size_det + (n_cols + 1) / 2
                    row_f = v / pixel_size_det + (n_rows + 1) / 2

                    # Check bounds
                    if !(1 <= col_f <= n_cols && 1 <= row_f <= n_rows)
                        continue
                    end

                    # Bilinear interpolation indices
                    col0 = floor(Int, col_f)
                    col1 = col0 + 1
                    row0 = floor(Int, row_f)
                    row1 = row0 + 1

                    # Fractional parts
                    fc = Float32(col_f - col0)
                    fr = Float32(row_f - row0)

                    # Clamp to valid range
                    col0 = clamp(col0, 1, n_cols)
                    col1 = clamp(col1, 1, n_cols)
                    row0 = clamp(row0, 1, n_rows)
                    row1 = clamp(row1, 1, n_rows)

                    # Linear indices for each corner (into flattened [n_cols, n_rows, n_angles] sinogram)
                    # Note: Julia is column-major, so linear index = col + (row-1)*n_cols + (angle-1)*n_cols*n_rows
                    linear_indices[1, ix, iy, iz, angle_idx] = col0 + (row0-1)*n_cols + (angle_idx-1)*n_cols*n_rows
                    linear_indices[2, ix, iy, iz, angle_idx] = col1 + (row0-1)*n_cols + (angle_idx-1)*n_cols*n_rows
                    linear_indices[3, ix, iy, iz, angle_idx] = col0 + (row1-1)*n_cols + (angle_idx-1)*n_cols*n_rows
                    linear_indices[4, ix, iy, iz, angle_idx] = col1 + (row1-1)*n_cols + (angle_idx-1)*n_cols*n_rows

                    # Bilinear weights
                    bilinear_weights[1, ix, iy, iz, angle_idx] = (1 - fc) * (1 - fr)  # col0, row0
                    bilinear_weights[2, ix, iy, iz, angle_idx] = fc * (1 - fr)        # col1, row0
                    bilinear_weights[3, ix, iy, iz, angle_idx] = (1 - fc) * fr        # col0, row1
                    bilinear_weights[4, ix, iy, iz, angle_idx] = fc * fr              # col1, row1

                    # Distance weighting
                    weight = Float32((geom.SAD / t)^2 * delta_angle)

                    # Tang 3D weighting
                    if tang_order > 0
                        cos_kappa = geom.SDD / sqrt(geom.SDD^2 + v^2)
                        tang_weight = Float32(cos_kappa^tang_order)
                        weight *= tang_weight
                    end

                    distance_weights[ix, iy, iz, angle_idx] = weight
                end
            end
        end
    end

    return BackprojectionGeometry(
        linear_indices, bilinear_weights, distance_weights,
        nx, ny, nz, n_cols, n_rows, n_angles
    )
end

# =============================================================================
# Functional (Non-mutating) Operations
# =============================================================================

"""
    preweight_cosine(sinogram, geom)

Apply cosine weighting for cone-beam geometry (functional, non-mutating).

Returns a new weighted sinogram array.
"""
function preweight_cosine(sinogram::AbstractArray{T,3}, geom::CTGeometry) where T
    n_cols, n_rows, n_angles = size(sinogram)

    # Detector pixel size at detector plane
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    # Create weight array
    weights = Array{T}(undef, n_cols, n_rows)

    for row in 1:n_rows
        for col in 1:n_cols
            u = (col - (n_cols + 1) / 2) * pixel_size_det
            v = (row - (n_rows + 1) / 2) * pixel_size_det
            dist_sq = geom.SDD^2 + u^2 + v^2
            weights[col, row] = T(geom.SDD / sqrt(dist_sq))
        end
    end

    # Broadcast multiply (non-mutating)
    return sinogram .* weights
end

"""
    create_ramp_kernel(n_cols, pixel_size)

Create a real-space ramp filter kernel for convolution.

This avoids FFTW dependency and is XLA-compatible.
The kernel is based on the Ram-Lak filter in spatial domain.
"""
function create_ramp_kernel(n_cols::Int, pixel_size::Float64)
    # Create spatial-domain ramp filter
    # Based on Kak & Slaney, equation 3.60
    n = 2 * n_cols  # Padded size for linear convolution
    kernel = zeros(Float32, n)

    center = n ÷ 2 + 1

    for i in 1:n
        k = i - center
        if k == 0
            # Central value: 1/(4*pixel_size^2)
            kernel[i] = Float32(1.0 / (4.0 * pixel_size^2))
        elseif k % 2 == 0
            # Even indices: 0
            kernel[i] = 0.0f0
        else
            # Odd indices: -1/(pi^2 * k^2 * pixel_size^2)
            kernel[i] = Float32(-1.0 / (π^2 * k^2 * pixel_size^2))
        end
    end

    return kernel
end

"""
    filter_ramp(sinogram, geom, kernel_type=:ramp)

Apply ramp filter to sinogram rows using real-space convolution.

This is XLA-compatible as it uses only array operations.
For efficiency, uses FFT-based convolution but with pre-computed filter.

# Arguments
- `sinogram`: Pre-weighted sinogram [n_cols, n_rows, n_angles]
- `geom`: Scanner geometry
- `kernel_type`: Kernel type (:ramp, :shepp_logan, :soft, :standard, :bone)
"""
function filter_ramp(sinogram::AbstractArray{T,3}, geom::CTGeometry;
                     kernel::ReconKernel=RampKernel()) where T
    n_cols, n_rows, n_angles = size(sinogram)

    # Pad for FFT
    n_fft = nextpow(2, 2 * n_cols)

    # Create frequency domain filter
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    filter_freq = create_kernel_filter(kernel, n_fft, pixel_size_det)

    # Output array
    filtered = similar(sinogram)

    # Process each row
    for angle_idx in 1:n_angles
        for row in 1:n_rows
            # Extract row with zero padding
            padded = zeros(Float64, n_fft)
            for col in 1:n_cols
                padded[col] = sinogram[col, row, angle_idx]
            end

            # FFT, filter, IFFT
            row_fft = fft(padded)
            row_fft .*= filter_freq
            filtered_row = real(ifft(row_fft))

            # Store result
            for col in 1:n_cols
                filtered[col, row, angle_idx] = T(filtered_row[col])
            end
        end
    end

    return filtered
end

# =============================================================================
# XLA-Compatible Backprojection
# =============================================================================

"""
    backproject_volume(sinogram_flat, bp_geom::BackprojectionGeometry)

Backproject filtered sinogram into volume using pre-computed geometry.

This is the XLA-compilable kernel - only uses array gather operations.

# Arguments
- `sinogram_flat`: Flattened filtered sinogram [n_cols * n_rows * n_angles]
- `bp_geom`: Pre-computed backprojection geometry

# Returns
Reconstructed volume [nx, ny, nz]
"""
function backproject_volume(sinogram_flat::AbstractVector{T}, bp_geom::BackprojectionGeometry) where T
    nx, ny, nz = bp_geom.nx, bp_geom.ny, bp_geom.nz
    n_angles = bp_geom.n_angles

    # Gather sinogram samples at all corners for all voxels and angles
    # Shape of linear_indices: [4, nx, ny, nz, n_angles]
    samples = sinogram_flat[bp_geom.linear_indices]  # [4, nx, ny, nz, n_angles]

    # Apply bilinear interpolation weights
    interpolated = samples .* bp_geom.bilinear_weights  # [4, nx, ny, nz, n_angles]

    # Sum over corners (bilinear combination)
    sino_sampled = dropdims(sum(interpolated, dims=1), dims=1)  # [nx, ny, nz, n_angles]

    # Apply distance weights
    weighted = sino_sampled .* bp_geom.distance_weights  # [nx, ny, nz, n_angles]

    # Sum over angles
    volume = dropdims(sum(weighted, dims=4), dims=4)  # [nx, ny, nz]

    return volume
end

"""
    fdk_reconstruct_xla(sinogram, geom, bp_geom; kernel=RampKernel())

XLA-compatible FDK reconstruction using pre-computed geometry.

# Arguments
- `sinogram`: Projections [n_cols, n_rows, n_angles]
- `geom`: Scanner geometry (for pre-weighting and filtering)
- `bp_geom`: Pre-computed backprojection geometry
- `kernel`: Reconstruction kernel

# Returns
Reconstructed volume [nx, ny, nz]

# Usage
```julia
# Pre-compute geometry once
bp_geom = precompute_backprojection_geometry(geom, output_size, fov)

# Run reconstruction (can be compiled with Reactant)
volume = fdk_reconstruct_xla(sinogram, geom, bp_geom)
```
"""
function fdk_reconstruct_xla(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    bp_geom::BackprojectionGeometry;
    kernel::ReconKernel=RampKernel()
) where T
    # Step 1: Pre-weight sinogram (functional)
    weighted = preweight_cosine(sinogram, geom)

    # Step 2: Filter rows
    filtered = filter_ramp(weighted, geom; kernel=kernel)

    # Step 3: Backproject using pre-computed geometry
    sinogram_flat = vec(filtered)
    volume = backproject_volume(sinogram_flat, bp_geom)

    return volume
end

# =============================================================================
# Legacy Implementation (Loop-based, non-differentiable)
# =============================================================================

"""
    fdk_reconstruct(sinogram::Array{Float32,3}, geom::CTGeometry, output_size::NTuple{3,Int}, fov::NTuple{3,Float64}; kernel=RampKernel(), tang_order=0)

Reconstruct a 3D volume from cone-beam projections using FDK.

# Arguments
- `sinogram::Array{Float32,3}`: Projections [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Scanner geometry
- `output_size::NTuple{3,Int}`: Output volume size (nx, ny, nz)
- `fov::NTuple{3,Float64}`: Field of view (cm) for output volume
- `kernel::ReconKernel`: Reconstruction kernel (default: RampKernel)
- `tang_order::Int`: Tang 3D weighting order (default: 0 = disabled)

# Returns
`Array{Float32,3}`: Reconstructed μ values (cm⁻¹)
"""
function fdk_reconstruct(
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64};
    kernel::ReconKernel=RampKernel(),
    tang_order::Int=0
)
    n_cols, n_rows, n_angles = size(sinogram)

    # Step 1: Pre-weight sinogram
    weighted = copy(sinogram)
    preweight_sinogram!(weighted, geom)

    # Step 2: Filter rows with specified kernel
    filtered = copy(weighted)
    filter_sinogram!(filtered, geom, kernel)

    # Step 3: Backproject with optional Tang weighting
    volume = zeros(Float32, output_size...)
    backproject!(volume, filtered, geom, fov, tang_order)

    return volume
end

"""
    preweight_sinogram!(sinogram, geom)

Apply cosine weighting for cone-beam geometry (in-place, legacy).
"""
function preweight_sinogram!(sinogram::Array{Float32,3}, geom::CTGeometry)
    n_cols, n_rows, n_angles = size(sinogram)

    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    for row in 1:n_rows
        for col in 1:n_cols
            u = (col - (n_cols + 1) / 2) * pixel_size_det
            v = (row - (n_rows + 1) / 2) * pixel_size_det
            dist_sq = geom.SDD^2 + u^2 + v^2
            weight = Float32(geom.SDD / sqrt(dist_sq))

            for angle_idx in 1:n_angles
                sinogram[col, row, angle_idx] *= weight
            end
        end
    end
end

"""
    filter_sinogram!(sinogram, geom, kernel)

Apply reconstruction filter to each row of the sinogram (in-place, legacy).
"""
function filter_sinogram!(sinogram::Array{Float32,3}, geom::CTGeometry,
                          kernel::ReconKernel=RampKernel())
    n_cols, n_rows, n_angles = size(sinogram)

    n_fft = nextpow(2, 2 * n_cols)
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    filter_freq = create_kernel_filter(kernel, n_fft, pixel_size_det)

    padded = zeros(Float64, n_fft)

    for angle_idx in 1:n_angles
        for row in 1:n_rows
            fill!(padded, 0.0)
            for col in 1:n_cols
                padded[col] = sinogram[col, row, angle_idx]
            end

            row_fft = fft(padded)
            row_fft .*= filter_freq
            filtered_row = real(ifft(row_fft))

            for col in 1:n_cols
                sinogram[col, row, angle_idx] = Float32(filtered_row[col])
            end
        end
    end
end

"""
    backproject!(volume, sinogram, geom, fov, tang_order=0)

Backproject filtered projections into volume (in-place, legacy).
"""
function backproject!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    fov::NTuple{3,Float64},
    tang_order::Int=0
)
    nx, ny, nz = size(volume)
    n_cols, n_rows, n_angles = size(sinogram)

    dx = fov[1] / nx
    dy = fov[2] / ny
    dz = fov[3] / nz

    x_coords = range(-fov[1]/2 + dx/2, fov[1]/2 - dx/2, length=nx)
    y_coords = range(-fov[2]/2 + dy/2, fov[2]/2 - dy/2, length=ny)
    z_coords = range(-fov[3]/2 + dz/2, fov[3]/2 - dz/2, length=nz)

    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    delta_angle = 2π / n_angles

    for angle_idx in 1:n_angles
        sx = geom.source_positions[1, angle_idx]
        sy = geom.source_positions[2, angle_idx]
        sz = geom.source_positions[3, angle_idx]

        dcx = geom.detector_centers[1, angle_idx]
        dcy = geom.detector_centers[2, angle_idx]
        dcz = geom.detector_centers[3, angle_idx]

        ux = geom.detector_u[1, angle_idx]
        uy = geom.detector_u[2, angle_idx]
        uz = geom.detector_u[3, angle_idx]

        vx = geom.detector_v[1, angle_idx]
        vy = geom.detector_v[2, angle_idx]
        vz = geom.detector_v[3, angle_idx]

        for iz in 1:nz
            z = z_coords[iz]

            for iy in 1:ny
                y = y_coords[iy]

                for ix in 1:nx
                    x = x_coords[ix]

                    rx = x - sx
                    ry = y - sy
                    rz = z - sz

                    sd_x = dcx - sx
                    sd_y = dcy - sy
                    sd_z = dcz - sz
                    sd_len = sqrt(sd_x^2 + sd_y^2 + sd_z^2)
                    sd_x /= sd_len
                    sd_y /= sd_len
                    sd_z /= sd_len

                    t = rx * sd_x + ry * sd_y + rz * sd_z
                    scale = geom.SDD / t

                    hit_x = sx + scale * rx
                    hit_y = sy + scale * ry
                    hit_z = sz + scale * rz

                    dhx = hit_x - dcx
                    dhy = hit_y - dcy
                    dhz = hit_z - dcz

                    u = dhx * ux + dhy * uy + dhz * uz
                    v = dhx * vx + dhy * vy + dhz * vz

                    col_f = u / pixel_size_det + (n_cols + 1) / 2
                    row_f = v / pixel_size_det + (n_rows + 1) / 2

                    if 1 <= col_f <= n_cols && 1 <= row_f <= n_rows
                        weight = Float32((geom.SAD / t)^2 * delta_angle)

                        if tang_order > 0
                            cos_kappa = geom.SDD / sqrt(geom.SDD^2 + v^2)
                            tang_weight = Float32(cos_kappa^tang_order)
                            weight *= tang_weight
                        end

                        sample = sample_sinogram_bilinear(sinogram, col_f, row_f, angle_idx)
                        volume[ix, iy, iz] += weight * sample
                    end
                end
            end
        end
    end
end

"""
    sample_sinogram_bilinear(sinogram, col, row, angle_idx)

Sample sinogram with bilinear interpolation (legacy helper).
"""
function sample_sinogram_bilinear(
    sinogram::Array{Float32,3},
    col::Float64,
    row::Float64,
    angle_idx::Int
)
    n_cols, n_rows, _ = size(sinogram)

    col0 = floor(Int, col)
    col1 = col0 + 1
    row0 = floor(Int, row)
    row1 = row0 + 1

    fc = col - col0
    fr = row - row0

    col0 = clamp(col0, 1, n_cols)
    col1 = clamp(col1, 1, n_cols)
    row0 = clamp(row0, 1, n_rows)
    row1 = clamp(row1, 1, n_rows)

    v00 = sinogram[col0, row0, angle_idx]
    v10 = sinogram[col1, row0, angle_idx]
    v01 = sinogram[col0, row1, angle_idx]
    v11 = sinogram[col1, row1, angle_idx]

    v0 = v00 * (1 - fc) + v10 * fc
    v1 = v01 * (1 - fc) + v11 * fc

    return v0 * (1 - fr) + v1 * fr
end

# =============================================================================
# Convenience Overloads
# =============================================================================

"""
    fdk_reconstruct(sinogram, geom; n_voxels=128, kernel=RampKernel(), tang_order=0)

Convenience version with automatic FOV calculation.
"""
function fdk_reconstruct(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry;
    n_voxels::Int=128,
    kernel::ReconKernel=RampKernel(),
    tang_order::Int=0
) where T
    sino32 = T == Float32 ? sinogram : Array{Float32}(sinogram)

    fov_xy = geom.n_cols * geom.pixel_size * 1.1
    fov_z = geom.n_rows * geom.pixel_size * 1.1

    output_size = (n_voxels, n_voxels, n_voxels)
    fov = (fov_xy, fov_xy, fov_z)

    return fdk_reconstruct(sino32, geom, output_size, fov; kernel=kernel, tang_order=tang_order)
end

"""
    fdk_reconstruct(sinogram, geom, output_size; fov=nothing, kernel=RampKernel(), tang_order=0)

Version with output_size tuple and optional FOV.
"""
function fdk_reconstruct(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int};
    fov::Union{NTuple{3,Float64},Nothing}=nothing,
    kernel::ReconKernel=RampKernel(),
    tang_order::Int=0
) where T
    sino32 = T == Float32 ? sinogram : Array{Float32}(sinogram)

    if fov === nothing
        fov_xy = geom.n_cols * geom.pixel_size * 1.1
        fov_z = geom.n_rows * geom.pixel_size * 1.1
        fov = (fov_xy, fov_xy, fov_z)
    end
    return fdk_reconstruct(sino32, geom, output_size, fov; kernel=kernel, tang_order=tang_order)
end

# =============================================================================
# Exports
# =============================================================================

export fdk_reconstruct, fdk_reconstruct_xla
export BackprojectionGeometry, precompute_backprojection_geometry
export preweight_cosine, filter_ramp, backproject_volume
export create_ramp_kernel
