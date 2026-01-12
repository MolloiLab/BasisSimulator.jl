"""
    Reconstruction/FDK.jl

Feldkamp-Davis-Kress (FDK) cone-beam CT reconstruction.

Standard FDK algorithm:
1. Pre-weight sinogram (cosine weighting)
2. Filter each row with ramp filter (FFT-based)
3. Backproject into volume

Designed for Reactant/XLA compatibility.
"""

using FFTW

# =============================================================================
# FDK Reconstruction
# =============================================================================

"""
    fdk_reconstruct(sinogram::Array{Float32,3}, geom::CTGeometry, output_size::NTuple{3,Int}, fov::NTuple{3,Float64})

Reconstruct a 3D volume from cone-beam projections using FDK.

# Arguments
- `sinogram::Array{Float32,3}`: Projections [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Scanner geometry
- `output_size::NTuple{3,Int}`: Output volume size (nx, ny, nz)
- `fov::NTuple{3,Float64}`: Field of view (cm) for output volume

# Returns
`Array{Float32,3}`: Reconstructed μ values (cm⁻¹)

# Algorithm
1. Cosine-weight projections for cone-beam geometry
2. Apply ramp filter via FFT
3. Backproject with distance weighting
"""
function fdk_reconstruct(
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    output_size::NTuple{3,Int},
    fov::NTuple{3,Float64}
)
    n_cols, n_rows, n_angles = size(sinogram)

    # Step 1: Pre-weight sinogram
    weighted = copy(sinogram)
    preweight_sinogram!(weighted, geom)

    # Step 2: Filter rows
    filtered = copy(weighted)
    filter_sinogram!(filtered, geom)

    # Step 3: Backproject
    volume = zeros(Float32, output_size...)
    backproject!(volume, filtered, geom, fov)

    return volume
end

"""
    preweight_sinogram!(sinogram, geom)

Apply cosine weighting for cone-beam geometry.
"""
function preweight_sinogram!(sinogram::Array{Float32,3}, geom::CTGeometry)
    n_cols, n_rows, n_angles = size(sinogram)

    # Detector pixel size at detector plane
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    for row in 1:n_rows
        for col in 1:n_cols
            # Offset from detector center
            u = (col - (n_cols + 1) / 2) * pixel_size_det
            v = (row - (n_rows + 1) / 2) * pixel_size_det

            # Distance from source to pixel
            dist_sq = geom.SDD^2 + u^2 + v^2

            # Cosine weight
            weight = Float32(geom.SDD / sqrt(dist_sq))

            for angle_idx in 1:n_angles
                sinogram[col, row, angle_idx] *= weight
            end
        end
    end
end

"""
    filter_sinogram!(sinogram, geom)

Apply ramp filter to each row of the sinogram.
Uses FFT-based filtering with Ram-Lak filter in frequency domain.
"""
function filter_sinogram!(sinogram::Array{Float32,3}, geom::CTGeometry)
    n_cols, n_rows, n_angles = size(sinogram)

    # Pad for FFT (next power of 2)
    n_fft = nextpow(2, 2 * n_cols)

    # Create frequency domain ramp filter
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    ramp_freq = create_ramp_filter_freq(n_fft, pixel_size_det)

    padded = zeros(Float64, n_fft)

    for angle_idx in 1:n_angles
        for row in 1:n_rows
            # Extract row with zero padding
            fill!(padded, 0.0)
            for col in 1:n_cols
                padded[col] = sinogram[col, row, angle_idx]
            end

            # FFT, multiply by ramp, inverse FFT
            row_fft = fft(padded)
            row_fft .*= ramp_freq
            filtered_row = real(ifft(row_fft))

            # Store back (extract valid portion)
            for col in 1:n_cols
                sinogram[col, row, angle_idx] = Float32(filtered_row[col])
            end
        end
    end
end

"""
    create_ramp_filter_freq(n_fft, pixel_size)

Create a frequency domain ramp filter (Ram-Lak).
Returns complex array for direct multiplication with FFT output.
"""
function create_ramp_filter_freq(n_fft::Int, pixel_size::Float64)
    ramp = zeros(ComplexF64, n_fft)

    # Nyquist frequency
    freq_max = 1.0 / (2.0 * pixel_size)

    for i in 1:n_fft
        # Frequency index (0 to n_fft-1, then wraps to negative)
        if i <= n_fft ÷ 2 + 1
            freq_idx = i - 1
        else
            freq_idx = i - 1 - n_fft
        end

        # Normalized frequency (0 to 0.5 for positive, -0.5 to 0 for negative)
        freq_normalized = freq_idx / n_fft

        # Frequency in physical units
        freq = freq_normalized / pixel_size

        # Ram-Lak filter: |freq| with cutoff at Nyquist
        if abs(freq) <= freq_max
            ramp[i] = abs(freq)
        else
            ramp[i] = 0.0
        end
    end

    return ramp
end

"""
    backproject!(volume, sinogram, geom, fov)

Backproject filtered projections into volume.
"""
function backproject!(
    volume::Array{Float32,3},
    sinogram::Array{Float32,3},
    geom::CTGeometry,
    fov::NTuple{3,Float64}
)
    nx, ny, nz = size(volume)
    n_cols, n_rows, n_angles = size(sinogram)

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
    delta_angle = 2π / n_angles

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

                    # Distance from source to voxel along source-detector axis
                    # The source-detector axis points from source toward detector center
                    sd_x = dcx - sx
                    sd_y = dcy - sy
                    sd_z = dcz - sz
                    sd_len = sqrt(sd_x^2 + sd_y^2 + sd_z^2)
                    sd_x /= sd_len
                    sd_y /= sd_len
                    sd_z /= sd_len

                    # Project voxel onto source-detector line
                    t = rx * sd_x + ry * sd_y + rz * sd_z

                    # Scale factor: where ray through voxel hits detector plane
                    scale = geom.SDD / t

                    # Find intersection point on detector
                    # Ray from source through voxel: S + t' * (voxel - source)
                    # where t' = SDD / t makes it hit detector plane
                    hit_x = sx + scale * rx
                    hit_y = sy + scale * ry
                    hit_z = sz + scale * rz

                    # Convert to detector coordinates (u, v)
                    # Vector from detector center to hit point
                    dhx = hit_x - dcx
                    dhy = hit_y - dcy
                    dhz = hit_z - dcz

                    # Project onto detector axes
                    u = dhx * ux + dhy * uy + dhz * uz
                    v = dhx * vx + dhy * vy + dhz * vz

                    # Convert to detector pixel indices (continuous)
                    col_f = u / pixel_size_det + (n_cols + 1) / 2
                    row_f = v / pixel_size_det + (n_rows + 1) / 2

                    # Bilinear interpolation
                    if 1 <= col_f <= n_cols && 1 <= row_f <= n_rows
                        # Distance weighting for cone beam
                        weight = Float32((geom.SAD / t)^2 * delta_angle)

                        # Sample sinogram with bilinear interpolation
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

Sample sinogram at continuous (col, row) coordinates using bilinear interpolation.
"""
function sample_sinogram_bilinear(
    sinogram::Array{Float32,3},
    col::Float64,
    row::Float64,
    angle_idx::Int
)
    n_cols, n_rows, _ = size(sinogram)

    # Integer indices
    col0 = floor(Int, col)
    col1 = col0 + 1
    row0 = floor(Int, row)
    row1 = row0 + 1

    # Fractional parts
    fc = col - col0
    fr = row - row0

    # Clamp to valid range
    col0 = clamp(col0, 1, n_cols)
    col1 = clamp(col1, 1, n_cols)
    row0 = clamp(row0, 1, n_rows)
    row1 = clamp(row1, 1, n_rows)

    # Bilinear interpolation
    v00 = sinogram[col0, row0, angle_idx]
    v10 = sinogram[col1, row0, angle_idx]
    v01 = sinogram[col0, row1, angle_idx]
    v11 = sinogram[col1, row1, angle_idx]

    v0 = v00 * (1 - fc) + v10 * fc
    v1 = v01 * (1 - fc) + v11 * fc

    return v0 * (1 - fr) + v1 * fr
end

# =============================================================================
# Exports
# =============================================================================

export fdk_reconstruct
