# =============================================================================
# FDK Filtering (Ramp Filter + Cosine Weighting)
# =============================================================================
#
# Implements standard CT reconstruction filtering:
#   1. Cosine weighting for cone-beam geometry
#   2. Ramp filter (Ram-Lak) in Fourier domain
#   3. Optional filter windows (Shepp-Logan, Cosine, Hamming, Hann)
#
# Reference:
#   - Feldkamp, Davis, Kress (1984)
#   - TIGRE filter options: ram_lak, shepp_logan, cosine, hamming, hann
#
# =============================================================================

using FFTW
import AcceleratedKernels as AK

export filter_sinogram!, filter_sinogram
export FilterType, RampFilter, SheppLoganFilter, CosineFilter, HammingFilter, HannFilter

# =============================================================================
# Filter Types
# =============================================================================

abstract type FilterType end

"""Ram-Lak (ramp) filter - standard FDK filter"""
struct RampFilter <: FilterType end

"""Shepp-Logan filter - ramp × sinc(f/2f_max)"""
struct SheppLoganFilter <: FilterType end

"""Cosine filter - ramp × cos(πf/2f_max)"""
struct CosineFilter <: FilterType end

"""Hamming filter - ramp × (0.54 + 0.46cos(πf/f_max))"""
struct HammingFilter <: FilterType end

"""Hann filter - ramp × 0.5(1 + cos(πf/f_max))"""
struct HannFilter <: FilterType end

# =============================================================================
# Filter Construction
# =============================================================================

"""
    create_ramp_filter(n, filter_type, cutoff=1.0)

Create a ramp filter in the frequency domain.

# Arguments
- `n`: Number of frequency bins (typically padded detector width)
- `filter_type`: Type of filter window (RampFilter, SheppLoganFilter, etc.)
- `cutoff`: Frequency cutoff as fraction of Nyquist (0-1)

# Returns
Filter array of length n (frequency domain)
"""
function create_ramp_filter(n::Int, filter_type::FilterType, cutoff::Float64=1.0)
    # Frequency axis (normalized to Nyquist)
    freq = fftfreq(n)

    # Ramp filter: |f|
    ramp = abs.(freq)

    # Apply cutoff
    ramp[abs.(freq) .> cutoff / 2] .= 0

    # Apply window function
    filter = apply_filter_window(ramp, freq, filter_type)

    return Float32.(filter)
end

"""
    apply_filter_window(ramp, freq, filter_type)

Apply windowing function to ramp filter.
"""
function apply_filter_window(ramp, freq, ::RampFilter)
    return ramp
end

function apply_filter_window(ramp, freq, ::SheppLoganFilter)
    # sinc(f / (2 * f_max)) where f_max = 0.5
    window = sinc.(freq)
    return ramp .* window
end

function apply_filter_window(ramp, freq, ::CosineFilter)
    # cos(π * f / (2 * f_max))
    window = cos.(π .* freq)
    return ramp .* window
end

function apply_filter_window(ramp, freq, ::HammingFilter)
    # 0.54 + 0.46 * cos(π * f / f_max)
    window = 0.54 .+ 0.46 .* cos.(2π .* freq)
    return ramp .* window
end

function apply_filter_window(ramp, freq, ::HannFilter)
    # 0.5 * (1 + cos(π * f / f_max))
    window = 0.5 .* (1 .+ cos.(2π .* freq))
    return ramp .* window
end

# =============================================================================
# Cosine Weighting for Cone-Beam Geometry
# =============================================================================

"""
    cosine_weight!(sinogram, geom)

Apply cosine weighting for cone-beam FDK reconstruction.

This pre-weights the projections to account for the cone-beam geometry
before filtering and backprojection.

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles] (modified in place)
- `geom`: CTGeometry with scanner parameters

# Reference
Feldkamp, Davis, Kress (1984) Eq. 5
"""
function cosine_weight!(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(sinogram)

    # Detector pixel positions relative to center (typed constants for GPU)
    pixel_size = T(geom.pixel_size)
    magnification = T(geom.SDD / geom.SAD)
    SDD = T(geom.SDD)
    SDD_sq = SDD * SDD

    # Pre-compute center offsets for GPU
    col_center = T((n_cols + 1) / 2)
    row_center = T((n_rows + 1) / 2)

    # Use AcceleratedKernels.jl for parallel cosine weighting
    AK.foreachindex(sinogram) do idx
        # Convert linear index to (col, row, angle) using integer arithmetic
        idx_0 = idx - 1
        col = (idx_0 % n_cols) + 1
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + 1
        # angle = (idx_0 ÷ n_rows) + 1  # not needed for cosine weight

        # Compute detector pixel position
        u = (T(col) - col_center) * pixel_size * magnification
        v = (T(row) - row_center) * pixel_size * magnification

        # Distance from source to detector pixel
        dist = sqrt(SDD_sq + u^2 + v^2)

        # Cosine weight = SDD / dist
        weight = SDD / dist

        sinogram[idx] *= weight
    end

    return sinogram
end

# =============================================================================
# Main Filtering Function
# =============================================================================

"""
    filter_sinogram!(sinogram, geom; filter=RampFilter(), cutoff=1.0)

Apply FDK filtering to sinogram in-place.

Steps:
1. Cosine weighting for cone-beam geometry
2. Row-by-row FFT filtering with ramp filter

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles] (modified in place)
- `geom`: CTGeometry with scanner parameters
- `filter`: Filter type (RampFilter, SheppLoganFilter, etc.)
- `cutoff`: Frequency cutoff as fraction of Nyquist (0-1)

# Returns
The filtered sinogram (modified in place)

# Example
```julia
filter_sinogram!(sinogram, geom; filter=SheppLoganFilter(), cutoff=0.8)
```
"""
function filter_sinogram!(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(sinogram)

    # Step 1: Cosine weighting
    cosine_weight!(sinogram, geom)

    # Step 2: Create filter (zero-padded for FFT)
    # Pad to next power of 2 for efficient FFT
    n_padded = nextpow(2, 2 * n_cols)
    filt = create_ramp_filter(n_padded, filter, cutoff)

    # Scale filter by detector spacing for proper units
    pixel_size = T(geom.pixel_size)
    filt .*= T(2 * n_cols / n_padded) / pixel_size

    # Step 3: Filter each row using FFT
    # Pre-allocate padded array
    padded = zeros(Complex{T}, n_padded)

    for angle in 1:n_angles
        for row in 1:n_rows
            # Zero-pad row
            fill!(padded, zero(Complex{T}))
            for col in 1:n_cols
                padded[col] = Complex{T}(sinogram[col, row, angle])
            end

            # FFT
            fft!(padded)

            # Apply filter
            padded .*= filt

            # Inverse FFT
            ifft!(padded)

            # Copy back (real part only)
            for col in 1:n_cols
                sinogram[col, row, angle] = T(real(padded[col]))
            end
        end
    end

    return sinogram
end

"""
    filter_sinogram(sinogram, geom; filter=RampFilter(), cutoff=1.0)

Allocating version of FDK filtering.

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `filter`: Filter type (RampFilter, SheppLoganFilter, etc.)
- `cutoff`: Frequency cutoff as fraction of Nyquist (0-1)

# Returns
New filtered sinogram

# Example
```julia
filtered = filter_sinogram(sinogram, geom; filter=RampFilter())
```
"""
function filter_sinogram(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    filtered = copy(sinogram)
    return filter_sinogram!(filtered, geom; filter=filter, cutoff=cutoff)
end
