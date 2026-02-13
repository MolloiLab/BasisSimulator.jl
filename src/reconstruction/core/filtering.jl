# =============================================================================
# FDK Filtering (GPU-Native Spatial Domain)
# =============================================================================
#
# Implements CT reconstruction filtering entirely on GPU:
#   1. Cosine weighting for cone-beam geometry
#   2. Spatial domain ramp filter convolution (no FFT needed)
#   3. Optional filter windows (Shepp-Logan, Cosine, Hamming, Hann)
#
# Reference:
#   - Feldkamp, Davis, Kress (1984)
#   - Kak & Slaney, "Principles of Computerized Tomographic Imaging"
#   - Ram-Lak filter spatial domain form
#
# =============================================================================

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
# Spatial Domain Filter Kernel
# =============================================================================

"""
    create_spatial_kernel(n, filter_type, pixel_size)

Create ramp filter kernel in spatial domain.

The Ram-Lak (ramp) filter in spatial domain:
- h[0] = 1/(4Δ²)
- h[n] = 0 for even n ≠ 0
- h[n] = -1/(π²n²Δ²) for odd n

# Arguments
- `n`: Kernel size (typically n_cols)
- `filter_type`: Type of filter window
- `pixel_size`: Detector pixel spacing

# Returns
Filter kernel array of length n (centered at n÷2+1)
"""
function create_spatial_kernel(n::Int, filter_type::FilterType, pixel_size::T) where T <: AbstractFloat
    kernel = zeros(T, n)
    center = n ÷ 2 + 1
    Δ = pixel_size

    # Ram-Lak (ramp) filter in spatial domain
    # Reference: Kak & Slaney, "Principles of Computerized Tomographic Imaging"
    #
    # The discrete ramp filter kernel:
    # h[0] = 1/(4Δ²)
    # h[n] = 0 for even n ≠ 0
    # h[n] = -1/(π²n²Δ²) for odd n
    #
    # Additional scaling by Δ to match FFT-based filter normalization
    # (FFT version scales by 1/pixel_size, spatial version has 1/Δ² built in)

    for i in 1:n
        k = i - center  # Distance from center

        if k == 0
            # Central value: 1/(4Δ²) * Δ = 1/(4Δ)
            kernel[i] = one(T) / (T(4) * Δ)
        elseif k % 2 == 0
            # Even indices (excluding center)
            kernel[i] = zero(T)
        else
            # Odd indices: -1/(π²n²Δ²) * Δ = -1/(π²n²Δ)
            kernel[i] = -one(T) / (T(π)^2 * T(k)^2 * Δ)
        end
    end

    # Apply window function
    apply_spatial_window!(kernel, filter_type)

    return kernel
end

"""Apply windowing to spatial domain kernel"""
function apply_spatial_window!(kernel::Vector{T}, ::RampFilter) where T
    # No windowing for pure ramp
    return kernel
end

function apply_spatial_window!(kernel::Vector{T}, ::SheppLoganFilter) where T
    n = length(kernel)
    center = n ÷ 2 + 1
    for i in 1:n
        k = i - center
        if k != 0
            # Shepp-Logan: multiply by sinc(k/n) in spatial domain
            # This is approximate - exact would require convolution
            x = T(k) / T(n)
            kernel[i] *= sinc(x)
        end
    end
    return kernel
end

function apply_spatial_window!(kernel::Vector{T}, ::CosineFilter) where T
    n = length(kernel)
    center = n ÷ 2 + 1
    for i in 1:n
        k = i - center
        # Cosine window in spatial domain
        x = T(k) / T(n)
        kernel[i] *= cos(T(π) * x / T(2))^2
    end
    return kernel
end

function apply_spatial_window!(kernel::Vector{T}, ::HammingFilter) where T
    n = length(kernel)
    center = n ÷ 2 + 1
    for i in 1:n
        k = i - center
        x = T(k) / T(n)
        kernel[i] *= T(0.54) + T(0.46) * cos(T(π) * x)
    end
    return kernel
end

function apply_spatial_window!(kernel::Vector{T}, ::HannFilter) where T
    n = length(kernel)
    center = n ÷ 2 + 1
    for i in 1:n
        k = i - center
        x = T(k) / T(n)
        kernel[i] *= T(0.5) * (one(T) + cos(T(π) * x))
    end
    return kernel
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

    # Get dimensions as Int32 for GPU compatibility
    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))

    # Detector pixel positions relative to center (typed constants for GPU)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    magnification = T(geom.SDD / geom.SAD)
    SDD = T(geom.SDD)
    SDD_sq = SDD * SDD

    # Pre-compute center offsets for GPU
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    # Use AcceleratedKernels.jl for parallel cosine weighting
    AK.foreachindex(sinogram) do idx
        # Convert linear index to (col, row) using integer arithmetic (Int32 for GPU)
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)

        # Compute detector pixel position (u uses col pixel_size, v uses row pixel_size)
        u = (T(col) - col_center) * pixel_size * magnification
        v = (T(row) - row_center) * pixel_row_size * magnification

        # Distance from source to detector pixel
        dist = sqrt(SDD_sq + u^2 + v^2)

        # Cosine weight = SDD / dist
        weight = SDD / dist

        sinogram[idx] *= weight
    end

    return sinogram
end

# =============================================================================
# GPU-Native Spatial Domain Filtering
# =============================================================================

"""
    filter_sinogram!(sinogram, geom; filter=RampFilter(), cutoff=1.0)

Apply FDK filtering to sinogram in-place using GPU-native spatial domain convolution.

Steps:
1. Cosine weighting for cone-beam geometry
2. Row-by-row convolution with ramp filter kernel

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles] (modified in place)
- `geom`: CTGeometry with scanner parameters
- `filter`: Filter type (RampFilter, SheppLoganFilter, etc.)
- `cutoff`: Frequency cutoff (0-1) - controls kernel truncation

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
    cutoff::Float64 = 1.0,
    ws_conv_scratch = nothing,
    ws_filter_kernel = nothing
) where T <: AbstractFloat

    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Step 1: Cosine weighting
    cosine_weight!(sinogram, geom)

    # Step 2: Create spatial domain filter kernel
    pixel_size = T(geom.pixel_size)

    # Kernel size based on cutoff (smaller cutoff = smaller kernel = faster)
    # Compute without reassignment to avoid GPU boxing issues
    raw_size = max(Int(ceil(Int(n_cols) * cutoff)), 32)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), Int(n_cols))  # Make odd, clamp to n_cols

    # Use pre-allocated filter kernel if provided (zero-alloc path)
    kernel = if ws_filter_kernel !== nothing
        ws_filter_kernel
    else
        kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size)
        _k = similar(sinogram, T, kernel_size_int)
        copyto!(_k, kernel_cpu)
        _k
    end

    # Use Int32 constants for GPU
    kernel_size = Int32(kernel_size_int)
    kernel_half = Int32(kernel_size_int ÷ 2)

    # Use pre-allocated convolution scratch buffer if provided (zero-alloc path)
    filtered = ws_conv_scratch !== nothing ? ws_conv_scratch : similar(sinogram)

    # Step 3: Convolve each row with the filter kernel (GPU-parallel)
    AK.foreachindex(sinogram) do idx
        # Convert linear index to (col, row, angle)
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)
        angle = (idx_0 ÷ n_rows) + Int32(1)

        # Convolution for this pixel
        acc = zero(T)

        for k in Int32(1):kernel_size
            # Source column index (with boundary handling)
            k_offset = k - kernel_half - Int32(1)
            src_col = col + k_offset

            # Clamp to valid range (zero-padding at boundaries)
            if src_col >= Int32(1) && src_col <= n_cols
                acc += sinogram[src_col, row, angle] * kernel[k]
            end
        end

        filtered[idx] = acc
    end

    # Copy result back to sinogram
    copyto!(sinogram, filtered)

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
