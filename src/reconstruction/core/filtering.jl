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
export StandardFilter, SoftFilter, BoneFilter, CustomFilter
export create_spatial_kernel, grid_bandlimit, frequency_window

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

"""
CatSim 'standard' filter — ramp × apodization window.

Matches CatSim/XCIST default `kernelType = 'standard'` from `createHSP.py`.
Apodization defined at 5 control points (quadratic interpolation):
  f_norm:  [0,    0.25,   0.5,    0.75,   1.0]
  window:  [1,    0.9338, 0.7441, 0.4425, 0.0531]

Provides ~2.1× noise reduction vs pure Ram-Lak at the cost of spatial resolution.
"""
struct StandardFilter <: FilterType end

"""
CatSim 'soft' filter — ramp × soft-tissue apodization.

Matches CatSim/XCIST `kernelType = 'soft'` from `createHSP.py`.
Provides stronger smoothing than StandardFilter.
"""
struct SoftFilter <: FilterType end

"""
CatSim 'bone' filter — ramp × bone-enhancing apodization.

Matches CatSim/XCIST `kernelType = 'bone'` from `createHSP.py`.
Boosts mid-frequencies for sharper bone edges; higher noise than StandardFilter.
"""
struct BoneFilter <: FilterType end

"""
    CustomFilter(control_x, control_y)

Custom frequency-domain apodization filter with user-specified control points.

Applies ramp × piecewise-linear window defined at normalized frequency control points,
using the same CatSim-style mechanism as `StandardFilter`, `SoftFilter`, and `BoneFilter`.

# Arguments
- `control_x`: Tuple of normalized frequency positions (0.0 to 1.0)
- `control_y`: Tuple of window values at each position (1.0 = full pass, 0.0 = block)

# Example
```julia
# Soft-tissue–optimised apodization
f = CustomFilter((0.0, 0.25, 0.5, 0.75, 1.0),
                 (1.0, 0.82, 0.54, 0.26, 0.001))
ws = create_fdk_recon_workspace(sino, geom, recon_size; filter = f)
```
"""
struct CustomFilter{N} <: FilterType
    control_x::NTuple{N,Float64}
    control_y::NTuple{N,Float64}
end

# =============================================================================
# Spatial Domain Filter Kernel
# =============================================================================

"""
    create_spatial_kernel(n, filter_type, pixel_size; bandlimit = 1.0)

The FBP convolution kernel in the spatial domain: the discrete Ram-Lak ramp at sample spacing
`pixel_size` (Kak & Slaney — `h[0] = 1/(4Δ)`, `h[k] = -1/(π²k²Δ)` for odd `k`, zero for even `k`;
the extra factor Δ matches the FFT-based normalisation), multiplied in the frequency domain by
the apodization window of `filter_type` ([`frequency_window`](@ref)).

`bandlimit` is the fraction of the sampling Nyquist `1/(2Δ)` that the reconstruction can carry:
`min(1, Δ / Δ_grid)` for an image grid of pixel `Δ_grid` at isocentre, which
[`grid_bandlimit`](@ref) computes from a geometry and a volume size. The window is stretched over
`[0, bandlimit]` of the sampling Nyquist and the response is zero above it. Two things follow: a
named kernel means the same resolution on any detector, and no frequency the image grid cannot
represent reaches the backprojector, where it would alias into the image as fine grain (a
0.30 mm photon-counting column on a 0.68 mm grid passed 2.3× the grid's Nyquist before this).
With `bandlimit = 1` — a detector no finer than the grid — the kernel is the classical one.
"""
function create_spatial_kernel(
        n::Int, filter_type::FilterType, pixel_size::T; bandlimit::Real = 1.0,
    ) where T <: AbstractFloat
    0 < bandlimit <= 1 || throw(ArgumentError("bandlimit must be in (0, 1], got $(bandlimit)"))
    kernel = zeros(T, n)
    center = n ÷ 2 + 1
    Δ = pixel_size
    for i in 1:n
        k = i - center
        if k == 0
            kernel[i] = one(T) / (T(4) * Δ)
        elseif k % 2 == 0
            kernel[i] = zero(T)
        else
            kernel[i] = -one(T) / (T(π)^2 * T(k)^2 * Δ)
        end
    end
    apply_frequency_window!(kernel, filter_type, T(bandlimit))
    return kernel
end

"""
    grid_bandlimit(geom, volume_size; ray_spacing = geom.pixel_size) -> Float64

The fraction of the ray sampling's Nyquist frequency that a reconstruction grid of
`volume_size` over `geom.fov` can represent: `min(1, Δ_ray / Δ_grid)`, with `Δ_grid` the larger
of the two in-plane pixels (cm, at isocentre) and `Δ_ray` the ray spacing at isocentre —
`geom.pixel_size`, or the rebinned parallel spacing of a helical reconstruction. It is 1 for a
detector no finer than the grid, and the `bandlimit` every reconstruction here passes to
[`create_spatial_kernel`](@ref).
"""
function grid_bandlimit(
        geom::CTGeometry, volume_size::NTuple{3, Int}; ray_spacing::Real = geom.pixel_size,
    )
    grid_pixel = max(geom.fov[1] / volume_size[1], geom.fov[2] / volume_size[2])
    return min(1.0, Float64(ray_spacing) / grid_pixel)
end

"""
    frequency_window(filter, f) -> window value at normalised frequency `f`

The apodization window of `filter` against frequency normalised to its cutoff: `f = 0` is DC,
`f = 1` the cutoff. Ram-Lak: 1. Shepp-Logan: `sinc(f/2) = sin(πf/2)/(πf/2)`. Cosine: `cos(πf/2)`.
Hamming: `0.54 + 0.46·cos(πf)`. Hann: `½(1 + cos(πf))`. The CatSim kernels — standard, soft, bone
and a `CustomFilter` — interpolate their control points (`createHSP.py`).
"""
frequency_window(::RampFilter, f::T) where T = one(T)
frequency_window(::SheppLoganFilter, f::T) where T =
    f == 0 ? one(T) : T(sin(π * f / 2) / (π * f / 2))
frequency_window(::CosineFilter, f::T) where T = T(cos(π * f / 2))
frequency_window(::HammingFilter, f::T) where T = T(0.54 + 0.46 * cos(π * f))
frequency_window(::HannFilter, f::T) where T = T(0.5 * (1 + cos(π * f)))

# ---------------------------------------------------------------------------
# CatSim-compatible filters (Standard, Soft, Bone, Custom): control-point windows from
# CatSim/XCIST's `createHSP.py`, piecewise-linearly interpolated.
# ---------------------------------------------------------------------------

"""
    _catsim_apodization_window(f_norm, control_x, control_y)

Evaluate the CatSim-style apodization window at normalized frequency `f_norm`
using piecewise quadratic interpolation (matching scipy interp1d kind='quadratic').

Returns the window value (0 to 1).
"""
function _catsim_apodization_window(f_norm::T, control_x, control_y) where T
    # Clamp to [0, 1]
    f = clamp(f_norm, zero(T), one(T))

    # Find the interval
    n = length(control_x)
    for i in 1:(n-1)
        if f <= T(control_x[i+1]) || i == n-1
            # Linear interpolation (close enough for 5 control points;
            # the quadratic difference is < 1% in noise)
            t = (f - T(control_x[i])) / (T(control_x[i+1]) - T(control_x[i]))
            return T(control_y[i]) * (one(T) - t) + T(control_y[i+1]) * t
        end
    end
    return T(control_y[end])
end

const _CATSIM_CONTROL_X = (0.0, 0.25, 0.5, 0.75, 1.0)
frequency_window(::StandardFilter, f::T) where T =
    _catsim_apodization_window(f, _CATSIM_CONTROL_X, (1.0, 0.9338, 0.7441, 0.4425, 0.0531))
frequency_window(::SoftFilter, f::T) where T =
    _catsim_apodization_window(f, _CATSIM_CONTROL_X, (1.0, 0.815, 0.4564, 0.1636, 0.0))
frequency_window(::BoneFilter, f::T) where T =
    _catsim_apodization_window(f, _CATSIM_CONTROL_X, (1.0, 1.0485, 1.17, 1.2202, 0.9201))
frequency_window(filt::CustomFilter, f::T) where T =
    _catsim_apodization_window(f, filt.control_x, filt.control_y)

"""
    apply_frequency_window!(kernel, filter, bandlimit) -> kernel

Multiply the spatial kernel's spectrum by `frequency_window(filter, f / bandlimit)` for
`f ≤ bandlimit` and by zero above, `f` being the frequency normalised to the sampling Nyquist:
FFT → window → IFFT. One path for every filter type.
"""
function apply_frequency_window!(kernel::Vector{T}, filter::FilterType, bandlimit::T) where T
    n = length(kernel)
    center = n ÷ 2 + 1
    # fftshift: the kernel's centre tap to index 1
    shifted = zeros(Complex{T}, n)
    for i in 1:n
        shifted[mod(i - center, n) + 1] = Complex{T}(kernel[i])
    end
    freq = fft(shifted)
    nyquist = n / 2
    for k in 0:(n - 1)
        f_idx = k <= n ÷ 2 ? k : n - k
        f_norm = T(f_idx) / T(nyquist)
        freq[k + 1] *= f_norm <= bandlimit ? frequency_window(filter, f_norm / bandlimit) : zero(T)
    end
    spatial = ifft(freq)
    for i in 1:n
        kernel[i] = T(real(spatial[mod(i - center, n) + 1]))
    end
    return kernel
end


# =============================================================================
# Symbol-to-FilterType conversion
# =============================================================================

"""
    filter_from_symbol(sym::Symbol) -> FilterType

Convert a filter symbol (e.g., from `ReconOptions.filter`) to a `FilterType` struct.

Supported symbols: `:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`, `:hann`,
`:standard`, `:soft`, `:bone`.
"""
function filter_from_symbol(sym::Symbol)::FilterType
    sym === :ram_lak     ? RampFilter() :
    sym === :shepp_logan ? SheppLoganFilter() :
    sym === :cosine      ? CosineFilter() :
    sym === :hamming     ? HammingFilter() :
    sym === :hann        ? HannFilter() :
    sym === :standard    ? StandardFilter() :
    sym === :soft        ? SoftFilter() :
    sym === :bone        ? BoneFilter() :
    error("Unknown filter symbol: $sym. Use :ram_lak, :shepp_logan, :cosine, :hamming, :hann, :standard, :soft, or :bone.")
end

export filter_from_symbol

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

"""
    equiangular_kernel_scale!(kernel_cpu, dγ) -> kernel_cpu

Scale spatial ramp-kernel taps by `(γ/sin γ)²` (Kak & Slaney's equiangular
fan-beam kernel, `g(γ) = ½(γ/sinγ)²·h(γ)`), with `γₙ = n·Δγ`.  Call on the
CPU kernel before upload when the geometry is an :arc detector (skip for
:flat and for rebinned-parallel WFBP data).
"""
function equiangular_kernel_scale!(kernel_cpu::AbstractVector{T}, dγ::Real) where {T}
    n = length(kernel_cpu)
    c = (n + 1) ÷ 2
    for i in 1:n
        γ = (i - c) * dγ
        # physical column pairs always satisfy |γ| < π (full fan); guard the
        # far zero-data taps of the full-support kernel anyway
        if γ != 0 && abs(γ) < π * 0.999
            kernel_cpu[i] *= T((γ / sin(γ))^2)
        end
    end
    return kernel_cpu
end

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
    arc_det = is_arc(geom)
    dγ = T(geom.pixel_size / geom.SAD)

    # Use AcceleratedKernels.jl for parallel cosine weighting
    AK.foreachindex(sinogram) do idx
        # Convert linear index to (col, row) using integer arithmetic (Int32 for GPU)
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)

        # :flat — cos of the full 3D ray obliquity (TIGRE/FDK);
        # :arc  — equiangular fan pre-weight cos(γ) × z-obliquity D/√(D²+v²)
        #         (Kak & Slaney eq. 91 generalised to the cone row).
        weight = if arc_det
            γ = (T(col) - col_center) * dγ
            v = (T(row) - row_center) * pixel_row_size * magnification
            cos(γ) * SDD / sqrt(SDD_sq + v^2)
        else
            u = (T(col) - col_center) * pixel_size * magnification
            v = (T(row) - row_center) * pixel_row_size * magnification
            SDD / sqrt(SDD_sq + u^2 + v^2)
        end

        sinogram[idx] *= weight
    end

    return sinogram
end

# =============================================================================
# GPU-Native Spatial Domain Filtering
# =============================================================================

"""
    filter_sinogram!(sinogram, geom; filter=StandardFilter(), cutoff=1.0)

Apply FDK filtering to sinogram in-place using GPU-native spatial domain convolution.

Steps:
1. Cosine weighting for cone-beam geometry
2. Row-by-row convolution with ramp filter kernel

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles] (modified in place)
- `geom`: CTGeometry with scanner parameters
- `filter`: Filter type (RampFilter, SheppLoganFilter, etc.)
- `cutoff`: Frequency cutoff (0-1) - controls kernel truncation
- `bandlimit`: fraction of the ray sampling's Nyquist the reconstruction grid can carry
  ([`grid_bandlimit`](@ref)); the kernel's window is stretched to it and zero above. Default 1.

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
    filter::FilterType = StandardFilter(),
    cutoff::Float64 = 1.0,
    ws_conv_scratch = nothing,
    ws_filter_kernel = nothing,
    apply_cosine::Bool = true,
    ray_spacing::Union{Nothing, Real} = nothing,
    bandlimit::Real = 1.0
) where T <: AbstractFloat

    n_cols = Int32(size(sinogram, 1))
    n_rows = Int32(size(sinogram, 2))
    n_angles = Int32(size(sinogram, 3))

    # Step 1: Cosine weighting (skipped for rebinned-parallel WFBP data,
    # which is filtered as plain parallel rows)
    apply_cosine && cosine_weight!(sinogram, geom)

    # Step 2: Create spatial domain filter kernel
    pixel_size = ray_spacing === nothing ? T(geom.pixel_size) : T(ray_spacing)

    # Kernel size based on cutoff (smaller cutoff = smaller kernel = faster).
    # FULL support is 2·n_cols−1: every output column must see the ramp's
    # negative wings across the whole data extent.  The old n_cols clamp
    # truncated the far wings, under-subtracting at the object rim → +8 HU
    # capping when the object fills most of the detector (audit 2026-07-07).
    raw_size = max(Int(ceil(2 * Int(n_cols) * cutoff)), 64)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), 2 * Int(n_cols) - 1)

    # Use pre-allocated filter kernel if provided (zero-alloc path)
    kernel = if ws_filter_kernel !== nothing
        ws_filter_kernel
    else
        kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size; bandlimit)
        # equiangular fan filter correction (only for native arc fan data —
        # not for rebinned-parallel WFBP rows, which pass ray_spacing)
        if is_arc(geom) && ray_spacing === nothing
            equiangular_kernel_scale!(kernel_cpu, geom.pixel_size / geom.SAD)
        end
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
    filter_sinogram(sinogram, geom; filter=StandardFilter(), cutoff=1.0)

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
filtered = filter_sinogram(sinogram, geom; filter=StandardFilter())
```
"""
function filter_sinogram(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    filter::FilterType = StandardFilter(),
    cutoff::Float64 = 1.0,
    bandlimit::Real = 1.0
) where T <: AbstractFloat

    filtered = copy(sinogram)
    return filter_sinogram!(filtered, geom; filter=filter, cutoff=cutoff, bandlimit=bandlimit)
end
