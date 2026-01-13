"""
    Reconstruction/Kernels.jl

Reconstruction kernels (filters) for CT image reconstruction.

Reconstruction kernels control the trade-off between spatial resolution
and noise. Common kernels:
- Soft/Smooth: Reduces noise, lower resolution (body imaging)
- Standard: Balanced (general purpose)
- Bone/Sharp: Enhances edges, higher noise (bone/lung)

Implementation follows CatSim/XCIST approach with:
- Analytical kernels (Ram-Lak, Shepp-Logan)
- Lookup-table kernels (Soft, Standard, Bone)
"""

# =============================================================================
# Reconstruction Kernel Types
# =============================================================================

"""
    ReconKernel

Abstract type for reconstruction kernels.
"""
abstract type ReconKernel end

"""
    RampKernel <: ReconKernel

Standard Ram-Lak (ramp) filter.

Pure ramp filter without apodization.
Maximum spatial resolution but also maximum noise amplification.
"""
struct RampKernel <: ReconKernel
    cutoff::Float64  # Cutoff frequency as fraction of Nyquist (0-1)
end
RampKernel() = RampKernel(1.0)

"""
    SheppLoganKernel <: ReconKernel

Shepp-Logan filter.

Multiplies ramp by sinc function for mild smoothing.
"""
struct SheppLoganKernel <: ReconKernel
    cutoff::Float64
end
SheppLoganKernel() = SheppLoganKernel(1.0)

"""
    HammingKernel <: ReconKernel

Hamming-windowed ramp filter.

Applies Hamming window for moderate smoothing.
"""
struct HammingKernel <: ReconKernel
    cutoff::Float64
    alpha::Float64  # Hamming parameter (0.5 = Hann, 0.54 = Hamming)
end
HammingKernel() = HammingKernel(1.0, 0.54)

"""
    CosineKernel <: ReconKernel

Cosine-windowed ramp filter.

Smooth roll-off at high frequencies.
"""
struct CosineKernel <: ReconKernel
    cutoff::Float64
end
CosineKernel() = CosineKernel(1.0)

"""
    SoftKernel <: ReconKernel

Soft (smooth) reconstruction kernel.

Significantly reduces noise at the expense of spatial resolution.
Recommended for: soft tissue, body imaging, low-dose protocols.
"""
struct SoftKernel <: ReconKernel end

"""
    StandardKernel <: ReconKernel

Standard reconstruction kernel.

Balanced trade-off between noise and resolution.
Recommended for: general purpose imaging.
"""
struct StandardKernel <: ReconKernel end

"""
    BoneKernel <: ReconKernel

Bone (sharp) reconstruction kernel.

Enhances edges and fine detail at the expense of higher noise.
Recommended for: bone imaging, lung (high contrast structures).
"""
struct BoneKernel <: ReconKernel end

"""
    LungKernel <: ReconKernel

Lung reconstruction kernel.

Similar to bone but slightly smoother, optimized for lung parenchyma.
"""
struct LungKernel <: ReconKernel end

# =============================================================================
# Convenience Constructors
# =============================================================================

"""
    kernel_ramp(; cutoff=1.0)

Create Ram-Lak (ramp) filter.
"""
kernel_ramp(; cutoff::Float64=1.0) = RampKernel(cutoff)

"""
    kernel_shepp_logan(; cutoff=1.0)

Create Shepp-Logan filter.
"""
kernel_shepp_logan(; cutoff::Float64=1.0) = SheppLoganKernel(cutoff)

"""
    kernel_hamming(; cutoff=1.0, alpha=0.54)

Create Hamming-windowed filter.
"""
kernel_hamming(; cutoff::Float64=1.0, alpha::Float64=0.54) = HammingKernel(cutoff, alpha)

"""
    kernel_cosine(; cutoff=1.0)

Create cosine-windowed filter.
"""
kernel_cosine(; cutoff::Float64=1.0) = CosineKernel(cutoff)

"""
    kernel_soft()

Create soft (smooth) kernel for body imaging.
"""
kernel_soft() = SoftKernel()

"""
    kernel_standard()

Create standard kernel for general purpose.
"""
kernel_standard() = StandardKernel()

"""
    kernel_bone()

Create bone (sharp) kernel for bone/lung imaging.
"""
kernel_bone() = BoneKernel()

"""
    kernel_lung()

Create lung kernel optimized for lung parenchyma.
"""
kernel_lung() = LungKernel()

# =============================================================================
# Kernel Frequency Response
# =============================================================================

"""
    create_kernel_filter(kernel::ReconKernel, n_fft::Int, pixel_size::Float64) -> Vector{ComplexF64}

Create frequency-domain filter for reconstruction.

# Arguments
- `kernel::ReconKernel`: Kernel specification
- `n_fft::Int`: FFT size
- `pixel_size::Float64`: Detector pixel size in cm

# Returns
Complex frequency-domain filter for direct multiplication with FFT output.
"""
function create_kernel_filter(kernel::RampKernel, n_fft::Int, pixel_size::Float64)
    filter = zeros(ComplexF64, n_fft)
    freq_max = 1.0 / (2.0 * pixel_size)
    cutoff_freq = kernel.cutoff * freq_max

    for i in 1:n_fft
        freq_idx = i <= n_fft ÷ 2 + 1 ? i - 1 : i - 1 - n_fft
        freq_normalized = freq_idx / n_fft
        freq = freq_normalized / pixel_size

        if abs(freq) <= cutoff_freq
            filter[i] = abs(freq)
        end
    end

    return filter
end

function create_kernel_filter(kernel::SheppLoganKernel, n_fft::Int, pixel_size::Float64)
    filter = zeros(ComplexF64, n_fft)
    freq_max = 1.0 / (2.0 * pixel_size)
    cutoff_freq = kernel.cutoff * freq_max

    for i in 1:n_fft
        freq_idx = i <= n_fft ÷ 2 + 1 ? i - 1 : i - 1 - n_fft
        freq_normalized = freq_idx / n_fft
        freq = freq_normalized / pixel_size

        if abs(freq) <= cutoff_freq && abs(freq) > 0
            # Shepp-Logan: ramp * sinc(freq / (2 * freq_max))
            x = π * abs(freq) / (2 * freq_max)
            sinc_val = x != 0 ? sin(x) / x : 1.0
            filter[i] = abs(freq) * sinc_val
        elseif abs(freq) == 0
            filter[i] = 0.0
        end
    end

    return filter
end

function create_kernel_filter(kernel::HammingKernel, n_fft::Int, pixel_size::Float64)
    filter = zeros(ComplexF64, n_fft)
    freq_max = 1.0 / (2.0 * pixel_size)
    cutoff_freq = kernel.cutoff * freq_max
    α = kernel.alpha

    for i in 1:n_fft
        freq_idx = i <= n_fft ÷ 2 + 1 ? i - 1 : i - 1 - n_fft
        freq_normalized = freq_idx / n_fft
        freq = freq_normalized / pixel_size

        if abs(freq) <= cutoff_freq
            # Hamming window: α + (1-α)*cos(π*freq/freq_max)
            window = α + (1 - α) * cos(π * abs(freq) / freq_max)
            filter[i] = abs(freq) * window
        end
    end

    return filter
end

function create_kernel_filter(kernel::CosineKernel, n_fft::Int, pixel_size::Float64)
    filter = zeros(ComplexF64, n_fft)
    freq_max = 1.0 / (2.0 * pixel_size)
    cutoff_freq = kernel.cutoff * freq_max

    for i in 1:n_fft
        freq_idx = i <= n_fft ÷ 2 + 1 ? i - 1 : i - 1 - n_fft
        freq_normalized = freq_idx / n_fft
        freq = freq_normalized / pixel_size

        if abs(freq) <= cutoff_freq
            # Cosine window: cos(π*freq/(2*freq_max))
            window = cos(π * abs(freq) / (2 * freq_max))
            filter[i] = abs(freq) * window
        end
    end

    return filter
end

# Lookup-table based kernels (CatSim-style)
function create_kernel_filter(kernel::SoftKernel, n_fft::Int, pixel_size::Float64)
    # Soft kernel: heavily attenuates high frequencies
    # Control points: (normalized_freq, response) at [0, 0.25, 0.5, 0.75, 1.0]
    control_points = [1.0, 0.8, 0.4, 0.15, 0.0]
    return create_lookup_kernel(control_points, n_fft, pixel_size)
end

function create_kernel_filter(kernel::StandardKernel, n_fft::Int, pixel_size::Float64)
    # Standard kernel: moderate roll-off
    control_points = [1.0, 0.95, 0.75, 0.45, 0.1]
    return create_lookup_kernel(control_points, n_fft, pixel_size)
end

function create_kernel_filter(kernel::BoneKernel, n_fft::Int, pixel_size::Float64)
    # Bone kernel: enhances high frequencies
    control_points = [1.0, 1.05, 1.0, 0.85, 0.5]
    return create_lookup_kernel(control_points, n_fft, pixel_size)
end

function create_kernel_filter(kernel::LungKernel, n_fft::Int, pixel_size::Float64)
    # Lung kernel: between standard and bone
    control_points = [1.0, 1.0, 0.9, 0.65, 0.3]
    return create_lookup_kernel(control_points, n_fft, pixel_size)
end

"""
    create_lookup_kernel(control_points, n_fft, pixel_size)

Create frequency-domain filter using lookup table interpolation.

Control points define the frequency response at normalized frequencies
[0, 0.25, 0.5, 0.75, 1.0] relative to Nyquist.
"""
function create_lookup_kernel(control_points::Vector{Float64}, n_fft::Int, pixel_size::Float64)
    filter = zeros(ComplexF64, n_fft)
    freq_max = 1.0 / (2.0 * pixel_size)

    # Control frequencies (normalized 0-1)
    ctrl_freqs = [0.0, 0.25, 0.5, 0.75, 1.0]

    for i in 1:n_fft
        freq_idx = i <= n_fft ÷ 2 + 1 ? i - 1 : i - 1 - n_fft
        freq_normalized = freq_idx / n_fft
        freq = freq_normalized / pixel_size
        freq_rel = abs(freq) / freq_max  # Normalized to Nyquist

        if freq_rel <= 1.0
            # Interpolate control points
            apod = interpolate_quadratic(ctrl_freqs, control_points, freq_rel)

            # Apply to ramp filter
            filter[i] = abs(freq) * max(apod, 0.0)
        end
    end

    return filter
end

"""
    interpolate_quadratic(x_points, y_points, x)

Quadratic interpolation at point x using control points.
"""
function interpolate_quadratic(x_points::Vector{Float64}, y_points::Vector{Float64}, x::Float64)
    n = length(x_points)

    # Find the nearest three control points
    idx = 1
    for i in 1:(n-1)
        if x >= x_points[i] && x <= x_points[i+1]
            idx = i
            break
        end
    end

    # Get three points for quadratic fit
    if idx == 1
        i0, i1, i2 = 1, 2, 3
    elseif idx >= n - 1
        i0, i1, i2 = n-2, n-1, n
    else
        i0, i1, i2 = idx, idx+1, min(idx+2, n)
    end

    x0, x1, x2 = x_points[i0], x_points[i1], x_points[i2]
    y0, y1, y2 = y_points[i0], y_points[i1], y_points[i2]

    # Lagrange quadratic interpolation
    if x2 == x0  # Degenerate case
        return y0
    end

    L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2) + 1e-10)
    L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2) + 1e-10)
    L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1) + 1e-10)

    return y0 * L0 + y1 * L1 + y2 * L2
end

# =============================================================================
# Kernel Information
# =============================================================================

"""
    get_kernel_info(kernel::ReconKernel) -> NamedTuple

Get diagnostic information about reconstruction kernel.
"""
function get_kernel_info(kernel::RampKernel)
    return (type = "Ram-Lak (Ramp)", cutoff = kernel.cutoff,
            noise_level = "High", resolution = "Maximum")
end

function get_kernel_info(kernel::SheppLoganKernel)
    return (type = "Shepp-Logan", cutoff = kernel.cutoff,
            noise_level = "Medium-High", resolution = "High")
end

function get_kernel_info(kernel::HammingKernel)
    return (type = "Hamming", cutoff = kernel.cutoff, alpha = kernel.alpha,
            noise_level = "Medium", resolution = "Medium")
end

function get_kernel_info(kernel::CosineKernel)
    return (type = "Cosine", cutoff = kernel.cutoff,
            noise_level = "Medium-Low", resolution = "Medium")
end

function get_kernel_info(::SoftKernel)
    return (type = "Soft", noise_level = "Low", resolution = "Low",
            recommended_for = "Body, soft tissue, low-dose")
end

function get_kernel_info(::StandardKernel)
    return (type = "Standard", noise_level = "Medium", resolution = "Medium",
            recommended_for = "General purpose")
end

function get_kernel_info(::BoneKernel)
    return (type = "Bone", noise_level = "High", resolution = "High",
            recommended_for = "Bone, temporal bone, high contrast")
end

function get_kernel_info(::LungKernel)
    return (type = "Lung", noise_level = "Medium-High", resolution = "Medium-High",
            recommended_for = "Lung parenchyma, airways")
end

# =============================================================================
# Exports
# =============================================================================

export ReconKernel
export RampKernel, SheppLoganKernel, HammingKernel, CosineKernel
export SoftKernel, StandardKernel, BoneKernel, LungKernel
export kernel_ramp, kernel_shepp_logan, kernel_hamming, kernel_cosine
export kernel_soft, kernel_standard, kernel_bone, kernel_lung
export create_kernel_filter, get_kernel_info
