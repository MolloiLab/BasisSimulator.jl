"""
    Forward/DetectorNoise.jl

Detector response and noise modeling for CT simulation.

Implements:
1. Detector blur (point spread function) - Phase 3 (convolution)
2. Quantum noise (Poisson statistics) - GPU-native with Gaussian approximation
3. Electronic noise (Gaussian additive) - GPU-native

GPU-native implementation using AcceleratedKernels.jl.

For GPU, random numbers are pre-generated on CPU and transferred to GPU.
This is a common pattern for simulation work where reproducibility is important.
"""

import AcceleratedKernels as AK
using Random

"""
    DetectorModel

Parameters for detector response simulation.
"""
struct DetectorModel
    # Detector MTF (blur) - FWHM in pixels
    blur_fwhm::Float64

    # Photon count at detector (affects quantum noise)
    # Higher = less noise, more dose
    I0::Float64

    # Electronic noise standard deviation (in detector units)
    electronic_noise_std::Float64

    # Random seed (for reproducibility)
    seed::Union{Nothing, Int}
end

"""
    default_detector_model(; blur_fwhm=1.5, I0=1e5, electronic_noise_std=10.0, seed=nothing)

Create detector model with default parameters.

# Parameters
- `blur_fwhm`: Detector PSF FWHM in pixels (typical: 1.0-2.0)
- `I0`: Incident photon count (typical: 1e4-1e6)
- `electronic_noise_std`: Electronic noise σ (typical: 5-20)
- `seed`: Random seed for reproducibility
"""
function default_detector_model(;
    blur_fwhm::Float64=1.5,
    I0::Float64=1e5,
    electronic_noise_std::Float64=10.0,
    seed::Union{Nothing, Int}=nothing
)
    return DetectorModel(blur_fwhm, I0, electronic_noise_std, seed)
end

# Maximum kernel size for detector blur
const MAX_DETECTOR_BLUR_KERNEL_SIZE = 15

"""
    apply_detector_blur!(sinogram, model::DetectorModel) -> sinogram

Apply detector blur (PSF convolution) to sinogram (in-place, GPU-native).

Uses spatial domain convolution for GPU compatibility.
"""
function apply_detector_blur!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    if model.blur_fwhm <= 0.0
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Create 2D Gaussian blur kernel on CPU
    sigma = model.blur_fwhm / (2 * sqrt(2 * log(2)))
    extent = min(MAX_DETECTOR_BLUR_KERNEL_SIZE ÷ 2, max(1, ceil(Int, 3 * sigma)))
    kernel_size = 2 * extent + 1
    half_k = extent

    kernel_cpu = zeros(T, kernel_size, kernel_size)
    center = extent + 1

    for dy in -extent:extent
        for dx in -extent:extent
            kernel_cpu[center + dx, center + dy] = exp(-(dx^2 + dy^2) / (2 * sigma^2))
        end
    end
    kernel_cpu ./= sum(kernel_cpu)

    # Transfer kernel to GPU
    kernel = similar(sinogram, kernel_size, kernel_size)
    copyto!(kernel, kernel_cpu)

    # Output buffer
    output = similar(sinogram)

    # GPU-native spatial convolution
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Apply kernel
        acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                ki = di + half_k + 1
                kj = dj + half_k + 1

                acc += sinogram[src_col, src_row, angle] * kernel[ki, kj]
            end
        end

        output[idx] = acc
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates
function apply_detector_blur(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return apply_detector_blur!(result, model)
end

"""
    add_quantum_noise!(sinogram, model::DetectorModel) -> sinogram

Add Poisson (quantum) noise to sinogram (in-place, GPU-native).

CT noise follows Poisson statistics: N(detected) ~ Poisson(I0 * exp(-attenuation))

Uses Gaussian approximation for all counts (valid for typical CT counts >> 100).
This is standard practice in CT simulation as counts are typically > 10^4.
"""
function add_quantum_noise!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    rng = isnothing(model.seed) ? Random.default_rng() : MersenneTwister(model.seed)

    n_elements = length(sinogram)
    I0 = T(model.I0)

    # Pre-generate Gaussian random numbers on CPU
    rand_cpu = randn(rng, T, n_elements)

    # Transfer to GPU (same type as sinogram)
    rand_gpu = similar(sinogram, n_elements)
    copyto!(rand_gpu, rand_cpu)

    # GPU-native noise application
    # Poisson noise with Gaussian approximation:
    # λ = I0 * exp(-sinogram), noisy_λ = λ + sqrt(λ) * randn
    # noisy_sinogram = -log(noisy_λ / I0)
    AK.foreachindex(sinogram) do idx
        # Convert to intensity (λ)
        λ = I0 * exp(-sinogram[idx])

        # Apply Gaussian approximation of Poisson noise
        # σ = sqrt(λ) for Poisson distribution
        λ_noisy = λ + sqrt(max(λ, T(1))) * rand_gpu[idx]

        # Clamp to positive
        λ_noisy = max(λ_noisy, T(1))

        # Convert back to attenuation
        sinogram[idx] = -log(λ_noisy / I0)
    end

    return sinogram
end

"""
    add_electronic_noise!(sinogram, model::DetectorModel) -> sinogram

Add Gaussian electronic noise to sinogram (in-place, GPU-native).

Electronic noise is additive in detector signal space, approximated
as additive noise in attenuation space scaled by I0.
"""
function add_electronic_noise!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    if model.electronic_noise_std <= 0.0
        return sinogram
    end

    rng = isnothing(model.seed) ? Random.default_rng() : MersenneTwister(model.seed + 1)

    n_elements = length(sinogram)
    noise_scale = T(model.electronic_noise_std / model.I0)

    # Pre-generate Gaussian random numbers on CPU
    rand_cpu = randn(rng, T, n_elements)

    # Transfer to GPU (same type as sinogram)
    rand_gpu = similar(sinogram, n_elements)
    copyto!(rand_gpu, rand_cpu)

    # GPU-native noise application
    AK.foreachindex(sinogram) do idx
        sinogram[idx] += noise_scale * rand_gpu[idx]
    end

    return sinogram
end

# Convenience wrappers that allocate (for backward compatibility during transition)
function add_quantum_noise(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return add_quantum_noise!(result, model)
end

function add_electronic_noise(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return add_electronic_noise!(result, model)
end

"""
    apply_detector_model!(sinogram, model::DetectorModel) -> sinogram

Apply full detector model in-place: blur + quantum noise + electronic noise (GPU-native).

All effects are GPU-native using AcceleratedKernels.jl.
"""
function apply_detector_model!(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    # Apply in order: blur -> quantum noise -> electronic noise
    # All operations are now GPU-native
    apply_detector_blur!(sinogram, model)
    add_quantum_noise!(sinogram, model)
    add_electronic_noise!(sinogram, model)
    return sinogram
end

# Convenience wrapper that allocates
function apply_detector_model(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    result = copy(sinogram)
    return apply_detector_model!(result, model)
end

"""
    compute_noise_level(sinogram_clean, sinogram_noisy) -> NamedTuple

Compute noise statistics comparing clean and noisy sinograms.
"""
function compute_noise_level(sinogram_clean, sinogram_noisy)
    diff = sinogram_noisy .- sinogram_clean
    return (
        mean_diff = mean(diff),
        std_diff = std(diff),
        max_diff = maximum(abs.(diff)),
        snr = mean(abs.(sinogram_clean)) / (std(diff) + 1e-10)
    )
end

"""
    estimate_dose_from_noise(noise_std::Float64, μ_water::Float64) -> Float64

Estimate relative dose level from noise standard deviation.
Lower noise = higher dose.
"""
function estimate_dose_from_noise(noise_std::Float64, μ_water::Float64)
    # Rough estimate: noise_std ∝ 1/√(dose)
    # Normalize to typical clinical noise level
    clinical_noise = 0.02 * μ_water  # ~2% noise at clinical dose
    return (clinical_noise / max(noise_std, 1e-10))^2
end

# =============================================================================
# Poisson Distribution (simple implementation)
# =============================================================================

struct Poisson
    λ::Float64
end

function Base.rand(rng::AbstractRNG, p::Poisson)
    # Knuth algorithm for small λ
    L = exp(-p.λ)
    k = 0
    prob = 1.0

    while prob > L
        k += 1
        prob *= rand(rng)
    end

    return k - 1
end

# =============================================================================
# Exports
# =============================================================================

export DetectorModel, default_detector_model
export apply_detector_blur!, apply_detector_blur
export add_quantum_noise!, add_electronic_noise!
export add_quantum_noise, add_electronic_noise
export apply_detector_model!, apply_detector_model
export compute_noise_level, estimate_dose_from_noise
