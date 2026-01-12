"""
    Forward/DetectorNoise.jl

Detector response and noise modeling for CT simulation.

Implements:
1. Detector blur (point spread function)
2. Quantum noise (Poisson statistics)
3. Electronic noise (Gaussian additive)
"""

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

"""
    apply_detector_blur(sinogram, model::DetectorModel) -> Array

Apply detector blur (PSF convolution) to sinogram.
"""
function apply_detector_blur(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    if model.blur_fwhm <= 0.0
        return copy(sinogram)
    end

    n_cols, n_rows, n_angles = size(sinogram)

    # Create 2D Gaussian blur kernel
    sigma = model.blur_fwhm / (2 * sqrt(2 * log(2)))
    kernel_size = max(3, 2 * ceil(Int, 3 * sigma) + 1)

    kernel = zeros(Float64, kernel_size, kernel_size)
    center = (kernel_size + 1) / 2

    for j in 1:kernel_size, i in 1:kernel_size
        dx = i - center
        dy = j - center
        kernel[i, j] = exp(-(dx^2 + dy^2) / (2 * sigma^2))
    end
    kernel ./= sum(kernel)

    # Pad for convolution
    pad_x = kernel_size ÷ 2
    pad_y = kernel_size ÷ 2

    blurred = similar(sinogram)

    for angle in 1:n_angles
        # Pad image
        img = sinogram[:, :, angle]
        padded = zeros(T, n_cols + 2*pad_x, n_rows + 2*pad_y)

        # Fill padded region with edge values
        for j in 1:n_rows+2*pad_y, i in 1:n_cols+2*pad_x
            src_i = clamp(i - pad_x, 1, n_cols)
            src_j = clamp(j - pad_y, 1, n_rows)
            padded[i, j] = img[src_i, src_j]
        end

        # Convolve
        result = zeros(T, n_cols, n_rows)
        for j in 1:n_rows, i in 1:n_cols
            val = zero(T)
            for kj in 1:kernel_size, ki in 1:kernel_size
                pi = i + ki - 1
                pj = j + kj - 1
                val += padded[pi, pj] * T(kernel[ki, kj])
            end
            result[i, j] = val
        end

        blurred[:, :, angle] = result
    end

    return blurred
end

"""
    add_quantum_noise(sinogram, model::DetectorModel) -> Array

Add Poisson (quantum) noise to sinogram.

CT noise follows Poisson statistics: N(detected) ~ Poisson(I0 * exp(-attenuation))
"""
function add_quantum_noise(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    rng = isnothing(model.seed) ? Random.default_rng() : MersenneTwister(model.seed)

    # Convert attenuation to detected photon counts
    # I = I0 * exp(-attenuation)
    intensity = model.I0 .* exp.(-sinogram)

    # Apply Poisson noise
    # For high counts, use Gaussian approximation: N ~ Normal(λ, √λ)
    noisy_intensity = similar(intensity)
    for i in eachindex(intensity)
        λ = max(intensity[i], 1.0)  # Ensure positive
        if λ > 100
            # Gaussian approximation for large λ
            noisy_intensity[i] = λ + sqrt(λ) * randn(rng)
        else
            # Direct Poisson for small counts
            noisy_intensity[i] = Float64(rand(rng, Poisson(λ)))
        end
    end

    # Clamp to positive
    noisy_intensity = max.(noisy_intensity, T(1.0))

    # Convert back to attenuation
    noisy_sinogram = -log.(noisy_intensity ./ model.I0)

    return T.(noisy_sinogram)
end

"""
    add_electronic_noise(sinogram, model::DetectorModel) -> Array

Add Gaussian electronic noise to sinogram.
"""
function add_electronic_noise(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    if model.electronic_noise_std <= 0.0
        return copy(sinogram)
    end

    rng = isnothing(model.seed) ? Random.default_rng() : MersenneTwister(model.seed + 1)

    # Electronic noise is additive in detector signal space
    # Approximate by adding noise in attenuation space scaled appropriately
    noise_scale = model.electronic_noise_std / model.I0

    noisy = sinogram .+ T(noise_scale) .* randn(rng, T, size(sinogram))

    return noisy
end

"""
    apply_detector_model(sinogram, model::DetectorModel) -> Array

Apply full detector model: blur + quantum noise + electronic noise.
"""
function apply_detector_model(sinogram::AbstractArray{T,3}, model::DetectorModel) where T
    # Apply in order: blur -> quantum noise -> electronic noise
    result = apply_detector_blur(sinogram, model)
    result = add_quantum_noise(result, model)
    result = add_electronic_noise(result, model)
    return result
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
export apply_detector_blur, add_quantum_noise, add_electronic_noise
export apply_detector_model
export compute_noise_level, estimate_dose_from_noise
