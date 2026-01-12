"""
    Forward/Scatter.jl

Scatter simulation for CT using analytic kernel convolution.

Scatter in CT is caused by Compton and Rayleigh scattering of X-ray photons.
The scatter-to-primary ratio (SPR) depends on:
- Patient size and density
- Collimation and air gap
- Beam energy

This module implements a simple analytic scatter kernel that approximates
the scatter distribution as a low-frequency component of the projection.
"""

using FFTW

"""
    ScatterModel

Parameters for analytic scatter simulation.
"""
struct ScatterModel
    # Scatter-to-primary ratio (SPR) at center
    spr::Float64

    # Scatter kernel FWHM (in detector pixels)
    kernel_fwhm::Float64

    # Scatter kernel type (:gaussian or :exponential)
    kernel_type::Symbol
end

"""
    default_scatter_model(; spr=0.15, kernel_fwhm=50.0, kernel_type=:gaussian)

Create a scatter model with default parameters for body CT.

# Parameters
- `spr`: Scatter-to-primary ratio (typical: 0.1-0.3 for body CT)
- `kernel_fwhm`: Full-width half-maximum of scatter kernel in pixels
- `kernel_type`: Kernel shape (:gaussian or :exponential)

# Typical SPR values:
- Head CT: 0.05-0.10
- Body CT: 0.10-0.20
- Large patient: 0.20-0.40
"""
function default_scatter_model(;
    spr::Float64=0.15,
    kernel_fwhm::Float64=50.0,
    kernel_type::Symbol=:gaussian
)
    return ScatterModel(spr, kernel_fwhm, kernel_type)
end

"""
    create_scatter_kernel(model::ScatterModel, n_cols::Int, n_rows::Int) -> Array{Float64,2}

Create 2D scatter kernel for convolution.
"""
function create_scatter_kernel(model::ScatterModel, n_cols::Int, n_rows::Int)
    # Kernel size (use full projection dimensions for circular convolution)
    kernel = zeros(Float64, n_cols, n_rows)

    # Convert FWHM to sigma (Gaussian) or decay constant (exponential)
    sigma = model.kernel_fwhm / (2 * sqrt(2 * log(2)))

    # Center of kernel
    cx = (n_cols + 1) / 2
    cy = (n_rows + 1) / 2

    if model.kernel_type == :gaussian
        for j in 1:n_rows, i in 1:n_cols
            dx = i - cx
            dy = j - cy
            r2 = dx^2 + dy^2
            kernel[i, j] = exp(-r2 / (2 * sigma^2))
        end
    elseif model.kernel_type == :exponential
        decay = sigma  # Use sigma as decay constant
        for j in 1:n_rows, i in 1:n_cols
            dx = i - cx
            dy = j - cy
            r = sqrt(dx^2 + dy^2)
            kernel[i, j] = exp(-r / decay)
        end
    else
        error("Unknown kernel type: $(model.kernel_type)")
    end

    # Normalize kernel
    kernel ./= sum(kernel)

    return kernel
end

"""
    add_scatter(sinogram, model::ScatterModel) -> Array

Add scatter to sinogram using analytic kernel convolution.

The scatter contribution is computed as:
    scatter = SPR × convolve(primary, kernel)

where the kernel approximates the scatter point spread function.

# Arguments
- `sinogram`: Primary sinogram [n_cols × n_rows × n_angles]
- `model::ScatterModel`: Scatter model parameters

# Returns
Sinogram with scatter added (still in line-integral/attenuation space).
"""
function add_scatter(sinogram::AbstractArray{T,3}, model::ScatterModel) where T
    n_cols, n_rows, n_angles = size(sinogram)

    # Create scatter kernel
    kernel = create_scatter_kernel(model, n_cols, n_rows)

    # FFT of kernel (for convolution)
    kernel_fft = fft(kernel)

    # Output array
    sinogram_with_scatter = similar(sinogram)

    # Process each angle
    for angle in 1:n_angles
        # Get primary signal at this angle
        primary = sinogram[:, :, angle]

        # Convert from attenuation to intensity domain for scatter addition
        # I = exp(-attenuation)
        intensity = exp.(-primary)

        # Scatter is proportional to intensity, spread by kernel
        scatter_contribution = real(ifft(fft(intensity) .* kernel_fft))

        # Total intensity = primary + scatter
        total_intensity = intensity .+ T(model.spr) .* scatter_contribution

        # Clamp to positive values
        total_intensity = max.(total_intensity, T(1e-10))

        # Convert back to attenuation
        sinogram_with_scatter[:, :, angle] = -log.(total_intensity)
    end

    return sinogram_with_scatter
end

"""
    add_scatter!(sinogram, model::ScatterModel)

In-place version of add_scatter.
"""
function add_scatter!(sinogram::AbstractArray{T,3}, model::ScatterModel) where T
    result = add_scatter(sinogram, model)
    sinogram .= result
    return sinogram
end

"""
    estimate_spr(phantom::Phantom, geom::CTGeometry) -> Float64

Estimate scatter-to-primary ratio based on phantom size and geometry.

Larger phantoms and closer detector geometry produce more scatter.
"""
function estimate_spr(phantom::Phantom, geom::CTGeometry)
    # Approximate patient diameter (cm)
    patient_diameter = max(phantom.fov[1], phantom.fov[2])

    # Base SPR increases with patient size
    # Empirical formula from literature
    base_spr = 0.05 + 0.01 * patient_diameter

    # Air gap factor (larger SDD reduces scatter)
    # Normalized to typical 100 cm SDD
    air_gap_factor = 100.0 / geom.SDD

    spr = base_spr * air_gap_factor

    # Clamp to reasonable range
    return clamp(spr, 0.01, 0.5)
end

"""
    compute_scatter_artifact_magnitude(sinogram_clean, sinogram_scatter) -> Float64

Compute the magnitude of scatter artifacts as relative difference.
"""
function compute_scatter_artifact_magnitude(
    sinogram_clean::AbstractArray,
    sinogram_scatter::AbstractArray
)
    diff = abs.(sinogram_scatter .- sinogram_clean)
    return mean(diff) / mean(abs.(sinogram_clean) .+ 1e-10)
end

# =============================================================================
# Exports
# =============================================================================

export ScatterModel, default_scatter_model
export add_scatter, add_scatter!
export estimate_spr, compute_scatter_artifact_magnitude
