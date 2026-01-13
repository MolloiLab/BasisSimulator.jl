"""
    Forward/Scatter.jl

Scatter simulation for CT using analytic kernel convolution.

Scatter in CT is caused by Compton and Rayleigh scattering of X-ray photons.
The scatter contribution depends on:
- Patient size and density (path length)
- Primary signal intensity
- Collimation and air gap
- Beam energy

This module implements the convolution-based scatter model inspired by
Ohnesorge et al. (1999) and used in XCIST/GeCATSim:

    scatter = convolve(intensity × path_length × C, kernel)

where C is a calibration constant (~0.02-0.03) and kernel is a broad
low-frequency scatter point spread function.

Reference:
- Ohnesorge B, Flohr T, Klingenbeck-Regn K. "Efficient object scatter
  correction algorithm for third and fourth generation CT scanners."
  Eur Radiol. 1999;9(3):563-9.
- XCIST: https://github.com/xcist/main
"""

using FFTW

"""
    ScatterModel

Parameters for analytic scatter simulation using XCIST-style convolution model.

The scatter contribution at each detector pixel is:
    S = convolve(I × p × C × scale, kernel)

where:
- I = primary intensity (exp(-projection))
- p = path length (-log(I/I₀) = projection value)
- C = base scatter coefficient (~0.025)
- scale = user-adjustable scale factor
- kernel = scatter point spread function
"""
struct ScatterModel
    # Base scatter coefficient (XCIST uses ~0.025)
    scatter_coefficient::Float64

    # User-adjustable scale factor (1.0 = nominal scatter)
    scale_factor::Float64

    # Scatter kernel FWHM (in detector pixels)
    # Scatter is a broad, low-frequency signal (typical: 30-100 pixels)
    kernel_fwhm::Float64

    # Scatter kernel type (:gaussian or :exponential)
    kernel_type::Symbol
end

"""
    default_scatter_model(; scale_factor=1.0, kernel_fwhm=50.0, kernel_type=:gaussian)

Create a scatter model with default parameters for body CT.

# Parameters
- `scale_factor`: Multiplier for scatter magnitude (1.0 = ~15% SPR for body CT)
- `kernel_fwhm`: Full-width half-maximum of scatter kernel in pixels
- `kernel_type`: Kernel shape (:gaussian or :exponential)
- `spr`: DEPRECATED - use scale_factor instead. If provided, converted to scale_factor.

# Notes
The base scatter coefficient (0.025) is calibrated to produce approximately
15% scatter-to-primary ratio for a typical body phantom, matching XCIST defaults.

To increase/decrease scatter:
- scale_factor=0.5 → ~7.5% SPR (less scatter, e.g., pediatric)
- scale_factor=1.0 → ~15% SPR (nominal body CT)
- scale_factor=2.0 → ~30% SPR (large patient)
"""
function default_scatter_model(;
    scale_factor::Float64=1.0,
    kernel_fwhm::Float64=50.0,
    kernel_type::Symbol=:gaussian,
    spr::Union{Nothing,Float64}=nothing  # Deprecated parameter
)
    # Handle deprecated spr parameter
    if spr !== nothing
        # Convert old SPR to approximate scale_factor
        # Old model: SPR=0.15 was "nominal", so scale_factor ≈ spr/0.15
        scale_factor = spr / 0.15
    end

    # Base coefficient from XCIST (produces ~15% SPR)
    scatter_coefficient = 0.025

    return ScatterModel(scatter_coefficient, scale_factor, kernel_fwhm, kernel_type)
end

"""
    create_scatter_kernel(model::ScatterModel, n_cols::Int, n_rows::Int) -> Array{Float64,2}

Create 2D scatter kernel for FFT-based convolution.

The kernel is centered at (1,1) with wrap-around for proper FFT convolution.
"""
function create_scatter_kernel(model::ScatterModel, n_cols::Int, n_rows::Int)
    # Kernel size (use full projection dimensions for circular convolution)
    kernel = zeros(Float64, n_cols, n_rows)

    # Convert FWHM to sigma (Gaussian) or decay constant (exponential)
    sigma = model.kernel_fwhm / (2 * sqrt(2 * log(2)))

    # For FFT convolution, kernel must be centered at (1,1) with wrap-around
    # Use mod1 to handle wrap-around indexing
    if model.kernel_type == :gaussian
        for j in 1:n_rows, i in 1:n_cols
            # Distance from (1,1) with wrap-around
            dx = min(i - 1, n_cols - i + 1)
            dy = min(j - 1, n_rows - j + 1)
            r2 = dx^2 + dy^2
            kernel[i, j] = exp(-r2 / (2 * sigma^2))
        end
    elseif model.kernel_type == :exponential
        decay = sigma  # Use sigma as decay constant
        for j in 1:n_rows, i in 1:n_cols
            # Distance from (1,1) with wrap-around
            dx = min(i - 1, n_cols - i + 1)
            dy = min(j - 1, n_rows - j + 1)
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

Add scatter to sinogram using XCIST-style convolution model.

The scatter contribution is computed as (Ohnesorge et al., 1999; XCIST):
    scatter_pre = intensity × path_length × C × scale_factor
    scatter = convolve(scatter_pre, kernel)

This produces physically realistic scatter that:
- Is stronger for thicker objects (more material to scatter)
- Is weighted by the primary signal (more photons = more scatter)
- Creates appropriate cupping artifacts without excessive darkening

# Arguments
- `sinogram`: Primary sinogram [n_cols × n_rows × n_angles] (line integrals)
- `model::ScatterModel`: Scatter model parameters

# Returns
Sinogram with scatter added (still in line-integral/attenuation space).
"""
function add_scatter(sinogram::AbstractArray{T,3}, model::ScatterModel) where T
    n_cols, n_rows, n_angles = size(sinogram)

    # Create scatter kernel
    kernel = create_scatter_kernel(model, n_cols, n_rows)

    # FFT of kernel (for convolution via FFT)
    kernel_fft = fft(kernel)

    # Output array
    sinogram_with_scatter = similar(sinogram)

    # Combined scatter coefficient
    C = model.scatter_coefficient * model.scale_factor

    # Process each angle
    for angle in 1:n_angles
        # Get projection at this angle (path length in line-integral space)
        projection = sinogram[:, :, angle]

        # Convert to intensity domain: I = exp(-projection)
        # Clamp projection to avoid overflow for very large values
        clamped_proj = min.(projection, T(20))  # exp(-20) ≈ 2e-9
        intensity = exp.(-clamped_proj)

        # XCIST formula: scatter_pre = intensity × path_length × C
        # path_length is just the projection value (line integral = -log(I/I₀))
        # This makes scatter proportional to both signal AND object thickness
        scatter_pre = intensity .* projection .* C

        # Convolve with scatter kernel (broad, low-frequency PSF)
        scatter_contribution = real(ifft(fft(scatter_pre) .* kernel_fft))

        # Ensure scatter is non-negative
        scatter_contribution = max.(scatter_contribution, T(0))

        # Add scatter to intensity: I_total = I_primary + I_scatter
        total_intensity = intensity .+ scatter_contribution

        # Clamp to positive values before log
        total_intensity = max.(total_intensity, T(1e-10))

        # Convert back to projection domain
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
    estimate_scale_factor(phantom::Phantom, geom::CTGeometry) -> Float64

Estimate scatter scale factor based on phantom size and geometry.

Larger phantoms and closer detector geometry produce more scatter.

# Returns
Scale factor for use with default_scatter_model(scale_factor=...)
- 1.0 = nominal body CT (~15% SPR)
- <1.0 = less scatter (smaller patient, pediatric)
- >1.0 = more scatter (large patient)
"""
function estimate_scale_factor(phantom::Phantom, geom::CTGeometry)
    # Approximate patient diameter (cm)
    patient_diameter = max(phantom.fov[1], phantom.fov[2])

    # Reference diameter for scale_factor=1.0 (typical adult body: ~30 cm)
    reference_diameter = 30.0

    # Scale factor increases with patient size
    size_factor = patient_diameter / reference_diameter

    # Air gap factor (larger SDD reduces scatter)
    # Normalized to typical 100 cm SDD
    air_gap_factor = 100.0 / geom.SDD

    scale = size_factor * air_gap_factor

    # Clamp to reasonable range
    return clamp(scale, 0.1, 3.0)
end

# Deprecated alias
function estimate_spr(phantom::Phantom, geom::CTGeometry)
    # Return approximate SPR for backward compatibility
    # scale_factor=1.0 corresponds to ~15% SPR
    scale = estimate_scale_factor(phantom, geom)
    return 0.15 * scale
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
export estimate_scale_factor, estimate_spr, compute_scatter_artifact_magnitude
