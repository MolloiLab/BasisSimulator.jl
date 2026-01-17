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

GPU-native implementation using AcceleratedKernels.jl with spatial domain convolution.
"""

import AcceleratedKernels as AK

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

# Maximum scatter kernel size (controls quality vs performance)
# Scatter kernels are large (FWHM ~50 pixels) but we truncate at 3σ
const MAX_SCATTER_KERNEL_SIZE = 63

"""
    create_scatter_kernel_spatial(model::ScatterModel) -> Matrix{Float64}

Create 2D scatter kernel for spatial domain convolution.

Returns a compact kernel (max size MAX_SCATTER_KERNEL_SIZE) for GPU-compatible
spatial convolution. The kernel is truncated at 3σ.
"""
function create_scatter_kernel_spatial(model::ScatterModel)
    # Convert FWHM to sigma
    sigma = model.kernel_fwhm / (2 * sqrt(2 * log(2)))

    # Compute kernel extent (truncate at 3σ or max size)
    extent = min(MAX_SCATTER_KERNEL_SIZE ÷ 2, ceil(Int, 3 * sigma))
    kernel_size = 2 * extent + 1

    kernel = zeros(Float64, kernel_size, kernel_size)
    center = extent + 1

    if model.kernel_type == :gaussian
        for dy in -extent:extent
            for dx in -extent:extent
                r2 = dx^2 + dy^2
                kernel[center + dx, center + dy] = exp(-r2 / (2 * sigma^2))
            end
        end
    elseif model.kernel_type == :exponential
        decay = sigma
        for dy in -extent:extent
            for dx in -extent:extent
                r = sqrt(dx^2 + dy^2)
                kernel[center + dx, center + dy] = exp(-r / decay)
            end
        end
    else
        error("Unknown kernel type: $(model.kernel_type)")
    end

    # Normalize kernel
    total = sum(kernel)
    if total > 0
        kernel ./= total
    else
        kernel[center, center] = 1.0
    end

    return kernel
end

"""
    add_scatter!(sinogram, model::ScatterModel) -> sinogram

Add scatter to sinogram (in-place, GPU-native).

Uses spatial domain convolution with truncated kernel for GPU compatibility.

The scatter contribution is computed as (Ohnesorge et al., 1999; XCIST):
    scatter_pre = intensity × path_length × C × scale_factor
    scatter = convolve(scatter_pre, kernel)

# Arguments
- `sinogram`: Primary sinogram [n_cols × n_rows × n_angles] (line integrals)
- `model::ScatterModel`: Scatter model parameters

# Returns
Modified sinogram with scatter added.
"""
function add_scatter!(sinogram::AbstractArray{T,3}, model::ScatterModel) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Combined scatter coefficient
    C = T(model.scatter_coefficient * model.scale_factor)

    # Create scatter kernel on CPU
    kernel_cpu = T.(create_scatter_kernel_spatial(model))
    kernel_size = size(kernel_cpu, 1)
    half_k = kernel_size ÷ 2

    # Transfer kernel to GPU
    kernel = similar(sinogram, size(kernel_cpu)...)
    copyto!(kernel, kernel_cpu)

    # Output buffer
    output = similar(sinogram)

    # GPU-native scatter computation
    # For each pixel: compute scatter pre-signal, convolve, add to intensity
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Current projection value
        proj = sinogram[idx]

        # Convert to intensity (clamp projection to avoid overflow)
        clamped_proj = min(proj, T(20))
        intensity = exp(-clamped_proj)

        # Compute scatter contribution via spatial convolution
        # scatter = convolve(intensity × projection × C, kernel)
        scatter_acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                # Source projection
                src_proj = sinogram[src_col, src_row, angle]
                src_clamped = min(src_proj, T(20))
                src_intensity = exp(-src_clamped)

                # Scatter pre-signal at source pixel
                scatter_pre = src_intensity * src_proj * C

                # Kernel weight
                ki = di + half_k + 1
                kj = dj + half_k + 1

                scatter_acc += scatter_pre * kernel[ki, kj]
            end
        end

        # Add scatter to intensity
        total_intensity = intensity + max(scatter_acc, T(0))

        # Clamp and convert back to projection domain
        output[idx] = -log(max(total_intensity, T(1e-10)))
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates
function add_scatter(sinogram::AbstractArray{T,3}, model::ScatterModel) where T
    result = copy(sinogram)
    return add_scatter!(result, model)
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
# Scatter Correction
# =============================================================================

"""
    ScatterCorrectionModel

Parameters for scatter correction using CatSim-style convolution-based estimation.

The estimated scatter at each detector pixel is computed as:
    scatter_est = convolve(intensity × prep^α × C × scale, kernel)

where:
- intensity = measured intensity (after calibration)
- prep = -log(calibrated_ratio), the line integral estimate
- α = 0.9 (empirical exponent from CatSim)
- C = base correction coefficient (~0.0268)
- scale = configurable scale factor
- kernel = scatter point spread function

This scatter estimate is then subtracted from the measured intensity.

Reference: CatSim Scatter_Correction.py
"""
struct ScatterCorrectionModel
    # Base correction coefficient (CatSim uses 0.0268)
    correction_coefficient::Float64

    # User-adjustable scale factor (1.0 = nominal correction)
    scale_factor::Float64

    # Exponent for prep term (CatSim uses 0.9)
    prep_exponent::Float64

    # Scatter kernel FWHM (in detector pixels)
    kernel_fwhm::Float64

    # Scatter kernel type (:gaussian or :exponential)
    kernel_type::Symbol
end

"""
    default_scatter_correction(; scale_factor=1.0, kernel_fwhm=50.0)

Create a scatter correction model with default CatSim parameters.

# Parameters
- `scale_factor`: Multiplier for correction magnitude (1.0 = nominal)
- `kernel_fwhm`: Full-width half-maximum of scatter kernel in pixels

# Notes
Uses CatSim-exact parameters:
- correction_coefficient = 0.0268
- prep_exponent = 0.9

Adjust scale_factor to fine-tune correction strength:
- scale_factor < 1.0: Under-correction (residual cupping)
- scale_factor = 1.0: Nominal correction
- scale_factor > 1.0: Over-correction (may cause ring artifacts)
"""
function default_scatter_correction(;
    scale_factor::Float64=1.0,
    kernel_fwhm::Float64=50.0,
    kernel_type::Symbol=:gaussian
)
    # CatSim-exact parameters
    correction_coefficient = 0.0268
    prep_exponent = 0.9

    return ScatterCorrectionModel(
        correction_coefficient,
        scale_factor,
        prep_exponent,
        kernel_fwhm,
        kernel_type
    )
end

"""
    correct_scatter!(sinogram, model::ScatterCorrectionModel) -> sinogram

Apply scatter correction to sinogram (in-place, GPU-native).

Uses CatSim-style convolution-based scatter estimation and subtraction.

The algorithm (per view):
1. Compute prep = sinogram value (already in log domain)
2. Compute scatter_pre = exp(-prep) × prep^α × C × scale
3. Scatter estimate = convolve(scatter_pre, kernel)
4. Convert sinogram to intensity: I = exp(-sinogram)
5. Subtract scatter: I_corrected = I - scatter_estimate
6. Convert back: sinogram = -log(I_corrected)

# Arguments
- `sinogram`: Sinogram [n_cols × n_rows × n_angles] (line integrals, log domain)
- `model::ScatterCorrectionModel`: Correction model parameters

# Returns
Modified sinogram with scatter correction applied.

# Notes
- Reduces cupping artifacts in uniform phantoms
- Center-to-edge HU difference should be < 20 HU after correction
- Works in log domain (takes sinogram, applies correction, returns sinogram)

Reference: CatSim Scatter_Correction.py
"""
function correct_scatter!(sinogram::AbstractArray{T,3}, model::ScatterCorrectionModel) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    n_angles = size(sinogram, 3)

    # Combined correction coefficient
    C = T(model.correction_coefficient * model.scale_factor)
    alpha = T(model.prep_exponent)

    # Create scatter kernel on CPU
    # Reuse the same kernel creation as for scatter addition
    scatter_model_temp = ScatterModel(
        model.correction_coefficient,
        model.scale_factor,
        model.kernel_fwhm,
        model.kernel_type
    )
    kernel_cpu = T.(create_scatter_kernel_spatial(scatter_model_temp))
    kernel_size = size(kernel_cpu, 1)
    half_k = kernel_size ÷ 2

    # Transfer kernel to GPU
    kernel = similar(sinogram, size(kernel_cpu)...)
    copyto!(kernel, kernel_cpu)

    # Output buffer
    output = similar(sinogram)

    eps = T(1e-10)

    # GPU-native scatter correction
    # For each pixel: estimate scatter, subtract from intensity, convert back
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Current value (log domain = line integral)
        prep = sinogram[idx]

        # Convert to intensity
        clamped_prep = min(max(prep, T(0)), T(15))
        intensity = exp(-clamped_prep)

        # Compute scatter estimate via spatial convolution
        # scatter_est = convolve(exp(-prep) × prep^α × C, kernel)
        scatter_est = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                # Source prep value
                src_prep = sinogram[src_col, src_row, angle]
                src_clamped = min(max(src_prep, eps), T(15))
                src_intensity = exp(-src_clamped)

                # Scatter pre-signal: intensity × prep^α × C
                # CatSim: sc_preConv = phantomScan × prep^0.9 × 0.0268
                scatter_pre = src_intensity * (src_clamped ^ alpha) * C

                # Kernel weight
                ki = di + half_k + 1
                kj = dj + half_k + 1

                scatter_est += scatter_pre * kernel[ki, kj]
            end
        end

        # Subtract scatter estimate from intensity
        # Ensure result is positive
        corrected_intensity = max(intensity - scatter_est, eps)

        # Convert back to log domain
        output[idx] = -log(corrected_intensity)
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates
function correct_scatter(sinogram::AbstractArray{T,3}, model::ScatterCorrectionModel) where T
    result = copy(sinogram)
    return correct_scatter!(result, model)
end

"""
    measure_cupping(recon_hu, center_radius_frac=0.1, edge_radius_frac=0.8)

Measure cupping artifact as center-to-edge HU difference in a uniform phantom.

# Arguments
- `recon_hu`: Reconstruction in HU [nx, ny, nz]
- `center_radius_frac`: Fraction of image radius for center ROI (default: 0.1)
- `edge_radius_frac`: Fraction of image radius for edge ROI (default: 0.8)

# Returns
Named tuple with:
- `center_hu`: Mean HU in center ROI
- `edge_hu`: Mean HU in edge annulus
- `cupping_hu`: Center - Edge HU difference (positive = cupping, negative = doming)
- `center_std`: Standard deviation in center
- `edge_std`: Standard deviation in edge

# Notes
For a water phantom, cupping_hu should be < 20 HU after scatter correction.
"""
function measure_cupping(
    recon_hu::AbstractArray{T,3};
    center_radius_frac::Float64=0.1,
    edge_radius_frac::Float64=0.8
) where T
    nx, ny, nz = size(recon_hu)

    # Use central slice
    central_slice = nz ÷ 2
    slice = Array(recon_hu[:, :, central_slice])

    cx, cy = nx ÷ 2, ny ÷ 2
    max_radius = min(nx, ny) / 2

    center_radius = center_radius_frac * max_radius
    edge_inner = edge_radius_frac * max_radius
    edge_outer = 0.95 * max_radius  # Leave small margin

    center_vals = T[]
    edge_vals = T[]

    for j in 1:ny
        for i in 1:nx
            r = sqrt((i - cx)^2 + (j - cy)^2)
            val = slice[i, j]

            # Skip air (outside phantom)
            if val < -500
                continue
            end

            if r <= center_radius
                push!(center_vals, val)
            elseif edge_inner <= r <= edge_outer
                push!(edge_vals, val)
            end
        end
    end

    if isempty(center_vals) || isempty(edge_vals)
        return (
            center_hu = T(NaN),
            edge_hu = T(NaN),
            cupping_hu = T(NaN),
            center_std = T(NaN),
            edge_std = T(NaN)
        )
    end

    center_hu = mean(center_vals)
    edge_hu = mean(edge_vals)

    return (
        center_hu = center_hu,
        edge_hu = edge_hu,
        cupping_hu = center_hu - edge_hu,  # Positive = cupping
        center_std = std(center_vals),
        edge_std = std(edge_vals)
    )
end

# =============================================================================
# Exports
# =============================================================================

export ScatterModel, default_scatter_model
export create_scatter_kernel_spatial
export add_scatter!, add_scatter
export estimate_scale_factor, estimate_spr, compute_scatter_artifact_magnitude
export ScatterCorrectionModel, default_scatter_correction
export correct_scatter!, correct_scatter
export measure_cupping
