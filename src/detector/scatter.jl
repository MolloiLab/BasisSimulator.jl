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
    kernel_type::Symbol=:gaussian
)
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

# =============================================================================
# Separable 1D Scatter Kernel (SPEED-BUILD-002)
# =============================================================================

"""
    create_scatter_kernel_1d(model::ScatterModel) -> Vector{Float64}

Create 1D Gaussian scatter kernel for separable convolution.

For Gaussian kernels: exp(-(dx²+dy²)/(2σ²)) = exp(-dx²/(2σ²)) × exp(-dy²/(2σ²)).
The 2D kernel is the outer product of two identical 1D kernels.

Returns nothing for non-Gaussian kernels (fall back to 2D).
"""
function create_scatter_kernel_1d(model::ScatterModel)
    model.kernel_type == :gaussian || return nothing

    sigma = model.kernel_fwhm / (2 * sqrt(2 * log(2)))
    extent = min(MAX_SCATTER_KERNEL_SIZE ÷ 2, ceil(Int, 3 * sigma))
    kernel_size = 2 * extent + 1

    kernel_1d = zeros(Float64, kernel_size)
    center = extent + 1
    for dx in -extent:extent
        kernel_1d[center + dx] = exp(-dx^2 / (2 * sigma^2))
    end
    # Normalize the 1D kernel so that outer_product sums to 1
    # Since 2D kernel = k1d ⊗ k1d, normalize so sum(k1d)² = 1 → sum(k1d) = 1
    kernel_1d ./= sum(kernel_1d)

    return kernel_1d
end

"""
    _convolve_separable_h!(output, input, kernel_1d, n_cols, n_rows)

Horizontal 1D convolution pass via AK.foreachindex.
Convolves each row of each angle slice with the 1D kernel.
"""
function _convolve_separable_h!(
    output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    kernel_1d,
    n_cols::Int, n_rows::Int
) where T
    half_k = size(kernel_1d, 1) ÷ 2
    let inp = input, out = output, k1d = kernel_1d, nc = Int32(n_cols), nr = Int32(n_rows), hk = Int32(half_k)
        AK.foreachindex(output) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            acc = zero(T)
            for di in -hk:hk
                src_col = clamp(col + di, Int32(1), nc)
                @inbounds acc += inp[src_col, row, angle] * k1d[di + hk + Int32(1)]
            end
            @inbounds out[idx] = acc
        end
    end
    return output
end

"""
    _convolve_separable_v!(output, input, kernel_1d, n_cols, n_rows)

Vertical 1D convolution pass via AK.foreachindex.
Convolves each column of each angle slice with the 1D kernel.
"""
function _convolve_separable_v!(
    output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    kernel_1d,
    n_cols::Int, n_rows::Int
) where T
    half_k = size(kernel_1d, 1) ÷ 2
    let inp = input, out = output, k1d = kernel_1d, nc = Int32(n_cols), nr = Int32(n_rows), hk = Int32(half_k)
        AK.foreachindex(output) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            acc = zero(T)
            for dj in -hk:hk
                src_row = clamp(row + dj, Int32(1), nr)
                @inbounds acc += inp[col, src_row, angle] * k1d[dj + hk + Int32(1)]
            end
            @inbounds out[idx] = acc
        end
    end
    return output
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
function add_scatter!(sinogram::AbstractArray{T,3}, model::ScatterModel;
                      ws_output=nothing, ws_kernel=nothing,
                      ws_scatter_temp=nothing, ws_kernel_1d=nothing) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Combined scatter coefficient
    C = T(model.scatter_coefficient * model.scale_factor)

    # Output buffer (use workspace or allocate)
    output = ws_output !== nothing ? ws_output : similar(sinogram)

    # ─── SEPARABLE PATH (Gaussian kernels only) ────────────────────────
    if model.kernel_type == :gaussian
        # Get or create 1D kernel on GPU
        kernel_1d = if ws_kernel_1d !== nothing
            ws_kernel_1d
        else
            k1d_cpu = T.(create_scatter_kernel_1d(model))
            k1d = similar(sinogram, T, length(k1d_cpu))
            copyto!(k1d, k1d_cpu)
            k1d
        end

        # Temp buffer for intermediate separable convolution result
        scatter_temp = ws_scatter_temp !== nothing ? ws_scatter_temp : similar(sinogram)

        # Step 1: Compute pre-signal into output buffer (reuse as scratch)
        let sino = sinogram, pre = output, c = C
            AK.foreachindex(sino) do idx
                proj = sino[idx]
                clamped = min(proj, T(20))
                @inbounds pre[idx] = exp(-clamped) * proj * c
            end
        end

        # Step 2: Horizontal 1D convolution: output(pre-signal) → scatter_temp
        _convolve_separable_h!(scatter_temp, output, kernel_1d, n_cols, n_rows)

        # Step 3: Vertical 1D convolution: scatter_temp → output (now contains scatter)
        _convolve_separable_v!(output, scatter_temp, kernel_1d, n_cols, n_rows)

        # Step 4: Apply scatter to sinogram: intensity + scatter → log domain
        let sino = sinogram, scatter = output
            AK.foreachindex(sino) do idx
                proj = sino[idx]
                clamped = min(proj, T(20))
                intensity = exp(-clamped)
                total_intensity = intensity + max(scatter[idx], zero(T))
                @inbounds sino[idx] = -log(max(total_intensity, T(1e-10)))
            end
        end

        return sinogram
    end

    # ─── FALLBACK: 2D convolution (non-Gaussian kernels) ──────────────
    if ws_kernel !== nothing
        kernel = ws_kernel
        kernel_size = size(kernel, 1)
    else
        kernel_cpu = T.(create_scatter_kernel_spatial(model))
        kernel_size = size(kernel_cpu, 1)
        kernel = similar(sinogram, size(kernel_cpu)...)
        copyto!(kernel, kernel_cpu)
    end
    half_k = kernel_size ÷ 2

    let kernel = kernel, output = output, half_k = half_k, n_cols = n_cols, n_rows = n_rows, C = C
        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % Int32(n_cols)) + Int32(1)
            idx_0 = idx_0 ÷ Int32(n_cols)
            row = (idx_0 % Int32(n_rows)) + Int32(1)
            angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

            proj = sinogram[idx]
            clamped_proj = min(proj, T(20))
            intensity = exp(-clamped_proj)

            scatter_acc = zero(T)
            for dj in -half_k:half_k
                for di in -half_k:half_k
                    src_col = clamp(col + di, 1, n_cols)
                    src_row = clamp(row + dj, 1, n_rows)

                    src_proj = sinogram[src_col, src_row, angle]
                    src_clamped = min(src_proj, T(20))
                    src_intensity = exp(-src_clamped)
                    scatter_pre = src_intensity * src_proj * C

                    ki = di + half_k + 1
                    kj = dj + half_k + 1
                    scatter_acc += scatter_pre * kernel[ki, kj]
                end
            end

            total_intensity = intensity + max(scatter_acc, T(0))
            output[idx] = -log(max(total_intensity, T(1e-10)))
        end
    end

    copyto!(sinogram, output)
    return sinogram
end

# Convenience wrapper that allocates
function add_scatter(sinogram::AbstractArray{T,3}, model::ScatterModel) where T
    result = copy(sinogram)
    return add_scatter!(result, model)
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

Uses convolution-based scatter estimation and subtraction. The algorithm matches
`add_scatter!()` to ensure consistent scatter estimation for simulation scenarios.

The algorithm (per view):
1. Compute prep = sinogram value (already in log domain)
2. Compute scatter_pre = exp(-prep) × prep × C × scale
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
- Algorithm matches `add_scatter!()` for consistent simulation behavior.
  The `prep_exponent` field is kept for API compatibility but is now ignored
  in favor of linear (exponent=1.0) model matching scatter addition.
"""
function correct_scatter!(sinogram::AbstractArray{T,3}, model::ScatterCorrectionModel;
                          ws_output=nothing, ws_kernel=nothing,
                          ws_scatter_temp=nothing, ws_kernel_1d=nothing) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    n_angles = size(sinogram, 3)

    # Combined correction coefficient
    C = T(model.correction_coefficient * model.scale_factor)
    scatter_damping = T(0.85)
    eps = T(1e-10)

    # Output buffer (use workspace or allocate)
    output = ws_output !== nothing ? ws_output : similar(sinogram)

    # ─── SEPARABLE PATH (Gaussian kernels only) ────────────────────────
    if model.kernel_type == :gaussian
        kernel_1d = if ws_kernel_1d !== nothing
            ws_kernel_1d
        else
            scatter_model_temp = ScatterModel(model.correction_coefficient, model.scale_factor, model.kernel_fwhm, model.kernel_type)
            k1d_cpu = T.(create_scatter_kernel_1d(scatter_model_temp))
            k1d = similar(sinogram, T, length(k1d_cpu))
            copyto!(k1d, k1d_cpu)
            k1d
        end

        scatter_temp = ws_scatter_temp !== nothing ? ws_scatter_temp : similar(sinogram)

        # Step 1: Compute pre-signal into output buffer
        let sino = sinogram, pre = output, c = C, eps = eps
            AK.foreachindex(sino) do idx
                prep = sino[idx]
                clamped = min(max(prep, eps), T(20))
                @inbounds pre[idx] = exp(-clamped) * clamped * c
            end
        end

        # Step 2: Horizontal convolution → scatter_temp
        _convolve_separable_h!(scatter_temp, output, kernel_1d, n_cols, n_rows)

        # Step 3: Vertical convolution → output (scatter estimate)
        _convolve_separable_v!(output, scatter_temp, kernel_1d, n_cols, n_rows)

        # Step 4: Subtract damped scatter from intensity
        let sino = sinogram, scatter = output, damp = scatter_damping, eps = eps
            AK.foreachindex(sino) do idx
                prep = sino[idx]
                clamped = min(max(prep, zero(T)), T(20))
                intensity = exp(-clamped)
                corrected = max(intensity - scatter[idx] * damp, eps)
                @inbounds sino[idx] = -log(corrected)
            end
        end

        return sinogram
    end

    # ─── FALLBACK: 2D convolution (non-Gaussian kernels) ──────────────
    if ws_kernel !== nothing
        kernel = ws_kernel
        kernel_size = size(kernel, 1)
    else
        scatter_model_temp = ScatterModel(model.correction_coefficient, model.scale_factor, model.kernel_fwhm, model.kernel_type)
        kernel_cpu = T.(create_scatter_kernel_spatial(scatter_model_temp))
        kernel_size = size(kernel_cpu, 1)
        kernel = similar(sinogram, size(kernel_cpu)...)
        copyto!(kernel, kernel_cpu)
    end
    half_k = kernel_size ÷ 2

    let kernel = kernel, output = output, half_k = half_k, n_cols = n_cols, n_rows = n_rows, C = C, scatter_damping = scatter_damping, eps = eps
        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % Int32(n_cols)) + Int32(1)
            idx_0 = idx_0 ÷ Int32(n_cols)
            row = (idx_0 % Int32(n_rows)) + Int32(1)
            angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

            prep = sinogram[idx]
            clamped_prep = min(max(prep, T(0)), T(20))
            intensity = exp(-clamped_prep)

            scatter_est = zero(T)
            for dj in -half_k:half_k
                for di in -half_k:half_k
                    src_col = clamp(col + di, 1, n_cols)
                    src_row = clamp(row + dj, 1, n_rows)

                    src_prep = sinogram[src_col, src_row, angle]
                    src_clamped = min(max(src_prep, eps), T(20))
                    src_intensity = exp(-src_clamped)
                    scatter_pre = src_intensity * src_clamped * C

                    ki = di + half_k + 1
                    kj = dj + half_k + 1
                    scatter_est += scatter_pre * kernel[ki, kj]
                end
            end

            scatter_est_damped = scatter_est * scatter_damping
            corrected_intensity = max(intensity - scatter_est_damped, eps)
            output[idx] = -log(corrected_intensity)
        end
    end

    copyto!(sinogram, output)
    return sinogram
end

# Convenience wrapper that allocates
function correct_scatter(sinogram::AbstractArray{T,3}, model::ScatterCorrectionModel) where T
    result = copy(sinogram)
    return correct_scatter!(result, model)
end


# =============================================================================
# Geometry-Aware Scatter (adapts to scanner configuration)
# =============================================================================

# Reference geometry constants (CatSim/BasisSimulator defaults)
# These values define the baseline for which the scatter coefficient was calibrated

"""Reference source-to-isocenter distance (mm) for scatter calibration."""
const SCATTER_REF_SID_MM = 540.0

"""Reference source-to-detector distance (mm) for scatter calibration."""
const SCATTER_REF_SDD_MM = 950.0

"""Reference air gap (mm) = SDD - SID for scatter calibration."""
const SCATTER_REF_AIR_GAP_MM = SCATTER_REF_SDD_MM - SCATTER_REF_SID_MM  # 410.0

"""Reference detector pixel pitch (mm) for scatter calibration."""
const SCATTER_REF_PIXEL_PITCH_MM = 1.0

"""Base scatter coefficient calibrated for reference geometry (~15% SPR)."""
const SCATTER_REF_COEFFICIENT = 0.025

"""Physical scatter kernel FWHM at detector (mm) - approximately constant."""
const SCATTER_PHYSICAL_KERNEL_FWHM_MM = 50.0

"""Reference scatter correction coefficient (CatSim-exact)."""
const SCATTER_REF_CORRECTION_COEFFICIENT = 0.0268

# =============================================================================
# Phantom Size Scaling Constants
# =============================================================================

"""Reference phantom diameter (cm) for scatter calibration (adult body)."""
const SCATTER_REF_PHANTOM_DIAMETER_CM = 30.0

"""SPR scaling exponent for phantom diameter (empirical: 1.5-2.0)."""
const SCATTER_SIZE_SCALING_EXPONENT = 1.5

# =============================================================================
# Energy-Dependent Scatter Constants
# =============================================================================

"""Reference mean photon energy (keV) for scatter calibration (corresponds to ~120 kVp)."""
const SCATTER_REF_ENERGY_KEV = 60.0

"""
SPR energy scaling exponent (empirical, 0.5-1.0 range).

Derived from literature:
- At 80 kVp (~50 keV mean): SPR is ~1.2-1.5× higher than at 120 kVp
- At 140 kVp (~70 keV mean): SPR is ~0.85-0.9× of 120 kVp

Conservative value of 0.6 balances physics (Klein-Nishina) with empirical observations.
"""
const SCATTER_ENERGY_EXPONENT = 0.6

"""
    estimate_phantom_diameter_cm(mask::AbstractArray{<:Unsigned,3}, voxel_size_mm) -> Float64

Estimate effective phantom diameter from material mask.

Computes the effective diameter as sqrt(width × height) from the bounding box of
non-air voxels (region index > 0) in the central slice.

# Arguments
- `mask`: Material index volume [nx, ny, nz], where 0 = air
- `voxel_size_mm`: Tuple of voxel dimensions (dx, dy, dz) in mm

# Returns
Effective diameter in cm.

# Notes
Uses the effective diameter formula: d_eff = sqrt(AP × LAT), which is standard
in CT dosimetry (AAPM Task Group 220).
"""
function estimate_phantom_diameter_cm(
    mask::AbstractArray{<:Unsigned,3},
    voxel_size_mm::Union{NTuple{3,<:Real}, AbstractVector{<:Real}}
)
    # Convert to CPU if on GPU (bounding box computation is fast on CPU)
    mask_cpu = Array(mask)

    nx, ny, nz = size(mask_cpu)
    dx, dy, dz = voxel_size_mm[1], voxel_size_mm[2], voxel_size_mm[3]

    # Find bounding box of non-air voxels across all slices
    min_x, max_x = nx, 1
    min_y, max_y = ny, 1

    # Sample central slices to estimate size (faster than full volume)
    z_mid = nz ÷ 2
    z_range = max(1, z_mid - 5):min(nz, z_mid + 5)

    for z in z_range
        for y in 1:ny
            for x in 1:nx
                if mask_cpu[x, y, z] > 0
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                end
            end
        end
    end

    # Compute extents in mm
    if max_x >= min_x && max_y >= min_y
        width_mm = (max_x - min_x + 1) * dx
        height_mm = (max_y - min_y + 1) * dy

        # Effective diameter = sqrt(AP × LAT)
        effective_diameter_mm = sqrt(width_mm * height_mm)

        # Convert to cm
        return effective_diameter_mm / 10.0
    else
        # No non-air voxels found, return reference size
        return SCATTER_REF_PHANTOM_DIAMETER_CM
    end
end

"""
    compute_scatter_size_scale(phantom_diameter_cm::Real) -> Float64

Compute scatter scaling factor based on phantom/patient size.

Returns a multiplier for the scatter coefficient. Larger phantoms produce
more scatter than the reference 30 cm body.

# Scaling Formula
scale = (diameter / SCATTER_REF_PHANTOM_DIAMETER_CM)^SCATTER_SIZE_SCALING_EXPONENT
      = (diameter / 30)^1.5

# Typical Values
| Diameter | Scale | Description |
|----------|-------|-------------|
| 15 cm | 0.35 | Pediatric head |
| 18 cm | 0.47 | Adult head |
| 20 cm | 0.59 | Pediatric body |
| 30 cm | 1.00 | Adult body (reference) |
| 40 cm | 1.54 | Large body |
| 50 cm | 2.15 | Very large body |

# Example
```julia
scale = compute_scatter_size_scale(20.0)  # Pediatric body → 0.59
scale = compute_scatter_size_scale(40.0)  # Large body → 1.54
```
"""
function compute_scatter_size_scale(phantom_diameter_cm::Real)
    ratio = phantom_diameter_cm / SCATTER_REF_PHANTOM_DIAMETER_CM
    scale = ratio ^ SCATTER_SIZE_SCALING_EXPONENT

    # Clamp to reasonable range
    return clamp(scale, 0.1, 10.0)
end

"""
    compute_scatter_energy_scale(mean_energy_keV::Real) -> Float64

Compute scatter scaling factor based on mean photon energy.

Lower energies have higher scatter (more Compton interactions relative to primary,
and photoelectric absorption decreases as 1/E³).

# Scaling Formula
scale = (SCATTER_REF_ENERGY_KEV / mean_energy_keV)^SCATTER_ENERGY_EXPONENT
      = (60 / mean_energy_keV)^0.6

# Typical Values
| Mean Energy | Scale | Description |
|-------------|-------|-------------|
| 45 keV | 1.20 | ~80 kVp (high scatter) |
| 50 keV | 1.13 | ~80 kVp |
| 60 keV | 1.00 | ~120 kVp (reference) |
| 70 keV | 0.91 | ~140 kVp |
| 75 keV | 0.87 | ~140 kVp (low scatter) |

# Example
```julia
scale = compute_scatter_energy_scale(50.0)  # 80 kVp → 1.13
scale = compute_scatter_energy_scale(70.0)  # 140 kVp → 0.91
```

# References
- PMC2674384: SPR decreases when x-ray kVp increases
- PMC8611284: SPRmax inversely proportional to beam energy
- Klein-Nishina formula: Compton cross-section slowly decreases with energy
"""
function compute_scatter_energy_scale(mean_energy_keV::Real)
    ratio = SCATTER_REF_ENERGY_KEV / mean_energy_keV
    scale = ratio ^ SCATTER_ENERGY_EXPONENT

    # Clamp to reasonable range (0.5 to 2.0)
    # Prevents extreme values at very low or high energies
    return clamp(scale, 0.5, 2.0)
end


"""
    compute_scatter_geometry_scale(scanner::Scanner) -> Float64

Compute scatter scaling factor based on scanner geometry relative to reference.

Returns a multiplier for the base scatter coefficient. Values < 1.0 indicate
less scatter than reference (e.g., larger air gap), values > 1.0 indicate
more scatter than reference.

# Physics
Scatter intensity at detector scales approximately with (air_gap_ref/air_gap)²
due to the geometric divergence of scattered photons. This inverse-square
behavior is consistent with Monte Carlo studies of scatter transport.

# Example
```julia
# Reference geometry scanner
scanner_ref = Scanner()  # SID=540, SDD=950
scale = compute_scatter_geometry_scale(scanner_ref)  # ≈ 1.0

# GE Revolution (larger air gap)
scanner_ge = Scanner(source_to_isocenter=626.0, source_to_detector=1097.0)
scale = compute_scatter_geometry_scale(scanner_ge)  # ≈ 0.76
```
"""
function compute_scatter_geometry_scale(scanner::Scanner)
    # Current air gap
    air_gap = scanner.source_to_detector - scanner.source_to_isocenter

    # Inverse square scaling: larger air gap → less scatter → scale < 1
    scale = (SCATTER_REF_AIR_GAP_MM / air_gap)^2

    # Clamp to reasonable range to avoid extreme values
    return clamp(scale, 0.1, 10.0)
end

"""
    compute_scatter_kernel_fwhm_pixels(scanner::Scanner) -> Float64

Compute scatter kernel FWHM in pixels for given scanner geometry.

The physical scatter kernel size (~50 mm FWHM at detector) is approximately
constant regardless of scanner geometry. This function converts to pixel units
based on detector pitch.

# Example
```julia
# 1.0 mm pitch (reference)
scanner = Scanner(detector_col_size=1.0)
fwhm = compute_scatter_kernel_fwhm_pixels(scanner)  # = 50.0 pixels

# 0.5 mm pitch (high resolution)
scanner = Scanner(detector_col_size=0.5)
fwhm = compute_scatter_kernel_fwhm_pixels(scanner)  # = 100.0 pixels
```
"""
function compute_scatter_kernel_fwhm_pixels(scanner::Scanner)
    # detector_col_size is at isocenter; scatter is physical at detector face
    magnification = scanner.source_to_detector / scanner.source_to_isocenter
    detector_face_pitch = scanner.detector_col_size * magnification
    return SCATTER_PHYSICAL_KERNEL_FWHM_MM / detector_face_pitch
end

"""
    geometry_aware_scatter_model(scanner::Scanner; scale_factor=1.0, kernel_type=:gaussian, phantom_diameter_cm=nothing, mean_energy_keV=nothing)

Create a scatter model with parameters automatically scaled for scanner geometry,
phantom/patient size, and beam energy.

This function computes appropriate scatter parameters based on the scanner's
physical geometry, ensuring consistent SPR (~15% for 30cm body at 120 kVp) regardless of
scanner configuration.

# Arguments
- `scanner::Scanner`: Scanner definition with geometry parameters

# Keyword Arguments
- `scale_factor::Float64 = 1.0`: Additional user multiplier for scatter magnitude
- `kernel_type::Symbol = :gaussian`: Kernel shape (:gaussian or :exponential)
- `phantom_diameter_cm::Union{Nothing, Real} = nothing`: Effective phantom diameter (cm).
  If `nothing`, uses reference diameter (30 cm). Smaller phantoms get less scatter,
  larger phantoms get more scatter.
- `mean_energy_keV::Union{Nothing, Real} = nothing`: Mean photon energy (keV).
  If `nothing`, uses reference energy (60 keV, ~120 kVp). Lower energies get more scatter,
  higher energies get less scatter. For dual-energy CT, use different values for each kVp.

# Returns
`ScatterModel` with geometry, size, and energy-appropriate parameters.

# Scaling Behavior
- Geometry: Scatter coefficient scales with (air_gap_ref / air_gap)²
- Size: Scatter coefficient scales with (diameter / 30)^1.5
- Energy: Scatter coefficient scales with (60 / mean_energy_keV)^0.6
- Kernel FWHM scales with physical_fwhm_mm / detector_pixel_pitch_mm
- `scale_factor` applies on top of all automatic scaling

# Energy Scaling (for Dual-Energy CT)
| Mean Energy | Scale | Description |
|-------------|-------|-------------|
| 45 keV | 1.20 | ~80 kVp (high scatter) |
| 50 keV | 1.13 | ~80 kVp |
| 60 keV | 1.00 | ~120 kVp (reference) |
| 70 keV | 0.91 | ~140 kVp |
| 75 keV | 0.87 | ~140 kVp (low scatter) |

# Example
```julia
# Default scanner (reference geometry), reference phantom and energy
scanner = Scanner()
model = geometry_aware_scatter_model(scanner)
# model.scatter_coefficient ≈ 0.025

# With energy for dual-energy low-kVp acquisition (80 kVp, mean ~50 keV)
model_low = geometry_aware_scatter_model(scanner; mean_energy_keV=50.0)
# energy_scale = 1.13, so coefficient ≈ 0.025 * 1.13 ≈ 0.028

# With energy for dual-energy high-kVp acquisition (140 kVp, mean ~70 keV)
model_high = geometry_aware_scatter_model(scanner; mean_energy_keV=70.0)
# energy_scale = 0.91, so coefficient ≈ 0.025 * 0.91 ≈ 0.023

# GE Revolution + large patient + low kVp
scanner = Scanner(source_to_isocenter=626.0, source_to_detector=1097.0)
model = geometry_aware_scatter_model(scanner; phantom_diameter_cm=40.0, mean_energy_keV=50.0)
# geometry_scale ≈ 0.76, size_scale ≈ 1.54, energy_scale ≈ 1.13
# model.scatter_coefficient ≈ 0.025 * 0.76 * 1.54 * 1.13 ≈ 0.033
```

See also: [`default_scatter_model`](@ref), [`geometry_aware_scatter_correction`](@ref),
[`estimate_phantom_diameter_cm`](@ref), [`compute_scatter_size_scale`](@ref),
[`compute_scatter_energy_scale`](@ref)
"""
function geometry_aware_scatter_model(
    scanner::Scanner;
    scale_factor::Float64 = 1.0,
    kernel_type::Symbol = :gaussian,
    phantom_diameter_cm::Union{Nothing, Real} = nothing,
    mean_energy_keV::Union{Nothing, Real} = nothing
)
    # Compute geometry-based scaling (air gap)
    geometry_scale = compute_scatter_geometry_scale(scanner)

    # Compute size-based scaling (phantom diameter)
    size_scale = if phantom_diameter_cm !== nothing
        compute_scatter_size_scale(phantom_diameter_cm)
    else
        1.0  # Use reference size (30 cm body)
    end

    # Compute energy-based scaling (mean photon energy)
    energy_scale = if mean_energy_keV !== nothing
        compute_scatter_energy_scale(mean_energy_keV)
    else
        1.0  # Use reference energy (60 keV, ~120 kVp)
    end

    # Scale the base coefficient by geometry, size, AND energy
    scatter_coefficient = SCATTER_REF_COEFFICIENT * geometry_scale * size_scale * energy_scale

    # Compute kernel FWHM in pixels for this detector
    kernel_fwhm = compute_scatter_kernel_fwhm_pixels(scanner)

    # Return model with combined scale_factor
    return ScatterModel(scatter_coefficient, scale_factor, kernel_fwhm, kernel_type)
end

"""
    geometry_aware_scatter_correction(scanner::Scanner; scale_factor=1.0, kernel_type=:gaussian, phantom_diameter_cm=nothing, mean_energy_keV=nothing)

Create a scatter correction model with parameters automatically scaled for scanner geometry,
phantom/patient size, and beam energy.

Uses the same geometry, size, and energy scaling AND base coefficient as `geometry_aware_scatter_model`
for consistent scatter estimation and correction in simulation scenarios.

# Arguments
- `scanner::Scanner`: Scanner definition with geometry parameters

# Keyword Arguments
- `scale_factor::Float64 = 1.0`: Additional user multiplier for correction strength
- `kernel_type::Symbol = :gaussian`: Kernel shape (:gaussian or :exponential)
- `phantom_diameter_cm::Union{Nothing, Real} = nothing`: Effective phantom diameter (cm).
  If `nothing`, uses reference diameter (30 cm). Must match the value used in
  `geometry_aware_scatter_model()` for consistent correction.
- `mean_energy_keV::Union{Nothing, Real} = nothing`: Mean photon energy (keV).
  If `nothing`, uses reference energy (60 keV, ~120 kVp). Must match the value used in
  `geometry_aware_scatter_model()` for consistent correction.

# Returns
`ScatterCorrectionModel` with geometry, size, and energy-appropriate parameters.

# Notes
The correction uses the SAME coefficient and scaling as scatter addition
(SCATTER_REF_COEFFICIENT × geometry_scale × size_scale × energy_scale) to ensure consistent behavior.
The `prep_exponent` field is set to 1.0 (linear model) to match the `add_scatter!()`
algorithm.

**CRITICAL for Dual-Energy:** The `mean_energy_keV` parameter MUST match the value used in
`geometry_aware_scatter_model()` for the same acquisition. Using mismatched energy values
will cause wave artifacts in material decomposition.

For CatSim-exact correction parameters, use `default_scatter_correction()` instead.

# Example
```julia
scanner = Scanner(source_to_isocenter=626.0, source_to_detector=1097.0)

# Reference size and energy correction
correction = geometry_aware_scatter_correction(scanner)

# With phantom size and energy (must match scatter model parameters)
correction_low = geometry_aware_scatter_correction(scanner;
    phantom_diameter_cm=30.0, mean_energy_keV=50.0)  # 80 kVp
correction_high = geometry_aware_scatter_correction(scanner;
    phantom_diameter_cm=30.0, mean_energy_keV=70.0)  # 140 kVp
```

See also: [`default_scatter_correction`](@ref), [`geometry_aware_scatter_model`](@ref),
[`estimate_phantom_diameter_cm`](@ref), [`compute_scatter_energy_scale`](@ref)
"""
function geometry_aware_scatter_correction(
    scanner::Scanner;
    scale_factor::Float64 = 1.0,
    kernel_type::Symbol = :gaussian,
    phantom_diameter_cm::Union{Nothing, Real} = nothing,
    mean_energy_keV::Union{Nothing, Real} = nothing
)
    # Same geometry scaling as scatter model
    geometry_scale = compute_scatter_geometry_scale(scanner)

    # Same size scaling as scatter model
    size_scale = if phantom_diameter_cm !== nothing
        compute_scatter_size_scale(phantom_diameter_cm)
    else
        1.0  # Use reference size (30 cm body)
    end

    # Same energy scaling as scatter model
    energy_scale = if mean_energy_keV !== nothing
        compute_scatter_energy_scale(mean_energy_keV)
    else
        1.0  # Use reference energy (60 keV, ~120 kVp)
    end

    # Use SAME coefficient and scaling as scatter addition for consistent simulation
    correction_coefficient = SCATTER_REF_COEFFICIENT * geometry_scale * size_scale * energy_scale

    # Compute kernel FWHM in pixels
    kernel_fwhm = compute_scatter_kernel_fwhm_pixels(scanner)

    # Linear model (exponent = 1.0) to match add_scatter!()
    # The prep_exponent field is kept for API compatibility but is ignored by correct_scatter!()
    prep_exponent = 1.0

    return ScatterCorrectionModel(
        correction_coefficient,
        scale_factor,
        prep_exponent,
        kernel_fwhm,
        kernel_type
    )
end





# =============================================================================
# Exports
# =============================================================================

export ScatterModel, default_scatter_model
export create_scatter_kernel_spatial
export add_scatter!, add_scatter
export ScatterCorrectionModel, default_scatter_correction
export correct_scatter!, correct_scatter

# Geometry-aware scatter API
export geometry_aware_scatter_model, geometry_aware_scatter_correction
export compute_scatter_geometry_scale, compute_scatter_kernel_fwhm_pixels

# Reference constants for scatter calibration
export SCATTER_REF_SID_MM, SCATTER_REF_SDD_MM, SCATTER_REF_AIR_GAP_MM
export SCATTER_REF_PIXEL_PITCH_MM, SCATTER_REF_COEFFICIENT
export SCATTER_PHYSICAL_KERNEL_FWHM_MM, SCATTER_REF_CORRECTION_COEFFICIENT

# Phantom size-aware scatter API
export estimate_phantom_diameter_cm, compute_scatter_size_scale
export SCATTER_REF_PHANTOM_DIAMETER_CM, SCATTER_SIZE_SCALING_EXPONENT

# Energy-dependent scatter API
export compute_scatter_energy_scale
export SCATTER_REF_ENERGY_KEV, SCATTER_ENERGY_EXPONENT

