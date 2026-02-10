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
function add_scatter!(sinogram::AbstractArray{T,3}, model::ScatterModel;
                      ws_output=nothing, ws_kernel=nothing) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Combined scatter coefficient
    C = T(model.scatter_coefficient * model.scale_factor)

    # Create scatter kernel on CPU (or use pre-computed workspace kernel)
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

    # Output buffer (use workspace or allocate)
    output = ws_output !== nothing ? ws_output : similar(sinogram)

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
                          ws_output=nothing, ws_kernel=nothing) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    n_angles = size(sinogram, 3)

    # Combined correction coefficient
    C = T(model.correction_coefficient * model.scale_factor)
    # NOTE: prep_exponent is now ignored - we use linear model to match add_scatter!()

    # Create scatter kernel (or use pre-computed workspace kernel)
    if ws_kernel !== nothing
        kernel = ws_kernel
        kernel_size = size(kernel, 1)
    else
        scatter_model_temp = ScatterModel(
            model.correction_coefficient,
            model.scale_factor,
            model.kernel_fwhm,
            model.kernel_type
        )
        kernel_cpu = T.(create_scatter_kernel_spatial(scatter_model_temp))
        kernel_size = size(kernel_cpu, 1)
        kernel = similar(sinogram, size(kernel_cpu)...)
        copyto!(kernel, kernel_cpu)
    end
    half_k = kernel_size ÷ 2

    # Output buffer (use workspace or allocate)
    output = ws_output !== nothing ? ws_output : similar(sinogram)

    eps = T(1e-10)

    # GPU-native scatter correction with damping to prevent over-correction
    #
    # ISSUE: When scatter was added, projection values were HIGHER.
    # After scatter addition, projection decreases. For p > 1, the function
    # f(p) = exp(-p) × p INCREASES as p DECREASES, so estimating scatter
    # from the current (lower) projection values OVER-estimates scatter by ~15-20%.
    #
    # FIX: Apply a damping factor to reduce the scatter estimate.
    # The damping factor is designed to compensate for the nonlinear bias
    # that arises from estimating scatter on already-scattered data.
    # Empirically, ~0.85 damping compensates for the ~17% over-estimation.
    scatter_damping = T(0.85)

    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Current value (log domain = line integral)
        prep = sinogram[idx]

        # Convert to intensity
        clamped_prep = min(max(prep, T(0)), T(20))
        intensity = exp(-clamped_prep)

        # Compute scatter estimate via spatial convolution
        scatter_est = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                # Source prep value
                src_prep = sinogram[src_col, src_row, angle]
                src_clamped = min(max(src_prep, eps), T(20))
                src_intensity = exp(-src_clamped)

                # Scatter pre-signal: intensity × prep × C
                scatter_pre = src_intensity * src_clamped * C

                # Kernel weight
                ki = di + half_k + 1
                kj = dj + half_k + 1

                scatter_est += scatter_pre * kernel[ki, kj]
            end
        end

        # Apply damping to prevent over-correction
        scatter_est_damped = scatter_est * scatter_damping

        # Subtract scatter estimate from intensity
        # Ensure result is positive
        corrected_intensity = max(intensity - scatter_est_damped, eps)

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
    estimate_phantom_diameter_cm(mask::AbstractArray{UInt8,3}, voxel_size_mm) -> Float64

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
    mask::AbstractArray{UInt8,3},
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
    return SCATTER_PHYSICAL_KERNEL_FWHM_MM / scanner.detector_col_size
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
# Dual-Energy Joint Scatter Estimation
# =============================================================================

"""
    estimate_scatter_joint(sino_low, sino_high, model::ScatterModel;
                           low_weight=0.5, high_weight=0.5) -> scatter_estimate

Estimate scatter jointly from both dual-energy sinograms.

Returns a single scatter estimate (in intensity domain) that can be applied to
BOTH sinograms using `correct_scatter_with_estimate!()`, ensuring identical
residual patterns that cancel in material decomposition.

# Algorithm
1. Combine sinograms: combined = low_weight × sino_low + high_weight × sino_high
2. Estimate scatter from combined using standard convolution model
3. Return scatter estimate (in intensity domain)

# Why Joint Estimation
Per-sinogram scatter correction creates DIFFERENT residuals at each energy.
Material decomposition takes a weighted DIFFERENCE of sinograms, amplifying
these different residuals into wave/stripe artifacts. Joint estimation ensures
the SAME residual pattern for both energies, which then cancels (or at least
doesn't amplify) in the decomposition.

# Arguments
- `sino_low`: Low-kVp sinogram (log domain, line integrals)
- `sino_high`: High-kVp sinogram (log domain, line integrals)
- `model::ScatterModel`: Scatter model (use average energy for dual-energy)

# Keyword Arguments
- `low_weight::Float64=0.5`: Weight for low-energy sinogram in combination
- `high_weight::Float64=0.5`: Weight for high-energy sinogram in combination

# Returns
Scatter estimate array (intensity domain) with same size as input sinograms.

# Example
```julia
# Create scatter model at average energy
model = geometry_aware_scatter_model(scanner; mean_energy_keV=60.0)

# Estimate scatter jointly
scatter_est = estimate_scatter_joint(sino_low, sino_high, model)

# Apply to both sinograms
correct_scatter_with_estimate!(sino_low, scatter_est)
correct_scatter_with_estimate!(sino_high, scatter_est)
```

See also: [`correct_scatter_with_estimate!`](@ref), [`correct_scatter_dual_energy!`](@ref)
"""
function estimate_scatter_joint(
    sino_low::AbstractArray{T,3},
    sino_high::AbstractArray{T,3},
    model::ScatterModel;
    low_weight::Float64 = 0.5,
    high_weight::Float64 = 0.5
) where T
    n_cols, n_rows, n_angles = size(sino_low)

    # Combined scatter coefficient
    C = T(model.scatter_coefficient * model.scale_factor)

    # Create kernel
    kernel_cpu = T.(create_scatter_kernel_spatial(model))
    kernel_size = size(kernel_cpu, 1)
    half_k = kernel_size ÷ 2

    # Transfer kernel to GPU (same device as input)
    kernel = similar(sino_low, size(kernel_cpu)...)
    copyto!(kernel, kernel_cpu)

    # Output: scatter estimate in INTENSITY domain
    scatter_estimate = similar(sino_low)

    w_low = T(low_weight)
    w_high = T(high_weight)

    AK.foreachindex(scatter_estimate) do idx
        ci = CartesianIndices(scatter_estimate)[idx]
        col, row, angle = Tuple(ci)

        # Scatter estimate via convolution on COMBINED signal
        scatter_acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                # Combined projection (weighted average in log domain)
                proj_low = sino_low[src_col, src_row, angle]
                proj_high = sino_high[src_col, src_row, angle]
                combined_proj = w_low * proj_low + w_high * proj_high

                # Convert to intensity (clamp to avoid overflow)
                clamped_proj = min(max(combined_proj, T(0)), T(15))
                intensity = exp(-clamped_proj)

                # Scatter pre-signal: intensity × projection × C
                scatter_pre = intensity * clamped_proj * C

                # Kernel weight
                ki = di + half_k + 1
                kj = dj + half_k + 1
                scatter_acc += scatter_pre * kernel[ki, kj]
            end
        end

        scatter_estimate[idx] = max(scatter_acc, zero(T))
    end

    return scatter_estimate
end

"""
    correct_scatter_with_estimate!(sinogram, scatter_estimate) -> sinogram

Apply a pre-computed scatter estimate to a sinogram (in-place).

This function subtracts the scatter estimate from the sinogram in intensity
domain and converts back to log domain. Used with `estimate_scatter_joint()`
for dual-energy joint scatter correction.

# Arguments
- `sinogram`: Input sinogram (log domain, line integrals)
- `scatter_estimate`: Pre-computed scatter estimate (intensity domain)

# Algorithm
For each pixel:
1. Convert sinogram to intensity: I = exp(-sinogram)
2. Subtract scatter: I_corrected = I - scatter_estimate
3. Convert back to log: sinogram = -log(I_corrected)

# Returns
Modified sinogram with scatter correction applied.

# Example
```julia
# After computing joint scatter estimate
correct_scatter_with_estimate!(sino_low, scatter_est)
correct_scatter_with_estimate!(sino_high, scatter_est)
```

See also: [`estimate_scatter_joint`](@ref), [`correct_scatter_dual_energy!`](@ref)
"""
function correct_scatter_with_estimate!(
    sinogram::AbstractArray{T,3},
    scatter_estimate::AbstractArray{T,3}
) where T
    output = similar(sinogram)
    eps = T(1e-10)

    AK.foreachindex(sinogram) do idx
        # Convert to intensity
        proj = sinogram[idx]
        clamped_proj = min(max(proj, T(0)), T(15))
        intensity = exp(-clamped_proj)

        # Subtract scatter estimate
        corrected_intensity = max(intensity - scatter_estimate[idx], eps)

        # Convert back to log domain
        output[idx] = -log(corrected_intensity)
    end

    copyto!(sinogram, output)
    return sinogram
end

"""
    correct_scatter_dual_energy!(sino_low, sino_high, scanner;
                                  phantom_diameter_cm=nothing,
                                  mean_energy_low_keV=50.0,
                                  mean_energy_high_keV=70.0) -> (sino_low, sino_high)

Apply joint scatter correction to dual-energy sinograms (in-place).

Uses a single scatter estimate from the combined signal to ensure identical
residual patterns that cancel in material decomposition. This avoids the
wave artifacts caused by per-sinogram scatter correction.

# Why Joint Correction
Per-sinogram scatter correction creates different residuals at 80 kVp vs 140 kVp
because:
1. SPR is ~5× higher at 80 kVp than 140 kVp
2. Different energy-dependent coefficients → different estimation errors
3. Material decomposition: material = a × sino_low + b × sino_high (b is NEGATIVE)
4. Different residuals get AMPLIFIED → wave artifacts

Joint correction ensures:
- Same scatter estimate → same residual pattern for both energies
- Decomposition: a × ε + b × ε = (a+b) × ε (single pattern, not amplified difference)

# Arguments
- `sino_low`: Low-kVp sinogram (modified in-place)
- `sino_high`: High-kVp sinogram (modified in-place)
- `scanner::Scanner`: Scanner geometry for scatter model

# Keyword Arguments
- `phantom_diameter_cm`: Phantom diameter for size scaling (nothing = reference)
- `mean_energy_low_keV`: Mean energy of low-kVp acquisition (default 50.0)
- `mean_energy_high_keV`: Mean energy of high-kVp acquisition (default 70.0)

# Returns
Tuple of modified sinograms `(sino_low, sino_high)`.

# Example
```julia
scanner = Scanner(source_to_isocenter=626.0, source_to_detector=1097.0, ...)
correct_scatter_dual_energy!(sino_low, sino_high, scanner)
```

See also: [`estimate_scatter_joint`](@ref), [`correct_scatter_with_estimate!`](@ref)
"""
function correct_scatter_dual_energy!(
    sino_low::AbstractArray{T,3},
    sino_high::AbstractArray{T,3},
    scanner::Scanner;
    phantom_diameter_cm::Union{Nothing,Real} = nothing,
    mean_energy_low_keV::Real = 50.0,
    mean_energy_high_keV::Real = 70.0
) where T
    # Use average energy for joint model
    mean_energy_joint = 0.5 * mean_energy_low_keV + 0.5 * mean_energy_high_keV

    # Create scatter model at joint (average) energy
    scatter_model = geometry_aware_scatter_model(scanner;
        phantom_diameter_cm = phantom_diameter_cm,
        mean_energy_keV = mean_energy_joint
    )

    # Estimate scatter jointly from both sinograms
    scatter_estimate = estimate_scatter_joint(sino_low, sino_high, scatter_model)

    # Apply SAME scatter estimate to BOTH sinograms
    # This ensures identical residual patterns that cancel in decomposition
    correct_scatter_with_estimate!(sino_low, scatter_estimate)
    correct_scatter_with_estimate!(sino_high, scatter_estimate)

    return (sino_low, sino_high)
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

# Dual-energy joint scatter correction API
export estimate_scatter_joint, correct_scatter_with_estimate!
export correct_scatter_dual_energy!
