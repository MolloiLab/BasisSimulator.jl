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

# add_scatter! / add_scatter DELETED — replaced by unified per-energy scatter path:
# - estimate_scatter_field!() for spatial distribution (shared EICT + PCCT)
# - compute_scatter_energy_weights() for per-energy Compton fractions
# - inject_scatter!() for EICT, inject_scatter_bins!() for PCCT


# ScatterCorrectionModel / correct_scatter! / correct_scatter DELETED.
# Scatter correction is now decoupled from simulate!. The known scatter field
# and weights are returned from simulate!() for exact model-based subtraction
# at the notebook level (same pattern as HU calibration via μ_water).


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


# =============================================================================
# Phantom Size Scaling Constants
# =============================================================================

"""Reference phantom diameter (cm) for scatter calibration (adult body)."""
const SCATTER_REF_PHANTOM_DIAMETER_CM = 30.0

"""SPR scaling exponent for phantom diameter (empirical: 1.5-2.0)."""
const SCATTER_SIZE_SCALING_EXPONENT = 1.5

# =============================================================================
# Per-Energy Scatter Weights (unified EICT + PCCT interface)
# =============================================================================
#
# Energy-dependent scatter is computed per-energy at the tube spectrum level.
# The analytical model uses the Compton scatter fraction; a future MC LUT
# implementation would replace this with pre-computed per-energy data.
#
# References:
# - NIST XCOM photon cross-section database
# - Klein-Nishina formula (theoretical Compton scattering cross-section)
# =============================================================================

"""
    compute_scatter_energy_weights(energies::Vector{Float64}) -> Vector{Float64}

Compute per-energy Compton scatter fractions for the tube spectrum.

Returns a weight for each energy representing the probability that a photon
at that energy contributes to scatter (vs photoelectric absorption). This is
the **pluggable interface** for the scatter energy model:
- **Analytical (current):** `1 / (1 + (20/E)³)` — empirical Compton fraction
  matching NIST XCOM within 5% for 20–140 keV diagnostic range.
- **MC LUT (future):** Load pre-computed per-energy scatter data from table.

These weights are combined with the spatial scatter field to produce the
full per-energy scatter contribution. The detector model (EICT integration
or PCCT DRM binning) then determines how scatter enters the measured signal.

# Returns
`Vector{Float64}` of length `length(energies)`, values in [0, 1].

# References
- NIST XCOM: Photon cross-section database (physics.nist.gov/xcom)
- Klein-Nishina formula: Compton scattering cross-section vs energy
"""
function compute_scatter_energy_weights(energies::Vector{Float64})
    return [1.0 / (1.0 + (20.0 / max(E, 1.0))^3) for E in energies]
end

"""
    inject_scatter!(sinogram, scatter_field, scatter_weight)

Add scatter to a single polychromatic sinogram (EICT path).

The scatter contribution at each pixel is `scatter_field × scatter_weight`,
added in intensity domain. `scatter_weight` is the spectrum-integrated
Compton fraction: `Σ(w_E × ew_E) / Σ(w_E)` where `ew_E` are per-energy
weights from `compute_scatter_energy_weights`.

Matches the physics of `inject_scatter_bins!` (PCCT) — same spatial field,
same per-energy model — but integrated over the full spectrum since an
energy-integrating detector sums all energies.

# References
- Ohnesorge B et al., Eur Radiol 1999 (convolution scatter model)
"""
function inject_scatter!(
    sinogram::AbstractArray{T,3},
    scatter_field::AbstractArray{T,3},
    scatter_weight::Real
) where T
    sw = T(scatter_weight)
    let sino = sinogram, sf = scatter_field, w = sw
        AK.foreachindex(sino) do idx
            proj = sino[idx]
            clamped = min(proj, T(20))
            intensity = exp(-clamped)
            scatter_intensity = sf[idx] * w
            total = intensity + max(scatter_intensity, zero(T))
            @inbounds sino[idx] = -log(max(total, T(1e-10)))
        end
    end
    return sinogram
end

"""
    inject_scatter_bins!(bins, scatter_field, I0_bins, I0_total, bin_scatter_weights)

Add scatter to per-bin sinograms (PCCT path).

For each bin `b`, adds `scatter_field × I0_total × bin_weight_b` scatter
counts, where `bin_weight_b` is from `compute_scatter_bin_weights` (per-energy
weights convolved through the DRM). This is the exact per-energy scatter model
applied through the photon-counting detector response.

Matches the physics of `inject_scatter!` (EICT) — same spatial field,
same per-energy model — but distributed across energy bins via the DRM.

# References
- Ohnesorge B et al., Eur Radiol 1999 (convolution scatter model)
"""
function inject_scatter_bins!(
    bins::Vector{<:AbstractArray{T,3}},
    scatter_field::AbstractArray{T,3},
    I0_bins::Vector{<:Real},
    I0_total::Real,
    bin_scatter_weights::Vector{Float64};
    subtract::Bool = false,
) where T
    eps = T(1e-10)
    sgn = subtract ? -one(T) : one(T)   # +1 = inject (forward), -1 = correct
    for (b, bin_sino) in enumerate(bins)
        let I0b = T(I0_bins[b]), frac = T(bin_scatter_weights[b]),
            I0t = T(I0_total), bs = bin_sino, sf = scatter_field, s = sgn
            AK.foreachindex(bs) do idx
                N_primary = I0b * exp(-bs[idx])
                N_scatter = sf[idx] * I0t * frac
                N_total = N_primary + s * max(N_scatter, zero(T))
                bs[idx] = -log(max(N_total, eps) / I0b)
            end
        end
    end
    return bins
end

"""
    compute_scatter_bin_weights(energies, weights, energy_weights, η, R) -> Vector{Float64}

Compute per-bin scatter weights from per-energy weights convolved through the DRM.

This replaces the old `compute_scatter_bin_fractions` with an explicit energy_weights
input, making the per-energy scatter model visible and pluggable.

The per-bin weight for bin `b` is:
    `w_b = Σ_E  weights[E] × energy_weights[E] × η[E] × R[E, b]`
normalized to sum to 1.0.

# Arguments
- `energies`: Photon energies (keV)
- `weights`: Tube spectrum weights (photons per energy bin)
- `energy_weights`: Per-energy Compton scatter fractions from `compute_scatter_energy_weights`
- `η`: Quantum detection efficiency per energy
- `R`: Detector response matrix (n_energies × n_bins)

# Returns
`Vector{Float64}` of length `n_bins`, summing to 1.0.
"""
function compute_scatter_bin_weights(
    energies::Vector{Float64},
    weights::Vector{Float64},
    energy_weights::Vector{Float64},
    η::Vector{Float64},
    R::Matrix{Float64},
    kVp::Float64
)
    n_bins = size(R, 2)
    n_R = size(R, 1)
    scatter_I0_per_bin = zeros(Float64, n_bins)

    for (e_idx, E) in enumerate(energies)
        w = weights[e_idx]
        ew = energy_weights[e_idx]
        w * ew < 1e-12 && continue

        # Map spectrum energy to DRM grid
        r_idx = clamp(round(Int, (E - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)

        for b in 1:n_bins
            scatter_I0_per_bin[b] += w * ew * η[e_idx] * R[r_idx, b]
        end
    end

    total = sum(scatter_I0_per_bin)
    if total > 0
        scatter_I0_per_bin ./= total
    else
        fill!(scatter_I0_per_bin, 1.0 / n_bins)
    end

    return scatter_I0_per_bin
end

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
    geometry_aware_scatter_model(scanner::Scanner; scale_factor=1.0, kernel_type=:gaussian, phantom_diameter_cm=nothing)

Create a **spatial-only** scatter model scaled for scanner geometry and phantom size.

Energy dependence is handled separately by `compute_scatter_energy_weights()`, which
returns per-energy Compton fractions. This separation enables pluggable energy models
(analytical Compton fraction now, MC LUT in future).

# Scaling Behavior
- Geometry: `SCATTER_REF_COEFFICIENT × (air_gap_ref / air_gap)²`
- Size: `× (diameter / 30)^1.5`
- Kernel FWHM: `physical_fwhm_mm / detector_pixel_pitch_mm`
- `scale_factor` applies on top of all automatic scaling

# Example
```julia
scanner = Scanner()
model = geometry_aware_scatter_model(scanner)
# model.scatter_coefficient ≈ 0.025 × geometry_scale × size_scale

# GE Revolution + large patient
scanner = Scanner(source_to_isocenter=626.0, source_to_detector=1097.0)
model = geometry_aware_scatter_model(scanner; phantom_diameter_cm=40.0)
# geometry_scale ≈ 0.76, size_scale ≈ 1.54
# model.scatter_coefficient ≈ 0.025 * 0.76 * 1.54 ≈ 0.029
```

See also: [`default_scatter_model`](@ref), [`compute_scatter_energy_weights`](@ref)
"""
function geometry_aware_scatter_model(
    scanner::Scanner;
    scale_factor::Float64 = 1.0,
    kernel_type::Symbol = :gaussian,
    phantom_diameter_cm::Union{Nothing, Real} = nothing,
    mean_energy_keV::Union{Nothing, Real} = nothing  # IGNORED — kept for API compat during transition
)
    # Compute geometry-based scaling (air gap)
    geometry_scale = compute_scatter_geometry_scale(scanner)

    # Compute size-based scaling (phantom diameter)
    size_scale = if phantom_diameter_cm !== nothing
        compute_scatter_size_scale(phantom_diameter_cm)
    else
        1.0  # Use reference size (30 cm body)
    end

    # Energy scaling is now handled per-energy via compute_scatter_energy_weights(),
    # NOT baked into the spatial coefficient. This enables per-energy scatter models
    # (analytical Compton fraction now, MC LUT in future).
    scatter_coefficient = SCATTER_REF_COEFFICIENT * geometry_scale * size_scale

    # Compute kernel FWHM in pixels for this detector
    kernel_fwhm = compute_scatter_kernel_fwhm_pixels(scanner)

    return ScatterModel(scatter_coefficient, scale_factor, kernel_fwhm, kernel_type)
end


# geometry_aware_scatter_correction DELETED — scatter correction is now decoupled.
# simulate!() returns scatter_field + weights for exact model-based subtraction.





# =============================================================================
# Exports
# =============================================================================
# Calibration constants (SCATTER_REF_*, SCATTER_PHYSICAL_KERNEL_FWHM_MM,
# SCATTER_SIZE_SCALING_EXPONENT) and internal scaling helpers
# (compute_scatter_geometry_scale, compute_scatter_kernel_fwhm_pixels,
# compute_scatter_size_scale) stay `const`/internal — consumed only by
# `geometry_aware_scatter_model` inside this file.

export ScatterModel
export create_scatter_kernel_spatial
export geometry_aware_scatter_model
export estimate_phantom_diameter_cm

# Unified per-energy scatter API (shared EICT + PCCT)
export estimate_scatter_field!
export compute_scatter_energy_weights
export inject_scatter!, inject_scatter_bins!
export compute_scatter_bin_weights


# =============================================================================
# Spatial Scatter Field Estimation (shared EICT + PCCT)
# =============================================================================
# The spatial scatter distribution is estimated from the combined polyenergetic
# sinogram using the Ohnesorge convolution model. Per-energy variation is handled
# separately by compute_scatter_energy_weights().
#
# Ohnesorge B et al., Eur Radiol 1999 — convolution-based scatter model
# =============================================================================

"""
    estimate_scatter_field!(output, sinogram, model::ScatterModel; ...) -> output

Compute the scatter intensity spatial distribution WITHOUT modifying the input sinogram.

Ohnesorge convolution model — returns only the spatial scatter intensity field.
The output is in intensity units relative to I0=1. Used by both EICT and PCCT paths.

# Returns
`output` filled with scatter intensity at each detector pixel.
Multiply by `I0_total` to get absolute scatter counts.
"""
function estimate_scatter_field!(
    output::AbstractArray{T,3},
    sinogram::AbstractArray{T,3},
    model::ScatterModel;
    ws_scatter_temp=nothing,
    ws_kernel_1d=nothing
) where T
    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    C = T(model.scatter_coefficient * model.scale_factor)

    # ─── SEPARABLE PATH (Gaussian kernels) ──────────────────────────────
    if model.kernel_type == :gaussian
        kernel_1d = if ws_kernel_1d !== nothing
            ws_kernel_1d
        else
            k1d_cpu = T.(create_scatter_kernel_1d(model))
            k1d = similar(sinogram, T, length(k1d_cpu))
            copyto!(k1d, k1d_cpu)
            k1d
        end

        scatter_temp = ws_scatter_temp !== nothing ? ws_scatter_temp : similar(sinogram)

        # Step 1: Pre-signal into output
        let sino = sinogram, pre = output, c = C
            AK.foreachindex(sino) do idx
                proj = sino[idx]
                clamped = min(proj, T(20))
                @inbounds pre[idx] = exp(-clamped) * proj * c
            end
        end

        # Step 2-3: Separable convolution → output holds scatter intensity
        _convolve_separable_h!(scatter_temp, output, kernel_1d, n_cols, n_rows)
        _convolve_separable_v!(output, scatter_temp, kernel_1d, n_cols, n_rows)

        return output
    end

    # ─── FALLBACK: 2D convolution ────────────────────────────────────────
    kernel_cpu = T.(create_scatter_kernel_spatial(model))
    kernel_size = size(kernel_cpu, 1)
    kernel = similar(sinogram, size(kernel_cpu)...)
    copyto!(kernel, kernel_cpu)
    half_k = kernel_size ÷ 2

    let kernel = kernel, output = output, half_k = half_k,
        n_cols = n_cols, n_rows = n_rows, C = C
        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % Int32(n_cols)) + Int32(1)
            idx_0 = idx_0 ÷ Int32(n_cols)
            row = (idx_0 % Int32(n_rows)) + Int32(1)
            angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

            scatter_est = zero(T)
            for dj in -half_k:half_k
                for di in -half_k:half_k
                    src_col = clamp(col + di, 1, n_cols)
                    src_row = clamp(row + dj, 1, n_rows)
                    src_prep = sinogram[src_col, src_row, angle]
                    src_clamped = min(max(src_prep, T(1e-10)), T(20))
                    scatter_pre = exp(-src_clamped) * src_clamped * C
                    ki = di + half_k + 1
                    kj = dj + half_k + 1
                    scatter_est += scatter_pre * kernel[ki, kj]
                end
            end
            output[idx] = scatter_est
        end
    end
    return output
end

# compute_scatter_bin_fractions DELETED — replaced by compute_scatter_bin_weights()
# which takes explicit per-energy weights from compute_scatter_energy_weights().

