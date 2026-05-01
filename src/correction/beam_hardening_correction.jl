# =============================================================================
# Beam Hardening Correction (BHC)
# =============================================================================
#
# Implements water-based beam hardening correction for polychromatic CT.
#
# ## Physics Background
#
# X-ray beams from clinical CT scanners are polychromatic (bremsstrahlung +
# characteristic lines). As the beam traverses tissue, lower-energy photons
# are preferentially absorbed, causing the mean energy to increase ("harden").
# This energy shift causes the measured line integral to be less than what
# would be measured with a monochromatic beam of the same effective energy.
#
# ## Beam Hardening Effects
#
# 1. **Cupping artifacts**: Center of uniform objects appears darker than edges
#    because X-rays traveling through the center traverse more material and
#    experience more hardening. Typical cupping in water can be 50-100 HU
#    without correction.
#
# 2. **Inaccurate HU values**: Dense materials appear less attenuating than
#    expected because the "effective μ" depends on path length and tissue
#    composition traversed.
#
# 3. **Dark bands/streaks**: Between dense objects (e.g., temporal bones)
#    where maximum beam hardening occurs along certain ray paths.
#
# ## Correction Method
#
# This module implements **water-based polynomial BHC**, the standard approach
# used in clinical CT (including CatSim/XCIST):
#
# 1. Generate calibration curve: simulate polychromatic transmission through
#    water of varying thickness (0 to ~50 cm)
#
# 2. For each path, compute:
#    - **Measured**: Polychromatic line integral p_meas = -log(Σᵢ wᵢ × exp(-μᵢ×d))
#    - **True**: Monochromatic equivalent p_true = μ_ref × d at reference energy
#
# 3. Fit polynomial: p_true = a₀ + a₁×p_meas + a₂×p_meas² + ... + aₙ×p_measⁿ
#
# 4. Apply polynomial to measured sinogram values before reconstruction
#
# ## CatSim Compatibility
#
# This implementation matches CatSim's Prep_BHC_Accurate.py approach:
# - Same polynomial fitting methodology
# - Same calibration curve generation
# - Applied post-log transform, pre-reconstruction
# - Order 5 polynomial typical (configurable)
#
# **Key difference**: CatSim fits per-detector coefficients to account for
# bowtie filter path variation. BasisSimulator uses uniform coefficients,
# which is accurate when bowtie attenuation is accounted for separately.
#
# ## Limitations
#
# Water-based BHC corrects for water-like tissues but under-corrects for
# bone, causing residual "dark band" artifacts between dense structures.
# More advanced methods (dual-material, iterative) can address this but
# are not implemented here.
#
# ## References
#
# 1. Joseph PM, Spital RD. "A method for correcting bone induced artifacts
#    in computed tomography scanners." J Comput Assist Tomogr. 1978;2(1):100-108.
#    doi:10.1097/00004728-197801000-00017
#
# 2. Herman GT. "Correction for beam hardening in computed tomography."
#    Phys Med Biol. 1979;24(1):81-106. doi:10.1088/0031-9155/24/1/008
#
# 3. Brooks RA, Di Chiro G. "Beam hardening in x-ray reconstructive tomography."
#    Phys Med Biol. 1976;21(3):390-398. doi:10.1088/0031-9155/21/3/004
#
# 4. De Man B, et al. "Metal streak artifacts in X-ray computed tomography:
#    a simulation study." IEEE Trans Nucl Sci. 1999;46(3):691-696.
#    doi:10.1109/23.775600
#
# 5. CatSim/XCIST: https://github.com/xcist/main - Prep_BHC_Accurate.py
#
# =============================================================================

import AcceleratedKernels as AK
import XrayAttenuation as XA
using Unitful: ustrip, @u_str

export BeamHardeningCorrection, BHCPolynomial
export calibrate_bhc, apply_bhc!
export get_bhc_coefficients

# Two-material (water + bone) sinogram-domain BHC
export bone_fraction_smooth, TwoMaterialBHC, TwoMaterialBHCPerColumn
export calibrate_bhc_two_material, apply_bhc_two_material
export bhc_spectrum_per_column, compute_polychromatic_μ_water

# Image-domain BHC (So et al. 2009)
export apply_bhc_image_domain

# =============================================================================
# BHC Types
# =============================================================================

"""
    BHCPolynomial

Polynomial beam hardening correction coefficients.

# Mathematical Formulation

The correction maps measured (polychromatic) line integrals to true
(monochromatic-equivalent) values using a polynomial:

    p_corrected = a₀ + a₁×p + a₂×p² + a₃×p³ + ... + aₙ×pⁿ

where `p` is the measured line integral and `p_corrected` is the
beam-hardening-corrected value that would have been measured with
a monochromatic beam at the reference energy.

# Fields
- `coefficients::Vector{Float64}`: Polynomial coefficients [a₀, a₁, a₂, ...].
   Stored in ascending order of power.
- `order::Int`: Polynomial order (length(coefficients) - 1). Typical: 3-5.
- `reference_energy_keV::Float64`: Reference energy for "true" monochromatic
   values (typically 70 keV for clinical CT).

# Typical Values

For 120 kVp tungsten spectrum with 70 keV reference:
- Order 3: RMS error ~0.01
- Order 5: RMS error <0.001 (recommended)
- Order 7+: Diminishing returns, risk of overfitting

# CatSim Compatibility

CatSim (Prep_BHC_Accurate.py) uses the same polynomial formulation.
Default order in CatSim: 5.
"""
struct BHCPolynomial
    coefficients::Vector{Float64}
    order::Int
    reference_energy_keV::Float64
end

"""
    BeamHardeningCorrection

Complete BHC model including calibration data and fitted polynomial.

This struct contains both the correction polynomial and the calibration
data used to generate it, enabling quality assessment and debugging.

# Fields
- `polynomial::BHCPolynomial`: Fitted correction polynomial
- `calibration_paths::Vector{Float64}`: Water path lengths (cm) used for calibration
- `calibration_measured::Vector{Float64}`: Polychromatic line integrals at each path
- `calibration_true::Vector{Float64}`: Monochromatic line integrals at each path

# Usage

Create via `calibrate_bhc()` which generates the calibration curve
and fits the polynomial:

```julia
energies, weights = load_spectrum(120)
bhc = calibrate_bhc(energies, weights; order=5)
apply_bhc!(sinogram, bhc)
```

"""
struct BeamHardeningCorrection
    polynomial::BHCPolynomial
    calibration_paths::Vector{Float64}
    calibration_measured::Vector{Float64}
    calibration_true::Vector{Float64}
end

"""
    get_bhc_coefficients(bhc) -> Vector{Float64}

Extract polynomial coefficients from either BHCPolynomial or BeamHardeningCorrection.
"""
get_bhc_coefficients(poly::BHCPolynomial) = poly.coefficients
get_bhc_coefficients(bhc::BeamHardeningCorrection) = bhc.polynomial.coefficients

# =============================================================================
# Water Calibration Curve Generation
# =============================================================================

"""
    generate_water_calibration_curve(energies, weights; max_path_cm=50.0, n_points=100, reference_energy_keV=70.0)

Generate water calibration curve for BHC by simulating polychromatic
transmission through various water path lengths.

# Arguments
- `energies`: Energy bin centers (keV)
- `weights`: Photon fluence weights per energy bin

# Keyword Arguments
- `max_path_cm`: Maximum water path length to simulate (default: 50.0 cm)
- `n_points`: Number of calibration points (default: 100)
- `reference_energy_keV`: Reference energy for "true" monochromatic values (default: 70.0)

# Returns
- `(paths, measured, true_values)`: Tuple of calibration data
  - `paths`: Water path lengths (cm)
  - `measured`: Polychromatic line integrals (-log(I/I₀))
  - `true_values`: Monochromatic line integrals at reference energy

# Example
```julia
energies, weights = load_spectrum(120)
paths, measured, true_vals = generate_water_calibration_curve(energies, weights)

# Plot calibration curve
plot(measured, true_vals, xlabel="Measured", ylabel="True")
plot!(measured, measured, linestyle=:dash, label="No BHC")
```
"""
function generate_water_calibration_curve(
    energies::Vector,
    weights::Vector;
    max_path_cm::Real = 50.0,
    n_points::Int = 100,
    reference_energy_keV::Real = 70.0
)
    # Get water attenuation coefficients at each energy
    water = get_material(:water)
    μ_water = [compute_μ_at_energy(water, Float64(e)) for e in energies]
    μ_water_ref = compute_μ_at_energy(water, Float64(reference_energy_keV))

    # Normalize weights
    weights_norm = weights ./ sum(weights)

    # Generate path lengths
    paths = range(0.0, Float64(max_path_cm), length=n_points) |> collect

    # Compute polychromatic transmission for each path
    measured = zeros(Float64, n_points)
    true_values = zeros(Float64, n_points)

    for (i, path) in enumerate(paths)
        # Polychromatic: I = Σ wₑ × exp(-μₑ × path)
        I_poly = sum(weights_norm .* exp.(-μ_water .* path))
        measured[i] = -log(max(I_poly, 1e-10))

        # Monochromatic at reference energy
        true_values[i] = μ_water_ref * path
    end

    return (paths, measured, true_values)
end

# =============================================================================
# BHC Calibration (Polynomial Fitting)
# =============================================================================

"""
    calibrate_bhc(energies, weights; order=3, max_path_cm=50.0, n_points=100, reference_energy_keV=70.0)

Calibrate beam hardening correction by fitting polynomial to water calibration curve.

This function generates a calibration curve by simulating polychromatic
X-ray transmission through water at various thicknesses, then fits a
polynomial to map measured (hardened) line integrals to "true"
(monochromatic-equivalent) values.

# Algorithm (CatSim-compatible)

1. For each water path d from 0 to max_path_cm:
   - Compute polychromatic transmission: I = Σᵢ wᵢ × exp(-μᵢ × d)
   - Measured line integral: p_meas = -log(I)
   - True line integral: p_true = μ_ref × d

2. Fit polynomial using least squares: p_true = Σⱼ aⱼ × p_meas^j

3. Return BeamHardeningCorrection with coefficients and calibration data

# Arguments
- `energies::Vector`: Energy bin centers (keV)
- `weights::Vector`: Photon fluence weights per energy bin (will be normalized)

# Keyword Arguments
- `order::Int=3`: Polynomial order. Recommended: 5 for clinical accuracy.
- `max_path_cm::Real=50.0`: Maximum water path length to calibrate (cm).
   Should cover expected body diameters (~40cm for abdominal CT).
- `n_points::Int=100`: Number of calibration points. More points improve
   polynomial stability but increase computation.
- `reference_energy_keV::Real=70.0`: Reference energy for "true" values (keV).
   Typically 60-70 keV for clinical CT (effective energy of filtered spectrum).

# Returns
- `BeamHardeningCorrection`: Complete BHC model with polynomial and calibration data

# GPU Compatibility
- ✅ Metal (via AcceleratedKernels.jl)
- ✅ CUDA
- ✅ CPU fallback

# Example

```julia
# Calibrate from spectrum
energies, weights = load_spectrum(120)
bhc = calibrate_bhc(energies, weights; order=5, reference_energy_keV=70.0)

# Apply to polychromatic sinogram (post-log transform)
apply_bhc!(sinogram, bhc)

apply_bhc!(sinogram, bhc)
```
"""
function calibrate_bhc(
    energies::Vector,
    weights::Vector;
    order::Int = 3,
    max_path_cm::Real = 50.0,
    n_points::Int = 100,
    reference_energy_keV::Real = 70.0
)
    # Generate calibration curve
    paths, measured, true_values = generate_water_calibration_curve(
        energies, weights;
        max_path_cm = max_path_cm,
        n_points = n_points,
        reference_energy_keV = reference_energy_keV
    )

    # Fit polynomial: true = Σ aᵢ × measured^i
    # Using simple least squares via normal equations
    coefficients = fit_polynomial(measured, true_values, order)

    polynomial = BHCPolynomial(coefficients, order, Float64(reference_energy_keV))

    return BeamHardeningCorrection(polynomial, paths, measured, true_values)
end

"""
    fit_polynomial(x, y, order)

Fit polynomial coefficients using least squares.
"""
function fit_polynomial(x::Vector, y::Vector, order::Int)
    n = length(x)

    # Build Vandermonde matrix
    V = zeros(n, order + 1)
    for i in 1:n
        for j in 0:order
            V[i, j+1] = x[i]^j
        end
    end

    # Solve normal equations: (VᵀV)a = Vᵀy
    coefficients = (V' * V) \ (V' * y)

    return coefficients
end

# =============================================================================
# BHC Application
# =============================================================================

"""
    apply_bhc!(sinogram, bhc)

Apply beam hardening correction to sinogram (in-place).

This function applies the polynomial BHC to each value in the sinogram,
mapping measured (beam-hardened) line integrals to corrected values
that approximate what would have been measured with a monochromatic
beam at the reference energy.

# Mathematical Operation

For each sinogram value p:

    p_corrected = a₀ + a₁×p + a₂×p² + ... + aₙ×pⁿ

where [a₀, a₁, ..., aₙ] are the polynomial coefficients.

# Signal Chain Position

BHC should be applied **after log transform, before reconstruction**:

1. Forward projection (polychromatic)
2. Air scan calibration
3. Log transform: p = -log(I/I₀)
4. **BHC ← applied here**
5. FDK/iterative reconstruction

# Arguments
- `sinogram::AbstractArray{T,3}`: Line integral sinogram [n_cols, n_rows, n_angles]
   where `T <: AbstractFloat`. Modified in place.
- `bhc::Union{BeamHardeningCorrection, BHCPolynomial}`: BHC model with
   polynomial coefficients.

# Returns
- The modified `sinogram` array (same object, mutated)

# GPU Compatibility
- ✅ Metal (via AcceleratedKernels.jl)
- ✅ CUDA
- ✅ CPU fallback

# Performance Notes

The polynomial evaluation uses Horner's method (iterative multiplication)
which is GPU-efficient and numerically stable.

# Example

```julia
energies, weights = load_spectrum(120)
bhc = calibrate_bhc(energies, weights; order=5)
apply_bhc!(sinogram, bhc)
```
"""
function apply_bhc!(
    sinogram::AbstractArray{T, 3},
    bhc::BeamHardeningCorrection;
    ws_coeffs_gpu=nothing
) where T <: AbstractFloat
    return apply_bhc!(sinogram, bhc.polynomial; ws_coeffs_gpu=ws_coeffs_gpu)
end

function apply_bhc!(
    sinogram::AbstractArray{T, 3},
    poly::BHCPolynomial;
    ws_coeffs_gpu=nothing
) where T <: AbstractFloat

    order = poly.order

    # Transfer coefficients to GPU (similar pattern for GPU compatibility)
    # Use pre-allocated buffer if provided
    coeffs = if ws_coeffs_gpu !== nothing
        ws_coeffs_gpu
    else
        coeffs_cpu = T.(poly.coefficients)
        c = similar(sinogram, T, length(coeffs_cpu))
        copyto!(c, coeffs_cpu)
        c
    end

    let coeffs = coeffs, order = order
        AK.foreachindex(sinogram) do idx
            p = sinogram[idx]

            # Evaluate polynomial: p_corrected = Σ aᵢ × p^i
            p_corrected = coeffs[1]
            p_power = one(T)
            for i in 1:order
                p_power *= p
                p_corrected += coeffs[i+1] * p_power
            end

            sinogram[idx] = p_corrected
        end
    end

    return sinogram
end

"""
    apply_bhc!(sinogram, bhcs_per_col::AbstractVector{BeamHardeningCorrection}; ws_coeffs_gpu=nothing)

Per-column BHC (bowtie-aware), convenience wrapper that unwraps the
`BeamHardeningCorrection` objects and delegates to the `BHCPolynomial`-vector
method below.
"""
function apply_bhc!(
    sinogram::AbstractArray{T, 3},
    bhcs_per_col::AbstractVector{BeamHardeningCorrection};
    ws_coeffs_gpu=nothing
) where T <: AbstractFloat
    polys = [b.polynomial for b in bhcs_per_col]
    return apply_bhc!(sinogram, polys; ws_coeffs_gpu=ws_coeffs_gpu)
end

"""
    apply_bhc!(sinogram, polys_per_col; ws_coeffs_gpu=nothing) -> sinogram

Per-column (bowtie-aware) BHC: applies a different polynomial to each detector
column, indexed off the sinogram's BS storage `[n_col, n_row, n_view]` layout.

Each `polys_per_col[c]` is a `BHCPolynomial` fit from the spectrum that column
actually saw (tube × filters × bowtie at fan angle `c`).  Coefficients are
stacked into a `(order+1) × n_col` matrix on the backend once, then the kernel
indexes `c = (idx-1) % n_col + 1` per ray.

All polynomials must share the same `order`.  Falls back to the single-poly
method automatically via multiple dispatch.
"""
function apply_bhc!(
    sinogram::AbstractArray{T, 3},
    polys_per_col::AbstractVector{BHCPolynomial};
    ws_coeffs_gpu=nothing
) where T <: AbstractFloat

    n_col = size(sinogram, 1)
    length(polys_per_col) == n_col ||
        error("apply_bhc!: polys_per_col length $(length(polys_per_col)) ≠ n_col $n_col")
    order = polys_per_col[1].order
    for c in 2:n_col
        polys_per_col[c].order == order ||
            error("apply_bhc!: polynomial orders must match across columns (col $c has order $(polys_per_col[c].order), expected $order)")
    end

    # Stack coefficients into a [order+1, n_col] matrix on the device.
    coeffs_cpu = zeros(T, order + 1, n_col)
    for col_idx in 1:n_col
        coeffs_cpu[:, col_idx] .= T.(polys_per_col[col_idx].coefficients)
    end
    coeffs = if ws_coeffs_gpu !== nothing
        copyto!(ws_coeffs_gpu, coeffs_cpu)
        ws_coeffs_gpu
    else
        cbuf = similar(sinogram, T, order + 1, n_col)
        copyto!(cbuf, coeffs_cpu)
        cbuf
    end

    # NOTE: renamed the local `col` to avoid collision with any outer `c`
    # scoped variable — same-name assignments in closures trigger Julia's
    # Core.Box heuristic and break GPU kernel compile with "non-bitstype
    # argument" errors.
    let coeffs = coeffs, order = order, n_col = n_col
        AK.foreachindex(sinogram) do idx
            col = ((idx - 1) % n_col) + 1
            p = sinogram[idx]
            p_corrected = coeffs[1, col]
            p_power = one(T)
            for i in 1:order
                p_power *= p
                p_corrected += coeffs[i+1, col] * p_power
            end
            sinogram[idx] = p_corrected
        end
    end

    return sinogram
end

# =============================================================================
# Two-Material (Water + Bone) Beam Hardening Correction
# =============================================================================
#
# Implements the Martinez/Fessler 2022 "2DCalBH" algorithm adapted for
# simulation where the spectrum is exactly known.
#
# References:
#   1. Martinez C, Fessler JA, et al. Phys Med Biol. 2022;67(11).
#   2. Joseph PM, Spital RD. J Comput Assist Tomogr. 1978;2(1):100-108.
#   3. Elbakri IA, Fessler JA. Phys Med Biol. 2003;48(15):2453-2477.
# =============================================================================

"""
    bone_fraction_smooth(hu; hu_low=100.0, hu_high=500.0) -> Float64

Compute smooth bone fraction for tissue decomposition (Elbakri/Fessler 2003).

Uses C1-continuous smoothstep (3t² - 2t³) to avoid hard segmentation artifacts.
Returns 0.0 for soft tissue (≤ hu_low), 1.0 for bone (≥ hu_high), smooth
transition between.
"""
function bone_fraction_smooth(hu::Real; hu_low::Real=100.0, hu_high::Real=500.0)
    hu <= hu_low && return 0.0
    hu >= hu_high && return 1.0
    t = (hu - hu_low) / (hu_high - hu_low)
    return t * t * (3.0 - 2.0 * t)  # C1 smoothstep
end

"""
    TwoMaterialBHC

Two-material (water + bone) beam hardening correction model.

Single-spectrum variant: one polynomial, one mean spectrum.  Assumes every
ray sees the same spectrum (valid when there's no bowtie, or when bowtie
is ignored).  Use `TwoMaterialBHCPerColumn` for bowtie-aware correction.
"""
struct TwoMaterialBHC
    water_bhc::BeamHardeningCorrection
    energies::Vector{Float64}
    w_norm::Vector{Float64}
    μ_water_E::Vector{Float64}
    μ_bone_E::Vector{Float64}
    μ_water_ref::Float64
    μ_bone_ref::Float64
    reference_energy_keV::Float64
    hu_low::Float64
    hu_high::Float64
end

"""
    TwoMaterialBHCPerColumn

Bowtie-aware two-material BHC: one polynomial per detector column, each fit
from the spectrum that column actually saw (tube × filters × bowtie at that
fan angle).  Eliminates the residual radial cupping that a single global
polynomial leaves behind when a bowtie is present.

- `water_bhc_per_col[c]` — `BeamHardeningCorrection` fit for column `c`.
  Length equals the sinogram's `n_col`.
- `w_norm_per_col` — normalized spectrum per column, `[n_E × n_col]`.  Used
  by the second-stage bone correction for the per-column polychromatic forward.
- `μ_water_ref`, `μ_bone_ref` — **scalar** global refs, same as `TwoMaterialBHC`.
  The mono-equivalent sinogram produced by the per-column polynomials targets
  this single global reference energy so FBP downstream remains physically
  consistent.

Share the same HU thresholds (`hu_low`, `hu_high`) and material μ(E) tables
as `TwoMaterialBHC`.
"""
struct TwoMaterialBHCPerColumn
    water_bhc_per_col::Vector{BeamHardeningCorrection}
    energies::Vector{Float64}
    w_norm_per_col::Matrix{Float64}      # [n_E, n_col]
    μ_water_E::Vector{Float64}
    μ_bone_E::Vector{Float64}
    μ_water_ref::Float64
    μ_bone_ref::Float64
    reference_energy_keV::Float64
    hu_low::Float64
    hu_high::Float64
end

"""
    calibrate_bhc_two_material(energies, weights; kwargs...) -> TwoMaterialBHC
    calibrate_bhc_two_material(energies, weights_per_col; kwargs...) -> TwoMaterialBHCPerColumn

Calibrate two-material BHC from a spectrum.

- Passing `weights::Vector` returns a single-polynomial `TwoMaterialBHC`.
- Passing `weights::Matrix` of shape `[n_E, n_col]` returns a
  `TwoMaterialBHCPerColumn` with one polynomial fit per column — that's the
  bowtie-aware path.  Build the matrix from
  `resolve_source_spectrum_with_bowtie(...)` output (collapse the row axis
  first — bowtie varies primarily with fan angle, not row).

The material μ(E) tables, HU thresholds, and the global `μ_water_ref` /
`μ_bone_ref` at `reference_energy_keV` are shared across all columns in the
bowtie-aware case — only the polynomial coefficients and per-column
normalized spectrum differ.
"""
function calibrate_bhc_two_material(
    energies::Vector,
    weights::Vector;
    order::Int = 5,
    max_path_cm::Real = 50.0,
    n_points::Int = 100,
    reference_energy_keV::Real = 70.0,
    hu_low::Real = 100.0,
    hu_high::Real = 500.0
)
    water_bhc = calibrate_bhc(energies, weights;
        order=order, max_path_cm=max_path_cm,
        n_points=n_points, reference_energy_keV=reference_energy_keV)

    water_mat = XA.Materials.water
    bone_mat = XA.Materials.corticalbone

    E_vec = Float64.(energies)
    w_norm = Float64.(weights) ./ sum(Float64.(weights))

    μ_water_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, E * u"keV")) for E in E_vec]
    μ_bone_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, E * u"keV")) for E in E_vec]
    μ_water_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, reference_energy_keV * u"keV"))
    μ_bone_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, reference_energy_keV * u"keV"))

    return TwoMaterialBHC(water_bhc, E_vec, w_norm, μ_water_E, μ_bone_E,
                          μ_water_ref, μ_bone_ref, Float64(reference_energy_keV),
                          Float64(hu_low), Float64(hu_high))
end

function calibrate_bhc_two_material(
    energies::Vector,
    weights_per_col::AbstractMatrix;
    order::Int = 5,
    max_path_cm::Real = 50.0,
    n_points::Int = 100,
    reference_energy_keV::Real = 70.0,
    hu_low::Real = 100.0,
    hu_high::Real = 500.0
)
    n_E, n_col = size(weights_per_col)
    length(energies) == n_E ||
        error("calibrate_bhc_two_material: energies length $(length(energies)) ≠ weights_per_col n_E $n_E")

    # Fit one water-BHC polynomial per column using that column's spectrum.
    water_bhc_per_col = Vector{BeamHardeningCorrection}(undef, n_col)
    w_norm_per_col = zeros(Float64, n_E, n_col)
    for c in 1:n_col
        w_c = Float64.(@view weights_per_col[:, c])
        w_norm_per_col[:, c] .= w_c ./ sum(w_c)
        water_bhc_per_col[c] = calibrate_bhc(energies, w_c;
            order=order, max_path_cm=max_path_cm,
            n_points=n_points, reference_energy_keV=reference_energy_keV)
    end

    water_mat = XA.Materials.water
    bone_mat = XA.Materials.corticalbone
    E_vec = Float64.(energies)
    μ_water_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, E * u"keV")) for E in E_vec]
    μ_bone_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, E * u"keV")) for E in E_vec]
    μ_water_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, reference_energy_keV * u"keV"))
    μ_bone_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, reference_energy_keV * u"keV"))

    @info "[calibrate_bhc_two_material] bowtie-aware: $n_col columns × order-$order polynomials, ref_E = $reference_energy_keV keV"
    return TwoMaterialBHCPerColumn(water_bhc_per_col, E_vec, w_norm_per_col,
                                    μ_water_E, μ_bone_E,
                                    μ_water_ref, μ_bone_ref,
                                    Float64(reference_energy_keV),
                                    Float64(hu_low), Float64(hu_high))
end

"""
    bhc_spectrum_per_column(e, w) -> (e, w_per_col)

Reduce a bowtie-aware 3D `ŵ[n_col, n_row, n_E]` (from
`resolve_source_spectrum_with_bowtie`) to 2D `w[n_E, n_col]` by taking the
centered detector row.  Bowtie variation is primarily in-plane (fan angle);
row-direction variation is small, so a center-row slice is an excellent
approximation for BHC calibration.

If `w` is already 1D or 2D, it's returned as-is (after transpose if 2D comes
in as `[n_col, n_E]` rather than `[n_E, n_col]`).

Usage:
```julia
e, ŵ = BS.resolve_source_spectrum_with_bowtie(sim_opts, prot; scanner, geom)
e, w_col = BS.bhc_spectrum_per_column(e, ŵ)
bhc = BS.calibrate_bhc_two_material(e, w_col; ...)
```
"""
function bhc_spectrum_per_column(e::AbstractVector, w::AbstractArray)
    if ndims(w) == 1
        return e, w
    elseif ndims(w) == 2
        # Accept either [n_E, n_col] or [n_col, n_E] and normalize to [n_E, n_col].
        size(w, 1) == length(e) && return e, w
        size(w, 2) == length(e) && return e, permutedims(w)
        error("bhc_spectrum_per_column: 2D spectrum shape $(size(w)) doesn't match energies length $(length(e))")
    elseif ndims(w) == 3
        # 3D bowtie-aware: [n_col, n_row, n_E] → take center row → [n_E, n_col]
        n_col, n_row, n_E = size(w)
        n_E == length(e) ||
            error("bhc_spectrum_per_column: 3D spectrum last-dim $n_E ≠ energies length $(length(e))")
        mid_r = n_row ÷ 2 + 1
        out = Array{Float64}(undef, n_E, n_col)
        @inbounds for c in 1:n_col, k in 1:n_E
            out[k, c] = Float64(w[c, mid_r, k])
        end
        return e, out
    else
        error("bhc_spectrum_per_column: unsupported ndims $(ndims(w))")
    end
end

"""
    compute_polychromatic_μ_water(sim_opts, protocol;
                                  scanner, geom, water_path_cm) -> Float64

Spectrum-analytic polychromatic-effective `μ_water` for an in-phantom
voxel — the analytic counterpart to "average μ in a solid-water ROI of
the FBP volume".  Use this as the HU-conversion reference for image-
domain pipelines that don't run BHC.

Pipeline:
1. Resolve the bowtie-aware source spectrum via
   [`resolve_source_spectrum_with_bowtie`](@ref) — picks up the scanner
   flat filter, `protocol.additional_filters`, and the bowtie attenuation.
2. Collapse the per-ray ŵ to the central-column 1D fluence profile via
   [`bhc_spectrum_per_column`](@ref).
3. **Pre-harden** the spectrum by `water_path_cm` of solid water at each
   energy bin via Beer-Lambert (low-E photons drop out preferentially,
   raising the mean energy — same effect a center-of-phantom voxel sees).
4. Return the fluence-weighted μ_water on the hardened spectrum.

# Arguments
- `sim_opts::SimOptions`
- `protocol::CTProtocol`

# Keyword arguments
- `scanner::Scanner` — flat filter + bowtie geometry
- `geom::CTGeometry` — for bowtie attenuation per column
- `water_path_cm::Real` — total chord-length of solid water seen by an
  FBP line integral through the reference voxel.  For a center voxel,
  the chord crosses the **full phantom diameter** (one entry + one exit
  half), so pass the phantom *diameter*, e.g. `33.0` for a Gammex 472
  (33 cm body), not the radius.

# Returns
- `μ_water::Float64` — linear attenuation coefficient (cm⁻¹), suitable
  for `to_hounsfield(...; μ_water = μ_water)`.

# Example
```julia
μw_80 = BS.compute_polychromatic_μ_water(sim_opts, protocol_low;
    scanner = scanner, geom = sim_low.geom, water_path_cm = 33.0)
hu = BS.to_hounsfield(vol_low_μ; μ_water = μw_80)
```
"""
function compute_polychromatic_μ_water(
        sim_opts,
        protocol;
        scanner,
        geom,
        water_path_cm::Real,
    )
    e, ŵ = resolve_source_spectrum_with_bowtie(
        sim_opts, protocol;
        scanner = scanner, geom = geom, include_bowtie = true,
        label = "compute_polychromatic_μ_water $(Int(protocol.kVp)) kVp",
    )
    _, w_per_col = bhc_spectrum_per_column(e, ŵ)
    mid_c   = size(w_per_col, 2) ÷ 2 + 1
    w_entr  = Float64.(@view w_per_col[:, mid_c])
    μ_per_E = Float64[compute_μ_at_energy(XA.Materials.water, Float64(eᵢ)) for eᵢ in e]
    w_hard  = w_entr .* exp.(-μ_per_E .* Float64(water_path_cm))
    return sum(w_hard .* μ_per_E) / sum(w_hard)
end


"""
    apply_bhc_two_material(sinogram_raw, bhc_2mat, geom, matrix_size; volume_extent=nothing)

Apply two-material (water + bone) beam hardening correction in sinogram domain.

Dispatches on the BHC type:
- `TwoMaterialBHC` — single polynomial water stage (1D spectrum).
- `TwoMaterialBHCPerColumn` — per-column polynomial water stage (bowtie-aware).
  Bone stage uses per-column spectrum + global μ refs.
"""
function apply_bhc_two_material(
    sinogram_raw::AbstractArray{T, 3},
    bhc_2mat::TwoMaterialBHC,
    geom,
    matrix_size::Tuple{Int,Int,Int};
    volume_extent=nothing
) where T <: AbstractFloat

    sino_water = similar(sinogram_raw)
    copyto!(sino_water, sinogram_raw)
    apply_bhc!(sino_water, bhc_2mat.water_bhc)
    recon_gpu = fdk_reconstruct(sino_water, geom, matrix_size)

    μ_w_ref = T(bhc_2mat.μ_water_ref)
    hu_low_T = T(bhc_2mat.hu_low)
    hu_high_T = T(bhc_2mat.hu_high)

    bone_μ_gpu = similar(recon_gpu)
    let recon_gpu = recon_gpu, bone_μ_gpu = bone_μ_gpu,
        μ_w_ref = μ_w_ref, hu_low_T = hu_low_T, hu_high_T = hu_high_T
        AK.foreachindex(recon_gpu) do idx
            μ_val = recon_gpu[idx]
            hu_val = T(1000) * (μ_val - μ_w_ref) / μ_w_ref
            bf = if hu_val <= hu_low_T
                zero(T)
            elseif hu_val >= hu_high_T
                one(T)
            else
                t = (hu_val - hu_low_T) / (hu_high_T - hu_low_T)
                t * t * (T(3) - T(2) * t)
            end
            bone_μ_gpu[idx] = bf * μ_val
        end
    end

    p_b_gpu = siddon_forward_project(bone_μ_gpu, geom; volume_extent=volume_extent)

    p_s_gpu = similar(sino_water)
    copyto!(p_s_gpu, sino_water)
    let p_s_gpu = p_s_gpu, p_b_gpu = p_b_gpu
        AK.foreachindex(p_s_gpu) do idx
            p_s_gpu[idx] = p_s_gpu[idx] - p_b_gpu[idx]
        end
    end

    n_e = length(bhc_2mat.w_norm)
    w_norm_cpu = T.(bhc_2mat.w_norm)
    μ_w_cpu = T.(bhc_2mat.μ_water_E)
    μ_b_cpu = T.(bhc_2mat.μ_bone_E)
    w_norm_gpu = similar(sinogram_raw, T, n_e)
    μ_w_gpu = similar(sinogram_raw, T, n_e)
    μ_b_gpu = similar(sinogram_raw, T, n_e)
    copyto!(w_norm_gpu, w_norm_cpu)
    copyto!(μ_w_gpu, μ_w_cpu)
    copyto!(μ_b_gpu, μ_b_cpu)

    μ_b_ref = T(bhc_2mat.μ_bone_ref)

    sino_out = similar(sinogram_raw)
    copyto!(sino_out, sinogram_raw)

    let sino_out = sino_out, p_s_gpu = p_s_gpu, p_b_gpu = p_b_gpu,
        w_norm_gpu = w_norm_gpu, μ_w_gpu = μ_w_gpu, μ_b_gpu = μ_b_gpu,
        μ_w_ref = μ_w_ref, μ_b_ref = μ_b_ref, n_e = n_e
        AK.foreachindex(sino_out) do idx
            Lw = max(p_s_gpu[idx] / μ_w_ref, zero(T))
            Lb = max(p_b_gpu[idx] / μ_b_ref, zero(T))
            if Lw > T(1e-6) || Lb > T(1e-6)
                I_poly = zero(T)
                for e in 1:n_e
                    I_poly += w_norm_gpu[e] * exp(-μ_w_gpu[e] * Lw - μ_b_gpu[e] * Lb)
                end
                F_2mat = I_poly > zero(T) ? -log(I_poly) : zero(T)
                p_mono = μ_w_ref * Lw + μ_b_ref * Lb
                sino_out[idx] = sino_out[idx] + (p_mono - F_2mat)
            end
        end
    end

    return Array(sino_out)
end

"""
    apply_bhc_two_material(sinogram_raw, bhc::TwoMaterialBHCPerColumn, geom, matrix_size; volume_extent=nothing)

Bowtie-aware two-material BHC.  Water stage uses the per-column polynomial;
bone stage uses per-column normalized spectrum for the polychromatic forward
model.  Global `μ_water_ref` / `μ_bone_ref` keep the mono-equivalent target
consistent across columns so downstream FBP sees a single effective energy.
"""
function apply_bhc_two_material(
    sinogram_raw::AbstractArray{T, 3},
    bhc::TwoMaterialBHCPerColumn,
    geom,
    matrix_size::Tuple{Int,Int,Int};
    volume_extent=nothing
) where T <: AbstractFloat

    n_col = size(sinogram_raw, 1)
    length(bhc.water_bhc_per_col) == n_col ||
        error("apply_bhc_two_material(PerColumn): polys $(length(bhc.water_bhc_per_col)) ≠ sino n_col $n_col")

    # Stage 1 — per-column polynomial water BHC.
    sino_water = similar(sinogram_raw)
    copyto!(sino_water, sinogram_raw)
    apply_bhc!(sino_water, bhc.water_bhc_per_col)
    recon_gpu = fdk_reconstruct(sino_water, geom, matrix_size)

    μ_w_ref   = T(bhc.μ_water_ref)
    μ_b_ref   = T(bhc.μ_bone_ref)
    hu_low_T  = T(bhc.hu_low)
    hu_high_T = T(bhc.hu_high)

    # Stage 2 — bone-fraction segmentation in image domain.
    bone_μ_gpu = similar(recon_gpu)
    let recon_gpu = recon_gpu, bone_μ_gpu = bone_μ_gpu,
        μ_w_ref = μ_w_ref, hu_low_T = hu_low_T, hu_high_T = hu_high_T
        AK.foreachindex(recon_gpu) do idx
            μ_val = recon_gpu[idx]
            hu_val = T(1000) * (μ_val - μ_w_ref) / μ_w_ref
            bf = if hu_val <= hu_low_T
                zero(T)
            elseif hu_val >= hu_high_T
                one(T)
            else
                t = (hu_val - hu_low_T) / (hu_high_T - hu_low_T)
                t * t * (T(3) - T(2) * t)
            end
            bone_μ_gpu[idx] = bf * μ_val
        end
    end

    p_b_gpu = siddon_forward_project(bone_μ_gpu, geom; volume_extent=volume_extent)

    p_s_gpu = similar(sino_water)
    copyto!(p_s_gpu, sino_water)
    let p_s_gpu = p_s_gpu, p_b_gpu = p_b_gpu
        AK.foreachindex(p_s_gpu) do idx
            p_s_gpu[idx] = p_s_gpu[idx] - p_b_gpu[idx]
        end
    end

    # Stage 3 — per-column polychromatic forward correction.  The w_norm matrix
    # is [n_E, n_col]; kernel indexes w_norm[e, c] for col c.
    n_e = length(bhc.energies)
    μ_w_cpu      = T.(bhc.μ_water_E)
    μ_b_cpu      = T.(bhc.μ_bone_E)
    w_norm_cpu   = T.(bhc.w_norm_per_col)  # [n_E, n_col]
    μ_w_gpu      = similar(sinogram_raw, T, n_e);           copyto!(μ_w_gpu,      μ_w_cpu)
    μ_b_gpu      = similar(sinogram_raw, T, n_e);           copyto!(μ_b_gpu,      μ_b_cpu)
    w_norm_gpu   = similar(sinogram_raw, T, n_e, n_col);    copyto!(w_norm_gpu,   w_norm_cpu)

    sino_out = similar(sinogram_raw)
    copyto!(sino_out, sinogram_raw)

    let sino_out = sino_out, p_s_gpu = p_s_gpu, p_b_gpu = p_b_gpu,
        w_norm_gpu = w_norm_gpu, μ_w_gpu = μ_w_gpu, μ_b_gpu = μ_b_gpu,
        μ_w_ref = μ_w_ref, μ_b_ref = μ_b_ref, n_e = n_e, n_col = n_col
        AK.foreachindex(sino_out) do idx
            col = ((idx - 1) % n_col) + 1
            Lw = max(p_s_gpu[idx] / μ_w_ref, zero(T))
            Lb = max(p_b_gpu[idx] / μ_b_ref, zero(T))
            if Lw > T(1e-6) || Lb > T(1e-6)
                I_poly = zero(T)
                for e in 1:n_e
                    I_poly += w_norm_gpu[e, col] * exp(-μ_w_gpu[e] * Lw - μ_b_gpu[e] * Lb)
                end
                F_2mat = I_poly > zero(T) ? -log(I_poly) : zero(T)
                p_mono = μ_w_ref * Lw + μ_b_ref * Lb
                sino_out[idx] = sino_out[idx] + (p_mono - F_2mat)
            end
        end
    end

    return Array(sino_out)
end

# =============================================================================
# Image-Domain BHC (So et al. 2009)
# =============================================================================
#
# Post-reconstruction beam hardening correction based on:
#   So A, Hsieh J, Li JY, Lee TY. "Beam hardening correction in CT myocardial
#   perfusion measurement." Phys Med Biol. 2009;54(10):3031-3050.
#
# Algorithm:
#   1. Start with water-BHC-corrected reconstructed image (μ domain)
#   2. Segment high-attenuation voxels via smoothstep threshold → weighting factor
#   3. Forward-project the weighted high-attenuation image → ξ (error sinogram)
#   4. Reconstruct the error sinogram via FDK → error image
#   5. Subtract scaled error image from original: corrected = original - scale × error
#
# Two tunable parameters:
#   - Threshold range [hu_low, hu_high]: what HU range counts as "high attenuation"
#   - Scale factor (0.0–1.0): strength of correction
#
# The scale factor empirically absorbs all polynomial coefficient differences
# (α_i - β_i from equation 8 in the paper) into a single tunable knob.
# =============================================================================

"""
    apply_bhc_image_domain(recon_μ, geom, matrix_size, μ_water_ref;
        hu_low=70.0, hu_high=150.0, scale_factor=0.5, volume_extent=nothing)

Apply image-domain beam hardening correction (So et al. 2009).

Corrects BH artifacts from high-attenuation materials (bone, contrast) by
segmenting the reconstructed image, forward-projecting the dense component,
reconstructing the resulting error sinogram, and subtracting a scaled version
of the error image from the original.

# Arguments
- `recon_μ::AbstractArray{T,3}`: Reconstructed image in μ (linear attenuation, cm⁻¹),
  already corrected with water-only polynomial BHC. GPU array.
- `geom`: CTGeometry used for reconstruction
- `matrix_size::Tuple{Int,Int,Int}`: Reconstruction matrix size (nx, ny, nz)
- `μ_water_ref::Real`: Water linear attenuation at reference energy (cm⁻¹)

# Keyword Arguments
- `hu_low::Real=70.0`: Lower HU threshold — voxels below this are 100% soft tissue (WF=0)
- `hu_high::Real=150.0`: Upper HU threshold — voxels above this are 100% high-atten (WF=1)
- `scale_factor::Real=0.5`: Scaling factor for error image subtraction (0.0 = no correction,
  1.0 = full subtraction). Typical range: 0.2–0.7.
- `volume_extent::Union{Nothing,NTuple{3,Float64}}=nothing`: Physical extent (cm) for
  forward projection. Pass `phantom.extent` for correct geometry.

# Returns
- `AbstractArray{T,3}`: Corrected image in μ domain (same type/device as input)

# Tuning Guide
- **hu_low/hu_high**: Use a narrow range (e.g., 70–150 or 100–200 HU) for contrast-only
  correction. Use a wider range to include bone. Paper recommends 100–150 HU.
- **scale_factor**: Start at 0.5. Increase if HU offset persists (under-correction).
  Decrease if artifacts appear (over-correction). Optimal is typically 0.2–0.7.

# References
So A, Hsieh J, Li JY, Lee TY. Phys Med Biol. 2009;54(10):3031-3050.
"""
function apply_bhc_image_domain(
    recon_μ::AbstractArray{T, 3},
    geom,
    matrix_size::Tuple{Int,Int,Int},
    μ_water_ref::Real;
    hu_low::Real = 70.0,
    hu_high::Real = 150.0,
    scale_factor::Real = 0.5,
    volume_extent = nothing
) where T <: AbstractFloat

    μ_w_ref = T(μ_water_ref)
    hu_low_T = T(hu_low)
    hu_high_T = T(hu_high)

    # Step 1: Segment high-attenuation voxels → weighted μ image (GPU)
    high_atten_μ = similar(recon_μ)
    let recon_μ = recon_μ, high_atten_μ = high_atten_μ,
        μ_w_ref = μ_w_ref, hu_low_T = hu_low_T, hu_high_T = hu_high_T
        AK.foreachindex(recon_μ) do idx
            μ_val = recon_μ[idx]
            hu_val = T(1000) * (μ_val - μ_w_ref) / μ_w_ref

            # Linear interpolation weighting factor (So et al. eq. in section 2.2)
            wf = if hu_val <= hu_low_T
                zero(T)
            elseif hu_val >= hu_high_T
                one(T)
            else
                (hu_val - hu_low_T) / (hu_high_T - hu_low_T)
            end

            # Paper works in HU space (water=0); in μ space we must subtract
            # the water baseline so we only capture excess attenuation above water.
            high_atten_μ[idx] = wf * (μ_val - μ_w_ref)
        end
    end

    # Step 2: Forward-project the high-attenuation image → ξ (error sinogram)
    ξ_gpu = siddon_forward_project(high_atten_μ, geom; volume_extent=volume_extent)

    # Step 3: Reconstruct the error sinogram → error image
    error_image = fdk_reconstruct(ξ_gpu, geom, matrix_size)

    # Step 4: Subtract scaled error image from original
    sf = T(scale_factor)
    let recon_μ = recon_μ, error_image = error_image, sf = sf
        AK.foreachindex(recon_μ) do idx
            recon_μ[idx] = recon_μ[idx] - sf * error_image[idx]
        end
    end

    return recon_μ
end

# =============================================================================
# Radial Cupping / Capping Correction (Post-Reconstruction)
# =============================================================================
#
# Corrects residual low-frequency radial non-uniformity in the HU image
# after all sinogram-domain and image-domain BHC stages.
#
# Works by fitting a low-order even polynomial in radius to the water-like
# background (excluding air and high-contrast inserts), then subtracting
# the fitted offset so the background becomes flat.
#
# This handles both cupping (center-dark) and capping (center-bright).
# =============================================================================

export apply_radial_cupping_correction!

"""
    apply_radial_cupping_correction!(hu_vol; fov_cm, hu_lo, hu_hi, poly_order, target_hu)

Post-reconstruction radial correction for residual cupping/capping artifacts.

Fits a low-order even polynomial in radius to the water-like background
(excluding air and inserts), then subtracts the fitted offset so the
background becomes flat at `target_hu`.

Operates slice-by-slice. Modifies `hu_vol` in place.

# Arguments
- `hu_vol::Array{Float32, 3}`: Reconstructed HU volume (modified in place)

# Keyword Arguments
- `fov_cm::Float64=35.0`: Field of view in cm
- `hu_lo::Float64=-100.0`: Lower HU threshold for water-like voxels
- `hu_hi::Float64=80.0`: Upper HU threshold for water-like voxels
- `poly_order::Int=2`: Number of even polynomial terms (fits c₀ + c₁r² + c₂r⁴ + ...)
- `target_hu::Float64=0.0`: Target HU for the water background after correction

# Returns
- The modified `hu_vol` array
"""
function apply_radial_cupping_correction!(
    hu_vol::Array{Float32, 3};
    fov_cm::Float64 = 35.0,
    hu_lo::Float64 = -100.0,
    hu_hi::Float64 = 80.0,
    poly_order::Int = 2,
    target_hu::Float64 = 0.0,
)
    nx, ny, nz = size(hu_vol)
    pixel_cm = fov_cm / nx
    cx = (nx + 1) / 2.0
    cy = (ny + 1) / 2.0

    for iz in 1:nz
        slice = @view hu_vol[:, :, iz]

        # Collect (radius, HU) samples from water-like voxels
        radii = Float64[]
        vals = Float64[]
        for j in 1:ny, i in 1:nx
            v = slice[i, j]
            if hu_lo <= v <= hu_hi
                r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
                push!(radii, r)
                push!(vals, Float64(v))
            end
        end

        length(radii) < 10 && continue

        # Fit even polynomial: offset = c₀ + c₁r² + c₂r⁴ + ...
        n_coeffs = poly_order + 1
        A = zeros(length(radii), n_coeffs)
        for (k, r) in enumerate(radii)
            for p in 0:poly_order
                A[k, p+1] = r^(2p)
            end
        end
        coeffs = A \ vals

        # Subtract fitted profile, shifting to target_hu
        for j in 1:ny, i in 1:nx
            r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
            fit_val = sum(coeffs[p+1] * r^(2p) for p in 0:poly_order)
            slice[i, j] -= Float32(fit_val - target_hu)
        end
    end

    return hu_vol
end
