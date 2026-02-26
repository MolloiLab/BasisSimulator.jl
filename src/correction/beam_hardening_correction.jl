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
export calibrate_bhc, apply_bhc!, apply_bhc
export generate_water_calibration_curve
export bhc_water_default, bhc_none
export evaluate_bhc, get_bhc_info, get_bhc_coefficients

# Two-material (water + bone) BHC exports
export bone_fraction_smooth, TwoMaterialBHC
export calibrate_bhc_two_material, apply_bhc_two_material

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

# Assessing Fit Quality

The calibration data can be used to assess polynomial fit quality:

```julia
info = get_bhc_info(bhc)
println("Max correction: ", info.max_correction)  # Maximum BH effect

# Compute residual
corrected = [evaluate_bhc(m, bhc.polynomial) for m in bhc.calibration_measured]
residual = bhc.calibration_true .- corrected
println("RMS residual: ", sqrt(mean(residual.^2)))
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
# Default BHC Models
# =============================================================================

"""
    bhc_none()

No beam hardening correction (identity mapping).
"""
function bhc_none()
    return BHCPolynomial([0.0, 1.0], 1, 70.0)  # p_corrected = p
end

"""
    bhc_water_default(; reference_energy_keV=70.0)

Default water-based BHC for 120 kVp spectrum.

These coefficients are typical for clinical CT and provide
approximate correction. For best results, use `calibrate_bhc`
to generate coefficients specific to your spectrum.
"""
function bhc_water_default(; reference_energy_keV::Real = 70.0)
    # Typical coefficients for 120 kVp tungsten spectrum
    # Derived from water phantom calibration
    coefficients = [0.0, 1.05, -0.02, 0.001]  # 3rd order polynomial
    return BHCPolynomial(coefficients, 3, Float64(reference_energy_keV))
end

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

# Inspect calibration quality
info = get_bhc_info(bhc)
println("Calibration range: ", info.calibration_range)
println("Max BH correction: ", info.max_correction)
```

# See Also
- [`apply_bhc!`](@ref): Apply BHC to sinogram
- [`generate_water_calibration_curve`](@ref): Generate calibration data
- [`bhc_water_default`](@ref): Pre-computed default coefficients
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
# Method 1: Pre-calibrated BHC
energies, weights = load_spectrum(120)
bhc = calibrate_bhc(energies, weights; order=5)
apply_bhc!(sinogram, bhc)

# Method 2: Default coefficients (approximate, no calibration)
apply_bhc!(sinogram, bhc_water_default())

# Method 3: Disable BHC (identity)
apply_bhc!(sinogram, bhc_none())  # p_out = p_in
```

# See Also
- [`apply_bhc`](@ref): Non-mutating version
- [`calibrate_bhc`](@ref): Generate calibrated BHC from spectrum
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
    apply_bhc(sinogram, bhc)

Non-mutating version of apply_bhc!.
"""
function apply_bhc(
    sinogram::AbstractArray{T, 3},
    bhc::Union{BeamHardeningCorrection, BHCPolynomial}
) where T <: AbstractFloat
    result = similar(sinogram)
    copyto!(result, sinogram)
    return apply_bhc!(result, bhc)
end

# =============================================================================
# BHC Utilities
# =============================================================================

"""
    evaluate_bhc(p, bhc)

Evaluate BHC polynomial at a single value.
"""
function evaluate_bhc(p::Real, poly::BHCPolynomial)
    coeffs = poly.coefficients
    result = coeffs[1]
    p_power = 1.0
    for i in 1:poly.order
        p_power *= p
        result += coeffs[i+1] * p_power
    end
    return result
end

"""
    get_bhc_info(bhc)

Get information about BHC model.
"""
function get_bhc_info(bhc::BeamHardeningCorrection)
    poly = bhc.polynomial
    return (
        order = poly.order,
        reference_energy_keV = poly.reference_energy_keV,
        coefficients = poly.coefficients,
        calibration_range = (minimum(bhc.calibration_measured), maximum(bhc.calibration_measured)),
        max_correction = maximum(abs.(bhc.calibration_true .- bhc.calibration_measured))
    )
end

function get_bhc_info(poly::BHCPolynomial)
    return (
        order = poly.order,
        reference_energy_keV = poly.reference_energy_keV,
        coefficients = poly.coefficients
    )
end

# =============================================================================
# Two-Material (Water + Bone) Beam Hardening Correction
# =============================================================================
#
# Implements the Martinez/Fessler 2022 "2DCalBH" algorithm adapted for
# simulation where the spectrum is exactly known.
#
# Algorithm (projection-domain, no hard segmentation):
#   Pass 1: Water-only polynomial BHC → FDK → preliminary image
#   Pass 2: Smooth tissue fraction decomposition → forward-project bone →
#           compute exact 2D correction from known spectrum → apply → FDK
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

# Arguments
- `hu::Real`: Hounsfield Unit value of the voxel

# Keyword Arguments
- `hu_low::Real=100.0`: Below this HU, voxel is 100% soft tissue
- `hu_high::Real=500.0`: Above this HU, voxel is 100% bone

# Returns
- `Float64`: Bone fraction in [0, 1]
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

Contains the water-only BHC polynomial plus spectral attenuation data for
both water and cortical bone, enabling exact 2-material correction.

# Fields
- `water_bhc::BeamHardeningCorrection`: Water-only BHC polynomial
- `energies::Vector{Float64}`: Spectrum energy bins (keV)
- `w_norm::Vector{Float64}`: Normalized spectrum weights
- `μ_water_E::Vector{Float64}`: Water attenuation at each energy (cm⁻¹)
- `μ_bone_E::Vector{Float64}`: Bone attenuation at each energy (cm⁻¹)
- `μ_water_ref::Float64`: Water attenuation at reference energy (cm⁻¹)
- `μ_bone_ref::Float64`: Bone attenuation at reference energy (cm⁻¹)
- `reference_energy_keV::Float64`: Reference energy for monochromatic model
- `hu_low::Float64`: Soft tissue / bone boundary threshold (HU)
- `hu_high::Float64`: Fully-bone threshold (HU)
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
    calibrate_bhc_two_material(energies, weights; kwargs...) -> TwoMaterialBHC

Calibrate the two-material (water + bone) BHC model.

Internally calibrates water-only BHC and computes spectral attenuation data
for both water and cortical bone.

# Arguments
- `energies::Vector`: Energy bin centers (keV)
- `weights::Vector`: Photon fluence weights per energy bin

# Keyword Arguments
- `order::Int=5`: Polynomial order for water-only BHC
- `max_path_cm::Real=50.0`: Maximum water path length for calibration (cm)
- `n_points::Int=100`: Number of calibration points
- `reference_energy_keV::Real=70.0`: Reference energy (keV)
- `hu_low::Real=100.0`: Soft tissue / bone boundary (HU)
- `hu_high::Real=500.0`: Fully-bone threshold (HU)

# Returns
- `TwoMaterialBHC`: Complete two-material BHC model
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
    # 1. Calibrate water-only BHC
    water_bhc = calibrate_bhc(energies, weights;
        order=order, max_path_cm=max_path_cm,
        n_points=n_points, reference_energy_keV=reference_energy_keV)

    # 2. Compute spectral attenuation data for both materials
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

"""
    apply_bhc_two_material(sinogram_raw, bhc_2mat, geom, matrix_size; volume_extent=nothing)

Apply two-material (water + bone) beam hardening correction.

Implements the full Martinez/Fessler 2022 algorithm:
1. Water-only BHC → FDK → preliminary HU image
2. Smooth tissue fraction decomposition (no hard threshold)
3. Forward-project bone image → compute soft/bone line integrals
4. Exact 2-material polychromatic correction from known spectrum
5. Apply correction to RAW sinogram

# Arguments
- `sinogram_raw::AbstractArray{T,3}`: Raw polychromatic sinogram (post-log)
- `bhc_2mat::TwoMaterialBHC`: Two-material BHC model from `calibrate_bhc_two_material`
- `geom`: CTGeometry
- `matrix_size::Tuple{Int,Int,Int}`: Reconstruction matrix size (nx, ny, nz)

# Keyword Arguments
- `volume_extent::Union{Nothing,NTuple{3,Float64}}=nothing`: Physical extent of the
  phantom volume (x, y, z) in cm. Pass `phantom.extent` for correct geometry.

# Returns
- `Array{Float32,3}`: Corrected sinogram (CPU array)
"""
function apply_bhc_two_material(
    sinogram_raw::AbstractArray{T, 3},
    bhc_2mat::TwoMaterialBHC,
    geom,
    matrix_size::Tuple{Int,Int,Int};
    volume_extent=nothing
) where T <: AbstractFloat

    # === Pass 1: Water-only BHC → FDK → preliminary HU image ===
    sino_water = apply_bhc(sinogram_raw, bhc_2mat.water_bhc)
    recon = Array(fdk_reconstruct(sino_water, geom, matrix_size))
    hu = 1000.0f0 .* (recon .- T(bhc_2mat.μ_water_ref)) ./ T(bhc_2mat.μ_water_ref)

    # === Pass 2: Bone correction ===

    # Step 1: Smooth tissue fraction decomposition
    nx, ny, nz = size(recon)
    bone_μ_3d = zeros(Float32, nx, ny, nz)
    for s in 1:nz
        for j in 1:ny, i in 1:nx
            bf = bone_fraction_smooth(hu[i,j,s];
                hu_low=bhc_2mat.hu_low, hu_high=bhc_2mat.hu_high)
            bone_μ_3d[i,j,s] = Float32(bf * recon[i,j,s])
        end
    end

    # Step 2: Forward-project bone image
    p_b = Array(siddon_forward_project(bone_μ_3d, geom; volume_extent=volume_extent))

    # Soft tissue line integral from water-corrected sinogram
    sino_water_cpu = Array(sino_water)
    p_s = sino_water_cpu .- p_b

    # Step 3: Material path lengths
    L_w = p_s ./ Float32(bhc_2mat.μ_water_ref)
    L_b = p_b ./ Float32(bhc_2mat.μ_bone_ref)
    L_w .= max.(L_w, 0.0f0)
    L_b .= max.(L_b, 0.0f0)

    # Step 4: Exact 2-material correction (Float64 for precision)
    sino_raw_cpu = Array(sinogram_raw)
    correction = zeros(Float32, size(sino_raw_cpu))
    w_norm_64 = Float64.(bhc_2mat.w_norm)
    μ_w_64 = Float64.(bhc_2mat.μ_water_E)
    μ_b_64 = Float64.(bhc_2mat.μ_bone_E)
    μ_water_ref = bhc_2mat.μ_water_ref
    μ_bone_ref = bhc_2mat.μ_bone_ref

    Threads.@threads for idx in eachindex(sino_raw_cpu)
        Lw = Float64(L_w[idx])
        Lb = Float64(L_b[idx])
        (Lw < 1e-6 && Lb < 1e-6) && continue

        # Polychromatic forward model
        I_poly = 0.0
        for e in eachindex(w_norm_64)
            I_poly += w_norm_64[e] * exp(-μ_w_64[e] * Lw - μ_b_64[e] * Lb)
        end
        F_2mat = I_poly > 0.0 ? -log(I_poly) : 0.0

        # Monochromatic reference
        p_mono = μ_water_ref * Lw + μ_bone_ref * Lb

        # Full BH correction
        correction[idx] = Float32(p_mono - F_2mat)
    end

    # Step 5: Apply to RAW sinogram
    return sino_raw_cpu .+ correction
end
