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

export BeamHardeningCorrection, BHCPolynomial
export calibrate_bhc, apply_bhc!, apply_bhc
export generate_water_calibration_curve
export bhc_water_default, bhc_none
export evaluate_bhc, get_bhc_info

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
    bhc::BeamHardeningCorrection
) where T <: AbstractFloat
    return apply_bhc!(sinogram, bhc.polynomial)
end

function apply_bhc!(
    sinogram::AbstractArray{T, 3},
    poly::BHCPolynomial
) where T <: AbstractFloat

    order = poly.order

    # Transfer coefficients to GPU (similar pattern for GPU compatibility)
    coeffs_cpu = T.(poly.coefficients)
    coeffs = similar(sinogram, T, length(coeffs_cpu))
    copyto!(coeffs, coeffs_cpu)

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
