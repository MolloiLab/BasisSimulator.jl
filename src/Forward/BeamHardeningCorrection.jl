# =============================================================================
# Beam Hardening Correction (BHC)
# =============================================================================
#
# Implements water-based beam hardening correction for polychromatic CT.
#
# Beam hardening causes:
# - Cupping artifacts in uniform water phantoms
# - Inaccurate HU values, especially for dense materials
# - Dark bands between dense objects
#
# This module provides:
# 1. Water calibration curve generation
# 2. Polynomial BHC fitting
# 3. BHC application to sinograms
#
# Reference: CatSim callback_post_log, Joseph & Spital (1978)
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

The correction maps measured line integrals to true (monochromatic-equivalent) values:
    p_corrected = a₀ + a₁×p + a₂×p² + a₃×p³ + ...

# Fields
- `coefficients`: Polynomial coefficients [a₀, a₁, a₂, ...]
- `order`: Polynomial order (length(coefficients) - 1)
- `reference_energy_keV`: Reference energy for "true" values
"""
struct BHCPolynomial
    coefficients::Vector{Float64}
    order::Int
    reference_energy_keV::Float64
end

"""
    BeamHardeningCorrection

Complete BHC model including calibration data.

# Fields
- `polynomial`: BHCPolynomial with correction coefficients
- `calibration_paths`: Water path lengths used for calibration (cm)
- `calibration_measured`: Measured line integrals at each path
- `calibration_true`: True (monochromatic) line integrals at each path
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

# Arguments
- `energies`: Energy bin centers (keV)
- `weights`: Photon fluence weights per energy bin

# Keyword Arguments
- `order`: Polynomial order (default: 3)
- `max_path_cm`: Maximum water path for calibration (default: 50.0 cm)
- `n_points`: Number of calibration points (default: 100)
- `reference_energy_keV`: Reference energy for true values (default: 70.0)

# Returns
- `BeamHardeningCorrection` with fitted polynomial

# Example
```julia
energies, weights = load_spectrum(120)
bhc = calibrate_bhc(energies, weights; order=4, reference_energy_keV=70.0)

# Apply to sinogram
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

# Arguments
- `sinogram`: Line integral sinogram [n_cols, n_rows, n_angles] (modified in place)
- `bhc`: Either `BeamHardeningCorrection` or `BHCPolynomial`

# Returns
- Modified sinogram with BHC applied

# Example
```julia
bhc = calibrate_bhc(energies, weights)
apply_bhc!(sinogram, bhc)
```
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
