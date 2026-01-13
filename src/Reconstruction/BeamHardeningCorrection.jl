"""
    Reconstruction/BeamHardeningCorrection.jl

Water beam hardening correction (BHC) for CT reconstruction.

Beam hardening occurs because lower-energy photons are preferentially
absorbed, causing the effective attenuation coefficient to decrease
with path length. This creates:
- Cupping artifacts in uniform water regions
- Dark bands between dense objects

Water BHC assumes the object is water-equivalent and corrects the
nonlinear relationship between polychromatic and monochromatic attenuation.

Implementation follows CatSim/XCIST approach:
- Polynomial correction based on simulated water attenuation
- Horner scheme for efficient polynomial evaluation
"""

# =============================================================================
# BHC Types
# =============================================================================

"""
    WaterBHC

Water beam hardening correction model.

# Fields
- `coefficients`: Polynomial coefficients [n_coeffs] (highest order first)
- `effective_μ`: Effective water attenuation at reference energy (cm⁻¹)
- `max_path_length`: Maximum calibrated path length (cm)
- `poly_order`: Polynomial order
"""
struct WaterBHC
    coefficients::Vector{Float64}
    effective_μ::Float64
    max_path_length::Float64
    poly_order::Int
end

# =============================================================================
# BHC Calibration
# =============================================================================

"""
    calibrate_water_bhc(spectrum_weights, spectrum_energies, μ_water_vec;
                        max_path_cm=50.0, n_samples=100, poly_order=5) -> WaterBHC

Calibrate water beam hardening correction polynomial.

Simulates polychromatic attenuation through water at various path lengths
and fits a polynomial to convert polychromatic to monochromatic-equivalent.

# Arguments
- `spectrum_weights`: Normalized spectrum weights [n_energies]
- `spectrum_energies`: Energy values in keV [n_energies]
- `μ_water_vec`: Water attenuation at each energy [n_energies] (cm⁻¹)
- `max_path_cm`: Maximum water path length to calibrate (default: 50 cm)
- `n_samples`: Number of path length samples (default: 100)
- `poly_order`: Polynomial order for fitting (default: 5)

# Returns
WaterBHC model for correction.
"""
function calibrate_water_bhc(
    spectrum_weights::Vector{Float64},
    spectrum_energies::Vector{Float64},
    μ_water_vec::Vector{Float64};
    max_path_cm::Float64=50.0,
    n_samples::Int=100,
    poly_order::Int=5
)
    # Effective monochromatic μ (weighted average)
    effective_μ = sum(spectrum_weights .* μ_water_vec)

    # Sample path lengths
    path_lengths = range(0.0, max_path_cm, length=n_samples+1)[2:end]  # Skip zero

    # Compute polychromatic and monochromatic attenuation at each path length
    p_poly = zeros(n_samples)   # Polychromatic projection values
    p_mono = zeros(n_samples)   # Monochromatic projection values

    for (i, L) in enumerate(path_lengths)
        # Polychromatic: -log(Σ wᵢ × exp(-μᵢ × L))
        transmitted = sum(spectrum_weights .* exp.(-μ_water_vec .* L))
        p_poly[i] = -log(max(transmitted, 1e-20))

        # Monochromatic: μ_eff × L
        p_mono[i] = effective_μ * L
    end

    # Fit polynomial: p_mono = f(p_poly)
    # We want p_corrected = c₀ + c₁×p + c₂×p² + ...
    coeffs = fit_polynomial(p_poly, p_mono, poly_order)

    return WaterBHC(coeffs, effective_μ, max_path_cm, poly_order)
end

"""
    calibrate_water_bhc_simple(kVp::Int; poly_order::Int=5) -> WaterBHC

Simplified BHC calibration using typical spectrum characteristics.

# Arguments
- `kVp`: Tube voltage (80, 100, 120, 140)
- `poly_order`: Polynomial order (default: 5)

# Returns
WaterBHC model.
"""
function calibrate_water_bhc_simple(kVp::Int; poly_order::Int=5)
    # Generate simple spectrum approximation
    n_bins = 50
    E_max = Float64(kVp)
    E_min = 20.0  # Filtered below this

    energies = range(E_min, E_max, length=n_bins)

    # Simplified Kramers spectrum with filtration
    weights = zeros(n_bins)
    for (i, E) in enumerate(energies)
        # Kramers continuum approximation
        brem = (E_max - E) / E
        # Approximate Al filtration effect
        μ_al = 0.5 * (60.0 / E)^2.5  # Simplified Al attenuation
        filtration = exp(-0.25 * μ_al)  # 2.5mm Al equivalent
        weights[i] = brem * filtration * E  # Energy weighting for detection
    end
    weights ./= sum(weights)

    # Water attenuation at each energy (NIST XCOM fit)
    # Fit to NIST data: μ = 2814/E^2.8 + 0.154 (cm⁻¹, E in keV)
    # Matches water μ: 0.81 @ 20keV, 0.21 @ 60keV, 0.16 @ 120keV
    μ_water = [2814.0 / E^2.8 + 0.154 for E in energies]

    return calibrate_water_bhc(weights, collect(energies), μ_water;
                               poly_order=poly_order)
end

"""
    fit_polynomial(x, y, order)

Fit polynomial y = p(x) using least squares.

Returns coefficients in descending order [cₙ, cₙ₋₁, ..., c₁, c₀].
"""
function fit_polynomial(x::Vector{Float64}, y::Vector{Float64}, order::Int)
    n = length(x)
    @assert length(y) == n

    # Build Vandermonde matrix
    A = zeros(n, order + 1)
    for i in 1:n
        for j in 0:order
            A[i, j+1] = x[i]^j
        end
    end

    # Solve least squares: A × c = y
    coeffs_ascending = A \ y

    # Reverse to descending order (highest power first)
    return reverse(coeffs_ascending)
end

# =============================================================================
# BHC Application
# =============================================================================

"""
    apply_water_bhc(sinogram, bhc::WaterBHC) -> Array

Apply water beam hardening correction to sinogram.

Uses Horner's scheme for efficient polynomial evaluation.

# Arguments
- `sinogram`: Projection data [n_cols, n_rows, n_angles]
- `bhc::WaterBHC`: Calibrated BHC model

# Returns
Corrected sinogram.
"""
function apply_water_bhc(sinogram::AbstractArray{T,3}, bhc::WaterBHC) where T
    result = similar(sinogram)
    coeffs = T.(bhc.coefficients)

    # Horner's scheme: p(x) = c₀ + x(c₁ + x(c₂ + ...))
    # Our coeffs are [cₙ, cₙ₋₁, ..., c₁, c₀] (descending)
    for idx in eachindex(sinogram)
        x = sinogram[idx]

        # Clamp to calibrated range
        x = clamp(x, T(0), T(bhc.effective_μ * bhc.max_path_length * 1.5))

        # Horner evaluation
        y = coeffs[1]
        for i in 2:length(coeffs)
            y = y * x + coeffs[i]
        end

        result[idx] = y
    end

    return result
end

"""
    apply_water_bhc!(sinogram, bhc::WaterBHC)

In-place water beam hardening correction.
"""
function apply_water_bhc!(sinogram::AbstractArray{T,3}, bhc::WaterBHC) where T
    coeffs = T.(bhc.coefficients)

    for idx in eachindex(sinogram)
        x = sinogram[idx]
        x = clamp(x, T(0), T(bhc.effective_μ * bhc.max_path_length * 1.5))

        y = coeffs[1]
        for i in 2:length(coeffs)
            y = y * x + coeffs[i]
        end

        sinogram[idx] = y
    end
end

# =============================================================================
# Pre-computed BHC Models
# =============================================================================

"""
    water_bhc_120kVp()

Pre-computed water BHC for 120 kVp spectrum.

Typical clinical setting.
"""
function water_bhc_120kVp()
    return calibrate_water_bhc_simple(120)
end

"""
    water_bhc_100kVp()

Pre-computed water BHC for 100 kVp spectrum.

Lower dose pediatric/contrast protocols.
"""
function water_bhc_100kVp()
    return calibrate_water_bhc_simple(100)
end

"""
    water_bhc_80kVp()

Pre-computed water BHC for 80 kVp spectrum.

Very low dose / high contrast protocols.
"""
function water_bhc_80kVp()
    return calibrate_water_bhc_simple(80)
end

"""
    water_bhc_140kVp()

Pre-computed water BHC for 140 kVp spectrum.

Large patient / obese protocols.
"""
function water_bhc_140kVp()
    return calibrate_water_bhc_simple(140)
end

# =============================================================================
# BHC Information
# =============================================================================

"""
    get_bhc_info(bhc::WaterBHC) -> NamedTuple

Get diagnostic information about BHC model.
"""
function get_bhc_info(bhc::WaterBHC)
    return (
        poly_order = bhc.poly_order,
        n_coefficients = length(bhc.coefficients),
        effective_μ_cm = bhc.effective_μ,
        max_path_cm = bhc.max_path_length,
        coefficients = bhc.coefficients
    )
end

"""
    evaluate_bhc_correction(bhc::WaterBHC, path_lengths::Vector{Float64}) -> Tuple

Evaluate BHC correction curve.

Returns (uncorrected_values, corrected_values) for plotting/analysis.
"""
function evaluate_bhc_correction(bhc::WaterBHC, path_lengths::Vector{Float64})
    # Uncorrected = effective_μ × L (what we'd get with monochromatic)
    uncorrected = bhc.effective_μ .* path_lengths

    # Apply correction polynomial
    corrected = similar(uncorrected)
    for (i, p) in enumerate(uncorrected)
        y = bhc.coefficients[1]
        for j in 2:length(bhc.coefficients)
            y = y * p + bhc.coefficients[j]
        end
        corrected[i] = y
    end

    return (uncorrected, corrected)
end

"""
    compute_bhc_error(bhc::WaterBHC; n_points::Int=50) -> Float64

Compute maximum relative error of BHC polynomial fit.
"""
function compute_bhc_error(bhc::WaterBHC; n_points::Int=50)
    paths = range(1.0, bhc.max_path_length, length=n_points)
    uncorr, corr = evaluate_bhc_correction(bhc, collect(paths))

    # Relative error
    rel_errors = abs.(corr .- uncorr) ./ max.(abs.(uncorr), 1e-10)
    return maximum(rel_errors)
end

# =============================================================================
# Exports
# =============================================================================

export WaterBHC
export calibrate_water_bhc, calibrate_water_bhc_simple
export apply_water_bhc, apply_water_bhc!
export water_bhc_120kVp, water_bhc_100kVp, water_bhc_80kVp, water_bhc_140kVp
export get_bhc_info, evaluate_bhc_correction, compute_bhc_error
