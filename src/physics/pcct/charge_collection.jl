# =============================================================================
# Charge Collection Efficiency — Improved Hecht Model with Small-Pixel Effect
# =============================================================================
#
# Implements the Hecht relation (Koch-Mehrin 2020, Eq. 15) with small-pixel
# weighting potential ψ(z) for pixelated CdTe PCCT detectors.
#
# The standard Hecht equation assumes a parallel-plate (planar) geometry where
# the weighting potential is linear: ψ(z) = z/L. For small pixels (w/L < 0.5),
# the weighting potential deviates from linear, concentrating near the collecting
# electrode (anode). This is the "small-pixel effect" (Barrett et al. 1995).
#
# Key insight: For NAEOTOM (w/L ≈ 0.17), the small-pixel effect REDUCES the
# impact of hole trapping because holes contribute less to the signal via the
# weighting potential when they are far from the anode.
#
# References:
# - Koch-Mehrin 2020, Eq. 15 (Hecht relation for pixelated detectors)
# - Barrett et al. 1995 (Charge transport in coplanar detectors, NIM-A 380:131)
# - Eskin et al. 1999 (Signals induced in semiconductor gamma-ray detectors)
# =============================================================================

"""
    small_pixel_weighting_potential(z_norm::Real, wL_ratio::Real) -> Float64

Compute the weighting potential ψ at normalized depth z/L for a pixel with
width-to-thickness ratio w/L.

For a square pixel of width w on a slab of thickness L, the weighting potential
can be approximated analytically. We use the leading-order Fourier approximation
from Barrett et al. 1995:

    ψ(z) ≈ [sinh(π·w·z/(2L²)) / sinh(π·w/(2L))]

For w/L → ∞ (planar limit): ψ → z/L (linear)
For w/L → 0 (point pixel): ψ concentrated near anode

# Arguments
- `z_norm`: Normalized depth from cathode (0 = cathode, 1 = anode)
- `wL_ratio`: Pixel width to thickness ratio (w/L)

# Returns
- Weighting potential value in [0, 1]

# References
- Barrett et al. 1995, NIM-A 380:131-154
- Eskin et al. 1999, J. Appl. Phys. 85:647
"""
function small_pixel_weighting_potential(z_norm::Real, wL_ratio::Real)
    z = clamp(Float64(z_norm), 0.0, 1.0)
    r = Float64(wL_ratio)

    if r >= 5.0
        # Planar limit: linear weighting potential
        return z
    end

    if r <= 0.01
        # Extreme small-pixel: essentially all weight at anode
        # ψ(z) ≈ 0 for z < 0.9, then rises steeply
        return z < 0.99 ? 0.0 : z
    end

    # Fourier series approximation (Barrett et al. 1995)
    # For a square pixel on a planar slab:
    # ψ(z) = Σ_n A_n × sinh(κ_n × z) / sinh(κ_n × L)
    # where κ_n = nπ/w for the dominant mode
    #
    # Leading order: κ₁ = π/w, so κ₁L = πL/w = π/r
    κL = π / r

    # sinh can overflow for large κL, so handle carefully
    if κL > 50.0
        # For very small pixels: ψ(z) ≈ exp(κL×(z-1)) near anode
        return exp(κL * (z - 1.0))
    end

    numerator = sinh(κL * z)
    denominator = sinh(κL)

    if denominator < 1e-300
        return z  # fallback to linear
    end

    # The Fourier mode gives a non-linear ψ, but the actual weighting potential
    # also includes higher-order corrections. For practical purposes, we blend
    # with the linear potential weighted by the small-pixel ratio:
    # At w/L = 1: roughly 50% small-pixel effect
    # At w/L = 0.17 (NAEOTOM): strong small-pixel effect
    ψ_smallpix = numerator / denominator
    ψ_linear = z

    # The leading Fourier mode overestimates the small-pixel effect.
    # Use a blending factor based on pixel geometry.
    # For w/L < 0.3: small-pixel effect dominates
    # For w/L > 2.0: linear dominates
    blend = clamp((r - 0.1) / 1.9, 0.0, 1.0)

    return ψ_linear * blend + ψ_smallpix * (1.0 - blend)
end

"""
    hecht_cce_weighted(z_cm::Real, L_cm::Real, V::Real,
                       μeτe::Real, μhτh::Real, wL_ratio::Real) -> Float64

Compute charge collection efficiency using the Hecht equation with
small-pixel weighting potential (Koch-Mehrin 2020, Eq. 15).

    CCE(z) = (λ_e/L) × ψ(L-z) × [1 - exp(-(L-z)/λ_e)]
           + (λ_h/L) × ψ(z) × [1 - exp(-z/λ_h)]

where ψ(z) is the small-pixel weighting potential.

# Arguments
- `z_cm`: Interaction depth from cathode [cm] (0 = cathode, L = anode)
- `L_cm`: Sensor thickness [cm]
- `V`: Bias voltage [V]
- `μeτe`: Electron mobility-lifetime product [cm²/V]
- `μhτh`: Hole mobility-lifetime product [cm²/V]
- `wL_ratio`: Pixel width-to-thickness ratio (for weighting potential)

# Returns
- CCE in [0, 1]

# References
- Koch-Mehrin 2020, Eq. 15
"""
function hecht_cce_weighted(z_cm::Real, L_cm::Real, V::Real,
                            μeτe::Real, μhτh::Real, wL_ratio::Real)
    L = Float64(L_cm)
    z = clamp(Float64(z_cm), 0.0, L)

    E_field = Float64(V) / L  # V/cm

    # Mean drift lengths
    λ_e = Float64(μeτe) * E_field  # cm
    λ_h = Float64(μhτh) * E_field  # cm

    z_norm = z / L  # Normalized depth (0 = cathode, 1 = anode)
    r = Float64(wL_ratio)

    # Koch-Mehrin 2020, Eq. 15 with small-pixel weighting potential ψ(z)
    #
    # The induced charge from carrier drift depends on the weighting potential
    # change Δψ along the drift path. For standard Hecht (planar, ψ=z/L), this
    # gives the familiar Hecht formula. For small pixels, the weighting potential
    # is non-linear, suppressing the hole contribution.
    #
    # Electron contribution: electrons drift from z toward anode (z=L)
    #   Δψ_e = ψ(1) - ψ(z_norm) = 1 - ψ(z_norm)
    #   With trapping: Q_e = Δψ_e × (effective collection fraction)
    # Hole contribution: holes drift from z toward cathode (z=0)
    #   Δψ_h = ψ(z_norm) - ψ(0) = ψ(z_norm)
    #   With trapping: Q_h = Δψ_h × (effective collection fraction)
    #
    # The Hecht formula factor (λ/L)(1-exp(-d/λ)) gives the fraction of
    # carriers that survive trapping. Combined with the weighting potential,
    # the signal is scaled by how much ψ changes during the surviving drift.

    ψ_z = small_pixel_weighting_potential(z_norm, r)

    # For planar (linear ψ): Δψ_e = 1 - z/L, Δψ_h = z/L
    # Standard Hecht: CCE_e = (λ_e/L)(1-exp(-(L-z)/λ_e)), CCE_h = (λ_h/L)(1-exp(-z/λ_h))
    # The weighting potential replaces the z/L factor:
    # CCE_e_planar × Δψ_e / (1-z/L) and CCE_h_planar × Δψ_h / (z/L)
    # But this has singularities at z=0 and z=L.
    #
    # More robust: compute the standard Hecht CCE and then scale the
    # hole contribution by the small-pixel suppression factor.

    # Standard Hecht (planar) components
    cce_e_planar = if λ_e > 1e-10
        (λ_e / L) * (1.0 - exp(-(L - z) / λ_e))
    else
        0.0
    end

    cce_h_planar = if λ_h > 1e-10
        (λ_h / L) * (1.0 - exp(-z / λ_h))
    else
        0.0
    end

    # Small-pixel correction: scale hole and electron contributions by
    # weighting potential factors
    # For electrons: signal ∝ 1-ψ(z) (unchanged for planar: 1-z/L → 1-z/L)
    # For holes: signal ∝ ψ(z) (unchanged for planar: z/L → z/L)
    # Normalization: at each depth, the planar factors sum to 1
    if z_norm > 1e-6 && z_norm < (1.0 - 1e-6)
        # Scale electron contribution: (1-ψ(z))/(1-z/L)
        # Scale hole contribution: ψ(z)/(z/L)
        e_scale = (1.0 - ψ_z) / (1.0 - z_norm)
        h_scale = ψ_z / z_norm
        cce = cce_e_planar * e_scale + cce_h_planar * h_scale
    elseif z_norm <= 1e-6
        # At cathode: only electrons contribute, ψ(0) = 0
        cce = cce_e_planar  # electrons see Δψ ≈ 1.0, same as planar
    else
        # At anode: only holes contribute, ψ(1) = 1.0
        cce = cce_h_planar  # holes see Δψ ≈ 1.0, same as planar
    end

    return clamp(cce, 0.0, 1.0)
end

"""
    mean_cce_beer_lambert(E_keV::Real, L_mm::Real, V::Real,
                          μeτe::Real, μhτh::Real, wL_ratio::Real;
                          μ_linear_per_mm::Real=0.0, n_depth::Int=50) -> Float64

Compute depth-averaged CCE weighted by Beer-Lambert absorption profile.

Instead of assuming uniform absorption (which overweights deep events),
weight each depth by the probability of absorption there:

    P(z) = μ × exp(-μ × z) / (1 - exp(-μ × L))

where μ is the linear attenuation coefficient of CdTe at energy E.

# Arguments
- `E_keV`: Photon energy [keV]
- `L_mm`: Sensor thickness [mm]
- `V`: Bias voltage [V]
- `μeτe`: Electron mobility-lifetime product [cm²/V]
- `μhτh`: Hole mobility-lifetime product [cm²/V]
- `wL_ratio`: Pixel width-to-thickness ratio
- `μ_linear_per_mm`: Linear attenuation coefficient [1/mm]. If 0, uses ~CdTe estimate.
- `n_depth`: Number of integration points

# Returns
- Depth-averaged CCE in [0, 1]
"""
function mean_cce_beer_lambert(E_keV::Real, L_mm::Real, V::Real,
                               μeτe::Real, μhτh::Real, wL_ratio::Real;
                               μ_linear_per_mm::Real=0.0, n_depth::Int=50)
    L_cm = Float64(L_mm) / 10.0
    L = Float64(L_mm)

    # Estimate CdTe linear attenuation if not provided
    μ = if Float64(μ_linear_per_mm) > 0.0
        Float64(μ_linear_per_mm)
    else
        # Rough CdTe μ estimate: from NIST data, CdTe density 5.85 g/cm³
        # At 60 keV: μ/ρ ≈ 1.5 cm²/g → μ ≈ 8.8 cm⁻¹ = 0.88 mm⁻¹
        # At 30 keV: μ/ρ ≈ 10 cm²/g → μ ≈ 58.5 cm⁻¹ = 5.85 mm⁻¹
        # Simple fit: μ ≈ A × E^(-2.5) for photoelectric-dominated regime
        E = max(Float64(E_keV), 10.0)
        # Calibrated to: μ(60 keV) ≈ 0.88 mm⁻¹, μ(30 keV) ≈ 5.85 mm⁻¹
        0.88 * (60.0 / E)^2.5
    end

    # Beer-Lambert absorption probability at depth z:
    # P(z) ∝ μ × exp(-μ × z), normalized over [0, L]
    # Normalization: ∫₀ᴸ μ exp(-μz) dz = 1 - exp(-μL)
    norm = 1.0 - exp(-μ * L)
    if norm < 1e-10
        norm = μ * L  # thin absorber limit
    end

    cce_weighted = 0.0
    weight_sum = 0.0

    dz = L / n_depth
    for i in 1:n_depth
        z_mm = (i - 0.5) * dz  # midpoint of each depth slice
        z_cm = z_mm / 10.0

        # Absorption weight (Beer-Lambert)
        w = μ * exp(-μ * z_mm) * dz / norm

        # CCE at this depth (with small-pixel weighting potential)
        cce = hecht_cce_weighted(z_cm, L_cm, Float64(V), Float64(μeτe), Float64(μhτh), Float64(wL_ratio))

        cce_weighted += w * cce
        weight_sum += w
    end

    return weight_sum > 0 ? cce_weighted / weight_sum : 0.9
end

"""
    hole_tailing_beer_lambert(E_incident_keV::Real, E_photon_keV::Real,
                              L_mm::Real, V::Real,
                              μeτe::Real, μhτh::Real, wL_ratio::Real;
                              μ_linear_per_mm::Real=0.0, n_depth::Int=20)
        -> Tuple{Vector{Float64}, Vector{Float64}}

Compute energy distribution from incomplete charge collection (hole tailing),
weighted by Beer-Lambert absorption profile.

Returns (energies, weights) where:
- energies[i] = E_incident × CCE(z_i)
- weights[i] = Beer-Lambert probability of absorption at depth z_i

# Arguments
- `E_incident_keV`: True photon energy [keV]
- `E_photon_keV`: Energy for Beer-Lambert attenuation (same as E_incident unless specified)
- Other args: same as `mean_cce_beer_lambert`
"""
function hole_tailing_beer_lambert(E_incident_keV::Real, E_photon_keV::Real,
                                   L_mm::Real, V::Real,
                                   μeτe::Real, μhτh::Real, wL_ratio::Real;
                                   μ_linear_per_mm::Real=0.0, n_depth::Int=20)
    L_cm = Float64(L_mm) / 10.0
    L = Float64(L_mm)
    E = Float64(E_incident_keV)

    # Attenuation coefficient
    μ = if Float64(μ_linear_per_mm) > 0.0
        Float64(μ_linear_per_mm)
    else
        Ep = max(Float64(E_photon_keV), 10.0)
        0.88 * (60.0 / Ep)^2.5
    end

    norm = 1.0 - exp(-μ * L)
    if norm < 1e-10
        norm = μ * L
    end

    energies = zeros(Float64, n_depth)
    weights = zeros(Float64, n_depth)

    dz = L / n_depth
    for i in 1:n_depth
        z_mm = (i - 0.5) * dz
        z_cm = z_mm / 10.0

        weights[i] = μ * exp(-μ * z_mm) * dz / norm
        cce = hecht_cce_weighted(z_cm, L_cm, Float64(V), Float64(μeτe), Float64(μhτh), Float64(wL_ratio))
        energies[i] = E * cce
    end

    # Normalize weights
    ws = sum(weights)
    if ws > 0
        weights ./= ws
    end

    return (energies, weights)
end

# =============================================================================
# Exports
# =============================================================================

export small_pixel_weighting_potential
export hecht_cce_weighted
export mean_cce_beer_lambert
export hole_tailing_beer_lambert
