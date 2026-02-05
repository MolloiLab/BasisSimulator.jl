# =============================================================================
# Unified Detector Response Matrix (DRM) for PCCT Detectors
# =============================================================================
#
# Combines all energy-dependent detector physics into a single pre-computed
# matrix D[E_detected | E_true] that maps true photon energy to detected
# energy distribution per pixel:
#
#   1. Charge cloud transport → energy-dependent resolution (Koch-Mehrin 2020)
#   2. Fano noise → intrinsic energy resolution (F=0.1)
#   3. Electronic noise → Gaussian broadening (typically 1-2 keV RMS)
#   4. K-fluorescence escape → escape peaks and spectral cross-talk
#   5. Charge collection efficiency → low-energy tailing (Hecht relation)
#
# Spatial effects (charge sharing between pixels) and flux-dependent effects
# (pileup spectral migration) are NOT part of the DRM — they are applied
# separately in the forward projection pipeline.
#
# The DRM is computed ONCE per detector configuration and reused for all
# projections (cached as a matrix on CPU, applied via GPU kernels).
#
# References:
# - Koch-Mehrin 2020, NIM-A 976:164241 (charge cloud, fluorescence, CCE)
# - Konrad 2025, PMB 70:065004 (NAEOTOM Alpha PCCT model)
# - Stierstorfer 2018, Med Phys 45:156-166 (DQE from event probabilities)
# =============================================================================

"""
    physics_energy_resolution_keV(E_keV::Real;
                                   fano_factor::Float64=0.1,
                                   pair_creation_eV::Float64=4.3,
                                   electronic_noise_keV::Float64=1.5) -> Float64

Compute the physics-based energy resolution (σ) at a given photon energy.

The total energy resolution combines two independent noise sources:
1. **Fano noise**: σ_Fano = √(F × ω × E)  [keV]
   where F = Fano factor, ω = pair creation energy [eV], E = photon energy [keV]
2. **Electronic noise**: σ_elec = constant [keV RMS]

Combined: σ_total = √(σ_Fano² + σ_elec²)

Note: This gives the ENERGY resolution (σ in keV), NOT the charge cloud
spatial resolution. The spatial charge cloud affects charge sharing between
pixels, not the energy resolution within a single pixel.

For CdTe at 60 keV:
- σ_Fano = √(0.1 × 4.3e-3 × 60) = √0.0258 = 0.161 keV
- σ_elec = 1.5 keV (NAEOTOM: electronic noise ~1.5 keV RMS)
- σ_total = √(0.026 + 2.25) = 1.51 keV
- FWHM = 2.355 × 1.51 = 3.55 keV

This is much better resolution than the ad-hoc 10 keV FWHM previously used.

# Arguments
- `E_keV`: Photon energy [keV]
- `fano_factor`: Fano factor (0.1 for CdTe, Redus 2009)
- `pair_creation_eV`: Pair creation energy [eV] (4.3 for CdTe, Koch-Mehrin 2020)
- `electronic_noise_keV`: Electronic noise RMS [keV] (1.5 for NAEOTOM, Konrad 2025)

# Returns
- σ_total in keV (Gaussian standard deviation)

# References
- Knoll 2010, "Radiation Detection and Measurement", Ch. 11 (Fano noise)
- Konrad 2025 (PMB 70:065004), Section 2.3 (electronic noise 0.6 keV RMS)
- Koch-Mehrin 2020, Section 2.3 (pair creation energy ω = 4.3 eV)
"""
function physics_energy_resolution_keV(E_keV::Real;
                                        fano_factor::Float64=0.1,
                                        pair_creation_eV::Float64=4.3,
                                        electronic_noise_keV::Float64=1.5)
    E = max(Float64(E_keV), 0.0)

    # Fano noise: σ² = F × ω × E  (all in keV)
    # ω in keV: 4.3 eV = 4.3e-3 keV
    ω_keV = pair_creation_eV * 1e-3
    σ_fano_sq = fano_factor * ω_keV * E

    # Electronic noise: constant σ
    σ_elec_sq = electronic_noise_keV^2

    # Combined (independent sources add in quadrature)
    return sqrt(σ_fano_sq + σ_elec_sq)
end

"""
    compute_unified_drm(detector::PhotonCountingDetector, kVp::Real;
                         n_energy_points::Int=200,
                         use_physics_resolution::Bool=true) -> Matrix{Float64}

Compute the unified Detector Response Matrix (DRM) combining all
energy-dependent physics effects for a PCCT detector.

D[i, b] = probability that a photon of energy E_i registers in bin b.

## Physics included:
1. **Energy resolution**: Physics-based (Fano + electronic noise) or fixed FWHM
2. **K-fluorescence escape**: Full Koch-Mehrin 2020 Table 1 model for CdTe
3. **Charge collection efficiency**: Hecht relation with small-pixel weighting
4. **Hole tailing**: Depth-dependent CCE creates low-energy tail

## Physics NOT included (applied separately):
- Charge sharing (spatial effect, depends on neighbor pixels)
- Pileup spectral migration (flux-dependent, varies per pixel)

## DRM Properties
- Size: [n_energy_points × n_bins]
- Each row sums to ≤ 1.0 (photons below threshold are undetected)
- Pre-computed once per detector configuration
- Compatible with `compute_spectral_response_matrix` interface

# Arguments
- `detector`: PhotonCountingDetector with all configuration
- `kVp`: Maximum tube voltage [keV]
- `n_energy_points`: Energy grid resolution (default: 200)
- `use_physics_resolution`: If true, use Fano + electronic noise resolution;
  if false, use detector.energy_resolution_keV (default: true)

# Returns
- `Matrix{Float64}`: [n_energy_points × n_bins] DRM

# Example
```julia
det = naeotom_detector_standard()
D = compute_unified_drm(det, 120.0)
# D[100, 3] → P(60 keV photon registered in bin 3)
```
"""
function compute_unified_drm(detector::PhotonCountingDetector, kVp::Real;
                              n_energy_points::Int=200,
                              use_physics_resolution::Bool=true)
    material = detector.material
    thickness_mm = detector.thickness_mm
    thresholds = detector.energy_thresholds_keV
    pixel_size_mm = detector.pixel_size_mm
    n_bins = length(thresholds)

    props = get_detector_material_properties(material)

    # Energy grid: from 1 keV to kVp
    E_max = Float64(kVp)
    energies = range(1.0, E_max, length=n_energy_points)

    # Fluorescence model — extended Koch-Mehrin 2020 for CdTe
    fluor_model = if material == CDTE_MATERIAL
        compute_cdte_fluorescence_model(pixel_size_mm, thickness_mm)
    else
        nothing
    end
    fluor_legacy = if fluor_model === nothing
        compute_fluorescence_escape_probability(material, thickness_mm, pixel_size_mm)
    else
        nothing
    end

    # Charge transport parameters for tailing
    transport_params = get_charge_transport_params(material, thickness_mm)

    # Build DRM
    D = zeros(Float64, n_energy_points, n_bins)

    for (i, E) in enumerate(energies)
        E_f = Float64(E)
        T_values = Float64.(thresholds)

        # Physics-based or fixed energy resolution
        σ_E = if use_physics_resolution
            physics_energy_resolution_keV(E_f;
                fano_factor=props.fano_factor,
                pair_creation_eV=props.pair_creation_energy_eV,
                electronic_noise_keV=Float64(detector.electronic_noise_keV))
        else
            if detector.energy_resolution_keV > 0.0
                detector.energy_resolution_keV / (2.0 * sqrt(2.0 * log(2.0)))
            else
                props.energy_resolution_fwhm_keV / (2.0 * sqrt(2.0 * log(2.0)))
            end
        end

        # Step 1: Hole tailing — distribute energy across CCE values
        # Uses Hecht relation with small-pixel weighting potential
        tail_energies, tail_weights = hole_tailing_distribution(
            E_f, material, thickness_mm;
            n_depth=20, pixel_pitch_mm=pixel_size_mm
        )

        for (t_idx, E_tail) in enumerate(tail_energies)
            w_tail = tail_weights[t_idx]

            # Step 2: K-fluorescence escape
            if fluor_model !== nothing
                p_escape, E_primary, E_neighbor = apply_fluorescence_escape_extended(
                    E_tail, fluor_model
                )
                energy_components = [
                    (E_tail, 1.0 - p_escape),         # no escape: photopeak
                    (E_primary, p_escape * 0.5),       # escape: primary gets reduced energy
                    (E_neighbor, p_escape * 0.5),      # escape: neighbor gets fluorescence
                ]
            elseif fluor_legacy !== nothing
                p_escape, E_escaped = apply_fluorescence_escape(E_tail, fluor_legacy)
                energy_components = [(E_tail, 1.0 - p_escape), (E_escaped, p_escape)]
            else
                energy_components = [(E_tail, 1.0)]
            end

            # Step 3: Gaussian energy blur and threshold binning
            for (E_comp, w_comp) in energy_components
                for b in 1:n_bins
                    T_low = T_values[b]

                    if σ_E > 0.0
                        z_low = (T_low - E_comp) / (σ_E * sqrt(2.0))
                        if b == n_bins
                            prob = 0.5 * (1.0 - _erf_approx(z_low))
                        else
                            T_high = T_values[b + 1]
                            z_high = (T_high - E_comp) / (σ_E * sqrt(2.0))
                            prob = 0.5 * (_erf_approx(z_high) - _erf_approx(z_low))
                        end
                    else
                        if b == n_bins
                            prob = E_comp >= T_low ? 1.0 : 0.0
                        else
                            T_high = T_values[b + 1]
                            prob = (E_comp >= T_low && E_comp < T_high) ? 1.0 : 0.0
                        end
                    end

                    D[i, b] += w_tail * w_comp * max(prob, 0.0)
                end
            end
        end
    end

    # Ensure physical constraint: each row sums to ≤ 1.0
    for i in 1:n_energy_points
        row_sum = sum(D[i, :])
        if row_sum > 1.0
            D[i, :] ./= row_sum
        end
    end

    return D
end

"""
    drm_energy_grid(kVp::Real; n_energy_points::Int=200) -> Vector{Float64}

Get the energy grid corresponding to the unified DRM.

Returns energy values [keV] for each row of the matrix returned by
`compute_unified_drm`.
"""
function drm_energy_grid(kVp::Real; n_energy_points::Int=200)
    return collect(range(1.0, Float64(kVp), length=n_energy_points))
end

"""
    drm_summary(D::Matrix{Float64}, thresholds::AbstractVector, kVp::Real;
                n_energy_points::Int=200)

Print a summary of the DRM properties for diagnostic purposes.
"""
function drm_summary(D::Matrix{Float64}, thresholds::AbstractVector, kVp::Real;
                     n_energy_points::Int=200)
    n_E, n_bins = size(D)
    energies = drm_energy_grid(kVp; n_energy_points=n_E)

    println("DRM Summary: $(n_E) energies × $(n_bins) bins")
    println("Energy range: $(energies[1])–$(energies[end]) keV")
    println("Thresholds: $(Float64.(thresholds)) keV")
    println()

    # Check key energies
    for E_test in [30.0, 45.0, 60.0, 80.0, 100.0, 120.0]
        if E_test > kVp
            continue
        end
        idx = clamp(round(Int, (E_test - 1.0) / (Float64(kVp) - 1.0) * (n_E - 1)) + 1, 1, n_E)
        row = D[idx, :]
        println("E=$(Int(E_test)) keV: bins=$(round.(row, digits=3)), sum=$(round(sum(row), digits=3))")
    end

    # Overall statistics
    row_sums = [sum(D[i, :]) for i in 1:n_E]
    println("\nRow sum stats: min=$(round(minimum(row_sums), digits=3)), " *
            "max=$(round(maximum(row_sums), digits=3)), " *
            "mean=$(round(sum(row_sums)/n_E, digits=3))")

    # Photopeak efficiency (fraction in correct bin)
    println("\nPhotopeak efficiency (fraction in 'correct' bin):")
    for b in 1:n_bins
        T_low = Float64(thresholds[b])
        T_high = b < n_bins ? Float64(thresholds[b+1]) : Float64(kVp)
        E_center = (T_low + T_high) / 2.0
        idx = clamp(round(Int, (E_center - 1.0) / (Float64(kVp) - 1.0) * (n_E - 1)) + 1, 1, n_E)
        println("  Bin $b ($(Int(T_low))–$(Int(T_high)) keV, center=$(round(E_center, digits=1))): " *
                "D=$(round(D[idx, b], digits=3))")
    end
end

# =============================================================================
# Exports
# =============================================================================

export physics_energy_resolution_keV, compute_unified_drm
export drm_energy_grid, drm_summary
