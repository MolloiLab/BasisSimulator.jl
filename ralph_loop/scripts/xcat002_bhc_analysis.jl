#!/usr/bin/env julia
# XCAT-002: BHC polynomial analysis — 120 kVp vs 140 kVp
#
# Computes:
# 1. Polychromatic line integrals for water at various path lengths
# 2. Monochromatic reference at 70 keV
# 3. BHC polynomial correction applied to both spectra
# 4. Cupping error estimate

using BasisSimulator
import XrayAttenuation as XA
using Unitful

println("="^70)
println("XCAT-002: BHC Polynomial Analysis")
println("="^70)

# ─── Load spectra ───
energies_120, weights_120 = load_spectrum(120)
energies_140, weights_140 = load_spectrum(140)

# Normalize weights
w120 = weights_120 ./ sum(weights_120)
w140 = weights_140 ./ sum(weights_140)

# ─── Mean energies ───
E_mean_120 = sum(energies_120 .* w120)
E_mean_140 = sum(energies_140 .* w140)
println("\n--- Spectrum Properties ---")
println("120 kVp: $(length(energies_120)) bins, mean energy = $(round(E_mean_120, digits=1)) keV")
println("140 kVp: $(length(energies_140)) bins, mean energy = $(round(E_mean_140, digits=1)) keV")

# ─── Water attenuation at each energy ───
water = XA.Materials.water
μ_water_120 = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water, E * u"keV")) for E in energies_120]
μ_water_140 = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water, E * u"keV")) for E in energies_140]

# Reference energy for BHC
μ_ref_70 = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water, 70.0 * u"keV"))
println("μ_water(70 keV) = $(round(μ_ref_70, digits=5)) cm⁻¹")

# Effective μ_water for each spectrum
μ_eff_120 = sum(w120 .* μ_water_120)
μ_eff_140 = sum(w140 .* μ_water_140)
println("μ_water_eff(120 kVp) = $(round(μ_eff_120, digits=5)) cm⁻¹")
println("μ_water_eff(140 kVp) = $(round(μ_eff_140, digits=5)) cm⁻¹")

# ─── BHC polynomial ───
bhc_coeffs = [0.0, 1.05, -0.02, 0.001]
bhc_eval(p) = bhc_coeffs[1] + bhc_coeffs[2]*p + bhc_coeffs[3]*p^2 + bhc_coeffs[4]*p^3

println("\n--- BHC Polynomial ---")
println("p_corrected = $(bhc_coeffs[1]) + $(bhc_coeffs[2])*p + $(bhc_coeffs[3])*p² + $(bhc_coeffs[4])*p³")

# ─── Compute polychromatic line integrals at various path lengths ───
path_lengths = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]  # cm of water

println("\n" * "="^70)
println("POLYCHROMATIC LINE INTEGRALS FOR WATER")
println("="^70)
println("Path(cm) | p_mono(70keV) | p_120kVp | p_140kVp | BH_120 | BH_140 | BHC(120) | BHC(140) | Δcorr_120 | Δcorr_140 | over-corr")
println("-"^140)

for d in path_lengths
    # Monochromatic reference
    p_mono = μ_ref_70 * d

    # Polychromatic line integrals
    I_120 = sum(w120 .* exp.(-μ_water_120 .* d))
    p_120 = -log(max(I_120, 1e-30))

    I_140 = sum(w140 .* exp.(-μ_water_140 .* d))
    p_140 = -log(max(I_140, 1e-30))

    # Beam hardening magnitude (how much p_poly underestimates p_mono)
    BH_120 = p_mono - p_120  # positive = BH present
    BH_140 = p_mono - p_140

    # BHC polynomial applied
    p_bhc_120 = bhc_eval(p_120)
    p_bhc_140 = bhc_eval(p_140)

    # Correction applied
    Δ_120 = p_bhc_120 - p_120  # how much BHC adds
    Δ_140 = p_bhc_140 - p_140

    # Over-correction = (correction applied) - (correction needed)
    # Positive = over-corrected
    over_corr_120 = Δ_120 - BH_120
    over_corr_140 = Δ_140 - BH_140

    println("$(lpad(round(d, digits=0), 5)) | $(lpad(round(p_mono, digits=4), 8)) | " *
            "$(lpad(round(p_120, digits=4), 8)) | $(lpad(round(p_140, digits=4), 8)) | " *
            "$(lpad(round(BH_120, digits=4), 6)) | $(lpad(round(BH_140, digits=4), 6)) | " *
            "$(lpad(round(p_bhc_120, digits=4), 8)) | $(lpad(round(p_bhc_140, digits=4), 8)) | " *
            "$(lpad(round(Δ_120, digits=4), 8)) | $(lpad(round(Δ_140, digits=4), 8)) | " *
            "$(lpad(round(over_corr_140, digits=4), 8))")
end

# ─── Cupping estimate ───
println("\n" * "="^70)
println("CUPPING ESTIMATE")
println("="^70)
println("\nCupping = μ_center - μ_edge after BHC, relative to mono reference")
println("For a 30cm water cylinder:")

# Center path = 30 cm, edge path ≈ 5 cm
d_center = 30.0
d_edge = 5.0

for (label, w, μ_w) in [("120 kVp", w120, μ_water_120), ("140 kVp", w140, μ_water_140)]
    p_mono_center = μ_ref_70 * d_center
    p_mono_edge = μ_ref_70 * d_edge

    I_center = sum(w .* exp.(-μ_w .* d_center))
    p_center = -log(max(I_center, 1e-30))

    I_edge = sum(w .* exp.(-μ_w .* d_edge))
    p_edge = -log(max(I_edge, 1e-30))

    # After BHC
    p_bhc_center = bhc_eval(p_center)
    p_bhc_edge = bhc_eval(p_edge)

    # Error relative to mono (in units of μ, which maps to HU)
    err_center = (p_bhc_center - p_mono_center) / d_center  # μ error at center
    err_edge = (p_bhc_edge - p_mono_edge) / d_edge         # μ error at edge

    # Cupping in HU (relative to μ_water at 70 keV)
    cupping_HU = 1000.0 * (err_center - err_edge) / μ_ref_70

    println("\n$(label):")
    println("  Center ($(d_center)cm): p_poly=$(round(p_center, digits=4)), p_bhc=$(round(p_bhc_center, digits=4)), p_mono=$(round(p_mono_center, digits=4))")
    println("  Edge ($(d_edge)cm):   p_poly=$(round(p_edge, digits=4)), p_bhc=$(round(p_bhc_edge, digits=4)), p_mono=$(round(p_mono_edge, digits=4))")
    println("  μ_err center = $(round(err_center, digits=6)) cm⁻¹")
    println("  μ_err edge   = $(round(err_edge, digits=6)) cm⁻¹")
    println("  CUPPING = $(round(cupping_HU, digits=1)) HU (center - edge, positive = center brighter)")
end

# ─── Also compare: what if we use calibrate_bhc() instead? ───
println("\n" * "="^70)
println("COMPARISON: calibrate_bhc() vs bhc_water_default()")
println("="^70)

for (label, energies, weights) in [("120 kVp", energies_120, weights_120),
                                    ("140 kVp", energies_140, weights_140)]
    bhc_cal = calibrate_bhc(energies, weights; order=3, reference_energy_keV=70.0)
    coeffs = bhc_cal.polynomial.coefficients
    println("\n$(label) calibrated coefficients (order 3):")
    println("  [$(round(coeffs[1], digits=6)), $(round(coeffs[2], digits=6)), $(round(coeffs[3], digits=6)), $(round(coeffs[4], digits=6))]")
    println("  vs default: [0.0, 1.05, -0.02, 0.001]")

    # Also compute order 5
    bhc_cal5 = calibrate_bhc(energies, weights; order=5, reference_energy_keV=70.0)
    coeffs5 = bhc_cal5.polynomial.coefficients
    println("  Order 5: [$(join([round(c, digits=6) for c in coeffs5], ", "))]")
end

# ─── Cupping with proper per-spectrum BHC ───
println("\n" * "="^70)
println("CUPPING WITH PER-SPECTRUM CALIBRATED BHC")
println("="^70)

for (label, energies, weights, w, μ_w) in [("120 kVp", energies_120, weights_120, w120, μ_water_120),
                                             ("140 kVp", energies_140, weights_140, w140, μ_water_140)]
    bhc_cal = calibrate_bhc(energies, weights; order=5, reference_energy_keV=70.0)
    coeffs = bhc_cal.polynomial.coefficients

    bhc_cal_eval(p) = sum(coeffs[i+1] * p^i for i in 0:length(coeffs)-1)

    p_mono_center = μ_ref_70 * d_center
    p_mono_edge = μ_ref_70 * d_edge

    I_center = sum(w .* exp.(-μ_w .* d_center))
    p_center = -log(max(I_center, 1e-30))

    I_edge = sum(w .* exp.(-μ_w .* d_edge))
    p_edge = -log(max(I_edge, 1e-30))

    p_bhc_center = bhc_cal_eval(p_center)
    p_bhc_edge = bhc_cal_eval(p_edge)

    err_center = (p_bhc_center - p_mono_center) / d_center
    err_edge = (p_bhc_edge - p_mono_edge) / d_edge
    cupping_HU = 1000.0 * (err_center - err_edge) / μ_ref_70

    println("\n$(label) (calibrated order-5 BHC):")
    println("  CUPPING = $(round(cupping_HU, digits=1)) HU (should be ~0 if BHC is correct)")
end

println("\n" * "="^70)
println("DONE")
println("="^70)
