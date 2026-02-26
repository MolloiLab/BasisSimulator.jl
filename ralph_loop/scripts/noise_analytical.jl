#!/usr/bin/env julia
"""
NOISE-009: Analytical Noise Budget

Compute theoretical FDK noise using the Siewerdsen-Jaffray model and compare
with measured values from BasisSimulator (σ=125 HU) and CatSim (σ=71 HU).

FDK noise model (from Siewerdsen & Jaffray, Med Phys 2001):
    σ² ∝ (1/N_photons) × (FDK_noise_gain²) / (voxel_size³)

More precisely, for parallel beam:
    σ_μ² ≈ (π / N_angles) × (1 / (2 × N_cols)) × (Σ |ω|² × S(ω))

For fan beam FDK with ramp filter (Ram-Lak):
    σ_μ = √(π / (2 × N_angles)) × (1/I0_eff) × G_filter
    where G_filter = sqrt(Σ h²) depends on filter kernel
    and I0_eff accounts for object attenuation

Let's work from first principles with our actual measured data.
"""

println("═══════════════════════════════════════════════════════")
println("  NOISE-009: Analytical Noise Budget")
println("═══════════════════════════════════════════════════════")

# ═══════════════════════════════════════════════════════════
# A. Measured reference data
# ═══════════════════════════════════════════════════════════
println("\n--- A. Reference Data ---")

# From notebook 01 CNR figure:
σ_HU_basis = 125.61  # BasisSimulator full-physics noise
σ_HU_catsim = 71.37   # CatSim full-physics noise

# From NOISE-006 test:
σ_HU_noiseonly = 92.99     # BasisSimulator noise-only (ideal + noise)
σ_HU_fullphys = 123.96     # BasisSimulator full-physics (matches notebook within 2%)
σ_HU_noiseless = 11.7      # BasisSimulator noiseless baseline
σ_sino_noiseonly = 0.0668   # Sinogram noise σ (noise-only)
σ_sino_fullphys = 0.1165    # Sinogram noise σ (full-physics)
μ_water_noiseonly = 0.2066  # cm⁻¹
μ_water_fullphys = 0.2597  # cm⁻¹

println("  BasisSimulator (full-physics):   σ = $σ_HU_basis HU")
println("  CatSim (full-physics):           σ = $σ_HU_catsim HU")
println("  Ratio:                            $(round(σ_HU_basis / σ_HU_catsim, digits=3))x")

# ═══════════════════════════════════════════════════════════
# B. Noise in μ domain (before HU conversion)
# ═══════════════════════════════════════════════════════════
println("\n--- B. Noise in Attenuation Domain ---")

# σ_HU = 1000 × σ_μ / μ_water
# → σ_μ = σ_HU × μ_water / 1000

σ_μ_basis = σ_HU_fullphys * μ_water_fullphys / 1000  # cm⁻¹
σ_μ_noiseonly = σ_HU_noiseonly * μ_water_noiseonly / 1000  # cm⁻¹

# CatSim: μ_water = 0.02 mm⁻¹ = 0.2 cm⁻¹
μ_water_catsim = 0.2  # cm⁻¹ (0.02 mm⁻¹)
σ_μ_catsim = σ_HU_catsim * μ_water_catsim / 1000  # cm⁻¹

println("  BasisSimulator σ_μ (full-physics):  $(round(σ_μ_basis, sigdigits=4)) cm⁻¹")
println("  BasisSimulator σ_μ (noise-only):    $(round(σ_μ_noiseonly, sigdigits=4)) cm⁻¹")
println("  CatSim σ_μ:                          $(round(σ_μ_catsim, sigdigits=4)) cm⁻¹")
println()
println("  Ratio in μ domain (basis/catsim):    $(round(σ_μ_basis / σ_μ_catsim, digits=3))x")

# ═══════════════════════════════════════════════════════════
# C. Check: Does the μ_water calibration explain the difference?
# ═══════════════════════════════════════════════════════════
println("\n--- C. HU Conversion Impact ---")

# If BasisSimulator used CatSim's μ_water for HU conversion, what would σ_HU be?
σ_HU_basis_with_catsim_muwater = 1000 * σ_μ_basis / μ_water_catsim
println("  If we used CatSim's μ_water (0.2 cm⁻¹): σ = $(round(σ_HU_basis_with_catsim_muwater, digits=1)) HU")
println("  Actual CatSim σ:                          $(σ_HU_catsim) HU")
println("  Ratio with matched μ_water:                $(round(σ_HU_basis_with_catsim_muwater / σ_HU_catsim, digits=3))x")
println()

# And for noise-only:
σ_HU_noiseonly_with_catsim_muwater = 1000 * σ_μ_noiseonly / μ_water_catsim
println("  Noise-only with CatSim's μ_water: σ = $(round(σ_HU_noiseonly_with_catsim_muwater, digits=1)) HU")

# ═══════════════════════════════════════════════════════════
# D. Full-physics μ_water anomaly check
# ═══════════════════════════════════════════════════════════
println("\n--- D. μ_water Calibration Analysis ---")
println("  NIST water @ 60 keV:      0.2059 cm⁻¹")
println("  NIST water @ 70 keV:      0.1928 cm⁻¹")
println("  Noise-only μ_water:        $μ_water_noiseonly cm⁻¹ (monochromatic 60 keV)")
println("  Full-physics μ_water:      $μ_water_fullphys cm⁻¹ (polychromatic + BHC)")
println("  CatSim μ_water:            0.2 cm⁻¹ (hardcoded)")
println()

# The noise-only μ_water = 0.2066 makes sense for 60 keV water
# The full-physics μ_water = 0.2597 is WAY too high!
# Possible explanations:
# 1. BHC overcorrecting (pushing μ_water up)
# 2. Beam hardening in polychromatic sim increasing effective μ
# 3. Bug in water calibration scan

μ_water_ratio = μ_water_fullphys / μ_water_noiseonly
println("  Full-physics / Noise-only μ_water ratio: $(round(μ_water_ratio, digits=3))")
println("  This means full-physics μ_water is $(round((μ_water_ratio - 1) * 100, digits=1))% higher than expected")
println()

# ═══════════════════════════════════════════════════════════
# E. What σ_HU would we get with correct μ_water?
# ═══════════════════════════════════════════════════════════
println("\n--- E. Expected Noise with Different μ_water Values ---")
println()

# The reconstruction noise σ_μ should be independent of μ_water choice
# (it's a property of the sinogram noise + FDK)
# Only the HU conversion changes: σ_HU = 1000 × σ_μ / μ_water

# What if we used the noise-only μ_water for the full-physics result?
σ_HU_fullphys_corrected_muwater = 1000 * σ_μ_basis / μ_water_noiseonly
println("  Full-physics with noise-only μ_water:  σ = $(round(σ_HU_fullphys_corrected_muwater, digits=1)) HU")

# What if we used NIST 70 keV?
σ_HU_fullphys_nist70 = 1000 * σ_μ_basis / 0.1928
println("  Full-physics with NIST 70 keV μ_water: σ = $(round(σ_HU_fullphys_nist70, digits=1)) HU")

# What if we used CatSim's 0.02 mm⁻¹ = 0.2 cm⁻¹?
println("  Full-physics with CatSim 0.2 cm⁻¹:    σ = $(round(σ_HU_basis_with_catsim_muwater, digits=1)) HU")
println("  CatSim reference:                       σ = $σ_HU_catsim HU")

# ═══════════════════════════════════════════════════════════
# F. Decompose the noise ratio into factors
# ═══════════════════════════════════════════════════════════
println("\n═══════════════════════════════════════════════════════")
println("  NOISE BUDGET DECOMPOSITION")
println("═══════════════════════════════════════════════════════")
println()

# Total ratio = σ_basis / σ_catsim = 125.61 / 71.37 = 1.76x
#
# Factor 1: μ_water calibration difference
#   σ_HU = 1000 × σ_μ / μ_water
#   If μ_water differs between us and CatSim, it scales noise
#   Factor = μ_water_catsim / μ_water_basis
factor_muwater = μ_water_catsim / μ_water_fullphys

# Factor 2: Sinogram noise difference (same I0, but physics effects differ)
#   CatSim physics may produce less sinogram noise or more sinogram noise
#   We can't directly measure this, but we know noise-only σ_sino = 0.0668
#   and full-physics σ_sino = 0.1165
#   Factor = σ_sino_basis / σ_sino_catsim (unknown)

# Factor 3: FDK noise gain difference
#   NOISE-008 showed this is ~0.934 (ours slightly lower)
factor_fdk = 0.934

# Factor 4: Noiseless baseline
#   Our noiseless σ = 11.7 HU adds in quadrature
#   σ_total² = σ_noise² + σ_artifact²
#   So σ_noise² = σ_total² - σ_artifact²
σ_noise_only_corrected = sqrt(σ_HU_fullphys^2 - σ_HU_noiseless^2)
println("  After subtracting noiseless baseline:")
println("    σ_basis (corrected)  = √($(round(σ_HU_fullphys, digits=1))² - $(σ_HU_noiseless)²) = $(round(σ_noise_only_corrected, digits=1)) HU")

σ_ratio_corrected = σ_noise_only_corrected / σ_HU_catsim
println("    Corrected ratio:     = $(round(σ_ratio_corrected, digits=3))x")
println()

println("  Noise budget breakdown:")
println("  ─────────────────────────────────────────")
println("  Measured ratio:                   $(round(σ_HU_basis / σ_HU_catsim, digits=3))x")
println("  Factor 1: μ_water calibration:    $(round(1/factor_muwater, digits=3))x  (μ_water_ours=$(round(μ_water_fullphys, digits=4)) vs CatSim=0.2)")
println("  Factor 2: FDK filter gain:        $(round(1/factor_fdk, digits=3))x  (NOISE-008 result)")
println("  Factor 3: Noiseless baseline:     adds $(round(σ_HU_noiseless, digits=1)) HU in quadrature")
println("  Residual:                          $(round(σ_ratio_corrected * factor_muwater * factor_fdk, digits=3))x")
println()

# ═══════════════════════════════════════════════════════════
# G. Key insight about μ_water
# ═══════════════════════════════════════════════════════════
println("  ═══ KEY INSIGHT ═══")
println()
println("  Our μ_water (full-physics) = 0.2597 cm⁻¹ is ANOMALOUSLY HIGH")
println("  Expected: ~0.19-0.21 cm⁻¹ for effective energy of 120 kVp beam")
println("  CatSim uses: 0.2 cm⁻¹")
println()
println("  High μ_water REDUCES σ_HU: σ_HU = 1000 × σ_μ / μ_water")
println("  So with μ_water=0.2597: σ_HU = 1000 × $(round(σ_μ_basis, sigdigits=4)) / 0.2597 = $(round(σ_HU_fullphys, digits=1)) HU")
println("  With μ_water=0.2:       σ_HU = 1000 × $(round(σ_μ_basis, sigdigits=4)) / 0.2 = $(round(σ_HU_basis_with_catsim_muwater, digits=1)) HU")
println()
println("  So the high μ_water is actually SUPPRESSING noise by a factor of $(round(μ_water_fullphys/μ_water_catsim, digits=3))x")
println("  Without this suppression, our noise would be $(round(σ_HU_basis_with_catsim_muwater, digits=1)) HU vs CatSim's $(σ_HU_catsim) HU")
println("  True attenuation-domain noise ratio: $(round(σ_μ_basis / σ_μ_catsim, digits=3))x")

println("\n═══════════════════════════════════════════════════════")
println("  NOISE-009 COMPLETE")
println("═══════════════════════════════════════════════════════")
