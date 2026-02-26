#!/usr/bin/env julia
"""
NOISE-008: Ramp Filter Noise Amplification Comparison

Compare noise gain from:
  1. Our spatial domain ramp filter (h[0] = 1/(4Δ), h[n_odd] = -1/(π²n²Δ))
  2. CatSim's dimensionless kernel + /DeltaUW (h[0] = 0.25/DeltaUW, h[n_odd] = -1/(π²n²·DeltaUW))

Both with n_cols = 900 and Δ = pixel_size_at_isocenter = 0.0569 cm

The noise gain from convolution with kernel h is: G = sqrt(Σ h[k]²)
"""

using Statistics

println("═══════════════════════════════════════════════════════")
println("  NOISE-008: Ramp Filter Noise Amplification Comparison")
println("═══════════════════════════════════════════════════════")

# Parameters from notebook 01
n_cols = 900
sid_mm = 540.0
sdd_mm = 950.0
magnification = sdd_mm / sid_mm
pixel_face_mm = 1.0  # detector pixel size at detector face
pixel_iso_mm = pixel_face_mm / magnification  # at isocenter = 0.5684 mm
pixel_iso_cm = pixel_iso_mm / 10.0  # 0.05684 cm (our internal units)

# CatSim equi-angle parameters
DecFanAng = 2 * atan(n_cols / 2 * pixel_face_mm / sdd_mm)
DeltaUW = DecFanAng / n_cols  # radians per detector element

println("\n--- Parameters ---")
println("  n_cols = $n_cols")
println("  SID = $sid_mm mm, SDD = $sdd_mm mm")
println("  magnification = $(round(magnification, digits=4))")
println("  pixel_face = $pixel_face_mm mm")
println("  pixel_iso = $(round(pixel_iso_mm, digits=4)) mm = $(round(pixel_iso_cm, digits=6)) cm")
println("  DecFanAng = $(round(DecFanAng, digits=6)) rad = $(round(rad2deg(DecFanAng), digits=2))°")
println("  DeltaUW = $(round(DeltaUW, sigdigits=6)) rad")
println("  pixel_iso / DeltaUW ratio = $(round(pixel_iso_mm / DeltaUW, digits=2)) (should ≈ SID for small angles)")
println("  SID = $sid_mm mm")

# ─── 1. Our kernel (spatial domain, equi-space) ───
Δ = pixel_iso_cm  # our code uses cm

our_kernel = zeros(Float64, n_cols)
center = n_cols ÷ 2 + 1
for i in 1:n_cols
    k = i - center
    if k == 0
        our_kernel[i] = 1.0 / (4.0 * Δ)
    elseif k % 2 == 0
        our_kernel[i] = 0.0
    else
        our_kernel[i] = -1.0 / (π^2 * k^2 * Δ)
    end
end

# ─── 2. CatSim kernel (dimensionless, equi-angle, then divided by DeltaUW) ───
# CatSim kernel BEFORE /DeltaUW:
catsim_kernel_raw = zeros(Float64, n_cols)
for i in 1:n_cols
    k = i - center
    if k == 0
        catsim_kernel_raw[i] = 0.25
    elseif k % 2 == 0
        catsim_kernel_raw[i] = 0.0
    else
        catsim_kernel_raw[i] = -sin(π * k / 2)^2 / (π^2 * k^2)
    end
end

# sin²(πk/2) = 1 for all odd k, so this is just -1/(π²k²) for odd k
# Same as our kernel * Δ (i.e., our kernel = catsim_raw / Δ)

# After /DeltaUW: effective kernel for noise
catsim_kernel_eff = catsim_kernel_raw ./ DeltaUW

# ─── 3. Compute noise gains ───
our_noise_gain = sqrt(sum(our_kernel .^ 2))
catsim_noise_gain_raw = sqrt(sum(catsim_kernel_raw .^ 2))
catsim_noise_gain_eff = sqrt(sum(catsim_kernel_eff .^ 2))

println("\n--- Kernel Comparison ---")
println("  Our h[0]    = $(round(our_kernel[center], digits=4))")
println("  CatSim h[0] = $(catsim_kernel_raw[center]) (raw)")
println("  CatSim h[0] = $(round(catsim_kernel_raw[center] / DeltaUW, digits=4)) (after /DeltaUW)")
println()
println("  Our h[0] / CatSim_eff h[0] ratio = $(round(our_kernel[center] / catsim_kernel_eff[center], digits=6))")
println("  This ratio should be 1.0 for equivalent noise. If not, it explains the noise discrepancy.")

println("\n--- Noise Gain (sqrt of sum of squared kernel) ---")
println("  Our noise gain        = $(round(our_noise_gain, digits=4))")
println("  CatSim noise gain_raw = $(round(catsim_noise_gain_raw, digits=4)) (before /DeltaUW)")
println("  CatSim noise gain_eff = $(round(catsim_noise_gain_eff, digits=4)) (after /DeltaUW)")
println()
println("  Ratio (ours / CatSim_eff) = $(round(our_noise_gain / catsim_noise_gain_eff, digits=6))")

# ─── 4. Now account for backprojection scaling differences ───
# CatSim backprojection: accumulates filtered/Dlocal², then *= -ScanR × π / ProjNum
# Ours: accumulates filtered × SAD²/dist², then *= π/N_angles
#
# For center voxel (Dlocal = dist = ScanR = SAD):
#   CatSim: ScanR × π / ProjNum × filtered / ScanR² = π × filtered / (ProjNum × ScanR)
#   Ours:   π / N × filtered × SAD² / SAD² = π × filtered / N
#
# Since ProjNum = N_angles, the backprojection scaling ratio at center is:
#   Ours / CatSim = (π × filtered_ours / N) / (π × filtered_catsim / (N × ScanR))
#                 = filtered_ours × ScanR / filtered_catsim
#
# But we want NOISE ratio, not value ratio. For noise:
#   σ_recon ∝ filter_noise_gain × backproj_scale

ScanR_mm = sid_mm  # 540 mm
ScanR_cm = ScanR_mm / 10.0  # 54 cm
N_angles = 984

# CatSim: backproj noise scale at center = ScanR × π / ProjNum / ScanR² = π / (N × ScanR)
catsim_bp_scale = π / (N_angles * ScanR_mm)  # mm⁻¹ per filtered value

# Ours: backproj noise scale at center = π / N × SAD² / SAD² = π / N (dimensionless)
# BUT our filtered values already have units of cm⁻¹ (due to 1/Δ in kernel)
our_bp_scale = π / N_angles

# Total noise gain = filter_noise_gain × backproj_scale
our_total_noise = our_noise_gain * our_bp_scale
catsim_total_noise = catsim_noise_gain_eff * catsim_bp_scale

println("\n--- Backprojection Scaling (center voxel) ---")
println("  CatSim BP scale = π / (N × ScanR) = $(round(catsim_bp_scale, sigdigits=4)) mm⁻¹")
println("  Our BP scale    = π / N            = $(round(our_bp_scale, sigdigits=4))")
println()
println("  CatSim total noise gain = $(round(catsim_total_noise, sigdigits=4))")
println("  Our total noise gain    = $(round(our_total_noise, sigdigits=4))")
println("  Ratio (ours / CatSim)   = $(round(our_total_noise / catsim_total_noise, sigdigits=6))")

# ─── 5. Alternative: compute the pixel-spacing relationship ───
# For equi-space flat detector, at small angles:
#   DeltaUW ≈ pixel_det / SDD = pixel_iso × magnification / SDD = pixel_iso / SAD
# Let's verify:
DeltaUW_approx = pixel_iso_mm / sid_mm
println("\n--- Geometric Relationship ---")
println("  DeltaUW (exact)  = $(round(DeltaUW, sigdigits=6)) rad")
println("  DeltaUW (approx) = pixel_iso / SAD = $(round(DeltaUW_approx, sigdigits=6))")
println("  Ratio            = $(round(DeltaUW / DeltaUW_approx, digits=6))")

# So: 1/Δ vs 1/DeltaUW: our kernel has 1/Δ_cm, CatSim effective kernel has 1/DeltaUW
# Ratio: (1/Δ_cm) / (1/DeltaUW) = DeltaUW / Δ_cm
# DeltaUW ≈ Δ_mm / SAD_mm (small angle approx for equi-space)
# So ratio = (Δ_mm / SAD_mm) / Δ_cm = (Δ_mm / SAD_mm) / (Δ_mm / 10)
#          = 10 / SAD_mm = 10 / 540 = 0.01852
# But then CatSim BP also has an extra ScanR factor...

# Let me be more careful about units
println("\n--- Unit-Consistent Comparison ---")
println("  Our kernel is in cm⁻¹ (because Δ is in cm)")
println("  CatSim effective kernel is in rad⁻¹ (dimensionless / DeltaUW)")
println("  These have different dimensions!")
println()
println("  To compare, multiply through full pipeline:")
println()

# For constant sinogram p:
# CatSim result = -ScanR_mm × π / ProjNum × Σ_angles [ (convolution result / DeltaUW) / Dlocal² ]
# At center (Dlocal = ScanR):
#   = -ScanR × π / N × N × (p × 0.25 / DeltaUW) / ScanR²
#   = -π × p × 0.25 / (DeltaUW × ScanR)
catsim_per_p = π * 0.25 / (DeltaUW * ScanR_mm)
println("  CatSim center voxel (per unit sinogram p) = $(round(catsim_per_p, sigdigits=4)) mm⁻¹")

# Our result = π / N × Σ_angles [ convolution result × SAD² / dist² ]
# At center (dist = SAD):
#   = π / N × N × (p × 1/(4Δ_cm))
#   = π × p / (4 × Δ_cm)
our_per_p = π / (4 * pixel_iso_cm)
println("  Our center voxel (per unit sinogram p)     = $(round(our_per_p, sigdigits=4)) cm⁻¹ = $(round(our_per_p/10, sigdigits=4)) mm⁻¹")
println("  Ratio = $(round(our_per_p/10 / catsim_per_p, digits=6)) (should be ~1.0)")

# For NOISE: the noise in the reconstructed image is proportional to filter noise gain × BP scale
# σ_recon = σ_sino × sqrt(Σ h²) × BP_scale × sqrt(N_angles)
# (The sqrt(N) comes from N independent noise additions in backprojection)

# CatSim: σ_recon = σ_sino × sqrt(Σ (h_raw/DeltaUW)²) × (ScanR × π / N) / ScanR² × sqrt(N)
#                 = σ_sino × (noise_gain_eff) × π / (N × ScanR) × ScanR × sqrt(N)
#   Wait, need to be more careful with the negative sign and scaling

# Actually, for noise purposes:
# Each angle contributes: σ_filtered × weight (where weight = 1/Dlocal² for CatSim, SAD²/dist² for ours)
# The contributions from N angles add in quadrature (independent noise)
# Then multiply by the overall scale factor

# CatSim noise per angle at center:
catsim_noise_per_angle = catsim_noise_gain_eff / ScanR_mm^2  # filtered noise / Dlocal²
catsim_noise_total = catsim_noise_per_angle * sqrt(Float64(N_angles)) * ScanR_mm * π / N_angles
# = noise_gain / ScanR² × sqrt(N) × ScanR × π / N
# = noise_gain × π × sqrt(N) / (ScanR × N)
# = noise_gain × π / (ScanR × sqrt(N))

# Our noise per angle at center:
our_noise_per_angle = our_noise_gain * ScanR_cm^2 / ScanR_cm^2  # filtered noise × SAD²/dist² = filtered noise
our_noise_total = our_noise_per_angle * sqrt(Float64(N_angles)) * π / N_angles
# = noise_gain × sqrt(N) × π / N
# = noise_gain × π / sqrt(N)

println("\n--- Noise Amplification Through Full Pipeline ---")
println("  CatSim σ_recon/σ_sino = noise_gain × π / (ScanR × sqrt(N))")
println("                        = $(round(catsim_noise_gain_eff, digits=2)) × π / ($(ScanR_mm) × $(round(sqrt(Float64(N_angles)), digits=2)))")
println("                        = $(round(catsim_noise_gain_eff * π / (ScanR_mm * sqrt(Float64(N_angles))), sigdigits=4)) mm⁻¹")
println()
println("  Our σ_recon/σ_sino    = noise_gain × π / sqrt(N)")
println("                        = $(round(our_noise_gain, digits=2)) × π / $(round(sqrt(Float64(N_angles)), digits=2))")
println("                        = $(round(our_noise_gain * π / sqrt(Float64(N_angles)), sigdigits=4)) cm⁻¹")
println("                        = $(round(our_noise_gain * π / sqrt(Float64(N_angles)) / 10, sigdigits=4)) mm⁻¹")
println()

noise_ratio = (our_noise_gain * π / sqrt(Float64(N_angles)) / 10) / (catsim_noise_gain_eff * π / (ScanR_mm * sqrt(Float64(N_angles))))
println("  Noise ratio (ours/CatSim) = $(round(noise_ratio, digits=4))")
println("  Expected if correct: ~1.0")
println("  If ~5.5: explains the noise discrepancy!")

# ─── 6. Direct approach: just compute Σ|h|² for both effective kernels ───
# "Effective kernel" = kernel as applied in the full pipeline, including all scaling
# CatSim effective: h_raw / DeltaUW, then each sample weighted by 1/Dlocal², scaled by ScanR×π/N
# At center: (h_raw / DeltaUW) × (1/ScanR²) × ScanR × π/N = h_raw × π / (DeltaUW × ScanR × N)
# Noise: sqrt(Σ (h_raw × π / (DeltaUW × ScanR × N))² × N)  [N angles add in quadrature]
#       = sqrt(N) × π / (DeltaUW × ScanR × N) × sqrt(Σ h_raw²)
#       = π / (DeltaUW × ScanR × sqrt(N)) × sqrt(Σ h_raw²)

# Ours: h_our, then each sample weighted by SAD²/dist²=1, scaled by π/N
# At center: h_our × π / N  [units: cm⁻¹]
# Noise: sqrt(N) × π / N × sqrt(Σ h_our²)
#       = π / sqrt(N) × sqrt(Σ h_our²)

# To compare in same units (mm⁻¹):
# CatSim: π / (DeltaUW × ScanR × sqrt(N)) × sqrt(Σ h_raw²) [mm⁻¹]
# Ours: π / (10 × sqrt(N)) × sqrt(Σ h_our²) [mm⁻¹, converting cm⁻¹ to mm⁻¹]

catsim_total = π / (DeltaUW * ScanR_mm * sqrt(Float64(N_angles))) * sqrt(sum(catsim_kernel_raw .^ 2))
our_total = π / (10.0 * sqrt(Float64(N_angles))) * sqrt(sum(our_kernel .^ 2))

println("\n═══════════════════════════════════════════════════════")
println("  FINAL NOISE COMPARISON (in mm⁻¹)")
println("═══════════════════════════════════════════════════════")
println("  CatSim noise gain = $(round(catsim_total, sigdigits=6)) mm⁻¹/σ_sino")
println("  Our noise gain    = $(round(our_total, sigdigits=6)) mm⁻¹/σ_sino")
println("  Ratio             = $(round(our_total / catsim_total, digits=6))")
println()

if abs(our_total / catsim_total - 1.0) < 0.1
    println("  ✓ Noise gains MATCH (within 10%) — filter is NOT the problem")
    println("  → Look elsewhere for the noise discrepancy")
elseif our_total > catsim_total
    println("  ✗ Our noise gain is $(round(our_total / catsim_total, digits=2))x HIGHER than CatSim")
    println("  → This is a significant noise amplification difference!")
else
    println("  ? Our noise gain is $(round(catsim_total / our_total, digits=2))x LOWER than CatSim")
end

println("\n═══════════════════════════════════════════════════════")
println("  NOISE-008 COMPLETE")
println("═══════════════════════════════════════════════════════")
