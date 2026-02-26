#!/usr/bin/env julia
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
#   1. Martinez C, Fessler JA, et al. "Simple beam hardening correction method
#      (2DCalBH) based on 2D linearization." Phys Med Biol. 2022;67(11).
#   2. Joseph PM, Spital RD. "A method for correcting bone induced artifacts
#      in CT scanners." J Comput Assist Tomogr. 1978;2(1):100-108.
#   3. Elbakri IA, Fessler JA. "Segmentation-free statistical image reconstruction
#      for polyenergetic x-ray CT." Phys Med Biol. 2003;48(15):2453-2477.
#
# Output: ralph_loop/results/bhc_two_material.png
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using BasisSimulator
import XrayAttenuation as XA
using Unitful: ustrip, @u_str
using Statistics
using CairoMakie

println("=" ^ 70)
println("  Two-Material Beam Hardening Correction Test")
println("=" ^ 70)

# =============================================================================
# [1] CONFIGURATION
# =============================================================================

const KVP = 120
const N_VOXELS = 256
const N_SLICES = 16
const N_ANGLES = 720
const N_ROWS = 16
const N_COLS = 736
const FOV_CM = 35.0
const Z_CM = 2.0
const N_ENERGY_BINS = 30
const REFERENCE_ENERGY_KEV = 70.0

# Smooth tissue fraction thresholds (in HU)
# Below HU_LOW: 100% soft tissue (water-like)
# Above HU_HIGH: 100% bone-like
# Between: smooth cubic interpolation (no hard segmentation)
const HU_LOW = 100.0    # soft tissue / bone boundary
const HU_HIGH = 500.0   # fully bone

println("\n[Config]")
println("  Phantom: Gammex 472, $(N_VOXELS)×$(N_VOXELS)×$(N_SLICES)")
println("  Geometry: $(N_ANGLES) angles, $(N_ROWS)×$(N_COLS) detector")
println("  Spectrum: $(KVP) kVp, $(N_ENERGY_BINS) energy bins")
println("  Reference energy: $(REFERENCE_ENERGY_KEV) keV")
println("  Bone fraction thresholds: $(HU_LOW) – $(HU_HIGH) HU")

# =============================================================================
# [2] LOAD SPECTRUM AND MATERIALS
# =============================================================================

println("\n[2] Loading spectrum and material data...")

energies, weights = load_spectrum(KVP)
energies, weights = downsample_spectrum(energies, weights, N_ENERGY_BINS)
w_norm = weights ./ sum(weights)

# Water attenuation at each spectrum energy
water_mat = XA.Materials.water
μ_water_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, E * u"keV")) for E in energies]
μ_water_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(water_mat, REFERENCE_ENERGY_KEV * u"keV"))

# Cortical bone attenuation at each spectrum energy
bone_mat = XA.Materials.corticalbone
μ_bone_E = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, E * u"keV")) for E in energies]
μ_bone_ref = ustrip(u"cm^-1", XA.linear_attenuation_coeff(bone_mat, REFERENCE_ENERGY_KEV * u"keV"))

println("  μ_water($(REFERENCE_ENERGY_KEV) keV) = $(round(μ_water_ref, digits=5)) cm⁻¹")
println("  μ_bone($(REFERENCE_ENERGY_KEV) keV)  = $(round(μ_bone_ref, digits=5)) cm⁻¹")
println("  Bone/water ratio at ref energy: $(round(μ_bone_ref/μ_water_ref, digits=2))×")
println("  Bone/water ratio at 30 keV:     $(round(μ_bone_E[1]/μ_water_E[1], digits=2))×")
println("  → Bone hardens spectrum differently than water (ratio changes with E)")

# =============================================================================
# [3] CREATE PHANTOM AND GEOMETRY
# =============================================================================

println("\n[3] Creating phantom and geometry...")

phantom = create_gammex_472(n_voxels=N_VOXELS, n_slices=N_SLICES, fov_cm=FOV_CM, z_cm=Z_CM)
geom = create_aquilion_one(n_angles=N_ANGLES, n_rows=N_ROWS, n_cols=N_COLS,
                           fov_cm=FOV_CM, z_cm=Z_CM)
materials = get_region_materials()

matrix_size = (N_VOXELS, N_VOXELS, N_SLICES)

println("  Phantom extent: $(phantom.extent)")
println("  Geometry FOV:   $(geom.fov)")
println("  Matrix size:    $(matrix_size)")

# =============================================================================
# [4] FORWARD PROJECT (POLYCHROMATIC, NO BHC, NO NOISE)
# =============================================================================
#
# We run the polychromatic forward projection ourselves so we get the raw
# (beam-hardened) sinogram before any correction. No noise — we want to
# see pure BHC effects.
# =============================================================================

println("\n[4] Running polychromatic forward projection (no BHC, no noise)...")

sino_raw = forward_project(phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    volume_extent=phantom.extent)

sino_raw_cpu = Array(sino_raw)

println("  Sinogram size: $(size(sino_raw_cpu))")
println("  Sinogram range: [$(round(minimum(sino_raw_cpu), digits=4)), $(round(maximum(sino_raw_cpu), digits=4))]")
println("  Mean value: $(round(mean(sino_raw_cpu), digits=4))")

# =============================================================================
# [5] RECONSTRUCT WITH NO BHC
# =============================================================================

println("\n[5] Reconstructing with NO BHC...")

recon_no_bhc = Array(fdk_reconstruct(sino_raw, geom, matrix_size))

hu_no_bhc = 1000f0 .* (recon_no_bhc .- μ_water_ref) ./ μ_water_ref

mid_slice = N_SLICES ÷ 2
cx, cy = N_VOXELS ÷ 2, N_VOXELS ÷ 2
r_roi = 10  # radius for center ROI

center_hu_no = mean(hu_no_bhc[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center water HU (no BHC): $(round(center_hu_no, digits=1)) HU")
println("  (Should be ~0 HU for water, negative = cupping artifact)")

# =============================================================================
# [6] WATER-ONLY BHC → RECONSTRUCT
# =============================================================================

println("\n[6] Applying water-only BHC → reconstructing...")

bhc_water = calibrate_bhc(energies, weights; order=5, reference_energy_keV=REFERENCE_ENERGY_KEV)

sino_water_bhc = apply_bhc(sino_raw, bhc_water)
recon_water_bhc = Array(fdk_reconstruct(sino_water_bhc, geom, matrix_size))
hu_water_bhc = 1000f0 .* (recon_water_bhc .- μ_water_ref) ./ μ_water_ref

center_hu_water = mean(hu_water_bhc[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center water HU (water BHC): $(round(center_hu_water, digits=1)) HU")
println("  (Cupping should be gone, but bone dark-band artifacts remain)")

# =============================================================================
# [7] TWO-MATERIAL BHC: THE ALGORITHM
# =============================================================================
#
# Martinez/Fessler 2022 adapted for known-spectrum simulation:
#
#   Step 1: From water-BHC recon, decompose into soft/bone using smooth
#           density-dependent tissue fractions (Elbakri/Fessler 2003).
#           NO hard threshold — uses C1-continuous smoothstep.
#
#   Step 2: Forward-project the bone-fraction-weighted μ image using Siddon
#           to get bone line integral p_b per ray.
#           Soft tissue line integral: p_s = p_water_corrected - p_b
#
#   Step 3: Convert to equivalent material path lengths:
#           L_w = p_s / μ_water_ref   (cm of water)
#           L_b = p_b / μ_bone_ref    (cm of bone)
#
#   Step 4: Compute exact 2-material polychromatic forward model:
#           F(L_w, L_b) = -log(Σ wₑ × exp(-μ_water(E)×L_w - μ_bone(E)×L_b))
#           And the monochromatic reference:
#           p_mono = μ_water_ref × L_w + μ_bone_ref × L_b = p_s + p_b
#
#   Step 5: The correction for each ray:
#           Δp = p_mono - F(L_w, L_b)
#           This captures the FULL beam hardening (both water and bone).
#           Apply to the RAW sinogram: p_corrected = p_raw + Δp
#
# Key insight: With known spectrum, Step 4 is EXACT. The only approximation
# is the tissue fraction decomposition in Step 1, which uses the preliminary
# (water-BHC) reconstruction. But this is very robust — even 10% errors in
# tissue fractions have negligible effect on the correction (Joseph & Spital 1978).
# =============================================================================

println("\n[7] Two-material BHC (water + bone)...")

# --- Step 1: Smooth tissue fraction decomposition ---
# Smoothstep (C1 Hermite): f(t) = 3t² - 2t³ for t ∈ [0,1]
# This avoids the hard threshold that plagues Joseph & Spital.
println("  Step 1: Smooth tissue fraction decomposition...")

function bone_fraction_smooth(hu::Real)
    hu <= HU_LOW && return 0.0
    hu >= HU_HIGH && return 1.0
    t = (hu - HU_LOW) / (HU_HIGH - HU_LOW)
    return t * t * (3.0 - 2.0 * t)  # smoothstep
end

# Apply to water-BHC reconstruction
bone_frac = bone_fraction_smooth.(hu_water_bhc[:, :, mid_slice])
bone_μ_image = bone_frac .* recon_water_bhc[:, :, mid_slice]   # bone-weighted μ
soft_μ_image = (1.0f0 .- bone_frac) .* recon_water_bhc[:, :, mid_slice]  # soft-weighted μ

# Extend to 3D (replicate mid-slice for all slices for forward projection)
bone_μ_3d = zeros(Float32, N_VOXELS, N_VOXELS, N_SLICES)
soft_μ_3d = zeros(Float32, N_VOXELS, N_VOXELS, N_SLICES)
for s in 1:N_SLICES
    bone_frac_s = bone_fraction_smooth.(hu_water_bhc[:, :, s])
    bone_μ_3d[:, :, s] .= bone_frac_s .* recon_water_bhc[:, :, s]
    soft_μ_3d[:, :, s] .= (1.0f0 .- bone_frac_s) .* recon_water_bhc[:, :, s]
end

n_bone_voxels = count(bone_frac .> 0.01)
println("    Bone voxels (f_b > 0.01): $(n_bone_voxels) / $(N_VOXELS^2)")
println("    Max bone fraction: $(round(maximum(bone_frac), digits=3))")

# --- Step 2: Forward-project bone image ---
println("  Step 2: Forward-projecting bone image (Siddon)...")

p_b = Array(siddon_forward_project(bone_μ_3d, geom; volume_extent=phantom.extent))

# Soft tissue line integral from water-corrected sinogram
sino_water_bhc_cpu = Array(sino_water_bhc)
p_s = sino_water_bhc_cpu .- p_b

println("    Bone sinogram range: [$(round(minimum(p_b), digits=4)), $(round(maximum(p_b), digits=4))]")
println("    Soft sinogram range: [$(round(minimum(p_s), digits=4)), $(round(maximum(p_s), digits=4))]")

# --- Step 3: Convert to equivalent path lengths ---
println("  Step 3: Converting to material path lengths...")

L_w = p_s ./ Float32(μ_water_ref)   # cm of water
L_b = p_b ./ Float32(μ_bone_ref)    # cm of bone

# Clamp negatives (can happen at edges from reconstruction artifacts)
L_w .= max.(L_w, 0.0f0)
L_b .= max.(L_b, 0.0f0)

println("    Water path range: [$(round(minimum(L_w), digits=2)), $(round(maximum(L_w), digits=2))] cm")
println("    Bone path range:  [$(round(minimum(L_b), digits=2)), $(round(maximum(L_b), digits=2))] cm")

# --- Step 4: Compute exact 2-material polychromatic + correction ---
println("  Step 4: Computing exact 2-material BH correction ($(length(sino_raw_cpu)) rays)...")

# Pre-convert to Float64 for precision
w_norm_64 = Float64.(w_norm)
μ_w_64 = Float64.(μ_water_E)
μ_b_64 = Float64.(μ_bone_E)

correction = zeros(Float32, size(sino_raw_cpu))

Threads.@threads for idx in eachindex(sino_raw_cpu)
    Lw = Float64(L_w[idx])
    Lb = Float64(L_b[idx])

    # Skip rays with no material
    if Lw < 1e-6 && Lb < 1e-6
        continue
    end

    # Polychromatic forward model: F(L_w, L_b) = -log(Σ wₑ exp(-μ_w(E)L_w - μ_b(E)L_b))
    I_poly = 0.0
    for e in eachindex(w_norm_64)
        I_poly += w_norm_64[e] * exp(-μ_w_64[e] * Lw - μ_b_64[e] * Lb)
    end
    F_2mat = I_poly > 0.0 ? -log(I_poly) : 0.0

    # Monochromatic reference at ref energy
    p_mono = μ_water_ref * Lw + μ_bone_ref * Lb

    # Full BH correction (captures both water and bone hardening)
    correction[idx] = Float32(p_mono - F_2mat)
end

println("    Correction range: [$(round(minimum(correction), digits=4)), $(round(maximum(correction), digits=4))]")
println("    Mean correction:  $(round(mean(correction), digits=4))")

# --- Step 5: Apply correction to RAW sinogram ---
println("  Step 5: Applying correction to raw sinogram...")

sino_2mat_bhc = sino_raw_cpu .+ correction

println("    Corrected sinogram range: [$(round(minimum(sino_2mat_bhc), digits=4)), $(round(maximum(sino_2mat_bhc), digits=4))]")

# =============================================================================
# [8] RECONSTRUCT WITH TWO-MATERIAL BHC
# =============================================================================

println("\n[8] Reconstructing with two-material BHC...")

recon_2mat_bhc = Array(fdk_reconstruct(sino_2mat_bhc, geom, matrix_size))
hu_2mat_bhc = 1000f0 .* (recon_2mat_bhc .- μ_water_ref) ./ μ_water_ref

center_hu_2mat = mean(hu_2mat_bhc[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center water HU (2-mat BHC): $(round(center_hu_2mat, digits=1)) HU")

# =============================================================================
# [9] QUANTITATIVE COMPARISON
# =============================================================================

println("\n" * "=" ^ 70)
println("  QUANTITATIVE RESULTS")
println("=" ^ 70)

println("\n--- Center Water ROI ($(2*r_roi+1)×$(2*r_roi+1) pixels) ---")
println("  No BHC:          $(round(center_hu_no, digits=1)) HU  (expect large negative = cupping)")
println("  Water-only BHC:  $(round(center_hu_water, digits=1)) HU  (cupping fixed, bone streaks remain)")
println("  Water+Bone BHC:  $(round(center_hu_2mat, digits=1)) HU  (both fixed)")

# Measure dark-band artifact between calcium inserts
# The Ca inserts are arranged in a circle at ~50mm radius from center
# Dark bands appear on lines connecting dense inserts through the center
println("\n--- Noise / Artifact Metrics ---")
σ_no = std(hu_no_bhc[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
σ_water = std(hu_water_bhc[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
σ_2mat = std(hu_2mat_bhc[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center ROI σ (no BHC):      $(round(σ_no, digits=1)) HU")
println("  Center ROI σ (water BHC):   $(round(σ_water, digits=1)) HU")
println("  Center ROI σ (2-mat BHC):   $(round(σ_2mat, digits=1)) HU")

# Check a few calcium insert HU values
# Ca inserts are at ~50mm = 5cm radius, in a circle, using Gammex layout
println("\n--- Global Image Statistics (mid-slice) ---")
for (label, hu) in [("No BHC", hu_no_bhc), ("Water BHC", hu_water_bhc), ("2-Mat BHC", hu_2mat_bhc)]
    sl = hu[:, :, mid_slice]
    # Mask out air (background)
    body_mask = sl .> -500
    body_vals = sl[body_mask]
    println("  $(rpad(label, 12)): mean=$(round(mean(body_vals), digits=1)) HU, " *
            "std=$(round(std(body_vals), digits=1)) HU, " *
            "range=[$(round(minimum(body_vals), digits=0)), $(round(maximum(body_vals), digits=0))] HU")
end

# =============================================================================
# [10] GENERATE COMPARISON FIGURE
# =============================================================================

println("\n[10] Generating comparison figure...")

output_dir = joinpath(@__DIR__, "..", "results")
mkpath(output_dir)
output_path = joinpath(output_dir, "bhc_two_material.png")

# Window/level for soft tissue (like the paper figure)
ww_soft = 400    # window width
wl_soft = 0      # window level (center)
clim_soft = (wl_soft - ww_soft/2, wl_soft + ww_soft/2)

# Wider window to see everything
ww_wide = 2000
wl_wide = 0
clim_wide = (wl_wide - ww_wide/2, wl_wide + ww_wide/2)

sl = mid_slice
img_no   = hu_no_bhc[:, :, sl]'      # transpose for image orientation
img_w    = hu_water_bhc[:, :, sl]'
img_2m   = hu_2mat_bhc[:, :, sl]'
img_diff = (hu_2mat_bhc[:, :, sl] .- hu_water_bhc[:, :, sl])'  # bone correction map

fig = Figure(size=(1800, 900), fontsize=14)

# Row 1: Soft tissue window (W:400 L:0) — shows dark-band artifacts
ax1 = Axis(fig[1, 1], title="No BHC\n(W:$(ww_soft) L:$(wl_soft))", aspect=DataAspect(),
           yreversed=true)
hm1 = heatmap!(ax1, img_no, colormap=:grays, colorrange=clim_soft)
hidedecorations!(ax1)

ax2 = Axis(fig[1, 2], title="Water-Only BHC\n(W:$(ww_soft) L:$(wl_soft))", aspect=DataAspect(),
           yreversed=true)
hm2 = heatmap!(ax2, img_w, colormap=:grays, colorrange=clim_soft)
hidedecorations!(ax2)

ax3 = Axis(fig[1, 3], title="Water + Bone BHC\n(W:$(ww_soft) L:$(wl_soft))", aspect=DataAspect(),
           yreversed=true)
hm3 = heatmap!(ax3, img_2m, colormap=:grays, colorrange=clim_soft)
hidedecorations!(ax3)

Colorbar(fig[1, 4], limits=clim_soft, colormap=:grays, label="HU",
         width=15, ticklabelsize=11)

# Row 2: Wide window + bone correction difference map
ax4 = Axis(fig[2, 1], title="No BHC\n(W:$(ww_wide) L:$(wl_wide))", aspect=DataAspect(),
           yreversed=true)
hm4 = heatmap!(ax4, img_no, colormap=:grays, colorrange=clim_wide)
hidedecorations!(ax4)

ax5 = Axis(fig[2, 2], title="Water-Only BHC\n(W:$(ww_wide) L:$(wl_wide))", aspect=DataAspect(),
           yreversed=true)
hm5 = heatmap!(ax5, img_w, colormap=:grays, colorrange=clim_wide)
hidedecorations!(ax5)

ax6 = Axis(fig[2, 3], title="Bone Correction Map\n(2-Mat minus Water-Only)", aspect=DataAspect(),
           yreversed=true)
diff_lim = 50  # ±50 HU
hm6 = heatmap!(ax6, img_diff, colormap=:RdBu, colorrange=(-diff_lim, diff_lim))
hidedecorations!(ax6)

Colorbar(fig[2, 4], limits=(-diff_lim, diff_lim), colormap=:RdBu,
         label="ΔHU (bone correction)", width=15, ticklabelsize=11)

# Title
Label(fig[0, :], "Two-Material Beam Hardening Correction — Gammex 472 at $(KVP) kVp",
      fontsize=18, font=:bold)

# Annotation with metrics
annotation = "Center water ROI:  No BHC = $(round(center_hu_no, digits=1)) HU  |  " *
             "Water BHC = $(round(center_hu_water, digits=1)) HU  |  " *
             "Water+Bone BHC = $(round(center_hu_2mat, digits=1)) HU"
Label(fig[3, :], annotation, fontsize=12)

save(output_path, fig, px_per_unit=2)
println("  Saved: $(output_path)")

println("\n" * "=" ^ 70)
println("  DONE — Two-Material BHC Test Complete")
println("=" ^ 70)
