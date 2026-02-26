#!/usr/bin/env julia
# =============================================================================
# Two-Material (Water + Bone) Beam Hardening Correction — Visual Test
# =============================================================================
#
# Exercises the core library functions calibrate_bhc_two_material() and
# apply_bhc_two_material() on a Gammex 472 phantom and generates a
# comparison PNG showing No BHC vs Water-only BHC vs Water+Bone BHC.
#
# Algorithm: Martinez/Fessler 2022 "2DCalBH" adapted for known spectrum.
#
# Output: ralph_loop/results/bhc_two_material.png
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using BasisSimulator
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

println("\n[Config]")
println("  Phantom: Gammex 472, $(N_VOXELS)×$(N_VOXELS)×$(N_SLICES)")
println("  Geometry: $(N_ANGLES) angles, $(N_ROWS)×$(N_COLS) detector")
println("  Spectrum: $(KVP) kVp, $(N_ENERGY_BINS) energy bins")
println("  Reference energy: $(REFERENCE_ENERGY_KEV) keV")

# =============================================================================
# [2] LOAD SPECTRUM, CREATE PHANTOM AND GEOMETRY
# =============================================================================

println("\n[2] Loading spectrum and creating phantom/geometry...")

energies, weights = load_spectrum(KVP)
energies, weights = downsample_spectrum(energies, weights, N_ENERGY_BINS)

phantom = create_gammex_472(n_voxels=N_VOXELS, n_slices=N_SLICES, fov_cm=FOV_CM, z_cm=Z_CM)
geom = create_aquilion_one(n_angles=N_ANGLES, n_rows=N_ROWS, n_cols=N_COLS,
                           fov_cm=FOV_CM, z_cm=Z_CM)
materials = get_region_materials()
matrix_size = (N_VOXELS, N_VOXELS, N_SLICES)

println("  Phantom extent: $(phantom.extent)")
println("  Geometry FOV:   $(geom.fov)")
println("  Matrix size:    $(matrix_size)")

# =============================================================================
# [3] FORWARD PROJECT (POLYCHROMATIC, NO BHC, NO NOISE)
# =============================================================================

println("\n[3] Running polychromatic forward projection...")

sino_raw = forward_project(phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    volume_extent=phantom.extent)

println("  Sinogram size: $(size(sino_raw))")

# =============================================================================
# [4] CALIBRATE BHC MODELS
# =============================================================================

println("\n[4] Calibrating BHC models...")

# Water-only BHC
bhc_water = calibrate_bhc(energies, weights; order=5, reference_energy_keV=REFERENCE_ENERGY_KEV)

# Two-material (water + bone) BHC
bhc_2mat = calibrate_bhc_two_material(energies, weights;
    order=5, reference_energy_keV=REFERENCE_ENERGY_KEV)

println("  μ_water($(REFERENCE_ENERGY_KEV) keV) = $(round(bhc_2mat.μ_water_ref, digits=5)) cm⁻¹")
println("  μ_bone($(REFERENCE_ENERGY_KEV) keV)  = $(round(bhc_2mat.μ_bone_ref, digits=5)) cm⁻¹")
println("  Bone/water ratio at ref energy: $(round(bhc_2mat.μ_bone_ref/bhc_2mat.μ_water_ref, digits=2))×")

# =============================================================================
# [5] RECONSTRUCT WITH NO BHC
# =============================================================================

println("\n[5] Reconstructing with NO BHC...")

recon_none = Array(fdk_reconstruct(sino_raw, geom, matrix_size))
μ_water_ref = bhc_2mat.μ_water_ref
hu_none = 1000f0 .* (recon_none .- Float32(μ_water_ref)) ./ Float32(μ_water_ref)

mid_slice = N_SLICES ÷ 2
cx, cy = N_VOXELS ÷ 2, N_VOXELS ÷ 2
r_roi = 10

center_hu_no = mean(hu_none[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center water HU (no BHC): $(round(center_hu_no, digits=1)) HU")

# =============================================================================
# [6] WATER-ONLY BHC → RECONSTRUCT
# =============================================================================

println("\n[6] Applying water-only BHC → reconstructing...")

sino_water = apply_bhc(sino_raw, bhc_water)
recon_water = Array(fdk_reconstruct(sino_water, geom, matrix_size))
hu_water = 1000f0 .* (recon_water .- Float32(μ_water_ref)) ./ Float32(μ_water_ref)

center_hu_water = mean(hu_water[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center water HU (water BHC): $(round(center_hu_water, digits=1)) HU")

# =============================================================================
# [7] TWO-MATERIAL BHC → RECONSTRUCT
# =============================================================================

println("\n[7] Applying two-material BHC (water + bone)...")

sino_2mat = apply_bhc_two_material(sino_raw, bhc_2mat, geom, matrix_size;
    volume_extent=phantom.extent)

recon_2mat = Array(fdk_reconstruct(sino_2mat, geom, matrix_size))
hu_2mat = 1000f0 .* (recon_2mat .- Float32(μ_water_ref)) ./ Float32(μ_water_ref)

center_hu_2mat = mean(hu_2mat[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center water HU (2-mat BHC): $(round(center_hu_2mat, digits=1)) HU")

# =============================================================================
# [8] QUANTITATIVE COMPARISON
# =============================================================================

println("\n" * "=" ^ 70)
println("  QUANTITATIVE RESULTS")
println("=" ^ 70)

println("\n--- Center Water ROI ($(2*r_roi+1)×$(2*r_roi+1) pixels) ---")
println("  No BHC:          $(round(center_hu_no, digits=1)) HU  (expect large negative = cupping)")
println("  Water-only BHC:  $(round(center_hu_water, digits=1)) HU  (cupping fixed, bone streaks remain)")
println("  Water+Bone BHC:  $(round(center_hu_2mat, digits=1)) HU  (both fixed)")

println("\n--- Noise / Artifact Metrics ---")
σ_no = std(hu_none[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
σ_water = std(hu_water[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
σ_2mat = std(hu_2mat[cx-r_roi:cx+r_roi, cy-r_roi:cy+r_roi, mid_slice])
println("  Center ROI σ (no BHC):      $(round(σ_no, digits=1)) HU")
println("  Center ROI σ (water BHC):   $(round(σ_water, digits=1)) HU")
println("  Center ROI σ (2-mat BHC):   $(round(σ_2mat, digits=1)) HU")

println("\n--- Global Image Statistics (mid-slice) ---")
for (label, hu) in [("No BHC", hu_none), ("Water BHC", hu_water), ("2-Mat BHC", hu_2mat)]
    sl = hu[:, :, mid_slice]
    body_mask = sl .> -500
    body_vals = sl[body_mask]
    println("  $(rpad(label, 12)): mean=$(round(mean(body_vals), digits=1)) HU, " *
            "std=$(round(std(body_vals), digits=1)) HU, " *
            "range=[$(round(minimum(body_vals), digits=0)), $(round(maximum(body_vals), digits=0))] HU")
end

# =============================================================================
# [9] GENERATE COMPARISON FIGURE
# =============================================================================

println("\n[9] Generating comparison figure...")

output_dir = joinpath(@__DIR__, "..", "results")
mkpath(output_dir)
output_path = joinpath(output_dir, "bhc_two_material.png")

# Window/level for soft tissue
ww_soft = 400
wl_soft = 0
clim_soft = (wl_soft - ww_soft/2, wl_soft + ww_soft/2)

# Wider window
ww_wide = 2000
wl_wide = 0
clim_wide = (wl_wide - ww_wide/2, wl_wide + ww_wide/2)

sl = mid_slice
img_no   = hu_none[:, :, sl]'
img_w    = hu_water[:, :, sl]'
img_2m   = hu_2mat[:, :, sl]'
img_diff = (hu_2mat[:, :, sl] .- hu_water[:, :, sl])'

fig = Figure(size=(1800, 900), fontsize=14)

# Row 1: Soft tissue window (W:400 L:0)
ax1 = Axis(fig[1, 1], title="No BHC\n(W:$(ww_soft) L:$(wl_soft))", aspect=DataAspect(),
           yreversed=true)
heatmap!(ax1, img_no, colormap=:grays, colorrange=clim_soft)
hidedecorations!(ax1)

ax2 = Axis(fig[1, 2], title="Water-Only BHC\n(W:$(ww_soft) L:$(wl_soft))", aspect=DataAspect(),
           yreversed=true)
heatmap!(ax2, img_w, colormap=:grays, colorrange=clim_soft)
hidedecorations!(ax2)

ax3 = Axis(fig[1, 3], title="Water + Bone BHC\n(W:$(ww_soft) L:$(wl_soft))", aspect=DataAspect(),
           yreversed=true)
heatmap!(ax3, img_2m, colormap=:grays, colorrange=clim_soft)
hidedecorations!(ax3)

Colorbar(fig[1, 4], limits=clim_soft, colormap=:grays, label="HU",
         width=15, ticklabelsize=11)

# Row 2: Wide window + bone correction difference map
ax4 = Axis(fig[2, 1], title="No BHC\n(W:$(ww_wide) L:$(wl_wide))", aspect=DataAspect(),
           yreversed=true)
heatmap!(ax4, img_no, colormap=:grays, colorrange=clim_wide)
hidedecorations!(ax4)

ax5 = Axis(fig[2, 2], title="Water-Only BHC\n(W:$(ww_wide) L:$(wl_wide))", aspect=DataAspect(),
           yreversed=true)
heatmap!(ax5, img_w, colormap=:grays, colorrange=clim_wide)
hidedecorations!(ax5)

ax6 = Axis(fig[2, 3], title="Bone Correction Map\n(2-Mat minus Water-Only)", aspect=DataAspect(),
           yreversed=true)
diff_lim = 50
heatmap!(ax6, img_diff, colormap=:RdBu, colorrange=(-diff_lim, diff_lim))
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
