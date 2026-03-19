# ============================================================================
# 06 — MC Detector Response Verification
# ============================================================================
# Author: Hamidreza Khodajou-Chokami, PhD.
#
# This notebook verifies the three MC-based PCCT additions:
#   1. MC detector response matrix (replaces analytical DRM)
#   2. MC pulse pileup (replaces analytical Taguchi model)
#   3. Cumulative threshold pipeline (T1/T4 sinograms)
#
# Runs 3 cases on a Gammex 472 phantom and compares CT image quality:
#   Case 1: Baseline — Analytical DRM + Analytical Pileup
#   Case 2: +MC DRM — MC response matrix (Stierstorfer 2018)
#   Case 3: +MC Pileup — MC DRM + MC seminonparalyzable pileup
#
# Outputs: 6 PNG comparison images + detailed text report
#
# Run with: julia --project=. verification/notebooks/06_mc_detector_response.jl
# ============================================================================

println("="^70)
println("  06 — MC Detector Response Verification")
println("="^70)

# Helper
mean_val(x) = sum(x) / length(x)

# ─── Load packages ───────────────────────────────────────────────────────────
println("\n[1/9] Loading packages...")
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BasisSimulator
import BasisSimulator as BS
import XrayAttenuation as XA
using CairoMakie
using Statistics

# ─── Configuration ───────────────────────────────────────────────────────────
println("\n[2/9] Configuration...")

# Phantom and protocol (use moderate size for verification)
const N_VOXELS = 256
const N_SLICES = 16
const KVP = 120.0
const MA = 300.0
const VIEWS = 512
const ROTATION_TIME = 1.0

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "..", "data", "mc_verification")
mkpath(OUTPUT_DIR)

println("  Phantom: $(N_VOXELS)³ × $(N_SLICES) slices")
println("  Protocol: $(KVP) kVp, $(MA) mA, $(VIEWS) views")
println("  Output: $(OUTPUT_DIR)")

# ─── Create phantom ──────────────────────────────────────────────────────────
println("\n[3/9] Creating phantom...")
phantom = BS.create_gammex_472(n_voxels=N_VOXELS, n_slices=N_SLICES, fov_cm=35.0, z_cm=2.0)
println("  Mask size: ", size(phantom.mask))

# ─── Set up common components ───────────────────────────────────────────────
println("\n[4/9] Setting up components...")
detector = BS.naeotom_detector_standard()
energies, weights = BS.load_spectrum(Int(KVP))
energies_ds, weights_ds = BS.downsample_spectrum(energies, weights, 30)
materials = BS.get_region_materials()

geom = BS.CTGeometry(
    BS.Scanner();
    n_angles=VIEWS,
    fov_cm=35.0,
    z_cm=nothing
)

protocol = BS.CTProtocol(kVp=KVP, mA=MA, views=VIEWS, rotation_time=ROTATION_TIME)
I0 = BS.compute_detector_I0(geom, protocol)
I0_bins = BS.compute_per_bin_I0(detector, energies_ds, weights_ds, I0)
recon_size = (N_VOXELS, N_VOXELS, N_SLICES)

println("  I₀ = $(round(I0, sigdigits=4)) photons/pixel/view")
println("  Thresholds: ", detector.energy_thresholds_keV, " keV")
println("  MC DRM path: ", BS.default_mc_drm_path())

# ─── Helper: Run one PCCT configuration ──────────────────────────────────────
function run_pcct_config(label, phantom, geom, detector, energies, weights,
                         materials, I0, I0_bins, recon_size;
                         use_mc_drm=false, use_mc_pileup=false)
    println("\n  ▶ $(label)...")
    t = @elapsed begin
        pcct_sino = BS.pcct_forward_project(
            phantom.mask, geom, detector;
            energies=energies, weights=weights,
            materials=materials,
            I0=I0,
            flux_rate=nothing,  # Auto from I₀
            apply_spectral_response=true,
            apply_detector_effects=true,
            apply_corrections=false,
            use_mc_drm=use_mc_drm,       # uses default bundled path
            use_mc_pileup=use_mc_pileup,
            mc_pileup_trials=3000,
        )
    end
    println("    Forward projection: $(round(t, digits=1))s")

    # Cumulative T1/T4
    low_sino, high_sino = BS.cumulative_threshold_sinograms(
        pcct_sino.bins, I0_bins;
        low_bins=1:4, high_bins=4:4
    )

    # Reconstruct
    low_recon = Array(BS.fdk_reconstruct(low_sino, geom, recon_size))
    high_recon = Array(BS.fdk_reconstruct(high_sino, geom, recon_size))

    # HU conversion
    μ_water_low = BS.get_reference_μ_water(35.0)
    μ_water_high = BS.get_reference_μ_water(85.0)
    low_hu = BS.to_hounsfield(low_recon; μ_water=μ_water_low)
    high_hu = BS.to_hounsfield(high_recon; μ_water=μ_water_high)

    # Per-bin stats
    bin_stats = [(round(minimum(Array(b)), digits=3),
                  round(maximum(Array(b)), digits=3),
                  round(mean_val(Array(b)), digits=3)) for b in pcct_sino.bins]

    mid = size(low_hu, 3) ÷ 2
    println("    Low  HU: [$(round(minimum(low_hu[:,:,mid]), digits=0)), $(round(maximum(low_hu[:,:,mid]), digits=0))]")
    println("    High HU: [$(round(minimum(high_hu[:,:,mid]), digits=0)), $(round(maximum(high_hu[:,:,mid]), digits=0))]")

    return (low_hu=low_hu, high_hu=high_hu, bin_stats=bin_stats, time=t, label=label)
end

# ═══════════════════════════════════════════════════════════════════════════
# Case 1: Baseline (Analytical DRM + Analytical Pileup)
# ═══════════════════════════════════════════════════════════════════════════
println("\n[5/9] Case 1: Baseline (Analytical DRM + Analytical Pileup)")
case1 = run_pcct_config("Baseline (Analytical)",
    phantom, geom, detector, energies_ds, weights_ds,
    materials, I0, I0_bins, recon_size;
    use_mc_drm=false, use_mc_pileup=false)

# ═══════════════════════════════════════════════════════════════════════════
# Case 2: +MC DRM (MC response + Analytical Pileup)
# ═══════════════════════════════════════════════════════════════════════════
println("\n[6/9] Case 2: +MC DRM (MC Response Matrix)")
case2 = run_pcct_config("+MC DRM (Stierstorfer)",
    phantom, geom, detector, energies_ds, weights_ds,
    materials, I0, I0_bins, recon_size;
    use_mc_drm=true, use_mc_pileup=false)

# ═══════════════════════════════════════════════════════════════════════════
# Case 3: +MC Pileup (MC DRM + MC Pulse Pileup)
# ═══════════════════════════════════════════════════════════════════════════
println("\n[7/9] Case 3: +MC Pileup (MC DRM + MC Pulse Pileup)")
case3 = run_pcct_config("+MC Pileup (full model)",
    phantom, geom, detector, energies_ds, weights_ds,
    materials, I0, I0_bins, recon_size;
    use_mc_drm=true, use_mc_pileup=true)

# ═══════════════════════════════════════════════════════════════════════════
# [8/9] Generate Comparison Images
# ═══════════════════════════════════════════════════════════════════════════
println("\n[8/9] Generating comparison images...")

cases = [case1, case2, case3]
mid_slice = N_SLICES ÷ 2

# --- Figure 1: Low-Energy CT (T1 ≥ 20 keV) ---
fig1 = Figure(size=(1600, 600), backgroundcolor=:black)
Label(fig1[0, 1:3], "Low-Energy CT (T1 ≥ 20 keV) — Abdomen Window W:400 / L:40",
    fontsize=22, color=:white, font=:bold)

wmin, wmax = -160.0, 240.0

for (i, c) in enumerate(cases)
    ax = Axis(fig1[1, i],
        title=c.label,
        titlecolor=:white, titlesize=14,
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)

    slice_data = clamp.(c.low_hu[:, :, mid_slice], wmin, wmax)
    slice_norm = (slice_data .- wmin) ./ (wmax - wmin)
    heatmap!(ax, slice_norm', colormap=:grays, colorrange=(0, 1))
end

save(joinpath(OUTPUT_DIR, "comparison_low_energy_abdomen.png"), fig1, px_per_unit=2)
println("  ✓ Saved: comparison_low_energy_abdomen.png")

# --- Figure 2: High-Energy CT (T4 ≥ 70 keV) ---
fig2 = Figure(size=(1600, 600), backgroundcolor=:black)
Label(fig2[0, 1:3], "High-Energy CT (T4 ≥ 70 keV) — Abdomen Window W:400 / L:40",
    fontsize=22, color=:white, font=:bold)

for (i, c) in enumerate(cases)
    ax = Axis(fig2[1, i],
        title=c.label,
        titlecolor=:white, titlesize=14,
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)

    slice_data = clamp.(c.high_hu[:, :, mid_slice], wmin, wmax)
    slice_norm = (slice_data .- wmin) ./ (wmax - wmin)
    heatmap!(ax, slice_norm', colormap=:grays, colorrange=(0, 1))
end

save(joinpath(OUTPUT_DIR, "comparison_high_energy_abdomen.png"), fig2, px_per_unit=2)
println("  ✓ Saved: comparison_high_energy_abdomen.png")

# --- Figure 3: HU Profile Comparison ---
fig4 = Figure(size=(1200, 800), backgroundcolor=:white)

ax4a = Axis(fig4[1, 1],
    title="HU Profile — Low-Energy CT (T1 ≥ 20 keV)",
    xlabel="Pixel Position", ylabel="HU",
    titlesize=16)

ny = size(case1.low_hu, 2)
cx = N_VOXELS ÷ 2
for (c, color, ls) in zip(cases, [:blue, :red, :green], [:solid, :dash, :dashdot])
    profile = c.low_hu[cx, :, mid_slice]
    lines!(ax4a, 1:ny, profile, color=color, linewidth=2, linestyle=ls, label=c.label)
end
hlines!(ax4a, [0.0], color=:gray, linestyle=:dot, linewidth=1, label="Water (0 HU)")
axislegend(ax4a, position=:rt, labelsize=11)

ax4b = Axis(fig4[2, 1],
    title="HU Profile — High-Energy CT (T4 ≥ 70 keV)",
    xlabel="Pixel Position", ylabel="HU",
    titlesize=16)

for (c, color, ls) in zip(cases, [:blue, :red, :green], [:solid, :dash, :dashdot])
    profile = c.high_hu[cx, :, mid_slice]
    lines!(ax4b, 1:ny, profile, color=color, linewidth=2, linestyle=ls, label=c.label)
end
hlines!(ax4b, [0.0], color=:gray, linestyle=:dot, linewidth=1, label="Water (0 HU)")
axislegend(ax4b, position=:rt, labelsize=11)

save(joinpath(OUTPUT_DIR, "comparison_hu_profiles.png"), fig4, px_per_unit=2)
println("  ✓ Saved: comparison_hu_profiles.png")

# --- Figure 4: Difference Maps ---
fig5 = Figure(size=(1600, 600), backgroundcolor=:black)
Label(fig5[0, 1:3], "Difference Maps — Low-Energy CT (ΔHU)",
    fontsize=22, color=:white, font=:bold)

ax5a = Axis(fig5[1, 1],
    title="Baseline (reference)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
slice_norm = (clamp.(case1.low_hu[:, :, mid_slice], wmin, wmax) .- wmin) ./ (wmax - wmin)
heatmap!(ax5a, slice_norm', colormap=:grays, colorrange=(0, 1))

diff_drm = case2.low_hu[:, :, mid_slice] .- case1.low_hu[:, :, mid_slice]
ax5b = Axis(fig5[1, 2],
    title="MC DRM Effect (ΔHU)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
heatmap!(ax5b, diff_drm', colormap=:RdBu, colorrange=(-200, 200))
Colorbar(fig5[2, 2], limits=(-200, 200), colormap=:RdBu,
    vertical=false, label="ΔHU", labelcolor=:white, ticklabelcolor=:white,
    labelsize=12, ticklabelsize=10, width=Relative(0.8))

diff_pileup = case3.low_hu[:, :, mid_slice] .- case2.low_hu[:, :, mid_slice]
ax5c = Axis(fig5[1, 3],
    title="MC Pileup Effect (ΔHU)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
heatmap!(ax5c, diff_pileup', colormap=:RdBu, colorrange=(-200, 200))
Colorbar(fig5[2, 3], limits=(-200, 200), colormap=:RdBu,
    vertical=false, label="ΔHU", labelcolor=:white, ticklabelcolor=:white,
    labelsize=12, ticklabelsize=10, width=Relative(0.8))

save(joinpath(OUTPUT_DIR, "comparison_difference_maps.png"), fig5, px_per_unit=2)
println("  ✓ Saved: comparison_difference_maps.png")

# ═══════════════════════════════════════════════════════════════════════════
# [9/9] Generate Text Report
# ═══════════════════════════════════════════════════════════════════════════
println("\n[9/9] Generating report...")

report = """
╔══════════════════════════════════════════════════════════════════════════╗
║              MC DETECTOR RESPONSE VERIFICATION REPORT                  ║
║    Author: Hamidreza Khodajou-Chokami, PhD.                            ║
╚══════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
 Configuration
═══════════════════════════════════════════════════════════════════════════
  Phantom:         Gammex 472, $(N_VOXELS)³ × $(N_SLICES) slices
  Protocol:        $(KVP) kVp, $(MA) mA, $(VIEWS) views
  I₀:              $(round(I0, sigdigits=4)) photons/pixel/view
  Thresholds:      $(detector.energy_thresholds_keV) keV
  MC DRM:          $(BS.default_mc_drm_path())

═══════════════════════════════════════════════════════════════════════════
 Simulation Cases
═══════════════════════════════════════════════════════════════════════════

  Case 1: BASELINE
    DRM:    Analytical (Gaussian charge sharing model)
    Pileup: Analytical (Taguchi 2010 nonparalyzable)
    Time:   $(round(case1.time, digits=1))s

  Case 2: +MC DETECTOR RESPONSE
    DRM:    Monte Carlo (Stierstorfer 2018 model)
            - Fano noise, charge cloud transport, pixel splitting
            - σ = 30µm charge cloud, 1.5 keV electronic noise
    Pileup: Analytical (Taguchi 2010)
    Time:   $(round(case2.time, digits=1))s

  Case 3: +MC PULSE PILEUP (FULL MODEL)
    DRM:    Monte Carlo (same as Case 2)
    Pileup: Monte Carlo seminonparalyzable
            - Poisson arrival times from spectrum
            - Dead-time gating with retrigger probability
            - Spectral migration matrix
            - flux_rate auto-coupled to mAs via I₀
    Time:   $(round(case3.time, digits=1))s

═══════════════════════════════════════════════════════════════════════════
 CT Image Quality Comparison (Slice $(mid_slice))
═══════════════════════════════════════════════════════════════════════════

  LOW-ENERGY CT (T1 ≥ 20 keV):
                           Min HU      Max HU
    Case 1 (Baseline):     $(round(minimum(case1.low_hu[:,:,mid_slice]), digits=0))       $(round(maximum(case1.low_hu[:,:,mid_slice]), digits=0))
    Case 2 (+MC DRM):      $(round(minimum(case2.low_hu[:,:,mid_slice]), digits=0))       $(round(maximum(case2.low_hu[:,:,mid_slice]), digits=0))
    Case 3 (+MC Pileup):   $(round(minimum(case3.low_hu[:,:,mid_slice]), digits=0))       $(round(maximum(case3.low_hu[:,:,mid_slice]), digits=0))

  HIGH-ENERGY CT (T4 ≥ 70 keV):
                           Min HU      Max HU
    Case 1 (Baseline):     $(round(minimum(case1.high_hu[:,:,mid_slice]), digits=0))       $(round(maximum(case1.high_hu[:,:,mid_slice]), digits=0))
    Case 2 (+MC DRM):      $(round(minimum(case2.high_hu[:,:,mid_slice]), digits=0))       $(round(maximum(case2.high_hu[:,:,mid_slice]), digits=0))
    Case 3 (+MC Pileup):   $(round(minimum(case3.high_hu[:,:,mid_slice]), digits=0))       $(round(maximum(case3.high_hu[:,:,mid_slice]), digits=0))

═══════════════════════════════════════════════════════════════════════════
 MC DRM Effect (Case 2 - Case 1)
═══════════════════════════════════════════════════════════════════════════
  The MC detector response matrix replaces the simplified analytical
  Gaussian model with full Monte Carlo simulation of CdTe physics:
  - Fano noise in charge generation
  - Charge cloud transport and diffusion (σ=30µm)
  - Sub-pixel edge splitting (3×3 grid)
  - Electronic noise (1.5 keV Gaussian)

  ΔHU (Low-E):  mean=$(round(mean(diff_drm), digits=1)), std=$(round(std(diff_drm), digits=1)), max|ΔHU|=$(round(maximum(abs.(diff_drm)), digits=1))

═══════════════════════════════════════════════════════════════════════════
 MC Pileup Effect (Case 3 - Case 2)
═══════════════════════════════════════════════════════════════════════════
  Monte Carlo pulse pileup replaces the analytical Taguchi model with:
  - Poisson photon arrival times
  - Seminonparalyzable dead-time with retrigger
  - Spectral migration (energy summing of overlapping pulses)
  - Count-rate dependent (coupled to mAs via I₀)

  ΔHU (Low-E):  mean=$(round(mean(diff_pileup), digits=1)), std=$(round(std(diff_pileup), digits=1)), max|ΔHU|=$(round(maximum(abs.(diff_pileup)), digits=1))

═══════════════════════════════════════════════════════════════════════════
 Output Files
═══════════════════════════════════════════════════════════════════════════
  $(OUTPUT_DIR)/comparison_low_energy_abdomen.png
  $(OUTPUT_DIR)/comparison_high_energy_abdomen.png
  $(OUTPUT_DIR)/comparison_hu_profiles.png
  $(OUTPUT_DIR)/comparison_difference_maps.png
  $(OUTPUT_DIR)/comparison_report.txt
"""

open(joinpath(OUTPUT_DIR, "comparison_report.txt"), "w") do f
    write(f, report)
end
println("  ✓ Saved: comparison_report.txt")
println(report)

println("\n" * "="^70)
println("  ✓ All verification results saved to: $(OUTPUT_DIR)")
println("="^70)
