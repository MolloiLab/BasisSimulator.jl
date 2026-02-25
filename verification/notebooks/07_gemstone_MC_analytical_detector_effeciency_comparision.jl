# ============================================================================
# 07 — GE Revolution Apex: Gemstone Detector Model Verification
#      Monte Carlo LUT vs Analytical Beer-Lambert (XCOM) Comparison
# ============================================================================
# BasisSimulator.jl Verification Notebook
# Hamidreza Khodajou-Chokami, PhD
#
# Purpose: Compare CT images produced by the Gemstone detector using
#          (A) MC-derived efficiency LUT (MCNP)  vs
#          (B) Analytical Beer-Lambert efficiency from NIST XCOM μ(E)
#
# Run with: julia --project=../.. verification/notebooks/07_gemstone_MC_analytical_detector_effeciency_comparision.jl
# ============================================================================

println("="^70)
println("  Verification 07: Gemstone Detector — MC LUT vs Analytical (XCOM)")
println("="^70)

# ─── Load Packages ───────────────────────────────────────────────────────────
println("\n[1/10] Loading packages...")
using CUDA
using BasisSimulator
import BasisSimulator as BS
import XrayAttenuation as XA
using CairoMakie
using Statistics

# ─── Check GPU ───────────────────────────────────────────────────────────────
println("\n[2/10] Checking CUDA GPU...")
if CUDA.functional()
    println("  ✓ CUDA is functional")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println("  Memory: ", round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=1), " GB")
else
    println("  ✗ CUDA not functional — falling back to CPU")
end

# ─── Verify Gemstone Material Data ───────────────────────────────────────────
println("\n[3/10] Verifying Gemstone material data...")

# MC LUT: check K-edge efficiency drops
println("  MC-LUT efficiency at K-edges:")
for (E, note) in [(52.0, "before Tb K-edge"), (53.0, "after Tb K-edge"),
    (63.0, "before Lu K-edge"), (64.0, "after Lu K-edge")]
    η = BS.get_gemstone_mc_efficiency(E)
    println("    E = $(Int(E)) keV → η = $(round(η, digits=4))  ($note)")
end

# XCOM Beer-Lambert: check K-edge μ jumps
println("\n  XCOM μ(E) at K-edges (from NIST XCOM, ρ=7.0 g/cm³):")
for E in [51.0, 52.0, 53.0, 60.0, 63.0, 64.0, 80.0, 100.0]
    μ = BS.get_scintillator_mu("Gemstone", E)
    println("    E = $(Int(E)) keV → μ = $(round(μ, digits=1)) cm⁻¹")
end

# Verify LUMEX alias
@assert BS.get_scintillator_mu("LUMEX", 60.0) == BS.get_scintillator_mu("Gemstone", 60.0)
println("\n  ✓ LUMEX alias → Gemstone (verified)")
println("  ✓ Material data verified")

# ─── Create Phantom ─────────────────────────────────────────────────────────
println("\n[4/10] Creating Gammex 472 phantom...")
phantom_cpu = BS.create_gammex_472(n_voxels=512, n_slices=32, fov_cm=35.0, z_cm=4.0)
println("  Mask size: ", size(phantom_cpu.mask))
println("  Materials: ", length(phantom_cpu.materials), " regions")

if CUDA.functional()
    println("  Moving phantom mask to GPU...")
    phantom_gpu_mask = CuArray(phantom_cpu.mask)
    phantom = BS.Phantom(
        phantom_gpu_mask,
        phantom_cpu.materials,
        phantom_cpu.voxel_size,
        phantom_cpu.origin,
        phantom_cpu.extent,
    )
    println("  ✓ Phantom on GPU")
else
    phantom = phantom_cpu
    println("  ⚠ Using CPU phantom")
end

# ─── Common Scanner & Protocol ──────────────────────────────────────────────
println("\n[5/10] Configuring GE Revolution Apex scanner...")

# GE Revolution Apex geometry with Gemstone detector
scanner = BS.Scanner(
    source_to_isocenter=626.0,     # GE Revolution Apex SID (mm)
    source_to_detector=1097.0,     # GE Revolution Apex SDD (mm)
    detector_rows=64,              # Subset of 256 rows for demo speed
    detector_cols=832,             # Full detector width
    detector_row_size=0.625,       # 0.625 mm native row size
    detector_col_size=0.625,       # At isocenter
    detector_depth=3.0,            # 3.0 mm garnet scintillator
    detector_material=:lumex,      # GE Gemstone Ce:(Tb,Lu)₃Al₅O₁₂
    fill_factor_row=0.90,
    fill_factor_col=0.90,
    target_angle=10.0,             # Quantix 160 tube
    focal_spot_width=1.0,
    focal_spot_length=0.7,
    flat_filter_material=:aluminum,
    flat_filter_thickness=7.0,     # mm Al inherent filtration
)

protocol = BS.CTProtocol(kVp=120.0, mA=300.0, views=912, rotation_time=1.0)
recon_opts = BS.ReconOptions(algorithm=:fdk, matrix_size=(512, 512, 32), fov_cm=35.0)

println("  Scanner: GE Revolution Apex (SID=$(scanner.source_to_isocenter)mm)")
println("  Detector: Gemstone Ce:(Tb,Lu)₃Al₅O₁₂, 3.0 mm")
println("  Protocol: $(protocol.kVp) kVp, $(protocol.mA) mA, $(protocol.views) views")

# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION A: Monte Carlo LUT Mode
# ═════════════════════════════════════════════════════════════════════════════
println("\n[6/10] Simulation A: Monte Carlo LUT mode...")

sim_opts_mc = BS.SimOptions(
    fidelity=:high,
    seed=42,
    detector_efficiency_mode=:mc_lut,
)

t_mc = @elapsed begin
    result_mc = BS.simulate(phantom, scanner, protocol, sim_opts_mc, recon_opts)
end
println("  ✓ MC simulation done in $(round(t_mc, digits=2))s")

# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION B: Analytical Beer-Lambert (XCOM) Mode
# ═════════════════════════════════════════════════════════════════════════════
println("\n[7/10] Simulation B: Analytical Beer-Lambert (XCOM) mode...")

sim_opts_bl = BS.SimOptions(
    fidelity=:high,
    seed=42,
    detector_efficiency_mode=:beer_lambert,
)

t_bl = @elapsed begin
    result_bl = BS.simulate(phantom, scanner, protocol, sim_opts_bl, recon_opts)
end
println("  ✓ Beer-Lambert simulation done in $(round(t_bl, digits=2))s")

# ─── Process Results ─────────────────────────────────────────────────────────
println("\n[8/10] Processing results...")

μ_water = BS.get_reference_μ_water(60.0)

recon_mc_cpu = Array(result_mc.reconstruction)
recon_bl_cpu = Array(result_bl.reconstruction)
hu_mc = BS.to_hounsfield(recon_mc_cpu; μ_water=μ_water)
hu_bl = BS.to_hounsfield(recon_bl_cpu; μ_water=μ_water)

nx, ny, nz = size(hu_mc)
mid_slice = nz ÷ 2

println("  MC  — center HU range: [$(round(minimum(hu_mc[:,:,mid_slice]), digits=0)), $(round(maximum(hu_mc[:,:,mid_slice]), digits=0))]")
println("  B-L — center HU range: [$(round(minimum(hu_bl[:,:,mid_slice]), digits=0)), $(round(maximum(hu_bl[:,:,mid_slice]), digits=0))]")

# Difference map
hu_diff = hu_mc .- hu_bl
println("  Difference (MC−BL): mean=$(round(mean(hu_diff[:,:,mid_slice]), digits=1)) HU, max|ΔHU|=$(round(maximum(abs.(hu_diff[:,:,mid_slice])), digits=1))")

# ─── Output Directory ───────────────────────────────────────────────────────
output_dir = joinpath(@__DIR__, "..", "data", "07_gemstone_output")
mkpath(output_dir)

# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════
println("\n[9/10] Generating comparison plots...")

# ─── Plot 1: Efficiency Curves — MC vs Beer-Lambert Side-by-Side ────────────
fig1 = Figure(size=(1200, 500), backgroundcolor=:white)

# Panel A: MC efficiency
ax1a = Axis(fig1[1, 1],
    title="MC-Derived Efficiency (MCNP)",
    xlabel="Energy (keV)", ylabel="η(E)",
    titlesize=16)

E_plot = collect(1.0:0.5:140.0)
η_mc_curve = [BS.get_gemstone_mc_efficiency(E) for E in E_plot]
lines!(ax1a, E_plot, η_mc_curve, color=:dodgerblue, linewidth=2)
vlines!(ax1a, [52.0], color=:red, linestyle=:dash, linewidth=1, label="Tb K-edge")
vlines!(ax1a, [63.3], color=:orange, linestyle=:dash, linewidth=1, label="Lu K-edge")
ylims!(ax1a, 0.7, 1.0)
axislegend(ax1a, position=:rb)

# Panel B: Beer-Lambert efficiency (η = 1 - exp(-μd))
ax1b = Axis(fig1[1, 2],
    title="Analytical Beer-Lambert (XCOM μ, d=3mm)",
    xlabel="Energy (keV)", ylabel="η(E)",
    titlesize=16)

E_bl = collect(10.0:0.5:150.0)
d_cm = 0.3  # 3 mm = 0.3 cm
η_bl_curve = [1.0 - exp(-BS.get_scintillator_mu("Gemstone", E) * d_cm) for E in E_bl]
lines!(ax1b, E_bl, η_bl_curve, color=:crimson, linewidth=2)
vlines!(ax1b, [52.0], color=:red, linestyle=:dash, linewidth=1, label="Tb K-edge")
vlines!(ax1b, [63.3], color=:orange, linestyle=:dash, linewidth=1, label="Lu K-edge")
ylims!(ax1b, 0.7, 1.0)
axislegend(ax1b, position=:rb)

save(joinpath(output_dir, "01_efficiency_mc_vs_analytical.png"), fig1, px_per_unit=2)
display(fig1)
println("  ✓ 01_efficiency_mc_vs_analytical.png")

# ─── Plot 2: Overlay — MC vs Analytical on Same Axes ────────────────────────
fig2 = Figure(size=(1000, 500), backgroundcolor=:white)
ax2 = Axis(fig2[1, 1],
    title="Gemstone Detector Efficiency: MC LUT vs Analytical (XCOM)",
    xlabel="Energy (keV)", ylabel="Detection Efficiency η(E)",
    titlesize=18)

lines!(ax2, E_plot, η_mc_curve, color=:dodgerblue, linewidth=2.5, label="MC LUT (MCNP)")
lines!(ax2, E_bl, η_bl_curve, color=:crimson, linewidth=2.5, linestyle=:dash, label="Beer-Lambert (XCOM)")
vlines!(ax2, [52.0], color=:gray60, linestyle=:dot, linewidth=1, label="Tb K-edge (52 keV)")
vlines!(ax2, [63.3], color=:gray60, linestyle=:dashdot, linewidth=1, label="Lu K-edge (63 keV)")
ylims!(ax2, 0.7, 1.0)
axislegend(ax2, position=:rb)

save(joinpath(output_dir, "02_efficiency_overlay.png"), fig2, px_per_unit=2)
display(fig2)
println("  ✓ 02_efficiency_overlay.png")

# ─── Plot 3: CT Reconstruction — MC vs Analytical vs Difference ─────────────
fig3 = Figure(size=(1400, 500), backgroundcolor=:black)
Label(fig3[0, 1:3], "GE Revolution Apex — MC LUT vs Analytical (Bone W/L=2000/500) — Slice $(mid_slice)",
    fontsize=18, color=:white, font=:bold)

wmin, wmax = -500.0, 1500.0  # Bone window: W=2000, L=500

# MC
ax3a = Axis(fig3[1, 1], title="MC LUT (MCNP)",
    titlecolor=:white, titlesize=14, aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
s_mc = clamp.(hu_mc[:, :, mid_slice], wmin, wmax)
heatmap!(ax3a, ((s_mc .- wmin) ./ (wmax - wmin))', colormap=:grays, colorrange=(0, 1))

# Beer-Lambert
ax3b = Axis(fig3[1, 2], title="Beer-Lambert (XCOM)",
    titlecolor=:white, titlesize=14, aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
s_bl = clamp.(hu_bl[:, :, mid_slice], wmin, wmax)
heatmap!(ax3b, ((s_bl .- wmin) ./ (wmax - wmin))', colormap=:grays, colorrange=(0, 1))

# Difference (MC - BL)
ax3c = Axis(fig3[1, 3], title="Difference (MC − Analytical)",
    titlecolor=:white, titlesize=14, aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
diff_slice = hu_diff[:, :, mid_slice]
hm3 = heatmap!(ax3c, diff_slice', colormap=:RdBu, colorrange=(-50, 50))
Colorbar(fig3[1, 4], hm3, label="ΔHU", labelcolor=:white, ticklabelcolor=:white)

save(joinpath(output_dir, "03_ct_mc_vs_analytical.png"), fig3, px_per_unit=2)
display(fig3)
println("  ✓ 03_ct_mc_vs_analytical.png")

# ─── Plot 4: HU Profiles — MC vs Analytical ─────────────────────────────────
fig4 = Figure(size=(1200, 500), backgroundcolor=:white)

ax4a = Axis(fig4[1, 1],
    title="HU Profile Through Center — Slice $(mid_slice)",
    xlabel="Pixel Position", ylabel="HU",
    titlesize=16)

profile_mc = hu_mc[nx÷2, :, mid_slice]
profile_bl = hu_bl[nx÷2, :, mid_slice]
lines!(ax4a, 1:ny, profile_mc, color=:dodgerblue, linewidth=2, label="MC LUT")
lines!(ax4a, 1:ny, profile_bl, color=:crimson, linewidth=2, linestyle=:dash, label="Beer-Lambert")
hlines!(ax4a, [0.0], color=:gray50, linestyle=:dot, linewidth=1, label="Water (0 HU)")
axislegend(ax4a, position=:rt)

ax4b = Axis(fig4[1, 2],
    title="Profile Difference (MC − Analytical)",
    xlabel="Pixel Position", ylabel="ΔHU",
    titlesize=16)

profile_diff = profile_mc .- profile_bl
lines!(ax4b, 1:ny, profile_diff, color=:purple, linewidth=2)
hlines!(ax4b, [0.0], color=:gray50, linestyle=:dot, linewidth=1)

save(joinpath(output_dir, "04_hu_profile_comparison.png"), fig4, px_per_unit=2)
display(fig4)
println("  ✓ 04_hu_profile_comparison.png")

# ─── Plot 5: Windowing Comparison (Bone & Abdomen) ──────────────────────────
fig5 = Figure(size=(1200, 1000), backgroundcolor=:black)
Label(fig5[0, 1:2], "MC LUT vs Analytical — Multiple Windows — Slice $(mid_slice)",
    fontsize=18, color=:white, font=:bold)

windows = [("Bone (W=2000/L=500)", 2000.0, 500.0), ("Abdomen (W=400/L=40)", 400.0, 40.0)]

for (row, (name, w, l)) in enumerate(windows)
    lo = l - w / 2
    hi = l + w / 2

    ax_mc = Axis(fig5[row, 1], title="MC LUT — $name",
        titlecolor=:white, titlesize=13, aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)
    s = clamp.(hu_mc[:, :, mid_slice], lo, hi)
    heatmap!(ax_mc, ((s .- lo) ./ (hi - lo))', colormap=:grays, colorrange=(0, 1))

    ax_bl = Axis(fig5[row, 2], title="Beer-Lambert — $name",
        titlecolor=:white, titlesize=13, aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)
    s = clamp.(hu_bl[:, :, mid_slice], lo, hi)
    heatmap!(ax_bl, ((s .- lo) ./ (hi - lo))', colormap=:grays, colorrange=(0, 1))
end

save(joinpath(output_dir, "05_windowing_mc_vs_analytical.png"), fig5, px_per_unit=2)
display(fig5)
println("  ✓ 05_windowing_mc_vs_analytical.png")

# ─── Plot 6: GOS vs Gemstone μ(E) ───────────────────────────────────────────
fig6 = Figure(size=(1000, 500), backgroundcolor=:white)
ax6 = Axis(fig6[1, 1],
    title="Linear Attenuation μ(E): GOS vs Gemstone (XCOM)",
    xlabel="Energy (keV)", ylabel="μ (cm⁻¹)",
    titlesize=18, yscale=log10)

E_mu = collect(10.0:1.0:150.0)
μ_gos = [BS.get_scintillator_mu("GOS", E) for E in E_mu]
μ_gem = [BS.get_scintillator_mu("Gemstone", E) for E in E_mu]

lines!(ax6, E_mu, μ_gos, color=:gray50, linewidth=2, label="GOS (Gd₂O₂S)")
lines!(ax6, E_mu, μ_gem, color=:dodgerblue, linewidth=2, label="Gemstone Ce:(Tb,Lu)₃Al₅O₁₂")
vlines!(ax6, [50.2], color=:gray50, linestyle=:dot, linewidth=1, label="Gd K-edge (50.2)")
vlines!(ax6, [52.0], color=:red, linestyle=:dash, linewidth=1, label="Tb K-edge (52)")
vlines!(ax6, [63.3], color=:orange, linestyle=:dash, linewidth=1, label="Lu K-edge (63)")
axislegend(ax6, position=:rt)

save(joinpath(output_dir, "06_mu_gos_vs_gemstone.png"), fig6, px_per_unit=2)
display(fig6)
println("  ✓ 06_mu_gos_vs_gemstone.png")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
println("\n[10/10] Verification Summary")
println("="^70)
println("  Simulation A (MC LUT):        $(round(t_mc, digits=1))s")
println("  Simulation B (Beer-Lambert):  $(round(t_bl, digits=1))s")
println("")
println("  MC LUT center HU:  [$(round(minimum(hu_mc[:,:,mid_slice]),digits=0)), $(round(maximum(hu_mc[:,:,mid_slice]),digits=0))]")
println("  B-L   center HU:   [$(round(minimum(hu_bl[:,:,mid_slice]),digits=0)), $(round(maximum(hu_bl[:,:,mid_slice]),digits=0))]")
println("  ΔHU (MC−BL):       mean=$(round(mean(hu_diff[:,:,mid_slice]),digits=1)), max|ΔHU|=$(round(maximum(abs.(hu_diff[:,:,mid_slice])),digits=1))")
println("")
println("  ✓ 6 comparison plots saved to: $(output_dir)")
println("="^70)
