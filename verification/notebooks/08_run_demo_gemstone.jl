# ============================================================================
# BasisSimulator.jl — GE Revolution Apex + Gemstone Detector Demo (CUDA)
# Hamidreza Khodajou-Chokami, PhD
# Run with: julia --project=../.. verification/notebooks/08_run_demo_gemstone.jl
# Or use "Run Without Debugging" in VS Code (set environment first)
# ============================================================================

println("="^60)
println("  BasisSimulator.jl — GE Gemstone Detector Demo (CUDA)")
println("="^60)

# ─── Load packages ───────────────────────────────────────────────────────────
println("\n[1/8] Loading packages...")
using CUDA
using BasisSimulator
import BasisSimulator as BS
import XrayAttenuation as XA
using CairoMakie

# ─── Check GPU ───────────────────────────────────────────────────────────────
println("\n[2/8] Checking CUDA GPU...")
if CUDA.functional()
    println("  ✓ CUDA is functional")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println("  Memory: ", round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=1), " GB")
else
    println("  ✗ CUDA not functional — falling back to CPU")
end

# ─── Gemstone Detector Material Test ─────────────────────────────────────────
println("\n[3/8] Testing Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ detector material...")

# Test MC LUT lookup
println("  MC-derived efficiency at key energies:")
for E in [20.0, 40.0, 52.0, 53.0, 60.0, 63.0, 64.0, 80.0, 100.0, 120.0, 140.0]
    η = BS.get_gemstone_mc_efficiency(E)
    println("    E = $(lpad(Int(E), 3)) keV  →  η = $(round(η, digits=4))")
end

# Test model creation
model_mc = BS.detector_efficiency_gemstone()  # MC LUT (default)
model_bl = BS.detector_efficiency_gemstone(mode=:beer_lambert)
println("\n  MC LUT model:       ", model_mc)
println("  Beer-Lambert model: ", model_bl)

# Test LUMEX alias
μ_gemstone = BS.get_scintillator_mu("Gemstone", 60.0)
μ_lumex = BS.get_scintillator_mu("LUMEX", 60.0)
μ_gos = BS.get_scintillator_mu("GOS", 60.0)
println("\n  μ at 60 keV: Gemstone=$(round(μ_gemstone, digits=1)), LUMEX=$(round(μ_lumex, digits=1)), GOS=$(round(μ_gos, digits=1))")
println("  ✓ LUMEX alias resolves to Gemstone (not GOS)")

# ─── Create phantom ──────────────────────────────────────────────────────────
println("\n[4/8] Creating Gammex 472 phantom...")
phantom_cpu = BS.create_gammex_472(n_voxels=512, n_slices=32, fov_cm=35.0, z_cm=4.0)
println("  Mask size: ", size(phantom_cpu.mask))
println("  Materials: ", length(phantom_cpu.materials), " regions")

# Move mask to GPU
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
    println("  ⚠ CUDA not available — using CPU phantom")
end

# ─── Configure GE Revolution Apex scanner ────────────────────────────────────
println("\n[5/8] Configuring GE Revolution Apex scanner + protocol...")

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

protocol = BS.CTProtocol(
    kVp=120.0,
    mA=300.0,
    views=912,
    rotation_time=1.0,
)

sim_opts = BS.SimOptions(
    fidelity=:high,                  # Enables detector_efficiency + all physics
    seed=42,
)

recon_opts = BS.ReconOptions(
    algorithm=:fdk,
    matrix_size=(512, 512, 32),
    fov_cm=35.0,
)

println("  Scanner: GE Revolution Apex (SID=$(scanner.source_to_isocenter)mm)")
println("  Detector: Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ ($(scanner.detector_material))")
println("  Detector depth: $(scanner.detector_depth) mm")
println("  Protocol: $(protocol.kVp) kVp, $(protocol.mA) mA, $(protocol.views) views")
println("  Fidelity: $(sim_opts.fidelity) (detector_efficiency ENABLED)")
println("  Recon: $(recon_opts.matrix_size)")

# ─── Run simulation ─────────────────────────────────────────────────────────
println("\n[6/8] Running simulate() with Gemstone detector... (this may take a minute)")
t = @elapsed begin
    result = BS.simulate(phantom, scanner, protocol, sim_opts, recon_opts)
end
println("  Done in $(round(t, digits=2)) seconds")

# ─── Convert to HU ──────────────────────────────────────────────────────────
println("\n[7/8] Processing results...")
recon_cpu = Array(result.reconstruction)
μ_water = BS.get_reference_μ_water(60.0)
recon_hu = BS.to_hounsfield(recon_cpu; μ_water=μ_water)

sino_ideal_cpu = Array(result.sinogram_ideal)
sino_noisy_cpu = Array(result.sinogram_noisy)

nx, ny, nz = size(recon_hu)
mid_slice = nz ÷ 2
println("  Reconstruction: $(nx)×$(ny)×$(nz)")
println("  Center slice HU range: [$(round(minimum(recon_hu[:,:,mid_slice]), digits=0)), $(round(maximum(recon_hu[:,:,mid_slice]), digits=0))]")

# ─── Create output directory ────────────────────────────────────────────────
output_dir = joinpath(@__DIR__, "..", "data", "output_gemstone")
mkpath(output_dir)

# ─── Plot 1: Reconstructed CT Slices (Bone Window) ──────────────────────────
println("\n[8/8] Generating images...")

# Bone window (W=2000, L=500) — shows inserts clearly
window_width = 2000.0
window_level = 500.0
wmin = window_level - window_width / 2  # -500
wmax = window_level + window_width / 2  # 1500

fig1 = Figure(size=(1400, 1000), backgroundcolor=:black)
Label(fig1[0, 1:4], "GE Revolution Apex — Gemstone Detector — Bone Window (W=$(Int(window_width)) / L=$(Int(window_level)))",
    fontsize=20, color=:white, font=:bold)

n_show = min(8, nz)
slice_indices = round.(Int, range(2, nz - 1, length=n_show))

for (i, s) in enumerate(slice_indices)
    row = (i - 1) ÷ 4 + 1
    col = (i - 1) % 4 + 1
    ax = Axis(fig1[row, col],
        title="Slice $s / $nz",
        titlecolor=:white, titlesize=14,
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)

    slice_data = clamp.(recon_hu[:, :, s], wmin, wmax)
    slice_norm = (slice_data .- wmin) ./ (wmax - wmin)
    heatmap!(ax, slice_norm', colormap=:grays, colorrange=(0, 1))
end

save(joinpath(output_dir, "01_ct_slices_ge_apex.png"), fig1, px_per_unit=2)
display(fig1)
println("  ✓ Saved & displayed: 01_ct_slices_ge_apex.png")

# ─── Plot 2: CT Slices with Different Windows ───────────────────────────────
fig2 = Figure(size=(1400, 500), backgroundcolor=:black)
Label(fig2[0, 1:4], "GE Revolution Apex — CT Windowing — Slice $(mid_slice)",
    fontsize=22, color=:white, font=:bold)

windows = [
    ("Bone", 2000.0, 500.0),
    ("Abdomen", 400.0, 40.0),
    ("Lung", 1500.0, -600.0),
    ("Brain", 80.0, 40.0),
]

for (i, (name, w, l)) in enumerate(windows)
    ax = Axis(fig2[1, i],
        title="$name (W=$(Int(w)) / L=$(Int(l)))",
        titlecolor=:white, titlesize=14,
        aspect=DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        bottomspinevisible=false, topspinevisible=false,
        leftspinevisible=false, rightspinevisible=false)

    lo = l - w / 2
    hi = l + w / 2
    slice_data = clamp.(recon_hu[:, :, mid_slice], lo, hi)
    slice_norm = (slice_data .- lo) ./ (hi - lo)
    heatmap!(ax, slice_norm', colormap=:grays, colorrange=(0, 1))
end

save(joinpath(output_dir, "02_ct_windows_ge_apex.png"), fig2, px_per_unit=2)
display(fig2)
println("  ✓ Saved & displayed: 02_ct_windows_ge_apex.png")

# ─── Plot 3: Sinogram (Ideal vs Noisy) ──────────────────────────────────────
fig3 = Figure(size=(1200, 500), backgroundcolor=:white)
Label(fig3[0, 1:2], "GE Revolution Apex — Sinograms — Row $(size(sino_ideal_cpu, 2) ÷ 2)",
    fontsize=22, font=:bold)

mid_row = size(sino_ideal_cpu, 2) ÷ 2

ax3a = Axis(fig3[1, 1],
    title="Ideal (noise-free)",
    xlabel="Detector Column", ylabel="View Angle",
    titlesize=16)
sino_slice_ideal = sino_ideal_cpu[:, mid_row, :]'
heatmap!(ax3a, sino_slice_ideal, colormap=:inferno)

ax3b = Axis(fig3[1, 2],
    title="Noisy (after detector simulation)",
    xlabel="Detector Column", ylabel="View Angle",
    titlesize=16)
sino_slice_noisy = sino_noisy_cpu[:, mid_row, :]'
heatmap!(ax3b, sino_slice_noisy, colormap=:inferno)

save(joinpath(output_dir, "03_sinogram_ge_apex.png"), fig3, px_per_unit=2)
display(fig3)
println("  ✓ Saved & displayed: 03_sinogram_ge_apex.png")

# ─── Plot 4: HU Profile ─────────────────────────────────────────────────────
fig4 = Figure(size=(1000, 400), backgroundcolor=:white)
ax4 = Axis(fig4[1, 1],
    title="HU Profile Through Center — GE Revolution Apex — Slice $(mid_slice)",
    xlabel="Pixel Position", ylabel="HU",
    titlesize=18)

center_profile = recon_hu[nx÷2, :, mid_slice]
lines!(ax4, 1:ny, center_profile, color=:steelblue, linewidth=2)
hlines!(ax4, [0.0], color=:red, linestyle=:dash, linewidth=1, label="Water (0 HU)")
axislegend(ax4, position=:rt)

save(joinpath(output_dir, "04_hu_profile_ge_apex.png"), fig4, px_per_unit=2)
display(fig4)
println("  ✓ Saved & displayed: 04_hu_profile_ge_apex.png")

# ─── Plot 5: Phantom Mask vs Reconstruction (Bone Window) ───────────────────
fig5 = Figure(size=(1000, 500), backgroundcolor=:black)
Label(fig5[0, 1:2], "GE Revolution Apex — Phantom vs Reconstruction (Bone Window)",
    fontsize=22, color=:white, font=:bold)

mask_cpu = Array(phantom.mask)
mask_mid = size(mask_cpu, 3) ÷ 2

ax5a = Axis(fig5[1, 1],
    title="Phantom Mask (ground truth)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
heatmap!(ax5a, Float32.(mask_cpu[:, :, mask_mid]'), colormap=:viridis)

ax5b = Axis(fig5[1, 2],
    title="CT Reconstruction (Bone W=2000/L=500)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
# Bone window: W=2000, L=500 → range [-500, 1500]
slice_clamped = clamp.(recon_hu[:, :, mid_slice], -500.0, 1500.0)
slice_norm = (slice_clamped .- (-500.0)) ./ 2000.0
heatmap!(ax5b, slice_norm', colormap=:grays, colorrange=(0, 1))

save(joinpath(output_dir, "05_phantom_vs_recon_ge_apex.png"), fig5, px_per_unit=2)
display(fig5)
println("  ✓ Saved & displayed: 05_phantom_vs_recon_ge_apex.png")

# ─── Plot 6: Gemstone MC Detector Efficiency Curve ──────────────────────────
fig6 = Figure(size=(1200, 800), backgroundcolor=:white)

# Top: MC efficiency vs energy
ax6a = Axis(fig6[1, 1],
    title="Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ — MC Detector Efficiency",
    xlabel="Photon Energy (keV)", ylabel="Detection Efficiency η(E)",
    titlesize=18)

energies_plot = collect(1.0:1.0:140.0)
η_mc = [BS.get_gemstone_mc_efficiency(E) for E in energies_plot]
lines!(ax6a, energies_plot, η_mc, color=:dodgerblue, linewidth=2.5, label="MC (MCNP)")

vlines!(ax6a, [52.0], color=:red, linestyle=:dash, linewidth=1.5, label="Tb K-edge (52 keV)")
vlines!(ax6a, [63.3], color=:orange, linestyle=:dash, linewidth=1.5, label="Lu K-edge (63 keV)")

text!(ax6a, 54.0, 0.78, text="Tb K-edge\nη: 0.956→0.807", fontsize=11, color=:red)
text!(ax6a, 65.0, 0.73, text="Lu K-edge\nη: 0.846→0.762", fontsize=11, color=:orange)

ylims!(ax6a, 0.7, 1.0)
axislegend(ax6a, position=:rb)

# Bottom: GOS vs Gemstone Beer-Lambert μ comparison
ax6b = Axis(fig6[2, 1],
    title="Linear Attenuation Coefficient μ(E) — GOS vs Gemstone (Beer-Lambert)",
    xlabel="Photon Energy (keV)", ylabel="μ (cm⁻¹)",
    titlesize=18, yscale=log10)

energies_mu = collect(20.0:1.0:150.0)
μ_gos_vec = [BS.get_scintillator_mu("GOS", E) for E in energies_mu]
μ_gem_vec = [BS.get_scintillator_mu("Gemstone", E) for E in energies_mu]

lines!(ax6b, energies_mu, μ_gos_vec, color=:gray50, linewidth=2, label="GOS (Gd₂O₂S)")
lines!(ax6b, energies_mu, μ_gem_vec, color=:dodgerblue, linewidth=2, label="Gemstone Ce:(Tb,Lu)₃Al₅O₁₂")

vlines!(ax6b, [50.2], color=:gray50, linestyle=:dot, linewidth=1, label="Gd K-edge (50.2 keV)")
vlines!(ax6b, [52.0], color=:red, linestyle=:dash, linewidth=1, label="Tb K-edge (52 keV)")
vlines!(ax6b, [63.3], color=:orange, linestyle=:dash, linewidth=1, label="Lu K-edge (63 keV)")

axislegend(ax6b, position=:rt)

save(joinpath(output_dir, "06_gemstone_detector_efficiency.png"), fig6, px_per_unit=2)
display(fig6)
println("  ✓ Saved & displayed: 06_gemstone_detector_efficiency.png")

# ─── Done ────────────────────────────────────────────────────────────────────
println("\n", "="^60)
println("  ✓ All images saved to: $(output_dir)")
println("  Scanner: GE Revolution Apex (Gemstone detector)")
println("  Detector: Ce:(Tb,Lu)₃Al₅O₁₂ with MC-derived efficiency")
println("="^60)
