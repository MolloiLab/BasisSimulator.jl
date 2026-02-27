# ============================================================================
# BasisSimulator.jl — Generic Scanner CUDA Demo with Visualization
# Hamidreza Khodajou-Chokami, PhD
# Run with: julia --project=../.. verification/notebooks/06_run_demo_generic_scanner.jl
# Or use "Run Without Debugging" in VS Code (set environment first)
# ============================================================================

println("="^60)
println("  BasisSimulator.jl — Generic Scanner CUDA Demo")
println("="^60)

# ─── Load packages ───────────────────────────────────────────────────────────
println("\n[1/7] Loading packages...")
using CUDA
using BasisSimulator
import BasisSimulator as BS
import XrayAttenuation as XA
using CairoMakie

# ─── Check GPU ───────────────────────────────────────────────────────────────
println("\n[2/7] Checking CUDA GPU...")
if CUDA.functional()
    println("  ✓ CUDA is functional")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println("  Memory: ", round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=1), " GB")
else
    println("  ✗ CUDA not functional — falling back to CPU")
end

# ─── Create phantom ──────────────────────────────────────────────────────────
# Step 1: Create phantom on CPU (geometry + materials)
println("\n[3/7] Creating Gammex 472 phantom...")
phantom_cpu = BS.create_gammex_472(n_voxels=512, n_slices=32, fov_cm=35.0, z_cm=4.0)
println("  Mask size: ", size(phantom_cpu.mask))
println("  Materials: ", length(phantom_cpu.materials), " regions")

# Step 2: Move mask to GPU, keep materials + metadata on CPU
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
    println("  ✓ Phantom mask on GPU ($(typeof(phantom.mask)))")
else
    phantom = phantom_cpu
    println("  ⚠ CUDA not available — using CPU phantom")
end

# ─── Configure simulation ───────────────────────────────────────────────────
println("\n[4/7] Configuring scanner + protocol...")

scanner = BS.Scanner()  # Default generic scanner

protocol = BS.CTProtocol(
    kVp=120.0,
    mA=300.0,
    views=912,           # More views for better image quality
    rotation_time=1.0,
)

sim_opts = BS.SimOptions(
    fidelity=:medium,    # Noise + focal spot + crosstalk + flat filter + BHC
    seed=42,
)

recon_opts = BS.ReconOptions(
    algorithm=:fdk,
    matrix_size=(512, 512, 32),  # Good resolution
    fov_cm=35.0,           # Full clinical FOV — matches phantom
)

println("  Scanner: generic ($(scanner.source_to_isocenter)mm SID)")
println("  Protocol: $(protocol.kVp) kVp, $(protocol.mA) mA, $(protocol.views) views")
println("  Fidelity: $(sim_opts.fidelity)")
println("  Recon: $(recon_opts.matrix_size)")

# ─── Run simulation ─────────────────────────────────────────────────────────
println("\n[5/7] Running simulate()... (this may take a minute)")
t = @elapsed begin
    result = BS.simulate(phantom, scanner, protocol, sim_opts, recon_opts)
end
println("  Done in $(round(t, digits=2)) seconds")

# ─── Convert to HU ──────────────────────────────────────────────────────────
println("\n[6/7] Processing results...")
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
output_dir = joinpath(@__DIR__, "..", "data", "output_generic_scanner")
mkpath(output_dir)

# ─── Plot 1: Reconstructed CT Slices (Abdomen Window) ───────────────────────
println("\n[7/7] Generating images...")

# Clinical windowing: Abdomen window (W=400, L=40)
window_width = 400.0
window_level = 40.0
wmin = window_level - window_width / 2  # -160
wmax = window_level + window_width / 2  # 240

fig1 = Figure(size=(1400, 1000), backgroundcolor=:black)
Label(fig1[0, 1:4], "CT Reconstruction — Abdomen Window (W=$(Int(window_width)) / L=$(Int(window_level)))",
    fontsize=22, color=:white, font=:bold)

# Show 8 evenly-spaced slices
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
    # Normalize to 0-1 for grayscale display
    slice_norm = (slice_data .- wmin) ./ (wmax - wmin)
    heatmap!(ax, slice_norm', colormap=:grays, colorrange=(0, 1))
end

save(joinpath(output_dir, "01_ct_slices_abdomen.png"), fig1, px_per_unit=2)
display(fig1)
println("  ✓ Saved & displayed: 01_ct_slices_abdomen.png")

# ─── Plot 2: CT Slices with Different Windows ───────────────────────────────
fig2 = Figure(size=(1400, 500), backgroundcolor=:black)
Label(fig2[0, 1:4], "CT Windowing Comparison — Slice $(mid_slice)",
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

save(joinpath(output_dir, "02_ct_windows_comparison.png"), fig2, px_per_unit=2)
display(fig2)
println("  ✓ Saved & displayed: 02_ct_windows_comparison.png")

# ─── Plot 3: Sinogram (Ideal vs Noisy) ──────────────────────────────────────
fig3 = Figure(size=(1200, 500), backgroundcolor=:white)
Label(fig3[0, 1:2], "Sinograms — Row $(size(sino_ideal_cpu, 2) ÷ 2)",
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

save(joinpath(output_dir, "03_sinogram_comparison.png"), fig3, px_per_unit=2)
display(fig3)
println("  ✓ Saved & displayed: 03_sinogram_comparison.png")

# ─── Plot 4: HU Profile through Center ──────────────────────────────────────
fig4 = Figure(size=(1000, 400), backgroundcolor=:white)
ax4 = Axis(fig4[1, 1],
    title="HU Profile Through Center of Slice $(mid_slice)",
    xlabel="Pixel Position", ylabel="HU",
    titlesize=18)

center_profile = recon_hu[nx÷2, :, mid_slice]
lines!(ax4, 1:ny, center_profile, color=:steelblue, linewidth=2)
hlines!(ax4, [0.0], color=:red, linestyle=:dash, linewidth=1, label="Water (0 HU)")
axislegend(ax4, position=:rt)

save(joinpath(output_dir, "04_hu_profile.png"), fig4, px_per_unit=2)
display(fig4)
println("  ✓ Saved & displayed: 04_hu_profile.png")

# ─── Plot 5: Phantom Mask vs Reconstruction Side-by-Side ────────────────────
fig5 = Figure(size=(1000, 500), backgroundcolor=:black)
Label(fig5[0, 1:2], "Phantom Mask vs CT Reconstruction",
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
    title="CT Reconstruction (Abdomen Window)",
    titlecolor=:white, titlesize=14,
    aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    bottomspinevisible=false, topspinevisible=false,
    leftspinevisible=false, rightspinevisible=false)
slice_clamped = clamp.(recon_hu[:, :, mid_slice], -160.0, 240.0)
slice_norm = (slice_clamped .- (-160.0)) ./ 400.0
heatmap!(ax5b, slice_norm', colormap=:grays, colorrange=(0, 1))

save(joinpath(output_dir, "05_phantom_vs_recon.png"), fig5, px_per_unit=2)
display(fig5)
println("  ✓ Saved & displayed: 05_phantom_vs_recon.png")

# ─── Done ────────────────────────────────────────────────────────────────────
println("\n", "="^60)
println("  ✓ All images saved to: $(output_dir)")
println("="^60)
