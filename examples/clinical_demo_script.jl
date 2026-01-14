# =============================================================================
# Clinical CT Simulation Demo (Fully GPU-Native)
# =============================================================================
#
# Interactive script for VSCode - step through with Shift+Enter
#
# Demonstrates:
# - Siddon forward projection on Metal GPU
# - FDK reconstruction (entirely on GPU - spatial domain filtering)
# - Polychromatic projection with beam hardening
# - HU validation
#
# All core operations run on Metal GPU via AcceleratedKernels.jl
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# %%
using BasisSimulator
using Statistics
using CairoMakie
using Metal  # GPU acceleration on Apple Silicon

# =============================================================================
# 1. GPU Setup and Configuration
# =============================================================================

# %%
# Check Metal GPU availability
println("Metal GPU Setup:")
println("  Device: ", Metal.current_device())
println("  Metal.jl version: ", pkgversion(Metal))
println()

# %%
# Simulation parameters
CONFIG = (
    n_cols = 256,
    n_rows = 32,
    n_angles = 180,
    n_voxels = 128,
    fov_cm = 35.0,
    z_cm = 4.0,
    n_energy_bins = 30,  # 30 bins ≈ 3 keV resolution for 120 kVp
)

# =============================================================================
# 2. Create Phantom and Geometry
# =============================================================================

# %%
phantom = create_gammex_472(
    n_voxels=CONFIG.n_voxels,
    fov_cm=CONFIG.fov_cm,
    z_cm=CONFIG.z_cm
)
println("Phantom size: ", size(phantom.μ))

# %%
geom = create_aquilion_one(
    n_angles=CONFIG.n_angles,
    n_rows=CONFIG.n_rows,
    n_cols=CONFIG.n_cols,
    fov_cm=CONFIG.fov_cm,
    z_cm=CONFIG.z_cm
)
println("Geometry: $(CONFIG.n_cols) × $(CONFIG.n_rows) × $(CONFIG.n_angles)")
println("FOV: $(geom.fov)")

# =============================================================================
# 3. Visualize Phantom
# =============================================================================

# %%
let
    slice = size(phantom.μ, 3) ÷ 2
    fig = Figure(size=(800, 400))

    ax1 = Axis(fig[1, 1], title="Attenuation (μ)", aspect=DataAspect())
    hm1 = heatmap!(ax1, phantom.μ[:, :, slice], colormap=:grays)
    Colorbar(fig[1, 2], hm1, label="μ (cm⁻¹)")

    ax2 = Axis(fig[1, 3], title="Material Mask", aspect=DataAspect())
    hm2 = heatmap!(ax2, phantom.mask[:, :, slice], colormap=:viridis)
    Colorbar(fig[1, 4], hm2, label="Region ID")

    display(fig)
end

# =============================================================================
# 4. Forward Projection (Metal GPU)
# =============================================================================

# %%
# Convert phantom to GPU array
println("Transferring phantom to Metal GPU...")
phantom_μ_gpu = MtlArray(Float32.(phantom.μ))
println("  GPU array type: ", typeof(phantom_μ_gpu))

# %%
println("Running Siddon forward projection on GPU...")
@time sinogram_gpu = siddon_forward_project(phantom_μ_gpu, geom)
println("Sinogram type: ", typeof(sinogram_gpu))
println("Sinogram size: ", size(sinogram_gpu))

# Second run to show actual performance (without compilation)
println("\nTimed run (after compilation):")
@time sinogram_gpu = siddon_forward_project(phantom_μ_gpu, geom)

sinogram = Array(sinogram_gpu)  # Transfer to CPU for visualization
println("Sinogram range: ", round(minimum(sinogram), digits=3), " to ", round(maximum(sinogram), digits=3))

# %%
let
    row = size(sinogram, 2) ÷ 2
    fig = Figure(size=(900, 400))

    ax1 = Axis(fig[1, 1], title="Sinogram (row $row)",
               xlabel="Detector Column", ylabel="Angle (°)")
    hm = heatmap!(ax1, 1:size(sinogram, 1), range(0, 360, length=size(sinogram, 3)),
                  sinogram[:, row, :]', colormap=:inferno)
    Colorbar(fig[1, 2], hm, label="Line Integral")

    ax2 = Axis(fig[1, 3], title="Central Profile",
               xlabel="Detector Column", ylabel="Line Integral")
    lines!(ax2, sinogram[:, row, size(sinogram,3)÷2], color=:blue, linewidth=2)

    display(fig)
end

# =============================================================================
# 5. FDK Reconstruction (Entirely on GPU)
# =============================================================================
#
# The entire FDK pipeline now runs on GPU:
# - Cosine weighting: GPU (AcceleratedKernels.jl)
# - Ramp filtering: GPU (spatial domain convolution)
# - Backprojection: GPU (AcceleratedKernels.jl)
#
# No CPU transfer needed!
#

# %%
println("Running full FDK reconstruction on GPU...")
@time recon_gpu = fdk_reconstruct(sinogram_gpu, geom, size(phantom.μ))
println("Reconstruction type: ", typeof(recon_gpu))

# Second run to show actual performance
println("\nTimed run (after compilation):")
@time recon_gpu = fdk_reconstruct(sinogram_gpu, geom, size(phantom.μ))

recon = Array(recon_gpu)  # Transfer to CPU for visualization
println("Reconstruction size: ", size(recon))

# %%
μ_water = get_reference_μ_water(60.0)
recon_hu = @. 1000 * (recon - μ_water) / μ_water
phantom_hu = @. 1000 * (phantom.μ - μ_water) / μ_water

# %%
let
    slice = size(recon, 3) ÷ 2
    fig = Figure(size=(900, 400))

    ax1 = Axis(fig[1, 1], title="Ground Truth", aspect=DataAspect())
    heatmap!(ax1, phantom_hu[:, :, slice], colormap=:grays, colorrange=(-200, 1500))

    ax2 = Axis(fig[1, 2], title="FDK Reconstruction", aspect=DataAspect())
    hm = heatmap!(ax2, recon_hu[:, :, slice], colormap=:grays, colorrange=(-200, 1500))
    Colorbar(fig[1, 3], hm, label="HU")

    ax3 = Axis(fig[1, 4], title="Difference", aspect=DataAspect())
    diff = recon_hu[:, :, slice] .- phantom_hu[:, :, slice]
    hm_diff = heatmap!(ax3, diff, colormap=:RdBu, colorrange=(-300, 300))
    Colorbar(fig[1, 5], hm_diff, label="ΔHU")

    display(fig)
end

# =============================================================================
# 6. HU Validation
# =============================================================================

# %%
function measure_hu(vol, mask, region_id, μ_water)
    m = mask .== UInt8(region_id)
    sum(m) < 100 && return (mean=NaN, std=NaN)
    vals = vol[m]
    hu = @. 1000 * (vals - μ_water) / μ_water
    (mean=mean(hu), std=std(hu))
end

regions = [
    ("Solid Water", REGION_SOLID_WATER, 0),
    ("Ca 50", REGION_CA_50, 180),
    ("Ca 100", REGION_CA_100, 375),
    ("Ca 200", REGION_CA_200, 750),
    ("Ca 300", REGION_CA_300, 1100),
    ("Ca 400", REGION_CA_400, 1500),
]

println("\nHU Measurements:")
println("-"^50)
for (name, id, expected) in regions
    stats = measure_hu(recon, phantom.mask, id, μ_water)
    println("  $name: $(round(stats.mean, digits=1)) ± $(round(stats.std, digits=1)) HU (expected: $expected)")
end

# %%
let
    hu_data = [(name, measure_hu(recon, phantom.mask, id, μ_water), expected)
               for (name, id, expected) in regions]

    fig = Figure(size=(700, 400))
    ax = Axis(fig[1, 1], title="HU Accuracy", xlabel="Material", ylabel="HU Value")

    names = [d[1] for d in hu_data]
    measured = [d[2].mean for d in hu_data]
    expected = [d[3] for d in hu_data]
    stds = [d[2].std for d in hu_data]

    x = 1:length(names)
    barplot!(ax, x .- 0.2, expected, width=0.35, label="Expected", color=:steelblue)
    barplot!(ax, x .+ 0.2, measured, width=0.35, label="Measured", color=:coral)
    errorbars!(ax, x .+ 0.2, measured, stds, color=:black, whiskerwidth=8)

    ax.xticks = (x, names)
    ax.xticklabelrotation = π/6
    axislegend(ax, position=:lt)

    display(fig)
end

# =============================================================================
# 7. Polychromatic Forward Projection
# =============================================================================
#
# Polychromatic uses CPU arrays because material lookup tables are CPU-based.
# Each internal Siddon projection still benefits from AcceleratedKernels.jl.
#
# Memory-efficient approach using the unified forward_project API:
# - Loops over energy bins internally
# - Accumulates Beer-Lambert: I = Σ w_e × exp(-∫μ_e dl)
# - Converts to line integral: -log(I / I_0)
#

# %%
energies_full, weights_full = load_spectrum(120)
energies, weights = downsample_spectrum(energies_full, weights_full, CONFIG.n_energy_bins)
materials = get_region_materials()

energy_range = round(maximum(energies) - minimum(energies), digits=1)
bin_width = round(energy_range / CONFIG.n_energy_bins, digits=1)
println("Spectrum: $(length(energies_full)) bins → $(CONFIG.n_energy_bins) bins")
println("  Energy range: $(round(minimum(energies), digits=1)) - $(round(maximum(energies), digits=1)) keV")
println("  Effective bin width: ~$(bin_width) keV")
println("  Materials: $(length(materials))")

# %%
println("\nRunning polychromatic projection ($(CONFIG.n_energy_bins) energies)...")
@time sino_poly = forward_project(
    phantom.mask, geom;
    energies=energies,
    weights=weights,
    materials=materials
)
println("Poly sinogram range: ", round(minimum(sino_poly), digits=3), " to ", round(maximum(sino_poly), digits=3))

# =============================================================================
# 8. Mono vs Poly Comparison (Beam Hardening)
# =============================================================================

# %%
# Also run monochromatic at 60 keV for comparison
println("Running monochromatic projection (60 keV)...")
@time sino_mono = forward_project(
    phantom.mask, geom;
    energy=60.0,
    materials=materials
)

# %%
let
    row = size(sinogram, 2) ÷ 2
    fig = Figure(size=(900, 400))

    ax1 = Axis(fig[1, 1], title="Monochromatic (60 keV)")
    heatmap!(ax1, sino_mono[:, row, :]', colormap=:inferno)

    ax2 = Axis(fig[1, 2], title="Polychromatic (120 kVp)")
    hm = heatmap!(ax2, sino_poly[:, row, :]', colormap=:inferno,
                  colorrange=extrema(sino_mono[:, row, :]))
    Colorbar(fig[1, 3], hm, label="Line Integral")

    ax3 = Axis(fig[1, 4], title="Profile Comparison",
               xlabel="Detector Column", ylabel="Line Integral")
    angle_mid = size(sinogram, 3) ÷ 2
    lines!(ax3, sino_mono[:, row, angle_mid], label="Mono", linewidth=2)
    lines!(ax3, sino_poly[:, row, angle_mid], label="Poly", linewidth=2)
    axislegend(ax3, position=:rb)

    display(fig)
end

# %%
diff = sino_poly .- sino_mono
println("\nBeam Hardening Effect (Sinograms):")
println("  Mean difference: ", round(mean(diff), digits=4))
println("  Max difference: ", round(maximum(abs.(diff)), digits=4))
println("  Poly max < Mono max: $(maximum(sino_poly) < maximum(sino_mono)) (expected for beam hardening)")

# =============================================================================
# 9. Mono vs Poly Reconstruction Comparison (GPU)
# =============================================================================

# %%
println("\nReconstructing monochromatic sinogram (GPU)...")
sino_mono_gpu = MtlArray(Float32.(sino_mono))
@time recon_mono_gpu = fdk_reconstruct(sino_mono_gpu, geom, size(phantom.μ))
recon_mono = Array(recon_mono_gpu)
println("Mono recon range: ", round(minimum(recon_mono), digits=4), " to ", round(maximum(recon_mono), digits=4))

# %%
println("Reconstructing polychromatic sinogram (GPU)...")
sino_poly_gpu = MtlArray(Float32.(sino_poly))
@time recon_poly_gpu = fdk_reconstruct(sino_poly_gpu, geom, size(phantom.μ))
recon_poly = Array(recon_poly_gpu)
println("Poly recon range: ", round(minimum(recon_poly), digits=4), " to ", round(maximum(recon_poly), digits=4))

# %%
# Convert to HU for comparison
recon_mono_hu = @. 1000 * (recon_mono - μ_water) / μ_water
recon_poly_hu = @. 1000 * (recon_poly - μ_water) / μ_water

# %%
let
    slice = size(recon_mono, 3) ÷ 2
    fig = Figure(size=(1100, 400))

    ax1 = Axis(fig[1, 1], title="Ground Truth", aspect=DataAspect())
    heatmap!(ax1, phantom_hu[:, :, slice], colormap=:grays, colorrange=(-200, 1500))

    ax2 = Axis(fig[1, 2], title="Mono Recon (60 keV)", aspect=DataAspect())
    heatmap!(ax2, recon_mono_hu[:, :, slice], colormap=:grays, colorrange=(-200, 1500))

    ax3 = Axis(fig[1, 3], title="Poly Recon (120 kVp)", aspect=DataAspect())
    hm = heatmap!(ax3, recon_poly_hu[:, :, slice], colormap=:grays, colorrange=(-200, 1500))
    Colorbar(fig[1, 4], hm, label="HU")

    ax4 = Axis(fig[1, 5], title="Poly - Mono", aspect=DataAspect())
    diff_recon = recon_poly_hu[:, :, slice] .- recon_mono_hu[:, :, slice]
    hm_diff = heatmap!(ax4, diff_recon, colormap=:RdBu, colorrange=(-200, 200))
    Colorbar(fig[1, 6], hm_diff, label="ΔHU")

    display(fig)
end

# %%
# Beam hardening cupping artifact analysis
println("\nBeam Hardening in Reconstruction:")
slice = size(recon_mono, 3) ÷ 2
center = size(recon_mono, 1) ÷ 2

# Profile through center
mono_profile = recon_mono_hu[center, :, slice]
poly_profile = recon_poly_hu[center, :, slice]

println("  Center profile (mono): min=$(round(minimum(mono_profile), digits=1)), max=$(round(maximum(mono_profile), digits=1))")
println("  Center profile (poly): min=$(round(minimum(poly_profile), digits=1)), max=$(round(maximum(poly_profile), digits=1))")
println("  Cupping visible in poly: center values lower than edges (beam hardening artifact)")

# %%
let
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], title="Center Profile: Mono vs Poly",
              xlabel="Position", ylabel="HU")

    lines!(ax, mono_profile, label="Mono (60 keV)", linewidth=2)
    lines!(ax, poly_profile, label="Poly (120 kVp)", linewidth=2, linestyle=:dash)
    axislegend(ax, position=:rb)

    display(fig)
end

# =============================================================================
# 10. Performance Summary
# =============================================================================

# %%
println("\n" * "="^60)
println("Performance Summary (Metal GPU)")
println("="^60)
println("\nForward Projection:")
@time siddon_forward_project(phantom_μ_gpu, geom)

println("\nFull FDK Reconstruction:")
@time fdk_reconstruct(sinogram_gpu, geom, size(phantom.μ))

println("\n" * "="^60)
println("Clinical Demo Complete!")
println("="^60)
println("\nAll core operations run on Metal GPU:")
println("  ✓ Forward projection (Siddon)")
println("  ✓ Cosine weighting")
println("  ✓ Ramp filtering (spatial domain)")
println("  ✓ Backprojection (FDK)")
