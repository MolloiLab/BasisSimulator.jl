# scripts/visualize.jl
# Run manually: julia --project scripts/visualize.jl
#
# Generates PNG visualizations for inspection.
# Requires CairoMakie: ] add CairoMakie

using BasisSimulator
using Statistics
using CairoMakie

output_dir = joinpath(@__DIR__, "..", "test", "outputs")
mkpath(output_dir)

println("Generating visualizations...")

# Create phantom and geometry
println("  Creating phantom...")
phantom = create_gammex_472(n_voxels=64)
geom = create_aquilion_one(n_angles=360, n_rows=16, n_cols=256, fov_cm=phantom.fov[1])

# Reference values
μ_water = get_reference_μ_water(60.0)
mid_slice = size(phantom.μ, 3) ÷ 2

# Helper function
function save_slice(filename, data; title="", colormap=:grays, vmin=-1000, vmax=1000)
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)
    hm = heatmap!(ax, data'; colormap=colormap, colorrange=(vmin, vmax))
    Colorbar(fig[1, 2], hm, label="HU")
    hidedecorations!(ax, label=false, ticklabels=false, ticks=false)
    save(joinpath(output_dir, filename), fig)
    println("    Saved: $filename")
end

# 1. Phantom ground truth
println("  Phantom...")
phantom_HU = μ_to_HU(phantom.μ, μ_water)
save_slice("01_phantom.png", phantom_HU[:, :, mid_slice]; title="Phantom (Ground Truth)")

# 2. Monochromatic forward + recon
println("  Monochromatic simulation...")
sino_mono = forward_project(phantom, geom)
recon_mono = fdk_reconstruct(sino_mono, geom, size(phantom.μ), phantom.fov)
recon_mono_HU = μ_to_HU(recon_mono, μ_water)
save_slice("02_recon_mono.png", recon_mono_HU[:, :, mid_slice]; title="Monochromatic Recon")

# 3. Sinogram
fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], xlabel="Detector", ylabel="Angle", title="Sinogram")
heatmap!(ax, sino_mono[:, size(sino_mono, 2)÷2, :]'; colormap=:hot)
save(joinpath(output_dir, "03_sinogram.png"), fig)
println("    Saved: 03_sinogram.png")

# 4. Polychromatic simulation
println("  Polychromatic simulation...")
projector = create_polychromatic_projector(phantom, geom, 120; n_bins=20)
sino_poly = forward_project_polychromatic(phantom, projector)
recon_poly = fdk_reconstruct(sino_poly, geom, size(phantom.μ), phantom.fov)
recon_poly_HU = μ_to_HU(recon_poly, get_effective_μ_water(projector))
save_slice("04_recon_poly.png", recon_poly_HU[:, :, mid_slice]; title="Polychromatic (Beam Hardening)")

# 5. Scatter (XCIST-style model)
println("  Scatter simulation...")
scatter_model = default_scatter_model(scale_factor=1.0)  # ~15% SPR
sino_scatter = add_scatter(sino_mono, scatter_model)
recon_scatter = fdk_reconstruct(sino_scatter, geom, size(phantom.μ), phantom.fov)
recon_scatter_HU = μ_to_HU(recon_scatter, μ_water)
save_slice("05_recon_scatter.png", recon_scatter_HU[:, :, mid_slice]; title="With Scatter (Cupping)")

# 6. Noisy
println("  Noise simulation...")
detector_model = default_detector_model(blur_fwhm=1.0, I0=5e4, electronic_noise_std=10.0, seed=42)
sino_noisy = apply_detector_model(sino_mono, detector_model)
recon_noisy = fdk_reconstruct(sino_noisy, geom, size(phantom.μ), phantom.fov)
recon_noisy_HU = μ_to_HU(recon_noisy, μ_water)
save_slice("06_recon_noisy.png", recon_noisy_HU[:, :, mid_slice]; title="With Noise")

# 7. Full realistic
println("  Full realistic simulation...")
sino_full = forward_project_polychromatic(phantom, projector)
sino_full = add_scatter(sino_full, scatter_model)
sino_full = apply_detector_model(sino_full, detector_model)
recon_full = fdk_reconstruct(sino_full, geom, size(phantom.μ), phantom.fov)
recon_full_HU = μ_to_HU(recon_full, get_effective_μ_water(projector))
save_slice("07_recon_full.png", recon_full_HU[:, :, mid_slice]; title="Full Realistic")

# 8. Comparison figure
println("  Comparison figure...")
fig = Figure(size=(1200, 800))
titles = ["Ground Truth", "Mono Recon", "Polychromatic", "With Scatter", "With Noise", "Full Realistic"]
data = [phantom_HU, recon_mono_HU, recon_poly_HU, recon_scatter_HU, recon_noisy_HU, recon_full_HU]
for (i, (t, d)) in enumerate(zip(titles, data))
    row, col = (i-1) ÷ 3 + 1, (i-1) % 3 + 1
    ax = Axis(fig[row, col], aspect=DataAspect(), title=t)
    heatmap!(ax, d[:, :, mid_slice]'; colormap=:grays, colorrange=(-1000, 1000))
    hidedecorations!(ax)
end
Colorbar(fig[1:2, 4], limits=(-1000, 1000), colormap=:grays, label="HU")
save(joinpath(output_dir, "08_comparison.png"), fig)
println("    Saved: 08_comparison.png")

println("\nDone! Outputs in: $output_dir")
