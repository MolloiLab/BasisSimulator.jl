# stuff/scripts/visualize.jl
# Run manually: julia --project stuff/scripts/visualize.jl
#
# Generates PNG visualizations for inspection.
# Requires CairoMakie: ] add CairoMakie

using BasisSimulator
using Statistics
using CairoMakie

output_dir = joinpath(@__DIR__, "outputs")
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

# Helper for sinogram display
function save_sinogram(filename, sino; title="")
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Detector", ylabel="Angle", title=title)
    mid_row = size(sino, 2) ÷ 2
    heatmap!(ax, sino[:, mid_row, :]'; colormap=:hot)
    save(joinpath(output_dir, filename), fig)
    println("    Saved: $filename")
end

# =============================================================================
# Basic Simulations
# =============================================================================

# 1. Phantom ground truth (categorical regions)
println("  Phantom...")
phantom_mask_slice = Float64.(phantom.mask[:, :, mid_slice])
fig = Figure(size=(650, 500))
ax = Axis(fig[1, 1], aspect=DataAspect(), title="Phantom (Region Labels)")
hm = heatmap!(ax, phantom_mask_slice'; colormap=:tab20, colorrange=(0, 26))
Colorbar(fig[1, 2], hm, label="Region ID")
hidedecorations!(ax, label=false, ticklabels=false, ticks=false)
save(joinpath(output_dir, "01_phantom.png"), fig)
println("    Saved: 01_phantom.png")

# 2. Monochromatic forward + recon
println("  Monochromatic simulation...")
sino_mono = forward_project(phantom, geom)
recon_mono = fdk_reconstruct(sino_mono, geom, size(phantom.μ), phantom.fov)
recon_mono_HU = μ_to_HU(recon_mono, μ_water)
save_slice("02_recon_mono.png", recon_mono_HU[:, :, mid_slice]; title="Monochromatic Recon")

# 3. Sinogram
save_sinogram("03_sinogram.png", sino_mono; title="Sinogram (Monochromatic)")

# 4. Polychromatic simulation
println("  Polychromatic simulation...")
projector = create_polychromatic_projector(phantom, geom, 120; n_bins=20)
sino_poly = forward_project_polychromatic(phantom, projector)
recon_poly = fdk_reconstruct(sino_poly, geom, size(phantom.μ), phantom.fov)
recon_poly_HU = μ_to_HU(recon_poly, get_effective_μ_water(projector))
save_slice("04_recon_poly.png", recon_poly_HU[:, :, mid_slice]; title="Polychromatic (Beam Hardening)")

# 5. Scatter (XCIST-style model)
println("  Scatter simulation...")
scatter_model = default_scatter_model(scale_factor=1.0)
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

# 7. Full realistic (original)
println("  Full realistic simulation...")
sino_full = forward_project_polychromatic(phantom, projector)
sino_full = add_scatter(sino_full, scatter_model)
sino_full = apply_detector_model(sino_full, detector_model)
recon_full = fdk_reconstruct(sino_full, geom, size(phantom.μ), phantom.fov)
recon_full_HU = μ_to_HU(recon_full, get_effective_μ_water(projector))
save_slice("07_recon_full.png", recon_full_HU[:, :, mid_slice]; title="Full Realistic")

# =============================================================================
# NEW: Bowtie Filter
# =============================================================================
println("  Bowtie filter...")
bowtie = bowtie_filter_large_body()
sino_bowtie = apply_bowtie_filter(sino_mono, bowtie, geom)

# Show bowtie attenuation profile
fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], xlabel="Detector Angle (degrees)", ylabel="Attenuation",
          title="Bowtie Filter Profile (Large Body)")
angles_deg = rad2deg.(bowtie.angles)
lines!(ax, angles_deg, bowtie.thickness .* bowtie.μ_ref, linewidth=2, color=:blue)
save(joinpath(output_dir, "08_bowtie_profile.png"), fig)
println("    Saved: 08_bowtie_profile.png")

# Show effect on sinogram
save_sinogram("09_sinogram_bowtie.png", sino_bowtie; title="Sinogram with Bowtie Filter")

# =============================================================================
# NEW: Focal Spot Blur
# =============================================================================
println("  Focal spot blur...")
# Use a larger focal spot for visible effect
focal_spot = FocalSpot(2.0, 2.0, :gaussian, 3)  # 2mm focal spot
sino_focal = apply_focal_spot_blur(sino_mono, focal_spot, geom)

# Show comparison
fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel="Detector", ylabel="Attenuation", title="Without Focal Spot Blur")
ax2 = Axis(fig[1, 2], xlabel="Detector", ylabel="Attenuation", title="With 2mm Focal Spot")
mid_row = size(sino_mono, 2) ÷ 2
mid_angle = size(sino_mono, 3) ÷ 2
lines!(ax1, sino_mono[:, mid_row, mid_angle], linewidth=1.5)
lines!(ax2, sino_focal[:, mid_row, mid_angle], linewidth=1.5)
save(joinpath(output_dir, "10_focal_spot_effect.png"), fig)
println("    Saved: 10_focal_spot_effect.png")

# =============================================================================
# NEW: Detector Crosstalk
# =============================================================================
println("  Detector crosstalk...")
crosstalk = crosstalk_high()  # 15% crosstalk for visible effect
sino_crosstalk = apply_crosstalk(sino_mono, crosstalk)

# Show crosstalk effect
fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel="Detector", ylabel="Attenuation", title="Without Crosstalk")
ax2 = Axis(fig[1, 2], xlabel="Detector", ylabel="Attenuation", title="With 15% Crosstalk")
lines!(ax1, sino_mono[:, mid_row, mid_angle], linewidth=1.5, color=:blue)
lines!(ax2, sino_crosstalk[:, mid_row, mid_angle], linewidth=1.5, color=:red)
save(joinpath(output_dir, "11_crosstalk_effect.png"), fig)
println("    Saved: 11_crosstalk_effect.png")

# Reconstruct with crosstalk
recon_crosstalk = fdk_reconstruct(sino_crosstalk, geom, size(phantom.μ), phantom.fov)
recon_crosstalk_HU = μ_to_HU(recon_crosstalk, μ_water)
save_slice("12_recon_crosstalk.png", recon_crosstalk_HU[:, :, mid_slice]; title="With Detector Crosstalk")

# =============================================================================
# NEW: Detector Lag
# =============================================================================
println("  Detector lag...")
lag_model = lag_high()  # High lag for visible effect
sino_lag = apply_lag(sino_mono, lag_model)

# Show impulse response
fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], xlabel="Frame Number", ylabel="Response",
          title="Detector Lag Impulse Response (High Lag)")
ir = compute_lag_impulse_response(lag_model, 50)
lines!(ax, 0:49, ir, linewidth=2, color=:purple)
scatter!(ax, 0:49, ir, markersize=6, color=:purple)
save(joinpath(output_dir, "13_lag_impulse_response.png"), fig)
println("    Saved: 13_lag_impulse_response.png")

# Compare sinograms angle-by-angle
fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel="Detector", ylabel="Attenuation", title="Without Lag")
ax2 = Axis(fig[1, 2], xlabel="Detector", ylabel="Attenuation", title="With Detector Lag")
lines!(ax1, sino_mono[:, mid_row, mid_angle], linewidth=1.5, color=:blue)
lines!(ax2, sino_lag[:, mid_row, mid_angle], linewidth=1.5, color=:orange)
save(joinpath(output_dir, "14_lag_effect.png"), fig)
println("    Saved: 14_lag_effect.png")

# =============================================================================
# NEW: Helical Scanning
# =============================================================================
println("  Helical scanning...")

# Create helical geometry
geom_helical = create_scan_geometry(
    mode=:helical,
    n_angles=180,
    n_rows=16,
    n_cols=256,
    fov_cm=phantom.fov[1],
    pitch=1.0,
    n_rotations=3.0,
    z_start=-3.0
)

# Forward project with helical geometry
sino_helical = forward_project(phantom, geom_helical)

# Show Z-position vs angle
params = get_helical_parameters(geom_helical)
z_positions = [geom_helical.source_positions[3, i] for i in 1:geom_helical.n_angles]
fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], xlabel="Projection Index", ylabel="Table Z Position (cm)",
          title="Helical Scan: Z Position vs Projection (Pitch=$(params.pitch))")
lines!(ax, 1:length(z_positions), z_positions, linewidth=2, color=:green)
save(joinpath(output_dir, "15_helical_trajectory.png"), fig)
println("    Saved: 15_helical_trajectory.png")

# Show helical sinogram (has more angles)
save_sinogram("16_sinogram_helical.png", sino_helical; title="Helical Sinogram (3 rotations)")

# Interpolate to axial at middle Z
z_mid = (params.z_start + params.z_end) / 2
sino_interp = interpolate_helical_to_axial(sino_helical, geom_helical, z_mid)
save_sinogram("17_sinogram_helical_interp.png", sino_interp;
              title="Interpolated to Axial at Z=$(round(z_mid, digits=1))cm")

# =============================================================================
# NEW: Combined Effects Comparison
# =============================================================================
println("  Combined effects comparison...")

# Start with monochromatic, add effects one by one
sino_chain = copy(sino_mono)

# Apply bowtie
bowtie_med = bowtie_filter_medium_body()
sino_chain = apply_bowtie_filter(sino_chain, bowtie_med, geom)

# Apply focal spot (smaller for subtle effect)
fs_small = focal_spot_medium()
sino_chain = apply_focal_spot_blur(sino_chain, fs_small, geom)

# Apply crosstalk (low)
ct_low = crosstalk_low()
sino_chain = apply_crosstalk(sino_chain, ct_low)

# Apply lag (Gadox)
lag_gos = lag_gadox()
sino_chain = apply_lag(sino_chain, lag_gos)

# Apply noise
sino_chain = apply_detector_model(sino_chain, detector_model)

recon_chain = fdk_reconstruct(sino_chain, geom, size(phantom.μ), phantom.fov)
recon_chain_HU = μ_to_HU(recon_chain, μ_water)
save_slice("18_recon_all_effects.png", recon_chain_HU[:, :, mid_slice];
           title="All Effects: Bowtie+Focal+Crosstalk+Lag+Noise")

# =============================================================================
# Comprehensive Comparison Figure
# =============================================================================
println("  Comprehensive comparison figure...")
fig = Figure(size=(1600, 800))

# Row 1: Basic effects
titles_r1 = ["Ideal (Mono)", "Beam Hardening", "Scatter", "Noise"]
data_r1 = [recon_mono_HU, recon_poly_HU, recon_scatter_HU, recon_noisy_HU]
for (i, (t, d)) in enumerate(zip(titles_r1, data_r1))
    local ax = Axis(fig[1, i], aspect=DataAspect(), title=t)
    heatmap!(ax, d[:, :, mid_slice]'; colormap=:grays, colorrange=(-1000, 1000))
    hidedecorations!(ax)
end

# Row 2: New effects
titles_r2 = ["Crosstalk", "All Effects", "Full Realistic", ""]
data_r2 = [recon_crosstalk_HU, recon_chain_HU, recon_full_HU]
for (i, (t, d)) in enumerate(zip(titles_r2[1:3], data_r2))
    local ax = Axis(fig[2, i], aspect=DataAspect(), title=t)
    heatmap!(ax, d[:, :, mid_slice]'; colormap=:grays, colorrange=(-1000, 1000))
    hidedecorations!(ax)
end

Colorbar(fig[1:2, 5], limits=(-1000, 1000), colormap=:grays, label="HU", height=Relative(0.8))
save(joinpath(output_dir, "19_comprehensive_comparison.png"), fig)
println("    Saved: 19_comprehensive_comparison.png")

# =============================================================================
# Summary: Effect Magnitudes
# =============================================================================
println("  Effect magnitudes summary...")

# Compute differences from ideal
diff_poly = mean(abs.(recon_poly_HU[:,:,mid_slice] - recon_mono_HU[:,:,mid_slice]))
diff_scatter = mean(abs.(recon_scatter_HU[:,:,mid_slice] - recon_mono_HU[:,:,mid_slice]))
diff_noisy = mean(abs.(recon_noisy_HU[:,:,mid_slice] - recon_mono_HU[:,:,mid_slice]))
diff_crosstalk = mean(abs.(recon_crosstalk_HU[:,:,mid_slice] - recon_mono_HU[:,:,mid_slice]))
diff_chain = mean(abs.(recon_chain_HU[:,:,mid_slice] - recon_mono_HU[:,:,mid_slice]))
diff_full = mean(abs.(recon_full_HU[:,:,mid_slice] - recon_mono_HU[:,:,mid_slice]))

fig = Figure(size=(800, 500))
ax = Axis(fig[1, 1], xlabel="Effect", ylabel="Mean Absolute Difference from Ideal (HU)",
          title="Effect Magnitudes on Reconstruction",
          xticks=(1:6, ["Poly\n(Beam Hard.)", "Scatter", "Noise", "Crosstalk",
                        "All New\nEffects", "Full\nRealistic"]))
barplot!(ax, 1:6, [diff_poly, diff_scatter, diff_noisy, diff_crosstalk, diff_chain, diff_full],
         color=[:blue, :orange, :green, :red, :purple, :brown])
save(joinpath(output_dir, "20_effect_magnitudes.png"), fig)
println("    Saved: 20_effect_magnitudes.png")

println("\nDone! Outputs in: $output_dir")
println("\nNew visualizations include:")
println("  08: Bowtie filter profile")
println("  09: Sinogram with bowtie")
println("  10: Focal spot blur effect")
println("  11: Crosstalk effect on projection")
println("  12: Reconstruction with crosstalk")
println("  13: Detector lag impulse response")
println("  14: Lag effect on projection")
println("  15: Helical scan trajectory")
println("  16: Helical sinogram")
println("  17: Interpolated axial from helical")
println("  18: All new effects combined")
println("  19: Comprehensive comparison")
println("  20: Effect magnitudes bar chart")
