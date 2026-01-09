"""
Quick Visualization - Fast simulation with image output
"""

using BasisSimulator
import XrayAttenuation as XA
using CairoMakie
using Statistics

println("="^70)
println("Quick Visualization Test")
println("="^70)

# Fast parameters
println("\n1. Creating small phantom...")
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=40.0)  # Small for speed

println("\n2. Setting up scanner...")
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=360)  # Match old version
geometry = create_aquilion_one(protocol=protocol)

println("\n3. Generating spectrum...")
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

println("\n4. Running forward simulation...")
sinogram_raw = simulate_ct_scan(
    phantom = phantom,
    geometry = geometry,
    spectrum = spectrum,
    verbose = false
)

println("   Raw sinogram (energy): $(size(sinogram_raw))")
println("   Range: $(minimum(sinogram_raw)) to $(maximum(sinogram_raw))")

println("\n5. Converting to attenuation line integrals...")
# Estimate air scan (I0 = incident intensity with no phantom)
I0 = estimate_air_scan(spectrum)
println("   I0 (air scan): $I0")

# Convert: -log(I/I0)
sinogram = convert_to_attenuation(sinogram_raw, I0)
println("   Attenuation sinogram: $(size(sinogram))")
println("   Range: $(minimum(sinogram)) to $(maximum(sinogram))")

println("\n6. Reconstructing...")
# Manual reconstruction grid
recon_fov_cm = 35.0  # 350 mm
recon_size = 256
z_thickness_cm = 4.0  # 40 mm

x_coords = range(-recon_fov_cm/2, recon_fov_cm/2, length=recon_size)
y_coords = range(-recon_fov_cm/2, recon_fov_cm/2, length=recon_size)
z_coords = range(-z_thickness_cm/2, z_thickness_cm/2, length=64)

reconstruction = reconstruct_fdk(
    sinogram,
    geometry.SAD_cm,
    geometry.SDD_cm,
    geometry.pixel_width_cm,
    geometry.pixel_height_cm,
    rad2deg.(geometry.angles),
    collect(x_coords),
    collect(y_coords),
    collect(z_coords)
)

# Debug: Check reconstruction μ values BEFORE HU conversion
println("   Reconstruction μ: $(size(reconstruction))")
println("   μ range: $(minimum(reconstruction)) to $(maximum(reconstruction)) cm^-1")

# Convert to HU using monochromatic μ_water at effective energy
# For 120 kVp with Al+Cu filtration, effective energy ~ 60-70 keV
μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
println("   μ_water (60 keV): $μ_water cm^-1")

reconstruction_hu = convert_to_hounsfield_units(reconstruction, μ_water)

println("   Reconstruction: $(size(reconstruction_hu))")
println("   HU range: $(minimum(reconstruction_hu)) to $(maximum(reconstruction_hu))")

println("\n6. Creating visualization...")
mkpath("test/outputs")

fig = Figure(size=(1200, 400))

# Phantom ground truth
ax1 = Axis(fig[1, 1], title="Phantom (Ground Truth)", aspect=DataAspect())
mid_z_phantom = phantom.grid.nz ÷ 2
heatmap!(ax1, phantom.grid.x, phantom.grid.y, phantom.material_ids[:, :, mid_z_phantom], colormap=:tab20)

# Sinogram
ax2 = Axis(fig[1, 2], title="Sinogram (Central Row)")
mid_row = size(sinogram, 1) ÷ 2
heatmap!(ax2, sinogram[mid_row, :, :], colormap=:grays)

# Reconstruction
ax3 = Axis(fig[1, 3], title="Reconstruction (HU)", aspect=DataAspect())
mid_z_recon = size(reconstruction_hu, 3) ÷ 2
hm = heatmap!(ax3, collect(x_coords), collect(y_coords), reconstruction_hu[:, :, mid_z_recon],
              colormap=:grays, colorrange=(-200, 400))
Colorbar(fig[1, 4], hm, label="HU")

output_file = "test/outputs/quick_validation.png"
save(output_file, fig, px_per_unit=2)

println("\n7. ✅ Visualization saved to: $output_file")
println("="^70)
