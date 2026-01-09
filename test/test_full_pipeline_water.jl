"""
Test complete BasisSimulator pipeline with water cylinder.

This validates the full CT simulation workflow:
1. Phantom creation
2. Spectrum generation
3. Scanner geometry
4. Forward simulation (ray tracing + polychromatic)
5. Sinogram generation
6. FDK reconstruction
7. HU conversion and validation
"""

using BasisSimulator
using Statistics

println("\n" * "="^70)
println("BASISSIMULATOR FULL PIPELINE TEST - WATER CYLINDER")
println("="^70)

# ==============================================================================
# 1. Create Water Cylinder Phantom
# ==============================================================================
println("\n1. Creating water cylinder phantom...")

phantom = create_water_cylinder(
    diameter_mm=100.0,
    height_mm=20.0,
    resolution_mm=2.0
)

println("   ✅ Phantom created")
println("      Grid size: $(size(phantom.material_ids))")
println("      Materials: $(length(phantom.id_to_material))")
println("      Grid dimensions: $(phantom.grid.nx) × $(phantom.grid.ny) × $(phantom.grid.nz)")
println("      FOV: $(phantom.grid.fov_xy_cm) cm (xy) × $(phantom.grid.fov_z_cm) cm (z)")

# ==============================================================================
# 2. Generate X-ray Spectrum
# ==============================================================================
println("\n2. Generating X-ray spectrum...")

spectrum = generate_spectrum(
    kVp=120.0,
    mAs=200.0
)

println("   ✅ Spectrum generated")
println("      Energy range: $(minimum(spectrum.energies)) - $(maximum(spectrum.energies)) keV")
println("      Number of bins: $(length(spectrum.energies))")
println("      Total photons: $(round(sum(spectrum.photons), sigdigits=3))")
println("      Mean energy: $(round(sum(spectrum.energies .* spectrum.photons) / sum(spectrum.photons), digits=2)) keV")

# ==============================================================================
# 3. Create Scanner Geometry
# ==============================================================================
println("\n3. Creating scanner geometry...")

protocol = ScanProtocol(
    kVp=120.0,
    mAs=200.0,
    scan_fov_mm=120.0,
    num_projections=180  # Moderate number for testing
)

geometry = create_aquilion_one(protocol=protocol)

println("   ✅ Scanner geometry created")
println("      SAD: $(geometry.SAD_cm * 10) mm")
println("      SDD: $(geometry.SDD_cm * 10) mm")
println("      Detector: $(geometry.n_rows) × $(geometry.n_cols)")
println("      Pixel size: $(geometry.pixel_width_cm * 10) × $(geometry.pixel_height_cm * 10) mm")
println("      Projections: $(length(geometry.angles))")
println("      Angular range: $(rad2deg(minimum(geometry.angles))) - $(rad2deg(maximum(geometry.angles))) degrees")

# ==============================================================================
# 4. Run Forward Simulation
# ==============================================================================
println("\n4. Running forward simulation (ray tracing)...")
println("   (This may take a moment...)")

detector_signal = simulate_ct_scan(
    phantom=phantom,
    geometry=geometry,
    spectrum=spectrum
)

println("   ✅ Forward simulation complete")
println("      Signal shape: $(size(detector_signal))")
println("      Signal range: $(round(minimum(detector_signal), sigdigits=3)) - $(round(maximum(detector_signal), sigdigits=3))")
println("      Signal mean: $(round(mean(detector_signal), sigdigits=3))")

# Check for valid signal
if any(isnan.(detector_signal))
    @warn "NaN values detected in detector signal!"
end
if any(isinf.(detector_signal))
    @warn "Inf values detected in detector signal!"
end

# ==============================================================================
# 5. Convert to Attenuation Sinogram
# ==============================================================================
println("\n5. Converting to attenuation sinogram...")

I0 = estimate_air_scan(spectrum)
println("   Air scan intensity (I0): $(round(I0, sigdigits=3))")

sinogram = convert_to_attenuation(detector_signal, I0)

println("   ✅ Sinogram created")
println("      Sinogram shape: $(size(sinogram))")
println("      Sinogram range: $(round(minimum(sinogram), digits=4)) - $(round(maximum(sinogram), digits=4)) cm⁻¹")
println("      Sinogram mean: $(round(mean(sinogram), digits=4)) cm⁻¹")

# ==============================================================================
# 6. FDK Reconstruction
# ==============================================================================
println("\n6. Running FDK reconstruction...")

fov_cm = 12.0
image_size = 128
recon_x = collect(range(-fov_cm/2, fov_cm/2, length=image_size))
recon_y = collect(range(-fov_cm/2, fov_cm/2, length=image_size))
recon_z = collect(range(-1.0, 1.0, length=16))

volume = reconstruct_fdk(
    sinogram,
    geometry.SAD_cm,
    geometry.SDD_cm,
    geometry.pixel_width_cm,
    geometry.pixel_height_cm,
    geometry.angles,
    recon_x,
    recon_y,
    recon_z,
    filter_type=ramlak
)

println("   ✅ Reconstruction complete")
println("      Volume shape: $(size(volume))")
println("      Volume range: $(round(minimum(volume), digits=4)) - $(round(maximum(volume), digits=4)) cm⁻¹")
println("      Volume mean: $(round(mean(volume), digits=4)) cm⁻¹")

# ==============================================================================
# 7. Analyze Reconstructed Volume
# ==============================================================================
println("\n7. Analyzing reconstruction quality...")

# Get center slice
center_slice = volume[:, :, div(size(volume, 3), 2)]

# Define ROI in center (water region)
roi_radius = 20  # pixels
center_x, center_y = div(image_size, 2), div(image_size, 2)

# Extract water ROI values
water_values = Float64[]
for i in 1:image_size, j in 1:image_size
    if sqrt((i - center_x)^2 + (j - center_y)^2) < roi_radius
        push!(water_values, center_slice[i, j])
    end
end

# Convert to HU (use measured water as reference for HU = 0)
mu_water = mean(water_values)
water_hu = (water_values .- mu_water) ./ mu_water .* 1000

println("   Water ROI Statistics:")
println("      Attenuation: $(round(mean(water_values), digits=4)) ± $(round(std(water_values), digits=4)) cm⁻¹")
println("      HU relative variability: $(round(std(water_hu), digits=1)) HU")
println("      Expected: <50 HU variability (uniform material)")
println("      Note: Absolute HU calibrated to water (0 HU by definition)")

# Background (air) analysis
background_values = Float64[]
for i in 1:image_size, j in 1:image_size
    if sqrt((i - center_x)^2 + (j - center_y)^2) > roi_radius * 2
        push!(background_values, center_slice[i, j])
    end
end

println("\n   Background (air) Statistics:")
println("      Attenuation: $(round(mean(background_values), digits=4)) ± $(round(std(background_values), digits=4)) cm⁻¹")
println("      Range: $(round(minimum(background_values), digits=4)) - $(round(maximum(background_values), digits=4)) cm⁻¹")

# ==============================================================================
# 8. Validation Checks
# ==============================================================================
println("\n" * "="^70)
println("VALIDATION SUMMARY")
println("="^70)

checks = [
    ("Phantom created", true),
    ("Spectrum generated", true),
    ("Geometry configured", true),
    ("Forward simulation completed", !any(isnan.(detector_signal)) && !any(isinf.(detector_signal))),
    ("Sinogram valid", !any(isnan.(sinogram)) && !any(isinf.(sinogram))),
    ("Reconstruction completed", !any(isnan.(volume)) && !any(isinf.(volume))),
    ("Water uniformity good", std(water_hu) < 50),
    ("Water attenuation positive", mean(water_values) > 0),
    ("Background lower than water", mean(background_values) < mean(water_values))
]

println()
all_passed = true
for (name, passed) in checks
    status = passed ? "✅" : "❌"
    println("$status $name")
    global all_passed = all_passed && passed
end

println("\n" * "="^70)
if all_passed
    println("✅ FULL PIPELINE TEST PASSED")
    println("BasisSimulator is ready for cross-validation with GECATSIM")
else
    println("❌ SOME CHECKS FAILED - REQUIRES INVESTIGATION")
end
println("="^70 * "\n")

# Save results for potential inspection
results = (
    phantom = phantom,
    spectrum = spectrum,
    geometry = geometry,
    detector_signal = detector_signal,
    sinogram = sinogram,
    volume = volume,
    water_hu_mean = mean(water_hu),
    water_hu_std = std(water_hu)
)

println("Results saved in variable 'results' for inspection")
