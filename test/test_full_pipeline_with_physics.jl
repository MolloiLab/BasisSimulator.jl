"""
Test full CT simulation pipeline with Phase 2 physics models integrated.

This script verifies that:
1. All new physics exports are available
2. Full simulation with scatter, noise, and bowtie filter works
3. Results are still reasonable (sinogram ~7 cm^-1, reconstruction valid HU)
"""

using BasisSimulator
import XrayAttenuation as XA
using Statistics

println("\n" * "="^70)
println("FULL PIPELINE TEST WITH PHASE 2 PHYSICS")
println("="^70)

# ==============================================================================
# 1. Verify Module Exports
# ==============================================================================
println("\n1. Verifying Phase 2 physics exports...")

# Scatter exports
@assert isdefined(BasisSimulator, :estimate_scatter_convolution) "Missing: estimate_scatter_convolution"
@assert isdefined(BasisSimulator, :klein_nishina_cross_section) "Missing: klein_nishina_cross_section"

# Noise exports
@assert isdefined(BasisSimulator, :apply_poisson_noise) "Missing: apply_poisson_noise"
@assert isdefined(BasisSimulator, :add_electronic_noise) "Missing: add_electronic_noise"
@assert isdefined(BasisSimulator, :add_1_over_f_noise) "Missing: add_1_over_f_noise"
@assert isdefined(BasisSimulator, :compute_nps) "Missing: compute_nps"

# Bowtie filter exports
@assert isdefined(BasisSimulator, :bowtie_thickness_profile) "Missing: bowtie_thickness_profile"
@assert isdefined(BasisSimulator, :apply_bowtie_filter) "Missing: apply_bowtie_filter"

# Iterative reconstruction exports
@assert isdefined(BasisSimulator, :reconstruct_sirt) "Missing: reconstruct_sirt"
@assert isdefined(BasisSimulator, :reconstruct_mlem) "Missing: reconstruct_mlem"
@assert isdefined(BasisSimulator, :reconstruct_tv) "Missing: reconstruct_tv"

println("   ✅ All Phase 2 exports available")

# ==============================================================================
# 2. Create Test Phantom (Small Water Cylinder)
# ==============================================================================
println("\n2. Creating test phantom...")

phantom = create_water_cylinder(
    diameter_mm=200.0,
    height_mm=40.0,
    resolution_mm=2.0
)

println("   ✅ Phantom created: $(phantom.name)")
println("      Grid size: $(size(phantom.material_ids))")
println("      Memory: $(get_memory_usage(phantom)) GB")

# ==============================================================================
# 3. Generate X-ray Spectrum
# ==============================================================================
println("\n3. Generating X-ray spectrum...")

spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

println("   ✅ Spectrum generated")
println("      Energy range: $(minimum(spectrum.energies)) - $(maximum(spectrum.energies)) keV")
println("      Mean energy: $(round(mean_energy(spectrum), digits=1)) keV")
println("      Total fluence: $(round(total_fluence(spectrum), digits=0)) photons")

# ==============================================================================
# 4. Create Scanner Geometry (Small Scan for Speed)
# ==============================================================================
println("\n4. Creating scanner geometry...")

protocol = ScanProtocol(
    kVp=120.0,
    mAs=200.0,
    scan_fov_mm=200.0,
    num_projections=10  # Small number for testing
)

geometry = create_aquilion_one(protocol=protocol)

println("   ✅ Scanner geometry created")
println("      Projections: $(length(geometry.angles))")
println("      Detector: $(geometry.n_rows)×$(geometry.n_cols)")
println("      SDD: $(geometry.SDD_cm) cm")

# ==============================================================================
# 5. Run Forward Simulation (Without Physics First)
# ==============================================================================
println("\n5. Running baseline forward simulation...")

detector_signal = simulate_ct_scan(
    phantom=phantom,
    geometry=geometry,
    spectrum=spectrum
)

# Convert to attenuation sinogram
I0 = estimate_air_scan(spectrum)
sinogram_baseline = convert_to_attenuation(detector_signal, I0)

sino_baseline_max = maximum(sinogram_baseline)
sino_baseline_mean = mean(sinogram_baseline)

println("   ✅ Baseline simulation complete")
println("      Detector signal max: $(round(maximum(detector_signal), digits=0))")
println("      I0: $(round(I0, digits=0))")
println("      Sinogram max: $(round(sino_baseline_max, digits=2)) cm^-1")
println("      Sinogram mean: $(round(sino_baseline_mean, digits=2)) cm^-1")
println("      Expected max: ~6.8 cm^-1 (for 20 cm water at 120 kVp)")

# ==============================================================================
# 6. Test Scatter Estimation
# ==============================================================================
println("\n6. Testing scatter estimation...")

# Primary is just the detector signal from simulation
primary = detector_signal

# Estimate scatter
scatter = estimate_scatter_convolution(
    primary,
    spr=0.15,
    kernel_sigma=10.0  # Smaller for small phantom
)

scatter_ratio = mean(scatter) / mean(primary)

println("   ✅ Scatter estimation complete")
println("      Primary mean: $(round(mean(primary), digits=0))")
println("      Scatter mean: $(round(mean(scatter), digits=0))")
println("      Scatter/Primary ratio: $(round(scatter_ratio, digits=3))")
println("      Expected ratio: ~0.15")

# Reconstruct with scatter
total_signal = primary .+ scatter
sinogram_with_scatter = convert_to_attenuation(total_signal, I0)

sino_scatter_max = maximum(sinogram_with_scatter)
println("      Sinogram (with scatter) max: $(round(sino_scatter_max, digits=2)) cm^-1")

# ==============================================================================
# 7. Test Noise Models
# ==============================================================================
println("\n7. Testing noise models...")

# Test Poisson noise
noisy_poisson = apply_poisson_noise(primary, dose_factor=1.0, seed=42)
poisson_std = std(noisy_poisson .- primary) / mean(primary)

println("   ✅ Poisson noise applied")
println("      Relative noise: $(round(poisson_std, digits=4))")

# Test electronic noise
noisy_electronic = add_electronic_noise(primary, sigma=500.0, seed=42)
electronic_std = std(noisy_electronic .- primary)

println("   ✅ Electronic noise applied")
println("      Absolute noise std: $(round(electronic_std, digits=1))")

# Combined noise
noisy_combined = add_electronic_noise(
    apply_poisson_noise(primary, dose_factor=1.0, seed=42),
    sigma=500.0,
    seed=43
)

# Convert to sinogram
sinogram_noisy = convert_to_attenuation(noisy_combined, I0)

sino_noisy_max = maximum(sinogram_noisy)
println("      Sinogram (with noise) max: $(round(sino_noisy_max, digits=2)) cm^-1")

# ==============================================================================
# 8. Test Bowtie Filter
# ==============================================================================
println("\n8. Testing bowtie filter...")

# Create fan angles for test
fan_angles = collect(range(-24.75, 24.75, length=geometry.n_cols))
bowtie_profile = bowtie_thickness_profile(
    fan_angles,
    max_thickness_cm=3.0,
    center_thickness_cm=0.5,
    profile=:parabolic
)

println("   ✅ Bowtie profile created")
println("      Center thickness: $(round(bowtie_profile[div(end,2)], digits=2)) cm")
println("      Edge thickness: $(round(bowtie_profile[1], digits=2)) cm")

# Test applying bowtie to spectrum
filtered_center = apply_bowtie_filter(
    spectrum.energies,
    spectrum.photons,
    div(geometry.n_cols, 2),  # Center column
    geometry.n_cols,
    max_thickness_cm=3.0
)

filtered_edge = apply_bowtie_filter(
    spectrum.energies,
    spectrum.photons,
    1,  # Edge column
    geometry.n_cols,
    max_thickness_cm=3.0
)

attenuation_center = 1.0 - sum(filtered_center) / sum(spectrum.photons)
attenuation_edge = 1.0 - sum(filtered_edge) / sum(spectrum.photons)

println("   ✅ Bowtie filter applied")
println("      Center attenuation: $(round(attenuation_center*100, digits=1))%")
println("      Edge attenuation: $(round(attenuation_edge*100, digits=1))%")
println("      Expected: center < 20%, edge > 40%")

# ==============================================================================
# 9. Test Reconstruction with Baseline
# ==============================================================================
println("\n9. Testing FDK reconstruction...")

# Create reconstruction grid (small for speed)
image_size = 128
fov_cm = 20.0  # 200mm FOV
voxel_size_cm = fov_cm / image_size
recon_x_cm = collect(range(-fov_cm/2, fov_cm/2, length=image_size))
recon_y_cm = collect(range(-fov_cm/2, fov_cm/2, length=image_size))
recon_z_cm = collect(range(-2.0, 2.0, length=20))  # 4cm Z coverage

volume = reconstruct_fdk(
    sinogram_baseline,
    geometry.SAD_cm,  # Source-to-axis distance
    geometry.SDD_cm,  # Source-to-detector distance
    geometry.pixel_width_cm,
    geometry.pixel_height_cm,
    geometry.angles,
    recon_x_cm,
    recon_y_cm,
    recon_z_cm,
    filter_type=ramlak
)

# Convert to HU
μ_water_expected = 0.2  # Approximate for 120 kVp
volume_hu = convert_to_hounsfield_units(volume, μ_water_expected)

# Get center slice
center_slice = div(size(volume_hu, 3), 2)
slice_hu = volume_hu[:, :, center_slice]

# Get central region stats (avoid edges)
center = div(size(slice_hu, 1), 2)
radius = 20
central_region = slice_hu[
    (center-radius):(center+radius),
    (center-radius):(center+radius)
]

hu_mean = mean(central_region)
hu_std = std(central_region)
hu_min = minimum(slice_hu)
hu_max = maximum(slice_hu)

println("   ✅ Reconstruction complete")
println("      Volume size: $(size(volume))")
println("      Central HU: $(round(hu_mean, digits=1)) ± $(round(hu_std, digits=1))")
println("      HU range: $(round(hu_min, digits=0)) to $(round(hu_max, digits=0))")
println("      Expected water: ~0 HU")

# ==============================================================================
# 10. Verify Iterative Placeholders Work
# ==============================================================================
println("\n10. Verifying iterative reconstruction placeholders...")

# These should throw errors with informative messages
try
    reconstruct_sirt(sinogram_baseline, geometry, 128)
    println("   ❌ SIRT should throw error (not implemented)")
catch e
    if occursin("not yet implemented", string(e))
        println("   ✅ SIRT placeholder works correctly")
    else
        println("   ❌ SIRT unexpected error: $e")
    end
end

try
    reconstruct_mlem(sinogram_baseline, geometry, 128)
    println("   ❌ MLEM should throw error (not implemented)")
catch e
    if occursin("not yet implemented", string(e))
        println("   ✅ MLEM placeholder works correctly")
    else
        println("   ❌ MLEM unexpected error: $e")
    end
end

try
    reconstruct_tv(sinogram_baseline, geometry, 128)
    println("   ❌ TV should throw error (not implemented)")
catch e
    if occursin("not yet implemented", string(e))
        println("   ✅ TV placeholder works correctly")
    else
        println("   ❌ TV unexpected error: $e")
    end
end

# ==============================================================================
# 11. Summary and Validation
# ==============================================================================
println("\n" * "="^70)
println("VALIDATION SUMMARY")
println("="^70)

# Check all critical metrics
checks = []

# Baseline sinogram
push!(checks, ("Baseline sinogram max", sino_baseline_max, 5.0, 8.0, "cm^-1"))
push!(checks, ("Baseline sinogram mean", sino_baseline_mean, 1.0, 4.0, "cm^-1"))

# Scatter
push!(checks, ("Scatter/Primary ratio", scatter_ratio, 0.10, 0.20, ""))

# Reconstruction
push!(checks, ("Central HU (water)", hu_mean, -100.0, 100.0, "HU"))
push!(checks, ("HU noise", hu_std, 0.0, 200.0, "HU"))

# Bowtie
push!(checks, ("Center attenuation", attenuation_center, 0.0, 0.25, ""))
push!(checks, ("Edge attenuation", attenuation_edge, 0.30, 0.70, ""))

println()
global all_passed = true
for (name, value, min_val, max_val, unit) in checks
    passed = min_val <= value <= max_val
    status = passed ? "✅" : "❌"
    global all_passed = all_passed && passed

    unit_str = isempty(unit) ? "" : " $unit"
    println("$status $name: $(round(value, digits=2))$unit_str (expected: $(min_val)-$(max_val))")
end

println("\n" * "="^70)
if all_passed
    println("✅ ALL CHECKS PASSED - PIPELINE WORKING WITH PHASE 2 PHYSICS")
else
    println("❌ SOME CHECKS FAILED - REQUIRES INVESTIGATION")
end
println("="^70 * "\n")
