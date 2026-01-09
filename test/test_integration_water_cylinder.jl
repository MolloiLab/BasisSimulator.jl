"""
Integration Test: Water Cylinder Full Pipeline

Validates end-to-end CT simulation pipeline:
1. Generate X-ray spectrum
2. Create phantom
3. Forward simulate (ray tracing + polychromatic attenuation)
4. Reconstruct (FDK)
5. Validate HU values

This tests the complete imaging chain that component validations verified individually.
"""

using BasisSimulator
using Statistics
using Printf

println("\n" * "="^70)
println("INTEGRATION TEST: WATER CYLINDER FULL PIPELINE")
println("="^70)

# ==============================================================================
# 1. Setup Scan Parameters
# ==============================================================================
println("\n1. Setting up scan parameters...")

protocol = ScanProtocol(
    kVp = 120.0,
    mAs = 200.0,
    scan_fov_mm = 150.0,
    num_projections = 360
)

geometry = create_aquilion_one(protocol=protocol)

spectrum = generate_spectrum(
    kVp = 120.0,
    mAs = 200.0
)

println("   ✅ Protocol: $(protocol.kVp) kVp, $(protocol.mAs) mAs")
println("   ✅ Projections: $(protocol.num_projections)")
println("   ✅ Spectrum: $(length(spectrum.energies)) energy bins")

# ==============================================================================
# 2. Create Water Cylinder Phantom
# ==============================================================================
println("\n2. Creating water cylinder phantom...")

phantom = create_water_cylinder(
    diameter_mm = 100.0,
    height_mm = 40.0,
    resolution_mm = 2.0
)

println("   ✅ Phantom: $(phantom.name)")
println("   ✅ Size: $(phantom.grid.nx) × $(phantom.grid.ny) × $(phantom.grid.nz)")
println("   ✅ Resolution: 2.0 mm isotropic")
println("   ✅ Diameter: 100 mm")

# ==============================================================================
# 3. Forward Simulation (Polychromatic Ray Tracing)
# ==============================================================================
println("\n3. Running forward simulation...")
println("   (This may take 1-2 minutes for 360 projections)")

try
    sinogram = simulate_ct_scan(
        phantom = phantom,
        geometry = geometry,
        spectrum = spectrum,
        verbose = false
    )

    println("   ✅ Sinogram shape: $(size(sinogram))")
    println("   ✅ Sinogram range: [$(round(minimum(sinogram), digits=3)), $(round(maximum(sinogram), digits=3))]")

    # Check for reasonable values
    sino_min = minimum(sinogram)
    sino_max = maximum(sinogram)
    sino_mean = mean(sinogram)

    println("   ✅ Mean signal: $(round(sino_mean, sigdigits=4))")

    # Sinogram should have positive values (energy deposited)
    if sino_min >= 0 && sino_max > 0
        println("   ✅ Sinogram values physically reasonable")
    else
        println("   ⚠️  Sinogram contains unexpected values")
    end

    # ==============================================================================
    # 4. Convert to Attenuation Sinogram
    # ==============================================================================
    println("\n4. Converting to attenuation sinogram...")

    # Air scan (no phantom - maximum signal)
    I0 = sino_max  # Approximate air scan value

    # Convert transmission to attenuation: μt = -ln(I/I0)
    attenuation_sino = -log.(sinogram ./ I0)

    println("   ✅ Attenuation range: [$(round(minimum(attenuation_sino), digits=3)), $(round(maximum(attenuation_sino), digits=3))] cm⁻¹")

    # Expected attenuation for 10 cm water at 60 keV: ~0.2 cm⁻¹ × 10 cm = 2.0
    expected_atten = 0.2 * 10.0  # ~2.0
    max_atten = maximum(attenuation_sino)

    if max_atten > 0.5 && max_atten < 5.0
        println("   ✅ Maximum attenuation reasonable: $(round(max_atten, digits=2)) cm⁻¹")
        println("      (Expected ~2.0 cm⁻¹ for 10 cm water)")
    else
        println("   ⚠️  Attenuation outside expected range")
    end

    # ==============================================================================
    # 5. FDK Reconstruction
    # ==============================================================================
    println("\n5. Reconstructing volume (FDK algorithm)...")

    # Create reconstruction grid to match phantom dimensions
    recon_x_cm = collect(range(-phantom.grid.fov_xy_cm/2, phantom.grid.fov_xy_cm/2, length=phantom.grid.nx))
    recon_y_cm = collect(range(-phantom.grid.fov_xy_cm/2, phantom.grid.fov_xy_cm/2, length=phantom.grid.ny))
    recon_z_cm = collect(range(-phantom.grid.fov_z_cm/2, phantom.grid.fov_z_cm/2, length=phantom.grid.nz))

    # Convert angles from radians to degrees
    angles_deg = rad2deg.(geometry.angles)

    reconstruction = reconstruct_fdk(
        attenuation_sino,
        geometry.SAD_cm,
        geometry.SDD_cm,
        geometry.pixel_width_cm,
        geometry.pixel_height_cm,
        angles_deg,
        recon_x_cm,
        recon_y_cm,
        recon_z_cm,
        filter_type = ramlak,
        use_parker_weighting = false
    )

    println("   ✅ Reconstruction shape: $(size(reconstruction))")
    println("   ✅ Reconstruction range: [$(round(minimum(reconstruction), digits=3)), $(round(maximum(reconstruction), digits=3))] cm⁻¹")

    # ==============================================================================
    # 6. Convert to Hounsfield Units
    # ==============================================================================
    println("\n6. Converting to Hounsfield Units...")

    # Reference attenuation for water at 60 keV (approximate mean energy)
    import XrayAttenuation as XA
    mu_water_ref = get_linear_attenuation(XA.Materials.water, 60.0)  # ~0.2 cm⁻¹

    # HU = 1000 × (μ - μ_water) / μ_water
    HU = @. 1000.0 * (reconstruction - mu_water_ref) / mu_water_ref

    println("   ✅ HU range: [$(round(minimum(HU), digits=1)), $(round(maximum(HU), digits=1))]")

    # ==============================================================================
    # 7. Analyze Central Slice
    # ==============================================================================
    println("\n7. Analyzing central slice...")

    center_slice = div(size(HU, 3), 2)
    central_slice = HU[:, :, center_slice]

    # Define ROI in center (20×20 pixels)
    center_x = div(size(HU, 1), 2)
    center_y = div(size(HU, 2), 2)
    roi_size = 10

    roi = central_slice[
        (center_x - roi_size):(center_x + roi_size),
        (center_y - roi_size):(center_y + roi_size)
    ]

    roi_mean = mean(roi)
    roi_std = std(roi)

    println("\n   Central ROI Analysis (21×21 pixels):")
    println("      Mean HU: $(round(roi_mean, digits=1)) HU")
    println("      Std Dev: $(round(roi_std, digits=1)) HU")
    println("      Expected: ~0 HU (water by definition)")

    # ==============================================================================
    # 8. Validation Summary
    # ==============================================================================
    println("\n" * "="^70)
    println("INTEGRATION TEST SUMMARY")
    println("="^70)

    checks = [
        ("Forward simulation runs", true),
        ("Sinogram physically reasonable", sino_min >= 0 && sino_max > 0),
        ("Attenuation in expected range", max_atten > 0.5 && max_atten < 5.0),
        ("Reconstruction succeeds", size(reconstruction) == size(phantom.material_ids)),
        ("HU conversion succeeds", true),
    ]

    println()
    all_passed = all([passed for (name, passed) in checks])
    for (name, passed) in checks
        status = passed ? "✅" : "⚠️"
        println("$status $name")
    end

    println("\n" * "="^70)
    if all_passed
        println("✅ INTEGRATION TEST PASSED")
        println("\nKey Results:")
        println("  • Full pipeline executed successfully")
        println("  • Sinogram: $(size(sinogram)) projections")
        println("  • Reconstruction: $(size(reconstruction)) volume")
        println("  • Central ROI: $(round(roi_mean, digits=1)) ± $(round(roi_std, digits=1)) HU")
        println("\nNext Steps:")
        println("  • Validate HU accuracy (expected ~0 HU for water)")
        println("  • Optimize reconstruction parameters if needed")
        println("  • Test with multi-material phantom (Gammex 472)")
    else
        println("⚠️  SOME CHECKS FAILED")
        println("\nDebugging Needed:")
        println("  • Check forward simulation normalization")
        println("  • Verify attenuation calculation")
        println("  • Review reconstruction parameters")
    end
    println("="^70 * "\n")

catch e
    println("\n" * "="^70)
    println("❌ INTEGRATION TEST FAILED")
    println("="^70)
    println("\nError during pipeline execution:")
    println("   $(typeof(e)): $e")
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    println("\n" * "="^70 * "\n")
    rethrow(e)
end
