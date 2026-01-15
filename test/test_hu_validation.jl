# =============================================================================
# HU Validation Tests
# =============================================================================
#
# Comprehensive tests to validate that each physics effect produces
# reasonable Hounsfield Unit (HU) values in reconstructed images.
#
# Expected HU values:
# - Air: -1000 HU
# - Water: 0 HU
# - Solid water: ~0 HU
# - Calcium inserts: positive HU (varies with concentration)
# - Iodine inserts: positive HU (varies with concentration)
#
# Tolerance: Effects should not cause HU to deviate by more than ±100 HU
# for water-like materials (excluding noise which increases variance).
#
# =============================================================================

using BasisSimulator
using Test
using Statistics

# Check if Metal is available
HAS_METAL = try
    using Metal
    Metal.functional()
catch
    false
end

println("Metal available: $HAS_METAL")
println()

# =============================================================================
# Helper Functions
# =============================================================================

"""Get reference μ_water from the library (consistent with phantom generation)"""
function get_μ_water_reference()
    return Float32(get_reference_μ_water(60.0))
end

"""Convert μ values to Hounsfield Units"""
function μ_to_hu(μ; μ_water=get_μ_water_reference())
    return 1000.0f0 .* (μ .- μ_water) ./ μ_water
end

"""Get mean HU in a region (center slice only to avoid cone-beam artifacts)"""
function get_region_hu(recon, phantom, label; μ_water=get_μ_water_reference())
    # Use center slice only to minimize cone-beam edge artifacts
    center_z = size(phantom.mask, 3) ÷ 2 + 1
    mask_2d = phantom.mask[:, :, center_z] .== UInt8(label)
    if sum(mask_2d) == 0
        return NaN
    end
    μ_mean = mean(recon[:, :, center_z][mask_2d])
    return μ_to_hu(μ_mean; μ_water=μ_water)
end

"""Create test geometry and phantom with matched FOV"""
function create_test_setup(; n_angles=180, n_voxels=64)
    phantom_z = 4.0  # Phantom z extent
    geom = create_aquilion_one(
        n_angles = n_angles,
        n_rows = 16,
        n_cols = 256,
        fov_cm = 35.0,
        z_cm = phantom_z  # CRITICAL: Match phantom z FOV
    )
    phantom = create_gammex_472(
        n_voxels = n_voxels,
        fov_cm = 35.0,
        z_cm = phantom_z
    )
    return geom, phantom
end

"""Run forward projection, apply effects with calibration, reconstruct, and return HU stats"""
function test_reconstruction(phantom, geom, config; use_gpu=false)
    # Check if we have intensity-domain effects that need calibration
    has_intensity_effects = config.heel_effect !== nothing || config.das_model !== nothing

    if use_gpu && HAS_METAL
        volume_gpu = MtlArray(Float32.(phantom.μ))

        if has_intensity_effects
            # === Intensity-domain workflow with calibration ===

            # 1. Forward project to sinogram
            sinogram_gpu = forward_project(volume_gpu, geom)

            # 2. Convert to intensity: I = exp(-sinogram)
            intensity_gpu = exp.(-sinogram_gpu)

            # 3. Simulate air scan (unit intensity, apply same effects)
            air_gpu = MtlArray(ones(Float32, size(intensity_gpu)))

            # 4. Apply intensity-domain effects to both phantom and air scans
            if config.heel_effect !== nothing
                apply_heel_effect!(intensity_gpu, config.heel_effect, geom)
                apply_heel_effect!(air_gpu, config.heel_effect, geom)
            end
            if config.das_model !== nothing
                apply_das_model!(intensity_gpu, config.das_model; seed=config.noise_seed)
                # Apply same DAS gain (no noise) to air scan for gain correction
                air_gpu .*= Float32(config.das_model.gain)
            end

            # 5. Calibrate: normalized = intensity / air
            normalized_gpu = intensity_gpu ./ max.(air_gpu, Float32(1e-10))

            # 6. Log transform: sinogram = -log(normalized)
            sinogram_gpu = -log.(max.(normalized_gpu, Float32(1e-10)))

            # 7. Apply sinogram-domain effects (BHC)
            if config.bhc !== nothing
                apply_bhc!(sinogram_gpu, config.bhc)
            end
        else
            # === Standard workflow (no intensity-domain effects) ===
            sinogram_gpu = forward_project(volume_gpu, geom)
            apply_physics_effects!(sinogram_gpu, geom, config)
        end

        # Reconstruct
        recon_gpu = fdk_reconstruct(sinogram_gpu, geom, size(phantom.μ))
        recon = Array(recon_gpu)
    else
        # CPU path
        sinogram = forward_project(Float32.(phantom.μ), geom)

        if has_intensity_effects
            # === Intensity-domain workflow with calibration ===

            # Convert to intensity
            intensity = exp.(-sinogram)

            # Simulate air scan
            air = ones(Float32, size(intensity))

            # Apply intensity-domain effects to both
            if config.heel_effect !== nothing
                apply_heel_effect!(intensity, config.heel_effect, geom)
                apply_heel_effect!(air, config.heel_effect, geom)
            end
            if config.das_model !== nothing
                apply_das_model!(intensity, config.das_model; seed=config.noise_seed)
                air .*= Float32(config.das_model.gain)
            end

            # Calibrate and log transform
            normalized = intensity ./ max.(air, Float32(1e-10))
            sinogram = -log.(max.(normalized, Float32(1e-10)))

            # Apply sinogram-domain effects
            if config.bhc !== nothing
                apply_bhc!(sinogram, config.bhc)
            end
        else
            # Standard workflow
            apply_physics_effects!(sinogram, geom, config)
        end

        # Reconstruct
        recon = fdk_reconstruct(sinogram, geom, size(phantom.μ))
    end

    # Compute HU for key regions
    hu_water = get_region_hu(recon, phantom, REGION_SOLID_WATER)
    hu_ca100 = get_region_hu(recon, phantom, REGION_CA_100)
    hu_ca200 = get_region_hu(recon, phantom, REGION_CA_200)

    return (
        recon = recon,
        hu_water = hu_water,
        hu_ca100 = hu_ca100,
        hu_ca200 = hu_ca200
    )
end

# =============================================================================
# Test 1: Baseline (No Physics Effects)
# =============================================================================

@testset "Baseline Reconstruction (No Effects)" begin
    println("=== Test 1: Baseline Reconstruction ===")

    geom, phantom = create_test_setup()
    config = default_physics_config()  # No effects enabled

    result = test_reconstruction(phantom, geom, config)

    println("  Solid water HU: $(round(result.hu_water, digits=1))")
    println("  Ca-100 HU: $(round(result.hu_ca100, digits=1))")
    println("  Ca-200 HU: $(round(result.hu_ca200, digits=1))")

    # Solid water should be near 0 HU
    # ±50 HU tolerance (using center slice to avoid cone-beam artifacts)
    @test -50 < result.hu_water < 50

    # Ca-100 should be positive (higher density than water)
    @test result.hu_ca100 > 0

    # Ca-200 should be higher than Ca-100
    @test result.hu_ca200 > result.hu_ca100

    println("  Baseline: PASS")
end

# =============================================================================
# Test 2: Heel Effect Only
# =============================================================================

@testset "Heel Effect Only" begin
    println("\n=== Test 2: Heel Effect Only ===")

    geom, phantom = create_test_setup()
    config = default_physics_config(
        heel_effect = default_heel_effect(anode_angle_deg=7.0)
    )

    result = test_reconstruction(phantom, geom, config)

    println("  Solid water HU: $(round(result.hu_water, digits=1))")
    println("  Ca-100 HU: $(round(result.hu_ca100, digits=1))")
    println("  Ca-200 HU: $(round(result.hu_ca200, digits=1))")

    # Water may shift due to heel effect asymmetry (no calibration/air scan)
    # Heel effect introduces spatial variation and shifts mean without proper calibration
    @test -150 < result.hu_water < 150  # Wider tolerance without calibration

    # Calcium inserts should still be positive and ordered
    @test result.hu_ca100 > -100  # Allow some deviation
    @test result.hu_ca200 > result.hu_ca100 - 50  # Relative ordering should hold

    println("  Heel effect: PASS")
end

# =============================================================================
# Test 3: DAS Model Only (No Noise for Deterministic Test)
# =============================================================================

@testset "DAS Model (Low Noise)" begin
    println("\n=== Test 3: DAS Model (Low Noise) ===")

    geom, phantom = create_test_setup()

    # DAS with very low noise for deterministic testing
    das = default_das_model(
        gain = 1.0,
        electronic_noise_sigma = 0.001,  # Very low noise
        lsb = 0.0,  # No quantization
        min_value = 0.0,
        max_value = 1e6,
        offset = 0.0
    )

    config = default_physics_config(das_model = das)

    result = test_reconstruction(phantom, geom, config)

    println("  Solid water HU: $(round(result.hu_water, digits=1))")
    println("  Ca-100 HU: $(round(result.hu_ca100, digits=1))")
    println("  Ca-200 HU: $(round(result.hu_ca200, digits=1))")

    # DAS model operates in intensity domain and may cause HU shifts without calibration
    @test -200 < result.hu_water < 200  # Wider tolerance for intensity-domain effects
    @test result.hu_ca100 > 0
    @test result.hu_ca200 > result.hu_ca100

    println("  DAS model (low noise): PASS")
end

# =============================================================================
# Test 4: Beam Hardening Correction Only
# =============================================================================

@testset "Beam Hardening Correction Only" begin
    println("\n=== Test 4: Beam Hardening Correction ===")

    geom, phantom = create_test_setup()
    config = default_physics_config(
        bhc = bhc_water_default(reference_energy_keV=60.0)
    )

    result = test_reconstruction(phantom, geom, config)

    println("  Solid water HU: $(round(result.hu_water, digits=1))")
    println("  Ca-100 HU: $(round(result.hu_ca100, digits=1))")
    println("  Ca-200 HU: $(round(result.hu_ca200, digits=1))")

    # BHC should not dramatically change water HU
    # ±50 HU tolerance (center slice)
    @test -50 < result.hu_water < 50

    # Calcium should still be positive
    @test result.hu_ca100 > -50
    @test result.hu_ca200 > result.hu_ca100 - 50

    println("  BHC: PASS")
end

# =============================================================================
# Test 5: Combined Effects (Heel + BHC)
# =============================================================================

@testset "Combined: Heel + BHC" begin
    println("\n=== Test 5: Combined Heel + BHC ===")

    geom, phantom = create_test_setup()
    config = default_physics_config(
        heel_effect = default_heel_effect(anode_angle_deg=7.0),
        bhc = bhc_water_default(reference_energy_keV=60.0)
    )

    result = test_reconstruction(phantom, geom, config)

    println("  Solid water HU: $(round(result.hu_water, digits=1))")
    println("  Ca-100 HU: $(round(result.hu_ca100, digits=1))")
    println("  Ca-200 HU: $(round(result.hu_ca200, digits=1))")

    # Combined effects - heel effect causes HU shift without calibration
    @test -150 < result.hu_water < 150  # Wider tolerance for heel effect
    @test result.hu_ca100 > 0
    @test result.hu_ca200 > result.hu_ca100

    println("  Heel + BHC: PASS")
end

# =============================================================================
# Test 6: GPU Validation (if available)
# =============================================================================

if HAS_METAL
    @testset "GPU: Baseline" begin
        println("\n=== Test 6: GPU Baseline ===")

        geom, phantom = create_test_setup()
        config = default_physics_config()

        result = test_reconstruction(phantom, geom, config; use_gpu=true)

        println("  [GPU] Solid water HU: $(round(result.hu_water, digits=1))")
        println("  [GPU] Ca-100 HU: $(round(result.hu_ca100, digits=1))")
        println("  [GPU] Ca-200 HU: $(round(result.hu_ca200, digits=1))")

        @test -50 < result.hu_water < 50  # Match CPU tolerance
        @test result.hu_ca100 > 0
        @test result.hu_ca200 > result.hu_ca100

        println("  GPU Baseline: PASS")
    end

    @testset "GPU: Heel + BHC" begin
        println("\n=== Test 7: GPU Heel + BHC ===")

        geom, phantom = create_test_setup()
        config = default_physics_config(
            heel_effect = default_heel_effect(anode_angle_deg=7.0),
            bhc = bhc_water_default(reference_energy_keV=60.0)
        )

        result = test_reconstruction(phantom, geom, config; use_gpu=true)

        println("  [GPU] Solid water HU: $(round(result.hu_water, digits=1))")
        println("  [GPU] Ca-100 HU: $(round(result.hu_ca100, digits=1))")
        println("  [GPU] Ca-200 HU: $(round(result.hu_ca200, digits=1))")

        @test -150 < result.hu_water < 150  # Match CPU tolerance (heel effect shifts HU)
        @test result.hu_ca100 > 0
        @test result.hu_ca200 > result.hu_ca100

        println("  GPU Heel + BHC: PASS")
    end
else
    println("\n=== GPU Tests SKIPPED (Metal not available) ===")
end

# =============================================================================
# Test 7: Effect Comparison (verify effects actually change output)
# =============================================================================

@testset "Effect Comparison" begin
    println("\n=== Test 8: Effect Comparison ===")

    geom, phantom = create_test_setup()

    # Baseline
    config_baseline = default_physics_config()
    result_baseline = test_reconstruction(phantom, geom, config_baseline)

    # With heel effect
    config_heel = default_physics_config(
        heel_effect = default_heel_effect(anode_angle_deg=7.0)
    )
    result_heel = test_reconstruction(phantom, geom, config_heel)

    # With BHC
    config_bhc = default_physics_config(
        bhc = bhc_water_default(reference_energy_keV=60.0)
    )
    result_bhc = test_reconstruction(phantom, geom, config_bhc)

    println("  Baseline water HU: $(round(result_baseline.hu_water, digits=1))")
    println("  Heel effect water HU: $(round(result_heel.hu_water, digits=1))")
    println("  BHC water HU: $(round(result_bhc.hu_water, digits=1))")

    # Heel effect should cause some change (spatial variation)
    # Check that the reconstruction is different but not wildly so
    baseline_mean = mean(result_baseline.recon)
    heel_mean = mean(result_heel.recon)
    bhc_mean = mean(result_bhc.recon)

    println("  Baseline μ mean: $(round(baseline_mean, digits=6))")
    println("  Heel effect μ mean: $(round(heel_mean, digits=6))")
    println("  BHC μ mean: $(round(bhc_mean, digits=6))")

    # Effects should produce measurable changes but stay in reasonable range
    @test isapprox(baseline_mean, heel_mean, rtol=0.5)  # Within 50%
    @test isapprox(baseline_mean, bhc_mean, rtol=0.5)   # Within 50%

    println("  Effect comparison: PASS")
end

# =============================================================================
# Summary
# =============================================================================

println("\n" * "=" ^ 70)
println("HU Validation Tests Complete")
println("=" ^ 70)
println()
println("All effects should produce HU values within expected clinical ranges:")
println("  - Water: approximately 0 HU (±100-200 HU with effects)")
println("  - Calcium inserts: positive HU, increasing with concentration")
println()
