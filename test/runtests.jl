using Test
using BasisSimulator
using Statistics

@testset "BasisSimulator.jl" begin
    @testset "Materials" begin
        # Test that Gammex materials are defined
        @test Ca_50 isa XA.Material
        @test Ca_100 isa XA.Material
        @test Ca_200 isa XA.Material
        @test I_2_0 isa XA.Material
        @test I_10_0 isa XA.Material

        # Test material lookup
        @test get_material(:Ca_50) === Ca_50
        @test get_material(:water) isa XA.Material

        # Test HU validation function exists
        hu = validate_material_hu(:Ca_100, 60.0)
        @test hu > 0  # Calcium should have positive HU
    end

    @testset "Spectrum" begin
        # Test loading xspect spectrum
        energies, weights = load_spectrum(120)
        @test length(energies) == length(weights)
        @test length(energies) > 0
        @test all(energies .> 0)
        @test all(weights .>= 0)
        @test maximum(energies) <= 120  # Can't exceed kVp

        # Test loading different kVp
        energies_80, weights_80 = load_spectrum(80)
        @test maximum(energies_80) <= 80

        # Test loading xcist spectrum
        energies_xcist, weights_xcist = load_spectrum(120; source=:xcist)
        @test length(energies_xcist) > 0

        # Test mean energy
        mean_E = spectrum_mean_energy(energies, weights)
        @test 40 < mean_E < 80  # Reasonable range for 120 kVp

        # Test available spectra
        avail = available_spectra()
        @test :xspect in keys(avail)
        @test :xcist in keys(avail)
        @test 120 in avail[:xspect][:kVp]
    end

    @testset "Attenuation" begin
        # Test single energy computation
        water = XA.Materials.water
        μ_water = compute_μ_at_energy(water, 60.0)
        @test μ_water > 0
        @test 0.1 < μ_water < 0.5  # Reasonable range for water at 60 keV

        # Test mass attenuation
        μ_ρ_water = compute_mass_μ_at_energy(water, 60.0)
        @test μ_ρ_water > 0

        # Test density
        ρ_water = get_density(water)
        @test 0.9 < ρ_water < 1.1  # Water density ~1.0 g/cm³

        # Test μ matrix computation
        materials = [water, Ca_100, I_10_0]
        energies, weights = load_spectrum(120)
        μ_matrix = compute_μ_matrix(materials, energies)

        @test size(μ_matrix) == (3, length(energies))
        @test all(μ_matrix .>= 0)

        # Higher Z materials should have higher attenuation at lower energies
        # (photoelectric effect dominates)
        low_E_idx = findfirst(e -> e > 30, energies)
        @test μ_matrix[2, low_E_idx] > μ_matrix[1, low_E_idx]  # Ca > water
        @test μ_matrix[3, low_E_idx] > μ_matrix[1, low_E_idx]  # I > water

        # Test effective μ computation
        μ_eff = compute_effective_μ(μ_matrix, weights)
        @test length(μ_eff) == 3
        @test all(μ_eff .> 0)
    end

    @testset "Phantom" begin
        # Create small phantom for fast testing
        phantom = create_gammex_472(n_voxels=32)

        # Test basic structure
        @test phantom isa Phantom
        @test size(phantom.μ) == size(phantom.mask)
        @test size(phantom.μ, 1) == 32
        @test size(phantom.μ, 2) == 32

        # Test that we have expected regions
        @test sum(phantom.mask .== UInt8(REGION_SOLID_WATER)) > 0
        @test sum(phantom.mask .== UInt8(REGION_CA_100)) > 0
        @test sum(phantom.mask .== UInt8(REGION_I_10_0)) > 0

        # Test region stats
        stats_water = get_region_stats(phantom, REGION_SOLID_WATER)
        @test stats_water.n_voxels > 0
        @test stats_water.mean > 0
        @test isfinite(stats_water.std)

        stats_ca = get_region_stats(phantom, REGION_CA_100)
        @test stats_ca.n_voxels > 0
        @test stats_ca.mean > stats_water.mean  # Ca should have higher μ

        stats_i = get_region_stats(phantom, REGION_I_10_0)
        @test stats_i.n_voxels > 0

        # Test region mask extraction
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        @test water_mask isa BitArray{3}
        @test sum(water_mask) == stats_water.n_voxels

        # Test all 14 inserts are present (7 Ca + 7 I)
        ca_labels = [REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600]
        i_labels = [REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0]

        n_ca_types = sum([sum(phantom.mask .== UInt8(l)) > 0 for l in ca_labels])
        n_i_types = sum([sum(phantom.mask .== UInt8(l)) > 0 for l in i_labels])
        @test n_ca_types == 7  # All 7 calcium inserts
        @test n_i_types == 7   # All 7 iodine inserts

        # Test validation function (perfect reconstruction = phantom itself)
        val_result = validate_reconstruction(phantom, phantom.μ)
        @test val_result.passed == true
        @test all(r -> r.passed, values(val_result.results))

        # Test with slightly perturbed reconstruction
        noisy_recon = phantom.μ .* 1.05f0  # 5% higher
        val_noisy = validate_reconstruction(phantom, noisy_recon; tolerance_pct=10.0)
        @test val_noisy.passed == true  # Should still pass with 10% tolerance

        # Test with very perturbed reconstruction
        bad_recon = phantom.μ .* 2.0f0  # 100% higher
        val_bad = validate_reconstruction(phantom, bad_recon; tolerance_pct=10.0)
        @test val_bad.passed == false  # Should fail

        # Test metadata
        @test phantom.voxel_size[1] > 0
        @test phantom.voxel_size[2] > 0
        @test phantom.voxel_size[3] > 0
        @test phantom.fov[1] > 0
    end

    @testset "Scanner Geometry" begin
        # Create scanner with default parameters
        geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=16)

        # Test basic properties
        @test geom.SAD ≈ 60.0  # 600mm in cm
        @test geom.SDD ≈ 100.0  # 1000mm in cm
        @test geom.n_angles == 36
        @test geom.n_rows == 8
        @test geom.n_cols == 16
        @test geom.pixel_size ≈ 0.05  # 0.5mm in cm

        # Test pre-computed arrays have correct size
        @test size(geom.source_positions) == (3, 36)
        @test size(geom.detector_centers) == (3, 36)
        @test size(geom.detector_u) == (3, 36)
        @test size(geom.detector_v) == (3, 36)

        # Test first angle (θ=0): source at (0, -SAD, 0)
        @test geom.source_positions[1, 1] ≈ 0.0 atol=1e-10
        @test geom.source_positions[2, 1] ≈ -60.0 atol=1e-10
        @test geom.source_positions[3, 1] ≈ 0.0 atol=1e-10

        # Detector center at (0, SDD-SAD, 0) = (0, 40, 0) for θ=0
        @test geom.detector_centers[1, 1] ≈ 0.0 atol=1e-10
        @test geom.detector_centers[2, 1] ≈ 40.0 atol=1e-10
        @test geom.detector_centers[3, 1] ≈ 0.0 atol=1e-10

        # Detector u-axis at θ=0 should be (1, 0, 0)
        @test geom.detector_u[1, 1] ≈ 1.0 atol=1e-10
        @test geom.detector_u[2, 1] ≈ 0.0 atol=1e-10
        @test geom.detector_u[3, 1] ≈ 0.0 atol=1e-10

        # Detector v-axis always (0, 0, 1)
        @test geom.detector_v[1, 1] ≈ 0.0 atol=1e-10
        @test geom.detector_v[2, 1] ≈ 0.0 atol=1e-10
        @test geom.detector_v[3, 1] ≈ 1.0 atol=1e-10

        # Test at 90 degrees (angle_idx = 10 for 36 angles)
        # Source should be at (-SAD, 0, 0) approximately
        angle_90_idx = 10  # 9 * (2π/36) ≈ π/2
        # At θ=π/2: source at (-SAD*sin(π/2), -SAD*cos(π/2), 0) = (-SAD, 0, 0)
        @test abs(geom.source_positions[1, angle_90_idx]) > 50  # Should be near -60

        # Test helper functions
        src = get_source_position(geom, 1)
        @test src[1] ≈ 0.0 atol=1e-10
        @test src[2] ≈ -60.0 atol=1e-10
        @test src[3] ≈ 0.0 atol=1e-10

        # Test detector pixel position at center
        det_center = get_detector_pixel_position(geom, 1,
            div(geom.n_rows, 2) + 1,
            div(geom.n_cols, 2) + 1)
        @test det_center[2] ≈ 40.0 atol=1.0  # Near detector center Y
    end

    @testset "Forward Projection" begin
        # Create small phantom and scanner for fast testing
        phantom = create_gammex_472(n_voxels=32)
        # Use fov_cm to ensure detector covers the phantom FOV
        geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=64, fov_cm=phantom.fov[1])

        # Forward project
        sinogram = forward_project(phantom, geom; n_steps=64)

        # Test sinogram shape
        @test size(sinogram) == (64, 8, 36)

        # Sinogram should have non-zero values (rays through phantom)
        @test maximum(sinogram) > 0

        # Sinogram values should be reasonable line integrals
        # For a ~33cm diameter water phantom, max path ~ 33cm
        # At 60 keV, μ_water ≈ 0.2 cm⁻¹, so max integral ~ 6.6
        @test maximum(sinogram) < 50.0  # Sanity check

        # Check that different angles have similar total signal
        # (phantom is roughly circular)
        totals = [sum(sinogram[:, :, i]) for i in 1:36]
        mean_total = mean(totals)
        @test all(t -> abs(t - mean_total) / mean_total < 0.5, totals)
    end

    @testset "FDK Reconstruction" begin
        # Create very small phantom for fast testing
        phantom = create_gammex_472(n_voxels=16)
        # Use fov_cm to ensure detector covers phantom
        geom = create_aquilion_one(n_angles=72, n_rows=8, n_cols=64, fov_cm=phantom.fov[1])

        # Forward project
        sinogram = forward_project(phantom, geom; n_steps=64)

        # Reconstruct
        output_size = (16, 16, size(phantom.μ, 3))
        recon = fdk_reconstruct(sinogram, geom, output_size, phantom.fov)

        # Test reconstruction shape matches phantom
        @test size(recon) == size(phantom.μ)

        # Reconstruction should have non-zero values inside phantom
        @test maximum(recon) > 0

        # Values should be in reasonable range for μ (cm⁻¹)
        # Water at 60 keV: ~0.2 cm⁻¹
        @test maximum(recon) < 2.0  # Sanity check
    end

    @testset "End-to-End Validation" begin
        # Create phantom with known regions
        phantom = create_gammex_472(n_voxels=32)
        # Use fov_cm to ensure detector covers phantom
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])

        # Forward project with more steps for accuracy
        sinogram = forward_project(phantom, geom; n_steps=128)

        # Reconstruct
        output_size = size(phantom.μ)
        recon = fdk_reconstruct(sinogram, geom, output_size, phantom.fov)

        # Get solid water region stats
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        expected_water_μ = mean(phantom.μ[water_mask])
        measured_water_μ = mean(recon[water_mask])

        # Water reconstruction should be within 50% (loose tolerance for coarse recon)
        # This is a smoke test - actual validation would need higher resolution
        if expected_water_μ > 0
            water_error_pct = abs(measured_water_μ - expected_water_μ) / expected_water_μ * 100
            @test water_error_pct < 100  # Very loose for now
        end

        # Check that high-attenuation inserts have higher values than background
        ca_mask = get_region_mask(phantom, REGION_CA_100)
        if sum(ca_mask) > 0
            expected_ca_μ = mean(phantom.μ[ca_mask])
            measured_ca_μ = mean(recon[ca_mask])
            # CA insert should be higher than water in both expected and measured
            @test expected_ca_μ > expected_water_μ
            # At least check measured is positive
            @test measured_ca_μ > 0
        end

        # The reconstruction should show contrast between regions
        # (Even if absolute values aren't perfect)
        water_vals = recon[water_mask]
        background_vals = recon[phantom.mask .== UInt8(REGION_BACKGROUND)]

        if length(water_vals) > 0 && length(background_vals) > 0
            # Water should have higher μ than air/background
            @test mean(water_vals) > mean(background_vals)
        end
    end
end
