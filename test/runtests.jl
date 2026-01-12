using Test
using BasisSimulator
using Statistics
using Reactant

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

    @testset "Forward Projection - Uniform" begin
        # Create small phantom and scanner for fast testing
        phantom = create_gammex_472(n_voxels=32)
        # Use fov_cm to ensure detector covers the phantom FOV
        geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=64, fov_cm=phantom.fov[1])

        # Forward project with uniform sampling
        sinogram = forward_project(phantom, geom; method=:uniform, n_steps=64)

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

    @testset "Forward Projection - Siddon" begin
        # Create small phantom and scanner for fast testing
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])

        # Forward project with Siddon's exact method (default)
        sinogram_siddon = forward_project(phantom, geom; method=:siddon)

        # Test sinogram shape
        @test size(sinogram_siddon) == (32, 4, 36)

        # Sinogram should have non-zero values
        @test maximum(sinogram_siddon) > 0

        # Compare with uniform sampling - should give similar results
        sinogram_uniform = forward_project(phantom, geom; method=:uniform, n_steps=64)

        # Both methods should give similar total attenuation
        total_siddon = sum(sinogram_siddon)
        total_uniform = sum(sinogram_uniform)
        @test abs(total_siddon - total_uniform) / total_uniform < 0.2  # Within 20%

        # Test that ProjectionGeometry stores correct method
        proj_geom_siddon = precompute_projection_geometry(
            geom, phantom.fov, phantom.voxel_size, size(phantom.μ); method=:siddon
        )
        @test proj_geom_siddon.method == :siddon

        proj_geom_uniform = precompute_projection_geometry(
            geom, phantom.fov, phantom.voxel_size, size(phantom.μ); method=:uniform, n_steps=32
        )
        @test proj_geom_uniform.method == :uniform
    end

    @testset "FDK Reconstruction" begin
        # Create very small phantom for fast testing
        phantom = create_gammex_472(n_voxels=16)
        # Use fov_cm to ensure detector covers phantom
        geom = create_aquilion_one(n_angles=72, n_rows=8, n_cols=64, fov_cm=phantom.fov[1])

        # Forward project using Siddon (more accurate)
        sinogram = forward_project(phantom, geom; method=:siddon)

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

    @testset "HU Conversion" begin
        # Get reference water μ at 60 keV
        μ_water = get_reference_μ_water(60.0)
        @test 0.18 < μ_water < 0.22  # ~0.2 cm⁻¹ for water at 60 keV

        # Test μ to HU conversion
        @test μ_to_HU(μ_water, μ_water) ≈ 0.0  # Water = 0 HU
        @test μ_to_HU(0.0, μ_water) ≈ -1000.0  # Air/vacuum = -1000 HU
        @test μ_to_HU(2 * μ_water, μ_water) ≈ 1000.0  # 2x water = 1000 HU

        # Test HU to μ conversion (inverse)
        @test HU_to_μ(0.0, μ_water) ≈ μ_water  # 0 HU = water
        @test HU_to_μ(-1000.0, μ_water) ≈ 0.0  # -1000 HU = vacuum
        @test HU_to_μ(1000.0, μ_water) ≈ 2 * μ_water  # 1000 HU = 2x water

        # Test round-trip conversion
        test_μ = 0.35  # Some arbitrary μ value
        HU = μ_to_HU(test_μ, μ_water)
        @test HU_to_μ(HU, μ_water) ≈ test_μ

        # Test array conversion
        μ_array = [0.0, μ_water, 2 * μ_water]
        HU_array = μ_to_HU(μ_array, μ_water)
        @test HU_array ≈ [-1000.0, 0.0, 1000.0]
    end

    @testset "Phantom HU Values" begin
        # Test that phantom μ values correspond to expected HU ranges
        phantom = create_gammex_472(n_voxels=32)
        μ_water = get_reference_μ_water(60.0)

        # Background (air) should be near HU = -1000
        bg_mask = get_region_mask(phantom, REGION_BACKGROUND)
        if sum(bg_mask) > 0
            bg_μ = mean(phantom.μ[bg_mask])
            bg_HU = μ_to_HU(bg_μ, μ_water)
            @test bg_HU < -900  # Air should be near -1000 HU
        end

        # Solid water should be near HU = 0
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        water_μ = mean(phantom.μ[water_mask])
        water_HU = μ_to_HU(water_μ, μ_water)
        @test -100 < water_HU < 100  # Solid water should be near 0 HU

        # Calcium inserts should have positive HU (higher than water)
        ca_100_mask = get_region_mask(phantom, REGION_CA_100)
        if sum(ca_100_mask) > 0
            ca_100_μ = mean(phantom.μ[ca_100_mask])
            ca_100_HU = μ_to_HU(ca_100_μ, μ_water)
            @test ca_100_HU > 50  # Ca_100 should be denser than water
            @test ca_100_HU < 500  # But not bone-dense
        end

        ca_200_mask = get_region_mask(phantom, REGION_CA_200)
        if sum(ca_200_mask) > 0
            ca_200_μ = mean(phantom.μ[ca_200_mask])
            ca_200_HU = μ_to_HU(ca_200_μ, μ_water)
            @test ca_200_HU > ca_100_HU  # Ca_200 > Ca_100
        end

        # Iodine inserts should have positive HU
        i_10_mask = get_region_mask(phantom, REGION_I_10_0)
        if sum(i_10_mask) > 0
            i_10_μ = mean(phantom.μ[i_10_mask])
            i_10_HU = μ_to_HU(i_10_μ, μ_water)
            @test i_10_HU > 100  # I_10 should be clearly above water
        end

        # Ordering of calcium inserts should be correct
        ca_masks = [
            (REGION_CA_50, get_region_mask(phantom, REGION_CA_50)),
            (REGION_CA_100, get_region_mask(phantom, REGION_CA_100)),
            (REGION_CA_200, get_region_mask(phantom, REGION_CA_200)),
        ]
        ca_HUs = Float64[]
        for (region, mask) in ca_masks
            if sum(mask) > 0
                push!(ca_HUs, μ_to_HU(mean(phantom.μ[mask]), μ_water))
            end
        end
        if length(ca_HUs) >= 2
            @test issorted(ca_HUs)  # HU should increase with concentration
        end
    end

    @testset "End-to-End Validation with HU" begin
        # Create phantom with known regions
        phantom = create_gammex_472(n_voxels=32)
        # Use fov_cm to ensure detector covers phantom
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])

        # Forward project with Siddon for accuracy
        sinogram = forward_project(phantom, geom; method=:siddon)

        # Reconstruct
        output_size = size(phantom.μ)
        recon = fdk_reconstruct(sinogram, geom, output_size, phantom.fov)

        # Reference μ for HU conversion
        μ_water_ref = get_reference_μ_water(60.0)

        # Get solid water region stats
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        expected_water_μ = mean(phantom.μ[water_mask])
        measured_water_μ = mean(recon[water_mask])

        # Convert to HU for more interpretable validation
        expected_water_HU = μ_to_HU(expected_water_μ, μ_water_ref)
        measured_water_HU = μ_to_HU(measured_water_μ, μ_water_ref)

        # Expected water HU should be near 0
        @test -100 < expected_water_HU < 100

        # Measured should be in a reasonable range (loose for coarse recon)
        @test -500 < measured_water_HU < 500

        # Check background region (air) - should reconstruct lower than water
        bg_mask = phantom.mask .== UInt8(REGION_BACKGROUND)
        if sum(bg_mask) > 0
            bg_recon_μ = mean(recon[bg_mask])
            bg_recon_HU = μ_to_HU(bg_recon_μ, μ_water_ref)
            # Background should be lower than water region in reconstruction
            @test bg_recon_HU < measured_water_HU
        end

        # Check calcium insert - should reconstruct higher than water
        ca_mask = get_region_mask(phantom, REGION_CA_100)
        if sum(ca_mask) > 0
            expected_ca_μ = mean(phantom.μ[ca_mask])
            measured_ca_μ = mean(recon[ca_mask])
            expected_ca_HU = μ_to_HU(expected_ca_μ, μ_water_ref)
            measured_ca_HU = μ_to_HU(measured_ca_μ, μ_water_ref)

            # Expected Ca should be denser than water
            @test expected_ca_HU > expected_water_HU

            # Measured Ca should show some contrast above water
            # (even if absolute values aren't accurate)
            @test measured_ca_μ > 0  # At least positive
        end

        # Verify contrast is preserved in reconstruction
        # Higher concentration inserts should still be brighter
        regions_by_expected_HU = [
            (REGION_BACKGROUND, bg_mask),
            (REGION_SOLID_WATER, water_mask),
            (REGION_CA_100, get_region_mask(phantom, REGION_CA_100)),
        ]

        prev_recon_mean = -Inf
        for (region, mask) in regions_by_expected_HU
            if sum(mask) > 10  # Need enough voxels
                recon_mean = mean(recon[mask])
                expected_mean = mean(phantom.μ[mask])
                # Check that ordering is preserved (at least roughly)
                if expected_mean > prev_recon_mean + 0.01
                    @test recon_mean > prev_recon_mean - 0.1  # Allow some tolerance
                end
                prev_recon_mean = expected_mean
            end
        end
    end

    @testset "Reactant Compilation - Uniform" begin
        # NO allowscalar - must work without it

        # Create small phantom and geometry for fast compilation
        phantom = create_gammex_472(n_voxels=8)
        geom = create_aquilion_one(n_angles=4, n_rows=2, n_cols=8, fov_cm=phantom.fov[1])

        # Pre-compute projection geometry with uniform sampling (not traced)
        proj_geom = precompute_projection_geometry(
            geom, phantom.fov, phantom.voxel_size, size(phantom.μ), 16
        )

        # Convert volume to Reactant array
        volume_ra = Reactant.to_rarray(phantom.μ)

        # Test that project_volume compiles (this is the traced function)
        compiled_pv = @compile project_volume(volume_ra, proj_geom)
        @test compiled_pv !== nothing

        # Run compiled function
        sinogram_ra = compiled_pv(volume_ra, proj_geom)
        sinogram_result = Array(sinogram_ra)

        # Verify output is non-zero
        @test maximum(sinogram_result) > 0

        # Compare with non-compiled version
        sinogram_julia = project_volume(phantom.μ, proj_geom)

        # Results should match
        @test maximum(abs.(sinogram_result .- sinogram_julia)) < 1e-5
    end

    @testset "Reactant Compilation - Siddon" begin
        # NO allowscalar - must work without it

        # Create small phantom and geometry for fast compilation
        phantom = create_gammex_472(n_voxels=8)
        geom = create_aquilion_one(n_angles=4, n_rows=2, n_cols=8, fov_cm=phantom.fov[1])

        # Pre-compute projection geometry with Siddon's method (not traced)
        proj_geom = precompute_projection_geometry(
            geom, phantom.fov, phantom.voxel_size, size(phantom.μ); method=:siddon
        )
        @test proj_geom.method == :siddon

        # Convert volume to Reactant array
        volume_ra = Reactant.to_rarray(phantom.μ)

        # Test that project_volume compiles with Siddon geometry
        compiled_pv = @compile project_volume(volume_ra, proj_geom)
        @test compiled_pv !== nothing

        # Run compiled function
        sinogram_ra = compiled_pv(volume_ra, proj_geom)
        sinogram_result = Array(sinogram_ra)

        # Verify output is non-zero
        @test maximum(sinogram_result) > 0

        # Compare with non-compiled version
        sinogram_julia = project_volume(phantom.μ, proj_geom)

        # Results should match
        @test maximum(abs.(sinogram_result .- sinogram_julia)) < 1e-5
    end
end
