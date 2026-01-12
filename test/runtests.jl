using Test
using BasisSimulator

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
end
