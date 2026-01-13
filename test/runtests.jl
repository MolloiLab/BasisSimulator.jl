using Test
using BasisSimulator
using Statistics

@testset "BasisSimulator.jl" begin

    # =========================================================================
    # PHYSICS-VALIDATED TESTS
    # These tests verify medically meaningful HU values for known materials
    # =========================================================================

    @testset "Medical Physics - Expected HU Values" begin
        # Reference values at 60 keV
        μ_water_ref = get_reference_μ_water(60.0)
        @test 0.19 < μ_water_ref < 0.22  # Water μ should be ~0.21 cm⁻¹

        # Expected HU values for Gammex materials (at 60 keV, approximate)
        # Water = 0 HU (by definition)
        # Air ≈ -1000 HU
        # Ca100 ≈ 350-400 HU
        # Ca200 ≈ 700-800 HU
        # I_10 ≈ 350-400 HU

        # Verify material HU calculations
        hu_water = validate_material_hu(:water, 60.0)
        @test abs(hu_water) < 1  # Water should be 0 HU

        hu_air = validate_material_hu(:air, 60.0)
        @test -1010 < hu_air < -990  # Air should be ~-1000 HU

        hu_ca100 = validate_material_hu(:Ca_100, 60.0)
        @test 300 < hu_ca100 < 450  # Ca100 should be ~375 HU

        hu_ca200 = validate_material_hu(:Ca_200, 60.0)
        @test 650 < hu_ca200 < 850  # Ca200 should be ~750 HU
    end

    @testset "Medical Physics - Ideal Reconstruction Accuracy" begin
        # Use n_voxels=64 for good resolution and accuracy
        phantom = create_gammex_472(n_voxels=64)
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])
        output_size = size(phantom.μ)
        μ_water_ref = get_reference_μ_water(60.0)

        # Ideal simulation (monochromatic, no effects)
        sino, recon = simulate_and_reconstruct(phantom, geom, output_size;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )

        # Use central slice only (edge slices are outside detector cone-beam coverage)
        mid_z = output_size[3] ÷ 2 + 1

        # Water region should be ~0 HU (tolerance: ±50 HU)
        water_mask_2d = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)
        water_HU = μ_to_HU(mean(recon[:, :, mid_z][water_mask_2d]), μ_water_ref)
        @test abs(water_HU) < 50

        # Calcium inserts should be higher than water
        ca100_mask_2d = phantom.mask[:, :, mid_z] .== UInt8(REGION_CA_100)
        if sum(ca100_mask_2d) > 0
            ca100_HU = μ_to_HU(mean(recon[:, :, mid_z][ca100_mask_2d]), μ_water_ref)
            @test ca100_HU > water_HU + 100  # Should be significantly higher
            @test 200 < ca100_HU < 600  # Ca100 expected ~375 HU
        end

        # Higher calcium should have higher HU
        ca200_mask_2d = phantom.mask[:, :, mid_z] .== UInt8(REGION_CA_200)
        if sum(ca200_mask_2d) > 0 && sum(ca100_mask_2d) > 0
            ca200_HU = μ_to_HU(mean(recon[:, :, mid_z][ca200_mask_2d]), μ_water_ref)
            ca100_HU = μ_to_HU(mean(recon[:, :, mid_z][ca100_mask_2d]), μ_water_ref)
            @test ca200_HU > ca100_HU  # Ca200 should be higher than Ca100
        end

        # Iodine inserts should be higher than water
        i10_mask_2d = phantom.mask[:, :, mid_z] .== UInt8(REGION_I_10_0)
        if sum(i10_mask_2d) > 0
            i10_HU = μ_to_HU(mean(recon[:, :, mid_z][i10_mask_2d]), μ_water_ref)
            @test i10_HU > water_HU + 50
        end
    end

    @testset "Medical Physics - Polychromatic Simulation" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=90, n_rows=4, n_cols=64, fov_cm=phantom.fov[1])
        output_size = size(phantom.μ)
        μ_water_ref = get_reference_μ_water(60.0)

        # Polychromatic (no detector effects for clean comparison)
        sino, recon = simulate_and_reconstruct(phantom, geom, output_size;
            polychromatic=true, kVp=120,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )

        # Check basic validity
        @test all(isfinite.(sino))
        @test all(isfinite.(recon))
        @test maximum(recon) > 0

        # Use central slice only (edge slices are outside detector cone-beam coverage)
        mid_z = output_size[3] ÷ 2 + 1

        # Water HU should still be reasonable (polychromatic causes beam hardening)
        water_mask_2d = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)
        water_HU = μ_to_HU(mean(recon[:, :, mid_z][water_mask_2d]), μ_water_ref)
        @test -150 < water_HU < 150  # Relaxed tolerance for polychromatic

        # Material ordering should still be preserved
        ca_mask_2d = phantom.mask[:, :, mid_z] .== UInt8(REGION_CA_100)
        if sum(ca_mask_2d) > 0 && sum(water_mask_2d) > 0
            @test mean(recon[:, :, mid_z][ca_mask_2d]) > mean(recon[:, :, mid_z][water_mask_2d])
        end
    end

    @testset "Medical Physics - Forward Projection Accuracy" begin
        # Test that forward projection produces correct line integrals
        phantom = create_gammex_472(n_voxels=64)
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])

        sino = forward_project_raymarching(phantom, geom)

        # Sinogram should have positive values where rays hit the phantom
        @test maximum(sino) > 0
        @test all(sino .>= 0)

        # Central ray through water should have line integral ~6.8
        # (μ_water ~0.21 × phantom_diameter ~33cm)
        center_col = geom.n_cols ÷ 2
        center_row = geom.n_rows ÷ 2
        center_sino = mean(sino[center_col, center_row, :])

        μ_water = get_reference_μ_water(60.0)
        expected_integral = μ_water * 33.0  # Approximate diameter
        @test 0.7 < center_sino / expected_integral < 1.5  # Within 50%
    end

    # =========================================================================
    # UNIFIED API TESTS
    # =========================================================================

    @testset "Unified API - simulate_sinogram" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=72, n_rows=4, n_cols=64, fov_cm=phantom.fov[1])

        # Full realistic simulation (default)
        sino_full = simulate_sinogram(phantom, geom; seed=42)
        @test size(sino_full) == (64, 4, 72)
        @test maximum(sino_full) > 0
        @test all(isfinite.(sino_full))

        # Ideal simulation (no effects)
        sino_ideal = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )
        @test size(sino_ideal) == (64, 4, 72)

        # Ideal should have lower values (no added attenuation from filters)
        @test mean(sino_ideal) < mean(sino_full)
    end

    @testset "Unified API - reconstruct" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=90, n_rows=4, n_cols=64, fov_cm=phantom.fov[1])
        output_size = size(phantom.μ)

        sino = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )

        # FDK reconstruction
        recon_fdk = reconstruct(sino, geom, output_size, phantom.fov)
        @test size(recon_fdk) == output_size
        @test all(isfinite.(recon_fdk))

        # Different kernels
        recon_soft = reconstruct(sino, geom, output_size, phantom.fov; kernel=kernel_soft())
        recon_bone = reconstruct(sino, geom, output_size, phantom.fov; kernel=kernel_bone())
        @test size(recon_soft) == output_size
        @test size(recon_bone) == output_size

        # SIRT reconstruction
        recon_sirt = reconstruct(sino, geom, output_size, phantom.fov;
            method=:sirt, n_iterations=2)
        @test size(recon_sirt) == output_size
        @test all(isfinite.(recon_sirt))

        # CGLS reconstruction
        recon_cgls = reconstruct(sino, geom, output_size, phantom.fov;
            method=:cgls, n_iterations=2)
        @test size(recon_cgls) == output_size
        @test all(isfinite.(recon_cgls))
    end

    @testset "Unified API - simulate_and_reconstruct" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=72, n_rows=4, n_cols=64, fov_cm=phantom.fov[1])
        output_size = size(phantom.μ)

        sino, recon = simulate_and_reconstruct(phantom, geom, output_size;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )

        @test size(sino) == (64, 4, 72)
        @test size(recon) == output_size
        @test all(isfinite.(sino))
        @test all(isfinite.(recon))
    end

    # =========================================================================
    # PHYSICAL EFFECT TESTS
    # =========================================================================

    @testset "Physical Effects - Filters" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])

        sino_baseline = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )

        # Flat filter adds attenuation
        sino_flat = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=DEFAULT_FLAT_FILTER,
            bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )
        @test mean(sino_flat) > mean(sino_baseline)

        # Bowtie filter adds attenuation
        sino_bowtie = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing,
            bowtie_filter=DEFAULT_BOWTIE_FILTER,
            scatter=nothing, detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )
        @test mean(sino_bowtie) > mean(sino_baseline)
    end

    @testset "Physical Effects - Scatter and Noise" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])

        sino_baseline = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )

        # Scatter changes sinogram
        sino_scatter = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing,
            scatter=DEFAULT_SCATTER_MODEL,
            detector=nothing, crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
        )
        @test sino_scatter != sino_baseline

        # Detector noise changes sinogram
        sino_noisy = simulate_sinogram(phantom, geom;
            polychromatic=false,
            flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
            detector=DEFAULT_DETECTOR_MODEL,
            crosstalk=nothing, lag=nothing,
            optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing,
            seed=42
        )
        @test sino_noisy != sino_baseline
        @test all(isfinite.(sino_noisy))
    end

    # =========================================================================
    # COMPONENT TESTS
    # =========================================================================

    @testset "Materials" begin
        @test Ca_50 isa XA.Material
        @test Ca_100 isa XA.Material
        @test I_2_0 isa XA.Material
        @test solid_water isa XA.Material

        @test get_material(:Ca_50) === Ca_50
        @test get_material(:water) isa XA.Material

        region_mats = get_region_materials()
        @test length(region_mats) == 27
    end

    @testset "Spectrum" begin
        energies, weights = load_spectrum(120)
        @test length(energies) == length(weights)
        @test length(energies) > 0
        @test all(energies .> 0)
        @test maximum(energies) <= 120

        mean_E = spectrum_mean_energy(energies, weights)
        @test 40 < mean_E < 80
    end

    @testset "HU Conversion" begin
        μ_water = get_reference_μ_water(60.0)
        @test 0.18 < μ_water < 0.22

        @test μ_to_HU(μ_water, μ_water) ≈ 0.0
        @test μ_to_HU(0.0, μ_water) ≈ -1000.0
        @test μ_to_HU(2 * μ_water, μ_water) ≈ 1000.0

        @test HU_to_μ(0.0, μ_water) ≈ μ_water
        @test HU_to_μ(-1000.0, μ_water) ≈ 0.0
    end

    @testset "Phantom" begin
        phantom = create_gammex_472(n_voxels=32)

        @test phantom isa Phantom
        @test size(phantom.μ, 1) == 32
        @test size(phantom.μ, 2) == 32

        @test sum(phantom.mask .== UInt8(REGION_SOLID_WATER)) > 0
        @test sum(phantom.mask .== UInt8(REGION_CA_100)) > 0

        stats_water = get_region_stats(phantom, REGION_SOLID_WATER)
        @test stats_water.n_voxels > 0
        @test stats_water.mean > 0
    end

    @testset "Scanner Geometry" begin
        geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=16)

        @test geom.SAD ≈ 60.0
        @test geom.SDD ≈ 100.0
        @test geom.n_angles == 36
        @test geom.n_rows == 8
        @test geom.n_cols == 16

        @test size(geom.source_positions) == (3, 36)
        @test size(geom.detector_centers) == (3, 36)
    end

    @testset "Reconstruction Kernels" begin
        k_ramp = kernel_ramp()
        k_soft = kernel_soft()
        k_bone = kernel_bone()

        @test k_ramp isa RampKernel
        @test k_soft isa SoftKernel
        @test k_bone isa BoneKernel

        n_fft = 256
        pixel_size = 0.1

        filter_ramp = create_kernel_filter(k_ramp, n_fft, pixel_size)
        filter_soft = create_kernel_filter(k_soft, n_fft, pixel_size)

        @test length(filter_ramp) == n_fft
        @test abs(filter_ramp[1]) < 1e-10  # DC = 0
    end

    @testset "Beam Hardening Correction" begin
        bhc_120 = water_bhc_120kVp()
        bhc_80 = water_bhc_80kVp()

        @test bhc_120 isa WaterBHC
        @test 0.1 < bhc_120.effective_μ < 0.3
        @test bhc_80.effective_μ > bhc_120.effective_μ

        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])
        sinogram = forward_project_raymarching(phantom, geom)

        sino_corrected = apply_water_bhc(sinogram, bhc_120)
        @test size(sino_corrected) == size(sinogram)
        @test all(isfinite.(sino_corrected))
    end

    @testset "Scanner Configurations" begin
        spec = GERevolutionApexElite()
        @test manufacturer(spec) == GE_HEALTHCARE
        @test model_name(spec) == "Revolution Apex Elite"
        @test fda_510k(spec) == "K213715"

        geom = create_geometry(spec; n_angles=180, n_rows=16, n_cols=256)
        @test geom isa CTGeometry
        @test geom.SAD ≈ 62.6
        @test geom.SDD ≈ 109.7
    end

    @testset "Helical Scanning" begin
        geom_axial = create_scan_geometry(mode=:axial, n_angles=36, n_rows=8, n_cols=32)
        @test !is_helical(geom_axial)

        geom_helical = create_scan_geometry(
            mode=:helical, n_angles=36, n_rows=8, n_cols=32,
            pitch=1.0, n_rotations=3.0, z_start=0.0
        )
        @test geom_helical.n_angles == 36 * 3
        @test is_helical(geom_helical)
    end

    @testset "Loss Functions" begin
        pred = [1.0f0, 2.0f0, 3.0f0]
        target = [1.0f0, 2.0f0, 3.0f0]
        @test mse_loss(pred, target) ≈ 0.0f0

        pred2 = [2.0f0, 3.0f0, 4.0f0]
        @test mse_loss(pred2, target) ≈ 1.0f0
        @test mae_loss(pred2, target) ≈ 1.0f0

        volume = ones(Float32, 4, 4, 4)
        @test l2_regularization(volume) ≈ 64.0f0
    end

end
