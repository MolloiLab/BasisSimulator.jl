# =============================================================================
# BasisSimulator.jl Test Suite
# =============================================================================
#
# Tests run on CPU by default. If Metal.jl is available and functional,
# GPU tests are also run automatically.
#
# =============================================================================

using Test
using BasisSimulator
using Statistics

# =============================================================================
# GPU Detection
# =============================================================================

const HAS_GPU = try
    using Metal
    Metal.functional()
catch
    false
end

if HAS_GPU
    using Metal
    println("GPU detected: ", Metal.current_device())
    println("Running tests on CPU and GPU")
else
    println("No GPU detected - running CPU tests only")
end

# =============================================================================
# Test Helpers
# =============================================================================

# Create test data on appropriate backend
function test_array(data)
    HAS_GPU ? MtlArray(data) : data
end

# Small phantom/geometry for fast tests
function small_test_setup()
    phantom = create_gammex_472(n_voxels=32, n_slices=8, fov_cm=35.0, z_cm=4.0)
    geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=64, fov_cm=35.0, z_cm=4.0)
    return phantom, geom
end

# =============================================================================
# Core Tests
# =============================================================================

@testset "BasisSimulator.jl" begin

    # -------------------------------------------------------------------------
    # Materials and Spectrum
    # -------------------------------------------------------------------------
    @testset "Materials" begin
        @test Ca_50 isa XA.Material
        @test Ca_100 isa XA.Material
        @test solid_water isa XA.Material

        materials = get_region_materials()
        @test length(materials) == 27
    end

    @testset "Spectrum" begin
        energies, weights = load_spectrum(120)
        @test length(energies) == length(weights)
        @test all(energies .> 0)
        @test maximum(energies) <= 120

        energies_ds, weights_ds = downsample_spectrum(energies, weights, 30)
        @test length(energies_ds) == 30
    end

    @testset "HU Conversion" begin
        μ_water = get_reference_μ_water(60.0)
        @test 0.18 < μ_water < 0.22

        @test μ_to_HU(μ_water, μ_water) ≈ 0.0
        @test μ_to_HU(0.0, μ_water) ≈ -1000.0
    end

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------
    @testset "Phantom" begin
        phantom = create_gammex_472(n_voxels=32)
        @test phantom isa Phantom
        @test size(phantom.μ, 1) == 32
        @test sum(phantom.mask .== UInt8(REGION_SOLID_WATER)) > 0
    end

    @testset "Scanner Geometry" begin
        geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=16)
        @test geom.SAD ≈ 60.0
        @test geom.SDD ≈ 100.0
        @test geom.n_angles == 36
        @test size(geom.source_positions) == (3, 36)
    end

    # -------------------------------------------------------------------------
    # Forward Projection (CPU)
    # -------------------------------------------------------------------------
    @testset "Forward Projection - CPU" begin
        phantom, geom = small_test_setup()

        # Monochromatic
        sino = forward_project(Float32.(phantom.μ), geom)
        @test size(sino) == (64, 8, 36)
        @test all(isfinite.(sino))
        @test maximum(sino) > 0
    end

    @testset "Polychromatic Forward Projection - CPU" begin
        phantom, geom = small_test_setup()
        energies, weights = load_spectrum(120)
        energies, weights = downsample_spectrum(energies, weights, 10)
        materials = get_region_materials()

        sino = forward_project(phantom.mask, geom;
            energies=energies, weights=weights, materials=materials)

        @test size(sino) == (64, 8, 36)
        @test all(isfinite.(sino))
        @test maximum(sino) > 0
    end

    # -------------------------------------------------------------------------
    # Reconstruction (CPU)
    # -------------------------------------------------------------------------
    @testset "FDK Reconstruction - CPU" begin
        phantom, geom = small_test_setup()
        sino = forward_project(Float32.(phantom.μ), geom)
        recon = fdk_reconstruct(sino, geom, size(phantom.μ))

        @test size(recon) == size(phantom.μ)
        @test all(isfinite.(recon))
    end

    # -------------------------------------------------------------------------
    # Physics Configuration
    # -------------------------------------------------------------------------
    @testset "Physics Config" begin
        # Default config - all nothing
        config = default_physics_config()
        info = get_physics_config_info(config)
        @test info.n_enabled == 0

        # Full config - all enabled
        config_full = full_physics_config()
        info_full = get_physics_config_info(config_full)
        @test info_full.n_enabled == 13

        # Realistic config
        config_real = realistic_physics_config()
        info_real = get_physics_config_info(config_real)
        @test info_real.n_enabled > 0
    end

    @testset "Physics Effects - CPU" begin
        phantom, geom = small_test_setup()

        # With some physics effects
        physics = default_physics_config(
            scatter = default_scatter_model(),
            noise = default_detector_model(I0=1e6, seed=42)
        )

        sino = forward_project(Float32.(phantom.μ), geom; physics=physics)
        @test all(isfinite.(sino))
    end

    # -------------------------------------------------------------------------
    # HU Validation (CPU)
    # -------------------------------------------------------------------------
    @testset "HU Validation - CPU" begin
        phantom, geom = small_test_setup()
        μ_water = get_reference_μ_water(60.0)

        sino = forward_project(Float32.(phantom.μ), geom)
        recon = fdk_reconstruct(sino, geom, size(phantom.μ))

        mid_z = size(recon, 3) ÷ 2 + 1
        water_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)

        if sum(water_mask) > 0
            water_hu = μ_to_HU(mean(recon[:, :, mid_z][water_mask]), μ_water)
            @test -100 < water_hu < 100  # Water should be ~0 HU
        end
    end

    # -------------------------------------------------------------------------
    # GPU Tests (only if Metal available)
    # -------------------------------------------------------------------------
    if HAS_GPU
        @testset "Forward Projection - GPU" begin
            phantom, geom = small_test_setup()
            mask_gpu = MtlArray(phantom.mask)

            energies, weights = load_spectrum(120)
            energies, weights = downsample_spectrum(energies, weights, 10)
            materials = get_region_materials()

            sino_gpu = forward_project(mask_gpu, geom;
                energies=energies, weights=weights, materials=materials)

            @test sino_gpu isa MtlArray
            @test size(sino_gpu) == (64, 8, 36)

            sino_cpu = Array(sino_gpu)
            @test all(isfinite.(sino_cpu))
        end

        @testset "FDK Reconstruction - GPU" begin
            phantom, geom = small_test_setup()
            mask_gpu = MtlArray(phantom.mask)

            energies, weights = load_spectrum(120)
            energies, weights = downsample_spectrum(energies, weights, 10)
            materials = get_region_materials()

            sino_gpu = forward_project(mask_gpu, geom;
                energies=energies, weights=weights, materials=materials)

            recon_gpu = fdk_reconstruct(sino_gpu, geom, size(phantom.μ))

            @test recon_gpu isa MtlArray
            @test size(recon_gpu) == size(phantom.μ)

            recon_cpu = Array(recon_gpu)
            @test all(isfinite.(recon_cpu))
        end

        @testset "Full Physics Pipeline - GPU" begin
            phantom, geom = small_test_setup()
            mask_gpu = MtlArray(phantom.mask)

            energies, weights = load_spectrum(120)
            energies, weights = downsample_spectrum(energies, weights, 10)
            materials = get_region_materials()

            # Full physics (excluding broken DAS for now)
            physics = default_physics_config(
                fill_factor = fill_factor_standard(),
                flat_filter = flat_filter_al(3.0),
                bowtie_filter = bowtie_filter_large_body(),
                scatter = default_scatter_model(),
                crosstalk = crosstalk_medium(),
                noise = default_detector_model(I0=1e6, seed=42),
                bhc = bhc_water_default(reference_energy_keV=65.0),
                energy_keV = 65.0,
                noise_seed = 42
            )

            sino_gpu = forward_project(mask_gpu, geom;
                energies=energies, weights=weights, materials=materials,
                physics=physics)

            @test sino_gpu isa MtlArray
            sino_cpu = Array(sino_gpu)
            @test all(isfinite.(sino_cpu))

            # Reconstruct and check HU
            recon_gpu = fdk_reconstruct(sino_gpu, geom, size(phantom.μ))
            recon_cpu = Array(recon_gpu)

            μ_water = get_effective_μ_water_kVp(120)
            recon_hu = 1000f0 .* (recon_cpu .- μ_water) ./ μ_water

            mid_z = size(recon_hu, 3) ÷ 2 + 1
            water_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)

            if sum(water_mask) > 0
                water_hu = mean(recon_hu[:, :, mid_z][water_mask])
                @test -200 < water_hu < 200  # Relaxed for full physics
            end
        end
    end

end

println("\nTests complete!")
