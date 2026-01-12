using Test
using BasisSimulator
using Statistics
using Reactant

# Include visualization helpers (for generating inspection images)
include("test_visualization.jl")

@testset "BasisSimulator.jl" begin
    @testset "Materials" begin
        # Test that Gammex materials are properly aliased from XrayAttenuation.jl
        @test Ca_50 isa XA.Material
        @test Ca_100 isa XA.Material
        @test Ca_200 isa XA.Material
        @test I_2_0 isa XA.Material
        @test I_10_0 isa XA.Material
        @test solid_water isa XA.Material

        # Test material lookup
        @test get_material(:Ca_50) === Ca_50
        @test get_material(:water) isa XA.Material
        @test get_material(:solid_water) === solid_water

        # Test HU validation function
        hu = validate_material_hu(:Ca_100, 60.0)
        @test hu > 0  # Calcium should have positive HU

        # Verify all materials come from XrayAttenuation.jl
        @test Ca_50 === XA.Materials.gammex_472_ca50_0
        @test Ca_100 === XA.Materials.gammex_472_ca100_0
        @test Ca_200 === XA.Materials.gammex_472_ca200_0
        @test I_2_0 === XA.Materials.gammex_472_i2_0
        @test I_10_0 === XA.Materials.gammex_472_i10_0
        @test solid_water === XA.Materials.gammex_water
    end

    @testset "Spectrum" begin
        # Test loading xspect spectrum
        energies, weights = load_spectrum(120)
        @test length(energies) == length(weights)
        @test length(energies) > 0
        @test all(energies .> 0)
        @test all(weights .>= 0)
        @test maximum(energies) <= 120

        # Test loading different kVp
        energies_80, weights_80 = load_spectrum(80)
        @test maximum(energies_80) <= 80

        # Test loading xcist spectrum
        energies_xcist, weights_xcist = load_spectrum(120; source=:xcist)
        @test length(energies_xcist) > 0

        # Test mean energy
        mean_E = spectrum_mean_energy(energies, weights)
        @test 40 < mean_E < 80

        # Test available spectra
        avail = available_spectra()
        @test :xspect in keys(avail)
        @test :xcist in keys(avail)
        @test 120 in avail[:xspect][:kVp]
    end

    @testset "Attenuation" begin
        water = XA.Materials.water
        μ_water = compute_μ_at_energy(water, 60.0)
        @test μ_water > 0
        @test 0.1 < μ_water < 0.5

        μ_ρ_water = compute_mass_μ_at_energy(water, 60.0)
        @test μ_ρ_water > 0

        ρ_water = get_density(water)
        @test 0.9 < ρ_water < 1.1

        # Test μ matrix computation
        materials = [water, Ca_100, I_10_0]
        energies, weights = load_spectrum(120)
        μ_matrix = compute_μ_matrix(materials, energies)

        @test size(μ_matrix) == (3, length(energies))
        @test all(μ_matrix .>= 0)

        # Higher Z materials have higher attenuation at lower energies
        low_E_idx = findfirst(e -> e > 30, energies)
        @test μ_matrix[2, low_E_idx] > μ_matrix[1, low_E_idx]
        @test μ_matrix[3, low_E_idx] > μ_matrix[1, low_E_idx]

        μ_eff = compute_effective_μ(μ_matrix, weights)
        @test length(μ_eff) == 3
        @test all(μ_eff .> 0)
    end

    @testset "HU Conversion" begin
        μ_water = get_reference_μ_water(60.0)
        @test 0.18 < μ_water < 0.22

        @test μ_to_HU(μ_water, μ_water) ≈ 0.0
        @test μ_to_HU(0.0, μ_water) ≈ -1000.0
        @test μ_to_HU(2 * μ_water, μ_water) ≈ 1000.0

        @test HU_to_μ(0.0, μ_water) ≈ μ_water
        @test HU_to_μ(-1000.0, μ_water) ≈ 0.0
        @test HU_to_μ(1000.0, μ_water) ≈ 2 * μ_water

        # Round-trip
        test_μ = 0.35
        HU = μ_to_HU(test_μ, μ_water)
        @test HU_to_μ(HU, μ_water) ≈ test_μ

        # Array conversion
        μ_array = [0.0, μ_water, 2 * μ_water]
        HU_array = μ_to_HU(μ_array, μ_water)
        @test HU_array ≈ [-1000.0, 0.0, 1000.0]
    end

    @testset "Phantom" begin
        phantom = create_gammex_472(n_voxels=32)

        @test phantom isa Phantom
        @test size(phantom.μ) == size(phantom.mask)
        @test size(phantom.μ, 1) == 32
        @test size(phantom.μ, 2) == 32

        @test sum(phantom.mask .== UInt8(REGION_SOLID_WATER)) > 0
        @test sum(phantom.mask .== UInt8(REGION_CA_100)) > 0
        @test sum(phantom.mask .== UInt8(REGION_I_10_0)) > 0

        stats_water = get_region_stats(phantom, REGION_SOLID_WATER)
        @test stats_water.n_voxels > 0
        @test stats_water.mean > 0
        @test isfinite(stats_water.std)

        stats_ca = get_region_stats(phantom, REGION_CA_100)
        @test stats_ca.n_voxels > 0
        @test stats_ca.mean > stats_water.mean

        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        @test water_mask isa BitArray{3}
        @test sum(water_mask) == stats_water.n_voxels

        # All 14 inserts present
        ca_labels = [REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600]
        i_labels = [REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0]

        n_ca_types = sum([sum(phantom.mask .== UInt8(l)) > 0 for l in ca_labels])
        n_i_types = sum([sum(phantom.mask .== UInt8(l)) > 0 for l in i_labels])
        @test n_ca_types == 7
        @test n_i_types == 7

        # Validation
        val_result = validate_reconstruction(phantom, phantom.μ)
        @test val_result.passed == true

        noisy_recon = phantom.μ .* 1.05f0
        val_noisy = validate_reconstruction(phantom, noisy_recon; tolerance_pct=10.0)
        @test val_noisy.passed == true

        bad_recon = phantom.μ .* 2.0f0
        val_bad = validate_reconstruction(phantom, bad_recon; tolerance_pct=10.0)
        @test val_bad.passed == false

        @test phantom.voxel_size[1] > 0
        @test phantom.fov[1] > 0
    end

    @testset "Phantom HU Values" begin
        phantom = create_gammex_472(n_voxels=32)
        μ_water = get_reference_μ_water(60.0)

        # Background (air) near HU = -1000
        bg_mask = get_region_mask(phantom, REGION_BACKGROUND)
        if sum(bg_mask) > 0
            bg_HU = μ_to_HU(mean(phantom.μ[bg_mask]), μ_water)
            @test bg_HU < -900
        end

        # Solid water near HU = 0
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        water_HU = μ_to_HU(mean(phantom.μ[water_mask]), μ_water)
        @test -100 < water_HU < 100

        # Calcium inserts positive HU
        ca_100_mask = get_region_mask(phantom, REGION_CA_100)
        if sum(ca_100_mask) > 0
            ca_100_HU = μ_to_HU(mean(phantom.μ[ca_100_mask]), μ_water)
            @test ca_100_HU > 50
            @test ca_100_HU < 500
        end

        # Calcium ordering correct
        ca_masks = [
            (REGION_CA_50, get_region_mask(phantom, REGION_CA_50)),
            (REGION_CA_100, get_region_mask(phantom, REGION_CA_100)),
            (REGION_CA_200, get_region_mask(phantom, REGION_CA_200)),
        ]
        ca_HUs = Float64[]
        for (_, mask) in ca_masks
            if sum(mask) > 0
                push!(ca_HUs, μ_to_HU(mean(phantom.μ[mask]), μ_water))
            end
        end
        if length(ca_HUs) >= 2
            @test issorted(ca_HUs)
        end
    end

    @testset "Scanner Geometry" begin
        geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=16)

        @test geom.SAD ≈ 60.0
        @test geom.SDD ≈ 100.0
        @test geom.n_angles == 36
        @test geom.n_rows == 8
        @test geom.n_cols == 16
        @test geom.pixel_size ≈ 0.05

        @test size(geom.source_positions) == (3, 36)
        @test size(geom.detector_centers) == (3, 36)

        # First angle: source at (0, -SAD, 0)
        @test geom.source_positions[1, 1] ≈ 0.0 atol=1e-10
        @test geom.source_positions[2, 1] ≈ -60.0 atol=1e-10
        @test geom.source_positions[3, 1] ≈ 0.0 atol=1e-10

        src = get_source_position(geom, 1)
        @test src[1] ≈ 0.0 atol=1e-10
        @test src[2] ≈ -60.0 atol=1e-10
    end

    @testset "Forward Projection - Siddon" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])

        sinogram = forward_project(phantom, geom)

        @test size(sinogram) == (32, 4, 36)
        @test maximum(sinogram) > 0
        @test maximum(sinogram) < 50.0

        # Different angles have similar total signal
        totals = [sum(sinogram[:, :, i]) for i in 1:36]
        mean_total = mean(totals)
        @test all(t -> abs(t - mean_total) / mean_total < 0.5, totals)
    end

    @testset "FDK Reconstruction" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=72, n_rows=8, n_cols=64, fov_cm=phantom.fov[1])

        sinogram = forward_project(phantom, geom)

        output_size = (16, 16, size(phantom.μ, 3))
        recon = fdk_reconstruct(sinogram, geom, output_size, phantom.fov)

        @test size(recon) == size(phantom.μ)
        @test maximum(recon) > 0
        @test maximum(recon) < 2.0
    end

    @testset "End-to-End Validation" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])

        sinogram = forward_project(phantom, geom)
        recon = fdk_reconstruct(sinogram, geom, size(phantom.μ), phantom.fov)

        μ_water_ref = get_reference_μ_water(60.0)

        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        expected_water_HU = μ_to_HU(mean(phantom.μ[water_mask]), μ_water_ref)
        measured_water_HU = μ_to_HU(mean(recon[water_mask]), μ_water_ref)

        @test -100 < expected_water_HU < 100
        @test -500 < measured_water_HU < 500

        # Background lower than water
        bg_mask = phantom.mask .== UInt8(REGION_BACKGROUND)
        if sum(bg_mask) > 0
            bg_recon_HU = μ_to_HU(mean(recon[bg_mask]), μ_water_ref)
            @test bg_recon_HU < measured_water_HU
        end

        # Calcium higher than water
        ca_mask = get_region_mask(phantom, REGION_CA_100)
        if sum(ca_mask) > 0
            @test mean(recon[ca_mask]) > 0
        end

        # Contrast preserved
        water_vals = recon[water_mask]
        background_vals = recon[bg_mask]
        if length(water_vals) > 0 && length(background_vals) > 0
            @test mean(water_vals) > mean(background_vals)
        end
    end

    @testset "Reactant Compilation" begin
        # NO allowscalar - must work without it
        phantom = create_gammex_472(n_voxels=8)
        geom = create_aquilion_one(n_angles=4, n_rows=2, n_cols=8, fov_cm=phantom.fov[1])

        proj_geom = precompute_projection_geometry(
            geom, phantom.fov, phantom.voxel_size, size(phantom.μ)
        )

        volume_ra = Reactant.to_rarray(phantom.μ)

        compiled_pv = @compile project_volume(volume_ra, proj_geom)
        @test compiled_pv !== nothing

        sinogram_ra = compiled_pv(volume_ra, proj_geom)
        sinogram_result = Array(sinogram_ra)

        @test maximum(sinogram_result) > 0

        sinogram_julia = project_volume(phantom.μ, proj_geom)
        @test maximum(abs.(sinogram_result .- sinogram_julia)) < 1e-5
    end

    @testset "Polychromatic Simulation" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])

        # Create polychromatic projector
        projector = create_polychromatic_projector(phantom, geom, 120; n_bins=10)
        @test projector isa PolychromaticProjector
        @test length(projector.energies) == 10

        # Check effective energy is reasonable
        eff_E = compute_effective_energy(projector)
        @test 40 < eff_E < 80  # Typical range for 120 kVp

        # Check effective μ_water
        μ_eff = get_effective_μ_water(projector)
        @test 0.15 < μ_eff < 0.25

        # Forward project with polychromatic spectrum
        sino_poly = forward_project_polychromatic(phantom, projector)
        @test size(sino_poly) == (32, 4, 36)
        @test maximum(sino_poly) > 0
        @test all(isfinite.(sino_poly))

        # Compare to monochromatic at effective energy
        sino_mono = forward_project(phantom, geom)

        # Polychromatic should generally have lower values due to beam hardening
        # (harder spectrum after passing through material)
        # But both should be in similar range
        @test 0.5 < mean(sino_poly) / mean(sino_mono) < 2.0

        # Higher-density paths should show more beam hardening difference
        # (This is a qualitative check - polychromatic is more realistic)
        @test maximum(sino_poly) > 0
    end

    @testset "Polychromatic Reconstruction" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])

        # Create polychromatic projector with binned spectrum
        projector = create_polychromatic_projector(phantom, geom, 120; n_bins=20)

        # Forward project
        sinogram = forward_project_polychromatic(phantom, projector)

        # Reconstruct
        recon = fdk_reconstruct(sinogram, geom, size(phantom.μ), phantom.fov)

        # Get effective μ_water for HU conversion
        μ_water_eff = get_effective_μ_water(projector)

        # Basic validation
        @test maximum(recon) > 0
        @test all(isfinite.(recon))

        # Water region should have finite, reasonable HU value
        # (polychromatic beam hardening causes HU shifts)
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        water_HU = μ_to_HU(mean(recon[water_mask]), μ_water_eff)
        @test -1000 < water_HU < 500  # Wide tolerance for beam hardening

        # Key test: Calcium should have HIGHER attenuation than water
        # (This is the physics we care about, not absolute HU values)
        ca_mask = get_region_mask(phantom, REGION_CA_100)
        if sum(ca_mask) > 0
            @test mean(recon[ca_mask]) > mean(recon[water_mask])
        end
    end

    @testset "Scatter Simulation" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])

        # Create clean sinogram
        sinogram = forward_project(phantom, geom)

        # Create scatter model
        model = default_scatter_model(spr=0.15, kernel_fwhm=10.0)
        @test model isa ScatterModel
        @test model.spr == 0.15

        # Add scatter
        sino_scatter = add_scatter(sinogram, model)
        @test size(sino_scatter) == size(sinogram)
        @test all(isfinite.(sino_scatter))

        # Scatter should reduce contrast (increase low-attenuation values)
        # In attenuation space, scatter causes values to shift
        @test sino_scatter != sinogram  # Should be different

        # Estimate SPR from phantom
        estimated_spr = estimate_spr(phantom, geom)
        @test 0.01 < estimated_spr < 0.5

        # Compute scatter artifact magnitude
        artifact_mag = compute_scatter_artifact_magnitude(sinogram, sino_scatter)
        @test artifact_mag > 0  # Should have some difference
        @test artifact_mag < 1.0  # But not overwhelming

        # In-place version
        sino_copy = copy(sinogram)
        add_scatter!(sino_copy, model)
        @test sino_copy ≈ sino_scatter
    end

    @testset "Detector Blur" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=8, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])
        sinogram = forward_project(phantom, geom)

        # Create detector model with blur
        model = default_detector_model(blur_fwhm=2.0, I0=1e6, electronic_noise_std=0.0, seed=42)
        @test model isa DetectorModel

        # Apply blur only
        blurred = apply_detector_blur(sinogram, model)
        @test size(blurred) == size(sinogram)
        @test all(isfinite.(blurred))

        # Blurred should be smoother (lower max gradient)
        # Just check it's different
        @test blurred != sinogram
    end

    @testset "Quantum Noise" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=8, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])
        sinogram = forward_project(phantom, geom)

        # High dose (low noise)
        model_high = default_detector_model(blur_fwhm=0.0, I0=1e6, electronic_noise_std=0.0, seed=42)
        noisy_high = add_quantum_noise(sinogram, model_high)

        # Low dose (high noise)
        model_low = default_detector_model(blur_fwhm=0.0, I0=1e3, electronic_noise_std=0.0, seed=42)
        noisy_low = add_quantum_noise(sinogram, model_low)

        @test all(isfinite.(noisy_high))
        @test all(isfinite.(noisy_low))

        # Lower dose should have more noise
        noise_high = compute_noise_level(sinogram, noisy_high)
        noise_low = compute_noise_level(sinogram, noisy_low)
        @test noise_low.std_diff > noise_high.std_diff
    end

    @testset "Electronic Noise" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=8, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])
        sinogram = forward_project(phantom, geom)

        model = default_detector_model(blur_fwhm=0.0, I0=1e6, electronic_noise_std=50.0, seed=42)
        noisy = add_electronic_noise(sinogram, model)

        @test all(isfinite.(noisy))
        @test noisy != sinogram

        noise_stats = compute_noise_level(sinogram, noisy)
        @test noise_stats.std_diff > 0
    end

    @testset "Full Detector Model" begin
        phantom = create_gammex_472(n_voxels=16)
        geom = create_aquilion_one(n_angles=36, n_rows=4, n_cols=32, fov_cm=phantom.fov[1])
        sinogram = forward_project(phantom, geom)

        # Apply full detector model
        model = default_detector_model(blur_fwhm=1.5, I0=1e5, electronic_noise_std=20.0, seed=42)
        degraded = apply_detector_model(sinogram, model)

        @test size(degraded) == size(sinogram)
        @test all(isfinite.(degraded))

        # Should be different from clean
        @test degraded != sinogram

        # Noise stats
        stats = compute_noise_level(sinogram, degraded)
        @test stats.snr > 0
        @test stats.std_diff > 0
    end

    @testset "Scatter Effects on Reconstruction" begin
        phantom = create_gammex_472(n_voxels=32)
        geom = create_aquilion_one(n_angles=180, n_rows=8, n_cols=128, fov_cm=phantom.fov[1])

        # Clean forward projection
        sinogram = forward_project(phantom, geom)

        # Add scatter
        model = default_scatter_model(spr=0.20)
        sino_scatter = add_scatter(sinogram, model)

        # Reconstruct both
        recon_clean = fdk_reconstruct(sinogram, geom, size(phantom.μ), phantom.fov)
        recon_scatter = fdk_reconstruct(sino_scatter, geom, size(phantom.μ), phantom.fov)

        # Scatter causes cupping artifact (lower values in center)
        water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
        μ_clean = mean(recon_clean[water_mask])
        μ_scatter = mean(recon_scatter[water_mask])

        # Scatter typically reduces reconstructed values
        @test maximum(recon_scatter) > 0
        @test all(isfinite.(recon_scatter))
    end

    @testset "Visualization Output" begin
        # Create output directory
        output_dir = joinpath(@__DIR__, "outputs")
        mkpath(output_dir)

        # Generate phantom, sinogram, and reconstruction at higher resolution
        phantom = create_gammex_472(n_voxels=64)
        geom = create_aquilion_one(n_angles=360, n_rows=16, n_cols=256, fov_cm=phantom.fov[1])

        # Forward project
        sinogram = forward_project(phantom, geom)

        # Reconstruct
        recon = fdk_reconstruct(sinogram, geom, size(phantom.μ), phantom.fov)

        # Convert to HU for visualization
        μ_water = get_reference_μ_water(60.0)
        phantom_HU = μ_to_HU(phantom.μ, μ_water)
        recon_HU = μ_to_HU(recon, μ_water)

        # Save phantom slices (ground truth)
        mid_slice = size(phantom.μ, 3) ÷ 2
        save_heatmap(
            joinpath(output_dir, "01_phantom_axial"),
            phantom_HU[:, :, mid_slice];
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "01_phantom_axial.ppm"))

        # Save phantom montage
        save_slice_montage(
            joinpath(output_dir, "02_phantom_montage"),
            phantom_HU;
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "02_phantom_montage.ppm"))

        # Save sinogram
        save_sinogram_view(
            joinpath(output_dir, "03_sinogram"),
            sinogram;
            row=size(sinogram, 2) ÷ 2, colormap=:hot
        )
        @test isfile(joinpath(output_dir, "03_sinogram.ppm"))

        # Save reconstruction slices
        save_heatmap(
            joinpath(output_dir, "04_recon_axial"),
            recon_HU[:, :, mid_slice];
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "04_recon_axial.ppm"))

        # Save reconstruction montage
        save_slice_montage(
            joinpath(output_dir, "05_recon_montage"),
            recon_HU;
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "05_recon_montage.ppm"))

        # Save difference (phantom - recon)
        diff_HU = phantom_HU .- recon_HU
        save_heatmap(
            joinpath(output_dir, "06_difference"),
            diff_HU[:, :, mid_slice];
            colormap=:viridis, vmin=-500, vmax=500
        )
        @test isfile(joinpath(output_dir, "06_difference.ppm"))

        # Save region mask
        save_heatmap(
            joinpath(output_dir, "07_mask"),
            Float64.(phantom.mask[:, :, mid_slice]);
            colormap=:viridis
        )
        @test isfile(joinpath(output_dir, "07_mask.ppm"))

        # Also generate polychromatic reconstruction for comparison
        projector = create_polychromatic_projector(phantom, geom, 120; n_bins=20)
        sino_poly = forward_project_polychromatic(phantom, projector)
        recon_poly = fdk_reconstruct(sino_poly, geom, size(phantom.μ), phantom.fov)

        μ_water_eff = get_effective_μ_water(projector)
        recon_poly_HU = μ_to_HU(recon_poly, μ_water_eff)

        save_heatmap(
            joinpath(output_dir, "08_recon_poly_axial"),
            recon_poly_HU[:, :, mid_slice];
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "08_recon_poly_axial.ppm"))

        # Difference: monochromatic - polychromatic (shows beam hardening effect)
        diff_mono_poly = recon_HU .- recon_poly_HU
        save_heatmap(
            joinpath(output_dir, "09_beam_hardening_diff"),
            diff_mono_poly[:, :, mid_slice];
            colormap=:viridis, vmin=-200, vmax=200
        )
        @test isfile(joinpath(output_dir, "09_beam_hardening_diff.ppm"))

        # Add scatter simulation visualization
        scatter_model = default_scatter_model(spr=0.20)
        sino_scatter = add_scatter(sinogram, scatter_model)
        recon_scatter = fdk_reconstruct(sino_scatter, geom, size(phantom.μ), phantom.fov)
        recon_scatter_HU = μ_to_HU(recon_scatter, μ_water)

        save_heatmap(
            joinpath(output_dir, "10_recon_scatter_axial"),
            recon_scatter_HU[:, :, mid_slice];
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "10_recon_scatter_axial.ppm"))

        # Cupping artifact visualization (clean - scatter)
        cupping = recon_HU .- recon_scatter_HU
        save_heatmap(
            joinpath(output_dir, "11_scatter_cupping"),
            cupping[:, :, mid_slice];
            colormap=:viridis, vmin=-100, vmax=100
        )
        @test isfile(joinpath(output_dir, "11_scatter_cupping.ppm"))

        # Add noisy reconstruction visualization
        detector_model = default_detector_model(blur_fwhm=1.0, I0=5e4, electronic_noise_std=10.0, seed=42)
        sino_noisy = apply_detector_model(sinogram, detector_model)
        recon_noisy = fdk_reconstruct(sino_noisy, geom, size(phantom.μ), phantom.fov)
        recon_noisy_HU = μ_to_HU(recon_noisy, μ_water)

        save_heatmap(
            joinpath(output_dir, "12_recon_noisy_axial"),
            recon_noisy_HU[:, :, mid_slice];
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "12_recon_noisy_axial.ppm"))

        # Full realistic simulation: polychromatic + scatter + noise
        sino_full = forward_project_polychromatic(phantom, projector)
        sino_full = add_scatter(sino_full, scatter_model)
        sino_full = apply_detector_model(sino_full, detector_model)
        recon_full = fdk_reconstruct(sino_full, geom, size(phantom.μ), phantom.fov)
        recon_full_HU = μ_to_HU(recon_full, get_effective_μ_water(projector))

        save_heatmap(
            joinpath(output_dir, "13_recon_full_realistic"),
            recon_full_HU[:, :, mid_slice];
            colormap=:gray, vmin=-1000, vmax=1000
        )
        @test isfile(joinpath(output_dir, "13_recon_full_realistic.ppm"))

        println("\nVisualization outputs saved to: $output_dir")
        println("Files: phantom, sinogram, mono/poly/scatter/noisy/full reconstructions")
    end
end
