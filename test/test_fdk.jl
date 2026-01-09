"""
Test suite for Reconstruction/FDK.jl

Tests Feldkamp-Davis-Kress cone-beam reconstruction with validation against
known geometric and physical properties.
"""

using Test
using BasisSimulator
using LinearAlgebra
using Statistics

@testset "FDK Reconstruction" begin

    # =========================================================================
    # Test 1: Reconstruction Filter Creation
    # =========================================================================
    @testset "Reconstruction Filters" begin
        n_pixels = 512
        pixel_width = 0.1  # cm

        @testset "Ram-Lak Filter" begin
            filter = create_reconstruction_filter(n_pixels, pixel_width, ramlak)

            # Should be a ramp (linear in frequency)
            @test length(filter) == nextpow(2, n_pixels * 2)
            @test all(filter .>= 0)  # Ramp is always non-negative

            # DC component should be zero (ramp starts at zero)
            mid_idx = div(length(filter), 2) + 1
            @test filter[mid_idx] ≈ 0 atol=1e-10
        end

        @testset "Shepp-Logan Filter" begin
            filter = create_reconstruction_filter(n_pixels, pixel_width, shepplogan)

            @test length(filter) == nextpow(2, n_pixels * 2)
            @test all(isfinite.(filter))

            # Shepp-Logan should suppress high frequencies more than Ram-Lak
            ramlak_filter = create_reconstruction_filter(n_pixels, pixel_width, ramlak)

            # Compare high-frequency region
            high_freq_start = div(3 * length(filter), 4)
            @test mean(filter[high_freq_start:end]) <
                  mean(ramlak_filter[high_freq_start:end])
        end

        @testset "Hann Filter" begin
            filter = create_reconstruction_filter(n_pixels, pixel_width, hann)

            @test length(filter) == nextpow(2, n_pixels * 2)
            @test all(isfinite.(filter))

            # Hann should be smoothest (most high-frequency suppression)
            ramlak_filter = create_reconstruction_filter(n_pixels, pixel_width, ramlak)
            shepp_filter = create_reconstruction_filter(n_pixels, pixel_width, shepplogan)

            high_freq_start = div(3 * length(filter), 4)
            hann_high = mean(filter[high_freq_start:end])
            shepp_high = mean(shepp_filter[high_freq_start:end])
            ramlak_high = mean(ramlak_filter[high_freq_start:end])

            @test hann_high < shepp_high < ramlak_high
        end
    end

    # =========================================================================
    # Test 2: Input Validation
    # =========================================================================
    @testset "Input Validation" begin
        # Valid inputs
        projections = rand(128, 256, 360)
        SAD = 60.0
        SDD = 100.0
        pixel_w = 0.1
        pixel_h = 0.1
        angles = collect(0.0:1.0:359.0)
        x = collect(-10.0:0.2:10.0)
        y = collect(-10.0:0.2:10.0)
        z = collect(-5.0:0.2:5.0)

        @test validate_fdk_inputs(
            projections, SAD, SDD, pixel_w, pixel_h, angles, x, y, z
        ) == true

        # Invalid: angle count mismatch
        @test_throws DimensionMismatch validate_fdk_inputs(
            projections, SAD, SDD, pixel_w, pixel_h, angles[1:180], x, y, z
        )

        # Invalid: SAD <= 0
        @test_throws ArgumentError validate_fdk_inputs(
            projections, -10.0, SDD, pixel_w, pixel_h, angles, x, y, z
        )

        # Invalid: SDD < SAD
        @test_throws ArgumentError validate_fdk_inputs(
            projections, 100.0, 60.0, pixel_w, pixel_h, angles, x, y, z
        )

        # Invalid: negative pixel size
        @test_throws ArgumentError validate_fdk_inputs(
            projections, SAD, SDD, -0.1, pixel_h, angles, x, y, z
        )

        # Invalid: empty reconstruction grid
        @test_throws ArgumentError validate_fdk_inputs(
            projections, SAD, SDD, pixel_w, pixel_h, angles, Float64[], y, z
        )
    end

    # =========================================================================
    # Test 3: Circular Phantom Reconstruction
    # =========================================================================
    @testset "Circular Phantom Geometry" begin
        # Create simple circular phantom in projection space
        # This is a basic test to verify geometric correctness

        # Geometry
        SAD = 60.0  # cm
        SDD = 100.0  # cm

        # Detector
        n_rows = 64
        n_cols = 128
        pixel_width = 0.2  # cm
        pixel_height = 0.2  # cm

        # Scan
        n_angles = 360
        angles = collect(0.0:1.0:359.0)

        # Create synthetic projections of a circle
        # For a circle of radius R at isocenter, the projection is constant
        phantom_radius = 5.0  # cm
        phantom_mu = 0.2  # cm⁻¹ (typical soft tissue)

        projections = zeros(Float64, n_rows, n_cols, n_angles)

        # Fill with simple circular projection
        # Path length through circle at distance d from center: 2√(R² - d²)
        for k in 1:n_angles
            for c in 1:n_cols
                u = (c - n_cols/2 - 0.5) * pixel_width

                # Distance from ray to circle center
                # For parallel-beam approximation at isocenter
                dist_to_center = abs(u)

                if dist_to_center < phantom_radius
                    path_length = 2 * sqrt(phantom_radius^2 - dist_to_center^2)
                    # All rows see same projection (cylinder along z)
                    projections[:, c, k] .= phantom_mu * path_length
                end
            end
        end

        # Reconstruct
        recon_size = 32  # Small for speed
        recon_range = 10.0  # cm
        x = collect(range(-recon_range, recon_range, length=recon_size))
        y = collect(range(-recon_range, recon_range, length=recon_size))
        z = collect(range(-2.0, 2.0, length=8))  # Just a few z-slices

        volume = reconstruct_fdk(
            projections, SAD, SDD, pixel_width, pixel_height, angles, x, y, z,
            filter_type = shepplogan
        )

        # Test 1: Circular symmetry
        # Extract central slice
        central_slice = volume[:, :, div(length(z), 2)]

        # Create radial distance map
        cx, cy = div(recon_size, 2) + 1, div(recon_size, 2) + 1
        radius_map = zeros(recon_size, recon_size)
        for j in 1:recon_size
            for i in 1:recon_size
                dx = x[i] - x[cx]
                dy = y[j] - y[cy]
                radius_map[i, j] = sqrt(dx^2 + dy^2)
            end
        end

        # Points at same radius should have similar values
        test_radius = 3.0  # cm (inside phantom)
        points_at_radius = findall(abs.(radius_map .- test_radius) .< 0.5)

        if length(points_at_radius) > 4
            values_at_radius = [central_slice[p] for p in points_at_radius]
            # Check coefficient of variation is small
            cv = std(values_at_radius) / (mean(values_at_radius) + 1e-10)
            @test cv < 0.3  # Within 30% variation (lenient for small test)
        end

        # Test 2: Background should be lower than center
        # (Allow for reconstruction artifacts - this is a simple test phantom)
        background_radius = 8.0  # cm (outside phantom)
        background_points = findall(radius_map .> background_radius)
        background_values = [central_slice[p] for p in background_points]

        # Center should have higher intensity than background
        center_value = abs(central_slice[cx, cy])
        @test mean(abs.(background_values)) < center_value
    end

    # =========================================================================
    # Test 4: Hounsfield Unit Conversion
    # =========================================================================
    @testset "Hounsfield Units" begin
        # Create synthetic volume with known attenuation values
        nx, ny, nz = 10, 10, 10

        # Water attenuation at 60 keV ≈ 0.206 cm⁻¹
        mu_water = 0.206

        # Create volume with water, air, and bone regions
        volume = zeros(Float64, nx, ny, nz)
        volume[3:5, 3:5, :] .= 0.0  # Air (μ ≈ 0)
        volume[6:8, 6:8, :] .= mu_water  # Water
        volume[1:2, 8:10, :] .= 0.5  # Bone-like (higher μ)

        # Convert to HU
        hu_volume = convert_to_hounsfield_units(volume, mu_water)

        # Test 1: Water should be 0 HU
        water_hu = hu_volume[6:8, 6:8, :]
        @test mean(water_hu) ≈ 0.0 atol=1e-10

        # Test 2: Air should be ≈ -1000 HU
        air_hu = hu_volume[3:5, 3:5, :]
        expected_air_hu = 1000.0 * (0.0 - mu_water) / mu_water
        @test mean(air_hu) ≈ expected_air_hu atol=1.0

        # Test 3: Denser material should be positive HU
        bone_hu = hu_volume[1:2, 8:10, :]
        @test mean(bone_hu) > 100.0  # Should be well above water

        # Test 4: Invalid input (negative mu_water)
        @test_throws AssertionError convert_to_hounsfield_units(volume, -0.1)
    end

    # =========================================================================
    # Test 5: Filter Comparison (Noise vs Sharpness Trade-off)
    # =========================================================================
    @testset "Filter Comparison" begin
        # Create simple test projections
        SAD = 60.0
        SDD = 100.0
        n_rows, n_cols = 32, 64
        pixel_w, pixel_h = 0.2, 0.2
        angles = collect(0.0:2.0:358.0)  # 180 angles

        # Simple uniform cylinder
        projections = ones(Float64, n_rows, n_cols, length(angles)) .* 0.5

        # Add some noise
        projections .+= randn(size(projections)) .* 0.05

        # Reconstruction grid
        x = collect(range(-8.0, 8.0, length=32))
        y = collect(range(-8.0, 8.0, length=32))
        z = collect(range(-2.0, 2.0, length=8))

        # Reconstruct with different filters
        vol_ramlak = reconstruct_fdk(
            projections, SAD, SDD, pixel_w, pixel_h, angles, x, y, z,
            filter_type = ramlak
        )

        vol_shepp = reconstruct_fdk(
            projections, SAD, SDD, pixel_w, pixel_h, angles, x, y, z,
            filter_type = shepplogan
        )

        vol_hann = reconstruct_fdk(
            projections, SAD, SDD, pixel_w, pixel_h, angles, x, y, z,
            filter_type = hann
        )

        # Test: Hann should be smoothest (lowest std dev in homogeneous region)
        # Extract central region
        c_slice = div(length(z), 2)
        c_x = div(length(x), 2)
        c_y = div(length(y), 2)

        # 5x5 central region
        region_ramlak = vol_ramlak[(c_x-2):(c_x+2), (c_y-2):(c_y+2), c_slice]
        region_shepp = vol_shepp[(c_x-2):(c_x+2), (c_y-2):(c_y+2), c_slice]
        region_hann = vol_hann[(c_x-2):(c_x+2), (c_y-2):(c_y+2), c_slice]

        std_ramlak = std(region_ramlak)
        std_shepp = std(region_shepp)
        std_hann = std(region_hann)

        # Hann should have lowest noise, Ram-Lak highest
        @test std_hann < std_shepp < std_ramlak
    end

    # =========================================================================
    # Test 6: Linearity Test
    # =========================================================================
    @testset "Linearity" begin
        # FDK should be linear: reconstruct(a·p + b·q) = a·reconstruct(p) + b·reconstruct(q)

        # Create two simple projection sets
        SAD, SDD = 60.0, 100.0
        n_rows, n_cols = 32, 64
        pixel_w, pixel_h = 0.2, 0.2
        angles = collect(0.0:4.0:356.0)

        # Simple projections
        proj1 = rand(n_rows, n_cols, length(angles)) .* 0.3
        proj2 = rand(n_rows, n_cols, length(angles)) .* 0.3

        # Scales
        a, b = 0.5, 1.5

        # Reconstruction grid
        x = collect(range(-6.0, 6.0, length=16))
        y = collect(range(-6.0, 6.0, length=16))
        z = collect(range(-1.0, 1.0, length=4))

        # Reconstruct individually
        vol1 = reconstruct_fdk(proj1, SAD, SDD, pixel_w, pixel_h, angles, x, y, z)
        vol2 = reconstruct_fdk(proj2, SAD, SDD, pixel_w, pixel_h, angles, x, y, z)

        # Reconstruct linear combination
        proj_combined = a .* proj1 .+ b .* proj2
        vol_combined = reconstruct_fdk(proj_combined, SAD, SDD, pixel_w, pixel_h, angles, x, y, z)

        # Compare
        vol_expected = a .* vol1 .+ b .* vol2

        # Test: Should be nearly identical (allowing for numerical precision)
        @test isapprox(vol_combined, vol_expected, rtol=1e-10)
    end

    # =========================================================================
    # Test 7: Angular Coverage Effect
    # =========================================================================
    @testset "Angular Coverage" begin
        # More angles should give better reconstruction

        SAD, SDD = 60.0, 100.0
        n_rows, n_cols = 32, 64
        pixel_w, pixel_h = 0.2, 0.2

        # Create test projections
        projections_360 = rand(n_rows, n_cols, 360) .* 0.5

        # Subsample to fewer angles
        angles_180 = collect(0.0:1.0:359.0)[1:2:end]  # Every other angle
        projections_180 = projections_360[:, :, 1:2:end]

        angles_360 = collect(0.0:1.0:359.0)

        # Reconstruct
        x = collect(range(-6.0, 6.0, length=16))
        y = collect(range(-6.0, 6.0, length=16))
        z = collect(range(-1.0, 1.0, length=4))

        vol_180 = reconstruct_fdk(projections_180, SAD, SDD, pixel_w, pixel_h, angles_180, x, y, z)
        vol_360 = reconstruct_fdk(projections_360, SAD, SDD, pixel_w, pixel_h, angles_360, x, y, z)

        # Test: 360 angles should capture more detail
        # (This is a qualitative test - both should be valid reconstructions,
        #  but 360 will have better angular sampling)
        @test !isapprox(vol_180, vol_360)  # Should be different
        @test all(isfinite.(vol_180))
        @test all(isfinite.(vol_360))
    end

    # =========================================================================
    # Test 8: Reconstruction Grid Size
    # =========================================================================
    @testset "Reconstruction Grid Variations" begin
        SAD, SDD = 60.0, 100.0
        pixel_w, pixel_h = 0.2, 0.2

        # Fixed projections
        projections = rand(32, 64, 180) .* 0.5
        angles = collect(0.0:2.0:358.0)

        # Different grid sizes
        x_coarse = collect(range(-8.0, 8.0, length=8))
        y_coarse = collect(range(-8.0, 8.0, length=8))
        z_coarse = collect(range(-2.0, 2.0, length=4))

        x_fine = collect(range(-8.0, 8.0, length=32))
        y_fine = collect(range(-8.0, 8.0, length=32))
        z_fine = collect(range(-2.0, 2.0, length=16))

        # Reconstruct both
        vol_coarse = reconstruct_fdk(
            projections, SAD, SDD, pixel_w, pixel_h, angles,
            x_coarse, y_coarse, z_coarse
        )

        vol_fine = reconstruct_fdk(
            projections, SAD, SDD, pixel_w, pixel_h, angles,
            x_fine, y_fine, z_fine
        )

        # Test: Both should succeed with correct dimensions
        @test size(vol_coarse) == (length(x_coarse), length(y_coarse), length(z_coarse))
        @test size(vol_fine) == (length(x_fine), length(y_fine), length(z_fine))

        # Test: All values should be finite
        @test all(isfinite.(vol_coarse))
        @test all(isfinite.(vol_fine))
    end
end

# ==============================================================================
# Integration Test: End-to-End Reconstruction
# ==============================================================================

@testset "FDK Integration Test" begin
    @testset "Full Reconstruction Pipeline" begin
        # This test simulates a complete reconstruction workflow

        # Scanner geometry (typical clinical CT)
        SAD = 60.0  # cm
        SDD = 100.0  # cm

        # Detector
        n_detector_rows = 64
        n_detector_cols = 128
        detector_pixel_width = 0.1  # cm (1 mm)
        detector_pixel_height = 0.1  # cm

        # Acquisition
        n_projections = 360
        projection_angles = collect(range(0.0, 360.0, length=n_projections+1))[1:end-1]

        # Create synthetic projections (simple cylinder)
        projections = zeros(Float64, n_detector_rows, n_detector_cols, n_projections)
        cylinder_radius = 5.0  # cm
        cylinder_mu = 0.2  # cm⁻¹

        for k in 1:n_projections
            for r in 1:n_detector_rows
                for c in 1:n_detector_cols
                    u = (c - n_detector_cols/2 - 0.5) * detector_pixel_width
                    dist = abs(u)
                    if dist < cylinder_radius
                        path_length = 2 * sqrt(cylinder_radius^2 - dist^2)
                        projections[r, c, k] = cylinder_mu * path_length
                    end
                end
            end
        end

        # Reconstruction grid
        recon_fov = 15.0  # cm
        voxel_size = 0.2  # cm
        n_voxels = round(Int, recon_fov / voxel_size)

        x_recon = collect(range(-recon_fov/2, recon_fov/2, length=n_voxels))
        y_recon = collect(range(-recon_fov/2, recon_fov/2, length=n_voxels))
        z_recon = collect(range(-3.0, 3.0, length=32))

        # Reconstruct with Shepp-Logan filter
        volume = reconstruct_fdk(
            projections,
            SAD, SDD,
            detector_pixel_width, detector_pixel_height,
            projection_angles,
            x_recon, y_recon, z_recon,
            filter_type = shepplogan
        )

        # Convert to Hounsfield Units
        mu_water = 0.206  # cm⁻¹ at ~60 keV
        hu_volume = convert_to_hounsfield_units(volume, mu_water)

        # Validation tests
        @testset "Pipeline Outputs" begin
            @test size(volume) == (n_voxels, n_voxels, length(z_recon))
            @test all(isfinite.(volume))
            @test all(isfinite.(hu_volume))
        end

        @testset "Physical Realism" begin
            # Central region should have positive attenuation
            center_idx = div(n_voxels, 2)
            center_region = volume[(center_idx-2):(center_idx+2),
                                   (center_idx-2):(center_idx+2),
                                   div(length(z_recon), 2)]
            @test mean(center_region) > 0

            # Background should be near zero
            background_region = volume[1:5, 1:5, div(length(z_recon), 2)]
            @test mean(abs.(background_region)) < 0.5 * mean(abs.(center_region))
        end

        @testset "HU Calibration" begin
            # HU conversion should produce finite values
            # (Actual calibration accuracy will be tested with real phantoms)
            @test all(isfinite.(hu_volume))

            # Basic sanity check - values shouldn't be astronomical
            # (This is a synthetic test, so calibration may be off)
            @test maximum(abs.(hu_volume)) < 1e6  # Very lenient sanity bound
        end
    end
end

# ==============================================================================
# Performance Benchmarks (not run in standard test suite)
# ==============================================================================

function benchmark_fdk_reconstruction()
    println("\n=== FDK Reconstruction Benchmarks ===")

    # Only run if BenchmarkTools is available
    if !isdefined(Main, :BenchmarkTools)
        try
            using BenchmarkTools
        catch
            println("BenchmarkTools not available. Skipping benchmarks.")
            println("Run: using Pkg; Pkg.add(\"BenchmarkTools\")")
            return
        end
    else
        using BenchmarkTools
    end

    # Clinical CT parameters
    SAD, SDD = 60.0, 100.0
    projections = rand(64, 128, 360)
    angles = collect(0.0:1.0:359.0)

    x = collect(range(-10.0, 10.0, length=64))
    y = collect(range(-10.0, 10.0, length=64))
    z = collect(range(-5.0, 5.0, length=32))

    println("\nSmall volume (64×64×32, 360 projections):")
    @btime reconstruct_fdk(
        $projections, $SAD, $SDD, 0.1, 0.1, $angles, $x, $y, $z
    )

    # Larger volume
    projections_large = rand(128, 256, 360)
    x_large = collect(range(-15.0, 15.0, length=128))
    y_large = collect(range(-15.0, 15.0, length=128))
    z_large = collect(range(-8.0, 8.0, length=64))

    println("\nLarge volume (128×128×64, 360 projections):")
    @btime reconstruct_fdk(
        $projections_large, $SAD, $SDD, 0.1, 0.1, $angles,
        $x_large, $y_large, $z_large
    )
end

# Uncomment to run benchmarks:
# benchmark_fdk_reconstruction()
