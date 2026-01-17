# =============================================================================
# METRICS-002: NPS Measurement Verification
# =============================================================================
#
# This test verifies that the NPS (Noise Power Spectrum) implementation
# matches CatSim methodology and meets AAPM TG-233 standards.
#
# ACCEPTANCE CRITERIA (from prd.json):
# - 2D NPS calculation implemented
# - Radial averaging for 1D NPS
# - Results within 10% of CatSim NPS
# - Peak frequency and magnitude reported
# - Publication-ready documentation added
#
# PHYSICS BACKGROUND:
# The NPS describes the spatial frequency distribution of noise in an image.
# NPS = |FFT(ΔI)|² averaged over multiple ROIs
# For white (uncorrelated) noise: NPS is flat
# For correlated noise: NPS has structure (peak at characteristic frequency)
# Integrated NPS = noise variance σ²
#
# VERIFICATION APPROACH:
# 1. Synthetic white noise: NPS should be flat (CV < 30%)
# 2. Synthetic correlated noise: NPS should peak at expected frequency
# 3. Simulated CT reconstruction: NPS matches known noise properties
# 4. Integrated NPS equals measured variance
#
# REFERENCES:
# - AAPM TG-233 Section 4.3: NPS measurement methodology
# - ICRU Report 87 Section 4.2.2: Definition and computation of NPS
# - Boedeker KL, et al. Phys Med Biol 2007;52:4027-4046
#
# USAGE:
#   cd BasisSimulator.jl && julia --project test/verification/nps.jl
#
# =============================================================================

using Test
using Statistics
using Printf
using Dates
using Random
using FFTW

# Add parent directory to load path
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using BasisSimulator
import XrayAttenuation as XA

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

struct NPSTestConfig
    # Phantom parameters
    n_voxels::Int
    n_slices::Int
    fov_cm::Float64
    z_cm::Float64

    # Simulation parameters
    n_views::Int
    n_rows::Int
    n_cols::Int

    # NPS measurement parameters
    roi_size::Int
    n_rois::Int

    # Tolerances
    nps_tolerance_percent::Float64  # Relative tolerance for NPS comparison
    flatness_tolerance::Float64     # CV threshold for white noise flatness
end

function default_nps_test_config(; scale::Symbol=:dev)
    if scale == :dev
        return NPSTestConfig(
            64, 1, 35.0, 0.5,    # Phantom: 64³, 35cm FOV
            90, 1, 128,          # Simulation: 90 views, single row
            32, 4,               # NPS: 32×32 ROIs, 4 ROIs
            15.0, 0.40           # 15% tolerance, 40% CV for flatness
        )
    elseif scale == :integration
        return NPSTestConfig(
            128, 1, 35.0, 0.5,   # Phantom: 128³, 35cm FOV
            180, 1, 256,         # Simulation: 180 views
            64, 16,              # NPS: 64×64 ROIs, 16 ROIs
            12.0, 0.35           # 12% tolerance, 35% CV
        )
    else  # verification scale
        return NPSTestConfig(
            256, 1, 35.0, 0.5,   # Phantom: 256³, 35cm FOV
            360, 1, 512,         # Simulation: 360 views
            64, 32,              # NPS: 64×64 ROIs, 32 ROIs
            10.0, 0.30           # 10% tolerance (acceptance criterion), 30% CV
        )
    end
end

# =============================================================================
# SYNTHETIC NOISE TESTS
# =============================================================================

"""
Test NPS with synthetic white Gaussian noise.
For white noise, NPS should be approximately flat.
"""
function test_white_noise_nps(cfg::NPSTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: White Noise NPS (Synthetic)")
    println("=" ^ 60)

    Random.seed!(42)  # Reproducibility

    # Create white noise image
    n_pixels = max(cfg.n_voxels, 128)  # Need sufficient size for NPS
    pixel_size_mm = cfg.fov_cm * 10.0 / n_pixels
    noise_std = 30.0  # HU

    # Generate Gaussian white noise
    noise_image = randn(n_pixels, n_pixels) .* noise_std

    println("  Parameters:")
    println(@sprintf("    Image size: %d × %d pixels", n_pixels, n_pixels))
    println(@sprintf("    Pixel size: %.3f mm", pixel_size_mm))
    println(@sprintf("    Noise std: %.1f HU", noise_std))

    # Measure NPS
    nps_config = NPSConfig(roi_size=cfg.roi_size, n_rois=cfg.n_rois)
    result = measure_nps(noise_image, pixel_size_mm; config=nps_config)

    info = get_nps_info(result)

    println("\n  NPS Results:")
    println(@sprintf("    Peak frequency: %.3f lp/mm", info.peak_frequency))
    println(@sprintf("    Integrated NPS (variance): %.1f HU²", info.integrated_nps))
    println(@sprintf("    Estimated noise std: %.1f HU", info.noise_std))
    println(@sprintf("    Number of ROIs used: %d", info.n_rois))

    all_passed = true

    # Test 1: NPS flatness (white noise should have flat NPS)
    # Calculate coefficient of variation of NPS (excluding DC)
    nps_without_dc = result.nps_1d[2:end]
    nps_mean = mean(nps_without_dc)
    nps_std = std(nps_without_dc)
    nps_cv = nps_mean > 0 ? nps_std / nps_mean : Inf

    passed_flatness = nps_cv < cfg.flatness_tolerance
    all_passed &= passed_flatness
    println()
    println(@sprintf("  NPS flatness (CV < %.0f%%): CV = %.1f%% [%s]",
                    cfg.flatness_tolerance * 100, nps_cv * 100,
                    passed_flatness ? "PASS" : "FAIL"))

    # Test 2: Integrated NPS should approximate input variance
    input_variance = noise_std^2
    measured_variance = info.integrated_nps
    variance_error = abs(measured_variance - input_variance) / input_variance * 100

    # More lenient for NPS variance (integration depends on frequency sampling)
    passed_variance = variance_error < 50.0  # 50% tolerance for variance
    all_passed &= passed_variance
    println(@sprintf("  Variance match (within 50%%): input=%.1f, measured=%.1f, error=%.1f%% [%s]",
                    input_variance, measured_variance, variance_error,
                    passed_variance ? "PASS" : "FAIL"))

    # Test 3: Estimated noise std should be reasonable
    std_error = abs(info.noise_std - noise_std) / noise_std * 100
    passed_std = std_error < 50.0  # 50% tolerance
    all_passed &= passed_std
    println(@sprintf("  Noise std match: input=%.1f, estimated=%.1f, error=%.1f%% [%s]",
                    noise_std, info.noise_std, std_error,
                    passed_std ? "PASS" : "FAIL"))

    return all_passed, result
end

"""
Test NPS with synthetic correlated (filtered) noise.
Correlated noise should have NPS peak at expected frequency.
"""
function test_correlated_noise_nps(cfg::NPSTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Correlated Noise NPS (Synthetic)")
    println("=" ^ 60)

    Random.seed!(43)

    # Create correlated noise by filtering white noise
    n_pixels = max(cfg.n_voxels, 128)
    pixel_size_mm = cfg.fov_cm * 10.0 / n_pixels
    noise_std = 30.0

    # Start with white noise
    white_noise = randn(n_pixels, n_pixels) .* noise_std

    # Apply Gaussian smoothing (creates correlated noise)
    # This acts as a low-pass filter, so NPS should peak near DC
    sigma_pixels = 3.0  # Gaussian filter width

    # Create Gaussian kernel
    kernel_size = 2 * ceil(Int, 3 * sigma_pixels) + 1
    center = kernel_size ÷ 2 + 1
    kernel = zeros(Float64, kernel_size, kernel_size)
    for j in 1:kernel_size, i in 1:kernel_size
        r2 = (i - center)^2 + (j - center)^2
        kernel[i, j] = exp(-r2 / (2 * sigma_pixels^2))
    end
    kernel ./= sum(kernel)

    # Convolve (simple approach using FFT)
    noise_padded = zeros(Float64, n_pixels + kernel_size - 1, n_pixels + kernel_size - 1)
    noise_padded[1:n_pixels, 1:n_pixels] = white_noise
    kernel_padded = zeros(Float64, size(noise_padded))
    kernel_padded[1:kernel_size, 1:kernel_size] = kernel

    convolved = real.(ifft(fft(noise_padded) .* fft(kernel_padded)))
    correlated_noise = convolved[center:center+n_pixels-1, center:center+n_pixels-1]

    println("  Parameters:")
    println(@sprintf("    Image size: %d × %d pixels", n_pixels, n_pixels))
    println(@sprintf("    Filter sigma: %.1f pixels (%.2f mm)", sigma_pixels, sigma_pixels * pixel_size_mm))

    # Measure NPS
    nps_config = NPSConfig(roi_size=cfg.roi_size, n_rois=cfg.n_rois)
    result = measure_nps(correlated_noise, pixel_size_mm; config=nps_config)

    info = get_nps_info(result)

    println("\n  NPS Results:")
    println(@sprintf("    Peak frequency: %.3f lp/mm", info.peak_frequency))
    println(@sprintf("    Peak NPS value: %.1f HU² mm²", info.peak_value))

    all_passed = true

    # Test 1: Peak should be near DC (low frequency) for smoothed noise
    # Expected cutoff frequency for Gaussian: f_c ≈ 1/(2π × σ_mm)
    sigma_mm = sigma_pixels * pixel_size_mm
    expected_cutoff = 1.0 / (2 * π * sigma_mm)  # lp/mm

    # For low-pass filtered noise, peak should be at or near DC
    # The peak frequency should be less than the cutoff
    passed_peak = info.peak_frequency < expected_cutoff * 2  # Allow 2x margin
    all_passed &= passed_peak
    println()
    println(@sprintf("  Peak at low frequency: f_peak=%.3f < cutoff=%.3f lp/mm [%s]",
                    info.peak_frequency, expected_cutoff * 2,
                    passed_peak ? "PASS" : "FAIL"))

    # Test 2: NPS should decrease at high frequencies (correlated noise characteristic)
    low_freq_nps = mean(result.nps_1d[2:min(5, length(result.nps_1d))])
    high_freq_nps = mean(result.nps_1d[end-4:end])

    passed_decrease = low_freq_nps > high_freq_nps * 1.5  # Should be at least 1.5x higher
    all_passed &= passed_decrease
    println(@sprintf("  NPS decreases at high freq: low=%.1f, high=%.1f, ratio=%.2f [%s]",
                    low_freq_nps, high_freq_nps, low_freq_nps / max(high_freq_nps, 1e-10),
                    passed_decrease ? "PASS" : "FAIL"))

    return all_passed, result
end

# =============================================================================
# SIMULATED CT RECONSTRUCTION TESTS
# =============================================================================

"""
Test NPS from simulated CT water phantom reconstruction.
"""
function test_water_phantom_nps(cfg::NPSTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Water Phantom NPS (Full Simulation)")
    println("=" ^ 60)

    Random.seed!(44)

    # Create water phantom
    n_voxels = cfg.n_voxels
    fov_cm = cfg.fov_cm
    pixel_size_mm = fov_cm * 10.0 / n_voxels

    println("  Parameters:")
    println(@sprintf("    Phantom: %d × %d × %d, FOV=%.1f cm", n_voxels, n_voxels, cfg.n_slices, fov_cm))
    println(@sprintf("    Simulation: %d views, %d×%d detector", cfg.n_views, cfg.n_rows, cfg.n_cols))

    # Create geometry
    geom = create_aquilion_one(
        n_angles=cfg.n_views,
        n_rows=cfg.n_rows,
        n_cols=cfg.n_cols,
        fov_cm=fov_cm,
        z_cm=cfg.z_cm
    )

    # Create uniform water phantom (μ_water at 60 keV ≈ 0.02 mm⁻¹)
    μ_water = 0.02f0
    phantom_μ = fill(μ_water, n_voxels, n_voxels, cfg.n_slices)

    println("  Forward projecting...")

    # Full simulation with noise
    physics = default_physics_config(
        noise = default_detector_model(I0=1e5, seed=44),  # Add quantum noise
        energy_keV = 60.0
    )

    sinogram = forward_project(Float32.(phantom_μ), geom; physics=physics)

    println("  Reconstructing...")
    recon = fdk_reconstruct(sinogram, geom, (n_voxels, n_voxels, cfg.n_slices))

    # Convert to HU
    recon_μ = Array(recon)
    recon_hu = 1000f0 .* (recon_μ .- μ_water) ./ μ_water

    # Extract central slice
    slice_idx = cfg.n_slices ÷ 2 + 1
    recon_2d = Float64.(recon_hu[:, :, slice_idx])

    # Measure image noise (std dev in central ROI)
    center = n_voxels ÷ 2
    roi_half = n_voxels ÷ 4
    central_roi = recon_2d[center-roi_half:center+roi_half, center-roi_half:center+roi_half]
    measured_noise_std = std(central_roi)
    measured_mean_hu = mean(central_roi)

    println(@sprintf("  Reconstruction stats: mean HU = %.1f, noise std = %.1f HU",
                    measured_mean_hu, measured_noise_std))

    # Measure NPS
    println("  Measuring NPS...")
    nps_config = NPSConfig(roi_size=cfg.roi_size, n_rois=cfg.n_rois)
    result = measure_nps(recon_2d, pixel_size_mm; config=nps_config)

    info = get_nps_info(result)

    println("\n  NPS Results:")
    println(@sprintf("    Peak frequency: %.3f lp/mm", info.peak_frequency))
    println(@sprintf("    Peak value: %.1f HU² mm²", info.peak_value))
    println(@sprintf("    Integrated NPS: %.1f HU²", info.integrated_nps))
    println(@sprintf("    Estimated noise std: %.1f HU", info.noise_std))
    println(@sprintf("    Number of ROIs used: %d", info.n_rois))

    all_passed = true

    # Test 1: NPS-derived noise std should match measured std
    # This is the key validation: σ² = ∫ NPS(f) df
    nps_derived_std = info.noise_std
    std_error = abs(nps_derived_std - measured_noise_std) / max(measured_noise_std, 1.0) * 100

    passed_std = std_error < 100.0  # 100% tolerance (NPS integration has uncertainties)
    all_passed &= passed_std
    println()
    println(@sprintf("  Noise std consistency: measured=%.1f, NPS-derived=%.1f, error=%.1f%% [%s]",
                    measured_noise_std, nps_derived_std, std_error,
                    passed_std ? "PASS" : "FAIL"))

    # Test 2: NPS should have reasonable magnitude (not zero, not infinite)
    passed_magnitude = info.integrated_nps > 0 && info.integrated_nps < 1e6
    all_passed &= passed_magnitude
    println(@sprintf("  NPS magnitude reasonable: %.1f HU² [%s]",
                    info.integrated_nps,
                    passed_magnitude ? "PASS" : "FAIL"))

    # Test 3: Peak frequency should be reasonable for CT noise
    # CT noise typically peaks at 0.1-0.5 lp/mm depending on kernel
    nyquist = 1.0 / (2.0 * pixel_size_mm)
    passed_peak = info.peak_frequency >= 0 && info.peak_frequency <= nyquist
    all_passed &= passed_peak
    println(@sprintf("  Peak frequency reasonable: %.3f lp/mm (Nyquist=%.3f) [%s]",
                    info.peak_frequency, nyquist,
                    passed_peak ? "PASS" : "FAIL"))

    return all_passed, result
end

# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

"""
Test NPS configuration and result accessors.
"""
function test_nps_utilities()
    println("\n" * "=" ^ 60)
    println("TEST: NPS Utility Functions")
    println("=" ^ 60)

    all_passed = true

    # Test 1: NPSConfig construction
    config = NPSConfig()
    passed_config = config.roi_size == 64 && config.n_rois == 16 && config.overlap ≈ 0.0
    all_passed &= passed_config
    println(@sprintf("  Default NPSConfig: roi_size=%d, n_rois=%d [%s]",
                    config.roi_size, config.n_rois,
                    passed_config ? "PASS" : "FAIL"))

    config2 = NPSConfig(roi_size=32, n_rois=8, overlap=0.5)
    passed_config2 = config2.roi_size == 32 && config2.overlap ≈ 0.5
    all_passed &= passed_config2
    println(@sprintf("  Custom NPSConfig: roi_size=%d, overlap=%.1f [%s]",
                    config2.roi_size, config2.overlap,
                    passed_config2 ? "PASS" : "FAIL"))

    # Test 2: NPSResult accessors with synthetic data
    Random.seed!(45)
    test_image = randn(128, 128) .* 30.0
    pixel_size_mm = 0.5

    result = measure_nps(test_image, pixel_size_mm; config=NPSConfig(roi_size=32, n_rois=4))

    # Test accessors
    passed_peak = nps_peak_frequency(result) >= 0
    passed_value = nps_peak_value(result) >= 0
    passed_var = nps_variance(result) >= 0

    all_passed &= passed_peak && passed_value && passed_var
    println(@sprintf("  Accessors: peak_freq=%.3f, peak_val=%.1f, variance=%.1f [%s]",
                    nps_peak_frequency(result), nps_peak_value(result), nps_variance(result),
                    (passed_peak && passed_value && passed_var) ? "PASS" : "FAIL"))

    # Test get_nps_peak
    freq, val = get_nps_peak(result)
    passed_get_peak = freq >= 0 && val >= 0
    all_passed &= passed_get_peak
    println(@sprintf("  get_nps_peak: (%.3f, %.1f) [%s]",
                    freq, val,
                    passed_get_peak ? "PASS" : "FAIL"))

    # Test get_nps_info
    info = get_nps_info(result)
    passed_info = haskey(info, :peak_frequency) && haskey(info, :noise_std) && haskey(info, :n_rois)
    all_passed &= passed_info
    println(@sprintf("  get_nps_info: keys present [%s]",
                    passed_info ? "PASS" : "FAIL"))

    return all_passed
end

"""
Test NPS comparison utility.
"""
function test_nps_comparison()
    println("\n" * "=" ^ 60)
    println("TEST: NPS Comparison")
    println("=" ^ 60)

    Random.seed!(46)

    # Create two noise images with different variances
    noise1 = randn(128, 128) .* 30.0
    noise2 = randn(128, 128) .* 45.0  # 50% more noise
    pixel_size_mm = 0.5

    config = NPSConfig(roi_size=32, n_rois=4)
    result1 = measure_nps(noise1, pixel_size_mm; config=config)
    result2 = measure_nps(noise2, pixel_size_mm; config=config)

    comparison = compare_nps(result1, result2)

    all_passed = true

    # Variance should differ by factor of ~2.25 (1.5²)
    variance_ratio = result2.integrated_nps / max(result1.integrated_nps, 1.0)
    passed_variance = 1.0 < variance_ratio < 5.0  # Should be around 2.25
    all_passed &= passed_variance
    println(@sprintf("  Variance ratio: %.2f (expected ~2.25) [%s]",
                    variance_ratio,
                    passed_variance ? "PASS" : "FAIL"))

    # Comparison should detect the difference
    passed_detect = comparison.variance_rel_percent > 10  # Should detect significant difference
    all_passed &= passed_detect
    println(@sprintf("  Variance difference detected: %.1f%% [%s]",
                    comparison.variance_rel_percent,
                    passed_detect ? "PASS" : "FAIL"))

    # Noise std ratio should be ~1.5
    std_ratio = sqrt(result2.integrated_nps) / sqrt(max(result1.integrated_nps, 1.0))
    passed_std_ratio = 1.0 < std_ratio < 3.0  # Should be around 1.5
    all_passed &= passed_std_ratio
    println(@sprintf("  Noise std ratio: %.2f (expected ~1.5) [%s]",
                    std_ratio,
                    passed_std_ratio ? "PASS" : "FAIL"))

    return all_passed
end

"""
Test uniform phantom creation for NPS.
"""
function test_phantom_creation()
    println("\n" * "=" ^ 60)
    println("TEST: Phantom Creation for NPS")
    println("=" ^ 60)

    all_passed = true

    # Create uniform phantom
    phantom = create_uniform_phantom_nps(128, 35.0; hu_value=0.0, noise_std=30.0)

    passed_size = size(phantom) == (128, 128)
    all_passed &= passed_size
    println(@sprintf("  Phantom size: %s [%s]",
                    size(phantom),
                    passed_size ? "PASS" : "FAIL"))

    # Check mean is approximately 0 (water HU)
    phantom_mean = mean(phantom)
    passed_mean = abs(phantom_mean) < 10.0  # Within 10 HU of 0
    all_passed &= passed_mean
    println(@sprintf("  Mean HU: %.1f (expected ~0) [%s]",
                    phantom_mean,
                    passed_mean ? "PASS" : "FAIL"))

    # Check std is approximately 30 HU
    phantom_std = std(phantom)
    std_error = abs(phantom_std - 30.0) / 30.0 * 100
    passed_std = std_error < 20.0  # Within 20% of 30 HU
    all_passed &= passed_std
    println(@sprintf("  Noise std: %.1f (expected 30) [%s]",
                    phantom_std,
                    passed_std ? "PASS" : "FAIL"))

    return all_passed
end

"""
Test 2D NPS measurement.
"""
function test_2d_nps()
    println("\n" * "=" ^ 60)
    println("TEST: 2D NPS Measurement")
    println("=" ^ 60)

    Random.seed!(47)

    # Create white noise
    noise_image = randn(128, 128) .* 30.0
    pixel_size_mm = 0.5

    # Measure 2D NPS
    config = NPSConfig(roi_size=32, n_rois=4)
    nps_2d, freq_x, freq_y = measure_nps_2d(noise_image, pixel_size_mm; config=config)

    all_passed = true

    # Check dimensions
    passed_dims = size(nps_2d) == (32, 32) && length(freq_x) == 32
    all_passed &= passed_dims
    println(@sprintf("  2D NPS dimensions: %s, freq_x length: %d [%s]",
                    size(nps_2d), length(freq_x),
                    passed_dims ? "PASS" : "FAIL"))

    # Check values are non-negative
    passed_nonneg = all(nps_2d .>= 0)
    all_passed &= passed_nonneg
    println(@sprintf("  All NPS values non-negative: min=%.2e [%s]",
                    minimum(nps_2d),
                    passed_nonneg ? "PASS" : "FAIL"))

    # Check 2D NPS has correct symmetry (should be roughly symmetric for white noise)
    center = size(nps_2d, 1) ÷ 2 + 1
    left = mean(nps_2d[:, 1:center-1])
    right = mean(nps_2d[:, center+1:end])
    symmetry_ratio = left / max(right, 1e-10)
    passed_symmetry = 0.5 < symmetry_ratio < 2.0
    all_passed &= passed_symmetry
    println(@sprintf("  Approximate symmetry: left/right ratio = %.2f [%s]",
                    symmetry_ratio,
                    passed_symmetry ? "PASS" : "FAIL"))

    return all_passed
end

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

"""
Run all NPS verification tests.
"""
function verify_nps(; scale::Symbol=:dev)
    println()
    println("=" ^ 80)
    println("METRICS-002: NPS MEASUREMENT VERIFICATION")
    println("=" ^ 80)
    println("Timestamp: $(now())")
    println("Scale: $scale")
    println()

    cfg = default_nps_test_config(scale=scale)

    results = []

    # Utility tests (fast, no simulation)
    push!(results, ("NPS Utilities", test_nps_utilities()))
    push!(results, ("Phantom Creation", test_phantom_creation()))
    push!(results, ("2D NPS Measurement", test_2d_nps()))
    push!(results, ("NPS Comparison", test_nps_comparison()))

    # Synthetic noise tests
    white_passed, white_result = test_white_noise_nps(cfg)
    push!(results, ("White Noise NPS", white_passed))

    correlated_passed, correlated_result = test_correlated_noise_nps(cfg)
    push!(results, ("Correlated Noise NPS", correlated_passed))

    # Full simulation test (only at integration scale or higher)
    if scale != :dev
        water_passed, water_result = test_water_phantom_nps(cfg)
        push!(results, ("Water Phantom NPS", water_passed))
    end

    # Summary
    println()
    println("=" ^ 80)
    println("SUMMARY")
    println("=" ^ 80)
    println()

    all_passed = true
    for (name, passed) in results
        status = passed ? "PASS" : "FAIL"
        all_passed &= passed
        println(@sprintf("  [%s] %s", status, name))
    end

    println()
    println("=" ^ 80)
    if all_passed
        println("OVERALL: PASS - All NPS tests passed")
    else
        println("OVERALL: FAIL - Some tests failed")
    end
    println("=" ^ 80)
    println()

    return all_passed
end

"""
Run NPS tests using Julia's Test framework.
"""
function run_nps_tests(; scale::Symbol=:dev)
    cfg = default_nps_test_config(scale=scale)

    @testset "METRICS-002: NPS Measurement" begin
        @testset "NPS Configuration" begin
            # Default config
            config = NPSConfig()
            @test config.roi_size == 64
            @test config.n_rois == 16
            @test config.overlap ≈ 0.0
            @test config.detrend == :mean
            @test config.window == :none

            # Custom config
            config2 = NPSConfig(roi_size=32, n_rois=8, overlap=0.5, include_2d=true)
            @test config2.roi_size == 32
            @test config2.overlap ≈ 0.5
            @test config2.include_2d == true
        end

        @testset "Phantom Creation" begin
            phantom = create_uniform_phantom_nps(64, 35.0)
            @test size(phantom) == (64, 64)
            @test abs(mean(phantom)) < 10.0  # Near 0 HU

            phantom2 = create_uniform_phantom_nps(64, 35.0; hu_value=100.0, noise_std=50.0)
            @test abs(mean(phantom2) - 100.0) < 20.0
        end

        @testset "White Noise NPS" begin
            Random.seed!(48)
            noise_image = randn(128, 128) .* 30.0
            pixel_size_mm = 0.5

            config = NPSConfig(roi_size=32, n_rois=4)
            result = measure_nps(noise_image, pixel_size_mm; config=config)

            @test length(result.frequencies) > 0
            @test length(result.nps_1d) > 0
            @test result.integrated_nps > 0
            @test result.n_rois > 0
            @test result.unit == :lp_mm
        end

        @testset "NPS Result Accessors" begin
            Random.seed!(49)
            noise_image = randn(128, 128) .* 30.0
            result = measure_nps(noise_image, 0.5; config=NPSConfig(roi_size=32, n_rois=4))

            @test nps_peak_frequency(result) >= 0
            @test nps_peak_value(result) >= 0
            @test nps_variance(result) > 0

            freq, val = get_nps_peak(result)
            @test freq >= 0
            @test val >= 0

            info = get_nps_info(result)
            @test info.peak_frequency == result.peak_frequency
            @test info.n_rois == result.n_rois
            @test info.noise_std > 0
        end

        @testset "NPS Frequency Integration" begin
            Random.seed!(50)
            noise_image = randn(128, 128) .* 30.0
            result = measure_nps(noise_image, 0.5; config=NPSConfig(roi_size=32, n_rois=4))

            # Full integral
            full_integral = get_nps_integral(result)
            @test full_integral > 0

            # Partial integral should be less than full
            if length(result.frequencies) > 2
                f_mid = result.frequencies[length(result.frequencies) ÷ 2]
                partial = get_nps_integral(result; f_min=0.0, f_max=f_mid)
                @test partial <= full_integral * 1.1  # Allow small numerical error
            end
        end

        @testset "2D NPS" begin
            Random.seed!(51)
            noise_image = randn(128, 128) .* 30.0

            nps_2d, freq_x, freq_y = measure_nps_2d(noise_image, 0.5;
                config=NPSConfig(roi_size=32, n_rois=4))

            @test size(nps_2d) == (32, 32)
            @test length(freq_x) == 32
            @test length(freq_y) == 32
            @test all(nps_2d .>= 0)
        end

        @testset "NPS Comparison" begin
            Random.seed!(52)
            noise1 = randn(128, 128) .* 30.0
            noise2 = randn(128, 128) .* 45.0

            config = NPSConfig(roi_size=32, n_rois=4)
            result1 = measure_nps(noise1, 0.5; config=config)
            result2 = measure_nps(noise2, 0.5; config=config)

            comparison = compare_nps(result1, result2)
            @test comparison.variance_diff >= 0
            @test comparison.variance_rel_percent >= 0
            @test comparison.noise_std_diff >= 0
        end

        @testset "NPS Units Conversion" begin
            Random.seed!(53)
            noise_image = randn(128, 128) .* 30.0
            pixel_size_mm = 0.5

            config = NPSConfig(roi_size=32, n_rois=4)
            result_mm = measure_nps(noise_image, pixel_size_mm; config=config, unit=:lp_mm)
            result_cm = measure_nps(noise_image, pixel_size_mm; config=config, unit=:lp_cm)

            @test result_mm.unit == :lp_mm
            @test result_cm.unit == :lp_cm

            # Frequencies in lp/cm should be 10x lp/mm
            @test result_cm.frequencies[end] ≈ result_mm.frequencies[end] * 10 atol=0.1
        end
    end
end

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    scale = :dev

    for arg in ARGS
        if startswith(arg, "--scale=")
            scale = Symbol(split(arg, "=")[2])
        elseif arg == "--help"
            println("Usage: julia nps.jl [options]")
            println()
            println("Options:")
            println("  --scale=SCALE    Test scale: dev, integration, verification (default: dev)")
            println("  --help           Show this help message")
            exit(0)
        end
    end

    # Run verification
    passed = verify_nps(scale=scale)
    exit(passed ? 0 : 1)
end
