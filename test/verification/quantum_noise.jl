# =============================================================================
# Quantum Noise Verification (PHYSICS-009)
# =============================================================================
#
# Verifies that BasisSimulator's quantum noise implementation matches CatSim
# physics and produces correct Poisson noise statistics.
#
# CatSim Reference: pyfiles/Detection_EI.py, pyfiles/randpf.py, clib_build/src/rndpoi.c
#
# Key Physics:
# - Quantum noise follows Poisson statistics: N ~ Poisson(λ) where λ = photon count
# - For Poisson: variance = mean, so σ = √λ
# - In CT, λ = I₀ × exp(-∫μ dl) where I₀ is incident photon count
# - BasisSimulator uses Gaussian approximation: N = λ + √λ × randn (valid for λ > 100)
#
# CatSim Implementation:
# - Uses compiled C library (rndpoi.c) implementing PTRD algorithm
# - PTRD = Poisson Transformed Rejection with Squeeze (Hormann 1992)
# - For λ < 10: direct inversion method
# - For λ >= 10: normal approximation with acceptance-rejection
#
# =============================================================================

using BasisSimulator
using Statistics
using Test
using FFTW

"""
    run_quantum_noise_verification()

Comprehensive verification of quantum noise implementation.
"""
function run_quantum_noise_verification()
    println("\n" * "=" ^ 70)
    println("PHYSICS-009: Quantum Noise Verification")
    println("=" ^ 70)

    results = Dict{String, Bool}()

    # =========================================================================
    # Test 1: Variance scales with 1/√(photons)
    # =========================================================================
    println("\n[1] Variance Scaling with Photon Count")
    println("-" ^ 50)

    # Test multiple photon counts
    I0_values = [1e3, 1e4, 1e5, 1e6]
    n_cols, n_rows, n_angles = 128, 32, 1
    n_trials = 10

    println("\n  I₀           Expected σ/μ    Measured σ/μ    Error")
    println("  " * "-" ^ 56)

    all_scaling_pass = true
    for I0 in I0_values
        # Create uniform sinogram (all zeros = full transmission)
        sinogram_template = zeros(Float32, n_cols, n_rows, n_angles)

        # Run multiple trials to get statistics
        variances = Float64[]
        means = Float64[]

        for trial in 1:n_trials
            model = default_detector_model(I0=I0, electronic_noise_std=0.0, seed=trial)
            sinogram = copy(sinogram_template)
            add_quantum_noise!(sinogram, model)

            # Convert to intensity domain to check Poisson statistics
            intensity = I0 .* exp.(-sinogram)
            push!(variances, var(intensity[:]))
            push!(means, mean(intensity[:]))
        end

        # Average statistics across trials
        avg_variance = mean(variances)
        avg_mean = mean(means)

        # For Poisson: variance = mean, so σ/μ = 1/√μ
        measured_ratio = sqrt(avg_variance) / avg_mean
        expected_ratio = 1.0 / sqrt(avg_mean)

        error_pct = abs(measured_ratio - expected_ratio) / expected_ratio * 100
        status = error_pct < 5.0 ? "PASS" : "FAIL"

        if error_pct >= 5.0
            all_scaling_pass = false
        end

        println("  $(lpad(Int(I0), 8))      $(round(expected_ratio, digits=5))        $(round(measured_ratio, digits=5))        $(round(error_pct, digits=1))% ($status)")
    end

    results["variance_scaling"] = all_scaling_pass
    println("\n  Result: Variance scaling $(all_scaling_pass ? "PASSED" : "FAILED")")

    # =========================================================================
    # Test 2: Noise variance in projection domain
    # =========================================================================
    println("\n[2] Noise Variance in Projection Domain")
    println("-" ^ 50)

    I0 = 1e5
    n_trials = 20
    projection_values = [0.0, 0.5, 1.0, 2.0]  # Different attenuation levels

    println("\n  Projection    λ = I₀×e⁻ᵖ    Expected σ_p    Measured σ_p    Error")
    println("  " * "-" ^ 65)

    all_proj_pass = true
    for p in projection_values
        sinogram_template = fill(Float32(p), n_cols, n_rows, n_angles)

        # Collect noise samples
        noise_samples = Float64[]
        for trial in 1:n_trials
            model = default_detector_model(I0=I0, electronic_noise_std=0.0, seed=trial)
            sinogram = copy(sinogram_template)
            add_quantum_noise!(sinogram, model)

            # Collect all pixels
            append!(noise_samples, sinogram[:] .- p)
        end

        measured_std = std(noise_samples)

        # Expected: σ_p = 1/√λ where λ = I₀ × exp(-p)
        λ = I0 * exp(-p)
        expected_std = 1.0 / sqrt(λ)

        error_pct = abs(measured_std - expected_std) / expected_std * 100
        status = error_pct < 10.0 ? "PASS" : "FAIL"

        if error_pct >= 10.0
            all_proj_pass = false
        end

        println("  $(lpad(p, 8))        $(lpad(round(Int, λ), 8))       $(round(expected_std, digits=5))        $(round(measured_std, digits=5))        $(round(error_pct, digits=1))% ($status)")
    end

    results["projection_variance"] = all_proj_pass
    println("\n  Result: Projection variance $(all_proj_pass ? "PASSED" : "FAILED")")

    # =========================================================================
    # Test 3: Noise Power Spectrum (NPS) Shape
    # =========================================================================
    println("\n[3] Noise Power Spectrum (NPS) Shape")
    println("-" ^ 50)

    I0 = 1e5
    n_cols_nps, n_rows_nps = 256, 128
    n_trials_nps = 50

    # Accumulate NPS over multiple realizations
    nps_sum = zeros(Float64, n_cols_nps, n_rows_nps)

    for trial in 1:n_trials_nps
        sinogram_clean = zeros(Float32, n_cols_nps, n_rows_nps, 1)
        model = default_detector_model(I0=I0, electronic_noise_std=0.0, seed=trial)
        sinogram_noisy = copy(sinogram_clean)
        add_quantum_noise!(sinogram_noisy, model)

        # Compute noise
        noise = sinogram_noisy[:, :, 1] .- sinogram_clean[:, :, 1]

        # Compute 2D FFT
        noise_fft = fft(noise)
        nps_sum .+= abs2.(noise_fft) ./ (n_cols_nps * n_rows_nps)
    end

    nps_avg = nps_sum ./ n_trials_nps

    # Shift to center
    nps_centered = fftshift(nps_avg)

    # Compute radial average
    center_x, center_y = n_cols_nps ÷ 2 + 1, n_rows_nps ÷ 2 + 1
    max_radius = min(center_x, center_y) - 1
    radial_nps = zeros(Float64, max_radius)
    radial_count = zeros(Int, max_radius)

    for i in 1:n_cols_nps
        for j in 1:n_rows_nps
            r = sqrt((i - center_x)^2 + (j - center_y)^2)
            r_idx = round(Int, r)
            if 1 <= r_idx <= max_radius
                radial_nps[r_idx] += nps_centered[i, j]
                radial_count[r_idx] += 1
            end
        end
    end

    radial_nps ./= max.(radial_count, 1)

    # Check flatness (white noise should have flat NPS)
    # Compute coefficient of variation of NPS
    valid_bins = findall(radial_count .> 10)
    nps_cv = std(radial_nps[valid_bins]) / mean(radial_nps[valid_bins])

    # For white noise, CV should be < 30% (allowing for statistical fluctuations)
    nps_is_flat = nps_cv < 0.30

    println("\n  NPS Analysis:")
    println("  Mean NPS value: $(round(mean(radial_nps[valid_bins]), sigdigits=4))")
    println("  NPS coefficient of variation: $(round(nps_cv * 100, digits=1))%")
    println("  Flatness criterion (CV < 30%): $(nps_is_flat ? "PASSED" : "FAILED")")

    results["nps_flatness"] = nps_is_flat

    # =========================================================================
    # Test 4: Noise is Spatially Uncorrelated (White Noise)
    # =========================================================================
    println("\n[4] Spatial Autocorrelation (White Noise Test)")
    println("-" ^ 50)

    I0 = 1e5
    n_cols_ac, n_rows_ac = 128, 64
    n_trials_ac = 30

    autocorr_sum = zeros(Float64, 21)  # Lag 0-20

    for trial in 1:n_trials_ac
        sinogram_clean = zeros(Float32, n_cols_ac, n_rows_ac, 1)
        model = default_detector_model(I0=I0, electronic_noise_std=0.0, seed=trial)
        sinogram_noisy = copy(sinogram_clean)
        add_quantum_noise!(sinogram_noisy, model)

        noise = vec(sinogram_noisy[:, :, 1])
        noise_centered = noise .- mean(noise)
        var_noise = var(noise_centered)

        for lag in 0:20
            if lag < length(noise_centered)
                autocorr = mean(noise_centered[1:end-lag] .* noise_centered[1+lag:end]) / var_noise
                autocorr_sum[lag+1] += autocorr
            end
        end
    end

    autocorr_avg = autocorr_sum ./ n_trials_ac

    # For white noise: autocorr[0] ≈ 1, autocorr[lag>0] ≈ 0
    lag0_pass = abs(autocorr_avg[1] - 1.0) < 0.05
    lag_others_pass = all(abs.(autocorr_avg[2:end]) .< 0.1)

    println("\n  Autocorrelation:")
    println("  Lag 0: $(round(autocorr_avg[1], digits=4)) (expected: 1.0)")
    println("  Lag 1: $(round(autocorr_avg[2], digits=4)) (expected: ~0)")
    println("  Lag 5: $(round(autocorr_avg[6], digits=4)) (expected: ~0)")
    println("  Max |autocorr[lag>0]|: $(round(maximum(abs.(autocorr_avg[2:end])), digits=4))")

    white_noise_pass = lag0_pass && lag_others_pass
    println("\n  Result: White noise test $(white_noise_pass ? "PASSED" : "FAILED")")

    results["white_noise"] = white_noise_pass

    # =========================================================================
    # Test 5: Gaussian Approximation Validity
    # =========================================================================
    println("\n[5] Gaussian Approximation Validity")
    println("-" ^ 50)

    # For high counts (CT regime), Gaussian approximation should be excellent
    I0_values_gauss = [1e3, 1e4, 1e5, 1e6]

    println("\n  I₀           Skewness (exp 0)    Kurtosis (exp 3)")
    println("  " * "-" ^ 50)

    all_gauss_pass = true
    for I0 in I0_values_gauss
        sinogram_template = zeros(Float32, 256, 64, 1)

        # Collect noise samples
        all_samples = Float64[]
        for trial in 1:20
            model = default_detector_model(I0=I0, electronic_noise_std=0.0, seed=trial)
            sinogram = copy(sinogram_template)
            add_quantum_noise!(sinogram, model)

            # Convert to intensity to check distribution
            intensity = I0 .* exp.(-sinogram)
            # Normalize to unit variance for comparison
            normalized = (intensity[:] .- mean(intensity)) ./ std(intensity)
            append!(all_samples, normalized)
        end

        # Compute skewness and kurtosis
        n = length(all_samples)
        m = mean(all_samples)
        s = std(all_samples)
        skewness = mean((all_samples .- m).^3) / s^3
        kurtosis = mean((all_samples .- m).^4) / s^4

        # For Gaussian: skewness ≈ 0, kurtosis ≈ 3
        # For Poisson with high λ: skewness ≈ 1/√λ ≈ 0, kurtosis ≈ 3 + 1/λ ≈ 3
        skew_pass = abs(skewness) < 0.1
        kurt_pass = abs(kurtosis - 3.0) < 0.2

        status = (skew_pass && kurt_pass) ? "PASS" : "FAIL"
        if !skew_pass || !kurt_pass
            all_gauss_pass = false
        end

        println("  $(lpad(Int(I0), 8))            $(round(skewness, digits=4))              $(round(kurtosis, digits=3))      ($status)")
    end

    results["gaussian_approximation"] = all_gauss_pass
    println("\n  Result: Gaussian approximation $(all_gauss_pass ? "PASSED" : "FAILED")")

    # =========================================================================
    # Test 6: CatSim Consistency
    # =========================================================================
    println("\n[6] CatSim Implementation Consistency")
    println("-" ^ 50)

    # CatSim applies Poisson noise in intensity domain:
    # λ = photon_count (input), output = Poisson(λ)
    # BasisSimulator uses Gaussian approximation in same domain

    # Test that mean is preserved (no bias)
    I0 = 1e5
    sinogram_template = fill(1.0f0, 128, 32, 1)  # p=1 → λ = I₀×e⁻¹

    mean_intensities = Float64[]
    for trial in 1:100
        model = default_detector_model(I0=I0, electronic_noise_std=0.0, seed=trial)
        sinogram = copy(sinogram_template)
        add_quantum_noise!(sinogram, model)

        # Back to intensity
        intensity = I0 .* exp.(-sinogram)
        push!(mean_intensities, mean(intensity))
    end

    expected_λ = I0 * exp(-1.0)
    measured_mean = mean(mean_intensities)
    bias_pct = abs(measured_mean - expected_λ) / expected_λ * 100

    unbiased = bias_pct < 1.0

    println("\n  Mean Preservation Test:")
    println("  Expected λ: $(round(Int, expected_λ))")
    println("  Measured mean: $(round(measured_mean, digits=1))")
    println("  Bias: $(round(bias_pct, digits=2))%")
    println("  Result: $(unbiased ? "PASSED" : "FAILED") (bias < 1%)")

    results["catsim_consistency"] = unbiased

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    all_passed = all(values(results))

    println("\n  Test                        Result")
    println("  " * "-" ^ 40)
    for (test, passed) in results
        status = passed ? "PASS" : "FAIL"
        println("  $(rpad(test, 28)) $status")
    end
    println("  " * "-" ^ 40)
    println("  OVERALL                     $(all_passed ? "PASS" : "FAIL")")

    return all_passed, results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_quantum_noise_verification()
end
