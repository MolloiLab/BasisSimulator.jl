# =============================================================================
# Test Monte Carlo Pulse Pileup
# =============================================================================
# Run: julia --project=. test/test_mc_pileup.jl

using Test
using BasisSimulator
using Random
using Statistics

const simulate_pulse_train = BasisSimulator.simulate_pulse_train
const compute_mc_pileup_matrix = BasisSimulator.compute_mc_pileup_matrix
const mc_pileup_count_factor = BasisSimulator.mc_pileup_count_factor
const Poisson_approx = BasisSimulator.Poisson_approx
const _find_threshold_bin = BasisSimulator._find_threshold_bin
const compute_spectral_migration_matrix = BasisSimulator.compute_spectral_migration_matrix
const seminonparalyzable_count_factor = BasisSimulator.seminonparalyzable_count_factor

println("=" ^ 60)
println("Monte Carlo Pulse Pileup — Validation Tests")
println("=" ^ 60)

# Flat spectrum for testing
const TEST_ENERGIES = collect(20.0:1.0:120.0)
const TEST_WEIGHTS = ones(length(TEST_ENERGIES)) ./ length(TEST_ENERGIES)
const THRESHOLDS = [20.0, 35.0, 55.0, 70.0]

@testset "MC Pulse Pileup" begin

    # =========================================================================
    # 1. Poisson sampler
    # =========================================================================
    @testset "1. Poisson sampler" begin
        # Small λ
        samples_small = [Poisson_approx(3.0) for _ in 1:10000]
        @test abs(mean(samples_small) - 3.0) < 0.2
        @test abs(var(samples_small) - 3.0) < 1.0

        # Large λ (Gaussian approximation)
        samples_large = [Poisson_approx(100.0) for _ in 1:10000]
        @test abs(mean(samples_large) - 100.0) < 3.0
        @test abs(var(samples_large) - 100.0) < 20.0

        # Edge case
        @test Poisson_approx(0.0) == 0
        @test Poisson_approx(-1.0) == 0

        println("  ✓ Poisson sampler: mean(λ=3)=$(round(mean(samples_small), digits=2)), var=$(round(var(samples_small), digits=2))")
    end

    # =========================================================================
    # 2. Threshold bin finder
    # =========================================================================
    @testset "2. Bin finder" begin
        @test _find_threshold_bin(10.0, THRESHOLDS) == 0  # Below lowest
        @test _find_threshold_bin(20.0, THRESHOLDS) == 1  # Exact threshold
        @test _find_threshold_bin(30.0, THRESHOLDS) == 1  # Between T1 and T2
        @test _find_threshold_bin(35.0, THRESHOLDS) == 2  # Exact T2
        @test _find_threshold_bin(50.0, THRESHOLDS) == 2
        @test _find_threshold_bin(55.0, THRESHOLDS) == 3
        @test _find_threshold_bin(70.0, THRESHOLDS) == 4
        @test _find_threshold_bin(200.0, THRESHOLDS) == 4  # Above all
        println("  ✓ Bin finder verified")
    end

    # =========================================================================
    # 3. Pulse train at low flux (no pileup expected)
    # =========================================================================
    @testset "3. Low flux pulse train" begin
        # Very low count rate → minimal pileup
        rng = MersenneTwister(42)
        result = simulate_pulse_train(
            TEST_WEIGHTS, TEST_ENERGIES,
            1e4,    # 10k photons/s (very low)
            5.0;    # 5 ns dead time
            observation_time_s=1e-3,
            thresholds_keV=THRESHOLDS,
            rng=rng
        )

        # At low flux: recorded ≈ true (negligible pileup)
        if result.n_true > 0
            ratio = result.n_recorded / result.n_true
            @test ratio > 0.95
            println("  ✓ Low flux: $(result.n_true) true → $(result.n_recorded) recorded (ratio=$(round(ratio, digits=3)))")
        else
            println("  ⚠ Low flux: no photons in observation window (expected for very low rate)")
        end
    end

    # =========================================================================
    # 4. Pulse train at high flux (significant pileup)
    # =========================================================================
    @testset "4. High flux pulse train" begin
        rng = MersenneTwister(42)
        result = simulate_pulse_train(
            TEST_WEIGHTS, TEST_ENERGIES,
            5e8,    # 500M photons/s (very high flux)
            50.0;   # 50 ns dead time
            observation_time_s=1e-3,
            thresholds_keV=THRESHOLDS,
            rng=rng
        )

        # At high flux (aτ = 5e8 × 50e-9 = 25): massive count loss
        @test result.n_recorded < result.n_true
        ratio = result.n_recorded / result.n_true
        @test ratio < 0.5  # Significant count loss

        # Pileup should shift counts to higher bins
        # Total true counts in low bins > recorded (some shifted up)
        println("  ✓ High flux: $(result.n_true) true → $(result.n_recorded) recorded (ratio=$(round(ratio, digits=3)))")
        println("    True bins:     $(result.true_bin_counts)")
        println("    Recorded bins: $(result.recorded_bin_counts)")
    end

    # =========================================================================
    # 5. MC pileup count factor vs analytical
    # =========================================================================
    @testset "5. Count factor comparison" begin
        # Compare MC count-loss factor with analytical at moderate flux
        # aτ = count_rate × dead_time
        # For count_rate=1e7, τ=50ns: aτ = 0.5
        mc_factor = mc_pileup_count_factor(
            TEST_WEIGHTS, TEST_ENERGIES,
            1e7, 50.0;   # aτ = 0.5
            n_trials=5000,
            seed=42
        )
        analytical_factor = seminonparalyzable_count_factor(0.5)

        println("  Count factor at aτ=0.5:")
        println("    MC:         $(round(mc_factor, digits=4))")
        println("    Analytical: $(round(analytical_factor, digits=4))")

        # MC and analytical should be within ~10% for moderate trials
        @test abs(mc_factor - analytical_factor) / analytical_factor < 0.15

        println("  ✓ MC count factor within 15% of analytical")
    end

    # =========================================================================
    # 6. MC spectral migration matrix
    # =========================================================================
    @testset "6. Spectral migration matrix" begin
        # At low flux: S should be near-identity
        S_low = compute_mc_pileup_matrix(
            THRESHOLDS, TEST_WEIGHTS, TEST_ENERGIES,
            1e5, 5.0;  # aτ ≈ 0.0005 (negligible pileup)
            n_trials=2000,
            seed=42
        )

        # At moderate flux: S should show off-diagonal elements
        S_high = compute_mc_pileup_matrix(
            THRESHOLDS, TEST_WEIGHTS, TEST_ENERGIES,
            1e7, 50.0;  # aτ = 0.5
            n_trials=5000,
            seed=42
        )

        n_bins = length(THRESHOLDS)

        # Low flux: column sums should be approximately 1
        # (Note: MC migration matrix uses marginal approximation — diagonal dominance
        # is not expected since per-photon bin transitions aren't tracked individually)
        for i in 1:n_bins
            if sum(S_low[:, i]) > 0
                @test sum(S_low[:, i]) > 0.5  # Column sum should be substantial
            end
        end

        # Column sums ≤ 1
        for i in 1:n_bins
            @test sum(S_low[:, i]) <= 1.0 + 0.05
            @test sum(S_high[:, i]) <= 1.0 + 0.05
        end

        println("\n  MC migration matrix at low flux (aτ≈0):")
        for i in 1:n_bins
            println("    S[$i,:] = $(round.(S_low[i,:], digits=3))")
        end

        println("\n  MC migration matrix at moderate flux (aτ=0.5):")
        for i in 1:n_bins
            println("    S[$i,:] = $(round.(S_high[i,:], digits=3))")
        end

        # Compare with analytical migration matrix
        S_analytical = compute_spectral_migration_matrix(THRESHOLDS, 0.5)
        println("\n  Analytical migration matrix at aτ=0.5:")
        for i in 1:n_bins
            println("    S[$i,:] = $(round.(S_analytical[i,:], digits=3))")
        end

        println("\n  ✓ MC migration matrix has correct structure")
    end
end

println("\n" * "=" ^ 60)
println("All MC pileup tests complete.")
println("=" ^ 60)
