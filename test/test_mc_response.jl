# =============================================================================
# Test MC Response Loader
# =============================================================================
# Run: julia --project=. test/test_mc_response.jl

using Test
using BasisSimulator

const load_mc_response = BasisSimulator.load_mc_response
const mc_cumulative_to_bins = BasisSimulator.mc_cumulative_to_bins
const compute_mc_drm = BasisSimulator.compute_mc_drm
const mc_drm_summary = BasisSimulator.mc_drm_summary
const naeotom_detector_standard = BasisSimulator.naeotom_detector_standard
const default_mc_drm_path = BasisSimulator.default_mc_drm_path

println("=" ^ 60)
println("MC Response Loader — Validation Tests")
println("=" ^ 60)

# Use bundled response file
const NPZ_PATH = default_mc_drm_path()
println("DRM path: $NPZ_PATH")

@testset "MC Response Loader" begin

    # =========================================================================
    # 1. Load NPZ file
    # =========================================================================
    @testset "1. Load NPZ" begin
        @test isfile(NPZ_PATH)
        data = load_mc_response(NPZ_PATH)

        @test length(data.energies_keV) == 140
        @test data.energies_keV[1] == 1
        @test data.energies_keV[end] == 140

        @test length(data.thresholds_keV) == 9
        @test data.thresholds_keV == [20, 25, 30, 35, 55, 60, 70, 75, 90]

        @test size(data.R_total) == (140, 9)
        @test size(data.R_perpixel) == (140, 3, 3, 9)
        @test size(data.Var_total) == (140, 9)
        @test size(data.Cov_total) == (140, 9, 9)

        println("  ✓ NPZ loaded: $(size(data.R_total)) R_total, thresholds=$(data.thresholds_keV)")
    end

    # =========================================================================
    # 2. Cumulative → bin conversion
    # =========================================================================
    @testset "2. Cumulative to bins" begin
        data = load_mc_response(NPZ_PATH)
        R_bins = mc_cumulative_to_bins(data.R_total, data.thresholds_keV)

        @test size(R_bins) == size(data.R_total)
        @test all(R_bins .>= 0.0)

        # At 1 keV (below all thresholds): all bins should be ~0
        @test sum(R_bins[1, :]) ≈ 0.0 atol=1e-6

        # The sum of bins should equal the lowest-threshold cumulative count
        # R_bins[e, 1] + R_bins[e, 2] + ... + R_bins[e, n] = R_total[e, 1]
        for e in [20, 50, 100]
            bin_sum = sum(R_bins[e, :])
            cumulative_lowest = data.R_total[e, 1]
            @test isapprox(bin_sum, cumulative_lowest, atol=1e-6)
        end

        println("  ✓ Cumulative → bin conversion verified")
        println("    R_bins at 50 keV: $(round.(R_bins[50, :], digits=4))")
        println("    R_bins at 100 keV: $(round.(R_bins[100, :], digits=4))")
    end

    # =========================================================================
    # 3. DRM computation with NAEOTOM thresholds
    # =========================================================================
    @testset "3. MC DRM" begin
        det = naeotom_detector_standard()
        D = compute_mc_drm(det, 120.0; n_energy_points=200)  # uses default path

        @test size(D) == (200, length(det.energy_thresholds_keV))

        # Row sums ≤ 1 (probability conservation)
        for i in 1:200
            @test sum(D[i, :]) <= 1.0 + 1e-10
        end

        # All values ≥ 0
        @test all(D .>= 0.0)

        # Below 20 keV threshold: should be ~0
        idx_10 = argmin(abs.(collect(range(1.0, 120.0, length=200)) .- 10.0))
        @test sum(D[idx_10, :]) < 0.1

        # At 60 keV: should have substantial counts in some bin
        idx_60 = argmin(abs.(collect(range(1.0, 120.0, length=200)) .- 60.0))
        @test sum(D[idx_60, :]) > 0.3

        println("  ✓ MC DRM computed: $(size(D))")
        println("\n  DRM summary:")
        mc_drm_summary(D, det.energy_thresholds_keV, 120.0)
    end

    # =========================================================================
    # 4. Comparison: MC DRM vs analytical DRM
    # =========================================================================
    @testset "4. MC vs Analytical DRM" begin
        det = naeotom_detector_standard()
        D_mc = compute_mc_drm(det, 120.0; n_energy_points=200)
        D_analytical = compute_unified_drm(det, 120.0; n_energy_points=200)

        @test size(D_mc) == size(D_analytical)

        E_grid = collect(range(1.0, 120.0, length=200))

        println("\n  MC vs Analytical DRM at key energies:")
        for E in [30.0, 50.0, 60.0, 80.0, 100.0]
            idx = argmin(abs.(E_grid .- E))
            row_mc = D_mc[idx, :]
            row_analytic = D_analytical[idx, :]
            println("    E=$(Int(E)) keV:")
            println("      MC:         $(round.(row_mc, digits=4))  sum=$(round(sum(row_mc), digits=4))")
            println("      Analytical:  $(round.(row_analytic, digits=4))  sum=$(round(sum(row_analytic), digits=4))")
        end
    end
end

println("\n" * "=" ^ 60)
println("All MC response loader tests complete.")
println("=" ^ 60)
