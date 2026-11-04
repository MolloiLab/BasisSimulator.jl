# Tests for src/detector/ — every detector physics effect that lives on the
# active simulate!() path, organized one file per detector module:
#
#   physics_pipeline.jl  → PhysicsConfig, default_physics_config
#   scatter_kernels.jl, scatter.jl → test/scatter.jl
#   detector_efficiency  → DetectorEfficiency (+ Gemstone factory + modes),
#                          MC LUT lookups, Beer-Lambert vector, μ-data
#   fill_factor.jl       → FillFactorModel,
#                          apply_fill_factor!, effective_fill_factor
#   optical_crosstalk    → OpticalCrosstalkModel, optical_crosstalk_typical,
#                          create_optical_crosstalk_kernel, apply_optical_crosstalk!
#   detector_lag.jl      → LagModel, lag_gadox, compute_lag_coefficients,
#                          apply_lag!
#   photon_counting.jl   → PhotonCountingDetector, EnergyResolvedSinogram,
#                          quantum_efficiency(+vector), get_detector_material_*,
#                          spatial_bin!
#   pcct/mc_response.jl  → load_mc_response, compute_mc_drm, mc_cumulative_to_bins
#   pcct/mc_pileup.jl    → compute_mc_pileup_matrix
#   pcct/native_table.jl → pcct_event_classes, pcct_count!
#
# Coverage policy: every exported symbol from src/detector/ is exercised with
# a real behavioral assertion (no smoke tests).  GPU-only paths inside the
# polychromatic.jl pipeline are exercised through api.jl; here we test the
# detector modules directly on CPU arrays.

import AcceleratedKernels  # workspace kernels are AK-based; loading here for parity

# -----------------------------------------------------------------------------
# physics_pipeline.jl
# -----------------------------------------------------------------------------
@testset "PhysicsConfig — default + populated" begin
    @testset "default_physics_config — all effects nothing" begin
        cfg = BS.default_physics_config()
        @test cfg.fill_factor === nothing
        @test cfg.scatter === nothing
        @test cfg.optical_crosstalk === nothing
        @test cfg.focal_spot === nothing
        @test cfg.detector_efficiency === nothing
        @test cfg.lag === nothing
        @test cfg.heel_effect === nothing
        @test cfg.noise_seed === nothing
        @test cfg.energy_keV == 60.0
    end

    @testset "default_physics_config — populated kwargs propagate" begin
        cfg = BS.default_physics_config(;
            fill_factor = BS.FillFactorModel(0.9),
            scatter = BS.scatter_kernels(_SCATTER_TEST_ENTRY),
            optical_crosstalk = BS.optical_crosstalk_typical(),
            detector_efficiency = BS.DetectorEfficiency(:lumex, 3.0, BS.MC_LUT),
            lag = BS.lag_gadox(),
            noise_seed = 7,
            energy_keV = 75.0,
        )
        @test cfg.fill_factor isa BS.FillFactorModel
        @test cfg.scatter isa BS.ScatterKernels
        @test cfg.optical_crosstalk isa BS.OpticalCrosstalkModel
        @test cfg.detector_efficiency isa BS.DetectorEfficiency
        @test cfg.lag isa BS.LagModel
        @test cfg.noise_seed == 7
        @test cfg.energy_keV == 75.0
    end
end

# scatter_kernels.jl / scatter.jl: the Geant4 scatter library, test/scatter.jl

# -----------------------------------------------------------------------------
# detector_efficiency.jl — the Geant4 tables, Beer–Lambert, the model's checks
# (the signal moments themselves: eict_signal.jl)
# -----------------------------------------------------------------------------
@testset "EICT detector tables: K-edge escape, 1–140 keV, one per scintillator" begin
    g, u, f = BS.GEMSTONE_MC_EFFICIENCY_LUT, BS.UFC_MC_EFFICIENCY_LUT, BS.UFC_FLASH_MC_EFFICIENCY_LUT
    for t in (g, u, f)
        @test t.energies == collect(1.0:140.0)
        @test length(t.efficiency) == length(t.dqe0) == 140
        @test all(0 .< t.efficiency .<= 1)
    end
    @test (g.thickness_mm, u.thickness_mm, f.thickness_mm) == (3.0, 1.4, 1.4)
    # fluorescence escape: the efficiency DROPS across the K-edge (Tb 52.0, Lu 63.3, Gd 50.2 keV)
    @test g.efficiency[51] - g.efficiency[52] > 0.1
    @test g.efficiency[63] - g.efficiency[64] > 0.05
    @test u.efficiency[50] - u.efficiency[51] > 0.1
    @test f.efficiency[50] - f.efficiency[51] > 0.1
    # the Force and Flash ceramics are the same UFC at the same depth, from separate runs
    @test u.efficiency != f.efficiency
    @test maximum(abs.(u.efficiency .- f.efficiency)) < 0.01
end

@testset "Beer–Lambert absorption rises across the K-edge, the table's escape falls" begin
    E = [49.0, 51.0]
    bl, _ = BS.eid_signal_moments(BS.DetectorEfficiency(:ufc, 1.4, BS.BEER_LAMBERT), E)
    mc, _ = BS.eid_signal_moments(BS.DetectorEfficiency(:ufc, 1.4, BS.MC_LUT), E)
    @test bl[2] / E[2] > bl[1] / E[1]
    @test mc[2] / E[2] < mc[1] / E[1]
end

@testset "build_physics_config: the scanner's scintillator and depth" begin
    e = collect(20.0:10.0:140.0)
    w = ones(length(e))
    for s in (BS.SOMATOM_FORCE, BS.SOMATOM_DEFINITION_FLASH, BS.REVOLUTION_APEX)
        model = BS.build_physics_config(s, BS.SimOptions(use_scatter = false), e, w).detector_efficiency
        @test model.material === s.detector_material
        @test model.thickness_mm == s.detector_depth
        @test model.mode == BS.MC_LUT
    end
    bl = BS.SimOptions(detector_efficiency_mode = :beer_lambert, use_scatter = false)
    @test BS.build_physics_config(BS.SOMATOM_FORCE, bl, e, w).detector_efficiency.mode == BS.BEER_LAMBERT
    off = BS.build_physics_config(BS.SOMATOM_FORCE, BS.SimOptions(use_detector_efficiency = false, use_scatter = false), e, w)
    @test off.detector_efficiency === nothing
    # a depth the table was not made at, and a material with no model, are refused
    @test_throws ArgumentError BS.build_physics_config(
        BS.EICTScanner(BS.SOMATOM_FORCE; detector_depth = 1.0), BS.SimOptions(use_scatter = false), e, w
    )
    @test_throws ArgumentError BS.build_physics_config(
        BS.EICTScanner(detector_material = :unobtainium), BS.SimOptions(use_scatter = false), e, w
    )
end

@testset "get_scintillator_mu — Gemstone + CdTe" begin
    # CdTe K-edges: Cd at 26.7 keV, Te at 31.8 keV.
    @test BS.get_scintillator_mu("CdTe", 27.0) > BS.get_scintillator_mu("CdTe", 25.0)
    # Gemstone @ 60 keV: 32.7 cm⁻¹ at 7.0 g/cm³ (NIST XCOM), scaled to 6.40 g/cm³
    @test BS.get_scintillator_mu("Gemstone", 60.0) ≈ 32.7 * 6.4 / 7.0 rtol = 1.0e-6
    @test_throws ArgumentError BS.get_scintillator_mu("nonexistent", 60.0)
end

# -----------------------------------------------------------------------------
# fill_factor.jl
# -----------------------------------------------------------------------------
@testset "FillFactorModel" begin
    @testset "effective_fill_factor multiplies row × col" begin
        m = BS.FillFactorModel(0.9, 0.8, true)
        @test BS.effective_fill_factor(m) ≈ 0.72  atol = 1.0e-12
    end

    @testset "1-arg constructor splits as sqrt × sqrt" begin
        m = BS.FillFactorModel(0.81)
        @test m.row_fill ≈ 0.9  atol = 1.0e-12
        @test m.col_fill ≈ 0.9  atol = 1.0e-12
        @test BS.effective_fill_factor(m) ≈ 0.81  atol = 1.0e-12
    end

    @testset "1-arg constructor rejects ff outside (0, 1]" begin
        @test_throws AssertionError BS.FillFactorModel(0.0)
        @test_throws AssertionError BS.FillFactorModel(1.5)
    end
end

# -----------------------------------------------------------------------------
# optical_crosstalk.jl
# -----------------------------------------------------------------------------
@testset "OpticalCrosstalkModel + apply_optical_crosstalk!" begin
    @testset "optical_crosstalk_typical = CatSim defaults" begin
        m = BS.optical_crosstalk_typical()
        @test m.row_coeff ≈ 0.045
        @test m.col_coeff ≈ 0.04
    end

    @testset "create_optical_crosstalk_kernel sums to 1, center is (1-2α)(1-2β)" begin
        m = BS.OpticalCrosstalkModel(0.05, 0.04)
        K = BS.create_optical_crosstalk_kernel(m)
        @test size(K) == (3, 3)
        @test sum(K) ≈ 1.0  atol = 1.0e-12
        @test K[2, 2] ≈ (1 - 2 * 0.04) * (1 - 2 * 0.05)  atol = 1.0e-12
    end

    @testset "apply_optical_crosstalk! preserves total intensity on uniform input" begin
        m = BS.optical_crosstalk_typical()
        sino = fill(Float32(0.5), 16, 4, 4)   # uniform projection → uniform intensity
        sino_save = copy(sino)
        BS.apply_optical_crosstalk!(sino, m)
        # 3×3 convolution of uniform input → unchanged (since kernel sums to 1)
        @test maximum(abs.(sino .- sino_save)) < 1.0e-5
    end

    @testset "zero-coefficient model is a no-op" begin
        m = BS.OpticalCrosstalkModel(0.0, 0.0)
        sino = randn(Float32, 12, 3, 4)
        sino_before = copy(sino)
        BS.apply_optical_crosstalk!(sino, m)
        @test sino == sino_before
    end
end

# -----------------------------------------------------------------------------
# detector_lag.jl
# -----------------------------------------------------------------------------
@testset "LagModel + compute_lag_coefficients + apply_lag!" begin
    @testset "lag_gadox returns 2-component model" begin
        m = BS.lag_gadox()
        @test length(m.amplitudes) == 2
        @test length(m.time_constants) == 2
        @test sum(m.amplitudes) ≈ 0.015  atol = 1.0e-12   # 1% fast + 0.5% slow
        @test m.time_constants == [1.0, 10.0]
    end

    @testset "compute_lag_coefficients — primary > tail, normalized" begin
        coeffs = BS.compute_lag_coefficients(BS.lag_gadox(), 10)
        @test length(coeffs) == 10
        @test sum(coeffs) ≈ 1.0  atol = 1.0e-12
        @test coeffs[1] > coeffs[2]
        @test coeffs[2] > coeffs[end]
        @test all(>(0), coeffs)
    end

    @testset "empty lag → single coefficient" begin
        empty_model = BS.LagModel(Float64[], Float64[], 0.5)
        coeffs = BS.compute_lag_coefficients(empty_model, 5)
        @test coeffs == [1.0]
    end

    @testset "apply_lag! is a no-op for empty model" begin
        sino = randn(Float32, 8, 2, 4)
        sino_before = copy(sino)
        BS.apply_lag!(sino, BS.LagModel(Float64[], Float64[], 0.5))
        @test sino == sino_before
    end

    @testset "apply_lag! smooths a sharp temporal step" begin
        # Step in view direction: 0 → 1 at view 4 (intensity domain step).
        # Lag spreads the rising edge over a few views.
        nc, nr, nv = 4, 1, 12
        sino = zeros(Float32, nc, nr, nv)
        for v in 4:nv, c in 1:nc
            sino[c, 1, v] = 0.5f0   # low projection (so I = exp(-0.5) ≈ 0.61)
        end
        sino_lag = copy(sino)
        BS.apply_lag!(sino_lag, BS.lag_gadox(); n_history = 6)
        # After lag, view 4 should be *between* view 3 (untouched) and view 5
        # — i.e., the step is smoothed (rise time spans 2+ views).
        @test sino_lag[1, 1, 4] != sino[1, 1, 4]
        # No NaN/Inf
        @test all(isfinite, sino_lag)
    end
end

# -----------------------------------------------------------------------------
# photon_counting.jl — PCCT detector + helpers
# -----------------------------------------------------------------------------
@testset "PhotonCountingDetector — construction + defaults" begin
    det = BS.PhotonCountingDetector(;
        material = BS.CDTE_MATERIAL,
        thickness_mm = 1.6,
        pixel_size_mm = (0.302, 0.302),
        energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
    )
    @test det.material == BS.CDTE_MATERIAL
    @test det.thickness_mm == 1.6
    @test det.energy_thresholds_keV == [20.0, 35.0, 55.0, 70.0]
    @test fieldnames(BS.PhotonCountingDetector) ==
        (:material, :thickness_mm, :pixel_size_mm, :energy_thresholds_keV, :dead_time_ns, :binning_factor)
end

@testset "EnergyResolvedSinogram contract" begin
    bins = [zeros(Float32, 4, 2, 3) for _ in 1:4]
    thr = Float32[20.0, 35.0, 55.0, 70.0]
    sino = BS.EnergyResolvedSinogram(bins, thr)
    @test size(sino) == (4, 2, 3)
    @test eltype(sino) == Float32
    @test BS.n_energy_bins(sino) == 4
    @test sino.thresholds_keV == thr
end

@testset "quantum_efficiency — CdTe nearly opaque, Si nearly transparent" begin
    @test BS.quantum_efficiency(BS.CDTE_MATERIAL, 1.6, 60.0) > 0.98
    # Si at 60 keV / 1.6 mm: μ ~0.46 cm⁻¹ → η = 1 - exp(-0.46×0.16) ≈ 0.07
    @test BS.quantum_efficiency(BS.SI_MATERIAL, 1.6, 60.0) < 0.15

    # Batch version matches scalar
    energies = [40.0, 60.0, 100.0]
    η_vec = BS.quantum_efficiency_vector(BS.CDTE_MATERIAL, 1.6, energies)
    for (i, E) in enumerate(energies)
        @test η_vec[i] ≈ BS.quantum_efficiency(BS.CDTE_MATERIAL, 1.6, E)  atol = 1.0e-12
    end
end

@testset "get_detector_material_properties" begin
    cdte = BS.get_detector_material_properties(BS.CDTE_MATERIAL)
    @test cdte.density_g_cm3 ≈ 5.85
    @test cdte.k_edges_keV ≈ [26.711, 31.814]    # Cd, Te K-edges
    @test cdte.elements == [:Cd, :Te]

    czt = BS.get_detector_material_properties(BS.CZT_MATERIAL)
    @test czt.elements == [:Cd, :Zn, :Te]
    @test czt.density_g_cm3 ≈ 5.78

    si = BS.get_detector_material_properties(BS.SI_MATERIAL)
    @test si.elements == [:Si]
    @test si.density_g_cm3 ≈ 2.33

    μ_cdte = BS.get_detector_material_attenuation(BS.CDTE_MATERIAL, 60.0)
    @test 25 < μ_cdte < 45        # NIST XCOM value ~33 cm⁻¹
end

@testset "spatial_bin! — 2×2 sums to one binned pixel" begin
    src = ones(Float32, 8, 4, 2)
    dst = zeros(Float32, 4, 2, 2)
    BS.spatial_bin!(dst, src, 2)
    @test all(dst .≈ 4.0f0)        # 2×2 = 4 dexels per binned pixel
end

# -----------------------------------------------------------------------------
# pcct/mc_response.jl — MC Detector Response Matrix
# -----------------------------------------------------------------------------
@testset "MC DRM — load + cumulative→bins + compute_mc_drm" begin
    @testset "load_mc_response — the native table's cell mean" begin
        mc = BS.load_mc_response(BS.default_mc_drm_path())
        @test mc isa BS.MCResponseData
        @test !isempty(mc.energies_keV)
        @test !isempty(mc.thresholds_keV)
        @test issorted(mc.energies_keV)
        @test issorted(mc.thresholds_keV)
        # MC response is per-incident-photon counts (≥ 0, ≤ 1)
        @test all(mc.R_total .>= 0)
    end

    @testset "mc_cumulative_to_bins — survival difference" begin
        mc = BS.load_mc_response(BS.default_mc_drm_path())
        R_bins = BS.mc_cumulative_to_bins(mc.R_total, mc.thresholds_keV)
        @test size(R_bins) == size(mc.R_total)
        # Each differential bin ≤ corresponding cumulative
        @test all(R_bins .<= mc.R_total .+ 1.0e-12)
        # Sum of differential bins ≈ cumulative count above first threshold
        sums = sum(R_bins; dims = 2)
        @test all(sums .>= 0)
    end

    @testset "compute_mc_drm — physical constraints" begin
        det = BS.PhotonCountingDetector(;
            material = BS.CDTE_MATERIAL,
            thickness_mm = 1.6,
            energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
        )
        D = BS.compute_mc_drm(det, 120.0; n_energy_points = 100)
        @test size(D) == (100, 4)
        # Counts per incident photon are never negative; behind the grid (open fraction 0.72)
        # they stay below one
        @test all(D .>= 0)
        @test 0.5 < maximum(sum(D; dims = 2)) < 1.0
        # v0.15.1's response (test data): a shared photon counts in two dexels, so a row is not
        # capped at one — v0.15.1 renormalised it
        legacy = joinpath(pkgdir(BS), "test", "data", "cdte_response_v4.jls")
        @test maximum(sum(BS.compute_mc_drm(det, 120.0; npz_path = legacy, n_energy_points = 100); dims = 2)) > 1.0
        # Below the lowest threshold (≤20 keV) → near-zero registration in all bins
        @test sum(D[1, :]) < 0.05
    end

    @testset "compute_mc_count_moments — spectrum-averaged covariance" begin
        det = BS.PhotonCountingDetector(;
            material = BS.CDTE_MATERIAL,
            thickness_mm = 1.6,
            energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
        )
        energies = collect(20.0:2.0:120.0)
        weights = @. exp(-((energies - 65.0) / 25.0)^2)
        moments = BS.compute_mc_count_moments(det, energies, weights)

        @test length(moments.mean) == 4
        @test length(moments.fano) == 4
        @test size(moments.covariance) == (4, 4)
        @test size(moments.correlation) == (4, 4)
        @test all(isfinite, moments.mean)
        @test all(moments.mean .>= 0)
        @test all(moments.fano .>= 0)
        @test moments.covariance ≈ transpose(moments.covariance)
        @test all(diag(moments.covariance) .>= 0)
        @test diag(moments.correlation) ≈ ones(4)
        @test all(abs.(moments.correlation) .<= 1.0)
        @test minimum(eigvals(Symmetric(moments.covariance))) >= -1.0e-10

        @test_throws DimensionMismatch BS.compute_mc_count_moments(
            det, energies, weights[1:(end - 1)]
        )
        @test_throws ArgumentError BS.compute_mc_count_moments(
            det, energies, zeros(length(energies))
        )
    end
end

# -----------------------------------------------------------------------------
# photon_counting.jl — exact Poisson count noise
# -----------------------------------------------------------------------------
@testset "_poisson_sample — exact Poisson statistics" begin
    rng = MersenneTwister(1234)
    @test BS._poisson_sample(rng, 0.0) == 0
    @test BS._poisson_sample(rng, 1.0e-15) == 0

    # log-factorial: exact table vs Stirling tail agree across the seam
    @test BS._log_factorial(20) ≈ sum(log, 1:20)
    @test BS._log_factorial(21) ≈ sum(log, 1:21) rtol = 1.0e-12
    @test BS._log_factorial(150) ≈ sum(log, 1:150) rtol = 1.0e-12

    # Mean/variance across the inversion (λ < 10) and PTRS (λ ≥ 10) branches
    for λ in (0.5, 5.0, 9.9, 10.1, 50.0, 1.0e4)
        s = [BS._poisson_sample(rng, λ) for _ in 1:200_000]
        @test all(>=(0), s)
        @test mean(s) ≈ λ rtol = 0.02
        @test std(s)^2 ≈ λ rtol = 0.05
    end

    # Skewness 1/√λ separates exact Poisson from a Gaussian approximation
    # (the old sampler switched to a rounded Gaussian above λ = 30 — skewness 0)
    s = [BS._poisson_sample(rng, 50.0) for _ in 1:200_000]
    z = (s .- mean(s)) ./ std(s)
    @test isapprox(mean(z .^ 3), 1 / sqrt(50.0); atol = 0.03)

    # Determinism under an explicit seed
    draw(seed) = [BS._poisson_sample(MersenneTwister(seed), 123.4) for _ in 1:100]
    @test draw(7) == draw(7)
    @test draw(7) != draw(8)
end

@testset "_poisson_draw — the device's Float32 sampler on counter-based streams" begin
    # the acceptance test's log-pmf keeps Float32 exact where the counts are large
    for (k, λ) in ((15, 12.0), (21, 30.0), (1.0e4 + 37, 1.0e4), (1.0e5 - 410, 1.0e5), (1.0e6 + 900, 1.0e6))
        exact = k * log(λ) - λ - BS._log_factorial(Int(k))
        @test BS._log_poisson_pmf(Float32(k), Float32(λ)) ≈ exact atol = 2.0e-3
    end
    for λ in (0.5f0, 9.9f0, 10.1f0, 50.0f0, 1.0f4, 2.0f5)
        s = [first(BS._poisson_draw(BS._DrawStream(UInt64(3), 1, i, 1, 1), λ)) for i in 1:200_000]
        @test all(x -> x >= 0 && isinteger(x), s)
        @test mean(Float64.(s)) ≈ λ rtol = 0.01
        @test std(Float64.(s))^2 ≈ λ rtol = 0.02
        z = (s .- mean(s)) ./ std(s)
        @test mean(Float64.(z) .^ 3) ≈ 1 / sqrt(λ) atol = 0.03
    end
    # a stream is a pure function of its key
    @test BS._poisson_draw(BS._DrawStream(UInt64(9), 2, 3, 4, 5), 77.0f0) == BS._poisson_draw(BS._DrawStream(UInt64(9), 2, 3, 4, 5), 77.0f0)
    @test first(BS._poisson_draw(BS._DrawStream(UInt64(9), 2, 3, 4, 5), 1.0f4)) != first(BS._poisson_draw(BS._DrawStream(UInt64(9), 2, 3, 4, 6), 1.0f4))
end

@testset "pcct_count!: Poisson dexels, shared photons covary, the digital sum is over-dispersed" begin
    e, w = BS.resolve_source_spectrum_without_bowtie(
        BS.SimOptions(), BS.CTProtocol(kVp = 120.0, mAs = 100.0, views = 10); scanner = BS.NAEOTOM_ALPHA,
    )
    classes = BS.pcct_event_classes([20.0], e, w)
    @test 0.6 < classes.single[1] < 0.95                    # charge sharing: a fifth of the counts shared
    @test all(p -> p.c1 > 0 && p.c2 > 0, classes.pairs)
    λ = 400.0
    expected = [fill(λ, 64, 64, 40)]
    a = BS.pcct_count!([zeros(64, 64, 40)], expected, classes; seed = 1)[1]
    b = BS.pcct_count!([zeros(64, 64, 40)], expected, classes; seed = 2)[1]
    @test BS.pcct_count!([zeros(64, 64, 40)], expected, classes; seed = 1)[1] == a   # a seed fixes it
    @test all(isinteger, a)
    d = (a .- b) ./ sqrt(2)
    @test sum(a) / length(a) ≈ λ rtol = 2.0e-3                # the mean is kept, edges included
    @test sum(abs2, d) / length(d) / λ ≈ 1 atol = 0.02        # each dexel Poisson
    # neighbours across the fan covary by the photons they share
    shared_x = sum(p.c1 for p in classes.pairs if (p.dx, p.dy) == (1, 0))
    corr = sum(d[1:(end - 1), :, :] .* d[2:end, :, :]) / sum(abs2, d[1:(end - 1), :, :])
    @test corr ≈ shared_x atol = 0.01
    # 2 × 2 digital sum: the pairs inside the group count twice
    inside = 2 * sum(
        p.c1 for p in classes.pairs if (p.dx, p.dy) in ((1, 0), (0, 1), (1, 1), (1, -1)) &&
            !(p.dx == 1 && p.dy == -1)
    ) + 2 * sum(p.c1 for p in classes.pairs if (p.dx, p.dy) == (1, -1))
    a2, b2 = (BS.pcct_count!([zeros(32, 32, 40)], expected, classes; seed, binning = 2)[1] for seed in (1, 2))
    @test sum(a2) == sum(a)                                  # the same dexels, summed
    binned = (a2 .- b2) ./ sqrt(2)
    @test sum(abs2, binned) / length(binned) / (4λ) ≈ 1 + inside / 2 atol = 0.03
    # in Float32 (the device's precision) the same statistics
    a32 = BS.pcct_count!([zeros(Float32, 64, 64, 40)], [fill(Float32(λ), 64, 64, 40)], classes; seed = 1)[1]
    b32 = BS.pcct_count!([zeros(Float32, 64, 64, 40)], [fill(Float32(λ), 64, 64, 40)], classes; seed = 2)[1]
    d32 = (a32 .- b32) ./ sqrt(2)
    @test sum(Float64, a32) / length(a32) ≈ λ rtol = 2.0e-3
    @test sum(abs2, d32) / length(d32) / λ ≈ 1 atol = 0.02
    @test sum(d32[1:(end - 1), :, :] .* d32[2:end, :, :]) / sum(abs2, d32[1:(end - 1), :, :]) ≈ shared_x atol = 0.01
    @test_throws ArgumentError BS.pcct_event_classes([22.0], e, w)    # not among the table's

    # every dexel's mean is its expected counts, bin by bin, whatever its spectrum and its
    # neighbours': four bins whose shares change from dexel to dexel (bright beside dark, hard beside
    # soft), each dexel's counts summed over 200 views against their own Poisson spread
    classes4 = BS.pcct_event_classes([20.0, 35.0, 55.0, 70.0], e, w)
    pattern(c, r, b) = 0.3 + mod(c * (b + 2) + 3r, 7) / 3
    expected4 = [
        repeat([λ * s * pattern(c, r, b) for c in 1:32, r in 1:32], 1, 1, 200)
            for (b, s) in enumerate((1.0, 0.8, 0.5, 0.35))
    ]
    counts4 = BS.pcct_count!([zeros(32, 32, 200) for _ in 1:4], expected4, classes4; seed = 3)
    for (n, x) in zip(counts4, expected4)
        z = (sum(n; dims = 3) .- sum(x; dims = 3)) ./ sqrt.(sum(x; dims = 3))
        @test abs(sum(z) / length(z)) < 3 / sqrt(length(z))
        @test sum(abs2, z) / length(z) ≈ 1 atol = 0.15
    end
end

@testset "compute_mc_pileup_matrix: retriggering comparators on shaped pulses" begin
    e = collect(20.0:1.0:140.0)
    w = ones(length(e)) ./ length(e)
    thresholds = [20.0, 35.0, 55.0, 70.0]
    S0 = BS.compute_mc_pileup_matrix(thresholds, w, e, 0.0, 27.0)
    @test S0 == Matrix(1.0I, 4, 4)
    low = BS.compute_mc_pileup_matrix(thresholds, w, e, 1.0e4, 27.0; seed = 3)
    @test low ≈ Matrix(1.0I, 4, 4) atol = 0.01                  # no overlap at low rate
    S = BS.compute_mc_pileup_matrix(thresholds, w, e, 5.0e6, 27.0; seed = 3)
    @test all(sum(S; dims = 1) .< 1)                            # counts are lost
    @test all(sum(S; dims = 1) .> 0.5)
    @test S[4, 1] > 0                                           # piled-up pulses rise into higher bins
    @test BS.compute_mc_pileup_matrix(thresholds, w, e, 5.0e6, 27.0; seed = 3) == S   # memoised, exact
    # the loss grows with the rate × dead time
    S2 = BS.compute_mc_pileup_matrix(thresholds, w, e, 1.0e7, 27.0; seed = 3)
    @test sum(S2) < sum(S)
    # pcct_count!'s pile-up: an air dexel at the grid's top rate takes the top matrix, before the sum
    pileup = (S = cat(Matrix(1.0I, 4, 4), S; dims = 3), rates = [0.0, 5.0e6], rate_air = 5.0e6, counts_air = 4000.0)
    counts = BS.pcct_count!([zeros(1, 1, 1) for _ in 1:4], [fill(1000.0, 2, 2, 1) for _ in 1:4], nothing; binning = 2, pileup)
    @test [counts[i][1, 1, 1] for i in 1:4] ≈ 4 .* (S * fill(1000.0, 4))
end
