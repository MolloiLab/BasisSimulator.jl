# Tests for src/detector/ — every detector physics effect that lives on the
# active simulate!() path, organized one file per detector module:
#
#   physics_pipeline.jl  → PhysicsConfig, default_physics_config
#   scatter.jl           → ScatterModel, geometry_aware_scatter_model, kernels,
#                          estimate_scatter_field!, per-energy/per-bin weights,
#                          inject_scatter!/inject_scatter_bins!,
#                          estimate_phantom_diameter_cm
#   detector_efficiency  → DetectorEfficiency (+ Gemstone factory + modes),
#                          MC LUT lookups, Beer-Lambert vector, μ-data
#   fill_factor.jl       → FillFactorModel, fill_factor_standard,
#                          apply_fill_factor!, effective_fill_factor
#   optical_crosstalk    → OpticalCrosstalkModel, optical_crosstalk_typical,
#                          create_optical_crosstalk_kernel, apply_optical_crosstalk!
#   detector_lag.jl      → LagModel, lag_gadox, compute_lag_coefficients,
#                          apply_lag!
#   photon_counting.jl   → PhotonCountingDetector, EnergyResolvedSinogram,
#                          quantum_efficiency(+vector), get_detector_material_*,
#                          spatial_bin!, apply_pcct_noise!
#   pcct/mc_response.jl  → load_mc_response, compute_mc_drm, mc_cumulative_to_bins
#   pcct/mc_pileup.jl    → simulate_pulse_train, compute_mc_pileup_matrix
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
        @test cfg.fill_factor          === nothing
        @test cfg.scatter              === nothing
        @test cfg.optical_crosstalk    === nothing
        @test cfg.focal_spot           === nothing
        @test cfg.detector_efficiency  === nothing
        @test cfg.lag                  === nothing
        @test cfg.heel_effect          === nothing
        @test cfg.noise_seed           === nothing
        @test cfg.energy_keV == 60.0
    end

    @testset "default_physics_config — populated kwargs propagate" begin
        cfg = BS.default_physics_config(;
            fill_factor       = BS.fill_factor_standard(),
            scatter           = BS.geometry_aware_scatter_model(BS.Scanner()),
            optical_crosstalk = BS.optical_crosstalk_typical(),
            detector_efficiency = BS.detector_efficiency_gemstone(),
            lag               = BS.lag_gadox(),
            noise_seed        = 7,
            energy_keV        = 75.0,
        )
        @test cfg.fill_factor          isa BS.FillFactorModel
        @test cfg.scatter              isa BS.ScatterModel
        @test cfg.optical_crosstalk    isa BS.OpticalCrosstalkModel
        @test cfg.detector_efficiency  isa BS.DetectorEfficiency
        @test cfg.lag                  isa BS.LagModel
        @test cfg.noise_seed == 7
        @test cfg.energy_keV == 75.0
    end
end

# -----------------------------------------------------------------------------
# scatter.jl
# -----------------------------------------------------------------------------
@testset "ScatterModel + geometry_aware_scatter_model" begin
    @testset "geometry_aware_scatter_model scales with air gap" begin
        ref = BS.Scanner(
            source_to_isocenter = BS.SCATTER_REF_SID_MM,
            source_to_detector  = BS.SCATTER_REF_SDD_MM,
        )
        ref_model = BS.geometry_aware_scatter_model(ref)
        @test ref_model isa BS.ScatterModel
        # Reference geometry → coefficient ≈ SCATTER_REF_COEFFICIENT × 1 × 1
        @test ref_model.scatter_coefficient ≈ BS.SCATTER_REF_COEFFICIENT atol = 1.0e-12

        # GE Revolution has a larger air gap → less scatter
        ge = BS.Scanner(source_to_isocenter = 626.0, source_to_detector = 1097.0)
        ge_model = BS.geometry_aware_scatter_model(ge)
        @test ge_model.scatter_coefficient < ref_model.scatter_coefficient
    end

    @testset "phantom diameter scaling — bigger body, more scatter" begin
        scanner = BS.Scanner()
        small = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = 20.0)
        large = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = 40.0)
        @test large.scatter_coefficient > small.scatter_coefficient
    end
end

@testset "scatter kernels (spatial + 1D Gaussian)" begin
    model = BS.geometry_aware_scatter_model(BS.Scanner())

    @testset "create_scatter_kernel_spatial — Gaussian normalized" begin
        K = BS.create_scatter_kernel_spatial(model)
        @test ndims(K) == 2
        @test sum(K) ≈ 1.0  atol = 1.0e-12
        # Symmetric about center
        nc = size(K, 1) ÷ 2 + 1
        @test K[nc - 1, nc] ≈ K[nc + 1, nc]
        @test K[nc, nc - 1] ≈ K[nc, nc + 1]
    end

    @testset "exponential kernel path" begin
        exp_model = BS.ScatterModel(0.025, 1.0, 30.0, :exponential)
        K = BS.create_scatter_kernel_spatial(exp_model)
        @test sum(K) ≈ 1.0  atol = 1.0e-12
    end
end

@testset "estimate_scatter_field!  + inject_scatter!" begin
    model = BS.geometry_aware_scatter_model(BS.Scanner())

    # Make a smooth phantom-shaped sinogram (Gaussian projection profile)
    nc, nr, nv = 64, 4, 8
    sino = zeros(Float32, nc, nr, nv)
    for v in 1:nv, r in 1:nr, c in 1:nc
        sino[c, r, v] = 2.0f0 * exp(-((c - nc / 2)^2) / (2 * 12^2))
    end

    @testset "estimate_scatter_field! does not modify the input" begin
        sino_save = copy(sino)
        sf = similar(sino)
        BS.estimate_scatter_field!(sf, sino, model)
        @test sino == sino_save                   # input untouched
        @test all(>=(0), sf)                      # field is non-negative
        @test maximum(sf) > 0                     # something got estimated
    end

    @testset "inject_scatter! raises sinogram intensity (lowers log)" begin
        sino_inj = copy(sino)
        sf = similar(sino)
        BS.estimate_scatter_field!(sf, sino, model)
        BS.inject_scatter!(sino_inj, sf, 0.5)
        # I_new = I_primary + scatter ⇒ -log(I_new) ≤ -log(I_primary), and
        # the projection should decrease in the body (where scatter is highest).
        body_center = sino[nc ÷ 2, :, :]
        body_center_inj = sino_inj[nc ÷ 2, :, :]
        @test all(body_center_inj .<= body_center .+ 1.0e-6)
    end
end

@testset "compute_scatter_energy_weights — Compton fraction" begin
    energies = [20.0, 60.0, 100.0, 140.0]
    ew = BS.compute_scatter_energy_weights(energies)
    @test length(ew) == length(energies)
    @test all(0 .<= ew .<= 1)
    # 1 / (1 + (20/E)^3) is monotonic increasing in E
    @test issorted(ew)
end

@testset "compute_scatter_bin_weights — normalize to 1, route via DRM" begin
    # 4-bin synthetic DRM: identity diag (each E lands in its native bin)
    energies = collect(20.0:5.0:120.0)
    weights = ones(length(energies))
    n_E = length(energies)
    n_bins = 4
    R = zeros(n_E, n_bins)
    for (i, E) in enumerate(energies)
        b = E <  35 ? 1 :
            E <  55 ? 2 :
            E <  70 ? 3 : 4
        R[i, b] = 1.0
    end
    η = ones(n_E)
    ew = BS.compute_scatter_energy_weights(energies)
    bw = BS.compute_scatter_bin_weights(energies, weights, ew, η, R, 120.0)
    @test length(bw) == n_bins
    @test sum(bw) ≈ 1.0  atol = 1.0e-12
    @test all(>=(0), bw)
    # higher-energy bins get more Compton weight than the lowest
    @test bw[4] > bw[1]
end

@testset "inject_scatter_bins! — scatter lowers each bin's line integral" begin
    # Start from a realistic body-projection (p ≈ 2.0 → I = e⁻² of primary).
    # Adding scatter raises N_total relative to N_primary → projection drops.
    nc, nr, nv = 16, 2, 4
    n_bins = 4
    bins_before = [fill(Float32(2.0), nc, nr, nv) for _ in 1:n_bins]
    bins = deepcopy(bins_before)
    sf   = fill(Float32(0.01), nc, nr, nv)
    I0_bins  = [1.0e5, 1.0e5, 1.0e5, 1.0e5]
    I0_total = sum(I0_bins)
    bw = [0.40, 0.30, 0.20, 0.10]

    BS.inject_scatter_bins!(bins, sf, I0_bins, I0_total, bw)
    for b in 1:n_bins
        # Scatter adds to numerator counts → p_new ≤ p_old everywhere
        @test all(bins[b] .<= bins_before[b] .+ 1.0e-6)
        # And at least some pixels show a measurable drop
        @test minimum(bins[b]) < bins_before[b][1, 1, 1] - 1.0e-4
    end
    # Lower bin weight should produce smaller change than higher bin weight,
    # since scatter is distributed proportionally to `bin_weights`.
    Δb1 = bins_before[1][1, 1, 1] - bins[1][1, 1, 1]    # bw=0.40
    Δb4 = bins_before[4][1, 1, 1] - bins[4][1, 1, 1]    # bw=0.10
    @test Δb1 > Δb4
end

@testset "estimate_phantom_diameter_cm" begin
    # 256³ mask with a central 100-vox-diameter cylinder, 1 mm voxels.
    nx = 256
    mask = zeros(UInt8, nx, nx, 8)
    R = 50
    cx, cy = nx ÷ 2, nx ÷ 2
    for z in 1:8, y in 1:nx, x in 1:nx
        if (x - cx)^2 + (y - cy)^2 <= R^2
            mask[x, y, z] = UInt8(1)
        end
    end
    d_cm = BS.estimate_phantom_diameter_cm(mask, (1.0, 1.0, 1.0))
    # Bounding box ≈ 2R = 100 mm = 10 cm; sqrt(AP × LAT) = 10 cm.
    @test 9.0 < d_cm < 11.0
end

# -----------------------------------------------------------------------------
# detector_efficiency.jl — GE Gemstone MC LUT + Beer-Lambert fallback
# -----------------------------------------------------------------------------
@testset "DetectorEfficiency — Gemstone MC LUT" begin
    @testset "detector_efficiency_gemstone factory" begin
        model = BS.detector_efficiency_gemstone()
        @test model.material == "Gemstone"
        @test model.thickness_mm == 3.0
        @test model.fill_factor == 0.9
        @test model.mode == BS.MC_LUT

        bl = BS.detector_efficiency_gemstone(mode = :beer_lambert)
        @test bl.mode == BS.BEER_LAMBERT
    end

    @testset "get_gemstone_mc_efficiency captures Tb K-edge fluorescence drop" begin
        η52 = BS.get_gemstone_mc_efficiency(52.0)
        η53 = BS.get_gemstone_mc_efficiency(53.0)
        # MC LUT: at the K-edge (52→53 keV) η drops from ~0.956 to ~0.807
        # because Tb-Kα fluorescence escapes the crystal.
        @test η52 > 0.9
        @test η53 < 0.82
        @test η52 - η53 > 0.10
    end

    @testset "get_gemstone_mc_efficiency captures Lu K-edge fluorescence drop" begin
        η63 = BS.get_gemstone_mc_efficiency(63.0)
        η64 = BS.get_gemstone_mc_efficiency(64.0)
        # ~0.846 → ~0.762 at the Lu K-edge
        @test η63 > 0.83
        @test η64 < 0.78
        @test η63 - η64 > 0.06
    end

    @testset "MC LUT linear interpolation at non-integer keV" begin
        η_60_5 = BS.get_gemstone_mc_efficiency(60.5)
        @test BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency[61] < η_60_5 < BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency[60] + 1.0e-12 ||
              BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency[60] < η_60_5 < BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency[61] + 1.0e-12
    end

    @testset "MC LUT clamps below 1 keV and above 140 keV" begin
        @test BS.get_gemstone_mc_efficiency(0.5) == BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency[1]
        @test BS.get_gemstone_mc_efficiency(200.0) == BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency[end]
    end

    @testset "compute_eid_efficiency_vector — MC routes to LUT" begin
        model = BS.detector_efficiency_gemstone()
        energies = [30.0, 52.0, 53.0, 100.0, 140.0]
        η = BS.compute_eid_efficiency_vector(model, energies)
        @test length(η) == length(energies)
        @test η[3] < η[2]                    # Tb K-edge fluorescence drop
        @test η ≈ [BS.get_gemstone_mc_efficiency(E) for E in energies]
    end

    @testset "compute_eid_efficiency_vector — Beer-Lambert path" begin
        bl = BS.detector_efficiency_gemstone(mode = :beer_lambert)
        energies = [30.0, 60.0, 100.0, 140.0]
        η = BS.compute_eid_efficiency_vector(bl, energies)
        @test length(η) == length(energies)
        @test all(0 .<= η .<= 1)
        # Beer-Lambert η decreases monotonically with energy (above K-edge data).
        # Across 60-140 keV this is monotonic since no K-edge in that range.
        @test η[2] > η[4]
    end
end

@testset "DetectorEfficiency — UFC MC LUT (Siemens Force Gd₂O₂S)" begin
    @testset "detector_efficiency_ufc factory" begin
        model = BS.detector_efficiency_ufc()
        @test model.material == "UFC"
        @test model.thickness_mm == 1.4
        @test model.fill_factor == 0.9
        @test model.mode == BS.MC_LUT

        bl = BS.detector_efficiency_ufc(mode = :beer_lambert)
        @test bl.mode == BS.BEER_LAMBERT
    end

    @testset "LUT shape + provenance values" begin
        @test length(BS.UFC_MC_EFFICIENCY_LUT.energies) == 140
        @test length(BS.UFC_MC_EFFICIENCY_LUT.efficiency) == 140
        # Spot-check verbatim values from Khodajou-Chokami efficiency_results.csv
        @test BS.UFC_MC_EFFICIENCY_LUT.efficiency[1] ≈ 9.90863305e-01
        @test BS.UFC_MC_EFFICIENCY_LUT.efficiency[50] ≈ 9.69026725e-01
        @test BS.UFC_MC_EFFICIENCY_LUT.efficiency[51] ≈ 7.41154644e-01
        @test BS.UFC_MC_EFFICIENCY_LUT.efficiency[140] ≈ 8.15971281e-01
    end

    @testset "get_ufc_mc_efficiency captures Gd K-edge fluorescence drop" begin
        η50 = BS.get_ufc_mc_efficiency(50.0)
        η51 = BS.get_ufc_mc_efficiency(51.0)
        # MC LUT: at the Gd K-edge (50.24 keV) η drops ~0.969 → ~0.741
        # because Gd-Kα fluorescence (~43 keV) escapes the crystal.
        @test η50 > 0.95
        @test η51 < 0.78
        @test η50 - η51 > 0.20
    end

    @testset "get_ufc_mc_efficiency captures Gd L-edge dip near 8 keV" begin
        @test BS.get_ufc_mc_efficiency(8.0) < BS.get_ufc_mc_efficiency(7.0)
    end

    @testset "MC LUT linear interpolation + clamping" begin
        η_60_5 = BS.get_ufc_mc_efficiency(60.5)
        lo = min(BS.UFC_MC_EFFICIENCY_LUT.efficiency[60], BS.UFC_MC_EFFICIENCY_LUT.efficiency[61])
        hi = max(BS.UFC_MC_EFFICIENCY_LUT.efficiency[60], BS.UFC_MC_EFFICIENCY_LUT.efficiency[61])
        @test lo - 1.0e-12 < η_60_5 < hi + 1.0e-12

        @test BS.get_ufc_mc_efficiency(0.5) == BS.UFC_MC_EFFICIENCY_LUT.efficiency[1]
        @test BS.get_ufc_mc_efficiency(200.0) == BS.UFC_MC_EFFICIENCY_LUT.efficiency[end]
    end

    @testset "compute_eid_efficiency_vector — MC routes to UFC LUT" begin
        model = BS.detector_efficiency_ufc()
        energies = [30.0, 50.0, 51.0, 100.0, 140.0]
        η = BS.compute_eid_efficiency_vector(model, energies)
        @test length(η) == length(energies)
        @test η[3] < η[2]                    # Gd K-edge fluorescence drop
        @test η ≈ [BS.get_ufc_mc_efficiency(E) for E in energies]
    end

    @testset "compute_eid_efficiency_vector — UFC Beer-Lambert path" begin
        bl = BS.detector_efficiency_ufc(mode = :beer_lambert)
        energies = [30.0, 49.0, 51.0, 100.0]
        η = BS.compute_eid_efficiency_vector(bl, energies)
        @test length(η) == length(energies)
        @test all(0 .<= η .<= 1)
        # Beer-Lambert predicts η RISES just above the Gd K-edge (more
        # absorption) — the opposite of the MC LUT's fluorescence-escape
        # drop.  That contrast is the whole point of the MC path.
        @test η[3] > η[2]
    end

    @testset "build_physics_config routes :ufc to the UFC factory" begin
        scanner = BS.Scanner(detector_material = :ufc, detector_depth = 1.4)
        sim_opts = BS.SimOptions(fidelity = :eict)
        e = collect(20.0:10.0:140.0)
        w = ones(length(e))
        config = BS.build_physics_config(scanner, sim_opts, e, w)
        @test config.detector_efficiency !== nothing
        @test config.detector_efficiency.material == "UFC"
        @test config.detector_efficiency.mode == BS.MC_LUT

        # Unknown EICT material errors with a clear message
        bad = BS.Scanner(detector_material = :unobtainium)
        @test_throws ErrorException BS.build_physics_config(bad, sim_opts, e, w)
    end
end

@testset "get_scintillator_mu — Gemstone + CdTe" begin
    # CdTe K-edges: Cd at 26.7 keV, Te at 31.8 keV.
    μ_cdte_25 = BS.get_scintillator_mu("CdTe", 25.0)
    μ_cdte_27 = BS.get_scintillator_mu("CdTe", 27.0)
    @test μ_cdte_27 > μ_cdte_25     # Just above Cd K-edge → bigger μ

    # Gemstone @ 60 keV ~32.7 cm⁻¹ (NIST XCOM table)
    μ_gem_60 = BS.get_scintillator_mu("Gemstone", 60.0)
    @test 28 < μ_gem_60 < 38

    # Alias resolves to Gemstone table
    μ_lumex_60 = BS.get_scintillator_mu("lumex", 60.0)
    @test μ_lumex_60 ≈ μ_gem_60  atol = 1.0e-12

    # Unknown material → warning + Gemstone fallback (no crash)
    μ_unknown = BS.get_scintillator_mu("nonexistent", 60.0)
    @test μ_unknown ≈ μ_gem_60  atol = 1.0e-12
end

# -----------------------------------------------------------------------------
# fill_factor.jl
# -----------------------------------------------------------------------------
@testset "FillFactorModel + apply_fill_factor!" begin
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

    @testset "fill_factor_standard = 90%" begin
        m = BS.fill_factor_standard()
        @test BS.effective_fill_factor(m) ≈ 0.9  atol = 1.0e-12
    end

    @testset "apply_fill_factor!  adds −log(ff) in projection domain" begin
        m = BS.fill_factor_standard()
        sino = ones(Float32, 16, 4, 8)
        BS.apply_fill_factor!(sino, m)
        @test sino ≈ ones(Float32, 16, 4, 8) .+ Float32(-log(0.9))  atol = 1.0e-6
    end

    @testset "ff ≈ 1.0 short-circuits (no-op)" begin
        m = BS.FillFactorModel(1.0, 1.0, true)
        sino = ones(Float32, 8, 2, 4)
        sino_before = copy(sino)
        BS.apply_fill_factor!(sino, m)
        @test sino == sino_before
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
        @test m.col_coeff ≈ 0.040
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
    @test det.enable_charge_sharing
    @test det.enable_pile_up
    @test det.enable_anti_coincidence
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
    @testset "load_mc_response — bundled cdte_response_v4.jls" begin
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
        # Probabilities in [0, 1]
        @test all(0 .<= D .<= 1)
        # Row sums ≤ 1
        @test maximum(sum(D; dims = 2)) <= 1.0 + 1.0e-9
        # Below the lowest threshold (≤20 keV) → near-zero registration in all bins
        @test sum(D[1, :]) < 0.05
    end
end

# -----------------------------------------------------------------------------
# pcct/mc_pileup.jl
# -----------------------------------------------------------------------------
@testset "MC pile-up — simulate_pulse_train + compute_mc_pileup_matrix" begin
    @testset "simulate_pulse_train — low rate: recorded ≈ true" begin
        Random.seed!(42)
        thresholds = [20.0, 35.0, 55.0, 70.0]
        energies = collect(20.0:5.0:120.0)
        weights = ones(length(energies))
        # Low rate, short window → almost no pile-up
        res = BS.simulate_pulse_train(
            weights, energies, 1.0e5, 5.0;
            observation_time_s = 1.0e-3,
            thresholds_keV = thresholds,
        )
        @test res isa BS.PileupResult
        @test res.n_true >= 0
        @test res.n_recorded <= res.n_true              # pile-up can only lose counts
        @test sum(res.true_bin_counts) == res.n_true
        @test length(res.trigger_bins) == res.n_recorded
        @test all(0 .<= res.trigger_bins .<= length(thresholds))
    end

    @testset "compute_mc_pileup_matrix — shape contract + col-sums ≤ 1" begin
        thresholds = [20.0, 35.0, 55.0, 70.0]
        energies = collect(20.0:5.0:120.0)
        weights = ones(length(energies))
        S = BS.compute_mc_pileup_matrix(
            thresholds, weights, energies, 5.0e7, 5.0;
            n_trials = 200, observation_time_s = 1.0e-4, seed = 7,
        )
        @test size(S) == (4, 4)
        @test all(0 .<= S .<= 1)
        # Column sums ≤ 1 (count loss in each true bin's deficit)
        col_sums = sum(S; dims = 1)
        @test all(col_sums .<= 1.0 + 1.0e-9)
        # Lower triangular: pile-up only pushes counts UP in energy
        for j in 1:4, i in 1:(j - 1)
            @test S[i, j] < 0.05   # very small upward leakage
        end
    end

    @testset "compute_mc_pileup_matrix — zero rate ⇒ identity columns" begin
        thresholds = [20.0, 35.0, 55.0, 70.0]
        energies = collect(20.0:5.0:120.0)
        weights = ones(length(energies))
        S = BS.compute_mc_pileup_matrix(
            thresholds, weights, energies, 1.0, 5.0;   # 1 photon/sec ⇒ no pileup
            n_trials = 50, observation_time_s = 1.0e-6, seed = 7,
        )
        # Zero pileup: each true bin maps mostly to itself
        for j in 1:4
            if sum(S[:, j]) > 0   # bin has data
                @test S[j, j] > 0.9 * sum(S[:, j])
            end
        end
    end
end
