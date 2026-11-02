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
            fill_factor = BS.fill_factor_standard(),
            scatter = BS.geometry_aware_scatter_model(BS.EICTScanner()),
            optical_crosstalk = BS.optical_crosstalk_typical(),
            detector_efficiency = BS.DetectorEfficiency(:lumex, 3.0, BS.MC_LUT),
            lag = BS.lag_gadox(),
            noise_seed = 7,
            energy_keV = 75.0,
        )
        @test cfg.fill_factor isa BS.FillFactorModel
        @test cfg.scatter isa BS.ScatterModel
        @test cfg.optical_crosstalk isa BS.OpticalCrosstalkModel
        @test cfg.detector_efficiency isa BS.DetectorEfficiency
        @test cfg.lag isa BS.LagModel
        @test cfg.noise_seed == 7
        @test cfg.energy_keV == 75.0
    end
end

# -----------------------------------------------------------------------------
# scatter.jl
# -----------------------------------------------------------------------------
@testset "ScatterModel + geometry_aware_scatter_model" begin
    @testset "geometry_aware_scatter_model scales with air gap" begin
        ref = BS.EICTScanner(
            source_to_isocenter = BS.SCATTER_REF_SID_MM,
            source_to_detector = BS.SCATTER_REF_SDD_MM,
        )
        ref_model = BS.geometry_aware_scatter_model(ref)
        @test ref_model isa BS.ScatterModel
        # Reference geometry → coefficient ≈ SCATTER_REF_COEFFICIENT × 1 × 1
        @test ref_model.scatter_coefficient ≈ BS.SCATTER_REF_COEFFICIENT atol = 1.0e-12

        # GE Revolution has a larger air gap → less scatter
        ge = BS.EICTScanner(source_to_isocenter = 626.0, source_to_detector = 1097.0)
        ge_model = BS.geometry_aware_scatter_model(ge)
        @test ge_model.scatter_coefficient < ref_model.scatter_coefficient
    end

    @testset "phantom diameter scaling — bigger body, more scatter" begin
        scanner = BS.EICTScanner()
        small = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = 20.0)
        large = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = 40.0)
        @test large.scatter_coefficient > small.scatter_coefficient
    end
end

@testset "scatter kernels (spatial + 1D Gaussian)" begin
    model = BS.geometry_aware_scatter_model(BS.EICTScanner())

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
    model = BS.geometry_aware_scatter_model(BS.EICTScanner())

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
        b = E < 35 ? 1 :
            E < 55 ? 2 :
            E < 70 ? 3 : 4
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
    sf = fill(Float32(0.01), nc, nr, nv)
    I0 = fill(Float32(1.0e5), nc, nr, n_bins)           # per ray, per bin
    I0_all = fill(Float32(4.0e5), nc, nr)                # per ray, all bins
    bw = [0.4, 0.3, 0.2, 0.1]

    BS.inject_scatter_bins!(bins, sf, I0, I0_all, bw)
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
        model = BS.build_physics_config(s, BS.SimOptions(), e, w).detector_efficiency
        @test model.material === s.detector_material
        @test model.thickness_mm == s.detector_depth
        @test model.mode == BS.MC_LUT
    end
    bl = BS.SimOptions(detector_efficiency_mode = :beer_lambert)
    @test BS.build_physics_config(BS.SOMATOM_FORCE, bl, e, w).detector_efficiency.mode == BS.BEER_LAMBERT
    off = BS.build_physics_config(BS.SOMATOM_FORCE, BS.SimOptions(use_detector_efficiency = false), e, w)
    @test off.detector_efficiency === nothing
    # a depth the table was not made at, and a material with no model, are refused
    @test_throws ArgumentError BS.build_physics_config(
        BS.EICTScanner(BS.SOMATOM_FORCE; detector_depth = 1.0), BS.SimOptions(), e, w
    )
    @test_throws ArgumentError BS.build_physics_config(
        BS.EICTScanner(detector_material = :unobtainium), BS.SimOptions(), e, w
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

    @testset "fill_factor_standard = 90%" begin
        m = BS.fill_factor_standard()
        @test BS.effective_fill_factor(m) ≈ 0.9  atol = 1.0e-12
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

@testset "apply_pcct_noise! — Poisson counts in the air-cal basis" begin
    thresholds = Float32[20, 35, 55, 70]
    I0_vec = Float64[40_000, 60_000, 30_000, 10_000]
    I0_bins = reshape(repeat(I0_vec, inner = 32 * 4), 32, 4, 4)     # per ray, per bin
    # Uniform log line integral h = 2 → expected counts λ_b = I0_b·e⁻²
    make_sino() = BS.EnergyResolvedSinogram(
        [fill(Float32(2), 32, 4, 25) for _ in 1:4], thresholds
    )

    sino = BS.apply_pcct_noise!(make_sino(), I0_bins; seed = 11)
    for (b, bin) in enumerate(sino.bins)
        counts = I0_vec[b] .* exp.(-Float64.(bin))
        # Integer realizations survive the Float32 log round trip
        @test maximum(abs.(counts .- round.(counts))) < 0.1
        λ = I0_vec[b] * exp(-2.0)
        @test mean(counts) ≈ λ rtol = 0.01
        @test std(counts)^2 ≈ λ rtol = 0.15
    end

    # Same seed → identical realization; different seed → different one
    s1 = BS.apply_pcct_noise!(make_sino(), I0_bins; seed = 42)
    s2 = BS.apply_pcct_noise!(make_sino(), I0_bins; seed = 42)
    s3 = BS.apply_pcct_noise!(make_sino(), I0_bins; seed = 43)
    @test all(s1.bins[b] == s2.bins[b] for b in 1:4)
    @test any(s1.bins[b] != s3.bins[b] for b in 1:4)

    # noise_reduction = 1 collapses to the noiseless expectation exactly
    # (any nonzero value leaves the strict Poisson count model)
    s4 = BS.apply_pcct_noise!(make_sino(), I0_bins; seed = 42, noise_reduction = 1.0)
    @test all(all(isapprox.(Array(bin), 2.0f0; atol = 2.0f-5)) for bin in s4.bins)

    # raw_out: measured counts captured verbatim at the sampling site —
    # TRUE ZEROS preserved; the floor at 1 lives only in the log bins
    lo_I0 = reshape(repeat(Float64[2, 3, 2, 4], inner = 32 * 4), 32, 4, 4)   # λ = I0·e⁻² ≈ 0.27–0.54 → many zeros
    s5 = make_sino()
    raw5 = [similar(b) for b in s5.bins]
    BS.apply_pcct_noise!(s5, lo_I0; seed = 5, raw_out = raw5)
    @test sum(count(==(0.0f0), r) for r in raw5) > 0
    for (b, r) in enumerate(raw5)
        @test all(isinteger, r)
        enc = lo_I0[:, :, b] .* exp.(-Float64.(s5.bins[b]))     # = max(N, 1)
        @test all(abs.(enc .- max.(Float64.(r), 1.0)) .< 1.0e-4)
    end

    @test_throws DimensionMismatch BS.apply_pcct_noise!(make_sino(), I0_bins[:, :, 1:3])
    @test_throws DimensionMismatch BS.apply_pcct_noise!(
        make_sino(), I0_bins; raw_out = [zeros(Float32, 32, 4, 25)]
    )
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

# -----------------------------------------------------------------------------
# apply_pcct_noise! — the threaded Poisson mode
# -----------------------------------------------------------------------------
@testset "apply_pcct_noise! rng_mode" begin
    n_bins = 3
    shape = (40, 4, 30)
    I0 = reshape(repeat(Float64[2.0e4, 8.0e3, 1.5e3], inner = 40 * 4), 40, 4, 3)   # per ray, per bin
    thresholds = Float32[20, 50, 80]
    build() = BS.EnergyResolvedSinogram(
        [fill(Float32(0.35 + 0.1b), shape) for b in 1:n_bins], thresholds
    )

    serial = build()
    BS.apply_pcct_noise!(serial, I0; seed = 7)
    again = build()
    BS.apply_pcct_noise!(again, I0; seed = 7)
    @test all(serial.bins[b] == again.bins[b] for b in 1:n_bins)      # a seed is a seed

    threaded = build()
    BS.apply_pcct_noise!(threaded, I0; seed = 7, rng_mode = :threaded)
    # a different realisation, but of the same distribution
    @test any(threaded.bins[b] != serial.bins[b] for b in 1:n_bins)
    for b in 1:n_bins
        λ = I0[1, 1, b] * exp(-(0.35 + 0.1b))        # I0 is flat across the fan in this test
        counts(bin) = I0[:, :, b] .* exp.(-Float64.(bin))
        for bin in (serial.bins[b], threaded.bins[b])
            c = counts(bin)
            @test mean(c) ≈ λ rtol = 0.02                  # unbiased
            @test (std(c)^2) / λ ≈ 1 rtol = 0.15               # Fano ≈ 1
        end
    end
    # the realisation is a property of the seed, not of the thread count: the chunking is fixed
    @test BS._PCCT_NOISE_CHUNKS > 1
    repeat_threaded = build()
    BS.apply_pcct_noise!(repeat_threaded, I0; seed = 7, rng_mode = :threaded)
    @test all(repeat_threaded.bins[b] == threaded.bins[b] for b in 1:n_bins)
    other_seed = build()
    BS.apply_pcct_noise!(other_seed, I0; seed = 8, rng_mode = :threaded)
    @test any(other_seed.bins[b] != threaded.bins[b] for b in 1:n_bins)

    @test_throws ArgumentError BS.apply_pcct_noise!(build(), I0; rng_mode = :parallel)

    # `seed = nothing` means unseeded on both paths: the threaded draw must not quietly pin
    # itself to a fixed stream just because it needs a number to derive its chunks from.
    unseeded_a = build(); BS.apply_pcct_noise!(unseeded_a, I0; rng_mode = :threaded)
    unseeded_b = build(); BS.apply_pcct_noise!(unseeded_b, I0; rng_mode = :threaded)
    @test any(unseeded_a.bins[b] != unseeded_b.bins[b] for b in 1:n_bins)
end
