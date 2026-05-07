# Tests for src/api/ — top-level orchestration: options, workspace, driver.
#
# Coverage policy: every public function exported (or otherwise intended for
# user consumption) from src/api/{options,workspace,driver}.jl is exercised
# here OR explicitly listed below as deliberately-untested-because-stale.
#
# Stale / removed (do not test, candidate for deletion from src/):
#   - (none yet — fill in as we audit)

# -----------------------------------------------------------------------------
# GPU backend detection — mirrors nb04's preamble.  The PCCT spectral forward
# project is GPU-only in practice: on CPU, the K=16 tiled path JIT-compiles
# for 14+ min on even a 32³ phantom.  When no Metal / CUDA / AMDGPU backend
# is available we skip the integration tests with a clear warning rather
# than running the CPU fallback.
# -----------------------------------------------------------------------------
const GPU_BACKEND = let
    candidates = [
        (:Metal,  "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
        (:CUDA,   "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
        (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
    ]
    detected = (name = "CPU", to_gpu = identity)
    for (pkg, uuid, ctor) in candidates
        pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
        Base.locate_package(pkg_id) === nothing && continue
        try
            m = Base.require(pkg_id)
            if Base.invokelatest(getfield(m, :functional))
                detected = (name = string(pkg), to_gpu = getfield(m, ctor))
                break
            end
        catch
        end
    end
    detected
end

const HAS_GPU = GPU_BACKEND.name != "CPU"
to_gpu(x) = GPU_BACKEND.to_gpu(x)
@info "[api.jl] GPU backend = $(GPU_BACKEND.name)"

# Tiny progress logger — lets us see WHERE time is going during the slow first
# compile pass.  Pluto/Pkg.test buffers stdout until @testset exits, so without
# this we stare at a black screen for a minute.
_ts(label) = (println(stderr, "[api.jl t=$(round(time()-_T0;digits=1))s] ", label); flush(stderr))
const _T0 = time()

# Helper: build a minimally-valid (Scanner, CTGeometry, CTProtocol) triple
# with an in-test name so failures point straight at the offending case.
function _toy_scan(;
        SAD_mm = 540.0,
        SDD_mm = 950.0,
        col_mm = 1.0,
        row_mm = 1.0,
        n_cols = 16,
        n_rows = 4,
        n_angles = 8,
        mA = 200.0,
        rotation_time = 1.0,
        views = 1000,
    )
    scanner = BS.Scanner(
        source_to_isocenter = SAD_mm,
        source_to_detector  = SDD_mm,
        detector_cols       = n_cols,
        detector_rows       = n_rows,
        detector_col_size   = col_mm,
        detector_row_size   = row_mm,
    )
    geom     = BS.CTGeometry(scanner; n_angles = n_angles)
    protocol = BS.CTProtocol(; mA = mA, kVp = 120.0, views = views,
                               rotation_time = rotation_time)
    return scanner, geom, protocol
end

# Closed-form reference, mirrors `compute_detector_I0` line-for-line.
_expected_I0(geom, protocol, flux_sum) = begin
    M  = geom.SDD / geom.SAD
    px = (geom.pixel_size     * 10.0) * M    # mm at detector
    py = (geom.pixel_row_size * 10.0) * M
    flux_sum * protocol.mA * (protocol.rotation_time / protocol.views) * px * py
end

# -----------------------------------------------------------------------------
# compute_detector_I0
#
# Reference: XCIST/CatSim — `gecatsim/pyfiles/Spectrum.py` (mA × viewTime
# scaling) + `gecatsim/pyfiles/Detection_Flux.py` (× detector active area
# × distance factor).  Test invariants follow CatSim's flux semantics:
# linear in mA, linear in viewTime (= 1/views at fixed rotation), quadratic
# in magnification M = SDD/SAD.
# -----------------------------------------------------------------------------
_ts("entering compute_detector_I0 testset")
@testset "compute_detector_I0" begin
    flux_sum = 1.0e6  # photons / mAs / mm² at SDD (post-filter, summed over E)

    @testset "closed-form value" begin
        _, geom, protocol = _toy_scan()
        got      = BS.compute_detector_I0(geom, protocol, flux_sum)
        expected = _expected_I0(geom, protocol, flux_sum)
        @test got ≈ expected
        @test got > 0
        @test isfinite(got)
    end

    @testset "linearity in mA" begin
        _, g, p1 = _toy_scan(mA = 100.0)
        _, _, p2 = _toy_scan(mA = 200.0)
        I1 = BS.compute_detector_I0(g, p1, flux_sum)
        I2 = BS.compute_detector_I0(g, p2, flux_sum)
        @test I2 ≈ 2 * I1
    end

    @testset "linearity in spectrum flux" begin
        _, g, p = _toy_scan()
        I1 = BS.compute_detector_I0(g, p, flux_sum)
        I2 = BS.compute_detector_I0(g, p, 2 * flux_sum)
        @test I2 ≈ 2 * I1
    end

    @testset "inverse linearity in views (time_per_view)" begin
        # Halving views at fixed rotation_time doubles time_per_view → doubles I0.
        _, g, p1 = _toy_scan(views = 1000)
        _, _, p2 = _toy_scan(views = 500)
        I1 = BS.compute_detector_I0(g, p1, flux_sum)
        I2 = BS.compute_detector_I0(g, p2, flux_sum)
        @test I2 ≈ 2 * I1
    end

    @testset "linearity in rotation_time" begin
        _, g, p1 = _toy_scan(rotation_time = 1.0)
        _, _, p2 = _toy_scan(rotation_time = 2.0)
        I1 = BS.compute_detector_I0(g, p1, flux_sum)
        I2 = BS.compute_detector_I0(g, p2, flux_sum)
        @test I2 ≈ 2 * I1
    end

    @testset "magnification M² scaling" begin
        # Doubling SDD at fixed SAD doubles M → pixel area at detector × 4.
        _, g1, p = _toy_scan(SAD_mm = 500.0, SDD_mm = 1000.0)  # M = 2
        _, g2, _ = _toy_scan(SAD_mm = 500.0, SDD_mm = 2000.0)  # M = 4
        I1 = BS.compute_detector_I0(g1, p, flux_sum)
        I2 = BS.compute_detector_I0(g2, p, flux_sum)
        @test I2 / I1 ≈ 4.0
    end

    @testset "pixel-area scaling (col × row mm)" begin
        # Each detector edge enters linearly in mm, so a 2×3 pixel-size scale
        # multiplies I0 by 6 (with all other params held fixed).
        _, g1, p = _toy_scan(col_mm = 1.0, row_mm = 1.0)
        _, g2, _ = _toy_scan(col_mm = 2.0, row_mm = 3.0)
        I1 = BS.compute_detector_I0(g1, p, flux_sum)
        I2 = BS.compute_detector_I0(g2, p, flux_sum)
        @test I2 / I1 ≈ 6.0
    end
end

# -----------------------------------------------------------------------------
# simulate!(::PCCTWorkspace, phantom, protocol, sim_opts)
#
# Contract:
# - Always uses MC-LUT detector response (`compute_mc_drm`) — no analytical
#   fallback — `pcct_forward_project` now hardcodes the MC DRM path.
# - Always uses MC-LUT spectral-migration matrix S for pulse pileup
#   (`compute_mc_pileup_matrix`) — no analytical Taguchi count factor.
# - Pileup application is gated on `sim_opts.use_pcct_pileup` (default true).
# - Returns `(pcct_sino, I0_bins)` — bin-combine + scatter correction live
#   at the notebook level.
# -----------------------------------------------------------------------------
function _toy_pcct_setup(; use_pcct_pileup = true, kwargs...)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector  = 1080.0,
        detector_rows       = 8,
        detector_cols       = 64,
        detector_row_size   = 1.0,
        detector_col_size   = 1.0,
        detector_type       = :photon_counting,
        detector_material   = :CdTe,
        detector_depth      = 1.6,
        n_energy_bins       = 4,
        energy_thresholds   = [20.0, 35.0, 55.0, 70.0],
        dead_time_ns        = 25.0,
    )
    # mAs-per-view kept near nb04 (~0.07 mAs/view) so the MC pile-up trial
    # samples a realistic number of photon arrivals (~1e7 per trial), not 1e11.
    protocol = BS.CTProtocol(mA = 2.5, kVp = 120.0, views = 16, rotation_time = 0.5)
    sim_opts = BS.SimOptions(;
        fidelity = :pcct,
        use_noise = false, use_scatter = false, use_lag = false,
        use_focal_spot = false, use_optical_crosstalk = false,
        use_pcct_pileup = use_pcct_pileup,
        kwargs...
    )
    recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
    phantom_cpu = BS.create_gammex_472(n_voxels = 32, fov_cm = 20.0, z_cm = 2.0)
    # Move the mask to GPU — nb04-style — so the workspace allocates all
    # `similar(ref_mask, …)` buffers on the same backend and the K=16 tiled
    # spectral forward project actually runs on the GPU.
    phantom = BS.Phantom(
        to_gpu(phantom_cpu.mask),
        phantom_cpu.materials,
        phantom_cpu.voxel_size,
        phantom_cpu.origin,
        phantom_cpu.extent,
    )
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    return (; scanner, protocol, sim_opts, recon_opts, phantom, ws)
end

_ts("entering simulate!(PCCTWorkspace) — return contract testset")
@testset "simulate!(PCCTWorkspace) — return contract" begin
    if !HAS_GPU
        @warn "Skipping PCCT integration test: no GPU backend (Metal/CUDA/AMDGPU). \
               PCCT spectral forward project requires GPU; CPU JIT alone takes 14+ min on a 32³ phantom."
    else
        _ts("  building toy_pcct_setup")
        s = _toy_pcct_setup()
        _ts("  running simulate!")
        res = BS.simulate!(s.ws, s.phantom, s.protocol, s.sim_opts)
        _ts("  simulate! returned")
        # Trimmed return tuple — pcct_sino + I0_bins + pileup_S (combine/scatter
        # decoupled).  pileup_S is `nothing` when use_pcct_pileup=false.
        @test propertynames(res) == (:pcct_sino, :I0_bins, :pileup_S)
        @test length(res.pcct_sino.bins) == 4
        @test length(res.I0_bins) == 4
        @test all(>(0), res.I0_bins)
        @test res.pileup_S isa Matrix{Float64}
        @test size(res.pileup_S) == (4, 4)
        for bin in res.pcct_sino.bins
            @test all(isfinite, Array(bin))
        end
    end
end

_ts("entering simulate!(PCCTWorkspace) — MC-LUT pileup wiring testset")
@testset "simulate!(PCCTWorkspace) — MC-LUT pileup wiring" begin
    if !HAS_GPU
        @warn "Skipping PCCT integration test: no GPU backend."
    else
        _ts("  workspace ON")
        on  = _toy_pcct_setup(use_pcct_pileup = true)
        _ts("  workspace OFF")
        off = _toy_pcct_setup(use_pcct_pileup = false)
        _ts("  workspaces built")

        # Workspace state reflects the toggle.
        @test on.ws.use_pcct_pileup === true
        @test off.ws.use_pcct_pileup === false
        @test on.ws.pileup_S isa Matrix{Float64}
        @test size(on.ws.pileup_S) == (4, 4)
        @test off.ws.pileup_S === nothing

        _ts("  simulate! ON")
        res_on  = BS.simulate!(on.ws,  on.phantom,  on.protocol,  on.sim_opts)
        _ts("  simulate! OFF")
        res_off = BS.simulate!(off.ws, off.phantom, off.protocol, off.sim_opts)
        _ts("  simulate! both done")

        # Both finite.
        for bins in (res_on.pcct_sino.bins, res_off.pcct_sino.bins)
            for bin in bins
                @test all(isfinite, Array(bin))
            end
        end

        # Pileup ON should produce DIFFERENT log line integrals than OFF.
        diff_any = false
        for b in 1:4
            on_arr  = Array(res_on.pcct_sino.bins[b])
            off_arr = Array(res_off.pcct_sino.bins[b])
            if maximum(abs.(on_arr .- off_arr)) > 1e-6
                diff_any = true
                break
            end
        end
        @test diff_any  # pileup must measurably perturb at least one bin
    end
end

# -----------------------------------------------------------------------------
# compute_mc_pileup_matrix — physical-shape contract
# -----------------------------------------------------------------------------
# Pure-CPU test: build S directly from `compute_mc_pileup_matrix` at high
# count-rate / dead-time so pile-up is significant, then assert the matrix
# encodes the right physics:
#
#   • Columns are NOT all identical (the original PR-#10 algorithm broke this).
#   • Column sums ≤ 1 — column-j-deficit = count-loss for true bin j.
#   • Counts only migrate UP (low-E photons sum to higher-E recorded events) —
#     S[i, j] ≈ 0 for i < j (high-E bins do not seed low-E bins, since
#     pile-up only adds energy).
#   • At high aτ, S[i, j] for i > j must be visibly non-zero.
# -----------------------------------------------------------------------------
@testset "compute_mc_pileup_matrix — physical-shape contract" begin
    energies   = collect(20.0:1.0:140.0)
    weights    = ones(length(energies)) ./ length(energies)   # flat spectrum
    thresholds = [20.0, 35.0, 55.0, 70.0]
    n_bins     = length(thresholds)

    # aτ ≈ 0.5 → significant pile-up, well within the regime our PCCT runs at.
    count_rate   = 1e8        # 100 Mcps per pixel
    dead_time_ns = 5.0
    S = BS.compute_mc_pileup_matrix(thresholds, weights, energies,
                                    count_rate, dead_time_ns;
                                    n_trials = 2000, seed = 1)

    @test size(S) == (n_bins, n_bins)
    @test all(isfinite, S)
    @test all(>=(0), S)

    # Non-degeneracy: at least one pair of columns must differ materially.
    max_col_diff = 0.0
    for j in 2:n_bins
        max_col_diff = max(max_col_diff, maximum(abs.(S[:, j] .- S[:, 1])))
    end
    @test max_col_diff > 0.05   # original PR's broken matrix gave 0 here

    # Column sums ≤ 1 (deficit = count loss).
    for j in 1:n_bins
        s = sum(@view S[:, j])
        @test 0.0 < s ≤ 1.0 + 1e-6
    end

    # Pile-up only adds energy → migration is upward only.
    # S[i, j] for i < j (lower recorded bin from higher true bin) must be ~0.
    for j in 2:n_bins, i in 1:(j-1)
        @test S[i, j] < 0.05
    end

    # At meaningful aτ, low-energy bins MUST visibly leak into higher bins.
    @test S[2, 1] > 0.01    # bin-1 photons piling up to bin 2 or higher
end

# -----------------------------------------------------------------------------
# Pile-up I0 renormalization (math-level, no phantom)
# -----------------------------------------------------------------------------
# Locks down `simulate!`'s pile-up math contract on a truth-basis air ray:
# bins are `-log(recorded / I0_truth)`, so an air ray (counts == I0_truth)
# gives `bins[i] = -log(I0_recorded[i] / I0_truth[i])` — a small per-bin
# offset, NOT zero.  That offset is the price of pile-up redistribution and
# is what makes `I0_b · exp(-bin)` round-trip to the recorded count for
# downstream count-domain math (scatter correction in nb04, etc.).
# -----------------------------------------------------------------------------
@testset "Pile-up math: truth-basis bins + count round-trip" begin
    n_bins = 4
    I0_truth = [1.3e12, 1.0e12, 4.2e11, 2.1e11]
    energies   = collect(20.0:1.0:140.0)
    weights    = ones(length(energies)) ./ length(energies)
    thresholds = [20.0, 35.0, 55.0, 70.0]
    S = BS.compute_mc_pileup_matrix(thresholds, weights, energies,
                                    1e8, 5.0; n_trials = 2000, seed = 1)

    # Air ray: pre-pileup counts == truth I0.
    counts_true     = copy(I0_truth)
    counts_recorded = [sum(S[i, j] * counts_true[j] for j in 1:n_bins) for i in 1:n_bins]

    # `simulate!` semantics: bins[i] = -log(recorded / I0_truth[i])
    bins_air = [-log(counts_recorded[i] / I0_truth[i]) for i in 1:n_bins]
    for v in bins_air
        @test isfinite(v)
    end

    # Round-trip identity: I0_truth[i] · exp(-bins[i]) == counts_recorded[i]
    for i in 1:n_bins
        @test I0_truth[i] * exp(-bins_air[i]) ≈ counts_recorded[i] rtol = 1e-12
    end

    # The air-ray offset equals log(I0_truth / I0_recorded) — physically the
    # signature of pile-up "dimming" each bin's air baseline.
    I0_recorded = [sum(S[i, j] * I0_truth[j] for j in 1:n_bins) for i in 1:n_bins]
    for i in 1:n_bins
        @test bins_air[i] ≈ log(I0_truth[i] / I0_recorded[i]) rtol = 1e-12
    end
end

# -----------------------------------------------------------------------------
# simulate!(::EICTWorkspace, phantom, protocol, sim_opts)
#
# Contract:
# - Signature is `(ws, phantom, protocol, sim_opts)` — scanner is consumed at
#   workspace creation time (η_eff, σ_e_photon baked in there).
# - Returns `nothing`; mutates `ws.sinogram` in place.
# - Notebooks read `ws.sinogram` directly (and `ws.geom`) — no separate
#   `sino_noisy_out` field.
# -----------------------------------------------------------------------------
function _toy_eict_setup(; use_noise = false, use_scatter = false, kwargs...)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector  = 1080.0,
        detector_rows       = 8,
        detector_cols       = 64,
        detector_row_size   = 1.0,
        detector_col_size   = 1.0,
        detector_material   = :lumex,
        detector_depth      = 3.0,
        electronic_noise    = 5.0,
        detection_gain      = 10.0,
    )
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    sim_opts = BS.SimOptions(;
        fidelity = :eict,
        use_noise = use_noise, use_scatter = use_scatter,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false,
        kwargs...
    )
    recon_opts  = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
    phantom_cpu = BS.create_gammex_472(n_voxels = 32, fov_cm = 20.0, z_cm = 2.0)
    phantom = BS.Phantom(
        to_gpu(phantom_cpu.mask),
        phantom_cpu.materials,
        phantom_cpu.voxel_size,
        phantom_cpu.origin,
        phantom_cpu.extent,
    )
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    return (; scanner, protocol, sim_opts, recon_opts, phantom, ws)
end

_ts("entering simulate!(EICTWorkspace) — return contract testset")
@testset "simulate!(EICTWorkspace) — return contract" begin
    if !HAS_GPU
        @warn "Skipping EICT integration test: no GPU backend (Metal/CUDA/AMDGPU)."
    else
        _ts("  building toy_eict_setup")
        s = _toy_eict_setup()
        # Workspace baked-in noise constants (Pass 2 contract).
        @test s.ws.η_eff isa Float32
        @test 0 < s.ws.η_eff ≤ 1
        @test s.ws.σ_e_photon isa Float32
        @test s.ws.σ_e_photon ≥ 0
        # Sino shape matches (n_cols, n_rows, n_views).
        @test size(s.ws.sinogram) == (64, 8, 16)
        _ts("  running simulate!")
        ret = BS.simulate!(s.ws, s.phantom, s.protocol, s.sim_opts)
        _ts("  simulate! returned")
        # Contract: returns nothing; ws.sinogram populated and finite.
        @test ret === nothing
        sino = Array(s.ws.sinogram)
        @test all(isfinite, sino)
        # Phantom is solid water+inserts → at least some non-trivial line integrals.
        @test maximum(sino) > 0
    end
end

_ts("entering simulate!(EICTWorkspace) — noise on/off seed reproducibility testset")
@testset "simulate!(EICTWorkspace) — noise on/off + seed reproducibility" begin
    if !HAS_GPU
        @warn "Skipping EICT noise test: no GPU backend."
    else
        _ts("  noise OFF")
        off = _toy_eict_setup(use_noise = false)
        BS.simulate!(off.ws, off.phantom, off.protocol, off.sim_opts)
        sino_off = Array(off.ws.sinogram)

        _ts("  noise ON (seed=42)")
        on1 = _toy_eict_setup(use_noise = true, seed = 42)
        BS.simulate!(on1.ws, on1.phantom, on1.protocol, on1.sim_opts)
        sino_on1 = Array(on1.ws.sinogram)

        _ts("  noise ON (seed=42, second run)")
        on2 = _toy_eict_setup(use_noise = true, seed = 42)
        BS.simulate!(on2.ws, on2.phantom, on2.protocol, on2.sim_opts)
        sino_on2 = Array(on2.ws.sinogram)

        # Noise ON must perturb the noise-free sinogram somewhere.
        @test maximum(abs.(sino_on1 .- sino_off)) > 1e-6
        # Same seed → bit-identical re-run under our `randn!` + copyto! pattern
        # (Phase-1 CPU RNG → GPU staging means this is deterministic).
        @test sino_on1 == sino_on2
    end
end

_ts("entering add_system_noise_floor! testset")
@testset "add_system_noise_floor!" begin
    # Pure-CPU function — no GPU gating needed.  Empirical post-recon noise
    # floor; concept covered in Kalender / Hsieh CT physics texts.
    @testset "σ ≤ 0 is a no-op (returns identity)" begin
        for σ in (0.0, -1e-9, -5.0)
            vol = randn(Float32, 8, 8, 4)
            ref = copy(vol)
            ret = BS.add_system_noise_floor!(vol, σ)
            @test ret === vol      # in-place identity contract
            @test vol == ref       # bit-identical
        end
    end

    @testset "σ > 0 produces empirical std ≈ σ" begin
        # Large enough volume that sample-std is tight to population σ.
        N = 64^3
        vol = zeros(Float32, 64, 64, 64)
        BS.add_system_noise_floor!(vol, 28.0; seed = 1234)
        # 64³ ≈ 262k samples → 99% CI on std is ~σ × (1 ± 0.005).  Use 5% rtol.
        @test std(vol) ≈ 28.0f0 rtol = 5e-2
        @test mean(vol) ≈ 0.0f0 atol = 1.0     # mean ≈ 0
        @test all(isfinite, vol)
    end

    @testset "seeded reproducibility" begin
        vol1 = zeros(Float32, 16, 16, 4)
        vol2 = zeros(Float32, 16, 16, 4)
        BS.add_system_noise_floor!(vol1, 28.0; seed = 42)
        BS.add_system_noise_floor!(vol2, 28.0; seed = 42)
        @test vol1 == vol2
        # Different seed → different realization
        vol3 = zeros(Float32, 16, 16, 4)
        BS.add_system_noise_floor!(vol3, 28.0; seed = 43)
        @test vol1 != vol3
    end

    @testset "additive in-place: floor stacks on existing values" begin
        # vol_after - vol_before ~ N(0, σ²); pre-existing content is preserved
        # in mean.
        base = fill(40.0f0, 32, 32, 4)
        vol  = copy(base)
        BS.add_system_noise_floor!(vol, 10.0; seed = 7)
        diff = vol .- base
        @test std(diff)  ≈ 10.0f0 rtol = 1e-1   # smaller volume → looser tol
        @test mean(diff) ≈ 0.0f0  atol = 1.0
        # Mean of vol stays near base value (within sample noise).
        @test mean(vol) ≈ 40.0f0 atol = 1.0
    end

    @testset "private-RNG isolation: doesn't disturb default_rng" begin
        # The seeded code path (seed::Int) builds its OWN MersenneTwister, so
        # calling it must not advance Random.default_rng().  Compare two runs
        # at the same global-RNG state: with and without the seeded floor
        # call between them.
        Random.seed!(123)
        _ = randn(Float32, 5)                     # advance to a known state
        target = randn(Float32, 5)                # next 5 draws

        Random.seed!(123)
        _ = randn(Float32, 5)                     # same prefix
        BS.add_system_noise_floor!(zeros(Float32, 32, 32, 4), 28.0; seed = 999)
        after  = randn(Float32, 5)                # next 5 draws — must match target

        @test target == after
    end
end

_ts("entering simulate!(EICTWorkspace) — scatter on/off testset")
@testset "simulate!(EICTWorkspace) — scatter on/off" begin
    if !HAS_GPU
        @warn "Skipping EICT scatter test: no GPU backend."
    else
        _ts("  scatter OFF")
        off = _toy_eict_setup(use_noise = false, use_scatter = false)
        BS.simulate!(off.ws, off.phantom, off.protocol, off.sim_opts)
        sino_off = Array(off.ws.sinogram)

        _ts("  scatter ON")
        on = _toy_eict_setup(use_noise = false, use_scatter = true)
        BS.simulate!(on.ws, on.phantom, on.protocol, on.sim_opts)
        sino_on = Array(on.ws.sinogram)

        # Scatter must measurably perturb the line integrals (DAS subtraction
        # of estimated scatter happens before the −log step, so the sinogram
        # does not equal the scatter-OFF sinogram).
        @test maximum(abs.(sino_on .- sino_off)) > 1e-6
        @test all(isfinite, sino_on)
    end
end
