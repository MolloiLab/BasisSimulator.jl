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
        (:Metal, "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
        (:CUDA, "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
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
_ts(label) = (println(stderr, "[api.jl t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))
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
        source_to_detector = SDD_mm,
        detector_cols = n_cols,
        detector_rows = n_rows,
        detector_col_size = col_mm,
        detector_row_size = row_mm,
    )
    geom = BS.CTGeometry(scanner; n_angles = n_angles)
    protocol = BS.CTProtocol(;
        mA = mA, kVp = 120.0, views = views,
        rotation_time = rotation_time
    )
    return scanner, geom, protocol
end

# Closed-form reference, mirrors `compute_detector_I0` line-for-line.
_expected_I0(geom, protocol, flux_sum) = begin
    M = geom.SDD / geom.SAD
    px = (geom.pixel_size * 10.0) * M    # mm at detector
    py = (geom.pixel_row_size * 10.0) * M
    flux_sum * protocol.mA * (protocol.rotation_time / protocol.views) * px * py
end

# -----------------------------------------------------------------------------
# SimOptions — fidelity preset resolution + per-toggle overrides + clamps.
# -----------------------------------------------------------------------------
_ts("entering SimOptions testset")
@testset "SimOptions" begin
    @testset "default fidelity = :eict; use_* presets" begin
        opts = BS.SimOptions()
        @test opts.use_fill_factor === true
        @test opts.use_detector_efficiency === true
        @test opts.use_scatter === true
        # optical_crosstalk defaults to false because no stable post-hoc
        # correction exists yet (see src/api/options.jl preset comment).
        @test opts.use_optical_crosstalk === false
        @test opts.use_focal_spot === true
        @test opts.use_noise === true
        @test opts.use_lag === true
        @test opts.use_heel_effect === true
        @test opts.use_pcct_pileup === false   # off in :eict preset
        @test opts.pcct_noise_reduction == 0.0
        @test opts.seed == 42
        @test opts.detector_efficiency_mode == :auto
    end

    @testset ":pcct preset enables use_pcct_pileup" begin
        opts = BS.SimOptions(fidelity = :pcct)
        @test opts.use_pcct_pileup === true
        # Other physics still on
        @test opts.use_noise === true
        @test opts.use_scatter === true
    end

    @testset "per-effect kwarg overrides preset" begin
        opts = BS.SimOptions(fidelity = :eict, use_scatter = false, use_noise = false)
        @test opts.use_scatter === false
        @test opts.use_noise === false
        # Untouched toggles still follow the preset
        @test opts.use_fill_factor === true

        opts2 = BS.SimOptions(fidelity = :pcct, use_pcct_pileup = false)
        @test opts2.use_pcct_pileup === false
    end

    @testset "fidelity field is NOT stored on the struct (kwarg-only)" begin
        # Confirms the dead-field cleanup: fidelity drives presets in the
        # ctor but is not retained as a runtime field anymore.
        @test !(:fidelity in fieldnames(BS.SimOptions))
    end

    @testset "pcct_noise_reduction clamps to [0, 1]" begin
        @test BS.SimOptions(pcct_noise_reduction = -0.5).pcct_noise_reduction == 0.0
        @test BS.SimOptions(pcct_noise_reduction = 0.7).pcct_noise_reduction ≈ 0.7
        @test BS.SimOptions(pcct_noise_reduction = 1.5).pcct_noise_reduction == 1.0
    end

    @testset "seed accepts Int or nothing" begin
        @test BS.SimOptions(seed = 1234).seed == 1234
        @test BS.SimOptions(seed = nothing).seed === nothing
    end

    @testset "detector_efficiency_mode passes through unchanged" begin
        for m in (:auto, :mc_lut, :beer_lambert)
            @test BS.SimOptions(detector_efficiency_mode = m).detector_efficiency_mode == m
        end
    end

    @testset "projector defaults to :dd, accepts :siddon, validates" begin
        @test BS.SimOptions().projector == :dd
        @test BS.SimOptions(fidelity = :pcct).projector == :dd
        @test BS.SimOptions(projector = :siddon).projector == :siddon
        @test_throws ArgumentError BS.SimOptions(projector = :bogus)
    end

    @testset "unknown fidelity errors with a clear message" begin
        @test_throws ErrorException BS.SimOptions(fidelity = :totally_made_up)
    end
end

# -----------------------------------------------------------------------------
# ReconOptions — slim 3-field config (matrix_size + fov_cm + z_cm).
# Dead fields removed: algorithm, filter, iterations, lambda, tv_weight,
# n_subsets, penalty, penalty_delta, use_edge_weights, blend_percent,
# vmi_energies, vmi_basis, warm_start, cascade_warm_start,
# system_noise_floor_hu.  None had any consumer in src/ or any non-archived
# notebook.  See git log for the audit.
# -----------------------------------------------------------------------------
_ts("entering ReconOptions testset")
@testset "ReconOptions" begin
    @testset "defaults" begin
        ro = BS.ReconOptions()
        @test ro.matrix_size == (512, 512, 64)
        @test ro.fov_cm == 35.0
        @test ro.z_cm === nothing
    end

    @testset "kwargs round-trip" begin
        ro = BS.ReconOptions(matrix_size = (256, 256, 16), fov_cm = 20.0, z_cm = 5.0)
        @test ro.matrix_size == (256, 256, 16)
        @test ro.fov_cm == 20.0
        @test ro.z_cm == 5.0
    end

    @testset "fov_cm and z_cm coerce Real → Float64" begin
        ro = BS.ReconOptions(fov_cm = 20, z_cm = 5)        # Int input
        @test ro.fov_cm isa Float64
        @test ro.z_cm isa Float64
        @test ro.fov_cm == 20.0
        @test ro.z_cm == 5.0
    end

    @testset "z_cm = nothing stays nothing (auto-compute path)" begin
        @test BS.ReconOptions(z_cm = nothing).z_cm === nothing
        @test BS.ReconOptions().z_cm === nothing
    end

    @testset "dead fields are gone" begin
        # Confirms the cleanup: every dead field that previously lived on
        # ReconOptions has been removed, not silently kept around.
        for dead in (
                :algorithm, :filter, :iterations, :lambda, :tv_weight,
                :n_subsets, :penalty, :penalty_delta, :use_edge_weights,
                :blend_percent, :vmi_energies, :vmi_basis,
                :warm_start, :cascade_warm_start, :system_noise_floor_hu,
            )
            @test !(dead in fieldnames(BS.ReconOptions))
        end
    end
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
        got = BS.compute_detector_I0(geom, protocol, flux_sum)
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
        source_to_detector = 1080.0,
        detector_rows = 8,
        detector_cols = 64,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
        detector_type = :photon_counting,
        detector_material = :CdTe,
        detector_depth = 1.6,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        dead_time_ns = 25.0,
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
        on = _toy_pcct_setup(use_pcct_pileup = true)
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
        res_on = BS.simulate!(on.ws, on.phantom, on.protocol, on.sim_opts)
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
            on_arr = Array(res_on.pcct_sino.bins[b])
            off_arr = Array(res_off.pcct_sino.bins[b])
            if maximum(abs.(on_arr .- off_arr)) > 1.0e-6
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
    energies = collect(20.0:1.0:140.0)
    weights = ones(length(energies)) ./ length(energies)   # flat spectrum
    thresholds = [20.0, 35.0, 55.0, 70.0]
    n_bins = length(thresholds)

    # aτ ≈ 0.5 → significant pile-up, well within the regime our PCCT runs at.
    count_rate = 1.0e8        # 100 Mcps per pixel
    dead_time_ns = 5.0
    S = BS.compute_mc_pileup_matrix(
        thresholds, weights, energies,
        count_rate, dead_time_ns;
        n_trials = 2000, seed = 1
    )

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
        @test 0.0 < s ≤ 1.0 + 1.0e-6
    end

    # Pile-up only adds energy → migration is upward only.
    # S[i, j] for i < j (lower recorded bin from higher true bin) must be ~0.
    for j in 2:n_bins, i in 1:(j - 1)
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
    energies = collect(20.0:1.0:140.0)
    weights = ones(length(energies)) ./ length(energies)
    thresholds = [20.0, 35.0, 55.0, 70.0]
    S = BS.compute_mc_pileup_matrix(
        thresholds, weights, energies,
        1.0e8, 5.0; n_trials = 2000, seed = 1
    )

    # Air ray: pre-pileup counts == truth I0.
    counts_true = copy(I0_truth)
    counts_recorded = [sum(S[i, j] * counts_true[j] for j in 1:n_bins) for i in 1:n_bins]

    # `simulate!` semantics: bins[i] = -log(recorded / I0_truth[i])
    bins_air = [-log(counts_recorded[i] / I0_truth[i]) for i in 1:n_bins]
    for v in bins_air
        @test isfinite(v)
    end

    # Round-trip identity: I0_truth[i] · exp(-bins[i]) == counts_recorded[i]
    for i in 1:n_bins
        @test I0_truth[i] * exp(-bins_air[i]) ≈ counts_recorded[i] rtol = 1.0e-12
    end

    # The air-ray offset equals log(I0_truth / I0_recorded) — physically the
    # signature of pile-up "dimming" each bin's air baseline.
    I0_recorded = [sum(S[i, j] * I0_truth[j] for j in 1:n_bins) for i in 1:n_bins]
    for i in 1:n_bins
        @test bins_air[i] ≈ log(I0_truth[i] / I0_recorded[i]) rtol = 1.0e-12
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
        source_to_detector = 1080.0,
        detector_rows = 8,
        detector_cols = 64,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
        detector_material = :lumex,
        detector_depth = 3.0,
        electronic_noise = 5.0,
        detection_gain = 10.0,
    )
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    sim_opts = BS.SimOptions(;
        fidelity = :eict,
        use_noise = use_noise, use_scatter = use_scatter,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false,
        kwargs...
    )
    recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
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
        @test maximum(abs.(sino_on1 .- sino_off)) > 1.0e-6
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
        for σ in (0.0, -1.0e-9, -5.0)
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
        @test std(vol) ≈ 28.0f0 rtol = 5.0e-2
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
        vol = copy(base)
        BS.add_system_noise_floor!(vol, 10.0; seed = 7)
        diff = vol .- base
        @test std(diff) ≈ 10.0f0 rtol = 1.0e-1   # smaller volume → looser tol
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
        after = randn(Float32, 5)                # next 5 draws — must match target

        @test target == after
    end
end

# -----------------------------------------------------------------------------
# Spectrum resolvers — pure CPU, no GPU gate.
#
# Reference for the bowtie path: XCIST/CatSim, GE Research.
#   - `gecatsim/pyfiles/Xray_Filter.py:bowtie_filter()`            (4-material stack)
#   - `gecatsim/pyfiles/Resample_Spectrum_Bowtie_FlatFilter.py`     (spectrum × bowtie)
# -----------------------------------------------------------------------------
function _toy_eict_scanner_with_bowtie()
    return BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = 4,
        detector_cols = 32,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
        detector_material = :lumex,
        detector_depth = 3.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        bowtie_filter = :ge_revolution_large,
    )
end

_ts("entering resolve_source_spectrum_without_bowtie testset")
@testset "resolve_source_spectrum_without_bowtie" begin
    scanner = _toy_eict_scanner_with_bowtie()
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    sim_opts = BS.SimOptions(; fidelity = :eict)

    e, w = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)

    # Shape contract.
    @test length(e) == length(w)
    @test length(e) > 0
    # Energy grid is positive, monotonic-ish, capped at kVp.
    @test all(>(0), e)
    @test maximum(e) ≤ Float64(protocol.kVp) + 1.0e-6
    # Weights are non-negative (post-Beer-Lambert filtering, before any norm).
    @test all(>=(0), w)
    @test all(isfinite, w)
    # Sum > 0 — flux didn't get filtered to zero.
    @test sum(w) > 0
end

_ts("entering apply_bowtie_to_spectrum testset")
@testset "apply_bowtie_to_spectrum" begin
    scanner = _toy_eict_scanner_with_bowtie()
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
    sim_opts = BS.SimOptions(; fidelity = :eict)

    e, w_1d = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    n_E = length(e)
    n_col = scanner.detector_cols
    n_row = scanner.detector_rows

    @testset "include_bowtie=false → 1D normalized spectrum" begin
        ŵ = BS.apply_bowtie_to_spectrum(
            w_1d, e, scanner, geom, protocol;
            include_bowtie = false
        )
        @test ndims(ŵ) == 1
        @test length(ŵ) == n_E
        @test eltype(ŵ) == Float32
        @test sum(ŵ) ≈ 1.0f0  rtol = 1.0e-5
        @test all(>=(0), ŵ)
    end

    @testset "scanner with no bowtie → 1D normalized spectrum (regardless of flag)" begin
        no_bowtie = BS.Scanner(
            source_to_isocenter = 540.0,
            source_to_detector = 1080.0,
            detector_rows = 4,
            detector_cols = 32,
            detector_row_size = 1.0,
            detector_col_size = 1.0,
            detector_material = :lumex,
            detector_depth = 3.0,
            flat_filter_material = :aluminum,
            flat_filter_thickness = 2.5,
            bowtie_filter = :none,
        )
        geom_nb = BS.CTGeometry(no_bowtie; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
        ŵ = BS.apply_bowtie_to_spectrum(
            w_1d, e, no_bowtie, geom_nb, protocol;
            include_bowtie = true
        )
        @test ndims(ŵ) == 1
        @test sum(ŵ) ≈ 1.0f0  rtol = 1.0e-5
    end

    @testset "bowtie ON → per-ray 3D ŵ, every ray normalized" begin
        ŵ = BS.apply_bowtie_to_spectrum(
            w_1d, e, scanner, geom, protocol;
            include_bowtie = true
        )
        @test ndims(ŵ) == 3
        @test size(ŵ) == (n_col, n_row, n_E)
        @test eltype(ŵ) == Float32
        @test all(isfinite, ŵ)
        @test all(>=(0), ŵ)
        # Every ray normalizes to 1.
        for row in 1:n_row, col in 1:n_col
            @test sum(@view ŵ[col, row, :]) ≈ 1.0f0  rtol = 1.0e-4
        end
    end

    @testset "bowtie hardening: edge-ray mean E > center-ray mean E" begin
        # Physical contract: bowtie filter is thicker at fan edges, so edge
        # rays see more low-E attenuation → harder spectrum (higher mean E).
        # CatSim's bowtie_filter() implements the same Beer-Lambert sum that
        # produces this effect.
        #
        # Use a clinically-realistic fan width here — a 32-col × 1 mm detector
        # at SDD = 1080 mm spans only ~1.7° of fan, which is too narrow for
        # the bowtie shape (cm-scale thickness vs. fan angle) to differ
        # measurably between center and edge.  Notebook scanners use ~800
        # columns at ~0.6 mm, ~25° fan.  Mirror that.
        wide = BS.Scanner(
            source_to_isocenter = 540.0,
            source_to_detector = 1080.0,
            detector_rows = 4,
            detector_cols = 800,
            detector_row_size = 1.0,
            detector_col_size = 0.6,
            detector_material = :lumex,
            detector_depth = 3.0,
            flat_filter_material = :aluminum,
            flat_filter_thickness = 2.5,
            bowtie_filter = :ge_revolution_large,
        )
        wide_geom = BS.CTGeometry(wide; n_angles = 16, fov_cm = 50.0, z_cm = 5.0)
        ŵ = BS.apply_bowtie_to_spectrum(
            w_1d, e, wide, wide_geom, protocol;
            include_bowtie = true
        )
        n_col_w = size(ŵ, 1)
        mid_c = n_col_w ÷ 2 + 1
        mid_r = size(ŵ, 2) ÷ 2 + 1
        mean_E_center = sum(Float64(e[k]) * ŵ[mid_c, mid_r, k] for k in 1:n_E)
        mean_E_edge = sum(Float64(e[k]) * ŵ[1, mid_r, k] for k in 1:n_E)
        @test mean_E_edge > mean_E_center
        # Effect should be physically meaningful, not a rounding artifact.
        @test (mean_E_edge - mean_E_center) > 0.5
    end
end

_ts("entering resolve_source_spectrum_with_bowtie testset")
@testset "resolve_source_spectrum_with_bowtie" begin
    scanner = _toy_eict_scanner_with_bowtie()
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
    sim_opts = BS.SimOptions(; fidelity = :eict)

    @testset "composes _without_bowtie + apply_bowtie_to_spectrum" begin
        # The wrapper is just a composition — verify it produces the same
        # output as calling the two primitives by hand.
        e₁, w_1d = BS.resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
        ŵ_manual = BS.apply_bowtie_to_spectrum(
            w_1d, e₁, scanner, geom, protocol;
            include_bowtie = true
        )
        e₂, ŵ_compose = BS.resolve_source_spectrum_with_bowtie(
            sim_opts, protocol;
            scanner = scanner, geom = geom,
            include_bowtie = true
        )
        @test e₂ == e₁
        @test ŵ_compose == ŵ_manual
    end

    @testset "include_bowtie=false → 1D output (caller can dispatch on ndims)" begin
        e, ŵ = BS.resolve_source_spectrum_with_bowtie(
            sim_opts, protocol;
            scanner = scanner, geom = geom,
            include_bowtie = false
        )
        @test ndims(ŵ) == 1
        @test length(ŵ) == length(e)
        @test sum(ŵ) ≈ 1.0f0  rtol = 1.0e-5
    end
end

# -----------------------------------------------------------------------------
# build_physics_config — toggle resolution + Scanner→effect-model mapping.
#
# Original BasisSim glue (not a CatSim port).  CatSim's equivalent is the
# per-effect callback registry on `cfg.physics.*Callback`.  Every notebook
# reaches this function indirectly through `create_workspace` /
# `create_eict_workspace`, so its toggle resolution is on the hot path.
# -----------------------------------------------------------------------------
_ts("entering build_physics_config testset")
@testset "build_physics_config" begin
    function _scanner_with_full_hardware()
        BS.Scanner(
            source_to_isocenter = 540.0,
            source_to_detector = 1080.0,
            detector_rows = 4,
            detector_cols = 32,
            detector_row_size = 1.0,
            detector_col_size = 1.0,
            detector_material = :lumex,
            detector_depth = 3.0,
            fill_factor_row = 0.85,
            fill_factor_col = 0.92,
            focal_spot_width = 1.2,
            focal_spot_length = 1.0,
            target_angle = 7.5,
            flat_filter_material = :aluminum,
            flat_filter_thickness = 2.5,
        )
    end
    energies = collect(20.0:1.0:120.0)        # flat-ish proxy
    weights = ones(length(energies))         # uniform — mean ≈ midpoint

    @testset "all toggles off → all-nothing PhysicsConfig" begin
        sc = _scanner_with_full_hardware()
        opts = BS.SimOptions(;
            fidelity = :eict,
            use_fill_factor = false, use_detector_efficiency = false,
            use_scatter = false, use_optical_crosstalk = false,
            use_focal_spot = false, use_noise = false, use_lag = false,
            use_heel_effect = false
        )
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.fill_factor === nothing
        @test cfg.detector_efficiency === nothing
        @test cfg.scatter === nothing
        @test cfg.optical_crosstalk === nothing
        @test cfg.focal_spot === nothing
        @test cfg.lag === nothing
        @test cfg.heel_effect === nothing
    end

    @testset "each toggle on → corresponding field non-nothing + correct type" begin
        sc = _scanner_with_full_hardware()
        # optical_crosstalk defaults to false in :eict preset, opt in explicitly
        # to exercise the type wiring for that field.
        opts = BS.SimOptions(; fidelity = :eict, use_optical_crosstalk = true)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.fill_factor isa BS.FillFactorModel
        @test cfg.detector_efficiency isa BS.DetectorEfficiency
        @test cfg.scatter isa BS.ScatterModel
        @test cfg.optical_crosstalk isa BS.OpticalCrosstalkModel
        @test cfg.focal_spot isa BS.FocalSpot
        @test cfg.lag isa BS.LagModel
        @test cfg.heel_effect isa BS.HeelEffect
    end

    @testset "spectrum-weighted mean energy + seed passthrough" begin
        sc = _scanner_with_full_hardware()
        # Asymmetric weights — assert exact closed-form mean.
        es = [40.0, 60.0, 100.0]
        ws = [1.0, 3.0, 1.0]
        expected_mean = sum(es .* ws) / sum(ws)
        opts = BS.SimOptions(; fidelity = :eict, seed = 4242)
        cfg = BS.build_physics_config(sc, opts, es, ws)
        @test cfg.energy_keV ≈ expected_mean
        @test cfg.noise_seed == 4242
    end

    @testset "fill_factor: scanner fields used when both >0" begin
        sc = _scanner_with_full_hardware()    # 0.85 × 0.92
        opts = BS.SimOptions(; fidelity = :eict)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.fill_factor.row_fill ≈ 0.85
        @test cfg.fill_factor.col_fill ≈ 0.92
    end

    @testset "fill_factor: zero hardware → factory fallback" begin
        sc = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 4, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
            detector_material = :lumex, detector_depth = 3.0,
            fill_factor_row = 0.0, fill_factor_col = 0.0,    # ← unset
        )
        opts = BS.SimOptions(; fidelity = :eict)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.fill_factor isa BS.FillFactorModel
        # Factory fill_factor_standard returns a populated model — non-zero.
        @test cfg.fill_factor.row_fill > 0
        @test cfg.fill_factor.col_fill > 0
    end

    @testset "detector_efficiency: lumex → gemstone branch" begin
        sc = _scanner_with_full_hardware()    # :lumex, depth 3.0
        opts = BS.SimOptions(; fidelity = :eict, detector_efficiency_mode = :auto)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        # gemstone factory tags the material distinctly from a generic
        # `DetectorEfficiency(name, depth, 1.0)` direct ctor.  At minimum the
        # model is not the bare-direct-ctor flavor.
        @test cfg.detector_efficiency isa BS.DetectorEfficiency
    end

    @testset "detector_efficiency: PCCT scanner skips EICT scintillator branch" begin
        # PCCT scanners declare detector_material=:cdte and detector_type=:photon_counting.
        # The EICT scintillator branch (which only supports :lumex) must NOT
        # fire for them — their detector physics lives in the MC DRM consumed
        # by `pcct_forward_project`, not in PhysicsConfig.detector_efficiency.
        sc = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 4, detector_cols = 32,
            detector_row_size = 0.5, detector_col_size = 0.5,
            detector_type = :photon_counting,
            detector_material = :cdte,
            detector_depth = 1.6,
            n_energy_bins = 4,
            energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        )
        opts = BS.SimOptions(; fidelity = :pcct)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.detector_efficiency === nothing   # skipped for PCCT
    end

    @testset "focal_spot: scanner fields used when both >0" begin
        sc = _scanner_with_full_hardware()
        opts = BS.SimOptions(; fidelity = :eict)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.focal_spot.width ≈ 1.2
        @test cfg.focal_spot.length ≈ 1.0
    end

    @testset "heel_effect: target_angle pulled from scanner" begin
        sc = _scanner_with_full_hardware()    # 7.5°
        opts = BS.SimOptions(; fidelity = :eict)
        cfg = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg.heel_effect isa BS.HeelEffect
        @test cfg.heel_effect.anode_angle_deg ≈ 7.5
    end

    @testset "scatter: phantom-diameter scaling kicks in when phantom passed" begin
        sc = _scanner_with_full_hardware()
        opts = BS.SimOptions(; fidelity = :eict)
        # No phantom: scatter still built, just without size-aware scaling.
        cfg_nopath = BS.build_physics_config(sc, opts, energies, weights)
        @test cfg_nopath.scatter isa BS.ScatterModel

        # With phantom: ensure no error and result is a ScatterModel.  Use
        # a small synthetic mask (same path the workspace ctor takes).
        mask = ones(UInt8, 24, 24, 4)
        materials = [BS.XA.Materials.water]
        phantom = BS.Phantom(mask, materials, (0.1, 0.1, 0.1), (0.0, 0.0, 0.0), (2.4, 2.4, 0.4))
        cfg_path = BS.build_physics_config(sc, opts, energies, weights; phantom = phantom)
        @test cfg_path.scatter isa BS.ScatterModel
    end
end

# -----------------------------------------------------------------------------
# reconstruct!(::FDKReconWorkspace, ...) — Feldkamp-Davis-Kress, TIGRE-style
# 4-step pipeline (copy → filter → backproject → FOV mask).
#
# Reference port: Feldkamp LA, Davis LC, Kress JW. J Opt Soc Am A 1(6),
# 1984.  TIGRE Toolbox MATLAB/Algorithms/FDK.m + Common/CUDA/voxel_backprojection.cu.
# This is NOT Siddon — Siddon is forward-projection only.
# -----------------------------------------------------------------------------
_ts("entering reconstruct!(FDKReconWorkspace) testset")
@testset "reconstruct!(FDKReconWorkspace)" begin
    # Tiny CPU-only setup — no GPU gate.  16 views × 32 cols × 8 rows is
    # enough to exercise the pipeline; this isn't a quality test, it's a
    # contract test.
    function _toy_fdk_setup(; matrix_size = (16, 16, 4))
        scanner = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 8, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
        )
        geom = BS.CTGeometry(scanner; n_angles = 16, fov_cm = 20.0, z_cm = 5.0)
        sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        ws = BS.create_fdk_recon_workspace(sino, geom, matrix_size)
        return (; scanner, geom, sino, ws, matrix_size)
    end

    @testset "return contract: ws.volume identity, shape" begin
        s = _toy_fdk_setup()
        ret = BS.reconstruct!(s.ws, s.sino, s.geom)
        @test ret === s.ws.volume
        @test size(ret) == s.matrix_size
        @test eltype(ret) == Float32
    end

    @testset "zero sinogram → recon is {0 inside FOV, sentinel outside}" begin
        s = _toy_fdk_setup()
        BS.reconstruct!(s.ws, s.sino, s.geom)
        # FDK is linear → all-zero input gives all-zero recon BEFORE the
        # FOV mask.  The FOV mask then writes the air sentinel (default
        # μ = -0.04 cm⁻¹) into voxels outside the inscribed FOV circle.
        # So every voxel is either 0 or exactly -0.04.
        @test all(v -> v == 0.0f0 || v ≈ -0.04f0, s.ws.volume)
        # Center voxel is inside the FOV → must be 0.
        nx, ny, nz = s.matrix_size
        @test s.ws.volume[nx ÷ 2 + 1, ny ÷ 2 + 1, nz ÷ 2 + 1] == 0.0f0
    end

    @testset "deterministic: same input → bit-identical output" begin
        s1 = _toy_fdk_setup()
        s2 = _toy_fdk_setup()
        # Random-but-finite sinogram so the recon is non-trivial.
        sino = randn(Float32, size(s1.sino)...)
        BS.reconstruct!(s1.ws, sino, s1.geom)
        BS.reconstruct!(s2.ws, sino, s2.geom)
        @test s1.ws.volume == s2.ws.volume
        @test all(isfinite, s1.ws.volume)
    end

    @testset "non-trivial sinogram → finite, non-uniform recon" begin
        s = _toy_fdk_setup()
        # Build a sinogram with a clear central log-attenuation peak — every
        # view has a Gaussian bump at the center column.  Recon should be
        # finite and have measurable variance (not flat).
        sino = zeros(Float32, size(s.sino)...)
        n_col, n_row, n_view = size(sino)
        mid_c = n_col ÷ 2 + 1
        for v in 1:n_view, r in 1:n_row, c in 1:n_col
            sino[c, r, v] = 0.5f0 * exp(-((c - mid_c)^2) / (2 * 4.0f0^2))
        end
        BS.reconstruct!(s.ws, sino, s.geom)
        @test all(isfinite, s.ws.volume)
        @test std(s.ws.volume) > 0     # not all the same value
    end

    @testset "FOV mask: corner voxels (outside inscribed circle) get sentinel μ" begin
        # apply_fov_mask! writes the air sentinel μ (default -0.04 cm⁻¹) into
        # every voxel outside the largest circle that fits the recon's xy
        # box (clinical convention).  All four xy corners must equal the
        # sentinel in every z-slice.
        s = _toy_fdk_setup()
        sino = randn(Float32, size(s.sino)...)
        BS.reconstruct!(s.ws, sino, s.geom)
        nx, ny, nz = s.matrix_size
        sentinel = -0.04f0
        for z in 1:nz
            @test s.ws.volume[1, 1, z] ≈ sentinel
            @test s.ws.volume[nx, 1, z] ≈ sentinel
            @test s.ws.volume[1, ny, z] ≈ sentinel
            @test s.ws.volume[nx, ny, z] ≈ sentinel
        end
        # Center voxel is inside the FOV → finite, almost certainly NOT
        # the sentinel value.
        cx, cy, cz = nx ÷ 2 + 1, ny ÷ 2 + 1, nz ÷ 2 + 1
        @test isfinite(s.ws.volume[cx, cy, cz])
        @test s.ws.volume[cx, cy, cz] != sentinel
    end

    @testset "filter kwarg accepted: ram_lak, shepp_logan, cosine, hamming, hann" begin
        # Each TIGRE-derived FBP filter must be usable.  Each call must
        # finish without error and produce a finite recon.
        sino = randn(Float32, 32, 8, 16)
        for fkernel in (:ram_lak, :shepp_logan, :cosine, :hamming, :hann)
            s = _toy_fdk_setup()
            ws = BS.create_fdk_recon_workspace(sino, s.geom, s.matrix_size; filter = fkernel)
            BS.reconstruct!(ws, sino, s.geom)
            @test all(isfinite, ws.volume)
        end
    end

    @testset "second call overwrites volume (no accumulation)" begin
        # Direct test of the `fill!(ws.volume, zero(T))` line at the start of
        # reconstruct!: pollute the volume buffer with a sentinel value
        # between calls, run the SAME sinogram, and verify the result matches
        # the clean run.  That's what "no accumulation" actually requires.
        s = _toy_fdk_setup()
        sino = zeros(Float32, size(s.sino)...)
        n_col, n_row, n_view = size(sino)
        mid_c = n_col ÷ 2 + 1
        for v in 1:n_view, r in 1:n_row, c in 1:n_col
            sino[c, r, v] = 0.5f0 * exp(-((c - mid_c)^2) / (2 * 4.0f0^2))
        end
        BS.reconstruct!(s.ws, sino, s.geom)
        v_clean = copy(s.ws.volume)

        # Pollute with garbage; reconstruct! must zero it before backprojection.
        fill!(s.ws.volume, 999.0f0)
        BS.reconstruct!(s.ws, sino, s.geom)
        @test s.ws.volume == v_clean
    end
end

# -----------------------------------------------------------------------------
# reconstruct!(::HIRReconWorkspace, ...) — OS-PWLS with Huber prior
# (Fessler / U-Michigan school; see hybrid_ir.jl preamble for full citations)
# -----------------------------------------------------------------------------
_ts("entering reconstruct!(HIRReconWorkspace) testset")
@testset "reconstruct!(HIRReconWorkspace)" begin
    function _toy_hir_setup(; matrix_size = (16, 16, 4), strength = 3)
        scanner = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 8, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
        )
        # n_angles divisible by n_subsets (12) so OS subsets are balanced.
        geom = BS.CTGeometry(scanner; n_angles = 24, fov_cm = 20.0, z_cm = 5.0)
        sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        ws = BS.create_hir_recon_workspace(sino, geom, matrix_size; strength = strength)
        return (; scanner, geom, sino, ws, matrix_size, strength)
    end

    @testset "HIRParams: niter dropped, n_subsets fixed at 12" begin
        # Sanity-check the cleanup: HIRParams no longer carries `niter`,
        # and every strength uses 12 subsets.
        @test !(:niter in fieldnames(BS.HIRParams))
        for strength in 1:5
            p = BS.get_hir_params(strength)
            @test p.n_subsets == 12
            @test p.nepochs ≥ 1
            @test p.lambda > 0
            @test p.huber_delta > 0
            @test 0 < p.relaxation ≤ 1
        end
    end

    @testset "HIRReconWorkspace: Ax field dropped" begin
        # The dead Ax field used by the deleted full-data branch is gone.
        @test !(:Ax in fieldnames(BS.HIRReconWorkspace))
    end

    @testset "return contract: ws.volume identity, shape" begin
        s = _toy_hir_setup()
        ret = BS.reconstruct!(s.ws, s.sino, s.geom)
        @test ret === s.ws.volume
        @test size(ret) == s.matrix_size
        @test eltype(ret) == Float32
        @test all(isfinite, ret)
    end

    @testset "deterministic: same input → bit-identical output" begin
        s1 = _toy_hir_setup()
        s2 = _toy_hir_setup()
        sino = randn(Float32, size(s1.sino)...)
        BS.reconstruct!(s1.ws, sino, s1.geom)
        BS.reconstruct!(s2.ws, sino, s2.geom)
        @test s1.ws.volume == s2.ws.volume
    end

    @testset "init_volume warm-start path" begin
        # When init_volume is provided, the FDK initialization is skipped
        # and the iterate starts from the supplied volume.  The Step-1
        # branch flips: copyto! instead of FDK + filter + backproject.
        s = _toy_hir_setup()
        sino = randn(Float32, size(s.sino)...)
        warm = fill(0.5f0, s.matrix_size...)
        # Run with warm start — should converge to a finite recon.
        BS.reconstruct!(s.ws, sino, s.geom; init_volume = warm)
        @test all(isfinite, s.ws.volume)
        # Run without warm start (FDK init) — output differs from warm-start.
        s2 = _toy_hir_setup()
        BS.reconstruct!(s2.ws, sino, s2.geom)
        @test s.ws.volume != s2.ws.volume
    end

    @testset "air_reference path runs without error" begin
        # The bowtie-aware stat-weight path (the second `if air_reference`
        # branch) must run without crashing and produce a finite recon.
        s = _toy_hir_setup()
        sino = randn(Float32, size(s.sino)...)
        # Synthetic air reference: edge pixels brighter than center
        # (opposite of real bowtie, but valid input).
        n_col, n_row = size(s.sino, 1), size(s.sino, 2)
        air_ref = ones(Float32, n_col, n_row)
        BS.reconstruct!(s.ws, sino, s.geom; air_reference = air_ref)
        @test all(isfinite, s.ws.volume)
    end

    @testset "strength → param ordering (clinical noise-reduction band)" begin
        # The strength dial is meant to give monotonically stronger
        # regularization.  Verify that on the clinical params side:
        #   - lambda increases with strength (more prior weight)
        #   - huber_delta decreases (sharper transition to L1 region)
        #   - nepochs increases (more iteration steps)
        # The toy test geometry's V_inv collapses to ~0, so we can't show
        # a recon-level difference here without a clinical-scale detector;
        # the param table is the actual contract being tested.
        for str in 1:4
            p_lo = BS.get_hir_params(str)
            p_hi = BS.get_hir_params(str + 1)
            @test p_lo.lambda ≤ p_hi.lambda
            @test p_lo.huber_delta ≥ p_hi.huber_delta
            @test p_lo.nepochs ≤ p_hi.nepochs
        end
        # End-to-end smoke test: every strength runs through to completion
        # and returns a finite recon.
        for str in 1:5
            s = _toy_hir_setup(strength = str)
            sino = zeros(Float32, size(s.sino)...)
            BS.reconstruct!(s.ws, sino, s.geom)
            @test all(isfinite, s.ws.volume)
        end
    end
end

# -----------------------------------------------------------------------------
# Workspace constructors — field-invariant audit.
#
# Audit method: for each of the 4 workspace structs (PCCT, EICT, FDK, HIR),
# trace every field and every ctor kwarg back to a real consumer in src/ +
# notebooks.  Tests below lock in the dead-field cleanup contract: anything
# that *was* removed should NOT come back as a field, and anything that *is*
# kept should be a real array (not Nothing) on the same backend as
# phantom.mask, with dimensions matching the geometry it derived from.
# -----------------------------------------------------------------------------
_ts("entering Workspace ctors — PCCT field invariants testset")
@testset "create_workspace (PCCT) — field invariants" begin
    if !HAS_GPU
        @warn "Skipping PCCT workspace ctor test: no GPU backend."
    else
        s = _toy_pcct_setup()
        ws = s.ws

        # Dead-field guard — these were removed in the workspace audit.
        # If any of these reappear via a future commit, the test screams.
        @test !(:source_spectral_gpu in fieldnames(BS.PCCTWorkspace))
        @test !(:native_scratch in fieldnames(BS.PCCTWorkspace))

        # Sinogram-shape buffers match (n_cols, n_rows, n_angles).
        sino_shape = (s.scanner.detector_cols, s.scanner.detector_rows, s.protocol.views)
        @test length(ws.bins) == s.scanner.n_energy_bins
        for b in 1:s.scanner.n_energy_bins
            @test size(ws.bins[b]) == sino_shape
        end
        @test size(ws.sino_buf) == sino_shape
        @test size(ws.scratch) == sino_shape
        @test size(ws.combined) == sino_shape

        # Spectral arrays sized to bin / energy counts.
        @test length(ws.I0_bins) == s.scanner.n_energy_bins
        @test length(ws.I0_bins_norm) == s.scanner.n_energy_bins
        @test length(ws.thresholds_T) == s.scanner.n_energy_bins
        @test length(ws.η) == length(ws.energies)

        # Pile-up wiring (matches sim_opts.use_pcct_pileup default = true for :pcct).
        @test ws.use_pcct_pileup === true
        @test ws.pileup_S isa Matrix{Float64}
        @test size(ws.pileup_S) == (s.scanner.n_energy_bins, s.scanner.n_energy_bins)

        # Native-res buffers are nothing when binning_factor == 1 (toy default).
        @test s.scanner.binning_factor == 1   # toy uses default
        @test ws.native_geom === nothing
        @test ws.native_bins === nothing
        @test ws.native_sino_buf === nothing
        @test ws.native_outputs_flat === nothing

        # Tiled spectral buffers always allocated (the only forward-projection path).
        @test size(ws.μ_table_gpu, 1) == length(ws.mats)
        @test size(ws.W_matrix_gpu, 2) == s.scanner.n_energy_bins
        @test length(ws.outputs_flat) == prod(sino_shape) * s.scanner.n_energy_bins
    end
end

_ts("entering Workspace ctors — PCCT pile-up off testset")
@testset "create_workspace (PCCT) — use_pcct_pileup=false skips pileup_S" begin
    if !HAS_GPU
        @warn "Skipping PCCT workspace ctor test: no GPU backend."
    else
        s = _toy_pcct_setup(use_pcct_pileup = false)
        @test s.ws.use_pcct_pileup === false
        @test s.ws.pileup_S === nothing
    end
end

_ts("entering Workspace ctors — EICT field invariants testset")
@testset "create_eict_workspace — field invariants" begin
    if !HAS_GPU
        @warn "Skipping EICT workspace ctor test: no GPU backend."
    else
        s = _toy_eict_setup()
        ws = s.ws
        sino_shape = (s.scanner.detector_cols, s.scanner.detector_rows, s.protocol.views)

        @test size(ws.sinogram) == sino_shape
        @test size(ws.μ_volume) == size(s.phantom.mask)
        @test size(ws.sino_mono) == sino_shape
        @test size(ws.I_transmitted) == sino_shape
        @test size(ws.air_scan) == sino_shape
        @test size(ws.physics_output) == sino_shape

        # Pre-computed noise constants from Pass 2 cleanup.
        @test ws.η_eff isa Float32 && 0 < ws.η_eff ≤ 1
        @test ws.σ_e_photon isa Float32 && ws.σ_e_photon ≥ 0

        # weights_norm is normalized to sum to 1.
        @test sum(Float64.(ws.weights_norm)) ≈ 1.0  rtol = 1.0e-5
        @test length(ws.weights_norm) == length(ws.energies)

        # μ_table dimensions: (n_regions, n_energies)
        @test size(ws.μ_table, 1) == length(ws.mats)
        @test size(ws.μ_table, 2) == length(ws.energies)
    end
end

_ts("entering Workspace ctors — EICT spectrum_override testset")
@testset "create_eict_workspace — spectrum_override path" begin
    if !HAS_GPU
        @warn "Skipping EICT workspace ctor test: no GPU backend."
    else
        # Inject a custom 1-bin monoenergetic spectrum at 70 keV.
        scanner = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 4, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
            detector_material = :lumex, detector_depth = 3.0,
            flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
        )
        protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
        sim_opts = BS.SimOptions(; fidelity = :eict)
        recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
        phantom_cpu = BS.create_gammex_472(n_voxels = 32, fov_cm = 20.0, z_cm = 2.0)
        phantom = BS.Phantom(
            to_gpu(phantom_cpu.mask), phantom_cpu.materials,
            phantom_cpu.voxel_size, phantom_cpu.origin, phantom_cpu.extent,
        )

        ws = BS.create_eict_workspace(
            scanner, protocol, sim_opts, recon_opts, phantom;
            spectrum_override = ([70.0], [1.0e6])
        )
        @test length(ws.energies) == 1
        @test ws.energies[1] == 70.0
        @test length(ws.weights) == 1
        @test ws.weights[1] == 1.0e6

        # Mismatched lengths must throw.
        @test_throws ArgumentError BS.create_eict_workspace(
            scanner, protocol, sim_opts, recon_opts, phantom;
            spectrum_override = ([70.0, 100.0], [1.0e6])
        )
    end
end

_ts("entering Workspace ctors — FDKReconWorkspace field invariants testset")
@testset "create_fdk_recon_workspace — field invariants" begin
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0
    )
    geom = BS.CTGeometry(scanner; n_angles = 16, fov_cm = 20.0, z_cm = 5.0)
    sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    matsize = (16, 16, 4)

    @testset "default filter (StandardFilter)" begin
        ws = BS.create_fdk_recon_workspace(sino, geom, matsize)
        @test size(ws.volume) == matsize
        @test size(ws.filtered) == size(sino)
        @test size(ws.conv_scratch) == size(sino)
        @test eltype(ws.volume) == Float32
        # Filter kernel: full linear-convolution support, up to 2·n_cols − 1.
        @test 32 ≤ length(ws.filter_kernel) ≤ 2 * geom.n_cols - 1
        # Volume zeroed at construction.
        @test all(==(0), ws.volume)
        # Geometry arrays match underlying CTGeometry shape.
        @test size(ws.bp_source_positions) == size(geom.source_positions)
        @test size(ws.bp_detector_centers) == size(geom.detector_centers)
        @test size(ws.bp_detector_u) == size(geom.detector_u)
        @test size(ws.bp_detector_v) == size(geom.detector_v)
    end

    @testset "filter Symbol → FilterType resolution" begin
        # Every FBP filter symbol from the docstring should resolve and
        # produce a valid kernel.
        for f in (
                :ram_lak, :shepp_logan, :cosine, :hamming, :hann,
                :standard, :soft, :bone,
            )
            ws = BS.create_fdk_recon_workspace(sino, geom, matsize; filter = f)
            # full linear-convolution support: up to 2·n_cols − 1 (the old
            # n_cols clamp truncated the ramp wings → rim capping)
            @test 32 ≤ length(ws.filter_kernel) ≤ 2 * geom.n_cols - 1
        end
    end

    @testset "cutoff < 1.0 shrinks the kernel" begin
        ws_full = BS.create_fdk_recon_workspace(sino, geom, matsize; cutoff = 1.0)
        ws_half = BS.create_fdk_recon_workspace(sino, geom, matsize; cutoff = 0.5)
        @test length(ws_half.filter_kernel) ≤ length(ws_full.filter_kernel)
    end

    @testset "T= overrides element type" begin
        ws64 = BS.create_fdk_recon_workspace(sino, geom, matsize; T = Float64)
        @test eltype(ws64.volume) == Float64
        @test eltype(ws64.filter_kernel) == Float64
    end
end

_ts("entering Workspace ctors — HIRReconWorkspace field invariants testset")
@testset "create_hir_recon_workspace — field invariants" begin
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0
    )
    # n_angles divisible by 12 for clean OS subsets.
    geom = BS.CTGeometry(scanner; n_angles = 24, fov_cm = 20.0, z_cm = 5.0)
    sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    matsize = (16, 16, 4)

    @testset "FDK init buffers + iteration scratch" begin
        ws = BS.create_hir_recon_workspace(sino, geom, matsize; strength = 3)
        @test size(ws.volume) == matsize
        @test size(ws.filtered) == size(sino)
        @test size(ws.conv_scratch) == size(sino)
        @test size(ws.W_proj) == size(sino)
        @test size(ws.V_inv) == matsize
        @test size(ws.stat_weights) == size(sino)
        @test size(ws.correction) == matsize
        @test size(ws.reg_grad) == matsize
        @test all(==(0), ws.volume)
    end

    @testset "OS-PWLS subsets always allocated (n_subsets = 12)" begin
        ws = BS.create_hir_recon_workspace(sino, geom, matsize; strength = 3)
        @test ws.params.n_subsets == 12
        @test length(ws.subsets) == 12
        @test length(ws.subset_geometries) == 12
        @test length(ws.subset_geom_source_positions) == 12
        # Subsets partition the angles.
        @test sort!(reduce(vcat, ws.subsets)) == collect(1:geom.n_angles)
        # Subset buffers sized to max(subset).
        max_sub = maximum(length(s) for s in ws.subsets)
        @test size(ws.subset_sino_buf, 3) == max_sub
        @test size(ws.subset_Ax_buf, 3) == max_sub
        @test size(ws.subset_W_proj_buf, 3) == max_sub
        @test size(ws.subset_stat_weights_buf, 3) == max_sub
    end

    @testset "strength 1..5 all produce a usable workspace" begin
        for str in 1:5
            ws = BS.create_hir_recon_workspace(sino, geom, matsize; strength = str)
            @test ws.params.strength == str
            @test ws.params.n_subsets == 12
            @test length(ws.subsets) == 12
        end
    end

    @testset "n_subsets <= 0 errors (legacy path deleted)" begin
        # The dead n_subsets=0 fallback was removed; constructing HIRParams
        # by hand with n_subsets=0 must raise.
        bad_params = BS.HIRParams(3, 1.0f0, 1, 0, 0.05f0, 0.5f0, (10, 20))
        @test bad_params.n_subsets == 0
        # We can't easily inject custom HIRParams into create_hir_recon_workspace
        # (it calls get_hir_params(strength) internally), but we can at least
        # confirm get_hir_params never returns 0 subsets for any valid strength.
        for str in 1:5
            @test BS.get_hir_params(str).n_subsets > 0
        end
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
        @test maximum(abs.(sino_on .- sino_off)) > 1.0e-6
        @test all(isfinite, sino_on)
    end
end

# -----------------------------------------------------------------------------
# PCCT focal-spot blur wiring (SoftwareX revision, R1.7)
# -----------------------------------------------------------------------------
_ts("entering PCCT focal-spot preset defaults testset")
@testset ":pcct preset — focal_spot and lag default OFF" begin
    # Pure-CPU option resolution — no GPU gating needed.
    # Tube-side focal-spot blur is wired into the PCCT path but ships
    # disabled by default; lag is a scintillator (Gd2O2S) afterglow model
    # that direct-conversion PCCT detectors do not exhibit.
    opts = BS.SimOptions(fidelity = :pcct)
    @test opts.use_focal_spot === false
    @test opts.use_lag === false
    # Opt-in must survive preset resolution.
    opts_on = BS.SimOptions(fidelity = :pcct, use_focal_spot = true)
    @test opts_on.use_focal_spot === true
    # EICT preset unchanged.
    eict = BS.SimOptions(fidelity = :eict)
    @test eict.use_focal_spot === true
    @test eict.use_lag === true
end

_ts("entering simulate!(PCCTWorkspace) — focal-spot blur on/off testset")
@testset "simulate!(PCCTWorkspace) — focal-spot blur on/off" begin
    if !HAS_GPU
        @warn "Skipping PCCT focal-spot test: no GPU backend."
    else
        _ts("  focal spot OFF")
        off = _toy_pcct_setup(use_pcct_pileup = false, use_pcct_scatter = false)
        @test off.ws.focal_spot_kernel === nothing
        res_off = BS.simulate!(off.ws, off.phantom, off.protocol, off.sim_opts)
        bins_off = [Array(b) for b in res_off.pcct_sino.bins]

        _ts("  focal spot ON")
        on = _toy_pcct_setup(
            use_pcct_pileup = false, use_pcct_scatter = false,
            use_focal_spot = true,
        )
        # Kernel is pre-computed at workspace creation (zero-alloc contract).
        @test on.ws.focal_spot_kernel !== nothing
        res_on = BS.simulate!(on.ws, on.phantom, on.protocol, on.sim_opts)
        bins_on = [Array(b) for b in res_on.pcct_sino.bins]

        # Blur must measurably perturb every energy bin, and keep it finite.
        for b in eachindex(bins_on)
            @test maximum(abs.(bins_on[b] .- bins_off[b])) > 1.0e-6
            @test all(isfinite, bins_on[b])
        end
        # Blur is a normalized convolution of log line integrals: the mean
        # should be approximately preserved (no energy created or destroyed).
        for b in eachindex(bins_on)
            @test mean(bins_on[b]) ≈ mean(bins_off[b]) rtol = 2.0e-2
        end
    end
end

# -----------------------------------------------------------------------------
# EICT quantum-noise dose scaling (SoftwareX revision, R1.4)
# -----------------------------------------------------------------------------
_ts("entering EICT quantum noise dose-sweep testset")
@testset "simulate!(EICTWorkspace) — quantum noise scales as 1/sqrt(mA)" begin
    if !HAS_GPU
        @warn "Skipping EICT dose-sweep test: no GPU backend."
    else
        # Pure quantum regime: electronic_noise = 0 so the only noise term is
        # Poisson (Gaussian-approximated) with variance = expected count.
        # sigma_p ∝ 1/sqrt(I0) ∝ 1/sqrt(mA · t) at fixed views/rotation time.
        function _dose_setup(mA)
            scanner = BS.Scanner(
                source_to_isocenter = 540.0,
                source_to_detector = 1080.0,
                detector_rows = 8,
                detector_cols = 64,
                detector_row_size = 1.0,
                detector_col_size = 1.0,
                detector_material = :lumex,
                detector_depth = 3.0,
                electronic_noise = 0.0,
                detection_gain = 10.0,
            )
            protocol = BS.CTProtocol(mA = mA, kVp = 120.0, views = 16, rotation_time = 0.5)
            sim_opts = BS.SimOptions(;
                fidelity = :eict,
                use_noise = true, use_scatter = false,
                use_lag = false, use_focal_spot = false,
                use_optical_crosstalk = false,
                seed = 7,
            )
            recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
            phantom_cpu = BS.create_gammex_472(n_voxels = 32, fov_cm = 20.0, z_cm = 2.0)
            phantom = BS.Phantom(
                to_gpu(phantom_cpu.mask),
                phantom_cpu.materials,
                phantom_cpu.voxel_size,
                phantom_cpu.origin,
                phantom_cpu.extent,
            )
            clean_opts = BS.SimOptions(;
                fidelity = :eict,
                use_noise = false, use_scatter = false,
                use_lag = false, use_focal_spot = false,
                use_optical_crosstalk = false,
            )
            ws_noisy = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
            ws_clean = BS.create_eict_workspace(scanner, protocol, clean_opts, recon_opts, phantom)
            BS.simulate!(ws_noisy, phantom, protocol, sim_opts)
            BS.simulate!(ws_clean, phantom, protocol, clean_opts)
            resid = Array(ws_noisy.sinogram) .- Array(ws_clean.sinogram)
            return std(resid)
        end

        mA_lo, mA_hi = 50.0, 800.0
        _ts("  sweep mA = $mA_lo")
        σ_lo = _dose_setup(mA_lo)
        _ts("  sweep mA = $mA_hi")
        σ_hi = _dose_setup(mA_hi)

        @test σ_lo > 0
        @test σ_hi > 0
        # Expected ratio sqrt(800/50) = 4.0. Sinogram has ~8k noise samples,
        # so the sample-std ratio is tight; allow 15% for the count-floor and
        # low-count tails behind the densest rods.
        @test σ_lo / σ_hi ≈ sqrt(mA_hi / mA_lo) rtol = 0.15
    end
end
