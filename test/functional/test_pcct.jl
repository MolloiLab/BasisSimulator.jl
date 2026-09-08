# =============================================================================
# test/functional/test_pcct.jl — parity of the functional PCCT chain vs legacy
#
# Standalone script:  julia --project=. -t 2 test/functional/test_pcct.jl
# Oracles: `dd_fast_fused_spectral_project!`, `apply_pcct_noise!`, the driver's
# MC pile-up register code, `apply_pcct_pileup_correction!`,
# `combine_pcct_bin_counts!`, and `simulate!(::PCCTWorkspace)` on the toy
# PCCT setup of test/api.jl (CPU, bf = 1, scatter off, focal spot off).
# =============================================================================
module TestFunctionalPCCT   # include-safe: no top-level names leak into the including scope

using Test, BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics, Random
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "src", "functional", "pcct.jl"))
end
const F = FStage

_t0 = time()
_ts(msg) = println("[test_pcct] +", lpad(round(time() - _t0; digits = 1), 6), " s  ", msg)

# -----------------------------------------------------------------------------
# Fixtures (test/projection.jl:246-268 — 64-col, 3-material, N_E = 8, K = 8)
# -----------------------------------------------------------------------------
function _toy_proj_geom(; n_cols = 16, n_rows = 4, n_angles = 4, fov_cm = 5.0)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = n_rows, detector_cols = n_cols,
        detector_row_size = 1.0, detector_col_size = 1.0,
    )
    return BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = fov_cm, z_cm = 1.0)
end

const GEOM = _toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
const N_E = 8
const MASK = let nx = 64, ny = 64, nz = 8
    mask = zeros(UInt16, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - 32.5)^2 + (j - 32.5)^2 <= (0.6 * 32)^2
            mask[i, j, k] = UInt16(1)
        end
        if (i - 45.5)^2 + (j - 32.5)^2 <= 3.0^2
            mask[i, j, k] = UInt16(2)
        end
    end
    mask
end
const μ_TABLE = let μ = zeros(Float32, 3, N_E)
    for e in 1:N_E
        μ[2, e] = 0.30f0 - 0.12f0 * (e - 1) / (N_E - 1)
        μ[3, e] = 1.50f0 - 0.90f0 * (e - 1) / (N_E - 1)
    end
    μ
end
const Wη = fill(Float32(1 / N_E), N_E)
const W2 = let W = zeros(Float32, N_E, 2)
    for e in 1:N_E
        W[e, 1] = Wη[e] * (1.0f0 - (e - 1) / (N_E - 1))
        W[e, 2] = Wη[e] * ((e - 1) / (N_E - 1))
    end
    W
end

# Legacy spectral kernel → (n_col, n_row, n_view, n_bins)
function _legacy_spectral(mask, geom, μ_table, W; volume_extent = nothing, bowtie = nothing)
    n_bins = size(W, 2)
    K = size(W, 1)
    pilot = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    out = zeros(Float32, length(pilot) * n_bins)
    BS.dd_fast_fused_spectral_project!(
        pilot, out, Int32(n_bins), mask, geom, μ_table, W, Val(K), Int32(1);
        volume_extent = volume_extent, ws_bowtie_spectral = bowtie)
    return reshape(out, geom.n_cols, geom.n_rows, geom.n_angles, n_bins)
end

# Scalar Float64 reference of Σ_e W[e,b]·bt·exp(-Σ_m μ[m,e]·P_m)
function _direct_spectral(P::Array{Float64, 4}, μ::Matrix{Float64}, W::Matrix{Float64}, bt)
    n_col, n_row, n_view, n_mat = size(P)
    n_E, n_bins = size(W)
    I = zeros(Float64, n_col, n_row, n_view, n_bins)
    for b in 1:n_bins, v in 1:n_view, r in 1:n_row, c in 1:n_col
        acc = 0.0
        for e in 1:n_E
            L = 0.0
            for m in 1:n_mat
                L += P[c, r, v, m] * μ[m, e]
            end
            tr = exp(-L)
            if bt !== nothing
                tr = min(tr, 1.0e30) * bt[c, r, e]
            end
            acc += W[e, b] * tr
        end
        I[c, r, v, b] = acc
    end
    return I
end

# The driver's MC pile-up register code (src/api/driver.jl), verbatim semantics.
function _driver_pileup!(bins::Vector{Array{T, 3}}, I0_bins, S) where {T}
    eps = T(1.0e-10)
    b1, b2, b3, b4 = bins
    I0_t1, I0_t2, I0_t3, I0_t4 = T(I0_bins[1]), T(I0_bins[2]), T(I0_bins[3]), T(I0_bins[4])
    S11 = T(S[1, 1])
    S21 = T(S[2, 1]); S22 = T(S[2, 2])
    S31 = T(S[3, 1]); S32 = T(S[3, 2]); S33 = T(S[3, 3])
    S41 = T(S[4, 1]); S42 = T(S[4, 2]); S43 = T(S[4, 3]); S44 = T(S[4, 4])
    for idx in eachindex(b1)
        c1 = I0_t1 * exp(-b1[idx]); c2 = I0_t2 * exp(-b2[idx])
        c3 = I0_t3 * exp(-b3[idx]); c4 = I0_t4 * exp(-b4[idx])
        r1 = S11 * c1
        r2 = S21 * c1 + S22 * c2
        r3 = S31 * c1 + S32 * c2 + S33 * c3
        r4 = S41 * c1 + S42 * c2 + S43 * c3 + S44 * c4
        b1[idx] = -log(max(r1, eps) / I0_t1)
        b2[idx] = -log(max(r2, eps) / I0_t2)
        b3[idx] = -log(max(r3, eps) / I0_t3)
        b4[idx] = -log(max(r4, eps) / I0_t4)
    end
    return bins
end

# The driver's combine (scatter step): comb = Σ_b I0_b·exp(-b_b); -log(max(comb,eps)/I0_total)
function _driver_combine(bins::Vector{Array{T, 3}}, I0_bins) where {T}
    comb = zeros(T, size(bins[1]))
    for (b, bs) in enumerate(bins)
        I0b = T(I0_bins[b])
        for idx in eachindex(bs)
            comb[idx] += I0b * exp(-bs[idx])
        end
    end
    I0t = T(sum(I0_bins)); eps = T(1.0e-10)
    for idx in eachindex(comb)
        comb[idx] = -log(max(comb[idx], eps) / I0t)
    end
    return comb
end

_unstack(A::Array{T, 4}) where {T} = [copy(A[:, :, :, b]) for b in 1:size(A, 4)]
_maxabs(a, b) = maximum(abs.(a .- b))
_maxrel(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps(eltype(b))))

# test/api.jl:302-341 toy PCCT setup (CPU; `use_pcct_scatter = false` is the
# scope of the functional chain — the :pcct preset would otherwise inject
# scatter through `config.scatter` even with `use_scatter = false`).
function _toy_pcct_setup(; use_pcct_pileup = true, kwargs...)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 64,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_type = :photon_counting, detector_material = :CdTe,
        detector_depth = 1.6, n_energy_bins = 4,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0], dead_time_ns = 25.0,
    )
    protocol = BS.CTProtocol(mA = 2.5, kVp = 120.0, views = 16, rotation_time = 0.5)
    sim_opts = BS.SimOptions(;
        fidelity = :pcct,
        use_noise = false, use_scatter = false, use_lag = false,
        use_focal_spot = false, use_optical_crosstalk = false,
        use_pcct_pileup = use_pcct_pileup, use_pcct_scatter = false,
        kwargs...
    )
    recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
    phantom = BS.create_gammex_472(n_voxels = 32, fov_cm = 20.0, z_cm = 2.0)
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    return (; scanner, protocol, sim_opts, recon_opts, phantom, ws)
end
_noise_opts(s) = BS.SimOptions(;
    fidelity = :pcct, use_noise = true, use_scatter = false, use_lag = false,
    use_focal_spot = false, use_optical_crosstalk = false,
    use_pcct_pileup = s.sim_opts.use_pcct_pileup, use_pcct_scatter = false)

# =============================================================================
@testset "Functional PCCT chain" begin
    P32 = F.material_path_lengths(MASK, 3, GEOM)
    @test size(P32) == (64, 8, 8, 3)
    _ts("fixture path lengths built")

    @testset "spectral bins ≡ dd_fast_fused_spectral_project! (single tile, 2 bins, K=8)" begin
        I_legacy = _legacy_spectral(MASK, GEOM, μ_TABLE, W2)
        I_f = F.spectral_bin_intensities(P32, μ_TABLE, W2)
        @test size(I_f) == size(I_legacy)
        @test eltype(I_f) == Float32
        Δ = _maxabs(I_f, I_legacy)
        println("  spectral bins vs legacy kernel: max abs = ", Δ)
        @test Δ < 1.0f-4
        # chunked views are the same program (BLAS may reassociate per chunk size)
        I_c = F.spectral_bin_intensities(P32, μ_TABLE, W2; view_chunks = 3)
        println("  view_chunks = 3 vs 1: max rel = ", _maxrel(I_c, I_f), (I_c == I_f ? " (bit-identical)" : ""))
        @test _maxrel(I_c, I_f) < 1.0e-6
        # purity
        P_copy = copy(P32)
        F.spectral_bin_intensities(P32, μ_TABLE, W2; view_chunks = 2)
        @test P32 == P_copy
    end

    @testset "spectral bins + spectral bowtie ≡ legacy (arc helical, test/projection.jl:340)" begin
        scanner_h = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 4, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
            detector_shape = :arc,
        )
        geom_h = BS.CTGeometry(scanner_h; n_angles = 16, fov_cm = 10.0,
            z_cm = 1.0, pitch = 1.0, n_rotations = 2.0)
        mask_h = MASK[17:48, 17:48, 3:6]
        bowtie = Array{Float32}(undef, geom_h.n_cols, geom_h.n_rows, N_E)
        for e in 1:N_E, r in 1:geom_h.n_rows, c in 1:geom_h.n_cols
            bowtie[c, r, e] = 0.7f0 + 0.25f0 * (c - 1) / (geom_h.n_cols - 1) +
                0.03f0 * (e - 1) / (N_E - 1)
        end
        ext = (10.0, 10.0, 1.0)
        P_h = F.material_path_lengths(mask_h, 3, geom_h; volume_extent = ext)
        I_legacy = _legacy_spectral(mask_h, geom_h, μ_TABLE, W2; volume_extent = ext, bowtie = bowtie)
        I_f = F.spectral_bin_intensities(P_h, μ_TABLE, W2, bowtie; view_chunks = 4)
        Δ = _maxabs(I_f, I_legacy)
        println("  spectral bins + bowtie vs legacy kernel: max abs = ", Δ)
        @test Δ < 1.0f-4
        # Float64 direct reference with bowtie
        P64 = Float64.(P_h); μ64 = Float64.(μ_TABLE); W64 = Float64.(W2); bt64 = Float64.(bowtie)
        I64 = F.spectral_bin_intensities(P64, μ64, W64, bt64; view_chunks = 2)
        I_ref = _direct_spectral(P64, μ64, W64, bt64)
        Δ64 = _maxabs(I64, I_ref)
        println("  Float64 direct reference (bowtie): max abs = ", Δ64)
        @test Δ64 < 1.0e-10
    end

    @testset "Float64 direct reference ≤ 1e-10" begin
        P64 = Float64.(P32); μ64 = Float64.(μ_TABLE); W64 = Float64.(W2)
        I64 = F.spectral_bin_intensities(P64, μ64, W64)
        I_ref = _direct_spectral(P64, μ64, W64, nothing)
        Δ64 = _maxabs(I64, I_ref)
        println("  Float64 direct reference: max abs = ", Δ64)
        @test Δ64 < 1.0e-10
        # log step matches the legacy formula bit-for-bit
        I0 = Float32[sum(W2[:, b]) for b in 1:2]
        p = F.bin_log_sinograms(Float32.(I64), I0, 1.0f-10)
        p_ref = similar(p)
        for b in 1:2, idx in CartesianIndices(size(p)[1:3])
            p_ref[idx, b] = -log(max(Float32(I64[idx, b]), 1.0f-10) / I0[b])
        end
        @test p == p_ref
    end
    _ts("spectral parity done")

    @testset "closed-form VJP vs finite differences (Float64)" begin
        rng = MersenneTwister(11)
        n_col, n_row, n_view, n_mat, n_E, n_bins = 3, 2, 4, 3, 5, 2
        P = 2.0 .* rand(rng, n_col, n_row, n_view, n_mat)
        μ = 0.5 .* rand(rng, n_mat, n_E)
        W = rand(rng, n_E, n_bins) .+ 0.1
        bt = 0.6 .+ 0.4 .* rand(rng, n_col, n_row, n_E)
        I0 = [sum(W[:, b]) for b in 1:n_bins]
        C = randn(rng, n_col, n_row, n_view, n_bins)
        eps64 = 1.0e-10
        for (label, bt_arg, chunks) in (("no bowtie", nothing, 1), ("bowtie", bt, 2))
            loss(Px) = sum(C .* F.bin_log_sinograms(
                F.spectral_bin_intensities(Px, μ, W, bt_arg; view_chunks = chunks), I0, eps64))
            I = F.spectral_bin_intensities(P, μ, W, bt_arg; view_chunks = chunks)
            Ibar = F.bin_log_sinograms_vjp(I, I0, eps64, C)
            Pbar = F.spectral_bins_vjp(P, μ, W, bt_arg, Ibar; view_chunks = chunks)
            @test size(Pbar) == size(P)
            Pbar_fd = similar(P)
            h = 1.0e-6
            for idx in eachindex(P)
                Pp = copy(P); Pp[idx] += h
                Pm = copy(P); Pm[idx] -= h
                Pbar_fd[idx] = (loss(Pp) - loss(Pm)) / (2h)
            end
            rel = maximum(abs.(Pbar .- Pbar_fd)) / maximum(abs.(Pbar_fd))
            println("  VJP vs FD (", label, "): max rel = ", rel)
            @test rel < 1.0e-6
        end
    end
    _ts("VJP done")

    @testset "pile-up apply / correct ≡ driver register code (synthetic)" begin
        rng = MersenneTwister(3)
        n_col, n_row, n_view = 5, 3, 4
        I0 = [2.8e5, 2.5e5, 1.1e5, 5.8e4]
        # lower-triangular MC-like S + DELIBERATE upper-triangle garbage the driver ignores
        S = [0.21 0.0 0.0 0.0;
             0.0074 0.211 0.0 0.0;
             0.037 0.0085 0.211 0.0;
             0.247 0.283 0.291 0.502]
        S_dirty = copy(S); S_dirty[1, 2] = 0.3; S_dirty[2, 4] = 0.1
        for (T, tol_apply, tol_corr) in ((Float32, 1.0f-6, 2.0f-5), (Float64, 1.0e-12, 1.0e-10))
            p = T.(4.0 .* rand(rng, n_col, n_row, n_view, 4))
            plan = F.pcct_plan(zeros(T, 2, 3), zeros(T, 3, 4), I0, S_dirty; T = T)
            @test plan.pileup_St == permutedims(T.(LinearAlgebra.tril(S)))
            out = F.pileup_apply(p, plan.I0_bins, plan.pileup_St, plan.eps)
            ref = _driver_pileup!(_unstack(p), I0, S)
            Δ = _maxabs(out, F.stack_bins(ref))
            println("  pile-up apply (", T, "): max abs = ", Δ)
            @test Δ <= tol_apply
            @test size(out) == size(p)
            # correction ≡ apply_pcct_pileup_correction! and inverts apply
            corr = F.pileup_correct(out, plan.I0_bins, plan.pileup_Sinv_t, plan.eps)
            ref_corr = BS.apply_pcct_pileup_correction!(_unstack(F.stack_bins(ref)), I0, S)
            Δc = _maxabs(corr, F.stack_bins(ref_corr))
            println("  pile-up correct (", T, "): max abs vs legacy = ", Δc,
                    ", round-trip max abs = ", _maxabs(corr, p))
            @test Δc <= tol_corr
            @test _maxabs(corr, p) <= tol_corr
        end
    end

    @testset "bin combine ≡ combine_pcct_bin_counts! and driver combine (synthetic)" begin
        rng = MersenneTwister(5)
        I0 = [2.8e5, 2.5e5, 1.1e5, 5.8e4]
        p = Float32.(6.0 .* rand(rng, 6, 2, 3, 4))
        groups = [[1, 2, 3], [4]]
        G, I0g = F.combine_matrix(I0, groups, Float32)
        @test I0g == Float32[Float32(sum(I0[1:3])), Float32(I0[4])]
        counts = F.combine_bin_counts(p, G)
        out_bins = [zeros(Float32, 6, 2, 3) for _ in groups]
        I0_leg = BS.combine_pcct_bin_counts!(out_bins, _unstack(p), I0, groups)
        @test I0_leg ≈ Float64.(I0g)
        Δ = _maxrel(counts, F.stack_bins(out_bins))
        println("  combine counts vs combine_pcct_bin_counts!: max rel = ", Δ)
        @test Δ < 1.0e-6
        # driver combine = single group of all bins, log domain
        q = F.combine_bins(p, I0, [[1, 2, 3, 4]], Float32)
        q_ref = _driver_combine(_unstack(p), I0)
        Δq = _maxabs(q[:, :, :, 1], q_ref)
        println("  combine log vs driver: max abs = ", Δq)
        @test Δq < 1.0e-6
        # Float64 exactness of the algebra
        p64 = Float64.(p)
        q64 = F.combine_bins(p64, I0, [[1, 2, 3, 4]], Float64)
        @test _maxabs(q64[:, :, :, 1], _driver_combine(_unstack(p64), I0)) < 1.0e-12
    end

    @testset "noise: draw_pcct_counts + pcct_counts_from_input ≡ apply_pcct_noise!" begin
        rng = MersenneTwister(9)
        I0 = [2.8e5, 2.5e5, 1.1e5, 5.8e4]
        p = Float32.(rand(rng, 8, 3, 4, 4) .* 12.0)     # λ from ~1.7e-0 to 2.8e5 → both sampler branches
        p[1, 1, 1, 1] = 40.0f0                          # λ < 1e-10 → N = 0 branch, floor at 1
        for nr in (0.0, 0.3)
            sino = BS.EnergyResolvedSinogram(_unstack(p), Float32[20, 35, 55, 70])
            raw_out = [similar(b) for b in sino.bins]
            BS.apply_pcct_noise!(sino, I0; seed = 123, noise_reduction = nr, raw_out = raw_out)
            N = F.draw_pcct_counts(p, I0, 123)
            I0_T = Float32.(I0)
            p_noisy, raw = F.pcct_counts_from_input(p, I0_T, N, Float32(nr))
            if nr == 0.0
                @test raw === N
                @test N == F.stack_bins(raw_out)           # bit-exact draws
                @test all(isinteger, N)
                @test any(iszero, N)
            else
                @test _maxrel(raw, F.stack_bins(raw_out)) < 1.0e-5
            end
            Δ = _maxabs(p_noisy, F.stack_bins(sino.bins))
            println("  noise map (nr = ", nr, "): max abs = ", Δ)
            @test Δ <= 2.0e-6          # ≤ 2 ulp(Float32) at p ≈ 12.6: legacy logs in Float64 then rounds
            @test !any(isnan, p_noisy)
        end
        # surrogate: implied ε reproduces the draw; gradient path is smooth
        N = F.draw_pcct_counts(p, I0, 123)
        λ = F.pcct_expected_counts(p, Float32.(I0))
        ε = F.implied_noise_eps(N, λ)
        N_sur = F.pcct_noise_surrogate(p, Float32.(I0), ε)
        @test _maxabs(N_sur, N) < 0.1f0          # ≤ ulp(λ) rounding of λ + √λ·ε
        ε0 = zeros(Float32, size(p))
        @test F.pcct_noise_surrogate(p, Float32.(I0), ε0) == λ
    end
    _ts("kernels done")

    # -------------------------------------------------------------------------
    # Full chain vs simulate!(PCCTWorkspace)
    # -------------------------------------------------------------------------
    @testset "full chain ≡ simulate!(PCCTWorkspace)" begin
        # ONE workspace (pile-up off at construction — the 5000-trial MC pile-up
        # matrix `create_workspace` builds costs ~25 s).  For the pile-up oracle
        # the SAME driver code path is exercised by injecting a 50-trial MC `S`
        # (`compute_mc_pileup_matrix`, the workspace's own routine) into the
        # mutable workspace fields — parity of the driver algebra does not
        # depend on how many trials shaped S.
        s = _toy_pcct_setup(use_pcct_pileup = false)
        _ts("toy workspace built")
        ws = s.ws
        n_mat = length(s.phantom.materials)
        @test size(ws.μ_table, 1) == n_mat
        P = F.material_path_lengths(s.phantom.mask, n_mat, ws.geom; volume_extent = s.phantom.extent)
        @test size(P) == (64, 8, 16, n_mat)
        plan = F.pcct_plan(ws)
        @test plan.pileup_St === nothing
        @test plan.I0_bins == Float32.(ws.I0_bins)
        @test size(plan.W) == (length(ws.energies), 4)

        # ---- noise OFF, pile-up OFF ----
        res = BS.simulate!(ws, s.phantom, s.protocol, s.sim_opts)
        _ts("legacy simulate! (noise off) done")
        @test propertynames(res) == (:pcct_sino, :I0_bins, :pileup_S, :raw_counts)
        @test res.pileup_S === nothing
        bins_leg = F.stack_bins(res.pcct_sino.bins)     # copy — ws.bins are reused by the next call
        raw_leg = F.stack_bins(res.raw_counts)
        chain = F.pcct_chain(P, plan)
        _ts("functional chain (noise off) done")
        @test propertynames(chain) == (:bins, :raw_counts)
        @test size(chain.bins) == (64, 8, 16, 4)
        Δb = _maxabs(chain.bins, bins_leg)
        Δr = _maxrel(chain.raw_counts, raw_leg)
        println("  chain (noise off, pile-up off): bins max abs = ", Δb, ", raw counts (λ) max rel = ", Δr)
        @test Δb < 1.0e-4
        @test Δr < 2.0e-4
        plan2 = F.pcct_plan(ws; view_chunks = 4)
        @test _maxabs(F.pcct_chain(P, plan2).bins, chain.bins) < 1.0e-5
        # purity of the chain w.r.t. its inputs
        @test P == F.material_path_lengths(s.phantom.mask, n_mat, ws.geom; volume_extent = s.phantom.extent)

        # ---- noise ON (seed 42 = SimOptions default), pile-up OFF ----
        so_n = _noise_opts(s)
        res_n = BS.simulate!(ws, s.phantom, s.protocol, so_n)
        bins_leg_n = F.stack_bins(res_n.pcct_sino.bins)
        raw_leg_n = F.stack_bins(res_n.raw_counts)
        _ts("legacy simulate! (noise on) done")
        # (a) host sampler on the LEGACY noise-free bins: bit-for-bit the legacy draws
        N_leg = F.draw_pcct_counts(bins_leg, ws.I0_bins, so_n.seed)
        println("  sampler on legacy bins: bit-identical draws = ", count(N_leg .== raw_leg_n), " / ", length(N_leg))
        @test N_leg == raw_leg_n
        @test all(isinteger, N_leg)
        # (b) deterministic remainder on the legacy draws
        p_n, raw_n = F.pcct_counts_from_input(bins_leg, plan.I0_bins, N_leg, plan.noise_reduction)
        Δn = _maxabs(p_n, bins_leg_n)
        println("  noise map on legacy draws: max abs = ", Δn)
        @test Δn <= 1.0e-6
        @test raw_n === N_leg
        # (c) chain from P with the legacy draws as the input tensor
        chain_n = F.pcct_chain(P, plan, N_leg)
        Δbn = _maxabs(chain_n.bins, bins_leg_n)
        println("  chain (noise on, legacy draws): bins max abs = ", Δbn)
        @test Δbn < 1.0e-4
        @test chain_n.raw_counts === N_leg
        # (d) chain from P with OUR draws: λ differs from legacy by float ordering
        #     (~1e-6 rel), so the RNG stream is not reproducible bit-for-bit for
        #     every ray — statistics must agree within 2 %
        N_mine = F.draw_pcct_counts(chain.bins, plan.I0_bins_f64, so_n.seed)
        println("  chain-drawn counts: bit-identical fraction = ",
                round(count(N_mine .== raw_leg_n) / length(N_mine); digits = 4))
        for b in 1:4
            m_leg = mean(raw_leg_n[:, :, :, b]); m_mine = mean(N_mine[:, :, :, b])
            v_leg = var(raw_leg_n[:, :, :, b]); v_mine = var(N_mine[:, :, :, b])
            println("    bin ", b, ": mean ", round(m_mine; digits = 1), " vs ", round(m_leg; digits = 1),
                    "  var ", round(v_mine; digits = 1), " vs ", round(v_leg; digits = 1))
            @test abs(m_mine - m_leg) / m_leg < 0.02
            @test abs(v_mine - v_leg) / v_leg < 0.02
        end
        λ_mine = F.pcct_expected_counts(chain.bins, plan.I0_bins)
        for b in 1:4
            resid = (N_mine[:, :, :, b] .- λ_mine[:, :, :, b]) ./ sqrt.(λ_mine[:, :, :, b])
            @test abs(mean(resid)) < 0.05
            @test abs(var(resid) - 1) < 0.1
        end
        # (e) straight-through surrogate reproduces the realized draw
        ε = F.implied_noise_eps(N_leg, F.pcct_expected_counts(bins_leg, plan.I0_bins))
        N_sur = F.pcct_noise_surrogate(bins_leg, plan.I0_bins, ε)
        @test _maxabs(N_sur, N_leg) < 0.1f0
        @test _maxabs(F.pcct_counts_from_input(bins_leg, plan.I0_bins, N_sur, plan.noise_reduction)[1], bins_leg_n) < 1.0e-4
        _ts("chain (pile-up off) done")

        # ---- pile-up ON (MC S injected into the workspace; driver code path) ----
        det = ws.pcct_detector
        w_norm = ws.weights ./ sum(ws.weights)
        rate = BS.compute_detector_I0(ws.geom, s.protocol, sum(ws.weights)) /
            (s.protocol.rotation_time / s.protocol.views)
        S = BS.compute_mc_pileup_matrix(det.energy_thresholds_keV, w_norm, ws.energies, rate,
            Float64(det.dead_time_ns); n_trials = 50, seed = 42)
        @test size(S) == (4, 4)
        @test all(S[i, j] == 0 for i in 1:4 for j in (i + 1):4)   # MC S is lower-triangular
        ws.pileup_S = S
        ws.use_pcct_pileup = true
        so_on = BS.SimOptions(; fidelity = :pcct, use_noise = false, use_scatter = false, use_lag = false,
            use_focal_spot = false, use_optical_crosstalk = false, use_pcct_pileup = true, use_pcct_scatter = false)
        plan_on = F.pcct_plan(ws)
        @test plan_on.pileup_St == permutedims(Float32.(LinearAlgebra.tril(S)))

        res_on = BS.simulate!(ws, s.phantom, s.protocol, so_on)
        @test res_on.pileup_S === S
        bins_leg_on = F.stack_bins(res_on.pcct_sino.bins)
        raw_leg_on = F.stack_bins(res_on.raw_counts)
        chain_on = F.pcct_chain(P, plan_on)
        Δb = _maxabs(chain_on.bins, bins_leg_on)
        Δr = _maxrel(chain_on.raw_counts, raw_leg_on)
        println("  chain (noise off, pile-up on): bins max abs = ", Δb, ", recorded counts max rel = ", Δr)
        @test Δb < 1.0e-4
        @test Δr < 2.0e-4
        # pile-up correction inverts the apply, and matches apply_pcct_pileup_correction!
        corr = F.pileup_correct(chain_on.bins, plan_on.I0_bins, plan_on.pileup_Sinv_t, plan_on.eps)
        @test _maxabs(corr, chain.bins) < 1.0e-3
        leg_corr = BS.apply_pcct_pileup_correction!(_unstack(bins_leg_on), ws.I0_bins, S)
        @test _maxabs(corr, F.stack_bins(leg_corr)) < 1.0e-3

        # noise ON + pile-up ON: the legacy draw happens on the un-piled bins
        # (identical to the pile-up-off noise-free bins), THEN S mixes the
        # floored counts.  The legacy draws fed to the chain must reproduce the
        # legacy bins and recorded counts deterministically.
        so_on_n = BS.SimOptions(; fidelity = :pcct, use_noise = true, use_scatter = false, use_lag = false,
            use_focal_spot = false, use_optical_crosstalk = false, use_pcct_pileup = true, use_pcct_scatter = false)
        res_on_n = BS.simulate!(ws, s.phantom, s.protocol, so_on_n)
        bins_leg_on_n = F.stack_bins(res_on_n.pcct_sino.bins)
        raw_leg_on_n = F.stack_bins(res_on_n.raw_counts)          # S × floored counts (fractional)
        chain_on_n = F.pcct_chain(P, plan_on, N_leg)
        Δbn = _maxabs(chain_on_n.bins, bins_leg_on_n)
        Δrn = _maxrel(chain_on_n.raw_counts, raw_leg_on_n)
        println("  chain (noise on, pile-up on, legacy draws): bins max abs = ", Δbn,
                ", recorded counts max rel = ", Δrn)
        @test Δbn < 1.0e-4
        @test Δrn < 2.0e-4
        @test all(isfinite, chain_on_n.bins)

        # combined chain = chain + combine_bins (driver combine on the same bins)
        G, I0g = F.combine_matrix(plan_on.I0_bins_f64, [[1, 2, 3], [4]], Float32)
        q = F.pcct_chain_combined(P, plan_on, G, I0g)
        @test size(q) == (64, 8, 16, 2)
        @test q == F.combine_bins(chain_on.bins, G, I0g, plan_on.eps)
        q_all = F.combine_bins(chain_on.bins, plan_on.I0_bins_f64, [[1, 2, 3, 4]], Float32)
        @test _maxabs(q_all[:, :, :, 1], _driver_combine(_unstack(bins_leg_on), ws.I0_bins)) < 1.0e-4
        _ts("chain (pile-up on) done")
    end
end

end # module TestFunctionalPCCT
