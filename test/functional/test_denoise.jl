# =============================================================================
# Oracle tests for the functional denoisers (src/functional/denoise.jl).
#
# Standalone:   julia --project=. -t 2 test/functional/test_denoise.jl
# From runtests: the file is self-contained (it builds its own `FStage` module).
#
# Every stage is compared against the legacy in-place implementation in
# src/denoising/ on synthetic sinograms / volumes that mimic the notebook usage
# (2 channels, (n_col, n_row, n_view) = (96, 4, 48), smooth object + Poisson-
# like noise).  Tolerances are those in the stage brief:
#   SVD-bilateral ≤ 1e-4 rel, ACNR-Kalender ≤ 1e-5 rel, median-z exact on
#   interior slices (boundary documented), SF-JSD ≤ 1e-4 rel with captured
#   constants.  A Float64 finite-difference smoothness/gradient check covers the
#   closed-form eigen path of the SVD-bilateral denoiser.
#
# The whole file lives in its own module so it can be `include`d from
# test/functional/runtests.jl without colliding with other test helpers in Main;
# nested @testsets still report into the enclosing testset.
# =============================================================================
module FunctionalDenoiseTests
using Test, Random, LinearAlgebra, Statistics, Logging
using BasisSimulator
const BS = BasisSimulator

module FStage
using BasisSimulator, LinearAlgebra, Statistics, FFTW
const BS = BasisSimulator
include(joinpath(@__DIR__, "..", "..", "src", "functional", "denoise.jl"))
end

# quiet the legacy @info chatter
const _quiet = Logging.SimpleLogger(stderr, Logging.Warn)

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
_relmax(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / maximum(abs.(Float64.(b)))

# Two (or three) co-registered log sinograms: smooth object (a broad blob and a
# thin rod moving with view angle, row-dependent scale) with different spectral
# weights per channel (rank ≥ 2), plus Gaussian noise of the log-Poisson
# variance 1/(I0·e^{-p}).
function synth_channels(::Type{T} = Float32; n_col = 96, n_row = 4, n_view = 48, seed = 7,
        I0 = (1.0e5, 6.0e4, 8.0e4, 5.0e4), n_ch = 2) where {T}
    rng = MersenneTwister(seed)
    cs = range(-1, 1; length = n_col); vs = range(0, 2π; length = n_view)
    wts = ((1.3, 0.9), (0.8, 0.35), (1.0, 0.7), (0.6, 0.5))
    out = ntuple(n_ch) do b
        p = Array{Float64}(undef, n_col, n_row, n_view)
        for (iv, v) in enumerate(vs), r in 1:n_row, (ic, c) in enumerate(cs)
            s = 1 + 0.06 * (r - 1)
            e1 = exp(-(c - 0.35 * cos(v))^2 / 0.08)
            e2 = exp(-(c + 0.4 * sin(v))^2 / 0.01)
            p[ic, r, iv] = s * (wts[b][1] * e1 + wts[b][2] * e2)
        end
        p .+= randn(rng, size(p)) ./ sqrt.(I0[b] .* exp.(-p))
        T.(p)
    end
    return out
end

# Water / iodine basis volumes with anti-correlated noise (ACNR fixture).
function synth_basis_pair(::Type{T} = Float32; n = 48, nz = 3, seed = 11) where {T}
    rng = MersenneTwister(seed)
    xs = range(-1, 1; length = n)
    W = [Float64((x^2 + y^2 < 0.6^2)) + 0.3 * ((x - 0.2)^2 + (y + 0.1)^2 < 0.15^2) for x in xs, y in xs, z in 1:nz]
    I = [2.0 * ((x - 0.2)^2 + (y + 0.1)^2 < 0.15^2) + 0.5 * ((x + 0.3)^2 + (y - 0.3)^2 < 0.1^2) for x in xs, y in xs, z in 1:nz]
    n1 = randn(rng, n, n, nz); n2 = randn(rng, n, n, nz)
    W .+= 0.03 .* n1 .+ 0.01 .* n2
    I .-= 0.06 .* n1 .- 0.02 .* n2
    return T.(W), T.(I)
end

# ----------------------------------------------------------------------------
# 0. Gram eigendecomposition vs legacy svd (closed form N=2, Jacobi N=3,4)
# ----------------------------------------------------------------------------
@testset "gram_eigen vs LinearAlgebra.svd" begin
    for (T, tol_s, tol_v) in ((Float64, 1e-10, 1e-8), (Float32, 2e-3, 2e-3)), N in (2, 3, 4)
        chs = synth_channels(T; n_col = 24, n_row = 3, n_view = 16, n_ch = N)
        cvr = map(c -> permutedims(c, (1, 3, 2)), chs)
        λ, V = FStage.gram_eigen(cvr)
        for row in 1:3
            M = hcat(ntuple(b -> vec(chs[b][:, row, :]), N)...)
            F = svd(M; full = false)
            S_f = [sqrt(max(λ[k][1, 1, row], zero(T))) for k in 1:N]
            @test maximum(abs.(S_f .- F.S) ./ F.S[1]) < tol_s
            Vf = [V[i][k][1, 1, row] for i in 1:N, k in 1:N]
            # columns match up to sign; our convention V[1,k] ≥ 0
            @test all(Vf[1, :] .>= 0)
            for k in 1:N
                sgn = sign(F.V[1, k]) == 0 ? 1 : sign(F.V[1, k])
                # Gram-based eigenvectors of the SMALL singular values carry an
                # ~eps·λ₁/λ_k error (condition number squared); in Float32 that is
                # ~1e-2 for λ₁/λ_k ~ 1e5.  The denoiser output is insensitive to it
                # (those components re-enter scaled by σ_k) — see the 3-ch parity test.
                # (scaled by the observed condition number so the gate is not fixture-marginal)
                tol_k = (T === Float32 && k >= 2) ? max(2e-2, 20 * eps(T) * (F.S[1] / max(F.S[k], eps(T)))^2) : tol_v
                @test maximum(abs.(Vf[:, k] .- sgn .* F.V[:, k])) < tol_k
            end
            # orthonormality
            @test maximum(abs.(Vf' * Vf .- I(N))) < (T === Float64 ? 1e-12 : 1e-5)
        end
    end
end

# ----------------------------------------------------------------------------
# 1. SVD + joint bilateral
# ----------------------------------------------------------------------------
@testset "sino_svd_denoise_bilateral parity (2-channel, (96,4,48))" begin
    a, b = synth_channels(Float32)
    kw = (bilat_radius = 3, bilat_sigma_s = 2.0, bilat_range_k = 2.0)
    ref = with_logger(_quiet) do
        BS.apply_sino_svd_denoise_bilateral([a, b]; kw...)
    end
    plan = FStage.SinoSVDBilateralPlan((a, b); kw...)
    a0 = copy(a); b0 = copy(b)
    out = FStage.sino_svd_denoise_bilateral((a, b), plan)
    @test a == a0 && b == b0                       # purity
    @test size(out[1]) == size(a) && eltype(out[1]) == Float32
    r1 = _relmax(out[1], ref[1]); r2 = _relmax(out[2], ref[2])
    @info "SVD-bilateral 2ch parity: max rel = $(max(r1, r2))"
    @test r1 < 1e-4 && r2 < 1e-4
    # it actually did something
    @test _relmax(out[1], a) > 1e-4
    # host-constant plan shapes
    @test size(plan.σ_guide) == (1, 1, 4) && length(plan.σ_tgt) == 1
end

@testset "sino_svd_denoise_bilateral parity (3-channel Jacobi path)" begin
    chs = synth_channels(Float32; n_col = 64, n_row = 3, n_view = 32, n_ch = 3)
    kw = (bilat_radius = 2, bilat_sigma_s = 1.5, bilat_range_k = 2.0)
    ref = with_logger(_quiet) do
        BS.apply_sino_svd_denoise_bilateral(collect(chs); kw...)
    end
    plan = FStage.SinoSVDBilateralPlan(chs; kw...)
    out = FStage.sino_svd_denoise_bilateral(chs, plan)
    r = maximum(_relmax(out[k], ref[k]) for k in 1:3)
    @info "SVD-bilateral 3ch parity: max rel = $r"
    @test r < 1e-4
end

@testset "sino_svd_denoise_bilateral: degenerate / passthrough" begin
    flat = fill(0.5f0, 16, 2, 8)
    chs = (flat, 2.0f0 .* flat, 3.0f0 .* flat, 4.0f0 .* flat)   # rank 1 → λ₂..₄ = 0
    plan = FStage.SinoSVDBilateralPlan(chs; bilat_radius = 2, bilat_sigma_s = 1.5, bilat_range_k = 2.0)
    out = FStage.sino_svd_denoise_bilateral(chs, plan)
    @test all(o -> all(isfinite, o), out)
    @test maximum(_relmax(out[k], chs[k]) for k in 1:4) < 1e-5   # rank-1 → reconstitution ≈ identity
    a, b = synth_channels(Float32; n_col = 16, n_row = 2, n_view = 8)
    plan0 = FStage.SinoSVDBilateralPlan((a, b); bilat_range_k = 0.0)
    out0 = FStage.sino_svd_denoise_bilateral((a, b), plan0)
    @test out0[1] == a && out0[2] == b && out0[1] !== a
end

# ----------------------------------------------------------------------------
# 2. ACNR-Kalender (the live nb03/nb04 denoiser)
# ----------------------------------------------------------------------------
@testset "acnr_kalender parity (nb03 / nb04 kwargs)" begin
    for (T, tol) in ((Float32, 1e-5), (Float64, 1e-12))
        for kw in ((hp_sigma_px = 1.5, window = 4, passes = 5, beta_max = 14.0),   # nb03
                   (hp_sigma_px = 1.5, window = 4, passes = 4, beta_max = 20.0),   # nb04
                   (hp_sigma_px = 1.5, window = 4, passes = 2, beta_max = 8.0))    # src defaults
            W, I = synth_basis_pair(T)
            Wr = copy(W); Ir = copy(I)
            info_ref = BS.apply_acnr_kalender!(Wr, Ir; kw...)
            plan = FStage.ACNRKalenderPlan(T; kw...)
            W0 = copy(W); I0 = copy(I)
            Wf, If, info = FStage.acnr_kalender(W, I, plan)
            @test W == W0 && I == I0
            rW = _relmax(Wf, Wr); rI = _relmax(If, Ir)
            @info "ACNR $(T) passes=$(kw.passes) beta_max=$(kw.beta_max): rel W=$rW I=$rI  " *
                "ρ_hp=$(info.ρ_hp) (legacy $(info_ref.ρ_hp))"
            @test rW < tol && rI < tol
            @test abs(info.ρ_hp - info_ref.ρ_hp) < 100 * eps(T)
            @test abs(info.σ_hW - info_ref.σ_hW) < 100 * eps(T) * abs(info_ref.σ_hW)
            @test abs(info.σ_hI - info_ref.σ_hI) < 100 * eps(T) * abs(info_ref.σ_hI)
            @test _relmax(If, I) > 1e-3                      # it corrected something
        end
    end
end

# ----------------------------------------------------------------------------
# 3. median-z
# ----------------------------------------------------------------------------
@testset "median_z: interior exact, boundary documented, :shrink exact" begin
    rng = MersenneTwister(5)
    vol = Float32.(randn(rng, 12, 10, 9))
    for n in (1, 2)
        ref = BS.apply_median_z(vol; adjacent_slices = n)
        rep = FStage.median_z(vol, FStage.MedianZPlan(; adjacent_slices = n, boundary = :replicate))
        shr = FStage.median_z(vol, FStage.MedianZPlan(; adjacent_slices = n, boundary = :shrink))
        @test size(rep) == size(vol) && size(shr) == size(vol)
        @test rep[:, :, (n + 1):(end - n)] == ref[:, :, (n + 1):(end - n)]   # interior bit-exact
        @test shr == ref                                                      # :shrink bit-exact everywhere
        nb = 2n * 12 * 10
        ndiff = count(rep[:, :, [1:n; (10 - n):9]] .!= ref[:, :, [1:n; (10 - n):9]])
        md = maximum(abs.(rep .- ref))
        @info "median_z n=$n :replicate boundary vs legacy shrink: $(ndiff)/$(nb) boundary voxels differ, max |Δ| = $md"
        @test ndiff > 0                     # documented, expected difference
    end
    # adjacent_slices = 0 → copy
    z = FStage.median_z(vol, FStage.MedianZPlan(; adjacent_slices = 0))
    @test z == vol && z !== vol
    # nz ≤ 2n path (all slices are boundary)
    small = Float32.(randn(rng, 4, 3, 3))
    @test FStage.median_z(small, FStage.MedianZPlan(; adjacent_slices = 2, boundary = :shrink)) ==
          BS.apply_median_z(small; adjacent_slices = 2)
end

# ----------------------------------------------------------------------------
# 4. SF-JSD with constants captured from a legacy run
# ----------------------------------------------------------------------------
@testset "sfjsd_denoise parity (captured constants)" begin
    # Oracle = the legacy algorithm run SEQUENTIALLY with the legacy internals
    # (`FStage.sfjsd_capture` advances ξ row by row with `BS._sfjsd_pass!` and
    # returns the inverse-whitened result next to the captured plan).  The
    # public threaded driver has a captured-variable data race (`slice_lo` /
    # `slice_hi` are function-scope locals reassigned inside `Threads.@threads`),
    # so with nthreads() > 1 it mixes rows; with nthreads() == 1 it is bit-exact
    # against the sequential oracle — both facts are checked/reported below.
    function _check(a, b, I0, σ₀, label; tol = 1e-4)
        # NB: locals here must not share names with testset-scope variables —
        # a closure assigning an outer local rebinds it (the legacy race is the
        # same mechanism with Threads.@threads).
        pl, r_lo, r_hi = with_logger(_quiet) do
            FStage.sfjsd_capture([a, b], I0; σ₀ = σ₀)
        end
        pub = with_logger(_quiet) do
            BS.apply_sino_sfjsd_denoise([a, b], I0; σ₀ = σ₀, verbose = false)
        end
        rp = max(_relmax(pub[1], r_lo), _relmax(pub[2], r_hi))
        if Threads.nthreads() == 1
            @test rp < 1e-6
        else
            @info "$label: public threaded legacy vs sequential legacy = $rp (nthreads = $(Threads.nthreads()); nonzero ⇒ the captured-variable race fired)"
        end
        fo = FStage.sfjsd_denoise(a, b, pl)
        rr = max(_relmax(fo[1], r_lo), _relmax(fo[2], r_hi))
        @info "$label (stride=$(pl.stride), n_iter=$(pl.n_iter), R=$(pl.radius_max), σ₀=$(pl.σ0_iter)): max rel = $rr"
        @test rr < tol
        @test _relmax(fo[1], a) > 1e-4
        return pl
    end

    # (a) user σ₀ (SURE skipped), high flux → n_iter = 1, stride 3
    a, b = synth_channels(Float32; n_col = 64, n_row = 4, n_view = 32, I0 = (1.0e5, 6.0e4))
    plan = _check(a, b, [1.0e5, 6.0e4], 1.0, "SF-JSD σ₀=1.0")
    @test plan.n_iter == 1

    # (b) photon-starved (n_iter = 2, per-iteration scales on the legacy trajectory)
    a2, b2 = synth_channels(Float32; n_col = 48, n_row = 2, n_view = 24, I0 = (300.0, 200.0), seed = 3)
    plan2 = _check(a2, b2, [300.0, 200.0], 0.8, "SF-JSD σ₀=0.8 starved")
    @test plan2.n_iter == 2

    # (c) SURE auto-σ₀ capture on a tiny grid (the RNG lives in the host helper)
    a3, b3 = synth_channels(Float32; n_col = 24, n_row = 2, n_view = 12, I0 = (2.0e4, 1.5e4), seed = 9)
    _check(a3, b3, [2.0e4, 1.5e4], 0.0, "SF-JSD SURE auto-σ₀")

    # single core pass: finite, shape-preserving, pure
    a0 = copy(a)
    ξa, ξb = FStage.sfjsd_pass(a, b, plan, 1)
    @test a == a0
    @test size(ξa) == size(a) && all(isfinite, ξa) && all(isfinite, ξb)
end

# ----------------------------------------------------------------------------
# 5. Float64 finite-difference gradient / smoothness check (SVD-bilateral)
# ----------------------------------------------------------------------------
@testset "sino_svd_denoise_bilateral: Float64 FD gradient check" begin
    a, b = synth_channels(Float64; n_col = 20, n_row = 2, n_view = 12)
    plan = FStage.SinoSVDBilateralPlan((a, b); bilat_radius = 2, bilat_sigma_s = 1.5, bilat_range_k = 2.0)
    rng = MersenneTwister(3)
    c1 = randn(rng, size(a)); c2 = randn(rng, size(b))
    L(x, y) = (o = FStage.sino_svd_denoise_bilateral((x, y), plan); sum(c1 .* o[1]) + sum(c2 .* o[2]))
    # well-separated singular values at this point
    λ, _ = FStage.gram_eigen((permutedims(a, (1, 3, 2)), permutedims(b, (1, 3, 2))))
    @test minimum(λ[1] ./ λ[2]) > 10 && minimum(λ[2]) > 0
    u = randn(rng, size(a)); v = randn(rng, size(b))
    dd(h) = (L(a .+ h .* u, b .+ h .* v) - L(a .- h .* u, b .- h .* v)) / (2h)
    d1 = dd(1e-3); d2 = dd(5e-4); d3 = dd(2.5e-4)
    # central differences converge as O(h²): successive estimates must agree
    # to far better than 1e-3 relative if the map is smooth here.
    @test abs(d1 - d2) < 1e-3 * abs(d2)
    @test abs(d2 - d3) < 1e-3 * abs(d3)
    # Richardson-extrapolated reference vs the finest estimate
    d_rich = (4d3 - d2) / 3
    @test abs(d3 - d_rich) < 1e-3 * abs(d_rich)
    # optional: a real AD gradient when ForwardDiff is on the load path
    if Base.find_package("ForwardDiff") !== nothing
        FD = Base.require(Main, :ForwardDiff)
        g = FD.gradient(x -> L(reshape(x[1:length(a)], size(a)), reshape(x[(length(a) + 1):end], size(b))), vcat(vec(a), vec(b)))
        ad_dir = dot(g, vcat(vec(u), vec(v)))
        @test abs(ad_dir - d_rich) < 1e-6 * abs(d_rich)
        @info "ForwardDiff directional derivative vs Richardson FD: $(ad_dir) vs $(d_rich)"
    else
        @info "ForwardDiff not on the load path — AD gradient check skipped (FD smoothness only)"
    end
end

end # module FunctionalDenoiseTests
