# Reactant / Enzyme smoke test for the functional PCCT chain.
#
#   julia --project=envs/reactant -t 2 test/functional/reactant/smoke_pcct.jl
#
# Compiles the chain from per-material path lengths to bin-combined log
# sinograms (`pcct_chain_combined`, with pile-up S, optional spectral bowtie,
# and externally supplied counts) on a toy size, and differentiates the
# surrogate-noise chain w.r.t. the path lengths with Enzyme under Reactant.
#
# Gates:
#   • compiled == plain-Array to ≤ 1e-5 rel of max|ref| (Float32 and Float64,
#     noise-free, with input counts, with bowtie);
#   • `Enzyme.gradient(Reverse, …)` of `sum(C .* chain(P))` w.r.t. `P`
#     (Float64, noise-free AND straight-through surrogate noise) agrees with
#     plain-Array central differences to ≤ 1e-3 rel (directional derivative +
#     per-entry probes) and with the closed-form `spectral_bins_vjp` chain.

using Test
using Reactant, Enzyme
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics, Random
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "pcct.jl"))
end
const F = FStage

const _T0 = time()
_ts(label) = (println(stderr, "[smoke_pcct t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))
relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))

# ----------------------------------------------------------------------------- fixture
const N_COL, N_ROW, N_VIEW, N_MAT, N_E, N_BINS = 16, 2, 6, 3, 12, 4
const GROUPS = [[1, 2, 3], [4]]

function toy(T; view_chunks = 1)
    rng = MersenneTwister(7)
    P = T.(3.0 .* rand(rng, N_COL, N_ROW, N_VIEW, N_MAT))
    e = range(0.0, 1.0; length = N_E)
    μ64 = vcat(1.0e-4 .* ones(1, N_E), 0.4 .* exp.(-e)', 2.0 .* exp.(-1.5 .* e)')   # air, water-ish, dense
    W64 = 1.0e5 .* (0.2 .+ rand(rng, N_E, N_BINS))
    I0 = [sum(W64[:, b]) for b in 1:N_BINS]
    S64 = LinearAlgebra.tril(0.05 .+ 0.1 .* rand(rng, N_BINS, N_BINS)) + 0.4 .* LinearAlgebra.I(N_BINS)
    bt = T.(0.6 .+ 0.4 .* rand(rng, N_COL, N_ROW, N_E))
    plan = F.pcct_plan(T.(μ64), T.(W64), I0, S64; T, view_chunks)
    G, I0g = F.combine_matrix(I0, GROUPS, T)
    return (; P, plan, G, I0g, bt, I0)
end

_ts("start")
@testset "Reactant smoke: functional PCCT chain" begin
    for T in (Float32, Float64)
        _ts("T = $T: fixture")
        fx = toy(T)
        P, plan, G, I0g, bt = fx.P, fx.plan, fx.G, fx.I0g, fx.bt
        # host-realized exact Poisson counts on the plain-Array PRE-noise bins
        # (noise is drawn before pile-up, so not `pcct_chain(P, plan).bins`)
        p_clean = F.bin_log_sinograms(F.spectral_bin_intensities(P, plan.μ_table, plan.W), plan.I0_bins, plan.eps)
        N = F.draw_pcct_counts(p_clean, plan.I0_bins_f64, 42)

        # Config tensors (plan, combine matrix, bowtie) become traced INPUTS of
        # the compiled program — no host Array may meet a traced array in `*`
        # or a broadcast (see the Reactant note in src/functional/pcct.jl).
        f0(Px, pl, Gx, gx) = F.pcct_chain_combined(Px, pl, Gx, gx)                 # noise-free
        f1(Px, pl, Gx, gx, Nx) = F.pcct_chain_combined(Px, pl, Gx, gx, Nx)         # input counts
        f2(Px, pl, Gx, gx, btx) = F.pcct_chain_combined(Px, pl, Gx, gx; bt = btx)  # spectral bowtie
        ref0 = f0(P, plan, G, I0g); ref1 = f1(P, plan, G, I0g, N); ref2 = f2(P, plan, G, I0g, bt)

        P_r = Reactant.to_rarray(P); N_r = Reactant.to_rarray(N); bt_r = Reactant.to_rarray(bt)
        plan_r = Reactant.to_rarray(plan); G_r = Reactant.to_rarray(G); I0g_r = Reactant.to_rarray(I0g)
        @test plan_r isa F.PCCTPlan{T}
        _ts("T = $T: compile noise-free chain")
        t0 = time(); c0 = @compile f0(P_r, plan_r, G_r, I0g_r)
        println("  [$T] compile (noise-free): $(round(time() - t0; digits = 1)) s")
        out0 = Array(c0(P_r, plan_r, G_r, I0g_r))
        e0 = relmax(out0, ref0); println("  [$T] noise-free compiled vs plain rel = $e0")
        @test e0 ≤ 1e-5
        @test all(isfinite, out0)

        _ts("T = $T: compile chain with input counts")
        t0 = time(); c1 = @compile f1(P_r, plan_r, G_r, I0g_r, N_r)
        println("  [$T] compile (input counts): $(round(time() - t0; digits = 1)) s")
        out1 = Array(c1(P_r, plan_r, G_r, I0g_r, N_r))
        e1 = relmax(out1, ref1); println("  [$T] input-counts compiled vs plain rel = $e1")
        @test e1 ≤ 1e-5

        _ts("T = $T: compile chain with bowtie")
        t0 = time(); c2 = @compile f2(P_r, plan_r, G_r, I0g_r, bt_r)
        println("  [$T] compile (bowtie): $(round(time() - t0; digits = 1)) s")
        out2 = Array(c2(P_r, plan_r, G_r, I0g_r, bt_r))
        e2 = relmax(out2, ref2); println("  [$T] bowtie compiled vs plain rel = $e2")
        @test e2 ≤ 1e-5

        if T === Float64
            C = randn(MersenneTwister(11), T, N_COL, N_ROW, N_VIEW, length(GROUPS))
            C_r = Reactant.to_rarray(C)
            λ = F.pcct_expected_counts(p_clean, plan.I0_bins)
            ε = F.implied_noise_eps(N, λ)
            ε_r = Reactant.to_rarray(ε)
            loss0(Px, pl, Gx, gx, Cx) = sum(Cx .* f0(Px, pl, Gx, gx))
            loss_s(Px, pl, Gx, gx, Cx, εx) = sum(Cx .* F.pcct_chain_surrogate_combined(Px, pl, Gx, gx, εx))
            # straight-through value check: surrogate with implied ε == exact-count chain
            @test relmax(F.pcct_chain_surrogate_combined(P, plan, G, I0g, ε), ref1) ≤ 1e-8

            for (label, lossf, consts_r, consts_h) in (
                    ("noise-free", loss0, (plan_r, G_r, I0g_r, C_r), (plan, G, I0g, C)),
                    ("surrogate noise", loss_s, (plan_r, G_r, I0g_r, C_r, ε_r), (plan, G, I0g, C, ε)))
                _ts("T = $T: compile Enzyme.gradient ($label)")
                t0 = time()
                cargs = map(Const, consts_r)
                grad_c = @compile Enzyme.gradient(Reverse, lossf, P_r, cargs...)
                println("  [$T] gradient compile ($label): $(round(time() - t0; digits = 1)) s")
                g = Array(grad_c(Reverse, lossf, P_r, cargs...)[1])
                @test size(g) == size(P)
                @test all(isfinite, g)
                loss_h(Px) = lossf(Px, consts_h...)
                d = randn(MersenneTwister(21), T, size(P))
                h = 1e-6
                dd_fd = (loss_h(P .+ h .* d) - loss_h(P .- h .* d)) / (2h)
                dd_ad = sum(g .* d)
                rel_dir = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
                println("  [$T] $label directional derivative: AD = $dd_ad, FD = $dd_fd, rel = $rel_dir")
                @test rel_dir ≤ 1e-3
                rng = MersenneTwister(5)
                worst = 0.0
                for I in rand(rng, CartesianIndices(P), 4)
                    e1 = zeros(T, size(P)); e1[I] = h
                    fd = (loss_h(P .+ e1) - loss_h(P .- e1)) / (2h)
                    rel = abs(g[I] - fd) / max(abs(fd), 1e-12)
                    worst = max(worst, rel)
                    println("  [$T] $label ∂loss/∂P$(Tuple(I)): AD = $(g[I]), FD = $fd, rel = $rel")
                end
                @test worst ≤ 1e-3
            end

            # closed-form VJP of the spectral+log front end vs Enzyme on the same sub-chain
            front(Px, pl) = F.bin_log_sinograms(F.spectral_bin_intensities(Px, pl.μ_table, pl.W), pl.I0_bins, pl.eps)
            Cb = randn(MersenneTwister(13), T, N_COL, N_ROW, N_VIEW, N_BINS)
            Cb_r = Reactant.to_rarray(Cb)
            loss_f(Px, pl, Cbx) = sum(Cbx .* front(Px, pl))
            gf_c = @compile Enzyme.gradient(Reverse, loss_f, P_r, Const(plan_r), Const(Cb_r))
            gf = Array(gf_c(Reverse, loss_f, P_r, Const(plan_r), Const(Cb_r))[1])
            Ib = F.spectral_bin_intensities(P, plan.μ_table, plan.W)
            g_closed = F.spectral_bins_vjp(P, plan.μ_table, plan.W, nothing,
                F.bin_log_sinograms_vjp(Ib, plan.I0_bins, plan.eps, Cb))
            e_vjp = relmax(gf, g_closed)
            println("  [$T] Enzyme vs closed-form spectral_bins_vjp rel = $e_vjp")
            @test e_vjp ≤ 1e-8
        end
    end

    # view chunking (static pairwise `cat` on the view axis) also traces
    T = Float32
    fx = toy(T; view_chunks = 3)
    f3(Px, pl, Gx, gx) = F.pcct_chain_combined(Px, pl, Gx, gx)
    P_r = Reactant.to_rarray(fx.P); plan_r = Reactant.to_rarray(fx.plan)
    G_r = Reactant.to_rarray(fx.G); I0g_r = Reactant.to_rarray(fx.I0g)
    _ts("view_chunks = 3: compile")
    c3 = @compile f3(P_r, plan_r, G_r, I0g_r)
    e3 = relmax(Array(c3(P_r, plan_r, G_r, I0g_r)), f3(fx.P, fx.plan, fx.G, fx.I0g))
    println("  [view_chunks = 3] compiled vs plain rel = $e3")
    @test e3 ≤ 1e-5
end
_ts("done")
