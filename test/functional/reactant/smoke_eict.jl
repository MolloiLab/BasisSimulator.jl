# Reactant / Enzyme smoke test for the functional EICT detector chain.
#
#   julia --project=envs/reactant -t 2 --heap-size-hint=3G test/functional/reactant/smoke_eict.jl
#
# Memory rule: Reactant runs are serialized system-wide (one at a time) with
# the atomic lock `/tmp/bs_reactant.lock` — acquire with
#   until mkdir /tmp/bs_reactant.lock 2>/dev/null; do sleep 30; done
# and ALWAYS release it (`rmdir /tmp/bs_reactant.lock`, e.g. via `trap ... EXIT`).
# Toy sizes only; XLA compile of this chain takes 1-3 min.
#
# Compiles `eict_chain` (per-material path lengths → fill factor → scatter
# estimate/inject → reparameterised noise → air normalisation → log → BHC) on a
# toy size with EVERY variant enabled (bowtie/heel weights, air reference, fill
# factor, quantum + electronic noise, separable-Gaussian scatter, order-3 BHC).
#
# Gates:
#   • compiled Float64 and Float32 == plain-Array to ≤ 1e-5 rel of max|ref|;
#   • `Enzyme.gradient(Reverse, …)` of `sum(w .* eict_chain(P))` w.r.t. the path
#     lengths agrees with plain-Array Float64 central differences to ≤ 1e-3 rel
#     (directional derivative + per-entry probes) and with the hand-derived
#     `eict_chain_vjp` adjoint.
# Reported, not gated: the static-chunked spectral variant under @compile.

using Test
using Reactant, Enzyme
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "eict.jl"))
end
const F = FStage

const _T0 = time()
_ts(label) = (println(stderr, "[smoke_eict t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))
relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))

# ----------------------------------------------------------------------------- fixture
const N_COL, N_ROW, N_VIEW, N_MAT, N_E = 16, 4, 6, 3, 8
const SHAPE = (N_COL, N_ROW, N_VIEW)

function fixture(::Type{T}) where {T}
    rng = MersenneTwister(0xE1C7)
    P = T.(3.0 .* rand(rng, N_COL, N_ROW, N_VIEW, N_MAT))
    μ = zeros(T, N_MAT, N_E)
    for e in 1:N_E
        μ[2, e] = T(0.30 - 0.12 * (e - 1) / (N_E - 1))
        μ[3, e] = T(1.50 - 0.90 * (e - 1) / (N_E - 1))
    end
    wη = T.([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
    bt = T.(0.6 .+ 0.4 .* rand(rng, N_COL, N_ROW, N_E))
    air = reshape(vec(sum(reshape(wη, 1, 1, :) .* bt; dims = 3)), N_COL, N_ROW)
    k1d = T.([0.05, 0.2, 0.5, 0.2, 0.05])
    Hc = F.clamped_conv_matrix(k1d, N_COL); Hr = F.clamped_conv_matrix(k1d, N_ROW)
    coeffs = T.([0.01 .+ 0.001 .* (1:N_COL)'; 1.02 .+ 0.001 .* (1:N_COL)'; 0.003 .* ones(1, N_COL); -0.0004 .* ones(1, N_COL)])
    ε = T.(randn(rng, N_COL, N_ROW, N_VIEW))
    εe = T.(randn(rng, N_COL, N_ROW, N_VIEW))
    w = T.(randn(rng, N_COL, N_ROW, N_VIEW))
    return (; P, μ, wη, bt, air, Hc, Hr, coeffs, ε, εe, w)
end

# Host scalars of the plan (config-only).  The plan is rebuilt INSIDE the traced
# function from the (traced) arrays so every array flows through Reactant as a
# program input while the variant selection stays a host decision.
const I0 = 2.0e6
const σ_E = 3.0
const FF_LOG = log(0.81)
const SC_C = 0.02
const SC_SW = 0.9

# `eltype` is the HOST scalar type (unwrapped from traced element types by the
# plan constructor itself); all scalar kwargs are converted to it.
make_plan(μ, wη, bt, air, Hc, Hr, coeffs) = F.EICTPlan(;
    μ_tbl = μ, wη, sino_shape = SHAPE, I0, bt, air_ref = air, σ_e = σ_E, use_noise = true,
    ff_log = FF_LOG, scatter_Hc = Hc, scatter_Hr = Hr, scatter_C = SC_C, scatter_sw = SC_SW,
    bhc_coeffs = coeffs)

run_chain(P, μ, wη, bt, air, Hc, Hr, coeffs, ε, εe) =
    F.eict_chain(P, make_plan(μ, wη, bt, air, Hc, Hr, coeffs), ε, εe)

run_chunked(P, μ, wη, bt) = F.poly_log_sinogram_chunked(P, μ, wη, bt, Val(2))

_ts("start")
@testset "Reactant smoke: functional EICT chain" begin
    for T in (Float64, Float32)
        _ts("T = $T: fixture + plain-Array reference")
        fx = fixture(T)
        ref = run_chain(fx.P, fx.μ, fx.wη, fx.bt, fx.air, fx.Hc, fx.Hr, fx.coeffs, fx.ε, fx.εe)
        @test all(isfinite, ref)
        println("  [$T] plain-Array chain: range = $(extrema(ref))")

        args = (fx.P, fx.μ, fx.wη, fx.bt, fx.air, fx.Hc, fx.Hr, fx.coeffs, fx.ε, fx.εe)
        rargs = map(Reactant.to_rarray, args)
        _ts("T = $T: compile eict_chain")
        t0 = time()
        chain_c = @compile run_chain(rargs...)
        println("  [$T] compile time: $(round(time() - t0; digits = 1)) s")
        out = Array(chain_c(rargs...))
        e = relmax(out, ref)
        println("  [$T] compiled vs plain-Array rel = $e")
        @test e ≤ 1e-5
        @test all(isfinite, out)

        if T === Float64
            w = fx.w
            loss(P, μ, wη, bt, air, Hc, Hr, coeffs, ε, εe) = sum(w .* run_chain(P, μ, wη, bt, air, Hc, Hr, coeffs, ε, εe))
            _ts("T = $T: compile Enzyme.gradient wrt path lengths")
            t0 = time()
            cargs = (rargs[1], map(Const, rargs[2:end])...)
            grad_c = @compile Enzyme.gradient(Reverse, loss, cargs...)
            println("  [$T] gradient compile time: $(round(time() - t0; digits = 1)) s")
            g = Array(grad_c(Reverse, loss, cargs...)[1])
            @test size(g) == size(fx.P)
            @test all(isfinite, g)

            # hand-derived adjoint (plain Arrays)
            plan_h = make_plan(fx.μ, fx.wη, fx.bt, fx.air, fx.Hc, fx.Hr, fx.coeffs)
            g_vjp = F.eict_chain_vjp(fx.P, plan_h, fx.ε, fx.εe, w)
            e_vjp = relmax(g, g_vjp)
            println("  [$T] Enzyme gradient vs eict_chain_vjp rel = $e_vjp")
            @test e_vjp ≤ 1e-6

            # finite differences (plain Arrays)
            loss_h(P) = sum(w .* F.eict_chain(P, plan_h, fx.ε, fx.εe))
            h = 1e-6
            d = randn(MersenneTwister(21), T, size(fx.P))
            dd_fd = (loss_h(fx.P .+ h .* d) - loss_h(fx.P .- h .* d)) / (2h)
            dd_ad = sum(g .* d)
            rel_dir = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
            println("  [$T] directional derivative: AD = $dd_ad, FD = $dd_fd, rel = $rel_dir")
            @test rel_dir ≤ 1e-3
            rng = MersenneTwister(5)
            worst = 0.0
            for I in rand(rng, CartesianIndices(fx.P), 5)
                e1 = zeros(T, size(fx.P)); e1[I] = h
                fd = (loss_h(fx.P .+ e1) - loss_h(fx.P .- e1)) / (2h)
                rel = abs(g[I] - fd) / max(abs(fd), 1e-12)
                worst = max(worst, rel)
                println("  [$T] ∂loss/∂P$(Tuple(I)): AD = $(g[I]), FD = $fd, rel = $rel")
            end
            @test worst ≤ 1e-3
        end
    end
end

@testset "Reactant smoke: chunked spectral variant (reported)" begin
    T = Float32
    fx = fixture(T)
    ref = run_chunked(fx.P, fx.μ, fx.wη, fx.bt)
    rargs = map(Reactant.to_rarray, (fx.P, fx.μ, fx.wη, fx.bt))
    ok = try
        _ts("chunked: compile")
        t0 = time()
        c = @compile run_chunked(rargs...)
        println("  chunked compile time: $(round(time() - t0; digits = 1)) s")
        e = relmax(Array(c(rargs...)), ref)
        println("  chunked compiled vs plain-Array rel = $e")
        e ≤ 1e-5
    catch err
        println("  chunked variant did not compile under Reactant: ", sprint(showerror, err)[1:min(end, 600)])
        false
    end
    println("  chunked Reactant status: ", ok ? "PASS" : "NOT PASSING")
    @test true   # reported, not gated
end
_ts("done")
