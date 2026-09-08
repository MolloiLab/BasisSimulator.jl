# Reactant / Enzyme smoke test for the functional n-channel VMI estimator.
#
#   julia --project=envs/reactant -t 2 test/functional/reactant/smoke_nchannel.jl
#
# Compiles `nchannel_estimate_tile` (the published profiled Poisson
# quasi-likelihood estimator: linear initializer → aggregate bisection →
# outer iodine / inner water Newton with freeze masks → Fisher diagnostics)
# on a toy K = 4 tile with REAL 120 kVp spectra, and differentiates a scalar
# loss of the basis sinograms with respect to the channel log-transmissions.
#
# Gates:
#   • compiled Float64 == plain-Array to ≤ 1e-5 rel of max|ref| for both basis
#     sinograms (Float32 too, reported);
#   • `Enzyme.gradient(Reverse, …)` of `sum(wI .* A) + sum(wW .* C)` w.r.t. `h`
#     agrees with plain-Array Float64 central differences to ≤ 1e-3 rel
#     (directional derivative + per-entry probes).
#
# Iteration ceilings: the production plan unrolls 28 + 16×12 + 12 fixed steps
# into the trace.  `SMOKE_FULL=1` compiles those; the default uses reduced
# ceilings (12 bisections, 6 outer × 4 inner) so the smoke stays quick — the
# same fixed-count program, just shorter, converged to `parameter_tolerance`
# on every fixture ray (checked: quality flag == 0).

using Test
using Reactant, Enzyme
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "nchannel.jl"))
end
const F = FStage

const _T0 = time()
_ts(label) = (println(stderr, "[smoke_nchannel t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))
relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))

const FULL = get(ENV, "SMOKE_FULL", "0") == "1"
const N_COL, N_ROW, N_VIEW = 4, 1, 4          # 16 rays
const E_GRID = collect(20.0:4.0:140.0)         # 31 energies

# ----------------------------------------------------------------------------- fixture
function rebin_spectrum(kVp)
    e, w = BS.load_spectrum(kVp)
    out = zeros(Float64, length(E_GRID))
    for (ei, wi) in zip(e, w)
        j = round(Int, (ei - E_GRID[1]) / 4) + 1
        1 <= j <= length(out) && (out[j] += wi)
    end
    return out ./ sum(out)
end

function pcct4_tables(; I0_total = 4.0e4)
    w = rebin_spectrum(120)
    edges = (45.0, 60.0, 75.0)
    soft(x) = 1 / (1 + exp(-x / 2.0))
    R = hcat(
        [1 - soft(e - edges[1]) for e in E_GRID],
        [soft(e - edges[1]) * (1 - soft(e - edges[2])) for e in E_GRID],
        [soft(e - edges[2]) * (1 - soft(e - edges[3])) for e in E_GRID],
        [soft(e - edges[3]) for e in E_GRID],
    )
    Φ = I0_total .* w .* R
    return Φ, vec(sum(Φ; dims = 1))
end

function make_plan(Φ, I0, ::Type{T}) where {T}
    kw = FULL ? (;) : (; bisection_iterations = 12, outer_iterations = 6, inner_iterations = 4)
    return F.nchannel_plan(Φ, E_GRID, I0; T = T, parameter_tolerance = (T === Float64 ? 1.0e-11 : 5.0e-5), kw...)
end

# Channel log-transmissions of a smooth (A, C) field through the exact model.
function fixture(::Type{T}, plan64) where {T}
    rng = MersenneTwister(0x4c4a)
    A = reshape([0.02 + 0.25 * (i - 1) / 15 for i in 1:16], N_COL, N_ROW, N_VIEW)
    C = reshape([2.0 + 30.0 * ((i * 7) % 16) / 15 for i in 1:16], N_COL, N_ROW, N_VIEW)
    λ, _, _ = F.nchannel_moments(A, C, plan64)
    y = λ .* (1 .+ 0.02 .* randn(rng, size(λ)))
    h = -log.(y ./ plan64.I0)
    return T.(h), T.(randn(rng, size(A))), T.(randn(rng, size(A)))
end

estimate_pair(h, plan) = (r = F.nchannel_estimate_tile(h, plan); (r.sino_iodine, r.sino_water))
loss(h, plan, wI, wW) = (p = estimate_pair(h, plan); sum(wI .* p[1]) + sum(wW .* p[2]))

_ts("start (FULL = $FULL)")
Φ, I0 = pcct4_tables()
plan64 = make_plan(Φ, I0, Float64)
h64, wI64, wW64 = fixture(Float64, plan64)
ref64 = F.nchannel_estimate_tile(h64, plan64)
println("  converged flags (plain Float64): ", count(==(0.0), ref64.quality_flag), " / ", length(ref64.quality_flag),
    "  max outer/inner used = ", maximum(ref64.outer_iterations), "/", maximum(ref64.inner_iterations))

@testset "Reactant smoke: n-channel estimator" begin
    for T in (Float64, Float32)
        plan = T === Float64 ? plan64 : make_plan(Φ, I0, Float32)
        h = T.(h64)
        ref = estimate_pair(h, plan)
        @test all(isfinite, ref[1]) && all(isfinite, ref[2])
        rh = Reactant.to_rarray(h)
        rplan = Reactant.to_rarray(plan)
        _ts("T = $T: compile nchannel_estimate_tile")
        t0 = time()
        est_c = @compile estimate_pair(rh, rplan)
        println("  [$T] compile time: $(round(time() - t0; digits = 1)) s")
        out = est_c(rh, rplan)
        eI = relmax(Array(out[1]), ref[1]); eW = relmax(Array(out[2]), ref[2])
        println("  [$T] compiled vs plain-Array rel: iodine = $eI, water = $eW")
        @test eI ≤ 1e-5
        @test eW ≤ 1e-5

        if T === Float64
            rwI = Reactant.to_rarray(wI64); rwW = Reactant.to_rarray(wW64)
            _ts("T = $T: compile Enzyme.gradient wrt channel sinograms")
            t0 = time()
            grad_c = @compile Enzyme.gradient(Reverse, loss, rh, Const(rplan), Const(rwI), Const(rwW))
            println("  [$T] gradient compile time: $(round(time() - t0; digits = 1)) s")
            g = Array(grad_c(loss, rh, Const(rplan), Const(rwI), Const(rwW))[1])
            @test size(g) == size(h64)
            @test all(isfinite, g)

            loss_h(hh) = loss(hh, plan64, wI64, wW64)
            δ = 1e-5
            d = randn(MersenneTwister(21), Float64, size(h64))
            dd_fd = (loss_h(h64 .+ δ .* d) - loss_h(h64 .- δ .* d)) / (2δ)
            dd_ad = sum(g .* d)
            rel_dir = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
            println("  [$T] directional derivative: AD = $dd_ad, FD = $dd_fd, rel = $rel_dir")
            @test rel_dir ≤ 1e-3
            rng = MersenneTwister(5)
            worst = 0.0
            for I in rand(rng, CartesianIndices(h64), 6)
                e1 = zeros(Float64, size(h64)); e1[I] = δ
                fd = (loss_h(h64 .+ e1) - loss_h(h64 .- e1)) / (2δ)
                rel = abs(g[I] - fd) / max(abs(fd), 1e-12)
                worst = max(worst, rel)
                println("  [$T] ∂loss/∂h$(Tuple(I)): AD = $(g[I]), FD = $fd, rel = $rel")
            end
            @test worst ≤ 1e-3
        end
    end
end
_ts("done")
