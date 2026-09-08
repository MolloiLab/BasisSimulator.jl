# Reactant / Enzyme smoke test for the functional denoisers.
#
#   julia --project=envs/reactant -t 2 test/functional/reactant/smoke_denoise.jl
#
# Compiles each denoiser in src/functional/denoise.jl on a toy size and gates:
#   • compiled == plain-Array to ≤ 1e-5 rel of max|ref| (Float32 for every
#     stage; Float64 additionally for the SVD-bilateral stage);
#   • `Enzyme.gradient(Reverse, …)` of a scalar loss of the SVD-bilateral
#     denoiser w.r.t. both input channels agrees with plain-Array Float64
#     central differences to ≤ 1e-3 rel (directional derivative + per-entry
#     probes).
# Host constants (MAD range scales, SF-JSD captured constants) live in the plan
# structs, which are captured as constants by the traced closures.

using Test
using Reactant, Enzyme
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "denoise.jl"))
end
const F = FStage

const _T0 = time()
_ts(label) = (println(stderr, "[smoke_denoise t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))
relmax(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / max(maximum(abs.(Float64.(b))), 1e-300)

# ----------------------------------------------------------------------------- fixtures
function synth_channels(::Type{T}; n_col, n_row, n_view, seed = 7, I0 = (1.0e5, 6.0e4)) where {T}
    rng = MersenneTwister(seed)
    cs = range(-1, 1; length = n_col); vs = range(0, 2π; length = n_view)
    wts = ((1.3, 0.9), (0.8, 0.35))
    out = ntuple(2) do b
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

function synth_basis_pair(::Type{T}; n = 16, nz = 2, seed = 11) where {T}
    rng = MersenneTwister(seed)
    xs = range(-1, 1; length = n)
    W = [Float64((x^2 + y^2 < 0.6^2)) + 0.3 * ((x - 0.2)^2 + (y + 0.1)^2 < 0.3^2) for x in xs, y in xs, z in 1:nz]
    I = [2.0 * ((x - 0.2)^2 + (y + 0.1)^2 < 0.3^2) for x in xs, y in xs, z in 1:nz]
    n1 = randn(rng, n, n, nz); n2 = randn(rng, n, n, nz)
    W .+= 0.03 .* n1 .+ 0.01 .* n2
    I .-= 0.06 .* n1 .- 0.02 .* n2
    return T.(W), T.(I)
end

_ts("start")

# ----------------------------------------------------------------------------- 1. SVD-bilateral
@testset "Reactant smoke: sino_svd_denoise_bilateral" begin
    for T in (Float32, Float64)
        a, b = synth_channels(T; n_col = 16, n_row = 2, n_view = 12)
        plan = F.SinoSVDBilateralPlan((a, b); bilat_radius = 2, bilat_sigma_s = 1.5, bilat_range_k = 2.0)
        ref = F.sino_svd_denoise_bilateral((a, b), plan)
        f(x, y) = F.sino_svd_denoise_bilateral((x, y), plan)
        ra, rb = Reactant.to_rarray(a), Reactant.to_rarray(b)
        _ts("SVD-bilateral $T: compile")
        t0 = time()
        fc = @compile f(ra, rb)
        println("  [$T] compile time: $(round(time() - t0; digits = 1)) s")
        out = fc(ra, rb)
        e = max(relmax(Array(out[1]), ref[1]), relmax(Array(out[2]), ref[2]))
        println("  [$T] compiled vs plain-Array rel = $e")
        @test e ≤ 1e-5

        if T === Float64
            rng = MersenneTwister(3)
            c1 = randn(rng, size(a)); c2 = randn(rng, size(b))
            loss(x, y, w1, w2) = (o = F.sino_svd_denoise_bilateral((x, y), plan); sum(w1 .* o[1]) + sum(w2 .* o[2]))
            rc1, rc2 = Reactant.to_rarray(c1), Reactant.to_rarray(c2)
            _ts("SVD-bilateral $T: compile Enzyme.gradient")
            t0 = time()
            cargs = (ra, rb, Const(rc1), Const(rc2))
            grad_c = @compile Enzyme.gradient(Reverse, loss, cargs...)
            println("  [$T] gradient compile time: $(round(time() - t0; digits = 1)) s")
            g = grad_c(Reverse, loss, cargs...)
            ga = Array(g[1]); gb = Array(g[2])
            @test size(ga) == size(a) && size(gb) == size(b)
            @test all(isfinite, ga) && all(isfinite, gb)

            loss_h(x, y) = loss(x, y, c1, c2)
            h = 1e-6
            u = randn(MersenneTwister(21), size(a)); v = randn(MersenneTwister(22), size(b))
            dd_fd = (loss_h(a .+ h .* u, b .+ h .* v) - loss_h(a .- h .* u, b .- h .* v)) / (2h)
            dd_ad = sum(ga .* u) + sum(gb .* v)
            rel_dir = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
            println("  [$T] directional derivative: AD = $dd_ad, FD = $dd_fd, rel = $rel_dir")
            @test rel_dir ≤ 1e-3
            worst = 0.0
            for I in rand(MersenneTwister(5), CartesianIndices(a), 4)
                e1 = zeros(T, size(a)); e1[I] = h
                fd = (loss_h(a .+ e1, b) - loss_h(a .- e1, b)) / (2h)
                rel = abs(ga[I] - fd) / max(abs(fd), 1e-12)
                worst = max(worst, rel)
                println("  [$T] ∂loss/∂a$(Tuple(I)): AD = $(ga[I]), FD = $fd, rel = $rel")
            end
            @test worst ≤ 1e-3
        end
    end
end

# ----------------------------------------------------------------------------- 2. ACNR-Kalender
@testset "Reactant smoke: acnr_kalender" begin
    T = Float32
    W, I = synth_basis_pair(T)
    plan = F.ACNRKalenderPlan(T; hp_sigma_px = 1.5, window = 4, passes = 5, beta_max = 14.0)   # nb03 setting
    Wr, Ir, info_ref = F.acnr_kalender(W, I, plan)
    f(w, i) = F.acnr_kalender(w, i, plan)
    rW, rI = Reactant.to_rarray(W), Reactant.to_rarray(I)
    _ts("ACNR: compile")
    t0 = time()
    fc = @compile f(rW, rI)
    println("  compile time: $(round(time() - t0; digits = 1)) s")
    Wc, Ic, info = fc(rW, rI)
    e = max(relmax(Array(Wc), Wr), relmax(Array(Ic), Ir))
    println("  compiled vs plain-Array rel = $e   ρ_hp = $(Float64(info.ρ_hp)) (plain $(info_ref.ρ_hp))")
    @test e ≤ 1e-5
    @test abs(Float64(info.ρ_hp) - info_ref.ρ_hp) ≤ 1e-4
end

# ----------------------------------------------------------------------------- 3. median-z
@testset "Reactant smoke: median_z" begin
    T = Float32
    vol = T.(randn(MersenneTwister(5), 8, 6, 7))
    for boundary in (:replicate, :shrink)
        plan = F.MedianZPlan(; adjacent_slices = 1, boundary = boundary)
        ref = F.median_z(vol, plan)
        f(v) = F.median_z(v, plan)
        rv = Reactant.to_rarray(vol)
        _ts("median_z $boundary: compile")
        fc = @compile f(rv)
        out = Array(fc(rv))
        e = relmax(out, ref)
        println("  [$boundary] compiled vs plain-Array rel = $e")
        @test e ≤ 1e-5
    end
end

# ----------------------------------------------------------------------------- 4. SF-JSD
@testset "Reactant smoke: sfjsd_denoise" begin
    T = Float32
    a, b = synth_channels(T; n_col = 16, n_row = 2, n_view = 12, I0 = (2.0e4, 1.5e4))
    plan = F.SFJSDPlan([a, b], [2.0e4, 1.5e4]; σ₀ = 0.5)
    println("  captured: stride=$(plan.stride) n_iter=$(plan.n_iter) R=$(plan.radius_max) σ₀=$(plan.σ0_iter)")
    ref = F.sfjsd_denoise(a, b, plan)
    f(x, y) = F.sfjsd_denoise(x, y, plan)
    ra, rb = Reactant.to_rarray(a), Reactant.to_rarray(b)
    _ts("SF-JSD: compile")
    t0 = time()
    fc = @compile f(ra, rb)
    println("  compile time: $(round(time() - t0; digits = 1)) s")
    out = fc(ra, rb)
    e = max(relmax(Array(out[1]), ref[1]), relmax(Array(out[2]), ref[2]))
    println("  compiled vs plain-Array rel = $e")
    @test e ≤ 1e-5
end
_ts("done")
