"""
Parity tests: `BasisSimulator.brent_solve` vs `Roots.find_zero(..., Roots.Brent())`.

Roots.jl is the authoritative reference — our port must match its
trajectory **bit-exactly** up to tolerance on all bracketed inputs we
run through Cong's per-ray decomposition.  Rather than claim bit-exact
(fp reorderings make that flaky across targets), we assert:

    | brent_solve(f, a, b) - Roots.find_zero(f, (a, b), Brent()) | ≤ 4·eps · max(1, |root|)

which is ~4 ULP — tighter than any downstream consumer would notice.
"""

using Test
using BasisSimulator: brent_solve
import Roots
using Random

Random.seed!(0xC0FFEE)

const _REL_TOL = 4 * eps(Float64)

"Helper: compare our brent vs Roots.Brent on a single (f, a, b)."
function _parity_check(f, a, b)
    r_ours, ok_ours = brent_solve(f, Float64(a), Float64(b))
    @test ok_ours
    r_roots = Roots.find_zero(f, (Float64(a), Float64(b)), Roots.Brent())
    # |our - roots| ≤ 4 ULP (relative for non-zero roots, absolute otherwise)
    tol = _REL_TOL * max(1.0, abs(r_roots))
    @test abs(r_ours - r_roots) ≤ tol
    # Residual must also be essentially zero (both find the same root).
    @test abs(f(r_ours)) ≤ 1e-10
    (r_ours, r_roots)
end

@testset "Brent parity vs Roots.jl" begin

    # ────────────────────────────────────────────────────────────────
    # Classic Brent test functions from the literature.
    # ────────────────────────────────────────────────────────────────
    @testset "classical bracketed roots" begin
        _parity_check(x -> x^3 - 2x - 5,         2.0, 3.0)          # Wallis
        _parity_check(x -> x^2 - 2,              1.0, 2.0)          # √2
        _parity_check(x -> cos(x) - x,           0.0, 1.0)          # Dottie
        _parity_check(x -> sin(x),               3.0, 4.0)          # π
        _parity_check(x -> exp(x) - 3,           0.0, 2.0)
        _parity_check(x -> log(x) - 1,           1.0, 10.0)          # e
        _parity_check(x -> (x - 1) * (x - 2) * (x - 3), 0.0, 1.5)   # root at 1
        _parity_check(x -> (x - 1) * (x - 2) * (x - 3), 2.5, 3.5)   # root at 3
    end

    # ────────────────────────────────────────────────────────────────
    # CT-flavoured: Beer–Lambert integral shape (monotonic decreasing).
    # Matches Cong's outer Brent on water-T_L → L.
    # ────────────────────────────────────────────────────────────────
    @testset "Beer-Lambert water-L shape" begin
        ŵ  = [0.10, 0.18, 0.22, 0.22, 0.15, 0.08, 0.05]      # toy spectrum
        μ  = [0.28, 0.23, 0.20, 0.18, 0.16, 0.14, 0.12]      # ~water μ
        for T_meas in (0.9, 0.5, 0.1, 0.01, 1e-4)
            f = L -> sum(ŵ[i] * exp(-μ[i] * L) for i in eachindex(ŵ)) - T_meas
            _parity_check(f, 0.0, 100.0)
        end
    end

    # ────────────────────────────────────────────────────────────────
    # Randomised: quintic polynomials with known bracketed real roots.
    # Catches tie-breaks in the mflag / guard-step logic if the ports
    # diverge from Roots.jl.
    # ────────────────────────────────────────────────────────────────
    @testset "random quintics (200 cases)" begin
        for _ in 1:200
            # Build a degree-5 polynomial with a known real root in [-1, 1]
            x_root = rand() * 2 - 1
            c      = randn(5) .* 0.2                  # small-ish
            f      = x -> c[1]*x^5 + c[2]*x^4 + c[3]*x^3 + c[4]*x^2 + c[5]*x + 0.0 -
                         (c[1]*x_root^5 + c[2]*x_root^4 + c[3]*x_root^3 +
                          c[4]*x_root^2 + c[5]*x_root)
            # Find a bracket by expanding outward until sign flip.
            a, b = x_root - 0.1, x_root + 0.1
            while sign(f(a)) == sign(f(b)) && b - a < 10
                a -= 0.1
                b += 0.1
            end
            if sign(f(a)) != sign(f(b))
                _parity_check(f, a, b)
            end
        end
    end

    # ────────────────────────────────────────────────────────────────
    # Cong-shaped test: outer Brent on G(y) from Cong Eq 10.
    # ────────────────────────────────────────────────────────────────
    @testset "Cong outer Brent G(y)" begin
        ŵ_L = [0.08, 0.15, 0.22, 0.25, 0.18, 0.08, 0.04]
        p_L = [1.2e-5, 5.8e-6, 3.1e-6, 1.8e-6, 1.1e-6, 7.0e-7, 4.8e-7]
        q_L = [0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
        ŵ_H = [0.05, 0.10, 0.15, 0.20, 0.22, 0.18, 0.10]
        p_H = [1.2e-5, 5.8e-6, 3.1e-6, 1.8e-6, 1.1e-6, 7.0e-7, 4.8e-7]
        q_H = [0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26]

        T_H_pred = function (y, C)
            T = 0.0
            for i in eachindex(ŵ_H)
                T += ŵ_H[i] * exp(-p_H[i] * y - q_H[i] * C)
            end
            T
        end

        # A simplified stand-in where C is a known linear function of y.
        for T_H_meas in (0.8, 0.4, 0.08)
            G = y -> T_H_pred(y, 0.5 + 0.01 * y) - T_H_meas
            if sign(G(0.0)) != sign(G(1000.0))
                _parity_check(G, 0.0, 1000.0)
            end
        end
    end

    # ────────────────────────────────────────────────────────────────
    # Edge cases — fast exits.
    # ────────────────────────────────────────────────────────────────
    @testset "fast exits" begin
        # Root exactly at a.
        f1 = x -> x - 2.0
        r, ok = brent_solve(f1, 2.0, 5.0)
        @test ok && r == 2.0
        # Root exactly at b.
        r, ok = brent_solve(f1, -1.0, 2.0)
        @test ok && r == 2.0
        # Invalid bracket (same sign) — must NOT converge.
        f2 = x -> x^2 + 1.0
        r, ok = brent_solve(f2, 0.0, 1.0)
        @test !ok
    end
end
