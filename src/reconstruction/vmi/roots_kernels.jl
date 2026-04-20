"""
GPU-safe, allocation-free 1:1 port of Roots.jl's **Brent's method** for
bracketed root-finding in one variable.

Targets Roots.jl v3.0.0 source:
  • src/Bracketing/brent.jl   (update_state dispatch, mflag logic)
  • src/Bracketing/bisection.jl (_middle, __middle bit-level)
  • src/utils.jl              (secant_step, inverse_quadratic_step)
  • src/convergence.jl        (iszero_Δx_xexact, default tolerances)

The port is **line-for-line equivalent** to Roots.jl's Brent algorithm
(same mflag tracking, same guard-step bound `(3a+b)/4`, same fallback to
secant when inverse-quadratic is NaN/Inf, same bracket-update sign
tiebreak, same `nextfloat(a) == b` bit-exact termination).  The only
changes from the upstream source are:

  1. Flat function (no struct / dispatch tower) — lives in one call
     so it compiles into a single GPU kernel without virtual dispatch.
  2. Bounded `maxiters` (default 100) since GPU kernels cannot throw.
     Brent's bisection fallback guarantees convergence in ≤ 64 steps
     on Float64, so 100 is a safety margin.
  3. Returns `(root, converged)` tuple instead of raising on failure.

Bit-for-bit parity against Roots.jl is verified in
`test/vmi/test_brent_parity.jl` on randomised inputs.
"""

# ─────────────────────────────────────────────────────────────────────
# Bit-level midpoint helpers (Roots.jl bisection.jl lines 111-152).
# ─────────────────────────────────────────────────────────────────────

# Roots.jl __middle(x::Float64, y::Float64): reinterpret over UInt64,
# bit-shift >> 1, reinterpret back.  Deterministic, ≤ 64 bisection
# steps to machine precision.
@inline function __middle_bits(x::Float64, y::Float64)
    xint = reinterpret(UInt64, abs(x))
    yint = reinterpret(UInt64, abs(y))
    mid  = (xint + yint) >> 1
    sign(x + y) * reinterpret(Float64, mid)
end

@inline function __middle_bits(x::Float32, y::Float32)
    xint = reinterpret(UInt32, abs(x))
    yint = reinterpret(UInt32, abs(y))
    mid  = (xint + yint) >> 1
    sign(x + y) * reinterpret(Float32, mid)
end

# Roots.jl _middle(x, y): handle inf and opposite-sign inputs, else
# fall through to __middle_bits.
@inline function _middle_gpu(x::T, y::T) where {T<:AbstractFloat}
    a = isinf(x) ? nextfloat(x) : x
    b = isinf(y) ? prevfloat(y) : y
    if sign(a) * sign(b) < 0
        return zero(a)
    else
        return __middle_bits(a, b)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Core step formulas (Roots.jl utils.jl lines 74, 118).
# ─────────────────────────────────────────────────────────────────────

@inline secant_step_gpu(a, b, fa, fb) = a - fa * (b - a) / (fb - fa)

@inline function inverse_quadratic_step_gpu(a, b, c, fa, fb, fc)
    s = zero(a)
    s += a * fb * fc / (fa - fb) / (fa - fc)
    s += b * fa * fc / (fb - fa) / (fb - fc)
    s += c * fa * fb / (fc - fa) / (fc - fb)
    s
end

# ─────────────────────────────────────────────────────────────────────
# Brent's method — flat, GPU-safe, 1:1 with Roots.jl Brent.update_state.
# ─────────────────────────────────────────────────────────────────────

"""
    brent_solve(f, a, b; xabstol=eps(T), xreltol=eps(T), maxiters=100)

Find a root of `f` in the bracket `[a, b]` using Brent's method.

Returns `(root, converged::Bool)`.  On failure (non-finite value, or
maxiters hit without `nextfloat(a) == b`), `converged = false` and
`root` is the best of `(a, b)` by `|f|`.

Identical algorithm to `Roots.find_zero(f, (a, b), Roots.Brent())` for
Float64 / Float32 inputs.  Tolerances default to `eps(T)` (matches
Roots.jl `default_tolerances` at `convergence.jl:55-56`).

# Caller contract
- `f(a)` and `f(b)` must have opposite signs (bracket must be valid).
- `f` must be pure (no allocations, no I/O) for GPU execution.
"""
@inline function brent_solve(
        f,
        a::T,
        b::T;
        xabstol::T = eps(T) * oneunit(T),
        xreltol::T = eps(T),
        maxiters::Int = 100,
    ) where {T<:AbstractFloat}
    fa = f(a)
    fb = f(b)

    # Fast exits — match Roots.jl init_state behaviour.
    iszero(fa) && return (a, true)
    iszero(fb) && return (b, true)

    # Init ensures |fa| ≥ |fb| (Roots brent.jl init_state lines 27-29).
    if abs(fa) < abs(fb)
        a,  b  = b,  a
        fa, fb = fb, fa
    end

    # Invalid bracket: `fa` and `fb` must have opposite signs.
    if sign(fa) * sign(fb) > 0
        # Return best-of-(a,b) by |f|; flag as not converged.
        return (abs(fa) < abs(fb) ? a : b, false)
    end

    c  = a;   fc = fa                  # initial c = a per Brent init
    d  = c                             # d is dummy until second iter
    mflag = true

    for _ in 1:maxiters
        # ── Choose trial step: inverse quadratic, else secant ──
        s = if fa != fc && fb != fc
            inverse_quadratic_step_gpu(a, b, c, fa, fb, fc)
        else
            secant_step_gpu(a, b, fa, fb)
        end
        if isnan(s) || isinf(s)
            s = secant_step_gpu(a, b, fa, fb)
        end

        # ── Guard step (brent.jl lines 55-71) ──
        u, v = (3a + b) / 4, b
        if u > v
            u, v = v, u
        end

        tol = max(xabstol, max(abs(b), abs(c), abs(d)) * xreltol)
        force_bisect =
            !(u < s < v) ||
            (mflag  && abs(s - b) >= abs(b - c) / 2) ||
            (!mflag && abs(s - b) >= abs(c - d) / 2) ||
            (mflag  && abs(b - c) <= tol) ||
            (!mflag && abs(c - d) <= tol)

        if force_bisect
            s = _middle_gpu(a, b)
            mflag = true
        else
            mflag = false
        end

        fs = f(s)

        # Exact zero — done (brent.jl line 76).
        iszero(fs) && return (s, true)

        # Non-finite eval — abort with best bracket endpoint.
        if isnan(fs) || isinf(fs)
            return (abs(fa) < abs(fb) ? a : b, false)
        end

        # ── Bracket update (brent.jl lines 79-90) ──
        d  = c
        c  = b
        fc = fb
        if sign(fa) * sign(fs) < 0
            b  = s
            fb = fs
        else
            a  = s
            fa = fs
        end
        if abs(fa) < abs(fb)
            a,  b  = b,  a
            fa, fb = fb, fa
        end

        # ── Termination (convergence.jl iszero_Δx_xexact, Float64 path) ──
        # `nextfloat(min(a,b)) == max(a,b)` → bracket is one ULP wide.
        lo = min(a, b); hi = max(a, b)
        if nextfloat(lo) == hi
            return (abs(fa) < abs(fb) ? a : b, true)
        end
    end

    # maxiters exhausted.
    (abs(fa) < abs(fb) ? a : b, false)
end

export brent_solve
