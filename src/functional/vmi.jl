#=
Functional VMI stage — projection-domain two-material decomposition and
virtual-monoenergetic synthesis as pure, vectorized tensor programs.

Legacy oracles (numerics unchanged, semantics reproduced with masks):

  * `src/reconstruction/vmi/cong.jl`          → `cong_solve` / `cong_decompose`
  * `src/reconstruction/vmi/roots_kernels.jl` (Brent) → fixed-count vectorized
    bracketing (bisection) + masked Newton polish
  * `src/reconstruction/vmi/cmv.jl`           → `cmv_decompose`
  * `src/reconstruction/vmi/image_domain_decomp.jl` `synth_vmi_2basis` (the LIVE
    notebook synthesis) → `synth_vmi_2basis`
  * `src/reconstruction/vmi/vmi_synth.jl` `synth_vmi_hu` (deprecated photo/Compton
    pairing) → `synth_vmi_hu`

Conventions shared by every stage function in this file:

  * A *ray array* is any `AbstractArray{<:Any,N}` (e.g. `(n_col, n_row, n_view)`);
    every operation is broadcast over all rays at once.  The energy axis is
    appended as axis `N+1`, so a spectral sum is
    `sum(w .* exp.(-(μ .* x)); dims = N+1)` and each spectral moment is
    `sum(z .* k; dims = N+1)` against a host-built `(1,…,1,n_E)` moment table
    (no matmul: a host `Matrix` times a traced array falls back to a generic
    `Matrix{TracedRNumber}` under Reactant).
  * Nothing is mutated; nothing is scalar-indexed; there are no data-dependent
    loops.  Every legacy `if`/`break`/`return` becomes an `ifelse.` mask, and
    every root-find is a host-constant number of vectorized iterations.
  * Element type `T <: AbstractFloat` is taken ONLY from the plan
    (`plan::CongPlan{T}`), never from an array's eltype: under Reactant a
    `TracedRArray{T,N} <: AbstractArray{TracedRNumber{T},N}`, so array
    arguments are typed `AbstractArray{<:Any,N}` and all constants live in the
    immutable plan structs as `T` scalars (no Float64 literals in hot code).
  * Broadcast expressions are kept SHALLOW (binary/ternary) on purpose: XLA
    fuses them anyway, while on plain Julia every distinct deep fusion is a
    separate multi-second compile.  Polynomials go through `_horner`.
  * Plans are built on the host from config tables (legacy helpers are called
    there and only there).

Memory model: the spectral kernels are `O(n_rays · n_E)` temporaries.  XLA
fuses the broadcast into the reduction; on plain `Array`s they are
materialised, so tile over views when running large sinograms natively.
=#

@static if !isdefined(@__MODULE__, :BS)
    import ..BasisSimulator as BS
end

# ════════════════════════════════════════════════════════════════════════
#  Shape / broadcast helpers (host-side, static)
# ════════════════════════════════════════════════════════════════════════

# Append a singleton energy axis to a ray array: (S...) → (S..., 1).
@inline _ea(x::AbstractArray) = reshape(x, size(x)..., 1)

# 1-D energy table → (1,…,1, n_E) so it broadcasts against (S..., 1).
_energy_table(v::AbstractVector, N::Int) = reshape(v, ntuple(_ -> 1, N)..., length(v))

# Spectrum weights: 1-D shared → (1,…,1,n_E); 3-D per-ray (n_col,n_row,n_E) →
# (n_col, n_row, 1,…,1, n_E) — ray dims 1:2 must be (col,row), as in legacy.
function _weights_table(w::AbstractArray, N::Int)
    if ndims(w) == 1
        return reshape(w, ntuple(_ -> 1, N)..., length(w))
    elseif ndims(w) == 3
        N >= 2 || error("per-ray spectrum weights need ray arrays with ≥ 2 dims (col,row,…)")
        return reshape(w, size(w, 1), size(w, 2), ntuple(_ -> 1, N - 2)..., size(w, 3))
    else
        error("spectrum weights must be 1-D (shared) or 3-D (per-ray); got ndims=$(ndims(w))")
    end
end

# Σ over the trailing energy axis, back to ray shape S.
@inline _esum(z::AbstractArray, S::Dims) = reshape(sum(z; dims = length(S) + 1), S)

# Spectral moment against a (1,…,1,n_E) table: Σ_E z·k, back to ray shape S.
@inline _emoment(z::AbstractArray, ktab::AbstractArray, S::Dims) = _esum(z .* ktab, S)

# Ray-shaped zeros derived from data (traceable; no `similar`/`fill!`).
# `T` is always the plan's scalar type — never the array eltype.
@inline _zeros_like(x::AbstractArray, ::Type{T}) where {T} = zero(T) .* x
@inline _true_like(x::AbstractArray) = isfinite.(x)       # all-true for finite data

# Horner evaluation c₀ + x(c₁ + x(c₂ + …)) with a homogeneous coefficient tuple;
# one small broadcast type reused for every polynomial in the file.
@inline function _horner(x, cs::Tuple)
    acc = cs[end]
    for k in (length(cs) - 1):-1:1
        acc = cs[k] .+ x .* acc
    end
    acc
end

# Safe divisor: replace |d| < floor by 1 (masked lanes never use the result).
# `oneT` is a ray-shaped array of ones so every `ifelse.` in the file has the
# same (mask, array, array) signature (one compiled kernel, not a dozen).
@inline function _safe(d, floor, oneT)
    m = abs.(d) .< floor
    ifelse.(m, oneT, d)
end

# ════════════════════════════════════════════════════════════════════════
#  CongPlan
# ════════════════════════════════════════════════════════════════════════

"""
    CongPlan{T, AW, AV}

Immutable, host-built plan for [`cong_decompose`](@ref).  Holds the staged
spectral basis (`ŵ_L`, `p_L`, `q_L`, `ŵ_H`, `p_H`, `q_H` — as in
`CongWorkspace`), the water basis constants, every constant the legacy
kernel hard-codes (air gate, bracket limits, `y_max` factor/cap, Newton
tolerance, output clamps) and the fixed vectorized iteration counts that
replace the legacy data-dependent Brent/`while` loops:

  * `n_bisect_water`   bisection steps on the water anchor `L ∈ [L_lo, L_hi]`
  * `n_newton_water`   masked Newton polish steps on `L` (clamped to the bracket)
  * `n_newton_quintic` Newton steps on the Eq-8 quintic (legacy `newton_max_iter`, 12)
  * `n_expand`         geometric bracket expansions on `y` (legacy ≤ 24)
  * `n_bisect_y`       bisection steps on the outer root `G(y) = 0`
  * `n_newton_y`       masked Newton polish steps on `y` (clamped to the bracket)

`ŵ_L`/`ŵ_H` are 1-D (shared spectrum) or 3-D `(n_col, n_row, n_E)` (per-ray
bowtie) exactly like the legacy workspace.  Build with [`cong_plan`](@ref).
"""
struct CongPlan{T <: AbstractFloat, AW <: AbstractArray{T}, AV <: AbstractVector{T}}
    ŵ_L::AW; p_L::AV; q_L::AV
    ŵ_H::AW; p_H::AV; q_H::AV
    a_w::T; c_w::T
    p_L_min_safe::T      # max(min(p_L), eps(Float32))            (legacy y_max denominator)
    μ_w_min_safe::T      # max(min(p_L·a_w + q_L·c_w), 1e-4)       (legacy L_hi denominator)
    air_gate::T          # 5e-3  (symmetric |p_L|,|p_H| gate)
    L_lo::T              # -1, or 0 when legacy's W(-1) is non-finite (see cong_plan)
    L_hi_floor::T        # 60
    L_hi_factor::T       # 1.5
    y_max_factor::T      # 0.99
    y_max_cap::T         # 1e7
    y_start::T           # 0.01  (first positive bracket end)
    y_neg::T             # -0.1  (noise-scale negative window)
    newton_tol::T        # eps(Float32)
    dF_floor::T          # 1e-30
    clamp_lo::T          # -5
    clamp_hi::T          # 1e4
    n_bisect_water::Int
    n_newton_water::Int
    n_newton_quintic::Int
    n_expand::Int
    n_bisect_y::Int
    n_newton_y::Int
end

Base.eltype(::CongPlan{T}) where {T} = T

"""
    cong_plan(basis; water_basis, T = Float32,
              newton_max_iter = 12, newton_tol = eps(Float32),
              y_max_factor = 0.99, y_max_cap = 1f7,
              n_bisect_water = 30, n_newton_water = 2,
              n_expand = 24, n_bisect_y = 30, n_newton_y = 2) -> CongPlan{T}

Build a [`CongPlan`](@ref) from anything exposing `ŵ_L, p_L, q_L, ŵ_H, p_H,
q_H` — a legacy `CongWorkspace` (temporary bridge; device arrays are copied to
the host) or the raw basis NamedTuple.  `water_basis` is the legacy kwarg:
`(a = 0f0, c = 1f0)` for the LIVE material-direct `(μρ_iodine, μρ_water)`
basis (outputs are then iodine g/cm² and water g/cm²), or
`BS.water_basis_constants()` for the deprecated photo/Compton basis.

Legacy-derived constants (`p_L_min`, `μ_w_min`) are computed on the host from
the `T`-converted tables exactly as `apply_cong!` does.

Water-bracket lower end: legacy brackets `L ∈ [-1, L_hi]`.  With real spectra
the lowest bins have `ŵ = 0` and `μ_w > 88`, so legacy's `W(-1) = Σ ŵ·exp(μ_w)`
is `0·Inf = NaN` in Float32; the Roots-style Brent then takes its first forced
bisection step at `_middle(-1, L_hi) = 0` (opposite-sign endpoints) and
continues on the valid bracket `[0, L_hi]`.  That is reproduced here on the
host: if `W(L_lo)` is non-finite for any ray, `L_lo = 0`.
"""
function cong_plan(basis;
        water_basis,
        T::Type{<:AbstractFloat} = Float32,
        newton_max_iter::Int = 12,
        newton_tol::Real = eps(Float32),
        y_max_factor::Real = 0.99,
        y_max_cap::Real = 1f7,
        n_bisect_water::Int = 30,
        n_newton_water::Int = 2,
        n_expand::Int = 24,
        n_bisect_y::Int = 30,
        n_newton_y::Int = 2,
    )
    tohost(x) = T.(Array(x))
    ŵ_L = tohost(basis.ŵ_L); p_L = tohost(basis.p_L); q_L = tohost(basis.q_L)
    ŵ_H = tohost(basis.ŵ_H); p_H = tohost(basis.p_H); q_H = tohost(basis.q_H)
    ndims(ŵ_L) == ndims(ŵ_H) || error("cong_plan: ŵ_L and ŵ_H must share ndims")
    length(p_L) == length(q_L) == size(ŵ_L, ndims(ŵ_L)) || error("cong_plan: low-kVp table lengths disagree")
    length(p_H) == length(q_H) == size(ŵ_H, ndims(ŵ_H)) || error("cong_plan: high-kVp table lengths disagree")
    a_w = T(water_basis.a); c_w = T(water_basis.c)
    p_L_min_safe = max(T(minimum(p_L)), T(eps(Float32)))
    μ_w = p_L .* a_w .+ q_L .* c_w
    μ_w_min_safe = max(T(minimum(μ_w)), T(1f-4))
    L_lo = T(-1)
    W_lo = if ndims(ŵ_L) == 1
        sum(ŵ_L .* exp.(.-(μ_w .* L_lo)))
    else
        sum(ŵ_L .* reshape(exp.(.-(μ_w .* L_lo)), 1, 1, :); dims = 3)
    end
    all(isfinite, W_lo) || (L_lo = zero(T))
    CongPlan{T, typeof(ŵ_L), typeof(p_L)}(
        ŵ_L, p_L, q_L, ŵ_H, p_H, q_H, a_w, c_w,
        p_L_min_safe, μ_w_min_safe,
        T(5f-3), L_lo, T(60), T(1.5f0), T(y_max_factor), T(y_max_cap),
        T(0.01f0), T(-0.1f0), T(newton_tol), T(1f-30), T(-5), T(1f4),
        n_bisect_water, n_newton_water, newton_max_iter, n_expand, n_bisect_y, n_newton_y,
    )
end

# ── per-call expanded tables (host reshapes + host moment matrices) ─────

function _cong_tables(plan::CongPlan{T}, N::Int) where {T}
    p_L = plan.p_L; q_L = plan.q_L
    # KP[k+1] = (-q)^k / k!  for k = 0..6  (P_6 only feeds the Jacobian);
    # KQ[k+1] = p (-q)^k / k! for k = 0..5 (∂P_k/∂y = -Q_k).  Energy tables.
    fact = T[one(T); cumprod(T.(1:6))]                      # k! for k = 0..6
    KP = ntuple(k -> _energy_table((-q_L) .^ (k - 1) ./ fact[k], N), 7)
    KQ = ntuple(k -> _energy_table(p_L .* (-q_L) .^ (k - 1) ./ fact[k], N), 6)
    μ_w = p_L .* plan.a_w .+ q_L .* plan.c_w
    (
        ŵ_L = _weights_table(plan.ŵ_L, N), p_L = _energy_table(p_L, N), q_L = _energy_table(q_L, N),
        ŵ_H = _weights_table(plan.ŵ_H, N), p_H = _energy_table(plan.p_H, N), q_H = _energy_table(plan.q_H, N),
        μ_w = _energy_table(μ_w, N), KP = KP, KQ = KQ,
    )
end

# ── spectral primitives ────────────────────────────────────────────────

# z = w · exp(-(p·y + q·c))  over (S..., nE)
@inline function _zexp(w, p, q, y, c)
    e = p .* _ea(y) .+ q .* _ea(c)
    w .* exp.(.-e)
end
# Water-only low-kVp transmission  W(L) = Σ ŵ_L e^{-μ_w L}
@inline _water_z(L, tabs) = tabs.ŵ_L .* exp.(.-(tabs.μ_w .* _ea(L)))
@inline _water_T(L, tabs, S::Dims) = _esum(_water_z(L, tabs), S)
@inline function _water_T_dT(L, tabs, S::Dims)
    z = _water_z(L, tabs)
    (_esum(z, S), .-_esum(z .* tabs.μ_w, S))
end
@inline _zL(y, c̄, tabs) = _zexp(tabs.ŵ_L, tabs.p_L, tabs.q_L, y, c̄)
@inline _zH(y, C, tabs)  = _zexp(tabs.ŵ_H, tabs.p_H, tabs.q_H, y, C)

# Eq-8 quintic Newton from x = 0: fixed `n_newton_quintic` masked iterations
# reproducing legacy `abs(dF) < 1e-30 && break` and `abs(Δ) < tol && break`.
function _quintic_newton(P::NTuple{6, <:AbstractArray}, T_L, plan::CongPlan{T}) where {T}
    P0, P1, P2, P3, P4, P5 = P
    Fc  = (P0 .- T_L, P1, P2, P3, P4, P5)
    dFc = (P1, T(2) .* P2, T(3) .* P3, T(4) .* P4, T(5) .* P5)
    x = _zeros_like(P0, T)
    oneT = one(T) .+ x
    active = _true_like(P0)
    for _ in 1:plan.n_newton_quintic
        F  = _horner(x, Fc)
        dF = _horner(x, dFc)
        active = active .& (abs.(dF) .>= plan.dF_floor)
        Δ = F ./ ifelse.(active, dF, oneT)
        Δ = ifelse.(active, Δ, x .* zero(T))
        x = x .- Δ
        active = active .& (abs.(Δ) .>= plan.newton_tol)
    end
    x
end

@inline _P6(z, tabs, S) = ntuple(j -> _emoment(z, tabs.KP[j], S), 6)

# x = h(y) from the quintic at (y, c̄).
function _solve_quintic(y, c̄, T_L, tabs, plan::CongPlan, S::Dims)
    z = _zL(y, c̄, tabs)
    _quintic_newton(_P6(z, tabs, S), T_L, plan)
end

# G(y) = T_H(y, c̄ + h(y)) - T_H_meas
function _G(y, c̄, T_L, T_H, tabs, plan::CongPlan, S::Dims)
    x = _solve_quintic(y, c̄, T_L, tabs, plan, S)
    _esum(_zH(y, c̄ .+ x, tabs), S) .- T_H
end

# G(y), dG/dy (implicit-function derivative through the quintic), plus the
# pieces the IFT Jacobian needs: x, dF (=∂R2/∂x, floored), Qx (=Σ Q_k x^k =
# -∂R2/∂y), R2_L (=∂R2/∂L), H_y (=∂R3/∂y), H_C (=∂R3/∂C).
function _G_full(y, c̄, T_L, T_H, tabs, plan::CongPlan{T}, S::Dims) where {T}
    z  = _zL(y, c̄, tabs)
    P  = _P6(z, tabs, S)
    P6 = _emoment(z, tabs.KP[7], S)
    Q  = ntuple(j -> _emoment(z, tabs.KQ[j], S), 6)
    x  = _quintic_newton(P, T_L, plan)
    P1, P2, P3, P4, P5 = P[2], P[3], P[4], P[5], P[6]
    dFc  = (P1, T(2) .* P2, T(3) .* P3, T(4) .* P4, T(5) .* P5)
    dF   = _horner(x, dFc)
    Qx   = _horner(x, Q)
    R2_L = plan.c_w .* _horner(x, (dFc..., T(6) .* P6))
    zH  = _zH(y, c̄ .+ x, tabs)
    TH  = _esum(zH, S)
    H_y = .-_emoment(zH, tabs.p_H, S)
    H_C = .-_emoment(zH, tabs.q_H, S)
    dF_safe = _safe(dF, plan.dF_floor, one(T) .+ _zeros_like(x, T))
    dxdy = Qx ./ dF_safe                        # dx/dy = -F_y/F_x = Qx/dF
    G  = TH .- T_H
    dG = H_y .+ H_C .* dxdy
    (G = G, dG = dG, x = x, dF = dF_safe, Qx = Qx, R2_L = R2_L, H_y = H_y, H_C = H_C)
end

# ════════════════════════════════════════════════════════════════════════
#  cong_solve / cong_decompose
# ════════════════════════════════════════════════════════════════════════

"""
    cong_solve(p_L, p_H, plan::CongPlan) -> NamedTuple

Full vectorized Cong 2022 solve.  Returns the final basis line integrals
`a`, `c` (shape `size(p_L)`) plus the solver state needed by
[`cong_decompose_jacobian`](@ref) and diagnostics:

  * `L`      water-equivalent path (anchor root)
  * `y`      iodine/photoelectric root of `G(y)=0` (0 on non-root branches)
  * `x`      quintic correction at `y`
  * `air`, `ok_L`, `ymax_le0`, `ok_y`, `fixed_y`, `main` — the legacy branch masks
  * `n_exp_used` — number of active geometric expansions per ray (`T` array)
  * `G_res`  — residual `|G(y)|` at the returned root (main-path rays)
  * `W_res`  — residual of the water anchor equation

Legacy branch semantics (all reproduced exactly with masks):

  1. symmetric air gate `|p_L| < 5e-3 && |p_H| < 5e-3` → `(0, 0)`
  2. invalid water bracket (`W(L_lo) - T_L` and `W(L_hi) - T_L` same sign, or
     non-finite) → `(0, 0)`; with `L_lo = 0` (see [`cong_plan`](@ref)) this is
     also what makes every `p_L < 0` ray return `(0, 0)` exactly like legacy
  3. `y_max ≤ 0` → `(0, c̄)`
  4. `G(0) == 0` → `y = 0` ; `G(0) > 0` and `G(-0.1) ≥ 0` (or non-finite) → `y = 0`
  5. `G(0) < 0` and expansion never finds `G ≥ 0` (or non-finite) → `(a_w L, c_w L)`
  6. otherwise the root, clamped to `[-5, 1e4]` on both outputs.

The only semantic deviation from legacy: a non-finite `G` at an interior
bisection point marks the ray `!ok_y` (fallback 5); legacy Brent would abort
with the same fallback but from a different trajectory.  Not observed on any
fixture ray.
"""
function cong_solve(p_L::AbstractArray{<:Any, N}, p_H::AbstractArray{<:Any, N},
                    plan::CongPlan{T}) where {T, N}
    size(p_L) == size(p_H) || error("cong_solve: p_L and p_H must share shape")
    S    = size(p_L)
    tabs = _cong_tables(plan, N)
    zeroT = _zeros_like(p_L, T)
    oneT  = one(T) .+ zeroT

    # ── gates & transmissions ─────────────────────────────────────────
    air  = (abs.(p_L) .< plan.air_gate) .& (abs.(p_H) .< plan.air_gate)
    T_L  = exp.(.-p_L)
    T_H  = exp.(.-p_H)

    # ── Step 1: water anchor  W(L) = T_L  on [L_lo, L_hi] ───────────────
    lo = plan.L_lo .+ zeroT
    hi = max.(plan.L_hi_floor, (plan.L_hi_factor / plan.μ_w_min_safe) .* p_L)
    f_lo = _water_T(lo, tabs, S) .- T_L
    f_hi = _water_T(hi, tabs, S) .- T_L
    ok_L = isfinite.(f_lo) .& isfinite.(f_hi)
    ok_L = ok_L .& ((sign.(f_lo) .* sign.(f_hi)) .<= zero(T))
    for _ in 1:plan.n_bisect_water
        mid = (lo .+ hi) ./ 2
        fm  = _water_T(mid, tabs, S) .- T_L
        up  = sign.(fm) .== sign.(f_lo)
        lo  = ifelse.(up, mid, lo); f_lo = ifelse.(up, fm, f_lo)
        hi  = ifelse.(up, hi, mid); f_hi = ifelse.(up, f_hi, fm)
    end
    L = (lo .+ hi) ./ 2
    for _ in 1:plan.n_newton_water
        W, dW = _water_T_dT(L, tabs, S)
        step = (W .- T_L) ./ _safe(dW, plan.dF_floor, oneT)
        L = clamp.(L .- step, lo, hi)
    end
    W_res = _water_T(L, tabs, S) .- T_L
    c̄ = plan.c_w .* L

    # ── y_max gate ───────────────────────────────────────────────────
    y_max    = min.((plan.y_max_factor / plan.p_L_min_safe) .* p_L, plan.y_max_cap)
    ymax_le0 = y_max .<= zero(T)

    # ── Step 3: bracket for G(y) = 0 ─────────────────────────────────
    G0 = _G(zeroT, c̄, T_L, T_H, tabs, plan, S)
    zero_branch = G0 .== zero(T)
    pos_branch  = G0 .< zero(T)
    neg_branch  = G0 .> zero(T)

    # positive branch: geometric expansion from min(y_start, y_max)
    y_hi = min.(plan.y_start .+ zeroT, y_max)
    G_hi = _G(y_hi, c̄, T_L, T_H, tabs, plan, S)
    n_exp_used = zeroT
    for _ in 1:plan.n_expand
        expanding = pos_branch .& isfinite.(G_hi)
        expanding = expanding .& (G_hi .< zero(T))
        expanding = expanding .& (y_hi .< y_max)
        y2    = min.(T(2) .* y_hi, y_max)
        y_new = ifelse.(expanding, y2, y_hi)
        G_new = _G(y_new, c̄, T_L, T_H, tabs, plan, S)
        G_hi  = ifelse.(expanding, G_new, G_hi)
        y_hi  = y_new
        n_exp_used = n_exp_used .+ ifelse.(expanding, oneT, zeroT)
    end
    pos_ok = pos_branch .& isfinite.(G_hi)
    pos_ok = pos_ok .& (G_hi .>= zero(T))

    # negative branch: noise-scale window [y_neg, 0]
    y_neg = plan.y_neg .+ zeroT
    G_lo  = _G(y_neg, c̄, T_L, T_H, tabs, plan, S)
    neg_ok   = neg_branch .& isfinite.(G_lo)
    neg_ok   = neg_ok .& (G_lo .< zero(T))
    neg_skip = neg_branch .& .!neg_ok

    bisect = pos_ok .| neg_ok
    lo_y = ifelse.(neg_ok, y_neg, zeroT)
    hi_y = ifelse.(pos_ok, y_hi, zeroT)
    f_lo_y = ifelse.(neg_ok, G_lo, G0)
    f_hi_y = ifelse.(pos_ok, G_hi, G0)
    bad = .!isfinite.(G0)
    for _ in 1:plan.n_bisect_y
        mid = (lo_y .+ hi_y) ./ 2
        fm  = _G(mid, c̄, T_L, T_H, tabs, plan, S)
        bad = bad .| (bisect .& .!isfinite.(fm))
        up  = sign.(fm) .== sign.(f_lo_y)
        lo_y = ifelse.(up, mid, lo_y); f_lo_y = ifelse.(up, fm, f_lo_y)
        hi_y = ifelse.(up, hi_y, mid); f_hi_y = ifelse.(up, f_hi_y, fm)
    end
    y = (lo_y .+ hi_y) ./ 2
    for _ in 1:plan.n_newton_y
        g = _G_full(y, c̄, T_L, T_H, tabs, plan, S)
        step = g.G ./ _safe(g.dG, plan.dF_floor, oneT)
        y = clamp.(y .- step, lo_y, hi_y)
    end
    y = ifelse.(bisect, y, zeroT)              # fixed-y branches sit at exactly 0
    ok_y    = zero_branch .| neg_skip
    ok_y    = ok_y .| (bisect .& .!bad)
    fixed_y = zero_branch .| neg_skip

    # ── final quintic at the root, clamps, branch selection ───────────
    x_final = _solve_quintic(y, c̄, T_L, tabs, plan, S)
    G_res   = abs.(_esum(_zH(y, c̄ .+ x_final, tabs), S) .- T_H)
    a_main  = clamp.(y, plan.clamp_lo, plan.clamp_hi)
    c_main  = clamp.(c̄ .+ x_final, plan.clamp_lo, plan.clamp_hi)
    a_fb    = plan.a_w .* L
    c_fb    = plan.c_w .* L

    a = ifelse.(ok_y, a_main, a_fb)
    c = ifelse.(ok_y, c_main, c_fb)
    a = ifelse.(ymax_le0, zeroT, a)
    c = ifelse.(ymax_le0, c̄, c)
    dead = air .| .!ok_L
    a = ifelse.(dead, zeroT, a)
    c = ifelse.(dead, zeroT, c)
    main = .!dead .& .!ymax_le0
    main = main .& ok_y

    (a = a, c = c, L = L, y = y, x = x_final,
     air = air, ok_L = ok_L, ymax_le0 = ymax_le0, ok_y = ok_y, fixed_y = fixed_y, main = main,
     n_exp_used = n_exp_used, G_res = G_res, W_res = W_res,
     lo_y = lo_y, hi_y = hi_y)
end

"""
    cong_decompose(p_L, p_H, plan::CongPlan) -> (a, c)

Pure, vectorized Cong 2022 two-material projection-domain decomposition —
the functional twin of `apply_cong!`.  `p_L`, `p_H` are measured low/high-kVp
log line integrals of any (shared) shape; returns basis line integrals `a`
(first basis: iodine g/cm² with the live material-direct plan) and `c`
(second basis: water g/cm²) of the same shape.  See [`cong_solve`](@ref) for
the branch semantics and [`cong_plan`](@ref) for the fixed iteration counts.
"""
function cong_decompose(p_L::AbstractArray, p_H::AbstractArray, plan::CongPlan)
    s = cong_solve(p_L, p_H, plan)
    (s.a, s.c)
end

# ════════════════════════════════════════════════════════════════════════
#  Implicit-function-theorem Jacobian / VJP
# ════════════════════════════════════════════════════════════════════════

"""
    cong_decompose_jacobian(p_L, p_H, plan; state = cong_solve(p_L, p_H, plan))
        -> (J_aL, J_aH, J_cL, J_cH)

Per-ray Jacobian `∂(a, c)/∂(p_L, p_H)` of [`cong_decompose`](@ref) by the
implicit-function theorem, with the same vectorized primitives (one 3×3
lower-triangular solve per ray, done as a scalar solve for `L` followed by a
2×2 solve for `(x, y)`).

At the root the solver state `z = (L, x, y)` satisfies `R(z; p) = 0` with

    R1 = Σ ŵ_L e^{-μ_w L}                       - e^{-p_L}
    R2 = Σ_k P_k(y, c̄) x^k,  c̄ = c_w L           - e^{-p_L}
    R3 = Σ ŵ_H e^{-p_H y - q_H (c̄ + x)}          - e^{-p_H}

so `∂z/∂p = -J_z⁻¹ J_p` and `(a, c) = (y, c_w L + x)`.  With
`W1 = ∂R1/∂L`, `dF = ∂R2/∂x`, `Qx = -∂R2/∂y`, `R2_L = ∂R2/∂L`,
`H_y = ∂R3/∂y`, `H_C = ∂R3/∂C`:

    dL = -e^{-p_L} dp_L / W1
    [dF  -Qx] [dx]   = -[e^{-p_L} dp_L + R2_L dL ]
    [H_C  H_y] [dy]     [e^{-p_H} dp_H + c_w H_C dL]
    da = dy,  dc = c_w dL + dx

Non-root branches are handled by the same masks as the forward solve:
air / invalid-water-bracket rays → 0; `y_max ≤ 0` → `dc = c_w dL`;
fallback (`!ok_y`) → `(a_w dL, c_w dL)`; fixed-`y` branches (`G(0)=0`, or
`G(0)>0` without a negative-window sign change) → `dy = 0`,
`dx = -(e^{-p_L} dp_L + R2_L dL)/dF`; active output clamps → 0.

This is the VJP to use at stage boundaries (Reactant cannot register custom
rules inside a trace) and the reference that Enzyme's unrolled reverse pass
must agree with.
"""
function cong_decompose_jacobian(p_L::AbstractArray{<:Any, N}, p_H::AbstractArray{<:Any, N},
                                 plan::CongPlan{T};
                                 state = cong_solve(p_L, p_H, plan)) where {T, N}
    S    = size(p_L)
    tabs = _cong_tables(plan, N)
    zeroT = _zeros_like(p_L, T)
    oneT  = one(T) .+ zeroT
    T_L  = exp.(.-p_L)
    T_H  = exp.(.-p_H)
    L = state.L; y = state.y
    c̄ = plan.c_w .* L

    # water anchor derivative
    _, dW = _water_T_dT(L, tabs, S)
    dL_dpL = .-T_L ./ _safe(dW, plan.dF_floor, oneT)   # dL/dp_L ; dL/dp_H = 0

    g = _G_full(y, c̄, T_L, T_H, tabs, plan, S)
    m11 = g.dF; m12 = .-g.Qx; m21 = g.H_C; m22 = g.H_y
    det = m11 .* m22 .- m12 .* m21
    det_safe = _safe(det, plan.dF_floor, oneT)

    # column p_L : dp_L = 1, dp_H = 0
    r1 = .-(T_L .+ g.R2_L .* dL_dpL)
    r2 = .-(plan.c_w .* g.H_C .* dL_dpL)
    dx_L = (m22 .* r1 .- m12 .* r2) ./ det_safe
    dy_L = (m11 .* r2 .- m21 .* r1) ./ det_safe
    # column p_H : dp_L = 0, dp_H = 1  (r1 = 0, r2 = -T_H)
    dx_H = (m12 .* T_H) ./ det_safe
    dy_H = .-(m11 .* T_H) ./ det_safe

    # fixed-y branches: dy = 0, dx = r1 / dF
    dx_fix = r1 ./ g.dF
    dx_L = ifelse.(state.fixed_y, dx_fix, dx_L)
    dy_L = ifelse.(state.fixed_y, zeroT, dy_L)
    dx_H = ifelse.(state.fixed_y, zeroT, dx_H)
    dy_H = ifelse.(state.fixed_y, zeroT, dy_H)

    # output clamps (main path)
    c_unc = c̄ .+ state.x
    a_open = (y .> plan.clamp_lo) .& (y .< plan.clamp_hi)
    c_open = (c_unc .> plan.clamp_lo) .& (c_unc .< plan.clamp_hi)
    J_aL = ifelse.(a_open, dy_L, zeroT)
    J_aH = ifelse.(a_open, dy_H, zeroT)
    J_cL = ifelse.(c_open, plan.c_w .* dL_dpL .+ dx_L, zeroT)
    J_cH = ifelse.(c_open, dx_H, zeroT)

    # fallback / gate branches
    fb = .!state.ok_y
    J_aL = ifelse.(fb, plan.a_w .* dL_dpL, J_aL)
    J_aH = ifelse.(fb, zeroT, J_aH)
    J_cL = ifelse.(fb, plan.c_w .* dL_dpL, J_cL)
    J_cH = ifelse.(fb, zeroT, J_cH)
    J_aL = ifelse.(state.ymax_le0, zeroT, J_aL)
    J_aH = ifelse.(state.ymax_le0, zeroT, J_aH)
    J_cL = ifelse.(state.ymax_le0, plan.c_w .* dL_dpL, J_cL)
    J_cH = ifelse.(state.ymax_le0, zeroT, J_cH)
    dead = state.air .| .!state.ok_L
    J_aL = ifelse.(dead, zeroT, J_aL)
    J_aH = ifelse.(dead, zeroT, J_aH)
    J_cL = ifelse.(dead, zeroT, J_cL)
    J_cH = ifelse.(dead, zeroT, J_cH)
    (J_aL, J_aH, J_cL, J_cH)
end

"""
    cong_decompose_ift_vjp(p_L, p_H, ā, c̄, plan; state = cong_solve(p_L, p_H, plan))
        -> (p̄_L, p̄_H)

Vector-Jacobian product of [`cong_decompose`](@ref): given output cotangents
`(ā, c̄)` returns input cotangents `(p̄_L, p̄_H) = Jᵀ (ā, c̄)` with `J` from
[`cong_decompose_jacobian`](@ref) (implicit-function theorem at the root).
"""
function cong_decompose_ift_vjp(p_L::AbstractArray, p_H::AbstractArray,
                                ā::AbstractArray, c̄::AbstractArray, plan::CongPlan;
                                state = cong_solve(p_L, p_H, plan))
    J_aL, J_aH, J_cL, J_cH = cong_decompose_jacobian(p_L, p_H, plan; state = state)
    (ā .* J_aL .+ c̄ .* J_cL, ā .* J_aH .+ c̄ .* J_cH)
end

# ════════════════════════════════════════════════════════════════════════
#  CMV — linear 2×2 baseline
# ════════════════════════════════════════════════════════════════════════

"""
    CMVPlan{T}

Spectrum-effective 2×2 mass-attenuation matrix for [`cmv_decompose`](@ref):
`μ̄_I_L, μ̄_W_L, μ̄_I_H, μ̄_W_H` and `inv_det`, rounded to `T` exactly as
`apply_cmv!` rounds them to Float32.  Build with [`cmv_plan`](@ref).
"""
struct CMVPlan{T <: AbstractFloat}
    μ̄_I_L::T; μ̄_W_L::T; μ̄_I_H::T; μ̄_W_H::T; inv_det::T
end

"""
    cmv_plan(basis; T = Float32) -> CMVPlan{T}

Host-side port of the `apply_cmv!` preamble: collapse a 3-D per-ray `ŵ` to
its centered ray, renormalise, and form the effective matrix in Float64.
"""
function cmv_plan(basis; T::Type{<:AbstractFloat} = Float32)
    ŵ_L_raw = Array(basis.ŵ_L); ŵ_H_raw = Array(basis.ŵ_H)
    ŵ_L_1d, ŵ_H_1d = if ndims(ŵ_L_raw) == 3
        nc = size(ŵ_L_raw, 1); nr = size(ŵ_L_raw, 2)
        mc = nc ÷ 2 + 1;       mr = nr ÷ 2 + 1
        (Float64.(ŵ_L_raw[mc, mr, :]), Float64.(ŵ_H_raw[mc, mr, :]))
    else
        (Float64.(ŵ_L_raw), Float64.(ŵ_H_raw))
    end
    ŵ_L_1d ./= sum(ŵ_L_1d); ŵ_H_1d ./= sum(ŵ_H_1d)
    p_L = Float64.(Array(basis.p_L)); q_L = Float64.(Array(basis.q_L))
    p_H = Float64.(Array(basis.p_H)); q_H = Float64.(Array(basis.q_H))
    μ̄_I_L = sum(ŵ_L_1d .* p_L); μ̄_W_L = sum(ŵ_L_1d .* q_L)
    μ̄_I_H = sum(ŵ_H_1d .* p_H); μ̄_W_H = sum(ŵ_H_1d .* q_H)
    det_M = μ̄_W_L * μ̄_I_H - μ̄_I_L * μ̄_W_H
    abs(det_M) < eps(Float64) * 1e3 && error("cmv_plan: 2×2 effective matrix is singular (det = $det_M).")
    CMVPlan{T}(T(μ̄_I_L), T(μ̄_W_L), T(μ̄_I_H), T(μ̄_W_H), T(1.0 / det_M))
end

"""
    cmv_decompose(p_L, p_H, plan::CMVPlan) -> (a_iodine, a_water)

Linear DE decomposition with ReLU projection — the functional twin of
`apply_cmv!` (same operation order, so bit-parity at matching `T`).
"""
function cmv_decompose(p_L::AbstractArray, p_H::AbstractArray, plan::CMVPlan{T}) where {T}
    a_w = plan.inv_det .* ( plan.μ̄_I_H .* p_L .- plan.μ̄_I_L .* p_H)
    a_i = plan.inv_det .* (.-plan.μ̄_W_H .* p_L .+ plan.μ̄_W_L .* p_H)
    (max.(a_i, zero(T)), max.(a_w, zero(T)))
end

# ════════════════════════════════════════════════════════════════════════
#  VMI synthesis — live 2-basis form (image_domain_decomp.jl)
# ════════════════════════════════════════════════════════════════════════

"""
    VMI2BasisPlan{T}

Per-energy constants for [`synth_vmi_2basis`](@ref): `energies` and
`α_E = μρ_iodine(E)/μρ_water(E)` (formed in Float64 and rounded to `T`, exactly
as `synth_vmi_2basis!` rounds to Float32).  Build with [`vmi_2basis_plan`](@ref).
"""
struct VMI2BasisPlan{T <: AbstractFloat}
    energies::Vector{T}
    α_E::Vector{T}
end

"""
    vmi_2basis_plan(energies; T = Float32,
                    water_material = BS.XA.Materials.water,
                    iodine_material = BS.XA.Elements.Iodine) -> VMI2BasisPlan{T}
"""
function vmi_2basis_plan(energies::AbstractVector;
        T::Type{<:AbstractFloat} = Float32,
        water_material  = BS.XA.Materials.water,
        iodine_material = BS.XA.Elements.Iodine,
    )
    Es = Float64.(energies)
    α = T[T(BS.compute_mass_μ_at_energy(iodine_material, E) / BS.compute_mass_μ_at_energy(water_material, E)) for E in Es]
    VMI2BasisPlan{T}(T.(Es), α)
end

"""
    synth_vmi_2basis(c_water, c_iodine, k::Int, plan::VMI2BasisPlan) -> HU_E
    synth_vmi_2basis(c_water, c_iodine, plan::VMI2BasisPlan) -> Vector of HU volumes

The LIVE notebook VMI synthesis (nb03/04/07/08/09/12), functional twin of
`synth_vmi_2basis!`:

    HU(E) = 1000·(c_water − 1) + c_iodine · α_E,   α_E = μρ_I(E)/μρ_w(E)

`c_water` in g/mL, `c_iodine` in mg/mL (the notebooks pass `iodine .* 1000f0`).
Same operation order as legacy, so bit-parity at matching `T`.
"""
function synth_vmi_2basis(c_water::AbstractArray, c_iodine::AbstractArray, k::Int, plan::VMI2BasisPlan{T}) where {T}
    T(1000) .* (c_water .- one(T)) .+ c_iodine .* plan.α_E[k]
end
synth_vmi_2basis(c_water::AbstractArray, c_iodine::AbstractArray, plan::VMI2BasisPlan) =
    [synth_vmi_2basis(c_water, c_iodine, k, plan) for k in eachindex(plan.energies)]

# ════════════════════════════════════════════════════════════════════════
#  VMI synthesis — deprecated photo/Compton form (vmi_synth.jl)
# ════════════════════════════════════════════════════════════════════════

"""
    VMISynthPlan{T, AM}

Per-energy synthesis constants for [`synth_vmi_hu`](@ref): `energies`, basis
attenuations `p_E`, `q_E` (`μ(E) = p_E·a + q_E·c`), reference water `μ_w_E`,
and the FOV mask as a precomputed `Bool` tensor `(nx, ny, 1)` (all-true when
masking is disabled).  Build with [`vmi_synth_plan`](@ref).
"""
struct VMISynthPlan{T <: AbstractFloat, AM <: AbstractArray{Bool}}
    energies::Vector{T}
    p_E::Vector{T}
    q_E::Vector{T}
    μ_w_E::Vector{T}
    fov_mask::AM
    hu_air::T
end

"""
    vmi_synth_plan(energies, (nx, ny); T = Float32, basis = :photo_compton,
                   fov_mask_radius_frac = 0.5,
                   μ_water_fn = E -> BS.compute_μ_at_energy(BS.XA.Materials.water, E))

`basis = :photo_compton` reproduces legacy `synth_vmi_hu`
(`p = p_photoelectric(E)`, `q = q_compton(E)`); `basis = :material` uses the
material-direct `(μρ_iodine(E), μρ_water(E))` pair.  The FOV mask follows the
legacy rule `(i - cx)^2 + (j - cy)^2 > (frac·nx)^2 → -1000 HU`, `cx = (nx+1)/2`;
pass `fov_mask_radius_frac = nothing` to disable.
"""
function vmi_synth_plan(energies::AbstractVector, matrix_xy::Tuple{Int, Int};
        T::Type{<:AbstractFloat} = Float32,
        basis::Symbol = :photo_compton,
        fov_mask_radius_frac = 0.5,
        μ_water_fn = (E -> BS.compute_μ_at_energy(BS.XA.Materials.water, Float64(E))),
    )
    Es = Float64.(energies)
    if basis === :photo_compton
        p_E = T[T(BS.p_photoelectric(E)) for E in Es]
        q_E = T[T(BS.q_compton(E)) for E in Es]
    elseif basis === :material
        p_E = T[T(BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)) for E in Es]
        q_E = T[T(BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)) for E in Es]
    else
        error("vmi_synth_plan: basis must be :photo_compton or :material")
    end
    μ_w_E = T[T(μ_water_fn(E)) for E in Es]
    nx, ny = matrix_xy
    cx, cy = (nx + 1) / 2, (ny + 1) / 2
    mask = if fov_mask_radius_frac === nothing
        trues(nx, ny, 1)
    else
        r2 = (Float64(fov_mask_radius_frac) * nx)^2
        reshape(Bool[(i - cx)^2 + (j - cy)^2 <= r2 for i in 1:nx, j in 1:ny], nx, ny, 1)
    end
    VMISynthPlan{T, typeof(mask)}(T.(Es), p_E, q_E, μ_w_E, mask, T(-1000))
end

"""
    synth_vmi_hu(a, c, k::Int, plan::VMISynthPlan) -> hu
    synth_vmi_hu(a, c, plan::VMISynthPlan) -> Vector of hu volumes (one per energy)

Image-domain VMI synthesis `μ = p_E·a + q_E·c → HU = 1000 (μ - μ_w)/μ_w`,
voxels outside the FOV mask set to -1000 — the functional twin of the
deprecated legacy `synth_vmi_hu` (which forms HU in Float64 and rounds; here
HU is formed in `T`, differing at the 1-ulp level).
"""
function synth_vmi_hu(a::AbstractArray, c::AbstractArray, k::Int, plan::VMISynthPlan{T}) where {T}
    p_E = plan.p_E[k]; q_E = plan.q_E[k]; μ_w = plan.μ_w_E[k]
    μ  = p_E .* a .+ q_E .* c
    hu = T(1000) .* (μ .- μ_w) ./ μ_w
    ifelse.(plan.fov_mask, hu, plan.hu_air)
end
synth_vmi_hu(a::AbstractArray, c::AbstractArray, plan::VMISynthPlan) =
    [synth_vmi_hu(a, c, k, plan) for k in eachindex(plan.energies)]
