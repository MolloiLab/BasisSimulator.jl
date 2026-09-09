# =============================================================================
# BasisSimulator.Functional — energy-integrating (EICT) detector chain
# =============================================================================
#
# PURE, mutation-free, array-generic re-implementation of the EICT signal chain
# from per-material PATH LENGTHS to the calibrated log sinogram (optionally BHC
# corrected).  Every function here is a pure function of arrays + host scalars
# (no RNG, no `push!`, no data-dependent shapes/branches, no scalar indexing),
# so the whole chain can be traced by Reactant.jl and differentiated by Enzyme.
#
# Numerical oracle (must match to the tolerances in test/functional/test_eict.jl):
#
#   * `src/projection/dd_fast.jl:163-175`  — cell-level spectral conversion
#         I_total = Σ_e wη[e] · bt[col,row,e] · exp(-Σ_m P_m · μ[m,e])
#         sino    = -log(max(I_total, 1e-10))
#   * `src/api/driver.jl` `simulate!(::EICTWorkspace)` lines ~423-646 — every
#     post-projection step, in this order:
#         STEP 2a  apply_fill_factor!        p += -log(ff_eff)          (fill_factor.jl:88)
#         STEP 2b  estimate_scatter_field! + inject_scatter!            (scatter.jl:317,712)
#         STEP 3   reparameterised noise kernel (+ scatter subtraction) (driver.jl:522-578)
#         STEP 4   q = exp(-clamp(p,-1,15)); q /= max(air_ref,eps);
#                  low_signal_correction_gpu!; p = -log(max(q,eps));
#                  p += log(ff_eff)                                     (driver.jl:583-642)
#     `simulate!` STOPS there: BHC is decoupled ("applied at notebook level"),
#     so `eict_chain` reproduces `ws.sinogram` exactly and applies BHC only
#     when the plan carries polynomial coefficients.
#   * `src/correction/bhc_sinogram.jl:296-342` `apply_bhc!` — per-column
#     ascending-power polynomial (running-power accumulation).
#
# Deviations from legacy semantics (all documented in the final report):
#   * focal-spot blur, optical crosstalk, detector lag — NOT reproduced; the
#     plan constructor throws if the workspace has them enabled.
#   * scatter estimate: only the separable `:gaussian` kernel path (the one the
#     workspace precomputes); the 2-D `:exponential` fallback is refused.
#   * `low_signal_correction_gpu!` (≤0 → eps) followed by `-log(max(·,eps))`
#     is reproduced as the single, exactly equivalent `-log(max(·,eps))`.
#   * the legacy kernels iterate over the ZERO-PADDED energy axis (multiple of
#     16); the pads contribute exactly 0, so the plan stores the unpadded n_E.
#
# Conventions: sinograms are `(n_col, n_row, n_view)`; path lengths are
# `(n_col, n_row, n_view, n_mat)` in cm; `μ_tbl` is `(n_mat, n_E)` in cm⁻¹;
# `wη` is `(n_E,)`; the bowtie/heel spectral weight `bt` is
# `(n_col, n_row, n_E)`; the air reference is `(n_col, n_row)`.
#
# Element types: array ELEMENT types are left unconstrained so traced arrays
# (whose `eltype` is a wrapper such as `Reactant.TracedRNumber{Float32}`) flow
# through unchanged; the HOST scalar type `T` (Float32/Float64) comes from the
# plan or from `_scalar_type(array)`, and every literal is written `T(...)` so
# Float32 programs stay Float32.
#
# This file expects the enclosing module to provide `BS = BasisSimulator`
# (`const BS = BasisSimulator` or `import ..BasisSimulator as BS`).
# =============================================================================

# Host scalar type of an array: `Float32` for `Array{Float32}`, and for wrapped
# element types (e.g. `TracedRNumber{Float32}`) the wrapped float.  Pure type
# computation — resolved on the host, never inside array code.
_scalar_type(::Type{S}) where {S <: AbstractFloat} = S
function _scalar_type(::Type{S}) where {S <: Number}
    ps = S.parameters
    return (length(ps) >= 1 && ps[1] isa Type && ps[1] <: AbstractFloat) ? ps[1] : Float64
end
_scalar_type(x::AbstractArray) = _scalar_type(eltype(x))

const _A3 = AbstractArray{<:Any, 3}
const _A4 = AbstractArray{<:Any, 4}

# -----------------------------------------------------------------------------
# 1. Spectral conversion: per-material path lengths → polychromatic log sinogram
# -----------------------------------------------------------------------------

"""
    poly_log_sinogram(P, μ_tbl, wη, bt) -> sino

Polychromatic Beer–Lambert log sinogram from per-material path lengths — the
cell-level energy loop of `dd_fast.jl:163-175` written over the whole detector:

    L      = reshape(P, n_cells, n_mat) * μ_tbl          # (n_cells, n_E)
    I      = Σ_e wη[e] · bt[col,row,e] · exp(-L[:, e])
    sino   = -log(max(I, 1e-10))

`bt` is either `nothing` (no bowtie/heel weighting; the sum becomes the
matrix–vector product `exp.(-L) * wη`) or a `(n_col, n_row, n_E)` array
(dispatch on the type is resolved on the host).

Memory: the transient `exp.(-L)` is `n_cells × n_E` elements (e.g. 736·16·720
cells × 234 energies × 4 B ≈ 7.9 GB for a clinical scan).  Use
[`poly_log_sinogram_chunked`](@ref) to bound it by a static view-chunk count.
"""
function poly_log_sinogram(P::_A4, μ_tbl::AbstractMatrix, wη::AbstractVector, bt::Union{Nothing, _A3})
    T = _scalar_type(P)
    n_col, n_row, n_view, n_mat = size(P)
    L = reshape(P, :, n_mat) * μ_tbl                 # (n_cells, n_E)
    E = exp.(-L)                                     # (n_cells, n_E)
    I = _spectral_sum(E, wη, bt, n_col, n_row, n_view)
    return reshape(-log.(max.(I, T(1.0e-10))), n_col, n_row, n_view)
end

# Σ_e wη[e]·exp(-L_e)  — no bowtie: a single matrix–vector product.
_spectral_sum(E::AbstractMatrix, wη::AbstractVector, ::Nothing, n_col, n_row, n_view) = E * wη

# Σ_e (wη[e]·bt[col,row,e])·exp(-L_e) — legacy product order `(wη*bt)*exp`; per detector pixel a
# dot over the energies, i.e. a contraction batched over (col, row) (`_bmm_spectral`: the host keeps
# the legacy broadcast-and-sum; the device hook is one `dot_general`, whose reverse is a contraction).
function _spectral_sum(E::AbstractMatrix, wη::AbstractVector, bt::_A3, n_col, n_row, n_view)
    n_E = length(wη)
    E4 = reshape(E, n_col, n_row, n_view, n_E)
    W3 = reshape(wη, 1, 1, n_E) .* bt                    # (n_col, n_row, n_E)
    return vec(_bmm_spectral(E4, W3))
end

"""
    poly_log_sinogram_chunked(P, μ_tbl, wη, bt, ::Val{K}) -> sino

Same result as [`poly_log_sinogram`](@ref) with the transient `n_cells × n_E`
buffer bounded to `n_cells/K × n_E`: the view axis is split into `K` STATIC,
equal chunks (host-constant ranges; `n_view % K == 0` is required) and the
chunks are concatenated.  `K` is a compile-time constant so the program stays
static-shaped.
"""
function poly_log_sinogram_chunked(P::_A4, μ_tbl::AbstractMatrix, wη::AbstractVector,
        bt::Union{Nothing, _A3}, ::Val{K}) where {K}
    n_view = size(P, 3)
    n_view % K == 0 || throw(ArgumentError("poly_log_sinogram_chunked: n_view=$n_view must be a multiple of K=$K"))
    nv = n_view ÷ K
    parts = ntuple(k -> poly_log_sinogram(P[:, :, ((k - 1) * nv + 1):(k * nv), :], μ_tbl, wη, bt), Val(K))
    return cat(parts...; dims = 3)
end

"""
    poly_log_sinogram_looped(P, μ_tbl, wη, bt, B) -> sino

Same result as [`poly_log_sinogram`](@ref) with the `n_cells × n_E` transient
bounded to `B` views: the view axis is processed in batches of `B` inside one
compiled loop (`_loop_over_batches`; a remainder batch runs once), so the
program size does not depend on the number of views.
"""
function poly_log_sinogram_looped(P::_A4, μ_tbl::AbstractMatrix, wη::AbstractVector,
        bt::Union{Nothing, _A3}, B::Int)
    T = _scalar_type(P)
    n_col, n_row, n_view, _ = size(P)
    out = _zeros(P, T, (n_col, n_row, n_view))
    chunk_fn = (start, len, P, μ_tbl, wη, bt) -> poly_log_sinogram(_dslice(P, start, len, 3), μ_tbl, wη, bt)
    return _loop_over_batches(chunk_fn, n_view, B, out, (P, μ_tbl, wη, bt), P)
end

"""
    poly_log_sinogram_batched(P, μ_tbl, wη, bt, B) -> sino

[`poly_log_sinogram`](@ref) over static view batches of `B` (last batch shorter),
unrolled and concatenated pairwise — the unrolled twin of
[`poly_log_sinogram_looped`](@ref).
"""
function poly_log_sinogram_batched(P::_A4, μ_tbl::AbstractMatrix, wη::AbstractVector,
        bt::Union{Nothing, _A3}, B::Int)
    n_view = size(P, 3)
    parts = [poly_log_sinogram(P[:, :, lo:min(lo + B - 1, n_view), :], μ_tbl, wη, bt) for lo in 1:B:n_view]
    return reduce((a, b) -> cat(a, b; dims = 3), parts)
end

# Spectral weight matrix (n_cells, n_E) as used in the forward sum: wη ⊗ 1 or wη·bt.
_spectral_weights(wη::AbstractVector, ::Nothing, n_col, n_row, n_view) = reshape(wη, 1, :)
function _spectral_weights(wη::AbstractVector, bt::_A3, n_col, n_row, n_view)
    T = _scalar_type(wη)
    n_E = length(wη)
    W4 = (reshape(wη, 1, 1, 1, n_E) .* reshape(bt, n_col, n_row, 1, n_E)) .* ones(T, 1, 1, n_view, 1)
    return reshape(W4, n_col * n_row * n_view, n_E)
end

"""
    poly_log_sinogram_vjp(P, μ_tbl, wη, bt, s̄) -> P̄

Closed-form vector–Jacobian product of [`poly_log_sinogram`](@ref) with respect
to the path lengths (the memory-efficient adjoint):

    ∂sino/∂P_m = Σ_e wη·bt·μ[m,e]·exp(-L_e) / Σ_e wη·bt·exp(-L_e)

so `P̄[:,:,:,m] = s̄ .* (A * μ_tblᵀ)[:, m] ./ I` with `A = W .* exp.(-L)`.
Cells whose spectral sum hit the `1e-10` floor get zero gradient (the floor
is a `max`, whose sub-gradient is 0 on the clamped side).  Transient memory
is the same `n_cells × n_E` as the forward pass.
"""
function poly_log_sinogram_vjp(P::_A4, μ_tbl::AbstractMatrix, wη::AbstractVector,
        bt::Union{Nothing, _A3}, s̄::_A3)
    T = _scalar_type(P)
    n_col, n_row, n_view, n_mat = size(P)
    eps = T(1.0e-10)
    L = reshape(P, :, n_mat) * μ_tbl                                   # (n_cells, n_E)
    A = exp.(-L) .* _spectral_weights(wη, bt, n_col, n_row, n_view)    # (n_cells, n_E)
    I = sum(A; dims = 2)                                               # (n_cells, 1)
    gate = ifelse.(I .> eps, one(T), zero(T))
    G = (A * permutedims(μ_tbl, (2, 1))) ./ max.(I, eps)               # (n_cells, n_mat)
    P̄ = (reshape(s̄, :, 1) .* gate) .* G
    return reshape(P̄, n_col, n_row, n_view, n_mat)
end

# -----------------------------------------------------------------------------
# 2. Fill factor (STEP 2a inject / STEP 4 cancel)
# -----------------------------------------------------------------------------

"""
    fill_factor_inject(sino, ff_log) -> sino + (-log ff)

`apply_fill_factor!` (fill_factor.jl:88): adds `T(-log(ff_eff))` to every log
line integral.  `ff_log = T(log(ff_eff))`; `nothing` = effect disabled (or
`ff_eff ≈ 1`), identity.
"""
fill_factor_inject(sino::_A3, ff_log::Real) = sino .- _scalar_type(sino)(ff_log)
fill_factor_inject(sino::_A3, ::Nothing) = sino

"""
    fill_factor_offset(sino, ff_log) -> sino + log ff

driver.jl:632-642 — cancels the injected fill-factor offset after the air-scan
log (real scanners absorb detector gain into the air reference).
"""
fill_factor_offset(sino::_A3, ff_log::Real) = sino .+ _scalar_type(sino)(ff_log)
fill_factor_offset(sino::_A3, ::Nothing) = sino

# -----------------------------------------------------------------------------
# 2b. Scatter (Ohnesorge separable Gaussian estimate + intensity-domain inject)
# -----------------------------------------------------------------------------

"""
    clamped_conv_matrix(k1d, n) -> H (n × n)

Dense matrix form of the legacy 1-D replicate-boundary convolution
(`_convolve_separable_h!`/`_v!`, scatter.jl:154-206): `out[i] = Σ_d
in[clamp(i+d,1,n)]·k1d[d]`, i.e. `H[i, clamp(i+d,1,n)] += k1d[d]`.  Host-side
precomputation (config-only).
"""
function clamped_conv_matrix(k1d::AbstractVector{T}, n::Integer) where {T}
    hk = length(k1d) ÷ 2
    H = zeros(T, n, n)
    for i in 1:n, d in (-hk):hk
        H[i, clamp(i + d, 1, n)] += k1d[d + hk + 1]
    end
    return H
end

# Apply H along axis 1 (columns) of an (n_col, n_row, n_view) array.
_conv_axis1(x::_A3, H::AbstractMatrix) = reshape(H * reshape(x, size(x, 1), :), size(x))
# Apply H along axis 2 (rows): transpose the two leading axes, convolve, transpose back.
_conv_axis2(x::_A3, H::AbstractMatrix) = permutedims(_conv_axis1(permutedims(x, (2, 1, 3)), H), (2, 1, 3))

"""
    scatter_estimate(sino, Hc, Hr, C) -> scatter_field

`estimate_scatter_field!` (scatter.jl:712, separable `:gaussian` path):
`pre = exp(-min(p,20)) · p · C`, then a replicate-boundary 1-D Gaussian pass
along columns (`Hc`) and along rows (`Hr`).  `C = scatter_coefficient ·
scale_factor`.  `Hc === nothing` (scatter disabled) returns `nothing`.
"""
function scatter_estimate(sino::_A3, Hc::AbstractMatrix, Hr::AbstractMatrix, C::Real)
    T = _scalar_type(sino)
    pre = (exp.(-min.(sino, T(20))) .* sino) .* T(C)
    return _conv_axis2(_conv_axis1(pre, Hc), Hr)
end
scatter_estimate(sino::_A3, ::Nothing, ::Nothing, C::Real) = nothing

"""
    scatter_inject(sino, scatter_field, sw) -> sino

`inject_scatter!` (scatter.jl:317): `-log(max(exp(-min(p,20)) + max(sf·sw,0), 1e-10))`.
`scatter_field === nothing` is the identity.
"""
function scatter_inject(sino::_A3, sf::_A3, sw::Real)
    T = _scalar_type(sino)
    return -log.(max.(exp.(-min.(sino, T(20))) .+ max.(sf .* T(sw), zero(T)), T(1.0e-10)))
end
scatter_inject(sino::_A3, ::Nothing, sw::Real) = sino

# -----------------------------------------------------------------------------
# 3. Reparameterised noise + scatter subtraction (driver.jl:522-578)
# -----------------------------------------------------------------------------

"""
    eict_noise(p, I0, ε, ε_e, σ_e, scatter_field, scatter_weight) -> p

The fused counts-domain kernel of `simulate!` STEP 3, exactly:

    λ   = I0 · exp(-p)
    λn  = λ + sqrt(max(λ,1)) · ε        (ε === nothing → skipped)
    λn += σ_e · ε_e                     (ε_e === nothing → skipped)
    λp  = max(λn - I0 · sf · sw, 1)     (sf === nothing → max(λn, 1))
    p'  = -log(λp / I0)

`ε`, `ε_e` are caller-supplied N(0,1) tensors of the sinogram shape (the
legacy `ws.noise_rand_cpu` / `ws.enoise_rand_cpu` draws).  All four legacy
variants (with/without electronic noise, with/without scatter, and the
noise-free-but-scatter branch) are selected by the `Nothing` types on the
host.  NOTE: legacy applies NOTHING when noise is off and scatter is off — the
chain must not call this in that case (see [`eict_chain`](@ref)).
"""
function eict_noise(p::_A3, I0::Real, ε::Union{Nothing, _A3}, ε_e::Union{Nothing, _A3}, σ_e::Real,
        sf::Union{Nothing, _A3}, sw::Real)
    T = _scalar_type(p)
    I0t = T(I0)
    λ = I0t .* exp.(-p)
    λn = _add_electronic(_add_quantum(λ, ε, T), ε_e, T(σ_e))
    λp = _subtract_scatter(λn, sf, I0t, T(sw), T)
    return -log.(λp ./ I0t)
end

_add_quantum(λ, ::Nothing, ::Type{T}) where {T} = λ
_add_quantum(λ::_A3, ε::_A3, ::Type{T}) where {T} = λ .+ sqrt.(max.(λ, one(T))) .* ε
_add_electronic(λn, ::Nothing, σ_e) = λn
_add_electronic(λn::_A3, ε_e::_A3, σ_e::Real) = λn .+ σ_e .* ε_e
_subtract_scatter(λn::_A3, ::Nothing, I0, sw, ::Type{T}) where {T} = max.(λn, one(T))
_subtract_scatter(λn::_A3, sf::_A3, I0, sw, ::Type{T}) where {T} = max.(λn .- I0 .* sf .* sw, one(T))

# -----------------------------------------------------------------------------
# 4. Calibration: intensity → air normalisation → floor → log (driver.jl:583-621)
# -----------------------------------------------------------------------------

"""
    to_intensity(sino) -> exp(-clamp(sino, -1, 15))
"""
function to_intensity(sino::_A3)
    T = _scalar_type(sino)
    return exp.(-clamp.(sino, T(-1), T(15)))
end

"""
    air_normalize(prep, air_ref) -> prep ./ max(air_ref, eps)

Bowtie/heel air-scan reference `(n_col, n_row)` broadcast over views
(driver.jl:593-611).  `nothing` = flat air scan of ones → identity.
"""
function air_normalize(prep::_A3, air_ref::AbstractMatrix)
    T = _scalar_type(prep)
    return prep ./ max.(reshape(air_ref, size(air_ref, 1), size(air_ref, 2), 1), T(1.0e-10))
end
air_normalize(prep::_A3, ::Nothing) = prep

"""
    low_signal_floor(prep, eps) -> max(prep, eps)

`low_signal_correction_gpu!` (≤0 → eps) composed with the legacy
`-log(max(·, eps))` is exactly `-log(max(prep, eps))`, so the floor is a
single `max`.
"""
low_signal_floor(prep::_A3, eps::Real) = max.(prep, _scalar_type(prep)(eps))

"""
    neg_log(prep) -> -log(prep)
"""
neg_log(prep::_A3) = -log.(prep)

# -----------------------------------------------------------------------------
# 5. Beam-hardening correction (bhc_sinogram.jl:296-342)
# -----------------------------------------------------------------------------

"""
    bhc_coeff_matrix(::Type{T}, polys) -> coeffs (order+1, n_col)

Host-side conversion of per-column `BHCPolynomial`s (or the
`BeamHardeningCorrection` / `WaterBHC` wrappers) into the `(order+1, n_col)`
matrix consumed by [`bhc_apply`](@ref) — the same matrix `apply_bhc!` builds.
"""
function bhc_coeff_matrix(::Type{T}, polys::AbstractVector{BS.BHCPolynomial}) where {T}
    n_col = length(polys)
    order = polys[1].order
    all(p.order == order for p in polys) ||
        throw(ArgumentError("bhc_coeff_matrix: polynomial orders must match across columns"))
    coeffs = zeros(T, order + 1, n_col)
    for c in 1:n_col
        coeffs[:, c] .= T.(polys[c].coefficients)
    end
    return coeffs
end
bhc_coeff_matrix(::Type{T}, bhcs::AbstractVector{BS.BeamHardeningCorrection}) where {T} =
    bhc_coeff_matrix(T, [b.polynomial for b in bhcs])
bhc_coeff_matrix(::Type{T}, bhc::BS.WaterBHC) where {T} = bhc_coeff_matrix(T, bhc.water_bhc_per_col)
bhc_coeff_matrix(::Type{T}, ::Nothing) where {T} = nothing

"""
    bhc_apply(sino, coeffs) -> corrected sinogram

Per-column polynomial of STATIC order `size(coeffs,1)-1`, evaluated exactly
like `apply_bhc!` (ascending powers with a running `p_power`):

    p' = c₀[col] + c₁[col]·p + c₂[col]·p² + …

`coeffs === nothing` is the identity (what `simulate!` returns).  Legacy only
has per-COLUMN polynomials (no per-row variant exists).
"""
function bhc_apply(sino::_A3, coeffs::AbstractMatrix)
    T = _scalar_type(sino)
    order = size(coeffs, 1) - 1
    n_col = size(sino, 1)
    c(i) = reshape(coeffs[i, :], n_col, 1, 1)
    order >= 1 || return c(1) .+ zero(T) .* sino
    acc = c(1) .+ c(2) .* sino
    pw = sino
    for i in 2:order
        pw = pw .* sino
        acc = acc .+ c(i + 1) .* pw
    end
    return acc
end
bhc_apply(sino::_A3, ::Nothing) = sino

# d/dp of the per-column polynomial: Σ_{i≥1} i·c_i·p^(i-1)  (used by the VJP)
function _bhc_derivative(sino::_A3, coeffs::AbstractMatrix)
    T = _scalar_type(sino)
    order = size(coeffs, 1) - 1
    n_col = size(sino, 1)
    c(i) = reshape(coeffs[i, :], n_col, 1, 1)
    order >= 1 || return zero(T) .* sino
    acc = c(2) .+ zero(T) .* sino
    pw = one(T) .+ zero(T) .* sino
    for i in 2:order
        pw = pw .* sino
        acc = acc .+ (T(i) .* c(i + 1)) .* pw
    end
    return acc
end
_bhc_derivative(sino::_A3, ::Nothing) = one(_scalar_type(sino)) .+ zero(_scalar_type(sino)) .* sino

# -----------------------------------------------------------------------------
# 6. Plan (host-precomputed, config-only tensors) and the composed chain
# -----------------------------------------------------------------------------

"""
    EICTPlan{T}

Immutable bundle of the host-precomputed, config-only tensors and scalars the
EICT chain needs.  `T` is the HOST scalar type (Float32/Float64) used for every
literal; the array fields may be plain or traced arrays.  Variant selection
(bowtie / air reference / scatter / BHC / fill factor) is encoded in the field
TYPES (`Nothing` vs array/scalar) and `use_noise::Bool`, so every branch is
resolved on the host before any array code runs.

Fields
- `μ_tbl::(n_mat, n_E)`, `wη::(n_E,)`, `bt::nothing | (n_col, n_row, n_E)`
- `air_ref::nothing | (n_col, n_row)` — bowtie/heel air-scan reference
- `I0::T` — photons per detector cell (`compute_detector_I0 · η_eff`)
- `σ_e::T` — DAS electronic noise in photon units; `use_noise`, `use_enoise`
- `ff_log::nothing | T` — `T(log(ff_eff))`
- `scatter_Hc::nothing | (n_col,n_col)`, `scatter_Hr::nothing | (n_row,n_row)`,
  `scatter_C::T`, `scatter_sw::T` — separable Gaussian estimate + spectral weight
- `bhc_coeffs::nothing | (order+1, n_col)`
- `eps::T` (= 1e-10), `sino_shape::(n_col, n_row, n_view)`
"""
struct EICTPlan{T <: AbstractFloat, MT, VT, BT, AR, FF, HC, HR, BC}
    μ_tbl::MT
    wη::VT
    bt::BT
    air_ref::AR
    I0::T
    σ_e::T
    use_noise::Bool
    use_enoise::Bool
    ff_log::FF
    scatter_Hc::HC
    scatter_Hr::HR
    scatter_C::T
    scatter_sw::T
    bhc_coeffs::BC
    eps::T
    sino_shape::NTuple{3, Int}
    energies::Vector{Float64}       # the spectral grid of `wη` / `μ_tbl` columns (keV)
end

"""
    EICTPlan(; μ_tbl, wη, sino_shape, I0, bt=nothing, air_ref=nothing, σ_e=0,
             use_noise=false, ff_log=nothing, scatter_Hc=nothing, scatter_Hr=nothing,
             scatter_C=0, scatter_sw=0, bhc_coeffs=nothing, eps=1e-10, energies=Float64[],
             eltype=_scalar_type(μ_tbl))

Keyword constructor for hand-built plans (tests, toy problems, Reactant smokes).
`eltype` is the host scalar type `T` (defaults to the unwrapped element type of
`μ_tbl`); all scalar kwargs are converted to it.
"""
function EICTPlan(;
        μ_tbl::AbstractMatrix, wη::AbstractVector, sino_shape::NTuple{3, Int}, I0::Real,
        bt = nothing, air_ref = nothing, σ_e::Real = 0, use_noise::Bool = false,
        ff_log::Union{Nothing, Real} = nothing, scatter_Hc = nothing, scatter_Hr = nothing,
        scatter_C::Real = 0, scatter_sw::Real = 0, bhc_coeffs = nothing, eps::Real = 1.0e-10,
        energies::AbstractVector{<:Real} = Float64[],
        eltype::Type{T} = _scalar_type(μ_tbl),
    ) where {T <: AbstractFloat}
    σ = T(σ_e)
    ff = ff_log === nothing ? nothing : T(ff_log)
    isempty(energies) || length(energies) == length(wη) ||
        throw(ArgumentError("EICTPlan: energies has $(length(energies)) entries but wη has $(length(wη))"))
    return EICTPlan{T, typeof(μ_tbl), typeof(wη), typeof(bt), typeof(air_ref), typeof(ff),
        typeof(scatter_Hc), typeof(scatter_Hr), typeof(bhc_coeffs)}(
        μ_tbl, wη, bt, air_ref, T(I0), σ, use_noise, use_noise && σ > zero(T),
        ff, scatter_Hc, scatter_Hr, T(scatter_C), T(scatter_sw), bhc_coeffs, T(eps), sino_shape,
        Vector{Float64}(energies))
end

"""
    eict_plan(ws::EICTWorkspace, protocol, sim_opts; bhc=nothing) -> EICTPlan

TEMPORARY BRIDGE: harvest the host-precomputed tensors of a legacy
`EICTWorkspace` (built by `create_eict_workspace`) into an [`EICTPlan`](@ref).
Everything harvested is config-only (spectrum, μ table, bowtie×heel spectral
weights and air reference, detector efficiency, I0, electronic noise, fill
factor, scatter kernel/weight) — legacy host helpers are called for these:
`compute_detector_I0`, `effective_fill_factor`,
`compute_scatter_energy_weights`, and the workspace's own precomputed fields.

`bhc` may be `nothing` (default — matches what `simulate!` returns), a
`WaterBHC`, or a per-column `Vector{BHCPolynomial}` /
`Vector{BeamHardeningCorrection}`.

Throws if the workspace enables focal-spot blur, optical crosstalk, or
detector lag (not reproduced by the functional chain), or a non-Gaussian
scatter kernel.
"""
function eict_plan(ws::BS.EICTWorkspace{T}, protocol::BS.CTProtocol, sim_opts::BS.SimOptions; bhc = nothing) where {T}
    config = ws.config
    ws.optical_crosstalk_kernel === nothing ||
        throw(ArgumentError("eict_plan: optical crosstalk is not reproduced by the functional EICT chain (use_optical_crosstalk=false)"))
    ws.focal_spot_kernel === nothing ||
        throw(ArgumentError("eict_plan: focal-spot blur is not reproduced by the functional EICT chain (use_focal_spot=false)"))
    config.lag === nothing ||
        throw(ArgumentError("eict_plan: detector lag is not reproduced by the functional EICT chain (use_lag=false)"))

    n_E = length(ws.energies)
    n_col, n_row, n_view = size(ws.sinogram)
    μ_tbl = Matrix{T}(Array(ws.μ_table)[:, 1:n_E])
    wη = Vector{T}(Array(ws.wη_gpu)[1:n_E])
    bt = ws.bowtie_spectral === nothing ? nothing : Array{T, 3}(Array(ws.bowtie_spectral)[:, :, 1:n_E])
    air_ref = ws.bowtie_air_reference === nothing ? nothing : Matrix{T}(Array(ws.bowtie_air_reference))

    # driver.jl:511-512
    I0 = T(BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights))) * ws.η_eff
    σ_e = ws.σ_e_photon

    # fill_factor.jl:88 (`ff ≈ 1.0` short-circuit) / driver.jl:632-642
    ff_log = if config.fill_factor === nothing
        nothing
    else
        ff = BS.effective_fill_factor(config.fill_factor)
        ff ≈ 1.0 ? nothing : T(log(ff))
    end

    # driver.jl:482-497 + scatter.jl estimate/inject
    if config.scatter !== nothing
        model = config.scatter
        model.kernel_type == :gaussian ||
            throw(ArgumentError("eict_plan: only the separable :gaussian scatter kernel is reproduced (got $(model.kernel_type))"))
        k1d = Vector{T}(Array(ws.scatter_kernel_1d))
        Hc = clamped_conv_matrix(k1d, n_col)
        Hr = clamped_conv_matrix(k1d, n_row)
        C = T(model.scatter_coefficient * model.scale_factor)
        ew = BS.compute_scatter_energy_weights(Float64.(ws.energies))
        wn = Float64.(ws.weights_norm)
        η = ws.η_vec
        sw = T(sum(wn[i] * ew[i] * η[i] for i in eachindex(wn)) /
            max(sum(wn[i] * η[i] for i in eachindex(wn)), 1.0e-30))
    else
        Hc = nothing; Hr = nothing; C = zero(T); sw = zero(T)
    end

    return EICTPlan(; μ_tbl, wη, sino_shape = (n_col, n_row, n_view), I0, bt, air_ref, σ_e,
        use_noise = sim_opts.use_noise, ff_log, scatter_Hc = Hc, scatter_Hr = Hr,
        scatter_C = C, scatter_sw = sw, bhc_coeffs = bhc_coeff_matrix(T, bhc), energies = Float64.(ws.energies), eltype = T)
end

# Host-level selection of the STEP 3 variant (mirrors driver.jl:522-578).
function _noise_stage(p::_A3, plan::EICTPlan{T}, sf, ε, ε_e) where {T}
    shape = size(p)
    if plan.use_noise
        ε === nothing && throw(ArgumentError("eict_chain: plan.use_noise=true requires the ε ~ N(0,1) tensor"))
        εr = reshape(ε, shape)
        if plan.use_enoise
            ε_e === nothing && throw(ArgumentError("eict_chain: σ_e > 0 requires the electronic-noise ε_e tensor"))
            return eict_noise(p, plan.I0, εr, reshape(ε_e, shape), plan.σ_e, sf, plan.scatter_sw)
        else
            return eict_noise(p, plan.I0, εr, nothing, plan.σ_e, sf, plan.scatter_sw)
        end
    elseif sf !== nothing
        return eict_noise(p, plan.I0, nothing, nothing, plan.σ_e, sf, plan.scatter_sw)
    else
        return p
    end
end

"""
    eict_chain(P, plan, ε=nothing, ε_e=nothing) -> sinogram

Full EICT chain from per-material path lengths `P::(n_col, n_row, n_view,
n_mat)` to the calibrated log line-integral sinogram, in exactly the legacy
`simulate!(::EICTWorkspace)` order:

 1. `poly_log_sinogram`             (STEP 1, `_forward_project_poly!` spectral sum)
 2. `fill_factor_inject`            (STEP 2, `apply_fill_factor!`)
 3. `scatter_estimate` + `scatter_inject`  (STEP 2, unified per-energy scatter)
 4. `eict_noise`                    (STEP 3, only when noise or scatter is on)
 5. `to_intensity` → `air_normalize` → `low_signal_floor` → `neg_log`
    → `fill_factor_offset`          (STEP 4)
 6. `bhc_apply`                     (decoupled in legacy; identity unless the
                                     plan carries `bhc_coeffs`)

`ε`, `ε_e` are N(0,1) tensors with `prod(plan.sino_shape)` elements (any
shape; reshaped on the host) — the exact draws of `ws.noise_rand_cpu` /
`ws.enoise_rand_cpu` reproduce `simulate!` bit-for-bit up to float ordering.
`view_batch > 0` bounds the spectral transient to that many views per compiled
loop iteration ([`poly_log_sinogram_looped`](@ref)); `0` = all views at once.
"""
function eict_chain(P::_A4, plan::EICTPlan{T}, ε = nothing, ε_e = nothing; view_batch::Int = 0, loop::Bool = true) where {T}
    n_view = size(P, 3)
    p = if view_batch <= 0 || view_batch >= n_view
        poly_log_sinogram(P, plan.μ_tbl, plan.wη, plan.bt)
    elseif loop
        poly_log_sinogram_looped(P, plan.μ_tbl, plan.wη, plan.bt, view_batch)
    else
        poly_log_sinogram_batched(P, plan.μ_tbl, plan.wη, plan.bt, view_batch)
    end
    p = fill_factor_inject(p, plan.ff_log)
    sf = scatter_estimate(p, plan.scatter_Hc, plan.scatter_Hr, plan.scatter_C)
    p = scatter_inject(p, sf, plan.scatter_sw)
    p = _noise_stage(p, plan, sf, ε, ε_e)
    q = to_intensity(p)
    q = air_normalize(q, plan.air_ref)
    q = low_signal_floor(q, plan.eps)
    p = neg_log(q)
    p = fill_factor_offset(p, plan.ff_log)
    return bhc_apply(p, plan.bhc_coeffs)
end

# -----------------------------------------------------------------------------
# 7. Hand-derived adjoint of the whole chain (memory-efficient VJP)
# -----------------------------------------------------------------------------

_zero_like(x::AbstractArray) = zero(_scalar_type(x)) .* x

"""
    eict_chain_vjp(P, plan, ε, ε_e, s̄) -> P̄

Vector–Jacobian product of [`eict_chain`](@ref) with respect to the path
lengths: `P̄ = (∂ Σ s̄ ⋅ eict_chain(P) / ∂P)`.  Every stage's local
derivative is hand-derived (all stages are element-wise except the linear
scatter convolution, whose adjoint is the transposed convolution matrices,
and the spectral sum, handled by [`poly_log_sinogram_vjp`](@ref)).  Clamps
(`max`, `clamp`, `min`) contribute their sub-gradient (0 on the clamped side).
Same variant selection as the forward chain.
"""
function eict_chain_vjp(P::_A4, plan::EICTPlan{T}, ε, ε_e, s̄::_A3) where {T}
    eps = plan.eps
    # ---- forward intermediates ----
    p0 = poly_log_sinogram(P, plan.μ_tbl, plan.wη, plan.bt)
    p1 = fill_factor_inject(p0, plan.ff_log)
    sf = scatter_estimate(p1, plan.scatter_Hc, plan.scatter_Hr, plan.scatter_C)
    p2 = scatter_inject(p1, sf, plan.scatter_sw)
    p3 = _noise_stage(p2, plan, sf, ε, ε_e)
    q0 = to_intensity(p3)
    q1 = air_normalize(q0, plan.air_ref)
    q2 = low_signal_floor(q1, eps)
    p4 = neg_log(q2)
    p5 = fill_factor_offset(p4, plan.ff_log)
    # ---- backward ----
    p̄5 = s̄ .* _bhc_derivative(p5, plan.bhc_coeffs)
    q̄2 = p̄5 .* (-one(T) ./ q2)
    q̄1 = q̄2 .* ifelse.(q1 .> eps, one(T), zero(T))
    q̄0 = _air_normalize_vjp(q̄1, plan.air_ref, T)
    p̄3 = q̄0 .* (-q0) .* ifelse.((p3 .> T(-1)) .& (p3 .< T(15)), one(T), zero(T))
    p̄2, s̄f = _noise_stage_vjp(p2, plan, sf, ε, ε_e, p̄3)
    p̄1, s̄f = _scatter_inject_vjp(p1, sf, plan.scatter_sw, p̄2, s̄f, T)
    p̄1 = p̄1 .+ _scatter_estimate_vjp(p1, plan.scatter_Hc, plan.scatter_Hr, plan.scatter_C, s̄f, T)
    return poly_log_sinogram_vjp(P, plan.μ_tbl, plan.wη, plan.bt, p̄1)
end

_air_normalize_vjp(q̄1, ::Nothing, ::Type{T}) where {T} = q̄1
_air_normalize_vjp(q̄1::_A3, air_ref::AbstractMatrix, ::Type{T}) where {T} =
    q̄1 ./ max.(reshape(air_ref, size(air_ref, 1), size(air_ref, 2), 1), T(1.0e-10))

# STEP 3 adjoint: returns (p̄2, s̄f) with s̄f === nothing when scatter is off.
function _noise_stage_vjp(p2::_A3, plan::EICTPlan{T}, sf, ε, ε_e, p̄3) where {T}
    has_noise = plan.use_noise
    if !has_noise && sf === nothing
        return p̄3, nothing
    end
    I0 = plan.I0
    λ = I0 .* exp.(-p2)
    if has_noise
        εr = reshape(ε, size(p2))
        λn = λ .+ sqrt.(max.(λ, one(T))) .* εr
        dλn_dλ = one(T) .+ εr .* ifelse.(λ .> one(T), one(T) ./ (T(2) .* sqrt.(max.(λ, one(T)))), zero(T))
        λn = plan.use_enoise ? λn .+ plan.σ_e .* reshape(ε_e, size(p2)) : λn
    else
        λn = λ
        dλn_dλ = one(T) .+ _zero_like(λ)
    end
    sub = sf === nothing ? λn : λn .- I0 .* sf .* plan.scatter_sw
    gate = ifelse.(sub .> one(T), one(T), zero(T))
    λp = max.(sub, one(T))
    λ̄p = p̄3 .* (-one(T) ./ λp)
    λ̄n = λ̄p .* gate
    p̄2 = λ̄n .* dλn_dλ .* (-λ)
    s̄f = sf === nothing ? nothing : λ̄n .* (-(I0 * plan.scatter_sw))
    return p̄2, s̄f
end

# inject adjoint: p2 = -log(max(exp(-min(p1,20)) + max(sf·sw,0), 1e-10))
_scatter_inject_vjp(p1, ::Nothing, sw, p̄2, ::Nothing, ::Type{T}) where {T} = p̄2, nothing
function _scatter_inject_vjp(p1::_A3, sf::_A3, sw::Real, p̄2, s̄f, ::Type{T}) where {T}
    swt = T(sw)
    m = min.(p1, T(20))
    e = exp.(-m)
    ss = sf .* swt
    tot = e .+ max.(ss, zero(T))
    t̄ = p̄2 .* (-one(T) ./ max.(tot, T(1.0e-10))) .* ifelse.(tot .> T(1.0e-10), one(T), zero(T))
    p̄1 = t̄ .* (-e) .* ifelse.(p1 .< T(20), one(T), zero(T))
    s̄f2 = s̄f .+ t̄ .* swt .* ifelse.(ss .> zero(T), one(T), zero(T))
    return p̄1, s̄f2
end

# estimate adjoint: sf = Hr ⊛ Hc ⊛ pre,  pre = exp(-min(p,20))·p·C
_scatter_estimate_vjp(p1::_A3, ::Nothing, ::Nothing, C, ::Nothing, ::Type{T}) where {T} = _zero_like(p1)
function _scatter_estimate_vjp(p1::_A3, Hc::AbstractMatrix, Hr::AbstractMatrix, C::Real, s̄f::_A3, ::Type{T}) where {T}
    p̄re = _conv_axis1(_conv_axis2(s̄f, permutedims(Hr, (2, 1))), permutedims(Hc, (2, 1)))
    m = min.(p1, T(20))
    e = exp.(-m)
    dpre = T(C) .* e .* (one(T) .- p1 .* ifelse.(p1 .< T(20), one(T), zero(T)))
    return p̄re .* dpre
end
