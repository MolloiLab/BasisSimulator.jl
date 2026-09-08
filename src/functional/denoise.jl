"""
Pure, static-shape, array-generic denoisers (`BasisSimulator.Functional`).

Functional re-implementations of the projection/image-domain denoisers in
`src/denoising/`, written as tensor programs suitable for Reactant (XLA) and
Enzyme: every stage is a pure function of input arrays + an immutable plan
struct that holds all host constants.  No in-place mutation, no scalar indexing
of data arrays, no data-dependent shapes / trip counts / early returns, no RNG,
no order statistics inside a stage.

| stage                          | legacy oracle (`src/denoising/`)                   |
|--------------------------------|----------------------------------------------------|
| `sino_svd_denoise_bilateral`   | `apply_sino_svd_denoise_bilateral!` (sino_svd.jl)  |
| `acnr_kalender`                | `apply_acnr_kalender!` (acnr.jl)                   |
| `median_z`                     | `apply_median_z!` (median_z.jl)                    |
| `sfjsd_denoise` / `sfjsd_pass` | `apply_sino_sfjsd_denoise` (sino_sfjsd.jl)         |

# Host / device split

Order statistics (median, MAD) and the legacy data heuristics are not tensor
programs.  Wherever the legacy derives a constant from the data that way, the
functional stage reads it from the plan and a host helper computes it ONCE by
reusing the legacy code:

* `mad_range_scales(channels; bilat_range_k)` — per-row bilateral range scales
  (`k · 1.4826 · MAD`) for the SVD-bilateral stage, from the legacy `svd` +
  `_mad_scale_2d`.  Stored in **Y-units** (see below).
* `SFJSDPlan(channels, I0; σ₀)` — SF-JSD stride (noise-correlation length),
  `n_iter` (min photon count), `σ₀★` (SURE golden-section, RNG-seeded), the
  per-iteration per-row MAD range scales and the maximum spatial-kernel radius,
  all captured along the legacy trajectory (the legacy `_sfjsd_pass!` is run to
  advance ξ between iterations so iteration-2 scales are legacy-exact).

# Per-row SVD without an SVD

The legacy stacks the `N` channels of a detector row as columns of
`M = [vec(ξ₁) … vec(ξ_N)]`, computes `M = U Σ Vᵀ`, keeps `U[:,1]`, filters
`U[:,2..N]`, and reconstitutes `U_d Σ Vᵀ`.  Here the `N×N` Gram matrix `MᵀM`
is eigendecomposed in closed form (2×2) or by a fixed number of cyclic Jacobi
sweeps (3×3, 4×4), batched over rows, giving `V` and `λ = Σ²`.  The stage then
works on `Y_k = M V[:,k] = σ_k U[:,k]` — the *unnormalised* left singular
vectors — so no `sqrt(λ)` / division by a tiny `σ_k` is ever taken:

    M_d = Y₁ V[:,1]ᵀ + Σ_{k≥2} bilateral(Y_k) V[:,k]ᵀ .

The joint bilateral filters used by both SVD stages are homogeneous of degree
one in their target once the range scales are expressed in the same units
(`bilateral(σ u; σ·s) = σ · bilateral(u; s)`), so filtering `Y_k` with range
scale `σ_k · (k · MAD(U[:,k]))` is *mathematically identical* to the legacy
filtering of `U[:,k]` with `k · MAD(U[:,k])` followed by the `σ_k` rescale.
The host helpers therefore return scales already multiplied by the legacy
singular value (“Y-units”).  Sign convention: `V[1,k] ≥ 0`; the outputs are
sign-invariant anyway (each `V[:,k]` enters twice and the bilateral is odd).

# Layout

Spatial filters act on dims `(1, 2)` of a 3-D array and batch over dim 3.
Sinograms `(n_col, n_row, n_view)` are permuted to `(n_col, n_view, n_row)`
internally (row = batch); volumes `(nx, ny, nz)` are used as-is (z = batch).
Boundary handling mirrors each legacy kernel exactly: truncated-renormalised
Gaussian (SVD / SF-JSD), replicate-padded Gaussian (ACNR), zero-weight
out-of-range bilateral taps, zero-padded box sums.
"""

import LinearAlgebra
import Statistics

# The legacy package (numerical oracle) — resolved whether this file is
# included into `BasisSimulator.Functional` (submodule) or into a scratch
# module that did `using BasisSimulator`.
const _LEGACY = let m = @__MODULE__
    if isdefined(m, :BasisSimulator) && getfield(m, :BasisSimulator) isa Module
        getfield(m, :BasisSimulator)
    elseif nameof(parentmodule(m)) === :BasisSimulator
        parentmodule(m)
    else
        ks = collect(keys(Base.loaded_modules))
        idx = findfirst(p -> p.name == "BasisSimulator", ks)
        idx === nothing && error("denoise.jl: BasisSimulator is not loaded")
        Base.loaded_modules[ks[idx]]
    end
end

# =============================================================================
#  Shared static-shape building blocks (dims (1, 2) spatial, dim 3 batch)
# =============================================================================

# Replicate-pad dims (1, 2) by (r1, r2) using only slicing + `cat`.
function _pad_rep12(x::AbstractArray{<:Any, 3}, r1::Int, r2::Int)
    xp = if r1 == 0
        x
    else
        cat(ntuple(_ -> x[1:1, :, :], r1)..., x, ntuple(_ -> x[end:end, :, :], r1)...; dims = 1)
    end
    return if r2 == 0
        xp
    else
        cat(ntuple(_ -> xp[:, 1:1, :], r2)..., xp, ntuple(_ -> xp[:, end:end, :], r2)...; dims = 2)
    end
end

# Host constant: ones on the interior, zeros in the (r1, r2) pad frame.
function _valid_mask(::Type{T}, n1::Int, n2::Int, r1::Int, r2::Int) where {T}
    m = zeros(T, n1 + 2r1, n2 + 2r2, 1)
    m[(r1 + 1):(r1 + n1), (r2 + 1):(r2 + n2), 1] .= one(T)
    return m
end

# Fixed slice of a padded array shifted by (d1, d2): xp[i + d1, j + d2].
@inline function _shift12(xp::AbstractArray{<:Any, 3}, d1::Int, d2::Int,
        n1::Int, n2::Int, r1::Int, r2::Int)
    return xp[(r1 + 1 + d1):(r1 + n1 + d1), (r2 + 1 + d2):(r2 + n2 + d2), :]
end

# Zero-pad dims (1, 2) (replicate pad × host mask — no `zeros` of a data array).
# `T` is the HOST scalar type (from the plan), never the array eltype.
function _pad_zero12(::Type{T}, x::AbstractArray{<:Any, 3}, r1::Int, r2::Int) where {T}
    return _pad_rep12(x, r1, r2) .* _valid_mask(T, size(x, 1), size(x, 2), r1, r2)
end

# Separable Gaussian on dims (1, 2), truncated-renormalised at the boundary
# (legacy `_separable_gauss_2d` / `_sfjsd_gauss_2d`: out-of-range taps are
# skipped and the weight sum re-normalised).  Same tap order as the legacy.
function _gauss12_trunc(x::AbstractArray{<:Any, 3}, ks::AbstractVector{T}, r::Int) where {T}
    n1, n2, _ = size(x)
    m1 = _valid_mask(T, n1, 1, r, 0)
    xp = _pad_rep12(x, r, 0)
    ms = _shift12(m1, -r, 0, n1, 1, r, 0)
    num = ks[1] .* _shift12(xp, -r, 0, n1, n2, r, 0) .* ms
    den = ks[1] .* ms
    for (i, d) in enumerate(-r:r)
        i == 1 && continue
        ms = _shift12(m1, d, 0, n1, 1, r, 0)
        num = num .+ ks[i] .* _shift12(xp, d, 0, n1, n2, r, 0) .* ms
        den = den .+ ks[i] .* ms
    end
    tmp = num ./ den
    m2 = _valid_mask(T, 1, n2, 0, r)
    tp = _pad_rep12(tmp, 0, r)
    ms = _shift12(m2, 0, -r, 1, n2, 0, r)
    num = ks[1] .* _shift12(tp, 0, -r, n1, n2, 0, r) .* ms
    den = ks[1] .* ms
    for (i, d) in enumerate(-r:r)
        i == 1 && continue
        ms = _shift12(m2, 0, d, 1, n2, 0, r)
        num = num .+ ks[i] .* _shift12(tp, 0, d, n1, n2, 0, r) .* ms
        den = den .+ ks[i] .* ms
    end
    return num ./ den
end

# Separable Gaussian on dims (1, 2) with replicate (clamped-index) boundary
# (legacy ACNR `G`).  Same tap order as the legacy.
function _gauss12_rep(x::AbstractArray{<:Any, 3}, k::AbstractVector{T}, r::Int) where {T}
    n1, n2, _ = size(x)
    xp = _pad_rep12(x, r, 0)
    tmp = k[1] .* _shift12(xp, -r, 0, n1, n2, r, 0)
    for t in (-r + 1):r
        tmp = tmp .+ k[t + r + 1] .* _shift12(xp, t, 0, n1, n2, r, 0)
    end
    tp = _pad_rep12(tmp, 0, r)
    out = k[1] .* _shift12(tp, 0, -r, n1, n2, 0, r)
    for t in (-r + 1):r
        out = out .+ k[t + r + 1] .* _shift12(tp, 0, t, n1, n2, 0, r)
    end
    return out
end

# (2w+1)² box sum on dims (1, 2), zero-padded (= legacy "skip out-of-range"),
# accumulated in the legacy order (dim-2 offset outer, dim-1 offset inner).
function _boxsum12_zero(::Type{T}, x::AbstractArray{<:Any, 3}, w::Int) where {T}
    n1, n2, _ = size(x)
    xp = _pad_zero12(T, x, w, w)
    acc = _shift12(xp, -w, -w, n1, n2, w, w)
    for dj in -w:w, di in -w:w
        (dj == -w && di == -w) && continue
        acc = acc .+ _shift12(xp, di, dj, n1, n2, w, w)
    end
    return acc
end

# Elementwise ascending sort of an M-tuple of arrays: fixed bubble network of
# min/max compare-exchanges (static, data-independent).
function _sort_network(xs::NTuple{M, Any}) where {M}
    v = xs
    for i in 1:(M - 1), j in 1:(M - i)
        vj = v
        lo = min.(vj[j], vj[j + 1])
        hi = max.(vj[j], vj[j + 1])
        v = ntuple(k -> k == j ? lo : (k == j + 1 ? hi : vj[k]), Val(M))
    end
    return v
end

# =============================================================================
#  Batched N×N Gram eigendecomposition (N ≤ 4), rows = dim 3
# =============================================================================

# G[i][j] :: (1, 1, n_batch) — Gram matrix of the channel tuple over dims (1, 2).
function _gram(chs::NTuple{N, Any}) where {N}
    up = ntuple(i -> ntuple(j -> j >= i ? sum(chs[i] .* chs[j]; dims = (1, 2)) : nothing, Val(N)), Val(N))
    return ntuple(i -> ntuple(j -> j >= i ? up[i][j] : up[j][i], Val(N)), Val(N))
end

# Closed-form symmetric 2×2 eigenpairs, λ descending.  V[i][k] = component i
# of eigenvector k.  Branch-free (`ifelse`) and NaN-free in the unselected
# branches (important for AD).  Degenerate G ∝ I → V = I.
function _eig_sym2(::Type{T}, G) where {T}
    a = G[1][1]; b = G[1][2]; c = G[2][2]
    h = (a .+ c) .* T(0.5)
    d = (a .- c) .* T(0.5)
    r = sqrt.(d .* d .+ b .* b)
    λ1 = h .+ r
    λ2 = h .- r
    pos = d .>= zero(T)
    vx = ifelse.(pos, d .+ r, b)
    vy = ifelse.(pos, b, r .- d)
    nrm = sqrt.(vx .* vx .+ vy .* vy)
    deg = nrm .<= zero(T)
    nrm_s = ifelse.(deg, one(T), nrm)
    v1x = ifelse.(deg, one(T), vx ./ nrm_s)
    v1y = ifelse.(deg, zero(T), vy ./ nrm_s)
    v2x = zero(T) .- v1y
    v2y = v1x
    return (λ1, λ2), ((v1x, v2x), (v1y, v2y))
end

# Cyclic Jacobi eigen-solver on the batched Gram (N ≥ 3), fixed sweep count.
# Numerical-Recipes rotation convention; branch-free via `ifelse`.
function _jacobi_eig(::Type{T}, G::NTuple{N, NTuple{N, Any}}, n_sweeps::Int) where {T, N}
    z = zero(T) .* G[1][1]
    Gc = G
    V = ntuple(i -> ntuple(j -> (i == j ? one(T) : zero(T)) .+ z, Val(N)), Val(N))
    for _ in 1:n_sweeps, p in 1:(N - 1), q in (p + 1):N
        app = Gc[p][p]; aqq = Gc[q][q]; apq = Gc[p][q]
        isz = apq .== zero(T)
        apq_s = ifelse.(isz, one(T), apq)
        θ = (aqq .- app) ./ (T(2) .* apq_s)
        t = ifelse.(θ .< zero(T), -one(T), one(T)) ./ (abs.(θ) .+ sqrt.(θ .* θ .+ one(T)))
        t = ifelse.(isz, zero(T), t)
        c = one(T) ./ sqrt.(t .* t .+ one(T))
        s = t .* c
        Go = Gc; Vo = V
        Gc = ntuple(i -> ntuple(j -> begin
            if i == p && j == p
                app .- t .* apq
            elseif i == q && j == q
                aqq .+ t .* apq
            elseif (i == p && j == q) || (i == q && j == p)
                zero(T) .* apq
            elseif j == p
                c .* Go[i][p] .- s .* Go[i][q]
            elseif j == q
                s .* Go[i][p] .+ c .* Go[i][q]
            elseif i == p
                c .* Go[p][j] .- s .* Go[q][j]
            elseif i == q
                s .* Go[p][j] .+ c .* Go[q][j]
            else
                Go[i][j]
            end
        end, Val(N)), Val(N))
        V = ntuple(i -> ntuple(j -> j == p ? c .* Vo[i][p] .- s .* Vo[i][q] :
            (j == q ? s .* Vo[i][p] .+ c .* Vo[i][q] : Vo[i][j]), Val(N)), Val(N))
    end
    λ = ntuple(k -> Gc[k][k], Val(N))
    return λ, V
end

# Sort eigenpairs descending (compare-exchange network) and fix V[1][k] ≥ 0.
function _eig_sort_sign(::Type{T}, λ::NTuple{N, Any}, V::NTuple{N, NTuple{N, Any}}) where {T, N}
    for i in 1:(N - 1), j in 1:(N - i)
        λo = λ; Vo = V
        sw = λo[j] .< λo[j + 1]
        λ = ntuple(k -> k == j ? ifelse.(sw, λo[j + 1], λo[j]) :
            (k == j + 1 ? ifelse.(sw, λo[j], λo[j + 1]) : λo[k]), Val(N))
        V = ntuple(r -> ntuple(k -> k == j ? ifelse.(sw, Vo[r][j + 1], Vo[r][j]) :
            (k == j + 1 ? ifelse.(sw, Vo[r][j], Vo[r][j + 1]) : Vo[r][k]), Val(N)), Val(N))
    end
    Vs = V
    sg = ntuple(k -> ifelse.(Vs[1][k] .< zero(T), -one(T), one(T)), Val(N))
    V = ntuple(r -> ntuple(k -> Vs[r][k] .* sg[k], Val(N)), Val(N))
    return λ, V
end

"""
    gram_eigen(T, chs::NTuple{N, AbstractArray{<:Any,3}}, n_sweeps = 10) -> (λ, V)
    gram_eigen(chs::NTuple{N, Array{T,3}}, n_sweeps = 10)                  # host convenience

Batched (over dim 3) eigendecomposition of the `N×N` Gram matrix of the
channel tuple over dims `(1, 2)`: `λ[k]` (`(1,1,n_batch)`, descending, `= Σ_k²`
of the per-batch SVD) and `V[i][k]` (component `i` of eigenvector `k`, sign
fixed so `V[1][k] ≥ 0`).  Closed form for `N = 2`, `n_sweeps` cyclic Jacobi
sweeps for `N ≥ 3`.  `T` is the host scalar type (a traced array's eltype is
`TracedRNumber{T}`, so it is never taken from the arrays).
"""
function gram_eigen(::Type{T}, chs::NTuple{N, Any}, n_sweeps::Int = 10) where {T, N}
    N >= 2 || error("gram_eigen: need ≥ 2 channels (got $N)")
    G = _gram(chs)
    λ, V = N == 2 ? _eig_sym2(T, G) : _jacobi_eig(T, G, n_sweeps)
    return _eig_sort_sign(T, λ, V)
end
gram_eigen(chs::NTuple{N, <:Array{T, 3}}, n_sweeps::Int = 10) where {T, N} = gram_eigen(T, chs, n_sweeps)

# Y_k = Σ_i chs[i] · V[i][k]
function _combine(chs::NTuple{N, Any}, V, k::Int) where {N}
    acc = chs[1] .* V[1][k]
    for i in 2:N
        acc = acc .+ chs[i] .* V[i][k]
    end
    return acc
end

# out_j = Σ_k Yd[k] · V[j][k]
function _reconstitute(Yd::NTuple{N, Any}, V, j::Int) where {N}
    acc = Yd[1] .* V[j][1]
    for k in 2:N
        acc = acc .+ Yd[k] .* V[j][k]
    end
    return acc
end

# =============================================================================
#  1. SVD + edge-aware joint bilateral  (oracle: apply_sino_svd_denoise_bilateral!)
# =============================================================================

# Joint bilateral on dims (1, 2): range-weighted by the guide and by the
# target itself; out-of-range taps get zero weight (legacy `continue`).  Same
# tap order and same floating-point expression as `_joint_bilateral_2d`.
function _joint_bilateral12(tgt::AbstractArray{<:Any, 3}, guide::AbstractArray{<:Any, 3},
        σg, σt, r::Int, σs2::T) where {T}
    n1, n2, _ = size(tgt)
    mask = _valid_mask(T, n1, n2, r, r)
    tp = _pad_rep12(tgt, r, r)
    gp = _pad_rep12(guide, r, r)
    local acc, wsum
    for dv in -r:r, dc in -r:r
        ms = _shift12(mask, dc, dv, n1, n2, r, r)
        tq = _shift12(tp, dc, dv, n1, n2, r, r)
        gq = _shift12(gp, dc, dv, n1, n2, r, r)
        zg = (gq .- guide) ./ σg
        zt = (tq .- tgt) ./ σt
        sp = T(-(dc * dc + dv * dv)) / σs2
        w = exp.(sp .- T(0.5) .* zg .* zg .- T(0.5) .* zt .* zt) .* ms
        if dv == -r && dc == -r
            acc = w .* tq
            wsum = w
        else
            acc = acc .+ w .* tq
            wsum = wsum .+ w
        end
    end
    return acc ./ wsum
end

_host_mad_scale(x::AbstractMatrix{Float32}) = _LEGACY._mad_scale_2d(x)
function _host_mad_scale(x::AbstractMatrix{T}) where {T}
    v = vec(x)
    med = Statistics.median(v)
    mad = Statistics.median(abs.(v .- med))
    return max(T(1.4826 * mad), T(1.0e-12))
end

"""
    mad_range_scales(channels::NTuple{N, AbstractArray{T,3}}; bilat_range_k = 2.0)
        -> (σ_guide::Array{T,3}, σ_tgt::NTuple{N-1, Array{T,3}})

HOST helper (order statistics).  Per detector row, the legacy `svd` of the
stacked channels and the legacy `_mad_scale_2d` give the joint-bilateral range
scales `σ_g = k·MAD(U[:,1])`, `σ_t,k = k·MAD(U[:,k])`.  Returned multiplied by
the corresponding singular value (Y-units — see the file docstring) as
`(1, 1, n_row)` arrays ready to broadcast over the `(n_col, n_view, n_row)`
batch layout.
"""
function mad_range_scales(channels::NTuple{N, <:Array{T, 3}};
        bilat_range_k::Real = 2.0) where {N, T}   # HOST helper: plain Arrays only
    n_col, n_row, n_view = size(channels[1])
    kf = T(bilat_range_k)
    σg = zeros(T, 1, 1, n_row)
    σt = ntuple(_ -> zeros(T, 1, 1, n_row), N - 1)
    for row in 1:n_row
        M = hcat(ntuple(b -> vec(channels[b][:, row, :]), N)...)
        F = LinearAlgebra.svd(M; full = false)
        U = F.U; S = F.S
        σg[1, 1, row] = max(kf * _host_mad_scale(reshape(U[:, 1], n_col, n_view)) * S[1], T(1.0e-12))
        for k in 2:N
            σt[k - 1][1, 1, row] = max(kf * _host_mad_scale(reshape(U[:, k], n_col, n_view)) * S[k], T(1.0e-12))
        end
    end
    return σg, σt
end

"""
    SinoSVDBilateralPlan(channels; bilat_radius = 3, bilat_sigma_s = 2.0,
                         bilat_range_k = 2.0, n_sweeps = 10)

Immutable host plan for [`sino_svd_denoise_bilateral`](@ref): kernel radius,
`2σ_s²`, the per-row MAD range scales (from [`mad_range_scales`](@ref), i.e.
computed ONCE on the host from the same channels), and the Jacobi sweep count
(`N ≥ 3` only).  `bilat_range_k ≤ 0` ⇒ passthrough (legacy semantics).
"""
struct SinoSVDBilateralPlan{T, A, M}
    radius::Int
    σs2::T
    passthrough::Bool
    σ_guide::A
    σ_tgt::NTuple{M, A}
    n_sweeps::Int
end

function SinoSVDBilateralPlan(channels::NTuple{N, <:Array{T, 3}};
        bilat_radius::Integer = 3, bilat_sigma_s::Real = 2.0,
        bilat_range_k::Real = 2.0, n_sweeps::Integer = 10) where {N, T}   # HOST constructor
    N >= 2 || error("SinoSVDBilateralPlan: requires ≥ 2 channels (got $N)")
    σg, σt = mad_range_scales(channels; bilat_range_k = max(bilat_range_k, 0))
    return SinoSVDBilateralPlan{T, typeof(σg), N - 1}(
        Int(bilat_radius), T(2 * bilat_sigma_s^2), bilat_range_k <= 0, σg, σt, Int(n_sweeps))
end

"""
    sino_svd_denoise_bilateral(channels::NTuple{N, AbstractArray{T,3}}, plan)
        -> NTuple{N}

Pure counterpart of `apply_sino_svd_denoise_bilateral!`.  Per detector row:
batched Gram eigendecomposition → `Y_k = M V[:,k]`; `Y₁` (anatomy) untouched;
`Y_{2..N}` cleaned with the edge-aware joint bilateral guided by `Y₁`; then
`out_j = Σ_k Yd_k V[j,k]`.  Channels are `(n_col, n_row, n_view)`.
"""
function sino_svd_denoise_bilateral(channels::NTuple{N, <:AbstractArray{<:Any, 3}},
        plan::SinoSVDBilateralPlan{T}) where {N, T}
    plan.passthrough && return map(copy, channels)
    chs = map(c -> permutedims(c, (1, 3, 2)), channels)
    _, V = gram_eigen(T, chs, plan.n_sweeps)
    Y = ntuple(k -> _combine(chs, V, k), Val(N))
    Yd = ntuple(k -> k == 1 ? Y[1] :
        _joint_bilateral12(Y[k], Y[1], plan.σ_guide, plan.σ_tgt[k - 1], plan.radius, plan.σs2), Val(N))
    return ntuple(j -> permutedims(_reconstitute(Yd, V, j), (1, 3, 2)), Val(N))
end

# =============================================================================
#  2. Kalender-1988 ACNR (per-pixel HF regression)  (oracle: apply_acnr_kalender!)
# =============================================================================

"""
    ACNRKalenderPlan(T = Float32; hp_sigma_px = 1.5, window = 4, beta_max = 8.0, passes = 2)

Host constants for [`acnr_kalender`](@ref): the separable high-pass Gaussian
taps (radius `max(2, ceil(3σ))`, built exactly as the legacy `G`), the
regression window half-width, `beta_max`, and the (unrolled) pass count.
Live-notebook settings: nb03 `(1.5, 4, 14.0, 5)`, nb04 `(1.5, 4, 20.0, 4)`.
"""
struct ACNRKalenderPlan{T}
    kernel::Vector{T}
    radius::Int
    window::Int
    beta_max::T
    passes::Int
end

function ACNRKalenderPlan(::Type{T} = Float32; hp_sigma_px::Real = 1.5, window::Integer = 4,
        beta_max::Real = 8.0, passes::Integer = 2) where {T}
    σ = Float64(hp_sigma_px)
    r = max(2, ceil(Int, 3σ))
    k = T.(exp.(-(collect(-r:r) .^ 2) ./ (2σ^2)))
    k ./= sum(k)
    return ACNRKalenderPlan{T}(k, r, Int(window), T(beta_max), Int(passes))
end

function _acnr_kalender_pass(W::AbstractArray{<:Any, 3}, I::AbstractArray{<:Any, 3},
        plan::ACNRKalenderPlan{T}) where {T}
    r = plan.radius; k = plan.kernel; w = plan.window
    hW = W .- _gauss12_rep(W, k, r)
    hI = I .- _gauss12_rep(I, k, r)
    sHW = sum(hW .* hW)
    sHI = sum(hI .* hI)
    λ_I = sqrt(sHI / max(sHW, T(1.0e-30)))
    λ_W = sqrt(sHW / max(sHI, T(1.0e-30)))
    βmaxI = min(plan.beta_max, λ_I)
    βmaxW = min(plan.beta_max, λ_W)
    sWW = _boxsum12_zero(T, hW .* hW, w)
    sII = _boxsum12_zero(T, hI .* hI, w)
    sWI = _boxsum12_zero(T, hW .* hI, w)
    βI = clamp.(sWI ./ max.(sWW, T(1.0e-20)), -βmaxI, zero(T))
    βW = clamp.(sWI ./ max.(sII, T(1.0e-20)), -βmaxW, zero(T))
    outI = I .- βI .* hW
    outW = W .- βW .* hI
    ρ = sum(hW .* hI) / sqrt(max(sHW * sHI, T(1.0e-30)))
    info = (ρ_hp = ρ, σ_hW = sqrt(sHW / T(length(hW))), σ_hI = sqrt(sHI / T(length(hI))))
    return outW, outI, info
end

"""
    acnr_kalender(W, I, plan::ACNRKalenderPlan) -> (W′, I′, info)

Pure counterpart of `apply_acnr_kalender!` on basis volumes `(nx, ny, nz)`:
`passes` unrolled rounds of {replicate-padded separable Gaussian high-pass,
global HF-ratio bound, `(2w+1)²` zero-padded box-summed `sWI/sWW` regression
slope clamped to `[-β_max, 0]`, per-pixel linear correction}.  `info` holds
the last pass's `(ρ_hp, σ_hW, σ_hI)` exactly like the legacy return value.
"""
function acnr_kalender(W::AbstractArray{<:Any, 3}, I::AbstractArray{<:Any, 3},
        plan::ACNRKalenderPlan{T}) where {T}
    size(W) == size(I) || error("acnr_kalender: W $(size(W)) and I $(size(I)) must match")
    plan.passes >= 1 || error("acnr_kalender: passes must be ≥ 1")
    Wc, Ic, info = _acnr_kalender_pass(W, I, plan)
    for _ in 2:plan.passes
        Wc, Ic, info = _acnr_kalender_pass(Wc, Ic, plan)
    end
    return Wc, Ic, info
end

# =============================================================================
#  3. z-median  (oracle: apply_median_z!)
# =============================================================================

"""
    MedianZPlan(; adjacent_slices = 1, boundary = :replicate)

Host constants for [`median_z`](@ref).  `boundary = :replicate` (default)
pads the z-axis with the edge slices and applies ONE fixed `2n+1` sorting
network everywhere (exact on interior slices; on the `n` slices at each end
the legacy instead shrinks the window and takes the LOWER median of the
available slices, so e.g. for `n = 1` the legacy end slice is `min(s₁, s₂)`
while replicate gives `s₁`).  `boundary = :shrink` reproduces the legacy
exactly with a separate static network per boundary slice.
"""
struct MedianZPlan
    adjacent_slices::Int
    boundary::Symbol
end

function MedianZPlan(; adjacent_slices::Integer = 1, boundary::Symbol = :replicate)
    adjacent_slices >= 0 || error("MedianZPlan: adjacent_slices must be ≥ 0")
    boundary in (:replicate, :shrink) || error("MedianZPlan: boundary must be :replicate or :shrink")
    return MedianZPlan(Int(adjacent_slices), boundary)
end

# Lower median of slices klo:khi (static window), as an (nx, ny, 1) slab.
function _lower_median_slices(vol::AbstractArray{<:Any, 3}, klo::Int, khi::Int)
    m = khi - klo + 1
    s = _sort_network(ntuple(d -> vol[:, :, (klo + d - 1):(klo + d - 1)], m))
    return s[(m + 1) ÷ 2]
end

"""
    median_z(vol, plan::MedianZPlan) -> Array

Pure counterpart of `apply_median_z!`: per `(x, y)`, the median over a
`2n+1` z-window realised as a min/max sorting network (static, no `sort`).
"""
function median_z(vol::AbstractArray{<:Any, 3}, plan::MedianZPlan)
    n = plan.adjacent_slices
    n == 0 && return copy(vol)
    nx, ny, nz = size(vol)
    m = 2n + 1
    if plan.boundary === :replicate
        vp = cat(ntuple(_ -> vol[:, :, 1:1], n)..., vol, ntuple(_ -> vol[:, :, nz:nz], n)...; dims = 3)
        slabs = ntuple(d -> vp[:, :, d:(d + nz - 1)], m)
        return _sort_network(slabs)[n + 1]
    else
        if nz > 2n
            interior = _sort_network(ntuple(d -> vol[:, :, d:(d + nz - 2n - 1)], m))[n + 1]
            top = ntuple(k -> _lower_median_slices(vol, 1, k + n), n)
            bot = ntuple(kk -> _lower_median_slices(vol, nz - 2n + kk, nz), n)
            return cat(top..., interior, bot...; dims = 3)
        else
            return cat(ntuple(k -> _lower_median_slices(vol, max(1, k - n), min(nz, k + n)), nz)...; dims = 3)
        end
    end
end

# =============================================================================
#  4. SF-JSD  (oracle: apply_sino_sfjsd_denoise)
# =============================================================================

"""
    SFJSDPlan(channels, I0; σ₀ = 0.0, T = Float32)

HOST capture of every data-derived SF-JSD constant, obtained by running the
legacy heuristics on `channels = [p_lo, p_hi]` (Float32 log sinograms
`(n_col, n_row, n_view)`) with per-channel flux `I0 = [I0_lo, I0_hi]`:

* `stride`  — `_sfjsd_pick_stride(max(_sfjsd_corr_length(p_lo), _sfjsd_corr_length(p_hi)))`
* `n_iter`  — `_sfjsd_pick_n_iter(min N)`
* `σ0_iter` — `σ₀★` (user value, or `_sfjsd_sure_optimize` on the mid row when
  `σ₀ ≤ 0`) decayed by `α = 0.7` per iteration, sequentially in Float32
* `σ1_rng[t], σ2_rng[t]` — per-iteration per-row MAD range scales
  `max(_sfjsd_mad_scale(U[:,k]), eps) · Σ_k` (Y-units), captured along the
  legacy trajectory (the legacy `_sfjsd_pass!` advances ξ between iterations)
* `radius_max` — max over rows/iterations of the legacy per-row kernel radius
  `max(1, ceil(3σ_sp))`; it is the static tap-loop bound of the functional pass
  (per-row radii are re-derived in-kernel and masked — see `sfjsd_pass`).

Also carried: the fixed 10-px reference Gaussian taps, `σ_cap = 12`, the 5×5
local-average half-width, `I0`, and diagnostics `corr_len`, `min_N`.
"""
struct SFJSDPlan{T, A, NI}
    stride::Int
    n_iter::Int
    σ0_iter::NTuple{NI, T}
    σ_cap::T
    radius_max::Int
    lavg_half::Int
    ref_kernel::Vector{T}
    ref_radius::Int
    I0_lo::T
    I0_hi::T
    σ1_rng::NTuple{NI, A}
    σ2_rng::NTuple{NI, A}
    n_sweeps::Int
    corr_len::Float64
    min_N::Float64
end

"""
    sfjsd_capture(channels, I0; σ₀ = 0.0, T = Float32) -> (plan, p_lo_ref, p_hi_ref)

HOST helper behind [`SFJSDPlan`](@ref).  Runs the legacy SF-JSD algorithm
*sequentially* (row by row, no threads) with the legacy internals, capturing
every data-derived constant into the plan, and also returns the resulting
denoised pair — the row-independent, single-threaded semantics of
`apply_sino_sfjsd_denoise`.  This is the oracle the parity tests use: the
threaded public driver assigns `slice_lo` / `slice_hi` inside its
`Threads.@threads` row loop while those names also exist at function scope
(SURE block), so under `nthreads() > 1` the closure shares them across threads
and rows get mixed (reproducible: only the first row of each thread's chunk is
corrupted).  With `nthreads() == 1` the public driver and this reference agree
bit for bit.
"""
function sfjsd_capture(channels::AbstractVector{<:AbstractArray{<:Real, 3}}, I0::AbstractVector{<:Real};
        σ₀::Real = 0.0, T::Type = Float32)
    L = _LEGACY
    length(channels) == 2 || error("SFJSDPlan: requires exactly 2 channels")
    length(I0) == 2 || error("SFJSDPlan: I0 must have length 2")
    p_lo = Float32.(channels[1]); p_hi = Float32.(channels[2])
    size(p_lo) == size(p_hi) || error("SFJSDPlan: channel shapes differ")
    n_col, n_row, n_view = size(p_lo)
    I0_lo = Float32(I0[1]); I0_hi = Float32(I0[2])
    N_lo = I0_lo .* exp.(-p_lo); N_hi = I0_hi .* exp.(-p_hi)
    min_N = Float64(min(minimum(N_lo), minimum(N_hi)))
    corr_len = max(L._sfjsd_corr_length(p_lo), L._sfjsd_corr_length(p_hi))
    stride = L._sfjsd_pick_stride(corr_len)
    n_iter = L._sfjsd_pick_n_iter(min_N)
    σ_cap = L._SFJSD_σ_cap
    α = L._SFJSD_α
    lavg_half = L._SFJSD_lavg ÷ 2
    # heavy low-pass reference + whitening (legacy expressions)
    p_ref_lo = L._sfjsd_sep_gauss_3d(p_lo, L._SFJSD_σ_ref_px)
    p_ref_hi = L._sfjsd_sep_gauss_3d(p_hi, L._SFJSD_σ_ref_px)
    w_lo = sqrt.(max.(N_lo, 1.0f0)); w_hi = sqrt.(max.(N_hi, 1.0f0))
    ξ_lo = w_lo .* (p_lo .- p_ref_lo)
    ξ_hi = w_hi .* (p_hi .- p_ref_hi)
    σ★ = if Float32(σ₀) > 0
        Float32(σ₀)
    else
        mid_r = n_row ÷ 2 + 1
        M_mid = hcat(vec(Float32.(ξ_lo[:, mid_r, :])), vec(Float32.(ξ_hi[:, mid_r, :])))
        L._sfjsd_sure_optimize(M_mid, n_col, n_view, stride; verbose = false)
    end
    σ0s = Float32[]
    s1s = Array{Float32, 3}[]
    s2s = Array{Float32, 3}[]
    radius_max = 1
    σ0 = σ★
    for _ in 0:(n_iter - 1)
        push!(σ0s, σ0)
        s1 = zeros(Float32, 1, 1, n_row); s2 = zeros(Float32, 1, 1, n_row)
        for r in 1:n_row
            M = hcat(vec(Float32.(ξ_lo[:, r, :])), vec(Float32.(ξ_hi[:, r, :])))
            F = LinearAlgebra.svd(M; full = false)
            U, Σ, V = F.U, F.S, F.V
            σ1 = min(σ0, σ_cap)
            σ2 = min(σ0 * sqrt(Σ[1] / max(Σ[2], 1.0f-12)), σ_cap)
            Λ1 = reshape(copy(U[:, 1]), n_col, n_view)
            Λ2 = reshape(copy(U[:, 2]), n_col, n_view)
            σ1_rng = max(L._sfjsd_mad_scale(Λ1), eps(Float32))
            σ2_rng = max(L._sfjsd_mad_scale(Λ2), eps(Float32))
            s1[1, 1, r] = σ1_rng * Σ[1]
            s2[1, 1, r] = σ2_rng * Σ[2]
            radius_max = max(radius_max, max(1, ceil(Int, 3 * Float64(σ1))), max(1, ceil(Int, 3 * Float64(σ2))))
            # advance ξ along the legacy trajectory (needed for iteration-2 scales)
            Λ1_d = similar(Λ1); Λ2_d = similar(Λ2)
            L._sfjsd_pass!(Λ1_d, Λ1, σ1, Λ1, Λ2, σ1_rng, σ2_rng, stride)
            L._sfjsd_pass!(Λ2_d, Λ2, σ2, Λ1, Λ2, σ1_rng, σ2_rng, stride)
            M_d = hcat(vec(Λ1_d), vec(Λ2_d)) * LinearAlgebra.Diagonal(Σ) * V'
            ξ_lo[:, r, :] .= reshape(view(M_d, :, 1), n_col, n_view)
            ξ_hi[:, r, :] .= reshape(view(M_d, :, 2), n_col, n_view)
        end
        push!(s1s, s1); push!(s2s, s2)
        σ0 *= α
    end
    # reference Gaussian taps exactly as `_sfjsd_sep_gauss_3d`
    σr = Float64(L._SFJSD_σ_ref_px)
    ref_radius = max(1, ceil(Int, 3σr))
    ks = Float32[exp(-(k^2) / (2σr^2)) for k in -ref_radius:ref_radius]
    ks ./= sum(ks)
    NI = n_iter
    plan = SFJSDPlan{T, Array{T, 3}, NI}(
        stride, n_iter, ntuple(t -> T(σ0s[t]), NI), T(σ_cap), radius_max, lavg_half,
        T.(ks), ref_radius, T(I0_lo), T(I0_hi),
        ntuple(t -> T.(s1s[t]), NI), ntuple(t -> T.(s2s[t]), NI), 10, corr_len, min_N)
    # inverse-whiten the sequentially advanced ξ → the legacy (single-threaded) result
    p_lo_d = ξ_lo ./ w_lo .+ p_ref_lo
    p_hi_d = ξ_hi ./ w_hi .+ p_ref_hi
    return plan, p_lo_d, p_hi_d
end

SFJSDPlan(channels::AbstractVector{<:AbstractArray{<:Real, 3}}, I0::AbstractVector{<:Real}; kwargs...) =
    sfjsd_capture(channels, I0; kwargs...)[1]

# Joint bilateral of `_sfjsd_pass!` on the (col, view, row) layout.
# `σ_sp` is a host scalar (component 1 → static legacy tap set) or a
# `(1,1,n_row)` array (component 2 → per-row radius/stride-phase re-derived
# in-kernel with `ceil`/`floor`, taps masked with `ifelse`, loop bound R).
function _sfjsd_bilateral12(::Type{T}, target::AbstractArray{<:Any, 3}, Λ1::AbstractArray{<:Any, 3},
        Λ2::AbstractArray{<:Any, 3}, σ_sp, inv2σr1, inv2σr2, stride::Int, R::Int, lh::Int) where {T}
    n1, n2, _ = size(target)
    mask = _valid_mask(T, n1, n2, R, R)
    tp = _pad_rep12(target, R, R)
    Λ1p = _pad_rep12(Λ1, R, R)
    Λ2p = _pad_rep12(Λ2, R, R)
    inv2σsp = one(T) ./ (T(2) .* σ_sp .* σ_sp .+ T(1.0e-30))
    if σ_sp isa Number
        rad = max(1, ceil(Int, 3 * Float64(σ_sp)))
        rad <= R || error("_sfjsd_bilateral12: host radius $rad exceeds the plan's radius_max = $R")
        taps = [(dc, dv) for dv in -rad:stride:rad for dc in -rad:stride:rad]
        rad_a = nothing; radm = nothing
    else
        rad_a = max.(one(T), ceil.(T(3) .* σ_sp))
        radm = rad_a .- T(stride) .* floor.(rad_a ./ T(stride))
        taps = [(dc, dv) for dv in -R:R for dc in -R:R]
    end
    local acc, wtot
    for (n, (dc, dv)) in enumerate(taps)
        pair = _shift12(mask, dc, dv, n1, n2, R, R)            # host (n1, n2, 1)
        cnt = max.(_boxsum12_zero(T, pair, lh), one(T))         # host
        d1 = _shift12(Λ1p, dc, dv, n1, n2, R, R) .- Λ1
        d2 = _shift12(Λ2p, dc, dv, n1, n2, R, R) .- Λ2
        Δ1 = _boxsum12_zero(T, d1 .* d1 .* pair, lh) ./ cnt
        Δ2 = _boxsum12_zero(T, d2 .* d2 .* pair, lh) ./ cnt
        nsp = T(-(dc * dc + dv * dv))
        lw = nsp .* inv2σsp .- Δ1 .* inv2σr1 .- Δ2 .* inv2σr2
        w = exp.(lw) .* pair
        if rad_a !== nothing
            act = (T(abs(dc)) .<= rad_a) .& (T(abs(dv)) .<= rad_a) .&
                (radm .== T(mod(-dc, stride))) .& (radm .== T(mod(-dv, stride)))
            w = ifelse.(act, w, zero(T))
        end
        tq = _shift12(tp, dc, dv, n1, n2, R, R)
        if n == 1
            acc = w .* tq
            wtot = w
        else
            acc = acc .+ w .* tq
            wtot = wtot .+ w
        end
    end
    return acc ./ max.(wtot, T(1.0e-30))
end

# One SF-JSD iteration on the (col, view, row) layout.
function _sfjsd_pass_cvr(A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3},
        plan::SFJSDPlan{T}, t::Int) where {T}
    _, V = gram_eigen(T, (A, B), plan.n_sweeps)
    Y1 = _combine((A, B), V, 1)
    Y2 = _combine((A, B), V, 2)
    nrm1 = sum(Y1 .* Y1; dims = (1, 2))
    nrm2 = sum(Y2 .* Y2; dims = (1, 2))
    ratio = sqrt.(nrm1 ./ max.(nrm2, T(1.0e-24)))               # Σ₁ / max(Σ₂, 1e-12)
    σ0_t = plan.σ0_iter[t]
    σ1 = min(σ0_t, plan.σ_cap)
    σ2 = min.(σ0_t .* sqrt.(ratio), plan.σ_cap)
    s1 = plan.σ1_rng[t]; s2 = plan.σ2_rng[t]
    inv1 = one(T) ./ (T(2) .* s1 .* s1 .+ T(1.0e-30))
    inv2 = one(T) ./ (T(2) .* s2 .* s2 .+ T(1.0e-30))
    Y1d = _sfjsd_bilateral12(T, Y1, Y1, Y2, σ1, inv1, inv2, plan.stride, plan.radius_max, plan.lavg_half)
    Y2d = _sfjsd_bilateral12(T, Y2, Y1, Y2, σ2, inv1, inv2, plan.stride, plan.radius_max, plan.lavg_half)
    return _reconstitute((Y1d, Y2d), V, 1), _reconstitute((Y1d, Y2d), V, 2)
end

"""
    sfjsd_pass(ξ_lo, ξ_hi, plan::SFJSDPlan, t = 1) -> (ξ_lo′, ξ_hi′)

CORE SF-JSD operator `D` (paper Eq 15) for iteration `t` on whitened residual
sinograms `(n_col, n_row, n_view)`: per-row Gram eigen → `Y₁, Y₂`; spatial
bandwidths `σ₁ = min(σ₀⁽ᵗ⁾, cap)` (host) and `σ₂ = min(σ₀⁽ᵗ⁾ √(Σ₁/Σ₂), cap)`
(in-kernel from `‖Y_k‖`, differentiable); product-of-channels range kernel
with 5×5 locally-averaged squared differences and the plan's captured MAD
scales; stride sub-sampling; reconstitution `Σ_k Yd_k V[j,k]`.
"""
function sfjsd_pass(ξ_lo::AbstractArray{<:Any, 3}, ξ_hi::AbstractArray{<:Any, 3},
        plan::SFJSDPlan{T}, t::Int = 1) where {T}
    1 <= t <= plan.n_iter || error("sfjsd_pass: iteration $t outside 1:$(plan.n_iter)")
    A = permutedims(ξ_lo, (1, 3, 2)); B = permutedims(ξ_hi, (1, 3, 2))
    oa, ob = _sfjsd_pass_cvr(A, B, plan, t)
    return permutedims(oa, (1, 3, 2)), permutedims(ob, (1, 3, 2))
end

"""
    sfjsd_denoise(p_lo, p_hi, plan::SFJSDPlan) -> (p_lo′, p_hi′)

Pure counterpart of `apply_sino_sfjsd_denoise` given a captured plan:
10-px truncated-renormalised reference Gaussian, Poisson whitening
`ξ = √max(N,1) · (p − p̄)`, `n_iter` unrolled [`sfjsd_pass`](@ref) iterations,
inverse whitening.
"""
function sfjsd_denoise(p_lo::AbstractArray{<:Any, 3}, p_hi::AbstractArray{<:Any, 3},
        plan::SFJSDPlan{T}) where {T}
    size(p_lo) == size(p_hi) || error("sfjsd_denoise: channel shapes differ")
    A = permutedims(p_lo, (1, 3, 2)); B = permutedims(p_hi, (1, 3, 2))
    N_lo = plan.I0_lo .* exp.(zero(T) .- A)
    N_hi = plan.I0_hi .* exp.(zero(T) .- B)
    w_lo = sqrt.(max.(N_lo, one(T)))
    w_hi = sqrt.(max.(N_hi, one(T)))
    ref_lo = _gauss12_trunc(A, plan.ref_kernel, plan.ref_radius)
    ref_hi = _gauss12_trunc(B, plan.ref_kernel, plan.ref_radius)
    ξ_lo = w_lo .* (A .- ref_lo)
    ξ_hi = w_hi .* (B .- ref_hi)
    for t in 1:plan.n_iter
        ξ_lo, ξ_hi = _sfjsd_pass_cvr(ξ_lo, ξ_hi, plan, t)
    end
    out_lo = ξ_lo ./ w_lo .+ ref_lo
    out_hi = ξ_hi ./ w_hi .+ ref_hi
    return permutedims(out_lo, (1, 3, 2)), permutedims(out_hi, (1, 3, 2))
end
