# =============================================================================
# BasisSimulator.Functional — photon-counting (PCCT) detector chain
# =============================================================================
#
# Pure, mutation-free, array-generic re-implementation of the PCCT detector
# chain that `simulate!(::PCCTWorkspace)` runs in `src/api/driver.jl`, from
# per-material PATH LENGTHS to per-bin (and optionally bin-combined) log
# sinograms.  Every stage is a pure function of arrays plus host scalars /
# an immutable `PCCTPlan`; nothing is mutated, every array size is a
# function of host config, there is no RNG inside any stage, and the only
# array primitives used are broadcasting, `reshape`, matrix products and
# static slices — so the whole chain traces under Reactant and
# differentiates under Enzyme.
#
# Numerical oracle: the legacy AcceleratedKernels code.
#   * `_dd_fused_spectral_plen!` (src/projection/dd_fast.jl)
#         I_b = Σ_e W[e,b] · bt[col,row,e] · exp(-Σ_m μ[m,e]·P_m)
#     (the bowtie factor `bt` is optional; when present the legacy kernel
#      clamps `min(exp(-L), 1e30)` before multiplying — reproduced.)
#   * `pcct_forward_project` (src/detector/photon_counting.jl)
#         p_b = -log(max(I_b, 1e-10) / I0_b)          (in T, I0_b = T(I0_bins_norm[b]))
#   * `apply_pcct_noise!` (src/detector/photon_counting.jl)
#         λ = I0_b·exp(-p_b);  N ~ Poisson(λ)  [HOST, `draw_pcct_counts`];
#         optional blend N ← λ + (1-nr)·(N-λ);  raw = N;
#         p_b ← -log(max(N, 1) / I0_b)
#   * MC pile-up apply (src/api/driver.jl, `simulate!(::PCCTWorkspace)`)
#         c_j = I0_j·exp(-p_j);  r_i = Σ_{j≤i} S[i,j]·c_j;  p_i ← -log(max(r_i,1e-10)/I0_i)
#     (only the LOWER triangle of S is read by the legacy register code —
#      `pcct_plan` masks S with `tril` so the matmul form is identical.)
#   * `apply_pcct_pileup_correction!` (src/correction/pcct_pileup_correction.jl)
#         forward substitution S·t = r  →  here `t = r · (S⁻¹)ᵀ`, S⁻¹ host-precomputed.
#   * bin combine (driver scatter step / `combine_pcct_bin_counts!`)
#         comb_k = Σ_{b∈group k} I0_b·exp(-p_b);  q_k = -log(max(comb_k,1e-10)/Σ_{b∈k} I0_b)
#
# Typing: EVERY array argument (data and config alike) is eltype-free
# (`AbstractArray{<:Any, N}` / `AbstractMatrix` / `AbstractVector`) so Reactant
# traced arrays — whose element type is `TracedRNumber{T}`, NOT `T` — dispatch
# whether they hold data or a `Reactant.to_rarray`-converted plan.  The scalar
# type `T` is never bound from an array's eltype: it comes from `PCCTPlan{T}`,
# from a typed scalar argument (`eps::T`, `noise_reduction::T`), or from an
# explicit `::Type{T}`.  Never `T(...)` with `T = eltype(x)` in stage code.
#
# Layout: per-material path lengths `P` are `(n_col, n_row, n_view, n_mat)`
# in cm, material index m ↔ mask value m-1 (0-based ids in the mask, exactly
# as `_dd_fused_spectral_plen!` does `mat = Int32(mask) + 1`).  Per-bin
# sinograms are `(n_col, n_row, n_view, n_bins)`; the legacy
# `Vector{Array{T,3}}` of bins is bin `b` = `bins[:, :, :, b]`.
#
# Noise design (straight-through): the exact integer Poisson draw is NOT
# differentiable and lives on the host (`draw_pcct_counts`, which reproduces
# `apply_pcct_noise!`'s RNG consumption order bit-for-bit).  The chain takes
# the realized counts `N_input` as an INPUT tensor and applies the
# deterministic remainder of `apply_pcct_noise!` (`pcct_counts_from_input`).
# For gradients, `pcct_noise_surrogate(p, I0, ε) = λ + √λ·ε` is the Gaussian
# reparameterization; feeding it the *implied* ε of a realized draw
# (`implied_noise_eps(N, λ) = (N-λ)/√λ`, a host constant) makes the forward
# value equal the exact draw (to rounding) while the reverse pass flows
# through ∂λ/∂p — the value comes from the exact counts, the gradient from
# the surrogate.  `pcct_chain(P, plan, N)` is the exact-count chain,
# `pcct_chain_surrogate(P, plan, ε)` the differentiable one.
#
# Reactant notes (verified 0.2.285, test/functional/reactant/smoke_pcct.jl):
#   * `Reactant.TracedRArray{T,N} <: AbstractArray{TracedRNumber{T},N}`, so any
#     `AbstractArray{T,N}` / `AbstractVector{T}` argument with `T<:AbstractFloat`
#     is a MethodError at trace time — hence the eltype-free signatures above.
#   * A broadcast `min.(x, T(1.0e30))` with `T` a method static parameter bound
#     from a host matrix's eltype sent the Reactant/Enzyme abstract interpreter
#     into unbounded recursion (StackOverflowError after ~1–5 min) even when the
#     branch containing it was never executed; the identical stage without that
#     constant traces in < 1 s (scratch bisect, isolate3.jl).  Do not build
#     scalar constants from array-derived `T` inside stage code.
#   * Pass the plan and combine matrix through `Reactant.to_rarray` and hand them
#     to the compiled function as arguments (`f(P, plan, G, I0g)`); mixed
#     traced×host operands also trace, but traced config keeps one code path.
#   * A compiled gradient thunk is called with the mode first:
#     `grad_c(Reverse, loss, P_r, Const(plan_r), …)`.
#
# Module requirements when included into `BasisSimulator.Functional`:
#   `import ..BasisSimulator as BS`, `using LinearAlgebra: LinearAlgebra`,
#   `using Random: Random` (host bridges call `BS._poisson_sample`,
#   `BS.dd_forward_project`, `LinearAlgebra.tril`, `Random.MersenneTwister`).
# =============================================================================

# -----------------------------------------------------------------------------
# Plan (immutable, host-side config).  `pcct_plan(ws)` is a temporary bridge
# that harvests the config-only tensors the legacy workspace already
# precomputes (μ table, DRM-shaped W, per-bin air calibration, pile-up S).
# -----------------------------------------------------------------------------

"""
    PCCTPlan{T}

Immutable host-side configuration for [`pcct_chain`](@ref).  All fields are
config-only tensors / scalars (never data-dependent):

- `μ_table::Matrix{T}`  — `(n_mat, n_E)` linear attenuation (1/cm) per material × energy.
- `W::Matrix{T}`        — `(n_E, n_bins)` energy→bin weights `I0·w(E)·η(E)·R(E,b)`
  (× centre-pixel bowtie when the scanner has one), exactly `ws.W_matrix_gpu[1:n_E, :]`.
- `I0_bins::Vector{T}`  — per-bin air calibration in `T` (what the legacy log step divides by).
- `I0_bins_f64::Vector{Float64}` — the same values in Float64 for the host Poisson sampler
  (`apply_pcct_noise!` samples in Float64).
- `pileup_St::Union{Nothing, Matrix{T}}` — `permutedims(tril(S))`, or `nothing` when pile-up is off.
- `pileup_Sinv_t::Union{Nothing, Matrix{T}}` — `permutedims(inv(tril(S)))` for [`pileup_correct`](@ref).
- `eps::T`              — count floor before the log (`1e-10` in every legacy site).
- `noise_reduction::T`  — `sim_opts.pcct_noise_reduction` blend (0 = exact counts).
- `view_chunks::Int`    — static number of view chunks for the energy-axis matmul.
"""
struct PCCTPlan{
        T <: AbstractFloat,
        MU <: AbstractMatrix, MW <: AbstractMatrix, VI <: AbstractVector, VF <: AbstractVector,
        SS <: Union{Nothing, AbstractMatrix}, SI <: Union{Nothing, AbstractMatrix},
    }
    μ_table::MU
    W::MW
    I0_bins::VI
    I0_bins_f64::VF
    pileup_St::SS
    pileup_Sinv_t::SI
    eps::T
    noise_reduction::T
    view_chunks::Int
end

# Array fields are typed abstractly (not `Matrix{T}`) so `Reactant.to_rarray(plan)`
# can rebuild the plan with device/traced arrays: passing the plan (and the
# combine matrix) to a compiled function as traced inputs keeps EVERY tensor op
# traced×traced — see the Reactant note in the header.
function PCCTPlan{T}(
        μ_table::AbstractMatrix, W::AbstractMatrix, I0_bins::AbstractVector, I0_bins_f64::AbstractVector,
        pileup_St, pileup_Sinv_t, eps::T, noise_reduction::T, view_chunks::Int,
    ) where {T <: AbstractFloat}
    return PCCTPlan{
        T, typeof(μ_table), typeof(W), typeof(I0_bins), typeof(I0_bins_f64),
        typeof(pileup_St), typeof(pileup_Sinv_t),
    }(μ_table, W, I0_bins, I0_bins_f64, pileup_St, pileup_Sinv_t, eps, noise_reduction, view_chunks)
end

"""
    pcct_plan(ws::BS.PCCTWorkspace; use_pileup = ws.use_pcct_pileup,
              noise_reduction = 0.0, view_chunks = 1, T = eltype(ws.μ_table))

Harvest a [`PCCTPlan`](@ref) from a legacy `PCCTWorkspace` (temporary bridge —
the plan only needs `ws.μ_table`, `ws.W_matrix_gpu`, `ws.I0_bins` and
`ws.pileup_S`, all config-only tensors the workspace precomputes on the host).
`noise_reduction` mirrors `sim_opts.pcct_noise_reduction` (the workspace does
not store it).  `view_chunks` must divide `size(P, 3)` or leave a static remainder
(both are host arithmetic on config).
"""
function pcct_plan(
        ws;
        use_pileup::Bool = ws.use_pcct_pileup,
        noise_reduction::Real = 0.0,
        view_chunks::Integer = 1,
        T::Type{<:AbstractFloat} = eltype(ws.μ_table),
    )
    n_E = length(ws.energies)
    μ_table = Matrix{T}(Array(ws.μ_table))
    W = Matrix{T}(Array(ws.W_matrix_gpu)[1:n_E, :])
    I0_f64 = Vector{Float64}(ws.I0_bins)
    S = (use_pileup && ws.pileup_S !== nothing) ? Matrix{Float64}(ws.pileup_S) : nothing
    return pcct_plan(μ_table, W, I0_f64, S; noise_reduction, view_chunks, T)
end

"""
    pcct_plan(μ_table, W, I0_bins_f64, S_or_nothing; noise_reduction=0, view_chunks=1, T)

Build a [`PCCTPlan`](@ref) from explicit host tensors (no workspace needed).
`S` is masked to its lower triangle (`tril`) because the legacy pile-up
register code only reads `S[i, j]` for `j ≤ i`.
"""
function pcct_plan(
        μ_table::AbstractMatrix, W::AbstractMatrix, I0_bins_f64::AbstractVector,
        S::Union{Nothing, AbstractMatrix};
        noise_reduction::Real = 0.0,
        view_chunks::Integer = 1,
        T::Type{<:AbstractFloat} = eltype(W),
    )
    size(μ_table, 2) == size(W, 1) ||
        throw(DimensionMismatch("μ_table is (n_mat, n_E) and W is (n_E, n_bins): n_E mismatch"))
    size(W, 2) == length(I0_bins_f64) ||
        throw(DimensionMismatch("one I0 value is required per PCCT bin"))
    view_chunks >= 1 || throw(ArgumentError("view_chunks must be ≥ 1"))
    St, Sinv_t = if S === nothing
        nothing, nothing
    else
        size(S, 1) == size(S, 2) == length(I0_bins_f64) ||
            throw(DimensionMismatch("pile-up S must be (n_bins, n_bins)"))
        S_lt = LinearAlgebra.tril(Matrix{Float64}(S))
        # T-round S the way the driver does (`S11 = T(S[1,1])` …) BEFORE transposing.
        S_T = Matrix{T}(S_lt)
        Matrix{T}(permutedims(S_T)), Matrix{T}(permutedims(inv(Matrix{Float64}(S_T))))
    end
    return PCCTPlan{T}(
        Matrix{T}(μ_table), Matrix{T}(W),
        Vector{T}(T.(Vector{Float64}(I0_bins_f64))), Vector{Float64}(I0_bins_f64),
        St, Sinv_t, T(1.0e-10), T(noise_reduction), Int(view_chunks),
    )
end

# -----------------------------------------------------------------------------
# Host helpers (config-only arithmetic)
# -----------------------------------------------------------------------------

# Static view-chunk ranges: `n_chunks` is host config, so the trip count and
# every slice size are fixed before any array code runs.  A non-dividing
# `n_chunks` leaves one static remainder range at the end.
function _view_chunk_ranges(n_view::Int, n_chunks::Int)
    step = cld(n_view, clamp(n_chunks, 1, n_view))
    return [lo:min(lo + step - 1, n_view) for lo in 1:step:n_view]
end

# Per-bin scalar vector → broadcastable (1, 1, 1, n_bins) tensor.
_bin_axis(v::AbstractVector) = reshape(v, 1, 1, 1, length(v))

# Concatenate view chunks along dim 3 with a static-trip-count pairwise
# reduce.  NOT `cat(parts...; dims = 3)`: a varargs splat of traced arrays
# sends the Reactant/Enzyme abstract interpreter into unbounded recursion
# (StackOverflowError at trace time) — pairwise binary `cat` traces cleanly.
_cat_views(parts::AbstractVector) = reduce((a, b) -> cat(a, b; dims = 3), parts)

"""
    combine_matrix(I0_bins_f64, groups, T) -> (G::Matrix{T}, I0_groups::Vector{T})

Host constructor for the bin-combine matmul: `G[b, k] = I0_b` if bin `b` is in
`groups[k]`, else 0; `I0_groups[k] = T(Σ_{b∈k} I0_b)` (summed in Float64 then
rounded, like the driver's `T(sum(I0_bins))`).
"""
function combine_matrix(
        I0_bins_f64::AbstractVector, groups::AbstractVector{<:AbstractVector{<:Integer}},
        ::Type{T},
    ) where {T <: AbstractFloat}
    n_bins = length(I0_bins_f64)
    G = zeros(T, n_bins, length(groups))
    I0g = zeros(T, length(groups))
    for (k, grp) in enumerate(groups)
        for b in grp
            1 <= b <= n_bins || throw(BoundsError(I0_bins_f64, b))
            G[b, k] = T(I0_bins_f64[b])
        end
        I0g[k] = T(sum(Float64(I0_bins_f64[b]) for b in grp))
    end
    return G, I0g
end

# -----------------------------------------------------------------------------
# 1. Spectral bin intensities
# -----------------------------------------------------------------------------

# Bowtie handling resolved by dispatch on the (host) type of `bt`, never on data.
_apply_bowtie(trans3::AbstractArray{<:Any, 3}, ::Nothing) = trans3
function _apply_bowtie(trans3::AbstractArray{<:Any, 3}, bt::AbstractArray{<:Any, 3})
    # trans3 is (n_col·n_row, nv, n_E); bt is (n_col, n_row, n_E) (view-independent).
    # The legacy kernel multiplies `min(exp(-L), 1e30) · bt`; the clamp is
    # inactive for every admissible input (path lengths and μ are ≥ 0, so
    # exp(-L) ≤ 1) and is omitted here, which keeps the spectral stage free of
    # any scalar-type dependence.
    n_px = size(trans3, 1); n_E = size(trans3, 3)
    btm = reshape(bt, n_px, 1, n_E)
    return trans3 .* btm
end

# Chunk of views: (n_col, n_row, nv, n_mat) → (n_col, n_row, nv, n_bins).
function _spectral_chunk(
        Pc::AbstractArray{<:Any, 4}, μ_table::AbstractMatrix, W::AbstractMatrix, bt,
    )
    n_col, n_row, nv, n_mat = size(Pc)
    n_E = size(μ_table, 2)
    n_bins = size(W, 2)
    n_px = n_col * n_row
    N = n_px * nv
    # L[ray, e] = Σ_m P[ray, m] · μ[m, e]   (matrix product over materials)
    L = reshape(Pc, N, n_mat) * μ_table
    trans = exp.(-L)
    trans_bt = reshape(_apply_bowtie(reshape(trans, n_px, nv, n_E), bt), N, n_E)
    # I[ray, b] = Σ_e trans[ray, e] · W[e, b]   (matrix product over energies)
    I = trans_bt * W
    return reshape(I, n_col, n_row, nv, n_bins)
end

"""
    spectral_bin_intensities(P, μ_table, W, bt = nothing; view_chunks = 1)

Per-bin detected intensities `(n_col, n_row, n_view, n_bins)` from per-material
path lengths `P` `(n_col, n_row, n_view, n_mat)`:

    I[·, b] = Σ_e W[e, b] · bt[col, row, e] · exp(-Σ_m μ_table[m, e] · P[·, m])

exactly the accumulation of `_dd_fused_spectral_plen!`, as two matrix
products (materials → energies, energies → bins).  `bt` is the optional
spectral bowtie `(n_col, n_row, n_E)` (the legacy `min(exp(-L), 1e30)` guard
is inactive for admissible inputs and omitted).  Memory: the energy axis
is materialized once per view chunk (`n_col·n_row·nv·n_E` elements);
`view_chunks` is host config (static trip count, static slices, `cat` on the
view axis).
"""
function spectral_bin_intensities(
        P::AbstractArray{<:Any, 4}, μ_table::AbstractMatrix, W::AbstractMatrix,
        bt::Union{Nothing, AbstractArray{<:Any, 3}} = nothing;
        view_chunks::Integer = 1,
    )
    size(P, 4) == size(μ_table, 1) ||
        throw(DimensionMismatch("P has $(size(P, 4)) materials but μ_table has $(size(μ_table, 1)) rows"))
    size(μ_table, 2) == size(W, 1) ||
        throw(DimensionMismatch("μ_table has $(size(μ_table, 2)) energies but W has $(size(W, 1)) rows"))
    ranges = _view_chunk_ranges(size(P, 3), Int(view_chunks))
    if length(ranges) == 1
        return _spectral_chunk(P, μ_table, W, bt)
    end
    parts = map(r -> _spectral_chunk(P[:, :, r, :], μ_table, W, bt), ranges)
    return _cat_views(parts)
end

"""
    spectral_bins_vjp(P, μ_table, W, bt, Ibar; view_chunks = 1) -> Pbar

Closed-form reverse-mode pullback of [`spectral_bin_intensities`](@ref):
given the cotangent `Ibar` `(n_col, n_row, n_view, n_bins)` returns
`Pbar = ∂⟨Ibar, I⟩/∂P` `(n_col, n_row, n_view, n_mat)`.  Per view chunk:

    Ltrans = Ibar · Wᵀ                  (n_ray, n_E)
    Lbar   = -(Ltrans ⊙ bt) ⊙ exp(-L)
    Pbar   = Lbar · μ_tableᵀ            (n_ray, n_mat)

so reverse mode needs one `(n_ray, n_E)` temporary pair instead of the three
a generic tape keeps (`L`, `trans`, `trans̄`), and never stores the energy
axis across chunks.
"""
function spectral_bins_vjp(
        P::AbstractArray{<:Any, 4}, μ_table::AbstractMatrix, W::AbstractMatrix,
        bt::Union{Nothing, AbstractArray{<:Any, 3}}, Ibar::AbstractArray{<:Any, 4};
        view_chunks::Integer = 1,
    )
    ranges = _view_chunk_ranges(size(P, 3), Int(view_chunks))
    if length(ranges) == 1
        return _spectral_chunk_vjp(P, μ_table, W, bt, Ibar)
    end
    parts = map(r -> _spectral_chunk_vjp(P[:, :, r, :], μ_table, W, bt, Ibar[:, :, r, :]), ranges)
    return _cat_views(parts)
end

_bowtie_pullback(Ltrans3::AbstractArray{<:Any, 3}, ::Nothing) = Ltrans3
function _bowtie_pullback(Ltrans3::AbstractArray{<:Any, 3}, bt::AbstractArray{<:Any, 3})
    n_px = size(Ltrans3, 1); n_E = size(Ltrans3, 3)
    return Ltrans3 .* reshape(bt, n_px, 1, n_E)
end

function _spectral_chunk_vjp(
        Pc::AbstractArray{<:Any, 4}, μ_table::AbstractMatrix, W::AbstractMatrix, bt,
        Ibar_c::AbstractArray{<:Any, 4},
    )
    n_col, n_row, nv, n_mat = size(Pc)
    n_E = size(μ_table, 2)
    n_bins = size(W, 2)
    n_px = n_col * n_row
    N = n_px * nv
    L = reshape(Pc, N, n_mat) * μ_table
    trans = exp.(-L)
    Ltrans = reshape(Ibar_c, N, n_bins) * permutedims(W)          # (N, n_E)
    Ltrans_bt = reshape(_bowtie_pullback(reshape(Ltrans, n_px, nv, n_E), bt), N, n_E)
    Lbar = -(Ltrans_bt .* trans)
    Pbar = Lbar * permutedims(μ_table)                              # (N, n_mat)
    return reshape(Pbar, n_col, n_row, nv, n_mat)
end

# -----------------------------------------------------------------------------
# 2. Per-bin log sinograms
# -----------------------------------------------------------------------------

"""
    bin_log_sinograms(I_bins, I0_bins, eps::T)

`p_b = -log(max(I_b, eps) / I0_b)` — the per-bin normalisation of
`pcct_forward_project` (division and log in `T`, `I0_b` already `T`-rounded,
exactly as the legacy `-log(max(ba[idx], eps) / I0_bin_T)`).
"""
function bin_log_sinograms(
        I_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector, eps::T,
    ) where {T <: AbstractFloat}
    I0r = _bin_axis(I0_bins)
    return @. -log(max(I_bins, eps) / I0r)
end

"""
    bin_log_sinograms_vjp(I_bins, I0_bins, eps, pbar) -> Ibar

Pullback of [`bin_log_sinograms`](@ref): `∂p/∂I = -1/I` where `I > eps`, 0 where the floor is active.
"""
function bin_log_sinograms_vjp(
        I_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector, eps::T, pbar::AbstractArray{<:Any, 4},
    ) where {T <: AbstractFloat}
    return @. pbar * ifelse(I_bins > eps, -one(T) / max(I_bins, eps), zero(T))
end

# -----------------------------------------------------------------------------
# 3. Noise
# -----------------------------------------------------------------------------

"""
    pcct_expected_counts(p_bins, I0_bins) -> λ

`λ_b = I0_b · exp(-p_b)` — the Poisson mean per bin and ray (count domain).
"""
function pcct_expected_counts(p_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector)
    I0r = _bin_axis(I0_bins)
    return @. I0r * exp(-p_bins)
end

"""
    draw_pcct_counts(p_bins, I0_bins_f64, seed_or_rng) -> N::Array{T,4}

HOST-ONLY exact integer Poisson draw reproducing `apply_pcct_noise!`
bit-for-bit: bins in order, elements in linear `(col, row, view)` order, each
`λ = I0_b · exp(-Float64(p))` sampled with `BS._poisson_sample` from a
`MersenneTwister` seeded like the workspace RNG (`seed = nothing` → 0).
Returns the raw integer draws stored as `T` (the legacy `T(N)`), with NO
`noise_reduction` blend — the blend is part of the deterministic map
[`pcct_counts_from_input`](@ref).  Composing the two with `noise_reduction = 0`
yields the legacy `raw_counts` exactly.
"""
function draw_pcct_counts(
        p_bins::AbstractArray{<:Any, 4}, I0_bins_f64::AbstractVector, rng::Random.AbstractRNG,
    )
    p_host = Array(p_bins)          # host copy; its (real) eltype is the count storage type
    T = eltype(p_host)
    n_col, n_row, n_view, n_bins = size(p_host)
    length(I0_bins_f64) == n_bins ||
        throw(DimensionMismatch("one I0 value is required per PCCT bin"))
    n_elem = n_col * n_row * n_view
    p_flat = reshape(p_host, n_elem, n_bins)
    N = Array{T}(undef, n_elem, n_bins)
    for b in 1:n_bins
        I0_bin = Float64(I0_bins_f64[b])
        @inbounds for idx in 1:n_elem
            λ = I0_bin * exp(-Float64(p_flat[idx, b]))
            N[idx, b] = T(Float64(BS._poisson_sample(rng, λ)))
        end
    end
    return reshape(N, n_col, n_row, n_view, n_bins)
end

function draw_pcct_counts(
        p_bins::AbstractArray{<:Any, 4}, I0_bins_f64::AbstractVector, seed::Union{Nothing, Integer},
    )
    rng = Random.MersenneTwister(seed === nothing ? 0 : Int(seed))
    return draw_pcct_counts(p_bins, I0_bins_f64, rng)
end

"""
    pcct_counts_from_input(p_bins, I0_bins, N_input, noise_reduction::T) -> (p_noisy, raw)

The deterministic remainder of `apply_pcct_noise!` given externally supplied
counts `N_input` (same shape as `p_bins`):

    λ   = I0_b · exp(-p_b)
    raw = N_input                                    (noise_reduction == 0)
        = λ + (1 - noise_reduction) · (N_input - λ)  (otherwise; host branch on config)
    p   = -log(max(raw, 1) / I0_b)

Returns the per-bin log sinograms and the pre-floor `raw` counts (the legacy
`raw_out`, true zeros preserved).  With `noise_reduction == 0` the counts pass
through untouched (no `λ + (N - λ)` round trip), so `raw == N_input` exactly.
"""
function pcct_counts_from_input(
        p_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector, N_input::AbstractArray{<:Any, 4},
        noise_reduction::T,
    ) where {T <: AbstractFloat}
    size(N_input) == size(p_bins) ||
        throw(DimensionMismatch("N_input must have the shape of p_bins"))
    I0r = _bin_axis(I0_bins)
    nr = noise_reduction
    raw = if iszero(nr)                  # host branch on config, not on data
        N_input
    else
        λ = @. I0r * exp(-p_bins)
        nr_scale = one(T) - nr
        @. λ + nr_scale * (N_input - λ)
    end
    p_noisy = @. -log(max(raw, one(T)) / I0r)
    return p_noisy, raw
end

"""
    pcct_noise_surrogate(p_bins, I0_bins, ε) -> N

Gaussian reparameterization of the Poisson draw, `N = λ + √λ · ε` with
`λ = I0_b · exp(-p_b)` and `ε` an INPUT tensor of standard-normal deviates
(same shape as `p_bins`).  Differentiable in `p_bins`; feed the result to
[`pcct_counts_from_input`](@ref).  Straight-through use: draw the exact
integer counts on the host, convert them to `ε = implied_noise_eps(N, λ)`,
and run the chain with this surrogate — the forward value reproduces the
exact draw to rounding while gradients flow through `∂λ/∂p`.
"""
function pcct_noise_surrogate(
        p_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector, ε::AbstractArray{<:Any, 4},
    )
    I0r = _bin_axis(I0_bins)
    λ = @. I0r * exp(-p_bins)
    return @. λ + sqrt(max(λ, 0)) * ε
end

"""
    implied_noise_eps(N, λ) -> ε

Host helper: the standard-normal deviate that makes the surrogate reproduce a
realized draw, `ε = (N - λ) / √λ` (0 where `λ` is 0).
"""
implied_noise_eps(N::AbstractArray, λ::AbstractArray) =
    @. ifelse(λ > 0, (N - λ) / sqrt(max(λ, 0)), 0 * λ)

# -----------------------------------------------------------------------------
# 4. MC pile-up (apply / correct)
# -----------------------------------------------------------------------------

# Batched (n_bins × n_bins) mixing over the bin axis via one matmul:
# out[ray, i] = Σ_j M[i, j] · x[ray, j]  ⇔  out = x · Mᵀ  (Mᵀ host-precomputed).
function _mix_bins(x::AbstractArray{<:Any, 4}, Mt::AbstractMatrix)
    n_col, n_row, n_view, n_bins = size(x)
    y = reshape(x, n_col * n_row * n_view, n_bins) * Mt
    return reshape(y, n_col, n_row, n_view, size(Mt, 2))
end

"""
    pileup_apply(p_bins, I0_bins, St, eps::T)

The driver's MC pile-up step: truth counts `c_j = I0_j·exp(-p_j)`, recorded
counts `r = S·c` (`St = permutedims(tril(S))`, one batched matmul over the bin
axis), then `-log(max(r, eps)/I0)` against the TRUTH `I0` — so
`I0_b · exp(-p_b)` still equals the recorded count downstream.
"""
function pileup_apply(
        p_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector, St::AbstractMatrix, eps::T,
    ) where {T <: AbstractFloat}
    I0r = _bin_axis(I0_bins)
    c = @. I0r * exp(-p_bins)
    r = _mix_bins(c, St)
    return @. -log(max(r, eps) / I0r)
end

"""
    pileup_correct(p_bins, I0_bins, Sinv_t, eps::T)

Model-based inverse of [`pileup_apply`](@ref) (`apply_pcct_pileup_correction!`):
recorded counts `r = I0·exp(-p)`, truth estimate `t = S⁻¹·r`
(`Sinv_t = permutedims(inv(tril(S)))`), then `-log(max(t, eps)/I0)`.
The legacy code does forward substitution; the matmul form is algebraically
identical and agrees to conditioning-level rounding.
"""
function pileup_correct(
        p_bins::AbstractArray{<:Any, 4}, I0_bins::AbstractVector, Sinv_t::AbstractMatrix, eps::T,
    ) where {T <: AbstractFloat}
    I0r = _bin_axis(I0_bins)
    r = @. I0r * exp(-p_bins)
    t = _mix_bins(r, Sinv_t)
    return @. -log(max(t, eps) / I0r)
end

# -----------------------------------------------------------------------------
# 5. Bin combine
# -----------------------------------------------------------------------------

"""
    combine_bin_counts(p_bins, G) -> counts (n_col, n_row, n_view, n_groups)

`counts[·, k] = Σ_b G[b, k] · exp(-p_b)` with `G` from [`combine_matrix`](@ref)
(`G[b, k] = I0_b · 1{b ∈ group k}`) — the streaming
`combine_pcct_bin_counts!` accumulation as one matmul.
"""
function combine_bin_counts(p_bins::AbstractArray{<:Any, 4}, G::AbstractMatrix)
    return _mix_bins(exp.(-p_bins), G)
end

"""
    combine_bins(p_bins, G, I0_groups, eps::T)

Combined log sinograms `q_k = -log(max(Σ_{b∈k} I0_b·exp(-p_b), eps) / Σ_{b∈k} I0_b)`
— the driver's combine (`groups = [[1:n_bins]]` gives its `combined_primary`
with `I0_total`).  `G, I0_groups = combine_matrix(I0_bins_f64, groups, T)`.
"""
function combine_bins(
        p_bins::AbstractArray{<:Any, 4}, G::AbstractMatrix, I0_groups::AbstractVector, eps::T,
    ) where {T <: AbstractFloat}
    comb = combine_bin_counts(p_bins, G)
    I0g = _bin_axis(I0_groups)
    return @. -log(max(comb, eps) / I0g)
end

"""
    combine_bins(p_bins, I0_bins_f64, groups, ::Type{T}; eps = T(1e-10))

Convenience form that builds the combine matrix on the host first; `T` is the
scalar type of the combine matrix (never inferred from `p_bins`, which may be
a traced array).
"""
function combine_bins(
        p_bins::AbstractArray{<:Any, 4}, I0_bins_f64::AbstractVector,
        groups::AbstractVector{<:AbstractVector{<:Integer}}, ::Type{T};
        eps::T = T(1.0e-10),
    ) where {T <: AbstractFloat}
    G, I0g = combine_matrix(I0_bins_f64, groups, T)
    return combine_bins(p_bins, G, I0g, eps)
end

# -----------------------------------------------------------------------------
# 6. Chain
# -----------------------------------------------------------------------------

# Noise stage resolved by dispatch on the host type of the counts argument.
_noise_stage(p::AbstractArray{<:Any, 4}, ::PCCTPlan, ::Nothing) = (p, nothing)
function _noise_stage(p::AbstractArray{<:Any, 4}, plan::PCCTPlan, N_input::AbstractArray{<:Any, 4})
    return pcct_counts_from_input(p, plan.I0_bins, N_input, plan.noise_reduction)
end

_pileup_stage(p::AbstractArray{<:Any, 4}, ::PCCTPlan, ::Nothing) = p
_pileup_stage(p::AbstractArray{<:Any, 4}, plan::PCCTPlan, St::AbstractMatrix) =
    pileup_apply(p, plan.I0_bins, St, plan.eps)

# raw_counts: verbatim draws when noise is on and pile-up off (the legacy
# capture at the sampling site); otherwise `I0·exp(-p)` of the final bins
# (`_capture_pcct_raw_counts`: expected counts λ with noise off, recorded
# fractional counts with pile-up on).
_raw_counts(p::AbstractArray{<:Any, 4}, plan::PCCTPlan, raw, ::Nothing) =
    raw === nothing ? pcct_expected_counts(p, plan.I0_bins) : raw
_raw_counts(p::AbstractArray{<:Any, 4}, plan::PCCTPlan, raw, ::AbstractMatrix) =
    pcct_expected_counts(p, plan.I0_bins)

"""
    pcct_chain(P, plan, N_input = nothing; bt = nothing) -> (; bins, raw_counts)

The `simulate!(::PCCTWorkspace)` chain for `binning_factor = 1`,
`use_pcct_scatter = false`, focal spot off, pile-up correction off, in the
legacy order:

1. [`spectral_bin_intensities`](@ref) (fused spectral DD accumulation),
2. [`bin_log_sinograms`](@ref) (`-log(max(I, eps)/I0_b)`),
3. noise: [`pcct_counts_from_input`](@ref) when `N_input` is an array of
   realized counts (host-drawn with [`draw_pcct_counts`](@ref) or the
   [`pcct_noise_surrogate`](@ref)); skipped when `N_input === nothing`,
4. [`pileup_apply`](@ref) when the plan carries `S`.

Returns `bins` `(n_col, n_row, n_view, n_bins)` — the legacy
`pcct_sino.bins[b]` is `bins[:, :, :, b]` — and `raw_counts`, defined as in
the driver: the verbatim draws when noise is on and pile-up off, otherwise
`I0_b·exp(-bins_b)` (λ with noise off; recorded fractional counts with
pile-up on).  `bt` is the optional per-pixel spectral bowtie (the legacy
driver folds the centre-pixel bowtie into `W` and passes none).
"""
function pcct_chain(
        P::AbstractArray{<:Any, 4}, plan::PCCTPlan{T},
        N_input::Union{Nothing, AbstractArray{<:Any, 4}} = nothing;
        bt::Union{Nothing, AbstractArray{<:Any, 3}} = nothing,
    ) where {T <: AbstractFloat}
    I = spectral_bin_intensities(P, plan.μ_table, plan.W, bt; view_chunks = plan.view_chunks)
    p = bin_log_sinograms(I, plan.I0_bins, plan.eps)
    p, raw = _noise_stage(p, plan, N_input)
    p = _pileup_stage(p, plan, plan.pileup_St)
    raw_counts = _raw_counts(p, plan, raw, plan.pileup_St)
    return (bins = p, raw_counts = raw_counts)
end

"""
    pcct_chain_combined(P, plan, G, I0_groups, N_input = nothing; bt = nothing)

[`pcct_chain`](@ref) followed by [`combine_bins`](@ref) with a host-built
combine matrix; returns the combined log sinograms `(n_col, n_row, n_view, n_groups)`.
"""
function pcct_chain_combined(
        P::AbstractArray{<:Any, 4}, plan::PCCTPlan{T}, G::AbstractMatrix, I0_groups::AbstractVector,
        N_input::Union{Nothing, AbstractArray{<:Any, 4}} = nothing;
        bt::Union{Nothing, AbstractArray{<:Any, 3}} = nothing,
    ) where {T <: AbstractFloat}
    bins = pcct_chain(P, plan, N_input; bt).bins
    return combine_bins(bins, G, I0_groups, plan.eps)
end

"""
    pcct_chain_surrogate(P, plan, ε; bt = nothing) -> (; bins, raw_counts)

[`pcct_chain`](@ref) with the DIFFERENTIABLE noise path: the counts are the
Gaussian reparameterization [`pcct_noise_surrogate`](@ref) of the input
deviates `ε` (same shape as the per-bin sinograms) instead of externally
realized integer draws.  With `ε = implied_noise_eps(N, λ)` of a host draw
`N` the forward value reproduces `pcct_chain(P, plan, N)` to rounding while
reverse mode flows through `∂λ/∂P` (straight-through: exact value, surrogate
gradient).
"""
function pcct_chain_surrogate(
        P::AbstractArray{<:Any, 4}, plan::PCCTPlan{T}, ε::AbstractArray{<:Any, 4};
        bt::Union{Nothing, AbstractArray{<:Any, 3}} = nothing,
    ) where {T <: AbstractFloat}
    I = spectral_bin_intensities(P, plan.μ_table, plan.W, bt; view_chunks = plan.view_chunks)
    p = bin_log_sinograms(I, plan.I0_bins, plan.eps)
    N = pcct_noise_surrogate(p, plan.I0_bins, ε)
    p, raw = pcct_counts_from_input(p, plan.I0_bins, N, plan.noise_reduction)
    p = _pileup_stage(p, plan, plan.pileup_St)
    raw_counts = _raw_counts(p, plan, raw, plan.pileup_St)
    return (bins = p, raw_counts = raw_counts)
end

"""
    pcct_chain_surrogate_combined(P, plan, G, I0_groups, ε; bt = nothing)

[`pcct_chain_surrogate`](@ref) followed by [`combine_bins`](@ref).
"""
function pcct_chain_surrogate_combined(
        P::AbstractArray{<:Any, 4}, plan::PCCTPlan{T}, G::AbstractMatrix, I0_groups::AbstractVector,
        ε::AbstractArray{<:Any, 4};
        bt::Union{Nothing, AbstractArray{<:Any, 3}} = nothing,
    ) where {T <: AbstractFloat}
    bins = pcct_chain_surrogate(P, plan, ε; bt).bins
    return combine_bins(bins, G, I0_groups, plan.eps)
end

# -----------------------------------------------------------------------------
# Host helpers to build inputs from legacy objects (tests / bridges only)
# -----------------------------------------------------------------------------

"""
    material_path_lengths(mask, n_mat, geom; volume_extent = nothing, T = Float32)

HOST bridge: per-material path lengths `(n_col, n_row, n_view, n_mat)` from a
0-based material mask via the LEGACY mono DD projector applied to one-hot
volumes (`P[:, :, :, m] = dd_forward_project(T.(mask .== m-1), geom)`).
Not a stage function — it calls the mutating legacy projector.
"""
function material_path_lengths(
        mask::AbstractArray{<:Integer, 3}, n_mat::Integer, geom;
        volume_extent = nothing, T::Type{<:AbstractFloat} = Float32,
    )
    mask_h = Array(mask)
    P = Array{T}(undef, geom.n_cols, geom.n_rows, geom.n_angles, n_mat)
    for m in 1:n_mat
        vol = T.(mask_h .== (m - 1))
        P[:, :, :, m] = Array(BS.dd_forward_project(vol, geom; volume_extent = volume_extent))
    end
    return P
end

"""
    stack_bins(bins::AbstractVector) -> Array{T,4}

HOST helper: legacy `Vector` of per-bin 3-D sinograms → `(n_col, n_row, n_view, n_bins)`.
"""
function stack_bins(bins::AbstractVector)
    first_bin = Array(bins[1])
    out = Array{eltype(first_bin)}(undef, size(first_bin)..., length(bins))
    for (b, bin) in enumerate(bins)
        out[:, :, :, b] = Array(bin)
    end
    return out
end
