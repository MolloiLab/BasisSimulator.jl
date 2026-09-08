# =============================================================================
# BasisSimulator.Functional — the published n-channel VMI estimator
# =============================================================================
#
# Pure, mutation-free, array-generic port of the notebook-only production
# estimator `nchannel_profile_tile!` (docs/notebooks/04_pcct_vmi.jl, mirrored
# in notebooks 03 and 12).  Per ray it maximises the Poisson quasi-likelihood
# of the corrected channel counts `y_k = I0_k·exp(-h_k)` under the exact
# discrete polychromatic mean
#
#     λ_k(A, C) = Σ_e Φ[e,k] · exp(-μρ_I[e]·A − μρ_W[e]·C)
#
# via a nested profile: an inner scalar water Newton solve `C*(A)` inside an
# outer iodine Newton update with the Fisher Schur-complement profile
# curvature.  See `nchannel_NOTES.md` next to this file for the complete map of
# the notebook code and the exact correspondence.
#
# Rules honoured here (the Reactant / Enzyme contract):
#   * every function is a pure function of arrays + an immutable `NChannelPlan`;
#   * every loop has a HOST-constant trip count (28 bisections, 16 × 12 + 12
#     Newton steps); the notebook's per-ray `break`s become per-ray freeze
#     masks (`ifelse.`), so the fixed-count program computes the same function;
#   * no scalar indexing, no data-dependent slicing, no RNG, no try/catch,
#     no device→host round trips; the energy contraction is a matrix product
#     (`nE × 3K` moment table `[Φ | μρ_I⊙Φ | μρ_W⊙Φ]`) or, for per-ray
#     response tables, a broadcast-reduce over the energy axis;
#   * element type generic `T <: AbstractFloat`; every literal goes through `T`.
#
# Layout: channel log-transmissions are ONE tensor `h::(n_col, n_row, n_view, K)`
# (the notebook's `NTuple{K}` of 3-D arrays — `nchannel_stack_channels`
# converts).  Outputs are `(n_col, n_row, n_view)`.
# =============================================================================

const _NCH_BS = isdefined(@__MODULE__, :BS) ? BS : parentmodule(@__MODULE__)

# -----------------------------------------------------------------------------
# Plan
# -----------------------------------------------------------------------------

"""
    NChannelPlan{T}

Immutable host-side configuration for [`nchannel_estimate_tile`](@ref).  Build
with [`nchannel_plan`](@ref).  Array fields are config-only tensors shaped so
they broadcast against a `(n_col, n_row, n_view)` tile:

- `moment_table` — `(nE, 3K)` global or `(n_col, n_row, nE, 3K)` per-ray:
  columns `[Φ_1..Φ_K | (μρ_I⊙Φ)_1..K | (μρ_W⊙Φ)_1..K]`.
- `total_table`  — `Σ_k Φ[…,k]`: `(nE,)` global or `(n_col, n_row, nE)` per-ray.
- `μρ_I`, `μρ_W` — `(nE,)` mass attenuation (cm²/g) of iodine and water.
- `I0`, `μI_eff`, `μW_eff` — `(1,1,1,K)` global or `(n_col, n_row, 1, K)` per-ray.
- `normal_II/IW/WW`, `attainable_max/min` — `(1,1,1)` or `(n_col, n_row, 1)`.

All remaining fields are the notebook's `nchannel_controls` plus the constants
that were hard-wired inside the kernel.
"""
struct NChannelPlan{T <: AbstractFloat, AM <: AbstractArray, AT <: AbstractArray,
        AV <: AbstractArray, AI <: AbstractArray, AN <: AbstractArray}
    K::Int
    nE::Int
    per_ray::Bool
    moment_table::AM
    total_table::AT
    μρ_I::AV
    μρ_W::AV
    I0::AI
    μI_eff::AI
    μW_eff::AI
    normal_II::AN
    normal_IW::AN
    normal_WW::AN
    attainable_max::AN
    attainable_min::AN
    # nchannel_controls
    iodine_bounds::NTuple{2, T}
    water_bounds::NTuple{2, T}
    outer_iterations::Int
    inner_iterations::Int
    bisection_iterations::Int
    max_iodine_step::T
    max_water_step::T
    parameter_tolerance::T
    fisher_condition_limit::T
    air_gate::T
    tile_views::Int
    # constants hard-wired in the notebook kernel
    count_floor::T          # 1f-6   (λ and y floors)
    curvature_floor::T      # 1f-12  (det0, FCC, Hprof, eigenvalue floors)
    bound_tol::T            # 2f-4   (bound-contact flag)
    init_fallback::NTuple{2, T}   # (0, 20) when the linear initializer is invalid
    I0_relerr::Float64
end

"""
    nchannel_plan(Φ, energies, I0; T = Float32, kwargs...) -> NChannelPlan{T}

Derive the estimator tables exactly as the notebooks do.

- `Φ::(nE, K)` with `I0::(K,)` — nb04's global tables (`W_applied`, `I0_bins`).
- `Φ::(n_col, n_row, nE, K)` with `I0::(n_col, n_row, K)` — nb03/nb12's per-ray
  tables (see [`nchannel_merge_channels`](@ref) for the energy-grid merge).

`μρ_I`/`μρ_W` default to `compute_mass_μ_at_energy` of iodine / water at
`Float64.(energies)` (the notebook recipe); pass vectors to override.  `scale`
multiplies `Φ` and `I0` AFTER the effective-energy tables are formed (nb04's
`nrows` slab factor).  The `I0_relerr < I0_tolerance` assertion of the
notebooks is kept (host `error`; `check_I0 = false` disables it).
"""
function nchannel_plan(
        Φ::AbstractArray, energies::AbstractVector, I0::AbstractArray;
        T::Type{<:AbstractFloat} = Float32,
        μρ_I = nothing, μρ_W = nothing,
        iodine_material = _NCH_BS.XA.Elements.Iodine,
        water_material = _NCH_BS.XA.Materials.water,
        scale::Real = 1,
        iodine_bounds = (-0.10, 0.40), water_bounds = (-2.0, 50.0),
        outer_iterations::Int = 16, inner_iterations::Int = 12,
        bisection_iterations::Int = 28,
        max_iodine_step = 0.05, max_water_step = 5.0,
        parameter_tolerance = 5.0e-5, fisher_condition_limit = 1.0e8,
        air_gate = 0.0, tile_views::Int = 8,
        count_floor = 1.0e-6, curvature_floor = 1.0e-12, bound_tol = 2.0e-4,
        init_fallback = (0.0, 20.0),
        check_I0::Bool = true, I0_tolerance::Real = 5.0e-5,
    )
    per_ray = ndims(Φ) == 4
    per_ray || ndims(Φ) == 2 || error("nchannel_plan: Φ must be (nE, K) or (n_col, n_row, nE, K)")
    nE = per_ray ? size(Φ, 3) : size(Φ, 1)
    K = per_ray ? size(Φ, 4) : size(Φ, 2)
    length(energies) == nE || error("nchannel_plan: length(energies) ≠ energy axis of Φ")
    if per_ray
        size(I0) == (size(Φ, 1), size(Φ, 2), K) ||
            error("nchannel_plan: per-ray I0 must be (n_col, n_row, K)")
    else
        length(I0) == K || error("nchannel_plan: I0 must have K entries")
    end

    ΦT = T.(Φ)
    I0T = T.(I0)
    # Both notebooks round the energy grid to Float32 before looking up μ/ρ
    # (`E = Float32.(energies)`; `compute_mass_μ_at_energy(…, Float64(e))`).
    E_T = T.(energies)
    μI = μρ_I === nothing ?
        T[_NCH_BS.compute_mass_μ_at_energy(iodine_material, Float64(e)) for e in E_T] :
        T.(collect(μρ_I))
    μW = μρ_W === nothing ?
        T[_NCH_BS.compute_mass_μ_at_energy(water_material, Float64(e)) for e in E_T] :
        T.(collect(μρ_W))

    # I0 consistency check (nb04 `nchannel_basis` / nb03 `build_nchannel_basis`).
    I0_from_Φ = per_ray ? dropdims(sum(Float64.(ΦT); dims = 3); dims = 3) :
        vec(sum(Float64.(ΦT); dims = 1))
    I0_relerr = maximum(abs.(I0_from_Φ .- Float64.(I0)) ./
        max.(Float64.(I0), eps(Float64)))
    if check_I0 && !(I0_relerr < I0_tolerance)
        error("nchannel_plan: applied response and I0 disagree (max relative error = $(I0_relerr)).")
    end

    # Effective-energy linearisation tables — the initializer only.  Formed in
    # T arithmetic with the same expressions as the notebooks so the values
    # match bit-for-bit when T = Float32.
    if per_ray
        n_col, n_row = size(ΦT, 1), size(ΦT, 2)
        Φsum = max.(I0T, eps(T))                                    # nb03: I0-normalised
        μI_eff = dropdims(sum(ΦT .* reshape(μI, 1, 1, nE, 1); dims = 3); dims = 3) ./ Φsum
        μW_eff = dropdims(sum(ΦT .* reshape(μW, 1, 1, nE, 1); dims = 3); dims = 3) ./ Φsum
        nII = dropdims(sum(abs2, μI_eff; dims = 3); dims = 3)
        nIW = dropdims(sum(μI_eff .* μW_eff; dims = 3); dims = 3)
        nWW = dropdims(sum(abs2, μW_eff; dims = 3); dims = 3)
        I0_b = reshape(I0T, n_col, n_row, 1, K)
        μI_eff_b = reshape(μI_eff, n_col, n_row, 1, K)
        μW_eff_b = reshape(μW_eff, n_col, n_row, 1, K)
        nII_b = reshape(nII, n_col, n_row, 1)
        nIW_b = reshape(nIW, n_col, n_row, 1)
        nWW_b = reshape(nWW, n_col, n_row, 1)
    else
        Φsum = vec(sum(ΦT; dims = 1))                               # nb04: Φ-normalised
        μI_eff = T[sum(view(ΦT, :, k) .* μI) / Φsum[k] for k in 1:K]
        μW_eff = T[sum(view(ΦT, :, k) .* μW) / Φsum[k] for k in 1:K]
        I0_b = reshape(I0T, 1, 1, 1, K)
        μI_eff_b = reshape(μI_eff, 1, 1, 1, K)
        μW_eff_b = reshape(μW_eff, 1, 1, 1, K)
        nII_b = fill(sum(abs2, μI_eff), 1, 1, 1)
        nIW_b = fill(sum(μI_eff .* μW_eff), 1, 1, 1)
        nWW_b = fill(sum(abs2, μW_eff), 1, 1, 1)
    end

    # Kernel tables carry the slab scale (nb04 multiplies Φ and I0 by nrows).
    s = T(scale)
    Φs = s .* ΦT
    I0s = s .* I0_b
    if per_ray
        n_col, n_row = size(Φs, 1), size(Φs, 2)
        moment_table = cat(Φs, Φs .* reshape(μI, 1, 1, nE, 1),
            Φs .* reshape(μW, 1, 1, nE, 1); dims = 4)
        total_table = dropdims(sum(Φs; dims = 4); dims = 4)
    else
        moment_table = hcat(Φs, Φs .* μI, Φs .* μW)
        total_table = vec(sum(Φs; dims = 2))
    end

    A_lo, A_hi = T(iodine_bounds[1]), T(iodine_bounds[2])
    C_lo, C_hi = T(water_bounds[1]), T(water_bounds[2])
    # Global attainable count range (kernel: Σ_{k,e} Φ·exp(-μI·A_lo − μW·C_lo) etc.)
    e_max = exp.(-(μI .* A_lo) .- (μW .* C_lo))
    e_min = exp.(-(μI .* A_hi) .- (μW .* C_hi))
    if per_ray
        att_max = reshape(sum(total_table .* reshape(e_max, 1, 1, nE); dims = 3), size(total_table, 1), size(total_table, 2), 1)
        att_min = reshape(sum(total_table .* reshape(e_min, 1, 1, nE); dims = 3), size(total_table, 1), size(total_table, 2), 1)
    else
        att_max = fill(sum(total_table .* e_max), 1, 1, 1)
        att_min = fill(sum(total_table .* e_min), 1, 1, 1)
    end

    return NChannelPlan{T, typeof(moment_table), typeof(total_table), typeof(μI),
        typeof(I0s), typeof(nII_b)}(
        K, nE, per_ray, moment_table, total_table, μI, μW,
        I0s, μI_eff_b, μW_eff_b, nII_b, nIW_b, nWW_b, att_max, att_min,
        (A_lo, A_hi), (C_lo, C_hi),
        outer_iterations, inner_iterations, bisection_iterations,
        T(max_iodine_step), T(max_water_step), T(parameter_tolerance),
        T(fisher_condition_limit), T(air_gate), tile_views,
        T(count_floor), T(curvature_floor), T(bound_tol),
        (T(init_fallback[1]), T(init_fallback[2])), I0_relerr)
end

"""
    nchannel_merge_channels(energies_per_channel, Φ_per_channel) -> (E, Φ)

Host helper reproducing nb03's `build_nchannel_basis` energy-grid merge: the
union of the per-channel energy grids (sorted, unique) and a per-ray
`Φ::(n_col, n_row, nE, K)` table with zeros where a channel has no support.
Each `Φ_per_channel[k]` is `(n_col, n_row, nE_k)` (already multiplied by that
channel's per-ray `I0`, i.e. absolute counts per energy).
"""
function nchannel_merge_channels(energies_per_channel, Φ_per_channel)
    E = sort!(unique(vcat(energies_per_channel...)))
    n_col, n_row = size(first(Φ_per_channel))[1:2]
    K = length(Φ_per_channel)
    Tf = eltype(first(Φ_per_channel))
    Φ = zeros(Tf, n_col, n_row, length(E), K)
    lookup = Dict(e => i for (i, e) in enumerate(E))
    for k in 1:K
        for (source_index, e) in enumerate(energies_per_channel[k])
            Φ[:, :, lookup[e], k] .= Φ_per_channel[k][:, :, source_index]
        end
    end
    return E, Φ
end

# -----------------------------------------------------------------------------
# Small array helpers
# -----------------------------------------------------------------------------

@inline _nch_sum4(x) = reshape(sum(x; dims = 4), size(x, 1), size(x, 2), size(x, 3))
@inline _nch_falses(x) = zero(x) .!= zero(x)
@inline _nch_trues(x) = zero(x) .== zero(x)
@inline _nch_bit(mask, v::T) where {T} = ifelse.(mask, v, zero(T))
@inline _nch_col(x::AbstractArray, n) = reshape(x, n, 1)
@inline _nch_col(x, n) = x     # scalars broadcast as they are

_nch_tile_ranges(n_view::Integer, tile::Integer) =
    (i:min(i + tile - 1, n_view) for i in 1:tile:n_view)

"""
    nchannel_stack_channels(hs::Tuple) -> (n_col, n_row, n_view, K)

Stack the notebook's `NTuple{K}` of channel sinograms on a fourth axis.
"""
nchannel_stack_channels(hs::Tuple) = cat(hs...; dims = 4)
nchannel_stack_channels(hs::AbstractVector{<:AbstractArray}) = cat(hs...; dims = 4)

"""
    nchannel_flags_u8(flag) / nchannel_counts_u8(count)

Host conversions of the `T`-valued diagnostic outputs to the notebook's `UInt8`.
"""
nchannel_flags_u8(flag::AbstractArray) = UInt8.(round.(Int, flag))
nchannel_counts_u8(count::AbstractArray) = UInt8.(min.(round.(Int, count), 255))

# -----------------------------------------------------------------------------
# Forward model over a tile
# -----------------------------------------------------------------------------

# exp(-μρ_I·A − μρ_W·C) for every ray × energy, as an (n_rays, nE) matrix.
# `C` may be a scalar (bracket endpoints).
@inline function _nch_exp_table(A::AbstractArray{<:Any, 3}, C, plan::NChannelPlan)
    n = length(A)
    μI = reshape(plan.μρ_I, 1, plan.nE)
    μW = reshape(plan.μρ_W, 1, plan.nE)
    return exp.(-(_nch_col(A, n) .* μI) .- (_nch_col(C, n) .* μW))
end

"""
    nchannel_moments(A, C, plan) -> (λ, dA, dC)

Exact polychromatic channel means and their analytic derivatives with respect
to the iodine (`A`) and water (`C`) area densities, each `(n_col, n_row,
n_view, K)`.  This is `nchannel_forward` of the notebooks, vectorised over the
tile.
"""
function nchannel_moments(A::AbstractArray{<:Any, 3}, C::AbstractArray{<:Any, 3}, plan::NChannelPlan)
    K = plan.K
    rs = size(A)
    E = _nch_exp_table(A, C, plan)
    M = if plan.per_ray
        z = reshape(E, rs[1], rs[2], rs[3], plan.nE, 1) .*
            reshape(plan.moment_table, rs[1], rs[2], 1, plan.nE, 3K)
        reshape(sum(z; dims = 4), rs[1], rs[2], rs[3], 3K)
    else
        reshape(E * plan.moment_table, rs[1], rs[2], rs[3], 3K)
    end
    λ = M[:, :, :, 1:K]
    dA = -M[:, :, :, (K + 1):(2K)]
    dC = -M[:, :, :, (2K + 1):(3K)]
    return λ, dA, dC
end

"""
    nchannel_total_counts(A, C, plan) -> Σ_k λ_k  as (n_col, n_row, n_view)

`C` may be a scalar.
"""
function nchannel_total_counts(A::AbstractArray{<:Any, 3}, C, plan::NChannelPlan)
    rs = size(A)
    E = _nch_exp_table(A, C, plan)
    if plan.per_ray
        z = reshape(E, rs[1], rs[2], rs[3], plan.nE) .*
            reshape(plan.total_table, rs[1], rs[2], 1, plan.nE)
        return reshape(sum(z; dims = 4), rs)
    else
        return reshape(E * plan.total_table, rs)
    end
end

# Score and Fisher blocks of the Poisson quasi-likelihood at (A, C).
function _nch_profile_terms(A, C, y, plan::NChannelPlan{T}) where {T}
    λ, dA, dC = nchannel_moments(A, C, plan)
    λf = max.(λ, plan.count_floor)
    r = one(T) .- y ./ λf
    gA = _nch_sum4(r .* dA)
    gC = _nch_sum4(r .* dC)
    FAA = _nch_sum4(dA .* dA ./ λf)
    FAC = _nch_sum4(dA .* dC ./ λf)
    FCC = _nch_sum4(dC .* dC ./ λf)
    return gA, gC, FAA, FAC, FCC
end

# Inner scalar solve C*(A) = argmin_C L(A, C): `inner_iterations` clamped
# Newton steps with per-ray freeze once |ΔC| ≤ tol·(1+|C|) (the notebook's
# `break`).  `active0` masks rays that take part at all.
function _nch_profile_water(A, C, y, plan::NChannelPlan{T}, active0) where {T}
    C_lo, C_hi = plan.water_bounds
    C_step = plan.max_water_step
    tol = plan.parameter_tolerance
    cf = plan.curvature_floor
    done = _nch_falses(C)
    n_used = zero(C)
    for _ in 1:plan.inner_iterations
        active = active0 .& .!done
        n_used = n_used .+ _nch_bit(active, one(T))
        _, gC, _, _, FCC = _nch_profile_terms(A, C, y, plan)
        C_new = clamp.(C .- clamp.(gC ./ max.(FCC, cf), -C_step, C_step), C_lo, C_hi)
        done_now = abs.(C_new .- C) .<= tol .* (one(T) .+ abs.(C))
        C = ifelse.(active, C_new, C)
        done = done .| (active .& done_now)
    end
    return C, n_used, done
end

# -----------------------------------------------------------------------------
# The estimator
# -----------------------------------------------------------------------------

"""
    nchannel_estimate_tile(h, plan) -> NamedTuple

Pure port of `nchannel_profile_tile!` for one tile `h::(n_col, n_row, n_view,
K)` of channel log-transmissions.  Returns

    (sino_iodine, sino_water, quality_flag, fisher = (AA, AC, CC),
     score_norm, outer_iterations, inner_iterations, aggregate_feasible)

all `(n_col, n_row, n_view)` arrays of `T` (`quality_flag`, the iteration
counters and `aggregate_feasible` hold exact small integers — see
[`nchannel_flags_u8`](@ref)).  Flag bits: 1 iodine at bound, 2 water at
bound, 4 not converged, 8 ill-conditioned Fisher or invalid initializer,
16 aggregate count outside the attainable range, 32 non-finite.
"""
function nchannel_estimate_tile(h::AbstractArray{<:Any, 4}, plan::NChannelPlan{T}) where {T}
    size(h, 4) == plan.K || throw(DimensionMismatch("nchannel_estimate_tile: h has $(size(h, 4)) channels, plan has $(plan.K)"))
    rs = (size(h, 1), size(h, 2), size(h, 3))
    A_lo, A_hi = plan.iodine_bounds
    C_lo, C_hi = plan.water_bounds
    A_step = plan.max_iodine_step
    tol = plan.parameter_tolerance
    cf = plan.curvature_floor
    half = T(0.5)
    z0 = zero(T)
    o1 = one(T)

    # Corrected counts (fractional after detector correction), floored.
    y = max.(plan.I0 .* exp.(-h), plan.count_floor)
    y_total = _nch_sum4(y)
    air = reshape(maximum(abs.(h); dims = 4), rs) .< plan.air_gate

    # K-channel linear initializer; everything after it is polychromatic.
    rhs_I = _nch_sum4(plan.μI_eff .* h)
    rhs_W = _nch_sum4(plan.μW_eff .* h)
    nII, nIW, nWW = plan.normal_II, plan.normal_IW, plan.normal_WW
    det0_raw = nII .* nWW .- nIW .* nIW
    init_valid = isfinite.(det0_raw) .& (det0_raw .> cf)
    det0 = ifelse.(init_valid, det0_raw, o1)
    A0 = ifelse.(init_valid,
        clamp.((nWW .* rhs_I .- nIW .* rhs_W) ./ det0, A_lo, A_hi),
        clamp(plan.init_fallback[1], A_lo, A_hi))
    C0 = ifelse.(init_valid,
        clamp.((nII .* rhs_W .- nIW .* rhs_I) ./ det0, C_lo, C_hi),
        clamp(plan.init_fallback[2], C_lo, C_hi))

    # Guaranteed-monotone aggregate equation Σ_k λ_k(A0, C) = y_total, used only
    # to stabilise the initial water value at the initial iodine value.
    total_lo = nchannel_total_counts(A0, C_lo, plan)
    total_hi = nchannel_total_counts(A0, C_hi, plan)
    bracketed = (total_lo .>= y_total) .& (total_hi .<= y_total)
    feasible = (plan.attainable_max .>= y_total) .& (plan.attainable_min .<= y_total)
    lo = zero(A0) .+ C_lo
    hi = zero(A0) .+ C_hi
    for _ in 1:plan.bisection_iterations
        mid = (lo .+ hi) .* half
        up = nchannel_total_counts(A0, mid, plan) .> y_total
        lo = ifelse.(up, mid, lo)
        hi = ifelse.(up, hi, mid)
    end
    C = ifelse.(bracketed, (lo .+ hi) .* half, C0)
    A = A0

    # Outer iodine Newton on the profile likelihood, inner water re-profiling.
    converged = _nch_falses(A)
    used_outer = zero(A)
    used_inner = zero(A)
    for _ in 1:plan.outer_iterations
        active_o = .!converged
        used_outer = used_outer .+ _nch_bit(active_o, o1)
        C, n_used, _ = _nch_profile_water(A, C, y, plan, active_o)
        used_inner = used_inner .+ n_used
        gA, _, FAA, FAC, FCC = _nch_profile_terms(A, C, y, plan)
        Hprof = max.(FAA .- FAC .* FAC ./ max.(FCC, cf), cf)
        A_new = clamp.(A .- clamp.(gA ./ Hprof, -A_step, A_step), A_lo, A_hi)
        conv_now = abs.(A_new .- A) .<= tol .* (o1 .+ abs.(A))
        A = ifelse.(active_o, A_new, A)
        converged = converged .| (active_o .& conv_now)
    end

    # Re-profile water at the final iodine iterate.
    C, n_used, c_converged = _nch_profile_water(A, C, y, plan, _nch_trues(A))
    used_inner = used_inner .+ n_used
    converged = converged .& c_converged

    # Final score and Fisher conditioning — recorded, never used to regularise.
    gA, gC, FAA, FAC, FCC = _nch_profile_terms(A, C, y, plan)
    score = sqrt.(gA .* gA .+ gC .* gC) ./ sqrt.(max.(FAA .+ FCC, cf))
    fisher_det = max.(FAA .* FCC .- FAC .* FAC, z0)
    fisher_trace = FAA .+ FCC
    fisher_disc = sqrt.(max.(fisher_trace .* fisher_trace .- T(4) .* fisher_det, z0))
    eig_max_raw = max.((fisher_trace .+ fisher_disc) .* half, cf)
    eig_min = max.(fisher_det ./ eig_max_raw, cf)
    eig_max = max.(eig_max_raw, eig_min)
    ill_conditioned = eig_max ./ eig_min .> plan.fisher_condition_limit

    bt = plan.bound_tol
    hit_A = (A .<= A_lo + bt) .| (A .>= A_hi - bt)
    hit_C = (C .<= C_lo + bt) .| (C .>= C_hi - bt)
    invalid = .!(isfinite.(A) .& isfinite.(C) .& isfinite.(score) .&
        isfinite.(FAA) .& isfinite.(FAC) .& isfinite.(FCC))
    flag = _nch_bit(hit_A, T(1)) .+ _nch_bit(hit_C, T(2)) .+
        _nch_bit(.!converged, T(4)) .+ _nch_bit(ill_conditioned .| .!init_valid, T(8)) .+
        _nch_bit(.!feasible, T(16)) .+ _nch_bit(invalid, T(32))

    # Air gate (disabled in production, `air_gate = 0`): compute-then-select.
    return (
        sino_iodine = ifelse.(air, z0, A),
        sino_water = ifelse.(air, z0, C),
        quality_flag = ifelse.(air, z0, flag),
        fisher = (AA = ifelse.(air, z0, FAA), AC = ifelse.(air, z0, FAC), CC = ifelse.(air, z0, FCC)),
        score_norm = ifelse.(air, z0, score),
        outer_iterations = ifelse.(air, z0, used_outer),
        inner_iterations = ifelse.(air, z0, used_inner),
        aggregate_feasible = _nch_bit(feasible, o1),
    )
end

"""
    nchannel_estimate(h, plan; tile_views = plan.tile_views) -> NamedTuple

Run [`nchannel_estimate_tile`](@ref) over view tiles of `h::(n_col, n_row,
n_view, K)` (host loop over static ranges, like the notebook's
`BS.tile_ranges(n_view, 8)`) and concatenate the outputs along the view axis.
"""
function nchannel_estimate(h::AbstractArray{<:Any, 4}, plan::NChannelPlan;
        tile_views::Int = plan.tile_views)
    n_view = size(h, 3)
    parts = [nchannel_estimate_tile(h[:, :, r, :], plan) for r in _nch_tile_ranges(n_view, tile_views)]
    cat3(f) = cat((f(p) for p in parts)...; dims = 3)
    return (
        sino_iodine = cat3(p -> p.sino_iodine),
        sino_water = cat3(p -> p.sino_water),
        quality_flag = cat3(p -> p.quality_flag),
        fisher = (AA = cat3(p -> p.fisher.AA), AC = cat3(p -> p.fisher.AC), CC = cat3(p -> p.fisher.CC)),
        score_norm = cat3(p -> p.score_norm),
        outer_iterations = cat3(p -> p.outer_iterations),
        inner_iterations = cat3(p -> p.inner_iterations),
        aggregate_feasible = cat3(p -> p.aggregate_feasible),
    )
end

"""
    nchannel_combine_rows(h, rows) -> (n_col, 1, n_view, K)

nb04's detector-row count combination: sum the count-domain equivalents of
the selected rows per channel and re-apply the logarithm,
`h_Σ = -log(max(Σ_r exp(-h_r), 1e-12) / n_rows)`.  The matching plan must be
built with `scale = length(rows)`.
"""
function nchannel_combine_rows(h::AbstractArray{<:Any, 4}, rows::AbstractUnitRange,
        ::Type{T} = Float32) where {T <: AbstractFloat}
    n = T(length(rows))
    s = sum(exp.(-h[:, rows, :, :]); dims = 2)
    return -log.(max.(s, T(1.0e-12)) ./ n)
end

# -----------------------------------------------------------------------------
# Lee-2025 total-likelihood bilateral filter (nb04 only)
# -----------------------------------------------------------------------------

"""
    NChannelTLBFPlan{T}

Static 25-tap (5×5) neighbourhood of `tlbf_filter_pair`: circular wrap along
views, non-wrapping (masked) along detector columns, each detector row
filtered independently (nb04 runs it on a single combined row).  Built by
[`nchannel_tlbf_plan`](@ref).
"""
struct NChannelTLBFPlan{T <: AbstractFloat, AI <: AbstractArray, AM <: AbstractArray}
    n_col::Int
    n_row::Int
    n_view::Int
    col_index::NTuple{5, AI}      # (n_col, 1, 1) clamped column index per dc
    col_valid::NTuple{5, AM}      # (n_col, 1, 1) 1 where col+dc is inside
    row_offset::AI                # (1, n_row, 1)  (row-1)·n_col
    view_offset::NTuple{5, AI}    # (1, 1, n_view) (mod1(view+dv, n_view)-1)·n_col·n_row
    spatial::NTuple{25, T}        # exp(-(dc²+dv²)/(2α1²)), dv outer, dc inner
    alpha1::T
    alpha2::T
end

"""
    nchannel_tlbf_plan(n_col, n_row, n_view; alpha1 = 0.9, alpha2, T = Float32)
"""
function nchannel_tlbf_plan(n_col::Int, n_row::Int, n_view::Int;
        alpha1::Real = 0.9, alpha2::Real, T::Type{<:AbstractFloat} = Float32)
    offsets = -2:2
    col_index = ntuple(i -> reshape(Int32.(clamp.((1:n_col) .+ offsets[i], 1, n_col)), n_col, 1, 1), 5)
    col_valid = ntuple(i -> reshape(T.(1 .<= (1:n_col) .+ offsets[i] .<= n_col), n_col, 1, 1), 5)
    row_offset = reshape(Int32.(((1:n_row) .- 1) .* n_col), 1, n_row, 1)
    view_offset = ntuple(j -> reshape(Int32.((mod1.((1:n_view) .+ offsets[j], n_view) .- 1) .* (n_col * n_row)), 1, 1, n_view), 5)
    a1 = T(alpha1)
    spatial = ntuple(25) do t
        dv = offsets[(t - 1) ÷ 5 + 1]
        dc = offsets[(t - 1) % 5 + 1]
        exp(-T(dc * dc + dv * dv) / (T(2) * a1 * a1))
    end
    return NChannelTLBFPlan{T, typeof(row_offset), typeof(col_valid[1])}(
        n_col, n_row, n_view, col_index, col_valid, row_offset, view_offset,
        spatial, a1, T(alpha2))
end

@inline _nch_gather(A::AbstractArray, L::AbstractArray{<:Integer}) = reshape(vec(A)[vec(L)], size(L))

"""
    nchannel_tlbf(sino_iodine, sino_water, expected, measured, tp) -> (sino_iodine, sino_water)

Pure `tlbf_filter_pair`: one normalised scalar weight per neighbour — the
spatial Gaussian times `exp(-(Δℓ)²/α2²)` with `Δℓ` the difference of the
Poisson log-likelihood `-λ + Y·log λ` of the neighbour's completed material
pair against the centre's — applied jointly to both bases.  `expected` is
`max(Σ_k λ_k, 1e-6)` from [`nchannel_total_counts`](@ref) and `measured` is
`Σ_k I0_k·exp(-h_k)` ([`nchannel_total_measured`](@ref)).
"""
function nchannel_tlbf(sino_I::AbstractArray{<:Any, 3}, sino_W::AbstractArray{<:Any, 3},
        expected::AbstractArray{<:Any, 3}, measured::AbstractArray{<:Any, 3},
        tp::NChannelTLBFPlan{T}) where {T}
    floor = T(1.0e-6)
    a2 = tp.alpha2
    Y = measured
    center_expected = max.(expected, floor)
    center_likelihood = -center_expected .+ Y .* log.(center_expected)
    weight_sum = zero(sino_I)
    iodine_sum = zero(sino_I)
    water_sum = zero(sino_W)
    for j in 1:5, i in 1:5              # dv outer, dc inner — the notebook's order
        t = (j - 1) * 5 + i
        L = tp.col_index[i] .+ tp.row_offset .+ tp.view_offset[j]
        nI = _nch_gather(sino_I, L)
        nW = _nch_gather(sino_W, L)
        likelihood_weight = if isinf(a2)
            one(T)
        elseif a2 <= zero(T)
            (i == 3 && j == 3) ? one(T) : zero(T)
        else
            ne = max.(_nch_gather(expected, L), floor)
            δ = (-ne .+ Y .* log.(ne)) .- center_likelihood
            exp.(-(δ .* δ) ./ (a2 * a2))
        end
        w = (tp.spatial[t] .* tp.col_valid[i]) .* likelihood_weight
        weight_sum = weight_sum .+ w
        iodine_sum = iodine_sum .+ w .* nI
        water_sum = water_sum .+ w .* nW
    end
    inverse_weight = one(T) ./ max.(weight_sum, eps(T))
    return (sino_iodine = iodine_sum .* inverse_weight, sino_water = water_sum .* inverse_weight)
end

"""
    nchannel_total_measured(h, plan) -> Σ_k I0_k·exp(-h_k)   (no floor; nb04 `total_measured_counts`)
"""
nchannel_total_measured(h::AbstractArray{<:Any, 4}, plan::NChannelPlan) =
    _nch_sum4(plan.I0 .* exp.(-h))

"""
    nchannel_total_expected(sino_iodine, sino_water, plan) -> max(Σ_k λ_k, 1e-6)   (nb04 `total_expected_counts`)
"""
nchannel_total_expected(A::AbstractArray{<:Any, 3}, C::AbstractArray{<:Any, 3}, plan::NChannelPlan{T}) where {T} =
    max.(nchannel_total_counts(A, C, plan), T(1.0e-6))

# -----------------------------------------------------------------------------
# nb04 angular anti-alias apodization (before the common-kernel FBP)
# -----------------------------------------------------------------------------

"""
    NChannelAngularPlan{T}

nb04's deterministic angular anti-alias projection (`nchannel_fbp_angular_response`
+ FFT/IFFT along the view axis) as a circulant matrix product: Fourier modes
through `pass_mode = min(n_view ÷ 2, ⌈π·N/4⌉)` pass with gain one, then a
raised-cosine roll-off over the angular oversampling margin.  `matrix` is
`(n_view, n_view)` in `T`; `response` keeps the notebook's Float64 window.
"""
struct NChannelAngularPlan{T <: AbstractFloat, AM <: AbstractArray}
    n_view::Int
    pass_mode::Int
    response::Vector{Float64}
    matrix::AM
end

"""
    nchannel_angular_plan(n_view, matrix_n; T = Float32)

`matrix_n` is the reconstruction matrix size (nb04: 512).
"""
function nchannel_angular_plan(n_view::Int, matrix_n::Int; T::Type{<:AbstractFloat} = Float32)
    pass_mode = min(n_view ÷ 2, ceil(Int, π * matrix_n / 4))
    response = [
        let mode = min(j - 1, n_view - (j - 1))
            mode <= pass_mode ? 1.0 :
            0.5 * (1 + cos(π * (mode - pass_mode) / (n_view ÷ 2 - pass_mode)))
        end
        for j in 1:n_view
    ]
    # Circular-convolution kernel c[m] = (1/N) Σ_j H_j e^{2πi jm/N}: real because
    # H is symmetric (H_j = H_{N-j}).
    c = [sum(response[j + 1] * cos(2π * j * m / n_view) for j in 0:(n_view - 1)) / n_view
         for m in 0:(n_view - 1)]
    # Y[:, v] = Σ_{v'} X[:, v'] · c[v − v']  ⇔  Y = X * matrix, matrix[v', v] = c[mod(v − v', N)]
    matrix = T.([c[mod(v - vp, n_view) + 1] for vp in 1:n_view, v in 1:n_view])
    return NChannelAngularPlan{T, typeof(matrix)}(n_view, pass_mode, response, matrix)
end

"""
    nchannel_angular_apodize(sino, ap::NChannelAngularPlan) -> same shape

Pure `nchannel_common_fbp_slice` pre-filter: `real(ifft(fft(sino, 3) .* H, 3))`
as one matrix product along the view axis.
"""
function nchannel_angular_apodize(sino::AbstractArray{<:Any, 3}, ap::NChannelAngularPlan)
    n_col, n_row, n_view = size(sino)
    n_view == ap.n_view || throw(DimensionMismatch("nchannel_angular_apodize: view count mismatch"))
    return reshape(reshape(sino, n_col * n_row, n_view) * ap.matrix, n_col, n_row, n_view)
end

# -----------------------------------------------------------------------------
# VMI synthesis and the pluggable chain
# -----------------------------------------------------------------------------

"""
    nchannel_vmi_alphas(energies; T = Float32) -> Vector{T}

`α_E = μρ_I(E)/μρ_w(E)` of `synth_vmi_2basis` for each VMI energy (host).
"""
function nchannel_vmi_alphas(energies; T::Type{<:AbstractFloat} = Float32,
        water_material = _NCH_BS.XA.Materials.water,
        iodine_material = _NCH_BS.XA.Elements.Iodine)
    return T[_NCH_BS.compute_mass_μ_at_energy(iodine_material, Float64(e)) /
             _NCH_BS.compute_mass_μ_at_energy(water_material, Float64(e)) for e in energies]
end

"""
    nchannel_synth_vmi(vol_water, vol_iodine, alphas) -> (nx, ny, nz, n_energies)

Pure `synthesize_vmi_stack`: `HU_E = 1000·(c_w − 1) + (c_I·1000)·α_E` with
`c_w`, `c_I` in g/cm³ (the notebook passes `iodine .* 1000` mg/mL).
"""
function nchannel_synth_vmi(vol_W::AbstractArray{<:Any, 3}, vol_I::AbstractArray{<:Any, 3},
        alphas::AbstractVector, ::Type{T} = Float32) where {T <: AbstractFloat}
    nx, ny, nz = size(vol_W)
    α = reshape(alphas, 1, 1, 1, length(alphas))
    return T(1000) .* (reshape(vol_W, nx, ny, nz, 1) .- one(T)) .+
        (reshape(vol_I, nx, ny, nz, 1) .* T(1000)) .* α
end

"""
    nchannel_vmi_chain(h, plan; fbp_iodine, fbp_water, synth, energies,
                       sino_filter = nothing, acnr = nothing, tile_views = plan.tile_views)

The published chain with the downstream stages injected as callables:

1. `nchannel_estimate` → basis sinograms;
2. optional `sino_filter(sino_iodine, sino_water, estimate) -> (; sino_iodine, sino_water)`
   (nb04's T-LBF);
3. `fbp_iodine(sino)` / `fbp_water(sino)` → basis volumes (per-basis kernels);
4. optional `acnr(vol_water, vol_iodine) -> (; water, iodine)`;
5. `synth(vol_water, vol_iodine, energies)` → VMI stack.
"""
function nchannel_vmi_chain(h::AbstractArray{<:Any, 4}, plan::NChannelPlan;
        fbp_iodine, fbp_water, synth, energies,
        sino_filter = nothing, acnr = nothing, tile_views::Int = plan.tile_views)
    est = nchannel_estimate(h, plan; tile_views = tile_views)
    sI, sW = est.sino_iodine, est.sino_water
    if sino_filter !== nothing
        f = sino_filter(sI, sW, est)
        sI, sW = f.sino_iodine, f.sino_water
    end
    vol_I = fbp_iodine(sI)
    vol_W = fbp_water(sW)
    if acnr !== nothing
        r = acnr(vol_W, vol_I)
        vol_W, vol_I = r.water, r.iodine
    end
    vmis = synth(vol_W, vol_I, energies)
    return (; estimate = est, sino_iodine = sI, sino_water = sW,
        vol_iodine = vol_I, vol_water = vol_W, vmis, energies)
end
