# =============================================================================
# PCCT Pulse-Pileup Correction
# =============================================================================
#
# Inverse of the MC-LUT pile-up degradation applied inside
# `simulate!(::PCCTWorkspace, …)`.  Recovers an estimate of the truth-domain
# bin counts from the recorded ones — this is the analogue of what a clinical
# PCCT scanner's manufacturer-supplied recon software does before any
# downstream processing (material decomposition, VMI, etc.) sees the data.
#
# ## Provenance
#
# This module is BasisSim-original.  CatSim/XCIST ships PCCT spectral
# response matrices (e.g. `response_matrix/PC_spectral_response_*.mat`) and
# applies them in the forward direction, but does not open-source an
# inverse-pile-up step.  The MC migration matrix S used here is built by
# `compute_mc_pileup_matrix` in `src/detector/pcct/mc_response.jl`,
# following the standard cascaded photon-counting detector model:
#
#   Taguchi K, Frenkel J, Doi K, et al. "Modeling the performance of a
#   photon counting x-ray detector for CT: Energy response and pulse pileup
#   effects." Med Phys. 2011;38(2):1089-1102. doi:10.1118/1.3539602
#
#   Roessl E, Proksa R. "K-edge imaging in x-ray computed tomography using
#   multi-bin photon counting detectors." Phys Med Biol. 2007;52(15):4679-96.
#   doi:10.1088/0031-9155/52/15/020
#
# The inverse `t̂ = S \ r` is the standard linear-algebra unfolding once `S`
# is in hand (S lower-triangular by construction → forward substitution).
#
# ## Why this is decoupled
#
# `simulate!` always applies the forward pile-up degradation when
# `PCCTScanner(; pileup = true)` so its returned bins reflect what the
# detector actually records.  Pile-up correction lives **outside** the
# simulator (mirroring how BHC, ACNR, and capping live outside) so a notebook
# can choose:
#
#   - call `apply_pcct_pileup_correction!` first to undo pile-up before
#     calibration / decomposition (the production setup, matching clinical
#     scanners — calibration math then doesn't need to know about pile-up);
#   - or skip it to study the raw degraded bins.
#
# ## Math
#
# Given a recorded count vector `r ∈ ℝⁿ` (with `n_bins` entries per pixel)
# and the MC migration matrix `S ∈ ℝⁿˣⁿ` such that `r = S · t` (where `t`
# is the truth-domain count vector), recover an estimate `t̂ = S \ r`.
#
# `S` from `compute_mc_pileup_matrix` is **lower triangular** (pile-up only
# pushes counts UP in energy via energy summation), so `S \ r` is solved by
# forward substitution per pixel — no allocation, no LU factorization.
#
# ## Caveat (the "imperfect" piece)
#
# Noise on `r` propagates through `S⁻¹` and gets amplified by roughly
# `1 / S[i, i]` per bin.  At low aτ (S diagonals ≈ 0.7 – 0.9) this is a
# 10 – 40 % noise inflation per bin — the price of un-degrading the data,
# and the same imperfection real PCCT scanners' inverse-pile-up corrections
# carry.

# ─── Pile-up at each ray's own count rate ─────────────────────────────────────────────────────
# `S[:, :, k]` is the MC migration matrix at count rate `rates[k]`, a grid LINEAR in rate from 0
# (S = I) to the central air rate (pile-up loss is linear in rate·τ).  A ray's rate is the air
# rate scaled by its truth counts relative to the unattenuated central beam's, and its matrix is
# the linear blend of the two grid matrices around that rate.  Above the grid (impossible: truth
# ≤ air) the top matrix is used.
@inline function _pileup_blend(rate::T, r_min::T, dr::T, K::Int32) where {T}
    x = (rate - r_min) / dr
    x = clamp(x, zero(T), T(K - 1))
    k = min(Int32(floor(x)), K - Int32(2))
    return k, x - T(k)            # lower index (0-based) and the weight of the upper matrix
end
@inline _S(Sflat, k::Int32, i::Int32, j::Int32) = @inbounds Sflat[i + (j - Int32(1)) * Int32(4) + k * Int32(16)]
@inline function _S_at(Sflat, k::Int32, f::T, i::Int32, j::Int32) where {T}
    return (one(T) - f) * _S(Sflat, k, i, j) + f * _S(Sflat, k + Int32(1), i, j)
end

"""
    apply_pcct_pileup!(bins, I0, S, rates, rate_air)

Record the pile-up of every ray at that ray's own count rate: the truth counts
`c_b = I0[col, row, b]·exp(-bin_b)` are mixed by the lower-triangular `S(rate)` of the ray's
rate (`rate_air · Σ_b c_b / Σ_b I0_centre_b`), and the bins are rewritten as
`-log(recorded / I0[col, row, b])` so `I0 · exp(-bin) = recorded count` holds per ray.
"""
function apply_pcct_pileup!(
        bins::Vector{A}, I0::AbstractArray{T, 3}, S::AbstractArray{<:Real, 3},
        rates::AbstractVector{<:Real}, rate_air::Real,
    ) where {T <: AbstractFloat, A <: AbstractArray{T, 3}}
    length(bins) == 4 || error("apply_pcct_pileup!: specialized to 4 bins, got $(length(bins))")
    size(S, 1) == 4 && size(S, 2) == 4 && size(S, 3) == length(rates) >= 2 ||
        error("apply_pcct_pileup!: S must be 4×4×n_rates with n_rates ≥ 2")
    size(I0, 3) == 4 || error("apply_pcct_pileup!: I0 must have one [n_cols, n_rows] plane per bin (4), got $(size(I0, 3))")
    Sflat = similar(bins[1], T, length(S)); copyto!(Sflat, T.(vec(S)))
    nc = size(I0, 1); nr = size(I0, 2); m = Int32(nc * nr)
    counts_air = T(sum(view(I0, (nc + 1) ÷ 2, (nr + 1) ÷ 2, :)))   # the central ray's air counts
    r_min = T(rates[1]); dr = T(rates[2] - rates[1]); K = Int32(length(rates))
    eps = T(1.0e-10)
    let b1 = bins[1], b2 = bins[2], b3 = bins[3], b4 = bins[4], i0 = I0, m = m, Sf = Sflat,
            ra = T(rate_air), ca = counts_air, lm = r_min, dl = dr, K = K, eps = eps
        AK.foreachindex(b1) do idx
            ray = (Int32(idx - 1) % m) + Int32(1)
            i1 = i0[ray]; i2 = i0[ray + m]; i3 = i0[ray + 2m]; i4 = i0[ray + 3m]
            c1 = i1 * exp(-b1[idx]); c2 = i2 * exp(-b2[idx]); c3 = i3 * exp(-b3[idx]); c4 = i4 * exp(-b4[idx])
            k, f = _pileup_blend(ra * (c1 + c2 + c3 + c4) / ca, lm, dl, K)
            r1 = _S_at(Sf, k, f, Int32(1), Int32(1)) * c1
            r2 = _S_at(Sf, k, f, Int32(2), Int32(1)) * c1 + _S_at(Sf, k, f, Int32(2), Int32(2)) * c2
            r3 = _S_at(Sf, k, f, Int32(3), Int32(1)) * c1 + _S_at(Sf, k, f, Int32(3), Int32(2)) * c2 + _S_at(Sf, k, f, Int32(3), Int32(3)) * c3
            r4 = _S_at(Sf, k, f, Int32(4), Int32(1)) * c1 + _S_at(Sf, k, f, Int32(4), Int32(2)) * c2 + _S_at(Sf, k, f, Int32(4), Int32(3)) * c3 + _S_at(Sf, k, f, Int32(4), Int32(4)) * c4
            b1[idx] = -log(max(r1, eps) / i1); b2[idx] = -log(max(r2, eps) / i2)
            b3[idx] = -log(max(r3, eps) / i3); b4[idx] = -log(max(r4, eps) / i4)
        end
    end
    return bins
end

"""
    apply_pcct_pileup_correction!(bins, I0, S, rates, rate_air)

Undo [`apply_pcct_pileup!`](@ref) per ray: the recorded counts are known, the truth counts fix the
rate, so `S(rate)·t = r` is solved by forward substitution with the rate re-estimated from the
solution (three fixed-point rounds — `S(rate)` varies slowly in log rate, and the first round
already starts within a few percent).  The bins come back as log-transmissions against
`I0[col, row, b]`.
"""
function apply_pcct_pileup_correction!(
        bins::Vector{A}, I0::AbstractArray{T, 3}, S::AbstractArray{<:Real, 3},
        rates::AbstractVector{<:Real}, rate_air::Real,
    ) where {T <: AbstractFloat, A <: AbstractArray{T, 3}}
    length(bins) == 4 || error("apply_pcct_pileup_correction!: specialized to 4 bins, got $(length(bins))")
    size(S, 1) == 4 && size(S, 2) == 4 && size(S, 3) == length(rates) >= 2 ||
        error("apply_pcct_pileup_correction!: S must be 4×4×n_rates with n_rates ≥ 2")
    size(I0, 3) == 4 || error("apply_pcct_pileup_correction!: I0 must have one [n_cols, n_rows] plane per bin (4), got $(size(I0, 3))")
    Sflat = similar(bins[1], T, length(S)); copyto!(Sflat, T.(vec(S)))
    nc = size(I0, 1); nr = size(I0, 2); m = Int32(nc * nr)
    counts_air = T(sum(view(I0, (nc + 1) ÷ 2, (nr + 1) ÷ 2, :)))
    r_min = T(rates[1]); dr = T(rates[2] - rates[1]); K = Int32(length(rates))
    eps = T(1.0e-10)
    let b1 = bins[1], b2 = bins[2], b3 = bins[3], b4 = bins[4], i0 = I0, m = m, Sf = Sflat,
            ra = T(rate_air), ca = counts_air, lm = r_min, dl = dr, K = K, eps = eps
        AK.foreachindex(b1) do idx
            ray = (Int32(idx - 1) % m) + Int32(1)
            i1 = i0[ray]; i2 = i0[ray + m]; i3 = i0[ray + 2m]; i4 = i0[ray + 3m]
            r1 = i1 * exp(-b1[idx]); r2 = i2 * exp(-b2[idx]); r3 = i3 * exp(-b3[idx]); r4 = i4 * exp(-b4[idx])
            t1 = r1; t2 = r2; t3 = r3; t4 = r4
            for _ in 1:3
                k, f = _pileup_blend(ra * (t1 + t2 + t3 + t4) / ca, lm, dl, K)
                t1 = r1 / _S_at(Sf, k, f, Int32(1), Int32(1))
                t2 = (r2 - _S_at(Sf, k, f, Int32(2), Int32(1)) * t1) / _S_at(Sf, k, f, Int32(2), Int32(2))
                t3 = (r3 - _S_at(Sf, k, f, Int32(3), Int32(1)) * t1 - _S_at(Sf, k, f, Int32(3), Int32(2)) * t2) / _S_at(Sf, k, f, Int32(3), Int32(3))
                t4 = (r4 - _S_at(Sf, k, f, Int32(4), Int32(1)) * t1 - _S_at(Sf, k, f, Int32(4), Int32(2)) * t2 - _S_at(Sf, k, f, Int32(4), Int32(3)) * t3) / _S_at(Sf, k, f, Int32(4), Int32(4))
            end
            b1[idx] = -log(max(t1, eps) / i1); b2[idx] = -log(max(t2, eps) / i2)
            b3[idx] = -log(max(t3, eps) / i3); b4[idx] = -log(max(t4, eps) / i4)
        end
    end
    return bins
end

export apply_pcct_pileup!, apply_pcct_pileup_correction!
