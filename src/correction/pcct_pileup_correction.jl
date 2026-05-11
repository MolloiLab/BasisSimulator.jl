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
# `sim_opts.use_pcct_pileup = true` so its returned bins reflect what the
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

"""
    apply_pcct_pileup_correction!(bins::Vector{<:AbstractArray{T,3}},
                                   I0_bins::AbstractVector,
                                   S::AbstractMatrix) where {T<:AbstractFloat}

In-place pile-up correction on `bins` (per-bin log-line-integral sinograms).

Takes:
- `bins`     : `Vector` of n_bins GPU/CPU sinograms in the
               `-log(N_recorded / I0_truth[b])` form returned by
               `simulate!(::PCCTWorkspace, …)` with pile-up enabled.
- `I0_bins`  : truth I0 per bin (== `result.I0_bins` from `simulate!`).
- `S`        : MC pile-up migration matrix from
               `compute_mc_pileup_matrix` (== `result.pileup_S`).
               Lower-triangular by construction.

Replaces each `bins[b][idx]` with the truth-equivalent log-line-integral
`-log(t̂[b] / I0_bins[b])` where `t̂ = S \\ r` via forward substitution.

Currently specialized to the canonical 4-bin PCCT layout; passing any other
`length(bins)` raises an error.
"""
function apply_pcct_pileup_correction!(
        bins::Vector{A},
        I0_bins::AbstractVector,
        S::AbstractMatrix,
    ) where {T <: AbstractFloat, A <: AbstractArray{T, 3}}
    n_bins = length(bins)
    n_bins == 4 || error("apply_pcct_pileup_correction!: specialized to 4 bins, got $(n_bins)")
    size(S) == (4, 4) || error("apply_pcct_pileup_correction!: S must be 4×4, got $(size(S))")
    length(I0_bins) == 4 || error("apply_pcct_pileup_correction!: I0_bins must have 4 entries")

    # All inputs share `idx` index space; capture them by `let` so the AK
    # closure pulls scalars/refs by value.
    eps_corr = T(1.0e-10)
    let b1 = bins[1], b2 = bins[2], b3 = bins[3], b4 = bins[4],
            I0_1 = T(I0_bins[1]), I0_2 = T(I0_bins[2]),
            I0_3 = T(I0_bins[3]), I0_4 = T(I0_bins[4]),
            S11 = T(S[1, 1]),
            S21 = T(S[2, 1]), S22 = T(S[2, 2]),
            S31 = T(S[3, 1]), S32 = T(S[3, 2]), S33 = T(S[3, 3]),
            S41 = T(S[4, 1]), S42 = T(S[4, 2]), S43 = T(S[4, 3]), S44 = T(S[4, 4]),
            eps = eps_corr
        AK.foreachindex(b1) do idx
            # Recorded counts from log-line-integrals (against truth I0).
            r1 = I0_1 * exp(-b1[idx])
            r2 = I0_2 * exp(-b2[idx])
            r3 = I0_3 * exp(-b3[idx])
            r4 = I0_4 * exp(-b4[idx])
            # Forward substitution: solve S · t̂ = r (S lower triangular).
            t1 = r1 / S11
            t2 = (r2 - S21 * t1) / S22
            t3 = (r3 - S31 * t1 - S32 * t2) / S33
            t4 = (r4 - S41 * t1 - S42 * t2 - S43 * t3) / S44
            # Back to log-line-integrals against truth I0.
            b1[idx] = -log(max(t1, eps) / I0_1)
            b2[idx] = -log(max(t2, eps) / I0_2)
            b3[idx] = -log(max(t3, eps) / I0_3)
            b4[idx] = -log(max(t4, eps) / I0_4)
        end
    end
    return bins
end

export apply_pcct_pileup_correction!
