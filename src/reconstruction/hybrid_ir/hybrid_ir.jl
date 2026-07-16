# =============================================================================
# Hybrid IR — strength-keyed parameter table for the OS-PWLS hot path
# =============================================================================
#
# This file ONLY defines `HIRParams` + `get_hir_params(strength)`.  The actual
# reconstruction is `reconstruct!(::HIRReconWorkspace, …)` in
# `src/api/driver.jl`, which runs ordered-subsets PWLS with a Huber prior.
#
# Algorithm provenance (Fessler / U-Michigan school — NOT a TIGRE port):
#   • Sauer & Bouman 1993 — PWLS for transmission tomography
#   • Fessler 1994 — IEEE TMI 13(2):290-300, PWLS framework
#   • Erdoğan & Fessler 1999 — PMB 44, 2835-2851, ordered-subsets PWLS for
#     transmission CT.  Reference impl: MIRT `transmission/tpl_os_sps.m`
#   • Huber 1964 — robust regression.  MIRT `penalty/huber_pot.m` /
#     `huber_dpot.m` for potential + derivative; we GPU-port `huber_dpot`
#     in `src/reconstruction/ir/utils.jl::compute_huber_gradient!`.
#
# `strength` is a PERCENTAGE in 10 % steps: 0 (pure FBP) … 100 (max smoothing),
# matching how vendors expose the dial (GE ASIR-V is literally "ASIR-V 60 %").
# The anchor table below is defined at 0/20/40/60/80/100 and the intermediate
# decades are linearly interpolated, so the five anchors reproduce the historical
# strength-1..5 levels EXACTLY (level n ⇔ strength = 20n) with no re-tuning.
#
# The table targets the noise-reduction performance of vendor IR (GE ASIR-V,
# Siemens SAFIRE, Philips iDose-4, Canon AIDR 3D) as reported in Geyer 2015,
# Willemink & Noël 2019, Ghetti PMC5714520.  Those papers describe the TARGET;
# the algorithm itself is the open-literature Fessler PWLS substitute, since the
# vendor algorithms are proprietary.
# =============================================================================

export get_hir_params, HIRParams

"""
    HIRParams

Parameters for OS-PWLS HIR reconstruction at a given strength.
Tuned via sensitivity analysis to land in the SAFIRE / ASIR-V / iDose-4 /
AIDR 3D noise-reduction band reported in Ghetti PMC5714520 + Geyer 2015.

# Fields
- `strength`: Strength percentage, 0–100 in steps of 10.  Higher → more
  regularization → more noise reduction.  `0` is pure FBP.
- `lambda`: Regularization strength (Huber prior weight).
- `nepochs`: Number of OS epochs.  One epoch with `n_subsets` subsets ≈
  `n_subsets` full-data iterations of convergence (Erdoğan & Fessler 1999).
  `0` at `strength = 0`, which skips the PWLS loop entirely.
- `n_subsets`: Number of ordered subsets — fixed at 12 for all strengths.
- `huber_delta`: Huber penalty edge threshold δ.  Below δ the penalty is
  quadratic (smoothing); above δ it's linear (edge-preserving).
- `relaxation`: SIRT-style relaxation parameter on the data-fit update,
  chosen `< 1.0` for stability at higher `lambda`.
- `target_noise_reduction`: Expected noise reduction band `(min%, max%)`.
"""
struct HIRParams
    strength::Int
    lambda::Float32
    nepochs::Int
    n_subsets::Int
    huber_delta::Float32
    relaxation::Float32
    target_noise_reduction::Tuple{Int, Int}  # (min%, max%)
end

# Ordered subsets: fixed at 12 for every strength.  One OS epoch ≈ 12 full-data
# iterations of convergence (Erdoğan & Fessler 1999).
const _HIR_N_SUBSETS = 12

# Anchor table.  Tuning notes (v29 HIR-DISCOVER / HIR-FIX-WEIGHTS): V_inv
# normalization (~0.03) and the statistical weights (~0.14 mean) suppress the
# effective regularization, so λ must be O(1-10); relaxation < 1.0 is needed for
# stability at higher λ. The exact-DD solver converts this legacy calibration
# to its adjoint normalization internally; Siddon retains the historical scale.
#
# The 0 row is pure FBP; its δ and relaxation are inert (nepochs = 0) and carry
# the 20 % values so that interpolation toward strength = 10 is continuous.
#
# λ must stay STRICTLY increasing across the anchors.  The historical level-4/5
# pair shared λ = 6, δ = 0.05 and differed only in `nepochs` (3 vs 4); with an
# interpolated dial that made 90 % round to 4 epochs and come out bit-identical
# to 100 %, so two adjacent settings did nothing.  The 100 % row now carries its
# own λ/δ/relaxation, which keeps every 10 % step distinct (asserted in
# `test/api.jl`) and monotone in measured σ.
const _HIR_ANCHORS = (
    #  %   λ       nepochs  δ        relaxation  target band
    (  0, 0.0f0,   0,       0.08f0,  0.5f0,      ( 0,  0)),
    ( 20, 1.0f0,   1,       0.08f0,  0.5f0,      ( 8, 15)),
    ( 40, 2.0f0,   2,       0.07f0,  0.4f0,      (15, 25)),
    ( 60, 4.0f0,   2,       0.06f0,  0.35f0,     (25, 35)),
    ( 80, 6.0f0,   3,       0.05f0,  0.3f0,      (30, 42)),
    (100, 8.0f0,   4,       0.045f0, 0.28f0,     (35, 50)),
)

@inline _hir_anchor(a) = HIRParams(a[1], a[2], a[3], _HIR_N_SUBSETS, a[4], a[5], a[6])

"""
    get_hir_params(strength::Integer) -> HIRParams

Parameter lookup for a strength **percentage**: `0` (pure FBP) … `100`
(maximum smoothing), in steps of 10.

`strength` is the single dial for the whole HIR pipeline — it is the only knob
`create_hir_recon_workspace` needs, and it moves λ, the Huber edge threshold, the
relaxation, and the epoch count together along one tuned trajectory.

# Strength → clinical use
| Strength | Noise red. | Use case                            |
|----------|------------|-------------------------------------|
| 0 %      |  0 %       | pure FBP (no PWLS refinement)       |
| 20 %     |  8–15 %    | preserve FBP texture (lung nodules) |
| 40 %     | 15–25 %    | light smoothing                     |
| 60 %     | 25–35 %    | standard clinical (recommended)     |
| 80 %     | 30–42 %    | strong smoothing                    |
| 100 %    | 35–50 %    | maximum noise reduction             |

The odd decades (10, 30, 50, 70, 90) are linearly interpolated between the
neighbouring anchors; `nepochs` rounds to the nearest whole epoch (never below 1
once `strength > 0`).

# Vendor-equivalent target bands (clinical-validation sources, NOT
# algorithm sources — see file header)
- GE ASIR-V — `strength` IS the ASIR-V percentage.
- Siemens SAFIRE — `strength` 20/40/60/80/100 = SAFIRE S1-S5.
- Philips iDose-4 — `strength` 20/40/60/80/100 ≈ iDose-4 levels 1-5.
- Canon AIDR 3D — `strength` 40/60/80 ≈ Mild / Standard / Strong.
"""
function get_hir_params(strength::Integer)
    if !(0 ≤ strength ≤ 100) || strength % 10 != 0
        # The old API was an integer level 1..5.  Every such value fails the
        # multiple-of-10 test, so a stale call errors loudly here instead of
        # silently reconstructing at 3 % strength.
        hint = 1 ≤ strength ≤ 5 ?
            "  `strength` is now a percentage: the old level $strength is `strength = $(20 * strength)`." : ""
        throw(ArgumentError(
            "HIR strength must be 0-100 in steps of 10, got $strength.$hint"
        ))
    end

    # Exact anchors return their stored constants verbatim (bit-identical to the
    # historical strength-1..5 table); only the odd decades interpolate.
    for a in _HIR_ANCHORS
        a[1] == strength && return _hir_anchor(a)
    end

    i = findlast(a -> a[1] < strength, _HIR_ANCHORS)::Int
    lo, hi = _HIR_ANCHORS[i], _HIR_ANCHORS[i + 1]
    t = Float32(strength - lo[1]) / Float32(hi[1] - lo[1])
    lerp(x, y) = x + t * (y - x)

    return HIRParams(
        Int(strength),
        lerp(lo[2], hi[2]),
        max(1, round(Int, lerp(Float32(lo[3]), Float32(hi[3])))),
        _HIR_N_SUBSETS,
        lerp(lo[4], hi[4]),
        lerp(lo[5], hi[5]),
        (round(Int, lerp(Float32(lo[6][1]), Float32(hi[6][1]))),
         round(Int, lerp(Float32(lo[6][2]), Float32(hi[6][2])))),
    )
end
