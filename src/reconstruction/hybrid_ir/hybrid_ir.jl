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
# The strength-1..5 lookup table targets the noise-reduction performance of
# vendor IR (GE ASIR-V, Siemens SAFIRE, Philips iDose-4, Canon AIDR 3D) as
# reported in Geyer 2015, Willemink & Noël 2019, Ghetti PMC5714520.  Those
# papers describe the TARGET; the algorithm itself is the open-literature
# Fessler PWLS substitute, since the vendor algorithms are proprietary.
# =============================================================================

export get_hir_params, HIRParams

"""
    HIRParams

Parameters for OS-PWLS HIR reconstruction at a given strength level.
Tuned via sensitivity analysis to land in the SAFIRE / ASIR-V / iDose-4 /
AIDR 3D noise-reduction band reported in Ghetti PMC5714520 + Geyer 2015.

# Fields
- `strength`: Strength level (1-5).  Higher → more regularization → more
  noise reduction.
- `lambda`: Regularization strength (Huber prior weight).
- `nepochs`: Number of OS epochs.  One epoch with `n_subsets` subsets ≈
  `n_subsets` full-data iterations of convergence (Erdoğan & Fessler 1999).
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

"""
    get_hir_params(strength::Int) -> HIRParams

Strength-keyed parameter lookup, 1 (preserve texture) … 5 (max smoothing).

# Strength → clinical use
| Strength | Noise red. | Use case                         |
|----------|------------|----------------------------------|
| 1        |  8–15 %    | preserve FBP texture (lung nodules) |
| 2        | 15–25 %    | light smoothing                  |
| 3        | 25–35 %    | standard clinical (recommended)  |
| 4        | 30–42 %    | strong smoothing                 |
| 5        | 35–50 %    | maximum noise reduction          |

# Vendor-equivalent target bands (clinical-validation sources, NOT
# algorithm sources — see file header)
- GE ASIR-V    strength 1-5 ≈ ASIR-V 20 % … 100 %
- Siemens SAFIRE strength 1-5 = SAFIRE S1-S5
- Philips iDose-4 strength 1-5 ≈ iDose-4 levels 1-5
- Canon AIDR 3D strength 2/3/4 ≈ Mild / Standard / Strong
"""
function get_hir_params(strength::Int)
    1 ≤ strength ≤ 5 || error("Strength must be 1-5, got $strength")

    # Tuning notes (v29 HIR-DISCOVER / HIR-FIX-WEIGHTS):
    # V_inv normalization (~0.03) and stat_weights (~0.14 mean) suppress the
    # effective regularization, so λ must be O(1-10).  Relaxation < 1.0 is
    # needed for stability at higher λ.  n_subsets fixed at 12; one OS epoch
    # ≈ 12 full iterations of convergence (Erdoğan & Fessler 1999).
    params = Dict(
        #             strength, lambda, nepochs, n_subsets, huber_δ, relaxation, target%
        1 => HIRParams(1, 1.0f0, 1, 12, 0.08f0, 0.5f0,  ( 8, 15)),
        2 => HIRParams(2, 2.0f0, 2, 12, 0.07f0, 0.4f0,  (15, 25)),
        3 => HIRParams(3, 4.0f0, 2, 12, 0.06f0, 0.35f0, (25, 35)),
        4 => HIRParams(4, 6.0f0, 3, 12, 0.05f0, 0.3f0,  (30, 42)),
        5 => HIRParams(5, 6.0f0, 4, 12, 0.05f0, 0.3f0,  (35, 50)),
    )

    return params[strength]
end
