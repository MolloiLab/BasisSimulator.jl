# =============================================================================
# Calibration — Low-Signal Correction (GPU)
# =============================================================================
#
# Simplified Julia port of CatSim/XCIST's `LowSignalCorr.py` (BSD 3-Clause,
# GE Precision HealthCare).  CatSim does:
#   1. find pixels with prep ≤ 0
#   2. replace them with a 3-tap column-direction convolution of neighbors
#      `[0.5, 0, 0.5]`
#   3. apply negative-log via a C extension
#
# BasisSim's `low_signal_correction_gpu!` keeps only step 1 — but degenerate:
# clamp ≤ 0 → ε.  We skip the convolution-replace because:
#   - it's a CatSim implementation detail for handling air-scan-divided raw
#     counts that went sub-zero, not a fundamental physics need;
#   - our pipeline applies it to `e^(-line-integral)` immediately before the
#     negative-log, where ≤ 0 values only happen at over-saturation, not
#     spatially-correlated detector noise.
# Negative-log is performed separately downstream in `polychromatic.jl`.
#
# Upstream reference:
#   `gecatsim/pyfiles/LowSignalCorr.py` — BSD 3-Clause, GE Precision HealthCare.
# =============================================================================

import AcceleratedKernels as AK

"""
    low_signal_correction_gpu!(prep::AbstractArray{T,3}) where {T<:AbstractFloat}

Clamp non-positive entries to `T(1e-10)` in place. GPU-compatible via
AcceleratedKernels. Called by `polychromatic.jl` to guarantee a strictly-
positive `prep` array before the downstream `-log` transform.

Simplification of CatSim's `LowSignalCorr.py` — see the module header for
the full reasoning.
"""
function low_signal_correction_gpu!(
        prep::AbstractArray{T, 3},
    ) where {T <: AbstractFloat}

    eps = T(1.0e-10)

    AK.foreachindex(prep) do idx
        if prep[idx] <= zero(T)
            prep[idx] = eps
        end
    end

    return prep
end
