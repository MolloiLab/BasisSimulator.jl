# =============================================================================
# Calibration — Low Signal Correction (GPU)
# =============================================================================
#
# Only the GPU low-signal correction is used (called by Polychromatic.jl).
# The rest of the CatSim-exact calibration pipeline was dead code and removed.
#
# =============================================================================

import AcceleratedKernels as AK

"""
    low_signal_correction_gpu!(prep) - GPU version

GPU-compatible low signal correction using AcceleratedKernels.
Clamps non-positive values to eps for numerical stability.
Called by the polychromatic forward projection pipeline.
"""
function low_signal_correction_gpu!(
    prep::AbstractArray{T, 3}
) where T <: AbstractFloat

    eps = T(1e-10)

    AK.foreachindex(prep) do idx
        if prep[idx] <= zero(T)
            prep[idx] = eps
        end
    end

    return prep
end
