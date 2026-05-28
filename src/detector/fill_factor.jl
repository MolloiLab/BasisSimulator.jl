"""
    Forward/FillFactor.jl

Detector pixel fill factor modeling for CT simulation.

# Overview

Fill factor (geometric efficiency) = active detector area / total pixel area.
A fill factor < 1.0 represents gaps between detector elements (electrode
grids, scintillator packaging, anti-scatter grid supports, pixel isolation).

# Physics

Detected photons scale linearly:
    I_out = I_in × ff
    p_out = p_in − log(ff)   (projection domain, since p = −log(I))

`apply_fill_factor!` operates in projection domain (sinogram space).

# CatSim parity
Matches CatSim's `Detector_ThirdgenCurved.py`:
    activeArea = colSize × colFillFraction × rowSize × rowFillFraction
The effective fill factor is `row_fill × col_fill`.
"""

import AcceleratedKernels as AK

"""
    FillFactorModel

Detector fill factor specification.

# Fields
- `row_fill`: Fill factor in row direction (0-1)
- `col_fill`: Fill factor in column direction (0-1)
- `uniform`: Whether fill factor is uniform across detector

The effective fill factor is `row_fill × col_fill`.
"""
struct FillFactorModel
    row_fill::Float64
    col_fill::Float64
    uniform::Bool
end

"""
    FillFactorModel(fill::Float64)

Create uniform fill factor model with equal row and column fill.
"""
function FillFactorModel(fill::Float64)
    @assert 0 < fill <= 1 "Fill factor must be in (0, 1]"
    return FillFactorModel(sqrt(fill), sqrt(fill), true)
end

"""
    fill_factor_standard()

Standard CT detector fill factor (90%) — third-generation default,
used as the driver fallback when scanner row/col fill fields are 0.
"""
fill_factor_standard() = FillFactorModel(0.9)

"""
    effective_fill_factor(model::FillFactorModel)

Effective (total) fill factor = `row_fill × col_fill`.
"""
effective_fill_factor(model::FillFactorModel) = model.row_fill * model.col_fill

"""
    apply_fill_factor!(sinogram, model::FillFactorModel) -> sinogram

Apply fill factor effect to projection-domain sinogram (in-place, GPU-native).

Mathematical formulation: `p_out = p_in − log(ff)` where `ff` is the effective
fill factor. Since `ff < 1`, `−log(ff) > 0`, so projection values increase
uniformly (more apparent attenuation due to reduced active area).

!!! note "auto-cancelled by `simulate!`"
    The EICT `simulate!` driver subtracts this same `−log(ff_eff)` offset
    after the air-scan / log step, matching the way real scanners absorb
    detector gain into their calibration reference.  Notebooks that call
    `simulate!` get back log-line-integrals with this effect already
    cancelled; only direct callers of `apply_fill_factor!` need to worry
    about the residual offset.
"""
function apply_fill_factor!(
    sinogram::AbstractArray{T,3},
    model::FillFactorModel
) where T
    ff = effective_fill_factor(model)

    if ff ≈ 1.0
        return sinogram
    end

    offset = T(-log(ff))

    AK.foreachindex(sinogram) do idx
        sinogram[idx] += offset
    end

    return sinogram
end

# =============================================================================
# Exports
# =============================================================================

export FillFactorModel
export fill_factor_standard
export effective_fill_factor
export apply_fill_factor!
