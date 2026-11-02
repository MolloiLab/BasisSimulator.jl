"""
    Forward/FillFactor.jl

Detector pixel fill factor modeling for CT simulation.

# Overview

Fill factor (geometric efficiency) = active detector area / total pixel area.
A fill factor < 1.0 represents gaps between detector elements (electrode
grids, scintillator packaging, anti-scatter grid supports, pixel isolation).

# Physics

A dexel collects the fraction `ff` of the quanta landing on it, so the energy-integrating noise
draws from `I₀·ff` (`simulate!`); the air calibration divides the same factor out of the mean.

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
    effective_fill_factor(model::FillFactorModel)

Effective (total) fill factor = `row_fill × col_fill`.
"""
effective_fill_factor(model::FillFactorModel) = model.row_fill * model.col_fill

# =============================================================================
# Exports
# =============================================================================

export FillFactorModel
export effective_fill_factor
