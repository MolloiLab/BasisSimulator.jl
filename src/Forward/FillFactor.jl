"""
    Forward/FillFactor.jl

Detector pixel fill factor modeling for CT simulation.

Fill factor is the ratio of active detector area to total pixel area.
A fill factor < 1.0 means there are gaps between detector elements (e.g., due to
electrode grid, scintillator packaging, or anti-scatter grid supports).

Effects of fill factor:
- Reduces signal proportionally (fewer photons detected)
- Can cause aliasing at high spatial frequencies
- May introduce structured noise patterns if non-uniform

Typical values:
- High-quality flat panel: 0.85-0.95
- Standard CT detector: 0.90
- Photon counting detector: 0.70-0.85 (smaller pixels, more dead area)
"""

# =============================================================================
# Fill Factor Types
# =============================================================================

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

# =============================================================================
# Pre-defined Fill Factor Models
# =============================================================================

"""
    fill_factor_ideal()

Ideal detector with 100% fill factor (no dead area).
"""
fill_factor_ideal() = FillFactorModel(1.0, 1.0, true)

"""
    fill_factor_standard()

Standard CT detector fill factor (90%).

Typical for third-generation CT with scintillator array.
"""
fill_factor_standard() = FillFactorModel(0.9)

"""
    fill_factor_high()

High-quality detector fill factor (95%).

Modern flat panel or high-end CT detector.
"""
fill_factor_high() = FillFactorModel(0.95)

"""
    fill_factor_low()

Low fill factor detector (80%).

Older detectors or photon counting with larger dead zones.
"""
fill_factor_low() = FillFactorModel(0.8)

"""
    fill_factor_photon_counting()

Typical photon counting detector fill factor (75%).

Smaller pixels with more relative dead area.
"""
fill_factor_photon_counting() = FillFactorModel(0.75)

"""
    fill_factor_custom(row_fill, col_fill)

Custom fill factor with separate row and column values.

# Arguments
- `row_fill`: Fill factor in row (z) direction
- `col_fill`: Fill factor in column (in-plane) direction
"""
function fill_factor_custom(row_fill::Float64, col_fill::Float64)
    @assert 0 < row_fill <= 1 "row_fill must be in (0, 1]"
    @assert 0 < col_fill <= 1 "col_fill must be in (0, 1]"
    return FillFactorModel(row_fill, col_fill, true)
end

# =============================================================================
# Fill Factor Application
# =============================================================================

"""
    effective_fill_factor(model::FillFactorModel)

Get the effective (total) fill factor.
"""
effective_fill_factor(model::FillFactorModel) = model.row_fill * model.col_fill

"""
    apply_fill_factor(sinogram, model::FillFactorModel) -> Array

Apply fill factor effect to sinogram.

This reduces the detected signal proportional to the fill factor,
modeling the loss of photons hitting dead areas between detector elements.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles] (projection domain)
- `model::FillFactorModel`: Fill factor specification

# Returns
Sinogram with fill factor effects (increased projection values due to fewer detected photons).

# Physics
In projection domain: p_out = p_in - log(fill_factor)
This is equivalent to multiplying intensity by fill_factor:
  I_out = I_in × fill_factor
  p_out = -log(I_out) = -log(I_in × ff) = p_in - log(ff)
"""
function apply_fill_factor(
    sinogram::AbstractArray{T,3},
    model::FillFactorModel
) where T
    ff = effective_fill_factor(model)

    # No effect if fill factor is 1.0
    if ff ≈ 1.0
        return sinogram
    end

    # In projection domain: add -log(fill_factor) to each value
    offset = T(-log(ff))
    return sinogram .+ offset
end

"""
    apply_fill_factor_intensity(intensity, model::FillFactorModel) -> Array

Apply fill factor directly to intensity-domain data.

This multiplies the intensity by the fill factor.

# Arguments
- `intensity`: Input intensity [n_cols, n_rows, n_angles]
- `model::FillFactorModel`: Fill factor specification

# Returns
Intensity with fill factor effects (reduced signal).
"""
function apply_fill_factor_intensity(
    intensity::AbstractArray{T,3},
    model::FillFactorModel
) where T
    ff = T(effective_fill_factor(model))

    if ff ≈ 1.0
        return intensity
    end

    return intensity .* ff
end

"""
    get_fill_factor_info(model::FillFactorModel) -> NamedTuple

Get diagnostic information about fill factor model.
"""
function get_fill_factor_info(model::FillFactorModel)
    eff_ff = effective_fill_factor(model)
    signal_loss_percent = (1.0 - eff_ff) * 100

    return (
        row_fill = model.row_fill,
        col_fill = model.col_fill,
        effective_fill_factor = eff_ff,
        signal_loss_percent = signal_loss_percent,
        uniform = model.uniform
    )
end

# =============================================================================
# Exports
# =============================================================================

export FillFactorModel
export fill_factor_ideal, fill_factor_standard, fill_factor_high
export fill_factor_low, fill_factor_photon_counting, fill_factor_custom
export effective_fill_factor
export apply_fill_factor, apply_fill_factor_intensity
export get_fill_factor_info
