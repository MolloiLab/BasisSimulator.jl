"""
    Forward/FillFactor.jl

Detector pixel fill factor modeling for CT simulation.

# Overview

Fill factor (also called geometric efficiency) is the ratio of active detector area
to total pixel area. A fill factor < 1.0 means there are gaps between detector
elements due to:

- Electrode grid structures for charge collection
- Scintillator packaging/mounting
- Anti-scatter grid supports
- Pixel isolation/crosstalk barriers

# Physics Background

The fill factor directly affects photon detection efficiency:

    detected_photons = incident_photons × fill_factor

In CT signal processing, this manifests as:
- **Intensity domain**: I_out = I_in × ff
- **Projection domain**: p_out = p_in - log(ff)

The second form follows from the relationship p = -log(I), where:
    p_out = -log(I_out) = -log(I_in × ff) = -log(I_in) - log(ff) = p_in - log(ff)

Since ff < 1, -log(ff) > 0, so projections increase (appears as additional attenuation).

# CatSim Compatibility

This implementation matches CatSim's fill factor handling in `Detector_ThirdgenCurved.py`:

```python
cfg.det.activeArea = colSize * detectorColFillFraction * rowSize * detectorRowFillFraction
cfg.detFlux = Ivec * (activeArea * distanceFactor)
```

CatSim applies separate row and column fill fractions, yielding:
    effective_fill_factor = row_fill × col_fill

# Typical Values

| Detector Type              | Fill Factor | Notes                          |
|----------------------------|-------------|--------------------------------|
| High-quality flat panel    | 0.85-0.95   | Modern indirect conversion     |
| Standard CT scintillator   | 0.90        | Third-generation CT default    |
| Photon counting (CdTe)     | 0.70-0.85   | Smaller pixels, more dead area |
| Ideal (theoretical)        | 1.0         | No dead area                   |

# References

1. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent
   Advances." 3rd ed. SPIE Press; 2015. Chapter 3: System Design.

2. Flohr TG, et al. "Photon-counting CT review." Physica Medica. 2020;79:126-136.
   doi:10.1016/j.ejmp.2020.10.030

3. XCIST/CatSim: `pyfiles/Detector_ThirdgenCurved.py`, `pyfiles/Detection_Flux.py`
   https://github.com/xcist/main

# GPU Compatibility

✅ Metal (Apple Silicon)
✅ CUDA (NVIDIA)
✅ ROCm (AMD)
✅ CPU fallback

All operations use AcceleratedKernels.jl for backend-agnostic GPU execution.

# Example

```julia
using BasisSimulator

# Standard 90% fill factor
model = fill_factor_standard()
@assert effective_fill_factor(model) ≈ 0.9

# Apply to intensity data
intensity = ones(Float32, 128, 32, 360)
apply_fill_factor_intensity!(intensity, model)
# Now mean(intensity) ≈ 0.9

# Apply to projection data (sinogram)
sinogram = ones(Float32, 128, 32, 360)
apply_fill_factor!(sinogram, model)
# Now mean(sinogram) ≈ 1.0 - log(0.9) ≈ 1.105
```

# Verified Against

- CatSim `Detector_ThirdgenCurved.py` (geometry setup)
- CatSim `Detection_Flux.py` (flux calculation)
- PHYSICS-001 verification test suite
"""

import AcceleratedKernels as AK

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
# Fill Factor Application (GPU-native)
# =============================================================================

"""
    effective_fill_factor(model::FillFactorModel)

Get the effective (total) fill factor.
"""
effective_fill_factor(model::FillFactorModel) = model.row_fill * model.col_fill

"""
    apply_fill_factor!(sinogram, model::FillFactorModel) -> sinogram

Apply fill factor effect to sinogram data (in-place, GPU-native).

Models the reduced photon detection due to inactive detector area by adding
a uniform offset to projection values.

# Mathematical Formulation

In the projection domain:

    p_out = p_in - log(ff)

where `ff` is the effective fill factor (0 < ff ≤ 1).

This is equivalent to multiplying intensity by the fill factor:
    I_out = I_in × ff
    p = -log(I)  →  p_out = -log(I_in × ff) = p_in - log(ff)

Since ff < 1, -log(ff) > 0, so projection values increase uniformly.

# Arguments
- `sinogram::AbstractArray{T,3}`: Input sinogram [n_cols, n_rows, n_angles] in projection domain
- `model::FillFactorModel`: Fill factor specification

# Returns
Modified sinogram (same array, modified in-place).

# GPU Compatibility
✅ Metal, CUDA, ROCm, CPU via AcceleratedKernels.jl

# CatSim Reference
This matches CatSim's fill factor effect on detected flux.
See `Detection_Flux.py` line 30: `detFlux = Ivec * (activeArea * distanceFactor)`

# Example
```julia
sinogram = ones(Float32, 128, 32, 360)  # Unit projection
model = fill_factor_standard()          # ff = 0.9
apply_fill_factor!(sinogram, model)
# mean(sinogram) ≈ 1.0 - log(0.9) ≈ 1.105
```

# See Also
- [`apply_fill_factor_intensity!`](@ref): For intensity-domain data
- [`effective_fill_factor`](@ref): Get combined fill factor value
"""
function apply_fill_factor!(
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

    # GPU-native element-wise operation
    AK.foreachindex(sinogram) do idx
        sinogram[idx] += offset
    end

    return sinogram
end

"""
    apply_fill_factor_intensity!(intensity, model::FillFactorModel) -> intensity

Apply fill factor directly to intensity-domain data (in-place, GPU-native).

Models the reduced photon detection by directly scaling the intensity.

# Mathematical Formulation

In the intensity domain:

    I_out = I_in × ff

where `ff` is the effective fill factor (0 < ff ≤ 1).

For ff = 0.9, intensity is reduced by 10%.

# Arguments
- `intensity::AbstractArray{T,3}`: Input intensity [n_cols, n_rows, n_angles]
- `model::FillFactorModel`: Fill factor specification

# Returns
Modified intensity array (same array, modified in-place).

# GPU Compatibility
✅ Metal, CUDA, ROCm, CPU via AcceleratedKernels.jl

# CatSim Reference
Matches CatSim `Detection_Flux.py`: `detFlux = Ivec * activeArea * distanceFactor`
where `activeArea = colSize × colFillFraction × rowSize × rowFillFraction`

# Example
```julia
intensity = ones(Float32, 128, 32, 360)
model = fill_factor_standard()  # ff = 0.9
apply_fill_factor_intensity!(intensity, model)
# mean(intensity) ≈ 0.9
```

# See Also
- [`apply_fill_factor!`](@ref): For projection-domain (sinogram) data
- [`effective_fill_factor`](@ref): Get combined fill factor value
"""
function apply_fill_factor_intensity!(
    intensity::AbstractArray{T,3},
    model::FillFactorModel
) where T
    ff = T(effective_fill_factor(model))

    if ff ≈ 1.0
        return intensity
    end

    # GPU-native element-wise operation
    AK.foreachindex(intensity) do idx
        intensity[idx] *= ff
    end

    return intensity
end

# Convenience wrappers that allocate (for backward compatibility during transition)
function apply_fill_factor(sinogram::AbstractArray{T,3}, model::FillFactorModel) where T
    result = copy(sinogram)
    return apply_fill_factor!(result, model)
end

function apply_fill_factor_intensity(intensity::AbstractArray{T,3}, model::FillFactorModel) where T
    result = copy(intensity)
    return apply_fill_factor_intensity!(result, model)
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
export apply_fill_factor!, apply_fill_factor_intensity!
export apply_fill_factor, apply_fill_factor_intensity
export get_fill_factor_info
