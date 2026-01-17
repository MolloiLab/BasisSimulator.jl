# =============================================================================
# Heel Effect (Anode Self-Attenuation)
# =============================================================================
#
# Models the heel effect: X-rays emitted toward the anode side travel through
# more target material, resulting in lower intensity on the anode side.
#
# The intensity variation across the field follows the CatSim-exact formula:
#
#   I(θ) = I₀ × exp(-μ × d × cos(θ_target) / sin(θ_target + θ))
#
# where:
#   θ_target = anode (target) angle (typically 7-12° for CT)
#   θ = take-off angle deviation from central ray
#   d = electron penetration depth in anode material (mm)
#   μ = linear attenuation coefficient of target material (cm⁻¹)
#
# The cos(θ_target) factor accounts for the effective electron beam penetration
# depth projected onto the anode surface.
#
# References:
# 1. Bushberg JT, et al. "The Essential Physics of Medical Imaging", 3rd ed.
#    Chapter 6: X-ray Production, X-ray Tubes, and X-ray Generators.
# 2. CatSim CreateHeelEffect.py - HeelEffectIntensity function
#    https://github.com/xcist/main
# 3. Podgorsak EB. "Radiation Physics for Medical Physicists", Chapter 4.
#
# =============================================================================

import AcceleratedKernels as AK

export HeelEffect
export default_heel_effect, heel_effect_none
export apply_heel_effect!, apply_heel_effect
export get_heel_effect_info

# =============================================================================
# Heel Effect Model
# =============================================================================

"""
    HeelEffect

Model for X-ray tube heel effect (anode self-attenuation).

# Fields
- `anode_angle_deg`: Anode angle in degrees (typically 7-12°)
- `target_material`: Target material (:tungsten, :molybdenum, :rhodium)
- `effective_thickness_mm`: Effective target thickness for attenuation
- `enabled`: Whether heel effect is active

# Notes
- Anode side has lower intensity (more self-attenuation)
- Effect is more pronounced at steeper anode angles
- Affects both intensity and effective spectrum (beam hardening on anode side)
"""
struct HeelEffect
    anode_angle_deg::Float64
    target_material::Symbol
    effective_thickness_mm::Float64
    enabled::Bool
end

# =============================================================================
# Default Models
# =============================================================================

"""
    default_heel_effect(; anode_angle_deg=7.0, target_material=:tungsten, effective_thickness_mm=0.01)

Create heel effect model with specified parameters.

# Default values for typical CT tube:
- anode_angle: 7° (common for CT)
- target: tungsten
- effective_thickness: 0.01 mm (conservative, produces ~5-10% intensity variation)

Note: Real CT tubes have heel effects producing 10-30% intensity variation across the field.
Use effective_thickness_mm=0.02-0.05 for stronger effects.
"""
function default_heel_effect(;
    anode_angle_deg::Real = 7.0,
    target_material::Symbol = :tungsten,
    effective_thickness_mm::Real = 0.01
)
    return HeelEffect(
        Float64(anode_angle_deg),
        target_material,
        Float64(effective_thickness_mm),
        true
    )
end

"""
    heel_effect_none()

Disabled heel effect (no intensity variation).
"""
function heel_effect_none()
    return HeelEffect(7.0, :tungsten, 0.0, false)
end

# =============================================================================
# Heel Effect Application
# =============================================================================

"""
    apply_heel_effect!(intensity, heel, geom)

Apply heel effect to intensity data (in-place) using CatSim-exact formula.

# Algorithm
The heel effect is modeled using the CatSim formula (from CreateHeelEffect.py):

    I(θ) = I₀ × exp(-μ × d × cos(θ_target) / sin(θ_target + θ))

where:
- μ = target material attenuation coefficient (cm⁻¹)
- d = electron penetration depth (cm)
- θ_target = anode (target) angle
- θ = take-off angle deviation from central ray

The cos(θ_target) factor accounts for the electron beam penetration geometry
in the tilted anode surface.

# Arguments
- `intensity`: Intensity array [n_cols, n_rows, n_angles] (modified in place)
- `heel`: HeelEffect model
- `geom`: CTGeometry

# Returns
- Modified intensity array with heel effect applied

# GPU Compatibility
- ✅ Metal (via AcceleratedKernels.jl)
- ✅ CUDA
- ✅ ROCm
- ✅ CPU fallback

# References
1. CatSim CreateHeelEffect.py - HeelEffectIntensity function
2. Bushberg JT, et al. "The Essential Physics of Medical Imaging"

# Example
```julia
heel = default_heel_effect(anode_angle_deg=7.0)
apply_heel_effect!(intensity, heel, geom)
```
"""
function apply_heel_effect!(
    intensity::AbstractArray{T, 3},
    heel::HeelEffect,
    geom::CTGeometry
) where T <: AbstractFloat

    if !heel.enabled || heel.effective_thickness_mm <= 0
        return intensity
    end

    n_cols, n_rows, n_angles = size(intensity)

    # Get target material attenuation (approximate at mean energy ~60 keV)
    # Convert all to T for GPU compatibility
    μ_target = T(get_target_attenuation(heel.target_material))

    # Anode angle in radians
    θ_anode = T(heel.anode_angle_deg * π / 180)

    # Electron penetration depth in cm (CatSim uses mm, we convert)
    d = T(heel.effective_thickness_mm / 10)

    # CatSim-exact: cos(θ_target) factor for electron penetration geometry
    cos_θ_anode = cos(θ_anode)

    # Fan angle range (assumes symmetric detector)
    fan_angle_max = T(atan(geom.fov[1] / 2 / geom.SAD))

    # Precompute reference angle attenuation for normalization
    # We normalize to the central ray (θ = 0) so that center intensity = 1.0
    sin_ref = sin(θ_anode)  # Reference at central ray
    sin_ref = max(sin_ref, T(0.01))  # Prevent division by zero
    exp_ref = -μ_target * d * cos_θ_anode / sin_ref
    I_ref = exp(clamp(exp_ref, T(-700), T(700)))

    # Minimum effective angle (prevent extreme attenuation at anode limit)
    # Clamp to at least θ_anode/3 to keep attenuation physically reasonable
    θ_min = θ_anode / T(3)

    AK.foreachindex(intensity) do idx
        ci = CartesianIndices(intensity)[idx]
        col, row, angle = Tuple(ci)

        # Fan angle for this column (negative = anode side, positive = cathode side)
        # Convention: col=1 is anode side, col=n_cols is cathode side
        n_cols_T = T(n_cols)
        γ = (T(col) - n_cols_T/T(2) - T(0.5)) / (n_cols_T/T(2)) * fan_angle_max

        # Effective angle through target material
        # On anode side (negative γ), angle is smaller, more attenuation
        θ_effective = θ_anode + γ

        # Clamp to minimum angle (prevents extreme attenuation at anode edge)
        θ_effective = max(θ_effective, θ_min)

        # CatSim-exact formula: exp(-μ × d × cos(θ_target) / sin(θ_target + θ))
        sin_effective = sin(θ_effective)
        sin_effective = max(sin_effective, T(0.01))  # Prevent division by zero

        exp_term = -μ_target * d * cos_θ_anode / sin_effective
        attenuation = exp(clamp(exp_term, T(-700), T(700)))

        # Normalize to central ray (so center stays at ~1.0)
        intensity[idx] *= attenuation / I_ref
    end

    return intensity
end

"""
    apply_heel_effect(intensity, heel, geom)

Non-mutating version of apply_heel_effect!.
"""
function apply_heel_effect(
    intensity::AbstractArray{T, 3},
    heel::HeelEffect,
    geom::CTGeometry
) where T <: AbstractFloat
    result = similar(intensity)
    copyto!(result, intensity)
    return apply_heel_effect!(result, heel, geom)
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    get_target_attenuation(material)

Get approximate attenuation coefficient for target material at ~60 keV.
"""
function get_target_attenuation(material::Symbol)
    # Approximate μ values at 60 keV (cm⁻¹)
    μ_values = Dict(
        :tungsten => 85.0,    # W, Z=74
        :molybdenum => 20.0,  # Mo, Z=42
        :rhodium => 25.0      # Rh, Z=45
    )
    return get(μ_values, material, 85.0)
end

"""
    get_heel_effect_info(heel)

Get information about heel effect model.

# Returns
Named tuple with:
- `enabled`: Whether heel effect is active
- `anode_angle_deg`: Target angle in degrees
- `target_material`: Target material symbol
- `effective_thickness_mm`: Electron penetration depth in mm
- `expected_variation`: Human-readable description of expected intensity variation
"""
function get_heel_effect_info(heel::HeelEffect)
    if !heel.enabled
        return (
            enabled = false,
            anode_angle_deg = heel.anode_angle_deg,
            target_material = heel.target_material,
            effective_thickness_mm = heel.effective_thickness_mm,
            expected_variation = "disabled"
        )
    end

    # Calculate expected intensity drop using CatSim formula
    # At anode limit (θ ≈ 0), most attenuation
    μ = get_target_attenuation(heel.target_material)
    d = heel.effective_thickness_mm / 10  # Convert to cm
    θ_anode = heel.anode_angle_deg * π / 180

    # CatSim formula: exp(-μ × d × cos(θ_anode) / sin(θ_anode + θ))
    # At central ray (θ = 0): exp(-μ × d × cos(θ_anode) / sin(θ_anode))
    # Intensity drop ≈ 1 - exp(...)
    central_atten = 1 - exp(-μ * d * cos(θ_anode) / sin(θ_anode))

    return (
        enabled = heel.enabled,
        anode_angle_deg = heel.anode_angle_deg,
        target_material = heel.target_material,
        effective_thickness_mm = heel.effective_thickness_mm,
        expected_variation = "~$(round(Int, central_atten * 100))% intensity drop at central ray"
    )
end
