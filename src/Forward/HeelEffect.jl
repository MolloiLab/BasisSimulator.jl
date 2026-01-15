# =============================================================================
# Heel Effect (Anode Self-Attenuation)
# =============================================================================
#
# Models the heel effect: X-rays emitted toward the anode side travel through
# more target material, resulting in lower intensity on the anode side.
#
# The intensity variation across the field follows approximately:
#   I(θ) = I₀ × exp(-μ_target × t / sin(θ_target + θ))
#
# where:
#   θ_target = anode angle (typically 7-12°)
#   θ = angle from central ray
#   t = effective target thickness
#
# Reference: Bushberg et al., "The Essential Physics of Medical Imaging"
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
    default_heel_effect(; anode_angle_deg=7.0, target_material=:tungsten, effective_thickness_mm=0.05)

Create heel effect model with specified parameters.

# Default values for typical CT tube:
- anode_angle: 7° (common for CT)
- target: tungsten
- effective_thickness: 0.05 mm (empirical, produces ~10-20% variation)
"""
function default_heel_effect(;
    anode_angle_deg::Real = 7.0,
    target_material::Symbol = :tungsten,
    effective_thickness_mm::Real = 0.05
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

Apply heel effect to intensity data (in-place).

# Arguments
- `intensity`: Intensity array [n_cols, n_rows, n_angles] (modified in place)
- `heel`: HeelEffect model
- `geom`: CTGeometry

# Returns
- Modified intensity array with heel effect applied

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

    # Effective thickness
    t = T(heel.effective_thickness_mm / 10)  # Convert to cm

    # Fan angle range (assumes symmetric detector)
    fan_angle_max = T(atan(geom.fov[1] / 2 / geom.SAD))

    # Precompute cathode side (maximum effective angle) for normalization
    # This ensures intensity is always ≤ original
    θ_max = θ_anode + fan_angle_max  # Maximum angle = cathode side
    max_path = t / sin(θ_max)
    max_atten = T(exp(-μ_target * max_path))  # Minimum attenuation (maximum transmission)

    AK.foreachindex(intensity) do idx
        ci = CartesianIndices(intensity)[idx]
        col, row, angle = Tuple(ci)

        # Fan angle for this column (negative = anode side, positive = cathode side)
        # Convention: col=1 is anode side, col=n_cols is cathode side
        n_cols_T = T(n_cols)
        γ = (T(col) - n_cols_T/T(2) - T(0.5)) / (n_cols_T/T(2)) * fan_angle_max

        # Angle through target material
        # On anode side (negative γ), angle is steeper, more attenuation
        θ_effective = θ_anode + γ

        # Ensure angle is positive and reasonable
        θ_effective = max(θ_effective, T(0.01))

        # Path length through target
        path_length = t / sin(θ_effective)

        # Attenuation factor
        attenuation = exp(-μ_target * path_length)

        # Normalize to cathode side (maximum transmission)
        # This ensures multiplier is always ≤ 1.0
        intensity[idx] *= attenuation / max_atten
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
"""
function get_heel_effect_info(heel::HeelEffect)
    return (
        enabled = heel.enabled,
        anode_angle_deg = heel.anode_angle_deg,
        target_material = heel.target_material,
        effective_thickness_mm = heel.effective_thickness_mm,
        expected_variation = heel.enabled ?
            "~$(round(Int, (1 - exp(-get_target_attenuation(heel.target_material) * heel.effective_thickness_mm/10 / sin(heel.anode_angle_deg*π/180))) * 100))% intensity drop on anode side" :
            "disabled"
    )
end
