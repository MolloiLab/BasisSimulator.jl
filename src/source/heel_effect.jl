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
- effective_thickness: 0.001 mm — the mean x-ray production depth; gives ≈ 2 %/degree of cone
  angle for a 7° tungsten anode, the order of the textbook heel effect (30–45 % across ±11°).

The anode angle must exceed the half-cone angle of the collimation (a 160 mm cone at 610 mm is
±7.5°, so a 7° anode cannot be used with it): rays that would leave below the target surface are
an invalid geometry, not a clamp.
"""
function default_heel_effect(;
        anode_angle_deg::Real = 7.0,
        target_material::Symbol = :tungsten,
        # The mean depth in the target at which the x-rays are produced, the one free parameter
        # of the model: 1 µm gives a 35 % anode-to-cathode fall across a ±7.5° cone (160 mm
        # collimation) and 3 % across ±0.7° (15 mm) for a 7° anode, the order the literature
        # reports for a 7–12° anode (30–45 % across ±11°). The former 0.01 mm gave 99 % and 28 %,
        # which the old fan mapping hid behind its angle clamp.
        effective_thickness_mm::Real = 0.001
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
    ) where {T <: AbstractFloat}

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

    # The takeoff angle changes with the CONE angle of the row (the anode axis is z in a CT),
    # not with the fan angle of the column; anode on the +z side, as in `compute_heel_spectral`.
    row_center_T = (T(n_rows) + one(T)) / T(2)
    row_pitch_det = T(geom.pixel_row_size * (geom.SDD / geom.SAD))
    SDD_T = T(geom.SDD)
    _heel_geometry_valid(Float64(θ_anode), atan((n_rows - Float64(row_center_T)) * Float64(row_pitch_det) / Float64(SDD_T)))

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
        idx_0 = Int32(idx - 1)
        col = (idx_0 % Int32(n_cols)) + Int32(1)
        idx_0 = idx_0 ÷ Int32(n_cols)
        row = (idx_0 % Int32(n_rows)) + Int32(1)
        angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

        # Cone angle of this row (positive towards +z, the anode side: smaller takeoff angle,
        # more self-absorption)
        α = atan((T(row) - row_center_T) * row_pitch_det / SDD_T)
        θ_effective = max(θ_anode - α, θ_min)

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
    ) where {T <: AbstractFloat}
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

"An anode-side ray that leaves below the target surface is not a tube geometry: the anode angle must exceed the half-cone."
function _heel_geometry_valid(θ_anode, half_cone)
    θ_anode > half_cone || throw(ArgumentError(
        "heel effect: the anode angle ($(round(θ_anode * 180 / π; digits = 2))°) must exceed the half-cone angle " *
        "($(round(half_cone * 180 / π; digits = 2))°) or anode-side rays would leave below the target surface — " *
        "use a larger anode angle or a narrower collimation"))
    return true
end

# =============================================================================
# Spectral Heel Effect (energy-dependent, per-row transmission)
# =============================================================================

"""
    compute_heel_spectral(heel, geom, energies_keV) -> Array{Float64, 3}

Compute the per-row, per-energy heel effect transmission: [n_cols, n_rows, n_energies] (flat along the fan).

Models anode self-attenuation with energy-dependent tungsten μ(E):
    T(row, E) = exp(-μ_W(E) × d × cos(θ_anode) / sin(θ_anode − α(row)))
normalized to central ray.

This is the spectral-domain heel effect, analogous to bowtie spectral transmission.
Applied during forward projection by multiplying into the spectral weight matrix.
"""
function compute_heel_spectral(
        heel::HeelEffect,
        geom::CTGeometry,
        energies_keV::Vector{Float64}
    )
    if !heel.enabled || heel.effective_thickness_mm <= 0
        return ones(Float64, geom.n_cols, geom.n_rows, length(energies_keV))
    end

    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_energies = length(energies_keV)

    θ_anode = heel.anode_angle_deg * π / 180.0
    d_cm = heel.effective_thickness_mm / 10.0
    cos_θ = cos(θ_anode)
    θ_min = θ_anode / 3.0

    # The heel effect is a property of the anode's takeoff angle. In a third-generation CT the
    # anode–cathode axis is parallel to z, so a ray's takeoff angle changes with its CONE angle —
    # the detector row — and not with the fan angle of its column: the gradient runs along the
    # rows and is flat along the fan. Convention here: the anode is on the +z side (the
    # `detector_v` direction), so rays towards +z rows leave the target at a smaller angle,
    # cross more of it and are attenuated more (the anode-side fall-off). Across the ±0.7° cone
    # of a 15 mm collimation this is a few percent; across a 160 mm cone (±7.5°) tens of percent.
    row_center = (n_rows + 1) / 2 + 0.0
    cone(row) = atan((row - row_center) * geom.pixel_row_size * (geom.SDD / geom.SAD) / geom.SDD)
    _heel_geometry_valid(θ_anode, cone(n_rows))

    # Get energy-dependent μ for target material
    target_mat = if heel.target_material == :tungsten
        XA.Elements.Tungsten
    elseif heel.target_material == :molybdenum
        XA.Elements.Molybdenum
    else
        XA.Elements.Tungsten
    end

    transmission = ones(Float64, n_cols, n_rows, n_energies)

    for (e_idx, E) in enumerate(energies_keV)
        # Energy-dependent linear attenuation of target material
        μ_E = compute_μ_at_energy(target_mat, E)

        # Reference attenuation at the central ray
        sin_ref = max(sin(θ_anode), 0.01)
        I_ref = exp(-μ_E * d_cm * cos_θ / sin_ref)

        for row in 1:n_rows
            θ_eff = max(θ_anode - cone(row), θ_min)
            sin_eff = max(sin(θ_eff), 0.01)
            I_row = exp(clamp(-μ_E * d_cm * cos_θ / sin_eff, -700.0, 700.0))
            # When I_ref is vanishingly small (e.g., < 1e-30 at very low energies), no photons
            # survive at any angle and the ratio is meaningless: no modulation.
            ratio = I_ref > 1.0e-30 ? I_row / I_ref : 1.0
            for col in 1:n_cols
                transmission[col, row, e_idx] = ratio
            end
        end
    end

    return transmission
end

export compute_heel_spectral
