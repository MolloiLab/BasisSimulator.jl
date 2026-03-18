"""
    Forward/PhysicsPipeline.jl

Physics configuration for CT simulation effects pipeline.

The PhysicsConfig struct holds all optional physics effect models.
Effects are applied by simulate!() via the workspace pathway.
"""

import AcceleratedKernels as AK

# =============================================================================
# Physics Configuration
# =============================================================================

"""
    PhysicsConfig

Configuration for physics effects pipeline.

All fields are optional - set to `nothing` to skip that effect.

# Fields (Standard Physics)
- `fill_factor`: FillFactorModel for detector active area
- `flat_filter`: FlatFilter for uniform beam filtration
- `bowtie_filter`: BowtieFilter for angle-dependent filtration
- `scatter`: ScatterModel for patient scatter (adds scatter)
- `scatter_correction`: ScatterCorrectionModel for scatter correction (removes scatter)
- `crosstalk`: CrosstalkModel for detector pixel coupling
- `optical_crosstalk`: OpticalCrosstalkModel for optical crosstalk
- `focal_spot`: FocalSpot for geometric blur
- `detector_efficiency`: DetectorEfficiency for scintillator response
- `noise`: DetectorModel for noise parameters
- `lag`: LagModel for temporal persistence
- `noise_seed`: Random seed for noise (for reproducibility)
- `energy_keV`: X-ray energy for filter calculations (default: 60.0)

# Fields (Signal Chain)
- `heel_effect`: HeelEffect for anode self-attenuation (intensity domain)
- `bhc`: BHCPolynomial or BeamHardeningCorrection for beam hardening correction
"""
struct PhysicsConfig
    fill_factor::Union{Nothing, FillFactorModel}
    flat_filter::Union{Nothing, FlatFilter}
    bowtie_filter::Union{Nothing, BowtieFilter}
    scatter::Union{Nothing, ScatterModel}
    scatter_correction::Union{Nothing, ScatterCorrectionModel}
    crosstalk::Union{Nothing, CrosstalkModel}
    optical_crosstalk::Union{Nothing, OpticalCrosstalkModel}
    focal_spot::Union{Nothing, FocalSpot}
    detector_efficiency::Union{Nothing, DetectorEfficiency}
    noise::Union{Nothing, DetectorModel}
    lag::Union{Nothing, LagModel}
    noise_seed::Union{Nothing, Int}
    energy_keV::Float64
    # Signal chain effects
    heel_effect::Union{Nothing, HeelEffect}
    bhc::Union{Nothing, BHCPolynomial, BeamHardeningCorrection}
end

"""
    default_physics_config(; kwargs...) -> PhysicsConfig

Create a physics configuration with default settings.

By default, all effects are disabled. Use keyword arguments to enable
specific effects with their models.
"""
function default_physics_config(;
    fill_factor::Union{Nothing, FillFactorModel}=nothing,
    flat_filter::Union{Nothing, FlatFilter}=nothing,
    bowtie_filter::Union{Nothing, BowtieFilter}=nothing,
    scatter::Union{Nothing, ScatterModel}=nothing,
    scatter_correction::Union{Nothing, ScatterCorrectionModel}=nothing,
    crosstalk::Union{Nothing, CrosstalkModel}=nothing,
    optical_crosstalk::Union{Nothing, OpticalCrosstalkModel}=nothing,
    focal_spot::Union{Nothing, FocalSpot}=nothing,
    detector_efficiency::Union{Nothing, DetectorEfficiency}=nothing,
    noise::Union{Nothing, DetectorModel}=nothing,
    lag::Union{Nothing, LagModel}=nothing,
    noise_seed::Union{Nothing, Int}=nothing,
    energy_keV::Float64=60.0,
    # Signal chain effects
    heel_effect::Union{Nothing, HeelEffect}=nothing,
    bhc::Union{Nothing, BHCPolynomial, BeamHardeningCorrection}=nothing
)
    return PhysicsConfig(
        fill_factor,
        flat_filter,
        bowtie_filter,
        scatter,
        scatter_correction,
        crosstalk,
        optical_crosstalk,
        focal_spot,
        detector_efficiency,
        noise,
        lag,
        noise_seed,
        energy_keV,
        heel_effect,
        bhc
    )
end

# =============================================================================
# Exports
# =============================================================================

export PhysicsConfig
export default_physics_config
