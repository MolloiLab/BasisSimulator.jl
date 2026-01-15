"""
    Forward/PhysicsPipeline.jl

Unified physics effects pipeline for CT simulation.

This module provides a single entry point to apply all physics effects
to a sinogram in the correct order. All effects are GPU-native and
will automatically run on the same backend as the input array.

The recommended order for applying physics effects is:
1. Forward projection (produces ideal sinogram)
2. Fill factor (detector active area)
3. Flat filter (uniform beam filtration)
4. Bowtie filter (angle-dependent filtration)
5. Scatter (patient-dependent)
6. Crosstalk (detector pixel coupling)
7. Focal spot blur (geometric blur)
8. Detector efficiency (scintillator response)
9. Detector noise (quantum + electronic)
10. Detector lag (temporal persistence)

All effects are GPU-native using AcceleratedKernels.jl.
"""

import AcceleratedKernels as AK

# =============================================================================
# Physics Configuration
# =============================================================================

"""
    PhysicsConfig

Configuration for physics effects pipeline.

All fields are optional - set to `nothing` to skip that effect.

# Fields
- `fill_factor`: FillFactorModel for detector active area
- `flat_filter`: FlatFilter for uniform beam filtration
- `bowtie_filter`: BowtieFilter for angle-dependent filtration
- `scatter`: ScatterModel for patient scatter
- `crosstalk`: CrosstalkModel for detector pixel coupling
- `optical_crosstalk`: OpticalCrosstalkModel for optical crosstalk
- `focal_spot`: FocalSpot for geometric blur
- `detector_efficiency`: DetectorEfficiency for scintillator response
- `detector_model`: DetectorModel for noise parameters
- `lag`: LagModel for temporal persistence
- `noise_seed`: Random seed for noise (for reproducibility)
- `energy_keV`: X-ray energy for filter calculations (default: 60.0)
"""
struct PhysicsConfig
    fill_factor::Union{Nothing, FillFactorModel}
    flat_filter::Union{Nothing, FlatFilter}
    bowtie_filter::Union{Nothing, BowtieFilter}
    scatter::Union{Nothing, ScatterModel}
    crosstalk::Union{Nothing, CrosstalkModel}
    optical_crosstalk::Union{Nothing, OpticalCrosstalkModel}
    focal_spot::Union{Nothing, FocalSpot}
    detector_efficiency::Union{Nothing, DetectorEfficiency}
    noise::Union{Nothing, DetectorModel}
    lag::Union{Nothing, LagModel}
    noise_seed::Union{Nothing, Int}
    energy_keV::Float64
end

"""
    default_physics_config(; kwargs...) -> PhysicsConfig

Create a physics configuration with default settings.

By default, all effects are disabled. Use keyword arguments to enable
specific effects with their models.

# Example
```julia
# Enable scatter and noise
config = default_physics_config(
    scatter = default_scatter_model(),
    noise = default_detector_model()
)

# Apply to sinogram
apply_physics_effects!(sinogram, geom, config)
```
"""
function default_physics_config(;
    fill_factor::Union{Nothing, FillFactorModel}=nothing,
    flat_filter::Union{Nothing, FlatFilter}=nothing,
    bowtie_filter::Union{Nothing, BowtieFilter}=nothing,
    scatter::Union{Nothing, ScatterModel}=nothing,
    crosstalk::Union{Nothing, CrosstalkModel}=nothing,
    optical_crosstalk::Union{Nothing, OpticalCrosstalkModel}=nothing,
    focal_spot::Union{Nothing, FocalSpot}=nothing,
    detector_efficiency::Union{Nothing, DetectorEfficiency}=nothing,
    noise::Union{Nothing, DetectorModel}=nothing,
    lag::Union{Nothing, LagModel}=nothing,
    noise_seed::Union{Nothing, Int}=nothing,
    energy_keV::Float64=60.0
)
    return PhysicsConfig(
        fill_factor,
        flat_filter,
        bowtie_filter,
        scatter,
        crosstalk,
        optical_crosstalk,
        focal_spot,
        detector_efficiency,
        noise,
        lag,
        noise_seed,
        energy_keV
    )
end

"""
    realistic_physics_config(; kwargs...) -> PhysicsConfig

Create a physics configuration with typical realistic settings for body CT.

Enables scatter, crosstalk, focal spot blur, noise, and lag with
nominal parameters.

# Keyword Arguments
- `scatter_scale`: Scatter scale factor (default: 1.0)
- `noise_level`: Noise multiplier (default: 1.0, higher = more noise)
- `energy_keV`: X-ray energy (default: 60.0)

# Example
```julia
config = realistic_physics_config(scatter_scale=1.5, noise_level=0.5)
```
"""
function realistic_physics_config(;
    scatter_scale::Float64=1.0,
    noise_level::Float64=1.0,
    energy_keV::Float64=60.0,
    noise_seed::Union{Nothing, Int}=nothing
)
    # Adjust I0 (photon count) inversely with noise_level
    # Higher noise_level = lower I0 = more quantum noise
    I0 = 1e6 / noise_level

    return PhysicsConfig(
        nothing,  # fill_factor - typically not needed for most simulations
        nothing,  # flat_filter - depends on specific scanner
        nothing,  # bowtie_filter - depends on specific scanner
        default_scatter_model(scale_factor=scatter_scale),
        crosstalk_medium(),
        nothing,  # optical_crosstalk - use crosstalk instead
        focal_spot_medium(),
        nothing,  # detector_efficiency - often implicit in calibration
        default_detector_model(I0=I0, seed=noise_seed),
        lag_gadox(),
        noise_seed,
        energy_keV
    )
end

"""
    minimal_physics_config(; kwargs...) -> PhysicsConfig

Create a physics configuration with minimal realistic effects.

Only enables noise (which is always present in real CT).

# Keyword Arguments
- `noise_level`: Noise multiplier (default: 1.0)
- `noise_seed`: Random seed for reproducibility

# Example
```julia
config = minimal_physics_config(noise_level=0.5)
```
"""
function minimal_physics_config(;
    noise_level::Float64=1.0,
    noise_seed::Union{Nothing, Int}=nothing
)
    I0 = 1e6 / noise_level

    return PhysicsConfig(
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing,
        default_detector_model(I0=I0, seed=noise_seed),
        nothing,
        noise_seed,
        60.0
    )
end

# =============================================================================
# Physics Pipeline Application
# =============================================================================

"""
    apply_physics_effects!(sinogram, geom::CTGeometry, config::PhysicsConfig) -> sinogram

Apply all configured physics effects to sinogram (in-place, GPU-native).

Effects are applied in the recommended order for physical correctness.
All operations are GPU-native and will run on the same backend as the
input sinogram.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles]
- `geom::CTGeometry`: Scanner geometry
- `config::PhysicsConfig`: Physics configuration

# Returns
Modified sinogram with all configured physics effects applied.

# Example
```julia
# Create configuration
config = realistic_physics_config()

# Forward project
sinogram = siddon_forward_project(volume, geom)

# Apply all physics effects
apply_physics_effects!(sinogram, geom, config)

# Reconstruct
recon = fdk_reconstruct(sinogram, geom, volume_size)
```
"""
function apply_physics_effects!(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    config::PhysicsConfig
) where T
    # 1. Fill factor (detector active area)
    if config.fill_factor !== nothing
        apply_fill_factor!(sinogram, config.fill_factor)
    end

    # 2. Flat filter (uniform beam filtration)
    if config.flat_filter !== nothing
        apply_flat_filter!(sinogram, config.flat_filter, geom; energy_keV=config.energy_keV)
    end

    # 3. Bowtie filter (angle-dependent filtration)
    if config.bowtie_filter !== nothing
        apply_bowtie_filter!(sinogram, config.bowtie_filter, geom; energy_keV=config.energy_keV)
    end

    # 4. Scatter (patient-dependent, large kernel convolution)
    if config.scatter !== nothing
        add_scatter!(sinogram, config.scatter)
    end

    # 5. Crosstalk (detector pixel coupling)
    if config.crosstalk !== nothing
        apply_crosstalk!(sinogram, config.crosstalk)
    end

    # 5b. Optical crosstalk (alternative model)
    if config.optical_crosstalk !== nothing
        apply_optical_crosstalk!(sinogram, config.optical_crosstalk)
    end

    # 6. Focal spot blur (geometric blur)
    if config.focal_spot !== nothing
        apply_focal_spot_blur!(sinogram, config.focal_spot, geom)
    end

    # 7. Detector efficiency (scintillator response)
    if config.detector_efficiency !== nothing
        apply_detector_efficiency!(sinogram, config.detector_efficiency; energy_keV=config.energy_keV)
    end

    # 8. Detector noise (quantum + electronic)
    # This should be applied late, after deterministic effects
    if config.noise !== nothing
        apply_detector_model!(sinogram, config.noise)
    end

    # 9. Detector lag (temporal persistence)
    # This is applied last as it's a temporal effect
    if config.lag !== nothing
        apply_lag!(sinogram, config.lag)
    end

    return sinogram
end

# Convenience wrapper that allocates
function apply_physics_effects(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    config::PhysicsConfig
) where T
    result = copy(sinogram)
    return apply_physics_effects!(result, geom, config)
end

"""
    get_physics_config_info(config::PhysicsConfig) -> NamedTuple

Get summary information about enabled physics effects.
"""
function get_physics_config_info(config::PhysicsConfig)
    enabled = String[]
    disabled = String[]

    effects = [
        ("fill_factor", config.fill_factor),
        ("flat_filter", config.flat_filter),
        ("bowtie_filter", config.bowtie_filter),
        ("scatter", config.scatter),
        ("crosstalk", config.crosstalk),
        ("optical_crosstalk", config.optical_crosstalk),
        ("focal_spot", config.focal_spot),
        ("detector_efficiency", config.detector_efficiency),
        ("noise", config.noise),
        ("lag", config.lag)
    ]

    for (name, effect) in effects
        if effect !== nothing
            push!(enabled, name)
        else
            push!(disabled, name)
        end
    end

    return (
        enabled_effects = enabled,
        disabled_effects = disabled,
        n_enabled = length(enabled),
        n_total = length(effects),
        energy_keV = config.energy_keV,
        noise_seed = config.noise_seed
    )
end

# =============================================================================
# Exports
# =============================================================================

export PhysicsConfig
export default_physics_config, realistic_physics_config, minimal_physics_config
export apply_physics_effects!, apply_physics_effects
export get_physics_config_info
