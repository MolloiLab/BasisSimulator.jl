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

CatSim-style signal processing (separate pipeline):
- Heel effect (intensity domain - before log transform)
- DAS model (raw signal domain - ADC effects)
- Calibration (air scan normalization)
- Beam hardening correction (sinogram domain)

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

# Fields (Standard Physics)
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

# Fields (CatSim-style Signal Processing)
- `heel_effect`: HeelEffect for anode self-attenuation (intensity domain)
- `das_model`: DASModel for signal chain effects (intensity domain)
- `bhc`: BHCPolynomial for beam hardening correction (sinogram domain)
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
    # CatSim-style effects
    heel_effect::Union{Nothing, HeelEffect}
    das_model::Union{Nothing, DASModel}
    bhc::Union{Nothing, BHCPolynomial}
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

# With CatSim-style effects
config = default_physics_config(
    heel_effect = default_heel_effect(),
    das_model = das_clinical(),
    bhc = bhc_water_default()
)
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
    energy_keV::Float64=60.0,
    # CatSim-style effects
    heel_effect::Union{Nothing, HeelEffect}=nothing,
    das_model::Union{Nothing, DASModel}=nothing,
    bhc::Union{Nothing, BHCPolynomial}=nothing
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
        energy_keV,
        heel_effect,
        das_model,
        bhc
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
- `heel_effect`: Enable heel effect (default: nothing)
- `das_model`: Enable DAS model (default: nothing)
- `bhc`: Enable beam hardening correction (default: nothing)

# Example
```julia
config = realistic_physics_config(scatter_scale=1.5, noise_level=0.5)

# With CatSim-style effects
config = realistic_physics_config(
    heel_effect = default_heel_effect(),
    bhc = bhc_water_default()
)
```
"""
function realistic_physics_config(;
    scatter_scale::Float64=1.0,
    noise_level::Float64=1.0,
    energy_keV::Float64=60.0,
    noise_seed::Union{Nothing, Int}=nothing,
    heel_effect::Union{Nothing, HeelEffect}=nothing,
    das_model::Union{Nothing, DASModel}=nothing,
    bhc::Union{Nothing, BHCPolynomial}=nothing
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
        energy_keV,
        heel_effect,
        das_model,
        bhc
    )
end

"""
    minimal_physics_config(; kwargs...) -> PhysicsConfig

Create a physics configuration with minimal realistic effects.

Only enables noise (which is always present in real CT).

# Keyword Arguments
- `noise_level`: Noise multiplier (default: 1.0)
- `noise_seed`: Random seed for reproducibility
- `bhc`: Enable beam hardening correction (default: nothing)

# Example
```julia
config = minimal_physics_config(noise_level=0.5)
```
"""
function minimal_physics_config(;
    noise_level::Float64=1.0,
    noise_seed::Union{Nothing, Int}=nothing,
    bhc::Union{Nothing, BHCPolynomial}=nothing
)
    I0 = 1e6 / noise_level

    return PhysicsConfig(
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing,
        default_detector_model(I0=I0, seed=noise_seed),
        nothing,
        noise_seed,
        60.0,
        nothing,  # heel_effect
        nothing,  # das_model
        bhc
    )
end

"""
    full_physics_config(; kwargs...) -> PhysicsConfig

Create a physics configuration with ALL 13 effects enabled by default.

This is the recommended configuration for realistic clinical CT simulation.
Includes ALL physics effects AND signal chain effects.

## ALL Effects Enabled (13 total):

**Physics Pipeline (10):**
1. fill_factor: 0.9 (detector dead area)
2. flat_filter: 3mm Al (beam hardening, dose reduction)
3. bowtie_filter: Large body (peripheral dose reduction)
4. detector_efficiency: GOS 0.5mm (scintillator absorption)
5. scatter: Convolution-based (Compton/Rayleigh)
6. crosstalk: Medium X-ray pixel coupling
7. optical_crosstalk: Typical scintillator light spread
8. focal_spot: Medium geometric blur
9. noise: Quantum + electronic noise
10. lag: GadOx afterglow

**Signal Chain (3):**
11. heel_effect: 7° tungsten anode
12. das_model: Gain + electronic noise
13. bhc: Water polynomial correction

## Scanner-Specific Notes:
The flat filter, bowtie filter, detector efficiency, fill factor, heel effect,
DAS model, and BHC are SCANNER-SPECIFIC and will vary by manufacturer/model.
The defaults here represent typical body CT parameters.

# Keyword Arguments
- `energy_keV`: X-ray mean energy for filter calculations (default: 60.0)
- `noise_seed`: Random seed for reproducibility (default: nothing)
- `scatter_scale`: Scatter multiplier (default: 1.0)
- `noise_level`: Noise multiplier, higher = more noise (default: 1.0)
- `das_noise_sigma`: DAS electronic noise sigma (default: 100.0)
- `anode_angle_deg`: Heel effect anode angle (default: 7.0)

# Example
```julia
# Full physics for clinical simulation - just pass physics config!
config = full_physics_config(energy_keV=65.0, noise_seed=42)

sinogram = forward_project(phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    physics=config
)
```
"""
function full_physics_config(;
    energy_keV::Float64=60.0,
    noise_seed::Union{Nothing, Int}=nothing,
    scatter_scale::Float64=1.0,
    noise_level::Float64=1.0,
    das_noise_sigma::Float64=100.0,
    anode_angle_deg::Float64=7.0
)
    # Adjust I0 (photon count) inversely with noise_level
    I0 = 1e6 / noise_level

    return PhysicsConfig(
        # Physics pipeline effects (10)
        fill_factor_standard(),                    # 0.9 fill factor
        flat_filter_al(3.0),                       # 3mm Al flat filter
        bowtie_filter_large_body(),                # Large body bowtie
        default_scatter_model(scale_factor=scatter_scale),  # Scatter
        crosstalk_medium(),                        # X-ray crosstalk
        optical_crosstalk_typical(),               # Optical crosstalk
        focal_spot_medium(),                       # Focal spot blur
        detector_efficiency_gos(0.5),              # GOS 0.5mm scintillator
        default_detector_model(I0=I0, seed=noise_seed),  # Quantum + electronic noise
        lag_gadox(),                               # GadOx afterglow
        noise_seed,
        energy_keV,
        # Signal chain effects (3) - ALL ENABLED
        default_heel_effect(anode_angle_deg=anode_angle_deg),  # Heel effect
        default_das_model(gain=1.0, electronic_noise_sigma=das_noise_sigma),  # DAS model
        bhc_water_default(reference_energy_keV=energy_keV)  # BHC
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

# Signal Processing Order
Standard effects (sinogram domain):
1. Fill factor, flat/bowtie filter, scatter, crosstalk, focal spot, DQE, noise, lag

CatSim-style effects:
- Heel effect and DAS model operate in intensity domain (converted internally)
- BHC operates in sinogram domain (applied at end)

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

# With CatSim-style effects
config = default_physics_config(
    heel_effect = default_heel_effect(),
    bhc = bhc_water_default()
)
```
"""
function apply_physics_effects!(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    config::PhysicsConfig
) where T

    # =========================================================================
    # INTENSITY-DOMAIN EFFECTS (heel effect, DAS model)
    # Convert sinogram to intensity, apply effects, convert back
    # =========================================================================
    has_intensity_effects = config.heel_effect !== nothing || config.das_model !== nothing

    if has_intensity_effects
        eps = T(1e-10)

        # Convert to intensity domain: I = exp(-sinogram)
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = exp(-sinogram[idx])
        end

        # Apply heel effect (intensity domain)
        if config.heel_effect !== nothing
            apply_heel_effect!(sinogram, config.heel_effect, geom)
        end

        # Apply DAS model (intensity domain - noise, gain, quantization)
        if config.das_model !== nothing
            apply_das_model!(sinogram, config.das_model; seed=config.noise_seed)
        end

        # Convert back to sinogram domain: sinogram = -log(intensity)
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = -log(max(sinogram[idx], eps))
        end
    end

    # =========================================================================
    # STANDARD SINOGRAM-DOMAIN EFFECTS
    # =========================================================================

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
        apply_detector_efficiency!(sinogram, config.detector_efficiency, geom; energy_keV=config.energy_keV)
    end

    # 8. Detector noise (quantum + electronic)
    # This should be applied late, after deterministic effects
    # Skip if DAS model was used (it has its own noise)
    if config.noise !== nothing && config.das_model === nothing
        apply_detector_model!(sinogram, config.noise)
    end

    # 9. Detector lag (temporal persistence)
    # This is applied last as it's a temporal effect
    if config.lag !== nothing
        apply_lag!(sinogram, config.lag)
    end

    # =========================================================================
    # BEAM HARDENING CORRECTION (sinogram domain, applied last)
    # =========================================================================
    if config.bhc !== nothing
        apply_bhc!(sinogram, config.bhc)
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
        ("lag", config.lag),
        # CatSim-style effects
        ("heel_effect", config.heel_effect),
        ("das_model", config.das_model),
        ("bhc", config.bhc)
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
export default_physics_config, realistic_physics_config, minimal_physics_config, full_physics_config
export apply_physics_effects!, apply_physics_effects
export get_physics_config_info
