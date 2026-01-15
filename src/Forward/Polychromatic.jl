# =============================================================================
# Polychromatic Forward Projection with CatSim Signal Chain
# =============================================================================
#
# Memory-efficient polychromatic CT simulation using Beer-Lambert physics
# with optional CatSim-exact signal chain built-in.
#
# NOT from TIGRE - TIGRE is monochromatic only.
# This implements proper spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
#
# CatSim Signal Chain (when enabled via kwargs):
# 1. Polychromatic forward projection (Beer-Lambert)
# 2. Physics pipeline (scatter, crosstalk, focal spot, noise, lag)
# 3. Heel effect (intensity domain)
# 4. DAS model (gain + electronic noise)
# 5. Air scan calibration (CatSim-exact: noise-free air reference)
# 6. Low signal correction (replace negatives with smoothed neighbors)
# 7. Log transform
# 8. Beam hardening correction
#
# Uses AcceleratedKernels.jl for GPU/CPU acceleration.
#
# =============================================================================

import AcceleratedKernels as AK

export forward_project!, forward_project

# =============================================================================
# Create μ Volume from Material Mask
# =============================================================================

"""
    create_μ_volume!(μ_volume, mask, materials, energy_keV)

Create attenuation volume for a single energy from material mask.

# Arguments
- `μ_volume`: Output volume [nx, ny, nz] (modified in place)
- `mask`: 3D material mask (UInt8 region indices)
- `materials`: Vector of materials indexed by region
- `energy_keV`: Energy in keV
"""
function create_μ_volume!(
    μ_volume::AbstractArray{T, 3},
    mask::AbstractArray{UInt8, 3},
    materials::Vector,
    energy_keV::Real
) where T <: AbstractFloat

    # Pre-compute μ for all regions at this energy (on CPU)
    n_regions = length(materials)
    μ_at_energy_cpu = Vector{T}(undef, n_regions)
    for i in 1:n_regions
        μ_at_energy_cpu[i] = T(compute_μ_at_energy(materials[i], Float64(energy_keV)))
    end

    # Transfer lookup table to same device as mask (GPU or CPU)
    μ_at_energy = similar(mask, T, n_regions)
    copyto!(μ_at_energy, μ_at_energy_cpu)

    # Use AcceleratedKernels.jl for parallel execution
    AK.foreachindex(mask) do idx
        region_idx = mask[idx] + 1  # Convert 0-based region to 1-based array index
        μ_volume[idx] = μ_at_energy[region_idx]
    end

    return μ_volume
end

# =============================================================================
# Unified Forward Projection API with CatSim Signal Chain
# =============================================================================

"""
    forward_project!(sinogram, volume_or_mask, geom; kwargs...)

Unified forward projection with monochromatic/polychromatic options,
physics effects, and optional CatSim-exact signal chain.

# Arguments
- `sinogram`: Output sinogram [n_cols, n_rows, n_angles] (modified in place)
- `volume_or_mask`: Either a 3D μ volume (Float32/64) or a UInt8 material mask
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments

**Spectrum Mode** (choose one):

*Monochromatic* (direct volume input):
- Just pass a μ volume directly - no spectrum kwargs needed

*Monochromatic* (mask + single energy):
- `energy`: Single energy in keV (e.g., 60.0)
- `materials`: Vector of materials from `get_region_materials()`

*Polychromatic* (mask + spectrum):
- `energies`: Vector of energy bin centers (keV)
- `weights`: Vector of photon fluence weights
- `materials`: Vector of materials from `get_region_materials()`

**Physics Effects** (optional):
- `physics`: PhysicsConfig from `realistic_physics_config()`, `minimal_physics_config()`,
             or `default_physics_config(...)`. If `nothing`, no physics effects applied.

**CatSim Signal Chain** (optional - enables full clinical pipeline):
- `heel_effect`: HeelEffect model for anode self-attenuation
- `das_model`: DASModel for detector signal chain (gain + electronic noise)
- `bhc`: BHCPolynomial for beam hardening correction
- `calibrate`: Enable full CatSim calibration (default: true when signal chain enabled)
- `max_prep`: Maximum sinogram value for clamping (default: nothing)

When any signal chain parameter is provided, the function automatically:
1. Applies physics effects (if physics provided)
2. Converts to intensity domain
3. Applies heel effect (if provided)
4. Applies DAS model with noise (if provided)
5. Creates noise-free air scan reference (CatSim-exact)
6. Applies air scan calibration with low signal correction
7. Applies log transform
8. Applies BHC (if provided)

# Returns
The modified sinogram array

# Examples

```julia
# Simple monochromatic projection (no physics, no signal chain)
forward_project!(sinogram, Float32.(phantom.μ), geom)

# Polychromatic with physics pipeline only
physics = realistic_physics_config(scatter_scale=1.0, noise_level=1.0)
forward_project!(sinogram, phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    physics=physics
)

# FULL CatSim signal chain (recommended for clinical simulation)
forward_project!(sinogram, phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    physics=realistic_physics_config(scatter_scale=1.0, noise_level=0.01),
    heel_effect=default_heel_effect(anode_angle_deg=7.0),
    das_model=default_das_model(gain=1.0, electronic_noise_sigma=100.0),
    bhc=bhc_water_default()
)
```
"""
function forward_project!(
    sinogram::AbstractArray{T, 3},
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    # Spectrum parameters
    energy::Union{Nothing, Real} = nothing,
    energies::Union{Nothing, Vector} = nothing,
    weights::Union{Nothing, Vector} = nothing,
    materials::Union{Nothing, Vector} = nothing,
    # Physics pipeline
    physics::Union{Nothing, PhysicsConfig} = nothing,
    # CatSim signal chain (can override PhysicsConfig values)
    heel_effect::Union{Nothing, HeelEffect} = nothing,
    das_model::Union{Nothing, DASModel} = nothing,
    bhc::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}} = nothing,
    calibrate::Bool = true,
    max_prep::Union{Nothing, Real} = nothing,
    noise_seed::Union{Nothing, Int} = nothing
) where T <: AbstractFloat

    # Get signal chain effects from PhysicsConfig if not provided as kwargs
    # This allows full_physics_config() to include everything
    effective_heel = heel_effect
    effective_das = das_model
    effective_bhc = bhc
    effective_seed = noise_seed

    if physics !== nothing
        if effective_heel === nothing && physics.heel_effect !== nothing
            effective_heel = physics.heel_effect
        end
        if effective_das === nothing && physics.das_model !== nothing
            effective_das = physics.das_model
        end
        if effective_bhc === nothing && physics.bhc !== nothing
            effective_bhc = physics.bhc
        end
        if effective_seed === nothing && physics.noise_seed !== nothing
            effective_seed = physics.noise_seed
        end
    end

    # Check if CatSim signal chain is requested (from kwargs OR PhysicsConfig)
    has_signal_chain = effective_heel !== nothing || effective_das !== nothing || effective_bhc !== nothing

    if has_signal_chain && calibrate
        # === FULL CatSim SIGNAL CHAIN ===
        return _forward_project_with_signal_chain!(
            sinogram, volume_or_mask, geom;
            energy=energy, energies=energies, weights=weights, materials=materials,
            physics=physics, heel_effect=effective_heel, das_model=effective_das,
            bhc=effective_bhc, max_prep=max_prep, noise_seed=effective_seed
        )
    end

    # === STANDARD PROJECTION (no signal chain) ===

    # Determine mode based on input type and kwargs
    if eltype(volume_or_mask) <: AbstractFloat
        # Direct volume input - simple monochromatic projection
        siddon_forward_project!(sinogram, volume_or_mask, geom)

    elseif eltype(volume_or_mask) == UInt8
        # Mask input - need energy specification
        mask = volume_or_mask

        if materials === nothing
            error("materials must be provided when using a mask input")
        end

        if energy !== nothing
            # Monochromatic mode with single energy
            _forward_project_mono!(sinogram, mask, geom, T(energy), materials)

        elseif energies !== nothing && weights !== nothing
            # Polychromatic mode
            _forward_project_poly!(sinogram, mask, geom, energies, weights, materials)

        else
            error("Must specify either `energy` (single keV) or `energies` + `weights` (spectrum)")
        end
    else
        error("volume_or_mask must be Float32/Float64 (μ volume) or UInt8 (material mask)")
    end

    # Apply physics effects if specified (but not signal chain)
    if physics !== nothing
        apply_physics_effects!(sinogram, geom, physics)
    end

    # Apply BHC separately if signal chain not used but BHC provided
    if effective_bhc !== nothing && !has_signal_chain
        apply_bhc!(sinogram, effective_bhc)
    end

    return sinogram
end

"""
    forward_project(volume_or_mask, geom; kwargs...)

Allocating version of forward_project!. See `forward_project!` for details.

The output sinogram is allocated on the same device as the input (CPU or GPU).

# Examples

```julia
# Simple monochromatic (CPU)
sinogram = forward_project(Float32.(phantom.μ), geom)

# GPU input -> GPU output
using Metal
volume_gpu = MtlArray(Float32.(phantom.μ))
sinogram_gpu = forward_project(volume_gpu, geom)

# Full CatSim signal chain
sinogram = forward_project(phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    physics=realistic_physics_config(),
    heel_effect=default_heel_effect(),
    das_model=default_das_model(),
    bhc=bhc_water_default()
)
```
"""
function forward_project(
    volume_or_mask::AbstractArray{T},
    geom::CTGeometry;
    kwargs...
) where T
    # Determine element type for output
    out_type = T <: AbstractFloat ? T : Float32

    # Create sinogram on same device as input
    sinogram = similar(volume_or_mask, out_type, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, zero(out_type))

    return forward_project!(sinogram, volume_or_mask, geom; kwargs...)
end

# =============================================================================
# Internal: Full CatSim Signal Chain Implementation
# =============================================================================

"""
Internal function implementing full CatSim-exact signal chain.

Pipeline:
1. Polychromatic forward projection (Beer-Lambert) -> returns INTENSITY
2. Apply physics pipeline in sinogram domain (scatter, noise, etc.)
3. Convert to intensity domain
4. Apply heel effect
5. Apply DAS model (gain + noise to phantom only)
6. Create noise-free air scan (CatSim-exact)
7. Calibrate: prep = phantom / air
8. Low signal correction
9. Log transform
10. BHC
"""
function _forward_project_with_signal_chain!(
    sinogram::AbstractArray{T, 3},
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    energy::Union{Nothing, Real},
    energies::Union{Nothing, Vector},
    weights::Union{Nothing, Vector},
    materials::Union{Nothing, Vector},
    physics::Union{Nothing, PhysicsConfig},
    heel_effect::Union{Nothing, HeelEffect},
    das_model::Union{Nothing, DASModel},
    bhc::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}},
    max_prep::Union{Nothing, Real},
    noise_seed::Union{Nothing, Int}
) where T <: AbstractFloat

    # =========================================================================
    # LOG SIGNAL CHAIN CONFIGURATION
    # =========================================================================
    _log_signal_chain_config(physics, heel_effect, das_model, bhc, max_prep, noise_seed)

    # =========================================================================
    # STEP 1: Get raw sinogram (line integrals)
    # =========================================================================
    if eltype(volume_or_mask) <: AbstractFloat
        siddon_forward_project!(sinogram, volume_or_mask, geom)
    elseif eltype(volume_or_mask) == UInt8
        mask = volume_or_mask
        if materials === nothing
            error("materials must be provided when using a mask input")
        end
        if energy !== nothing
            _forward_project_mono!(sinogram, mask, geom, T(energy), materials)
        elseif energies !== nothing && weights !== nothing
            _forward_project_poly!(sinogram, mask, geom, energies, weights, materials)
        else
            error("Must specify either `energy` or `energies` + `weights`")
        end
    else
        error("volume_or_mask must be Float32/Float64 or UInt8")
    end

    # =========================================================================
    # STEP 2: Apply physics pipeline (in sinogram domain EXCEPT noise/DAS)
    # =========================================================================
    if physics !== nothing
        # Apply deterministic physics effects only (not noise - that's in DAS)
        # Create modified physics config without noise since DAS handles it
        _apply_physics_no_noise!(sinogram, geom, physics)
    end

    # =========================================================================
    # STEP 3: Convert to intensity domain
    # =========================================================================
    eps = T(1e-10)

    # Clamp sinogram to reasonable range before exp (avoid extreme intensities)
    AK.foreachindex(sinogram) do idx
        sinogram[idx] = exp(-clamp(sinogram[idx], T(-1), T(15)))
    end

    # Now sinogram contains INTENSITY values

    # =========================================================================
    # STEP 4: Apply heel effect to phantom intensity
    # =========================================================================
    if heel_effect !== nothing
        apply_heel_effect!(sinogram, heel_effect, geom)
    end

    # =========================================================================
    # STEP 5: Apply DAS model (gain + noise) to phantom
    # =========================================================================
    if das_model !== nothing
        apply_das_model!(sinogram, das_model; seed=noise_seed)
    end

    # =========================================================================
    # STEP 6: Create noise-free air scan (CatSim-exact)
    # =========================================================================
    # Air scan has same deterministic effects but NO noise
    air_scan = similar(sinogram)
    fill!(air_scan, one(T))

    if heel_effect !== nothing
        apply_heel_effect!(air_scan, heel_effect, geom)
    end

    if das_model !== nothing
        # Apply gain ONLY (no noise) - CatSim exact
        gain = T(das_model.gain)
        AK.foreachindex(air_scan) do idx
            air_scan[idx] *= gain
        end
    end

    # =========================================================================
    # STEP 7: Calibration (prep = phantom / air)
    # =========================================================================
    AK.foreachindex(sinogram) do idx
        air_val = max(air_scan[idx], eps)
        sinogram[idx] = sinogram[idx] / air_val
    end

    # =========================================================================
    # STEP 8: Low signal correction (CatSim-exact)
    # =========================================================================
    low_signal_correction_gpu!(sinogram)

    # =========================================================================
    # STEP 9: Log transform
    # =========================================================================
    if max_prep !== nothing
        max_val = T(max_prep)
        AK.foreachindex(sinogram) do idx
            val = -log(max(sinogram[idx], eps))
            sinogram[idx] = min(val, max_val)
        end
    else
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = -log(max(sinogram[idx], eps))
        end
    end

    # =========================================================================
    # STEP 10: Beam hardening correction
    # =========================================================================
    if bhc !== nothing
        apply_bhc!(sinogram, bhc)
    end

    return sinogram
end

"""
Apply physics effects except noise (which is handled by DAS model).
"""
function _apply_physics_no_noise!(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    config::PhysicsConfig
) where T

    # Apply deterministic physics effects only
    # Skip noise since DAS model handles it

    # Fill factor
    if config.fill_factor !== nothing
        apply_fill_factor!(sinogram, config.fill_factor)
    end

    # Flat filter
    if config.flat_filter !== nothing
        apply_flat_filter!(sinogram, config.flat_filter, geom; energy_keV=config.energy_keV)
    end

    # Bowtie filter
    if config.bowtie_filter !== nothing
        apply_bowtie_filter!(sinogram, config.bowtie_filter, geom; energy_keV=config.energy_keV)
    end

    # Scatter
    if config.scatter !== nothing
        add_scatter!(sinogram, config.scatter)
    end

    # Crosstalk
    if config.crosstalk !== nothing
        apply_crosstalk!(sinogram, config.crosstalk)
    end

    # Optical crosstalk
    if config.optical_crosstalk !== nothing
        apply_optical_crosstalk!(sinogram, config.optical_crosstalk)
    end

    # Focal spot blur
    if config.focal_spot !== nothing
        apply_focal_spot_blur!(sinogram, config.focal_spot, geom)
    end

    # Detector efficiency
    if config.detector_efficiency !== nothing
        apply_detector_efficiency!(sinogram, config.detector_efficiency, geom; energy_keV=config.energy_keV)
    end

    # NOTE: Skip noise - handled by DAS model in signal chain

    # Detector lag
    if config.lag !== nothing
        apply_lag!(sinogram, config.lag)
    end

    return sinogram
end

# =============================================================================
# Internal Implementation Functions
# =============================================================================

"""Monochromatic forward projection from mask + single energy"""
function _forward_project_mono!(
    sinogram::AbstractArray{T, 3},
    mask::AbstractArray{UInt8, 3},
    geom::CTGeometry,
    energy_keV::T,
    materials::Vector
) where T <: AbstractFloat

    # Create μ volume at this energy
    μ_volume = similar(sinogram, T, size(mask))
    create_μ_volume!(μ_volume, mask, materials, energy_keV)

    # Forward project
    return siddon_forward_project!(sinogram, μ_volume, geom)
end

"""Polychromatic forward projection from mask + spectrum"""
function _forward_project_poly!(
    sinogram::AbstractArray{T, 3},
    mask::AbstractArray{UInt8, 3},
    geom::CTGeometry,
    energies::Vector,
    weights::Vector,
    materials::Vector
) where T <: AbstractFloat

    n_energies = length(energies)

    # Normalize weights
    weights_norm = T.(weights ./ sum(weights))

    # Allocate temporary arrays
    μ_volume = similar(sinogram, T, size(mask))
    sino_mono = similar(sinogram)
    I_transmitted = similar(sinogram)

    # Initialize transmitted intensity
    fill!(I_transmitted, zero(T))

    # Loop over energies (memory-efficient approach)
    for e_idx in 1:n_energies
        # Create μ volume for this energy
        create_μ_volume!(μ_volume, mask, materials, energies[e_idx])

        # Forward project at this energy
        fill!(sino_mono, zero(T))
        siddon_forward_project!(sino_mono, μ_volume, geom)

        # Accumulate Beer-Lambert: I += w × exp(-line_integral)
        w = weights_norm[e_idx]

        AK.foreachindex(I_transmitted) do idx
            I_transmitted[idx] += w * exp(-sino_mono[idx])
        end
    end

    # Convert back to line integral: sinogram = -log(I / I₀)
    eps = T(1e-10)

    AK.foreachindex(sinogram) do idx
        sinogram[idx] = -log(max(I_transmitted[idx], eps))
    end

    return sinogram
end

# =============================================================================
# Signal Chain Logging
# =============================================================================

"""
Log CatSim signal chain configuration with scanner-specific notes.

SCANNER-SPECIFIC parameters (vary by manufacturer/model):
- Flat filter (material, thickness)
- Bowtie filter (profile shape)
- Detector efficiency (scintillator type, thickness)
- Fill factor (detector geometry)
- Heel effect (anode angle, target material)
- DAS model (gain, noise characteristics)
- BHC coefficients (calibration-dependent)

PHYSICS parameters (generally applicable):
- Scatter (depends on patient size, not scanner)
- Crosstalk (optional, can be disabled)
- Focal spot blur (optional, can be disabled)
- Detector lag (optional, can be disabled)
"""
function _log_signal_chain_config(
    physics::Union{Nothing, PhysicsConfig},
    heel_effect::Union{Nothing, HeelEffect},
    das_model::Union{Nothing, DASModel},
    bhc,
    max_prep::Union{Nothing, Real},
    noise_seed::Union{Nothing, Int}
)
    println("\n" * "=" ^ 60)
    println("CATSIM SIGNAL CHAIN ACTIVE")
    println("=" ^ 60)

    # --- Physics Pipeline ---
    if physics !== nothing
        info = get_physics_config_info(physics)
        println("\n[PHYSICS PIPELINE] $(info.n_enabled) effects enabled:")

        # CatSim ESSENTIAL (scanner-specific)
        scanner_specific = ["fill_factor", "flat_filter", "bowtie_filter", "detector_efficiency"]
        optional = ["scatter", "crosstalk", "optical_crosstalk", "focal_spot", "lag", "noise"]

        for effect in info.enabled_effects
            if effect in scanner_specific
                println("  ✓ $effect  [SCANNER-SPECIFIC]")
            elseif effect in optional
                println("  ✓ $effect  [OPTIONAL]")
            else
                println("  ✓ $effect")
            end
        end
    else
        println("\n[PHYSICS PIPELINE] DISABLED")
    end

    # --- Heel Effect ---
    println("\n[HEEL EFFECT]")
    if heel_effect !== nothing
        info = get_heel_effect_info(heel_effect)
        println("  ✓ ENABLED  [SCANNER-SPECIFIC: anode geometry]")
        println("    Anode angle: $(info.anode_angle_deg)°")
        println("    Target: $(info.target_material)")
    else
        println("  ✗ DISABLED")
    end

    # --- DAS Model ---
    println("\n[DAS MODEL]")
    if das_model !== nothing
        info = get_das_info(das_model)
        println("  ✓ ENABLED  [SCANNER-SPECIFIC: electronics]")
        println("    Gain: $(info.gain)")
        println("    Electronic noise σ: $(info.electronic_noise_sigma)")
        if noise_seed !== nothing
            println("    Noise seed: $noise_seed (reproducible)")
        end
    else
        println("  ✗ DISABLED")
    end

    # --- Air Scan Calibration ---
    println("\n[AIR SCAN CALIBRATION]")
    println("  ✓ ENABLED  [CATSIM-EXACT: air scan has NO noise]")
    println("    Low signal correction: replace negatives with smoothed neighbors")
    if max_prep !== nothing
        println("    Max prep clamp: $max_prep")
    end

    # --- BHC ---
    println("\n[BEAM HARDENING CORRECTION]")
    if bhc !== nothing
        if hasproperty(bhc, :coefficients)
            order = length(bhc.coefficients) - 1
            println("  ✓ ENABLED  [SCANNER-SPECIFIC: calibration-dependent]")
            println("    Polynomial order: $order")
        else
            println("  ✓ ENABLED")
        end
    else
        println("  ✗ DISABLED")
    end

    println("\n" * "=" ^ 60)
    println()
end
