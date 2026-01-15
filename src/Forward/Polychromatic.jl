# =============================================================================
# Polychromatic Forward Projection
# =============================================================================
#
# Memory-efficient polychromatic CT simulation using Beer-Lambert physics.
#
# NOT from TIGRE - TIGRE is monochromatic only.
# This implements proper spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
#
# Uses AcceleratedKernels.jl for GPU/CPU acceleration.
#
# References:
# - Beer-Lambert law for polychromatic X-rays
# - XCIST/CatSim approach (loop over energies)
# - PMC8126163 for energy bin resolution guidance
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

    # Pre-compute μ for all regions at this energy
    n_regions = length(materials)
    μ_at_energy = Vector{T}(undef, n_regions)
    for i in 1:n_regions
        μ_at_energy[i] = T(compute_μ_at_energy(materials[i], Float64(energy_keV)))
    end

    # Use AcceleratedKernels.jl for parallel execution
    AK.foreachindex(mask) do idx
        region_idx = mask[idx] + 1  # Convert 0-based region to 1-based array index
        μ_volume[idx] = μ_at_energy[region_idx]
    end

    return μ_volume
end

# =============================================================================
# Unified Forward Projection API
# =============================================================================

"""
    forward_project!(sinogram, volume_or_mask, geom; energy=nothing, energies=nothing, weights=nothing, materials=nothing, physics=nothing)

Unified forward projection with monochromatic/polychromatic options and physics effects.

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

# Returns
The modified sinogram array (with physics effects if specified)

# Examples

```julia
# Simple monochromatic projection (no physics)
forward_project!(sinogram, Float32.(phantom.μ), geom)

# Monochromatic with realistic physics
physics = realistic_physics_config(scatter_scale=1.0, noise_level=1.0)
forward_project!(sinogram, Float32.(phantom.μ), geom; physics=physics)

# Polychromatic with custom physics
materials = get_region_materials()
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)
physics = default_physics_config(
    scatter = default_scatter_model(),
    noise = default_detector_model(I0=1e5)
)
forward_project!(sinogram, phantom.mask, geom;
    energies=energies, weights=weights, materials=materials,
    physics=physics
)
```
"""
function forward_project!(
    sinogram::AbstractArray{T, 3},
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    energy::Union{Nothing, Real} = nothing,
    energies::Union{Nothing, Vector} = nothing,
    weights::Union{Nothing, Vector} = nothing,
    materials::Union{Nothing, Vector} = nothing,
    physics::Union{Nothing, PhysicsConfig} = nothing
) where T <: AbstractFloat

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

    # Apply physics effects if specified
    if physics !== nothing
        apply_physics_effects!(sinogram, geom, physics)
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
sinogram_gpu = forward_project(volume_gpu, geom; physics=realistic_physics_config())

# Polychromatic with physics
sinogram = forward_project(phantom.mask, geom;
    energies = energies,
    weights = weights,
    materials = materials,
    physics = realistic_physics_config(scatter_scale=1.0, noise_level=0.5)
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
