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
    forward_project!(sinogram, volume_or_mask, geom; energy=nothing, energies=nothing, weights=nothing, materials=nothing)

Unified forward projection with monochromatic or polychromatic options.

# Arguments
- `sinogram`: Output sinogram [n_cols, n_rows, n_angles] (modified in place)
- `volume_or_mask`: Either a 3D μ volume (Float32/64) or a UInt8 material mask
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments (choose one mode)

**Monochromatic mode** (direct volume input):
- Just pass a μ volume directly - no kwargs needed

**Monochromatic mode** (mask + single energy):
- `energy`: Single energy in keV (e.g., 60.0)
- `materials`: Vector of materials from `get_region_materials()`

**Polychromatic mode** (mask + spectrum):
- `energies`: Vector of energy bin centers (keV)
- `weights`: Vector of photon fluence weights
- `materials`: Vector of materials from `get_region_materials()`

# Returns
The modified sinogram array

# Examples

```julia
# Direct volume (monochromatic at whatever energy the μ values represent)
sinogram = zeros(Float32, 256, 32, 180)
forward_project!(sinogram, Float32.(phantom.μ), geom)

# Monochromatic with mask and specified energy
materials = get_region_materials()
forward_project!(sinogram, phantom.mask, geom; energy=60.0, materials=materials)

# Polychromatic with full spectrum
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)
forward_project!(sinogram, phantom.mask, geom; energies=energies, weights=weights, materials=materials)
```
"""
function forward_project!(
    sinogram::AbstractArray{T, 3},
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    energy::Union{Nothing, Real} = nothing,
    energies::Union{Nothing, Vector} = nothing,
    weights::Union{Nothing, Vector} = nothing,
    materials::Union{Nothing, Vector} = nothing
) where T <: AbstractFloat

    # Determine mode based on input type and kwargs
    if eltype(volume_or_mask) <: AbstractFloat
        # Direct volume input - simple monochromatic projection
        return siddon_forward_project!(sinogram, volume_or_mask, geom)

    elseif eltype(volume_or_mask) == UInt8
        # Mask input - need energy specification
        mask = volume_or_mask

        if materials === nothing
            error("materials must be provided when using a mask input")
        end

        if energy !== nothing
            # Monochromatic mode with single energy
            return _forward_project_mono!(sinogram, mask, geom, T(energy), materials)

        elseif energies !== nothing && weights !== nothing
            # Polychromatic mode
            return _forward_project_poly!(sinogram, mask, geom, energies, weights, materials)

        else
            error("Must specify either `energy` (single keV) or `energies` + `weights` (spectrum)")
        end
    else
        error("volume_or_mask must be Float32/Float64 (μ volume) or UInt8 (material mask)")
    end
end

"""
    forward_project(volume_or_mask, geom; kwargs...)

Allocating version of forward_project!. See `forward_project!` for details.
"""
function forward_project(
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    kwargs...
)
    sinogram = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
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
