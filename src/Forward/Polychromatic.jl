"""
    Forward/Polychromatic.jl

Polychromatic X-ray simulation with energy-dependent attenuation.

Implements the Beer-Lambert law across the full X-ray spectrum:
    I/I₀ = Σ S(E) exp(-∫ μ(x,E) dx)

This naturally produces beam hardening artifacts and enables
multi-energy CT simulations.
"""

"""
    PolychromaticProjector

Pre-computed data for polychromatic forward projection.
"""
struct PolychromaticProjector
    # Projection geometry (Siddon indices and weights)
    proj_geom::ProjectionGeometry

    # Energy bins
    energies::Vector{Float64}      # keV
    spectrum_weights::Vector{Float64}  # Normalized spectrum

    # Material μ values at each energy: [n_regions × n_energies]
    μ_by_energy::Matrix{Float64}

    # Reference μ_water for each energy
    μ_water::Vector{Float64}
end

"""
    create_polychromatic_projector(phantom, geom, kVp=120; source=:xspect, n_bins=nothing)

Create a polychromatic projector for energy-dependent simulation.

# Arguments
- `phantom::Phantom`: Phantom with μ and mask arrays
- `geom::CTGeometry`: Scanner geometry
- `kVp::Int`: Tube voltage (determines spectrum)
- `source::Symbol`: Spectrum source (:xspect or :xcist)
- `n_bins::Union{Nothing,Int}`: Number of energy bins (nothing = use full spectrum)

# Returns
`PolychromaticProjector` ready for forward projection.
"""
function create_polychromatic_projector(
    phantom::Phantom,
    geom::CTGeometry,
    kVp::Int=120;
    source::Symbol=:xspect,
    n_bins::Union{Nothing,Int}=nothing
)
    # Load and optionally bin spectrum
    energies, weights = load_spectrum(kVp; source=source)

    if !isnothing(n_bins) && n_bins < length(energies)
        energies, weights = bin_spectrum(energies, weights, n_bins)
    end

    # Normalize spectrum weights
    weights = weights ./ sum(weights)

    # Pre-compute projection geometry
    proj_geom = precompute_projection_geometry(
        geom, phantom.fov, phantom.voxel_size, size(phantom.μ)
    )

    # Get all unique materials from phantom mask
    μ_by_energy = compute_region_μ_matrix(phantom, energies)

    # Reference water attenuation at each energy
    μ_water = [compute_μ_at_energy(XA.Materials.water, E) for E in energies]

    return PolychromaticProjector(proj_geom, energies, weights, μ_by_energy, μ_water)
end

"""
    compute_region_μ_matrix(phantom, energies) -> Matrix{Float64}

Compute μ values for each phantom region at each energy.

Returns matrix of size [n_regions × n_energies] where n_regions = 27
(matching RegionLabel enum values 0-26).
"""
function compute_region_μ_matrix(phantom::Phantom, energies::Vector{Float64})
    n_energies = length(energies)
    n_regions = 27  # RegionLabel enum range (0-26)

    μ_matrix = zeros(Float64, n_regions, n_energies)

    # Map region labels to materials
    region_materials = Dict{RegionLabel, Any}(
        REGION_BACKGROUND => XA.Materials.air,
        REGION_AIR => XA.Materials.air,
        REGION_WATER => XA.Materials.water,
        REGION_SOLID_WATER => solid_water,
        REGION_CA_50 => Ca_50,
        REGION_CA_100 => Ca_100,
        REGION_CA_200 => Ca_200,
        REGION_CA_300 => Ca_300,
        REGION_CA_400 => Ca_400,
        REGION_CA_500 => Ca_500,
        REGION_CA_600 => Ca_600,
        REGION_I_2_0 => I_2_0,
        REGION_I_2_5 => I_2_5,
        REGION_I_5_0 => I_5_0,
        REGION_I_7_5 => I_7_5,
        REGION_I_10_0 => I_10_0,
        REGION_I_15_0 => I_15_0,
        REGION_I_20_0 => I_20_0,
    )

    for (region, material) in region_materials
        region_idx = Int(region) + 1  # 0-indexed enum to 1-indexed array
        for (e_idx, energy) in enumerate(energies)
            μ_matrix[region_idx, e_idx] = compute_μ_at_energy(material, energy)
        end
    end

    return μ_matrix
end

"""
    forward_project_polychromatic(phantom, projector) -> Array{Float32,3}

Compute polychromatic projections using pre-computed projector.

Implements: -log(Σ S(E) exp(-∫ μ(E) dx))

# Arguments
- `phantom::Phantom`: Input phantom
- `projector::PolychromaticProjector`: Pre-computed projector data

# Returns
Sinogram in attenuation form (ready for reconstruction).
"""
function forward_project_polychromatic(
    phantom::Phantom,
    projector::PolychromaticProjector
)
    T = Float32
    n_cols, n_rows, n_angles, n_samples = size(projector.proj_geom.linear_indices)
    n_energies = length(projector.energies)

    # Flatten mask for linear indexing
    mask_flat = vec(phantom.mask)

    # OPTIMIZATION: Gather mask samples ONCE (not per energy)
    # This avoids creating a full μ volume for each energy bin
    # mask_samples: [n_cols, n_rows, n_angles, n_samples] of UInt8 region indices
    mask_samples = mask_flat[projector.proj_geom.linear_indices]

    # Pre-compute sample weights once
    weights_T = T.(projector.proj_geom.sample_weights)

    # Initialize transmission (accumulated I/I₀)
    transmission = zeros(T, n_cols, n_rows, n_angles)

    # For each energy bin
    for e_idx in 1:n_energies
        weight = T(projector.spectrum_weights[e_idx])

        # Get μ values for all 27 regions at this energy (small array!)
        μ_at_energy = T.(projector.μ_by_energy[:, e_idx])

        # OPTIMIZATION: Direct lookup from mask samples into small μ table
        # mask_samples contains region indices (0-26), add 1 for Julia indexing
        # μ_at_energy is only [27] elements - the lookup is O(n_samples), not O(n_voxels)
        μ_samples = μ_at_energy[mask_samples .+ 1]

        # Compute line integrals (path length weighted sum)
        line_integrals = dropdims(sum(μ_samples .* weights_T, dims=4), dims=4)

        # Accumulate: weight * exp(-line_integral)
        transmission .+= weight .* exp.(-line_integrals)
    end

    # Convert transmission to attenuation: -log(I/I₀)
    # Clamp to avoid log(0)
    transmission = clamp.(transmission, T(1e-10), T(Inf))
    sinogram = -log.(transmission)

    return sinogram
end

"""
    bin_spectrum(energies, weights, n_bins) -> (binned_energies, binned_weights)

Reduce spectrum to fewer energy bins by averaging.
"""
function bin_spectrum(energies::Vector{Float64}, weights::Vector{Float64}, n_bins::Int)
    n_orig = length(energies)
    bin_size = n_orig ÷ n_bins

    binned_energies = Float64[]
    binned_weights = Float64[]

    for i in 1:n_bins
        start_idx = (i - 1) * bin_size + 1
        end_idx = i == n_bins ? n_orig : i * bin_size

        # Weighted average energy in this bin
        bin_weights = weights[start_idx:end_idx]
        bin_energies = energies[start_idx:end_idx]

        total_weight = sum(bin_weights)
        if total_weight > 0
            avg_energy = sum(bin_energies .* bin_weights) / total_weight
            push!(binned_energies, avg_energy)
            push!(binned_weights, total_weight)
        end
    end

    return binned_energies, binned_weights
end

"""
    compute_effective_energy(projector) -> Float64

Compute the spectrum-weighted mean energy (effective energy).
"""
function compute_effective_energy(projector::PolychromaticProjector)
    return sum(projector.energies .* projector.spectrum_weights)
end

"""
    get_effective_μ_water(projector) -> Float64

Get spectrum-weighted water attenuation coefficient.
"""
function get_effective_μ_water(projector::PolychromaticProjector)
    return sum(projector.μ_water .* projector.spectrum_weights)
end

# =============================================================================
# Exports
# =============================================================================

export PolychromaticProjector, create_polychromatic_projector
export forward_project_polychromatic
export compute_effective_energy, get_effective_μ_water
