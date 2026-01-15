"""
    Physics/Attenuation.jl

Pre-compute attenuation coefficient matrices for simulation.

Wraps XrayAttenuation.jl for NIST XCOM database queries.
"""

using Unitful
import XrayAttenuation as XA

"""
    compute_μ_matrix(materials::Vector, energies::Vector{Float64})

Pre-compute linear attenuation coefficients for all materials at all energies.

This is computed ONCE before simulation and injected into the forward model.

# Arguments
- `materials::Vector`: Vector of XA.Material objects
- `energies::Vector{Float64}`: Energy values in keV

# Returns
- `μ_matrix::Matrix{Float64}`: [n_materials × n_energies] linear attenuation (cm⁻¹)

# Example
```julia
materials = [XA.Materials.water, Ca_100, I_10_0]
energies, weights = load_spectrum(120)
μ_matrix = compute_μ_matrix(materials, energies)
# μ_matrix[2, 50] = linear attenuation of Ca_100 at energies[50]
```
"""
function compute_μ_matrix(materials::Vector, energies::Vector{Float64})
    n_materials = length(materials)
    n_energies = length(energies)

    μ_matrix = Matrix{Float64}(undef, n_materials, n_energies)

    for (i, mat) in enumerate(materials)
        for (j, E) in enumerate(energies)
            μ_unitful = XA.linear_attenuation_coeff(mat, E * u"keV")
            μ_matrix[i, j] = ustrip(u"cm^-1", μ_unitful)
        end
    end

    return μ_matrix
end

"""
    compute_μ_at_energy(material, energy_keV::Float64)

Get linear attenuation coefficient for a single material at single energy.

# Arguments
- `material`: XA.Material object
- `energy_keV::Float64`: Energy in keV

# Returns
- `μ::Float64`: Linear attenuation coefficient (cm⁻¹)
"""
function compute_μ_at_energy(material, energy_keV::Float64)
    μ_unitful = XA.linear_attenuation_coeff(material, energy_keV * u"keV")
    return ustrip(u"cm^-1", μ_unitful)
end

"""
    compute_mass_μ_at_energy(material, energy_keV::Float64)

Get mass attenuation coefficient for a single material at single energy.

# Arguments
- `material`: XA.Material object
- `energy_keV::Float64`: Energy in keV

# Returns
- `μ_ρ::Float64`: Mass attenuation coefficient (cm²/g)
"""
function compute_mass_μ_at_energy(material, energy_keV::Float64)
    μ_ρ_unitful = XA.mass_attenuation_coeff(material, energy_keV * u"keV")
    return ustrip(u"cm^2/g", μ_ρ_unitful)
end

"""
    get_density(material)

Get density of a material in g/cm³.
"""
function get_density(material)
    return ustrip(u"g/cm^3", material.density)
end

"""
    compute_effective_μ(μ_matrix::Matrix{Float64}, weights::Vector{Float64})

Compute effective (spectrum-weighted) linear attenuation for each material.

# Arguments
- `μ_matrix::Matrix{Float64}`: [n_materials × n_energies]
- `weights::Vector{Float64}`: Spectrum weights [n_energies]

# Returns
- `μ_eff::Vector{Float64}`: Effective attenuation for each material (cm⁻¹)

This gives the "average" attenuation across the polychromatic spectrum.
Useful for monochromatic approximations or initial estimates.
"""
function compute_effective_μ(μ_matrix::Matrix{Float64}, weights::Vector{Float64})
    n_materials = size(μ_matrix, 1)
    total_weight = sum(weights)

    μ_eff = Vector{Float64}(undef, n_materials)
    for i in 1:n_materials
        μ_eff[i] = sum(μ_matrix[i, :] .* weights) / total_weight
    end

    return μ_eff
end

# =============================================================================
# HU Conversion
# =============================================================================

"""
    μ_to_HU(μ::Real, μ_water::Real)

Convert linear attenuation coefficient to Hounsfield Units (HU).

HU = 1000 × (μ - μ_water) / μ_water

# Arguments
- `μ`: Linear attenuation coefficient (cm⁻¹)
- `μ_water`: Linear attenuation of water at the same energy (cm⁻¹)

# Returns
- `HU::Float64`: Hounsfield Units

# Reference values
- Air: HU ≈ -1000
- Water: HU = 0
- Bone: HU ≈ 400-1000
- Soft tissue: HU ≈ 40-80
"""
function μ_to_HU(μ::Real, μ_water::Real)
    return 1000.0 * (μ - μ_water) / μ_water
end

"""
    μ_to_HU(μ::AbstractArray, μ_water::Real)

Convert array of linear attenuation coefficients to HU.
"""
function μ_to_HU(μ::AbstractArray, μ_water::Real)
    return 1000.0 .* (μ .- μ_water) ./ μ_water
end

"""
    HU_to_μ(HU::Real, μ_water::Real)

Convert Hounsfield Units back to linear attenuation coefficient.

μ = μ_water × (1 + HU/1000)

# Arguments
- `HU`: Hounsfield Units
- `μ_water`: Linear attenuation of water (cm⁻¹)

# Returns
- `μ::Float64`: Linear attenuation coefficient (cm⁻¹)
"""
function HU_to_μ(HU::Real, μ_water::Real)
    return μ_water * (1.0 + HU / 1000.0)
end

# =============================================================================
# Effective Water μ for Different kVp (DukeSim Calibration Values)
# =============================================================================
#
# These are pre-computed effective μ_water values for water calibration
# at different tube voltages. Using these ensures HU ≈ 0 for water when
# scanning with a polychromatic spectrum.
#
# Source: DukeSim team pre-calculated values
# =============================================================================

"""
Pre-computed effective μ_water values for different kVp settings.

When using a polychromatic spectrum at a given kVp, use these values
for HU conversion to ensure water = 0 HU (water calibration).

Source: DukeSim team
"""
const EFFECTIVE_μ_WATER_KVP = Dict{Int, Float64}(
    80  => 0.2206,
    90  => 0.2122,
    100 => 0.2059,
    110 => 0.2026,
    120 => 0.1992
)

"""
    get_effective_μ_water_kVp(kVp::Int)

Get effective water linear attenuation coefficient for a given tube voltage.

These values are pre-computed for polychromatic spectra and ensure
water = 0 HU when used for HU calibration.

# Arguments
- `kVp::Int`: Tube voltage (80, 90, 100, 110, or 120)

# Returns
- `μ_eff::Float64`: Effective water attenuation coefficient (cm⁻¹)

# Source
DukeSim team pre-calculated values

# Example
```julia
# For 120 kVp scan
μ_water = get_effective_μ_water_kVp(120)  # Returns 0.1992
hu = μ_to_HU(μ_recon, μ_water)
```
"""
function get_effective_μ_water_kVp(kVp::Int)
    if haskey(EFFECTIVE_μ_WATER_KVP, kVp)
        return EFFECTIVE_μ_WATER_KVP[kVp]
    else
        # Interpolate or warn
        available = sort(collect(keys(EFFECTIVE_μ_WATER_KVP)))
        @warn "No pre-computed μ_water for $(kVp) kVp. Available: $(available). Using XrayAttenuation.jl lookup."
        # Estimate effective energy (rough approximation)
        effective_energy = 0.4 * kVp  # Rough rule of thumb
        return compute_μ_at_energy(XA.Materials.water, Float64(effective_energy))
    end
end

"""
    get_reference_μ_water(energy_keV::Float64=60.0)

Get linear attenuation coefficient of water at specified energy from XrayAttenuation.jl.

For polychromatic spectra, prefer `get_effective_μ_water_kVp(kVp)` instead.

Default energy is 60 keV (approximate effective energy for 100 kVp spectrum).
"""
function get_reference_μ_water(energy_keV::Float64=60.0)
    return compute_μ_at_energy(XA.Materials.water, energy_keV)
end

# Exports
export compute_μ_matrix, compute_μ_at_energy, compute_mass_μ_at_energy
export get_density, compute_effective_μ
export μ_to_HU, HU_to_μ, get_reference_μ_water
export get_effective_μ_water_kVp, EFFECTIVE_μ_WATER_KVP
