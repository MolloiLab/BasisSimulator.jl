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

# =============================================================================
# NIST-Based Expected HU Calculations (Polychromatic)
# =============================================================================

"""
    compute_effective_μ_material(material, energies::Vector{Float64}, weights::Vector{Float64})

Compute effective (spectrum-weighted) linear attenuation for a single material.

μ_eff = Σ(w_E × μ(E)) / Σ(w_E)

This is the "thin sample" approximation - good for HU prediction.

# Arguments
- `material`: XA.Material object
- `energies`: Energy values in keV
- `weights`: Spectrum weights (photon counts or relative intensities)

# Returns
- `μ_eff::Float64`: Effective attenuation coefficient (cm⁻¹)
"""
function compute_effective_μ_material(material, energies::Vector{Float64}, weights::Vector{Float64})
    total_weight = sum(weights)
    μ_weighted_sum = 0.0

    for (E, w) in zip(energies, weights)
        μ = compute_μ_at_energy(material, E)
        μ_weighted_sum += w * μ
    end

    return μ_weighted_sum / total_weight
end

"""
    compute_expected_hu_spectrum(material, energies::Vector{Float64}, weights::Vector{Float64})

Compute expected HU for a material given a polychromatic spectrum using NIST data.

This computes:
1. μ_eff(material) = spectrum-weighted average of μ(E)
2. μ_eff(water) = spectrum-weighted average of μ_water(E)
3. HU = 1000 × (μ_eff(material) - μ_eff(water)) / μ_eff(water)

# Arguments
- `material`: XA.Material object (or Symbol to look up)
- `energies`: Energy values in keV
- `weights`: Spectrum weights

# Returns
- `HU_expected::Float64`: Expected HU value from NIST data

# Example
```julia
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)
hu = compute_expected_hu_spectrum(Ca_100, energies, weights)
# Returns expected HU for Ca-100 insert at 120 kVp
```
"""
function compute_expected_hu_spectrum(material, energies::Vector{Float64}, weights::Vector{Float64})
    # Handle Symbol input
    if material isa Symbol
        material = get_material(material)
    end

    μ_eff_material = compute_effective_μ_material(material, energies, weights)
    μ_eff_water = compute_effective_μ_material(XA.Materials.water, energies, weights)

    return 1000.0 * (μ_eff_material - μ_eff_water) / μ_eff_water
end

"""
    compute_expected_hu_spectrum(material_symbol::Symbol, kVp::Int; n_bins::Int=30)

Convenience method: compute expected HU for a material at given kVp.

# Arguments
- `material_symbol`: Symbol for material (e.g., :Ca_100, :solid_water)
- `kVp`: Tube voltage
- `n_bins`: Number of energy bins for spectrum (default 30)

# Returns
- `HU_expected::Float64`: Expected HU value
"""
function compute_expected_hu_spectrum(material_symbol::Symbol, kVp::Int; n_bins::Int=30)
    energies, weights = load_spectrum(kVp)
    energies, weights = downsample_spectrum(energies, weights, n_bins)
    material = get_material(material_symbol)
    return compute_expected_hu_spectrum(material, energies, weights)
end

"""
    NistExpectedHU

Container for NIST-derived expected HU values.

# Fields
- `region::UInt8`: Region ID (matches phantom mask values)
- `material_symbol::Symbol`: Material symbol
- `expected_hu::Float64`: Expected HU from NIST data
- `μ_eff::Float64`: Effective attenuation coefficient (cm⁻¹)
"""
struct NistExpectedHU
    region::UInt8
    material_symbol::Symbol
    expected_hu::Float64
    μ_eff::Float64
end

# Region constants for NIST table (mirrors Phantom.jl RegionLabel enum)
const _REGION_SOLID_WATER = UInt8(3)
const _REGION_CA_50 = UInt8(10)
const _REGION_CA_100 = UInt8(11)
const _REGION_CA_200 = UInt8(12)
const _REGION_CA_300 = UInt8(13)
const _REGION_CA_400 = UInt8(14)
const _REGION_CA_500 = UInt8(15)
const _REGION_CA_600 = UInt8(16)
const _REGION_I_2_0 = UInt8(20)
const _REGION_I_2_5 = UInt8(21)
const _REGION_I_5_0 = UInt8(22)
const _REGION_I_7_5 = UInt8(23)
const _REGION_I_10_0 = UInt8(24)
const _REGION_I_15_0 = UInt8(25)
const _REGION_I_20_0 = UInt8(26)

"""
    get_nist_expected_hu_table(kVp::Int; n_bins::Int=30) -> Vector{NistExpectedHU}

Compute expected HU values for all Gammex 472 materials from NIST data.

This is the "ground truth" for validating simulation accuracy.

# Arguments
- `kVp`: Tube voltage (e.g., 120)
- `n_bins`: Number of energy bins for spectrum

# Returns
- `Vector{NistExpectedHU}`: Expected HU for each phantom region

# Example
```julia
expected = get_nist_expected_hu_table(120)
for e in expected
    println("\$(e.material_symbol): \$(round(e.expected_hu, digits=1)) HU")
end
```
"""
function get_nist_expected_hu_table(kVp::Int; n_bins::Int=30)
    energies, weights = load_spectrum(kVp)
    energies, weights = downsample_spectrum(energies, weights, n_bins)

    μ_eff_water = compute_effective_μ_material(XA.Materials.water, energies, weights)

    results = NistExpectedHU[]

    # All testable regions with their materials (region ID, material symbol)
    test_regions = [
        (_REGION_SOLID_WATER, :solid_water),
        (_REGION_CA_50, :Ca_50),
        (_REGION_CA_100, :Ca_100),
        (_REGION_CA_200, :Ca_200),
        (_REGION_CA_300, :Ca_300),
        (_REGION_CA_400, :Ca_400),
        (_REGION_CA_500, :Ca_500),
        (_REGION_CA_600, :Ca_600),
        (_REGION_I_2_0, :I_2_0),
        (_REGION_I_2_5, :I_2_5),
        (_REGION_I_5_0, :I_5_0),
        (_REGION_I_7_5, :I_7_5),
        (_REGION_I_10_0, :I_10_0),
        (_REGION_I_15_0, :I_15_0),
        (_REGION_I_20_0, :I_20_0),
    ]

    for (region_id, mat_sym) in test_regions
        material = get_material(mat_sym)
        μ_eff = compute_effective_μ_material(material, energies, weights)
        expected_hu = 1000.0 * (μ_eff - μ_eff_water) / μ_eff_water
        push!(results, NistExpectedHU(region_id, mat_sym, expected_hu, μ_eff))
    end

    return results
end

"""
    print_nist_expected_hu_table(kVp::Int; n_bins::Int=30)

Print formatted table of expected HU values from NIST data.
"""
function print_nist_expected_hu_table(kVp::Int; n_bins::Int=30)
    expected = get_nist_expected_hu_table(kVp; n_bins=n_bins)

    println("=" ^ 60)
    println("NIST-Derived Expected HU Values ($(kVp) kVp, $(n_bins) bins)")
    println("=" ^ 60)
    println()
    println("Material        | Expected HU | μ_eff (cm⁻¹)")
    println("-" ^ 50)

    for e in expected
        name = rpad(string(e.material_symbol), 14)
        hu = lpad(round(Int, e.expected_hu), 8)
        μ = round(e.μ_eff, digits=4)
        println("  $(name) | $(hu)    | $(μ)")
    end
    println()
end

# =============================================================================
# Monochromatic (BHC-Referenced) Expected HU
# =============================================================================

"""
    MonochromaticExpectedHU

Expected HU values at a specific reference energy (for BHC-corrected scans).

# Fields
- `region::UInt8`: Region ID (matches phantom mask values)
- `material_symbol::Symbol`: Material symbol
- `expected_hu::Float64`: Expected HU at reference energy
- `μ::Float64`: Linear attenuation coefficient at reference energy (cm⁻¹)
- `reference_energy_keV::Float64`: Reference energy used
"""
struct MonochromaticExpectedHU
    region::UInt8
    material_symbol::Symbol
    expected_hu::Float64
    μ::Float64
    reference_energy_keV::Float64
end

"""
    get_monochromatic_expected_hu_table(reference_energy_keV::Float64=60.0) -> Vector{MonochromaticExpectedHU}

Compute expected HU values for all Gammex 472 materials at a specific reference energy.

This is the correct "ground truth" for BHC-corrected polychromatic scans,
where BHC maps polychromatic line integrals to monochromatic equivalents
at the reference energy.

# Arguments
- `reference_energy_keV`: Reference energy in keV (default 60.0, matching CatSim)

# Returns
- `Vector{MonochromaticExpectedHU}`: Expected HU for each phantom region

# Note
For BHC-corrected scans, use this instead of `get_nist_expected_hu_table()`.
The thin-sample approximation (spectrum-weighted) gives higher HU values
that don't match what water-based BHC actually produces.

# Example
```julia
# Get expected HU for BHC at 60 keV reference
expected = get_monochromatic_expected_hu_table(60.0)
for e in expected
    println("\$(e.material_symbol): \$(round(e.expected_hu, digits=1)) HU")
end
```
"""
function get_monochromatic_expected_hu_table(reference_energy_keV::Float64=60.0)
    μ_water_ref = compute_μ_at_energy(XA.Materials.water, reference_energy_keV)

    results = MonochromaticExpectedHU[]

    # All testable regions with their materials (region ID, material symbol)
    test_regions = [
        (_REGION_SOLID_WATER, :solid_water),
        (_REGION_CA_50, :Ca_50),
        (_REGION_CA_100, :Ca_100),
        (_REGION_CA_200, :Ca_200),
        (_REGION_CA_300, :Ca_300),
        (_REGION_CA_400, :Ca_400),
        (_REGION_CA_500, :Ca_500),
        (_REGION_CA_600, :Ca_600),
        (_REGION_I_2_0, :I_2_0),
        (_REGION_I_2_5, :I_2_5),
        (_REGION_I_5_0, :I_5_0),
        (_REGION_I_7_5, :I_7_5),
        (_REGION_I_10_0, :I_10_0),
        (_REGION_I_15_0, :I_15_0),
        (_REGION_I_20_0, :I_20_0),
    ]

    for (region_id, mat_sym) in test_regions
        material = get_material(mat_sym)
        μ = compute_μ_at_energy(material, reference_energy_keV)
        expected_hu = 1000.0 * (μ - μ_water_ref) / μ_water_ref
        push!(results, MonochromaticExpectedHU(region_id, mat_sym, expected_hu, μ, reference_energy_keV))
    end

    return results
end

"""
    compute_expected_hu_monochromatic(material_symbol::Symbol, reference_energy_keV::Float64=60.0)

Compute expected HU for a single material at a specific reference energy.

Use this for comparing against BHC-corrected polychromatic reconstruction.
"""
function compute_expected_hu_monochromatic(material_symbol::Symbol, reference_energy_keV::Float64=60.0)
    material = get_material(material_symbol)
    μ = compute_μ_at_energy(material, reference_energy_keV)
    μ_water = compute_μ_at_energy(XA.Materials.water, reference_energy_keV)
    return 1000.0 * (μ - μ_water) / μ_water
end

# Exports
export compute_μ_matrix, compute_μ_at_energy, compute_mass_μ_at_energy
export get_density, compute_effective_μ
export μ_to_HU, HU_to_μ, get_reference_μ_water
export get_effective_μ_water_kVp, EFFECTIVE_μ_WATER_KVP
export compute_effective_μ_material, compute_expected_hu_spectrum
export NistExpectedHU, get_nist_expected_hu_table, print_nist_expected_hu_table
export MonochromaticExpectedHU, get_monochromatic_expected_hu_table, compute_expected_hu_monochromatic
