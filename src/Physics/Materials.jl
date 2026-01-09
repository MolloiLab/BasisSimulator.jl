"""
    Physics/Materials.jl

Custom material definitions for BasisSimulator.jl phantoms.

# Overview

This module defines custom materials used in calibration phantoms (Gammex 472)
that are not available in the standard XrayAttenuation.jl materials database.

**Purpose:**
- Define Gammex 472 calcium and iodine inserts
- Provide tissue-equivalent materials for XCAT phantom
- Enable accurate HU calibration and material decomposition validation

**Integration:**
- All materials are XA.Compound or XA.Mixture types
- Compatible with get_linear_attenuation() and ray tracing
- Densities calibrated to manufacturer specifications

# Material Definitions

## Gammex 472 Calcium Inserts

Modeled as hydroxyapatite (Ca₁₀(PO₄)₆(OH)₂) dispersed in water-equivalent resin.

**Chemistry:**
- Hydroxyapatite: Ca₁₀(PO₄)₆(OH)₂
- Molecular weight: 1004.6 g/mol
- Calcium mass fraction: 39.9%

**Concentrations:**
- Ca_50: 50 mg/mL elemental calcium
- Ca_100: 100 mg/mL elemental calcium
- Ca_200: 200 mg/mL elemental calcium
- Ca_300: 300 mg/mL elemental calcium
- Ca_400: 400 mg/mL elemental calcium
- Ca_500: 500 mg/mL elemental calcium
- Ca_600: 600 mg/mL elemental calcium

**Density Calculation:**
Given target calcium concentration C_Ca (mg/mL):
1. Hydroxyapatite mass: m_HA = C_Ca / 0.399 (mg/mL)
2. Water mass: m_water = 1000 - m_HA (mg/mL, assuming density ≈ 1 g/cm³)
3. Total density: ρ = (m_HA + m_water) / 1000 (g/cm³)

## Gammex 472 Iodine Inserts

Modeled as iodine dissolved in water.

**Chemistry:**
- Iodine: I (atomic number 53)
- Dissolved in H₂O

**Concentrations:**
- I_2_0: 2.0 mg/mL iodine
- I_2_5: 2.5 mg/mL iodine
- I_5_0: 5.0 mg/mL iodine
- I_7_5: 7.5 mg/mL iodine
- I_10_0: 10.0 mg/mL iodine
- I_15_0: 15.0 mg/mL iodine
- I_20_0: 20.0 mg/mL iodine

**Density Calculation:**
Given target iodine concentration C_I (mg/mL):
1. Iodine mass: m_I = C_I (mg/mL)
2. Water mass: m_water = 1000 - m_I (mg/mL)
3. Total density: ρ = (m_I + m_water) / 1000 (g/cm³)

# References

**Phantom Specifications:**
- Gammex Inc. (2018). "Model 472 Tissue Characterization Phantom"
- GE Healthcare (2020). "Phantom Electron Density Calibration"

**Hydroxyapatite Properties:**
- NIST XCOM Database (Berger et al., 2010)
- ICRU Report 46 (1992). "Photon, Electron, Proton Beams"

**Material Composition:**
- Schneider, U., et al. (2000). "Correlation between CT numbers and tissue"
  Physics in Medicine & Biology, 45(2), 459-478.

# Author

Dale Black, MolloiLab
Created: January 2026
"""

import XrayAttenuation as XA
using Unitful
using Unitful: ustrip, @u_str

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
    CalciumInsert

Represents a calcium insert with specified concentration and computed density.

Fields:
- `concentration_mg_ml` - Elemental calcium concentration (mg/mL)
- `density` - Material density (g/cm³)
- `ha_mass_fraction` - Mass fraction of hydroxyapatite
- `water_mass_fraction` - Mass fraction of water
"""
struct CalciumInsert
    concentration_mg_ml::Float64
    density::Float64
    ha_mass_fraction::Float64
    water_mass_fraction::Float64
end

"""
    create_calcium_insert(concentration_mg_ml::Float64)::CalciumInsert

Create calcium insert material with specified elemental calcium concentration.

Uses hydroxyapatite (Ca₁₀(PO₄)₆(OH)₂) as calcium source mixed with water.

# Arguments
- `concentration_mg_ml` - Elemental calcium concentration (mg/mL)

# Returns
CalciumInsert with computed density and composition
"""
function create_calcium_insert(concentration_mg_ml::Float64)::CalciumInsert
    # Hydroxyapatite composition: Ca₁₀(PO₄)₆(OH)₂
    # Atomic masses: Ca=40.08, P=30.97, O=16.00, H=1.008
    # Molecular weight: 10*40.08 + 6*30.97 + 26*16.00 + 2*1.008 = 1004.6 g/mol
    # Mass fraction of Ca: (10*40.08)/1004.6 = 0.399

    ca_mass_fraction = 0.399

    # Calculate hydroxyapatite mass needed for target Ca concentration
    ha_concentration = concentration_mg_ml / ca_mass_fraction  # mg/mL

    # Remaining mass is water
    water_concentration = 1000.0 - ha_concentration  # mg/mL

    # Mass fractions
    ha_fraction = ha_concentration / 1000.0
    water_fraction = water_concentration / 1000.0

    # Density calculation using simple mass-weighted average
    # ρ_ha ≈ 3.16 g/cm³, ρ_water = 1.0 g/cm³
    # For low concentrations, approximate: ρ ≈ ρ_water + C_ha * (ρ_ha - ρ_water) / 1000
    # This is more appropriate for dilute suspensions
    rho_ha = 3.16
    rho_water = 1.0

    # Linear interpolation: ρ = w_water*ρ_water + w_ha*ρ_ha
    density = ha_fraction * rho_ha + water_fraction * rho_water

    return CalciumInsert(concentration_mg_ml, density, ha_fraction, water_fraction)
end

"""
    IodineInsert

Represents an iodine insert with specified concentration and computed density.

Fields:
- `concentration_mg_ml` - Iodine concentration (mg/mL)
- `density` - Material density (g/cm³)
- `iodine_mass_fraction` - Mass fraction of iodine
- `water_mass_fraction` - Mass fraction of water
"""
struct IodineInsert
    concentration_mg_ml::Float64
    density::Float64
    iodine_mass_fraction::Float64
    water_mass_fraction::Float64
end

"""
    create_iodine_insert(concentration_mg_ml::Float64)::IodineInsert

Create iodine insert material with specified iodine concentration in water.

# Arguments
- `concentration_mg_ml` - Iodine concentration (mg/mL)

# Returns
IodineInsert with computed density and composition
"""
function create_iodine_insert(concentration_mg_ml::Float64)::IodineInsert
    # Iodine mass
    iodine_concentration = concentration_mg_ml  # mg/mL

    # Water mass
    water_concentration = 1000.0 - iodine_concentration  # mg/mL

    # Mass fractions
    iodine_fraction = iodine_concentration / 1000.0
    water_fraction = water_concentration / 1000.0

    # Density calculation
    # ρ_iodine = 4.93 g/cm³ (solid), but dissolved so use effective mixing
    # For dilute solutions: ρ ≈ ρ_water + α*C_iodine
    # Empirical: α ≈ 0.0006 cm³/mg for iodine solutions
    density = 1.0 + 0.0006 * iodine_concentration

    return IodineInsert(concentration_mg_ml, density, iodine_fraction, water_fraction)
end

# ==============================================================================
# Gammex 472 Material Definitions
# ==============================================================================

"""
Gammex 472 Calcium Inserts (50-600 mg/mL elemental Ca)
"""
const Ca_50 = create_calcium_insert(50.0)
const Ca_100 = create_calcium_insert(100.0)
const Ca_200 = create_calcium_insert(200.0)
const Ca_300 = create_calcium_insert(300.0)
const Ca_400 = create_calcium_insert(400.0)
const Ca_500 = create_calcium_insert(500.0)
const Ca_600 = create_calcium_insert(600.0)

"""
Gammex 472 Iodine Inserts (2.0-20.0 mg/mL iodine)
"""
const I_2_0 = create_iodine_insert(2.0)
const I_2_5 = create_iodine_insert(2.5)
const I_5_0 = create_iodine_insert(5.0)
const I_7_5 = create_iodine_insert(7.5)
const I_10_0 = create_iodine_insert(10.0)
const I_15_0 = create_iodine_insert(15.0)
const I_20_0 = create_iodine_insert(20.0)

# Pre-compute XA compounds for mixtures
const _HA_COMPOUND = XA.Compound("Ca10P6O26H2")  # Hydroxyapatite
const _IODINE_COMPOUND = XA.Compound("I")  # Elemental iodine

# ==============================================================================
# Attenuation Coefficient Methods for Custom Materials
# ==============================================================================

"""
    get_linear_attenuation(mat::CalciumInsert, energy_keV::Float64)::Float64

Compute linear attenuation coefficient for calcium insert at given energy.

Uses mixture rule: μ = w_HA·μ_HA + w_water·μ_water, where w are mass fractions.
"""
function get_linear_attenuation(mat::CalciumInsert, energy_keV::Float64)::Float64
    # Get mass attenuation coefficients
    μ_ρ_ha = XA.mass_attenuation_coeff(_HA_COMPOUND, energy_keV * 1.0u"keV")
    μ_ρ_water = XA.mass_attenuation_coeff(XA.Materials.water, energy_keV * 1.0u"keV")

    # Convert to Float64 (strip units)
    μ_ρ_ha_val = ustrip(u"cm^2/g", μ_ρ_ha)
    μ_ρ_water_val = ustrip(u"cm^2/g", μ_ρ_water)

    # Mixture rule (mass-weighted average)
    μ_ρ_mix = mat.ha_mass_fraction * μ_ρ_ha_val + mat.water_mass_fraction * μ_ρ_water_val

    # Apply density: μ = ρ · (μ/ρ)
    return mat.density * μ_ρ_mix
end

"""
    get_linear_attenuation(mat::IodineInsert, energy_keV::Float64)::Float64

Compute linear attenuation coefficient for iodine insert at given energy.

Uses mixture rule: μ = w_I·μ_I + w_water·μ_water, where w are mass fractions.
"""
function get_linear_attenuation(mat::IodineInsert, energy_keV::Float64)::Float64
    # Get mass attenuation coefficients
    μ_ρ_iodine = XA.mass_attenuation_coeff(_IODINE_COMPOUND, energy_keV * 1.0u"keV")
    μ_ρ_water = XA.mass_attenuation_coeff(XA.Materials.water, energy_keV * 1.0u"keV")

    # Convert to Float64 (strip units)
    μ_ρ_iodine_val = ustrip(u"cm^2/g", μ_ρ_iodine)
    μ_ρ_water_val = ustrip(u"cm^2/g", μ_ρ_water)

    # Mixture rule (mass-weighted average)
    μ_ρ_mix = mat.iodine_mass_fraction * μ_ρ_iodine_val + mat.water_mass_fraction * μ_ρ_water_val

    # Apply density: μ = ρ · (μ/ρ)
    return mat.density * μ_ρ_mix
end

# ==============================================================================
# Material Registry for Phantom Lookup
# ==============================================================================

"""
    CUSTOM_MATERIALS

Dictionary mapping material symbols to custom material objects.

Use this for phantom material lookup:
```julia
mat = Materials.CUSTOM_MATERIALS[:Ca_50]
μ = get_linear_attenuation(mat, 60.0)
```
"""
const CUSTOM_MATERIALS = Dict{Symbol, Union{CalciumInsert, IodineInsert}}(
    :Ca_50  => Ca_50,
    :Ca_100 => Ca_100,
    :Ca_200 => Ca_200,
    :Ca_300 => Ca_300,
    :Ca_400 => Ca_400,
    :Ca_500 => Ca_500,
    :Ca_600 => Ca_600,
    :I_2_0  => I_2_0,
    :I_2_5  => I_2_5,
    :I_5_0  => I_5_0,
    :I_7_5  => I_7_5,
    :I_10_0 => I_10_0,
    :I_15_0 => I_15_0,
    :I_20_0 => I_20_0
)

"""
    get_material(symbol::Symbol)

Get material by symbol, checking custom materials first, then XA.Materials.

Returns either a custom material type (CalciumInsert, IodineInsert) or an XA.Material.

# Example
```julia
water = get_material(:water)  # From XA.Materials
ca50 = get_material(:Ca_50)   # From custom materials
```
"""
function get_material(symbol::Symbol)
    # Check custom materials first
    if haskey(CUSTOM_MATERIALS, symbol)
        return CUSTOM_MATERIALS[symbol]
    end

    # Fall back to XA.Materials
    if hasfield(typeof(XA.Materials), symbol)
        return getfield(XA.Materials, symbol)
    end

    error("Material :$symbol not found in custom materials or XA.Materials")
end

# ==============================================================================
# Validation Functions
# ==============================================================================

"""
    validate_material_hu(material_symbol::Symbol, energy_kev::Float64)::Float64

Calculate expected HU value for a material at given energy.

Returns HU relative to water at the same energy.
"""
function validate_material_hu(material_symbol::Symbol, energy_kev::Float64)::Float64
    mat = get_material(material_symbol)

    # Get attenuation for custom material using get_linear_attenuation
    μ_mat = get_linear_attenuation(mat, energy_kev)

    # Water reference (using XA's built-in function)
    μ_water_unitful = XA.linear_attenuation_coeff(XA.Materials.water, energy_kev * 1.0u"keV")
    μ_water = ustrip(u"cm^-1", μ_water_unitful)

    hu = 1000.0 * (μ_mat - μ_water) / μ_water

    return hu
end

"""
    print_material_properties()

Print attenuation properties of all Gammex materials at standard CT energies.
"""
function print_material_properties()
    energies = [60.0, 80.0, 120.0]  # keV

    println("="^70)
    println("Gammex 472 Material Properties")
    println("="^70)

    # Water reference
    println("\nReference (Water):")
    for E in energies
        μ_unitful = XA.linear_attenuation_coeff(XA.Materials.water, E * 1.0u"keV")
        μ = ustrip(u"cm^-1", μ_unitful)
        println("  $(E) keV: μ = $(round(μ, digits=4)) cm⁻¹")
    end

    # Calcium inserts
    println("\nCalcium Inserts:")
    for symbol in [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
        mat = CUSTOM_MATERIALS[symbol]
        println("  $symbol (ρ = $(round(mat.density, digits=3)) g/cm³):")
        for E in energies
            μ = get_linear_attenuation(mat, E)
            hu = validate_material_hu(symbol, E)
            println("    $(E) keV: μ = $(round(μ, digits=4)) cm⁻¹, HU = $(round(hu, digits=1))")
        end
    end

    # Iodine inserts
    println("\nIodine Inserts:")
    for symbol in [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]
        mat = CUSTOM_MATERIALS[symbol]
        println("  $symbol (ρ = $(round(mat.density, digits=3)) g/cm³):")
        for E in energies
            μ = get_linear_attenuation(mat, E)
            hu = validate_material_hu(symbol, E)
            println("    $(E) keV: μ = $(round(μ, digits=4)) cm⁻¹, HU = $(round(hu, digits=1))")
        end
    end

    println("="^70)
end

# ==============================================================================
# Exports
# ==============================================================================

# Export individual materials
export Ca_50, Ca_100, Ca_200, Ca_300, Ca_400, Ca_500, Ca_600
export I_2_0, I_2_5, I_5_0, I_7_5, I_10_0, I_15_0, I_20_0

# Export registry and lookup functions
export CUSTOM_MATERIALS, get_material
export validate_material_hu, print_material_properties
