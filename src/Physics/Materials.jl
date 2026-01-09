"""
    Physics/Materials.jl

Custom material definitions for BasisSimulator.jl phantoms (Gammex 472).

# Overview

This module defines Gammex 472 calibration phantom materials using full elemental
compositions from the manufacturer specifications.

**Purpose:**
- Define Gammex 472 calcium and iodine inserts with accurate elemental compositions
- Provide tissue-equivalent materials for calibration and validation
- Enable accurate HU calibration and material decomposition validation

**Integration:**
- All materials are XA.Material types compatible with XrayAttenuation.jl
- Uses NIST XCOM database for energy-dependent attenuation coefficients
- Proper density and elemental composition from manufacturer specs

# Material Definitions

## Gammex 472 Calcium Inserts

Tissue-equivalent materials with varying calcium concentrations (50-600 mg/mL).

**Elemental Composition**: H, C, N, O, S, Cl, Ca (mass fractions from manufacturer)
**Densities**: 1.17-2.01 g/cm³ (calibrated to manufacturer specifications)

## Gammex 472 Iodine Inserts

Water-equivalent materials with varying iodine concentrations (2-20 mg/mL).

**Elemental Composition**: H, C, N, O, Na, S, Cl, Ca, I (mass fractions from manufacturer)
**Densities**: 1.03-1.04 g/cm³

# References

**Phantom Specifications:**
- Gammex Inc. (2018). "Model 472 Tissue Characterization Phantom"
- Material compositions from official Gammex 472 specification sheet

**NIST Data:**
- Berger, M.J., et al. (2010). "XCOM: Photon Cross Section Database."

# Author

Dale Black, MolloiLab
Created: January 2026
"""

import XrayAttenuation as XA
using Unitful

# ==============================================================================
# Gammex 472 Calcium Inserts
# ==============================================================================

"""
Gammex 472 calcium inserts with manufacturer-specified elemental compositions.

Reference: Gammex 472 specification sheet
"""
const Ca_50 = XA.Material(
    "Gammex 472 (50.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    1.17 * u"g/cm^3",
    Dict(
        1  => 0.0710, # H
        6  => 0.6266, # C
        7  => 0.0270, # N
        8  => 0.2308, # O
        16 => 0.0007, # S
        17 => 0.0012, # Cl
        20 => 0.0427  # Ca
    )
)

const Ca_100 = XA.Material(
    "Gammex 472 (100.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    1.24 * u"g/cm^3",
    Dict(
        1  => 0.0635, # H
        6  => 0.5720, # C
        7  => 0.0241, # N
        8  => 0.2579, # O
        16 => 0.0012, # S
        17 => 0.0010, # Cl
        20 => 0.0802  # Ca
    )
)

const Ca_200 = XA.Material(
    "Gammex 472 (200.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    1.40 * u"g/cm^3",
    Dict(
        1  => 0.0509, # H
        6  => 0.4806, # C
        7  => 0.0193, # N
        8  => 0.3031, # O
        16 => 0.0022, # S
        17 => 0.0008, # Cl
        20 => 0.1431  # Ca
    )
)

const Ca_300 = XA.Material(
    "Gammex 472 (300.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    1.55 * u"g/cm^3",
    Dict(
        1  => 0.0408, # H
        6  => 0.4070, # C
        7  => 0.0154, # N
        8  => 0.3396, # O
        16 => 0.0030, # S
        17 => 0.0007, # Cl
        20 => 0.1936  # Ca
    )
)

const Ca_400 = XA.Material(
    "Gammex 472 (400.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    1.70 * u"g/cm^3",
    Dict(
        1  => 0.0325, # H
        6  => 0.3465, # C
        7  => 0.0121, # N
        8  => 0.3695, # O
        16 => 0.0036, # S
        17 => 0.0005, # Cl
        20 => 0.2352  # Ca
    )
)

const Ca_500 = XA.Material(
    "Gammex 472 (500.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    1.85 * u"g/cm^3",
    Dict(
        1  => 0.0256, # H
        6  => 0.2958, # C
        7  => 0.0095, # N
        8  => 0.3946, # O
        16 => 0.0041, # S
        17 => 0.0004, # Cl
        20 => 0.2700  # Ca
    )
)

const Ca_600 = XA.Material(
    "Gammex 472 (600.0 mg/ml Calcium)",
    0.0,
    0.0 * u"eV",
    2.01 * u"g/cm^3",
    Dict(
        1  => 0.0196, # H
        6  => 0.2525, # C
        7  => 0.0072, # N
        8  => 0.4161, # O
        16 => 0.0046, # S
        17 => 0.0003, # Cl
        20 => 0.2998  # Ca
    )
)

# ==============================================================================
# Gammex 472 Iodine Inserts
# ==============================================================================

"""
Gammex 472 iodine inserts with manufacturer-specified elemental compositions.

Reference: Gammex 472 specification sheet
"""
const I_2_0 = XA.Material(
    "Gammex 472 (2.0 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.03 * u"g/cm^3",
    Dict(
        1  => 0.0864, # H
        6  => 0.6953, # C
        7  => 0.0215, # N
        8  => 0.1751, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0181, # Ca
        53 => 0.0020  # I
    )
)

const I_2_5 = XA.Material(
    "Gammex 472 (2.5 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.03 * u"g/cm^3",
    Dict(
        1  => 0.0863, # H
        6  => 0.6950, # C
        7  => 0.0214, # N
        8  => 0.1750, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0181, # Ca
        53 => 0.0025  # I
    )
)

const I_5_0 = XA.Material(
    "Gammex 472 (5.0 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.03 * u"g/cm^3",
    Dict(
        1  => 0.0861, # H
        6  => 0.6937, # C
        7  => 0.0214, # N
        8  => 0.1743, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0181, # Ca
        53 => 0.0049  # I
    )
)

const I_7_5 = XA.Material(
    "Gammex 472 (7.5 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.03 * u"g/cm^3",
    Dict(
        1  => 0.0859, # H
        6  => 0.6924, # C
        7  => 0.0213, # N
        8  => 0.1736, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0180, # Ca
        53 => 0.0073  # I
    )
)

const I_10_0 = XA.Material(
    "Gammex 472 (10.0 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.03 * u"g/cm^3",
    Dict(
        1  => 0.0856, # H
        6  => 0.6911, # C
        7  => 0.0212, # N
        8  => 0.1729, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0179, # Ca
        53 => 0.0097  # I
    )
)

const I_15_0 = XA.Material(
    "Gammex 472 (15.0 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.03 * u"g/cm^3",
    Dict(
        1  => 0.0851, # H
        6  => 0.6885, # C
        7  => 0.0210, # N
        8  => 0.1715, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0178, # Ca
        53 => 0.0146  # I
    )
)

const I_20_0 = XA.Material(
    "Gammex 472 (20.0 mg/ml Iodine)",
    0.0,
    0.0 * u"eV",
    1.04 * u"g/cm^3",
    Dict(
        1  => 0.0846, # H
        6  => 0.6859, # C
        7  => 0.0209, # N
        8  => 0.1701, # O
        11 => 0.0003, # Na
        16 => 0.0003, # S
        17 => 0.0013, # Cl
        20 => 0.0176, # Ca
        53 => 0.0194  # I
    )
)

# ==============================================================================
# Material Registry for Phantom Lookup
# ==============================================================================

"""
    CUSTOM_MATERIALS

Dictionary mapping material symbols to XA.Material objects.

Use this for phantom material lookup:
```julia
mat = Materials.CUSTOM_MATERIALS[:Ca_50]
μ = XA.linear_attenuation_coeff(mat, 60.0u"keV")
```
"""
const CUSTOM_MATERIALS = Dict{Symbol, XA.Material}(
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

Returns XA.Material type.

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

    # Get attenuation coefficients
    μ_mat_unitful = XA.linear_attenuation_coeff(mat, energy_kev * u"keV")
    μ_mat = ustrip(u"cm^-1", μ_mat_unitful)

    μ_water_unitful = XA.linear_attenuation_coeff(XA.Materials.water, energy_kev * u"keV")
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
        μ_unitful = XA.linear_attenuation_coeff(XA.Materials.water, E * u"keV")
        μ = ustrip(u"cm^-1", μ_unitful)
        println("  $(E) keV: μ = $(round(μ, digits=4)) cm⁻¹")
    end

    # Calcium inserts
    println("\nCalcium Inserts:")
    for symbol in [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
        mat = CUSTOM_MATERIALS[symbol]
        ρ = ustrip(u"g/cm^3", mat.density)
        println("  $symbol (ρ = $(round(ρ, digits=3)) g/cm³):")
        for E in energies
            μ_unitful = XA.linear_attenuation_coeff(mat, E * u"keV")
            μ = ustrip(u"cm^-1", μ_unitful)
            hu = validate_material_hu(symbol, E)
            println("    $(E) keV: μ = $(round(μ, digits=4)) cm⁻¹, HU = $(round(hu, digits=1))")
        end
    end

    # Iodine inserts
    println("\nIodine Inserts:")
    for symbol in [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]
        mat = CUSTOM_MATERIALS[symbol]
        ρ = ustrip(u"g/cm^3", mat.density)
        println("  $symbol (ρ = $(round(ρ, digits=3)) g/cm³):")
        for E in energies
            μ_unitful = XA.linear_attenuation_coeff(mat, E * u"keV")
            μ = ustrip(u"cm^-1", μ_unitful)
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
