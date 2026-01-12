"""
    Physics/Materials.jl

Gammex 472 phantom materials for BasisSimulator.jl.

All materials are sourced directly from XrayAttenuation.jl which contains
the official Gammex 472 CT phantom specifications.
"""

import XrayAttenuation as XA
using Unitful: ustrip, @u_str

# =============================================================================
# Gammex 472 Calcium Inserts (from XrayAttenuation.jl)
# =============================================================================

const Ca_50 = XA.Materials.gammex_472_ca50_0
const Ca_100 = XA.Materials.gammex_472_ca100_0
const Ca_200 = XA.Materials.gammex_472_ca200_0
const Ca_300 = XA.Materials.gammex_472_ca300_0
const Ca_400 = XA.Materials.gammex_472_ca400_0
const Ca_500 = XA.Materials.gammex_472_ca500_0
const Ca_600 = XA.Materials.gammex_472_ca600_0

# =============================================================================
# Gammex 472 Iodine Inserts (from XrayAttenuation.jl)
# =============================================================================

const I_2_0 = XA.Materials.gammex_472_i2_0
const I_2_5 = XA.Materials.gammex_472_i2_5
const I_5_0 = XA.Materials.gammex_472_i5_0
const I_7_5 = XA.Materials.gammex_472_i7_5
const I_10_0 = XA.Materials.gammex_472_i10_0
const I_15_0 = XA.Materials.gammex_472_i15_0
const I_20_0 = XA.Materials.gammex_472_i20_0

# =============================================================================
# Background Materials (from XrayAttenuation.jl)
# =============================================================================

const solid_water = XA.Materials.gammex_water

# =============================================================================
# Material Registry
# =============================================================================

const MATERIALS_REGISTRY = Dict{Symbol, XA.Material}(
    :Ca_50 => Ca_50, :Ca_100 => Ca_100, :Ca_200 => Ca_200, :Ca_300 => Ca_300,
    :Ca_400 => Ca_400, :Ca_500 => Ca_500, :Ca_600 => Ca_600,
    :I_2_0 => I_2_0, :I_2_5 => I_2_5, :I_5_0 => I_5_0, :I_7_5 => I_7_5,
    :I_10_0 => I_10_0, :I_15_0 => I_15_0, :I_20_0 => I_20_0,
    :solid_water => solid_water, :water => XA.Materials.water, :air => XA.Materials.air,
)

"""
    get_material(symbol::Symbol) -> XA.Material

Get material by symbol from registry or XA.Materials.
"""
function get_material(symbol::Symbol)
    haskey(MATERIALS_REGISTRY, symbol) && return MATERIALS_REGISTRY[symbol]
    hasproperty(XA.Materials, symbol) && return getproperty(XA.Materials, symbol)
    error("Material :$symbol not found")
end

"""
    validate_material_hu(material_symbol::Symbol, energy_keV::Float64) -> Float64

Calculate expected HU value for a material at given energy.
"""
function validate_material_hu(material_symbol::Symbol, energy_keV::Float64)
    mat = get_material(material_symbol)
    μ_mat = ustrip(u"cm^-1", XA.linear_attenuation_coeff(mat, energy_keV * u"keV"))
    μ_water = ustrip(u"cm^-1", XA.linear_attenuation_coeff(XA.Materials.water, energy_keV * u"keV"))
    return 1000.0 * (μ_mat - μ_water) / μ_water
end

# =============================================================================
# Exports
# =============================================================================

export Ca_50, Ca_100, Ca_200, Ca_300, Ca_400, Ca_500, Ca_600
export I_2_0, I_2_5, I_5_0, I_7_5, I_10_0, I_15_0, I_20_0
export solid_water
export get_material, MATERIALS_REGISTRY, validate_material_hu
