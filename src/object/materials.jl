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

# Gammex Model 451 Solid Water — proper composition from XrayAttenuation.jl >= 0.2.3
const solid_water = XA.Materials.gammex_472_solidwater

# =============================================================================
# Material Registry
# =============================================================================

const MATERIALS_REGISTRY = Dict{Symbol, XA.Material}(
    # Gammex 472 inserts
    :Ca_50 => Ca_50, :Ca_100 => Ca_100, :Ca_200 => Ca_200, :Ca_300 => Ca_300,
    :Ca_400 => Ca_400, :Ca_500 => Ca_500, :Ca_600 => Ca_600,
    :I_2_0 => I_2_0, :I_2_5 => I_2_5, :I_5_0 => I_5_0, :I_7_5 => I_7_5,
    :I_10_0 => I_10_0, :I_15_0 => I_15_0, :I_20_0 => I_20_0,
    :solid_water => solid_water,
    # Common materials
    :water => XA.Materials.water,
    :air => XA.Materials.air,
    :bone => XA.Materials.corticalbone,
    :cortical_bone => XA.Materials.corticalbone,
    :corticalbone => XA.Materials.corticalbone,
    :blood => XA.Materials.blood,
    :muscle => XA.Materials.muscle,
    :soft_tissue => XA.Materials.softtissue,
    :softtissue => XA.Materials.softtissue,
    :lung => XA.Materials.lung,
    :adipose => XA.Materials.adipose,
    :brain => XA.Materials.brain,
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

# =============================================================================
# Region to Material Mapping for Polychromatic Simulation
# =============================================================================

"""
    get_region_materials() -> Vector{XA.Material}

Return a vector of materials indexed by region number (1-based).
Used for polychromatic simulation where μ_by_energy[region, energy] is needed.

The vector has 27 elements (indices 1-27, but only 18 are used):
- Index 1 (REGION 0): air (background)
- Index 2 (REGION 1): air
- Index 3 (REGION 2): water
- Index 4 (REGION 3): solid_water
- Indices 5-10: unused (filled with air)
- Index 11-17 (REGION 10-16): Ca_50 through Ca_600
- Indices 18-20: unused (filled with air)
- Index 21-27 (REGION 20-26): I_2_0 through I_20_0
"""
function get_region_materials()
    # Max region index is 26, so we need 27 elements (0-indexed regions become 1-indexed)
    materials = fill(XA.Materials.air, 27)

    # Map region indices to materials
    materials[1] = XA.Materials.air       # REGION_BACKGROUND = 0
    materials[2] = XA.Materials.air       # REGION_AIR = 1
    materials[3] = XA.Materials.water     # REGION_WATER = 2
    materials[4] = solid_water            # REGION_SOLID_WATER = 3

    # Calcium inserts (REGION 10-16 -> indices 11-17)
    materials[11] = Ca_50   # REGION_CA_50 = 10
    materials[12] = Ca_100  # REGION_CA_100 = 11
    materials[13] = Ca_200  # REGION_CA_200 = 12
    materials[14] = Ca_300  # REGION_CA_300 = 13
    materials[15] = Ca_400  # REGION_CA_400 = 14
    materials[16] = Ca_500  # REGION_CA_500 = 15
    materials[17] = Ca_600  # REGION_CA_600 = 16

    # Iodine inserts (REGION 20-26 -> indices 21-27)
    materials[21] = I_2_0   # REGION_I_2_0 = 20
    materials[22] = I_2_5   # REGION_I_2_5 = 21
    materials[23] = I_5_0   # REGION_I_5_0 = 22
    materials[24] = I_7_5   # REGION_I_7_5 = 23
    materials[25] = I_10_0  # REGION_I_10_0 = 24
    materials[26] = I_15_0  # REGION_I_15_0 = 25
    materials[27] = I_20_0  # REGION_I_20_0 = 26

    return materials
end


# =============================================================================
# Mixture builder
# =============================================================================
# Element tables for Z/A and Bethe mean-excitation, used so create_mixture can
# synthesize an XA.Material from an arbitrary composition Dict (Z → mass
# fraction) without depending on a precomputed `entries` vector.

const _MIX_ATOMIC_MASS = Dict{Int,Float64}(
    1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990,
    12=>24.305, 15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078,
    26=>55.845, 53=>126.904,
)
const _MIX_MEAN_EXCITATION_EV = Dict{Int,Float64}(
    1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
    15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0,
)

_zoa_ratio(comp::Dict{Int,Float64}) =
    sum(w * Z / get(_MIX_ATOMIC_MASS, Z, Float64(Z)*2) for (Z, w) in comp) /
    sum(values(comp))

function _mean_excitation(comp::Dict{Int,Float64})
    log_I = sum(w * (Z / get(_MIX_ATOMIC_MASS, Z, Float64(Z)*2)) *
                log(get(_MIX_MEAN_EXCITATION_EV, Z, 10.0*Z)) for (Z, w) in comp)
    z_a   = sum(w * (Z / get(_MIX_ATOMIC_MASS, Z, Float64(Z)*2)) for (Z, w) in comp)
    exp(log_I / z_a) * u"eV"
end

"""
    create_mixture(materials, fractions; by_volume=true, name="mixture") -> XA.Material

Mix `materials` by volume (default) or mass fractions; fractions must sum to 1.
"""
function create_mixture(
    materials::Vector{<:XA.Material},
    fractions::Vector{Float64};
    by_volume::Bool=true,
    name::String="mixture",
)::XA.Material
    length(materials) == length(fractions) ||
        throw(ArgumentError("create_mixture: length(materials)=$(length(materials)) != length(fractions)=$(length(fractions))"))
    isapprox(sum(fractions), 1.0; atol=1e-6) ||
        throw(ArgumentError("create_mixture: fractions must sum to 1.0 (got $(sum(fractions)))"))

    densities    = [ustrip(u"g/cm^3", m.density) for m in materials]
    compositions = [m.composition for m in materials]

    if by_volume
        ρ_mix      = sum(fractions .* densities)
        masses     = fractions .* densities
        mass_fracs = masses ./ sum(masses)
    else
        mass_fracs = fractions
        ρ_mix      = 1.0 / sum(fractions ./ densities)
    end

    elements = Dict{Int, Float64}()
    for (i, comp) in enumerate(compositions)
        for (Z, wf) in comp
            elements[Z] = get(elements, Z, 0.0) + mass_fracs[i] * wf
        end
    end
    s    = sum(values(elements))
    comp = Dict{Int,Float64}(Z => w/s for (Z, w) in elements)

    return XA.Material(name, _zoa_ratio(comp), _mean_excitation(comp),
                       ρ_mix * u"g/cm^3", comp)
end


# =============================================================================
# Exports
# =============================================================================

export Ca_50, Ca_100, Ca_200, Ca_300, Ca_400, Ca_500, Ca_600
export I_2_0, I_2_5, I_5_0, I_7_5, I_10_0, I_15_0, I_20_0
export solid_water
export get_material, MATERIALS_REGISTRY
export get_region_materials
export create_mixture
