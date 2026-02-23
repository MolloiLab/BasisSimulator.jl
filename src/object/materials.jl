"""
    Physics/Materials.jl

Gammex 472 materials and mixture calculations for BasisSimulator.jl.

All materials are sourced directly from XrayAttenuation.jl which contains
the official Gammex 472 CT phantom specifications.

# Mixture Calculation
Use `create_mixture()` to create custom materials from base components:
```julia
# Create 50% water + 50% bone mixture by volume
mixture = create_mixture(
    [water, bone],
    [0.5, 0.5];
    by_volume=true
)

# Create iodine contrast agent (mass fractions)
iodine_concentration = create_mixture(
    [water, iodine],
    [0.99, 0.01];  # 1% iodine by mass
    by_volume=false
)
```
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
# Brain Tissue Materials (Woodard & White 1986 / ICRU-44)
# =============================================================================

# Z/A ratio helper used for custom brain tissue construction
function _zoa_ratio(comp::Dict{Int,Float64})::Float64
    atomic_masses = Dict(1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990,
        12=>24.305, 15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078,
        26=>55.845, 53=>126.904)
    num = sum(w * Z / get(atomic_masses, Z, Float64(Z)*2) for (Z, w) in comp)
    den = sum(values(comp))
    num / den
end

function _mean_excitation(comp::Dict{Int,Float64})
    I_values = Dict(1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
        15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0)
    atomic_masses = Dict(1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990,
        12=>24.305, 15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078,
        26=>55.845, 53=>126.904)
    log_I = sum(w * (Z / get(atomic_masses, Z, Float64(Z)*2)) * log(get(I_values, Z, 10.0*Z))
                for (Z, w) in comp)
    z_a  = sum(w * (Z / get(atomic_masses, Z, Float64(Z)*2)) for (Z, w) in comp)
    exp(log_I / z_a) * u"eV"
end

# Gray matter: Woodard & White 1986 (mass fractions)
const _gm_comp = Dict{Int,Float64}(
    1 => 0.1070,   # H
    6 => 0.0955,   # C
    7 => 0.0180,   # N
    8 => 0.7620,   # O
    11 => 0.0020,  # Na
    15 => 0.0035,  # P
    16 => 0.0020,  # S
    17 => 0.0030,  # Cl
    19 => 0.0070,  # K
)
const gray_matter = XA.Material(
    "Gray Matter (Woodard & White 1986)",
    _zoa_ratio(_gm_comp),
    _mean_excitation(_gm_comp),
    1.04u"g/cm^3",
    _gm_comp,
)

# White matter: Woodard & White 1986 (mass fractions)
const _wm_comp = Dict{Int,Float64}(
    1 => 0.1060,   # H
    6 => 0.1980,   # C
    7 => 0.0130,   # N
    8 => 0.6703,   # O
    11 => 0.0020,  # Na
    15 => 0.0040,  # P
    16 => 0.0017,  # S
    17 => 0.0020,  # Cl
    19 => 0.0030,  # K
)
const white_matter = XA.Material(
    "White Matter (Woodard & White 1986)",
    _zoa_ratio(_wm_comp),
    _mean_excitation(_wm_comp),
    1.04u"g/cm^3",
    _wm_comp,
)

# =============================================================================
# Material Registry
# =============================================================================

const MATERIALS_REGISTRY = Dict{Symbol, XA.Material}(
    # Gammex 472 Calcium
    :Ca_50 => Ca_50, :Ca_100 => Ca_100, :Ca_200 => Ca_200, :Ca_300 => Ca_300,
    :Ca_400 => Ca_400, :Ca_500 => Ca_500, :Ca_600 => Ca_600,
    # Gammex 472 Iodine
    :I_2_0 => I_2_0, :I_2_5 => I_2_5, :I_5_0 => I_5_0, :I_7_5 => I_7_5,
    :I_10_0 => I_10_0, :I_15_0 => I_15_0, :I_20_0 => I_20_0,
    # Basic
    :solid_water => solid_water, :water => XA.Materials.water, :air => XA.Materials.air,
    # Tissue types
    :bone => XA.Materials.corticalbone,
    :cortical_bone => XA.Materials.corticalbone,
    :blood => XA.Materials.blood,
    :brain => XA.Materials.brain,
    :muscle => XA.Materials.muscle,
    :soft_tissue => XA.Materials.softtissue,
    :lung => XA.Materials.lung,
    :csf => XA.Materials.cerebrospinal_fluid,
    :gray_matter => gray_matter,
    :white_matter => white_matter,
    :iodine => XA.Materials.iodine,
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
# Region to Material Mapping for Polychromatic Simulation
# =============================================================================

"""
    get_region_materials() -> Vector{XA.Material}

Return a vector of materials indexed by region number (1-based).
Used for polychromatic simulation where μ_by_energy[region, energy] is needed.
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
# Mixture Calculations
# =============================================================================

"""
    create_mixture(materials, fractions; by_volume=true, name="mixture") -> XA.Material

Create a mixture material from base materials.

# Arguments
- `materials::Vector{XA.Material}`: Base materials to mix
- `fractions::Vector{Float64}`: Fractions (must sum to 1.0)
- `by_volume::Bool`: If true, fractions are volume fractions; if false, mass fractions
- `name::String`: Name for the mixture

# Example
```julia
# Create 50/50 water/bone by volume
mixture = create_mixture([XA.Materials.water, XA.Materials.bone], [0.5, 0.5]; by_volume=true)

# Create 1% iodine contrast (by mass)
iodine = create_mixture([XA.Materials.water, get_material(:I_10_0)], [0.99, 0.01]; by_volume=false)
```
"""
function create_mixture(
    materials::Vector{<:XA.Material},
    fractions::Vector{Float64};
    by_volume::Bool=true,
    name::String="mixture"
)::XA.Material
    @assert length(materials) == length(fractions) "Materials and fractions must have same length"
    @assert isapprox(sum(fractions), 1.0, atol=1e-6) "Fractions must sum to 1.0"

    n = length(materials)

    densities    = [ustrip(u"g/cm^3", mat.density) for mat in materials]
    compositions = [mat.composition for mat in materials]

    if by_volume
        mixture_density = sum(fractions[i] * densities[i] for i in 1:n)
        masses     = [fractions[i] * densities[i] for i in 1:n]
        total_mass = sum(masses)
        mass_fracs = masses ./ total_mass
    else
        mass_fracs      = fractions
        inv_rho_mix     = sum(fractions[i] / densities[i] for i in 1:n)
        mixture_density = 1.0 / inv_rho_mix
    end

    all_elements = Dict{Int, Float64}()
    for i in 1:n
        for (Z, wf) in compositions[i]
            all_elements[Z] = get(all_elements, Z, 0.0) + mass_fracs[i] * wf
        end
    end

    total     = sum(values(all_elements))
    Zs        = sort(collect(keys(all_elements)))
    final_wfs = [all_elements[Z]/total for Z in Zs]
    comp_dict = Dict{Int64, Float64}(Z => w for (Z, w) in zip(Zs, final_wfs))

    za_ratio = _zoa_ratio(comp_dict)
    I_mean   = _mean_excitation(comp_dict)

    return XA.Material(name, za_ratio, I_mean, mixture_density * u"g/cm^3", comp_dict)
end

"""
    update_region_with_contrast(
        base_material::XA.Material,
        contrast_material::XA.Material,
        region_voxel_count::Int,
        voxel_volume_cm3::Float64,
        target_concentration::Float64;
        by_mass::Bool=true
    ) -> XA.Material

Update a region's material with contrast agent.

# Arguments
- `base_material::XA.Material`: Base material (e.g., blood, tissue)
- `contrast_material::XA.Material`: Contrast agent (e.g., iodine solution)
- `region_voxel_count::Int`: Number of voxels in the region
- `voxel_volume_cm3::Float64`: Volume of one voxel in cm³
- `target_concentration::Float64`: Target concentration (0-1, fraction)
- `by_mass::Bool`: If true, target_concentration is mass fraction; if false, volume fraction

# Returns
New XA.Material with contrast added

# Example
```julia
# Update 1000 voxels of blood (0.001 cm³ each) with 5% iodine by mass
blood = XA.Materials.blood
iodine = get_material(:I_10_0)
updated_blood = update_region_with_contrast(
    blood, iodine, 1000, 0.001, 0.05; by_mass=true
)
```
"""
function update_region_with_contrast(
    base_material::XA.Material,
    contrast_material::XA.Material,
    region_voxel_count::Int,
    voxel_volume_cm3::Float64,
    target_concentration::Float64;
    by_mass::Bool=true
)::XA.Material
    @assert target_concentration >= 0 && target_concentration <= 1

    total_volume = region_voxel_count * voxel_volume_cm3

    if target_concentration == 0
        return base_material
    elseif target_concentration == 1
        return contrast_material
    end

    # Calculate volumes of each component
    base_volume = total_volume * (1 - target_concentration)
    contrast_volume = total_volume * target_concentration

    # Get densities
    ρ_base = base_material.density
    ρ_contrast = contrast_material.density

    # Calculate masses
    m_base = base_volume * ρ_base
    m_contrast = contrast_volume * ρ_contrast
    total_mass = m_base + m_contrast

    # Get mass fractions
    w_base = m_base / total_mass
    w_contrast = m_contrast / total_mass

    # Create mixture using mass fractions
    return create_mixture(
        [base_material, contrast_material],
        [w_base, w_contrast];
        by_volume=false,
        name = "$(base_material.name)_with_$(contrast_material.name)"
    )
end

# =============================================================================
# Iodine Contrast Material (clinical mg I/mL units)
# =============================================================================

"""Atomic masses (g/mol) for elements common in biological tissues."""
const _ATOMIC_MASSES = Dict(
    1  => 1.008,   6  => 12.011,  7  => 14.007,  8  => 15.999,
    11 => 22.990,  12 => 24.305,  15 => 30.974,  16 => 32.06,
    17 => 35.45,   19 => 39.098,  20 => 40.078,  26 => 55.845,
    53 => 126.904,
)

"""Mean excitation energies I (eV) for Bethe stopping-power formula (ICRU 37)."""
const _I_VALUES = Dict(
    1  => 19.2,   6  => 81.0,   7  => 82.0,   8  => 95.0,
    11 => 149.0,  12 => 156.0,  15 => 173.0,  16 => 180.0,
    17 => 174.0,  19 => 190.0,  20 => 191.0,  26 => 286.0,
    53 => 491.0,
)

"""
    iodine_contrast_material(base_mat, conc_mg_per_mL; density_g_per_mL=nothing) -> XA.Material

Create an iodine-doped material at the given clinical concentration.
# Arguments
- `base_mat::XA.Material`: base tissue (blood, gray matter, etc.)
- `conc_mg_per_mL::Float64`: iodine concentration in mg I/mL (= mg I/cm³)
- `density_g_per_mL`: override base material density (default: use base_mat.density)
# Example
```julia
blood_contrast = iodine_contrast_material(XA.Materials.blood, 5.0)  # 5 mg I/mL
```
"""
function iodine_contrast_material(
    base_mat::XA.Material,
    conc_mg_per_mL::Float64;
    density_g_per_mL::Union{Nothing,Float64}=nothing,
)::XA.Material
    conc_mg_per_mL < 0 && error("iodine_contrast_material: concentration must be ≥ 0")
    conc_mg_per_mL < 1e-6 && return base_mat

    rho = density_g_per_mL !== nothing ? density_g_per_mL :
          ustrip(u"g/cm^3", base_mat.density)

    # mg I/mL ÷ (g/mL × 1000 mg/g) = mass fraction of iodine
    f_I = (conc_mg_per_mL / 1000.0) / rho
    f_I = min(f_I, 1.0)
    f_s = 1.0 - f_I

    base_comp = Dict{Int,Float64}(base_mat.composition)
    new_comp  = Dict{Int,Float64}(Z => w * f_s for (Z, w) in base_comp)
    new_comp[53] = get(new_comp, 53, 0.0) + f_I

    new_density = rho + conc_mg_per_mL / 1000.0  # iodine mass adds to density

    log_I_num = sum(
        new_comp[Z] * (Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2)) *
        log(get(_I_VALUES, Z, 10.0 * Z))
        for Z in keys(new_comp)
    )
    log_I_den = sum(
        new_comp[Z] * (Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2))
        for Z in keys(new_comp)
    )
    I_mean = exp(log_I_num / log_I_den) * u"eV"
    ZA     = sum(w * Z / get(_ATOMIC_MASSES, Z, Float64(Z) * 2) for (Z, w) in new_comp)

    name = "$(base_mat.name)_$(round(Int, conc_mg_per_mL))mgI_per_mL"
    return XA.Material(name, ZA, I_mean, new_density * u"g/cm^3", new_comp)
end

"""
    calculate_mixture_attenuation(mixture::XA.Material, energy_keV::Float64) -> Float64

Calculate linear attenuation coefficient for a mixture at given energy.
"""
function calculate_mixture_attenuation(mixture::XA.Material, energy_keV::Float64)::Float64
    μ = XA.linear_attenuation_coeff(mixture, energy_keV * u"keV")
    return ustrip(u"cm^-1", μ)
end

# =============================================================================
# Exports
# =============================================================================
export Ca_50, Ca_100, Ca_200, Ca_300, Ca_400, Ca_500, Ca_600
export I_2_0, I_2_5, I_5_0, I_7_5, I_10_0, I_15_0, I_20_0
export solid_water
export gray_matter, white_matter
export get_material, MATERIALS_REGISTRY, validate_material_hu
export get_region_materials
export create_mixture, update_region_with_contrast
export calculate_mixture_attenuation
export iodine_contrast_material
