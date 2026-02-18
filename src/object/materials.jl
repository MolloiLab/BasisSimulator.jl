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

# NOTE: The Gammex phantom body is water-equivalent. We use pure water here.
# The XrayAttenuation.jl "gammex_water" material has incorrect composition (13% Cl).
# Real Gammex solid water should be nearly identical to pure water in attenuation.
const solid_water = XA.Materials.water

# =============================================================================
# Material Registry
# =============================================================================

# Helper to safely get material from XA.Materials with different naming conventions
function _get_xa_material(name::String)::XA.Material
    # Try direct name first
    sym = Symbol(name)
    if haskey(XA.Materials, sym)
        return XA.Materials[sym]
    end
    # Try with underscore removed
    name_no_underscore = replace(name, "_" => "")
    sym2 = Symbol(name_no_underscore)
    if haskey(XA.Materials, sym2)
        return XA.Materials[sym2]
    end
    # Try common variations
    if name == "corticalbone"
        return _get_xa_material("corticalbone")
    end
    error("Material $name not found in XrayAttenuation")
end

const MATERIALS_REGISTRY = Dict{Symbol, XA.Material}(
    # Gammex 472 Calcium
    :Ca_50 => Ca_50, :Ca_100 => Ca_100, :Ca_200 => Ca_200, :Ca_300 => Ca_300,
    :Ca_400 => Ca_400, :Ca_500 => Ca_500, :Ca_600 => Ca_600,
    # Gammex 472 Iodine
    :I_2_0 => I_2_0, :I_2_5 => I_2_5, :I_5_0 => I_5_0, :I_7_5 => I_7_5,
    :I_10_0 => I_10_0, :I_15_0 => I_15_0, :I_20_0 => I_20_0,
    # Basic
    :solid_water => solid_water, :water => XA.Materials.water, :air => XA.Materials.air,
    # Tissue types for XCAT
    :bone => _get_xa_material("corticalbone"),
    :cortical_bone => _get_xa_material("corticalbone"),
    :blood => _get_xa_material("blood"),
    :brain => _get_xa_material("brain"),
    :muscle => _get_xa_material("muscle"),
    :soft_tissue => _get_xa_material("softtissue"),
    :lung => _get_xa_material("lung"),
    :csf => _get_xa_material("cerebrospinal_fluid"),
    :gray_matter => _get_xa_material("brain"),
    :white_matter => _get_xa_material("brain"),
    :iodine => _get_xa_material("iodine"),
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
# Mixture Calculations
# =============================================================================

"""
    MixtureComponent

A component in a material mixture with its fraction and properties.
"""
struct MixtureComponent
    material::XA.Material
    mass_fraction::Float64
    volume_fraction::Float64
    density::Float64  # g/cm³
    atomic_numbers::Vector{Int}
    mass_fractions::Vector{Float64}
end

"""
    create_mixture(
        materials::Vector{XA.Material},
        fractions::Vector{Float64};
        by_volume::Bool=true,
        name::String="mixture"
    ) -> XA.Material

Create a mixture material from base materials.

# Arguments
- `materials::Vector{XA.Material}`: Base materials to mix
- `fractions::Vector{Float64}`: Fractions (must sum to 1.0)
- `by_volume::Bool`: If true, fractions are volume fractions; if false, mass fractions
- `name::String`: Name for the mixture

# Returns
XA.Material with computed mixture properties

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
    
    # Get densities and compositions for each material
    densities = Float64[]
    compositions = []
    
    for mat in materials
        # Extract density as Float64 using ustrip
        density_val = mat.density
        rho = try 
            ustrip(density_val)
        catch
            Float64(density_val)
        end
        push!(densities, rho)
        
        # Get composition (Dict{Z, mass_fraction})
        push!(compositions, mat.composition)
    end
    
    # Calculate mixture density
    if by_volume
        # Volume-weighted: ρ_mix = Σ(v_i * ρ_i)
        mixture_density = sum(fractions[i] * densities[i] for i in 1:n)
        
        # Convert volume fractions to mass fractions
        # m_i = v_i * ρ_i, then normalize
        masses = [fractions[i] * densities[i] for i in 1:n]
        total_mass = sum(masses)
        mass_fracs = masses ./ total_mass
    else
        # Mass fractions provided directly
        mass_fracs = fractions
        
        # Calculate density: 1/ρ_mix = Σ(w_i/ρ_i)
        inv_rho_mix = sum(fractions[i] / densities[i] for i in 1:n)
        mixture_density = 1.0 / inv_rho_mix
    end
    
    # Calculate effective mixture composition
    # Combine all elements from all materials
    all_elements = Dict{Int, Float64}()
    
    for i in 1:n
        for (Z, wf) in compositions[i]
            element_mass = mass_fracs[i] * wf
            all_elements[Z] = get(all_elements, Z, 0.0) + element_mass
        end
    end
    
    # Normalize
    total = sum(values(all_elements))
    Zs = sort(collect(keys(all_elements)))
    final_wfs = [all_elements[Z]/total for Z in Zs]
    
    # Create composition dict (required by XA.Material)
    comp_dict = Dict{Int64, Float64}(Z => w for (Z, w) in zip(Zs, final_wfs))
    
    # Calculate Z/A ratio for the mixture
    # Z/A ≈ sum(w_i * Z_i / A_i), simplified: use weighted average
    za_ratio = mixture_density / sum(Zs[i] * final_wfs[i] for i in 1:length(Zs))
    
    # Create the material using XA.Material (5 positional args)
    return XA.Material(
        name,
        za_ratio,  # Z/A ratio
        75.0,  # I (mean excitation energy in eV - typical value)
        mixture_density,  # density in g/cm³
        comp_dict  # composition dict
    )
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

"""
    create_iodine_blood_mixture(
        blood::XA.Material,
        iodine_concentration_mg_g::Float64;
        base_iodine::XA.Material=get_material(:I_10_0)
    ) -> XA.Material

Create iodine contrast blood mixture given concentration.

# Arguments
- `blood::XA.Material`: Base blood material
- `iodine_concentration_mg_g::Float64`: Iodine concentration in mg/g (mass of iodine per gram tissue)
- `base_iodine::XA.Material`: Base iodine material to use

# Returns
XA.Material with updated iodine concentration

# Example
```julia
blood = XA.Materials.blood
iodine_5mg_g = create_iodine_blood_mixture(blood, 5.0)  # 5 mg/g iodine
```
"""
function create_iodine_blood_mixture(
    blood::XA.Material,
    iodine_concentration_mg_g::Float64;
    base_iodine::XA.Material=get_material(:I_10_0)
)::XA.Material
    # Convert: mg/g -> mass fraction (mg/g * 1g/1000mg = 0.001)
    mass_fraction = iodine_concentration_mg_g / 1000.0
    
    # Cap at reasonable maximum
    mass_fraction = min(mass_fraction, 0.5)  # Max 50% iodine by mass
    
    if mass_fraction < 1e-6
        return blood  # No significant contrast
    end
    
    return create_mixture(
        [blood, base_iodine],
        [1 - mass_fraction, mass_fraction];
        by_volume=false,
        name = "blood_iodine_$(round(Int, iodine_concentration_mg_g))mg"
    )
end

"""
    calculate_mixture_attenuation(mixture::XA.Material, energy_keV::Float64) -> Float64

Calculate linear attenuation coefficient for a mixture at given energy.
"""
function calculate_mixture_attenuation(mixture::XA.Material, energy_keV::Float64)::Float64
    μ = 0.0
    comp = XA.composition(mixture)
    Zs = comp.first
    wfs = comp.second
    
    for (Z, wf) in zip(Zs, wfs)
        μ += wf * XA.attenuation_coefficient(Z, energy_keV)
    end
    
    return μ * mixture.density
end

# =============================================================================
# Exports
# =============================================================================

export Ca_50, Ca_100, Ca_200, Ca_300, Ca_400, Ca_500, Ca_600
export I_2_0, I_2_5, I_5_0, I_7_5, I_10_0, I_15_0, I_20_0
export solid_water
export get_material, MATERIALS_REGISTRY, validate_material_hu
export get_region_materials
export create_mixture, update_region_with_contrast, create_iodine_blood_mixture
export calculate_mixture_attenuation
export MixtureComponent
