"""
    Physics/Attenuation.jl

X-ray attenuation coefficients using XrayAttenuation.jl (NIST XCOM database).

# Physics Background

X-ray attenuation in materials is described by the Beer-Lambert law:

**Equation 1 - Beer-Lambert Law:**
```
I(E) = I₀(E) · exp(-∫ μ(E, x) dx)
```

where:
- `I₀(E)` is the incident intensity at energy E
- `μ(E, x)` is the linear attenuation coefficient at position x
- The integral is taken along the ray path

**Equation 2 - Linear Attenuation Coefficient:**
```
μ(E) = ρ · (μ/ρ)(E)
```

where:
- `ρ` is the material density (g/cm³)
- `μ/ρ` is the mass attenuation coefficient (cm²/g) from NIST XCOM

# Key Physics Interactions

At diagnostic CT energies (20-150 keV), three interactions dominate:

1. **Photoelectric Effect** - Dominates at low energies, scales as Z⁴⁻⁵/E³
2. **Compton Scattering** - Energy-independent at high E, scales with electron density
3. **Coherent (Rayleigh) Scattering** - Small contribution, forward-peaked

# XrayAttenuation.jl Integration

This module uses XrayAttenuation.jl which provides:
- Embedded local XCOM database (no HTTP requests)
- Support for all 92 elements (H through U)
- Pre-configured materials (water, tissues, bones, polymers, contrast agents)
- K-edge discontinuity handling for accurate interpolation
- Full Unitful.jl integration

# Material Library

Access pre-defined materials via `XA.Materials`:

**Biological:**
- Water, Air, Soft Tissue, Adipose, Muscle, Blood, Cortical Bone

**Contrast Agents:**
- Iodine, Calcium, Hydroxyapatite

**Calibration Materials (Gammex 472, CatPhan):**
- Acrylic (PMMA), Teflon, various bone-equivalent inserts

**Custom Materials:**
- Create via `XA.Compound("H2O")` or `XA.Mixture(Dict("H2O"=>0.7, "Ca"=>0.3))`

# Inverse Problem Considerations

For material decomposition via backpropagation:

1. **Discrete Materials**: Optimize over material_id per voxel (categorical)
2. **Material Fractions**: Optimize over mixture fractions (continuous)
3. **Density**: Optimize density per voxel (continuous)

All functions are designed to be:
- **Enzyme.jl compatible** - Autodifferentiable
- **Reactant.jl compatible** - XLA-compilable
- **NIST-accurate** - Based on peer-reviewed data

# References

**NIST X-ray Data:**
- Hubbell, J.H., & Seltzer, S.M. (1995). "Tables of X-ray mass attenuation
  coefficients." NIST Physical Reference Data.
  https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients

- Berger, M.J., et al. (2010). "XCOM: Photon Cross Section Database."
  NIST Standard Reference Database 8.
  https://www.nist.gov/pml/xcom-photon-cross-sections-database

**XrayAttenuation.jl:**
- Dale Black. XrayAttenuation.jl: X-ray photon attenuation coefficients from
  the NIST XCOM database with K-edge aware interpolation.
  https://github.com/Dale-Black/XrayAttenuation.jl

**Tissue Composition:**
- ICRU Report 44 (1989). "Tissue Substitutes in Radiation Dosimetry and
  Measurement." International Commission on Radiation Units and Measurements.

- White, D.R., et al. (1987). "The composition of body tissues." British
  Journal of Radiology, 59(703), 1209-1219.

**Contrast Agents:**
- Alvarez, R.E., & Macovski, A. (1976). "Energy-selective reconstructions in
  X-ray computerized tomography." Physics in Medicine & Biology, 21(5), 733.

# Author

Dale Black, MolloiLab
Created: January 2026
"""

import XrayAttenuation as XA
using Unitful: ustrip, @u_str

# Re-export commonly used XA types and materials
export Materials, Elements, Compound, Mixture
const Materials = XA.Materials
const Elements = XA.Elements

# ==============================================================================
# Attenuation Queries (Autodiff-Compatible Wrappers)
# ==============================================================================

"""
    get_mass_attenuation(
        material,  # XA.Material, XA.Compound, or XA.Mixture
        energy_keV::Float64
    )::Float64

Get mass attenuation coefficient μ/ρ for a material at given energy.

**Autodiff-compatible**: This function is differentiable with respect to
`energy_keV` via Enzyme.jl.

# Arguments

- `material` - XA.Material (e.g., `XA.Materials.water`), XA.Compound, or XA.Mixture
- `energy_keV` - Photon energy in keV

# Returns

Mass attenuation coefficient μ/ρ in cm²/g

# Example

```julia
using BasisSimulator
import XrayAttenuation as XA

# Using pre-defined material
μ_ρ_water = get_mass_attenuation(XA.Materials.water, 60.0)

# Using element
μ_ρ_iron = get_mass_attenuation(XA.Elements.Fe, 70.0)

# Using compound
water_compound = XA.Compound("H2O")
μ_ρ = get_mass_attenuation(water_compound, 60.0)
```

# Physics

Uses NIST XCOM database with K-edge aware interpolation. The function
handles discontinuities at absorption edges automatically.
"""
function get_mass_attenuation(material, energy_keV::Float64)::Float64
    # Convert keV to Unitful quantity
    E = energy_keV * u"keV"

    # Call XA function and strip units
    μ_ρ = XA.mass_attenuation_coeff(material, E)

    return ustrip(u"cm^2/g", μ_ρ)
end

"""
    get_linear_attenuation(
        material,  # XA.Material with embedded density
        energy_keV::Float64
    )::Float64

Get linear attenuation coefficient μ for a material at given energy.

**Autodiff-compatible**: Differentiable with respect to `energy_keV`.

# Formula

```
μ(E) = ρ · (μ/ρ)(E)
```

# Arguments

- `material` - XA.Material (has embedded density), XA.Compound, or XA.Mixture
- `energy_keV` - Photon energy in keV

# Returns

Linear attenuation coefficient μ in cm⁻¹

# Example

```julia
import XrayAttenuation as XA

# Pre-defined material (has density)
μ_bone = get_linear_attenuation(XA.Materials.corticalbone, 60.0)  # cm⁻¹

# Custom mixture with density
tissue_mix = XA.Mixture(Dict("H2O"=>0.70, "C"=>0.20, "Ca"=>0.10))
μ_tissue = get_linear_attenuation(tissue_mix, 60.0)
```

# Note

For XA.Materials, density is embedded in the material definition.
For XA.Compound or XA.Mixture without density, use `get_mass_attenuation`
and multiply by your custom density.
"""
function get_linear_attenuation(material, energy_keV::Float64)::Float64
    # Convert keV to Unitful quantity
    E = energy_keV * u"keV"

    # Call XA function and strip units
    μ = XA.linear_attenuation_coeff(material, E)

    return ustrip(u"cm^-1", μ)
end

# ==============================================================================
# Batch Processing (Vectorized)
# ==============================================================================

"""
    get_mass_attenuation(
        material,
        energies_keV::AbstractVector{Float64}
    )::Vector{Float64}

Vectorized version: Get mass attenuation coefficients for multiple energies.

**Autodiff-compatible**: Differentiable with respect to energy array.

# Arguments

- `material` - XA.Material, XA.Compound, or XA.Mixture
- `energies_keV` - Array of photon energies in keV

# Returns

Vector of μ/ρ values in cm²/g

# Example

```julia
import XrayAttenuation as XA

energies = [30.0, 50.0, 70.0, 100.0, 150.0]
μ_ρ_array = get_mass_attenuation(XA.Materials.water, energies)
```
"""
function get_mass_attenuation(
        material,
        energies_keV::AbstractVector{Float64}
    )::Vector{Float64}

    # Convert to Unitful array
    E = energies_keV .* u"keV"

    # Call XA function (supports vector input)
    μ_ρ = XA.mass_attenuation_coeff(material, E)

    return ustrip.(u"cm^2/g", μ_ρ)
end

"""
    get_linear_attenuation(
        material,
        energies_keV::AbstractVector{Float64}
    )::Vector{Float64}

Vectorized version: Get linear attenuation coefficients for multiple energies.
"""
function get_linear_attenuation(
        material,
        energies_keV::AbstractVector{Float64}
    )::Vector{Float64}

    # Convert to Unitful array
    E = energies_keV .* u"keV"

    # Call XA function (supports vector input)
    μ = XA.linear_attenuation_coeff(material, E)

    return ustrip.(u"cm^-1", μ)
end

# ==============================================================================
# Polychromatic Attenuation (Spectrum Integration)
# ==============================================================================

"""
    compute_polychromatic_attenuation(
        material,
        spectrum  # XRaySpectrum from Spectrum.jl
    )::Float64

Compute effective attenuation coefficient for polychromatic spectrum.

**Autodiff-compatible**: Differentiable with respect to spectrum energies.

# Formula

```
μ_eff = Σᵢ [μ(Eᵢ) · Φ(Eᵢ)] / Σᵢ Φ(Eᵢ)
```

where Φ(Eᵢ) is the photon fluence at energy Eᵢ.

# Arguments

- `material` - XA.Material, XA.Compound, or XA.Mixture
- `spectrum` - X-ray spectrum (from `generate_spectrum`)

# Returns

Effective linear attenuation coefficient in cm⁻¹

# Example

```julia
import XrayAttenuation as XA

# Generate spectrum
spec = generate_spectrum(kVp=120.0, mAs=200.0)

# Compute effective attenuation
μ_eff_water = compute_polychromatic_attenuation(XA.Materials.water, spec)
μ_eff_bone = compute_polychromatic_attenuation(XA.Materials.corticalbone, spec)
```

# Physics

This accounts for beam hardening by computing the fluence-weighted
average attenuation coefficient across the entire spectrum.
"""
function compute_polychromatic_attenuation(material, spectrum)::Float64
    # Energy-weighted average of attenuation
    μ_sum = 0.0
    weight_sum = 0.0

    for (i, E) in enumerate(spectrum.energies)
        photons = spectrum.photons[i]

        if photons > 0
            μ = get_linear_attenuation(material, E)
            μ_sum += μ * photons
            weight_sum += photons
        end
    end

    if weight_sum == 0
        return 0.0
    end

    return μ_sum / weight_sum
end

# ==============================================================================
# Material Decomposition (Two-Material and Three-Material)
# ==============================================================================

"""
    compute_two_material_decomposition(
        material1,
        material2,
        energies_keV::Vector{Float64},
        measurements::Vector{Float64}
    )::Tuple{Float64, Float64}

Solve two-material decomposition using dual-energy measurements.

**Uses Alvarez-Macovski framework** (Alvarez & Macovski 1976).

# Arguments

- `material1` - First basis material (e.g., `XA.Materials.water`)
- `material2` - Second basis material (e.g., `XA.Materials.hydroxyapatite`)
- `energies_keV` - Two energies [E_low, E_high] in keV
- `measurements` - Two attenuation measurements [μ_low, μ_high] in cm⁻¹

# Returns

Tuple of path lengths (L1, L2) in cm for each material

# Example - Calcium Quantification

```julia
import XrayAttenuation as XA

# Dual-energy measurements
E_low, E_high = 80.0, 140.0
μ_low = 0.45  # cm⁻¹
μ_high = 0.30  # cm⁻¹

# Decompose into water and hydroxyapatite
L_water, L_HA = compute_two_material_decomposition(
    XA.Materials.water,
    XA.Materials.hydroxyapatite,
    [E_low, E_high],
    [μ_low, μ_high]
)

# Convert to calcium density
ρ_calcium_mg_cm3 = L_HA * 399.0  # 399 mg Ca per cm³ of HA
```

# References

- Alvarez & Macovski (1976) PMB - Energy-selective reconstructions
- Johnson et al. (2007) Med Phys - Material decomposition methods
"""
function compute_two_material_decomposition(
        material1,
        material2,
        energies_keV::Vector{Float64},
        measurements::Vector{Float64}
    )::Tuple{Float64, Float64}

    if length(energies_keV) != 2 || length(measurements) != 2
        error("Two-material decomposition requires exactly 2 energies and 2 measurements")
    end

    E1, E2 = energies_keV
    μ_meas_1, μ_meas_2 = measurements

    # Get attenuation coefficients for both materials at both energies
    μ1_E1 = get_linear_attenuation(material1, E1)
    μ1_E2 = get_linear_attenuation(material1, E2)
    μ2_E1 = get_linear_attenuation(material2, E1)
    μ2_E2 = get_linear_attenuation(material2, E2)

    # Solve linear system:
    # [μ1_E1  μ2_E1] [L1]   [μ_meas_1]
    # [μ1_E2  μ2_E2] [L2] = [μ_meas_2]

    # Determinant
    det = μ1_E1 * μ2_E2 - μ1_E2 * μ2_E1

    if abs(det) < 1e-10
        error("Materials are not linearly independent at these energies")
    end

    # Cramer's rule
    L1 = (μ_meas_1 * μ2_E2 - μ_meas_2 * μ2_E1) / det
    L2 = (μ1_E1 * μ_meas_2 - μ1_E2 * μ_meas_1) / det

    return (L1, L2)
end

"""
    compute_mixture_attenuation(
        material_fractions::Dict,
        energy_keV::Float64
    )::Float64

Compute attenuation for a volume-weighted mixture of materials.

**Critical for inverse problems**: Allows continuous optimization over
material fractions during backpropagation.

**Autodiff-compatible**: Differentiable with respect to both
`material_fractions` and `energy_keV`.

# Formula (Volume Mixture)

```
μ_mixture(E) = Σᵢ [fᵢ · μᵢ(E)]
```

where fᵢ is the volume fraction of material i.

# Arguments

- `material_fractions` - Dict mapping XA.Material → volume fraction
- `energy_keV` - Photon energy in keV

# Returns

Linear attenuation coefficient μ in cm⁻¹

# Example - Three-Material Decomposition

```julia
import XrayAttenuation as XA

# Blood vessel with iodine contrast
# 70% blood, 29.7% soft tissue, 0.3% iodine
mixture = Dict(
    XA.Materials.blood => 0.70,
    XA.Materials.softtissue => 0.297,
    XA.Materials.iodine => 0.003
)

μ_mix = compute_mixture_attenuation(mixture, 60.0)
```

# Example - Inverse Problem (Material Decomposition)

```julia
using Enzyme

# Initialize with random material fractions
θ = randn(3)
fractions_raw = softmax(θ)  # Ensure fractions sum to 1

# Create mixture dict
materials_list = [XA.Materials.water, XA.Materials.softtissue, XA.Materials.bone]
mixture = Dict(materials_list[i] => fractions_raw[i] for i in 1:3)

# Forward model
predicted_μ = compute_mixture_attenuation(mixture, 60.0)

# Define loss
loss = (predicted_μ - measured_μ)^2

# Backprop through Enzyme.jl
∇θ = Enzyme.gradient(Reverse, loss_fn, θ)
```

# Note

The mixture model assumes volume fractions (not mass fractions).
Ensure fractions sum to 1.0 for physical consistency.
"""
function compute_mixture_attenuation(
        material_fractions::Dict,
        energy_keV::Float64
    )::Float64

    # Validate fractions sum to ~1.0 (volume fractions)
    total_fraction = sum(values(material_fractions))
    if !isapprox(total_fraction, 1.0, atol=1e-4)
        @warn "Material fractions sum to $(total_fraction), should be 1.0"
    end

    μ_total = 0.0

    for (material, fraction) in material_fractions
        if fraction > 0
            μ_mat = get_linear_attenuation(material, energy_keV)
            μ_total += fraction * μ_mat
        end
    end

    return μ_total
end

# ==============================================================================
# Custom Material Creation Helpers
# ==============================================================================

"""
    create_custom_compound(chemical_formula::String)

Create a custom compound from a chemical formula.

# Arguments

- `chemical_formula` - String like "H2O", "CaCO3", "Ca10(PO4)6(OH)2"

# Returns

XA.Compound object that can be used with attenuation functions

# Example

```julia
# Create hydroxyapatite
ha = create_custom_compound("Ca10(PO4)6(OH)2")
μ_ρ = get_mass_attenuation(ha, 60.0)

# Create calcium carbonate
caco3 = create_custom_compound("CaCO3")
```

# Note

The compound will not have a density. To get linear attenuation (μ),
you must multiply μ/ρ by your custom density value.
"""
function create_custom_compound(chemical_formula::String)
    return XA.Compound(chemical_formula)
end

"""
    create_custom_mixture(mass_fractions::Dict{String, Float64})

Create a custom mixture from chemical formulas with mass fractions.

# Arguments

- `mass_fractions` - Dict mapping chemical formula → mass fraction

# Returns

XA.Mixture object that can be used with attenuation functions

# Example

```julia
# Create contrast-enhanced blood
# 99.5% blood composition, 0.5% iodine
blood_iodine = create_custom_mixture(Dict(
    "H2O" => 0.70,
    "C" => 0.15,
    "N" => 0.03,
    "O" => 0.10,
    "I" => 0.005,  # 5 mg/ml iodine
    "Na" => 0.01,
    "Ca" => 0.005
))

μ = get_linear_attenuation(blood_iodine, 60.0)
```

# Note

Mass fractions must sum to 1.0. Use this for creating custom tissue
compositions or contrast agent mixtures.
"""
function create_custom_mixture(mass_fractions::Dict{String, Float64})
    # Validate mass fractions sum to 1.0
    total = sum(values(mass_fractions))
    if !isapprox(total, 1.0, atol=1e-6)
        error("Mass fractions must sum to 1.0, got $(total)")
    end

    return XA.Mixture(mass_fractions)
end

# ==============================================================================
# Material Library Utilities
# ==============================================================================

"""
    list_available_materials()::Vector{Symbol}

List all available pre-defined materials from XrayAttenuation.jl

# Returns

Vector of material names that can be accessed via `XA.Materials.name`

# Example

```julia
materials = list_available_materials()
println("Available materials:")
for mat in materials
    println("  - ", mat)
end
```
"""
function list_available_materials()::Vector{Symbol}
    # Get all field names from XA.Materials NamedTuple
    return collect(propertynames(XA.Materials))
end

"""
    list_available_elements()::Vector{Symbol}

List all available elements from XrayAttenuation.jl

# Returns

Vector of element symbols (e.g., :H, :C, :O, :Ca, :I)
"""
function list_available_elements()::Vector{Symbol}
    return collect(propertynames(XA.Elements))
end

# ==============================================================================
# Exports
# ==============================================================================

export get_mass_attenuation, get_linear_attenuation
export compute_polychromatic_attenuation
export compute_mixture_attenuation
export compute_two_material_decomposition
export create_custom_compound, create_custom_mixture
export list_available_materials, list_available_elements
