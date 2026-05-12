"""
    src/object/attenuation.jl

Per-energy and per-material attenuation coefficient queries for
simulation.  Thin wrapper over `XrayAttenuation.jl`, which reads NIST
XCOM photon cross-section data and returns linear (cm⁻¹) and mass
(cm²/g) attenuation coefficients.

Also exposes the `to_hounsfield(μ_volume; ...)` HU conversion entry —
the recommended way to map a reconstructed μ-domain volume to HU for
display / measurement.  Empirical water calibration (via `water_mask`)
is the most accurate path for polychromatic simulation; bowtie-aware
analytic `μ_water` from `compute_polychromatic_μ_water` in
`src/correction/bhc_sinogram.jl` is the equivalent without a mask.
"""

using Unitful
import XrayAttenuation as XA

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
    get_reference_μ_water(energy_keV::Float64=60.0)

Get linear attenuation coefficient of water at specified energy from XrayAttenuation.jl.

For polychromatic spectra, use `compute_effective_μ_material(water, energies, weights)` instead.

Default energy is 60 keV (approximate effective energy for 100 kVp spectrum).
"""
function get_reference_μ_water(energy_keV::Float64 = 60.0)
    return compute_μ_at_energy(XA.Materials.water, energy_keV)
end

# =============================================================================
# Convenience API for Reconstruction HU Conversion
# =============================================================================

"""
    to_hounsfield(reconstruction::AbstractArray; water_mask=nothing, μ_water=nothing)

Convert reconstruction from linear attenuation (cm⁻¹) to Hounsfield Units.

This is the recommended way to convert reconstruction results to HU.

# Arguments
- `reconstruction`: 3D array in μ (cm⁻¹) from FDK or iterative reconstruction

# Keyword Arguments
- `water_mask::Union{Nothing,AbstractArray{Bool}}=nothing`: Boolean mask identifying
  water regions. If provided, μ_water is empirically calibrated from the mean
  attenuation in the masked region. This is the most accurate approach.
- `μ_water::Union{Nothing,Real}=nothing`: Manual water attenuation value (cm⁻¹).
  If not provided and no water_mask given, defaults to NIST water at 70 keV.

# Returns
- Array in Hounsfield Units where water ≈ 0 HU

# Best Practices

**Option 1: Empirical calibration with water_mask (most accurate)**
```julia
recon = fdk_reconstruct(sinogram, geom, matrix_size)

# Create water mask from phantom
water_mask = phantom.mask .== REGION_SOLID_WATER

# Convert to HU with empirical calibration
recon_hu = to_hounsfield(result.reconstruction; water_mask=water_mask)
# Water region will be ~0 HU by construction
```

**Option 2: Manual μ_water specification**
```julia
# Measure μ_water from reconstruction
recon = result.reconstruction
cx, cy, cz = size(recon) .÷ 2
μ_water_measured = mean(recon[cx-5:cx+5, cy-5:cy+5, cz])

# Convert
recon_hu = to_hounsfield(recon; μ_water=μ_water_measured)
```

**Option 3: Use NIST reference (less accurate)**
```julia
# Uses NIST water at 70 keV (~0.193 cm⁻¹)
# May have offset due to reconstruction scaling
recon_hu = to_hounsfield(result.reconstruction)
```

# Notes
- For polychromatic CT, empirical calibration is preferred because
  the effective attenuation depends on spectrum, filtration, and patient size.
- For VMI from dual-energy, use `vmi_to_hu()` instead which accounts for
  the VMI synthesis process.
"""
function to_hounsfield(
        reconstruction::AbstractArray{T};
        water_mask::Union{Nothing, AbstractArray{Bool}} = nothing,
        μ_water::Union{Nothing, Real} = nothing
    ) where {T}
    # Determine μ_water for calibration
    if water_mask !== nothing
        # Empirical calibration from water region
        μ_cal = T(mean(reconstruction[water_mask]))
    elseif μ_water !== nothing
        μ_cal = T(μ_water)
    else
        # Default: NIST water at 70 keV (approximate effective energy for 120 kVp)
        μ_cal = T(get_reference_μ_water(70.0))
    end

    # Convert to HU
    return μ_to_HU(reconstruction, μ_cal)
end

# =============================================================================
# Basis Material Attenuation (for VMI / spectral decomposition)
# =============================================================================

"""
    get_basis_mu(material::Symbol, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for a basis material at the given energy.

Supports: `:water`, `:iodine` (5 mg/mL solution), `:calcium` (200 mg/cc equivalent).
"""
function get_basis_mu(material::Symbol, energy_keV::Float64)
    if material == :water
        return compute_μ_at_energy(XA.Materials.water, energy_keV)
    elseif material == :iodine
        μ_water = compute_μ_at_energy(XA.Materials.water, energy_keV)
        μ_ρ_I = compute_mass_μ_at_energy(XA.Elements.Iodine, energy_keV)
        return μ_water + (5.0 / 1000.0) * μ_ρ_I
    elseif material == :calcium
        μ_water = compute_μ_at_energy(XA.Materials.water, energy_keV)
        μ_ρ_Ca = compute_mass_μ_at_energy(XA.Elements.Calcium, energy_keV)
        return μ_water + (200.0 / 1000.0) * μ_ρ_Ca
    else
        error("Unknown basis material: $material. Use :water, :iodine, or :calcium")
    end
end

# Exports
export compute_μ_at_energy, compute_mass_μ_at_energy
export μ_to_HU, HU_to_μ, get_reference_μ_water, to_hounsfield
export get_basis_mu
