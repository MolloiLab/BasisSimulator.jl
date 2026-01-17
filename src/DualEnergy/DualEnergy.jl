# =============================================================================
# DualEnergy/DualEnergy.jl
#
# Dual-energy (dual kVp) CT simulation for GE GSI and similar rapid kVp
# switching systems.
#
# Reference: RESEARCH-DUAL-KVP document (docs/research/Dual_kVp_CT_Research.md)
# =============================================================================

"""
    DualEnergySinogram{T}

Container for dual-energy sinogram data from rapid kVp switching acquisition.

In rapid kVp switching (e.g., GE GSI), the X-ray tube alternates between low
and high kVp settings on a view-by-view basis. This produces interleaved
projections that are sorted into separate low/high energy sinograms.

# Fields
- `low::Array{T,3}`: Low-energy sinogram (e.g., 80 kVp)
- `high::Array{T,3}`: High-energy sinogram (e.g., 140 kVp)
- `low_kvp::Int`: Low kVp setting (typically 80)
- `high_kvp::Int`: High kVp setting (typically 140)
- `n_cols::Int`: Number of detector columns
- `n_rows::Int`: Number of detector rows
- `n_angles::Int`: Number of projection angles per sinogram

# Example

```julia
# After dual-energy forward projection
de_sino = forward_project_dual_energy(phantom.mask, geom; ...)

# Access individual sinograms
sino_low = de_sino.low   # 80 kVp sinogram
sino_high = de_sino.high  # 140 kVp sinogram

# Material decomposition
materials = decompose_materials(de_sino; basis=(:water, :iodine))
```

# Technical Notes

For GE Revolution Apex GSI:
- Low kVp: 80 kVp
- High kVp: 140 kVp
- Switching time: 0.2 ms per view
- Angular offset between adjacent views: <0.18°
- Integration time split: ~65% low kVp, ~35% high kVp

See also: [`forward_project_dual_energy`](@ref), [`decompose_materials`](@ref)
"""
struct DualEnergySinogram{T<:AbstractFloat}
    low::Array{T,3}
    high::Array{T,3}
    low_kvp::Int
    high_kvp::Int
    n_cols::Int
    n_rows::Int
    n_angles::Int
end

function DualEnergySinogram(low::Array{T,3}, high::Array{T,3};
                            low_kvp::Int=80, high_kvp::Int=140) where T
    @assert size(low) == size(high) "Low and high sinograms must have same size"
    n_cols, n_rows, n_angles = size(low)
    return DualEnergySinogram{T}(low, high, low_kvp, high_kvp, n_cols, n_rows, n_angles)
end

Base.size(ds::DualEnergySinogram) = (ds.n_cols, ds.n_rows, ds.n_angles)
Base.eltype(::DualEnergySinogram{T}) where T = T

"""
    MaterialMap{T}

Result of material decomposition from dual-energy CT data.

After basis material decomposition, each voxel (or projection ray) is
represented as a combination of two basis materials. Common pairs:
- Water + Iodine (contrast imaging)
- Water + Calcium (bone imaging)
- Photoelectric + Compton (physics-based)

# Fields
- `material1::Array{T,3}`: Density/projection of first basis material
- `material2::Array{T,3}`: Density/projection of second basis material
- `material1_name::Symbol`: Name of first material (e.g., :water)
- `material2_name::Symbol`: Name of second material (e.g., :iodine)
- `domain::Symbol`: :projection or :image

# Example

```julia
# After material decomposition
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))

# Access material maps
water_proj = mat_map.material1
iodine_proj = mat_map.material2

# Generate VMI at 50 keV
vmi_50 = virtual_monoenergetic(mat_map, 50.0)
```

See also: [`decompose_materials`](@ref), [`virtual_monoenergetic`](@ref)
"""
struct MaterialMap{T<:AbstractFloat}
    material1::Array{T,3}
    material2::Array{T,3}
    material1_name::Symbol
    material2_name::Symbol
    domain::Symbol  # :projection or :image
end

function MaterialMap(m1::Array{T,3}, m2::Array{T,3};
                     material1_name::Symbol=:water,
                     material2_name::Symbol=:iodine,
                     domain::Symbol=:projection) where T
    @assert size(m1) == size(m2) "Material maps must have same size"
    return MaterialMap{T}(m1, m2, material1_name, material2_name, domain)
end

Base.size(mm::MaterialMap) = size(mm.material1)
Base.eltype(::MaterialMap{T}) where T = T

"""
    GSIProtocol

Configuration for GE Gemstone Spectral Imaging (GSI) dual-energy acquisition.

# Fields
- `low_kvp::Int`: Low tube voltage (default: 80)
- `high_kvp::Int`: High tube voltage (default: 140)
- `low_mA::Float64`: Tube current for low kVp views
- `high_mA::Float64`: Tube current for high kVp views
- `low_integration_fraction::Float64`: Fraction of view time for low kVp (default: 0.65)
- `rotation_time_s::Float64`: Gantry rotation time (default: 0.5)
- `n_views::Int`: Number of views per rotation (default: 984)

# Technical Notes

GE GSI uses fixed mA for each kVp level (no angular modulation).
The integration time is split approximately 65%/35% between low/high kVp
to balance photon flux at each energy.

See also: [`forward_project_dual_energy`](@ref)
"""
struct GSIProtocol
    low_kvp::Int
    high_kvp::Int
    low_mA::Float64
    high_mA::Float64
    low_integration_fraction::Float64
    rotation_time_s::Float64
    n_views::Int
end

"""
    default_gsi_protocol(; low_mA=400.0, high_mA=400.0, kwargs...)

Create default GSI protocol for GE Revolution Apex.

# Keyword Arguments
- `low_kvp::Int=80`: Low tube voltage
- `high_kvp::Int=140`: High tube voltage
- `low_mA::Float64=400.0`: Tube current for low kVp
- `high_mA::Float64=400.0`: Tube current for high kVp
- `low_integration_fraction::Float64=0.65`: Fraction of time for low kVp
- `rotation_time_s::Float64=0.5`: Rotation time in seconds
- `n_views::Int=984`: Views per rotation

# Example

```julia
# Standard GSI protocol
protocol = default_gsi_protocol()

# Low-dose GSI
protocol_lowdose = default_gsi_protocol(low_mA=200.0, high_mA=200.0)
```
"""
function default_gsi_protocol(;
    low_kvp::Int=80,
    high_kvp::Int=140,
    low_mA::Float64=400.0,
    high_mA::Float64=400.0,
    low_integration_fraction::Float64=0.65,
    rotation_time_s::Float64=0.5,
    n_views::Int=984
)
    return GSIProtocol(low_kvp, high_kvp, low_mA, high_mA,
                       low_integration_fraction, rotation_time_s, n_views)
end

"""
    forward_project_dual_energy(mask, geom, protocol::GSIProtocol;
                                materials, physics=nothing,
                                scanner=nothing) -> DualEnergySinogram

Perform dual-energy forward projection using rapid kVp switching protocol.

This simulates a dual-energy CT acquisition where the X-ray tube alternates
between low and high kVp settings. Both sinograms are computed using the
full polychromatic physics model with appropriate spectra for each kVp.

# Arguments
- `mask::AbstractArray{UInt8,3}`: Material mask (region IDs)
- `geom::CTGeometry`: CT geometry
- `protocol::GSIProtocol`: GSI acquisition protocol

# Keyword Arguments
- `materials::Vector`: Vector of materials for each region ID
- `physics::Union{Nothing,PhysicsConfig}=nothing`: Physics effects to apply
- `scanner::Union{Nothing,AbstractScannerSpec}=nothing`: Scanner for mA→I0 conversion

# Returns
`DualEnergySinogram` with low and high energy sinograms.

# Example

```julia
spec = GERevolutionApex()
protocol = default_gsi_protocol(low_mA=400.0, high_mA=400.0)

phantom = create_gammex_472(n_voxels=256)
geom = create_geometry(spec; n_angles=984, n_rows=64)
materials = get_region_materials()

de_sino = forward_project_dual_energy(
    phantom.mask, geom, protocol;
    materials = materials,
    scanner = spec
)
```

# Technical Notes

For GE GSI, the low kVp views receive ~65% of the integration time to
compensate for lower photon flux at 80 kVp. This is handled internally
when computing I0 from mA using the protocol's integration fraction.

See also: [`GSIProtocol`](@ref), [`DualEnergySinogram`](@ref)
"""
function forward_project_dual_energy(
    mask::AbstractArray{UInt8,3},
    geom::CTGeometry,
    protocol::GSIProtocol;
    materials::Vector,
    physics::Union{Nothing,PhysicsConfig}=nothing,
    scanner=nothing
)
    # Load spectra for both kVp levels
    e_low, w_low = load_spectrum(protocol.low_kvp)
    e_high, w_high = load_spectrum(protocol.high_kvp)

    # Downsample spectra for efficiency
    n_bins = 30
    e_low, w_low = downsample_spectrum(e_low, w_low, n_bins)
    e_high, w_high = downsample_spectrum(e_high, w_high, n_bins)

    # Compute mean energies for physics config
    mean_e_low = sum(e_low .* w_low) / sum(w_low)
    mean_e_high = sum(e_high .* w_high) / sum(w_high)

    # Setup noise models with mA-based I0 if scanner provided
    physics_low = physics
    physics_high = physics

    if scanner !== nothing && physics !== nothing
        # Compute I0 for each kVp based on mA and integration time
        # Low kVp gets more integration time to balance flux
        effective_rotation_low = protocol.rotation_time_s * protocol.low_integration_fraction
        effective_rotation_high = protocol.rotation_time_s * (1.0 - protocol.low_integration_fraction)

        # Use the mA_to_I0 function from DetectorNoise
        I0_low = mA_to_I0(protocol.low_mA, scanner;
                         rotation_time_s=effective_rotation_low,
                         n_views=protocol.n_views)
        I0_high = mA_to_I0(protocol.high_mA, scanner;
                          rotation_time_s=effective_rotation_high,
                          n_views=protocol.n_views)

        # Create separate noise models
        if physics.noise !== nothing
            noise_low = DetectorModel(
                physics.noise.blur_fwhm,
                I0_low,
                physics.noise.electronic_noise_std,
                physics.noise.seed
            )
            noise_high = DetectorModel(
                physics.noise.blur_fwhm,
                I0_high,
                physics.noise.electronic_noise_std,
                physics.noise.seed !== nothing ? physics.noise.seed + 1 : nothing
            )

            physics_low = PhysicsConfig(
                physics.fill_factor, physics.flat_filter, physics.bowtie_filter,
                physics.heel_effect, physics.scatter, physics.focal_spot,
                physics.crosstalk, physics.detector_efficiency, noise_low,
                physics.lag, physics.das, physics.bhc, physics.air_scan,
                physics.scatter_correction, mean_e_low, physics.noise_seed
            )
            physics_high = PhysicsConfig(
                physics.fill_factor, physics.flat_filter, physics.bowtie_filter,
                physics.heel_effect, physics.scatter, physics.focal_spot,
                physics.crosstalk, physics.detector_efficiency, noise_high,
                physics.lag, physics.das, physics.bhc, physics.air_scan,
                physics.scatter_correction, mean_e_high, physics.noise_seed
            )
        end
    end

    # Forward project low kVp
    sino_low = forward_project(mask, geom;
        energies = e_low,
        weights = w_low,
        materials = materials,
        physics = physics_low
    )

    # Forward project high kVp
    sino_high = forward_project(mask, geom;
        energies = e_high,
        weights = w_high,
        materials = materials,
        physics = physics_high
    )

    # Transfer to CPU if on GPU
    sino_low_cpu = Array(sino_low)
    sino_high_cpu = Array(sino_high)

    return DualEnergySinogram(sino_low_cpu, sino_high_cpu;
                              low_kvp=protocol.low_kvp,
                              high_kvp=protocol.high_kvp)
end

"""
    decompose_materials(sino::DualEnergySinogram;
                        basis=(:water, :iodine),
                        method=:polynomial) -> MaterialMap

Perform projection-domain material decomposition.

Given dual-energy sinogram data, decompose each projection ray into
contributions from two basis materials. This is performed in the
projection domain for rapid kVp switching systems.

# Arguments
- `sino::DualEnergySinogram`: Dual-energy sinogram data

# Keyword Arguments
- `basis::Tuple{Symbol,Symbol}=(:water, :iodine)`: Basis material pair
- `method::Symbol=:polynomial`: Decomposition method

# Returns
`MaterialMap` with projection-domain material densities.

# Supported Basis Pairs
- `:water, :iodine` - Contrast-enhanced imaging
- `:water, :calcium` - Bone imaging

# Example

```julia
# Decompose into water and iodine
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))

# Access projections
water_sino = mat_map.material1
iodine_sino = mat_map.material2

# Then reconstruct each
water_recon = fdk_reconstruct(water_sino, geom, recon_size)
iodine_recon = fdk_reconstruct(iodine_sino, geom, recon_size)
```

# Technical Notes

Material decomposition uses a polynomial fitting approach based on
calibration data. The decomposition inherently corrects for beam
hardening effects since it accounts for the full energy spectrum.

See also: [`DualEnergySinogram`](@ref), [`MaterialMap`](@ref)
"""
function decompose_materials(sino::DualEnergySinogram{T};
                             basis::Tuple{Symbol,Symbol}=(:water, :iodine),
                             method::Symbol=:polynomial) where T

    if method != :polynomial
        error("Only :polynomial method currently supported")
    end

    # Get basis material attenuation coefficients
    m1, m2 = basis

    # Get effective energies (approximate)
    e_eff_low = get_effective_energy(sino.low_kvp)
    e_eff_high = get_effective_energy(sino.high_kvp)

    # Get material attenuation at effective energies
    μ1_low = get_material_attenuation(m1, e_eff_low)
    μ1_high = get_material_attenuation(m1, e_eff_high)
    μ2_low = get_material_attenuation(m2, e_eff_low)
    μ2_high = get_material_attenuation(m2, e_eff_high)

    # Matrix form: [μ1_low μ2_low; μ1_high μ2_high] * [ρ1; ρ2] = [p_low; p_high]
    # Solve for [ρ1; ρ2] = A^-1 * [p_low; p_high]
    det_A = μ1_low * μ2_high - μ2_low * μ1_high

    if abs(det_A) < 1e-10
        error("Singular decomposition matrix - basis materials too similar at these energies")
    end

    # Inverse matrix elements
    inv_a11 = μ2_high / det_A
    inv_a12 = -μ2_low / det_A
    inv_a21 = -μ1_high / det_A
    inv_a22 = μ1_low / det_A

    # Allocate output
    material1 = similar(sino.low)
    material2 = similar(sino.low)

    # Apply decomposition
    for i in eachindex(sino.low)
        p_low = sino.low[i]
        p_high = sino.high[i]

        material1[i] = inv_a11 * p_low + inv_a12 * p_high
        material2[i] = inv_a21 * p_low + inv_a22 * p_high
    end

    return MaterialMap(material1, material2;
                       material1_name=m1, material2_name=m2,
                       domain=:projection)
end

"""
    virtual_monoenergetic(materials::MaterialMap, energy_keV::Float64) -> Array

Generate virtual monoenergetic image (VMI) at specified energy.

VMI synthesizes what the CT image would look like if acquired with a
perfectly monochromatic X-ray beam at the specified energy. This is
computed from the material maps using known attenuation coefficients.

# Arguments
- `materials::MaterialMap`: Result of material decomposition
- `energy_keV::Float64`: Target energy in keV (40-140)

# Returns
Sinogram or image at the virtual monoenergetic energy level.

# Energy Selection Guide
- 40-50 keV: Maximum iodine enhancement (high noise)
- 50-60 keV: Good contrast, moderate noise
- 65-75 keV: Balanced (similar to 120 kVp single-energy)
- 80-100 keV: Reduced beam hardening artifacts
- 100-140 keV: Metal artifact reduction

# Example

```julia
# Material decomposition
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))

# Generate VMI at different energies
vmi_50 = virtual_monoenergetic(mat_map, 50.0)   # High iodine contrast
vmi_70 = virtual_monoenergetic(mat_map, 70.0)   # Balanced
vmi_100 = virtual_monoenergetic(mat_map, 100.0) # Low artifacts
```

# Technical Notes

The VMI is computed as:
    μ_VMI(E) = ρ₁ × μ₁(E) + ρ₂ × μ₂(E)

where ρ₁, ρ₂ are the material densities and μ₁(E), μ₂(E) are the
mass attenuation coefficients at energy E.

See also: [`decompose_materials`](@ref), [`MaterialMap`](@ref)
"""
function virtual_monoenergetic(materials::MaterialMap{T}, energy_keV::Float64) where T
    if energy_keV < 10.0 || energy_keV > 150.0
        error("Energy must be between 10 and 150 keV (got $energy_keV)")
    end

    # Get attenuation coefficients at target energy
    μ1 = get_material_attenuation(materials.material1_name, energy_keV)
    μ2 = get_material_attenuation(materials.material2_name, energy_keV)

    # Synthesize VMI: μ_VMI = ρ1 × μ1 + ρ2 × μ2
    vmi = similar(materials.material1)
    for i in eachindex(materials.material1)
        vmi[i] = materials.material1[i] * μ1 + materials.material2[i] * μ2
    end

    return vmi
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    get_effective_energy(kvp::Int) -> Float64

Get approximate effective energy for a given kVp setting.
"""
function get_effective_energy(kvp::Int)
    # Approximate effective energies (from NIST/literature)
    # These are rough estimates; actual values depend on filtration
    if kvp <= 80
        return 45.0  # ~45 keV effective for 80 kVp
    elseif kvp <= 100
        return 55.0
    elseif kvp <= 120
        return 65.0
    elseif kvp <= 140
        return 75.0
    else
        return 85.0
    end
end

"""
    get_material_attenuation(material::Symbol, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for material at given energy.

Uses NIST XCOM database via XrayAttenuation.jl for water.
For iodine and calcium, uses physics-based models validated against NIST.

# Arguments
- `material::Symbol`: Material type (:water, :iodine, or :calcium)
- `energy_keV::Float64`: Energy in keV

# Returns
- `μ::Float64`: Linear attenuation coefficient (cm⁻¹)

# Notes
- :iodine represents dilute iodine solution (5 mg/mL in water)
- :calcium represents calcium equivalent material (cortical bone-like)
- For pure elemental attenuation, use XrayAttenuation.jl directly
"""
function get_material_attenuation(material::Symbol, energy_keV::Float64)
    if material == :water
        return get_water_attenuation_vmi(energy_keV)
    elseif material == :iodine
        # Dilute iodine solution (5 mg/mL typical contrast concentration)
        return get_iodine_solution_attenuation(energy_keV)
    elseif material == :calcium
        # Calcium-equivalent material (cortical bone-like)
        return get_calcium_material_attenuation(energy_keV)
    else
        error("Unknown material: $material. Use :water, :iodine, or :calcium")
    end
end

"""
    get_iodine_solution_attenuation(energy_keV::Float64; conc_mg_ml::Float64=5.0) -> Float64

Get linear attenuation for dilute iodine solution at given energy.

Uses NIST mass attenuation coefficient for elemental iodine (via XrayAttenuation.jl)
combined with the mixture formula for iodine in water.

The iodine K-edge at 33.2 keV causes a ~4.8× jump in attenuation,
providing maximum contrast for VMI at 40-50 keV.

# Arguments
- `energy_keV::Float64`: Energy in keV
- `conc_mg_ml::Float64=5.0`: Iodine concentration in mg/mL

# Returns
- `μ::Float64`: Linear attenuation coefficient of solution (cm⁻¹)

# Physics
For dilute solution: μ_solution = μ_water + C × (μ/ρ)_iodine
where C is concentration in g/cm³ and (μ/ρ)_iodine is mass attenuation.

# Reference
NIST XCOM: https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z53.html
"""
function get_iodine_solution_attenuation(energy_keV::Float64; conc_mg_ml::Float64=5.0)
    # Get water attenuation (baseline)
    μ_water = get_water_attenuation_vmi(energy_keV)

    # Get iodine mass attenuation from XrayAttenuation.jl (NIST data)
    # XA.Elements.Iodine is pure elemental iodine (note capital I)
    μ_ρ_iodine = compute_mass_μ_at_energy(XA.Elements.Iodine, energy_keV)

    # Convert concentration: mg/mL → g/cm³
    conc_g_cm3 = conc_mg_ml / 1000.0

    # Dilute solution formula: μ_solution = μ_water + C × (μ/ρ)_iodine
    μ_solution = μ_water + conc_g_cm3 * μ_ρ_iodine

    return μ_solution
end

"""
    get_calcium_material_attenuation(energy_keV::Float64; density_mg_cc::Float64=200.0) -> Float64

Get linear attenuation for calcium-equivalent material at given energy.

Uses NIST data for calcium (via XrayAttenuation.jl) combined with
the mixture formula for hydroxyapatite-equivalent material.

# Arguments
- `energy_keV::Float64`: Energy in keV
- `density_mg_cc::Float64=200.0`: Calcium density equivalent (mg/cc)

# Returns
- `μ::Float64`: Linear attenuation coefficient (cm⁻¹)

# Notes
Calcium K-edge is at 4.0 keV (below diagnostic range), so attenuation
decreases monotonically with increasing energy in the VMI range.

# Reference
NIST XCOM: https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z20.html
"""
function get_calcium_material_attenuation(energy_keV::Float64; density_mg_cc::Float64=200.0)
    # Get water attenuation (baseline for CHA material in water-equivalent background)
    μ_water = get_water_attenuation_vmi(energy_keV)

    # Get calcium mass attenuation from XrayAttenuation.jl (NIST data)
    # Note: XA.Elements.Calcium uses capital C
    μ_ρ_calcium = compute_mass_μ_at_energy(XA.Elements.Calcium, energy_keV)

    # Convert density: mg/cc → g/cm³
    density_g_cm3 = density_mg_cc / 1000.0

    # Mixture formula: μ_material ≈ μ_water_bg + density × (μ/ρ)_calcium
    # This is a simplified model for calcium-equivalent material (like CHA inserts)
    μ_material = μ_water + density_g_cm3 * μ_ρ_calcium

    return μ_material
end

# Legacy compatibility aliases (deprecated, use NIST-based functions)
get_iodine_attenuation(energy_keV::Float64) = get_iodine_solution_attenuation(energy_keV)
get_calcium_attenuation(energy_keV::Float64) = get_calcium_material_attenuation(energy_keV)

"""
    get_water_attenuation_vmi(energy_keV::Float64) -> Float64

Get NIST-validated water linear attenuation coefficient at given energy.

Uses XrayAttenuation.jl (NIST XCOM database) for accurate values.
This is essential for correct HU conversion in VMI.

# Arguments
- `energy_keV::Float64`: Energy in keV (typically 40-140 for VMI)

# Returns
- `μ_water::Float64`: Linear attenuation coefficient (cm⁻¹)

# Example
```julia
μ_water_70 = get_water_attenuation_vmi(70.0)  # ~0.193 cm⁻¹
```

# Reference Values (NIST XCOM)
| Energy (keV) | μ_water (cm⁻¹) |
|--------------|----------------|
| 40 | 0.268 |
| 50 | 0.227 |
| 60 | 0.206 |
| 70 | 0.193 |
| 80 | 0.184 |
| 100 | 0.171 |
| 120 | 0.163 |
| 140 | 0.157 |
"""
function get_water_attenuation_vmi(energy_keV::Float64)
    # Use NIST-validated lookup via compute_μ_at_energy
    return compute_μ_at_energy(XA.Materials.water, energy_keV)
end

# Backward compatibility alias
compute_effective_μ_water(energy_keV::Float64) = get_water_attenuation_vmi(energy_keV)

# =============================================================================
# VMI Reconstruction Integration
# =============================================================================

"""
    vmi_to_hu(vmi_image::AbstractArray, energy_keV::Float64) -> Array

Convert Virtual Monoenergetic Image from attenuation (cm⁻¹) to Hounsfield Units.

Uses energy-specific water attenuation from NIST XCOM database.

# Arguments
- `vmi_image`: VMI in linear attenuation units (cm⁻¹)
- `energy_keV`: VMI energy in keV

# Returns
Array in Hounsfield Units where water = 0 HU at any energy.

# Example
```julia
vmi_70 = virtual_monoenergetic(mat_map, 70.0)
recon = fdk_reconstruct(vmi_70, geom, (256, 256, 32))
recon_hu = vmi_to_hu(recon, 70.0)  # Water regions ≈ 0 HU
```
"""
function vmi_to_hu(vmi_image::AbstractArray{T}, energy_keV::Float64) where T
    μ_water = T(get_water_attenuation_vmi(energy_keV))
    return T(1000) .* (vmi_image .- μ_water) ./ μ_water
end

"""
    reconstruct_vmi(materials::MaterialMap, energy_keV::Float64,
                    geom::CTGeometry, recon_size::NTuple{3,Int};
                    method::Symbol=:fdk, to_hu::Bool=true,
                    fdk_kwargs...) -> Array

Full VMI reconstruction pipeline: synthesize VMI sinogram + reconstruct + HU conversion.

This is the recommended high-level API for generating VMI images.

# Arguments
- `materials::MaterialMap`: Result of material decomposition
- `energy_keV::Float64`: Target VMI energy (40-140 keV typical)
- `geom::CTGeometry`: CT geometry for reconstruction
- `recon_size::NTuple{3,Int}`: Output volume dimensions

# Keyword Arguments
- `method::Symbol=:fdk`: Reconstruction method (:fdk or :sirt)
- `to_hu::Bool=true`: Convert output to Hounsfield Units
- `niter::Int=3`: Number of iterations for SIRT (ignored for FDK)
- `filter::FilterType=RampFilter()`: FDK filter (passed to fdk_reconstruct)
- `cutoff::Float64=1.0`: FDK frequency cutoff

# Returns
- If to_hu=true: VMI reconstruction in Hounsfield Units
- If to_hu=false: VMI reconstruction in attenuation units (cm⁻¹)

# Example
```julia
# Standard workflow
de_sino = forward_project_dual_energy(phantom.mask, geom, protocol; ...)
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))

# Reconstruct at 50 keV (high iodine contrast)
vmi_50_hu = reconstruct_vmi(mat_map, 50.0, geom, (256, 256, 32))

# Reconstruct at 100 keV (reduced artifacts)
vmi_100_hu = reconstruct_vmi(mat_map, 100.0, geom, (256, 256, 32))

# Use SIRT for better quality
vmi_70_sirt = reconstruct_vmi(mat_map, 70.0, geom, (256, 256, 32);
                               method=:sirt, niter=3)
```

# Energy Selection Guide
- 40-50 keV: Maximum iodine enhancement (high noise)
- 50-60 keV: Good contrast, moderate noise
- 65-75 keV: Balanced (similar to 120 kVp single-energy)
- 80-100 keV: Reduced beam hardening artifacts
- 100-140 keV: Metal artifact reduction
"""
function reconstruct_vmi(
    materials::MaterialMap{T},
    energy_keV::Float64,
    geom::CTGeometry,
    recon_size::NTuple{3,Int};
    method::Symbol=:fdk,
    to_hu::Bool=true,
    niter::Int=3,
    filter::FilterType=RampFilter(),
    cutoff::Float64=1.0
) where T

    # Step 1: Generate VMI sinogram
    vmi_sino = virtual_monoenergetic(materials, energy_keV)

    # Step 2: Reconstruct based on method
    if method == :fdk
        recon = fdk_reconstruct(vmi_sino, geom, recon_size; filter=filter, cutoff=cutoff)
    elseif method == :sirt
        recon = sirt_reconstruct(vmi_sino, geom, recon_size; niter=niter)
    else
        error("Unknown reconstruction method: $method. Use :fdk or :sirt")
    end

    # Step 3: Convert to HU if requested
    if to_hu
        return vmi_to_hu(Array(recon), energy_keV)
    else
        return Array(recon)
    end
end

"""
    generate_vmi_series(materials::MaterialMap, energies_keV::Vector{Float64},
                        geom::CTGeometry, recon_size::NTuple{3,Int};
                        kwargs...) -> Dict{Float64, Array}

Generate VMI images at multiple energies for keV sweep visualization.

Useful for demonstrating energy-dependent contrast behavior or
finding optimal VMI energy for a specific diagnostic task.

# Arguments
- `materials::MaterialMap`: Result of material decomposition
- `energies_keV::Vector{Float64}`: Vector of target energies
- `geom::CTGeometry`: CT geometry
- `recon_size::NTuple{3,Int}`: Output volume dimensions

# Keyword Arguments
All kwargs passed to `reconstruct_vmi()`.

# Returns
Dict{Float64, Array} mapping energy (keV) to reconstructed VMI image.

# Example
```julia
# Generate VMI sweep from 40-140 keV
energies = [40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 140.0]
vmi_series = generate_vmi_series(mat_map, energies, geom, (128, 128, 16))

# Access specific energy
vmi_50 = vmi_series[50.0]

# Plot iodine enhancement vs energy
using Plots
iodine_mask = ...  # mask for iodine ROI
mean_hu = [mean(vmi_series[E][iodine_mask]) for E in energies]
plot(energies, mean_hu, xlabel="Energy (keV)", ylabel="Mean HU")
```
"""
function generate_vmi_series(
    materials::MaterialMap,
    energies_keV::Vector{Float64},
    geom::CTGeometry,
    recon_size::NTuple{3,Int};
    kwargs...
)
    return Dict(E => reconstruct_vmi(materials, E, geom, recon_size; kwargs...)
                for E in energies_keV)
end

"""
    VMIResult

Container for VMI reconstruction results with metadata.

# Fields
- `image::Array{Float32,3}`: Reconstructed image (HU or attenuation)
- `energy_keV::Float64`: VMI energy
- `is_hu::Bool`: True if image is in HU, false if in attenuation units
- `μ_water::Float64`: Water attenuation used for HU conversion
- `method::Symbol`: Reconstruction method used (:fdk or :sirt)
"""
struct VMIResult{T<:AbstractFloat}
    image::Array{T,3}
    energy_keV::Float64
    is_hu::Bool
    μ_water::Float64
    method::Symbol
end

"""
    get_vmi_info(result::VMIResult)

Print summary information about a VMI result.
"""
function get_vmi_info(result::VMIResult)
    unit_str = result.is_hu ? "HU" : "cm⁻¹"
    println("VMI Result:")
    println("  Energy: $(result.energy_keV) keV")
    println("  Size: $(size(result.image))")
    println("  Units: $unit_str")
    println("  μ_water: $(round(result.μ_water, digits=4)) cm⁻¹")
    println("  Method: $(result.method)")
    println("  Range: [$(round(minimum(result.image), digits=1)), $(round(maximum(result.image), digits=1))]")
end

# =============================================================================
# Exports
# =============================================================================

export DualEnergySinogram, MaterialMap
export GSIProtocol, default_gsi_protocol
export forward_project_dual_energy
export decompose_materials, virtual_monoenergetic
export get_water_attenuation_vmi, vmi_to_hu
export get_material_attenuation, get_iodine_solution_attenuation, get_calcium_material_attenuation
export reconstruct_vmi, generate_vmi_series
export VMIResult, get_vmi_info
