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
struct DualEnergySinogram{T<:AbstractFloat, A<:AbstractArray{T,3}}
    low::A
    high::A
    low_kvp::Int
    high_kvp::Int
    n_cols::Int
    n_rows::Int
    n_angles::Int
end

function DualEnergySinogram(low::A, high::A;
                            low_kvp::Int=80, high_kvp::Int=140) where {T<:AbstractFloat, A<:AbstractArray{T,3}}
    @assert size(low) == size(high) "Low and high sinograms must have same size"
    n_cols, n_rows, n_angles = size(low)
    return DualEnergySinogram{T,A}(low, high, low_kvp, high_kvp, n_cols, n_rows, n_angles)
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
struct MaterialMap{T<:AbstractFloat, A<:AbstractArray{T,3}}
    material1::A
    material2::A
    material1_name::Symbol
    material2_name::Symbol
    domain::Symbol  # :projection or :image
end

function MaterialMap(m1::A, m2::A;
                     material1_name::Symbol=:water,
                     material2_name::Symbol=:iodine,
                     domain::Symbol=:projection) where {T<:AbstractFloat, A<:AbstractArray{T,3}}
    @assert size(m1) == size(m2) "Material maps must have same size"
    return MaterialMap{T,A}(m1, m2, material1_name, material2_name, domain)
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
    mask::AbstractArray{<:Unsigned,3},
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

    # Setup physics configs for low and high kVp
    physics_low = physics
    physics_high = physics

    # Get Scanner geometry for energy-dependent scatter
    # geometry_aware_scatter_model requires a Scanner object, not a GeometrySpecification
    scanner_geom = if scanner isa Scanner
        scanner
    elseif scanner !== nothing && hasmethod(geometry, Tuple{typeof(scanner)})
        # AbstractScannerSpec - extract geometry and create a Scanner with those values
        geom_spec = geometry(scanner)
        det_spec = detector(scanner)
        Scanner(
            source_to_isocenter = geom_spec.sid_mm.value,
            source_to_detector = geom_spec.sdd_mm.value,
            detector_cols = det_spec.n_cols.value,
            detector_rows = det_spec.n_rows.value,
            detector_col_size = det_spec.col_size_mm.value,
            detector_row_size = det_spec.row_size_mm.value
        )
    else
        nothing
    end

    # Initialize flag for joint scatter correction (applied after both sinograms generated)
    apply_joint_scatter_correction = false

    if physics !== nothing
        # Create ENERGY-DEPENDENT scatter models if scatter is enabled AND we have scanner geometry
        # See SCATTER-ENERGY-RESEARCH: Lower kVp (80) has higher SPR than higher kVp (140)
        # Using mean_energy_keV ensures scatter coefficient scales appropriately for each energy
        scatter_low = physics.scatter
        scatter_correction_low = physics.scatter_correction
        scatter_high = physics.scatter
        scatter_correction_high = physics.scatter_correction

        if scanner_geom !== nothing && physics.scatter !== nothing
            phantom_diameter_cm = nothing  # Will use reference size if not estimable

            scatter_low = geometry_aware_scatter_model(scanner_geom;
                scale_factor=physics.scatter.scale_factor,
                kernel_type=physics.scatter.kernel_type,
                phantom_diameter_cm=phantom_diameter_cm,
                mean_energy_keV=mean_e_low)  # Energy-dependent scaling for 80 kVp
            scatter_high = geometry_aware_scatter_model(scanner_geom;
                scale_factor=physics.scatter.scale_factor,
                kernel_type=physics.scatter.kernel_type,
                phantom_diameter_cm=phantom_diameter_cm,
                mean_energy_keV=mean_e_high)  # Energy-dependent scaling for 140 kVp
        end

        if scanner_geom !== nothing && physics.scatter_correction !== nothing
            # Use JOINT scatter correction instead of per-sinogram correction
            #
            # Per-sinogram scatter correction causes wave artifacts in material
            # decomposition because:
            # 1. 80 kVp and 140 kVp have different SPR (up to 5x difference per PMC3097788)
            # 2. Different energy-dependent coefficients → different estimation residuals
            # 3. Material decomposition takes a weighted DIFFERENCE of sinograms
            # 4. Different residual errors get AMPLIFIED in the decomposition
            #
            # Joint scatter correction uses a single scatter estimate for BOTH sinograms:
            # - Combines sinograms to create average signal
            # - Estimates scatter from combined signal at average energy
            # - Applies SAME scatter estimate to BOTH sinograms
            # - Results in identical residual patterns that DON'T amplify in decomposition
            #
            # See DE-CROSS-SCATTER-RESEARCH in progress.md for full analysis with citations.
            @info "Using joint scatter correction for dual-energy (avoids decomposition artifacts)" maxlog=1

            # Disable per-sinogram scatter correction (will use joint correction instead)
            scatter_correction_low = nothing
            scatter_correction_high = nothing

            # Flag to apply joint correction after both sinograms are generated
            apply_joint_scatter_correction = true
        end

        # Use noise from physics config directly
        noise_low = physics.noise
        noise_high = physics.noise

        # Create PhysicsConfig with correct field order matching struct definition
        # Scatter ADDITION is energy-dependent, but scatter CORRECTION is disabled
        # (correction causes wave artifacts in decomposition - see DE-SCATTER-RESEARCH)
        physics_low = PhysicsConfig(
            physics.fill_factor,        # 1. fill_factor
            physics.flat_filter,        # 2. flat_filter
            physics.bowtie_filter,      # 3. bowtie_filter
            scatter_low,                # 4. scatter (energy-dependent for 80 kVp)
            scatter_correction_low,     # 5. scatter_correction (=nothing, disabled for DE)
            physics.crosstalk,          # 6. crosstalk
            physics.optical_crosstalk,  # 7. optical_crosstalk
            physics.focal_spot,         # 8. focal_spot
            physics.detector_efficiency,# 9. detector_efficiency
            noise_low,                  # 10. noise
            physics.lag,                # 11. lag
            physics.noise_seed,         # 12. noise_seed
            mean_e_low,                 # 13. energy_keV
            physics.heel_effect,        # 14. heel_effect
            physics.das_model,          # 15. das_model
            physics.bhc                 # 16. bhc
        )
        physics_high = PhysicsConfig(
            physics.fill_factor,        # 1. fill_factor
            physics.flat_filter,        # 2. flat_filter
            physics.bowtie_filter,      # 3. bowtie_filter
            scatter_high,               # 4. scatter (energy-dependent for 140 kVp)
            scatter_correction_high,    # 5. scatter_correction (=nothing, disabled for DE)
            physics.crosstalk,          # 6. crosstalk
            physics.optical_crosstalk,  # 7. optical_crosstalk
            physics.focal_spot,         # 8. focal_spot
            physics.detector_efficiency,# 9. detector_efficiency
            noise_high,                 # 10. noise
            physics.lag,                # 11. lag
            physics.noise_seed,         # 12. noise_seed
            mean_e_high,                # 13. energy_keV
            physics.heel_effect,        # 14. heel_effect
            physics.das_model,          # 15. das_model
            physics.bhc                 # 16. bhc
        )
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

    # Apply joint scatter correction if enabled
    # This uses a single scatter estimate for BOTH sinograms to ensure identical
    # residual patterns that don't amplify in material decomposition
    if physics !== nothing && apply_joint_scatter_correction && scanner_geom !== nothing
        correct_scatter_dual_energy!(sino_low, sino_high, scanner_geom;
            mean_energy_low_keV = mean_e_low,
            mean_energy_high_keV = mean_e_high
        )
    end

    # Keep arrays on same device as input (GPU-native)
    # No forced CPU transfer - preserves GPU arrays if input was GPU
    return DualEnergySinogram(sino_low, sino_high;
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
function decompose_materials(sino::DualEnergySinogram{T,A};
                             basis::Tuple{Symbol,Symbol}=(:water, :iodine),
                             method::Symbol=:polynomial,
                             ws_material1=nothing,
                             ws_material2=nothing,
                             ws_inv_a11=nothing,
                             ws_inv_a12=nothing,
                             ws_inv_a21=nothing,
                             ws_inv_a22=nothing) where {T, A}

    if method != :polynomial
        error("Only :polynomial method currently supported")
    end

    # Use pre-computed inverse matrix elements if provided (zero-alloc path)
    if ws_inv_a11 !== nothing
        inv_a11 = T(ws_inv_a11)
        inv_a12 = T(ws_inv_a12)
        inv_a21 = T(ws_inv_a21)
        inv_a22 = T(ws_inv_a22)
    else
        e_eff_low = get_effective_energy(sino.low_kvp)
        e_eff_high = get_effective_energy(sino.high_kvp)
        inv_a11, inv_a12, inv_a21, inv_a22 = compute_decomposition_matrix(basis, e_eff_low, e_eff_high; T=T)
    end

    # Use pre-allocated output buffers if provided (zero-alloc path)
    material1 = ws_material1 === nothing ? similar(sino.low) : ws_material1
    material2 = ws_material2 === nothing ? similar(sino.low) : ws_material2

    m1_name, m2_name = basis

    spectral_decompose!(material1, material2, sino.low, sino.high,
                        inv_a11, inv_a12, inv_a21, inv_a22)

    return MaterialMap(material1, material2;
                       material1_name=m1_name, material2_name=m2_name,
                       domain=:projection)
end

"""
    virtual_monoenergetic(materials::MaterialMap, energy_keV::Float64; ws_output=nothing) -> Array

Generate virtual monoenergetic image (VMI) at specified energy.

Delegates to `spectral_vmi!()` from the unified VMI pipeline.

# Arguments
- `materials::MaterialMap`: Result of material decomposition
- `energy_keV::Float64`: Target energy in keV (40-140)
"""
function virtual_monoenergetic(materials::MaterialMap{T,A}, energy_keV::Float64;
                               ws_output=nothing) where {T, A}
    if energy_keV < 10.0 || energy_keV > 150.0
        error("Energy must be between 10 and 150 keV (got $energy_keV)")
    end

    μ1 = T(get_basis_mu(materials.material1_name, energy_keV))
    μ2 = T(get_basis_mu(materials.material2_name, energy_keV))

    vmi = ws_output === nothing ? similar(materials.material1) : ws_output
    spectral_vmi!(vmi, materials.material1, materials.material2, μ1, μ2)
    return vmi
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    get_effective_energy(kvp::Int) -> Float64

Get spectrum-weighted mean effective energy for a given kVp setting.

These values are computed from the actual spectra used in forward projection
to ensure consistency between projection and material decomposition.
"""
function get_effective_energy(kvp::Int)
    # Compute mean energy from actual spectrum for accurate decomposition
    # These match the spectra used in forward_project_dual_energy()
    energies, weights = load_spectrum(kvp)
    return sum(energies .* weights) / sum(weights)
end

"""
    get_material_attenuation(material::Symbol, energy_keV::Float64) -> Float64

Deprecated: use `get_basis_mu()` from the unified VMI pipeline instead.
Kept as thin wrapper for backward compatibility.
"""
get_material_attenuation(material::Symbol, energy_keV::Float64) = get_basis_mu(material, energy_keV)

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

"""
    reconstruct_vmi(materials::MaterialMap, energy_keV, geom, recon_size; kwargs...) -> Array

DEPRECATED — projection-domain VMI pipeline. Use image-domain VMI instead:

    # Image-domain VMI (Wu et al. 2009 — the actual GE GSI algorithm):
    # 1. Reconstruct material sinograms → density images m₁, m₂
    # 2. image_domain_vmi!(output, m₁, m₂, compute_vmi_g(E, basis[2]))
    # 3. image_domain_vmi_to_hu!(hu, output)

See: compute_vmi_g, image_domain_vmi!, image_domain_vmi_to_hu!,
     noise_compensated_vmi!, extract_noise_map
"""
function reconstruct_vmi(
    materials::MaterialMap{T,A},
    energy_keV::Float64,
    geom::CTGeometry,
    recon_size::NTuple{3,Int};
    method::Symbol=:fdk,
    to_hu::Bool=true,
    water_mask=nothing,
    niter::Int=3,
    filter::FilterType=StandardFilter(),
    cutoff::Float64=1.0
) where {T, A}

    vmi_sino = virtual_monoenergetic(materials, energy_keV)

    if method == :fdk
        recon = fdk_reconstruct(vmi_sino, geom, recon_size; filter=filter, cutoff=cutoff)
    elseif method == :sirt
        recon = sirt_reconstruct(vmi_sino, geom, recon_size; niter=niter)
    else
        error("Unknown reconstruction method: $method. Use :fdk or :sirt")
    end

    if to_hu
        recon_cpu = Array(recon)
        if water_mask !== nothing
            μ_w = mean(recon_cpu[water_mask])
            return vmi_to_hu(recon_cpu, energy_keV; μ_water=μ_w)
        else
            return vmi_to_hu(recon_cpu, energy_keV)
        end
    else
        return Array(recon)
    end
end


# =============================================================================
# Exports
# =============================================================================

export DualEnergySinogram, MaterialMap
export GSIProtocol, default_gsi_protocol
export forward_project_dual_energy
export decompose_materials, virtual_monoenergetic
export get_water_attenuation_vmi
export get_material_attenuation, get_iodine_solution_attenuation, get_calcium_material_attenuation
export reconstruct_vmi
