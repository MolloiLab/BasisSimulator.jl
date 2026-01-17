# =============================================================================
# Polychromatic Forward Projection with CatSim Signal Chain
# =============================================================================
#
# Memory-efficient polychromatic CT simulation using Beer-Lambert physics
# with optional CatSim-exact signal chain built-in.
#
# NOT from TIGRE - TIGRE is monochromatic only.
# This implements proper spectral physics: I = Σ wₑ × exp(-∫μₑ dl)
#
# Algorithm Overview
# ------------------
# Polychromatic projection models the energy-dependent attenuation of X-rays
# as they traverse a heterogeneous medium. Unlike monochromatic projection
# (single energy), polychromatic simulation accounts for the broad energy
# spectrum produced by an X-ray tube and the energy-dependence of material
# attenuation coefficients.
#
# Mathematical Foundation: Beer-Lambert Law
# -----------------------------------------
# For a monochromatic beam of intensity I₀ at energy E traversing a path:
#
#     I(E) = I₀(E) × exp(-∫ μ(E, l) dl)
#
# where μ(E, l) is the linear attenuation coefficient (mm⁻¹) at energy E and
# position l along the ray path.
#
# For a polychromatic beam with spectral distribution w(E) (photon fluence
# weights normalized such that Σ w(E) = 1), the transmitted intensity is:
#
#     I_total = ∫ w(E) × I₀(E) × exp(-∫ μ(E, l) dl) dE
#
# In discretized form with N energy bins:
#
#     I_total = Σₑ wₑ × exp(-∫ μₑ(l) dl) = Σₑ wₑ × exp(-Lₑ)
#
# where wₑ is the normalized weight for energy bin e, and Lₑ is the line
# integral of attenuation at energy e computed via ray tracing.
#
# The measured projection value is converted back to an "effective" line
# integral via the log transform:
#
#     p = -log(I_total / I₀) = -log(Σₑ wₑ × exp(-Lₑ))
#
# Spectral Integration Method
# ---------------------------
# This implementation uses an energy-sequential approach for memory efficiency:
#
# 1. For each energy bin e in spectrum:
#    a. Create μ-volume: μᵢⱼₖ(e) = μ(material[i,j,k], E_e) for all voxels
#    b. Forward project: Lₑ = Siddon ray-trace through μ-volume
#    c. Accumulate: I_total += wₑ × exp(-Lₑ)
#
# 2. After all energies: p = -log(I_total)
#
# This approach requires O(1) additional memory per energy bin (one μ-volume),
# versus O(N) memory for storing all energy-specific projections simultaneously.
#
# Beam Hardening
# --------------
# The nonlinear log-sum-exp operation causes beam hardening artifacts:
# - p_poly ≠ Σₑ wₑ × pₑ (projection is NOT a linear combination)
# - Low-energy photons are preferentially absorbed, "hardening" the beam
# - Results in cupping artifacts in reconstructed images
# - Corrected via polynomial BHC (beam hardening correction) post-projection
#
# CatSim Signal Chain (when enabled via kwargs):
# 1. Polychromatic forward projection (Beer-Lambert)
# 2. Physics pipeline (scatter, crosstalk, focal spot, noise, lag)
# 3. Heel effect (intensity domain)
# 4. DAS model (gain + electronic noise)
# 5. Air scan calibration (CatSim-exact: noise-free air reference)
# 6. Low signal correction (replace negatives with smoothed neighbors)
# 7. Log transform
# 8. Beam hardening correction
#
# Physical Units
# --------------
# - Energy bins: keV
# - Spectral weights: normalized photon fluence (dimensionless, sum to 1)
# - Attenuation coefficients μ: mm⁻¹
# - Line integrals: dimensionless (mm⁻¹ × mm)
# - Intensity: relative transmitted photon counts (dimensionless)
#
# GPU Compatibility (via AcceleratedKernels.jl)
# ---------------------------------------------
# - ✅ Metal (Apple Silicon) - Primary development platform
# - ✅ CUDA (NVIDIA GPUs)
# - ✅ ROCm (AMD GPUs)
# - ✅ Intel oneAPI
# - ✅ CPU fallback (multi-threaded)
#
# The algorithm auto-detects backend from array type. Pass MtlArray (Metal),
# CuArray (CUDA), ROCArray (ROCm), or regular Array (CPU).
#
# References
# ----------
# 1. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent
#    Advances." 3rd ed. SPIE Press; 2015. Chapter 3: X-ray Production and
#    Interactions. doi:10.1117/3.2197756
#
# 2. Buzug TM. "Computed Tomography: From Photon Statistics to Modern Cone-Beam
#    CT." Springer; 2008. Section 5.3: Polychromatic Effects and Beam Hardening.
#    doi:10.1007/978-3-540-39408-2
#
# 3. De Man B, Nuyts J, Dupont P, Marchal G, Suetens P. "An iterative maximum-
#    likelihood polychromatic algorithm for CT." IEEE Trans Med Imaging.
#    2001;20(10):999-1008. doi:10.1109/42.959297
#
# 4. Joseph PM, Spital RD. "A method for correcting bone induced artifacts in
#    computed tomography scanners." J Comput Assist Tomogr. 1978;2(1):100-108.
#    doi:10.1097/00004728-197801000-00017
#
# 5. Hubbell JH, Seltzer SM. "Tables of X-Ray Mass Attenuation Coefficients and
#    Mass Energy-Absorption Coefficients." NIST Standard Reference Database 126.
#    https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
#
# Uses AcceleratedKernels.jl for GPU/CPU acceleration.
#
# =============================================================================

import AcceleratedKernels as AK

export forward_project!, forward_project

# =============================================================================
# Create μ Volume from Material Mask
# =============================================================================

"""
    create_μ_volume!(μ_volume, mask, materials, energy_keV) -> μ_volume

Create a 3D attenuation coefficient volume for a single energy from a material
mask. This is a helper function for polychromatic projection that maps material
indices to energy-specific attenuation coefficients.

# Algorithm

For each voxel in the volume:
1. Read material index m from mask (0-indexed UInt8)
2. Look up material composition: composition = materials[m + 1]
3. Compute attenuation: μᵢⱼₖ = μ(composition, E) using XrayAttenuation.jl

The attenuation lookup uses NIST XCOM cross-section data interpolated to the
specified energy via `compute_μ_at_energy()`.

# Arguments

- `μ_volume::AbstractArray{T,3}`: Output attenuation volume of size `[nx, ny, nz]`,
  modified in place. After execution, contains linear attenuation coefficients
  μ in mm⁻¹ for each voxel at the specified energy.

- `mask::AbstractArray{UInt8,3}`: Input material index volume of size `[nx, ny, nz]`.
  Values are 0-indexed region indices (0 = region 1, 1 = region 2, etc.).
  Typically created by phantom generators with material segmentation.

- `materials::Vector`: Vector of material definitions indexed by region.
  Each material should be compatible with `compute_μ_at_energy(material, E)`.
  Typically from `get_region_materials()` which returns Gammex 472 or custom
  tissue compositions.

- `energy_keV::Real`: X-ray photon energy in keV for attenuation lookup.
  Valid range: typically 10-150 keV for diagnostic CT.

# Returns

- `μ_volume::AbstractArray{T,3}`: The modified input array (same reference)

# GPU Compatibility

Uses `AK.foreachindex` for parallel execution on any backend:

| Array Type | Backend | Notes |
|------------|---------|-------|
| `Array` | CPU | Multi-threaded via Julia |
| `MtlArray` | Metal | Apple Silicon GPU |
| `CuArray` | CUDA | NVIDIA GPU |
| `ROCArray` | ROCm | AMD GPU |

# Example

```julia
using BasisSimulator

# Create a simple 2-material phantom mask
mask = zeros(UInt8, 128, 128, 32)
mask[40:90, 40:90, :] .= 1  # Water region
# Region 0 = air (background), Region 1 = water

# Get material definitions
materials = get_region_materials()  # Returns [air, water, ...]

# Create μ-volume at 60 keV
μ_volume = zeros(Float32, 128, 128, 32)
create_μ_volume!(μ_volume, mask, materials, 60.0)

# Result: μ_volume contains ~0.0 for air, ~0.02 mm⁻¹ for water at 60 keV
println("Water μ at 60 keV: ", μ_volume[65, 65, 16], " mm⁻¹")
```

# Notes

- The lookup table is computed on CPU then transferred to GPU for efficiency
- UInt8 mask values are converted to 1-based Julia indices internally
- Attenuation values are linearly interpolated from NIST XCOM data
- For very large volumes, this is the memory bottleneck (one full μ-volume)

# See Also

- [`forward_project!`](@ref): High-level projection interface
- [`get_region_materials`](@ref): Standard phantom material definitions
- [`compute_μ_at_energy`](@ref): Single-material attenuation lookup
"""
function create_μ_volume!(
    μ_volume::AbstractArray{T, 3},
    mask::AbstractArray{UInt8, 3},
    materials::Vector,
    energy_keV::Real
) where T <: AbstractFloat

    # Pre-compute μ for all regions at this energy (on CPU)
    n_regions = length(materials)
    μ_at_energy_cpu = Vector{T}(undef, n_regions)
    for i in 1:n_regions
        μ_at_energy_cpu[i] = T(compute_μ_at_energy(materials[i], Float64(energy_keV)))
    end

    # Transfer lookup table to same device as mask (GPU or CPU)
    μ_at_energy = similar(mask, T, n_regions)
    copyto!(μ_at_energy, μ_at_energy_cpu)

    # Use AcceleratedKernels.jl for parallel execution
    AK.foreachindex(mask) do idx
        region_idx = mask[idx] + 1  # Convert 0-based region to 1-based array index
        μ_volume[idx] = μ_at_energy[region_idx]
    end

    return μ_volume
end

# =============================================================================
# Unified Forward Projection API with CatSim Signal Chain
# =============================================================================

"""
    forward_project!(sinogram, volume_or_mask, geom; kwargs...) -> sinogram

Unified forward projection with monochromatic or polychromatic X-ray physics,
optional detector effects, and CatSim-exact calibration signal chain.

This is the primary interface for CT simulation in BasisSimulator.jl, supporting
three modes of operation:
1. **Monochromatic**: Single effective energy (fast, simple)
2. **Polychromatic**: Full spectral simulation (accurate, slower)
3. **Full Signal Chain**: CatSim-exact clinical simulation pipeline

# Algorithm

## Polychromatic Projection (Beer-Lambert Law)

For a polychromatic X-ray beam with N energy bins, the transmitted intensity is:

    I_total = Σₑ wₑ × exp(-Lₑ)

where:
- wₑ = normalized spectral weight for energy bin e (Σ wₑ = 1)
- Lₑ = ∫ μₑ(l) dl = line integral of attenuation at energy e

The projection value is the negative log of transmitted intensity:

    p = -log(I_total) = -log(Σₑ wₑ × exp(-Lₑ))

This nonlinear operation produces beam hardening, where low-energy photons are
preferentially absorbed, shifting the effective energy along the ray path.

## Spectral Integration Method

Uses energy-sequential integration for memory efficiency:

```
for e in 1:N_energies:
    μ_volume[i,j,k] = μ(material[i,j,k], E_e)     # Create energy-specific volume
    L_e = siddon_ray_trace(μ_volume)               # Ray trace at this energy
    I_total += w_e × exp(-L_e)                     # Accumulate Beer-Lambert
end
sinogram = -log(I_total)                           # Convert to projection
```

Memory: O(volume_size) additional storage (one μ-volume), independent of N_energies.

## CatSim Signal Chain

When signal chain parameters are provided, applies clinical CT pipeline:

1. **Polychromatic FP**: Beer-Lambert projection as above
2. **Physics Pipeline**: scatter, crosstalk, focal spot, detector lag
3. **Intensity Domain**: exp(-sinogram) to get transmitted photon counts
4. **Heel Effect**: anode self-attenuation intensity gradient
5. **DAS Model**: gain and electronic noise (phantom only)
6. **Air Calibration**: ratio to noise-free air scan (CatSim-exact)
7. **Low Signal Correction**: smooth replacement of negative values
8. **Log Transform**: -log(calibrated_intensity)
9. **BHC**: polynomial beam hardening correction

# Arguments

- `sinogram::AbstractArray{T,3}`: Output sinogram of size `[n_cols, n_rows, n_angles]`,
  modified in place. Contains line integrals (projection domain) after execution.
  - `n_cols`: Detector columns (transaxial direction)
  - `n_rows`: Detector rows (axial direction, typically 1 for 2D, >1 for cone-beam)
  - `n_angles`: Number of projection angles

- `volume_or_mask::AbstractArray`: Input volume, either:
  - `AbstractArray{T,3}` where `T <: AbstractFloat`: Pre-computed attenuation volume
    μ in mm⁻¹. Used directly for monochromatic projection.
  - `AbstractArray{UInt8,3}`: Material index mask. Combined with `materials` and
    `energies`/`weights` for polychromatic projection.

- `geom::CTGeometry`: Scanner geometry containing source positions, detector
  geometry, and field of view. See `CTGeometry` for required fields.

# Keyword Arguments

## Spectrum Parameters (required for mask input)

- `energy::Union{Nothing,Real}=nothing`: Single energy in keV for monochromatic
  projection from mask input. Mutually exclusive with `energies`/`weights`.

- `energies::Union{Nothing,Vector}=nothing`: Vector of energy bin centers in keV.
  Typically 10-60 bins spanning 20-140 keV. From `load_spectrum()` or custom.

- `weights::Union{Nothing,Vector}=nothing`: Vector of spectral weights (photon
  fluence). Will be normalized internally (sum to 1). Paired with `energies`.

- `materials::Union{Nothing,Vector}=nothing`: Vector of material definitions for
  each region in the mask. From `get_region_materials()`. Required when using
  mask input.

## Physics Effects

- `physics::Union{Nothing,PhysicsConfig}=nothing`: Physics configuration from:
  - `realistic_physics_config()`: Common clinical effects
  - `minimal_physics_config()`: Noise only
  - `full_physics_config()`: All 13 effects
  - `default_physics_config(...)`: Custom via kwargs

## Signal Chain Parameters

- `heel_effect::Union{Nothing,HeelEffect}=nothing`: Anode heel effect model.
  From `default_heel_effect(anode_angle_deg=7.0)`.

- `das_model::Union{Nothing,DASModel}=nothing`: Data acquisition system model
  with gain and electronic noise. From `default_das_model(gain=1.0, ...)`.

- `bhc::Union{Nothing,Union{BHCPolynomial,BeamHardeningCorrection}}=nothing`:
  Beam hardening correction polynomial. From `bhc_water_default()`.

- `calibrate::Bool=true`: Enable full CatSim calibration when signal chain
  parameters are provided. Set to `false` for raw projections.

- `max_prep::Union{Nothing,Real}=nothing`: Maximum projection value for clamping.
  Prevents extreme values from saturation/negative values.

- `noise_seed::Union{Nothing,Int}=nothing`: Random seed for reproducible noise.

# Returns

- `sinogram::AbstractArray{T,3}`: The modified sinogram array (same reference)

# GPU Compatibility

Backend auto-detected from array type via AcceleratedKernels.jl:

| Array Type | Backend | Speedup vs CPU |
|------------|---------|----------------|
| `Array` | CPU (multi-threaded) | 1× (baseline) |
| `MtlArray` | Metal (Apple Silicon) | ~50-100× |
| `CuArray` | CUDA (NVIDIA) | ~50-100× |
| `ROCArray` | ROCm (AMD) | ~50-100× |

# Examples

## Simple Monochromatic Projection

```julia
using BasisSimulator

# Create geometry and phantom
scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=360, fov=(350.0, 350.0, 40.0))
phantom = create_water_cylinder(128, diameter_mm=200.0)

# Direct μ-volume projection (fastest)
sinogram = similar(phantom.μ, Float32, geom.n_cols, geom.n_rows, geom.n_angles)
fill!(sinogram, 0f0)
forward_project!(sinogram, phantom.μ, geom)
```

## Polychromatic Projection

```julia
# Load 120 kVp spectrum and materials
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)  # 30 bins
materials = get_region_materials()

# Polychromatic projection (includes beam hardening)
forward_project!(sinogram, phantom.mask, geom;
    energies=energies,
    weights=weights,
    materials=materials
)
```

## Full Clinical Simulation (CatSim Signal Chain)

```julia
using Metal  # GPU acceleration

# GPU arrays
mask_gpu = MtlArray(phantom.mask)
sinogram_gpu = MtlArray(zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles))

# Full physics + signal chain
forward_project!(sinogram_gpu, mask_gpu, geom;
    energies=energies,
    weights=weights,
    materials=materials,
    physics=full_physics_config(energy_keV=65.0, noise_seed=42),
    heel_effect=default_heel_effect(anode_angle_deg=7.0),
    das_model=default_das_model(gain=1.0, electronic_noise_sigma=100.0),
    bhc=bhc_water_default(reference_energy_keV=65.0)
)

# sinogram_gpu now contains calibrated, BHC-corrected projections
```

# Performance Notes

- **Memory**: O(volume_size) working memory for μ-volume at each energy
- **Time complexity**: O(N_energies × N_rays × N_voxels_per_ray)
- **Bottleneck**: Ray tracing (GPU) or memory bandwidth (CPU)
- For repeated projections, prefer in-place version to avoid allocations

# References

1. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent
   Advances." 3rd ed. SPIE Press; 2015. doi:10.1117/3.2197756

2. Buzug TM. "Computed Tomography: From Photon Statistics to Modern Cone-Beam
   CT." Springer; 2008. doi:10.1007/978-3-540-39408-2

3. De Man B, et al. "An iterative maximum-likelihood polychromatic algorithm
   for CT." IEEE Trans Med Imaging. 2001;20(10):999-1008.
   doi:10.1109/42.959297

4. Hubbell JH, Seltzer SM. "Tables of X-Ray Mass Attenuation Coefficients."
   NIST Standard Reference Database 126.

# See Also

- [`forward_project`](@ref): Allocating version (creates sinogram)
- [`siddon_forward_project!`](@ref): Low-level ray tracing
- [`fdk_reconstruct`](@ref): Filtered backprojection reconstruction
- [`PhysicsConfig`](@ref): Physics effects configuration
"""
function forward_project!(
    sinogram::AbstractArray{T, 3},
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    # Spectrum parameters
    energy::Union{Nothing, Real} = nothing,
    energies::Union{Nothing, Vector} = nothing,
    weights::Union{Nothing, Vector} = nothing,
    materials::Union{Nothing, Vector} = nothing,
    # Physics pipeline
    physics::Union{Nothing, PhysicsConfig} = nothing,
    # CatSim signal chain (can override PhysicsConfig values)
    heel_effect::Union{Nothing, HeelEffect} = nothing,
    das_model::Union{Nothing, DASModel} = nothing,
    bhc::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}} = nothing,
    calibrate::Bool = true,
    max_prep::Union{Nothing, Real} = nothing,
    noise_seed::Union{Nothing, Int} = nothing
) where T <: AbstractFloat

    # Get signal chain effects from PhysicsConfig if not provided as kwargs
    # This allows full_physics_config() to include everything
    effective_heel = heel_effect
    effective_das = das_model
    effective_bhc = bhc
    effective_seed = noise_seed

    if physics !== nothing
        if effective_heel === nothing && physics.heel_effect !== nothing
            effective_heel = physics.heel_effect
        end
        if effective_das === nothing && physics.das_model !== nothing
            effective_das = physics.das_model
        end
        if effective_bhc === nothing && physics.bhc !== nothing
            effective_bhc = physics.bhc
        end
        if effective_seed === nothing && physics.noise_seed !== nothing
            effective_seed = physics.noise_seed
        end
    end

    # Check if CatSim signal chain is requested (from kwargs OR PhysicsConfig)
    has_signal_chain = effective_heel !== nothing || effective_das !== nothing || effective_bhc !== nothing

    if has_signal_chain && calibrate
        # === FULL CatSim SIGNAL CHAIN ===
        return _forward_project_with_signal_chain!(
            sinogram, volume_or_mask, geom;
            energy=energy, energies=energies, weights=weights, materials=materials,
            physics=physics, heel_effect=effective_heel, das_model=effective_das,
            bhc=effective_bhc, max_prep=max_prep, noise_seed=effective_seed
        )
    end

    # === STANDARD PROJECTION (no signal chain) ===

    # Determine mode based on input type and kwargs
    if eltype(volume_or_mask) <: AbstractFloat
        # Direct volume input - simple monochromatic projection
        siddon_forward_project!(sinogram, volume_or_mask, geom)

    elseif eltype(volume_or_mask) == UInt8
        # Mask input - need energy specification
        mask = volume_or_mask

        if materials === nothing
            error("materials must be provided when using a mask input")
        end

        if energy !== nothing
            # Monochromatic mode with single energy
            _forward_project_mono!(sinogram, mask, geom, T(energy), materials)

        elseif energies !== nothing && weights !== nothing
            # Polychromatic mode
            _forward_project_poly!(sinogram, mask, geom, energies, weights, materials)

        else
            error("Must specify either `energy` (single keV) or `energies` + `weights` (spectrum)")
        end
    else
        error("volume_or_mask must be Float32/Float64 (μ volume) or UInt8 (material mask)")
    end

    # Apply physics effects if specified (but not signal chain)
    if physics !== nothing
        apply_physics_effects!(sinogram, geom, physics)
    end

    # Apply BHC separately if signal chain not used but BHC provided
    if effective_bhc !== nothing && !has_signal_chain
        apply_bhc!(sinogram, effective_bhc)
    end

    return sinogram
end

"""
    forward_project(volume_or_mask, geom; kwargs...) -> sinogram

Compute forward projection and return a newly allocated sinogram array.
This is the allocating version of [`forward_project!`](@ref).

# Algorithm

Implements polychromatic X-ray projection via Beer-Lambert law:

    I_total = Σₑ wₑ × exp(-Lₑ)
    p = -log(I_total)

where wₑ are spectral weights and Lₑ are energy-specific line integrals.
See [`forward_project!`](@ref) for detailed algorithm description.

# Arguments

- `volume_or_mask::AbstractArray`: Input volume, either:
  - `AbstractArray{T,3}` where `T <: AbstractFloat`: Pre-computed μ-volume (mm⁻¹)
  - `AbstractArray{UInt8,3}`: Material mask for polychromatic projection

- `geom::CTGeometry`: Scanner geometry (source positions, detector, FOV)

# Keyword Arguments

See [`forward_project!`](@ref) for complete list. Key parameters:
- `energies`, `weights`, `materials`: For polychromatic mode
- `physics`: PhysicsConfig for detector effects
- `heel_effect`, `das_model`, `bhc`: CatSim signal chain

# Returns

- `sinogram::AbstractArray{T,3}`: Newly allocated sinogram of size
  `[n_cols, n_rows, n_angles]`. The array is allocated on the same device
  as `volume_or_mask` (CPU or GPU).

# GPU Compatibility

The returned sinogram is allocated on the same device as input:

```julia
# CPU
sinogram = forward_project(Array(phantom.μ), geom)      # returns Array

# Metal (Apple Silicon)
sinogram = forward_project(MtlArray(phantom.μ), geom)   # returns MtlArray

# CUDA (NVIDIA)
sinogram = forward_project(CuArray(phantom.μ), geom)    # returns CuArray
```

# Examples

## Simple Monochromatic Projection

```julia
using BasisSimulator

scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=180, fov=(300.0, 300.0, 32.0))

# Create uniform water phantom (μ ≈ 0.02 mm⁻¹ at 60 keV)
phantom_μ = fill(0.02f0, 128, 128, 32)

# Forward projection - sinogram auto-allocated
sinogram = forward_project(phantom_μ, geom)
println("Sinogram size: ", size(sinogram))  # (n_cols, n_rows, 180)
```

## Polychromatic with GPU

```julia
using Metal

# Load spectrum and materials
energies, weights = load_spectrum(120)
materials = get_region_materials()

# GPU phantom mask
mask_gpu = MtlArray(phantom.mask)

# Polychromatic projection on GPU
sinogram_gpu = forward_project(mask_gpu, geom;
    energies=energies,
    weights=weights,
    materials=materials
)
```

## Full Clinical Pipeline

```julia
# Complete CatSim-exact simulation
sinogram = forward_project(phantom.mask, geom;
    energies=energies,
    weights=weights,
    materials=materials,
    physics=full_physics_config(energy_keV=65.0),
    heel_effect=default_heel_effect(anode_angle_deg=7.0),
    das_model=default_das_model(gain=1.0, electronic_noise_sigma=100.0),
    bhc=bhc_water_default()
)
```

# Performance Notes

For iterative algorithms or repeated projections, prefer [`forward_project!`](@ref)
to avoid allocation overhead. This function allocates O(n_cols × n_rows × n_angles)
elements for the output sinogram.

# References

1. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent
   Advances." 3rd ed. SPIE Press; 2015. doi:10.1117/3.2197756

2. Buzug TM. "Computed Tomography: From Photon Statistics to Modern Cone-Beam
   CT." Springer; 2008. doi:10.1007/978-3-540-39408-2

# See Also

- [`forward_project!`](@ref): In-place version (avoids allocation)
- [`siddon_forward_project`](@ref): Low-level allocating ray tracing
- [`fdk_reconstruct`](@ref): Filtered backprojection reconstruction
"""
function forward_project(
    volume_or_mask::AbstractArray{T},
    geom::CTGeometry;
    kwargs...
) where T
    # Determine element type for output
    out_type = T <: AbstractFloat ? T : Float32

    # Create sinogram on same device as input
    sinogram = similar(volume_or_mask, out_type, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, zero(out_type))

    return forward_project!(sinogram, volume_or_mask, geom; kwargs...)
end

# =============================================================================
# Internal: Full CatSim Signal Chain Implementation
# =============================================================================

"""
Internal function implementing full CatSim-exact signal chain.

Pipeline:
1. Polychromatic forward projection (Beer-Lambert) -> returns INTENSITY
2. Apply physics pipeline in sinogram domain (scatter, noise, etc.)
3. Convert to intensity domain
4. Apply heel effect
5. Apply DAS model (gain + noise to phantom only)
6. Create noise-free air scan (CatSim-exact)
7. Calibrate: prep = phantom / air
8. Low signal correction
9. Log transform
10. BHC
"""
function _forward_project_with_signal_chain!(
    sinogram::AbstractArray{T, 3},
    volume_or_mask::AbstractArray,
    geom::CTGeometry;
    energy::Union{Nothing, Real},
    energies::Union{Nothing, Vector},
    weights::Union{Nothing, Vector},
    materials::Union{Nothing, Vector},
    physics::Union{Nothing, PhysicsConfig},
    heel_effect::Union{Nothing, HeelEffect},
    das_model::Union{Nothing, DASModel},
    bhc::Union{Nothing, Union{BHCPolynomial, BeamHardeningCorrection}},
    max_prep::Union{Nothing, Real},
    noise_seed::Union{Nothing, Int}
) where T <: AbstractFloat

    # =========================================================================
    # LOG SIGNAL CHAIN CONFIGURATION
    # =========================================================================
    _log_signal_chain_config(physics, heel_effect, das_model, bhc, max_prep, noise_seed)

    # =========================================================================
    # STEP 1: Get raw sinogram (line integrals)
    # =========================================================================
    if eltype(volume_or_mask) <: AbstractFloat
        siddon_forward_project!(sinogram, volume_or_mask, geom)
    elseif eltype(volume_or_mask) == UInt8
        mask = volume_or_mask
        if materials === nothing
            error("materials must be provided when using a mask input")
        end
        if energy !== nothing
            _forward_project_mono!(sinogram, mask, geom, T(energy), materials)
        elseif energies !== nothing && weights !== nothing
            _forward_project_poly!(sinogram, mask, geom, energies, weights, materials)
        else
            error("Must specify either `energy` or `energies` + `weights`")
        end
    else
        error("volume_or_mask must be Float32/Float64 or UInt8")
    end

    # =========================================================================
    # STEP 2: Apply physics pipeline (in sinogram domain EXCEPT noise/DAS)
    # =========================================================================
    if physics !== nothing
        # Apply deterministic physics effects only (not noise - that's in DAS)
        # Create modified physics config without noise since DAS handles it
        _apply_physics_no_noise!(sinogram, geom, physics)
    end

    # =========================================================================
    # STEP 3: Convert to intensity domain
    # =========================================================================
    eps = T(1e-10)

    # Clamp sinogram to reasonable range before exp (avoid extreme intensities)
    AK.foreachindex(sinogram) do idx
        sinogram[idx] = exp(-clamp(sinogram[idx], T(-1), T(15)))
    end

    # Now sinogram contains INTENSITY values

    # =========================================================================
    # STEP 4: Apply heel effect to phantom intensity
    # =========================================================================
    if heel_effect !== nothing
        apply_heel_effect!(sinogram, heel_effect, geom)
    end

    # =========================================================================
    # STEP 5: Apply DAS model (gain + noise) to phantom
    # =========================================================================
    if das_model !== nothing
        apply_das_model!(sinogram, das_model; seed=noise_seed)
    end

    # =========================================================================
    # STEP 6: Create noise-free air scan (CatSim-exact)
    # =========================================================================
    # Air scan has same deterministic effects but NO noise
    air_scan = similar(sinogram)
    fill!(air_scan, one(T))

    if heel_effect !== nothing
        apply_heel_effect!(air_scan, heel_effect, geom)
    end

    if das_model !== nothing
        # Apply gain ONLY (no noise) - CatSim exact
        gain = T(das_model.gain)
        AK.foreachindex(air_scan) do idx
            air_scan[idx] *= gain
        end
    end

    # =========================================================================
    # STEP 7: Calibration (prep = phantom / air)
    # =========================================================================
    AK.foreachindex(sinogram) do idx
        air_val = max(air_scan[idx], eps)
        sinogram[idx] = sinogram[idx] / air_val
    end

    # =========================================================================
    # STEP 8: Low signal correction (CatSim-exact)
    # =========================================================================
    low_signal_correction_gpu!(sinogram)

    # =========================================================================
    # STEP 9: Log transform
    # =========================================================================
    if max_prep !== nothing
        max_val = T(max_prep)
        AK.foreachindex(sinogram) do idx
            val = -log(max(sinogram[idx], eps))
            sinogram[idx] = min(val, max_val)
        end
    else
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = -log(max(sinogram[idx], eps))
        end
    end

    # =========================================================================
    # STEP 10: Scatter correction (after log transform, before BHC)
    # =========================================================================
    # Apply scatter correction if specified in physics config
    # This estimates and subtracts scatter to reduce cupping artifacts
    if physics !== nothing && physics.scatter_correction !== nothing
        correct_scatter!(sinogram, physics.scatter_correction)
    end

    # =========================================================================
    # STEP 11: Beam hardening correction
    # =========================================================================
    if bhc !== nothing
        apply_bhc!(sinogram, bhc)
    end

    return sinogram
end

"""
Apply physics effects except noise (which is handled by DAS model).
"""
function _apply_physics_no_noise!(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    config::PhysicsConfig
) where T

    # Apply deterministic physics effects only
    # Skip noise since DAS model handles it

    # Fill factor
    if config.fill_factor !== nothing
        apply_fill_factor!(sinogram, config.fill_factor)
    end

    # Flat filter
    if config.flat_filter !== nothing
        apply_flat_filter!(sinogram, config.flat_filter, geom; energy_keV=config.energy_keV)
    end

    # Bowtie filter
    if config.bowtie_filter !== nothing
        apply_bowtie_filter!(sinogram, config.bowtie_filter, geom; energy_keV=config.energy_keV)
    end

    # Scatter
    if config.scatter !== nothing
        add_scatter!(sinogram, config.scatter)
    end

    # Crosstalk
    if config.crosstalk !== nothing
        apply_crosstalk!(sinogram, config.crosstalk)
    end

    # Optical crosstalk
    if config.optical_crosstalk !== nothing
        apply_optical_crosstalk!(sinogram, config.optical_crosstalk)
    end

    # Focal spot blur
    if config.focal_spot !== nothing
        apply_focal_spot_blur!(sinogram, config.focal_spot, geom)
    end

    # Detector efficiency
    if config.detector_efficiency !== nothing
        apply_detector_efficiency!(sinogram, config.detector_efficiency, geom; energy_keV=config.energy_keV)
    end

    # NOTE: Skip noise - handled by DAS model in signal chain

    # Detector lag
    if config.lag !== nothing
        apply_lag!(sinogram, config.lag)
    end

    return sinogram
end

# =============================================================================
# Internal Implementation Functions
# =============================================================================

"""Monochromatic forward projection from mask + single energy"""
function _forward_project_mono!(
    sinogram::AbstractArray{T, 3},
    mask::AbstractArray{UInt8, 3},
    geom::CTGeometry,
    energy_keV::T,
    materials::Vector
) where T <: AbstractFloat

    # Create μ volume at this energy
    μ_volume = similar(sinogram, T, size(mask))
    create_μ_volume!(μ_volume, mask, materials, energy_keV)

    # Forward project
    return siddon_forward_project!(sinogram, μ_volume, geom)
end

"""
    _forward_project_poly!(sinogram, mask, geom, energies, weights, materials) -> sinogram

Internal implementation of polychromatic forward projection using Beer-Lambert
physics with energy-sequential integration.

# Algorithm

Implements the polychromatic Beer-Lambert formula:

    I_total = Σₑ wₑ × exp(-Lₑ)

where:
- wₑ = normalized weight for energy bin e
- Lₑ = Siddon line integral through μ(E_e) volume

The algorithm proceeds energy-by-energy for memory efficiency:

```
I_transmitted = 0
for e in 1:N_energies:
    μ_volume = create_μ_volume(mask, materials, E_e)  # Energy-specific μ
    L_e = siddon_project(μ_volume)                     # Ray trace
    I_transmitted += w_e × exp(-L_e)                   # Accumulate intensity
end
sinogram = -log(max(I_transmitted, ε))                 # Convert to projection
```

# Mathematical Formulation

For a polychromatic beam passing through heterogeneous material:

    I/I₀ = ∫ w(E) × exp(-∫ μ(E, l) dl) dE

Discretized with N energy bins:

    I/I₀ = Σₑ₌₁ᴺ wₑ × exp(-Σᵢ μᵢ(Eₑ) × Δlᵢ)

The projection (negative log-intensity) is:

    p = -log(Σₑ wₑ × exp(-Lₑ))

Note: This is NOT equal to Σₑ wₑ × Lₑ (weighted average of monochromatic
projections) due to the nonlinearity of log-sum-exp. This nonlinearity
causes beam hardening artifacts.

# Arguments

- `sinogram::AbstractArray{T,3}`: Output array [n_cols, n_rows, n_angles], modified
  in place with line integral values.

- `mask::AbstractArray{UInt8,3}`: Material index volume [nx, ny, nz]. Values 0-255
  index into materials array (0-indexed, converted internally).

- `geom::CTGeometry`: Scanner geometry with source/detector positions and FOV.

- `energies::Vector`: Energy bin centers in keV. Length N_energies.
  Example: [20.0, 30.0, ..., 120.0] for 10 keV bins.

- `weights::Vector`: Photon fluence weights per energy bin. Will be normalized
  internally to sum to 1.0. Typically from X-ray tube spectrum.

- `materials::Vector`: Material definitions for each region index. Must include
  all indices present in mask. From `get_region_materials()`.

# Returns

- `sinogram::AbstractArray{T,3}`: The modified sinogram (same reference)

# Memory Usage

- One temporary μ-volume: O(nx × ny × nz) elements
- One temporary monochromatic sinogram: O(n_cols × n_rows × n_angles)
- One intensity accumulator: O(n_cols × n_rows × n_angles)
- Total: ~3× sinogram size additional memory, independent of N_energies

# Notes

- Weights are normalized internally (sum to 1.0)
- Epsilon clamping (1e-10) prevents log(0) errors
- GPU execution via AcceleratedKernels.jl when arrays are GPU arrays

# See Also

- [`forward_project!`](@ref): High-level interface
- [`create_μ_volume!`](@ref): Energy-specific μ volume creation
- [`siddon_forward_project!`](@ref): Monochromatic ray tracing
"""
function _forward_project_poly!(
    sinogram::AbstractArray{T, 3},
    mask::AbstractArray{UInt8, 3},
    geom::CTGeometry,
    energies::Vector,
    weights::Vector,
    materials::Vector
) where T <: AbstractFloat

    n_energies = length(energies)

    # Normalize weights
    weights_norm = T.(weights ./ sum(weights))

    # Allocate temporary arrays
    μ_volume = similar(sinogram, T, size(mask))
    sino_mono = similar(sinogram)
    I_transmitted = similar(sinogram)

    # Initialize transmitted intensity
    fill!(I_transmitted, zero(T))

    # Loop over energies (memory-efficient approach)
    for e_idx in 1:n_energies
        # Create μ volume for this energy
        create_μ_volume!(μ_volume, mask, materials, energies[e_idx])

        # Forward project at this energy
        fill!(sino_mono, zero(T))
        siddon_forward_project!(sino_mono, μ_volume, geom)

        # Accumulate Beer-Lambert: I += w × exp(-line_integral)
        w = weights_norm[e_idx]

        AK.foreachindex(I_transmitted) do idx
            I_transmitted[idx] += w * exp(-sino_mono[idx])
        end
    end

    # Convert back to line integral: sinogram = -log(I / I₀)
    eps = T(1e-10)

    AK.foreachindex(sinogram) do idx
        sinogram[idx] = -log(max(I_transmitted[idx], eps))
    end

    return sinogram
end

# =============================================================================
# Signal Chain Logging
# =============================================================================

"""
Log CatSim signal chain configuration with scanner-specific notes.

SCANNER-SPECIFIC parameters (vary by manufacturer/model):
- Flat filter (material, thickness)
- Bowtie filter (profile shape)
- Detector efficiency (scintillator type, thickness)
- Fill factor (detector geometry)
- Heel effect (anode angle, target material)
- DAS model (gain, noise characteristics)
- BHC coefficients (calibration-dependent)

PHYSICS parameters (generally applicable):
- Scatter (depends on patient size, not scanner)
- Crosstalk (optional, can be disabled)
- Focal spot blur (optional, can be disabled)
- Detector lag (optional, can be disabled)
"""
function _log_signal_chain_config(
    physics::Union{Nothing, PhysicsConfig},
    heel_effect::Union{Nothing, HeelEffect},
    das_model::Union{Nothing, DASModel},
    bhc,
    max_prep::Union{Nothing, Real},
    noise_seed::Union{Nothing, Int}
)
    println("\n" * "=" ^ 60)
    println("CATSIM SIGNAL CHAIN ACTIVE")
    println("=" ^ 60)

    # --- Physics Pipeline ---
    if physics !== nothing
        info = get_physics_config_info(physics)
        println("\n[PHYSICS PIPELINE] $(info.n_enabled) effects enabled:")

        # CatSim ESSENTIAL (scanner-specific)
        scanner_specific = ["fill_factor", "flat_filter", "bowtie_filter", "detector_efficiency"]
        optional = ["scatter", "crosstalk", "optical_crosstalk", "focal_spot", "lag", "noise"]

        for effect in info.enabled_effects
            if effect in scanner_specific
                println("  ✓ $effect  [SCANNER-SPECIFIC]")
            elseif effect in optional
                println("  ✓ $effect  [OPTIONAL]")
            else
                println("  ✓ $effect")
            end
        end
    else
        println("\n[PHYSICS PIPELINE] DISABLED")
    end

    # --- Heel Effect ---
    println("\n[HEEL EFFECT]")
    if heel_effect !== nothing
        info = get_heel_effect_info(heel_effect)
        println("  ✓ ENABLED  [SCANNER-SPECIFIC: anode geometry]")
        println("    Anode angle: $(info.anode_angle_deg)°")
        println("    Target: $(info.target_material)")
    else
        println("  ✗ DISABLED")
    end

    # --- DAS Model ---
    println("\n[DAS MODEL]")
    if das_model !== nothing
        info = get_das_info(das_model)
        println("  ✓ ENABLED  [SCANNER-SPECIFIC: electronics]")
        println("    Gain: $(info.gain)")
        println("    Electronic noise σ: $(info.electronic_noise_sigma)")
        if noise_seed !== nothing
            println("    Noise seed: $noise_seed (reproducible)")
        end
    else
        println("  ✗ DISABLED")
    end

    # --- Air Scan Calibration ---
    println("\n[AIR SCAN CALIBRATION]")
    println("  ✓ ENABLED  [CATSIM-EXACT: air scan has NO noise]")
    println("    Low signal correction: replace negatives with smoothed neighbors")
    if max_prep !== nothing
        println("    Max prep clamp: $max_prep")
    end

    # --- BHC ---
    println("\n[BEAM HARDENING CORRECTION]")
    if bhc !== nothing
        if hasproperty(bhc, :coefficients)
            order = length(bhc.coefficients) - 1
            println("  ✓ ENABLED  [SCANNER-SPECIFIC: calibration-dependent]")
            println("    Polynomial order: $order")
        else
            println("  ✓ ENABLED")
        end
    else
        println("  ✗ DISABLED")
    end

    println("\n" * "=" ^ 60)
    println()
end
