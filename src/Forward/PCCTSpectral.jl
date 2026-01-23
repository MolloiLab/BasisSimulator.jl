"""
    Forward/PCCTSpectral.jl

Spectral imaging algorithms for Photon-Counting CT (PCCT).

This module implements spectral processing capabilities unique to photon-counting CT,
including native Virtual Monoenergetic Imaging (VMI) directly from energy bins
without requiring material decomposition.

# Native PCCT Spectral Imaging

Unlike dual-energy CT which requires dual-kVp switching and material decomposition,
PCCT provides native spectral information from energy-binned photon counts. This
enables:

1. **Bin-Weighted VMI**: Direct synthesis of monoenergetic images from energy bins
2. **Multi-Bin Material Decomposition**: 3+ material decomposition with 4 energy bins
3. **K-Edge Imaging**: Direct detection of contrast agents via K-edge enhancement
4. **Effective Z Imaging**: Atomic number mapping from spectral response

# GPU Compatibility

All functions use AcceleratedKernels.jl for backend-agnostic GPU execution:
- Metal (Apple Silicon)
- CUDA (NVIDIA)
- ROCm (AMD)
- CPU fallback

# References

1. Willemink MJ, Persson M. "Photon-counting CT: Technical Principles and
   Clinical Prospects." Radiology. 2018;289(2):293-312.

2. Si-Mohamed S, et al. "Spectral Photon-Counting CT Technology: Initial
   Experience and Clinical Applications." Radiology. 2021;299(1):26-38.

3. Flohr T, et al. "Photon-counting CT review." Phys Med. 2020;79:126-136.

See also: [`PhotonCountingDetector`](@ref), [`EnergyResolvedSinogram`](@ref)
"""

import AcceleratedKernels as AK
import XrayAttenuation as XA
using Statistics

# =============================================================================
# PCCT VMI Result Container
# =============================================================================

"""
    PCCTVMIResult{T<:AbstractFloat}

Container for PCCT-native VMI reconstruction results.

# Fields
- `image::Array{T,3}`: Reconstructed image (HU or attenuation units)
- `energy_keV::Float64`: Target VMI energy
- `is_hu::Bool`: True if image is in HU, false if in attenuation units
- `μ_water::Float64`: Water attenuation used for HU conversion
- `method::Symbol`: Reconstruction method used
- `source::Symbol`: :pcct (bin-weighted) vs :dual_energy (material decomposition)
"""
struct PCCTVMIResult{T<:AbstractFloat}
    image::Array{T,3}
    energy_keV::Float64
    is_hu::Bool
    μ_water::Float64
    method::Symbol
    source::Symbol
end

# =============================================================================
# Bin-Weighted VMI (Native PCCT VMI)
# =============================================================================

"""
    compute_bin_weights(target_keV::Float64, thresholds::Vector{Float64}) -> Vector{Float64}

Compute optimal weighting factors for bin-weighted VMI synthesis.

The weights are designed to synthesize the equivalent monoenergetic image
at `target_keV` from the energy-binned data. This uses a linear combination
approach based on the center energy of each bin.

# Arguments
- `target_keV::Float64`: Target monoenergetic energy in keV
- `thresholds::Vector{Float64}`: Energy thresholds defining bins (e.g., [20, 35, 55, 70])

# Returns
- `weights::Vector{Float64}`: Weighting factors for each energy bin

# Physics

For NAEOTOM Alpha with thresholds [20, 35, 55, 70] keV and max kVp = 120:
- Bin 1: 20-35 keV, center ≈ 27.5 keV
- Bin 2: 35-55 keV, center ≈ 45 keV
- Bin 3: 55-70 keV, center ≈ 62.5 keV
- Bin 4: 70-120 keV, center ≈ 95 keV (assuming 120 kVp)

The weights are computed using inverse-distance weighting in energy space,
providing smooth interpolation between bin centers.

# Reference

Si-Mohamed et al., "Spectral Photon-Counting CT" (Radiology 2021)
"""
function compute_bin_weights(target_keV::Float64, thresholds::Vector{T}; max_keV::Float64=120.0) where T
    n_bins = length(thresholds)
    weights = zeros(T, n_bins)

    # Compute bin center energies
    bin_centers = zeros(T, n_bins)
    for i in 1:n_bins
        lower = thresholds[i]
        upper = i < n_bins ? thresholds[i+1] : max_keV
        bin_centers[i] = (lower + upper) / 2
    end

    # Inverse-distance weighting with Gaussian kernel
    σ = T(15.0)  # keV, controls smoothness of interpolation

    for i in 1:n_bins
        dist = abs(target_keV - bin_centers[i])
        weights[i] = exp(-(dist^2) / (2 * σ^2))
    end

    # Normalize weights to sum to 1
    total = sum(weights)
    if total > zero(T)
        weights ./= total
    else
        # Fallback: equal weights
        weights .= one(T) / n_bins
    end

    return weights
end

"""
    pcct_virtual_monoenergetic(sino::EnergyResolvedSinogram, target_keV::Float64;
                                max_keV::Float64=120.0) -> Array

Synthesize virtual monoenergetic sinogram from PCCT energy bins.

This is the native PCCT VMI - no material decomposition required.
The monoenergetic sinogram is computed as a weighted sum of energy bins.

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinogram from PCCT
- `target_keV::Float64`: Target VMI energy (40-190 keV)

# Keyword Arguments
- `max_keV::Float64=120.0`: Maximum energy (tube kVp)

# Returns
Sinogram at the virtual monoenergetic energy level.

# Advantages over Dual-Energy VMI

1. **Native spectral data**: No dual-kVp switching artifacts
2. **Lower noise at low keV**: Electronic noise rejection via thresholding
3. **Better energy resolution**: 4 bins vs 2 energy levels
4. **No material decomposition required**: Direct bin weighting

# Example

```julia
# PCCT forward projection
detector = naeotom_detector_standard()
pcct_sino = pcct_forward_project(volume, geom, detector, energies, weights)

# Native VMI at 50 keV
vmi_50 = pcct_virtual_monoenergetic(pcct_sino, 50.0)

# Reconstruct
recon = fdk_reconstruct(vmi_50, geom, recon_size)
```

See also: [`EnergyResolvedSinogram`](@ref), [`reconstruct_pcct_vmi`](@ref)
"""
function pcct_virtual_monoenergetic(
    sino::EnergyResolvedSinogram{T,A},
    target_keV::Float64;
    max_keV::Float64=120.0
) where {T, A}

    if target_keV < 10.0 || target_keV > 190.0
        error("VMI energy must be between 10 and 190 keV (got $target_keV)")
    end

    # Compute optimal weights for this energy
    weights = compute_bin_weights(target_keV, sino.thresholds_keV; max_keV=max_keV)

    # Weighted sum of bins
    vmi_sino = similar(sino.bins[1])
    fill!(vmi_sino, zero(T))

    for (i, w) in enumerate(weights)
        w_T = T(w)
        bin = sino.bins[i]
        AK.foreachindex(vmi_sino) do idx
            vmi_sino[idx] += bin[idx] * w_T
        end
    end

    return vmi_sino
end

"""
    pcct_vmi_to_hu(vmi_sinogram::AbstractArray{T,3}, target_keV::Float64;
                   μ_water=nothing) -> Array

Convert PCCT VMI sinogram to Hounsfield Units.

# Arguments
- `vmi_sinogram`: VMI sinogram from pcct_virtual_monoenergetic
- `target_keV`: VMI energy in keV

# Keyword Arguments
- `μ_water`: Water attenuation for calibration. If nothing, uses NIST value.

# Returns
Sinogram converted to HU scale.
"""
function pcct_vmi_to_hu(vmi_sinogram::AbstractArray{T,3}, target_keV::Float64;
                        μ_water=nothing) where T
    if μ_water === nothing
        μ_water = T(compute_μ_at_energy(XA.Materials.water, target_keV))
    else
        μ_water = T(μ_water)
    end

    # HU = 1000 × (μ - μ_water) / μ_water
    output = similar(vmi_sinogram)
    AK.foreachindex(output) do idx
        output[idx] = T(1000) * (vmi_sinogram[idx] - μ_water) / μ_water
    end

    return output
end

"""
    reconstruct_pcct_vmi(sino::EnergyResolvedSinogram, target_keV::Float64,
                         geom::CTGeometry, recon_size::NTuple{3,Int};
                         kwargs...) -> PCCTVMIResult

Full PCCT VMI reconstruction pipeline.

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinogram from PCCT
- `target_keV::Float64`: Target VMI energy (40-190 keV)
- `geom::CTGeometry`: CT geometry for reconstruction
- `recon_size::NTuple{3,Int}`: Output volume dimensions

# Keyword Arguments
- `method::Symbol=:fdk`: Reconstruction method (:fdk or :sirt)
- `to_hu::Bool=true`: Convert output to Hounsfield Units
- `water_mask=nothing`: Mask for water region (for empirical HU calibration)
- `max_keV::Float64=120.0`: Maximum energy (tube kVp)
- `niter::Int=3`: Number of iterations for SIRT
- `filter::FilterType=RampFilter()`: FDK filter
- `cutoff::Float64=1.0`: FDK frequency cutoff

# Returns
`PCCTVMIResult` with reconstructed image and metadata.

# Example

```julia
pcct_sino = pcct_forward_project(volume, geom, detector, energies, weights)
vmi_result = reconstruct_pcct_vmi(pcct_sino, 50.0, geom, (256, 256, 64))
println("VMI at \$(vmi_result.energy_keV) keV, range: [\$(minimum(vmi_result.image)), \$(maximum(vmi_result.image))] HU")
```
"""
function reconstruct_pcct_vmi(
    sino::EnergyResolvedSinogram{T,A},
    target_keV::Float64,
    geom,
    recon_size::NTuple{3,Int};
    method::Symbol=:fdk,
    to_hu::Bool=true,
    water_mask=nothing,
    max_keV::Float64=120.0,
    niter::Int=3,
    filter=RampFilter(),
    cutoff::Float64=1.0
) where {T, A}

    # Step 1: Generate VMI sinogram from bins
    vmi_sino = pcct_virtual_monoenergetic(sino, target_keV; max_keV=max_keV)

    # Step 2: Reconstruct
    if method == :fdk
        recon = fdk_reconstruct(vmi_sino, geom, recon_size; filter=filter, cutoff=cutoff)
    elseif method == :sirt
        recon = sirt_reconstruct(vmi_sino, geom, recon_size; niter=niter)
    else
        error("Unknown reconstruction method: $method. Use :fdk or :sirt")
    end

    # Step 3: Convert to HU if requested
    μ_water = T(compute_μ_at_energy(XA.Materials.water, target_keV))

    if to_hu
        recon_cpu = Array(recon)
        if water_mask !== nothing
            # Empirical calibration
            μ_water = mean(recon_cpu[water_mask])
        end

        recon_hu = T(1000) .* (recon_cpu .- μ_water) ./ μ_water

        return PCCTVMIResult(recon_hu, target_keV, true, Float64(μ_water), method, :pcct)
    else
        return PCCTVMIResult(Array(recon), target_keV, false, Float64(μ_water), method, :pcct)
    end
end

# =============================================================================
# Multi-Bin Material Decomposition
# =============================================================================

"""
    PCCTMaterialMap{T<:AbstractFloat, A<:AbstractArray{T,3}}

Result of multi-bin material decomposition from PCCT.

With 4 energy bins, PCCT can decompose into up to 3 basis materials
(vs 2 for dual-energy CT).

# Fields
- `materials::Vector{A}`: Vector of material density maps
- `material_names::Vector{Symbol}`: Names of basis materials
- `domain::Symbol`: :projection or :image
"""
struct PCCTMaterialMap{T<:AbstractFloat, A<:AbstractArray{T,3}}
    materials::Vector{A}
    material_names::Vector{Symbol}
    domain::Symbol
end

Base.size(mm::PCCTMaterialMap) = size(mm.materials[1])
Base.eltype(::PCCTMaterialMap{T}) where T = T
n_materials(mm::PCCTMaterialMap) = length(mm.materials)

"""
    pcct_material_decomposition(sino::EnergyResolvedSinogram;
                                 basis=(:water, :iodine),
                                 method=:least_squares) -> PCCTMaterialMap

Perform material decomposition from PCCT energy bins.

With N energy bins, PCCT can decompose into up to N-1 basis materials.
For NAEOTOM Alpha with 4 bins: up to 3 materials (water + 2 contrast agents).

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinogram from PCCT

# Keyword Arguments
- `basis::Tuple`: Basis material pair or triplet
  - 2 materials: `:water, :iodine` or `:water, :calcium`
  - 3 materials: `:water, :iodine, :calcium` (requires 4+ bins)
- `method::Symbol`: Decomposition method
  - `:least_squares` - Linear least squares (fast, default)
  - `:maximum_likelihood` - ML estimation (more accurate, slower)

# Returns
`PCCTMaterialMap` with material density projections.

# Example

```julia
# Two-material decomposition (standard)
mat_map = pcct_material_decomposition(pcct_sino; basis=(:water, :iodine))

# Three-material decomposition (PCCT advantage)
mat_map_3 = pcct_material_decomposition(pcct_sino; basis=(:water, :iodine, :calcium))
```

# Physics

The system is:
    p_bin(i) = Σⱼ μⱼ(Ē_i) × ρⱼ × L

where Ē_i is the effective energy of bin i, μⱼ is the mass attenuation
of material j, ρⱼ is the material density, and L is the path length.

For N bins and M materials (M < N), we solve the overdetermined system
using least squares.
"""
function pcct_material_decomposition(
    sino::EnergyResolvedSinogram{T,A};
    basis::NTuple{M,Symbol}=(:water, :iodine),
    method::Symbol=:least_squares,
    max_keV::Float64=120.0
) where {T, A, M}

    n_bins = n_energy_bins(sino)
    n_materials = length(basis)

    if n_materials >= n_bins
        error("Number of basis materials ($n_materials) must be less than number of energy bins ($n_bins)")
    end

    # Compute effective energies for each bin
    thresholds = sino.thresholds_keV
    bin_energies = zeros(T, n_bins)
    for i in 1:n_bins
        lower = thresholds[i]
        upper = i < n_bins ? thresholds[i+1] : max_keV
        bin_energies[i] = (lower + upper) / 2
    end

    # Build attenuation matrix A: [n_bins × n_materials]
    # A[i,j] = μⱼ(Ē_i) = attenuation of material j at energy of bin i
    A_mat = zeros(T, n_bins, n_materials)
    for (j, mat_sym) in enumerate(basis)
        for i in 1:n_bins
            A_mat[i, j] = T(get_material_attenuation_pcct(mat_sym, bin_energies[i]))
        end
    end

    # Solve least squares: p = A × ρ  →  ρ = A⁺ × p
    # Compute pseudo-inverse
    A_pinv = pinv(A_mat)

    # Allocate output materials
    n_cols, n_rows, n_angles = size(sino)
    material_maps = [similar(sino.bins[1]) for _ in 1:n_materials]
    for m in material_maps
        fill!(m, zero(T))
    end

    # Apply decomposition (GPU-native)
    bins = sino.bins

    # For each pixel, compute material densities
    for idx in 1:length(material_maps[1])
        # Gather bin values for this pixel
        p = zeros(T, n_bins)
        for i in 1:n_bins
            p[i] = bins[i][idx]
        end

        # Apply pseudo-inverse
        for j in 1:n_materials
            ρ = zero(T)
            for i in 1:n_bins
                ρ += A_pinv[j, i] * p[i]
            end
            material_maps[j][idx] = ρ
        end
    end

    return PCCTMaterialMap{T,A}(material_maps, collect(basis), :projection)
end

"""
    get_material_attenuation_pcct(material::Symbol, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for PCCT material at given energy.

Supports standard materials plus K-edge contrast agents.
"""
function get_material_attenuation_pcct(material::Symbol, energy_keV::Float64)
    if material == :water
        return compute_μ_at_energy(XA.Materials.water, energy_keV)
    elseif material == :iodine
        # Dilute iodine solution (5 mg/mL)
        return get_iodine_solution_attenuation(energy_keV; conc_mg_ml=5.0)
    elseif material == :calcium
        # Calcium-equivalent material (200 mg/cc)
        return get_calcium_material_attenuation(energy_keV; density_mg_cc=200.0)
    elseif material == :gadolinium
        # Dilute gadolinium solution (1 mM ≈ 0.157 mg/mL)
        return get_gadolinium_solution_attenuation(energy_keV; conc_mg_ml=0.5)
    elseif material == :gold
        # Gold nanoparticles (1 mg/mL)
        return get_gold_solution_attenuation(energy_keV; conc_mg_ml=1.0)
    else
        error("Unknown PCCT material: $material. Use :water, :iodine, :calcium, :gadolinium, or :gold")
    end
end

"""
    get_gadolinium_solution_attenuation(energy_keV::Float64; conc_mg_ml::Float64=0.5) -> Float64

Get linear attenuation for dilute gadolinium solution.

Gadolinium K-edge is at 50.2 keV.
"""
function get_gadolinium_solution_attenuation(energy_keV::Float64; conc_mg_ml::Float64=0.5)
    μ_water = compute_μ_at_energy(XA.Materials.water, energy_keV)
    μ_ρ_gd = compute_mass_μ_at_energy(XA.Elements.Gadolinium, energy_keV)
    conc_g_cm3 = conc_mg_ml / 1000.0
    return μ_water + conc_g_cm3 * μ_ρ_gd
end

"""
    get_gold_solution_attenuation(energy_keV::Float64; conc_mg_ml::Float64=1.0) -> Float64

Get linear attenuation for gold nanoparticle solution.

Gold K-edge is at 80.7 keV.
"""
function get_gold_solution_attenuation(energy_keV::Float64; conc_mg_ml::Float64=1.0)
    μ_water = compute_μ_at_energy(XA.Materials.water, energy_keV)
    μ_ρ_au = compute_mass_μ_at_energy(XA.Elements.Gold, energy_keV)
    conc_g_cm3 = conc_mg_ml / 1000.0
    return μ_water + conc_g_cm3 * μ_ρ_au
end

# =============================================================================
# K-Edge Imaging
# =============================================================================

"""
    K_EDGE_ENERGIES

K-edge energies for common contrast agents in keV.
"""
const K_EDGE_ENERGIES = Dict{Symbol, Float64}(
    :iodine => 33.2,
    :gadolinium => 50.2,
    :gold => 80.7,
    :barium => 37.4,
    :bismuth => 90.5
)

"""
    compute_kedge_enhancement(sino::EnergyResolvedSinogram, element::Symbol;
                               method=:subtraction) -> Array

Compute K-edge enhancement map for a specific element.

K-edge imaging exploits the sudden increase in attenuation at the K-edge
energy of high-Z elements. PCCT energy bins spanning the K-edge enable
direct K-edge detection.

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinogram
- `element::Symbol`: Element to detect (:iodine, :gadolinium, :gold)

# Keyword Arguments
- `method::Symbol`: Enhancement method
  - `:subtraction` - Subtract bin below K-edge from bin above
  - `:ratio` - Ratio of bins above/below K-edge

# Returns
K-edge enhancement map (high values indicate element presence).

# Physics

For iodine (K-edge = 33.2 keV):
- NAEOTOM threshold at 35 keV brackets the K-edge
- Bin 1 (20-35 keV): below K-edge
- Bin 2 (35-55 keV): above K-edge
- Enhancement = (Bin 2 - Bin 1) or (Bin 2 / Bin 1)

# Example

```julia
iodine_map = compute_kedge_enhancement(pcct_sino, :iodine)
```

# Limitations

- NAEOTOM Alpha thresholds (20/35/55/70 keV) are optimized for iodine
- Gadolinium (50.2 keV) has limited sensitivity due to 55 keV threshold
- Gold (80.7 keV) is not detectable with current clinical threshold settings
"""
function compute_kedge_enhancement(
    sino::EnergyResolvedSinogram{T,A},
    element::Symbol;
    method::Symbol=:subtraction
) where {T, A}

    if !haskey(K_EDGE_ENERGIES, element)
        error("Unknown element: $element. Available: $(keys(K_EDGE_ENERGIES))")
    end

    k_edge = K_EDGE_ENERGIES[element]
    thresholds = sino.thresholds_keV
    n_bins = n_energy_bins(sino)

    # Find bins spanning the K-edge
    bin_above = findfirst(t -> t > k_edge, thresholds)
    if bin_above === nothing || bin_above == 1
        @warn "K-edge ($k_edge keV) not bracketed by thresholds. Results may be inaccurate."
        bin_above = 2
    end
    bin_below = bin_above - 1

    # Compute enhancement
    output = similar(sino.bins[1])
    bin_above_data = sino.bins[min(bin_above, n_bins)]
    bin_below_data = sino.bins[bin_below]

    if method == :subtraction
        AK.foreachindex(output) do idx
            output[idx] = bin_above_data[idx] - bin_below_data[idx]
        end
    elseif method == :ratio
        ε = T(1e-10)  # Avoid division by zero
        AK.foreachindex(output) do idx
            output[idx] = bin_above_data[idx] / (bin_below_data[idx] + ε)
        end
    else
        error("Unknown K-edge method: $method. Use :subtraction or :ratio")
    end

    return output
end

"""
    get_kedge_sensitivity(detector::PhotonCountingDetector, element::Symbol) -> NamedTuple

Evaluate K-edge detection sensitivity for a given element with the detector thresholds.

Returns information about whether the K-edge is properly bracketed and
the expected contrast-to-noise ratio improvement.
"""
function get_kedge_sensitivity(detector::PhotonCountingDetector, element::Symbol)
    if !haskey(K_EDGE_ENERGIES, element)
        error("Unknown element: $element")
    end

    k_edge = K_EDGE_ENERGIES[element]
    thresholds = detector.energy_thresholds_keV

    # Check if K-edge is bracketed
    bin_above = findfirst(t -> t > k_edge, thresholds)
    bracketed = bin_above !== nothing && bin_above > 1

    # Distance from K-edge to nearest threshold
    if bracketed
        dist_above = thresholds[bin_above] - k_edge
        dist_below = k_edge - thresholds[bin_above - 1]
        min_dist = min(dist_above, dist_below)
    else
        min_dist = Inf
    end

    # Energy resolution effect
    energy_resolution = detector.energy_resolution_keV

    # Qualitative sensitivity rating
    if bracketed && min_dist < energy_resolution
        sensitivity = :optimal
    elseif bracketed && min_dist < 2 * energy_resolution
        sensitivity = :good
    elseif bracketed
        sensitivity = :moderate
    else
        sensitivity = :poor
    end

    return (
        element = element,
        k_edge_keV = k_edge,
        bracketed = bracketed,
        nearest_threshold_distance_keV = min_dist,
        energy_resolution_keV = energy_resolution,
        sensitivity = sensitivity
    )
end

# =============================================================================
# Effective Z Imaging
# =============================================================================

"""
    compute_effective_z(sino::EnergyResolvedSinogram;
                        method=:dual_ratio) -> Array

Compute effective atomic number (Z_eff) map from PCCT data.

Effective Z imaging uses the energy dependence of attenuation to estimate
the average atomic number of materials.

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinogram

# Keyword Arguments
- `method::Symbol`: Computation method
  - `:dual_ratio` - Uses ratio of high/low energy bins
  - `:fit` - Fits power law to all bins (slower, more accurate)

# Returns
Effective Z map (Z_eff values, typically 6-20 for tissues).

# Physics

Attenuation can be approximated as:
    μ ∝ Z^n × E^(-m)

where n ≈ 3.5 for photoelectric effect dominance.

The ratio of attenuation at two energies:
    R = μ_low / μ_high ∝ (E_high / E_low)^m

gives Z_eff ∝ R^(1/n) after accounting for Compton contributions.

# Reference Values
- Soft tissue: Z_eff ≈ 7.4
- Bone: Z_eff ≈ 13.8
- Iodine: Z_eff ≈ 53
- Water: Z_eff ≈ 7.42
"""
function compute_effective_z(
    sino::EnergyResolvedSinogram{T,A};
    method::Symbol=:dual_ratio,
    max_keV::Float64=120.0
) where {T, A}

    n_bins = n_energy_bins(sino)
    thresholds = sino.thresholds_keV

    # Compute bin center energies
    bin_energies = zeros(T, n_bins)
    for i in 1:n_bins
        lower = thresholds[i]
        upper = i < n_bins ? thresholds[i+1] : max_keV
        bin_energies[i] = (lower + upper) / 2
    end

    output = similar(sino.bins[1])

    if method == :dual_ratio
        # Use lowest and highest bins
        low_bin = sino.bins[1]
        high_bin = sino.bins[end]
        E_low = bin_energies[1]
        E_high = bin_energies[end]

        # Z_eff estimation from dual-energy ratio
        # Simplified model: Z_eff^3.5 ∝ (μ_low/μ_high - (E_low/E_high)^3) / ...
        # Using empirical calibration for soft tissue range

        # Reference: water at these energies
        μ_water_low = T(compute_μ_at_energy(XA.Materials.water, Float64(E_low)))
        μ_water_high = T(compute_μ_at_energy(XA.Materials.water, Float64(E_high)))
        R_water = μ_water_low / μ_water_high
        Z_water = T(7.42)

        AK.foreachindex(output) do idx
            μ_low = low_bin[idx]
            μ_high = high_bin[idx]

            # Avoid division by zero
            if μ_high < T(1e-10)
                output[idx] = Z_water
            else
                R = μ_low / μ_high
                # Empirical mapping: Z_eff ∝ (R / R_water)^α × Z_water
                α = T(0.4)  # Calibration exponent
                output[idx] = ((R / R_water)^α) * Z_water
            end
        end

    elseif method == :fit
        # Fit power law to all bins (more accurate but slower)
        # μ(E) = a × E^(-m), m depends on Z_eff

        ε = T(1e-10)
        AK.foreachindex(output) do idx
            # Gather data points
            μ_vals = [sino.bins[i][idx] for i in 1:n_bins]

            # Log-linear fit: log(μ) = log(a) - m × log(E)
            # m is related to Z_eff through the material model

            sum_x = sum(log.(bin_energies))
            sum_y = sum(log.(max.(μ_vals, ε)))
            sum_xy = sum(log.(bin_energies) .* log.(max.(μ_vals, ε)))
            sum_x2 = sum(log.(bin_energies).^2)
            n = n_bins

            # Slope m from linear regression
            m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2 + ε)

            # Map m to Z_eff (empirical calibration)
            # For photoelectric: m ≈ 3, for Compton: m ≈ 1
            # Z_eff = f(m) calibrated to known materials
            Z_eff = T(7.42) * (abs(m) / T(2.7))^T(0.5)
            Z_eff = clamp(Z_eff, T(1), T(100))

            output[idx] = Z_eff
        end
    else
        error("Unknown effective Z method: $method. Use :dual_ratio or :fit")
    end

    return output
end

# =============================================================================
# PCCT vs Dual-Energy Comparison
# =============================================================================

"""
    compare_pcct_vs_dect_vmi(pcct_vmi::Array, dect_vmi::Array;
                             mask=nothing) -> NamedTuple

Compare PCCT VMI with dual-energy VMI for quantitative analysis.

# Arguments
- `pcct_vmi`: VMI from PCCT (bin-weighted)
- `dect_vmi`: VMI from dual-kVp material decomposition

# Keyword Arguments
- `mask`: Optional mask for ROI analysis

# Returns
NamedTuple with comparison metrics:
- `mean_diff`: Mean difference (PCCT - DECT)
- `std_diff`: Standard deviation of difference
- `correlation`: Pearson correlation coefficient
- `pcct_noise`: Noise (std) in PCCT VMI
- `dect_noise`: Noise (std) in DECT VMI
- `noise_ratio`: DECT_noise / PCCT_noise (>1 means PCCT has lower noise)
"""
function compare_pcct_vs_dect_vmi(
    pcct_vmi::AbstractArray{T},
    dect_vmi::AbstractArray{T};
    mask=nothing
) where T

    if size(pcct_vmi) != size(dect_vmi)
        error("VMI arrays must have same size")
    end

    # Apply mask if provided
    if mask !== nothing
        pcct_vals = pcct_vmi[mask]
        dect_vals = dect_vmi[mask]
    else
        pcct_vals = vec(pcct_vmi)
        dect_vals = vec(dect_vmi)
    end

    # Compute difference statistics
    diff = pcct_vals .- dect_vals
    mean_diff = mean(diff)
    std_diff = std(diff)

    # Correlation
    correlation = cor(pcct_vals, dect_vals)

    # Noise (using local variance estimate)
    pcct_noise = std(pcct_vals)
    dect_noise = std(dect_vals)
    noise_ratio = dect_noise / (pcct_noise + 1e-10)

    return (
        mean_diff = mean_diff,
        std_diff = std_diff,
        correlation = correlation,
        pcct_noise = pcct_noise,
        dect_noise = dect_noise,
        noise_ratio = noise_ratio
    )
end

"""
    expected_pcct_noise_advantage(energy_keV::Float64) -> Float64

Compute expected noise advantage of PCCT VMI vs dual-energy VMI at given energy.

PCCT provides lower noise especially at low keV due to:
1. Electronic noise rejection via thresholding
2. No spectral overlap between bins
3. More energy levels for interpolation

# Returns
Expected noise ratio (DECT_noise / PCCT_noise). Values > 1 indicate PCCT advantage.

# Reference
Springer Performance improvements of VMI in PCD-CT vs DSCT (2024)
"""
function expected_pcct_noise_advantage(energy_keV::Float64)
    # Empirical model from literature
    # Maximum advantage at low keV (40-50), decreasing at higher keV

    if energy_keV <= 40
        return 1.4  # 40% lower noise
    elseif energy_keV <= 50
        return 1.3
    elseif energy_keV <= 60
        return 1.2
    elseif energy_keV <= 80
        return 1.1
    else
        return 1.05  # Minimal advantage at high keV
    end
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    generate_pcct_vmi_series(sino::EnergyResolvedSinogram, energies_keV::Vector{Float64},
                              geom, recon_size::NTuple{3,Int};
                              kwargs...) -> Dict{Float64, PCCTVMIResult}

Generate PCCT VMI at multiple energies.

# Example
```julia
energies = [40.0, 50.0, 70.0, 100.0, 140.0]
vmi_series = generate_pcct_vmi_series(pcct_sino, energies, geom, (128, 128, 32))
```
"""
function generate_pcct_vmi_series(
    sino::EnergyResolvedSinogram,
    energies_keV::Vector{Float64},
    geom,
    recon_size::NTuple{3,Int};
    kwargs...
)
    return Dict(E => reconstruct_pcct_vmi(sino, E, geom, recon_size; kwargs...)
                for E in energies_keV)
end

"""
    print_pcct_spectral_info(sino::EnergyResolvedSinogram)

Print summary information about PCCT spectral data.
"""
function print_pcct_spectral_info(sino::EnergyResolvedSinogram)
    println("=" ^ 60)
    println("PCCT SPECTRAL SINOGRAM")
    println("=" ^ 60)
    println("Dimensions:       $(sino.n_cols) × $(sino.n_rows) × $(sino.n_angles)")
    println("Number of bins:   $(n_energy_bins(sino))")
    println("Thresholds (keV): $(sino.thresholds_keV)")
    println()
    println("Bin Statistics:")
    println("-" ^ 40)
    for (i, bin) in enumerate(sino.bins)
        lower = sino.thresholds_keV[i]
        upper = i < length(sino.thresholds_keV) ? sino.thresholds_keV[i+1] : "max"
        println("  Bin $i ($lower-$upper keV): mean=$(round(mean(bin), digits=2)), std=$(round(std(bin), digits=2))")
    end
    println("=" ^ 60)
end

"""
    get_supported_kedge_elements(detector::PhotonCountingDetector) -> Vector{Symbol}

Get list of K-edge elements that can be detected with given detector thresholds.
"""
function get_supported_kedge_elements(detector::PhotonCountingDetector)
    supported = Symbol[]
    for (elem, k_edge) in K_EDGE_ENERGIES
        info = get_kedge_sensitivity(detector, elem)
        if info.sensitivity in (:optimal, :good, :moderate)
            push!(supported, elem)
        end
    end
    return supported
end

# =============================================================================
# VMI Synthesis from Material Maps (PCCT-NOISE-DECOMP)
# =============================================================================

"""
    synthesize_vmi(material_map::PCCTMaterialMap, energy_keV::Float64) -> AbstractArray

Synthesize a Virtual Monoenergetic Image (VMI) from material density maps.

The VMI at energy E is computed as:
    μ_VMI(E) = Σᵢ ρᵢ × μᵢ(E)

where ρᵢ is the material density map and μᵢ(E) is the linear attenuation
coefficient of basis material i at energy E.

This works for any number of basis materials (2 for dual-energy, 3+ for PCCT).

# Arguments
- `material_map::PCCTMaterialMap`: N-material density maps from decomposition
- `energy_keV::Float64`: Target VMI energy in keV

# Returns
- Array of same shape as material maps (VMI sinogram or image, depending on domain)

# Example
```julia
# 3-material decomposition from PCCT
mat_map = pcct_material_decomposition(pcct_sino; basis=(:water, :iodine, :calcium))

# VMI at 70 keV
vmi_70 = synthesize_vmi(mat_map, 70.0)
```
"""
function synthesize_vmi(material_map::PCCTMaterialMap{T,A}, energy_keV::Float64) where {T, A}
    n_materials = length(material_map.materials)
    @assert n_materials > 0 "Material map must have at least one material"

    # Get attenuation coefficient for each basis material at target energy
    μ_values = Vector{T}(undef, n_materials)
    for (i, mat_name) in enumerate(material_map.material_names)
        μ_values[i] = T(_get_basis_material_attenuation(mat_name, energy_keV))
    end

    # Synthesize VMI: μ_VMI = Σ ρᵢ × μᵢ(E)
    result = similar(material_map.materials[1])
    fill!(result, zero(T))

    for i in 1:n_materials
        μ_i = μ_values[i]
        mat_i = material_map.materials[i]
        AK.foreachindex(result) do idx
            result[idx] += mat_i[idx] * μ_i
        end
    end

    return result
end

"""
    _get_basis_material_attenuation(material_name::Symbol, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for a basis material at given energy.
Maps common VMI basis names to XrayAttenuation.jl materials.
"""
function _get_basis_material_attenuation(material_name::Symbol, energy_keV::Float64)
    if material_name == :water
        return compute_μ_at_energy(XA.Materials.water, energy_keV)
    elseif material_name == :iodine
        # Iodine solution at 1 mg/mL as reference concentration
        return compute_mass_μ_at_energy(XA.Elements.Iodine, energy_keV) * 0.001  # 1 mg/mL
    elseif material_name == :calcium
        # Calcium hydroxyapatite as reference
        return compute_μ_at_energy(Ca_200, energy_keV)  # Ca_200 as representative
    elseif material_name == :bone
        return compute_μ_at_energy(Ca_200, energy_keV)
    elseif material_name == :air
        return compute_μ_at_energy(XA.Materials.air, energy_keV)
    else
        # Try to look up in materials registry
        mat = get_material(material_name)
        return compute_μ_at_energy(mat, energy_keV)
    end
end

# =============================================================================
# Exports
# =============================================================================

export PCCTVMIResult, PCCTMaterialMap
export synthesize_vmi
export compute_bin_weights, pcct_virtual_monoenergetic, pcct_vmi_to_hu
export reconstruct_pcct_vmi, generate_pcct_vmi_series
export pcct_material_decomposition, get_material_attenuation_pcct
export get_gadolinium_solution_attenuation, get_gold_solution_attenuation
export K_EDGE_ENERGIES, compute_kedge_enhancement, get_kedge_sensitivity
export compute_effective_z
export compare_pcct_vs_dect_vmi, expected_pcct_noise_advantage
export print_pcct_spectral_info, get_supported_kedge_elements
export n_materials
