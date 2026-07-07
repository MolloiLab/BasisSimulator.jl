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
# Spectrum-Weighted Bin Energies
# =============================================================================

"""
    compute_pcct_bin_energies(thresholds; max_keV=120.0, kvp=120) -> Vector{Float64}

Compute spectrum-weighted effective energy for each PCCT energy bin.

Uses the X-ray spectrum at the given kVp to compute the photon-weighted mean
energy within each bin window. This is more accurate than arithmetic mean of
bin boundaries because real spectra are not uniform.

# Arguments
- `thresholds::Vector`: Energy thresholds defining bins (e.g., [20, 35, 55, 70])

# Keyword Arguments
- `max_keV::Float64=120.0`: Maximum energy (tube kVp)
- `kvp::Int=120`: Tube kVp for spectrum loading

# Returns
Vector of effective energies for each bin (keV).

# Example
```julia
thresholds = [20.0, 35.0, 55.0, 70.0]
E_eff = compute_pcct_bin_energies(thresholds; kvp=120)
# [28.3, 44.5, 62.3, 88.7]  (vs arithmetic: [27.5, 45.0, 62.5, 95.0])
```
"""
function compute_pcct_bin_energies(thresholds::Vector{T}; max_keV::Float64=120.0, kvp::Int=120) where T
    n_bins = length(thresholds)
    bin_energies = zeros(Float64, n_bins)

    # Load spectrum for this kVp
    energies, weights = load_spectrum(kvp)

    for i in 1:n_bins
        lower = Float64(thresholds[i])
        upper = i < n_bins ? Float64(thresholds[i+1]) : max_keV

        # Find spectrum components within this bin window
        total_weight = 0.0
        weighted_energy = 0.0
        for k in eachindex(energies)
            if lower ≤ energies[k] < upper
                weighted_energy += energies[k] * weights[k]
                total_weight += weights[k]
            end
        end

        if total_weight > 0
            bin_energies[i] = weighted_energy / total_weight
        else
            # Fallback to arithmetic mean if no spectrum data in this window
            bin_energies[i] = (lower + upper) / 2
        end
    end

    return bin_energies
end

export compute_pcct_bin_energies

# =============================================================================
# Bin-Weighted VMI (Native PCCT VMI)
# =============================================================================



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
    basis::Union{NTuple{M,Symbol} where M, Vector{Symbol}} = (:water, :iodine),
    method::Symbol=:least_squares,
    max_keV::Float64=120.0,
    kvp::Int=120,
    # Workspace buffers (optional — allocate internally if not provided)
    ws_bins_cpu = nothing,
    ws_material_maps = nothing,
    ws_decomp_pixel_buf = nothing,
    ws_A_pinv = nothing,  # Pre-computed pseudo-inverse (n_materials × n_bins)
    ws_basis_vec = nothing  # Pre-computed basis vector
) where {T, A}
    Base.depwarn("pcct_material_decomposition is a legacy effective-energy LSQ (ill-conditioned iodine basis); use the Cong projection-domain chain (apply_cong! with Phi_k).", :pcct_material_decomposition)

    basis_vec = if ws_basis_vec !== nothing
        ws_basis_vec
    else
        basis isa Tuple ? collect(Symbol, basis) : Vector{Symbol}(basis)
    end
    n_bins = n_energy_bins(sino)
    n_materials = length(basis_vec)

    if n_materials >= n_bins
        error("Number of basis materials ($n_materials) must be less than number of energy bins ($n_bins)")
    end

    A_pinv = if ws_A_pinv !== nothing
        ws_A_pinv
    else
        # Compute spectrum-weighted effective energies for each bin
        bin_energies = compute_pcct_bin_energies(sino.thresholds_keV; max_keV=max_keV, kvp=kvp)

        # Build attenuation matrix A: [n_bins × n_materials]
        A_mat = zeros(T, n_bins, n_materials)
        for (j, mat_sym) in enumerate(basis_vec)
            for i in 1:n_bins
                A_mat[i, j] = T(get_basis_mu(mat_sym, Float64(bin_energies[i])))
            end
        end

        pinv(A_mat)
    end

    # Copy bins to CPU only if needed (avoid redundant copy when already CPU)
    bins_cpu = if ws_bins_cpu !== nothing
        # Use workspace buffers — copyto! each bin
        if eltype(sino.bins) <: Array
            # Already CPU, just reference directly
            for (i, b) in enumerate(sino.bins)
                copyto!(ws_bins_cpu[i], b)
            end
        else
            # GPU→CPU transfer into pre-allocated buffers
            for (i, b) in enumerate(sino.bins)
                copyto!(ws_bins_cpu[i], b)
            end
        end
        ws_bins_cpu
    elseif eltype(sino.bins) <: Array
        sino.bins  # Already CPU arrays — use directly, no copy
    else
        [Array(b) for b in sino.bins]  # GPU→CPU transfer
    end

    # Allocate output on CPU (or use workspace)
    n_elements = length(bins_cpu[1])
    material_maps_cpu = if ws_material_maps !== nothing
        for m in ws_material_maps
            fill!(m, zero(T))
        end
        ws_material_maps
    else
        [zeros(T, size(bins_cpu[1])) for _ in 1:n_materials]
    end

    # Apply decomposition on CPU (fast memory access)
    p = if ws_decomp_pixel_buf !== nothing
        ws_decomp_pixel_buf
    else
        zeros(T, n_bins)
    end
    @inbounds for idx in 1:n_elements
        # Gather bin values for this pixel
        for i in 1:n_bins
            p[i] = bins_cpu[i][idx]
        end

        # Apply pseudo-inverse
        for j in 1:n_materials
            ρ = zero(T)
            for i in 1:n_bins
                ρ += A_pinv[j, i] * p[i]
            end
            material_maps_cpu[j][idx] = ρ
        end
    end

    # Return CPU arrays directly — no redundant copy to GPU then back
    # The caller (driver) will use these as-is or copy to GPU if needed
    return PCCTMaterialMap{T,typeof(material_maps_cpu[1])}(material_maps_cpu, basis_vec, :projection)
end


"""
    get_material_attenuation_pcct(material::Symbol, energy_keV::Float64) -> Float64

Deprecated: use `get_basis_mu()` from the unified VMI pipeline instead.
Kept as thin wrapper for backward compatibility.
"""
get_material_attenuation_pcct(material::Symbol, energy_keV::Float64) = get_basis_mu(material, energy_keV)

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
    max_keV::Float64=120.0,
    kvp::Int=120
) where {T, A}

    n_bins = n_energy_bins(sino)
    thresholds = sino.thresholds_keV

    # Compute spectrum-weighted bin center energies
    bin_energies_f64 = compute_pcct_bin_energies(thresholds; max_keV=max_keV, kvp=kvp)
    bin_energies = T.(bin_energies_f64)

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

"""
    synthesize_vmi(material_map::PCCTMaterialMap, energy_keV::Float64; output=nothing, ws_μ_values=nothing) -> AbstractArray

Synthesize VMI from N-material density maps.

For 2-material maps, delegates to `spectral_vmi!()`. For 3+ materials,
uses a loop over materials (general N-material path).
"""
function synthesize_vmi(material_map::PCCTMaterialMap{T,A}, energy_keV::Float64;
                        output=nothing,
                        ws_μ_values=nothing) where {T, A}
    n_mats = length(material_map.materials)
    @assert n_mats > 0 "Material map must have at least one material"

    μ_values = ws_μ_values !== nothing ? ws_μ_values : Vector{T}(undef, n_mats)
    for (i, mat_name) in enumerate(material_map.material_names)
        μ_values[i] = T(get_basis_mu(mat_name, energy_keV))
    end

    result = output === nothing ? similar(material_map.materials[1]) : output

    if n_mats == 2
        spectral_vmi!(result, material_map.materials[1], material_map.materials[2],
                      μ_values[1], μ_values[2])
    else
        fill!(result, zero(T))
        for i in 1:n_mats
            μ_i = μ_values[i]
            mat_i = material_map.materials[i]
            let result=result, mat_i=mat_i, μ_i=μ_i
                AK.foreachindex(result) do idx
                    result[idx] += mat_i[idx] * μ_i
                end
            end
        end
    end

    return result
end

# =============================================================================
# Exports
# =============================================================================

export PCCTMaterialMap, n_materials
export synthesize_vmi
export pcct_material_decomposition, get_material_attenuation_pcct
export K_EDGE_ENERGIES, compute_kedge_enhancement, get_kedge_sensitivity
export compute_effective_z
export get_supported_kedge_elements
