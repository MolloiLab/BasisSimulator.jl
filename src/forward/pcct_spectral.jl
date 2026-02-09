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

See also: [`EnergyResolvedSinogram`](@ref)
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
    # Workspace buffers (optional — allocate internally if not provided)
    ws_bins_cpu = nothing,
    ws_material_maps = nothing,
    ws_decomp_pixel_buf = nothing
) where {T, A}

    basis_vec = basis isa Tuple ? collect(Symbol, basis) : Vector{Symbol}(basis)
    n_bins = n_energy_bins(sino)
    n_materials = length(basis_vec)

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
    for (j, mat_sym) in enumerate(basis_vec)
        for i in 1:n_bins
            A_mat[i, j] = T(get_material_attenuation_pcct(mat_sym, Float64(bin_energies[i])))
        end
    end

    # Solve least squares: p = A × ρ  →  ρ = A⁺ × p
    # Compute pseudo-inverse
    A_pinv = pinv(A_mat)

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
    pcct_material_decomposition_mle(sino, detector; basis, energies, weights, max_iterations, max_keV) -> PCCTMaterialMap

Maximum Likelihood Estimation (MLE) material decomposition from N energy bins.

Uses the full forward model with Poisson log-likelihood to find optimal
material densities. Statistically optimal (approaches Cramér-Rao lower bound)
but more computationally expensive than least-squares.

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinograms
- `detector::PhotonCountingDetector`: Detector for spectral response

# Keyword Arguments
- `basis::Union{NTuple{M,Symbol}, Vector{Symbol}}`: Basis materials
- `energies::AbstractVector`: Spectrum energies (keV)
- `weights::AbstractVector`: Spectrum weights (normalized photon fluence)
- `max_iterations::Int=20`: Maximum Newton-Raphson iterations
- `max_keV::Float64=120.0`: Maximum energy (tube kVp)
- `I0::Float64=1e6`: Reference photon count

# Physics

Maximizes the Poisson log-likelihood:
    L(ρ) = Σ_b [ N_b × log(N̄_b(ρ)) - N̄_b(ρ) ]

where N̄_b(ρ) is the expected count in bin b given material densities ρ:
    N̄_b(ρ) = Σ_E R(E,b) × S(E) × η(E) × exp(-Σ_j ρ_j × μ_j(E) × L)

Uses Newton-Raphson iteration with the Hessian of the log-likelihood.

# References
- Roessl & Proksa (2007), "K-edge imaging in x-ray computed tomography..."
- Alvarez & Macovski (1976), "Energy-selective reconstructions in X-ray CT"
"""
function pcct_material_decomposition_mle(
    sino::EnergyResolvedSinogram{T,A},
    detector::PhotonCountingDetector;
    basis::Union{NTuple{M,Symbol} where M, Vector{Symbol}} = (:water, :iodine),
    energies::AbstractVector = Float64[],
    weights::AbstractVector = Float64[],
    max_iterations::Int = 20,
    max_keV::Float64 = 120.0,
    I0::Float64 = 1e6
) where {T, A}

    basis_vec = basis isa Tuple ? collect(Symbol, basis) : Vector{Symbol}(basis)
    n_bins = n_energy_bins(sino)
    n_materials = length(basis_vec)

    if n_materials >= n_bins
        error("Number of basis materials ($n_materials) must be less than number of energy bins ($n_bins)")
    end

    if isempty(energies) || isempty(weights)
        error("MLE decomposition requires energies and weights (tube spectrum)")
    end

    thresholds = sino.thresholds_keV
    n_cols, n_rows, n_angles = size(sino)

    # Pre-compute material attenuation coefficients at each energy
    # μ_mat[j, e] = μ_j(E_e) for material j at energy e
    n_energies = length(energies)
    μ_mat = zeros(Float64, n_materials, n_energies)
    for j in 1:n_materials
        for e in 1:n_energies
            μ_mat[j, e] = get_material_attenuation_pcct(basis_vec[j], Float64(energies[e]))
        end
    end

    # Pre-compute quantum efficiency
    η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)

    # Pre-compute spectral response matrix R(E,b) or use ideal binning
    R = compute_spectral_response_matrix(
        detector.material, detector.thickness_mm, collect(Float64, thresholds),
        max_keV;
        energy_resolution_keV=detector.energy_resolution_keV,
        n_energy_points=n_energies
    )

    # Energy grid for R matrix mapping
    R_energy_grid = collect(range(1.0, max_keV, length=n_energies))

    # Pre-compute bin assignment or R-mapping for each spectral energy
    # Maps each energy in `energies` to the closest R matrix row
    r_indices = zeros(Int, n_energies)
    for e in 1:n_energies
        r_indices[e] = clamp(
            round(Int, (Float64(energies[e]) - 1.0) / (max_keV - 1.0) * (n_energies - 1)) + 1,
            1, n_energies
        )
    end

    # Pre-compute S(E) × η(E) for each energy
    Sη = zeros(Float64, n_energies)
    for e in 1:n_energies
        Sη[e] = Float64(weights[e]) * η[e]
    end

    # Get initial estimate from polynomial decomposition (warm start for MLE)
    init_map = pcct_material_decomposition(sino; basis=Tuple(basis_vec), method=:least_squares, max_keV=max_keV)

    # Allocate output materials on CPU (transfer after)
    material_maps = [similar(sino.bins[1]) for _ in 1:n_materials]

    # Work on CPU for MLE (iterative per-pixel)
    bins_cpu = [Array(b) for b in sino.bins]
    init_cpu = [Array(m) for m in init_map.materials]

    # Per-pixel Newton-Raphson MLE
    n_pixels = length(bins_cpu[1])
    for idx in 1:n_pixels
        # Measured bin values (line-integral domain)
        p = zeros(Float64, n_bins)
        for b in 1:n_bins
            p[b] = Float64(bins_cpu[b][idx])
        end

        # Initial guess from polynomial decomposition
        ρ = zeros(Float64, n_materials)
        for j in 1:n_materials
            ρ[j] = Float64(init_cpu[j][idx])
        end

        # Newton-Raphson iterations
        for iter in 1:max_iterations
            # Compute expected counts per bin: N̄_b(ρ)
            N_bar = zeros(Float64, n_bins)
            # And gradient: ∂N̄_b/∂ρ_j
            grad_N = zeros(Float64, n_bins, n_materials)

            for e in 1:n_energies
                if Sη[e] < 1e-15
                    continue
                end

                # Line integral: Σ_j ρ_j × μ_j(E)
                line_integral = 0.0
                for j in 1:n_materials
                    line_integral += ρ[j] * μ_mat[j, e]
                end

                transmission = exp(-line_integral)
                r_idx = r_indices[e]

                for b in 1:n_bins
                    R_val = R[r_idx, b]
                    if R_val < 1e-15
                        continue
                    end
                    contribution = I0 * Sη[e] * R_val * transmission
                    N_bar[b] += contribution

                    # Gradient: ∂N̄_b/∂ρ_j = -μ_j(E) × contribution
                    for j in 1:n_materials
                        grad_N[b, j] -= μ_mat[j, e] * contribution
                    end
                end
            end

            # Compute Poisson log-likelihood gradient and Hessian
            # ∂L/∂ρ_j = Σ_b (N_b/N̄_b - 1) × ∂N̄_b/∂ρ_j
            # where N_b = I0 × exp(-p[b]) (measured counts from sinogram)
            gradient = zeros(Float64, n_materials)
            hessian = zeros(Float64, n_materials, n_materials)

            for b in 1:n_bins
                N_b_meas = I0 * exp(-p[b])  # Measured counts
                N_bar_b = max(N_bar[b], 1.0)  # Expected counts (floor)

                ratio = N_b_meas / N_bar_b - 1.0

                for j in 1:n_materials
                    gradient[j] += ratio * grad_N[b, j]
                end

                # Hessian (Fisher information approximation):
                # H[j,k] ≈ -Σ_b (1/N̄_b) × ∂N̄_b/∂ρ_j × ∂N̄_b/∂ρ_k
                for j in 1:n_materials
                    for k in j:n_materials
                        h_val = -(1.0 / N_bar_b) * grad_N[b, j] * grad_N[b, k]
                        hessian[j, k] += h_val
                        if k != j
                            hessian[k, j] += h_val
                        end
                    end
                end
            end

            # Newton step: Δρ = -H⁻¹ × ∇L
            # Use regularized Hessian for stability
            for j in 1:n_materials
                hessian[j, j] -= 1e-8  # Small negative definite regularization
            end

            # Solve via Cholesky or fallback to gradient step
            Δρ = try
                -hessian \ gradient
            catch
                # Fallback: steepest descent with small step
                0.01 * gradient
            end

            # Update with damping
            step_size = 1.0
            for j in 1:n_materials
                ρ[j] += step_size * Δρ[j]
            end

            # Check convergence
            if maximum(abs.(Δρ)) < 1e-6
                break
            end
        end

        # Store results
        for j in 1:n_materials
            bins_cpu[1][idx] = T(ρ[1])  # Temporary storage reuse
        end
        for j in 1:n_materials
            init_cpu[j][idx] = T(ρ[j])
        end
    end

    # Copy results to output (same device as input)
    for j in 1:n_materials
        copyto!(material_maps[j], init_cpu[j])
    end

    return PCCTMaterialMap{T,A}(material_maps, basis_vec, :projection)
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
function synthesize_vmi(material_map::PCCTMaterialMap{T,A}, energy_keV::Float64;
                        output=nothing,
                        ws_μ_values=nothing) where {T, A}
    n_materials = length(material_map.materials)
    @assert n_materials > 0 "Material map must have at least one material"

    # Get attenuation coefficient for each basis material at target energy
    μ_values = ws_μ_values !== nothing ? ws_μ_values : Vector{T}(undef, n_materials)
    for (i, mat_name) in enumerate(material_map.material_names)
        μ_values[i] = T(_get_basis_material_attenuation(mat_name, energy_keV))
    end

    # Synthesize VMI: μ_VMI = Σ ρᵢ × μᵢ(E)
    # Reuse pre-allocated output buffer if provided
    result = output === nothing ? similar(material_map.materials[1]) : output
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

IMPORTANT: This MUST return the same values as `get_material_attenuation_pcct`
to ensure consistency between the decomposition system matrix and VMI synthesis.
The decomposition solves: p = A × ρ, and synthesis computes: μ_VMI = Σ ρᵢ × μᵢ(E).
If A and μᵢ(E) use different basis functions, the result is physically wrong.
"""
function _get_basis_material_attenuation(material_name::Symbol, energy_keV::Float64)
    return get_material_attenuation_pcct(material_name, energy_keV)
end

# =============================================================================
# Exports
# =============================================================================

export PCCTMaterialMap, n_materials
export synthesize_vmi
export compute_bin_weights, pcct_virtual_monoenergetic, pcct_vmi_to_hu
export pcct_material_decomposition, pcct_material_decomposition_mle, get_material_attenuation_pcct
export get_gadolinium_solution_attenuation, get_gold_solution_attenuation
export K_EDGE_ENERGIES, compute_kedge_enhancement, get_kedge_sensitivity
export compute_effective_z
export get_supported_kedge_elements
