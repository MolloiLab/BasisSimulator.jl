# =============================================================================
# Model-Based Iterative Reconstruction (MBIR)
# =============================================================================
#
# GPU-native implementation using AcceleratedKernels.jl
#
# This implements full model-based iterative reconstruction similar to:
#   - GE TrueFidelity
#   - Siemens ADMIRE/QIR
#
# Key features:
#   1. Ordered subsets for faster convergence
#   2. Physics-based edge-preserving regularization (Huber, Hyperbola)
#   3. 3D neighborhood analysis for structure/noise separation
#   4. Spectral MBIR for PCCT (QIR-style)
#   5. Configurable strength/quality levels
#
# Algorithm (OS-SQS):
#   minimize: Φ(x) = L(x) + λR(x)
#
# where:
#   - L(x) = data fidelity (penalized weighted least squares)
#   - R(x) = edge-preserving regularization (Huber/Hyperbola)
#   - Ordered subsets: process projection subsets sequentially
#
# References:
#   - Thibault, Sauer, Bouman, Hsieh: "A three-dimensional statistical approach
#     to improved image quality for multislice helical CT" (Med Phys 2007)
#   - Siemens ADMIRE white paper
#   - Siemens QIR for PCCT
#   - Geyer et al. "State of the Art: IR" (Radiology 2015)
#
# CRITICAL: All operations are GPU-native via AK.foreachindex()
#
# =============================================================================

import AcceleratedKernels as AK

export mbir_reconstruct, mbir_reconstruct!
export admire_style_reconstruct, qir_spectral_reconstruct
export HyperbolaPenalty, compute_hyperbola_penalty, compute_hyperbola_gradient!
export MBIRStrengthLevel, get_mbir_strength_params
export create_ordered_subsets, os_sqs_iteration!
export compute_3d_neighborhood_weights, compute_adaptive_regularization_gradient!
export create_subset_geometry, extract_subset_sinogram

# =============================================================================
# Hyperbola Penalty (Smooth TV Approximation)
# =============================================================================

"""
    HyperbolaPenalty <: PenaltyType

Hyperbola penalty function for edge-preserving regularization.

    ψ(t) = √(t² + ε²) - ε

This is a smooth (differentiable everywhere) approximation to total variation:
- Behaves like |t| for large |t| (preserves edges)
- Behaves like t²/(2ε) for small |t| (smooth penalty for noise)

The ε parameter controls the edge threshold:
- Smaller ε → sharper edges but potentially staircasing
- Larger ε → smoother transitions

# References
- Charbonnier et al. "Two deterministic half-quadratic regularization algorithms
  for computed imaging" (ICIP 1994)
- Thibault et al. "A three-dimensional statistical approach to improved image
  quality for multislice helical CT" (Med Phys 2007)
"""
struct HyperbolaPenalty <: PenaltyType
    epsilon::Float32  # Edge threshold (corner rounding)
end

HyperbolaPenalty() = HyperbolaPenalty(0.01f0)

"""
    compute_hyperbola_penalty(x, epsilon)

Compute hyperbola penalty value: R(x) = Σ [√(|∇x|² + ε²) - ε]

Uses 6-connected neighborhood in 3D with forward differences.
"""
function compute_hyperbola_penalty(
    x::AbstractArray{T, 3},
    epsilon::Real
) where T <: AbstractFloat

    ε = T(epsilon)
    nx, ny, nz = size(x)
    penalty_vals = similar(x)
    backend = AK.get_backend(x)

    AK.foreachindex(penalty_vals, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        val = x[i, j, k]

        # Forward differences
        dx = (i < nx) ? (x[i+1, j, k] - val) : zero(T)
        dy = (j < ny) ? (x[i, j+1, k] - val) : zero(T)
        dz = (k < nz) ? (x[i, j, k+1] - val) : zero(T)

        # Hyperbola penalty at this voxel: √(|∇x|² + ε²) - ε
        grad_sq = dx*dx + dy*dy + dz*dz
        penalty_vals[linear_idx] = sqrt(grad_sq + ε*ε) - ε
    end

    return AK.mapreduce(identity, +, penalty_vals; init=zero(T))
end

"""
    compute_hyperbola_gradient!(grad, x, epsilon)

Compute gradient of hyperbola penalty in-place.

    ∇ψ(t) = t / √(t² + ε²)

The gradient naturally becomes small for large differences (edges)
and approaches t/ε for small differences (noise), providing
edge-preserving behavior.
"""
function compute_hyperbola_gradient!(
    grad::AbstractArray{T, 3},
    x::AbstractArray{T, 3},
    epsilon::Real
) where T <: AbstractFloat

    ε = T(epsilon)
    nx, ny, nz = size(x)
    backend = AK.get_backend(x)

    AK.foreachindex(grad, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        # Compute divergence of normalized gradient field
        # Similar to isotropic TV but with hyperbola normalization
        grad[linear_idx] = _hyperbola_gradient_point(x, i, j, k, nx, ny, nz, ε)
    end

    return grad
end

@inline function _hyperbola_gradient_point(
    x::AbstractArray{T, 3},
    i::Int, j::Int, k::Int,
    nx::Int, ny::Int, nz::Int,
    ε::T
) where T <: AbstractFloat

    x_ijk = x[i, j, k]

    # Forward differences at (i,j,k)
    dx_ijk = (i < nx) ? (x[i+1, j, k] - x_ijk) : zero(T)
    dy_ijk = (j < ny) ? (x[i, j+1, k] - x_ijk) : zero(T)
    dz_ijk = (k < nz) ? (x[i, j, k+1] - x_ijk) : zero(T)

    # Hyperbola normalization at (i,j,k)
    norm_ijk = sqrt(dx_ijk*dx_ijk + dy_ijk*dy_ijk + dz_ijk*dz_ijk + ε*ε)

    # Normalized gradient components at (i,j,k)
    px_ijk = dx_ijk / norm_ijk
    py_ijk = dy_ijk / norm_ijk
    pz_ijk = dz_ijk / norm_ijk

    # Normalized gradients at neighboring points for divergence
    px_im1 = zero(T)
    if i > 1
        x_im1jk = x[i-1, j, k]
        dx_im1 = x_ijk - x_im1jk
        dy_im1 = (j < ny) ? (x[i-1, j+1, k] - x_im1jk) : zero(T)
        dz_im1 = (k < nz) ? (x[i-1, j, k+1] - x_im1jk) : zero(T)
        norm_im1 = sqrt(dx_im1*dx_im1 + dy_im1*dy_im1 + dz_im1*dz_im1 + ε*ε)
        px_im1 = dx_im1 / norm_im1
    end

    py_jm1 = zero(T)
    if j > 1
        x_ijm1k = x[i, j-1, k]
        dx_jm1 = (i < nx) ? (x[i+1, j-1, k] - x_ijm1k) : zero(T)
        dy_jm1 = x_ijk - x_ijm1k
        dz_jm1 = (k < nz) ? (x[i, j-1, k+1] - x_ijm1k) : zero(T)
        norm_jm1 = sqrt(dx_jm1*dx_jm1 + dy_jm1*dy_jm1 + dz_jm1*dz_jm1 + ε*ε)
        py_jm1 = dy_jm1 / norm_jm1
    end

    pz_km1 = zero(T)
    if k > 1
        x_ijkm1 = x[i, j, k-1]
        dx_km1 = (i < nx) ? (x[i+1, j, k-1] - x_ijkm1) : zero(T)
        dy_km1 = (j < ny) ? (x[i, j+1, k-1] - x_ijkm1) : zero(T)
        dz_km1 = x_ijk - x_ijkm1
        norm_km1 = sqrt(dx_km1*dx_km1 + dy_km1*dy_km1 + dz_km1*dz_km1 + ε*ε)
        pz_km1 = dz_km1 / norm_km1
    end

    # Divergence: div(p) = ∂px/∂x + ∂py/∂y + ∂pz/∂z (backward differences)
    div_p = (px_ijk - px_im1) + (py_ijk - py_jm1) + (pz_ijk - pz_km1)

    # Gradient = -div(p)
    return -div_p
end

# =============================================================================
# 3D Neighborhood Analysis (ADMIRE-style)
# =============================================================================

"""
    compute_3d_neighborhood_weights(x, radius=1; threshold=0.0)

Compute adaptive weights based on 3D neighborhood analysis.

This implements the ADMIRE-style neighborhood analysis that separates
anatomical structures from noise by examining local intensity variations.

# Arguments
- `x`: Current reconstruction estimate [nx, ny, nz]
- `radius`: Neighborhood radius (default: 1, gives 3×3×3 neighborhood)
- `threshold`: Edge detection threshold (default: 0.0, auto-computed)

# Returns
Edge-preserving weight map [nx, ny, nz] with values in [0, 1]:
- Near 1 at edges (high local variance) → less regularization
- Near 0 in smooth regions → more regularization
"""
function compute_3d_neighborhood_weights(
    x::AbstractArray{T, 3};
    radius::Int = 1,
    threshold::Real = 0.0
) where T <: AbstractFloat

    nx, ny, nz = size(x)
    weights = similar(x)
    backend = AK.get_backend(x)

    # Compute local variance at each voxel
    local_vars = similar(x)

    AK.foreachindex(local_vars, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        # Compute local mean and variance in neighborhood
        sum_val = zero(T)
        sum_sq = zero(T)
        count = 0

        for di in -radius:radius
            ii = i + di
            (ii < 1 || ii > nx) && continue
            for dj in -radius:radius
                jj = j + dj
                (jj < 1 || jj > ny) && continue
                for dk in -radius:radius
                    kk = k + dk
                    (kk < 1 || kk > nz) && continue

                    val = x[ii, jj, kk]
                    sum_val += val
                    sum_sq += val * val
                    count += 1
                end
            end
        end

        n = T(count)
        mean_val = sum_val / n
        local_vars[linear_idx] = sum_sq / n - mean_val * mean_val
    end

    # Compute edge weights from local variance
    # Auto-compute threshold if not provided
    max_var = maximum(local_vars)

    # Compute threshold with stable type (fix GPU compilation issue)
    thresh::T = if threshold > zero(T)
        T(threshold)
    elseif max_var > T(1e-12)
        # Compute threshold based on variance statistics
        T(0.5) * sqrt(max_var)
    else
        # Constant image (no edges) - return zero weights (full regularization)
        fill!(weights, zero(T))
        return weights
    end

    # Ensure type stability for GPU kernel capture
    safe_thresh::T = max(thresh, T(1e-12))

    AK.foreachindex(weights, backend) do idx
        var = local_vars[idx]
        # Sigmoid-like function: high variance → high weight (edge)
        weights[idx] = one(T) - one(T) / (one(T) + sqrt(var) / safe_thresh)
    end

    return weights
end

"""
    compute_adaptive_regularization_gradient!(grad, x, edge_weights, penalty, lambda)

Compute adaptive regularization gradient modulated by edge weights.

# Arguments
- `grad`: Output gradient array (modified in place)
- `x`: Current reconstruction
- `edge_weights`: Edge-preserving weights from `compute_3d_neighborhood_weights`
- `penalty`: Penalty type (HuberPenalty or HyperbolaPenalty)
- `lambda`: Base regularization strength

# Notes
At edges (high weight): reduced regularization → preserved structure
In smooth regions (low weight): full regularization → noise reduction
"""
function compute_adaptive_regularization_gradient!(
    grad::AbstractArray{T, 3},
    x::AbstractArray{T, 3},
    edge_weights::AbstractArray{T, 3},
    penalty::PenaltyType,
    lambda::Real
) where T <: AbstractFloat

    λ = T(lambda)

    # Compute base penalty gradient
    if penalty isa HuberPenalty
        compute_huber_gradient!(grad, x, penalty.delta)
    elseif penalty isa HyperbolaPenalty
        compute_hyperbola_gradient!(grad, x, penalty.epsilon)
    elseif penalty isa QuadraticPenalty
        compute_quadratic_gradient!(grad, x)
    else
        error("Unsupported penalty type: $(typeof(penalty))")
    end

    # Modulate by edge weights: less regularization at edges
    backend = AK.get_backend(grad)
    AK.foreachindex(grad, backend) do idx
        # edge_weight near 1 → edge → reduce regularization
        # edge_weight near 0 → smooth → full regularization
        modulation = one(T) - edge_weights[idx]
        grad[idx] = λ * modulation * grad[idx]
    end

    return grad
end

# =============================================================================
# Ordered Subsets
# =============================================================================

"""
    create_ordered_subsets(n_angles::Int, n_subsets::Int) -> Vector{Vector{Int}}

Create ordered subsets of projection angles for OS-SQS algorithm.

# Arguments
- `n_angles`: Total number of projection angles
- `n_subsets`: Number of subsets (typically 8-24)

# Returns
Vector of vectors, each containing angle indices for one subset.
Subsets are distributed to maximize angular separation.

# Example
```julia
subsets = create_ordered_subsets(360, 12)  # 12 subsets of 30 angles each
```

# Notes
Ordered subsets accelerate convergence by a factor of ~n_subsets.
More subsets = faster but potentially less stable.
Recommended: 8-16 subsets for clinical use.
"""
function create_ordered_subsets(n_angles::Int, n_subsets::Int)
    subsets = [Int[] for _ in 1:n_subsets]

    for i in 1:n_angles
        # Distribute angles to maximize angular separation within each subset
        subset_idx = mod1(i, n_subsets)
        push!(subsets[subset_idx], i)
    end

    return subsets
end

"""
    extract_subset_sinogram(sinogram, angle_indices)

Extract subset of sinogram for given angle indices.
"""
function extract_subset_sinogram(
    sinogram::AbstractArray{T, 3},
    angle_indices::Vector{Int}
) where T <: AbstractFloat

    n_cols, n_rows, _ = size(sinogram)
    n_angles_subset = length(angle_indices)

    subset = similar(sinogram, T, n_cols, n_rows, n_angles_subset)

    for (i, angle_idx) in enumerate(angle_indices)
        subset[:, :, i] = sinogram[:, :, angle_idx]
    end

    return subset
end

"""
    create_subset_geometry(geom::CTGeometry, angle_indices::Vector{Int})

Create geometry for a subset of projection angles.

This extracts the pre-computed source/detector positions for the specified
angle indices to create a subset geometry suitable for ordered subsets iteration.
"""
function create_subset_geometry(geom::CTGeometry, angle_indices::Vector{Int})
    n_subset = length(angle_indices)

    # Extract subset of pre-computed arrays
    angles_subset = geom.angles[angle_indices]
    source_positions_subset = geom.source_positions[:, angle_indices]
    detector_centers_subset = geom.detector_centers[:, angle_indices]
    detector_u_subset = geom.detector_u[:, angle_indices]
    detector_v_subset = geom.detector_v[:, angle_indices]

    # Create new CTGeometry with subset data
    return CTGeometry(
        geom.SAD,
        geom.SDD,
        n_subset,
        geom.n_rows,
        geom.n_cols,
        geom.pixel_size,
        angles_subset,
        source_positions_subset,
        detector_centers_subset,
        detector_u_subset,
        detector_v_subset,
        geom.fov
    )
end

# =============================================================================
# OS-SQS Iteration (Ordered Subsets Separable Quadratic Surrogate)
# =============================================================================

"""
    os_sqs_iteration!(recon, sinogram, geom, subsets, W, V_inv, stat_weights,
                       reg_grad, edge_weights, penalty, lambda_reg, relaxation)

Perform one OS-SQS iteration (all subsets).

The ordered subsets approach processes projection data in subsets,
updating the image after each subset. This accelerates convergence
by a factor of approximately n_subsets compared to standard SIRT.

# Algorithm
For each subset s:
    1. Compute forward projection for subset: A_s·x
    2. Compute residual: y_s - A_s·x
    3. Weight by statistical weights: W_s · stat_weights_s · residual
    4. Backproject weighted residual
    5. Add regularization gradient (modulated by edge weights)
    6. Update image: x = x + λ · correction
"""
function os_sqs_iteration!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    subsets::Vector{Vector{Int}},
    W::AbstractArray{T, 3},
    V_inv::AbstractArray{T, 3},
    stat_weights::AbstractArray{T, 3},
    reg_grad::AbstractArray{T, 3},
    edge_weights::AbstractArray{T, 3},
    penalty::PenaltyType,
    lambda_reg::T,
    relaxation::T
) where T <: AbstractFloat

    n_subsets = length(subsets)
    backend = AK.get_backend(recon)

    for (s, angle_indices) in enumerate(subsets)
        # Extract subset data
        sino_subset = extract_subset_sinogram(sinogram, angle_indices)
        geom_subset = create_subset_geometry(geom, angle_indices)
        W_subset = extract_subset_sinogram(W, angle_indices)
        stat_weights_subset = extract_subset_sinogram(stat_weights, angle_indices)

        # Forward project: A_s·x
        Ax_subset = siddon_forward_project(recon, geom_subset)

        # Compute weighted residual: (W_s ⊙ stat_weights_s) · (y_s - A_s·x)
        AK.foreachindex(Ax_subset, backend) do idx
            residual = sino_subset[idx] - Ax_subset[idx]
            Ax_subset[idx] = W_subset[idx] * stat_weights_subset[idx] * residual
        end

        # Backproject weighted residual: A_s^T · weighted_residual
        correction = backproject(Ax_subset, geom_subset, size(recon); weighted=false)

        # Compute adaptive regularization gradient
        compute_adaptive_regularization_gradient!(reg_grad, recon, edge_weights, penalty, lambda_reg)

        # Scale for subset: multiply by n_subsets to account for using only 1/n_subsets of data
        subset_scale = T(n_subsets)

        # Apply update: x = x + λ · V^{-1} · (subset_scale · correction - reg_grad)
        AK.foreachindex(recon, backend) do idx
            data_update = subset_scale * V_inv[idx] * correction[idx]
            reg_update = V_inv[idx] * reg_grad[idx]
            recon[idx] += relaxation * (data_update - reg_update)
        end
    end

    return recon
end

# =============================================================================
# MBIR Reconstruction (TrueFidelity/ADMIRE-style)
# =============================================================================

"""
    mbir_reconstruct!(recon, sinogram, geom; niter=30, n_subsets=12,
                       lambda=0.02, penalty=HyperbolaPenalty(), relaxation=1.0,
                       use_edge_weights=true, update_weights_interval=5, verbose=false)

In-place model-based iterative reconstruction (MBIR).

This implements TrueFidelity/ADMIRE-style reconstruction with:
- Ordered subsets for fast convergence
- Physics-based edge-preserving regularization
- 3D neighborhood analysis for structure preservation
- Statistical weighting (Poisson noise model)

# Arguments
- `recon`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram`: Measured sinogram (log-transformed) [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `niter`: Number of outer iterations (default: 30)
- `n_subsets`: Number of ordered subsets (default: 12)
- `lambda`: Regularization strength (default: 0.02)
- `penalty`: PenaltyType - HyperbolaPenalty(ε), HuberPenalty(δ), QuadraticPenalty() (default: HyperbolaPenalty())
- `relaxation`: Relaxation parameter (default: 1.0)
- `use_edge_weights`: Enable adaptive regularization (default: true)
- `update_weights_interval`: How often to update weights (default: 5)
- `verbose`: Print progress (default: false)

# Returns
The modified reconstruction array

# Notes
- MBIR provides superior image quality compared to statistical IR
- Ordered subsets accelerate convergence by ~n_subsets factor
- Edge-preserving regularization maintains anatomical detail
- Recommended: 20-50 iterations for convergence

# References
- Thibault et al., "A three-dimensional statistical approach to improved
  image quality for multislice helical CT" (Med Phys 2007)
- Siemens ADMIRE/TrueFidelity technical documentation
"""
function mbir_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    niter::Int = 30,
    n_subsets::Int = 12,
    lambda::Real = 0.02,
    penalty::PenaltyType = HyperbolaPenalty(),
    relaxation::Real = 1.0,
    use_edge_weights::Bool = true,
    update_weights_interval::Int = 5,
    verbose::Bool = false
) where T <: AbstractFloat

    λ = T(lambda)
    λ_relax = T(relaxation)
    volume_size = size(recon)

    verbose && println("Initializing MBIR reconstruction...")
    verbose && println("  Penalty: $(typeof(penalty))")
    verbose && println("  λ = $lambda, n_subsets = $n_subsets")
    verbose && println("  Edge-preserving: $use_edge_weights")

    # Create ordered subsets
    subsets = create_ordered_subsets(geom.n_angles, n_subsets)
    verbose && println("  Created $n_subsets subsets (~$(length(subsets[1])) angles each)")

    # Pre-compute normalization weights
    verbose && println("  Computing normalization weights...")
    W = compute_projection_weights(geom, volume_size, T)
    V_inv = compute_image_weights(geom, volume_size, T)

    # Transfer to same device
    W_gpu = similar(sinogram, T, size(W)...)
    copyto!(W_gpu, W)
    V_inv_gpu = similar(recon, T, size(V_inv)...)
    copyto!(V_inv_gpu, V_inv)

    # Initialize statistical weights (from sinogram)
    stat_weights = compute_simple_weights(sinogram)

    # Initialize edge weights
    edge_weights = use_edge_weights ?
        compute_3d_neighborhood_weights(recon) :
        similar(recon)
    if !use_edge_weights
        fill!(edge_weights, zero(T))  # No modulation
    end

    # Regularization gradient storage
    reg_grad = similar(recon)

    backend = AK.get_backend(recon)

    verbose && println("  Running $niter MBIR iterations...")

    for iter in 1:niter
        # Periodically update weights
        if iter == 1 || iter % update_weights_interval == 0
            # Update statistical weights
            stat_weights = compute_statistical_weights(sinogram, geom, recon)

            # Update edge weights
            if use_edge_weights
                edge_weights = compute_3d_neighborhood_weights(recon)
            end
        end

        # OS-SQS iteration (processes all subsets)
        os_sqs_iteration!(recon, sinogram, geom, subsets, W_gpu, V_inv_gpu,
                          stat_weights, reg_grad, edge_weights, penalty, λ, λ_relax)

        if verbose && iter % 5 == 0
            println("    Iteration $iter/$niter")
        end
    end

    return recon
end

"""
    mbir_reconstruct(sinogram, geom, volume_size; niter=30, n_subsets=12,
                      lambda=0.02, penalty=HyperbolaPenalty(), relaxation=1.0,
                      use_edge_weights=true, init=:fdk, verbose=false)

MBIR reconstruction with initialization options.

# Arguments
- `sinogram`: Measured sinogram (log-transformed) [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `niter`: Number of iterations (default: 30)
- `n_subsets`: Number of ordered subsets (default: 12)
- `lambda`: Regularization strength (default: 0.02)
- `penalty`: PenaltyType (default: HyperbolaPenalty())
- `relaxation`: Relaxation parameter (default: 1.0)
- `use_edge_weights`: Enable adaptive regularization (default: true)
- `init`: Initialization - :zeros, :fdk, or an array (default: :fdk)
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Standard MBIR with hyperbola penalty
recon = mbir_reconstruct(sinogram, geom, (256, 256, 128);
                         niter=30, lambda=0.02)

# MBIR with Huber penalty for stronger edge preservation
recon = mbir_reconstruct(sinogram, geom, (256, 256, 128);
                         niter=50, penalty=HuberPenalty(0.01f0))
```
"""
function mbir_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 30,
    n_subsets::Int = 12,
    lambda::Real = 0.02,
    penalty::PenaltyType = HyperbolaPenalty(),
    relaxation::Real = 1.0,
    use_edge_weights::Bool = true,
    update_weights_interval::Int = 5,
    init::Union{Symbol, AbstractArray} = :fdk,
    verbose::Bool = false
) where T <: AbstractFloat

    # Initialize reconstruction
    if init == :zeros
        recon = similar(sinogram, T, volume_size...)
        fill!(recon, zero(T))
    elseif init == :fdk
        verbose && println("Initializing with FDK...")
        recon = fdk_reconstruct(sinogram, geom, volume_size)
    elseif init isa AbstractArray
        recon = similar(sinogram, T, volume_size...)
        copyto!(recon, T.(init))
    else
        error("init must be :zeros, :fdk, or an array")
    end

    return mbir_reconstruct!(recon, sinogram, geom;
                              niter=niter, n_subsets=n_subsets,
                              lambda=lambda, penalty=penalty,
                              relaxation=relaxation, use_edge_weights=use_edge_weights,
                              update_weights_interval=update_weights_interval,
                              verbose=verbose)
end

# =============================================================================
# ADMIRE-Style Interface (Strength Levels)
# =============================================================================

"""
    MBIRStrengthLevel

MBIR strength levels similar to ADMIRE 1-5:

| Level | Noise Reduction | Edge Preservation | Clinical Use |
|-------|-----------------|-------------------|--------------|
| 1 | Minimal (~15%) | Maximum | Texture-critical imaging |
| 2 | Light (~25%) | High | General imaging |
| 3 | Standard (~35%) | Good | Most clinical use |
| 4 | Strong (~45%) | Moderate | Higher noise reduction |
| 5 | Maximum (~55%) | Lower | Maximum dose reduction |

Maps to (n_subsets, lambda, niter, penalty_param) tuples.
"""
struct MBIRStrengthLevel
    level::Int
    function MBIRStrengthLevel(level::Int)
        1 ≤ level ≤ 5 || error("MBIR strength level must be 1-5, got $level")
        new(level)
    end
end

Base.convert(::Type{MBIRStrengthLevel}, x::Int) = MBIRStrengthLevel(x)

"""
    get_mbir_strength_params(level::Union{Int, MBIRStrengthLevel})

Get optimized MBIR parameters for a given strength level.

Returns named tuple with:
- `n_subsets`: Number of ordered subsets
- `lambda`: Regularization strength
- `niter`: Number of iterations
- `epsilon`: Hyperbola penalty parameter
- `use_edge_weights`: Whether to use adaptive regularization

# Example
```julia
params = get_mbir_strength_params(3)
# (n_subsets=12, lambda=0.025, niter=30, epsilon=0.01, use_edge_weights=true)
```
"""
function get_mbir_strength_params(level::Union{Int, MBIRStrengthLevel})
    l = level isa MBIRStrengthLevel ? level.level : level

    # Optimized parameters based on ADMIRE/TrueFidelity behavior
    params = Dict(
        1 => (n_subsets=16, lambda=0.01,  niter=20, epsilon=0.02f0,  use_edge_weights=true),
        2 => (n_subsets=14, lambda=0.015, niter=25, epsilon=0.015f0, use_edge_weights=true),
        3 => (n_subsets=12, lambda=0.025, niter=30, epsilon=0.01f0,  use_edge_weights=true),
        4 => (n_subsets=10, lambda=0.04,  niter=40, epsilon=0.008f0, use_edge_weights=true),
        5 => (n_subsets=8,  lambda=0.06,  niter=50, epsilon=0.005f0, use_edge_weights=false),
    )

    return params[l]
end

"""
    admire_style_reconstruct(sinogram, geom, volume_size; strength=3, verbose=false)

ADMIRE-style MBIR reconstruction using clinical strength levels (1-5).

This provides a simplified, clinically-oriented interface similar to
Siemens ADMIRE. Just specify a strength level and get optimized parameters.

# Arguments
- `sinogram`: Measured sinogram (log-transformed) [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `strength`: Noise reduction level 1-5 (default: 3)
  - 1: Minimal (preserves texture, like FBP)
  - 2: Light
  - 3: Standard clinical (recommended)
  - 4: Strong
  - 5: Maximum (most noise reduction)
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Standard ADMIRE-3 equivalent
recon = admire_style_reconstruct(sinogram, geom, (256, 256, 128); strength=3)

# Maximum noise reduction (ADMIRE-5)
recon = admire_style_reconstruct(sinogram, geom, (256, 256, 128); strength=5)
```

# Clinical Guidelines (similar to ADMIRE)
- **Level 1**: Lung nodule evaluation, texture-critical imaging
- **Level 2-3**: General clinical imaging
- **Level 4**: Higher dose reduction needed
- **Level 5**: Maximum dose reduction (may affect fine detail)

See also: [`mbir_reconstruct`](@ref), [`qir_spectral_reconstruct`](@ref)
"""
function admire_style_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    strength::Union{Int, MBIRStrengthLevel} = 3,
    verbose::Bool = false
) where T <: AbstractFloat

    params = get_mbir_strength_params(strength)

    verbose && println("ADMIRE-style MBIR (Level $(strength isa MBIRStrengthLevel ? strength.level : strength))...")
    verbose && println("  Parameters: n_subsets=$(params.n_subsets), λ=$(params.lambda), niter=$(params.niter)")

    penalty = HyperbolaPenalty(params.epsilon)

    return mbir_reconstruct(sinogram, geom, volume_size;
                            niter=params.niter,
                            n_subsets=params.n_subsets,
                            lambda=params.lambda,
                            penalty=penalty,
                            use_edge_weights=params.use_edge_weights,
                            init=:fdk,
                            verbose=verbose)
end

# =============================================================================
# QIR-Style Spectral MBIR for PCCT
# =============================================================================

"""
    qir_spectral_reconstruct(energy_bin_sinograms, geom, volume_size;
                              strength=3, combine_structural=true, verbose=false)

QIR-style spectral MBIR for photon-counting CT.

This implements Siemens QIR-style reconstruction for PCCT data:
- Multi-spectral consistency: All energy bins share structural priors
- Spectral-aware regularization: Uses information from all bins
- Energy-specific noise handling: Different weights per bin

# Arguments
- `energy_bin_sinograms`: Vector of sinograms, one per energy bin [n_bins]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `strength`: QIR strength level 1-4 (default: 3)
  - 1: Minimal (~22% noise reduction)
  - 2: Light (~41% noise reduction)
  - 3: Standard (~57% noise reduction)
  - 4: Maximum (~71% noise reduction)
- `combine_structural`: Use shared structural prior from all bins (default: true)
- `verbose`: Print progress (default: false)

# Returns
Vector of reconstructed volumes, one per energy bin

# Notes
QIR differs from standard MBIR:
- Uses spectral information to enhance regularization
- Structural features are assumed consistent across spectra
- Provides superior noise reduction for PCCT data

# Example
```julia
# Reconstruct all energy bins with QIR-3
bin_recons = qir_spectral_reconstruct(energy_bins, geom, (256, 256, 128); strength=3)

# VMI can then be computed from bin_recons
```

# References
- Siemens QIR white paper
- PMC10321251: Photon-counting detector CT review
"""
function qir_spectral_reconstruct(
    energy_bin_sinograms::Vector{<:AbstractArray{T, 3}},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    strength::Int = 3,
    combine_structural::Bool = true,
    verbose::Bool = false
) where T <: AbstractFloat

    # Validate strength level (QIR uses 1-4)
    1 ≤ strength ≤ 4 || error("QIR strength level must be 1-4, got $strength")

    n_bins = length(energy_bin_sinograms)
    verbose && println("QIR spectral reconstruction (Level $strength, $n_bins bins)...")

    # QIR-specific parameters (different from ADMIRE)
    qir_params = Dict(
        1 => (lambda=0.015, niter=25, n_subsets=12),
        2 => (lambda=0.025, niter=30, n_subsets=10),
        3 => (lambda=0.04,  niter=40, n_subsets=8),
        4 => (lambda=0.06,  niter=50, n_subsets=6),
    )
    params = qir_params[strength]

    # Initialize reconstructions for each bin with FDK
    bin_recons = Vector{typeof(energy_bin_sinograms[1])}(undef, n_bins)
    for i in 1:n_bins
        verbose && println("  Initializing bin $i with FDK...")
        bin_recons[i] = fdk_reconstruct(energy_bin_sinograms[i], geom, volume_size)
    end

    # Compute shared structural prior from all bins
    structural_weights = nothing
    if combine_structural
        verbose && println("  Computing shared structural prior...")

        # Average edge weights across all bins for structural consistency
        combined_edges = similar(bin_recons[1])
        fill!(combined_edges, zero(T))

        for i in 1:n_bins
            edges_i = compute_3d_neighborhood_weights(bin_recons[i])
            combined_edges .+= edges_i
        end
        combined_edges ./= T(n_bins)
        structural_weights = combined_edges
    end

    # Reconstruct each bin with spectral-aware regularization
    for i in 1:n_bins
        verbose && println("  Reconstructing bin $i/$n_bins...")

        # Pre-compute weights for this bin
        W = compute_projection_weights(geom, volume_size, T)
        V_inv = compute_image_weights(geom, volume_size, T)
        W_gpu = similar(energy_bin_sinograms[i], T, size(W)...)
        copyto!(W_gpu, W)
        V_inv_gpu = similar(bin_recons[i], T, size(V_inv)...)
        copyto!(V_inv_gpu, V_inv)

        subsets = create_ordered_subsets(geom.n_angles, params.n_subsets)
        stat_weights = compute_simple_weights(energy_bin_sinograms[i])
        reg_grad = similar(bin_recons[i])

        # Use structural prior if available, otherwise compute locally
        edge_weights = structural_weights !== nothing ?
            structural_weights :
            compute_3d_neighborhood_weights(bin_recons[i])

        penalty = HyperbolaPenalty(0.01f0)
        λ = T(params.lambda)

        for iter in 1:params.niter
            os_sqs_iteration!(bin_recons[i], energy_bin_sinograms[i], geom,
                              subsets, W_gpu, V_inv_gpu, stat_weights,
                              reg_grad, edge_weights, penalty, λ, one(T))
        end
    end

    verbose && println("QIR reconstruction complete.")
    return bin_recons
end
