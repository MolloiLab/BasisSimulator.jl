# =============================================================================
# SIRT - Simultaneous Iterative Reconstruction Technique
# =============================================================================
#
# GPU-native implementation using AcceleratedKernels.jl
#
# Reference:
#   - TIGRE: CERN/TIGRE/MATLAB/Algorithms/SIRT.m
#   - Kak & Slaney, "Principles of Computerized Tomographic Imaging"
#
# Algorithm:
#   x_{k+1} = x_k + λ · V⁻¹ · Aᵀ · W · (b - A·x_k)
#
# Where:
#   - x is the reconstruction
#   - b is the measured sinogram
#   - A is the forward projection operator
#   - Aᵀ is the backprojection operator
#   - W = 1 / (A · 1) - projection domain weights (ray length normalization)
#   - V = 1 / (Aᵀ · 1) - image domain weights (voxel sensitivity)
#   - λ is the relaxation parameter
#
# =============================================================================

import AcceleratedKernels as AK

export sirt_reconstruct, sirt_reconstruct!

# =============================================================================
# Weight Computation
# =============================================================================

"""
    compute_projection_weights(geom, volume_size)

Compute projection domain weights W = 1 / (A · 1).

These weights normalize for ray length differences - longer rays get lower weights.
"""
function compute_projection_weights(
    geom::CTGeometry,
    volume_size::NTuple{3, Int},
    ::Type{T}
) where T <: AbstractFloat

    # Create ones volume
    ones_volume = ones(T, volume_size...)

    # Forward project ones to get ray lengths
    ray_sums = siddon_forward_project(ones_volume, geom)

    # W = 1 / ray_sums, with protection against division by zero
    eps = T(1e-8)

    AK.foreachindex(ray_sums) do idx
        val = ray_sums[idx]
        ray_sums[idx] = val > eps ? one(T) / val : zero(T)
    end

    return ray_sums
end

"""
    compute_image_weights(geom, volume_size)

Compute image domain weights V = 1 / (Aᵀ · 1).

These weights account for non-uniform voxel sensitivity in cone-beam geometry.
"""
function compute_image_weights(
    geom::CTGeometry,
    volume_size::NTuple{3, Int},
    ::Type{T}
) where T <: AbstractFloat

    # Create ones sinogram
    ones_sino = ones(T, geom.n_cols, geom.n_rows, geom.n_angles)

    # Backproject ones to get voxel sensitivities
    voxel_sums = backproject(ones_sino, geom, volume_size)

    # V_inv = 1 / voxel_sums, with protection against division by zero
    eps = T(1e-8)

    AK.foreachindex(voxel_sums) do idx
        val = voxel_sums[idx]
        voxel_sums[idx] = val > eps ? one(T) / val : zero(T)
    end

    return voxel_sums
end

# =============================================================================
# SIRT Iteration
# =============================================================================

"""
    sirt_iteration!(recon, sinogram, geom, W, V_inv, lambda)

Perform one SIRT iteration in-place.

x = x + λ · V⁻¹ · Aᵀ · W · (b - A·x)
"""
function sirt_iteration!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    W::AbstractArray{T, 3},
    V_inv::AbstractArray{T, 3},
    lambda::T
) where T <: AbstractFloat

    # Forward project current estimate: A·x
    projected = siddon_forward_project(recon, geom)

    # Compute residual: b - A·x
    # And apply projection weights: W · (b - A·x)
    AK.foreachindex(projected) do idx
        residual = sinogram[idx] - projected[idx]
        projected[idx] = W[idx] * residual
    end

    # Backproject weighted residual: Aᵀ · W · (b - A·x)
    correction = backproject(projected, geom, size(recon))

    # Apply image weights and update: x + λ · V⁻¹ · correction
    AK.foreachindex(recon) do idx
        recon[idx] += lambda * V_inv[idx] * correction[idx]
    end

    return recon
end

# =============================================================================
# High-Level Interface
# =============================================================================

"""
    sirt_reconstruct!(recon, sinogram, geom; niter=50, lambda=1.0, verbose=false)

In-place SIRT reconstruction.

# Arguments
- `recon`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `niter`: Number of iterations (default: 50)
- `lambda`: Relaxation parameter (default: 1.0)
- `verbose`: Print progress (default: false)

# Returns
The modified reconstruction array
"""
function sirt_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    niter::Int = 50,
    lambda::Real = 1.0,
    verbose::Bool = false
) where T <: AbstractFloat

    λ = T(lambda)
    volume_size = size(recon)

    # Pre-compute weights (expensive, but only done once)
    verbose && println("Computing SIRT weights...")

    # Transfer sinogram type info to weight computation
    W = compute_projection_weights(geom, volume_size, T)
    V_inv = compute_image_weights(geom, volume_size, T)

    # Transfer weights to same device as sinogram
    W_gpu = similar(sinogram, T, size(W)...)
    copyto!(W_gpu, W)
    V_inv_gpu = similar(recon, T, size(V_inv)...)
    copyto!(V_inv_gpu, V_inv)

    verbose && println("Running $niter SIRT iterations...")

    for iter in 1:niter
        sirt_iteration!(recon, sinogram, geom, W_gpu, V_inv_gpu, λ)

        if verbose && iter % 10 == 0
            println("  Iteration $iter/$niter")
        end
    end

    return recon
end

"""
    sirt_reconstruct(sinogram, geom, volume_size; niter=50, lambda=1.0, init=:zeros, verbose=false)

SIRT reconstruction with various initialization options.

# Arguments
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions
- `niter`: Number of iterations (default: 50)
- `lambda`: Relaxation parameter (default: 1.0)
- `init`: Initialization method - :zeros, :fdk, or an array (default: :zeros)
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Basic SIRT
recon = sirt_reconstruct(sinogram, geom, (128, 128, 64); niter=100)

# SIRT initialized with FDK
recon = sirt_reconstruct(sinogram, geom, (128, 128, 64); niter=50, init=:fdk)
```
"""
function sirt_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 50,
    lambda::Real = 1.0,
    init::Union{Symbol, AbstractArray} = :zeros,
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

    return sirt_reconstruct!(recon, sinogram, geom; niter=niter, lambda=lambda, verbose=verbose)
end
