# =============================================================================
# CGLS - Conjugate Gradient Least Squares
# =============================================================================
#
# GPU-native implementation using AcceleratedKernels.jl
#
# Reference:
#   - TIGRE: CERN/TIGRE/MATLAB/Algorithms/CGLS.m
#   - Björck, "Numerical Methods for Least Squares Problems"
#
# Algorithm (solves min ||Ax - b||²):
#   1. r = b - A·x
#   2. p = Aᵀ·r
#   3. γ = ||p||²
#   4. Loop:
#      - q = A·p
#      - α = γ / ||q||²
#      - x = x + α·p
#      - r = r - α·q
#      - s = Aᵀ·r
#      - γ₁ = ||s||²
#      - β = γ₁ / γ
#      - p = s + β·p
#      - γ = γ₁
#
# =============================================================================

import AcceleratedKernels as AK

export cgls_reconstruct, cgls_reconstruct!

# =============================================================================
# Norm Computation (GPU-friendly)
# =============================================================================

"""
    compute_norm_squared(x)

Compute ||x||² using AcceleratedKernels.jl reduction.
"""
function compute_norm_squared(x::AbstractArray{T}) where T <: AbstractFloat
    # Use mapreduce for GPU-compatible norm computation
    return AK.mapreduce(v -> v * v, +, x; init=zero(T))
end

# =============================================================================
# CGLS Iteration
# =============================================================================

"""
    cgls_iteration!(x, r, p, gamma, sinogram, geom)

Perform one CGLS iteration, returning new gamma value.

Updates x, r, p in place.
"""
function cgls_iteration!(
    x::AbstractArray{T, 3},
    r::AbstractArray{T, 3},
    p::AbstractArray{T, 3},
    gamma::T,
    geom::CTGeometry
) where T <: AbstractFloat

    # q = A·p (forward project search direction)
    q = siddon_forward_project(p, geom)

    # α = γ / ||q||²
    q_norm_sq = compute_norm_squared(q)
    alpha = gamma / max(q_norm_sq, T(1e-12))

    # x = x + α·p (update solution)
    AK.foreachindex(x) do idx
        x[idx] += alpha * p[idx]
    end

    # r = r - α·q (update residual)
    AK.foreachindex(r) do idx
        r[idx] -= alpha * q[idx]
    end

    # s = Aᵀ·r (backproject residual)
    s = backproject(r, geom, size(x))

    # γ₁ = ||s||²
    gamma_new = compute_norm_squared(s)

    # β = γ₁ / γ
    beta = gamma_new / max(gamma, T(1e-12))

    # p = s + β·p (update search direction)
    AK.foreachindex(p) do idx
        p[idx] = s[idx] + beta * p[idx]
    end

    return gamma_new
end

# =============================================================================
# High-Level Interface
# =============================================================================

"""
    cgls_reconstruct!(recon, sinogram, geom; niter=20, verbose=false)

In-place CGLS reconstruction.

# Arguments
- `recon`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `niter`: Number of iterations (default: 20)
- `verbose`: Print progress (default: false)

# Returns
The modified reconstruction array

# Notes
CGLS typically converges faster than SIRT but may exhibit semi-convergence
(error increases after optimal point). For noisy data, fewer iterations
(10-20) often give better results than many iterations.
"""
function cgls_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    niter::Int = 20,
    verbose::Bool = false
) where T <: AbstractFloat

    verbose && println("Initializing CGLS...")

    # r = b - A·x (initial residual)
    projected = siddon_forward_project(recon, geom)
    r = similar(sinogram)
    AK.foreachindex(r) do idx
        r[idx] = sinogram[idx] - projected[idx]
    end

    # p = Aᵀ·r (initial search direction)
    p = backproject(r, geom, size(recon))

    # γ = ||p||²
    gamma = compute_norm_squared(p)

    verbose && println("Running $niter CGLS iterations...")

    for iter in 1:niter
        gamma = cgls_iteration!(recon, r, p, gamma, geom)

        if verbose && iter % 5 == 0
            residual_norm = sqrt(compute_norm_squared(r))
            println("  Iteration $iter/$niter, ||r|| = $(round(residual_norm, digits=4))")
        end

        # Check for convergence (gamma → 0 means gradient is zero)
        if gamma < T(1e-20)
            verbose && println("  Converged at iteration $iter")
            break
        end
    end

    return recon
end

"""
    cgls_reconstruct(sinogram, geom, volume_size; niter=20, init=:zeros, verbose=false)

CGLS reconstruction with various initialization options.

# Arguments
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions
- `niter`: Number of iterations (default: 20)
- `init`: Initialization method - :zeros, :fdk, or an array (default: :zeros)
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Basic CGLS (typically fewer iterations than SIRT)
recon = cgls_reconstruct(sinogram, geom, (128, 128, 64); niter=15)

# CGLS initialized with FDK
recon = cgls_reconstruct(sinogram, geom, (128, 128, 64); niter=10, init=:fdk)
```

# Notes
CGLS minimizes ||Ax - b||² directly using conjugate gradients.
It typically converges faster than SIRT but may exhibit semi-convergence
behavior where the error increases after an optimal iteration count.
For noisy data, 10-20 iterations often works well.
"""
function cgls_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 20,
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

    return cgls_reconstruct!(recon, sinogram, geom; niter=niter, verbose=verbose)
end
