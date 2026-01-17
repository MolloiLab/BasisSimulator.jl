# =============================================================================
# CGLS - Conjugate Gradient Least Squares
# =============================================================================
#
# GPU-native implementation using AcceleratedKernels.jl
#
# Reference:
#   - TIGRE: CERN/TIGRE/MATLAB/Algorithms/CGLS.m
#   - Björck, "Numerical Methods for Least Squares Problems"
#   - Hansen, "Discrete Inverse Problems: Insight and Algorithms" (Tikhonov)
#
# Standard CGLS (solves min ||Ax - b||²):
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
# Tikhonov-Regularized CGLS (solves min ||Ax - b||² + λ||x||²):
#   - Modified gradient includes regularization term
#   - s = Aᵀ·r - λ·x
#   - Provides noise regularization while preserving CGLS convergence
#
# CRITICAL: CGLS requires matched/unweighted backprojection (weighted=false).
# Using FDK-weighted backprojection would violate the adjoint property A*A'.
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
    cgls_iteration!(x, r, p, gamma, geom; lambda=0.0)

Perform one CGLS iteration, returning new gamma value.

Updates x, r, p in place.

# Arguments
- `x`: Current reconstruction estimate [nx, ny, nz]
- `r`: Current residual [n_cols, n_rows, n_angles]
- `p`: Current search direction [nx, ny, nz]
- `gamma`: Current gamma value (||Aᵀr||² or ||Aᵀr - λx||²)
- `geom`: CT geometry

# Keyword Arguments
- `lambda`: Tikhonov regularization parameter (default: 0.0)

# Returns
New gamma value for next iteration

# Notes
Uses matched/unweighted backprojection (weighted=false) to ensure A and Aᵀ
form a proper adjoint pair, which is essential for CGLS convergence.
"""
function cgls_iteration!(
    x::AbstractArray{T, 3},
    r::AbstractArray{T, 3},
    p::AbstractArray{T, 3},
    gamma::T,
    geom::CTGeometry;
    lambda::T = zero(T)
) where T <: AbstractFloat

    # q = A·p (forward project search direction)
    q = siddon_forward_project(p, geom)

    # For regularized CGLS, augment q with regularization term
    # The effective system is [A; √λI], so ||[A; √λI]·p||² = ||A·p||² + λ||p||²
    q_norm_sq = compute_norm_squared(q)
    if lambda > zero(T)
        p_norm_sq = compute_norm_squared(p)
        q_norm_sq += lambda * p_norm_sq
    end

    # α = γ / ||q||² (or augmented norm)
    alpha = gamma / max(q_norm_sq, T(1e-12))

    # x = x + α·p (update solution)
    AK.foreachindex(x) do idx
        x[idx] += alpha * p[idx]
    end

    # r = r - α·q (update residual in data space)
    AK.foreachindex(r) do idx
        r[idx] -= alpha * q[idx]
    end

    # s = Aᵀ·r (backproject residual)
    # CRITICAL: Use matched backprojection (weighted=false) for correct adjoint
    s = backproject(r, geom, size(x); weighted=false)

    # For regularized CGLS, subtract λ·x from gradient
    # This implements min ||Ax - b||² + λ||x||²
    if lambda > zero(T)
        AK.foreachindex(s) do idx
            s[idx] -= lambda * x[idx]
        end
    end

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
    cgls_reconstruct!(recon, sinogram, geom; niter=20, lambda=0.0, tol=0.0, verbose=false)

In-place CGLS reconstruction with optional Tikhonov regularization.

# Arguments
- `recon`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `niter`: Maximum number of iterations (default: 20)
- `lambda`: Tikhonov regularization parameter (default: 0.0, no regularization)
            Higher values provide more noise suppression but may reduce resolution.
            Typical range: 1e-6 to 1e-2 depending on noise level.
- `tol`: Convergence tolerance for relative residual reduction (default: 0.0)
         If > 0, stops when ||r_k|| / ||r_0|| < tol
- `verbose`: Print progress (default: false)

# Returns
The modified reconstruction array

# Notes
- CGLS typically converges faster than SIRT but may exhibit semi-convergence
  (reconstruction error increases after optimal iteration count) for noisy data.
- For noisy data without regularization, fewer iterations (10-20) often work better.
- Tikhonov regularization (lambda > 0) can prevent semi-convergence and allow
  more iterations without degradation.
- Uses matched/unweighted backprojection (weighted=false) to ensure correct
  adjoint property for the conjugate gradient algorithm.

# Example
```julia
# Basic CGLS
recon = cgls_reconstruct!(zeros(Float32, 128,128,64), sinogram, geom; niter=15)

# Regularized CGLS for noisy data
recon = cgls_reconstruct!(zeros(Float32, 128,128,64), sinogram, geom;
                          niter=50, lambda=1e-4)

# With convergence tolerance
recon = cgls_reconstruct!(zeros(Float32, 128,128,64), sinogram, geom;
                          niter=100, tol=1e-4)
```
"""
function cgls_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    niter::Int = 20,
    lambda::Real = 0.0,
    tol::Real = 0.0,
    verbose::Bool = false
) where T <: AbstractFloat

    λ = T(lambda)
    tolerance = T(tol)

    verbose && println("Initializing CGLS$(lambda > 0 ? " (Tikhonov λ=$lambda)" : "")...")

    # r = b - A·x (initial residual in data space)
    projected = siddon_forward_project(recon, geom)
    r = similar(sinogram)
    AK.foreachindex(r) do idx
        r[idx] = sinogram[idx] - projected[idx]
    end

    # Store initial residual norm for convergence check
    initial_residual_norm = sqrt(compute_norm_squared(r))

    # p = Aᵀ·r (initial search direction)
    # CRITICAL: Use matched backprojection (weighted=false) for correct adjoint
    p = backproject(r, geom, size(recon); weighted=false)

    # For regularized CGLS, modify initial gradient
    if λ > zero(T)
        AK.foreachindex(p) do idx
            p[idx] -= λ * recon[idx]
        end
    end

    # γ = ||p||²
    gamma = compute_norm_squared(p)

    verbose && println("Running up to $niter CGLS iterations...")
    verbose && println("  Initial ||r|| = $(round(initial_residual_norm, digits=4))")

    iterations_completed = 0

    for iter in 1:niter
        gamma = cgls_iteration!(recon, r, p, gamma, geom; lambda=λ)
        iterations_completed = iter

        if verbose && iter % 5 == 0
            residual_norm = sqrt(compute_norm_squared(r))
            rel_residual = residual_norm / max(initial_residual_norm, T(1e-12))
            println("  Iteration $iter/$niter, ||r|| = $(round(residual_norm, digits=4)), rel = $(round(rel_residual, digits=6))")
        end

        # Check for convergence based on gradient (gamma → 0)
        if gamma < T(1e-20)
            verbose && println("  Converged (gradient) at iteration $iter")
            break
        end

        # Check for convergence based on relative residual
        if tolerance > zero(T)
            residual_norm = sqrt(compute_norm_squared(r))
            rel_residual = residual_norm / max(initial_residual_norm, T(1e-12))
            if rel_residual < tolerance
                verbose && println("  Converged (tolerance) at iteration $iter, rel_residual = $(round(rel_residual, digits=6))")
                break
            end
        end
    end

    verbose && println("Completed $iterations_completed iterations")

    return recon
end

"""
    cgls_reconstruct(sinogram, geom, volume_size; niter=20, lambda=0.0, tol=0.0, init=:zeros, verbose=false)

CGLS reconstruction with optional Tikhonov regularization and various initialization options.

# Arguments
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `niter`: Maximum number of iterations (default: 20)
- `lambda`: Tikhonov regularization parameter (default: 0.0, no regularization)
            Higher values provide more noise suppression but may reduce resolution.
            Typical range: 1e-6 to 1e-2 depending on noise level.
- `tol`: Convergence tolerance for relative residual reduction (default: 0.0)
         If > 0, stops when ||r_k|| / ||r_0|| < tol
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

# Regularized CGLS for noisy data
recon = cgls_reconstruct(sinogram, geom, (128, 128, 64); niter=50, lambda=1e-4)

# With convergence tolerance
recon = cgls_reconstruct(sinogram, geom, (128, 128, 64); niter=100, tol=1e-4)
```

# Notes
CGLS minimizes ||Ax - b||² (or ||Ax - b||² + λ||x||² with regularization)
using conjugate gradients. Key properties:

- **Faster convergence**: Typically converges faster than SIRT
- **Semi-convergence**: Without regularization, reconstruction error may
  increase after optimal iteration count (10-20 iterations often best)
- **Tikhonov regularization**: Setting lambda > 0 prevents semi-convergence
  and allows more iterations without degradation
- **Matched backprojection**: Uses weighted=false for correct adjoint property,
  which is essential for conjugate gradient convergence

CGLS is particularly effective when:
- You need faster convergence than SIRT
- You have noisy data (use regularization or limit iterations)
- You want similar quality to SIRT with fewer iterations
"""
function cgls_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 20,
    lambda::Real = 0.0,
    tol::Real = 0.0,
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

    return cgls_reconstruct!(recon, sinogram, geom; niter=niter, lambda=lambda, tol=tol, verbose=verbose)
end
