# =============================================================================
# Total Variation Regularization for CT Reconstruction
# =============================================================================
#
# GPU-native implementation using AcceleratedKernels.jl
#
# Reference:
#   - Rudin, Osher, Fatemi (ROF): "Nonlinear total variation based noise removal
#     algorithms" Physica D: Nonlinear Phenomena, 1992
#   - Chambolle: "An algorithm for total variation minimization and applications"
#     J. Math. Imaging Vis. 2004
#   - TIGRE: CERN/TIGRE/MATLAB/Algorithms/OS_SART_TV.m
#
# Total Variation:
#   TV(x) = Σ_j ||∇x_j||_p
#
#   - Isotropic (p=2): TV(x) = Σ √[(∂x/∂x₁)² + (∂x/∂x₂)² + (∂x/∂x₃)²]
#   - Anisotropic (p=1): TV(x) = Σ [|∂x/∂x₁| + |∂x/∂x₂| + |∂x/∂x₃|]
#
# TV Gradient (isotropic):
#   ∇TV(x)_j = -div(∇x_j / ||∇x_j||)
#
# Integration with iterative methods:
#   - TV-SIRT: SIRT update followed by TV denoising (proximal operator)
#   - TV-CGLS: CGLS with TV as regularization term (gradient descent on TV)
#
# CRITICAL: All operations are GPU-native via AK.foreachindex()
#
# =============================================================================

import AcceleratedKernels as AK

export compute_tv, compute_tv_gradient, compute_tv_gradient!
export tv_denoise!, tv_denoise
export tv_sirt_reconstruct, tv_sirt_reconstruct!
export tv_cgls_reconstruct, tv_cgls_reconstruct!
export TVType, IsotropicTV, AnisotropicTV

# =============================================================================
# TV Type Dispatch
# =============================================================================

"""
    TVType

Abstract type for total variation formulations.
"""
abstract type TVType end

"""
    IsotropicTV <: TVType

Isotropic total variation: TV(x) = Σ √[(∂x/∂x₁)² + (∂x/∂x₂)² + (∂x/∂x₃)²]

Rotationally invariant, better preserves edges in all directions.
"""
struct IsotropicTV <: TVType end

"""
    AnisotropicTV <: TVType

Anisotropic total variation: TV(x) = Σ [|∂x/∂x₁| + |∂x/∂x₂| + |∂x/∂x₃|]

Computationally simpler, may produce staircase artifacts along axes.
"""
struct AnisotropicTV <: TVType end

# =============================================================================
# TV Value Computation
# =============================================================================

"""
    compute_tv(x; tv_type=IsotropicTV(), eps=1e-8)

Compute total variation value TV(x) using forward differences.

# Arguments
- `x`: 3D volume [nx, ny, nz]

# Keyword Arguments
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `eps`: Small constant to avoid division by zero (default: 1e-8)

# Returns
Scalar TV value

# Notes
Uses forward finite differences with Neumann boundary conditions (zero gradient at boundary).
"""
function compute_tv(
    x::AbstractArray{T, 3};
    tv_type::TVType = IsotropicTV(),
    eps::Real = 1e-8
) where T <: AbstractFloat

    ε = T(eps)
    nx, ny, nz = size(x)

    # Compute TV contributions at each voxel
    tv_contributions = similar(x)
    backend = AK.get_backend(x)

    AK.foreachindex(tv_contributions, backend) do linear_idx
        # Convert linear index to Cartesian
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        # Forward differences with Neumann BC (zero at boundary)
        dx = (i < nx) ? (x[i+1, j, k] - x[i, j, k]) : zero(T)
        dy = (j < ny) ? (x[i, j+1, k] - x[i, j, k]) : zero(T)
        dz = (k < nz) ? (x[i, j, k+1] - x[i, j, k]) : zero(T)

        # TV contribution based on type
        tv_contributions[linear_idx] = _tv_point_value(dx, dy, dz, tv_type, ε)
    end

    # Sum all contributions
    return AK.mapreduce(identity, +, tv_contributions; init=zero(T))
end

# Point-wise TV value (for dispatch)
@inline function _tv_point_value(dx::T, dy::T, dz::T, ::IsotropicTV, ε::T) where T
    return sqrt(dx*dx + dy*dy + dz*dz + ε)
end

@inline function _tv_point_value(dx::T, dy::T, dz::T, ::AnisotropicTV, ε::T) where T
    return abs(dx) + abs(dy) + abs(dz)
end

# =============================================================================
# TV Gradient Computation
# =============================================================================

"""
    compute_tv_gradient!(grad, x; tv_type=IsotropicTV(), eps=1e-8)

Compute TV gradient in-place: ∇TV(x) = -div(∇x / ||∇x||)

# Arguments
- `grad`: Output gradient array [nx, ny, nz] (modified in place)
- `x`: Input volume [nx, ny, nz]

# Keyword Arguments
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `eps`: Small constant to avoid division by zero (default: 1e-8)

# Returns
The modified gradient array

# Notes
The TV gradient is computed as:
- Isotropic: ∇TV(x)_j = -div(∇x / ||∇x||₂)
- Anisotropic: ∇TV(x)_j = -div(sign(∇x))

Uses forward differences for gradient and backward differences for divergence,
with Neumann boundary conditions.
"""
function compute_tv_gradient!(
    grad::AbstractArray{T, 3},
    x::AbstractArray{T, 3};
    tv_type::TVType = IsotropicTV(),
    eps::Real = 1e-8
) where T <: AbstractFloat

    ε = T(eps)
    nx, ny, nz = size(x)
    backend = AK.get_backend(x)

    AK.foreachindex(grad, backend) do linear_idx
        # Convert linear index to Cartesian
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        # Compute TV gradient at this voxel
        grad[linear_idx] = _tv_gradient_point(x, i, j, k, nx, ny, nz, tv_type, ε)
    end

    return grad
end

"""
    compute_tv_gradient(x; tv_type=IsotropicTV(), eps=1e-8)

Compute TV gradient, allocating new output array.

See `compute_tv_gradient!` for full documentation.
"""
function compute_tv_gradient(
    x::AbstractArray{T, 3};
    tv_type::TVType = IsotropicTV(),
    eps::Real = 1e-8
) where T <: AbstractFloat

    grad = similar(x)
    return compute_tv_gradient!(grad, x; tv_type=tv_type, eps=eps)
end

# =============================================================================
# TV Gradient Point-wise Computation (Isotropic)
# =============================================================================

"""
Compute isotropic TV gradient at point (i,j,k).

∇TV(x)_ijk = -div(∇x / ||∇x||)

The divergence of normalized gradient requires computing:
1. Forward differences ∇x at current and neighboring points
2. Normalization ||∇x||
3. Backward differences (divergence) of normalized field
"""
@inline function _tv_gradient_point(
    x::AbstractArray{T, 3},
    i::Int, j::Int, k::Int,
    nx::Int, ny::Int, nz::Int,
    ::IsotropicTV,
    ε::T
) where T <: AbstractFloat

    # Get current value
    x_ijk = x[i, j, k]

    # Forward differences at (i,j,k)
    dx_ijk = (i < nx) ? (x[i+1, j, k] - x_ijk) : zero(T)
    dy_ijk = (j < ny) ? (x[i, j+1, k] - x_ijk) : zero(T)
    dz_ijk = (k < nz) ? (x[i, j, k+1] - x_ijk) : zero(T)

    # Gradient magnitude at (i,j,k)
    norm_ijk = sqrt(dx_ijk*dx_ijk + dy_ijk*dy_ijk + dz_ijk*dz_ijk + ε)

    # Normalized gradient components at (i,j,k)
    px_ijk = dx_ijk / norm_ijk
    py_ijk = dy_ijk / norm_ijk
    pz_ijk = dz_ijk / norm_ijk

    # Need normalized gradients at neighboring points for divergence
    # Point (i-1,j,k) - for x-divergence
    px_im1 = zero(T)
    if i > 1
        x_im1jk = x[i-1, j, k]
        dx_im1 = x_ijk - x_im1jk  # forward diff at i-1
        dy_im1 = (j < ny) ? (x[i-1, j+1, k] - x_im1jk) : zero(T)
        dz_im1 = (k < nz) ? (x[i-1, j, k+1] - x_im1jk) : zero(T)
        norm_im1 = sqrt(dx_im1*dx_im1 + dy_im1*dy_im1 + dz_im1*dz_im1 + ε)
        px_im1 = dx_im1 / norm_im1
    end

    # Point (i,j-1,k) - for y-divergence
    py_jm1 = zero(T)
    if j > 1
        x_ijm1k = x[i, j-1, k]
        dx_jm1 = (i < nx) ? (x[i+1, j-1, k] - x_ijm1k) : zero(T)
        dy_jm1 = x_ijk - x_ijm1k  # forward diff at j-1
        dz_jm1 = (k < nz) ? (x[i, j-1, k+1] - x_ijm1k) : zero(T)
        norm_jm1 = sqrt(dx_jm1*dx_jm1 + dy_jm1*dy_jm1 + dz_jm1*dz_jm1 + ε)
        py_jm1 = dy_jm1 / norm_jm1
    end

    # Point (i,j,k-1) - for z-divergence
    pz_km1 = zero(T)
    if k > 1
        x_ijkm1 = x[i, j, k-1]
        dx_km1 = (i < nx) ? (x[i+1, j, k-1] - x_ijkm1) : zero(T)
        dy_km1 = (j < ny) ? (x[i, j+1, k-1] - x_ijkm1) : zero(T)
        dz_km1 = x_ijk - x_ijkm1  # forward diff at k-1
        norm_km1 = sqrt(dx_km1*dx_km1 + dy_km1*dy_km1 + dz_km1*dz_km1 + ε)
        pz_km1 = dz_km1 / norm_km1
    end

    # Divergence: div(p) = ∂px/∂x + ∂py/∂y + ∂pz/∂z (backward differences)
    div_p = (px_ijk - px_im1) + (py_ijk - py_jm1) + (pz_ijk - pz_km1)

    # TV gradient = -div(p)
    return -div_p
end

# =============================================================================
# TV Gradient Point-wise Computation (Anisotropic)
# =============================================================================

"""
Compute anisotropic TV gradient at point (i,j,k).

∇TV(x)_ijk = -div(sign(∇x))

For anisotropic TV, the gradient simplifies to differences of sign functions.
"""
@inline function _tv_gradient_point(
    x::AbstractArray{T, 3},
    i::Int, j::Int, k::Int,
    nx::Int, ny::Int, nz::Int,
    ::AnisotropicTV,
    ε::T
) where T <: AbstractFloat

    # Get current value
    x_ijk = x[i, j, k]

    # Forward differences at (i,j,k)
    dx_ijk = (i < nx) ? (x[i+1, j, k] - x_ijk) : zero(T)
    dy_ijk = (j < ny) ? (x[i, j+1, k] - x_ijk) : zero(T)
    dz_ijk = (k < nz) ? (x[i, j, k+1] - x_ijk) : zero(T)

    # Sign of gradients at (i,j,k)
    sx_ijk = _smooth_sign(dx_ijk, ε)
    sy_ijk = _smooth_sign(dy_ijk, ε)
    sz_ijk = _smooth_sign(dz_ijk, ε)

    # Sign of gradients at neighboring points
    sx_im1 = zero(T)
    if i > 1
        dx_im1 = x_ijk - x[i-1, j, k]
        sx_im1 = _smooth_sign(dx_im1, ε)
    end

    sy_jm1 = zero(T)
    if j > 1
        dy_jm1 = x_ijk - x[i, j-1, k]
        sy_jm1 = _smooth_sign(dy_jm1, ε)
    end

    sz_km1 = zero(T)
    if k > 1
        dz_km1 = x_ijk - x[i, j, k-1]
        sz_km1 = _smooth_sign(dz_km1, ε)
    end

    # Divergence of sign field
    div_s = (sx_ijk - sx_im1) + (sy_ijk - sy_jm1) + (sz_ijk - sz_km1)

    # TV gradient = -div(s)
    return -div_s
end

# Smooth sign function for numerical stability
@inline function _smooth_sign(x::T, ε::T) where T <: AbstractFloat
    return x / sqrt(x*x + ε)
end

# =============================================================================
# TV Denoising (Proximal Operator)
# =============================================================================

"""
    tv_denoise!(x, lambda; tv_type=IsotropicTV(), niter=20, eps=1e-8)

In-place TV denoising using gradient descent.

Solves: min_x (1/2)||x - x₀||² + λ·TV(x)

This is the proximal operator of TV, used as a denoising step in TV-SIRT.

# Arguments
- `x`: Volume to denoise [nx, ny, nz] (modified in place)
- `lambda`: TV regularization strength (λ)

# Keyword Arguments
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `niter`: Number of denoising iterations (default: 20)
- `eps`: Small constant for numerical stability (default: 1e-8)

# Returns
The modified (denoised) volume

# Notes
Uses gradient descent with adaptive step size. For the ROF problem:
- Objective: (1/2)||x - x₀||² + λ·TV(x)
- Gradient: (x - x₀) + λ·∇TV(x)
The step size is chosen based on the Lipschitz constant of the gradient.

Reference: Rudin, Osher, Fatemi, "Nonlinear total variation based noise removal
algorithms" Physica D 1992
"""
function tv_denoise!(
    x::AbstractArray{T, 3},
    lambda::Real;
    tv_type::TVType = IsotropicTV(),
    niter::Int = 20,
    eps::Real = 1e-8
) where T <: AbstractFloat

    λ = T(lambda)
    ε = T(eps)

    # Early return for zero regularization
    if λ ≤ zero(T)
        return x
    end

    # Store original for data fidelity term
    x0 = copy(x)

    # Gradient array
    grad = similar(x)

    backend = AK.get_backend(x)

    # Step size for gradient descent on (1/2)||x-x0||² + λ·TV(x)
    # The TV gradient is not Lipschitz continuous, so we use a conservative step size.
    # Empirically, step = 0.125 works well for a wide range of λ values.
    # This ensures stability while maintaining reasonable convergence speed.
    step = T(0.125)

    for _ in 1:niter
        # Compute TV gradient
        compute_tv_gradient!(grad, x; tv_type=tv_type, eps=eps)

        # Gradient descent update: x = x - step * ((x - x0) + λ·∇TV(x))
        AK.foreachindex(x, backend) do idx
            data_fidelity_grad = x[idx] - x0[idx]
            tv_grad_term = λ * grad[idx]
            total_grad = data_fidelity_grad + tv_grad_term
            x[idx] -= step * total_grad
        end
    end

    return x
end

"""
    tv_denoise(x, lambda; tv_type=IsotropicTV(), niter=20, eps=1e-8)

TV denoising, allocating new output array.

See `tv_denoise!` for full documentation.
"""
function tv_denoise(
    x::AbstractArray{T, 3},
    lambda::Real;
    tv_type::TVType = IsotropicTV(),
    niter::Int = 20,
    eps::Real = 1e-8
) where T <: AbstractFloat

    result = copy(x)
    return tv_denoise!(result, lambda; tv_type=tv_type, niter=niter, eps=eps)
end

# =============================================================================
# TV-SIRT Reconstruction
# =============================================================================

"""
    tv_sirt_reconstruct!(recon, sinogram, geom; niter=50, lambda_sirt=1.0,
                         lambda_tv=0.01, tv_niter=20, tv_type=IsotropicTV(),
                         verbose=false)

In-place TV-SIRT reconstruction.

Algorithm: Alternating SIRT updates and TV denoising
1. x_{k+1/2} = SIRT_step(x_k)  (data fidelity)
2. x_{k+1} = TV_denoise(x_{k+1/2})  (regularization)

# Arguments
- `recon`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `niter`: Number of outer iterations (default: 50)
- `lambda_sirt`: SIRT relaxation parameter (default: 1.0)
- `lambda_tv`: TV regularization strength (default: 0.01)
- `tv_niter`: Number of TV denoising iterations per outer iteration (default: 20)
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `verbose`: Print progress (default: false)

# Returns
The modified reconstruction array

# Notes
- TV regularization promotes piecewise-constant images, good for edge preservation
- Higher lambda_tv gives more smoothing but may lose detail
- Typical range: lambda_tv ∈ [0.001, 0.1]
"""
function tv_sirt_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    niter::Int = 50,
    lambda_sirt::Real = 1.0,
    lambda_tv::Real = 0.01,
    tv_niter::Int = 20,
    tv_type::TVType = IsotropicTV(),
    verbose::Bool = false
) where T <: AbstractFloat

    λ_sirt = T(lambda_sirt)
    λ_tv = T(lambda_tv)
    volume_size = size(recon)

    # Pre-compute SIRT weights
    verbose && println("Computing TV-SIRT weights...")
    W = compute_projection_weights(geom, volume_size, T)
    V_inv = compute_image_weights(geom, volume_size, T)

    # Transfer weights to same device as sinogram
    W_gpu = similar(sinogram, T, size(W)...)
    copyto!(W_gpu, W)
    V_inv_gpu = similar(recon, T, size(V_inv)...)
    copyto!(V_inv_gpu, V_inv)

    verbose && println("Running $niter TV-SIRT iterations (TV: $(tv_type), λ_tv=$lambda_tv)...")

    for iter in 1:niter
        # Step 1: SIRT update
        sirt_iteration!(recon, sinogram, geom, W_gpu, V_inv_gpu, λ_sirt)

        # Step 2: TV denoising
        tv_denoise!(recon, λ_tv; tv_type=tv_type, niter=tv_niter)

        if verbose && iter % 10 == 0
            tv_val = compute_tv(recon; tv_type=tv_type)
            println("  Iteration $iter/$niter, TV = $(round(tv_val, digits=4))")
        end
    end

    return recon
end

"""
    tv_sirt_reconstruct(sinogram, geom, volume_size; niter=50, lambda_sirt=1.0,
                        lambda_tv=0.01, tv_niter=20, tv_type=IsotropicTV(),
                        init=:zeros, verbose=false)

TV-SIRT reconstruction with various initialization options.

# Arguments
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `niter`: Number of outer iterations (default: 50)
- `lambda_sirt`: SIRT relaxation parameter (default: 1.0)
- `lambda_tv`: TV regularization strength (default: 0.01)
- `tv_niter`: Number of TV denoising iterations per outer iteration (default: 20)
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `init`: Initialization method - :zeros, :fdk, or an array (default: :zeros)
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Basic TV-SIRT with isotropic TV
recon = tv_sirt_reconstruct(sinogram, geom, (128, 128, 64);
                            niter=100, lambda_tv=0.02)

# TV-SIRT with FDK initialization and anisotropic TV
recon = tv_sirt_reconstruct(sinogram, geom, (128, 128, 64);
                            niter=50, lambda_tv=0.01, tv_type=AnisotropicTV(),
                            init=:fdk)
```
"""
function tv_sirt_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 50,
    lambda_sirt::Real = 1.0,
    lambda_tv::Real = 0.01,
    tv_niter::Int = 20,
    tv_type::TVType = IsotropicTV(),
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

    return tv_sirt_reconstruct!(recon, sinogram, geom;
                                 niter=niter, lambda_sirt=lambda_sirt,
                                 lambda_tv=lambda_tv, tv_niter=tv_niter,
                                 tv_type=tv_type, verbose=verbose)
end

# =============================================================================
# TV-CGLS Reconstruction
# =============================================================================

"""
    tv_cgls_reconstruct!(recon, sinogram, geom; niter=20, lambda_tv=0.01,
                         tv_niter=5, tv_type=IsotropicTV(), tol=0.0, verbose=false)

In-place TV-CGLS reconstruction.

Algorithm: Interleaved CGLS and TV gradient descent
1. x_{k+1/2} = CGLS_step(x_k)  (data fidelity via conjugate gradients)
2. x_{k+1} = x_{k+1/2} - α·∇TV(x_{k+1/2})  (TV regularization)

# Arguments
- `recon`: Initial reconstruction [nx, ny, nz] (modified in place)
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters

# Keyword Arguments
- `niter`: Number of CGLS iterations (default: 20)
- `lambda_tv`: TV regularization strength (default: 0.01)
- `tv_niter`: Number of TV gradient descent steps per CGLS iteration (default: 5)
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `tol`: Convergence tolerance for relative residual (default: 0.0)
- `verbose`: Print progress (default: false)

# Returns
The modified reconstruction array

# Notes
- Combines fast CGLS convergence with TV edge preservation
- TV steps are applied as gradient descent, not full proximal solve
- Suitable for moderate noise levels
"""
function tv_cgls_reconstruct!(
    recon::AbstractArray{T, 3},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry;
    niter::Int = 20,
    lambda_tv::Real = 0.01,
    tv_niter::Int = 5,
    tv_type::TVType = IsotropicTV(),
    tol::Real = 0.0,
    verbose::Bool = false
) where T <: AbstractFloat

    λ_tv = T(lambda_tv)
    tolerance = T(tol)
    ε = T(1e-8)

    # TV gradient step size
    tv_step = one(T) / (T(8) * λ_tv + ε)

    verbose && println("Initializing TV-CGLS (TV: $(tv_type), λ_tv=$lambda_tv)...")

    # Initialize CGLS variables
    # r = b - A·x (initial residual)
    projected = siddon_forward_project(recon, geom)
    r = similar(sinogram)
    AK.foreachindex(r) do idx
        r[idx] = sinogram[idx] - projected[idx]
    end

    initial_residual_norm = sqrt(compute_norm_squared(r))

    # p = Aᵀ·r (initial search direction)
    p = backproject(r, geom, size(recon); weighted=false)

    # γ = ||p||²
    gamma = compute_norm_squared(p)

    # TV gradient storage
    tv_grad = similar(recon)

    backend = AK.get_backend(recon)

    verbose && println("Running up to $niter TV-CGLS iterations...")
    verbose && println("  Initial ||r|| = $(round(initial_residual_norm, digits=4))")

    for iter in 1:niter
        # Standard CGLS iteration
        gamma = cgls_iteration!(recon, r, p, gamma, geom; lambda=zero(T))

        # TV regularization steps
        for _ in 1:tv_niter
            compute_tv_gradient!(tv_grad, recon; tv_type=tv_type, eps=ε)
            AK.foreachindex(recon, backend) do idx
                recon[idx] -= tv_step * λ_tv * tv_grad[idx]
            end
        end

        if verbose && iter % 5 == 0
            residual_norm = sqrt(compute_norm_squared(r))
            tv_val = compute_tv(recon; tv_type=tv_type)
            rel_residual = residual_norm / max(initial_residual_norm, T(1e-12))
            println("  Iteration $iter/$niter, ||r|| = $(round(residual_norm, digits=4)), TV = $(round(tv_val, digits=4))")
        end

        # Check convergence
        if gamma < T(1e-20)
            verbose && println("  Converged (gradient) at iteration $iter")
            break
        end

        if tolerance > zero(T)
            residual_norm = sqrt(compute_norm_squared(r))
            rel_residual = residual_norm / max(initial_residual_norm, T(1e-12))
            if rel_residual < tolerance
                verbose && println("  Converged (tolerance) at iteration $iter")
                break
            end
        end
    end

    return recon
end

"""
    tv_cgls_reconstruct(sinogram, geom, volume_size; niter=20, lambda_tv=0.01,
                        tv_niter=5, tv_type=IsotropicTV(), tol=0.0,
                        init=:zeros, verbose=false)

TV-CGLS reconstruction with various initialization options.

# Arguments
- `sinogram`: Measured sinogram [n_cols, n_rows, n_angles]
- `geom`: CTGeometry with scanner parameters
- `volume_size`: (nx, ny, nz) output volume dimensions

# Keyword Arguments
- `niter`: Number of CGLS iterations (default: 20)
- `lambda_tv`: TV regularization strength (default: 0.01)
- `tv_niter`: Number of TV gradient descent steps per CGLS iteration (default: 5)
- `tv_type`: TVType instance - IsotropicTV() or AnisotropicTV() (default: IsotropicTV())
- `tol`: Convergence tolerance for relative residual (default: 0.0)
- `init`: Initialization method - :zeros, :fdk, or an array (default: :zeros)
- `verbose`: Print progress (default: false)

# Returns
Reconstructed volume [nx, ny, nz]

# Example
```julia
# Basic TV-CGLS
recon = tv_cgls_reconstruct(sinogram, geom, (128, 128, 64);
                            niter=30, lambda_tv=0.01)

# TV-CGLS with FDK initialization
recon = tv_cgls_reconstruct(sinogram, geom, (128, 128, 64);
                            niter=15, lambda_tv=0.02, init=:fdk)
```
"""
function tv_cgls_reconstruct(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    niter::Int = 20,
    lambda_tv::Real = 0.01,
    tv_niter::Int = 5,
    tv_type::TVType = IsotropicTV(),
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

    return tv_cgls_reconstruct!(recon, sinogram, geom;
                                 niter=niter, lambda_tv=lambda_tv,
                                 tv_niter=tv_niter, tv_type=tv_type,
                                 tol=tol, verbose=verbose)
end
