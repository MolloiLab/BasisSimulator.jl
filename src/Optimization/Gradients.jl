"""
    Optimization/Gradients.jl

Gradient computation utilities for CT optimization using Reactant/Enzyme.

This module provides wrappers for computing gradients of loss functions
with respect to volume or other parameters, enabling gradient-based
optimization for iterative reconstruction and material decomposition.
"""

using Reactant

# =============================================================================
# Compiled Forward and Gradient Functions
# =============================================================================

"""
    compile_forward_projection(proj_geom, volume_size, T=Float32)

Compile the forward projection for fast repeated evaluation.

Returns a compiled function that takes a volume and returns the sinogram.
"""
function compile_forward_projection(
    proj_geom::ProjectionGeometry,
    volume_size::NTuple{3,Int};
    T::Type=Float32
)
    # Create template volume
    volume_template = Reactant.to_rarray(zeros(T, volume_size...))

    # Compile the projection
    compiled_proj = @compile project_volume(volume_template, proj_geom)

    return compiled_proj
end

"""
    compile_loss_and_gradient(proj_geom, sinogram_target;
                              λ_tv=0.0, λ_l2=0.0, λ_nn=1e6)

Compile the loss function and its gradient for iterative reconstruction.

Returns a function that takes a volume and returns (loss, gradient).

# Arguments
- `proj_geom`: Pre-computed projection geometry
- `sinogram_target`: Target sinogram to match
- `λ_tv`, `λ_l2`, `λ_nn`: Regularization weights

# Returns
Function `(volume) -> (loss, gradient)` for optimization.
"""
function compile_loss_and_gradient(
    proj_geom::ProjectionGeometry,
    sinogram_target::AbstractArray{T};
    λ_tv::T=T(0.0),
    λ_l2::T=T(0.0),
    λ_nn::T=T(1e6)
) where T
    # Convert target to Reactant array
    target_ra = Reactant.to_rarray(sinogram_target)

    # Create loss function closure
    function loss_fn(volume)
        return reconstruction_loss(volume, target_ra, proj_geom;
                                   λ_tv=λ_tv, λ_l2=λ_l2, λ_nn=λ_nn)
    end

    # Return function that computes loss and gradient
    # Note: Enzyme.autodiff integration with Reactant
    function loss_and_grad(volume::AbstractArray{T}) where T
        volume_ra = Reactant.to_rarray(volume)

        # Forward pass
        loss_val = loss_fn(volume_ra)

        # For now, use finite differences as fallback
        # Full Enzyme integration requires careful setup
        grad = finite_difference_gradient(loss_fn, volume_ra)

        return Array(loss_val), Array(grad)
    end

    return loss_and_grad
end

"""
    finite_difference_gradient(f, x; epsilon=1e-5)

Compute gradient using central finite differences.

This is a fallback for when automatic differentiation is not available.
Note: This is slow and should only be used for debugging or small problems.
"""
function finite_difference_gradient(f, x::AbstractArray{T}; epsilon::T=T(1e-5)) where T
    grad = similar(x)
    x_flat = vec(x)
    grad_flat = vec(grad)

    f_x = f(x)

    for i in eachindex(x_flat)
        x_plus = copy(x_flat)
        x_plus[i] += epsilon

        x_minus = copy(x_flat)
        x_minus[i] -= epsilon

        f_plus = f(reshape(x_plus, size(x)))
        f_minus = f(reshape(x_minus, size(x)))

        grad_flat[i] = (f_plus - f_minus) / (2 * epsilon)
    end

    return reshape(grad_flat, size(x))
end

# =============================================================================
# Adjoint Operator (Backprojection)
# =============================================================================

"""
    backproject_volume(sinogram, proj_geom, volume_size)

Backproject sinogram to volume space (adjoint of forward projection).

This is the transpose of the forward projection operator:
    A' y = Σ_i w_i × y_i  (for each voxel, sum weighted contributions)

# Arguments
- `sinogram`: Sinogram [n_cols, n_rows, n_angles]
- `proj_geom`: Pre-computed projection geometry
- `volume_size`: Size of output volume (nx, ny, nz)

# Returns
Volume array of shape [nx, ny, nz]
"""
function backproject_volume(
    sinogram::AbstractArray{T},
    proj_geom::ProjectionGeometry,
    volume_size::NTuple{3,Int}
) where T
    nx, ny, nz = volume_size
    n_cols, n_rows, n_angles = size(sinogram)
    n_samples = size(proj_geom.linear_indices, 4)

    # Initialize output volume
    volume = zeros(T, nx, ny, nz)
    volume_flat = vec(volume)

    # Weights as correct type
    weights = T.(proj_geom.sample_weights)

    # Accumulate contributions from each ray
    for angle in 1:n_angles
        for row in 1:n_rows
            for col in 1:n_cols
                sino_val = sinogram[col, row, angle]

                for s in 1:n_samples
                    idx = proj_geom.linear_indices[col, row, angle, s]
                    w = weights[col, row, angle, s]

                    if w > 0
                        volume_flat[idx] += w * sino_val
                    end
                end
            end
        end
    end

    return reshape(volume_flat, volume_size)
end

"""
    compute_gradient_data_term(volume, sinogram_target, proj_geom)

Compute gradient of MSE data term: ∇_x ||Ax - y||²

    ∇ = 2 A' (Ax - y)

# Arguments
- `volume`: Current volume estimate
- `sinogram_target`: Target sinogram
- `proj_geom`: Projection geometry

# Returns
Gradient array with same shape as volume.
"""
function compute_gradient_data_term(
    volume::AbstractArray{T},
    sinogram_target::AbstractArray{T},
    proj_geom::ProjectionGeometry
) where T
    # Forward project
    sinogram_pred = project_volume(volume, proj_geom)

    # Residual
    residual = sinogram_pred .- sinogram_target

    # Backproject residual (adjoint)
    grad = backproject_volume(residual, proj_geom, size(volume))

    # Scale by 2/N for MSE gradient
    grad .*= T(2) / length(sinogram_target)

    return grad
end

# =============================================================================
# Gradient Descent Optimizer
# =============================================================================

"""
    GradientDescentResult

Result of gradient descent optimization.
"""
struct GradientDescentResult{T}
    volume::Array{T,3}
    loss_history::Vector{T}
    converged::Bool
    iterations::Int
end

"""
    gradient_descent_reconstruction(sinogram_target, proj_geom, volume_size;
                                    n_iterations=100, learning_rate=1e-4,
                                    λ_tv=0.0, λ_l2=1e-6, verbose=true)

Simple gradient descent for iterative reconstruction.

# Arguments
- `sinogram_target`: Target sinogram to match
- `proj_geom`: Pre-computed projection geometry
- `volume_size`: Size of volume to reconstruct
- `n_iterations`: Maximum iterations
- `learning_rate`: Step size for gradient descent
- `λ_tv`, `λ_l2`: Regularization weights
- `verbose`: Print progress

# Returns
`GradientDescentResult` with reconstructed volume and loss history.
"""
function gradient_descent_reconstruction(
    sinogram_target::AbstractArray{T},
    proj_geom::ProjectionGeometry,
    volume_size::NTuple{3,Int};
    n_iterations::Int=100,
    learning_rate::T=T(1e-4),
    λ_tv::T=T(0.0),
    λ_l2::T=T(1e-6),
    verbose::Bool=true
) where T
    # Initialize with backprojection (good starting point)
    volume = backproject_volume(sinogram_target, proj_geom, volume_size)

    # Normalize initial estimate
    volume ./= maximum(abs.(volume)) + T(1e-10)
    volume .*= T(0.5)  # Scale to reasonable μ range

    loss_history = T[]
    converged = false

    for iter in 1:n_iterations
        # Compute gradient of data term
        grad = compute_gradient_data_term(volume, sinogram_target, proj_geom)

        # Add L2 regularization gradient
        if λ_l2 > 0
            grad .+= T(2) .* λ_l2 .* volume
        end

        # Update
        volume .-= learning_rate .* grad

        # Enforce non-negativity
        volume .= max.(volume, T(0))

        # Compute loss for monitoring
        sino_pred = project_volume(volume, proj_geom)
        loss = mse_loss(sino_pred, sinogram_target)

        if λ_l2 > 0
            loss += λ_l2 * l2_regularization(volume)
        end

        push!(loss_history, loss)

        if verbose && (iter == 1 || iter % 10 == 0)
            println("  Iteration $iter: loss = $loss")
        end

        # Check convergence
        if length(loss_history) > 1
            rel_change = abs(loss_history[end] - loss_history[end-1]) /
                        (abs(loss_history[end-1]) + T(1e-10))
            if rel_change < T(1e-6)
                converged = true
                if verbose
                    println("  Converged at iteration $iter")
                end
                break
            end
        end
    end

    return GradientDescentResult(volume, loss_history, converged, length(loss_history))
end

# =============================================================================
# Exports
# =============================================================================

export compile_forward_projection, compile_loss_and_gradient
export finite_difference_gradient
export backproject_volume, compute_gradient_data_term
export GradientDescentResult, gradient_descent_reconstruction
