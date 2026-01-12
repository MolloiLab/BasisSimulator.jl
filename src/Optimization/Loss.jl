"""
    Optimization/Loss.jl

Loss functions for CT optimization problems.

These loss functions are designed to be differentiable through Reactant/Enzyme
for gradient-based optimization of reconstruction and material decomposition.
"""

# =============================================================================
# Loss Functions
# =============================================================================

"""
    mse_loss(predicted, target)

Mean squared error loss.

    L = mean((predicted - target)²)

# Arguments
- `predicted`: Predicted values (e.g., forward projected sinogram)
- `target`: Target values (e.g., measured sinogram)

# Returns
Scalar loss value.
"""
function mse_loss(predicted::AbstractArray{T}, target::AbstractArray{T}) where T
    diff = predicted .- target
    return sum(diff .* diff) / length(diff)
end

"""
    weighted_mse_loss(predicted, target, weights)

Weighted mean squared error loss.

    L = sum(weights .* (predicted - target)²) / sum(weights)

Useful for emphasizing certain regions or handling variable noise.
"""
function weighted_mse_loss(
    predicted::AbstractArray{T},
    target::AbstractArray{T},
    weights::AbstractArray{T}
) where T
    diff = predicted .- target
    return sum(weights .* diff .* diff) / sum(weights)
end

"""
    mae_loss(predicted, target)

Mean absolute error loss.

    L = mean(|predicted - target|)

More robust to outliers than MSE.
"""
function mae_loss(predicted::AbstractArray{T}, target::AbstractArray{T}) where T
    diff = abs.(predicted .- target)
    return sum(diff) / length(diff)
end

"""
    huber_loss(predicted, target; delta=1.0)

Huber loss - quadratic for small errors, linear for large errors.

    L = { 0.5 * (x)²           if |x| ≤ δ
        { δ * (|x| - 0.5δ)     if |x| > δ

where x = predicted - target

# Arguments
- `predicted`: Predicted values
- `target`: Target values
- `delta`: Threshold between quadratic and linear regions
"""
function huber_loss(
    predicted::AbstractArray{T},
    target::AbstractArray{T};
    delta::T=T(1.0)
) where T
    diff = predicted .- target
    abs_diff = abs.(diff)

    # Quadratic for small errors, linear for large
    quadratic = T(0.5) .* diff .* diff
    linear = delta .* (abs_diff .- T(0.5) .* delta)

    loss_per_element = ifelse.(abs_diff .<= delta, quadratic, linear)
    return sum(loss_per_element) / length(diff)
end

# =============================================================================
# Regularization Terms
# =============================================================================

"""
    tv_regularization(volume; epsilon=1e-8)

Total Variation (TV) regularization for promoting piecewise constant solutions.

    TV(x) = Σ √(∂x² + ∂y² + ∂z² + ε)

Isotropic TV computed using finite differences.

# Arguments
- `volume`: 3D volume to regularize
- `epsilon`: Small constant for numerical stability
"""
function tv_regularization(volume::AbstractArray{T,3}; epsilon=T(1e-8)) where T
    nx, ny, nz = size(volume)

    # Compute squared gradients using finite differences
    # Pad with zeros at boundaries
    dx = volume[2:end, :, :] .- volume[1:end-1, :, :]
    dy = volume[:, 2:end, :] .- volume[:, 1:end-1, :]
    dz = volume[:, :, 2:end] .- volume[:, :, 1:end-1]

    # Sum squared gradients (using common region)
    nx_c, ny_c, nz_c = nx - 1, ny - 1, nz - 1

    grad_sq = dx[1:nx_c, 1:ny_c, 1:nz_c].^2 .+
              dy[1:nx_c, 1:ny_c, 1:nz_c].^2 .+
              dz[1:nx_c, 1:ny_c, 1:nz_c].^2

    return sum(sqrt.(grad_sq .+ epsilon))
end

"""
    l1_regularization(volume)

L1 regularization (sparsity promoting).

    L1(x) = Σ |x|
"""
function l1_regularization(volume::AbstractArray{T}) where T
    return sum(abs.(volume))
end

"""
    l2_regularization(volume)

L2 regularization (Tikhonov).

    L2(x) = Σ x²
"""
function l2_regularization(volume::AbstractArray{T}) where T
    return sum(volume .* volume)
end

"""
    non_negativity_penalty(volume; strength=1e6)

Penalty for negative values (soft constraint).

    P(x) = strength * Σ max(0, -x)²
"""
function non_negativity_penalty(volume::AbstractArray{T}; strength::T=T(1e6)) where T
    negative_vals = min.(volume, T(0))
    return strength * sum(negative_vals .* negative_vals)
end

# =============================================================================
# Combined Loss Functions
# =============================================================================

"""
    reconstruction_loss(volume, sinogram_target, proj_geom;
                        λ_tv=0.0, λ_l2=0.0, λ_nn=1e6)

Combined loss for iterative reconstruction.

    L = MSE(A·x, y) + λ_tv·TV(x) + λ_l2·L2(x) + λ_nn·NN(x)

where A is the forward projector, x is the volume, y is the target sinogram.

# Arguments
- `volume`: Current volume estimate
- `sinogram_target`: Target sinogram (measured data)
- `proj_geom`: Pre-computed projection geometry
- `λ_tv`: Total variation weight
- `λ_l2`: L2 regularization weight
- `λ_nn`: Non-negativity penalty weight
"""
function reconstruction_loss(
    volume::AbstractArray{T},
    sinogram_target::AbstractArray{T},
    proj_geom::ProjectionGeometry;
    λ_tv::T=T(0.0),
    λ_l2::T=T(0.0),
    λ_nn::T=T(1e6)
) where T
    # Data fidelity term
    sinogram_pred = project_volume(volume, proj_geom)
    data_loss = mse_loss(sinogram_pred, sinogram_target)

    # Regularization terms
    reg_loss = T(0)

    if λ_tv > 0
        reg_loss += λ_tv * tv_regularization(volume)
    end

    if λ_l2 > 0
        reg_loss += λ_l2 * l2_regularization(volume)
    end

    if λ_nn > 0
        reg_loss += non_negativity_penalty(volume; strength=λ_nn)
    end

    return data_loss + reg_loss
end

# =============================================================================
# Exports
# =============================================================================

export mse_loss, weighted_mse_loss, mae_loss, huber_loss
export tv_regularization, l1_regularization, l2_regularization, non_negativity_penalty
export reconstruction_loss
