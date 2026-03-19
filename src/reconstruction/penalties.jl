# =============================================================================
# Regularization Penalties
# =============================================================================
#
# Edge-preserving penalty functions for iterative reconstruction.
#
# Currently provides:
#   - HuberPenalty: quadratic/linear hybrid, controlled by delta threshold
#
# GPU-native via AcceleratedKernels.jl foreachindex.
# =============================================================================

import AcceleratedKernels as AK

export PenaltyType, HuberPenalty
export compute_huber_penalty, compute_huber_gradient!

# =============================================================================
# Abstract Type
# =============================================================================

"""
    PenaltyType

Abstract type for regularization penalties.
"""
abstract type PenaltyType end

# =============================================================================
# Huber Penalty
# =============================================================================

"""
    HuberPenalty <: PenaltyType

Huber penalty: edge-preserving quadratic/linear hybrid.

ψ(t) = t²/2           if |t| ≤ δ
ψ(t) = δ|t| - δ²/2    if |t| > δ

The δ parameter controls the edge threshold.
"""
struct HuberPenalty <: PenaltyType
    delta::Float32  # Edge threshold
end

HuberPenalty() = HuberPenalty(0.01f0)

# =============================================================================
# Huber Penalty Computation
# =============================================================================

@inline function _huber(t::T, δ::T) where T
    abs_t = abs(t)
    if abs_t ≤ δ
        return t * t / T(2)
    else
        return δ * abs_t - δ * δ / T(2)
    end
end

@inline function _huber_deriv(t::T, δ::T) where T
    abs_t = abs(t)
    if abs_t ≤ δ
        return t
    else
        return δ * sign(t)
    end
end

"""
    compute_huber_penalty(x, delta)

Compute Huber penalty value.

ψ(t) = t²/2           if |t| ≤ δ
ψ(t) = δ|t| - δ²/2    if |t| > δ
"""
function compute_huber_penalty(
    x::AbstractArray{T, 3},
    delta::Real
) where T <: AbstractFloat

    δ = T(delta)
    nx, ny, nz = size(x)
    penalty_vals = similar(x)
    backend = AK.get_backend(x)

    AK.foreachindex(penalty_vals, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        val = x[i, j, k]
        penalty = zero(T)

        # Apply Huber to each neighbor difference
        if i < nx
            diff = x[i+1, j, k] - val
            penalty += _huber(diff, δ)
        end
        if j < ny
            diff = x[i, j+1, k] - val
            penalty += _huber(diff, δ)
        end
        if k < nz
            diff = x[i, j, k+1] - val
            penalty += _huber(diff, δ)
        end

        penalty_vals[linear_idx] = penalty
    end

    return AK.mapreduce(identity, +, penalty_vals; init=zero(T))
end

"""
    compute_huber_gradient!(grad, x, delta)

Compute gradient of Huber penalty in-place.

ψ'(t) = t        if |t| ≤ δ
ψ'(t) = δ·sign(t) if |t| > δ
"""
function compute_huber_gradient!(
    grad::AbstractArray{T, 3},
    x::AbstractArray{T, 3},
    delta::Real
) where T <: AbstractFloat

    δ = T(delta)
    nx, ny, nz = size(x)
    backend = AK.get_backend(x)

    AK.foreachindex(grad, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        val = x[i, j, k]
        g = zero(T)

        # Forward differences
        if i < nx
            diff = x[i+1, j, k] - val
            g -= _huber_deriv(diff, δ)
        end
        if j < ny
            diff = x[i, j+1, k] - val
            g -= _huber_deriv(diff, δ)
        end
        if k < nz
            diff = x[i, j, k+1] - val
            g -= _huber_deriv(diff, δ)
        end

        # Backward differences
        if i > 1
            diff = val - x[i-1, j, k]
            g += _huber_deriv(diff, δ)
        end
        if j > 1
            diff = val - x[i, j-1, k]
            g += _huber_deriv(diff, δ)
        end
        if k > 1
            diff = val - x[i, j, k-1]
            g += _huber_deriv(diff, δ)
        end

        grad[linear_idx] = g
    end

    return grad
end
