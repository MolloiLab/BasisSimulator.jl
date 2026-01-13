"""
    Optimization/Gradients.jl

Gradient computation utilities for CT optimization.

This module provides basic gradient utilities. For full iterative reconstruction,
use the ray marching forward/backprojection functions which are designed to work
with Reactant compilation.
"""

# =============================================================================
# Finite Difference Gradient (Fallback)
# =============================================================================

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
# Exports
# =============================================================================

export finite_difference_gradient
