# =============================================================================
# Oracle-harness helpers for BasisSimulator.Functional stages.
#
# Every functional stage is validated three ways against the legacy kernels:
#   1. parity      — `parity_report(new, oracle)` (max abs / max rel / mean rel)
#   2. adjointness — `adjoint_dot_test(A, At, x, y)`: ⟨A x, y⟩ = ⟨x, Aᵀ y⟩
#   3. operator    — `brute_force_matrix(A, vol_shape, sino_shape)`: every column
#                    of A formed explicitly, so Aᵀ y can be checked entry-wise.
# Reusable for every stage that follows the DD projector.
# =============================================================================

"""
    parity_report(a, b; floor=0)

Compare array `a` against oracle `b`.  Returns a NamedTuple with `max_abs`,
`max_rel` (relative to `max(|b|)`), `mean_rel` (mean |a−b|/|b| over entries with
`|b| > floor`), and `n` (number of entries used for `mean_rel`).
"""
function parity_report(a::AbstractArray, b::AbstractArray; floor::Real = 0)
    size(a) == size(b) || throw(DimensionMismatch("parity_report: $(size(a)) vs $(size(b))"))
    d = abs.(Float64.(a) .- Float64.(b))
    bmax = maximum(abs.(Float64.(b)))
    m = abs.(b) .> floor
    n = count(m)
    mean_rel = n == 0 ? 0.0 : sum(d[m] ./ abs.(Float64.(b[m]))) / n
    return (max_abs = maximum(d), max_rel = bmax > 0 ? maximum(d) / bmax : maximum(d),
        mean_rel = mean_rel, n = n)
end

"""
    adjoint_dot_test(A, At, x, y)

`A(x)` must have the shape of `y` and `At(y)` the shape of `x`.  Returns
`(lhs = ⟨A x, y⟩, rhs = ⟨x, Aᵀ y⟩, rel = |lhs − rhs| / max(|lhs|, |rhs|))`,
accumulated in Float64.
"""
function adjoint_dot_test(A, At, x::AbstractArray, y::AbstractArray)
    Ax = A(x); Aty = At(y)
    size(Ax) == size(y) || throw(DimensionMismatch("A(x) $(size(Ax)) vs y $(size(y))"))
    size(Aty) == size(x) || throw(DimensionMismatch("At(y) $(size(Aty)) vs x $(size(x))"))
    lhs = sum(Float64.(Ax) .* Float64.(y))
    rhs = sum(Float64.(x) .* Float64.(Aty))
    return (lhs = lhs, rhs = rhs, rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps()))
end

"""
    brute_force_matrix(A, vol_shape, sino_shape; T=Float64)

Explicit matrix of the linear operator `A` (a function volume → sinogram):
column `j` is `vec(A(e_j))`.  Sized `prod(sino_shape) × prod(vol_shape)`; only
sensible for tiny fixtures.
"""
function brute_force_matrix(A, vol_shape::Dims, sino_shape::Dims; T::Type = Float64)
    nvox = prod(vol_shape); nrays = prod(sino_shape)
    M = Matrix{T}(undef, nrays, nvox)
    basis = zeros(T, vol_shape)
    for j in 1:nvox
        fill!(basis, zero(T)); basis[j] = one(T)
        col = A(basis)
        size(col) == sino_shape || throw(DimensionMismatch("A(e_j) $(size(col)) vs $(sino_shape)"))
        M[:, j] .= vec(col)
    end
    return M
end
