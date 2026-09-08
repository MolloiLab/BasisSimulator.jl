# =============================================================================
# Static-tap box-overlap resampling primitives
# =============================================================================
#
# Distance-driven projection integrates the *overlap* of two piecewise-constant
# partitions of one axis: the voxel boundaries, mapped affinely onto the
# iso-plane, and the detector-cell boundaries.  For one detector cell whose
# iso-plane interval is `[c_lo, c_hi]` (in voxel-boundary index units, voxel
# `i` spanning `[i-1, i]`), the weight of voxel `i` is
#
#     w_i = overlap([i-1, i], [c_lo, c_hi]) · (v · mag)        (iso-plane units)
#
# and only voxels `i ∈ (c_lo, c_hi + 1)` are non-zero, i.e. at most
# `floor(c_hi - c_lo) + 2` consecutive voxels starting at `floor(c_lo) + 1`.
# Because the span `c_hi - c_lo` is bounded by the geometry (max cell width ÷
# min magnified voxel width) the number of taps `K` is a *static* host constant,
# so the whole operator becomes: compute the first tap index by broadcasting,
# then `K` gathers with broadcast weights.  No data-dependent loops, no scatter,
# no scalar indexing — the form Reactant traces and Enzyme differentiates.
#
# The transpose is the same construction with cells and voxels swapped: each
# voxel overlaps at most `K'` consecutive cells whose first index is the floor
# of the inverse boundary map, so the adjoint is again a static-tap gather.
#
# Everything here is elementwise (usable in broadcasts on host arrays and on
# traced arrays alike); the projector in `dd_projector.jl` composes them.
#
# Typing rule (Reactant): `TracedRArray{T,N} <: AbstractArray{TracedRNumber{T},N}`,
# so nothing here constrains element types or binds a `where T` from an array —
# scalar types come from the caller (plan / explicit `::Type{T}`) only.
# =============================================================================

"""
    _overlap(a_lo, a_hi, b_lo, b_hi)

Length of `[a_lo, a_hi] ∩ [b_lo, b_hi]`, zero when disjoint.  Identical to the
legacy `_dd_overlap` (`hi > lo ? hi - lo : 0`) for finite inputs.
"""
@inline function _overlap(a_lo, a_hi, b_lo, b_hi)
    return max(zero(a_lo), min(a_hi, b_hi) - max(a_lo, b_lo))
end

"""
    _inrange(i, n)

Boolean mask `1 ≤ i ≤ n` for integer tap indices (used to zero the weight of
taps that fall outside the array; the gather index itself is clamped).
"""
@inline _inrange(i, n) = (i >= 1) & (i <= n)

"""
    _clamp_start(c, lo, hi)

Clamp a window-start coordinate (voxel-boundary or cell units) to `[lo, hi]`
(`lo = -2`, `hi = n + 2`, both host scalars of the plan's element type) so the
subsequent `floor` / `floor(Int32, ·)` are always safe; taps outside `1:n` are
masked to zero weight by the caller.  The floating and integer twins are taken
from this same clamped value so they agree bit-for-bit.
"""
@inline _clamp_start(c, lo, hi) = clamp(c, lo, hi)

"""
    _gather(A, L)

Reshape-free gather: `A[L]` with a linear-index array `L`, returned in the shape
of `L`.  Written as `vec(A)[vec(L)]` so that it lowers to a single XLA gather.
"""
@inline function _gather(A::AbstractArray, L::AbstractArray)
    return reshape(vec(A)[vec(L)], size(L))
end

"""
    _max_cells_spanned(boundaries::AbstractVector{<:Real}, W::Real)

Host helper for the transpose tap count: the maximum number of consecutive
cells (partition of the line by the sorted `boundaries`) that any interval of
length `W` can overlap.  Exact sliding-window count: for each boundary `X_j`,
the number of boundaries in `[X_j, X_j + W)` plus one.
"""
function _max_cells_spanned(boundaries::AbstractVector{<:Real}, W::Real)
    X = sort(collect(Float64, boundaries))
    n = length(X)
    best = 1
    for j in 1:n
        k = j
        while k <= n && X[k] < X[j] + W
            k += 1
        end
        best = max(best, (k - j) + 1)
    end
    return best
end
