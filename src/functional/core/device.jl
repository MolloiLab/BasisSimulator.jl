# =============================================================================
# Backend hooks.  Host defaults here; ext/BasisSimulatorReactantExt.jl overrides
# them for traced arrays so the SAME stage code runs on plain arrays and inside
# `Reactant.@compile` (plan tensors lifted as constants, in-graph index vectors,
# compiled view loops — see core/loop.jl).
# =============================================================================

"""
    _iota(ref, T, n) -> [1, 2, …, n] of `T`

Index vector in the array world of `ref`: a host constant, or an in-graph iota
under Reactant (so every geometry array derived from it is computed inside the
compiled program instead of being embedded as a literal).
"""
_iota(::AbstractArray, ::Type{T}, n::Integer) where {T} = collect(T, 1:n)


"""
    _on_device(x, ref)

Return `x` in the array world of `ref`. Identity for plain arrays; the Reactant
extension lifts host plan tensors into the traced graph as constants when `ref`
is a traced array (a host `Matrix * traced` otherwise degrades to a scalar
fallback with one op per multiply-add).
"""
_on_device(x, ref) = x

_on_device(::Nothing, ref) = nothing

"""
    _plain(x) -> x

The array behind a lazy wrapper (`reshape` / `dropdims` / view of a traced
array) as a plain array of the same world; identity on the host.  Use before
broadcasts that mix wrapper and array styles.
"""
_plain(x) = x
