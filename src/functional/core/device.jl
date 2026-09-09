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

"""
    _bmm_t(Wx, V) -> A

Batched transverse contraction of the dense DD projector:
`A[col, k, l, b] = Σ_t Wx[col, t, l, b] · V[t, k, l]` (`V` is view-independent).
Host: one matmul per (slab, view); Reactant: a single `dot_general`.
"""
function _bmm_t(Wx::AbstractArray{<:Any, 4}, V::AbstractArray{<:Any, 3})
    n_cols, n_t, n_long, B = size(Wx); K = size(V, 2)
    A = similar(Wx, n_cols, K, n_long, B)
    for b in 1:B, l in 1:n_long
        A[:, :, l, b] = Wx[:, :, l, b] * V[:, :, l]
    end
    return A
end

"""
    _bmm_zl(Wz, A) -> P

Batched axial + slab contraction of the dense DD projector:
`P[col, row, m, b] = Σ_l Σ_z Wz[col, row, z, l, b] · A[col, z, m, l, b]`.
Host: a broadcast-and-reduce; Reactant: a single `dot_general` batched over (col, b).
"""
function _bmm_zl(Wz::AbstractArray{<:Any, 5}, A::AbstractArray{<:Any, 5})
    n_cols, n_rows, nz, n_long, B = size(Wz); M = size(A, 3)
    P = similar(Wz, n_cols, n_rows, M, B)
    for b in 1:B, m in 1:M
        acc = zero(similar(Wz, n_cols, n_rows))
        for l in 1:n_long, z in 1:nz
            acc = acc .+ Wz[:, :, z, l, b] .* reshape(A[:, z, m, l, b], n_cols, 1)
        end
        P[:, :, m, b] = acc
    end
    return P
end

"""
    _bmm_rows(Wz, S) -> G

Transpose-side axial contraction: `G[col, z, l, b] = Σ_row Wz[col,row,z,l,b] · S[col,row,b]`.
"""
function _bmm_rows(Wz::AbstractArray{<:Any, 5}, S::AbstractArray{<:Any, 3})
    n_cols, n_rows, nz, n_long, B = size(Wz)
    G = similar(Wz, n_cols, nz, n_long, B)
    for b in 1:B, l in 1:n_long, z in 1:nz
        G[:, z, l, b] = vec(sum(Wz[:, :, z, l, b] .* S[:, :, b]; dims = 2))
    end
    return G
end

"""
    _bmm_cols(Wx, G) -> V

Transpose-side transverse contraction: `V[t, z, l] = Σ_{col,b} Wx[col,t,l,b] · G[col,z,l,b]`.
"""
function _bmm_cols(Wx::AbstractArray{<:Any, 4}, G::AbstractArray{<:Any, 4})
    n_cols, n_t, n_long, B = size(Wx); nz = size(G, 2)
    V = zeros(eltype(Wx), n_t, nz, n_long)
    for b in 1:B, l in 1:n_long
        V[:, :, l] .+= transpose(Wx[:, :, l, b]) * G[:, :, l, b]
    end
    return V
end

"""
    _bmm_tb(Wx, Vw) -> A

Transverse contraction with a per-view window of the volume:
`A[col, k, l, b] = Σ_t Wx[col, t, l, b] · Vw[t, k, l, b]`.
"""
function _bmm_tb(Wx::AbstractArray{<:Any, 4}, Vw::AbstractArray{<:Any, 4})
    n_cols, w, n_long, B = size(Wx); K = size(Vw, 2)
    A = similar(Wx, n_cols, K, n_long, B)
    for b in 1:B, l in 1:n_long
        A[:, :, l, b] = Wx[:, :, l, b] * Vw[:, :, l, b]
    end
    return A
end

"""
    _to_float(T, x) -> x as an array of `T`

Element type conversion (integer index arrays → the plan's float type) in the
array world of `x`: `T.(x)` on the host, a StableHLO convert under Reactant.
"""
_to_float(::Type{T}, x::AbstractArray) where {T} = T.(x)
