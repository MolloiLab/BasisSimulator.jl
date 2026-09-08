# =============================================================================
# Batched-loop hooks: a fixed-trip-count loop over view batches whose body is
# ONE tensor program, so program size is independent of the number of views.
#
# Host (plain arrays): an ordinary `for` loop with slices/copies.
# Reactant (ext/BasisSimulatorReactantExt.jl): a StableHLO `while` loop
# (`@trace for`) with `dynamic_slice` / `dynamic_update_slice`, which Enzyme
# differentiates in reverse mode (probed: gradient ≡ unrolled ≡ finite
# differences).  Every stage that loops over views (DD forward/transpose, the
# spectral sum, FDK backprojection) is written against these four hooks only.
#
# Contract: `body(state, b, consts)` maps the loop-carried `state` (ONE array),
# the 1-based batch index `b` (a traced integer under Reactant — never convert
# it to a host `Int`) and the loop-invariant `consts` (a Tuple holding EVERY
# traced array the body reads — the tracer sees loop arguments, not closure
# captures; `body` itself must close over host data only) to the new state; `_dslice(x, start, len, dim)` is
# `x[.., start:start+len-1, ..]` along `dim`; `_dupdate(x, chunk, start, dim)`
# returns `x` with `chunk` written at `start` along `dim`; `_zeros(ref, T, dims)`
# is a zero tensor in the array world of `ref`.
# =============================================================================

function _batched_loop(body, n::Int, state, consts::Tuple, ::AbstractArray)
    for b in 1:n
        state = body(state, b, consts)
    end
    return state
end

function _dslice(x::AbstractArray{<:Any, N}, start, len::Int, dim::Int) where {N}
    idx = ntuple(d -> d == dim ? (start:(start + len - 1)) : Colon(), N)
    return x[idx...]
end

# Host: in place (the destination is always a fresh `_zeros` buffer owned by the caller).
function _dupdate(x::AbstractArray{<:Any, N}, chunk, start, dim::Int) where {N}
    idx = ntuple(d -> d == dim ? (start:(start + size(chunk, dim) - 1)) : Colon(), N)
    x[idx...] = chunk
    return x
end

_zeros(::AbstractArray, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"""
    _loop_over_batches(chunk_fn, n::Int, B::Int, out, consts::Tuple, ref) -> out

Apply `chunk_fn(start, len, consts...)` — which returns the result for views
`start:start+len-1` — over `1:n` in batches of `B`: the full batches run inside
[`_batched_loop`](@ref) (one program), a remainder batch runs once after it
(static shape), and every chunk is written into `out` along its third axis
(the view axis of sinograms).  `n ≤ B` runs a single chunk with no loop.
`consts` carries every traced array the chunk needs (see the contract above).
"""
function _loop_over_batches(chunk_fn, n::Int, B::Int, out, consts::Tuple, ref)
    B = min(B, n)
    nb = n ÷ B; tail = n % B
    if nb == 1 && tail == 0
        return _dupdate(out, chunk_fn(1, n, consts...), 1, 3)
    end
    # no variable named inside the closure may be assigned anywhere else in this scope (boxing)
    body = (state, b, cs) -> (s0 = (b - 1) * B + 1; _dupdate(state, chunk_fn(s0, B, cs...), s0, 3))
    out = _batched_loop(body, nb, out, consts, ref)
    if tail > 0
        s_tail = nb * B + 1
        out = _dupdate(out, chunk_fn(s_tail, tail, consts...), s_tail, 3)
    end
    return out
end

"""
    _sum_over_batches(chunk_fn, n::Int, B::Int, acc0, consts::Tuple, ref) -> acc

Like [`_loop_over_batches`](@ref) but the chunks are ACCUMULATED (`acc .+ chunk`)
instead of written side by side (backprojection-type reductions over views).
"""
function _sum_over_batches(chunk_fn, n::Int, B::Int, acc0, consts::Tuple, ref)
    B = min(B, n)
    nb = n ÷ B; tail = n % B
    if nb == 1 && tail == 0
        return acc0 .+ chunk_fn(1, n, consts...)
    end
    body = (state, b, cs) -> (s0 = (b - 1) * B + 1; state .+ chunk_fn(s0, B, cs...))
    acc = _batched_loop(body, nb, acc0, consts, ref)
    if tail > 0
        s_tail = nb * B + 1
        acc = acc .+ chunk_fn(s_tail, tail, consts...)
    end
    return acc
end
