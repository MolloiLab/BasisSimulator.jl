"""
    BasisSimulatorReactantExt

Loaded automatically when `Reactant` is imported alongside `BasisSimulator`.

The functional core (`BasisSimulator.Functional`) is written against plain
`AbstractArray`s and needs nothing from Reactant to *trace*. This extension only
provides the few device-aware hooks the pure code cannot express itself:

* `Functional._on_device(x, ref)` — when `ref` is a traced array (i.e. we are
  inside `Reactant.@compile`), lift a host plan tensor `x` into the graph as an
  MLIR constant so that `x * traced` / `traced * x` become real XLA matmuls.
  Without this, a host `Matrix` times a `TracedRArray` falls back to the generic
  scalar `Matrix{TracedRNumber}` product and emits one op per scalar
  multiply-add (minutes of tracing for a toy problem).  Constants are the right
  choice for plan tables (μ tables, filter Toeplitz matrices, bowtie weights);
  large data tensors should be passed as thunk arguments instead.
"""
module BasisSimulatorReactantExt

using BasisSimulator
using Reactant

const BSF = BasisSimulator.Functional

BSF._on_device(x::AbstractArray, ::Reactant.TracedRArray) = Reactant.Ops.constant(Array(x))
BSF._on_device(x::AbstractArray, ::Reactant.TracedRNumber) = Reactant.Ops.constant(Array(x))

# In-graph index vectors for the projector: with the plain-array fallback (`collect(T, 1:n)`) every
# geometry/index tensor derived from them is evaluated on the HOST and embedded in the module as a
# literal constant (megabytes per view; trace time then scales with detector columns × slabs).
# An `Ops.iota` makes all of that in-graph arithmetic instead.
BSF._iota(::Reactant.TracedRArray, ::Type{T}, n::Integer) where {T} =
    Reactant.Ops.iota(T, [Int(n)]; iota_dimension = 1) .+ one(T)

# Batched-loop hooks (see src/functional/core/loop.jl): a StableHLO while loop over view batches
# with dynamic slices, so the compiled program does not grow with the number of views.
# `track_numbers = false`: host integers in the body (batch sizes, static shapes) must stay
# host integers; only the loop index is traced.
# Wrapper arrays over traced data (`reshape`, `dropdims`, views produce `ReshapedArray` /
# `SubArray` of a TracedRArray) are materialized into plain traced arrays before the ops.
_mat(x::Reactant.TracedRArray) = x
_mat(x::AbstractArray) = Reactant.TracedUtils.materialize_traced_array(x)
const _AnyTraced = Union{Reactant.TracedRArray, Base.ReshapedArray{<:Any, <:Any, <:Reactant.TracedRArray},
    SubArray{<:Any, <:Any, <:Reactant.TracedRArray}}
# `state`, `consts` and `b` are the loop arguments the tracer sees (they are named in the
# loop body); traced arrays hidden inside closures are NOT, so `body` closes over host data only.
# `checkpointing = true`: reverse mode through the loop recomputes iterations instead of caching
# every iteration's intermediates (the dense weights of a view batch are hundreds of MB; caching
# all iterations killed a 256-grid / 984-view gradient with an out-of-memory).
function BSF._batched_loop(body, n::Int, state, consts::Tuple, ::_AnyTraced)
    state = _mat(state)
    @trace track_numbers = false checkpointing = true for b in 1:n
        state = _mat(body(state, b, consts))
    end
    return state
end
function BSF._dslice(x::_AnyTraced, start, len::Int, dim::Int)
    xm = _mat(x); N = ndims(xm)
    starts = Any[d == dim ? start : 1 for d in 1:N]
    sizes = Int[d == dim ? len : size(xm, d) for d in 1:N]
    return Reactant.Ops.dynamic_slice(xm, starts, sizes)
end
function BSF._dupdate(x::_AnyTraced, chunk, start, dim::Int)
    xm = _mat(x); N = ndims(xm)
    starts = Any[d == dim ? start : 1 for d in 1:N]
    return Reactant.Ops.dynamic_update_slice(xm, _mat(chunk), starts)
end
BSF._zeros(::_AnyTraced, ::Type{T}, dims::Dims) where {T} =
    Reactant.Ops.fill(zero(T), collect(Int, dims))
BSF._dslice_at(x::_AnyTraced, starts::Tuple, sizes::Tuple) =
    Reactant.Ops.dynamic_slice(_mat(x), Any[starts...], Int[sizes...])
BSF._dupdate_at(x::_AnyTraced, chunk, starts::Tuple) =
    Reactant.Ops.dynamic_update_slice(_mat(x), _mat(chunk), Any[starts...])
BSF._plain(x::_AnyTraced) = _mat(x)

# Dense DD contractions as single XLA dot_general ops (result dims: batch…, lhs free…, rhs free…).
function BSF._bmm_t(Wx::_AnyTraced, V::_AnyTraced)
    Wxm = _mat(Wx); Vm = _mat(V)                             # (n_cols,n_t,n_long,B), (n_t,K,n_long)
    n_cols, n_t, n_long, B = size(Wxm); K = size(Vm, 2)
    Vb = Vm .* Reactant.Ops.fill(one(Reactant.unwrapped_eltype(Vm)), [1, 1, 1, B])   # (n_t,K,n_long,B)
    r = Reactant.Ops.dot_general(Wxm, Vb; contracting_dimensions = ([2], [1]), batching_dimensions = ([3, 4], [3, 4]))
    # r :: (n_long, B, n_cols, K) → (n_cols, K, n_long, B)
    return permutedims(r, (3, 4, 1, 2))
end
function BSF._bmm_rows(Wz::_AnyTraced, S::_AnyTraced)
    Wzm = _mat(Wz); Sm = _mat(S)                             # (n_cols,n_rows,nz,n_long,B), (n_cols,n_rows,B)
    r = Reactant.Ops.dot_general(Wzm, Sm; contracting_dimensions = ([2], [2]), batching_dimensions = ([1, 5], [1, 3]))
    # r :: (n_cols, B, nz, n_long) → (n_cols, nz, n_long, B)
    return permutedims(r, (1, 3, 4, 2))
end
function BSF._bmm_cols(Wx::_AnyTraced, G::_AnyTraced)
    Wxm = _mat(Wx); Gm = _mat(G)                             # (n_cols,n_t,n_long,B), (n_cols,nz,n_long,B)
    r = Reactant.Ops.dot_general(Wxm, Gm; contracting_dimensions = ([1, 4], [1, 4]), batching_dimensions = ([3], [3]))
    # r :: (n_long, n_t, nz) → (n_t, nz, n_long)
    return permutedims(r, (2, 3, 1))
end
function BSF._bmm_tb(Wx::_AnyTraced, Vw::_AnyTraced)
    Wxm = _mat(Wx); Vm = _mat(Vw)
    r = Reactant.Ops.dot_general(Wxm, Vm; contracting_dimensions = ([2], [1]), batching_dimensions = ([3, 4], [3, 4]))
    return permutedims(r, (3, 4, 1, 2))                       # (n_long, B, n_cols, K) → (n_cols, K, n_long, B)
end
BSF._scalar_start(st::_AnyTraced, j::Int, b::Int) = _mat(st)[j, b]     # TracedRNumber{Int32}
function BSF._bmm_zl(Wz::_AnyTraced, A::_AnyTraced)
    Wzm = _mat(Wz); Am = _mat(A)                             # (n_cols,n_rows,nz,n_long,B), (n_cols,nz,M,n_long,B)
    r = Reactant.Ops.dot_general(Wzm, Am; contracting_dimensions = ([3, 4], [2, 4]), batching_dimensions = ([1, 5], [1, 5]))
    # r :: (n_cols, B, n_rows, M) → (n_cols, n_rows, M, B)
    return permutedims(r, (1, 3, 4, 2))
end

# Reactant 0.2.28x caps the number of same-named elementwise helper functions per module at
# 10 000 (`__lookup_unique_name_in_module` probes name, name_1, … against a freshly built symbol
# table on every call). A single full pipeline exceeds it once every stage loops over views.
# Replaced at load time by a monotonic per-name counter (no cap, O(1) per call); names stay
# unique, so the emitted MLIR is unchanged.
# Counters are PER MODULE (reset whenever a new module is being built): the entry function of
# every compile keeps its plain name, exactly as Reactant's own symbol-table probe would give it.
const _UNIQUE_NAME_COUNTERS = Dict{String, Int}()
const _UNIQUE_NAME_MODULE = Ref{Any}(nothing)
function __init__()
    @eval Reactant.TracedUtils function __lookup_unique_name_in_module(mod, name)
        if $_UNIQUE_NAME_MODULE[] !== mod
            empty!($_UNIQUE_NAME_COUNTERS)
            $_UNIQUE_NAME_MODULE[] = mod
        end
        i = get($_UNIQUE_NAME_COUNTERS, name, 0)
        $_UNIQUE_NAME_COUNTERS[name] = i + 1
        return i == 0 ? name : name * "_" * string(i)
    end
    return nothing
end

end # module
