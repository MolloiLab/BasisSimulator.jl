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

# Batched-loop hooks (see src/functional/loop.jl): a StableHLO while loop over view batches
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
function BSF._batched_loop(body, n::Int, state, consts::Tuple, ::_AnyTraced)
    state = _mat(state)
    @trace track_numbers = false for b in 1:n
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

# Reactant 0.2.28x caps the number of same-named elementwise helper functions per module at
# 10 000 (`__lookup_unique_name_in_module` probes name, name_1, … against a freshly built symbol
# table on every call). A single full pipeline exceeds it once every stage loops over views.
# Replaced at load time by a monotonic per-name counter (no cap, O(1) per call); names stay
# unique, so the emitted MLIR is unchanged.
const _UNIQUE_NAME_COUNTERS = Dict{String, Int}()
function __init__()
    @eval Reactant.TracedUtils function __lookup_unique_name_in_module(mod, name)
        i = get($_UNIQUE_NAME_COUNTERS, name, 0)
        $_UNIQUE_NAME_COUNTERS[name] = i + 1
        return i == 0 ? name : name * "_" * string(i)
    end
    return nothing
end

end # module
