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
function BSF._batched_loop(body, n::Int, state, ::Reactant.TracedRArray)
    @trace track_numbers = false for b in 1:n
        state = body(state, b)
    end
    return state
end
function BSF._dslice(x::Reactant.TracedRArray{<:Any, N}, start, len::Int, dim::Int) where {N}
    starts = Any[d == dim ? start : 1 for d in 1:N]
    sizes = Int[d == dim ? len : size(x, d) for d in 1:N]
    return Reactant.Ops.dynamic_slice(x, starts, sizes)
end
function BSF._dupdate(x::Reactant.TracedRArray{<:Any, N}, chunk, start, dim::Int) where {N}
    starts = Any[d == dim ? start : 1 for d in 1:N]
    return Reactant.Ops.dynamic_update_slice(x, chunk, starts)
end
BSF._zeros(::Reactant.TracedRArray, ::Type{T}, dims::Dims) where {T} =
    Reactant.Ops.fill(zero(T), collect(Int, dims))

# Reactant 0.2.28x caps the number of same-named elementwise helper functions per module at
# 10 000 (`__lookup_unique_name_in_module` probes name, name_1, … against a freshly built symbol
# table on every call). Two full pipelines in one graph exceed it. Opt-in override: a monotonic
# per-name counter (no cap, O(1) per call). Names stay unique, so the emitted MLIR is unchanged.
const _UNIQUE_NAME_COUNTERS = Dict{String, Int}()
function _uncap_reactant_names!()
    @eval Reactant.TracedUtils function __lookup_unique_name_in_module(mod, name)
        i = get($_UNIQUE_NAME_COUNTERS, name, 0)
        $_UNIQUE_NAME_COUNTERS[name] = i + 1
        return i == 0 ? name : name * "_" * string(i)
    end
    return nothing
end
__init__() = (BSF._UNCAP_REACTANT_HOOK[] = _uncap_reactant_names!; nothing)

end # module
