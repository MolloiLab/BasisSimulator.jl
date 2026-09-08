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

end # module
