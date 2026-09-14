# Run one Reactant smoke with every `@compile` / `@jit` traced at FULL f32 precision.
#
# XLA:GPU executes f32 `stablehlo.dot_general` at TF32 unless the op's precision_config says
# otherwise, which biases the ramp filter and shifts every reconstruction by ~-31 HU — and inflates
# any gradient whose loss residual is small. `compile_pipeline` sets this itself; the stage smokes
# call `@compile` directly, so they run inside `BSF.with_full_precision` here. On CPU (no TF32 path)
# this is a no-op, so one runner serves both backends.
#   julia --project=envs/reactant test/functional/reactant/run_full_precision.jl test/functional/reactant/smoke_fbp.jl
using Reactant, BasisSimulator
BasisSimulator.Functional.with_full_precision() do
    include(abspath(ARGS[1]))
end
