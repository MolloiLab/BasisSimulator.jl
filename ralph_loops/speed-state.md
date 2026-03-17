# 10x Speed Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** CRITIQUE (SPEED-000 + SPEED-001)
**Current topic:** Challenge profiling estimates and fusion claims

## What was done

**Iteration 1 (SPEED-000 DISCOVERY — static profiling audit):**
- Traced complete simulate!() call graph through driver.jl, polychromatic.jl, siddon.jl, physics_pipeline.jl
- Analyzed all 14 detector physics effects for computational complexity
- Produced quantitative time breakdown with concrete parameters (XCAT f4 phantom)

**HEADLINE FINDING:** Forward projection (energy loop × Siddon ray tracing) is 90-97%
of simulate!() time. The 30× redundant DDA traversal across energy bins is the root cause.
Energy loop fusion is THE path to 10×.

**Iteration 2 (SPEED-001 DISCOVERY — energy loop fusion deep dive):**
- Proved mathematical equivalence of fused kernel (bit-identical to current)
- Analyzed register pressure: 78 regs/thread (fits CUDA and Metal comfortably)
- Designed AK.jl fused kernel using foreachindex + NTuple accumulators
- Quantified memory bandwidth reduction: 2,790 GB → 65-115 GB (24-43×)
- Confirmed PCCT path equally fusible (same DDA + accumulate, output via DRM)
- Reference: gVXR does this; CatSim/TIGRE don't (sequential or mono-only)

## What to do next

**SPEED-000 + SPEED-001 CRITIQUE — Challenge Every Claim**

Phase rotation: 2 discoveries done, time to stress-test the findings.

1. **Challenge the 90-97% forward projection claim:**
   - Is the static complexity analysis realistic?
   - Could AK.jl overhead, synchronization, or GPU launch patterns change the picture?
   - What if scatter (50×50 convolution) is more expensive than estimated?

2. **Challenge the 24-43× bandwidth reduction:**
   - Are the cache assumptions valid? (17.5 MB mask fits L2 on M1 Max?)
   - Random access patterns: is effective bandwidth really 50-100 GB for mask?
   - What about cache thrashing with 57.6M rays accessing 17.5M voxels?

3. **Challenge the NTuple register claim:**
   - Will Julia's GPU compiler (GPUCompiler.jl → LLVM) keep NTuples in registers?
   - Or will it generate tuple reconstruction code that spills to global memory?
   - Has anyone tested NTuple{30,Float32} in a KernelAbstractions.jl kernel?

4. **Challenge the "bit-identical" claim:**
   - Float32 addition order: is DDA traversal order truly deterministic?
   - Does `ntuple(Val(N))` generate the same FMA patterns as explicit loops?
   - Could compiler optimizations (e.g., FMA fusion) change results?

5. **Identify show-stoppers:**
   - AK.jl closure size limits on Metal backend
   - GPU compiler timeout for large kernels (200+ lines in one closure)
   - Edge cases: rays parallel to axes, empty rays, boundary voxels

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| SPEED-000 Profiling Audit | **done** | **open → DO THIS** | open |
| SPEED-001 Energy Loop Fusion | **done** | **open → DO THIS** | open |
| SPEED-002 Ray Tracing Algorithm | open | open | open |
| SPEED-003 GPU Kernel Optimization (AK.jl) | open | open | open |
| SPEED-004 Physics Pipeline Batching | open | open | open |
| SPEED-005 Reference Implementations | open | open | open |
| SPEED-006 Precision Analysis | open | open | open |
| SPEED-007 SYNTHESIS | open | open | open |

## Completed iterations

1. **Iteration 1** — SPEED-000 DISCOVERY: Static profiling audit. Forward projection = 90-97% of time. Energy loop fusion = path to 10×.
2. **Iteration 2** — SPEED-001 DISCOVERY: Energy loop fusion deep dive. Math proof, register analysis, AK.jl kernel design, 24-43× bandwidth reduction, PCCT fusible.
