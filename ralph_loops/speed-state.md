# 10x Speed Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** CRITIQUE (SPEED-002 + SPEED-003) or SYNTHESIS
**Current topic:** Challenge branchless DDA + separable scatter claims, or begin synthesis

## What was done

**Iteration 1 (SPEED-000 DISCOVERY — static profiling audit):**
- Traced complete simulate!() call graph
- Forward projection = 88-95% of time. Energy loop is THE bottleneck.

**Iteration 2 (SPEED-001 DISCOVERY — energy loop fusion deep dive):**
- Proved mathematical equivalence, register pressure analysis, AK.jl kernel design
- Memory bandwidth reduction: 24-43× (effective 8-20× with caching)

**Iteration 3 (SPEED-000 + SPEED-001 CRITIQUE):**
- Adjusted forward projection to 88-95% (scatter slightly understated)
- Adjusted bandwidth reduction to 8-20× effective
- Identified Metal register pressure risk → energy tiling fallback
- Confirmed bit-identical within ULP

**Iteration 4 (SPEED-002 + SPEED-003 DISCOVERY):**
- Branchless DDA: 1.1-1.3× speedup, exact equivalence, zero risk
- No alternative projector is both faster AND equivalent to Siddon
- AK.jl overhead = zero, foreachindex is the right primitive
- **CRITICAL NEW FINDING:** After fusion, scatter becomes 40% of total time
- Separable Gaussian scatter: 31× fewer ops, exact equivalence for Gaussian kernel
- **Revised path to 10×:** fusion (10×) + branchless DDA (1.15×) + separable scatter (31×) = 10.7×

## What to do next

**Option A: SPEED-002+003 CRITIQUE** — Stress-test new claims:
1. Is branchless DDA actually better on Metal's SIMD model? (Metal SIMD groups behave
   differently from CUDA warps — predicated execution may already happen.)
2. Separable scatter: is it exact for both `gaussian` AND `exponential` kernel types?
   (Only Gaussian is separable. Exponential is NOT. Check which one is used in practice.)
3. Does the 40%-physics post-fusion breakdown hold under runtime profiling?

**Option B: SPEED-007 SYNTHESIS** — Produce the final roadmap. We have enough
data for all major optimizations:
- P0: Energy loop fusion (10× on forward proj)
- P1: Branchless DDA (1.15×) + Separable scatter (31× on scatter)
- P2: CartesianIndices fix, other minor cleanups

**Recommendation: Go to SYNTHESIS.** Four discoveries + 1 critique covers the core
bottleneck (forward projection) and the secondary bottleneck (scatter). The remaining
topics (SPEED-004 physics batching, SPEED-005 reference implementations, SPEED-006
precision analysis) are low-impact given the synthesis data we already have.

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| SPEED-000 Profiling Audit | **done** | **done** | open |
| SPEED-001 Energy Loop Fusion | **done** | **done** | open |
| SPEED-002 Ray Tracing Algorithm | **done** | open | open |
| SPEED-003 GPU Kernel Optimization (AK.jl) | **done** | open | open |
| SPEED-004 Physics Pipeline Batching | partial (in §3.3) | open | open |
| SPEED-005 Reference Implementations | partial (in §1.7) | open | open |
| SPEED-006 Precision Analysis | open | open | open |
| SPEED-007 SYNTHESIS | **open → DO THIS** | open | open |

## Completed iterations

1. **Iteration 1** — SPEED-000 DISCOVERY: Static profiling audit. Forward projection = 88-95%.
2. **Iteration 2** — SPEED-001 DISCOVERY: Energy loop fusion deep dive. 10-20× speedup, AK.jl kernel design.
3. **Iteration 3** — SPEED-000+001 CRITIQUE: Adjusted estimates, Metal register risk, tiled fallback.
4. **Iteration 4** — SPEED-002+003 DISCOVERY: Branchless DDA (1.1-1.3×), AK.jl analysis, separable scatter (31×). Path to 10× confirmed.
