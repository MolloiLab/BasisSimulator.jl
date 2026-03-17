# 10x Speed Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** COMPLETE
**Status:** SPEED_COMPLETE

## Summary

The 10x speed discovery is **complete**. Five iterations produced a concrete,
implementable roadmap with 3 prioritized stories that stack to 10.9× speedup:

1. **Energy loop fusion (P0):** 30s → 3s on forward projection (10×)
2. **Separable Gaussian scatter (P1):** 2s → 0.065s on scatter (31×)
3. **Branchless DDA (P1):** 1.15× on DDA inner loop

**Conservative stacked total: 7.5× (tiled fusion). Optimistic: 10.9× (full fusion).**

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
- Post-fusion scatter becomes 40% of total → separable Gaussian = 31× fix

**Iteration 5 (SPEED-002+003 CRITIQUE + SPEED-004 + SPEED-007 SYNTHESIS):**
- Validated separable scatter: confirmed Gaussian IS separable (read source code)
- Validated signal chain: scatter >95% of physics pipeline, elementwise fusion not worth it
- Wrote complete synthesis with 3 stories, acceptance tests, risk register

## What to do next

**SPEED_COMPLETE.** Hand off to build loop agent for implementation.

Implementation order:
1. Story 1: Energy loop fusion (Week 1 — the big win)
2. Story 2: Separable scatter (Week 2 — the missing piece for 10×)
3. Story 3: Branchless DDA (Week 1, alongside Story 1 — free performance)
4. Story 4: CartesianIndices fix (Week 2, trivial cleanup)

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| SPEED-000 Profiling Audit | **done** | **done** | **done** (in synthesis) |
| SPEED-001 Energy Loop Fusion | **done** | **done** | **done** (in synthesis) |
| SPEED-002 Ray Tracing Algorithm | **done** | **done** | **done** (in synthesis) |
| SPEED-003 GPU Kernel Optimization | **done** | **done** | **done** (in synthesis) |
| SPEED-004 Physics Pipeline Batching | **done** | **done** | **done** (in synthesis) |
| SPEED-005 Reference Implementations | **done** (inline) | N/A | N/A |
| SPEED-006 Precision Analysis | **done** (inline) | N/A | N/A |
| SPEED-007 SYNTHESIS | **done** | N/A | N/A |

## Completed iterations

1. **Iteration 1** — SPEED-000 DISCOVERY: Static profiling audit. Forward projection = 88-95%.
2. **Iteration 2** — SPEED-001 DISCOVERY: Energy loop fusion deep dive. 10-20× speedup, AK.jl kernel design.
3. **Iteration 3** — SPEED-000+001 CRITIQUE: Adjusted estimates, Metal register risk, tiled fallback.
4. **Iteration 4** — SPEED-002+003 DISCOVERY: Branchless DDA (1.1-1.3×), AK.jl analysis, separable scatter (31×).
5. **Iteration 5** — SPEED-002+003 CRITIQUE + SPEED-004 + SPEED-007 SYNTHESIS: Final roadmap complete.
