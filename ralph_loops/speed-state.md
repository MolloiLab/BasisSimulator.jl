# 10x Speed Loop — Current State (v2 — GPU-First, Discovery+Build)

**Last updated:** 2026-03-18
**Current phase:** RESET — Starting over with GPU-first approach
**Status:** IN PROGRESS

## v2 Design: Discovery Creates Build Stories

The loop is now unified:
1. **Discovery** profiles on Metal GPU, finds real optimizations, writes GPU-validated build stories
2. **Critique** re-measures on GPU, invalidates bogus claims, updates/removes stories
3. **Refinement** finalizes implementation details with GPU prototype benchmarks
4. **After discovery completes:** `speed-build-prd.json` has a complete set of GPU-validated stories
5. **Build loop** picks up those stories and implements them, verifying GPU speedup at each step

## Why v2?

v1 discovery (5 iterations) + v1 build (4 stories) produced ZERO GPU speedup despite
claiming 6-10x. Everything was CPU benchmarks and "static analysis." GPU performance
is fundamentally different from CPU. v2 requires Metal GPU measurements for every claim.

## What exists from v1 (ON THE BRANCH speed/fused-projection)

The v1 build loop committed 4 changes. These are implemented but **GPU-unvalidated**:
1. `siddon_fused_poly_project!()` — energy loop fusion (6.5x on CPU, unknown on GPU)
2. Separable Gaussian scatter convolution (147x on CPU, unknown on GPU)
3. Branchless DDA inner loop (no measured effect on CPU)
4. CartesianIndices → mod/div in 24 GPU closures

**These may be net-negative on GPU.** SPEED-000 must profile them on Metal
and determine what's actually helping vs hurting.

## What to do next

**SPEED-000 DISCOVERY: GPU Profiling Baseline**

This is the ONLY acceptable starting point. You must:

1. Run `simulate!()` on Metal GPU with `@elapsed` + `Metal.@sync`
2. Profile individual components: forward projection, scatter, each physics effect
3. Establish REAL timing baseline on GPU
4. Compare the CURRENT branch code (with v1 optimizations) against `fused=false` path
5. Determine which v1 changes actually help or hurt on GPU
6. **Write the first build stories if any v1 changes show real GPU speedup**

**DO NOT proceed to any other topic until SPEED-000 has real GPU numbers.**

### Phase tracking

| Topic | Discovery | Critique | Refinement | Build Stories |
|-------|-----------|----------|------------|---------------|
| SPEED-000 GPU Profiling Baseline | **TODO** | — | — | — |
| SPEED-001 Energy Loop Fusion | **TODO** | — | — | — |
| SPEED-002 Ray Tracing Algorithm | **TODO** | — | — | — |
| SPEED-003 GPU Kernel Optimization | **TODO** | — | — | — |
| SPEED-004 Physics Pipeline | **TODO** | — | — | — |
| SPEED-007 SYNTHESIS | **TODO** | — | — | — |

## Completed iterations

None yet for v2. (v1 archived in speed-progress-v1.md)
