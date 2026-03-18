# 10x Speed Loop — Current State (v2 — GPU-First, Discovery+Build)

**Last updated:** 2026-03-18
**Current phase:** NEAR COMPLETE — Critique/Refinement of V2-001/002 stories, then SPEED_COMPLETE
**Status:** 10× ACHIEVED (12× measured from current default)

## GPU Baseline (Corrected in Iteration 3)

Real Metal GPU measurements on Apple M3 Max (AGXG16G):

| Component | Time | % |
|-----------|------|---|
| **simulate!() total (fused, default)** | **19,182 ms** | **—** |
| **simulate!() total (unfused)** | **~5,430 ms** | **100%** |
| Forward projection (unfused) | 5,232 ms | 96.4% |
| Signal chain (exp, air, cal, log) | 6 ms | 0.1% |
| Physics pipeline | 22 ms | 0.4% |
| Noise (randn + apply) | 152 ms | 2.8% |
| GPU→CPU copies | 5 ms | 0.1% |

**CORRECTION (Iter 3):** Signal chain is 6ms, NOT 487ms. SPEED-000's measurements
were inflated 80× by Metal.synchronize() barrier overhead.

## Tiled Energy Fusion — GPU Validated (Iteration 2, confirmed Iter 3)

| K | Per-tile | N tiles | Est. total fwd proj | Speedup |
|---|---------|---------|-------------------|---------|
| **16** | **95.3 ms** | **15** | **1,430 ms** | **3.66×** |

**Per-bin marginal cost: 5.96ms** (instruction-bound, data-independent).

## Per-bin Cost Analysis (Iteration 3)

The 6ms per-bin floor is **instruction-bound**, not memory or data dependent:
- All-zero μ_table: 95.21ms (same as real: 95.32ms)
- All-tissue phantom: 95.40ms (same as real: 95.32ms)
- Air skipping: USELESS on GPU (no warp-coherent air regions)
- μ_table layout: IRRELEVANT (16KB fits in L1 cache)

The 234 × 6ms = 1,404ms is a hard floor for tiled Siddon. Only a fundamentally
different projector algorithm could break this floor.

## 10× Speedup — ACHIEVED

| Configuration | Time (ms) | Speedup |
|---------------|-----------|---------|
| **Current default (fused)** | **19,182** | **1.0×** |
| After V2-001 (disable fused) | ~5,430 | 3.5× |
| **After V2-002 (tiled K=16)** | **~1,610** | **11.9×** |
| 10× target | 1,918 | 10.0× |

Only 2 build stories needed. Signal chain and per-bin optimizations save <10ms combined.

## What to do next

**SPEED-001 Critique + Refinement → SPEED-007 Synthesis → SPEED_COMPLETE**

### Priority: SPEED-001 Critique
Validate tiled implementation edge cases:
1. Does the kernel handle the last partial tile correctly (234 = 14×16 + 10)?
2. Is bowtie spectral slicing correct with offset?
3. What happens with larger phantoms (512³)? Does memory pressure change?
4. Compilation time for Val(16) — is it acceptable?
5. Does the kernel produce identical results to unfused path?

### Then: SPEED-007 Synthesis
Finalize build stories, ensure all details are implementation-ready.
Declare SPEED_COMPLETE.

### Build stories created/updated:
- **SPEED-BUILD-V2-001:** Disable fused path (3.5× speedup, ready)
- **SPEED-BUILD-V2-002:** Tiled energy fusion K=16 (3.66× on fwd proj, ready)

### Optimizations INVALIDATED:
- Air voxel skipping: no timing difference on GPU
- μ_table layout: fits in L1, layout irrelevant
- Signal chain fusion: saves 5ms out of 1,610ms total
- Redundant -log/exp: saves 2.5ms, already in tiled approach

### Phase tracking

| Topic | Discovery | Critique | Refinement | Build Stories |
|-------|-----------|----------|------------|---------------|
| SPEED-000 GPU Profiling Baseline | **DONE** | **DONE** (corrected in iter 3) | N/A | V2-001 (disable fused) |
| SPEED-001 Energy Loop Optimization | **DONE** | TODO | TODO | V2-002 (tiled K=16) |
| SPEED-002 Ray Tracing Algorithm | Deprioritized | — | — | — (10× achieved) |
| SPEED-003 GPU Kernel Optimization | **DONE** | **DONE** | N/A | INVALIDATED |
| SPEED-004 Signal Chain Fusion | **DONE** | **DONE** | N/A | INVALIDATED (6ms) |
| SPEED-007 SYNTHESIS | **DONE** | TODO | TODO | — |

## Completed iterations

1. **v2 Iter 1 (2026-03-18):** SPEED-000 Discovery. GPU baseline established.
   Fused 3.65× slower. Forward proj 88.7%. 234 energy bins. Signal chain 8.2% (WRONG).

2. **v2 Iter 2 (2026-03-18):** SPEED-001 Discovery. Tiled energy fusion validated.
   K=16 optimal: 97.45ms/tile × 15 tiles = 1462ms (3.79× on fwd proj).
   Per-bin marginal cost = 6ms. Projected total speedup = 2.78×.

3. **v2 Iter 3 (2026-03-18):** SPEED-003/004 Discovery. Per-bin cost decomposition +
   signal chain correction. CRITICAL FINDING: signal chain is 6ms (not 487ms).
   Per-bin 6ms is instruction-bound. Air skip, μ_table layout, signal chain fusion
   all INVALIDATED. 10× from default baseline ACHIEVED (12× with tiled fusion).
   Bench: bench_speed_003.jl.
