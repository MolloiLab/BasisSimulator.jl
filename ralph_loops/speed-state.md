# 10x Speed Loop — Current State (v2 — GPU-First, Discovery+Build)

**Last updated:** 2026-03-18
**Current phase:** DISCOVERY — SPEED-003 or SPEED-001 Critique next
**Status:** IN PROGRESS

## GPU Baseline Established (Iteration 1)

Real Metal GPU measurements on Apple M3 Max (AGXG16G):

| Component | Time | % |
|-----------|------|---|
| **simulate!() total (unfused)** | **5925 ms** | **100%** |
| Forward projection (unfused) | 5254 ms | 88.7% |
| Signal chain (exp, air, cal, log, etc.) | 487 ms | 8.2% |
| Noise (randn + apply) | 152 ms | 2.6% |
| Physics pipeline | 22 ms | 0.4% |
| GPU→CPU copies | 5 ms | 0.1% |

**CRITICAL:** Fused kernel is **3.65× SLOWER** on GPU (19.16s vs 5.25s).
234 energy bins causes register spilling on Metal.

## Tiled Energy Fusion — GPU Validated (Iteration 2)

| K | Per-tile | N tiles | Est. total fwd proj | Speedup |
|---|---------|---------|-------------------|---------|
| 8 | 51.95 ms | 30 | 1558 ms | 3.56× |
| **16** | **97.45 ms** | **15** | **1462 ms** | **3.79×** |
| 32 | 195.23 ms | 8 | 1562 ms | 3.55× |

**Per-bin marginal cost: 6ms** (constant). DDA overhead shared across K bins.
K=16 is 96% of theoretical optimum (234 × 6ms = 1404ms).

**Projected total:** tiled fwd (1462ms) + signal chain (487ms) + rest (183ms) = **2132ms → 2.78×**

## What to do next

**The 6ms per-bin floor is the new bottleneck.** Tiled fusion was the biggest single
win, but 10× requires reducing this floor OR finding additional orthogonal optimizations.

### Priority options for next iteration:

**Option A: SPEED-003 DISCOVERY — Reduce per-bin 6ms cost**
- Empty-space skipping: if mask[v] == 0 (air), skip all K accumulations
- μ_table layout optimization: test transposed layout for better coalescing
- Measure how much of 6ms is exp() vs table lookup vs FMA
- Could Float16 accumulation help? Must quantify error.

**Option B: SPEED-001 CRITIQUE — Validate tiled approach edge cases**
- Does the kernel handle the last partial tile correctly (234 = 14×16 + 10)?
- Is bowtie spectral slicing correct with offset?
- What happens with larger phantoms (512³)? Does memory pressure change?
- Compilation time for Val(16) — is it acceptable?

**Option C: SPEED-002 DISCOVERY — Alternative projector algorithms**
- Distance-driven projection: better memory coalescing on GPU?
- Joseph's method: trilinear interpolation instead of DDA stepping?
- Read TIGRE/ASTRA GPU implementations for reference
- Would need full prototype — high effort, high potential payoff

**Option D: SPEED-004 DISCOVERY — Signal chain fusion**
- Quick win: fuse exp→air→cal→-log into single kernel
- Saves ~287ms (4.8% of baseline) — easy to implement
- Low effort, guaranteed payoff

### Recommended: Option A (SPEED-003) — highest leverage

The per-bin 6ms cost × 234 bins = 1404ms. If we can halve it to 3ms, total becomes:
- Fwd proj: 234 × 3ms = 702ms (from 1462ms)
- Total: 702 + 487 + 183 = 1372ms → **4.3×**
With signal chain fusion: 702 + 200 + 183 = 1085ms → **5.5×**

### Build stories created/updated:
- **SPEED-BUILD-V2-001:** Disable fused path (3.65× speedup, ready)
- **SPEED-BUILD-V2-002:** Tiled energy fusion K=16 (3.79× on fwd proj, ready)

### Phase tracking

| Topic | Discovery | Critique | Refinement | Build Stories |
|-------|-----------|----------|------------|---------------|
| SPEED-000 GPU Profiling Baseline | **DONE** | N/A | N/A | V2-001 (disable fused) |
| SPEED-001 Energy Loop Optimization | **DONE** | TODO | TODO | V2-002 (tiled K=16) |
| SPEED-002 Ray Tracing Algorithm | TODO | — | — | — |
| SPEED-003 GPU Kernel Optimization | **TODO** | — | — | — |
| SPEED-004 Signal Chain Fusion | TODO | — | — | — |
| SPEED-007 SYNTHESIS | TODO | — | — | — |

## Completed iterations

1. **v2 Iter 1 (2026-03-18):** SPEED-000 Discovery. GPU baseline established.
   Fused 3.65× slower. Forward proj 88.7%. 234 energy bins. Signal chain 8.2%.

2. **v2 Iter 2 (2026-03-18):** SPEED-001 Discovery. Tiled energy fusion validated.
   K=16 optimal: 97.45ms/tile × 15 tiles = 1462ms (3.79× on fwd proj).
   Per-bin marginal cost = 6ms. Projected total speedup = 2.78×.
   Build story V2-002 created.
