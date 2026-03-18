# 10x Speed Loop — Current State (v2 — GPU-First, Discovery+Build)

**Last updated:** 2026-03-18
**Current phase:** DISCOVERY — SPEED-001 next
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

**Per Siddon call:** 22.5ms for 2.95M rays through 128³ volume.

## What to do next

**SPEED-001 DISCOVERY: Energy Loop Optimization**

The dominant cost is 234 × siddon_forward_project! at 22.5ms each = 5.27s.
Two approaches to research:

### ~~Approach A: Spectrum Downsampling~~ — REJECTED (cheating, changes physics)

### Approach A: Tiled Energy Fusion (8-16 bins per DDA pass) — PRIMARY PATH
- Process 8-16 energy bins per DDA traversal of the mask
- 234/8 = 30 passes (vs 234 full Siddon calls)
- Each pass reads mask once, looks up μ for 8 energies at each voxel
- 8 accumulators = 32 bytes (fits in registers, unlike 234 = 936 bytes)
- **Must prototype on Metal GPU via AK.jl and measure**

### Approach B: Signal Chain Fusion
- 5-6 elementwise AK.foreachindex kernels → 1 fused kernel
- exp(-sino) → heel → air/cal → -log → BHC
- Each is ~120ms launch overhead on 2.95M elements
- Fusing saves ~367ms (6.2% of total)
- Low-hanging fruit, but won't reach 10× alone

### Priority:
1. Prototype tiled energy fusion on Metal GPU (SPEED-001/003) — highest leverage
2. Signal chain fusion (SPEED-004) — easy win, saves ~367ms
3. Siddon kernel optimization (SPEED-002) — if tiled fusion alone isn't enough

### Build stories created:
- SPEED-BUILD-V2-001: Disable fused path (3.65× speedup, ready to implement)

### Phase tracking

| Topic | Discovery | Critique | Refinement | Build Stories |
|-------|-----------|----------|------------|---------------|
| SPEED-000 GPU Profiling Baseline | **DONE** | N/A | N/A | V2-001 (disable fused) |
| SPEED-001 Energy Loop Optimization | **TODO** | — | — | — |
| SPEED-002 Ray Tracing Algorithm | TODO | — | — | — |
| SPEED-003 GPU Kernel Optimization | **TODO** | — | — | — |
| SPEED-004 Signal Chain Fusion | **TODO** | — | — | — |
| SPEED-007 SYNTHESIS | TODO | — | — | — |

## Completed iterations

1. **v2 Iter 1 (2026-03-18):** SPEED-000 Discovery. GPU baseline established.
   Fused 3.65× slower. Forward proj 88.7%. 234 energy bins. Signal chain 8.2%.
