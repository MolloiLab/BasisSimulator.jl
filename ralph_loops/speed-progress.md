# BasisSimulator.jl 10x Speed — Discovery Progress Log (v2 — GPU-First)

v1 progress archived in `speed-progress-v1.md`. v1 produced ZERO GPU speedup despite
claiming 6-10x from CPU-only benchmarks. v2 requires Metal GPU measurements for every claim.

---

## v2 Iteration 1 — SPEED-000 DISCOVERY: GPU Profiling Baseline (2026-03-18)

**Phase:** Discovery
**Topic:** SPEED-000 GPU Profiling Baseline
**GPU:** Apple M3 Max (AGXG16G) via Metal.jl
**Benchmark script:** `ralph_loops/bench_speed.jl`

### Setup

- Phantom: Gammex 472, 128×128×15, 35cm FOV
- Sinogram: 256×32×360 (2.95M elements)
- Scanner: GE Revolution-like (256 cols, 64 rows)
- Fidelity: `:high` (IPEM spectrum, 234 energy bins at 0.5 keV resolution)
- Protocol: 120 kVp, 150 mA, 360 views

### GPU Benchmark Results

#### Full simulate!() (5 runs, median)

| Run | Time |
|-----|------|
| 1 | 19.20s |
| 2 | 19.19s |
| 3 | 19.28s |
| 4 | 19.27s |
| 5 | 19.27s |
| **Median** | **19.27s** |

Note: This uses the FUSED path (default). The fused path is broken on GPU.

#### Forward Projection: FUSED vs UNFUSED (3 runs, median)

| Path | Median | Speedup vs Fused |
|------|--------|-----------------|
| FUSED | 19.16s | 1.00× |
| UNFUSED | 5.25s | **3.65× faster** |

**CRITICAL FINDING:** The v1 fused kernel (`siddon_fused_poly_project!`) is 3.65× SLOWER
on Metal GPU than the sequential unfused path. The fused kernel needs 234 Float32
accumulators per thread (936 bytes), causing massive register spilling on Metal.

Both paths produce identical results (max abs diff = 1.0e-6).

#### Step-by-Step simulate!() Timing (UNFUSED path)

| Step | Time | % |
|------|------|---|
| Forward projection (unfused) | 5254 ms | 88.7% |
| Physics pipeline | 26 ms | 0.4% |
| exp(-sino) | 112 ms | 1.9% |
| Heel effect | 3 ms | 0.0% |
| Air scan build | 133 ms | 2.2% |
| Calibration (÷ air) | 115 ms | 1.9% |
| Low signal correction | 4 ms | 0.1% |
| -log(sino) | 121 ms | 2.0% |
| BHC | 0 ms | 0.0% |
| GPU→CPU copy (ideal) | 3 ms | 0.1% |
| Noise (randn + apply) | 152 ms | 2.6% |
| GPU→CPU copy (noisy) | 1 ms | 0.0% |
| **Manual total** | **5925 ms** | **100%** |

#### Per-Energy-Bin Breakdown (10 runs, median)

| Component | Time | % of bin |
|-----------|------|----------|
| create_μ_volume! | 0.27 ms | 1.1% |
| siddon_forward_project! | 22.55 ms | 93.9% |
| fill + accumulate | ~1.2 ms | 5.0% |
| **Total per bin** | **~24.0 ms** | |
| × 234 bins = | **5614 ms** | |

10 sequential bins measured: 240ms (24ms/bin average, projecting to 5.6s for 234 bins).

#### Physics Pipeline

| Benchmark | Time |
|-----------|------|
| _apply_physics_no_noise!() (5 runs, median) | 21.8 ms |

The separable scatter convolution (v1 optimization) appears to be working well on GPU.
The entire physics pipeline is negligible at 0.4% of total time.

### Key Findings

1. **Fused kernel is 3.65× SLOWER on GPU.** Must disable fused path as default.
   The v1 "optimization" is actually a massive regression on Metal.

2. **234 energy bins is the real operating point.** v1 assumed ~30 bins. The IPEM
   spectrum at `:high` fidelity produces 234 bins. This is why the fused kernel
   fails — 234 accumulators per thread causes register spilling.

3. **Forward projection is 88.7% of time** (confirmed). Siddon ray tracing is
   93.9% of each energy bin's time.

4. **Signal chain is 8.2% of time.** Five separate elementwise kernels (exp, air,
   cal, log) each take ~120ms. Could be fused into 1 kernel.

5. **Physics pipeline is only 0.4%.** Separable scatter is fast. No optimization needed.

6. **Per Siddon call: 22.5ms** for 2.95M rays through 128³ volume. This is the
   atomic unit we need to make faster or call fewer times.

### Optimization Implications

To reach 10× (5925ms → ~593ms):

- **Option A: Reduce energy bins.** 234→30 bins = 7.8× fewer Siddon calls.
  30 × 24ms = 720ms forward proj + 487ms chain = 1207ms → 4.9× speedup.
  Need to quantify error from spectrum downsampling.

- **Option B: Tiled energy fusion.** Process 8-16 bins per DDA pass.
  234/8 = ~30 passes, each with 8 accumulators (32 bytes — fits in registers).
  Each pass reads mask once (not μ_volume). Saves DDA + volume read cost.
  If siddon per-tile-pass ≈ 25ms: 30 × 25ms = 750ms. But unclear if AK.jl supports this.

- **Option C: Faster Siddon kernel.** Optimize memory access pattern, better coalescing.
  Maybe 1.5-2× per call → 234 × 12ms = 2.8s. Only 2.1× total — not enough alone.

- **Option D: Combine A+B.** Downsample to ~50 bins, tile at 10/pass = 5 passes.
  5 × 25ms = 125ms forward proj. + 120ms fused chain = 245ms → **24× speedup**.

### What's Next

SPEED-000 Discovery is DONE. Baseline established with real GPU numbers.

Next priorities:
1. **SPEED-001 Discovery:** Test spectrum downsampling error (234→50→30 bins).
   Must quantify if the error is below detector noise floor.
2. **SPEED-003 Discovery:** Test tiled energy fusion prototype on Metal.
3. **SPEED-004 Discovery:** Test signal chain fusion (5 kernels → 1).

The fused kernel should be DISABLED as default immediately (it's a 3.65× regression).
