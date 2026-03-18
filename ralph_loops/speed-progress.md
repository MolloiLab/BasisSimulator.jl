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

---

## v2 Iteration 2 — SPEED-001 DISCOVERY: Tiled Energy Fusion (2026-03-18)

**Phase:** Discovery
**Topic:** SPEED-001 Energy Loop Optimization — Tiled Fusion via small Val(K) fused kernel
**GPU:** Apple M3 Max (AGXG16G) via Metal.jl
**Benchmark script:** `ralph_loops/bench_speed_001.jl`

### Approach

The existing `siddon_fused_poly_project!` kernel with Val(234) is 3.65× slower due to register
spilling (234 accumulators = 936 bytes). But at small Val(K), the kernel should be fast because
K accumulators fit in registers. The "tiled" approach: call the fused kernel ceil(234/K) times,
each time processing a tile of K energy bins.

This avoids 234 separate Siddon DDA traversals (each 22.5ms). Instead: ~15 DDA traversals
(at K=16), each processing 16 energies per voxel via μ_table lookup.

### GPU Benchmark Results

#### Reference baselines (this run)

| Metric | Time |
|--------|------|
| Unfused forward proj (234 bins) | 5546 ms |
| Single Siddon (Float32 volume) | 22.85 ms |

#### Tiled Fusion — Fused kernel at small Val(K)

| K | Per-tile time | N tiles | Est. total | Speedup vs unfused |
|---|--------------|---------|-----------|-------------------|
| 8 | 51.95 ms | 30 | 1558 ms | 3.56× |
| **16** | **97.45 ms** | **15** | **1462 ms** | **3.79×** |
| 32 | 195.23 ms | 8 | 1562 ms | 3.55× |

**Optimal tile size: K=16** (3.79× speedup on forward projection)

#### Per-bin marginal cost analysis

| K | Per-tile | Per-bin (tile/K) | DDA overhead (tile - K×6ms) |
|---|---------|-----------------|---------------------------|
| 1 (unfused) | 22.85 ms | 22.85 ms | ~17 ms |
| 8 | 51.95 ms | 6.49 ms | ~4 ms |
| 16 | 97.45 ms | 6.09 ms | ~1.5 ms |
| 32 | 195.23 ms | 6.10 ms | ~3 ms |

**Critical insight:** The marginal cost per energy bin is ~6ms (constant), vs 22.85ms for
separate Siddon calls. The DDA traversal overhead (~17ms per call) is shared across K bins.
At K=16+, DDA overhead is amortized away — the ~6ms per-bin cost dominates.

**Theoretical minimum for 234 bins: 234 × 6ms = 1404ms** (one giant tile with no DDA
overhead). K=16 at 1462ms is already within 4% of this limit. Increasing K beyond 16
yields no further benefit.

#### Overhead costs (negligible)

| Operation | Time |
|-----------|------|
| fill!(sinogram) | 0.34 ms |
| accumulate (+=) | 0.97 ms |
| -log(I_total) | 0.96 ms |

#### Full tiled estimate (DDA + accumulation + overhead)

| K | Est. total forward proj | Speedup vs unfused |
|---|------------------------|-------------------|
| 8 | 1560 ms | 3.56× |
| **16** | **1463 ms** | **3.79×** |
| 32 | 1563 ms | 3.55× |

### Key Findings

1. **Tiled fusion WORKS on Metal GPU** — 3.79× measured speedup on forward projection.
   The existing fused kernel at Val(16) compiles cleanly and runs without register spilling.

2. **Per-bin marginal cost is 6ms** — This is 3.8× cheaper than the 22.85ms single
   Siddon call because DDA traversal overhead is shared across K bins.

3. **K=16 is the optimal tile size.** K=8 and K=32 both ~3.55×. K=16 hits the sweet
   spot between tile count and per-tile cost.

4. **Diminishing returns beyond K=16.** The theoretical minimum (234 × 6ms = 1404ms) is
   only 4% better than K=16 (1462ms). Larger tiles don't help.

5. **The 6ms per-bin floor is the new bottleneck.** This is dominated by μ_table lookup +
   FMA accumulation + exp() in Beer-Lambert sum. To go faster, we'd need a fundamentally
   different projector algorithm (SPEED-002) or reduced precision.

6. **GPU compiler note:** The fused kernel requires a non-nothing bowtie spectral array.
   The Metal compiler can't optimize away dead branches with `Nothing` types. Pass a
   valid (possibly all-ones) array.

### Projected Total Speedup

| Component | Baseline | After tiled fusion | After + signal chain fusion |
|-----------|----------|-------------------|---------------------------|
| Forward proj | 5546 ms | 1462 ms (3.79×) | 1462 ms |
| Signal chain | 487 ms | 487 ms | ~200 ms (est. fused) |
| Physics + noise + copies | 183 ms | 183 ms | 183 ms |
| **Total simulate!()** | **6216 ms** | **2132 ms (2.9×)** | **~1845 ms (~3.4×)** |

**10× is NOT achievable with tiled fusion alone.** Tiled fusion is the biggest single win
(3.79× on the dominant cost), but it only reaches ~2.9-3.4× total. Reaching 10× requires
either: (a) a fundamentally faster projector algorithm (SPEED-002), or (b) reduced-precision
accumulation, or (c) some combination of multiple orthogonal optimizations.

### Implementation Notes for Build Story

The real tiled implementation needs a MODIFIED fused kernel that:
1. Takes `tile_start::Int32` and `Val(K)` — processes energies [tile_start, tile_start+K-1]
2. Offsets μ_table reads: `μ_table[mat, tile_start + e - 1]` instead of `μ_table[mat, e]`
3. Offsets bowtie reads: `bt[bt_base + (tile_start + e - 2) * ncnr]`
4. ADDS partial Beer-Lambert sum to `I_transmitted` buffer (no -log)
5. After all tiles: `sinogram[idx] = -log(max(I_transmitted[idx], eps))`
6. Pad μ_table and wη to be divisible by K (zeros for unused bins)

Can reuse 95% of the existing `siddon_fused_poly_project!` code.

### What's Next

SPEED-001 Discovery is DONE. Tiled fusion validated at 3.79× on Metal GPU.

Next priorities:
1. **SPEED-002 Discovery:** Explore alternative projector algorithms (distance-driven,
   Joseph's) that might reduce the 6ms per-bin floor cost on GPU.
2. **SPEED-004 Discovery:** Signal chain fusion (5 elementwise kernels → 1-2).
3. **SPEED-001 Critique:** What might go wrong with tiled implementation? Register pressure
   at K=16 with all DDA locals? Compilation time for multiple Val(K) specializations?
