# BasisSimulator.jl — 10x Speed Specification

> **Goal:** 10x or greater speedup of simulate!() with identical mathematical
> results. No cheating — same physics, same fidelity, same flexibility.
>
> **Scope:** simulate!() ONLY — forward projection + detector physics pipeline.
> Reconstruction (reconstruct!) is a separate function and is OUT OF SCOPE.
>
> **Hard constraint:** All GPU code MUST use AcceleratedKernels.jl (AK.jl).
> No backend-specific kernels. Metal/CUDA/ROCm/CPU portability is non-negotiable.
>
> **Method:** Profile first, optimize what matters, verify correctness after
> each change.

---

## 0. Profiling Baseline — Where Does the Time Go?

**Method:** Static analysis of source code call graph + computational complexity estimation.
Runtime profiling recommended as follow-up to validate these estimates.

### 0.1 simulate!() Call Graph

Both EICT and PCCT paths share the same bottleneck structure:

```
simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
│
├── STEP 1: _forward_project_poly!()                    ← 90-97% of time
│   │
│   └── for e_idx in 1:n_energies (~30 bins):           ← Sequential loop
│       ├── create_μ_volume!(μ_vol, mask, mats, E)      ← Elementwise GPU lookup (<1%)
│       ├── fill!(sino_mono, 0)                         ← Memset (<0.1%)
│       ├── siddon_forward_project!(sino_mono, μ_vol)   ← FULL ray trace (85-95%)
│       └── AK.foreachindex: I += w*η*exp(-sino_mono)   ← Elementwise accumulate (<1%)
│
├── STEP 2: apply_physics_effects!() / signal chain      ← 3-8% of time
│   ├── fill_factor          (elementwise add)           ← <0.001%
│   ├── flat_filter           (elementwise mult)         ← <0.01%
│   ├── scatter add          (2D conv, kernel ~50×50)    ← 1-3% (if enabled)
│   ├── scatter correct      (2D conv, kernel ~50×50)    ← 1-3% (if enabled)
│   ├── bowtie_filter        (elementwise mult)          ← <0.01%
│   ├── crosstalk            (3×3 conv)                  ← <0.01%
│   ├── optical_crosstalk    (3×3 conv)                  ← <0.01%
│   ├── focal_spot_blur      (small 2D conv)             ← <0.1%
│   ├── noise                (elementwise + blur)        ← <0.5%
│   ├── lag                  (temporal weighted sum)      ← <0.1%
│   └── bhc                  (polynomial eval)           ← <0.01%
│
├── STEP 3: Noise application (quantum + electronic)     ← <1%
│   ├── randn!() on CPU + copyto!() to GPU               ← CPU-bound
│   └── AK.foreachindex: Poisson + Gaussian noise         ← Elementwise
│
└── STEP 4: copyto!(cpu_output, gpu_sinogram)            ← <1%
```

### 0.2 Quantitative Complexity Analysis

**Reference parameters (XCAT factor-4 phantom, GE Apex Elite-like scanner):**

| Parameter | Value |
|-----------|-------|
| Volume (XCAT f4) | 400 × 350 × 125 = 17.5M voxels |
| Volume (XCAT f2) | 800 × 700 × 250 = 140M voxels |
| Detector | 900 cols × 64 rows = 57,600 pixels |
| Angles | 1,000 views |
| **Total rays** | **57.6M** |
| Energy bins | ~30 |
| Avg voxels traversed/ray | ~400 (for 400×350 volume) |
| Materials (XCAT) | ~15 tissue types |

#### Forward Projection (per energy bin)

| Sub-step | Operations | Est. GFLOP | Notes |
|----------|-----------|------------|-------|
| create_μ_volume! | 17.5M lookups | 0.02 | GPU table lookup, trivial |
| fill!(sino_mono) | 57.6M × 4B | 0.0 | memset, <1ms |
| **siddon_forward_project!** | **57.6M rays × 400 vox × 15 FLOP** | **345** | **Memory-bandwidth bound** |
| Beer-Lambert accumulate | 57.6M × (exp + mult + add) | 1.7 | Elementwise, trivial |
| **Subtotal per bin** | | **~347** | |

**30 energy bins total: ~10,400 GFLOP** ← but bandwidth-bound, not compute-bound.

#### Memory Bandwidth Analysis (THE Real Bottleneck)

Siddon is **memory-bandwidth bound** due to scattered volume access:

| Metric | Value |
|--------|-------|
| Volume size (f4, Float32) | 17.5M × 4B = 70 MB |
| Volume reads per bin | 57.6M rays × 400 vox × 4B = 92 GB |
| **Volume reads for 30 bins** | **2,760 GB** |
| Apple M1 Max bandwidth | ~400 GB/s (peak), ~50-100 GB/s (random access) |
| **Est. time for ray tracing** | **27-55 seconds** |
| μ-volume writes per bin | 17.5M × 4B = 70 MB |
| μ-volume writes for 30 bins | 2.1 GB (negligible) |

**Key insight:** The volume is re-read 30 times (once per energy bin), but with DIFFERENT
μ values each time. The ray geometry (DDA traversal) is computed 30× redundantly — same
source, same detector, same phantom shape, only the attenuation coefficients change.

#### Physics Pipeline

| Effect | Ops/element | Total GFLOP | Est. time | % of simulate!() |
|--------|-------------|-------------|-----------|-------------------|
| Scatter add | 3,136 (50×50 kernel) | 181 | 0.5-1s | 1-3% |
| Scatter correct | 3,136 | 181 | 0.5-1s | 1-3% |
| Detector blur | 225 (15×15 kernel) | 13 | <0.1s | <0.5% |
| Focal spot | ~100 | 5.8 | <0.05s | <0.2% |
| Lag (weighted) | 20 | 1.2 | <10ms | <0.1% |
| Crosstalk | 9 | 0.5 | <1ms | <0.01% |
| BHC polynomial | 5 | 0.3 | <1ms | <0.01% |
| Fill factor | 1 | 0.06 | <1ms | <0.001% |
| Noise (elem.) | 3 | 0.17 | <1ms | <0.01% |

### 0.3 Time Breakdown Summary

| Component | Est. Time (XCAT f4) | % of simulate!() |
|-----------|---------------------|-------------------|
| **Forward projection total** | **28-56s** | **90-97%** |
| ├── Siddon ray tracing (30×) | 27-55s | 85-95% |
| ├── create_μ_volume! (30×) | <0.3s | <1% |
| ├── Beer-Lambert accumulate (30×) | <0.3s | <1% |
| └── Kernel launch overhead (90+ launches) | 0.5-1s | 1-2% |
| **Physics pipeline** | **1-3s** | **3-8%** |
| ├── Scatter (if enabled) | 1-2s | 3-5% |
| └── All other effects | <0.5s | <2% |
| **Noise application** | <0.1s | <1% |
| **GPU↔CPU copies** | <0.1s | <1% |
| **TOTAL** | **~30-60s** | **100%** |

### 0.4 Root Cause Analysis

**Why is forward projection 90-97% of simulate!()?**

The energy-sequential architecture causes **30× redundant work**:

1. **30× redundant DDA traversal.** Every ray computes the same geometric path
   (entry point, voxel stepping, exit point) 30 times. The phantom geometry doesn't
   change between energies — only μ values change.

2. **30× redundant volume reads.** Each Siddon trace reads ~400 voxels from a 70MB
   volume. 30 bins × 92 GB/bin = 2.76 TB of memory traffic. The bottleneck is
   GPU memory bandwidth, not compute.

3. **30 separate kernel launches.** Each siddon_forward_project! call dispatches
   a fresh AK.foreachindex kernel. Plus 30 create_μ_volume! launches and 30
   accumulation launches = 90+ GPU kernel launches with synchronization.

4. **30× μ-volume creation.** The entire volume (17.5M voxels) is overwritten
   with new μ values for each energy bin, even though the mapping is a simple
   table lookup (material_id → μ_at_energy).

**The material mask (UInt8) is the invariant.** The phantom is stored as a UInt8 mask
(17.5M bytes = 17.5 MB). The μ_table mapping materials to energies is tiny
(15 materials × 30 energies × 4 bytes = 1.8 KB). A fused kernel that traces
through the mask ONCE and looks up μ for all energies at each voxel would
eliminate all four sources of redundancy.

### 0.5 Preliminary Optimization Opportunity Ranking

| Optimization | Target | Est. Speedup | Effort | Priority |
|-------------|--------|-------------|--------|----------|
| **Energy loop fusion** | Forward proj (90-97%) | **10-20×** | High | **P0** |
| Branchless DDA | Siddon kernel | 1.2-1.5× | Medium | P1 |
| Mixed precision geometry | Siddon kernel | up to 2× (if Float64→32) | Low | P1 |
| Physics pipeline fusion | Detector chain (3-8%) | 2× on pipeline | Low | P2 |
| Scatter kernel optimization | Scatter (1-3%) | 2-3× | Medium | P2 |

**Energy loop fusion alone can hit 10×.** Other optimizations stack multiplicatively
but have diminishing returns since they target the remaining 3-10%.

### 0.6 Validation: Must Confirm With Runtime Profiling

These are **static estimates** based on complexity analysis. Before implementing:

1. **Instrument with @elapsed** around each step of simulate!() to get wall-clock times
2. **Measure GPU utilization** — is Siddon actually bandwidth-bound or compute-bound?
3. **Count actual kernel launches** — AK.jl may batch/fuse internally
4. **Profile XCAT f2 vs f4** — does scaling match O(volume × rays × energies)?
5. **Measure scatter cost** — is the 50×50 kernel estimate correct?

**Action: Next iteration should run an instrumented simulation to validate or
correct these estimates.**

---

## 1. Energy Loop Analysis

(To be filled by SPEED-001 — ONLY if SPEED-000 shows this is significant)

---

## 2. Ray Tracing Algorithm Analysis

(To be filled by SPEED-002 — ONLY if SPEED-000 shows projection is a bottleneck)

---

## 3. GPU Kernel Optimization via AK.jl

(To be filled by SPEED-003)

---

## 4. Physics Pipeline Batching

(To be filled by SPEED-004)

---

## 5. Reference Implementation Benchmarks

(To be filled by SPEED-005)

---

## 6. Precision Analysis

(To be filled by SPEED-006)

---

## 7. SYNTHESIS: 10x Speedup Roadmap for simulate!()

(To be filled by SPEED-007 — blocked until profiling and top bottleneck analyses complete)

---
