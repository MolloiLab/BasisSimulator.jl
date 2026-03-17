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

## 1. Energy Loop Fusion — Deep Analysis (SPEED-001)

### 1.1 Mathematical Proof of Equivalence

**Current implementation** (sequential, per energy bin `e`):
```
For each energy e in 1:N_E:
    μ_vol[x,y,z] = μ_table[mask[x,y,z]+1, e]           // create_μ_volume!
    L_e = Σ_v μ_vol[v] × path_v                         // siddon_trace_ray (DDA)
    I_total += w_e × η_e × exp(-L_e)                    // accumulate
sinogram = -log(I_total)
```

**Proposed fused kernel** (single DDA pass, all energies inline):
```
For each ray (col, row, angle):
    accum[1:N_E] = 0
    // DDA traversal through mask (ONCE)
    for each voxel v along ray:
        mat = mask[v]
        path = path_length(v)
        for e in 1:N_E:
            accum[e] += μ_table[mat+1, e] × path
    // Beer-Lambert after traversal
    I = Σ_e w_e × η_e × exp(-accum[e])
    sinogram = -log(I)
```

**Proof:** The line integral for energy `e` is:
```
L_e = Σ_v μ(v, e) × path_v
    = Σ_v μ_table[mask[v]+1, e] × path_v    (by definition of create_μ_volume!)
```

In the current code, `siddon_trace_ray` traverses voxels in DDA order and accumulates
`μ_vol[v] × path_v`. Since `μ_vol[v] = μ_table[mask[v]+1, e]`, the terms are identical.

In the fused kernel, `accum[e]` accumulates the same terms in the same DDA traversal
order. Therefore `accum[e] = L_e` exactly — same floating-point operations in the same
order, producing **bit-identical results**.

The final Beer-Lambert computation `I = Σ_e w_e × η_e × exp(-accum[e])` is the same
elementwise sum that the current accumulation kernel computes. **QED.**

**Bowtie spectral transmission:** The current code multiplies `bt[col, row, e]` into
each energy's contribution. In the fused kernel, this is handled identically after
traversal: `I += w_e × η_e × bt[col, row, e] × exp(-accum[e])`. The bowtie factor
is per-pixel per-energy (not per-voxel), so it doesn't affect the DDA loop.

**PCCT path equivalence:** The PCCT energy loop (`pcct_forward_project`, lines 1340-1398
of `photon_counting.jl`) has the same structure. Instead of accumulating a single
`I_total`, it distributes to `n_bins` outputs via a DRM: `bins[b] += I0 × w × η × R[e,b] × exp(-L_e)`.
The fused kernel computes the same `L_e = accum[e]` values, then applies the DRM after
traversal. Mathematically identical.

### 1.2 Register Pressure Analysis

**Registers needed per GPU thread (fused kernel):**

| Component | Count | Bytes | Notes |
|-----------|-------|-------|-------|
| DDA state (ix, iy, iz, step_x/y/z) | 6 Int32 | 24 | Mutable traversal state |
| DDA timing (t_next_x/y/z, t_current, dt_x/y/z) | 7 Float32 | 28 | DDA step parameters |
| Ray geometry (ray_x/y/z, ray_length) | 4 Float32 | 16 | Per-ray constants |
| Volume params (captured in closure) | ~16 Float32 | 64 | vol_min/max, voxel_size, etc. |
| Index decomposition | 3 Int32 | 12 | col, row, angle |
| Detector geometry | 9 Float32 | 36 | src, det_center, u, v per angle |
| **Energy accumulators** | **30 Float32** | **120** | **One per energy bin** |
| Per-voxel temporaries | 3 | 12 | mat, path, t_next |
| **Total** | **~78** | **~312 bytes** |

**Platform register budgets:**

| GPU | Max registers/thread | 78 regs = | Occupancy estimate |
|-----|---------------------|-----------|-------------------|
| CUDA (Ampere) | 255 | 31% used | ~48% occupancy (limited by blocks/SM) |
| CUDA (Ada) | 255 | 31% used | ~48% occupancy |
| Metal (M1 Max) | Automatic spill | N/A | Compiler manages; 312B reasonable |
| Metal (M3 Max) | Automatic spill | N/A | Should fit in register file |

**Verdict: 30 accumulators fit comfortably on all targets.**

Apple Metal GPUs use a register spill model — the compiler automatically spills to
threadgroup memory if registers are exhausted. 312 bytes/thread is well within typical
limits. CUDA GPUs have 255 registers (1020 bytes) per thread, and 78 registers leaves
ample headroom.

**Fallback: energy tiling.** If register pressure becomes an issue on some backend:
- Tile energy dimension: process 8-10 energies per DDA traversal pass
- 4 passes × 8 energies = 32 (only 8 accumulators per pass = 32 bytes)
- Still reads mask 4× (not 30×): 4 × 17.5 MB = 70 MB vs. current 30 × 70 MB = 2.1 GB
- 7.5× bandwidth reduction even with tiling

### 1.3 AK.jl Fused Kernel Design

**Primary entry point: `siddon_fused_poly_project!`**

```julia
function siddon_fused_poly_project!(
    sinogram::AbstractArray{T,3},
    mask::AbstractArray{UInt8,3},
    geom::CTGeometry,
    μ_table_gpu,          # [n_materials, n_energies] on GPU
    weights_gpu,          # [n_energies] on GPU (normalized)
    η_gpu;                # [n_energies] on GPU
    volume_extent = nothing,
    ws_source_positions = nothing,
    ws_detector_centers = nothing,
    ws_detector_u = nothing,
    ws_detector_v = nothing,
    ws_bowtie_spectral = nothing   # [n_cols, n_rows, n_energies] or nothing
) where T

    n_energies = Int32(size(μ_table_gpu, 2))
    # ... volume bounds, geometry setup identical to current siddon_forward_project! ...

    AK.foreachindex(sinogram) do idx
        # --- Index decomposition (same as current) ---
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)
        angle = (idx_0 ÷ n_rows) + Int32(1)

        # --- Source/detector geometry (same as current) ---
        # src_x, src_y, src_z, det_x, det_y, det_z from geometry arrays

        # --- DDA setup (same as siddon_trace_ray) ---
        # ray direction, t_enter/t_exit, initial ix/iy/iz, step/dt/t_next

        # --- Initialize energy accumulators ---
        # Option A: NTuple (immutable, compiler optimizes to registers)
        accums = ntuple(_ -> zero(T), Val(N_E))  # N_E = compile-time constant

        # --- DDA traversal (same loop structure as current) ---
        while t_current < t_exit && iter < max_iter
            iter += Int32(1)
            if ix < Int32(0) || ix >= nx || ...
                break
            end

            t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
            path_length = (t_next - t_current) * ray_length

            if path_length > eps
                mat = Int32(mask[ix+1, iy+1, iz+1]) + Int32(1)

                # INNER ENERGY LOOP: accumulate line integrals
                accums = ntuple(Val(N_E)) do e
                    accums[e] + μ_table_gpu[mat, e] * path_length
                end
            end

            # Step to next voxel (same as current)
            # ...
        end

        # --- Beer-Lambert summation (replaces separate accumulation kernel) ---
        I_total = zero(T)
        for e in 1:N_E
            w_η = weights_gpu[e] * η_gpu[e]
            if ws_bowtie_spectral !== nothing
                bt_val = ws_bowtie_spectral[col + (row-1)*n_cols + (e-1)*n_cols*n_rows]
                I_total += w_η * bt_val * exp(-accums[e])
            else
                I_total += w_η * exp(-accums[e])
            end
        end

        sinogram[idx] = -log(max(I_total, T(1e-10)))
    end
end
```

**Key design decisions:**

1. **NTuple accumulators:** Julia NTuples are stored in registers on GPU. The `ntuple(Val(N_E)) do e ... end` pattern generates unrolled register updates.

2. **N_E as compile-time constant:** Use `Val{N}` or generated functions so the energy loop unrolls at compile time. This avoids dynamic loops on GPU.

3. **μ_table in GPU global memory:** At 1.8 KB (15 mat × 30 energy × 4B), this fits in L1 cache and will be cached after first access. All threads in a warp read the same material's row → broadcast.

4. **Single kernel launch:** Replaces 90+ kernel launches (30 create_μ + 30 siddon + 30 accumulate) with exactly 1.

5. **mask reads UInt8:** 4× smaller than Float32 volume → 4× more cache-friendly. 17.5 MB mask fits in L2 cache on most GPUs.

**PCCT variant: `siddon_fused_pcct_project!`**

Same DDA structure, but after traversal distributes to n_bins outputs via DRM:
```julia
# After DDA traversal:
for b in 1:n_bins
    count_b = zero(T)
    for e in 1:N_E
        R_val = DRM_gpu[e, b]   # spectral response
        count_b += I0 * w[e] * η[e] * R_val * exp(-accums[e])
    end
    bins_gpu[b][col, row, angle] = count_b
end
```

### 1.4 Memory Access Pattern Comparison

#### Current: Sequential Energy Loop

| Data | Size | Accesses | Total traffic | Pattern |
|------|------|----------|---------------|---------|
| mask → μ_volume (write) | 70 MB | 30× | 2.1 GB | Sequential write |
| mask (read for μ creation) | 17.5 MB | 30× | 525 MB | Sequential read |
| μ_volume (Siddon reads) | 70 MB | 30× @ 57.6M rays × 400 vox | **2,760 GB** | **Scattered random** |
| sino_mono (write, clear) | 230 MB | 30× × 2 | 13.8 GB | Sequential |
| I_transmitted (accum) | 230 MB | 30× RW | 13.8 GB | Sequential |
| **TOTAL** | | | **~2,790 GB** | |

#### Fused: Single Pass

| Data | Size | Accesses | Total traffic | Pattern |
|------|------|----------|---------------|---------|
| mask (UInt8, Siddon reads) | 17.5 MB | 1× @ 57.6M rays × 400 vox | **23 GB raw** | Scattered random |
| ↳ with L2 cache (17.5 MB fits) | | | **~50-100 GB effective** | Cached random |
| μ_table (read) | 1.8 KB | Continuously from L1 | **~0 effective** | L1 cached |
| weights, η (read) | 240 B | Once per ray × N_E | **~7 GB** | L1 cached |
| bowtie_spectral (read) | 6.9 MB | Once per ray × N_E | **~7 GB** | Coalesced |
| sinogram (write) | 230 MB | 1× | **230 MB** | Sequential |
| **TOTAL** | | | **~65-115 GB** | |

**Bandwidth reduction: 2,790 GB → ~65-115 GB = 24-43×**

Since Siddon is memory-bandwidth bound, this translates almost directly to speedup.
Combined with eliminating 89 kernel launches and 30 synchronization points,
**the expected speedup on forward projection is 15-30×**.

Since forward projection is 90-97% of simulate!(), **total simulate!() speedup: 10-20×**.

### 1.5 Additional Benefits of Fusion

1. **Kernel launch elimination:** 90+ launches → 1. Each GPU kernel launch has 5-20μs
   overhead (dispatch + synchronization). At 90 launches: 0.5-1.8ms wasted. Small in
   absolute terms, but meaningful at short runtimes.

2. **No temporary μ_volume allocation.** Saves 70 MB GPU memory (XCAT f4) or 560 MB
   (XCAT f2). The fused kernel doesn't need it at all.

3. **No temporary sino_mono allocation.** Saves another 230 MB. The monochromatic
   sinogram is never materialized — line integrals live in per-thread registers.

4. **Better GPU utilization.** Current code has 30 sync barriers (between kernel
   launches). Fused kernel runs one continuous computation with maximum parallelism.

### 1.6 Implementation Complexity and Risks

**Complexity: HIGH but well-defined.**

The fused kernel is a direct mechanical combination of existing code:
- DDA traversal: copy from `siddon_trace_ray` (lines 150-293 of siddon.jl)
- Inner energy loop: new but trivial (1 multiply-add per energy per voxel)
- Beer-Lambert sum: copy from current accumulation kernel
- Geometry setup: copy from `siddon_forward_project!`

**Risks:**

1. **NTuple code generation for variable N_E.** If N_E varies at runtime (e.g., 20 vs 30
   energy bins), we need either: (a) a generated function dispatching on Val{N_E}, or
   (b) a fixed maximum N_E with unused bins zeroed. Option (b) is simpler and
   has negligible cost (unused bins = 0 weight = free FMAs).

2. **GPU compiler register allocation.** If the compiler spills accumulators to global
   memory, the kernel becomes write-bandwidth-bound instead of read-bandwidth-bound.
   Mitigation: test with small N_E first, monitor register usage.

3. **Closure size for AK.foreachindex.** The closure captures μ_table_gpu, weights_gpu,
   η_gpu, bowtie_spectral, geometry arrays, and volume parameters. This is more state
   than the current Siddon closure. Verify AK.jl doesn't have a closure size limit on
   Metal/CUDA backends.

4. **Debugging difficulty.** A fused kernel with 200+ lines is harder to debug than
   separate modular functions. Mitigation: keep the unfused path for validation.

### 1.7 Reference Implementations

**CatSim/XCIST (GE Research):** Uses sequential energy loop in "accurate" mode (like our
current implementation). Their "fast" mode uses basis material decomposition (2 basis
sinograms instead of 30), which is an approximation — not applicable here.

**gVXR (GPU Virtual X-Ray):** Fuses the energy loop in OpenGL fragment shaders. Each
fragment (detector pixel) traces through the volume once and accumulates across all
energies. This is exactly the approach proposed here.

**TIGRE:** Monochromatic only. External energy loop required for polychromatic simulation
(same as our current unfused approach).

**Key takeaway:** Energy loop fusion for polychromatic GPU ray tracing is an established
technique, but rarely implemented in research codes because it requires careful kernel
engineering. BasisSimulator.jl would be among the first open-source Julia implementations.

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
