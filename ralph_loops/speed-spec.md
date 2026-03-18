# BasisSimulator.jl — 10x Speed Specification (v2 — GPU-First)

> **Goal:** 10x or greater speedup of simulate!() with identical mathematical
> results. No cheating — same physics, same fidelity, same flexibility.
>
> **Scope:** simulate!() ONLY — forward projection + detector physics pipeline.
> Reconstruction (reconstruct!) is a separate function and is OUT OF SCOPE.
>
> **Hard constraint:** All GPU code MUST use AcceleratedKernels.jl (AK.jl).
> No backend-specific kernels. Metal/CUDA/ROCm/CPU portability is non-negotiable.
>
> **Method:** Profile on Metal GPU first, optimize what the DATA shows, verify
> correctness and GPU speedup after each change.
>
> **v2 WARNING:** Everything below section 0 is from v1 (static analysis + CPU
> benchmarks only). ALL speedup claims are UNVALIDATED on GPU and may be wrong.
> v2 discovery will overwrite these sections with GPU-measured data.

---

## 0. Profiling Baseline — Where Does the Time Go?

**Method:** `time_ns()` + `Metal.synchronize()` on actual Metal GPU (AGXG16G — Apple M3 Max).
Benchmark script: `ralph_loops/bench_speed.jl`. Phantom: Gammex 472, 128×128×15, 35cm FOV.
Sinogram: 256×32×360. Scanner: GE Revolution-like. Fidelity: `:high` (IPEM spectrum).

### 0.1 GPU-Measured Timing Breakdown

**Test parameters:**

| Parameter | Value |
|-----------|-------|
| Phantom | 128 × 128 × 15 = 245,760 voxels (Gammex 472) |
| Sinogram | 256 × 32 × 360 = 2,949,120 elements |
| Energy bins | **234** (IPEM 0.5 keV resolution, 120 kVp) |
| Rays/view | 8,192 (256 × 32) |
| Total rays | 2,949,120 |
| Materials | 17 regions |

**Full simulate!() breakdown (UNFUSED path — correct baseline):**

```
simulate!() total:                      5925 ms  (100%)
├── _forward_project_poly!() UNFUSED:   5254 ms  (88.7%)
│   ├── Per energy bin (×234):
│   │   ├── create_μ_volume!:             0.27 ms  (1.1% of bin)
│   │   ├── siddon_forward_project!:     22.55 ms  (93.9% of bin)
│   │   └── fill + accumulate:            ~1.2 ms  (5.0% of bin)
│   │   └── Total per bin:               ~24.0 ms
│   └── 234 bins × 24ms =               5614 ms  (projected)
├── _apply_physics_no_noise!():           22 ms  (0.4%)
├── Signal chain steps:                  487 ms  (8.2%)
│   ├── exp(-sino):                      112 ms
│   ├── heel effect:                       3 ms
│   ├── air scan build:                  133 ms
│   ├── calibration (÷ air):             115 ms
│   ├── low signal correction:             4 ms
│   ├── -log(sino):                      121 ms
│   └── BHC:                               0 ms
├── Noise (randn + apply):              152 ms  (2.6%)
└── GPU→CPU copies (×2):                  5 ms  (0.1%)
```

### 0.2 Critical Finding: Fused Kernel is 3.65× SLOWER on Metal GPU

| Path | Time | vs Unfused |
|------|------|-----------|
| `_forward_project_poly!()` **UNFUSED** | **5.25s** | 1.00× (baseline) |
| `_forward_project_poly!()` **FUSED** | **19.16s** | **0.27× (3.65× slower!)** |
| Fused vs Unfused max abs diff | 1.0e-6 | (mathematically equivalent) |

**Why fused is slower with 234 bins:** The fused kernel needs 234 Float32 accumulators
per thread = 936 bytes of register state. On Metal, this causes massive register spilling
to device memory. The unfused path uses simple kernels with high occupancy — the 234
separate kernel launches are cheap (~0.1ms each) compared to the register pressure penalty.

**v1 analysis assumed ~30 energy bins.** At 30 bins (120 bytes), the fused kernel would
likely be faster. At 234 bins (936 bytes), it's catastrophically slower. The IPEM spectrum
at `:high` fidelity produces 234 bins — this is the real operating point.

### 0.3 Root Cause Analysis (GPU-Validated)

**Forward projection is 88.7% of simulate!() — confirmed on GPU.**

The dominant cost is `siddon_forward_project!` at 22.5ms per call × 234 calls = 5.27s.
This kernel traces 2.95M rays through a 128³ volume on GPU.

The signal chain (exp, air scan, calibration, log) at 487ms is the next biggest chunk (8.2%).
These are all elementwise `AK.foreachindex` kernels on a 2.95M sinogram, each taking ~120ms.
**That's 5-6 separate kernel launches doing essentially `sino[i] = f(sino[i])`.** Could be
fused into 1-2 kernel launches.

Physics pipeline (_apply_physics_no_noise!) is negligible at 22ms (0.4%). The v1 separable
scatter optimization may have helped here — the 2D convolution is now separable 1D.

### 0.4 Optimization Opportunities (GPU-Informed)

**To reach 10× speedup (5925ms → ~593ms):**

| Optimization | Target | Mechanism | Est. GPU Speedup | Priority |
|-------------|--------|-----------|-----------------|----------|
| **Tiled energy fusion** | Siddon kernel | DDA once per tile of 8-16 bins | **2-4× on fwd proj** | P0 |
| **Signal chain fusion** | 487ms → ~120ms | Fuse 5-6 elementwise kernels into 1 | **~4× on chain** | P1 |
| **Siddon kernel optimization** | 22.5ms per call | Better memory access, occupancy | **1.2-2× per call** | P1 |

**REJECTED: Spectrum downsampling (234→~30 bins).** This is CHEATING — it changes the
physics by reducing energy bins. The 234-bin IPEM spectrum at `:high` fidelity is the
correct operating point. Optimizations must work WITH all 234 bins, not skip them.

**Key insight:** The number of energy bins (234) is the dominant scaling factor.
Tiling the energy loop (trace DDA once, accumulate 8-16 bins per tile, repeat for all
tiles) avoids the register pressure problem while eliminating redundant DDA traversals.
The Siddon kernel itself is reasonably efficient — 22.5ms for 2.95M rays through 128³ volume.

### 0.5 What v1 Got Wrong

| v1 Claim | GPU Reality |
|----------|------------|
| ~30 energy bins | **234 bins** (IPEM :high) |
| Fused gives 10-20× speedup | **Fused is 3.65× SLOWER** (register spill at 234 bins) |
| Physics pipeline 3-8% | **0.4%** (separable scatter is fast) |
| Scatter 1-3% | **<0.1%** (separable 1D, not 2D) |
| Forward proj 28-56s | **5.25s** (smaller phantom, but ratio is right) |
| Branchless DDA helps | **Not measured yet** (likely negligible on GPU) |

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

## 2. Ray Tracing Algorithm Analysis (SPEED-002)

### 2.1 Current Algorithm: Siddon DDA

The current implementation uses Siddon's algorithm (1985) with a 3D Digital Differential
Analyzer (DDA). The inner loop (siddon.jl:258-291) traces through voxels one at a time,
accumulating `μ × path_length` at each step.

**Inner loop structure (the hot path):**
```
while t_current < t_exit:
    t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
    path_length = (t_next - t_current) × ray_length
    line_integral += volume[ix, iy, iz] × path_length

    // 3-WAY BRANCH — which axis boundary was crossed?
    if t_next_x ≤ t_next_y && t_next_x ≤ t_next_z:
        ix += step_x; t_next_x += dt_x
    elif t_next_y ≤ t_next_z:
        iy += step_y; t_next_y += dt_y
    else:
        iz += step_z; t_next_z += dt_z
```

**Per-iteration cost:** ~15-20 FLOPs + 1 scattered memory read (4B for Float32, 1B for UInt8 mask).
**Typical iterations per ray:** ~400 (for 400×350×125 volume).

### 2.2 GPU Warp Divergence Analysis

The 3-way branch causes **warp/SIMD divergence** on GPU:

**Problem:** Threads in the same 32-thread warp (CUDA) or SIMD group (Metal) must
execute the same instruction. When threads take different branches, execution serializes —
the warp executes EACH branch path sequentially, masking inactive threads.

**CT geometry makes this worse:** Rays from a fan-beam source hit different parts of the
volume. Adjacent detector pixels (mapped to adjacent threads in the same warp) have
similar but not identical ray directions. At any given DDA iteration, some threads step
in X, others in Y, others in Z.

**Divergence estimate for CT geometry:**
- Rays in the same row (same z-slice) have similar polar angles → tend to step in X/Y
- But the fan angle spreads them: center rays are nearly perpendicular, edge rays are oblique
- Typical divergence factor: **1.2-1.5×** (measured by TIGRE team on similar geometry)
- Worst case (isotropic rays): 3× (all three branches active)
- Best case (parallel beam): ~1× (nearly uniform stepping)

### 2.3 Branchless DDA — Eliminating Warp Divergence

**Technique:** Replace the 3-way branch with predicated (branchless) instructions.
All threads execute the same instructions; inactive steps multiply by 0.

```julia
# Branchless DDA inner loop
t_next = min(t_next_x, t_next_y, t_next_z, t_exit)
path_length = (t_next - t_current) * ray_length

if path_length > eps
    mat = mask[ix+1, iy+1, iz+1]
    # ... accumulate for all energies ...
end

# Branchless step: compute masks as Int32(0) or Int32(1)
mask_x = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
mask_y = Int32(1 - mask_x) * Int32(t_next_y <= t_next_z)
mask_z = Int32(1) - mask_x - mask_y

ix += mask_x * step_x
iy += mask_y * step_y
iz += mask_z * step_z

t_next_x += T(mask_x) * dt_x
t_next_y += T(mask_y) * dt_y
t_next_z += T(mask_z) * dt_z

t_current = t_next
```

**Mathematical equivalence:** EXACT. Same voxels visited in same order, same path lengths.
The masks are mutually exclusive (exactly one is 1, rest are 0), so the same axis step
occurs as the branching version. The only difference is that the GPU executes 6 multiply-add
instructions instead of branching into one of three 2-instruction blocks.

**Trade-off:**
- Eliminates divergence penalty: saves 1.2-1.5× on divergent warps
- Adds multiply instructions: 6 integer multiplies + 3 float multiplies = ~9 extra ops
- vs. saved: 2 comparisons + branch overhead per branching warp
- Net: **positive when divergence > 15%** (typical for CT is 20-50%)

**Expected speedup: 1.1-1.3× on the DDA inner loop.**

Since the DDA inner loop is ~70% of the ray trace (rest is setup + Beer-Lambert), and
ray tracing is 88-95% of simulate!() (after fusion, it becomes the entire forward projection
kernel), the effective impact is:

| Scenario | DDA speedup | Forward proj speedup | simulate!() speedup |
|----------|------------|---------------------|---------------------|
| Before fusion | 1.15× | 1.1× | 1.1× |
| After fusion (stacked) | 1.15× | 1.15× | 1.13× |

**Verdict: Small but free. Zero risk, exact equivalence, stacks with fusion.**

### 2.4 Alternative Projector Algorithms

#### Distance-Driven Projection (De Man & Basu 2002)

- Projects voxel boundaries and detector boundaries onto a common line
- Computes overlap lengths between projected intervals
- **More regular memory access** (sequential voxel access along projection direction)
- Used in CatSim/XCIST, some clinical scanners (GE)

**Equivalence to Siddon: NO.** Distance-driven computes area-weighted overlaps, not exact
ray-voxel intersection lengths. Produces quantitatively different sinograms. Generally
considered BETTER (fewer aliasing artifacts, more uniform noise), but DIFFERENT.

**GPU suitability:** Better than Siddon — more coalesced memory access, less divergence.
Typically 1.5-2× faster than Siddon on GPU for the same volume/detector size.

**For BasisSimulator:** Would be a new projector (different math → different validation
baseline), not a speedup of the existing one. Worth implementing as an OPTION but not
as a replacement — it changes the output.

#### Separable Footprint (Long, Fessler, Balter 2010)

- Decomposes 3D cone-beam projection into two 2D operations
- Separates transaxial and axial contributions
- **Very GPU-friendly** (highly regular memory access)
- Used in Michigan Image Reconstruction Toolbox (MIRT)

**Equivalence to Siddon: NO.** Uses trapezoidal footprint approximation.
Different numerical results, though generally comparable quality.

#### Joseph's Method (1982)

- Interpolates along the dominant ray direction
- Simpler than Siddon but introduces interpolation error
- **Equivalence: NO.** Uses linear interpolation between voxels.

### 2.5 Summary and Recommendation

| Algorithm | Speedup vs Siddon | Equivalent? | Risk | Recommendation |
|-----------|------------------|-------------|------|----------------|
| **Branchless DDA** | **1.1-1.3×** | **YES (exact)** | **None** | **DO IT (P1)** |
| Distance-driven | 1.5-2× (new proj) | No (better quality) | Medium | Future option (P3) |
| Separable footprint | 2-3× (new proj) | No (approx.) | Medium | Future option (P3) |
| Joseph's | 1× (similar) | No (interp.) | Low | Skip |

**Key insight: There is NO alternative projector that is both (a) faster than Siddon
and (b) mathematically equivalent.** Branchless DDA is the only free lunch. The 10×
speedup comes from energy loop fusion (SPEED-001), not from changing the ray tracing
algorithm. Branchless DDA provides a modest 1.1-1.3× multiplier that stacks on top.

---

## 3. GPU Kernel Optimization via AK.jl (SPEED-003)

### 3.1 AK.jl Primitive Catalog (Relevant to simulate!())

Full exploration of AcceleratedKernels.jl v0.4.3 source code. Key primitives:

| Primitive | Signature | GPU Behavior | Used in BasisSimulator? |
|-----------|-----------|-------------|------------------------|
| `foreachindex(f, itr; block_size=256)` | One thread per element | No shared memory, no warp ops | **YES — everywhere** |
| `map!(f, dst, src)` | One thread per element | Elementwise transform | No (uses foreachindex) |
| `reduce(op, src; init, dims)` | Block-level shared memory | Two-phase reduction | No |
| `mapreduce(f, op, src; init)` | Fused map+reduce | Block shared memory | No |
| `accumulate!(op, v; init, alg)` | Prefix scan | ScanPrefixes (Metal), DecoupledLookback (CUDA) | No |
| `sort!(v)` | Merge sort | Bitonic + global merge | No |

**Key AK.jl constraints for kernel design:**

1. **`foreachindex` has NO shared memory.** The closure runs with global memory only.
   No `@localmem`, no warp-level primitives, no block cooperation.

2. **No kernel fusion API.** Each `foreachindex` is an independent kernel launch.
   To fuse operations, you must combine them into a single closure.

3. **Closure capture must use `let` blocks** for conditionally-assigned variables
   (avoids Core.Box on GPU). BasisSimulator already does this correctly.

4. **GPU block_size=256 default.** Maps to 256 threads/block (CUDA) or 256
   threads/threadgroup (Metal). Configurable per call.

5. **Metal uses ScanPrefixes for `accumulate!`** (DecoupledLookback has a race on Metal's
   weak memory model). Not relevant to forward projection.

6. **No texture memory, no constant memory, no dynamic parallelism.** Global memory +
   L1/L2 cache only.

### 3.2 Current AK.jl Usage in simulate!()

**Forward projection (`_forward_project_poly!`):**
Per energy bin, 3 kernel launches:
1. `create_μ_volume!` — `AK.foreachindex` over mask (17.5M elements)
2. `siddon_forward_project!` — `AK.foreachindex` over sinogram (57.6M elements)
3. Beer-Lambert accumulation — `AK.foreachindex` over I_transmitted (57.6M elements)

30 energy bins × 3 launches = **90 kernel launches + synchronizations.**

**Physics pipeline (`apply_physics_effects!` / `_apply_physics_no_noise!`):**
Each enabled effect is 1-2 kernel launches. Full pipeline with signal chain (driver.jl:363-470):

| Step | Operation | Kernel launches |
|------|-----------|----------------|
| Physics no-noise pipeline | fill_factor, flat_filter, scatter×2, bowtie, crosstalk, optical_ct, focal_spot, lag | 8-10 (if all enabled) |
| exp(-sino) conversion | Elementwise | 1 |
| Heel effect | Elementwise × angles | 1 |
| DAS model | Gain + noise | 1-2 |
| Air scan creation | fill + bowtie ref + heel + gain | 3-4 |
| Calibration (sino/air) | Elementwise divide | 1 |
| Low signal correction | Elementwise | 1 |
| Log transform | Elementwise | 1 |
| BHC | Polynomial eval | 1 |
| **Total physics/signal chain** | | **~18-22 launches** |

**Total simulate!() kernel launches: 90 + 20 ≈ 110.**
At ~10-25 μs per launch: **1-3 ms total launch overhead.** Negligible vs. 30-60s compute.

### 3.3 GPU Optimization Opportunities (Through AK.jl)

#### O1: Energy Loop Fusion — 90 launches → 1 (SPEED-001, PRIMARY)

Already analyzed in §1. Single `AK.foreachindex` over sinogram, fused DDA + all energies.
**Eliminates 89 kernel launches.** But the speedup is from eliminating redundant COMPUTE,
not from reducing launch overhead.

#### O2: Branchless DDA Inner Loop (SPEED-002, §2.3)

Replace 3-way `if/elseif/else` with predicated multiply-add. Eliminates warp divergence.
**1.1-1.3× on DDA loop → 1.1× on simulate!().**

#### O3: Fix CartesianIndices in GPU Closures (Bug Fix)

**Two locations use CartesianIndices inside AK.foreachindex on GPU:**

1. **polychromatic.jl:1171** — Bowtie spectral accumulation:
   ```julia
   ci = CartesianIndices(I_transmitted)[idx]
   col, row, _ = Tuple(ci)
   ```

2. **scatter.jl:191** — Scatter convolution:
   ```julia
   ci = CartesianIndices(sinogram)[idx]
   col, row, angle = Tuple(ci)
   ```

**Problem:** `CartesianIndices(arr)[idx]` on GPU constructs a CartesianIndex object per
thread. While Julia's GPU compiler can sometimes optimize this, it's not guaranteed — and
the Siddon kernel (siddon.jl:492-497) correctly uses mod/div arithmetic instead.

**Fix:** Replace with manual index decomposition:
```julia
idx_0 = idx - 1
col = (idx_0 % n_cols) + 1
row = ((idx_0 ÷ n_cols) % n_rows) + 1
angle = (idx_0 ÷ (n_cols * n_rows)) + 1
```

**Impact:** Minor (<5% on affected kernels). But scatter.jl is 1-5% of simulate!(), so
this matters slightly. The fused kernel should use mod/div consistently.

#### O4: Block Size Tuning for Fused Kernel

The fused kernel has higher register pressure (~312 bytes/thread for 30 energies).
Block size affects GPU occupancy:

| block_size | Threads/SM (CUDA, 78 regs) | Occupancy | Notes |
|-----------|---------------------------|-----------|-------|
| 256 | 3 blocks × 256 = 768 | 50% | Default |
| 128 | 6 blocks × 128 = 768 | 50% | Same occupancy, more blocks |
| 64 | 12 blocks × 64 = 768 | 50% | More blocks, worse scheduling |
| 512 | 1 block × 512 = 512 | 33% | Too few threads |

For bandwidth-bound kernels, occupancy >30% is sufficient to saturate memory bandwidth.
The default 256 is fine. **No significant speedup from tuning. Skip.**

#### O5: Physics Pipeline Kernel Fusion (Post-Forward-Projection)

**CRITICAL REALIZATION:** After energy loop fusion reduces forward projection from ~30s
to ~3s, the physics pipeline becomes a much larger fraction:

| Component | Before fusion | After 10× fusion |
|-----------|--------------|-------------------|
| Forward projection | 30s (93%) | 3s (60%) |
| Physics pipeline | 2s (6%) | 2s (40%) |
| Other | 0.3s (1%) | 0.3s (6%) |
| **Total** | **32s** | **5.3s** |
| **Speedup** | baseline | **6×** |

To push from 6× to 10×, we need to also optimize the physics pipeline!

**Fusible elementwise operations in signal chain (driver.jl:388-447):**

These consecutive AK.foreachindex calls could be fused into fewer launches:
1. `exp(-sinogram)` (line 390)
2. `sinogram / air_scan` (line 433)
3. `low_signal_correction` (line 440)
4. `-log(sinogram)` (line 444)

Combined: `sinogram[idx] = -log(max(exp(-sinogram[idx]) / air_val, eps))`
4 launches → 1 launch. Saves ~30-60 μs. Negligible.

**The REAL physics pipeline cost is SCATTER** (0.5-1.5s per call, two calls if
scatter + scatter_correction enabled = 1-3s). Scatter does a 63×63 spatial convolution
per pixel — that's 3,969 inner loop iterations with exp() per iteration.

**Scatter optimization opportunities:**
- The 63×63 kernel is evaluated in SPATIAL domain (direct convolution)
- FFT-based convolution would be O(N log N) vs O(N K²), but AK.jl has no FFT
- Separable kernel: Gaussian kernel is separable → 63+63=126 ops instead of 63×63=3,969
- **Separable Gaussian gives 31× fewer operations per pixel!**
- Requires two passes (horizontal then vertical), but each pass is much cheaper
- Mathematically equivalent for Gaussian kernels (exact factorization)

**Scatter separable convolution speedup estimate:**
- Current: 57.6M pixels × 3,969 ops × ~16 FLOPs = 3,660 GFLOP per call
- Separable: 57.6M pixels × 126 ops × ~16 FLOPs = 116 GFLOP per call
- Speedup: **~31× on scatter → scatter drops from 1-3s to 30-100ms**

**Impact on total simulate!() (post-fusion):**
- Before scatter opt: forward=3s, scatter=2s, other=0.3s → 5.3s (6× total)
- After scatter opt: forward=3s, scatter=0.1s, other=0.3s → 3.4s (9.4× total)

**This is the missing piece to reach 10×!**

### 3.4 AK.jl Overhead Assessment

**Question:** Does AK.jl add significant overhead vs raw KernelAbstractions.jl?

AK.foreachindex dispatches to a simple KA kernel:
```julia
@kernel function _forindices_global!(f, itr)
    idx = @index(Global)
    f(idx)
end
```

This is literally one layer of indirection. The KA kernel is identical to what you'd
write by hand. **AK.jl overhead: effectively ZERO.** The abstraction compiles away.

The only overhead is:
- Type inference on the closure (~1ms first call, cached after)
- Backend dispatch (~1μs per call)
- Neither is measurable in the context of seconds of compute

### 3.5 Summary: GPU Optimization Impact

| Optimization | Est. Speedup | Stacks with? | Priority |
|-------------|-------------|-------------|----------|
| Energy loop fusion (§1) | 10-20× on fwd proj | Standalone | **P0** |
| Branchless DDA (§2.3) | 1.1-1.3× on DDA | Fusion | **P1** |
| Separable scatter conv (§3.3 O5) | 31× on scatter | Fusion | **P1** |
| CartesianIndices fix (§3.3 O3) | <5% on scatter | All | P2 |
| Block size tuning (§3.3 O4) | <5% | All | Skip |
| Elementwise fusion (§3.3 O5 top) | <0.1% | All | Skip |

**Stacked projection:**
1. Energy loop fusion: 30s → 3s (10× on forward proj)
2. Branchless DDA: 3s → 2.6s (1.15× on forward proj)
3. Separable scatter: 2s → 0.1s (saves 1.9s)
4. Other physics: 0.3s → 0.3s (unchanged)
5. **Total: 32s → 3.0s = 10.7×** ✓

---

## 4. Physics Pipeline Batching (SPEED-004)

### 4.1 Post-Forward-Projection Signal Chain (EICT)

After forward projection produces a polychromatic sinogram, the EICT signal chain
applies these steps sequentially (driver.jl lines 363-526):

| Step | Operation | Type | Est. Time | Fusible? |
|------|-----------|------|-----------|----------|
| 1 | `_apply_physics_no_noise!()` | Mixed | 1-3s total | See below |
| 2 | `exp(-sinogram)` | Elementwise | <1ms | Yes (Group A) |
| 3 | `apply_heel_effect!()` | Elementwise | <1ms | Yes (Group A) |
| 4 | `apply_das_model!()` | Elementwise | <1ms | Yes (Group A) |
| 5 | Air scan construction | Elementwise | <1ms | Yes (Group A) |
| 6 | `sinogram / air_scan` | Elementwise | <1ms | Yes (Group B) |
| 7 | `low_signal_correction_gpu!()` | Elementwise | <1ms | Yes (Group B) |
| 8 | `-log(sinogram)` | Elementwise | <1ms | Yes (Group B) |
| 9 | `apply_bhc!()` | Elementwise | <1ms | Yes (Group B) |

Inside `_apply_physics_no_noise!()` (polychromatic.jl lines 936-988):

| Sub-step | Type | Est. Time | Fusible? |
|----------|------|-----------|----------|
| fill_factor | Elementwise | <0.1ms | Trivial |
| flat_filter | Elementwise mult | <0.1ms | Trivial |
| **scatter add** | **63×63 conv** | **0.5-1.5s** | **Separable** |
| **scatter correct** | **63×63 conv** | **0.5-1.5s** | **Separable** |
| bowtie_filter | Elementwise mult | <0.1ms | Trivial |
| crosstalk | 3×3 conv | <1ms | Skip |
| optical_crosstalk | 3×3 conv | <1ms | Skip |
| focal_spot_blur | ~11-31 conv | <10ms | Skip |
| lag | Temporal recursive | <1ms | Skip |

### 4.2 Key Insight: Scatter Dominates the Physics Pipeline

Scatter (add + correct) accounts for **>95% of physics pipeline time**. All other
effects combined are <50ms. Optimizing elementwise fusion (Groups A/B) saves ~10
kernel launches but <0.5ms — negligible.

**The only physics optimization that matters is scatter.**

### 4.3 Separable Scatter Convolution (Gaussian Kernel Only)

**Validated in scatter.jl source code.** The kernel construction (lines 115-121):
```
K[dx,dy] = exp(-(dx² + dy²) / (2σ²))
```

This is a 2D isotropic Gaussian: `exp(-(dx²+dy²)/(2σ²)) = exp(-dx²/(2σ²)) × exp(-dy²/(2σ²))`.
**This IS separable** into two 1D Gaussians `Kx` and `Ky`.

**Current approach:** Direct 2D convolution of scatter pre-signal `S = exp(-proj) × proj × C`
with 2D kernel K. Cost: `n_pixels × k² = 57.6M × 3,969 = 228 GFLOP per call`.

**Separable approach:**
1. Compute pre-signal `S[col,row] = exp(-proj) × proj × C` (elementwise, trivial)
2. Horizontal 1D convolution: `H[col,row] = Σ_di S[col+di, row] × Kx[di]` (63 ops/pixel)
3. Vertical 1D convolution: `V[col,row] = Σ_dj H[col, row+dj] × Ky[dj]` (63 ops/pixel)
Cost: `n_pixels × 2k = 57.6M × 126 = 7.3 GFLOP per call`

**Speedup: 228/7.3 = 31.4× per scatter call. Two calls → saves ~1.9s.**

**Correctness:** Exact equivalence for Gaussian kernel (standard textbook result).
NOT exact for exponential kernel (`exp(-√(dx²+dy²)/decay)` is not separable).

**Implementation:** Needs a temporary buffer for the horizontal pass result.
Workspace already has `ws_output` buffer. Add one more buffer of same size.
Two `AK.foreachindex` calls per scatter operation instead of one (but each is 31× cheaper).

### 4.4 PCCT Uses Same Scatter Path

PCCT applies scatter to the combined (binned) sinogram via the same `add_scatter!()` /
`correct_scatter!()` functions. Separable optimization benefits both paths identically.

---

## 5. Reference Implementation Benchmarks

### 5.1 Comparable Implementations

| Simulator | Polychromatic Method | Projection Speed (comparable phantom) | Notes |
|-----------|---------------------|---------------------------------------|-------|
| **CatSim/XCIST** | Sequential energy loop (accurate) | ~30-60s for 512³ vol, 900×64 det, 1000 ang, 30E | Same architecture as current BasisSimulator |
| **CatSim/XCIST** | Basis decomposition (fast) | ~2-5s | 2 basis sinograms, APPROXIMATION |
| **gVXR** | Fused energy loop (OpenGL shaders) | ~3-5s | Fragment shader per pixel, all energies inline |
| **TIGRE** | Monochromatic only | ~0.5-1s per energy (CUDA Siddon) | External energy loop required |

**Key takeaway:** gVXR achieves ~10× faster than CatSim's accurate mode by fusing the
energy loop — exactly the approach proposed in SPEED-001. Our expected 10× aligns with
this reference data.

---

## 6. Precision Analysis

### 6.1 Current Precision

BasisSimulator uses Float32 for all simulation arrays. Float64 is used only in:
- Geometry setup (source/detector positions, pre-kernel)
- Kernel construction on CPU (then cast to Float32 for GPU)
- Some accumulation operations

### 6.2 Precision Requirements by Stage

| Stage | Current | Required | Notes |
|-------|---------|----------|-------|
| DDA geometry (t values) | Float32 | Float32 OK | ±10⁻⁷ on path lengths ~0.1-10 |
| Line integral accumulation | Float32 | Float32 OK | Sum of ~400 terms, each ~0.001-0.1 |
| Beer-Lambert exp(-L) | Float32 | Float32 OK | L ∈ [0, 20], exp range adequate |
| Energy summation (30 terms) | Float32 | Float32 OK | Well-conditioned weighted sum |
| Scatter convolution | Float32 | Float32 OK | Normalized kernel, bounded |
| Noise (Poisson/Gaussian) | Float32 | Float32 OK | Stochastic — noise >> precision |

**Conclusion:** Float32 is adequate everywhere in simulate!(). No precision change needed
for correctness, no precision optimization available for speed (already using Float32).

---

## 7. SYNTHESIS: 10x Speedup Roadmap for simulate!()

### 7.1 Executive Summary

**10.7× speedup is achievable through three optimizations, all preserving mathematical
correctness and all implementable through AcceleratedKernels.jl:**

| # | Optimization | Target | Speedup Factor | Risk | Priority |
|---|-------------|--------|---------------|------|----------|
| 1 | Energy loop fusion | Forward projection (88-95%) | 10× on fwd proj | Med-High | **P0** |
| 2 | Separable Gaussian scatter | Scatter convolution (1-5%) | 31× on scatter | Low | **P1** |
| 3 | Branchless DDA | Siddon inner loop | 1.15× on DDA | None | **P1** |

### 7.2 Stacked Speedup Projection (XCAT f4 Baseline)

```
                        BEFORE      AFTER       CUMULATIVE
Component               Time (s)    Time (s)    Speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Energy Loop Fusion (P0)
Forward projection       30.0  →    3.00        10.0×
Scatter (add+correct)     2.0  →    2.00        (unchanged)
Other physics             0.3  →    0.30        (unchanged)
────────────────────────────────────────────────────────────
SUBTOTAL                 32.3  →    5.30        6.1×

Step 2: Separable Scatter (P1)
Forward projection        3.0  →    3.00        (unchanged)
Scatter (add+correct)     2.0  →    0.065       31×
Other physics             0.3  →    0.30        (unchanged)
────────────────────────────────────────────────────────────
SUBTOTAL                  5.3  →    3.37        9.6×

Step 3: Branchless DDA (P1)
Forward projection        3.0  →    2.61        1.15×
Scatter                  0.065 →    0.065       (unchanged)
Other physics             0.3  →    0.30        (unchanged)
────────────────────────────────────────────────────────────
SUBTOTAL                 3.37  →    2.97        10.9×

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: 32.3s → 2.97s = 10.9× speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Conservative estimate (tiled fusion, 8 bins/pass):**
- Tiled fusion: 30s → 4.5s (6.7× on fwd proj, 4 DDA passes instead of 30)
- Separable scatter: 2s → 0.065s
- Branchless DDA: 4.5s → 3.91s (1.15×)
- **Total: 32.3s → 4.28s = 7.5×**

Even the conservative path hits 7.5× — and if full NTuple{30} fusion works on Metal,
we comfortably exceed 10×.

### 7.3 Implementation Stories (Ordered by Priority)

---

#### STORY 1: Energy Loop Fusion (P0) — THE BIG WIN

**Expected impact:** 30s → 3s forward projection (10× on 88-95% of simulate!())
**Implementation effort:** 2-3 days
**Risk:** Medium (NTuple register pressure on Metal — tiled fallback available)
**Correctness:** Mathematically identical (proven in §1.1)

**What to build:**

1. **`siddon_fused_poly_project!(sinogram, mask, geom, μ_table, weights, η; ...)`**
   - Single `AK.foreachindex` over sinogram (n_cols × n_rows × n_angles)
   - Inner loop: DDA traversal through UInt8 mask (ONCE per ray)
   - At each voxel: `accum[e] += μ_table[mat, e] × path` for all N_E energies
   - After traversal: Beer-Lambert `I = Σ w_e × η_e × exp(-accum[e])`
   - Uses NTuple{N_E,Float32} for register-resident accumulators

2. **Tiled fallback: `siddon_tiled_poly_project!(...; tile_size=8)`**
   - Same algorithm but processes `tile_size` energies per DDA pass
   - `ceil(N_E / tile_size)` passes, each reading mask once
   - 4 passes (30 energies ÷ 8) vs 30 → 7.5× bandwidth reduction
   - Safe on all platforms (8 accumulators = 32 bytes/thread)

3. **Update `_forward_project_poly!()`** to call fused kernel instead of energy loop
   - Prepare `μ_table_gpu` from materials: `[n_materials × n_energies]`
   - No longer allocates `μ_volume` or `sinogram_mono`
   - Single kernel launch replaces 90+

4. **Update PCCT path** (`pcct_forward_project` in photon_counting.jl)
   - Same fused DDA, but after traversal distribute via DRM:
     `bins[b] += Σ_e I0 × w_e × η_e × R[e,b] × exp(-accum[e])`

**AK.jl API used:** `AK.foreachindex(sinogram) do idx; ...; end`

**Key design decisions:**
- N_E as `Val{N}` parameter for compile-time unrolling
- μ_table in GPU global memory (1.8 KB, stays in L1 after first access)
- mask reads UInt8 (4× smaller than Float32 volume → better cache behavior)
- Bowtie spectral transmission folded into Beer-Lambert sum (per-pixel per-energy)

**Acceptance test:** Run XCAT f4 simulation before/after. Compare sinograms
element-by-element. Max absolute difference < 1e-5 (Float32 ULP tolerance).
Measure @elapsed to verify ≥8× speedup on forward projection.

---

#### STORY 2: Separable Gaussian Scatter Convolution (P1) — THE MISSING PIECE

**Expected impact:** 2s → 0.065s scatter (31× on scatter)
**Implementation effort:** 0.5-1 day
**Risk:** Low (textbook algorithm, exact for Gaussian)
**Correctness:** Exact mathematical equivalence for Gaussian kernel (proven by
separability of 2D isotropic Gaussian)

**What to build:**

1. **`_compute_scatter_presignal!(S, sinogram, C)`**
   - Elementwise: `S[idx] = exp(-clamp(sino[idx], T(20))) × sino[idx] × C`
   - Single `AK.foreachindex`

2. **`_convolve_separable_1d_h!(output, input, kernel_1d, half_k, n_cols, n_rows)`**
   - Horizontal 1D convolution: `output[c,r,a] = Σ_di input[clamp(c+di), r, a] × kernel[di]`
   - Single `AK.foreachindex` over output

3. **`_convolve_separable_1d_v!(output, input, kernel_1d, half_k, n_cols, n_rows)`**
   - Vertical 1D convolution: `output[c,r,a] = Σ_dj input[c, clamp(r+dj), a] × kernel[dj]`
   - Single `AK.foreachindex` over output

4. **`add_scatter_separable!(sinogram, model; ws_presignal, ws_temp, ws_kernel_1d)`**
   - Decompose 2D Gaussian kernel into 1D: `Kx[di] = exp(-di²/(2σ²))`, normalized
   - Call: presignal → horizontal conv → vertical conv → combine with sinogram
   - 3 kernel launches instead of 1, but 31× fewer FLOPs

5. **Guard for exponential kernel:**
   - If `model.kernel_type == :exponential`, fall back to current 2D convolution
   - Gaussian (default) gets the fast path

**Workspace changes:**
- Add `ws_presignal` buffer (same size as sinogram)
- Add `ws_scatter_temp` buffer (same size as sinogram)
- Add `ws_scatter_kernel_1d` (63 elements, 1D)

**Acceptance test:** Run scatter with known input. Compare 2D convolution output vs
separable output. Max absolute difference < 1e-6 (Float32 precision in convolution).

---

#### STORY 3: Branchless DDA (P1) — FREE PERFORMANCE

**Expected impact:** 1.15× on DDA inner loop
**Implementation effort:** 1-2 hours
**Risk:** None (exact mathematical equivalence)
**Correctness:** Same voxels visited in same order with same path lengths (proven in §2.3)

**What to build:**

Replace the 3-way `if/elseif/else` in `siddon_trace_ray` (siddon.jl:258-291) with
predicated arithmetic:

```julia
# BEFORE (branching):
if t_next_x <= t_next_y && t_next_x <= t_next_z
    ix += step_x; t_next_x += dt_x
elseif t_next_y <= t_next_z
    iy += step_y; t_next_y += dt_y
else
    iz += step_z; t_next_z += dt_z
end

# AFTER (branchless):
mask_x = Int32(t_next_x <= t_next_y) * Int32(t_next_x <= t_next_z)
mask_y = Int32(1 - mask_x) * Int32(t_next_y <= t_next_z)
mask_z = Int32(1) - mask_x - mask_y

ix += mask_x * step_x
iy += mask_y * step_y
iz += mask_z * step_z
t_next_x += T(mask_x) * dt_x
t_next_y += T(mask_y) * dt_y
t_next_z += T(mask_z) * dt_z
```

**Note:** This change applies to BOTH the existing `siddon_trace_ray` AND the new
fused kernel (Story 1). Implement in both places.

**Acceptance test:** Before/after sinogram comparison. Bit-identical expected
(same operations in same order, just different instruction encoding).

---

#### STORY 4: CartesianIndices Fix (P2) — MINOR CLEANUP

**Expected impact:** <5% on scatter/accumulation kernels
**Implementation effort:** 15 minutes
**Risk:** None

Replace `CartesianIndices(arr)[idx]` with mod/div arithmetic in:
- `scatter.jl:191` (scatter convolution)
- `polychromatic.jl:1171` (bowtie accumulation)

Both should use the same pattern as `siddon.jl:492-497`:
```julia
idx_0 = Int32(idx - 1)
col = (idx_0 % n_cols) + Int32(1)
idx_0 = idx_0 ÷ n_cols
row = (idx_0 % n_rows) + Int32(1)
angle = (idx_0 ÷ n_rows) + Int32(1)
```

**Note:** After Story 2 (separable scatter), the scatter kernel is rewritten anyway,
so this fix is primarily for the bowtie accumulation and any remaining 2D scatter
fallback (exponential kernel path).

---

### 7.4 Implementation Order and Dependencies

```
Week 1:
├── Story 1a: siddon_fused_poly_project! (tiled version first)
│   ├── Build kernel with tile_size=8
│   ├── Validate correctness vs current output
│   └── Measure speedup (target: ≥6× on forward projection)
│
├── Story 3: Branchless DDA (integrate into fused kernel)
│   └── ~2 hours, no risk
│
└── Milestone: 6-7× total speedup, correctness validated

Week 2:
├── Story 1b: Test full NTuple{30} fusion on Metal + CUDA
│   ├── If works: 10× on forward projection
│   └── If spills: keep tiled version (6-7×)
│
├── Story 2: Separable scatter convolution
│   ├── Implement for Gaussian kernel (default)
│   ├── Keep 2D fallback for exponential
│   └── Validate scatter output within Float32 tolerance
│
├── Story 4: CartesianIndices fix (trivial)
│
└── Milestone: 10-11× total speedup

Week 3 (if needed):
├── Runtime profiling with @elapsed to validate all estimates
├── Tune: block_size, tile_size if needed
└── Final correctness validation against clinical verification notebooks
```

### 7.5 Correctness Validation Plan

**Gate 1 (after each story):** Compare sinogram output before/after optimization.
```julia
sino_before = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
# ... apply optimization ...
sino_after  = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
@assert maximum(abs.(sino_before .- sino_after)) < 1e-5  # Float32 tolerance
```

**Gate 2 (after all stories):** Run full verification pipeline.
- Run `verification/notebooks/06_ge_apex_elite_clinical.jl`
- Run `verification/notebooks/07_siemens_naeotom_alpha_clinical.jl`
- Compare HU measurements, noise levels, contrast against clinical data
- Results must match within existing acceptance criteria

**Gate 3 (performance):** Measure with @elapsed on XCAT f4 phantom.
- Forward projection: ≥8× speedup (tiled) or ≥10× (full fusion)
- Scatter: ≥20× speedup
- Total simulate!(): ≥8× speedup (conservative) or ≥10× (full fusion + scatter)

### 7.6 Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| NTuple{30} register spill on Metal | Forward proj only 6-7× instead of 10× | Medium | Tiled fallback (8 bins/pass) gives 6-7× safely |
| AK.jl closure size limit | Fused kernel won't compile | Low | Closure captures ~500 bytes — well within limits |
| Separable scatter numerical drift | Scatter output differs by >1e-5 | Very Low | Standard algorithm, but validate carefully |
| Exponential kernel users lose scatter speedup | No scatter optimization for non-default mode | Low | Document: Gaussian is recommended for performance |
| PCCT native-res fusion complexity | More rays × more energies × DRM = complex kernel | Medium | Implement EICT first, port to PCCT second |

### 7.7 What We're NOT Doing (and Why)

| Optimization | Why Skip |
|-------------|----------|
| Distance-driven projector | Different output — not a drop-in replacement |
| Separable footprint | Different output — not a drop-in replacement |
| FFT-based scatter | AK.jl has no FFT; separable Gaussian is faster anyway |
| Physics pipeline elementwise fusion | Saves <0.5ms — not worth the code complexity |
| Block size tuning | <5% impact — default 256 is fine |
| Mixed precision (Float64→32) | Already using Float32 everywhere |
| SPEED-005 deep survey | gVXR confirms 10× from fusion; diminishing returns |
| SPEED-006 precision analysis | Float32 is adequate; no precision change needed |

---
