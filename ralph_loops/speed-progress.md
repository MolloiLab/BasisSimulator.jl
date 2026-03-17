# BasisSimulator.jl 10x Speed — Discovery Progress Log

Each iteration logs what was researched, what was found, and what gaps remain.

---

## Iteration 1 — SPEED-000 DISCOVERY: Static Profiling Audit

**Date:** 2026-03-17
**Phase:** Discovery
**Topic:** SPEED-000 (Profile simulate!() — where does the time go?)

### What was done

Traced the complete simulate!() call graph by reading source code:
- `src/api/driver.jl` — both EICT and PCCT simulate!() methods
- `src/projection/polychromatic.jl` — `_forward_project_poly!()` energy loop (lines 1100-1195)
- `src/projection/siddon.jl` — `siddon_trace_ray()` + `siddon_forward_project!()` (full file)
- `src/detector/physics_pipeline.jl` — `apply_physics_effects!()` 14-effect chain
- All individual detector physics implementations (scatter, crosstalk, noise, lag, etc.)
- `src/api/workspace.jl` — PCCTWorkspace and EICTWorkspace structures

Estimated computational complexity for each component using concrete parameters
(XCAT f4: 400×350×125 volume, 900×64 detector, 1000 angles, 30 energy bins).

### Key findings

#### 1. THE BOTTLENECK: Energy-sequential forward projection is 90-97% of simulate!()

`_forward_project_poly!()` loops sequentially over ~30 energy bins. Each iteration:
1. `create_μ_volume!()` — elementwise GPU lookup (negligible)
2. `siddon_forward_project!()` — **FULL Siddon ray trace** (dominant)
3. Beer-Lambert accumulation — elementwise exp/add (negligible)

**Siddon ray tracing alone consumes 85-95% of total simulate!() time.**

#### 2. ROOT CAUSE: 30× redundant geometric computation

The key insight: **the DDA traversal path is identical across all energy bins.**
Same source positions, same detector positions, same phantom geometry. Only the
attenuation coefficients (μ values) change per energy. Yet the current code:
- Recomputes the full DDA path 30 times per ray
- Reads the volume 30 times (different μ values each time) → 2.76 TB memory traffic
- Launches 90+ GPU kernels with synchronization points
- Overwrites 17.5M-voxel μ_volume 30 times

The phantom's **material mask (UInt8, 17.5 MB)** is the invariant. The
**μ_table (15 materials × 30 energies, 1.8 KB)** fits in L1 cache.

#### 3. Physics pipeline is NOT the bottleneck (3-8% of time)

Even scatter (the most expensive effect at ~3136 ops/element for a 50×50 kernel)
only accounts for 1-3% of total time. All other effects (crosstalk, noise, lag,
BHC, fill factor, etc.) are elementwise or small-kernel operations — collectively <2%.

#### 4. Energy loop fusion is THE path to 10×

A fused kernel that traces each ray ONCE through the material mask, looking up μ
for all 30 energies at each voxel intersection, would:
- Eliminate 30× redundant DDA traversal → 30× less geometry compute
- Read mask (UInt8, 17.5MB) once instead of volume (Float32, 70MB) 30× → 120× less bandwidth
- 1 kernel launch instead of 90+ → eliminate sync overhead
- μ_table (1.8 KB) stays in L1 cache → effectively free lookups

**Estimated speedup from fusion alone: 10-20× on forward projection → 10-19× on simulate!().**

#### 5. Siddon is memory-bandwidth bound, not compute-bound

Each ray accesses ~400 scattered voxels from a 70MB volume. This is random access
(rays fan out from source), so effective bandwidth is ~50-100 GB/s on Apple M1 Max
(vs 400 GB/s peak for sequential access). The fused kernel improves this by reading
a 4× smaller array (UInt8 mask vs Float32 volume).

### What gaps remain

1. **No runtime profiling.** These are static estimates. Need @elapsed measurements
   to validate the 90-97% forward projection claim and check if AK.jl adds overhead.
2. **Register pressure unknown.** Fused kernel needs 30 Float32 accumulators (120 bytes)
   per thread — need to verify this doesn't kill GPU occupancy on Metal/CUDA.
3. **AK.jl fusion feasibility.** Need to confirm that AK.foreachindex can handle the
   fused kernel (larger closure, more registers) without hitting backend limits.
4. **PCCT path not analyzed in detail.** PCCT has native-res binning that adds another
   layer — need to trace pcct_forward_project() for differences.

### What should happen next

1. **SPEED-000 DISCOVERY iteration 2:** Runtime profiling with @elapsed to validate
   static estimates. Even a simple test (small phantom, few angles) would confirm ratios.
2. **SPEED-001 DISCOVERY:** Deep dive on energy loop fusion algorithm — the #1 optimization.
   Math proof of Beer-Lambert equivalence, register pressure analysis, AK.jl pseudocode.
3. Phase rotation: 2 more discoveries → 1 critique.

---

## Iteration 2 — SPEED-001 DISCOVERY: Energy Loop Fusion Deep Dive

**Date:** 2026-03-17
**Phase:** Discovery
**Topic:** SPEED-001 (Fuse the Sequential Energy Loop)

### What was done

Deep analysis of energy loop fusion covering all 5 research questions:

1. **Read actual source code:**
   - `_forward_project_poly!()` (polychromatic.jl:1100-1195) — EICT energy loop
   - `pcct_forward_project()` (photon_counting.jl:1340-1460) — PCCT energy loop
   - `siddon_trace_ray()` (siddon.jl:150-293) — DDA traversal kernel
   - `siddon_forward_project!()` (siddon.jl:414-538) — AK.foreachindex dispatch
   - `create_μ_volume!()` (polychromatic.jl:214-262) — GPU table lookup

2. **Full AcceleratedKernels.jl v0.4.3 API exploration:**
   - `foreachindex(f, itr; block_size=256)` — primary primitive, closure must have known types
   - NTuple approach works for fixed-count accumulators in GPU kernels
   - No explicit register limit in AK; backend compiler manages spill
   - Metal auto-selects ScanPrefixes for accumulate (DecoupledLookback has 0.38% race)

3. **Analyzed both EICT and PCCT paths for fusion compatibility**

### Key findings

1. **Mathematical equivalence PROVEN (bit-identical).** Same voxels, same DDA order, same μ values via table lookup. No approximation.

2. **Register pressure: 78 regs/thread (manageable).** 30 Float32 accumulators (120B) + DDA state (192B) = 312B. CUDA budget: 255 regs (31% used). Metal: auto-spill handles it. Fallback: tile 8 energies × 4 passes.

3. **AK.jl kernel design feasible.** Single `foreachindex` closure. NTuple{N_E,T} for accumulators. μ_table (1.8 KB) stays in L1. Full pseudocode in spec §1.3.

4. **Memory bandwidth reduction: 24-43×.** Current 2,790 GB → fused ~65-115 GB. UInt8 mask (17.5 MB, fits L2) replaces 30× Float32 volume (70 MB) reads.

5. **Reference implementations confirm.** gVXR fuses energy loop. CatSim uses sequential (accurate) or basis decomposition (approximate). TIGRE is mono only.

6. **PCCT equally fusible.** Same DDA + accumulate pattern; output via DRM after traversal.

### Speedup estimate

- **Forward projection: 15-30× speedup** (bandwidth-bound → 24-43× less traffic)
- **simulate!() total: 10-20× speedup** (forward proj = 90-97% of total)

### What gaps remain

1. Variable N_E support (Val{N_E} dispatch vs fixed-max)
2. GPU compiler verification (NTuple register allocation)
3. PCCT fused variant concrete pseudocode
4. AK.jl closure size limits on Metal
5. No runtime profiling yet (static analysis only)

### What should happen next

Phase rotation: 2 discoveries done → **CRITIQUE time.**
Next iteration: **SPEED-000 + SPEED-001 CRITIQUE** — challenge all claims, find show-stoppers.

---
