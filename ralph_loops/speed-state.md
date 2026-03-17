# 10x Speed Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** SPEED-001 DISCOVERY (energy loop fusion)
**Current topic:** SPEED-001 (Fuse the Sequential Energy Loop)

## What was done

**Iteration 1 (SPEED-000 DISCOVERY — static profiling audit):**
- Traced complete simulate!() call graph through driver.jl, polychromatic.jl, siddon.jl, physics_pipeline.jl
- Analyzed all 14 detector physics effects for computational complexity
- Produced quantitative time breakdown with concrete parameters (XCAT f4 phantom)

**HEADLINE FINDING:** Forward projection (energy loop × Siddon ray tracing) is 90-97%
of simulate!() time. The 30× redundant DDA traversal across energy bins is the root cause.
Energy loop fusion is THE path to 10×.

## What to do next

**SPEED-001 DISCOVERY — Energy Loop Fusion Algorithm**

The profiling audit identified energy loop fusion as the single highest-impact optimization
(est. 10-20× speedup, targeting 90-97% of runtime). Deep-dive this topic:

1. **Mathematical proof of equivalence:**
   - Current: `I = Σ_e w_e × exp(-siddon(μ_volume_e))` (30 separate traces)
   - Fused: `I = Σ_e w_e × exp(-Σ_v μ_table[mat_v, e] × path_v)` (1 trace, inline lookup)
   - Prove these are identical (they should be — Beer-Lambert is linear in μ×path)

2. **Register pressure analysis:**
   - 30 Float32 accumulators = 120 bytes per thread
   - Plus DDA state (~50 bytes): ix, iy, iz, steps, t_next, dt, etc.
   - Total ~200 bytes → verify this fits Metal/CUDA register files
   - Estimate occupancy impact

3. **AK.jl kernel design:**
   - Write pseudocode for `siddon_fused_poly!()` using AK.foreachindex
   - Input: UInt8 mask + μ_table (n_materials × n_energies) + weights
   - Output: I_transmitted (Float32 sinogram)
   - Handle: bowtie spectral transmission, detector efficiency η(E)

4. **Memory access pattern comparison:**
   - Current: 30 × read Float32 volume (70 MB) = 2.76 TB total
   - Fused: 1 × read UInt8 mask (17.5 MB) + μ_table in L1 (1.8 KB)
   - Quantify expected bandwidth reduction

5. **Reference implementations:**
   - Does CatSim/XCIST fuse the energy loop? How?
   - Does any other GPU CT simulator do fused polychromatic projection?

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| SPEED-000 Profiling Audit | **done** | open | open |
| SPEED-001 Energy Loop Fusion | **open → DO THIS** | open | open |
| SPEED-002 Ray Tracing Algorithm | open | open | open |
| SPEED-003 GPU Kernel Optimization (AK.jl) | open | open | open |
| SPEED-004 Physics Pipeline Batching | open | open | open |
| SPEED-005 Reference Implementations | open | open | open |
| SPEED-006 Precision Analysis | open | open | open |
| SPEED-007 SYNTHESIS | open | open | open |

## Completed iterations

1. **Iteration 1** — SPEED-000 DISCOVERY: Static profiling audit. Forward projection = 90-97% of time. Energy loop fusion = path to 10×.
