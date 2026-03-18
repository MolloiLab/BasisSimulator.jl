# BasisSimulator.jl 10x Speed — Build Progress Log (v2 — GPU-First)

v1 build progress archived in git history. v1 implemented 4 stories with CPU-only
benchmarks that produced ZERO GPU speedup. v2 stories will only be created by the
discovery loop after GPU validation, and every build completion requires GPU benchmarks.

Branch: `speed/fused-projection`

---

## SPEED-BUILD-V2-001: Disable fused path as default (DONE)

**Date:** 2026-03-18

**Change:** `fused::Bool=true` → `fused::Bool=false` in `_forward_project_poly!` (polychromatic.jl:1135)

**Why:** The fused kernel with 234 energy bins (NTuple{234,Float32} = 936 bytes of accumulators) causes massive register spilling on Metal GPU, making it 3.65× slower than the unfused sequential energy loop.

**GPU Benchmark (Metal M4 Max, AGXG16G):**
- Fused baseline (discovery): 19,182 ms
- Unfused (V2-001): **5,259 ms** (3.65× speedup)
- Target: < 6,500 ms ✓ PASS

**Fused kernel code preserved** for tiled fusion (V2-002).

---

## SPEED-BUILD-V2-002: Tiled energy fusion K=16 (DONE)

**Date:** 2026-03-18

**Approach:** Use proven-fast `siddon_fused_poly_project!` with `Val(16)` for each tile.
15 tiles × 16 energies = 240 bins (234 real + 6 zero-padded). Each tile:
1. Copy subset μ_table/wη/bowtie (1.7KB + 64B + 512KB) to per-tile buffers
2. Call `siddon_fused_poly_project!` with Val(16) — ~95ms per tile (single DDA pass)
3. Accumulate: `I_transmitted += exp(-sinogram)` (undo -log, add partial sum)

After all tiles: `sinogram = -log(I_transmitted)`.

**Key learnings:**
- Custom `_tiled_*` @generated helpers with runtime `ts` offset were 71% slower (162ms vs 95ms/tile) due to GPU compiler not optimizing runtime index arithmetic as well as compile-time constant indices. Reverting to original `_fused_*` helpers with subset copies solved this.
- `siddon_multitile_poly_project!` (single kernel, DDA re-traversed 15× per ray) gave no improvement — re-traversal is the dominant cost.
- Workspace buffers (μ_table_gpu, wη_gpu, bowtie_spectral) padded to multiples of 16 in `create_eict_workspace`.
- `let` blocks needed for all GPU closures referencing `I_transmitted` (assigned in multiple branches → Core.Box).

**GPU Benchmark (Metal M4 Max, AGXG16G):**
- Fused baseline: 19,182 ms
- Unfused (V2-001): 5,259 ms
- Tiled (V2-002): **1,496 ms** (12.83× from fused, 3.52× from unfused)
- 10× target (1,918 ms): ✓ **EXCEEDED** — 12.83×

**Correctness:**
- Tiled vs unfused max abs diff: 3.8e-6 (Float32 tolerance)

**Files modified:**
- `src/projection/siddon.jl` — Added `siddon_tiled_poly_project!`, `siddon_multitile_poly_project!`, `_tiled_*` helpers (kept for reference; production uses original `siddon_fused_poly_project!` with subsets)
- `src/projection/polychromatic.jl` — Tiled path in `_forward_project_poly!`, fixed `let` blocks for GPU closure safety
- `src/api/workspace.jl` — Padded μ_table_gpu, wη_gpu, bowtie_spectral to multiples of 16
