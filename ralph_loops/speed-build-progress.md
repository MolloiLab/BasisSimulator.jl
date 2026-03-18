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
