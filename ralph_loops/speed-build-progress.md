# BasisSimulator.jl 10x Speed — Build Progress Log

Branch: `speed/fused-projection`

---

## SPEED-BUILD-001: Energy Loop Fusion ✓

**Status:** Done
**Commit:** `c14a17a` on `speed/fused-projection`
**Date:** 2026-03-17

### What was done
- Added `siddon_fused_poly_project!()` to `src/projection/siddon.jl` — single AK.foreachindex kernel that traces each ray ONCE through the UInt16 material mask and accumulates line integrals for ALL energy bins via μ_table lookup
- Added three `@generated` helper functions for NTuple manipulation: `_fused_accum_energies`, `_fused_beer_lambert`, `_fused_beer_lambert_bt`
- Modified `_forward_project_poly!()` with `fused=true` kwarg (default) — gates between new fused path and legacy unfused path
- Added `wη_gpu` field to `EICTWorkspace` (pre-computed weights_norm .* η on GPU)
- Wired up in `driver.jl` simulate! to pass `ws_wη_gpu`

### Key design decision: @generated vs ntuple
The spec recommended `ntuple(Val(N_E)) do e ... end` for energy accumulators. This works on GPU (compiles to register operations) but is **939x slower on CPU** due to closure overhead inside a while loop. Solution: `@generated` functions that expand to explicit `tuple(old[1]+..., old[2]+..., ...)` at compile time. Both CPU and GPU compilers optimize this well.

### Performance (CPU, post-warmup)
| Test case | Unfused | Fused | Speedup | Max diff |
|-----------|---------|-------|---------|----------|
| 128³ × 3mat × 20E × 72ang | 1.04 s | 0.16 s | **6.65x** | 0.0 |
| 200³ × 4mat × 30E × 180ang | 9.93 s | 1.58 s | **6.28x** | 0.0 |

### Correctness
- Bit-identical output (max abs diff = 0.0) for all test cases
- All 1618 existing tests pass (35 failures + 13 errors are pre-existing recon/Enzyme issues)

### What's NOT done
- PCCT fused variant (separate `pcct_forward_project` path — needs its own fused kernel with DRM distribution)
- Tiled fallback (not needed — NTuple{30} with @generated works fine)
- GPU benchmarks (no GPU in test environment; CPU results validate correctness and algorithmic speedup)
