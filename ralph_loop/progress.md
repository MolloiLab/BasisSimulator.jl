# BasisSimulator — Two-Material BHC Implementation Progress

## Project Reset (2026-02-26)

Previous project (Verification Suite Migration) completed. Ralph loop repurposed for
two-material beam hardening correction implementation.

### Stories

| Story | Description | Status |
|-------|-------------|--------|
| BHC-CORE | Add two-material BHC functions to core library | open |
| BHC-SCRIPT | Create visual test script with comparison PNG | open |
| BHC-RUN | Run test script and verify PNG output | open |

---

### BHC-CORE — Iteration 1 (2026-02-26)
- **Done:** Added all four two-material BHC components to `src/correction/beam_hardening_correction.jl`:
  - `bone_fraction_smooth()` — C1 smoothstep tissue fraction decomposition
  - `TwoMaterialBHC` struct — holds water BHC + spectral bone data
  - `calibrate_bhc_two_material()` — calibrates water+bone model
  - `apply_bhc_two_material()` — full 2-material correction pipeline
  - Added exports and imports (`XA`, `Unitful`)
  - Existing water-only BHC code untouched
  - Syntax verified via `Meta.parseall`
- **Files:** `src/correction/beam_hardening_correction.jl` (modified)
- **Next:** Commit and move to BHC-SCRIPT

### BHC-SCRIPT — Iteration 1 (2026-02-26)
- **Done:** Rewrote `ralph_loop/scripts/bhc_two_material_test.jl` to use core library functions:
  - Replaced inline algorithm with `calibrate_bhc_two_material()` and `apply_bhc_two_material()`
  - Removed inline `bone_fraction_smooth`, inline material lookups, inline correction loop
  - Kept same visualization layout (2×4 grid, colorbars, annotation)
  - Kept same quantitative comparison printout
  - Script uses `mkpath` and `save(path, fig, px_per_unit=2)`
- **Files:** `ralph_loop/scripts/bhc_two_material_test.jl` (rewritten)
- **Next:** Commit and move to BHC-RUN

### BHC-RUN — Iteration 1 (2026-02-26)
- **Done:** Ran test script successfully. Results:
  - No BHC center water: +43.5 HU (beam hardening present)
  - Water-only BHC center water: +3.0 HU (cupping corrected)
  - Water+Bone BHC center water: +0.6 HU (both effects corrected)
  - Center ROI σ: 13.7 → 13.1 → 9.6 HU (bone BHC reduces artifact variance)
  - Bone correction map shows non-zero correction near Ca insert ray paths
  - PNG: 1.0 MB at `ralph_loop/results/bhc_two_material.png`
- **Files:** `ralph_loop/results/bhc_two_material.png` (generated)
- **Status:** ALL STORIES COMPLETE
