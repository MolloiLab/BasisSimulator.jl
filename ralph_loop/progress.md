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
