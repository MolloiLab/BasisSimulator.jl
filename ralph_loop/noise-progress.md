# Noise Diagnosis Progress Log

> Each agent iteration appends findings here. This is the persistent memory across iterations.

---

## Status Summary

| Story | Status | Verdict | Factor |
|-------|--------|---------|--------|
| NOISE-001 | done | NO | ~1.0x |
| NOISE-002 | open | — | — |
| NOISE-003 | open | — | — |
| NOISE-004 | open | — | — |
| NOISE-005 | open | — | — |
| NOISE-006 | open | — | — |
| NOISE-007 | open | — | — |
| NOISE-008 | open | — | — |
| NOISE-009 | open | — | — |
| NOISE-010 | open | — | — |
| NOISE-011 | open | — | — |
| NOISE-012 | open | — | — |
| NOISE-013 | open | — | — |
| NOISE-014 | open | — | — |

---

## Iteration Log

<!-- Agent iterations append below this line -->

### 2026-02-13: NOISE-001 [PASS — I0 computation is correct]

**Investigated:** Full I0 computation path for notebook 01.

**Path traced:**
1. Notebook 01 uses `SimOptions(fidelity=:high)` (line 575-580) → `use_noise=true`
2. Calls `BS.create_eict_workspace()` then `BS.simulate!()` (lines 636-637, 673-674)
3. `simulate!` for `EICTWorkspace` (driver.jl:684) applies noise at driver.jl:819-834
4. Noise uses `compute_detector_I0(geom, protocol)` at detector_noise.jl:724-743
5. **NOT** `sim_detect()` (used in `_simulate_axial_single` but not workspace path)
6. **NOT** `mA_to_I0()` (which includes η_det=0.85)
7. **NOT** hardcoded I0=1e6 from `full_physics_config()` (physics noise=nothing in driver)

**Exact computation (detector_noise.jl:724-743):**
```
I0 = flux_density × mA × (rotation_time/views) × pixel_area_mm² × (1000/SDD)²
```

**Notebook 01 parameters:**
- Scanner: SID=540mm, SDD=950mm → geom.SAD=54cm, geom.SDD=95cm
- Scanner: detector_col_size=0.569mm, detector_row_size=0.569mm (at isocenter)
- Protocol: mA=200, rotation_time=1.0s, views=984, flux_density=2.0e6 (default)

**Step-by-step I0 computation:**
- SDD_mm = 95.0 × 10 = 950.0
- SAD_mm = 54.0 × 10 = 540.0
- magnification = 950/540 = 1.7593
- pixel_col_det_mm = 0.569 × 1.7593 = 1.0010 mm
- pixel_row_det_mm = 0.569 × 1.7593 = 1.0010 mm
- pixel_area_mm² = 1.0010 × 1.0010 = 1.0020 mm²
- time_per_view = 1.0/984 = 0.001016 s
- dist_factor = (1000/950)² = 1.1080
- **I0 = 2.0e6 × 200 × 0.001016 × 1.0020 × 1.1080 = ~451,307 photons/pixel/view**

**CatSim comparison:**
- CatSim detector face pitch = 1.0mm → pixel_area = 1.0 mm²
- CatSim I0 ≈ 2.0e6 × 200 × (1/984) × 1.0 × (1000/950)² ≈ 449,630
- **Difference: <1%** (from slight pixel area difference: 1.002 vs 1.0 mm²)

**Missing η_det discrepancy:**
- `compute_detector_I0()` does NOT include η_det (quantum efficiency)
- `mA_to_I0()` DOES include η_det=0.85
- Effect: I0 is ~15% HIGHER than it should be → noise is ~7% LOWER
- **Wrong direction!** Missing η_det makes us LESS noisy, not more.

**Key discovery — workspace path vs allocating path:**
- `_simulate_axial_single()` (driver.jl:258) uses `sim_detect()` which calls `compute_detector_I0()`
- `simulate!()` for EICTWorkspace (driver.jl:684) inlines noise application with same `compute_detector_I0()`
- Both paths use identical I0 computation — no double-noise possible.

**Verdict: NO — I0 computation does NOT explain 2x noise.**
- I0 values match CatSim within ~1%
- Missing η_det makes noise ~7% too LOW (wrong direction)
- Impact factor: ~1.0x (no contribution to 2x noise)
- **Next: Look at FDK normalization (NOISE-004, NOISE-008, NOISE-009) or spectrum double-filtering (NOISE-002)**
