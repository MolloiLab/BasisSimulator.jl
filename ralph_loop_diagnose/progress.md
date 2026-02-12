# BasisSimulator — Ghost Artifact Diagnosis Progress

## Problem Statement (2026-02-12)

Ghost-like aliasing artifacts visible around ribs and high-contrast structures in XCAT CT reconstructions. The phantom ground truth is CLEAN — no aliasing in the labeled mask. The artifact is introduced by the forward projection → physics → reconstruction pipeline.

**Critical clue from user:** With HIGHER detector column count and SMALLER pixel size, artifacts get WORSE.

**Current settings when artifact is visible:**
- EICT 120 kVp, 1600 views, mA=700, noise OFF
- HannFilter, 512×512×64 recon at 35cm FOV
- All physics enabled except scatter, noise, DAS
- Phantom: XCAT downsampled ×2 (800×700×250, 0.6mm voxels)

---

### SETUP-DIAG — Iteration 1 (2026-02-12)
- **Task:** Created `ralph_loop_diagnose/diag.jl` diagnostic script
- **Result:** Script created with full XCAT loading, water calibration, simulate+reconstruct, PNG output
- **Features:**
  - Configurable physics toggles at top of file (PHYSICS_ENABLED master switch + individual USE_* flags)
  - Configurable detector geometry (col_size, row_size, cols, rows)
  - Water phantom calibration for HU conversion
  - Saves reconstruction PNG (HU window -300 to 400) and sinogram PNG
  - Mirrors NB05 settings exactly
- **Next:** Run baseline test to verify script works

### CODE-AUDIT — Iteration 1 (2026-02-12)
- **Task:** Line-by-line audit of FP/BP/filtering pixel coordinate consistency
- **Findings:**

#### pixel_size vs pixel_row_size (SELF-CONSISTENT BUG)
All three modules use `pixel_size` (column-based) for BOTH u and v directions:
  - **siddon.jl:511-512** — `u_offset` and `v_offset` both use `pixel_size * magnification`
  - **filtering.jl:204-205** — cosine weighting `u` and `v` both use `pixel_size * magnification`
  - **backprojection.jl:328** — `pixel_mag = pixel_size * magnification` used for both u and v

**Expected:** v direction should use `pixel_row_size` (0.625mm) instead of `pixel_size` (1.0mm)

**Key insight:** The bug is SELF-CONSISTENT across FP, filtering, and BP. Since all three use the same wrong value, the errors partially cancel. This may not be the primary artifact cause, but it means the z-direction ray geometry is wrong (stretched by 1.6x factor = 1.0/0.625).

#### col_center / row_center (CONSISTENT — OK)
  - siddon.jl:448-449: `(n_cols+1)/2` and `(n_rows+1)/2`
  - backprojection.jl:331-332: same formula
  - filtering.jl:192-193: same formula
  - **Status:** CONSISTENT across all modules ✓

#### CTGeometry struct and constructor (OK)
  - scanner.jl:514-528: struct has both `pixel_size` and `pixel_row_size` fields ✓
  - scanner.jl:582-585: constructor computes both separately from `detector_col_size` and `detector_row_size` ✓

#### FOV propagation (OK)
  - driver.jl:261-266: CTGeometry created with `fov_cm` from recon_opts
  - scanner.jl:596-603: z_cm auto-computed from detector row coverage when not specified
  - **Status:** Consistent ✓

#### Workspace geometry (OK)
  - workspace.jl:284-291: copies all geometry arrays faithfully ✓

- **Summary:** The pixel_size/pixel_row_size bug exists but is self-consistent. It won't cause ghost artifacts — just incorrect z-scaling. The ghost artifacts likely have a different root cause.
- **Next:** Run TEST-BARE-SIDDON to determine if artifacts are from physics effects or core ray tracing

### TEST-BARE-SIDDON — Iteration 1 (2026-02-12)
- **Test:** Bare Siddon + FDK with ALL physics disabled (PHYSICS_ENABLED=false)
- **Settings:** 844 cols × 64 rows, 1.0mm col / 0.625mm row, 1600 views, 120 kVp
- **Result:** **ARTIFACTS APPEAR ABSENT** — The bare Siddon reconstruction looks clean
- **Observation:** The XCAT chest phantom reconstructs with clear anatomical detail. Ribs appear as clean bright arcs without obvious ghost/aliasing halos. Soft tissue regions appear smooth. HU range [-1455, 2329] looks reasonable for bare (no physics) reconstruction.
- **μ_water:** 0.223 cm⁻¹ (reasonable for polychromatic 120 kVp — may include some BH since this is polychromatic FP even without physics effects)
- **Image:** diag_bare_siddon.png, sino_bare_siddon.png
- **Conclusion:** The core ray tracing (Siddon FP) + FDK reconstruction pipeline appears clean. Ghost artifacts are likely introduced by one or more physics effects.
- **Next:** Run baseline with ALL physics enabled to confirm artifacts appear, then physics ablation (TEST-PHYSICS-ABLATION)

### TEST-BARE-SIDDON — Iteration 2 (2026-02-12): All Physics Baseline
- **Test:** All physics enabled (fill_factor, flat_filter, bowtie, det_eff, crosstalk, optical, focal_spot, lag, heel, bhc)
- **Settings:** Same detector config (844×64, 1.0mm/0.625mm), 1600 views, 120 kVp
- **Result:** **ARTIFACTS PRESENT — SUBTLE** — The all-physics image shows slightly more blur/halos around high-contrast edges compared to bare Siddon
- **Observation:**
  - μ_water = 0.269 cm⁻¹ (higher than bare: 0.223, because physics effects add filtration/attenuation)
  - HU range [-1050, 1249] is more compressed than bare [-1455, 2329]
  - The image shows slightly smeared rib edges compared to bare Siddon
  - Simulation took 300s (vs 3.6s for bare) — physics effects are computationally expensive
- **Image:** diag_all_physics.png, sino_all_physics.png
- **Conclusion:** Artifacts are subtle at this display resolution but visible in comparison. Physics effects introduce the edge artifacts.
- **Next:** Begin physics ablation — test crosstalk first (highest priority suspect)

