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

### TEST-PHYSICS-ABLATION — Iteration 1 (2026-02-12)
- **Tests completed (4 individual effects):**

| Effect | Sim Time | μ_water | HU Range | Artifacts |
|--------|----------|---------|----------|-----------|
| bare_siddon (none) | 3.6s | 0.223 | [-1455, 2329] | ABSENT |
| all_physics | 300s | 0.269 | [-1050, 1249] | SUBTLE |
| crosstalk only | 3.2s | 0.223 | [-1409, 2195] | ABSENT |
| optical only | 3.1s | 0.223 | [-1390, 2148] | ABSENT |
| focal_spot only | 3.2s | 0.223 | [-1442, 2301] | ABSENT |
| bowtie only | 305s | 0.276 | [-1185, 1392] | SIMILAR TO ALL |
| bhc only | 302s | 0.213 | [-1388, 1826] | SIMILAR TO ALL |

- **Key Observations:**
  1. Effects that DON'T trigger spectral integration (crosstalk, optical, focal_spot) leave the image unchanged from bare Siddon — sim time ~3s, μ_water=0.223
  2. Effects that DO trigger spectral integration (bowtie, bhc) take ~300s and change μ_water significantly
  3. The bowtie filter and BHC both independently produce images similar to all-physics
  4. The "ghost artifacts" may be beam hardening artifacts (cupping, streaks near bone) rather than edge-ghost artifacts
  5. Need to look more carefully at what the user means by "ghost-like aliasing" — may need narrower window or difference imaging

- **Next:** Create difference images between bare and all-physics to isolate the artifact pattern. Also test lag and fill_factor.

### TEST-PHYSICS-ABLATION — Iteration 2 (2026-02-12): Code Path Analysis
- **Task:** Analyzed what triggers the polychromatic (spectral integration) code path
- **Finding:** `needs_polychromatic()` in driver.jl:1261-1266 returns true if ANY of these are enabled:
  - `use_flat_filter`
  - `use_bowtie_filter`
  - `use_detector_efficiency`
  - `use_bhc`
- **When polychromatic:**
  - Full spectrum loaded (30+ energy bins)
  - Siddon forward projection called 30× (once per energy)
  - Beer-Lambert spectral integration: `sinogram = -log(Σ wᵢ × exp(-μ(Eᵢ) × L))`
  - Explains the 3s → 300s slowdown (30× ray tracing)
- **When monochromatic (crosstalk/optical/focal_spot/lag/fill_factor/heel):**
  - Single energy: kVp × 0.5 = 60 keV
  - One Siddon call
  - No spectral integration
- **Implication:** The artifact is NOT caused by a specific physics effect (bowtie, bhc, etc.) — it's caused by the **polychromatic spectral integration pathway itself**. ANY of the 4 spectral triggers should reproduce it.
- **Created:** `ralph_loop_diagnose/diag_diff.jl` — difference image script that tests flat_filter, det_eff, lag, fill_factor, heel individually + creates difference images
- **Next:** Run diag_diff.jl to confirm flat_filter and det_eff also show the artifact (confirming polychromatic pathway hypothesis)

### TEST-PHYSICS-ABLATION — Iteration 3 (2026-02-12): Difference Images + All 10 Effects Tested
- **Task:** Ran diag_diff.jl — all 10 individual physics effects tested, plus difference images
- **Full results table (all 10 effects):**

| Effect | Sim Time | Center μ | μ Range | Spectral? | Artifact Change |
|--------|----------|----------|---------|-----------|-----------------|
| bare_siddon (none) | 5.9s | 0.334 | [-0.10, 0.74] | NO | baseline |
| flat_filter only | 277.5s | 0.291 | [-0.09, 0.63] | YES | PRESENT — same as all-physics |
| det_eff only | 277.9s | 0.287 | [-0.09, 0.63] | YES | PRESENT — same as all-physics |
| bowtie only | 275.1s | 0.340 | [-0.05, 0.66] | YES | PRESENT — same as all-physics |
| bhc only | 302s | 0.213 | [-1388, 1826]* | YES | PRESENT — same as all-physics |
| crosstalk only | 3.2s | 0.334 | same as bare | NO | ABSENT |
| optical only | 3.1s | 0.334 | same as bare | NO | ABSENT |
| focal_spot only | 3.2s | 0.334 | same as bare | NO | ABSENT |
| lag only | 3.1s | 0.334 | [-0.10, 0.72] | NO | ABSENT |
| fill_factor only | 2.9s | 0.339 | [-0.10, 0.75] | NO | ABSENT |
| heel only | 3.1s | 0.334 | same as bare | NO | ABSENT |

(*bhc μ values from earlier iteration with different calibration)

- **Difference Image Analysis (bare vs polychromatic):**
  - **diff_flat_filter.png:** Classic beam hardening pattern — blue (negative) halos around bone, red (positive) cupping in center body. Difference is smooth, no high-frequency "ghost" patterns.
  - **diff_det_eff.png:** Nearly identical pattern to flat_filter difference.
  - **diff_bowtie.png:** Same bone-boundary pattern plus radial modulation from bowtie filter shape (stronger attenuation at periphery).
  - **sino_diff_bowtie.png:** Sinogram difference is spatially smooth — dominated by bowtie attenuation pattern. No high-frequency artifacts or oscillations.
  - **profile_bare_vs_bowtie.png:** Line profile shows bowtie raises soft tissue μ, slightly reduces bone/soft-tissue contrast. Transitions remain smooth — no oscillations or ringing.
  - **narrow_window_comparison.png:** Narrow HU window (-50 to 100) — mostly clipped. No obvious ghost patterns visible beyond the expected contrast changes.

- **CRITICAL CONCLUSION: All artifacts correlate 100% with spectral integration (polychromatic pathway)**
  - ALL 4 spectral triggers (flat_filter, bowtie, det_eff, bhc) produce the same artifact pattern
  - ALL 6 non-spectral effects (crosstalk, optical, focal_spot, lag, fill_factor, heel) produce images identical to bare Siddon
  - The artifact pattern in difference images is **classic beam hardening** — cupping + bone-edge effects
  - This is **expected physics**, not a bug in the polychromatic code

- **GPU Closure Analysis:**
  - Investigated potential Core.Box GPU closure bugs in polychromatic.jl (CLAUDE.md anti-pattern)
  - polychromatic.jl:1105-1110 — conditional assignment of `weights_norm`, `μ_volume`, `sino_mono`, `I_transmitted`
  - polychromatic.jl:240 — conditional assignment of `μ_at_energy` in `create_μ_volume!`
  - **However:** When called through the workspace path (always the case in practice), these kwargs are ALWAYS non-nothing, so the conditional branch always takes one path. Julia's compiler likely doesn't box these.
  - siddon.jl:453-480 — conditional geometry arrays also always provided by workspace
  - **Verdict:** Core.Box may be a latent bug but is unlikely to be causing the artifacts in the workspace path

- **Revised hypothesis about user's "ghost-like aliasing":**
  The user may be describing **beam hardening artifacts** (cupping, dark streaks near bone) that are expected physical effects of polychromatic X-ray simulation. The "more dexels = worse" observation needs testing — could be that finer sampling better resolves these beam hardening patterns, making them more visible.

- **Next:** Complete TEST-PHYSICS-ABLATION story (all acceptance criteria met). Move to TEST-DEXEL-SWEEP to investigate the "more dexels = worse" observation.

### TEST-DEXEL-SWEEP — Iteration 1 (2026-02-12): Coarse / Default / Fine
- **Test:** Polychromatic (flat_filter only) at 3 detector resolutions
- **Settings:** All same: 64 rows, 1600 views, HannFilter recon to 512x512x64

| Config | Col Size | Cols | Sim Time | Center μ | μ Range | Observation |
|--------|----------|------|----------|----------|---------|-------------|
| coarse_2mm | 2.0mm | 422 | ~275s | ~0.29 | similar | Fine aliasing texture (horizontal striations), relatively smooth edges |
| default_1mm | 1.0mm | 844 | ~275s | ~0.29 | similar | Cleaner overall, sharper edges, subtle dark halos at bone boundaries |
| fine_05mm | 0.5mm | 1688 | ~275s | ~0.29 | similar | Dark halos/streaks around bone more pronounced, better resolved |

- **Trend:** Artifacts do get slightly more visible with finer sampling, BUT the artifact character is consistent beam hardening (not aliasing ghosts). Finer detector pixels better resolve the beam hardening patterns that exist in the sinogram data.
- **Image:** dexel_coarse_2mm.png, dexel_default_1mm.png, dexel_fine_05mm.png (+ zoom versions)
- **Note:** Ultrafine (0.25mm, 3376 cols) was not completed — script was killed by timeout during this run.

### TEST-DEXEL-SWEEP — Iteration 2 (2026-02-12): Square Pixel Test
- **Test:** Square pixels (col_size = row_size = 1.0mm) — tests pixel_size/pixel_row_size bug theory
- **Settings:** 844 cols × 64 rows, 1.0mm × 1.0mm, polychromatic (flat_filter only)
- **Result:** **ARTIFACTS IDENTICAL** to default (1.0mm col × 0.625mm row) configuration
- **Observation:**
  - Sim time: 652.9s (longer than expected — may be JIT recompilation due to different row_size)
  - Center μ = 0.3362, range [-0.1401, 0.6382]
  - Zoomed image looks essentially identical to default_1mm
  - The pixel_size/pixel_row_size self-consistent bug does NOT change the artifact pattern
- **Conclusion:** pixel_row_size bug is NOT the cause of the ghost artifacts. It causes z-direction scaling errors but since FP/BP/filter all use the same wrong value, the errors cancel.
- **Image:** dexel_square_1mm.png, dexel_square_1mm_zoom.png

### INSPECT-SINOGRAM — Iteration 1 (2026-02-12): Full Sinogram Analysis
- **Test:** Detailed sinogram inspection — polychromatic vs bare (monochromatic)
- **Settings:** 844 cols × 64 rows, 1.0mm col, 0.625mm row, 1600 views

#### Raw Sinogram
- **sino_full_polychromatic.png:** Full sinogram (all angles, middle row) — smooth sinusoidal traces of anatomical structures. No visible oscillations, ringing, or anomalous patterns.
- **sino_single_view.png:** Single detector view — clean 2D detector image, smooth attenuation profile across rows and columns.
- **sino_view_diff.png:** Mean absolute difference between adjacent views — shows smooth angular change concentrated at high-contrast boundaries (ribs, spine). No high-frequency oscillations.
- **sino_profiles.png:** Line profiles at 4 angles — smooth profiles with clean transitions at material boundaries. No ringing.
- **sino_angle_trace.png:** Value at peak column across all angles — smooth sinusoidal trace as bone rotates past that detector column.

#### Polychromatic vs Bare Comparison
- **sino_diff_poly_vs_bare.png:** Sinogram difference is SMOOTH — concentrated where rays pass through bone (beam hardening reduces measured μ·L). Maximum difference ~-1.0 at thickest bone paths. No high-frequency artifacts, no oscillations.
- **sino_profile_bare_vs_poly.png:** Polychromatic line profile is uniformly LOWER than bare through body center (beam hardening) but similar at edges (short path lengths). Transitions are smooth — no edge ringing.
- **sino_diff_profile.png:** Difference profile is smooth with peaks at -1.0 Δ(μ·L) through thickest bone paths. No oscillations or ringing.

#### Reconstruction Comparison
- **recon_bare_vs_poly_comparison.png:** Side-by-side bare / polychromatic / difference:
  - Bare: center=0.3804, range [-0.1015, 0.7422]
  - Poly: center=0.3204, range [-0.0852, 0.6297]
  - Difference range: [-0.1374, +0.0543]
  - Difference image shows CLASSIC beam hardening: negative (dark) halos around all bone structures, positive (bright) cupping in body center. The pattern matches textbook beam hardening artifacts exactly.

#### Key Conclusion
- **Artifact is in the SINOGRAM (forward projection)**, not introduced by reconstruction
- The sinogram difference is smooth and physically correct — it IS beam hardening
- No high-frequency "ghost" patterns visible in sinogram or reconstruction
- The "ghost-like aliasing" described by the user is most likely **beam hardening artifacts** (cupping + dark streaks near bone) — expected physics behavior of polychromatic X-ray simulation without perfect BHC

