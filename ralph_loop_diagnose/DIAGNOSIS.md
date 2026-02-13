# Ghost Artifact Diagnosis Report

## Summary

- **Root cause:** The "ghost-like aliasing" artifacts around ribs are **residual bone beam hardening artifacts** — an expected physical effect of polychromatic X-ray simulation with water-only beam hardening correction (BHC). The water-based BHC corrects for soft tissue but under-corrects for bone, leaving dark halos/streaks around dense structures. This is not a code bug.
- **Confidence:** HIGH
- **Recommended fix:** Implement multi-material (bone + water) BHC, or implement iterative BHC that accounts for the spectral behavior of high-Z materials. Alternatively, if the user wants monochromatic-equivalent images, run with all spectral triggers disabled (use_flat_filter=false, use_bowtie_filter=false, use_detector_efficiency=false, use_bhc=false).

---

## Evidence

### 1. Bare Siddon Test (TEST-BARE-SIDDON)

- **Result:** ARTIFACTS ABSENT
- **Images:** `diag_bare_siddon.png`, `sino_bare_siddon.png`
- **Details:** Bare Siddon FP + FDK reconstruction (no physics effects, monochromatic at 60 keV) produces a clean XCAT reconstruction with sharp rib edges and no halos. The core ray tracing and reconstruction pipeline is correct.
- **Metrics:** center μ = 0.223 cm⁻¹, HU range [-1455, 2329]

### 2. Physics Ablation (TEST-PHYSICS-ABLATION)

All 10 physics effects tested individually. Perfect binary split:

| Effect | Triggers Spectral? | Artifact Present? |
|--------|:---:|:---:|
| crosstalk | NO | NO |
| optical_crosstalk | NO | NO |
| focal_spot | NO | NO |
| detector_lag | NO | NO |
| fill_factor | NO | NO |
| heel_effect | NO | NO |
| **flat_filter** | **YES** | **YES** |
| **bowtie_filter** | **YES** | **YES** |
| **detector_efficiency** | **YES** | **YES** |
| **bhc** | **YES** | **YES** |

**Conclusion:** The artifact appears if and only if the polychromatic spectral integration pathway is activated. ANY of the 4 spectral triggers reproduces the same artifact pattern. The 6 non-spectral effects produce images identical to bare Siddon.

The triggering mechanism is `needs_polychromatic()` in `driver.jl:1261-1266`, which activates full spectrum simulation (30+ energy bins, Beer-Lambert integration) whenever flat_filter, bowtie, detector_efficiency, or BHC is enabled.

### 3. Dexel Sweep (TEST-DEXEL-SWEEP)

| Config | Col Size | Cols | Artifact Visibility |
|--------|----------|------|---------------------|
| coarse | 2.0mm | 422 | Present — lower spatial resolution smears the pattern |
| default | 1.0mm | 844 | Present — subtle dark halos at bone boundaries |
| fine | 0.5mm | 1688 | Present — halos more pronounced, better spatially resolved |
| square | 1.0mm×1.0mm | 844 | Identical to default (pixel_row_size bug ruled out) |

**Trend:** Artifacts become more visible with finer detector pixels. This is consistent with beam hardening — finer sampling better resolves the spatial pattern of the beam hardening difference signal, making the cupping/halo pattern more apparent. This is NOT aliasing (which would get better with finer sampling).

**Square pixel test:** Making detector_row_size = detector_col_size = 1.0mm (eliminating the pixel_size/pixel_row_size self-consistent bug) produces identical results. The pixel_row_size bug is confirmed as NOT causing these artifacts.

### 4. Sinogram Inspection (INSPECT-SINOGRAM)

- **Artifact visible in raw sinogram:** YES — the sinogram difference (polychromatic - bare) is smooth and concentrated at ray paths through bone, consistent with beam hardening.
- **Sinogram profile:** Polychromatic values are uniformly LOWER than monochromatic through body center (beam hardening reduces effective μ). No oscillations, no ringing, no high-frequency anomalies.
- **Difference profile:** Peak difference = -1.0 Δ(μ·L) through thickest bone paths. Smooth transitions, no edge artifacts.
- **Reconstruction difference:** Classic beam hardening pattern — negative (dark) halos around all bone structures, positive (bright) cupping in body center. Matches textbook beam hardening exactly.

**Conclusion:** The artifact is introduced during **forward projection** (polychromatic Beer-Lambert spectral integration), not during reconstruction. The sinogram data itself contains the beam hardening effect, which FDK faithfully reconstructs.

### 5. Code Audit (CODE-AUDIT)

- **pixel_size/pixel_row_size:** SELF-CONSISTENT BUG — siddon.jl:511-512, filtering.jl:204-205, and backprojection.jl:328 all use `pixel_size` (column-based) for both u and v. Should use `pixel_row_size` for v, but since all three are consistent, errors cancel. Does NOT cause ghost artifacts (confirmed by square pixel test).
- **col_center/row_center:** CONSISTENT across FP, BP, and filtering. All use `(n+1)/2`.
- **Detector position FP↔BP:** CONSISTENT — exact inverses of each other.
- **FOV propagation:** CONSISTENT — driver.jl and workspace.jl propagate FOV correctly.

### 6. BHC Implementation Analysis

The BHC is explicitly **water-based only** (documented at `beam_hardening_correction.jl:57-62`):

> "Water-based BHC corrects for water-like tissues but under-corrects for bone, causing residual 'dark band' artifacts between dense structures."

The implementation:
- Calibration creates a polynomial mapping measured (polychromatic) line integrals → monochromatic reference values, using **water only** as the calibration material
- Default coefficients: `[0.0, 1.05, -0.02, 0.001]` — a 3rd order polynomial
- Applied as a post-log transform on the sinogram

This means water-like tissues (soft tissue, organs, blood) get properly corrected, but **bone and other high-Z materials have different spectral behavior** that the water-based polynomial cannot model. The residual cupping/halos around bone are the expected result.

---

## Root Cause Analysis

The "ghost-like aliasing artifacts" around ribs in the XCAT reconstruction are **residual bone beam hardening artifacts** resulting from:

1. **Polychromatic X-ray simulation** — The Beer-Lambert spectral integration (`polychromatic.jl:1130-1143`) correctly models how a polychromatic X-ray beam is progressively hardened (low-energy photons preferentially absorbed) as it passes through the body. The resulting sinogram values are nonlinearly lower than what a monochromatic beam would produce, especially through bone.

2. **Water-only BHC** — The implemented beam hardening correction (`beam_hardening_correction.jl`) only calibrates against water. For water-like tissues, this correction is accurate. For bone (with its higher effective Z and different spectral attenuation curve), the correction is insufficient, leaving residual artifacts.

3. **Appearance as "ghost-like aliasing"** — The residual beam hardening manifests as:
   - **Dark halos** around ribs and spine (under-corrected bone paths)
   - **Cupping** in the body center (residual path-length-dependent error)
   - **Dark streaks** between dense structures (crossing bone paths from multiple angles)

   These patterns can appear "ghost-like" because they follow the shape of the bone structures but are displaced/smeared, and they can appear "aliasing-like" because the beam hardening effect is view-dependent (each projection angle sees bone at different locations).

4. **"More dexels = worse" explained** — Finer detector pixels better resolve the spatial frequency content of the beam hardening error signal. At coarse detector resolution (2mm), the beam hardening pattern is smoothed by the limited spatial bandwidth. At fine resolution (0.5mm), the same physical effect is resolved at higher spatial detail, making the halos more visible and sharper.

---

## Secondary Finding: pixel_row_size Bug

While not the cause of the ghost artifacts, the code audit revealed that `pixel_row_size` is never used in ray tracing or backprojection:
- `siddon.jl:511-512`: v_offset uses `pixel_size` (column-based) instead of `pixel_row_size`
- `backprojection.jl:328`: `pixel_mag` uses `pixel_size` for both u and v
- `filtering.jl:204-205`: cosine weighting uses `pixel_size` for both u and v

Since all three are self-consistent, this bug causes z-direction geometry to be scaled by a factor of `col_size/row_size` = 1.6x, but doesn't produce reconstruction artifacts because FP and BP agree. The z-dimension of the reconstruction will be stretched by 1.6x compared to ground truth.

---

## Recommended Fixes

### Primary: Multi-material BHC
Implement a two-material (water + bone) beam hardening correction:
1. **Joseph & Spital method**: Decompose each ray's line integral into water-equivalent and bone-equivalent path lengths using a 2D lookup table
2. **Calibrate**: Generate correction polynomials for water+bone combinations, not just water alone
3. **Apply**: Replace the 1D water-only polynomial with a 2D (water, bone) correction surface

This is the standard approach in clinical CT scanners and is described in the CatSim/XCIST documentation.

### Secondary: Fix pixel_row_size usage
In `siddon.jl`, `backprojection.jl`, and `filtering.jl`, use `pixel_row_size` for the v-direction (row direction) calculations instead of `pixel_size`. This will correct the z-direction geometry for non-square detector pixels.

### Alternative: Monochromatic-equivalent mode
For users who don't need polychromatic physics, provide a mode that runs at a single effective energy (e.g., 65-70 keV) without spectral integration. This is already available by disabling all spectral triggers, but could be made more explicit in the API.

---

## Files Referenced

| File | Key Lines | Purpose |
|------|-----------|---------|
| `src/projection/polychromatic.jl` | 1130-1143 | Beer-Lambert spectral integration |
| `src/correction/beam_hardening_correction.jl` | 57-62, 208, 247-279, 465-509 | Water-only BHC |
| `src/api/driver.jl` | 1261-1266 | `needs_polychromatic()` gate |
| `src/projection/siddon.jl` | 511-512 | pixel_size used for both u and v |
| `src/reconstruction/core/backprojection.jl` | 328 | pixel_mag for both u and v |
| `src/reconstruction/core/filtering.jl` | 204-205 | Cosine weighting pixel_size |
| `src/geometry/scanner.jl` | 514-528, 582-585 | CTGeometry with pixel_size + pixel_row_size |

---

## Test Outputs

All diagnostic images are in `ralph_loop_diagnose/outputs/`:

| Image | Description |
|-------|-------------|
| `diag_bare_siddon.png` | Clean bare Siddon reconstruction (no artifacts) |
| `diag_all_physics.png` | All physics enabled (artifacts present) |
| `diff_flat_filter.png` | Difference image showing beam hardening pattern |
| `dexel_coarse_2mm_zoom.png` | Coarse detector zoom |
| `dexel_default_1mm_zoom.png` | Default detector zoom |
| `dexel_fine_05mm_zoom.png` | Fine detector zoom (artifacts most visible) |
| `dexel_square_1mm_zoom.png` | Square pixel test (identical to default) |
| `sino_diff_poly_vs_bare.png` | Sinogram difference showing beam hardening in projection data |
| `sino_profile_bare_vs_poly.png` | Line profiles: bare vs polychromatic |
| `recon_bare_vs_poly_comparison.png` | Side-by-side: bare, polychromatic, difference |
