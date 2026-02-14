# Z/XY Geometry Scaling Audit — Progress Log

> Started: 2026-02-14
> Goal: Exhaustive audit of z-direction and xy-direction geometry scaling

---

## Pre-Audit Summary

**Recent fixes verified by Explore agent:**
- 7 files correctly use pixel_row_size for z-direction
- CTGeometry constructor correctly uses detector_row_size for z
- NB05 has per-scanner water calibration z_cm

**Gaps identified:**
- polychromatic.jl, driver.jl, photon_counting.jl: volume_fov threading not fully verified
- scatter.jl, crosstalk.jl, heel_effect.jl, PCCT files: not checked for pixel_size/pixel_row_size
- SIRT/CGLS/Hybrid IR: not verified for z-direction handling
- NB01-04: z_cm consistency not checked

---

## Audit Log

### 2026-02-14: GEO-001 — PASS (2 issues found)

**Agent:** Exhaustive grep of pixel_size/pixel_row_size/pixel_col_size in src/
**Method:** `grep -rn 'pixel_size\|pixel_row_size\|pixel_col_size' src/` — every hit checked

#### Files with NO pixel_size/pixel_row_size references (CLEAN — no action needed):
- src/projection/polychromatic.jl
- src/detector/scatter.jl
- src/detector/crosstalk.jl
- src/detector/detector_lag.jl
- src/detector/fill_factor.jl
- src/detector/physics_pipeline.jl
- src/detector/das_model.jl
- src/detector/pcct/charge_collection.jl
- src/detector/pcct/k_fluorescence.jl (uses pixel_pitch_mm tuple — correct)
- src/detector/pcct/pileup_model.jl
- src/detector/pcct/cdte_constants.jl (uses pixel_pitch_mm tuple — correct)
- src/source/heel_effect.jl
- src/source/protocol.jl
- src/source/spectrum.jl
- src/object/*.jl
- src/correction/*.jl
- src/spectral/*.jl
- src/reconstruction/ir/sirt.jl
- src/reconstruction/ir/cgls.jl
- src/reconstruction/hybrid_ir/hybrid_ir.jl
- src/reconstruction/statistical_ir.jl
- src/scanners/general_electric.jl
- src/scanners/siemens.jl
- src/api/options.jl

#### Detailed Audit Table

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| **src/projection/siddon.jl** | | | | |
| :339 | `pixel_size` | — (docstring) | N/A | Documentation |
| :451 | `pixel_size = T(geom.pixel_size)` | xy (col) | CORRECT | Stored for u_offset |
| :452 | `pixel_row_size = T(geom.pixel_row_size)` | z (row) | CORRECT | Stored for v_offset |
| :518 | `pixel_size * magnification` | xy (col) | CORRECT | u_offset = col * pixel_size * mag |
| :519 | `pixel_row_size * magnification` | z (row) | CORRECT | v_offset = row * pixel_row_size * mag |
| **src/source/flat_filter.jl** | | | | |
| :247 | `pixel_size_det = geom.pixel_size * (SDD/SAD)` | xy (col) | CORRECT | u_offset |
| :248 | `pixel_row_size_det = geom.pixel_row_size * (SDD/SAD)` | z (row) | CORRECT | v_offset |
| :253 | `pixel_row_size_det` | z (row) | CORRECT | v_offset |
| :258 | `pixel_size_det` | xy (col) | CORRECT | u_offset |
| :310-311 | Same pattern | xy/z | CORRECT | Second function |
| :315 | `pixel_row_size_det` | z (row) | CORRECT | v_offset |
| :319 | `pixel_size_det` | xy (col) | CORRECT | u_offset |
| **src/source/focal_spot.jl** | | | | |
| :35,37 | Comments | — | N/A | Documentation |
| :233,237,247 | Comments | — | N/A | Documentation |
| :295 | `pixel_size_det = geom.pixel_size * (SDD/SAD)` | xy (col) | CORRECT | blur_width |
| :296 | `pixel_row_size_det = geom.pixel_row_size * (SDD/SAD)` | z (row) | CORRECT | blur_length |
| :297 | `blur_width_pixels = blur_width_cm / pixel_size_det` | xy (col) | CORRECT | Width → xy |
| :298 | `blur_length_pixels = blur_length_cm / pixel_row_size_det` | z (row) | CORRECT | Length → z |
| **src/source/bowtie_filter.jl** | | | | |
| :711 | `pixel_size_det = geom.pixel_size * (SDD/SAD)` | xy (col) | CORRECT | u_offset |
| :712 | `pixel_row_size_det = geom.pixel_row_size * (SDD/SAD)` | z (row) | CORRECT | v_offset |
| :718 | `pixel_row_size_det` | z (row) | CORRECT | v_offset |
| :724 | `pixel_size_det` | xy (col) | CORRECT | u_offset |
| :777-778 | Same pattern | xy/z | CORRECT | Second function |
| :782 | `pixel_row_size_det` | z (row) | CORRECT | v_offset |
| :787 | `pixel_size_det` | xy (col) | CORRECT | u_offset |
| **src/detector/detector_efficiency.jl** | | | | |
| :323 | `pixel_row_size_det = geom.pixel_row_size * (SDD/SAD)` | z (row) | CORRECT | v_offset |
| :328 | `pixel_row_size_det` | z (row) | CORRECT | v_offset (only uses z) |
| :374 | `pixel_row_size_det = geom.pixel_row_size * (SDD/SAD)` | z (row) | CORRECT | Second function |
| :379 | `pixel_row_size_det` | z (row) | CORRECT | v_offset |
| **src/reconstruction/core/backprojection.jl** | | | | |
| :325 | `pixel_size = T(geom.pixel_size)` | xy (col) | CORRECT | pixel_mag |
| :326 | `pixel_row_size = T(geom.pixel_row_size)` | z (row) | CORRECT | pixel_row_mag |
| :329 | `pixel_mag = pixel_size * magnification` | xy (col) | CORRECT | u-direction |
| :330 | `pixel_row_mag = pixel_row_size * magnification` | z (row) | CORRECT | v-direction |
| **src/reconstruction/core/filtering.jl** | | | | |
| :77-108 | `pixel_size` (param/docstring) | — | N/A | create_spatial_kernel param — always receives col pixel |
| :94 | `pixel_size::T` (function param) | xy (col) | CORRECT | Ramp filter spacing = column pixel |
| :97 | `Δ = pixel_size` | xy (col) | CORRECT | Filter operates row-by-row on columns |
| :342 | `pixel_size = T(geom.pixel_size)` | xy (col) | CORRECT | cosine_weight! u direction |
| :343 | `pixel_row_size = T(geom.pixel_row_size)` | z (row) | CORRECT | cosine_weight! v direction |
| :361 | `pixel_size * magnification` | xy (col) | CORRECT | u position |
| :362 | `pixel_row_size * magnification` | z (row) | CORRECT | v position |
| :420 | `pixel_size = T(geom.pixel_size)` | xy (col) | CORRECT | filter kernel spacing |
| :431 | `create_spatial_kernel(..., pixel_size)` | xy (col) | CORRECT | Ramp filter in column direction |
| **src/reconstruction/fbp/fdk.jl** | | | | |
| :185 | `pixel_size` | — (docstring) | N/A | Documentation |
| :423 | `geom.pixel_size, geom.pixel_row_size` | xy, z | CORRECT | Passed to CTGeometry constructor |
| **src/reconstruction/mbir/mbir.jl** | | | | |
| :440 | `geom.pixel_size` | xy (col) | CORRECT | Passed to create_subset_geometry |
| :441 | `geom.pixel_row_size` | z (row) | CORRECT | Passed to create_subset_geometry |
| **src/detector/detector_noise.jl** | | | | |
| :721 | `pixel_size` | — (docstring) | N/A | Documentation |
| :731 | `pixel_col_det_mm = (geom.pixel_size * 10.0) * magnification` | xy (col) | CORRECT | Column pixel at detector |
| :732 | `pixel_row_det_mm = (geom.pixel_row_size * 10.0) * magnification` | z (row) | CORRECT | Row pixel at detector |
| **src/detector/photon_counting.jl** | | | | |
| :100 | `pixel_size_mm::Tuple{T,T}` — (row, col) | both | CORRECT | Direction-aware tuple |
| :131 | `pixel_size_mm = (0.302, 0.302)` | both | CORRECT | Default value (tuple) |
| :144 | `pixel_size_mm::Tuple{T,T}` | both | CORRECT | Struct field |
| :177-192-205 | `pixel_size_mm` kwarg/docstring | both | CORRECT | Tuple (row, col) |
| :235 | `pixel_size_mm = (0.302, 0.302)` | both | CORRECT | Preset |
| :265 | `pixel_size_mm = (0.151, 0.151)` | both | CORRECT | UHR preset |
| :289 | `pixel_size_mm = (0.302, 0.302)` | both | CORRECT | Preset |
| :537 | `Float64.(detector.pixel_size_mm)` | both | CORRECT | Tuple passthrough |
| :546 | `pixel_pitch = detector.pixel_size_mm` | both | CORRECT | Alias |
| :643-644 | `pixel_row = detector.pixel_size_mm[1]`, `pixel_col = detector.pixel_size_mm[2]` | row/col | CORRECT | Tuple [1]=row, [2]=col |
| :743-744 | Same pattern | row/col | CORRECT | Second usage |
| :873 | `detector.pixel_size_mm[1] * detector.pixel_size_mm[2]` | area | CORRECT | row × col = area |
| :1002 | Same area calculation | area | CORRECT | |
| :1407 | `pixel_size_mm=detector.pixel_size_mm` | both | CORRECT | Passthrough |
| :1705-1706 | `pixel_row/pixel_col` from tuple | row/col | CORRECT | |
| :1725 | Area from tuple product | area | CORRECT | |
| :1968 | `pixel_size_mm = detector.pixel_size_mm` | both | CORRECT | Info struct |
| :1994 | Print statement | both | CORRECT | Display |
| :2284-2337 | `pixel_size_mm` param + usage | both | CORRECT | Fluorescence model |
| :2323 | `min(pixel_size_mm[1], pixel_size_mm[2]) / 2` | both | CORRECT | Uses smaller dim |
| :2593-2699 | `pixel_size_mm` kwarg + passthrough | both | CORRECT | compute_spectral_response_matrix |
| **src/detector/pcct/detector_response.jl** | | | | |
| :136 | `pixel_size_mm = detector.pixel_size_mm` | both | CORRECT | Tuple |
| :147 | `compute_cdte_fluorescence_model(pixel_size_mm, ...)` | both | CORRECT | Tuple |
| :152 | `compute_fluorescence_escape_probability(..., pixel_size_mm)` | both | CORRECT | Tuple |
| :185 | `pixel_pitch_mm=pixel_size_mm` | both | CORRECT | Tuple |
| **src/detector/pcct/charge_transport.jl** | | | | |
| :337 | `σ/pixel_size` | — (comment) | N/A | Documentation only |
| **src/api/driver.jl** | | | | |
| :486 | `pixel_size_mm=detector.pixel_size_mm` | both | CORRECT | Passthrough to compute_spectral_response_matrix |
| :498 | `pixel_size_mm=detector.pixel_size_mm` | both | CORRECT | Same |
| **src/api/workspace.jl** | | | | |
| :211 | `pixel_size_mm=pcct_detector.pixel_size_mm` | both | CORRECT | Passthrough |
| :315 | `Float64.(pcct_detector.pixel_size_mm)` | both | CORRECT | Tuple conversion |
| :319 | `compute_cdte_fluorescence_model(pcct_detector.pixel_size_mm, ...)` | both | CORRECT | Tuple |
| :325 | `charge_sharing_probability(σ, pcct_detector.pixel_size_mm)` | both | CORRECT | Tuple |
| :1078 | `pixel_size = T(geom.pixel_size)` | xy (col) | CORRECT | Filter kernel spacing (FDK workspace) |
| :1082 | `create_spatial_kernel(..., pixel_size)` | xy (col) | CORRECT | Ramp filter |
| :1191 | `pixel_size = T(geom.pixel_size)` | xy (col) | CORRECT | Filter kernel spacing (HIR workspace) |
| :1195 | `create_spatial_kernel(..., pixel_size)` | xy (col) | CORRECT | Ramp filter |
| **src/scanners/scanners.jl** | | | | |
| :285 | `pixel_size_cm = (_fov_cm * 1.1) / _n_cols` | xy (col) | CORRECT | Column pixel from FOV |
| :323 | `pixel_row_size_cm = det.row_size_mm[] / 10.0` | z (row) | CORRECT | Row pixel from detector spec |
| :324 | `z_coverage_cm = n_rows * pixel_row_size_cm` | z (row) | CORRECT | Z coverage |
| :328 | `..., pixel_size_cm, pixel_row_size_cm, ...` | xy, z | CORRECT | Passed to CTGeometry |
| **src/geometry/scanner.jl** | | | | |
| :500 | `pixel_size` | — (docstring) | N/A | CTGeometry struct field doc |
| :520 | `pixel_size::Float64` | xy (col) | CORRECT | CTGeometry struct field |
| :521 | `pixel_row_size::Float64` | z (row) | CORRECT | CTGeometry struct field |
| :583 | `pixel_size = scanner.detector_col_size / 10.0` | xy (col) | CORRECT | create_ct_geometry |
| :584 | `pixel_row_size = scanner.detector_row_size / 10.0` | z (row) | CORRECT | create_ct_geometry |
| :591 | `fov_xy = _n_cols * pixel_size` | xy (col) | CORRECT | Default FOV |
| :641 | `..., pixel_size, pixel_row_size, ...` | xy, z | CORRECT | CTGeometry constructor |
| :691 | `pixel_size = pixel_pitch_mm / 10.0` | xy (col) | CORRECT | create_aquilion_one (single pitch for square pixels) |
| :692 | `fov_xy = pixel_size * n_cols` | xy (col) | CORRECT | FOV from cols |
| :696 | `pixel_size = (fov_cm * 1.1) / n_cols` | xy (col) | CORRECT | FOV override |
| :702 | `fov_z = pixel_size * n_rows` | **z (row)** | **WRONG** | Uses `pixel_size` (column) for z-direction FOV. Should use row pixel size. |
| :746 | `..., pixel_size, pixel_size, ...` | xy, **z** | **WRONG** | Passes `pixel_size` as BOTH pixel_size AND pixel_row_size to CTGeometry. Should be `pixel_size, pixel_row_size` (but Aquilion ONE has 0.5mm square pixels, so numerically same). |
| :809 | `pixel_size = 0.2` | both | N/A | NAEOTOM Alpha local var (square pixels, used for both row/col) |
| :813 | `pixel_size = 0.4` | both | N/A | NAEOTOM Alpha standard mode (square pixels) |
| :828 | `detector_row_size = pixel_size` | z (row) | CORRECT* | *Correct for square pixels |
| :829 | `detector_col_size = pixel_size` | xy (col) | CORRECT | |
| :832 | `detector_col_offset = pixel_size / 2` | xy (col) | CORRECT | Quarter-detector offset |
| :886 | `pixel_size_mm = (scanner.detector_row_size * mag, scanner.detector_col_size * mag)` | (row, col) | CORRECT | Tuple with explicit row/col |
| **src/metrics/psf.jl** | | | | |
| :183-807 | `pixel_size_mm` | in-plane | N/A | Function parameter for recon pixel size, NOT geom.pixel_size. Used for generic 2D image measurement. Not a geometry direction issue. |
| **src/metrics/nps.jl** | | | | |
| :173-417 | `pixel_size_mm` | in-plane | N/A | Same as psf.jl — generic recon pixel size parameter. Not geometry direction issue. |
| **src/metrics/mtf.jl** | | | | |
| :148-711 | `pixel_size_mm` | in-plane | N/A | Same as psf.jl — generic recon pixel size parameter. Not geometry direction issue. |

#### Issues Found: 2

**Issue 1: scanner.jl:702 — `fov_z = pixel_size * n_rows`**
- `create_aquilion_one()` uses `pixel_size` (column direction, 0.5mm) for z-direction FOV default.
- Should use a separate `pixel_row_size` for z-direction.
- **Impact: NONE for Aquilion ONE** — it has square 0.5mm pixels, so `pixel_size == pixel_row_size`.
- **Impact if code is copied for non-square scanner: BUG**
- Severity: LOW (cosmetic for current scanner, but incorrect pattern)

**Issue 2: scanner.jl:746 — `pixel_size, pixel_size` passed to CTGeometry**
- `create_aquilion_one()` passes `pixel_size` as both `pixel_size` and `pixel_row_size` in CTGeometry constructor.
- Should pass `pixel_size, pixel_row_size` (separate variables).
- **Impact: NONE for Aquilion ONE** — square 0.5mm pixels.
- **Impact if code is copied for non-square scanner: BUG** — z-direction would use wrong pixel pitch.
- Severity: LOW (cosmetic for current scanner, but incorrect pattern)

Note: Both issues are in `create_aquilion_one()` which hardcodes `pixel_pitch_mm = 0.5` (square). The `create_ct_geometry()` (scanner.jl:583-641) and `create_scanner_geometry()` (scanners.jl:285-331) correctly use separate row/col pixel sizes. The NAEOTOM Alpha `create_naeotom_alpha()` (scanner.jl:805-865) also uses square pixels and stores them separately in `detector_row_size`/`detector_col_size`.

### 2026-02-14: GEO-002 — PASS (0 issues found)

**Agent:** Exhaustive grep of detector_row_size/detector_col_size/detector_size in src/ AND verification/
**Method:** `grep -rn 'detector_row_size\|detector_col_size\|detector_size' src/ verification/` — every hit checked
**Also checked:** `\bdetector_size\b` (word-boundary) — **zero matches** (no ambiguous bare `detector_size` anywhere)

#### Files with NO detector_row_size/detector_col_size references (CLEAN):
- All src/ files not listed below (polychromatic.jl, siddon.jl, backprojection.jl, filtering.jl, flat_filter.jl, bowtie_filter.jl, focal_spot.jl, detector_efficiency.jl, detector_noise.jl, detector_lag.jl, crosstalk.jl, fill_factor.jl, physics_pipeline.jl, das_model.jl, heel_effect.jl, protocol.jl, spectrum.jl, object/*.jl, correction/*.jl, spectral/pcct_spectral.jl, reconstruction/*.jl, metrics/*.jl, api/options.jl, scanners/general_electric.jl, scanners/siemens.jl, scanners/helical_protocols.jl)

#### Detailed Audit Table — src/

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| **src/geometry/scanner.jl** | | | | |
| :62 | `detector_row_size` | — (docstring) | N/A | Documentation: "z (mm) at isocenter" |
| :63 | `detector_col_size` | — (docstring) | N/A | Documentation: "fan direction (mm) at isocenter" |
| :108 | `detector_row_size` | — (CatSim mapping table) | N/A | Doc: maps to detectorRowSize |
| :109 | `detector_col_size` | — (CatSim mapping table) | N/A | Doc: maps to detectorColSize |
| :134 | `detector_row_size::T` | z (row) | CORRECT | Scanner struct field definition |
| :135 | `detector_col_size::T` | xy (col) | CORRECT | Scanner struct field definition |
| :187 | `detector_row_size` | — (docstring) | N/A | Kwarg doc: "row pitch (mm)" |
| :188 | `detector_col_size` | — (docstring) | N/A | Kwarg doc: "column pitch (mm)" |
| :218 | `detector_row_size = 0.625` | — (docstring example) | N/A | Example code in docstring |
| :227 | `detector_row_size = 0.15` | — (docstring example) | N/A | Flat panel example |
| :228 | `detector_col_size = 0.15` | — (docstring example) | N/A | Flat panel example |
| :240 | `detector_row_size::Real = 1.0` | z (row) | CORRECT | Scanner constructor kwarg |
| :241 | `detector_col_size::Real = 1.0` | xy (col) | CORRECT | Scanner constructor kwarg |
| :302 | `T(detector_row_size)` | z (row) | CORRECT | Passed to Scanner struct field |
| :303 | `T(detector_col_size)` | xy (col) | CORRECT | Passed to Scanner struct field |
| :388 | `scanner.detector_row_size <= 0` | z (row) | CORRECT | Validation check |
| :389 | `scanner.detector_row_size` (error msg) | — | N/A | Error message |
| :393 | `scanner.detector_col_size <= 0` | xy (col) | CORRECT | Validation check |
| :394 | `scanner.detector_col_size` (error msg) | — | N/A | Error message |
| :422 | `# detector_col_size is already at isocenter` | — (comment) | N/A | Documentation |
| :423 | `scanner.detector_cols * scanner.detector_col_size` | xy (col) | CORRECT | XY detector coverage at iso |
| :428 | `# Z coverage (detector_row_size is already at isocenter)` | — (comment) | N/A | Documentation |
| :430 | `scanner.detector_rows * scanner.detector_row_size` | z (row) | CORRECT | Z coverage at isocenter |
| :460 | `scanner.detector_col_size × scanner.detector_row_size` | both (display) | CORRECT | Print col×row element size |
| :462 | `scanner.detector_rows * scanner.detector_row_size` | z (row) | CORRECT | Z coverage for display |
| :583 | `scanner.detector_col_size / 10.0` → `pixel_size` | xy (col) | CORRECT | CTGeometry: col → pixel_size |
| :584 | `scanner.detector_row_size / 10.0` → `pixel_row_size` | z (row) | CORRECT | CTGeometry: row → pixel_row_size |
| :598 | `# detector_row_size is already at isocenter` | — (comment) | N/A | Documentation |
| :599 | `scanner.detector_rows * scanner.detector_row_size` | z (row) | CORRECT | Default z coverage |
| :802 | `scanner_uhr.detector_row_size` | — (docstring) | N/A | Example showing 0.2mm |
| :828 | `detector_row_size = pixel_size` | z (row) | CORRECT | NAEOTOM: pixel_size is local var (square pixels) |
| :829 | `detector_col_size = pixel_size` | xy (col) | CORRECT | NAEOTOM: pixel_size is local var (square pixels) |
| :886 | `scanner.detector_row_size * magnification` | z (row) | CORRECT | PCCT detector tuple[1] = row |
| :886 | `scanner.detector_col_size * magnification` | xy (col) | CORRECT | PCCT detector tuple[2] = col |
| **src/detector/scatter.jl** | | | | |
| :708 | `Scanner(detector_col_size=1.0)` | — (docstring) | N/A | Example code |
| :712 | `Scanner(detector_col_size=0.5)` | — (docstring) | N/A | Example code |
| :717 | `# detector_col_size is at isocenter` | — (comment) | N/A | Documentation |
| :719 | `scanner.detector_col_size * magnification` | xy (col) | CORRECT | Scatter kernel is applied in-plane (xy), column-direction pitch is correct for the 2D Gaussian scatter kernel |
| **src/spectral/dual_energy.jl** | | | | |
| :282 | `detector_col_size = det_spec.col_size_mm.value` | xy (col) | CORRECT | Builds Scanner from CatSim spec, col → col |
| :283 | `detector_row_size = det_spec.row_size_mm.value` | z (row) | CORRECT | Builds Scanner from CatSim spec, row → row |

#### Detailed Audit Table — verification/

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| **NB01 (01_single_kvp_verification.jl)** | | | | |
| :542 | `detector_row_size = SIM_CONFIG.detectorRowSize` | z (row) | CORRECT | Row → row mapping |
| :545 | `detector_col_size = SIM_CONFIG.detectorColSize` | xy (col) | CORRECT | Col → col mapping |
| **NB02 (02_multi_dose_and_iterative_reconstruction.jl)** | | | | |
| :231 | `detector_row_size = SIM_CONFIG.detectorRowSize` | z (row) | CORRECT | Row → row mapping |
| :234 | `detector_col_size = SIM_CONFIG.detectorColSize` | xy (col) | CORRECT | Col → col mapping |
| **NB03 (03_dual_kvp_vmi_verification.jl)** | | | | |
| :118 | `detector_row_size = SIM_CONFIG.detectorRowSize` | z (row) | CORRECT | Row → row mapping |
| :121 | `detector_col_size = SIM_CONFIG.detectorColSize` | xy (col) | CORRECT | Col → col mapping |
| **NB04 (04_pcct_demonstration.jl)** | | | | |
| :110 | `detector_row_size = SIM_CONFIG.detectorRowSize` | z (row) | CORRECT | Row → row mapping |
| :113 | `detector_col_size = SIM_CONFIG.detectorColSize` | xy (col) | CORRECT | Col → col mapping |
| :162 | `naeotom.detector_col_size` | — (display) | CORRECT | Shows element pitch in markdown string (NAEOTOM has square pixels; col pitch is appropriate for a generic "pitch" display) |
| **NB05 (05_xcat_full.jl)** | | | | |
| :621 | `detector_row_size = 0.625` | z (row) | CORRECT | GE Revolution z-direction pitch |
| :622 | `detector_col_size = eict_col_size_iso` | xy (col) | CORRECT | GE Revolution in-plane pitch (computed variable) |
| :659 | `detector_row_size = 0.4` | z (row) | CORRECT | NAEOTOM z-direction pitch |
| :660 | `detector_col_size = 0.4` | xy (col) | CORRECT | NAEOTOM in-plane pitch |
| :785 | `scanner_eict.detector_row_size / 10.0` | z (row) | CORRECT | z_cm for water calibration: 8 × 0.625mm / 10 = 0.5cm |
| :794 | `scanner_pcct_standard.detector_row_size / 10.0` | z (row) | CORRECT | z_cm for water calibration: 8 × 0.4mm / 10 = 0.32cm |

#### Issues Found: 0

All occurrences of `detector_row_size` and `detector_col_size` are used for the correct direction (z and xy respectively). No ambiguous bare `detector_size` found anywhere in src/ or verification/.

**Summary:**
- **scanner.jl**: Struct definition, constructor, validation, CTGeometry conversion, NAEOTOM Alpha factory, PCCT builder — all correct
- **scatter.jl**: Uses `detector_col_size` for in-plane scatter kernel — correct (scatter is applied as 2D kernel in sinogram plane)
- **dual_energy.jl**: CatSim spec mapping col→col, row→row — correct
- **NB01-04**: All use `SIM_CONFIG.detectorRowSize` → `detector_row_size` and `SIM_CONFIG.detectorColSize` → `detector_col_size` — correct
- **NB05**: Hardcoded values and per-scanner z_cm calculations — all correct
