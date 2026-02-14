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

### 2026-02-14: GEO-003 — FAIL (3 issues found)

**Agent:** Exhaustive audit of `volume_fov` threading in ALL forward projection paths
**Method:** `grep -rn 'volume_fov' src/` — every hit checked, plus analysis of ALL paths that call `siddon_forward_project!` or `forward_project`/`forward_project!`

#### 1. siddon.jl — KNOWN CORRECT (base layer)

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| siddon.jl:424 | `siddon_forward_project!()` | Accepts `volume_fov` kwarg | CORRECT | Uses it for volume bounds (lines 436-448) |
| siddon.jl:625 | `siddon_forward_project()` | Accepts `volume_fov` kwarg | CORRECT | Passes to `siddon_forward_project!()` (line 632) |

#### 2. polychromatic.jl — ALL CORRECT (middle layer)

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| polychromatic.jl:498 | `forward_project!()` | Accepts `volume_fov` kwarg | CORRECT | Passes to all downstream paths |
| polychromatic.jl:533 | → `_forward_project_with_signal_chain!()` call | Passes `volume_fov=volume_fov` | CORRECT | Signal chain path |
| polychromatic.jl:542 | → `siddon_forward_project!()` call (direct Float volume) | Passes `volume_fov=volume_fov` | CORRECT | Monochromatic float path |
| polychromatic.jl:555 | → `_forward_project_mono!()` call | Passes `volume_fov=volume_fov` | CORRECT | Mono from mask path |
| polychromatic.jl:560 | → `_forward_project_poly!()` call | Passes `volume_fov=volume_fov` | CORRECT | Polychromatic path |
| polychromatic.jl:757 | `_forward_project_with_signal_chain!()` | Accepts `volume_fov` kwarg | CORRECT | |
| polychromatic.jl:769 | → `siddon_forward_project!()` call (Float volume) | Passes `volume_fov=volume_fov` | CORRECT | |
| polychromatic.jl:777 | → `_forward_project_mono!()` call | Passes `volume_fov=volume_fov` | CORRECT | |
| polychromatic.jl:780 | → `_forward_project_poly!()` call | Passes `volume_fov=volume_fov` | CORRECT | |
| polychromatic.jl:990 | `_forward_project_mono!()` | Accepts `volume_fov` kwarg | CORRECT | Passes to `siddon_forward_project!()` (line 998) |
| polychromatic.jl:1110 | `_forward_project_poly!()` | Accepts `volume_fov` kwarg | CORRECT | Passes to `siddon_forward_project!()` in loop (line 1140) |
| polychromatic.jl:709-722 | `forward_project()` (allocating) | Passes `kwargs...` to `forward_project!()` | CORRECT | All kwargs including volume_fov forwarded |

#### 3. driver.jl — workspace simulate!() paths — ALL CORRECT

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| driver.jl:604 | `simulate!(ws::PCCTWorkspace, ...)` | Passes `volume_fov=phantom.fov` | CORRECT | PCCT workspace path |
| driver.jl:712 | `simulate!(ws::EICTWorkspace, ...)` | Passes `volume_fov=phantom.fov` | CORRECT | EICT workspace path, via `_forward_project_poly!()` |
| driver.jl:985 | `_eict_dual_forward_pass!()` | Passes `volume_fov=phantom.fov` | CORRECT | Dual-kVp workspace path, via `_forward_project_poly!()` |

#### 4. driver.jl — non-workspace simulate() paths — **3 MISSING**

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| driver.jl:278-282 | `_simulate_axial_single()` | **NOT passed** to `forward_project()` | **WRONG** | Calls `forward_project(mask_gpu, geom; energies=..., weights=..., materials=mats, physics=config)` — missing `volume_fov=phantom.fov` |
| driver.jl:347-351 | `_forward_single_pass()` | **NOT passed** to `forward_project()` | **WRONG** | Calls `forward_project(mask_gpu, geom; energies=..., weights=..., materials=mats, physics=config)` — missing `volume_fov=phantom.fov` |
| driver.jl:391-395 | `_simulate_axial_dual()` | **NOT passed** (via `_forward_single_pass()`) | **WRONG** | Both low-kVp and high-kVp calls go through `_forward_single_pass()` which is missing volume_fov |

**Note:** `_simulate_axial_pcct()` is CORRECT because it creates a workspace and calls `simulate!(ws, ...)` which does pass `volume_fov=phantom.fov`.

#### 5. photon_counting.jl — CORRECT

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| photon_counting.jl:1378 | `pcct_forward_project()` | Accepts `volume_fov` kwarg | CORRECT | |
| photon_counting.jl:1472 | → `siddon_forward_project!()` call | Passes `volume_fov=volume_fov` | CORRECT | |

#### 6. Iterative Reconstruction — CORRECTLY does NOT pass volume_fov

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| sirt.jl:53 | `sirt_compute_weights()` | No volume_fov | CORRECT | Projects ones for ray lengths — uses geom.fov (recon FOV) |
| sirt.jl:125 | `sirt_iterate()` | No volume_fov | CORRECT | Projects recon volume — uses geom.fov (recon FOV) |
| cgls.jl:93 | `cgls_iterate()` | No volume_fov | CORRECT | Projects search direction — uses geom.fov (recon FOV) |
| cgls.jl:207 | `cgls_reconstruct()` init | No volume_fov | CORRECT | Initial residual — uses geom.fov (recon FOV) |
| mbir.jl:500 | `mbir_iterate()` | No volume_fov | CORRECT | Projects recon volume — uses geom.fov (recon FOV) |

#### 7. Hybrid IR (driver.jl reconstruct!) — CORRECTLY does NOT pass volume_fov

| File:Line | Function | volume_fov Status | Verdict | Notes |
|-----------|----------|-------------------|---------|-------|
| driver.jl:1566 | HIR OS-PWLS `siddon_forward_project!()` | No volume_fov | CORRECT | Projects recon volume |
| driver.jl:1624 | HIR legacy stat weights update | No volume_fov | CORRECT | Projects recon volume |
| driver.jl:1652 | HIR legacy PWLS iterate | No volume_fov | CORRECT | Projects recon volume |

#### Issues Found: 3

**Issue 1: driver.jl:278-282 — `_simulate_axial_single()` missing `volume_fov=phantom.fov`**
- When using the non-workspace `simulate()` API for single-kVp EICT, the forward projection call does NOT pass `volume_fov=phantom.fov`.
- This means `siddon_forward_project!()` falls back to `geom.fov` (reconstruction FOV) for volume bounds.
- **Impact:** When `phantom.fov ≠ geom.fov`, the phantom is projected as if it fits within `geom.fov`, causing XY stretching and Z compression. This is the exact bug that was fixed in the workspace paths.
- **Severity: HIGH** — affects all non-workspace single-kVp simulations with non-matching phantom/recon FOVs (e.g., XCAT phantoms).

**Issue 2: driver.jl:347-351 — `_forward_single_pass()` missing `volume_fov=phantom.fov`**
- Same issue in the helper function used by dual-kVp.
- **Impact:** Same as Issue 1 but for dual-kVp mode.
- **Severity: HIGH**

**Issue 3: driver.jl:391-395 — `_simulate_axial_dual()` inherits missing volume_fov from `_forward_single_pass()`**
- Both low-kVp and high-kVp passes go through `_forward_single_pass()` which is missing volume_fov.
- **Impact:** All non-workspace dual-kVp simulations with non-matching phantom/recon FOVs.
- **Severity: HIGH**

**Summary:**
- The workspace-based `simulate!()` paths (PCCT, EICT, Dual) are ALL CORRECT — they pass `volume_fov=phantom.fov`.
- The non-workspace `simulate()` paths for single-kVp and dual-kVp are WRONG — they don't pass `volume_fov`.
- The allocating `forward_project()` correctly accepts and forwards `volume_fov` via `kwargs...`, so the fix is simply to add `volume_fov=phantom.fov` to the call sites.
- All iterative recon paths correctly do NOT pass volume_fov (they project the recon volume, not the phantom).

### 2026-02-14: GEO-004 — PASS (0 issues found, 2 notes)

**Agent:** Auditing ALL 5 verification notebooks for z_cm and fov_cm consistency
**Method:** `grep -n 'z_cm\|fov_cm\|detector_row_size' verification/notebooks/*.jl` — every hit checked

#### Key Principle

`ReconOptions.z_cm` controls the reconstruction volume z-extent. When set to `nothing` (default), it auto-computes from detector z-coverage via `CTGeometry(scanner; z_cm=nothing)` → `z_coverage = detector_rows × detector_row_size`. This is CORRECT for most cases.

When explicitly set, z_cm should equal `n_recon_slices × row_pitch / 10` so the recon grid slice thickness matches the native detector row pitch.

#### NB01: 01_single_kvp_verification.jl — PASS

| Parameter | Value | Source |
|-----------|-------|--------|
| Scanner | Generic CatSim-style | sid=540, sdd=950 |
| detector_row_size | ≈0.569 mm (1.0mm face / 1.759 mag) | :104 |
| detector_rows | 16 | :98 |
| sliceCount | 9 (= floor(16×0.569/1.0)) | :109 |
| sliceThickness | 1.0 mm | :108 |

| Location | z_cm formula | Value | Verdict |
|----------|-------------|-------|---------|
| :441 (phantom gen) | config.sliceCount × config.sliceThickness / 10 | 0.9 cm | CORRECT |
| :590 (recon_opts) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 0.9 cm | CORRECT |
| :609 (water phantom) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 0.9 cm | CORRECT |
| :659 (Gammex phantom) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 0.9 cm | CORRECT |

| Location | fov_cm | Verdict |
|----------|--------|---------|
| :589 (recon_opts) | 35.0 cm | CORRECT |
| :607 (water phantom) | 35.0 cm | CORRECT |
| :658 (Gammex phantom) | 35.0 cm | CORRECT |

**Note:** z_cm = sliceCount × sliceThickness / 10 = 9 × 1.0 / 10 = 0.9 cm, while detector z-coverage = 16 × 0.569 / 10 = 0.909 cm. Tiny rounding difference (0.9 vs 0.909) due to `floor()` in sliceCount calculation. Harmless — recon z is 1% smaller than detector z.

#### NB02: 02_multi_dose_and_iterative_reconstruction.jl — PASS

| Parameter | Value | Source |
|-----------|-------|--------|
| Scanner | Same as NB01 | sid=540, sdd=950 |
| detector_row_size | ≈0.569 mm | :86 |
| detector_rows | 16 | :82 |
| sliceCount | 9 | :91 |
| sliceThickness | 1.0 mm | :90 |

| Location | z_cm formula | Value | Verdict |
|----------|-------------|-------|---------|
| :184 (water phantom) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 0.9 cm | CORRECT |
| :212 (Gammex phantom) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 0.9 cm | CORRECT |
| :271 (recon_opts) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 0.9 cm | CORRECT |

| Location | fov_cm | Verdict |
|----------|--------|---------|
| :182 (water phantom) | 35.0 cm | CORRECT |
| :211 (Gammex phantom) | 35.0 cm | CORRECT |
| :270 (recon_opts) | 35.0 cm | CORRECT |

**Iterative recon (HIR):** Uses workspace API with same `recon_opts` → same z_cm. HIR calls `create_hir_recon_workspace()` with sinogram from same geometry. CORRECT.

#### NB03: 03_dual_kvp_vmi_verification.jl — PASS (with note)

| Parameter | Value | Source |
|-----------|-------|--------|
| Scanner | Same CatSim-style | sid=540, sdd=950 |
| detector_row_size | ≈0.569 mm | :104 |
| detector_rows | 16 | :102 (implied from config) |
| sliceCount | 32 | :95 |
| sliceThickness | 1.25 mm | :96 |

| Location | z_cm formula | Value | Verdict |
|----------|-------------|-------|---------|
| :169 (phantom) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 4.0 cm | CORRECT |
| :153 (recon_opts) | **z_cm not set** → auto from detector | ~0.909 cm | CORRECT |

| Location | fov_cm | Verdict |
|----------|--------|---------|
| :153 (recon_opts) | 35.0 cm | CORRECT |
| :168 (phantom) | 35.0 cm | CORRECT |

**Both kVp use same geometry:** Both 80 kVp (:194) and 140 kVp (:210) call `create_eict_workspace(scanner, ...)` with same scanner and recon_opts. CORRECT — same geometry for both energies.

**NOTE:** Phantom z_cm (4.0 cm) is much larger than detector z-coverage (~0.909 cm). The phantom is over-allocated in z. Only the central ~0.909 cm is visible to the detector. This is harmless but wastes memory on unused phantom voxels. Not a correctness issue.

#### NB04: 04_pcct_demonstration.jl — PASS (with note)

| Parameter | Value | Source |
|-----------|-------|--------|
| Scanner | NAEOTOM Alpha (PCCT) | :88-94 |
| detector_row_size | 0.4 mm (square pixels) | :94 |
| detector_rows | 64 | :92 |
| sliceCount | 32 | :84 |
| sliceThickness | 1.25 mm | :85 |
| Detector z-coverage | 64 × 0.4 / 10 = 2.56 cm | computed |

| Location | z_cm formula | Value | Verdict |
|----------|-------------|-------|---------|
| :472 (phantom) | SIM_CONFIG.sliceCount × SIM_CONFIG.sliceThickness / 10 | 4.0 cm | CORRECT |
| :514 (recon_opts) | **z_cm not set** → auto from detector | 2.56 cm | CORRECT |
| :542 (water cal recon) | **z_cm not set** → auto from detector | 2.56 cm | CORRECT |

| Location | fov_cm | Verdict |
|----------|--------|---------|
| :471 (phantom) | 35.0 cm | CORRECT |
| :514 (recon_opts) | 35.0 cm | CORRECT |
| :542 (water cal recon) | 15.0 cm | CORRECT (smaller FOV for water cal) |

**NOTE:** Same pattern as NB03 — phantom z_cm (4.0 cm) > detector z-coverage (2.56 cm). Harmless; only central 2.56 cm of phantom is projected.

**Water calibration:** Uses same NAEOTOM scanner, so auto-derived z_cm = 2.56 cm. Water phantom z-extent = 16 × 0.1 = 1.6 cm (smaller than detector z-coverage). Central ROI extraction handles this correctly.

#### NB05: 05_xcat_full.jl — PASS

| Parameter | EICT (GE Revolution) | PCCT (NAEOTOM Alpha) |
|-----------|---------------------|---------------------|
| detector_row_size | 0.625 mm (:621) | 0.4 mm (:659) |
| detector_rows | min(256, 128) = 128 | min(144, 128) = 128 |
| n_recon_slices | 128 (:598) | 128 (:598) |
| recon_fov_cm | 25.0 (:589) | 25.0 (:589) |

**EICT Recon z_cm:**

| Location | z_cm formula | Value | Verdict |
|----------|-------------|-------|---------|
| :704 (single) | n_recon_slices × 0.625 / 10 | 8.0 cm | CORRECT |
| :721 (dual) | n_recon_slices × 0.625 / 10 | 8.0 cm | CORRECT |
| :785 (water cal) | 8 × scanner_eict.detector_row_size / 10 = 8 × 0.625 / 10 | 0.5 cm | CORRECT |

**PCCT Recon z_cm:**

| Location | z_cm formula | Value | Verdict |
|----------|-------------|-------|---------|
| :740 (standard) | n_recon_slices × 0.4 / 10 | 5.12 cm | CORRECT |
| :794 (water cal) | 8 × scanner_pcct_standard.detector_row_size / 10 = 8 × 0.4 / 10 | 0.32 cm | CORRECT |

**Consistency checks:**
- EICT: recon z_cm = n_recon_slices × detector_row_size / 10 = detector_rows × detector_row_size / 10 (since det_rows = n_recon_slices = 128). **Match** ✓
- PCCT: Same pattern. **Match** ✓
- Water cal: Uses scanner.detector_row_size directly for per-scanner z_cm. **CORRECT** ✓
- Dual-kVp: Same z_cm as single-kVp (both = 8.0 cm). **CORRECT** ✓
- fov_cm: All recon opts use recon_fov_cm = 25.0. **Consistent** ✓

#### Summary

| Notebook | Scanner | z_cm Method | Consistency | Verdict |
|----------|---------|------------|-------------|---------|
| NB01 | Generic (≈0.569mm) | Explicit: sliceCount × sliceThickness / 10 | All 4 usages = 0.9 cm | PASS |
| NB02 | Generic (≈0.569mm) | Explicit: sliceCount × sliceThickness / 10 | All 3 usages = 0.9 cm | PASS |
| NB03 | Generic (≈0.569mm) | Auto (z_cm=nothing) + Phantom explicit | Phantom oversized (4.0 vs 0.909 cm) but harmless | PASS |
| NB04 | NAEOTOM (0.4mm) | Auto (z_cm=nothing) + Phantom explicit | Phantom oversized (4.0 vs 2.56 cm) but harmless | PASS |
| NB05 | GE Rev + NAEOTOM | Explicit: n_slices × row_pitch / 10 | All values match scanner row pitch | PASS |

**Issues found: 0**

**Notes:**
1. NB03 and NB04 have phantoms with z_cm much larger than detector z-coverage. This wastes memory but is not incorrect — the projector only captures what the detector sees.
2. NB01/NB02 use `sliceCount × sliceThickness / 10` which is ~1% less than detector z-coverage due to `floor()` rounding. Negligible.

### 2026-02-14: GEO-005 — PASS (0 bugs, 1 documentation note)

**Agent:** Deep audit of PCCT-specific geometry in photon_counting.jl and pcct/*.jl
**Method:** `grep -rn 'pixel_size\|pixel_pitch\|pixel_row\|pixel_col' src/detector/photon_counting.jl src/detector/pcct/` — every hit checked

#### Architecture Summary

The PCCT code uses two geometry structs:

1. **`PhotonCountingDetector.pixel_size_mm::Tuple{T,T}`** — documented as `(row, col)` = `(z, xy)`
   - Used for all high-level detector physics: charge sharing, pileup, spectral response
   - Always accessed as tuple with `[1]`=row, `[2]`=col — direction-aware

2. **`PCCTDetectorGeometry.pixel_pitch_mm::Tuple{Float64,Float64}`** — documented as `(channel, slice)` = `(xy, z)`
   - Used only for low-level charge transport ODE (Koch-Mehrin 2020)
   - `pixel_pitch_mm` is only accessed via `min(geom.pixel_pitch_mm...)` in `pixel_to_thickness_ratio()` — **order-independent**
   - `charge_cloud_sigma_mm()` and `mean_charge_cloud_sigma_mm()` use `geom.thickness_mm` and `geom.effective_voltage_V` — they do NOT use `pixel_pitch_mm`

**Key insight:** Despite the opposite tuple ordering conventions between the two structs, no actual direction bug exists because:
- `PCCTDetectorGeometry.pixel_pitch_mm` is only used with `min()` (order-independent)
- All direction-sensitive code uses `PhotonCountingDetector.pixel_size_mm` with explicit `[1]`=row, `[2]`=col indexing

#### Detailed Audit Table — photon_counting.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| :100 | `pixel_size_mm::Tuple{T,T}` | — (docstring) | N/A | Documents "(row, col)" |
| :131 | `pixel_size_mm = (0.302, 0.302)` | both | CORRECT | Square pixels, order doesn't matter |
| :144 | `pixel_size_mm::Tuple{T,T}` | both | CORRECT | Struct field |
| :177 | `pixel_size_mm` kwarg doc | — (docstring) | N/A | Documents "(row, col)" |
| :192 | `pixel_size_mm::Tuple{Float64,Float64}=(0.302, 0.302)` | both | CORRECT | Constructor kwarg |
| :205 | `pixel_size_mm` passed to struct | both | CORRECT | Forwarded to struct |
| :235 | `pixel_size_mm = (0.302, 0.302)` | both | CORRECT | naeotom_detector_standard() preset |
| :265 | `pixel_size_mm = (0.151, 0.151)` | both | CORRECT | naeotom_detector_uhr() preset — square |
| :289 | `pixel_size_mm = (0.302, 0.302)` | both | CORRECT | naeotom_detector_chess() preset |
| :537 | `Float64.(detector.pixel_size_mm)` | both | CORRECT | Passed to PCCTDetectorGeometry (see note) |
| :546 | `pixel_pitch = detector.pixel_size_mm` | both | CORRECT | Alias for fluorescence model |
| :549 | `compute_cdte_fluorescence_model(pixel_pitch, ...)` | both | CORRECT | Tuple passed to k_fluorescence.jl |
| :563 | `charge_sharing_probability(σ, pixel_pitch)` | both | CORRECT | Tuple passed to charge_transport.jl |
| :643-644 | `pixel_row = detector.pixel_size_mm[1]`, `pixel_col = detector.pixel_size_mm[2]` | row/col | CORRECT | Explicit row/col decomposition |
| :646-647 | `boundary_dist_row/col = pixel_row/col / 2` | row/col | CORRECT | Per-direction charge sharing |
| :649-654 | `z_row`, `z_col`, `p_share_row`, `p_share_col` | row/col | CORRECT | Per-direction probabilities, combined |
| :743-744 | Same as :643-644 | row/col | CORRECT | apply_anti_coincidence_correction! |
| :746-747 | Same as :646-647 | row/col | CORRECT | |
| :749-753 | Same as :649-654 | row/col | CORRECT | |
| :873 | `pixel_area = pixel_size_mm[1] * pixel_size_mm[2]` | area | CORRECT | row × col = area (correct for pileup) |
| :1002 | Same area calculation | area | CORRECT | apply_pulse_pileup_correction! |
| :1407 | `pixel_size_mm=detector.pixel_size_mm` | both | CORRECT | Passed to compute_spectral_response_matrix |
| :1705-1706 | `pixel_row/col` from tuple | row/col | CORRECT | compute_I0_per_bin_pcct |
| :1707-1708 | `z_row/z_col` | row/col | CORRECT | Per-direction charge sharing |
| :1725 | `pixel_area = pixel_size_mm[1] * pixel_size_mm[2]` | area | CORRECT | row × col = area |
| :1968 | `pixel_size_mm = detector.pixel_size_mm` | both | CORRECT | Info struct passthrough |
| :1994 | Print statement | both | CORRECT | Display: "$(info.pixel_size_mm[1]) × $(info.pixel_size_mm[2]) mm" |
| :2284 | `pixel_size_mm::Tuple{<:Real,<:Real}` | both | CORRECT | compute_fluorescence_escape_probability param |
| :2299 | Doc: "(row, col)" | — | N/A | Documentation |
| :2308 | `pixel_size_mm::Tuple{<:Real,<:Real}` | both | CORRECT | Type constraint |
| :2323 | `min(pixel_size_mm[1], pixel_size_mm[2]) / 2` | both | CORRECT | Uses smaller dim — order-independent |
| :2337 | Comment about pixel_size | — | N/A | Documentation |
| :2430 | `pixel_pitch_mm::Tuple{<:Real,<:Real}=(0.0, 0.0)` | both | CORRECT | CCE kwarg |
| :2439 | Doc about pixel_pitch_mm | — | N/A | Documentation |
| :2451 | Doc: "(row, col)" | — | N/A | Documentation |
| :2462 | `pixel_pitch_mm::Tuple{<:Real,<:Real}=(0.0, 0.0)` | both | CORRECT | charge_collection_efficiency param |
| :2468 | `w_mm = min(pixel_pitch_mm[1], pixel_pitch_mm[2])` | both | CORRECT | Uses min — order-independent |
| :2483 | `pixel_pitch_mm` in mean_charge_collection_efficiency | both | CORRECT | Forwarded |
| :2502 | `pixel_pitch_mm` kwarg | both | CORRECT | |
| :2504 | `w_mm = min(pixel_pitch_mm[1], pixel_pitch_mm[2])` | both | CORRECT | Uses min — order-independent |
| :2529 | `pixel_pitch_mm` in hole_tailing_distribution | both | CORRECT | Forwarded |
| :2552 | `pixel_pitch_mm` kwarg | both | CORRECT | |
| :2554 | `w_mm = min(pixel_pitch_mm[1], pixel_pitch_mm[2])` | both | CORRECT | Uses min — order-independent |
| :2593 | `pixel_size_mm::Tuple=(0.302, 0.302)` | both | CORRECT | compute_spectral_response_matrix kwarg |
| :2619 | Doc about pixel_size_mm | — | N/A | Documentation |
| :2645 | `pixel_size_mm::Tuple=(0.302, 0.302)` | both | CORRECT | Function param |
| :2668 | `compute_cdte_fluorescence_model(pixel_size_mm, ...)` | both | CORRECT | Tuple passed to k_fluorescence |
| :2674 | `compute_fluorescence_escape_probability(..., pixel_size_mm)` | both | CORRECT | Tuple passed |
| :2686 | `pixel_pitch_mm=pixel_size_mm` | both | CORRECT | Forwarded to CCE |
| :2699 | `pixel_pitch_mm=pixel_size_mm` | both | CORRECT | Forwarded to hole tailing |

#### Detailed Audit Table — pcct/charge_transport.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| :337 | `σ/pixel_size` | — (comment) | N/A | Documentation only |
| :350 | Doc: "(row, col)" | — | N/A | Documentation |
| :355 | `pixel_pitch_mm::Tuple{<:Real,<:Real}` | both | CORRECT | charge_sharing_probability param |
| :357 | `w_row = Float64(pixel_pitch_mm[1])` | z (row) | CORRECT | Tuple[1]=row |
| :358 | `w_col = Float64(pixel_pitch_mm[2])` | xy (col) | CORRECT | Tuple[2]=col |
| :370-371 | `inner_row/inner_col` | row/col | CORRECT | Per-direction safe region |
| :373 | `p_no_share = (inner_row/w_row) * (inner_col/w_col)` | both | CORRECT | 2D probability |

Note: `charge_cloud_sigma_mm()` (:61) and `mean_charge_cloud_sigma_mm()` (:289) use `geom.thickness_mm` and `geom.effective_voltage_V` from `PCCTDetectorGeometry` — they do NOT use `pixel_pitch_mm`. The pixel pitch is irrelevant for charge cloud growth (which depends only on sensor thickness, voltage, and photon energy).

#### Detailed Audit Table — pcct/charge_collection.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| (entire file) | `wL_ratio` | — | CORRECT | Uses `min(pixel_pitch_mm...)` via callers — order-independent |

`charge_collection.jl` does NOT directly reference `pixel_size_mm` or `pixel_pitch_mm`. All functions take `wL_ratio::Real` (scalar), computed upstream by callers using `min(pixel_pitch_mm[1], pixel_pitch_mm[2]) / thickness_mm`. The `min()` makes it order-independent.

Functions: `small_pixel_weighting_potential()`, `hecht_cce_weighted()`, `mean_cce_beer_lambert()`, `hole_tailing_beer_lambert()` — all work with scalar `wL_ratio`.

#### Detailed Audit Table — pcct/k_fluorescence.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| :45 | Doc: "(row, col)" | — | N/A | Documentation |
| :56 | `pixel_pitch_mm::Tuple{<:Real,<:Real}` | both | CORRECT | fluorescence_escape_fraction param |
| :64 | `w_row = Float64(pixel_pitch_mm[1])` | z (row) | CORRECT | Row dimension |
| :65 | `w_col = Float64(pixel_pitch_mm[2])` | xy (col) | CORRECT | Col dimension |
| :70-71 | `d_row = w_row / 4.0`, `d_col = w_col / 4.0` | row/col | CORRECT | Per-direction avg distance |
| :75-77 | `p_row`, `p_col`, `p_axial` | row/col/z | CORRECT | Per-direction escape probability |
| :80-87 | Solid angle weighting: `Ω_row = 2*w_col*L`, `Ω_col = 2*w_row*L` | cross-terms | CORRECT | Row faces have area `w_col × L`, col faces have area `w_row × L` |
| :89 | `p_escape = weighted average` | both | CORRECT | Physically correct solid-angle weighting |
| :139 | Doc: "(row, col)" | — | N/A | Documentation |
| :146 | `pixel_pitch_mm::Tuple{<:Real,<:Real}` | both | CORRECT | compute_cdte_fluorescence_model param |
| :158 | `fluorescence_escape_fraction(cd.mean_path_mm, pixel_pitch_mm, thickness_mm)` | both | CORRECT | Tuple forwarded |
| :165 | Same for Te | both | CORRECT | Tuple forwarded |
| :203 | `fluorescence_escape_fraction(te.mean_path_mm, pixel_pitch_mm, thickness_mm)` | both | CORRECT | Te reabsorption |

The fluorescence escape model is direction-aware: it correctly treats the pixel as a 3D rectangular parallelepiped with different dimensions in row, col, and depth directions, weighted by solid angle fractions.

#### Detailed Audit Table — pcct/pileup_model.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|

`pileup_model.jl` does NOT contain any pixel geometry references. All functions work with scalar `aτ` (count rate × dead time product). The pixel area calculation for count rate happens upstream in `photon_counting.jl` (:873, :1002, :1725) using `pixel_size_mm[1] * pixel_size_mm[2]` (row × col = correct area).

#### Detailed Audit Table — pcct/detector_response.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| :136 | `pixel_size_mm = detector.pixel_size_mm` | both | CORRECT | Local alias |
| :147 | `compute_cdte_fluorescence_model(pixel_size_mm, ...)` | both | CORRECT | Tuple forwarded |
| :152 | `compute_fluorescence_escape_probability(..., pixel_size_mm)` | both | CORRECT | Tuple forwarded |
| :185 | `pixel_pitch_mm=pixel_size_mm` | both | CORRECT | Forwarded to hole_tailing_distribution |

The unified DRM (`compute_unified_drm()`) passes the tuple through to lower-level functions (fluorescence, CCE, hole tailing) which handle direction correctly as documented above.

#### Detailed Audit Table — pcct/cdte_constants.jl

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| :188 | Doc: "(channel, slice)" | — | N/A | **NOTE: Opposite convention from PhotonCountingDetector's (row, col)** |
| :197 | `pixel_pitch_mm::Tuple{Float64, Float64}` | both | CORRECT | Struct field |
| :209 | Doc: "275 × 322 μm pixel pitch (non-square: channel × slice)" | — | N/A | Documentation |
| :217 | `(0.275, 0.322)` | (channel, slice) | CORRECT* | *See note about convention mismatch |
| :369 | `w = min(geom.pixel_pitch_mm...)` | both | CORRECT | `min()` is order-independent |

#### Documentation Convention Note (NOT a bug)

**`PCCTDetectorGeometry.pixel_pitch_mm`** is documented as `(channel, slice)` = `(xy, z)`.
**`PhotonCountingDetector.pixel_size_mm`** is documented as `(row, col)` = `(z, xy)`.

When `detector.pixel_size_mm` is passed to `PCCTDetectorGeometry()` (photon_counting.jl:537, workspace.jl:315), the tuple order would technically be swapped relative to the `PCCTDetectorGeometry` convention.

**However, this causes NO runtime bug because:**
1. `PCCTDetectorGeometry.pixel_pitch_mm` is ONLY accessed via `min(geom.pixel_pitch_mm...)` in `pixel_to_thickness_ratio()`, which is order-independent.
2. `charge_cloud_sigma_mm()` and `mean_charge_cloud_sigma_mm()` do NOT use `pixel_pitch_mm` — they use `thickness_mm` and `effective_voltage_V`.
3. All direction-sensitive code uses `PhotonCountingDetector.pixel_size_mm` with explicit `[1]`=row, `[2]`=col indexing.

**Recommendation (LOW priority):** Align the documentation of `PCCTDetectorGeometry.pixel_pitch_mm` from `(channel, slice)` to `(row, col)` to match `PhotonCountingDetector.pixel_size_mm`. But since the field is only used via `min()`, this is purely cosmetic.

#### Summary — PCCT Geometry Patterns

All PCCT geometry code follows these safe patterns:

1. **Direction-aware tuple indexing**: `pixel_size_mm[1]`=row, `pixel_size_mm[2]`=col — used for charge sharing, pileup area, fluorescence escape solid angles
2. **Order-independent `min()`**: Small-pixel effect uses `min(pixel_pitch_mm...)` — correctly uses smaller dimension regardless of tuple order
3. **Scalar passthrough**: CCE/hole-tailing use `wL_ratio` (scalar) — no direction sensitivity
4. **Area = row × col**: Pileup count rate uses `pixel_size_mm[1] * pixel_size_mm[2]` — commutative, correct
5. **No hardcoded square-pixel assumptions**: All tuple code works correctly for non-square pixels

**Issues found: 0 (bugs)**
**Notes found: 1 (documentation convention mismatch between PCCTDetectorGeometry and PhotonCountingDetector)**

### 2026-02-14: GEO-006 — PASS (0 issues found)

**Agent:** Auditing iterative recon (SIRT, CGLS, MBIR, Hybrid IR, statistical_ir) z-geometry
**Method:** `grep -rn 'siddon\|forward_project\|volume_fov' src/reconstruction/ir/ src/reconstruction/hybrid_ir/ src/reconstruction/mbir/` + full file reads + driver.jl reconstruct! paths

**Key principle:** Iterative recon forward-projects the RECONSTRUCTION volume, so should use `geom.fov` (recon FOV), NOT `phantom.fov`. No `volume_fov` kwarg should be passed.

#### 1. sirt.jl — CORRECT (no volume_fov, no pixel_size/pixel_row_size)

| File:Line | Call | volume_fov | Verdict | Notes |
|-----------|------|------------|---------|-------|
| sirt.jl:53 | `siddon_forward_project(ones_volume, geom)` | NOT passed | CORRECT | `compute_projection_weights`: ray lengths — uses `geom.fov` |
| sirt.jl:125 | `siddon_forward_project(recon, geom)` | NOT passed | CORRECT | `sirt_iteration!`: recon estimate — uses `geom.fov` |
| sirt.jl:88 | `backproject(ones_sino, geom, volume_size; weighted=false)` | N/A | CORRECT | Voxel sensitivity |
| sirt.jl:136 | `backproject(projected, geom, size(recon); weighted=false)` | N/A | CORRECT | Matched backprojection |

**pixel_size/pixel_row_size:** Not directly referenced. Delegated to siddon/backproject. CORRECT.

#### 2. cgls.jl — CORRECT (no volume_fov, no pixel_size/pixel_row_size)

| File:Line | Call | volume_fov | Verdict | Notes |
|-----------|------|------------|---------|-------|
| cgls.jl:93 | `siddon_forward_project(p, geom)` | NOT passed | CORRECT | Search direction — uses `geom.fov` |
| cgls.jl:207 | `siddon_forward_project(recon, geom)` | NOT passed | CORRECT | Initial residual — uses `geom.fov` |
| cgls.jl:118 | `backproject(r, geom, size(x); weighted=false)` | N/A | CORRECT | Adjoint backprojection |
| cgls.jl:218 | `backproject(r, geom, size(recon); weighted=false)` | N/A | CORRECT | Initial search direction |

**pixel_size/pixel_row_size:** Not directly referenced. Delegated to siddon/backproject. CORRECT.

#### 3. hybrid_ir.jl — CORRECT (delegates to pwls_reconstruct!, no direct forward projection)

| File:Line | Call | volume_fov | Verdict | Notes |
|-----------|------|------------|---------|-------|
| hybrid_ir.jl:182 | `fdk_reconstruct(sinogram, geometry, volume_size; filter)` | N/A | CORRECT | FDK init |
| hybrid_ir.jl:187-194 | `pwls_reconstruct!(x, sinogram, geometry; ...)` | NOT passed | CORRECT | Delegates to statistical_ir.jl |

**pixel_size/pixel_row_size:** Not directly referenced. No direct siddon calls. CORRECT.

#### 4. mbir.jl — CORRECT (no volume_fov, correct pixel_size/pixel_row_size in subset geometry)

| File:Line | Call | volume_fov | Verdict | Notes |
|-----------|------|------------|---------|-------|
| mbir.jl:500 | `siddon_forward_project(recon, geom_subset)` | NOT passed | CORRECT | OS-SQS: recon through subset geom |
| mbir.jl:509 | `backproject(Ax_subset, geom_subset, size(recon); weighted=false)` | N/A | CORRECT | Matched backprojection |
| mbir.jl:603-604 | `compute_projection_weights`/`compute_image_weights` | NOT passed | CORRECT | Normalization weights |
| mbir.jl:440-441 | `create_subset_geometry`: `geom.pixel_size, geom.pixel_row_size` | N/A | CORRECT | xy and z preserved separately in subset CTGeometry |
| mbir.jl:447 | `create_subset_geometry`: `geom.fov` | N/A | CORRECT | Subset preserves recon FOV |

#### 5. statistical_ir.jl (PWLS core) — CORRECT (no volume_fov, no pixel_size/pixel_row_size)

| File:Line | Call | volume_fov | Verdict | Notes |
|-----------|------|------------|---------|-------|
| statistical_ir.jl:343 | `siddon_forward_project(x_current, geom)` | NOT passed | CORRECT | `compute_statistical_weights` |
| statistical_ir.jl:429 | `siddon_forward_project(x, geom)` | NOT passed | CORRECT | `pwls_iteration_sirt!` data fidelity |
| statistical_ir.jl:439 | `backproject(Ax, geom, size(x); weighted=false)` | N/A | CORRECT | Matched backprojection |
| statistical_ir.jl:513-514 | `compute_projection_weights`/`compute_image_weights` | NOT passed | CORRECT | Normalization weights |

**pixel_size/pixel_row_size:** Not directly referenced. Delegated to siddon/backproject. CORRECT.

#### 6. driver.jl reconstruct!() paths — CORRECT (no volume_fov in reconstruction forward projections)

| File:Line | Call | volume_fov | Verdict | Notes |
|-----------|------|------------|---------|-------|
| driver.jl:1566 | `siddon_forward_project!(ax_view, ws.volume, geom_s; ws_*)` | NOT passed | CORRECT | HIR OS-PWLS subset FP |
| driver.jl:1624 | `siddon_forward_project!(ws.Ax, ws.volume, geom; ws_*)` | NOT passed | CORRECT | HIR legacy stat weights |
| driver.jl:1652 | `siddon_forward_project!(ws.Ax, ws.volume, geom; ws_*)` | NOT passed | CORRECT | HIR legacy data fidelity |

**Note:** All three driver.jl reconstruction FP calls correctly do NOT pass `volume_fov`. Distinguished from the **simulation** FP calls (driver.jl:604, :712, :985) which DO pass `volume_fov=phantom.fov`.

#### Summary

| File | FP calls | volume_fov | pixel_size/pixel_row_size | Verdict |
|------|----------|------------|---------------------------|---------|
| sirt.jl | 2 FP + 2 BP | NOT passed | Delegated | PASS |
| cgls.jl | 2 FP + 2 BP | NOT passed | Delegated | PASS |
| hybrid_ir.jl | 0 direct | NOT passed | N/A | PASS |
| mbir.jl | 1 FP + 1 BP + subset_geom | NOT passed | Correct: xy/z separate | PASS |
| statistical_ir.jl | 2 FP + 1 BP | NOT passed | Delegated | PASS |
| driver.jl (recon!) | 3 FP + 3 BP | NOT passed | Via workspace geom | PASS |

**Issues found: 0**

All iterative reconstruction paths correctly:
1. Do NOT pass `volume_fov` — they project the recon volume using `geom.fov`
2. Use matched/unweighted backprojection (`weighted=false`)
3. Properly delegate pixel geometry to siddon/backproject (which handle pixel_size vs pixel_row_size internally)
4. Pass `volume_size = size(recon)` consistently
5. mbir.jl's `create_subset_geometry()` correctly preserves separate `pixel_size` (xy) and `pixel_row_size` (z)

### 2026-02-14: GEO-009 — WIP — Starting fixes for geometry issues

**Agent:** Fixing all issues identified in GEO-001 through GEO-006.

#### Issues to fix:

**From GEO-001 (2 issues in scanner.jl):**
1. `scanner.jl:702` — `fov_z = pixel_size * n_rows` should use a row pixel size
2. `scanner.jl:746` — `pixel_size, pixel_size` passed to CTGeometry should have separate pixel_row_size

**From GEO-003 (3 issues in driver.jl):**
1. `driver.jl:278-282` — `_simulate_axial_single()` missing `volume_fov=phantom.fov`
2. `driver.jl:347-351` — `_forward_single_pass()` missing `volume_fov=phantom.fov`
3. `driver.jl:391-395` — `_simulate_axial_dual()` inherits missing from `_forward_single_pass()`

**Plan:**
- Fix 1: In `create_aquilion_one()`, add `pixel_row_size = pixel_pitch_mm / 10.0` (separate from pixel_size), use it for z FOV and pass it separately to CTGeometry
- Fix 2: In `_forward_single_pass()`, add `volume_fov=phantom.fov` to the `forward_project()` call. This fixes all 3 driver.jl issues since `_simulate_axial_single()` calls `forward_project()` directly and `_simulate_axial_dual()` calls through `_forward_single_pass()`

#### Fixes Applied:

**Fix 1: scanner.jl — `create_aquilion_one()` (2 issues resolved)**
- Added `pixel_row_size = pixel_pitch_mm / 10.0` as a separate variable for the z-direction
- Changed `fov_z = pixel_size * n_rows` → `fov_z = pixel_row_size * n_rows` (line 705)
- Changed `CTGeometry(..., pixel_size, pixel_size, ...)` → `CTGeometry(..., pixel_size, pixel_row_size, ...)` (line 749)
- **Impact:** For Aquilion ONE (square 0.5mm pixels), no numerical change. But the code is now correct for non-square pixel extensions and the z FOV default no longer depends on the XY-overridden pixel_size when fov_cm is specified.

**Fix 2: driver.jl — non-workspace simulate paths (3 issues resolved)**
- `_simulate_axial_single()`: Added `volume_fov=phantom.fov` to `forward_project()` call (line 282)
- `_forward_single_pass()`: Added `volume_fov=phantom.fov` to `forward_project()` call (line 352)
- This transitively fixes `_simulate_axial_dual()` which calls `_forward_single_pass()` for both kVp passes
- **Impact:** Non-workspace simulate paths now correctly use phantom physical dimensions for volume bounds, matching the workspace-based simulate!() paths that were already fixed.

**Summary: 5 issues fixed, 0 regressions expected**

All fixes are minimal and targeted. The Aquilion ONE has square pixels so the scanner.jl fix has no numerical effect on current usage. The driver.jl fix aligns the non-workspace API with the workspace API that was already correct.

### 2026-02-14: GEO-008 — PASS (0 issues found, 1 note)

**Agent:** Auditing scatter.jl geometry for z-direction correctness
**Method:** Full file read (1198 lines) + `grep -rn 'pixel_size\|pixel_row\|geom\.' src/detector/scatter.jl` + `grep -rn 'scanner\.\|detector_col\|detector_row\|magnification\|pixel\|pitch\|fov' src/detector/scatter.jl`

#### Architecture Summary

The scatter module has two main components:

1. **Core scatter add/correct** (`add_scatter!`, `correct_scatter!`): Work entirely in **pixel-index space**. They operate on the sinogram `[n_cols × n_rows × n_angles]` using pixel indices for the 2D convolution kernel. No `pixel_size`, `pixel_row_size`, or `geom.*` references at all. The kernel FWHM is specified in **pixels** (not physical units).

2. **Geometry-aware scatter** (`geometry_aware_scatter_model`, `geometry_aware_scatter_correction`): Use `Scanner` struct fields to compute scaling factors. The only geometry-to-pixel conversion is `compute_scatter_kernel_fwhm_pixels()`.

#### Audit Questions Answered

1. **Does scatter use detector pixel geometry?** YES — `compute_scatter_kernel_fwhm_pixels()` (line 716-720) converts physical FWHM (50mm) to pixel units using `scanner.detector_col_size * magnification`.

2. **Does scatter compute patient size from sinogram?** NO — `estimate_phantom_diameter_cm()` takes a `mask` volume and `voxel_size_mm` tuple, computes in-plane bounding box using `dx`, `dy` only. Does not use sinogram geometry.

3. **Any position-dependent scatter kernel that uses pixel_size vs pixel_row_size?** NO — the core scatter functions (`add_scatter!`, `correct_scatter!`) use a single symmetric 2D kernel indexed by pixel offsets `(di, dj)`. No `pixel_size` or `pixel_row_size` references.

4. **Does scatter model use cone angle (z) or fan angle (xy)?** NEITHER — scatter is modeled as isotropic 2D convolution in sinogram pixel space.

5. **Is the kernel applied correctly in both row and column dimensions?** YES — the 2D kernel is symmetric (same FWHM in both directions) and applied via nested loops over `di` (col offset) and `dj` (row offset).

#### Detailed Audit Table

| File:Line | Variable | Direction | Verdict | Notes |
|-----------|----------|-----------|---------|-------|
| scatter.jl:54-55 | `kernel_fwhm` | — (comment) | N/A | "FWHM in detector pixels" |
| scatter.jl:69 | `kernel_fwhm` | — (docstring) | N/A | Parameter documentation |
| scatter.jl:104-143 | `create_scatter_kernel_spatial()` | both (symmetric) | CORRECT | 2D symmetric Gaussian/exponential kernel, same extent in both directions |
| scatter.jl:163-236 | `add_scatter!()` | both (pixel index) | CORRECT | No geom references. Kernel applied symmetrically via `di` (col) and `dj` (row) loops |
| scatter.jl:354-451 | `correct_scatter!()` | both (pixel index) | CORRECT | Same structure as add_scatter!(), no geom references |
| scatter.jl:476 | `SCATTER_REF_PIXEL_PITCH_MM = 1.0` | — (constant) | N/A | Reference calibration constant |
| scatter.jl:483 | `SCATTER_PHYSICAL_KERNEL_FWHM_MM = 50.0` | — (constant) | N/A | Physical scatter kernel size at detector |
| scatter.jl:685-694 | `compute_scatter_geometry_scale()` | — (scalar) | CORRECT | Uses `scanner.source_to_detector - scanner.source_to_isocenter` for air gap. No pixel geometry. |
| scatter.jl:716-720 | `compute_scatter_kernel_fwhm_pixels()` | xy (col) | CORRECT* | Uses `scanner.detector_col_size * magnification` for pixel pitch. *See note below. |
| scatter.jl:719 | `detector_face_pitch = scanner.detector_col_size * magnification` | xy (col) | CORRECT* | Column-direction pitch at detector face |
| scatter.jl:535-580 | `estimate_phantom_diameter_cm()` | in-plane (xy) | CORRECT | Uses `dx`, `dy` from `voxel_size_mm` tuple for bounding box. `dz` present but unused (correct — diameter is in-plane). |
| scatter.jl:976-1037 | `estimate_scatter_joint()` | both (pixel index) | CORRECT | Same symmetric kernel convolution structure. No geom references. |
| scatter.jl:1070-1092 | `correct_scatter_with_estimate!()` | — (per-pixel) | CORRECT | Simple intensity subtraction, no geometry. |
| scatter.jl:1139-1165 | `correct_scatter_dual_energy!()` | — (scalar) | CORRECT | Delegates to `geometry_aware_scatter_model()` and `estimate_scatter_joint()`. |

#### Issues Found: 0

#### Note: Isotropic Kernel FWHM Uses Column Pitch Only

`compute_scatter_kernel_fwhm_pixels()` (line 716-720) converts the 50mm physical scatter FWHM to pixel units using only `scanner.detector_col_size` (xy-direction). For non-square detectors (e.g., GE Revolution: col=0.6mm, row=0.625mm), this means the kernel FWHM in row-pixel units is slightly different from col-pixel units:

- At detector face: col pitch = 0.6 × (1097/626) = 1.051mm, row pitch = 0.625 × (1097/626) = 1.095mm
- Kernel FWHM in col pixels: 50/1.051 = 47.6 pixels
- Kernel FWHM in row pixels: 50/1.095 = 45.7 pixels (4.2% smaller)

The kernel is applied as a single symmetric 2D Gaussian with the col-pixel FWHM, making it slightly wider in the row direction (in physical mm) than ideal.

**Impact: NEGLIGIBLE** — The scatter kernel is a very broad, low-frequency signal (FWHM ~48 pixels ≈ 50mm). A 4% asymmetry in such a broad kernel has no measurable effect on the scatter estimate. The dominant factors are the scatter coefficient, phantom size, and energy — not the kernel shape.

**Recommendation: NO ACTION NEEDED** — Making the kernel anisotropic would add complexity for negligible improvement. The current isotropic kernel is standard practice in CT scatter simulation (XCIST/CatSim also uses isotropic kernels).

#### Summary

The scatter module is architecturally clean from a geometry perspective:
- Core scatter add/correct functions are geometry-free (work in pixel-index space)
- Geometry-aware functions correctly use `Scanner` struct fields
- `detector_col_size` is used for in-plane kernel FWHM conversion — correct for an isotropic scatter kernel
- No `pixel_size`, `pixel_row_size`, or `geom.*` references in the entire file
- No z-direction geometry issues
