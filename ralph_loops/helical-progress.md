# BasisSimulator.jl Helical CT — Discovery Progress Log

Each iteration logs what was researched, what was found, and what gaps remain.

---

## Iteration 1: HELI-000 DISCOVERY — Current Geometry and Projection Audit

**Date:** 2026-03-17
**Phase:** Discovery
**Topic:** HELI-000 (Current Geometry and Projection Audit)

### Summary

Performed a line-by-line audit of every file that touches source/detector positions, Z-coordinates, or sinogram dimensions. The result is a precise map of what assumes circular orbit (Z=0 at all angles), what is already general-purpose, and what must change for helical.

### File-by-File Findings

---

#### 1. `src/geometry/scanner.jl` — Scanner + CTGeometry

**Scanner struct (lines 127–179):** NO circular orbit assumptions. Scanner defines physical parameters only — geometry, detector, source, filters. Nothing about scan trajectory. **Already general.**

**CTGeometry struct (lines 552–566):** The struct itself is GENERAL. It stores:
- `source_positions::Matrix{Float64}` — `[3, n_angles]` — per-angle XYZ positions
- `detector_centers::Matrix{Float64}` — `[3, n_angles]` — per-angle XYZ positions
- `detector_u::Matrix{Float64}` — `[3, n_angles]` — per-angle u-axis vectors
- `detector_v::Matrix{Float64}` — `[3, n_angles]` — per-angle v-axis vectors
- `angles::Vector{Float64}` — angle values

The struct CAN hold helical positions. Nothing prevents non-zero Z components.

**CTGeometry constructor (lines 609–699):** THIS IS WHERE Z=0 IS HARDCODED.

```julia
# Line 673: source Z is hardcoded to 0.0
source_positions[3, i] = 0.0

# Line 679: detector center Z is hardcoded to 0.0
detector_centers[3, i] = 0.0

# Line 684-685: detector u has Z=0 (correct — u is always in XY plane for curved detectors)
detector_u[3, i] = 0.0

# Line 688-689: detector v is hardcoded to (0,0,1) — always +Z
detector_v[3, i] = 1.0
```

**What changes for helical:** Only the constructor needs modification. The Z components of source_positions and detector_centers should be `z_start + (θ / 2π) × pitch × collimation_cm` instead of `0.0`. The detector_u and detector_v vectors are CORRECT as-is for helical — the detector doesn't tilt, it translates along Z.

**`create_aquilion_one` (lines 725–807):** Same Z=0 pattern as the CTGeometry constructor. Legacy factory function with identical hardcoded `0.0` for Z components.

**Angles generation (line 658):**
```julia
angles = collect(range(0.0, 2π - 2π/n_angles, length=n_angles))
```
This generates exactly ONE rotation. For helical with `n_rotations > 1`, this must generate `n_rotations × views_per_rotation` angles spanning `[0, n_rotations × 2π)`.

---

#### 2. `src/projection/siddon.jl` — Ray Tracer

**`siddon_trace_ray` (lines 150–294):** COMPLETELY GENERAL. Takes arbitrary `(src_x, src_y, src_z)` and `(det_x, det_y, det_z)` positions. Uses parametric ray representation `P(t) = S + t(D - S)`. No circular orbit assumptions whatsoever. The algorithm works for any ray from any source to any detector position in 3D space. **NO CHANGES NEEDED.**

**`siddon_forward_project!` (lines 414–538):** ALMOST COMPLETELY GENERAL. Takes geometry arrays from CTGeometry:
- Reads `source_positions[1..3, angle]` — uses all three coordinates
- Reads `detector_centers[1..3, angle]` — uses all three coordinates
- Reads `detector_u[1..3, angle]` and `detector_v[1..3, angle]` — uses all three
- Computes detector pixel position as: `det_center + u_offset × du + v_offset × dv`

**The only assumption:** Volume bounds are centered at origin (lines 440–448):
```julia
vol_min_x = T(-vol_bounds[1] / 2)
vol_min_z = T(-vol_bounds[3] / 2)
```
For helical, the phantom may not be centered at Z=0 relative to the scan range. However, since the phantom is stationary and the source/detector translate, this is actually correct — the phantom's physical extent doesn't change. **NO CHANGES NEEDED for the projector itself.**

---

#### 3. `src/projection/polychromatic.jl` — Polychromatic Forward Projection

**`_forward_project_poly!` (lines 1100–1140):** Calls `siddon_forward_project!` in a loop over energies. Passes through all geometry arrays. **No Z assumptions of its own. NO CHANGES NEEDED** — it inherits whatever geometry is in CTGeometry.

The polychromatic pipeline accumulates `I_transmitted = Σ w_e × exp(-L_e)` where `L_e` is the line integral at energy `e`. This is purely ray-based and orbit-agnostic.

---

#### 4. `src/reconstruction/fbp/fdk.jl` — FDK Reconstruction

**`fdk_reconstruct` (lines 327–346):** Three steps:
1. `filter_sinogram` — cosine weighting + ramp filter
2. `backproject` — voxel-driven FDK backprojection
3. `apply_fov_mask!` — circular FOV mask

**CRITICAL HELICAL ISSUE:** FDK assumes a circular orbit. The math is:
```
f(x,y,z) = (π/N) Σ_θ [SAD² / dist²] × p̂(θ, u*, v*)
```
This formula is EXACT for circular orbit but APPROXIMATE for helical. For helical CT, the standard approaches are:
1. FDK with helical weighting (approximate but fast — used by most clinical scanners)
2. Katsevich exact algorithm (exact but complex)
3. Rebinning to parallel beam (fastest but lowest quality)

The backprojection kernel (see below) already uses arbitrary source/detector positions, so the geometric computation is helical-ready. What's missing is the **helical weighting function** that replaces the simple `SAD² / dist²` weight.

---

#### 5. `src/reconstruction/core/backprojection.jl` — Backprojection Kernel

**`backproject_voxel` (lines 37–146):** The FDK-weighted backprojection kernel. **ALREADY GENERAL in geometry:**
- Reads per-angle `source_positions[1..3, angle]` — all three components
- Reads per-angle `detector_centers[1..3, angle]` — all three components
- Computes ray from source to voxel, projects onto detector plane
- Uses bilinear interpolation to sample filtered sinogram

**The FDK weight (line 137–138):**
```julia
dist_sq = sv_x^2 + sv_y^2 + sv_z^2
weight = SAD_sq / dist_sq
```
This is the circular-orbit FDK weight. For helical, this needs to be replaced with a helical weight that accounts for redundant data (conjugate rays from different rotations). The exact form depends on the chosen helical reconstruction algorithm.

**`backproject_voxel_matched` (lines 159–264):** Unweighted backprojection for iterative methods. **ALREADY FULLY GENERAL** — no FDK weight applied, just bilinear interpolation from per-angle geometry. Works unchanged for helical iterative reconstruction (SIRT, CGLS).

**`backproject!` (lines 296–426):**
- `pi_over_angles = T(π) / T(n_angles)` — scaling factor. For helical multi-rotation, this needs adjustment to `T(π) / T(angles_per_rotation)` or similar, depending on the helical weighting scheme.

---

#### 6. `src/reconstruction/core/filtering.jl` — Sinogram Filtering

**`cosine_weight!` (lines 362–404):** Applies cosine weighting based on detector pixel position relative to central ray. Uses `SDD`, `pixel_size`, `pixel_row_size` — all per-detector-element quantities. **No orbit assumption. NO CHANGES NEEDED.**

**`filter_sinogram!` (lines 433–504):** Applies row-by-row ramp filter convolution. Operates on `sinogram[col, row, angle]` independently per angle. **No orbit assumption. NO CHANGES NEEDED.**

---

#### 7. `src/api/workspace.jl` — Workspace Allocation

**`EICTWorkspace` (lines 441–501):** Stores sinogram buffers as `(n_cols, n_rows, n_angles)`. **The sinogram dimension assumption is KEY:**
- Currently: `n_angles = protocol.views` (one rotation)
- For helical: `n_angles = protocol.views × n_rotations` (multi-rotation)

**`create_eict_workspace` (line 514):**
```julia
geom = CTGeometry(scanner; n_angles=protocol.views, ...)
```
This passes `protocol.views` as `n_angles`. For helical, this should be `protocol.views * ceil(Int, protocol.n_rotations)` to cover the full helical scan.

**`sino_shape = (geom.n_cols, geom.n_rows, geom.n_angles)` (line 528):** Buffer allocation is correct — it uses whatever `n_angles` comes from the geometry. **No change needed here** as long as the geometry is constructed correctly.

**`PCCTWorkspace`:** Same pattern — derives `sino_shape` from `geom`. **Same fix needed at the geometry construction call.**

---

#### 8. `src/api/driver.jl` — simulate!() Driver

**`simulate!(ws::EICTWorkspace, ...)` (lines 330–470):** The driver function:
1. Calls `_forward_project_poly!` with `ws.geom` — passes through geometry
2. Applies physics pipeline with `ws.geom`
3. Applies noise with `compute_detector_I0(geom, protocol, ...)`

**No circular orbit assumptions in the driver itself.** It uses whatever geometry the workspace provides. The physics pipeline calls pass `geom` through to their respective functions.

**`compute_detector_I0` (detector_noise.jl:457):** Computes I₀ per pixel per view. Uses `protocol.rotation_time / protocol.views` for time per view. For helical, if `views` is per-rotation and we have N rotations, total views = N × views, but time per view stays the same (rotation_time / views_per_rotation). **No change needed** if `protocol.views` stays as views-per-rotation and `n_angles` is computed separately.

---

#### 9. `src/source/protocol.jl` — CTProtocol

**`n_rotations` field (line 41):** `n_rotations::Float64` — already exists! Default is 1.0. Currently only used in `compute_dlp` (line 299) for dose calculation. **NOT used for geometry construction.**

**What's missing:** A `pitch` field. Helical requires:
- `pitch::Float64` — table distance per rotation / beam collimation (dimensionless, typically 0.5–1.5)
- Alternatively: `table_speed::Float64` (mm/s) from which pitch is derived

The `n_rotations` field is already there and ready to be connected to geometry construction.

---

#### 10. `src/geometry/affine.jl` — Coordinate Transforms

**`recon_to_world_affine` (lines 61–79):** Maps reconstruction voxel indices to world coordinates. Uses `geom.fov` for scale and translation. **Assumes reconstruction grid is centered at origin.** For helical, the reconstruction volume may need to be centered at a different Z position (e.g., mid-scan). This is a minor change — just add a Z offset to the affine.

**`resample_to_recon` (lines 113–232):** Resamples phantom onto recon grid. Uses the same FOV-centered-at-origin assumption. Same minor Z-offset change needed.

---

#### 11. `src/detector/physics_pipeline.jl` — Physics Effects

**`apply_physics_effects!` (lines 407–539):** Iterates through 10+ physics effects. Each effect takes `sinogram` and `geom`. Key observations:
- **Fill factor, scatter, crosstalk, noise, lag:** Operate per-pixel on the sinogram. Angle-independent. **NO CHANGES NEEDED.**
- **Flat filter, bowtie filter:** Use `geom` for pixel position computation. They compute per-column or per-(col,row) attenuation. **Angle-independent — NO CHANGES NEEDED.**
- **Heel effect:** Uses `geom` for source orientation. Per-angle computation but reads source positions from geometry. **Works as-is with helical geometry.**
- **Focal spot blur:** Spatial blur kernel. Angle-independent. **NO CHANGES NEEDED.**
- **BHC:** Polynomial correction applied per-pixel. **NO CHANGES NEEDED.**

**CONCLUSION: The entire 16-step physics pipeline is angle-agnostic and works unchanged for helical.**

---

### Classification Summary

#### ALREADY GENERAL — No Changes Needed
| Component | File | Why It's General |
|-----------|------|-----------------|
| Scanner struct | scanner.jl:127–179 | Defines physical params only, no trajectory |
| CTGeometry struct | scanner.jl:552–566 | Stores per-angle [3,N] positions — can hold helical |
| Siddon ray tracer | siddon.jl:150–294 | Takes arbitrary (src,det) positions |
| Siddon forward_project! | siddon.jl:414–538 | Reads per-angle geometry arrays |
| Polychromatic projector | polychromatic.jl:1100+ | Calls Siddon in energy loop |
| Cosine weighting | filtering.jl:362–404 | Per-pixel detector geometry only |
| Ramp filter | filtering.jl:433–504 | Row-by-row, angle-independent |
| Matched backprojection | backprojection.jl:159–264 | No FDK weight, reads per-angle geometry |
| Physics pipeline (ALL) | physics_pipeline.jl:407–539 | Angle-agnostic operations |
| simulate!() driver | driver.jl:330–470 | Passes through workspace geometry |

#### NEEDS PARAMETER ADDITION (Small Changes)
| Component | File:Line | What Changes |
|-----------|-----------|-------------|
| CTGeometry constructor | scanner.jl:666–689 | Add Z(θ) = z_start + pitch × collimation × θ/(2π) to source/detector Z |
| create_aquilion_one | scanner.jl:773–797 | Same Z(θ) pattern (or deprecate in favor of CTGeometry constructor) |
| Angles generation | scanner.jl:658 | Generate n_rotations × views angles instead of one rotation |
| CTProtocol | protocol.jl:35–49 | Add `pitch::Float64` field |
| Workspace geometry call | workspace.jl:514 | Pass `n_angles = views × n_rotations` |

#### NEEDS ALGORITHMIC ADDITION (Helical Reconstruction)
| Component | File:Line | What Changes |
|-----------|-----------|-------------|
| FDK backprojection weight | backprojection.jl:137–138 | Replace `SAD²/dist²` with helical weight |
| Backprojection scaling | backprojection.jl:335 | Adjust `π/n_angles` for multi-rotation |
| fdk_reconstruct | fdk.jl:327–346 | Add helical weighting step before/during backprojection |
| Affine Z-center | affine.jl:61–79 | Optional Z-offset for non-centered recon volumes |

### Key Insight: Minimal Changes Required

**The codebase is 90% helical-ready.** The architecture decision to pre-compute per-angle source/detector positions and pass them through means the forward projection path (Siddon + polychromatic + physics) works UNCHANGED for helical. Only three things need real work:

1. **Geometry construction** — compute Z(θ) positions instead of Z=0
2. **CTProtocol** — add `pitch` field
3. **Reconstruction** — helical weighting for FDK (or implement Katsevich)

The iterative reconstruction path (SIRT, CGLS) works UNCHANGED for helical because matched backprojection uses arbitrary per-angle geometry with no FDK weight.

### Gaps / Open Questions for Next Iterations

1. **Which helical reconstruction algorithm?** FDK+helical weights is simplest. Need HELI-003 research.
2. **Phantom Z-coverage:** For helical, phantom must extend beyond the reconstruction Z-FOV to cover the helical travel. How much padding? Depends on pitch.
3. **n_angles semantics:** Should `protocol.views` stay as per-rotation? Or become total views? Need HELI-004 design.
4. **Reconstruction Z-FOV:** Helical allows reconstructing arbitrary Z ranges (sliding window). How to specify?
5. **PCCT + helical:** Same geometry changes apply. PCCT detector physics is angle-agnostic. Native dexel resolution path would also work unchanged.

---

## Iteration 2: HELI-001 DISCOVERY — Helical CT Geometry (completed by prior agent)

**Date:** 2026-03-17
**Phase:** Discovery
**Topic:** HELI-001 (Helical CT Geometry — Table Motion, Pitch, Coordinate Transforms)

### Summary

Researched helical geometry: IEC pitch definition, Z(θ) formula, multi-rotation angle generation, collimation relationships, clinical pitch values, phantom Z-coverage requirements, reconstruction Z-range (Tam-Danielsson window), and reconstruction volume Z-centering. Proposed CTGeometry constructor changes with full pseudocode.

### Key Findings

- **Pitch definition:** IEC 60601-2-44 standard. `pitch = table_feed_per_rotation / total_collimation_width`.
- **Z(θ):** `Z(θ) = Z_start + (θ/2π) × pitch × total_collimation_cm`. Both source and detector share the same Z (rigid gantry).
- **Centering:** `z_start = -z_travel/2`, scan symmetric about Z=0. For pitch=0: `z_start = 0.0` (axial degenerate case).
- **Proposed struct additions:** `pitch`, `views_per_rotation`, `recon_center` fields to CTGeometry.
- **Clinical pitch values:** 0.2–1.5 typical, 3.0–3.4 for dual-source cardiac.

See spec section 1 for full details.

---

## Iteration 3: HELI-002 DISCOVERY — Forward Projection with Table Motion

**Date:** 2026-03-17
**Phase:** Discovery
**Topic:** HELI-002 (Forward Projection — Siddon with Table Motion)

### Summary

Deep audit of the entire forward projection pipeline for helical compatibility: Siddon ray tracer, polychromatic Beer-Lambert, PCCT photon-counting, `simulate!()` driver, noise model, iterative reconstruction projector, GPU kernel parallelism, and memory implications.

### Key Findings

**The entire forward projection pipeline works UNCHANGED for helical.** This is the central finding.

### Detailed Analysis

#### 1. Siddon Ray Tracer (`siddon.jl:150–294`)

Completely general. Takes arbitrary `(src_x, src_y, src_z, det_x, det_y, det_z)`. Uses parametric `P(t) = S + t(D-S)` with 3D-DDA. No orbit assumptions. **NO CHANGES.**

#### 2. `siddon_forward_project!` (`siddon.jl:414–538`)

Reads per-angle arrays from CTGeometry. Volume bounds via `volume_extent` kwarg (phantom's physical size) or `geom.fov` (recon FOV). Both are centered at origin. Phantom is stationary — source/detector Z varies — rays correctly intersect the stationary volume. **NO CHANGES** for phantom projection.

**One subtlety for iterative recon:** SIRT/CGLS call `siddon_forward_project(recon_volume, geom)` without `volume_extent`, defaulting to `geom.fov` centered at origin. For helical with `recon_center_z ≠ 0`, the volume bounds need a Z offset. This is a minor change:
```julia
vol_min_z = T(-vol_bounds[3] / 2 + geom.recon_center[3])
```
This is driven by reconstruction requirements, not forward projection requirements.

#### 3. Polychromatic Pipeline (`polychromatic.jl:1100–1195`)

Energy loop → `create_μ_volume!` → `siddon_forward_project!` → Beer-Lambert accumulation. Passes through geometry and `volume_extent`. Orbit-agnostic. **NO CHANGES.**

#### 4. PCCT Pipeline (`photon_counting.jl:1192–1440`)

Same structure as polychromatic. Energy loop → Siddon → spectral binning → detector physics (charge sharing, pileup, anti-coincidence). All per-pixel operations. Native-resolution path also just reads geometry. **NO CHANGES.**

#### 5. `compute_detector_I0` (`detector_noise.jl:457–473`)

Uses `time_per_view = rotation_time / views_per_rotation`. This is physically correct — the per-view integration time doesn't change with helical vs axial. `protocol.views` means views per rotation. **NO CHANGES.**

#### 6. GPU Kernel Parallelism

AK.foreachindex parallelizes over `n_cols × n_rows × n_angles`. More angles (multi-rotation) = more work items = higher GPU utilization. Each ray is independent. **Perfect scaling.**

#### 7. Memory Implications

Sinogram buffers scale linearly with `n_rotations`. For 5 rotations: ~6.5 GB workspace (900×64 cols/rows). For 10 rotations: ~13 GB. Within capacity of modern GPUs but may need optimization for very long scans (>10 rotations).

#### 8. Phantom Z-Coverage Validation

Recommended: warn if `phantom.extent[3] < z_travel + total_collimation_cm`. Not an error — projecting through empty space produces correct zero values.

### Classification

| Component | Change | Status |
|-----------|--------|--------|
| `siddon_trace_ray` | NONE | Already general |
| `siddon_forward_project!` | Minor: volume_center for iterative recon | Reconstruction-driven |
| `_forward_project_poly!` | NONE | Pass-through |
| `pcct_forward_project` | NONE | Pass-through |
| `compute_detector_I0` | NONE | Uses views_per_rotation |
| Noise model | NONE | Per-view, orbit-agnostic |
| Workspace buffers | SIZE ONLY | Automatic from geom.n_angles |

### Gaps for Next Iterations

1. **HELI-003 (Reconstruction):** The CRITICAL unknown. Forward projection is free, but reconstruction needs helical weighting. Must research FDK+weights vs Katsevich vs rebinning.
2. **HELI-005 (Physics Pipeline):** Partially addressed here (all physics is angle-agnostic), but a systematic audit of all 16 effects would be thorough.
3. **Volume center for iterative recon:** The `recon_center` Z offset needs to be threaded through to `siddon_forward_project!` when called from SIRT/CGLS. Design this in HELI-004.

---

## Iteration 4: HELI-003 DISCOVERY — Helical Reconstruction Algorithms

**Date:** 2026-03-17
**Phase:** Discovery
**Topic:** HELI-003 (Helical Reconstruction Algorithms — FDK-Helical vs Katsevich vs Rebinning)

### Summary

Deep research into helical CT reconstruction algorithms via web search, reference implementation analysis, and clinical scanner vendor surveys. Produced comprehensive analysis of four algorithm classes with exact mathematical formulations, GPU implementation feasibility, clinical usage, and quality comparisons.

### Key Findings

#### 1. ALGORITHM RECOMMENDATION: Two-Phase Approach Confirmed

**Phase 1 — Naive Helical FDK:** Change only `pi_over_angles = π/views_per_rotation`. The existing backprojection kernel naturally handles helical geometry because voxels outside the detector cone contribute zero. Works for pitch ≤ 1.

**Phase 2 — Weighted Helical FDK (WFBP-style):** Add z-distance-based W(q̂) weighting to the backprojection kernel. The weight function:
```
W(q̂) = 1                                    if |q̂| < 0.6
W(q̂) = cos²(π/2 × (|q̂| − 0.6) / 0.4)      if 0.6 ≤ |q̂| < 1.0
W(q̂) = 0                                    if |q̂| ≥ 1.0
```
Normalization: `f(x) = π × Σ(w_h × w_fdk × val) / Σ(w_h × w_fdk)`. For axial (pitch=0), w_h=1 for all, reduces to standard FDK.

#### 2. CLINICAL SCANNER ALGORITHMS (Detailed)

**Siemens (WFBP — Stierstorfer 2004, PMID 15248573):**
- Fan-to-parallel rebinning → ramp filter → 3D backprojection with z-distance cos² taper (Q=0.6)
- Key property: noise approximately pitch-independent (< 7% variation)
- Open-source: FreeCT_wFBP (github.com/xiehq/FreeCT_wFBP, PMID 26936725)
- All Siemens scanners including NAEOTOM Alpha use WFBP as base; SAFIRE/ADMIRE refine on top

**GE (3D CB-FBP — Tang/Hsieh 2006, PMID 16467583):**
- Row-wise fan-to-parallel rebinning ("tilted cone-beam reconstruction")
- Cone-angle-based conjugate ray weighting: w₁ = κ₂²/(κ₁²+κ₂²) for conjugate pair
- "Comparable to exact CB reconstruction under moderate cone angle (4°)"
- ASiR-V / TrueFidelity applied post-reconstruction

**Key: NO vendor uses Katsevich.** All use approximate weighted cone-beam FBP. DL reconstruction (TrueFidelity, AiCE) largely masks remaining cone-beam artifacts.

#### 3. KATSEVICH — NOT RECOMMENDED

- Theoretically exact FBP-type for helical cone-beam (Katsevich 2002, SIAM J. Appl. Math.)
- PI-line based: f(x) = -(1/2π²) ∫ [1/|x-y(s)|²] × Hilbert_filtered dq ds
- Filtering along oblique "K-lines" on detector (not rows) — requires interpolation to/from K-line coordinates
- Uses only PI-arc data (~73% of measurements) → **higher noise** than WFBP (100%)
- **NO open-source GPU implementation exists** (confirmed: TIGRE, RTK, ASTRA all lack it)
- Only published GPU paper: Yan et al. 2010 (PMID 20007041) — 16-year-old CUDA 2.x, no public code
- Estimated 3-5x computational cost vs FDK, ~1500-3000 lines of code
- Katsevich algorithm was patented (filed ~2002, expired ~2022)
- Only matters at cone angles > 5° — our target scanners are 1.6-3.3°

#### 4. REFERENCE IMPLEMENTATION SURVEY

- **XCIST/CatSim**: ONLY framework with complete helical FBP. Key file: `Parallel_FDK_Helical_3DWeighting.c`. Uses cos² view window + conjugate-ray Gamma^k1 weighting. Separate code path from axial.
- **TIGRE**: No helical FDK. Uses per-angle `offOrigin[z]` for helical geometry. Only iterative recon (SIRT, CGLS) works for helical. Confirmed by maintainer: "we don't have the Katsevich algorithm implemented."
- **ASTRA**: FDK explicitly blocked for non-circular geometry (`fdk.cu` line 420: "we don't support arbitrary cone_vec geometries here"). Iterative works via `cone_vec`.
- **RTK**: No helical support. Zero references to Katsevich in entire repository.
- **FreeCT_wFBP**: Complete open-source WFBP (Siemens-style). Best reference for our implementation.

#### 5. GPU KERNEL DESIGN

The modified `backproject_voxel` for helical needs:
- **2 new parameters:** `half_collimation_iso` (T), `is_helical` (Bool)
- **Replace** `pi_over_angles` accumulation with dual-accumulator (weighted values + weight sums)
- **Add** ~10 lines of helical weight computation per angle
- **Normalization:** `π × acc_val / acc_wt` (degenerates to `π/n_angles × acc` for axial)
- **No new buffers** — weight accumulator is per-voxel inline, not stored

Iterative recon (SIRT, CGLS) needs NO changes — matched backprojection is already orbit-agnostic.

#### 6. CONE ANGLE QUALITY ANALYSIS

From Tan et al. 2012 (PMID 28519621), Catphan 600 phantom at 360° helical:
- Katsevich noise σ = 28.07 HU vs FDK noise σ = 44.64 HU (37% improvement)
- But this was CBCT at cone half-angle ~8° — far beyond our target scanners

Clinical CT cone half-angles:
- NAEOTOM Alpha: ~1.6° (57.6mm/2 at SAD=595mm × SDD/SAD)
- GE Apex Elite: ~2.3° (80mm effective at isocenter)
- At these angles, WFBP artifacts are below the noise floor

### Normalization Analysis for Axial-Helical Seamlessness

**Critical subtlety identified:** For axial (pitch=0), a z-distance-based weight (using source_z - voxel_z) would cause problems because all source_z = 0, and voxels near the z-edges of the recon volume could have large |Δz|. The existing spec correctly uses **detector row position** (q̂ = v*/v_max) instead of z-distance — this naturally degenerates to constant weight for axial since the detector projection doesn't change shape.

However, for axial FDK the detector row projection `row_f` of a voxel at z=z_max relative to row_center IS nonzero (it's the cone-beam projection). So for a 64-row detector with a recon volume spanning the full z-extent, voxels at the z-edges would have |q̂| approaching ~1 even in axial mode. The W(q̂) weight would taper their contributions, which is **undesirable** for axial.

**Solution:** Use `is_helical` flag. For axial (pitch=0), skip helical weighting entirely (w_h=1 for all). For helical (pitch>0), compute W(q̂). Since this is a scene-wide constant (not per-voxel), all GPU threads take the same branch — zero performance cost.

### Open Questions for Future Iterations

1. **Fan-to-parallel rebinning (Phase 3):** Worth implementing? XCIST and FreeCT both use it. Improves filtering accuracy. But adds a preprocessing GPU kernel. Defer to HELI-004 API design.
2. **GE conjugate-ray approach vs Siemens z-distance approach:** The Tang/Hsieh cone-angle weighting (κ₂²/(κ₁²+κ₂²)) is more physically motivated than the cos² taper. Consider as an enhancement.
3. **PCCT + helical:** The PCCT workspace uses the same geometry → same helical changes apply. But multi-energy binning + helical is a combined complexity. Verify in HELI-005.

---

## Iteration 5: CRITIQUE of HELI-000 through HELI-003

**Date:** 2026-03-17
**Phase:** Critique
**Topics:** HELI-000 (Geometry Audit), HELI-001 (Helical Geometry), HELI-002 (Forward Projection), HELI-003 (Reconstruction Algorithms)

### Method

Cross-referenced every claim in `helical-spec.md` sections 0–3 against the actual BasisSimulator source code. Verified line numbers, tested mathematical formulas for self-consistency and axial degeneration, checked all function signatures and call sites, and tested integration concerns.

---

### CRITICAL Issues (Must Fix Before Implementation)

#### C1. Inconsistent normalization formulas in Section 3.3.6

The spec provides THREE different normalization formulas that give different results:

1. **Line 714:** `val_acc * pi_over_angles / wgt_acc * T(n_angles) / T(views_per_rotation)`
   - Expanding: `val_acc * (π/n_angles) / wgt_acc * (n_angles/views_per_rotation)` = `val_acc * π / (wgt_acc * views_per_rotation)`
2. **Line 726:** `val_acc / wgt_acc * T(π)`
   - This is: `val_acc * π / wgt_acc`
3. **Section 3.9, line 960:** `val_acc * T(π) / wgt_acc`
   - Same as formula 2.

Formula 1 differs from formulas 2/3 by a factor of `1/views_per_rotation`. These cannot all be correct. **The spec must settle on ONE formula and derive it from first principles.**

The correct WFBP formula (Stierstorfer 2004, Eq. 5) is:
```
f(x) = Δα × Σ_i [ W_norm(i,x) × w_fdk(i) × p̂(i, u*, v*) ]
```
where `Δα = 2π / views_per_rotation` is the angular step, and `W_norm(i,x) = W(q̂_i) / Σ_k W(q̂_{i+kπ})` is the group-normalized weight. For a simplified version using total-sum normalization:
```
f(x) = (2π / views_per_rotation) × Σ[ W × w_fdk × p̂ ] / Σ[ W × w_fdk ]
```
This simplifies to `val_acc * (2π / views_per_rotation) / wgt_acc`. Neither formula 1, 2, nor 3 matches this exactly.

**Action needed:** Derive the correct normalization from the WFBP integral and verify against FreeCT_wFBP source code (`cuda_kernels.cuh`, backprojection kernel). The factor of 2 (from 2π vs π) needs to be resolved by checking whether the FDK integral convention uses (1/2)∫₀²π or ∫₀π.

#### C2. Naive FDK fix (Section 3.2) only works for pitch ≈ 1, not pitch ≤ 1

The spec claims (line 534): "Works well for **pitch ≤ 1** and cone half-angle < 5°"

This is incorrect for pitch significantly below 1. Analysis:

- For pitch < 1, detector cones overlap — multiple rotations contribute data for the same voxel
- A central voxel sees approximately `1/pitch` rotations of data (e.g., ~2 rotations for pitch=0.5)
- With `pi_over_angles = π / views_per_rotation`, the sum of `~2N` non-zero terms × `π/N` = `~2× correct value`
- **Result: Voxels are ~(1/pitch)× too bright for pitch < 1**

Example: pitch=0.5, 3 rotations → central voxels see ~2 rotations → ~2× overestimation.

**Action needed:** Change claim to "Works well for **pitch ≈ 1** (±0.1). For pitch < 0.8, expect overestimation by a factor approaching 1/pitch due to unhandled data redundancy. Use Phase 2 (weighted WFBP) for pitch < 0.8."

#### C3. The normalized formula does NOT degenerate to standard FDK for axial

The spec claims (or implies) that the weighted helical formula degenerates to standard FDK when W(q̂)=1 for all angles. This is false.

**Standard axial FDK:**
```
f(x) = (π/N) × Σ_{i=1}^{N} [SAD²/dist²_i] × p̂_i
```

**Helical normalized (with W=1):**
```
f(x) = π × [Σ (SAD²/dist²_i) × p̂_i] / [Σ (SAD²/dist²_i)]
```

These are equivalent ONLY if `Σ(SAD²/dist²_i) = N`, which is true only at the isocenter (where dist = SAD for all angles). For off-center voxels, the FDK weight `SAD²/dist²` varies per angle, so `Σ w_fdk ≠ N`.

**Consequence:** The `is_helical` flag is necessary not just for the W(q̂) weighting but for the ENTIRELY DIFFERENT normalization strategy. For axial, use the existing `π/N × Σ(w×p̂)`. For helical, use the weight-normalized form. **The spec already proposes the `is_helical` flag (good), but the reasoning should be corrected — it's not just about cone-edge tapering in axial mode.**

#### C4. Adding fields to CTGeometry breaks 6 inner constructor call sites (spec lists only 2)

The spec proposes adding `pitch`, `views_per_rotation`, and `recon_center` to the `CTGeometry` struct. Julia's default inner constructor requires ALL fields in positional order. The spec identifies changes to `scanner.jl` and `workspace.jl`, but there are **6 call sites** using the positional inner constructor:

| # | File:Line | What It Does |
|---|-----------|-------------|
| 1 | `scanner.jl:694` | Main CTGeometry constructor return |
| 2 | `scanner.jl:802` | `create_aquilion_one` return |
| 3 | `fdk.jl:426` | FOV-override variant of `fdk_reconstruct` |
| 4 | `workspace.jl:364` | PCCT native-resolution geometry |
| 5 | `mbir.jl:434` | Ordered-subset geometry slicing |
| 6 | `scanners.jl:327` | Scanner factory function (`create_geometry`) |

All 6 must be updated when fields are added. The MBIR site (#5) is particularly subtle — it creates a geometry subset for ordered-subset iteration. The new fields (`pitch`, `views_per_rotation`, `recon_center`) must be propagated: `geom.pitch`, `geom.views_per_rotation`, `geom.recon_center`.

**Action needed:** List ALL 6 sites in the spec. For site #5 (MBIR), note that `views_per_rotation` stays the same even though `n_angles` changes (it's a subset of a single rotation).

#### C5. Helical `fov_z` computation not specified in constructor

The current CTGeometry constructor computes `fov_z` from single-rotation detector coverage:
```julia
# scanner.jl:653-655
z_coverage_mm = _n_rows * scanner.detector_row_size
fov_z = z_coverage_mm / 10.0  # mm → cm
```

For helical, the reconstruction z-FOV should be the **Tam-Danielsson window** (fully-sampled z-range):
```
z_recon = z_travel - total_collimation_cm
       = (n_rotations × pitch × collim) - collim
       = collim × (n_rotations × pitch - 1)
```

The proposed constructor changes (Section 1.10) modify the angle generation and Z-positions but do NOT change the `fov_z` computation. A helical scan with 3 rotations at pitch=0.8 on NAEOTOM (collim=5.76cm) should have:
```
z_recon = 5.76 × (3 × 0.8 - 1) = 5.76 × 1.4 = 8.06 cm
```
But the current code would give `fov_z = 5.76 cm` (single-rotation coverage).

**Action needed:** Add helical-aware `fov_z` computation to the constructor:
```julia
if pitch > 0 && z_cm === nothing
    total_collim_cm = _n_rows * scanner.detector_row_size / 10.0
    z_travel = n_rotations * pitch * total_collim_cm
    fov_z = z_travel - total_collim_cm  # Tam-Danielsson window
    fov_z = max(fov_z, total_collim_cm)  # At least single-rotation coverage
end
```

---

### IMPORTANT Issues (Should Fix)

#### C6. `recon_center` Z-offset needed in BOTH forward projector AND backprojection

The spec shows the forward projector change (Section 2.2):
```julia
vol_min_z = T(-vol_bounds[3] / 2 + geom.recon_center[3])
```

But the SAME offset must be added to `backproject!` (backprojection.jl:316-318):
```julia
vol_min_z = T(-geom.fov[3] / 2)  →  T(-geom.fov[3] / 2 + geom.recon_center[3])
```

Section 1.9 mentions this conceptually but doesn't list backprojection.jl as a change site. The `resample_to_recon` affine (affine.jl:69-71) also needs the same offset:
```julia
tz = -fov_z / 2 + sz / 2  →  -fov_z / 2 + sz / 2 + recon_center_z
```

**Action needed:** Explicitly list ALL sites needing the recon_center offset:
1. `siddon_forward_project!` (siddon.jl:440-442) — for iterative recon path
2. `backproject!` (backprojection.jl:316-318) — for FDK and matched BP
3. `recon_to_world_affine` (affine.jl:69-71) — for label resampling
4. `resample_to_recon` (affine.jl:128-130) — for phantom-to-recon mapping

#### C7. PCCT workspace has same `n_angles=protocol.views` issue

The spec identifies `create_eict_workspace` (workspace.jl:514) as needing `n_angles = views × n_rotations`. But `create_workspace` (PCCT, workspace.jl:174) has the IDENTICAL pattern:
```julia
geom = CTGeometry(scanner; n_angles=protocol.views, ...)
```

**Both** must pass helical-aware n_angles. The fix is in the CTGeometry constructor (compute total_views internally from pitch and n_rotations), NOT in each workspace function. The constructor should treat `n_angles` as views_per_rotation when pitch > 0.

**Action needed:** Clarify that the `n_angles` parameter to CTGeometry should retain its current meaning (views per rotation). The constructor internally computes `total_views = round(Int, n_angles * n_rotations)` when `pitch > 0`. This means NO changes to workspace call sites — the constructor handles it.

#### C8. Weight accumulator buffer contradiction

Section 3.3.6 (line 699) says:
```julia
weight_sum = similar(volume)
fill!(weight_sum, zero(T))
```

But Section 3.9 pseudocode (line 960) computes normalization inline:
```julia
return wgt_acc > T(1e-10) ? val_acc * T(π) / wgt_acc : zero(T)
```

The inline approach is correct and sufficient — the weight accumulator is per-voxel and computed in the inner loop, not stored as a separate buffer. **Remove the `weight_sum` buffer from section 3.3.6** to avoid confusion.

#### C9. Dual-energy + helical not addressed

The spec doesn't mention dual-energy helical at all. For dual-kVp mode, views alternate between high and low kVp. For helical, each view is at a different z-position, so the kVp alternation creates an interleaved z-coverage pattern. Material decomposition would need to handle the z-dependent alternation.

**Action needed:** Add a note to the spec: "Dual-energy + helical is out of initial scope. The current dual-kVp alternation pattern works geometrically (views at different z-positions have alternating kVp), but material decomposition validation is deferred."

#### C10. `create_aquilion_one` disposition unclear

The spec says "Same Z(θ) pattern (or deprecate)" for `create_aquilion_one`. Given the project policy (NO backward compatibility), this should be a clear decision:
- If `create_aquilion_one` is still used: update it to support helical parameters
- If it's superseded by the CTGeometry constructor: delete it

**Action needed:** Check whether `create_aquilion_one` is used anywhere outside tests. If not, mark it for deletion. If yes, update it to accept `pitch` and `n_rotations` kwargs.

---

### MINOR Issues

#### C11. Simplified q̂ is acknowledged as approximate but should be more explicit

The spec uses `q_hat = (row_f - row_center) / (n_rows/2)` as a simplified proxy for the full WFBP q̂ formula which includes the helical interpolation correction:
```
q̂ = (z_voxel − z_table + (z_rot/(2π)) arcsin(p̂/SAD)) / (l̂ × tan(θ_cone/2))
```

The simplified version ignores the `(z_rot/(2π)) arcsin(p̂/SAD)` term, which corrects for the oblique intersection of the helix. This matters more at high pitch and large fan angles. For our target scanners (pitch ≤ 1.5, fan angle ~25°), the correction is ~0.5° — likely below noise.

**Action needed:** Add a note that the full q̂ formula should be the Phase 2b target, and that the simplified version is explicitly chosen for Phase 2a to reduce implementation risk.

#### C12. `is_helical` derivation chain not specified

The spec says `is_helical` is a parameter to `backproject_voxel` but doesn't specify who computes it. It should be:
```
geom.pitch > 0  →  is_helical = true  (in backproject!)
```
Not a user-settable parameter.

---

### Internal Consistency Check

#### Line numbers: VERIFIED ✓

All spec line numbers were checked against actual source code (as of this iteration):
- `scanner.jl:673` = `source_positions[3, i] = 0.0` ✓
- `scanner.jl:679` = `detector_centers[3, i] = 0.0` ✓
- `scanner.jl:658` = angles range ✓
- `backprojection.jl:335` = `pi_over_angles = T(π) / T(n_angles)` ✓
- `backprojection.jl:137-138` = `dist_sq` and FDK weight ✓
- `siddon.jl:440-445` = volume bounds computation ✓

#### Angle formula consistency: VERIFIED ✓

Section 1.4 formula `angles = [2π × i / V for i in 0:total_views-1]` is equivalent to Section 1.10 range formula. Both give last_angle = `2πR - 2π/V`. For R=1, degenerates to current `2π - 2π/N` ✓

#### Forward projection claims: VERIFIED ✓

- Siddon ray tracer (siddon.jl:150-294): confirmed fully 3D, no orbit assumptions ✓
- `siddon_forward_project!`: reads all 3 components of per-angle arrays ✓
- Volume bounds centered at origin via `volume_extent` ✓
- PCCT path: same geometry pass-through ✓
- Physics pipeline: all effects are angle-agnostic ✓

#### Iterative recon: VERIFIED with NUANCE

SIRT/CGLS call `siddon_forward_project(recon, geom)` without `volume_extent` ✓
This uses `geom.fov` centered at origin — correct for `recon_center = (0,0,0)` but needs offset for non-zero recon_center (captured in C6 above).

---

### Cross-Section Contradictions

1. Section 0.4 says "BP scaling `π/n_angles` → `π/views_per_rotation`" but Section 3.3 says to use weight normalization instead. These are two DIFFERENT approaches (Phase 1 vs Phase 2). The spec should make clear that Section 0.4 describes the Phase 1 fix only.

2. Section 0.2 lists workspace.jl:514 as a change site, but Section 1.10 proposes handling n_angles→total_views computation INSIDE the CTGeometry constructor. If the constructor computes total_views internally, workspace.jl:514 needs NO change — `n_angles=protocol.views` stays as-is and the constructor handles the multiplication. These are contradictory approaches. Must pick one.

---

### Summary of Required Actions

| Issue | Severity | Action |
|-------|----------|--------|
| C1. Three normalization formulas | CRITICAL | Derive correct formula from WFBP integral, pick one |
| C2. Naive FDK pitch ≤ 1 claim | CRITICAL | Change to "pitch ≈ 1"; document overestimation for pitch < 0.8 |
| C3. Normalized ≠ standard FDK | CRITICAL | Correct reasoning; `is_helical` needed for normalization too |
| C4. 6 inner constructor sites | CRITICAL | List all 6 in spec |
| C5. Helical fov_z | CRITICAL | Add Tam-Danielsson fov_z computation to constructor |
| C6. recon_center in BP + affine | IMPORTANT | List all 4 offset sites |
| C7. PCCT workspace n_angles | IMPORTANT | Clarify constructor handles total_views |
| C8. Buffer contradiction | IMPORTANT | Remove weight_sum buffer from 3.3.6 |
| C9. Dual-energy + helical | IMPORTANT | Add out-of-scope note |
| C10. create_aquilion_one | IMPORTANT | Decide: update or delete |
| C11. Simplified q̂ | MINOR | Add Phase 2a/2b note |
| C12. is_helical derivation | MINOR | Specify derivation chain |

---

## Iteration 7: HELI-004 + HELI-005 CRITIQUE — API Design & Physics Pipeline

**Date:** 2026-03-17
**Phase:** Critique
**Topics:** HELI-004 (API Design, Section 4) + HELI-005 (Physics Pipeline, Section 5)

### Methodology

Cross-referenced every claim in Sections 4 and 5 of `helical-spec.md` against actual source code. Verified:
- All CTProtocol inner constructor call sites
- All CTGeometry inner constructor call sites (re-confirmed C4)
- `backproject!` and `backproject_voxel` code (actual `pi_over_angles` usage at `backprojection.jl:335,394`)
- `apply_lag!` and `apply_lag_catsim!` implementations (lag loop behavior at `detector_lag.jl:236-301, 535-643`)
- `apply_heel_effect!` implementation (fan angle from column index at `heel_effect.jl:175-196`)
- `siddon_forward_project!` volume bounds logic (two-path: phantom vs iterative at `siddon.jl:437-445`)
- FDK `reconstruct!` workspace path (`driver.jl:815-846`)
- `compute_ctdi_vol` and `compute_dlp` formulas (`protocol.jl:261-300`)

### New Issues Found

#### C13 (IMPORTANT): `n_angles` naming ambiguity — constructor param vs struct field

The CTGeometry constructor parameter `n_angles` means "views per rotation" (from `protocol.views`), but the struct field `n_angles` stores "total views" (= `views_per_rotation × n_rotations`). Example:

```
geom = CTGeometry(scanner; n_angles=984, pitch=0.8, n_rotations=3)
# geom.n_angles == 2952  (total views, not 984!)
# geom.views_per_rotation == 984
```

**Resolution:** Accept as design tradeoff. Add CLEAR docstring: "`n_angles` kwarg is views per rotation. The struct field `n_angles` = total views across all rotations."

#### C14 (IMPORTANT): `recon_center` ONLY applies to reconstruction volume bounds, not phantom projection

Spec C6 lists `siddon.jl:440-442` as needing `recon_center` offset. This is PARTIALLY correct.

`siddon_forward_project!` has TWO paths (siddon.jl:437-445):
```julia
vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
vol_min_z = T(-vol_bounds[3] / 2)
```

- **IF branch** (`volume_extent` provided): Phantom projection, centered at WORLD ORIGIN. NO `recon_center` offset.
- **ELSE branch** (`geom.fov`): Iterative recon projection. NEEDS `recon_center` offset.

Revised C6 site list:
| # | File:Line | Apply recon_center? | Reason |
|---|-----------|-------------------|--------|
| 1 | `siddon.jl:440-442` | ONLY in `geom.fov` path | Phantom stays at origin |
| 2 | `backprojection.jl:316-318` | YES always | FDK + matched BP use geom.fov |
| 3 | `affine.jl:69-71` | YES | recon_to_world_affine |
| 4 | `affine.jl:128-130` | YES | resample_to_recon |

#### C15 (IMPORTANT): `create_aquilion_one` duplicates geometry loop — refactor recommended

`create_aquilion_one` (scanner.jl:725-807) has its own geometry computation loop, duplicating CTGeometry constructor logic. Adding helical Z formula to BOTH places creates maintenance risk.

**Recommendation:** During helical implementation, refactor to:
```julia
function create_aquilion_one(; kwargs...)
    scanner = Scanner(source_to_isocenter=600.0, source_to_detector=1000.0, ...)
    return CTGeometry(scanner; n_angles, fov_cm, pitch, n_rotations, ...)
end
```
Eliminates duplicated geometry loop. Note: pixel_size derivation differs between the two functions — reconciliation needed during refactor.

#### C16 (IMPORTANT): GPU memory scaling not documented in Section 4.5

Section 4.5 says "No struct changes" for workspaces but doesn't note that sinogram-sized buffers scale with n_rotations. EICTWorkspace has ~8 sino-sized buffers; PCCTWorkspace has ~5 per energy bin. Practical limit: ~5 rotations on 24GB GPU for clinical geometry. Section 2.4 has memory table but Section 4.5 should cross-reference it.

#### C17 (MINOR): `compute_dlp` needs `scan_length_cm` semantics clarification

Spec proposes removing `* n_rotations` from DLP. Correct, but docstring should clarify:
- Helical: `scan_length_cm` = total z-travel = `n_rotations × pitch × total_collimation_cm`
- Axial: `scan_length_cm` = z-coverage per rotation. `n_rotations` = 1 default.

#### C18 (MINOR): `fan_angle_max` in heel effect uses recon FOV, not detector fan

`heel_effect.jl:176`: `fan_angle_max = atan(geom.fov[1] / 2 / geom.SAD)` — uses reconstruction FOV, not actual detector fan coverage. Pre-existing issue, not helical-specific. Doesn't affect helical compatibility.

### Verified Claims (ALL CORRECT)

**Section 4:**
- ✅ CTProtocol: 3 inner constructor call sites (protocol.jl:118, 418, 461)
- ✅ CTGeometry: 6 inner constructor call sites (scanner.jl:694,802; fdk.jl:426; workspace.jl:364; mbir.jl:434; scanners.jl:327)
- ✅ n_angles semantics: total_views in struct, correct for backward compat
- ✅ Workspace creation: only pitch/n_rotations kwargs needed
- ✅ simulate!(): no changes — confirmed all paths use ws.geom
- ✅ backproject! has access to geom.pitch and geom.views_per_rotation
- ✅ FDK reconstruct! path flows geometry correctly
- ✅ PCCT + helical: identical creation path, all detector physics angle-agnostic

**Section 5:**
- ✅ ALL 16 physics effects verified — zero changes needed
- ✅ Heel effect: fan angle from column index + SAD, no Z dependency (heel_effect.jl:175-196)
- ✅ Detector lag (apply_lag!): `prev_angle = angle - k`, clamp to frame 1 (detector_lag.jl:285-289) — correct for multi-rotation
- ✅ Detector lag (apply_lag_catsim!): sequential IIR state across views (detector_lag.jl:602-640) — correct
- ✅ Bowtie/flat filter: 2D maps, uniform per view
- ✅ Scatter: convolution approximation, same quality axial/helical
- ✅ Noise: `compute_detector_I0` uses `protocol.views` (per-rotation) — correct
- ✅ PCCT detector physics: per-pixel, no geometry references

### Summary

| Issue | Severity | Section | Action |
|-------|----------|---------|--------|
| C13: n_angles naming | IMPORTANT | 4.2, 4.13 | Add clear docstring; accept tradeoff |
| C14: recon_center conditional | IMPORTANT | 4.7, C6 | Clarify phantom path stays at origin |
| C15: create_aquilion_one duplication | IMPORTANT | 4.11 | Refactor to delegate to CTGeometry(scanner;...) |
| C16: GPU memory scaling | IMPORTANT | 4.5 | Cross-reference Section 2.4 |
| C17: DLP scan_length semantics | MINOR | 4.9 | Clarify docstring |
| C18: heel fan_angle uses FOV | MINOR | 5.3 | Pre-existing; note for future |

**Overall: Sections 4 and 5 are SOLID.** No critical issues. API design is clean. Physics pipeline confirmed 100% helical-compatible. Important issues are clarification and code hygiene, not correctness.

---
