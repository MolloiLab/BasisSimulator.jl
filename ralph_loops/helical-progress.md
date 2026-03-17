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
