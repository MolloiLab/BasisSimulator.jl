# BasisSimulator.jl — Helical CT Specification

> **Goal:** Add helical (spiral) CT scanning to BasisSimulator.jl, seamlessly
> integrated with the existing axial scan, physics pipeline, workspace pattern,
> and GPU-native architecture.
>
> **Key principle:** Helical is a GENERALIZATION of axial. Axial = helical with
> pitch=0, n_rotations=1. The same code paths handle both cases.
>
> **Current state:** Axial scanning works end-to-end with 16-step physics,
> FDK + iterative reconstruction, PCCT, dual-energy. CTProtocol has n_rotations
> field (unused). CTGeometry hardcodes Z=0 for all angles.

---

## 0. Current Geometry Audit — Where Z=0 Is Hardcoded

**Status: COMPLETE (HELI-000 Discovery, 2026-03-17)**

### 0.1 Architecture Summary

The codebase uses a position-based geometry architecture:
- `CTGeometry` pre-computes `source_positions[3, n_angles]` and `detector_centers[3, n_angles]`
- All downstream code (Siddon, polychromatic, physics, backprojection) reads these per-angle arrays
- **This means helical support requires changing geometry CONSTRUCTION, not CONSUMPTION**

### 0.2 Z=0 Hardcoded Locations (MUST CHANGE)

| Location | File:Line | What's Hardcoded |
|----------|-----------|-----------------|
| Source Z | `scanner.jl:673` | `source_positions[3, i] = 0.0` |
| Detector center Z | `scanner.jl:679` | `detector_centers[3, i] = 0.0` |
| Angles range | `scanner.jl:658` | `range(0, 2π - 2π/n_angles)` — exactly 1 rotation |
| `create_aquilion_one` source Z | `scanner.jl:780` | `source_positions[3, i] = 0.0` |
| `create_aquilion_one` det Z | `scanner.jl:786` | `detector_centers[3, i] = 0.0` |
| `create_aquilion_one` angles | `scanner.jl:765` | Same 1-rotation range |
| Workspace n_angles | `workspace.jl:514` | `n_angles=protocol.views` (1 rotation) |
| CTProtocol missing field | `protocol.jl:35–49` | No `pitch` field |

### 0.3 Already General — No Changes Needed

| Component | File | Why |
|-----------|------|-----|
| `CTGeometry` struct | `scanner.jl:552–566` | Stores `[3, n_angles]` matrices — can hold any Z |
| `Scanner` struct | `scanner.jl:127–179` | Physical params only, no trajectory |
| `siddon_trace_ray` | `siddon.jl:150–294` | Takes arbitrary `(src, det)` positions |
| `siddon_forward_project!` | `siddon.jl:414–538` | Reads per-angle geometry arrays |
| `_forward_project_poly!` | `polychromatic.jl:1100+` | Calls Siddon in energy loop |
| `cosine_weight!` | `filtering.jl:362–404` | Per-pixel detector geometry |
| `filter_sinogram!` | `filtering.jl:433–504` | Row-by-row, angle-independent |
| `backproject_voxel_matched` | `backprojection.jl:159–264` | No FDK weight, arbitrary geometry |
| All 16 physics effects | `physics_pipeline.jl` | Angle-agnostic operations |
| `simulate!()` driver | `driver.jl:330–470` | Passes through workspace geometry |
| SIRT/CGLS iterative recon | `ir/sirt.jl`, `ir/cgls.jl` | Use matched backprojection (no FDK weight) |

### 0.4 Needs Algorithmic Addition (Reconstruction Only)

| Component | File:Line | What Changes |
|-----------|-----------|-------------|
| FDK weight | `backprojection.jl:137–138` | `SAD²/dist²` → helical weight function |
| BP scaling | `backprojection.jl:335` | `π/n_angles` → `π/views_per_rotation` |
| `fdk_reconstruct` | `fdk.jl:327–346` | Add helical weighting step |
| Affine Z-center | `affine.jl:61–79` | Optional Z-offset for recon volumes |

### 0.5 Key Architectural Insight

**The codebase is ~90% helical-ready.** The decision to pre-compute per-angle position matrices means:
1. **Forward projection** (Siddon + polychromatic + 16 physics effects) works UNCHANGED
2. **Iterative reconstruction** (SIRT, CGLS) works UNCHANGED (matched backprojection)
3. **PCCT pipeline** works UNCHANGED (same geometry flow)
4. Only **geometry construction** and **FDK reconstruction** need modification

The `n_rotations` field already exists in `CTProtocol` (default 1.0, currently used only for dose calculation). Only `pitch` is missing.

---

## 1. Helical CT Geometry — Table Motion and Pitch

**Status: COMPLETE (HELI-001 Discovery, 2026-03-17)**

### 1.1 Pitch Definition (IEC 60601-2-44 / DICOM PS3.3 C.8.15.3.4)

The IEC standard defines **Spiral Pitch Factor** (DICOM tag 0018,9311):

```
pitch = table_feed_per_rotation / total_collimation_width
```

Where:
- **table_feed_per_rotation** = table_speed × rotation_time [mm]
- **total_collimation_width** = n_active_rows × single_collimation_width [mm]

This is the universally accepted "beam pitch" definition. The deprecated "detector pitch" (= N × beam_pitch) should NOT be used.

**Key relationships:**
```
table_speed        = pitch × total_collimation_width / rotation_time    [mm/s]
table_feed_per_rot = pitch × total_collimation_width                    [mm]
total_scan_length  = table_feed_per_rot × n_rotations                   [mm]
total_views        = views_per_rotation × n_rotations
```

**Ref:** IEC 60601-2-44:2009; Hsieh, "Computed Tomography," 3rd ed., SPIE Press, Ch. 8–9.

### 1.2 Z(θ) Position Formula

For a helical scan with constant pitch and rotation speed, the source and detector Z-positions are:

```
Z(θ) = Z_start + (θ / 2π) × pitch × total_collimation_cm
```

Where `θ` is the **cumulative** rotation angle (monotonically increasing, NOT wrapped to [0, 2π]).

For view index `i` with `views_per_rotation` V:
```
θ_i = 2π × i / V          for i = 0, 1, ..., total_views - 1
Z_i = Z_start + (i / V) × pitch × total_collimation_cm
```

**Both source AND detector translate by the same Z_i.** The gantry is a rigid unit — source and detector are at different XY positions (rotating) but the same Z (translating together). The detector_u vector stays in the XY plane; detector_v stays as (0,0,1).

**Sign convention:** In BasisSimulator, Z = inferior→superior. Table motion moves the patient in -Z → equivalent to source/detector translating in +Z. We choose Z increasing with θ.

**Ref:** Hsieh, "Computed Tomography," 3rd ed., Ch. 8, Eq. 8.1–8.3.

### 1.3 Z_start — Centering the Scan

For a scan centered at Z=0 (symmetric about isocenter):

```
z_travel = n_rotations × pitch × total_collimation_cm
z_start  = -z_travel / 2
z_end    = +z_travel / 2
```

For axial (pitch=0): `z_start = z_end = 0.0` → all views at Z=0. This is the correct degenerate case.

### 1.4 Multi-Rotation Angle Generation

For `n_rotations` R with `views_per_rotation` V:

```
total_views = round(Int, R × V)
angles = [2π × i / V  for i in 0 : total_views - 1]
```

These angles are **monotonically increasing** from 0 to R×2π. This differs from TIGRE (which concatenates wrapped [0,2π) arrays) and is more natural for the Z(θ) computation.

For axial (R=1): `total_views = V`, angles span [0, 2π), same as current code.

### 1.5 Collimation and Active Detector Rows

The existing CTGeometry constructor already handles collimation:

```julia
# scanner.jl:619-628 — current code
if collimation_mm !== nothing
    _n_rows = round(Int, collimation_mm / scanner.detector_row_size)
else
    _n_rows = n_rows !== nothing ? n_rows : scanner.detector_rows
end
```

**Total collimation width** for pitch calculation:
```
total_collimation_mm = _n_rows × scanner.detector_row_size    [mm at isocenter]
total_collimation_cm = total_collimation_mm / 10.0             [cm, internal units]
```

This correctly uses the **active** rows (from collimation or explicit n_rows), not the full scanner detector.

### 1.6 Clinical Pitch Values by Scanner

| Scanner | Detector | Total Collimation | Typical Pitch | Primary Mode |
|---------|----------|-------------------|---------------|-------------|
| **Siemens NAEOTOM Alpha** | 144×0.4mm | 57.6 mm | 0.8 (routine) | Helical |
| — Turbo Flash (dual-source) | | | 3.0–3.4 | Cardiac |
| **GE Revolution Apex Elite** | 256×0.625mm | 160 mm | 0.516, 0.984, 1.375 | Axial or Helical |
| **Canon Aquilion ONE** | 320×0.5mm | 160 mm | 0.813–1.484 | Axial volumetric |

**Clinical pitch ranges by application:**

| Application | Pitch Range | Rationale |
|-------------|------------|-----------|
| Head/brain | 0.5–0.75 | Low pitch for z-resolution |
| Routine body | 0.8–1.0 | Balance speed/quality |
| CTA (angiography) | 0.8–1.2 | Fast coverage |
| Lung screening | 1.0–1.5 | Fast, low dose |
| Trauma | 1.2–1.5 | Maximum speed |
| Cardiac retrospective | 0.2–0.3 | Oversampling for phase gating |
| Dual-source high-pitch | 3.0–3.4 | Single-heartbeat |

**Ref:** Siemens NAEOTOM Alpha — Konrad et al. 2025 (PMB 70:065004), DukeSim (PMC9125732); GE Revolution — project verification notebook `06_ge_apex_elite_clinical.jl`.

### 1.7 Phantom Z-Coverage Requirement

For helical, the phantom must cover the full scan travel range PLUS half the detector Z-coverage on each end (otherwise rays at the extremes miss the phantom):

```
detector_half_z = total_collimation_cm / 2
phantom_z_min  ≤  z_start - detector_half_z
phantom_z_max  ≥  z_end   + detector_half_z
```

For XCAT at factor 2 (extent 32×28×10 cm, z_extent = 10 cm):
- NAEOTOM Alpha at pitch=0.8, collimation 57.6mm = 5.76cm:
  - z_travel for 3 rotations: 3 × 0.8 × 5.76 = 13.82 cm → z_start = -6.91, z_end = +6.91
  - Phantom needs: z_min ≤ -6.91 - 2.88 = -9.79, z_max ≥ +9.79
  - XCAT z_extent is 10 cm (±5cm) → **barely sufficient for 3 rotations at pitch 0.8**

### 1.8 Reconstruction Z-Range (Fully Sampled Region)

For helical FDK, the fully-sampled region (where every voxel sees at least π + fan_angle of data) is narrower than the scan range:

```
z_recon_min = z_start + total_collimation_cm / 2
z_recon_max = z_end   - total_collimation_cm / 2
z_recon_range = z_travel - total_collimation_cm
```

This is the "Tam-Danielsson window" concept — data outside this range has incomplete angular sampling and produces artifacts.

The reconstruction volume should default to this fully-sampled Z-range, but the user can override it.

### 1.9 Reconstruction Volume Z-Centering

**Current code** (backprojection.jl:316-318):
```julia
vol_min_z = T(-geom.fov[3] / 2)   # Recon grid centered at Z=0
```

For helical, the reconstruction may need to be centered at a Z-position other than 0 (e.g., to reconstruct a specific anatomy). Two approaches:

**Option A: Add `recon_center_z` to CTGeometry** — shift the volume origin. Simplest.

**Option B: Add `recon_center_z` to ReconOptions** — more flexible, decoupled from geometry.

**Recommendation: Option A.** The recon center is a geometric property. Add `recon_center::NTuple{3,Float64}` to CTGeometry (defaults to (0,0,0)). The backprojection computes:
```julia
vol_min_z = T(recon_center[3] - geom.fov[3] / 2)
```
For axial, `recon_center = (0,0,0)` → identical to current behavior.

### 1.10 CTGeometry Constructor — Proposed Changes

The constructor needs THREE new keyword arguments:

```julia
function CTGeometry(scanner::Scanner;
    n_angles::Int = 360,              # views per rotation (UNCHANGED)
    pitch::Float64 = 0.0,            # NEW: IEC beam pitch (0.0 = axial)
    n_rotations::Float64 = 1.0,      # NEW: number of gantry rotations
    recon_center_z_cm::Float64 = 0.0, # NEW: reconstruction Z center
    fov_cm=nothing, z_cm=nothing,
    n_rows=nothing, n_cols=nothing,
    collimation_mm=nothing
)
```

**Changes to the loop (scanner.jl:658–690):**

```julia
# Compute collimation for pitch calculation
total_collim_cm = _n_rows * scanner.detector_row_size / 10.0  # mm → cm

# Total views
total_views = if pitch > 0
    round(Int, n_angles * n_rotations)
else
    n_angles  # axial: exactly 1 rotation
end

# Z travel
z_travel = n_rotations * pitch * total_collim_cm
z_start = -z_travel / 2

# Generate angles (monotonically increasing)
angles = collect(range(0.0, 2π * n_rotations - 2π * n_rotations / total_views,
                       length=total_views))

# Allocate position matrices (total_views, not n_angles)
source_positions = Matrix{Float64}(undef, 3, total_views)
detector_centers = Matrix{Float64}(undef, 3, total_views)
detector_u = Matrix{Float64}(undef, 3, total_views)
detector_v = Matrix{Float64}(undef, 3, total_views)

for (i, θ) in enumerate(angles)
    cosθ = cos(θ)
    sinθ = sin(θ)

    # Z position from helical formula
    z_i = z_start + (θ / (2π)) * pitch * total_collim_cm  # 0.0 for axial

    source_positions[1, i] = -SAD * sinθ
    source_positions[2, i] = -SAD * cosθ
    source_positions[3, i] = z_i

    det_dist = SDD - SAD
    detector_centers[1, i] = det_dist * sinθ
    detector_centers[2, i] = det_dist * cosθ
    detector_centers[3, i] = z_i  # Same Z as source

    # u-axis: in XY plane (unchanged)
    detector_u[1, i] = cosθ
    detector_u[2, i] = -sinθ
    detector_u[3, i] = 0.0

    # v-axis: always +Z (unchanged)
    detector_v[1, i] = 0.0
    detector_v[2, i] = 0.0
    detector_v[3, i] = 1.0
end
```

**Key insight:** For pitch=0 (axial), `z_i = 0.0` for all angles → identical to current behavior.

### 1.11 CTGeometry Struct — Proposed Additions

```julia
struct CTGeometry
    # ... existing fields ...
    fov::NTuple{3, Float64}
    # NEW fields:
    pitch::Float64                    # IEC beam pitch (0.0 = axial)
    views_per_rotation::Int           # n_angles per single rotation
    recon_center::NTuple{3, Float64}  # (x, y, z) center of recon volume in cm
end
```

`views_per_rotation` is needed by the backprojection for the correct scaling factor (`π / views_per_rotation` instead of `π / n_angles`).

### 1.12 Comparison with Reference Implementations

| Aspect | **BasisSimulator (proposed)** | **TIGRE** | **XCIST/CatSim** |
|--------|------------------------------|-----------|-------------------|
| Helical parameter | `pitch` (IEC standard) | None — user sets Z offsets | `tableSpeed` (mm/s) |
| Z computation | In CTGeometry constructor | User responsibility | In `Gantry_Helical.py` via 4×4 transforms |
| Multi-rotation | `n_rotations` field | Concatenate angle arrays | `viewCount / viewsPerRotation` |
| Angle convention | Monotonic 0→R×2π | Wrapped [0,2π) × R times | Monotonic (from time) |
| Helical recon | FDK+weights (planned) | Iterative only (FDK fails) | `helical_equiAngle` rebinning+FBP |
| Gantry tilt | Not supported (future) | Via offsets | Full 4×4 transform |

**BasisSimulator's approach is cleanest:** `pitch` is the standard clinical parameter, and deriving Z(θ) internally keeps the API simple. Users don't need to compute Z-offsets manually (TIGRE) or derive table speeds from pitch (XCIST).

---

## 2. Forward Projection — Siddon with Table Motion

**Status: COMPLETE (HELI-002 Discovery, 2026-03-17)**

### 2.1 Core Finding: Forward Projection Is ALREADY Helical-Ready

The entire forward projection pipeline — Siddon ray tracing, polychromatic Beer-Lambert, and PCCT photon-counting — works **UNCHANGED** for helical scanning. No code modifications are needed.

This is because:
1. The Siddon ray tracer (`siddon_trace_ray`, `siddon.jl:150–294`) takes arbitrary 3D source and detector positions as inputs. It uses parametric ray representation `P(t) = S + t(D-S)` and performs 3D-DDA traversal. No orbit shape is assumed.
2. The high-level projector (`siddon_forward_project!`, `siddon.jl:414–538`) reads per-angle positions from `CTGeometry` arrays: `source_positions[1..3, angle]`, `detector_centers[1..3, angle]`, `detector_u[1..3, angle]`, `detector_v[1..3, angle]`. All three coordinates are used. The only assumption is that the phantom volume is centered at the world origin — which remains true for helical (the phantom is stationary; the gantry translates in Z).
3. The polychromatic pipeline (`_forward_project_poly!`, `polychromatic.jl:1100–1195`) loops over energies and calls `siddon_forward_project!` at each energy. It accumulates Beer-Lambert: `I = Σ w_e × exp(-L_e)`. This is purely ray-based and orbit-agnostic.
4. The PCCT pipeline (`pcct_forward_project`, `photon_counting.jl:1192–1440`) follows the same pattern — energy loop → Siddon → spectral binning. Detector physics (charge sharing, pileup, anti-coincidence) are per-pixel/angle-agnostic.

### 2.2 Volume Bounds and Phantom Centering

**Phantom forward projection** (used by `simulate!`):

The projector receives the phantom's physical dimensions via the `volume_extent` kwarg (`siddon.jl:424`):
```julia
vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
vol_min_z = T(-vol_bounds[3] / 2)     # phantom centered at origin
vol_max_z = T(vol_bounds[3] / 2)
```

For helical, the phantom is **stationary at the world origin**. The source and detector translate in Z per the helical formula `Z(θ)`. Rays from source positions at Z ≠ 0 correctly intersect the phantom volume because the Siddon ray tracer performs full 3D intersection testing — it finds where each ray enters/exits the volume regardless of ray origin.

**No changes needed.** The `volume_extent` kwarg already decouples phantom physical size from reconstruction FOV (this was the fix from 2026-02-11 — see memory).

**Iterative reconstruction forward projection** (used by SIRT/CGLS):

SIRT (`sirt.jl:53,125`) and CGLS (`cgls.jl:93,207`) call `siddon_forward_project(recon_volume, geom)` **without** the `volume_extent` kwarg. This defaults to `geom.fov` for volume bounds, centered at the origin.

For helical with the proposed `recon_center` field (section 1.9), the forward projector needs to offset the volume bounds by `recon_center`. This is a **reconstruction concern** (the projector needs to know where the recon grid sits in world coordinates). The change is minimal:
```julia
# In siddon_forward_project! — when using geom.fov (no volume_extent override):
vol_min_z = T(-vol_bounds[3] / 2 + geom.recon_center[3])
vol_max_z = T(+vol_bounds[3] / 2 + geom.recon_center[3])
```
For axial, `recon_center = (0,0,0)` → identical to current behavior.

### 2.3 GPU Kernel Parallelism — Perfect Scaling

The Siddon GPU kernel (`siddon.jl:490–534`) parallelizes over ALL rays:
```julia
AK.foreachindex(sinogram) do idx
    # Convert linear index to (col, row, angle)
    # ... trace single ray ...
end
```

For helical with `n_rotations` R:
- Total work items: `n_cols × n_rows × (R × views_per_rotation)`
- Each ray is **independent** — no inter-angle dependencies
- GPU utilization scales linearly with total views
- No kernel code changes needed

For a clinical example (NAEOTOM Alpha, pitch=0.8, 5 rotations):
- 900 cols × 64 rows × (5 × 720 views) = 207M rays
- Each ray traces ~300 voxels → ~60B voxel accesses
- Within GPU throughput capacity for modern hardware

### 2.4 Memory Implications

Sinogram size scales linearly with `n_rotations`:
```
sinogram_bytes = n_cols × n_rows × total_views × sizeof(T)
```

| Config | Views | Sinogram Size | ×8 Buffers (EICTWorkspace) |
|--------|-------|---------------|---------------------------|
| Axial (1 rot) | 720 | 166 MB | 1.3 GB |
| 3 rotations | 2160 | 498 MB | 3.9 GB |
| 5 rotations | 3600 | 830 MB | 6.5 GB |
| 10 rotations | 7200 | 1.66 GB | 13.0 GB |

*(Assumes 900 cols × 64 rows × Float32)*

For long helical scans (>10 rotations), GPU memory may be limiting. This is an **optimization concern** for future work (e.g., chunked projection, sliding-window reconstruction). The initial implementation should work straightforwardly for ≤5 rotations on 24+ GB GPUs.

### 2.5 Phantom Z-Coverage Validation

For correct helical simulation, the phantom must be large enough in Z to cover the scan range. If the source is at Z = +6 cm but the phantom only extends to Z = +5 cm, off-center detector rows will see "air" (zero attenuation) — which is physically correct behavior but may surprise users.

**Recommended validation** (in workspace creation):
```julia
z_travel = n_rotations * pitch * total_collimation_cm
z_scan_extent = z_travel + total_collimation_cm  # scan range + detector coverage
if phantom.extent[3] < z_scan_extent
    @warn "Phantom Z-extent ($(phantom.extent[3]) cm) < helical scan extent ($z_scan_extent cm). " *
          "Outer views will project through empty space."
end
```

This is a warning, not an error — projecting through empty space produces correct (zero) sinogram values.

### 2.6 I₀ and Noise — No Changes Needed

`compute_detector_I0` (`detector_noise.jl:457–473`) computes photons per pixel per view:
```julia
time_per_view = protocol.rotation_time / protocol.views
I₀ = spectrum_flux_sum × mA × time_per_view × pixel_area_mm²
```

`protocol.views` is **views per rotation** (not total views). The time per view is `rotation_time / views_per_rotation`, which is the correct physical quantity regardless of how many rotations occur. The total number of views in the sinogram doesn't affect the per-view photon count.

For the noise model in `simulate!` (`driver.jl:480–520`), the noise is applied per-element of the sinogram using the same I₀ for all views. This is physically correct — each view has the same tube current and integration time, regardless of helical position.

### 2.7 PCCT + Helical — Works Unchanged

The PCCT forward projection path (`pcct_forward_project`, `photon_counting.jl:1192–1440`) follows the same structure as the EICT path:

1. Energy loop → `create_μ_volume!` → `siddon_forward_project!` → spectral binning
2. Detector physics at native dexel resolution (charge sharing, pileup, anti-coincidence)
3. Spatial binning (native → binned) via `spatial_bin!`
4. Combination via `_combine_pcct_bins`

All of these steps read geometry from `proj_geom` (either native or binned CTGeometry) and operate per-pixel. **None have orbit assumptions.** The PCCT pipeline works identically for helical — the only change is that the CTGeometry has helical Z-positions, which propagate through automatically.

The native-resolution geometry (`native_geom`) is constructed from the binned geometry in `create_workspace`. For helical, both geometries share the same source/detector Z-positions (they differ only in detector pixel grid resolution). No special handling needed.

### 2.8 Summary: Forward Projection Changes for Helical

| Component | Change Required | Reason |
|-----------|----------------|--------|
| `siddon_trace_ray` | **NONE** | Already takes arbitrary 3D positions |
| `siddon_forward_project!` | **MINOR** — add `volume_center` support for iterative recon | Recon volume may be centered at Z ≠ 0 |
| `_forward_project_poly!` | **NONE** | Passes through geometry |
| `pcct_forward_project` | **NONE** | Same geometry pass-through |
| `compute_detector_I0` | **NONE** | Uses views_per_rotation correctly |
| Noise model | **NONE** | Per-view, independent of orbit |
| Workspace buffers | **SIZE ONLY** — larger sinograms for multi-rotation | Automatic from `geom.n_angles` |

**The forward projection is the easiest part of helical CT integration.** The architecture decision to pre-compute per-angle position matrices made this essentially free.

---

## 3. Helical Reconstruction Algorithms

(To be filled by HELI-003 research)

---

## 4. API Design — Seamless Helical in the 5-Part API

(To be filled by HELI-004 research)

---

## 5. Physics Pipeline Compatibility

(To be filled by HELI-005 research)

---

## 6. Reference Implementations

(To be filled by HELI-006 research)

---

## 7. Validation and Testing Strategy

(To be filled by HELI-007 research)

---

## 8. SYNTHESIS: Implementation Roadmap

(To be filled by HELI-008 synthesis — blocked until all other topics complete)

---
