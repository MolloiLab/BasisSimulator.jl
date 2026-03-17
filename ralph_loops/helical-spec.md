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

> **⚠ CRITIQUE C6 (refined by C14):** The `recon_center` offset applies to
> reconstruction volume bounds, NOT phantom forward projection. The `siddon_forward_project!`
> function has two paths: (1) `volume_extent` provided → phantom centered at origin, NO offset;
> (2) `geom.fov` → iterative recon, NEEDS offset. Updated site list:
>
> | # | File:Line | Apply recon_center? | Code Changed |
> |---|-----------|-------------------|-------------|
> | 1 | `siddon.jl:440-442` | ONLY in `geom.fov` path | `vol_min_z` for iterative forward projector |
> | 2 | `backprojection.jl:316-318` | YES always | `vol_min_z` for FDK + matched backprojection |
> | 3 | `affine.jl:69-71` | YES | `tz` in `recon_to_world_affine` |
> | 4 | `affine.jl:128-130` | YES | `roz` in `resample_to_recon` |

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

> **⚠ CRITIQUE C5:** The proposed constructor changes above do NOT modify the `fov_z`
> computation for helical. Currently `fov_z = n_rows × row_size` (single-rotation
> detector coverage). For helical, the reconstruction z-FOV should default to the
> Tam-Danielsson window: `fov_z = collim × (n_rotations × pitch - 1)`. Add:
> ```julia
> if pitch > 0 && z_cm === nothing
>     z_travel = n_rotations * pitch * total_collim_cm
>     fov_z = max(z_travel - total_collim_cm, total_collim_cm)
> end
> ```

> **⚠ CRITIQUE C7:** The `n_angles` parameter should retain its meaning as
> views_per_rotation. The constructor internally computes
> `total_views = round(Int, n_angles * n_rotations)` when pitch > 0. This means
> workspace call sites (`create_eict_workspace`, `create_workspace`) need NO changes
> — they continue passing `n_angles=protocol.views` and the constructor handles
> the multiplication. The spec currently says workspace.jl:514 needs changing,
> but if the constructor handles it internally, this is unnecessary.

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

> **⚠ CRITIQUE C4:** Adding fields to CTGeometry breaks **6 inner constructor call sites**
> (not just 2). All must be updated to pass the new fields:
>
> | # | File:Line | Context |
> |---|-----------|---------|
> | 1 | `scanner.jl:694` | Main CTGeometry constructor return |
> | 2 | `scanner.jl:802` | `create_aquilion_one` return |
> | 3 | `fdk.jl:426` | FOV-override variant (copies all fields) |
> | 4 | `workspace.jl:364` | PCCT native-resolution geometry |
> | 5 | `mbir.jl:434` | Ordered-subset geometry slicing |
> | 6 | `scanners.jl:327` | Scanner factory `create_geometry` |
>
> Site #5 (MBIR) is subtle: `views_per_rotation` stays the same when subsetting
> angles. Site #4 (PCCT native geom) copies pitch/views_per_rotation from parent.

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

### 2.9 Dual-Energy + Helical (Out of Scope)

> **⚠ CRITIQUE C9:** Dual-energy (dual-kVp alternation) + helical is not addressed in
> the initial implementation. The kVp alternation pattern works geometrically (views at
> different z-positions have alternating kVp). Forward projection and noise model are
> unaffected. However, material decomposition validation with z-varying kVp alternation
> is deferred to a future iteration. For initial helical work, assume single-kVp only.

---

## 3. Helical Reconstruction Algorithms

**Status: COMPLETE (HELI-003 Discovery, 2026-03-17)**

### 3.1 Overview: Four Classes of Helical CT Reconstruction

| Algorithm Class | Type | Quality | Speed | Implementation Difficulty |
|----------------|------|---------|-------|---------------------------|
| **Naive Helical FDK** | Approximate | Adequate (pitch ≤ 1, cone < 5°) | Fast | Minimal (scaling fix only) |
| **Weighted Helical FDK (WFBP)** | Approximate | Clinical quality | Fast | Moderate |
| **Katsevich** | Exact | Artifact-free (theoretical) | Moderate | High |
| **Rebinning (ASSR)** | Approximate | Good (narrow detectors) | Fastest | Moderate |

**Recommendation: Implement Phase 1 (Naive) then Phase 2 (Weighted WFBP). Skip Katsevich and ASSR.**

**Rationale:**
- **No major clinical vendor uses Katsevich** (Ref: Hsieh 2021, "History of CT reconstruction algorithms"). WFBP is what Siemens uses for ALL scanners including NAEOTOM Alpha. GE uses FDK-type with proprietary weights. Canon uses ASSR-type rebinning.
- Katsevich uses only ~73% of measured data ("1-pi method") → **higher noise** than WFBP (100% utilization).
- For BasisSimulator's target detectors (NAEOTOM 3–4° cone, GE Apex 5–6° cone), WFBP cone-beam artifacts are clinically negligible.
- Iterative methods (SIRT/CGLS) already work unchanged for helical — they're the fallback for artifact-sensitive applications.

### 3.2 Naive Helical FDK (Phase 1 — Minimal Changes)

**What it is:** Use the EXISTING FDK backprojection kernel on helical data, with only the scaling factor corrected.

**Why it works:** The `backproject_voxel` function (`backprojection.jl:37–146`) already:
- Reads per-angle source positions in full 3D (including Z)
- Projects each voxel onto the detector using arbitrary geometry
- Applies bilinear interpolation
- Checks detector bounds (out-of-range voxels contribute zero)
- Applies FDK weight `SAD²/dist²`

For helical, the source/detector Z varies per angle. The backprojection correctly handles this because it computes the source→voxel→detector projection geometrically, not analytically. Voxels that fall outside the detector for a given angle automatically contribute zero.

**The ONLY change needed:**

```julia
# CURRENT (backprojection.jl:335):
pi_over_angles = T(π) / T(n_angles)  # n_angles = total views (multi-rotation)

# HELICAL FIX:
pi_over_angles = T(π) / T(views_per_rotation)
```

**Why:** The FDK integral is `(1/2) ∫₀²π ... dθ` for one rotation. Discretized: `(π/N_one_rot) × Σ`. For multi-rotation helical, each voxel is "seen" by approximately one rotation's worth of views (the rest have zero contribution because the voxel is outside the detector). The scaling must use `views_per_rotation`, not `total_views`.

**For axial (pitch=0, 1 rotation):** `views_per_rotation = n_angles` → identical to current.

**Quality characteristics:**
- Works well for **pitch ≈ 1** (within ±0.1) and **cone half-angle < 5°**
- **⚠ CRITIQUE C2:** For pitch < 0.8, voxels see >1 rotation of data → overestimation by ~1/pitch. For pitch=0.5, expect ~2× overbrightness. Use Phase 2 (WFBP) for pitch < 0.8.
- At the Z-edges of the reconstruction, voxels near the detector boundary will have slightly lower intensity (fewer contributing views)
- No handling of data redundancy for pitch < 1 (overlapping coverage)
- Adequate for initial validation and prototyping at pitch ≈ 1

**GPU kernel changes: NONE.** Only the scalar `pi_over_angles` changes.

### 3.3 Weighted Helical FDK / WFBP (Phase 2 — Clinical Quality)

**What it is:** The Stierstorfer WFBP algorithm — the same algorithm used by all Siemens clinical scanners, implemented as a modification to the FDK backprojection.

**Ref:** Stierstorfer K et al., "Weighted FBP — a simple approximate 3D FBP algorithm for multislice spiral CT with good dose usage for arbitrary pitch," Phys. Med. Biol. 49(11):2209–2218, 2004. DOI: 10.1088/0031-9155/49/11/007

**Ref (implementation):** Hoffman J et al., "Technical Note: FreeCT_wFBP: A robust, efficient, open-source implementation of weighted filtered backprojection for helical, fan-beam CT," Med. Phys. 43(3):1411–1420, 2016. PMID: 26936725. Source: github.com/xiehq/FreeCT_wFBP

#### 3.3.1 Algorithm Overview

WFBP has three stages:

1. **Fan-to-parallel rebinning** (sinogram preprocessing)
2. **1D ramp filtering** along detector columns (standard — we already have this)
3. **3D cone-angle-weighted backprojection** (modified backprojection kernel)

The key innovation is the **normalized cone-angle weight W(q̂)** that handles data redundancy by favoring rays with small cone angles and down-weighting rays with large cone angles. This is applied per-voxel per-view during backprojection.

#### 3.3.2 The Normalized Cone Angle Parameter q̂

For a voxel at world position `(x, y, z)` and a parallel-beam view at angle `α` with table position `z_table`:

```
p̂ = x sin(α) − y cos(α)                          # perpendicular distance to central ray
l̂ = √(SAD² − p̂²) − x cos(α) − y sin(α)           # distance along ray to voxel
q̂ = (z_voxel − z_table + (z_rot/(2π)) arcsin(p̂/SAD)) / (l̂ × tan(θ_cone/2))
```

Where:
- `SAD` = source-to-isocenter distance
- `z_rot` = table feed per gantry rotation (= pitch × total_collimation)
- `θ_cone` = full cone angle of the detector = 2 × arctan(detector_z_extent / (2 × SDD))
- The term `(z_rot/(2π)) arcsin(p̂/SAD)` is the **helical interpolation correction** — it accounts for the z-offset between the voxel plane and the oblique plane defined by the helical source trajectory

**q̂ ranges from −1 to +1**, where |q̂| = 1 means the ray reaches the detector edge. For axial (z_rot=0), q̂ depends only on the fixed v-position of the voxel on the detector.

**Ref:** FreeCT_wFBP `cuda_kernels.cuh`, backprojection kernel.

#### 3.3.3 The Weight Function W(q̂)

```
         ⎧ 1                                        if |q̂| < Q
W(q̂) =  ⎨ cos²(π/2 × (|q̂| − Q) / (1 − Q))        if Q ≤ |q̂| < 1
         ⎩ 0                                        if |q̂| ≥ 1
```

Where **Q = 0.6** is the flat-top width parameter.

This is a smooth cone-angle apodization window:
- |q̂| < 0.6: full weight (ray near detector center, small cone angle)
- 0.6 ≤ |q̂| < 1.0: smooth cos² taper (ray approaching detector edge)
- |q̂| ≥ 1.0: zero weight (ray outside detector z-extent)

**Ref:** FreeCT_wFBP `cuda_kernels.cuh`, `W()` function; Stierstorfer 2004, Eq. 3–5.

#### 3.3.4 Backprojection with Weight Normalization

For each voxel, the backprojection loops over "half-turns" (π-intervals) of the helix. Multiple half-turns contribute rays through the same voxel at the same parallel view direction — these are **complementary/redundant rays**. The normalized weighted backprojection:

```
                    Σ_k  W(q̂_k) × p̂_filtered(α_k, p*, q*)
f(x,y,z) += (2π/N) × ─────────────────────────────────────
                           Σ_k  W(q̂_k)
```

Where:
- `k` indexes half-turns (complementary rays)
- `N` = views per rotation
- The denominator `Σ W(q̂_k)` normalizes weights to sum to 1 for each view direction
- `p*` and `q*` are the detector coordinates of the voxel projection

For pitch = 1: typically 2 half-turns contribute per voxel, with one having small |q̂| (high weight) and one having large |q̂| (low weight). For pitch < 1: more half-turns overlap, providing more redundancy.

**Ref:** Stierstorfer 2004, Section 2.3; FreeCT_wFBP backprojection kernel.

#### 3.3.5 Fan-to-Parallel Rebinning (Optional — Can Be Deferred)

The full WFBP includes fan-to-parallel rebinning before filtering:

```
β = arcsin((channel − c_center) × Δ_col / SDD)     # parallel fan angle
α = θ − β                                            # parallel view angle
```

Data is resampled via 2D bilinear interpolation from fan-beam `(channel, θ)` to parallel-beam `(β, α)`.

**Key insight for BasisSimulator:** Rebinning can be DEFERRED. The q̂-based weighting works in fan-beam geometry too, with a modified q̂ formula that uses fan-beam coordinates instead of parallel-beam. The rebinning improves quality (filtering along the helix tangent reduces cone-beam artifacts) but the weight function alone provides 80–90% of the quality improvement.

**Simplified Phase 2 approach (no rebinning):** Apply W(q̂) weighting directly in the existing fan-beam FDK backprojection kernel:

```julia
# In backproject_voxel, after computing row_f (v* detector coordinate):

# Compute normalized cone angle q_hat
v_center_offset = row_f - row_center
v_max = T(n_rows) / T(2)
q_hat = v_center_offset / v_max  # simplified: normalized detector row position

# Apply WFBP weight
q_abs = abs(q_hat)
w_helical = if q_abs < T(0.6)
    one(T)
elseif q_abs < one(T)
    t = (q_abs - T(0.6)) / T(0.4)
    cos(T(π) / T(2) * t)^2
else
    zero(T)
end

# Combined FDK + helical weight
weight = (SAD_sq / dist_sq) * w_helical
```

The normalization is handled by tracking weight sums per voxel inline (no separate buffer needed — see Section 3.9).

This simplified approach uses `v*/v_max` instead of the full q̂ formula. The full q̂ includes the helical interpolation correction term `(z_rot/(2π)) arcsin(p̂/SAD)`, which can be added in Phase 2b for improved quality at high pitch and large fan angles.

#### 3.3.6 Integration with Existing Code

> **✅ CRITIQUE C1, C3 — RESOLVED (REFINEMENT, 2026-03-17):**
>
> **The normalization constant is `2π/views_per_rotation`**, not `π/views_per_rotation`.
>
> **Source:** FreeCT_wFBP (`include/backproject.cuh`): `output[...] += s[k] * 2*pi / d_cg.n_proj_turn`
>
> **Why 2π not π:** The standard FDK integral is `(1/2) ∫₀²π ... dθ`, discretized as
> `(π/N) Σ`. But in WFBP, the per-sample weight normalization (`Σ W_k p_k / Σ W_k`)
> absorbs the 1/2 redundancy factor. Each angular sample contributes a weighted
> average of its redundant half-turn copies. The outer integral is a plain Riemann sum
> with step `Δβ = 2π/N`, giving `(2π/N) × Σ` with no extra 1/2.
>
> **The `is_helical` flag** (from `geom.pitch > 0`) controls BOTH the W(q̂) weighting
> AND the normalization strategy. For axial (`is_helical=false`), the standard
> `π/n_angles` scaling is used. For helical, the weight-normalized path uses
> `2π/views_per_rotation`. These do NOT degenerate into each other — they are
> separate code paths selected by the flag.
>
> **Ref:** Stierstorfer 2004 (PMB 49:2209); Hoffman 2016 (Med Phys 43:1411);
> FreeCT_wFBP source `backproject.cuh`.

**Changes to `backproject_voxel` (`backprojection.jl:37–146`):**

```julia
# ADD parameters:
#   is_helical::Bool, Q_flat::T, n_rows_half::T
# (passed as captured scalars from backproject!)

# REPLACE lines 136-140:
dist_sq = sv_x^2 + sv_y^2 + sv_z^2
fdk_weight = SAD_sq / dist_sq

# NEW: helical weight (only when is_helical)
if is_helical
    q_abs = abs(row_f - row_center) / n_rows_half
    w_h = if q_abs < Q_flat
        one(T)
    elseif q_abs < one(T)
        cos(T(π) / T(2) * (q_abs - Q_flat) / (one(T) - Q_flat))^2
    else
        zero(T)
    end
    weight = fdk_weight * w_h
else
    weight = fdk_weight  # standard FDK (unchanged)
end
```

**For axial (is_helical=false):** The code path is identical to current — no performance impact.

**Normalization — helical path (single-pass, inline per voxel):**

> **✅ CRITIQUE C1 — RESOLVED:** The normalization constant is **`2π/views_per_rotation`**.
>
> The weight accumulation is done inline per voxel — NO separate `weight_sum` buffer
> is needed. See Section 3.9 for the GPU kernel pseudocode.
>
> The WFBP discretized integral is:
> ```
> f(x) = (2π / views_per_rotation) × Σ[W × w_fdk × p̂] / Σ[W × w_fdk]
> ```
>
> **Derivation:** The standard FDK integral `(1/2) ∫₀²π` has the `1/2` to handle
> parallel-beam redundancy. In WFBP, the per-angular-sample weight normalization
> (`Σ_k W_k p_k / Σ_k W_k` across redundant half-turns at each angle) already
> absorbs this redundancy factor. The outer integration is a plain Riemann sum
> with step `Δα = 2π/N_proj`, yielding `(2π/N) × Σ`. This is confirmed by
> FreeCT_wFBP source: `output += s[k] * 2*pi / n_proj_turn`.
>
> **Ref:** FreeCT_wFBP `backproject.cuh`; Stierstorfer 2004, Eq. 5.

### 3.4 Katsevich Exact Algorithm (Optional — For Research)

**Ref:** Katsevich A, "Analysis of an exact inversion algorithm for spiral cone-beam CT," Phys. Med. Biol. 47(15):2583–2597, 2002.

**Ref (implementation):** ASTRA helical-kats, github.com/astra-toolbox/helical-kats (Python + CuPy + ASTRA GPU backprojection).

**Ref (simplified):** Noo F et al., "Exact helical reconstruction using native cone-beam geometries," Phys. Med. Biol. 48(23):3787–3818, 2003.

#### 3.4.1 Algorithm Steps

1. **Differentiation + length correction:**
   ```
   g₁(λ, u, v) = [D / √(u² + D² + v²)] × [∂p/∂λ + ((u² + D²)/D) × ∂p/∂u + (uv/D) × ∂p/∂v]
   ```
   Where D = SDD, λ = view angle, (u, v) = flat detector coordinates.

2. **Forward height rebinning** to k-line coordinates:
   ```
   v_k(u, ψ) = (D × h) / (2π × R) × (ψ + (ψ/tan(ψ)) × u/D)
   ```
   Maps detector rows to a coordinate system aligned with the helix.

3. **Hilbert filtering** (1D convolution along detector columns):
   ```
   h_H[n] = (1 − cos(π(n − N/2 − 0.5))) / (π(n − N/2 − 0.5))
   ```

4. **Reverse height rebinning** back to detector coordinates.

5. **Tam-Danielsson windowing:**
   ```
   w_bottom(u) = −(h / (2πRD)) × (u² + D²) × (π/2 + arctan(u/D))
   w_top(u)    =  (h / (2πRD)) × (u² + D²) × (π/2 − arctan(u/D))
   ```

6. **Weighted backprojection:**
   ```
   f(x) = (1/2π) ∫_{λ₁}^{λ₂} [1/|x − a(λ)|] × g_filtered(λ, u*, v*) dλ
   ```
   Integration over the PI-arc [λ₁, λ₂] for each voxel.

#### 3.4.2 Why NOT to Implement First

- **Complexity:** 7-step pipeline vs 3-step FDK. Requires height rebinning, Hilbert transforms, T-D windowing — all new code.
- **~73% data utilization:** Uses only PI-arc data → higher noise than WFBP.
- **Not used clinically:** "The Katsevich algorithm did not find its way directly into commercial use" (Hsieh 2021).
- **Marginal quality improvement:** For cone half-angles < 5° (our target scanners), the difference between WFBP and Katsevich is negligible.
- **GPU parallelism:** Same as FDK (voxel-parallel backprojection), but filtering steps are more complex.

**When to implement:** Only if BasisSimulator needs to simulate very wide-cone detectors (>128 rows) or if an exact reference reconstruction is needed for algorithm validation.

### 3.5 Rebinning / ASSR (Not Recommended)

**Ref:** Kachelrieß M et al., "Advanced single-slice rebinning in cone-beam spiral CT," Med. Phys. 27(4):754–772, 2000.

**ASSR** rebins helical cone-beam data to virtual 2D fan-beam scans on tilted reconstruction planes, then applies standard 2D FBP per slice.

**Optimal tilt angle:** `θ_tilt = arctan(3h / (4πR))` where h = pitch per turn, R = source radius.

**Quality:** Good for ≤ 16 detector rows (cone half-angle < 3°). Degrades for wider detectors (64+ rows) because the planar approximation breaks down.

**Not recommended for BasisSimulator** because:
- Our target scanners have 64–320 detector rows (well beyond ASSR's useful range)
- Would require entirely new code path (2D FBP per slice)
- WFBP provides better quality with same or lower implementation effort

### 3.6 Iterative Methods (Already Work — No Changes)

**SIRT and CGLS** (`sirt.jl`, `cgls.jl`) use matched (unweighted) backprojection. They work UNCHANGED for helical because:
- Forward projection (`siddon_forward_project!`) already handles arbitrary 3D geometry
- Matched backprojection (`backproject_voxel_matched`) reads per-angle positions
- No FDK weight involved — the adjoint operator is correct for any orbit

Iterative methods are the recommended fallback for:
- Very wide cone angles where WFBP artifacts are unacceptable
- Research requiring artifact-free reconstruction
- Low-dose protocols where regularization improves quality

### 3.7 Clinical Scanner Reconstruction Algorithms

| Vendor | Base Algorithm | Weight Scheme | Iterative Layer | DL Layer |
|--------|---------------|---------------|----------------|----------|
| **Siemens** | **WFBP** (Stierstorfer 2004) | cos² z-distance taper (Q=0.6) | SAFIRE → ADMIRE | — |
| **GE** | **3D CB-FBP** (Tang/Hsieh 2006) | Cone-angle conjugate ray: κ₂²/(κ₁²+κ₂²) | ASiR → ASiR-V | TrueFidelity |
| **Philips** | FDK-variant (unpublished) | Unknown | iDose4 → IMR | Precise Image |
| **Canon** | FDK-variant / ASSR | Unknown | AIDR 3D → FIRST | AiCE |

#### 3.7.1 Siemens WFBP (Detailed)

**Pipeline:** Fan-to-parallel rebinning → ramp filter (parallel geometry) → 3D backprojection with z-dependent W(q̂) weights.

**Weight from FreeCT_wFBP (open-source Siemens WFBP):**
```
qhat = (z_voxel - z_table + (z_rot/(2π)) * arcsin(p̂/SAD)) / (l̂ * tan(cone_half_angle))

W(qhat):
  |qhat| < 0.6:   W = 1.0
  0.6 ≤ |qhat| < 1.0:   W = cos²(π/2 * (|qhat| - 0.6) / 0.4)
  |qhat| ≥ 1.0:   W = 0.0

Normalization: f(x) = (2π/N_proj) * Σ(W_i * fdk_i * p̂_i) / Σ(W_i * fdk_i)
```

**Key property:** Noise is approximately pitch-independent (< 7% variation). Dose efficiency is good at all pitch values.

**Ref:** Stierstorfer 2004 (PMID 15248573); Flohr 2005 (PMID 16193784); Hoffman et al. 2016 (FreeCT, PMID 26936725).

#### 3.7.2 GE 3D CB-FBP (Detailed)

**Pipeline:** Row-wise fan-to-parallel rebinning ("tilted cone-beam reconstruction") → ramp filter → backprojection with cone-angle-based conjugate ray weighting.

**Key innovation (Tang/Hsieh 2006, PMID 16467583):** Out of conjugate ray pairs, the ray with the smaller cone angle gets larger weight:
```
For conjugate rays with cone angles κ₁ and κ₂:
  w₁ = κ₂² / (κ₁² + κ₂²)
  w₂ = κ₁² / (κ₁² + κ₂²)
```
This ensures: w₁ + w₂ = 1 (normalized), smaller cone angle → larger weight.

**Performance:** "Comparable to exact CB reconstruction algorithms, such as the Katsevich algorithm, under a moderate cone angle (4 degrees)" — Tang/Hsieh 2006.

**Ref:** Tang/Hsieh 2006 (PMID 16467583); Tang/Hsieh 2007 (PMID 17654902); Hsieh/Tang 2006 (PMID 17019037).

#### 3.7.3 Key Finding: NO Vendor Uses Katsevich

All clinical vendors use APPROXIMATE algorithms:
- Katsevich was patented (~2002, 20-year term → expired ~2022) which limited adoption
- Katsevich uses ~73% of data (PI window) → higher noise than WFBP (100% data)
- GE collaborated with Katsevich (2004, PMID 15357183) but published approximate CB-FBP as their clinical algorithm
- Deep learning reconstruction (TrueFidelity, AiCE, Precise Image) applied post-reconstruction largely masks remaining cone-beam artifacts from approximate methods

**Ref:** Hsieh J et al., "A brief history of CT reconstruction algorithms and future directions," 2012; Hoffman et al. 2016.

### 3.7b Reference Implementations Survey

| Framework | Helical FDK/FBP | Helical Iterative | Weighting | Key Mechanism |
|-----------|----------------|-------------------|-----------|---------------|
| **XCIST/CatSim** | **YES (complete)** | No | cos² view window + 3D conjugate ray Gamma^k1 | Fan-to-parallel rebinning + dedicated C library |
| **TIGRE** | No | Yes (SIRT, CGLS, OS-SART) | None (circular 1/U² only) | Per-angle `offOrigin[z]` offset |
| **ASTRA** | No (explicitly blocked) | Yes (SIRT, CGLS via cone_vec) | None for FDK | 12-element `cone_vec` per angle |
| **RTK** | No | No helical | Circular FDK weights only | Per-projection geometry parameters |
| **FreeCT_wFBP** | **YES (complete)** | No | WFBP cos² taper (Q=0.6) | Fan-to-parallel + z-distance weights |

**XCIST is the only CT simulator framework with helical FBP.** Key file: `gecatsim/reconstruction/src/Parallel_FDK_Helical_3DWeighting.c` — complete helical FBP with cos² view window and 3D cone-beam redundancy weighting via conjugate ray Gamma^k1 formula.

**TIGRE and ASTRA** handle helical via per-angle geometry offsets → iterative algorithms work, but their FDK kernels have NO helical weights. TIGRE's helical demo (`Python/demos/d13_HelicalGeometry.py`) only uses iterative algorithms.

**FreeCT_wFBP** (open-source WFBP) is the best reference for the Siemens-style algorithm. Source: `github.com/xiehq/FreeCT_wFBP`.

**NO open-source GPU Katsevich implementation exists as of 2026.** The only published GPU paper (Yan et al. 2010, PMID 20007041) used 16-year-old CUDA 2.x architecture with no public code.

### 3.8 Recommended Implementation Roadmap

#### Phase 1: Naive Helical FDK (1 code change)

**Effort:** ~1 hour. Change `pi_over_angles` from `π/n_angles` to `π/views_per_rotation`.

**Validates:** Helical geometry generation, forward projection, basic reconstruction. Produces usable images for pitch ≤ 1.

#### Phase 2: Weighted Helical FDK (backprojection kernel modification)

**Effort:** ~1 day. Add q̂-based W(q̂) weighting to `backproject_voxel`. Add weight normalization buffer.

**Validates:** Clinical-quality helical reconstruction. Matches Siemens WFBP approach.

**Changes:**
1. Add `is_helical`, `Q_flat`, `n_rows_half` parameters to `backproject_voxel`
2. Add W(q̂) weight computation after bilinear interpolation
3. Add weight accumulation for normalization
4. Add weight normalization pass (divide by accumulated weights)

#### Phase 3 (Optional): Fan-to-Parallel Rebinning

**Effort:** ~2 days. Add sinogram rebinning kernel.

**Benefit:** Filtering along helix tangent reduces cone-beam artifacts. Full WFBP quality.

**Changes:**
1. New GPU kernel `rebin_fan_to_parallel!` — 2D bilinear interpolation
2. Apply ramp filter in parallel-beam domain
3. Backprojection uses parallel-beam geometry internally

#### Phase 4 (Optional): Katsevich Exact

**Effort:** ~1 week. Complete new pipeline with height rebinning, Hilbert filtering, T-D windowing.

**Benefit:** Exact reconstruction for research applications. Reference for validating WFBP.

### 3.9 GPU Kernel Design for Phase 2 (Weighted Helical FDK)

The modified `backproject_voxel` pseudocode for helical:

```julia
@inline function backproject_voxel_helical(
    sinogram, voxel_x, voxel_y, voxel_z,
    source_positions, detector_centers, detector_u, detector_v,
    n_cols, n_rows, n_angles,
    col_center, row_center,
    pixel_mag, pixel_row_mag, SAD, SAD_sq,
    Q_flat, n_rows_half
) where T

    val_acc = zero(T)   # weighted value accumulator
    wgt_acc = zero(T)   # weight sum accumulator

    for angle in Int32(1):n_angles
        # ... (IDENTICAL geometry: source pos, detector projection, bilinear interp) ...
        # ... (produces: val, row_f, dist_sq) ...

        # Standard FDK distance weight
        fdk_w = SAD_sq / dist_sq

        # Helical cone-angle weight W(q̂)
        q_abs = abs(row_f - row_center) / n_rows_half
        w_h = if q_abs < Q_flat
            one(T)
        elseif q_abs < one(T)
            cos(T(π) / T(2) * (q_abs - Q_flat) / (one(T) - Q_flat))^2
        else
            zero(T)
        end

        w_total = fdk_w * w_h
        val_acc += val * w_total
        wgt_acc += fdk_w * w_h  # accumulate for normalization
    end

    # Normalize: weight-normalized backprojection with 2π/views_per_rotation
    # (C1 RESOLVED: FreeCT uses 2π/N, not π/N, because per-sample W normalization
    #  absorbs the 1/2 redundancy factor from the FDK integral)
    two_pi_over_vpr = T(2π) / T(views_per_rotation)
    return wgt_acc > T(1e-10) ? val_acc * two_pi_over_vpr / wgt_acc : zero(T)
end
```

**Parallelism:** Same as existing FDK — one GPU thread per voxel. The inner loop over angles is sequential (same as current). The W(q̂) computation adds ~5 FLOPs per angle per voxel — negligible overhead.

**Memory:** No additional sinogram buffers. One weight accumulator per voxel (computed inline, not stored).

**For axial (pitch=0):** All voxels have the same fixed row_f per angle (no Z variation), so `q_abs` is constant. For central slices, `q_abs ≈ 0 < Q_flat` → `w_h = 1` → identical to standard FDK.

### 3.10 Cone Angle Thresholds and Quality Analysis

| Cone Half-Angle | Detector Rows | Naive FDK | Weighted FDK (WFBP) | Katsevich |
|-----------------|---------------|-----------|---------------------|-----------|
| < 3° | ≤ 16 rows | Good | Excellent | Excellent |
| 3°–5° | 32–64 rows | Adequate | Good–Excellent | Excellent |
| 5°–7° | 64–128 rows | Visible artifacts | Good | Excellent |
| > 7° | 128+ rows | Poor | Adequate | Excellent |

**BasisSimulator target scanners:**
- NAEOTOM Alpha: 144 × 0.4mm rows = 57.6mm. Cone half-angle ≈ 3.3° at SAD=595mm. → **WFBP is excellent.**
- GE Apex Elite: 256 × 0.625mm rows = 160mm at isocenter. Cone half-angle ≈ 5.7° at SAD=541mm. → **WFBP is good.** (GE uses FDK-type clinically.)
- Aquilion ONE: 320 × 0.5mm rows = 160mm. Primarily axial volumetric. Cone half-angle ≈ 5.5°.

**Ref:** Turbell H, "Cone-Beam Reconstruction Using Filtered Backprojection," PhD Thesis, Linköping University, 2001, Chapter 6; Hsieh, "Computed Tomography," 3rd ed., Ch. 9.

---

## 4. API Design — Seamless Helical in the 5-Part API

**Status: COMPLETE (HELI-004 Discovery, 2026-03-17)**

### 4.0 Design Principle: One-Parameter Helical Activation

The user switches from axial to helical by adding **one parameter**: `pitch` in `CTProtocol`.
Everything else follows automatically:

```julia
# AXIAL (current behavior — unchanged)
protocol = CTProtocol(kVp=120, mA=200, views=984)

# HELICAL — add pitch and n_rotations
protocol = CTProtocol(kVp=120, mA=200, views=984, pitch=0.8, n_rotations=3)
```

The `pitch` parameter activates helical mode. When `pitch > 0`:
- `CTGeometry` constructor generates helical Z(θ) positions
- `fov_z` defaults to the Tam-Danielsson fully-sampled window
- `backproject!` applies helical weighting (WFBP)
- Dose formulas account for pitch

When `pitch == 0` (default): **everything behaves identically to current code.** Zero regressions.

### 4.1 CTProtocol — Add `pitch` Field

**Current struct** (`protocol.jl:35–49`): 13 fields. Missing `pitch`.

**Proposed struct:**

```julia
struct CTProtocol
    mA::Float64
    kVp::Float64
    views::Int                    # views per rotation (UNCHANGED meaning)
    rotation_time::Float64
    spectrum_path::Union{String, Nothing}
    n_rotations::Float64          # EXISTING field (default 1.0)
    pitch::Float64                # NEW: IEC beam pitch (0.0 = axial)
    dual_energy::Bool
    kVp_low::Float64
    mA_low::Float64
    integration_fraction::Float64
    collimation_mm::Union{Float64, Nothing}
    anode_angle::Int
    additional_filters::Vector{Tuple{String,Float64}}
end
```

**Keyword constructor changes** (`protocol.jl:87–133`):

```julia
function CTProtocol(;
    # ... existing kwargs ...
    pitch::Real=0.0,              # NEW: IEC beam pitch (0.0 = axial)
    # ... rest unchanged ...
)
    # ... existing mA/mAs logic ...

    # Helical validation
    _pitch = Float64(pitch)
    if _pitch < 0.0
        error("pitch must be non-negative (got $_pitch)")
    end
    if _pitch > 0.0 && n_rotations < 1.0
        error("Helical scan (pitch > 0) requires n_rotations ≥ 1.0 (got $n_rotations)")
    end

    return CTProtocol(
        final_mA, Float64(kVp), Int(views), Float64(rotation_time),
        spectrum_path, Float64(n_rotations), _pitch,  # pitch added after n_rotations
        dual_energy, Float64(kVp_low), Float64(mA_low),
        Float64(integration_fraction),
        collimation_mm === nothing ? nothing : Float64(collimation_mm),
        anode_angle, additional_filters
    )
end
```

**`views` field meaning is UNCHANGED:** Views per rotation. This is the physically meaningful quantity — it determines angular sampling density, I₀ per view (via `rotation_time / views`), and reconstruction angular resolution. Total views for the full helical scan = `views × n_rotations`, computed internally by CTGeometry.

**Impact on inner constructor call sites:** There are **3 call sites** using the positional inner constructor that must be updated:
1. `protocol.jl:118-132` — Main keyword constructor
2. `protocol.jl:418-432` — `constant_dose_protocol`
3. `protocol.jl:461-475` — `constant_noise_protocol`

All 3 must insert `_pitch` (or `base.pitch`) in the correct position after `n_rotations`.

### 4.2 CTGeometry Struct — Three New Fields

**Current struct** (`scanner.jl:552–566`): 13 fields.

**Proposed struct:**

```julia
struct CTGeometry
    SAD::Float64
    SDD::Float64
    n_angles::Int                      # total views (not per-rotation!)
    n_rows::Int
    n_cols::Int
    pixel_size::Float64
    pixel_row_size::Float64
    angles::Vector{Float64}
    source_positions::Matrix{Float64}
    detector_centers::Matrix{Float64}
    detector_u::Matrix{Float64}
    detector_v::Matrix{Float64}
    fov::NTuple{3, Float64}
    # NEW fields:
    pitch::Float64                     # IEC beam pitch (0.0 = axial)
    views_per_rotation::Int            # views in one gantry rotation
    recon_center::NTuple{3, Float64}   # (x, y, z) center of recon volume (cm)
end
```

**`n_angles` is TOTAL views** (not per-rotation). This preserves backward compatibility — every piece of code that iterates `for angle in 1:geom.n_angles` works correctly because it iterates over all views. The `views_per_rotation` field is needed ONLY for the backprojection normalization factor.

> **⚠ CRITIQUE C13:** The naming is ambiguous: the constructor PARAMETER `n_angles` means
> "views per rotation" (from `protocol.views`), but the struct FIELD `n_angles` stores
> "total views" (= views_per_rotation × n_rotations). Example:
> `CTGeometry(scanner; n_angles=984, pitch=0.8, n_rotations=3)` → `geom.n_angles == 2952`.
> This is correct for backward compat but confusing. **Resolution:** Accept as design
> tradeoff; add CLEAR docstring warning in the constructor documentation.

**Default values:**
- `pitch = 0.0` → axial
- `views_per_rotation = n_angles` → matches current behavior for axial
- `recon_center = (0.0, 0.0, 0.0)` → centered at origin, same as current

**Impact on 6 inner constructor call sites** (per CRITIQUE C4):

| # | File:Line | Change |
|---|-----------|--------|
| 1 | `scanner.jl:694` | Main constructor — add 3 new fields |
| 2 | `scanner.jl:802` | `create_aquilion_one` — add defaults `0.0, n_angles, (0.0,0.0,0.0)` |
| 3 | `fdk.jl:426` | FOV-override — propagate: `geom.pitch, geom.views_per_rotation, geom.recon_center` |
| 4 | `workspace.jl:364` | PCCT native geom — propagate: `geom.pitch, geom.views_per_rotation, geom.recon_center` |
| 5 | `mbir.jl:434` | Ordered-subset slicing — propagate: `geom.pitch, geom.views_per_rotation, geom.recon_center` |
| 6 | `scanners.jl:327` | Scanner factory — add defaults `0.0, n_angles, (0.0,0.0,0.0)` |

For sites 3/4/5, the new fields are simply copied from the parent geometry (same pitch, same views_per_rotation, same recon_center). For sites 2/6, axial defaults are used.

### 4.3 CTGeometry Constructor — Helical Keyword Arguments

**Current signature** (`scanner.jl:609–616`):
```julia
function CTGeometry(scanner::Scanner{T};
    n_angles::Int = 360,
    fov_cm=nothing, z_cm=nothing,
    n_rows=nothing, n_cols=nothing,
    collimation_mm=nothing
) where T
```

**Proposed signature:**
```julia
function CTGeometry(scanner::Scanner{T};
    n_angles::Int = 360,              # views per rotation
    pitch::Float64 = 0.0,            # IEC beam pitch (0.0 = axial)
    n_rotations::Float64 = 1.0,      # gantry rotations
    recon_center_z::Float64 = 0.0,   # recon volume Z center (cm)
    fov_cm=nothing, z_cm=nothing,
    n_rows=nothing, n_cols=nothing,
    collimation_mm=nothing
) where T
```

**Key design decisions:**

1. **`n_angles` stays as views_per_rotation.** Callers (`create_eict_workspace`, `create_workspace`) pass `n_angles=protocol.views` unchanged. The constructor internally computes `total_views = round(Int, n_angles * n_rotations)` when pitch > 0. **This resolves CRITIQUE C7** — no workspace call sites need changing.

2. **`pitch` and `n_rotations` come from CTProtocol.** The workspace creators thread them through:
   ```julia
   # In create_eict_workspace / create_workspace:
   geom = CTGeometry(scanner;
       n_angles=protocol.views,
       pitch=protocol.pitch,
       n_rotations=protocol.n_rotations,
       fov_cm=recon_opts.fov_cm,
       z_cm=recon_opts.z_cm,
       collimation_mm=protocol.collimation_mm
   )
   ```

3. **`recon_center_z` defaults to 0.0** (centered, same as current). For helical, the constructor could auto-center the recon volume, but explicit user control is preferred.

4. **Helical `fov_z` auto-computation** (resolves CRITIQUE C5):
   ```julia
   if pitch > 0.0 && z_cm === nothing
       z_travel = n_rotations * pitch * total_collim_cm
       fov_z = max(z_travel - total_collim_cm, total_collim_cm)
   end
   ```
   This is the Tam-Danielsson fully-sampled window. The `max(...)` ensures at least single-rotation detector coverage.

### 4.4 How Pitch Flows Through the System

Complete data flow from user to GPU kernel:

```
User sets:                     CTProtocol.pitch = 0.8
                               CTProtocol.n_rotations = 3
                                       ↓
create_eict_workspace:         CTGeometry(scanner; pitch=0.8, n_rotations=3, ...)
                                       ↓
CTGeometry constructor:        total_views = round(Int, 984 * 3) = 2952
                               z_travel = 3 * 0.8 * 5.76 = 13.82 cm
                               z_start = -6.91 cm
                               For each view i: Z_i = z_start + (i/984)*0.8*5.76
                               Stores: geom.pitch=0.8, geom.views_per_rotation=984
                                       ↓
simulate!:                     Passes geom through — NO changes
                               Forward projection: reads per-angle Z positions automatically
                               Physics pipeline: angle-agnostic — NO changes
                                       ↓
reconstruct! / fdk_reconstruct:
                               is_helical = geom.pitch > 0.0   (from CRITIQUE C12)
                               if is_helical:
                                   Use weight-normalized backprojection (WFBP)
                                   pi_over_angles replaced by weight normalization
                               else:
                                   Standard FDK (unchanged)
```

### 4.5 Workspace Changes

**EICTWorkspace** (`workspace.jl:441–503`): **No struct changes.** The workspace stores `geom::CTGeometry`, and all buffers derive dimensions from `geom`. When `geom` has more angles (helical), buffers are automatically larger.

> **⚠ CRITIQUE C16:** While no struct changes are needed, GPU memory scales linearly with
> `n_rotations`. EICTWorkspace has ~8 sinogram-sized buffers; PCCTWorkspace has ~5 per
> energy bin. See Section 2.4 for the memory table. Practical limit: ~5 rotations on
> 24GB GPU for clinical geometry (900 cols × 64 rows).

The `create_eict_workspace` call site (`workspace.jl:514`) changes only to pass helical kwargs:
```julia
# CURRENT:
geom = CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm,
                  z_cm=recon_opts.z_cm, collimation_mm=protocol.collimation_mm)

# HELICAL:
geom = CTGeometry(scanner; n_angles=protocol.views,
                  pitch=protocol.pitch, n_rotations=protocol.n_rotations,
                  fov_cm=recon_opts.fov_cm, z_cm=recon_opts.z_cm,
                  collimation_mm=protocol.collimation_mm)
```

**PCCTWorkspace** (`workspace.jl:32–138`): Same change at `workspace.jl:174`:
```julia
geom = CTGeometry(scanner; n_angles=protocol.views,
                  pitch=protocol.pitch, n_rotations=protocol.n_rotations,
                  fov_cm=recon_opts.fov_cm, z_cm=recon_opts.z_cm,
                  collimation_mm=protocol.collimation_mm)
```

**FDKReconWorkspace / HIRReconWorkspace**: These take an existing `geom` and `sinogram` — no changes needed. They inherit helical properties from the geometry.

### 4.6 simulate!() — No Changes Needed

Both `simulate!(ws::EICTWorkspace, ...)` (`driver.jl:330`) and `simulate!(ws::PCCTWorkspace, ...)` (`driver.jl:180`) use `ws.geom` for all geometry operations. They:

1. Call forward projection with `ws.geom` → positions are helical, rays trace correctly
2. Apply physics pipeline with `ws.geom` → angle-agnostic, works unchanged
3. Apply noise with `compute_detector_I0(geom, protocol, ...)` → uses `protocol.views` (per-rotation), correct for helical

**No code changes needed in the simulate! driver.**

### 4.7 reconstruct!() — Helical Detection

The `reconstruct!` functions (`driver.jl:815`, `driver.jl:880`) call `fdk_reconstruct` or iterative methods. The helical path is determined by:

```julia
# In fdk_reconstruct (fdk.jl), before calling backproject!:
is_helical = geom.pitch > 0.0
```

This flag is passed to `backproject!`, which selects the normalization strategy:

```julia
# In backproject! (backprojection.jl:296):
function backproject!(volume, sinogram, geom; weighted=true, ...)
    is_helical = geom.pitch > 0.0
    views_per_rotation = Int32(geom.views_per_rotation)

    if weighted
        if is_helical
            # Helical WFBP: weight-normalized backprojection
            # See Section 3.9 for kernel pseudocode
        else
            # Axial FDK: standard pi/n_angles scaling (unchanged)
        end
    else
        # Matched/unweighted: unchanged for iterative methods
    end
end
```

**ReconOptions does not need helical-specific fields.** The reconstruction algorithm is determined by the geometry — not by the user. Users who want iterative recon for helical just set `algorithm=:sirt`.

### 4.8 ReconOptions — z_cm for Helical

For helical, the reconstruction z-extent defaults to the Tam-Danielsson window (computed in CTGeometry constructor). Users can override with `z_cm`:

```julia
# Default z_cm for helical (from CTGeometry constructor):
# fov_z = max(z_travel - total_collimation_cm, total_collimation_cm)

# User override for custom z-range:
recon_opts = ReconOptions(algorithm=:fdk, z_cm=8.0, matrix_size=(512,512,128))
```

This override is passed to `CTGeometry` via `z_cm=recon_opts.z_cm` (already the case in `create_eict_workspace`). No structural change to ReconOptions needed.

### 4.9 Dose Formulas — CTDIvol and DLP with Pitch

**Current bug:** `compute_ctdi_vol` docstring says `CTDIvol = C × mAs × (kVp/120)^2.5 / pitch` but the implementation **does not divide by pitch** (`protocol.jl:261-272`). This is incorrect for helical — CTDIvol is inversely proportional to pitch (lower pitch = more overlap = higher dose per unit length).

**Fix:**
```julia
function compute_ctdi_vol(protocol::CTProtocol; phantom_diameter::Real=320.0)
    mAs = protocol.mA * protocol.rotation_time
    kvp_factor = (protocol.kVp / 120.0)^2.5
    size_factor = (320.0 / phantom_diameter)^2
    pitch_factor = protocol.pitch > 0 ? protocol.pitch : 1.0  # axial: no pitch correction
    return _CTDI_CAL_CONSTANT * mAs * kvp_factor * size_factor / pitch_factor
end
```

**DLP for helical:**
```julia
function compute_dlp(protocol::CTProtocol, scan_length_cm::Real; phantom_diameter::Real=320.0)
    ctdi = compute_ctdi_vol(protocol; phantom_diameter)
    return ctdi * scan_length_cm  # NOT multiplied by n_rotations — scan_length already includes all rotations
end
```

Note: The current DLP formula multiplies by `n_rotations`, which double-counts for helical (where `scan_length_cm` already reflects the helical travel). For helical, `scan_length_cm = n_rotations * pitch * total_collimation_cm`. The fixed formula simply multiplies CTDIvol by scan length.

> **⚠ CRITIQUE C17:** The `scan_length_cm` parameter semantics must be clarified in the
> docstring. For helical: `scan_length_cm` = total z-travel = `n_rotations × pitch ×
> total_collimation_cm`. For axial: `scan_length_cm` = z-coverage per rotation (n_rotations
> defaults to 1).

### 4.10 validate_protocol — Helical Validation Rules

**Add to `validate_protocol` (`protocol.jl:160–226`):**

```julia
# Helical validation
if protocol.pitch > 0.0
    # Pitch range
    if protocol.pitch > 3.5
        push!(messages, "ERROR: pitch > 3.5 is beyond clinical range (got $(protocol.pitch)). " *
              "Only dual-source scanners support pitch > 1.5.")
        valid = false
    end
    if protocol.pitch < 0.1
        push!(messages, "WARNING: pitch $(protocol.pitch) is very low — consider axial (pitch=0)")
    end

    # n_rotations for helical
    if protocol.n_rotations < 1.5
        push!(messages, "WARNING: helical with n_rotations=$(protocol.n_rotations) — " *
              "consider at least 2 rotations for meaningful z-coverage")
    end

    # Collimation required for helical
    if protocol.collimation_mm === nothing
        push!(messages, "WARNING: helical scan without explicit collimation_mm — " *
              "using full detector. Set collimation_mm for clinical realism.")
    end

    # Check z-coverage against scanner
    total_collim_mm = if protocol.collimation_mm !== nothing
        protocol.collimation_mm
    else
        scanner.detector_rows * scanner.detector_row_size
    end
    z_travel_mm = protocol.n_rotations * protocol.pitch * total_collim_mm
    if z_travel_mm < total_collim_mm
        push!(messages, "WARNING: helical z-travel ($(round(z_travel_mm, digits=1)) mm) < " *
              "detector coverage ($(round(total_collim_mm, digits=1)) mm). " *
              "Increase n_rotations or pitch.")
    end

    # Dual-energy + helical warning
    if protocol.dual_energy
        push!(messages, "WARNING: Dual-energy + helical is not validated. " *
              "Material decomposition may have z-dependent artifacts.")
    end
end
```

### 4.11 `create_aquilion_one` Disposition (Resolves CRITIQUE C10)

**Decision: KEEP but REFACTOR to eliminate code duplication.**

> **⚠ CRITIQUE C15:** `create_aquilion_one` (`scanner.jl:725–807`) has its own independent
> geometry computation loop (lines 773-798), duplicating the CTGeometry constructor logic.
> Adding helical Z formula to BOTH places creates maintenance risk. **Recommended approach:**
> Refactor `create_aquilion_one` to construct a `Scanner` and delegate to
> `CTGeometry(scanner; ...)`, eliminating the duplicated loop entirely. This requires
> reconciling the slightly different pixel_size derivation logic between the two functions.

**Update:** Refactor to delegate, then add `pitch` and `n_rotations` kwargs:
```julia
function create_aquilion_one(;
    n_angles::Int=360, n_rows::Int=64, n_cols::Int=128,
    fov_cm=nothing, z_cm=nothing, sad=nothing, sdd=nothing,
    pitch::Float64=0.0, n_rotations::Float64=1.0  # NEW
)
```

The Z-position loop gains the same `z_i = z_start + (θ/(2π)) * pitch * total_collim_cm` formula as the main CTGeometry constructor. The return statement adds the 3 new fields: `pitch, n_angles, (0.0, 0.0, 0.0)`.

### 4.12 User-Facing Examples

#### Example 1: Simple Helical Scan (NAEOTOM Alpha)

```julia
using BasisSimulator

# Scanner
scanner = create_naeotom_alpha()

# Protocol — helical with pitch 0.8, 3 rotations
protocol = CTProtocol(
    kVp=120, mA=200, views=720,
    rotation_time=0.5,
    pitch=0.8,                    # ← this activates helical
    n_rotations=3,                # ← 3 full rotations
    collimation_mm=57.6           # full detector
)

# Everything else is identical to axial
sim_opts = SimOptions(fidelity=:high)
recon_opts = ReconOptions(algorithm=:fdk, matrix_size=(512,512,128), fov_cm=25.0)

# Workspace + simulate (same API as axial)
ws = create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
```

#### Example 2: GE Apex Elite Helical

```julia
scanner = GERevolutionApex()

protocol = CTProtocol(
    kVp=120, mA=300, views=984,
    rotation_time=0.35,
    pitch=0.984,
    n_rotations=5,
    collimation_mm=80.0           # 128 × 0.625mm
)

# Identical workflow
ws = create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
```

#### Example 3: Axial (Unchanged — Zero Regression)

```julia
# No pitch parameter → pitch defaults to 0.0 → axial mode
protocol = CTProtocol(kVp=120, mA=200, views=984)
# Everything works exactly as before
```

### 4.13 `n_angles` Semantics Resolution

This is a key design decision that resolves the tension identified in CRITIQUE C7.

**The question:** Should `CTGeometry.n_angles` mean "total views across all rotations" or "views per rotation"?

**Answer: `n_angles` = total views. `views_per_rotation` = new separate field.**

**Rationale:**
1. Every for-loop in the codebase uses `for angle in 1:n_angles` or `for angle in 1:geom.n_angles`. These must iterate over ALL views — including multi-rotation views. If `n_angles` meant "per rotation," every loop would need `n_angles * n_rotations`.
2. Sinogram buffers are `(n_cols, n_rows, n_angles)`. The third dimension must be the total number of views.
3. The `backproject!` inner loop iterates `for angle in 1:n_angles` — it must see all views for helical.

**So:** The constructor receives `n_angles` as views_per_rotation (from `protocol.views`), internally computes `total_views = round(Int, n_angles * n_rotations)`, and stores `total_views` in `geom.n_angles` and `n_angles` in `geom.views_per_rotation`.

### 4.14 PCCT + Helical Integration Path

**PCCT workspace creation** (`workspace.jl:174`) uses the same `CTGeometry` constructor. Adding `pitch` and `n_rotations` kwargs is identical to the EICT path. The PCCT-specific code that follows (detector construction, DRM, charge sharing, etc.) is entirely geometry-agnostic.

**Native-resolution geometry** (`workspace.jl:364`): Constructed from the binned geometry by copying all fields and overriding pixel counts. The new fields (`pitch`, `views_per_rotation`, `recon_center`) are simply propagated from the parent geometry.

**PCCT + helical works with zero additional changes** beyond the 2 workspace call sites and the 6 CTGeometry constructor sites already identified.

### 4.15 Summary: Complete Change List

| Component | File:Line | Change Type | Description |
|-----------|-----------|-------------|-------------|
| **CTProtocol struct** | `protocol.jl:35-49` | Add field | `pitch::Float64` after `n_rotations` |
| **CTProtocol kwargs** | `protocol.jl:87-133` | Add kwarg | `pitch::Real=0.0`, validation, pass to inner |
| **CTProtocol inner call** | `protocol.jl:118-132` | Add arg | Insert `_pitch` in positional args |
| **constant_dose_protocol** | `protocol.jl:418-432` | Add arg | Insert `base.pitch` in positional args |
| **constant_noise_protocol** | `protocol.jl:461-475` | Add arg | Insert `base.pitch` in positional args |
| **validate_protocol** | `protocol.jl:160-226` | Add rules | Pitch range, n_rotations, DE+helical warning |
| **compute_ctdi_vol** | `protocol.jl:261-272` | Fix formula | Divide by pitch when pitch > 0 |
| **compute_dlp** | `protocol.jl:297-300` | Fix formula | Remove n_rotations multiply (scan_length includes it) |
| **CTGeometry struct** | `scanner.jl:552-566` | Add 3 fields | `pitch`, `views_per_rotation`, `recon_center` |
| **CTGeometry constructor** | `scanner.jl:609-699` | Major change | Add pitch/n_rot/recon_center kwargs, Z(θ), fov_z |
| **CTGeometry inner #1** | `scanner.jl:694` | Add 3 args | Main constructor return |
| **CTGeometry inner #2** | `scanner.jl:802` | Add 3 args | `create_aquilion_one` return |
| **CTGeometry inner #3** | `fdk.jl:426` | Add 3 args | FOV-override (propagate from parent) |
| **CTGeometry inner #4** | `workspace.jl:364` | Add 3 args | PCCT native geom (propagate from parent) |
| **CTGeometry inner #5** | `mbir.jl:434` | Add 3 args | Ordered-subset (propagate from parent) |
| **CTGeometry inner #6** | `scanners.jl:327` | Add 3 args | Scanner factory (axial defaults) |
| **create_aquilion_one** | `scanner.jl:725-807` | Add kwargs | `pitch=0.0, n_rotations=1.0` |
| **create_eict_workspace** | `workspace.jl:514` | Pass kwargs | `pitch=protocol.pitch, n_rotations=protocol.n_rotations` |
| **create_workspace (PCCT)** | `workspace.jl:174` | Pass kwargs | Same as EICT |
| **backproject!** | `backprojection.jl:296-426` | Branch logic | `is_helical = geom.pitch > 0`, select normalization |
| **backproject_voxel** | `backprojection.jl:37-146` | Add weight | W(q̂) helical weight + normalization |
| **vol_min_z (4 sites)** | See C6 above | Add offset | `+ geom.recon_center[3]` |

**Total: ~25 change sites across ~10 files.** No new files. No new structs. No new code paths for forward projection, physics, or iterative recon.

---

## 5. Physics Pipeline Compatibility

**Status: COMPLETE (HELI-005 Discovery, 2026-03-17)**

### 5.1 Core Finding: The Entire Physics Pipeline Works UNCHANGED for Helical

All 16 physics effects (10 standard + 3 signal chain + 3 PCCT corrections) are **helical-compatible with zero code changes**. This is because the gantry is a rigid body — source, detector, bowtie filter, flat filter, and anode all rotate AND translate together. From the gantry's frame of reference, nothing changes between axial and helical scanning. All physics effects operate either:

1. **Per-pixel (element-wise):** The effect operates on each sinogram element independently, with no reference to geometry. Example: fill factor, BHC, detector efficiency, noise.
2. **In the gantry frame:** The effect computes fan/cone angles relative to the source-detector geometry, which is FIXED within the gantry regardless of Z translation. Example: bowtie filter, flat filter, heel effect.
3. **Temporally sequential:** The effect operates on sequential views by index. Example: detector lag. For multi-rotation helical, sequential index access is correct — view N is contaminated by view N-1 regardless of rotation boundaries.

### 5.2 Effect-by-Effect Audit

| # | Effect | File | Geometry Usage | Helical Status | Reason |
|---|--------|------|----------------|----------------|--------|
| 1 | Fill factor | `fill_factor.jl` | None | ✅ UNCHANGED | Element-wise scaling by fill fraction |
| 2 | Flat filter | `flat_filter.jl` | `geom.SDD`, `pixel_size` | ✅ UNCHANGED | Computes path length from fan/cone angle — gantry-frame quantity |
| 3 | Bowtie filter | `bowtie_filter.jl` | `geom.SDD`, `pixel_size` | ✅ UNCHANGED | Fan angle interpolation — gantry-frame |
| 4 | Scatter (add) | `scatter.jl` | None | ✅ UNCHANGED | Sinogram-domain convolution kernel |
| 5 | Scatter correction | `scatter.jl` | None | ✅ UNCHANGED | Same convolution deblur kernel |
| 6 | Crosstalk | `crosstalk.jl` | None | ✅ UNCHANGED | 3×3 spatial convolution per angle |
| 7 | Optical crosstalk | `crosstalk.jl` | None | ✅ UNCHANGED | 3×3 spatial convolution per angle |
| 8 | Focal spot blur | `focal_spot.jl` | `geom.SAD`, `geom.SDD` | ✅ UNCHANGED | Spatial blur kernel — gantry-frame |
| 9 | Detector efficiency | `detector_efficiency.jl` | None | ✅ UNCHANGED | η(E) per energy — no geometry |
| 10 | Detector noise | `detector_noise.jl` | `protocol.views` (per-rot) | ✅ UNCHANGED | I₀ from `time_per_view = rot_time / views_per_rot` |
| 11 | Detector lag | `detector_lag.jl` | `size(sino, 3)` | ✅ UNCHANGED | Sequential index `prev_angle = angle - k` (see 5.4) |
| 12 | Heel effect | `heel_effect.jl` | `geom.SAD`, `geom.fov` | ✅ UNCHANGED | Fan angle from column position — gantry-frame (see 5.3) |
| 13 | DAS model | `das_model.jl` | None | ✅ UNCHANGED | Gain + electronic noise, element-wise |
| 14 | BHC | `bhc.jl` | None | ✅ UNCHANGED | Element-wise polynomial |
| 15 | PCCT charge sharing | `photon_counting.jl` | None | ✅ UNCHANGED | Per-pixel detector physics |
| 16 | PCCT pileup/anti-coincidence | `photon_counting.jl` | None | ✅ UNCHANGED | Per-pixel detector physics |

### 5.3 Heel Effect — Why It's Correct for Helical

The heel effect (`heel_effect.jl:150–210`) computes the fan angle γ from the detector column position:

```julia
fan_angle_max = T(atan(geom.fov[1] / 2 / geom.SAD))
γ = (T(col) - n_cols_T/T(2) - T(0.5)) / (n_cols_T/T(2)) * fan_angle_max
θ_effective = θ_anode + γ
```

This uses `geom.SAD` (source-to-axis distance, a constant) and the column index. It does NOT use `source_positions` or any Z-coordinate. The fan angle is a gantry-frame quantity — the anode is fixed relative to the detector, so the heel pattern is identical for every view regardless of the gantry's Z position.

For helical, the heel effect correctly applies the same column-dependent intensity modulation to every view. This matches physical reality: the anode self-attenuation depends only on the angle from the anode surface to each detector pixel, which doesn't change with table translation.

> **⚠ CRITIQUE C18 (pre-existing, NOT helical-specific):** `fan_angle_max` uses
> `geom.fov[1]` (reconstruction FOV) rather than detector fan coverage. For small recon FOV
> with large detector, this underestimates heel effect at detector edges. Does not affect
> helical compatibility. Note for future fix.

### 5.4 Detector Lag — Correct Sequential Behavior for Multi-Rotation

The lag model (`detector_lag.jl:236–300`) operates on sequential view indices:

```julia
for k in 0:(n_frames-1)
    prev_angle = angle - k
    if prev_angle >= 1
        weighted_sum += coeffs[k+1] * intensity[col, row, prev_angle]
    end
end
```

For multi-rotation helical, the sinogram has `total_views = views_per_rotation × n_rotations` in the third dimension. Views are stored sequentially: view 1 (rotation 1, angle 0°), ..., view 720 (rotation 1, angle 359.5°), view 721 (rotation 2, angle 0°), etc.

The lag correctly references `intensity[col, row, prev_angle]` by sequential index. View 721 is contaminated by views 720, 719, ... — which is physically correct (the lag is temporal, operating across the rotation boundary). **No changes needed.**

The `n_frames = min(n_history, n_angles)` computation uses `n_angles = size(sinogram, 3) = total_views`. For helical with 3 rotations × 720 views = 2160 total views, the lag looks back at up to 20 previous frames — well within a single rotation. **Correct.**

### 5.5 Bowtie and Flat Filter — Why They're Angle-Independent

Both the bowtie filter (`bowtie_filter.jl`) and flat filter (`flat_filter.jl`) generate a 2D transmission map `[n_cols, n_rows]` and apply it uniformly to all views. This is correct for BOTH axial and helical because:

1. The bowtie and flat filter are physically mounted on the tube assembly
2. They rotate WITH the source and detector as a rigid unit
3. The transmission through the filter depends on the angle from the central ray to each detector pixel — a gantry-frame geometric property
4. This angle does NOT change with Z translation

The 2D map is applied per-view using `AK.foreachindex`:
```julia
sino[col, row, angle] -= filter_projection[col, row]
```

This correctly adds the same filter attenuation to every view, regardless of the view's Z position. **No changes needed.**

### 5.6 Scatter — Convolution Approximation Is Adequate

The scatter model (`scatter.jl`) uses a convolution-based approximation:
1. Primary sinogram → convolve with scatter kernel → scatter sinogram
2. Add scatter sinogram to primary (or subtract for correction)

This approximation does not model 3D scatter physics (which depends on the patient anatomy at each table position). For helical, the actual scatter distribution varies with Z — different anatomy produces different scatter patterns.

**However:** The convolution approximation is already an approximation for axial too (it doesn't model anatomy-dependent scatter). The same level of approximation applies for helical. A future upgrade to Monte Carlo scatter would need to be Z-aware, but that's orthogonal to the helical geometry integration.

**No changes needed for helical.**

---

## 7. Validation and Testing Strategy

**Status: COMPLETE (HELI-007 Discovery, 2026-03-17)**

### 7.0 Validation Philosophy

Helical validation follows the same principle as the helical implementation itself: **generalize, don't duplicate.** The existing axial test infrastructure (Gammex 472, `small_test_setup()`, HU/CNR/NPS/MTF metrics) must continue to work unchanged. Helical tests extend this infrastructure with pitch-aware geometry, multi-rotation sinograms, and z-axis quality metrics.

**Three validation goals:**
1. **Axial regression** — pitch=0 produces bit-identical results to the current code
2. **Helical correctness** — geometry, forward projection, and reconstruction produce physically correct results
3. **Clinical quality** — HU accuracy, noise, and spatial resolution meet clinical standards at various pitch values

### 7.1 Testing Tiers

| Tier | Scope | Speed | Where | What It Validates |
|------|-------|-------|-------|-------------------|
| **T1: Unit** | Single function | < 1s | `test/runtests.jl` | Z(θ) formula, W(q̂) weights, pitch validation, struct fields |
| **T2: Axial Regression** | Full pipeline pitch=0 | ~5s | `test/runtests.jl` | Zero regressions — identical outputs to current code |
| **T3: Helical Integration** | Forward proj + recon | ~30s | `test/runtests.jl` | Geometry → sinogram → reconstruction produces valid images |
| **T4: Quality Metrics** | HU/noise/resolution | ~2min | `verification/notebooks/` | Quantitative clinical quality at various pitch values |
| **T5: Cross-Reference** | Comparison to XCIST | ~10min | `verification/notebooks/` | Absolute accuracy against independent reference |

### 7.2 Tier 1: Unit Tests

#### 7.2.1 Z(θ) Position Computation

```julia
@testset "Helical Z(θ) positions" begin
    scanner = create_naeotom_alpha()
    geom = CTGeometry(scanner; n_angles=720, pitch=1.0, n_rotations=3,
                      collimation_mm=57.6)

    total_collim_cm = 57.6 / 10.0  # 5.76 cm
    z_travel = 3 * 1.0 * total_collim_cm  # 17.28 cm
    z_start = -z_travel / 2  # -8.64 cm

    # Total views
    @test geom.n_angles == 720 * 3  # = 2160
    @test geom.views_per_rotation == 720

    # Z positions monotonically increase
    z_source = geom.source_positions[3, :]
    @test all(diff(z_source) .> 0)  # strictly increasing

    # First and last Z positions
    @test z_source[1] ≈ z_start atol=0.01
    @test z_source[end] ≈ -z_start - z_travel/geom.n_angles atol=0.01

    # Source and detector at same Z for each view
    z_det = geom.detector_centers[3, :]
    @test z_source ≈ z_det atol=1e-12

    # Z travel covers expected range
    @test z_source[end] - z_source[1] ≈ z_travel * (1 - 1/geom.n_angles) atol=0.01

    # Pitch stored correctly
    @test geom.pitch == 1.0
    @test geom.recon_center == (0.0, 0.0, 0.0)
end
```

**Ground truth:** Analytical formula `Z_i = z_start + (i/V) × pitch × collimation_cm`. No external reference needed — this is a coordinate computation with an exact answer.

#### 7.2.2 Axial Degeneration (pitch=0)

```julia
@testset "Pitch=0 degenerates to axial" begin
    scanner = create_naeotom_alpha()

    # Axial geometry (current code path)
    geom_axial = CTGeometry(scanner; n_angles=720)

    # Helical with pitch=0 (should be identical)
    geom_helical = CTGeometry(scanner; n_angles=720, pitch=0.0, n_rotations=1.0)

    # All position arrays must match exactly
    @test geom_axial.source_positions == geom_helical.source_positions
    @test geom_axial.detector_centers == geom_helical.detector_centers
    @test geom_axial.detector_u == geom_helical.detector_u
    @test geom_axial.detector_v == geom_helical.detector_v
    @test geom_axial.angles == geom_helical.angles
    @test geom_axial.n_angles == geom_helical.n_angles

    # Z=0 for all views
    @test all(geom_helical.source_positions[3, :] .== 0.0)
    @test all(geom_helical.detector_centers[3, :] .== 0.0)
end
```

**This is the most important unit test.** If pitch=0 doesn't produce identical geometry, the integration has a bug.

#### 7.2.3 CTProtocol Pitch Validation

```julia
@testset "CTProtocol pitch validation" begin
    # Default pitch is 0.0 (axial)
    p = CTProtocol(kVp=120, mA=200, views=720)
    @test p.pitch == 0.0

    # Valid helical
    p = CTProtocol(kVp=120, mA=200, views=720, pitch=0.8, n_rotations=3)
    @test p.pitch == 0.8
    @test p.n_rotations == 3.0

    # Negative pitch rejected
    @test_throws ErrorException CTProtocol(kVp=120, mA=200, views=720, pitch=-0.5)

    # Helical with n_rotations < 1 rejected
    @test_throws ErrorException CTProtocol(kVp=120, mA=200, views=720,
                                           pitch=0.8, n_rotations=0.5)
end
```

#### 7.2.4 W(q̂) Helical Weight Function

```julia
@testset "WFBP weight function W(q̂)" begin
    Q = 0.6  # flat-top width

    # Compute W(q_abs) inline (matches Section 3.3.3)
    function W(q_abs, Q=0.6)
        if q_abs < Q
            return 1.0
        elseif q_abs < 1.0
            t = (q_abs - Q) / (1.0 - Q)
            return cos(π/2 * t)^2
        else
            return 0.0
        end
    end

    # Known values
    @test W(0.0) == 1.0              # center → full weight
    @test W(0.3) == 1.0              # within flat region
    @test W(0.6) == 1.0              # boundary of flat region
    @test W(0.8) ≈ cos(π/2 * 0.5)^2 # = cos(π/4)² = 0.5
    @test W(1.0) == 0.0              # detector edge → zero
    @test W(1.5) == 0.0              # outside detector → zero

    # Symmetry: W is applied to |q̂|, so negative values behave identically
    # (tested via abs() in the kernel)

    # Monotonic decrease in taper region
    qs = range(0.6, 1.0, length=100)
    ws = W.(qs)
    @test all(diff(ws) .<= 0)  # non-increasing

    # Smooth transition (no jump at Q boundary)
    @test W(0.6 + 1e-6) ≈ 1.0 atol=1e-4
end
```

**Ground truth:** Analytical formula from Stierstorfer 2004, Eq. 3–5. No ambiguity.

#### 7.2.5 Tam-Danielsson Window (fov_z Auto-Computation)

```julia
@testset "Helical fov_z defaults to Tam-Danielsson window" begin
    scanner = create_naeotom_alpha()

    # pitch=0.8, 3 rotations, collimation 57.6mm = 5.76cm
    geom = CTGeometry(scanner; n_angles=720, pitch=0.8, n_rotations=3,
                      collimation_mm=57.6)

    total_collim_cm = 5.76
    z_travel = 3 * 0.8 * total_collim_cm  # = 13.824 cm
    expected_fov_z = z_travel - total_collim_cm  # = 8.064 cm

    @test geom.fov[3] ≈ expected_fov_z atol=0.01

    # With explicit z_cm, override takes effect
    geom2 = CTGeometry(scanner; n_angles=720, pitch=0.8, n_rotations=3,
                       collimation_mm=57.6, z_cm=5.0)
    @test geom2.fov[3] ≈ 5.0 atol=0.01
end
```

#### 7.2.6 Dose Formulas with Pitch

```julia
@testset "CTDIvol and DLP with pitch" begin
    p_axial = CTProtocol(kVp=120, mA=200, views=720, rotation_time=0.5)
    p_helical = CTProtocol(kVp=120, mA=200, views=720, rotation_time=0.5,
                           pitch=0.8, n_rotations=3)

    ctdi_axial = compute_ctdi_vol(p_axial)
    ctdi_helical = compute_ctdi_vol(p_helical)

    # CTDIvol inversely proportional to pitch
    @test ctdi_helical ≈ ctdi_axial / 0.8 atol=ctdi_axial * 0.01

    # DLP = CTDIvol × scan_length
    # For helical: scan_length = n_rot × pitch × collimation
    collim_cm = 5.76
    scan_length = 3 * 0.8 * collim_cm
    dlp = compute_dlp(p_helical, scan_length)
    @test dlp ≈ ctdi_helical * scan_length atol=0.1
end
```

### 7.3 Tier 2: Axial Regression Tests

These tests confirm that the helical modifications do not change ANY existing behavior when pitch=0.

#### 7.3.1 Forward Projection Regression

```julia
@testset "Axial forward projection unchanged after helical modification" begin
    phantom, geom = small_test_setup()
    # small_test_setup uses create_aquilion_one which has pitch=0.0

    # Monochromatic forward projection
    sino = forward_project(compute_μ(phantom, 60.0), geom)

    # Save reference values BEFORE helical implementation.
    # After implementation, these must still match.
    @test size(sino) == (64, 8, 36)
    @test sino[32, 4, 18] ≈ SAVED_REFERENCE_VALUE atol=1e-6
    # (SAVED_REFERENCE_VALUE captured from current code)
end
```

**Implementation note:** Before implementing helical, run the existing test suite and save key sinogram/reconstruction values as reference constants. After implementation, verify bit-for-bit (or atol=1e-6) agreement.

**Key regression points to capture (before helical implementation):**
1. `forward_project(compute_μ(phantom, 60.0), geom)` — central element
2. `fdk_reconstruct(sino, geom, size(phantom.mask))` — center voxel HU
3. `sirt_reconstruct(sino, geom, size(phantom.mask); iterations=10)` — center voxel
4. Full `simulate!` pipeline with `SimOptions(fidelity=:high)` — center sinogram element

#### 7.3.2 Full Pipeline Regression

```julia
@testset "Full simulate! pipeline regression (pitch=0)" begin
    scanner = create_naeotom_alpha()
    phantom = create_gammex_472(n_voxels=64, n_slices=16, fov_cm=35.0, z_cm=4.0)
    protocol = CTProtocol(kVp=120, mA=200, views=360, rotation_time=0.5)
    # pitch defaults to 0.0 — must produce identical result to pre-helical code

    sim_opts = SimOptions(fidelity=:high)
    recon_opts = ReconOptions(algorithm=:fdk, matrix_size=(64,64,16), fov_cm=35.0)

    ws = create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    # Central slice water region should be ~0 HU
    center = result.reconstruction[32, 32, 8]
    @test -50 < center < 50  # water ≈ 0 HU

    # Compare to saved reference (captured before helical changes)
    @test center ≈ SAVED_AXIAL_CENTER_HU atol=1.0
end
```

### 7.4 Tier 3: Helical Integration Tests

#### 7.4.1 Helical Forward Projection — Sinogram Shape and Content

```julia
@testset "Helical forward projection" begin
    scanner = create_naeotom_alpha()
    phantom = create_gammex_472(n_voxels=64, n_slices=64, fov_cm=35.0, z_cm=10.0)

    # Helical geometry: 3 rotations, pitch 0.8
    geom = CTGeometry(scanner; n_angles=360, pitch=0.8, n_rotations=3,
                      collimation_mm=57.6, n_rows=16, n_cols=128, fov_cm=35.0)

    μ_vol = compute_μ(phantom, 60.0)
    sino = forward_project(μ_vol, geom; volume_extent=phantom.extent)

    # Shape: n_cols × n_rows × total_views
    @test size(sino, 3) == 360 * 3  # = 1080 total views

    # Non-zero: phantom is large enough to intercept rays
    @test maximum(sino) > 0.0

    # Z-varying content: sinogram should change across rotations
    # because the source moves through different z-levels of the phantom
    mean_rot1 = mean(sino[:, :, 1:360])
    mean_rot2 = mean(sino[:, :, 361:720])
    mean_rot3 = mean(sino[:, :, 721:1080])
    # For a cylindrical phantom, means should be similar
    @test abs(mean_rot1 - mean_rot2) / mean_rot1 < 0.1  # < 10% variation
end
```

#### 7.4.2 Helical Naive FDK Reconstruction — Uniform Cylinder

This is the key correctness test for Phase 1 (naive helical FDK).

```julia
@testset "Helical FDK — uniform water cylinder" begin
    scanner = create_naeotom_alpha()

    # Create uniform water cylinder phantom (large z-extent)
    n = 64
    mask = zeros(UInt8, n, n, n)
    center = n ÷ 2
    radius = n ÷ 3
    for i in 1:n, j in 1:n, k in 1:n
        if (i - center)^2 + (j - center)^2 < radius^2
            mask[i, j, k] = UInt8(3)  # REGION_WATER
        end
    end
    phantom = Phantom(mask, Dict(3 => :water), (0.05, 0.05, 0.05))
    # extent: 3.2 × 3.2 × 3.2 cm

    # Helical geometry
    geom = CTGeometry(scanner; n_angles=360, pitch=1.0, n_rotations=2,
                      n_rows=16, n_cols=128, fov_cm=3.2, collimation_mm=6.4)

    # Forward project + reconstruct
    energies, weights = load_spectrum(120)
    energies, weights = downsample_spectrum(energies, weights, 10)
    materials = get_region_materials()
    sino = forward_project_poly(phantom, geom, energies, weights, materials)
    recon = fdk_reconstruct(sino, geom, (n, n, n))

    # Convert to HU
    μ_water = compute_μ_water(60.0)
    hu = (recon .- μ_water) ./ μ_water .* 1000

    # Central axial slice — water region should be ~0 HU
    central_slice = hu[:, :, n÷2]
    water_mask_2d = mask[:, :, n÷2] .== 3
    water_hu = central_slice[water_mask_2d]

    @test abs(mean(water_hu)) < 30     # mean HU accuracy < 30 HU from 0
    @test std(water_hu) < 100          # noise < 100 HU (coarse phantom, few views)

    # Z-uniformity: compare central slice to off-center slices
    # (within Tam-Danielsson window)
    for z_offset in [-8, -4, 4, 8]
        z_slice = n÷2 + z_offset
        if 1 <= z_slice <= n
            off_slice = hu[:, :, z_slice]
            off_water = off_slice[mask[:, :, z_slice] .== 3]
            if length(off_water) > 10
                @test abs(mean(off_water) - mean(water_hu)) < 50  # z-uniformity
            end
        end
    end
end
```

**Ground truth:** A uniform water cylinder must reconstruct to ~0 HU everywhere, regardless of pitch. Deviations indicate incorrect normalization (C1), incorrect scaling (π vs 2π), or geometry errors.

**Acceptance criteria:**
- Mean HU within ±30 of 0 (relaxed for coarse phantom)
- Z-uniformity: slice-to-slice mean variation < 50 HU within Tam-Danielsson window
- No systematic z-trend (rules out incorrect normalization)

#### 7.4.3 Helical WFBP vs Naive FDK — Quality Comparison

```julia
@testset "WFBP vs Naive FDK at low pitch" begin
    # At pitch=0.5, naive FDK over-estimates by ~2× due to data redundancy.
    # WFBP should produce correct values.
    scanner = create_naeotom_alpha()
    phantom = create_uniform_water_cylinder(...)  # helper

    geom_low = CTGeometry(scanner; n_angles=360, pitch=0.5, n_rotations=3, ...)

    sino = forward_project_poly(phantom, geom_low, ...)
    recon_naive = fdk_reconstruct(sino, geom_low, ...; helical_weights=false)
    recon_wfbp  = fdk_reconstruct(sino, geom_low, ...; helical_weights=true)

    # Naive should over-estimate (redundant data at low pitch)
    hu_naive = to_hu(recon_naive)
    hu_wfbp  = to_hu(recon_wfbp)

    mean_naive = mean(hu_naive[water_mask])
    mean_wfbp  = mean(hu_wfbp[water_mask])

    # WFBP should be closer to 0 HU than naive
    @test abs(mean_wfbp) < abs(mean_naive)
    @test abs(mean_wfbp) < 30  # WFBP should be accurate
end
```

#### 7.4.4 Helical Iterative Reconstruction (SIRT) — Reference

Iterative recon (SIRT/CGLS) with matched backprojection is the **gold standard reference** for helical because it doesn't use FDK weights — the forward/adjoint pair is exact. This provides ground truth for validating WFBP.

```julia
@testset "Helical SIRT convergence" begin
    scanner = create_naeotom_alpha()
    phantom = create_gammex_472(n_voxels=32, n_slices=32, fov_cm=35.0, z_cm=6.0)

    geom = CTGeometry(scanner; n_angles=180, pitch=0.8, n_rotations=2,
                      n_rows=8, n_cols=64, fov_cm=35.0, collimation_mm=57.6)

    μ_vol = compute_μ(phantom, 60.0)
    sino = forward_project(μ_vol, geom; volume_extent=phantom.extent)

    # SIRT reconstruction (matched backprojection, no FDK weight)
    recon_sirt = sirt_reconstruct(sino, geom, size(phantom.mask); iterations=50)

    # Should converge to something reasonable
    @test !any(isnan, recon_sirt)
    @test !any(isinf, recon_sirt)
    @test maximum(recon_sirt) > 0  # non-trivial reconstruction

    # Compare to FDK
    recon_fdk = fdk_reconstruct(sino, geom, size(phantom.mask))

    # Both should reconstruct water region similarly
    # (SIRT converges to correct answer; FDK approximate but close for pitch≈1)
    center_sirt = recon_sirt[16, 16, 16]
    center_fdk  = recon_fdk[16, 16, 16]
    # Both should be in the same ballpark
    @test abs(center_sirt - center_fdk) < 0.05 * abs(center_sirt + 1e-6)
end
```

#### 7.4.5 PCCT + Helical

```julia
@testset "PCCT + Helical integration" begin
    scanner = create_naeotom_alpha()
    phantom = create_gammex_472(n_voxels=32, n_slices=32, fov_cm=35.0, z_cm=6.0)

    protocol = CTProtocol(kVp=120, mA=200, views=360, rotation_time=0.5,
                          pitch=0.8, n_rotations=2, collimation_mm=57.6)
    sim_opts = SimOptions(fidelity=:pcct)
    recon_opts = ReconOptions(algorithm=:fdk, matrix_size=(32,32,32), fov_cm=35.0)

    ws = create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)

    # Should not throw
    result = simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    # Basic sanity: reconstruction is non-trivial
    @test size(result.reconstruction) == (32, 32, 32)
    @test maximum(result.reconstruction) > 0
    @test !any(isnan, result.reconstruction)
end
```

### 7.5 Tier 4: Quality Metrics (Verification Notebook)

These tests belong in a verification notebook (e.g., `verification/notebooks/08_helical_validation.jl`) and use the same metrics infrastructure as notebooks 06 and 07.

#### 7.5.1 Helical Gammex 472 — HU Accuracy

**Test setup:**
- Scanner: NAEOTOM Alpha (or GE Apex Elite)
- Phantom: Gammex 472 at XCAT resolution (factor 2: 800×700×250, 0.04cm voxels)
- Protocol: 120 kVp, 200 mA, pitch=0.8, 3 rotations
- Reconstruction: FDK with WFBP weights, 512×512 matrix

**Metrics and acceptance criteria:**

| Metric | Target | Tolerance | How Measured |
|--------|--------|-----------|-------------|
| **Water HU** | 0 HU | ±10 HU | Mean HU in 2 water ROIs |
| **Ca_50 HU** | ~50 HU | ±15 HU | Mean HU in Ca_50 rod ROI |
| **Ca_100 HU** | ~100 HU | ±20 HU | Mean HU in Ca_100 rod ROI |
| **Ca_200 HU** | ~200 HU | ±30 HU | Mean HU in Ca_200 rod ROI |
| **I_5.0 HU** | ~250 HU | ±30 HU | Mean HU in Iodine 5.0 rod ROI |
| **σ_water (noise)** | Comparable to axial | <1.5× axial noise | Std of water ROI |
| **CNR (Ca_200)** | >5 | >3 minimum | (HU_rod − HU_water) / σ_water |
| **MTF50** | >2 lp/cm | >1.5 lp/cm | Circular edge method |
| **NPS peak** | ~1.5 lp/cm | 0.5–3.0 lp/cm | Local NPS radial average |

**Key comparison:** Run the same phantom/scanner/protocol in axial mode (pitch=0, 1 rotation, same total mAs). Helical metrics should be within 20% of axial metrics. If noise is significantly higher or HU accuracy degrades, the helical weighting has a bug.

**Ref:** Existing notebook `06_ge_apex_elite_clinical.jl` uses `segment_gammex_rods()` for automated ROI extraction and `measure_nps_local()` / `measure_mtf_circular_edge()` for image quality metrics.

#### 7.5.2 Z-Uniformity Test

**Purpose:** Verify that reconstruction quality is uniform along the z-axis within the Tam-Danielsson window.

**Method:** Reconstruct with 128+ z-slices. For each slice, measure water ROI mean HU and noise (σ). Plot mean(HU) and σ vs z.

**Acceptance criteria:**
- Mean HU variation < ±15 HU across all slices within Tam-Danielsson window
- No systematic z-trend (linear fit slope < 2 HU/cm)
- Noise variation < 20% across slices (√(σ_max/σ_min) < 1.2)
- Edge slices (outside Tam-Danielsson window) may have degraded quality — document but don't fail

**This is the single most diagnostic test for helical reconstruction.** Incorrect normalization (C1) manifests as z-dependent HU drift. Incorrect scaling manifests as global offset. Incorrect W(q̂) manifests as noise variation.

#### 7.5.3 Pitch Sweep

**Purpose:** Verify correct behavior across the clinical pitch range.

**Method:** Run identical phantom/scanner with pitch ∈ {0.5, 0.8, 1.0, 1.2, 1.5}. For each, measure:
1. Mean HU (water) — should be ~0 for all pitch values
2. Noise (σ_water) — should be approximately pitch-independent for WFBP (< 7% variation per Stierstorfer 2004)
3. CNR — should scale inversely with σ (higher pitch → higher CTDIvol-normalized noise)

**Acceptance criteria:**
- Mean HU within ±10 of 0 for all pitch values (WFBP)
- Noise ratio max/min across pitch values < 1.15 (WFBP)
- For naive FDK (Phase 1): mean HU may deviate at pitch < 0.8 (expected — see Section 3.2)

**Ref:** Stierstorfer 2004, Figure 3: noise vs pitch for WFBP shows < 5% variation for pitch 0.5–1.5 with 32-row detector.

#### 7.5.4 Cone-Beam Artifact Assessment

**Purpose:** Quantify remaining cone-beam artifacts in WFBP reconstruction.

**Method:** Use a phantom with a sharp planar interface (e.g., a slab of bone within water, perpendicular to the z-axis). Reconstruct and measure artifact magnitude near the interface.

**Metric:** Deviation from expected HU in regions adjacent to the slab boundary.

**Expected behavior:**
- NAEOTOM Alpha (3.3° cone half-angle): Artifacts < 5 HU
- GE Apex Elite (5.7° cone half-angle): Artifacts < 15 HU
- Naive FDK at high pitch: Artifacts may exceed 30 HU

**This test is informational (documenting limitations), not pass/fail.** Cone-beam artifacts are inherent to approximate algorithms.

### 7.6 Tier 5: Cross-Reference Validation

#### 7.6.1 XCIST/CatSim Helical Comparison

**Purpose:** Absolute validation against an independent helical CT simulator.

**Method:** Run identical helical scan parameters in BasisSimulator and XCIST. Compare:
1. Sinograms (line profiles through matching views)
2. Reconstructed HU values
3. Noise properties

**XCIST setup** (from existing notebook `01_single_kvp_verification.jl` pattern):
```python
# XCIST helical configuration
cfg.protocol.viewsPerRotation = 720
cfg.protocol.viewCount = 720 * 3  # 3 rotations
cfg.protocol.scanType = "helical"
cfg.protocol.pitch = 0.8
```

**Acceptance criteria:**
- HU agreement within ±20 HU for all Gammex rods (same tolerance as axial XCIST comparison)
- Sinogram central profile correlation > 0.95

**Dependencies:** XCIST must be accessible via PythonCall (existing infrastructure in notebook 01). XCIST helical reconstruction uses `helical_equiAngle` rebinning + FBP — a different algorithm than WFBP, so perfect agreement is not expected. The comparison validates the forward model (which should match closely) and confirms reconstruction is in the right ballpark.

#### 7.6.2 Self-Consistency Check: SIRT as Reference

Since iterative methods (SIRT, CGLS) with matched backprojection are mathematically correct for any geometry, a converged SIRT reconstruction serves as an internal ground truth:

```julia
# Run enough iterations for convergence
recon_sirt = sirt_reconstruct(sino, geom, vol_size; iterations=200)
recon_wfbp = fdk_reconstruct(sino, geom, vol_size)  # WFBP

# WFBP should approximate SIRT result
hu_sirt = to_hu(recon_sirt)
hu_wfbp = to_hu(recon_wfbp)

# Per-rod HU comparison
for rod in rods
    hu_s = mean(hu_sirt[rod.mask])
    hu_w = mean(hu_wfbp[rod.mask])
    @test abs(hu_s - hu_w) < 30  # WFBP vs converged SIRT < 30 HU
end
```

**This is the strongest internal validation** because it compares two independent reconstruction algorithms operating on the same helical sinogram data. SIRT is provably correct for the geometry; WFBP should approximate it.

### 7.7 Edge Cases and Error Handling

#### 7.7.1 Phantom Smaller Than Scan Range

```julia
@testset "Small phantom + helical warning" begin
    scanner = create_naeotom_alpha()
    phantom = create_gammex_472(n_voxels=32, n_slices=8, fov_cm=35.0, z_cm=2.0)
    # z_cm=2.0 cm phantom, helical travel = 3*0.8*5.76 = 13.82 cm → phantom too small

    protocol = CTProtocol(kVp=120, mA=200, views=360, pitch=0.8, n_rotations=3,
                          collimation_mm=57.6)
    recon_opts = ReconOptions(algorithm=:fdk, matrix_size=(32,32,8), fov_cm=35.0)

    # Should produce a warning about phantom z-extent but NOT error
    @test_logs (:warn,) create_eict_workspace(
        scanner, protocol, SimOptions(), recon_opts, phantom)

    # Sinogram should have mostly-zero regions (views outside phantom)
    # This is physically correct — air projects as zero attenuation
end
```

#### 7.7.2 Single Rotation with Pitch > 0

```julia
@testset "Single rotation helical" begin
    # pitch > 0 with n_rotations=1 — valid but minimal z-coverage
    scanner = create_naeotom_alpha()
    geom = CTGeometry(scanner; n_angles=360, pitch=0.8, n_rotations=1.0,
                      n_rows=8, n_cols=64, collimation_mm=57.6)

    @test geom.n_angles == 360  # single rotation
    @test geom.pitch == 0.8

    # Z should vary within single rotation
    z_range = geom.source_positions[3, end] - geom.source_positions[3, 1]
    expected_travel = 0.8 * 5.76  # ~4.6 cm within one rotation
    @test z_range ≈ expected_travel * (1 - 1/360) atol=0.1
end
```

#### 7.7.3 Very High Pitch (Cardiac/Dual-Source)

```julia
@testset "High pitch (dual-source style)" begin
    protocol = CTProtocol(kVp=120, mA=200, views=720, pitch=3.0, n_rotations=1)
    # pitch=3.0 valid (dual-source Flash mode)

    valid, msgs = validate_protocol(protocol, scanner)
    # Should produce a validation message but not error
    @test valid  # pitch ≤ 3.5 is accepted
end
```

#### 7.7.4 Reconstruction Center Z Offset

```julia
@testset "Recon center Z offset" begin
    scanner = create_naeotom_alpha()
    geom = CTGeometry(scanner; n_angles=360, pitch=0.8, n_rotations=3,
                      recon_center_z=2.0, n_rows=16, n_cols=128)

    @test geom.recon_center == (0.0, 0.0, 2.0)

    # Reconstruction should be centered at Z=2cm, not Z=0
    # Verify by checking that the recon volume's Z-bounds are offset
    fov_z = geom.fov[3]
    vol_min_z = geom.recon_center[3] - fov_z / 2
    vol_max_z = geom.recon_center[3] + fov_z / 2
    @test vol_min_z ≈ 2.0 - fov_z/2
    @test vol_max_z ≈ 2.0 + fov_z/2
end
```

### 7.8 Ground Truth Summary

| Source | What It Validates | Strength | Limitation |
|--------|-------------------|----------|------------|
| **Analytical formulas** | Z(θ), W(q̂), pitch, dose | Exact | Only validates math, not integration |
| **Axial regression** | No regressions from helical code | Complete | Only validates pitch=0 path |
| **Uniform cylinder** | HU accuracy, normalization | Simple, unambiguous | Insensitive to spatial resolution |
| **Converged SIRT** | FDK/WFBP correctness | Independent internal reference | Slow, limited by convergence |
| **XCIST/CatSim** | Absolute forward model accuracy | Independent external reference | Different recon algorithm |
| **Gammex 472** | Multi-material HU accuracy | Clinical relevance, existing infrastructure | Limited z-information |
| **Pitch sweep** | Weight normalization, dose efficiency | Exposes scaling bugs | Requires multiple runs |
| **Z-uniformity** | Helical normalization, weight correctness | Most diagnostic for helical bugs | Requires large phantom |

### 7.9 Recommended Test Implementation Order

1. **Before any helical code changes:** Capture regression reference values from current axial tests (sinogram elements, reconstruction center voxels). Store as constants in `test/runtests.jl`.

2. **Phase 1 (Geometry + Forward Projection):**
   - T1: Z(θ) computation (7.2.1)
   - T1: Axial degeneration (7.2.2)
   - T1: Protocol validation (7.2.3)
   - T2: Axial regression sinogram (7.3.1)
   - T3: Helical sinogram shape and content (7.4.1)

3. **Phase 2 (Naive Helical FDK):**
   - T1: Tam-Danielsson window (7.2.5)
   - T2: Full pipeline regression (7.3.2)
   - T3: Uniform cylinder HU (7.4.2)
   - T3: Helical SIRT convergence (7.4.4)

4. **Phase 3 (WFBP):**
   - T1: W(q̂) weight function (7.2.4)
   - T3: WFBP vs naive FDK (7.4.3)
   - T4: Gammex 472 HU accuracy (7.5.1)
   - T4: Z-uniformity (7.5.2)
   - T4: Pitch sweep (7.5.3)

5. **Phase 4 (Integration):**
   - T3: PCCT + helical (7.4.5)
   - T1: Dose formulas (7.2.6)
   - T5: XCIST cross-reference (7.6.1)
   - T5: SIRT self-consistency (7.6.2)
   - Edge cases (7.7.*)

### 7.10 Test Phantom Requirements

| Phantom | Purpose | Minimum Size | Notes |
|---------|---------|-------------|-------|
| **Gammex 472** (existing) | Multi-material HU, CNR, MTF | `n_slices≥32, z_cm≥6` for helical | Existing `create_gammex_472()` with larger z |
| **Uniform water cylinder** | Z-uniformity, normalization | `z_cm ≥ z_travel + collimation_cm` | Simple cylindrical mask, water material |
| **small_test_setup()** (existing) | Fast unit/regression | Existing 32×32×8 | Used only for axial regression |
| **XCAT** (existing) | Clinical realism | Factor 2 (800×700×250) | Existing phantom, z_extent=10cm |

**No new phantom types are needed.** The existing Gammex 472 with increased z-extent and a simple uniform cylinder (constructible from `Phantom()` with a cylindrical mask) cover all test scenarios.

### 7.11 Failure Mode Diagnosis Guide

| Symptom | Likely Cause | Diagnostic Test |
|---------|-------------|-----------------|
| **Global HU offset in helical** | Wrong normalization constant (π vs 2π) | Uniform cylinder (7.4.2): compare mean HU vs expected 0 |
| **HU drift along z** | Normalization depends on view count not weight sum | Z-uniformity (7.5.2): plot mean HU per slice |
| **HU correct at pitch=1, wrong at pitch=0.5** | Naive FDK (no weight normalization for redundancy) | Pitch sweep (7.5.3): should motivate WFBP |
| **Noise varies with pitch** | Weight normalization incorrect | Pitch sweep (7.5.3): σ should be ~constant for WFBP |
| **Axial results changed** | Helical code affects pitch=0 path | Axial regression (7.3.*): bit-exact comparison |
| **Streaks at z-edges** | Insufficient angular coverage (voxels outside Tam-Danielsson window) | Extend fov_z to full range and observe edge quality |
| **PCCT crash or NaN** | New CTGeometry fields not propagated to native-res geometry | PCCT + helical (7.4.5): check workspace.jl:364 |
| **SIRT diverges for helical** | Forward/adjoint mismatch (recon_center not applied symmetrically) | Check vol_min_z offset in both siddon and backprojection |

### 5.7 Noise — Per-View Photon Count Is Correct

`compute_detector_I0` (`detector_noise.jl:438–474`) computes I₀ per pixel per view:

```julia
time_per_view = protocol.rotation_time / protocol.views
I₀ = spectrum_flux_sum × mA × time_per_view × pixel_area_mm²
```

`protocol.views` means views per rotation. The time per view (`rotation_time / views_per_rotation`) is the correct physical integration time regardless of how many rotations occur. Each view has the same tube current and integration time in both axial and helical.

For helical, the total number of views in the sinogram is larger (multi-rotation), but I₀ per view is unchanged. The noise model in `simulate!` (`driver.jl:480–520`) applies the same I₀ to all views via element-wise operations. **Correct for helical.**

### 5.8 PCCT Detector Physics — Fully Angle-Agnostic

The PCCT detector physics chain (charge sharing, pileup, anti-coincidence, spectral binning) operates entirely per-pixel per-energy:

- `apply_charge_sharing!` — per-pixel probability based on charge cloud σ and pixel pitch
- `apply_pileup!` — per-pixel count rate effect
- `apply_anti_coincidence!` — per-pixel neighboring pixel logic
- `spatial_bin!` — bins native dexels to final pixels (pure summation)

None of these reference source/detector positions, angles, or orbit parameters. They process the spectral sinogram element-by-element. **Fully helical-compatible.**

### 5.9 Signal Chain (CatSim Path) — All Steps Angle-Agnostic

The `simulate!(ws::EICTWorkspace, ...)` driver (`driver.jl:363–470`) applies signal chain steps:

1. **Physics (no noise):** Calls `_apply_physics_no_noise!` — same pipeline as Section 5.2
2. **exp(-sino) conversion:** Element-wise
3. **Heel effect:** Gantry-frame (Section 5.3)
4. **DAS model:** Element-wise gain + noise
5. **Air scan generation:** Same heel/DAS applied to ones — gantry-frame
6. **Calibration (phantom/air):** Element-wise division
7. **Low signal correction:** Element-wise
8. **Log transform:** Element-wise
9. **BHC:** Element-wise polynomial

Every step is either element-wise or gantry-frame. **All steps work unchanged for helical.**

### 5.10 Summary

**The physics pipeline is the simplest part of helical integration.** Zero code changes are needed across ALL 16 physics effects, the signal chain, and the PCCT detector model. The architectural decision to separate physics (gantry-frame) from geometry (world-frame) made this entirely free.

| Category | Components | Change Required |
|----------|-----------|-----------------|
| Element-wise effects | fill_factor, BHC, DAS, noise, PCCT | **NONE** |
| Gantry-frame effects | flat_filter, bowtie, heel, focal_spot | **NONE** |
| Sequential effects | detector lag | **NONE** |
| Sinogram-domain effects | scatter, crosstalk | **NONE** |
| Forward projection | Siddon, polychromatic, PCCT | **NONE** (already confirmed in Section 2) |
| **Geometry construction** | CTGeometry constructor | **YES** (Section 1) |
| **Reconstruction** | FDK backprojection | **YES** (Section 3) |
| **API** | CTProtocol, workspace creation | **YES** (Section 4) |

---

## 6. Reference Implementations

**Status: COMPLETE (covered in Section 3.7b)**

See Section 3.7b for the complete survey of XCIST/CatSim, TIGRE, ASTRA, RTK, and FreeCT_wFBP. Key finding: XCIST and FreeCT_wFBP are the only open-source frameworks with helical FBP reconstruction. BasisSimulator's approach (WFBP) aligns with FreeCT_wFBP (Siemens-style).

---

## [Section 7 content is above — see "7. Validation and Testing Strategy" starting at the earlier position in this file]

---

## 8. SYNTHESIS: Implementation Roadmap

**Status: COMPLETE (HELI-008 REFINEMENT + SYNTHESIS, 2026-03-17)**

### 8.0 Critique Resolution Summary

All 18 critique items (C1-C18) resolved:

| ID | Severity | Status | Resolution |
|----|----------|--------|------------|
| **C1** | CRITICAL | **✅ RESOLVED** | Normalization = `2π/views_per_rotation`. FreeCT confirms. See Section 3.3.6. |
| **C2** | IMPORTANT | **✅ ACCEPTED** | Naive FDK at pitch<0.8 overestimates. Phase 1 for pitch≈1 only; Phase 2 WFBP fixes. |
| **C3** | IMPORTANT | **✅ RESOLVED** | `is_helical` flag controls both W(q̂) and normalization. See Section 3.3.6. |
| **C4** | IMPORTANT | **✅ RESOLVED** | 7 inner constructor call sites enumerated with exact changes. See Section 8.2. |
| **C5** | IMPORTANT | **✅ RESOLVED** | Helical fov_z auto-computation in constructor. See Section 4.3. |
| **C6** | IMPORTANT | **✅ RESOLVED** | 4 `recon_center` offset sites with exact diffs. See Section 8.3. |
| **C7** | MINOR | **✅ RESOLVED** | `n_angles` param = views/rot, struct field = total views. See Section 4.13. |
| **C9** | MINOR | **✅ ACCEPTED** | Dual-energy + helical deferred. See Section 2.9. |
| **C10** | MINOR | **✅ RESOLVED** | `create_aquilion_one` kept, add helical kwargs. See Section 8.4. |
| **C12** | MINOR | **✅ RESOLVED** | `is_helical = geom.pitch > 0.0` in `backproject!`. See Section 4.7. |
| **C13** | MINOR | **✅ ACCEPTED** | n_angles naming ambiguity accepted with docstring. See Section 4.13. |
| **C14** | IMPORTANT | **✅ RESOLVED** | `recon_center` conditional in forward projector. Exact diff in Section 8.3. |
| **C15** | IMPORTANT | **✅ RESOLVED** | `create_aquilion_one` adds helical kwargs to its loop. See Section 8.4. |
| **C16** | MINOR | **✅ ACCEPTED** | GPU memory limits for multi-rotation documented. See Section 2.4. |
| **C17** | MINOR | **✅ RESOLVED** | `scan_length_cm` docstring clarification. See Section 4.9. |
| **C18** | MINOR | **✅ NOTED** | Heel effect uses `geom.fov[1]` — pre-existing, not helical-specific. |

### 8.1 Implementation Phases (3 Stories)

#### Story 1: Helical Geometry + Forward Projection (Foundation)

**Goal:** Generate helical Z(θ) positions. Forward project through helical geometry. Validate sinogram.

**Changes:**

| # | File | Change | Lines |
|---|------|--------|-------|
| 1 | `protocol.jl` | Add `pitch::Float64` field to CTProtocol struct | After `n_rotations` |
| 2 | `protocol.jl` | Add `pitch` kwarg to keyword constructor + validation | In kwargs section |
| 3 | `protocol.jl` | Insert `_pitch` in 3 inner constructor calls | Lines ~118, ~418, ~461 |
| 4 | `protocol.jl` | Add helical validation rules to `validate_protocol` | Lines ~160-226 |
| 5 | `scanner.jl` | Add 3 fields to CTGeometry struct: `pitch`, `views_per_rotation`, `recon_center` | After `fov` |
| 6 | `scanner.jl` | Rewrite CTGeometry constructor for helical Z(θ) | Lines 609-699 |
| 7 | `scanner.jl` | Update `create_aquilion_one` with helical kwargs + Z(θ) | Lines 725-807 |
| 8 | `fdk.jl` | Propagate 3 new fields in FOV-override constructor | Line 426 |
| 9 | `workspace.jl` | Propagate 3 new fields in PCCT native geom | Line 364 |
| 10 | `workspace.jl` | Pass `pitch`, `n_rotations` to CTGeometry in create_workspace (PCCT) | Line 174 |
| 11 | `workspace.jl` | Pass `pitch`, `n_rotations` to CTGeometry in create_eict_workspace | Line 515 |
| 12 | `mbir.jl` | Propagate 3 new fields in ordered-subset geometry | Line 434 |
| 13 | `scanners.jl` | Add 3 default fields in scanner factory | Line 327 |

**Validation:** T1 unit tests (Z(θ), pitch=0 degeneration, protocol validation), T2 axial regression, T3 helical sinogram shape.

**Pre-implementation:** Capture regression reference values from current axial tests.

#### Story 2: Naive Helical FDK Reconstruction

**Goal:** Reconstruct helical sinograms with corrected scaling. Validate with uniform cylinder.

**Changes:**

| # | File | Change |
|---|------|--------|
| 1 | `backprojection.jl:335` | Change `pi_over_angles = T(π) / T(n_angles)` to branch on `is_helical` |
| 2 | `backprojection.jl:316-318` | Add `recon_center` Z offset to `vol_min_z` |
| 3 | `siddon.jl:440-442` | Add `recon_center` Z offset for iterative recon path |
| 4 | `affine.jl:69-71` | Add `recon_center` offset to `tz` |
| 5 | `affine.jl:128-130` | Add `recon_center` offset to `roz` |
| 6 | `protocol.jl` | Fix `compute_ctdi_vol` to divide by pitch |
| 7 | `protocol.jl` | Fix `compute_dlp` to use scan_length directly |

**Exact backprojection change (backprojection.jl:316-335):**

```julia
# BEFORE:
vol_min_z = T(-geom.fov[3] / 2)
# ...
pi_over_angles = T(π) / T(n_angles)

# AFTER:
vol_min_z = T(-geom.fov[3] / 2 + geom.recon_center[3])
vol_min_x = T(-geom.fov[1] / 2 + geom.recon_center[1])
vol_min_y = T(-geom.fov[2] / 2 + geom.recon_center[2])
# ...
is_helical = geom.pitch > 0.0
pi_over_angles = T(π) / T(geom.views_per_rotation)  # was: T(n_angles)
```

For axial: `recon_center = (0,0,0)` and `views_per_rotation == n_angles` → identical.

**Validation:** T1 Tam-Danielsson, T2 full pipeline regression, T3 uniform cylinder, T3 SIRT convergence.

#### Story 3: WFBP Helical Weighting (Clinical Quality)

**Goal:** Add W(q̂) weight function to backprojection kernel for pitch-independent quality.

**Changes:**

| # | File | Change |
|---|------|--------|
| 1 | `backprojection.jl` | Add `backproject_voxel_helical` function (new, ~80 lines) |
| 2 | `backprojection.jl` | Modify `backproject!` to select between axial and helical voxel functions |

**Complete `backproject_voxel_helical` (implementation-ready):**

```julia
@inline function backproject_voxel_helical(
    sinogram::AbstractArray{T, 3},
    voxel_x::T, voxel_y::T, voxel_z::T,
    source_positions::AbstractArray{T, 2},
    detector_centers::AbstractArray{T, 2},
    detector_u::AbstractArray{T, 2},
    detector_v::AbstractArray{T, 2},
    n_cols::Int32, n_rows::Int32, n_angles::Int32,
    col_center::T, row_center::T,
    pixel_mag::T, pixel_row_mag::T,
    SAD::T, SAD_sq::T,
    two_pi_over_vpr::T,     # 2π / views_per_rotation
    Q_flat::T,              # 0.6 (flat-top width of W(q̂))
    n_rows_half::T          # n_rows / 2 (for q̂ normalization)
) where T

    val_acc = zero(T)   # Σ(W × w_fdk × p̂)
    wgt_acc = zero(T)   # Σ(W × w_fdk)

    for angle in Int32(1):n_angles
        # --- Geometry (IDENTICAL to backproject_voxel lines 52-110) ---
        src_x = source_positions[1, angle]
        src_y = source_positions[2, angle]
        src_z = source_positions[3, angle]

        dcx = detector_centers[1, angle]
        dcy = detector_centers[2, angle]
        dcz = detector_centers[3, angle]

        dux = detector_u[1, angle]
        duy = detector_u[2, angle]
        duz = detector_u[3, angle]

        dvx = detector_v[1, angle]
        dvy = detector_v[2, angle]
        dvz = detector_v[3, angle]

        sv_x = voxel_x - src_x
        sv_y = voxel_y - src_y
        sv_z = voxel_z - src_z

        sd_x = dcx - src_x
        sd_y = dcy - src_y
        sd_z = dcz - src_z

        sd_len_sq = sd_x^2 + sd_y^2 + sd_z^2
        sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z

        if abs(sv_dot_sd) < T(1e-10)
            continue
        end

        t = sd_len_sq / sv_dot_sd

        proj_x = src_x + t * sv_x
        proj_y = src_y + t * sv_y
        proj_z = src_z + t * sv_z

        dp_x = proj_x - dcx
        dp_y = proj_y - dcy
        dp_z = proj_z - dcz

        u = (dp_x * dux + dp_y * duy + dp_z * duz) / pixel_mag
        v = (dp_x * dvx + dp_y * dvy + dp_z * dvz) / pixel_row_mag

        col_f = u + col_center
        row_f = v + row_center

        # --- Bounds check ---
        if col_f >= one(T) && col_f <= T(n_cols) && row_f >= one(T) && row_f <= T(n_rows)
            # --- Bilinear interpolation (IDENTICAL to backproject_voxel lines 115-134) ---
            col_lo = unsafe_trunc(Int32, col_f)
            col_hi = col_lo + Int32(1)
            row_lo = unsafe_trunc(Int32, row_f)
            row_hi = row_lo + Int32(1)

            w_col = col_f - T(col_lo)
            w_row = row_f - T(row_lo)

            col_lo = clamp(col_lo, Int32(1), n_cols)
            col_hi = clamp(col_hi, Int32(1), n_cols)
            row_lo = clamp(row_lo, Int32(1), n_rows)
            row_hi = clamp(row_hi, Int32(1), n_rows)

            val = (one(T) - w_col) * (one(T) - w_row) * sinogram[col_lo, row_lo, angle] +
                  w_col * (one(T) - w_row) * sinogram[col_hi, row_lo, angle] +
                  (one(T) - w_col) * w_row * sinogram[col_lo, row_hi, angle] +
                  w_col * w_row * sinogram[col_hi, row_hi, angle]

            # --- FDK distance weight ---
            dist_sq = sv_x^2 + sv_y^2 + sv_z^2
            fdk_w = SAD_sq / dist_sq

            # --- WFBP helical weight W(q̂) ---
            q_abs = abs(row_f - row_center) / n_rows_half
            w_h = if q_abs < Q_flat
                one(T)
            elseif q_abs < one(T)
                cos(T(π) / T(2) * (q_abs - Q_flat) / (one(T) - Q_flat))^2
            else
                zero(T)
            end

            w_total = fdk_w * w_h
            val_acc += val * w_total
            wgt_acc += w_total
        end
    end

    # Weight-normalized reconstruction: (2π/N) × Σ(W·fdk·p̂) / Σ(W·fdk)
    return wgt_acc > T(1e-10) ? val_acc * two_pi_over_vpr / wgt_acc : zero(T)
end
```

**Modified `backproject!` dispatch (backprojection.jl:370-396):**

```julia
if weighted
    is_helical = geom.pitch > 0.0
    if is_helical
        two_pi_over_vpr = T(2π) / T(geom.views_per_rotation)
        Q_flat = T(0.6)
        n_rows_half = T(n_rows) / T(2)

        AK.foreachindex(volume) do idx
            idx_0 = Int32(idx - 1)
            ix = (idx_0 % nx) + Int32(1)
            idx_0 = idx_0 ÷ nx
            iy = (idx_0 % ny) + Int32(1)
            iz = (idx_0 ÷ ny) + Int32(1)

            voxel_x = vol_min_x + (T(ix) - half) * voxel_size_x
            voxel_y = vol_min_y + (T(iy) - half) * voxel_size_y
            voxel_z = vol_min_z + (T(iz) - half) * voxel_size_z

            volume[idx] = backproject_voxel_helical(
                sinogram, voxel_x, voxel_y, voxel_z,
                source_positions, detector_centers,
                detector_u, detector_v,
                n_cols, n_rows, n_angles,
                col_center, row_center,
                pixel_mag, pixel_row_mag,
                SAD, SAD_sq, two_pi_over_vpr, Q_flat, n_rows_half
            )
        end
    else
        # Standard axial FDK (UNCHANGED from current code)
        AK.foreachindex(volume) do idx
            # ... existing backproject_voxel call (unchanged) ...
        end
    end
end
```

**For axial (`is_helical=false`):** The code path is IDENTICAL to current — same `backproject_voxel` function with same `pi_over_angles` constant. **Zero performance impact on axial.**

**Validation:** T1 W(q̂) unit tests, T3 WFBP vs naive comparison, T4 Gammex HU/noise/z-uniformity, T4 pitch sweep.

### 8.2 CTGeometry Inner Constructor — All 7 Sites (Complete Diffs)

Every positional `CTGeometry(...)` call must add 3 new trailing arguments: `pitch`, `views_per_rotation`, `recon_center`.

**Site 1 — `scanner.jl:694` (main constructor):**
```julia
# BEFORE:
return CTGeometry(
    SAD, SDD, n_angles, _n_rows, _n_cols, pixel_size, pixel_row_size,
    angles, source_positions, detector_centers, detector_u, detector_v,
    fov
)

# AFTER:
return CTGeometry(
    SAD, SDD, total_views, _n_rows, _n_cols, pixel_size, pixel_row_size,
    angles, source_positions, detector_centers, detector_u, detector_v,
    fov,
    pitch, n_angles, (0.0, 0.0, recon_center_z)
)
# Note: total_views stored in n_angles field; n_angles (param) stored in views_per_rotation
```

**Site 2 — `scanner.jl:802` (`create_aquilion_one`):**
```julia
# BEFORE:
return CTGeometry(
    SAD, SDD, n_angles, n_rows, n_cols, pixel_size, pixel_row_size,
    angles, source_positions, detector_centers, detector_u, detector_v,
    fov
)

# AFTER:
return CTGeometry(
    SAD, SDD, total_views, n_rows, n_cols, pixel_size, pixel_row_size,
    angles, source_positions, detector_centers, detector_u, detector_v,
    fov,
    pitch, n_angles, (0.0, 0.0, 0.0)
)
```

**Site 3 — `fdk.jl:426` (FOV override):**
```julia
# BEFORE:
geom_fov = CTGeometry(
    geom.SAD, geom.SDD,
    geom.n_angles, geom.n_rows, geom.n_cols, geom.pixel_size, geom.pixel_row_size,
    geom.angles, geom.source_positions, geom.detector_centers,
    geom.detector_u, geom.detector_v,
    fov
)

# AFTER:
geom_fov = CTGeometry(
    geom.SAD, geom.SDD,
    geom.n_angles, geom.n_rows, geom.n_cols, geom.pixel_size, geom.pixel_row_size,
    geom.angles, geom.source_positions, geom.detector_centers,
    geom.detector_u, geom.detector_v,
    fov,
    geom.pitch, geom.views_per_rotation, geom.recon_center
)
```

**Site 4 — `workspace.jl:364` (PCCT native geometry):**
```julia
# BEFORE:
_native_geom = CTGeometry(
    geom.SAD, geom.SDD, geom.n_angles, _native_n_rows, _native_n_cols,
    _native_pixel_size, _native_pixel_row_size,
    geom.angles,
    geom.source_positions, geom.detector_centers,
    geom.detector_u, geom.detector_v,
    geom.fov
)

# AFTER:
_native_geom = CTGeometry(
    geom.SAD, geom.SDD, geom.n_angles, _native_n_rows, _native_n_cols,
    _native_pixel_size, _native_pixel_row_size,
    geom.angles,
    geom.source_positions, geom.detector_centers,
    geom.detector_u, geom.detector_v,
    geom.fov,
    geom.pitch, geom.views_per_rotation, geom.recon_center
)
```

**Site 5 — `mbir.jl:434` (ordered-subset slicing):**
```julia
# BEFORE:
return CTGeometry(
    geom.SAD, geom.SDD,
    n_subset, geom.n_rows, geom.n_cols,
    geom.pixel_size, geom.pixel_row_size,
    angles_subset, source_positions_subset, detector_centers_subset,
    detector_u_subset, detector_v_subset,
    geom.fov
)

# AFTER:
return CTGeometry(
    geom.SAD, geom.SDD,
    n_subset, geom.n_rows, geom.n_cols,
    geom.pixel_size, geom.pixel_row_size,
    angles_subset, source_positions_subset, detector_centers_subset,
    detector_u_subset, detector_v_subset,
    geom.fov,
    geom.pitch, geom.views_per_rotation, geom.recon_center
)
# Note: views_per_rotation stays same (it's a property of the scan, not the subset)
```

**Site 6 — `scanners.jl:327` (scanner factory):**
```julia
# BEFORE:
return CTGeometry(
    sid_cm, sdd_cm, n_angles, n_rows, _n_cols, pixel_size_cm, pixel_row_size_cm,
    angles, source_positions, detector_centers, detector_u, detector_v,
    fov
)

# AFTER:
return CTGeometry(
    sid_cm, sdd_cm, n_angles, n_rows, _n_cols, pixel_size_cm, pixel_row_size_cm,
    angles, source_positions, detector_centers, detector_u, detector_v,
    fov,
    0.0, n_angles, (0.0, 0.0, 0.0)  # axial defaults
)
```

### 8.3 `recon_center` Offset — All 5 Sites (Complete Diffs)

The `recon_center` field offsets the reconstruction volume origin in world coordinates.

**Site 1 — `backprojection.jl:316-318` (FDK + matched backprojection):**
```julia
# BEFORE:
vol_min_x = T(-geom.fov[1] / 2)
vol_min_y = T(-geom.fov[2] / 2)
vol_min_z = T(-geom.fov[3] / 2)

# AFTER:
vol_min_x = T(-geom.fov[1] / 2 + geom.recon_center[1])
vol_min_y = T(-geom.fov[2] / 2 + geom.recon_center[2])
vol_min_z = T(-geom.fov[3] / 2 + geom.recon_center[3])
```
For axial: `recon_center = (0,0,0)` → identical.

**Site 2 — `siddon.jl:440-442` (forward projector, iterative recon path ONLY):**
```julia
# BEFORE:
vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
vol_min_x = T(-vol_bounds[1] / 2)
vol_min_y = T(-vol_bounds[2] / 2)
vol_min_z = T(-vol_bounds[3] / 2)
vol_max_x = T(vol_bounds[1] / 2)
vol_max_y = T(vol_bounds[2] / 2)
vol_max_z = T(vol_bounds[3] / 2)

# AFTER:
vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
# recon_center only applies when using geom.fov (iterative recon path).
# When volume_extent is provided (phantom projection), phantom is at origin.
_rc = volume_extent !== nothing ? (0.0, 0.0, 0.0) : geom.recon_center
vol_min_x = T(-vol_bounds[1] / 2 + _rc[1])
vol_min_y = T(-vol_bounds[2] / 2 + _rc[2])
vol_min_z = T(-vol_bounds[3] / 2 + _rc[3])
vol_max_x = T(vol_bounds[1] / 2 + _rc[1])
vol_max_y = T(vol_bounds[2] / 2 + _rc[2])
vol_max_z = T(vol_bounds[3] / 2 + _rc[3])
```
**Critical subtlety (C14):** The recon_center ONLY applies in the `geom.fov` path (iterative recon projecting the recon volume). When `volume_extent` is provided (phantom forward projection), the phantom is always centered at world origin — NO offset. This is enforced by the `_rc` conditional.

**Site 3 — `affine.jl:69-71` (recon_to_world_affine):**
```julia
# BEFORE:
tx = -fov_x / 2 + sx / 2
ty = -fov_y / 2 + sy / 2
tz = -fov_z / 2 + sz / 2

# AFTER:
rcx, rcy, rcz = geom.recon_center
tx = -fov_x / 2 + sx / 2 + rcx
ty = -fov_y / 2 + sy / 2 + rcy
tz = -fov_z / 2 + sz / 2 + rcz
```

**Site 4 — `affine.jl:128-130` (resample_to_recon):**
```julia
# BEFORE:
rox = -fov_x / 2 + rvx / 2
roy = -fov_y / 2 + rvy / 2
roz = -fov_z / 2 + rvz / 2

# AFTER:
rcx, rcy, rcz = geom.recon_center
rox = -fov_x / 2 + rvx / 2 + rcx
roy = -fov_y / 2 + rvy / 2 + rcy
roz = -fov_z / 2 + rvz / 2 + rcz
```

**Site 5 — `fdk.jl:454-481` (apply_fov_mask!):**
```julia
# BEFORE:
x = (T(ix) - T(0.5) - T(nx) / T(2)) * voxel_x
y = (T(iy) - T(0.5) - T(ny) / T(2)) * voxel_y

# AFTER:  (recon_center doesn't change the circular mask radius, only the center)
# Actually, the FOV mask is always centered at isocenter regardless of recon_center,
# so NO change is needed here. The mask checks distance from (0,0), which is correct.
```
**Decision: No change needed at Site 5.** The FOV mask clips based on distance from the XY origin (isocenter), not from the recon center. This is physically correct — the reconstruction circle is always centered at the scanner isocenter.

### 8.4 `create_aquilion_one` Helical Extension

**Decision: Add helical kwargs to `create_aquilion_one` directly (no refactoring to delegate).**

Rationale: The function has specific pixel_size logic (1.1× margin for fov_cm override) that differs from the main CTGeometry constructor. Refactoring to delegate would require reconciling this, adding complexity with no benefit. Instead, simply add helical Z(θ) to the existing loop.

```julia
# CHANGE function signature (line 725):
function create_aquilion_one(;
    n_angles::Int=360,
    n_rows::Int=64,
    n_cols::Int=128,
    fov_cm::Union{Float64,Nothing}=nothing,
    z_cm::Union{Float64,Nothing}=nothing,
    sad::Union{Float64,Nothing}=nothing,
    sdd::Union{Float64,Nothing}=nothing,
    pitch::Float64=0.0,             # NEW
    n_rotations::Float64=1.0        # NEW
)

# ADD after z FOV computation (after line 762):
    total_collim_cm = pixel_row_size * n_rows
    total_views = pitch > 0.0 ? round(Int, n_angles * n_rotations) : n_angles
    z_travel = n_rotations * pitch * total_collim_cm
    z_start = -z_travel / 2.0
    if pitch > 0.0 && z_cm === nothing
        fov_z = max(z_travel - total_collim_cm, total_collim_cm)
    end

# CHANGE angles (line 765):
    angles = collect(range(0.0, 2π * n_rotations - 2π * n_rotations / total_views,
                           length=total_views))

# CHANGE matrix allocation (lines 768-771):
    source_positions = Matrix{Float64}(undef, 3, total_views)
    detector_centers = Matrix{Float64}(undef, 3, total_views)
    detector_u = Matrix{Float64}(undef, 3, total_views)
    detector_v = Matrix{Float64}(undef, 3, total_views)

# CHANGE inside loop (lines 780, 786):
    for (i, θ) in enumerate(angles)
        # ... existing XY computation ...
        z_i = z_start + (θ / (2π)) * pitch * total_collim_cm  # 0.0 for axial
        source_positions[3, i] = z_i
        detector_centers[3, i] = z_i
        # ... existing u,v computation ...
    end

# CHANGE return (line 802):
    return CTGeometry(
        SAD, SDD, total_views, n_rows, n_cols, pixel_size, pixel_row_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov,
        pitch, n_angles, (0.0, 0.0, 0.0)
    )
```

### 8.5 CTGeometry Constructor — Complete Helical Logic

**The constructor body (`scanner.jl:617-698`) changes to:**

```julia
function CTGeometry(scanner::Scanner{T};
    n_angles::Int = 360,              # views per rotation
    pitch::Float64 = 0.0,            # IEC beam pitch (0.0 = axial)
    n_rotations::Float64 = 1.0,      # gantry rotations
    recon_center_z::Float64 = 0.0,   # recon volume Z center (cm)
    fov_cm::Union{Float64,Nothing} = nothing,
    z_cm::Union{Float64,Nothing} = nothing,
    n_rows::Union{Int,Nothing} = nothing,
    n_cols::Union{Int,Nothing} = nothing,
    collimation_mm::Union{Float64,Nothing} = nothing
) where T

    # ... existing collimation/detector logic (lines 618-631, unchanged) ...
    # ... existing SAD/SDD/pixel_size conversion (lines 633-639, unchanged) ...
    # ... existing fov_xy computation (lines 641-647, unchanged) ...

    # Total collimation for pitch calculation
    total_collim_cm = _n_rows * scanner.detector_row_size / 10.0  # mm → cm

    # Total views for helical
    total_views = pitch > 0.0 ? round(Int, n_angles * n_rotations) : n_angles

    # Z travel and start
    z_travel = n_rotations * pitch * total_collim_cm
    z_start = -z_travel / 2.0

    # Z FOV — helical auto-computation (Tam-Danielsson window)
    if z_cm !== nothing
        fov_z = z_cm
    elseif pitch > 0.0
        fov_z = max(z_travel - total_collim_cm, total_collim_cm)
    else
        z_coverage_mm = _n_rows * scanner.detector_row_size
        fov_z = z_coverage_mm / 10.0  # mm → cm (unchanged axial path)
    end

    # Generate angles (monotonically increasing for helical)
    angles = collect(range(0.0,
        2π * n_rotations - 2π * n_rotations / total_views,
        length=total_views))

    # Pre-compute all positions
    source_positions = Matrix{Float64}(undef, 3, total_views)
    detector_centers = Matrix{Float64}(undef, 3, total_views)
    detector_u = Matrix{Float64}(undef, 3, total_views)
    detector_v = Matrix{Float64}(undef, 3, total_views)

    for (i, θ) in enumerate(angles)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # Z position from helical formula (0.0 for pitch=0)
        z_i = z_start + (θ / (2π)) * pitch * total_collim_cm

        source_positions[1, i] = -SAD * sinθ
        source_positions[2, i] = -SAD * cosθ
        source_positions[3, i] = z_i

        det_dist = SDD - SAD
        detector_centers[1, i] = det_dist * sinθ
        detector_centers[2, i] = det_dist * cosθ
        detector_centers[3, i] = z_i

        detector_u[1, i] = cosθ
        detector_u[2, i] = -sinθ
        detector_u[3, i] = 0.0

        detector_v[1, i] = 0.0
        detector_v[2, i] = 0.0
        detector_v[3, i] = 1.0
    end

    fov = (fov_xy, fov_xy, fov_z)

    return CTGeometry(
        SAD, SDD, total_views, _n_rows, _n_cols, pixel_size, pixel_row_size,
        angles, source_positions, detector_centers, detector_u, detector_v,
        fov,
        pitch, n_angles, (0.0, 0.0, recon_center_z)
    )
end
```

### 8.6 Dependencies Between Stories

```
Story 1 (Geometry + Forward Projection)
    │
    ├── Story 2 (Naive Helical FDK)  ← requires helical geometry
    │       │
    │       └── Story 3 (WFBP)  ← requires naive FDK as baseline
    │
    └── (PCCT + Helical works after Story 1 with iterative recon)
```

**Story 1 is the foundation.** After Story 1, helical forward projection works, and iterative recon (SIRT/CGLS) works for helical. This is a useful checkpoint — users can do helical simulation + iterative reconstruction.

**Story 2 enables FDK for helical.** Correct for pitch ≈ 1.

**Story 3 enables clinical-quality FDK.** Correct for all pitch values.

### 8.7 Files Changed Per Story

| File | Story 1 | Story 2 | Story 3 |
|------|---------|---------|---------|
| `protocol.jl` | **YES** (pitch field, validation, dose) | YES (dose fix) | — |
| `scanner.jl` | **YES** (struct + constructor + create_aquilion_one) | — | — |
| `workspace.jl` | **YES** (2 create sites + PCCT native geom) | — | — |
| `scanners.jl` | **YES** (factory constructor) | — | — |
| `fdk.jl` | **YES** (FOV-override constructor) | — | — |
| `mbir.jl` | **YES** (ordered-subset constructor) | — | — |
| `backprojection.jl` | — | **YES** (recon_center + pi_over_angles) | **YES** (helical kernel) |
| `siddon.jl` | — | **YES** (recon_center in iterative path) | — |
| `affine.jl` | — | **YES** (recon_center offset) | — |

**Total: ~10 files, ~25 change sites, 0 new files.**

### 8.8 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Axial regression** | HIGH | T2 regression tests captured before any changes |
| **PCCT native geom breaks** | MEDIUM | Propagate all 3 new fields (Site 4 in Section 8.2) |
| **MBIR subset geom breaks** | MEDIUM | `views_per_rotation` stays same in subset (Site 5) |
| **Normalization wrong** | HIGH | Uniform cylinder test catches immediately (Section 7.4.2) |
| **GPU memory for long scans** | LOW | Warning at >5 rotations, document limit |
| **Dual-energy + helical** | LOW | Explicitly deferred, validation warning added |

### 8.9 Estimated Scope

- **Story 1:** ~150 lines changed across 6 files + ~50 lines of tests
- **Story 2:** ~30 lines changed across 3 files + ~100 lines of tests
- **Story 3:** ~120 lines new code (helical kernel) + ~50 lines of tests
- **Total:** ~350 lines of production code, ~200 lines of tests

---
