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

> **⚠ CRITIQUE C1, C3:** The normalization formula must be derived carefully. The
> helical weight-normalized formula does NOT degenerate to standard FDK for axial.
> This means `is_helical` controls BOTH the W(q̂) weighting AND the normalization
> strategy — not just one or the other. The `is_helical` flag is set from
> `geom.pitch > 0` inside `backproject!`.

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

> **⚠ CRITIQUE C1 (RESOLUTION PENDING):** The exact normalization factor needs to
> be derived from the WFBP integral (Stierstorfer 2004 Eq. 5) and verified
> against FreeCT_wFBP source code. The weight accumulation is done inline per
> voxel — NO separate `weight_sum` buffer is needed. See Section 3.9 for the
> GPU kernel pseudocode.
>
> The WFBP discretized integral is:
> ```
> f(x) = Δα × Σ_i [ W_norm(i,x) × w_fdk(i) × p̂(i) ]
> ```
> With total-sum normalization approximation and Δα = 2π/views_per_rotation:
> ```
> f(x) ≈ (2π / views_per_rotation) × Σ[W × w_fdk × p̂] / Σ[W × w_fdk]
> ```
> Whether this is `π` or `2π` depends on the FDK integral convention ((1/2)∫₀²π
> vs ∫₀π) — must be resolved by checking FreeCT_wFBP.

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

Normalization: f(x) = Σ(W_i * fdk_i * p̂_i) * π / Σ(W_i * fdk_i)
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

    # Normalize: divide by weight sum
    # ⚠ CRITIQUE C1: The exact normalization constant (π vs 2π/views_per_rotation)
    # must be resolved by checking FreeCT_wFBP source. Using π here as placeholder.
    return wgt_acc > T(1e-10) ? val_acc * T(π) / wgt_acc : zero(T)
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

(To be filled by HELI-006 research)

---

## 7. Validation and Testing Strategy

(To be filled by HELI-007 research)

---

## 8. SYNTHESIS: Implementation Roadmap

(To be filled by HELI-008 synthesis — blocked until all other topics complete)

---
