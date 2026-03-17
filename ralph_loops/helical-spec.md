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

(To be filled by HELI-001 research)

---

## 2. Forward Projection — Siddon with Table Motion

(To be filled by HELI-002 research)

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
