# Helical CT Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** HELI-000 DISCOVERY complete → next: HELI-001 DISCOVERY
**Current topic:** HELI-001 (Helical CT Geometry — Table Motion, Pitch, Coordinate Transforms)

## What was done

**HELI-000 DISCOVERY (Iteration 1):** Complete geometry and projection audit.

Key findings:
- CTGeometry struct is already general (per-angle [3,N] position matrices)
- Z=0 is hardcoded in only 2 places: CTGeometry constructor and create_aquilion_one
- Forward projection (Siddon + polychromatic), all 16 physics effects, and iterative recon work UNCHANGED
- Only geometry construction + FDK reconstruction need modification
- `n_rotations` field already exists in CTProtocol; only `pitch` is missing
- **The codebase is ~90% helical-ready**

## What to do next

**HELI-001 DISCOVERY — Helical CT Geometry: Table Motion, Pitch, and Z(θ) Formulas**

Research the helical geometry model:

1. **Pitch definition:** `pitch = table_distance_per_rotation / beam_collimation`. Derive Z(θ) = z_start + pitch × collimation × θ/(2π). What are clinically relevant pitch values? (0.2–2.0, typical 0.5–1.5)

2. **CTGeometry constructor changes:** How to compute Z(θ) for source and detector. Should Z be relative to scan center (symmetric about Z=0)? What is z_start = -(n_rotations/2) × pitch × collimation?

3. **Multi-rotation angles:** For n_rotations R, generate R × views_per_rotation angles spanning [0, R×2π). Confirm that source_positions size becomes [3, R×views].

4. **Coordinate conventions:** BasisSimulator uses Z = inferior-superior. Table motion moves the patient through the gantry → equivalent to source/detector translating along +Z. Confirm sign convention.

5. **Phantom Z extent:** For helical, the phantom must cover the full helical travel range. What's the relationship between phantom extent, pitch, n_rotations, and collimation?

6. **Reconstruction Z range:** Unlike axial (fixed Z-FOV), helical allows arbitrary Z-range reconstruction. How to specify which Z-range to reconstruct?

Also research:
- How CatSim handles helical geometry (scan parameters)
- How TIGRE specifies helical orbits
- Clinical scanner pitch values (GE, Siemens, Canon)

### Phase rotation plan

```
HELI-000 (priority 0): Discovery ✓ → Critique → Refinement
HELI-001 (priority 1): Discovery [NEXT] → Critique → Refinement
HELI-002 through HELI-006 (priority 1): Discovery → Critique → Refinement
HELI-007 (priority 2): Discovery → Critique → Refinement
HELI-008 (priority 3, SYNTHESIS): Blocked until P0+P1 topics complete
```

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| HELI-000 Current Geometry Audit | **DONE** | open | open |
| HELI-001 Helical Geometry | **NEXT** | open | open |
| HELI-002 Forward Projection | open | open | open |
| HELI-003 Reconstruction Algorithms | open | open | open |
| HELI-004 API Integration | open | open | open |
| HELI-005 Physics Pipeline Compat | open | open | open |
| HELI-006 Reference Implementations | open | open | open |
| HELI-007 Validation Strategy | open | open | open |
| HELI-008 SYNTHESIS | open | open | open |

## Completed iterations

1. **HELI-000 Discovery** (2026-03-17) — Full geometry audit. Found codebase is ~90% helical-ready.
