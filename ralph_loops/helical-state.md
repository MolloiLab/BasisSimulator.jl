# Helical CT Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** HELI-002 DISCOVERY complete → next: HELI-003 DISCOVERY
**Current topic:** HELI-003 (Helical Reconstruction Algorithms — FDK-Helical vs Katsevich vs Rebinning)

## What was done

**HELI-000 DISCOVERY (Iteration 1):** Complete geometry and projection audit.
**HELI-001 DISCOVERY (Iteration 2):** Helical CT geometry — pitch, Z(θ), clinical values, phantom coverage.
**HELI-002 DISCOVERY (Iteration 3):** Forward projection audit — confirmed entire pipeline is helical-ready.

Key findings so far:
- CTGeometry struct is already general (per-angle [3,N] position matrices)
- Z=0 is hardcoded in only 2 places: CTGeometry constructor and create_aquilion_one
- Forward projection (Siddon + polychromatic + PCCT), all 16 physics effects, and iterative recon work UNCHANGED
- Only geometry construction + FDK reconstruction need modification
- `n_rotations` field already exists in CTProtocol; only `pitch` is missing
- **The codebase is ~90% helical-ready — reconstruction is the ONLY significant addition**

## What to do next

**HELI-003 DISCOVERY — Helical Reconstruction Algorithms**

This is the CRITICAL research topic. Research the following reconstruction approaches:

1. **FDK with helical weighting (approximate):**
   - Standard approach: apply redundancy weights before FDK backprojection
   - Noo/Defrise/Clackdoyle 2003 weights (smooth normalized weights)
   - Parker-type weights extended to helical (HWS: Helical Weighting Scheme)
   - Turbell 2001 PhD thesis — comprehensive treatment of helical FDK
   - **Find the exact weight formula** — what is w(θ, u, v) for helical FDK?
   - How does it degenerate to standard FDK for pitch=0?
   - Quality: good for pitch ≤ 1.5, artifacts increase with pitch
   - **This is the PRIMARY candidate** for BasisSimulator

2. **Katsevich exact algorithm (2002):**
   - PI-line based exact reconstruction for helical cone-beam
   - Theoretically exact but complex implementation
   - Requires computing PI-line intervals and Hilbert transforms
   - GPU implementation exists (TIGRE has partial, RTK has C++)
   - Computational cost comparison vs FDK+weights
   - **Is the quality improvement worth the complexity?**

3. **Rebinning approaches:**
   - ASSR (Advanced Single-Slice Rebinning, Noo et al. 1999)
   - Full 3D rebinning to parallel beam + 2D FBP
   - Fastest but lowest quality — significant cone-beam artifacts for wide detectors
   - Not recommended as primary, but worth documenting

4. **What clinical scanners actually use:**
   - Siemens SAFIRE/WFBP (weighted FBP)
   - GE ASiR/TrueFidelity
   - Most use approximate FDK-type with vendor-specific weights

5. **GPU kernel design for the chosen algorithm:**
   - How does the helical weight integrate into the existing backprojection kernel?
   - Is it a per-voxel-per-angle weight that multiplies the existing FDK weight?
   - Can it be computed on-the-fly or must it be pre-computed?

### Phase rotation plan

```
HELI-000 (priority 0): Discovery ✓ → Critique → Refinement
HELI-001 (priority 1): Discovery ✓ → Critique → Refinement
HELI-002 (priority 1): Discovery ✓ → Critique → Refinement
HELI-003 (priority 1): Discovery [NEXT] → Critique → Refinement
HELI-004 (priority 1): Discovery → Critique → Refinement
HELI-005 (priority 1): Discovery → Critique → Refinement
HELI-006 (priority 1): Discovery → Critique → Refinement
HELI-007 (priority 2): Discovery → Critique → Refinement
HELI-008 (priority 3, SYNTHESIS): Blocked until P0+P1 topics complete
```

**Note:** After HELI-003 discovery, the next iteration should be a CRITIQUE cycle covering HELI-000 through HELI-003 together. We've done 3 discoveries in a row — time for a critique.

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| HELI-000 Current Geometry Audit | **DONE** | open | open |
| HELI-001 Helical Geometry | **DONE** | open | open |
| HELI-002 Forward Projection | **DONE** | open | open |
| HELI-003 Reconstruction Algorithms | **NEXT** | open | open |
| HELI-004 API Integration | open | open | open |
| HELI-005 Physics Pipeline Compat | open | open | open |
| HELI-006 Reference Implementations | open | open | open |
| HELI-007 Validation Strategy | open | open | open |
| HELI-008 SYNTHESIS | open | open | open |

## Completed iterations

1. **HELI-000 Discovery** (2026-03-17) — Full geometry audit. Found codebase is ~90% helical-ready.
2. **HELI-001 Discovery** (2026-03-17) — Helical geometry: pitch, Z(θ), clinical values, phantom coverage.
3. **HELI-002 Discovery** (2026-03-17) — Forward projection audit. Entire pipeline works unchanged for helical.
