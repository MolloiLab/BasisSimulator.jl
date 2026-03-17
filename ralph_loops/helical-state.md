# Helical CT Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** CRITIQUE of HELI-000 through HELI-003 complete → next: REFINEMENT or continue with HELI-004/005 DISCOVERY
**Current topic:** Choose between REFINEMENT (fix critique issues) or DISCOVERY (HELI-004 API Design)

## What was done

**HELI-000 DISCOVERY (Iteration 1):** Complete geometry and projection audit.
**HELI-001 DISCOVERY (Iteration 2):** Helical CT geometry — pitch, Z(θ), clinical values, phantom coverage.
**HELI-002 DISCOVERY (Iteration 3):** Forward projection audit — confirmed entire pipeline is helical-ready.
**HELI-003 DISCOVERY (Iteration 4):** Reconstruction algorithms — comprehensive research.
**HELI-000–003 CRITIQUE (Iteration 5):** Cross-referenced all spec claims against actual source code.

### Key critique findings (Iteration 5)

**5 CRITICAL issues found:**
1. **C1: Three inconsistent normalization formulas** in Section 3.3.6 — must derive correct formula from WFBP integral and verify against FreeCT_wFBP
2. **C2: Naive FDK pitch claim wrong** — "pitch ≤ 1" should be "pitch ≈ 1"; overestimates by ~1/pitch for low pitch
3. **C3: Normalized formula ≠ standard FDK** — `is_helical` flag needed for normalization strategy, not just weight function
4. **C4: 6 inner constructor sites** break when CTGeometry gets new fields (spec listed only 2). Missing: fdk.jl:426, workspace.jl:364, mbir.jl:434, scanners.jl:327
5. **C5: Helical fov_z not computed** — Constructor must derive Tam-Danielsson window for default z-FOV

**5 IMPORTANT issues found:**
6. C6: `recon_center` offset needed at 4 sites (not just the projector)
7. C7: PCCT workspace has same `n_angles=protocol.views` issue
8. C8: Weight buffer contradiction (inline normalization, no buffer needed)
9. C9: Dual-energy + helical not addressed (out of scope note added)
10. C10: `create_aquilion_one` disposition unclear

All issues annotated in-place in `helical-spec.md` with `⚠ CRITIQUE` markers.

## What to do next

**RECOMMENDED: REFINEMENT of Sections 0–3** to resolve the 5 critical issues.

The most urgent task is **C1 (normalization formula)** — this requires:
1. Reading FreeCT_wFBP source code (`cuda_kernels.cuh`, backprojection kernel) to determine the exact normalization constant
2. Settling on ONE formula for the helical backprojection
3. Proving it gives correct HU for known phantoms

After C1 is resolved, **C4 and C5** can be addressed by expanding the struct change section with all 6 call sites and adding helical fov_z computation.

**ALTERNATIVE: HELI-004 DISCOVERY (API Design)** — this would address C7 (n_angles semantics) and C10 (`create_aquilion_one` disposition) naturally as part of the API design.

**Suggested path:** Do HELI-004 DISCOVERY next (it naturally resolves several critique issues), then REFINEMENT of Sections 0–3, then HELI-005 DISCOVERY.

### Phase rotation plan

```
HELI-000 (priority 0): Discovery ✓ → Critique ✓ → Refinement
HELI-001 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-002 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-003 (priority 1): Discovery ✓ → Critique ✓ → Refinement (needs C1 resolution)
HELI-004 (priority 1): Discovery [NEXT] → Critique → Refinement
HELI-005 (priority 1): Discovery → Critique → Refinement
HELI-006 (priority 1): Discovery ✓ (partial, via HELI-003) → Critique → Refinement
HELI-007 (priority 2): Discovery → Critique → Refinement
HELI-008 (priority 3, SYNTHESIS): Blocked until P0+P1 topics complete
```

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| HELI-000 Current Geometry Audit | **DONE** | **DONE** | open |
| HELI-001 Helical Geometry | **DONE** | **DONE** | open |
| HELI-002 Forward Projection | **DONE** | **DONE** | open |
| HELI-003 Reconstruction Algorithms | **DONE** | **DONE** | open (needs C1) |
| HELI-004 API Integration | **NEXT** | open | open |
| HELI-005 Physics Pipeline Compat | open | open | open |
| HELI-006 Reference Implementations | **DONE** (partial) | open | open |
| HELI-007 Validation Strategy | open | open | open |
| HELI-008 SYNTHESIS | open | open | open |

## Completed iterations

1. **HELI-000 Discovery** (2026-03-17) — Full geometry audit. Found codebase is ~90% helical-ready.
2. **HELI-001 Discovery** (2026-03-17) — Helical geometry: pitch, Z(θ), clinical values, phantom coverage.
3. **HELI-002 Discovery** (2026-03-17) — Forward projection audit. Entire pipeline works unchanged for helical.
4. **HELI-003 Discovery** (2026-03-17) — Reconstruction algorithms: WFBP recommended, Katsevich rejected.
5. **HELI-000–003 Critique** (2026-03-17) — 5 critical + 5 important issues found. All annotated in spec.
