# Helical CT Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** CRITIQUE of HELI-004 + HELI-005 complete → next: HELI-007 DISCOVERY or REFINEMENT
**Current topic:** Choose between HELI-007 DISCOVERY (Validation) or REFINEMENT (resolve all critique issues)

## What was done

**HELI-000 DISCOVERY (Iteration 1):** Complete geometry and projection audit.
**HELI-001 DISCOVERY (Iteration 2):** Helical CT geometry — pitch, Z(θ), clinical values, phantom coverage.
**HELI-002 DISCOVERY (Iteration 3):** Forward projection audit — confirmed entire pipeline is helical-ready.
**HELI-003 DISCOVERY (Iteration 4):** Reconstruction algorithms — comprehensive research.
**HELI-000–003 CRITIQUE (Iteration 5):** Cross-referenced spec sections 0-3 against source code. 5 critical + 5 important issues found.
**HELI-004 DISCOVERY (Iteration 6a):** API design — CTProtocol pitch field, CTGeometry struct additions, workspace changes, dose formulas, validation.
**HELI-005 DISCOVERY (Iteration 6b):** Physics pipeline audit — confirmed all 16 effects are helical-compatible with zero changes.
**HELI-004+005 CRITIQUE (Iteration 7):** Cross-referenced sections 4-5 against source code. 4 important + 2 minor issues found (C13-C18). No critical issues.

### Key findings from Iteration 7 (HELI-004+005 Critique)

**4 IMPORTANT issues:**
1. **C13:** `n_angles` naming ambiguity (constructor param = views_per_rotation, struct field = total_views). Accept as tradeoff, add clear docs.
2. **C14:** `recon_center` offset must NOT apply to phantom forward projection (only iterative recon). Refines C6 site #1.
3. **C15:** `create_aquilion_one` duplicates geometry loop — refactor to delegate to `CTGeometry(scanner; ...)` instead of maintaining two copies of helical Z formula.
4. **C16:** GPU memory scaling (n_rotations × buffer sizes) not documented in Section 4.5. Cross-reference Section 2.4.

**2 MINOR issues:**
5. **C17:** `compute_dlp` fix needs `scan_length_cm` semantics clarification.
6. **C18:** Heel effect `fan_angle_max` uses recon FOV not detector fan (pre-existing, not helical).

**Overall: Sections 4 and 5 are SOLID.** No critical issues. API design is clean and physics pipeline is 100% helical-compatible.

## What to do next

**RECOMMENDED: HELI-007 DISCOVERY (Validation Strategy)** — define test cases, acceptance criteria, ground truth sources. This is the last major discovery topic before REFINEMENT and SYNTHESIS.

**ALTERNATIVE: REFINEMENT of Sections 0-4** — resolve all C1-C18 critique issues with implementation-ready detail. The most urgent unresolved items are:
- C1 (normalization formula) — requires reading FreeCT_wFBP source
- C15 (create_aquilion_one refactor) — straightforward but needs pixel_size reconciliation
- C14 (recon_center conditional) — needs exact code diff

**Suggested path:** HELI-007 DISCOVERY next, then a comprehensive REFINEMENT pass across all sections, then HELI-008 SYNTHESIS.

### Phase rotation plan

```
HELI-000 (priority 0): Discovery ✓ → Critique ✓ → Refinement
HELI-001 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-002 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-003 (priority 1): Discovery ✓ → Critique ✓ → Refinement (needs C1 resolution)
HELI-004 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-005 (priority 1): Discovery ✓ → Critique ✓ → Refinement (minimal — physics confirmed unchanged)
HELI-006 (priority 1): Discovery ✓ (partial, via HELI-003) → Critique → Refinement
HELI-007 (priority 2): Discovery [NEXT] → Critique → Refinement
HELI-008 (priority 3, SYNTHESIS): Blocked until P0+P1 topics complete
```

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| HELI-000 Current Geometry Audit | **DONE** | **DONE** | open |
| HELI-001 Helical Geometry | **DONE** | **DONE** | open |
| HELI-002 Forward Projection | **DONE** | **DONE** | open |
| HELI-003 Reconstruction Algorithms | **DONE** | **DONE** | open (needs C1) |
| HELI-004 API Integration | **DONE** | **DONE** | open |
| HELI-005 Physics Pipeline Compat | **DONE** | **DONE** | open (minimal) |
| HELI-006 Reference Implementations | **DONE** (partial) | open | open |
| HELI-007 Validation Strategy | **NEXT** | open | open |
| HELI-008 SYNTHESIS | open | open | open |

## Completed iterations

1. **HELI-000 Discovery** (2026-03-17) — Full geometry audit. Found codebase is ~90% helical-ready.
2. **HELI-001 Discovery** (2026-03-17) — Helical geometry: pitch, Z(θ), clinical values, phantom coverage.
3. **HELI-002 Discovery** (2026-03-17) — Forward projection audit. Entire pipeline works unchanged for helical.
4. **HELI-003 Discovery** (2026-03-17) — Reconstruction algorithms: WFBP recommended, Katsevich rejected.
5. **HELI-000–003 Critique** (2026-03-17) — 5 critical + 5 important issues found. All annotated in spec.
6. **HELI-004+005 Discovery** (2026-03-17) — API design + physics pipeline. Complete sections 4 and 5 in spec.
7. **HELI-004+005 Critique** (2026-03-17) — 4 important + 2 minor issues (C13-C18). No critical issues. Sections 4 and 5 confirmed solid.
