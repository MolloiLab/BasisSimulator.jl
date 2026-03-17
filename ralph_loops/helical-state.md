# Helical CT Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** HELI_COMPLETE — All topics researched, critiqued, refined, and synthesized.
**Status:** The helical-spec.md is implementation-ready.

## What was done

**HELI-000 DISCOVERY (Iteration 1):** Complete geometry and projection audit.
**HELI-001 DISCOVERY (Iteration 2):** Helical CT geometry — pitch, Z(θ), clinical values, phantom coverage.
**HELI-002 DISCOVERY (Iteration 3):** Forward projection audit — confirmed entire pipeline is helical-ready.
**HELI-003 DISCOVERY (Iteration 4):** Reconstruction algorithms — comprehensive research.
**HELI-000–003 CRITIQUE (Iteration 5):** Cross-referenced spec sections 0-3 against source code. 5 critical + 5 important issues found.
**HELI-004 DISCOVERY (Iteration 6a):** API design — CTProtocol pitch field, CTGeometry struct additions, workspace changes, dose formulas, validation.
**HELI-005 DISCOVERY (Iteration 6b):** Physics pipeline audit — confirmed all 16 effects are helical-compatible with zero changes.
**HELI-004+005 CRITIQUE (Iteration 7):** Cross-referenced sections 4-5 against source code. 4 important + 2 minor issues found (C13-C18). No critical issues.
**HELI-007 DISCOVERY (Iteration 8):** Validation strategy — 5-tier testing plan with test code, acceptance criteria, ground truth sources, failure mode diagnosis.
**ALL REFINEMENT + HELI-008 SYNTHESIS (Iteration 9):** Resolved ALL 18 critique items (C1-C18). Produced complete implementation roadmap with 3 stories, exact before/after code diffs for all ~25 change sites, GPU kernel implementation, and risk assessment.

### Key findings from Iteration 9 (Final REFINEMENT + SYNTHESIS)

1. **C1 RESOLVED (CRITICAL):** WFBP normalization is `2π/views_per_rotation`, NOT `π`. Confirmed by FreeCT_wFBP source (`backproject.cuh: output += s[k] * 2*pi / n_proj_turn`). The per-sample weight normalization absorbs the 1/2 FDK redundancy factor.
2. **C14 RESOLVED:** `recon_center` offset in forward projector is CONDITIONAL — only in `geom.fov` path (iterative recon), NOT when `volume_extent` is provided (phantom projection).
3. **C15 RESOLVED:** `create_aquilion_one` adds helical kwargs directly (no delegation refactor). Exact code provided.
4. **Section 8 complete:** 3 implementation stories with dependencies, 7 CTGeometry inner constructor diffs, 5 recon_center offset diffs, complete WFBP kernel code, risk assessment.
5. **Total scope:** ~350 lines production code, ~200 lines tests, 0 new files, ~25 change sites across ~10 files.

## What to do next

**HELI_COMPLETE.** The spec is ready for implementation. A coding agent should:

1. Capture axial regression reference values (pre-implementation)
2. Implement Story 1 (geometry + forward projection) with T1+T2+T3 tests
3. Implement Story 2 (naive helical FDK) with uniform cylinder validation
4. Implement Story 3 (WFBP) with full quality metric validation

### Phase tracking

| Topic | Discovery | Critique | Refinement |
|-------|-----------|----------|------------|
| HELI-000 Current Geometry Audit | **DONE** | **DONE** | **DONE** |
| HELI-001 Helical Geometry | **DONE** | **DONE** | **DONE** |
| HELI-002 Forward Projection | **DONE** | **DONE** | **DONE** |
| HELI-003 Reconstruction Algorithms | **DONE** | **DONE** | **DONE** |
| HELI-004 API Integration | **DONE** | **DONE** | **DONE** |
| HELI-005 Physics Pipeline Compat | **DONE** | **DONE** | **DONE** |
| HELI-006 Reference Implementations | **DONE** | **DONE** | **DONE** |
| HELI-007 Validation Strategy | **DONE** | **DONE** | **DONE** |
| HELI-008 SYNTHESIS | **DONE** | **DONE** | **DONE** |

## Completed iterations

1. **HELI-000 Discovery** (2026-03-17) — Full geometry audit. Found codebase is ~90% helical-ready.
2. **HELI-001 Discovery** (2026-03-17) — Helical geometry: pitch, Z(θ), clinical values, phantom coverage.
3. **HELI-002 Discovery** (2026-03-17) — Forward projection audit. Entire pipeline works unchanged for helical.
4. **HELI-003 Discovery** (2026-03-17) — Reconstruction algorithms: WFBP recommended, Katsevich rejected.
5. **HELI-000–003 Critique** (2026-03-17) — 5 critical + 5 important issues found. All annotated in spec.
6. **HELI-004+005 Discovery** (2026-03-17) — API design + physics pipeline. Complete sections 4 and 5 in spec.
7. **HELI-004+005 Critique** (2026-03-17) — 4 important + 2 minor issues (C13-C18). No critical issues. Sections 4 and 5 confirmed solid.
8. **HELI-007 Discovery** (2026-03-17) — Validation strategy: 5-tier plan, 16 test cases, ground truth sources, failure mode diagnosis. Section 7 added to spec.
9. **ALL REFINEMENT + HELI-008 SYNTHESIS** (2026-03-17) — FINAL ITERATION. All 18 critiques resolved. Section 8 added with complete implementation roadmap. Spec is implementation-ready.
