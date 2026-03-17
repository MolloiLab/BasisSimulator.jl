# Helical CT Discovery Loop — Current State

**Last updated:** 2026-03-17
**Current phase:** HELI-007 DISCOVERY complete → next: REFINEMENT pass (all sections) or HELI-008 SYNTHESIS
**Current topic:** All discovery and critique phases complete. Ready for comprehensive REFINEMENT.

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

### Key findings from Iteration 8 (HELI-007 Discovery)

1. **Existing test infrastructure is sufficient** — no new test utilities needed. Gammex 472, small_test_setup(), HU/CNR/NPS/MTF metrics, XCIST integration all reusable.
2. **5-tier validation:** Unit → Axial Regression → Helical Integration → Quality Metrics → Cross-Reference.
3. **Most diagnostic test:** Z-uniformity (mean HU per slice should be flat within Tam-Danielsson window). Catches normalization, scaling, and weight bugs.
4. **Strongest internal reference:** Converged SIRT with matched backprojection — provably correct for any geometry, serves as ground truth for WFBP validation.
5. **Pre-implementation step required:** Capture axial regression reference values before any code changes.
6. **16 specific test cases** with Julia code ready to implement, organized by coding phase.

## What to do next

**RECOMMENDED: Comprehensive REFINEMENT pass** — Resolve all open critique items (C1-C18) with implementation-ready detail. Most urgent:
- **C1 (CRITICAL):** WFBP normalization formula — must read FreeCT_wFBP source to determine π vs 2π constant. This is the single biggest unresolved question in the spec.
- **C15 (IMPORTANT):** `create_aquilion_one` refactoring to eliminate duplicated geometry loop.
- **C14 (IMPORTANT):** `recon_center` conditional in forward projector — needs exact code diff.

**ALTERNATIVE: HELI-008 SYNTHESIS** — Create phased implementation roadmap with stories and dependencies. Could proceed now since all discovery topics are complete, but REFINEMENT would make the implementation more precise.

**Suggested path:** One REFINEMENT pass resolving C1 + other open issues → HELI-008 SYNTHESIS → HELI_COMPLETE.

### Phase rotation plan

```
HELI-000 (priority 0): Discovery ✓ → Critique ✓ → Refinement
HELI-001 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-002 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-003 (priority 1): Discovery ✓ → Critique ✓ → Refinement (NEEDS C1 resolution)
HELI-004 (priority 1): Discovery ✓ → Critique ✓ → Refinement
HELI-005 (priority 1): Discovery ✓ → Critique ✓ → Refinement (minimal — physics confirmed unchanged)
HELI-006 (priority 1): Discovery ✓ (partial, via HELI-003) → Critique → Refinement
HELI-007 (priority 2): Discovery ✓ → Critique → Refinement
HELI-008 (priority 3, SYNTHESIS): All discovery complete → ready when refinement done
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
| HELI-007 Validation Strategy | **DONE** | open | open |
| HELI-008 SYNTHESIS | open | open | open |

## Completed iterations

1. **HELI-000 Discovery** (2026-03-17) — Full geometry audit. Found codebase is ~90% helical-ready.
2. **HELI-001 Discovery** (2026-03-17) — Helical geometry: pitch, Z(θ), clinical values, phantom coverage.
3. **HELI-002 Discovery** (2026-03-17) — Forward projection audit. Entire pipeline works unchanged for helical.
4. **HELI-003 Discovery** (2026-03-17) — Reconstruction algorithms: WFBP recommended, Katsevich rejected.
5. **HELI-000–003 Critique** (2026-03-17) — 5 critical + 5 important issues found. All annotated in spec.
6. **HELI-004+005 Discovery** (2026-03-17) — API design + physics pipeline. Complete sections 4 and 5 in spec.
7. **HELI-004+005 Critique** (2026-03-17) — 4 important + 2 minor issues (C13-C18). No critical issues. Sections 4 and 5 confirmed solid.
8. **HELI-007 Discovery** (2026-03-17) — Validation strategy: 5-tier plan, 16 test cases, ground truth sources, failure mode diagnosis. Section 7 added to spec.
