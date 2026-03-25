# PCCT VMI Discovery — Current State

Current phase: **REFINEMENT**
Current topic: VMI-004 (synthesis into implementation spec)

## Completed
- VMI-000 DISCOVERY: Audited current implementation. Identified 3 compounding problems: (1) 4→2 bin collapsing, (2) sinogram-domain polynomial decomposition, (3) Mono+ applied to excessively noisy base.
- VMI-001 DISCOVERY: Found CMV (constrained minimum-variance) bin weighting theory. Explains how VMI noise CAN be lower than poly FBP. Key papers: Gilat Schmidt 2009, Leng et al. 2011, Yu et al. 2012.
- VMI-002 DISCOVERY: Found Siemens patent US7856134B2 confirming image-domain approach. Confirmed Mono+ is built into clinical VMI (Grant et al. 2014, Siemens co-authors).
- VMI-003 DISCOVERY: No open-source CMV VMI implementation found. Closest references catalogued.
- VMI-000/001/002 CRITIQUE: 7 critique points identified. **CRITICAL finding: clinical noise targets include QIR-3 (~37% noise reduction).** FBP-equivalent targets are ~60% higher. CMV + Mono+ can plausibly reach FBP-equivalent targets. Anti-correlations absent from our DRM (Σ is diagonal). Mono+ role elevated from "finishing touch" to "essential component." 4-bin DOF benefit is real but modest vs skipping polynomial decomposition entirely.

## Key Critique Findings (summary)
1. **QIR-3 is in the clinical data** — FBP-equivalent targets: 40keV≈90, 70keV≈56, 100keV≈52, 140keV≈50 HU
2. **CMV at 70 keV ≈ poly FBP** (Bhattarai 2024) — does NOT go below poly FBP
3. **Σ is diagonal** in our sim (no anti-correlations) — CMV relies on DOF only
4. **Dominant fix: skip polynomial decomposition**, not just use 4 bins
5. **Mono+ provides 20-50% noise reduction** at extreme keV on top of CMV
6. **Numerical CMV noise prediction required** before implementation

## Next action
**REFINEMENT** (VMI-004): Synthesize all findings into final implementation spec. Key tasks:
1. Rewrite noise budget with corrected FBP-equivalent targets
2. Add numerical validation section (recipe for computing σ²_VMI from A, Σ, t(E))
3. Clarify the 3-layer noise model: CMV base → Mono+ → (clinical: QIR)
4. Write precise implementation pseudocode for notebook
5. Update citation table with new references from critique
6. Mark VMI-003 critique as "done" (no open-source implementations to critique)
