# PCCT VMI Discovery — Current State

Current phase: **DISC_COMPLETE**
Final iteration: VMI-004 REFINEMENT

## Completed
- VMI-000 DISCOVERY: Audited current implementation. Identified 3 compounding problems: (1) 4→2 bin collapsing, (2) sinogram-domain polynomial decomposition, (3) Mono+ applied to excessively noisy base.
- VMI-001 DISCOVERY: Found CMV (constrained minimum-variance) bin weighting theory. Explains how VMI noise CAN be lower than poly FBP. Key papers: Gilat Schmidt 2009, Leng et al. 2011, Yu et al. 2012.
- VMI-002 DISCOVERY: Found Siemens patent US7856134B2 confirming image-domain approach. Confirmed Mono+ is built into clinical VMI (Grant et al. 2014, Siemens co-authors).
- VMI-003 DISCOVERY: No open-source CMV VMI implementation found. Closest references catalogued.
- VMI-000/001/002 CRITIQUE: 7 critique points identified. **CRITICAL finding: clinical noise targets include QIR-3 (~37% noise reduction).** FBP-equivalent targets are ~60% higher. CMV + Mono+ can plausibly reach FBP-equivalent targets. Anti-correlations absent from our DRM (Σ is diagonal). Mono+ role elevated from "finishing touch" to "essential component."
- VMI-004 REFINEMENT: Final spec written. Includes 3-layer noise model, corrected FBP-equivalent targets, numerical validation recipe, precise Julia pseudocode, and 23 cited references.

## Summary of Algorithm

Replace sinogram-domain polynomial decomposition with:
1. FBP-reconstruct all 4 bins independently
2. Compute A matrix (4×2) from effective bin spectra (already available)
3. Compute diagonal Σ from I0 per bin (already available)
4. CMV weights: w*(E) = Σ⁻¹A(A'Σ⁻¹A)⁻¹t(E)
5. VMI(E) = Σ w_k(E) · bin_image_k
6. HU conversion
7. Mono+ frequency-split blending

## Next Steps (IMPLEMENTATION — outside discovery scope)
1. **Numerical validation** (spec Section 5): Compute σ²_VMI with actual A and Σ. Confirm noise predictions match FBP-equivalent targets before writing code.
2. **Implement CMV cell** in nb07 (spec Section 6 has drop-in pseudocode).
3. **Measure noise** from reconstructed CMV VMI images. Compare against budget.
4. **Tune Mono+** parameters if needed (increase σ_lp if noise at extreme keV too high).
