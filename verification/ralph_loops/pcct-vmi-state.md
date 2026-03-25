# PCCT VMI Discovery — Current State

Current phase: **CRITIQUE**
Current topic: VMI-000/001/002 (combined audit + noise pattern + Siemens algorithm)

## Completed
- VMI-000 DISCOVERY: Audited current implementation. Identified 3 compounding problems: (1) 4→2 bin collapsing, (2) sinogram-domain polynomial decomposition, (3) Mono+ applied to excessively noisy base.
- VMI-001 DISCOVERY: Found CMV (constrained minimum-variance) bin weighting theory. Explains how VMI noise CAN be lower than poly FBP. Key papers: Gilat Schmidt 2009, Leng et al. 2011, Yu et al. 2012.
- VMI-002 DISCOVERY: Found Siemens patent US7856134B2 confirming image-domain approach. Confirmed Mono+ is built into clinical VMI (Grant et al. 2014, Siemens co-authors).

## Next action
**CRITIQUE** of the combined VMI-000/001/002 findings. Key questions:
1. Validate: Can CMV with 4 bins and 2 materials mathematically produce the clinical noise levels?
2. Challenge: Is our assumption that "4 bins used directly" correct, or does Siemens collapse to 2 clinically?
3. Challenge: User previously tried image-domain decomposition and got σ=479. How is CMV different?
4. Challenge: Are the anti-correlations in our DRM realistic enough for CMV to work?
5. Gap: We have no numerical estimate of expected CMV noise — need to compute σ²_VMI(E) from our actual A matrix and Σ.
6. Gap: The Mono+ filter parameters are proprietary. How sensitive is the result to these?

After critique, move to VMI-003 DISCOVERY (reference implementations) if needed, then VMI-004 REFINEMENT.
