# PCCT VMI Discovery Progress

## Iteration 0 — Pre-loop context (from human)

### What we know:
- Material decomposition HU values are CORRECT (scatter plot matches clinical perfectly)
- Noiseless VMI (pcct_noise_reduction=1.0) produces clean images at all energies
- Poly FBP noise is reasonable: simulated σ=74.5 vs clinical σ=66.4 HU
- VMI noise is catastrophically wrong: simulated 40keV σ=1063 vs clinical σ=56.4
- Clinical VMI noise pattern: 40keV=56.4, 70keV=35.2, 100keV=32.5, 140keV=31.8 (DECREASING)
- Our VMI noise pattern: worst at 40 and 140 keV, best near 70 keV (INCREASING away from center)
- Clinical VMIs are FBP (NOT QIR) per user confirmation
- Siemens collapses 4 bins → 2 spectral channels (confirmed in literature)

### What we tried (all failed to fix noise):
1. Sinogram-domain polynomial decomposition (Alvarez & Macovski) — correct HU, 19× noise
2. Image-domain 2×2 linear decomposition — correct HU, still ~8× noise
3. Softer VMI reconstruction kernel — marginal
4. Median filtering on material sinograms — marginal
5. pcct_noise_reduction 0.3-0.7 — reduces total noise but pattern stays wrong
6. Mono+ blending — user says clinical doesn't use it (but literature suggests otherwise)

### Key unresolved question:
What algorithm produces VMI with DECREASING noise from 40→140 keV from FBP-only reconstruction?

---

## Iteration 1 — VMI-000 + VMI-001 + VMI-002 DISCOVERY (2026-03-24)

### Phase: A (DISCOVERY) — Combined audit + literature research

### VMI-000: Current Implementation Audit

**Audit of nb07 (lines 1515-1729):**

The current PCCT VMI pipeline has three stages:

**Stage 1 — Bin collapsing (lines 1517-1539):**
- 4 threshold bins collapsed to 2 sinograms: Low (bins 1+2, 20-55 keV), High (bins 3+4, >55 keV)
- Count-domain combination: `sino = -log((I0_a·exp(-sino_a) + I0_b·exp(-sino_b)) / (I0_a + I0_b))`
- This discards 2 degrees of freedom that could be used for noise optimization

**Stage 2 — Sinogram-domain polynomial decomposition (lines 1566-1686):**
- Effective bin spectra computed from source spectrum × DRM × quantum efficiency
- Synthetic step-wedge calibration: 40×25 grid (water × iodine), Chebyshev spacing
- 4th-order inverse polynomial: `(p_low, p_high) → (t_water, t_iodine)`, 15 terms
- Per-ray application to noisy sinograms → material sinograms `sino_w`, `sino_I`
- This is the catastrophic noise amplifier — polynomial terms like `p^4` map small noise to large errors

**Stage 3 — VMI synthesis + FBP + Mono+ (lines 1688-1727):**
- VMI sinogram: `vmi_sino(E) = μρ_w(E)·sino_w + μρ_I(E)·sino_I`
- FBP reconstruction per energy (ramp filter amplifies already-noisy material sinograms)
- Mono+ blending: `Mono+(E) = LP(VMI(E)) + HP(VMI(70keV))`
- σ_lp = 1.5 + 0.02·|E-70| mm

**Comparison with nb06 (GE dual-energy VMI):**
- Identical algorithm (sinogram poly decomp → VMI sinogram → FBP → Mono+)
- Uses a softer DE-specific filter kernel
- Same Mono+ parameters

**Key finding:** Both notebooks use the SAME algorithm. The GE DE works better because dual-kVp (80/140) gives much better spectral separation than single-kVp bin splitting, so the polynomial decomposition condition number is lower.

### VMI-001: Literature Research — Why Clinical Noise Decreases

**Finding 1: Constrained Minimum-Variance (CMV) bin weighting**

Multiple independent sources describe an approach that skips sinogram-domain decomposition entirely:

- **Gilat Schmidt TG, Med Phys 2009;36(7):3018-27**: "Optimal image-based weighting for energy-resolved CT." Optimal weight for each bin ∝ contrast/variance. CNR improvement 1.15-1.6× over energy-integrating.
- **Leng S et al., Med Phys 2011;38(9):4946-57**: "Noise reduction in spectral CT: reducing dose and breaking the trade-off between image noise and energy bin selection."
- **Yu L, Leng S, McCollough CH, AJR 2012;199:S9-S15**: Image-domain VMI framework.
- **Yang et al., Med Phys 2025;doi:10.1002/mp.17489**: Compares pre-log, post-log, and MD-based optimal weights — all achieve ~equal max CNR.

The CMV algorithm:
```
minimize    w^T Σ w           (noise variance)
subject to  A^T w = t(E)      (monoenergetic fidelity)
solution:   w* = Σ^{-1} A (A^T Σ^{-1} A)^{-1} t(E)
variance:   σ²_VMI = t(E)^T (A^T Σ^{-1} A)^{-1} t(E)
```

**Finding 2: VMI noise CAN be lower than poly FBP**

This is the key insight resolving the "impossible" σ(140keV)=31.8 < σ(poly)=66.4:
- Poly FBP uses a fixed, non-optimal energy weighting (detector response × spectrum)
- CMV finds the optimal weighting for each target energy
- With K>M bins (4 bins, 2 materials), extra DOF enable noise reduction below poly FBP
- Anti-correlated noise between PCCT bins (from anti-coincidence logic) provides additional noise cancellation when optimally combined

**Finding 3: Noise behavior with keV explained**

At high keV (140): target attenuation values are small and similar across materials → constraint easy to satisfy → weights nearly proportional to simple photon-count sum → noise ≈ or < poly FBP.

At low keV (40): target values are large and divergent (iodine photoelectric effect) → constraint forces extreme weights → higher noise but still manageable with CMV.

At optimal keV (~70): minimum noise, can be significantly below poly FBP.

### VMI-002: Siemens-Specific Implementation

**Finding 4: Siemens patent US7856134B2 (Stierstorfer/Ruhrnschopf)**
- Confirms image-domain material decomposition (NOT sinogram domain)
- Uses pre-computed lookup tables for weights (material-class dependent)
- Weights depend on virtual energy and material class
- VMI = λ₁(E)·I₁ + λ₂(E)·I₂ (direct image combination)

**Finding 5: Mono+ is built into clinical VMI (Grant et al. 2014)**
- The user rejected Mono+ as "clinical doesn't use it" — but literature strongly indicates Mono+ (or equivalent) IS the standard Siemens VMI algorithm
- Grant et al. 2014 has Siemens co-authors (Flohr, Krauss, Sedlmair, Schmidt)
- Siemens syngo.via "MonoE" applications use Mono+/VMI+ by default
- The exact frequency filter parameters are proprietary

**Finding 6: NAEOTOM Alpha uses 2 thresholds clinically (not 4)**
- Standard clinical mode: 2 energy thresholds → 2 bins
- 4-threshold mode is research only
- BUT our simulation uses 4 thresholds — we should use all 4 for CMV (advantage over clinical)

**Finding 7: No fundamentally different PCD-CT algorithm vs DECT**
- Same Mono+ approach, better results due to superior spectral data quality from PCD
- PCD advantage: photon counting (not energy weighting) gives equal weight to all energies
- This improves low-keV SNR compared to energy-integrating detectors

### Summary of Root Causes (ranked by impact)

1. **Sinogram-domain polynomial decomposition** — THE primary noise amplifier. Should be replaced with image-domain CMV weighting.
2. **Collapsing 4 bins → 2** — Discards noise optimization degrees of freedom. Should use all 4 bins with CMV.
3. **No noise covariance modeling** — Current approach ignores inter-bin noise correlations. CMV exploits these.
4. **Mono+ applied to excessively noisy images** — Mono+ is a finishing touch, not a rescue tool. Base noise must be reasonable first.

### Proposed Fix

Replace the entire VMI pipeline with:
1. FBP-reconstruct all 4 bins independently → 4 bin images
2. Compute A matrix (material-bin attenuation) from effective bin spectra
3. Estimate Σ (noise covariance) analytically from I0 per bin or empirically from water ROI
4. CMV weights: `w*(E) = Σ^{-1} A (A^T Σ^{-1} A)^{-1} t(E)` per target energy
5. VMI(E) = Σ w_k(E) · bin_image_k
6. HU conversion per energy
7. Mono+ frequency-split blending for extreme keV

### Open Questions for Critique Phase
1. Can we validate the CMV noise prediction formula against clinical data before implementing?
2. Does our DRM produce realistic inter-bin anti-correlations?
3. Should weights be spatially varying (local Σ estimation)?
4. Is the 2-material (water/iodine) basis sufficient, or do we need calcium?
5. What happens at object boundaries where different materials meet?
