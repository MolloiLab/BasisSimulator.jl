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

---

## Iteration 2 — CRITIQUE of VMI-000/001/002 (2026-03-24)

### Phase: B (CRITIQUE)

### Critique Point 1: THE QIR PROBLEM — Clinical targets likely include iterative reconstruction

**Severity: CRITICAL — may invalidate our noise targets entirely**

The discovery phase accepted the user's assertion that clinical VMI noise (40keV=56.4, 70keV=35.2, 100keV=32.5, 140keV=31.8 HU) represents pure FBP. Multiple lines of evidence challenge this:

1. **DICOM headers say "Q3"** — The user acknowledged this but dismissed it. In Siemens nomenclature, Q3 = QIR strength 3, which is well-documented to reduce noise by ~37% vs QIR-off (Eberhard et al., Eur Radiol Exp 2023, PMC9975359).

2. **NAEOTOM Alpha does NOT offer true FBP for VMI.** "QIR-off" (Q0) is described in the literature as having "minimally possible statistical optimization" — it is NOT identical to WFBP. True FBP (T1 images) is only available in Expert Service mode for non-spectral (total-energy) data. Every clinical VMI noise number includes at least some statistical processing.

3. **Quantitative QIR impact on VMI noise (Eberhard/Mergen 2023, coronary CTA, Bv40 kernel, 9.4 mGy):**

   | keV | QIR-off | QIR-1 | QIR-2 | QIR-3 | QIR-4 |
   |-----|---------|-------|-------|-------|-------|
   | 40  | 35 HU   | 31    | 27    | 22    | 18    |
   | 50  | 29      | 26    | 22    | 19    | 15    |
   | 60  | 24      | 21    | 19    | 16    | 13    |
   | 70  | 21      | 18    | 16    | 13    | 11    |

   QIR-3 at 70 keV: 13 HU (vs QIR-off 21 HU) = 38% reduction.

4. **Reverse-engineering the user's clinical numbers.** If QIR-3 reduces noise by ~37%:
   - 40 keV: 56.4 / 0.63 ≈ **89.5 HU** (FBP-equivalent)
   - 70 keV: 35.2 / 0.63 ≈ **55.9 HU** (FBP-equivalent)
   - 100 keV: 32.5 / 0.63 ≈ **51.6 HU** (FBP-equivalent)
   - 140 keV: 31.8 / 0.63 ≈ **50.5 HU** (FBP-equivalent)

   These FBP-equivalent targets are MUCH more achievable. Our poly FBP is 74.5 HU. CMV at 70 keV should match that (~74 HU), and with CMV noise optimization the 40 keV target of ~90 HU is plausible (ratio 40/70 ≈ 1.6×, consistent with literature).

5. **Bhattarai et al. (Med Phys 2024, PMC10796834) directly measured:** VMI 70 keV noise is nearly IDENTICAL to polychromatic T3D noise (differences <1 HU at same QIR level). This means at 70 keV, VMI noise ≈ poly FBP noise — it does NOT go below it.

**Implication:** If CMV VMI at 70 keV ≈ poly FBP ≈ 74 HU (our simulation), but clinical shows 35.2 HU at 70 keV, the ~50% gap is almost exactly what QIR-3 provides (38% reduction gives 74 × 0.63 = 46.6 HU; remaining gap from dose matching, kernel differences, and Mono+ blending). The "mystery" of VMI noise below poly FBP DISSOLVES once QIR is included.

**Verdict:** The spec MUST be updated to distinguish between:
- (a) FBP-only noise targets (what CMV can achieve)
- (b) Clinical noise targets (which include QIR contribution)

### Critique Point 2: CMV math is sound but noise predictions in spec are UNVALIDATED

The spec claims "expected noise" of 50-60 HU at 40 keV and 25-35 HU at 140 keV (Section 8). These are wishful thinking — no actual computation using our A matrix and Σ was done.

**What we need but don't have:**
- The actual I0 per bin (photon counts per bin from our simulation)
- The actual A matrix entries computed from our DRM × spectrum
- The actual noise covariance Σ (even diagonal-only from Poisson statistics)
- The resulting σ²_VMI(E) = t(E)^T (A^T Σ^{-1} A)^{-1} t(E)

**Without these, we cannot validate whether CMV can reach any specific noise target.** The spec should flag Section 8 as "TBD — requires numerical computation."

### Critique Point 3: 2-bin vs 4-bin difference is OVERSTATED

The spec claims collapsing 4→2 bins is "the single most important insight" and implies 4-bin CMV is dramatically better than 2-bin. The literature suggests the DOF effect alone is modest:

- **Yang et al. (Med Phys 2024):** Optimally weighted 2-channel compression of 4-channel data achieved noise "within 0.27% difference to ideal 4-input decomposition." This means optimal 2-bin ≈ optimal 4-bin when the 2 bins are chosen optimally.
- **However:** Our 4→2 collapsing is NOT optimal compression. It uses fixed sum (bins 1+2, bins 3+4), not CRLB-optimal weighted combination. So the current collapsing does lose information, but the fix isn't necessarily "use 4 bins" — it could be "use better-weighted 2 bins" or even just "skip polynomial decomposition."

**The dominant factor is NOT the number of bins — it's skipping the ill-conditioned matrix inversion entirely.** CMV with K=4 avoids forming material density maps. Even CMV with K=2 (done correctly — direct image combination, not via decomposition) would be dramatically better than our current polynomial decomposition approach.

### Critique Point 4: Anti-correlations in our DRM are ABSENT

The spec (Section 3d) claims "PCCT anti-coincidence logic creates negative off-diagonal entries in Σ" that enable additional noise cancellation. Examination of our actual DRM reveals:

- The MC DRM is cumulative (counts ≥ threshold), not differential
- Bins are extracted by subtraction of consecutive thresholds
- **No negative entries in the DRM or in the resulting bin count distributions**
- Anti-coincidence effects (reassigning multi-counted events) are NOT explicitly modeled in our MC transport

This means one claimed advantage of CMV — exploiting negative noise covariances — may not apply to our simulation. The covariance matrix Σ in our case will be approximately **diagonal** (independent Poisson per bin), not have the negative off-diagonal entries that physical PCCT detectors exhibit from anti-coincidence logic.

**Impact:** Without anti-correlations, CMV still benefits from the overdetermined system (2 extra DOF), but the noise reduction may be smaller than estimated. The spec must distinguish between:
- CMV with diagonal Σ (our case): noise reduction from DOF only
- CMV with full Σ including anti-correlations (physical detector): additional noise cancellation

### Critique Point 5: Mono+ IS part of the algorithm — spec correctly identifies this but UNDERESTIMATES its role

The discovery found that Mono+ is built into clinical VMI (Grant et al. 2014, Siemens co-authors). The spec treats it as a "finishing touch" (Section 4c). But the literature data suggests Mono+ does heavy lifting:

- Head CT (Boehm 2023, QIR-off): 40 keV noise = 12.8 HU, 190 keV = 1.6 HU. Ratio = 8×.
- But pure CMV theory predicts the ratio should be ~2-3× (from target vector scaling).
- The additional reduction at high keV beyond what CMV predicts suggests Mono+ is contributing significantly — replacing high-frequency noise content from the noisy low-keV VMI with the low-noise optimal-keV content.

**At extreme keV (40, 140+), Mono+ may contribute 30-50% of the noise reduction on top of CMV.** The spec should elevate Mono+ from "finishing touch" to "essential component."

### Critique Point 6: The A matrix formulation needs precision

Section 6a defines A_km but is vague about what "mean attenuation of material m as seen by bin k" means in practice. Two possible interpretations:

1. **Projection-domain:** A_km = effective linear attenuation coefficient of material m integrated over bin k's spectral response. Units: cm^{-1}. This works for line-integral-domain VMI.
2. **Image-domain:** A_km = mean CT number (HU) of material m in bin k's reconstructed image. Units: HU. This works for image-domain CMV.

For image-domain CMV (the proposed algorithm), interpretation (2) is correct. The A matrix should be constructed from:
```
A_km = ∫ S(E) × R_k(E) × μ_m(E) dE / ∫ S(E) × R_k(E) dE
```
where S(E) is the source spectrum, R_k(E) is the DRM response for bin k, and μ_m(E) is the linear attenuation coefficient of material m. This is the mean attenuation of material m as seen by bin k after FBP reconstruction.

### Critique Point 7: Covariance estimation from reconstructed images is WRONG for FBP

Section 5b suggests estimating Σ empirically from reconstructed bin images (uniform ROI covariance). This is problematic:

- FBP noise is spatially correlated (ramp filter creates directional correlations)
- The noise texture differs from pixel to pixel (depends on ray paths, object attenuation)
- A single ROI gives a biased estimate of the covariance

**Better approach:** Estimate Σ analytically from photon counts. For bin k, σ²_k ∝ 1/N_k where N_k is the mean detected count. For independent Poisson bins (our case — no anti-correlations), Σ = diag(σ²_1, ..., σ²_4). This is simpler and more accurate than empirical estimation.

### Summary of Critique Gaps

| # | Gap | Severity | Resolution |
|---|-----|----------|------------|
| 1 | Clinical targets include QIR-3 | **CRITICAL** | Recompute FBP-equivalent targets; validate against QIR-off literature data |
| 2 | No numerical CMV noise prediction | HIGH | Compute σ²_VMI(E) from actual A and Σ before implementing |
| 3 | 4-bin DOF benefit overstated | MEDIUM | The dominant fix is skipping decomposition, not number of bins |
| 4 | No anti-correlations in DRM | MEDIUM | Σ is diagonal in our sim; reduces CMV noise cancellation |
| 5 | Mono+ role underestimated | MEDIUM | Elevate to essential component; adjust noise budget |
| 6 | A matrix interpretation ambiguous | LOW | Clarify image-domain definition |
| 7 | Empirical Σ estimation flawed for FBP | LOW | Use analytical Poisson estimate instead |

### Revised Understanding

The "impossible" observation (σ_VMI(140)=31.8 < σ_poly=66.4) is now explained by the COMBINATION of:
1. **QIR-3:** ~37% noise reduction (66.4 → 41.8 HU for poly; VMI similarly reduced)
2. **CMV image-domain weighting:** Optimal combination at 140 keV ≈ poly FBP noise (or slightly below due to 2 extra DOF)
3. **Mono+:** Additional 20-30% reduction at extreme keV by replacing HF noise content

None of these alone explains the clinical numbers. All three together do.
