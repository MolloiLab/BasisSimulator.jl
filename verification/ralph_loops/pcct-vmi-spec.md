# PCCT VMI Algorithm Spec

## 1. Problem Statement

Our PCCT VMI has correct HU accuracy but catastrophically wrong noise:
- Clinical noise: 40keV=56.4, 70keV=35.2, 100keV=32.5, 140keV=31.8 HU (DECREASING)
- Our noise: 40keV=1063, 70keV=124, 100keV=201, 140keV=297 HU (V-shaped)
- Poly FBP: clinical 66.4, simulated 74.5 HU (close match)

**Central mystery:** VMI at 140 keV (31.8 HU) has LESS noise than poly FBP (66.4 HU). This is impossible with standard 2-material decomposition + recombination, which always amplifies noise.

## 2. Root Cause Analysis

### 2a. What Our Implementation Does Wrong

Current pipeline (nb07, lines 1515-1729):
1. Collapse 4 threshold bins → 2 sinograms (low: bins 1+2, high: bins 3+4)
2. Sinogram-domain 4th-order polynomial decomposition: `(p_low, p_high) → (t_water, t_iodine)`
3. VMI sinogram synthesis: `vmi_sino(E) = μρ_w(E)·sino_w + μρ_I(E)·sino_I`
4. FBP of VMI sinogram
5. Mono+ frequency-split blending

**Three compounding problems:**
1. **Collapsing 4→2 bins discards noise-optimization degrees of freedom.** With 4 bins and 2 material constraints, CMV has 2 extra DOF for noise minimization. By collapsing to 2 bins, these DOF are lost.
2. **Sinogram-domain polynomial decomposition amplifies noise catastrophically.** The 4th-order polynomial maps small noise perturbations in `(p_low, p_high)` to large errors in `(t_water, t_iodine)`. The polynomial includes terms like `p^4` that amplify noise enormously.
3. **FBP ramp filter further amplifies the already-noisy material sinograms.** The ramp filter boosts high-frequency content. Applied to noise-amplified material sinograms, it makes things much worse.

The result: Raw VMI noise is σ=1063 HU at 40 keV (before Mono+). Even Mono+ replacing HF content with 70 keV (σ=124) can't fix this — the 70 keV VMI itself is 3.5x too noisy.

### 2b. What the Clinical Algorithm Does Differently

Based on literature synthesis (see Section 3), clinical PCCT VMI uses **image-domain processing** that avoids sinogram-domain polynomial inversion entirely:

## 3. The Correct Algorithm: Constrained Minimum-Variance (CMV) Bin Weighting

### 3a. Overview

**Source:** Leng et al., Med Phys 2011; Yu et al., AJR 2012; Gilat Schmidt, Med Phys 2009. Siemens patent US7856134B2 (Stierstorfer/Ruhrnschopf) confirms image-domain approach.

The algorithm:
1. Reconstruct **each energy bin** independently via FBP → K bin images
2. Compute VMI at energy E as a weighted sum of bin images: `VMI(E) = Σ_k w_k(E) · I_k`
3. Weights `w_k(E)` are chosen to **minimize variance** subject to **monoenergetic fidelity constraints**
4. Apply Mono+ frequency-split blending (Grant et al. 2014) for further noise optimization at extreme keV

### 3b. Mathematical Formulation

**Given:**
- K energy bins, each reconstructed via FBP: `I_1(x,y), ..., I_K(x,y)`
- M basis materials (M=2: water + iodine)
- Noise covariance matrix of bin images: `Σ ∈ R^(K×K)` (estimated from data or analytically from photon counts)
- Material-bin attenuation matrix: `A ∈ R^(K×M)`, where `A_km` = mean attenuation of material m in bin k
- Target monoenergetic attenuation: `t(E) ∈ R^M`, where `t_m(E)` = mass attenuation of material m at energy E

**Optimization:**
```
minimize    w^T Σ w           (noise variance)
subject to  A^T w = t(E)      (monoenergetic fidelity for each basis material)
```

**Closed-form solution (Lagrange multipliers):**
```
w*(E) = Σ^{-1} A (A^T Σ^{-1} A)^{-1} t(E)
```

**Minimum achievable variance:**
```
σ²_VMI(E) = t(E)^T (A^T Σ^{-1} A)^{-1} t(E)
```

### 3c. Why Noise Decreases with Increasing keV

The target vector `t(E)` contains mass attenuation coefficients at energy E:
- At **low E** (40 keV): `t(40) = [μ_w(40), μ_I(40)]` — large, divergent values (photoelectric dominates for iodine). The constraint forces extreme weights → high variance.
- At **high E** (140 keV): `t(140) = [μ_w(140), μ_I(140)]` — small, similar values (Compton dominates). The constraint is easy to satisfy → low variance.
- At **optimal E** (~65-75 keV): minimum noise, can be lower than poly FBP.

### 3d. Why VMI Can Beat Poly FBP Noise

Poly FBP is a *specific* (non-optimal) energy weighting of all photons. The CMV approach finds the *optimal* weighting for a given target energy. With K>M bins:
1. **Extra degrees of freedom**: K-M=2 free DOF for noise minimization (with 4 bins, 2 materials)
2. **Anti-correlated noise**: PCCT anti-coincidence logic creates negative off-diagonal entries in Σ. The optimal weights exploit these anti-correlations to cancel noise.
3. **Energy-optimal combination**: At high keV where all materials look similar, the CMV weights approach the noise-optimal photon combination, which is better than energy-integrating (poly FBP) weighting.

This directly explains: σ_VMI(140) = 31.8 < σ_poly = 66.4 HU.

### 3e. Use All 4 Bins for Maximum Benefit, But 2-Bin CMV Also Helps

**REVISED (Critique Point 3):** The 4→2 bin collapsing matters, but it is NOT the dominant problem. Yang et al. (Med Phys 2024) showed optimally weighted 2-channel compression is within 0.27% of 4-channel noise. The REAL win is:

1. **Skipping the polynomial decomposition** (dominant factor, ~5-10× noise reduction)
2. **Never forming material density maps** (avoids ill-conditioned A^{-1}, ~2-4× noise reduction)
3. **Using all 4 bins for extra DOF** (modest additional benefit, ~1.3×)

Even 2-bin CMV (reconstructing 2 bins via FBP, then combining with optimal weights directly into VMI) would be dramatically better than our current polynomial decomposition approach. 4-bin CMV is better still.

With 4 bins: K=4, M=2 → 2 DOF for noise optimization → CMV can push noise below poly FBP at optimal keV.
With 2 bins: K=2, M=2 → 0 extra DOF → CMV reduces to unique solution but STILL avoids the noise amplification of polynomial inversion by operating in image domain with direct combination.

## 4. Mono+ Frequency-Split Blending (Grant et al. 2014)

### 4a. Algorithm

Applied after CMV weighting to further improve noise at extreme keV:

1. Compute CMV VMI at target energy E (high contrast, moderate noise at extreme keV)
2. Compute CMV VMI at optimal energy E_opt ≈ 70 keV (minimum noise)
3. Frequency decomposition: `Mono+(E) = LP(VMI(E)) + HP(VMI(E_opt))`
   - LP = Gaussian low-pass filter in Fourier domain
   - HP = I - LP (high-pass complement)
4. Low-frequency contrast from target E, high-frequency noise texture from optimal E

### 4b. Key Parameters (from GE notebook, validated)
- `E_opt` = 70 keV
- `σ_lp_mm` = 1.5 + 0.02 × |E - E_opt| mm (increases with distance from optimal)
- Applied per-slice in image domain

### 4c. Mono+ Role Is LARGER Than Initially Thought (Critique Point 5)

Our current implementation already uses Mono+. The problem is the *input* to Mono+ is too noisy:
- Our raw 70 keV VMI: σ=124 HU (FBP-equivalent target: ~56 HU)
- Mono+ replaces HF noise from 70 keV, but 70 keV itself is 2× too noisy
- Fix the raw VMI noise via CMV first, then Mono+ provides substantial additional reduction

**REVISED estimate of Mono+ contribution:** At extreme keV (40, 140+), Mono+ provides 20-50% noise reduction on top of CMV. Evidence: Head CT data (Boehm 2023) shows noise ratio 40/190 keV = 8× with QIR-off, while pure CMV theory predicts ~2-3× from target vector scaling alone. The additional reduction at high keV comes from Mono+ replacing noisy HF content with optimal-keV content.

**Mono+ is an essential component, not a finishing touch.** The noise budget is approximately:
- Polynomial decomp → CMV: 10× improvement (σ from ~1063 to ~100 at 40 keV)
- Mono+ blending: additional 20-50% reduction at extreme keV
- QIR-3: additional ~37% reduction (clinical only, not in our sim)

## 5. Covariance Matrix Estimation

### 5a. Analytical — Diagonal Poisson (RECOMMENDED for our simulation)

**REVISED (Critique Points 4 & 7):** Our MC DRM has no anti-correlations (cumulative → subtracted bins produce independent counts). The covariance matrix is therefore **diagonal**:

```
Σ = diag(σ²_1, σ²_2, σ²_3, σ²_4)
```

For FBP-reconstructed bin images, the noise variance of bin k is approximately:
```
σ²_k ≈ C / N_k
```
where N_k = mean photon count in bin k, C = constant depending on ramp filter, geometry, etc. Since C is the same for all bins (same geometry, same ramp filter), the relative variances scale as 1/N_k.

**Practical approach:** Measure σ²_k from reconstructed bin images in a uniform water ROI, or compute analytically from I0 per bin. Both should give consistent results for a diagonal Σ.

### 5b. Empirical (from reconstructed bin images)

Measuring from reconstructed images is acceptable for a **diagonal** Σ (just per-bin variance). However, note:
- FBP noise is spatially correlated (ramp filter creates streak-like correlations)
- Use a sufficiently large uniform ROI to average over these correlations
- Do NOT attempt to estimate off-diagonal terms from image ROIs (the spatial correlations contaminate the inter-bin covariance estimate)

### 5c. Future: Physical anti-correlations

If the DRM is enhanced to model anti-coincidence explicitly, Σ will have negative off-diagonal entries. This would enable additional CMV noise cancellation. For now, diagonal Σ is correct for our simulation.

## 6. Material-Bin Attenuation Matrix A

The matrix A has dimensions K×M. Each entry A_km represents the mean attenuation of material m as seen by bin k.

### 6a. Computation

For each bin k with effective spectral weights w_k(E):
```
A_km = Σ_E  w_k(E) · (μ/ρ)_m(E) · ρ_m
```
where the sum is over the bin's effective energy spectrum (source spectrum × DRM response × quantum efficiency for bin k).

**We already compute these effective spectra** in the polynomial calibration code (lines 1589-1596). The difference is we use them for polynomial fitting instead of for direct weight computation.

### 6b. Target Vector t(E)

For each target VMI energy E:
```
t_m(E) = (μ/ρ)_m(E) · ρ_m
```
For water (ρ=1.0 g/cm³) and iodine (mass per unit volume basis), these are just the linear attenuation coefficients at E.

## 7. Implementation Plan

### Step 1: Reconstruct all 4 bins via FBP
- Use the same filter kernel as poly FBP
- Result: 4 images, each with independent noise from their bin's photon statistics

### Step 2: Compute A matrix and Σ
- A: from effective bin spectra (already computed in calibration cell)
- Σ: either analytical (from I0 per bin) or empirical (from uniform ROI in bin images)

### Step 3: CMV weights per energy
```julia
w(E) = Σ_inv * A * inv(A' * Σ_inv * A) * t(E)
```

### Step 4: Weighted sum → raw VMI
```julia
VMI(E) = sum(w_k(E) * bin_image_k for k in 1:4)
```

### Step 5: HU conversion
```julia
VMI_HU(E) = (VMI(E) - μ_water(E)) / μ_water(E) * 1000
```

### Step 6: Mono+ blending (for keV far from optimal)
```julia
Mono+(E) = LP(VMI(E)) + HP(VMI(E_opt))
```

## 8. Expected Noise Performance

### 8a. REVISED: Clinical targets include QIR-3 (Critique Point 1)

**The user's clinical noise values (40keV=56.4, 70keV=35.2, 100keV=32.5, 140keV=31.8 HU) almost certainly include QIR-3 iterative reconstruction**, despite the user's belief that they are FBP. Evidence:

- DICOM headers say "Q3" = QIR strength 3
- NAEOTOM Alpha does NOT offer true FBP for VMI; even "QIR-off" has minimal statistical optimization
- QIR-3 reduces noise ~37% vs QIR-off (Eberhard/Mergen 2023, PMC9975359)
- Bhattarai 2024 (PMC10796834): VMI 70 keV noise ≈ T3D (polychromatic) noise at same QIR level (<1 HU difference)

### 8b. FBP-Only Noise Targets (CMV + Mono+, no QIR)

Reverse-engineering clinical numbers assuming QIR-3 (÷0.63):

| keV | Clinical (QIR-3) | FBP-equivalent (÷0.63) | CMV-only estimate | CMV + Mono+ target |
|-----|-------------------|------------------------|-------------------|--------------------|
| 40  | 56.4 HU           | ~89.5 HU               | ~120-150 HU       | ~80-100 HU         |
| 70  | 35.2              | ~55.9                   | ~70-80            | ~55-65             |
| 100 | 32.5              | ~51.6                   | ~60-70            | ~45-55             |
| 140 | 31.8              | ~50.5                   | ~55-65            | ~40-55             |

**Key insight:** CMV at 70 keV ≈ poly FBP ≈ 74.5 HU (our sim). With Mono+, ~55-65 HU. Remaining gap to clinical 35.2 HU is explained by QIR-3.

### 8c. Literature Benchmarks (QIR-off, closest to FBP)

| Source | Anatomy | Kernel | Dose | 40 keV | 70 keV | Ratio |
|--------|---------|--------|------|--------|--------|-------|
| Eberhard 2023 | Coronary phantom | Bv40 | 9.4 mGy | 35 HU | 21 HU | 1.67× |
| Boehm 2023 | Head cortical | Qr36 | 48.8 mGy | 12.8 HU | 5.0 HU | 2.56× |
| Greffier 2023 | Catphan 20cm | Qr40 | 10 mGy | — | 10.7 HU | — |

### 8d. UNVALIDATED — Needs numerical computation (Critique Point 2)

The CMV-only and CMV+Mono+ estimates above are back-of-envelope. To validate:
- Compute actual A matrix from DRM × spectrum
- Compute Σ from Poisson statistics (diagonal, I0 per bin)
- Evaluate σ²_VMI(E) = t(E)^T (A^T Σ^{-1} A)^{-1} t(E) at each energy
- This numerical prediction is **required** before implementation to confirm feasibility

## 9. Key Citations

| Reference | Relevance |
|-----------|-----------|
| Gilat Schmidt TG, Med Phys 2009;36(7):3018-27 | Optimal image-based weighting theory. Weight ∝ contrast/variance. CNR improvement 1.15-1.6× over energy-integrating. |
| Yu L, Leng S, McCollough CH, AJR 2012;199:S9-S15 | Image-domain VMI decomposition framework |
| Leng S et al., Med Phys 2011;38(9):4946-57 | Noise reduction via optimal energy weighting in spectral CT |
| Grant KL et al., Invest Radiol 2014;49(9):586-92 | Mono+ frequency-split algorithm (Siemens co-authors) |
| Alvarez RE, Macovski A, Phys Med Biol 1976 | Classical sinogram-domain decomposition (what NOT to do for noise) |
| Roessl E, Herrmann C, Phys Med Biol 2009;54(5):1307-18 | CRLB for VMI noise — theoretical minimum |
| US7856134B2 (Stierstorfer/Ruhrnschopf, Siemens) | Siemens patent: image-domain VMI, pre-computed weight LUTs |
| Rajendran K et al., Radiology 2022;303(1):130-8 | First clinical NAEOTOM Alpha evaluation — noise/NPS characterization |
| Yang et al., Med Phys 2025, doi:10.1002/mp.17489 | Pre-log/post-log/MD optimal weighting comparison |
| **NEW (Critique)** | |
| Bhattarai et al., Med Phys 2024;51(2), PMC10796834 | **VMI 70keV noise ≈ T3D (poly) noise** at same QIR level (<1 HU diff). Key validation. |
| Eberhard/Mergen et al., Eur Radiol Exp 2023, PMC9975359 | **QIR-off vs QIR-1/2/3/4 VMI noise table** for coronary CTA. QIR-3 = 38% reduction at 70keV. |
| Boehm/Michael et al., Clin Neuroradiol 2023, PMC10881631 | **Head CT VMI noise QIR-off**: 40keV=12.8, 66keV=5.0, 100keV=2.4, 190keV=1.6 HU |
| Greffier et al., Diagnostics 2023, PMC10092985 | **Catphan 70keV WFBP**: 10.7 HU. QIR-4 reduces to 4.4 HU (59% reduction). |
| Sartoretti et al., Br J Radiol 2023, PMC9975359 | **QIR-off VMI noise**: 40keV=35, 70keV=21 HU (ratio 1.67×) |
| Yang et al., Med Phys 2024;51(1):224-238 | **Optimal 2-channel ≈ 4-channel** (within 0.27%). Naive 2-channel = 1.3× worse. |
| Niu et al., Phys Med Biol 2018, PMC5903446 | **Condition numbers**: MECT(N=4) no-prior = 301, DECT with-prior = 17.86 |
| Heismann et al., Med Phys 2025, doi:10.1002/mp.17591 | PCCT 10% noise advantage from quantum counting; up to 1.7× RMS improvement |

## 10. Open Questions (Post-Critique)

### Resolved
1. ~~**QIR contribution**~~: **RESOLVED — QIR-3 IS applied** (Critique Point 1). Clinical targets are QIR-inclusive. FBP-only targets are ~60% higher. CMV + Mono+ can plausibly reach FBP-equivalent targets.

2. ~~**Role of anti-coincidence anti-correlations**~~: **RESOLVED — Our DRM has NO anti-correlations** (Critique Point 4). MC DRM is cumulative, bins extracted by subtraction. Σ is diagonal (independent Poisson). CMV still benefits from 2 extra DOF but cannot exploit anti-correlations.

### Remaining
3. **Exact Mono+ filter parameters**: Proprietary. Our `σ_lp = 1.5 + 0.02·|E-E_opt|` is a starting point. Mono+ role is LARGER than initially thought (Critique Point 5) — it provides 20-50% noise reduction at extreme keV on top of CMV.

4. **Spatial variation of weights**: Siemens uses pre-computed LUTs. Global Σ should work for initial validation. Local Σ is a refinement.

5. **Numerical CMV noise prediction**: MUST compute σ²_VMI(E) with actual A matrix and Σ before implementation (Critique Point 2). This is the gating validation step.

6. **Dominant fix: skipping decomposition vs number of bins**: The 4-bin advantage is modest (~1.3× over optimal 2-bin per Yang 2024). The REAL win is skipping the ill-conditioned matrix inversion entirely (Critique Point 3). Even 2-bin CMV (direct image combination, no material maps) would be dramatically better than polynomial decomposition.
