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

### 3e. Key: Use All 4 Bins, Don't Collapse to 2

With 4 bins collapsed to 2: K=2, M=2 → 0 DOF for noise optimization. CMV reduces to standard 2-material decomposition (same noise amplification as Alvarez-Macovski).

With all 4 bins: K=4, M=2 → 2 DOF for noise optimization. CMV can achieve significantly lower noise.

**This is the single most important insight:** the current 4→2 bin collapsing destroys the noise optimization capability.

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

### 4c. Why Mono+ Alone Is Insufficient

Our current implementation already uses Mono+. The problem is the *input* to Mono+ is too noisy:
- Our raw 70 keV VMI: σ=124 HU (should be ~35)
- Mono+ replaces HF noise from 70 keV, but 70 keV itself is 3.5x too noisy
- Fix the raw VMI noise via CMV first, then Mono+ provides the finishing touch

## 5. Covariance Matrix Estimation

### 5a. Analytical (from photon counts)

For FBP-reconstructed bin images, the noise variance of bin k is approximately:
```
σ²_k ≈ C / N_k
```
where N_k = mean photon count in bin k, C = constant depending on ramp filter, geometry, etc.

For PCCT with anti-coincidence logic, inter-bin covariances arise from charge sharing:
```
Σ_ij ≈ -C · N_shared_ij / (N_i · N_j)
```
where N_shared_ij = counts reassigned between bins i and j by anti-coincidence.

### 5b. Empirical (from data)

Alternatively, estimate Σ from the reconstructed bin images themselves:
1. Select a uniform ROI (e.g., water background)
2. Compute sample covariance across the K bin images

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

With CMV and 4 bins at 140 kVp, estimated noise levels:
- 40 keV: σ ≈ 50-60 HU (clinical: 56.4)
- 70 keV: σ ≈ 30-40 HU (clinical: 35.2) — can be below poly FBP
- 100 keV: σ ≈ 25-35 HU (clinical: 32.5) — anti-correlation helps
- 140 keV: σ ≈ 25-35 HU (clinical: 31.8) — near-optimal combination

## 9. Key Citations

| Reference | Relevance |
|-----------|-----------|
| Gilat Schmidt TG, Med Phys 2009;36(7):3018-27 | Optimal image-based weighting theory. Weight ∝ contrast/variance. |
| Yu L, Leng S, McCollough CH, AJR 2012;199:S9-S15 | Image-domain VMI decomposition framework |
| Leng S et al., Med Phys 2011;38(9):4946-57 | Noise reduction via optimal energy weighting in spectral CT |
| Grant KL et al., Invest Radiol 2014;49(9):586-92 | Mono+ frequency-split algorithm |
| Alvarez RE, Macovski A, Phys Med Biol 1976 | Classical sinogram-domain decomposition (what NOT to do for noise) |
| Roessl E, Herrmann C, Phys Med Biol 2009;54(5):1307-18 | CRLB for VMI noise — theoretical minimum |
| US7856134B2 (Stierstorfer/Ruhrnschopf, Siemens) | Siemens patent: image-domain VMI, pre-computed weight LUTs |
| Rajendran K et al., Radiology 2022;303(1):130-8 | First clinical NAEOTOM Alpha evaluation — noise curves |
| Yang et al., Med Phys 2025, doi:10.1002/mp.17489 | Pre-log/post-log/MD optimal weighting comparison |

## 10. Open Questions

1. **Exact Mono+ filter parameters**: The LP cutoff frequency/shape is proprietary (Siemens). Our current linear scaling `σ_lp = 1.5 + 0.02·|E-E_opt|` is a reasonable starting point but may need tuning.

2. **Spatial variation of weights**: The Siemens patent mentions material-class-dependent weights and pre-computed LUTs. A simple global Σ may not capture local noise variations.

3. **Role of anti-coincidence anti-correlations**: The magnitude of negative off-diagonal entries in Σ depends on charge sharing modeling. Need to verify our DRM produces realistic anti-correlations.

4. **QIR contribution**: The user says clinical VMI is FBP, but DICOM headers say "Q3". If QIR IS applied, our target noise levels may be unachievable with pure FBP + CMV. Should check whether CMV alone can reach σ≈32-56 HU range.
