# PCCT VMI Algorithm Spec — FINAL (VMI-004 Refinement)

## 1. Problem Statement

Our PCCT VMI has correct HU accuracy but catastrophically wrong noise:
- Clinical noise (QIR-3): 40keV=56.4, 70keV=35.2, 100keV=32.5, 140keV=31.8 HU (DECREASING)
- Our noise: 40keV=1063, 70keV=124, 100keV=201, 140keV=297 HU (V-shaped)
- Poly FBP: clinical 66.4, simulated 74.5 HU (close match)

**Root cause:** Sinogram-domain polynomial decomposition amplifies noise catastrophically (4th-order polynomial with 15 terms maps small noise → large errors). Compounded by collapsing 4 bins → 2 (losing noise-optimization DOF) and applying FBP ramp filter to already-noisy material sinograms.

**Solution:** Replace with image-domain Constrained Minimum-Variance (CMV) bin weighting + Mono+ frequency-split blending.

## 2. The 3-Layer Noise Model

Clinical VMI noise is the product of three independent noise-reduction layers:

```
Layer 1: CMV image-domain weighting     → σ_CMV(E)
Layer 2: Mono+ frequency-split blending → σ_Mono+(E) ≈ σ_CMV(E) × R_mono(E)
Layer 3: QIR iterative reconstruction   → σ_clinical(E) ≈ σ_Mono+(E) × R_qir
```

**Our simulation target: Layers 1+2 only (no QIR).** We match against FBP-equivalent targets.

### 2a. Noise Budget

| keV | Clinical (QIR-3) | FBP-equiv (÷0.63) | CMV-only target | CMV+Mono+ target | Our current |
|-----|-------------------|--------------------|-----------------|-------------------|-------------|
| 40  | 56.4 HU           | ~90 HU             | ~120-150 HU     | ~80-100 HU        | 1063 HU     |
| 70  | 35.2              | ~56 HU             | ~70-80          | ~55-65            | 124         |
| 100 | 32.5              | ~52 HU             | ~60-70          | ~45-55            | 201         |
| 140 | 31.8              | ~50 HU             | ~55-65          | ~40-55            | 297         |

**Key validations:**
- CMV at 70 keV ≈ poly FBP noise (Bhattarai 2024: VMI 70keV ≈ T3D, <1 HU difference at same QIR level)
- Our poly FBP = 74.5 HU → CMV 70 keV should be ~70-80 HU
- QIR-3 reduces noise by ~37% (Eberhard 2023: 21→13 HU at 70keV, 38% reduction)

### 2b. Why Clinical VMI(140keV)=31.8 < poly=66.4 HU

This is NOT achievable with CMV+Mono+ alone. It requires all three layers:
1. CMV: ~55-65 HU at 140 keV (slightly below poly FBP due to 2 extra DOF from 4 bins)
2. Mono+: ~40-50 HU (replaces HF noise content with optimal-keV texture)
3. QIR-3: ~25-35 HU (37% reduction → lands at ~31.8 HU)

The "impossible" observation dissolves once QIR-3 is included.

## 3. Algorithm: CMV Image-Domain Bin Weighting

### 3a. Mathematical Formulation

**Given:**
- K=4 energy bins, each reconstructed via FBP: `I_1(x,y), ..., I_4(x,y)` (linear attenuation, cm⁻¹)
- M=2 basis materials (water, iodine)
- Noise covariance: `Σ ∈ R^(4×4)` — **diagonal** in our simulation (no anti-correlations in MC DRM)
- Material-bin attenuation matrix: `A ∈ R^(4×2)`, where `A[k,m]` = mean linear attenuation of material m in bin k
- Target vector: `t(E) ∈ R^2`, where `t[m]` = linear attenuation of material m at monoenergetic energy E

**Optimization:**
```
minimize    w'Σw            (image noise variance)
subject to  A'w = t(E)      (monoenergetic fidelity: 2 constraints)
```

**Closed-form solution (Lagrange multipliers):**
```
w*(E) = Σ⁻¹ A (A'Σ⁻¹A)⁻¹ t(E)     ∈ R^4
```

**Minimum variance:**
```
σ²_VMI(E) = t(E)' (A'Σ⁻¹A)⁻¹ t(E)   (in same units² as bin images)
```

### 3b. A Matrix Construction

For each bin k ∈ {1,2,3,4} and material m ∈ {water, iodine}:

```
         Σ_i  w_full[i] × η[i] × R[i,k] × μ_m(E_i)
A[k,m] = ——————————————————————————————————————————————
              Σ_i  w_full[i] × η[i] × R[i,k]
```

Where:
- `w_full[i]`: source spectrum weight at energy E_i (photon fluence)
- `η[i]`: quantum efficiency at E_i (CdTe, thickness-dependent)
- `R[i,k]`: DRM probability that photon at E_i registers in bin k (from `compute_mc_drm`, shape [200,4])
- `μ_m(E_i)`: linear attenuation coefficient of material m at E_i (cm⁻¹)

**Units:** cm⁻¹ (linear attenuation at the spectrum-weighted mean energy of each bin).

**For water:** `μ_w(E) = mac_water(E) × 1.0` (g/cm³ density = 1.0)
**For iodine:** Use mass attenuation coefficient `(μ/ρ)_I(E)` directly, so A[k,2] has units cm²/g. Then t[2](E) = `(μ/ρ)_I(E)` (cm²/g) as well. This ensures the decomposition coefficients are (water fraction, iodine mass density in g/cm³).

**Important:** We already compute `w_full[i] × η[i] × R[i,k]` for each bin in the existing calibration code (nb07 lines 1589-1592). The only change is using these weights for CMV instead of polynomial fitting.

### 3c. Target Vector t(E)

For VMI at monoenergetic energy E:
```
t[1](E) = μ_water(E)      [cm⁻¹]    — water linear attenuation at E
t[2](E) = (μ/ρ)_iodine(E) [cm²/g]   — iodine mass attenuation at E
```

### 3d. Covariance Matrix Σ

**Our simulation: diagonal (no anti-correlations in MC DRM).**

```
Σ = diag(σ²_1, σ²_2, σ²_3, σ²_4)
```

**Two approaches (use BOTH for cross-validation):**

**Approach 1 — Analytical (from photon counts):**
```
σ²_k = C / N_k
```
where `N_k = I0_bins[k]` (unattenuated photon count per bin, available from `sim_scan2.I0_bins`). The constant C is identical for all bins (same geometry, filter, reconstruction). For relative weights, C cancels:
```
Σ_analytical = diag(1/N_1, 1/N_2, 1/N_3, 1/N_4)
```

**Approach 2 — Empirical (from reconstructed bin images):**
Reconstruct all 4 bins via FBP. Measure σ²_k from a large uniform water ROI in each bin image. Use ROI ≥ 20mm diameter to average over FBP noise correlations.
```
Σ_empirical = diag(var(ROI_1), var(ROI_2), var(ROI_3), var(ROI_4))
```

**Cross-validation:** Check that σ²_k ratios are consistent between analytical and empirical approaches: σ²_1/σ²_2 ≈ N_2/N_1 (should hold for Poisson-dominated noise).

### 3e. Why This Fixes the Noise Problem

| Factor | Current approach | CMV approach | Impact |
|--------|-----------------|--------------|--------|
| Polynomial amplification | 4th-order poly (15 terms) amplifies noise in sinogram domain | No polynomial — direct image-domain linear combination | **5-10× reduction** (dominant) |
| FBP on noisy sinograms | FBP ramp filter amplifies noisy material sinograms | FBP only on raw bin sinograms (reasonable noise) | **2-3× reduction** |
| Bin DOF | 4→2 collapsing (0 extra DOF) | 4 bins, 2 constraints (2 DOF for noise minimization) | **~1.3× reduction** (modest) |

**Combined expected improvement: ~13-40× noise reduction at 40 keV** (from 1063 → ~80-120 HU). This lands squarely in the FBP-equivalent target range.

## 4. Mono+ Frequency-Split Blending

### 4a. Algorithm (Grant et al. 2014, Invest Radiol 49:586-92)

Applied per-slice in image domain after CMV weighting:

```
Mono+(E) = LP(VMI(E)) + [VMI(E_opt) - LP(VMI(E_opt))]
         = LP(VMI(E)) + HP(VMI(E_opt))
```

Where:
- VMI(E) = CMV-weighted image at target energy E (high contrast, higher noise at extreme keV)
- VMI(E_opt) = CMV-weighted image at optimal keV ≈ 70 keV (lowest noise)
- LP = Gaussian low-pass filter in Fourier domain
- HP = I - LP (high-pass complement)
- Result: low-frequency contrast from target E, high-frequency noise texture from E_opt

### 4b. Filter Parameters

**The exact Siemens parameters are proprietary** (confirmed: no public source after exhaustive search). Our implementation uses:

```
E_opt = 70 keV
σ_lp(E) = 1.5 + 0.02 × |E - E_opt|  [mm]
```

At 40 keV: σ = 2.1 mm; at 70 keV: σ = 1.5 mm; at 140 keV: σ = 2.9 mm.

**NPS constraints from literature (Cester 2022, Monsivais 2025):**
- At 70 keV: NPS matches poly FBP — minimal/no frequency manipulation
- At 40 keV: low-frequency NPS peak at 0.0-0.1 mm⁻¹ — LP cutoff ~0.1-0.2 mm⁻¹
- Effect strengthens with keV distance from optimal

**Note:** Grant et al. describes a "regional-spatial" algorithm, suggesting possible spatial adaptivity (different blending in high-contrast vs low-contrast regions). Our global filter is a simplification. If contrast artifacts appear near iodine/bone edges, spatial adaptivity may be needed.

### 4c. Expected Noise Reduction from Mono+

At extreme keV (40, 140+): 20-50% noise reduction on top of CMV.
At optimal keV (70): ~0% (VMI(E) ≈ VMI(E_opt), filter has no effect).

Evidence: Literature shows noise ratio 40/70 keV = 1.67× (Eberhard 2023, QIR-off). Pure CMV theory (from target vector scaling) predicts ~2-3× ratio. The difference is Mono+ compressing the curve toward 70 keV noise at extreme energies.

## 5. Numerical Validation Recipe (MUST DO BEFORE IMPLEMENTATION)

### 5a. Predicted CMV Noise

Compute σ²_VMI(E) at 40, 70, 100, 140 keV using actual simulation parameters:

```julia
# 1. Build A matrix (4×2) from DRM and spectrum
A = zeros(4, 2)
for k in 1:4
    wb = [w_full[i] * η[i] * R[drm_row(e[i]), k] for i in eachindex(e)]
    wb_n = wb / sum(wb)
    A[k, 1] = sum(wb_n .* μ_water.(e))
    A[k, 2] = sum(wb_n .* mac_iodine.(e))  # mass attenuation (cm²/g)
end

# 2. Build Σ (diagonal, from I0 per bin — relative scale)
Σ = Diagonal(1.0 ./ I0_bins)

# 3. Compute noise amplification factor per energy
Σ_inv = inv(Σ)
F = A' * Σ_inv * A       # Fisher information (2×2)
F_inv = inv(F)

for E in [40, 70, 100, 140]
    t_E = [μ_water(E), mac_iodine(E)]
    σ²_rel = t_E' * F_inv * t_E          # Relative variance
    w_opt = Σ_inv * A * F_inv * t_E       # Optimal weights (4-vector)
    println("E=$E keV: σ²_rel=$(σ²_rel), weights=$(w_opt)")
end

# 4. Calibrate to absolute HU noise
# At 70 keV, CMV noise ≈ poly FBP noise ≈ 74.5 HU (our sim)
# Scale all σ²_rel values by: C = (74.5)² / σ²_rel(70keV)
# Then σ_HU(E) = sqrt(C × σ²_rel(E))
```

### 5b. Validation Criteria

| Check | Pass condition |
|-------|---------------|
| CMV noise at 70 keV ≈ poly FBP | σ_CMV(70) ≈ 74 ± 10 HU (after calibration) |
| Noise monotonically decreasing 40→140 keV | σ(40) > σ(70) > σ(100) > σ(140) |
| Noise ratio 40/70 keV | 1.5-3.0× (literature: 1.67× with Mono+, higher without) |
| CMV+Mono+ at 40 keV | ≤ 100 HU (FBP-equiv target ~90 HU) |
| CMV+Mono+ at 140 keV | ≤ 65 HU (FBP-equiv target ~50 HU) |
| HU accuracy preserved | Mean HU within ±5 HU of current (noiseless) VMI values |
| Weights at 70 keV | All positive, roughly proportional to N_k (energy-integrating-like) |
| Weights at 40 keV | Some negative weights expected (to boost photoelectric contrast) |

### 5c. If Validation Fails

If CMV+Mono+ cannot reach FBP-equivalent targets:
1. Check A matrix condition number: `cond(A'*Σ_inv*A)` — should be < 100
2. Check I0 per bin distribution — heavily unbalanced bins reduce CMV benefit
3. Consider: our simulation may simply have higher noise floor than clinical due to missing anti-correlations in DRM. Document the gap quantitatively.

## 6. Implementation Pseudocode (for nb07)

### Cell: CMV VMI Reconstruction

```julia
# ============================================================
# PCCT VMI via Constrained Minimum-Variance (CMV) Bin Weighting
# ============================================================
# References:
#   Gilat Schmidt, Med Phys 2009;36:3018-27
#   Leng et al., Med Phys 2011;38:4946-57
#   Yu et al., AJR 2012;199:S9-S15

using LinearAlgebra

# --- Step 1: Reconstruct all 4 bins independently via FBP ---
bin_images = Vector{Array{Float32,3}}(undef, 4)
for b in 1:4
    # Use sim_scan2.bins[b] (line-integral sinogram for bin b)
    # Use same filter kernel as poly FBP reconstruction
    bin_images[b] = Array(BS.reconstruct!(ws_fdk,
        CuArray(Float32.(sim_scan2.bins[b])), geom, recon_size))
end

# --- Step 2: Build A matrix (4×2) ---
# Effective bin spectra from DRM × source spectrum × QE
A = zeros(Float64, 4, 2)
for k in 1:4
    wb = [w_full[i] * η_vec[i] * R_mat[drm_row(e_full[i], kVp, size(R_mat,1)), k]
          for i in eachindex(e_full)]
    wb ./= sum(wb)  # Normalize to probability weights
    A[k, 1] = sum(wb .* μ_water_interp.(e_full))       # cm⁻¹
    A[k, 2] = sum(wb .* mac_iodine_interp.(e_full))     # cm²/g
end

# --- Step 3: Build Σ (diagonal from I0 per bin) ---
# Option A: Analytical (relative — sufficient since C cancels in weights)
Σ_inv = Diagonal(Float64.(sim_scan2.I0_bins))  # inv(diag(1/N_k)) = diag(N_k)

# Option B: Empirical (from reconstructed bin images, for cross-validation)
# roi_mask = circular_roi(center, radius_pix)
# for k in 1:4; σ²_emp[k] = var(bin_images[k][roi_mask]); end
# Σ_inv_emp = Diagonal(1.0 ./ σ²_emp)

# --- Step 4: CMV weights for each VMI energy ---
F = A' * Σ_inv * A           # Fisher information matrix (2×2)
F_inv = inv(F)                # Inverse Fisher (2×2)
P = Σ_inv * A * F_inv        # Pre-multiply (4×2) — reuse for all energies

vmi_energies = [40.0, 70.0, 100.0, 140.0]  # keV
vmi_images = Dict{Float64, Array{Float32,3}}()

for E in vmi_energies
    t_E = [μ_water(E), mac_iodine(E)]    # Target vector (2×1)
    w = P * t_E                            # Optimal weights (4×1)

    # Predicted noise (relative): σ²_rel = t' F⁻¹ t
    σ²_rel = t_E' * F_inv * t_E
    @info "VMI $E keV: weights=$w, σ²_rel=$σ²_rel"

    # --- Step 5: Weighted sum → VMI ---
    vmi_μ = zeros(Float32, size(bin_images[1]))
    for k in 1:4
        vmi_μ .+= Float32(w[k]) .* bin_images[k]
    end

    # --- Step 6: HU conversion ---
    μ_w_E = Float32(μ_water(E))
    vmi_hu = @. (vmi_μ - μ_w_E) / μ_w_E * 1000f0

    vmi_images[E] = vmi_hu
end

# --- Step 7: Mono+ blending ---
E_opt = 70.0
for E in vmi_energies
    E == E_opt && continue
    σ_lp = 1.5 + 0.02 * abs(E - E_opt)
    vmi_images[E] = mono_plus_vmi(vmi_images[E], vmi_images[E_opt];
        σ_lp_mm=σ_lp, pixel_mm=pixel_mm)
end
```

### Helper: drm_row mapping (already in codebase)
```julia
drm_row(E, kVp, n_R) = clamp(round(Int, (E - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
```

## 7. Literature Benchmarks

### 7a. NAEOTOM Alpha VMI Noise (QIR-off or WFBP, closest to our FBP)

| Source | Anatomy | Kernel | Dose | 40 keV | 70 keV | 140 keV | 40/70 ratio |
|--------|---------|--------|------|--------|--------|---------|-------------|
| Eberhard 2023 | Coronary phantom | Bv40 | 9.4 mGy | 35 HU | 21 HU | — | 1.67× |
| Boehm 2023 | Head cortical | Qr36 | 48.8 mGy | 12.8 HU | 5.0 HU | — | 2.56× |
| Greffier 2023 | Catphan 20cm | Qr40 | 10 mGy | — | 10.7 HU | — | — |
| Yalynska 2022 | Pulmonary | — | — | 27.3 HU | 15.6 HU | — | 1.75× |

**Note:** All "QIR-off" literature values include Siemens minimal statistical optimization (not true FBP). Our simulation FBP is strictly unoptimized → expect our noise to be ~10-20% higher than QIR-off literature values.

### 7b. 70 keV VMI vs Polychromatic Noise

| Source | Finding |
|--------|---------|
| Bhattarai 2024 (PMC10796834) | VMI 70keV ≈ T3D poly noise at same QIR (<1 HU difference) |
| Monsivais 2025 (Med Phys) | NPS shape and magnitude similar for T3D and 70keV VMI |
| Cester 2022 (QIMS) | VMI 70keV noise ~7.6% below poly (linear-blended images) on DECT |
| Yalynska 2022 (Diagnostics) | VMI 70keV 25% lower noise than poly SPP (suggests Mono+ applied) |

**Conclusion:** CMV alone at 70 keV ≈ poly FBP. With Mono+ at 70 keV, may be 0-25% below poly.

## 8. Key Citations

| Reference | DOI/PMID | Relevance |
|-----------|----------|-----------|
| Gilat Schmidt TG, Med Phys 2009;36(7):3018-27 | PMID:19673196 | Optimal image-based weighting theory |
| Yu L, Leng S, McCollough CH, AJR 2012;199:S9-S15 | PMID:23097173 | Image-domain VMI framework |
| Leng S et al., Med Phys 2011;38(9):4946-57 | PMID:21978039 | Noise reduction via optimal energy weighting |
| Grant KL et al., Invest Radiol 2014;49(9):586-92 | PMID:24710203 | Mono+ frequency-split algorithm |
| Alvarez RE, Macovski A, Phys Med Biol 1976 | — | Sinogram-domain decomposition (baseline) |
| Roessl E, Herrmann C, Phys Med Biol 2009;54(5):1307-18 | — | CRLB for VMI noise |
| US7856134B2 (Stierstorfer/Ruhrnschopf) | — | Siemens patent: image-domain VMI with LUTs |
| Rajendran K et al., Radiology 2022;303(1):130-8 | PMID:34904876 | First clinical NAEOTOM evaluation |
| Yang et al., Med Phys 2025, doi:10.1002/mp.17489 | — | Pre/post-log/MD optimal weighting comparison |
| Yang et al., Med Phys 2024;51(1):224-238, doi:10.1002/mp.16590 | — | Optimal 2-ch ≈ 4-ch (within 0.27%) |
| Bhattarai et al., Med Phys 2024;51(2), PMC10796834 | — | VMI 70keV ≈ T3D noise |
| Eberhard/Mergen et al., Eur Radiol Exp 2023, PMC9975359 | — | QIR-off vs QIR-1/2/3/4 VMI noise |
| Boehm/Michael et al., Clin Neuroradiol 2023, PMC10881631 | — | Head CT VMI noise QIR-off |
| Greffier et al., Diagnostics 2023, PMC10092985 | — | Catphan 70keV WFBP: 10.7 HU |
| Cester D et al., QIMS 2022;12(1):726-741 | — | NPS of VMI+ vs standard VMI; Mono+ filter constraints |
| Monsivais H et al., Med Phys 2025, doi:10.1002/mp.70067 | — | 3D NPS of NAEOTOM VMI; 70keV ≈ T3D shape |
| Yalynska T et al., Diagnostics 2022;12(11):2715 | — | VMI 70keV 25% below poly SPP noise |
| Kawashima H et al., Phys Eng Sci Med 2024;48(1):143-153 | — | PCCT 26-40% higher detectability than DSCT at 40keV |
| Wang AS, Pelc NJ, IEEE TRPMS 2021;5(4):453-464 | — | CRLB/Fisher info for spectral PCD CT |
| D'Angelo T et al., Br J Radiol 2019;92(1098):20180546 | — | Mono+ description: "regional-spatial frequency-split" |
| Heismann et al., Med Phys 2025, doi:10.1002/mp.17591 | — | PCCT 10% noise advantage from quantum counting |
| Niu et al., Phys Med Biol 2018, PMC5903446 | — | Condition numbers: MECT(N=4) = 301, DECT+prior = 17.86 |

## 9. Open Questions (Post-Refinement)

### Resolved
1. **QIR contribution** — Clinical targets include QIR-3 (~37% reduction). FBP-equiv targets computed.
2. **Anti-correlations** — Absent from our MC DRM. Σ is diagonal. CMV uses DOF only.
3. **Dominant fix** — Skipping polynomial decomposition, not bin count. Even 2-bin CMV >> poly decomp.
4. **Mono+ role** — Essential component, 20-50% noise reduction at extreme keV on top of CMV.
5. **A matrix interpretation** — Image-domain: mean linear attenuation of material m in bin k (cm⁻¹ or cm²/g).
6. **Σ estimation** — Analytical (1/N_k) preferred over empirical for diagonal case.

### Deferred (not blocking implementation)
7. **Mono+ exact filter params** — Proprietary. Our heuristic is reasonable. Tune against NPS data if available.
8. **Spatial adaptivity** — Grant et al. mentions "regional-spatial" variant. Global filter first, add adaptivity if edge artifacts appear.
9. **Physical anti-correlations** — Would require DRM enhancement (anti-coincidence model). Improves CMV noise cancellation. Future work.
10. **Local Σ** — Spatially varying weights via local covariance estimation. Not needed for phantom validation.
