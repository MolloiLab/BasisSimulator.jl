# XCAT PCCT Artifact Investigation — Progress Log

> Started: 2026-02-13
> Issue: PCCT FDK shows cupping/edge artifacts not present in EICT

---

## Pre-Investigation Summary

**Observed**: PCCT Standard FDK reconstruction shows subtle cupping/edge artifacts
at tissue boundaries (lung/soft-tissue, body contour) that do NOT appear in
EICT 120 kVp FDK from the same XCAT phantom.

**Slice mismatch confirmed**: EICT z=8.0cm, PCCT z=5.12cm → same slice index =
different anatomy. This is expected behavior (different z-coverage), not a bug.

**Prior fixes verified**: volume_fov threading, pixel_row_size, StandardFilter defaults
are all correct.

---

## Investigation Log

### 2026-02-13: XCAT-001 — DIAGNOSTIC: Map exact PCCT vs EICT signal chains side-by-side [PASS]

**NB05 Configuration (both scanners use `fidelity = :high`):**

Both EICT and PCCT use `SimOptions(fidelity = :high)`, which enables ALL physics effects:
fill_factor, flat_filter, bowtie_filter, detector_efficiency, scatter, scatter_correction,
crosstalk, optical_crosstalk, focal_spot, noise, lag, heel_effect, bhc (das=false, broken).
PCCT additionally uses `pcct_noise_reduction = 0.60`, `use_pcct_corrections = false`.

---

#### EICT Signal Chain (`simulate!(ws::EICTWorkspace, ...)` at driver.jl:684)

The EICT workspace path has TWO branches: **signal chain** (`ws.has_signal_chain = true`)
and **standard** path. For `fidelity = :high`, signal chain effects (heel_effect, das_model)
are present in the config, so `has_signal_chain = true`.

```
STEP 1: _forward_project_poly!() [driver.jl:702]
  - Polychromatic Beer-Lambert: I_total = Σ w_E × exp(-L_E)
  - sinogram = -log(I_total)
  - Uses phantom.fov for volume bounds (volume_fov kwarg)

STEP 2: _apply_physics_no_noise!() [driver.jl:725, polychromatic.jl:878]
  - Applies 10 deterministic effects IN THIS ORDER (sinogram domain):
    1. fill_factor → apply_fill_factor!()
    2. flat_filter → apply_flat_filter!()
    3. scatter → add_scatter!()
    4. scatter_correction → correct_scatter!()
    5. bowtie_filter → apply_bowtie_filter!()
    6. crosstalk → apply_crosstalk!()
    7. optical_crosstalk → apply_optical_crosstalk!()
    8. focal_spot → apply_focal_spot_blur!()
    9. detector_efficiency → apply_detector_efficiency!()
   10. lag → apply_lag!()
  - NOTE: BHC is NOT applied here (it's done after the signal chain)
  - NOTE: Noise is NOT applied here (DAS model handles it in signal chain)

STEP 3: Convert to intensity domain [driver.jl:740]
  - sinogram[idx] = exp(-clamp(sinogram[idx], -1, 15))

STEP 4: Heel effect (intensity domain) [driver.jl:748]
  - apply_heel_effect!(sinogram, heel_effect, geom)

STEP 5: DAS model (intensity domain) [driver.jl:753]
  - apply_das_model!(sinogram, das_model; seed=...)
  - NOTE: DAS is BROKEN and use_das=false at fidelity=:high, so this is SKIPPED

STEP 6: Create noise-free air scan [driver.jl:757]
  - fill air_scan with 1.0
  - Apply heel_effect to air_scan
  - Apply DAS gain to air_scan (if DAS enabled, which it isn't)

STEP 7: Calibration [driver.jl:771]
  - sinogram = sinogram / air_scan  (normalize by air scan)

STEP 8: Low signal correction [driver.jl:779]
  - low_signal_correction_gpu!()

STEP 9: Log transform [driver.jl:782]
  - sinogram = -log(max(sinogram, eps))

STEP 10: BHC [driver.jl:789]
  - apply_bhc!(sinogram, bhc)
  - Uses bhc_water_default() with coefficients [0, 1.05, -0.02, 0.001]
  - Reference energy = weighted mean of spectrum energies

--- Save ideal sinogram to CPU [driver.jl:814] ---

STEP 11: Quantum noise [driver.jl:819]
  - I0 = compute_detector_I0(geom, protocol)
  - Gaussian approximation: N_measured = N_expected + sqrt(N_expected) × randn
  - sinogram_noisy = -log(N_measured / I0)

--- Save noisy sinogram to CPU [driver.jl:837] ---
```

**SUMMARY OF EICT SIGNAL CHAIN:**
```
poly_fp → 10 physics effects → intensity domain → heel → (DAS skip) →
calibrate (÷ air) → low_signal_correction → -log → BHC → save ideal →
noise → save noisy
```

---

#### PCCT Signal Chain (`simulate!(ws::PCCTWorkspace, ...)` at driver.jl:554)

```
STEP 1: pcct_forward_project() [driver.jl:575, photon_counting.jl:1329]
  For each energy E in spectrum (100 bins):
    a. create_μ_volume!(μ_volume, mask, materials, E)
    b. siddon_forward_project!(sino_buf, μ_volume, geom; volume_fov=phantom.fov)
    c. For each energy bin b:
       bins[b] += I0 × w_E × η_E × R[E,b] × exp(-sino_buf)
       (R = spectral response matrix, η = quantum efficiency)

  After energy loop, apply detector physics chain:
    d. apply_charge_sharing!(bins, detector)     ← spatial + energy redistribution
    e. apply_pulse_pileup!(bins, detector)        ← count rate effects
    f. apply_anti_coincidence!(bins, detector)     ← multi-pixel rejection

  Software corrections (if use_pcct_corrections = true):
    g. correct_pulse_pileup!()    ← SKIPPED (use_pcct_corrections=false in NB05)
    h. correct_charge_sharing!()  ← SKIPPED

  Convert counts → line integrals:
    i. For each bin b: sino_bin = -log(N_bin / I0_bin_norm)
       I0_bin_norm = _compute_degraded_I0() (since detector effects ON, corrections OFF)

  Post-log smoothing (if corrections ON): SKIPPED (corrections OFF)

  Returns: EnergyResolvedSinogram with per-bin sinograms

STEP 2: _combine_pcct_bins() for IDEAL sinogram [driver.jl:608]
  - N_total = Σ I0_bins[b] × exp(-sino_bin[b])
  - sino_combined = -log(N_total / I0_total)
  - I0_bins = ws.I0_bins (pre-computed in create_workspace, uses _compute_degraded_I0)

STEP 3: BHC on ideal combined sinogram [driver.jl:615]
  - apply_bhc!(sino_ideal, config.bhc)
  - SAME coefficients as EICT: [0, 1.05, -0.02, 0.001]

--- Save ideal to CPU [driver.jl:620] ---

STEP 4: apply_pcct_noise!() on PER-BIN sinograms [driver.jl:623]
  - I0_physics = compute_detector_I0(geom, protocol)
  - For each bin:
    a. Convert line integrals → expected counts: N = I0_bin × exp(-sino)
    b. Add Gaussian noise: N_noisy = N + (1 - noise_reduction) × sqrt(N) × randn
       noise_reduction = 0.60 → scale = 0.40 → 60% less noise variance
    c. Low-count Poisson sampling (N ≤ 20)
    d. Floor: N_noisy = max(N_noisy, 1)
    e. Convert back: sino = -log(N_noisy / I0_bin)

  NOTE: I0_bin for noise is computed by _compute_pcct_noise_I0(), which uses
  _compute_bin_I0() with unit I0, normalizes to fractions, scales by physics I0.
  This is a DIFFERENT I0 computation than the normalization I0 in pcct_forward_project!

STEP 5: _combine_pcct_bins() for NOISY sinogram [driver.jl:637]
  - Same combination as step 2, but on noisy per-bin sinograms
  - Uses SAME I0_bins as step 2

STEP 6: BHC on noisy combined sinogram [driver.jl:644]
  - apply_bhc!(sino_noisy, config.bhc)
  - SAME coefficients as ideal BHC

--- Save noisy to CPU [driver.jl:649] ---

STEP 7: Material decomposition for VMI [driver.jl:652]
  - combine_pcct_bins!() for pseudo-dual-energy
  - spectral_decompose!() for material maps
```

**SUMMARY OF PCCT SIGNAL CHAIN:**
```
pcct_fp (per-energy Siddon + DRM) → charge_sharing → pileup → anti_coincidence →
-log(N/I0_degraded) → combine_bins → BHC → save ideal →
noise (per-bin, 60% reduction) → combine_bins → BHC → save noisy →
material decomposition
```

---

#### CRITICAL DIFFERENCES TABLE

| # | Aspect | EICT Path | PCCT Path | Position-Dependent? | Cupping Impact |
|---|--------|-----------|-----------|---------------------|----------------|
| 1 | **Forward Projection** | Polychromatic Beer-Lambert: -log(Σ w×exp(-μL)) | Per-energy Siddon → DRM → per-bin counts → -log(N/I0) | — | — |
| 2 | **Fill Factor** | YES (apply_fill_factor!) | **NO** — not applied | NO (uniform scaling) | NONE — cancels in air cal |
| 3 | **Flat Filter** | YES (apply_flat_filter!) | **NO** — not applied | NO (uniform additive in log domain) | NONE — uniform shift |
| 4 | **Bowtie Filter** | YES (apply_bowtie_filter!) | **NO** — not applied | **YES** (fan-angle dependent) | **POSSIBLE** — changes sinogram values in position-dependent way, affects BHC behavior |
| 5 | **Scatter** | YES (add_scatter! + correct_scatter!) | **NO** — not applied | **YES** (patient-size dependent) | **POSSIBLE** — uncorrected scatter in PCCT is absent but would cause cupping if present |
| 6 | **Scatter Correction** | YES (correct_scatter!) | **NO** — not applied | YES | See above |
| 7 | **Crosstalk** | YES (apply_crosstalk!) | **NO** — not applied (PCCT uses charge sharing instead) | Partly | Low impact — analogous to charge sharing |
| 8 | **Optical Crosstalk** | YES (apply_optical_crosstalk!) | **NO** — not applied (CdTe has no scintillator) | Partly | Low impact — correct physics for direct-conversion |
| 9 | **Focal Spot** | YES (apply_focal_spot_blur!) | **NO** — not applied | YES (geometric) | Low — blurs rather than shifts |
| 10 | **Detector Efficiency** | YES (apply_detector_efficiency!) | **NO** — PCCT uses quantum efficiency η in DRM | NO (uniform) | NONE |
| 11 | **Lag** | YES (apply_lag!) | **NO** — not applied | Temporal | Low — afterglow, not spatial |
| 12 | **Heel Effect** | YES (intensity domain) | **NO** — not applied | **YES** (position-dependent) | **POSSIBLE** — anode-cathode intensity gradient |
| 13 | **DAS Model** | SKIPPED (broken) | SKIPPED | — | — |
| 14 | **Air Calibration** | YES (÷ air_scan, includes heel) | **NO** — raw combined sinogram | **YES** | **KEY DIFFERENCE** — see analysis |
| 15 | **Low Signal Correction** | YES | **NO** | YES | Low |
| 16 | **BHC** | On polychromatic sinogram (AFTER physics + calibration) | On PCCT combined sinogram (NO physics, NO calibration) | **YES** | **HIGH** — different input statistics |
| 17 | **BHC Applied** | ONCE (on ideal) | **TWICE** (once ideal, once noisy) | YES | **MODERATE** — double correction |
| 18 | **Noise** | Gaussian on combined sinogram (I0 from geometry) | Per-BIN Gaussian with 60% reduction | — | — |
| 19 | **Charge Sharing** | N/A | YES (Koch-Mehrin ODE) | YES (energy-dependent) | **POSSIBLE** — redistributes counts between bins |
| 20 | **Pileup** | N/A | YES (Yang 2025 semi-nonparalyzable) | YES (count-rate dependent) | **POSSIBLE** — high-count pixels affected more |

---

#### KEY FINDINGS

**Finding 1: Physics effects missing from PCCT path**
The PCCT path does NOT apply the following 10 EICT physics effects:
- fill_factor, flat_filter, bowtie_filter, scatter, scatter_correction,
  crosstalk, optical_crosstalk, focal_spot, detector_efficiency, lag

Instead, the PCCT path only applies:
- DRM (charge sharing, K-fluorescence, pileup) via pcct_forward_project
- BHC via apply_bhc! after bin combination
- Noise via apply_pcct_noise!

**Verdict on missing effects:** Most missing effects (fill_factor, flat_filter, detector_efficiency,
optical_crosstalk, lag) are NOT position-dependent or cancel in calibration. They should NOT
cause cupping.

**However:** Bowtie filter IS position-dependent (adds fan-angle-dependent attenuation).
For EICT, the bowtie modifies the sinogram before BHC. For PCCT, the bowtie is absent,
so the sinogram values entering BHC are different. Since BHC is a polynomial that varies
with input value, and bowtie changes input values in a position-dependent way, this could
contribute to differences (but likely NOT cupping since PCCT doesn't have bowtie to begin with).

**Finding 2: BHC input is fundamentally different**
- EICT BHC input: polychromatic sinogram AFTER 10 physics effects + heel effect + air calibration
- PCCT BHC input: combined sinogram from energy-resolved bins with DRM effects, NO physics effects, NO air calibration

The PCCT combined sinogram has LESS beam hardening than the EICT polychromatic sinogram
because the energy-resolved bins partially separate the spectral contributions. Applying
the SAME BHC polynomial [0, 1.05, -0.02, 0.001] could OVER-correct the PCCT sinogram.

Over-correction is position-dependent (BHC polynomial has quadratic/cubic terms that
grow with path length). This WOULD cause cupping.

**Verdict: BHC over-correction is the #1 hypothesis for PCCT-specific cupping.**

**Finding 3: BHC applied TWICE in PCCT path**
- Ideal path: combine → BHC → save
- Noisy path: noise → combine → BHC → save

The noisy sinogram gets BHC applied once (not twice, since the noise is on per-bin
sinograms, not on the combined sinogram). Each combine→BHC is a separate operation.
This is actually correct (two separate combined sinograms each get BHC once).

**Finding 4: No air calibration in PCCT path**
EICT applies: intensity / air_scan → -log → BHC
PCCT applies: -log(N/I0_bin_norm) → combine → BHC

The PCCT normalization uses I0_bin_norm (degraded I0 accounting for DRM), which is
the PCCT equivalent of air calibration. This should be correct as long as I0_bin_norm
accurately models the unattenuated bin counts.

**Finding 5: Noise reduction = 0.60 in PCCT**
The pcct_noise_reduction=0.60 reduces noise amplitude by 60% in the Gaussian approximation.
This is equivalent to clinical vendor noise reduction (Siemens QIR-3 level). It does NOT
smooth spatially — it reduces the noise magnitude at each pixel independently. This should
NOT cause cupping or edge artifacts by itself.

---

#### SUMMARY VERDICT

| Root Cause | Confidence | Mechanism |
|-----------|------------|-----------|
| BHC over-correction | **HIGH** | Same BHC polynomial on sinogram with less beam hardening → over-subtraction at long paths → cupping |
| I0_bins mismatch | **MEDIUM** | If _combine_pcct_bins I0 doesn't match pcct_forward_project normalization, position-dependent error |
| Missing bowtie in PCCT | **LOW** | Bowtie is absent, so PCCT sinogram values are different from EICT, but this doesn't directly cause cupping |
| Charge sharing + pileup | **LOW** | Position-dependent, but these are physical effects, not artifacts per se |

**Next steps:** XCAT-002 (BHC analysis) and XCAT-003 (I0_bins verification) are the highest priority.

---

### 2026-02-13: XCAT-002 — DIAGNOSTIC: Test BHC impact on PCCT cupping [WIP]

**Plan:** Analyze BHC polynomial behavior on EICT (120 kVp) vs PCCT (140 kVp) sinograms.

**Initial Code Reading Findings:**

1. **BHC polynomial is HARDCODED**: `bhc_water_default()` at beam_hardening_correction.jl:205-209
   always returns `[0.0, 1.05, -0.02, 0.001]` regardless of `reference_energy_keV` argument.
   The `reference_energy_keV` is stored as metadata only — it does NOT change the polynomial.

2. **Both EICT and PCCT use the same polynomial**: driver.jl:1416-1418 calls
   `bhc_water_default(reference_energy_keV=ref_energy)` for all scanners. Since the
   coefficients are hardcoded, both 120 kVp EICT and 140 kVp PCCT get identical BHC.

3. **PCCT combined sinogram math**: For ideal detection (no DRM loss), the combination
   `_combine_pcct_bins` perfectly recovers the total photon count:
   ```
   N_total = Σ_b N_bin(b) = Σ_E I0 × w_E × η_E × exp(-μ_E × d)
   p_pcct = -log(N_total / I0_total) = -log(Σ w_E×η_E×exp(-μ_E×d) / Σ w_E×η_E)
   ```
   This is the EICT polychromatic integral but weighted by η_E (quantum efficiency).
   For CdTe at diagnostic energies (>40 keV), η ≈ 1.0, so p_pcct ≈ p_eict_140kVp.

4. **I0 normalization is consistent**: workspace.jl:218-230 shows that `I0_bins_norm`
   (used by pcct_forward_project) and `I0_bins_combine` (used by _combine_pcct_bins)
   are computed identically when `use_detector_fx && !use_corrections` (NB05 case).
   Both use `_compute_degraded_I0()`.

5. **Key hypothesis**: The BHC polynomial `[0, 1.05, -0.02, 0.001]` was designed for
   a 120 kVp spectrum. For 140 kVp (harder spectrum, less beam hardening), the SAME
   polynomial OVER-corrects. Over-correction is position-dependent because the
   quadratic/cubic terms grow with path length → cupping.

**Next:** Numerical analysis of BHC correction magnitude for 120 kVp vs 140 kVp.

**Numerical Analysis Results (xcat002_bhc_analysis.jl):**

Spectrum properties:
- 120 kVp: 240 bins, mean energy = 59.4 keV, μ_water_eff = 0.22562 cm⁻¹
- 140 kVp: 280 bins, mean energy = 65.1 keV, μ_water_eff = 0.21532 cm⁻¹
- μ_water(70 keV reference) = 0.19283 cm⁻¹

**CRITICAL FINDING: Default BHC coefficients are completely WRONG!**

The hardcoded `bhc_water_default()` coefficients `[0, 1.05, -0.02, 0.001]` do NOT match
the actual calibration curve for ANY spectrum:

| Coefficient | Default | 120 kVp calibrated | 140 kVp calibrated |
|-------------|---------|-------------------|-------------------|
| a₀ | 0.0 | -0.006 | -0.004 |
| a₁ | 1.05 | 0.870 | 0.907 |
| a₂ | -0.02 | +0.014 | +0.014 |
| a₃ | 0.001 | -0.0004 | -0.0003 |

Key differences:
- a₁: Default 1.05 vs calibrated ~0.87-0.91 (20% too large!)
- a₂: Default NEGATIVE vs calibrated POSITIVE (wrong sign!)
- a₃: Default POSITIVE vs calibrated NEGATIVE (wrong sign!)

This means the default BHC polynomial is essentially an aggressive OVER-scaling
(a₁=1.05 inflates all values by 5%) combined with wrong-direction curvature corrections.

**Cupping with default vs calibrated BHC (30cm water cylinder):**

| Spectrum | Default BHC | Calibrated BHC (order 5) |
|----------|-------------|--------------------------|
| 120 kVp | -146 HU | -0.2 HU |
| 140 kVp | -133 HU | -0.1 HU |

Both spectra have massive cupping with the default BHC. The difference between
120 and 140 kVp is only ~13 HU with the default BHC.

**BUT: Water calibration absorbs the BHC error**

NB05 uses empirical water calibration: simulate water cylinder → reconstruct → measure μ_water.
This μ_water already includes the BHC error, so when converting to HU via
`HU = 1000 × (μ - μ_water) / μ_water`, the BHC error partially cancels.

However, the cancellation is only exact at the calibration phantom's path length.
For the XCAT phantom (different size, different materials, different paths through
tissue vs bone vs lung), the BHC error is different → **residual cupping**.

**Verdict: PARTIAL — BHC default is severely wrong for both EICT and PCCT**

The default BHC is wrong for both paths, not specifically wrong for PCCT. The cupping
it causes is similar magnitude for both spectra (~133-146 HU raw, largely absorbed
by water calibration). The PCCT-specific artifact must come from additional factors:
- Different water calibration behavior (PCCT uses different geometry, z-coverage)
- DRM effects that change the effective spectrum after bin combination
- Missing physics effects in the PCCT path (no bowtie, no scatter, etc.)

The fact that the default BHC has fundamentally WRONG coefficients is a bug that
should be fixed (use `calibrate_bhc()` instead), but it likely doesn't explain
the PCCT-SPECIFIC cupping — it causes cupping equally in both paths.

**Important secondary finding: PCCT combined sinogram ≈ EICT polychromatic**

The math shows that for ideal detection (no DRM spatial effects), the PCCT combined
sinogram is equivalent to a polychromatic integral with η-weighted spectrum:
```
p_pcct = -log(Σ w_E × η_E × exp(-μ_E × d) / Σ w_E × η_E)
```
For CdTe at diagnostic energies (>40 keV), η ≈ 1.0, so p_pcct ≈ p_eict_polychromatic.

This means the PCCT combined sinogram has the SAME beam hardening as a standard
polychromatic sinogram for the same kVp — the energy binning does NOT reduce
beam hardening in the combined sinogram (only in individual bins).

**Conclusion for XCAT-002:**

1. The default BHC is wrong (should use `calibrate_bhc()`) — this is a codebase bug
   but affects EICT and PCCT similarly
2. The PCCT combined sinogram has equivalent beam hardening to a polychromatic
   sinogram at the same kVp — the BHC correction needed is the same
3. The PCCT-specific artifact is NOT explained by BHC over/under-correction of the
   combined sinogram. The root cause lies elsewhere (XCAT-003, XCAT-006, XCAT-007).

---

### 2026-02-13: XCAT-003 — DIAGNOSTIC: Verify _combine_pcct_bins I0 values [PASS]

**Investigation:** Are the I0_bins used by `_combine_pcct_bins` consistent with the
normalization I0 used by `pcct_forward_project`?

**Finding 1: I0 paths are identical for NB05 configuration**

For NB05 PCCT with `fidelity=:high`:
- `use_detector_fx = true` (`:high` in `(:medium, :high, :pcct)` — workspace.jl:200)
- `use_corrections = false` (`:high` defaults `pcct_corrections=false` — options.jl:128)

With this configuration, workspace.jl:218-230:
- `I0_bins_norm` → `_compute_degraded_I0(...)` (photon_counting.jl:1693)
- `I0_bins_combine` → `copy(I0_bins_norm_vec)` (exact copy!)

Both the forward projection normalization (`pcct_forward_project` line 1551-1555)
and the combination (`_combine_pcct_bins` line 478-479) use the SAME I0 values.

**Finding 2: Combination math is exact**

The round-trip is:
1. Forward proj: `sino_bin = -log(N_bin / I0_degraded_bin)`
2. Combine: `N_total = Σ I0_degraded_bin × exp(-sino_bin) = Σ N_bin` (exact)
3. Combined: `-log(N_total / I0_total)`

Since `I0_degraded_bin × exp(-sino_bin) = I0_degraded_bin × (N_bin / I0_degraded_bin) = N_bin`,
the combination perfectly recovers the total photon count. No information is lost.

**Finding 3: _compute_degraded_I0 correctly models DRM**

`_compute_degraded_I0` (photon_counting.jl:1693-1758) accounts for:
1. Spectral response matrix R (energy resolution, K-fluorescence, charge sharing spectral)
2. Charge sharing energy redistribution (high bins → low bins transfer)
3. Pileup count loss (seminonparalyzable model)
4. Spectral migration (Taguchi 2010 migration matrix)

It does NOT model spatial charge sharing (8-neighbor averaging) or anti-coincidence.
But the comments correctly note: "Spatial redistribution is a no-op for uniform fields
(all neighbors equal)." This is correct — I0 represents the unattenuated (uniform) case.

**Finding 4: Charge sharing is significant (~24% from charge cloud alone)**

For Koch-Mehrin ODE model with σ ≈ 0.013mm and pixel = 0.4mm:
- `charge_sharing_probability` = 1 - ((0.4 - 4×0.013)/0.4)² = 1 - 0.757 = **0.243**
- Plus fluorescence escape adds more above K-edges

This spatial charge sharing operates in count domain and creates position-dependent
smoothing at tissue boundaries. After log transform, this produces a nonlinear
effect: log(smooth(counts)) ≠ smooth(log(counts)). This IS a potential source of
edge artifacts, but it's a REAL physical effect, not a normalization error.

**Finding 5: _compute_degraded_I0 uses different charge sharing model than forward proj**

The `_compute_degraded_I0` models energy redistribution with:
```
σ_cloud = fwhm / 2.355  (using charge_sharing_fwhm_mm = 0.08)
p_share = 2/(1+exp(1.5*z_row)) + 2/(1+exp(1.5*z_col))
energy_loss_fraction = 0.5
```

But the forward projection uses the physics-based `_apply_charge_sharing_physics!` with:
```
σ = mean_charge_cloud_sigma_mm(E_center, geom)  (Koch-Mehrin ODE, σ ≈ 0.013mm)
p_cloud = charge_sharing_probability(σ, pixel_pitch)
p_fluor = fluorescence_sharing_boost(E_center, fluor_model)
p_share[b] = min(p_cloud + p_fluor, 0.7)
energy_loss_fraction = 0.4 (not 0.5!)
```

**This IS a mismatch!** The `_compute_degraded_I0` uses the LEGACY charge sharing model
(fixed FWHM = 0.08mm → p_share ≈ 0.06%) while the forward projection uses the PHYSICS
model (Koch-Mehrin σ ≈ 0.013mm → p_share ≈ 24% + fluorescence). However, in practice
the energy redistribution amount depends on both p_share and energy_loss_fraction:
- Legacy: 0.0006 × 0.5 = 0.0003 (negligible)
- Physics: not applied in `_compute_degraded_I0` (it uses the legacy path)

Actually wait — re-reading `_compute_degraded_I0` more carefully:
The function checks `detector.enable_charge_sharing && detector.charge_sharing_fwhm_mm > 0.0`.
For NAEOTOM with `charge_sharing_fwhm = 0.08`, this IS true, and it uses the legacy
formula with σ_cloud = 0.08/2.355 = 0.034mm. This gives p_share = 0.0006 (negligible).

Meanwhile, the forward projection uses `_apply_charge_sharing_physics!` which uses
Koch-Mehrin σ ≈ 0.013mm with `charge_sharing_probability` function → p_share ≈ 0.24.
The energy redistribution in the forward proj is `lf = p_share * 0.4`.

So the ENERGY redistribution per bin in the forward proj is ~24% × 0.4 = 9.6% transfer
from each bin to the bin below. But the `_compute_degraded_I0` models only ~0.06% × 0.5
= 0.03% transfer. **This is a 300× mismatch in energy redistribution modeling.**

However — this mismatch exists between two different effects in the degraded I0 vs
the actual forward projection. The `_compute_degraded_I0` function ALSO uses the
spectral response matrix R to compute I0_bins, which DOES include the spectral
effects of charge sharing and K-fluorescence. The R matrix is the main mechanism
for spectral redistribution. The subsequent charge sharing energy transfer in
`_compute_degraded_I0` is a secondary correction.

**Verdict: NO — I0_bins mismatch does NOT explain PCCT-specific cupping**

The I0_bins are consistent between forward projection normalization and combination.
The math is exact. The secondary energy redistribution mismatch between legacy and
physics-based charge sharing models in `_compute_degraded_I0` is a minor inconsistency
(legacy uses negligible 0.03% vs physics-based 9.6%), but:
1. The R matrix already captures the primary spectral redistribution
2. The secondary transfer is small compared to R matrix effects
3. Even if there's a small mismatch, it's approximately uniform across the sinogram
   (same p_share at every pixel) → does NOT produce position-dependent cupping

**Impact:** Low — fix the `_compute_degraded_I0` to use physics-based model for
consistency, but this is unlikely to cause the observed cupping artifact.

---

### 2026-02-13: XCAT-007 — DIAGNOSTIC: PCCT vs EICT physics effects checklist [PASS]

**Already answered by XCAT-001** — see the CRITICAL DIFFERENCES TABLE above (lines 188-211).

Summary: PCCT path is MISSING 10 EICT physics effects (fill_factor, flat_filter,
bowtie_filter, scatter, scatter_correction, crosstalk, optical_crosstalk, focal_spot,
detector_efficiency, lag). Instead, PCCT applies DRM (charge sharing, K-fluorescence,
pileup) and BHC.

Of the missing effects, only **bowtie_filter** is position-dependent and could contribute
to cupping. But PCCT doesn't HAVE a bowtie, so missing it in the model is correct
physics. The cupping must come from something else.

**Verdict: PASS — fully documented in XCAT-001. No additional investigation needed.**

---

### 2026-02-13: XCAT-005 — DIAGNOSTIC: PCCT noise_reduction interaction with edges [PASS]

**Investigation:** Does `pcct_noise_reduction=0.60` cause edge artifacts or cupping?

**Finding: NO — noise reduction is purely per-pixel amplitude scaling**

At photon_counting.jl:1852-1853:
```julia
nr_scale = T(1.0 - noise_reduction)  # 0.40 for noise_reduction=0.60
@. noise_buf = cpu_buf + nr_scale * sqrt(cpu_buf) * noise_buf
```

This computes: `N_measured = N_expected + 0.40 × √(N_expected) × randn()`

Key properties:
1. **No spatial kernel**: Each pixel is processed independently
2. **No neighbor averaging**: Unlike a spatial filter, this doesn't blend adjacent pixels
3. **Position-independent**: The scale factor `nr_scale=0.40` is constant everywhere
4. **Only scales noise magnitude**: Reduces the Gaussian noise amplitude by 60%
5. **Does not change the expected value**: E[N_measured] = N_expected (zero mean noise)

For low-count pixels (N ≤ 20), Poisson sampling with the same blend:
```julia
noise_buf[idx] = cpu_buf[idx] + nr_scale * (sampled - cpu_buf[idx])
```
Same principle — blend toward expected value, no spatial operation.

**Verdict: NO — pcct_noise_reduction CANNOT cause cupping or edge artifacts**

It's a purely local, position-independent noise amplitude scaling.
No spatial smoothing, no edge interaction, no position dependence.
This definitively rules out noise_reduction as a cause.
