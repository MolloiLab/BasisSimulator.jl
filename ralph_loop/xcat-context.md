# BasisSimulator.jl — XCAT PCCT Artifact Context

> Last updated: 2026-02-13

## The Problem

PCCT (photon-counting CT) FDK reconstructions from NB05 show cupping/edge artifacts at tissue boundaries that are NOT present in EICT (energy-integrating CT) reconstructions from the same XCAT phantom. Additionally, the same slice index shows slightly different anatomy between EICT and PCCT.

### Observed Artifacts (from screenshots)

1. **EICT 120 kVp FDK**: Clean reconstruction, no visible cupping
2. **PCCT Standard FDK**: Visible cupping/edge effects, especially at:
   - Lung/soft-tissue boundaries
   - Body contour edges
   - High-contrast interfaces
3. **Zoomed PCCT view**: Clear cupping/edge/aliasing-like artifact at tissue boundaries — subtle but definitely present

### Confirmed NOT the Cause

- **volume_fov threading**: Correct in both EICT and PCCT workspace paths
- **pixel_row_size**: Fixed in prior session (7 files corrected)
- **Reconstruction filter**: Both use StandardFilter (CatSim-matched apodized ramp)
- **Z-coverage mismatch**: EICT 8.0cm vs PCCT 5.12cm is expected behavior, not a bug

## NB05 Scanner Configurations

### EICT (GE Revolution Apex)
```
SID = 625.6mm, SDD = 1100mm
Detector: 128 rows × eict_det_cols, 0.625mm row × 0.6mm col (at isocenter)
z_cm = 128 * 0.625 / 10.0 = 8.0 cm
kVp = 120, rotation_time = 0.5s
```

### PCCT (Siemens NAEOTOM Alpha)
```
SID = 595mm, SDD = 1085.5mm
Detector: 128 rows × pcct_det_cols, 0.4mm row × 0.4mm col (at isocenter)
z_cm = 128 * 0.4 / 10.0 = 5.12 cm
kVp = 140, rotation_time = 0.25s
energy_thresholds = [20, 35, 55, 70] keV (4 bins)
detector_material = :cdte (direct-conversion)
pixel_mode = :standard (0.4mm binned from 0.2mm native)
```

### XCAT Phantom
```
Size: 800 × 700 × 250 (factor 2 downsampled from 1600×1400×500)
Voxel: (0.06, 0.06, 0.2) cm → FOV = 48 × 42 × 50 cm
Materials: ~30 XCAT tissue types from XLSX
Reconstruction: 512×512×128, fov_cm = 25.0
```

## PCCT Signal Chain Deep Dive

### Step 1: pcct_forward_project()

Located in `src/detector/photon_counting.jl`. This function:
1. For each energy bin defined by thresholds:
   a. Computes μ_volume at energy E for all materials
   b. Runs Siddon ray tracing to get line integrals L_E
   c. Computes photon counts per bin (uses spectral weighting + DRM)
   d. Normalizes: sino_bin = -log(N_bin / I0_bin_norm)
2. DRM effects applied:
   - Charge sharing (Koch-Mehrin ODE model)
   - K-fluorescence (5 K-lines per element, CdTe)
   - Charge collection efficiency (Hecht model + small-pixel weighting)
   - Pileup (Yang 2025 semi-nonparalyzable model)

### Step 2: _combine_pcct_bins()

Located in `src/api/driver.jl:467`. Combines energy-resolved bins:
```julia
# For each bin b:
N_total += I0_bins[b] * exp(-sino_bin[b])
# Then:
sino_combined = -log(N_total / I0_total)
```

Critical: I0_bins are computed using `_compute_bin_I0()` which accounts for:
- Spectrum weighting within each bin's energy range
- DRM effects (charge sharing probability matrix)
- Quantum efficiency per energy

If I0_bins don't match the normalization used in pcct_forward_project,
the combined sinogram has systematic errors.

### Step 3: BHC on Combined Sinogram

```julia
apply_bhc!(sino_combined, config.bhc)
# Uses bhc_water_default() coefficients: [0.0, 1.05, -0.02, 0.001]
# p_out = 0 + 1.05*p - 0.02*p^2 + 0.001*p^3
```

**Critical insight**: This same polynomial is applied to BOTH EICT and PCCT sinograms.
But the sinograms have different statistical properties:
- EICT: direct polychromatic integral (strong beam hardening)
- PCCT combined: sum of energy-resolved bins (less beam hardening because bins already separate spectrum)

If BHC over-corrects the PCCT sinogram, it produces inverted cupping.

### Step 4: Noise Application

PCCT noise is applied PER-BIN (not on the combined sinogram):
```julia
apply_pcct_noise!(pcct_sino, pcct_detector, protocol; ...)
```
With `noise_reduction=0.60` (60% noise reduction).

### Step 5: Recombination + BHC Again

After noise, bins are recombined and BHC applied again.
Note: BHC is applied TWICE — once on ideal combined, once on noisy combined.

## Top Hypotheses (Prioritized)

### Hypothesis 1: BHC Over-Correction (Most Likely)

The PCCT combined sinogram already has reduced beam hardening (energy-resolved bins
partially separate the spectral effects). Applying the same BHC polynomial designed
for polychromatic EICT sinograms may OVER-correct.

Over-correction mechanism:
- Short paths (thin tissue): BHC correction ≈ linear (1.05*p), minimal over-correction
- Long paths (thick tissue through center): BHC has quadratic/cubic terms (-0.02*p^2 + 0.001*p^3)
- If PCCT combined sinogram has LESS beam hardening than EICT, the quadratic term
  subtracts too much → center values are reduced → cupping

### Hypothesis 2: I0_bins Mismatch

If _combine_pcct_bins uses I0_bins that don't match the normalization in
pcct_forward_project, the combined sinogram has a path-length-dependent bias.

Mechanism:
- At low path lengths (air/lung): small sino_bin → exp(-sino_bin) ≈ 1 → bias is small
- At high path lengths (body center): large sino_bin → exp(-sino_bin) ≈ 0 → bias is amplified by I0 weighting

### Hypothesis 3: Missing Physics Effects

If EICT has fill_factor, flat_filter, bowtie, scatter, etc. but PCCT doesn't,
the PCCT sinogram has different attenuation values. Some of these effects are
position-dependent (bowtie) which could create cupping.

### Hypothesis 4: Noise Reduction Edge Effects

pcct_noise_reduction=0.60 may smooth sinogram values at boundaries,
creating partial-volume-like effects that appear as cupping after reconstruction.

## Key Files to Read

```
src/api/driver.jl                              — Both simulate!() paths
src/api/workspace.jl                           — PCCTWorkspace, EICTWorkspace
src/detector/photon_counting.jl                — pcct_forward_project, apply_pcct_noise!
src/detector/physics_pipeline.jl               — apply_physics_effects! (EICT)
src/correction/beam_hardening_correction.jl    — BHC coefficients and application
src/projection/polychromatic.jl                — EICT forward projection
verification/notebooks/05_xcat_full.jl         — NB05 with the artifact
```
