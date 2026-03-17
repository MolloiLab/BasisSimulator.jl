---
title: "Fix NB07 Brain Perfusion — BHC & Cupping Correction"
date: 2026-03-16
status: draft
---

# Fix NB07 Brain Perfusion Output

## Problem

NB07 (brain perfusion CT, `phantom-loading` branch) produces severely corrupted output:
- **White background** outside FOV circle, values reaching +4000 HU at corners
- **Extreme values** (max 1.6M HU) concentrated at top/bottom slices
- **Iodine enhancement invisible** — buried in artifact noise
- Air at -153 HU instead of -1000 HU

## Root Causes (Diagnosed)

### 1. Cupping Correction Extrapolation

`apply_radial_cupping_correction!` fits a `c₀ + c₁r² + c₂r⁴` polynomial to water-like voxels
([-100, 80] HU). For the Gammex phantom, water fills the entire FOV (r=0–17 cm) — the polynomial
is well-conditioned. For the brain, water-like tissue exists only at r=2–8 cm inside the skull.
The r⁴ polynomial extrapolates catastrophically to r=20–28 cm (FOV edge and corners).

**Evidence:** Radial profile shows monotonic increase from -878 HU at r=12 cm to +4026 HU at r=28 cm.

### 2. Sinogram-Domain Two-Material BHC + Wide Cone

`apply_bhc_two_material` performs an intermediate FDK reconstruction (line 690) to segment bone.
With 160 mm collimation (256 rows, 7.3° cone half-angle), FDK produces significant cone-beam
artifacts at top/bottom slices. These artifacts corrupt bone segmentation, which cascades through
the forward-projection → subtraction → energy-correction chain.

**Evidence:** Extreme voxels (>10,000 HU) exist only in top 30 slices. Slice 11 has 11,514 voxels
above 10,000 HU; center slice 81 has zero.

## Fix

### Change 1: Remove cupping correction from NB07

Cupping is a beam-hardening artifact. With proper BHC, it's already corrected. Clinical brain CT
does not apply empirical radial cupping correction on top of BHC. The Gammex needs it because it's
a large water cylinder where residual cupping after BHC is the dominant artifact. Brain tissue
doesn't need it.

### Change 2: Replace sinogram-domain two-material BHC with water-only polynomial

Instead of `apply_bhc_two_material` (deprecated, cone-beam sensitive), apply only the water
polynomial BHC: `apply_bhc!(sino, bhc_model.water_bhc)`. This corrects beam hardening assuming
water-only composition — appropriate for the soft-tissue-dominated brain.

Keep the image-domain BHC (`apply_bhc_image_domain`) for residual bone correction. It operates on
the final reconstruction and is robust to cone-beam artifacts because:
- Bone segmentation uses the final (custom-filtered) image, not an intermediate ramp-filtered one
- The error image is scaled by 0.2 before subtraction
- It's perturbative, not decomposition-based

### Change 3: Keep everything else

- Water polynomial BHC: yes (corrects polychromatic beam hardening)
- Image-domain BHC: yes (corrects residual bone artifacts, scale_factor=0.2)
- Custom FBP filter: yes (matches NB00 clinical kernel)
- Noise floor (28 HU): yes (models scanner system noise)
- μ_water from BHC model: yes (spectrum-analytical, consistent with BHC calibration)

## Files Changed

| File | Change |
|------|--------|
| `verification/notebooks/07_brain_perfusion.jl` | Replace `apply_bhc_two_material` with `apply_bhc!` water polynomial; remove `apply_radial_cupping_correction!` |

## Validation

Run single t=20s timepoint with fixes, compare:
- Center brain ROI mean should be 30–45 HU (gray matter)
- Air should be near -1000 HU
- No extreme values (max < 2000 HU)
- Iodine enhancement visible in contrast subtraction
