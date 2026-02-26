# BasisSimulator.jl — Noise Diagnosis Context

> Last updated: 2026-02-13

## The Problem

BasisSimulator.jl produces **~2x more noise** than CatSim/XCIST when simulating identical CT protocols. This manifests as:
- CNR in verification notebook 01 is ~2x worse than CatSim
- All downstream image quality metrics (NPS, MTF, SNR) are affected
- The 2x factor is consistent across different insert types (calcium, iodine)

## What Was Compared

**Notebook 01** (`verification/notebooks/01_single_kvp_verification.jl`) directly compares BasisSimulator vs CatSim using:
- Scanner: SID=540mm, SDD=950mm, 900 detector columns, 16 rows
- Protocol: 120 kVp, 200 mA, 984 views, 1.0s rotation
- Phantom: Gammex 472 (water body + calcium/iodine inserts)
- Reconstruction: FDK with Ram-Lak, 512x512x9, 350mm FOV
- Physics: fidelity=:high (all 13 effects except DAS)

## Signal Chain Overview

```
X-ray Source (spectrum.jl)
    │ load_spectrum(120) → energies, weights
    │ downsample_spectrum(energies, weights, 15)
    ▼
Polychromatic Forward Projection (polychromatic.jl)
    │ For each energy bin:
    │   μ_volume = material_attenuation(mask, materials, E)
    │   L_e = siddon_project(μ_volume, geom)
    │   I_total += w_e × exp(-L_e)
    │ sinogram = -log(I_total)
    ▼
Physics Effects Pipeline (physics_pipeline.jl) [noise=nothing in driver path!]
    │ 1. Heel effect (intensity domain)
    │ 2. Fill factor (sinogram domain, adds ~0.105 to p)
    │ 3. Flat filter (adds attenuation from 2.5mm Al)
    │ 4. Scatter + scatter correction
    │ 5. Bowtie filter
    │ 6. Crosstalk + optical crosstalk
    │ 7. Focal spot blur
    │ 8. [NOISE SKIPPED — handled by sim_detect]
    │ 9. Lag
    │ 10. BHC
    ▼
Detector Noise (detector_noise.jl via sim_detect)
    │ I0 = compute_detector_I0(geom, protocol)
    │   = flux × mA × (rot_time/views) × pixel_area × (1000/SDD)²
    │   NOTE: Missing η_det (quantum efficiency)!
    │ λ = I0 × exp(-sinogram)
    │ λ_noisy = λ + √λ × randn()
    │ sinogram_noisy = -log(λ_noisy / I0)
    ▼
FDK Reconstruction (fdk.jl)
    │ 1. Cosine weighting: p × D/√(D²+u²+v²)
    │ 2. Ramp filtering: convolution with h[n]
    │ 3. Backprojection: integral over angles with D²/(D-s)² weight
    ▼
HU Conversion
    │ μ_water = empirical (measured from water phantom scan)
    │ HU = 1000 × (μ - μ_water) / μ_water
    ▼
Noise Measurement
    │ σ_HU = std(water_ROI)
    │ CNR = |HU_insert - HU_water| / σ_HU
```

## Top Hypotheses (Prioritized)

### Tier 1: Most Likely (Each Could Cause 2x Alone)

1. **FDK Normalization Factor (NOISE-004, NOISE-009)**
   - If using 2π/N instead of π/N (or vice versa), reconstruction is 2x off
   - This is a COMMON porting error from TIGRE
   - Effect: EXACTLY 2x noise in HU

2. **Ramp Filter Pixel Size Units (NOISE-008)**
   - Ramp kernel h[n] = 1/(4Δ²) at n=0
   - If Δ is in wrong units (cm vs mm), the kernel is 100x wrong
   - Effect: Up to 10x wrong (but usually caught by visual inspection)

3. **μ_water Calibration Error (NOISE-011)**
   - If empirical μ_water is 2x too small, σ_HU is 2x too large
   - Could happen if reconstruction values are in wrong units
   - Effect: EXACTLY 2x noise in HU (but HU values would also be 2x)

### Tier 2: Compound Effects (Each 1.2-1.5x)

4. **Spectrum Double-Filtering (NOISE-002)**
   - Spectrum .dat files may already include Al filtration
   - Physics pipeline adds another 2.5mm Al via flat_filter
   - Effect: ~1.3-1.5x more noise (from reduced photon flux)

5. **I0 Missing η_det (Already Identified)**
   - compute_detector_I0() at line 742 doesn't include η_det = 0.85
   - mA_to_I0() at line 283 DOES include η_det
   - Effect: I0 is ~15% too high → noise is ~7% too LOW (wrong direction!)
   - Actually makes noise LESS, not more

6. **Physics Effects Before Noise (NOISE-003)**
   - Fill factor adds ~0.105 to sinogram values
   - Flat filter adds ~0.3-0.5 to sinogram values
   - Combined: λ is reduced by ~35-50%
   - Effect: ~1.2-1.4x more noise

### Tier 3: Possible But Less Likely

7. **CatSim Parameter Mismatch (NOISE-005)**
   - Different detector pixel interpretation
   - Different bowtie filter
   - Different spectrum source
   - Effect: Variable, depends on specific mismatch

8. **Detector Blur Before Noise (NOISE-007)**
   - Blur BEFORE noise is physically wrong (noise happens at photon absorption)
   - But the effect on total noise magnitude is small
   - Effect: Minimal (changes NPS shape, not total noise)

## Architecture Notes

### Two Noise Code Paths (CRITICAL TO UNDERSTAND)

**Path A: Driver API** (used by notebooks via simulate())
```julia
# build_physics_config() sets noise=nothing (line 1310)
# sim_detect() adds noise using compute_detector_I0() (no η_det)
# RESULT: Noise applied once, I0 from geometry
```

**Path B: Direct API** (used when calling forward_project directly)
```julia
# full_physics_config() includes noise with I0=1e6 (hardcoded!)
# RESULT: Noise applied once, but I0 may be wrong for the protocol
```

**Path C: Explicit mixing** (possible user error)
```julia
# User calls forward_project with physics=full_physics_config() (has noise)
# THEN calls sim_detect() again
# RESULT: DOUBLE NOISE! But this is unlikely in the notebooks.
```

### Spectrum Files

Located in `data/spectra/` (or similar path)
Format: `.dat` files with two columns (energy_keV, photon_fluence)
Key question: Does the filename include "_filt" (pre-filtered)?

CatSim uses `tungsten_tar7.0_120_filt.dat` — the "_filt" suffix strongly suggests
the spectrum is pre-filtered through inherent aluminum.

If BasisSimulator loads the SAME pre-filtered spectrum AND applies flat_filter_al(2.5),
the beam is DOUBLE-FILTERED, reducing photon flux and increasing noise.

## Debugging Tools

```julia
# Check I0 value for notebook 01 parameters
using BasisSimulator
scanner = Scanner(540.0, 950.0, 16, 900, 1.0, 1.0)  # approximate
geom = CTGeometry(scanner; n_angles=984, fov_cm=35.0)
protocol = CTProtocol(mA=200, kVp=120, views=984, rotation_time=1.0)
I0 = compute_detector_I0(geom, protocol)
println("I0 = $I0")

# Check spectrum weights
energies, weights = load_spectrum(120)
println("Sum of weights: $(sum(weights))")
e_ds, w_ds = downsample_spectrum(energies, weights, 15)
println("Sum after downsample: $(sum(w_ds))")
```
