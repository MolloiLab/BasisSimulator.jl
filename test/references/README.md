# Scanner Reference Library

Ground truth rod measurements for validating BasisSimulator reconstructions.

## Purpose

BasisSimulator aims to match the behavior of specific clinical CT scanners. This
directory holds **fixed reference data** — rod-by-rod HU measurements from the
Gammex 472 phantom — captured from two sources:

- **`catsim/`** — XCIST/CatSim simulation modeled after the GE Revolution Apex geometry
- **Scanner folders** — Real clinical acquisitions, preprocessed into PNGs

BasisSimulator results are compared **against** these references. The references
themselves do not change when BasisSimulator is tuned.

## Directory Structure

```
references/
  catsim/                          # CatSim baseline (GE Revolution Apex geometry)
    {protocol}/                    # e.g. 120kVp_10mGy
      recon.png                    # Middle-slice reconstruction (grayscale)
      mask.png                     # Segmented ROI mask overlay
      rod_measurements.csv         # Per-rod: mean HU, noise (σ), CNR
      nps.csv                      # Radial NPS profile (freq vs power)
      mtf.csv                      # MTF curve (freq vs modulation)
      metadata.txt                 # Protocol, geometry, bg noise, MTF50/MTF10
    summary.csv                    # Master: all protocols × 16 rods × metrics

  ge_revolution_apex/              # Clinical scanner references
    {protocol}/
      recon.png                    # Preprocessed DICOM → PNG (middle slice)
      mask.png                     # Corresponding segmentation mask
      rod_measurements.csv         # Per-rod: mean HU, noise (σ), CNR
      nps.csv                      # Radial NPS profile
      mtf.csv                      # MTF curve
      metadata.txt

  siemens_naeotom_alpha/
    ...
  siemens_force/
    ...
  canon_aquilion_one/
    ...
  philips_iqon/
    ...
```

### Key Points

- **No raw DICOMs** — clinical scans are preprocessed into PNGs before storage
- **Comprehensive metrics** — each protocol folder has rod HU + CNR, NPS curve,
  MTF curve, and scan-level summary (bg noise, MTF50/MTF10, NPS peak)
- **CatSim is one folder** — it models the GE Revolution Apex scanner geometry
  via XCIST/CatSim, serving as the simulated ground truth baseline
- **BasisSimulator results are never stored here** — references are fixed baselines;
  BasisSim is what changes during tuning and gets compared against these

## Scanners

| Directory                | Scanner                  | Status  |
|--------------------------|--------------------------|---------|
| `catsim/`                | XCIST/CatSim (GE Revolution Apex geometry) | Planned |
| `ge_revolution_apex/`    | GE Revolution Apex       | Planned |
| `siemens_naeotom_alpha/` | Siemens NAEOTOM Alpha    | Planned |
| `siemens_force/`         | Siemens SOMATOM Force    | Planned |
| `canon_aquilion_one/`    | Canon Aquilion ONE       | Planned |
| `philips_iqon/`          | Philips IQon Spectral CT | Planned |

## Protocol Naming Convention

```
{kVp}kVp_{dose}mGy          # e.g. 120kVp_10mGy
{kVp}kVp_{dose}mGy_{kernel}  # e.g. 120kVp_10mGy_standard
```

## CSV Formats

### `rod_measurements.csv` — per-rod metrics (16 rows)

| Column     | Description                           |
|------------|---------------------------------------|
| `name`     | Material name (e.g. "Ca 200", "I 5.0")|
| `label`    | Phantom label (10-16=Ca, 20-26=I)    |
| `ring`     | `outer` (Ca) or `inner` (I)          |
| `angle_deg`| Angular position (degrees)            |
| `mean_hu`  | Mean HU within ROI                    |
| `std_hu`   | Noise (σ) within ROI                  |
| `cnr`      | Contrast-to-noise ratio vs background |
| `n_pixels` | Number of pixels in ROI               |
| `cx`       | ROI center x (pixels)                 |
| `cy`       | ROI center y (pixels)                 |

### `nps.csv` — radial noise power spectrum

| Column            | Description                      |
|-------------------|----------------------------------|
| `freq_lp_per_mm`  | Spatial frequency (lp/mm)        |
| `nps_hu2_mm2`     | Noise power (HU² mm²)           |

### `mtf.csv` — modulation transfer function

| Column            | Description                      |
|-------------------|----------------------------------|
| `freq_lp_per_mm`  | Spatial frequency (lp/mm)        |
| `mtf`             | Modulation factor (0–1)          |

### `summary.csv` — master table (all protocols × 16 rods)

| Column            | Description                      |
|-------------------|----------------------------------|
| `protocol`        | Protocol name                    |
| `kvp`             | Tube voltage                     |
| `name`            | Rod material name                |
| `label`           | Phantom label                    |
| `ring`            | `outer` or `inner`               |
| `mean_hu`         | Mean HU                          |
| `std_hu`          | Noise (σ)                        |
| `cnr`             | Contrast-to-noise ratio          |
| `bg_noise`        | Background noise (σ) for this protocol |
| `mtf50_lp_per_mm` | Spatial frequency at 50% MTF     |
| `mtf10_lp_per_mm` | Spatial frequency at 10% MTF     |

## Validation Workflow

```julia
using BasisSimulator

# 1. Run BasisSimulator with current pipeline
recon_hu = my_reconstruction_pipeline(sinogram, geom)

# 2. Segment rods from BasisSim result
_, basis_rods, _ = segment_gammex_rods(recon_hu[:,:,mid_z]; fov_cm=35.0)

# 3. Load reference (CatSim or any clinical scanner)
ref_rods = load_rod_reference("test/references/catsim/120kVp_10mGy/rod_measurements.csv")

# 4. Compare HU accuracy
for (ref, basis) in zip(ref_rods, basis_rods)
    @test abs(basis.mean_hu - ref.mean_hu) < 10.0  # HU tolerance
end

# 5. Compare noise (from background ROI)
@test abs(basis_bg_noise - ref_bg_noise) / ref_bg_noise < 0.2  # 20% tolerance

# 6. Compare spatial resolution (load MTF curve)
ref_mtf = load_csv("test/references/catsim/120kVp_10mGy/mtf.csv")
# Compare MTF50 values

# 7. Compare noise texture (load NPS curve)
ref_nps = load_csv("test/references/catsim/120kVp_10mGy/nps.csv")
# Compare NPS peak frequency and shape
```

## Generating References

### CatSim references

Run notebook `verification/notebooks/06_ge_apex_catsim_baseline.jl`:
1. Enable all 6 CatSim scan cells
2. Enable the auto-save cell in section 9
3. References are saved to `test/references/catsim/{protocol}/`
4. Master summary saved to `test/references/catsim/summary.csv`

### Clinical references

1. Preprocess DICOM reconstruction → PNG (middle slice, grayscale HU)
2. Run `segment_gammex_rods(hu_slice; fov_cm=...)` on the slice
3. Compute NPS and MTF from the reconstruction
4. Save all artifacts to `test/references/{scanner}/{protocol}/`
