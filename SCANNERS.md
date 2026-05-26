# Scanner Specifications Reference

Reference parameters for clinical CT scanners used in BasisSimulator.jl verification.
These were previously implemented as factory functions and have been moved here for reference.

## GE Revolution Apex Elite (Energy-Integrating)

**FDA 510(k):** K213715

| Parameter | Value |
|-----------|-------|
| SID | 626.0 mm |
| SDD | 1097.0 mm |
| Detector Rows × Cols | 256 × 832 |
| Row Size (iso) | 0.625 mm |
| Col Size (iso) | 1.053 mm |
| Z Coverage | 160 mm |
| Detector Material | Lumex (garnet scintillator) |
| Detector Depth | 3.0 mm |
| Fill Factor | 0.90 |
| Focal Spot (small) | 1.0 × 0.7 mm |
| Focal Spot (large) | 1.6 × 1.2 mm |
| Target Angle | 10° |
| Flat Filter | 2.5 mm Al |
| Bowtie | large_body |
| Supported kVp | 70, 80, 100, 120, 140 |
| Max mA | 1300 |
| Min Rotation | 0.23 s |
| Bore | 800 mm |
| Scan Diameter | 500 mm |
| Detection Gain | 15.0 e⁻/keV |
| Electronic Noise | 5000 e⁻ |

## Siemens NAEOTOM Alpha (Photon-Counting)

**FDA 510(k):** K201501

| Parameter | Standard (2×2) | UHR (1×1) |
|-----------|---------------|-----------|
| SID | 595.0 mm | 595.0 mm |
| SDD | 1085.5 mm | 1085.5 mm |
| Subpixel (iso) | 0.300 × 0.176 mm | 0.150 × 0.176 mm |
| Binning Factor | 2 | 1 |
| **Pixel Size at iso (marketed)** | **0.4 mm row × 0.5 mm col** | **0.2 mm row × 0.25 mm col** |
| Detector Rows | 144 | 120 |
| **Z-coverage at iso (marketed)** | **57.6 mm** (144 × 0.4 mm) | **24.0 mm** (120 × 0.2 mm) |
| Detector Material | CdTe | CdTe |
| Detector Depth | 1.6 mm | 1.6 mm |
| Fill Factor | 0.95 | 0.95 |
| Energy Thresholds | 20, 35, 55, 70 keV | 20, 35, 55, 70 keV |
| Energy Resolution | 10 keV FWHM | 10 keV FWHM |
| Charge Sharing FWHM | 0.08 mm | 0.08 mm |
| Dead Time | 5.0 ns | 5.0 ns |
| Focal Spot | 1.0 × 1.0 mm | 0.4 × 0.5 mm |
| Target Angle | 7° | 7° |
| Flat Filter | 2.5 mm Al | 2.5 mm Al |
| Supported kVp | 70, 90, 120, 140 | 70, 90, 120, 140 |
| Min Rotation | 0.25 s | 0.25 s |
| Bore | 820 mm | 820 mm |
| Scan Diameter | 500 mm | 500 mm |

**Dual-source geometry**: Two source/detector pairs at ~90° angular offset around the
gantry — *both cover the same Z extent*. The second source is for temporal resolution
(66 ms quarter-rotation imaging) and high-pitch (3.2) helical Flash mode, **not** for
extended Z coverage. Max single-rotation axial Z coverage is **57.6 mm** in standard
mode; Flash mode achieves ~120 mm of *scan range* in ~160 ms via table translation, not
a wider detector.

For simulator scenarios that want to approximate Flash-mode coverage as a single axial
rotation (training-data generation, plaque-volume scans), use the `extended_collimation`
kwarg on `CTGeometry(scanner; collimation_mm=..., extended_collimation=true)` — see that
function's docstring.

**Note on row pitch**: The 0.4 mm marketed pitch is what Siemens publishes (Petersilka
et al. 2021; vendor planning docs) and is the canonical number used throughout the
literature. A strict geometric derivation from the 0.176 mm subpixel and 1.825×
magnification would give 0.176 × 2 = 0.352 mm — a ~13% underestimate. Internally
BasisSim adopts the published 0.4 mm so `detector_rows × detector_row_size` matches
the 57.6 mm spec.

## Canon Aquilion ONE (Legacy Reference)

| Parameter | Value |
|-----------|-------|
| SAD | 600 mm |
| SDD | 1000 mm |
| Pixel Pitch (iso) | 0.5 mm |
| Detector Type | Energy-integrating |
