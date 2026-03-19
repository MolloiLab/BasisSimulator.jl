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
| Native Dexel (detector) | 0.275 × 0.322 mm | 0.275 × 0.322 mm |
| Binning Factor | 2 | 1 |
| Pixel Size (iso) | 0.302 × 0.353 mm | 0.151 × 0.176 mm |
| Detector Rows | 144 | 120 |
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

## Canon Aquilion ONE (Legacy Reference)

| Parameter | Value |
|-----------|-------|
| SAD | 600 mm |
| SDD | 1000 mm |
| Pixel Pitch (iso) | 0.5 mm |
| Detector Type | Energy-integrating |
