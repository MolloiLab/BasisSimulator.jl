# XCAT Phantom Data

This directory contains data for the XCAT clinical workflow notebook (`05_xcat_full.jl`).

## Files

### Committed (in git)
- `Material_Spreadsheets/` — 3 XLSX files mapping XCAT material IDs to attenuation properties
  - `vmale_50_materials_heart_high_contrast.xlsx`
  - `vmale_50_materials_heart_low_contrast.xlsx`
  - `vmale_50_materials_heart_non_contrast.xlsx`

### Not committed (too large — 1.07 GB)
- `vmale_50_1600x1400x500_8bit_little_endian_act_1.bin`

## Setup

To run notebook 05, you need the XCAT binary file. Either:

### Option A: Symlink from BasisSimulatorVerification
```bash
ln -s /path/to/BasisSimulatorVerification/notebooks/phantom_export_xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin \
      verification/data/xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin
```

### Option B: Copy directly
```bash
cp /path/to/BasisSimulatorVerification/notebooks/phantom_export_xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin \
   verification/data/xcat/
```

The binary is a raw 8-bit volume: 1600 x 1400 x 500 voxels, little-endian activity map from the XCAT phantom (male, 50th percentile).
