# BasisSimulator Verification Suite

Pluto notebooks that verify BasisSimulator.jl against CatSim/XCIST reference simulations, validate physics pipelines, and demonstrate clinical workflows.

## Setup

```bash
cd verification/
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs all Julia dependencies and sets up the Python environment (gecatsim via CondaPkg).

## Running Notebooks

```bash
julia --project=. -e 'using Pluto; Pluto.run()'
```

Then open any notebook from the Pluto interface.

## Notebooks

| # | File | Description |
|---|------|-------------|
| 01 | `01_single_kvp_verification.jl` | Single-energy verification against CatSim (requires gecatsim) |
| 02 | `02_multi_dose_and_iterative_reconstruction.jl` | Multi-protocol dose comparison and Hybrid IR |
| 03 | `03_dual_kvp_vmi_verification.jl` | Dual-energy VMI validation against NIST |
| 04 | `04_pcct_demonstration.jl` | PCCT detector physics and spectral imaging |
| 05 | `05_xcat_full.jl` | Full XCAT clinical workflow (3 scanners) |
| 06 | `06_ge_apex_catsim_baseline.jl` | GE Revolution Apex — CatSim baseline reference across 6 protocols (3 dose levels × 3 kVp) |
| 07 | `07_brain_perfusion.jl` | Dynamic contrast-enhanced brain perfusion CT using P1/P2 XCAT phantom (6 time points, FDK + Hybrid IR) |

## Data Requirements

### Gammex 472 (included)
Small phantom data files in `data/gammex_basic/` (~2 MB). Committed directly.

### XCAT Phantom (notebook 05 only)
The XCAT binary (`vmale_50_1600x1400x500_8bit_little_endian_act_1.bin`, 1.07 GB) is too large for git. See `data/xcat/README.md` for download/symlink instructions. The Material Spreadsheets (~27 KB) are committed directly.

### Brain Perfusion Phantom (notebook 07 only)
The P1/P2 XCAT brain phantom raw files (`*_RAW_400_400_400.raw`, ~320 MB each) and two MAT files are not tracked in git.

Copy the following files from the group share drive into `verification/data/brain_perfusion/`:

**Share drive path:** `/Molloilab/Wenbo/brain phantom/Caedin Files/dynamic_brain_phantom`

| File | Size |
|------|------|
| `P1_brain_all_2020_RAW_400_400_400.raw` | ~320 MB |
| `P2_brain_all_2020_RAW_400_400_400.raw` | ~320 MB |
| `structure_info.mat` | small |
| `iodine_mass_data.mat` | small |

The voxelization lookup tables (`P1_voxelize_table.txt`, `P2_vozelize_table.txt`) and the reference contrast curve (`data/brain_perfusion/reference/Dynamic_Contrast.jl`) are committed directly.

> **Citation:** Sarah E. Divel et al. (2021), *Med. Phys.*, https://doi.org/10.1002/mp.14887

### No External Data Needed
Notebooks 02, 03, and 04 generate phantoms at runtime using `BasisSimulator.create_gammex_472()`.

## Generated Output

- `figures/` — Saved figures (gitignored, regenerated on run)
- `catsim_work/` — CatSim working directory (gitignored, notebook 01 only)
