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

## Data Requirements

### Gammex 472 (included)
Small phantom data files in `data/gammex_basic/` (~2 MB). Committed directly.

### XCAT Phantom (notebook 05 only)
The XCAT binary (`vmale_50_1600x1400x500_8bit_little_endian_act_1.bin`, 1.07 GB) is too large for git. See `data/xcat/README.md` for download/symlink instructions. The Material Spreadsheets (~27 KB) are committed directly.

### No External Data Needed
Notebooks 02, 03, and 04 generate phantoms at runtime using `BasisSimulator.create_gammex_472()`.

## Generated Output

- `figures/` — Saved figures (gitignored, regenerated on run)
- `catsim_work/` — CatSim working directory (gitignored, notebook 01 only)
