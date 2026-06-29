# BasisSimulator.jl

[![CI](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://molloilab.github.io/BasisSimulator.jl/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20262003.svg)](https://doi.org/10.5281/zenodo.20262003)


GPU-portable polychromatic CT simulator in Julia. Runs on Metal, CUDA, ROCm, or CPU
via [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl).
Models energy-integrating and photon-counting detectors, single- and dual-kVp
acquisitions, and reconstructs with FBP (FDK), OS-PWLS iterative reconstruction, and material-basis VMI.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/MolloiLab/BasisSimulator.jl")
```

For GPU support, also add your backend:

```julia
Pkg.add("Metal")     # Apple Silicon
Pkg.add("CUDA")      # NVIDIA
Pkg.add("AMDGPU")    # AMD
```

## Quick example

```julia
import BasisSimulator as BS
using Metal  # or CUDA / AMDGPU; omit for CPU

phantom_cpu = BS.create_gammex_472(n_voxels=256)
phantom = BS.Phantom(MtlArray(phantom_cpu.mask),
                     phantom_cpu.materials,
                     phantom_cpu.voxel_size)

scanner = BS.Scanner(
    source_to_isocenter = 626.0,   # mm
    source_to_detector  = 1097.0,
    detector_rows       = 64,
    detector_cols       = 832,
    detector_row_size   = 0.625,
    detector_col_size   = 1.053,
)
protocol = BS.CTProtocol(kVp=120.0, mA=200.0, views=984)
sim_opts = BS.SimOptions(fidelity=:eict)
rec_opts = BS.ReconOptions(matrix_size=(512, 512, 64), fov_cm=35.0)

ws = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
BS.simulate!(ws, phantom, scanner, protocol, sim_opts, rec_opts)

ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, rec_opts.matrix_size)
hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, rec_opts.matrix_size));
    μ_water = BS.get_reference_μ_water(70.0),
)
```

### Forward projector: speed vs. accuracy

`SimOptions(; projector=:dd)` (default) uses the **distance-driven** ray tracer —
anti-aliased and robust in severe beam-hardened regions. Pass `projector=:siddon`
for **~3.5–5.5× faster** forward projection, at the cost of aliasing in those
regions (use when speed outranks accuracy). If you reconstruct **iteratively**,
pass the *same* projector to `create_hir_recon_workspace(; projector=…)` so the IR
system matrix matches the data — FDK reconstruction is unaffected.

## Documentation

Full API reference, physics overview, and worked examples: **docs link TBD** (Documenter site
in preparation). Until then, see docstrings via `?Function` in the Julia REPL.

## License

MIT. Core ray tracing ported from [TIGRE](https://github.com/CERN/TIGRE);
calibration workflow follows [CatSim/XCIST](https://github.com/xcist/main).
