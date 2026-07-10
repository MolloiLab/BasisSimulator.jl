# BasisSimulator.jl

[![CI](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://molloilab.github.io/BasisSimulator.jl/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20262003.svg)](https://doi.org/10.5281/zenodo.20262003)


GPU-portable polychromatic CT simulator in Julia. Runs on Metal, CUDA, ROCm,
oneAPI, or CPU
via [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl).
Models energy-integrating and photon-counting detectors, single- and dual-kVp
acquisitions, and reconstructs with FBP (FDK), OS-PWLS iterative reconstruction, and material-basis VMI.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/MolloiLab/BasisSimulator.jl")
```

For portable device selection, add `GPUSelect` plus your backend:

```julia
Pkg.add(url="https://github.com/GroupTherapyOrg/GPUSelect.jl")
Pkg.add("Metal")     # Apple Silicon
Pkg.add("CUDA")      # NVIDIA
Pkg.add("AMDGPU")    # AMD
Pkg.add("oneAPI")    # Intel
```

## Quick example

```julia
import BasisSimulator as BS
import GPUSelect

AT = GPUSelect.Storage()  # MtlArray / CuArray / ROCArray / oneArray / Array
to_gpu(x) = AT(x)

phantom_cpu = BS.create_gammex_472(n_voxels=256)
phantom = BS.Phantom(to_gpu(phantom_cpu.mask),
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
BS.simulate!(ws, phantom, protocol, sim_opts)

bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner=scanner, geom=ws.geom)
sino_bhc = to_gpu(BS.apply_bhc_water(ws.sinogram, bhc))
ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, ws.geom, rec_opts.matrix_size)
hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, sino_bhc, ws.geom));
    μ_water = bhc.μ_water_ref,
)
```

### Forward projector: speed vs. accuracy

`SimOptions(; projector=:dd_fast)` (default) uses the single-pass
**distance-driven** projector — anti-aliased and robust in severe
beam-hardened regions. Pass `projector=:siddon`
for **~3.5–5.5× faster** forward projection, at the cost of aliasing in those
regions (use when speed outranks accuracy). If you reconstruct **iteratively**,
pass the *same* projector to `create_hir_recon_workspace(; projector=…)` so the IR
system matrix matches the data — FDK reconstruction is unaffected.

## Documentation

Full API reference, getting-started guide, and eleven worked-example notebooks:
**<https://molloilab.github.io/BasisSimulator.jl/>**. Docstrings are also
available via `?Function` in the Julia REPL.

## License

MIT. Core ray tracing ported from [TIGRE](https://github.com/CERN/TIGRE);
calibration workflow follows [CatSim/XCIST](https://github.com/xcist/main).
