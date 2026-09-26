# BasisSimulator.jl

[![CI](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://molloilab.github.io/BasisSimulator.jl/)
[![SoftwareX DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.102910-blue)](https://doi.org/10.1016/j.softx.2026.102910)
[![Zenodo archive](https://zenodo.org/badge/DOI/10.5281/zenodo.20262003.svg)](https://doi.org/10.5281/zenodo.20262003)

Polychromatic CT simulation and reconstruction in Julia, on any GPU or the CPU.

BasisSimulator.jl models energy-integrating and photon-counting scanners from the source spectrum
to the detector counts, and reconstructs what they measure: cone-beam FDK, helical WFBP, hybrid
iterative reconstruction, and virtual monoenergetic images from dual-kVp, dual-source and
photon-counting acquisitions, with the CTDIvol and DLP of every scan. The kernels are written once
against [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) and run on CUDA,
Metal, ROCm, oneAPI or the CPU.

## Install

BasisSimulator.jl requires Julia 1.12.

```julia
using Pkg
Pkg.add("BasisSimulator")
Pkg.add("GPUSelect")   # picks the device array type
Pkg.add("CUDA")        # or Metal, AMDGPU, oneAPI; nothing for the CPU
```

## Quick example

A Gammex 472 phantom on a model of the GE Revolution Apex Elite, reconstructed to Hounsfield units:

```julia
import BasisSimulator as BS
import GPUSelect

AT = GPUSelect.Storage()      # CuArray, MtlArray, ROCArray or oneArray; Array on the CPU
to_gpu(x) = AT(x)

cpu = BS.create_gammex_472(n_voxels = 256, n_slices = 4, z_cm = 1.0)
phantom = BS.Phantom(to_gpu(cpu.mask), cpu.materials, cpu.voxel_size, cpu.origin, cpu.extent)

scanner  = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0,
                          detector_rows = 32, detector_cols = 834,
                          detector_row_size = 0.625, detector_col_size = 0.6)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = 500, collimation_mm = 5.0)
sim_opts = BS.SimOptions(seed = 42)
rec_opts = BS.ReconOptions(matrix_size = (512, 512, 4), fov_cm = 35.0, z_cm = 0.5)

ws     = BS.create_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
result = BS.simulate!(ws, phantom, protocol, sim_opts)
result.dose                   # CTDIvol and DLP of this acquisition, from the simulated beam

bhc  = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom)
sino = to_gpu(BS.apply_bhc_water(ws.sinogram, bhc))
fdk  = BS.create_fdk_recon_workspace(sino, ws.geom, rec_opts.matrix_size)
hu   = BS.to_hounsfield(Array(BS.reconstruct!(fdk, sino, ws.geom)); μ_water = bhc.μ_water_ref)

# hybrid iterative reconstruction: one strength dial, 0 (FBP) to 100
hir    = BS.create_hir_recon_workspace(sino, ws.geom, rec_opts.matrix_size; strength = 60)
hu_hir = BS.to_hounsfield(Array(BS.reconstruct!(hir, sino, ws.geom)); μ_water = bhc.μ_water_ref)
```

A protocol with a `pitch` is helical, and `reconstruct!` switches to rebinned WFBP by itself. The
volume walk does not depend on the spectrum, so a study at several tube voltages walks once:

```julia
paths = BS.material_paths(ws, phantom)          # one walk, reused by every voltage below
for kvp in (80, 100, 120, 140)
    p = BS.CTProtocol(kVp = kvp, mA = 200.0, views = 500, collimation_mm = 5.0)
    w = BS.create_workspace(scanner, p, sim_opts, rec_opts, phantom)
    BS.simulate!(w, phantom, p, sim_opts; paths)
end
```

## Photon counting to virtual monoenergetic images

```julia
scanner_pc = BS.PCCTScanner(
    source_to_isocenter = 610.0, source_to_detector = 1113.0,
    detector_rows = 32, detector_cols = 1200,          # 1200 × 0.3 mm covers the 33 cm phantom
    detector_row_size = 0.35, detector_col_size = 0.3,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0],     # four counting bins, keV
    energy_resolution = 10.0, charge_sharing_fwhm = 0.08,
    dead_time_ns = 5.0,                               # pile-up is modelled once there is a dead time
    pileup_correction = true, scatter_correction = true,
)
ws_pc  = BS.create_workspace(scanner_pc, protocol, sim_opts, rec_opts, phantom)
res_pc = BS.simulate!(ws_pc, phantom, protocol, sim_opts)

channels = [Array(b) for b in res_pc.pcct_sino.bins]    # one corrected log sinogram per bin
basis    = BS.spectral_basis(ws_pc; I0 = res_pc.I0_bins) # the response the simulation applied

vmi = BS.vmi_pipeline(; channels, basis, geom = ws_pc.geom, to_backend = to_gpu,
                      matrix_size = rec_opts.matrix_size)
vmi.vmis                      # (512, 512, 4, 4) in HU at 40, 70, 100 and 140 keV

# with the SpectralHYPR denoiser, on the counts and on the reconstructed basis pair
vmi_denoised = BS.vmi_pipeline(; channels, basis, geom = ws_pc.geom, to_backend = to_gpu,
                               matrix_size = rec_opts.matrix_size,
                               denoiser = BS.SpectralHYPR())
```

The same `vmi_pipeline` takes rapid kVp-switching and dual-source pairs through
`BS.spectral_basis_from_acquisitions`; the
[getting-started guide](https://molloilab.github.io/BasisSimulator.jl/getting-started/#dual-energy)
shows it.

## Documentation

**<https://molloilab.github.io/BasisSimulator.jl/>**: a getting-started guide, twelve worked-example
notebooks (the five-struct API, XCAT anatomy, dual-kVp and photon-counting VMI, the Siemens SOMATOM
Force and Definition Flash, helical scanning, metal artifacts, a CatSim comparison), the
[scanner parameter sets](https://molloilab.github.io/BasisSimulator.jl/scanners/), and the API
reference. Docstrings are also available via `?BS.simulate!` in the REPL.

## Citation

If you use BasisSimulator.jl in your work, please cite the SoftwareX article:

> Black D, Khodajou-Chokami H, Molloi S. BasisSimulator.jl: Open-source
> polychromatic CT simulation with a GPU-portable reconstruction stack.
> *SoftwareX*. 2026;35:102910.
> <https://doi.org/10.1016/j.softx.2026.102910>

```bibtex
@article{BLACK2026102910,
  title = {BasisSimulator.jl: Open-source polychromatic CT simulation with a GPU-portable reconstruction stack},
  journal = {SoftwareX},
  volume = {35},
  pages = {102910},
  year = {2026},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2026.102910},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711026004012},
  author = {Dale Black and Hamidreza Khodajou-Chokami and Sabee Molloi}
}
```

## License

BasisSimulator.jl's original source code is released under the
[MIT License](LICENSE). The package also contains or adapts components under
their original licenses:

- The bundled [CatSim/XCIST](https://github.com/xcist/main) bowtie profiles are
  redistributed under the BSD 3-Clause License.
- The Siddon projector is ported from [TIGRE](https://github.com/CERN/TIGRE)
  under the BSD 3-Clause License; its upstream copyright and license notice are
  preserved in `src/projection/siddon.jl`.
- The distance-driven DD3 projector is ported from CatSim/XCIST under the
  Apache License 2.0, with attribution preserved in `src/projection/dd.jl`.
