# BasisSimulator.jl

[![CI](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MolloiLab/BasisSimulator.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://molloilab.github.io/BasisSimulator.jl/)
[![SoftwareX DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.102910-blue)](https://doi.org/10.1016/j.softx.2026.102910)
[![Zenodo archive](https://zenodo.org/badge/DOI/10.5281/zenodo.20262003.svg)](https://doi.org/10.5281/zenodo.20262003)

Polychromatic CT simulation and reconstruction in Julia, on a GPU or the CPU.

Put a phantom in a model of a clinical CT scanner and get back what that scanner would give you:
images in Hounsfield units, with realistic noise and artifacts, and the dose of the scan.
BasisSimulator.jl models energy-integrating and photon-counting scanners from the source spectrum to
the detector counts, and reconstructs what they measure: cone-beam FDK, helical WFBP, hybrid
iterative reconstruction, and virtual monoenergetic images from dual-kVp, dual-source and
photon-counting acquisitions. The kernels are written once against
[AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl); they are tested on NVIDIA
(CUDA) and Apple (Metal) GPUs and on the CPU.

## Install

Julia 1.12 or later, started with `julia -t auto`:

```julia
using Pkg
Pkg.add(["BasisSimulator", "GPUSelect"])
Pkg.add("Metal") # Apple silicon; "CUDA" for NVIDIA; leave it out to run on the CPU
```

## Start here

**[Getting Started](https://molloilab.github.io/BasisSimulator.jl/getting-started/)** takes you from
an empty environment to a scan of a calibration phantom, an anatomical chest with and without
contrast, and photon-counting virtual monoenergetic images. It explains each setting, lists every
rough edge users have hit with what to do about it, and covers how to contribute.

## Documentation

**<https://molloilab.github.io/BasisSimulator.jl/>**

- [Getting Started](https://molloilab.github.io/BasisSimulator.jl/getting-started/): the guide.
- [Examples](https://molloilab.github.io/BasisSimulator.jl/examples/): twelve Pluto notebooks, from
  the five-struct API to dual-source VMI, each checked against theory.
- [Scanners](https://molloilab.github.io/BasisSimulator.jl/scanners/): the clinical parameter sets
  to copy.
- [API](https://molloilab.github.io/BasisSimulator.jl/api/): every exported name, from its docstring
  (also `?BS.simulate!` in the REPL).

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
