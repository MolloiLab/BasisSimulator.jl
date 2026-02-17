# Running BasisSimulator.jl on Windows with CUDA (NVIDIA GPU)

This guide walks through running BasisSimulator.jl on **Windows** with an **NVIDIA GPU** using CUDA.

## Prerequisites

- **Julia 1.11+**  [julialang.org](https://julialang.org/downloads/)
- **NVIDIA GPU** with up-to-date driver  [nvidia.com](https://www.nvidia.com/Download/index.aspx)
- No separate CUDA Toolkit needed  Julia's CUDA.jl ships its own runtime.

## Setup

`powershell
git clone https://github.com/MolloiLab/BasisSimulator.jl.git
cd BasisSimulator.jl
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.add(\"CUDA\")"
`

## Verify GPU

`powershell
julia --project=. -e "using CUDA; println(CUDA.functional()); println(CUDA.device())"
`

## Run

`powershell
julia --project=. run_demo.jl
`

## Key Difference from macOS

| Platform | GPU backend | What to load |
|----------|-------------|--------------|
| macOS    | Metal.jl    | `using Metal` |
| Windows  | CUDA.jl     | `using CUDA`  |
| Linux    | CUDA.jl or AMDGPU.jl | `using CUDA` or `using AMDGPU` |

The source code is identical  `AcceleratedKernels.jl` handles GPU dispatch automatically.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA is not functional` | Update NVIDIA driver |
| Out of memory | Reduce `matrix_size` or `views` |
| Precompilation errors | `Pkg.precompile()` or `Pkg.update()` |
