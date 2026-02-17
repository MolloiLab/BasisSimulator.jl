# Running BasisSimulator.jl on Windows with CUDA (NVIDIA GPU)

This guide walks through running BasisSimulator.jl on **Windows** with an **NVIDIA GPU** using CUDA.

## Prerequisites

- **Julia 1.11+**  [julialang.org](https://julialang.org/downloads/)
- **NVIDIA GPU** with up-to-date driver  [nvidia.com](https://www.nvidia.com/Download/index.aspx)
- No separate CUDA Toolkit needed — Julia's CUDA.jl ships its own runtime.

## Setup

```powershell
git clone https://github.com/MolloiLab/BasisSimulator.jl.git
cd BasisSimulator.jl
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.add(\"CUDA\")"
```

## Verify GPU

```powershell
julia --project=. -e "using CUDA; println(CUDA.functional()); println(CUDA.device())"
```

## GPU Phantom Initialization

The phantom must be created on CPU first, then the mask is moved to GPU.
Materials and metadata stay on CPU — only the mask array goes to GPU memory.

```julia
using BasisSimulator as BS
using CUDA

# Step 1: Create phantom on CPU
phantom_cpu = BS.create_gammex_472(n_voxels=512, n_slices=32)

# Step 2: Move mask to GPU
phantom_gpu_mask = CuArray(phantom_cpu.mask)

# Step 3: Create final phantom struct with GPU mask + CPU metadata
phantom = BS.Phantom(
    phantom_gpu_mask,        # UInt8 mask on GPU
    phantom_cpu.materials,   # materials stay on CPU
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.fov,
)
```

> **Note:** On macOS, replace `CuArray` with `MtlArray` (from Metal.jl).

## Run

```powershell
julia --project=. run_demo.jl
```

## Key Difference from macOS

| Platform | GPU backend | Array type | What to load |
|----------|-------------|------------|--------------|
| macOS    | Metal.jl    | `MtlArray` | `using Metal` |
| Windows  | CUDA.jl     | `CuArray`  | `using CUDA`  |
| Linux    | CUDA.jl or AMDGPU.jl | `CuArray` / `ROCArray` | `using CUDA` or `using AMDGPU` |

The source code is identical — `AcceleratedKernels.jl` handles GPU dispatch automatically.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA is not functional` | Update NVIDIA driver |
| Out of memory | Reduce `n_voxels` or `views` |
| Precompilation errors | `Pkg.precompile()` or `Pkg.update()` |
