# BasisSimulator.jl - TIGRE-Style CT Simulator

## Overview

CT simulation with backend-agnostic GPU/CPU execution via **AcceleratedKernels.jl**.

Core ray tracing algorithms ported from TIGRE. Polychromatic physics is our own implementation (TIGRE is monochromatic only).

**Works on:** Metal (Apple), CUDA (NVIDIA), ROCm (AMD), Intel oneAPI, or CPU.

---

## Current Status

| Component | File | Reference | Status |
|-----------|------|-----------|--------|
| Forward Projection | `Forward/Siddon.jl` | TIGRE `Siddon_projection.cu` | ✅ Complete |
| Polychromatic FP | `Forward/Polychromatic.jl` | Beer-Lambert physics | ✅ Complete |
| Backprojection | `Reconstruction/Backprojection.jl` | TIGRE `voxel_backprojection.cu` | ✅ Complete |
| FDK Filtering | `Reconstruction/Filtering.jl` | Standard ramp filter | ✅ Complete |
| FDK Reconstruction | `Reconstruction/FDK.jl` | TIGRE FDK | ✅ Complete |

---

## AcceleratedKernels.jl Approach

We use [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) instead of raw KernelAbstractions.jl for cleaner code:

```julia
import AcceleratedKernels as AK

# Parallel loop - automatically runs on GPU or CPU
AK.foreachindex(sinogram) do idx
    # Convert linear index to (col, row, angle)
    ci = CartesianIndices(sinogram)[idx]
    col, row, angle = Tuple(ci)

    # Compute ray trace for this detector pixel
    sinogram[idx] = siddon_trace_ray(...)
end
```

**Key advantages:**
- No `@kernel` macros needed - just normal Julia code
- `AK.foreachindex` automatically parallelizes
- Works on any backend without code changes
- Cleaner, more readable than raw CUDA/KA code

---

## Siddon Forward Projection

**Reference:** `CERN/TIGRE/Common/CUDA/Siddon_projection.cu`

### Algorithm

Per-ray computation (each thread = one detector pixel):
1. Compute ray from source to detector pixel
2. Find entry/exit points where ray intersects volume
3. Traverse voxels using Siddon's parametric algorithm
4. Accumulate: `line_integral += μ[voxel] × path_length`

### Usage

```julia
using BasisSimulator

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=128, fov_cm=35.0, z_cm=4.0)
geom = create_aquilion_one(n_angles=180, n_rows=32, n_cols=256, fov_cm=35.0)

# Forward projection (automatically uses GPU if available)
sinogram = siddon_forward_project(Float32.(phantom.μ), geom)
```

---

## Voxel Backprojection

**Reference:** `CERN/TIGRE/Common/CUDA/voxel_backprojection.cu`

### Algorithm

Per-voxel computation (each thread = one voxel):
1. Compute voxel center in world coordinates
2. For each angle:
   - Project voxel onto detector plane
   - Bilinear interpolation of detector values
   - Apply FDK distance weight: `(SAD / dist)²`
3. Accumulate weighted contributions

### Usage

```julia
# Backproject filtered sinogram
volume = backproject(filtered_sinogram, geom, (128, 128, 64))
```

---

## FDK Reconstruction

**Reference:** Feldkamp, Davis, Kress (1984)

### Pipeline

1. **Cosine weighting** - Pre-weight for cone-beam geometry
2. **Ramp filtering** - Fourier domain filter (Ram-Lak, Shepp-Logan, etc.)
3. **Weighted backprojection** - FDK distance weights

### Usage

```julia
# Full FDK reconstruction
recon = fdk_reconstruct(sinogram, geom, size(phantom.μ))

# With filter options
recon = fdk_reconstruct(sinogram, geom, size(phantom.μ);
                        filter=SheppLoganFilter(), cutoff=0.8)
```

### Available Filters

- `RampFilter()` - Standard Ram-Lak
- `SheppLoganFilter()` - Ramp × sinc
- `CosineFilter()` - Ramp × cos
- `HammingFilter()` - Ramp × Hamming window
- `HannFilter()` - Ramp × Hann window

---

## Unified Forward Projection API

Single function for both monochromatic and polychromatic projection:

```julia
# Direct volume input (monochromatic)
sinogram = forward_project(Float32.(phantom.μ), geom)

# Mask + single energy (monochromatic)
materials = get_region_materials()
sinogram = forward_project(phantom.mask, geom; energy=60.0, materials=materials)

# Mask + spectrum (polychromatic)
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)
sinogram = forward_project(phantom.mask, geom; energies=energies, weights=weights, materials=materials)
```

### Polychromatic Physics

**NOT from TIGRE** - TIGRE is monochromatic only.

Beer-Lambert law:
```
I = Σₑ wₑ × exp(-∫μₑ dl)
sinogram = -log(I / I₀)
```

---

## File Structure

```
src/
├── BasisSimulator.jl
├── Forward/
│   ├── Siddon.jl              # ✅ TIGRE port (AcceleratedKernels.jl)
│   ├── Polychromatic.jl       # ✅ Our implementation
│   └── [physics effects]      # Scatter, noise, etc.
├── Reconstruction/
│   ├── Backprojection.jl      # ✅ TIGRE port (AcceleratedKernels.jl)
│   ├── Filtering.jl           # ✅ Ramp filter (FFTW)
│   └── FDK.jl                 # ✅ Full FDK pipeline
├── Geometry/
│   ├── Scanner.jl
│   ├── Phantom.jl
│   └── Helical.jl
├── Physics/
│   ├── Materials.jl
│   ├── Attenuation.jl
│   └── Spectrum.jl
└── Scanners/
    └── Scanners.jl
```

---

## GPU Usage

AcceleratedKernels.jl automatically detects the backend from the array type:

```julia
# CPU (default)
sinogram = siddon_forward_project(Float32.(volume), geom)

# Metal (Apple Silicon)
using Metal
volume_gpu = MtlArray(Float32.(volume))
sinogram_gpu = siddon_forward_project(volume_gpu, geom)

# CUDA (NVIDIA)
using CUDA
volume_gpu = CuArray(Float32.(volume))
sinogram_gpu = siddon_forward_project(volume_gpu, geom)
```

---

## References

1. **TIGRE**: https://github.com/CERN/TIGRE
   - `Common/CUDA/Siddon_projection.cu`
   - `Common/CUDA/voxel_backprojection.cu`

2. **AcceleratedKernels.jl**: https://github.com/JuliaGPU/AcceleratedKernels.jl

3. **Feldkamp, Davis, Kress (1984)**: "Practical cone-beam algorithm"

4. **Siddon (1985)**: "Fast calculation of the exact radiological path"

---

Last Updated: 2025-01-14
