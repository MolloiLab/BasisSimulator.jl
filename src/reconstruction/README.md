# CT Reconstruction Methods

This folder organizes CT reconstruction algorithms by clinical terminology and use case.

## Folder Structure

```
reconstruction/
├── core/                 # Shared building blocks
│   ├── backprojection.jl # GPU voxel-driven backprojection
│   └── filtering.jl      # Ramp/Ram-Lak and other filters
├── fbp/                  # Filtered Back-Projection (analytical)
│   └── fdk.jl            # Feldkamp-Davis-Kress cone-beam FBP
├── ir/                   # Classic Iterative Reconstruction
│   ├── sirt.jl           # Simultaneous Iterative Reconstruction
│   └── cgls.jl           # Conjugate Gradient Least Squares
├── hybrid_ir/            # Hybrid Iterative Reconstruction
│   └── hybrid_ir.jl      # TRUE Hybrid IR (PWLS with FDK init)
├── mbir/                 # Model-Based Iterative Reconstruction
│   └── mbir.jl           # Full statistical MBIR
├── regularization/       # Regularization penalties
│   └── tv_regularization.jl  # Total Variation
└── statistical_ir.jl     # PWLS core (used by hybrid_ir)
```

## Clinical Terminology Mapping

| Clinical Term | Vendor Examples | Our Implementation |
|---------------|-----------------|-------------------|
| **FBP** | Basic FBP, FDK | `fbp/fdk.jl` |
| **Hybrid IR** | GE ASIR-V, Siemens SAFIRE, Philips iDose4, Canon AIDR 3D | `hybrid_ir/hybrid_ir.jl` |
| **MBIR** | GE VEO, Siemens ADMIRE, Philips IMR | `mbir/mbir.jl` |

## When to Use Each Method

### FBP/FDK (`fbp/fdk.jl`)
- **Speed**: Fastest
- **Quality**: Good baseline
- **Use when**: Speed is critical, high-dose acquisitions, research baseline

### Hybrid IR (`hybrid_ir/hybrid_ir.jl`)
- **Speed**: 2-5x FDK (strength dependent)
- **Quality**: Significant noise reduction while preserving resolution
- **Use when**: Clinical imaging, moderate dose reduction, balanced quality/speed

### Classic IR (`ir/sirt.jl`, `ir/cgls.jl`)
- **Speed**: 5-20x FDK
- **Quality**: Good for limited-angle/sparse data
- **Use when**: Non-standard geometries, limited projections

### MBIR (`mbir/mbir.jl`)
- **Speed**: 10-50x FDK
- **Quality**: Best possible at very low dose
- **Use when**: Ultra-low dose, research on advanced reconstruction

## Hybrid IR Strength Levels

Hybrid IR uses strength levels 1-5 that map to clinical vendor settings:

| Strength | Noise Reduction | Speed | Vendor Equivalent |
|----------|-----------------|-------|-------------------|
| 1 | ~10% | ~1.5x FDK | ASIR-V 20%, SAFIRE S1 |
| 2 | ~23% | ~2x FDK | ASIR-V 40%, SAFIRE S2 |
| 3 | ~35% | ~3x FDK | ASIR-V 60%, SAFIRE S3 (recommended) |
| 4 | ~48% | ~4x FDK | ASIR-V 80%, SAFIRE S4 |
| 5 | ~59% | ~5x FDK | ASIR-V 100%, SAFIRE S5 |

## Quick Start

```julia
using BasisSimulator

# FBP reconstruction
fdk_result = fdk_reconstruct(sinogram, geom, (256, 256, 128))

# Hybrid IR (recommended for clinical use)
hir_result = hybrid_ir_reconstruct(sinogram, geom, (256, 256, 128); strength=3)

# MBIR (slow but highest quality at low dose)
mbir_result = mbir_reconstruct(sinogram, geom, (256, 256, 128); niter=100)
```
