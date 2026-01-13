# BasisSimulator.jl - Claude Development Guide

## Project Overview

BasisSimulator.jl is a differentiable 3D cone-beam CT simulator designed for:
- Inverse problems and iterative reconstruction
- Machine learning integration (Reactant/Enzyme compatible)
- CatSim/XCIST feature parity for realistic CT simulation
- Educational and research purposes

**Repository**: https://github.com/MolloiLab/BasisSimulator.jl

## CatSim/XCIST Reference

### Key Resources
- **Main Repository**: https://github.com/xcist/main
- **Documentation Wiki**: https://github.com/xcist/documentation/wiki
- **Paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10151073/
- **PyPI Package**: https://pypi.org/project/gecatsim/

### CatSim Source Code Locations
- **Detector Models**: `gecatsim/pyfiles/Detection_EI.py`
- **X-ray Filters**: `gecatsim/pyfiles/Xray_Filter.py`
- **Scatter Model**: `gecatsim/pyfiles/Scatter_ConvolutionModel.py`
- **FDK Reconstruction**: `gecatsim/reconstruction/pyfiles/fdk_equiAngle.py`
- **Recon Kernels**: `gecatsim/reconstruction/pyfiles/createHSP.py`
- **Crosstalk**: Uses `scipy.signal.convolve2d` on detector data
- **Detector Lag**: Dual-exponential decay with memory state

### CatSim Key Equations
- **Detector Efficiency**: `η = 1 - exp(-μ × d / cos(β))`
- **Bowtie Attenuation**: `trans = exp(-Σᵢ μᵢ(E) × tᵢ(θ) / cos(α))`
- **Scatter SPR**: ~15% for 35cm water, 120kVp, 40mm beam width
- **Lag Coefficient Normalization**: Coefficients sum to 1.0 to preserve signal

---

## Current Implementation Status

### ✅ Completed Features (CatSim Parity)

| Feature | File | Description |
|---------|------|-------------|
| Forward Projection | `src/Forward/Projector.jl` | Ray-driven, Siddon's method |
| Polychromatic X-ray | `src/Forward/Polychromatic.jl` | Energy-dependent attenuation, beam hardening |
| Scatter | `src/Forward/Scatter.jl` | Convolution-based model (XCIST-style) |
| Detector Noise | `src/Forward/DetectorNoise.jl` | Poisson + Gaussian noise |
| Detector Efficiency | `src/Forward/DetectorEfficiency.jl` | GOS, CsI, CdTe scintillators, DQE |
| Bowtie Filter | `src/Forward/BowtieFilter.jl` | Multi-material, NIST XCOM μ values |
| Flat Filter | `src/Forward/FlatFilter.jl` | Al, Cu, Ti source filtration |
| Focal Spot | `src/Forward/FocalSpot.jl` | FFT-based blur, multiple shapes |
| Crosstalk | `src/Forward/Crosstalk.jl` | 3x3 kernel convolution |
| Detector Lag | `src/Forward/DetectorLag.jl` | Multi-exponential decay |
| FDK Reconstruction | `src/Reconstruction/FDK.jl` | Cone-beam, Parker weighting |
| Recon Kernels | `src/Reconstruction/Kernels.jl` | Soft, Standard, Bone, Lung |
| Helical Forward | `src/Geometry/Helical.jl` | Pitch, rotations, interpolation |
| Phantoms | `src/Geometry/Phantom.jl` | Gammex 472 with semantic masks |

### ✅ Recently Completed

| Feature | File | Description |
|---------|------|-------------|
| Water BHC | `src/Reconstruction/BeamHardeningCorrection.jl` | Polynomial correction, Horner scheme |
| Helical FDK | `src/Geometry/Helical.jl` | Slice-by-slice interpolation + FDK |
| Tang 3D Weighting | `src/Reconstruction/FDK.jl` | Cone-beam artifact reduction |
| Optical Crosstalk | `src/Forward/Crosstalk.jl` | CatSim-style separable kernel |
| Fill Factors | `src/Forward/FillFactor.jl` | Detector pixel active area |
| Flying Focal Spot | `src/Forward/FlyingFocalSpot.jl` | 2/4-position sampling improvement |
| Iterative Recon | `src/Reconstruction/Iterative.jl` | SIRT, CGLS (XLA-compatible) |

### ❌ Not Yet Implemented

| Feature | Priority | Complexity |
|---------|----------|------------|
| Azimuthal Blur | Medium | Motion blur within view |
| Material Database | Medium | NIST XCOM 194 materials |
| XCAT Phantoms | High | Anatomical phantom support |
| Photon Counting | High | PC detector model |
| Dose Estimation | Medium | Kernel-based dose calc |
| DICOM Output | Medium | Standard medical format |

---

## Architecture

### Module Structure
```
src/
├── BasisSimulator.jl          # Main module
├── Physics/
│   ├── Materials.jl           # Gammex material definitions
│   ├── Spectrum.jl            # X-ray spectrum loading
│   └── Attenuation.jl         # μ calculation
├── Geometry/
│   ├── Phantom.jl             # Phantom generation
│   ├── Scanner.jl             # CT geometry (Aquilion One)
│   └── Helical.jl             # Helical scanning
├── Forward/
│   ├── Projector.jl           # Forward projection
│   ├── Polychromatic.jl       # Polychromatic simulation
│   ├── Scatter.jl             # Scatter modeling
│   ├── DetectorNoise.jl       # Noise modeling
│   ├── DetectorEfficiency.jl  # DQE/absorption
│   ├── BowtieFilter.jl        # Bowtie filter
│   ├── FlatFilter.jl          # Flat filter
│   ├── FocalSpot.jl           # Focal spot blur
│   ├── Crosstalk.jl           # Detector crosstalk
│   └── DetectorLag.jl         # Afterglow
├── Reconstruction/
│   ├── Kernels.jl             # Recon kernels
│   ├── FDK.jl                 # FDK reconstruction
│   └── Iterative.jl           # SIRT, CGLS iterative recon
├── Scanners/
│   ├── Scanners.jl            # Scanner module with base types
│   └── GeneralElectric.jl     # GE Revolution Apex Elite
└── Optimization/
    ├── Loss.jl                # Loss functions
    └── Gradients.jl           # Gradient computation
```

### Key Design Decisions
1. **FFT Kernel Centering**: Use `mod1(1 + dx, n_cols)` for wrap-around at (1,1)
2. **Intensity Domain**: Crosstalk/lag applied in intensity (exp(-sino)), focal blur in projection
3. **Coefficient Normalization**: Lag coefficients sum to 1.0 to preserve signal
4. **NIST XCOM Data**: Lookup tables with log-linear interpolation for μ values
5. **Differentiable Design**: Separate pre-computation (geometry/indices) from XLA-traceable code

### Differentiable Architecture (Reactant/XLA)

The entire imaging chain must be differentiable for inverse problems:

```
Volume → Forward Project → [Effects] → Sinogram → FDK Reconstruct → Volume
   ↑                                                                    |
   └──────────────────── ∇loss ←───────────────────────────────────────┘
```

**Pattern**: Pre-compute geometry indices once, then use gather/scatter operations.

| Component | Pre-computation | XLA-Traceable Kernel |
|-----------|----------------|---------------------|
| Forward Proj | `precompute_projection_geometry()` | `project_volume()` |
| FDK Recon | `precompute_backprojection_geometry()` | `backproject_volume()` |

**FDK Differentiable Requirements**:
- Functional (non-mutating) cosine pre-weighting
- FFT filtering without FFTW (use real-valued convolution or Reactant.fft)
- Pre-computed backprojection indices with gather-based sampling
- No scalar loops in traced code

### Clinical Scanner Configurations

Scanner configurations are modular and manufacturer-specific:

```
src/Scanners/
├── Scanners.jl           # Main module with base types
├── GeneralElectric.jl    # GE scanners (Revolution Apex Elite, etc.)
├── Siemens.jl            # Future: Siemens scanners
└── Canon.jl              # Future: Canon/Toshiba scanners
```

**Design Pattern**:
1. **Specification Structs**: Detailed hardware specs with FDA 510(k) citations
2. **Factory Functions**: Create `CTGeometry` from specs using base functions
3. **Protocol Presets**: Common clinical protocols (chest, head, cardiac, etc.)

**Adding a New Scanner**:
1. Create manufacturer file (e.g., `Siemens.jl`)
2. Define spec struct with all hardware parameters + source citations
3. Implement `create_[scanner]_geometry()` factory function
4. Add protocol presets as needed
5. Export from `Scanners.jl`

**Source Requirements** (for publication-ready configs):
- FDA 510(k) summaries (accessdata.fda.gov)
- Manufacturer technical specifications
- Peer-reviewed publications
- Each parameter must have a source URL or derivation note

---

## Roadmap

### Phase 1: CatSim Parity (COMPLETE ✅)
- [x] Core forward projection
- [x] Polychromatic/beam hardening
- [x] Scatter model
- [x] Detector effects (noise, blur, crosstalk, lag)
- [x] Source effects (bowtie, flat filter, focal spot)
- [x] FDK reconstruction with kernels
- [x] Helical forward projection
- [x] Water beam hardening correction
- [x] Helical FDK reconstruction

### Phase 2: Differentiable Reconstruction (COMPLETE ✅)
- [x] Differentiable forward projection (`project_volume`)
- [x] Differentiable FDK reconstruction
  - [x] Functional cosine pre-weighting (`preweight_cosine`)
  - [x] Pre-computed backprojection geometry (`BackprojectionGeometry`)
  - [x] Gather-based backprojection kernel (`backproject_volume`)
  - [x] XLA-compatible FDK (`fdk_reconstruct_xla`)
- [x] Iterative reconstruction (SIRT, CGLS)
- [x] Clinical Scanner Configurations
  - [x] Scanner configuration module (`src/Scanners/`)
  - [x] GE Revolution Apex Elite (K213715)
  - [ ] Siemens SOMATOM (future)
  - [ ] Canon Aquilion (future)
- [ ] Full material database (NIST XCOM)
- [ ] Photon counting detector model

### Phase 3: DukeSim Parity
- [ ] Research DukeSim features
- [ ] Implement missing physics models
- [ ] Validation against DukeSim outputs

### Phase 4: Documentation & Validation
- [ ] Pluto notebook demonstrating CatSim++ parity
- [ ] DICOM output support
- [ ] Comprehensive documentation
- [ ] Validation against clinical data

---

## Testing

Run tests: `julia --project -e 'using Pkg; Pkg.test()'`

Current test count: **564 tests**

### Test Categories
- Forward projection accuracy
- Polychromatic simulation
- Scatter effects
- Detector noise models
- FDK reconstruction
- Iterative reconstruction (SIRT, CGLS)
- Scanner configurations (GE Revolution Apex Elite)
- All CatSim parity features

---

## Visualization

Run visualization script:
```bash
julia --project stuff/scripts/visualize.jl
```

Outputs 20 PNG files to `stuff/scripts/outputs/`:
1. Phantom regions
2. Monochromatic reconstruction
3. Sinogram
4. Polychromatic (beam hardening)
5. Scatter effects
6. Noise effects
7. Full realistic simulation
8. Bowtie profile
9. Bowtie sinogram
10. Focal spot effect
11. Crosstalk effect
12. Crosstalk reconstruction
13. Lag impulse response
14. Lag effect
15. Helical trajectory
16. Helical sinogram
17. Interpolated axial
18. All effects combined
19. Comprehensive comparison
20. Effect magnitudes

---

## API Quick Reference

### Phantoms
```julia
phantom = create_gammex_472(n_voxels=64)
mask = get_region_mask(phantom, REGION_BONE_50)
```

### Geometry
```julia
geom = create_aquilion_one(n_angles=360, n_rows=64, n_cols=512)
geom_helical = create_scan_geometry(mode=:helical, pitch=1.0, n_rotations=3.0)
```

### Forward Projection
```julia
sino = forward_project(phantom, geom)
sino_poly = forward_project_polychromatic(phantom, projector)
```

### Effects
```julia
sino = apply_bowtie_filter(sino, bowtie_filter_medium_body(), geom)
sino = apply_flat_filter(sino, flat_filter_al_cu(), geom)
sino = apply_focal_spot_blur(sino, focal_spot_medium(), geom)
sino = apply_crosstalk(sino, crosstalk_medium())
sino = apply_lag(sino, lag_gadox())
sino = apply_detector_model(sino, default_detector_model())
```

### Reconstruction
```julia
# Legacy FDK (loop-based, faster for single runs)
recon = fdk_reconstruct(sino, geom, size(phantom.μ), phantom.fov)
recon = fdk_reconstruct(sino, geom, size, fov; kernel=kernel_soft())
recon_HU = μ_to_HU(recon, get_reference_μ_water(60.0))

# XLA-Compatible FDK (differentiable, for Reactant compilation)
bp_geom = precompute_backprojection_geometry(geom, output_size, fov)
recon = fdk_reconstruct_xla(sino, geom, bp_geom)

# Individual steps (for custom pipelines)
weighted = preweight_cosine(sino, geom)
filtered = filter_ramp(weighted, geom; kernel=kernel_soft())
volume = backproject_volume(vec(filtered), bp_geom)

# Iterative Reconstruction (XLA-compatible)
proj_geom = precompute_projection_geometry(geom, fov, voxel_size, output_size)
bp_geom = precompute_backprojection_geometry(geom, output_size, fov)

# SIRT (robust to noise, slower convergence)
result = sirt_reconstruct(sino, proj_geom, bp_geom; n_iterations=50)
recon = result.volume

# CGLS (faster convergence, semi-convergent)
result = cgls_reconstruct(sino, proj_geom, bp_geom; n_iterations=20)
recon = result.volume

# Convenience wrapper
recon = iterative_reconstruct(sino, geom, output_size, fov; method=:sirt, n_iterations=30)
```

### Detector Efficiency
```julia
det = detector_efficiency_gos(0.5)  # 0.5mm GOS
η = compute_detector_efficiency(det, geom; energy_keV=60.0)
dqe = compute_dqe(det, 60.0)
```

---

## Bugs Fixed (Reference)

### Focal Spot Kernel Centering
**Problem**: 1571 HU difference from ideal
**Solution**: FFT kernel must be centered at (1,1) with wrap-around using `mod1()`

### Detector Lag Normalization
**Problem**: Negative sinogram values due to coefficients summing to 1.26
**Solution**: Normalize coefficients to sum to 1.0: `coeffs ./= sum(coeffs)`

### Bowtie μ Values
**Problem**: μ values 10x too low due to incorrect fit formula
**Solution**: Use NIST XCOM lookup tables with log-linear interpolation

---

## User Goals (Project Owner)

1. ✅ Finish CatSim parity of BasisSimulator.jl (COMPLETE)
2. ✅ Add iterative reconstruction (SIRT, CGLS)
3. ✅ Add perfectly documented GE Revolution Apex scanner config
4. ⏳ Build out Pluto notebook showing the CatSim++ parity
5. ⏳ Begin building out DukeSim parity

---

## Session Continuation Notes

When continuing a session:
1. Run `git status` to check current state
2. Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
3. Check TODO status in this file
4. Consult CatSim source code before implementing new features
5. Commit and push incrementally

Last updated: 2026-01-12
Current test count: 564
Phase 1 (CatSim Parity): COMPLETE
Phase 2 (Differentiable Reconstruction): COMPLETE
Additional features: Tang 3D weighting, optical crosstalk, fill factors, flying focal spot, iterative recon (SIRT, CGLS)
XLA-Compatible: forward_project (project_volume), fdk_reconstruct_xla (backproject_volume), sirt_reconstruct, cgls_reconstruct
Clinical Scanners: GE Revolution Apex Elite (K213715) with FDA 510(k) sourced parameters
