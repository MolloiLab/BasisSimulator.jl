# BasisSimulator.jl Roadmap

## Project Vision

BasisSimulator.jl is a **differentiable 3D cone-beam CT simulator** for medical imaging research. The goal is to provide:
1. **Accurate physics**: Realistic X-ray attenuation based on NIST XCOM data
2. **Differentiability**: Gradients via Reactant/XLA for optimization and learning
3. **Validation**: Gammex 472 phantom with known material properties for HU verification
4. **Pure Julia**: No Python dependencies, full integration with Julia ecosystem

---

## Completed Phases

### Phase 0: Repository Cleanup
- Removed broken/incomplete code from initial development
- Established clean module structure

### Phase 1: Foundation
- **Spectrum Loading**: X-ray spectrum data (xspect, xcist sources) at various kVp
- **Attenuation Physics**: μ matrix computation, HU conversion functions
- **XrayAttenuation.jl Integration**: v0.2.0 for NIST XCOM database access

### Phase 2: Phantom Infrastructure
- **Gammex 472 Phantom**: Digital phantom with 14 inserts (7 calcium, 7 iodine)
- **Region Labels**: Enumerated regions with material-to-region mapping
- **Mask System**: Binary masks for each region, statistics computation
- **Validation Framework**: Compare reconstruction to ground truth by region

### Phase 3: Cone-Beam Imaging Chain
- **CT Geometry**: `CTGeometry` struct with Aquilion One presets
- **Source/Detector Positions**: Pre-computed for all angles
- **FDK Reconstruction**: Filtered backprojection for cone-beam

### Phase 3.5: Siddon's Method
- **Exact Voxel Traversal**: Siddon's 1985 algorithm for line integrals
- **Pre-computed Geometry**: `ProjectionGeometry` with indices and path-length weights
- **XLA Compatibility**: Traced code uses only gather/scatter operations

### Cleanup (Current)
- **XrayAttenuation.jl v0.2.0**: Upgraded for `gammex_472_*` materials
- **Materials.jl**: Now aliases XA.Materials directly (no duplicate definitions)
- **Siddon-Only**: Single forward projection algorithm

---

## Current Architecture

```
BasisSimulator.jl/
├── src/
│   ├── BasisSimulator.jl       # Main module
│   ├── Physics/
│   │   ├── Materials.jl        # Gammex 472 materials (from XA.Materials)
│   │   ├── Attenuation.jl      # μ computation, HU conversion
│   │   └── Spectrum.jl         # X-ray spectrum loading
│   ├── Phantom/
│   │   ├── Regions.jl          # Region labels enum
│   │   ├── Phantom.jl          # Phantom struct, Gammex 472 creation
│   │   └── Validation.jl       # Reconstruction validation
│   ├── Geometry/
│   │   └── CTGeometry.jl       # Scanner geometry
│   ├── Forward/
│   │   └── Projector.jl        # Siddon forward projection
│   └── Reconstruction/
│       └── FDK.jl              # FDK reconstruction
├── data/
│   └── spectra/                # X-ray spectrum files
└── test/
    └── runtests.jl             # 100 tests, all passing
```

### Key Design Decisions

1. **Siddon's Method**: Exact voxel traversal (not trilinear interpolation)
   - More accurate for validation
   - Pre-computed geometry separates traced/non-traced code

2. **Pre-computed Projection Geometry**:
   - `ProjectionGeometry` contains linear indices and weights
   - Traced code only does gather + weighted sum
   - Avoids scalar indexing in XLA

3. **Materials from XrayAttenuation.jl**:
   - All Gammex 472 materials sourced from XA.Materials
   - Single source of truth for elemental compositions and densities

---

## Phase 4: Realistic Physics (Next)

### 4.1 Polychromatic Simulation
**Goal**: Simulate energy-dependent attenuation across X-ray spectrum

```julia
# Current (monochromatic approximation)
sinogram = forward_project(phantom, geom)

# Target (polychromatic)
sinogram = forward_project(phantom, geom, spectrum)
```

**Tasks**:
- [ ] Energy-binned forward projection
- [ ] Spectrum-weighted line integrals: I/I₀ = Σ w(E) exp(-∫μ(E)ds)
- [ ] Beam hardening artifacts (naturally arise from polychromatic simulation)

### 4.2 Scatter Modeling
**Goal**: Add scatter contribution to projections

**Options** (in order of complexity):
1. Analytic scatter kernel (fast, approximate)
2. Single-scatter convolution
3. Monte Carlo scatter (most accurate, slow)

**Initial approach**: Start with analytic scatter kernel

### 4.3 Detector Response
**Goal**: Model detector physics

- [ ] Detector efficiency vs energy
- [ ] Crosstalk between pixels
- [ ] Afterglow (for temporal effects)

### 4.4 Noise Modeling
**Goal**: Realistic quantum and electronic noise

- [ ] Poisson noise (photon counting)
- [ ] Electronic noise (Gaussian additive)
- [ ] Noise variance estimation

---

## Phase 5: Optimization & Gradients

### 5.1 Reactant/XLA Optimization
**Goal**: Maximize compilation performance

- [ ] Profile current compiled code
- [ ] Optimize memory layout for XLA
- [ ] Batch processing for multiple volumes

### 5.2 Gradient Verification
**Goal**: Validate autodiff correctness

- [ ] Finite difference comparison
- [ ] Gradient through full forward+recon chain
- [ ] Test optimization convergence

### 5.3 Inverse Problems
**Goal**: Enable learned reconstruction

- [ ] Loss functions (MSE, SSIM, perceptual)
- [ ] Regularization terms
- [ ] Integration with Lux.jl for neural networks

---

## Phase 6: Advanced Features

### 6.1 Multi-Material Decomposition
**Goal**: Separate materials using multi-energy data

- [ ] Dual-energy CT simulation
- [ ] Basis material decomposition (water/calcium/iodine)
- [ ] K-edge imaging simulation

### 6.2 Motion Artifacts
**Goal**: Simulate respiratory/cardiac motion

- [ ] Time-varying phantom
- [ ] Motion blur in projections
- [ ] Motion-corrupted reconstruction

### 6.3 Metal Artifacts
**Goal**: Simulate metal implant effects

- [ ] High-Z material inserts
- [ ] Photon starvation
- [ ] Metal artifact reduction testing

---

## Testing Strategy

### Current (100 tests)
- Materials: XA.Materials aliasing, HU validation
- Spectrum: Loading, mean energy
- Attenuation: μ computation, HU conversion
- Phantom: Creation, region masks, validation
- Geometry: Scanner setup, source/detector positions
- Forward: Siddon projection
- Reconstruction: FDK
- End-to-end: Full chain validation
- Reactant: XLA compilation

### Future Tests
- [ ] Polychromatic vs monochromatic comparison
- [ ] Scatter contribution magnitude
- [ ] Noise statistics validation
- [ ] Gradient numerical checks
- [ ] Known analytic phantoms (Shepp-Logan)

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| XrayAttenuation.jl | 0.2.0 | NIST XCOM data, Gammex 472 materials |
| Reactant.jl | 0.2.x | XLA compilation, autodiff |
| FFTW.jl | 1.x | FFT for FDK reconstruction |
| Unitful.jl | 1.x | Physical units |

---

## Quick Start for Contributors

```julia
# Clone and setup
git clone https://github.com/MolloiLab/BasisSimulator.jl
cd BasisSimulator.jl
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project -e 'using Pkg; Pkg.test()'

# Basic usage
using BasisSimulator

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=64)
geom = create_aquilion_one(n_angles=360, n_rows=32, n_cols=256)

# Forward project
sinogram = forward_project(phantom, geom)

# Reconstruct
recon = fdk_reconstruct(sinogram, geom, size(phantom.μ), phantom.fov)

# Validate
result = validate_reconstruction(phantom, recon)
println("Validation passed: ", result.passed)
```

---

## Version History

- **v0.2.0** (current): Siddon's method, XrayAttenuation.jl v0.2.0, cleanup
- **v0.1.0**: Initial working version with uniform sampling

---

## Notes for Resuming Work

If restarting from scratch, follow this order:

1. **Phase 1**: Get spectrum loading and μ computation working
2. **Phase 2**: Build Gammex 472 phantom with region masks
3. **Phase 3**: Implement CT geometry and basic forward projection
4. **Phase 3.5**: Replace with Siddon's exact method
5. **Cleanup**: Ensure XA.Materials integration
6. **Phase 4+**: Add physics features incrementally

Key files to understand first:
- `src/Physics/Materials.jl` - Material definitions
- `src/Phantom/Phantom.jl` - Phantom creation
- `src/Forward/Projector.jl` - Siddon's algorithm
- `test/runtests.jl` - What should pass

---

*Last updated: January 2026*
