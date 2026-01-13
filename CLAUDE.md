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
│   ├── Projector.jl           # Forward projection (Siddon)
│   ├── RayMarching.jl         # Ray marching forward projection (NEW)
│   ├── ClinicalProjector.jl   # Angle-batched projection
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
│   ├── RayMarchingBackproj.jl # Ray marching backprojection (NEW)
│   ├── ClinicalFDK.jl         # Angle-batched FDK
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
- [x] Clinical-scale projection/reconstruction (angle-batched)
- [ ] Full material database (NIST XCOM)
- [ ] Photon counting detector model

### Phase 3: DukeSim Parity
- [ ] Research DukeSim features
- [ ] Implement missing physics models
- [ ] Validation against DukeSim outputs

### Phase 4: Documentation & Validation
- [x] Pluto notebook demonstrating realistic CT simulation
- [ ] DICOM output support
- [ ] Comprehensive documentation
- [ ] Validation against clinical data

---

## Pluto Notebook: Realistic CT Simulation

### Overview
A clean, comprehensive Pluto notebook demonstrating physically realistic CT simulation
using high-resolution phantoms, clinical scanner configurations, and multiple reconstruction
methods with Reactant/XLA compilation for performance.

### Design Philosophy
1. **High-resolution phantom**: Physical objects have atoms at much higher resolution than CT voxels
2. **Clinical realism**: Use GE Revolution Apex Elite scanner with real FDA 510(k) parameters
3. **Multiple protocols**: Compare different kVp/mAs settings (dose levels)
4. **Multiple reconstructions**: FDK (fast), SIRT (robust), CGLS (optimal)
5. **Compiled execution**: Use `@compile` throughout for performance

### Notebook Structure

#### 1. Setup & Configuration
- Load BasisSimulator, Reactant, CairoMakie
- Define output resolution: 512x512x20 (clinical-like slab)
- Configure Reactant compilation

#### 2. High-Resolution Phantom Creation
- Create Gammex 472 at 1024x1024x40 (4x oversampling in-plane, 2x in z)
- This represents the "true" physical object
- Display phantom slices with material labels

#### 3. Scanner Configuration
- Use `GERevolutionApexElite()` scanner spec
- Create geometry with clinical parameters:
  - 984 angles per rotation (clinical standard)
  - 64 detector rows (typical acquisition)
  - 512 detector columns
  - SAD/SDD from FDA 510(k): 626mm / 1097mm

#### 4. Acquisition Protocols
Three protocols to compare:
| Protocol | kVp | mAs | Use Case |
|----------|-----|-----|----------|
| Low Dose | 80 | 100 | Pediatric/screening |
| Standard | 120 | 200 | Routine diagnostic |
| High Dose | 140 | 400 | Obese/high contrast |

#### 5. Physical Effects Pipeline
For each protocol, apply full physics chain:
```
Phantom → Forward Project (polychromatic) → Flat Filter → Bowtie Filter
       → Detector Efficiency → Scatter → Quantum Noise → Sinogram
```

Effects included:
- **Polychromatic X-ray**: Energy-dependent μ, beam hardening
- **Flat filter**: 2.5mm Al + 0.1mm Cu (typical filtration)
- **Bowtie filter**: Medium body filter
- **Detector efficiency**: GOS scintillator model
- **Scatter**: Convolution-based SPR model
- **Quantum noise**: Poisson noise based on I0 (mAs-dependent)

#### 6. Reconstruction Methods
For each protocol, reconstruct with three methods:

| Method | Iterations | Characteristics |
|--------|------------|-----------------|
| FDK | N/A | Fast, analytical, some artifacts |
| SIRT | 30 | Robust to noise, slower convergence |
| CGLS | 15 | Fastest convergence, may amplify noise |

Output: 512x512x20 volume (downsampled from high-res phantom)

#### 7. Compilation Strategy
Pre-compile all compute-intensive operations:
```julia
# Geometry pre-computation (done once)
proj_geom = precompute_projection_geometry(...)
bp_geom = precompute_backprojection_geometry(...)

# Compile forward projection
compiled_project = @compile project_volume(volume_ra, proj_geom)

# Compile backprojection (for iterative methods)
compiled_backproject = @compile backproject_volume(sino_ra, bp_geom)
```

#### 8. Visualization
- Side-by-side comparison of protocols (80/120/140 kVp)
- Reconstruction method comparison (FDK/SIRT/CGLS)
- HU accuracy in ROIs (water, bone, iodine)
- Noise measurements in uniform regions
- Difference images showing reconstruction artifacts

#### 9. Quantitative Analysis
- Mean HU in each Gammex insert
- Noise (std dev) in water region
- CNR for calcium/iodine inserts
- Comparison table across protocols and methods

### Expected Output
- 9 reconstructed volumes (3 protocols × 3 methods)
- Comparison figures showing image quality trade-offs
- Quantitative metrics demonstrating physical realism

### Performance Notes
- High-res phantom creation: ~30 seconds
- Forward projection per protocol: ~2-5 minutes (compiled)
- FDK reconstruction: ~30 seconds (compiled)
- SIRT/CGLS reconstruction: ~2-5 minutes per protocol (compiled)
- Total notebook runtime: ~30-60 minutes (first run with compilation)

---

## Performance Optimization: Polychromatic Projection

### Key Insight
The number of distinct **materials** (27 regions) and **energy bins** (10-50) is fixed and small.
Only **ray tracing** and **reconstruction** scale with voxel count. The μ lookup should NOT.

### Current Implementation (src/Forward/Polychromatic.jl)
- Line 71: `μ_by_energy` is a small [27 × n_energies] lookup table (GOOD)
- Line 161: Creates full μ volume for EACH energy bin (INEFFICIENT)

```julia
# CURRENT (inefficient) - O(n_voxels × n_energies)
for e_idx in 1:n_energies
    μ_volume_flat = [μ_at_energy[mask_flat[i] + 1] for i in 1:length(mask_flat)]  # Creates n_voxel array
    samples = μ_volume_flat[linear_indices]  # Then gathers
    ...
end
```

### Optimized Approach
Gather mask values ONCE, then use direct indexing into the small μ table:

```julia
# OPTIMIZED - O(n_samples × n_energies), no temporary volume
mask_samples = UInt8.(mask_flat[linear_indices])  # Gather mask ONCE

for e_idx in 1:n_energies
    μ_samples = μ_at_energy[mask_samples .+ 1]  # Direct lookup, no temp array
    ...
end
```

### Memory Comparison (512×512×32 phantom, 10 energies, 360 angles)
- Current: Creates 10 × 8.4M = 84M element temp arrays
- Optimized: Zero temp arrays, just index into 27×10 = 270 element table

### Implementation Status
- [x] Optimize `forward_project_polychromatic` to gather mask samples once
- [x] Tests pass (564/564) - results identical to previous implementation
- [ ] Ensure XLA compatibility with optimized version (future)

---

## Clinical-Scale Architecture: Angle-Batched Processing

### The Problem
The standard pre-computed geometry approach stores indices/weights for ALL angles:
- `ProjectionGeometry`: [n_cols × n_rows × n_angles × max_samples]
- `BackprojectionGeometry`: [4 × nx × ny × nz × n_angles]

For clinical scale (1000 angles, 1000 cols, 512³ volume):
- Projection: 1000 × 64 × 1000 × 1500 = **96 trillion elements** = impossible
- Backprojection: 4 × 512 × 512 × 320 × 1000 = **335 billion elements** = impossible

### The Solution: Angle-Batched Processing
Process one angle (or small batch) at a time:
1. Pre-compute geometry for ONE angle (~1-2 GB)
2. XLA-compile the single-angle kernel
3. Loop over angles, accumulating results
4. Geometry gets GC'd between angles

Memory per angle:
- Projection: 1000 × 64 × 1500 = 96M elements = **~1.5 GB** (manageable!)
- Backprojection: 4 × 512 × 512 × 320 = 335M elements = **~5 GB** (manageable!)

### Implementation Files
- `src/Forward/ClinicalProjector.jl` - Angle-batched forward projection
- `src/Reconstruction/ClinicalFDK.jl` - Angle-batched FDK reconstruction

### Key Functions
```julia
# Forward projection (clinical scale)
sinogram = forward_project_clinical(phantom, geom; verbose=true)

# FDK reconstruction (clinical scale)
volume = fdk_reconstruct_clinical(sinogram, geom, output_size, fov; verbose=true)

# XLA-compilable kernels (for user-side Reactant integration)
project_single_angle(volume_flat, indices, weights)
backproject_single_angle(sino_flat, indices, bilinear_w, distance_w)
```

### Usage Pattern for Reactant Compilation
```julia
using Reactant

# Compile kernel once
volume_ra = Reactant.to_rarray(volume_flat)
indices_ra = Reactant.to_rarray(angle_geom.linear_indices)
weights_ra = Reactant.to_rarray(angle_geom.sample_weights)
compiled_project = @compile project_single_angle(volume_ra, indices_ra, weights_ra)

# Reuse for each angle
for angle_idx in 1:n_angles
    angle_geom = precompute_single_angle_geometry(geom, phantom, angle_idx)
    indices_ra = Reactant.to_rarray(angle_geom.linear_indices)
    weights_ra = Reactant.to_rarray(angle_geom.sample_weights)
    projection = Array(compiled_project(volume_ra, indices_ra, weights_ra))
    sinogram[:, :, angle_idx] = projection
end
```

### Trade-offs
| Approach | Memory | Speed | XLA-Compatible |
|----------|--------|-------|----------------|
| Full pre-computed | O(angles × detector × samples) | Fastest | Yes (if fits) |
| Angle-batched | O(detector × samples) | Slower | Yes |
| On-the-fly | O(1) | Slowest | Harder |

The angle-batched approach is the sweet spot for clinical scale: bounded memory with XLA compilation.

---

## Ray Marching Architecture (NEW - Optimal Approach)

### The Problem with Angle-Batched
While angle-batched processing bounds memory, it still requires:
- One kernel call per angle (e.g., 984 kernel launches for clinical CT)
- Pre-computed geometry arrays (~500MB-1GB per angle)
- Significant kernel launch overhead

### The Solution: Ray Marching with On-the-Fly Geometry
Compute geometry **inside** the kernel from compact parameters:
- Store only ray origins/directions (~600MB total for all rays)
- **ONE kernel call** for ALL angles
- Fixed-step ray marching with trilinear interpolation

### Memory Comparison

| Approach | Geometry Storage | Kernel Calls (360 angles) |
|----------|-----------------|---------------------------|
| Full pre-computed | ~500GB | 1 |
| Angle-batched | ~500MB/angle | 360 |
| **Ray marching** | **~600MB total** | **1** |

### Implementation Files
- `src/Forward/RayMarching.jl` - Ray marching forward projection
- `src/Reconstruction/RayMarchingBackproj.jl` - Ray marching backprojection

### Key Functions

```julia
# Forward Projection (ray marching)
ray_geom = compute_ray_geometry(geom, fov, volume_size)
sinogram = forward_project_raymarching_vectorized(
    volume, ray_geom.origins, ray_geom.directions,
    ray_geom.vol_min, voxel_size, n_samples, step_size
)

# Backprojection (ray marching)
bp_geom = compute_backproj_geometry(geom)
volume = backproject_raymarching_kernel(
    filtered_sinogram,
    bp_geom.source_positions, bp_geom.detector_centers,
    bp_geom.detector_u, bp_geom.detector_v, bp_geom.sd_axis,
    voxel_x, voxel_y, voxel_z,
    bp_geom.SAD, bp_geom.SDD, bp_geom.pixel_size_det, bp_geom.delta_angle,
    bp_geom.n_cols, bp_geom.n_rows
)

# FDK Reconstruction (complete)
recon = fdk_reconstruct_raymarching(sinogram, geom, output_size, fov)
```

### Ray Marching Parameters

```julia
# Step size: typically 0.5 × minimum voxel size
min_voxel = minimum(voxel_size)
step_size = Float32(min_voxel * 0.5)

# Number of samples: enough to traverse volume diagonal
diagonal = sqrt(sum(fov .^ 2))
n_samples = ceil(Int, diagonal / step_size) + 10
```

### Accuracy
Ray marching with step_size = 0.5 × voxel_size achieves accuracy indistinguishable from Siddon's exact method. The error is well below quantum noise levels. Many commercial scanners use similar approximate methods.

### Reactant Compilation (Future)
The ray marching kernels are designed for Reactant.@compile:

```julia
using Reactant

# Convert to Reactant arrays
vol_ra = Reactant.to_rarray(volume)
origins_ra = Reactant.to_rarray(ray_geom.origins)
dirs_ra = Reactant.to_rarray(ray_geom.directions)

# Compile once, run many times
compiled_proj = @compile forward_project_raymarching_compiled(
    vol_ra, origins_ra, dirs_ra,
    vol_min_x, vol_min_y, vol_min_z,
    voxel_size_x, voxel_size_y, voxel_size_z,
    n_samples, step_size
)
```

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

# Array-based functions for Reactant compilation
# (avoids embedding structs as constants)
proj_geom = precompute_projection_geometry(geom, fov, voxel_size, output_size)
bp_geom = precompute_backprojection_geometry(geom, output_size, fov)

# Compile with Reactant (extract arrays, pass directly)
vol_ra = Reactant.to_rarray(volume)
proj_indices_ra = Reactant.to_rarray(proj_geom.linear_indices)
proj_weights_ra = Reactant.to_rarray(Float32.(proj_geom.sample_weights))
compiled_proj = @compile project_volume_arrays(vol_ra, proj_indices_ra, proj_weights_ra)

sino_ra = Reactant.to_rarray(vec(sinogram))
bp_indices_ra = Reactant.to_rarray(bp_geom.linear_indices)
bp_bilinear_ra = Reactant.to_rarray(bp_geom.bilinear_weights)
bp_distance_ra = Reactant.to_rarray(bp_geom.distance_weights)
compiled_bp = @compile backproject_volume_arrays(sino_ra, bp_indices_ra, bp_bilinear_ra, bp_distance_ra)

# Iterative Reconstruction (XLA-compatible)
# Use struct-based functions for convenience (slower, not compiled)
result = sirt_reconstruct(sino, proj_geom, bp_geom; n_iterations=50)
recon = result.volume

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
4. ✅ Build out Pluto notebook showing realistic CT simulation
5. ⏳ Begin building out DukeSim parity

---

## Session Continuation Notes

When continuing a session:
1. Run `git status` to check current state
2. Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
3. Check TODO status in this file
4. Consult CatSim source code before implementing new features
5. Commit and push incrementally

Last updated: 2026-01-13
Current test count: 564
Phase 1 (CatSim Parity): COMPLETE
Phase 2 (Differentiable Reconstruction): COMPLETE
Additional features: Tang 3D weighting, optical crosstalk, fill factors, flying focal spot, iterative recon (SIRT, CGLS)
XLA-Compatible: forward_project (project_volume), fdk_reconstruct_xla (backproject_volume), sirt_reconstruct, cgls_reconstruct
Clinical Scanners: GE Revolution Apex Elite (K213715) with FDA 510(k) sourced parameters

**NEW: Ray Marching Architecture**
- `src/Forward/RayMarching.jl` - Single-kernel forward projection for ALL angles
- `src/Reconstruction/RayMarchingBackproj.jl` - Single-kernel backprojection for ALL angles
- Key functions: `compute_ray_geometry()`, `forward_project_raymarching_vectorized()`, `backproject_raymarching_kernel()`, `fdk_reconstruct_raymarching()`
- Pluto notebook updated to use ray marching approach

---

## Known Issues: Iterative Reconstruction (SIRT/CGLS)

### Current Status
- **FDK**: ✅ Works correctly (Water HU ≈ 0, all materials accurate)
- **SIRT**: ⚠️ Overshoots progressively with iterations
- **CGLS**: ⚠️ Converges slowly to wrong solution

### SIRT Overshoot Data
| Iterations | Water HU | Overshoot |
|------------|----------|-----------|
| 1 | -119 | -11.9% |
| 2 | 30 | +3.0% |
| 5 | 114 | +11.4% |
| 10 | 154 | +15.4% |
| 20 | 167 | +16.7% |

### Root Cause: Operator Mismatch
The forward and backward projection operators are not proper adjoints:

1. **Forward Projection** (`forward_project_raymarching_vectorized`):
   - Ray-driven: shoots rays from source through detector pixels
   - Accumulates μ values along ray path with trilinear interpolation
   - Each sample weighted by `step_size`

2. **Backprojection** (`backproject_raw_kernel`):
   - Voxel-driven: for each voxel, finds corresponding detector position
   - Samples sinogram with bilinear interpolation
   - No step_size weighting

For iterative methods to converge correctly, we need `<Ax, y> = <x, A^T y>` (adjoint property).
Currently this doesn't hold because the operators use different traversal strategies.

### Fix Options (Priority Order)

**Option 1: Matched Ray-Driven Operators** (Recommended)
- Implement ray-driven backprojection that mirrors forward projection
- For each ray (source → detector pixel), distribute sinogram value back to voxels along ray
- Use same step_size and interpolation as forward projection
- Pros: True adjoint, CGLS will converge correctly
- Cons: More memory access patterns, potentially slower

**Option 2: Matched Voxel-Driven Operators**
- Implement voxel-driven forward projection
- For each voxel, find all detector pixels it projects to
- Pros: Simpler geometry
- Cons: Harder to parallelize, may miss some rays

**Option 3: Normalization Correction**
- Keep mismatched operators but add correction factors
- Compute `D = A^T A` diagonal approximation
- Scale SIRT/CGLS updates by inverse of D
- Pros: No algorithm changes
- Cons: Approximate, may not fully fix convergence

### Implementation Plan
1. **Phase 1** (Current): FDK works, SIRT/CGLS marked experimental
2. **Phase 2**: Implement ray-driven backprojection (`backproject_raydriven_kernel`)
3. **Phase 3**: Test SIRT/CGLS with matched operators
4. **Phase 4**: Optimize ray-driven backprojection for performance

### Files to Modify
- `src/Reconstruction/RayMarchingBackproj.jl`: Add `backproject_raydriven_kernel`
- `src/API.jl`: Update `_reconstruct_sirt` and `_reconstruct_cgls` to use matched operators
- `test/runtests.jl`: Add convergence tests for iterative methods

### Reference: Ray-Driven Backprojection Algorithm
```
For each angle:
    For each detector pixel (col, row):
        ray_origin = source_position[angle]
        ray_direction = normalize(detector_pixel_position - ray_origin)
        sinogram_value = sinogram[col, row, angle]

        For each sample point along ray:
            voxel_coords = ray_origin + t * ray_direction
            # Distribute sinogram_value to nearby voxels using trilinear weights
            # Weight by step_size (same as forward projection)
```

This ensures the backprojection is the true transpose of forward projection.
