# BasisSimulator.jl - Development Planning and Implementation Status

**Last Updated**: January 8, 2026
**Target**: Medical Physics journal publication
**Version**: 0.1.0 (active development)

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Current Status](#current-status)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Testing Strategy](#testing-strategy)
6. [GECATSIM Validation](#gecatsim-validation)
7. [Development Roadmap](#development-roadmap)
8. [Publication Requirements](#publication-requirements)
9. [Contributing Guidelines](#contributing-guidelines)

---

## Project Goals

### Primary Objective
Create a publication-grade, fully differentiable CT simulator for inverse problems, validated against GECATSIM and suitable for Medical Physics journal publication.

### Key Claims for Publication
1. **First fully differentiable CT simulator** with end-to-end gradients through complete imaging chain
2. **Reactant/XLA compilation** for 10-100x speedup on GPU/TPU
3. **Validated against GECATSIM** (NIH standard CT simulator)
4. **Complete physics models** with peer-reviewed citations
5. **Enables novel applications**: gradient-based material decomposition and dose optimization

### Target Journal
**Medical Physics** (Official Journal of AAPM)
- Impact Factor: 3.8
- Requires rigorous validation and citations
- Code availability requirement (GitHub release)
- Supplementary data requirements (validation results)

---

## Current Status

### Implemented Modules (v0.1.0)

#### Physics
- **Spectrum.jl** (487 lines) - X-ray spectrum generation
  - Status: COMPLETE
  - Kramers' Law bremsstrahlung (Boone & Seibert 1997)
  - Tungsten K-lines (59.3 keV, 67.2 keV)
  - Energy-dependent filtration
  - Heel effect modeling
  - Test coverage: 98 tests passing

- **Attenuation.jl** - Material interaction
  - Status: COMPLETE
  - XrayAttenuation.jl integration (NIST XCOM database)
  - Energy-dependent mass/linear attenuation coefficients
  - Polychromatic attenuation computation
  - Test coverage: Integrated with physics validation tests

#### Geometry
- **ScannerGeometry.jl** (593 lines) - CT scanner configurations
  - Status: COMPLETE
  - Canon Aquilion ONE (320×800 detector, SAD=600mm, SDD=1000mm)
  - Pre-computed source/detector trajectories
  - Custom geometry support
  - Test coverage: 728 tests passing (geometry constraints)

- **RayTracing.jl** (721 lines) - Amanatides-Woo ray tracer
  - Status: COMPLETE
  - Pure functional (Reactant-compatible)
  - Material path length accumulation
  - Density-weighted radiological paths
  - Test coverage: Conservation laws validated

- **Phantoms.jl** - Digital phantom generation
  - Status: COMPLETE
  - Gammex 472 calibration phantom (7 Ca + 7 I inserts)
  - Water cylinder (basic validation)
  - Memory-optimized (UInt8 IDs, Float32 densities)
  - Test coverage: 10 tests passing (phantom generation)

#### Reconstruction
- **FDK.jl** - Feldkamp-Davis-Kress algorithm
  - Status: COMPLETE
  - Ram-Lak, Shepp-Logan, Hann, Cosine filters
  - Parker weighting for short-scan
  - Hounsfield Unit conversion
  - Test coverage: 37 tests passing

#### Simulation
- **Simulation.jl** - End-to-end forward model
  - Status: COMPLETE
  - Polychromatic ray tracing
  - Material attenuation matrix pre-computation
  - Parallel over projection angles
  - Sub-pixel sampling (2×2)
  - Test coverage: Integrated validation

#### Validation
- **test_physics_validation.jl** (420 lines)
  - Status: COMPLETE
  - 790 tests passing
  - NIST attenuation coefficient validation
  - kVp dependence (beam hardening)
  - Hounsfield Unit calibration
  - Material contrast and separability
  - Spectrum physics validation
  - Ray tracing conservation laws

- **test_gecatsim_validation.jl**
  - Status: INFRASTRUCTURE READY
  - PythonCall.jl integration configured
  - Test structure in place
  - Awaiting GECATSIM installation for full validation

### In Progress

- **Custom Gammex Materials**: Need to define Ca_50, Ca_100, etc. as XA.Compound or XA.Mixture with proper elemental compositions
- **GECATSIM Cross-Validation**: Infrastructure ready, needs GECATSIM Python package
- **Performance Benchmarking**: Forward model timing and memory profiling

### Not Yet Implemented

#### Physics
- **Scatter.jl** - Compton/Rayleigh scatter
  - Klein-Nishina differential cross section
  - Convolution-based approximation
  - SPR (Scatter-to-Primary Ratio) estimation

- **Detector.jl** - Detector response
  - Quantum detection efficiency (QDE)
  - Modulation Transfer Function (MTF)
  - Point Spread Function (PSF)

- **Noise.jl** - Realistic noise models
  - Poisson (quantum) noise
  - Electronic noise (Gaussian)
  - 1/f noise (low-frequency drift)

#### Reconstruction
- **Iterative.jl** - Iterative algorithms
  - SIRT (Simultaneous Iterative Reconstruction)
  - MLEM (Maximum Likelihood Expectation Maximization)
  - Total Variation (TV) regularization

- **Corrections.jl** - Physics corrections
  - Beam hardening correction
  - Scatter correction
  - Ring artifact removal

---

## Architecture

### Module Organization

```
src/
├── BasisSimulator.jl          # Main module, public API
├── Physics/
│   ├── Spectrum.jl            # X-ray source [COMPLETE]
│   ├── Attenuation.jl         # Material interaction [COMPLETE]
│   ├── Scatter.jl             # [TODO]
│   ├── Detector.jl            # [TODO]
│   └── Noise.jl               # [TODO]
├── Geometry/
│   ├── ScannerGeometry.jl     # Scanner config [COMPLETE]
│   ├── RayTracing.jl          # Ray tracer [COMPLETE]
│   └── Phantoms.jl            # Phantoms [COMPLETE]
├── Reconstruction/
│   ├── FDK.jl                 # Cone-beam FDK [COMPLETE]
│   ├── Iterative.jl           # [TODO]
│   └── Corrections.jl         # [TODO]
├── Simulation.jl              # Forward model [COMPLETE]
└── Validation/                # [EMPTY - future metrics]
```

### Design Principles

1. **Pure Functional**: All functions are stateless (Reactant requirement)
2. **Type Stable**: Avoid type instabilities for performance
3. **Documented**: Every exported function has docstring with citations
4. **Tested**: Unit tests with known solutions + integration tests
5. **Validated**: Cross-checked against NIST data and GECATSIM

---

## Implementation Details

### Data Types

#### Core Structures

```julia
struct XRaySpectrum
    energies::Vector{Float64}      # keV bins
    photons::Vector{Float64}       # Photons per bin
    kVp::Float64                   # Peak voltage
    mAs::Float64                   # Tube current-time product
end

struct ScanProtocol
    kVp::Float64                   # Peak voltage (kV)
    mAs::Float64                   # Tube current-time (mAs)
    scan_fov_mm::Float64           # Field of view (mm)
    num_projections::Int           # Number of projection angles
end

struct CTGeometry
    SDD_cm::Float64                # Source-detector distance
    SAD_cm::Float64                # Source-axis distance
    n_rows::Int                    # Detector rows
    n_cols::Int                    # Detector columns
    pixel_width_cm::Float64        # Detector pixel width
    pixel_height_cm::Float64       # Detector pixel height
    angles::Vector{Float64}        # Projection angles (radians)
    source_positions::Matrix{Float64}    # [3, N]
    det_centers::Matrix{Float64}         # [3, N]
    det_u_vecs::Matrix{Float64}          # [3, N]
    det_v_vecs::Matrix{Float64}          # [3, N]
end

struct PhantomData
    name::String
    grid::VoxelGrid
    material_ids::Array{UInt8, 3}        # Material ID per voxel
    densities::Array{Float32, 3}          # Density (g/cm³)
    id_to_material::Dict{UInt8, Symbol}   # ID → material name
end
```

### Key Algorithms

#### 1. Spectrum Generation (Spectrum.jl)

**Kramers' Law**: Bremsstrahlung continuum
```julia
dN/dE ∝ Z × (E_max - E) / E
```

**Characteristic Lines**: Tungsten K-α and K-β
```
K-α1: 59.318 keV (intensity: 100)
K-α2: 57.982 keV (intensity: 58)
K-β1: 67.244 keV (intensity: 29)
K-β2: 69.101 keV (intensity: 8)
```

**Filtration**: Exponential attenuation through Al, Cu
```julia
I(E) = I₀(E) × exp(-μ_Al(E) × t_Al - μ_Cu(E) × t_Cu)
```

**Citations**:
- Boone & Seibert (1997) Med Phys 24(11):1661-1670
- Tucker et al. (1991) Med Phys 18(2):211-218

#### 2. Ray Tracing (RayTracing.jl)

**Amanatides-Woo Algorithm**: 3D-DDA voxel traversal
```julia
# Initialize t_max for each axis
t_max_x = (next_x_plane - ray_origin_x) / ray_direction_x
t_max_y = (next_y_plane - ray_origin_y) / ray_direction_y
t_max_z = (next_z_plane - ray_origin_z) / ray_direction_z

# Step through grid
while inside_grid
    if t_max_x < t_max_y && t_max_x < t_max_z
        # Cross X plane
        x += step_x
        t_max_x += Δt_x
    elseif t_max_y < t_max_z
        # Cross Y plane
        y += step_y
        t_max_y += Δt_y
    else
        # Cross Z plane
        z += step_z
        t_max_z += Δt_z
    end
    accumulate_path_length()
end
```

**Citations**:
- Amanatides & Woo (1987) Eurographics 87(3):3-10
- Siddon (1985) Med Phys 12(2):252-255

#### 3. FDK Reconstruction (FDK.jl)

**Feldkamp-Davis-Kress Algorithm**:
```
1. Distance weighting: p_w(u,v,β) = (SAD/√(SAD² + u² + v²)) × p(u,v,β)
2. Filter in u-direction: q(u,v,β) = p_w(u,v,β) ⊗ h_filter(u)
3. Backproject with cone-beam weighting
```

**Citations**:
- Feldkamp et al. (1984) JOSA A 1(6):612-619
- Parker (1982) Med Phys 9(2):254-257

---

## Testing Strategy

### Test Hierarchy

1. **Unit Tests**: Individual functions with known solutions
2. **Integration Tests**: Full pipeline validation
3. **Physics Validation**: Against NIST data and principles
4. **GECATSIM Comparison**: Cross-simulator validation

### Physics Validation Tests (test_physics_validation.jl)

**790 tests covering**:

1. **NIST Attenuation Coefficients** (10 tests)
   - Water at 60, 80, 120 keV: μ within 5% of NIST XCOM
   - Bone, iodine validation at clinical energies
   - Higher energy → lower attenuation

2. **kVp Dependence** (6 tests)
   - Beam hardening: μ_80kVp > μ_120kVp > μ_140kVp
   - High-Z materials (iodine) show stronger dependence
   - Polychromatic effective attenuation

3. **Hounsfield Unit Calibration** (6 tests)
   - Water = 0 HU (by definition)
   - Air ≈ -1000 HU
   - Bone > 1000 HU
   - Iodine > 10000 HU (pure)

4. **Material Contrast** (6 tests)
   - Calcium concentration linearity
   - Iodine concentration linearity
   - Material separability at 60 keV

5. **X-ray Spectrum Physics** (10 tests)
   - Energy range: E_min > 0, E_max ≤ kVp
   - K-alpha lines present near 59.3 keV
   - Higher kVp → harder spectrum (higher mean E)
   - mAs scales fluence linearly

6. **Polychromatic Effects** (1 test)
   - Polychromatic μ < monochromatic μ at mean energy

7. **Scanner Geometry** (728 tests)
   - Canon Aquilion ONE specs validation
   - Trajectory smoothness and consistency
   - Magnification calculations

8. **Phantom Generation** (10 tests)
   - Material count verification
   - Density distribution checks
   - Memory footprint validation

9. **Ray Tracing Physics** (1 test)
   - Path length conservation
   - Total path ≈ source-detector distance

### Test Philosophy

- **Absolute tests** when literature values known (NIST data)
- **Relative tests** when only trends known (kVp effects)
- **Reasonable tolerances** (5-15%) for polychromatic spectra
- **Physical plausibility** checks (0 < QDE ≤ 1, etc.)

---

## GECATSIM Validation

### Infrastructure Status

**Completed**:
- PythonCall.jl integration configured in Project.toml
- test_gecatsim_validation.jl structure in place
- Graceful skipping if GECATSIM not installed

**Installation Required**:
```bash
pip install gecatsim
```

### Validation Plan

#### Phase 1: Attenuation Coefficients
- Compare BasisSimulator and GECATSIM attenuation against NIST XCOM
- Target: < 1% difference (both should match XCOM)

#### Phase 2: Simple Phantom (Water Cylinder)
- 200mm diameter, 40mm height
- 120 kVp, 200 mAs
- Compare:
  - Sinogram RMSE < 5%
  - Reconstruction SSIM > 0.95
  - HU values: water ≈ 0 HU in both

#### Phase 3: Gammex 472 Phantom
- Matching insert compositions
- Multiple kVp (80, 100, 120, 140)
- Compare:
  - Insert HU values
  - Contrast-to-noise ratio (CNR)
  - Beam hardening artifacts

#### Phase 4: Noise Statistics
- Low dose (50 mAs) vs high dose (400 mAs)
- Verify: σ_low / σ_high ≈ √(400/50) = 2.83
- Compare quantum noise distribution

### Validation Metrics

```julia
struct ValidationMetrics
    sinogram_rmse::Float64          # < 5% target
    sinogram_mae::Float64
    image_ssim::Float64             # > 0.95 target
    image_psnr::Float64
    hu_water_diff::Float64          # < 10 HU
    hu_bone_diff::Float64
    noise_ratio::Float64
end
```

### Visual Validation

**Location**: `test/visual_validation.jl`

**Purpose**: Generate publication-quality comparison figures for qualitative validation against GECATSIM.

**Output**: Single PNG file with comprehensive 8-panel comparison:
- Row 1: Sinogram comparisons (BasisSimulator, GECATSIM, Difference)
- Row 2: Reconstruction comparisons (BasisSimulator, GECATSIM, Difference)
- Row 3: Quantitative analysis (HU profiles, scatter plots, metrics)

**Usage**:
```bash
julia --project=. test/visual_validation.jl
```

**Output Location**:
```
test/outputs/visual_comparison.png
```

**Features**:
- Automatic detection of GECATSIM availability
- Falls back to BasisSimulator-only validation if GECATSIM not installed
- High-resolution output (2x pixel density for publication)
- Quantitative metrics displayed (RMSE, MAE)
- HU profile comparisons
- Pixel-by-pixel correlation analysis

**Requirements** (optional, auto-detected):
- CairoMakie.jl (visualization)
- PythonCall.jl (GECATSIM integration)
- CondaPkg.jl (Python environment)

**Validation Workflow**:
1. Run BasisSimulator forward simulation
2. If GECATSIM available, run equivalent simulation
3. Generate multi-panel comparison figure
4. Save to `test/outputs/`
5. Inspect visually for agreement

**What to Look For**:
- **Sinogram agreement**: Should have similar structure and intensity distribution
- **Reconstruction agreement**: HU values should match within ±10 HU for water
- **Difference maps**: Should show random noise, not systematic patterns
- **HU profiles**: Should overlap closely across phantom center
- **Scatter plot**: Should cluster tightly around y=x identity line

**Integration with CI/CD**:
- Currently manual (requires visual inspection)
- Future: Automated SSIM/RMSE checks with pass/fail thresholds
- Figures can be archived as artifacts for peer review

**Current Limitations**:
- Uses water cylinder phantom only (simple validation baseline)
- Gammex 472 custom materials (Ca_50, Ca_100, I_2_0, etc.) not yet defined
  - Need to create XA.Compound or XA.Mixture definitions with proper elemental compositions
  - Required for meaningful contrast in calibration phantom validation
- Full GECATSIM Python interop not yet implemented (placeholder code in place)
- Visual validation currently falls back to BasisSimulator-only mode

**Next Steps**:
1. Define custom Gammex materials in XrayAttenuation.jl format
2. Implement full GECATSIM Python interop for run_gecatsim()
3. Generate actual BasisSimulator vs GECATSIM comparison figures
4. Document validation results with quantitative metrics

---

## Development Roadmap

### Phase 1: Core Implementation (COMPLETE)
- Spectrum generation
- Material attenuation
- Ray tracing
- Scanner geometry
- Phantoms
- FDK reconstruction
- Forward simulation pipeline
- Physics validation tests

### Phase 2: Physics Extensions (Weeks 2-3)
**Priority**: Scatter → Detector → Noise

**Scatter.jl**:
- Klein-Nishina differential cross section
- Single-scatter approximation
- Convolution-based scatter estimation
- SPR (Scatter-to-Primary Ratio) tuning
- Citations: Klein & Nishina (1929), Siewerdsen et al. (2006)

**Detector.jl**:
- Quantum detection efficiency vs energy
- MTF modeling (Fujita 1992, Samei 1998)
- PSF convolution
- Charge sharing effects
- Citations: Fujita et al. (1992) IEEE TMI

**Noise.jl**:
- Poisson quantum noise
- Electronic noise (Gaussian)
- NPS (Noise Power Spectrum) computation
- Citations: Barrett & Myers (2004), Wagner et al. (1999)

### Phase 3: Advanced Reconstruction (Week 4)
**Priority**: Iterative → Corrections

**Iterative.jl**:
- SIRT (Simultaneous Iterative Reconstruction)
- MLEM (Maximum Likelihood)
- TV (Total Variation) regularization
- All Enzyme-differentiable

**Corrections.jl**:
- Beam hardening correction (water, bone)
- Scatter correction (convolution subtraction)
- Ring artifact removal

### Phase 4: GECATSIM Validation (Week 5)
- Install GECATSIM Python package
- Run Phase 1-4 validation plan (see above)
- Document all comparison results
- Generate supplementary figures for publication

### Phase 5: Examples & Documentation (Week 6)
**Examples**:
- Basic CT simulation
- Material decomposition (dual-energy)
- Dose optimization (gradient-based)
- Geometry calibration

**Documentation**:
- API reference (Documenter.jl)
- Physics model details
- Validation report
- Tutorial notebooks

### Phase 6: Performance Optimization (Week 7)
- Reactant compilation profiling
- GPU benchmarking
- Memory optimization
- Parallel scaling tests

### Phase 7: Manuscript Preparation (Week 8)
- Write Methods section (detailed algorithms)
- Generate all validation figures
- Create supplementary materials
- Code release preparation (GitHub)

---

## Publication Requirements

### Medical Physics Journal Standards

#### Methods Section
- Complete algorithm descriptions
- All physics equations with citations
- Validation methodology
- Computational environment details

#### Results Section
- GECATSIM comparison (sinogram, images, HU values)
- Physics validation (spectrum, attenuation, noise)
- Performance benchmarks (timing, memory)
- Example applications (material decomp, dose optimization)

#### Discussion
- Novel contribution: full differentiability
- Comparison to existing CT simulators
- Limitations and future work
- Clinical applications

#### Code Availability
- GitHub repository release (v1.0.0)
- Zenodo DOI for archival
- Documentation website
- Example datasets

#### Supplementary Materials
- Detailed validation results
- Additional figures
- Code examples
- Raw data (HDF5 format)

### Required Citations (Minimum)

**X-ray Physics**:
- Boone & Seibert (1997) - spectrum generation
- Hubbell (1999) - attenuation coefficients
- Klein & Nishina (1929) - Compton scattering

**CT Reconstruction**:
- Feldkamp et al. (1984) - FDK algorithm
- Kak & Slaney (1988) - CT principles

**Validation**:
- GECATSIM paper (De Man et al. 2022)
- NIST XCOM database

**Automatic Differentiation**:
- Enzyme.jl citation
- Reactant.jl/XLA citations

---

## Contributing Guidelines

### Code Standards

1. **Formatting**: Use JuliaFormatter.jl with default settings
2. **Naming**:
   - Functions: `snake_case`
   - Types: `PascalCase`
   - Constants: `UPPER_SNAKE_CASE`
3. **Documentation**: All exported functions need docstrings
4. **Citations**: Include references in docstrings

### Testing Requirements

**For New Physics Models**:
- Unit tests with known analytical solutions
- Comparison against reference implementation
- Physical validity checks (e.g., 0 < QDE ≤ 1)
- GECATSIM comparison (if applicable)

**For New Features**:
- Integration tests with full pipeline
- Performance regression tests
- Documentation updates
- Example usage

### Pull Request Checklist

- [ ] All tests pass (`julia --project=. test/runtests.jl`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code formatted with JuliaFormatter
- [ ] Citations added for new physics models
- [ ] GECATSIM comparison (if physics model)
- [ ] Performance benchmarks (if core algorithm)

---

## Development Notes

### Known Issues

1. **Gammex Materials**: Need custom compound definitions for Ca_50, Ca_100, etc.
2. **GECATSIM Installation**: Requires manual Python package install for validation
3. **Reactant Compilation**: Not yet tested (Phase 6)
4. **Memory Usage**: Large phantoms (512³) may exceed 16GB target

### Design Decisions

**Why Pure Functional?**
- Reactant requirement (stateless)
- Enzyme compatibility (differentiable)
- Easier testing and parallelization

**Why XrayAttenuation.jl over Attenuations.jl?**
- Local NIST XCOM database (no HTTP requests)
- Better K-edge handling
- Full Unitful.jl integration

**Why GECATSIM for Validation?**
- NIH-validated industry standard
- Peer-reviewed physics models
- Establishes credibility for clinical use

**Why Medical Physics Journal?**
- Standard venue for CT simulation papers
- AAPM community relevance
- Rigorous peer review establishes quality

---

## Timeline Summary

- **Week 1**: Core implementation (COMPLETE)
- **Week 2-3**: Physics extensions (scatter, detector, noise)
- **Week 4**: Advanced reconstruction
- **Week 5**: GECATSIM validation
- **Week 6**: Examples & documentation
- **Week 7**: Performance optimization
- **Week 8**: Manuscript preparation

**Target Submission**: End of Week 8
**Expected Review**: 8-12 weeks
**Revision**: 2-4 weeks
**Publication**: ~6 months from submission

---

## Contact & Resources

**Repository**: https://github.com/MolloiLab/BasisSimulator.jl
**Issues**: https://github.com/MolloiLab/BasisSimulator.jl/issues
**Discussion**: GitHub Discussions (TBD)

**Key Dependencies**:
- XrayAttenuation.jl: https://github.com/Dale-Black/XrayAttenuation.jl
- GECATSIM: https://github.com/xcist/main
- Enzyme.jl: https://github.com/EnzymeAD/Enzyme.jl
- Reactant.jl: https://github.com/EnzymeAD/Reactant.jl

---

**Document Version**: 1.0
**Last Updated**: January 8, 2026
**Next Review**: After Phase 2 completion
