# BasisSimulator.jl Architecture

**A fully differentiable, Reactant-compilable CT simulator for inverse problems**

## Design Philosophy

1. **Pure Functional**: All physics models are pure functions (no side effects)
2. **Differentiable**: Full Enzyme.jl autodiff support through entire pipeline
3. **Compilable**: Reactant/XLA compatible for GPU/TPU acceleration
4. **Validated**: Systematic testing against GECATSIM ground truth
5. **Modular**: Clean separation of physics, geometry, and reconstruction
6. **Performant**: Optimized for both forward simulation and gradient computation

---

## Package Structure

```
BasisSimulator.jl/
├── src/
│   ├── BasisSimulator.jl          # Main module, exports public API
│   │
│   ├── Physics/
│   │   ├── Spectrum.jl            # X-ray source models
│   │   ├── Attenuation.jl         # Material interaction
│   │   ├── Scatter.jl             # Compton/Rayleigh scatter
│   │   ├── Detector.jl            # Response, MTF, PSF
│   │   └── Noise.jl               # Quantum + electronic noise
│   │
│   ├── Geometry/
│   │   ├── ScannerGeometry.jl     # CT scanner configurations
│   │   ├── RayTracing.jl          # Amanatides-Woo tracer
│   │   └── Phantoms.jl            # Phantom generators
│   │
│   ├── Reconstruction/
│   │   ├── FDK.jl                 # Feldkamp-Davis-Kress
│   │   ├── Iterative.jl           # SIRT, MLEM, MBIR
│   │   └── Corrections.jl         # Beam hardening, scatter
│   │
│   ├── Simulation.jl              # Forward model orchestration
│   │
│   └── Validation/
│       ├── Metrics.jl             # RMSE, SSIM, SNR, MTF
│       ├── GECATSIM.jl            # Python interop for validation
│       └── ReferenceData.jl       # Ground truth datasets
│
├── test/
│   ├── runtests.jl                # Test entry point
│   ├── test_physics.jl            # Unit tests for physics
│   ├── test_geometry.jl           # Ray tracing validation
│   ├── test_reconstruction.jl     # FDK accuracy tests
│   ├── test_gradients.jl          # Enzyme autodiff validation
│   └── test_gecatsim.jl           # GECATSIM comparison
│
├── examples/
│   ├── 01_basic_simulation.jl     # Hello world
│   ├── 02_material_decomp.jl      # Dual-energy separation
│   ├── 03_dose_optimization.jl    # Gradient-based dose tuning
│   ├── 04_geometry_calib.jl       # Scanner calibration
│   └── 05_iterative_recon.jl      # MBIR with gradients
│
├── docs/
│   ├── src/
│   │   ├── index.md               # Main documentation
│   │   ├── physics.md             # Physics models
│   │   ├── api.md                 # API reference
│   │   └── validation.md          # Validation results
│   └── make.jl                    # Documenter.jl build
│
├── Project.toml                   # Package dependencies
├── Manifest.toml                  # Locked versions
└── README.md                      # Quick start guide
```

---

## Module Responsibilities

### 1. Physics Module

**Purpose**: Implement all physical models with exact physics

#### `Spectrum.jl`
- Bremsstrahlung generation (Kramers' Law with Z-dependence)
- Characteristic X-rays (K-α, K-β lines for tungsten)
- Filtration (Al, Cu, tissue equivalent)
- Heel effect modeling
- Multiple kVp protocols (80, 100, 120, 140 kVp)
- Spectral validation against NIST data

```julia
struct XRaySpectrum
    energies::Vector{Float64}        # keV bins
    photons::Vector{Float64}         # Photons per bin
    kVp::Float64                     # Peak voltage
    mAs::Float64                     # Tube current-time
    filtration::NamedTuple           # Material thicknesses
end

function generate_spectrum(;
    kVp::Float64,
    mAs::Float64,
    anode_material::Symbol = :tungsten,
    anode_angle_deg::Float64 = 7.0,
    filtration_al_mm::Float64 = 4.0,
    filtration_cu_mm::Float64 = 0.1
) -> XRaySpectrum
```

#### `Attenuation.jl`
- Energy-dependent mass attenuation coefficients
- Photoelectric absorption (∝ Z³/E³)
- Compton scattering (Klein-Nishina)
- Coherent (Rayleigh) scattering
- Material composition handling
- NIST XCOM database integration

```julia
function compute_attenuation(
    material::Material,
    energy_keV::Float64
) -> Float64  # cm⁻¹
```

#### `Scatter.jl`
- Klein-Nishina differential cross section
- Single-scatter Monte Carlo (optional)
- Convolution-based approximation (fast)
- Scatter-to-Primary Ratio (SPR) estimation
- Angular distribution modeling

```julia
function klein_nishina_dcs(
    energy_incident::Float64,  # keV
    scatter_angle::Float64      # radians
) -> Float64  # cm²/sr

function estimate_scatter_convolution(
    primary_signal::Array{Float64, 3};
    spr_factor::Float64 = 0.15,
    kernel_sigma_mm::Float64 = 30.0
) -> Array{Float64, 3}
```

#### `Detector.jl`
- Quantum detection efficiency (QDE)
- Modulation Transfer Function (MTF)
- Point Spread Function (PSF)
- Energy-dependent response
- Charge sharing and cross-talk
- Detector element variations

```julia
struct DetectorModel
    material::Material              # GOS, CsI, etc.
    thickness_mm::Float64
    pixel_pitch_mm::Float64
    mtf::MTFModel
    psf::PSFModel
end

function compute_mtf(
    detector::DetectorModel,
    frequency_lp_mm::Vector{Float64}
) -> Vector{Float64}
```

#### `Noise.jl`
- Poisson (quantum) noise
- Electronic noise (Gaussian)
- 1/f noise (low-frequency drift)
- Quantization noise (ADC)
- Dead time losses

```julia
function apply_quantum_noise(
    photon_counts::Array{Float64};
    dose_factor::Float64 = 1.0
) -> Array{Float64}
```

---

### 2. Geometry Module

**Purpose**: Scanner configurations and ray tracing

#### `ScannerGeometry.jl`
- Pre-defined scanner models:
  - Canon Aquilion ONE (320-slice, 16cm)
  - GE Revolution CT (256-slice)
  - Siemens SOMATOM Definition Flash
  - Custom specifications
- Cone-beam geometry
- Circular and helical trajectories
- Detector array layout

```julia
struct CTScanner
    name::String
    sad_mm::Float64          # Source-to-axis distance
    sdd_mm::Float64          # Source-to-detector distance
    n_detector_rows::Int
    n_detector_cols::Int
    pixel_pitch_mm::Float64
    rotation_time_s::Float64
end

function create_aquilion_one() -> CTScanner
function create_custom_scanner(...) -> CTScanner
```

#### `RayTracing.jl`
- Amanatides-Woo grid traversal
- Reactant-compilable (pure functional)
- Siddon algorithm (alternative)
- Parallel beam projection (for validation)
- Path length accumulation per material

```julia
function trace_ray_material_paths(
    grid::GridMeta,
    material_ids::Array{UInt8, 3},
    densities::Array{Float32, 3},
    id_lut::Vector{Int},
    n_materials::Int,
    p1x::Float64, p1y::Float64, p1z::Float64,  # Source
    p2x::Float64, p2y::Float64, p2z::Float64   # Detector
) -> Vector{Float64}  # Path lengths per material
```

#### `Phantoms.jl`
- Gammex 472 (calibration phantom)
- XCAT (anatomical torso phantom)
- Custom geometric phantoms
- Material composition database
- Texture generation (density variations)

```julia
function create_gammex_472(;
    resolution_mm::Float64 = 0.5,
    z_coverage_mm::Float64 = 40.0
) -> PhantomData

function create_xcat_phantom(;
    gender::Symbol = :male,
    cardiac_phase::Int = 0
) -> PhantomData
```

---

### 3. Reconstruction Module

**Purpose**: Image reconstruction algorithms

#### `FDK.jl`
- Feldkamp-Davis-Kress cone-beam reconstruction
- Multiple filters:
  - Ram-Lak (ramp filter)
  - Shepp-Logan
  - Hann (Hamming)
  - Cosine
- Parker short-scan weighting
- Distance weighting for cone-beam
- Parallel and threaded implementations

```julia
function reconstruct_fdk(
    projections::Array{Float64, 3},
    geometry::CTGeometry,
    recon_grid::VoxelGrid;
    filter_type::Symbol = :ramlak,
    cutoff_freq::Float64 = 1.0
) -> Array{Float64, 3}
```

#### `Iterative.jl`
- SIRT (Simultaneous Iterative Reconstruction Technique)
- MLEM (Maximum Likelihood Expectation Maximization)
- MBIR (Model-Based Iterative Reconstruction)
- Total Variation (TV) regularization
- All Enzyme-differentiable for optimization

```julia
function reconstruct_sirt(
    projections::Array{Float64, 3},
    geometry::CTGeometry,
    n_iterations::Int = 20;
    λ_tv::Float64 = 0.01
) -> Array{Float64, 3}
```

#### `Corrections.jl`
- Beam hardening correction
- Scatter correction
- Ring artifact removal
- Metal artifact reduction (MAR)
- Motion correction

```julia
function correct_beam_hardening(
    projections::Array{Float64, 3},
    spectrum::XRaySpectrum,
    materials::Vector{Material}
) -> Array{Float64, 3}
```

---

### 4. Simulation Module

**Purpose**: High-level forward model orchestration

```julia
function simulate_ct_scan(;
    phantom::PhantomData,
    scanner::CTScanner,
    spectrum::XRaySpectrum,
    protocol::ScanProtocol,
    # Physics toggles
    enable_scatter::Bool = true,
    enable_noise::Bool = true,
    enable_detector_blur::Bool = true,
    # Compilation
    use_reactant::Bool = true
) -> CTScanData

struct CTScanData
    sinogram::Array{Float64, 3}
    reconstruction::Array{Float64, 3}
    metadata::Dict{Symbol, Any}
end
```

---

### 5. Validation Module

**Purpose**: Testing against GECATSIM and reference data

#### `GECATSIM.jl`
- PythonCall.jl integration
- Run equivalent GECATSIM simulations
- Export comparison data
- Automated validation pipeline

```julia
function run_gecatsim_reference(
    phantom_spec::Dict,
    scanner_spec::Dict,
    output_dir::String
) -> GECATSIMResults

function compare_with_gecatsim(
    our_result::CTScanData,
    gecatsim_result::GECATSIMResults
) -> ValidationMetrics
```

#### `Metrics.jl`
- RMSE (Root Mean Square Error)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- MTF measurement
- Noise Power Spectrum (NPS)
- Material accuracy (HU units)

```julia
function compute_rmse(
    predicted::Array{Float64},
    truth::Array{Float64}
) -> Float64

function compute_ssim(
    img1::Matrix{Float64},
    img2::Matrix{Float64}
) -> Float64
```

---

## Testing Strategy

### Unit Tests

**Every physics function must have:**
1. Known input → expected output tests
2. Physical validity checks (e.g., QDE ∈ [0,1])
3. Symmetry/conservation tests
4. Edge case handling

**Example:**
```julia
@testset "Spectrum Generation" begin
    spec = generate_spectrum(kVp=120.0, mAs=200.0)

    # Test 1: Energy range
    @test minimum(spec.energies) > 0
    @test maximum(spec.energies) ≤ 120.0

    # Test 2: Normalization
    @test sum(spec.photons) > 0

    # Test 3: K-lines present
    kalpha_idx = findfirst(e -> abs(e - 59.3) < 0.5, spec.energies)
    @test !isnothing(kalpha_idx)
    @test spec.photons[kalpha_idx] > mean(spec.photons)
end
```

### Integration Tests

**Full pipeline tests:**
```julia
@testset "End-to-End Simulation" begin
    # Simple phantom
    phantom = create_water_cylinder(radius_cm=10.0)
    scanner = create_aquilion_one()
    spec = generate_spectrum(kVp=120.0, mAs=200.0)

    # Run simulation
    result = simulate_ct_scan(
        phantom=phantom,
        scanner=scanner,
        spectrum=spec
    )

    # Test 1: No NaN/Inf
    @test all(isfinite, result.reconstruction)

    # Test 2: HU calibration (water ≈ 0 HU)
    center_hu = result.reconstruction[256, 256, 32]
    @test abs(center_hu - 0.0) < 10.0  # Within 10 HU
end
```

### Gradient Tests

**Enzyme autodiff validation:**
```julia
@testset "Gradient Accuracy" begin
    # Simple function: forward pass
    function loss(densities)
        phantom = update_densities(PHANTOM, densities)
        sino = simulate_ct_scan(phantom=phantom, ...)
        return sum(sino)
    end

    # Enzyme gradient
    grad_enzyme = gradient(loss, densities)

    # Finite differences
    ε = 1e-6
    grad_fd = finite_difference_gradient(loss, densities, ε)

    # Test: Relative error < 1%
    rel_error = norm(grad_enzyme - grad_fd) / norm(grad_fd)
    @test rel_error < 0.01
end
```

### GECATSIM Comparison Tests

**Validation against ground truth:**
```julia
@testset "GECATSIM Validation" begin
    # Run both simulators on identical setup
    gecatsim_result = run_gecatsim_reference(
        phantom="Gammex472",
        scanner="AquilionONE",
        kVp=120.0
    )

    our_result = simulate_ct_scan(
        phantom=create_gammex_472(),
        scanner=create_aquilion_one(),
        spectrum=generate_spectrum(kVp=120.0, mAs=200.0)
    )

    # Sinogram comparison
    sino_rmse = compute_rmse(
        our_result.sinogram,
        gecatsim_result.sinogram
    )
    @test sino_rmse < 0.1  # Within 10% error

    # Image comparison
    img_ssim = compute_ssim(
        our_result.reconstruction[:,:,32],
        gecatsim_result.reconstruction[:,:,32]
    )
    @test img_ssim > 0.95  # SSIM > 0.95
end
```

---

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Forward simulation (512³ + 360 views) | < 10 seconds | TBD |
| Reactant compilation time | < 30 seconds | ~15 seconds |
| Gradient computation overhead | < 2x forward pass | TBD |
| Memory usage (512³ phantom) | < 16 GB RAM | TBD |
| GECATSIM sinogram RMSE | < 5% | TBD |
| GECATSIM image SSIM | > 0.95 | TBD |

---

## Roadmap

### Phase 1: Core Package Setup (Week 1)
- ✅ Directory structure
- ⏳ Module stubs with documentation
- ⏳ Test framework setup
- ⏳ CI/CD configuration

### Phase 2: Physics Implementation (Week 2-3)
- ⏳ Spectrum models with validation
- ⏳ Klein-Nishina scatter
- ⏳ MTF/PSF detector models
- ⏳ Comprehensive noise models

### Phase 3: Reconstruction (Week 4)
- ⏳ Advanced FDK filters
- ⏳ Iterative algorithms (SIRT, MLEM)
- ⏳ Beam hardening correction

### Phase 4: Validation (Week 5-6)
- ⏳ GECATSIM integration
- ⏳ Reference dataset comparison
- ⏳ Gradient validation
- ⏳ Performance benchmarking

### Phase 5: Documentation & Examples (Week 7)
- ⏳ API documentation
- ⏳ Physics model documentation
- ⏳ Tutorial examples
- ⏳ Validation report

---

## Dependencies

**Core Physics:**
- `Attenuations.jl` - Mass attenuation coefficients (NIST)
- `Unitful.jl` - Physical units
- `Interpolations.jl` - Energy interpolation

**Differentiation:**
- `Enzyme.jl` - Automatic differentiation
- `Reactant.jl` - XLA compilation

**Numerics:**
- `FFTW.jl` - Fourier transforms (FDK filtering)
- `LinearAlgebra.jl` - Matrix operations
- `Statistics.jl` - Statistical metrics

**Validation:**
- `PythonCall.jl` - GECATSIM integration
- `HDF5.jl` - Data I/O
- `Test.jl` - Testing framework

**Visualization:**
- `CairoMakie.jl` - Plotting
- `Colors.jl` - Colormaps

---

## Design Decisions

### Why Pure Functional?
- **Reactant requires** stateless functions
- **Enzyme requires** differentiable operations
- **Testing is easier** with deterministic functions
- **Parallelization** is straightforward

### Why Not GPU Arrays Directly?
- Reactant handles GPU compilation automatically
- Keeps code device-agnostic
- Easier to test on CPU first
- XLA optimizer handles memory layout

### Why Test Against GECATSIM?
- Industry-standard CT simulator
- NIH-validated physics models
- Reference datasets available
- Establishes credibility for clinical use

### Why Keep Notebook Versions?
- Rapid prototyping and exploration
- Interactive visualization
- Teaching and documentation
- Quick experiments before package integration

---

## Contributing Guidelines

1. **All new physics models require:**
   - Unit tests with known solutions
   - Comparison against reference implementation
   - Documentation with equations and citations
   - Reactant compilation verification

2. **All PRs must:**
   - Pass all existing tests
   - Add tests for new functionality
   - Update documentation
   - Include performance benchmarks

3. **Code style:**
   - Use JuliaFormatter.jl
   - Type annotations for public API
   - Docstrings for all exported functions
   - Comments for complex physics

---

## References

1. GECATSIM Documentation: https://github.com/xcist/main
2. Feldkamp, L. A., et al. "Practical cone-beam algorithm." JOSA A 1.6 (1984)
3. Klein-Nishina formula: Compton scattering cross section
4. NIST XCOM database: https://www.nist.gov/pml/xcom-photon-cross-sections-database
5. Attenuations.jl: https://github.com/JuliaPhysics/Attenuations.jl
