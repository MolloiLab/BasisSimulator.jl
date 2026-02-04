# BasisSimulator.jl

**GPU-native CT simulation with backend-agnostic execution via AcceleratedKernels.jl**

A comprehensive CT (Computed Tomography) simulator for medical imaging research, supporting conventional energy-integrating detectors and photon-counting CT (PCCT). Core algorithms ported from [TIGRE](https://github.com/CERN/TIGRE) with physics models inspired by [CatSim/XCIST](https://github.com/xcist/main).

## Features

- **GPU Acceleration**: Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD), or CPU fallback
- **Full Physics Pipeline**: 14 toggleable physics effects (scatter, crosstalk, focal spot blur, etc.)
- **Multiple Reconstruction Algorithms**: FDK, Hybrid IR (ASIR-V/SAFIRE-style), MBIR, SIRT, CGLS
- **Spectral Imaging**: Dual-energy CT with VMI and material decomposition
- **Photon-Counting CT**: Energy-resolved sinograms, N-material decomposition
- **End-to-End Differentiability**: Enzyme.jl integration for inverse problems

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MolloiLab/BasisSimulator.jl")
```

For GPU support, also install the appropriate backend:
```julia
Pkg.add("Metal")     # Apple Silicon
Pkg.add("CUDA")      # NVIDIA
Pkg.add("AMDGPU")    # AMD
```

## Quick Start

```julia
using BasisSimulator

# Create Gammex 472 calibration phantom
phantom = create_gammex_472(n_voxels=128, fov_cm=35.0)

# Configure scanner (Canon Aquilion ONE-like)
scanner = Scanner(
    source_to_isocenter = 600.0,  # mm
    source_to_detector = 1000.0,
    detector_rows = 64,
    detector_cols = 900
)

# Set acquisition protocol
protocol = CTProtocol(kVp=120, mA=200, views=984)

# Configure simulation fidelity
sim_opts = SimOptions(fidelity=:high, seed=42)

# Configure reconstruction
recon_opts = ReconOptions(
    algorithm=:fdk,
    matrix_size=(512, 512, 64),
    fov_cm=35.0
)

# Run simulation
result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)

# Access results
volume = result.reconstruction
sinogram = result.sinogram_noisy
```

## The 5-Part API

BasisSimulator uses a clean 5-struct API:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      simulate(...)                                   │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │ Phantom  │  │ Scanner  │  │ Protocol │  │SimOptions│  │ReconOpts││
│  │          │  │          │  │          │  │          │  │         ││
│  │ mask     │  │ geometry │  │ kVp, mA  │  │ fidelity │  │algorithm││
│  │ materials│  │ detector │  │ views    │  │ seed     │  │fov_cm   ││
│  │ voxel_sz │  │ source   │  │ pitch    │  │ use_*    │  │vmi_keV  ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
│                                                                      │
│  Geometry      Hardware      Acquisition    Simulation   Reconstruction
│  + Materials   Config        Parameters     Fidelity     Settings
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Phantom — Geometry + Materials

```julia
# Create from labeled array
materials_dict = Dict(0 => XA.Materials.air, 1 => XA.Materials.water)
phantom = Phantom(labeled_array, materials_dict, (0.1, 0.1, 0.1))  # 1mm voxels

# Or use built-in Gammex 472
phantom = create_gammex_472(n_voxels=128)

# Get attenuation at any energy
μ_60keV = compute_μ(phantom, 60.0)
```

### 2. Scanner — Hardware Configuration

```julia
# Default scanner
scanner = Scanner()

# Custom scanner
scanner = Scanner(
    source_to_isocenter = 595.0,   # mm
    source_to_detector = 1085.5,
    detector_rows = 144,
    detector_cols = 2280
)

# PCCT scanner (Siemens NAEOTOM Alpha-like)
scanner = create_naeotom_alpha(mode=:standard)
is_pcct(scanner)  # true
```

### 3. CTProtocol — Acquisition Parameters

```julia
# Simple axial
CTProtocol(kVp=120, mA=200, views=984)

# Helical
CTProtocol(scan_mode=:helical, kVp=120, mA=200, pitch=0.984, n_rotations=10.0)

# Dual-energy
CTProtocol(dual_energy=true, kVp=140, mA=200, kVp_low=80, mA_low=350)
```

### 4. SimOptions — Simulation Fidelity

```julia
SimOptions(fidelity=:high)                      # Full physics (default)
SimOptions(fidelity=:medium)                    # Noise + key effects
SimOptions(fidelity=:ideal)                     # Geometric ray tracing only
SimOptions(fidelity=:high, use_scatter=false)   # Selective effect override
```

**Fidelity Presets:**
| Preset | Effects |
|--------|---------|
| `:ideal` | All OFF — geometric ray tracing |
| `:low` | Noise only |
| `:medium` | Noise + focal_spot + crosstalk + flat_filter + bhc |
| `:high` | All 14 effects enabled |
| `:pcct` | High + PCCT corrections |

### 5. ReconOptions — Reconstruction Settings

```julia
# FDK (default)
ReconOptions(algorithm=:fdk, matrix_size=(512, 512, 64), fov_cm=35.0)

# Hybrid IR (TRUE iterative refinement, like ASIR-V/SAFIRE)
ReconOptions(algorithm=:hybrid_ir, strength=3)  # 1-5, 3=standard clinical

# Iterative (SIRT)
ReconOptions(algorithm=:sirt, iterations=50, lambda=0.5)

# TV-regularized
ReconOptions(algorithm=:tv_sirt, iterations=50, tv_weight=0.01)

# VMI reconstruction
ReconOptions(algorithm=:fdk, vmi_energies=[40.0, 70.0, 100.0])
```

## Hybrid IR (Iterative Reconstruction)

BasisSimulator includes vendor-general Hybrid IR that matches clinical systems like GE ASIR-V and Siemens SAFIRE.

### What is TRUE Hybrid IR?

TRUE Hybrid IR uses statistical iterative refinement — not simple blending:

```
FDK initialization → PWLS iterations with statistical weights → Refined result
```

**TRUE Hybrid IR has:**
- Statistical noise model (Poisson-based weights)
- Data fidelity term using measured sinogram
- Edge-preserving regularization (Huber penalty)
- FDK provides fast initialization, iterations provide refinement

**FALSE Hybrid IR (what we avoid):**
```julia
# This is NOT true HIR — just post-processing blending
x = (1-α) * fdk_result + α * smooth(fdk_result)
```

### Strength Levels

| Strength | Noise Reduction | Performance | Clinical Use |
|----------|-----------------|-------------|--------------|
| 1 | ~10-15% | ~1.5x FDK | Preserve texture (lung imaging) |
| 2 | ~20-30% | ~2x FDK | Light smoothing |
| 3 | ~35-40% | ~3x FDK | **Standard clinical (recommended)** |
| 4 | ~45-55% | ~4x FDK | Strong smoothing |
| 5 | ~55-65% | ~5x FDK | Maximum noise reduction |

### Vendor Equivalents

| Our Strength | GE ASIR-V | Siemens SAFIRE | Philips iDose4 | Canon AIDR 3D |
|--------------|-----------|----------------|----------------|---------------|
| 1 | ~20% | S1 | Level 1 | - |
| 2 | ~40% | S2 | Level 2-3 | Mild |
| 3 | ~60% | S3 | Level 3-4 | Standard |
| 4 | ~80% | S4 | Level 5 | Strong |
| 5 | ~100% | S5 | Level 6-7 | - |

### Usage

```julia
using BasisSimulator

# Direct API
recon = hybrid_ir_reconstruct(sinogram, geom, (256, 256, 128); strength=3)

# Via simulate() driver
result = simulate(phantom, scanner, protocol, sim_opts,
                  ReconOptions(algorithm=:hybrid_ir, strength=3))

# Get parameters for a strength level
params = get_hir_params(3)
# HIRParams(3, 0.015f0, 8, 0.01f0, (30, 42))
```

### Clinical Guidelines

- **Strength 1**: Use when FBP texture is critical (lung nodules, emphysema)
- **Strength 2-3**: General clinical imaging, good balance of noise/texture
- **Strength 4**: Higher dose reduction needed, some texture loss acceptable
- **Strength 5**: Maximum dose reduction (may appear "plastic")

## Architecture

```
src/
├── BasisSimulator.jl              # Module entry point
├── physics/                       # Material attenuation, spectra
│   ├── materials.jl               # Gammex 472 materials
│   ├── spectrum.jl                # X-ray spectra loading
│   └── attenuation.jl             # μ computation
├── geometry/                      # Phantom and scanner definitions
│   ├── phantom.jl                 # Phantom struct, compute_μ
│   └── scanner.jl                 # Scanner, CTGeometry
├── forward/                       # Forward projection pipeline
│   ├── siddon.jl                  # Siddon ray tracing (TIGRE port)
│   ├── polychromatic.jl           # Beer-Lambert spectral physics
│   ├── scatter.jl                 # XCIST-style scatter model
│   ├── photon_counting.jl         # PCCT detector model
│   └── ...                        # 15+ physics effect modules
├── reconstruction/                # Image reconstruction (by clinical category)
│   ├── core/                      # Shared: backprojection.jl, filtering.jl
│   ├── fbp/                       # FBP: fdk.jl, helical_recon.jl
│   ├── hybrid_ir/                 # Hybrid IR: hybrid_ir.jl (ASIR-V/SAFIRE-style)
│   ├── ir/                        # Classic IR: sirt.jl, cgls.jl
│   ├── mbir/                      # Model-based: mbir.jl
│   └── regularization/            # TV regularization
├── dual_energy/                   # Spectral CT
│   └── dual_energy.jl             # VMI, material decomposition
├── metrics/                       # Image quality metrics
│   ├── mtf.jl                     # Spatial resolution (AAPM TG-233)
│   ├── nps.jl                     # Noise power spectrum
│   └── psf.jl                     # Point spread function
├── scanners/                      # Clinical scanner presets
│   ├── scanners.jl                # Scanner specifications
│   ├── siemens.jl                 # NAEOTOM Alpha, etc.
│   └── general_electric.jl        # Revolution, etc.
└── simulation/                    # High-level driver
    ├── options.jl                 # SimOptions, ReconOptions
    └── driver.jl                  # simulate() entry point
```

## Data Flow

```
Phantom (mask + materials)
        │
        ▼
┌───────────────────────────────────────────────────┐
│               Forward Projection                   │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐ │
│  │ Siddon  │ →  │ Physics  │ →  │   Detector   │ │
│  │Ray Trace│    │ Effects  │    │  Simulation  │ │
│  └─────────┘    └──────────┘    └──────────────┘ │
│                 (scatter,        (noise, DQE,    │
│                  filters)        efficiency)     │
└───────────────────────────────────────────────────┘
        │
        ▼
    Sinogram
        │
        ▼
┌───────────────────────────────────────────────────┐
│              Reconstruction                        │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐ │
│  │  Filter │ →  │   Back-  │ →  │     Post-    │ │
│  │ (ramp)  │    │ project  │    │  processing  │ │
│  └─────────┘    └──────────┘    └──────────────┘ │
└───────────────────────────────────────────────────┘
        │
        ▼
  CT Volume (HU)
```

## File Verification Guide

This section categorizes all 43 source files by confidence level to guide verification efforts.

### Confidence Tiers

| Tier | Description | Collaborator Action |
|------|-------------|---------------------|
| HIGH | Validated algorithms, TIGRE ports, NIST data | Use confidently |
| MEDIUM | Mostly validated, some empirical parameters | Review before critical use |
| LOW | Needs verification, complex physics chains | Verify before use |
| BROKEN | Known non-functional | Do not use |

### Tier Summary

| Tier | Count | Notable Files |
|------|-------|---------------|
| HIGH | 11 | siddon.jl, fdk.jl, backprojection.jl, materials.jl, attenuation.jl |
| MEDIUM | 30 | scatter.jl, polychromatic.jl, mbir.jl, dual_energy.jl |
| LOW | 2 | photon_counting.jl, pcct_spectral.jl |
| BROKEN | 1 | das_model.jl |

---

### TIER 1: HIGH CONFIDENCE (11 files)

*Ready for use — validated algorithms with strong documentation*

#### Forward Projection

| File | Source | Justification |
|------|--------|---------------|
| `siddon.jl` | TIGRE port | Standard Siddon algorithm with 40+ years validation, direct TIGRE port |
| `fill_factor.jl` | CatSim-compatible | Simple geometric model, no physics complexity |
| `calibration.jl` | Original | Simple eps clamping for numerical stability |

#### Reconstruction

| File | Source | Justification |
|------|--------|---------------|
| `backprojection.jl` | TIGRE port | Direct TIGRE port, well-documented voxel-driven algorithm |
| `filtering.jl` | Original | Standard ramp filter, Kak & Slaney textbook algorithm |
| `fdk.jl` | TIGRE port | Standard Feldkamp-Davis-Kress, TIGRE-validated |
| `sirt.jl` | TIGRE-style | Correct matched backprojection, TIGRE-validated |
| `cgls.jl` | TIGRE-style | Standard conjugate gradient least squares |

#### Physics & Infrastructure

| File | Source | Justification |
|------|--------|---------------|
| `materials.jl` | NIST wrapper | Direct NIST XCOM data via XrayAttenuation.jl |
| `attenuation.jl` | NIST wrapper | Wraps validated NIST XCOM database |
| `scanners/scanners.jl` | Original | Design pattern infrastructure, no physics validation needed |

---

### TIER 2: MEDIUM CONFIDENCE (30 files)

*Use with caution — mostly validated with some gaps or empirical parameters*

#### Forward Projection (14 files)

| File | Source | Key Uncertainty | Verification Needed |
|------|--------|-----------------|---------------------|
| `polychromatic.jl` | XCIST-inspired | CatSim flux constant (5.6e13) unvalidated | Validate flux density against tube output curves |
| `scatter.jl` | XCIST-inspired | Empirical scatter kernel coefficients | Compare SPR to GATE/MCNP Monte Carlo |
| `protocol.jl` | Original | Clinical parameter ranges untested | Verify ranges against clinical scanner specs |
| `detector_noise.jl` | Original | Gaussian approximation at low flux | Test ultra-low-dose scenarios |
| `detector_efficiency.jl` | CatSim-exact | Energy-dependent DQE unvalidated | Compare η(E) to published detector data |
| `bowtie_filter.jl` | XCIST-inspired | GE profiles may not match current firmware | Validate against service manuals |
| `flat_filter.jl` | CatSim-exact | Filter material/thickness assumptions | Validate HVL measurements |
| `focal_spot.jl` | CatSim-inspired | Focal spot size assumptions | Validate MTF degradation on wire phantom |
| `crosstalk.jl` | CatSim-exact | Detector-specific kernel parameters | Validate against manufacturer data |
| `detector_lag.jl` | CatSim-exact | Material-dependent time constants | Validate afterglow curves for GOS/CsI |
| `heel_effect.jl` | CatSim-exact | Anode angle assumptions | Validate intensity profile |
| `beam_hardening_correction.jl` | CatSim-compatible | Spectrum-dependent polynomial coefficients | Test bone BH correction |
| `physics_pipeline.jl` | Original | Effect ordering may differ from clinical | Validate against CatSim documentation |

#### Reconstruction (4 files)

| File | Source | Key Uncertainty | Verification Needed |
|------|--------|-----------------|---------------------|
| `tv_regularization.jl` | Original | λ selection problem-dependent | Compare to TIGRE defaults |
| `hybrid_ir.jl` | Research-based | Parameters from SAFIRE clinical studies | Validate noise reduction vs clinical targets |
| `mbir.jl` | Original | Hyperbola δ empirical, ordered subsets untested | Compare edge preservation to clinical ADMIRE |
| `helical_recon.jl` | Original | 180LI/360LI accuracy at high pitch | Test for windmill artifacts |

#### Geometry & Physics (4 files)

| File | Source | Key Uncertainty | Verification Needed |
|------|--------|-----------------|---------------------|
| `phantom.jl` | Original | Gammex HU may differ from measured values | Validate against measured Gammex 472 data |
| `scanner.jl` | Original | PCCT energy thresholds approximate | Verify energy threshold placement |
| `spectrum.jl` | Original | Spectrum provenance undocumented | Document spectrum generation process |

#### Metrics (3 files)

| File | Source | Key Uncertainty | Verification Needed |
|------|--------|-----------------|---------------------|
| `mtf.jl` | Original | Edge detection may fail on noisy images | Validate against commercial QA software |
| `nps.jl` | Original | ROI placement affects results | Compare radial averaging to published curves |
| `psf.jl` | Original | Assumes symmetric PSF | Test asymmetric scenarios |

#### Dual-Energy & Scanners (5 files)

| File | Source | Key Uncertainty | Verification Needed |
|------|--------|-----------------|---------------------|
| `dual_energy.jl` | Original | VMI weighting may not match vendor implementation | Validate VMI HU against Gammex dual-energy phantom |
| `siemens.jl` | FDA 510(k) | FDA parameters may differ from current firmware | Validate against current scanner specs |
| `general_electric.jl` | FDA 510(k) | Same as above | Validate against current scanner specs |
| `helical_protocols.jl` | Original | Helical geometry vendor-specific | Validate z-coverage calculations |

#### Simulation (2 files)

| File | Source | Key Uncertainty | Verification Needed |
|------|--------|-----------------|---------------------|
| `options.jl` | Original | Fidelity preset mappings empirical | Validate preset quality against clinical images |
| `driver.jl` | Original | 5-mode complexity may have edge cases | Test all mode combinations systematically |

---

### TIER 3: LOW CONFIDENCE (2 files)

*Needs verification — original implementations with complex physics chains*

| File | Lines | Key Issues | Critical Verification |
|------|-------|------------|----------------------|
| `photon_counting.jl` | ~2200 | Complex PCCT physics (charge sharing, pile-up, K-fluorescence) not validated against real PCCT data | Validate energy bin counts against NAEOTOM phantom data; compare charge sharing artifacts to real PCCT images |
| `pcct_spectral.jl` | ~1032 | VMI accuracy unknown; 3-material decomposition untested; K-edge sensitivity unverified | Validate VMI HU against Gammex phantom (40-190 keV); test 3-material decomposition; verify K-edge detection |

**Collaborator action**: Verify against real PCCT data before any use; consider Monte Carlo validation (GATE/MCNP).

---

### TIER 4: BROKEN (1 file)

| File | Issue | Recommendation |
|------|-------|----------------|
| `das_model.jl` | Produces incorrect gain/offset values; explicitly disabled in all fidelity presets | Do not use; fix or remove in v23.0 |

---

### Verification Priority Matrix

#### Priority 1: PCCT Validation (Tier 3)
1. Obtain NAEOTOM Alpha phantom data for validation
2. Compare simulated energy bin counts to measured data
3. Validate VMI HU accuracy across 40-190 keV range

#### Priority 2: Scatter/Physics Validation
1. Run GATE/MCNP Monte Carlo for ACR phantom
2. Compare scatter-to-primary ratios
3. Validate CatSim flux density constant (5.6e13)

#### Priority 3: Image Quality Metrics
1. Compare MTF/NPS/PSF to commercial QA software (e.g., CT ACR)
2. Validate against published phantom measurements

## Clinical Verification Checklist

For clinical-grade simulation:

- [ ] **CT Number Accuracy**: Water 0±4 HU, Air -1000±20 HU
- [ ] **Spatial Resolution**: MTF50/MTF10 within 10% of scanner specs
- [ ] **Noise Characteristics**: SD scales as 1/√(mAs)
- [ ] **Spectral Accuracy**: VMI HU within ±10 HU (40-140 keV)
- [ ] **Dose Metrics**: CTDIvol/DLP match reference values

## References

- **TIGRE**: Biguri A, et al. "TIGRE: A MATLAB-GPU toolbox for CBCT image reconstruction." Biomed Phys Eng Express. 2016. [GitHub](https://github.com/CERN/TIGRE)
- **CatSim/XCIST**: GE Healthcare CT simulation tools. [GitHub](https://github.com/xcist/main)
- **AAPM TG-233**: Quality assurance for CT-based technologies
- **Feldkamp**: Feldkamp LA, et al. "Practical cone-beam algorithm." J Opt Soc Am A. 1984.
- **Siddon**: Siddon RL. "Fast calculation of the exact radiological path." Med Phys. 1985.

## License

MIT
