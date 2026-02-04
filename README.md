# BasisSimulator.jl

**GPU-native CT simulation with backend-agnostic execution via AcceleratedKernels.jl**

A comprehensive CT (Computed Tomography) simulator for medical imaging research, supporting conventional energy-integrating detectors and photon-counting CT (PCCT). Core algorithms ported from [TIGRE](https://github.com/CERN/TIGRE) with physics models inspired by [CatSim/XCIST](https://github.com/xcist/main).

## Features

- **GPU Acceleration**: Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD), or CPU fallback
- **Full Physics Pipeline**: 14 toggleable physics effects (scatter, crosstalk, focal spot blur, etc.)
- **Multiple Reconstruction Algorithms**: FDK, SIRT, CGLS, TV-regularized, MBIR
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

# Iterative (SIRT)
ReconOptions(algorithm=:sirt, iterations=50, lambda=0.5)

# TV-regularized
ReconOptions(algorithm=:tv_sirt, iterations=50, tv_weight=0.01)

# VMI reconstruction
ReconOptions(algorithm=:fdk, vmi_energies=[40.0, 70.0, 100.0])
```

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
├── reconstruction/                # Image reconstruction
│   ├── fdk.jl                     # FDK cone-beam reconstruction
│   ├── sirt.jl, cgls.jl           # Iterative algorithms
│   ├── mbir.jl                    # Model-based IR
│   └── helical_recon.jl           # Helical CT
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

## Code Quality Assessment

### Cleanest Files (Best for Learning)

| File | Notes |
|------|-------|
| `forward/siddon.jl` | Excellent documentation, mathematical foundation, TIGRE references |
| `reconstruction/fdk.jl` | Complete algorithm explanation, physical interpretation |
| `metrics/mtf.jl` | AAPM TG-233 references, clinical methodology |
| `forward/photon_counting.jl` | Comprehensive physics, FDA references |

### Files Needing Verification

| File | Concern |
|------|---------|
| `forward/scatter.jl` | Empirical model, needs Monte Carlo validation |
| `forward/das_model.jl` | **BROKEN** — disabled in all fidelity presets |
| `reconstruction/mbir.jl` | Regularization parameters need clinical tuning |

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
