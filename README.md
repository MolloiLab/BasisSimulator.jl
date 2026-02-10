# BasisSimulator.jl

**GPU-native CT simulation with backend-agnostic execution via AcceleratedKernels.jl**

A comprehensive CT (Computed Tomography) simulator for medical imaging research, supporting conventional energy-integrating detectors and photon-counting CT (PCCT). Core algorithms ported from [TIGRE](https://github.com/CERN/TIGRE) with physics models inspired by [CatSim/XCIST](https://github.com/xcist/main).

## Features

- **GPU Acceleration**: Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD), or CPU fallback
- **Full Physics Pipeline**: 13 toggleable physics effects (scatter, crosstalk, focal spot blur, etc.)
- **Multiple Reconstruction Algorithms**: FDK, Hybrid IR (ASIR-V/SAFIRE-style), MBIR, SIRT, CGLS
- **Spectral Imaging**: Dual-energy CT and PCCT with VMI and material decomposition
- **Photon-Counting CT**: Energy-resolved sinograms with CdTe detector physics (charge sharing, K-fluorescence, pileup, DRM)
- **Dose Validation**: Protocol validation, CTDIvol, DLP, constant-dose/constant-noise helpers
- **Zero-Allocation Workspaces**: Pre-allocated GPU buffers for production pipelines

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

---

## 1. Imaging Chain Architecture

The `src/` folder mirrors the physical X-ray imaging chain, making the codebase self-documenting:

```
                  X-ray Source           Object            Scanner
                  ──────────            ──────            ───────
                  source/               object/           geometry/
                  spectrum.jl           phantom.jl        scanner.jl
                  bowtie_filter.jl      materials.jl
                  flat_filter.jl        attenuation.jl
                  heel_effect.jl
                  focal_spot.jl
                  protocol.jl
                       │                    │                 │
                       └────────┬───────────┘                 │
                                ▼                             │
                        Ray Tracing                           │
                        ───────────                           │
                        projection/                           │
                        siddon.jl  ◄──────────────────────────┘
                        polychromatic.jl
                                │
                                ▼
                        Detector Response
                        ─────────────────
                        detector/
                        scatter.jl            detector_noise.jl
                        crosstalk.jl          detector_efficiency.jl
                        detector_lag.jl       fill_factor.jl
                        das_model.jl          physics_pipeline.jl
                        photon_counting.jl
                        └── pcct/
                            cdte_constants.jl       charge_collection.jl
                            charge_transport.jl     pileup_model.jl
                            k_fluorescence.jl       detector_response.jl
                                │
                                ▼
                        Post-Detection
                        ──────────────
                        correction/                 spectral/
                        beam_hardening_correction.jl pcct_spectral.jl
                        calibration.jl              dual_energy.jl
                                │
                                ▼
                        Reconstruction
                        ──────────────
                        reconstruction/
                        ├── core/           backprojection.jl, filtering.jl
                        ├── fbp/            fdk.jl, helical_recon.jl
                        ├── ir/             sirt.jl, cgls.jl
                        ├── hybrid_ir/      hybrid_ir.jl
                        ├── mbir/           mbir.jl
                        ├── regularization/ tv_regularization.jl
                        └── statistical_ir.jl
                                │
                                ▼
                        Image Quality
                        ─────────────
                        metrics/
                        mtf.jl    nps.jl    psf.jl
```

**Orchestration layer** (`api/`): `options.jl`, `workspace.jl`, `driver.jl`
**Scanner presets** (`scanners/`): `scanners.jl`, `general_electric.jl`, `siemens.jl`, `helical_protocols.jl`

---

## 2. API & Type Reference

### The 5-Part API

```
┌──────────────────────────────────────────────────────────────────┐
│                        simulate(...)                             │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌─────┐│
│  │ Phantom  │  │ Scanner  │  │ Protocol │  │SimOpts  │  │Recon││
│  │          │  │          │  │          │  │         │  │Opts ││
│  │ mask     │  │ geometry │  │ kVp, mA  │  │fidelity │  │algo ││
│  │ materials│  │ detector │  │ views    │  │ seed    │  │fov  ││
│  │ voxel_sz │  │ source   │  │ pitch    │  │ use_*   │  │vmi  ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘  └─────┘│
│  Geometry      Hardware      Acquisition    Simulation   Recon  │
│  + Materials   Config        Parameters     Fidelity     Config │
└──────────────────────────────────────────────────────────────────┘
```

### Key Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `Phantom` | Labeled voxel volume + materials | `mask`, `materials`, `voxel_size` |
| `Scanner` | Hardware configuration | `ScannerType`, detector, geometry specs |
| `CTProtocol` | Acquisition parameters | `kVp`, `mA`, `views`, `rotation_time`, `pitch`, `dual_energy` |
| `CTGeometry` | Pre-computed geometry arrays | `SAD`, `SDD`, `angles`, `source_positions`, `detector_centers` |
| `SimOptions` | Simulation fidelity control | `fidelity` preset, per-effect `use_*` flags |
| `ReconOptions` | Reconstruction settings | `algorithm`, `matrix_size`, `fov_cm`, `vmi_energies` |
| `PhysicsConfig` | Low-level physics effect config | Per-effect model structs (scatter, crosstalk, etc.) |
| `SimulationResult` | All simulation outputs | `sinogram_noisy`, `reconstruction`, `pcct_material_maps`, `vmi_volumes` |

### Key Functions

**Simulation:**
```julia
simulate(phantom, scanner, protocol, sim_opts, recon_opts; materials=nothing) -> SimulationResult
simulate!(ws::PCCTWorkspace, ...)    # Zero-allocation PCCT
simulate!(ws::EICTWorkspace, ...)    # Zero-allocation single-kVp
simulate!(ws::EICTDualWorkspace, ...)# Zero-allocation dual-kVp
```

**Reconstruction:**
```julia
fdk_reconstruct(sinogram, geom, volume_size)
hybrid_ir_reconstruct(sinogram, geom, volume_size; strength=3)
reconstruct!(ws::FDKReconWorkspace, ...)
reconstruct!(ws::HIRReconWorkspace, ...)
```

**Forward projection:**
```julia
siddon_forward_project(volume, geom)
forward_project(mask, geom; energies, weights, materials, physics)
```

**Spectral imaging:**
```julia
pcct_material_decomposition(pcct_sino; basis, kvp)
synthesize_vmi(material_map, target_keV)
decompose_materials(de_sino; basis)
virtual_monoenergetic(material_map, target_keV)
compute_pcct_bin_energies(thresholds; kvp, max_keV)
```

**Dose validation:**
```julia
validate_protocol(protocol, scanner) -> (valid, messages)
compute_ctdi_vol(protocol; phantom_diameter=320)
compute_dlp(protocol, scan_length_cm)
dose_report(protocol, geom)
constant_dose_protocol(base, new_views)
constant_noise_protocol(base, new_views)
```

**Phantoms:**
```julia
create_gammex_472(; n_voxels=512, n_slices=32, fov_cm=35.0, z_cm=4.0)
create_acr_464(; n_voxels=512)
```

---

## 3. GPU Computation Model

### Backend-Agnostic Execution

All GPU operations use [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) (AK) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) (KA). The backend is determined by the array type passed in — no backend-specific imports in source code.

```julia
using BasisSimulator

# CPU (default)
sinogram = forward_project(phantom.mask, geom; ...)

# Metal (Apple Silicon)
using Metal
sinogram = forward_project(MtlArray(phantom.mask), geom; ...)

# CUDA (NVIDIA)
using CUDA
sinogram = forward_project(CuArray(phantom.mask), geom; ...)
```

### Pre-Computed on CPU (Once)

These are computed during workspace creation and do not require GPU:

- Scanner geometry (source positions, detector centers, angles)
- Spectrum loading and downsampling
- Attenuation lookup tables (NIST XCOM via XrayAttenuation.jl)
- SIRT normalization weights (W_proj, V_inv)
- Physics effect kernels (scatter, crosstalk, bowtie profiles)

### Runs on GPU Every Call

These execute via `AK.foreachindex` or KA kernels on whichever backend the arrays live on:

- Siddon ray tracing (forward projection)
- Beer-Lambert spectral integration
- All 13 physics effects (scatter, crosstalk, noise, etc.)
- Voxel-driven backprojection
- PWLS iterative updates (Hybrid IR)
- Material decomposition

### Zero-Allocation Pattern

Production pipelines use pre-allocated workspace structs:

```julia
# Create workspace (allocates all GPU buffers once)
ws = create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom;
                           T=Float32, materials=materials)

# Simulate (zero buffer allocations — writes into workspace buffers)
simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

# Reconstruct (also zero-allocation)
recon_ws = create_hir_recon_workspace(sinogram, geom, volume_size; strength=3)
reconstruct!(recon_ws, sinogram, geom, volume_size)
```

**Workspace types:** `PCCTWorkspace`, `EICTWorkspace`, `EICTDualWorkspace`, `FDKReconWorkspace`, `HIRReconWorkspace`

---

## 4. Physics Effects Reference

### All 13 Effects

| # | Effect | Config | Domain | Description |
|---|--------|--------|--------|-------------|
| 1 | Fill factor | `fill_factor_standard()` | Attenuation | Detector active area fraction (0.9 typical) |
| 2 | Flat filter | `flat_filter_al(3.0)` | Attenuation | Inherent filtration (Al, Cu) |
| 3 | Bowtie filter | `bowtie_filter_large_body()` | Attenuation | Angle-dependent beam shaping |
| 4 | Scatter | `default_scatter_model()` | Attenuation | Patient scatter (spatial convolution) |
| 5 | Scatter correction | scatter_correction model | Attenuation | Scatter removal (if applied) |
| 6 | Crosstalk | `crosstalk_medium()` | Attenuation | Electronic pixel-to-pixel coupling |
| 7 | Optical crosstalk | `optical_crosstalk_typical()` | Attenuation | Optical light spread in scintillator |
| 8 | Focal spot | `focal_spot_medium()` | Attenuation | Geometric blur from finite focal spot |
| 9 | Detector efficiency | `detector_efficiency_gos(0.5)` | Attenuation | Scintillator DQE |
| 10 | Lag | `lag_gadox()` | Attenuation | Temporal persistence (afterglow) |
| 11 | Heel effect | `default_heel_effect()` | Intensity | Anode self-attenuation |
| 12 | DAS model | `default_das_model()` | Intensity | Signal chain (BROKEN) |
| 13 | BHC | `bhc_water_default()` | Post-log | Water-based beam hardening correction |

Noise is always applied via `sim_detect()` (Poisson → Gaussian approximation).

### Preset Configurations

```julia
full_physics_config()       # ALL 13 effects enabled (complete clinical simulation)
realistic_physics_config()  # Common subset (scatter, crosstalk, focal_spot, noise, lag)
minimal_physics_config()    # Noise only
default_physics_config()    # All disabled — use kwargs to enable selectively
```

### SimOptions Fidelity Presets

| Preset | Effects |
|--------|---------|
| `:ideal` | All OFF — geometric ray tracing only |
| `:low` | Noise only |
| `:medium` | Noise + focal_spot + crosstalk + flat_filter + bhc |
| `:high` | All effects enabled (except DAS) |
| `:pcct` | High + PCCT detector corrections |

---

## 5. Clinical Scanner Support

### Supported Scanners

| Scanner | Manufacturer | Type | Detector | Key Specs |
|---------|-------------|------|----------|-----------|
| GE Revolution Apex Elite | GE Healthcare | EICT | 256 x 832, Gemstone Clarity (LUMEX) | SID=626mm, SDD=1097mm, 160mm z-coverage |
| Siemens NAEOTOM Alpha | Siemens Healthineers | PCCT | 144 x 2280 (std), CdTe | SID=600mm, SDD=1072mm, 4 energy bins [20,35,55,70] keV |
| Siemens NAEOTOM Alpha UHR | Siemens Healthineers | PCCT | 120 x 2280, CdTe | 0.2mm pixels, 24mm z-coverage |
| Canon Aquilion ONE | Canon Medical | EICT | Configurable | Generic EICT geometry |

```julia
# Clinical scanner with validated specs (from FDA 510(k))
spec = GERevolutionApexElite()
geom = create_geometry(spec; n_angles=984, fov_cm=35.0)

spec = SiemensNAEOTOMAlpha(:standard)
geom = create_geometry(spec; n_angles=984)

# Protocol presets
axial = AxialProtocol(kvp=120, ma=200, rotation_time_s=1.0, n_angles=984)
helical = HelicalProtocol(kvp=120, ma=200, pitch=0.8, n_rotations=10)
geom = create_geometry(spec, helical)
```

### PCCT Detector Physics (CdTe)

The PCCT model includes physics-based CdTe detector simulation:

| Component | Model | Reference |
|-----------|-------|-----------|
| Charge cloud transport | Koch-Mehrin ODE (sigma ~12-14 um) | Koch-Mehrin 2020 (NIM-A) |
| K-fluorescence | 5 K-lines per element, Te to Cd cascade | Koch-Mehrin Table 1 |
| Charge collection | Hecht CCE + Barrett small-pixel weighting | Barrett 1995 |
| Pileup | Yang 2025 seminonparalyzable | Yang 2025 |
| DRM | Unified detector response matrix (FWHM ~3.55 keV) | Konrad 2025 (PMB) |

---

## 6. Quick Start Examples

### Basic Simulation (Single-kVp)

```julia
using BasisSimulator

phantom = create_gammex_472(n_voxels=128, fov_cm=35.0)
scanner = Scanner(ScannerType.EICT)
protocol = CTProtocol(kVp=120, mA=200, views=984)
sim_opts = SimOptions(fidelity=:high, seed=42)
recon_opts = ReconOptions(algorithm=:fdk, matrix_size=(512, 512, 64))

result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)
volume = result.reconstruction
```

### Hybrid IR Reconstruction

```julia
recon_opts = ReconOptions(algorithm=:hybrid_ir, strength=3)
result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)

# Or direct API
recon = hybrid_ir_reconstruct(sinogram, geom, (256, 256, 128); strength=3)
```

| Strength | Noise Reduction | Iterations | Use Case |
|----------|-----------------|------------|----------|
| 1 | ~9% | 8 | Minimal smoothing, preserve texture |
| 2 | ~20% | 15 | Moderate smoothing |
| 3 | ~32% | 30 | Balanced (recommended) |
| 4 | ~38% | 60 | Strong smoothing |
| 5 | ~38% | 100 | Maximum (SIRT ceiling) |

### Dual-Energy CT with VMI

```julia
protocol = CTProtocol(dual_energy=true, kVp=140, mA=200, kVp_low=80, mA_low=350)
recon_opts = ReconOptions(algorithm=:fdk, vmi_energies=[40.0, 70.0, 100.0])

result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)
vmi_70 = result.vmi_volumes[70.0]
```

### PCCT Simulation with Material Decomposition

```julia
scanner = Scanner(ScannerType.PCCT)
protocol = CTProtocol(kVp=120, mA=200, views=984)
sim_opts = SimOptions(fidelity=:pcct)

result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)

# Access energy-resolved data
bin_sinograms = result.pcct_sinogram
material_maps = result.pcct_material_maps
vmi_volumes = result.pcct_vmi_volumes
```

### VMI Physics

```julia
# Spectrum-weighted bin energies (not arithmetic mean)
E_eff = compute_pcct_bin_energies([20.0, 35.0, 55.0, 70.0]; kvp=120)

# Material decomposition + VMI synthesis
mat_map = pcct_material_decomposition(pcct_sino; basis=(:water, :iodine), kvp=120)
vmi_70 = synthesize_vmi(mat_map, 70.0)
```

**Expected physics:**
- Water: HU = 0 at all energies (by definition)
- Iodine: Maximum contrast at ~40-50 keV (above K-edge at 33.2 keV)
- Calcium: Contrast decreases monotonically (K-edge at 4 keV, below diagnostic range)

### Dose Validation and Reporting

```julia
protocol = CTProtocol(kVp=120, mA=200, views=984)
scanner = Scanner(ScannerType.EICT)

# Validate protocol parameters
result = validate_protocol(protocol, scanner)
println(result.valid, " — ", result.messages)

# Compute dose metrics
ctdi = compute_ctdi_vol(protocol)          # mGy
dlp = compute_dlp(protocol, 30.0)          # mGy*cm (30cm scan)

# Formatted dose report
report = dose_report(protocol, geom)

# Adjust protocol for different view counts
proto_2000 = constant_dose_protocol(protocol, 2000)   # Same dose, more views
proto_2000n = constant_noise_protocol(protocol, 2000)  # Same noise/view, more dose
```

### Image Quality Metrics

```julia
# Spatial resolution
mtf_result = compute_mtf(volume, roi_center, roi_radius)
f50, f10 = mtf_result.f50, mtf_result.f10

# Noise texture
nps_result = compute_nps(volume, roi_center, roi_size)

# Point spread function
psf_result = compute_psf(volume, wire_center)
```

### GPU Execution

```julia
using Metal  # or CUDA

# Option 1: Via workspaces (zero-allocation)
ws = create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom;
                           T=Float32, materials=get_region_materials())
simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

# Option 2: Via forward_project API
mask_gpu = MtlArray(phantom.mask)
sinogram = forward_project(mask_gpu, geom;
    energies=energies, weights=weights, materials=materials,
    physics=full_physics_config())

# Option 3: Via simulate() — auto-detects GPU from workspace
result = simulate(phantom, scanner, protocol, sim_opts, recon_opts)
```

---

## References

- **TIGRE**: Biguri A, et al. "TIGRE: A MATLAB-GPU toolbox for CBCT image reconstruction." Biomed Phys Eng Express. 2016. [GitHub](https://github.com/CERN/TIGRE)
- **CatSim/XCIST**: GE Healthcare CT simulation tools. [GitHub](https://github.com/xcist/main)
- **Koch-Mehrin 2020**: Koch-Mehrin KAF, et al. "Charge transport in CdTe photon-counting detectors." NIM-A 976:164241.
- **Konrad 2025**: Konrad U, et al. "Validated NAEOTOM Alpha MC model." PMB 70:065004.
- **Yang 2025**: Yang Q, et al. "Seminonparalyzable pileup model for PCCT."
- **AAPM TG-233**: Quality assurance for CT-based technologies.
- **Feldkamp 1984**: Feldkamp LA, et al. "Practical cone-beam algorithm." J Opt Soc Am A.
- **Siddon 1985**: Siddon RL. "Fast calculation of the exact radiological path." Med Phys.

## License

MIT
