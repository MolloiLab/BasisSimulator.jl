# BasisSimulator.jl - CT Simulator

## Overview

CT simulation with backend-agnostic GPU/CPU execution via **AcceleratedKernels.jl**.

- Core ray tracing: ported from TIGRE
- Polychromatic physics: our own implementation (TIGRE is monochromatic only)
- Signal chain: CatSim-exact calibration workflow

**Works on:** Metal (Apple), CUDA (NVIDIA), ROCm (AMD), Intel oneAPI, or CPU.

---

## Quick Start

```julia
using BasisSimulator
using Metal  # or CUDA

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=512, n_slices=32, fov_cm=35.0, z_cm=4.0)
geom = create_aquilion_one(n_angles=1160, n_rows=64, n_cols=736, fov_cm=35.0, z_cm=4.0)

# Load spectrum
energies, weights = load_spectrum(120)
energies, weights = downsample_spectrum(energies, weights, 30)
materials = get_region_materials()

# Full clinical simulation (ALL 13 physics effects)
mask_gpu = MtlArray(phantom.mask)
sinogram = forward_project(mask_gpu, geom;
    energies=energies, weights=weights, materials=materials,
    physics=full_physics_config(energy_keV=65.0, noise_seed=42)
)

# Reconstruct
recon = fdk_reconstruct(sinogram, geom, (512, 512, 32))

# Convert to HU
μ_water = get_effective_μ_water_kVp(120)
recon_hu = 1000f0 .* (Array(recon) .- μ_water) ./ μ_water
```

---

## Physics Configuration

### Enabling/Disabling Effects

Use `default_physics_config()` with explicit kwargs. **Comment out or set to `nothing` to disable** (NOT `false`):

```julia
physics = default_physics_config(
    # --- SCANNER-SPECIFIC (CatSim essential) ---
    fill_factor = fill_factor_standard(),           # 0.9 fill factor
    flat_filter = flat_filter_al(3.0),              # 3mm Al
    bowtie_filter = bowtie_filter_large_body(),     # Large body
    detector_efficiency = detector_efficiency_gos(0.5),  # GOS 0.5mm

    # --- OPTIONAL PHYSICS ---
    scatter = default_scatter_model(scale_factor=1.0),
    crosstalk = crosstalk_medium(),
    optical_crosstalk = optical_crosstalk_typical(),
    # focal_spot = focal_spot_medium(),             # <-- commented out = disabled
    noise = default_detector_model(I0=1e6, seed=42),
    lag = lag_gadox(),

    # --- SIGNAL CHAIN (CatSim-exact) ---
    heel_effect = default_heel_effect(anode_angle_deg=7.0),
    das_model = default_das_model(gain=1.0, electronic_noise_sigma=100.0),
    bhc = bhc_water_default(reference_energy_keV=65.0),

    energy_keV = 65.0,
    noise_seed = 42
)
```

### Preset Configurations

```julia
full_physics_config()       # ALL 13 effects enabled
realistic_physics_config()  # Common subset
minimal_physics_config()    # Noise only
default_physics_config()    # All nothing (use kwargs to enable)
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Forward Projection (Siddon) | ✅ | TIGRE port |
| Polychromatic FP | ✅ | Beer-Lambert |
| FDK Reconstruction | ✅ | TIGRE port |
| SIRT/CGLS | ✅ | TIGRE port |
| Physics Pipeline (10 effects) | ✅ | All GPU-native |
| Signal Chain (heel, BHC) | ✅ | CatSim-exact |
| DAS Model | ⚠️ BROKEN | Needs fixing |
| Scatter | ⚠️ NO CORRECTION | Requires scatter correction |

### Physics Effects (13 total)

**Physics Pipeline (10):**
- fill_factor, flat_filter, bowtie_filter, detector_efficiency*
- scatter**, crosstalk, optical_crosstalk, focal_spot
- noise (quantum), lag (afterglow)

*detector_efficiency: no-op in calibrated mode (efficiency cancels in air scan normalization)
**scatter: adds scatter but has NO correction - will produce cupping artifacts

**Signal Chain (3):**
- heel_effect (anode self-attenuation)
- das_model (gain + electronic noise) - **BROKEN**
- bhc (beam hardening correction)

---

## CatSim Signal Chain

The signal chain follows CatSim-exact methodology:

1. **Polychromatic projection** (Beer-Lambert)
2. **Physics effects** (scatter, crosstalk, focal spot, lag, etc.)
3. **Heel effect** (intensity domain)
4. **DAS model** (gain + noise to phantom only)
5. **Air scan calibration** (noise-free reference - CatSim exact!)
6. **Low signal correction** (smooth negatives, not clamp)
7. **Log transform**
8. **Beam hardening correction**

Key design decisions:
- Air scan has NO noise (simulates averaged reference)
- Low signal correction uses smoothed neighbors
- Calibration: `prep = phantom_intensity / air_intensity`

---

## File Structure

```
src/
├── BasisSimulator.jl
├── Forward/
│   ├── Siddon.jl                    # Ray tracing (TIGRE)
│   ├── Polychromatic.jl             # Beer-Lambert + signal chain
│   ├── PhysicsPipeline.jl           # Unified physics config
│   ├── Scatter.jl, Crosstalk.jl     # Physics effects
│   ├── FocalSpot.jl, DetectorLag.jl
│   ├── FillFactor.jl, FlatFilter.jl, BowtieFilter.jl
│   ├── DetectorNoise.jl, DetectorEfficiency.jl
│   ├── HeelEffect.jl                # Signal chain
│   ├── DASModel.jl                  # BROKEN
│   ├── Calibration.jl
│   └── BeamHardeningCorrection.jl
├── Reconstruction/
│   ├── FDK.jl, Backprojection.jl, Filtering.jl
│   ├── SIRT.jl, CGLS.jl
│   └── HelicalRecon.jl
├── Geometry/
│   ├── Scanner.jl, Phantom.jl, Helical.jl
│   └── AnalyticalPhantom.jl
├── Physics/
│   ├── Materials.jl, Attenuation.jl, Spectrum.jl
└── Scanners/
    ├── Scanners.jl, GeneralElectric.jl
```

---

## GPU Usage

AcceleratedKernels.jl auto-detects backend from array type:

```julia
# CPU
sinogram = forward_project(phantom.mask, geom; ...)

# Metal (Apple)
using Metal
sinogram = forward_project(MtlArray(phantom.mask), geom; ...)

# CUDA (NVIDIA)
using CUDA
sinogram = forward_project(CuArray(phantom.mask), geom; ...)
```

---

## References

1. **TIGRE**: https://github.com/CERN/TIGRE
2. **CatSim/XCIST**: https://github.com/xcist/main
3. **AcceleratedKernels.jl**: https://github.com/JuliaGPU/AcceleratedKernels.jl

---

Last Updated: 2026-01-15
