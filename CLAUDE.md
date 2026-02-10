# BasisSimulator.jl - CT Simulator

## CRITICAL: GPU Closure Type Stability Rule

**NEVER conditionally assign a variable and then capture it in an `AK.foreachindex` closure.**

This causes `Core.Box` wrapping which makes the closure non-bitstype — Metal/CUDA GPU compilation will fail.

**BAD — causes Core.Box on GPU:**
```julia
if ws_kernel !== nothing
    kernel = ws_kernel
else
    kernel = similar(sinogram, 3, 3)
    copyto!(kernel, kernel_cpu)
end
AK.foreachindex(sinogram) do idx
    val = kernel[idx]  # Core.Box! GPU compilation fails
end
```

**GOOD — use `let` to capture with concrete type:**
```julia
if ws_kernel !== nothing
    kernel = ws_kernel
else
    kernel = similar(sinogram, 3, 3)
    copyto!(kernel, kernel_cpu)
end
let kernel = kernel
    AK.foreachindex(sinogram) do idx
        val = kernel[idx]  # Concrete type, GPU compiles
    end
end
```

This applies to ALL variables captured from conditional branches (if/else, ternary).
Same rule for any `ws_*` kwargs with `nothing` defaults.

---

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
├── forward/
│   ├── siddon.jl                    # Ray tracing (TIGRE)
│   ├── polychromatic.jl             # Beer-Lambert + signal chain
│   ├── physics_pipeline.jl          # Unified physics config
│   ├── scatter.jl, crosstalk.jl     # Physics effects
│   ├── focal_spot.jl, detector_lag.jl
│   ├── fill_factor.jl, flat_filter.jl, bowtie_filter.jl
│   ├── detector_noise.jl, detector_efficiency.jl
│   ├── heel_effect.jl               # Signal chain
│   ├── das_model.jl                 # BROKEN
│   ├── calibration.jl
│   ├── beam_hardening_correction.jl
│   ├── photon_counting.jl           # PCCT detector model
│   └── pcct_spectral.jl             # PCCT spectral imaging
├── reconstruction/
│   ├── fdk.jl, backprojection.jl, filtering.jl
│   ├── sirt.jl, cgls.jl, mbir.jl
│   └── helical_recon.jl
├── geometry/
│   ├── scanner.jl                   # Scanner, CTGeometry
│   └── phantom.jl                   # Phantom, compute_μ
├── physics/
│   ├── materials.jl, attenuation.jl, spectrum.jl
├── dual_energy/
│   └── dual_energy.jl               # VMI, material decomposition
├── metrics/
│   ├── mtf.jl, nps.jl, psf.jl       # AAPM TG-233 metrics
├── scanners/
│   ├── scanners.jl, general_electric.jl, siemens.jl
└── simulation/
    ├── options.jl                   # SimOptions, ReconOptions
    └── driver.jl                    # simulate() entry point
```

---

## Dual-Energy (Dual kVp) CT

BasisSimulator supports GE GSI-style dual-energy CT with rapid kVp switching:

```julia
using BasisSimulator
using Metal  # or CUDA

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=256, n_slices=32)
spec = GERevolutionApex()
geom = create_geometry(spec; n_angles=984, n_rows=64)
materials = get_region_materials()

# Define GSI protocol (80/140 kVp rapid switching)
protocol = default_gsi_protocol(
    low_mA = 400.0,   # 80 kVp tube current
    high_mA = 400.0   # 140 kVp tube current
)

# Dual-energy forward projection
de_sino = forward_project_dual_energy(
    MtlArray(phantom.mask), geom, protocol;
    materials = materials,
    scanner = spec
)

# Access low/high energy sinograms
sino_80kVp = de_sino.low    # 80 kVp sinogram
sino_140kVp = de_sino.high  # 140 kVp sinogram

# Material decomposition (water/iodine basis)
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))
water_sino = mat_map.material1
iodine_sino = mat_map.material2

# Virtual Monoenergetic Imaging (VMI)
vmi_50keV = virtual_monoenergetic(mat_map, 50.0)   # High iodine contrast
vmi_70keV = virtual_monoenergetic(mat_map, 70.0)   # Balanced
vmi_100keV = virtual_monoenergetic(mat_map, 100.0) # Metal artifact reduction

# Reconstruct VMI
recon_70keV = fdk_reconstruct(vmi_70keV, geom, size(phantom.μ))
```

### VMI Energy Selection Guide

| Energy (keV) | Use Case |
|--------------|----------|
| 40-50 | Maximum iodine enhancement (high noise) |
| 50-60 | Good contrast, moderate noise |
| 65-75 | Balanced (similar to 120 kVp single-energy) |
| 80-100 | Reduced beam hardening artifacts |
| 100-140 | Metal artifact reduction |

### VMI Reconstruction Integration

The `reconstruct_vmi()` function provides a complete VMI reconstruction pipeline:

```julia
# Full VMI reconstruction with HU conversion
vmi_70_hu = reconstruct_vmi(mat_map, 70.0, geom, (256, 256, 32);
    method=:fdk,        # or :sirt
    to_hu=true,         # Convert to Hounsfield Units
    niter=3             # SIRT iterations (ignored for FDK)
)

# Get attenuation values instead of HU
vmi_70_mu = reconstruct_vmi(mat_map, 70.0, geom, (256, 256, 32);
    to_hu=false
)

# Use SIRT for better quality (slower)
vmi_70_sirt = reconstruct_vmi(mat_map, 70.0, geom, (256, 256, 32);
    method=:sirt, niter=5
)
```

### Batch VMI Generation (keV Sweep)

Generate VMI at multiple energies for visualization:

```julia
energies = [40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 140.0]
vmi_series = generate_vmi_series(mat_map, energies, geom, (128, 128, 16))

# Access specific energy
vmi_50 = vmi_series[50.0]

# Plot iodine enhancement vs energy
using Statistics
iodine_mask = phantom.mask[:, :, 8] .== UInt8(REGION_I_10_0)
mean_hu = [mean(vmi_series[E][:, :, 8][iodine_mask]) for E in energies]
```

### VMI HU Conversion

Energy-specific water attenuation is used for correct HU conversion:

```julia
# Get NIST-validated water attenuation
μ_water_70 = get_water_attenuation_vmi(70.0)  # ~0.193 cm⁻¹

# Manual HU conversion (if needed)
vmi_sino = virtual_monoenergetic(mat_map, 70.0)
recon = fdk_reconstruct(vmi_sino, geom, recon_size)
recon_hu = vmi_to_hu(Array(recon), 70.0)  # Water = 0 HU
```

### Physics Behavior in VMI

- **Water:** HU = 0 at all VMI energies (by definition)
- **Iodine:** Maximum HU at ~40-50 keV (above K-edge at 33.2 keV), decreases at high keV
- **Calcium:** HU decreases monotonically with keV (K-edge at 4 keV, below diagnostic range)
- **Noise:** Increases at low keV (~3× at 40 keV vs 70 keV baseline)

Example: `examples/vmi_keV_sweep.jl` demonstrates energy-dependent contrast behavior.

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

Last Updated: 2026-01-16
