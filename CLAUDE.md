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

## File Structure

Organized by imaging chain: source → object → geometry → projection → detector → correction → spectral → reconstruction → metrics → scanners → api

```
src/
├── BasisSimulator.jl
├── source/                         # X-ray source + beam shaping
│   ├── spectrum.jl                 # Spectrum loading (.dat files)
│   ├── bowtie_filter.jl            # Bowtie filter modeling
│   ├── flat_filter.jl              # Flat (inherent) filter
│   ├── heel_effect.jl              # Anode self-attenuation
│   ├── focal_spot.jl               # Finite focal spot
│   └── protocol.jl                 # CTProtocol, dose validation/reporting
├── object/                         # Scanned object
│   ├── materials.jl                # Gammex 472 materials
│   ├── attenuation.jl              # Attenuation coefficient computation
│   └── phantom.jl                  # Phantom generation (semantic masks)
├── geometry/
│   └── scanner.jl                  # Scanner, CTGeometry
├── projection/                     # Ray tracing
│   ├── siddon.jl                   # Siddon forward projection (TIGRE)
│   └── polychromatic.jl            # Beer-Lambert + unified API
├── detector/                       # ALL detector effects + noise
│   ├── scatter.jl                  # Scatter simulation
│   ├── crosstalk.jl                # Electronic + optical crosstalk
│   ├── detector_lag.jl             # Afterglow modeling
│   ├── detector_noise.jl           # Quantum noise + I0 computation
│   ├── detector_efficiency.jl      # DQE (no-op in calibrated mode)
│   ├── fill_factor.jl              # Fill factor modeling
│   ├── das_model.jl                # DAS model (BROKEN)
│   ├── physics_pipeline.jl         # Unified PhysicsConfig
│   ├── photon_counting.jl          # PCCT detector model
│   └── pcct/                       # CdTe-specific physics
│       ├── cdte_constants.jl       # Material constants
│       ├── charge_transport.jl     # Charge cloud transport (Koch-Mehrin ODE)
│       ├── k_fluorescence.jl       # K-fluorescence (5 K-lines/element)
│       ├── charge_collection.jl    # Hecht CCE + small-pixel weighting
│       ├── pileup_model.jl         # Yang 2025 seminonparalyzable pileup
│       └── detector_response.jl    # Unified DRM
├── correction/                     # Post-detection corrections
│   ├── beam_hardening_correction.jl # Water-based polynomial BHC
│   └── calibration.jl              # Air scan calibration pipeline
├── spectral/                       # Spectral imaging (PCCT + dual-energy)
│   ├── pcct_spectral.jl            # PCCT VMI, K-edge, effective Z, decomposition
│   └── dual_energy.jl              # Dual-kVp VMI, material decomposition
├── reconstruction/                 # Image reconstruction
│   ├── core/
│   │   ├── backprojection.jl       # Voxel-driven backprojection (TIGRE)
│   │   └── filtering.jl           # Ramp filter, cosine weighting
│   ├── fbp/
│   │   ├── fdk.jl                 # FDK reconstruction
│   │   └── helical_recon.jl       # Helical (spiral) CT
│   ├── ir/
│   │   ├── sirt.jl                # SIRT iterative reconstruction
│   │   └── cgls.jl                # CGLS iterative reconstruction
│   ├── regularization/
│   │   └── tv_regularization.jl   # Total Variation (ROF model)
│   ├── hybrid_ir/
│   │   └── hybrid_ir.jl           # Hybrid IR (FDK + PWLS refinement)
│   ├── mbir/
│   │   └── mbir.jl                # Model-Based IR
│   └── statistical_ir.jl          # PWLS core (used by Hybrid IR)
├── metrics/                        # Image quality (AAPM TG-233)
│   ├── mtf.jl                     # Modulation Transfer Function
│   ├── nps.jl                     # Noise Power Spectrum
│   └── psf.jl                     # Point Spread Function
├── scanners/                       # Clinical scanner configurations
│   ├── scanners.jl                # Scanner specifications
│   ├── general_electric.jl        # GE Revolution Apex
│   ├── siemens.jl                 # Siemens NAEOTOM Alpha
│   └── helical_protocols.jl       # Helical protocol integration
└── api/                           # Top-level orchestration
    ├── options.jl                 # SimOptions, ReconOptions
    ├── workspace.jl               # Workspace structs (pre-allocated buffers)
    └── driver.jl                  # simulate!(), reconstruct!() entry points
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

## Hybrid Iterative Reconstruction (HIR)

FDK initialization + PWLS (Penalized Weighted Least Squares) refinement with Huber edge-preserving regularization. Strength 1-5 provides progressive noise reduction.

```julia
# Create HIR workspace
hir_ws = create_hir_recon_workspace(geom, sinogram; strength=3, array_type=MtlArray)
reconstruct!(hir_ws, geom, sinogram)
hir_result = Array(hir_ws.volume)
```

### Noise Reduction by Strength

| Strength | Noise Reduction | Iterations | Use Case |
|----------|-----------------|------------|----------|
| 1 | ~9% | 8 | Minimal smoothing |
| 2 | ~20% | 15 | Moderate |
| 3 | ~32% | 30 | Balanced (recommended) |
| 4 | ~38% | 60 | Strong smoothing |
| 5 | ~38% | 100 | Maximum (SIRT ceiling) |

---

## Dose Validation and Reporting

Protocol validation and dose estimation functions (CPU-only, no GPU):

```julia
# Validate protocol parameters
result = validate_protocol(protocol, scanner)
println(result.valid, " — ", result.messages)

# Compute dose metrics
ctdi = compute_ctdi_vol(protocol)
dlp = compute_dlp(protocol, 30.0)  # 30 cm scan length

# Formatted dose report
report = dose_report(protocol, geom)

# Constant-dose protocol (same dose, different views)
proto_2000 = constant_dose_protocol(protocol, 2000)

# Constant-noise protocol (same noise/view, more dose)
proto_2000_noise = constant_noise_protocol(protocol, 2000)
```

---

## VMI (Virtual Monoenergetic Imaging)

### PCCT VMI (Material Decomposition)

Uses spectrum-weighted bin energies for accurate physics:

```julia
# Material decomposition from PCCT energy bins
mat_map = pcct_material_decomposition(pcct_sino; basis=(:water, :iodine), kvp=120)

# Synthesize VMI at target energy
vmi_70 = synthesize_vmi(mat_map, 70.0)

# Bin energies use spectrum-weighted mean (not arithmetic mean)
E_eff = compute_pcct_bin_energies([20.0, 35.0, 55.0, 70.0]; kvp=120)
```

### Dual-Energy VMI

```julia
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))
vmi_50 = virtual_monoenergetic(mat_map, 50.0)
```

### Physics Behavior

- **Water:** HU = 0 at all energies (by definition)
- **Iodine:** Maximum contrast at ~40-50 keV (above K-edge at 33.2 keV)
- **Calcium:** Contrast decreases monotonically (K-edge at 4 keV, below diagnostic)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Forward Projection (Siddon) | OK | TIGRE port |
| Polychromatic FP | OK | Beer-Lambert |
| FDK Reconstruction | OK | TIGRE port |
| SIRT/CGLS | OK | TIGRE port |
| Hybrid IR (HIR) | OK | Strength 1-5, PWLS-based |
| Physics Pipeline (10 effects) | OK | All GPU-native |
| Signal Chain (heel, BHC) | OK | CatSim-exact |
| Dose Validation/Reporting | OK | validate_protocol, dose_report |
| PCCT VMI | OK | Spectrum-weighted bin energies |
| Dual-Energy VMI | OK | GE GSI-style |
| PCCT Detector Physics | OK | Koch-Mehrin charge cloud, K-fluorescence, pileup, DRM |
| DAS Model | BROKEN | Guarded in driver.jl |
| Scatter | NO CORRECTION | Adds scatter but no correction |

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

Last Updated: 2026-02-09
