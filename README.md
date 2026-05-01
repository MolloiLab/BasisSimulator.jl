# BasisSimulator.jl

GPU-native CT simulation with backend-agnostic execution via [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl). Works on Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD), or CPU.

Core ray tracing ported from [TIGRE](https://github.com/CERN/TIGRE). Polychromatic physics and the full signal chain are our own implementation -- TIGRE is monochromatic only. The calibration workflow follows [CatSim/XCIST](https://github.com/xcist/main) exactly.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MolloiLab/BasisSimulator.jl")
```

For GPU support, also install your backend:
```julia
Pkg.add("Metal")     # Apple Silicon
Pkg.add("CUDA")      # NVIDIA
Pkg.add("AMDGPU")    # AMD
```

---

## The 5-Part API

Every simulation is defined by 5 structs. Configure each one, then call `simulate!()`:

```
                          simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Phantom  │  │ Scanner  │  │CTProtocol│  │SimOptions│  │  Recon   │
  │          │  │          │  │          │  │          │  │ Options  │
  │ mask     │  │ geometry │  │ kVp, mA  │  │ fidelity │  │ algo     │
  │ materials│  │ detector │  │ views    │  │ seed     │  │ fov      │
  │ voxel_sz │  │ source   │  │ dual_kVp │  │ use_*    │  │ vmi      │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
  What to scan   Hardware     Acquisition    Simulation     Output
                                              fidelity      config
```

### 1. Phantom -- What to Scan

A labeled 3D volume where each voxel holds a material index (UInt8), plus a materials dictionary and voxel dimensions.

```julia
import BasisSimulator as BS
import XrayAttenuation as XA
using Metal

# Define materials mapping label → material
materials_dict = Dict{Int, XA.Material}(
    0 => air_material,
    1 => XA.Materials.water,
    2 => bone_material,
    # ... up to 255 labels
)

# Voxel dimensions in cm
voxel_size_cm = (0.06, 0.06, 0.2)  # 0.6mm × 0.6mm × 2mm

# Create phantom — mask goes on GPU, materials stay on CPU
phantom_mask_gpu = Metal.MtlArray(labeled_array)
phantom = BS.Phantom(phantom_mask_gpu, materials_dict, voxel_size_cm)
```

The `Phantom()` constructor accepts any `AbstractArray{<:Integer, 3}` for the mask (CPU or GPU), a `Dict{Int, XA.Material}` mapping label values to materials, and a 3-tuple of voxel sizes in cm. It automatically computes the FOV and centers the phantom at isocenter.

For testing, a built-in Gammex 472 QA phantom is available via `create_gammex_472()`.

### 2. Scanner -- Hardware Configuration

Defines the physical scanner: source-detector distances, detector array, pixel pitch, filtration, and detector type (energy-integrating or photon-counting). All distances in mm.

**EICT (energy-integrating):**
```julia
scanner_eict = BS.Scanner(
    # Geometry
    source_to_isocenter = 625.6,  # mm (SID)
    source_to_detector = 1100.0,  # mm (SDD)

    # Detector array
    detector_rows = 64,
    detector_cols = 845,
    detector_row_size = 0.625,    # mm
    detector_col_size = 1.0,      # mm
    detector_shape = BS.CURVED_DETECTOR,

    # X-ray source
    focal_spot_width = 1.0,       # mm
    focal_spot_length = 1.0,      # mm
    target_angle = 7.0,           # degrees

    # Filtration
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,  # mm

    # Detector physics (GOS scintillator)
    detector_material = :gos,
    detector_depth = 3.0,         # mm
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    detection_gain = 1.0,
)
```

**PCCT (photon-counting) -- adds energy-resolved fields:**
```julia
scanner_pcct = BS.Scanner(
    # Geometry
    source_to_isocenter = 595.0,
    source_to_detector = 1085.5,

    # Detector array
    detector_rows = 64,
    detector_cols = 2189,
    detector_row_size = 0.4,      # mm (2×2 binned from 0.2mm native)
    detector_col_size = 0.4,
    detector_shape = BS.CURVED_DETECTOR,
    detector_row_offset = 0.0,
    detector_col_offset = 0.2,    # quarter-detector offset

    # X-ray source
    focal_spot_width = 0.4,
    focal_spot_length = 0.5,
    target_angle = 7.0,

    # Filtration
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,

    # CdTe direct-conversion detector
    detector_material = :cdte,
    detector_depth = 1.6,
    fill_factor_row = 0.95,
    fill_factor_col = 0.95,
    detection_gain = 1.0,
    electronic_noise = 0.0,

    # PCCT-specific fields
    detector_type = :photon_counting,
    n_energy_bins = 4,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0],  # keV
    energy_resolution = 10.0,       # keV FWHM
    charge_sharing_fwhm = 0.08,     # mm
    dead_time_ns = 5.0,             # ns
    pixel_mode = :standard,         # :standard, :uhr, :macro
)
```

The `detector_type` field determines the simulation path: `:energy_integrating` (default) runs the standard EICT pipeline, `:photon_counting` activates the full PCCT detector model (charge cloud transport, K-fluorescence, pileup, DRM).

Every Scanner field that maps to a physics effect is only used when the corresponding `use_*` flag in SimOptions is `true`. For example, `flat_filter_material` and `flat_filter_thickness` are only read when `use_flat_filter=true`.

### 3. CTProtocol -- Acquisition Parameters

Defines how the scan is acquired: tube voltage, current, number of views, and dual-energy settings.

```julia
# Single-kVp
protocol = BS.CTProtocol(kVp=120.0, mA=300.0, views=984, rotation_time=0.5)

# Dual-energy (kVp/mA are HIGH energy; kVp_low/mA_low are LOW energy)
protocol_dual = BS.CTProtocol(
    dual_energy = true,
    kVp = 140.0,   mA = 200.0,      # high energy
    kVp_low = 80.0, mA_low = 350.0,  # low energy
    views = 984,
    rotation_time = 0.5,
)

# PCCT (always single-kVp -- spectral info comes from energy bins, not dual kVp)
protocol_pcct = BS.CTProtocol(kVp=140.0, mA=300.0, views=984)
```

### 4. SimOptions -- Simulation Fidelity

Controls which physics effects are enabled. The `fidelity` preset sets defaults for all 15 `use_*` toggles; individual overrides win.

```julia
# Full physics (all 13 effects except DAS which is broken)
sim_opts = BS.SimOptions(fidelity=:high, seed=42)

# PCCT with noise reduction (approximates clinical vendor reconstruction like Siemens QIR)
sim_opts_pcct = BS.SimOptions(fidelity=:high, pcct_noise_reduction=0.60, seed=42)
```

**Fidelity presets:**

| Preset | Effects Enabled |
|--------|----------------|
| `:ideal` | None -- pure geometric ray tracing |
| `:low` | Noise only |
| `:medium` | Noise + focal spot + crosstalk + flat filter + BHC |
| `:high` | All 13 effects except DAS (broken) |
| `:pcct` | Same as `:high` + PCCT detector corrections |

**All `use_*` toggles:**

| Toggle | What it controls |
|--------|-----------------|
| `use_fill_factor` | Detector active area fraction |
| `use_flat_filter` | Inherent Al/Cu filtration (from Scanner fields) |
| `use_bowtie_filter` | Angle-dependent beam shaping |
| `use_detector_efficiency` | Energy-dependent DQE (from Scanner fields) |
| `use_scatter` | Patient scatter (spatial convolution) |
| `use_scatter_correction` | Scatter removal estimate |
| `use_crosstalk` | Electronic pixel-to-pixel coupling |
| `use_optical_crosstalk` | Scintillator light spread |
| `use_focal_spot` | Geometric blur from finite source (from Scanner fields) |
| `use_noise` | Poisson quantum noise |
| `use_lag` | Temporal persistence / afterglow |
| `use_heel_effect` | Anode self-attenuation (from Scanner fields) |
| `use_das` | DAS model (always false -- broken) |
| `use_bhc` | Water-based beam hardening correction |
| `use_pcct_corrections` | Inverse pileup + inverse charge sharing (PCCT only) |

**Other fields:**
- `pcct_noise_reduction::Float64` -- 0.0 (raw, default) to 1.0 (maximum). Only affects PCCT sinogram noise.
- `n_energy_bins::Int` -- Number of spectrum bins for polychromatic mode (default 30).
- `seed::Union{Int, Nothing}` -- Random seed for reproducibility (default 42).

### 5. ReconOptions -- Reconstruction Output

Controls reconstruction algorithm, output matrix size, VMI, and iterative parameters.

```julia
# FDK (standard filtered backprojection)
recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (512, 512, 64),
    fov_cm = 35.0,
)

# FDK with VMI (for dual-energy or PCCT)
recon_opts_vmi = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (512, 512, 64),
    fov_cm = 35.0,
    vmi_energies = [40.0, 70.0, 100.0, 140.0],
    vmi_basis = (:water, :iodine),
)

# PCCT with 3-material VMI basis
recon_opts_pcct = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (512, 512, 64),
    fov_cm = 35.0,
    vmi_energies = [40.0, 70.0, 100.0, 140.0],
    vmi_basis = [:water, :iodine, :calcium],
)
```

**Core fields:** `algorithm` (`:fdk`, `:sirt`, `:cgls`, `:tv_sirt`, `:tv_cgls`, `:asir`, `:mbir`), `matrix_size`, `fov_cm`, `filter` (`:ram_lak` default).

**Iterative fields:** `iterations`, `lambda`, `tv_weight`, `n_subsets`, `penalty` (`:quadratic`, `:huber`, `:hyperbola`).

**VMI fields:** `vmi_energies` (keV values), `vmi_basis` (material symbols for decomposition).

---

## simulate!() and reconstruct!()

The workspace pattern pre-allocates all GPU buffers once, then runs with zero allocations. This is how the verification notebooks run all simulations.

### EICT Single-kVp Workflow

```julia
# 1. Create workspace (allocates all GPU buffers once)
ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)

# 2. Simulate (writes into workspace buffers)
BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

# 3. FDK reconstruction
recon_size = (512, 512, 64)
ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
fdk_hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
    μ_water = μ_water
)

# 4. Hybrid IR (strength 3) reconstruction
ws_hir = BS.create_hir_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; strength=3)
hir_hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_hir, ws.sino_noisy_out, ws.geom, recon_size));
    μ_water = μ_water
)

# 5. Free GPU memory
ws = nothing; ws_fdk = nothing; ws_hir = nothing
GC.gc(true)
```

### EICT Dual-kVp Workflow

```julia
# 1. Create separate workspaces for low and high kVp
protocol_low = BS.CTProtocol(kVp=80.0, mA=350.0, views=984, rotation_time=0.5)
protocol_high = BS.CTProtocol(kVp=140.0, mA=200.0, views=984, rotation_time=0.5)

ws_low = BS.create_eict_workspace(scanner, protocol_low, sim_opts, recon_opts, phantom)
ws_high = BS.create_eict_workspace(scanner, protocol_high, sim_opts, recon_opts, phantom)

# 2. Simulate each kVp independently
BS.simulate!(ws_low, phantom, scanner, protocol_low, sim_opts, recon_opts)
BS.simulate!(ws_high, phantom, scanner, protocol_high, sim_opts, recon_opts)

# 3. Reconstruct each kVp
ws_fdk_low = BS.create_fdk_recon_workspace(ws_low.sino_noisy_out, ws_low.geom, recon_size)
fdk_80_hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk_low, ws_low.sino_noisy_out, ws_low.geom, recon_size));
    μ_water = μ_water_low
)

# 4. Material decomposition + VMI from the two sinograms
de_sino = BS.DualEnergySinogram(ws_low.sino_noisy_out, ws_high.sino_noisy_out;
    low_kvp=80, high_kvp=140)
mat_map = BS.spectral_decompose(de_sino)
for E in [40.0, 70.0, 100.0, 140.0]
    vmi_sino = BS.virtual_monoenergetic(mat_map, E)
    ws_fdk_vmi = BS.create_fdk_recon_workspace(vmi_sino, ws_low.geom, recon_size)
    vmi_vol = Array(BS.reconstruct!(ws_fdk_vmi, vmi_sino, ws_low.geom, recon_size))
    ws_fdk_vmi = nothing; GC.gc(true)
end
```

### PCCT Workflow

```julia
# 1. Create PCCT workspace (auto-detected from scanner.detector_type)
ws = BS.create_workspace(scanner_pcct, protocol_pcct, sim_opts_pcct, recon_opts, phantom)

# 2. Simulate — returns energy-resolved sinograms + material maps
result = BS.simulate!(ws, phantom, scanner_pcct, protocol_pcct, sim_opts_pcct, recon_opts)
mat_map = result.mat_map

# 3. Reconstruct from combined sinogram
ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
fdk_hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
    μ_water = μ_water_pcct
)

# 4. VMI synthesis (reuses workspace buffer via ws_output)
for E in [40.0, 70.0, 100.0, 140.0]
    vmi_sino = BS.virtual_monoenergetic(mat_map, E; ws_output=ws.vmi_sino)
    ws_fdk_vmi = BS.create_fdk_recon_workspace(vmi_sino, ws.geom, recon_size)
    vmi_vol = Array(BS.reconstruct!(ws_fdk_vmi, vmi_sino, ws.geom, recon_size))
    ws_fdk_vmi = nothing; GC.gc(true)
end
```

### Workspace Types

| Workspace | Creator | Use Case |
|-----------|---------|----------|
| `EICTWorkspace` | `create_eict_workspace(...)` | Single-kVp energy-integrating CT |
| `PCCTWorkspace` | `create_workspace(...)` | Photon-counting CT |
| `FDKReconWorkspace` | `create_fdk_recon_workspace(...)` | FDK reconstruction |
| `HIRReconWorkspace` | `create_hir_recon_workspace(...)` | Hybrid IR (FDK init + PWLS refinement) |

### The Allocating Path: simulate()

For quick prototyping, `simulate()` (no `!`) handles workspace creation internally:

```julia
result = BS.simulate(phantom, scanner, protocol, sim_opts, recon_opts)
result.sinogram_noisy      # noisy sinogram
result.reconstruction      # first reconstruction volume
result.vmi_volumes         # Dict{Float64, Array} if VMI requested
```

---

## GPU Memory Management

Workspaces hold references to GPU arrays. The verification notebooks use `let` blocks to scope GPU lifetimes:

```julia
(fdk_hu, hir_hu) = let
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

    # FDK
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
    fdk = BS.to_hounsfield(
        Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
        μ_water = μ_water
    )
    ws_fdk = nothing; GC.gc(true)

    # Hybrid IR
    ws_hir = BS.create_hir_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; strength=3)
    hir = BS.to_hounsfield(
        Array(BS.reconstruct!(ws_hir, ws.sino_noisy_out, ws.geom, recon_size));
        μ_water = μ_water
    )
    ws_hir = nothing; ws = nothing; GC.gc(true)

    (fdk, hir)  # only CPU arrays survive
end
```

The GPU backend is determined by the phantom mask array type. `MtlArray` for Metal, `CuArray` for CUDA, or plain `Array` for CPU. All workspace buffers are allocated on the same device automatically.

---

## The Imaging Chain

The `src/` directory mirrors the physical X-ray imaging chain:

```
  X-ray Tube                    Patient/Phantom              Scanner Geometry
  ──────────                    ───────────────              ────────────────
  source/                       object/                      geometry/
    spectrum.jl                   phantom.jl                   scanner.jl
    bowtie_filter.jl              materials.jl
    flat_filter.jl                attenuation.jl
    heel_effect.jl
    focal_spot.jl
    protocol.jl
          │                           │                            │
          └───────────┬───────────────┘                            │
                      ▼                                            │
              Forward Projection                                   │
              ──────────────────                                   │
              projection/                                          │
                siddon.jl           ◄──────────────────────────────┘
                polychromatic.jl
                      │
                      ▼
              Detector Physics
              ────────────────
              detector/
                physics_pipeline.jl ─── orchestrates all effects
                  scatter.jl              crosstalk.jl
                  detector_noise.jl       detector_lag.jl
                  detector_efficiency.jl  fill_factor.jl
                  focal_spot.jl           das_model.jl
                photon_counting.jl
                └── pcct/
                    cdte_constants.jl     charge_collection.jl
                    charge_transport.jl   pileup_model.jl
                    k_fluorescence.jl     detector_response.jl
                      │
                      ▼
              Corrections & Spectral
              ──────────────────────
              correction/                spectral/
                calibration.jl             pcct_spectral.jl
                beam_hardening_correction  dual_energy.jl
                      │
                      ▼
              Reconstruction
              ──────────────
              reconstruction/
                core/         backprojection.jl, filtering.jl
                fbp/          fdk.jl, helical_recon.jl
                ir/           sirt.jl, cgls.jl
                hybrid_ir/    hybrid_ir.jl
                mbir/         mbir.jl
                regularization/  tv_regularization.jl
                statistical_ir.jl
                      │
                      ▼
              Image Quality Metrics
              ─────────────────────
              metrics/
                mtf.jl   nps.jl   psf.jl

  Orchestration:  api/driver.jl, api/workspace.jl, api/options.jl
  Scanner Presets: scanners/general_electric.jl, scanners/siemens.jl
```

### Signal Chain Order

When all effects are enabled (`fidelity=:high`), `simulate!()` runs this pipeline:

1. **Polychromatic forward projection** -- Beer-Lambert integration across spectrum energies
2. **Fill factor** -- detector active area fraction
3. **Flat filter** -- inherent Al/Cu filtration
4. **Scatter** -- patient scatter (spatial convolution kernel)
5. **Scatter correction** -- removal estimate
6. **Bowtie filter** -- angle-dependent beam shaping
7. **Crosstalk** -- electronic pixel-to-pixel coupling
8. **Optical crosstalk** -- scintillator light spread
9. **Focal spot** -- geometric blur from finite source
10. **Detector efficiency** -- scintillator DQE
11. **Noise** -- Poisson quantum noise (I0-scaled Gaussian approximation)
12. **Detector lag** -- temporal persistence / afterglow
13. **Heel effect** -- anode self-attenuation (intensity domain)
14. **Air scan calibration** -- noise-free reference normalization (CatSim-exact)
15. **Log transform** -- intensity to line-integral domain
16. **Beam hardening correction** -- water-based polynomial BHC

---

## Reconstruction

### FDK (Feldkamp-Davis-Kress)

Standard filtered backprojection for cone-beam CT:

```julia
ws_fdk = BS.create_fdk_recon_workspace(sinogram, geom, (512, 512, 64))
volume = BS.reconstruct!(ws_fdk, sinogram, geom, (512, 512, 64))
hu = BS.to_hounsfield(Array(volume); μ_water=μ_water)
```

### Hybrid IR

FDK initialization + PWLS (Penalized Weighted Least Squares) refinement with Huber edge-preserving regularization. Modeled after clinical Hybrid IR (ASIR-V, SAFIRE):

```julia
ws_hir = BS.create_hir_recon_workspace(sinogram, geom, (512, 512, 64); strength=3)
volume = BS.reconstruct!(ws_hir, sinogram, geom, (512, 512, 64))
hu = BS.to_hounsfield(Array(volume); μ_water=μ_water)
```

| Strength | Noise Reduction | Iterations | Use Case |
|----------|-----------------|------------|----------|
| 1 | ~9% | 8 | Minimal smoothing, preserve texture |
| 2 | ~20% | 15 | Moderate |
| 3 | ~32% | 30 | Balanced (recommended) |
| 4 | ~38% | 60 | Strong smoothing |
| 5 | ~38% | 100 | Maximum |

### Also Available

- **SIRT** / **CGLS** -- algebraic iterative reconstruction
- **MBIR** -- model-based iterative reconstruction
- **TV regularization** -- total variation (ROF model)

---

## Spectral Imaging

### Dual-Energy VMI

After dual-kVp `simulate!()`, the returned `mat_map` enables VMI synthesis at any energy:

```julia
result = BS.simulate!(ws, phantom, scanner, protocol_dual, sim_opts, recon_opts)
mat_map = result.mat_map

vmi_sino = BS.virtual_monoenergetic(mat_map, 70.0)  # 70 keV
```

### PCCT VMI

After PCCT `simulate!()`, the same `virtual_monoenergetic()` function works with PCCT material maps. Use `ws_output` to reuse a pre-allocated buffer:

```julia
result = BS.simulate!(ws, phantom, scanner_pcct, protocol_pcct, sim_opts_pcct, recon_opts)
mat_map = result.mat_map

vmi_sino = BS.virtual_monoenergetic(mat_map, 40.0; ws_output=ws.vmi_sino)
```

### Expected VMI Physics

- **Water**: HU = 0 at all energies (by definition)
- **Iodine**: Maximum contrast at ~40-50 keV (K-edge at 33.2 keV)
- **Calcium**: Contrast decreases monotonically with energy (K-edge at 4 keV, below diagnostic range)

---

## PCCT Detector Physics (CdTe)

When `fidelity=:high` or `:pcct` with a photon-counting scanner, the full CdTe detector model runs:

| Component | Model | Reference |
|-----------|-------|-----------|
| Charge cloud transport | Koch-Mehrin ODE (sigma ~12-14 um) | Koch-Mehrin 2020 (NIM-A) |
| K-fluorescence | 5 K-lines per element, Te to Cd cascade | Koch-Mehrin Table 1 |
| Charge collection | Hecht CCE + Barrett small-pixel weighting | Barrett 1995 |
| Pileup | Seminonparalyzable model | Yang 2025 |
| DRM | Unified detector response matrix (FWHM ~3.55 keV) | Konrad 2025 (PMB) |

---

## Verification & Examples

Verification notebooks (CatSim baseline, dual-kVp VMI vs NIST, PCCT physics, XCAT clinical workflows) live in the companion repo [**MolloiLab/basis-verification**](https://github.com/MolloiLab/basis-verification). It carries its own `Project.toml` and pulls `BasisSimulator` directly from this GitHub URL via `[sources]`, so no sibling-checkout is required.

---

## References

- **TIGRE**: Biguri A, et al. "TIGRE: A MATLAB-GPU toolbox for CBCT image reconstruction." *Biomed Phys Eng Express.* 2016. [GitHub](https://github.com/CERN/TIGRE)
- **CatSim/XCIST**: GE Healthcare CT simulation tools. [GitHub](https://github.com/xcist/main)
- **Koch-Mehrin 2020**: Koch-Mehrin KAF, et al. "Charge transport in CdTe photon-counting detectors." *NIM-A* 976:164241.
- **Konrad 2025**: Konrad U, et al. "Validated NAEOTOM Alpha MC model." *PMB* 70:065004.
- **Yang 2025**: Yang Q, et al. "Seminonparalyzable pileup model for PCCT."
- **Feldkamp 1984**: Feldkamp LA, et al. "Practical cone-beam algorithm." *J Opt Soc Am A.*
- **Siddon 1985**: Siddon RL. "Fast calculation of the exact radiological path." *Med Phys.*

## License

MIT
