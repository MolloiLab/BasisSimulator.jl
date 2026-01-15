# BasisSimulator.jl vs CatSim/XCIST: Comprehensive Comparison

This document provides a detailed comparison between BasisSimulator.jl and CatSim/XCIST to identify feature gaps and guide future development.

**References:**
- [XCIST Paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10151073/)
- [CatSim Documentation Wiki](https://github.com/xcist/documentation/wiki)
- [CatSim Source Code](https://github.com/xcist/main)

---

## Executive Summary

| Category | BasisSimulator | CatSim | Gap Status |
|----------|---------------|--------|------------|
| Forward Projection | ✅ Complete | ✅ | Parity |
| Physics Effects | ✅ 10/12 effects | ✅ | Minor gaps |
| Calibration Pipeline | ⚠️ Partial | ✅ | **Significant gap** |
| Reconstruction | ✅ FDK/SIRT/CGLS | ✅ | Minor gaps |
| Scan Modes | ⚠️ Axial only | ✅ | **Significant gap** |
| Phantom Types | ⚠️ Voxelized only | ✅ | Moderate gap |

---

## 1. What BasisSimulator HAS (Complete)

### 1.1 Forward Projection
| Feature | Status | Implementation |
|---------|--------|----------------|
| Siddon ray tracing | ✅ | `Forward/Siddon.jl` - GPU native |
| Polychromatic projection | ✅ | `Forward/Polychromatic.jl` - Beer-Lambert |
| Cone-beam geometry | ✅ | Full 3D support |

### 1.2 Physics Effects (All GPU-Native)
| Effect | Status | Notes |
|--------|--------|-------|
| Scatter (convolution) | ✅ | Ohnesorge model, up to 63x63 kernel |
| Quantum noise (Poisson) | ✅ | Gaussian approximation for GPU |
| Electronic noise | ✅ | Gaussian additive |
| Detector efficiency/DQE | ✅ | Energy-dependent, NIST XCOM data |
| Bowtie filter | ✅ | Angle-dependent attenuation |
| Flat filter | ✅ | Multi-material support |
| Focal spot blur | ✅ | Convolution-based |
| Crosstalk | ✅ | X-ray + optical models |
| Detector lag | ✅ | Multi-exponential, parallel |
| Fill factor | ✅ | Row/column separable |
| Flying focal spot | ✅ | 2/4 position patterns |

### 1.3 Reconstruction
| Algorithm | Status | Notes |
|-----------|--------|-------|
| FDK | ✅ | Cosine weighting + ramp filter |
| SIRT | ✅ | TIGRE port |
| CGLS | ✅ | TIGRE port |
| Multiple filters | ✅ | Ramp, Shepp-Logan, Cosine, Hamming, Hann |

### 1.4 Spectrum & Materials
| Feature | Status | Notes |
|---------|--------|-------|
| Spectrum loading | ✅ | .dat files, 80-140 kVp |
| Spectrum downsampling | ✅ | Energy bin reduction |
| XrayAttenuation.jl integration | ✅ | NIST XCOM data |
| Material database | ✅ | Gammex 472 materials |

### 1.5 Geometry
| Feature | Status | Notes |
|---------|--------|-------|
| Clinical scanner configs | ✅ | Aquilion ONE, etc. |
| CTGeometry struct | ✅ | SAD, SDD, angles, detector size |
| Gammex 472 phantom | ✅ | With semantic masks |

---

## 2. What BasisSimulator is MISSING

### 2.1 Calibration/Preprocessing Pipeline (HIGH PRIORITY)

**CatSim outputs three scan types:**
```
[basename].air      - Air calibration scan (no phantom)
[basename].offset   - Offset scan (tube OFF, dark current)
[basename].scan     - Phantom projection data
```

**CatSim signal processing chain:**
```
Raw Signal → Offset Correction → Gain Correction → Log Transform → BHC → HU Calibration
```

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| **Air scan simulation** | HIGH | Scan with no phantom to get I₀ reference |
| **Offset scan** | HIGH | Dark current measurement (tube off) |
| **Gain/offset correction** | HIGH | `corrected = (raw - offset) / (air - offset)` |
| **Log transformation** | HIGH | `sinogram = -log(corrected)` - we do this but not with air reference |
| **Beam hardening correction** | HIGH | Water-based polynomial correction |
| **HU calibration** | MEDIUM | Proper -1000 offset, μ_water reference |

**Why this matters:**
- Clinical CT simulators produce raw detector counts, not line integrals
- Air scan provides the I₀ normalization reference
- BHC is essential for accurate HU values with polychromatic spectra
- Without proper calibration, HU values won't match clinical scanners

### 2.2 Beam Hardening Correction (HIGH PRIORITY)

CatSim implements water-based BHC via `callback_post_log`:

```python
# CatSim approach
# 1. Measure path lengths through water phantom at multiple thicknesses
# 2. Fit polynomial: true_path = a₀ + a₁×measured + a₂×measured² + ...
# 3. Apply correction to all projections
```

**BasisSimulator gap:**
- We simulate beam hardening (polychromatic creates cupping)
- We do NOT correct for it
- This causes HU inaccuracy, especially for dense materials

**Implementation needed:**
```julia
# Water-based BHC
struct BeamHardeningCorrection
    coefficients::Vector{Float64}  # Polynomial coefficients
    water_equivalent_path::Bool    # Whether to use water-equivalent path
end

function apply_bhc!(sinogram, bhc::BeamHardeningCorrection)
    # Apply polynomial correction: p_corrected = Σ aᵢ × pⁱ
end

function calibrate_bhc(spectrum, energies, weights; max_path_cm=50.0)
    # Generate calibration curve from water phantom simulation
end
```

### 2.3 Data Acquisition System (DAS) Model (MEDIUM PRIORITY)

CatSim has detailed DAS modeling:

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| DAS gain | MEDIUM | Electrons per keV X-ray (`cfg.das_gain`) |
| DAS quantization | LOW | LSB quantization (`cfg.das_lsb`) |
| DAS minimum value | LOW | Output truncation (`cfg.das_minvalue`) |
| Signal conversion | MEDIUM | keV → electrons → digital counts |

**Current BasisSimulator approach:**
- We work in line integral space (μ×path)
- We don't model the full signal conversion chain

### 2.4 Source Model Enhancements (MEDIUM PRIORITY)

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| Tube current (mA) scaling | MEDIUM | Per-view flux scaling |
| Target angle | LOW | Anode angle (typically 7°) |
| Spectrum generation | LOW | Parameterized model vs. lookup tables |
| 3D focal spot profiles | LOW | Uniform, Gaussian, user-defined |

### 2.5 Scan Modes (MEDIUM PRIORITY)

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| **Helical scanning** | MEDIUM | Spiral acquisition with z-interpolation |
| Fan-to-parallel rebinning | MEDIUM | Required for helical FDK |
| Short-scan support | LOW | Parker weighting for <360° |
| Axial step-and-shoot | LOW | Multi-rotation axial |

**Current status:**
- BasisSimulator has `Helical.jl` but reconstruction doesn't handle helical properly
- CatSim uses Tang's 3D weighting for helical cone-beam

### 2.6 Reconstruction Enhancements (LOW PRIORITY)

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| Tang's 3D weighting | LOW | Improved cone-beam handling |
| Multiple kernels | LOW | Soft, standard, bone |
| Parker weighting | LOW | Short-scan support |
| Iterative recon (TV, etc.) | LOW | Beyond SIRT/CGLS |

### 2.7 Phantom Types (LOW PRIORITY)

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| Analytical phantoms | LOW | Ellipsoids, cylinders with exact intersection |
| NURBS phantoms | LOW | Smooth surface representation |
| Polygonal phantoms | LOW | CAD-style geometry |

**Current status:**
- BasisSimulator uses voxelized phantoms only
- Analytical phantoms would allow exact ray-object intersection (no discretization)

### 2.8 Physics Effects (LOW PRIORITY)

| Missing Feature | Priority | Description |
|-----------------|----------|-------------|
| Monte Carlo scatter | LOW | More accurate than convolution |
| Heel effect | LOW | Anode self-attenuation |
| Photon counting detectors | LOW | Energy-resolved detection |

---

## 3. CatSim Signal Processing Chain (Reference)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CatSim Simulation Pipeline                    │
└─────────────────────────────────────────────────────────────────┘

1. SPECTRUM GENERATION
   ├── Load spectrum file (tungsten, 80-140 kVp)
   ├── Apply focal spot weighting
   └── Output: Photon fluence per energy bin

2. PRE-FILTRATION
   ├── Flat filter attenuation
   ├── Bowtie filter attenuation
   └── Output: Filtered spectrum at each detector position

3. PER-VIEW LOOP
   │
   ├── 3a. FLUX SCALING
   │       └── Scale by tube current (mA)
   │
   ├── 3b. RAY TRACING
   │       ├── Compute ray paths through phantom
   │       └── Output: Path lengths per material per energy
   │
   ├── 3c. ATTENUATION (Beer-Lambert)
   │       ├── I_transmitted = Σ w_e × exp(-Σ μ_e × path_e)
   │       └── Output: Transmitted intensity per detector pixel
   │
   ├── 3d. SCATTER (optional)
   │       ├── Convolution-based or Monte Carlo
   │       └── Output: Scatter contribution added
   │
   └── 3e. DETECTION
           ├── Apply DQE (energy & position dependent)
           ├── Poisson noise on photon counts
           ├── Energy integration (sum over bins)
           ├── Electronic noise (Gaussian)
           ├── Crosstalk convolution
           ├── Lag/afterglow (temporal)
           └── Output: Raw detector signal

4. POST-PROCESSING (to get sinogram)
   │
   ├── 4a. OFFSET CORRECTION
   │       └── signal_corrected = signal_raw - signal_offset
   │
   ├── 4b. GAIN CORRECTION (Air normalization)
   │       └── signal_norm = signal_corrected / (air_signal - offset)
   │
   ├── 4c. LOG TRANSFORMATION
   │       └── sinogram = -log(signal_norm)
   │
   └── 4d. BEAM HARDENING CORRECTION (optional)
           └── sinogram_bhc = polynomial(sinogram)

5. RECONSTRUCTION
   ├── FDK with cosine weighting
   ├── Ramp filter (various kernels)
   ├── Backprojection
   └── HU calibration: HU = 1000 × (μ - μ_water) / μ_water - 1000
```

---

## 4. BasisSimulator Current Pipeline (Comparison)

```
┌─────────────────────────────────────────────────────────────────┐
│                  BasisSimulator Current Pipeline                 │
└─────────────────────────────────────────────────────────────────┘

1. SPECTRUM LOADING
   ├── Load spectrum file
   ├── Downsample to N bins
   └── ✅ Same as CatSim

2. PRE-FILTRATION
   ├── Flat filter ✅
   ├── Bowtie filter ✅
   └── ✅ Same as CatSim

3. FORWARD PROJECTION
   ├── Create μ volume from mask + materials
   ├── Siddon ray tracing
   ├── Beer-Lambert per energy
   └── Output: Line integrals (NOT raw counts)
       ⚠️ DIFFERENCE: We output -log(I/I₀) directly, not raw I

4. PHYSICS EFFECTS (on line integrals)
   ├── Scatter ✅
   ├── Crosstalk ✅
   ├── Focal spot blur ✅
   ├── Detector efficiency ✅
   ├── Quantum noise ✅
   ├── Electronic noise ✅
   ├── Lag ✅
   └── Fill factor ✅

5. ❌ MISSING: CALIBRATION PIPELINE
   ├── ❌ No air scan
   ├── ❌ No offset scan
   ├── ❌ No gain/offset correction
   ├── ❌ No beam hardening correction
   └── ⚠️ Log transform happens inside forward_project

6. RECONSTRUCTION
   ├── FDK ✅
   ├── SIRT ✅
   ├── CGLS ✅
   └── HU conversion ✅ (but without BHC, values may be off)
```

---

## 5. Recommended Implementation Order

### Phase 1: Calibration Pipeline (HIGH IMPACT)

1. **Air scan simulation**
   - Add `simulate_air_scan(geom, spectrum)` function
   - Returns I₀ reference for each detector pixel

2. **Offset scan**
   - Add `simulate_offset_scan(geom, noise_model)` function
   - Returns dark current / electronic offset

3. **Raw signal mode**
   - Add option to output raw intensity (not log-transformed)
   - `forward_project(...; output_mode=:raw)` vs `:line_integral`

4. **Calibration correction**
   - `apply_calibration!(raw, air, offset)` → normalized signal
   - `apply_log_transform!(normalized)` → sinogram

5. **Beam hardening correction**
   - `calibrate_bhc(spectrum, water_paths)` → polynomial coeffs
   - `apply_bhc!(sinogram, bhc)` → corrected sinogram

### Phase 2: Scan Mode Enhancements (MEDIUM IMPACT)

1. **Helical reconstruction**
   - Fan-to-parallel rebinning
   - Z-interpolation
   - Tang's 3D weighting (optional)

2. **Short-scan support**
   - Parker weighting
   - Angular range < 360°

### Phase 3: Advanced Features (LOW IMPACT)

1. Monte Carlo scatter
2. Heel effect
3. Analytical phantoms
4. Photon counting detectors

---

## 6. Key Differences Summary

| Aspect | CatSim | BasisSimulator | Impact |
|--------|--------|----------------|--------|
| Output domain | Raw counts → calibrated | Line integrals directly | May affect noise modeling |
| Air calibration | Explicit air scan | Implicit I₀ | HU accuracy |
| BHC | Water-based polynomial | None | HU accuracy with poly spectrum |
| Scan modes | Axial + Helical | Axial only | Limited use cases |
| Phantoms | Analytical + Voxelized | Voxelized only | Inverse crime possible |

---

## 7. Validation Targets

To achieve CatSim parity, BasisSimulator should match:

1. **Air HU**: -1000 ± 5 HU (CatSim: -995 HU)
2. **Water HU**: 0 ± 5 HU (CatSim: 0 HU)
3. **Bone HU**: Within 2% of expected (CatSim: 1551 vs 1552 expected)
4. **Cupping**: < 10 HU variation across water phantom after BHC
5. **Noise texture**: Qualitatively similar to CatSim

---

Last Updated: 2025-01-14
