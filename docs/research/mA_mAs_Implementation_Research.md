# mA/mAs Implementation Research for BasisSimulator.jl

> **Document Purpose:** Research document for implementing clinical mA/mAs parameters in BasisSimulator.jl to enable realistic dose-based noise modeling.
>
> **Story ID:** RESEARCH-MA-MAS
>
> **Status:** COMPLETE
>
> **Last Updated:** 2026-01-16

---

## Executive Summary

This document researches the implementation of true clinical mA (milliampere) and mAs (milliampere-seconds) parameters for CT simulation. Currently, BasisSimulator uses an abstract photon count parameter `I0` that controls noise levels. This research determines how to map clinical mA values to realistic photon counts, enabling users to specify dose parameters that match real CT scanner interfaces.

### Key Findings

1. **Linear Relationship:** Photon flux is directly proportional to tube current (mA)
2. **CatSim Standard:** Uses flux density of ~2×10⁶ photons/mA/mm²/s at 1 meter
3. **GE Revolution Apex:** Supports 10-740 mA (10-1300 mA at low kVp) with 5 mA increments
4. **Implementation Path:** Calibrate I0 using scanner-specific flux density and geometry

---

## 1. Current Implementation Analysis

### 1.1 Where I0 is Defined

The `I0` parameter is defined in `src/Forward/DetectorNoise.jl`:

```julia
struct DetectorModel
    blur_fwhm::Float64           # Detector PSF FWHM in pixels
    I0::Float64                  # Incident photon count (photons/detector element)
    electronic_noise_std::Float64 # Electronic noise σ
    seed::Union{Nothing, Int}    # Random seed
end

function default_detector_model(;
    blur_fwhm::Float64=1.5,
    I0::Float64=1e5,           # Default: 100,000 photons/detector
    electronic_noise_std::Float64=10.0,
    seed::Union{Nothing, Int}=nothing
)
```

### 1.2 How I0 Affects Noise

In the quantum noise model (Poisson statistics):

```
λ = I₀ × exp(-∫μ dl)    # Expected photon count at detector
N ~ Poisson(λ)           # Detected photons
σ_p = 1/√λ              # Noise std in projection domain
```

The code in `add_quantum_noise!` implements:

```julia
# Convert projection to intensity
λ = I0 * exp(-sinogram[idx])

# Gaussian approximation of Poisson noise (valid for λ > 100)
λ_noisy = λ + sqrt(max(λ, T(1))) * rand_gpu[idx]

# Convert back to attenuation
sinogram[idx] = -log(λ_noisy / I0)
```

### 1.3 Current I0 Ranges in Use

| Config Function | I0 Value | Noise Level Description |
|-----------------|----------|-------------------------|
| `default_detector_model()` | 1×10⁵ | Medium noise |
| `realistic_physics_config(noise_level=1.0)` | 1×10⁶ | Low noise (clinical standard) |
| `realistic_physics_config(noise_level=2.0)` | 5×10⁵ | Higher noise (low dose) |
| `full_physics_config(noise_level=1.0)` | 1×10⁶ | Full physics, clinical |

---

## 2. Physics of mA and Photon Flux

### 2.1 Fundamental Relationships

The tube current (mA) directly controls the electron flux striking the anode:

**Electrons per second:**
```
electrons/s = mA × 10⁻³ × 6.24×10¹⁸ electrons/coulomb
```

For 100 mA: 6.24×10¹⁷ electrons/second

**X-ray Production Efficiency:**
Only ~1% of electron kinetic energy converts to X-rays (bremsstrahlung), with efficiency:
```
η ≈ 9×10⁻¹⁰ × Z × V
```
Where Z is atomic number (74 for tungsten) and V is tube voltage.

At 120 kV: η ≈ 0.8% efficiency

### 2.2 Photon Flux Formula

The photon flux at the detector depends on:

```
Φ(mA, t, d, A, η_det) = (mA × t × flux_density × A × η_det) / d²
```

Where:
- `mA` = tube current (milliamperes)
- `t` = exposure time per projection (seconds)
- `flux_density` = source photon output per mA (scanner-specific)
- `A` = detector element area (mm²)
- `η_det` = detector quantum efficiency
- `d` = source-to-detector distance (mm)

### 2.3 CatSim/XCIST Reference Values

From XCIST documentation and publications [1]:

| Parameter | Value | Source |
|-----------|-------|--------|
| Flux density | 2×10⁶ photons/mA/mm²/s | XCIST default |
| Reference distance | 1000 mm | Standard |
| Electronic noise | 20,000 electrons σ | ~20 X-ray equivalents |

**Formula from CatSim:**
```
I₀ = flux_density × mA × t_proj × (1000/SDD)² × A_det × η_det
```

---

## 3. GE Revolution Apex Specifications

### 3.1 Tube Current Range

From GE documentation and FDA 510(k) submissions [2,3]:

| kVp | mA Range | Step | Max mAs/rotation |
|-----|----------|------|------------------|
| 70 kVp | 10-1300 mA | 5 mA | 1300 (0.5s) |
| 80 kVp | 10-1300 mA | 5 mA | 1300 (0.5s) |
| 100 kVp | 10-1000 mA | 5 mA | 500 (0.5s) |
| 120 kVp | 10-740 mA | 5 mA | 370 (0.5s) |
| 140 kVp | 10-555 mA | 5 mA | 277 (0.5s) |

### 3.2 Typical Clinical Protocols

| Protocol | kVp | mA | Rotation | mAs/rotation |
|----------|-----|-----|----------|--------------|
| Chest (standard) | 120 | 400 | 0.5s | 200 |
| Chest (low dose) | 100 | 100 | 0.5s | 50 |
| Abdomen (standard) | 120 | 450 | 0.5s | 225 |
| Head (standard) | 120 | 300 | 1.0s | 300 |
| Cardiac | 120 | 600 | 0.28s | 168 |
| Pediatric | 80 | 150 | 0.5s | 75 |

### 3.3 Detector Geometry

For GE Revolution Apex:
- SID = 626 mm
- SDD = 1097 mm
- Detector element: ~1.05 mm × 0.625 mm at isocenter
- At detector: ~1.84 mm × 1.10 mm (magnification = 1.752)
- Element area at detector: ~2.0 mm²

---

## 4. Proposed Implementation

### 4.1 New API Design

```julia
"""
    mA_to_I0(mA, scanner; rotation_time_s=0.5, n_views=984) -> Float64

Convert clinical mA setting to photon count (I0) per detector element.

# Arguments
- `mA::Float64`: Tube current in milliamperes
- `scanner::ScannerSpec`: Scanner specification (provides SDD, detector size)

# Keyword Arguments
- `rotation_time_s::Float64=0.5`: Gantry rotation time in seconds
- `n_views::Int=984`: Number of projection views per rotation

# Returns
- `I0::Float64`: Photon count per detector element per projection
"""
function mA_to_I0(mA::Float64, scanner::AbstractScannerSpec;
                  rotation_time_s::Float64=0.5,
                  n_views::Int=984)

    # Get scanner geometry
    SDD = source_to_detector_distance(scanner) / 1000.0  # Convert mm to m
    det_area_mm2 = detector_element_area(scanner)  # At detector

    # CatSim reference flux density
    flux_density = 2e6  # photons/mA/mm²/s at 1m

    # Time per projection
    t_proj = rotation_time_s / n_views

    # Distance scaling (inverse square law from 1m reference)
    distance_factor = (1.0 / SDD)^2

    # Detector quantum efficiency (typical GOS)
    η_det = 0.85

    # Calculate I0
    I0 = flux_density * mA * t_proj * distance_factor * det_area_mm2 * η_det

    return I0
end
```

### 4.2 Enhanced DetectorModel

```julia
"""
    DetectorModel with mA support
"""
struct DetectorModel
    blur_fwhm::Float64
    I0::Float64  # Computed from mA or set directly
    electronic_noise_std::Float64
    seed::Union{Nothing, Int}

    # Optional clinical parameters (for reference)
    mA::Union{Nothing, Float64}
    rotation_time_s::Union{Nothing, Float64}
end

"""
    clinical_detector_model(; mA, scanner, rotation_time_s=0.5, ...)

Create detector model with clinical mA specification.
"""
function clinical_detector_model(;
    mA::Float64,
    scanner::AbstractScannerSpec,
    rotation_time_s::Float64=0.5,
    n_views::Int=984,
    blur_fwhm::Float64=1.5,
    electronic_noise_std::Float64=10.0,
    seed::Union{Nothing, Int}=nothing
)
    I0 = mA_to_I0(mA, scanner; rotation_time_s, n_views)
    return DetectorModel(blur_fwhm, I0, electronic_noise_std, seed, mA, rotation_time_s)
end
```

### 4.3 PhysicsConfig Integration

```julia
"""
    default_physics_config(; mA=nothing, scanner=nothing, ...)

Extended physics config with optional mA support.
"""
function default_physics_config(;
    # Existing parameters
    noise::Union{Nothing, DetectorModel}=nothing,
    # NEW: clinical mA parameters
    mA::Union{Nothing, Float64}=nothing,
    scanner::Union{Nothing, AbstractScannerSpec}=nothing,
    rotation_time_s::Float64=0.5,
    n_views::Int=984,
    ...
)
    # If mA specified, compute I0 and create detector model
    if mA !== nothing && scanner !== nothing
        noise = clinical_detector_model(
            mA=mA, scanner=scanner,
            rotation_time_s=rotation_time_s,
            n_views=n_views
        )
    end
    ...
end
```

### 4.4 Example Usage

```julia
using BasisSimulator

# Get scanner specification
spec = GERevolutionApex()

# Create physics config with clinical mA
physics = default_physics_config(
    mA = 400.0,                    # 400 mA tube current
    scanner = spec,
    rotation_time_s = 0.5,         # 0.5s rotation
    n_views = 984,                 # 984 views per rotation
    fill_factor = fill_factor_standard(),
    flat_filter = flat_filter_al(3.0),
    bowtie_filter = bowtie_filter_large_body(),
    energy_keV = 60.0
)

# Or create detector model directly
detector = clinical_detector_model(
    mA = 200.0,
    scanner = spec,
    rotation_time_s = 0.5
)
println("mA: 200, I0: $(detector.I0)")  # ~3.5e4 photons/detector/projection
```

---

## 5. Calibration and Validation

### 5.1 Expected I0 Ranges

Using the proposed formula for GE Revolution Apex (984 views per rotation):

| mA | Rotation | I0 (photons/det/proj) | Noise Level |
|----|----------|----------------------|-------------|
| 100 | 0.5s | ~145,000 | High (low dose) |
| 200 | 0.5s | ~290,000 | Medium-high |
| 400 | 0.5s | ~580,000 | Medium (standard) |
| 600 | 0.5s | ~870,000 | Medium-low |
| 740 | 0.5s | ~1,070,000 | Low (max at 120kVp) |

**Note:** These values are consistent with CatSim/XCIST which uses flux density of 2×10⁶ photons/mA/mm²/s at 1m reference distance.

### 5.2 Validation Approach

1. **Noise vs mAs relationship:** Image noise σ should scale as 1/√(mAs)
2. **Clinical comparison:** Compare simulated noise levels to published clinical data
3. **CTDIvol correlation:** Higher mA → higher CTDIvol → lower noise

### 5.3 CTDI Relationship

For dose estimation, CTDI relates to photon flux:
```
CTDIvol ∝ mAs × (kVp)^n / pitch
```
Where n ≈ 2.6 for CT.

The simulator can optionally output estimated CTDIvol for clinical reference:
```julia
function estimate_ctdivol(mA, kVp, rotation_time_s, pitch, scanner)
    # Simplified CTDI estimation (scanner-specific calibration needed)
    mAs = mA * rotation_time_s
    ctdi_factor = scanner.ctdi_calibration_factor  # mGy per mAs at 120kVp
    kVp_factor = (kVp / 120.0)^2.6
    return ctdi_factor * mAs * kVp_factor / pitch
end
```

---

## 6. Implementation Plan

### Phase 1: Core mA Support
1. Add `mA_to_I0()` function in `src/Forward/DetectorNoise.jl`
2. Add `clinical_detector_model()` constructor
3. Update `default_physics_config()` with optional mA parameters
4. Add tests for mA → I0 conversion

### Phase 2: Scanner Integration
1. Add `detector_element_area()` method to scanner interface
2. Add scanner-specific flux calibration factors
3. Implement for GE Revolution Apex

### Phase 3: Validation
1. Create validation example comparing noise vs mA
2. Compare to published clinical noise data
3. Document calibration procedure

### Phase 4: Advanced Features (optional)
1. Tube current modulation (TCM) support
2. CTDI estimation
3. DLP calculation

---

## 7. References

1. De Man B, Basu S, Chandra N, et al. "XCIST—an open access x-ray/CT simulation toolkit." Phys Med Biol. 2022;67(18). PMC10151073
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10151073/

2. FDA 510(k) K213715 - GE Revolution Apex Elite CT
   https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf

3. GE HealthCare Revolution Apex Specifications
   https://www.gehealthcare.com/products/computed-tomography/revolution-apex

4. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and Recent Advances." 3rd ed. SPIE Press, 2015. Chapter 7: Noise.

5. AAPM Task Group 204: "Size-Specific Dose Estimates (SSDE) in Pediatric and Adult Body CT Examinations."
   https://www.aapm.org/pubs/reports/RPT_204.pdf

6. SUNY Upstate Radiology - CT Radiographic Techniques
   https://www.upstate.edu/radiology/education/rsna/ct/technique.php

7. Correlation between X-ray tube current exposure time and photon number. PMC10754355
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10754355/

---

## Appendix A: Derivation of I0 Formula

### Step 1: Electron Flux

```
electrons/s = (mA × 10⁻³ C/s) × (6.24×10¹⁸ e⁻/C)
            = mA × 6.24×10¹⁵ electrons/s
```

### Step 2: X-ray Production

X-ray production efficiency: η ≈ 1% (varies with kVp)
```
photons/s ≈ mA × 6.24×10¹³ × η_spectrum
```

### Step 3: Solid Angle to Detector

For detector element at distance SDD:
```
Ω = A_det / SDD²
fraction = Ω / (4π) = A_det / (4π × SDD²)
```

### Step 4: Per-Projection Count

```
I0 = (photons/s) × t_proj × (A_det / SDD²) × η_det

   = flux_density × mA × t_proj × (1/SDD)² × A_det × η_det
```

Where `flux_density` incorporates electron flux, X-ray efficiency, and geometric factors into a single calibration constant (~2×10⁶ photons/mA/mm²/s at 1m for typical diagnostic CT).

---

## Appendix B: Noise Relationships

### Quantum Noise in Projection Domain

```
σ_projection = 1 / √(I0 × exp(-μL))
             = 1 / √I0 × exp(μL/2)
```

### Noise in Reconstructed Image (HU)

After FDK reconstruction with ramp filter:
```
σ_HU ∝ 1 / √(mAs) × f(object_size, kVp, reconstruction_kernel)
```

Doubling mAs reduces noise by factor of √2 ≈ 1.41

### Clinical Reference Noise Levels

| Protocol | kVp | mAs | Expected σ_HU (water) |
|----------|-----|-----|------------------------|
| Standard body | 120 | 200 | 10-15 HU |
| Low dose chest | 100 | 50 | 25-35 HU |
| Head | 120 | 300 | 3-5 HU |
| Ultra-low dose | 80 | 10 | 50-80 HU |
