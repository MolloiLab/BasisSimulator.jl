# Dual kVp CT Implementation Research for BasisSimulator.jl

> **Document Purpose:** Research document for implementing dual-energy (dual kVp) CT simulation in BasisSimulator.jl, with focus on GE Revolution Apex GSI technology.
>
> **Story ID:** RESEARCH-DUAL-KVP
>
> **Status:** COMPLETE
>
> **Last Updated:** 2026-01-16

---

## Executive Summary

This document researches the implementation of dual-energy CT (DECT) simulation for BasisSimulator.jl. The focus is on GE's Gemstone Spectral Imaging (GSI) technology using rapid kVp switching, as implemented in the GE Revolution Apex scanner.

### Key Findings

1. **GE GSI Technology:** Uses rapid kVp switching (80/140 kVp alternating every 0.2 ms)
2. **Angular Interleaving:** Low/high kVp projections are interleaved view-by-view (<0.18° angular offset)
3. **Material Decomposition:** Performed in projection domain using polynomial expansion
4. **Virtual Monoenergetic Imaging (VMI):** Generates images at 40-140 keV from basis material maps
5. **Unified API:** Single scanner specification with mode parameter for single/dual kVp acquisition

---

## 1. Dual-Energy CT Architecture Types

### 1.1 Overview of Available Technologies

There are six main approaches to dual-energy CT imaging [1]:

| Type | Manufacturer | Mechanism | Temporal Registration |
|------|--------------|-----------|----------------------|
| Rapid kVp Switching | GE Healthcare | Fast tube voltage alternation | Excellent (<0.5 ms) |
| Dual Source | Siemens | Two X-ray tube/detector pairs | Limited (angular offset) |
| Dual Layer (Sandwich) | Philips | Spectral detector | Perfect (simultaneous) |
| Sequential | Canon/Toshiba | Two sequential scans | Poor (motion artifacts) |
| Twin Beam | Siemens | Split beam with filters | Good (simultaneous) |
| Helical DECT | Siemens | Helical acquisition | Variable |

### 1.2 GE GSI: Rapid kVp Switching

GE's Gemstone Spectral Imaging (GSI) platform was introduced in 2010 on the Discovery CT750 HD and has evolved through subsequent generations [2]:

- **Discovery CT750 HD** (2010): First GSI platform
- **Revolution CT Frontier**: Improved spectral capabilities
- **Revolution CT/Revolution Apex**: GSI-Xtream with better energy separation

#### Technical Advantages

- Excellent temporal registration (projections <0.5 ms apart)
- Full 50 cm scan field of view for material decomposition
- Projection-domain material decomposition (more accurate)
- Single source design (no cross-scatter between sources)

#### Technical Challenges

- Incomplete energy separation due to voltage rise/fall times
- Fixed tube current during GSI acquisition (no mA modulation)
- Increased noise at low keV VMI

---

## 2. GE Revolution Apex GSI Technical Specifications

### 2.1 kVp Switching Parameters

Based on FDA documentation and published research [2,3,4]:

| Parameter | Value | Source |
|-----------|-------|--------|
| Low kVp | 80 kVp | GE Documentation |
| High kVp | 140 kVp | GE Documentation |
| Switching time | 0.2 ms | PMC6909110 |
| Views per rotation | 2496 max | FDA K213715 |
| Angular offset (adjacent views) | <0.18° | Literature |
| Integration time (low kVp) | ~65% of view | Radiologykey |
| Integration time (high kVp) | ~35% of view | Radiologykey |

### 2.2 Acquisition Modes

The GE Revolution Apex supports multiple GSI preset families [3]:

- **Gantry rotation time:** 0.5 - 1.0 seconds
- **Helical pitch:** 0.508 - 1.531
- **Detector collimation:** 40 mm or 80 mm
- **Focal spot:** Large or Extra Large
- **Bowtie filter:** Body or Head

**Important constraint:** During GSI acquisition, tube current (mA) is fixed for each kVp level and cannot be modulated longitudinally or angularly.

### 2.3 Data Structure

The acquired projection data forms an **interleaved sinogram**:

```
View 1:  80 kVp
View 2:  140 kVp
View 3:  80 kVp
View 4:  140 kVp
...
```

Before material decomposition, the interleaved data is sorted into two separate sinograms:

```
Low-energy sinogram:   Views 1, 3, 5, 7, ...  (80 kVp)
High-energy sinogram:  Views 2, 4, 6, 8, ...  (140 kVp)
```

---

## 3. Physics of Dual-Energy CT

### 3.1 Basis Material Decomposition

The fundamental principle of DECT is that X-ray attenuation can be decomposed into contributions from two basis materials [1,5]:

```
μ(E) = ρ₁ × μ₁(E) + ρ₂ × μ₂(E)
```

Where:
- `μ(E)` = total linear attenuation at energy E
- `ρ₁, ρ₂` = mass densities of basis materials
- `μ₁(E), μ₂(E)` = mass attenuation coefficients of basis materials

Common basis material pairs:
- **Water + Iodine** (most common clinical)
- **Water + Calcium** (bone imaging)
- **Photoelectric + Compton** (physics-based)

### 3.2 Projection-Domain Material Decomposition

For rapid kVp switching systems, material decomposition is performed in the projection domain [4,6]:

Given low and high energy projections `p_L` and `p_H`:

```
p_L = ∫ [ρ₁(x,y) × μ₁(E_L) + ρ₂(x,y) × μ₂(E_L)] dl
p_H = ∫ [ρ₁(x,y) × μ₁(E_H) + ρ₂(x,y) × μ₂(E_H)] dl
```

These form a system of two equations that can be solved for the line integrals of `ρ₁` and `ρ₂`.

In practice, a **polynomial expansion** with calibration lookup table is used:

```
A₁ = f₁(p_L, p_H) = Σᵢⱼ aᵢⱼ × p_L^i × p_H^j
A₂ = f₂(p_L, p_H) = Σᵢⱼ bᵢⱼ × p_L^i × p_H^j
```

Where `A₁` and `A₂` are the line integrals of basis material densities, and coefficients `aᵢⱼ, bᵢⱼ` are determined from calibration measurements.

### 3.3 Virtual Monoenergetic Imaging (VMI)

After material decomposition, Virtual Monoenergetic Images at any energy E can be synthesized [1,5]:

```
μ_VMI(E) = ρ₁ × μ₁(E) + ρ₂ × μ₂(E)
```

For water-iodine basis:
```
μ_VMI(E) = ρ_water × μ_water(E) + ρ_iodine × μ_iodine(E)
```

Clinical VMI energy ranges:
- **40-70 keV:** Enhanced iodine contrast (high noise)
- **70-80 keV:** Standard imaging (balanced)
- **80-140 keV:** Artifact reduction, beam hardening correction

---

## 4. Proposed API Design

### 4.1 Unified Scanner API

The GE Revolution Apex scanner specification should support both single and dual kVp modes through a unified API:

```julia
# Get scanner specification
spec = GERevolutionApex()

# Single kVp mode (existing)
geom = create_geometry(spec; n_angles=984, n_rows=64)
sinogram = forward_project(phantom.μ, geom; physics=physics)

# Dual kVp mode (new)
geom_gsi = create_geometry(spec;
    n_angles = 984,
    n_rows = 64,
    mode = :dual_kvp  # NEW: Enable GSI mode
)

# Forward project returns DualEnergySinogram
de_sinogram = forward_project(phantom.mask, geom_gsi;
    physics = physics,
    energies_low = energies_80kVp,
    weights_low = weights_80kVp,
    energies_high = energies_140kVp,
    weights_high = weights_140kVp,
    materials = materials
)
```

### 4.2 Data Types

```julia
"""
    DualEnergySinogram

Container for dual-energy sinogram data with interleaved low/high kVp projections.

# Fields
- `low::Array{T,3}`: Low-energy sinogram (80 kVp)
- `high::Array{T,3}`: High-energy sinogram (140 kVp)
- `geometry::CTGeometry`: Acquisition geometry
- `interleave_pattern::Symbol`: :view_by_view or :block
"""
struct DualEnergySinogram{T<:AbstractFloat}
    low::Array{T,3}
    high::Array{T,3}
    geometry::CTGeometry
    interleave_pattern::Symbol
end

"""
    MaterialMap

Result of material decomposition.

# Fields
- `material1::Array{T,3}`: Density map of first basis material (e.g., water)
- `material2::Array{T,3}`: Density map of second basis material (e.g., iodine)
- `material1_name::Symbol`: Name of first material
- `material2_name::Symbol`: Name of second material
"""
struct MaterialMap{T<:AbstractFloat}
    material1::Array{T,3}
    material2::Array{T,3}
    material1_name::Symbol
    material2_name::Symbol
end
```

### 4.3 Material Decomposition API

```julia
"""
    decompose_materials(sino::DualEnergySinogram;
                        basis=(:water, :iodine),
                        method=:polynomial) -> MaterialMap

Perform projection-domain material decomposition.

# Arguments
- `sino::DualEnergySinogram`: Dual-energy sinogram data
- `basis`: Tuple of basis materials (:water, :iodine), (:water, :calcium), etc.
- `method`: Decomposition method (:polynomial, :direct)

# Returns
MaterialMap with density projections for each basis material.
"""
function decompose_materials(sino::DualEnergySinogram;
                             basis::Tuple{Symbol,Symbol}=(:water, :iodine),
                             method::Symbol=:polynomial)
    # ... implementation
end

"""
    virtual_monoenergetic(materials::MaterialMap, energy_keV::Float64) -> Array

Generate virtual monoenergetic image at specified energy.

# Arguments
- `materials::MaterialMap`: Result of material decomposition
- `energy_keV::Float64`: Target energy in keV (40-140)

# Returns
Virtual monoenergetic image at the specified energy.
"""
function virtual_monoenergetic(materials::MaterialMap, energy_keV::Float64)
    # ... implementation
end
```

### 4.4 High-Level Workflow

```julia
using BasisSimulator

# 1. Setup
spec = GERevolutionApex()
phantom = create_gammex_472(n_voxels=256, n_slices=32, fov_cm=35.0)
materials = get_region_materials()

# 2. Load spectra for both energies
e_low, w_low = load_spectrum(80)
e_high, w_high = load_spectrum(140)

# 3. Create GSI geometry
geom = create_geometry(spec; n_angles=984, n_rows=64, mode=:dual_kvp)

# 4. Forward project (GPU)
de_sino = forward_project(phantom.mask, geom;
    energies_low = e_low, weights_low = w_low,
    energies_high = e_high, weights_high = w_high,
    materials = materials,
    physics = gsi_physics_config()
)

# 5. Material decomposition
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))

# 6. Generate VMI at different energies
vmi_50keV = virtual_monoenergetic(mat_map, 50.0)
vmi_70keV = virtual_monoenergetic(mat_map, 70.0)
vmi_100keV = virtual_monoenergetic(mat_map, 100.0)

# 7. Reconstruct
recon_50keV = fdk_reconstruct(vmi_50keV, geom)
```

---

## 5. Implementation Plan

### Phase 1: Core Dual-Energy Data Structures

1. Add `DualEnergySinogram` type
2. Add `MaterialMap` type
3. Extend `CTGeometry` with `mode` field
4. Update `create_geometry()` for dual kVp mode

### Phase 2: Dual-Energy Forward Projection

1. Implement `forward_project()` dispatch for dual kVp mode
2. Generate interleaved low/high energy sinograms
3. Handle angular offset between views (~0.18°)
4. Apply appropriate physics for each kVp level

### Phase 3: Material Decomposition

1. Implement polynomial-based material decomposition
2. Add calibration data for water-iodine, water-calcium pairs
3. Handle noise propagation in decomposition
4. GPU-accelerate material decomposition

### Phase 4: Virtual Monoenergetic Imaging

1. Implement VMI synthesis from material maps
2. Add energy-dependent μ interpolation (NIST data)
3. Validate VMI against expected HU values
4. Support clinical keV range (40-140)

### Phase 5: Validation and Examples

1. Create dual-energy example with Gammex phantom
2. Validate iodine quantification
3. Compare VMI at different energies
4. Document noise characteristics

---

## 6. Considerations for Physics Pipeline

### 6.1 Bowtie Filter

Different attenuation for 80 and 140 kVp spectra:
- Low kVp: More attenuation, more beam hardening correction
- High kVp: Less attenuation

### 6.2 Scatter

Scatter characteristics differ between energies:
- Low kVp: Higher scatter fraction
- High kVp: Lower scatter fraction

### 6.3 Noise Model

Fixed mA for each kVp during GSI:
- Low kVp views: Longer integration time (~65%)
- High kVp views: Shorter integration time (~35%)

Need to use `mA_to_I0()` separately for each kVp level.

### 6.4 Detector Response

Same detector, but energy-dependent efficiency:
- Lower efficiency at low keV
- DQE varies with energy

---

## 7. References

1. Dual energy computed tomography virtual monoenergetic imaging: technique and clinical applications. PMC6592074
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6592074/

2. Block Imaging - What Is Gemstone Spectral Imaging?
   https://www.blockimaging.com/blog/what-is-gemstone-spectral-imaging

3. A suggested method for setting up GSI profiles on the GE Revolution CT scanner. PMC6909110
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6909110/

4. Rapid kV Switching Dual-Energy CT Imaging. Radiology Key
   https://radiologykey.com/rapid-kv-switching-dual-energy-ct-imaging/

5. Dual-Energy CT–Based Monochromatic Imaging. AJR
   https://ajronline.org/doi/10.2214/AJR.12.9121

6. Analysis of fast kV-switching in dual energy CT. ResearchGate
   https://www.researchgate.net/publication/252513571_Analysis_of_fast_kV-switching_in_dual_energy_CT_using_a_pre-reconstruction_decomposition_technique

7. Principles and applications of multienergy CT: Report of AAPM Task Group 291. Medical Physics
   https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14157

8. Comparison of image quality between two rapid kVp-switching dual-energy CT scanners. PMC10613589
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10613589/

9. FDA 510(k) K213715 - GE Revolution Apex Elite CT
   https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf

---

## Appendix A: GE GSI Preset Families

Common GSI acquisition presets available on GE Revolution CT:

| Preset | Rotation | Pitch | Collimation | Application |
|--------|----------|-------|-------------|-------------|
| GSI-01 | 0.5s | 0.508 | 80mm | Abdomen |
| GSI-02 | 0.5s | 0.992 | 40mm | Chest |
| GSI-03 | 0.7s | 0.508 | 80mm | Abdomen (lower dose) |
| GSI-04 | 1.0s | 0.508 | 80mm | Abdomen (quality) |

---

## Appendix B: Material Attenuation at Dual-Energy kVp

Reference attenuation values at effective energies:

| Material | μ at ~50 keV (80kVp) | μ at ~75 keV (140kVp) | K-edge |
|----------|---------------------|----------------------|--------|
| Water | 0.22 cm⁻¹ | 0.19 cm⁻¹ | - |
| Iodine (5 mg/mL) | 0.35 cm⁻¹ | 0.21 cm⁻¹ | 33.2 keV |
| Calcium | 0.85 cm⁻¹ | 0.45 cm⁻¹ | 4.0 keV |
| Bone | 0.55 cm⁻¹ | 0.35 cm⁻¹ | - |

Note: Iodine shows maximum contrast difference due to K-edge at 33.2 keV, which is above 80 kVp effective energy but below 140 kVp effective energy.

---

## Appendix C: VMI Noise Characteristics

Expected noise behavior in VMI:

| Energy (keV) | Relative Noise | Iodine CNR | Use Case |
|--------------|---------------|------------|----------|
| 40 | 3.0x | High | Maximum iodine boost |
| 50 | 2.0x | High | Liver lesions |
| 60 | 1.5x | Medium | Vascular imaging |
| 70 | 1.0x (baseline) | Medium | Standard imaging |
| 80 | 0.9x | Low | Balanced |
| 100 | 0.8x | Low | Artifact reduction |
| 140 | 0.7x | Minimal | Metal artifact |

The 70 keV VMI is designed to match the noise level of a dose-matched single-energy acquisition.
