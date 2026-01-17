# Virtual Monoenergetic Imaging (VMI) Algorithm Research for BasisSimulator.jl

> **Document Purpose:** Research document for implementing Virtual Monoenergetic Imaging in the dual-energy CT reconstruction workflow of BasisSimulator.jl.
>
> **Story ID:** RESEARCH-VMI
>
> **Status:** COMPLETE
>
> **Last Updated:** 2026-01-16

---

## Executive Summary

This document researches Virtual Monoenergetic Imaging (VMI) algorithms for implementation in BasisSimulator.jl. VMI synthesizes CT images at arbitrary monochromatic energies from dual-energy data, providing improved contrast, reduced beam hardening artifacts, and material-specific visualization.

### Key Findings

1. **Two Main Algorithm Classes:** Projection-domain and image-domain VMI generation
2. **BasisSimulator Approach:** Projection-domain method recommended (matches GE GSI architecture)
3. **Optimal Energy Ranges:** 40-70 keV for iodine enhancement, 80-140 keV for artifact reduction
4. **Noise Characteristics:** Anti-correlated noise amplification at low keV is inherent to VMI physics
5. **Validation Strategy:** Water HU = 0, iodine/calcium monotonic with keV energy

---

## 1. VMI Algorithm Fundamentals

### 1.1 Physical Basis

Virtual Monoenergetic Images emulate what a CT acquisition would produce with a true monochromatic X-ray source (e.g., synchrotron). The fundamental equation is:

```
μ_VMI(E) = ρ₁ × μ₁(E) + ρ₂ × μ₂(E)
```

Where:
- `ρ₁, ρ₂` are the mass densities of basis materials (from decomposition)
- `μ₁(E), μ₂(E)` are mass attenuation coefficients at energy E

For water-iodine basis pair:
```
μ_VMI(E) = ρ_water × μ_water(E) + ρ_iodine × μ_iodine(E)
```

CITE: AAPM TG-291 (McCollough et al., Medical Physics 2020) - https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14157

### 1.2 Two Algorithm Classes

VMI generation can be performed in either domain, depending on the DECT system architecture:

| Method | Domain | Systems | BHC Correction | Registration |
|--------|--------|---------|----------------|--------------|
| Projection-domain | Raw sinogram | GE GSI, Dual-layer | Intrinsic | Excellent |
| Image-domain | Reconstructed images | Dual-source | Requires pre-correction | Limited |

CITE: AJR "Dual-Energy CT-Based Monochromatic Imaging" - https://ajronline.org/doi/10.2214/AJR.12.9121

---

## 2. Projection-Domain VMI (Recommended for BasisSimulator)

### 2.1 Algorithm Overview

For rapid kVp switching systems (GE GSI), VMI is generated in the projection domain:

**Step 1: Material Decomposition (already implemented)**
```
Given: p_L (80 kVp sinogram), p_H (140 kVp sinogram)
Solve: A₁ = ∫ρ₁ dl, A₂ = ∫ρ₂ dl (line integrals of basis material densities)
```

**Step 2: Virtual Monoenergetic Projection Synthesis**
```
p_VMI(E) = A₁ × μ₁(E) + A₂ × μ₂(E)
```

**Step 3: Standard Reconstruction**
```
image_VMI = FDK(p_VMI)
```

**Step 4: HU Conversion**
```
HU_VMI = 1000 × (μ_VMI - μ_water(E)) / μ_water(E)
```

CITE: GE Gemstone Spectral Imaging documentation, Springer Nature - https://link.springer.com/chapter/10.1007/174_2010_35

### 2.2 Advantages of Projection-Domain Method

1. **Intrinsic Beam Hardening Correction:** Virtual monochromatic CT images synthesized in the projection domain can fully eliminate beam-hardening artifacts [1]
2. **More Accurate Iodine Detection:** Projection-based method delivers more accurate detection of iodine [2]
3. **No Pre-Correction Required:** Raw data-based decomposition is exact [3]
4. **Temporal Registration:** Excellent for rapid kVp switching systems [4]

CITE:
1. PMC6792008 - https://pmc.ncbi.nlm.nih.gov/articles/PMC6792008/
2. Radiology Dual-Energy comparison study
3. PMC9297826 - https://pmc.ncbi.nlm.nih.gov/articles/PMC9297826/
4. AAPM TG-291

### 2.3 Workflow in BasisSimulator

```julia
# EXISTING: Dual-energy forward projection
de_sino = forward_project_dual_energy(phantom.mask, geom, protocol; ...)

# EXISTING: Material decomposition
mat_map = decompose_materials(de_sino; basis=(:water, :iodine))

# EXISTING: VMI synthesis (projection domain)
vmi_sino = virtual_monoenergetic(mat_map, 70.0)  # Returns sinogram

# NEW: Reconstruct VMI sinogram
vmi_image = fdk_reconstruct(vmi_sino, geom, recon_size)

# NEW: Convert to HU
μ_water_70 = get_water_attenuation(70.0)  # ~0.19 cm⁻¹
vmi_hu = 1000f0 * (vmi_image .- μ_water_70) / μ_water_70
```

---

## 3. Image-Domain VMI (Alternative)

### 3.1 Algorithm Overview

For dual-source systems (Siemens), VMI is generated from reconstructed images:

**Step 1: Independent Reconstruction**
```
image_L = FDK(p_L)  # 80 kVp reconstruction
image_H = FDK(p_H)  # 140 kVp reconstruction
```

**Step 2: Image-Domain Material Decomposition**
```
Per voxel: [μ₁_img; μ₂_img] = A⁻¹ × [image_L; image_H]
```

**Step 3: VMI Synthesis**
```
image_VMI(E) = μ₁_img × μ₁(E)/μ₁_ref + μ₂_img × μ₂(E)/μ₂_ref
```

CITE: Siemens Mono+ Algorithm, Syngo.via documentation

### 3.2 Limitations

1. **Beam Hardening Artifacts:** Requires ad hoc pre-correction [1]
2. **Spatial Registration:** Can be problematic for dual-source helical [2]
3. **Quantitative Accuracy:** Less accurate material density maps [3]

CITE:
1. PMC4409135 - https://pmc.ncbi.nlm.nih.gov/articles/PMC4409135/
2. AJR 12.9121
3. PMC6792008

### 3.3 When to Use Image-Domain

Image-domain VMI is appropriate when:
- Projection data registration is imperfect (dual-source helical)
- Only reconstructed images are available
- Real-time interactive VMI keV selection is needed

**NOT recommended for BasisSimulator** because:
- GE GSI architecture uses projection-domain decomposition
- We have full access to projection data
- Projection-domain provides better beam hardening correction

---

## 4. Optimal Energy Selection Guide

### 4.1 Clinical Energy Recommendations

| Energy (keV) | Relative Noise | Iodine Enhancement | Clinical Use |
|--------------|----------------|-------------------|--------------|
| 40 | 3.0× baseline | Maximum | Salvage low-contrast studies |
| 50 | 2.0× baseline | Very high | Liver lesion detection |
| 55-60 | 1.5× baseline | High | Vascular imaging |
| 65-70 | 1.0× (baseline) | Moderate | Standard imaging (≈120 kVp equivalent) |
| 80 | 0.9× baseline | Low | Balanced quality |
| 100-120 | 0.8× baseline | Minimal | Beam hardening reduction |
| 140 | 0.7× baseline | Minimal | Metal artifact reduction |

CITE: PMC6592074 "Dual energy computed tomography virtual monoenergetic imaging" - https://pmc.ncbi.nlm.nih.gov/articles/PMC6592074/

### 4.2 Iodine K-Edge Effect

Iodine has K-edge at **33.2 keV**, which causes:
- **Below K-edge (< 33 keV):** Lower photoelectric absorption
- **Above K-edge (33-70 keV):** Maximum photoelectric enhancement
- **High energy (> 100 keV):** Compton scatter dominates, low contrast

For iodine-enhanced imaging, **40-60 keV** provides maximum contrast because:
1. Energy is above K-edge (maximum absorption jump)
2. Energy is still low enough for strong photoelectric effect

CITE: NIST XCOM database, Radiology Key dual-energy chapter

### 4.3 Calcium/Bone Behavior

Calcium K-edge is at **4.0 keV** (below diagnostic range), so:
- Calcium HU decreases monotonically with increasing keV
- No K-edge enhancement effect in clinical energy range
- At 140 keV, calcium contrast is reduced (useful for virtual non-contrast)

---

## 5. Noise Characteristics

### 5.1 Anti-Correlated Noise in Material Decomposition

During material decomposition, anti-correlated noise is introduced:

```
Cov(ρ₁, ρ₂) < 0
```

This means:
- When noise causes overestimation of ρ₁, it causes underestimation of ρ₂
- VMI noise = f(σ₁², σ₂², Cov(ρ₁,ρ₂), μ₁(E), μ₂(E))

CITE: PMC9860003 "Noise correlation and its impact on multi-material decomposition" - https://pmc.ncbi.nlm.nih.gov/articles/PMC9860003/

### 5.2 Energy-Dependent Noise Behavior

The VMI noise variance follows:

```
σ²_VMI(E) = (∂p_VMI/∂A₁)² × σ²_A1 + (∂p_VMI/∂A₂)² × σ²_A2 + 2×(∂p_VMI/∂A₁)×(∂p_VMI/∂A₂)×Cov(A₁,A₂)
         = μ₁(E)² × σ²_A1 + μ₂(E)² × σ²_A2 + 2 × μ₁(E) × μ₂(E) × Cov(A₁,A₂)
```

**Key insight:** At low keV, μ_iodine(E) is large, amplifying the iodine noise contribution.

Measured values from phantom studies:
- **40 keV:** Noise = 5.3 ± 0.2 HU (highest)
- **70 keV:** Noise = 3.6 ± 0.2 HU (minimum)
- **200 keV:** Noise = 3.5 ± 0.2 HU (constant)

CITE: PMC8666791 - https://pmc.ncbi.nlm.nih.gov/articles/PMC8666791/

### 5.3 Noise-Optimized VMI+ Algorithm

Siemens developed a "frequency-split" noise-optimized algorithm (VMI+ or Monoenergetic+):

```
VMI+(E) = LowFreq(VMI at optimal_keV) + HighFreq(VMI at target_keV)
```

This preserves high-contrast information from low-keV VMI while reducing noise using optimal-keV background.

**NOT RECOMMENDED** for BasisSimulator v1:
- Adds significant complexity
- Requires tuning frequency thresholds
- Standard VMI is sufficient for research/simulation purposes

---

## 6. Validation Strategy

### 6.1 Water HU Validation

Water should measure **0 HU at ALL VMI energies**:

```
HU_water = 1000 × (μ_water(E) - μ_water(E)) / μ_water(E) = 0
```

**Acceptance Criterion:** |HU_water| < 5 HU at 40, 70, 100, 140 keV

CITE: PMC5903965 QC program development - https://pmc.ncbi.nlm.nih.gov/articles/PMC5903965/

### 6.2 Iodine Enhancement Validation

Iodine HU should INCREASE at LOW keV:

| Energy (keV) | Expected Iodine Factor (vs 120 kVp) |
|--------------|-------------------------------------|
| 40 | ~3.0× |
| 50 | ~1.9× |
| 60 | ~1.3× |
| 70 | ~1.0× (reference) |
| 100 | ~0.6× |
| 140 | ~0.4× |

**Acceptance Criterion:** HU_iodine(40 keV) > HU_iodine(70 keV) > HU_iodine(140 keV)

CITE: PMC8219825 - https://pmc.ncbi.nlm.nih.gov/articles/PMC8219825/

### 6.3 Calcium HU Validation

Calcium HU should DECREASE at HIGH keV (monotonically):

**Acceptance Criterion:** HU_calcium(40 keV) > HU_calcium(70 keV) > HU_calcium(140 keV)

### 6.4 Phantom-Based Testing

Recommended phantom configuration:
- **Water insert:** Verify HU = 0 at all energies
- **Iodine inserts (2-20 mg/mL):** Verify K-edge enhancement
- **Calcium inserts (50-600 mg/cc):** Verify monotonic decrease

Use existing Gammex 472 phantom with:
- Calcium series: 50, 100, 200, 300, 400, 500, 600 mg/cc
- Iodine series: 2, 2.5, 5, 7.5, 10, 15, 20 mg/mL

---

## 7. Proposed API Design

### 7.1 Enhanced virtual_monoenergetic() Function

The existing function returns a projection-domain sinogram. We need to add:

1. **HU conversion support**
2. **Integrated reconstruction option**
3. **Energy-specific water attenuation**

```julia
"""
    virtual_monoenergetic(materials::MaterialMap, energy_keV::Float64;
                          output::Symbol=:projection,
                          geom::Union{Nothing,CTGeometry}=nothing,
                          recon_size::Union{Nothing,NTuple{3,Int}}=nothing) -> Array

Generate virtual monoenergetic image at specified energy.

# Arguments
- `materials::MaterialMap`: Result of material decomposition
- `energy_keV::Float64`: Target energy in keV (40-140)

# Keyword Arguments
- `output::Symbol=:projection`: Output type (:projection, :image, :hu)
- `geom::CTGeometry`: Required if output is :image or :hu
- `recon_size::NTuple{3,Int}`: Required if output is :image or :hu

# Returns
- If output=:projection: Virtual monoenergetic sinogram
- If output=:image: Reconstructed image (μ values)
- If output=:hu: Reconstructed image in HU

# Example
```julia
# Just sinogram (existing behavior)
vmi_sino = virtual_monoenergetic(mat_map, 70.0)

# Full reconstruction to HU
vmi_hu = virtual_monoenergetic(mat_map, 70.0;
    output=:hu,
    geom=geom,
    recon_size=(256, 256, 32)
)
```
"""
function virtual_monoenergetic(materials::MaterialMap{T}, energy_keV::Float64;
                                output::Symbol=:projection,
                                geom=nothing,
                                recon_size=nothing) where T
    # ...
end
```

### 7.2 Reconstruction Integration

```julia
"""
    fdk_reconstruct(sino, geom, recon_size; vmi_keV=nothing)

FDK reconstruction with optional VMI energy specification.

If `vmi_keV` is provided and `sino` is a MaterialMap, automatically
generates VMI at the specified energy before reconstruction.
"""
function fdk_reconstruct(materials::MaterialMap, geom::CTGeometry, recon_size::NTuple{3,Int};
                         vmi_keV::Float64)
    # 1. Generate VMI sinogram
    vmi_sino = virtual_monoenergetic(materials, vmi_keV)

    # 2. Reconstruct
    vmi_image = fdk_reconstruct(vmi_sino, geom, recon_size)

    # 3. Convert to HU
    μ_water = get_water_attenuation(vmi_keV)
    vmi_hu = 1000f0 * (vmi_image .- μ_water) / μ_water

    return vmi_hu
end
```

### 7.3 Batch VMI Generation

```julia
"""
    generate_vmi_series(materials::MaterialMap, energies_keV::Vector{Float64},
                        geom::CTGeometry, recon_size::NTuple{3,Int})

Generate VMI images at multiple energies for keV sweep visualization.

# Returns
Dict{Float64, Array{Float32,3}} mapping energy to HU image.
"""
function generate_vmi_series(materials, energies_keV, geom, recon_size)
    return Dict(E => virtual_monoenergetic(materials, E; output=:hu, geom=geom, recon_size=recon_size)
                for E in energies_keV)
end
```

---

## 8. Implementation Plan

### Phase 1: Core VMI Enhancements

1. **Add energy-specific water attenuation lookup**
   - `get_water_attenuation(energy_keV)` using XrayAttenuation.jl
   - Validate against NIST data

2. **Enhance virtual_monoenergetic() API**
   - Add `output` kwarg for :projection/:image/:hu
   - Add `geom` and `recon_size` kwargs for integrated reconstruction
   - Maintain backward compatibility (default output=:projection)

3. **Add HU conversion utilities**
   - `vmi_to_hu(vmi_image, energy_keV)`
   - Handle energy-specific μ_water normalization

### Phase 2: Reconstruction Integration

1. **Add MaterialMap dispatch to fdk_reconstruct()**
   - Accept MaterialMap + vmi_keV kwarg
   - Internally generate VMI sinogram and reconstruct

2. **Add batch VMI generation**
   - `generate_vmi_series()` for keV sweep
   - Efficient multi-energy reconstruction

### Phase 3: Validation

1. **Water HU stability test**
   - Verify HU = 0 ± 5 at 40, 50, 60, 70, 80, 100, 120, 140 keV

2. **Iodine enhancement test**
   - Verify HU increases at low keV
   - Measure enhancement factor vs 70 keV baseline

3. **Calcium monotonicity test**
   - Verify HU decreases at high keV

4. **Noise characterization**
   - Measure σ vs keV
   - Verify noise minimum at ~70 keV

### Phase 4: Example and Documentation

1. **Create VMI example**
   - `examples/vmi_keV_sweep.jl`
   - Show Gammex phantom at 40, 70, 100, 140 keV
   - Visualize iodine/calcium behavior

2. **Update CLAUDE.md**
   - Add VMI workflow documentation
   - Energy selection guide

---

## 9. References

1. McCollough CH, et al. "Principles and applications of multienergy CT: Report of AAPM Task Group 291." Medical Physics. 2020.
   https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14157

2. Yu L, et al. "Dual-Energy CT–Based Monochromatic Imaging." AJR. 2012.
   https://ajronline.org/doi/10.2214/AJR.12.9121

3. "Gemstone Detector: Dual Energy Imaging via Fast kVp Switching." Springer. 2010.
   https://link.springer.com/chapter/10.1007/174_2010_35

4. Tao S, et al. "Dual energy computed tomography virtual monoenergetic imaging." Ann Transl Med. 2020.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6592074/

5. Cester D, et al. "Virtual monoenergetic images from dual-energy CT: systematic assessment of task-based image quality." QIMS. 2021.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8666791/

6. Nute JL, et al. "Development of a dual-energy computed tomography quality control program." Medical Physics. 2018.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5903965/

7. Euler A, et al. "Quantitative benchmarking of iodine imaging for two CT spectral imaging technologies." Eur Radiol Exp. 2021.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8219825/

8. Wang X, et al. "Noise correlation and its impact on multi-material decomposition in photon-counting CT." Medical Physics. 2023.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9860003/

9. Zhou W, et al. "Image-domain multimaterial decomposition for dual-energy CT." Physics in Medicine & Biology. 2019.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6792008/

10. Keeler A, et al. "Fast, automated optimization of virtual monoenergetic images with the dual-energy image synthesizer for cone-beam CT." J Appl Clin Med Phys. 2025.
    https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.70083

---

## Appendix A: Material Attenuation Coefficients

Reference values for water at various energies (used for HU conversion):

| Energy (keV) | μ_water (cm⁻¹) | Source |
|--------------|----------------|--------|
| 40 | 0.268 | NIST XCOM |
| 50 | 0.227 | NIST XCOM |
| 60 | 0.206 | NIST XCOM |
| 70 | 0.193 | NIST XCOM |
| 80 | 0.184 | NIST XCOM |
| 100 | 0.171 | NIST XCOM |
| 120 | 0.163 | NIST XCOM |
| 140 | 0.157 | NIST XCOM |

---

## Appendix B: Iodine Enhancement Factors

Expected iodine HU multiplication factors relative to 70 keV baseline:

| Energy (keV) | Factor | Notes |
|--------------|--------|-------|
| 40 | 2.8-3.2 | Above K-edge, maximum enhancement |
| 50 | 1.8-2.0 | Still strong enhancement |
| 60 | 1.2-1.4 | Moderate enhancement |
| 70 | 1.0 | Reference baseline |
| 80 | 0.8-0.9 | Slight reduction |
| 100 | 0.5-0.6 | Significant reduction |
| 140 | 0.3-0.4 | Minimal iodine contrast |

---

## Appendix C: Noise vs Energy Model

Simplified noise model for VMI:

```
σ_VMI(E) ≈ σ_70 × sqrt(1 + α × (70/E - 1)²)
```

Where:
- `σ_70` is the noise at 70 keV (baseline)
- `α` is a scanner-dependent constant (~0.5-1.0)

This gives approximately:
- σ(40 keV) ≈ 1.5-2.0 × σ(70 keV)
- σ(100 keV) ≈ 0.9 × σ(70 keV)
- σ(140 keV) ≈ 0.85 × σ(70 keV)
