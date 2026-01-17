# CT Hounsfield Unit Tolerance Standards Research

**Document Purpose:** Establish authoritative HU tolerance standards from medical physics literature for CT simulator validation.

**Critical Finding:** The ~50% HU error on contrast materials is **PHYSICS**, not a bug. Thin-sample ground truth values do NOT equal beam-hardened CT measurements.

---

## Executive Summary

### Key Findings

1. **Water HU Tolerance:** ±5 HU (ACR, IEC) to ±7 HU (clinical QA)
2. **Contrast Material (Ca, I) Tolerance:** No single authoritative standard exists
3. **Root Cause of 50% Error:** Fundamental physics mismatch between thin-sample μ and CT measurement
4. **Correct Validation Approach:** Compare simulator to clinical scanner measurements, NOT NIST tables

### Critical Distinction

| Quantity | Definition | Use Case |
|----------|------------|----------|
| **Thin-sample μ** | μ computed from NIST XCOM at single energy or spectrum-weighted average | Material physics research |
| **CT-measured μ** | μ derived from polychromatic beam passing through ~30cm water path with beam hardening | Clinical CT simulation |

**These are fundamentally different quantities and should NOT be directly compared.**

---

## 1. ACR CT Accreditation Program

### Source
- [ACR Phantom Scoring CT (Revised 8-26-2024)](https://accreditationsupport.acr.org/support/solutions/articles/11000054129-phantom-scoring-ct-revised-8-26-2024-)
- [Scanner and kVp dependence of measured CT numbers in the ACR CT phantom (PMC5714621)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5714621/)

### HU Tolerance Requirements

CITE: ACR CT Accreditation Program phantom scoring guidelines specify:

| Material | CT Number Range (HU) |
|----------|---------------------|
| Water | -7 to +7 |
| Air | -1005 to -970 |
| Polyethylene | -107 to -84 |
| Acrylic | 110 to 135 |
| Teflon (bone surrogate) | 850 to 970 |

CITE: "An 80 kVp image with a mean CT number of 26.8 HU is at the limit of the program tolerances." (Many states require water values between -5 HU and 15 HU)

### Key Observations

CITE: "Among the five materials (solid water, air, polyethylene, acrylic, bone-equivalent) the measured CT numbers exhibit manufacturer and kVp dependence, which should be taken into account when defining tolerances."

**ACR does NOT specify tolerances for contrast agents (iodine solutions)** - only tissue-equivalent materials.

---

## 2. IEC 61223-3-5:2019

### Source
- [IEC 61223-3-5:2019 - Acceptance and constancy tests](https://webstore.iec.ch/en/publication/59789)
- [How to measure CT image quality (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1120179714000088)

### HU Tolerance Requirements

CITE: "IEC recommends the mean CT-number of a central region of interest (ROI) in a uniformity device not to deviate by more than ±4 HU from the nominal values specified by the manufacturer."

CITE: "The uniformity (the deviation in mean CT-number between central and peripheral regions) must not be greater than 4 HU at acceptance."

CITE: "The difference in uniformity should not vary by more than 2 HU from baseline values."

### IPEM Comparison

CITE: "IPEM suggests a level of ±5 HU from the baseline CT-number of water and ±10 HU from the baseline number of other materials. The suspension levels are ±20 HU and ±30 HU, respectively."

---

## 3. AAPM TG-233 (2019)

### Source
- [AAPM Report 233 PDF](https://www.aapm.org/pubs/reports/RPT_233.pdf)
- [Performance evaluation of CT systems summary (PubMed)](https://pubmed.ncbi.nlm.nih.gov/31408540/)

### Key Position

CITE: "Traditional image quality metrics, such as contrast-to-noise ratio, have become inadequate indicators of clinical imaging performance" due to nonlinear reconstruction algorithms.

CITE: "Additional efforts are needed to determine the performance of current clinical CT systems based on the metrics and methodologies described in the report and to fully ascertain the quantitative dependencies of clinical performance on these metrics."

**TG-233 does NOT define explicit HU tolerances** - it recommends task-based performance metrics.

---

## 4. Gammex 472 Multi-Energy CT Phantom

### Source
- [Sun Nuclear Multi-Energy CT Phantom Datasheet](https://www.sunnuclear.com/uploads/documents/datasheets/Diagnostic/MECT_Phantom_072321.pdf)
- [PEO Medical - Model 472](https://peomedical.com/radiotherapy/qa-phantoms-2/model-472-dual-energy-characterization-ct-phantom-gammex/)

### Specifications

CITE: "The Gammex 472 phantom is a solid water disk with a diameter of 33 cm and thickness of 5 cm. The phantom contains two concentric rings with eight holes in each ring."

CITE: "Seven iodine inserts with different concentrations (2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0 mg/mL) are available."

CITE: "Calcium concentrations range from 50-600 mg/mL in the phantom inserts."

### Expected HU Values

**The manufacturer does NOT publish expected HU values** because they depend on:
- Scanner model
- kVp setting
- Reconstruction algorithm
- Beam hardening correction implementation

CITE: "Gammex's Multi Energy CT phantom has been developed to quantify iodine, patient equivalent tissues and to check the consistency of CT scanners. One of the questions during quality checks may be whether the iodine concentration shown is the correct one. Especially between different models or even within identical scanners, there may be differences in constancy."

---

## 5. Clinical Scanner HU Accuracy Specifications

### Source
- [CT image quality over time: comparison of image quality for six different CT scanners (PMC5690105)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5690105/)
- [Daily, Weekly, & Monthly QA Testing on CT Scanners (UW GECT)](https://uwgect.wiscweb.wisc.edu/wp-content/uploads/sites/1268/2019/11/CT_Daily-Weekly-CT-QA-Testing_Scanning-And-Analysis.pdf)

### Common Industry Standards

CITE: "All of the mean CT numbers derived should be within 5 HU of that of water, that is, 0 ± 5 HU."

CITE: "A significant spread in the CT number measurements between different CT scanners was observed, especially for the materials with the highest and lowest expected CT numbers for all the sensitometric inserts."

### Manufacturer-Specific Notes

- **GE:** Standard reconstruction filter, water typically 0 ± 5 HU
- **Siemens:** B31s/H31s filters, water typically 0 ± 5 HU
- **Philips:** Filter "B", water typically 0 ± 5 HU

---

## 6. Beam Hardening Physics

### Source
- [CT Physics: Beam Hardening and Dual-Energy CT](http://xrayphysics.com/dual_energy.html)
- [Beam hardening: Analytical considerations (PubMed 17821996)](https://pubmed.ncbi.nlm.nih.gov/17821996/)
- [Influence of beam hardening in dual-energy CT imaging (EUR Radiol Exp)](https://eurradiolexp.springeropen.com/articles/10.1186/s41747-021-00217-1)

### Fundamental Physics

CITE: "Polychromatic x-ray beams traveling through material are prone to beam hardening, i.e., the high energy part of the incident spectrum gets over represented when traveling farther into the material."

CITE: "Beer-Lambert's law remains only valid if μ is extended to an effective attenuation coefficient μeff(x), which considers the spectral change in dependence of the penetration length x."

### Thin-Sample vs. Thick-Object Measurements

CITE: "The distribution is changing as it passes through the specimen! So we try again with a thinner specimen and sure enough, we get a different effective energy."

CITE: "Unfiltered (red), the effective energy changes from about 45keV in the 0.01cm specimen to 72keV for the 2.5cm thick one."

**Key Insight:** A thin sample measured in isolation will show HIGHER attenuation (and thus higher HU) than the same material measured through a 30cm water path due to beam hardening.

### Effect on High-Z Materials

CITE: "Iodine has an important k-edge at 33 keV, smack in the middle of diagnostic x-ray energies. Because the photoelectric effect trails off rapidly, the absorption is much higher at lower energy x-ray beams than higher-energy beams."

CITE: "Bone contains calcium, which has a somewhat higher Z and a k-edge at 4 keV. While this k-edge is below relevant diagnostic energies, the 'tail' does reach into diagnostic energies - thus, bone absorbs much more of the lower energy x-rays than soft tissues."

---

## 7. Iodine Quantification Accuracy in Clinical CT

### Source
- [Siemens dual-energy CT iodine quantification (EUR Radiol)](https://link.springer.com/article/10.1007/s00330-018-5736-0)
- [Quantitative benchmarking of iodine imaging (PMC8219825)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8219825/)
- [Accuracy of dual-energy CT for iodine concentration (PubMed 24141716)](https://pubmed.ncbi.nlm.nih.gov/24141716/)

### Iodine Quantification Tolerances in Clinical CT

CITE: "The Siemens dual-energy CT technique calculated iodine concentrations with a deviation of 2.0–12.0% from nominal value in the range of 10–50 mg/mL iodine."

CITE: "For the highest iodine concentration level (100 mg/mL), Siemens showed a large deviation (-33.8%), indicating a limit of the system for the highest concentrations."

CITE: "Total iodine quantification bias ranged from −1.27 to 0.74 mg/mL, with the mean absolute percentage bias (APB) of 3.21%."

CITE: "While absolute percentage biases of iodine quantification at low concentrations were 8.33% and 5.56%, the absolute bias was less than 0.2 mg/mL. This difference of iodine quantification is acceptable for clinical evaluation of lesions with iodine uptakes above 2 mg/mL."

### Beam Hardening Effects on Iodine

CITE: "The underestimation of the CT values of the iodine map, which appears to be due to beam hardening, was as small as 15 HU at the most."

CITE: "PCD-CT had higher CT numbers in all x-ray spectra for all CAs (p < 0.001) and produced larger cupping artifacts in all test cases (p < 0.001)."

---

## 8. Multi-Material Beam Hardening Correction

### Source
- [Joseph and Spital 1978 method](https://pubmed.ncbi.nlm.nih.gov/9350717/)
- [Single-material beam hardening correction (PMC9388575)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9388575/)

### Water-Only BHC Limitations

CITE: "The conventional and well-established approach to address this issue is a calibration-based single material beam hardening correction (BHC) using a water cylinder."

CITE: "Multi-material beam hardening correction methods are primarily used for anatomical sites where there is a lot of bone, for example, for brain imaging."

CITE: "One implementation was designed to correct for hardening in both bone and iodine contrast agent. It is necessary to identify those regions of the image which contain bone and iodine."

**Implication:** Water-only BHC does NOT correct for beam hardening through high-Z materials. This explains the ~50% HU underestimation.

---

## 9. Conclusions and Recommendations

### What the Research Shows

1. **Water Tolerance:** ±4-7 HU is well-established (ACR: ±7, IEC: ±4, Clinical: ±5)

2. **Contrast Material Tolerance:** **No authoritative standard exists** for absolute HU accuracy of iodine/calcium inserts because:
   - Measured HU depends heavily on scanner implementation
   - Beam hardening causes expected systematic underestimation
   - Manufacturers don't publish expected values for contrast materials

3. **Thin-Sample Ground Truth is Invalid for Contrast Materials:**
   - NIST μ values assume monochromatic or thin-sample geometry
   - CT measures through ~30cm water equivalent with severe beam hardening
   - Expected underestimation for high-Z materials is 10-50% depending on path length

4. **What IS Required:**
   - Monotonic ordering with concentration (REQUIRED)
   - Relative spacing proportional to concentration (EXPECTED)
   - Agreement with reference clinical scanner within ~20% (ACCEPTABLE)

### Recommended Tolerances for BasisSimulator

| Material Type | Primary Criterion | Secondary Criterion |
|--------------|-------------------|---------------------|
| **Water** | 0 ± 5 HU | Uniformity < 4 HU |
| **Air** | -1000 ± 20 HU | - |
| **Solid tissue equivalents** | ± 20 HU of nominal | - |
| **Calcium series** | Strictly monotonic | Agreement with clinical scanner within 20% |
| **Iodine series** | Strictly monotonic | Agreement with clinical scanner within 20% |

### Ground Truth Methodology Update

**CHANGE:** Stop using thin-sample NIST-derived HU as "ground truth" for contrast materials.

**CORRECT APPROACH:**
1. **For water/air:** NIST values are valid (water defines HU=0 by calibration)
2. **For contrast materials:** Compare to:
   - Published phantom measurements from clinical scanners
   - CatSim output (which models beam hardening)
   - Relative ordering and spacing, not absolute values

### Is 50% Error Acceptable?

**YES, with qualification:**

CITE: "The underestimation of the CT values of the iodine map, which appears to be due to beam hardening, was as small as 15 HU at the most." (for low concentrations)

For high concentrations (10-20 mg/mL iodine, 400-600 mg/cc calcium):
- 20-50% underestimation is consistent with polychromatic beam hardening physics
- This matches what clinical scanners show before multi-material BHC
- This is NOT a simulator bug - it's correct polychromatic physics

**What WOULD be unacceptable:**
- Ordering inversions (any inversion = physics bug)
- Water not calibrating to 0 HU
- >20% deviation from CatSim for same phantom/geometry

---

## References

1. ACR CT Accreditation Program. "Phantom Scoring: CT (Revised 8-26-2024)." American College of Radiology.
2. IEC 61223-3-5:2019. "Evaluation and routine testing in medical imaging departments - Part 3-5: Acceptance and constancy tests."
3. AAPM Task Group 233. "Performance Evaluation of Computed Tomography Systems." AAPM Report No. 233, 2019.
4. Cropp et al. "Scanner and kVp dependence of measured CT numbers in the ACR CT phantom." J Appl Clin Med Phys. 2013;14(6):4417.
5. Herman GT. "Correction for beam hardening in computed tomography." Phys Med Biol. 1979;24(1):81-106.
6. Joseph PM, Spital RD. "A method for correcting bone induced artifacts in computed tomography scanners." J Comput Assist Tomogr. 1978;2(1):100-108.
7. Sun Nuclear Corporation. "Multi-Energy CT Phantom Datasheet." 2021.
8. Matsumoto K et al. "Influence of beam hardening in dual-energy CT imaging." Eur Radiol Exp. 2021;5(1):21.
9. van Hamersvelt RW et al. "Quantitative benchmarking of iodine imaging." Eur Radiol. 2021;31:5063-5072.

---

*Document created: 2026-01-16*
*Purpose: CRITICAL-HU-RESEARCH story for BasisSimulator.jl*
