"""
    src/geometry/clinical_scanners.jl

The clinical scanners the documentation models, each a documented constant: an instance of
[`EICTScanner`](@ref) or [`PCCTScanner`](@ref), never a preset function. A different scan of the
same scanner is a variant made with the copy constructor, e.g.
`EICTScanner(SOMATOM_FORCE; detector_rows = 48)`. A dual-source scanner is two constants, one per
tube/detector chain (`SOMATOM_FORCE` and `SOMATOM_FORCE_B`); a dual-source or dual-kVp scan is two
ordinary acquisitions, each with its own protocol.

The values are those the documentation's notebooks pass to the constructors, so the constants
reproduce BasisSimulator's documented behaviour. A value the vendor does not publish is marked as
an assumption in the docstring.
"""

"""
    REVOLUTION_APEX

The GE Revolution Apex Elite: a 256-row wide-cone energy-integrating scanner with a curved
Gemstone garnet array (`detector_material = :lumex`, the Geant4 table
[`GEMSTONE_MC_EFFICIENCY_LUT`](@ref)).

- 256 × 0.625 mm rows, 160 mm coverage: FDA K133705, K213715.
- 834 columns of 0.6 mm at the isocentre, source–isocentre 625.6 mm, source–detector 1100 mm:
  not published; the documentation notebooks' values.
- Focal spot 1.0 × 1.0 mm, 10° anode, 2.5 mm Al flat filter, GE large-body bowtie: the notebooks.
- 3.0 mm garnet at 6.40 g/cm³, the depth of its Geant4 table: an assumption (XCIST's default).
- Fill factor 0.9 × 0.9 and detection gain 10 e⁻/keV: assumptions.
- Electronic noise 3500 e⁻, the clinical DAS readout of notebooks 01, 02 and 05. The VMI notebooks
  (03, 06, 07) run without it: `EICTScanner(REVOLUTION_APEX; electronic_noise = 0.0)`.

Rapid kVp switching is two acquisitions on this scanner, one protocol per kVp, each with its
share of the view period (notebook 03).
"""
const REVOLUTION_APEX = EICTScanner(;
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_col_offset = 0.25,
    detector_shape = :arc,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 3500.0,
    detection_gain = 10.0,
)

"""
    NAEOTOM_ALPHA

The Siemens NAEOTOM Alpha's photon-counting detector A in its standard mode: 2 × 2 native CdTe
dexels summed digitally after counting, 1376 pixel columns (2752 native) over the 50 cm field and
144 rows (288 native), 57.6 mm along z. The pixel pitch at the isocentre is 0.373 × 0.399 mm
(0.68 × 0.727 mm at the detector, the native pitch 0.34 × 0.364 mm: the 0.275 × 0.322 mm
electrode plus its share of the anti-scatter grid's gaps), magnified 1113/610.

- Source–isocentre 610 mm, source–detector 1113 mm: Li 2025; Hennemann 2026.
- Pitch 0.37 × 0.40 mm at the isocentre: Li 2025, Table 1. 144 × 0.4 mm, 1376 columns (2752
  native): K243523, Tables 3–4.
- CdTe 1.6 mm, the 2-D grid of 4 × 6 dexel cells, the transported charge (16 µm) and 1.9 keV of
  electronic noise: the response table, [`PCCT_NATIVE_TABLE_PATH`](@ref) (Geant4, BasisMonteCarlo
  notebook 03; Flohr & Schmidt 2023; Zhan 2026).
- Thresholds 20, 35, 55 and 70 keV (bins 20–35, 35–55, 55–70, > 70 keV): Sartoretti 2023.
- Dead time 27 ns per native dexel, retriggering comparators after a 10 ns CR-RC shaper: the fit
  of notebook 03 to Li 2025's count loss.
- Focal spot 0.4 × 0.5 mm, 7° anode geometry, 3 mm Al, large-body bowtie, 820 mm bore: the notebooks.
- Scatter correction on, exact Poisson counts: the notebooks' detector toggles.

Its dexels are ideal (no gain or threshold spread). BasisMonteCarlo notebook 03 assumes a 2 % gain
spread and 0.35 keV of threshold spread (unpublished); that detector is
`PCCTScanner(NAEOTOM_ALPHA; gain_spread = 0.02, threshold_spread_keV = 0.35)`, corrected by
[`calibrate_flat_field`](@ref) → [`apply_flat_field!`](@ref).

Its ultra-high-resolution mode is [`NAEOTOM_ALPHA_UHR`](@ref). A different scan is a variant:
`PCCTScanner(NAEOTOM_ALPHA; pileup = false)`.
"""
const NAEOTOM_ALPHA = let native_pitch = (0.34, 1.13 * 0.322), sid = 610.0, sdd = 1113.0, binning = 2
    PCCTScanner(;
        source_to_isocenter = sid,
        source_to_detector = sdd,
        detector_rows = 144,
        detector_cols = 1376,
        detector_row_size = native_pitch[2] * binning * sid / sdd,   # 0.399 mm
        detector_col_size = native_pitch[1] * binning * sid / sdd,   # 0.373 mm
        detector_shape = :arc,
        detector_col_offset = 0.25,
        focal_spot_width = 0.4,
        focal_spot_length = 0.5,
        target_angle = 7.0,
        gantry_rotation_time = 0.5,
        scan_diameter = 500.0,
        gantry_aperture = 820.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        bowtie_filter = :large_body,
        detector_material = :CdTe,
        detector_depth = 1.6,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        dead_time_ns = 27.0,
        binning_factor = binning,
        scatter_correction = true,
        noise_reduction = 0.0,
    )
end

"""
    NAEOTOM_ALPHA_UHR

The NAEOTOM Alpha in its ultra-high-resolution mode: every native dexel read on its own,
0.186 × 0.199 mm at the isocentre, 120 rows of 0.2 mm (24 mm along z) and 2752 columns. A native
dexel's counts are Poisson; its neighbours covary through the photons they share. Each dexel
piles up at its own rate, a quarter of a standard pixel's.

The mode reads one threshold at native resolution, 20 keV; the other thresholds are read binned
(standard mode, [`NAEOTOM_ALPHA`](@ref)).
"""
const NAEOTOM_ALPHA_UHR = PCCTScanner(
    NAEOTOM_ALPHA;
    detector_rows = 120,
    detector_cols = 2752,
    detector_row_size = NAEOTOM_ALPHA.detector_row_size / 2,
    detector_col_size = NAEOTOM_ALPHA.detector_col_size / 2,
    binning_factor = 1,
    energy_thresholds = [20.0],
)

"""
    SOMATOM_FORCE

The Siemens SOMATOM Force's tube/detector A, a third-generation dual-source scanner whose second
chain is [`SOMATOM_FORCE_B`](@ref): 96 rows of 0.6 mm (57.6 mm) and 920 channels of 0.054°
(0.561 mm at the isocentre), with a UFC Gd₂O₂S detector (`detector_material = :ufc`,
BasisSimulator's Force Monte Carlo efficiency table).

- 96 × 0.6 mm rows, 920 channels: Force white paper §1.2; FDA K133589.
- Source–isocentre 595 mm, source–detector 1085.6 mm, 0.561 mm columns: not published; the
  documentation notebook's values.
- Focal spot 0.8 × 1.2 mm, 8° anode, 3 mm Al flat filter: assumptions (the Vectron-family stack;
  the protocol adds the tube's 0.9 mm titanium window, `additional_filters = [("Ti", 0.9)]`).
- Large-body bowtie: an assumption (Siemens' profile is unpublished).
- UFC depth 1.4 mm, the only UFC thickness Siemens quotes; fill factor 0.9 × 0.9.
- Electronic noise 0 and detection gain 10 e⁻/keV: notebook 09's assumptions.

The real detector is spherical; this one is the equiangular arc, which has the same angles
across the fan and along z.
"""
const SOMATOM_FORCE = EICTScanner(;
    source_to_isocenter = 595.0,
    source_to_detector = 1085.6,
    detector_rows = 96,
    detector_cols = 920,
    detector_row_size = 0.6,
    detector_col_size = 0.561,
    focal_spot_width = 0.8,
    focal_spot_length = 1.2,
    target_angle = 8.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 3.0,
    bowtie_filter = :large_body,
    detector_material = :ufc,
    detector_depth = 1.4,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 0.0,
    detection_gain = 10.0,
)

"""
    SOMATOM_FORCE_B

The SOMATOM Force's tube/detector B: [`SOMATOM_FORCE`](@ref) with detector B's 640 channels and its
35.5 cm field of view (Force white paper; Wang et al. 2021; Flohr & Schmidt, DE-DSCT chapter).
Tube B sits 95° from tube A in the gantry, and detector B 0.88 mm from A along z (Wang et al.
2021): neither is a scanner parameter, since each tube's scan is its own full acquisition; the
z offset is applied to tube B's phantom, as notebook 09 does.

A dual-energy scan is two acquisitions, one per chain: `SOMATOM_FORCE` at the low kV and
`SOMATOM_FORCE_B` at the high, tin-filtered kV, e.g.
`CTProtocol(kVp = 140, …, additional_filters = [("Ti", 0.9), ("Sn", 0.6)])`. To co-register the
two sinograms ray by ray, run tube B on tube A's arc (`SOMATOM_FORCE`) with an object inside
detector B's 35.5 cm field, as notebook 09 does.
"""
const SOMATOM_FORCE_B = EICTScanner(SOMATOM_FORCE; detector_cols = 640, scan_diameter = 355.0)

"""
    SOMATOM_DEFINITION_FLASH

The Siemens SOMATOM Definition Flash's tube/detector A, a second-generation dual-source scanner
(2008) whose second chain is [`SOMATOM_DEFINITION_FLASH_B`](@ref): 64 rows of 0.6 mm (38.4 mm)
and 736 channels of 0.70473 mm at the isocentre (49.94°, a 50.2 cm field), with a UFC Gd₂O₂S
detector of its own Monte Carlo efficiency table (`detector_material = :ufc_flash`, never the
Force's `:ufc`).

- 64 × 0.6 mm rows, 736 channels: FDA K133589 (comparison table); Siemens datasheet (2010).
- Source–isocentre 595 mm, source–detector 1085.6 mm, arc detector: DICOM-CT-PD / AAPM LDCT
  projection data.
- Focal spot 0.7 × 0.7 mm, 7° anode (IEC 60336); 78 cm bore, 50 cm scan field: Siemens datasheet.
- 8.4 mm Al-equivalent flat filtration (6.8 tube + 1.6 beam-limiting device): Siemens datasheet.
- Large-body bowtie: an assumption (profile unpublished).
- UFC depth 1.4 mm, the only UFC thickness Siemens quotes, and the depth of its Geant4 table
  ([`UFC_FLASH_MC_EFFICIENCY_LUT`](@ref)); fill factor 0.9 × 0.9: assumptions (proprietary). The
  fill factor scales the quanta, hence the noise, and cancels in the air calibration.
- Electronic noise 1500 e⁻ (the Stellar detector's floor; ≈ 3500 e⁻ before 2011) and detection
  gain 10 e⁻/keV: assumptions.
"""
const SOMATOM_DEFINITION_FLASH = EICTScanner(;
    source_to_isocenter = 595.0,
    source_to_detector = 1085.6,
    detector_rows = 64,
    detector_cols = 736,
    detector_row_size = 0.6,
    detector_col_size = 0.70473,
    detector_shape = :arc,
    focal_spot_width = 0.7,
    focal_spot_length = 0.7,
    target_angle = 7.0,
    gantry_rotation_time = 0.5,
    scan_diameter = 500.0,
    gantry_aperture = 780.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 8.4,
    bowtie_filter = :large_body,
    detector_material = :ufc_flash,
    detector_depth = 1.4,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 1500.0,
    detection_gain = 10.0,
)

"""
    SOMATOM_DEFINITION_FLASH_B

The SOMATOM Definition Flash's tube/detector B: [`SOMATOM_DEFINITION_FLASH`](@ref) with detector
B's 480 channels (30,720 elements over 64 rows, Siemens datasheet), 32.57° and a 33.4 cm field of
view. Tube B sits 95° from tube A (Petersilka et al. 2008; 94° also appears in the literature);
its z offset is not published for the Flash.

A dual-energy scan is two acquisitions, one per chain: `SOMATOM_DEFINITION_FLASH` at 80 or 100 kV
and `SOMATOM_DEFINITION_FLASH_B` at 140 kV behind the 0.4 mm tin Selective Photon Shield (Primak
et al. 2010), `CTProtocol(kVp = 140, …, additional_filters = [("Sn", 0.4)])`. To co-register the
two sinograms ray by ray, run tube B on tube A's arc with an object inside detector B's field, as
notebook 12 does.
"""
const SOMATOM_DEFINITION_FLASH_B = EICTScanner(SOMATOM_DEFINITION_FLASH; detector_cols = 480, scan_diameter = 334.0)

export REVOLUTION_APEX, NAEOTOM_ALPHA, NAEOTOM_ALPHA_UHR, SOMATOM_FORCE, SOMATOM_FORCE_B, SOMATOM_DEFINITION_FLASH, SOMATOM_DEFINITION_FLASH_B
