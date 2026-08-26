# Siemens SOMATOM Definition Flash — Complete Scanner Spec Dossier

Prepared for BasisSimulator.jl scanner modeling · 2026-08-20

> ## ⚠️ Repo addendum — 2026-08-26 (READ FIRST)
>
> This dossier was prepared on 2026-08-20, **before** Hamidreza Khodajou-Chokami
> finished the Flash-specific Monte Carlo detector simulation
> (`flash_efficiency_results.csv`, CRSP `share/`, received 2026-08-26).
> Two statements below are now **superseded by measured data**:
>
> 1. **Part 0's "Key consequence" claim** — *"the existing `UFC_MC_EFFICIENCY_LUT`
>    appl[ies] verbatim. No new Monte Carlo is required"* — is **WRONG**. The Flash
>    MC shows the same Gd₂O₂S material signatures (L-edge dip at 8 keV, K-edge
>    fluorescence-escape crash at 50→51 keV) but a **distinctly thinner crystal**:
>    η is identical below 30 keV, −4.6% at 50 keV, −8% at 100 keV, and
>    **−28% at 140 keV (0.588 vs the Force's 0.816)**. The repo now ships a
>    dedicated `UFC_FLASH_MC_EFFICIENCY_LUT` + `detector_efficiency_ufc_flash()`
>    (`src/detector/detector_efficiency.jl`), and a regression test forbids
>    aliasing the two LUTs.
> 2. **Part 3's ready-to-use config** should therefore read
>    `detector_material = :ufc_flash` (routed in `build_physics_config`) and
>    `detector_depth = 1.0` (documented assumption, inert in `:mc_lut` mode;
>    the MC high-energy roll-off implies a Beer-Lambert-effective depth of
>    ≈1.07 mm).
>
> Open question for Hamid: the Beer-Lambert-effective Flash/Force depth ratio is
> **not constant** (0.77 at 100 keV → 0.52 at 140 keV), so the difference is not
> purely a thickness change — what crystal depth, reflector, and backing were
> modeled for each scanner? (The 1–5 keV values are bit-identical between the two
> runs — same MC framework/seed; low-energy photons are absorbed before the
> geometries differ — which is a good internal-consistency signal.)
>
> Everything else in the dossier (geometry, tubes, filtration, protocols,
> timing) stands as the parity checklist for the simulated dual-source Flash.

---

## Part 0 — What we already have modeled (orientation)

Three detectors currently carry Monte-Carlo-derived physics in this repo:

| # | Detector | Scanner | MC artifact | What the MC captures |
|---|----------|---------|-------------|----------------------|
| 1 | **GE Gemstone / Lumex** Ce:(Tb,Lu)₃Al₅O₁₂ garnet, 3.0 mm | GE Revolution Apex Elite (FDA K213715) | `GEMSTONE_MC_EFFICIENCY_LUT` (`src/detector/detector_efficiency.jl`), 1–140 keV on a 1-keV grid, MCNP | K-fluorescence **escape** dips at Tb 52.0 keV and Lu 63.31 keV (η *drops*, Beer-Lambert predicts a rise); high-E roll-off |
| 2 | **Siemens UFC** Gd₂O₂S:Pr,Ce, 1.4 mm | SOMATOM Force *StellarInfinity* | `UFC_MC_EFFICIENCY_LUT`, 1–140 keV, Khodajou-Chokami `efficiency_results.csv` 2026-06-08 | Gd K-edge escape 50.24 keV (0.969 → 0.741 between 50 and 51 keV); Gd L-edges 7.24/7.93/8.38 keV; 0.897@100 → 0.816@140 keV |
| 3 | **CdTe** 1.6 mm direct-conversion | Siemens NAEOTOM Alpha (FDA K201501) | Full **DRM** `cdte_response_v4.jls` + MC pile-up matrix (`mc_response.jl`, `mc_pileup.jl`) | Full photon transport (PE/Compton/Rayleigh), Fano noise, Dreier-2018 charge cloud, 3×3 erf charge splitting, electronic noise, threshold binning, per-photon dead-time/pile-up migration matrix S |

Legacy reference in `SCANNERS.md`: Canon Aquilion ONE (geometry only, no MC).

**Key consequence for the Flash: it uses the *same UFC Gd₂O₂S:Pr,Ce scintillator* as the Force.** ~~The existing `UFC_MC_EFFICIENCY_LUT` and the `Scanner(detector_material = :ufc)` dispatch in `src/api/driver.jl` apply verbatim. No new Monte Carlo is required — this is a geometry + source + filtration job only.~~ **← SUPERSEDED, see the 2026-08-26 addendum above: the Flash has its own MC LUT (`:ufc_flash`) with a −28% high-energy divergence from the Force.**

---

## Part 1 — The spec checklist to model a NEW scanner

Everything a `Scanner` needs, in three tiers.

### Tier A — hard requirements (`Scanner` struct fields, `src/geometry/scanner.jl`)

| Field | Units | Notes |
|-------|-------|-------|
| `source_to_isocenter` | mm | SID |
| `source_to_detector` | mm | SDD (mag = SDD/SID) |
| `detector_rows` / `detector_cols` | count | physical, not "slices" |
| `detector_row_size` / `detector_col_size` | mm **at isocenter** | col size at iso = (pitch at detector)/mag |
| `detector_row_offset` / `detector_col_offset` | rows / cols | quarter-detector offset |
| `detector_shape` | `:arc` (default) / `:flat` | all clinical 3rd-gen MDCT = `:arc` |
| `focal_spot_width` / `focal_spot_length` | mm | IEC 60336 nominal |
| `target_angle` | ° | anode angle → drives heel effect + spectrum |
| `gantry_rotation_time` | s | |
| `scan_diameter` / `gantry_aperture` | mm | SFOV / bore |
| `flat_filter_material` / `flat_filter_thickness` | `:aluminum/:copper/:titanium`, mm | spectrum module also accepts `"Sn"`, `"Ti"`, `"C"`, `"W"` |
| `bowtie_filter` | `:large_body/:medium_body/:small_body/:head/:none` | |
| `detector_material` | `:lumex`, `:ufc`, or `:ufc_flash` for EID; `:cdte/:czt/:si` for PCCT | **must map to an existing MC LUT or you must generate one** |
| `detector_depth` | mm | inert in `:mc_lut` mode |
| `fill_factor_row` / `fill_factor_col` | 0–1 | cancels in air calibration |
| `detection_gain` | e⁻/keV | |
| `electronic_noise` | e⁻ | |
| `detector_type` | `:energy_integrating` / `:photon_counting` | |

PCCT-only extras: `n_energy_bins`, `energy_thresholds`, `energy_resolution`, `charge_sharing_fwhm`, `dead_time_ns`, `pixel_mode`, `native_dexel_col_mm`, `native_dexel_row_mm`, `binning_factor`.

### Tier B — protocol / source (`CTProtocol`, `src/source/protocol.jl` + `spectrum.jl`)
kVp list · mA range and max · rotation time · views per rotation · collimation options · added filters per mode (e.g. tin) · pitch range.

### Tier C — the things vendors never publish (must be *documented assumptions*)
Bowtie profile shape · scintillator layer depth · fill factor · absolute electronic-noise figure · exact flat-filter stack composition · tube-B z-offset. The nb09 convention is to declare these in a `!!! warning "Documented modeling assumptions"` admonition and use the CatSim stand-in.

---

## Part 2 — SOMATOM Definition Flash: full specification

Second-generation dual-source CT (2008/2009). Predecessor of the SOMATOM Force already modeled in nb09.

### 2.1 Regulatory / identification

| Item | Value | Source |
|------|-------|--------|
| Original FDA 510(k) | **K082220**, cleared 10/10/2008 (product name "SOMATOM P47", trade name *SOMATOM Flash DS*) | K082220 510(k) summary |
| Stellar Detector variant | **K113342**, cleared 12/29/2011 ("SOMATOM Definition Flash (with Stellar Detector)"), predicate K082220 | K113342 |
| SOMARIS/7 VA44 + 0.5 mm slices | **K121072**, cleared 05/08/2012 | K121072 |
| Later bundled clearances | K122471 (2012), K173630 (2017, syngo CT VB10), K230421 (2023) | FDA |
| Successor predicate use | SOMATOM Force **K133589** (2014) cites K121072 as predicate | K133589 |
| Product code / class | 90 JAK, Class II, 21 CFR §892.1750 | FDA |

### 2.2 Gantry geometry — **THE CORE NUMBERS**

| Parameter | Value | Source |
|-----------|-------|--------|
| **SID** (source→isocenter) | **595.0 mm** | DICOM-CT-PD / AAPM LDCT projection data acquired on Definition Flash |
| **SDD** (source→detector) | **1085.6 mm** | same |
| Magnification SDD/SID | **1.824538** | derived |
| **Detector shape** | **Arc / cylindrical, equiangular, concentric on the focal spot** | LDCT-PD ("equiangularly distributed along a curvilinear array") |
| **Angular offset tube A ↔ tube B** | **95°** (increased from 90° on the 1st-gen Definition to fit the wider B detector; 94° also appears in the literature) | multiple technical reviews |
| Gantry bore (aperture) | **78 cm** | Siemens datasheet Dec-2010 + 2016 brochure |
| Scan field of view (A) | **50 cm** (78 cm extended-FOV option) | Siemens datasheet |
| Gantry front → scan plane | 35 cm | Siemens datasheet |
| Gantry tilt | **None** (two tube/detector arrays prevent tilting) | NHS attribute sheet |

### 2.3 Detector — **fully pinned down, cross-validated 3 ways**

| Parameter | Detector A | Detector B | Source |
|-----------|-----------|-----------|--------|
| **Number of detector elements** | **47,104** | **30,720** | **Siemens datasheet: "77,824 (47,104 system A; 30,720 system B)"** |
| **Rows** | **64** | **64** | Siemens datasheet ("Number of detector rows 2 × 64") |
| **Channels (= elements / rows)** | **736** | **480** | derived, exact; independently confirmed by NICE MIB54 ("736 detectors per row") and by LDCT-PD ("736 detector elements") |
| FDA "measuring channels" | 1472 | 960 | K133589 — exactly 2× the physical count (same convention gives Force 1840/2 = 920 ✓) |
| **Column pitch at detector** | **1.2858 mm** | 1.2858 mm | LDCT-PD dataset geometry |
| **Column pitch at isocenter** | **0.70473 mm** | 0.70473 mm | derived = 1.2858 / 1.824538 |
| Angular channel pitch | **0.0678497° (1.18427 mrad)** | same | derived = 1.2858 / 1085.6 |
| **Fan angle** | **49.94°** | **32.57°** | derived |
| **FOV at isocenter** | **502.4 mm (50 cm)** ✓ | **333.7 mm (33 cm)** ✓ | derived — matches published 50 cm / 33 cm exactly |
| **Row pitch at isocenter** | **0.600 mm** | 0.600 mm | 64 × 0.6 = 38.4 mm ✓ |
| Row pitch at detector | 1.0947 mm | 1.0947 mm | derived = 0.6 × 1.824538 (Force measured 1.094 mm — same detector module) |
| **z-coverage per rotation** | **38.4 mm** | 38.4 mm | FDA K173630 ("2 × 38.4 mm"); NHS ("2 × 38.4, 2 × 64 × 0.6 mm") |
| Max slices/rotation | **2 × 128** (z-FFS doubles 64 rows) | | Siemens datasheet |
| DAS electronic channels | 2 × 128 | | Siemens datasheet |
| **Scintillator** | **UFC — Ultra Fast Ceramic, Gd₂O₂S:Pr,Ce** (ρ ≈ 7.34 g/cm³) | | Siemens datasheet ("2 × Multislice UFC Detector"); NHS ("Detector type: Siemens UFC") |
| Detector generation | pre-2011: UFC + discrete photodiode/ASIC. 2011+: **Stellar** = UFC with the ASIC integrated into the photodiode (TrueSignal), lower electronic noise + less cross-talk, enables 0.5 mm slices / 0.30 mm cross-plane | K113342; Siemens brochure |

> **Note (1st-gen contrast):** the original SOMATOM Definition (K052216, 2005) had detector B at only **26 cm** FOV and 90° offset. The Flash's 33 cm B-FOV is a defining second-gen change.

### 2.4 X-ray tubes

| Parameter | Value | Source |
|-----------|-------|--------|
| Tube | **2 × STRATON MX P** (rotating-envelope, direct anode cooling) | FDA K133589 comparison table; K173630 Table 4 |
| **Generator power** | **200 kW total (2 × 100 kW)** | Siemens datasheet + brochure + K173630 |
| **kV steps** | **80, 100, 120, 140 kV** (Dec-2010 datasheet) → **70, 80, 100, 120, 140 kV** on later software | Siemens datasheet 2010; Siemens brochure 2016; FDA K173630 Table 4; NHS |
| **Tube current** | single-source **20–800 mA**; dual-source **40–1600 mA** | Siemens datasheet; NHS |
| **Focal spot (IEC 60336)** | **0.7 × 0.7 mm @ 7°** and **0.9 × 1.1 mm @ 7°** | **Siemens datasheet** |
| **Anode angle** | **7°** | Siemens datasheet (both focal spots quoted "/7°") |
| Straton design point | 0.6 × 0.7 mm² focal spot, focal-track radius 48 mm, anode rotation 150 Hz, anode angle 7–9° | Schardt et al., Med Phys 2004 |
| Anode heat storage | **0 MHU** (0.53 MHU effective capacity), cooling **7.3 MHU/min (5,400 kJ/min)** ≈ a conventional 30 MHU tube | Siemens datasheet |
| Flying focal spot | **z-Sharp**: electron beam deflected, two focal spots **alternating 4,608 times/s**; interleaves half a slice width in z → doubles rows to 128 | Siemens datasheet |
| In-plane FFS | "Multifan principle with Flying Focal Spot" (φ-deflection also available) | Siemens datasheet |

### 2.5 Filtration — **this is the number you need for spectra**

| Layer | Value | Source |
|-------|-------|--------|
| **Tube assembly, Al equivalent** | **6.8 mm Al** | **Siemens datasheet, "CARE Filter → Al equivalent tube: 6.8 mm Al"**; corroborated by NHS attribute sheet ("Minimum filtration 6.8") |
| **Beam-limiting device, permanent** | **1.6 mm Al equivalent** | Siemens datasheet |
| **CARE Filter, mode-dependent** | **additional 0.5 mm Al** | Siemens datasheet |
| **⇒ total permanent filtration** | **8.4 mm Al eq.** (8.9 mm Al eq. with the mode-dependent CARE Filter) | derived |
| **Selective Photon Shield (tin filter)** | **0.4 mm Sn**, a flat tin sheet mounted to the **bottom of the bowtie filter, directly under the collimator**, on the high-kV tube (tube B) | Primak et al., AJR 2010 — "a single filter thickness of 0.4 mm was chosen for commercial implementation on the second-generation DSCT system (SOMATOM Definition Flash)"; AJR *Dual-Energy CT: General Principles* |
| Bowtie / form filters | "Multifan principle" — a set of shaped form filters (head / body). **Exact profile unpublished.** | Siemens datasheet |
| Reference HVL | ~7.27 mm Al @ 120 kVp (simulated CT beam of this class) | dosimetry literature |
| Dynamic z-collimator | **Adaptive Dose Shield** (asymmetric pre-/post-spiral collimation) | Siemens datasheet |
| Organ-based modulation | **X-CARE**; tube-current modulation **CARE Dose4D**; kV selection **CARE kV** | Siemens datasheet |

**Dual-energy pairs**: 80/Sn140 kV (head, extremities, bone-iodine — 80% better bone/iodine separation), **100/Sn140 kV** (routine abdomen/liver VNC; +30% bone-iodine contrast, more power reserve for large patients). A typical Flash liver-VNC protocol: tube A/B = 100/Sn140 kV, quality-reference **230/178 mAs**, collimation **32 × 0.6 mm**.

### 2.6 Acquisition / timing

| Parameter | Value | Source |
|-----------|-------|--------|
| **Rotation times** | **0.28 s** (optional), 0.33, 0.5, 1.0 s | Siemens datasheet; NHS |
| **Temporal resolution** | **75 ms**, heart-rate independent (down to 37.5 ms with 2-segment recon, except Flash Spiral) | Siemens datasheet + brochure |
| **Projections / rotation** | **up to 4,608 per 360° per data-acquisition unit** (FFS deflects 4,608×/s ⇒ **2,304 readings/rotation at 0.5 s = 1,152 per focal-spot position**; the AAPM LDCT Flash data is 2,304 views/rotation) | Siemens datasheet; AAPM LDCT projection data |
| **Pitch** | 0.35–3.0 routine; **up to 3.2 / 3.4 ECG-triggered Flash Spiral**; down to 0.3 (z-UHR), 0.17 (cardiac) | NHS; Flohr et al. Med Phys 2009 |
| **Max table/scan speed** | 400 mm/s Flash Spiral; **458 mm/s ECG-triggered Flash Spiral** | Siemens datasheet |
| Table feed | 2–200 mm/s; scannable range 200 cm | Siemens datasheet |
| Table load | 220 kg standard / 300 kg multi-purpose / up to 307 kg (later) | Siemens datasheet; 2016 brochure |
| Continuous scan time | 100 s | NHS |

### 2.7 Acquisition modes / collimations

- **Dual-source cardio / Flash Spiral / dual-power / dual-energy**: 2 × 128 × 0.6 mm
- **Single-source spiral**: 16 × 0.3 (z-UHR), 8 × 0.3 (z-UHR), 128 × 0.6, 64 × 0.6, 40 × 0.6, 32 × 0.6, 20 × 0.6, 16 × 0.6 (UHR comb), 10 × 0.6, 8 × 0.6 (UHR comb), 32 × 1.2 mm
- **Sequence**: 64 × 0.6, 32 × 0.6, 8 × 0.6 (UHR), 2 × 1, 32 × 1.2, 12 × 1.2, 1 × 5, 1 × 10 mm
- **Adaptive 4D Spiral Plus**: dynamic coverage up to 48 cm
- Acquired slice widths 0.6 / 1 / 1.2 / 5 / 10 mm; reconstructed 0.5–20 mm (increment down to 0.1 mm)

### 2.8 Image quality / reconstruction

| Parameter | Value |
|-----------|-------|
| Isotropic resolution | 0.33 mm (z-Sharp); **0.30 mm cross-plane with Stellar** |
| z-UHR option | 30 lp/cm (0.17 mm) at 0% MTF |
| Reconstruction matrix | 512 × 512; recon FOV min 5 cm |
| Recon speed | ~50 fps |
| CT number range | −1,024 to +3,071 HU |
| IR | SAFIRE (K103424, 2011), later ADMIRE |
| Beam hardening | **PFO — Posterior Fossa Optimization** (the Force upgraded this to IBHC, raw-data 3D forward-projection with a two-compartment iodine/water model) |
| MAR | iMAR (adaptive sinogram mixing) |

---

## Part 3 — Ready-to-use BasisSimulator config

> **2026-08-26 correction applied below**: `detector_material = :ufc_flash`
> (Flash-specific MC LUT), `detector_depth = 1.0`. See the addendum at top.

```julia
# ── Siemens SOMATOM Definition Flash — tube/detector A ────────────────────
scanner_A = BS.Scanner(
    source_to_isocenter = 595.0,      # mm — LDCT-PD / DICOM
    source_to_detector  = 1085.6,     # mm — LDCT-PD / DICOM
    detector_shape      = :arc,       # equiangular, focal-spot-concentric

    detector_rows       = 64,         # physical rows (128 slices via z-Sharp)
    detector_cols       = 736,        # 47,104 elements / 64 rows  (Siemens datasheet)
    detector_row_size   = 0.6,        # mm at iso  → 64 × 0.6 = 38.4 mm z-coverage
    detector_col_size   = 0.704729,   # mm at iso  = 1.2858 mm / 1.824538

    focal_spot_width    = 0.7,        # mm — IEC 60336 small focus, 0.7 × 0.7 @ 7°
    focal_spot_length   = 0.7,        # mm   (large focus: 0.9 × 1.1 mm)
    target_angle        = 7.0,        # ° — Siemens datasheet

    gantry_rotation_time = 0.28,      # s (min; 0.33 / 0.5 / 1.0 also available)
    scan_diameter        = 500.0,     # mm
    gantry_aperture      = 780.0,     # mm

    flat_filter_material = :aluminum,
    flat_filter_thickness = 8.4,      # mm Al eq. = 6.8 (tube) + 1.6 (BLD)
    bowtie_filter        = :large_body,   # ASSUMPTION — Siemens profile unpublished

    detector_material   = :ufc_flash, # → UFC_FLASH_MC_EFFICIENCY_LUT (Flash-specific MC)
    detector_depth      = 1.0,        # mm — inert in :mc_lut mode (BL-effective ≈1.07 mm)
    fill_factor_row     = 0.9,        # ASSUMPTION (cancels in air cal)
    fill_factor_col     = 0.9,
    detection_gain      = 10.0,       # e⁻/keV — repo UFC convention
    electronic_noise    = 0.0,        # Stellar/TrueSignal: negligible vs quantum
)

# ── Detector B: identical except channel count / FOV ──────────────────────
# detector_cols = 480  → 30,720 elements / 64 rows → 32.57° fan → 33.4 cm FOV

```

**Protocol (clinical dual-energy pair):**

```julia
protocol_low  = BS.CTProtocol(kVp = 100, mA = 460.0, rotation_time = 0.5)   # tube A
protocol_high = BS.CTProtocol(kVp = 140, mA = 356.0, rotation_time = 0.5,
                              added_filters = [("Sn", 0.4)])                # tube B
# 230 / 178 quality-reference mAs at 0.5 s ⇒ 460 / 356 mA (≈1.29:1 A:B)
# NOTE: bundled IPEM spectra top out at 140 kVp — that is exactly the Flash's
# top kV, so unlike the Force (150 kV) there is NO kV substitution needed here.

```

---

## Part 4 — Gaps and required documented assumptions

| Item | Status | Recommendation |
| --- | --- | --- |
| Bowtie profile | **Unpublished** | CatSim `:large_body` stand-in (repo convention, same as nb04/08/09) |
| Flat-filter *composition* | Only the **Al equivalent (6.8 + 1.6 mm)** is published, not the actual stack | Model as 8.4 mm Al equivalent, or split 3.0 mm Al + 0.9 mm Ti + balance as in nb09 — but 8.4 mm Al eq. is the *measured* figure here, so prefer it |
| UFC layer depth | Proprietary | 1.0 mm (inert in `:mc_lut`; MC BL-effective ≈1.07 mm — see addendum) |
| Fill factor | Proprietary | 0.9 (cancels in air calibration) |
| Absolute electronic noise | Never published | 0 for Stellar; a small nonzero value for pre-2011 UFC builds |
| Tube-B z-offset | Not published for the Flash (Force ≈ 0.88 mm) | Assume 0 or reuse the Force's 0.88 mm; a no-op for z-invariant phantoms |
| 94° vs 95° tube offset | Both appear in the literature; **95°** dominates | Irrelevant for full-rotation axial sims (nb09 §3 argument) |
| Row pitch at detector (1.0947 mm) | Derived, not directly published | Cross-checked against the Force's measured 1.094 mm — same module |
| Detector B fan truncation | Real: 33 cm | For a ≤33 cm phantom, give B the A arc (nb09 convention) so the (low, high) pair is per-ray co-registered |

---

## Sources

* FDA 510(k): [K082220](https://www.accessdata.fda.gov/cdrh_docs/pdf8/K082220.pdf) · [K113342](https://www.accessdata.fda.gov/cdrh_docs/pdf11/K113342.pdf) · [K121072](https://www.accessdata.fda.gov/cdrh_docs/pdf12/K121072.pdf) · [K122471](https://www.accessdata.fda.gov/cdrh_docs/pdf12/K122471.pdf) · [K133589 (Force, Flash comparison table)](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133589.pdf) · [K173630 (Table 4 hardware)](https://www.accessdata.fda.gov/cdrh_docs/pdf17/K173630.pdf) · [K230421](https://fda.innolitics.com/submissions/RA/subpart-b%E2%80%94diagnostic-devices/JAK/K230421)
* [Siemens SOMATOM Definition Flash datasheet, Dec 2010 (NHS Supply Chain)](https://media.supplychain.nhs.uk/media/documents/N0899429/Specification/33081_Datasheet_Definition_Flash_December_2010.pdf) — detector element counts, focal spots, filtration, tube, projections
* [NHS Supply Chain CT scanner attributes — SOMATOM Definition Flash](https://media.supplychain.nhs.uk/media/documents/N0899429/Specification/33080_CT%20scanner%20Attributes.pdf)
* [Siemens SOMATOM Definition Flash brochure (2016)](https://www.kbdentalconsulting.com/file/Siemens-Somatom-Definition-Flash-CT-Scanner.pdf)
* [Siemens SOMATOM Definition Flash Basic Planning Information](https://doclib.siemens-healthineers.com/rest/v1/view?document-id=412163)
* [DICOM-CT-PD User Manual v3 (TCIA)](https://www.cancerimagingarchive.net/wp-content/uploads/DICOM-CT-PD-User-Manual_Version-3.pdf) + [LDCT-and-Projection-data](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) — 595 / 1085.6 mm, 736 elements @ 1.2858 mm
* [NICE MIB54 — SOMATOM Definition Edge](https://www.nice.org.uk/advice/mib54/resources/somatom-definition-edge-ct-scanner-for-imaging-coronary-artery-disease-in-adults-in-whom-imaging-is-difficult-pdf-63499226527429) — "736 detectors per row"
* Petersilka M, Bruder H, Krauss B, Stierstorfer K, Flohr TG. [Technical principles of dual source CT](https://www.sciencedirect.com/science/article/abs/pii/S0720048X08004713). Eur J Radiol 2008;68:362–8.
* Flohr TG et al. [Dual-source spiral CT with pitch up to 3.2 and 75 ms temporal resolution](https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1118/1.3259739). Med Phys 2009;36:5641–53.
* Primak AN, Giraldo JCR, Eusemann CD, et al. [Dual-Source Dual-Energy CT With Additional Tin Filtration](https://www.ajronline.org/doi/10.2214/AJR.09.3956). AJR 2010 — 0.4 mm Sn.
* [Dual-Energy CT: General Principles](https://www.ajronline.org/doi/10.2214/AJR.12.9116). AJR 2012.
* Schardt P et al. New x-ray tube performance in CT by introducing the rotating envelope tube technology. Med Phys 2004;31:2699 — STRATON.
* Wang et al., [Physics-Based Iterative Reconstruction for Dual Source and Flying Focal Spot CT](https://arxiv.org/pdf/2001.09471) — Force geometry cross-check (595 / 1085.6 / 920 ch / 0.054°).
* [Siemens Spectral Imaging with Dual Energy CT white paper](https://cdn0.scrvt.com/39b415fb07de4d9656c7b516d8e2d907/1800000005822487/5c82b2d4571a/CT_Dual-Energy_White_Paper_Spectral-Imaging-with-Dual-Energy-CT_HOOD05162002840252_152102617_1800000005822487.pdf)
