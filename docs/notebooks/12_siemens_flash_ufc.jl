### A Pluto.jl notebook ###
# v0.4.2

using Markdown
using InteractiveUtils

# ╔═╡ 12000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 12000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 12000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 12000001-0000-4000-8000-000000000010
md"""
# 12 · Siemens SOMATOM Definition Flash · Dual-Source Regular + VMI

The **second-generation dual-source** scanner (2008, FDA K082220; Stellar
detector K113342) as a complete simulated system: **two STRATON tubes + two
UFC detectors at 95°**, its published geometry and filtration, and the
**Flash-specific Monte-Carlo detector LUT**, exercised through *both* of the
scanner's signature acquisition classes in one notebook. The scanner, its
physics switches, the dual-energy protocol and the VMI chain are the Flash of
the basis-spectral-denoising paper, carried over unchanged.

```
Flash UFC MC η(E) LUT  (BS.UFC_FLASH_MC_EFFICIENCY_LUT, 1–140 keV)
        │ via EICTScanner(detector_material = :ufc_flash)
        ▼
┌─ REGULAR (dual power): 120 kVp on BOTH tubes ──────────────────────────┐
│  one projection, two independent noise draws → per-tube η-aware BHC    │
│  → FDK → HU → combined image: water ≈ 0 HU ×3, σ ≈ σ_single/√2    (§6) │
└────────────────────────────────────────────────────────────────────────┘
┌─ DUAL ENERGY: 80 kVp (A) / Sn140 kVp (B, 0.4 mm Sn) ───────────────────┐
│  each tube projected once → noise-free, measured, calibration draws    │
│  POLY: per-tube η-aware BHC → FDK → HU → mixed image M_w          (§8) │
│  VMI:  BS.vmi_pipeline(; denoiser = SpectralHYPR(),                    │
│          fbp_filter = the fitted Flash PairFilter,                     │
│          pair_basis, composite_energy from the calibration draw)       │
│        → VMI 50/70/100/140 keV, scored against the noise-free draw     │
│        → per-rod regression                                 (§9–§11)   │
└────────────────────────────────────────────────────────────────────────┘
        ▼
   Automated PASS/FAIL verification (water HU, √2 noise, VMI noise and
   bias against the noise-free reference, the chain, per-rod regression)
```

!!! danger "The Flash is NOT the Force — measured, not assumed"
    The Flash uses the same UFC Gd₂O₂S:Pr,Ce *material* as the Force, but
    Hamid Khodajou-Chokami's dedicated Flash MC (2026-08-26) shows a
    **distinctly thinner crystal**: η identical below 30 keV, −4.6% at
    50 keV, −8% at 100 keV, **−28% at 140 keV (0.588 vs 0.816)**.  This
    notebook therefore runs `detector_material = :ufc_flash`
    (`BS.UFC_FLASH_MC_EFFICIENCY_LUT`), never `:ufc`.  §1 plots the
    difference; `test/detector.jl` forbids aliasing the two LUTs.

!!! note "Spec provenance"
    Every published number below comes from the sourced
    [SOMATOM Definition Flash entry of the scanners page](../../scanners/#somatom-definition-flash)
    (FDA 510(k)s, Siemens Dec-2010 datasheet, AAPM LDCT-PD projection
    geometry, Primak AJR 2010 for the 0.4 mm Sn Selective Photon Shield).  Unpublished
    items are declared as documented modeling assumptions in §3.
"""

# ╔═╡ 12000001-0000-4000-8000-000000000020
md"""
## Notebook Setup

Activate the shared docs environment, load the simulator and plotting stack,
detect the available compute backend, and build the notebook table of contents.
"""

# ╔═╡ 12000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 12000001-0000-4000-8000-000000000031
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 12000001-0000-4000-8000-000000000032
import PlutoUI

# ╔═╡ 12000001-0000-4000-8000-000000000033
PlutoUI.TableOfContents()

# ╔═╡ 12000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 12000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 12000002-0000-4000-8000-000000000001
md"""
## 1. The Flash UFC Monte-Carlo Efficiency LUT

Per-energy absorbed fraction η(E) for the **Definition Flash** UFC Gd₂O₂S
scintillator, from a full Monte-Carlo transport simulation of the Flash
detector.

**Provenance**: a Monte-Carlo transport simulation of the Definition Flash
detector by Hamidreza Khodajou-Chokami, PhD (UC Irvine Medical Imaging
Laboratory), `flash_efficiency_results.csv`, 2026-08-26.  The table ships in
the package as `BS.UFC_FLASH_MC_EFFICIENCY_LUT`
(`src/detector/detector_efficiency.jl`): 140 values on a 1-keV grid
(1–140 keV), verbatim from that dataset, next to the Force
`BS.UFC_MC_EFFICIENCY_LUT`.

**Physics signatures** (MC-only features Beer-Lambert *cannot* model):

1. **Gd K-edge fluorescence escape at 50.24 keV**: η drops
   `0.925 → 0.739` between 50 and 51 keV — Gd Kα fluorescence (~43 keV)
   escapes the crystal, so the *deposited* fraction falls even though
   attenuation rises.
2. **Gd L-edge structure near 7–8 keV** (L₃ 7.24 / L₂ 7.93 / L₁ 8.38 keV).
3. **Steep high-energy roll-off** (0.825 at 100 keV → **0.588 at 140 keV**)
   — the Flash-vs-Force fingerprint.  The thinner Flash crystal transmits
   more high-energy primaries; below ~30 keV, where everything is absorbed
   near the entrance surface, the two scanners are indistinguishable.
"""

# ╔═╡ 12000002-0000-4000-8000-000000000030
let
    Es = collect(1.0:0.25:140.0)
    η_flash = BS.get_ufc_flash_mc_efficiency.(Es)
    η_force = BS.get_ufc_mc_efficiency.(Es)
    η_gem = BS.get_gemstone_mc_efficiency.(Es)

    fig = Mke.Figure(size = (1180, 580))
    ax = Mke.Axis(
        fig[1, 1];
        title = "MC Detector Efficiency",
        subtitle = "Flash UFC vs Force UFC vs Gemstone — same Gd₂O₂S, thinner Flash crystal",
        xlabel = "Photon Energy (keV)",
        ylabel = "Absorbed Fraction η(E)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.lines!(ax, Es, η_flash; color = :crimson, linewidth = 3, label = "Flash UFC (BS.UFC_FLASH_MC_EFFICIENCY_LUT)")
    Mke.lines!(ax, Es, η_force; color = :gray35, linewidth = 2.5, linestyle = :dash, label = "Force UFC (notebook 09)")
    Mke.lines!(ax, Es, η_gem; color = :steelblue, linewidth = 2.5, label = "Gemstone garnet (GE Apex)")

    Mke.vlines!(ax, [50.24]; color = :crimson, linestyle = :dash, linewidth = 1.5)
    Mke.text!(ax, 50.24, 0.62; text = "Gd K-edge\n50.2 keV", fontsize = 16, align = (:left, :top), offset = (4, 0))
    # the 140 keV values, marked on both curves and labelled in the free space below them
    Mke.scatter!(ax, [140.0, 140.0], [η_flash[end], η_force[end]];
        color = [:crimson, :gray35], markersize = 12)
    Mke.text!(
        ax, 104.0, 0.52;
        text = "140 keV: Flash $(round(η_flash[end]; digits = 3)), Force $(round(η_force[end]; digits = 3))\n($(round(Int, 100 * (η_flash[end] / η_force[end] - 1)))% for the Flash)",
        fontsize = 16, align = (:left, :bottom), color = :crimson,
    )

    Mke.ylims!(ax, 0.5, 1.02)
    Mke.axislegend(ax; position = :rt, framevisible = true, labelsize = 18)
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_lut_comparison.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000003-0000-4000-8000-000000000001
md"""
## 2. `Phantom`: Gammex Model 472
"""

# ╔═╡ 12000003-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 12000003-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 12000004-0000-4000-8000-000000000001
md"""
## 3. `EICTScanner`: Siemens SOMATOM Definition Flash

Second-generation dual-source: **two STRATON MX P tubes + two UFC detectors
at 95°** in the same gantry.  Every value below is sourced in the
[SOMATOM Definition Flash entry of the scanners page](../../scanners/#somatom-definition-flash):

| Parameter | Value | Source |
|-----------|-------|--------|
| Source–isocenter (SID) | 595.0 mm | AAPM LDCT-PD / DICOM-CT-PD (acquired on a Flash) |
| Source–detector (SDD)  | 1085.6 mm | LDCT-PD |
| Tube A–B angular offset | 95° | Petersilka 2008; technical reviews |
| Detector A channels | 736 (50 cm FOV) | Siemens datasheet 47,104 el / 64 rows; NICE MIB54; LDCT-PD |
| Detector B channels | 480 (33 cm FOV) | Siemens datasheet 30,720 el / 64 rows |
| Rows | 64 × 0.6 mm at iso (38.4 mm z-coverage) | Siemens datasheet; FDA K173630 |
| Column pitch at iso | 0.70473 mm (1.2858 mm at detector / 1.8245× mag) | LDCT-PD |
| Scintillator | UFC Gd₂O₂S:Pr,Ce → **`:ufc_flash` MC LUT** | Siemens datasheet + Khodajou MC |
| Tube (STRATON MX P) | 2 × 100 kW, 80/100/120/140 kV, ≤800 mA each | Siemens datasheet |
| Focal spots (IEC 60336) | 0.7 × 0.7 mm and 0.9 × 1.1 mm, **anode 7°** | Siemens datasheet |
| Flat filtration | **8.4 mm Al equivalent** (6.8 tube + 1.6 BLD) | Siemens datasheet |
| Tin filter | **0.4 mm Sn** on tube B (Selective Photon Shield, DE mode) | Primak AJR 2010 |
| Rotation | 0.28 s min (0.5 s routine) | Siemens datasheet |
| Projections | 1152 / rotation per focal-spot position (2304 with z-FFS) | Siemens datasheet; AAPM LDCT |

`flash_scanner` below is the Flash of the basis-spectral-denoising paper
(basis-vmi's `flash_scanner`), verbatim.

!!! warning "Documented modeling assumptions (no public source exists)"
    - **Anode angle in the spectrum model**: the Flash's anode is a
      *published* 7°, but the bundled IPEM spectra come only in 8°/10° →
      protocols use the closest available **8°** (`anode_angle = 8`);
      the `EICTScanner`'s `target_angle` keeps the true 7.0° (heel-effect metadata,
      inert with `use_heel_effect = false`).
    - **Bowtie**: Siemens form-filter shape is unpublished → CatSim
      **large-body** profile as stand-in (same convention as notebooks 04, 08 and 09).
    - **Flat-filter composition**: only the **Al equivalent** is published
      (8.4 mm) → modeled as 8.4 mm of aluminum.  Unlike notebook 09's Force
      (assumed Al+Ti stack), this figure is a *measured* datasheet value.
    - **Scintillator thickness 1.0 mm / fill factor 0.9**: proprietary;
      thickness is inert here (η comes from the Flash MC LUT; the MC
      high-energy roll-off implies a Beer-Lambert-effective ≈1.07 mm),
      fill factor auto-cancels in the air-scan calibration.
    - **Electronic noise = 1500 e⁻** (≈2.1 photon-equivalents at 70 keV
      with the 10 e⁻/keV gain): modeling the Stellar (K113342) build.
      Siemens publishes no absolute figure for either build, but a hard
      zero is *wrong* — the DAS floor enters the counts before the log
      transform and dominates in photon starvation (low dose, 80 kVp,
      dense anatomy), so `simulate!` must be allowed to propagate it
      (see `add_system_noise_floor!` in `src/api/driver.jl`).
      The value is ~40% of the conventional-DAS class figure
      (≈3500 e⁻ at this gain — same class as the repo's GE Apex
      5000 e⁻ @ 15 e⁻/keV), reflecting TrueSignal's integrated-ASIC
      reduction.  For the **pre-2011 original Flash** (K082220, discrete
      photodiode/ASIC) set `electronic_noise = 3500`.  At this
      notebook's well-exposed protocols the image-σ contribution is
      ≲1%; the term matters when this scanner model is reused at low
      dose.
    - **Tube-B z-offset**: unpublished for the Flash (Force ≈ 0.88 mm) →
      not modeled.  The Gammex is z-invariant, so an offset would change
      nothing here; notebook 09 carries the mechanism for the Force.
    - **z-FFS not modeled** → 1152 views/rotation (one focal-spot position,
      half of the 2304 FFS-interleaved readings).

!!! info "Dual source → two co-registered scans (the accepted hack)"
    Exactly like notebook 03 models rapid-kVp switching and notebook 09 models the Force,
    the Flash's two tubes are modeled as **two `CTProtocol`s run
    back-to-back on one `EICTScanner`** — but with **two unique tube sources**: each
    tube gets its own protocol (kVp, mA, filtration) and its own
    **independent noise seed** (two physically separate tube/detector
    chains must not share a noise realization — critical for the §6
    dual-power √2 check).

    - **Detector arc**: tube B gets detector-A's 736-channel arc so the
      (low, high) sinogram pair is per-ray co-registered for the
      n-channel estimator — defensible because the 33 cm Gammex body fits inside
      detector B's real 33.4 cm FOV (tighter than the Force's 35.5 cm —
      the Flash B-fan is the actual clinical DE-FOV limit).
    - **95° in-plane tube offset**: both modeled scans run a full axial
      rotation on the same angle grid — a constant angular offset has no
      effect on a full-rotation axial scan (notebook 09 §3 argument).
    - **DE-mode collimation**: the Flash reads 2 × 128 × 0.6 mm in DE mode;
      we use 4.8 mm (8 × 0.6 mm), the collimation of the
      basis-spectral-denoising acquisition — the thin-collimation equivalent
      that fits the 1 cm Gammex z-extent.
"""

# ╔═╡ 12000004-0000-4000-8000-000000000010
# Siemens SOMATOM Definition Flash: basis-spectral-denoising's `flash_scanner`, verbatim. Both
# tubes run on detector A's arc so the two channels stay co-registered ray by ray (see md above).
flash_scanner = BS.EICTScanner(
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
    flat_filter_thickness = 8.4,          # 6.8 mm Al tube assembly + 1.6 mm beam limiter
    bowtie_filter = :large_body,
    detector_material = :ufc_flash,       # the Flash's own Monte Carlo table (NOT :ufc)
    detector_depth = 1.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 1500,              # e⁻ — Stellar DAS floor; pre-2011 build ≈ 3500 (§3)
    detection_gain = 10.0,
);

# ╔═╡ 12000006-0000-4000-8000-000000000001
md"""
## 4. Physics Switches: `SimOptions`

`flash_opts` are the Flash's physics switches in the basis-spectral-denoising
paper, verbatim: every effect of the EICT forward model on, the heel effect
off as there, and **view integration**.

- **Detector efficiency.** `use_detector_efficiency = true` (the default)
  routes through the Flash UFC MC LUT: `build_physics_config` sees
  `EICTScanner(detector_material = :ufc_flash)` and dispatches to
  `detector_efficiency_ufc_flash()`, so every energy is weighted by
  `w(E) · η_Flash(E)` and the detected flux, and therefore the Poisson
  noise, reflects the Flash absorption.
- **Heel effect off** (`use_heel_effect = false`) keeps the forward spectral
  model exactly equal to the η-folded response the spectral basis inverts
  (with 4.8 mm collimation the heel is a small row-direction effect).
- **View integration** (`view_samples = 5`). A detector integrates while
  the gantry turns, so each view reads the transmission averaged over the
  arc it sweeps, `r · Δθ` at radius `r`: a blur of the object, largest far
  from the isocentre, that the noise, counted once per view, does not
  share.  Five sub-views sample that arc (0.2 mm apart at the edge of a
  35 cm field, a third of a pixel).  Without it the simulated signal is
  sharper, relative to its noise, than a physical scanner's.  Each Flash
  tube reads its own detector for the whole view period, so both integrate
  over the full view arc (`view_arc = 1.0`); GE's rapid kVp switching, by
  contrast, reads each energy for its duty cycle of the view only.

`with_option` replaces one field (a seed, `use_noise`, `view_arc`) and keeps
every other switch.
"""

# ╔═╡ 12000006-0000-4000-8000-000000000010
begin
    VIEW_SAMPLES = 5
    flash_opts = BS.SimOptions(projector = :dd_fast, use_heel_effect = false, view_samples = VIEW_SAMPLES)
    "The same physics switches with one field replaced."
    with_option(opts, field, value) = BS.SimOptions(;
        (f => getfield(opts, f) for f in fieldnames(BS.SimOptions) if f !== field)...,
        (field => value,)...,
    )
end;

# ╔═╡ 12000005-0000-4000-8000-000000000001
md"""
## 5. Protocols, Dose and Grid

**Dual energy** is the protocol of the physical Flash scan of the Gammex that
the basis-spectral-denoising paper compares its simulations with:
**80 kVp / Sn140 kVp** at 470 / 182 mA, the 0.4 mm Sn Selective Photon
Shield on tube B (Primak AJR 2010), 1152 views in 0.5 s, 4.8 mm collimation,
reconstructed as 4 × 1.0 mm slices over 33 cm (its clinical series: DE
monoenergetic, D40f).  `FLASH` holds every value, as `SCANNERS.flash` does
there.  The bundled IPEM spectra top out at 140 kVp, exactly the Flash's top
kV, so no kV substitution is needed.

**Dose matching (`DOSE_SCALE`).**  A physical tube's output per mA is its
own and the simulator's is generic, so the tube currents alone do not fix
the dose.  Both tube currents are scaled by one factor, which keeps the
physical 470 : 182 split, so that the simulated CTDIvol (the package's Monte
Carlo dose, 32 cm body phantom, summed over both tubes) equals the value the
physical scan reported.

**Regular (dual power)** is the Flash's bread-and-butter dual-source mode:
both tubes at the **same** kVp, doubling the available power.  Routine
abdomen 120 kV at 420 mA per tube (210 quality-reference mAs at 0.5 s),
scaled by the same `DOSE_SCALE`, which calibrates the simulated tube whatever
protocol it runs.

!!! note "Why the views are NOT halved per tube"
    Each tube/detector system has its **own DAS sampling a full
    rotation** — the datasheet quotes "up to 4,608 projections per 360°
    *per data-acquisition unit*."  In DE mode tube A acquires a complete
    1152-view rotation at 80 kV **and** tube B acquires a complete
    1152-view rotation at Sn140: full angular sampling per energy is the
    defining advantage of dual-source DE over rapid-kVp *switching*,
    where a single tube alternates kV between views.  The 2,304
    readings/rotation figure is the z-FFS focal-spot doubling (not
    modeled → 1152), and cardiac quarter-rotation segments are a recon
    mode, not an acquisition split.
"""

# ╔═╡ 12000005-0000-4000-8000-000000000005
# The dual-energy acquisition: basis-spectral-denoising's `SCANNERS.flash`, verbatim. Seeds are per
# exposure: a shared seed would correlate the two channels' noise.
FLASH = (name = "Siemens SOMATOM Flash", mechanism = "dual source", kind = :eict,
    scanner = flash_scanner, opts = flash_opts, views = 1152, rotation_time = 0.5,
    collimation_mm = 4.8, slice_mm = 1.0, n_slices = 4, fov_cm = 33.0, anode_angle = 8,
    clinical_recon = "DE monoenergetic, D40f, 1.0 mm, 330 mm", ctdi_vol_mGy = 10.01,
    exposures = [
        (label = "80 kVp", kVp = 80, mA = 470.0, view_arc = 1.0, seed = 1234, filters = Tuple{String, Float64}[]),
        (label = "Sn140 kVp", kVp = 140, mA = 182.0, view_arc = 1.0, seed = 4321, filters = [("Sn", 0.4)]),
    ]);

# ╔═╡ 12000005-0000-4000-8000-000000000010
begin
    "The CTProtocol of one exposure of scanner `s`, its tube current scaled by `mA_scale` (`DOSE_SCALE`)."
    exposure_protocol(s, e; mA_scale = 1.0) = BS.CTProtocol(;
        kVp = e.kVp, mA = e.mA * mA_scale, views = s.views, rotation_time = s.rotation_time,
        collimation_mm = s.collimation_mm, additional_filters = e.filters,
        (s.anode_angle === nothing ? (;) : (; anode_angle = s.anode_angle))...,
    )
    "Reconstruction grid of scanner `s`: its clinical series' slices, field of view and 512²."
    recon_options(s) = BS.ReconOptions(
        matrix_size = (512, 512, s.n_slices),
        fov_cm = s.fov_cm, z_cm = s.n_slices * s.slice_mm / 10,
    )
end;

# ╔═╡ 12000005-0000-4000-8000-000000000020
# One factor for both tubes: the simulated CTDIvol (32 cm body phantom, summed over the exposures)
# equals the physical scan's.
DOSE_SCALE = FLASH.ctdi_vol_mGy / sum(FLASH.exposures) do e
    p = exposure_protocol(FLASH, e)
    BS.compute_dose(BS.dose_source(FLASH.scanner, p), p).ctdi_vol_mGy
end

# ╔═╡ 12000005-0000-4000-8000-000000000030
begin
    # dual energy: tube A (low) and tube B (high), at the physical scan's dose
    protocol_low, protocol_high = (exposure_protocol(FLASH, e; mA_scale = DOSE_SCALE) for e in FLASH.exposures)
    # regular (dual power): 120 kVp, 420 mA on each tube, on the same simulated tube
    protocol_reg = BS.CTProtocol(
        kVp = 120, mA = 420.0 * DOSE_SCALE, views = FLASH.views, rotation_time = FLASH.rotation_time,
        collimation_mm = FLASH.collimation_mm, anode_angle = FLASH.anode_angle,
    )
end;

# ╔═╡ 12000005-0000-4000-8000-000000000040
let
    filt(p) = isempty(p.additional_filters) ? "none" :
        join(["$(f[2]) mm $(f[1])" for f in p.additional_filters], ", ")
    rows = ["| $(m) | $(t) | $(p.kVp) | $(filt(p)) | $(round(Int, p.mA / DOSE_SCALE)) | $(round(p.mA, digits = 1)) |"
            for (m, t, p) in (("Regular", "A and B", protocol_reg), ("Dual energy", "A (low)", protocol_low),
                              ("Dual energy", "B (high)", protocol_high))]
    Markdown.parse("""
    `DOSE_SCALE` = $(round(DOSE_SCALE, digits = 3)): at the scaled currents the dual-energy scan's
    simulated CTDIvol is the physical scan's $(FLASH.ctdi_vol_mGy) mGy.

    | Acquisition | Tube | kVp | Added filter | Physical mA | Simulated mA |
    |---|---|---:|---|---:|---:|
    $(join(rows, "\n"))

    Every exposure: $(FLASH.views) views in $(FLASH.rotation_time) s, $(FLASH.collimation_mm) mm
    collimation, the 8.4 mm Al flat filter and the large-body bowtie.
    """)
end

# ╔═╡ 12000006-0000-4000-8000-000000000020
# The clinical series' grid (`recon_options`): 4 × 1.0 mm slices over 33 cm, 512².
recon_opts = recon_options(FLASH);

# ╔═╡ 12000006-0000-4000-8000-000000000025
# The Gammex's labels on the reconstruction grid (the phantom itself is voxelized over 35 cm):
# every ROI below is drawn on this map.
roi_labels = BS.create_gammex_472(
    n_voxels = recon_opts.matrix_size[1], n_slices = 1, fov_cm = recon_opts.fov_cm, z_cm = 0.1,
).mask[:, :, 1];

# ╔═╡ 12000006-0000-4000-8000-000000000030
"""
    flash_detected_spectrum(protocol) -> (e, w_eta)

Display helper: tube spectrum × flat filtration × protocol filters
(IPEM, absolute flux) with the src Flash UFC MC η(E) folded in — what the
detector actually integrates (centered ray, no bowtie).
"""
function flash_detected_spectrum(protocol)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        flash_opts, protocol; scanner = flash_scanner,
    )
    return e, Float64.(w) .* BS.get_ufc_flash_mc_efficiency.(e)
end;

# ╔═╡ 12000006-0000-4000-8000-000000000040
let
    specs = (
        ("120 kVp · both tubes (regular)", protocol_reg, :seagreen),
        ("80 kVp · tube A (DE)", protocol_low, :royalblue),
        ("Sn140 kVp · tube B (DE)", protocol_high, :crimson),
    )

    fig = Mke.Figure(size = (1180, 580))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Flash-Detected Spectra",
        subtitle = "w(E) · η_Flash(E), normalized — 0.4 mm Sn hardening on tube B",
        xlabel = "Energy (keV)",
        ylabel = "Relative Detected Fluence",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )

    for (label, prot, color) in specs
        e, wη = flash_detected_spectrum(prot)
        wn = wη ./ sum(wη)
        mean_E = sum(e .* wn)
        Mke.lines!(
            ax, Float64.(e), wn ./ maximum(wn);
            color = color, linewidth = 3,
            label = "$(label)  (mean $(round(mean_E, digits = 1)) keV)",
        )
    end
    Mke.axislegend(ax; position = :rt, framevisible = true, labelsize = 18)
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_detected_spectra.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000007-0000-4000-8000-000000000001
md"""
## 6. Dual-Source Regular Scan (Dual Power, 120/120 kVp)

The Flash's routine dual-source mode: both tubes at the **same** kVp.
Clinically this buys temporal resolution (cardiac) or power reserve
(dual power for large patients); the image-formation consequence we can
verify quantitatively is that averaging the two **independent** tube
acquisitions leaves accuracy untouched and cuts noise by **√2**:

- water ≈ 0 HU in the tube-A recon, the tube-B recon, *and* the combined
  image (the η-aware BHC is right on both chains), and
- `σ_combined ≈ σ_single / √2` (the two noise realizations are
  independent — this is exactly what tube B's own seed buys).

The two tubes run one protocol through one geometry, so their noise-free
expectation is a single projection: tube A keeps it (`keep_projection = true`)
and tube B passes it back (`projection`), drawing only its own noise from its
own seed — bit-identical to projecting again.  Each tube then runs the
standard correction stack: parameter-free η-aware water sinogram BHC → FDK →
HU.
"""

# ╔═╡ 12000007-0000-4000-8000-000000000005
"""
    flash_bhc_calibration(protocol, geom)

η-aware per-tube BHC: bowtie-hardened per-column spectrum × Flash UFC η(E)
→ per-column two-material polynomial.  Returns `(model, μ_water, ref_E_keV)`.
"""
function flash_bhc_calibration(protocol, geom)
    # resolve_source_spectrum_full folds bowtie AND the src Flash UFC η(E)
    # (via the same build_physics_config the forward model used).
    e, ŵ = BS.resolve_source_spectrum_full(
        flash_opts, protocol; scanner = flash_scanner, geom = geom,
    )
    e2, w_col = BS.bhc_spectrum_per_column(e, ŵ)          # [n_E, n_col]

    # Single mono-equivalent target = mean energy of the η-folded mean spectrum
    w_mean = vec(sum(w_col; dims = 2)) ./ size(w_col, 2)
    ref_E = sum(e2 .* w_mean) / sum(w_mean)

    # Parameter-free water BHC from the per-column Flash-η spectrum.
    model = BS.calibrate_bhc_water(
        e2, w_col;
        reference_energy_keV = ref_E,
    )
    return (model = model, μ_water = model.μ_water_ref, ref_E_keV = model.reference_energy_keV)
end;

# ╔═╡ 12000007-0000-4000-8000-000000000008
"""
    flash_poly_recon(sino_cpu, geom, bhc) -> Array{Float32, 3}

Doctrine correction stack for one tube: knobless water sino-BHC → FDK → HU.
"""
function flash_poly_recon(sino_cpu, geom, bhc)
    matrix_size = recon_opts.matrix_size

    sino_gpu = to_gpu(sino_cpu)
    # Knobless water BHC: one sinogram-domain pass, no recon round-trip.
    sino_bhc = BS.apply_bhc_water(sino_gpu, bhc.model)
    sino_gpu = sino_bhc

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom)

    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water))

    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing
    GC.gc(true)
    return hu
end;

# ╔═╡ 12000007-0000-4000-8000-000000000010
sim_reg = let projection = nothing, out = Dict{Symbol, Any}()
    for (tube, seed) in ((:a, 1234), (:b, 4321))
        opts = with_option(flash_opts, :seed, seed)
        ws = BS.create_workspace(flash_scanner, protocol_reg, opts, recon_opts, phantom)
        try
            t = time()
            result = BS.simulate!(ws, phantom, protocol_reg, opts; report_dose = false,
                keep_projection = projection === nothing, projection)
            projection === nothing && (projection = result.projection)   # tube B reuses tube A's
            out[tube] = (sino = Array(ws.sinogram), geom = ws.geom, seconds = time() - t)
        finally
            BS.release_backend!(ws)
        end
    end
    projection = nothing; GC.gc(true)
    (a = out[:a], b = out[:b])
end;

# ╔═╡ 12000007-0000-4000-8000-000000000015
bhc_reg = flash_bhc_calibration(protocol_reg, sim_reg.a.geom);

# ╔═╡ 12000007-0000-4000-8000-000000000020
hu_reg = (
    a = flash_poly_recon(sim_reg.a.sino, sim_reg.a.geom, bhc_reg),
    b = flash_poly_recon(sim_reg.b.sino, sim_reg.b.geom, bhc_reg),
);

# ╔═╡ 12000007-0000-4000-8000-000000000025
# Dual power: the two independent same-kV acquisitions average into the
# combined image (image domain — per-ray sinogram averaging is equivalent
# to first order on the log data at these noise levels).
hu_reg_combined = 0.5f0 .* hu_reg.a .+ 0.5f0 .* hu_reg.b;

# ╔═╡ 12000007-0000-4000-8000-000000000030
dp_stats = let
    ERODE_PX = 12.0
    sw_bool = BS.erode_mask_2d(
        roi_labels .== UInt8(BS.REGION_SOLID_WATER); erode_px = ERODE_PX,
    )
    sw_idx = findall(sw_bool)

    # Central noise ROI (12 px ≈ 7.7 mm at 0.645 mm/px) — background water.
    nx_r, ny_r, _ = size(hu_reg_combined)
    cx = nx_r ÷ 2 + 1; cy = ny_r ÷ 2 + 1
    noise_bool = falses(nx_r, ny_r)
    r² = 12.0^2
    @inbounds for j in 1:ny_r, i in 1:nx_r
        ((i - cx)^2 + (j - cy)^2) ≤ r² && (noise_bool[i, j] = true)
    end
    noise_idx = findall(noise_bool)
    n_z = size(hu_reg_combined, 3)

    function _roi(vol, idx)
        vals = Float64[Float64(vol[ci, z]) for z in 1:n_z, ci in idx]
        (mean = mean(vals), std = std(vals), n = length(vals))
    end

    water = (
        a = _roi(hu_reg.a, sw_idx),
        b = _roi(hu_reg.b, sw_idx),
        combined = _roi(hu_reg_combined, sw_idx),
    )
    noise = (
        a = _roi(hu_reg.a, noise_idx),
        b = _roi(hu_reg.b, noise_idx),
        combined = _roi(hu_reg_combined, noise_idx),
    )
    σ_single = 0.5 * (noise.a.std + noise.b.std)
    ratio = noise.combined.std / σ_single

    for (tag, s) in pairs(water)
        @info "[regular 120/120 · Flash] $(tag) SW ROI: ⟨HU⟩ = $(round(s.mean, digits = 2)), σ = $(round(s.std, digits = 2)) HU"
    end
    @info "[regular 120/120 · Flash] noise σ: A = $(round(noise.a.std, digits = 2)), " *
        "B = $(round(noise.b.std, digits = 2)), combined = $(round(noise.combined.std, digits = 2)) HU " *
        "→ ratio $(round(ratio, digits = 3)) (ideal 1/√2 = 0.707)"

    (
        water = water, noise = noise,
        noise_ratio = ratio,
        sw_mask_2d = collect(sw_bool), noise_mask_2d = noise_bool,
    )
end;

# ╔═╡ 12000007-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)
    mid = size(hu_reg_combined, 3) ÷ 2

    fig = Mke.Figure(size = (1400, 1000))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    panels = (
        (1, "Tube A (120 kVp)", hu_reg.a),
        (2, "Tube B (120 kVp)", hu_reg.b),
        (3, "Dual Power (A+B)/2", hu_reg_combined),
    )
    for (c, ttl, vol) in panels
        ax = Mke.Axis(
            fig[1, c]; title = ttl,
            aspect = Mke.DataAspect(), axis_kwargs...,
        )
        Mke.heatmap!(ax, vol[:, :, mid]; colormap = :grays, colorrange = HU_window)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1, 4]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    entries = (
        ("Tube A", dp_stats.noise.a.std),
        ("Tube B", dp_stats.noise.b.std),
        ("Combined", dp_stats.noise.combined.std),
    )
    n = length(entries)
    σs = [e[2] for e in entries]
    bar_colors = [Mke.cgrad(:plasma, n; categorical = true)[i] for i in 1:n]
    ax2 = Mke.Axis(
        fig[2, 1:3];
        title = "Dual-Power Noise",
        subtitle = "Central water ROI σ — combined ≈ single/√2 (independent tubes)",
        xlabel = "Reconstruction", ylabel = "σ (HU)",
        xticks = (collect(1:n), [e[1] for e in entries]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.barplot!(ax2, 1:n, σs; color = bar_colors, strokecolor = :black, strokewidth = 1)
    σ_pred = 0.5 * (σs[1] + σs[2]) / sqrt(2.0)
    Mke.hlines!(ax2, [σ_pred]; color = :black, linewidth = 2, linestyle = :dash,
        label = "predicted single/√2 = $(round(σ_pred, digits = 2)) HU")
    for (k, σv) in enumerate(σs)
        Mke.text!(
            ax2, k, σv;
            text = "σ = $(round(σv, digits = 2)) HU",
            align = (:center, :bottom), fontsize = 18, offset = (0, 6),
        )
    end
    Mke.ylims!(ax2, 0, 1.35 * maximum(σs))   # headroom for the bar labels and the legend
    Mke.axislegend(ax2; position = :rt, framevisible = true, labelsize = 16)
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.45))

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_dual_power.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000008-0000-4000-8000-000000000001
md"""
## 7. DE Acquisition: One Projection per Tube, Three Draws

`acquire` simulates the dual-energy acquisition the way basis-spectral-denoising
does.  Each tube builds its workspace (`BS.create_workspace`; the Flash UFC η
enters through the `detector_efficiency` pathway) and is **projected once**:
everything before the noise is the same for every draw, so the first draw
keeps it (`keep_projection = true`) and every further draw passes it back
(`projection`) and only draws its noise, bit-identically to simulating
afresh.  Three draws share the projection:

- draw **0**, `use_noise = false`: the **noise-free reference**, the
  expectation every noisy result below is scored against;
- draw **1**: the **measured** acquisition every result comes from;
- draw **9**: the **calibration** draw, which only fixes the
  reconstruction's pair (§10); nothing is measured on it.

Each draw keeps the two corrected log sinograms.  The first also keeps what
the spectral basis needs: the per-ray air counts `I0_ray` (detector air count
× bowtie air profile) and the per-ray detected spectrum from
`resolve_source_spectrum_full` (source × filtration × bowtie × Flash η), the
model the forward projector applied.  The basis does not depend on the
noise, so every draw shares it.
"""

# ╔═╡ 12000006-0000-4000-8000-000000000012
begin
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each exposure the seed
    # `seed + 1000 (r − 1)`, so the two tubes stay independent within a draw, and draws of each other.
    DE_DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    opts_for(e, r) = with_option(with_option(with_option(FLASH.opts, :seed, realization_seed(e, r)),
        :use_noise, r > 0), :view_arc, e.view_arc)
end;

# ╔═╡ 12000008-0000-4000-8000-000000000005
"""
    acquire(draws) -> Dict{Int, NamedTuple}

The dual-energy acquisition of the Gammex (both tubes of `FLASH`) for every draw in `draws`: draw 0
the noise-free expectation, draw `r ≥ 1` a noisy realisation with the seeds `realization_seed(e, r)`.
Each exposure is projected once, by the first draw, and every other draw re-uses that projection
(`keep_projection`, `projection`), only its noise drawn. Each entry holds the two log-transmission
channels, the spectral basis the decomposition inverts (shared by every draw), the geometry and the
wall time of the draw per tube.
"""
function acquire(draws)
    parts = map(FLASH.exposures) do e
        protocol = exposure_protocol(FLASH, e; mA_scale = DOSE_SCALE)
        projection = nothing
        channels = Dict{Int, Array{Float32, 3}}()
        seconds = Dict{Int, Float64}()
        model = nothing
        for r in draws
            opts = opts_for(e, r)
            t = time()
            ws = BS.create_workspace(FLASH.scanner, protocol, opts, recon_opts, phantom)
            try
                result = BS.simulate!(ws, phantom, protocol, opts; report_dose = false,
                    keep_projection = projection === nothing, projection)
                projection === nothing && (projection = result.projection)
                channels[r] = Array(ws.sinogram)
                if model === nothing
                    air = ws.bowtie_air_reference === nothing ?
                        ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
                        Float32.(Array(ws.bowtie_air_reference))
                    I0 = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights)) * Float64(ws.η_eff)
                    energies, response = BS.resolve_source_spectrum_full(
                        opts, protocol; scanner = FLASH.scanner, geom = ws.geom)
                    model = (geom = ws.geom, I0_ray = Float32.(I0 .* air),
                        energies = Float64.(energies), response = Float32.(response))
                end
            finally
                BS.release_backend!(ws)
            end
            seconds[r] = time() - t
        end
        projection = nothing
        (; channels, seconds, model...)
    end
    GC.gc(true)
    basis = BS.spectral_basis_from_acquisitions(acquisitions = [
        (energies = p.energies, response = p.response, I0_ray = p.I0_ray) for p in parts])
    Dict(r => (draw = r, channels = [p.channels[r] for p in parts], basis = basis,
               geom = first(parts).geom, seconds = [p.seconds[r] for p in parts])
         for r in draws)
end;

# ╔═╡ 12000008-0000-4000-8000-000000000020
de_acq = acquire((DE_DRAWS.reference, DE_DRAWS.measured, DE_DRAWS.calibration));

# ╔═╡ 12000008-0000-4000-8000-000000000030
let
    role = Dict(DE_DRAWS.reference => "noise-free reference", DE_DRAWS.measured => "measured",
        DE_DRAWS.calibration => "calibration")
    seeds(r) = r == 0 ? "none" : join([realization_seed(e, r) for e in FLASH.exposures], ", ")
    secs(v) = join([string(round(s, digits = 1)) for s in v], ", ")
    rows = ["| $(r) | $(role[r]) | $(seeds(r)) | $(secs(de_acq[r].seconds)) |" for r in sort(collect(keys(de_acq)))]
    Markdown.parse("""
    | Draw | Role | Seeds (A, B) | Seconds (A, B) |
    |---:|---|---|---|
    $(join(rows, "\n"))

    Draw 0 projects each tube, $(VIEW_SAMPLES) sub-views per view; draws 1 and 9 re-use that
    projection and only draw their noise. In the regular mode of §6, tube A took
    $(round(sim_reg.a.seconds, digits = 1)) s and tube B, re-using its projection,
    $(round(sim_reg.b.seconds, digits = 1)) s (wall time on this render, compilation included in the first call).
    """)
end

# ╔═╡ 12000008-0000-4000-8000-000000000035
# The measured draw: every DE result below comes from it.
de_measured = de_acq[DE_DRAWS.measured];

# ╔═╡ 12000008-0000-4000-8000-000000000040
let
    n_row = size(de_measured.channels[1], 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(de_measured.channels[1][:, mid_r, :], (2, 1))
    slice_hi = permutedims(de_measured.channels[2][:, mid_r, :], (2, 1))

    all_v = vcat(vec(slice_lo), vec(slice_hi))
    sino_window = (
        Float64(quantile(all_v, 0.01)),
        Float64(quantile(all_v, 0.99)),
    )

    fig = Mke.Figure(size = (1180, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    panels = (
        (1, 1, "80 kVp (tube A)", slice_lo),
        (1, 2, "Sn140 kVp (tube B)", slice_hi),
    )

    for (r, c, ttl, slice) in panels
        ax = Mke.Axis(fig[r, c]; title = ttl, axis_kwargs...)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    Mke.Colorbar(
        fig[1, 3]; colormap = :viridis, colorrange = sino_window,
        label = "Log Line Integral", width = 16, labelsize = 22, ticklabelsize = 18
    )
    fig
end

# ╔═╡ 12000009-0000-4000-8000-000000000001
md"""
## 8. DE Poly Readout: Per-Tube EICT Recon + Siemens Mixed Image

In DE mode the scanner's routine-equivalent grayscale output is the
**mixed image** — a linear image-domain blend of the two per-tube
reconstructions (Yu et al., *Med Phys* 2009: `M = w·I_low + (1−w)·I_high`;
Eusemann et al., SPIE 2008).  This notebook shows the equal blend,
w = 0.5; clinical Flash blends weight the low-kV image between about 0.3
and 0.6.

So the poly validation of the Flash LUT runs the standard correction stack
**per tube** — η-aware water sinogram BHC → FDK → HU — then blends.  If the
η fold is right, solid water lands at ≈ 0 HU in *both* per-tube recons (and
therefore in any blend).
"""

# ╔═╡ 12000009-0000-4000-8000-000000000010
bhc_low = flash_bhc_calibration(protocol_low, de_measured.geom);

# ╔═╡ 12000009-0000-4000-8000-000000000012
bhc_high = flash_bhc_calibration(protocol_high, de_measured.geom);

# ╔═╡ 12000009-0000-4000-8000-000000000015
md"""
**Calibrated (η-aware):**
tube A ref energy = $(round(bhc_low.ref_E_keV, digits = 1)) keV ·
lac water = $(round(bhc_low.μ_water, digits = 5)) cm⁻¹ —
tube B ref energy = $(round(bhc_high.ref_E_keV, digits = 1)) keV ·
lac water = $(round(bhc_high.μ_water, digits = 5)) cm⁻¹ —
regular 120 kVp ref energy = $(round(bhc_reg.ref_E_keV, digits = 1)) keV
"""

# ╔═╡ 12000009-0000-4000-8000-000000000020
hu_tube = (
    low = flash_poly_recon(de_measured.channels[1], de_measured.geom, bhc_low),
    high = flash_poly_recon(de_measured.channels[2], de_measured.geom, bhc_high),
);

# ╔═╡ 12000009-0000-4000-8000-000000000025
# Siemens linear mixed image: M = w·I_low + (1−w)·I_high (image domain, Yu 2009): the equal blend.
MIX_W_LOW = 0.5f0;

# ╔═╡ 12000009-0000-4000-8000-000000000028
hu_mixed = MIX_W_LOW .* hu_tube.low .+ (1.0f0 - MIX_W_LOW) .* hu_tube.high;

# ╔═╡ 12000009-0000-4000-8000-000000000030
poly_water_stats = let
    ERODE_PX = 12.0
    sw_bool = BS.erode_mask_2d(
        roi_labels .== UInt8(BS.REGION_SOLID_WATER); erode_px = ERODE_PX,
    )
    sw_idx = findall(sw_bool)
    n_z = size(hu_mixed, 3)

    function _stats(vol)
        vals = Float64[Float64(vol[ci, z]) for z in 1:n_z, ci in sw_idx]
        (mean = mean(vals), std = std(vals), n = length(vals))
    end
    stats = (
        low = _stats(hu_tube.low),
        high = _stats(hu_tube.high),
        mixed = _stats(hu_mixed),
    )
    for (tag, s) in pairs(stats)
        @info "[poly · Flash] $(tag) SW ROI: ⟨HU⟩ = $(round(s.mean, digits = 2)), σ = $(round(s.std, digits = 2)) HU (n = $(s.n))"
    end
    (stats..., mask_2d = collect(sw_bool))
end;

# ╔═╡ 12000009-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)
    mid = size(hu_mixed, 3) ÷ 2

    fig = Mke.Figure(size = (1400, 520))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    panels = (
        (1, "80 kVp (tube A)", hu_tube.low),
        (2, "Sn140 kVp (tube B)", hu_tube.high),
        (3, "Mixed M$(MIX_W_LOW)", hu_mixed),
    )
    for (c, ttl, vol) in panels
        ax = Mke.Axis(
            fig[1, c]; title = ttl,
            aspect = Mke.DataAspect(), axis_kwargs...,
        )
        Mke.heatmap!(ax, vol[:, :, mid]; colormap = :grays, colorrange = HU_window)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1, 4]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_poly_recon.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000009-0000-4000-8000-000000000050
md"""
### Poly Water Values

Water-HU validation of the Flash LUT in the plain EICT chain, **before**
any spectral decomposition: solid-water ⟨HU⟩ ± σ for each per-tube recon
and the mixed image.  All three should cluster at ≈ 0 HU.
"""

# ╔═╡ 12000009-0000-4000-8000-000000000060
let
    entries = (
        ("80 kVp", poly_water_stats.low),
        ("Sn140 kVp", poly_water_stats.high),
        ("Mixed M$(MIX_W_LOW)", poly_water_stats.mixed),
    )
    n = length(entries)
    means = [e[2].mean for e in entries]
    stds = [e[2].std for e in entries]

    fig = Mke.Figure(size = (1180, 580))

    # ─── Left panel — eroded SW ROI on the mixed image ──────────────────
    HU_window = (-200, 500)
    mid = size(hu_mixed, 3) ÷ 2
    overlay = Float32[b ? 1.0f0 : NaN32 for b in poly_water_stats.mask_2d]

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "Eroded Water Region",
        subtitle = "Overlaid on mixed image",
        aspect = Mke.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, hu_mixed[:, :, mid]; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(ax1)

    # ─── Right panel — water ⟨HU⟩ ± σ per poly readout ───────────────────
    bar_colors = [Mke.cgrad(:plasma, n; categorical = true)[i] for i in 1:n]
    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Poly Water HU",
        subtitle = "Solid-water ROI, mean ± σ",
        xlabel = "Reconstruction", ylabel = "HU",
        xticks = (collect(1:n), [e[1] for e in entries]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.barplot!(
        ax2, 1:n, means;
        color = bar_colors, strokecolor = :black, strokewidth = 1,
    )
    Mke.errorbars!(ax2, 1:n, means, stds; color = :black, whiskerwidth = 14, linewidth = 2)
    Mke.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, (m, s)) in enumerate(zip(means, stds))
        Mke.text!(
            ax2, k, m + s;
            text = "$(round(m, digits = 1)) ± $(round(s, digits = 1)) HU",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 6),
        )
    end

    y_max = max(25.0, 1.4 * maximum(abs.(means) .+ stds))
    Mke.ylims!(ax2, -y_max, y_max)

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_poly_water_values.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000a-0000-4000-8000-000000000001
md"""
## 9. Spectral Basis from the Two Tubes

`acquire` built the basis with `spectral_basis_from_acquisitions`, which merges
the two tubes' energy grids onto their union and scales each tube's per-ray
detected spectrum by its own air counts, so the likelihood sees the absolute
response ``\Phi_k(E)`` of every ray and channel: **source × flat filter ×
bowtie × Flash UFC η(E)**, the identical model the forward projector applied
(it is resolved from the same `build_physics_config`, and therefore the same
Flash table, that `simulate!` used).  No calibration scan is involved, and
the noise-free, measured and calibration draws all share it.
"""

# ╔═╡ 1200000a-0000-4000-8000-000000000010
basis = de_measured.basis;

# ╔═╡ 1200000a-0000-4000-8000-000000000015
Markdown.parse("""
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point
energy grid for $(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response
sums to the air counts to within $(round(basis.I0_relerr, sigdigits = 2))
(relative).
""")

# ╔═╡ 1200000b-0000-4000-8000-000000000001
md"""
## 10. The VMI Chain: `vmi_pipeline`

One package call from the two corrected sinograms to the VMI stack, the chain
of the basis-spectral-denoising paper:
`vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PAIR_FILTER.filter, pair_basis, composite_energy)`.

1. **Projection HYPR-LR** (`ProjectionHYPR`): a 3 × 3 window (columns ×
   views of one parity, `view_stride = 2`, so the odd and even views stay
   independent) on the counts of each detector row, with each tube's
   dispersion measured from its own air rays.
2. **K = 2 n-channel decomposition**: per ray, the Poisson maximum-likelihood
   iodine + water pair under the exact polychromatic mean of both tubes,
   every detector row kept.
3. **The spectral pair** (`spectral_pair`): FDK of the composite `M`, the
   minimum-noise VMI, and of its complement `I⊥`, whose noise is
   uncorrelated with `M`'s, each with its own window of the fitted
   `PairFilter`, on the pair `pair_basis` fixed from the calibration draw.
4. **ACNR** on the complement only (`acnr_complement!`): the composite is
   left as reconstructed.
5. **Image HYPR** (`ImageHYPR`): the composite kept, the complement pooled
   within its slice with weights from the composite, over the window with
   the least estimated risk.
6. **VMI synthesis** at 50 / 70 / 100 / 140 keV from the one basis pair.

Nothing in the chain averages detector rows or slices.

**The fitted windows (`PAIR_FILTER`).** Each window is an apodized ramp
`W(f) = exp(-(f / f_c)^p)` on the grid's Nyquist axis.  The Flash's pair
(composite `f_c = 1.05, p = 1.5`; complement `f_c = 0.8, p = 1.0`) was
fitted in basis-spectral-denoising to the MTF and NPS of the physical Flash
scan of the Gammex, on 512² over 33 cm, the grid used here.  A window is a
function of physical frequency, so on another grid it is rescaled by the
ratio of the two grids' `grid_bandlimit`; the ratio on this grid is printed
below.

**`pair_basis` and `composite_energy`, from the calibration draw.** The pair
the windows act on (`E*`, `β`) and the composite energy ACNR and the image
HYPR act on are properties of the scanner and protocol, not of one noise
draw: near its minimum the VMI noise hardly changes with energy, so an
argmin measured on the draw being evaluated would itself be noise.  They are
measured once, on the calibration draw, with a standard soft-tissue window
(`SoftFilter`): `E*` and `β` from its plain decomposition, the composite
energy from its decomposition after the projection HYPR.  The measured
draw, the same draw without denoising, and the noise-free reference are then
reconstructed alike.
"""

# ╔═╡ 1200000b-0000-4000-8000-000000000005
# basis-spectral-denoising's HYPR_CHAIN: ProjectionHYPR(3 × 3 Box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 1200000b-0000-4000-8000-000000000006
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 1200000b-0000-4000-8000-000000000007
# The Flash's fitted pair of FDK windows (basis-spectral-denoising's `RECON.flash`), fitted on
# 512² over 33 cm and rescaled onto this grid's frequency axis as its `RECON_BY` does.
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11)),
        fitted = (composite = (fc = 1.05, p = 1.5), complement = (fc = 0.8, p = 1.0)),
        g = de_measured.geom,
        r = BS.grid_bandlimit(g, recon_opts.matrix_size) / min(1.0, g.pixel_size / (33.0 / 512))
    window(q) = BS.CustomFilter(knots, Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots))
    (filter = BS.PairFilter(window(fitted.composite), window(fitted.complement)), kernels = fitted, scale = r)
end;

# ╔═╡ 1200000b-0000-4000-8000-000000000008
# `pair_basis` and `composite_energy` (basis-spectral-denoising's `PAIR_BASIS`), from the calibration draw
calibration = let a = de_acq[DE_DRAWS.calibration]
    decompose(denoiser) = BS.vmi_pipeline(; channels = a.channels, basis = a.basis, geom = a.geom,
        to_backend = to_gpu, matrix_size = recon_opts.matrix_size, vmi_energies = Tuple(de_vmi_energies),
        denoiser, keep_sinograms = true, VMI_CHAIN..., use_acnr = false).sinograms
    sp(d; kw...) = BS.spectral_pair(d.water, d.iodine, a.geom, recon_opts.matrix_size;
        filter = BS.SoftFilter(), antialias = VMI_CHAIN.antialias, to_backend = to_gpu, kw...)
    x = sp(decompose(nothing))                                   # the plain decomposition
    c = sp(decompose(BS.SpectralHYPR(image = nothing));          # after the projection HYPR
        basis = (Estar = x.Estar, β = x.β))
    (pair_basis = (Estar = x.Estar, β = x.β), composite_energy = c.Estar)
end;

# ╔═╡ 1200000c-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 1200000b-0000-4000-8000-000000000010
de_vmi = BS.vmi_pipeline(;
    channels = de_measured.channels,
    basis,
    geom = de_measured.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 1200000b-0000-4000-8000-000000000011
# The same draw and reconstruction without the denoising (no HYPR, no ACNR), and the noise-free
# reference (draw 0, the same projection), reconstructed alike.
de_none, de_ref = (BS.vmi_pipeline(;
        channels = de_acq[r].channels, basis, geom = de_measured.geom, to_backend = to_gpu,
        matrix_size = recon_opts.matrix_size, vmi_energies = Tuple(de_vmi_energies),
        denoiser = nothing, use_acnr = false, fbp_filter = PAIR_FILTER.filter,
        pair_basis = calibration.pair_basis, composite_energy = calibration.composite_energy,
        VMI_CHAIN...,
    ) for r in (DE_DRAWS.measured, DE_DRAWS.reference));

# ╔═╡ 1200000b-0000-4000-8000-000000000015
let
    q = de_vmi.quality
    d = de_vmi.settings.denoiser
    pct(x) = round(100x, digits = 3)
    pb = calibration.pair_basis
    Markdown.parse("""
    The decomposition solved $(q.n_rays) rays in $(round(de_vmi.elapsed_s, digits = 1)) s
    with $(round(q.outer_mean, digits = 1)) outer iterations on average; $(pct(q.frac_not_converged))% did not
    converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions (variance / mean of the counts on the air rays) of
    $(join(round.(d.dispersion, digits = 2), " and ")) for tube A and tube B.

    From the calibration draw: the windows act on the pair at E* = $(round(pb.Estar, digits = 1)) keV,
    β = $(round(pb.β, sigdigits = 3)), and ACNR and the image HYPR on the composite at
    $(round(calibration.composite_energy, digits = 1)) keV. The image HYPR pooled the complement over a
    $(d.image_estimates.window) × $(d.image_estimates.window) window. The fitted windows were rescaled by
    $(round(PAIR_FILTER.scale, digits = 3)) onto this grid's frequency axis.
    """)
end

# ╔═╡ 1200000a-0000-4000-8000-000000000040
let
    n_row = size(de_vmi.sinograms.iodine, 2)
    mid_r = n_row ÷ 2 + 1

    fig = Mke.Figure(size = (1400, 580))
    axis_kwargs = (
        titlesize = 32, subtitlesize = 24,
        xlabel = "View", ylabel = "Detector Column",
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 16, yticklabelsize = 16,
    )

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = permutedims(de_vmi.sinograms.iodine[:, mid_r, :], (2, 1))
    slice_wat = permutedims(de_vmi.sinograms.water[:, mid_r, :], (2, 1))

    panels = (
        (1, 1, 2, "Iodine Basis Sinogram", "g/cm²", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis Sinogram", "g/cm²", slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = Mke.Axis(fig[r, panel_c]; title = ttl, axis_kwargs...)
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 1200000b-0000-4000-8000-000000000040
let
    fig = Mke.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = (size(de_vmi.images.iodine, 3) + 1) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = de_vmi.images.iodine[:, :, mid]
    slice_wat = de_vmi.images.water[:, :, mid]

    panels = (
        (1, 1, 2, "Iodine Basis", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis", "g/cm³", slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = Mke.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = Mke.DataAspect(), axis_kwargs...
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 1200000c-0000-4000-8000-000000000001
md"""
## 11. VMIs

`vmi_pipeline` synthesizes each VMI from the basis pair (McCollough 2015):

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

The figure shows the three reconstructions of §10 side by side: the measured
draw without denoising, the measured draw through the full chain, and the
noise-free reference.  The `solid_water_basis` diagnostic reports the basis
pair in the eroded solid-water region: a perfect decomposition reads water
density ≈ 1 g/cm³ (solid water is not pure water, so a small offset is
expected) and iodine ≈ 0.
"""

# ╔═╡ 1200000c-0000-4000-8000-000000000010
solid_water_basis = let
    ERODE_PX = 12.0

    sw_bool_raw = (roi_labels .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)

    n_raw = count(sw_bool_raw); n_eroded = count(sw_bool)
    n_eroded == 0 && error(
        "solid_water_basis: deep erosion (σ = $(ERODE_PX) px) wiped out the SW " *
            "ROI (raw count = $(n_raw)).  Reduce erode_px or check phantom mask."
    )

    sw_idx = findall(sw_bool)
    _mean(vol) = mean(vol[ci, z] for z in axes(vol, 3) for ci in sw_idx)

    c_w = Float64(_mean(de_vmi.images.water))
    c_i = Float64(_mean(de_vmi.images.iodine))
    @info "solid_water_basis: ⟨c_water⟩_SW = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_SW = $(round(1000c_i, digits = 3)) mg/mL"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * size(de_vmi.images.water, 3),
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 1200000c-0000-4000-8000-000000000020
# The VMI stack as one volume per energy (keV → (nx, ny, nz) HU).
vmi_HU_final = Dict(
    Float64(E) => de_vmi.vmis[:, :, :, k] for (k, E) in pairs(de_vmi.energies)
);

# ╔═╡ 1200000c-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)
    mid = (size(de_vmi.vmis, 3) + 1) ÷ 2
    runs = (("No denoising", de_none), ("SpectralHYPR", de_vmi), ("Noise-free reference", de_ref))

    fig = Mke.Figure(size = (1500, 1150))
    for (r, (label, run)) in enumerate(runs)
        Mke.Label(fig[r, 1], label; rotation = π / 2, fontsize = 26, tellheight = false)
        for (c, E) in enumerate(de_vmi_energies)
            ax = Mke.Axis(
                fig[r, c + 1]; title = r == 1 ? "$(Int(E)) keV" : "",
                aspect = Mke.DataAspect(), titlesize = 30,
            )
            Mke.heatmap!(ax, run.vmis[:, :, mid, c]; colormap = :grays, colorrange = HU_window)
            Mke.hidedecorations!(ax)
        end
    end
    Mke.Colorbar(
        fig[1:3, length(de_vmi_energies) + 2];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000e-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV.

!!! info "Methodology"
    - **Measured HU** = mean over an 8-px-radius circular ROI at the rod
      centroid, broadcast across all z slices.
    - **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
      `BS.compute_μ_at_energy` — pure physics, no fitting.
"""

# ╔═╡ 1200000e-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 1200000e-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 1200000e-0000-4000-8000-000000000030
rod_data = let
    materials = phantom_cpu.materials
    mask_2d = roi_labels
    nx, ny = size(mask_2d)
    ROI_RADIUS_PX = 8

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function rod_roi_mask(label::UInt8)
        cx, cy = rod_centroid(label)
        i_lo = max(1, floor(Int, cx - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_RADIUS_PX))
        roi = CartesianIndex{2}[]
        r² = Float64(ROI_RADIUS_PX)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        return roi
    end

    rod_rois = Dict(
        lab => rod_roi_mask(lab)
            for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )

    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
            for E in de_vmi_energies
    )

    function theoretical_hu(material, E::Float64)
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[E]) / μ_water_E[E]
    end

    function measured_hu(vmi_vol, label::UInt8)
        roi = rod_rois[label]
        s = 0.0; n = 0
        for z in 1:size(vmi_vol, 3), ci in roi
            s += vmi_vol[ci, z]; n += 1
        end
        return s / n
    end

    out = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        labels = ROD_LABELS[group]
        n_rods = length(labels)
        n_E = length(de_vmi_energies)
        meas = zeros(Float64, n_rods, n_E)
        theo = zeros(Float64, n_rods, n_E)
        for (i, lab) in pairs(labels)
            mat = materials[Int(lab) + 1]   # mask_value + 1
            for (j, E) in pairs(de_vmi_energies)
                meas[i, j] = measured_hu(vmi_HU_final[E], lab)
                theo[i, j] = theoretical_hu(mat, E)
            end
        end
        out[group] = (
            labels = labels, names = ROD_NAMES[group],
            measured = meas, theoretical = theo,
        )
    end
    out
end;

# ╔═╡ 1200000e-0000-4000-8000-000000000040
md"""
### Water ROI
"""

# ╔═╡ 1200000e-0000-4000-8000-000000000050
let
    fig = Mke.Figure(size = (1180, 580))

    HU_window = (-200, 500)
    mid = (size(vmi_HU_final[70.0], 3) + 1) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "Eroded Water Region",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = Mke.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(
        ax1, overlay;
        colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(ax1)

    sw_hu_per_keV = [vmi_noise_by_keV[E].mean for E in de_vmi_energies]
    ref_hu_per_keV = [vmi_noise_by_keV[E].reference for E in de_vmi_energies]

    n_E = length(de_vmi_energies)
    bar_colors = [Mke.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Water Region Mean HU",
        subtitle = "SpectralHYPR (bars) vs noise-free reference (◆)",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 22,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.barplot!(
        ax2, 1:n_E, sw_hu_per_keV;
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    Mke.scatter!(ax2, 1:n_E, ref_hu_per_keV; color = :black, marker = :diamond, markersize = 18)
    Mke.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    y_max = max(15.0, 1.2 * maximum(abs, vcat(sw_hu_per_keV, ref_hu_per_keV)))
    Mke.ylims!(ax2, -y_max, y_max)
    # the values, above the bars and clear of the reference markers
    for (k, (h, r)) in enumerate(zip(sw_hu_per_keV, ref_hu_per_keV))
        Mke.text!(
            ax2, k, 0.6 * y_max;
            text = "$(round(h, digits = 1)) HU\nref $(round(r, digits = 1))",
            align = (:center, :center), fontsize = 16,
        )
    end

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_water_roi.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000e-0000-4000-8000-000000000060
md"""
### Water-Region Noise

The mean and σ are both measured on the **deeply eroded solid-water region**
(the same 12-px-eroded mask as the accuracy ROI, as in notebook 03), over
every slice: a large region keeps the per-keV statistics stable where FBP
noise is spatially correlated.  The noise is measured **against the
noise-free reference**, σ = std(VMI − reference): the reference shares the
projection and the reconstruction, so the difference is the noise alone,
with no structure (residual cupping, the rods' edges) mixed in.  The same
draw without the denoising shows what the chain removes.
"""

# ╔═╡ 1200000e-0000-4000-8000-000000000080
vmi_noise_by_keV = let
    roi_idx = findall(solid_water_basis.mask_2d)
    nz_r = size(de_vmi.vmis, 3)
    vals(run, k) = Float64[Float64(run.vmis[ci, z, k]) for z in 1:nz_r, ci in roi_idx]

    out = Dict{Float64, NamedTuple}()
    for (k, E) in enumerate(de_vmi_energies)
        p, n, r = vals(de_vmi, k), vals(de_none, k), vals(de_ref, k)
        out[E] = (mean = mean(p), reference = mean(r), std = std(p .- r), std_none = std(n .- r),
            n = length(p))
        @info "water region @ $(Int(E)) keV: ⟨HU⟩ = $(round(mean(p), digits = 2)) " *
            "(reference $(round(mean(r), digits = 2))), σ = $(round(std(p .- r), digits = 2)) HU " *
            "(no denoising $(round(std(n .- r), digits = 2)) HU, n = $(length(p)))"
    end
    out
end;

# ╔═╡ 1200000e-0000-4000-8000-000000000090
let
    HU_window = (-200, 500)
    mid = (size(vmi_HU_final[70.0], 3) + 1) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    fig = Mke.Figure(size = (1180, 580))

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "Water-Region Noise ROI",
        subtitle = "Eroded solid water, overlaid on 70 keV VMI",
        aspect = Mke.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(ax1)

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σs = [vmi_noise_by_keV[E].std for E in Es]
    σn = [vmi_noise_by_keV[E].std_none for E in Es]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Water-Region Noise vs Energy",
        subtitle = "σ of (VMI − noise-free reference)",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        titlesize = 32, subtitlesize = 22,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.scatterlines!(ax2, Es, σn; color = :gray45, markersize = 14, linewidth = 2.5,
        linestyle = :dash, label = "No denoising")
    Mke.scatterlines!(ax2, Es, σs; color = :tomato, markersize = 18, linewidth = 3,
        label = "SpectralHYPR")
    for (E, σ) in zip(Es, σs)
        Mke.text!(
            ax2, E, σ;
            text = "$(round(σ; digits = 1))",
            align = (:center, :top),
            fontsize = 16, offset = (0, -12),
        )
    end
    for (E, σ) in zip(Es, σn)
        Mke.text!(
            ax2, E, σ;
            text = "$(round(σ; digits = 1))",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 10), color = :gray35,
        )
    end
    Mke.xlims!(ax2, first(Es) - 15, last(Es) + 15)
    Mke.ylims!(ax2, 0, 1.25 * maximum(σn))
    Mke.axislegend(ax2; position = :rt, framevisible = true, labelsize = 16)

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_water_noise.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000f-0000-4000-8000-000000000001
md"""
### Per-Rod Regression
"""

# ╔═╡ 1200000f-0000-4000-8000-000000000010
let
    fig = Mke.Figure(size = (1180, 580))

    cmap_ca = Mke.cgrad(:Oranges, 7; categorical = true)
    cmap_i = Mke.cgrad(:GnBu, 7; categorical = true)

    panels = (
        (
            group = :Ca, title = "Calcium rods",
            subtitle = "50–600 mg/mL",
            cmap = cmap_ca, ylim = (0, 4200),
        ),
        (
            group = :I, title = "Iodine rods",
            subtitle = "2–20 mg/mL",
            cmap = cmap_i, ylim = (0, 1500),
        ),
    )

    for (col, p) in pairs(panels)
        ax = Mke.Axis(
            fig[1, col];
            title = p.title,
            subtitle = p.subtitle,
            xlabel = "VMI energy (keV)",
            ylabel = "HU",
            xticks = de_vmi_energies,
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 18, yticklabelsize = 16,
        )
        Mke.ylims!(ax, p.ylim...)

        d = rod_data[p.group]
        rod_lines = Vector{Any}(undef, length(d.names))
        for i in eachindex(d.names)
            color = p.cmap[i]
            Mke.scatterlines!(
                ax, de_vmi_energies, vec(d.measured[i, :]);
                color = color, linewidth = 2.5, markersize = 9,
            )
            Mke.lines!(
                ax, de_vmi_energies, vec(d.theoretical[i, :]);
                color = color, linewidth = 1.6, linestyle = :dash,
            )
            rod_lines[i] = Mke.LineElement(color = color, linewidth = 2.5)
        end

        style_meas = Mke.MarkerElement(
            color = :black, marker = :circle, markersize = 9,
            strokecolor = :black, strokewidth = 1,
        )
        style_theo = Mke.LineElement(
            color = :black, linewidth = 1.6, linestyle = :dash,
        )
        Mke.axislegend(
            ax,
            vcat([style_meas, style_theo], rod_lines),
            vcat(["Measured", "Theoretical"], collect(d.names));
            position = :rt, framevisible = true, labelsize = 18,
            rowgap = 1, padding = (6, 6, 6, 6),
        )
    end

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000f-0000-4000-8000-000000000020
md"""
### Linear Regression
"""

# ╔═╡ 1200000f-0000-4000-8000-000000000030
let
    fig = Mke.Figure(size = (1000, 1200))

    energy_colors = Dict(
        50.0 => Mke.RGBf(0.85, 0.27, 0.1),
        70.0 => Mke.RGBf(0.95, 0.65, 0.13),
        100.0 => Mke.RGBf(0.13, 0.59, 0.85),
        140.0 => Mke.RGBf(0.1, 0.27, 0.65),
    )

    function fit_lr(x::Vector{Float64}, y::Vector{Float64})
        x̄ = mean(x); ȳ = mean(y)
        sxx = sum((x .- x̄) .^ 2)
        sxy = sum((x .- x̄) .* (y .- ȳ))
        β = sxy / sxx
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        ss_res = sum((y .- ŷ) .^ 2)
        ss_tot = sum((y .- ȳ) .^ 2)
        r² = 1 - ss_res / ss_tot
        rmse = sqrt(ss_res / length(y))
        return (slope = β, intercept = α, r² = r², rmse = rmse)
    end

    panels = (
        (:Ca, "Calcium Rods", "50–600 mg/mL"),
        (:I, "Iodine Rods", "2–20 mg/mL"),
    )

    for (row, (group, title, subtitle)) in pairs(panels)
        d = rod_data[group]
        ax = Mke.Axis(
            fig[row, 1];
            title = title,
            subtitle = subtitle,
            xlabel = "Theoretical HU",
            ylabel = "Measured HU",
            titlesize = 32, subtitlesize = 24,
            xlabelsize = 22, ylabelsize = 22,
            xticklabelsize = 16, yticklabelsize = 16,
        )

        lim_lo = min(0.0, minimum(d.measured), minimum(d.theoretical))
        lim_hi = max(maximum(d.measured), maximum(d.theoretical)) * 1.05
        Mke.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "Unity (y = x)",
        )

        for (j, E) in pairs(de_vmi_energies)
            x = Vector{Float64}(vec(d.theoretical[:, j]))
            y = Vector{Float64}(vec(d.measured[:, j]))
            color = energy_colors[E]
            Mke.scatter!(ax, x, y; color = color, markersize = 11)

            f = fit_lr(x, y)
            xrange = [minimum(x), maximum(x)]
            yrange = f.intercept .+ f.slope .* xrange
            sign_str = f.intercept ≥ 0 ? "+" : "−"
            label = "$(Int(E)) keV: y = $(round(f.slope, digits = 2))·x " *
                "$(sign_str) $(round(abs(f.intercept), digits = 0)) HU   " *
                "R² = $(round(f.r², digits = 3))   " *
                "RMSE = $(round(f.rmse, digits = 1)) HU"
            Mke.lines!(
                ax, xrange, yrange;
                color = color, linewidth = 2, label = label,
            )
        end

        Mke.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 16, padding = (6, 6, 6, 6), rowgap = 1,
        )
    end

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000010-0000-4000-8000-000000000001
md"""
## Verification

Automated PASS/FAIL gates over both acquisition classes (notebook 01 convention):

1. **Regular water accuracy** — |⟨HU⟩| ≤ 5 in tube A, tube B, combined.
2. **Dual-power √2** — `σ_combined / σ_single` ∈ [0.62, 0.80]
   (ideal 0.707; fails if the tube seeds were ever shared).
3. **DE poly water accuracy** — |⟨HU⟩| ≤ 5 in low, high, mixed.
4. **VMI water accuracy** — |⟨HU⟩| ≤ 10 at every synthesized keV,
   measured on the deeply eroded solid-water region (notebook 03 convention).
5. **VMI bias against the noise-free reference** — the chain's water
   ⟨HU⟩ within 2 HU of the noise-free reference's at every keV: the
   denoising moves the noise, not the CT number.
6. **VMI noise flat in energy** — max / min σ over 50–140 keV ≤ 1.10, σ
   against the noise-free reference.  Without denoising the VMI noise is
   U-shaped: highest at 50 keV, lowest near the minimum-noise energy E*,
   rising again above it.  The chain keeps the composite (the VMI at E*) as
   reconstructed and pools only the complement, which carries the noise
   that grows away from E*; what remains in every VMI is mostly the
   composite's noise, and the curve flattens (the undenoised curve's
   max / min is printed beside the gate).  The gate this notebook used
   before, σ strictly decreasing from 50 to 140 keV, described the earlier
   chain and no longer holds; the check for this chain is that the noise
   stays near its minimum at every energy.
7. **The denoising lowers the noise** — σ below the undenoised
   reconstruction of the same draw at every keV.
8. **The chain that ran** — both tubes in the likelihood, projection and
   image HYPR, ACNR on the complement, the fitted `PairFilter`, the pair and
   composite energy of the calibration draw, view integration
   (`de_vmi.settings`, `FLASH.opts`).
9. **Per-rod regression** — measured-vs-theoretical slope ∈ [0.85, 1.15]
   and R² ≥ 0.99 at every keV, both rod groups.
"""

# ╔═╡ 12000010-0000-4000-8000-000000000010
verification = let
    checks = Tuple{String, Bool, String}[]

    # 1. Regular (dual-power) water accuracy
    for (tag, s) in (("tube A", dp_stats.water.a), ("tube B", dp_stats.water.b),
        ("combined", dp_stats.water.combined))
        push!(checks, (
            "regular water $(tag)", abs(s.mean) ≤ 5.0,
            "⟨HU⟩ = $(round(s.mean, digits = 2)) (gate ±5)",
        ))
    end

    # 2. Dual-power √2 noise reduction
    push!(checks, (
        "dual-power σ ratio", 0.62 ≤ dp_stats.noise_ratio ≤ 0.80,
        "`σ_combined / σ_single` = $(round(dp_stats.noise_ratio, digits = 3)) (ideal 0.707, gate [0.62, 0.80])",
    ))

    # 3. DE poly water accuracy
    for (tag, s) in (("80 kVp", poly_water_stats.low), ("Sn140 kVp", poly_water_stats.high),
        ("mixed", poly_water_stats.mixed))
        push!(checks, (
            "DE poly water $(tag)", abs(s.mean) ≤ 5.0,
            "⟨HU⟩ = $(round(s.mean, digits = 2)) (gate ±5)",
        ))
    end

    # 4. VMI water accuracy per keV
    for E in de_vmi_energies
        m = vmi_noise_by_keV[E].mean
        push!(checks, (
            "VMI water @ $(Int(E)) keV", abs(m) ≤ 10.0,
            "⟨HU⟩ = $(round(m, digits = 2)) (gate ±10)",
        ))
    end

    # 5. Bias against the noise-free reference (same projection, same reconstruction)
    bias = [vmi_noise_by_keV[E].mean - vmi_noise_by_keV[E].reference for E in de_vmi_energies]
    push!(checks, (
        "VMI bias vs noise-free reference", maximum(abs, bias) ≤ 2.0,
        "⟨HU⟩ − ⟨HU⟩_ref = " * join([string(round(b, digits = 2)) for b in bias], ", ") * " (gate ±2)",
    ))

    # 6. The noise is flat in energy: the chain carries the composite's noise to every VMI
    σs = [vmi_noise_by_keV[E].std for E in de_vmi_energies]
    σn_all = [vmi_noise_by_keV[E].std_none for E in de_vmi_energies]
    flat = maximum(σs) / minimum(σs)
    push!(checks, (
        "VMI noise flat in energy", flat ≤ 1.10,
        "σ = " * join([string(round(σ, digits = 1)) for σ in σs], ", ") *
            " HU, max / min = $(round(flat, digits = 3)) (gate ≤ 1.10; no denoising " *
            "$(round(maximum(σn_all) / minimum(σn_all), digits = 2)))",
    ))

    # 7. The denoising lowers the noise at every keV
    σn = [vmi_noise_by_keV[E].std_none for E in de_vmi_energies]
    push!(checks, (
        "denoising lowers σ", all(σs .< σn),
        "σ / σ_none = " * join([string(round(a / b, digits = 2)) for (a, b) in zip(σs, σn)], ", "),
    ))

    # 8. The chain that ran
    chain = de_vmi.settings
    push!(checks, (
        "VMI chain",
        chain.n_channels == 2 && chain.denoiser !== nothing &&
            chain.denoiser.projection !== nothing && chain.denoiser.image !== nothing &&
            chain.acnr !== nothing && chain.acnr.on === :complement &&
            chain.recon.filter isa BS.PairFilter &&
            chain.pair.basis == calibration.pair_basis &&
            chain.pair.Estar == calibration.composite_energy &&
            FLASH.opts.view_samples == 5 && all(e -> e.view_arc == 1.0, FLASH.exposures),
        "K = $(chain.n_channels), projection + image HYPR, ACNR on the `$(chain.acnr === nothing ? "none" : chain.acnr.on)`, " *
            "`PairFilter`, calibration pair E* = $(round(chain.pair.basis.Estar, digits = 1)) keV, " *
            "composite $(round(chain.pair.Estar, digits = 1)) keV, `view_samples` = $(FLASH.opts.view_samples)",
    ))

    # 9. Per-rod regression gates
    function _fit(x, y)
        x̄ = mean(x); ȳ = mean(y)
        β = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        r² = 1 - sum((y .- ŷ) .^ 2) / sum((y .- ȳ) .^ 2)
        (slope = β, r² = r²)
    end
    for group in (:Ca, :I), (j, E) in pairs(de_vmi_energies)
        d = rod_data[group]
        f = _fit(Vector{Float64}(vec(d.theoretical[:, j])), Vector{Float64}(vec(d.measured[:, j])))
        push!(checks, (
            "$(group) regression @ $(Int(E)) keV",
            (0.85 ≤ f.slope ≤ 1.15) && (f.r² ≥ 0.99),
            "slope = $(round(f.slope, digits = 3)), R² = $(round(f.r², digits = 4)) (gates slope [0.85, 1.15], R² ≥ 0.99)",
        ))
    end

    n_pass = count(c -> c[2], checks)
    for (name, ok, detail) in checks
        @info "$(ok ? "✅ PASS" : "❌ FAIL") · $(name) — $(detail)"
    end
    overall = n_pass == length(checks)
    @info (overall ? "🎉 VERIFICATION PASS $(n_pass)/$(length(checks))" :
        "🚨 VERIFICATION FAIL $(n_pass)/$(length(checks))")

    (checks = checks, n_pass = n_pass, n_total = length(checks), pass = overall)
end;

# ╔═╡ 12000010-0000-4000-8000-000000000020
let
    rows = join([
        "| $(name) | $(detail) | $(ok ? "✅" : "❌") |"
        for (name, ok, detail) in verification.checks
    ], "\n")
    Markdown.parse("""
### $(verification.pass ? "✅ Verification: PASS" : "❌ Verification: CHECK") — $(verification.n_pass)/$(verification.n_total) gates

| check | value | pass |
|---|---|:---:|
$rows
""")
end

# ╔═╡ 12000011-0000-4000-8000-000000000001
md"""
## Summary

```
Flash UFC MC η(E) LUT (BS.UFC_FLASH_MC_EFFICIENCY_LUT, Gd₂O₂S, 1–140 keV)
   → EICTScanner(detector_material = :ufc_flash) → detector_efficiency_ufc_flash()
   flash_opts: view_samples = 5 (view_arc 1.0 per tube), heel effect off
REGULAR: 120 kVp × 2 tubes, one projection, two seeds
   → per-tube η-aware BHC → FDK → HU → (A+B)/2
   → water ≈ 0 HU ×3, σ_combined ≈ σ_single/√2                    (§6)
DUAL ENERGY: 80 kVp (A) + Sn140 kVp (B, 0.4 mm Sn), × DOSE_SCALE
   each tube projected once → draws 0 (noise-free), 1 (measured), 9 (calibration)
   ├─→ POLY: per-tube η-aware BHC → FDK → HU → mixed image M_w       (§8)
   └─→ VMI:  spectral basis (bowtie + η_Flash per ray, shared by every draw)
             pair_basis, composite_energy ← calibration draw
             → BS.vmi_pipeline(; denoiser = SpectralHYPR(),
                   fbp_filter = PairFilter(fitted Flash windows), …)
             → VMI 50/70/100/140 keV vs the noise-free reference
             → per-rod measured vs theoretical regression        (§9–11)
   → automated PASS/FAIL verification over both classes
```

**What this notebook establishes:**

1. **The Flash is modeled as itself, not as a re-badged Force**: its own
   MC detector LUT (−28% η at 140 keV vs the Force), its published
   geometry (595/1085.6 mm, 64 × 736 @ 0.70473 mm iso), published anode
   angle (7°), published flat filtration (8.4 mm Al eq.), and the physical
   Flash scan's DE protocol (80/Sn140 with 0.4 mm Sn, dose-matched to its
   CTDIvol) — the scanner and acquisition of the basis-spectral-denoising
   paper.
2. **Both dual-source acquisition classes in one place**: the regular
   (dual-power) readout verifies accuracy and the √2 independence of the
   two tube chains; the DE readout verifies the poly/mixed chain and the
   full VMI chain (`vmi_pipeline`) on two very different detected spectra
   on opposite sides of the Gd K-edge fluorescence-escape cliff, each
   result scored against the noise-free draw of the same projection.
3. **Documented assumptions are explicit** (§3): bowtie profile, crystal
   depth, fill factor, electronic noise, tube-B z-offset, z-FFS — the
   remaining gaps on the parity checklist of the
   [SOMATOM Definition Flash entry of the scanners page](../../scanners/#somatom-definition-flash).

**src status:** `UFC_FLASH_MC_EFFICIENCY_LUT`,
`get_ufc_flash_mc_efficiency`, `detector_efficiency_ufc_flash()`, and the
`:ufc_flash` branches in `compute_eid_efficiency_vector` +
`build_physics_config` live in `src/detector/detector_efficiency.jl` /
`src/api/driver.jl`, covered by `test/detector.jl` (including a
never-alias-the-Force regression test).
"""

# ╔═╡ Cell order:
# ╟─12000001-0000-4000-8000-000000000010
# ╠═12000001-0000-4000-8000-000000000001
# ╠═12000001-0000-4000-8000-000000000002
# ╠═12000001-0000-4000-8000-000000000003
# ╟─12000001-0000-4000-8000-000000000020
# ╠═12000001-0000-4000-8000-000000000030
# ╠═12000001-0000-4000-8000-000000000031
# ╠═12000001-0000-4000-8000-000000000032
# ╠═12000001-0000-4000-8000-000000000033
# ╠═12000001-0000-4000-8000-000000000040
# ╟─12000001-0000-4000-8000-000000000050
# ╟─12000002-0000-4000-8000-000000000001
# ╟─12000002-0000-4000-8000-000000000030
# ╟─12000003-0000-4000-8000-000000000001
# ╠═12000003-0000-4000-8000-000000000010
# ╠═12000003-0000-4000-8000-000000000020
# ╟─12000004-0000-4000-8000-000000000001
# ╠═12000004-0000-4000-8000-000000000010
# ╟─12000006-0000-4000-8000-000000000001
# ╠═12000006-0000-4000-8000-000000000010
# ╟─12000005-0000-4000-8000-000000000001
# ╠═12000005-0000-4000-8000-000000000005
# ╠═12000005-0000-4000-8000-000000000010
# ╠═12000005-0000-4000-8000-000000000020
# ╠═12000005-0000-4000-8000-000000000030
# ╟─12000005-0000-4000-8000-000000000040
# ╠═12000006-0000-4000-8000-000000000020
# ╠═12000006-0000-4000-8000-000000000025
# ╠═12000006-0000-4000-8000-000000000030
# ╟─12000006-0000-4000-8000-000000000040
# ╟─12000007-0000-4000-8000-000000000001
# ╠═12000007-0000-4000-8000-000000000005
# ╠═12000007-0000-4000-8000-000000000008
# ╠═12000007-0000-4000-8000-000000000010
# ╠═12000007-0000-4000-8000-000000000015
# ╠═12000007-0000-4000-8000-000000000020
# ╠═12000007-0000-4000-8000-000000000025
# ╠═12000007-0000-4000-8000-000000000030
# ╟─12000007-0000-4000-8000-000000000040
# ╟─12000008-0000-4000-8000-000000000001
# ╠═12000006-0000-4000-8000-000000000012
# ╠═12000008-0000-4000-8000-000000000005
# ╠═12000008-0000-4000-8000-000000000020
# ╟─12000008-0000-4000-8000-000000000030
# ╠═12000008-0000-4000-8000-000000000035
# ╟─12000008-0000-4000-8000-000000000040
# ╟─12000009-0000-4000-8000-000000000001
# ╠═12000009-0000-4000-8000-000000000010
# ╠═12000009-0000-4000-8000-000000000012
# ╟─12000009-0000-4000-8000-000000000015
# ╠═12000009-0000-4000-8000-000000000020
# ╠═12000009-0000-4000-8000-000000000025
# ╠═12000009-0000-4000-8000-000000000028
# ╠═12000009-0000-4000-8000-000000000030
# ╟─12000009-0000-4000-8000-000000000040
# ╟─12000009-0000-4000-8000-000000000050
# ╟─12000009-0000-4000-8000-000000000060
# ╟─1200000a-0000-4000-8000-000000000001
# ╠═1200000a-0000-4000-8000-000000000010
# ╟─1200000a-0000-4000-8000-000000000015
# ╟─1200000b-0000-4000-8000-000000000001
# ╠═1200000b-0000-4000-8000-000000000005
# ╠═1200000b-0000-4000-8000-000000000006
# ╠═1200000b-0000-4000-8000-000000000007
# ╠═1200000b-0000-4000-8000-000000000008
# ╠═1200000c-0000-4000-8000-000000000015
# ╠═1200000b-0000-4000-8000-000000000010
# ╠═1200000b-0000-4000-8000-000000000011
# ╟─1200000b-0000-4000-8000-000000000015
# ╟─1200000a-0000-4000-8000-000000000040
# ╟─1200000b-0000-4000-8000-000000000040
# ╟─1200000c-0000-4000-8000-000000000001
# ╠═1200000c-0000-4000-8000-000000000010
# ╠═1200000c-0000-4000-8000-000000000020
# ╟─1200000c-0000-4000-8000-000000000040
# ╟─1200000e-0000-4000-8000-000000000001
# ╠═1200000e-0000-4000-8000-000000000010
# ╠═1200000e-0000-4000-8000-000000000020
# ╠═1200000e-0000-4000-8000-000000000030
# ╟─1200000e-0000-4000-8000-000000000040
# ╟─1200000e-0000-4000-8000-000000000050
# ╟─1200000e-0000-4000-8000-000000000060
# ╠═1200000e-0000-4000-8000-000000000080
# ╟─1200000e-0000-4000-8000-000000000090
# ╟─1200000f-0000-4000-8000-000000000001
# ╟─1200000f-0000-4000-8000-000000000010
# ╟─1200000f-0000-4000-8000-000000000020
# ╟─1200000f-0000-4000-8000-000000000030
# ╟─12000010-0000-4000-8000-000000000001
# ╠═12000010-0000-4000-8000-000000000010
# ╟─12000010-0000-4000-8000-000000000020
# ╟─12000011-0000-4000-8000-000000000001
