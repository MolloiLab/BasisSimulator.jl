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
UFC detectors at 95°**, its published geometry/filtration, and the new
**Flash-specific Monte-Carlo detector LUT** — exercised through *both* of the
scanner's signature acquisition classes in one notebook:

```
Flash UFC MC η(E) LUT  (Khodajou-Chokami MC, 2026-08-26, 1–140 keV)
        │ via Scanner(detector_material = :ufc_flash)  (src MC-LUT pathway)
        ▼
┌─ REGULAR (dual power): 120 kVp on BOTH tubes ──────────────────────────┐
│  tube A + tube B (independent noise) → per-tube η-aware BHC → FDK → HU │
│  → combined image → water ≈ 0 HU ×3, σ_combined ≈ σ_single/√2     (§6) │
└────────────────────────────────────────────────────────────────────────┘
┌─ DUAL ENERGY: 100 kVp (A) / Sn140 kVp (B, 0.4 mm Sn) ──────────────────┐
│  POLY: per-tube η-aware BHC → FDK → HU → Siemens mixed image M_w  (§8) │
│  VMI:  Cong decomp (raw sinograms) → FBP × 2 → cov-ACNR                │
│        → VMI 50/70/100/140 keV → per-rod regression         (§9–§11+)  │
└────────────────────────────────────────────────────────────────────────┘
        ▼
   Automated PASS/FAIL verification (water HU, √2 noise, monotonic
   VMI noise, per-rod regression gates)
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
    Every published number below comes from the sourced dossier at
    `docs/scanner_dossiers/somatom_definition_flash.md` (FDA 510(k)s,
    Siemens Dec-2010 datasheet, AAPM LDCT-PD projection geometry, Primak
    AJR 2010 for the 0.4 mm Sn Selective Photon Shield).  Unpublished
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
end

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

**Provenance**: Hamidreza Khodajou-Chokami, PhD (UC Irvine Medical Imaging
Laboratory), `flash_efficiency_results.csv`, CRSP lab share, received
2026-08-26.  The 140 values (1-keV grid, 1–140 keV) live verbatim in src as
`BS.UFC_FLASH_MC_EFFICIENCY_LUT` (`src/detector/detector_efficiency.jl`),
sister of the Force `BS.UFC_MC_EFFICIENCY_LUT`; an archival copy of the CSV
is at `docs/notebooks/data/ufc_flash_mc_efficiency_v1.csv` (gitignored, with
a PROVENANCE sidecar).

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
    Mke.lines!(ax, Es, η_force; color = :gray35, linewidth = 2.5, linestyle = :dash, label = "Force UFC (nb09)")
    Mke.lines!(ax, Es, η_gem; color = :steelblue, linewidth = 2.5, label = "Gemstone garnet (GE Apex)")

    Mke.vlines!(ax, [50.24]; color = :crimson, linestyle = :dash, linewidth = 1.5)
    Mke.text!(ax, 50.24, 0.62; text = "Gd K-edge\n50.2 keV", fontsize = 16, align = (:left, :top), offset = (4, 0))
    Mke.text!(
        ax, 140.0, 0.60;
        text = "Flash 0.588\nForce 0.816\n(−28%)", fontsize = 16,
        align = (:right, :bottom), offset = (-6, 4), color = :crimson,
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
## 3. `Scanner`: Siemens SOMATOM Definition Flash

Second-generation dual-source: **two STRATON MX P tubes + two UFC detectors
at 95°** in the same gantry.  Every value below is sourced in
`docs/scanner_dossiers/somatom_definition_flash.md`:

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

!!! warning "Documented modeling assumptions (no public source exists)"
    - **Anode angle in the spectrum model**: the Flash's anode is a
      *published* 7°, but the bundled IPEM spectra come only in 8°/10° →
      protocols use the closest available **8°** (`anode_angle = 8`);
      `Scanner.target_angle` keeps the true 7.0° (heel-effect metadata,
      inert with `use_heel_effect = false`).
    - **Bowtie**: Siemens form-filter shape is unpublished → CatSim
      **large-body** profile as stand-in (same convention as nb04/nb08/nb09).
    - **Flat-filter composition**: only the **Al equivalent** is published
      (8.4 mm) → modeled as 8.4 mm of aluminum.  Unlike nb09's Force
      (assumed Al+Ti stack), this figure is a *measured* datasheet value.
    - **Scintillator thickness 1.0 mm / fill factor 0.9**: proprietary;
      thickness is inert here (η comes from the Flash MC LUT; the MC
      high-energy roll-off implies a Beer-Lambert-effective ≈1.07 mm),
      fill factor auto-cancels in the air-scan calibration.
    - **Electronic noise = 0**: modeling the Stellar (K113342) build —
      TrueSignal's design point is "electronic noise negligible vs quantum";
      pre-2011 UFC builds would need a small nonzero value.
    - **Tube-B z-offset**: unpublished for the Flash (Force ≈ 0.88 mm) →
      **0.0 mm**, kept as an explicit constant so the mechanism is in place.
    - **z-FFS not modeled** → 1152 views/rotation (one focal-spot position,
      half of the 2304 FFS-interleaved readings).

!!! info "Dual source → two co-registered scans (the accepted hack)"
    Exactly like nb03 models rapid-kVp switching and nb09 models the Force,
    the Flash's two tubes are modeled as **two `Scanner` + `CTProtocol`
    configs run back-to-back** — but with **two unique tube sources**: each
    tube gets its own protocol (kVp, mA, filtration) and its own
    **independent noise seed** (two physically separate tube/detector
    chains must not share a noise realization — critical for the §6
    dual-power √2 check).

    - **Detector arc**: tube B gets detector-A's 736-channel arc so the
      (low, high) sinogram pair is per-ray co-registered for the Cong
      solver — defensible because the 33 cm Gammex body fits inside
      detector B's real 33.4 cm FOV (tighter than the Force's 35.5 cm —
      the Flash B-fan is the actual clinical DE-FOV limit).
    - **95° in-plane tube offset**: both modeled scans run a full axial
      rotation on the same angle grid — a constant angular offset has no
      effect on a full-rotation axial scan (nb09 §3 argument).
    - **DE-mode collimation**: the Flash reads 2 × 128 × 0.6 mm in DE mode;
      we use 4.8 mm (8 × 0.6 mm) — the thin-collimation equivalent that
      fits the 1 cm Gammex z-extent, same convention as nb03/nb09.
"""

# ╔═╡ 12000004-0000-4000-8000-000000000010
# Tube/detector A geometry — shared by both modeled tubes (see md above).
scanner = BS.Scanner(
    source_to_isocenter = 595.0,
    source_to_detector = 1085.6,

    detector_rows = 64,
    detector_cols = 736,
    detector_row_size = 0.6,
    detector_col_size = 0.70473,

    focal_spot_width = 0.7,
    focal_spot_length = 0.7,
    target_angle = 7.0,

    flat_filter_material = :aluminum,
    flat_filter_thickness = 8.4,
    bowtie_filter = :large_body,

    detector_material = :ufc_flash,   # → Flash-specific MC LUT (NOT :ufc)
    detector_depth = 1.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,

    electronic_noise = 0,
    detection_gain = 10.0,
);

# ╔═╡ 12000005-0000-4000-8000-000000000001
md"""
## 4. Protocols: Two Acquisition Classes

**Regular (dual power)** — the Flash's bread-and-butter dual-source mode:
both tubes at the **same** kVp, doubling the available power (40–1600 mA
combined).  Routine abdomen 120 kV; each tube at 420 mA (210 quality-ref
mAs at 0.5 s).

**Dual energy** — the Flash's clinical abdomen pair is **100/Sn140** with
the 0.4 mm Sn Selective Photon Shield on tube B (Primak AJR 2010).  The
bundled IPEM spectra top out at 140 kVp — **exactly the Flash's top kV**,
so unlike the Force (Sn150) *no kV substitution is needed*: this notebook
runs the true clinical pair.  Currents follow the published liver-VNC
quality-reference 230/178 mAs at 0.5 s → 460/356 mA (≈1.29 : 1 A : B).

| Acquisition | Tube | kVp | Filters | mA | views | rotation |
|-------------|------|-----|---------|----|-------|----------|
| Regular | A | 120 | 8.4 Al | 420 | 1152 | 0.5 s |
| Regular | B | 120 | 8.4 Al | 420 | 1152 | 0.5 s |
| DE | A (low)  | 100 | 8.4 Al | 460 | 1152 | 0.5 s |
| DE | B (high) | 140 | 8.4 Al + **0.4 Sn** | 356 | 1152 | 0.5 s |
"""

# ╔═╡ 12000005-0000-4000-8000-000000000005
protocol_reg = BS.CTProtocol(
    kVp = 120,
    mA = 420.0,
    views = 1152,
    rotation_time = 0.5,
    collimation_mm = 4.8,    # nominal beam width; axial cone guards are automatic
    anode_angle = 8,         # IPEM bundle has 8°/10° only; published anode is 7° (see §3)
);

# ╔═╡ 12000005-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 100,
    mA = 460.0,
    views = 1152,
    rotation_time = 0.5,
    collimation_mm = 4.8,
    anode_angle = 8,         # IPEM bundle has 8°/10° only; published anode is 7° (see §3)
);

# ╔═╡ 12000005-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 356.0,
    views = 1152,
    rotation_time = 0.5,
    collimation_mm = 4.8,
    anode_angle = 8,         # IPEM bundle has 8°/10° only; published anode is 7° (see §3)
    additional_filters = [("Sn", 0.4)],
);

# ╔═╡ 12000006-0000-4000-8000-000000000001
md"""
## 5. `SimOptions` and `ReconOptions`

`use_detector_efficiency = true` (the `:eict` preset default) routes
through the **src Flash UFC MC LUT**: `build_physics_config` sees
`Scanner(detector_material = :ufc_flash)` and dispatches to
`detector_efficiency_ufc_flash()`, so the EICT forward model weights every
energy by `w(E) · η_Flash(E)` and the detected flux (and therefore the
Poisson noise level) automatically reflects the Flash absorption.

**Tube B gets its own seed** (`sim_opts_b`): the two tube/detector chains
are physically independent, so their noise realizations must be too —
otherwise the §6 dual-power average would cancel *nothing*.

`use_heel_effect = false` keeps the forward spectral model exactly equal
to the η-folded basis the Cong inversion uses (heel is a small
row-direction effect; with 4.8 mm collimation at center it is negligible).
"""

# ╔═╡ 12000006-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,               # tube A chain
    use_heel_effect = false,   # exact forward/inverse spectral match
    projector = :dd_fast,      # same DD physics, single-pass fused kernels.
                               #  BHC (flash_poly_recon) reads sim_opts.projector to match.
);

# ╔═╡ 12000006-0000-4000-8000-000000000012
# Independent noise chain for tube B (spectra/geometry identical handling;
# only the random stream differs).
sim_opts_b = BS.SimOptions(
    fidelity = :eict,
    seed = 4321,               # tube B chain — MUST differ from tube A
    use_heel_effect = false,
    projector = :dd_fast,
);

# ╔═╡ 12000006-0000-4000-8000-000000000020
# Keep the intended centered 5 × 0.6 mm saved grid. The axial workspace
# automatically adds symmetric detector guard rows so peripheral voxels
# on both terminal slices retain measured cone-beam support.
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 5),
    fov_cm = 35.0,
    z_cm = 0.30,
);

# ╔═╡ 12000006-0000-4000-8000-000000000030
"""
    flash_detected_spectrum(protocol) -> (e, w_eta)

Display helper: tube spectrum × flat filtration × protocol filters
(IPEM, absolute flux) with the src Flash UFC MC η(E) folded in — what the
detector actually integrates (centered ray, no bowtie).
"""
function flash_detected_spectrum(protocol)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        sim_opts, protocol; scanner = scanner,
    )
    return e, Float64.(w) .* BS.get_ufc_flash_mc_efficiency.(e)
end;

# ╔═╡ 12000006-0000-4000-8000-000000000040
let
    specs = (
        ("120 kVp · both tubes (regular)", protocol_reg, :seagreen),
        ("100 kVp · tube A (DE)", protocol_low, :royalblue),
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
  independent — this is exactly what `sim_opts_b`'s separate seed buys).

Each tube runs the doctrine correction stack: knobless η-aware water
sinogram BHC → FDK → HU.
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
        sim_opts, protocol; scanner = scanner, geom = geom,
    )
    e2, w_col = BS.bhc_spectrum_per_column(e, ŵ)          # [n_E, n_col]
    w_col_η = w_col

    # Single mono-equivalent target = mean energy of the η-folded mean spectrum
    w_mean = vec(sum(w_col_η; dims = 2)) ./ size(w_col_η, 2)
    ref_E = sum(e2 .* w_mean) / sum(w_mean)

    # KNOBLESS water BHC from the per-column Flash-η spectrum.
    model = BS.calibrate_bhc_water(
        e2, w_col_η;
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
sim_reg_a = let
    @info "Simulating: 120 kVp / 420 mA (tube A, regular mode, Flash η folded)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_reg, sim_opts, recon_opts, phantom,
    )
    BS.simulate!(ws, phantom, protocol_reg, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 12000007-0000-4000-8000-000000000012
sim_reg_b = let
    @info "Simulating: 120 kVp / 420 mA (tube B, regular mode, independent seed)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_reg, sim_opts_b, recon_opts, phantom,
    )
    BS.simulate!(ws, phantom, protocol_reg, sim_opts_b)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 12000007-0000-4000-8000-000000000015
bhc_reg = flash_bhc_calibration(protocol_reg, sim_reg_a.geom);

# ╔═╡ 12000007-0000-4000-8000-000000000020
hu_reg = (
    a = flash_poly_recon(sim_reg_a.sino, sim_reg_a.geom, bhc_reg),
    b = flash_poly_recon(sim_reg_b.sino, sim_reg_b.geom, bhc_reg),
);

# ╔═╡ 12000007-0000-4000-8000-000000000025
# Dual power: the two independent same-kV acquisitions average into the
# combined image (image domain — per-ray sinogram averaging is equivalent
# to first order on the log data at these noise levels).
hu_reg_combined = 0.5f0 .* hu_reg.a .+ 0.5f0 .* hu_reg.b;

# ╔═╡ 12000007-0000-4000-8000-000000000030
dp_stats = let
    ERODE_PX = 12.0
    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool = BS.erode_mask_2d(
        mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER); erode_px = ERODE_PX,
    )
    sw_idx = findall(sw_bool)

    # Central noise ROI (12 px ≈ 8.2 mm at 0.683 mm/px) — background water.
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

    fig = Mke.Figure(size = (1400, 640))
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
    Mke.hlines!(ax2, [σ_pred]; color = :black, linewidth = 2, linestyle = :dash)
    for (k, σv) in enumerate(σs)
        Mke.text!(
            ax2, k, σv;
            text = "σ = $(round(σv, digits = 2)) HU",
            align = (:center, :bottom), fontsize = 16, offset = (0, 4),
        )
    end
    Mke.text!(
        ax2, n, σ_pred;
        text = "predicted single/√2 = $(round(σ_pred, digits = 2))",
        align = (:right, :top), fontsize = 16, offset = (0, -6),
    )

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_dual_power.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000008-0000-4000-8000-000000000001
md"""
## 7. DE Forward Project (one DE acquisition = two tube scans)

Each tube builds its own workspace (the Flash UFC η enters via the src
`detector_efficiency` pathway) and keeps only the noisy log-line-integral
sinogram + geometry.  Tube B runs on its **independent noise chain**
(`sim_opts_b`) and sees the phantom through the tube-B z-offset — which for
the Flash is unpublished and set to **0.0 mm** (documented assumption; the
mechanism mirrors nb09's Force −0.88 mm so a measured value can drop in).
"""

# ╔═╡ 12000008-0000-4000-8000-000000000005
begin
    # Tube-B z-offset: unpublished for the Flash → 0.0 mm documented
    # assumption (Force: −0.88 mm).  The mechanism stays in place.
    FLASH_TUBE_B_Z_OFFSET_MM = 0.0
    phantom_b = BS.Phantom(
        phantom.mask,
        phantom.materials,
        phantom.voxel_size,
        (phantom.origin[1], phantom.origin[2],
            phantom.origin[3] - FLASH_TUBE_B_Z_OFFSET_MM / 10.0),
        phantom.extent,
    )
end;

# ╔═╡ 12000008-0000-4000-8000-000000000020
sim_low = let
    @info "Simulating: 100 kVp / $(round(protocol_low.mA, digits = 1)) mA (tube A, Flash η folded)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_low, sim_opts, recon_opts, phantom,
    )
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    I0_scalar = BS.compute_detector_I0(ws.geom, protocol_low, sum(ws.weights)) * Float64(ws.η_eff)
    air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
        Array(ws.bowtie_air_reference)
    result = (sino = Array(ws.sinogram), geom = ws.geom,
        I0_ray = Float32.(I0_scalar .* Float64.(air_ref)))
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 12000008-0000-4000-8000-000000000030
sim_high = let
    @info "Simulating: Sn140 kVp / $(round(protocol_high.mA, digits = 1)) mA (tube B, Flash η folded, independent seed)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_high, sim_opts_b, recon_opts, phantom_b,
    )
    BS.simulate!(ws, phantom_b, protocol_high, sim_opts_b)
    I0_scalar = BS.compute_detector_I0(ws.geom, protocol_high, sum(ws.weights)) * Float64(ws.η_eff)
    air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
        Array(ws.bowtie_air_reference)
    result = (sino = Array(ws.sinogram), geom = ws.geom,
        I0_ray = Float32.(I0_scalar .* Float64.(air_ref)))
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 12000008-0000-4000-8000-000000000040
let
    n_row = size(sim_low.sino, 2)
    mid_r = n_row ÷ 2 + 1

    slice_lo = permutedims(sim_low.sino[:, mid_r, :], (2, 1))
    slice_hi = permutedims(sim_high.sino[:, mid_r, :], (2, 1))

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
        (1, 1, "100 kVp (tube A)", slice_lo),
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
Eusemann et al., SPIE 2008).  On the Flash's 100/Sn140 pair the historical
clinical default is w = 0.5 (later w ≈ 0.5–0.6).

So the poly validation of the Flash LUT runs the doctrine correction stack
**per tube** — η-aware water sinogram BHC → FDK → HU — then blends.  If the
η fold is right, solid water lands at ≈ 0 HU in *both* per-tube recons (and
therefore in any blend).
"""

# ╔═╡ 12000009-0000-4000-8000-000000000010
bhc_low = flash_bhc_calibration(protocol_low, sim_low.geom);

# ╔═╡ 12000009-0000-4000-8000-000000000012
bhc_high = flash_bhc_calibration(protocol_high, sim_high.geom);

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
    low = flash_poly_recon(sim_low.sino, sim_low.geom, bhc_low),
    high = flash_poly_recon(sim_high.sino, sim_high.geom, bhc_high),
);

# ╔═╡ 12000009-0000-4000-8000-000000000025
# Siemens linear mixed image: M = w·I_low + (1−w)·I_high (image domain,
# Yu 2009).  w = 0.5 is the Flash-era 100/Sn140 default.
MIX_W_LOW = 0.5f0;

# ╔═╡ 12000009-0000-4000-8000-000000000028
hu_mixed = MIX_W_LOW .* hu_tube.low .+ (1.0f0 - MIX_W_LOW) .* hu_tube.high;

# ╔═╡ 12000009-0000-4000-8000-000000000030
poly_water_stats = let
    ERODE_PX = 12.0
    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool = BS.erode_mask_2d(
        mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER); erode_px = ERODE_PX,
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
        (1, "100 kVp (tube A)", hu_tube.low),
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
        ("100 kVp", poly_water_stats.low),
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
## 9. Projection-Domain Material Decomposition (Cong)

Per-ray Cong univariate solver on the polychromatic transmission integral,
iodine + water material basis, running on the **raw noisy sinograms** (no
projection-domain denoising — the anti-correlated basis noise is handled
by cov-ACNR after FBP).  The per-ray spectral weights are
**source × flat filter × bowtie × Flash UFC η(E)** — the identical model
the forward projector applied, because `resolve_source_spectrum_full`
builds ŵ from the same `build_physics_config` (and therefore the same src
Flash LUT) that `simulate!` used.
"""

# ╔═╡ 1200000a-0000-4000-8000-000000000010
material_basis = let
    # Per-ray effective spectrum: source × filters × bowtie × src Flash η(E),
    # normalized per ray (Σ_E ŵ = 1) — the nb07 pattern.
    e_L, ŵ_L = BS.resolve_source_spectrum_full(
        sim_opts, protocol_low; scanner = scanner, geom = sim_low.geom,
        diagnostic = true, label = "low·Flash",
    )
    e_H, ŵ_H = BS.resolve_source_spectrum_full(
        sim_opts, protocol_high; scanner = scanner, geom = sim_high.geom,
        diagnostic = true, label = "high·Flash",
    )

    iodine_mat = BS.XA.Elements.Iodine
    water_mat = BS.XA.Materials.water

    p_L = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_L]
    q_L = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_L]
    p_H = Float32[Float32(BS.compute_mass_μ_at_energy(iodine_mat, Float64(E))) for E in e_H]
    q_H = Float32[Float32(BS.compute_mass_μ_at_energy(water_mat, Float64(E))) for E in e_H]

    (
        ŵ_L = ŵ_L, p_L = p_L, q_L = q_L,
        ŵ_H = ŵ_H, p_H = p_H, q_H = q_H,
    )
end;

# ╔═╡ 1200000a-0000-4000-8000-000000000020
sino_basis = let
    # First-order log-Poisson DEBIAS (deterministic; matches nb03).
    debias(p, I0_ray) = begin
        out = Float32.(p)
        nc, nr, nv = size(out)
        for v in 1:nv, r in 1:nr, c in 1:nc
            N = max(I0_ray[c, r] * exp(-out[c, r, v]), 1.0f0)
            out[c, r, v] -= 1.0f0 / (2.0f0 * N)
        end
        out
    end
    sino_low_gpu = to_gpu(debias(sim_low.sino, sim_low.I0_ray))
    sino_high_gpu = to_gpu(debias(sim_high.sino, sim_high.I0_ray))

    sino_y = similar(sino_low_gpu)   # iodine basis line integrals
    sino_c = similar(sino_low_gpu)   # water  basis line integrals
    fill!(sino_y, 0.0f0); fill!(sino_c, 0.0f0)

    cong_ws = BS.create_cong_workspace(sino_low_gpu, material_basis)
    BS.apply_cong!(
        cong_ws, sino_y, sino_c, sino_low_gpu, sino_high_gpu;
        water_basis = (a = 0.0f0, c = 1.0f0),
    )

    result = (
        sino_iodine = Array(sino_y),
        sino_water = Array(sino_c),
        geom = sim_low.geom,
    )
    sino_low_gpu = nothing; sino_high_gpu = nothing
    sino_y = nothing; sino_c = nothing; cong_ws = nothing
    GC.gc(true)
    result
end;

# ╔═╡ 1200000a-0000-4000-8000-000000000040
let
    n_row = size(sino_basis.sino_iodine, 2)
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

    slice_iod = permutedims(sino_basis.sino_iodine[:, mid_r, :], (2, 1))
    slice_wat = permutedims(sino_basis.sino_water[:, mid_r, :], (2, 1))

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

# ╔═╡ 1200000b-0000-4000-8000-000000000001
md"""
## 10. FBP × 2 + Kalender ACNR

Identical post-decomposition chain to nb03/nb09: two FDK passes
(`BS.SoftFilter()`) → image-domain data-adaptive cov-ACNR
(`BS.apply_acnr_kalender!`, per-pixel regression, zero blur).
"""

# ╔═╡ 1200000b-0000-4000-8000-000000000010
basis_volumes = let
    matrix_size = recon_opts.matrix_size
    geom = sino_basis.geom

    function _fbp(sino_cpu)
        sino_gpu = to_gpu(Float32.(sino_cpu))
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, geom, matrix_size; filter = BS.SoftFilter(),
        )
        recon = Array(BS.reconstruct!(ws, sino_gpu, geom))
        ws = nothing; sino_gpu = nothing
        GC.gc(true)
        return Float32.(recon)
    end

    (
        vol_iodine_raw = _fbp(sino_basis.sino_iodine),
        vol_water_raw = _fbp(sino_basis.sino_water),
        geom = geom,
    )
end;

# ╔═╡ 1200000b-0000-4000-8000-000000000020
basis_acnr = let
    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)

    info = BS.apply_acnr_kalender!(W, I)
    @info "[ACNR · Kalender-1988 true ACNR] ρ_hp(W,I)=$(round(info.ρ_hp, digits = 3))"

    (vol_iodine_raw = I, vol_water_raw = W, geom = basis_volumes.geom)
end;

# ╔═╡ 1200000b-0000-4000-8000-000000000040
let
    fig = Mke.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_acnr.vol_iodine_raw, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_acnr.vol_iodine_raw[:, :, mid]
    slice_wat = basis_acnr.vol_water_raw[:, :, mid]

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
## 11. VMI Synthesis

Textbook 2-basis mix (McCollough 2015) at 50 / 70 / 100 / 140 keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

The `solid_water_basis` diagnostic logs the basis-decomp residual bias as
a Δ% between the SW-ROI synth μ_water and the textbook mono divisor.
"""

# ╔═╡ 1200000c-0000-4000-8000-000000000010
solid_water_basis = let
    ERODE_PX = 12.0

    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool_raw = (mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)

    n_raw = count(sw_bool_raw); n_eroded = count(sw_bool)
    n_eroded == 0 && error(
        "solid_water_basis: deep erosion (σ = $(ERODE_PX) px) wiped out the SW " *
            "ROI (raw count = $(n_raw)).  Reduce erode_px or check phantom mask."
    )
    @info "solid_water_basis: SW mid-slice voxel count $(n_raw) → $(n_eroded) " *
        "after $(ERODE_PX)-px erosion"

    sw_idx = findall(sw_bool)
    n_z = size(basis_acnr.vol_water_raw, 3)
    function _mean(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    c_w = Float64(_mean(basis_acnr.vol_water_raw))
    c_i = Float64(_mean(basis_acnr.vol_iodine_raw))
    @info "solid_water_basis: ⟨c_water⟩_SW = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_SW = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * n_z,
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 1200000c-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 1200000c-0000-4000-8000-000000000020
vmi_HU_final = let
    # synth_vmi_2basis expects c_iodine in mg/mL; basis maps are g/cm³ (= g/mL)
    c_iodine_mg_per_mL = basis_acnr.vol_iodine_raw .* 1000.0f0

    out = Dict{Float64, Array{Float32, 3}}()
    for E in de_vmi_energies
        μρ_w = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
        μρ_I = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E)
        μ_water_anchor = solid_water_basis.c_water * μρ_w +
            solid_water_basis.c_iodine * μρ_I
        Δ_pct = 100.0 * (μ_water_anchor - μρ_w) / μρ_w
        @info "VMI synth @ $(Int(E)) keV: divisor = $(round(μρ_w, digits = 5)) cm⁻¹ " *
            "(mono μρ_water);  SW-ROI anchor = " *
            "$(round(μ_water_anchor, digits = 5)) → Δ = $(round(Δ_pct, digits = 2))%"

        out[E] = BS.synth_vmi_2basis(
            basis_acnr.vol_water_raw, c_iodine_mg_per_mL;
            energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 1200000c-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[50.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = Mke.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            aspect = Mke.DataAspect(), axis_kwargs...,
        )
        Mke.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[1:2, 3];
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
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
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
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
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

    sw_idx = findall(solid_water_basis.mask_2d)
    n_z = size(vmi_HU_final[70.0], 3)
    function _mean_hu(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end
    sw_hu_per_keV = [_mean_hu(vmi_HU_final[E]) for E in de_vmi_energies]

    n_E = length(de_vmi_energies)
    bar_colors = [Mke.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Water Region Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.barplot!(
        ax2, 1:n_E, sw_hu_per_keV;
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    Mke.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, h) in pairs(sw_hu_per_keV)
        Mke.text!(
            ax2, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4),
        )
    end

    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    Mke.ylims!(ax2, -y_max, y_max)

    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_water_roi.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000e-0000-4000-8000-000000000060
md"""
### Water-Region Noise

Mean and σ are both measured on the **deeply eroded solid-water region**
(the same 12-px-eroded mask as the accuracy ROI) — the certified nb03
convention.  The large region (~25k px × z) makes the 50-keV mean and the
per-keV σ estimates statistically stable; a small central circle is far
too underpowered when FBP noise is spatially correlated (a 441-px ROI at
σ ≈ 105 HU carries an effective ±10–15 HU standard error on its mean).
"""

# ╔═╡ 1200000e-0000-4000-8000-000000000080
vmi_noise_by_keV = let
    roi_idx = findall(solid_water_basis.mask_2d)
    nz_r = size(vmi_HU_final[70.0], 3)

    out = Dict{Float64, NamedTuple}()
    for E in de_vmi_energies
        vol = vmi_HU_final[E]
        vals = Float64[Float64(vol[ci, z]) for z in 1:nz_r, ci in roi_idx]
        μ = mean(vals); σ = std(vals)
        out[E] = (mean = μ, std = σ, n = length(vals))
        @info "water-region noise @ $(Int(E)) keV: ⟨HU⟩ = $(round(μ, digits = 2)),  σ = $(round(σ, digits = 2)) HU  (n = $(length(vals)))"
    end
    out
end;

# ╔═╡ 1200000e-0000-4000-8000-000000000090
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
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
    σs = [vmi_noise_by_keV[E].std  for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]

    ax2 = Mke.Axis(
        fig[1, 2];
        title = "Water-Region Noise vs Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.scatterlines!(
        ax2, Es, σs;
        color = :tomato, markersize = 18, linewidth = 3,
    )
    for (E, σ, μ) in zip(Es, σs, μs)
        Mke.text!(
            ax2, E, σ;
            text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 8),
        )
    end

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

Automated PASS/FAIL gates over both acquisition classes (nb01 convention):

1. **Regular water accuracy** — |⟨HU⟩| ≤ 5 in tube A, tube B, combined.
2. **Dual-power √2** — σ_combined / σ_single ∈ [0.62, 0.80]
   (ideal 0.707; fails if the tube seeds were ever shared).
3. **DE poly water accuracy** — |⟨HU⟩| ≤ 5 in low, high, mixed.
4. **VMI water accuracy** — |⟨HU⟩| ≤ 10 at every synthesized keV,
   measured on the deeply eroded solid-water region (nb03 convention).
5. **Monotonic VMI noise** — σ(50) > σ(70) > σ(100) > σ(140) on the same
   region (clinical truth; hard gate per nb03 doctrine).
6. **Per-rod regression** — measured-vs-theoretical slope ∈ [0.85, 1.15]
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
        "σ_comb/σ_single = $(round(dp_stats.noise_ratio, digits = 3)) (ideal 0.707, gate [0.62, 0.80])",
    ))

    # 3. DE poly water accuracy
    for (tag, s) in (("100 kVp", poly_water_stats.low), ("Sn140 kVp", poly_water_stats.high),
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

    # 5. Monotonic noise decrease with keV
    σs = [vmi_noise_by_keV[E].std for E in de_vmi_energies]
    push!(checks, (
        "VMI noise monotonic ↓", all(diff(σs) .< 0),
        "σ = " * join([string(round(σ, digits = 1)) for σ in σs], " > "),
    ))

    # 6. Per-rod regression gates
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
md"""
**Verification: $(verification.pass ? "PASS" : "FAIL") — $(verification.n_pass)/$(verification.n_total) gates**
"""

# ╔═╡ 12000011-0000-4000-8000-000000000001
md"""
## Summary

```
Flash UFC MC η(E) LUT (Khodajou-Chokami, Gd₂O₂S, 1–140 keV, 2026-08-26)
   → src pathway: Scanner(detector_material = :ufc_flash)
                → detector_efficiency_ufc_flash()
REGULAR: 120 kVp × 2 tubes (independent seeds)
   → per-tube η-aware BHC → FDK → HU → (A+B)/2
   → water ≈ 0 HU ×3, σ_combined ≈ σ_single/√2               (§6)
DUAL ENERGY: 100 kVp (A) + Sn140 kVp (B, 0.4 mm Sn)
   ├─→ POLY: per-tube η-aware BHC → FDK → HU → mixed image M_w  (§8)
   └─→ VMI:  Cong decomposition on raw sinograms (iodine + water,
             bowtie + η_Flash per ray)
             → FBP × 2 → cov-ACNR
             → VMI 50/70/100/140 keV
             → per-rod measured vs theoretical regression   (§9–11+)
   → automated PASS/FAIL verification over both classes
```

**What this notebook establishes:**

1. **The Flash is modeled as itself, not as a re-badged Force**: its own
   MC detector LUT (−28% η at 140 keV vs the Force), its published
   geometry (595/1085.6 mm, 64 × 736 @ 0.70473 mm iso), published anode
   angle (7°), published flat filtration (8.4 mm Al eq.), and its actual
   clinical DE pair (100/Sn140 with 0.4 mm Sn — no kV substitution
   needed, unlike the Force's Sn150).
2. **Both dual-source acquisition classes in one place**: the regular
   (dual-power) readout verifies accuracy + the √2 independence of the
   two tube chains; the DE readout verifies the poly/mixed chain and the
   full VMI chain through the Cong inversion on two very different
   detected spectra sitting on opposite sides of the Gd K-edge
   fluorescence-escape cliff.
3. **Documented assumptions are explicit** (§3): bowtie profile, crystal
   depth, fill factor, electronic noise, tube-B z-offset, z-FFS — the
   remaining gaps on the 1-1 parity checklist in
   `docs/scanner_dossiers/somatom_definition_flash.md`.

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
# ╟─12000005-0000-4000-8000-000000000001
# ╠═12000005-0000-4000-8000-000000000005
# ╠═12000005-0000-4000-8000-000000000010
# ╠═12000005-0000-4000-8000-000000000020
# ╟─12000006-0000-4000-8000-000000000001
# ╠═12000006-0000-4000-8000-000000000010
# ╠═12000006-0000-4000-8000-000000000012
# ╠═12000006-0000-4000-8000-000000000020
# ╠═12000006-0000-4000-8000-000000000030
# ╟─12000006-0000-4000-8000-000000000040
# ╟─12000007-0000-4000-8000-000000000001
# ╠═12000007-0000-4000-8000-000000000005
# ╠═12000007-0000-4000-8000-000000000008
# ╠═12000007-0000-4000-8000-000000000010
# ╠═12000007-0000-4000-8000-000000000012
# ╠═12000007-0000-4000-8000-000000000015
# ╠═12000007-0000-4000-8000-000000000020
# ╠═12000007-0000-4000-8000-000000000025
# ╠═12000007-0000-4000-8000-000000000030
# ╟─12000007-0000-4000-8000-000000000040
# ╟─12000008-0000-4000-8000-000000000001
# ╠═12000008-0000-4000-8000-000000000005
# ╠═12000008-0000-4000-8000-000000000020
# ╠═12000008-0000-4000-8000-000000000030
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
# ╠═1200000a-0000-4000-8000-000000000020
# ╟─1200000a-0000-4000-8000-000000000040
# ╟─1200000b-0000-4000-8000-000000000001
# ╠═1200000b-0000-4000-8000-000000000010
# ╠═1200000b-0000-4000-8000-000000000020
# ╟─1200000b-0000-4000-8000-000000000040
# ╟─1200000c-0000-4000-8000-000000000001
# ╠═1200000c-0000-4000-8000-000000000010
# ╠═1200000c-0000-4000-8000-000000000015
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
