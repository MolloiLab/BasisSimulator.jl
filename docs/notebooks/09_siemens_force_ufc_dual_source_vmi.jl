### A Pluto.jl notebook ###
# v0.2.1

using Markdown
using InteractiveUtils

# ╔═╡ 09000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 09000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 09000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 09000001-0000-4000-8000-000000000010
md"""
# 09 · Siemens SOMATOM Force · UFC MC LUT · Dual-Source VMI

The Siemens **SOMATOM Force** third-generation dual-source scanner with its
**UFC (Ultra-Fast Ceramic, Gd₂O₂S:Pr,Ce)** scintillator, modelled through the
package's Monte-Carlo detector-efficiency table `BS.UFC_MC_EFFICIENCY_LUT`.

One dual-source DE acquisition feeds **both** outputs, the routine-looking
mixed image and the spectral results:

```
UFC MC η(E) LUT  (BS.UFC_MC_EFFICIENCY_LUT, 1–140 keV)
        │ via EICTScanner(detector_material = :ufc)
        ▼
Simulate 100 kVp (tube A) ──┬─→ POLY: per-tube η-aware BHC → FDK → HU
Simulate Sn140 kVp (tube B)─┘         → Siemens-style mixed image M_w
        │                               (water-HU validation, §7)
        ▼
   BS.spectral_basis_from_acquisitions                           (§8)
   BS.vmi_pipeline: projection HYPR → K = 2 n-channel decomposition
        → image HYPR → ACNR → VMI 50/70/100/140 keV              (§9–10)
        → Per-Rod Measured vs Theoretical Regression
```

!!! note "Single-energy vs dual-energy on the Force"
    The Force is *not* inherently spectral: routine protocols run both
    tubes at the **same** kVp (dual source buys temporal resolution and
    power, not spectra) and DE is a selectable mode.  But within a DE
    acquisition there is no third "plain" scan — the routine-equivalent
    grayscale output is the **mixed image**, a weighted blend of the
    low-kV and high-kV reconstructions.  This notebook models the DE
    acquisition and derives both readouts from it.

!!! info "How the UFC table enters the simulation"
    `src/detector/detector_efficiency.jl` ships `UFC_MC_EFFICIENCY_LUT`,
    `get_ufc_mc_efficiency(E)` and the `detector_efficiency_ufc()`
    factory; `build_physics_config` dispatches `detector_material = :ufc`
    to it (as `:lumex` dispatches to the GE Gemstone table).  This notebook
    sets `EICTScanner(detector_material = :ufc)` with the default
    `use_detector_efficiency = true`, and the EICT forward model weights
    every energy by `w(E) · η_UFC(E) · exp(-∫μ dl)`.  The spectral basis and
    the BHC calibration resolve the *same* η-folded spectrum through
    `resolve_source_spectrum_full`, so the forward and inverse spectral
    models match exactly.
"""

# ╔═╡ 09000001-0000-4000-8000-000000000020
md"""
## Notebook Setup

Activate the shared docs environment, load the simulator and plotting stack,
detect the available compute backend, and build the notebook table of contents.
"""

# ╔═╡ 09000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 09000001-0000-4000-8000-000000000031
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 09000001-0000-4000-8000-000000000032
import PlutoUI

# ╔═╡ 09000001-0000-4000-8000-000000000033
PlutoUI.TableOfContents()

# ╔═╡ 09000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    AT = GPUSelect.Storage()     # the backend array type, directly: MtlArray / CuArray / ROCArray
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 09000001-0000-4000-8000-000000000050
md"""
**Backend detected:** $(GPU_BACKEND.name)
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
md"""
## 1. The UFC Monte-Carlo Efficiency LUT

Per-energy absorbed fraction η(E) for the Siemens UFC Gd₂O₂S scintillator,
from a full Monte-Carlo transport simulation of the SOMATOM Force
StellarInfinity detector.

**Provenance**: a Monte-Carlo transport simulation of the Force detector by
Hamidreza Khodajou-Chokami, PhD (UC Irvine Medical Imaging Laboratory),
`efficiency_results.csv`, 2026-06-08.  The table ships in the package as
`BS.UFC_MC_EFFICIENCY_LUT` (`src/detector/detector_efficiency.jl`): 140 values
on a 1-keV grid (1–140 keV), verbatim from that dataset, next to the GE
`BS.GEMSTONE_MC_EFFICIENCY_LUT`.

**Physics signatures** (same class of MC-only features the Gemstone LUT
captures — Beer-Lambert *cannot* model these):

1. **Gd K-edge fluorescence escape at 50.24 keV**: η drops
   `0.969 → 0.741` between 50 and 51 keV.  Just above the K-edge,
   photoabsorption produces Gd Kα fluorescence (~43 keV) that escapes the
   thin crystal, so the *deposited* fraction falls even though attenuation
   rises.  Beer-Lambert would predict the opposite jump.
2. **Gd L-edge structure near 7–8 keV** (L₃ 7.24 / L₂ 7.93 / L₁ 8.38 keV):
   the small dip at 8 keV.
3. **Gradual high-energy roll-off** (0.897 at 100 keV → 0.816 at 140 keV)
   from primary transmission + Compton escape.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000030
let
    Es = collect(1.0:0.25:140.0)
    η_ufc = BS.get_ufc_mc_efficiency.(Es)
    η_gem = BS.get_gemstone_mc_efficiency.(Es)

    fig = Mke.Figure(size = (1180, 580))
    ax = Mke.Axis(
        fig[1, 1];
        title = "MC Detector Efficiency",
        subtitle = "UFC (SOMATOM Force) vs Gemstone (GE Apex Elite)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Absorbed Fraction η(E)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    Mke.lines!(ax, Es, η_ufc; color = :crimson, linewidth = 3, label = "UFC Gd₂O₂S (BS.UFC_MC_EFFICIENCY_LUT)")
    Mke.lines!(ax, Es, η_gem; color = :steelblue, linewidth = 3, label = "Gemstone garnet (src)")

    Mke.vlines!(ax, [50.24]; color = :crimson, linestyle = :dash, linewidth = 1.5)
    Mke.text!(ax, 50.24, 0.70; text = "Gd K-edge\n50.2 keV", fontsize = 16, align = (:left, :top), offset = (4, 0))
    Mke.vlines!(ax, [52.0, 63.31]; color = :steelblue, linestyle = :dash, linewidth = 1.5)
    Mke.text!(ax, 63.31, 0.99; text = "Tb / Lu K-edges", fontsize = 16, align = (:left, :top), offset = (4, 0))

    Mke.ylims!(ax, 0.6, 1.02)
    Mke.axislegend(ax; position = :rt, framevisible = true, labelsize = 18)
    fig
end

# ╔═╡ 09000003-0000-4000-8000-000000000001
md"""
## 2. `Phantom`: Gammex Model 472
"""

# ╔═╡ 09000003-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 09000003-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 09000004-0000-4000-8000-000000000001
md"""
## 3. `EICTScanner`: Siemens SOMATOM Force

Third-generation dual-source: **two Vectron tubes + two StellarInfinity
UFC detectors at 95°** in the same gantry.  Spec sheet (sources below;
the geometry was measured from an actual clinical Force by Wang et al.):

| Parameter | Value | Source |
|-----------|-------|--------|
| Source–isocenter (SID) | 595.0 mm | Wang et al. 2021, arXiv:2001.09471 §IV |
| Source–detector (SDD)  | 1085.6 mm | Wang et al. 2021 |
| Tube A–B angular offset | 95° (z-offset 0.88 mm) | Siemens Force whitepaper; Wang et al. |
| Detector A channels | 920 (50 cm FOV) | Wang et al.; whitepaper §1.2 |
| Detector B channels | 640 (35.5 cm FOV) | Wang et al.; Flohr/Schmidt DE-DSCT chapter |
| Rows | 96 × 0.6 mm at iso (57.6 mm z-coverage) | whitepaper §1.2 |
| Column pitch at iso | 0.054°/ch ≈ 0.561 mm | Wang et al. (1.023 mm at detector / 1.825× mag) |
| Scintillator | UFC Gd₂O₂S:Pr,Ce, ρ ≈ 7.34 g/cm³ | Siemens UFC page; Rossner & Grabmaier 1991 |
| Tube (Vectron) | 2 × 120 kW, 70–150 kVp / 10 kV steps, ≤1300 mA | whitepaper §1.1; datasheet |
| Focal spots | 0.4×0.5 mm² smallest; ~0.8 / 1.2 mm nominal | whitepaper; Duan AAPM 2014 |
| Tin filter | 0.6 mm Sn on the high-kV tube (SnX modes) | Faby et al. Siemens DECT white paper; PMC12745887 |
| Rotation | 0.25 s min (0.5 s routine) | whitepaper |
| Projections | 1160 / rotation per focal-spot position | Flohr et al. Med Phys 2005 (Siemens family figure) |

!!! warning "Documented modeling assumptions (no public source exists)"
    - **Anode angle**: Vectron's angle is unpublished → IPEM **8°** spectrum
      (typical 7–9° CT anode).
    - **Flat filtration**: unpublished → **3.0 mm Al + 0.9 mm Ti**, the same
      Vectron-family stack this repo already uses for the Naeotom Alpha
      (nb08).  The 0.6 mm Sn is added on tube B only.
    - **Bowtie**: Siemens body bowtie shape is unpublished → CatSim
      **large-body** profile as stand-in (same convention as nb04/nb08).
    - **Scintillator thickness 1.4 mm / fill factor 0.9**: proprietary;
      thickness is inert here (η comes from the MC LUT, and the
      Beer-Lambert fallback is not used), fill factor auto-cancels in the
      air-scan calibration.
    - **Electronic noise = 0**: Stellar's TrueSignal ASIC has no published
      absolute noise figure; its design point is "electronic noise
      negligible vs quantum noise" (Duan et al. AJR 2013).

!!! info "Dual source → two co-registered scans"
    Exactly like nb03 models GE rapid-kVp switching as two sequential
    scans, the Force's two tubes are modeled as **two `CTProtocol`s run
    back-to-back on one `EICTScanner`** (identical detector geometry), each
    with its own noise seed:

    - **Detector arc**: tube B gets detector-A's 920-channel arc so the
      (low, high) sinogram pair is per-ray co-registered for the K = 2
      decomposition — physically defensible because the 33 cm Gammex body fits
      inside detector B's real 35.5 cm FOV, so no ray we use would be
      missing on the real detector B.
    - **95° in-plane tube offset**: both modeled scans run a full axial
      rotation on the same angle grid, which is exactly what the clinical
      rebinning produces when it aligns the B data onto the A grid — a
      constant angular offset has no effect on a full-rotation axial scan.
    - **0.88 mm tube-B z-offset**: modeled explicitly in §6 by shifting
      the phantom origin −0.88 mm in z for the tube-B scan (tube B images
      a z-shifted slab during the same rotation).  For the z-invariant
      Gammex 472 this is an exact no-op, but the mechanism is in place so
      z-varying phantoms (XCAT, QRM) inherit the real misalignment.
    - **DE-mode collimation**: the Force reads out 128 × 0.6 mm in DE mode;
      we use 4.8 mm (8 × 0.6 mm) — the thin-collimation equivalent that
      fits the 1 cm Gammex z-extent, same convention as nb03's 5 mm.
"""

# ╔═╡ 09000004-0000-4000-8000-000000000010
# Tube/detector A geometry — shared by both modeled tubes (see md above).
scanner = BS.EICTScanner(
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

    detector_material = :ufc,    # free-form tag; η comes from the MC LUT below
    detector_depth = 1.4,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,

    electronic_noise = 0,
    detection_gain = 10.0,
);

# ╔═╡ 09000005-0000-4000-8000-000000000001
md"""
## 4. Protocols: One Dual-Source DE Acquisition

The Force's clinical abdomen DE pairs are **x/Sn150** (x ∈ 70–100 kVp,
0.6 mm Sn on tube B).  The bundled IPEM spectra top out at **140 kVp**, so
this notebook runs the **100 / Sn140** pair — the same pair the
second-generation Definition Flash ran clinically (with 0.4 mm Sn; we keep
the Force's 0.6 mm).  Mean-energy separation is within ~2 keV of the
100/Sn150 target.

Tube currents follow the published Force abdomen 100/Sn150 reference
(190 mAs A / 95 mAs B at 0.5 s → 2:1).  This single acquisition feeds
**both** the §7 poly/mixed readout and the §8+ VMI readout — no separate
plain scan exists on the real scanner in DE mode.

| Tube | kVp | Filters | mA | views | rotation |
|------|-----|---------|----|-------|----------|
| A (low)  | 100   | 3 Al + 0.9 Ti | 380 | 1160 | 0.5 s |
| B (high) | 140   | 3 Al + 0.9 Ti + **0.6 Sn** | 190 | 1160 | 0.5 s |
"""

# ╔═╡ 09000005-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 100,
    mA = 380.0,
    views = 1160,
    rotation_time = 0.5,
    collimation_mm = 4.8,    # nominal beam width; axial cone guards are automatic
    anode_angle = 8,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 09000005-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 190.0,
    views = 1160,
    rotation_time = 0.5,
    collimation_mm = 4.8,
    anode_angle = 8,
    additional_filters = [("Ti", 0.9), ("Sn", 0.6)],
);

# ╔═╡ 09000006-0000-4000-8000-000000000001
md"""
## 5. `SimOptions` and `ReconOptions`

`use_detector_efficiency = true` (the `:eict` preset default) routes
through the **src UFC MC LUT**: `build_physics_config` sees
`EICTScanner(detector_material = :ufc)` and dispatches to
`detector_efficiency_ufc()`, so the EICT forward model weights every
energy by `w(E) · η_UFC(E)` and the detected flux (and therefore the
Poisson noise level) automatically reflects the UFC absorption.

**Tube B gets its own seed** (`sim_opts_b`): the two tube/detector chains
are physically independent, so their noise must be too.  Two acquisitions
drawn from one seed carry the same noise pattern, correlated between the
channels, which the decomposition would then amplify.

`use_heel_effect = false` keeps the forward spectral model exactly equal
to the η-folded response the spectral basis uses (heel is a small
row-direction effect; with 4.8 mm collimation at center it is negligible).
"""

# ╔═╡ 09000006-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    seed = 1234,               # tube A chain
    use_heel_effect = false,   # exact forward/inverse spectral match
    projector = :dd_fast,      # same DD physics, single-pass fused kernels.
);

# ╔═╡ 09000006-0000-4000-8000-000000000012
# Independent noise chain for tube B (identical physics; only the random
# stream differs).
sim_opts_b = BS.SimOptions(
    seed = 4321,               # tube B chain — must differ from tube A
    use_heel_effect = false,
    projector = :dd_fast,
);

# ╔═╡ 09000006-0000-4000-8000-000000000020
# Keep the intended centered 5 × 0.6 mm saved grid. The axial workspace
# automatically adds symmetric detector guard rows so peripheral voxels
# on both terminal slices retain measured cone-beam support.
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 5),
    fov_cm = 35.0,
    z_cm = 0.30,
);

# ╔═╡ 09000006-0000-4000-8000-000000000030
"""
    ufc_detected_spectrum(protocol) -> (e, w_eta)

Display helper: tube spectrum × flat filtration × protocol filters
(IPEM, absolute flux) with the src UFC MC η(E) folded in — what the
detector actually integrates (centered ray, no bowtie).
"""
function ufc_detected_spectrum(protocol)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        sim_opts, protocol; scanner = scanner,
    )
    return e, Float64.(w) .* BS.get_ufc_mc_efficiency.(e)
end;

# ╔═╡ 09000006-0000-4000-8000-000000000040
let
    specs = (
        ("100 kVp · tube A", protocol_low, :royalblue),
        ("Sn140 kVp · tube B", protocol_high, :crimson),
    )

    fig = Mke.Figure(size = (1180, 580))
    ax = Mke.Axis(
        fig[1, 1];
        title = "UFC-Detected Spectra",
        subtitle = "w(E) · η_UFC(E), normalized — Sn hardening on tube B",
        xlabel = "Energy (keV)",
        ylabel = "Relative Detected Fluence",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )

    for (label, prot, color) in specs
        e, wη = ufc_detected_spectrum(prot)
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_detected_spectra.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 09000007-0000-4000-8000-000000000001
md"""
## 6. Forward Project (one DE acquisition = two tube scans)

Each tube builds its own workspace (the UFC η enters through the
`detector_efficiency` pathway) and keeps what the spectral basis needs: the
noisy corrected log sinogram, its per-ray air counts `I0_ray` (detector air
count × bowtie air profile) and its per-ray detected spectrum from
`resolve_source_spectrum_full` (source × filtration × bowtie × η_UFC), the
same model the forward projector applied.  Tube B runs on its own noise
chain (`sim_opts_b`).

Tube B sees the phantom through a **−0.88 mm z-shifted origin** — the real
detector-B z-offset (Wang et al. 2021).  For the z-invariant Gammex this
changes nothing, but the mechanism mirrors the physical scanner so
z-varying phantoms inherit the misalignment (and any future z-rebinning
step has something real to correct).
"""

# ╔═╡ 09000007-0000-4000-8000-000000000005
# Tube-B view of the phantom: origin shifted by the real −0.88 mm detector
# z-offset (no-op for the z-invariant Gammex; see §6 md).
phantom_b = BS.Phantom(
    phantom.mask,
    phantom.materials,
    phantom.voxel_size,
    (phantom.origin[1], phantom.origin[2], phantom.origin[3] - 0.088),
    phantom.extent,
);

# ╔═╡ 09000007-0000-4000-8000-000000000020
sim_low = let
    @info "Simulating: 100 kVp / $(round(protocol_low.mA, digits = 1)) mA (tube A, UFC η folded)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_low, sim_opts, recon_opts, phantom,
    )
    try
        BS.simulate!(ws, phantom, protocol_low, sim_opts)
        I0_scalar = BS.compute_detector_I0(ws.geom, protocol_low, sum(ws.weights)) * Float64(ws.η_eff)
        air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
            Array(ws.bowtie_air_reference)
        energies, response = BS.resolve_source_spectrum_full(
            sim_opts, protocol_low; scanner = scanner, geom = ws.geom,
        )
        (sino = Array(ws.sinogram), geom = ws.geom,
            I0_ray = Float32.(I0_scalar .* Float64.(air_ref)),
            energies = Float64.(energies), response = Float32.(response))
    finally
        BS.release_backend!(ws)
    end
end;

# ╔═╡ 09000007-0000-4000-8000-000000000030
sim_high = let
    @info "Simulating: Sn140 kVp / $(round(protocol_high.mA, digits = 1)) mA (tube B, UFC η folded, z-offset −0.88 mm, own seed)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_high, sim_opts_b, recon_opts, phantom_b,
    )
    try
        BS.simulate!(ws, phantom_b, protocol_high, sim_opts_b)
        I0_scalar = BS.compute_detector_I0(ws.geom, protocol_high, sum(ws.weights)) * Float64(ws.η_eff)
        air_ref = ws.bowtie_air_reference === nothing ? ones(Float32, ws.geom.n_cols, ws.geom.n_rows) :
            Array(ws.bowtie_air_reference)
        energies, response = BS.resolve_source_spectrum_full(
            sim_opts_b, protocol_high; scanner = scanner, geom = ws.geom,
        )
        (sino = Array(ws.sinogram), geom = ws.geom,
            I0_ray = Float32.(I0_scalar .* Float64.(air_ref)),
            energies = Float64.(energies), response = Float32.(response))
    finally
        BS.release_backend!(ws)
    end
end;

# ╔═╡ 09000007-0000-4000-8000-000000000040
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

# ╔═╡ 09000008-0000-4000-8000-000000000001
md"""
## 7. Poly Readout: Per-Tube EICT Recon + Siemens Mixed Image

In DE mode the scanner's routine-equivalent grayscale output is the
**mixed image** — a linear image-domain blend of the two per-tube
reconstructions (Yu et al., *Med Phys* 2009: `M = w·I_low + (1−w)·I_high`;
Eusemann et al., SPIE 2008).  On Sn150-class pairs the clinical weight is
w ≈ 0.5–0.6 (Lenga et al., *Br J Radiol* 2021).

So the poly validation of the UFC LUT runs the current nb01 correction stack
**per tube** — η-aware water sinogram BHC → FDK → HU, with residual
cupping measured as QA — then blends. If the η fold is right, solid water
lands at ≈ 0 HU in *both* per-tube recons (and therefore in any blend).

!!! info "η-aware BHC"
    The BHC water polynomial must see the same detected spectrum the
    forward model used.  We resolve the bowtie-hardened per-column
    spectrum with the UFC η(E) already folded in
    (`resolve_source_spectrum_full`), and feed the low-level
    `calibrate_bhc_water(e, w_col)` — same per-column fit the
    high-level API performs, but with the UFC fold included.
"""

# ╔═╡ 09000008-0000-4000-8000-000000000005
"""
    ufc_bhc_calibration(protocol, geom)

η-aware per-tube BHC: bowtie-hardened per-column spectrum × UFC η(E) →
per-column two-material polynomial.  Returns `(model, μ_water, ref_E_keV)`.
"""
function ufc_bhc_calibration(protocol, geom)
    # resolve_source_spectrum_full folds bowtie AND the src UFC η(E)
    # (via the same build_physics_config the forward model used).
    e, ŵ = BS.resolve_source_spectrum_full(
        sim_opts, protocol; scanner = scanner, geom = geom,
    )
    e2, w_col = BS.bhc_spectrum_per_column(e, ŵ)          # [n_E, n_col]
    w_col_η = w_col

    # Single mono-equivalent target = mean energy of the η-folded mean spectrum
    w_mean = vec(sum(w_col_η; dims = 2)) ./ size(w_col_η, 2)
    ref_E = sum(e2 .* w_mean) / sum(w_mean)

    # KNOBLESS water BHC from the custom UFC-η per-column spectrum — zero
    # segmentation thresholds (the two-material bone pass is deprecated:
    # its 450–600 HU window misclassified dense iodine as bone).
    model = BS.calibrate_bhc_water(
        e2, w_col_η;
        reference_energy_keV = ref_E,
    )
    return (model = model, μ_water = model.μ_water_ref, ref_E_keV = model.reference_energy_keV)
end;

# ╔═╡ 09000008-0000-4000-8000-000000000008
"""
    ufc_poly_recon(sino_cpu, geom, bhc) -> Array{Float32, 3}

Doctrine correction stack for one tube: knobless water sino-BHC → FDK → HU.
"""
function ufc_poly_recon(sino_cpu, geom, bhc)
    matrix_size = recon_opts.matrix_size

    sino_gpu = to_gpu(sino_cpu)
    # Knobless water BHC: one sinogram-domain pass, no recon round-trip.
    sino_bhc = BS.apply_bhc_water(sino_gpu, bhc.model)
    sino_gpu = sino_bhc

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom)

    # (image-domain BHC + applied cupping removed — deprecated; cupping is a
    #  QA metric via measure_radial_cupping)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water))

    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing
    GC.gc(true)
    return hu
end;

# ╔═╡ 09000008-0000-4000-8000-000000000010
bhc_low = ufc_bhc_calibration(protocol_low, sim_low.geom);

# ╔═╡ 09000008-0000-4000-8000-000000000012
bhc_high = ufc_bhc_calibration(protocol_high, sim_high.geom);

# ╔═╡ 09000008-0000-4000-8000-000000000015
md"""
**Calibrated (η-aware):**
tube A ref energy = $(round(bhc_low.ref_E_keV, digits = 1)) keV ·
lac water = $(round(bhc_low.μ_water, digits = 5)) cm⁻¹ —
tube B ref energy = $(round(bhc_high.ref_E_keV, digits = 1)) keV ·
lac water = $(round(bhc_high.μ_water, digits = 5)) cm⁻¹
"""

# ╔═╡ 09000008-0000-4000-8000-000000000020
hu_tube = (
    low = ufc_poly_recon(sim_low.sino, sim_low.geom, bhc_low),
    high = ufc_poly_recon(sim_high.sino, sim_high.geom, bhc_high),
);

# ╔═╡ 09000008-0000-4000-8000-000000000025
# Siemens linear mixed image: M = w·I_low + (1−w)·I_high (image domain,
# Yu 2009).  w = 0.5 is the common Sn150-pair default (0.3–0.7 clinical).
MIX_W_LOW = 0.5f0;

# ╔═╡ 09000008-0000-4000-8000-000000000028
hu_mixed = MIX_W_LOW .* hu_tube.low .+ (1.0f0 - MIX_W_LOW) .* hu_tube.high;

# ╔═╡ 09000008-0000-4000-8000-000000000030
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
        @info "[poly · UFC] $(tag) SW ROI: ⟨HU⟩ = $(round(s.mean, digits = 2)), σ = $(round(s.std, digits = 2)) HU (n = $(s.n))"
    end
    (stats..., mask_2d = collect(sw_bool))
end;

# ╔═╡ 09000008-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)
    mid = size(hu_mixed, 3) ÷ 2 + 1

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
        joinpath(@__DIR__, "..", "assets", "force_ufc_poly_recon.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 09000008-0000-4000-8000-000000000050
md"""
### Poly Water Values

Water-HU validation of the UFC LUT in the plain EICT chain, **before**
any spectral decomposition: solid-water ⟨HU⟩ ± σ for each per-tube recon
and the mixed image.  All three should cluster at ≈ 0 HU.
"""

# ╔═╡ 09000008-0000-4000-8000-000000000060
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
    mid = size(hu_mixed, 3) ÷ 2 + 1
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_poly_water_values.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000a-0000-4000-8000-000000000001
md"""
## 8. Spectral Basis from the Two Tubes

`spectral_basis_from_acquisitions` merges the two tubes' energy grids onto
their union and scales each tube's per-ray detected spectrum by its own air
counts, so the likelihood sees the absolute response ``\Phi_k(E)`` of every
ray and channel: **source × flat filters × bowtie × UFC η(E)**, the identical
model the forward projector applied (`resolve_source_spectrum_full` builds it
from the same `build_physics_config`, and therefore the same UFC table, that
`simulate!` used).  No calibration scan is involved.
"""

# ╔═╡ 0900000a-0000-4000-8000-000000000010
basis = BS.spectral_basis_from_acquisitions(acquisitions = [
    (energies = s.energies, response = s.response, I0_ray = s.I0_ray)
    for s in (sim_low, sim_high)
]);

# ╔═╡ 0900000a-0000-4000-8000-000000000015
Markdown.parse("""
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point
energy grid for $(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response
sums to the air counts to within $(round(basis.I0_relerr, sigdigits = 2))
(relative).
""")

# ╔═╡ 0900000b-0000-4000-8000-000000000001
md"""
## 9. The VMI Chain: `vmi_pipeline`

One package call from the two corrected sinograms to the VMI stack, the chain
of the basis-vmi and basis-spectral-denoising papers:

1. **Projection HYPR-LR** (`BS.SpectralHYPR`'s projection instance): a 3 × 3
   (column × view) window on the counts of each detector row, with each
   tube's dispersion measured from its own air rays.
2. **K = 2 n-channel decomposition**: per ray, the Poisson maximum-likelihood
   iodine + water pair under the exact polychromatic mean of both tubes.
3. **Image HYPR** on the FDK-reconstructed basis pair (a 3 × 3 × 7 composite
   and a 15 × 15 × 7 complement window), then **Kalender ACNR**.
4. **VMI synthesis** at 50 / 70 / 100 / 140 keV from the one basis pair.

The FBP kernel is the notebook's soft-tissue kernel, `BS.SoftFilter()`; the
grid is `recon_opts.matrix_size` with every detector row kept.
"""

# ╔═╡ 0900000b-0000-4000-8000-000000000005
HYPR_CHAIN = BS.SpectralHYPR(
    projection = BS.ProjectionHYPR(kernel = BS.HYPRKernel((3, 3), BS.BoxProfile())),
    image = BS.ImageHYPR(
        composite = BS.HYPRKernel((3, 3, 7), BS.BoxProfile(); linear = false),
        complement = BS.HYPRKernel((15, 15, 7), BS.BoxProfile(); linear = false),
    ),
)

# ╔═╡ 0900000b-0000-4000-8000-000000000006
VMI_CHAIN = (method = :nchannel, controls = BS.NChannelControls(), use_tlbf = false, antialias = true);

# ╔═╡ 0900000c-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 0900000b-0000-4000-8000-000000000010
de_vmi = BS.vmi_pipeline(;
    channels = [sim_low.sino, sim_high.sino],
    basis,
    geom = sim_low.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    fbp_filter = BS.SoftFilter(),
    denoiser = HYPR_CHAIN,
    use_acnr = true,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 0900000b-0000-4000-8000-000000000015
let
    q = de_vmi.quality
    d = de_vmi.settings.denoiser
    pct(x) = round(100x, digits = 3)
    Markdown.parse("""
    The decomposition solved $(q.n_rays) rays in $(round(de_vmi.elapsed_s, digits = 1)) s
    ($(round(q.outer_mean, digits = 1)) outer iterations on average); $(pct(q.frac_not_converged))% did not
    converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions (variance / mean of the counts on the air rays) of
    $(join(round.(d.dispersion, digits = 2), " and ")) for tube A and tube B. The image HYPR's
    minimum-noise composite energy is E* = $(round(Int, d.image_estimates.Estar)) keV.
    """)
end

# ╔═╡ 0900000a-0000-4000-8000-000000000040
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

# ╔═╡ 0900000b-0000-4000-8000-000000000040
let
    fig = Mke.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(de_vmi.images.iodine, 3) ÷ 2 + 1

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

# ╔═╡ 0900000c-0000-4000-8000-000000000001
md"""
## 10. VMIs

`vmi_pipeline` synthesizes each VMI from the basis pair (McCollough 2015):

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

The `solid_water_basis` diagnostic reports the basis pair in the eroded
solid-water region: a perfect decomposition reads water density ≈ 1 g/cm³
(solid water is not pure water, so a small offset is expected) and iodine ≈ 0.
"""

# ╔═╡ 0900000c-0000-4000-8000-000000000010
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

# ╔═╡ 0900000c-0000-4000-8000-000000000020
# The VMI stack as one volume per energy (keV → (nx, ny, nz) HU).
vmi_HU_final = Dict(
    Float64(E) => de_vmi.vmis[:, :, :, k] for (k, E) in pairs(de_vmi.energies)
);

# ╔═╡ 0900000c-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = Mke.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(de_vmi.vmis, 3) ÷ 2 + 1

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
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000e-0000-4000-8000-000000000001
md"""
## Results

Per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV.

!!! info "Methodology"
    - **Measured HU** = mean over an 8-px-radius circular ROI at the rod
      centroid, broadcast across all z slices.
    - **Theoretical HU** = `1000 · (μ_r(E) − μ_water(E)) / μ_water(E)` from
      `BS.compute_μ_at_energy` — pure physics, no fitting.
"""

# ╔═╡ 0900000e-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0900000e-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = ("50 mg/mL", "100 mg/mL", "200 mg/mL", "300 mg/mL", "400 mg/mL", "500 mg/mL", "600 mg/mL"),
    I = ("2.0 mg/mL", "2.5 mg/mL", "5.0 mg/mL", "7.5 mg/mL", "10.0 mg/mL", "15.0 mg/mL", "20.0 mg/mL"),
);

# ╔═╡ 0900000e-0000-4000-8000-000000000030
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

# ╔═╡ 0900000e-0000-4000-8000-000000000040
md"""
### Water ROI
"""

# ╔═╡ 0900000e-0000-4000-8000-000000000050
let
    fig = Mke.Figure(size = (1180, 580))

    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_water_roi.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000e-0000-4000-8000-000000000060
md"""
### Water-Region Noise
"""

# ╔═╡ 0900000e-0000-4000-8000-000000000065
const WATER_NOISE_ROI_RADIUS_PX = 12;   # ≈8.2 mm at 0.683 mm/px

# ╔═╡ 0900000e-0000-4000-8000-000000000070
water_noise_roi = let
    nx_r, ny_r, nz_r = size(de_vmi.images.water)
    cx = nx_r ÷ 2 + 1
    cy = ny_r ÷ 2 + 1

    roi_bool = falses(nx_r, ny_r)
    r² = Float64(WATER_NOISE_ROI_RADIUS_PX)^2
    @inbounds for j in 1:ny_r, i in 1:nx_r
        ((i - cx)^2 + (j - cy)^2) ≤ r² && (roi_bool[i, j] = true)
    end

    n_vox = count(roi_bool)
    @info "water_noise_roi: center = ($(cx), $(cy)), radius = $(WATER_NOISE_ROI_RADIUS_PX) px, " *
        "$(n_vox) vx × $(nz_r) z = $(n_vox * nz_r) total"

    (
        center_xy = (Float64(cx), Float64(cy)), mask_2d = roi_bool,
        n_voxels = n_vox, n_total = n_vox * nz_r,
    )
end;

# ╔═╡ 0900000e-0000-4000-8000-000000000080
vmi_noise_by_keV = let
    roi_idx = findall(water_noise_roi.mask_2d)
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

# ╔═╡ 0900000e-0000-4000-8000-000000000090
let
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in water_noise_roi.mask_2d]

    fig = Mke.Figure(size = (1180, 580))

    ax1 = Mke.Axis(
        fig[1, 1];
        title = "Water-Region Noise ROI",
        subtitle = "Overlaid on 70 keV VMI",
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_water_noise.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000f-0000-4000-8000-000000000001
md"""
### Per-Rod Regression
"""

# ╔═╡ 0900000f-0000-4000-8000-000000000010
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_vs_theoretical.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000f-0000-4000-8000-000000000020
md"""
### Linear Regression
"""

# ╔═╡ 0900000f-0000-4000-8000-000000000030
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 09000010-0000-4000-8000-000000000001
md"""
## Summary

```
UFC MC η(E) LUT (BS.UFC_MC_EFFICIENCY_LUT, Gd₂O₂S, 1–140 keV)
   → EICTScanner(detector_material = :ufc) → detector_efficiency_ufc()
Simulate 100 kVp + Sn140 kVp   (one SOMATOM Force dual-source DE acquisition,
                                one noise seed per tube)
   ├─→ POLY: per-tube η-aware BHC → FDK → HU → mixed image M_w     (§7)
   └─→ VMI:  BS.spectral_basis_from_acquisitions (bowtie + η_UFC per ray)
             → BS.vmi_pipeline(; denoiser = HYPR_CHAIN, use_acnr = true):
               projection HYPR → K = 2 n-channel decomposition
               → image HYPR → ACNR → VMI 50/70/100/140 keV      (§8–10)
             → Per-Rod Measured vs Theoretical Regression
```

**What validates the UFC table here:**

1. **§7 poly**: solid water in the 100 kVp recon, the Sn140 recon and the
   Siemens-style mixed image, each under its own η-aware BHC: the table's
   spectral shape is consistent with the detected sinogram on two very
   different tube spectra independently.
2. **§8–10 VMI**: water-region HU at every keV and the per-rod
   measured-vs-theoretical overlays: the table survives the harsher test of
   per-ray spectral inversion across the two detected spectra (100 kVp vs
   Sn140 kVp, separated by the 0.6 mm tin filter and sitting on opposite
   sides of the Gd K-edge fluorescence-escape cliff).
"""

# ╔═╡ Cell order:
# ╠═09000001-0000-4000-8000-000000000001
# ╠═09000001-0000-4000-8000-000000000002
# ╠═09000001-0000-4000-8000-000000000003
# ╟─09000001-0000-4000-8000-000000000010
# ╟─09000001-0000-4000-8000-000000000020
# ╠═09000001-0000-4000-8000-000000000030
# ╠═09000001-0000-4000-8000-000000000031
# ╠═09000001-0000-4000-8000-000000000032
# ╠═09000001-0000-4000-8000-000000000033
# ╠═09000001-0000-4000-8000-000000000040
# ╟─09000001-0000-4000-8000-000000000050
# ╟─09000002-0000-4000-8000-000000000001
# ╟─09000002-0000-4000-8000-000000000030
# ╟─09000003-0000-4000-8000-000000000001
# ╠═09000003-0000-4000-8000-000000000010
# ╠═09000003-0000-4000-8000-000000000020
# ╟─09000004-0000-4000-8000-000000000001
# ╠═09000004-0000-4000-8000-000000000010
# ╟─09000005-0000-4000-8000-000000000001
# ╠═09000005-0000-4000-8000-000000000010
# ╠═09000005-0000-4000-8000-000000000020
# ╟─09000006-0000-4000-8000-000000000001
# ╠═09000006-0000-4000-8000-000000000010
# ╠═09000006-0000-4000-8000-000000000012
# ╠═09000006-0000-4000-8000-000000000020
# ╠═09000006-0000-4000-8000-000000000030
# ╟─09000006-0000-4000-8000-000000000040
# ╟─09000007-0000-4000-8000-000000000001
# ╠═09000007-0000-4000-8000-000000000005
# ╠═09000007-0000-4000-8000-000000000020
# ╠═09000007-0000-4000-8000-000000000030
# ╟─09000007-0000-4000-8000-000000000040
# ╟─09000008-0000-4000-8000-000000000001
# ╠═09000008-0000-4000-8000-000000000005
# ╠═09000008-0000-4000-8000-000000000008
# ╠═09000008-0000-4000-8000-000000000010
# ╠═09000008-0000-4000-8000-000000000012
# ╟─09000008-0000-4000-8000-000000000015
# ╠═09000008-0000-4000-8000-000000000020
# ╠═09000008-0000-4000-8000-000000000025
# ╠═09000008-0000-4000-8000-000000000028
# ╠═09000008-0000-4000-8000-000000000030
# ╟─09000008-0000-4000-8000-000000000040
# ╟─09000008-0000-4000-8000-000000000050
# ╟─09000008-0000-4000-8000-000000000060
# ╟─0900000a-0000-4000-8000-000000000001
# ╠═0900000a-0000-4000-8000-000000000010
# ╟─0900000a-0000-4000-8000-000000000015
# ╟─0900000b-0000-4000-8000-000000000001
# ╠═0900000b-0000-4000-8000-000000000005
# ╠═0900000b-0000-4000-8000-000000000006
# ╠═0900000c-0000-4000-8000-000000000015
# ╠═0900000b-0000-4000-8000-000000000010
# ╟─0900000b-0000-4000-8000-000000000015
# ╟─0900000a-0000-4000-8000-000000000040
# ╟─0900000b-0000-4000-8000-000000000040
# ╟─0900000c-0000-4000-8000-000000000001
# ╠═0900000c-0000-4000-8000-000000000010
# ╠═0900000c-0000-4000-8000-000000000020
# ╟─0900000c-0000-4000-8000-000000000040
# ╟─0900000e-0000-4000-8000-000000000001
# ╠═0900000e-0000-4000-8000-000000000010
# ╠═0900000e-0000-4000-8000-000000000020
# ╠═0900000e-0000-4000-8000-000000000030
# ╟─0900000e-0000-4000-8000-000000000040
# ╟─0900000e-0000-4000-8000-000000000050
# ╟─0900000e-0000-4000-8000-000000000060
# ╠═0900000e-0000-4000-8000-000000000065
# ╠═0900000e-0000-4000-8000-000000000070
# ╠═0900000e-0000-4000-8000-000000000080
# ╟─0900000e-0000-4000-8000-000000000090
# ╟─0900000f-0000-4000-8000-000000000001
# ╟─0900000f-0000-4000-8000-000000000010
# ╟─0900000f-0000-4000-8000-000000000020
# ╟─0900000f-0000-4000-8000-000000000030
# ╟─09000010-0000-4000-8000-000000000001
