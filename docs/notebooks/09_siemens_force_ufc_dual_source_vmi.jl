### A Pluto.jl notebook ###
# v0.19.0

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

Standalone validation of a **new EICT Monte-Carlo detector-response LUT**
for the Siemens **UFC (Ultra-Fast Ceramic, Gd₂O₂S:Pr,Ce)** scintillator on
the **SOMATOM Force** third-generation dual-source scanner — *before* it is
promoted into `src/` alongside the GE Gemstone MC LUT.

One dual-source DE acquisition feeds **both** outputs — exactly like a
PCCT scan can be read out as a routine-looking image *or* as spectral
results:

```
UFC MC η(E) LUT  (Khodajou-Chokami MC, 1–140 keV)
        │ folded into the source spectrum  (spectrum_override)
        ▼
Simulate 100 kVp (tube A) ──┬─→ POLY: per-tube η-aware BHC → FDK → HU
Simulate Sn140 kVp (tube B)─┘         → Siemens-style mixed image M_w
        │                               (water-HU validation, §7)
        ▼
   Cong Decomp (raw sinograms) → FBP × 2
        → cov-ACNR → z-median → VMI 50/70/100/140 → Mono+
        → Per-Rod Measured vs Theoretical Regression          (§8–11)
```

!!! note "Single-energy vs dual-energy on the Force"
    The Force is *not* inherently spectral: routine protocols run both
    tubes at the **same** kVp (dual source buys temporal resolution and
    power, not spectra) and DE is a selectable mode.  But within a DE
    acquisition there is no third "plain" scan — the routine-equivalent
    grayscale output is the **mixed image**, a weighted blend of the
    low-kV and high-kV reconstructions.  This notebook models the DE
    acquisition and derives both readouts from it.

!!! info "How the UFC LUT enters the pipeline (no src changes)"
    The EICT forward model weights every energy bin by
    `w(E) · η(E) · exp(-∫μ dl)`.  Instead of routing through
    `SimOptions(use_detector_efficiency = true)` (which dispatches to the
    **Gemstone** MC LUT and asserts a `:lumex` detector), this notebook:

    1. sets `use_detector_efficiency = false`, and
    2. passes `spectrum_override = (e, w .* η_UFC.(e))` into
       `create_eict_workspace` — the UFC absorbed fraction is folded
       directly into the source weights.

    The two are algebraically identical inside `simulate!` (same `w·η`
    product, same detected-flux `I₀ = Σ w·η`), and the same η-folded
    weights drive the Cong basis and the BHC calibration — so the forward
    and inverse spectral models match *exactly*.

!!! success "Once validated here → src integration"
    The promotion path mirrors Gemstone 1:1: a `UFC_MC_EFFICIENCY_LUT`
    const + `get_ufc_mc_efficiency(E)` in
    `src/detector/detector_efficiency.jl`, a `detector_efficiency_ufc()`
    factory, a `:ufc` material branch in `compute_eid_efficiency_vector`,
    and a relaxed material assert in `build_physics_config`.
"""

# ╔═╡ 09000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 09000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 09000001-0000-4000-8000-000000000031
import CairoMakie as CM

# ╔═╡ 09000001-0000-4000-8000-000000000040
begin
    GPU_BACKEND = let
        candidates = [
            (:Metal, "dde4c033-4e86-420c-a63e-0dd931031962", :MtlArray),
            (:CUDA, "052768ef-5323-5732-b1bb-66c8b64840ba", :CuArray),
            (:AMDGPU, "21141c5a-9bdb-4563-92ae-f87d6854732e", :ROCArray),
        ]
        detected = (name = "CPU", to_gpu = identity)
        for (pkg, uuid, ctor) in candidates
            pkg_id = Base.PkgId(Base.UUID(uuid), String(pkg))
            Base.locate_package(pkg_id) === nothing && continue
            try
                m = Base.require(pkg_id)
                if Base.invokelatest(getfield(m, :functional))
                    detected = (name = string(pkg), to_gpu = getfield(m, ctor))
                    break
                end
            catch
            end
        end
        detected
    end

    to_gpu(x) = GPU_BACKEND.to_gpu(x)
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

**Provenance**: Hamidreza Khodajou-Chokami, PhD (UC Irvine Medical Imaging
Laboratory), `efficiency_results.csv`, received 2026-06-08.  The 140 values
below are verbatim from that CSV (1-keV grid, 1–140 keV); an archival copy
lives at `docs/notebooks/data/ufc_mc_efficiency_v1.csv` (gitignored — this
inline vector is the tracked copy, same convention as the inline
`GEMSTONE_MC_EFFICIENCY_LUT` in src).

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

# ╔═╡ 09000002-0000-4000-8000-000000000010
# Verbatim from ufc_mc_efficiency_v1.csv (Khodajou-Chokami MC, 2026-06-08).
# 1-keV grid: UFC_MC_EFFICIENCY_LUT.efficiency[k] is η at k keV.
const UFC_MC_EFFICIENCY_LUT = (
    energies = collect(1.0:1.0:140.0),
    efficiency = Float64[
        9.90863305e-01, 9.90798712e-01, 9.90207366e-01, 9.89968109e-01, 9.89353993e-01,
        9.88877621e-01, 9.88764364e-01, 9.69360328e-01, 9.72729277e-01, 9.76089134e-01,
        9.78741846e-01, 9.80889490e-01, 9.82528622e-01, 9.83559147e-01, 9.84355716e-01,
        9.84992948e-01, 9.85387625e-01, 9.85860336e-01, 9.86036647e-01, 9.86192298e-01,
        9.86036424e-01, 9.86016013e-01, 9.85866718e-01, 9.85760036e-01, 9.85619041e-01,
        9.85427871e-01, 9.85144176e-01, 9.84997524e-01, 9.84686433e-01, 9.84226060e-01,
        9.83886432e-01, 9.83888151e-01, 9.83530746e-01, 9.82826388e-01, 9.82299763e-01,
        9.81754579e-01, 9.81145416e-01, 9.80406222e-01, 9.79366721e-01, 9.78673909e-01,
        9.78040413e-01, 9.77222704e-01, 9.75291071e-01, 9.74675907e-01, 9.73959516e-01,
        9.72894908e-01, 9.71301325e-01, 9.70168735e-01, 9.69081131e-01, 9.69026725e-01,
        7.41154644e-01, 7.47765810e-01, 7.54996954e-01, 7.61970239e-01, 7.68339996e-01,
        7.74502998e-01, 7.80637525e-01, 7.85478337e-01, 7.91350826e-01, 7.97260988e-01,
        8.01716927e-01, 8.06821449e-01, 8.12187403e-01, 8.16816937e-01, 8.21093164e-01,
        8.25750637e-01, 8.29601348e-01, 8.33439150e-01, 8.36683997e-01, 8.40730609e-01,
        8.44150931e-01, 8.47429296e-01, 8.50435158e-01, 8.53887225e-01, 8.56827475e-01,
        8.59963243e-01, 8.62840135e-01, 8.65738358e-01, 8.68314163e-01, 8.70655876e-01,
        8.72908866e-01, 8.74932562e-01, 8.77096478e-01, 8.78888967e-01, 8.80993084e-01,
        8.82948845e-01, 8.84599046e-01, 8.86221492e-01, 8.87676325e-01, 8.88783948e-01,
        8.90449254e-01, 8.91856769e-01, 8.92648690e-01, 8.93756963e-01, 8.94433775e-01,
        8.95239544e-01, 8.95655380e-01, 8.96253439e-01, 8.96718581e-01, 8.96705367e-01,
        8.96467303e-01, 8.96790377e-01, 8.96711472e-01, 8.96650987e-01, 8.96839385e-01,
        8.96598287e-01, 8.96387397e-01, 8.95719919e-01, 8.94503447e-01, 8.93412046e-01,
        8.92210714e-01, 8.90899489e-01, 8.89779601e-01, 8.88385073e-01, 8.87015853e-01,
        8.85130923e-01, 8.83793476e-01, 8.82173648e-01, 8.80395144e-01, 8.78167359e-01,
        8.76309315e-01, 8.73851763e-01, 8.71613318e-01, 8.69143288e-01, 8.66828531e-01,
        8.64327296e-01, 8.61391429e-01, 8.58218989e-01, 8.55394561e-01, 8.52652873e-01,
        8.49222638e-01, 8.45710682e-01, 8.41841634e-01, 8.37917905e-01, 8.34416213e-01,
        8.30333026e-01, 8.26984885e-01, 8.23573340e-01, 8.20082743e-01, 8.15971281e-01,
    ],
);

# ╔═╡ 09000002-0000-4000-8000-000000000020
"""
    ufc_mc_efficiency(E_keV) -> Float64

UFC MC LUT lookup with linear interpolation on the 1-keV grid (clamped
to 1–140 keV).  Mirror of `BS.get_gemstone_mc_efficiency`.
"""
function ufc_mc_efficiency(E_keV::Real)
    η = UFC_MC_EFFICIENCY_LUT.efficiency
    E = clamp(Float64(E_keV), 1.0, 140.0)
    i = clamp(floor(Int, E), 1, 139)
    t = E - i
    return η[i] * (1.0 - t) + η[i + 1] * t
end;

# ╔═╡ 09000002-0000-4000-8000-000000000030
let
    Es = collect(1.0:0.25:140.0)
    η_ufc = ufc_mc_efficiency.(Es)
    η_gem = BS.get_gemstone_mc_efficiency.(Es)

    fig = CM.Figure(size = (1180, 580))
    ax = CM.Axis(
        fig[1, 1];
        title = "MC Detector Efficiency",
        subtitle = "UFC (SOMATOM Force) vs Gemstone (GE Apex Elite)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Absorbed Fraction η(E)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.lines!(ax, Es, η_ufc; color = :crimson, linewidth = 3, label = "UFC Gd₂O₂S (this notebook)")
    CM.lines!(ax, Es, η_gem; color = :steelblue, linewidth = 3, label = "Gemstone garnet (src)")

    CM.vlines!(ax, [50.24]; color = :crimson, linestyle = :dash, linewidth = 1.5)
    CM.text!(ax, 50.24, 0.70; text = "Gd K-edge\n50.2 keV", fontsize = 16, align = (:left, :top), offset = (4, 0))
    CM.vlines!(ax, [52.0, 63.31]; color = :steelblue, linestyle = :dash, linewidth = 1.5)
    CM.text!(ax, 63.31, 0.99; text = "Tb / Lu K-edges", fontsize = 16, align = (:left, :top), offset = (4, 0))

    CM.ylims!(ax, 0.6, 1.02)
    CM.axislegend(ax; position = :rt, framevisible = true, labelsize = 18)
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
## 3. `Scanner`: Siemens SOMATOM Force

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
    scans, the Force's two tubes are modeled as **two `Scanner` +
    `CTProtocol` configs run back-to-back** on identical detector geometry:

    - **Detector arc**: tube B gets detector-A's 920-channel arc so the
      (low, high) sinogram pair is per-ray co-registered for the Cong
      solver — physically defensible because the 33 cm Gammex body fits
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
scanner = BS.Scanner(
    source_to_isocenter = 595.0,
    source_to_detector = 1085.6,

    detector_rows = 96,
    detector_cols = 920,
    detector_row_size = 0.6,
    detector_col_size = 0.561,
    detector_shape = BS.CURVED_DETECTOR,

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
    collimation_mm = 4.8,    # 8 × 0.6 mm rows
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
## 5. `SimOptions`, `ReconOptions`, and the UFC Spectrum Fold

`use_detector_efficiency = false` keeps the src Gemstone LUT out of the
loop; the UFC η(E) is folded into the **absolute-flux source weights**
instead and handed to `create_eict_workspace` via `spectrum_override`.
Inside `simulate!` the math is identical (`I = Σ w·η·e^{-∫μ}`, detected
`I₀ = Σ w·η`), so the Poisson noise level automatically reflects the
UFC-detected flux.

`use_heel_effect = false` keeps the forward spectral model exactly equal
to the η-folded basis the Cong inversion uses (heel is a small
row-direction effect; with 5 mm collimation at center it is negligible).
"""

# ╔═╡ 09000006-0000-4000-8000-000000000010
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
    use_detector_efficiency = false,   # UFC η folded into spectrum_override instead
    use_heel_effect = false,           # exact forward/inverse spectral match
);

# ╔═╡ 09000006-0000-4000-8000-000000000020
# Cone-beam usable-z budget (same constraint nb07/nb08 respect): voxels
# near the source at slice-edge z fall off the detector unless
#   z_recon ≤ collimation × (1 − R_body/SID) = 4.8 × (1 − 165/595) ≈ 3.5 mm.
# Reconstructing the full 4.8 mm would put the two edge slices outside
# coverage (they read ~0.37× — verified during bring-up).  5 × 0.6 mm
# slices = 3.0 mm sits safely inside the budget.
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 5),
    fov_cm = 35.0,
    z_cm = 0.30,
);

# ╔═╡ 09000006-0000-4000-8000-000000000030
"""
    ufc_spectrum(protocol) -> (e, w_eta)

Tube spectrum × flat filtration × protocol filters (IPEM, absolute flux)
with the UFC MC η(E) folded in — the `spectrum_override` pair for
`create_eict_workspace`.
"""
function ufc_spectrum(protocol)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        sim_opts, protocol; scanner = scanner,
    )
    return e, Float64.(w) .* ufc_mc_efficiency.(e)
end;

# ╔═╡ 09000006-0000-4000-8000-000000000040
let
    specs = (
        ("100 kVp · tube A", protocol_low, :royalblue),
        ("Sn140 kVp · tube B", protocol_high, :crimson),
    )

    fig = CM.Figure(size = (1180, 580))
    ax = CM.Axis(
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
        e, wη = ufc_spectrum(prot)
        wn = wη ./ sum(wη)
        mean_E = sum(e .* wn)
        CM.lines!(
            ax, Float64.(e), wn ./ maximum(wn);
            color = color, linewidth = 3,
            label = "$(label)  (mean $(round(mean_E, digits = 1)) keV)",
        )
    end
    CM.axislegend(ax; position = :rt, framevisible = true, labelsize = 18)
    fig
end

# ╔═╡ 09000007-0000-4000-8000-000000000001
md"""
## 6. Forward Project (one DE acquisition = two tube scans)

Each tube builds its own η-folded workspace via `spectrum_override` and
keeps only the noisy log-line-integral sinogram + geometry.

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
        scanner, protocol_low, sim_opts, recon_opts, phantom;
        spectrum_override = ufc_spectrum(protocol_low),
    )
    BS.simulate!(ws, phantom, protocol_low, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
end;

# ╔═╡ 09000007-0000-4000-8000-000000000030
sim_high = let
    @info "Simulating: Sn140 kVp / $(round(protocol_high.mA, digits = 1)) mA (tube B, UFC η folded, z-offset −0.88 mm)…"
    ws = BS.create_eict_workspace(
        scanner, protocol_high, sim_opts, recon_opts, phantom_b;
        spectrum_override = ufc_spectrum(protocol_high),
    )
    BS.simulate!(ws, phantom_b, protocol_high, sim_opts)
    result = (sino = Array(ws.sinogram), geom = ws.geom)
    ws = nothing; GC.gc(true)
    result
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

    fig = CM.Figure(size = (1180, 580))
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
        ax = CM.Axis(fig[r, c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    CM.Colorbar(
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

So the poly validation of the UFC LUT runs the nb01 correction stack
**per tube** — η-aware sinogram BHC → FDK → image-domain BHC → HU →
residual cupping — then blends.  If the η fold is right, solid water
lands at ≈ 0 HU in *both* per-tube recons (and therefore in any blend).

!!! info "η-aware BHC"
    The BHC water polynomial must see the same detected spectrum the
    forward model used.  We resolve the bowtie-hardened per-column
    spectrum, multiply by η_UFC(E), and feed the low-level
    `calibrate_bhc_two_material(e, w_col)` — same per-column fit the
    high-level API performs, but with the UFC fold included.
"""

# ╔═╡ 09000008-0000-4000-8000-000000000005
"""
    ufc_bhc_calibration(protocol, geom)

η-aware per-tube BHC: bowtie-hardened per-column spectrum × UFC η(E) →
per-column two-material polynomial.  Returns `(model, μ_water, ref_E_keV)`.
"""
function ufc_bhc_calibration(protocol, geom)
    e, ŵ = BS.resolve_source_spectrum_with_bowtie(
        sim_opts, protocol; scanner = scanner, geom = geom,
    )
    e2, w_col = BS.bhc_spectrum_per_column(e, ŵ)          # [n_E, n_col]
    η = ufc_mc_efficiency.(e2)
    w_col_η = w_col .* η                                  # broadcast along energy dim

    # Single mono-equivalent target = mean energy of the η-folded mean spectrum
    w_mean = vec(sum(w_col_η; dims = 2)) ./ size(w_col_η, 2)
    ref_E = sum(e2 .* w_mean) / sum(w_mean)

    model = BS.calibrate_bhc_two_material(
        e2, w_col_η;
        order = 2,
        reference_energy_keV = ref_E,
        hu_low = 450.0,
        hu_high = 600.0,
    )
    return (model = model, μ_water = model.μ_water_ref, ref_E_keV = model.reference_energy_keV)
end;

# ╔═╡ 09000008-0000-4000-8000-000000000008
"""
    ufc_poly_recon(sino_cpu, geom, bhc) -> Array{Float32, 3}

nb01 correction stack for one tube: sino BHC → FDK → image BHC → HU →
residual radial cupping.
"""
function ufc_poly_recon(sino_cpu, geom, bhc)
    matrix_size = recon_opts.matrix_size

    sino_gpu = to_gpu(sino_cpu)
    sino_bhc = BS.apply_bhc_two_material(sino_gpu, bhc.model, geom, matrix_size)
    sino_gpu = to_gpu(sino_bhc)

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom)

    BS.apply_bhc_image_domain(
        recon_μ, geom, matrix_size, bhc.μ_water;
        hu_low = 50.0, hu_high = 150.0, scale_factor = 0.2,
    )

    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water))
    BS.apply_radial_cupping_correction!(hu; fov_cm = 35.0)

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
    mid = size(hu_mixed, 3) ÷ 2

    fig = CM.Figure(size = (1400, 520))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    panels = (
        (1, "100 kVp (tube A)", hu_tube.low),
        (2, "Sn140 kVp (tube B)", hu_tube.high),
        (3, "Mixed M$(MIX_W_LOW)", hu_mixed),
    )
    for (c, ttl, vol) in panels
        ax = CM.Axis(
            fig[1, c]; title = ttl,
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(ax, vol[:, :, mid]; colormap = :grays, colorrange = HU_window)
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1, 4]; colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    CM.save(
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

    fig = CM.Figure(size = (1180, 580))

    # ─── Left panel — eroded SW ROI on the mixed image ──────────────────
    HU_window = (-200, 500)
    mid = size(hu_mixed, 3) ÷ 2
    overlay = Float32[b ? 1.0f0 : NaN32 for b in poly_water_stats.mask_2d]

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Eroded Water Region",
        subtitle = "Overlaid on mixed image",
        aspect = CM.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    CM.heatmap!(ax1, hu_mixed[:, :, mid]; colormap = :grays, colorrange = HU_window)
    CM.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    CM.hidedecorations!(ax1)

    # ─── Right panel — water ⟨HU⟩ ± σ per poly readout ───────────────────
    bar_colors = [CM.cgrad(:plasma, n; categorical = true)[i] for i in 1:n]
    ax2 = CM.Axis(
        fig[1, 2];
        title = "Poly Water HU",
        subtitle = "Solid-water ROI, mean ± σ",
        xlabel = "Reconstruction", ylabel = "HU",
        xticks = (collect(1:n), [e[1] for e in entries]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.barplot!(
        ax2, 1:n, means;
        color = bar_colors, strokecolor = :black, strokewidth = 1,
    )
    CM.errorbars!(ax2, 1:n, means, stds; color = :black, whiskerwidth = 14, linewidth = 2)
    CM.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, (m, s)) in enumerate(zip(means, stds))
        CM.text!(
            ax2, k, m + s;
            text = "$(round(m, digits = 1)) ± $(round(s, digits = 1)) HU",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 6),
        )
    end

    y_max = max(25.0, 1.4 * maximum(abs.(means) .+ stds))
    CM.ylims!(ax2, -y_max, y_max)

    CM.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_poly_water_values.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000a-0000-4000-8000-000000000001
md"""
## 8. Projection-Domain Material Decomposition (Cong)

Per-ray Cong univariate solver on the polychromatic transmission integral,
iodine + water material basis, running on the **raw noisy sinograms** (no
projection-domain denoising — the anti-correlated basis noise is handled
by cov-ACNR after FBP).  The per-ray spectral weights are
**source × flat filters × bowtie × UFC η(E)** — the identical model the
forward projector applied, because we fold the same `ufc_mc_efficiency.(e)`
into the bowtie-resolved 1D spectrum before the per-ray normalization.
"""

# ╔═╡ 0900000a-0000-4000-8000-000000000010
material_basis = let
    function ŵ_ufc(protocol, geom, label)
        e, w1d = BS.resolve_source_spectrum_without_bowtie(
            sim_opts, protocol; scanner = scanner,
        )
        wη = Float64.(w1d) .* ufc_mc_efficiency.(e)
        # Per-ray bowtie fold + per-ray normalization (Σ_E ŵ = 1) → Float32 3D
        ŵ = BS.apply_bowtie_to_spectrum(wη, e, scanner, geom, protocol; label = label)
        return e, ŵ
    end

    e_L, ŵ_L = ŵ_ufc(protocol_low, sim_low.geom, "low·UFC")
    e_H, ŵ_H = ŵ_ufc(protocol_high, sim_high.geom, "high·UFC")

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

# ╔═╡ 0900000a-0000-4000-8000-000000000020
sino_basis = let
    sino_low_gpu = to_gpu(Float32.(sim_low.sino))
    sino_high_gpu = to_gpu(Float32.(sim_high.sino))

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

# ╔═╡ 0900000a-0000-4000-8000-000000000040
let
    n_row = size(sino_basis.sino_iodine, 2)
    mid_r = n_row ÷ 2 + 1

    fig = CM.Figure(size = (1400, 580))
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
        ax = CM.Axis(fig[r, panel_c]; title = ttl, axis_kwargs...)
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 0900000b-0000-4000-8000-000000000001
md"""
## 9. FBP × 2, cov-ACNR, Z-Median

Identical post-decomposition chain to nb03: two FDK passes
(`BS.SoftFilter()`) → image-domain data-adaptive cov-ACNR
(`BS.apply_image_acnr!`, structure axis pixel-perfect) → z-median
(5-slice window, exploits the Gammex z-invariance).
"""

# ╔═╡ 0900000b-0000-4000-8000-000000000010
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

# ╔═╡ 0900000b-0000-4000-8000-000000000020
basis_acnr = let
    GAMMA = 1.0
    BILAT_RADIUS = 3
    BILAT_SIGMA_S = 1.0
    BILAT_RANGE_K = 2.5

    W = copy(basis_volumes.vol_water_raw)
    I = copy(basis_volumes.vol_iodine_raw)

    info = BS.apply_image_acnr!(
        W, I;
        gamma = GAMMA, bilat_radius = BILAT_RADIUS,
        bilat_sigma_s = BILAT_SIGMA_S, bilat_range_k = BILAT_RANGE_K
    )
    @info "[ACNR · cov] θ=$(round(info.θ_deg, digits = 1))° · ρ(W,I)=$(round(info.ρ_struct, digits = 3)) · γ=$(GAMMA) · σ_noise(W)=$(round(info.σ_W, sigdigits = 3)), σ_noise(I)=$(round(info.σ_I, sigdigits = 3))"

    (vol_iodine_raw = I, vol_water_raw = W, geom = basis_volumes.geom)
end;

# ╔═╡ 0900000b-0000-4000-8000-000000000025
Z_MEDIAN_ADJACENT = 2;

# ╔═╡ 0900000b-0000-4000-8000-000000000030
basis_z = let
    (
        vol_iodine = BS.apply_median_z(
            basis_acnr.vol_iodine_raw;
            adjacent_slices = Z_MEDIAN_ADJACENT,
        ),
        vol_water = BS.apply_median_z(
            basis_acnr.vol_water_raw;
            adjacent_slices = Z_MEDIAN_ADJACENT,
        ),
        geom = basis_acnr.geom,
    )
end;

# ╔═╡ 0900000b-0000-4000-8000-000000000040
let
    fig = CM.Figure(size = (1180, 580))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    mid = size(basis_z.vol_iodine, 3) ÷ 2

    _qrange(arr) = (
        Float64(quantile(vec(arr), 0.01)),
        Float64(quantile(vec(arr), 0.99)),
    )

    slice_iod = basis_z.vol_iodine[:, :, mid]
    slice_wat = basis_z.vol_water[:, :, mid]

    panels = (
        (1, 1, 2, "Iodine Basis", "g/cm³", slice_iod, _qrange(slice_iod)),
        (1, 3, 4, "Water Basis", "g/cm³", slice_wat, _qrange(slice_wat)),
    )

    for (r, panel_c, cbar_c, ttl, cbar_label, slice, range) in panels
        ax = CM.Axis(
            fig[r, panel_c]; title = ttl,
            aspect = CM.DataAspect(), axis_kwargs...
        )
        CM.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        CM.hidedecorations!(ax)
        CM.Colorbar(
            fig[r, cbar_c]; colormap = :viridis, colorrange = range,
            label = cbar_label, width = 16, labelsize = 22, ticklabelsize = 18
        )
    end
    fig
end

# ╔═╡ 0900000c-0000-4000-8000-000000000001
md"""
## 10. VMI Synthesis

Textbook 2-basis mix (McCollough 2015) at 50 / 70 / 100 / 140 keV:

```
μ(E)  = c_water(r) · (μ/ρ)_water(E) + c_iodine(r) · (μ/ρ)_iodine(E)
HU(E) = 1000 · (μ(E) − (μ/ρ)_water(E)) / (μ/ρ)_water(E)
```

The `solid_water_basis` diagnostic logs the basis-decomp residual bias as
a Δ% between the SW-ROI synth μ_water and the textbook mono divisor.
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
    @info "solid_water_basis: SW mid-slice voxel count $(n_raw) → $(n_eroded) " *
        "after $(ERODE_PX)-px erosion"

    sw_idx = findall(sw_bool)
    n_z = size(basis_z.vol_water, 3)
    function _mean(vol)
        s = 0.0; n = 0
        for z in 1:n_z, ci in sw_idx
            s += vol[ci, z]; n += 1
        end
        return s / n
    end

    c_w = Float64(_mean(basis_z.vol_water))
    c_i = Float64(_mean(basis_z.vol_iodine))
    @info "solid_water_basis: ⟨c_water⟩_SW = $(round(c_w, digits = 4)) g/cm³, " *
        "⟨c_iodine⟩_SW = $(round(c_i, digits = 6)) g/cm³"

    (
        c_water = c_w, c_iodine = c_i, n_voxels = length(sw_idx) * n_z,
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 0900000c-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 0900000c-0000-4000-8000-000000000020
vmi_HU_by_keV = let
    # synth_vmi_2basis expects c_iodine in mg/mL; basis maps are g/cm³ (= g/mL)
    c_iodine_mg_per_mL = basis_z.vol_iodine .* 1000.0f0

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
            basis_z.vol_water, c_iodine_mg_per_mL;
            energy_keV = E,
        )
    end
    out
end;

# ╔═╡ 0900000c-0000-4000-8000-000000000040
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_by_keV[50.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_by_keV[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_grid.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000d-0000-4000-8000-000000000001
md"""
## 11. VMI Post-Processing (Mono+)

Grant 2014 frequency split toward the 70 keV noise-optimal anchor.
σ pairs element-wise with `de_vmi_energies`; 0 ⇒ identity.
"""

# ╔═╡ 0900000d-0000-4000-8000-000000000005
# (50, 70, 100, 140) keV
σ_vmi_lp_px = Float64[1.0, 0.0, 1.0, 1.0];

# ╔═╡ 0900000d-0000-4000-8000-000000000010
vmi_HU_final = let
    volumes = [vmi_HU_by_keV[E] for E in de_vmi_energies]

    ws = BS.create_mono_plus_workspace(
        volumes[1];
        n_energies = length(de_vmi_energies)
    )
    BS.apply_mono_plus!(
        ws, volumes, de_vmi_energies;
        E_noise_opt = 70.0,
        σ_lp_px = σ_vmi_lp_px,
        verbose = true,
    )

    out = Dict{Float64, Array{Float32, 3}}()
    for (i, E) in enumerate(de_vmi_energies)
        out[E] = copy(ws.out_vols[i])
    end
    ws = nothing; GC.gc(true)
    out
end;

# ╔═╡ 0900000d-0000-4000-8000-000000000030
let
    HU_window = (-200, 500)

    fig = CM.Figure(size = (1180, 1180))
    axis_kwargs = (titlesize = 32, subtitlesize = 24)

    sample = vmi_HU_final[50.0]
    mid = size(sample, 3) ÷ 2

    for (k, E) in enumerate(de_vmi_energies)
        r = ((k - 1) ÷ 2) + 1
        c = ((k - 1) % 2) + 1
        ax = CM.Axis(
            fig[r, c]; title = "$(Int(E)) keV VMI",
            subtitle = "Mono+",
            aspect = CM.DataAspect(), axis_kwargs...,
        )
        CM.heatmap!(
            ax, vmi_HU_final[E][:, :, mid];
            colormap = :grays, colorrange = HU_window,
        )
        CM.hidedecorations!(ax)
    end
    CM.Colorbar(
        fig[1:2, 3];
        colormap = :grays, colorrange = HU_window,
        label = "HU", width = 16, labelsize = 22, ticklabelsize = 18,
    )

    CM.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_monoplus.png"),
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
    fig = CM.Figure(size = (1180, 580))

    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in solid_water_basis.mask_2d]

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Eroded Water Region",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = CM.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    CM.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    CM.heatmap!(
        ax1, overlay;
        colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    CM.hidedecorations!(ax1)

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
    bar_colors = [CM.cgrad(:plasma, n_E; categorical = true)[i] for i in 1:n_E]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water Region Mean HU",
        subtitle = "Per VMI Energy",
        xlabel = "VMI Energy (keV)", ylabel = "HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.barplot!(
        ax2, 1:n_E, sw_hu_per_keV;
        color = bar_colors,
        strokecolor = :black, strokewidth = 1,
    )
    CM.hlines!(ax2, [0.0]; color = :black, linewidth = 1, linestyle = :dash)

    for (k, h) in pairs(sw_hu_per_keV)
        CM.text!(
            ax2, k, h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 16, offset = (0, h ≥ 0 ? 4 : -4),
        )
    end

    y_max = max(15.0, 1.2 * maximum(abs, sw_hu_per_keV))
    CM.ylims!(ax2, -y_max, y_max)

    CM.save(
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
    nx_r, ny_r, nz_r = size(basis_z.vol_water)
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
    mid = size(vmi_HU_final[70.0], 3) ÷ 2
    bg = vmi_HU_final[70.0][:, :, mid]

    overlay = Float32[b ? 1.0f0 : NaN32 for b in water_noise_roi.mask_2d]

    fig = CM.Figure(size = (1180, 580))

    ax1 = CM.Axis(
        fig[1, 1];
        title = "Water-Region Noise ROI",
        subtitle = "Overlaid on 70 keV VMI",
        aspect = CM.DataAspect(),
        titlesize = 32, subtitlesize = 24,
    )
    CM.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    CM.heatmap!(
        ax1, overlay; colormap = :reds, alpha = 0.5,
        nan_color = (:white, 0.0),
    )
    CM.hidedecorations!(ax1)

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σs = [vmi_noise_by_keV[E].std  for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]

    ax2 = CM.Axis(
        fig[1, 2];
        title = "Water-Region Noise vs Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        titlesize = 32, subtitlesize = 24,
        xlabelsize = 22, ylabelsize = 22,
        xticklabelsize = 18, yticklabelsize = 16,
    )
    CM.scatterlines!(
        ax2, Es, σs;
        color = :tomato, markersize = 18, linewidth = 3,
    )
    for (E, σ, μ) in zip(Es, σs, μs)
        CM.text!(
            ax2, E, σ;
            text = "σ=$(round(σ; digits = 1))\n⟨HU⟩=$(round(μ; digits = 1))",
            align = (:center, :bottom),
            fontsize = 16, offset = (0, 8),
        )
    end

    CM.save(
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
    fig = CM.Figure(size = (1180, 580))

    cmap_ca = CM.cgrad(:Oranges, 7; categorical = true)
    cmap_i = CM.cgrad(:GnBu, 7; categorical = true)

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
        ax = CM.Axis(
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
        CM.ylims!(ax, p.ylim...)

        d = rod_data[p.group]
        rod_lines = Vector{Any}(undef, length(d.names))
        for i in eachindex(d.names)
            color = p.cmap[i]
            CM.scatterlines!(
                ax, de_vmi_energies, vec(d.measured[i, :]);
                color = color, linewidth = 2.5, markersize = 9,
            )
            CM.lines!(
                ax, de_vmi_energies, vec(d.theoretical[i, :]);
                color = color, linewidth = 1.6, linestyle = :dash,
            )
            rod_lines[i] = CM.LineElement(color = color, linewidth = 2.5)
        end

        style_meas = CM.MarkerElement(
            color = :black, marker = :circle, markersize = 9,
            strokecolor = :black, strokewidth = 1,
        )
        style_theo = CM.LineElement(
            color = :black, linewidth = 1.6, linestyle = :dash,
        )
        CM.axislegend(
            ax,
            vcat([style_meas, style_theo], rod_lines),
            vcat(["Measured", "Theoretical"], collect(d.names));
            position = :rt, framevisible = true, labelsize = 18,
            rowgap = 1, padding = (6, 6, 6, 6),
        )
    end

    CM.save(
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
    fig = CM.Figure(size = (1000, 1200))

    energy_colors = Dict(
        50.0 => CM.RGBf(0.85, 0.27, 0.1),
        70.0 => CM.RGBf(0.95, 0.65, 0.13),
        100.0 => CM.RGBf(0.13, 0.59, 0.85),
        140.0 => CM.RGBf(0.1, 0.27, 0.65),
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
        ax = CM.Axis(
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
        CM.lines!(
            ax, [lim_lo, lim_hi], [lim_lo, lim_hi];
            color = :black, linestyle = :dash, linewidth = 2,
            label = "Unity (y = x)",
        )

        for (j, E) in pairs(de_vmi_energies)
            x = Vector{Float64}(vec(d.theoretical[:, j]))
            y = Vector{Float64}(vec(d.measured[:, j]))
            color = energy_colors[E]
            CM.scatter!(ax, x, y; color = color, markersize = 11)

            f = fit_lr(x, y)
            xrange = [minimum(x), maximum(x)]
            yrange = f.intercept .+ f.slope .* xrange
            sign_str = f.intercept ≥ 0 ? "+" : "−"
            label = "$(Int(E)) keV: y = $(round(f.slope, digits = 2))·x " *
                "$(sign_str) $(round(abs(f.intercept), digits = 0)) HU   " *
                "R² = $(round(f.r², digits = 3))   " *
                "RMSE = $(round(f.rmse, digits = 1)) HU"
            CM.lines!(
                ax, xrange, yrange;
                color = color, linewidth = 2, label = label,
            )
        end

        CM.axislegend(
            ax; position = :rb, framevisible = true,
            labelsize = 16, padding = (6, 6, 6, 6), rowgap = 1,
        )
    end

    CM.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_regression.png"),
        fig; px_per_unit = 2,
    )
    fig
end

# ╔═╡ 09000010-0000-4000-8000-000000000001
md"""
## Summary

```
UFC MC η(E) LUT (Khodajou-Chokami, Gd₂O₂S, 1–140 keV)
   → folded into the source spectrum  (spectrum_override; η-exact forward + inverse)
Simulate 100 kVp + Sn140 kVp   (one SOMATOM Force dual-source DE acquisition)
   ├─→ POLY: per-tube η-aware BHC → FDK → HU → mixed image M_w  (SW ≈ 0 HU ×3)
   └─→ VMI:  Cong Decomposition on raw sinograms  (iodine + water,
             bowtie + η_UFC per ray)
             → FBP × 2 → cov-ACNR → z-median
             → VMI 50/70/100/140 keV → Mono+
             → Per-Rod Measured vs Theoretical Regression
```

**What validates the UFC LUT here:**

1. **§7 poly**: solid water ≈ 0 HU in the 100 kVp recon, the Sn140 recon,
   and the Siemens-style mixed image, each under its own η-aware BHC —
   the LUT's spectral shape is consistent with the detected sinogram on
   both very different tube spectra independently.
2. **§10–11 VMI**: water-ROI bars ≈ 0 HU at every keV and tight per-rod
   measured-vs-theoretical overlays — the LUT survives the much harsher
   test of per-ray spectral inversion across the two detected spectra
   (100 kVp vs Sn140 kVp, separated by the 0.6 mm tin filter and sitting
   on opposite sides of the Gd K-edge fluorescence-escape cliff).

**Next step (after sign-off):** promote the LUT into
`src/detector/detector_efficiency.jl` next to `GEMSTONE_MC_EFFICIENCY_LUT`
(`UFC_MC_EFFICIENCY_LUT` + `get_ufc_mc_efficiency` +
`detector_efficiency_ufc()` + a `:ufc` branch in
`compute_eid_efficiency_vector` and `build_physics_config`), then this
notebook collapses to `use_detector_efficiency = true` with
`detector_material = :ufc` and no `spectrum_override`.
"""

# ╔═╡ Cell order:
# ╠═09000001-0000-4000-8000-000000000001
# ╠═09000001-0000-4000-8000-000000000002
# ╠═09000001-0000-4000-8000-000000000003
# ╟─09000001-0000-4000-8000-000000000010
# ╟─09000001-0000-4000-8000-000000000020
# ╠═09000001-0000-4000-8000-000000000030
# ╠═09000001-0000-4000-8000-000000000031
# ╠═09000001-0000-4000-8000-000000000040
# ╟─09000001-0000-4000-8000-000000000050
# ╟─09000002-0000-4000-8000-000000000001
# ╠═09000002-0000-4000-8000-000000000010
# ╠═09000002-0000-4000-8000-000000000020
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
# ╠═0900000a-0000-4000-8000-000000000020
# ╟─0900000a-0000-4000-8000-000000000040
# ╟─0900000b-0000-4000-8000-000000000001
# ╠═0900000b-0000-4000-8000-000000000010
# ╠═0900000b-0000-4000-8000-000000000020
# ╠═0900000b-0000-4000-8000-000000000025
# ╠═0900000b-0000-4000-8000-000000000030
# ╟─0900000b-0000-4000-8000-000000000040
# ╟─0900000c-0000-4000-8000-000000000001
# ╠═0900000c-0000-4000-8000-000000000010
# ╠═0900000c-0000-4000-8000-000000000015
# ╠═0900000c-0000-4000-8000-000000000020
# ╟─0900000c-0000-4000-8000-000000000040
# ╟─0900000d-0000-4000-8000-000000000001
# ╠═0900000d-0000-4000-8000-000000000005
# ╠═0900000d-0000-4000-8000-000000000010
# ╟─0900000d-0000-4000-8000-000000000030
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
