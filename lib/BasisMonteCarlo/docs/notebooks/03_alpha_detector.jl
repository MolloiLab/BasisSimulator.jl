### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 6e30fdb2-879c-52f4-99ab-666bbb68cd4f
md"""
# The NAEOTOM Alpha Detector in Geant4

The Siemens NAEOTOM Alpha is a photon-counting CT scanner: its detector counts x rays one by one
and sorts each into an energy bin. This notebook builds that detector in Geant4, from the
smallest piece up, starting with **one dexel** (detector element) and its eight neighbours.

Two things decide what a dexel reports, and both are simulated here:

- **What the photon does in the crystal** (Geant4): where it is absorbed, whether its
  characteristic x ray escapes or is absorbed under a neighbouring dexel, whether it scatters.
- **What the electronics make of it** (Julia, after Geant4): the cloud of charge each deposit
  becomes, how that cloud is shared between dexels, noise, and the energy thresholds.

The result is the dexel's **response**: for a photon of each energy, how many counts it makes
above each threshold, in the dexel it hit and in each neighbour. BasisSimulator already uses
such a table, computed by Hamidreza Khodajou-Chokami with MCNP (`cdte_response_v4`, PR #10),
whose code was never committed. The last section compares the two point by point.

**The parts.** (1) One dexel. (2) The anti-scatter grid over it, as the Alpha has one.
(3) Photons arriving at an angle. (4) The 2 × 2 binned dexels the scanner reads out. (5) The
whole detector, one to one, and the symmetry that reduces it to six dexels. (6) The Alpha's
x-ray beam. (7) Pile-up at clinical count rates.
"""

# ╔═╡ df200390-005d-5328-870a-9a821744961d
md"""
## How a Dexel Counts
"""

# ╔═╡ e71ceeb9-0d88-5608-8588-ee48a312de90
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 330))
    ax = Mke.Axis(fig[1, 1]; limits = ((0, 130), (0, 30)))
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)

    function box!(x, y, w, h, color)
        r = 1.6
        θ = range(0, π / 2; length = 10)
        corners = vcat(
            [Mke.Point2f(x + w - r + r * cos(t), y + h - r + r * sin(t)) for t in θ],
            [Mke.Point2f(x + r - r * sin(t), y + h - r + r * cos(t)) for t in θ],
            [Mke.Point2f(x + r - r * cos(t), y + r - r * sin(t)) for t in θ],
            [Mke.Point2f(x + w - r + r * sin(t), y + r - r * cos(t)) for t in θ],
        )
        return Mke.poly!(
            ax,
            corners;
            color = (color, 0.1),
            strokecolor = color,
            strokewidth = 2,
        )
    end

    steps = (
        (
            title = "1  Photon",
            body = "one x ray of\nenergy E lands\non the dexel",
            where = "beam",
            color = PALETTE.photon,
        ),
        (
            title = "2  Crystal",
            body = "absorbed, scattered;\nits x rays may land\nunder a neighbour",
            where = "Geant4",
            color = PALETTE.xray,
        ),
        (
            title = "3  Charge",
            body = "each deposit becomes\na cloud of charge,\nσ = 30 µm",
            where = "Julia",
            color = PALETTE.electron,
        ),
        (
            title = "4  Dexels",
            body = "each dexel collects\nthe part of every\ncloud above it",
            where = "Julia",
            color = PALETTE.slab_edge,
        ),
        (
            title = "5  Counts",
            body = "noise, then one count\nfor every threshold\nits energy exceeds",
            where = "Julia",
            color = PALETTE.kept,
        ),
    )
    w, gap = 23.0, 3.0
    for (i, s) in enumerate(steps)
        x = 1 + (i - 1) * (w + gap)
        box!(x, 3, w, 25, s.color)
        Mke.text!(
            ax,
            x + 1.5,
            26.0;
            text = s.title,
            font = :bold,
            fontsize = 20,
            color = s.color,
            align = (:left, :top),
        )
        Mke.text!(
            ax,
            x + 1.5,
            20.5;
            text = s.body,
            fontsize = 15,
            color = PALETTE.ink,
            align = (:left, :top),
        )
        Mke.text!(
            ax,
            x + 1.5,
            4.5;
            text = s.where,
            fontsize = 14,
            font = :italic,
            color = s.color,
            align = (:left, :bottom),
        )
        if i < length(steps)
            Mke.arrows2d!(
                ax,
                [Mke.Point2f(x + w + 0.3, 15.5)],
                [Mke.Vec2f(gap - 0.6, 0)];
                color = PALETTE.muted,
                shaftwidth = 3,
                tipwidth = 12,
                tiplength = 10,
            )
        end
    end
    fig
end

# ╔═╡ 8326156c-5e6a-5f16-89ab-5e8b268b2e0a
md"""
Steps 1–2 are Geant4: it records where every bit of energy is left in the crystal. Steps 3–5 are
the detector's electronics, modelled in Julia on those positions, so the dexel size, the charge
cloud, the noise and the thresholds can all be changed without simulating the photons again.
"""

# ╔═╡ f7ad4569-b274-5714-83ab-bc7ad0497b53
md"""
## The NAEOTOM Alpha, as Documented

Every hardware number this notebook uses, checked against the FDA 510(k) summaries, Siemens'
own technical documents and peer-reviewed measurements of clinical systems. Where a source and
BasisSimulator disagree the row is marked **⚠ discrepancy**; the notebook then keeps
BasisSimulator's value, so the two stay comparable, unless the physics being built needs the
documented one (the anti-scatter grid, which BasisSimulator does not model, is built as
documented).

**The family.** Three cleared products share one detector (QuantaMax, CdTe): the original
**NAEOTOM Alpha**, called **Alpha.Peak** on new systems (dual source, 2 × 288 rows, 2 × 57.6 mm),
**Alpha.Pro** (dual source, 2 × 192 rows, 2 × 38.4 mm) and **Alpha.Prime** (single source,
288 rows). Detector A has 2752 channels and a 50 cm field of view, detector B 1984 channels and
36 cm (K211591, K243523).

| quantity | BasisSimulator | documented | status | source |
|---|---|---|---|---|
| sensor | CdTe, 1.6 mm | CdTe, 1.6 mm | ✓ | K211591 §VI; Sartoretti 2023 |
| dexel (sub-pixel) at the detector face | 0.275 × 0.322 mm | 0.275 × 0.322 mm (0.151 × 0.176 mm at isocentre) | ✓ | Flohr 2023; Li 2025 |
| standard-mode binning | 2 × 2 | 2 × 2 | ✓ | Flohr 2023 |
| sampling pitch (dexel + its share of the grid gap) | 0.301 × 0.353 mm at isocentre | 0.37 × 0.40 mm at isocentre | **⚠ discrepancy** | Li 2025, Table 1 |
| rows × coverage | 144 × 0.353 = 50.8 mm | 144 × 0.4 = 57.6 mm | **⚠ discrepancy** | K243523 Tables 3–4 |
| columns (binned) | 1659 | 1376 (2752 native) | **⚠ discrepancy** | K243523 Tables 3–4 |
| fill factor (x, z) | 0.95, 0.95 | ≈ 0.81, ≈ 0.885 | **⚠ discrepancy** | Rajendran 2021 §2.1; derived from Li 2025 |
| anti-scatter grid | not modelled | 2-D, cells of 4 × 6 dexels; dimensions and material proprietary | built here (see section 5) | Flohr 2023; Hennemann 2026 §2 |
| source–isocentre, source–detector | 610, 1113 mm | 610, 1113 mm | ✓ | Li 2025; Hennemann 2026 |
| anode angle | 7° (geometry), 10° (spectrum default) | 8° | **⚠ discrepancy** | K233657 Table 4; K243523 Table 3; Siemens white paper p. 6 |
| thresholds at 140 kV | 20 / 35 / 55 / 70 keV | 20/35/55/70 (kV not stated); clinical dual-energy split 20/70 keV at 140 kV | ✓ (consistent) | Sartoretti 2023; Jost 2023 |
| dead time | 5.0 ns | 5.3–7.0 ns (measured) | **⚠ discrepancy** | Li 2025 |
| electronic noise, charge cloud | σ_cloud 30 µm, 1.5 keV (`cdte_response_v4`) | model fit to clinical data: 16.4 µm charge-spreading radius, 1.9 keV noise | different model; both shown | Zhan 2026 |
| tin filters | none in the Alpha notebooks | Sn 0.4 mm on both tubes; Sn 0.7 mm extra on B | n/a here | K233657; K243523 |
| measured HVL at 140 kV, isocentre | – | 7.7 mm Al (with table), 7.9 mm Al (without) | **⚠ discrepancy**: BasisSimulator's beam gives 9.6 mm (section 9) | Zhan 2026 Table 3; Li 2025 Table 2 |

**The grid, derived.** The walls enclose 4 × 6 dexels. Per dexel the published pitch is
0.68/2 = 0.34 mm across the fan and 1.13 × 0.322 = 0.364 mm along z, against 0.275 × 0.322 mm of
dexel: so each 4 × 6 group is followed by a gap of 4 × (0.34 − 0.275) ≈ **0.26 mm** across the
fan and 6 × (0.364 − 0.322) ≈ **0.25 mm** along z, under a wall of that thickness. This reproduces
both published fill factors (1.1 / 1.36 = 0.81 and 1.932 / 2.182 = 0.885). The wall height is not
published; **10 mm of tungsten** is assumed, and section 6 shows where it matters.

Sources: FDA 510(k) summaries
[K211591](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K211591.pdf),
[K233657](https://www.accessdata.fda.gov/cdrh_docs/pdf23/K233657.pdf),
[K243523](https://www.accessdata.fda.gov/cdrh_docs/pdf24/K243523.pdf);
Siemens [white paper](https://marketing.webassets.siemens-healthineers.com/358113cc7cf4f651/8f3e60d17f5e/ct_naeotom_alpha_whitepaper.pdf);
[Flohr & Schmidt 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10259209/);
[Rajendran et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8551012/);
[Sartoretti et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10321251/);
[Li et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12497395/);
[Zhan et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13691410/);
[Hennemann et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13125421/);
[Jost et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10259215/).
"""

# ╔═╡ 943d5189-ca8c-5e1b-860d-dad9ed192843
md"""
## Notebook Setup
"""

# ╔═╡ ed5d0968-55b8-54bb-be08-b8b3e5231445
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 73911ef9-afec-5057-9406-29f4217f5d15
using Markdown: @md_str, Markdown

# ╔═╡ aa57fb94-4f51-5853-8cf5-c0598ee6b717
import PlutoUI

# ╔═╡ 9215c3a9-8e71-5f66-b12a-2779e5350eda
import BasisMonteCarlo as MC

# ╔═╡ 1ff7c097-d985-5412-9686-8e062018d0b2
import CairoMakie as Mke

# ╔═╡ bc2a467c-d990-556b-b955-dd91539122ab
import BasisSimulator as BS

# ╔═╡ fdaa0020-4804-5c58-8453-668f6c70ebf3
import Serialization

# ╔═╡ 0df22b09-66d5-59e9-9f68-79a2b1886e38
PlutoUI.TableOfContents()

# ╔═╡ 40986dbf-a0d8-51ea-aa7b-75b21dcc7eaa
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ a944bba9-6761-5648-b028-0d90a7b56e29
# One colour per kind of particle, used by every figure
PALETTE = (
    photon = Mke.RGBf(0.13, 0.36, 0.66),  # the incident photon
    xray = Mke.RGBf(0.9, 0.45, 0.05),  # characteristic x rays
    electron = Mke.RGBf(0.7, 0.12, 0.17),  # electrons and their charge
    kept = Mke.RGBf(0.27, 0.55, 0.34),  # counts
    slab = Mke.RGBAf(0.45, 0.52, 0.62, 0.3),
    slab_edge = Mke.RGBf(0.3, 0.36, 0.44),
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator's current table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ c5fa8b52-766b-5c4a-9794-a056c8e00dc2
# Large, uncluttered type for every figure
THEME = Mke.Theme(
    fontsize = 18,
    figure_padding = (12, 24, 12, 12),
    Axis = (
        titlesize = 21,
        titlealign = :left,
        titlegap = 14,
        xlabelsize = 19,
        ylabelsize = 19,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xgridcolor = (:black, 0.07),
        ygridcolor = (:black, 0.07),
        topspinevisible = false,
        rightspinevisible = false,
        xlabelpadding = 8,
        ylabelpadding = 12,
    ),
    Axis3 = (
        titlesize = 21,
        titlealign = :left,
        xlabelsize = 18,
        ylabelsize = 18,
        zlabelsize = 18,
        xticklabelsize = 15,
        yticklabelsize = 15,
        zticklabelsize = 15,
    ),
    Legend = (
        labelsize = 17,
        framevisible = false,
        patchsize = (30, 14),
        rowgap = 8,
    ),
    Lines = (
        linewidth = 2.5,
    ),
)

# ╔═╡ eb7fb66a-4e78-5a01-ada0-e323524fde34
"""
An in-figure annotation on a translucent rounded background, so it never fights the data.
"""
function note!(
        ax, x, y, text;
        align = (:left, :center),
        color = PALETTE.ink,
        fontsize = 16,
        offset = (0, 0),
        alpha = 0.85,
    )
    return Mke.textlabel!(
        ax, Mke.Point2f(x, y);
        text,
        text_align = align,
        text_color = color,
        fontsize,
        offset,
        background_color = (:white, alpha),
        strokewidth = 0,
        padding = (7, 7, 4, 4),
        cornerradius = 5,
    )
end

# ╔═╡ a75d1e50-0d30-51f0-9004-d0a001673992
md"""
## 1. One Dexel and Its Neighbours

The Alpha's sensor is a continuous slab of cadmium telluride, 1.6 mm deep. Photons enter through
the front face, which is one continuous electrode (the cathode). The back face carries a grid of
small electrodes (the anodes); each one, with the crystal above it, is a **dexel**. The bias
voltage between the faces drives the electrons each absorbed photon frees toward the back, and
the dexel they arrive at records the charge.

BasisSimulator models the Alpha with native dexels of **0.275 × 0.322 mm** (x across the fan,
y along the patient), read out in 2 × 2 groups. The response of one native dexel is simulated
here: photons land uniformly over the central dexel, and the charge is recorded in it and its
eight neighbours.
"""

# ╔═╡ d1967848-6509-5e0e-8f13-db8ea4993799
crystal = MC.SlabDetector(:CdTe, 1.6)

# ╔═╡ c03a39db-4010-59b6-8c41-be48372ae822
readout = MC.DexelReadout(;
    pitch_x_mm = 0.275,
    pitch_y_mm = 0.322,
    neighborhood = 3,
    cloud_sigma_um = 30.0,
)

# ╔═╡ c9a17736-48ea-5cf4-acdd-39000bd8e7ae
# Photons land uniformly over the central dexel
beam = MC.AreaBeam(;
    width_mm = readout.pitch_x_mm,
    height_mm = readout.pitch_y_mm,
)

# ╔═╡ 497d96da-68e6-5194-ac65-4f174288ace0
let
    Mke.set_theme!(THEME)
    px, py = readout.pitch_x_mm, readout.pitch_y_mm
    d = crystal.thickness_mm
    fig = Mke.Figure(size = (1400, 640))

    # (a) The crystal around the central dexel, in 3-D, with the dexel grid on its back face
    ax3 = Mke.Axis3(
        fig[1, 1];
        title = "(a) 3 × 3 dexels, to scale",
        aspect = :data,
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        zlabel = "depth z (mm)",
        azimuth = 1.3π,
        elevation = 0.16π,
        zreversed = true,
        protrusions = (30, 10, 30, 30),
    )
    for i in -1:1, j in -1:1
        block = Mke.Rect3f(
            Mke.Vec3f((i - 0.5) * px, (j - 0.5) * py, 0),
            Mke.Vec3f(px, py, d),
        )
        central = i == 0 && j == 0
        Mke.mesh!(
            ax3,
            block;
            color = central ? (PALETTE.photon, 0.25) : (PALETTE.slab_edge, 0.1),
        )
        Mke.wireframe!(
            ax3,
            block;
            color = central ? PALETTE.photon : PALETTE.slab_edge,
            linewidth = central ? 2 : 0.8,
        )
    end
    # the photons arrive from above (z < 0) onto the central dexel
    for (x, y) in ((0.0, 0.0), (0.08, -0.1), (-0.09, 0.11), (0.1, 0.12), (-0.11, -0.12))
        Mke.lines!(ax3, [x, x], [y, y], [-0.8, 0.0]; color = PALETTE.photon, linewidth = 2)
    end
    Mke.text!(
        ax3,
        Mke.Point3f(0, 0, -0.8);
        text = "photons",
        align = (:center, :bottom),
        color = PALETTE.photon,
        fontsize = 17,
        font = :bold,
        offset = (0, 6),
    )
    Mke.text!(
        ax3,
        Mke.Point3f(1.5px, -1.5py, 0);
        text = "front face:\ncathode",
        align = (:left, :center),
        fontsize = 15,
        color = PALETTE.slab_edge,
        offset = (8, 0),
    )
    Mke.text!(
        ax3,
        Mke.Point3f(1.5px, -1.5py, d);
        text = "back face:\none anode\nper dexel",
        align = (:left, :center),
        fontsize = 15,
        color = PALETTE.slab_edge,
        offset = (8, 0),
    )

    # (b) The dexel grid seen from the beam
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the beam",
        xlabel = "x, across the fan (mm)",
        ylabel = "y, along the patient (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6px, 2.3px), (-2.0py, 1.6py)),
    )
    for i in -1:1, j in -1:1
        central = i == 0 && j == 0
        Mke.poly!(
            axb, Mke.Rect2f((i - 0.5) * px, (j - 0.5) * py, px, py);
            color = central ? (PALETTE.photon, 0.18) : (PALETTE.slab_edge, 0.06),
            strokecolor = PALETTE.slab_edge,
            strokewidth = 1.5,
        )
    end
    note!(
        axb,
        0,
        0.12,
        "central dexel:\nphotons land\nanywhere here";
        align = (:center, :center),
        color = PALETTE.photon,
        fontsize = 14,
    )
    note!(
        axb,
        px,
        py,
        "neighbour";
        align = (:center, :center),
        color = PALETTE.muted,
        fontsize = 13,
        alpha = 0.0,
    )
    Mke.bracket!(
        axb,
        -0.5px,
        -1.5py,
        0.5px,
        -1.5py;
        text = "$(px) mm",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )
    Mke.bracket!(
        axb,
        1.5px,
        -0.5py,
        1.5px,
        0.5py;
        text = "$(py) mm",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )

    # (c) A cut through the central row: depth and the drift of the charge
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side view",
        xlabel = "x, across the fan (mm)",
        ylabel = "depth z (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.6px, 2.6px), (-0.35, d + 0.15)),
    )
    Mke.poly!(
        axc,
        Mke.Rect2f(-1.5px, 0, 3px, d);
        color = PALETTE.slab,
        strokecolor = PALETTE.slab_edge,
        strokewidth = 1.5,
    )
    for i in -1:1
        Mke.lines!(
            axc,
            [(i - 0.5) * px + 0.01, (i + 0.5) * px - 0.01],
            [d, d];
            color = PALETTE.slab_edge,
            linewidth = 6,
        )
    end
    Mke.lines!(axc, [-1.5px, 1.5px], [0, 0]; color = PALETTE.slab_edge, linewidth = 3)
    Mke.lines!(axc, [0.03, 0.03], [-0.3, 0.45]; color = PALETTE.photon, linewidth = 3)
    Mke.scatter!(axc, [Mke.Point2f(0.03, 0.45)]; color = PALETTE.electron, markersize = 12)
    Mke.arrows2d!(
        axc,
        [Mke.Point2f(0.03, 0.5)],
        [Mke.Vec2f(0, d - 0.6)];
        color = PALETTE.electron,
        shaftwidth = 3,
        tipwidth = 12,
        tiplength = 12,
    )
    note!(
        axc,
        0.06,
        0.95,
        "the freed\nelectrons drift\nto the anode,\nspreading";
        align = (:left, :center),
        color = PALETTE.electron,
        fontsize = 13,
    )
    note!(
        axc,
        -1.45px,
        d - 0.08,
        "anodes (one per dexel)";
        align = (:left, :bottom),
        color = PALETTE.slab_edge,
        fontsize = 13,
    )
    note!(
        axc,
        -1.45px,
        -0.05,
        "cathode";
        align = (:left, :bottom),
        color = PALETTE.slab_edge,
        fontsize = 13,
    )
    Mke.bracket!(
        axc,
        1.5px,
        0,
        1.5px,
        d;
        text = "$(d) mm",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    Mke.colgap!(fig.layout, 30)
    fig
end

# ╔═╡ 05b29a50-b080-5d3b-b8f3-a51d0c6ee344
md"""
## 2. From a Photon to Counts

One photon's journey, in pictures. `record_tracks` follows a few hundred photons landing on the
central dexel; the figure picks one whose characteristic x ray was absorbed **under a
neighbouring dexel**, so its energy is split between dexels. (a) The photon (blue) is absorbed
near the front face; the Cd or Te x ray it releases (orange) travels sideways and is absorbed
under the neighbour. (b) Each energy deposit becomes a charge cloud; a cloud near a dexel edge
is itself split between dexels. (c) The energy each dexel collects: one photon, counted in more
than one dexel if more than one crosses the lowest threshold.
"""

# ╔═╡ 1ac6bf6f-9249-590c-8f75-7fd754caab4b
tracks = MC.record_tracks(
    crystal, 60.0;
    n = 300,
    beam,
    seed = 3,
)

# ╔═╡ 03f7a698-8e12-5541-ac92-3ebf124799f5
# A photon whose characteristic x ray was absorbed under a neighbouring dexel
shared_event = let
    in_central(x, y) = abs(x) <= readout.pitch_x_mm / 2 && abs(y) <= readout.pitch_y_mm / 2
    candidates = [
        s.event for s in tracks.steps
            if s.particle === :fluorescence && s.process == "phot" && !in_central(s.stop[1], s.stop[2]) &&
            0 <= s.stop[3] <= crystal.thickness_mm
    ]
    first(candidates)
end

# ╔═╡ c4a5a27a-e7b8-54f7-aa8b-7b3cf1059bb2
# The same photons again, now with every energy deposit recorded for the dexel model
deposits_run = MC.simulate_monoenergetic(
    crystal, 60.0;
    n = 300,
    beam,
    seed = 3,
    record_deposits = true,
)

# ╔═╡ 110f9a44-e004-5d43-a2b2-391a4bd18b99
single_response = MC.dexel_response(deposits_run, readout)

# ╔═╡ fb9ed65f-1ea8-557f-9703-76b935d46a75
let
    Mke.set_theme!(THEME)
    px, py = readout.pitch_x_mm, readout.pitch_y_mm
    σ = readout.cloud_sigma_um / 1000
    d = crystal.thickness_mm
    steps = [s for s in tracks.steps if s.event == shared_event]
    dep = deposits_run.deposits
    k = findall(==(shared_event), dep.event)
    fig = Mke.Figure(size = (1400, 620))

    # (a) Side view of the photon and its x ray, with the dexel boundaries
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The photon and its x ray",
        xlabel = "x, across the fan (mm)",
        ylabel = "depth z (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.6px, 1.6px), (-0.25, d + 0.1)),
    )
    Mke.poly!(
        axa,
        Mke.Rect2f(-1.5px, 0, 3px, d);
        color = PALETTE.slab,
        strokecolor = PALETTE.slab_edge,
        strokewidth = 1.5,
    )
    for i in -1:2
        Mke.lines!(
            axa,
            [(i - 0.5) * px, (i - 0.5) * px],
            [0, d];
            color = (PALETTE.slab_edge, 0.6),
            linestyle = :dash,
            linewidth = 1.5,
        )
    end
    for s in steps
        s.particle in (:primary, :fluorescence) || continue
        color = s.particle === :primary ? PALETTE.photon : PALETTE.xray
        Mke.lines!(
            axa,
            [s.start[1], s.stop[1]],
            [s.start[3], s.stop[3]];
            color,
            linewidth = 3,
        )
    end
    Mke.scatter!(
        axa,
        [Mke.Point2f(dep.x_mm[i], dep.z_mm[i]) for i in k];
        color = PALETTE.electron,
        markersize = 9,
    )
    note!(
        axa,
        -1.45px,
        d - 0.05,
        "dashed: dexel\nboundaries";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )

    # (b) Top view: the energy deposits and their charge clouds over the dexel grid
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Its charge clouds, from above",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.5px, 1.5px), (-1.5py, 1.5py)),
    )
    for i in -1:1, j in -1:1
        Mke.poly!(
            axb,
            Mke.Rect2f((i - 0.5) * px, (j - 0.5) * py, px, py);
            color = (:white, 0),
            strokecolor = PALETTE.slab_edge,
            strokewidth = 1.5,
        )
    end
    for i in k
        Mke.poly!(
            axb,
            Mke.Circle(Mke.Point2f(dep.x_mm[i], dep.y_mm[i]), 2σ);
            color = (PALETTE.electron, 0.12),
            strokecolor = (PALETTE.electron, 0.5),
            strokewidth = 1,
        )
    end
    Mke.scatter!(
        axb,
        [Mke.Point2f(dep.x_mm[i], dep.y_mm[i]) for i in k];
        color = PALETTE.electron,
        markersize = 6,
    )
    note!(
        axb,
        0,
        -1.4py,
        "circles: 2σ = 60 µm";
        align = (:center, :bottom),
        color = PALETTE.electron,
        fontsize = 14,
    )

    # (c) What each dexel collected, and which thresholds it crossed
    energies = single_response.dexel_energy_keV[shared_event, :, :]
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) What each dexel collects",
        xlabel = "x (dexel)",
        ylabel = "y (dexel)",
        aspect = Mke.DataAspect(),
        xticks = (1:3, ["−1", "0", "+1"]),
        yticks = (1:3, ["−1", "0", "+1"]),
    )
    Mke.heatmap!(
        axc,
        1:3,
        1:3,
        energies;
        colormap = :Greens,
        colorrange = (0, tracks.energy_keV),
    )
    for i in 1:3, j in 1:3
        e = energies[i, j]
        e > 0.5 || continue
        note!(
            axc,
            i,
            j,
            "$(round(e; digits = 1)) keV";
            align = (:center, :center),
            fontsize = 15,
        )
    end

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.24))
    Mke.colgap!(fig.layout, 30)
    fig
end

# ╔═╡ a1c0bfc5-a8eb-51f4-8a11-429b5604368d
md"""
## 3. The Response of One Dexel

The full simulation: at every whole keV from 1 to 140 keV (the grid of BasisSimulator's table),
100 000 photons land uniformly on the central dexel. Each run is reduced to the energy each of the nine dexels collected from each
photon right where it ran, on the eight workers; the deposits themselves never travel back.
"""

# ╔═╡ e9d84ab8-bfdf-59c7-813b-78b03c2c4a7a
energies = 1.0:1.0:140.0

# ╔═╡ efb0402d-648c-5ab8-a06d-f031e9550e90
responses = MC.simulate_sweep(
    [(crystal, E) for E in energies];
    n = 100_000,
    beam,
    seed = 1,
    readout,
)

# ╔═╡ 4f6dd6cb-cb42-5da9-9692-b257389cacf7
md"""
The spectrum each dexel sees, for 60 keV photons. The central dexel shows the full-energy peak,
the escape peaks (Cd and Te x rays that left it), and a low-energy tail from charge clouds split
across its edges. Its neighbours see the other side of the same events: the x rays it lost, and
the edge of the split clouds.
"""

# ╔═╡ b334ec12-f83a-59bc-8936-c7e16badc21c
spectrum_energy = 60.0

# ╔═╡ 6a162bd9-0b36-5cf6-b16e-6b6d095b5ee8
let
    Mke.set_theme!(THEME)
    r = responses[findfirst(r -> r.energy_keV == spectrum_energy, responses)]
    E0 = r.energy_keV
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "What each dexel collects from $(round(Int, E0)) keV photons landing on the central one",
        xlabel = "energy collected by the dexel (keV)",
        ylabel = "per keV, per photon",
        yscale = log10,
        limits = ((0, E0 + 3), (1.0e-5, 3)),
    )
    for (dexel, label, color) in (
            ((0, 0), "central dexel", PALETTE.photon),
            ((1, 0), "side neighbour (along x)", PALETTE.xray),
            ((0, 1), "side neighbour (along y)", PALETTE.kept),
            ((1, 1), "corner neighbour", PALETTE.muted),
        )
        s = MC.dexel_spectrum(r, dexel; bin_keV = 0.5)
        Mke.stairs!(
            ax,
            s.energy_keV,
            max.(s.density, 1.0e-9);
            step = :center,
            color,
            linewidth = 2,
            label,
        )
    end
    for e in MC.k_edges(crystal.material)
        note!(
            ax,
            E0 - e.kalpha_keV,
            0.6,
            "E − $(e.element) Kα";
            align = (:center, :bottom),
            color = PALETTE.xray,
            fontsize = 14,
        )
        note!(
            ax,
            e.kalpha_keV,
            0.2,
            "$(e.element) Kα";
            align = (:center, :bottom),
            color = PALETTE.xray,
            fontsize = 14,
        )
    end
    note!(
        ax,
        3,
        2.0e-5,
        "split charge clouds: anything from 0 to E";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )
    Mke.axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 65f1babc-4849-5e3a-8404-738562c88837
md"""
## 4. Compared with BasisSimulator's Current Table

BasisSimulator's photon-counting detector model uses `cdte_response_v4`: for photons of each
energy from 1 to 140 keV landing on a dexel, the mean number of counts above each of nine
thresholds (20, 25, 30, 35, 55, 60, 70, 75, 90 keV), in the 3 × 3 neighbourhood and per dexel,
from MCNP with a 30 µm Gaussian charge cloud and 1.5 keV electronic noise. Its file does not
record the dexel size or crystal depth; the native 0.275 × 0.322 mm dexels and 1.6 mm of CdTe
used here are the geometry it matches (the 2 × 2 binned dexels, 0.55 × 0.644 mm, do not).

The same thresholds and the same noise are applied to the Geant4 dexel signals below.
"""

# ╔═╡ 0eff78e6-df72-522a-b5db-d70421a6cdec
# BasisSimulator's bundled table, read from the file the simulator itself loads
current_table = Serialization.deserialize(BS.default_mc_drm_path())

# ╔═╡ 4e507ffb-04ca-5f56-a577-e2715de205a4
thresholds = current_table["thresholds_keV"]

# ╔═╡ 124978ab-3b89-5c61-996d-64d7f222d405
counts = [
    MC.threshold_counts(
            r,
            thresholds;
            electronic_noise_keV = current_table["sigma_elec_keV"],
        )
        for r in responses
]

# ╔═╡ 28a04c08-f2be-5b0b-8487-d1953b4ffca4
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    rows = Int.(E)                       # the table is on a 1-keV grid starting at 1 keV
    fig = Mke.Figure(size = (1200, 900))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Counts per photon in the 3 × 3 dexels, above each threshold",
        ylabel = "counts per photon",
        xticks = 10:10:140,
        limits = ((8, 142), (0, 1.3)),
    )
    colors = Mke.cgrad(:viridis, length(thresholds); categorical = true)
    for (t, threshold) in enumerate(thresholds)
        ours = [c.total[t] for c in counts]
        Mke.lines!(
            ax,
            E,
            ours;
            color = colors[t],
            linewidth = 2.5,
            label = "≥ $(round(Int, threshold)) keV",
        )
        Mke.scatter!(
            ax,
            E[1:3:end],
            current_table["R_total"][rows[1:3:end], t];
            color = colors[t],
            marker = :circle,
            markersize = 7,
            strokecolor = :white,
            strokewidth = 0.8,
        )
    end
    note!(
        ax,
        140,
        1.25,
        "lines: Geant4 (this notebook)    dots: cdte_response_v4 (MCNP)";
        align = (:right, :top),
        fontsize = 15,
    )
    note!(
        ax,
        12,
        1.1,
        "above 1: charge sharing makes one\nphoton count in two dexels";
        align = (:left, :center),
        fontsize = 14,
        color = PALETTE.muted,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center, title = "threshold")

    axd = Mke.Axis(
        fig[2, 1];
        xlabel = "photon energy (keV)",
        ylabel = "Geant4 − MCNP\n(counts per photon)",
        xticks = 10:10:140,
        limits = ((8, 142), (-0.06, 0.06)),
    )
    Mke.hspan!(axd, -0.01, 0.01; color = (PALETTE.muted, 0.15))
    Mke.hlines!(axd, 0; color = PALETTE.ink, linewidth = 1.5)
    for (t, _) in enumerate(thresholds)
        ours = [c.total[t] for c in counts]
        Mke.lines!(
            axd,
            E,
            ours .- current_table["R_total"][rows, t];
            color = colors[t],
            linewidth = 2,
        )
    end
    note!(
        axd,
        140,
        0.01,
        "grey band: ±0.01 counts per photon";
        align = (:right, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )

    Mke.linkxaxes!(ax, axd)
    Mke.hidexdecorations!(ax; grid = false)
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.32))
    fig
end

# ╔═╡ c0ec94cf-51a6-5967-8b5a-2a582ed1ebf9
let
    rows = Int.(collect(energies))
    differences = [counts[k].total[t] - current_table["R_total"][rows[k], t] for k in eachindex(rows), t in eachindex(thresholds)]
    largest = maximum(abs, differences)
    worst = Tuple(argmax(abs.(differences)))
    mean_abs = sum(abs, differences) / length(differences)

    Markdown.parse(
        """
        Over $(length(rows)) energies and $(length(thresholds)) thresholds, the mean difference between
        the Geant4 dexel model and `cdte_response_v4` is **$(round(mean_abs; digits = 4))** counts per
        photon, and the largest is **$(round(largest; digits = 3))**, at $(rows[worst[1]]) keV and the
        $(round(Int, thresholds[worst[2]])) keV threshold.
        """
    )
end

# ╔═╡ db9638a4-4721-5644-b84e-bec6fa1a80e3
md"""
## 5. The Anti-Scatter Grid

The Alpha's detector sits behind a two-dimensional grid of thin walls that stops photons
arriving at an angle (scattered in the patient) while letting through those coming straight
from the focal spot. Its walls enclose **4 × 6 dexels** (4 across the fan, 6 along z) and stand
over a narrow gap between neighbouring groups of dexels. Its dimensions are not published; the
gaps and wall thickness here are derived from the published fill factors (see *The NAEOTOM
Alpha, as Documented* above), the wall height (10 mm) and material (tungsten) are assumed.

BasisSimulator does not model the grid: it multiplies every dexel's signal by a fill factor of
0.95. With the grid, the photons that reach the crystal are those that miss the walls, which
stand over the gaps: per unit area of the detector, an open fraction of
`(4 × 0.275) × (6 × 0.322) / (1.36 × 2.182)` ≈ **0.72**. A dexel at the edge of a cell sits next
to a wall, one in the middle between two dexels; by symmetry there are six different positions
in a cell (A–F below), and every dexel of the detector is one of them.
"""

# ╔═╡ 9b183cd4-f849-5eb9-bbc9-f62a327ba741
# The Alpha's dexels and grid cells at the detector face (mm)
alpha = (
    dexel = (0.275, 0.322),
    group = (4, 6),
    gap = (0.26, 0.25),
    wall_height = 10.0,
)

# ╔═╡ 26699732-4c76-5ecd-ae55-ed5aff70d36f
cell_pitch = (
    alpha.group[1] * alpha.dexel[1] + alpha.gap[1],
    alpha.group[2] * alpha.dexel[2] + alpha.gap[2],
)

# ╔═╡ 8bc9b5e0-ff52-51d2-b662-82db9b772e06
# The Alpha's grid as seen from a dexel at cone angle θ (degrees): focused, so its walls lean
# with the rays
grid_at(θ) = MC.Collimator(;
    height_mm = alpha.wall_height,
    wall_mm = alpha.gap[1],
    pitch_x_mm = cell_pitch[1],
    pitch_y_mm = cell_pitch[2],
    walls = :both,
    cells = 5,
    tilt_y_deg = θ,
)

# ╔═╡ 43251201-987e-59a2-ba3e-9e1785549002
grid = grid_at(0.0)

# ╔═╡ c5fb6ae2-0681-5f93-a2d7-05dffa64530e
alpha_crystal = MC.SlabDetector(:CdTe, 1.6; collimator = grid)

# ╔═╡ d952031c-89aa-5eb9-9697-ef27eff6d6b0
# The six distinct positions in a cell: column 0 or 1 (2 and 3 mirror them), row 0, 1 or 2 (3–5 mirror)
positions = [
    (label = "A", target = (0, 0)),
    (label = "B", target = (1, 0)),
    (label = "C", target = (0, 1)),
    (label = "D", target = (1, 1)),
    (label = "E", target = (0, 2)),
    (label = "F", target = (1, 2)),
]

# ╔═╡ 9f9f79cf-284d-5ef9-98c7-5cf015924059
grid_readouts = [
    MC.DexelReadout(;
            pitch_x_mm = alpha.dexel[1],
            pitch_y_mm = alpha.dexel[2],
            group = alpha.group,
            gap_mm = alpha.gap,
            target = p.target,
            neighborhood = 3,
            cloud_sigma_um = 30.0,
        )
        for p in positions
]

# ╔═╡ e5b42b5a-564f-5ba8-b8a1-6a302df8e3ad
let
    Mke.set_theme!(THEME)
    p, G, g = alpha.dexel, alpha.group, alpha.gap
    H = alpha.wall_height
    fig = Mke.Figure(size = (1400, 700))
    centre(m, axis) = MC.dexel_centre(m, p[axis], G[axis], g[axis])
    tungsten = Mke.RGBf(0.35, 0.35, 0.4)

    # (a) One grid cell in 3-D: the walls standing on the gaps, the dexels below
    ax3 = Mke.Axis3(
        fig[1, 1];
        title = "(a) One grid cell: 4 × 6 dexels",
        aspect = :data,
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        zlabel = "z (mm)",
        azimuth = 1.25π,
        elevation = 0.2π,
        zreversed = true,
        protrusions = (30, 10, 30, 30),
    )
    for i in 0:(G[1] - 1), j in 0:(G[2] - 1)
        block = Mke.Rect3f(
            Mke.Vec3f(centre(i, 1) - p[1] / 2, centre(j, 2) - p[2] / 2, 0),
            Mke.Vec3f(p[1], p[2], 1.6),
        )
        Mke.mesh!(ax3, block; color = (PALETTE.photon, 0.18))
        Mke.wireframe!(ax3, block; color = PALETTE.slab_edge, linewidth = 0.6)
    end
    half_x, half_y = cell_pitch[1] / 2, cell_pitch[2] / 2
    for sx in (-1, 1)
        wall = Mke.Rect3f(
            Mke.Vec3f(sx * half_x - g[1] / 2, -half_y - g[2] / 2, -H),
            Mke.Vec3f(g[1], cell_pitch[2] + g[2], H),
        )
        Mke.mesh!(ax3, wall; color = (tungsten, 0.3))
    end
    for sy in (-1, 1)
        wall = Mke.Rect3f(
            Mke.Vec3f(-half_x + g[1] / 2, sy * half_y - g[2] / 2, -H),
            Mke.Vec3f(cell_pitch[1] - g[1], g[2], H),
        )
        Mke.mesh!(ax3, wall; color = (tungsten, 0.2))
    end
    Mke.text!(
        ax3,
        Mke.Point3f(half_x, -half_y, -H);
        text = "tungsten walls,\n$(H) mm tall",
        align = (:left, :center),
        offset = (8, 0),
        fontsize = 15,
        color = tungsten,
    )
    Mke.text!(
        ax3,
        Mke.Point3f(half_x, -half_y, 1.6);
        text = "CdTe, 1.6 mm",
        align = (:left, :center),
        offset = (8, 0),
        fontsize = 15,
        color = PALETTE.photon,
    )

    # (b) Seen from the beam: three by two cells, and the six distinct positions
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the beam: the six kinds of dexel",
        xlabel = "x, across the fan (mm)",
        ylabel = "y, along z (mm)",
        aspect = Mke.DataAspect(),
    )
    for cx in -1:1, cy in -1:0
        x0 = cx * cell_pitch[1] - half_x
        y0 = (cy + 0.5) * cell_pitch[2] - half_y
        Mke.poly!(
            axb,
            Mke.Rect2f(x0, y0, cell_pitch[1], cell_pitch[2]);
            color = (tungsten, 0.55),
            strokewidth = 0,
        )
        for i in 0:(G[1] - 1), j in 0:(G[2] - 1)
            x = cx * cell_pitch[1] + centre(i, 1)
            y = (cy + 0.5) * cell_pitch[2] + centre(j, 2)
            Mke.poly!(
                axb,
                Mke.Rect2f(x - p[1] / 2, y - p[2] / 2, p[1], p[2]);
                color = :white,
                strokecolor = PALETTE.slab_edge,
                strokewidth = 0.8,
            )
        end
    end
    mirror(k, n) = min(k, n - 1 - k)
    letter(i, j) = positions[findfirst(q -> q.target == (mirror(i, G[1]), mirror(j, G[2])), positions)].label
    for i in 0:(G[1] - 1), j in 0:(G[2] - 1)
        Mke.text!(
            axb,
            centre(i, 1),
            centre(j, 2) + 0.5 * cell_pitch[2];
            text = letter(i, j),
            align = (:center, :center),
            fontsize = 15,
            font = :bold,
            color = PALETTE.photon,
        )
    end
    note!(
        axb,
        1.5 * cell_pitch[1],
        cell_pitch[2] + 0.1,
        "grey: wall over a gap";
        align = (:right, :bottom),
        color = tungsten,
        fontsize = 14,
    )

    # (c) Side view of a cell along x: walls over the gaps, a photon blocked, one let through
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side view",
        xlabel = "x (mm)",
        ylabel = "z (mm)",
        yreversed = true,
        limits = ((-0.85 * cell_pitch[1], 0.85 * cell_pitch[1]), (-H - 1, 2.6)),
    )
    Mke.poly!(
        axc,
        Mke.Rect2f(-0.85 * cell_pitch[1], 0, 1.7 * cell_pitch[1], 1.6);
        color = PALETTE.slab,
        strokecolor = PALETTE.slab_edge,
    )
    for sx in (-1, 1)
        Mke.poly!(axc, Mke.Rect2f(sx * half_x - g[1] / 2, -H, g[1], H); color = tungsten)
    end
    for i in 0:(G[1] - 1)
        Mke.lines!(
            axc,
            [centre(i, 1) - p[1] / 2 + 0.01, centre(i, 1) + p[1] / 2 - 0.01],
            [1.6, 1.6];
            color = PALETTE.slab_edge,
            linewidth = 6,
        )
    end
    Mke.lines!(axc, [0.1, 0.1], [-H - 0.8, 0.7]; color = PALETTE.photon, linewidth = 3)
    Mke.lines!(axc, [half_x, half_x], [-H - 0.8, -4]; color = PALETTE.photon, linewidth = 3)
    Mke.scatter!(
        axc,
        [Mke.Point2f(half_x, -4)];
        marker = :xcross,
        color = PALETTE.electron,
        markersize = 16,
    )
    note!(
        axc,
        0.12,
        -H + 0.5,
        "straight from\nthe focus:\npasses";
        align = (:left, :top),
        color = PALETTE.photon,
        fontsize = 13,
    )
    note!(
        axc,
        half_x - 0.05,
        -2.5,
        "on a wall:\nstopped";
        align = (:right, :top),
        color = PALETTE.electron,
        fontsize = 13,
    )

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.3))
    Mke.colsize!(fig.layout, 3, Mke.Relative(0.22))
    fig
end

# ╔═╡ d09b836e-5751-5269-b5c0-c3ad1f8a94ef
md"""
The response of each of the six positions, with photons landing uniformly over the dexel's
electrode (what an air scan delivers to it). Six positions × 66 energies, 100 000 photons
each, on the workers.
"""

# ╔═╡ 888de7f9-befc-515e-a23a-f7db676ac39e
grid_energies = 10.0:2.0:140.0

# ╔═╡ 4c53f560-8613-5038-b0d9-25180b502e73
grid_responses = [
    MC.simulate_sweep(
            [(alpha_crystal, E) for E in grid_energies];
            n = 100_000,
            beam = MC.dexel_beam(ro),
            seed = 1,
            readout = ro,
        )
        for ro in grid_readouts
]

# ╔═╡ 87a3f02f-8d85-57ec-ab9b-78b53a80b7c0
md"""
The six positions count almost the same: the walls shade the gaps, not the dexels. What the grid
changes for an air scan is the share of the detector that sees photons (0.72 instead of
BasisSimulator's 0.95), and a slight loss at the edge of a cell, where a split charge cloud's
far half falls in the gap under the wall instead of in a neighbour.
"""

# ╔═╡ 71102eb4-ce19-584d-809f-e69e4f3561bd
grid_counts = [
    [MC.threshold_counts(r, thresholds; electronic_noise_keV = 1.5) for r in responses_at]
        for responses_at in grid_responses
]

# ╔═╡ 650a008b-622d-5948-8485-96d50b16b5b9
let
    Mke.set_theme!(THEME)
    E = collect(grid_energies)
    fig = Mke.Figure(size = (1200, 620))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Counts above 20 keV in the dexel, per photon landing on it",
        xlabel = "photon energy (keV)",
        ylabel = "counts per photon",
        xticks = 10:10:140,
        limits = ((8, 142), (0, 1.05)),
    )
    no_grid = [c.per_dexel[2, 2, 1] for c in counts]
    Mke.lines!(
        ax,
        collect(energies),
        no_grid;
        color = PALETTE.muted,
        linestyle = :dash,
        linewidth = 3,
        label = "no grid (section 3)",
    )
    colors = Mke.cgrad(:tableau_10, 6; categorical = true)
    for (k, pos) in enumerate(positions)
        Mke.lines!(
            ax,
            E,
            [c.per_dexel[2, 2, 1] for c in grid_counts[k]];
            color = colors[k],
            linewidth = 2.5,
            label = "position $(pos.label)",
        )
    end
    note!(
        ax,
        140,
        0.05,
        "every position counts almost the same: the walls stand over the gaps, not over dexels";
        align = (:right, :bottom),
        fontsize = 14,
        color = PALETTE.muted,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 46b15a3d-c453-593c-bb39-2b0659dc1631
md"""
## 6. Photons Arriving at an Angle

The Alpha's detector is an arc centred on the focal spot, so across the fan every dexel faces
the source head on. Along z it is flat: a dexel in an outer row sees the source at the **cone
angle**, up to `atan(52 mm / 1113 mm)` ≈ 2.7° at the detector's edge. The grid is focused, so its
walls lean with the rays and do not shadow more; what changes is the path in the crystal: a
photon absorbed deep in the CdTe has drifted sideways into the next row.
"""

# ╔═╡ c6682374-bbef-5638-9592-82c5c8e5aff1
let
    Mke.set_theme!(THEME)
    sdd = 1113.0
    half_z = 24 * cell_pitch[2]
    fig = Mke.Figure(size = (1200, 460))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Side view of the scanner along z (not to scale across)",
        xlabel = "distance from the focal spot (mm)",
        ylabel = "z (mm)",
        limits = ((-40, sdd + 230), (-1.6 * half_z, 1.6 * half_z)),
    )
    Mke.scatter!(ax, [Mke.Point2f(0, 0)]; color = PALETTE.xray, markersize = 18)
    Mke.lines!(ax, [sdd, sdd], [-half_z, half_z]; color = PALETTE.slab_edge, linewidth = 10)
    for zt in (-half_z, -half_z / 2, 0, half_z / 2, half_z)
        Mke.lines!(ax, [0, sdd], [0, zt]; color = (PALETTE.photon, 0.6), linewidth = 1.5)
    end
    angle = round(atand(half_z / sdd); digits = 2)
    note!(
        ax,
        10,
        6,
        "focal spot";
        align = (:left, :bottom),
        color = PALETTE.xray,
        fontsize = 15,
    )
    note!(
        ax,
        sdd - 10,
        half_z,
        "outer row: $(angle)° cone angle";
        align = (:right, :bottom),
        color = PALETTE.photon,
        fontsize = 15,
    )
    note!(
        ax,
        sdd + 15,
        0,
        "detector,\n$(round(2half_z; digits = 1)) mm\nalong z";
        align = (:left, :center),
        color = PALETTE.slab_edge,
        fontsize = 15,
    )
    fig
end

# ╔═╡ f167cd8c-aea8-5090-ab71-7a82f303c13a
cone_angles = [0.0, 1.0, 2.0, 2.7]

# ╔═╡ a7975678-b7d2-52c0-a314-7c1bc4549cff
# Position F (the middle of a cell), at each cone angle, with the grid's walls leaning with the rays
cone_responses = [
    MC.simulate_sweep(
            [(MC.SlabDetector(:CdTe, 1.6; collimator = grid_at(θ)), E) for E in grid_energies];
            n = 100_000,
            beam = MC.dexel_beam(grid_readouts[6]; angle_y_deg = θ),
            seed = 1,
            readout = grid_readouts[6],
        )
        for θ in cone_angles
]

# ╔═╡ 49a1c7cc-171e-5724-a834-f334c5e9c2ea
let
    Mke.set_theme!(THEME)
    E = collect(grid_energies)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Position F: where the counts land, by cone angle",
        xlabel = "photon energy (keV)",
        ylabel = "counts above 20 keV per photon",
        xticks = 10:10:140,
        limits = ((8, 142), (0, 1.0)),
    )
    colors = Mke.cgrad(:viridis, length(cone_angles); categorical = true)
    for (k, θ) in enumerate(cone_angles)
        c = [MC.threshold_counts(r, [20.0]) for r in cone_responses[k]]
        Mke.lines!(
            ax,
            E,
            [x.per_dexel[2, 2, 1] for x in c];
            color = colors[k],
            linewidth = 3,
            label = "$(θ)°: the dexel aimed at",
        )
        Mke.lines!(
            ax,
            E,
            [x.per_dexel[2, 3, 1] for x in c];
            color = colors[k],
            linewidth = 2,
            linestyle = :dash,
            label = "$(θ)°: the next row",
        )
    end
    note!(
        ax,
        140,
        0.35,
        "the deeper the photon goes (high energy),\nthe further it drifts into the next row";
        align = (:right, :center),
        fontsize = 14,
        color = PALETTE.muted,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center, nbanks = 1)
    fig
end

# ╔═╡ 25fb922f-615e-5b76-9048-7acf73bb0f3a
md"""
## 7. The 2 × 2 Binned Dexels

In its standard mode the Alpha reads its dexels out in groups of 2 × 2: each of the four
counts its own photons against the thresholds, and the group reports the sum. Binning after
counting keeps charge sharing *inside* the group as double counting: a photon split between two
dexels of the same group still counts twice. A grid cell holds 2 × 3 binned dexels, of two kinds:
a corner group (A, B, C, D) and an edge-middle group (E, F and their mirrors).
"""

# ╔═╡ 39f04615-41a5-5f60-894d-f8e636e68f8c
# Counts of a binned dexel for photons landing uniformly on it: the four members' counts, averaged
# over the four landing dexels (by symmetry, two targets of each group are mirrors of the other two)
binned_counts = let
    corner = [1, 2, 3, 4]       # A, B, C, D: the corner group, members (0,0) (1,0) (0,1) (1,1)
    middle = [5, 6]             # E, F: the middle group, members (0,2) (1,2) and their mirrors (0,3) (1,3)
    member_offsets = Dict(
        1 => [(0, 0), (1, 0), (0, 1), (1, 1)],
        2 => [(-1, 0), (0, 0), (-1, 1), (0, 1)],
        3 => [(0, -1), (1, -1), (0, 0), (1, 0)],
        4 => [(-1, -1), (0, -1), (-1, 0), (0, 0)],
        5 => [(0, 0), (1, 0), (0, 1), (1, 1)],
        6 => [(-1, 0), (0, 0), (-1, 1), (0, 1)],
    )
    group_sum(k, c) = sum(c.per_dexel[2 + dx, 2 + dy, :] for (dx, dy) in member_offsets[k])
    average(ks, e) = sum(group_sum(k, grid_counts[k][e]) for k in ks) / length(ks)
    (
        corner = [average(corner, e) for e in eachindex(grid_energies)],
        middle = [average(middle, e) for e in eachindex(grid_energies)],
    )
end

# ╔═╡ 52e68256-0954-5993-b7ea-a0ef64dbde97
let
    Mke.set_theme!(THEME)
    E = collect(grid_energies)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "A binned (2 × 2) dexel: counts above each threshold, per photon landing on it",
        xlabel = "photon energy (keV)",
        ylabel = "counts per photon",
        xticks = 10:10:140,
        limits = ((8, 142), (0, 1.25)),
    )
    colors = Mke.cgrad(:viridis, length(thresholds); categorical = true)
    for (t, threshold) in enumerate(thresholds)
        Mke.lines!(
            ax,
            E,
            [c[t] for c in binned_counts.corner];
            color = colors[t],
            linewidth = 3,
            label = "≥ $(round(Int, threshold)) keV",
        )
        Mke.lines!(
            ax,
            E,
            [c[t] for c in binned_counts.middle];
            color = colors[t],
            linewidth = 2,
            linestyle = :dash,
        )
    end
    note!(
        ax,
        140,
        1.2,
        "solid: corner group   dashed: middle group";
        align = (:right, :top),
        fontsize = 14,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center, title = "threshold")
    fig
end

# ╔═╡ 9133ec80-9f4b-5fbd-8d36-626ca9c12719
md"""
## 8. The Whole Detector

Everything so far happened in one small patch of crystal. The real detector is an arc of CdTe
around the focal spot: detector A, **2752 dexel columns × 288 rows**, 1113 mm from the focal
spot, and the dual-source systems' detector B, **1984 columns**, the same rows and dexels. In
Geant4 it is built as it is: one shell of CdTe, 689 tungsten plates between the columns of grid
cells, each pointing at the focal spot, and a wall segment between the rows of every cell,
tilted toward the focal spot (33 712 of them for detector A).

An **air scan** (photons straight from the focal spot, nothing in between) illuminates every
dexel equally, so it is the test of the whole detector: every dexel should behave as one of the
six positions of section 5, at the cone angle of its row.
"""

# ╔═╡ 7081fb53-7350-5ef5-82fc-c3866ca8dc5a
detector_a = MC.ArcDetector(; columns = 2752, rows = 288)

# ╔═╡ bfee1bcb-7273-5f67-b684-42e8be79191d
detector_b = MC.ArcDetector(; columns = 1984, rows = 288)

# ╔═╡ c0f06895-b97e-5bc3-81dd-bdd5b1a8a0c3
let
    row(name, d) = string(
        "| $(name) | $(d.columns) × $(d.rows) | $(round(MC.fan_angle_deg(d); digits = 1))° ",
        "| $(round(MC.face_size_mm(d)[1]; digits = 1)) × $(round(MC.face_size_mm(d)[2]; digits = 1)) mm ",
        "| $(MC.cell_count(d)[1]) × $(MC.cell_count(d)[2]) |",
    )
    Markdown.parse(
        """
        | detector | dexels (columns × rows) | fan angle | front face (arc × z) | grid cells |
        |---|---|---|---|---|
        $(row("A", detector_a))
        $(row("B", detector_b))
        """
    )
end

# ╔═╡ 39c48e80-8b7e-5910-9909-463d8f715b0f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 700))
    R = detector_a.radius_mm

    # (a) The gantry from the patient's feet: focal spot and the two detector arcs
    ax = Mke.Axis(
        fig[1, 1];
        title = "(a) The detectors around the focal spot",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-620, 620), (-150, 1420)),
    )
    for (d, color, label) in ((detector_a, PALETTE.photon, "detector A"), (detector_b, PALETTE.xray, "detector B"))
        half = deg2rad(MC.fan_angle_deg(d)) / 2
        θ = range(-half, half; length = 200)
        r = d === detector_a ? R : R + 60
        Mke.lines!(ax, r .* sin.(θ), r .* cos.(θ); color, linewidth = 8, label)
        for t in (-half, half)
            Mke.lines!(
                ax,
                [0, r * sin(t)],
                [0, r * cos(t)];
                color = (color, 0.5),
                linewidth = 1.5,
            )
        end
    end
    Mke.scatter!(ax, [Mke.Point2f(0, 0)]; color = PALETTE.xray, markersize = 18)
    Mke.poly!(
        ax,
        Mke.Circle(Mke.Point2f(0, 610), 250);
        color = (PALETTE.muted, 0.08),
        strokecolor = PALETTE.muted,
        strokewidth = 1,
        linestyle = :dash,
    )
    note!(
        ax,
        0,
        610,
        "50 cm scan field\n(detector A)";
        align = (:center, :center),
        color = PALETTE.muted,
        fontsize = 14,
    )
    note!(
        ax,
        15,
        -20,
        "focal spot";
        align = (:left, :top),
        color = PALETTE.xray,
        fontsize = 15,
    )
    note!(
        ax,
        0,
        R + 110,
        "B is drawn outside A for clarity:\nboth sit 1113 mm from their own focal spot";
        align = (:center, :bottom),
        fontsize = 13,
        color = PALETTE.muted,
    )
    Mke.axislegend(ax; position = :lb)

    # (b) A patch of detector A seen from the focal spot: dexels, gaps and walls, to scale
    pitch = MC.cell_pitch(detector_a)
    p, G, g = detector_a.dexel_mm, detector_a.group, detector_a.gap_mm
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) A patch of the face: 3 × 2 of its 688 × 48 grid cells",
        xlabel = "along the arc (mm)",
        ylabel = "z (mm)",
        aspect = Mke.DataAspect(),
    )
    tungsten = Mke.RGBf(0.35, 0.35, 0.4)
    Mke.poly!(axb, Mke.Rect2f(0, 0, 3 * pitch[1], 2 * pitch[2]); color = (tungsten, 0.7))
    for cx in 0:2, cy in 0:1, i in 0:(G[1] - 1), j in 0:(G[2] - 1)
        x = cx * pitch[1] + g[1] / 2 + i * p[1]
        y = cy * pitch[2] + g[2] / 2 + j * p[2]
        Mke.poly!(
            axb,
            Mke.Rect2f(x, y, p[1], p[2]);
            color = Mke.RGBf(0.86, 0.9, 0.97),
            strokecolor = PALETTE.slab_edge,
            strokewidth = 0.8,
        )
    end
    note!(
        axb,
        3 * pitch[1],
        2 * pitch[2],
        "blue: dexels   grey: walls over the gaps";
        align = (:right, :bottom),
        fontsize = 14,
        offset = (0, 6),
    )

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.55))
    fig
end

# ╔═╡ 2ae8390f-0dcc-5f4b-85ca-ca5cb562bb98
md"""
The air scan: 60 keV photons from the focal spot, spread evenly over detector A, about 30 per
dexel (24 million photons in 48 chunks of 500 000, each on its own seed, spread over the
workers). Each chunk is counted where it ran and only the dexel counts travel back.
"""

# ╔═╡ 1e681426-7ee2-5fe5-a150-92b684d4aed3
air_readout = MC.ArcReadout(;
    thresholds_keV = [20.0, 35.0, 55.0, 70.0],
    cloud_sigma_um = 30.0,
    electronic_noise_keV = 1.5,
)

# ╔═╡ c96473dd-3ca7-54aa-87ce-af04055fc5d1
air_scan_60 = sum(
    MC.simulate_sweep(
        [(detector_a, 60.0, chunk) for chunk in 1:48];
        n = 500_000,
        beam = MC.focal_spot(detector_a),
        readout = air_readout,
    )
)

# ╔═╡ 2ffccd1c-e5ea-5da2-b746-7ee1f88daeb5
let
    Mke.set_theme!(THEME)
    c = air_scan_60
    # photons that landed on each dexel's electrode: the share of the face its electrode covers
    on_electrode = c.n * prod(detector_a.dexel_mm) / prod(MC.face_size_mm(detector_a))
    per_photon = c.counts[:, :, 1] ./ on_electrode
    fig = Mke.Figure(size = (1400, 760))

    # (a) The whole detector's counts per dexel (normalised to the mean photons per dexel)
    ax = Mke.Axis(
        fig[1, 1:2];
        title = "(a) Every dexel of detector A: counts above 20 keV per photon landing on it",
        xlabel = "dexel column",
        ylabel = "row",
    )
    hm = Mke.heatmap!(
        ax,
        1:detector_a.columns,
        1:detector_a.rows,
        per_photon;
        colormap = :viridis,
        colorrange = (0.6, 1.3),
    )
    Mke.Colorbar(fig[1, 3], hm; label = "counts per photon")

    # (b) A close-up: three grid cells across, two along z
    axz = Mke.Axis(
        fig[2, 1];
        title = "(b) Close-up: the 4 × 6 cells",
        xlabel = "dexel column",
        ylabel = "row",
        aspect = Mke.DataAspect(),
    )
    cols = 1373:1384
    rows = 133:144
    Mke.heatmap!(
        axz,
        cols,
        rows,
        per_photon[cols, rows];
        colormap = :viridis,
        colorrange = (0.6, 1.3),
    )

    # (c) Averaged over the detector by position in the cell, against the six-position runs of section 5
    axc = Mke.Axis(
        fig[2, 2];
        title = "(c) By position in the cell",
        xlabel = "position in the grid cell",
        ylabel = "counts above 20 keV per photon",
        xticks = (1:6, [p.label for p in positions]),
        limits = (nothing, (0.95, 1.3)),
    )
    mirror(k, n) = min(k, n - 1 - k)
    sums = zeros(6)
    numbers = zeros(Int, 6)
    for col in 1:detector_a.columns, row in 1:detector_a.rows
        i, j = mod(col - 1, 4), mod(row - 1, 6)
        k = findfirst(q -> q.target == (mirror(i, 4), mirror(j, 6)), positions)
        sums[k] += per_photon[col, row]
        numbers[k] += 1
    end
    e60 = findfirst(==(60.0), collect(grid_energies))
    Mke.scatter!(
        axc,
        1:6,
        sums ./ numbers;
        color = PALETTE.photon,
        markersize = 16,
        label = "whole detector (this air scan)",
    )
    Mke.scatter!(
        axc,
        1:6,
        [grid_counts[k][e60].total[1] for k in 1:6];
        color = PALETTE.xray,
        marker = :diamond,
        markersize = 16,
        label = "one dexel at a time, 3 × 3 total (section 5)",
    )
    Mke.axislegend(axc; position = :lb)

    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.45))
    fig
end

# ╔═╡ dd7dbc05-6496-563a-ab62-46a547a22324
md"""
In an air scan a dexel counts its own photons *and* the shares of its neighbours' photons that
reach it: about 1.15 counts per photon landing on it at 60 keV, not the 0.95 of its own. By
symmetry (what a neighbour's photon gives this dexel equals what this dexel's photon gives the
neighbour), that is the 3 × 3 total of the one-dexel runs, which panel (c) compares. The whole
detector agrees with the six dexels simulated one at a time: a full detector's response is the
six positions' responses, at each row's cone angle. That is what lets a response table be built
from six dexels instead of 792 576.
"""

# ╔═╡ c89fca63-6462-5f25-b16e-d491689a48a3
md"""
## 9. The Alpha's X-Ray Beam

The photons come from BasisSimulator itself: its IPEM Report 78 tungsten spectrum at 140 kVp,
through the scanner's 3 mm of aluminium and the protocol's 0.9 mm of titanium, scaled to the
detector at 1113 mm, exactly the spectrum `simulate!` uses for the Alpha (BasisSimulator's notebooks 04 and 08).
The IPEM spectrum's anode angle is BasisSimulator's default, 10°; the Alpha's anode is 8°
(⚠ *discrepancy* above), so the 8° spectrum is shown beside it.
"""

# ╔═╡ 5d8d5cd1-8ff7-5356-87bb-24b57c3281f1
# BasisSimulator's Alpha, as on its scanner page (BasisSimulator's notebooks 04 and 08)
bs_scanner = let native_col = 0.275, native_row = 0.322, sid = 610.0, sdd = 1113.0, bf = 2
    col_iso = native_col * bf / (sdd / sid)
    row_iso = native_row * bf / (sdd / sid)
    BS.PCCTScanner(
        source_to_isocenter = sid,
        source_to_detector = sdd,
        detector_rows = 144,
        detector_cols = ceil(Int, 500.0 / col_iso),
        detector_row_size = row_iso,
        detector_col_size = col_iso,
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
        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        energy_resolution = 10.0,
        charge_sharing_fwhm = 0.08,
        dead_time_ns = 5.0,
        pixel_mode = :standard,
        native_dexel_col_mm = native_col,
        native_dexel_row_mm = native_row,
        binning_factor = bf,
        pileup_correction = true,
        scatter_correction = true,
        noise_reduction = 0.0,
    )
end

# ╔═╡ 1d5d02ab-485b-567f-ac5d-aaea105040a4
bs_protocol(anode_angle) = BS.CTProtocol(;
    kVp = 140,
    mA = 174.0,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    anode_angle,
    additional_filters = [("Ti", 0.9)],
)

# ╔═╡ 15bdf1fd-777f-50c0-ad69-59c912741e2e
# Photons per mAs per mm² at the detector, per 0.5 keV bin, for each anode angle
spectra = Dict(
    angle => BS.resolve_source_spectrum_without_bowtie(
            BS.SimOptions(),
            bs_protocol(angle);
            scanner = bs_scanner,
        )
        for angle in (10, 8)
)

# ╔═╡ 877c4989-7b59-5741-a50a-433cf4a4a14b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The photons reaching the detector: 140 kVp, 3 mm Al + 0.9 mm Ti",
        xlabel = "photon energy (keV)",
        ylabel = "photons per mAs per mm²\nper keV, at 1113 mm",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    for (angle, color, label) in ((10, PALETTE.photon, "anode 10° (BasisSimulator)"), (8, PALETTE.xray, "anode 8° (documented)"))
        e, w = spectra[angle]
        bin = e[2] - e[1]
        Mke.lines!(ax, e, w ./ bin; color, linewidth = 3, label)
    end
    for t in air_readout.thresholds_keV
        Mke.vlines!(ax, t; color = (PALETTE.muted, 0.6), linestyle = :dash, linewidth = 1.5)
    end
    note!(
        ax,
        21,
        0,
        "thresholds: 20, 35, 55, 70 keV";
        align = (:left, :bottom),
        offset = (0, 8),
        color = PALETTE.muted,
        fontsize = 14,
    )
    note!(
        ax,
        59,
        0,
        "tungsten K lines";
        align = (:center, :bottom),
        offset = (0, 230),
        color = PALETTE.ink,
        fontsize = 14,
    )
    Mke.axislegend(ax; position = :rt)
    fig
end

# ╔═╡ 848bfaff-246d-4084-989c-c17ef4008e41
md"""
**The beam's quality, against measurement.** A beam's spectrum is checked against a scanner by its
half-value layer (HVL): the thickness of aluminium that halves its air kerma, measured at the
isocentre, where the beam has passed the bowtie's thinnest part (2.5 mm of aluminium in
BasisSimulator's large-body bowtie). The Alpha's measured HVL at 140 kV is 7.9 mm without the table
in the beam and 7.7 mm with it (Li 2025; Zhan 2026).
"""

# ╔═╡ 648db0cd-57fc-46a7-894e-06ab540140a2
# The half-value layer at the isocentre: BasisSimulator's beam through the bowtie's centre, with
# and without the protocol's 0.9 mm of titanium
hvl_check = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium = MC.DetectorMaterial(:Al)

    function hvl(anode_angle, filters)
        protocol = BS.CTProtocol(;
            kVp = 140,
            mA = 174.0,
            views = 1200,
            rotation_time = 0.5,
            collimation_mm = 5.0,
            anode_angle,
            additional_filters = filters,
        )
        e, w = BS.resolve_source_spectrum_without_bowtie(
            BS.SimOptions(),
            protocol;
            scanner = bs_scanner,
        )
        μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
        return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
    end

    [
        (
            label = "BasisSimulator: anode 10°,\n3 mm Al + 0.9 mm Ti",
            hvl = hvl(10, [("Ti", 0.9)]),
        ),
        (
            label = "anode 8° (documented),\n3 mm Al + 0.9 mm Ti",
            hvl = hvl(8, [("Ti", 0.9)]),
        ),
        (label = "anode 8°,\n3 mm Al, no titanium", hvl = hvl(8, Tuple{String, Float64}[])),
    ]
end

# ╔═╡ f68a9548-e730-4402-af6d-ed2f2a2dc3a9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-value layer at the isocentre, 140 kVp",
        xlabel = "half-value layer (mm of aluminium)",
        yticks = (1:3, [c.label for c in hvl_check]),
        yreversed = true,
        limits = ((0, 12), (0.4, 4.2)),
    )
    Mke.vspan!(ax, 7.7, 7.9; color = (PALETTE.kept, 0.35))
    note!(
        ax,
        7.8,
        3.85,
        "measured on the Alpha: 7.7–7.9 mm";
        align = (:center, :center),
        color = PALETTE.kept,
        fontsize = 16,
    )
    Mke.barplot!(
        ax,
        1:3,
        [c.hvl for c in hvl_check];
        direction = :x,
        width = 0.6,
        color = [PALETTE.reference, PALETTE.xray, PALETTE.photon],
    )
    for (k, c) in enumerate(hvl_check)
        note!(
            ax,
            c.hvl,
            k,
            "$(round(c.hvl; digits = 1)) mm";
            align = (:left, :center),
            offset = (8, 0),
            fontsize = 16,
            alpha = 0.0,
        )
    end
    fig
end

# ╔═╡ 0e2ac434-ad27-4763-bbb7-f1a231b06087
let
    basis, documented, without = (c.hvl for c in hvl_check)
    Markdown.parse(
        """
        **⚠ Discrepancy: BasisSimulator's Alpha beam is too hard.** At the isocentre its 140 kVp beam has a
        half-value layer of **$(round(basis; digits = 1)) mm** of aluminium ($(round(documented; digits = 1)) mm with the
        documented 8° anode), against the measured 7.7–7.9 mm. Without the protocol's 0.9 mm of titanium it is
        **$(round(without; digits = 1)) mm**, on the measurement. The titanium is in BasisSimulator's Alpha protocol;
        the measured HVL says the Alpha's 140 kVp beam carries no such extra filtration. The counts below keep
        BasisSimulator's beam, so the two stay comparable.
        """
    )
end

# ╔═╡ 0534c5aa-7947-53bd-b463-865d06524b9c
md"""
**Counts per bin, three ways.** For photons of this spectrum landing on the detector:

- **BasisSimulator today**: its response matrix (`compute_mc_drm`, built from `cdte_response_v4`)
  folded with the spectrum, times its fill factor (0.95);
- **Geant4, one dexel**: the same calculation with section 3's Geant4 response (no grid);
- **Geant4, the whole detector**: an air scan of all of detector A with the spectrum itself,
  grid and all, counted dexel by dexel (24 million photons).

All three are per photon arriving at the detector's front, per native dexel.
"""

# ╔═╡ 5282c597-c178-50a2-a9e2-9ac36262852e
# Four bins (20–35, 35–55, 55–70, >70 keV) from counts above the four thresholds
bins_from(above) = [above[1] - above[2], above[2] - above[3], above[3] - above[4], above[4]]

# ╔═╡ 26b7ba11-b550-531e-a435-c117d0ddf3d3
bs_drm = BS.compute_mc_drm(BS._build_pcct_detector(bs_scanner), 140.0)

# ╔═╡ eefd3a0b-3fd0-529b-a9b2-b48dd07f032c
# BasisSimulator: counts per bin per photon arriving, for the 10° spectrum, with its fill factor
bs_bins = let
    e, w = spectra[10]
    grid = range(1.0, 140.0; length = size(bs_drm, 1))
    weights = [sum(w[abs.(e .- E) .<= step(grid) / 2]) for E in grid]
    0.95 * vec(sum(weights .* bs_drm; dims = 1)) ./ sum(weights)
end

# ╔═╡ ee77cb7f-65ec-5973-bee7-d09ea418ad7c
# Geant4, one native dexel without a grid (section 3), folded with the same spectrum
geant4_bins = let
    e, w = spectra[10]
    thresholds_at = [findfirst(==(t), thresholds) for t in air_readout.thresholds_keV]
    per_energy = [bins_from(c.total[thresholds_at]) for c in counts]
    weights = [sum(w[abs.(e .- E) .<= 0.5]) for E in energies]
    sum(weights .* per_energy) ./ sum(weights)
end

# ╔═╡ bba393d6-aba8-5c04-b19e-aabfbee54a17
# The same, with BasisSimulator's rule applied: at any energy where a photon makes more than one
# count (charge sharing), its counts are scaled down to one, and the fill factor 0.95 applied
geant4_bins_capped = let
    e, w = spectra[10]
    thresholds_at = [findfirst(==(t), thresholds) for t in air_readout.thresholds_keV]
    capped(b) = sum(b) > 1 ? b ./ sum(b) : b
    per_energy = [capped(bins_from(c.total[thresholds_at])) for c in counts]
    weights = [sum(w[abs.(e .- E) .<= 0.5]) for E in energies]
    0.95 * sum(weights .* per_energy) ./ sum(weights)
end

# ╔═╡ b6f087da-6174-536f-bcc8-cb3162a5672a
spectrum_10 = MC.Spectrum(spectra[10]...)

# ╔═╡ 386f7529-23ab-58b1-8cb8-b437f6202d92
air_scan_spectrum = sum(
    MC.simulate_sweep(
        [(detector_a, spectrum_10, chunk) for chunk in 1:48];
        n = 500_000,
        beam = MC.focal_spot(detector_a),
        readout = air_readout,
    )
)

# ╔═╡ ffbb7b94-2da1-50b4-a068-3db6b4722420
# The whole detector: total counts per bin over all dexels, per photon, per dexel
whole_bins = let c = air_scan_spectrum
    above = [sum(c.counts[:, :, t]) for t in 1:4] ./ c.n
    bins_from(above)
end

# ╔═╡ 5138ed44-8263-5dbf-9d6e-d78c085c4a85
let
    Mke.set_theme!(THEME)
    labels = ["20–35", "35–55", "55–70", "> 70"]
    fig = Mke.Figure(size = (1200, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Counts in each energy bin, per photon reaching the detector (140 kVp)",
        xlabel = "energy bin (keV)",
        ylabel = "counts per photon",
        xticks = (1:4, labels),
        limits = (nothing, (0, 0.42)),
    )
    series = (
        (
            bs_bins,
            PALETTE.reference,
            "BasisSimulator today (cdte_response_v4, capped at 1 count per photon, × 0.95)",
        ),
        (
            geant4_bins_capped,
            (PALETTE.reference, 0.45),
            "Geant4, one dexel, with the same cap and × 0.95",
        ),
        (geant4_bins, PALETTE.muted, "Geant4, one dexel, no grid, every count kept"),
        (whole_bins, PALETTE.photon, "Geant4, whole detector A with its grid"),
    )
    width = 0.2
    for (k, (values, color, label)) in enumerate(series)
        x = (1:4) .+ (k - 2.5) * width
        Mke.barplot!(ax, x, values; width, color, label)
        for (xi, v) in zip(x, values)
            note!(
                ax,
                xi,
                v,
                string(round(v; digits = 3));
                align = (:center, :bottom),
                offset = (0, 4),
                fontsize = 13,
                alpha = 0.0,
            )
        end
    end
    Mke.Legend(
        fig[2, 1],
        ax;
        orientation = :vertical,
        framevisible = false,
        nbanks = 2,
        tellheight = true,
    )
    Mke.rowgap!(fig.layout, 10)
    fig
end

# ╔═╡ f47124cb-a324-5c43-b885-8868f266f9e8
let
    total(v) = sum(v)
    Markdown.parse(
        """
        Counts per photon summed over the four bins: BasisSimulator **$(round(total(bs_bins); digits = 3))**;
        Geant4 one dexel with BasisSimulator's cap **$(round(total(geant4_bins_capped); digits = 3))**; Geant4
        one dexel keeping every count **$(round(total(geant4_bins); digits = 3))**; Geant4 whole detector
        **$(round(total(whole_bins); digits = 3))**.

        Two things separate BasisSimulator from the detector. **⚠ It caps each photon at one count**
        (`compute_mc_drm` rescales any energy whose counts per photon exceed 1), removing the double counts
        that charge sharing makes; the documented Alpha has no anti-coincidence logic, so the real detector
        keeps them. With the same cap, Geant4 lands on BasisSimulator's numbers. **⚠ Its fill factor (0.95)**
        is the share of the face that collects photons; with the grid stopping the photons over the gaps the
        whole detector's is about 0.72.
        """
    )
end

# ╔═╡ fe7057a7-e676-5a62-899c-da11603bb9ed
md"""
## 10. Pile-Up at Clinical Count Rates

At clinical tube currents a dexel receives tens of millions of photons per second. Its
electronics need a few nanoseconds to process each pulse; a second pulse inside that **dead
time** is either lost or merged with the first into one pulse of their summed energy (pile-up).
BasisSimulator models this with a pulse-train Monte Carlo (`simulate_pulse_train`,
seminonparalyzable, Hamid's PR #10): Poisson arrivals, dead-time gating, energy summation.

What a dexel's electronics see is not the incident spectrum, though: it is the **pulse
heights** the crystal produces, including the split charge clouds and the K x rays shared with
neighbours, more pulses than photons and many of low energy. Going beyond BasisSimulator's
input, the same pile-up code is fed here the Geant4 pulse-height spectrum of a dexel at its
true pulse rate.
"""

# ╔═╡ b42aaddf-0087-5766-91d5-7a5d1fd04261
# Photons per second reaching one native dexel in an air scan at 174 mA (no patient: the highest rate)
photon_rate = let
    e, w = spectra[10]
    sum(w) * 174.0 * prod(alpha.dexel)
end

# ╔═╡ 4d962a9a-3fd6-5fab-a132-c9c9907d4abc
# The pulse heights one dexel produces, from section 3's 3 × 3 run at each energy: every dexel that
# collected energy from a photon is one pulse (by symmetry, a neighbour's pulse from a photon on
# the central dexel is the central dexel's pulse from a photon on that neighbour)
pulses = let
    e, w = spectra[10]
    edges = 0.0:0.5:150.0
    heights = zeros(length(edges) - 1)
    pulses_per_photon = 0.0
    for (k, r) in enumerate(responses)
        weight = sum(w[abs.(e .- r.energy_keV) .<= 0.5])
        weight > 0 || continue
        for value in r.dexel_energy_keV
            value > 1.0 || continue
            b = min(searchsortedlast(edges, value), length(heights))
            heights[b] += weight / r.n
            pulses_per_photon += weight / r.n
        end
    end
    normalisation = sum(sum(w[abs.(e .- E) .<= 0.5]) for E in energies)
    (
        energies = collect(edges[1:(end - 1)] .+ 0.25),
        weights = heights,
        per_photon = pulses_per_photon / normalisation,
    )
end

# ╔═╡ 98710092-8d7f-56d4-964f-36441ac6613a
# Recorded against true counts per bin over many 1 ms pulse trains, for both inputs and two dead times
pileup = let
    e, w = spectra[10]
    thresholds = air_readout.thresholds_keV
    function run(weights, energies, rate, τ)
        true_counts = zeros(4)
        recorded = zeros(4)
        rng = BS.Random.Xoshiro(7)
        for _ in 1:200
            r = BS.simulate_pulse_train(
                weights,
                energies,
                rate,
                τ;
                thresholds_keV = thresholds,
                rng,
            )
            true_counts .+= r.true_bin_counts
            recorded .+= r.recorded_bin_counts
        end
        return (true_counts = true_counts, recorded = recorded)
    end
    Dict(
        (input, τ) => (
                input === :incident ? run(w, e, photon_rate, τ) :
                run(pulses.weights, pulses.energies, photon_rate * pulses.per_photon, τ)
            )
            for input in (:incident, :pulses), τ in (5.0, 6.0)
    )
end

# ╔═╡ 4c43a065-f125-5640-8df8-a2b16fdbdd4e
let
    Mke.set_theme!(THEME)
    labels = ["20–35", "35–55", "55–70", "> 70"]
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "What pile-up does at $(round(photon_rate / 1.0e6; digits = 1)) million photons per second per dexel",
        xlabel = "energy bin (keV)",
        ylabel = "recorded ÷ true counts",
        xticks = (1:4, labels),
        limits = (nothing, (0.5, nothing)),
    )
    Mke.hlines!(ax, 1; color = PALETTE.ink, linewidth = 1.5)
    series = (
        ((:incident, 5.0), PALETTE.reference, "incident spectrum, 5.0 ns (BasisSimulator)"),
        (
            (:incident, 6.0),
            (PALETTE.reference, 0.5),
            "incident spectrum, 6.0 ns (measured 5.3–7.0)",
        ),
        ((:pulses, 5.0), PALETTE.photon, "Geant4 pulse heights, 5.0 ns"),
        ((:pulses, 6.0), (PALETTE.photon, 0.5), "Geant4 pulse heights, 6.0 ns"),
    )
    width = 0.2
    for (k, (key, color, label)) in enumerate(series)
        r = pileup[key]
        Mke.barplot!(
            ax,
            (1:4) .+ (k - 2.5) * width,
            r.recorded ./ r.true_counts;
            width,
            color,
            label,
        )
    end
    Mke.axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 455fbb1d-4de4-588e-98b7-7d0159a29d63
let
    loss(key) = 1 - sum(pileup[key].recorded) / sum(pileup[key].true_counts)
    Markdown.parse(
        """
        Each dexel receives **$(round(photon_rate / 1.0e6; digits = 1)) million photons per second** in this air
        scan, and produces $(round(pulses.per_photon; digits = 2)) pulses per photon in the crystal. Counts lost
        overall at 5 ns: **$(round(100 * loss((:incident, 5.0)); digits = 1)) %** with the incident spectrum
        (BasisSimulator's input), **$(round(100 * loss((:pulses, 5.0)); digits = 1)) %** with the Geant4
        pulse heights.
        """
    )
end

# ╔═╡ 92576991-a76a-5062-b4fc-e5d8604dd672
md"""
## Saving the Tables

The tables BasisSimulator would load, written in the layout of its current `cdte_response_v4`
(the same keys, dimensions and meaning; `BasisSimulator.load_mc_response` reads them), with the
provenance alongside. Two tables:

- **`naeotom_alpha_cdte_response_geant4.jls`**: one native dexel without a grid, 1–140 keV, the
  nine thresholds, 30 µm cloud and 1.5 keV noise: a drop-in replacement for `cdte_response_v4`,
  computed from first principles.
- **`naeotom_alpha_cdte_response_grid_geant4.jls`**: the six positions in a grid cell, with the
  grid, plus its open fraction: what BasisSimulator would need to model the grid itself.
"""

# ╔═╡ fd10aa2e-f7ac-5b15-b095-6f2ead637241
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ d090515d-ebde-502a-88ff-a1801857dd84
provenance = Dict(
    "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 03_alpha_detector.jl",
    "detector" => "Siemens NAEOTOM Alpha, CdTe 1.6 mm, native dexel 0.275 × 0.322 mm",
    "physics" => string(MC.PhysicsSettings()),
    "photons_per_energy" => 100_000,
    "illumination" => "uniform over the central dexel's electrode, normal incidence",
)

# ╔═╡ 8ba64dd5-6b2c-5c06-8a3b-0e94a0f5941d
alpha_table = MC.response_table(
    responses,
    thresholds;
    electronic_noise_keV = current_table["sigma_elec_keV"],
    metadata = merge(provenance, Dict("grid" => "none")),
)

# ╔═╡ 043f5250-d081-56b7-9379-acbcfea44d1a
grid_table = Dict(
    "positions" => [p.label for p in positions],
    "targets" => [p.target for p in positions],
    "tables" => [
        MC.response_table(
                grid_responses[k],
                thresholds;
                electronic_noise_keV = current_table["sigma_elec_keV"],
            )
            for k in eachindex(positions)
    ],
    "open_fraction" => prod(alpha.group .* alpha.dexel) / prod(cell_pitch),
    "grid" => "2-D, 4 × 6 dexel cells, walls over the gaps, $(alpha.wall_height) mm tall, tungsten (assumed)",
    "provenance" => provenance,
)

# ╔═╡ ba9df479-0755-53ad-83b1-635caea0f8ec
saved = let
    mkpath(table_dir)
    paths = (
        joinpath(table_dir, "naeotom_alpha_cdte_response_geant4.jls"),
        joinpath(table_dir, "naeotom_alpha_cdte_response_grid_geant4.jls"),
    )
    Serialization.serialize(paths[1], alpha_table)
    Serialization.serialize(paths[2], grid_table)
    basename.(paths)
end

# ╔═╡ b5b66b9b-9dd8-58c6-967a-3571f17ffd31
# BasisSimulator reads the new table with the same loader it uses for cdte_response_v4
reloaded = (saved; BS.load_mc_response(joinpath(table_dir, "naeotom_alpha_cdte_response_geant4.jls")))

# ╔═╡ 288a1e92-09f9-55d2-b6e3-06b2ff4adc96
let
    largest = maximum(abs, reloaded.R_total .- current_table["R_total"])
    Markdown.parse(
        """
        BasisSimulator loads the Geant4 table: $(length(reloaded.energies_keV)) energies × $(length(reloaded.thresholds_keV))
        thresholds, and its counts differ from `cdte_response_v4`'s by at most **$(round(largest; digits = 3))** counts per
        photon.
        """
    )
end

# ╔═╡ 6cd7d71c-20c3-568f-9386-3ee1047ed406
md"""
## Where This Leaves the Alpha

- **One dexel**, simulated in Geant4 from first principles, reproduces BasisSimulator's current
  MCNP table (`cdte_response_v4`) to about 0.01 counts per photon at every threshold.
- **The anti-scatter grid**, absent from BasisSimulator, changes the share of the detector that
  sees photons from 0.95 to about 0.72, without making dexels unequal in an air scan.
- **The whole detector**, built one to one in Geant4, behaves as its six cell positions: a
  full response table needs six dexels per row, not 792 576.
- **The count rates** and pulse heights feed BasisSimulator's own pile-up model, with the
  crystal's real pulse-height spectrum in place of the incident one.

What would replace BasisSimulator's tables: the six positions' responses on its energy grid,
at the cone angle of each row, binned 2 × 2, with the grid's open fraction, from this notebook.
"""

# ╔═╡ Cell order:
# ╟─6e30fdb2-879c-52f4-99ab-666bbb68cd4f
# ╟─df200390-005d-5328-870a-9a821744961d
# ╟─e71ceeb9-0d88-5608-8588-ee48a312de90
# ╟─8326156c-5e6a-5f16-89ab-5e8b268b2e0a
# ╟─f7ad4569-b274-5714-83ab-bc7ad0497b53
# ╟─943d5189-ca8c-5e1b-860d-dad9ed192843
# ╠═ed5d0968-55b8-54bb-be08-b8b3e5231445
# ╠═73911ef9-afec-5057-9406-29f4217f5d15
# ╠═aa57fb94-4f51-5853-8cf5-c0598ee6b717
# ╠═9215c3a9-8e71-5f66-b12a-2779e5350eda
# ╠═1ff7c097-d985-5412-9686-8e062018d0b2
# ╠═bc2a467c-d990-556b-b955-dd91539122ab
# ╠═fdaa0020-4804-5c58-8453-668f6c70ebf3
# ╠═0df22b09-66d5-59e9-9f68-79a2b1886e38
# ╠═40986dbf-a0d8-51ea-aa7b-75b21dcc7eaa
# ╠═a944bba9-6761-5648-b028-0d90a7b56e29
# ╠═c5fa8b52-766b-5c4a-9794-a056c8e00dc2
# ╠═eb7fb66a-4e78-5a01-ada0-e323524fde34
# ╟─a75d1e50-0d30-51f0-9004-d0a001673992
# ╠═d1967848-6509-5e0e-8f13-db8ea4993799
# ╠═c03a39db-4010-59b6-8c41-be48372ae822
# ╠═c9a17736-48ea-5cf4-acdd-39000bd8e7ae
# ╟─497d96da-68e6-5194-ac65-4f174288ace0
# ╟─05b29a50-b080-5d3b-b8f3-a51d0c6ee344
# ╠═1ac6bf6f-9249-590c-8f75-7fd754caab4b
# ╠═03f7a698-8e12-5541-ac92-3ebf124799f5
# ╠═c4a5a27a-e7b8-54f7-aa8b-7b3cf1059bb2
# ╠═110f9a44-e004-5d43-a2b2-391a4bd18b99
# ╟─fb9ed65f-1ea8-557f-9703-76b935d46a75
# ╟─a1c0bfc5-a8eb-51f4-8a11-429b5604368d
# ╠═e9d84ab8-bfdf-59c7-813b-78b03c2c4a7a
# ╠═efb0402d-648c-5ab8-a06d-f031e9550e90
# ╟─4f6dd6cb-cb42-5da9-9692-b257389cacf7
# ╠═b334ec12-f83a-59bc-8936-c7e16badc21c
# ╟─6a162bd9-0b36-5cf6-b16e-6b6d095b5ee8
# ╟─65f1babc-4849-5e3a-8404-738562c88837
# ╠═0eff78e6-df72-522a-b5db-d70421a6cdec
# ╠═4e507ffb-04ca-5f56-a577-e2715de205a4
# ╠═124978ab-3b89-5c61-996d-64d7f222d405
# ╟─28a04c08-f2be-5b0b-8487-d1953b4ffca4
# ╟─c0ec94cf-51a6-5967-8b5a-2a582ed1ebf9
# ╟─db9638a4-4721-5644-b84e-bec6fa1a80e3
# ╠═9b183cd4-f849-5eb9-bbc9-f62a327ba741
# ╠═26699732-4c76-5ecd-ae55-ed5aff70d36f
# ╠═8bc9b5e0-ff52-51d2-b662-82db9b772e06
# ╠═43251201-987e-59a2-ba3e-9e1785549002
# ╠═c5fb6ae2-0681-5f93-a2d7-05dffa64530e
# ╠═d952031c-89aa-5eb9-9697-ef27eff6d6b0
# ╠═9f9f79cf-284d-5ef9-98c7-5cf015924059
# ╟─e5b42b5a-564f-5ba8-b8a1-6a302df8e3ad
# ╟─d09b836e-5751-5269-b5c0-c3ad1f8a94ef
# ╠═888de7f9-befc-515e-a23a-f7db676ac39e
# ╠═4c53f560-8613-5038-b0d9-25180b502e73
# ╟─87a3f02f-8d85-57ec-ab9b-78b53a80b7c0
# ╠═71102eb4-ce19-584d-809f-e69e4f3561bd
# ╟─650a008b-622d-5948-8485-96d50b16b5b9
# ╟─46b15a3d-c453-593c-bb39-2b0659dc1631
# ╟─c6682374-bbef-5638-9592-82c5c8e5aff1
# ╠═f167cd8c-aea8-5090-ab71-7a82f303c13a
# ╠═a7975678-b7d2-52c0-a314-7c1bc4549cff
# ╟─49a1c7cc-171e-5724-a834-f334c5e9c2ea
# ╟─25fb922f-615e-5b76-9048-7acf73bb0f3a
# ╠═39f04615-41a5-5f60-894d-f8e636e68f8c
# ╟─52e68256-0954-5993-b7ea-a0ef64dbde97
# ╟─9133ec80-9f4b-5fbd-8d36-626ca9c12719
# ╠═7081fb53-7350-5ef5-82fc-c3866ca8dc5a
# ╠═bfee1bcb-7273-5f67-b684-42e8be79191d
# ╟─c0f06895-b97e-5bc3-81dd-bdd5b1a8a0c3
# ╟─39c48e80-8b7e-5910-9909-463d8f715b0f
# ╟─2ae8390f-0dcc-5f4b-85ca-ca5cb562bb98
# ╠═1e681426-7ee2-5fe5-a150-92b684d4aed3
# ╠═c96473dd-3ca7-54aa-87ce-af04055fc5d1
# ╟─2ffccd1c-e5ea-5da2-b746-7ee1f88daeb5
# ╟─dd7dbc05-6496-563a-ab62-46a547a22324
# ╟─c89fca63-6462-5f25-b16e-d491689a48a3
# ╠═5d8d5cd1-8ff7-5356-87bb-24b57c3281f1
# ╠═1d5d02ab-485b-567f-ac5d-aaea105040a4
# ╠═15bdf1fd-777f-50c0-ad69-59c912741e2e
# ╟─877c4989-7b59-5741-a50a-433cf4a4a14b
# ╟─848bfaff-246d-4084-989c-c17ef4008e41
# ╠═648db0cd-57fc-46a7-894e-06ab540140a2
# ╟─f68a9548-e730-4402-af6d-ed2f2a2dc3a9
# ╟─0e2ac434-ad27-4763-bbb7-f1a231b06087
# ╟─0534c5aa-7947-53bd-b463-865d06524b9c
# ╠═5282c597-c178-50a2-a9e2-9ac36262852e
# ╠═26b7ba11-b550-531e-a435-c117d0ddf3d3
# ╠═eefd3a0b-3fd0-529b-a9b2-b48dd07f032c
# ╠═ee77cb7f-65ec-5973-bee7-d09ea418ad7c
# ╠═bba393d6-aba8-5c04-b19e-aabfbee54a17
# ╠═b6f087da-6174-536f-bcc8-cb3162a5672a
# ╠═386f7529-23ab-58b1-8cb8-b437f6202d92
# ╠═ffbb7b94-2da1-50b4-a068-3db6b4722420
# ╟─5138ed44-8263-5dbf-9d6e-d78c085c4a85
# ╟─f47124cb-a324-5c43-b885-8868f266f9e8
# ╟─fe7057a7-e676-5a62-899c-da11603bb9ed
# ╠═b42aaddf-0087-5766-91d5-7a5d1fd04261
# ╠═4d962a9a-3fd6-5fab-a132-c9c9907d4abc
# ╠═98710092-8d7f-56d4-964f-36441ac6613a
# ╟─4c43a065-f125-5640-8df8-a2b16fdbdd4e
# ╟─455fbb1d-4de4-588e-98b7-7d0159a29d63
# ╟─92576991-a76a-5062-b4fc-e5d8604dd672
# ╠═fd10aa2e-f7ac-5b15-b095-6f2ead637241
# ╠═d090515d-ebde-502a-88ff-a1801857dd84
# ╠═8ba64dd5-6b2c-5c06-8a3b-0e94a0f5941d
# ╠═043f5250-d081-56b7-9379-acbcfea44d1a
# ╠═ba9df479-0755-53ad-83b1-635caea0f8ec
# ╠═b5b66b9b-9dd8-58c6-967a-3571f17ffd31
# ╟─288a1e92-09f9-55d2-b6e3-06b2ff4adc96
# ╟─6cd7d71c-20c3-568f-9386-3ee1047ed406
