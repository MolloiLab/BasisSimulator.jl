### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end


# ╔═╡ 982eeab8-0545-51f1-b14c-f4984bc5ce21
md"""
# The GE Gemstone Detector in Geant4: Revolution Apex

The GE Revolution Apex is an energy-integrating CT scanner with a 256-row, 160 mm wide detector
(GE's "Gemstone Clarity"). Its scintillator is **Gemstone**, a cerium-activated garnet of
terbium, lutetium and aluminium, (Tb,Lu)₃Al₅O₁₂:Ce, made as a transparent polycrystalline
ceramic and cut into **dexels** (detector elements), with a reflector between them and a
backlit photodiode under every dexel. In front of it stands GE's "3D collimator", a
two-dimensional anti-scatter grid.

BasisSimulator describes how efficiently this detector turns x rays into signal with one table,
η(E) from 1 to 140 keV (`GEMSTONE_MC_EFFICIENCY_LUT`), computed with MCNP by Hamidreza
Khodajou-Chokami; the inputs to that simulation were never committed. This notebook builds the
detector in Geant4, one piece at a time, and

1. **checks every piece against the documentation**: the FDA clearances, GE's patents and
   brochures, and the literature;
2. **reproduces the current table**, which shows how it was made;
3. **builds a better table** for a garnet that can exist (BasisSimulator's density cannot), with
   its noise (the Swank factor), and saves it for BasisSimulator.
"""

# ╔═╡ 7d76168f-6aaf-5cac-ad93-06b3d605dce4
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a Gemstone dexel does not count at all.
Every x ray it absorbs leaves energy in the garnet, which glows; the reflector keeps the light
inside the dexel, and the photodiode under it adds up the light of every photon over the reading.
The signal is therefore proportional to the **energy** left in the dexel, and the efficiency
that matters is the energy-absorption efficiency: the mean energy a dexel collects per photon,
divided by the photon's energy.
"""

# ╔═╡ f53ca02e-1359-50fb-ab81-6e8c8c5249a4
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
            body = "an x ray of energy E\nlands on a dexel",
            where = "the beam",
            color = PALETTE.photon,
        ),
        (
            title = "2  Garnet",
            body = "absorbed in Gemstone;\nits Tb or Lu x ray\nmay land next door",
            where = "Geant4",
            color = PALETTE.xray,
        ),
        (
            title = "3  Light",
            body = "the energy left\nbecomes light, kept\nin by the reflector",
            where = "Gemstone glows for 30 ns",
            color = PALETTE.electron,
        ),
        (
            title = "4  Photodiode",
            body = "adds up the light\nunder the dexel:\nsignal ∝ energy",
            where = "one per dexel",
            color = PALETTE.slab_edge,
        ),
        (
            title = "5  Efficiency",
            body = "energy collected\n÷ photon energy\n= η(E)",
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
            Mke.scatter!(
                ax,
                [x + w + gap / 2],
                [15.5];
                marker = :rtriangle,
                markersize = 14,
                color = PALETTE.muted,
            )
        end
    end
    fig
end

# ╔═╡ 0d89ac30-a1cd-555c-a60a-934112484f93
md"""
## The Revolution Apex, as Documented

Every hardware number this notebook uses, checked against the FDA 510(k) summaries, GE's patents
and brochures, and the literature. Where BasisSimulator and a source disagree the row is marked
**⚠ discrepancy**; where no source gives a number the row says so, and the notebook states its
assumption. GE publishes very little about the detector itself.

| quantity | BasisSimulator | documented | status | source |
|---|---|---|---|---|
| scintillator | Gemstone, (Tb,Lu)₃Al₅O₁₂:Ce | a Ce-activated rare-earth garnet; "(Tb,Lu)" named in the literature; GE's patents claim (Tb₁₋ₓLuₓ)₃Al₅O₁₂:Ce with x from 0 to about 0.3 | ✓ (family) | LLNL-TR-587512 p. 7; US 6,793,848; You et al. 2023 |
| composition | Tb 0.287, Lu 0.316, Al 0.163, O 0.231, Ce 0.002 by mass (an even Tb/Lu split) | not published; GE's patent example Lu₀.₈Tb₂.₂Al₅O₁₂ is Tb-rich (Tb 0.428, Lu 0.171) | not confirmed; bracketed in section 4 | US 6,793,848 |
| citation for the composition | You et al. 2023 (PMC10179960) | that paper names Gemstone only, with no composition | **⚠ discrepancy** (miscitation) | You et al., Materials 2023 |
| density | 7.0 g/cm³ | any (Tb,Lu)₃Al₅O₁₂ lies between 6.07 (Tb₃Al₅O₁₂) and 6.72 (Lu₃Al₅O₁₂) | **⚠ discrepancy** (impossible); 6.40 used here | Konoshima; Zhu et al. 2022 |
| crystal depth | 3.0 mm | not published; 3.0 mm is XCIST's default detector depth (for its "Lumex", a different material); GE's reflector patent gives cells about 1.5–3 mm tall | not confirmed | XCIST `Scanner_Default.cfg`; US 8,385,499 |
| the `:lumex` name | `:lumex` = Gemstone | XCIST's "Lumex" is (Y,Gd)₂O₃:Eu, GE's older HiLight ceramic | **⚠ discrepancy** (naming) | XCIST `material/Lumex` |
| rows × row width | 256 × 0.625 mm | 256 × 0.625 mm, 160 mm | ✓ | K133705; K213715 |
| columns, column width, source–isocentre, source–detector | 834; 0.6 mm; 625.6, 1100 mm | not published | not confirmed | – |
| reflector between dexels | not modelled (fill factor 0.9 × 0.9) | not published for the product; GE's patent lists TiO₂ among reflector materials | assumed: 0.1 mm of TiO₂-loaded epoxy, which gives the same fill factor | US 8,385,499 |
| anti-scatter grid | not modelled | "3D collimator (post patient)", two-dimensional; product dimensions not published; GE's patent: walls 0.10–0.20 mm, about 8 mm tall, one channel per dexel | built here: tungsten, 0.15 mm walls, 8 mm tall | K133705; US 8,385,499 |
| photodiode | – | "low capacitance backlit photodiode" | – | K213715 |
| beam quality (Revolution CT, large bowtie) | – | half-value layer 7.08, 9.20, 10.0 mm Al at 80, 120, 140 kV, isocentre | a check in section 8 | Kawahara et al. 2023 |

Sources: FDA [K133705](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf),
[K213715](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf);
GE patents [US 6,793,848](https://patents.google.com/patent/US6793848B2/en),
[US 6,630,077](https://patents.google.com/patent/US6630077B2/en),
[US 8,385,499](https://patents.google.com/patent/US8385499B2/en);
Lanier, [LLNL-TR-587512](https://www.osti.gov/servlets/purl/1053658);
You et al., [Materials 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10179960/);
Zhu et al., [J. Adv. Ceram. 2022](https://d-nb.info/1280575069/34);
Konoshima Chemical, [TAG ceramics](https://www.konoshima.co.jp/eng/ceramics/resdev2.html);
XCIST, [github.com/xcist/main](https://github.com/xcist/main);
Kawahara et al., [Pol. J. Radiol. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317008/);
Siemens, [Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ b5d65082-f8d7-5b19-a30a-117dcd788835
md"""
## Notebook Setup
"""

# ╔═╡ c7b8ce23-df15-5445-8f9c-1d533b22b055
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ b9afe761-a41a-56bd-89b3-aa4d1f850542
using Markdown: @md_str, Markdown

# ╔═╡ 1114b922-2909-5382-ad68-7870b9f17a49
import PlutoUI

# ╔═╡ 3c7da9cf-d44f-5117-a6cd-ab70462594ba
import BasisMonteCarlo as MC

# ╔═╡ d7980232-34d1-599f-89a1-18c144d7468b
import CairoMakie as Mke

# ╔═╡ 2e1263bc-6a15-5dad-9da8-0c5ea9d9a1ec
import BasisSimulator as BS

# ╔═╡ 6f0a1390-4e16-5007-aff0-205efb24710a
import Serialization

# ╔═╡ 6650624a-9201-5026-9af0-2852cb32ff32
import DelimitedFiles

# ╔═╡ 065c2cd0-3724-5ae0-a368-5c8eb5540a1a
using Statistics: mean

# ╔═╡ 0f1f7711-29fa-5149-b284-b47dcdc295c0
PlutoUI.TableOfContents()

# ╔═╡ e3e9c212-4583-5b56-8c63-55a23aceccc6
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 3b5f881c-54d6-5378-ad04-eefcacc3c8ea
# One colour per kind of particle, used by every figure
PALETTE = (
    photon = Mke.RGBf(0.13, 0.36, 0.66),  # the incident photon
    xray = Mke.RGBf(0.9, 0.45, 0.05),  # characteristic x rays
    electron = Mke.RGBf(0.7, 0.12, 0.17),  # electrons, and the light they make
    kept = Mke.RGBf(0.27, 0.55, 0.34),  # the signal
    slab = Mke.RGBAf(0.45, 0.52, 0.62, 0.3),
    slab_edge = Mke.RGBf(0.3, 0.36, 0.44),
    reflector = Mke.RGBf(0.85, 0.83, 0.7),
    ceramic = Mke.RGBf(0.9, 0.92, 0.95),
    ceramic_hit = Mke.RGBf(0.78, 0.85, 0.95),
    tungsten = Mke.RGBf(0.25, 0.25, 0.3),
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator's current table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ 425bf6f0-bb29-56ec-9883-45afcf680eda
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

# ╔═╡ 4551bbf4-0c6f-558d-8dd3-19061afbeee1
"""
An in-figure annotation on a translucent rounded background, so it never fights the data.
"""
function note!(
        ax,
        x,
        y,
        text;
        align = (:left, :center),
        color = PALETTE.ink,
        fontsize = 16,
        offset = (0, 0),
        alpha = 0.85,
    )
    return Mke.textlabel!(
        ax,
        Mke.Point2f(x, y);
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

# ╔═╡ d852372b-0f59-5689-a6f2-a96f8756249b
"""Probability per keV per photon of `values` falling in each bin of width `bin_keV` up to `top_keV`."""
function histogram(values, top_keV; bin_keV = 0.5)
    edges = 0:bin_keV:top_keV
    counts = zeros(length(edges) - 1)
    for v in values
        v > 0 || continue
        k = clamp(floor(Int, v / bin_keV) + 1, 1, length(counts))
        counts[k] += 1
    end
    return (
        centres = edges[1:(end - 1)] .+ bin_keV / 2,
        density = counts ./ (length(values) * bin_keV),
    )
end

# ╔═╡ 9ed4e154-0c35-5252-b2dc-388f22a1c47f
md"""
## 1. The Scintillator

Gemstone is a garnet, (Tb,Lu)₃Al₅O₁₂ activated with cerium: terbium and lutetium share the large
sites of the crystal in a proportion GE does not publish. BasisSimulator's composition splits
them evenly (by atoms) at **7.0 g/cm³**, but no such garnet can be that dense: pure
Tb₃Al₅O₁₂ is 6.07 g/cm³ and pure Lu₃Al₅O₁₂ 6.72 g/cm³, and a mixture lies between (⚠
*discrepancy* above). The even split, interpolated between the two, is **6.40 g/cm³**: this
notebook builds the detector from BasisSimulator's composition at that density, keeps 7.0 to
reproduce BasisSimulator's table, and brackets the unknown composition with the Tb-rich garnet of
GE's patent (6.24 g/cm³).
"""

# ╔═╡ 0565c39d-9720-5863-a0b6-c3e104a475d1
# BasisSimulator's composition (an even Tb/Lu split) at a density such a garnet can have
gemstone = MC.DetectorMaterial(
    "Gemstone 6.40";
    density = 6.4,
    mass_fractions = Dict(
        "Tb" => 0.287,
        "Lu" => 0.316,
        "Al" => 0.163,
        "O" => 0.231,
        "Ce" => 0.002,
    ),
)

# ╔═╡ b5c946ee-f16f-5cca-82cf-b5040fe481b8
# BasisSimulator's Gemstone: the same composition at 7.0 g/cm³
gemstone_basis = MC.DetectorMaterial(:Gemstone)

# ╔═╡ 8fba75c0-6a9f-5bbe-9a10-9d52e2296292
# The Tb-rich garnet of GE's patent, Lu₀.₈Tb₂.₂Al₅O₁₂ (6.24 g/cm³ by the same interpolation)
gemstone_tb_rich = MC.DetectorMaterial(
    "Gemstone Tb-rich";
    density = 6.24,
    mass_fractions = Dict("Tb" => 0.428, "Lu" => 0.171, "Al" => 0.165, "O" => 0.235),
)

# ╔═╡ b349f0be-6494-5c81-80ea-82a481df3b90
# Terbium's and lutetium's K-shells: their edges and the energies of the x rays they emit (Kα)
terbium, lutetium = sort(MC.k_edges(gemstone); by = e -> e.edge_keV)

# ╔═╡ 3e134eba-e90e-5fef-845b-b81b61397dca
md"""
Gemstone has **two K-edges** in the CT spectrum: terbium's at 52.0 keV and lutetium's at
63.3 keV. Above each, absorption jumps, and every absorption in that element's K shell leaves the
atom to emit, most of the time, its K x ray: **44.5 keV for terbium, 54.1 keV for lutetium**.
Each is just below its own edge, so weakly absorbed by its own element, and can travel a
millimetre or more before stopping, about the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ f88f3daf-cbd3-55ec-9361-63ae45a21b33
# The ways a photon can interact, as the figures name them
processes = (
    (kind = :total, label = "all processes together (total)", color = PALETTE.ink),
    (
        kind = :photoelectric,
        label = "photon absorbed (photoelectric)",
        color = PALETTE.electron,
    ),
    (
        kind = :compton,
        label = "bounces off, loses energy (Compton)",
        color = PALETTE.photon,
    ),
    (
        kind = :rayleigh,
        label = "bounces off, keeps energy (Rayleigh)",
        color = PALETTE.muted,
    ),
)

# ╔═╡ d6256293-43fa-5626-bba6-6b1c01af6dda
mu_energies = 10.0:0.25:150.0

# ╔═╡ 2bcabe1a-a3fd-5fa7-8e2a-2a88d9126859
# Geant4's attenuation of Gemstone, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(gemstone, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 7b39f748-54d2-5a67-977a-b81bf47b6b27
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(gemstone, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 79bf52e2-7776-5cc1-8d0b-4e264aa8530e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How strongly Gemstone stops x rays (6.40 g/cm³)",
        xlabel = "photon energy (keV)",
        ylabel = "chance of interacting per mm, μ (1/mm)",
        yscale = log10,
        xticks = 10:20:150,
        limits = ((10, 150), (4.0e-3, 400)),
    )
    for p in processes
        Mke.lines!(
            ax,
            mu_energies,
            mu_geant4[p.kind];
            color = p.color,
            label = p.label,
            linewidth = p.kind === :total ? 3.5 : 2.5,
        )
        Mke.lines!(
            ax,
            mu_energies,
            mu_xcom[p.kind];
            color = (p.color, 0.7),
            linestyle = :dash,
            linewidth = 1.6,
        )
    end
    for (k, edge) in enumerate((terbium, lutetium))
        Mke.vlines!(
            ax,
            edge.edge_keV;
            color = (PALETTE.xray, 0.8),
            linestyle = :dot,
            linewidth = 2,
        )
        note!(
            ax,
            edge.edge_keV,
            k == 1 ? 0.03 : 0.012,
            "$(edge.element) K-edge, $(round(edge.edge_keV; digits = 1)) keV";
            offset = (8, 0),
            color = PALETTE.xray,
        )
    end
    note!(
        ax,
        148,
        250,
        "solid: Geant4 (used)    dashed: NIST XCOM";
        align = (:right, :top),
        fontsize = 15,
        color = PALETTE.muted,
    )
    Mke.Legend(
        fig[2, 1],
        ax;
        orientation = :horizontal,
        nbanks = 2,
        tellheight = true,
    )
    fig
end

# ╔═╡ d0f1d5ee-5eb0-5fa2-bcd6-74a957ed0358
let
    total = maximum(abs.(mu_geant4[:total] ./ mu_xcom[:total] .- 1))
    rayleigh = maximum(abs.(mu_geant4[:rayleigh] ./ mu_xcom[:rayleigh] .- 1))
    Markdown.parse(
        """
        Geant4 and XCOM agree on the total attenuation to within **$(round(100total; digits = 1)) %** from 10 to
        150 keV. They differ most on Rayleigh scattering (up to $(round(100rayleigh; digits = 0)) %), which Geant4
        computes with anomalous scattering near the K-edge; Rayleigh is a small part of the total.
        """
    )
end

# ╔═╡ ea752deb-d5f8-530f-a2bc-d8f70ac7f9d6
md"""
### A First Check: Siemens' Curve for LuTAG

No absorption curve for Gemstone is published, but Siemens, comparing its own ceramic with its
rivals', plots the fraction of x rays that **1.4 mm of LuTAG** (lutetium terbium aluminium
garnet, Gemstone's family) absorbs, against energy (*Detectors & Crystals*, p. 19). Its steps at
the K-edges go *up*: it is the probability that a photon interacts at all, `1 − exp(−μd)`, so it
depends only on composition, density and depth. Siemens gives neither LuTAG's composition nor
its density, so this is a loose check; but it can tell a garnet that exists from one that cannot.
The values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ ad4b77f0-e633-5a1f-8474-144d368eb858
# Siemens' LuTAG curve, read from the vector figure (reference_data/siemens_lutag_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_lutag_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ 7fea9a31-343e-5084-b697-c614d31d86cc
# The depth of Siemens' figure
siemens_depth_mm = 1.4

# ╔═╡ c6bf2e2b-147e-5c10-a14e-53a1f4421e56
# The probability a photon interacts in 1.4 mm of each candidate garnet
interaction_curves = [
    (
            label = label,
            color = color,
            curve = [1 - exp(-MC.geant4_attenuation(material, E) * siemens_depth_mm) for E in mu_energies],
        )
        for (label, color, material) in (
            (
                "BasisSimulator's: even split, 7.0 g/cm³ (impossible)",
                PALETTE.reference,
                gemstone_basis,
            ),
            ("even split, 6.40 g/cm³ (used here)", PALETTE.photon, gemstone),
            ("Tb-rich (GE's patent), 6.24 g/cm³", PALETTE.kept, gemstone_tb_rich),
        )
]

# ╔═╡ 7e771ff9-6f1a-5c5b-8469-30e1a7ef0a6e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The fraction of photons 1.4 mm of garnet absorbs",
        xlabel = "photon energy (keV)",
        ylabel = "fraction of photons absorbed",
        xticks = 20:20:150,
        limits = ((15, 152), (0.3, 1.04)),
    )
    for c in interaction_curves
        Mke.lines!(
            ax,
            mu_energies,
            c.curve;
            color = c.color,
            linewidth = 3,
            label = c.label,
        )
    end
    Mke.rangebars!(
        ax,
        siemens_curve.energy_keV,
        siemens_curve.absorption .- 0.025,
        siemens_curve.absorption .+ 0.025;
        color = (PALETTE.ink, 0.5),
        whiskerwidth = 8,
    )
    Mke.scatter!(
        ax,
        siemens_curve.energy_keV,
        siemens_curve.absorption;
        color = PALETTE.ink,
        markersize = 13,
        label = "Siemens' LuTAG figure (± 0.025)",
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ ba7bd8e3-3fe8-51fb-a149-651b7415cecf
let
    high = siemens_curve.energy_keV .>= 80
    rms(c) = sqrt(
        sum(
            abs2,
            [MC.interpolate_linear(collect(mu_energies), c.curve, E) for E in siemens_curve.energy_keV[high]] .-
                siemens_curve.absorption[high],
        ) / count(high),
    )
    rows = ["| $(c.label) | $(round(rms(c); digits = 3)) |" for c in interaction_curves]
    Markdown.parse(
        """
        | garnet, 1.4 mm | rms difference from Siemens' LuTAG figure, 80–150 keV |
        |---|---|
        $(join(rows, "\n"))

        Above 80 keV, where density matters most, every candidate absorbs more than Siemens' LuTAG does, and the
        lighter the garnet, the closer it comes: BasisSimulator's 7.0 g/cm³ is furthest. LuTAG's own composition and
        depth are not published, so this does not pin Gemstone down, but it points the same way as the density limits:
        a lutetium-terbium garnet absorbs less than BasisSimulator's Gemstone.
        """
    )
end

# ╔═╡ 76d9fe06-65d2-50ec-a57f-9d13c0424438
md"""
## 2. One Dexel and Its Neighbours

A dexel is one block of garnet with reflector on its four sides. Its pitch at the detector face
is the Apex's dexel at the isocentre as BasisSimulator has it, 0.6 mm across the fan by 0.625 mm
along z, magnified by the source-to-detector over the source-to-isocentre distance,
1100 / 625.6. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy is assumed
(TiO₂ is among the reflector materials of GE's patent). The crystal is BasisSimulator's 3.0 mm
deep, inherited from XCIST, and within the 1.5–3 mm cells of GE's reflector patent.
"""

# ╔═╡ 59b35b57-f457-5c5a-89fd-f215ac91ec5f
# The Revolution Apex's geometry, as BasisSimulator has it
apex = (
    source_to_isocenter_mm = 625.6,
    source_to_detector_mm = 1100.0,
    columns = 834,
    rows = 256,
    column_at_isocenter_mm = 0.6,
    row_at_isocenter_mm = 0.625,
)

# ╔═╡ 87c01464-7920-5151-aaf1-668655c36c98
# The crystal depth: BasisSimulator's (no source gives the Apex's)
depth_mm = 3.0

# ╔═╡ 66875788-7213-5a45-93df-f02a465feab5
# The dexel pitch at the detector face (across the fan, along z), in mm
pitch_mm = (apex.column_at_isocenter_mm, apex.row_at_isocenter_mm) .*
    (apex.source_to_detector_mm / apex.source_to_isocenter_mm)

# ╔═╡ 85332bcb-a653-5577-986e-55557180a527
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ 43a2476d-f0b3-5c4e-ad60-a79093af88bc
# The block of garnet each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ b937f150-98dd-52ad-a1a9-39606e1cc17d
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 69f88251-23f2-5b2f-8e30-f8037e72aabd
let
    fill = crystal_mm ./ pitch_mm
    Markdown.parse(
        """
        The dexel pitch at the face is **$(round(pitch_mm[1]; digits = 3)) × $(round(pitch_mm[2]; digits = 3)) mm**, and
        the garnet block inside it $(round(crystal_mm[1]; digits = 3)) × $(round(crystal_mm[2]; digits = 3)) mm: the crystal
        covers $(round(fill[1]; digits = 3)) × $(round(fill[2]; digits = 3)) of the face, BasisSimulator's fill factor of 0.9 × 0.9.
        The assumed reflector and BasisSimulator's fill factor are the same detector.
        """
    )
end

# ╔═╡ 3cd39c08-6d7d-5852-911d-70b71149b9ca
# The Geant4 detector: a slab of garnet `depth` deep, cut into 13 × 13 dexels by reflector
# septa (ample for the x rays that leave a dexel), optionally under an anti-scatter grid
function dexel_array(depth; material = gemstone, grid = nothing)
    septa = MC.Septa(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        width_x_mm = septum_mm,
        width_y_mm = septum_mm,
        material = reflector,
        cells = 13,
    )
    return MC.SlabDetector(
        material,
        depth;
        width_mm = 13 * maximum(pitch_mm) + 1,
        septa,
        collimator = grid,
    )
end

# ╔═╡ 13a363b3-f112-59a0-9192-be6a9bef1026
# Photons landing uniformly on one dexel's garnet, and on its whole cell (garnet and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ e340add3-b5f6-5242-93ea-81ef5ba335a2
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 0df4a950-2c66-5f98-827e-d647b3db7d5c
# Each dexel collects the energy left in its own block of garnet (the reflector keeps the light
# in); the dexel the photons land on and its eight neighbours are recorded
readout = MC.DexelEnergyReadout(
    MC.DexelReadout(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        neighborhood = 3,
        cloud_sigma_um = 0.01,
    ),
)

# ╔═╡ 3a51adf8-1bd6-5027-a06e-b288b5ea7b6e
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1400, 640))

    # (a) The 3 × 3 blocks of garnet, to scale, with the beam arriving from above
    ax3 = Mke.Axis3(
        fig[1, 1];
        title = "(a) 3 × 3 dexels, to scale; beam from above",
        aspect = :data,
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        zlabel = "depth (mm)",
        azimuth = 1.3π,
        elevation = 0.17π,
        zreversed = true,
        protrusions = (75, 40, 40, 40),
        zticks = [0, d],
    )
    for i in -1:1, j in -1:1
        block = Mke.Rect3f(
            Mke.Vec3f(i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, 0),
            Mke.Vec3f(p[1] - s, p[2] - s, d),
        )
        central = i == 0 && j == 0
        Mke.mesh!(
            ax3,
            block;
            color = central ? (PALETTE.photon, 0.3) : (PALETTE.slab_edge, 0.12),
        )
        Mke.wireframe!(
            ax3,
            block;
            color = central ? PALETTE.photon : PALETTE.slab_edge,
            linewidth = central ? 2 : 0.7,
        )
    end
    Mke.lines!(
        ax3,
        [Mke.Point3f(0, 0, -1.2d), Mke.Point3f(0, 0, 0)];
        color = PALETTE.photon,
        linewidth = 4,
    )

    # (b) Seen from the beam: blocks of garnet and the reflector between them
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the beam",
        xlabel = "x, across the fan (mm)",
        ylabel = "y, along z (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 2.3p[1]), (-2.3p[2], 1.9p[2])),
    )
    Mke.poly!(
        axb,
        Mke.Rect2f(-1.5p[1], -1.5p[2], 3p[1], 3p[2]);
        color = PALETTE.reflector,
    )
    for i in -1:1, j in -1:1
        Mke.poly!(
            axb,
            Mke.Rect2f(
                i * p[1] - (p[1] - s) / 2,
                j * p[2] - (p[2] - s) / 2,
                p[1] - s,
                p[2] - s,
            );
            color = i == 0 && j == 0 ? PALETTE.ceramic_hit : PALETTE.ceramic,
            strokecolor = PALETTE.slab_edge,
            strokewidth = 1,
        )
    end
    Mke.bracket!(
        axb,
        -0.5p[1],
        -1.6p[2],
        0.5p[1],
        -1.6p[2];
        text = "$(round(p[1]; digits = 3)) mm",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )
    Mke.bracket!(
        axb,
        1.6p[1],
        -0.5p[2],
        1.6p[1],
        0.5p[2];
        text = "$(round(p[2]; digits = 3)) mm",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )
    note!(
        axb,
        0,
        0,
        "dexel";
        align = (:center, :center),
        color = PALETTE.photon,
        fontsize = 15,
    )
    note!(
        axb,
        1.5p[1],
        1.5p[2],
        "reflector, $(round(Int, 1000s)) µm";
        align = (:right, :bottom),
        offset = (0, 6),
        fontsize = 14,
        color = PALETTE.muted,
    )

    # (c) Side view: a photon absorbed, and its x ray heading for the next dexel
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side view",
        xlabel = "x (mm)",
        ylabel = "depth (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 1.9p[1]), (-0.6, d + 0.3)),
    )
    Mke.poly!(axc, Mke.Rect2f(-1.5p[1], 0, 3p[1], d); color = PALETTE.reflector)
    for i in -1:1
        Mke.poly!(
            axc,
            Mke.Rect2f(i * p[1] - (p[1] - s) / 2, 0, p[1] - s, d);
            color = PALETTE.ceramic,
            strokecolor = PALETTE.slab_edge,
        )
    end
    Mke.lines!(axc, [0.1, 0.1], [-0.6, 0.35]; color = PALETTE.photon, linewidth = 3)
    Mke.lines!(axc, [0.1, 0.8p[1]], [0.35, 1.0]; color = PALETTE.xray, linewidth = 3)
    Mke.scatter!(
        axc,
        [Mke.Point2f(0.1, 0.35), Mke.Point2f(0.8p[1], 1.0)];
        color = PALETTE.electron,
        markersize = 12,
    )
    note!(
        axc,
        0.18,
        -0.4,
        "photon";
        align = (:left, :center),
        color = PALETTE.photon,
        fontsize = 14,
    )
    note!(
        axc,
        0.35p[1],
        0.62,
        "Tb or Lu x ray";
        align = (:left, :bottom),
        color = PALETTE.xray,
        fontsize = 14,
    )
    Mke.bracket!(
        axc,
        1.6p[1],
        0,
        1.6p[1],
        d;
        text = "$(d) mm",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    fig
end

# ╔═╡ 3d4f918b-7df5-5eed-becb-a6dd61ba726c
md"""
### What a Photon Does in the Garnet

`record_tracks` follows a few hundred 70 keV photons (above both K-edges) landing on the
central dexel, every step of every particle. Two of them are drawn: one whose Tb or Lu x ray is
absorbed **in the next dexel**, so its energy is split between two photodiodes, and one whose
x ray **leaves through the front face**, so its energy is lost to every dexel.
"""

# ╔═╡ bfafebed-31d8-53f5-a471-25993d2ed385
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    70.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ 32ca1abd-3897-5936-a735-a52773f47bf5
# Two photons to follow: one whose Tb or Lu x ray stops in the next dexel across the fan (in the same
# row, so the side view shows it), one whose x ray leaves through the front face without stopping
followed = let
    in_ceramic(z) = 0 <= z <= depth_mm
    xray_steps(event) = [s for s in tracks.steps if s.event == event && s.particle === :fluorescence]
    events = unique(s.event for s in tracks.steps if s.particle === :fluorescence)

    across(s) = abs(s.stop[1]) > pitch_mm[1] / 2 && abs(s.stop[2]) < crystal_mm[2] / 2
    next_door = [
        e for e in events
            if any(s.process == "phot" && in_ceramic(s.stop[3]) && across(s) for s in xray_steps(e))
    ]
    front = [
        e for e in events
            if !any(s.process == "phot" for s in xray_steps(e)) && any(s.stop[3] < -0.01 for s in xray_steps(e))
    ]
    (next_door = first(next_door), front = first(front))
end

# ╔═╡ 76876cdf-0539-5b54-a239-98f3e02df836
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    top = -0.9
    fig = Mke.Figure(size = (1300, 520))

    # Clip a step to the drawing (above the garnet it may run off to the world's edge)
    function clipped(a, b)
        a[3] >= top && b[3] >= top && return (a, b)
        t = (top - a[3]) / (b[3] - a[3])
        cut = a .+ t .* (b .- a)
        return a[3] < top ? (cut, b) : (a, cut)
    end

    function panel!(ax, event)
        Mke.poly!(ax, Mke.Rect2f(-1.5p[1], 0, 3p[1], d); color = PALETTE.reflector)
        for i in -1:1
            Mke.poly!(
                ax,
                Mke.Rect2f(i * p[1] - (p[1] - s) / 2, 0, p[1] - s, d);
                color = PALETTE.ceramic,
                strokecolor = PALETTE.slab_edge,
            )
        end
        kept = Dict(-1 => 0.0, 0 => 0.0, 1 => 0.0)
        for st in tracks.steps
            st.event == event || continue
            if st.edep_keV > 0 && 0 <= st.stop[3] <= d && abs(st.stop[2]) < p[2] / 2
                i = round(Int, st.stop[1] / p[1])
                haskey(kept, i) && (kept[i] += st.edep_keV)
            end
            st.particle in (:primary, :fluorescence) || continue
            a, b = clipped(st.start, st.stop)
            Mke.lines!(
                ax,
                [a[1], b[1]],
                [a[3], b[3]];
                color = st.particle === :primary ? PALETTE.photon : PALETTE.xray,
                linewidth = 3,
            )
            if st.process == "phot" && 0 <= st.stop[3] <= d
                Mke.scatter!(
                    ax,
                    [st.stop[1]],
                    [st.stop[3]];
                    color = PALETTE.electron,
                    markersize = 13,
                )
            end
        end
        for (i, e) in kept
            e > 0.5 || continue
            note!(
                ax,
                i * p[1],
                d + 0.12,
                "$(round(e; digits = 1)) keV";
                align = (:center, :top),
                color = PALETTE.kept,
                fontsize = 15,
            )
        end
        return nothing
    end

    common = (
        xlabel = "x, across the fan (mm)",
        ylabel = "depth (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.55p[1], 1.55p[1]), (top, d + 0.55)),
    )
    axa = Mke.Axis(fig[1, 1]; title = "(a) Its x ray stops in the next dexel", common...)
    axb = Mke.Axis(fig[1, 2]; title = "(b) Its x ray leaves through the front", common...)
    panel!(axa, followed.next_door)
    panel!(axb, followed.front)
    Mke.Label(
        fig[2, 1:2],
        "blue: the 70 keV photon     orange: its Tb or Lu x ray     red dots: where they are absorbed     green: the energy each dexel keeps";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ fdb32583-deac-5102-80c8-d5c90e587dc2
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is a spread: the full
photon energy when everything stays in the dexel, the energy minus 44.5 keV when a terbium x ray
leaves, minus 54.1 keV when a lutetium x ray leaves, a little when the photon only
Compton-scatters once and leaves. Three energies show the regimes: 50 keV (below both K-edges),
70 keV (above both) and 120 keV.
"""

# ╔═╡ a2a74ae1-42e5-5360-b821-fa1c5835af23
# 100 000 photons at each energy, every energy deposit recorded
signal_runs = Dict(
    E => MC.simulate_monoenergetic(
            dexel_array(depth_mm),
            E;
            n = 100_000,
            beam = crystal_beam,
            seed = 5,
            record_deposits = true,
        )
        for E in (50.0, 70.0, 120.0)
)

# ╔═╡ d465f476-4b7a-5a26-8485-e167450cf6e3
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ be0eef36-6795-5094-814c-7838d52f5e92
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 520))
    for (k, E) in enumerate((50.0, 70.0, 120.0))
        h = histogram(own_signal[E], E + 4)
        ax = Mke.Axis(
            fig[1, k];
            title = "$(Int(E)) keV photons",
            xlabel = "energy the dexel collects (keV)",
            ylabel = k == 1 ? "probability per keV" : "",
            yscale = log10,
            limits = ((0, E + 4), (1.0e-5, 2)),
        )
        Mke.lines!(
            ax,
            h.centres,
            max.(h.density, 1.0e-6);
            color = PALETTE.kept,
            linewidth = 2.5,
        )
        note!(
            ax,
            E,
            0.9,
            "all of it";
            align = (:right, :center),
            offset = (-6, 0),
            color = PALETTE.kept,
            fontsize = 14,
        )
        for (edge, y) in ((terbium, 0.3), (lutetium, 0.05))
            E > edge.edge_keV || continue
            note!(
                ax,
                E - edge.kalpha_keV,
                y,
                "$(edge.element) x ray lost:\nE − $(round(edge.kalpha_keV; digits = 1)) keV";
                align = (:left, :center),
                offset = (12, 0),
                color = PALETTE.xray,
                fontsize = 14,
            )
        end
        if E > 100
            note!(
                ax,
                25,
                0.002,
                "Compton scatters\nthat left";
                align = (:center, :center),
                color = PALETTE.photon,
                fontsize = 14,
            )
        end
    end
    fig
end

# ╔═╡ fb4ffcd0-bac4-5ce3-858c-c45c2085c83b
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the garnet or
leaves it: as a Tb or Lu x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of garnet (no reflector), so that whatever
leaves the garnet leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ 71ce3fe1-447b-54b5-8ded-a897f1c8636e
budget_energies = 15.0:5.0:140.0

# ╔═╡ c1642d2c-c743-50d8-8fa0-b307a93526da
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(gemstone, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ 590bc8d2-76b3-5e54-9bc8-d8aa7104552e
let
    Mke.set_theme!(THEME)
    E = collect(budget_energies)
    parts = (
        (
            values = [b.absorbed for b in budgets],
            label = "stays in the garnet (the signal)",
            color = PALETTE.kept,
        ),
        (
            values = [b.fluorescence for b in budgets],
            label = "leaves as a Tb or Lu x ray",
            color = PALETTE.xray,
        ),
        (
            values = [b.scattered for b in budgets],
            label = "leaves as a scattered photon",
            color = PALETTE.photon,
        ),
        (
            values = [b.transmitted for b in budgets],
            label = "passes straight through",
            color = PALETTE.muted,
        ),
    )
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Where the photon's energy goes: $(depth_mm) mm of Gemstone",
        xlabel = "photon energy (keV)",
        ylabel = "fraction of the photon's energy",
        xticks = 20:20:140,
        limits = ((15, 140), (0, 1)),
    )
    below = zeros(length(E))
    for part in parts
        above = below .+ part.values
        Mke.band!(ax, E, below, above; color = (part.color, 0.75), label = part.label)
        below = above
    end
    Mke.vlines!(
        ax,
        [terbium.edge_keV, lutetium.edge_keV];
        color = PALETTE.ink,
        linestyle = :dot,
        linewidth = 1.5,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 90bbeea9-e955-5350-9c91-055bb731efb2
let
    k = findfirst(>(lutetium.edge_keV), collect(budget_energies))
    b = budgets[k]
    closure = maximum(abs(b.total - 1) for b in budgets)
    Markdown.parse(
        """
        Just above the Lu K-edge, at $(Int(budget_energies[k])) keV, **$(round(100b.fluorescence; digits = 1)) %** of the energy leaves
        as Tb or Lu x rays and $(round(100b.absorbed; digits = 1)) % stays. Gemstone is $(depth_mm) mm deep, so far fewer photons pass
        straight through than in the thinner Siemens ceramics. The energy balance closes to within
        $(round(100closure; digits = 2)) % at every energy.
        """
    )
end

# ╔═╡ 76eecb6b-bcfa-5b64-820a-0ab315fad43f
md"""
## 4. The Efficiency and Noise of One Dexel

The full simulation: at every whole keV from 1 to 140 (the grid of BasisSimulator's table),
100 000 photons land uniformly on one dexel's garnet, and each run is reduced, where it ran, to
the energy every dexel of the 3 × 3 collected and the energy left anywhere in the garnet. From
them come four curves:

- **interaction probability**: the photon interacts at all (`1 − exp(−μd)`, the check);
- **one-dexel efficiency**: the energy the dexel the photon landed on keeps, ÷ E;
- **flat-field efficiency**: the energy left anywhere in the garnet, ÷ E. In a scan every dexel
  is illuminated, and what one dexel loses to its neighbours it gets back from them, so this is
  what each dexel collects per photon landing on it: **the efficiency a CT detector has**;
- **Swank factor** `I = M₁² / (M₀ M₂)` of the flat-field signal: the noise the spread of
  collected energies adds. A detector's zero-frequency DQE is `M₁² / M₂` per incident photon, the
  interaction probability times `I`.
"""

# ╔═╡ 27dc6d6f-6037-5f6a-b642-1c5af6a9bc4d
energies = 1.0:1.0:140.0

# ╔═╡ 05c86c98-403d-55b7-ae80-f23a4cbbfcd5
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 651673f4-ac81-5aa2-9b6c-4d3a1f232562
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ fdb38e4c-3f74-5de4-be98-cdda136f1a56
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1400, 600))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Efficiency, $(depth_mm) mm of Gemstone",
        xlabel = "photon energy (keV)",
        ylabel = "efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        axa,
        E,
        efficiency.interaction;
        color = PALETTE.ink,
        linestyle = :dot,
        linewidth = 2.5,
        label = "photon interacts at all",
    )
    Mke.lines!(
        axa,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3.5,
        label = "flat field (what a scan sees)",
    )
    Mke.lines!(
        axa,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2.5,
        linestyle = :dash,
        label = "kept by the dexel it landed on",
    )
    note!(
        axa,
        terbium.edge_keV,
        0.75,
        "Tb and Lu\nK-edges";
        align = (:right, :center),
        offset = (-8, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.vlines!(
        axa,
        [terbium.edge_keV, lutetium.edge_keV];
        color = (PALETTE.xray, 0.6),
        linestyle = :dot,
    )
    Mke.axislegend(axa; position = :lb, margin = (10, 10, 40, 10))

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Noise: the Swank factor and DQE(0)",
        xlabel = "photon energy (keV)",
        ylabel = "Swank factor I,  DQE(0)",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        axb,
        E,
        efficiency.swank;
        color = PALETTE.electron,
        linewidth = 3,
        label = "Swank factor I (spread of the signal)",
    )
    Mke.lines!(
        axb,
        E,
        efficiency.dqe;
        color = PALETTE.photon,
        linewidth = 3,
        label = "DQE(0) = interaction × I",
    )
    Mke.vlines!(
        axb,
        [terbium.edge_keV, lutetium.edge_keV];
        color = (PALETTE.xray, 0.6),
        linestyle = :dot,
    )
    Mke.axislegend(axb; position = :lb)
    fig
end

# ╔═╡ f192b404-d795-5a67-8bf5-87aa47aeda2e
let
    at(E) = findfirst(==(E), collect(energies))
    k70, k140 = at(70.0), at(140.0)
    beer = 1 - exp(-MC.geant4_attenuation(gemstone, 140.0) * depth_mm)
    Markdown.parse(
        """
        At 70 keV, above both K-edges, the dexel a photon lands on keeps **$(round(efficiency.one_dexel[k70]; digits = 3))** of its
        energy; with every dexel illuminated each collects **$(round(efficiency.flat_field[k70]; digits = 3))**: the Tb and Lu x rays a
        dexel loses to its neighbours, it gets back from them. The Swank factor there is $(round(efficiency.swank[k70]; digits = 3)).
        At 140 keV the photon interacts with probability $(round(efficiency.interaction[k140]; digits = 3)) (Beer–Lambert:
        $(round(beer; digits = 3))), and the flat-field efficiency is $(round(efficiency.flat_field[k140]; digits = 3)).
        """
    )
end

# ╔═╡ 243b6cd3-d4a7-5e9a-a7d0-39a424919164
md"""
### How Much the Unknown Composition Matters

The garnet's Tb/Lu split is the one material number nobody publishes. The flat-field efficiency
at 3.0 mm for the even split (6.40 g/cm³, used here), for the Tb-rich garnet of GE's patent
(6.24 g/cm³), and for BasisSimulator's impossible 7.0 g/cm³:
"""

# ╔═╡ ee6081a3-7bb9-5320-a338-909cf1251027
composition_runs = Dict(
    label => MC.energy_absorption_efficiency.(
            MC.simulate_sweep(
                [(dexel_array(depth_mm; material), E) for E in energies];
                n = 50_000,
                beam = crystal_beam,
                seed = 23,
                readout,
            ),
        )
        for (label, material) in (("Tb-rich, 6.24", gemstone_tb_rich), ("even split, 7.0", gemstone_basis))
)

# ╔═╡ 7f4daa38-7a47-5d7f-81c6-2aa821f3715e
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 520))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Flat-field efficiency at 3.0 mm, three garnets",
        xlabel = "photon energy (keV)",
        ylabel = "efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.6, 1.02)),
    )
    Mke.lines!(
        ax,
        E,
        composition_runs["even split, 7.0"];
        color = PALETTE.reference,
        linewidth = 2.5,
        linestyle = :dash,
        label = "even split, 7.0 g/cm³ (BasisSimulator's, impossible)",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.flat_field;
        color = PALETTE.photon,
        linewidth = 3.5,
        label = "even split, 6.40 g/cm³ (used here)",
    )
    Mke.lines!(
        ax,
        E,
        composition_runs["Tb-rich, 6.24"];
        color = PALETTE.kept,
        linewidth = 2.5,
        label = "Tb-rich (GE's patent), 6.24 g/cm³",
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 208a44d2-987f-5887-b584-1f976fcc90a5
let
    E = collect(energies)
    spread = maximum(abs.(composition_runs["Tb-rich, 6.24"] .- efficiency.flat_field))
    density = maximum(abs.(composition_runs["even split, 7.0"] .- efficiency.flat_field))
    Markdown.parse(
        """
        Between the two possible garnets the flat-field efficiency differs by at most **$(round(spread; digits = 3))**
        (largest where the Tb and Lu edges sit); BasisSimulator's impossible density moves it by up to
        $(round(density; digits = 3)). Both possible garnets are saved with the table.
        """
    )
end

# ╔═╡ 56b952ce-233b-5076-a0ad-fa6756455914
md"""
### A Check Against Published Monte Carlo

The efficiency above the K-edges is shaped by fluorescence escape. No benchmark exists for a
garnet, but Boone et al. (Med. Phys. 1999) followed the same physics in thin Gd₂O₂S screens with
the TART 98 code: a pencil beam at normal incidence on Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S), the energy tallied in a circle 0.1 mm across the beam and in rings out to
4.5 mm, and ideal sensors in front of the screen and behind it. At 55 keV they report (p. 908)
how the incident energy divides. The same screen and the same tallies in Geant4:
"""

# ╔═╡ 1d5ea56f-89a5-535f-a2ab-dce36fcb20a7
boone = let
    # Gd₂O₂S with 5 % binder by weight (Carboset, an acrylic, taken as C₅H₈O₂)
    fractions = Dict(
        "Gd" => 0.95 * 0.83083,
        "O" => 0.95 * 0.08453 + 0.05 * 0.3196,
        "S" => 0.95 * 0.08469,
        "C" => 0.05 * 0.5998,
        "H" => 0.05 * 0.0805,
    )
    material = MC.DetectorMaterial(
        "Gd2O2S screen with binder";
        density = 7.34,
        mass_fractions = fractions,
    )
    screen = MC.SlabDetector(
        material,
        (0.09 / 0.95) / 7.34 * 10;
        width_mm = 30.0,
        shape = :disc,
    )

    n = 400_000
    run = MC.simulate_monoenergetic(
        screen,
        55.0;
        n,
        beam = MC.PencilBeam(),
        seed = 11,
        record_deposits = true,
    )
    incident = n * 55.0
    dep, esc = run.deposits, run.escapes
    radius(k) = hypot(dep.x_mm[k], dep.y_mm[k])
    (
        primary = sum(dep.energy_keV[k] for k in eachindex(dep.event) if radius(k) <= 0.1) / incident,
        lateral = sum(dep.energy_keV[k] for k in eachindex(dep.event) if 0.1 < radius(k) <= 4.5) / incident,
        backward = sum(esc.energy_keV[i] for i in eachindex(esc.kind) if esc.backward[i]) / incident,
        forward = sum(
            esc.energy_keV[i] for i in eachindex(esc.kind) if !esc.backward[i] && esc.kind[i] !== :transmitted
        ) / incident,
        transmitted = sum(esc.energy_keV[i] for i in eachindex(esc.kind) if esc.kind[i] === :transmitted) / incident,
    )
end

# ╔═╡ 8ec58a1a-ddc9-51e5-a0e4-6ced725ac8fe
let
    pct(x) = round(100x; digits = 1)
    rows = [
        ("absorbed within 0.1 mm of the beam (\"primary\")", 27.9, boone.primary),
        ("reabsorbed 0.1–4.5 mm away (\"lateral\")", 7.6, boone.lateral),
        ("carried back toward the tube", 16.5, boone.backward),
        ("carried forward as secondary radiation", 14.8, boone.forward),
        (
            "transmitted primary (Boone: the rest)",
            round(100 - 27.9 - 7.6 - 16.5 - 14.8; digits = 1),
            boone.transmitted,
        ),
    ]
    escaped = boone.backward + boone.forward
    Markdown.parse(
        """
        | 55 keV, 90 mg/cm² Gd₂O₂S screen | Boone et al. 1999 (TART 98) | Geant4 |
        |---|---|---|
        $(join(["| $(r[1]) | $(r[2]) % | $(pct(r[3])) % |" for r in rows], "\n"))
        | **escaped, back and forward** | **31.3 %** | **$(pct(escaped)) %** |

        The photons that pass straight through agree, so the screen and its attenuation are the same, and so does the
        split of what escapes between back and forward ($(round(boone.backward / boone.forward; digits = 2)) against 1.11). But Geant4 lets
        $(round(31.3 - 100escaped; digits = 1)) points less escape and keeps $(round(100boone.primary - 27.9; digits = 1)) more within 0.1 mm of the beam: **⚠ TART 98 and Geant4 differ by
        about $(round(Int, 100 * (1 - 100escaped / 31.3))) % in the energy the screen re-emits.** The same Geant4 setup reproduces EGSnrc's Swank factors
        for Gd₂O₂S to a few thousandths (Hajdok et al., notebook 02), and those depend directly on how much fluorescence
        escapes; the difference most likely lies in the 1999 code's atomic data rather than in this screen, but that is an
        inference, not a demonstration.
        """
    )
end

# ╔═╡ 4ace8ef9-4637-5260-9f14-5f7685e76537
md"""
## 5. Compared with BasisSimulator's Current Table

BasisSimulator's table is an energy-absorption efficiency (its K-edge steps go down, as
fluorescence escape makes them), for a crystal BasisSimulator documents as 3.0 mm deep. A photon
that never interacts leaves no energy, so the probability of interacting at all is a ceiling on
the efficiency.
"""

# ╔═╡ 797de5bb-99b6-5094-a316-d8ca2bfb284f
# BasisSimulator's table (MCNP, Hamidreza Khodajou-Chokami)
# BasisSimulator v0.15.1's table, as BasisMonteCarlo keeps it for comparison
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.gemstone

# ╔═╡ 6830b6a0-8ae0-5e43-80e9-da1cc552508c
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ f27dd4ed-4167-5bd4-ac84-a8f98e88754b
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(gemstone_basis, e) * depth_mm) for e in E]
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "BasisSimulator's table against 3.0 mm of Gemstone",
        xlabel = "photon energy (keV)",
        ylabel = "efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.55, 1.06)),
    )
    Mke.band!(ax, E, ceiling, fill(1.06, length(E)); color = (PALETTE.electron, 0.08))
    Mke.lines!(
        ax,
        E,
        ceiling;
        color = PALETTE.electron,
        linewidth = 2,
        linestyle = :dot,
        label = "ceiling: 3.0 mm interacts at all (7.0 g/cm³)",
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 3.5,
        label = "BasisSimulator's table (MCNP)",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, 3.0 mm, flat field",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 3.0 mm, one dexel",
    )
    note!(
        ax,
        120,
        1.02,
        "impossible for 3.0 mm";
        align = (:center, :center),
        color = PALETTE.electron,
        fontsize = 15,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 2ad68ded-ca19-504d-a805-f0caf48d398b
let
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(gemstone_basis, e) * depth_mm) for e in E]
    margin = minimum(ceiling .- table_at.(E))
    Markdown.parse(
        """
        Unlike the Siemens tables, this one stays under its ceiling at every energy (closest at $(Int(E[argmin(ceiling .- table_at.(E))])) keV,
        $(round(margin; digits = 3)) below it): it can describe a 3.0 mm crystal of its material. But it sits below Geant4's
        flat field above the K-edges, the signature of one dexel under a pencil beam. Which crystal does it describe?
        """
    )
end

# ╔═╡ 23cc9c86-3ad7-5182-a4e2-9857fcf1c6f2
md"""
### Which Crystal Did the Table Describe?

The table's shape fixes how it was made. A pencil beam into the centre of one dexel, tallied in
that dexel, gives its deep step at the K-edge (with a uniform beam or a flat field the step is
shallower); the fall-off at high energy then fixes the depth. So: pencil beams at 60–140 keV into
dexels 1.5 to 4.0 mm deep, in BasisSimulator's garnet at its 7.0 g/cm³, compared with the
table, with one constant factor for the table's flat 1 % loss at low energy.
"""

# ╔═╡ 2f3c0378-a2dd-5a8a-aac2-8ecba127fdc3
scan_depths = 1.5:0.25:4.0

# ╔═╡ 1ae1a7b5-befc-57dd-8506-a3904c6e56e2
scan_energies = 60.0:5.0:140.0

# ╔═╡ 876a93ce-7ca7-5f15-a44d-80a8c18bd676
# One column per depth, one row per energy
pencil_scan = reshape(
    MC.simulate_sweep(
        [
            (dexel_array(depth; material = gemstone_basis), E)
                for depth in scan_depths for E in scan_energies
        ];
        n = 50_000,
        beam = MC.PencilBeam(; start_mm = 12.0),
        seed = 13,
        readout,
    ),
    length(scan_energies),
    length(scan_depths),
)

# ╔═╡ 24568e75-698d-5a5e-9401-f553959c2742
# How far one depth's pencil-beam efficiencies are from the table, after the best constant factor
function misfit(runs, target)
    ours = MC.dexel_efficiency.(runs)
    scale = sum(ours .* target) / sum(ours .^ 2)
    rms = sqrt(sum(abs2, scale .* ours .- target) / length(target))
    return (scale = scale, rms = rms)
end

# ╔═╡ e41cfdd8-b032-5c29-abef-a5c856dec82f
fit = let
    target = table_at.(scan_energies)
    misfits = [misfit(pencil_scan[:, k], target) for k in eachindex(scan_depths)]
    best = argmin([m.rms for m in misfits])
    (depth_mm = scan_depths[best], scale = misfits[best].scale, misfits = misfits)
end

# ╔═╡ ba31b322-bf2a-57ca-8068-9858e8ac5d00
let
    Mke.set_theme!(THEME)
    rms = [m.rms for m in fit.misfits]
    fig = Mke.Figure(size = (1200, 500))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Which crystal depth reproduces BasisSimulator's table?",
        xlabel = "crystal depth in the simulation (mm)",
        ylabel = "rms difference from the table\n(60–140 keV)",
        yscale = log10,
        limits = (nothing, (1.0e-3, 0.5)),
    )
    Mke.vlines!(ax, depth_mm; color = PALETTE.reference, linestyle = :dash, linewidth = 2)
    Mke.scatterlines!(
        ax,
        collect(scan_depths),
        rms;
        color = PALETTE.photon,
        markersize = 11,
    )
    note!(
        ax,
        fit.depth_mm,
        minimum(rms),
        "best: $(fit.depth_mm) mm";
        align = (:center, :top),
        offset = (0, -12),
        color = PALETTE.photon,
        fontsize = 16,
    )
    note!(
        ax,
        depth_mm,
        0.3,
        "BasisSimulator documents 3.0 mm";
        align = (:left, :center),
        offset = (8, 0),
        color = PALETTE.reference,
        fontsize = 15,
    )
    fig
end

# ╔═╡ 05b60d54-a668-5497-9fc5-dcedf626515e
# The table's recipe at the depth that fits, at every energy
reproduction = MC.simulate_sweep(
    [(dexel_array(fit.depth_mm; material = gemstone_basis), E) for E in energies];
    n = 100_000,
    beam = MC.PencilBeam(; start_mm = 12.0),
    seed = 15,
    readout,
)

# ╔═╡ d067a0b4-d46d-5a21-a351-d3c7e347f9b7
reproduced = fit.scale .* MC.dexel_efficiency.(reproduction)

# ╔═╡ 54179647-5aa9-520a-a0ce-933b5e136cbf
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The table, and Geant4's pencil beam into one dexel at $(fit.depth_mm) mm (× $(round(fit.scale; digits = 3)))",
        ylabel = "efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 5,
        label = "BasisSimulator's table (MCNP)",
    )
    Mke.lines!(
        ax,
        E,
        reproduced;
        color = PALETTE.photon,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, reproduced",
    )
    Mke.axislegend(ax; position = :lb)
    axr = Mke.Axis(
        fig[2, 1];
        xlabel = "photon energy (keV)",
        ylabel = "Geant4 − table",
        xticks = 0:20:140,
        limits = ((0, 141), (-0.04, 0.04)),
    )
    Mke.hspan!(axr, -0.01, 0.01; color = (:black, 0.06))
    Mke.hlines!(axr, 0; color = PALETTE.ink, linewidth = 1)
    Mke.lines!(axr, E, reproduced .- table_at.(E); color = PALETTE.photon, linewidth = 2)
    Mke.linkxaxes!(ax, axr)
    Mke.hidexdecorations!(ax; grid = false)
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.25))
    fig
end

# ╔═╡ 25b60608-28d7-5442-89df-fb1e7ac471c8
let
    E = collect(energies)
    difference = reproduced .- table_at.(E)
    away_from_edge = abs.(E .- terbium.edge_keV) .> 0.5
    typical = sum(abs, difference) / length(difference)
    Markdown.parse(
        """
        The table is reproduced to **$(round(typical; digits = 4))** on average by a pencil beam into the centre of one dexel,
        tallied in that dexel, in BasisSimulator's garnet at 7.0 g/cm³ **$(fit.depth_mm) mm deep**, times $(round(fit.scale; digits = 3))$(abs(fit.depth_mm - depth_mm) <= 0.25 ? ": within one step of the scan of the 3.0 mm BasisSimulator documents" : ", not the 3.0 mm BasisSimulator documents").
        Away from 52 keV the largest difference is $(round(maximum(abs, difference[away_from_edge]); digits = 3)).
        At 52 keV, on the terbium K-edge (51.996 keV), Geant4 places the photon 4 eV above the edge and the table
        below it: one point, a few eV of edge energy or a shifted energy grid in the MCNP run.

        So the table is BasisSimulator's documented detector, with two problems: **⚠ its density is impossible**
        (section 1), and it is one dexel under a pencil beam rather than the flat field a scan sees (section 4).
        """
    )
end

# ╔═╡ 55dad12b-a818-55c3-b5ba-aea9be0cd98c
md"""
## 6. The Anti-Scatter Grid

GE calls it a **"3D collimator"**: a two-dimensional grid behind the patient, walls between
columns and between rows, focused on the focal spot. Its product dimensions are not published;
GE's patent for a combined reflector and collimator (US 8,385,499) gives walls **0.10–0.20 mm**
thick, **about 8 mm** tall, with "a pitch of channels identical to a pitch of detector
elements", made of tungsten (or lead, tantalum) powder in a binder. Built here: **solid
tungsten walls 0.15 mm thick and 8 mm tall**, one cell per dexel, standing on the reflector.
0.15 mm walls on 0.1 mm of reflector shade a 25 µm strip of garnet on each side.
"""

# ╔═╡ d89950fe-ea70-592b-998b-3e954534f5c9
grid = MC.Collimator(;
    height_mm = 8.0,
    wall_mm = 0.15,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :both,
    cells = 13,
)

# ╔═╡ 6b89ebe5-d2ca-5d77-93e1-ec26b393c271
let
    Mke.set_theme!(THEME)
    p, s, d, H, w = pitch_mm, septum_mm, depth_mm, grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1400, 700))

    # (a) From the beam: tungsten walls over the reflector, over the garnet
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Seen from the beam",
        xlabel = "x, across the fan (mm)",
        ylabel = "y, along z (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 1.6p[1]), (-1.6p[2], 1.6p[2])),
    )
    Mke.poly!(axa, Mke.Rect2f(-1.5p[1], -1.5p[2], 3p[1], 3p[2]); color = PALETTE.reflector)
    for i in -1:1, j in -1:1
        Mke.poly!(
            axa,
            Mke.Rect2f(
                i * p[1] - (p[1] - s) / 2,
                j * p[2] - (p[2] - s) / 2,
                p[1] - s,
                p[2] - s,
            );
            color = PALETTE.ceramic,
        )
    end
    for i in -1:2
        Mke.poly!(
            axa,
            Mke.Rect2f((i - 0.5) * p[1] - w / 2, -1.5p[2], w, 3p[2]);
            color = PALETTE.tungsten,
        )
        Mke.poly!(
            axa,
            Mke.Rect2f(-1.5p[1], (i - 0.5) * p[2] - w / 2, 3p[1], w);
            color = PALETTE.tungsten,
        )
    end
    note!(
        axa,
        0,
        0,
        "garnet";
        align = (:center, :center),
        color = PALETTE.slab_edge,
        fontsize = 15,
    )
    note!(
        axa,
        -1.45p[1],
        1.45p[2],
        "black: tungsten walls, $(round(Int, 1000w)) µm\nbeige: reflector, $(round(Int, 1000s)) µm";
        align = (:left, :top),
        fontsize = 14,
        color = PALETTE.ink,
    )

    # (b) Side view, to scale: a photon from the focal spot passes, a scattered one is stopped
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Side view, to scale",
        xlabel = "x, across the fan (mm)",
        ylabel = "height above the garnet (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        yticks = ([-10, -5, 0], ["10", "5", "0"]),
        limits = ((-1.6p[1], 1.6p[1] + 4.2), (-H - 1.2, d + 0.3)),
    )
    Mke.poly!(axb, Mke.Rect2f(-1.5p[1], 0, 3p[1], d); color = PALETTE.reflector)
    for i in -1:1
        Mke.poly!(
            axb,
            Mke.Rect2f(i * p[1] - (p[1] - s) / 2, 0, p[1] - s, d);
            color = PALETTE.ceramic,
            strokecolor = PALETTE.slab_edge,
        )
    end
    for i in -1:2
        Mke.poly!(
            axb,
            Mke.Rect2f((i - 0.5) * p[1] - w / 2, -H, w, H);
            color = PALETTE.tungsten,
        )
    end

    # From the focal spot: straight down the middle of a channel, into the garnet
    Mke.lines!(axb, [0.15, 0.15], [-H - 1.2, 0.5]; color = PALETTE.photon, linewidth = 3)

    # Scattered at 20°: enters the left channel and runs into its wall
    slope = tand(20)
    top_x = -1.42p[1]
    hit = Mke.Point2f(-0.5p[1] - w / 2, -H + (-0.5p[1] - w / 2 - top_x) / slope)
    start = Mke.Point2f(top_x - 1.2 * slope, -H - 1.2)
    Mke.lines!(axb, [start, hit]; color = PALETTE.xray, linewidth = 3)
    Mke.scatter!(axb, [hit]; color = PALETTE.xray, markersize = 16, marker = :xcross)

    note!(
        axb,
        1.6p[1] + 0.55,
        -H + 1.0,
        "blue: from the focal spot,\npasses into the garnet";
        align = (:left, :center),
        color = PALETTE.photon,
        fontsize = 14,
    )
    note!(
        axb,
        1.6p[1] + 0.55,
        -H + 3.5,
        "orange: scattered in the\npatient, stopped by a wall";
        align = (:left, :center),
        color = PALETTE.xray,
        fontsize = 14,
    )
    Mke.bracket!(
        axb,
        1.5p[1] + 0.25,
        -H,
        1.5p[1] + 0.25,
        0;
        text = "$(round(Int, H)) mm of tungsten",
        orientation = :down,
        fontsize = 15,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.5))
    fig
end

# ╔═╡ c599896a-d8dc-5d31-8aa6-d04f7eb6bf7e
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (garnet and reflector), with and without the grid: the grid
should cost only the strip of garnet its walls shade, and its own tungsten x rays (59 and
67 keV) reach the garnet above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ 367261c6-ffb8-5661-970e-6494b0a10197
grid_energies = 20.0:5.0:140.0

# ╔═╡ 8c9ab3eb-23b7-57a5-9c13-0aa983dae5e0
# Photons over one dexel's whole cell, straight down, without and with the grid
grid_runs = (
    without = MC.simulate_sweep(
        [(dexel_array(depth_mm), E) for E in grid_energies];
        n = 100_000,
        beam = cell_beam,
        seed = 17,
        readout,
    ),
    with = MC.simulate_sweep(
        [(dexel_array(depth_mm; grid), E) for E in grid_energies];
        n = 100_000,
        beam = cell_beam,
        seed = 17,
        readout,
    ),
)

# ╔═╡ 387074f5-5299-5768-b50a-e1a315621ab8
angles_deg = 0.0:1.0:20.0

# ╔═╡ 173503d6-98e5-51d2-b783-7e5885c408eb
# 60 keV photons arriving at an angle across the fan, over a whole cell, without and with the grid
angle_runs = let
    tilted(angle) = MC.AreaBeam(;
        width_mm = pitch_mm[1],
        height_mm = pitch_mm[2],
        angle_x_deg = angle,
        start_mm = 12.0,
    )
    (
        without = MC.simulate_sweep(
            [(dexel_array(depth_mm), 60.0, 21, tilted(a)) for a in angles_deg];
            n = 100_000,
            readout,
        ),
        with = MC.simulate_sweep(
            [(dexel_array(depth_mm; grid), 60.0, 21, tilted(a)) for a in angles_deg];
            n = 100_000,
            readout,
        ),
    )
end

# ╔═╡ 8c9677c3-aac1-593c-a53c-ccc5218c3b3e
let
    Mke.set_theme!(THEME)
    E = collect(grid_energies)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Straight from the focal spot",
        xlabel = "photon energy (keV)",
        ylabel = "energy collected per photon on the cell, ÷ E",
        xticks = 20:20:140,
        limits = ((20, 140), (0.4, 1.0)),
    )
    Mke.lines!(
        axa,
        E,
        MC.energy_absorption_efficiency.(grid_runs.without);
        color = PALETTE.muted,
        linewidth = 3,
        label = "no grid",
    )
    Mke.lines!(
        axa,
        E,
        MC.energy_absorption_efficiency.(grid_runs.with);
        color = PALETTE.tungsten,
        linewidth = 2.5,
        linestyle = :dash,
        label = "with the 2-D grid",
    )
    Mke.vlines!(axa, 69.5; color = (PALETTE.xray, 0.6), linestyle = :dot)
    note!(
        axa,
        69.5,
        0.45,
        "W K-edge";
        align = (:left, :bottom),
        offset = (6, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.axislegend(axa; position = :rt)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) 60 keV photons arriving at an angle",
        xlabel = "angle across the fan (degrees)",
        ylabel = "energy collected, relative to\nstraight down without a grid",
        limits = ((0, 20), (0, 1.35)),
    )
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    Mke.lines!(
        axb,
        collect(angles_deg),
        MC.energy_absorption_efficiency.(angle_runs.without) ./ reference;
        color = PALETTE.muted,
        linewidth = 3,
        label = "no grid",
    )
    Mke.lines!(
        axb,
        collect(angles_deg),
        MC.energy_absorption_efficiency.(angle_runs.with) ./ reference;
        color = PALETTE.tungsten,
        linewidth = 3,
        label = "with the 2-D grid",
    )
    opening = pitch_mm[1] - grid.wall_mm
    geometric = [max(0, 1 - grid.height_mm * tand(a) / opening) for a in angles_deg]
    Mke.lines!(
        axb,
        collect(angles_deg),
        geometric;
        color = PALETTE.xray,
        linewidth = 2,
        linestyle = :dash,
        label = "geometry alone: 1 − H tan θ / opening",
    )
    Mke.axislegend(axb; position = :rc)
    fig
end

# ╔═╡ 9af6dffc-d99a-54bf-97cf-dd20156bbe48
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) %: the walls are 50 µm wider than the reflector they stand on, so the face open to the
        beam is $(round(prod((pitch_mm .- grid.wall_mm) ./ pitch_mm); digits = 3)) of the cell, against $(round(prod(crystal_mm ./ pitch_mm); digits = 3)) of garnet without the grid. If
        GE's walls are as thick as its patent's, BasisSimulator's fill factor (0.81) overstates the collecting face.
        (Above 69.5 keV the walls' own x rays add a little.)
        At an angle it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %. A
        photon that scatters in the patient arrives from anywhere, so most of it is stopped.
        """
    )
end

# ╔═╡ 89f83a80-82e6-5d12-8616-36560dba03a1
md"""
## 7. The Whole Detector

The whole detector, as BasisSimulator has it: 834 columns × 256 rows on an arc of radius 1100 mm
around the focal spot, 160 mm along z, every dexel a block of garnet in reflector, under the 2-D
grid, all of it focused on the focal spot (GE's "focally aligned" sub-modules). An air scan fires
photons uniformly from the focal spot over the whole detector, and every dexel's signal is
summed. Two questions: is every dexel the same, and does the whole detector collect what the
single dexel of sections 4 and 6 predicts?
"""

# ╔═╡ 8074b52c-eaab-5eef-ab41-b3482d729940
detector = MC.ArcDetector(;
    columns = apex.columns,
    rows = apex.rows,
    radius_mm = apex.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :both,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = gemstone,
)

# ╔═╡ bd408664-f831-56e0-971a-4d7c4f14fa33
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ 92105c08-3d1a-533c-8939-3c30460be4b4
# 3.2 million photons per energy from the focal spot, in 16 independent chunks
air_scans = let
    chunks = MC.simulate_sweep(
        [(detector, E, chunk) for E in air_energies for chunk in 1:16];
        n = 200_000,
        beam = MC.focal_spot(detector),
        readout = MC.ArcEnergyReadout(),
    )
    [sum(chunks[(16(k - 1) + 1):(16k)]) for k in eachindex(air_energies)]
end

# ╔═╡ 59e3353e-13f5-53f2-be17-2062c579f044
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 620))

    # (a) The scanner from above: focal spot, the detector arc, the isocentre
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The detector from above",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-560, 560), (-170, 1260)),
    )
    R = apex.source_to_detector_mm
    half = MC.fan_angle_deg(detector) / 2
    θ = range(-deg2rad(half), deg2rad(half); length = 200)
    Mke.lines!(axa, R .* sin.(θ), R .* cos.(θ); color = PALETTE.kept, linewidth = 6)
    for t in (first(θ), last(θ))
        Mke.lines!(
            axa,
            [0, R * sin(t)],
            [0, R * cos(t)];
            color = (PALETTE.photon, 0.5),
            linewidth = 1.5,
        )
    end
    Mke.scatter!(axa, [0], [0]; color = PALETTE.photon, markersize = 14)
    Mke.scatter!(
        axa,
        [0],
        [apex.source_to_isocenter_mm];
        color = PALETTE.ink,
        marker = :cross,
        markersize = 16,
    )
    note!(
        axa,
        0,
        0,
        "focal spot";
        align = (:center, :top),
        offset = (0, -10),
        color = PALETTE.photon,
        fontsize = 15,
    )
    note!(
        axa,
        0,
        apex.source_to_isocenter_mm,
        "isocentre";
        align = (:left, :center),
        offset = (12, 0),
        fontsize = 15,
    )
    note!(
        axa,
        0,
        R,
        "$(apex.columns) × $(apex.rows) dexels, $(round(MC.fan_angle_deg(detector); digits = 1))° fan";
        align = (:center, :bottom),
        offset = (0, 10),
        color = PALETTE.kept,
        fontsize = 15,
    )
    Mke.hidespines!(axa)

    # (b) The signal per photon arriving, across the fan, in groups of 40 columns
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Energy collected per photon, across the fan",
        xlabel = "column (across the fan)",
        ylabel = "energy collected per photon\non the cell, ÷ E",
        limits = ((1, apex.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:apex.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / apex.columns
        centres = [mean(g) for g in groups]
        per_group = [sum(scan.energy_keV[g, :]) / (length(g) * per_photon) for g in groups]
        Mke.scatterlines!(
            axb,
            centres,
            per_group;
            color = colors[k],
            markersize = 9,
            linewidth = 2,
            label = "$(Int(air_energies[k])) keV",
        )
        slab = MC.energy_absorption_efficiency(grid_runs.with[findfirst(==(air_energies[k]), collect(grid_energies))])
        Mke.hlines!(axb, slab; color = colors[k], linestyle = :dash, linewidth = 2)
    end
    note!(
        axb,
        apex.columns / 2,
        0.42,
        "dashed: one dexel's cell, section 6 (with the grid)";
        align = (:center, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )
    Mke.axislegend(
        axb;
        position = :rt,
        orientation = :horizontal,
        backgroundcolor = (:white, 0.85),
        framevisible = false,
    )

    # (c) The same along z, in groups of rows: the outer rows see the beam at the cone angle
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Along z",
        xlabel = "row (along z)",
        limits = ((1, apex.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:apex.rows, max(1, apex.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / apex.rows
        Mke.scatterlines!(
            axc,
            [mean(g) for g in row_groups],
            [sum(scan.energy_keV[:, g]) / (length(g) * per_photon) for g in row_groups];
            color = colors[k],
            markersize = 9,
            linewidth = 2,
        )
    end
    Mke.hideydecorations!(axc; grid = false)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.3))
    Mke.colsize!(fig.layout, 3, Mke.Relative(0.22))
    fig
end

# ╔═╡ ace5c787-13e0-57f3-93f1-345861f9c001
let
    lines = String[]
    ratios = Float64[]
    for (k, scan) in enumerate(air_scans)
        whole = sum(scan.energy_keV) / scan.photon_keV
        slab = MC.energy_absorption_efficiency(grid_runs.with[findfirst(==(air_energies[k]), collect(grid_energies))])
        push!(ratios, whole / slab)
        push!(
            lines,
            "| $(Int(air_energies[k])) keV | $(round(whole; digits = 4)) | $(round(slab; digits = 4)) |",
        )
    end

    # Outer rows against central rows, at each energy; the largest difference is reported
    function edge_of(scan)
        per_row = vec(sum(scan.energy_keV; dims = 1))
        n = length(per_row)
        eighth = max(1, n ÷ 8)
        centre = mean(per_row[(n ÷ 2 - eighth + 1):(n ÷ 2 + eighth)])
        outer = mean(vcat(per_row[1:eighth], per_row[(end - eighth + 1):end]))
        return outer / centre - 1
    end
    edges = edge_of.(air_scans)
    k = argmax(abs.(edges))
    edge = edges[k]
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / apex.source_to_detector_mm))
    worst = maximum(abs.(ratios .- 1))

    why = edge > 0 ?
        "a photon that lands on the reflector between rows at an angle passes through it into the next ring of crystal, so the outer rows catch photons the reflector would stop face-on" :
        "a photon that goes deep crosses from its ring of crystal into the reflector between rows"
    verdict = worst <= 0.01 || abs(edge) < 0.01 ?
        """The whole detector collects what one dexel's cell does, at every energy, to within $(round(100worst; digits = 1)) %: in an air
        scan each dexel gets back from its neighbours what it loses to them, and no column or row differs from the others
        beyond the statistics of the photons that reached it. One dexel, simulated in detail, stands for them all.""" :
        """Across the fan the whole detector collects what one dexel's cell does; the difference, up to
        $(round(100worst; digits = 1)) %, is along z. At $(Int(air_energies[k])) keV the outer rows collect $(round(100abs(edge); digits = 1)) % $(edge > 0 ? "more" : "less") than the central
        ones: the beam reaches them up to $(round(cone; digits = 1))° off the axis, and in this model the rings of crystal are perpendicular
        to the axis (only the grid's walls, where there are any, lean toward the focal spot), so $(why). Real detectors
        are built from modules turned toward the focal spot along z as well, which this model's single arc is not; the
        efficiency table, from one dexel facing the beam, is not affected."""
    Markdown.parse(
        """
        | photons of | whole detector (air scan) | one dexel's cell, with the grid (section 6) |
        |---|---|---|
        $(join(lines, "\n"))

        $(verdict)
        """
    )
end

# ╔═╡ 96693605-bc02-50f2-a3ea-da32946c6c79
md"""
## 8. The Apex's X-Ray Beam

The photons come from BasisSimulator itself: its IPEM Report 78 tungsten spectra (10° anode),
through the scanner's 2.5 mm aluminium flat filter and the protocol's 4.5 mm of aluminium,
scaled to the detector at 1100 mm, exactly as `simulate!` uses them for the Apex.
BasisSimulator's spectra are tabulated at 80, 100, 120 and 140 kV; its Apex runs 120 kV, and
80/140 kV with rapid kV switching.

The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ 4575a19b-f864-53fc-a050-40b57b16fc8f
# BasisSimulator's Revolution Apex, as on its scanner page
apex_scanner = BS.EICTScanner(
    source_to_isocenter = apex.source_to_isocenter_mm,
    source_to_detector = apex.source_to_detector_mm,
    detector_rows = apex.rows,
    detector_cols = apex.columns,
    detector_row_size = apex.row_at_isocenter_mm,
    detector_col_size = apex.column_at_isocenter_mm,
    detector_col_offset = 0.25,
    detector_shape = :arc,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex,
    detector_depth = depth_mm,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 3500.0,
    detection_gain = 10.0,
)

# ╔═╡ 9f4fa6af-f684-525c-9389-5081654e6d71
"""
The photons per mAs per mm² reaching the detector, per bin, for an Apex protocol at `kVp`, with
the protocol's 4.5 mm of aluminium.
"""
function apex_spectrum(kVp)
    protocol = BS.CTProtocol(;
        kVp,
        mA = 200.0,
        views = 984,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        additional_filters = [("Al", 4.5)],
    )
    return BS.resolve_source_spectrum_without_bowtie(
        BS.SimOptions(),
        protocol;
        scanner = apex_scanner,
    )
end

# ╔═╡ cf2e67e6-4692-5f99-bd79-a5fbc8fb9abf
@bind kvp PlutoUI.Slider([80, 100, 120, 140]; default = 120, show_value = true)

# ╔═╡ 1bae766b-aa3d-5389-937c-13cb98623d47
"""
For a spectrum `(e, w)`: the photons per keV, and the signal they make per keV in a detector of
efficiency `η` on BasisSimulator's 1-keV grid (`Φ(E) · E · η(E)`).
"""
function signal_per_kev(spectrum, η)
    e, w = spectrum
    bin = e[2] - e[1]
    at(E) = MC.interpolate_linear(collect(energies), η, E)
    return (
        energy_keV = e,
        photons = w ./ bin,
        signal = [wi / bin * E * at(E) for (E, wi) in zip(e, w)],
    )
end

# ╔═╡ 7f435d0f-3376-5aba-9422-da95aeee9fd4
let
    Mke.set_theme!(THEME)
    spectrum = apex_spectrum(kvp)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The photons reaching the detector, $(kvp) kV",
        xlabel = "photon energy (keV)",
        ylabel = "photons per mAs per mm²\nper keV",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    s = signal_per_kev(spectrum, efficiency.flat_field)
    Mke.lines!(axa, s.energy_keV, s.photons; color = PALETTE.photon, linewidth = 3)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The signal they make",
        xlabel = "photon energy (keV)",
        ylabel = "signal per keV\n(photons × E × η, arbitrary units)",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    theirs = signal_per_kev(spectrum, table_at.(collect(energies)))
    Mke.lines!(
        axb,
        theirs.energy_keV,
        theirs.signal;
        color = PALETTE.reference,
        linewidth = 3,
        label = "BasisSimulator's table",
    )
    Mke.lines!(
        axb,
        s.energy_keV,
        s.signal;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, flat field, 3.0 mm",
    )
    Mke.vlines!(
        axb,
        [terbium.edge_keV, lutetium.edge_keV];
        color = (PALETTE.xray, 0.6),
        linestyle = :dot,
    )
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ e8fa69d7-5554-59ca-9c0b-ab33f82e52dc
md"""
**The beam's quality, against measurement.** Kawahara et al. (Pol. J. Radiol. 2023) measured the
half-value layer of a Revolution CT (the Apex's predecessor, with the same Gemstone Clarity
detector) at the isocentre with a stationary tube and the large bowtie: **7.08, 9.20 and
10.0 mm of aluminium at 80, 120 and 140 kV**. At the isocentre the beam has passed the bowtie's
thinnest part, 2.5 mm of aluminium in BasisSimulator's GE large bowtie.
"""

# ╔═╡ a5ab816b-0816-53c4-a84d-40a2992a1de7
hvl_check = let
    bowtie = BS.resolve_bowtie_filter(:ge_revolution_large)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium = MC.DetectorMaterial(:Al)

    function hvl(kVp)
        e, w = apex_spectrum(kVp)
        μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
        return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
    end

    [(kVp = k, basis = hvl(k), measured = m) for (k, m) in ((80, 7.08), (120, 9.2), (140, 10.0))]
end

# ╔═╡ 8eae3d73-e14e-5d93-9420-74e73deb379c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-value layer at the isocentre",
        xlabel = "tube voltage",
        ylabel = "half-value layer (mm of aluminium)",
        xticks = (1:3, ["$(c.kVp) kV" for c in hvl_check]),
        limits = ((0.4, 3.6), (0, 13)),
    )
    x = 1:3
    Mke.barplot!(
        ax,
        x .- 0.18,
        [c.basis for c in hvl_check];
        width = 0.34,
        color = PALETTE.reference,
        label = "BasisSimulator's beam",
    )
    Mke.barplot!(
        ax,
        x .+ 0.18,
        [c.measured for c in hvl_check];
        width = 0.34,
        color = PALETTE.kept,
        label = "measured (Kawahara et al. 2023)",
    )
    for (k, c) in enumerate(hvl_check)
        for (dx, v) in ((-0.18, c.basis), (0.18, c.measured))
            note!(
                ax,
                k + dx,
                v,
                "$(round(v; digits = 1))";
                align = (:center, :bottom),
                offset = (0, 4),
                fontsize = 15,
                alpha = 0.0,
            )
        end
    end
    Mke.axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 5ab9e1a0-492b-54b0-bbdb-f272e3fa8bd1
let
    worst = maximum(abs(c.basis - c.measured) for c in hvl_check)
    verdict = worst <= 0.5 ?
        "BasisSimulator's Apex beam matches the measured beam quality at every tube voltage." :
        "**⚠ Discrepancy:** BasisSimulator's Apex beam differs from the measured one by up to $(round(worst; digits = 1)) mm of aluminium."
    Markdown.parse(
        """
        $(verdict) (The measurement is on a Revolution CT with its large bowtie; the Apex shares its tube filtration
        and bowtie family.) The rest of this notebook keeps BasisSimulator's beam, so the two stay comparable.
        """
    )
end

# ╔═╡ ba5b5a00-0e75-5e2c-9bae-727aec37b499
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. Every Apex tube voltage BasisSimulator can make:
"""

# ╔═╡ f775068c-2324-5cf4-8811-cd75f968a57d
scan_protocols = [(label = "$(k) kV", spectrum = apex_spectrum(k)) for k in (80, 100, 120, 140)]

# ╔═╡ 354a335b-59d0-565d-a29c-4a3242eac17c
# Each protocol's spectrum on the 1-keV grid, then its energy-weighted efficiencies and DQE(0)
scan_metrics = let
    E = collect(energies)
    m1 = [d.crystal_keV / d.n for d in one_dexel]
    m2 = [d.crystal2_keV2 / d.n for d in one_dexel]
    function metrics(spectrum)
        e, w = spectrum
        φ = [sum(w[abs.(e .- Ek) .<= 0.5]) for Ek in E]
        weighted(η) = sum(φ .* E .* η) / sum(φ .* E)
        return (
            table = weighted(table_at.(E)),
            geant4 = weighted(efficiency.flat_field),
            dqe = sum(φ .* m1)^2 / (sum(φ) * sum(φ .* m2)),
        )
    end
    [merge((label = p.label,), metrics(p.spectrum)) for p in scan_protocols]
end

# ╔═╡ ffeaa78c-e964-5348-9a8d-7e17230dbfc1
let
    Mke.set_theme!(THEME)
    x = 1:length(scan_metrics)
    labels = [m.label for m in scan_metrics]
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Mean efficiency over each protocol's spectrum",
        ylabel = "energy-weighted efficiency",
        xticks = (x, labels),
        xticklabelrotation = π / 5,
        limits = ((0.4, length(x) + 0.6), (0.5, 1.0)),
    )
    Mke.scatterlines!(
        axa,
        x,
        [m.table for m in scan_metrics];
        color = PALETTE.reference,
        markersize = 13,
        linewidth = 2.5,
        label = "BasisSimulator's table",
    )
    Mke.scatterlines!(
        axa,
        x,
        [m.geant4 for m in scan_metrics];
        color = PALETTE.kept,
        markersize = 13,
        linewidth = 2.5,
        label = "Geant4, 3.0 mm, flat field",
    )
    Mke.axislegend(axa; position = :lb)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) DQE(0) over each spectrum (Geant4)",
        ylabel = "DQE(0)",
        xticks = (x, labels),
        xticklabelrotation = π / 5,
        limits = ((0.4, length(x) + 0.6), (0.5, 1.0)),
    )
    Mke.scatterlines!(
        axb,
        x,
        [m.dqe for m in scan_metrics];
        color = PALETTE.photon,
        markersize = 13,
        linewidth = 2.5,
    )
    fig
end

# ╔═╡ 2077865c-68c8-5b80-8b56-275858a50a24
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | BasisSimulator's table | Geant4, 3.0 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's garnet; BasisSimulator's fill factor (0.9 × 0.9) applies to both
        alike.
        """
    )
end

# ╔═╡ c582ddd0-5d0e-5091-bd69-4e30c9217079
md"""
## The Table to Use

When BasisSimulator replaces its current table, this is the one to use, and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **BasisSimulator's composition at 6.40 g/cm³**, a density such a garnet can have, instead of the
  impossible 7.0 (section 1). The Tb-rich garnet of GE's patent is saved beside it; between the two
  the efficiency moves by at most a few hundredths (section 4).
- **3.0 mm deep.** BasisSimulator documents it, the current table was made at it, and it is
  within the 1.5–3 mm reflector cells of GE's patent (sections 2 and 5).
- **BasisSimulator's fill factor, 0.9 × 0.9, unchanged.** The assumed reflector reproduces it.
  With the 0.15 mm grid walls of GE's patent example it would be about 0.74 of the face
  (section 6), but that is a patent example, not the product, so it is not adopted.

The efficiency for depths from 1.5 to 4.0 mm is saved beside it.
"""

# ╔═╡ 93ff99c5-0fce-54a2-bb23-da7eb0316397
md"""
## Saving the Table

One file in `lib/BasisMonteCarlo/tables/`, in the layout of BasisSimulator's
`GEMSTONE_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of 3.0 mm of
Gemstone at a density a garnet can have (BasisSimulator's composition at 6.40 g/cm³), beside the
one-dexel efficiency, the interaction probability, the Swank factor and DQE(0), the Tb-rich
garnet of GE's patent, and the flat-field efficiency for depths from 1.5 to 4.0 mm, since no
source gives the Apex's depth. The reproduction of the current table is kept with it, as the
record of how that table was made.
"""

# ╔═╡ 7b873d56-4442-5532-9a8c-1d466d90360f
table_depths = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# ╔═╡ 82ad04aa-e3c4-51e2-a29b-5e9290db49e4
# One column per depth, one row per energy
by_depth = reshape(
    MC.simulate_sweep(
        [(dexel_array(depth), E) for depth in table_depths for E in energies];
        n = 50_000,
        beam = crystal_beam,
        seed = 19,
        readout,
    ),
    length(energies),
    length(table_depths),
)

# ╔═╡ 3e1c3140-a3e4-5297-9faa-a403a0c3482d
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ 2ba11eeb-f02c-5028-8c7b-3ec4edc3219d
saved = let
    path = joinpath(table_dir, "ge_revolution_apex_gemstone_efficiency_geant4.jls")
    contents = Dict{String, Any}(
        "energies" => collect(energies),
        "efficiency" => efficiency.flat_field,
        "efficiency_one_dexel" => efficiency.one_dexel,
        "interaction_probability" => efficiency.interaction,
        "swank_factor" => efficiency.swank,
        "dqe0" => efficiency.dqe,
        "depths_mm" => table_depths,
        "efficiency_by_depth" => MC.energy_absorption_efficiency.(by_depth),
        "efficiency_tb_rich" => composition_runs["Tb-rich, 6.24"],
        "efficiency_at_7g_cm3" => composition_runs["even split, 7.0"],
        "reproduced_table" => reproduced,
        "reproduced_depth_mm" => fit.depth_mm,
        "reproduced_scale" => fit.scale,
        "decision" => "use `efficiency`: flat field, BasisSimulator's Gemstone composition at 6.40 g/cm3 (7.0 is impossible for any (Tb,Lu)3Al5O12), 3.0 mm deep; BasisSimulator's fill factor 0.9 x 0.9 unchanged",
        "definition" => "mean energy collected per photon landing on a dexel's garnet with every dexel illuminated (flat field), divided by the photon energy; BasisSimulator's fill factor applies separately",
        "detector" => Dict(
            "scanner" => "GE Revolution Apex (Gemstone Clarity)",
            "material" => "Gemstone, (Tb,Lu)3Al5O12:Ce, BasisSimulator's mass fractions at 6.40 g/cm3",
            "depth_mm" => depth_mm,
            "pitch_mm" => pitch_mm,
            "reflector" => "0.1 mm TiO2-loaded epoxy (assumed)",
            "grid" => "2-D tungsten, 0.15 mm walls, 8 mm tall (GE patent example); not in the flat field",
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 06_ge_revolution_apex.jl",
    )
    Serialization.serialize(path, contents)
    path
end

# ╔═╡ fe7f4f17-2218-5691-8c95-15a0e171c1ce
# Read it back in BasisSimulator's layout
let
    t = Serialization.deserialize(saved)
    lut = (energies = t["energies"], efficiency = t["efficiency"])
    (
        file = basename(saved),
        energies = extrema(lut.energies),
        at_60keV = lut.efficiency[60],
        at_120keV = lut.efficiency[120],
    )
end

# ╔═╡ 1eb7490b-a247-5aff-829e-55401af71db1
md"""
## Where This Leaves the Apex
"""

# ╔═╡ f0fd4be7-bd7e-5437-b754-3370abcdbd27
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    worst_hvl = maximum(abs(c.basis - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        - **The density is impossible.** BasisSimulator's Gemstone is 7.0 g/cm³; any (Tb,Lu)₃Al₅O₁₂ lies between 6.07 and
          6.72. This notebook uses BasisSimulator's composition at 6.40 g/cm³ and brackets the unpublished Tb/Lu split
          with GE's Tb-rich patent garnet; Siemens' own LuTAG curve sides with the lighter garnets.
        - **BasisSimulator's table is its documented detector**, at the impossible density: a pencil beam into one dexel,
          $(fit.depth_mm) mm deep, tallied in that dexel, reproduces it to about 0.01 (one point on the Tb K-edge apart).
        - **What a scan sees** is the flat-field efficiency: each dexel gets back from its neighbours the Tb and Lu
          x rays it loses to them. At 3.0 mm and 6.40 g/cm³ it is $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, against the
          table's $(round(m120.table; digits = 3)) ($(round(100(m120.geant4 / m120.table - 1); digits = 1)) %), with DQE(0) $(round(m120.dqe; digits = 3)).
        - **The grid** (2-D, tungsten, GE's patent dimensions) shades a thin strip of garnet and stops most scatter;
          **the whole detector** collects what one dexel's cell does.
        - **The beam:** BasisSimulator's Apex beam $(worst_hvl <= 0.5 ? "agrees with" : "differs from") the measured Revolution CT half-value layers $(worst_hvl <= 0.5 ? "to within" : "by up to") $(round(worst_hvl; digits = 1)) mm of aluminium.
        - **Saved:** `$(basename(saved))`, with both possible garnets and depths 1.5–4.0 mm, for BasisSimulator.
        """
    )
end

# ╔═╡ Cell order:
# ╟─982eeab8-0545-51f1-b14c-f4984bc5ce21
# ╟─7d76168f-6aaf-5cac-ad93-06b3d605dce4
# ╟─f53ca02e-1359-50fb-ab81-6e8c8c5249a4
# ╟─0d89ac30-a1cd-555c-a60a-934112484f93
# ╟─b5d65082-f8d7-5b19-a30a-117dcd788835
# ╠═c7b8ce23-df15-5445-8f9c-1d533b22b055
# ╠═b9afe761-a41a-56bd-89b3-aa4d1f850542
# ╠═1114b922-2909-5382-ad68-7870b9f17a49
# ╠═3c7da9cf-d44f-5117-a6cd-ab70462594ba
# ╠═d7980232-34d1-599f-89a1-18c144d7468b
# ╠═2e1263bc-6a15-5dad-9da8-0c5ea9d9a1ec
# ╠═6f0a1390-4e16-5007-aff0-205efb24710a
# ╠═6650624a-9201-5026-9af0-2852cb32ff32
# ╠═065c2cd0-3724-5ae0-a368-5c8eb5540a1a
# ╠═0f1f7711-29fa-5149-b284-b47dcdc295c0
# ╠═e3e9c212-4583-5b56-8c63-55a23aceccc6
# ╠═3b5f881c-54d6-5378-ad04-eefcacc3c8ea
# ╠═425bf6f0-bb29-56ec-9883-45afcf680eda
# ╠═4551bbf4-0c6f-558d-8dd3-19061afbeee1
# ╠═d852372b-0f59-5689-a6f2-a96f8756249b
# ╟─9ed4e154-0c35-5252-b2dc-388f22a1c47f
# ╠═0565c39d-9720-5863-a0b6-c3e104a475d1
# ╠═b5c946ee-f16f-5cca-82cf-b5040fe481b8
# ╠═8fba75c0-6a9f-5bbe-9a10-9d52e2296292
# ╠═b349f0be-6494-5c81-80ea-82a481df3b90
# ╟─3e134eba-e90e-5fef-845b-b81b61397dca
# ╠═f88f3daf-cbd3-55ec-9361-63ae45a21b33
# ╠═d6256293-43fa-5626-bba6-6b1c01af6dda
# ╠═2bcabe1a-a3fd-5fa7-8e2a-2a88d9126859
# ╠═7b39f748-54d2-5a67-977a-b81bf47b6b27
# ╟─79bf52e2-7776-5cc1-8d0b-4e264aa8530e
# ╠═d0f1d5ee-5eb0-5fa2-bcd6-74a957ed0358
# ╟─ea752deb-d5f8-530f-a2bc-d8f70ac7f9d6
# ╠═ad4b77f0-e633-5a1f-8474-144d368eb858
# ╠═7fea9a31-343e-5084-b697-c614d31d86cc
# ╠═c6bf2e2b-147e-5c10-a14e-53a1f4421e56
# ╟─7e771ff9-6f1a-5c5b-8469-30e1a7ef0a6e
# ╠═ba7bd8e3-3fe8-51fb-a149-651b7415cecf
# ╟─76d9fe06-65d2-50ec-a57f-9d13c0424438
# ╠═59b35b57-f457-5c5a-89fd-f215ac91ec5f
# ╠═87c01464-7920-5151-aaf1-668655c36c98
# ╠═66875788-7213-5a45-93df-f02a465feab5
# ╠═85332bcb-a653-5577-986e-55557180a527
# ╠═43a2476d-f0b3-5c4e-ad60-a79093af88bc
# ╠═b937f150-98dd-52ad-a1a9-39606e1cc17d
# ╠═69f88251-23f2-5b2f-8e30-f8037e72aabd
# ╠═3cd39c08-6d7d-5852-911d-70b71149b9ca
# ╠═13a363b3-f112-59a0-9192-be6a9bef1026
# ╠═e340add3-b5f6-5242-93ea-81ef5ba335a2
# ╠═0df4a950-2c66-5f98-827e-d647b3db7d5c
# ╟─3a51adf8-1bd6-5027-a06e-b288b5ea7b6e
# ╟─3d4f918b-7df5-5eed-becb-a6dd61ba726c
# ╠═bfafebed-31d8-53f5-a471-25993d2ed385
# ╠═32ca1abd-3897-5936-a735-a52773f47bf5
# ╟─76876cdf-0539-5b54-a239-98f3e02df836
# ╟─fdb32583-deac-5102-80c8-d5c90e587dc2
# ╠═a2a74ae1-42e5-5360-b821-fa1c5835af23
# ╠═d465f476-4b7a-5a26-8485-e167450cf6e3
# ╟─be0eef36-6795-5094-814c-7838d52f5e92
# ╟─fb4ffcd0-bac4-5ce3-858c-c45c2085c83b
# ╠═71ce3fe1-447b-54b5-8ded-a897f1c8636e
# ╠═c1642d2c-c743-50d8-8fa0-b307a93526da
# ╟─590bc8d2-76b3-5e54-9bc8-d8aa7104552e
# ╠═90bbeea9-e955-5350-9c91-055bb731efb2
# ╟─76eecb6b-bcfa-5b64-820a-0ab315fad43f
# ╠═27dc6d6f-6037-5f6a-b642-1c5af6a9bc4d
# ╠═05c86c98-403d-55b7-ae80-f23a4cbbfcd5
# ╠═651673f4-ac81-5aa2-9b6c-4d3a1f232562
# ╟─fdb38e4c-3f74-5de4-be98-cdda136f1a56
# ╠═f192b404-d795-5a67-8bf5-87aa47aeda2e
# ╟─243b6cd3-d4a7-5e9a-a7d0-39a424919164
# ╠═ee6081a3-7bb9-5320-a338-909cf1251027
# ╟─7f4daa38-7a47-5d7f-81c6-2aa821f3715e
# ╠═208a44d2-987f-5887-b584-1f976fcc90a5
# ╟─56b952ce-233b-5076-a0ad-fa6756455914
# ╠═1d5ea56f-89a5-535f-a2ab-dce36fcb20a7
# ╠═8ec58a1a-ddc9-51e5-a0e4-6ced725ac8fe
# ╟─4ace8ef9-4637-5260-9f14-5f7685e76537
# ╠═797de5bb-99b6-5094-a316-d8ca2bfb284f
# ╠═6830b6a0-8ae0-5e43-80e9-da1cc552508c
# ╟─f27dd4ed-4167-5bd4-ac84-a8f98e88754b
# ╠═2ad68ded-ca19-504d-a805-f0caf48d398b
# ╟─23cc9c86-3ad7-5182-a4e2-9857fcf1c6f2
# ╠═2f3c0378-a2dd-5a8a-aac2-8ecba127fdc3
# ╠═1ae1a7b5-befc-57dd-8506-a3904c6e56e2
# ╠═876a93ce-7ca7-5f15-a44d-80a8c18bd676
# ╠═24568e75-698d-5a5e-9401-f553959c2742
# ╠═e41cfdd8-b032-5c29-abef-a5c856dec82f
# ╟─ba31b322-bf2a-57ca-8068-9858e8ac5d00
# ╠═05b60d54-a668-5497-9fc5-dcedf626515e
# ╠═d067a0b4-d46d-5a21-a351-d3c7e347f9b7
# ╟─54179647-5aa9-520a-a0ce-933b5e136cbf
# ╠═25b60608-28d7-5442-89df-fb1e7ac471c8
# ╟─55dad12b-a818-55c3-b5ba-aea9be0cd98c
# ╠═d89950fe-ea70-592b-998b-3e954534f5c9
# ╟─6b89ebe5-d2ca-5d77-93e1-ec26b393c271
# ╟─c599896a-d8dc-5d31-8aa6-d04f7eb6bf7e
# ╠═367261c6-ffb8-5661-970e-6494b0a10197
# ╠═8c9ab3eb-23b7-57a5-9c13-0aa983dae5e0
# ╠═387074f5-5299-5768-b50a-e1a315621ab8
# ╠═173503d6-98e5-51d2-b783-7e5885c408eb
# ╟─8c9677c3-aac1-593c-a53c-ccc5218c3b3e
# ╠═9af6dffc-d99a-54bf-97cf-dd20156bbe48
# ╟─89f83a80-82e6-5d12-8616-36560dba03a1
# ╠═8074b52c-eaab-5eef-ab41-b3482d729940
# ╠═bd408664-f831-56e0-971a-4d7c4f14fa33
# ╠═92105c08-3d1a-533c-8939-3c30460be4b4
# ╟─59e3353e-13f5-53f2-be17-2062c579f044
# ╠═ace5c787-13e0-57f3-93f1-345861f9c001
# ╟─96693605-bc02-50f2-a3ea-da32946c6c79
# ╠═4575a19b-f864-53fc-a050-40b57b16fc8f
# ╠═9f4fa6af-f684-525c-9389-5081654e6d71
# ╠═cf2e67e6-4692-5f99-bd79-a5fbc8fb9abf
# ╠═1bae766b-aa3d-5389-937c-13cb98623d47
# ╟─7f435d0f-3376-5aba-9422-da95aeee9fd4
# ╟─e8fa69d7-5554-59ca-9c0b-ab33f82e52dc
# ╠═a5ab816b-0816-53c4-a84d-40a2992a1de7
# ╟─8eae3d73-e14e-5d93-9420-74e73deb379c
# ╠═5ab9e1a0-492b-54b0-bbdb-f272e3fa8bd1
# ╟─ba5b5a00-0e75-5e2c-9bae-727aec37b499
# ╠═f775068c-2324-5cf4-8811-cd75f968a57d
# ╠═354a335b-59d0-565d-a29c-4a3242eac17c
# ╟─ffeaa78c-e964-5348-9a8d-7e17230dbfc1
# ╠═2077865c-68c8-5b80-8b56-275858a50a24
# ╟─c582ddd0-5d0e-5091-bd69-4e30c9217079
# ╟─93ff99c5-0fce-54a2-bb23-da7eb0316397
# ╠═7b873d56-4442-5532-9a8c-1d466d90360f
# ╠═82ad04aa-e3c4-51e2-a29b-5e9290db49e4
# ╠═3e1c3140-a3e4-5297-9faa-a403a0c3482d
# ╠═2ba11eeb-f02c-5028-8c7b-3ec4edc3219d
# ╠═fe7f4f17-2218-5691-8c95-15a0e171c1ce
# ╟─1eb7490b-a247-5aff-829e-55401af71db1
# ╠═f0fd4be7-bd7e-5437-b754-3370abcdbd27
