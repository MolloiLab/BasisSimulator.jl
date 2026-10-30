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


# ╔═╡ 80569823-8008-530d-a0df-34c4e21e0758
md"""
# The Canon Aquilion ONE Detector in Geant4: GENESIS and PRISM

The Canon (formerly Toshiba) Aquilion ONE is an energy-integrating CT scanner with a 320-row,
16 cm wide detector, wide enough to cover an organ in one rotation. Two current editions are modelled
here, the **Aquilion ONE GENESIS** (TSX-305A) and the **Aquilion ONE PRISM** (TSX-306A). They share
the detector, Canon's **PUREViSION**, and the tube: the PRISM's FDA clearance lists "no change in the
hardware" of the detector. A PUREViSION dexel (detector element) is a block of praseodymium-activated
ceramic with a reflector around it and a photodiode under it.

BasisSimulator has no Canon scanner yet: no geometry, no beam and no detector table. This notebook
builds the detector in Geant4, one piece at a time, and

1. **gathers what is documented** (the FDA clearances, Canon's brochures, the literature) and
   states every assumption where nothing is;
2. **fits the x-ray beam** to measured half-value layers, and checks it on Canon's silver filter;
3. **builds the detector's efficiency table**, with its noise (the Swank factor), and saves it for
   BasisSimulator, once for each edition.
"""

# ╔═╡ 0e0805a1-0452-5360-8e0e-27c7450b5769
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a PUREViSION dexel does not count at all.
Every x ray it absorbs leaves energy in the ceramic, which glows; the reflector keeps the light
inside the dexel, and the photodiode under it adds up the light of every photon over the reading.
The signal is therefore proportional to the **energy** left in the dexel, and the efficiency that
matters is the energy-absorption efficiency: the mean energy a dexel collects per photon, divided
by the photon's energy.
"""

# ╔═╡ 3c17a4f3-92ba-516a-beba-e0b6a903377b
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
            title = "2  Ceramic",
            body = "absorbed in Gd₂O₂S;\nits Gd x ray may\nland next door",
            where = "Geant4",
            color = PALETTE.xray,
        ),
        (
            title = "3  Light",
            body = "the energy left\nbecomes light, kept\nin by the reflector",
            where = "the ceramic glows",
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

# ╔═╡ db0ace6b-e7f9-56bf-850a-db1a6d6e2e06
md"""
## The Aquilion ONE, as Documented

Every hardware number this notebook uses, from the FDA 510(k) summaries, Canon's brochures, Toshiba's
patents and the literature. Where no source gives a number the row says so, and the notebook states
its assumption. BasisSimulator has no Canon values to compare against. Canon publishes even less about
its detector than Philips does.

| quantity | documented | status | source |
|---|---|---|---|
| detector | PUREViSION (since 2014), the same on GENESIS and PRISM ("no change in the hardware") | ✓ | K133497; K192188 |
| scintillator | "praseodymium-activated" ceramic, cut from a solid ingot, with reflective isolation; host material not stated | assumed here: Gd₂O₂S at 7.31 g/cm³ | Canon PUREViSION brochure |
| crystal depth | not published | assumed here: 1.4 mm; 0.8–2.0 mm saved | – |
| dexels | 320 rows × 0.5 mm (16 cm at the isocentre); 896 channels per row | ✓ | Canon brochures; K160587 |
| source–isocentre | 600 mm (original Aquilion ONE and ONE ViSION) | used; not confirmed for GENESIS/PRISM | Geleijns et al. 2009 |
| source–detector | 1072 mm, from a secondary table only | used; not confirmed | – |
| dexel pitch at the detector | not published | calc: 0.5 mm × 1072/600 along z; across the fan, 896 channels over the 50 cm field | – |
| reflector between dexels | "reflective isolation", width not published | assumed: 0.1 mm of TiO₂-loaded epoxy | Canon PUREViSION brochure |
| anti-scatter grid | not published for any edition; Toshiba's patents describe 1-D and 2-D tungsten or molybdenum collimators | assumed here: 1-D tungsten plates, 0.1 mm, 10 mm tall | US 7,526,070; US 9,320,476 |
| tube | CXB-750F on both; 80, 100, 120, 135 kV | ✓ | K160587; K192188 |
| anode angle | 10° (ONE ViSION); not stated for GENESIS/PRISM | 10° used | K122109 |
| filtration | not published | fitted in section 8 | – |
| beam quality (PRISM) | half-value layer 6.0, 7.4, 8.5, 9.2 mm Al at 80, 100, 120, 135 kV; 13.7 mm Al with SilverBeam (0.5 mm silver) at 120 kV | the target of the beam fit, and its check | Ito et al. 2026 |
| bowtie filters | GENESIS: two, unnamed; PRISM: M and L, plus SilverBeam | – | K160587; K213504 |

Sources: FDA [K133497](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133497.pdf),
[K160587](https://www.accessdata.fda.gov/cdrh_docs/pdf16/K160587.pdf),
[K192188](https://www.accessdata.fda.gov/cdrh_docs/pdf19/K192188.pdf),
[K213504](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213504.pdf),
[K122109](https://www.accessdata.fda.gov/cdrh_docs/pdf12/K122109.pdf);
Canon, [GENESIS brochure](https://us.medical.canon/download/ct-br-aq-one-genesis),
[PRISM brochure](https://us.medical.canon/download/ct-br-aquilion-one-prism),
[PUREViSION detector](https://us.medical.canon/products/computed-tomography/pure-vision-detector);
Toshiba patents [US 7,526,070](https://patents.google.com/patent/US7526070B2/en),
[US 9,320,476](https://patents.google.com/patent/US9320476B2/en);
Geleijns et al., [Phys. Med. Biol. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2948862/);
Ito et al., [J. Appl. Clin. Med. Phys. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13486965/);
Siemens, [Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ 33e46fd5-c542-55e8-bcb5-192a0bac3d5e
md"""
## Notebook Setup
"""

# ╔═╡ ab9db025-f507-5388-bf67-4c0d170d0e28
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ bc041287-f4f5-5ce7-8924-9633ea5dc417
using Markdown: @md_str, Markdown

# ╔═╡ 808a4723-438f-586f-b92c-6e49eb9f89e6
import PlutoUI

# ╔═╡ 8b53812d-c201-51bb-8847-1b66b5f1f09c
import BasisMonteCarlo as MC

# ╔═╡ 429139dd-55bf-52bd-a9d5-9691d14d06a2
import CairoMakie as Mke

# ╔═╡ 1a29df66-ff12-5ff0-acdd-9c0e6b2b13d6
import BasisSimulator as BS

# ╔═╡ 726ef0af-666e-5b87-9ea4-02801135cab1
import Serialization

# ╔═╡ 3accdd63-8b76-5d41-a238-01a65ba33416
import DelimitedFiles

# ╔═╡ 10fb07d1-38ee-5894-b238-df1f1e70be84
using Statistics: mean

# ╔═╡ 558bad13-644e-5e3d-be50-5c5850d10928
PlutoUI.TableOfContents()

# ╔═╡ ed268a94-6b31-5c94-87d3-d11efbaeb1a0
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ c9b96524-0916-5e0a-bd06-366cce225226
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

# ╔═╡ 57dd0152-ffba-53a8-a5fc-33d79b6ce3ac
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

# ╔═╡ 1b1c6cc3-ced3-5a5c-affb-f6456a4012f2
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

# ╔═╡ bfe11ba8-730e-58f7-9f66-212ffb916313
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

# ╔═╡ ddadd9bd-a2b7-58f8-a970-3bff3a89201e
md"""
## 1. The Scintillator

Canon describes the PUREViSION scintillator only as a "praseodymium-activated" ceramic, cut from a
solid ingot. Praseodymium is the activator of the gadolinium oxysulfide ceramics used across CT,
Gd₂O₂S:Pr, so this notebook takes the PUREViSION ceramic to be Gd₂O₂S at **7.31 g/cm³**, the density
Siemens measures for its own Gd₂O₂S ceramic. That is an assumption: Canon publishes neither the host
material nor the density.
"""

# ╔═╡ 54cfabcd-ec9e-5930-af26-fe46ae2f6df2
# Gd₂O₂S ceramic at 7.31 g/cm³ (assumed: Canon publishes neither the host material nor the density)
gos = MC.DetectorMaterial("GOS ceramic"; density = 7.31, formula = "Gd2O2S")

# ╔═╡ bfd257ef-c874-5d3e-865b-91eca06e30f1
# Gadolinium's K-shell: its edge and the energy of the x ray it emits (Kα)
gadolinium = only(MC.k_edges(gos))

# ╔═╡ ce933054-651a-5933-9566-efb31407e7cd
md"""
Gadolinium's **K-edge, 50.24 keV**, sits in the middle of every CT spectrum. Below it a photon
cannot free a K-shell electron; just above it, it can, and absorption jumps sevenfold. Every
such absorption leaves the atom to emit, nine times in ten, a **Gd K x ray of about 43 keV**,
which is just below the edge and so is absorbed only weakly: it can travel a millimetre or more
before stopping, which is the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ f2132f86-2dcb-5010-bb46-b3216f321502
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

# ╔═╡ f7c3b5dc-b20b-5db8-bbc9-8b7b9ad389bb
mu_energies = 10.0:0.25:150.0

# ╔═╡ 4e886e38-891b-5244-b3f7-f4217cb5785c
# Geant4's attenuation of the ceramic, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 919250d4-f19e-5252-bad8-a0191e7f3dd9
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 94ece8ce-3e4d-53b1-8769-420e883094e7
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How strongly Gd₂O₂S ceramic stops x rays (7.31 g/cm³)",
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
    Mke.vlines!(
        ax,
        gadolinium.edge_keV;
        color = (PALETTE.xray, 0.8),
        linestyle = :dot,
        linewidth = 2,
    )
    note!(
        ax,
        gadolinium.edge_keV,
        0.4,
        "Gd K-edge, $(round(gadolinium.edge_keV; digits = 2)) keV";
        offset = (8, 0),
        color = PALETTE.xray,
    )
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

# ╔═╡ 6926962b-931d-56b2-be23-590de184f48e
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

# ╔═╡ fe6f7616-c5c3-5864-b959-5fd0d16bb6be
md"""
### A First Check: A Published Gd₂O₂S Curve

Canon publishes no absorption curve, but Siemens publishes the fraction of x rays that **1.4 mm
of its own Gd₂O₂S ceramic** absorbs (*Detectors & Crystals*, p. 19). It is the probability that a
photon interacts at all, `1 − exp(−μd)`, so it checks the material model and the density assumed
here. The values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ 4a1a5d21-5581-50d2-b02a-9174b8bfcd02
# Siemens' curve, read from the vector figure (reference_data/siemens_ufc_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_ufc_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ 3e588088-f4e1-5470-a7d8-f02f18437d48
# The crystal depth (assumed: not published)
depth_mm = 1.4

# ╔═╡ 17fe60de-9741-5ec5-9555-7afb4820e74b
# The probability a photon interacts in 1.4 mm of the ceramic
interaction_curve = [1 - exp(-MC.geant4_attenuation(gos, E) * 1.4) for E in mu_energies]

# ╔═╡ 3d6ad596-a1da-5c24-b39a-8cdc0a1314da
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The fraction of photons 1.4 mm of Gd₂O₂S ceramic absorbs",
        xlabel = "photon energy (keV)",
        ylabel = "fraction of photons absorbed",
        xticks = 20:20:150,
        limits = ((15, 152), (0.45, 1.04)),
    )
    Mke.lines!(
        ax,
        mu_energies,
        interaction_curve;
        color = PALETTE.photon,
        linewidth = 3.5,
        label = "Geant4, Gd₂O₂S at 7.31 g/cm³",
    )
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
        label = "Siemens' figure for its Gd₂O₂S ceramic (± 0.025)",
    )
    note!(
        ax,
        gadolinium.edge_keV,
        0.88,
        "Gd K-edge:\nabsorption jumps";
        align = (:right, :center),
        offset = (-12, 0),
        color = PALETTE.xray,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 157eec79-393b-5b59-bb8a-950a2c4f5802
let
    ours = [MC.interpolate_linear(collect(mu_energies), interaction_curve, E) for E in siemens_curve.energy_keV]
    low = siemens_curve.energy_keV .<= 90
    Markdown.parse(
        """
        Up to 90 keV the published curve and Geant4 for 1.4 mm of Gd₂O₂S at 7.31 g/cm³ agree to within
        **$(round(maximum(abs, (siemens_curve.absorption .- ours)[low]); digits = 3))**, inside the figure's reading accuracy: the material model is sound.
        Whether the PUREViSION ceramic is this material at this density, nothing published says.
        """
    )
end

# ╔═╡ fcee931d-a949-5b8c-9bee-d59e556f0fc1
md"""
## 2. One Dexel and Its Neighbours

A dexel is one block of ceramic with reflector on its four sides. Its pitch at the detector face is
not published, so it is worked out from what is: 320 rows of 0.5 mm at the isocentre, magnified by
1072 / 600 along z; and 896 channels across the 50 cm scan field, on an arc of 1072 mm across the
fan. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy is assumed. Neither is the
crystal's depth; 1.4 mm is assumed, and the table is saved for other depths too.
"""

# ╔═╡ e01b99a8-d89e-53cd-9b56-b4dcc5c374db
# The Aquilion ONE's geometry: 600 mm (Geleijns et al. 2009) and 1072 mm (secondary source)
aquilion = (
    source_to_isocenter_mm = 600.0,
    source_to_detector_mm = 1072.0,
    columns = 896,
    rows = 320,
    row_at_isocenter_mm = 0.5,
    scan_field_mm = 500.0,
)

# ╔═╡ 2594a24f-dc08-533c-8b9d-2f99108aab4e
# The dexel pitch at the detector face (across the fan, along z), in mm: the 50 cm scan field's fan
# angle shared among the channels, and the row width magnified to the detector
pitch_mm = let
    fan = 2 * asin(aquilion.scan_field_mm / 2 / aquilion.source_to_isocenter_mm)
    (
        aquilion.source_to_detector_mm * fan / aquilion.columns,
        aquilion.row_at_isocenter_mm * aquilion.source_to_detector_mm / aquilion.source_to_isocenter_mm,
    )
end

# ╔═╡ 219cb299-5acc-5911-9eaf-2e86161e449d
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ 92ec3c2f-5d09-56f1-95fd-e5d2d1450ff2
# The block of ceramic each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ f266e1ff-8e7c-52ae-acfb-8fc1c6f2ca9f
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 498b5b6a-a948-5126-92a3-364216f7b20e
let
    fill = crystal_mm ./ pitch_mm
    Markdown.parse(
        """
        The dexel pitch at the face is **$(round(pitch_mm[1]; digits = 3)) × $(round(pitch_mm[2]; digits = 3)) mm**, and the ceramic block inside it
        $(round(crystal_mm[1]; digits = 3)) × $(round(crystal_mm[2]; digits = 3)) mm: the ceramic covers $(round(fill[1]; digits = 3)) × $(round(fill[2]; digits = 3)) of the face, the fill factor a
        BasisSimulator model of this scanner would use.
        """
    )
end

# ╔═╡ 85053c4d-c787-591e-868e-b0548f93a32f
# The Geant4 detector: a slab of ceramic `depth` deep, cut into 13 × 13 dexels by reflector
# septa (ample for the x rays that leave a dexel), optionally under an anti-scatter grid
function dexel_array(depth; material = gos, grid = nothing)
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

# ╔═╡ f18449da-5fb9-5dd1-8889-b6f1fd902f59
# Photons landing uniformly on one dexel's ceramic, and on its whole cell (ceramic and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ 076c1cfb-65a3-5d2e-b5b0-1d264e756fe8
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 77ad16fd-5153-5a4b-801a-57c22ebbd8b7
# Each dexel collects the energy left in its own block of ceramic (the reflector keeps the light
# in); the dexel the photons land on and its eight neighbours are recorded
readout = MC.DexelEnergyReadout(
    MC.DexelReadout(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        neighborhood = 3,
        cloud_sigma_um = 0.01,
    ),
)

# ╔═╡ 5c906809-30a7-5e8f-b82e-78981da811f3
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1400, 640))

    # (a) The 3 × 3 blocks of ceramic, to scale, with the beam arriving from above
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

    # (b) Seen from the beam: blocks of ceramic and the reflector between them
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

    # (c) Side view: a photon absorbed, and its Gd x ray heading for the next dexel
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
        "Gd x ray";
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

# ╔═╡ 4121046c-46f1-52bc-b36f-5e41408abb40
md"""
### What a Photon Does in the Ceramic

`record_tracks` follows a few hundred 60 keV photons landing on the central dexel, every step of
every particle. Two of them are drawn: one whose Gd x ray is absorbed **in the next dexel**, so
its energy is split between two photodiodes, and one whose Gd x ray **leaves through the front
face**, so its 43 keV are lost to every dexel. Both happen often, because the Gd x ray is so
weakly absorbed.
"""

# ╔═╡ 4ee27e1b-2c0f-5ba4-baaa-6e883057b012
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    60.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ ac571320-09f8-5d9b-843c-50d2bef6fc49
# Two photons to follow: one whose Gd x ray stops in the next dexel across the fan (in the same
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

# ╔═╡ e71c343e-4038-52da-8770-c462750ba0bc
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    top = -0.9
    fig = Mke.Figure(size = (1300, 520))

    # Clip a step to the drawing (above the ceramic it may run off to the world's edge)
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
    axa = Mke.Axis(fig[1, 1]; title = "(a) The Gd x ray stops in the next dexel", common...)
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Gd x ray leaves through the front",
        common...,
    )
    panel!(axa, followed.next_door)
    panel!(axb, followed.front)
    Mke.Label(
        fig[2, 1:2],
        "blue: the 60 keV photon     orange: its Gd x ray     red dots: where they are absorbed     green: the energy each dexel keeps";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 2baedffc-48d3-58a5-9816-c781f3598720
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is not one number but a
spread: the full photon energy when everything stays in the dexel, the energy minus 43 keV when
the Gd x ray leaves, a little when the photon only Compton-scatters once and leaves. Three
energies show the three regimes: 45 keV (below the K-edge), 60 keV (just above it) and 120 keV.
"""

# ╔═╡ 767031ad-4240-58a7-8784-73ed6193049b
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
        for E in (45.0, 60.0, 120.0)
)

# ╔═╡ ac0ae21f-6ba3-5b81-b801-b226d34a8645
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ 3734b13a-da38-504d-bb4a-ef79fccc86b9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 520))
    kα = gadolinium.kalpha_keV
    for (k, E) in enumerate((45.0, 60.0, 120.0))
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
        if E > gadolinium.edge_keV
            note!(
                ax,
                E - kα,
                0.3,
                "Gd x ray lost:\nE − $(round(kα; digits = 1)) keV";
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
                0.02,
                "Compton scatters\nthat left";
                align = (:center, :center),
                color = PALETTE.photon,
                fontsize = 14,
            )
        end
    end
    fig
end

# ╔═╡ 4456dbc6-d2ac-5b8b-b744-cd7109af51ea
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the ceramic or
leaves it: as the Gd x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of ceramic (no reflector), so that whatever
leaves the ceramic leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ b0e5a025-3a3b-5d9d-8ae0-d597c6507d3d
budget_energies = 15.0:5.0:140.0

# ╔═╡ d21e6491-d12e-52ae-b383-bd12d671b4e9
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(gos, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ c012a993-2845-5b07-be0a-f019bf4fea80
let
    Mke.set_theme!(THEME)
    E = collect(budget_energies)
    parts = (
        (
            values = [b.absorbed for b in budgets],
            label = "stays in the ceramic (the signal)",
            color = PALETTE.kept,
        ),
        (
            values = [b.fluorescence for b in budgets],
            label = "leaves as a Gd x ray",
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
        title = "Where the photon's energy goes: $(depth_mm) mm of the ceramic",
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
        gadolinium.edge_keV;
        color = PALETTE.ink,
        linestyle = :dot,
        linewidth = 1.5,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ b2e83820-62f2-50f2-9aea-a57e382bfc14
let
    k = findfirst(>(gadolinium.edge_keV), collect(budget_energies))
    b = budgets[k]
    closure = maximum(abs(b.total - 1) for b in budgets)
    Markdown.parse(
        """
        Just above the K-edge, at $(Int(budget_energies[k])) keV, **$(round(100b.fluorescence; digits = 1)) %** of the energy leaves as
        Gd x rays and $(round(100b.absorbed; digits = 1)) % stays. The energy balance closes to within
        $(round(100closure; digits = 2)) % at every energy (the rest is binding energy below Geant4's tracking cuts).
        """
    )
end

# ╔═╡ cf122acc-fc8b-5d8d-ad7d-84a701424756
md"""
## 4. The Efficiency and Noise of One Dexel

The full simulation: at every whole keV from 1 to 140 (the grid of BasisSimulator's table),
100 000 photons land uniformly on one dexel's ceramic, and each run is reduced, where it ran, to
the energy every dexel of the 3 × 3 collected and the energy left anywhere in the ceramic. From
them come four curves:

- **interaction probability**: the photon interacts at all (`1 − exp(−μd)`, the check);
- **one-dexel efficiency**: the energy the dexel the photon landed on keeps, ÷ E;
- **flat-field efficiency**: the energy left anywhere in the ceramic, ÷ E. In a scan every dexel
  is illuminated, and what one dexel loses to its neighbours it gets back from them, so this is
  what each dexel collects per photon landing on it: **the efficiency a CT detector has**;
- **Swank factor** `I = M₁² / (M₀ M₂)` of the flat-field signal: the noise the spread of
  collected energies adds. A detector's zero-frequency DQE is `M₁² / M₂` per incident photon, the
  interaction probability times `I`.
"""

# ╔═╡ e4c398e8-b01d-565f-81ed-8a19750e5e33
energies = 1.0:1.0:140.0

# ╔═╡ 98a5f0ad-bbe1-5aab-8dc8-4e2aecbe102c
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 76f1fe13-108c-5f56-aeb2-5b626860523d
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ 89682fb9-9156-5081-9815-65ae00e73940
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1400, 600))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Efficiency, $(depth_mm) mm of the ceramic",
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
        gadolinium.edge_keV,
        0.75,
        "Gd K-edge";
        align = (:right, :center),
        offset = (-8, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.vlines!(axa, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
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
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :lb)
    fig
end

# ╔═╡ ff5bcd43-c401-569f-bb51-3b3163ebcde3
let
    at(E) = findfirst(==(E), collect(energies))
    k60, k140 = at(60.0), at(140.0)
    beer = 1 - exp(-MC.geant4_attenuation(gos, 140.0) * depth_mm)
    Markdown.parse(
        """
        At 60 keV, just above the K-edge, the dexel a photon lands on keeps **$(round(efficiency.one_dexel[k60]; digits = 3))**
        of its energy; with every dexel illuminated each collects **$(round(efficiency.flat_field[k60]; digits = 3))**: the Gd
        x rays a dexel loses to its neighbours, it gets back from them. The Swank factor there is
        $(round(efficiency.swank[k60]; digits = 3)): the lost x rays also make the signal noisier. At 140 keV the photon interacts
        with probability $(round(efficiency.interaction[k140]; digits = 3)) (Beer–Lambert: $(round(beer; digits = 3))), and the flat-field
        efficiency is $(round(efficiency.flat_field[k140]; digits = 3)).
        """
    )
end

# ╔═╡ b67523ad-a398-5025-a89e-31bbc644a886
md"""
### A Check Against Published Monte Carlo

Boone et al. (Med. Phys. 1999) followed x-ray fluorescence in thin Gd₂O₂S screens with the TART 98
code: a pencil beam at normal incidence on a screen of Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S, the binder not counted), the energy tallied in a circle 0.1 mm across the
beam ("absorbed as primary") and in rings out to 4.5 mm ("lateral"), and ideal sensors in front of
the screen and behind it. At 55 keV they report (p. 908) how the incident energy divides. The same
screen and the same tallies in Geant4:
"""

# ╔═╡ b807babb-788e-50b5-9857-ded2f7dfff09
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

# ╔═╡ 8a97e96d-38c0-52f0-a4b4-9c835b6413c8
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

# ╔═╡ f4b79fc1-aa23-56f6-8bc1-eb2b3db7b47e
md"""
## 5. Compared with BasisSimulator's Nearest Table

BasisSimulator has no Canon detector. The nearest table it has is its SOMATOM Definition Flash
table, `UFC_FLASH_MC_EFFICIENCY_LUT`: Gd₂O₂S, and, judging by its shape, made as a pencil beam into
the centre of one dexel 1.4 mm deep at 7.44 g/cm³, tallied in that dexel. It is what a Canon model
would use today if it borrowed a Siemens table. How it compares with the PUREViSION's flat field:
"""

# ╔═╡ fdbc6f9a-497d-588a-b8b5-6d7bd3b5d4e7
# BasisSimulator's nearest table: the Definition Flash's Gd₂O₂S (MCNP, Hamidreza Khodajou-Chokami)
# BasisSimulator v0.15.1's table, as BasisMonteCarlo keeps it for comparison
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash

# ╔═╡ 65264960-57e5-5281-b471-39fa9e34f538
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ dac0f49a-6dcf-5351-843b-5bf3d2d603df
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The nearest BasisSimulator table against the PUREViSION at 1.4 mm",
        xlabel = "photon energy (keV)",
        ylabel = "efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 3.5,
        label = "BasisSimulator's Flash table (MCNP, one dexel, pencil beam)",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, 1.4 mm, flat field",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 1.4 mm, one dexel",
    )
    Mke.vlines!(ax, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 89ee2f6a-8d3d-5837-a72b-b16ac6dd781c
let
    at(E) = findfirst(==(E), collect(energies))
    rows = [
        "| $(Int(E)) keV | $(round(table_at(E); digits = 3)) | $(round(efficiency.flat_field[at(E)]; digits = 3)) |"
            for E in (40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
    ]
    Markdown.parse(
        """
        | photons of | BasisSimulator's Flash table | Geant4, PUREViSION flat field |
        |---|---|---|
        $(join(rows, "\n"))

        Just above the K-edge the flat field is higher (each dexel gets back the Gd x rays its neighbours lose); at high
        energy it is lower (7.31 g/cm³ rather than 7.44, and photons landing near a dexel's edge lose more). Borrowing
        the Siemens table would misplace the Canon detector's efficiency by several hundredths at the energies a
        scan uses most.
        """
    )
end

# ╔═╡ eab56392-a661-5fc6-8f95-068d32573a2b
md"""
## 6. The Anti-Scatter Grid

Canon publishes nothing about the Aquilion ONE's anti-scatter grid; Toshiba's patents describe both
one-dimensional collimators (plates between the channels) and two-dimensional ones, of tungsten or
molybdenum. Canon does not advertise a two-dimensional grid, as Siemens, GE and Philips do for
theirs, so a **one-dimensional** grid is assumed: **tungsten plates 0.1 mm thick and 10 mm tall**, one
per channel, standing on the reflector. The grid does not enter the efficiency table; it matters for
scatter.
"""

# ╔═╡ f2759008-f052-5b85-92fb-04c23dfa5842
grid = MC.Collimator(;
    height_mm = 10.0,
    wall_mm = 0.1,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :x,
    cells = 13,
)

# ╔═╡ 9dbcae9a-cb06-5a25-a722-9fb39b6e7114
let
    Mke.set_theme!(THEME)
    p, s, d, H, w = pitch_mm, septum_mm, depth_mm, grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1400, 700))

    # (a) From the beam: tungsten walls over the reflector, over the ceramic
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
    end
    note!(
        axa,
        0,
        0,
        "ceramic";
        align = (:center, :center),
        color = PALETTE.slab_edge,
        fontsize = 15,
    )
    note!(
        axa,
        -1.45p[1],
        1.45p[2],
        "black: tungsten plates, $(round(Int, 1000w)) µm\nbeige: reflector, $(round(Int, 1000s)) µm\n(no plates between rows)";
        align = (:left, :top),
        fontsize = 14,
        color = PALETTE.ink,
    )

    # (b) Side view, to scale: a photon from the focal spot passes, a scattered one is stopped
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Side view, to scale",
        xlabel = "x, across the fan (mm)",
        ylabel = "height above the ceramic (mm)",
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

    # From the focal spot: straight down the middle of a channel, into the ceramic
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
        "blue: from the focal spot,\npasses into the ceramic";
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

# ╔═╡ 66a50ae8-d09e-57d8-b69c-664f6b362651
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (ceramic and reflector), with and without the grid: the grid
should cost almost nothing, apart from its own tungsten x rays (59 and 67 keV) reaching the
ceramic above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ da786503-a2d6-5880-9361-079b494dabcc
grid_energies = 20.0:5.0:140.0

# ╔═╡ 9ad0c644-b539-5e83-a22e-35fc92391228
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

# ╔═╡ f1873392-5be8-5055-9c30-fdbe75ee33ab
angles_deg = 0.0:1.0:20.0

# ╔═╡ a6d94647-e210-576c-9539-a0ae4159ad3f
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

# ╔═╡ 3aea3630-70c7-5d4f-ab87-489a1320cce4
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
        label = "with the 1-D grid",
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
        label = "with the 1-D grid",
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

# ╔═╡ d8ad9d29-97f4-5325-8ec6-68943312f265
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) % (the walls stand on the reflector; above 69.5 keV the walls' own x rays add a little).
        At an angle across the fan it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %. A
        photon that scatters in the patient arrives from anywhere, so most of it is stopped.
        """
    )
end

# ╔═╡ e259dd7e-12c9-56b0-b77b-93c23dd38226
md"""
## 7. The Whole Detector

The whole detector: 896 channels × 320 rows on an arc of radius 1072 mm around the focal spot,
160 mm along z at the isocentre, every dexel a block of ceramic in reflector, under the plates of the
1-D grid, focused on the focal spot across the fan. An air scan fires photons uniformly from the
focal spot over the whole detector, and every dexel's signal is summed. Two questions: is every dexel
the same, and does the whole detector collect what the single dexel of sections 4 and 6 predicts?
"""

# ╔═╡ 0ffbdeda-bc29-5233-9b92-7f5d1f13573b
detector = MC.ArcDetector(;
    columns = aquilion.columns,
    rows = aquilion.rows,
    radius_mm = aquilion.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :across,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = gos,
)

# ╔═╡ 1c62ec1e-8fd3-52a8-b84f-5610a8257098
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ 53b69f21-a0cd-5ce4-bede-baa7b0519223
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

# ╔═╡ 23ab1779-eac6-5e29-af6d-d465174c5654
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
    R = aquilion.source_to_detector_mm
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
        [aquilion.source_to_isocenter_mm];
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
        aquilion.source_to_isocenter_mm,
        "isocentre";
        align = (:left, :center),
        offset = (12, 0),
        fontsize = 15,
    )
    note!(
        axa,
        0,
        R,
        "$(aquilion.columns) × $(aquilion.rows) dexels, $(round(MC.fan_angle_deg(detector); digits = 1))° fan";
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
        limits = ((1, aquilion.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:aquilion.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / aquilion.columns
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
        aquilion.columns / 2,
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
        limits = ((1, aquilion.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:aquilion.rows, max(1, aquilion.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / aquilion.rows
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

# ╔═╡ d887e3ee-97ca-53ea-8ecd-ee1a7b5b6ea0
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
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / aquilion.source_to_detector_mm))
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

# ╔═╡ 2ab149ac-24be-56fc-9c18-fc41227b7113
md"""
## 8. The Aquilion ONE's X-Ray Beam

Canon publishes no filtration, so the beam is fitted. The spectra are BasisSimulator's IPEM Report 78
tungsten spectra at a 10° anode (the ONE ViSION's; not stated for GENESIS or PRISM), through a flat
aluminium filter whose thickness is chosen so that the half-value layer at the isocentre, through
the thinnest part of BasisSimulator's large-body bowtie (2.5 mm of aluminium), matches Ito et al.'s
measurements on an Aquilion ONE PRISM: **6.0, 7.4 and 8.5 mm of aluminium at 80, 100 and 120 kV**
(BasisSimulator has no 135 kV spectrum; the measured 9.2 mm there is left out). The PRISM's
**SilverBeam** filter, 0.5 mm of silver at 120 kV, measured at **13.7 mm** of aluminium, then checks
the fitted beam independently.
"""

# ╔═╡ 10c68b00-24d3-58cd-9f5b-2b25c917fb5f
# Measured half-value layers at the isocentre, Aquilion ONE PRISM (Ito et al., J. Appl. Clin. Med. Phys. 2026)
measured_hvl = [(kVp = 80, hvl = 6.0), (kVp = 100, hvl = 7.4), (kVp = 120, hvl = 8.5)]

# ╔═╡ 28f3d388-5432-5c8b-83dd-17ff0ca8fdd6
aluminium = MC.DetectorMaterial(:Al)

# ╔═╡ 5ae61127-e981-5614-b7cc-49a51e5fa77e
# The large-body bowtie at its thinnest, the ray through the isocentre: aluminium only, in mm
bowtie_al_mm = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
end

# ╔═╡ 77c888f8-546b-55bd-8e6d-76a9c2068b16
"""
The photons per mAs per mm² reaching the detector, per bin, for an Aquilion ONE protocol at `kVp`, with
`flat_mm` of aluminium as the beam's filtration and a 10° anode.
"""
function aquilion_spectrum(kVp; flat_mm)
    scanner = BS.EICTScanner(
        source_to_isocenter = aquilion.source_to_isocenter_mm,
        source_to_detector = aquilion.source_to_detector_mm,
        detector_rows = aquilion.rows,
        detector_cols = aquilion.columns,
        detector_row_size = aquilion.row_at_isocenter_mm,
        detector_col_size = pitch_mm[1] * aquilion.source_to_isocenter_mm / aquilion.source_to_detector_mm,
        detector_shape = :arc,
        target_angle = 10.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = flat_mm,
        bowtie_filter = :large_body,
        detector_material = :ufc,
        detector_depth = depth_mm,
    )
    protocol = BS.CTProtocol(;
        kVp,
        mA = 200.0,
        views = 1200,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        anode_angle = 10,
    )
    return BS.resolve_source_spectrum_without_bowtie(BS.SimOptions(), protocol; scanner)
end

# ╔═╡ 6002b833-7de7-5e06-a753-277aab945c0e
# The half-value layer at the isocentre for `flat_mm` of filtration, through the bowtie's centre
function hvl_at(kVp, flat_mm)
    e, w = aquilion_spectrum(kVp; flat_mm)
    μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
    return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
end

# ╔═╡ 36ead4eb-6aa3-5a38-82f0-7427bcd7c338
# The aluminium filtration whose half-value layers best match the measured ones (least squares,
# golden-section search over 0–20 mm)
filtration = let
    misfit(t) = sum(abs2, [hvl_at(m.kVp, t) - m.hvl for m in measured_hvl])
    a, b = 0.0, 20.0
    φ = (sqrt(5) - 1) / 2
    for _ in 1:40
        c = b - φ * (b - a)
        d = a + φ * (b - a)
        if misfit(c) < misfit(d)
            b = d
        else
            a = c
        end
    end
    (a + b) / 2
end

# ╔═╡ be6ef043-75d4-59cf-b172-491e2200a90a
hvl_check = [(kVp = m.kVp, fitted = hvl_at(m.kVp, filtration), measured = m.hvl) for m in measured_hvl]

# ╔═╡ 81e9027d-a62a-5a0c-a091-f587d7c0a121
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-value layer at the isocentre: the fitted beam against measurement",
        xlabel = "tube voltage",
        ylabel = "half-value layer (mm of aluminium)",
        xticks = (1:length(hvl_check), ["$(c.kVp) kV" for c in hvl_check]),
        limits = ((0.4, length(hvl_check) + 0.6), (0, 12)),
    )
    x = 1:length(hvl_check)
    Mke.barplot!(
        ax,
        x .- 0.18,
        [c.fitted for c in hvl_check];
        width = 0.34,
        color = PALETTE.photon,
        label = "fitted beam: $(round(filtration; digits = 1)) mm Al, 10° anode",
    )
    Mke.barplot!(
        ax,
        x .+ 0.18,
        [c.measured for c in hvl_check];
        width = 0.34,
        color = PALETTE.kept,
        label = "measured (Ito et al. 2026)",
    )
    for (k, c) in enumerate(hvl_check)
        for (dx, v) in ((-0.18, c.fitted), (0.18, c.measured))
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

# ╔═╡ 0f0250c8-45d8-56d7-b1e5-f15bcf7a6714
let
    worst = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        **$(round(filtration; digits = 1)) mm of aluminium** at a 10° anode reproduces the measured half-value layers to within
        **$(round(worst; digits = 2)) mm** at every tube voltage: one filtration fits all three, which is what a real tube's fixed filtration
        does. This fitted beam is the Aquilion ONE beam for the rest of the notebook and is saved with the table. (The fit
        lumps the unstated titanium filter, the tube's own filtration and the anode angle into one aluminium thickness;
        the bowtie the measurement used is not stated.)
        """
    )
end

# ╔═╡ 5c7fff9b-6630-5e79-a499-efc14c14e3bc
md"""
**An independent check: SilverBeam.** The PRISM's SilverBeam adds 0.5 mm of silver to the 120 kV
beam, for low-dose lung screening; Ito et al. measured its half-value layer at 13.7 mm of aluminium.
The fit never saw this number, so the fitted beam through 0.5 mm of silver should land on it if the
fit captured the real spectrum, not just three numbers.
"""

# ╔═╡ 6b66d29d-5dab-579c-a5d8-1233dc4f35b9
silverbeam = let
    silver = MC.DetectorMaterial("Ag"; density = 10.49, formula = "Ag")
    e, w = aquilion_spectrum(120; flat_mm = filtration)
    μ_al = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
    μ_ag = [MC.linear_attenuation(silver, max(E, 1.0)) for E in e]
    filtered = w .* exp.(-μ_ag .* 0.5)
    (
        spectrum = (e, filtered),
        hvl = MC.half_value_layer(e, filtered .* exp.(-μ_al .* bowtie_al_mm)),
        measured = 13.7,
    )
end

# ╔═╡ 1703f60a-9d85-5a57-962d-91c4f8b0bece
Markdown.parse(
    """
    Through 0.5 mm of silver the fitted 120 kV beam has a half-value layer of **$(round(silverbeam.hvl; digits = 1)) mm** of aluminium,
    against the measured 13.7 mm: $(abs(silverbeam.hvl - 13.7) <= 0.7 ? "the fitted beam passes a check it was not fitted to." : "**⚠ the fitted beam misses a check it was not fitted to**; one aluminium thickness does not capture the spectrum's shape well enough for a filter this strong.")
    """
)

# ╔═╡ d10caf51-f7a2-5154-8d24-e1ed46c4d86c
md"""
The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ ff9e2424-574b-5274-9c8f-adbad84450d6
@bind kvp PlutoUI.Slider([80, 100, 120]; default = 120, show_value = true)

# ╔═╡ 45043ecb-063f-5b53-bd1f-5b559c97199a
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

# ╔═╡ b8e5ce21-e91b-5d3a-a6eb-942d1fb42af8
let
    Mke.set_theme!(THEME)
    spectrum = aquilion_spectrum(kvp; flat_mm = filtration)
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
        label = "BasisSimulator's Flash table",
    )
    Mke.lines!(
        axb,
        s.energy_keV,
        s.signal;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, PUREViSION flat field",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ 5ca60f2c-a230-5894-a8fa-1a5dcc16b64c
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. The Aquilion ONE's tube voltages BasisSimulator can make,
through the fitted beam, and the PRISM's SilverBeam:
"""

# ╔═╡ 2e1b5a19-af32-5b42-8f00-e79cbdc0fad1
scan_protocols = vcat(
    [(label = "$(k) kV", spectrum = aquilion_spectrum(k; flat_mm = filtration)) for k in (80, 100, 120)],
    [(label = "120 kV SilverBeam", spectrum = silverbeam.spectrum)],
)

# ╔═╡ b247d175-c3ff-5561-98c7-bbc089fbf17a
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

# ╔═╡ baaa15f4-add3-5d99-9695-3d6721e6d6f0
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
        label = "BasisSimulator's Flash table",
    )
    Mke.scatterlines!(
        axa,
        x,
        [m.geant4 for m in scan_metrics];
        color = PALETTE.kept,
        markersize = 13,
        linewidth = 2.5,
        label = "Geant4, 1.4 mm, flat field",
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

# ╔═╡ 47436151-80ce-52d6-856e-b75b78deee48
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | BasisSimulator's Flash table | Geant4, PUREViSION 1.4 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's ceramic; BasisSimulator's fill factor (0.9 × 0.9) applies to both
        alike.
        """
    )
end

# ╔═╡ c48615a2-707b-5ccc-b824-acf128726517
md"""
## The Table to Use

When BasisSimulator gains an Aquilion ONE, this is the table to use, and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **Gd₂O₂S ceramic at 7.31 g/cm³, 1.4 mm deep** (assumed). Canon publishes only "praseodymium-
  activated ceramic"; 1.4 mm is the one Gd₂O₂S CT thickness a vendor (Siemens) quotes. Every depth from
  0.8 to 2.0 mm is saved beside it.
- **The fill factor of the assumed reflector** (section 2); the 1-D grid's plates stand on the
  reflector and take nothing more (section 6).
- **The fitted beam**: BasisSimulator's IPEM spectra at a 10° anode through the fitted aluminium,
  which reproduces the measured half-value layers and is checked on SilverBeam (section 8).

The GENESIS and the PRISM share the detector and the tube, so both files hold the same efficiency and
beam; the PRISM's adds SilverBeam.
"""

# ╔═╡ 8b77445b-bc28-5204-89ef-89814e715f0d
md"""
## Saving the Tables

Two files in `lib/BasisMonteCarlo/tables/`, one per edition, in the layout of BasisSimulator's
`*_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of the PUREViSION as
modelled here, beside the one-dexel efficiency, the interaction probability, the Swank factor,
DQE(0), the efficiency for depths from 0.8 to 2.0 mm, and the fitted beam.
"""

# ╔═╡ 14c7a701-5529-58e5-bd40-1d6d46907e66
table_depths = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# ╔═╡ fd3f9361-3413-50e2-8dca-dadab6f783b7
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

# ╔═╡ 1549c982-b5aa-54ba-80c0-000deeceece7
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ 95459b06-583d-5fe0-bf96-830980415543
saved = let
    common = Dict{String, Any}(
        "energies" => collect(energies),
        "efficiency" => efficiency.flat_field,
        "efficiency_one_dexel" => efficiency.one_dexel,
        "interaction_probability" => efficiency.interaction,
        "swank_factor" => efficiency.swank,
        "dqe0" => efficiency.dqe,
        "depths_mm" => table_depths,
        "efficiency_by_depth" => MC.energy_absorption_efficiency.(by_depth),
        "decision" => "use `efficiency`: flat field, Gd2O2S ceramic at 7.31 g/cm3 and 1.4 mm deep (both assumed; Canon publishes only 'praseodymium-activated ceramic'); fill factor of the assumed 0.1 mm reflector",
        "definition" => "mean energy collected per photon landing on a dexel's ceramic with every dexel illuminated (flat field), divided by the photon energy; the fill factor applies separately",
        "beam" => Dict(
            "flat_filter_al_mm" => filtration,
            "anode_angle_deg" => 10,
            "bowtie" => "BasisSimulator :large_body",
            "fitted_to" => "measured HVL 6.0, 7.4, 8.5 mm Al at 80, 100, 120 kV, Aquilion ONE PRISM (Ito et al. 2026)",
            "silverbeam_hvl_mm_al" => silverbeam.hvl,
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 08_canon_aquilion_one.jl",
    )
    paths = String[]
    for (file, scanner, rows, grid) in (
            (
                "canon_aquilion_one_genesis_efficiency_geant4.jls",
                "Canon Aquilion ONE GENESIS (TSX-305A)",
                aquilion.rows,
                "1-D (assumed; not published)",
            ),
            (
                "canon_aquilion_one_prism_efficiency_geant4.jls",
                "Canon Aquilion ONE PRISM (TSX-306A), with SilverBeam",
                aquilion.rows,
                "1-D (assumed; not published)",
            ),
        )
        contents = merge(
            common,
            Dict{String, Any}(
                "detector" => Dict(
                    "scanner" => scanner,
                    "material" => "praseodymium-activated ceramic, modelled as Gd2O2S at 7.31 g/cm3 (assumed)",
                    "depth_mm" => depth_mm,
                    "pitch_mm" => pitch_mm,
                    "columns" => aquilion.columns,
                    "rows" => rows,
                    "reflector" => "0.1 mm TiO2-loaded epoxy (assumed)",
                    "grid" => grid,
                ),
            ),
        )
        path = joinpath(table_dir, file)
        Serialization.serialize(path, contents)
        push!(paths, path)
    end
    paths
end

# ╔═╡ 09fd9bf9-da85-5488-ac38-c1bd9bf15da8
# Read them back in BasisSimulator's layout
[
    let t = Serialization.deserialize(path)
            (
                file = basename(path),
                rows = t["detector"]["rows"],
                at_60keV = t["efficiency"][60],
                at_120keV = t["efficiency"][120],
            )
    end
        for path in saved
]

# ╔═╡ db084ca4-290d-5555-bba5-06a94a2b6348
md"""
## Where This Leaves the Aquilion ONE
"""

# ╔═╡ 8c1258f9-0411-5633-807d-fa9809e266d0
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    worst_hvl = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        - **Little is documented**: 320 × 0.5 mm rows, 896 channels, the PUREViSION detector shared by GENESIS and PRISM.
          The distances (600 / 1072 mm), the pitch at the detector, the scintillator (Gd₂O₂S at 7.31 g/cm³, 1.4 mm), the
          reflector and the 1-D grid are worked out or assumed, and said so.
        - **The beam is fitted**: $(round(filtration; digits = 1)) mm of aluminium at a 10° anode reproduces the PRISM's measured half-value
          layers to within $(round(worst_hvl; digits = 2)) mm, and gives $(round(silverbeam.hvl; digits = 1)) mm through SilverBeam against the measured 13.7 mm.
        - **What a scan sees** is the flat-field efficiency: $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, with DQE(0)
          $(round(m120.dqe; digits = 3)); borrowing BasisSimulator's Flash table would give $(round(m120.table; digits = 3)).
        - **The whole detector**: 16 cm wide, its outer rows see the beam up to $(round(rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / aquilion.source_to_detector_mm)); digits = 1))° off the axis (section 7).
        - **Saved:** $(join(["`$(basename(p))`" for p in saved], " and ")), for BasisSimulator.
        """
    )
end

# ╔═╡ Cell order:
# ╟─80569823-8008-530d-a0df-34c4e21e0758
# ╟─0e0805a1-0452-5360-8e0e-27c7450b5769
# ╟─3c17a4f3-92ba-516a-beba-e0b6a903377b
# ╟─db0ace6b-e7f9-56bf-850a-db1a6d6e2e06
# ╟─33e46fd5-c542-55e8-bcb5-192a0bac3d5e
# ╠═ab9db025-f507-5388-bf67-4c0d170d0e28
# ╠═bc041287-f4f5-5ce7-8924-9633ea5dc417
# ╠═808a4723-438f-586f-b92c-6e49eb9f89e6
# ╠═8b53812d-c201-51bb-8847-1b66b5f1f09c
# ╠═429139dd-55bf-52bd-a9d5-9691d14d06a2
# ╠═1a29df66-ff12-5ff0-acdd-9c0e6b2b13d6
# ╠═726ef0af-666e-5b87-9ea4-02801135cab1
# ╠═3accdd63-8b76-5d41-a238-01a65ba33416
# ╠═10fb07d1-38ee-5894-b238-df1f1e70be84
# ╠═558bad13-644e-5e3d-be50-5c5850d10928
# ╠═ed268a94-6b31-5c94-87d3-d11efbaeb1a0
# ╠═c9b96524-0916-5e0a-bd06-366cce225226
# ╠═57dd0152-ffba-53a8-a5fc-33d79b6ce3ac
# ╠═1b1c6cc3-ced3-5a5c-affb-f6456a4012f2
# ╠═bfe11ba8-730e-58f7-9f66-212ffb916313
# ╟─ddadd9bd-a2b7-58f8-a970-3bff3a89201e
# ╠═54cfabcd-ec9e-5930-af26-fe46ae2f6df2
# ╠═bfd257ef-c874-5d3e-865b-91eca06e30f1
# ╟─ce933054-651a-5933-9566-efb31407e7cd
# ╠═f2132f86-2dcb-5010-bb46-b3216f321502
# ╠═f7c3b5dc-b20b-5db8-bbc9-8b7b9ad389bb
# ╠═4e886e38-891b-5244-b3f7-f4217cb5785c
# ╠═919250d4-f19e-5252-bad8-a0191e7f3dd9
# ╟─94ece8ce-3e4d-53b1-8769-420e883094e7
# ╠═6926962b-931d-56b2-be23-590de184f48e
# ╟─fe6f7616-c5c3-5864-b959-5fd0d16bb6be
# ╠═4a1a5d21-5581-50d2-b02a-9174b8bfcd02
# ╠═3e588088-f4e1-5470-a7d8-f02f18437d48
# ╠═17fe60de-9741-5ec5-9555-7afb4820e74b
# ╟─3d6ad596-a1da-5c24-b39a-8cdc0a1314da
# ╠═157eec79-393b-5b59-bb8a-950a2c4f5802
# ╟─fcee931d-a949-5b8c-9bee-d59e556f0fc1
# ╠═e01b99a8-d89e-53cd-9b56-b4dcc5c374db
# ╠═2594a24f-dc08-533c-8b9d-2f99108aab4e
# ╠═219cb299-5acc-5911-9eaf-2e86161e449d
# ╠═92ec3c2f-5d09-56f1-95fd-e5d2d1450ff2
# ╠═f266e1ff-8e7c-52ae-acfb-8fc1c6f2ca9f
# ╠═498b5b6a-a948-5126-92a3-364216f7b20e
# ╠═85053c4d-c787-591e-868e-b0548f93a32f
# ╠═f18449da-5fb9-5dd1-8889-b6f1fd902f59
# ╠═076c1cfb-65a3-5d2e-b5b0-1d264e756fe8
# ╠═77ad16fd-5153-5a4b-801a-57c22ebbd8b7
# ╟─5c906809-30a7-5e8f-b82e-78981da811f3
# ╟─4121046c-46f1-52bc-b36f-5e41408abb40
# ╠═4ee27e1b-2c0f-5ba4-baaa-6e883057b012
# ╠═ac571320-09f8-5d9b-843c-50d2bef6fc49
# ╟─e71c343e-4038-52da-8770-c462750ba0bc
# ╟─2baedffc-48d3-58a5-9816-c781f3598720
# ╠═767031ad-4240-58a7-8784-73ed6193049b
# ╠═ac0ae21f-6ba3-5b81-b801-b226d34a8645
# ╟─3734b13a-da38-504d-bb4a-ef79fccc86b9
# ╟─4456dbc6-d2ac-5b8b-b744-cd7109af51ea
# ╠═b0e5a025-3a3b-5d9d-8ae0-d597c6507d3d
# ╠═d21e6491-d12e-52ae-b383-bd12d671b4e9
# ╟─c012a993-2845-5b07-be0a-f019bf4fea80
# ╠═b2e83820-62f2-50f2-9aea-a57e382bfc14
# ╟─cf122acc-fc8b-5d8d-ad7d-84a701424756
# ╠═e4c398e8-b01d-565f-81ed-8a19750e5e33
# ╠═98a5f0ad-bbe1-5aab-8dc8-4e2aecbe102c
# ╠═76f1fe13-108c-5f56-aeb2-5b626860523d
# ╟─89682fb9-9156-5081-9815-65ae00e73940
# ╠═ff5bcd43-c401-569f-bb51-3b3163ebcde3
# ╟─b67523ad-a398-5025-a89e-31bbc644a886
# ╠═b807babb-788e-50b5-9857-ded2f7dfff09
# ╠═8a97e96d-38c0-52f0-a4b4-9c835b6413c8
# ╟─f4b79fc1-aa23-56f6-8bc1-eb2b3db7b47e
# ╠═fdbc6f9a-497d-588a-b8b5-6d7bd3b5d4e7
# ╠═65264960-57e5-5281-b471-39fa9e34f538
# ╟─dac0f49a-6dcf-5351-843b-5bf3d2d603df
# ╠═89ee2f6a-8d3d-5837-a72b-b16ac6dd781c
# ╟─eab56392-a661-5fc6-8f95-068d32573a2b
# ╠═f2759008-f052-5b85-92fb-04c23dfa5842
# ╟─9dbcae9a-cb06-5a25-a722-9fb39b6e7114
# ╟─66a50ae8-d09e-57d8-b69c-664f6b362651
# ╠═da786503-a2d6-5880-9361-079b494dabcc
# ╠═9ad0c644-b539-5e83-a22e-35fc92391228
# ╠═f1873392-5be8-5055-9c30-fdbe75ee33ab
# ╠═a6d94647-e210-576c-9539-a0ae4159ad3f
# ╟─3aea3630-70c7-5d4f-ab87-489a1320cce4
# ╠═d8ad9d29-97f4-5325-8ec6-68943312f265
# ╟─e259dd7e-12c9-56b0-b77b-93c23dd38226
# ╠═0ffbdeda-bc29-5233-9b92-7f5d1f13573b
# ╠═1c62ec1e-8fd3-52a8-b84f-5610a8257098
# ╠═53b69f21-a0cd-5ce4-bede-baa7b0519223
# ╟─23ab1779-eac6-5e29-af6d-d465174c5654
# ╠═d887e3ee-97ca-53ea-8ecd-ee1a7b5b6ea0
# ╟─2ab149ac-24be-56fc-9c18-fc41227b7113
# ╠═10c68b00-24d3-58cd-9f5b-2b25c917fb5f
# ╠═28f3d388-5432-5c8b-83dd-17ff0ca8fdd6
# ╠═5ae61127-e981-5614-b7cc-49a51e5fa77e
# ╠═77c888f8-546b-55bd-8e6d-76a9c2068b16
# ╠═6002b833-7de7-5e06-a753-277aab945c0e
# ╠═36ead4eb-6aa3-5a38-82f0-7427bcd7c338
# ╠═be6ef043-75d4-59cf-b172-491e2200a90a
# ╟─81e9027d-a62a-5a0c-a091-f587d7c0a121
# ╠═0f0250c8-45d8-56d7-b1e5-f15bcf7a6714
# ╟─5c7fff9b-6630-5e79-a499-efc14c14e3bc
# ╠═6b66d29d-5dab-579c-a5d8-1233dc4f35b9
# ╠═1703f60a-9d85-5a57-962d-91c4f8b0bece
# ╟─d10caf51-f7a2-5154-8d24-e1ed46c4d86c
# ╠═ff9e2424-574b-5274-9c8f-adbad84450d6
# ╠═45043ecb-063f-5b53-bd1f-5b559c97199a
# ╟─b8e5ce21-e91b-5d3a-a6eb-942d1fb42af8
# ╟─5ca60f2c-a230-5894-a8fa-1a5dcc16b64c
# ╠═2e1b5a19-af32-5b42-8f00-e79cbdc0fad1
# ╠═b247d175-c3ff-5561-98c7-bbc089fbf17a
# ╟─baaa15f4-add3-5d99-9695-3d6721e6d6f0
# ╠═47436151-80ce-52d6-856e-b75b78deee48
# ╟─c48615a2-707b-5ccc-b824-acf128726517
# ╟─8b77445b-bc28-5204-89ef-89814e715f0d
# ╠═14c7a701-5529-58e5-bd40-1d6d46907e66
# ╠═fd3f9361-3413-50e2-8dca-dadab6f783b7
# ╠═1549c982-b5aa-54ba-80c0-000deeceece7
# ╠═95459b06-583d-5fe0-bf96-830980415543
# ╠═09fd9bf9-da85-5488-ac38-c1bd9bf15da8
# ╟─db084ca4-290d-5555-bba5-06a94a2b6348
# ╠═8c1258f9-0411-5633-807d-fa9809e266d0
