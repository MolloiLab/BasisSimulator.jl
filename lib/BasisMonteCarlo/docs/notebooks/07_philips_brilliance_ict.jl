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


# ╔═╡ 1e8019db-b5b1-5ad1-a89a-ffa9fb0e8341
md"""
# The Philips Brilliance iCT Detector in Geant4

The Philips Brilliance iCT is an energy-integrating CT scanner, made in two widths: the
**iCT 256** (128 detector rows, 8 cm) and the **iCT SP**, sold as the iCT 128 (64 rows, 4 cm). Both
are at UCI Health, and both use the same detector, Philips' **NanoPanel**: a ceramic scintillator
cut into **dexels** (detector elements), with a reflector between them and a photodiode under every
dexel, curved in two directions (a spherical detector) behind a two-dimensional anti-scatter grid.

BasisSimulator has no Philips scanner yet: no geometry, no beam and no detector table. This
notebook builds the detector in Geant4, one piece at a time, and

1. **gathers what is documented** (the FDA clearances, Philips' Dunlee detector datasheet, the
   literature) and states every assumption where nothing is;
2. **fits the x-ray beam** to measured half-value layers, since Philips publishes no filtration;
3. **builds the detector's efficiency table**, with its noise (the Swank factor), and saves it for
   BasisSimulator, once for each of the two scanners.
"""

# ╔═╡ c34c0bd0-aa5c-5e4a-8bda-740e8ca07147
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a NanoPanel dexel does not count at all.
Every x ray it absorbs leaves energy in the ceramic, which glows; the reflector keeps the light
inside the dexel, and the photodiode under it adds up the light of every photon over the reading.
The signal is therefore proportional to the **energy** left in the dexel, and the efficiency that
matters is the energy-absorption efficiency: the mean energy a dexel collects per photon, divided
by the photon's energy.
"""

# ╔═╡ 5018fd7e-6324-5297-b849-332efe0aeed1
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

# ╔═╡ e1a742dd-c09a-5ada-87f2-df7c195ed69b
md"""
## The Brilliance iCT, as Documented

Every hardware number this notebook uses, from the FDA 510(k) summaries, Philips' Dunlee OEM
detector datasheet (which matches the iCT's dexel count and distances exactly), Philips' patents and
the literature. Where no source gives a number the row says so, and the notebook states its
assumption. BasisSimulator has no Philips values to compare against.

| quantity | documented | status | source |
|---|---|---|---|
| scintillator | "NanoPanel: ceramic scintillator + photodiode"; composition not stated (Philips' GOS patents describe Gd₂O₂S:Pr,Ce) | assumed here: Gd₂O₂S at 7.31 g/cm³ | K162838; US 8,492,724 |
| crystal depth | not published; Philips' GOS-array patent: "about 2 mm or less" | assumed here: 1.4 mm; 0.8–2.0 mm saved | US 7,728,302 |
| dexels | 86 016 (iCT 256) and 43 008 (iCT SP): 672 columns × 128 or 64 rows | ✓ | K162838 |
| dexel pitch at the detector | 1.408 mm across the fan × 1.14 mm along z (0.772 × 0.625 mm at the isocentre, calc) | ✓ (OEM datasheet) | Dunlee CD500 |
| z coverage | 8 cm (iCT 256), 4 cm (iCT SP) | ✓ | K162838 |
| source–isocentre, source–detector | 570, 1040 mm | ✓ | K162838 |
| detector shape | spherical | built here as a cylindrical arc (geometry only) | Dunlee CD500 |
| reflector between dexels | not published | assumed: 0.1 mm of TiO₂-loaded epoxy | – |
| anti-scatter grid | two-dimensional, pre-focused (iCT 256); the iCT SP's may be one-dimensional; material and dimensions not published | built here: tungsten, 0.1 mm walls, 10 mm tall (assumed) | Dunlee CD500; EP 2,433,156 |
| tube | 80, 100, 120, 140 kV; focal spots 0.6 × 0.7 and 1.1 × 1.2 mm, flying in x and z | ✓ | K162838 |
| anode angle, filtration | not published (a titanium filter of unstated thickness) | anode 10° assumed; filtration fitted in section 8 | – |
| beam quality | half-value layer 6.4, 7.7, 8.8 mm Al at 80, 100, 120 kV | the target of the beam fit | Trattner et al. 2017 |

Sources: FDA [K162838](https://www.accessdata.fda.gov/cdrh_docs/pdf16/K162838.pdf),
[K131773](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K131773.pdf);
Dunlee (Philips), [CD500 128-row detector datasheet](https://www.philips.com/c-dam/dunlee/downloads/5218142_Dunlee_Datasheet_Detector_CD500_128Rows.pdf),
[CD300 64-row detector datasheet](https://www.philips.com/c-dam/dunlee/downloads/5218150_Dunlee_Datasheet_Detector_CD300_64rows.pdf);
Philips patents [US 7,728,302](https://patents.google.com/patent/US7728302B2/en),
[US 8,492,724](https://patents.google.com/patent/US8492724),
[EP 2,433,156](https://patents.google.com/patent/EP2433156);
Trattner et al., [Med. Phys. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5734630/);
Siemens, [Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ ec086b05-5dc0-5983-bb6f-4f7ebd3781af
md"""
## Notebook Setup
"""

# ╔═╡ 03b0eb03-c3fe-59cb-acd8-5a14b0d80207
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ e457121e-c315-501e-aa97-d71761eb2100
using Markdown: @md_str, Markdown

# ╔═╡ cf074860-7eb8-551a-9545-d52b492f3d22
import PlutoUI

# ╔═╡ a281979d-cff0-5fa9-87ca-baf1abec600c
import BasisMonteCarlo as MC

# ╔═╡ 6ded5641-4694-5be0-8dd3-61dbc4c47466
import CairoMakie as Mke

# ╔═╡ 6399c96e-ad68-5802-b2fa-ed8629257ff3
import BasisSimulator as BS

# ╔═╡ 5fab2bcc-a52b-5a40-bfd2-4c3ab57a916f
import Serialization

# ╔═╡ 98b6fe03-3bc8-58a3-a3d2-51eb0a66db65
import DelimitedFiles

# ╔═╡ 673ac0c5-46e5-5961-9fcd-014c44cf2edd
using Statistics: mean

# ╔═╡ 092f835e-0e32-57d5-ba25-6a5783c55235
PlutoUI.TableOfContents()

# ╔═╡ 293d3699-48da-599a-bef5-fdd75fda7e1b
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 88fd7a1a-b153-5163-82fe-94476836d0c4
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

# ╔═╡ 43f718d9-bdcd-55dc-8181-8ec3ca691693
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

# ╔═╡ 68f8c99b-e252-5cf4-a55b-ad5517bc4017
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

# ╔═╡ fe7cf193-49ce-5b3e-8d57-69460f810f9a
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

# ╔═╡ 4a2ba4d6-c8ee-536d-b050-8d8e63896fc4
md"""
## 1. The Scintillator

Philips' FDA filings say only "ceramic scintillator"; its patents for CT detector arrays describe
gadolinium oxysulfide, Gd₂O₂S, activated with praseodymium and cerium at levels far too small to
matter for x rays. This notebook therefore takes the NanoPanel to be a dense Gd₂O₂S ceramic, at
**7.31 g/cm³**, the density Siemens measures for its own Gd₂O₂S ceramic (99.95 % of theoretical).
That is an assumption: Philips publishes neither composition nor density.
"""

# ╔═╡ 3c064df0-e2d1-5119-8221-98b267523e6c
# Gd₂O₂S ceramic at 7.31 g/cm³ (assumed: Philips publishes neither composition nor density)
gos = MC.DetectorMaterial("GOS ceramic"; density = 7.31, formula = "Gd2O2S")

# ╔═╡ f9c206fd-83ce-5049-a55f-7e6f67e51826
# Gadolinium's K-shell: its edge and the energy of the x ray it emits (Kα)
gadolinium = only(MC.k_edges(gos))

# ╔═╡ 02efe6e9-af41-50a5-ab20-dd5e3ea56823
md"""
Gadolinium's **K-edge, 50.24 keV**, sits in the middle of every CT spectrum. Below it a photon
cannot free a K-shell electron; just above it, it can, and absorption jumps sevenfold. Every
such absorption leaves the atom to emit, nine times in ten, a **Gd K x ray of about 43 keV**,
which is just below the edge and so is absorbed only weakly: it can travel a millimetre or more
before stopping, which is the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ 9adff7b9-1f90-5e35-bfba-1fb3b0ce2850
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

# ╔═╡ 8bef983f-486a-5add-af56-509a943fc213
mu_energies = 10.0:0.25:150.0

# ╔═╡ ce839469-d815-502c-a9a8-b239ef98e6e4
# Geant4's attenuation of the ceramic, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ f54d6623-1392-523e-9669-cda9e27df0e6
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 58cb196e-00e0-5355-9388-1c235999f0e2
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

# ╔═╡ b702ba31-bdd4-535a-8bb5-82d4998424ad
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

# ╔═╡ 9de8c04a-d7fa-563a-9037-a24a4874949f
md"""
### A First Check: A Published Gd₂O₂S Curve

Philips publishes no absorption curve, but Siemens publishes the fraction of x rays that **1.4 mm
of its own Gd₂O₂S ceramic** absorbs (*Detectors & Crystals*, p. 19). It is the probability that a
photon interacts at all, `1 − exp(−μd)`, so it checks the material model and the density assumed
here. The values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ 4dbb12b4-8b84-55f4-971d-834621179fd2
# Siemens' curve, read from the vector figure (reference_data/siemens_ufc_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_ufc_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ e3e5fb2d-8839-5c6d-bc55-6ab33f4fbdc0
# The crystal depth (assumed: not published; Philips' patent says "about 2 mm or less")
depth_mm = 1.4

# ╔═╡ 1c6c8e35-e26f-5c80-af70-8886225448fd
# The probability a photon interacts in 1.4 mm of the ceramic
interaction_curve = [1 - exp(-MC.geant4_attenuation(gos, E) * 1.4) for E in mu_energies]

# ╔═╡ 9494b99e-6ee9-5e7c-bfc6-4107cf482f79
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

# ╔═╡ c250a1e3-30e5-53ce-91ca-7e6d01da00c9
let
    ours = [MC.interpolate_linear(collect(mu_energies), interaction_curve, E) for E in siemens_curve.energy_keV]
    low = siemens_curve.energy_keV .<= 90
    Markdown.parse(
        """
        Up to 90 keV the published curve and Geant4 for 1.4 mm of Gd₂O₂S at 7.31 g/cm³ agree to within
        **$(round(maximum(abs, (siemens_curve.absorption .- ours)[low]); digits = 3))**, inside the figure's reading accuracy: the material model is sound.
        Whether the NanoPanel is this material at this density, nothing published says.
        """
    )
end

# ╔═╡ 9d862dad-bf55-5fdf-a8aa-42c15cea2306
md"""
## 2. One Dexel and Its Neighbours

A dexel is one block of ceramic with reflector on its four sides. Its pitch at the detector face
is documented in Philips' Dunlee datasheet for the 128-row detector: **1.408 mm across the fan by
1.14 mm along z**. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy, the usual CT
reflector, is assumed. The crystal's depth is not published either; 1.4 mm is assumed (within the
"about 2 mm or less" of Philips' patent), and the table is saved for other depths too.
"""

# ╔═╡ 394d3e3f-93d1-56fe-bcdb-6c13f6cb7849
# The Brilliance iCT's geometry: FDA K162838 and the Dunlee CD500 datasheet
ict = (
    source_to_isocenter_mm = 570.0,
    source_to_detector_mm = 1040.0,
    columns = 672,
    rows = 128,          # iCT 256; the iCT SP has 64
    rows_sp = 64,
    pitch_at_detector_mm = (1.408, 1.14),
)

# ╔═╡ a754a356-83f4-547f-9d84-1cdda3dbe6b2
# The dexel pitch at the detector face (across the fan, along z), in mm
pitch_mm = ict.pitch_at_detector_mm

# ╔═╡ 6ab628e2-352e-5a0a-b3ea-dbbc5afeacdd
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ bb3446dc-7de0-5105-905b-874f3cf8b572
# The block of ceramic each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ c6349ec8-c5a9-58ff-a25f-2246c548381a
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 8670cbca-682f-56ab-b865-7b3a01ff1c2d
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

# ╔═╡ de7908ce-c3e4-5a45-89cd-759a322c3d82
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

# ╔═╡ 6f1cc53f-bdcd-544a-a60c-d85e6ad99214
# Photons landing uniformly on one dexel's ceramic, and on its whole cell (ceramic and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ 8cac5e6c-cede-53b4-ab8a-d56c561ebadd
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 5d77a48b-c23d-5aad-a0c9-ad585a95cd1f
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

# ╔═╡ 894ed03e-aca5-5497-a05e-e250aa7186a7
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

# ╔═╡ 5e60b6ee-9554-5f5c-bc2f-ab22c98206f3
md"""
### What a Photon Does in the Ceramic

`record_tracks` follows a few hundred 60 keV photons landing on the central dexel, every step of
every particle. Two of them are drawn: one whose Gd x ray is absorbed **in the next dexel**, so
its energy is split between two photodiodes, and one whose Gd x ray **leaves through the front
face**, so its 43 keV are lost to every dexel. Both happen often, because the Gd x ray is so
weakly absorbed.
"""

# ╔═╡ 6a1403cc-ac6c-5bdb-9833-3abe703e67ce
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    60.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ 9777254d-9999-50b0-833f-4545c6b84cfb
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

# ╔═╡ 7cf5e41e-9f00-5457-aa85-61be46632fe0
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

# ╔═╡ 59507a74-4442-5ca0-a716-8ddd907568c7
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is not one number but a
spread: the full photon energy when everything stays in the dexel, the energy minus 43 keV when
the Gd x ray leaves, a little when the photon only Compton-scatters once and leaves. Three
energies show the three regimes: 45 keV (below the K-edge), 60 keV (just above it) and 120 keV.
"""

# ╔═╡ 9948f898-1d1d-5a61-856c-15b2704e1306
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

# ╔═╡ 0773a247-588d-565c-8752-5d4e21bb843f
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ 39a75aab-0cbc-5ab9-82fa-77f14140b2e0
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

# ╔═╡ d52dcc27-2572-524f-9683-7ba09b3e847d
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the ceramic or
leaves it: as the Gd x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of ceramic (no reflector), so that whatever
leaves the ceramic leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ 11d971ce-2bac-52bc-87d5-f5d096b8a656
budget_energies = 15.0:5.0:140.0

# ╔═╡ 7295db49-8d5d-59de-a43e-113ed6a28a2e
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(gos, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ 9169c0cc-f78b-5697-ab12-33ad5ca5654c
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

# ╔═╡ 3741ebf6-6078-592b-afc7-ea3706eb3386
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

# ╔═╡ 728d0bdf-38e7-5044-adfa-594b5bd87bdf
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

# ╔═╡ 3f1986f0-e3f2-51c7-925c-25ca42d42836
energies = 1.0:1.0:140.0

# ╔═╡ 099118be-25b9-5d87-a5f1-00931a2c71f7
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 3b4afd60-19f1-52ec-9355-73a9faa66245
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ db494983-9552-582e-857d-9dffb7afdb00
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

# ╔═╡ 62aad628-ba2a-50a0-9040-c5742f0c9d13
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

# ╔═╡ 0c5f1902-7ad2-5f34-9e0d-4a9dd16e3262
md"""
### A Check Against Published Monte Carlo

Boone et al. (Med. Phys. 1999) followed x-ray fluorescence in thin Gd₂O₂S screens with the TART 98
code: a pencil beam at normal incidence on a screen of Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S, the binder not counted), the energy tallied in a circle 0.1 mm across the
beam ("absorbed as primary") and in rings out to 4.5 mm ("lateral"), and ideal sensors in front of
the screen and behind it. At 55 keV they report (p. 908) how the incident energy divides. The same
screen and the same tallies in Geant4:
"""

# ╔═╡ b8d0877a-78e8-5adf-8850-d22dbfe2e2bf
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

# ╔═╡ fcd5bd5f-321e-5322-abcd-872d1486dc05
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

# ╔═╡ 4343e931-d6ea-5aa8-86b7-a3f9ad3e149a
md"""
## 5. Compared with BasisSimulator's Nearest Table

BasisSimulator has no Philips detector. The nearest table it has is its SOMATOM Definition Flash
table, `UFC_FLASH_MC_EFFICIENCY_LUT`: Gd₂O₂S, and, judging by its shape, made as a pencil beam into
the centre of one dexel 1.4 mm deep at 7.44 g/cm³, tallied in that dexel. It is what a Philips model
would use today if it borrowed a Siemens table. How it compares with the NanoPanel's flat field:
"""

# ╔═╡ 97418e01-601a-5e85-bdb4-0c291a14f494
# BasisSimulator's nearest table: the Definition Flash's Gd₂O₂S (MCNP, Hamidreza Khodajou-Chokami)
# BasisSimulator v0.15.1's table, as BasisMonteCarlo keeps it for comparison
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash

# ╔═╡ 7cb3e6da-71fa-5220-bc3b-46f8c9659eaa
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ 5bafab62-de83-566e-8809-4542f9986240
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The nearest BasisSimulator table against the NanoPanel at 1.4 mm",
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

# ╔═╡ 4a9d192f-853c-5c30-b135-b9c654181bd1
let
    at(E) = findfirst(==(E), collect(energies))
    rows = [
        "| $(Int(E)) keV | $(round(table_at(E); digits = 3)) | $(round(efficiency.flat_field[at(E)]; digits = 3)) |"
            for E in (40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
    ]
    Markdown.parse(
        """
        | photons of | BasisSimulator's Flash table | Geant4, NanoPanel flat field |
        |---|---|---|
        $(join(rows, "\n"))

        Just above the K-edge the flat field is higher (each dexel gets back the Gd x rays its neighbours lose); at high
        energy it is lower (7.31 g/cm³ rather than 7.44, and photons landing near a dexel's edge lose more). Borrowing
        the Siemens table would misplace the Philips detector's efficiency by several hundredths at the energies a
        scan uses most.
        """
    )
end

# ╔═╡ f62ede7a-50a6-528f-9fe9-1eee52766a58
md"""
## 6. The Anti-Scatter Grid

The iCT 256's detector has a **two-dimensional**, pre-focused anti-scatter grid (Dunlee's datasheet
for the 128-row detector; Philips' patent EP 2,433,156); the iCT SP's 64-row sibling may have a
one-dimensional one, which nothing published confirms. Material and dimensions are not published.
Built here: **tungsten walls 0.1 mm thick and 10 mm tall** (assumed), one cell per dexel, standing on
the reflector, so they shade no ceramic.
"""

# ╔═╡ 39bf1600-bece-5065-8028-4cbdb0f46cfd
grid = MC.Collimator(;
    height_mm = 10.0,
    wall_mm = 0.1,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :both,
    cells = 13,
)

# ╔═╡ ab633fe6-4662-5646-a9bf-e1b2cb27eea4
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
        "ceramic";
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

# ╔═╡ 5e8374e9-8118-5d45-a15d-37291ee50143
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (ceramic and reflector), with and without the grid: the grid
should cost almost nothing, apart from its own tungsten x rays (59 and 67 keV) reaching the
ceramic above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ 2c65e6be-5d25-5bbe-9f42-1fb5f613e03e
grid_energies = 20.0:5.0:140.0

# ╔═╡ 4c4fce07-e9dc-5385-9787-cdc0dbfed941
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

# ╔═╡ 476ceedb-0ddf-50a6-93ee-0f2c6f0d2440
angles_deg = 0.0:1.0:20.0

# ╔═╡ 81032d50-5b45-54e9-98dc-7a3b9d343c87
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

# ╔═╡ 1b7c92ca-1c9f-5bf2-af18-b8f3b682798f
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

# ╔═╡ 1fa6a519-6041-510f-a5c2-8a4ccdfcc902
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) % (the walls stand on the reflector; above 69.5 keV the walls' own x rays add a little).
        At an angle it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %. A
        photon that scatters in the patient arrives from anywhere, so most of it is stopped.
        """
    )
end

# ╔═╡ 44ac8c50-3613-5fa6-91fb-68206beb55d4
md"""
## 7. The Whole Detector

The iCT 256's whole detector: 672 columns × 128 rows on an arc of radius 1040 mm around the focal
spot, every dexel a block of ceramic in reflector, under the 2-D grid, all of it focused on the
focal spot. (The real detector is spherical; across the fan and along z every dexel faces the focal
spot either way.) The iCT SP's is the same detector, half as wide. An air scan fires photons
uniformly from the focal spot over the whole detector, and every dexel's signal is summed. Two
questions: is every dexel the same, and does the whole detector collect what the single dexel of
sections 4 and 6 predicts?
"""

# ╔═╡ 1811f34e-14b5-59d0-9953-667d0f42d13c
detector = MC.ArcDetector(;
    columns = ict.columns,
    rows = ict.rows,
    radius_mm = ict.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :both,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = gos,
)

# ╔═╡ c1a130ac-f15e-571a-af8a-01ca4badfbcf
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ a2b31647-666c-54be-a436-657c08249224
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

# ╔═╡ eb74df12-b344-5362-87f4-55d783c39c99
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
    R = ict.source_to_detector_mm
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
        [ict.source_to_isocenter_mm];
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
        ict.source_to_isocenter_mm,
        "isocentre";
        align = (:left, :center),
        offset = (12, 0),
        fontsize = 15,
    )
    note!(
        axa,
        0,
        R,
        "$(ict.columns) × $(ict.rows) dexels, $(round(MC.fan_angle_deg(detector); digits = 1))° fan";
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
        limits = ((1, ict.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:ict.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / ict.columns
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
        ict.columns / 2,
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
        limits = ((1, ict.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:ict.rows, max(1, ict.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / ict.rows
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

# ╔═╡ 69491fa0-423d-5973-9fe6-6aa45682ee37
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
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / ict.source_to_detector_mm))
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

# ╔═╡ 50184044-477a-5299-90e6-416d67bf057a
md"""
## 8. The iCT's X-Ray Beam

Philips publishes neither the anode angle nor the filtration (beyond "a titanium filter" of no
stated thickness), so the beam is fitted. The spectra are BasisSimulator's IPEM Report 78 tungsten
spectra at a 10° anode (assumed), through a flat aluminium filter whose thickness is chosen so that
the half-value layer at the isocentre, through the thinnest part of BasisSimulator's large-body
bowtie (2.5 mm of aluminium), matches Trattner et al.'s measurements on a Brilliance iCT:
**6.4, 7.7 and 8.8 mm of aluminium at 80, 100 and 120 kV**.
"""

# ╔═╡ 8c632429-70dc-57e1-a968-7f7f0c831027
# Measured half-value layers at the isocentre, Brilliance iCT (Trattner et al., Med. Phys. 2017)
measured_hvl = [(kVp = 80, hvl = 6.4), (kVp = 100, hvl = 7.7), (kVp = 120, hvl = 8.8)]

# ╔═╡ ad0ee5e7-30f3-5144-9618-99680c8b58c0
aluminium = MC.DetectorMaterial(:Al)

# ╔═╡ 2b4ddb85-5b0e-5572-a884-08c034f20cbe
# The large-body bowtie at its thinnest, the ray through the isocentre: aluminium only, in mm
bowtie_al_mm = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
end

# ╔═╡ ed171294-6af7-58ed-bceb-efff28a6b4d8
"""
The photons per mAs per mm² reaching the detector, per bin, for an iCT protocol at `kVp`, with
`flat_mm` of aluminium as the beam's filtration and a 10° anode.
"""
function ict_spectrum(kVp; flat_mm)
    scanner = BS.EICTScanner(
        source_to_isocenter = ict.source_to_isocenter_mm,
        source_to_detector = ict.source_to_detector_mm,
        detector_rows = ict.rows,
        detector_cols = ict.columns,
        detector_row_size = pitch_mm[2] * ict.source_to_isocenter_mm / ict.source_to_detector_mm,
        detector_col_size = pitch_mm[1] * ict.source_to_isocenter_mm / ict.source_to_detector_mm,
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

# ╔═╡ e676d602-7cac-54de-999e-a7465e957763
# The half-value layer at the isocentre for `flat_mm` of filtration, through the bowtie's centre
function hvl_at(kVp, flat_mm)
    e, w = ict_spectrum(kVp; flat_mm)
    μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
    return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
end

# ╔═╡ 0fcfa999-406e-5a3f-9e9c-4494ebe0d683
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

# ╔═╡ 659f32de-5a65-54da-a298-5515dc4eb347
hvl_check = [(kVp = m.kVp, fitted = hvl_at(m.kVp, filtration), measured = m.hvl) for m in measured_hvl]

# ╔═╡ ffcbc0d1-307e-598b-8a88-9ae6a0955a96
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
        label = "measured (Trattner et al. 2017)",
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

# ╔═╡ de610460-4367-5ed4-8880-f975f3c423c0
let
    worst = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        **$(round(filtration; digits = 1)) mm of aluminium** at a 10° anode reproduces the measured half-value layers to within
        **$(round(worst; digits = 2)) mm** at every tube voltage: one filtration fits all three, which is what a real tube's fixed filtration
        does. This fitted beam is the iCT beam for the rest of the notebook and is saved with the table. (The fit
        lumps the unstated titanium filter, the tube's own filtration and the anode angle into one aluminium thickness;
        the bowtie the measurement used is not stated.)
        """
    )
end

# ╔═╡ 4d4fe7bd-017b-56cd-9fb4-71993782ef42
md"""
The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ 3d965bcf-74b8-502a-8ea1-896161f71704
@bind kvp PlutoUI.Slider([80, 100, 120, 140]; default = 120, show_value = true)

# ╔═╡ 458f37c2-1cda-54e0-805f-21366ba9fb57
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

# ╔═╡ 75de37dc-82d1-5ea1-a8fc-f86b81c2f954
let
    Mke.set_theme!(THEME)
    spectrum = ict_spectrum(kvp; flat_mm = filtration)
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
        label = "Geant4, NanoPanel flat field",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ 4070778b-47ba-561c-86c5-99008cb991ba
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. Every iCT tube voltage, through the fitted beam:
"""

# ╔═╡ 566d74f6-a104-5697-b4cd-15e7cd351eed
scan_protocols = [(label = "$(k) kV", spectrum = ict_spectrum(k; flat_mm = filtration)) for k in (80, 100, 120, 140)]

# ╔═╡ e73cb88d-3684-5b40-b2b1-549d4394a785
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

# ╔═╡ 90dc1e82-8b70-5ec7-99a3-af34e29179aa
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

# ╔═╡ c65ca608-7514-50d2-ace5-31a7f8abbfce
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | BasisSimulator's Flash table | Geant4, NanoPanel 1.4 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's ceramic; BasisSimulator's fill factor (0.9 × 0.9) applies to both
        alike.
        """
    )
end

# ╔═╡ 899d1884-4c2a-51ff-ab93-a5b7af30bc97
md"""
## The Table to Use

When BasisSimulator gains a Philips Brilliance iCT, this is the table to use, and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **Gd₂O₂S ceramic at 7.31 g/cm³, 1.4 mm deep** (assumed). Philips publishes neither; its patents
  describe Gd₂O₂S arrays "about 2 mm or less" deep, and 1.4 mm is the one Gd₂O₂S CT thickness a
  vendor (Siemens) quotes. Every depth from 0.8 to 2.0 mm is saved beside it.
- **The fill factor of the assumed reflector**, about 0.93 across × 0.91 along z; the grid's walls
  stand on the reflector and take nothing more (section 6).
- **The fitted beam**: BasisSimulator's IPEM spectra at a 10° anode through the fitted aluminium,
  which reproduces the measured half-value layers (section 8).

The iCT 256 and the iCT SP share the detector, so both files hold the same efficiency; they differ in
rows, coverage and grid.
"""

# ╔═╡ 753b7d0c-b3e0-59e2-a8b8-a3ce8d806c00
md"""
## Saving the Tables

Two files in `lib/BasisMonteCarlo/tables/`, one per scanner, in the layout of BasisSimulator's
`*_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of the NanoPanel as
modelled here, beside the one-dexel efficiency, the interaction probability, the Swank factor,
DQE(0), the efficiency for depths from 0.8 to 2.0 mm, and the fitted beam.
"""

# ╔═╡ c7de753a-cca5-5efe-97f6-71973736c4e4
table_depths = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# ╔═╡ 9c331ef8-e137-562d-a92c-5035f6921af6
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

# ╔═╡ 55a7667e-c0bc-5685-8ac3-dd8dc1077898
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ ff228af5-db4f-5f96-bdf7-fd9de37b4fea
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
        "decision" => "use `efficiency`: flat field, Gd2O2S ceramic at 7.31 g/cm3 and 1.4 mm deep (both assumed; Philips publishes neither); fill factor of the assumed 0.1 mm reflector",
        "definition" => "mean energy collected per photon landing on a dexel's ceramic with every dexel illuminated (flat field), divided by the photon energy; the fill factor applies separately",
        "beam" => Dict(
            "flat_filter_al_mm" => filtration,
            "anode_angle_deg" => 10,
            "bowtie" => "BasisSimulator :large_body",
            "fitted_to" => "measured HVL 6.4, 7.7, 8.8 mm Al at 80, 100, 120 kV (Trattner et al. 2017)",
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 07_philips_brilliance_ict.jl",
    )
    paths = String[]
    for (file, scanner, rows, grid) in (
            (
                "philips_brilliance_ict256_efficiency_geant4.jls",
                "Philips Brilliance iCT 256",
                ict.rows,
                "2-D, pre-focused (Dunlee CD500)",
            ),
            (
                "philips_brilliance_ict_sp_efficiency_geant4.jls",
                "Philips Brilliance iCT SP (iCT 128)",
                ict.rows_sp,
                "1-D or 2-D: not confirmed",
            ),
        )
        contents = merge(
            common,
            Dict{String, Any}(
                "detector" => Dict(
                    "scanner" => scanner,
                    "material" => "Gd2O2S ceramic, 7.31 g/cm3 (assumed)",
                    "depth_mm" => depth_mm,
                    "pitch_mm" => pitch_mm,
                    "columns" => ict.columns,
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

# ╔═╡ c57c863f-8b8d-514b-9a96-a1dac0cd9a41
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

# ╔═╡ b47feb77-eb44-530c-9eb8-8993d09a5760
md"""
## Where This Leaves the Brilliance iCT
"""

# ╔═╡ 41a509c6-8f7f-5833-b89a-224880f73922
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    worst_hvl = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        - **The geometry is documented**: 570 / 1040 mm, 672 × 128 dexels (64 on the iCT SP) of 1.408 × 1.14 mm at the
          detector, a spherical detector behind a 2-D grid. **The scintillator is not**: Gd₂O₂S ceramic at 7.31 g/cm³ and
          1.4 mm deep are assumptions, with other depths saved.
        - **The beam is fitted**: $(round(filtration; digits = 1)) mm of aluminium at a 10° anode reproduces the measured half-value layers to
          within $(round(worst_hvl; digits = 2)) mm.
        - **What a scan sees** is the flat-field efficiency: $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, with DQE(0)
          $(round(m120.dqe; digits = 3)); borrowing BasisSimulator's Flash table would give $(round(m120.table; digits = 3)).
        - **The whole detector** collects what one dexel's cell does.
        - **Saved:** $(join(["`$(basename(p))`" for p in saved], " and ")), for BasisSimulator.
        """
    )
end

# ╔═╡ Cell order:
# ╟─1e8019db-b5b1-5ad1-a89a-ffa9fb0e8341
# ╟─c34c0bd0-aa5c-5e4a-8bda-740e8ca07147
# ╟─5018fd7e-6324-5297-b849-332efe0aeed1
# ╟─e1a742dd-c09a-5ada-87f2-df7c195ed69b
# ╟─ec086b05-5dc0-5983-bb6f-4f7ebd3781af
# ╠═03b0eb03-c3fe-59cb-acd8-5a14b0d80207
# ╠═e457121e-c315-501e-aa97-d71761eb2100
# ╠═cf074860-7eb8-551a-9545-d52b492f3d22
# ╠═a281979d-cff0-5fa9-87ca-baf1abec600c
# ╠═6ded5641-4694-5be0-8dd3-61dbc4c47466
# ╠═6399c96e-ad68-5802-b2fa-ed8629257ff3
# ╠═5fab2bcc-a52b-5a40-bfd2-4c3ab57a916f
# ╠═98b6fe03-3bc8-58a3-a3d2-51eb0a66db65
# ╠═673ac0c5-46e5-5961-9fcd-014c44cf2edd
# ╠═092f835e-0e32-57d5-ba25-6a5783c55235
# ╠═293d3699-48da-599a-bef5-fdd75fda7e1b
# ╠═88fd7a1a-b153-5163-82fe-94476836d0c4
# ╠═43f718d9-bdcd-55dc-8181-8ec3ca691693
# ╠═68f8c99b-e252-5cf4-a55b-ad5517bc4017
# ╠═fe7cf193-49ce-5b3e-8d57-69460f810f9a
# ╟─4a2ba4d6-c8ee-536d-b050-8d8e63896fc4
# ╠═3c064df0-e2d1-5119-8221-98b267523e6c
# ╠═f9c206fd-83ce-5049-a55f-7e6f67e51826
# ╟─02efe6e9-af41-50a5-ab20-dd5e3ea56823
# ╠═9adff7b9-1f90-5e35-bfba-1fb3b0ce2850
# ╠═8bef983f-486a-5add-af56-509a943fc213
# ╠═ce839469-d815-502c-a9a8-b239ef98e6e4
# ╠═f54d6623-1392-523e-9669-cda9e27df0e6
# ╟─58cb196e-00e0-5355-9388-1c235999f0e2
# ╠═b702ba31-bdd4-535a-8bb5-82d4998424ad
# ╟─9de8c04a-d7fa-563a-9037-a24a4874949f
# ╠═4dbb12b4-8b84-55f4-971d-834621179fd2
# ╠═e3e5fb2d-8839-5c6d-bc55-6ab33f4fbdc0
# ╠═1c6c8e35-e26f-5c80-af70-8886225448fd
# ╟─9494b99e-6ee9-5e7c-bfc6-4107cf482f79
# ╠═c250a1e3-30e5-53ce-91ca-7e6d01da00c9
# ╟─9d862dad-bf55-5fdf-a8aa-42c15cea2306
# ╠═394d3e3f-93d1-56fe-bcdb-6c13f6cb7849
# ╠═a754a356-83f4-547f-9d84-1cdda3dbe6b2
# ╠═6ab628e2-352e-5a0a-b3ea-dbbc5afeacdd
# ╠═bb3446dc-7de0-5105-905b-874f3cf8b572
# ╠═c6349ec8-c5a9-58ff-a25f-2246c548381a
# ╠═8670cbca-682f-56ab-b865-7b3a01ff1c2d
# ╠═de7908ce-c3e4-5a45-89cd-759a322c3d82
# ╠═6f1cc53f-bdcd-544a-a60c-d85e6ad99214
# ╠═8cac5e6c-cede-53b4-ab8a-d56c561ebadd
# ╠═5d77a48b-c23d-5aad-a0c9-ad585a95cd1f
# ╟─894ed03e-aca5-5497-a05e-e250aa7186a7
# ╟─5e60b6ee-9554-5f5c-bc2f-ab22c98206f3
# ╠═6a1403cc-ac6c-5bdb-9833-3abe703e67ce
# ╠═9777254d-9999-50b0-833f-4545c6b84cfb
# ╟─7cf5e41e-9f00-5457-aa85-61be46632fe0
# ╟─59507a74-4442-5ca0-a716-8ddd907568c7
# ╠═9948f898-1d1d-5a61-856c-15b2704e1306
# ╠═0773a247-588d-565c-8752-5d4e21bb843f
# ╟─39a75aab-0cbc-5ab9-82fa-77f14140b2e0
# ╟─d52dcc27-2572-524f-9683-7ba09b3e847d
# ╠═11d971ce-2bac-52bc-87d5-f5d096b8a656
# ╠═7295db49-8d5d-59de-a43e-113ed6a28a2e
# ╟─9169c0cc-f78b-5697-ab12-33ad5ca5654c
# ╠═3741ebf6-6078-592b-afc7-ea3706eb3386
# ╟─728d0bdf-38e7-5044-adfa-594b5bd87bdf
# ╠═3f1986f0-e3f2-51c7-925c-25ca42d42836
# ╠═099118be-25b9-5d87-a5f1-00931a2c71f7
# ╠═3b4afd60-19f1-52ec-9355-73a9faa66245
# ╟─db494983-9552-582e-857d-9dffb7afdb00
# ╠═62aad628-ba2a-50a0-9040-c5742f0c9d13
# ╟─0c5f1902-7ad2-5f34-9e0d-4a9dd16e3262
# ╠═b8d0877a-78e8-5adf-8850-d22dbfe2e2bf
# ╠═fcd5bd5f-321e-5322-abcd-872d1486dc05
# ╟─4343e931-d6ea-5aa8-86b7-a3f9ad3e149a
# ╠═97418e01-601a-5e85-bdb4-0c291a14f494
# ╠═7cb3e6da-71fa-5220-bc3b-46f8c9659eaa
# ╟─5bafab62-de83-566e-8809-4542f9986240
# ╠═4a9d192f-853c-5c30-b135-b9c654181bd1
# ╟─f62ede7a-50a6-528f-9fe9-1eee52766a58
# ╠═39bf1600-bece-5065-8028-4cbdb0f46cfd
# ╟─ab633fe6-4662-5646-a9bf-e1b2cb27eea4
# ╟─5e8374e9-8118-5d45-a15d-37291ee50143
# ╠═2c65e6be-5d25-5bbe-9f42-1fb5f613e03e
# ╠═4c4fce07-e9dc-5385-9787-cdc0dbfed941
# ╠═476ceedb-0ddf-50a6-93ee-0f2c6f0d2440
# ╠═81032d50-5b45-54e9-98dc-7a3b9d343c87
# ╟─1b7c92ca-1c9f-5bf2-af18-b8f3b682798f
# ╠═1fa6a519-6041-510f-a5c2-8a4ccdfcc902
# ╟─44ac8c50-3613-5fa6-91fb-68206beb55d4
# ╠═1811f34e-14b5-59d0-9953-667d0f42d13c
# ╠═c1a130ac-f15e-571a-af8a-01ca4badfbcf
# ╠═a2b31647-666c-54be-a436-657c08249224
# ╟─eb74df12-b344-5362-87f4-55d783c39c99
# ╠═69491fa0-423d-5973-9fe6-6aa45682ee37
# ╟─50184044-477a-5299-90e6-416d67bf057a
# ╠═8c632429-70dc-57e1-a968-7f7f0c831027
# ╠═ad0ee5e7-30f3-5144-9618-99680c8b58c0
# ╠═2b4ddb85-5b0e-5572-a884-08c034f20cbe
# ╠═ed171294-6af7-58ed-bceb-efff28a6b4d8
# ╠═e676d602-7cac-54de-999e-a7465e957763
# ╠═0fcfa999-406e-5a3f-9e9c-4494ebe0d683
# ╠═659f32de-5a65-54da-a298-5515dc4eb347
# ╟─ffcbc0d1-307e-598b-8a88-9ae6a0955a96
# ╠═de610460-4367-5ed4-8880-f975f3c423c0
# ╟─4d4fe7bd-017b-56cd-9fb4-71993782ef42
# ╠═3d965bcf-74b8-502a-8ea1-896161f71704
# ╠═458f37c2-1cda-54e0-805f-21366ba9fb57
# ╟─75de37dc-82d1-5ea1-a8fc-f86b81c2f954
# ╟─4070778b-47ba-561c-86c5-99008cb991ba
# ╠═566d74f6-a104-5697-b4cd-15e7cd351eed
# ╠═e73cb88d-3684-5b40-b2b1-549d4394a785
# ╟─90dc1e82-8b70-5ec7-99a3-af34e29179aa
# ╠═c65ca608-7514-50d2-ace5-31a7f8abbfce
# ╟─899d1884-4c2a-51ff-ab93-a5b7af30bc97
# ╟─753b7d0c-b3e0-59e2-a8b8-a3ce8d806c00
# ╠═c7de753a-cca5-5efe-97f6-71973736c4e4
# ╠═9c331ef8-e137-562d-a92c-5035f6921af6
# ╠═55a7667e-c0bc-5685-8ac3-dd8dc1077898
# ╠═ff228af5-db4f-5f96-bdf7-fd9de37b4fea
# ╠═c57c863f-8b8d-514b-9a96-a1dac0cd9a41
# ╟─b47feb77-eb44-530c-9eb8-8993d09a5760
# ╠═41a509c6-8f7f-5833-b89a-224880f73922
