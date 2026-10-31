### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ ed5d0968-55b8-54bb-be08-b8b3e5231445
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 73911ef9-afec-5057-9406-29f4217f5d15
using Markdown: @md_str, Markdown

# ╔═╡ 5285d69f-7fb4-488a-9954-5e2d36bf1204
using Statistics: mean

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
above each threshold, in the dexel it hit and in each neighbour. Up to and including
**v0.15.1**, BasisSimulator uses such a table, computed by Hamidreza Khodajou-Chokami with MCNP
(`cdte_response_v4`, PR #10), whose code was never committed. Section 4 compares the two point by
point.

!!! note "Which BasisSimulator this notebook compares against"
    Every comparison with "BasisSimulator v0.15.1" (its `cdte_response_v4`, its fill factor, its
    one-count cap, its pile-up model's inputs) is with **BasisSimulator v0.15.1**, the last release
    before this notebook's tables. The x-ray spectra and the bowtie come from the **installed**
    BasisSimulator, whatever its version; nothing here pins BasisSimulator, or BasisMonteCarlo,
    to a version. BasisMonteCarlo keeps v0.15.1's table (`basissimulator_v0_15_1_response`), so
    section 4's comparison stays the same after BasisSimulator moves on.

**The parts.** (1) One dexel and its neighbours. (2) One photon's journey. (3) The response of
one dexel. (4) Against BasisSimulator v0.15.1's table. (5) The anti-scatter grid over it, as the
Alpha has one. (6) Photons arriving at an angle. (7) The 2 × 2 binned dexels the scanner reads
out. (8) The whole detector, one to one. (9) The Alpha's x-ray beam. (10) Pile-up at clinical count
rates. (11) A clinical beam, photon by photon. Every section opens with a **Where We Are** band: the
gantry, one dexel's stack with the part under study lit, and the dexel's five steps.
"""

# ╔═╡ df200390-005d-5328-870a-9a821744961d
md"""
## How a Dexel Counts
"""

# ╔═╡ 8326156c-5e6a-5f16-89ab-5e8b268b2e0a
md"""
Steps 1–2 are Geant4: the grid, and where every bit of energy is left in the crystal. Steps 3–5
are the detector's charge and electronics, modelled in Julia on those positions: the cloud of
charge each deposit becomes, the share of it each electrode collects, the noise, and the
thresholds. So the dexel size, the charge cloud, the noise and the thresholds can all be changed
without simulating the photons again. The cathode film, the solder bumps and the board are not
simulated (they do not change where energy is left), and of the chip only its noise, thresholds
and, in section 10, its dead time are modelled.

### What Makes a Dexel

In a scintillator detector such as the SOMATOM Force's, the crystal is sawn into blocks and a
dexel is one block, its reflector and the photodiode under it. The Alpha's CdTe is **one
continuous crystal**: nothing inside it separates one dexel from the next. A dexel is a **pixel
electrode** on the crystal's back face, and the crystal above it; where two electrodes'
territories meet, a cloud of charge can be shared between them (*charge sharing*). The scanner
then reads native dexels singly (ultra-high resolution) or adds them 2 × 2 (standard mode), and
the grid's walls stand around every 4 × 6 native dexels.
"""

# ╔═╡ 22a88d29-5771-4e17-ab70-f33d13c288a8
md"""
### Which Beam, and Why

How the photons arrive decides what a simulation measures, so each section says which beam it
uses, from the simplest to the most realistic:

| beam | what it measures | where |
|---|---|---|
| **uniform on one dexel**, no grid | its response: the counts in it and its eight neighbours, per photon landing on it | sections 1–4 |
| **uniform on one dexel under the grid**, at each of the six positions in a cell | what the walls change, dexel by dexel | sections 5, 7 |
| **tilted along z** at the cone angle | the outer rows | section 6 |
| **from the focal spot** over the whole arc, one energy and the 140 kVp spectrum | everything at once: every dexel, its grid, real angles | sections 8, 9 |
| **clinical spectra, photon by photon**, bare and after 20 and 35 cm of water | whether the single-energy response, weighted by a spectrum, is the clinical beam | section 11 |

There is no pencil beam here: a photon-counting response is per photon landing anywhere on the
dexel, so photons land uniformly over it. The response is built from single energies because
BasisSimulator weights it by whatever spectrum a scan uses; section 11 checks that the weighting
is exact.
"""

# ╔═╡ f7ad4569-b274-5714-83ab-bc7ad0497b53
md"""
## The NAEOTOM Alpha, as Documented

Every hardware number this notebook uses, checked against the FDA 510(k) summaries, Siemens'
own technical documents and peer-reviewed measurements of clinical systems. Where a source and
BasisSimulator v0.15.1 disagree the row is marked **⚠ discrepancy**; the notebook then keeps
BasisSimulator v0.15.1's value, so the two stay comparable, unless the physics being built needs the
documented one (the anti-scatter grid, which BasisSimulator does not model, is built as
documented).

**The family.** Three cleared products share one detector (QuantaMax, CdTe): the original
**NAEOTOM Alpha**, called **Alpha.Peak** on new systems (dual source, 2 × 288 rows, 2 × 57.6 mm),
**Alpha.Pro** (dual source, 2 × 192 rows, 2 × 38.4 mm) and **Alpha.Prime** (single source,
288 rows). Detector A has 2752 channels and a 50 cm field of view, detector B 1984 channels and
36 cm (K211591, K243523).

| quantity | BasisSimulator v0.15.1 | documented | status | source |
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
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator v0.15.1's table
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

# ╔═╡ f3001712-cc0a-48db-8c7e-27e06c5aa3fd
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses (the dexel pitch, the gaps,
the grid cell, the crystal's depth), with a small kit: blocks in 3-D, seen from any direction,
shaded by one light and drawn back to front. The colours are the materials': grey-blue CdTe, a
pale cathode film, gold electrodes, grey tungsten, the dark counting chip, amber x rays, red
charge.
"""

# ╔═╡ 2444710c-f16a-47aa-9ea5-d3cdaa2c6d19
MATERIALS = (
    cdte = Mke.RGBf(0.6, 0.65, 0.76),  # cadmium telluride: a dark, metallic grey-blue
    cdte_lit = Mke.RGBf(0.78, 0.86, 0.98),  # the part of the crystal under one dexel, when marked
    cathode = Mke.RGBf(0.86, 0.87, 0.9),  # the thin metal film on the front face
    electrode = Mke.RGBf(0.86, 0.66, 0.2),  # the pixel electrodes (anodes): gold
    bump = Mke.RGBf(0.75, 0.75, 0.78),  # the solder bumps to the chip
    asic = Mke.RGBf(0.2, 0.22, 0.27),  # the counting chip
    board = Mke.RGBf(0.13, 0.4, 0.29),  # circuit board
    tungsten = Mke.RGBf(0.42, 0.44, 0.5),
    water = Mke.RGBf(0.62, 0.78, 0.92),
    beam = Mke.RGBf(0.98, 0.7, 0.25),  # x rays, as in the lab's banner
    charge = Mke.RGBf(0.7, 0.12, 0.17),  # the freed electrons
    night = Mke.RGBf(0.05, 0.06, 0.08),  # the dark backdrop of the lab's banner
    night_ring = Mke.RGBf(0.11, 0.13, 0.16),
)

# ╔═╡ 18913250-92bd-415e-9559-55e73934e2eb
begin
    """A block in 3-D (axis-aligned, `lo` to `hi` in mm), and how to draw it."""
    Base.@kwdef struct Block
        lo::NTuple{3, Float64}
        hi::NTuple{3, Float64}
        color::Mke.RGBf
        alpha::Float64 = 1.0
        stroke::Float64 = 0.6
    end

    Block(lo, hi, color; kw...) = Block(; lo = Float64.(Tuple(lo)), hi = Float64.(Tuple(hi)), color, kw...)
end

# ╔═╡ de754d25-442e-4d20-8565-4e0b9a502db7
"""
An orthographic camera: seen from `azimuth` (degrees, about the vertical axis z) and `elevation`
(degrees above the horizontal). Returns the screen's right and up and the direction toward the viewer.
"""
function camera(azimuth, elevation)
    θ, φ = deg2rad(azimuth), deg2rad(elevation)
    toward = (cos(φ) * cos(θ), cos(φ) * sin(θ), sin(φ))
    right = (-sin(θ), cos(θ), 0.0)
    up = (
        toward[2] * right[3] - toward[3] * right[2],
        toward[3] * right[1] - toward[1] * right[3],
        toward[1] * right[2] - toward[2] * right[1],
    )
    return (; right, up, toward)
end

# ╔═╡ dc842efe-90d4-435c-a6de-fc2375005697
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ 72cb803c-129a-4891-ac42-8308cc4ce50a
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ ee6c802f-d1f0-432e-bae6-166cf16320d3
"""The six faces of a block: outward normal and corners."""
function faces(b::Block)
    (x0, y0, z0), (x1, y1, z1) = b.lo, b.hi
    return (
        ((0, 0, 1), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0, 0, -1), ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))),
        ((1, 0, 0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1, 0, 0), ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))),
        ((0, 1, 0), ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))),
        ((0, -1, 0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
    )
end

# ╔═╡ 792826ce-b8c6-4fbd-9919-8c719d716f49
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ eff17814-a2f6-4d61-a9cd-df7458f5b5b0
"""
Draw `blocks` back to front (the painter's way), each with its visible faces shaded, and any
`extras` (functions drawing lines or arrows, with their depth) in their place among them.
Blocks should be of similar size (split long walls into pieces) for the order to be right.
"""
function draw_blocks!(ax, cam, blocks; extras = [], edge = Mke.RGBf(0.1, 0.12, 0.16), edge_alpha = 0.35)
    items = Any[(dot3(map(+, b.lo, b.hi) ./ 2, cam.toward), b) for b in blocks]
    append!(items, extras)
    sort!(items; by = first)
    polygons = Vector{Mke.Point2f}[]
    colors = Mke.RGBAf[]
    strokes = Mke.RGBAf[]
    function flush!()
        isempty(polygons) && return
        Mke.poly!(
            ax,
            copy(polygons);
            color = copy(colors),
            strokecolor = copy(strokes),
            strokewidth = 0.7,
        )
        empty!(polygons)
        empty!(colors)
        return empty!(strokes)
    end
    for (_, item) in items
        if item isa Block
            for (normal, corners) in faces(item)
                dot3(normal, cam.toward) > 1.0e-9 || continue
                push!(polygons, [project(cam, c) for c in corners])
                c = shade(item.color, normal, cam)
                push!(colors, Mke.RGBAf(c.r, c.g, c.b, item.alpha))
                # a seam between tiles of one layer is drawn in the face's own colour
                push!(
                    strokes,
                    item.stroke > 0 ? Mke.RGBAf(edge.r, edge.g, edge.b, edge_alpha * item.alpha) :
                        Mke.RGBAf(c.r, c.g, c.b, item.alpha),
                )
            end
        else
            flush!()
            item(ax)
        end
    end
    flush!()
    return ax
end

# ╔═╡ 896c9200-6700-48f4-beb2-cffee0c1ec0f
"""
A label with a leader line: a dot at `anchor`, a line to `at`, and the text there, on a
translucent rounded background.
"""
function callout!(
        ax,
        anchor,
        at,
        text;
        color = Mke.RGBf(0.13, 0.13, 0.16),
        align = (:left, :center),
        fontsize = 17,
        background = (:white, 0.9),
        line = (Mke.RGBf(0.3, 0.3, 0.35), 0.8),
        font = :regular,
    )
    Mke.lines!(ax, Mke.Point2f[anchor, at]; color = line, linewidth = 1.3)
    Mke.scatter!(ax, [anchor]; color = line, markersize = 7)
    return Mke.textlabel!(
        ax,
        at;
        text,
        text_align = align,
        text_color = color,
        fontsize,
        font,
        background_color = background,
        strokewidth = 0,
        padding = (8, 8, 5, 5),
        cornerradius = 6,
    )
end

# ╔═╡ e6249cb2-6672-4f2c-956d-a4e947fa0d07
"""A blank canvas for an illustration: equal scales, no axes."""
function canvas(position; background = :white, title = "", kw...)
    ax = Mke.Axis(
        position;
        aspect = Mke.DataAspect(),
        backgroundcolor = background,
        title,
        kw...,
    )
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    return ax
end

# ╔═╡ ed2d4d8c-1329-47a4-841f-7affbc4d204f
"""
A photon drawn as a wave from `a` to `b` (2-D), with `cycles` oscillations (more for a higher
energy) and an arrowhead at `b`.
"""
function photon!(ax, a, b; color = MATERIALS.beam, cycles = 6, amplitude = 0.06, linewidth = 3, head = 14)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    d = b - a
    L = sqrt(d[1]^2 + d[2]^2)
    t = d / L
    n = Mke.Point2f(-t[2], t[1])
    s = range(0, 1; length = 240)
    ramp(u) = min(1, 8u, 8(1 - u))  # the wave starts and ends on the line
    points = Mke.Point2f[a + u * d + amplitude * L * ramp(u) * sin(2π * cycles * u) * n for u in s]
    Mke.lines!(ax, points[1:(end - 6)]; color, linewidth)
    Mke.scatter!(
        ax,
        [b];
        marker = :utriangle,
        markersize = head,
        rotation = atan(t[2], t[1]) - π / 2,
        color,
    )
    return nothing
end

# ╔═╡ 1d5e84c1-714b-47d6-a120-71068b37aebc
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ dd76ac67-b4fa-4c94-bdf6-4b431f3d14a4
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ fe495bf4-f87f-45c4-b1ab-16c586bc4df5
"""
Spread labels down the right of a drawing: each `entry` is `(; anchor, title, detail, color, tag)`
(`tag` is `nothing` or `(text, color)`). Labels keep at least `spacing` apart (data units), in the
order of their anchors, and a bent leader line runs from each anchor to its label at `x`.
"""
function labels_right!(ax, entries, x; spacing = 1.0, tag_width = 0.0, title_size = 19, detail_size = 15.5)
    order = sortperm([e.anchor[2] for e in entries]; rev = true)
    ys = Float64[]
    for k in order
        y = entries[k].anchor[2]
        isempty(ys) || (y = min(y, last(ys) - spacing))
        push!(ys, y)
    end
    # centre the column of labels on the anchors
    shift = (sum(e.anchor[2] for e in entries) - sum(ys)) / length(ys)
    ys .+= max(0, shift)
    for (k, y) in zip(order, ys)
        e = entries[k]
        elbow = Mke.Point2f(x - 0.35 * spacing, y)
        Mke.lines!(ax, Mke.Point2f[e.anchor, Mke.Point2f(elbow[1] - 0.3 * spacing, e.anchor[2]), elbow, Mke.Point2f(x, y)]; color = (:black, 0.32), linewidth = 1.2)
        Mke.scatter!(ax, [Mke.Point2f(e.anchor)]; color = (:black, 0.55), markersize = 7)
        at = Mke.Point2f(x, y)
        if e.tag !== nothing
            Mke.textlabel!(ax, at; text = e.tag[1], text_align = (:left, :center), text_color = :white, font = :bold, fontsize = 13, offset = (6, 0), background_color = e.tag[2], strokewidth = 0, padding = (8, 8, 3, 3), cornerradius = 9)
        end
        dx = e.tag === nothing ? 6 : 6 + tag_width
        Mke.text!(ax, at; text = e.title, font = :bold, fontsize = title_size, align = (:left, :bottom), offset = (dx, 1), color = e.color)
        e.detail == "" || Mke.text!(ax, at; text = e.detail, fontsize = detail_size, align = (:left, :top), offset = (dx, -3), color = Mke.RGBf(0.42, 0.42, 0.46))
    end
    return nothing
end

# ╔═╡ a05ae43e-08a8-4696-b931-100fe6023d33
"""A vertical bracket at `x` from `y0` to `y1`, with `text` to its left."""
function side_bracket!(ax, x, y0, y1, text; color, fontsize = 16, tick = 0.08)
    Mke.lines!(ax, Mke.Point2f[(x + tick, y0), (x, y0), (x, y1), (x + tick, y1)]; color, linewidth = 2)
    return Mke.text!(ax, x - tick, (y0 + y1) / 2; text, color, fontsize, font = :bold, align = (:right, :center), justification = :right)
end

# ╔═╡ a2be090c-6854-4fa7-90ce-5494da5bd57c
"""A small rounded tag ("In Geant4" / "Not Simulated")."""
function tag!(ax, at, text, color; align = (:left, :center), fontsize = 14, offset = (0, 0))
    return Mke.textlabel!(
        ax,
        at;
        text,
        text_align = align,
        text_color = :white,
        font = :bold,
        fontsize,
        offset,
        background_color = color,
        strokewidth = 0,
        padding = (8, 8, 3, 3),
        cornerradius = 9,
    )
end

# ╔═╡ cf02ddf8-fcef-413d-a30d-d51071fdc761
TAGS = (
    geant4 = Mke.RGBf(0.2, 0.55, 0.32),
    julia = Mke.RGBf(0.16, 0.38, 0.68),
    not_simulated = Mke.RGBf(0.52, 0.52, 0.57),
)

# ╔═╡ 2a420206-70e3-4019-945c-f628bd6ee0df
"""A block drawn faded toward the background (`amount` 0: as is, 1: gone)."""
function faded(b::Block, amount; to = Mke.RGBf(1, 1, 1))
    mix(a, c) = a + amount * (c - a)
    color = Mke.RGBf(mix(b.color.r, to.r), mix(b.color.g, to.g), mix(b.color.b, to.b))
    return Block(; lo = b.lo, hi = b.hi, color, alpha = b.alpha, stroke = b.stroke)
end

# ╔═╡ 290c44b9-29d5-4d71-ae2b-b14fa5fc02d5
"""The convex hull of points (2-D), counter-clockwise, closed."""
function hull(points)
    pts = sort(unique(Mke.Point2f.(points)); by = p -> (p[1], p[2]))
    length(pts) < 3 && return pts
    cross2(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    lower, upper = Mke.Point2f[], Mke.Point2f[]
    for p in pts
        while length(lower) >= 2 && cross2(lower[end - 1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    for p in reverse(pts)
        while length(upper) >= 2 && cross2(upper[end - 1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    return vcat(lower[1:(end - 1)], upper[1:(end - 1)], lower[1:1])
end

# ╔═╡ c7ef0930-55ad-4a4f-8296-db3571c54a88
"""Outline a box (lo, hi) as seen by `cam`: the edge of its silhouette."""
function outline!(ax, cam, lo, hi; color = MATERIALS.beam, linewidth = 3, linestyle = :solid)
    corners = [(x, y, z) for x in (lo[1], hi[1]) for y in (lo[2], hi[2]) for z in (lo[3], hi[3])]
    return Mke.lines!(ax, hull([project(cam, c) for c in corners]); color, linewidth, linestyle)
end

# ╔═╡ 06101f43-57fd-44a5-b7aa-cd22bfaa5eb1
"""A curved strip (part of a cylinder about the z axis at the focal spot), as quads with depth, for the painter."""
function arc_quads(cam, R, half_angle, half_z; columns = 46, thickness = 18.0, color = MATERIALS.ufc)
    quads = Any[]
    for k in 1:columns
        a0 = -half_angle + (k - 1) * 2half_angle / columns
        a1 = a0 + 2half_angle / columns
        for (r, normal_sign) in ((R, -1), (R + thickness, 1))
            corners = [(r * sin(a), -r * cos(a), z) for (a, z) in ((a0, -half_z), (a1, -half_z), (a1, half_z), (a0, half_z))]
            mid = (a0 + a1) / 2
            n = (normal_sign * sin(mid), -normal_sign * cos(mid), 0.0)
            depth = dot3(corners[1] .+ corners[3], cam.toward) / 2
            push!(
                quads, (
                    depth, ax -> begin
                        dot3(n, cam.toward) > 0 || return
                        c = shade(color, n, cam)
                        Mke.poly!(ax, [project(cam, q) for q in corners]; color = c, strokecolor = (:black, 0.25), strokewidth = 0.5)
                    end,
                )
            )
        end
    end
    return quads
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
    transport = nothing,  # the fixed cloud, not the transported charge (the default)
)

# ╔═╡ c9a17736-48ea-5cf4-acdd-39000bd8e7ae
# Photons land uniformly over the central dexel
beam = MC.AreaBeam(;
    width_mm = readout.pitch_x_mm,
    height_mm = readout.pitch_y_mm,
)

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
        title = "What Each Dexel Collects from $(round(Int, E0)) keV Photons Landing on the Middle One",
        xlabel = "Energy Collected by the Dexel (keV)",
        ylabel = "Per keV, per Photon",
        yscale = log10,
        limits = ((0, E0 + 3), (1.0e-5, 3)),
    )
    for (dexel, label, color) in (
            ((0, 0), "Middle Dexel", PALETTE.photon),
            ((1, 0), "Side Neighbour (Along X)", PALETTE.xray),
            ((0, 1), "Side Neighbour (Along Y)", PALETTE.kept),
            ((1, 1), "Corner Neighbour", PALETTE.muted),
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
        "Split Charge Clouds: Anything from 0 to E";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )
    Mke.axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 043f6510-e50d-4d4c-b343-8fee4e5918f3
md"""
### How the Charge Travels

Up to here every deposit became one Gaussian cloud **30 µm** wide, whatever its depth or energy:
BasisSimulator v0.15.1's model, kept so that section 4 compares like with like. The real charge
does more on its way to the electrodes, and all of it can be computed from where Geant4 left
the energy (each deposit's depth is recorded):

- **Drift.** The bias, 800–1000 V across the 1.6 mm (Flohr & Schmidt 2023; 900 V here), pulls the
  electrons to the pixel electrodes at the back and the holes to the cathode at the front.
- **Diffusion.** On the way the electron cloud spreads: σ² = 2 (kT/q) d / E for a drift of d at
  field E. The mobility cancels, so it depends only on how deep the photon was absorbed: about
  12 µm from the front face, nothing from the back.
- **Repulsion.** The electrons of one cloud push each other apart: a uniform sphere grows as
  R³ = 3 Q d / (4π ε E) (Gatti et al. 1987), about 10 µm for 60 keV. Deposits within 50 µm of
  each other count as one cloud; a characteristic x ray absorbed further away is its own.
- **The photoelectron's track.** Geant4 follows every electron in 1 µm steps, so the track's own
  length is already in where the deposits are: nothing is added for it.
- **Induction.** A dexel's signal is the charge the moving carriers induce on its electrode
  (Shockley–Ramo): the electron's charge times the change of the dexel's **weighting potential**
  along its path. For electrodes much smaller than the crystal is thick that potential is close to
  zero except right at the electrode (the "small-pixel effect"), so the electrons, arriving, make
  almost all of the signal, and the slow holes very little.
- **Trapping.** Electrons and holes are lost on the way with mean drift lengths λ = μτE (Hecht):
  μτ = 3.3 × 10⁻³ cm²/V for electrons and 2 × 10⁻⁴ cm²/V for holes in CdTe (Del Sordo et al.
  2009), λ ≈ 19 cm and 1.1 cm at 900 V: a little signal is lost, mostly for photons absorbed near
  the back.

What is not modelled: the electrode's own gaps and the field's bending around them (charge in a gap
goes to the nearer dexel, as before), polarization of the crystal at very high flux, and the
pulse shaper's finite time (the induced charge is taken whole).
"""

# ╔═╡ 9950b293-c447-457b-bd7b-5fa3d4a8eddd
# The charge transport in the Alpha's CdTe (BasisMonteCarlo's `ChargeTransport`; these are its
# defaults, written out with the source of every number)
charge_transport = MC.ChargeTransport(;
    bias_V = 900.0,  # Flohr & Schmidt 2023: 800–1000 V across the crystal
    thickness_mm = 1.6,
    mu_tau_electron = 3.3e-3,  # cm²/V, CdTe (Del Sordo et al. 2009, Sensors 9:3491)
    mu_tau_hole = 2.0e-4,  # cm²/V, same source
    thermal_voltage = 0.02585,  # kT/q at 300 K, V
    relative_permittivity = 10.9,  # CdTe
    pair_energy_eV = 4.43,  # one electron–hole pair per 4.43 eV, as in the threshold model
    cluster_mm = 0.05,  # deposits closer than this repel each other as one cloud
)

# ╔═╡ 8e06edcd-ec4f-4332-a291-921130d544c2
# Zhan et al. 2026 (PMC13691410), fitted to clinical NAEOTOM Alpha scans of PMMA, 70–140 kV: a
# Gaussian charge-spreading kernel of standard deviation r_e and Gaussian electronic noise σ_e,
# with Fano factor 0.1; across their subsets and bootstrap groups r_e ranged about 14–17 µm
zhan = (
    sigma_um = 16.4,
    sigma_range_um = (14.0, 17.0),
    noise_keV = 1.9,
    fano = 0.1,
)

# ╔═╡ 29622b50-54f1-4b0c-a988-724a4732970d
import Distributed

# ╔═╡ 88e2e00a-8c74-4c9e-ac31-88663466235e
let
    Mke.set_theme!(THEME)
    p = charge_transport
    L = p.thickness_mm
    λ = MC.transport_lengths(p)
    t = MC.induction_table(readout.pitch_x_mm, readout.pitch_y_mm, p)
    z = collect(range(0.0, L; length = 161))
    fig = Mke.Figure(size = (1600, 560))

    # (a) How wide the cloud arrives, by the depth it started from
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) How Wide the Charge Arrives",
        xlabel = "Depth Where the Photon Was Absorbed (mm)",
        ylabel = "Width of the Cloud, σ (µm)",
        limits = ((0, L), (0, 34)),
    )
    Mke.hspan!(axa, zhan.sigma_range_um...; color = (PALETTE.kept, 0.12))
    note!(axa, 0.05, zhan.sigma_range_um[2], "Zhan 2026's Fit: $(zhan.sigma_range_um[1])–$(zhan.sigma_range_um[2]) µm"; align = (:left, :bottom), color = PALETTE.kept, fontsize = 14)
    Mke.hlines!(axa, 30.0; color = PALETTE.reference, linestyle = :dash, linewidth = 2)
    note!(axa, L - 0.02, 30.0, "BasisSimulator v0.15.1: 30 µm"; align = (:right, :bottom), color = PALETTE.reference, fontsize = 14)
    Mke.lines!(axa, z, [1000MC.diffusion_sigma(L - zz, p) for zz in z]; color = PALETTE.muted, linewidth = 2.5, linestyle = :dot, label = "Diffusion Alone")
    for (E, colour) in ((30.0, PALETTE.photon), (60.0, MATERIALS.beam), (120.0, PALETTE.electron))
        Mke.lines!(axa, z, [1000sqrt(MC.diffusion_sigma(L - zz, p)^2 + MC.repulsion_sigma(E, L - zz, p)^2) for zz in z]; color = colour, linewidth = 3, label = "With Repulsion, $(Int(E)) keV")
    end
    Mke.axislegend(axa; position = :lb)

    # (b) The weighting potential of one dexel, along three vertical lines
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) One Dexel's Weighting Potential",
        xlabel = "Depth from the Front (mm)",
        ylabel = "Weighting Potential",
        limits = ((0, L), (-0.02, 1.02)),
    )
    for (ox, label, colour) in ((0.0, "Above Its Centre", PALETTE.kept), (readout.pitch_x_mm / 2, "Above Its Edge", MATERIALS.beam), (readout.pitch_x_mm, "Above the Neighbour's Centre", PALETTE.muted))
        Mke.lines!(axb, z, [MC.induction_lookup(t, t.φ, ox, 0.0, zz) for zz in z]; color = colour, linewidth = 3, label)
    end
    Mke.lines!(axb, z, z ./ L; color = PALETTE.ink, linestyle = :dash, linewidth = 1.5, label = "A Whole-Face Electrode: z / L")
    Mke.axislegend(axb; position = :lt)

    # (c) The charge the middle dexel records, by depth, with trapping
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) The Charge the Middle Dexel Records",
        xlabel = "Depth Where the Photon Was Absorbed (mm)",
        ylabel = "Share of the Freed Charge",
        limits = ((0, L), (0.9, 1.005)),
    )
    induced(ox, zz, f) = exp(-(L - zz) / λ.electron) * f + MC.induction_lookup(t, t.ge, ox, 0.0, zz) + MC.induction_lookup(t, t.gh, ox, 0.0, zz)
    Mke.lines!(axc, z, [induced(0.0, zz, 1.0) for zz in z]; color = PALETTE.kept, linewidth = 3, label = "Absorbed Above It: All Arrives There")
    Mke.lines!(axc, z, [λ.electron / L * (1 - exp(-(L - zz) / λ.electron)) + λ.hole / L * (1 - exp(-zz / λ.hole)) for zz in z]; color = PALETTE.ink, linestyle = :dash, linewidth = 1.5, label = "Hecht: a Whole-Face Electrode")
    note!(axc, 0.75, 0.935, "Holes Trapped on Their Way to the Front:\nThe Small Dexel Hardly Feels Them,\na Whole-Face Electrode Would"; align = (:center, :center), fontsize = 14, color = PALETTE.photon)
    Mke.axislegend(axc; position = :lb)
    fig
end

# ╔═╡ 44c8451f-2fdf-4a30-b09c-0a0bf51f7ad2
# The induction model's own check: summed over every dexel, the charge a deposit induces is the
# Hecht relation's collection efficiency (a 41 × 41 array of dexels, computed directly)
let
    p = charge_transport
    L = p.thickness_mm
    λ = MC.transport_lengths(p)
    wx, wy = readout.pitch_x_mm, readout.pitch_y_mm
    zs = range(0.0, L; length = 161)
    h = step(zs)
    rows = String[]
    worst = 0.0
    for z in (0.1, 0.4, 0.8, 1.2, 1.5)
        total = 0.0
        for i in -20:20, j in -20:20
            φ(zz) = MC.weighting_potential(0.03 - i * wx, 0.05 - j * wy, L - zz, wx, wy, L)
            arrival = (i == 0 && j == 0) ? exp(-(L - z) / λ.electron) : 0.0
            # electrons from z up to L, holes from z down to 0, each weighted by the share not yet trapped
            above = [zz for zz in zs if zz >= z]
            below = reverse([zz for zz in zs if zz <= z])
            electrons = arrival - φ(z) + sum(exp(-(zz - z) / λ.electron) * φ(zz) for zz in above) * h / λ.electron
            holes = φ(z) - sum(exp(-(z - zz) / λ.hole) * φ(zz) for zz in below) * h / λ.hole
            total += electrons + holes
        end
        hecht = λ.electron / L * (1 - exp(-(L - z) / λ.electron)) + λ.hole / L * (1 - exp(-z / λ.hole))
        worst = max(worst, abs(total - hecht))
        push!(rows, "| $(z) mm | $(round(total; digits = 4)) | $(round(hecht; digits = 4)) |")
    end
    Markdown.parse(
        """
        | depth from the front | induced, summed over 41 × 41 dexels | Hecht's collection efficiency |
        |---|---|---|
        $(join(rows, "\n"))

        The induction model and the Hecht relation agree to $(round(worst; digits = 4)) at every depth (the rest is the sum's
        finite steps and the dexels beyond the 41 × 41): the weighting potential, the trapping and the bookkeeping are
        consistent. About $(round(100 * (1 - λ.hole / L * (1 - exp(-L / λ.hole))); digits = 1)) % of the charge is lost for a photon absorbed at the back, almost nothing at the front.
        """
    )
end

# ╔═╡ 6eaa219f-a038-4754-a84a-692db7826816
md"""
Section 3's photons again: the same energies, the same seeds, so the very same photons. This time
each run's deposits come back from the worker that simulated it, and the charge is transported
here. Two things are computed beside it from the same deposits: the fixed 30 µm cloud again, which
must give exactly section 3's responses (a check of the bookkeeping), and, every 5 keV, fixed
clouds from 6 to 30 µm, to find the one that counts like the transported charge (Zhan's model,
below).
"""

# ╔═╡ 7ce20715-234a-4095-a100-0206821b7cfd
md"""
### Checked Against the Clinical Alpha

Zhan et al. (2026) fitted a detector model to clinical NAEOTOM Alpha scans: PMMA from 0 to 13
inches at 70, 90, 120 and 140 kV, the ratio of counts above 20 keV to counts above a higher
threshold. Their model spreads each interaction's charge with **one Gaussian of standard deviation
r_e**, whatever its depth or energy, and adds Gaussian electronic noise σ_e. The fit gives
**r_e = 16.4 µm** (14–17 µm across their subsets and bootstrap groups) and **σ_e = 1.9 keV**.

Their r_e lumps together everything that spreads the charge, so the comparison is made their way:
the single Gaussian width whose counts above all nine thresholds come closest to the transported
charge's (least squares), at every energy and over the Alpha's 140 kV beams, with their electronic
noise. A high threshold matters: counts above 20 keV alone barely depend on the width over a whole
spectrum (a wider cloud loses low-energy photons and double-counts high-energy ones), which is why
Zhan fitted a ratio to a higher threshold. Nothing in the transport model was fitted to them.
"""

# ╔═╡ 65f1babc-4849-5e3a-8404-738562c88837
md"""
## 4. Compared with BasisSimulator v0.15.1's Table

Up to and including v0.15.1, BasisSimulator's photon-counting detector model uses `cdte_response_v4`: for photons of each
energy from 1 to 140 keV landing on a dexel, the mean number of counts above each of nine
thresholds (20, 25, 30, 35, 55, 60, 70, 75, 90 keV), in the 3 × 3 neighbourhood and per dexel,
from MCNP with a 30 µm Gaussian charge cloud and 1.5 keV electronic noise. Its file does not
record the dexel size or crystal depth; the native 0.275 × 0.322 mm dexels and 1.6 mm of CdTe
used here are the geometry it matches (the 2 × 2 binned dexels, 0.55 × 0.644 mm, do not).

The same thresholds and the same noise are applied to the Geant4 dexel signals below.
"""

# ╔═╡ 0eff78e6-df72-522a-b5db-d70421a6cdec
# BasisSimulator v0.15.1's table (`cdte_response_v4`), as BasisMonteCarlo keeps it for comparison
current_table = MC.basissimulator_v0_15_1_response()

# ╔═╡ 0c8a2fcc-ca65-4040-9b13-973725659b2e
# Is BasisMonteCarlo's copy the table the installed BasisSimulator ships? Up to v0.15.1 it must be,
# byte for byte; a later release may ship this notebook's table instead, and the copy stands alone
let
    installed = pkgversion(BS)
    kept = joinpath(pkgdir(MC), "src", "data", "basissimulator_v0_15_1_cdte_response_v4.jls")
    shipped = isdefined(BS, :default_mc_drm_path) ? BS.default_mc_drm_path() : ""
    same = isfile(shipped) && read(shipped) == read(kept)
    message = if installed <= v"0.15.1"
        same ?
            "✓ The installed BasisSimulator, v$(installed), ships exactly this table." :
            "✗ **The installed BasisSimulator, v$(installed), ships a different table from BasisMonteCarlo's copy of v0.15.1's `cdte_response_v4`.** Check the copy against BasisSimulator v0.15.1."
    else
        "The installed BasisSimulator is v$(installed), which no longer needs this table. The comparisons below are with BasisSimulator v0.15.1's `cdte_response_v4`, which BasisMonteCarlo keeps (`basissimulator_v0_15_1_response`): they do not change with BasisSimulator's version."
    end
    Markdown.parse(message)
end

# ╔═╡ 4e507ffb-04ca-5f56-a577-e2715de205a4
thresholds = current_table["thresholds_keV"]

# ╔═╡ d771059c-df37-4346-a774-fa326eb07309
# Each energy's run, deposits included, simulated on a free worker; the charge transported here
transport_runs = let
    pool = Distributed.WorkerPool(workers)
    scan_energies = 15.0:5.0:140.0
    scan_sigmas = 6.0:2.0:30.0
    fixed_cloud(σ) = MC.DexelReadout(;
        pitch_x_mm = readout.pitch_x_mm,
        pitch_y_mm = readout.pitch_y_mm,
        neighborhood = 3,
        cloud_sigma_um = σ,
        transport = nothing,
    )
    transported_readout = MC.DexelReadout(;
        pitch_x_mm = readout.pitch_x_mm,
        pitch_y_mm = readout.pitch_y_mm,
        neighborhood = readout.neighborhood,
        transport = charge_transport,
    )
    results = asyncmap(collect(energies); ntasks = length(workers)) do E
        run = Distributed.remotecall_fetch(
            MC.simulate_monoenergetic,
            pool,
            crystal,
            E;
            n = 100_000,
            beam,
            seed = 1,
            record_deposits = true,
        )
        transported = MC.dexel_response(run, transported_readout)
        k = findfirst(==(E), energies)
        reproduces = MC.dexel_response(run, readout).dexel_energy_keV == responses[k].dexel_energy_keV
        scan = E in scan_energies ?
            [
                MC.threshold_counts(MC.dexel_response(run, fixed_cloud(σ)), thresholds; electronic_noise_keV = zhan.noise_keV).total
                for σ in scan_sigmas
            ] :
            nothing
        (; transported, reproduces, scan)
    end
    (
        responses = [r.transported for r in results],
        reproduces = all(r.reproduces for r in results),
        scan_energies,
        scan_sigmas,
        scan = Dict(E => r.scan for (E, r) in zip(energies, results) if r.scan !== nothing),
    )
end

# ╔═╡ 1c1f1e65-29b9-4154-b499-ad21f8a26d31
# Counts per threshold with the transported charge and Zhan's electronic noise
transport_counts = [
    MC.threshold_counts(r, thresholds; electronic_noise_keV = zhan.noise_keV)
        for r in transport_runs.responses
]

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
        title = "Counts per Photon in the 3 × 3 Dexels, Above Each Threshold",
        ylabel = "Counts per Photon",
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
        "Lines: Geant4 (This Notebook)    Dots: BasisSimulator v0.15.1 (cdte_response_v4, MCNP)";
        align = (:right, :top),
        fontsize = 15,
    )
    note!(
        ax,
        12,
        1.1,
        "Above 1: Charge Sharing Makes One\nPhoton Count in Two Dexels";
        align = (:left, :center),
        fontsize = 14,
        color = PALETTE.muted,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center, title = "Threshold")

    axd = Mke.Axis(
        fig[2, 1];
        xlabel = "Photon Energy (keV)",
        ylabel = "Geant4 − MCNP\n(Counts per Photon)",
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
        "Grey Band: ±0.01 Counts per Photon";
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

# ╔═╡ 7b539e6e-ee97-4464-a2c5-f09abe2a32f7
"""A box cut into about-dexel-sized tiles of one colour (so the painter's order holds), seams hidden."""
function tiles(lo, hi, color; size = alpha.dexel)
    nx = max(1, round(Int, (hi[1] - lo[1]) / size[1]))
    ny = max(1, round(Int, (hi[2] - lo[2]) / size[2]))
    dx, dy = (hi[1] - lo[1]) / nx, (hi[2] - lo[2]) / ny
    return [
        Block((lo[1] + i * dx, lo[2] + j * dy, lo[3]), (lo[1] + (i + 1) * dx, lo[2] + (j + 1) * dy, hi[3]), color; stroke = 0.0)
            for i in 0:(nx - 1) for j in 0:(ny - 1)
    ]
end

# ╔═╡ 26699732-4c76-5ecd-ae55-ed5aff70d36f
cell_pitch = (
    alpha.group[1] * alpha.dexel[1] + alpha.gap[1],
    alpha.group[2] * alpha.dexel[2] + alpha.gap[2],
)

# ╔═╡ 246bff85-ebee-4478-a199-b432c91f57ad
"""
Where native dexel `(k, l)` sits on the detector's face (its lower-left corner, mm), counted from
the corner of a grid cell: dexels come in groups of 4 × 6, each group inside one grid cell, half a
gap in from the cell's edge.
"""
dexel_corner(k, l) = (
    (k ÷ alpha.group[1]) * cell_pitch[1] + alpha.gap[1] / 2 + (k % alpha.group[1]) * alpha.dexel[1],
    (l ÷ alpha.group[2]) * cell_pitch[2] + alpha.gap[2] / 2 + (l % alpha.group[2]) * alpha.dexel[2],
)

# ╔═╡ f6b40d89-2b55-4c5e-b8e7-881636d05f6e
let
    Mke.set_theme!(THEME)
    p, g = alpha.dexel, alpha.gap
    d = crystal.thickness_mm
    fig = Mke.Figure(size = (1450, 820))
    ax = canvas(fig[1, 1])

    # One row of a grid cell, side view (x across, up toward the tube): four dexels and the walls
    xL, xR = 0.0, cell_pitch[1]
    y_board, y_asic, y_el = -1.35, -0.95, -0.12
    rect!(ax, xL, y_board, xR - xL, 0.28, MATERIALS.board)
    rect!(ax, xL, y_asic, xR - xL, 0.52, MATERIALS.asic)
    rect!(ax, xL, 0, xR - xL, d, MATERIALS.cdte; strokecolor = (:black, 0.25), strokewidth = 1)
    rect!(ax, xL, d, xR - xL, 0.05, MATERIALS.cathode; strokecolor = (:black, 0.3), strokewidth = 0.8)
    for k in 0:3
        x = dexel_corner(k, 0)[1]
        rect!(ax, x + 0.02, y_el, p[1] - 0.04, 0.1, MATERIALS.electrode)
        rect!(ax, x + p[1] / 2 - 0.035, y_asic + 0.52, 0.07, y_el - (y_asic + 0.52), MATERIALS.bump)
    end
    for x in (xL, xR)
        rect!(ax, x - g[1] / 2, d + 0.05, g[1], 1.3, MATERIALS.tungsten)
    end

    # (1) the x ray, (2) absorbed, (3) the charge drifts and spreads, (4) two electrodes share it
    hit = Mke.Point2f(dexel_corner(1, 0)[1] + 0.92p[1], d - 0.35)
    photon!(ax, (hit[1], d + 2.2), hit; cycles = 9, amplitude = 0.02, color = MATERIALS.beam, linewidth = 3.5)
    Mke.scatter!(ax, [hit]; marker = :star5, markersize = 28, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.5)
    for (k, z) in enumerate(range(hit[2] - 0.15, 0.08; length = 5))
        spread = 0.012 + 0.02k
        xs = hit[1] .+ spread .* [-1.6, -0.8, 0.0, 0.8, 1.6]
        Mke.scatter!(ax, xs, fill(z, 5); color = (MATERIALS.charge, 0.35 + 0.1k), markersize = 6)
    end
    arrow!(ax, (hit[1] - 0.2, hit[2] - 0.1), (hit[1] - 0.2, 0.25); color = MATERIALS.charge, linewidth = 2, head = 10)
    edge = dexel_corner(2, 0)[1]
    Mke.lines!(ax, [edge, edge], [-0.12, d]; color = (PALETTE.ink, 0.35), linestyle = :dash, linewidth = 1.5)

    # (5) the chip: the shared cloud's two pulses, against the four thresholds
    for (k, height) in ((1, 0.62), (2, 0.38))
        x = dexel_corner(k, 0)[1] + 0.05
        base = y_asic + 0.06
        Mke.lines!(ax, Mke.Point2f[(x, base), (x + 0.06, base + 0.4height), (x + 0.11, base), (x + 0.2, base)]; color = :white, linewidth = 2)
        for (t, level) in enumerate((0.2, 0.35, 0.55, 0.7))
            on = height >= level
            Mke.lines!(ax, [x - 0.01, x + 0.2], fill(base + 0.4level, 2); color = (on ? MATERIALS.beam : :white, on ? 0.9 : 0.25), linewidth = 1)
        end
    end

    entries = [
        (anchor = (hit[1], d + 1.6), title = "1  An X Ray Arrives", detail = "Through the Open Middle of a Grid Cell", color = Mke.RGBf(0.85, 0.5, 0.05), tag = nothing),
        (anchor = hit, title = "2  The CdTe Absorbs It", detail = "Freeing Electron–Hole Pairs in Proportion to Its Energy", color = Mke.RGBf(0.7, 0.55, 0.05), tag = nothing),
        (anchor = (hit[1] - 0.2, 0.6), title = "3  The Charge Drifts and Spreads", detail = "Pulled by 800–1000 V to the Back; About 10–20 µm Wide on Arrival", color = MATERIALS.charge, tag = nothing),
        (anchor = (edge, y_el + 0.05), title = "4  The Electrodes Collect It", detail = "Each Takes the Part Above It: a Cloud on an Edge Is Shared", color = Mke.RGBf(0.7, 0.5, 0.1), tag = nothing),
        (anchor = (dexel_corner(2, 0)[1] + 0.1, y_asic + 0.3), title = "5  Thresholds Count It", detail = "Each Pulse Against 20, 35, 55, 70 keV:\nOne Count per Threshold It Crosses", color = PALETTE.ink, tag = nothing),
    ]
    x_text = xR + 0.55
    labels_right!(ax, entries, x_text; spacing = 0.55)
    Mke.text!(ax, xR + g[1] / 2 + 0.05, d + 1.3; text = "Grid Walls\n(Cut Short)", fontsize = 14, color = PALETTE.muted, align = (:left, :top))

    xl = xL - 0.2
    side_bracket!(ax, xl, 0.0, d + 1.35, "Geant4\nSimulates"; color = TAGS.geant4)
    side_bracket!(ax, xl, y_asic, -0.04, "Julia\nModels"; color = TAGS.julia)
    Mke.text!(ax, xL, y_board - 0.08; text = "The Charge Cloud, the Electrodes' Shares, Noise and Thresholds Are Applied in Julia to Geant4's Energy Deposits", fontsize = 15, color = PALETTE.muted, align = (:left, :top))
    Mke.limits!(ax, xl - 1.25, x_text + 4.4, y_board - 0.45, d + 2.35)
    fig
end

# ╔═╡ debadf41-b238-45ca-a2f2-eeecff9a20f5
let
    Mke.set_theme!(THEME)
    p, g = alpha.dexel, alpha.gap
    d = crystal.thickness_mm
    fig = Mke.Figure(size = (1600, 620))

    # (a) A scintillator detector: blocks sawn apart, reflector between, a photodiode under each
    axa = canvas(fig[1, 1]; title = "(a) The Force: a Dexel Is a Sawn Block", titlesize = 20)
    w, s = 1.0, 0.1
    for k in 0:2
        rect!(axa, k * w, 0, w, 1.4, Mke.RGBf(0.97, 0.97, 0.95); strokecolor = (:black, 0.25))
        rect!(axa, k * w + s / 2, 0, w - s, 1.4, Mke.RGBf(0.74, 0.84, 0.93))
        rect!(axa, k * w + s / 2, -0.2, w - s, 0.12, Mke.RGBf(0.24, 0.27, 0.55))
    end
    Mke.text!(axa, 1.5, -0.3; text = "White Reflector Walls Separate the Blocks;\nOne Photodiode Under Each", align = (:center, :top), fontsize = 15, color = PALETTE.muted)
    Mke.limits!(axa, -0.3, 3.3, -1.2, 1.8)

    # (b) The Alpha: one crystal; the electrodes underneath decide where one dexel ends
    axb = canvas(fig[1, 2]; title = "(b) The Alpha: a Dexel Is an Electrode", titlesize = 20)
    sc = 1 / p[1]
    rect!(axb, 0, 0, 4.0 + g[1] * sc, d * sc / 3.2, MATERIALS.cdte; strokecolor = (:black, 0.25))
    for k in 0:3
        x = (g[1] / 2 + k * p[1]) * sc
        rect!(axb, x + 0.05, -0.12, 0.9, 0.08, MATERIALS.electrode)
        Mke.lines!(axb, [x, x], [0, d * sc / 3.2]; color = (PALETTE.ink, 0.25), linestyle = :dash)
    end
    rect!(axb, 0, d * sc / 3.2, 4.0 + g[1] * sc, 0.04, MATERIALS.cathode)
    Mke.text!(axb, (4.0 + g[1] * sc) / 2, -0.3; text = "No Walls Inside the Crystal: Dashed Lines Are Only\nWhere the Electrodes' Territories Meet", align = (:center, :top), fontsize = 15, color = PALETTE.muted)
    Mke.limits!(axb, -0.3, 4.6, -1.2, 2.4)

    # (c) Seen from the beam: native dexels, the 2 × 2 binned dexels, the grid cell
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Native, Binned, and the Grid Cell",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Along Z (mm)",
        aspect = Mke.DataAspect(),
    )
    rect!(axc, -g[1] / 2, -g[2] / 2, cell_pitch[1], cell_pitch[2], MATERIALS.tungsten)
    for k in 0:3, l in 0:5
        c = dexel_corner(k, l) .- g ./ 2
        rect!(axc, c[1], c[2], p[1], p[2], MATERIALS.cdte_lit; strokecolor = PALETTE.slab_edge, strokewidth = 0.8)
    end
    for k in 0:2:2, l in 0:2:4
        c = dexel_corner(k, l) .- g ./ 2
        Mke.poly!(axc, Mke.Rect2f(c[1], c[2], 2p[1], 2p[2]); color = (:white, 0), strokecolor = PALETTE.photon, strokewidth = 2.5, linestyle = :dash)
    end
    c = dexel_corner(0, 0) .- g ./ 2
    Mke.poly!(axc, Mke.Rect2f(c[1], c[2], p[1], p[2]); color = (:white, 0), strokecolor = MATERIALS.beam, strokewidth = 3.5)
    note!(axc, cell_pitch[1] + 0.05, 0.1, "Amber: One Native Dexel,\n0.275 × 0.322 mm"; align = (:left, :bottom), color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 14)
    note!(axc, cell_pitch[1] + 0.05, 0.85, "Dashed Blue: 2 × 2 Binned\nDexels (Standard Mode)"; align = (:left, :bottom), color = PALETTE.photon, fontsize = 14)
    note!(axc, cell_pitch[1] + 0.05, 1.6, "Grey Frame: the Grid's Walls,\nAround 4 × 6 Native Dexels"; align = (:left, :bottom), color = PALETTE.slab_edge, fontsize = 14)
    Mke.xlims!(axc, -0.35, cell_pitch[1] + 1.25)
    Mke.colsize!(fig.layout, 3, Mke.Relative(0.36))
    fig
end

# ╔═╡ 7cefafb2-4fa6-43f2-9cc7-c64d016df12b
"""
The blocks of a patch of the Alpha's detector, `nx` × `ny` native dexels from the corner of a grid
cell, pulled apart layer by layer: board, counting chip (ASIC), solder bumps, one gold electrode
per dexel, the continuous CdTe crystal (drawn `depth` mm deep), the cathode film over it, and the
grid's walls (drawn `grid_drawn` mm tall) where the patch meets a cell's edge; `thin` scales the
thin layers, for small drawings. Returns each layer's blocks
(`layers[key]`), its (bottom, top) (`levels[key]`), all the blocks (`blocks`) and the patch's
extent on the face (`extent`).
"""
function exploded_alpha(nx, ny; gap = 0.8, grid_drawn = 2.4, depth = 1.6, thin = 1.0)
    p, g = alpha.dexel, alpha.gap
    x1 = dexel_corner(nx - 1, 0)[1] + p[1] + (nx % alpha.group[1] == 0 ? g[1] / 2 : 0.04)
    y1 = dexel_corner(0, ny - 1)[2] + p[2] + (ny % alpha.group[2] == 0 ? g[2] / 2 : 0.04)
    x0, y0 = 0.0, 0.0
    stack = (
        (key = :board, h = 0.3thin, gap = 0.45thin),
        (key = :asic, h = 0.4thin, gap = 0.0),
        (key = :bumps, h = 0.18thin, gap = 0.0),
        (key = :electrodes, h = 0.05thin, gap = gap),
        (key = :cdte, h = depth, gap = 0.0),
        (key = :cathode, h = 0.05thin, gap = gap),
        (key = :grid, h = grid_drawn, gap = 0.0),
    )
    layers = Dict{Symbol, Vector{Block}}()
    levels = Dict{Symbol, Tuple{Float64, Float64}}()
    z = 0.0
    for layer in stack
        z0, z1 = z, z + layer.h
        levels[layer.key] = (z0, z1)
        layers[layer.key] = if layer.key === :board
            tiles((x0, y0, z0), (x1, y1, z1), MATERIALS.board)
        elseif layer.key === :asic
            tiles((x0, y0, z0), (x1, y1, z1), MATERIALS.asic)
        elseif layer.key in (:bumps, :electrodes)
            inset = layer.key === :bumps ? 0.09 : 0.025
            color = layer.key === :bumps ? MATERIALS.bump : MATERIALS.electrode
            [
                Block((c[1] + inset, c[2] + inset, z0), (c[1] + p[1] - inset, c[2] + p[2] - inset, z1), color)
                    for c in (dexel_corner(k, l) for k in 0:(nx - 1) for l in 0:(ny - 1))
            ]
        elseif layer.key === :cdte
            tiles((x0, y0, z0), (x1, y1, z1), MATERIALS.cdte)
        elseif layer.key === :cathode
            tiles((x0, y0, z0), (x1, y1, z1), MATERIALS.cathode)
        else
            walls = Block[]
            for c in 0:(nx ÷ alpha.group[1])
                x = c * cell_pitch[1]
                x <= x1 + 1.0e-9 || continue
                append!(walls, tiles((x - g[1] / 2, y0 - g[2] / 2, z0), (x + g[1] / 2, y1, z1), MATERIALS.tungsten; size = (g[1], p[2])))
            end
            for r in 0:(ny ÷ alpha.group[2])
                y = r * cell_pitch[2]
                y <= y1 + 1.0e-9 || continue
                append!(walls, tiles((x0 + g[1] / 2, y - g[2] / 2, z0), (x1, y + g[2] / 2, z1), MATERIALS.tungsten; size = (p[1], g[2])))
            end
            walls
        end
        z = z1 + layer.gap
    end
    return (; layers, levels, blocks = reduce(vcat, values(layers)), extent = ((x0, y0), (x1, y1)))
end

# ╔═╡ e71ceeb9-0d88-5608-8588-ee48a312de90
let
    Mke.set_theme!(THEME)
    p = alpha.dexel
    nx, ny = alpha.group
    cam = camera(-56, 30)
    fig = Mke.Figure(size = (1450, 1000))
    ax = canvas(fig[1, 1])
    ex = exploded_alpha(nx, ny)
    blocks, levels = ex.blocks, ex.levels
    (x0, y0), (x1, y1) = ex.extent

    # X rays from the focal spot, far above, into the open cell
    top = levels[:grid][2]
    rays = map([(0, 0), (1, 1), (2, 0), (3, 2)]) do (k, l)
        c = dexel_corner(k, l) .+ p ./ 2
        a, b = project(cam, (c..., top + 3.0)), project(cam, (c..., top + 0.3))
        (dot3((c..., top), cam.toward) + 50, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3.5, head = 15))
    end
    draw_blocks!(ax, cam, blocks; extras = rays)

    # One dexel: the front-right one, its electrode, bump, chip channel and the crystal above
    k, l = nx - 1, 0
    c0 = dexel_corner(k, l)
    for key in (:cdte, :electrodes, :bumps, :asic)
        z0, z1 = levels[key]
        outline!(ax, cam, (c0[1], c0[2], z0), (c0[1] + p[1], c0[2] + p[2], z1); linewidth = 3)
    end

    rows = (
        (:grid, "Anti-Scatter Grid", "2-D Tungsten Walls Around Each 4 × 6 Dexels\n(10 mm Tall, Assumed; Drawn Short Here)", ("In Geant4", TAGS.geant4)),
        (:cathode, "Cathode", "One Thin Metal Film over the Whole Crystal,\nat 800–1000 V", ("Not Simulated", TAGS.not_simulated)),
        (:cdte, "CdTe Crystal, One Piece", "1.6 mm Deep, Continuous: Absorbs the X Ray,\nwhich Frees a Cloud of Charge", ("In Geant4", TAGS.geant4)),
        (:electrodes, "Pixel Electrodes (Anodes)", "One Gold Pad per Dexel: Each Collects the Charge\nThat Drifts Down onto It", ("In Julia", TAGS.julia)),
        (:asic, "Counting Chip (ASIC)", "Per Dexel: Pulse Shaping, Four Thresholds,\nFour Counters", ("In Julia", TAGS.julia)),
        (:board, "Circuit Board", "Carries Every Dexel's Counts Out", ("Not Simulated", TAGS.not_simulated)),
    )
    x_text = maximum(project(cam, (x1, y, 0))[1] for y in (y0, y1)) + 2.2
    entries = [
        (
                anchor = project(cam, (x1, (y0 + y1) * 0.6, sum(levels[key]) / 2)),
                title = name,
                detail,
                color = PALETTE.ink,
                tag,
            )
            for (key, name, detail, tag) in rows
    ]
    labels_right!(ax, entries, x_text; spacing = 1.25, tag_width = 122)
    Mke.xlims!(ax, nothing, x_text + 9.2)
    Mke.Label(
        fig[2, 1],
        "One Dexel, Outlined in Amber: Its Electrode, Its Chip Channel and the Crystal Above It. The Crystal Is One Piece: the Electrode Alone Makes the Dexel.";
        font = :bold,
        fontsize = 17,
        color = Mke.RGBf(0.8, 0.5, 0.05),
        halign = :left,
        tellwidth = false,
    )
    fig
end

# ╔═╡ 20aa77fa-7c82-4c2a-b72d-58c13baead11
let
    Mke.set_theme!(THEME)
    p = alpha.dexel
    fig = Mke.Figure(size = (1700, 560))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        ("Uniform on One Dexel", "No Grid; the Dexel and Its\nEight Neighbours Read (1–4)", :one),
        ("Under the Grid", "Uniform on Each of the Six\nPositions in a Cell (5, 7)", :grid),
        ("At the Cone Angle", "The Outer Rows' Photons\nArrive Tilted (6)", :tilt),
        ("From the Focal Spot", "The Whole Arc: One Energy,\nthe 140 kVp Spectrum (8, 9)", :focal),
        ("Clinical Spectra", "Bare and After 20 and 35 cm\nof Water, Photon by Photon (11)", :clinical),
    )
    for (k, (title, detail, kind)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        ex = exploded_alpha(4, 3; grid_drawn = 1.6)
        zc = ex.levels[:cathode][2]
        blocks = Block[]
        append!(blocks, ex.layers[:cdte]); append!(blocks, ex.layers[:cathode])
        if kind in (:grid, :tilt, :focal)
            for b in ex.layers[:grid]
                push!(blocks, Block((b.lo[1], b.lo[2], zc), (b.hi[1], b.hi[2], zc + 1.6), b.color))
            end
        end
        extras = Any[]
        c = dexel_corner(1, 1)
        if kind in (:one, :grid)
            column = Block((c[1], c[2], zc), (c[1] + p[1], c[2] + p[2], zc + 3.0), amber; alpha = 0.3, stroke = 0.0)
            push!(extras, (1.0e3, a -> draw_blocks!(a, cam, [column]; edge_alpha = 0.0)))
        elseif kind === :tilt
            for (dx, dy) in ((0.3, 0.3), (0.7, 0.4), (0.5, 0.7))
                from = project(cam, (c[1] + dx * p[1], c[2] + dy * p[2] - 1.2, zc + 3.4))
                to = project(cam, (c[1] + dx * p[1], c[2] + dy * p[2], zc))
                push!(extras, (1.0e3, a -> arrow!(a, from, to; color = amber, linewidth = 2.5, head = 11)))
            end
        else
            focal = ((ex.extent[2][1]) / 2, (ex.extent[2][2]) / 2, zc + 5.0)
            for (x, y) in ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.5))
                from, to = project(cam, focal), project(cam, (x * ex.extent[2][1], y * ex.extent[2][2], zc))
                push!(extras, (1.0e3, a -> arrow!(a, from, to; color = kind === :clinical ? PALETTE.xray : amber, linewidth = 2.2, head = 10)))
            end
            push!(extras, (1.0e3 + 1, a -> Mke.scatter!(a, [project(cam, focal)]; color = PALETTE.ink, markersize = 12)))
        end
        draw_blocks!(ax, cam, blocks; extras)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    fig
end

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
            transport = nothing,
        )
        for p in positions
]

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
        title = "Counts Above 20 keV in the Dexel, per Photon Landing on It",
        xlabel = "Photon Energy (keV)",
        ylabel = "Counts per Photon",
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
        label = "No Grid (Section 3)",
    )
    colors = Mke.cgrad(:tableau_10, 6; categorical = true)
    for (k, pos) in enumerate(positions)
        Mke.lines!(
            ax,
            E,
            [c.per_dexel[2, 2, 1] for c in grid_counts[k]];
            color = colors[k],
            linewidth = 2.5,
            label = "Position $(pos.label)",
        )
    end
    note!(
        ax,
        140,
        0.05,
        "Every Position Counts Almost the Same: the Walls Stand over the Gaps, Not over Dexels";
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
        title = "Position F: Where the Counts Land, by Cone Angle",
        xlabel = "Photon Energy (keV)",
        ylabel = "Counts Above 20 keV per Photon",
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
            label = "$(θ)°: the Dexel Aimed At",
        )
        Mke.lines!(
            ax,
            E,
            [x.per_dexel[2, 3, 1] for x in c];
            color = colors[k],
            linewidth = 2,
            linestyle = :dash,
            label = "$(θ)°: the Next Row",
        )
    end
    note!(
        ax,
        140,
        0.35,
        "The Deeper the Photon Goes (High Energy),\nthe Further It Drifts into the Next Row";
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
        title = "A Binned (2 × 2) Dexel: Counts Above Each Threshold, per Photon Landing on It",
        xlabel = "Photon Energy (keV)",
        ylabel = "Counts per Photon",
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
        "Solid: Corner Group   Dashed: Middle Group";
        align = (:right, :top),
        fontsize = 14,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center, title = "Threshold")
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

# ╔═╡ 9893a0c1-2162-4290-8273-19740f076747
"""
The master map, as a band across the top of a section's first illustration: the gantry, one
dexel's stack (a dexel at the corner of a grid cell) with the part under study lit and outlined,
and the dexel's five steps with this one's lit. `focus` is `:tube`, `:arc`, `:whole`, or one or a
tuple of the stack's layers, `:grid`, `:cdte`, `:electrodes`, `:asic`. Place it with
`locator!(fig[0, :], focus; note)`.
"""
function locator!(position, focus; note = "")
    amber, dim = MATERIALS.beam, Mke.RGBf(0.82, 0.83, 0.86)
    Mke.Box(position; color = Mke.RGBf(0.965, 0.965, 0.975), strokevisible = false, cornerradius = 14)
    layout = Mke.GridLayout(position; halign = :center, tellwidth = false)
    focus_set = focus isa Tuple ? focus : (focus,)

    # The gantry, small: tube A, its fan, detector A's arc, the one dexel below
    ax = canvas(layout[1, 1]; width = 230, height = 230, backgroundcolor = :transparent)
    θ = range(0, 2π; length = 120)
    R_iso, R_det = 610.0, detector_a.radius_mm
    half = deg2rad(MC.fan_angle_deg(detector_a)) / 2
    lit(k) = k in focus_set || :whole in focus_set
    Mke.lines!(ax, 820 .* cos.(θ), 820 .* sin.(θ); color = dim, linewidth = 1.5)
    Mke.poly!(ax, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = (PALETTE.muted, 0.25))
    src = Mke.Point2f(0, R_iso)
    φ = range(-half, half; length = 60)
    arc = [src + R_det * Mke.Point2f(sin(t), -cos(t)) for t in φ]
    Mke.poly!(ax, Mke.Point2f.(vcat([src], arc)); color = (amber, lit(:tube) ? 0.45 : 0.12))
    Mke.lines!(ax, Mke.Point2f.(arc); color = lit(:arc) ? PALETTE.ink : dim, linewidth = lit(:arc) ? 7 : 4)
    Mke.scatter!(ax, [src]; marker = :rect, markersize = 16, color = lit(:tube) ? PALETTE.ink : dim)
    Mke.scatter!(ax, [arc[30]]; marker = :rect, markersize = 9, color = amber, strokecolor = PALETTE.ink, strokewidth = 1)
    Mke.limits!(ax, -900, 900, -760, 900)

    # One dexel's stack, exploded; the focus lit and outlined, the rest faded
    axs = canvas(layout[1, 2]; width = 150, height = 250, backgroundcolor = :transparent)
    ex = exploded_alpha(1, 1; gap = 0.2, grid_drawn = 0.5, depth = 0.42, thin = 0.4)
    on = :whole in focus_set ? Tuple(keys(ex.layers)) :
        Tuple(k for k in (:grid, :cathode, :cdte, :electrodes, :bumps, :asic) if k in focus_set || (k === :cathode && :cdte in focus_set))
    blocks = Block[]
    for (key, bs) in ex.layers
        append!(blocks, key in on ? bs : [faded(b, 0.86) for b in bs])
    end
    cam = camera(-56, 30)
    draw_blocks!(axs, cam, blocks; edge_alpha = 0.15)
    (x0, y0), (x1, y1) = ex.extent
    for key in on
        key in (:cathode, :bumps) && continue
        z0, z1 = ex.levels[key]
        lo = key === :grid ? (x0 - alpha.gap[1] / 2, y0 - alpha.gap[2] / 2) : (x0, y0)
        outline!(axs, cam, (lo[1] - 0.04, lo[2] - 0.04, z0 - 0.04), (x1 + 0.04, y1 + 0.04, z1 + 0.04); linewidth = 2.5)
    end

    # Where we are, in words, and the dexel's five steps with this one's lit
    text = Mke.GridLayout(layout[1, 3]; valign = :center)
    Mke.Label(text[1, 1], "Where We Are"; font = :bold, fontsize = 20, color = PALETTE.ink, halign = :left)
    Mke.Label(text[2, 1], note; fontsize = 17, color = PALETTE.ink, halign = :left, justification = :left)
    steps = (
        (:grid, "1  The Grid"),
        (:cdte, "2  The CdTe Absorbs It"),
        (:cdte, "3  Its Charge Drifts"),
        (:electrodes, "4  The Electrodes Collect It"),
        (:asic, "5  Thresholds Count It"),
    )
    for (k, (key, label)) in enumerate(steps)
        active = key in on
        Mke.Label(
            text[k + 2, 1],
            label;
            fontsize = 15,
            font = active ? :bold : :regular,
            color = active ? Mke.RGBf(0.8, 0.5, 0.05) : PALETTE.muted,
            halign = :left,
        )
    end
    Mke.rowgap!(text, 3)
    Mke.rowgap!(text, 1, 6)
    Mke.rowgap!(text, 2, 10)
    Mke.colgap!(layout, 28)
    return layout
end

# ╔═╡ 497d96da-68e6-5194-ac65-4f174288ace0
let
    Mke.set_theme!(THEME)
    px, py = readout.pitch_x_mm, readout.pitch_y_mm
    d = crystal.thickness_mm
    fig = Mke.Figure(size = (1600, 960))

    # (a) What Geant4 builds: one slab of CdTe; the 3 × 3 dexels are the electrodes under it
    ax = canvas(fig[1, 1]; title = "(a) The Crystal, Lifted off Its Electrodes", titlesize = 20)
    cam = camera(-52, 30)
    lift = 0.9
    blocks = Block[]
    for i in -1:1, j in -1:1
        x0, y0 = (i - 0.5) * px, (j - 0.5) * py
        push!(blocks, Block((x0 + 0.012, y0 + 0.012, 0.0), (x0 + px - 0.012, y0 + py - 0.012, 0.05), i == 0 && j == 0 ? MATERIALS.beam : MATERIALS.electrode))
    end
    append!(blocks, tiles((-1.5px, -1.5py, lift), (1.5px, 1.5py, lift + d), MATERIALS.cdte; size = (px, py)))
    append!(blocks, tiles((-1.5px, -1.5py, lift + d), (1.5px, 1.5py, lift + d + 0.04), MATERIALS.cathode; size = (px, py)))
    column = Block((-px / 2, -py / 2, lift + d + 0.04), (px / 2, py / 2, lift + d + 1.6), MATERIALS.beam; alpha = 0.3, stroke = 0.0)
    draw_blocks!(ax, cam, blocks; extras = [(1.0e3, a -> draw_blocks!(a, cam, [column]; edge_alpha = 0.0))])
    outline!(ax, cam, (-px / 2, -py / 2, lift), (px / 2, py / 2, lift + d); linewidth = 2.5, linestyle = :dash)
    Mke.Label(
        fig[2, 1],
        "Amber Column: Photons Landing Uniformly over the Middle Dexel.\nDashed: the Crystal Above Its Electrode. Below: the Nine Electrodes.";
        fontsize = 15,
        color = PALETTE.ink,
        tellwidth = false,
    )

    # (b) Seen from the beam
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the Beam",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Along the Patient (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6px, 2.3px), (-2.0py, 1.6py)),
    )
    for i in -1:1, j in -1:1
        central = i == 0 && j == 0
        rect!(axb, (i - 0.5) * px, (j - 0.5) * py, px, py, central ? Mke.RGBf(0.98, 0.88, 0.68) : MATERIALS.cdte_lit; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    end
    note!(axb, 0, 0.0, "Middle\nDexel"; align = (:center, :center), color = Mke.RGBf(0.75, 0.45, 0.05), fontsize = 14)
    Mke.bracket!(axb, -0.5px, -1.5py, 0.5px, -1.5py; text = "$(px) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.bracket!(axb, 1.5px, -0.5py, 1.5px, 0.5py; text = "$(py) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)

    # (c) Side view: the cathode, the crystal, the anodes; the charge drifts down
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side View",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Depth (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.6px, 2.6px), (-0.35, d + 0.15)),
    )
    rect!(axc, -1.5px, 0, 3px, d, MATERIALS.cdte)
    rect!(axc, -1.5px, -0.04, 3px, 0.04, MATERIALS.cathode)
    for i in -1:1
        rect!(axc, (i - 0.5) * px + 0.01, d, px - 0.02, 0.05, i == 0 ? MATERIALS.beam : MATERIALS.electrode)
    end
    Mke.lines!(axc, [0.03, 0.03], [-0.3, 0.45]; color = MATERIALS.beam, linewidth = 3)
    Mke.scatter!(axc, [Mke.Point2f(0.03, 0.45)]; marker = :star5, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, markersize = 16)
    Mke.arrows2d!(axc, [Mke.Point2f(0.03, 0.5)], [Mke.Vec2f(0, d - 0.6)]; color = MATERIALS.charge, shaftwidth = 3, tipwidth = 12, tiplength = 12)
    note!(axc, 0.06, 0.95, "The Freed Electrons\nDrift to the Anodes,\nSpreading"; align = (:left, :center), color = MATERIALS.charge, fontsize = 13)
    Mke.bracket!(axc, 1.5px, 0, 1.5px, d; text = "$(d) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.38))
    Mke.colgap!(fig.layout, 30)
    locator!(fig[0, :], (:cdte, :electrodes); note = "Steps 2–4 · One Dexel\nand Its Neighbours")
    fig
end

# ╔═╡ fb9ed65f-1ea8-557f-9703-76b935d46a75
let
    Mke.set_theme!(THEME)
    px, py = readout.pitch_x_mm, readout.pitch_y_mm
    σ = readout.cloud_sigma_um / 1000
    d = crystal.thickness_mm
    steps = [s for s in tracks.steps if s.event == shared_event]
    dep = deposits_run.deposits
    k = findall(==(shared_event), dep.event)
    fig = Mke.Figure(size = (1400, 910))

    # (a) Side view of the photon and its x ray, with the dexel boundaries
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The Photon and Its X Ray",
        xlabel = "Across the Fan (mm)",
        ylabel = "Depth (mm)",
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
        "Dashed: Dexel\nBoundaries";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )

    # (b) Top view: the energy deposits and their charge clouds over the dexel grid
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Its Charge Clouds, from Above",
        xlabel = "X (mm)",
        ylabel = "Y (mm)",
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
        "Circles: 2σ = 60 µm";
        align = (:center, :bottom),
        color = PALETTE.electron,
        fontsize = 14,
    )

    # (c) What each dexel collected, and which thresholds it crossed
    energies = single_response.dexel_energy_keV[shared_event, :, :]
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) What Each Dexel Collects",
        xlabel = "X (Dexel)",
        ylabel = "Y (Dexel)",
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
    locator!(fig[0, :], (:cdte, :electrodes); note = "Steps 2–4 · One Photon's\nJourney, Followed")
    fig
end

# ╔═╡ 8b38185f-e423-49ca-af97-536138ddc8be
let
    Mke.set_theme!(THEME)
    p, g = alpha.dexel, alpha.gap
    d = crystal.thickness_mm
    edges = MC.k_edges(crystal.material)
    kα = Dict(e.element => e.kalpha_keV for e in edges)
    fig = Mke.Figure(size = (1700, 820))
    cases = (
        ("All of It Stays", "Absorbed, Its Charge Under This Dexel:\nthe Full-Energy Peak at E", PALETTE.kept),
        ("A Cd or Te X Ray Escapes", "$(round(kα["Cd"]; digits = 1)) or $(round(kα["Te"]; digits = 1)) keV Leave the Crystal:\nEscape Peaks at E − $(round(kα["Cd"]; digits = 1)) and E − $(round(kα["Te"]; digits = 1))", PALETTE.xray),
        ("Its X Ray Lands Next Door", "The Neighbour Sees $(round(Int, kα["Cd"]))–$(round(Int, kα["Te"])) keV,\nThis Dexel the Rest", Mke.RGBf(0.45, 0.3, 0.6)),
        ("The Cloud Lands on an Edge", "Charge Shared Between Two Dexels:\nAnything from 0 to E in Each", MATERIALS.charge),
    )
    function slab!(ax)
        rect!(ax, -1.5p[1], 0, 3p[1], d, MATERIALS.cdte; strokecolor = (:black, 0.25))
        rect!(ax, -1.5p[1], d, 3p[1], 0.04, MATERIALS.cathode)
        for k in -1:1
            rect!(ax, (k - 0.5) * p[1] + 0.02, -0.1, p[1] - 0.04, 0.07, k == 0 ? MATERIALS.beam : MATERIALS.electrode)
        end
        for x in (-0.5p[1], 0.5p[1])
            Mke.lines!(ax, [x, x], [0, d]; color = (PALETTE.ink, 0.3), linestyle = :dash, linewidth = 1.2)
        end
        return
    end
    function cloud!(ax, x, z; split = false)
        Mke.lines!(ax, [x, x], [z - 0.05, 0.05]; color = (MATERIALS.charge, 0.5), linewidth = 1.5, linestyle = :dot)
        return Mke.poly!(ax, Mke.Circle(Mke.Point2f(x, 0.08), 0.06); color = (MATERIALS.charge, 0.35), strokecolor = MATERIALS.charge, strokewidth = 1)
    end
    for (k, (title, detail, color)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 20)
        slab!(ax)
        x_hit = k == 4 ? 0.5p[1] - 0.025 : 0.0
        hit = Mke.Point2f(x_hit, d - 0.45)
        photon!(ax, (x_hit, d + 0.9), hit; cycles = 5, amplitude = 0.03, color = MATERIALS.beam)
        Mke.scatter!(ax, [hit]; marker = :star5, markersize = 22, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)
        if k == 1
            cloud!(ax, hit[1], hit[2])
        elseif k == 2
            cloud!(ax, hit[1], hit[2])
            photon!(ax, hit, (-0.33, d + 0.8); cycles = 3, amplitude = 0.06, color = PALETTE.xray, head = 11)
        elseif k == 3
            cloud!(ax, hit[1], hit[2])
            stop = Mke.Point2f(1.15p[1], 0.5)
            photon!(ax, hit, stop; cycles = 3, amplitude = 0.06, color = PALETTE.xray, head = 10)
            cloud!(ax, stop[1], stop[2])
        else
            cloud!(ax, hit[1], hit[2])
        end
        Mke.limits!(ax, -1.7p[1], 1.7p[1], -0.3, d + 1.0)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :cdte; note = "Steps 2–4 · Where One\nPhoton's Charge Goes")
    fig
end

# ╔═╡ 8ea66d84-4cce-4ff9-881f-1ea60174ef27
let
    Mke.set_theme!(THEME)
    p = charge_transport
    L = p.thickness_mm
    wx, wy = readout.pitch_x_mm, readout.pitch_y_mm
    fig = Mke.Figure(size = (1500, 1080))

    # (a) Three dexels from the side, to scale, coloured by the middle dexel's weighting potential
    ax = Mke.Axis(
        fig[1, 1];
        title = "(a) Three Dexels from the Side, to Scale",
        xlabel = "Across the Fan (mm)",
        ylabel = "Depth from the Front (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.5wx - 0.02, 1.5wx + 0.02), (-0.32, L + 0.1)),
    )
    xs = range(-1.5wx, 1.5wx; length = 161)
    zs = range(0.0, L; length = 161)
    φ = [MC.weighting_potential(x, 0.0, L - z, wx, wy, L) for x in xs, z in zs]
    map_colours = Mke.cgrad([Mke.RGBf(0.97, 0.97, 0.99), MATERIALS.cdte_lit, MATERIALS.electrode])
    hm = Mke.heatmap!(ax, xs, zs, φ; colormap = map_colours, colorrange = (0, 1))
    Mke.contour!(ax, xs, zs, φ; levels = [0.02, 0.05, 0.1, 0.2, 0.5], color = (PALETTE.ink, 0.45), linewidth = 1.2)
    rect!(ax, -1.5wx, -0.05, 3wx, 0.05, MATERIALS.cathode; strokecolor = (:black, 0.3))
    for i in -1:1
        rect!(ax, i * wx - wx / 2 + 0.008, L, wx - 0.016, 0.05, MATERIALS.electrode)
    end
    # a 60 keV photon absorbed 0.45 mm deep: its electrons drift to the back (the band is ±2σ, to
    # scale), its holes to the front
    x0, z0, e0 = 0.05, 0.45, 60.0
    photon!(ax, (x0, -0.32), (x0, z0 - 0.02); cycles = 6, amplitude = 0.04, color = MATERIALS.beam)
    path = range(z0, L; length = 60)
    width(z) = 2 * sqrt(MC.diffusion_sigma(z - z0, p)^2 + MC.repulsion_sigma(e0, z - z0, p)^2)
    Mke.poly!(
        ax,
        vcat([Mke.Point2f(x0 - width(z), z) for z in path], [Mke.Point2f(x0 + width(z), z) for z in reverse(path)]);
        color = (PALETTE.electron, 0.55),
    )
    arrow!(ax, (x0 - 0.035, z0 - 0.02), (x0 - 0.035, 0.03); color = PALETTE.photon, linewidth = 2.5, head = 11)
    Mke.scatter!(ax, [Mke.Point2f(x0, z0)]; marker = :star5, markersize = 20, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)
    note!(ax, x0 + 0.03, 1.05, "Electrons Drift\nto the Electrodes,\nSpreading (±2σ)"; align = (:left, :center), color = PALETTE.electron, fontsize = 14)
    note!(ax, x0 - 0.05, 0.22, "Holes Drift\nto the Front"; align = (:right, :center), color = PALETTE.photon, fontsize = 14)
    note!(ax, -1.5wx + 0.01, L - 0.08, "The Middle Dexel's\nWeighting Potential:\nNear 0 Except Close\nto Its Electrode"; align = (:left, :bottom), fontsize = 13)
    Mke.Colorbar(fig[1, 2], hm; label = "Weighting Potential of the Middle Dexel", labelsize = 16, ticklabelsize = 14)

    # (b) Where the electrons arrive: the cloud over the three electrodes, from three depths
    axb = Mke.Axis(
        fig[1, 3];
        title = "(b) Where the Electrons Arrive (60 keV, Absorbed Near the Edge)",
        xlabel = "Across the Fan (mm)",
        ylabel = "Charge Arriving per µm (Share)",
        limits = ((-0.6wx, 0.9wx), (0, nothing)),
    )
    edge = wx / 2
    Mke.vspan!(axb, -0.6wx, edge; color = (MATERIALS.cdte_lit, 0.5))
    Mke.vspan!(axb, edge, 0.9wx; color = (MATERIALS.cathode, 0.6))
    note!(axb, 0.0, 0.0, "Middle Dexel"; align = (:center, :bottom), offset = (0, 6), fontsize = 14)
    note!(axb, 0.72wx, 0.0, "Its Neighbour"; align = (:center, :bottom), offset = (0, 6), fontsize = 14)
    x_hit = edge - 0.02
    u = range(-0.6wx, 0.9wx; length = 600)
    gauss(x, σ) = exp(-(x - x_hit)^2 / (2σ^2)) / (σ * sqrt(2π)) / 1000
    for (z, label, colour) in ((0.1, "From the Front (0.1 mm)", PALETTE.electron), (0.8, "From the Middle (0.8 mm)", MATERIALS.beam), (1.5, "From the Back (1.5 mm)", PALETTE.kept))
        σ = sqrt(MC.diffusion_sigma(L - z, p)^2 + MC.repulsion_sigma(60.0, L - z, p)^2)
        Mke.lines!(axb, u, gauss.(u, σ); color = colour, linewidth = 3, label = "$(label): σ = $(round(1000σ; digits = 1)) µm")
    end
    Mke.lines!(axb, u, gauss.(u, 0.03); color = PALETTE.reference, linewidth = 2.5, linestyle = :dash, label = "BasisSimulator v0.15.1: σ = 30 µm Always")
    Mke.vlines!(axb, edge; color = PALETTE.ink, linewidth = 1.5)
    Mke.axislegend(axb; position = :lt)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.3))
    locator!(fig[0, :], (:cdte, :electrodes); note = "Steps 3–4 · The Charge\nDrifts and Is Collected")
    fig
end

# ╔═╡ 5667adb0-4579-4f57-bee7-4b081cdb4f9c
let
    Mke.set_theme!(THEME)
    p = charge_transport
    L = p.thickness_mm
    cdte = MC.DetectorMaterial(:CdTe)
    shown = (30.0, 60.0, 120.0)
    fig = Mke.Figure(size = (1600, 980))

    # Top: where each energy is absorbed in the crystal, and how wide its charge arrives
    for (k, E) in enumerate(shown)
        ax = canvas(fig[1, k]; title = "$(Int(E)) keV: Where It Is Absorbed", titlesize = 19)
        μ = MC.linear_attenuation(cdte, E)
        rect!(ax, 0, 0, 1.0, L, MATERIALS.cdte; strokecolor = PALETTE.slab_edge)
        # shading: the share of photons absorbed at each depth
        for z0 in range(0.0, L - 0.04; length = 40)
            density = μ * exp(-μ * z0) / μ
            rect!(ax, 0, L - z0 - 0.04, 1.0, 0.04, (MATERIALS.beam, clamp(0.85density, 0.0, 0.85)))
        end
        mean_depth = 1 / μ - L * exp(-μ * L) / (1 - exp(-μ * L))
        σ = sqrt(MC.diffusion_sigma(L - mean_depth, p)^2 + MC.repulsion_sigma(E, L - mean_depth, p)^2)
        rect!(ax, 0, -0.06, 1.0, 0.06, MATERIALS.electrode)
        Mke.text!(ax, 1.05, L - mean_depth; text = "Typical Depth\n$(round(mean_depth; digits = 2)) mm\nσ ≈ $(round(1000σ; digits = 1)) µm", align = (:left, :center), fontsize = 14, color = PALETTE.ink)
        Mke.text!(ax, 0.5, L + 0.05; text = "Front", align = (:center, :bottom), fontsize = 13, color = PALETTE.muted)
        Mke.text!(ax, 0.5, -0.1; text = "Electrodes", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
        Mke.limits!(ax, -0.1, 1.9, -0.35, L + 0.25)
    end

    # Bottom: what the middle dexel records from each photon, old cloud and transported charge
    for (k, E) in enumerate(shown)
        e = findfirst(==(E), energies)
        old = vec(responses[e].dexel_energy_keV[:, 2, 2])
        new = vec(transport_runs.responses[e].dexel_energy_keV[:, 2, 2])
        bins = 0.0:1.0:(E + 5)
        density(v) = [count(x -> b <= x < b + 1, v) / length(v) for b in bins[1:(end - 1)]]
        ax = Mke.Axis(
            fig[2, k];
            title = "What the Middle Dexel Records",
            xlabel = "Energy Recorded (keV)",
            ylabel = k == 1 ? "Share of Photons per keV" : "",
            yscale = log10,
            limits = ((0, E + 5), (1.0e-5, 1)),
        )
        centres = bins[1:(end - 1)] .+ 0.5
        Mke.lines!(ax, centres, max.(density(old), 1.0e-6); color = PALETTE.reference, linewidth = 2, linestyle = :dash, label = "Fixed 30 µm Cloud")
        Mke.lines!(ax, centres, max.(density(new), 1.0e-6); color = PALETTE.kept, linewidth = 3, label = "Transported Charge")
        k == 1 && Mke.axislegend(ax; position = :lt)
        note!(ax, E, 0.3, "Full Energy"; align = (:right, :center), offset = (-6, 0), color = PALETTE.kept, fontsize = 14)
        note!(ax, E / 2, 2.0e-3, "Shared with\na Neighbour"; align = (:center, :center), color = PALETTE.muted, fontsize = 14)
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.42))
    locator!(fig[0, :], (:cdte, :electrodes); note = "Steps 2–4 · The Same Photons,\nTheir Charge Transported")
    fig
end

# ╔═╡ cb866af1-b9f8-467d-bab4-77815e589bd9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 520))
    levels = [20, 25, 30, 35, 55, 60, 70, 75, 90]

    # (a) One 60 keV pulse against the nine thresholds: one count for each it crosses
    ax = Mke.Axis(
        fig[1, 1];
        title = "(a) One Pulse, Nine Thresholds",
        titlesize = 20,
        xlabel = "Time (ns)",
        ylabel = "Pulse Height (keV)",
        limits = ((0, 40), (0, 100)),
        yticks = [20, 35, 55, 70, 90],
    )
    t = range(0, 40; length = 300)
    pulse = [60 * (x / 8) * exp(1 - x / 8) for x in t]
    for L in levels
        crossed = maximum(pulse) >= L
        Mke.hlines!(ax, L; color = crossed ? MATERIALS.beam : (PALETTE.muted, 0.5), linewidth = crossed ? 2 : 1, linestyle = crossed ? :solid : :dash)
    end
    Mke.lines!(ax, t, pulse; color = PALETTE.photon, linewidth = 3.5)
    note!(ax, 39, 57, "60 keV Crosses 20–55:\n+1 in Each of Six Counters"; align = (:right, :top), color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 15)
    note!(ax, 39, 94, "Dashed: Not Crossed"; align = (:right, :top), color = PALETTE.muted, fontsize = 14)

    # (b) A photon whose charge is shared: two pulses, in two dexels, both above 20 keV
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Charge Shared: Two Dexels, Two Counts",
        titlesize = 20,
        xlabel = "Time (ns)",
        ylabel = "Pulse Height (keV)",
        limits = ((0, 40), (0, 100)),
        yticks = [20, 35, 55, 70],
    )
    for (E, color, label) in ((38, PALETTE.photon, "This Dexel: 38 keV"), (22, MATERIALS.charge, "Its Neighbour: 22 keV"))
        Mke.lines!(axb, t, [E * (x / 8) * exp(1 - x / 8) for x in t]; color, linewidth = 3, label)
    end
    Mke.hlines!(axb, 20; color = MATERIALS.beam, linewidth = 2)
    note!(axb, 39, 22, "Both Cross 20 keV: One Photon,\nTwo Counts Above 20 keV"; align = (:right, :bottom), color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 15)
    Mke.axislegend(axb; position = :rt)
    locator!(fig[0, :], :asic; note = "Step 5 · Thresholds\nCount Each Pulse")
    fig
end

# ╔═╡ e5b42b5a-564f-5ba8-b8a1-6a302df8e3ad
let
    Mke.set_theme!(THEME)
    p, G, g = alpha.dexel, alpha.group, alpha.gap
    H = alpha.wall_height
    fig = Mke.Figure(size = (1700, 1000))

    # (a) One grid cell, to scale: 10 mm walls around 4 × 6 dexels, the near walls cut away
    ax = canvas(fig[1, 1]; title = "(a) One Grid Cell, to Scale", titlesize = 20)
    cam = camera(-50, 24)
    ex = exploded_alpha(2G[1], G[2]; grid_drawn = H)
    zc = ex.levels[:cathode][2]
    blocks = Block[]
    append!(blocks, ex.layers[:cdte]); append!(blocks, ex.layers[:cathode])
    for b in ex.layers[:grid]
        b2 = Block((b.lo[1], b.lo[2], zc), (b.hi[1], b.hi[2], zc + H), b.color)
        b2.lo[2] < 0.3 && b2.lo[1] > 0.3 && continue        # the near wall, cut away
        push!(blocks, b2)
    end
    c = dexel_corner(5, 2) .+ p ./ 2
    straight = ((c..., zc + H + 3.0), (c..., zc))
    extras = [(1.0e4, a -> arrow!(a, project(cam, straight[1]), project(cam, straight[2]); color = MATERIALS.beam, linewidth = 3.5))]
    draw_blocks!(ax, cam, blocks; extras)
    for k in 0:(2G[1] - 1), l in 0:(G[2] - 1)
        q = dexel_corner(k, l)
        corners = [(q[1], q[2], zc), (q[1] + p[1], q[2], zc), (q[1] + p[1], q[2] + p[2], zc), (q[1], q[2] + p[2], zc), (q[1], q[2], zc)]
        Mke.lines!(ax, [project(cam, x) for x in corners]; color = (PALETTE.ink, 0.25), linewidth = 0.6)
    end
    Mke.Label(fig[2, 1], "Tungsten Walls over the Gaps, $(round(Int, H)) mm Tall (Assumed),\n$(round(Int, 1000g[1])) µm Thick; the Near Wall Cut Away.\nThin Lines: Where the Dexels Are (Electrodes Below)"; fontsize = 15, color = PALETTE.ink, tellwidth = false)

    # (b) Seen from the beam: three by two cells, and the six distinct positions
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the Beam: the Six Kinds of Dexel",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Along Z (mm)",
        aspect = Mke.DataAspect(),
    )
    centre(m, axis) = MC.dexel_centre(m, p[axis], G[axis], g[axis])
    half_x, half_y = cell_pitch[1] / 2, cell_pitch[2] / 2
    for cx in -1:1, cy in -1:0
        x0 = cx * cell_pitch[1] - half_x
        y0 = (cy + 0.5) * cell_pitch[2] - half_y
        rect!(axb, x0, y0, cell_pitch[1], cell_pitch[2], MATERIALS.tungsten)
        for i in 0:(G[1] - 1), j in 0:(G[2] - 1)
            x = cx * cell_pitch[1] + centre(i, 1)
            y = (cy + 0.5) * cell_pitch[2] + centre(j, 2)
            rect!(axb, x - p[1] / 2, y - p[2] / 2, p[1], p[2], MATERIALS.cdte_lit; strokecolor = PALETTE.slab_edge, strokewidth = 0.8)
        end
    end
    mirror(k, n) = min(k, n - 1 - k)
    letter(i, j) = positions[findfirst(q -> q.target == (mirror(i, G[1]), mirror(j, G[2])), positions)].label
    for i in 0:(G[1] - 1), j in 0:(G[2] - 1)
        Mke.text!(axb, centre(i, 1), centre(j, 2) + 0.5 * cell_pitch[2]; text = letter(i, j), align = (:center, :center), fontsize = 15, font = :bold, color = PALETTE.photon)
    end
    note!(axb, 1.5 * cell_pitch[1], cell_pitch[2] + 0.12, "Grey: the Walls, over the Gaps"; align = (:right, :bottom), color = PALETTE.slab_edge, fontsize = 14)

    # (c) Side view along x: walls over the gaps, a photon on a wall stopped, one in the open passes
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side View (Width Stretched)",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Height Above the Crystal (mm)",
        limits = ((-0.85 * cell_pitch[1], 0.85 * cell_pitch[1]), (-1.9, H + 1.6)),
    )
    rect!(axc, -0.85 * cell_pitch[1], -1.6, 1.7 * cell_pitch[1], 1.6, MATERIALS.cdte)
    for sx in (-1, 1)
        rect!(axc, sx * half_x - g[1] / 2, 0, g[1], H, MATERIALS.tungsten)
    end
    for i in 0:(G[1] - 1)
        rect!(axc, centre(i, 1) - p[1] / 2 + 0.01, -1.75, p[1] - 0.02, 0.1, MATERIALS.electrode)
    end
    Mke.lines!(axc, [0.1, 0.1], [H + 1.3, -0.8]; color = MATERIALS.beam, linewidth = 3)
    Mke.lines!(axc, [half_x, half_x], [H + 1.3, 4.0]; color = MATERIALS.beam, linewidth = 3)
    Mke.scatter!(axc, [Mke.Point2f(half_x, 4.0)]; marker = :xcross, color = PALETTE.xray, markersize = 18)
    note!(axc, 0.06, H - 0.3, "Straight from the\nFocal Spot: Passes"; align = (:right, :top), color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 14)
    note!(axc, half_x - 0.04, 3.4, "On a Wall:\nStopped"; align = (:right, :top), color = PALETTE.xray, fontsize = 14)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    Mke.colsize!(fig.layout, 3, Mke.Relative(0.22))
    locator!(fig[0, :], :grid; note = "Step 1 · The Grid,\nin Front of the CdTe")
    fig
end

# ╔═╡ c6682374-bbef-5638-9592-82c5c8e5aff1
let
    Mke.set_theme!(THEME)
    sdd = 1113.0
    half_z = 24 * cell_pitch[2]
    θ_edge = atand(half_z / sdd)
    fig = Mke.Figure(size = (1600, 760))
    ax = Mke.Axis(
        fig[1, 1];
        title = "(a) The Scanner Along Z (Not to Scale Across)",
        titlesize = 20,
        xlabel = "Distance from the Focal Spot (mm)",
        ylabel = "Z (mm)",
        limits = ((-40, sdd + 230), (-1.6 * half_z, 1.6 * half_z)),
    )
    Mke.scatter!(ax, [Mke.Point2f(0, 0)]; color = MATERIALS.beam, markersize = 18)
    Mke.lines!(ax, [sdd, sdd], [-half_z, half_z]; color = MATERIALS.cdte, linewidth = 12)
    for zt in (-half_z, -half_z / 2, 0, half_z / 2, half_z)
        Mke.lines!(ax, [0, sdd], [0, zt]; color = (MATERIALS.beam, 0.7), linewidth = 1.5)
    end
    note!(ax, 10, 6, "Focal Spot"; align = (:left, :bottom), color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 15)
    note!(ax, sdd - 10, half_z, "Outer Row: $(round(θ_edge; digits = 2))° Cone Angle"; align = (:right, :bottom), color = PALETTE.ink, fontsize = 15)
    note!(ax, sdd + 15, 0, "Detector,\n$(round(2half_z; digits = 1)) mm\nAlong Z"; align = (:left, :center), color = PALETTE.slab_edge, fontsize = 15)

    # (b) In the crystal of an outer row, to scale: a deep photon's charge lands one row over
    p = alpha.dexel[2]
    d = crystal.thickness_mm
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) An Outer Row, to Scale",
        titlesize = 20,
        xlabel = "Z (mm)",
        ylabel = "Depth (mm)",
        aspect = Mke.DataAspect(),
        yreversed = true,
        limits = ((-0.55, 0.85), (-0.55, d + 0.25)),
    )
    rect!(axb, -0.55, 0, 1.4, d, MATERIALS.cdte)
    for k in -1:1
        rect!(axb, (k - 0.5) * p + 0.015, d, p - 0.03, 0.06, MATERIALS.electrode)
        Mke.lines!(axb, [(k + 0.5) * p, (k + 0.5) * p], [0, d]; color = (PALETTE.ink, 0.3), linestyle = :dash)
    end
    z_in = 0.5p - 0.06
    stop = z_in + d * 0.95 * tand(θ_edge)
    Mke.lines!(axb, [z_in - 0.45 * tand(θ_edge), stop], [-0.45, 0.95d]; color = MATERIALS.beam, linewidth = 3)
    Mke.scatter!(axb, [Mke.Point2f(stop, 0.95d)]; marker = :star5, markersize = 18, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam)
    note!(axb, -0.52, 0.25, "Absorbed Deep, a $(round(θ_edge; digits = 1))° Photon\nHas Crossed into the Next\nRow's Territory"; align = (:left, :center), color = PALETTE.ink, fontsize = 14)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.42))
    locator!(fig[0, :], (:grid, :cdte); note = "Steps 1–3 · The Outer\nRows' Photons Arrive Tilted")
    fig
end

# ╔═╡ 313586e2-12f3-49c0-8418-027e1f21533c
let
    Mke.set_theme!(THEME)
    p, G, g = alpha.dexel, alpha.group, alpha.gap
    fig = Mke.Figure(size = (1500, 900))
    ax = Mke.Axis(
        fig[1, 1];
        title = "One Grid Cell: 4 × 6 Native Dexels, Read as 2 × 3 Binned Dexels",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Along Z (mm)",
        aspect = Mke.DataAspect(),
    )
    rect!(ax, -g[1] / 2, -g[2] / 2, cell_pitch[1], cell_pitch[2], MATERIALS.tungsten)
    corner_colour, middle_colour = Mke.RGBf(0.78, 0.86, 0.98), Mke.RGBf(0.86, 0.93, 0.84)
    for k in 0:3, l in 0:5
        c = dexel_corner(k, l) .- g ./ 2
        rect!(ax, c[1], c[2], p[1], p[2], l in (2, 3) ? middle_colour : corner_colour; strokecolor = PALETTE.slab_edge, strokewidth = 0.8)
    end
    for k in 0:2:2, l in 0:2:4
        c = dexel_corner(k, l) .- g ./ 2
        Mke.poly!(ax, Mke.Rect2f(c[1], c[2], 2p[1], 2p[2]); color = (:white, 0), strokecolor = PALETTE.photon, strokewidth = 3, linestyle = :dash)
    end
    # a photon on the edge between two members of one group: both count it, the group twice
    c = dexel_corner(0, 0) .- g ./ 2
    star = Mke.Point2f(c[1] + p[1], c[2] + 0.55p[2])
    Mke.scatter!(ax, [star]; marker = :star5, markersize = 26, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.5)
    Mke.poly!(ax, Mke.Circle(star, 0.06); color = (MATERIALS.charge, 0.25), strokecolor = MATERIALS.charge)
    note!(ax, cell_pitch[1] - 0.1, 0.25, "A Photon Shared Between Two Members\nof One Group: Each Counts It, So the\nBinned Dexel Counts It Twice"; align = (:left, :center), color = MATERIALS.charge, fontsize = 15)
    note!(ax, cell_pitch[1] - 0.1, 1.05, "Green: the Middle Group (E, F)\nBlue: the Corner Groups (A–D)"; align = (:left, :center), color = PALETTE.ink, fontsize = 15)
    note!(ax, cell_pitch[1] - 0.1, 1.75, "Dashed: One Binned Dexel,\n2 × 2 Native (0.55 × 0.644 mm)"; align = (:left, :center), color = PALETTE.photon, fontsize = 15)
    Mke.xlims!(ax, -0.3, cell_pitch[1] + 1.6)
    locator!(fig[0, :], :asic; note = "Step 5 · Counting,\nThen Adding 2 × 2")
    fig
end

# ╔═╡ bfee1bcb-7273-5f67-b684-42e8be79191d
detector_b = MC.ArcDetector(; columns = 1984, rows = 288)

# ╔═╡ 97e89e01-c1b8-4d6e-892f-b17751b636d1
let
    Mke.set_theme!(THEME)
    night, ink, sub = MATERIALS.night, Mke.RGBf(0.93, 0.94, 0.96), Mke.RGBf(0.6, 0.63, 0.68)
    fig = Mke.Figure(size = (1500, 600), backgroundcolor = night, figure_padding = (10, 10, 10, 10))

    # (1) The gantry, to scale: tube A with detector A (50 cm), tube B with detector B (36 cm)
    ax1 = canvas(fig[1, 1]; background = night)
    R_iso, R_det = 610.0, detector_a.radius_mm
    θ = range(0, 2π; length = 200)
    Mke.poly!(ax1, Mke.Point2f.(830 .* cos.(θ), 830 .* sin.(θ)); color = MATERIALS.night_ring)
    Mke.poly!(ax1, Mke.Point2f.(400 .* cos.(θ), 400 .* sin.(θ)); color = night)
    Mke.poly!(ax1, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = Mke.RGBf(0.42, 0.45, 0.5))
    for x in (-75, 75)
        Mke.poly!(ax1, Mke.Point2f.(x .+ 50 .* cos.(θ), 10 .+ 60 .* sin.(θ)); color = Mke.RGBf(0.16, 0.18, 0.21))
    end
    Mke.poly!(ax1, Mke.Point2f.(22 .* cos.(θ), -105 .+ 20 .* sin.(θ)); color = Mke.RGBf(0.85, 0.85, 0.82))
    rot(v, t) = Mke.Point2f(v[1] * cos(t) - v[2] * sin(t), v[1] * sin(t) + v[2] * cos(t))
    for (angle_deg, d, alpha_fill, label) in ((90, detector_a, 0.22, "Tube A"), (90 + 95, detector_b, 0.14, "Tube B"))
        a = deg2rad(angle_deg)
        src = Mke.Point2f(R_iso * cos(a), R_iso * sin(a))
        toward = -Mke.Point2f(cos(a), sin(a))
        half = deg2rad(MC.fan_angle_deg(d)) / 2
        φ = range(-half, half; length = 80)
        arc = [src + R_det * rot(toward, t) for t in φ]
        Mke.poly!(ax1, Mke.Point2f.(vcat([src], arc)); color = (MATERIALS.beam, alpha_fill))
        inner = [src + (R_det + 8) * rot(toward, t) for t in φ]
        outer = [src + (R_det + 50) * rot(toward, t) for t in φ]
        Mke.poly!(ax1, Mke.Point2f.(vcat(inner, reverse(outer))); color = MATERIALS.cdte)
        Mke.poly!(ax1, Mke.Rect2f(src[1] - 45, src[2] - 45, 90, 90); color = Mke.RGBf(0.8, 0.83, 0.88))
        Mke.scatter!(ax1, [src]; color = MATERIALS.beam, markersize = 12)
        Mke.text!(ax1, src + Mke.Point2f(0, 85); text = label, color = ink, fontsize = 17, font = :bold, align = (:center, :center))
    end
    Mke.limits!(ax1, -1030, 1030, -1030, 1030)

    # (2) A patch of the face, to scale: CdTe under the 2-D grid, the near cells cut away
    ax2 = canvas(fig[1, 2]; background = night)
    cam = camera(-52, 28)
    nx, ny = 8, 12
    ex = exploded_alpha(nx, ny; grid_drawn = 5.0)
    keep = Block[]
    append!(keep, ex.layers[:cdte])
    append!(keep, ex.layers[:cathode])
    zc = ex.levels[:cathode][2]
    for b in ex.layers[:grid]
        b2 = Block((b.lo[1], b.lo[2], zc), (b.hi[1], b.hi[2], zc + 5.0), b.color)
        b2.hi[1] > 0.3 && b2.lo[2] < 1.6 * cell_pitch[2] && continue
        push!(keep, b2)
    end
    rays = map([(2, 1), (5, 3), (6, 6)]) do (k, l)
        c = dexel_corner(k, l) .+ alpha.dexel ./ 2
        a, b = project(cam, (c..., zc + 7.5)), project(cam, (c..., zc + 0.05))
        (1.0e4, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 12))
    end
    draw_blocks!(ax2, cam, keep; extras = rays, edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    # (3) One dexel, taken apart
    ax3 = canvas(fig[1, 3]; background = night)
    draw_blocks!(ax3, camera(-56, 30), exploded_alpha(1, 1; gap = 0.35, grid_drawn = 1.1, depth = 1.0, thin = 0.6).blocks; edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    for (k, (title, subtitle)) in enumerate(
            (
                ("Two Tubes, Two Detectors", "The NAEOTOM Alpha: Detector A 50 cm, B 36 cm"),
                ("CdTe Under a 2-D Grid", "One Crystal; Walls Around Every 4 × 6 Dexels"),
                ("One Dexel, Taken Apart", "Grid, Cathode, CdTe, Electrode, Chip, Board"),
            )
        )
        Mke.Label(fig[2, k], title; color = ink, fontsize = 26, font = :bold, tellwidth = false)
        Mke.Label(fig[3, k], subtitle; color = sub, fontsize = 18, tellwidth = false)
    end
    Mke.rowgap!(fig.layout, 1, 8)
    Mke.rowgap!(fig.layout, 2, 4)
    fig
end

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

# ╔═╡ 8a8a6147-7140-4c47-8268-4d0002045acf
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 1000))
    ax = canvas(fig[1, 1]; title = "Detector A, from the Focal Spot (Depth Exaggerated)", titlesize = 20)
    cam = camera(-62, 26)
    R = detector_a.radius_mm
    half = deg2rad(MC.fan_angle_deg(detector_a)) / 2
    half_z = MC.face_size_mm(detector_a)[2] / 2
    amber = MATERIALS.beam
    items = arc_quads(cam, R, half, half_z; thickness = 25.0, color = MATERIALS.cdte)
    focal = (0.0, 0.0, 0.0)
    corners = [(R * sin(a), -R * cos(a), z) for (a, z) in ((-half, -half_z), (half, -half_z), (half, half_z), (-half, half_z))]
    iso = (0.0, -610.0, 0.0)
    push!(
        items,
        (
            1.0e4,
            a -> begin
                for (c1, c2) in zip(corners, circshift(corners, -1))
                    Mke.poly!(a, Mke.Point2f[project(cam, focal), project(cam, c1), project(cam, c2)]; color = (amber, 0.1))
                end
                for c in corners
                    Mke.lines!(a, [project(cam, focal), project(cam, c)]; color = (amber, 0.7), linewidth = 1.5)
                end
                Mke.scatter!(a, [project(cam, focal)]; color = PALETTE.ink, markersize = 14)
                Mke.scatter!(a, [project(cam, iso)]; marker = :cross, markersize = 18, color = PALETTE.ink)
            end,
        ),
    )
    sort!(items; by = first)
    for (_, f) in items
        f(ax)
    end
    callout!(ax, project(cam, focal), project(cam, focal) + Mke.Point2f(150, 80), "Focal Spot"; fontsize = 16)
    callout!(ax, project(cam, iso), project(cam, iso) + Mke.Point2f(260, 60), "Isocentre, 610 mm Away"; fontsize = 16)
    edge = (R * sin(half), -R * cos(half), half_z)
    callout!(ax, project(cam, edge), project(cam, edge) + Mke.Point2f(120, 120), "$(detector_a.columns) Columns × $(detector_a.rows) Rows of Native Dexels\n$(round(MC.fan_angle_deg(detector_a); digits = 1))° Fan, at $(round(R; digits = 1)) mm"; fontsize = 16)
    low = (R * sin(-half / 2), -R * cos(-half / 2), -half_z)
    callout!(ax, project(cam, low), project(cam, low) + Mke.Point2f(-120, -160), "The Outer Rows See the Beam\nat the Cone Angle (Section 6)"; fontsize = 16, align = (:right, :center))
    xs = [project(cam, c)[1] for c in vcat(corners, [focal])]
    ys = [project(cam, c)[2] for c in vcat(corners, [focal])]
    Mke.limits!(ax, minimum(xs) - 520, maximum(xs) + 650, minimum(ys) - 240, maximum(ys) + 200)
    locator!(fig[0, :], :arc; note = "The Whole Arc:\n$(detector_a.columns) × $(detector_a.rows) Dexels")
    fig
end

# ╔═╡ 39c48e80-8b7e-5910-9909-463d8f715b0f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 700))
    R = detector_a.radius_mm

    # (a) The gantry from the patient's feet: focal spot and the two detector arcs
    ax = Mke.Axis(
        fig[1, 1];
        title = "(a) The Detectors Around the Focal Spot",
        xlabel = "X (mm)",
        ylabel = "Y (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-620, 620), (-150, 1420)),
    )
    for (d, color, label) in ((detector_a, PALETTE.photon, "Detector A"), (detector_b, PALETTE.xray, "Detector B"))
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
        "50 cm Scan Field\n(Detector A)";
        align = (:center, :center),
        color = PALETTE.muted,
        fontsize = 14,
    )
    note!(
        ax,
        15,
        -20,
        "Focal Spot";
        align = (:left, :top),
        color = PALETTE.xray,
        fontsize = 15,
    )
    note!(
        ax,
        0,
        R + 110,
        "B Is Drawn Outside A for Clarity:\nBoth Sit 1113 mm from Their Own Focal Spot";
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
        title = "(b) A Patch of the Face: 3 × 2 of Its 688 × 48 Grid Cells",
        xlabel = "Along the Arc (mm)",
        ylabel = "Z (mm)",
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
        "Blue: Dexels   Grey: Walls over the Gaps";
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
        title = "(a) Every Dexel of Detector A: Counts Above 20 keV per Photon Landing on It",
        xlabel = "Dexel Column",
        ylabel = "Row",
    )
    hm = Mke.heatmap!(
        ax,
        1:detector_a.columns,
        1:detector_a.rows,
        per_photon;
        colormap = :viridis,
        colorrange = (0.6, 1.3),
    )
    Mke.Colorbar(fig[1, 3], hm; label = "Counts per Photon")

    # (b) A close-up: three grid cells across, two along z
    axz = Mke.Axis(
        fig[2, 1];
        title = "(b) Close-Up: the 4 × 6 Cells",
        xlabel = "Dexel Column",
        ylabel = "Row",
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
        title = "(c) By Position in the Cell",
        xlabel = "Position in the Grid Cell",
        ylabel = "Counts Above 20 keV per Photon",
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
        label = "Whole Detector (This Air Scan)",
    )
    Mke.scatter!(
        axc,
        1:6,
        [grid_counts[k][e60].total[1] for k in 1:6];
        color = PALETTE.xray,
        marker = :diamond,
        markersize = 16,
        label = "One Dexel at a Time, 3 × 3 Total (Section 5)",
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

# ╔═╡ c81376fc-7b62-4e44-af3b-ec6cba4a8a9c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))
    ax = canvas(fig[1, 1])
    amber = MATERIALS.beam
    Mke.poly!(ax, Mke.Rect2f(-0.6, -1.5, 2.4, 3.0); color = Mke.RGBf(0.93, 0.94, 0.96), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    rect!(ax, -0.45, -0.25, 0.3, 0.5, PALETTE.slab_edge)
    Mke.text!(ax, -0.3, -0.45; text = "Cathode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.poly!(ax, Mke.Point2f[(1.2, -0.9), (1.55, -0.9), (1.55, 0.9), (1.35, 0.9)]; color = Mke.RGBf(0.6, 0.45, 0.3))
    Mke.text!(ax, 1.42, -1.0; text = "Tungsten Anode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    arrow!(ax, (-0.1, 0.0), (1.22, 0.0); color = PALETTE.photon, linewidth = 2.5, head = 11)
    Mke.text!(ax, 0.55, 0.12; text = "Electrons", align = (:center, :bottom), fontsize = 14, color = PALETTE.photon)
    Mke.text!(ax, 0.6, 1.6; text = "X-Ray Tube, 140 kVp", align = (:center, :bottom), fontsize = 18, font = :bold, color = PALETTE.ink)
    Mke.text!(ax, 0.6, -1.65; text = "Anode Angle: 8° Documented;\nBasisSimulator v0.15.1: 10° Spectrum", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    focal = Mke.Point2f(1.28, 0.0)
    x_end = 13.5
    Mke.poly!(ax, Mke.Point2f[focal, (x_end, 2.6), (x_end, -2.6)]; color = (amber, 0.13))
    for y in (-2.6, 0.0, 2.6)
        Mke.lines!(ax, [focal, Mke.Point2f(x_end, y)]; color = (amber, 0.5), linewidth = 1)
    end
    filters = (
        (2.4, 0.18, Mke.RGBf(0.75, 0.77, 0.8), "3 mm Al", "Always"),
        (4.0, 0.12, Mke.RGBf(0.6, 0.62, 0.68), "0.9 mm Ti", "BasisSimulator v0.15.1's\nAlpha Protocol"),
        (5.7, 0.08, Mke.RGBf(0.55, 0.6, 0.5), "0.4 mm Sn", "Documented on Both\nTubes; Not Used Here"),
    )
    for (x, w, c, name, when) in filters
        h = 0.25 + (x - focal[1]) * 2.6 / (x_end - focal[1])
        rect!(ax, x, -h, w, 2h, c; strokecolor = (:black, 0.3))
        Mke.text!(ax, x + w / 2, h + 0.12; text = name, align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
        Mke.text!(ax, x + w / 2, -h - 0.12; text = when, align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    end
    xb = 7.2
    hb = 0.25 + (xb - focal[1]) * 2.6 / (x_end - focal[1])
    Mke.poly!(ax, Mke.Point2f[(xb, -hb), (xb + 0.55, -hb), (xb + 0.08, 0), (xb + 0.55, hb), (xb, hb)]; color = Mke.RGBf(0.75, 0.77, 0.8), strokecolor = (:black, 0.3))
    Mke.text!(ax, xb + 0.25, hb + 0.12; text = "Bowtie", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.text!(ax, xb + 0.25, -hb - 0.12; text = "2.5 mm Al\nat the Centre", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    θ = range(0, 2π; length = 100)
    xi = 10.3
    Mke.poly!(ax, Mke.Point2f.(xi .+ 1.1 .* cos.(θ), 0.8 .* sin.(θ)); color = (MATERIALS.water, 0.6), strokecolor = (PALETTE.photon, 0.5))
    Mke.scatter!(ax, [Mke.Point2f(xi, 0)]; marker = :cross, markersize = 16, color = PALETTE.ink)
    Mke.text!(ax, xi, 0.95; text = "Patient (Isocentre, 610 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    rect!(ax, x_end, -2.7, 0.15, 5.4, MATERIALS.cdte; strokecolor = PALETTE.slab_edge)
    for y in range(-2.6, 2.6; length = 8)
        rect!(ax, x_end - 0.35, y - 0.012, 0.35, 0.024, MATERIALS.tungsten)
    end
    Mke.text!(ax, x_end + 0.1, 2.85; text = "Grid + CdTe\n(1113 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -1.4, 14.6, -3.4, 3.9)
    locator!(fig[0, :], :tube; note = "The X-Ray Tube and\nWhat the Beam Passes")
    fig
end

# ╔═╡ 877c4989-7b59-5741-a50a-433cf4a4a14b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Photons Reaching the Detector: 140 kVp, 3 mm Al + 0.9 mm Ti",
        xlabel = "Photon Energy (keV)",
        ylabel = "Photons per mAs per mm²\nper keV, at 1113 mm",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    for (angle, color, label) in ((10, PALETTE.photon, "Anode 10° (BasisSimulator v0.15.1)"), (8, PALETTE.xray, "Anode 8° (Documented)"))
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
        "Thresholds: 20, 35, 55, 70 keV";
        align = (:left, :bottom),
        offset = (0, 8),
        color = PALETTE.muted,
        fontsize = 14,
    )
    note!(
        ax,
        59,
        0,
        "Tungsten K Lines";
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
            label = "BasisSimulator v0.15.1: Anode 10°,\n3 mm Al + 0.9 mm Ti",
            hvl = hvl(10, [("Ti", 0.9)]),
        ),
        (
            label = "Anode 8° (Documented),\n3 mm Al + 0.9 mm Ti",
            hvl = hvl(8, [("Ti", 0.9)]),
        ),
        (label = "Anode 8°,\n3 mm Al, No Titanium", hvl = hvl(8, Tuple{String, Float64}[])),
    ]
end

# ╔═╡ 01ccb19c-6a12-4958-b5c0-acb5ea96acfa
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 380))
    for (k, (al, reading, title)) in enumerate(((0, "100 %", "No Aluminium in the Beam"), (7.8, "50 %", "7.7–7.9 mm of Aluminium: Half Is Left")))
        ax = canvas(fig[1, k]; title, titlesize = 20)
        Mke.scatter!(ax, [Mke.Point2f(0, 0)]; marker = :rect, markersize = 22, color = PALETTE.ink)
        Mke.text!(ax, 0, -0.45; text = "Tube\n(140 kVp, Through\nthe Bowtie's Centre)", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
        Mke.poly!(ax, Mke.Point2f[(0.1, 0), (6, 0.45), (6, -0.45)]; color = (MATERIALS.beam, 0.25))
        if al > 0
            for i in 0:3
                rect!(ax, 2.8 + 0.18i, -0.35, 0.12, 0.7, Mke.RGBf(0.75, 0.77, 0.8); strokecolor = (:black, 0.3))
            end
            Mke.text!(ax, 3.1, 0.5; text = "Aluminium Sheets", align = (:center, :bottom), fontsize = 14, color = PALETTE.ink)
        end
        Mke.poly!(ax, Mke.Point2f.(6.0 .+ 0.35 .* cos.(range(0, 2π; length = 60)), 0.35 .* sin.(range(0, 2π; length = 60))); color = Mke.RGBf(0.85, 0.87, 0.9), strokecolor = PALETTE.slab_edge)
        Mke.text!(ax, 6.0, -0.5; text = "Ion Chamber\nat the Isocentre", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
        Mke.textlabel!(ax, Mke.Point2f(6.0, 0.75); text = "Air Kerma: $(reading)", text_align = (:center, :bottom), fontsize = 16, font = :bold, text_color = PALETTE.kept, background_color = (:white, 0.9), strokewidth = 0)
        Mke.limits!(ax, -0.8, 7.0, -1.3, 1.3)
    end
    fig
end

# ╔═╡ f68a9548-e730-4402-af6d-ed2f2a2dc3a9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-Value Layer at the Isocentre, 140 kVp",
        xlabel = "Half-Value Layer (mm of Aluminium)",
        yticks = (1:3, [c.label for c in hvl_check]),
        yreversed = true,
        limits = ((0, 12), (0.4, 4.2)),
    )
    Mke.vspan!(ax, 7.7, 7.9; color = (PALETTE.kept, 0.35))
    note!(
        ax,
        7.8,
        3.85,
        "Measured on the Alpha: 7.7–7.9 mm";
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

- **BasisSimulator v0.15.1**: its response matrix (`compute_mc_drm`, built from `cdte_response_v4`)
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

# ╔═╡ b3df600c-9c49-48f5-89b1-760870751d97
let
    Mke.set_theme!(THEME)
    p, G, g = alpha.dexel, alpha.group, alpha.gap
    fig = Mke.Figure(size = (1600, 560))

    # (a) One photon, shared by two dexels: the detector counts it twice; v0.15.1 rescales to one
    ax = canvas(fig[1, 1]; title = "(a) One Shared Photon: Two Counts, or One?", titlesize = 20)
    for k in 0:1
        rect!(ax, k * 1.1, 0, 1.0, 0.6, MATERIALS.cdte_lit; strokecolor = PALETTE.slab_edge)
        Mke.textlabel!(ax, Mke.Point2f(k * 1.1 + 0.5, 0.3); text = "+1", text_align = (:center, :center), fontsize = 18, font = :bold, text_color = :white, background_color = PALETTE.photon, strokewidth = 0, cornerradius = 8)
    end
    Mke.scatter!(ax, [Mke.Point2f(1.05, 0.62)]; marker = :star5, markersize = 26, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.5)
    Mke.text!(ax, 1.05, -0.15; text = "The Detector (No Anti-Coincidence Logic\nDocumented): Both Counts Kept, 2 per Photon", align = (:center, :top), fontsize = 15, color = PALETTE.ink)
    Mke.text!(ax, 1.05, -0.75; text = "BasisSimulator v0.15.1: Any Energy That Makes\nMore Than 1 Count per Photon Is Scaled Back to 1", align = (:center, :top), fontsize = 15, color = PALETTE.reference)
    Mke.limits!(ax, -0.3, 2.4, -1.35, 1.0)

    # (b) The share of the face that sees photons: v0.15.1's fill factor against the grid's open area
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Share of the Face That Collects Photons",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Along Z (mm)",
        aspect = Mke.DataAspect(),
    )
    rect!(axb, -g[1] / 2, -g[2] / 2, cell_pitch[1], cell_pitch[2], MATERIALS.tungsten)
    for k in 0:3, l in 0:5
        c = dexel_corner(k, l) .- g ./ 2
        rect!(axb, c[1], c[2], p[1], p[2], MATERIALS.cdte_lit; strokecolor = PALETTE.slab_edge, strokewidth = 0.6)
    end
    open_fraction = prod(G .* p) / prod(cell_pitch)
    note!(axb, cell_pitch[1] - 0.05, 1.3, "Open (Light): $(round(open_fraction; digits = 2)) of the Face\nWith the Grid, as Built Here"; align = (:left, :center), color = PALETTE.photon, fontsize = 15)
    note!(axb, cell_pitch[1] - 0.05, 0.5, "BasisSimulator v0.15.1:\nFill Factor 0.95"; align = (:left, :center), color = PALETTE.reference, fontsize = 15)
    Mke.xlims!(axb, -0.3, cell_pitch[1] + 1.5)
    fig
end

# ╔═╡ 5138ed44-8263-5dbf-9d6e-d78c085c4a85
let
    Mke.set_theme!(THEME)
    labels = ["20–35", "35–55", "55–70", "> 70"]
    fig = Mke.Figure(size = (1200, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Counts in Each Energy Bin, per Photon Reaching the Detector (140 kVp)",
        xlabel = "Energy Bin (keV)",
        ylabel = "Counts per Photon",
        xticks = (1:4, labels),
        limits = (nothing, (0, 0.42)),
    )
    series = (
        (
            bs_bins,
            PALETTE.reference,
            "BasisSimulator v0.15.1 (cdte_response_v4, Capped at 1 Count per Photon, × 0.95)",
        ),
        (
            geant4_bins_capped,
            (PALETTE.reference, 0.45),
            "Geant4, One Dexel, with the Same Cap and × 0.95",
        ),
        (geant4_bins, PALETTE.muted, "Geant4, One Dexel, No Grid, Every Count Kept"),
        (whole_bins, PALETTE.photon, "Geant4, Whole Detector A with Its Grid"),
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

# ╔═╡ dc89fc6d-57b5-42e2-97ba-04cb2fb1614d
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 800))
    ax = Mke.Axis(
        fig[1, 1];
        title = "One Dexel's Pulses in 60 ns, at Clinical Rates",
        titlesize = 20,
        xlabel = "Time (ns)",
        ylabel = "Pulse Height (keV)",
        limits = ((0, 60), (0, 110)),
        yticks = [20, 35, 55, 70],
    )
    τ = 5.0
    shape(t, t0, E) = t < t0 ? 0.0 : E * ((t - t0) / 2.5) * exp(1 - (t - t0) / 2.5)
    pulses = [(4.0, 45.0), (19.0, 62.0), (21.5, 40.0), (38.0, 30.0), (50.0, 75.0)]
    t = range(0, 60; length = 1200)
    for (t0, _) in pulses
        Mke.vspan!(ax, t0, t0 + τ; color = (PALETTE.muted, 0.12))
    end
    signal = [sum(shape(x, t0, E) for (t0, E) in pulses) for x in t]
    for level in (20, 35, 55, 70)
        Mke.hlines!(ax, level; color = (MATERIALS.beam, 0.7), linewidth = 1.2, linestyle = :dash)
    end
    Mke.lines!(ax, t, signal; color = PALETTE.photon, linewidth = 3)
    note!(ax, 20.2, 104, "Two Photons 2.5 ns Apart Merge:\nOne Count, at a Higher Energy (Pile-Up)"; align = (:center, :top), color = MATERIALS.charge, fontsize = 15)
    note!(ax, 59, 10, "Grey: the $(round(Int, τ)) ns Dead Time After Each Pulse"; align = (:right, :bottom), color = PALETTE.muted, fontsize = 14)
    note!(ax, 1, 72, "Dashed: the Four Thresholds"; align = (:left, :bottom), color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 14)
    locator!(fig[0, :], :asic; note = "Step 5 at Clinical\nCount Rates")
    fig
end

# ╔═╡ 4c43a065-f125-5640-8df8-a2b16fdbdd4e
let
    Mke.set_theme!(THEME)
    labels = ["20–35", "35–55", "55–70", "> 70"]
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "What Pile-Up Does at $(round(photon_rate / 1.0e6; digits = 1)) Million Photons per Second per Dexel",
        xlabel = "Energy Bin (keV)",
        ylabel = "Recorded ÷ True Counts",
        xticks = (1:4, labels),
        limits = (nothing, (0.5, nothing)),
    )
    Mke.hlines!(ax, 1; color = PALETTE.ink, linewidth = 1.5)
    series = (
        ((:incident, 5.0), PALETTE.reference, "Incident Spectrum, 5.0 ns (BasisSimulator v0.15.1)"),
        (
            (:incident, 6.0),
            (PALETTE.reference, 0.5),
            "Incident Spectrum, 6.0 ns (Measured 5.3–7.0)",
        ),
        ((:pulses, 5.0), PALETTE.photon, "Geant4 Pulse Heights, 5.0 ns"),
        ((:pulses, 6.0), (PALETTE.photon, 0.5), "Geant4 Pulse Heights, 6.0 ns"),
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

# ╔═╡ 399f9fe4-230f-450c-80b4-669326efdbf2
md"""
### Pulses Shaped and Counted, Fitted to the Clinical Alpha

The comparison above feeds BasisSimulator's parametric model a dead time. Here the pile-up follows
from the pulses themselves, and its two free numbers are fitted to the clinical Alpha:

- **The pulses.** Every dexel that collects charge from a photon makes a pulse, its height the
  charge that dexel recorded with the charge transported (section 3): more pulses than photons,
  many of them small. A pulse below the lowest threshold is never counted, but it still occupies
  the shaper and can lift its neighbour over a threshold.
- **The shaping.** Each pulse is a CR-RC pulse, 10–15 ns wide at half its height (Flohr & Schmidt
  2023), with Zhan's 1.9 keV electronic noise and the Fano noise on its height.
- **The counting.** One comparator per threshold counts at most once per dead time τ_d. A plain
  comparator counts each upward crossing of its threshold, so pulses that overlap above it count
  once, and a merged pulse can land in a higher energy bin; a retriggering one counts every peak
  above its threshold (Ballabriga et al. 2016 describe both). Which one the Alpha uses is not
  published; the fit chooses.
- **The binning is digital** (Li et al. 2025; Yu et al. 2016 for the prototype): each subpixel
  counts its own pulses at a quarter of the binned pixel's rate, and standard mode adds the four
  counters. So a subpixel loses the same share of its counts at a given tube current in both modes.
- **Fitted** to Li et al. 2025, who measured two Alphas with Siemens' pile-up correction off: their
  low-flux counts per binned pixel per mAs (β, Table 2) and their dead-time fits to the measured
  noise against tube current, per kV. Their loss curve, with a band from their 95 % confidence
  interval on the dead time (5.6–7.1 ns), is the reference; the shaper's width (within 10–15 ns)
  τ_d and the comparator logic are fitted to it at 120 and 140 kV, the voltages BasisSimulator has spectra for.
- **Assumed**, because the Alpha's ASIC is not published: the CR-RC shape, noise added to each
  pulse's height, and no pile-up trigger.

One tension stays open. Li fit a dead time to the counts per **read-out** pixel and find about
the same value in UHR mode, where a pixel is one subpixel with a quarter of the rate. Per-subpixel
pile-up with digital binning would instead give UHR the same loss at a given tube current, so four
times the dead time on Li's scale. This notebook follows the binning Li infer and fits standard
mode; its UHR pile-up is a prediction.
"""

# ╔═╡ 67a5c658-bcfe-41b5-a154-f5f04a8cfd9f
# The pulses one native dexel makes per photon landing on it, at every energy, with the charge
# transported: every dexel of the 3 × 3 that recorded charge from a photon makes a pulse (by
# symmetry, a neighbour's pulse from a photon on the centre is the centre's pulse from a photon on
# that neighbour)
pulse_heights = let
    edges = 0.0:0.5:150.0
    floor_keV = 1.0
    histograms = permutedims(reduce(hcat, [MC.pulse_heights(r; floor_keV, edges_keV = edges) for r in transport_runs.responses]))
    (
        energies = collect(energies),
        edges = collect(edges),
        centres = collect(edges[1:(end - 1)] .+ 0.25),
        floor_keV,
        histograms,
        per_photon = vec(sum(histograms; dims = 2)),
    )
end

# ╔═╡ cf22256e-4caa-4a14-86f2-6bda4c461ff8
# Li et al. 2025 (Med Phys 52:e70045), two NAEOTOM Alphas in service mode with the pile-up
# correction off, standard mode, scanner 1: low-flux output counts per binned pixel per mAs (β,
# Table 2), and the composite dead-time model fitted to the measured noise against tube current
# (non-paralyzable with weight 1 − w, paralyzable with weight w), per kV; the 95 % confidence
# interval on the dead time, from all kV, is 5.6–7.1 ns
li2025 = (
    beta = Dict(70 => 0.83e5, 90 => 1.82e5, 120 => 3.61e5, 140 => 4.93e5),
    tau_ns = Dict(70 => 5.9, 90 => 7.0, 120 => 6.7, 140 => 6.1),
    w = Dict(70 => 0.0, 90 => 0.11, 120 => 0.05, 140 => 0.05),
    tau_ci_ns = (5.6, 7.1),
    max_mA = Dict(70 => 1200.0, 90 => 1200.0, 120 => 1000.0, 140 => 850.0),
)

# ╔═╡ e4f9fe55-8ec8-44df-9d56-d70389f7015c
"""Li et al. 2025's count loss at tube current `mA` (air, standard mode): their composite dead-time model."""
function li_loss(kV, mA; τ_ns = li2025.tau_ns[kV], w = li2025.w[kV])
    x = li2025.beta[kV] * mA * τ_ns * 1.0e-9
    return 1 - ((1 - w) / (1 + x) + w * exp(-x))
end

# ╔═╡ 7a122efb-af3a-4f8f-862d-8196f6246c53
# The Alpha's 120 and 140 kV beams at the detector, from the installed BasisSimulator (photons per
# mAs per mm², 0.5 keV bins): the voltages at which it has spectra, and Li's 120 and 140 kV
li_beams = Dict(
    kV => BS.resolve_source_spectrum_without_bowtie(
            BS.SimOptions(),
            BS.CTProtocol(;
                kVp = kV,
                mA = 1.0,
                views = 1200,
                rotation_time = 0.5,
                collimation_mm = 5.0,
                anode_angle = 8,
                additional_filters = [("Ti", 0.9)],
            );
            scanner = bs_scanner,
        )
        for kV in (120, 140)
)

# ╔═╡ 0cb750de-25a1-4668-9665-42b7a8fca2f7
"""
The pulses a native dexel makes under `spectrum = (e, w)`: their heights (`values`, cumulative
`cdf`, probabilities `p`), pulses per photon, the height noise `noise(h)`, and the share of pulses
that cross each of `thresholds` when they arrive alone.
"""
function pulse_mix(spectrum, thresholds; noise_keV = zhan.noise_keV, fano = zhan.fano, pair_eV = 4.43)
    e, w = spectrum
    weights = zeros(length(pulse_heights.centres))
    photons = 0.0
    for (E, wi) in zip(e, w)
        k = round(Int, E)
        1 <= k <= length(pulse_heights.energies) || continue
        weights .+= wi .* pulse_heights.histograms[k, :]
        photons += wi
    end
    p = weights ./ sum(weights)
    noise(h) = sqrt(noise_keV^2 + fano * pair_eV / 1000 * max(h, 0.0))
    return (
        values = pulse_heights.centres,
        cdf = cumsum(p),
        p,
        per_photon = sum(weights) / photons,
        noise,
        crossing = MC.counts_above(pulse_heights.centres, p, thresholds; electronic_noise_keV = noise_keV, fano, pair_energy_eV = pair_eV),
    )
end

# ╔═╡ c4a1eb26-d693-427b-9b8f-95d586a55a88
# The shaper's width and the comparators' dead time that make the shaped pulses lose counts as the
# clinical Alpha did: a grid search over 10–15 ns (Flohr & Schmidt 2023), 0–40 ns and the two comparator
# logics (plain, or retriggering on every peak), against Li's loss curves at 120 and 140 kV. Each loss is against the same pulses arriving far apart (same seed),
# so the comparison carries no sampling noise of its own.
pileup_fit = let
    fit_mA = Dict(
        120 => [50.0, 100.0, 200.0, 350.0, 500.0, 700.0, 1000.0],
        140 => [50.0, 100.0, 200.0, 350.0, 500.0, 700.0, 850.0],
    )
    mixes = Dict(kV => pulse_mix(li_beams[kV], [20.0]) for kV in (120, 140))
    n_pulses = 15_000
    function losses(fwhm, dead, retrigger, kV, mAs)
        m = mixes[kV]
        alone = MC.pulse_counts(BS.Random.Xoshiro(11), MC.PulseShaper(; fwhm_ns = fwhm, dead_ns = dead, retrigger), 1.0e-7, n_pulses, m.values, m.cdf, [20.0]; noise = m.noise)[1]
        return map(mAs) do I
            rate = li2025.beta[kV] * I / 4 / m.crossing[1] * 1.0e-9  # pulses per ns on one subpixel
            counted = MC.pulse_counts(BS.Random.Xoshiro(11), MC.PulseShaper(; fwhm_ns = fwhm, dead_ns = dead, retrigger), rate, n_pulses, m.values, m.cdf, [20.0]; noise = m.noise)[1]
            1 - counted / alone
        end
    end
    grid = [(f, d, r) for f in (10.0, 12.5, 15.0) for d in 0.0:1.0:40.0 for r in (false, true)]
    cost(f, d, r) = sum(
        sum(abs2, losses(f, d, r, kV, fit_mA[kV]) .- [li_loss(kV, I) for I in fit_mA[kV]])
            for kV in (120, 140)
    )
    costs = [cost(g...) for g in grid]
    fwhm, dead, retrigger = grid[argmin(costs)]
    best_plain = minimum(c for (c, g) in zip(costs, grid) if !g[3])
    best_retrigger = minimum(c for (c, g) in zip(costs, grid) if g[3])
    curves = Dict(kV => (mA = fit_mA[kV], loss = losses(fwhm, dead, retrigger, kV, fit_mA[kV])) for kV in (120, 140))
    (; fwhm_ns = fwhm, dead_ns = dead, retrigger, best_plain, best_retrigger, grid, costs, curves, mixes, n_pulses)
end

# ╔═╡ 8b62777b-7fe3-4637-aa92-73ba515ddb45
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 1000))

    # (a) One subpixel's shaped pulses and its lowest comparator, with a dead time
    fwhm, dead, T = 12.0, 20.0, 20.0
    τ = fwhm / MC.shaper_fwhm(1)
    arrivals = [(5.0, 48.0), (34.0, 58.0), (37.5, 36.0), (70.0, 30.0), (82.0, 64.0)]
    t = collect(range(0, 110; length = 2200))
    signal = [sum(A * MC.shaper_pulse(x - t0, τ, 1) for (t0, A) in arrivals) for x in t]
    counted, missed = Float64[], Float64[]
    last = -Inf
    for k in 2:length(t)
        if signal[k - 1] < T <= signal[k]
            t[k] - last >= dead ? (push!(counted, t[k]); last = t[k]) : push!(missed, t[k])
        end
    end
    ax = Mke.Axis(
        fig[1, 1];
        title = "(a) One Subpixel's Shaped Pulses and Its 20 keV Comparator",
        xlabel = "Time (ns)",
        ylabel = "Pulse Height (keV)",
        limits = ((0, 110), (0, 105)),
    )
    for c in counted
        Mke.vspan!(ax, c, c + dead; color = (PALETTE.muted, 0.13))
    end
    Mke.hlines!(ax, T; color = MATERIALS.beam, linestyle = :dash, linewidth = 2)
    Mke.lines!(ax, t, signal; color = PALETTE.photon, linewidth = 3)
    Mke.scatter!(ax, counted, fill(T, length(counted)); color = PALETTE.kept, markersize = 18)
    Mke.scatter!(ax, missed, fill(T, length(missed)); color = PALETTE.electron, marker = :xcross, markersize = 18)
    note!(ax, 36, 99, "Two Pulses 3.5 ns Apart Merge:\nOne Count, Higher Up"; align = (:center, :top), fontsize = 15, color = MATERIALS.charge)
    note!(ax, 86, 92, "Inside the Dead Time:\nNot Counted"; align = (:center, :top), fontsize = 15, color = PALETTE.electron)
    note!(ax, 108, 8, "Grey: $(round(Int, dead)) ns Dead Time After Each Count · Pulses $(round(Int, fwhm)) ns Wide"; align = (:right, :bottom), fontsize = 14, color = PALETTE.muted)
    note!(ax, 1, 23, "20 keV Threshold"; align = (:left, :bottom), fontsize = 14, color = Mke.RGBf(0.8, 0.5, 0.05))

    # (b) Standard mode: four subpixels, each with its own counter, added after counting
    axb = canvas(fig[1, 2]; title = "(b) Standard Mode: Four Counters, Added", titlesize = 20)
    counts = [7, 6, 8, 7]
    for (k, (i, j)) in enumerate(((0, 1), (1, 1), (0, 0), (1, 0)))
        rect!(axb, i * 1.15, j * 1.3, 1.0, 1.15, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        Mke.text!(axb, i * 1.15 + 0.5, j * 1.3 + 0.75; text = "Subpixel", align = (:center, :center), fontsize = 14, color = PALETTE.slab_edge)
        Mke.textlabel!(axb, Mke.Point2f(i * 1.15 + 0.5, j * 1.3 + 0.38); text = "Counted $(counts[k])", text_align = (:center, :center), fontsize = 14, font = :bold, text_color = :white, background_color = PALETTE.kept, strokewidth = 0, cornerradius = 7)
    end
    arrow!(axb, (2.35, 1.22), (3.25, 1.22); color = PALETTE.ink, linewidth = 2.5, head = 14)
    rect!(axb, 3.35, 0.55, 1.5, 1.35, Mke.RGBf(0.93, 0.95, 0.93); strokecolor = PALETTE.kept, strokewidth = 2)
    Mke.text!(axb, 4.1, 1.45; text = "Binned Pixel", align = (:center, :center), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.text!(axb, 4.1, 1.0; text = "7 + 6 + 8 + 7\n= $(sum(counts))", align = (:center, :center), fontsize = 15, color = PALETTE.kept)
    Mke.text!(axb, 2.4, -0.25; text = "Each Subpixel Piles Up at Its Own Rate, a Quarter of the Binned Pixel's.\nUHR Mode Reads Each Counter on Its Own (Li et al. 2025: Digital Binning).", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.limits!(axb, -0.3, 5.1, -0.95, 2.75)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.38))
    locator!(fig[0, :], :asic; note = "Step 5 at Clinical Rates:\nShaped Pulses, Comparators,\nFour Counters Added")
    fig
end

# ╔═╡ 7baa1504-7942-48ec-93f2-a03d48d3430f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 620))
    for (k, kV) in enumerate((120, 140))
        ax = Mke.Axis(
            fig[1, k];
            title = "($(('a', 'b')[k])) $(kV) kV, Air, Standard Mode",
            xlabel = "Tube Current (mA)",
            ylabel = k == 1 ? "Counts Lost to Pile-Up" : "",
            limits = ((0, 1050), (0, 0.85)),
        )
        mA = collect(range(1.0, li2025.max_mA[kV]; length = 200))
        lo = [li_loss(kV, I; τ_ns = li2025.tau_ci_ns[1]) for I in mA]
        hi = [li_loss(kV, I; τ_ns = li2025.tau_ci_ns[2]) for I in mA]
        Mke.band!(ax, mA, lo, hi; color = (PALETTE.reference, 0.2), label = "Li et al. 2025, Measured (95 % Band)")
        Mke.lines!(ax, mA, [li_loss(kV, I) for I in mA]; color = PALETTE.reference, linewidth = 2.5, label = "Li et al. 2025, Their Fit")
        r = pileup_fit.curves[kV]
        Mke.scatterlines!(ax, r.mA, r.loss; color = PALETTE.photon, markersize = 12, linewidth = 2.5, label = "Shaped Pulses, Fitted Here")
        k == 1 && Mke.axislegend(ax; position = :lt)
    end
    fig
end

# ╔═╡ ccc7fe92-998c-48d6-8ff5-bf063506827e
let
    within(kV) = all(
        li_loss(kV, I; τ_ns = li2025.tau_ci_ns[1]) - 0.01 <= L <= li_loss(kV, I; τ_ns = li2025.tau_ci_ns[2]) + 0.01
            for (I, L) in zip(pileup_fit.curves[kV].mA, pileup_fit.curves[kV].loss)
    )
    m = pileup_fit.mixes[140]
    Markdown.parse(
        """
        Fitted: pulses **$(pileup_fit.fwhm_ns) ns** wide at half height, a dead time of **$(pileup_fit.dead_ns) ns** after each
        count, and a **$(pileup_fit.retrigger ? "retriggering" : "plain")** comparator (squared misfit $(round(pileup_fit.best_retrigger; sigdigits = 2))
        retriggering, $(round(pileup_fit.best_plain; sigdigits = 2)) plain: a plain comparator piles up like a paralyzable counter, which
        Li's air scans rule out). With them a subpixel loses counts as Li measured: within their 95 % band at
        every tube current at 120 kV ($(within(120) ? "yes" : "no")) and at 140 kV ($(within(140) ? "yes" : "no")). At 140 kV a subpixel makes
        $(round(1 / m.crossing[1]; digits = 2)) pulses for every count above 20 keV, so the dead time measured per count is longer than
        the shaper alone would suggest; Siemens co-authors (Taguchi et al. 2022) assumed 23 ns per subpixel, the same scale.
        """
    )
end

# ╔═╡ e9d0ef22-c3ce-4023-a86d-963a234c51d6
# What pile-up does to each energy bin at 140 kV in air, with the fitted shaper and dead time
spectral_pileup = let
    thresholds_4 = [20.0, 35.0, 55.0, 70.0]
    m = pulse_mix(li_beams[140], thresholds_4)
    n_pulses = 30_000
    bins(c) = [c[1] - c[2], c[2] - c[3], c[3] - c[4], c[4]]
    count_at(rate) = MC.pulse_counts(BS.Random.Xoshiro(13), MC.PulseShaper(; fwhm_ns = pileup_fit.fwhm_ns, dead_ns = pileup_fit.dead_ns, retrigger = pileup_fit.retrigger), rate, n_pulses, m.values, m.cdf, thresholds_4; noise = m.noise)
    alone = bins(count_at(1.0e-7))
    mA = [25.0, 50.0, 100.0, 200.0, 350.0, 500.0, 700.0, 850.0]
    ratios = [bins(count_at(li2025.beta[140] * I / 4 / m.crossing[1] * 1.0e-9)) ./ alone for I in mA]
    (; mA, ratios, labels = ["20–35", "35–55", "55–70", "> 70"])
end

# ╔═╡ 2fe8db09-89e3-43fb-99a9-2470d61844eb
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 900))

    # what a merge does: two pulses become one, higher up
    axi = canvas(fig[1, 1]; title = "Two Pulses Merge: One Count Lost, and the Survivor Climbs a Bin", titlesize = 19)
    τ = pileup_fit.fwhm_ns / MC.shaper_fwhm(1)
    t = collect(range(0, 60; length = 600))
    single(t0, A) = [A * MC.shaper_pulse(x - t0, τ, 1) for x in t]
    a, b = single(10, 40), single(14, 30)
    for (level, label) in ((20, "20"), (35, "35"), (55, "55"), (70, "70 keV"))
        Mke.hlines!(axi, level; color = (MATERIALS.beam, 0.6), linestyle = :dash, linewidth = 1.2)
        Mke.text!(axi, 61, level; text = label, align = (:left, :center), fontsize = 13, color = Mke.RGBf(0.8, 0.5, 0.05))
    end
    Mke.lines!(axi, t, a; color = (PALETTE.photon, 0.5), linewidth = 2)
    Mke.lines!(axi, t, b; color = (PALETTE.photon, 0.5), linewidth = 2)
    Mke.lines!(axi, t, a .+ b; color = MATERIALS.charge, linewidth = 3.5)
    Mke.text!(axi, 30, 64; text = "40 + 30 keV, 4 ns Apart:\nCounted Once, Above 55 keV", align = (:left, :center), fontsize = 14, color = MATERIALS.charge)
    Mke.limits!(axi, -2, 72, -4, 80)

    ax = Mke.Axis(
        fig[1, 2];
        title = "Recorded ÷ True Counts, per Bin, 140 kV in Air",
        xlabel = "Tube Current (mA)",
        ylabel = "Recorded ÷ True",
        limits = ((0, 880), (0, nothing)),
    )
    Mke.hlines!(ax, 1; color = PALETTE.ink, linewidth = 1.2)
    colors = (PALETTE.photon, PALETTE.kept, PALETTE.xray, PALETTE.electron)
    for b in 1:4
        Mke.scatterlines!(ax, spectral_pileup.mA, [r[b] for r in spectral_pileup.ratios]; color = colors[b], markersize = 10, linewidth = 2.5, label = "$(spectral_pileup.labels[b]) keV")
    end
    Mke.axislegend(ax; position = :lb)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.42))
    fig
end

# ╔═╡ a1549f34-7dfe-431b-9d68-43c201ed3f6c
# Bhattarai et al. 2023 (Med Phys 50:6693): a 20 cm ACR phantom on an Alpha at 120 kV, 40–1000 mA,
# with Siemens' pile-up correction on: the noise at the centre rose above the 1/√mA expectation by
# 0.9 % at 120 mA and 11.1 % at 1000 mA, monotonically. The fitted model's noise excess for the
# central ray, behind 20 cm of water, and in air, taking a counter's noise to grow as 1/√(1 − loss)
pileup_context = let
    water = MC.DetectorMaterial(:Water)
    e, w = li_beams[120]
    through = w .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* 200.0)
    transmission = sum(through) / sum(w)
    mixes = (air = pileup_fit.mixes[120], water = pulse_mix((e, through), [20.0]))
    mA = [40.0, 80.0, 120.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    n_pulses = 15_000
    function excess(m, scale)
        alone = MC.pulse_counts(BS.Random.Xoshiro(17), MC.PulseShaper(; fwhm_ns = pileup_fit.fwhm_ns, dead_ns = pileup_fit.dead_ns, retrigger = pileup_fit.retrigger), 1.0e-7, n_pulses, m.values, m.cdf, [20.0]; noise = m.noise)[1]
        return map(mA) do I
            rate = li2025.beta[120] * I * scale / 4 / m.crossing[1] * 1.0e-9
            counted = MC.pulse_counts(BS.Random.Xoshiro(17), MC.PulseShaper(; fwhm_ns = pileup_fit.fwhm_ns, dead_ns = pileup_fit.dead_ns, retrigger = pileup_fit.retrigger), rate, n_pulses, m.values, m.cdf, [20.0]; noise = m.noise)[1]
            1 / sqrt(counted / alone) - 1
        end
    end
    (
        mA,
        transmission,
        air = excess(mixes.air, 1.0),
        water = excess(mixes.water, transmission),
        measured = (mA = [120.0, 1000.0], excess = [0.009, 0.111]),
    )
end

# ╔═╡ 905e2665-4c7e-4686-baed-5ec32a2cf16d
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 820))

    # the setup: the 20 cm phantom, the central ray, the subpixel behind it
    axi = canvas(fig[1, 1]; title = "Behind 20 cm of Water the Central Subpixel Sees $(round(100 * pileup_context.transmission; digits = 1)) % of the Air Rate", titlesize = 19)
    θ = range(0, 2π; length = 100)
    Mke.poly!(axi, Mke.Point2f.(3.0 .+ 1.6 .* cos.(θ), 1.6 .* sin.(θ)); color = (MATERIALS.water, 0.7), strokecolor = (PALETTE.photon, 0.5))
    Mke.text!(axi, 3.0, 0.0; text = "20 cm\nWater", align = (:center, :center), fontsize = 15, color = PALETTE.ink)
    Mke.scatter!(axi, [Mke.Point2f(-0.6, 0)]; marker = :rect, markersize = 20, color = PALETTE.ink)
    Mke.text!(axi, -0.6, -0.45; text = "Tube, 120 kV", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.lines!(axi, [-0.5, 6.8], [0.0, 0.0]; color = (MATERIALS.beam, 0.9), linewidth = 3)
    Mke.lines!(axi, [-0.5, 6.8], [0.0, 2.6]; color = (MATERIALS.beam, 0.5), linewidth = 2)
    rect!(axi, 6.8, -0.15, 0.25, 0.3, MATERIALS.cdte; strokecolor = PALETTE.slab_edge)
    Mke.text!(axi, 7.2, 0.0; text = "Central Subpixel:\nFew Photons, Little Pile-Up", align = (:left, :center), fontsize = 14, color = PALETTE.ink)
    Mke.text!(axi, 7.2, 2.6; text = "Edge Rays: Little Water,\nHigh Rates", align = (:left, :center), fontsize = 14, color = PALETTE.muted)
    Mke.limits!(axi, -2.2, 13.2, -2.0, 3.2)

    ax = Mke.Axis(
        fig[2, 1];
        title = "Noise Above the 1/√mA Expectation, 120 kV",
        xlabel = "Tube Current (mA)",
        ylabel = "Excess Noise (%)",
        limits = ((0, 1050), (-0.5, nothing)),
    )
    Mke.lines!(ax, pileup_context.mA, 100 .* pileup_context.air; color = (PALETTE.photon, 0.6), linewidth = 2.5, linestyle = :dash, label = "Model, Central Ray in Air")
    Mke.scatterlines!(ax, pileup_context.mA, 100 .* pileup_context.water; color = PALETTE.photon, markersize = 10, linewidth = 2.5, label = "Model, Central Ray Behind 20 cm Water")
    Mke.scatterlines!(ax, pileup_context.measured.mA, 100 .* pileup_context.measured.excess; color = PALETTE.reference, marker = :diamond, markersize = 16, linewidth = 1.5, linestyle = :dot, label = "Bhattarai et al. 2023, Measured (Correction On)")
    Mke.axislegend(ax; position = :lt)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.36))
    fig
end

# ╔═╡ 18d124be-b21d-4496-b644-c00fe5abae68
let
    k = findfirst(==(1000.0), pileup_context.mA)
    Markdown.parse(
        """
        **Context, not a check.** At 1000 mA the model's pile-up raises the noise of the central ray behind 20 cm of water by
        **$(round(100 * pileup_context.water[k]; digits = 1)) %**; Bhattarai et al. measured **11.1 %** at the centre of their 20 cm phantom. The two are not
        the same quantity: their scans ran with Siemens' pile-up correction on, their image noise at the centre also holds rays
        that crossed less of the phantom, and they name pile-up, the signal-dependent filtration and nonlinearities of the
        reconstruction as possible causes together. The model accounts for a part of what they saw; it is not a pass/fail test
        of the pile-up model, which Li et al.'s air scans, with the correction off, are.
        """
    )
end

# ╔═╡ e9a4425a-7892-48c8-8065-9733bc99b607
# What BasisSimulator needs to pile up counts itself, per mode (saved with the response table)
pileup_inputs = Dict{String, Any}(
    "model" => "pulses from the transported charge (every dexel that records charge from a photon makes one); CR-RC shaping; Gaussian noise on each pulse's height (Zhan's 1.9 keV and Fano 0.1); one comparator per threshold (see comparator)",
    "pulse_floor_keV" => pulse_heights.floor_keV,
    "pulse_height_edges_keV" => pulse_heights.edges,
    "pulse_heights_per_photon" => pulse_heights.histograms,
    "pulses_per_photon" => pulse_heights.per_photon,
    "energies_keV" => pulse_heights.energies,
    "shaper" => "CR-RC, order 1",
    "shaper_fwhm_ns" => pileup_fit.fwhm_ns,
    "dead_time_ns" => pileup_fit.dead_ns,
    "comparator" => pileup_fit.retrigger ? "retriggering: counts every peak above its threshold, at least dead_time_ns after its last count" : "plain: counts every upward crossing of its threshold, at least dead_time_ns after its last count",
    "fitted_to" => "Li et al. 2025, Med Phys 52:e70045: the loss against tube current in air at 120 and 140 kV, standard mode, pile-up correction off (beta and the composite dead-time model of their Table 2 and Figs 5-7)",
    "modes" => Dict(
        "native" => "a subpixel's pulses at its own rate, the inputs above as they are",
        "binned_2x2" => "digital binning: each of the 4 subpixels piles up on its own, at a quarter of the binned pixel's rate, with the same inputs; the binned counts are the sum of the 4 counters",
    ),
    "assumed" => "the CR-RC shape, noise added to each pulse's height, the comparator logic, chosen by the fit (no pile-up trigger): the Alpha's ASIC is not published",
)

# ╔═╡ 88117c52-14b8-4302-a0e5-2231d12504e2
md"""
## 11. A Clinical Beam, for Real

Every count so far came from photons of **one energy**. A scan is a spectrum, hardened by the
bowtie and by the patient, and BasisSimulator never follows it photon by photon: it weights the
single-energy response by the spectrum. This section checks that shortcut by doing it the long
way:

- the Alpha's **140 kVp beam** (BasisSimulator's spectrum: the 10° anode, 3 mm Al + 0.9 mm Ti,
  through the bowtie's centre), **bare and after 20 and 35 cm of water** (the patient's primary
  photons; the patient's scatter is notebook 10's);
- each one **drawn photon by photon from its spectrum** onto one native dexel, as in sections 1–4
  (800 000 photons, in 16 independent chunks), with the same charge cloud, noise and thresholds;
- against the single-energy counts of section 4 **weighted by the same spectrum**, in each of the
  four bins (20–35, 35–55, 55–70 and above 70 keV) and in all of them together.

Charge sharing makes the counts nonlinear in the photon (one photon can count twice), but each
photon still counts independently of the others below pile-up rates, so the weighted sum must
hold exactly; the check is whether it does.
"""

# ╔═╡ 888c0650-ebc4-4bfa-8880-335e82f146c0
# The Alpha's 140 kVp beam on the central ray: BasisSimulator's spectrum (10° anode) through the
# bowtie's centre, then through 0, 20 and 35 cm of water (the primary photons)
clinical_beams = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium, water = MC.DetectorMaterial(:Al), MC.DetectorMaterial(:Water)
    e, w = spectra[10]
    through_bowtie = w .* exp.(-[MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e] .* bowtie_al_mm)
    [
        (
                label = L == 0 ? "No Patient" : "$(round(Int, L / 10)) cm of Water",
                water_mm = L,
                spectrum = MC.Spectrum(
                    collect(Float64, e),
                    through_bowtie .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* L),
                ),
            )
            for L in (0.0, 200.0, 350.0)
    ]
end

# ╔═╡ 5eb2d62c-5e6b-48f3-b451-cb69665bebc1
# What changes in the counts: per photon of the bare 140 kV beam, in each of the Alpha's four bins
let
    b = first(clinical_beams)
    e, w = b.spectrum.energies_keV, b.spectrum.weights
    at = [findfirst(==(t), thresholds) for t in air_readout.thresholds_keV]
    E = collect(energies)
    weighted(cs) = [sum(w .* [MC.interpolate_linear(E, [c.total[t] for c in cs], x) for x in e]) / sum(w) for t in at]
    old = bins_from(weighted(counts))
    new = bins_from(weighted(transport_counts))
    labels = ["20–35 keV", "35–55 keV", "55–70 keV", "> 70 keV"]
    rows = ["| $(labels[i]) | $(round(old[i]; digits = 4)) | $(round(new[i]; digits = 4)) | $(round(100 * (new[i] / old[i] - 1); digits = 1)) % |" for i in 1:4]
    Markdown.parse(
        """
        | bin | fixed 30 µm cloud, 1.5 keV noise | transported charge, 1.9 keV noise | change |
        |---|---|---|---|
        $(join(rows, "\n"))
        | all above 20 keV | $(round(sum(old); digits = 4)) | $(round(sum(new); digits = 4)) | $(round(100 * (sum(new) / sum(old) - 1); digits = 1)) % |

        Counts per photon of $(b.label)'s 140 kV beam landing on one native dexel. With narrower clouds fewer photons are
        shared with a neighbour: fewer double counts at low energy, more photons counted at their full energy.
        """
    )
end

# ╔═╡ 788bd9cd-7e98-4bba-a04a-97b4f96f1e2e
# The single Gaussian width that counts most like the transported charge, above all nine
# thresholds (least squares): at each energy, and over each clinical beam
zhan_check = let
    E = collect(transport_runs.scan_energies)
    σs = collect(transport_runs.scan_sigmas)
    target = [transport_counts[findfirst(==(e), energies)].total for e in E]
    curves = [transport_runs.scan[e] for e in E]
    function best(curve, value)
        r = [sum(abs2, c .- value) for c in curve]
        k = argmin(r)
        (k == 1 || k == length(r)) && return σs[k]
        a, b, c = r[k - 1], r[k], r[k + 1]
        return σs[k] + (a - c) / (2 * (a - 2b + c)) * (σs[k + 1] - σs[k])
    end
    per_energy = [best(c, v) for (c, v) in zip(curves, target)]
    beams = map(clinical_beams) do b
        # each beam's photons, gathered into the 5 keV steps of the scan
        φ = [sum(b.spectrum.weights[(e - 2.5) .< b.spectrum.energies_keV .<= (e + 2.5)]) for e in E]
        curve = [sum(φ .* [c[s] for c in curves]) / sum(φ) for s in eachindex(σs)]
        (label = b.label, sigma_um = best(curve, sum(φ .* target) / sum(φ)))
    end
    (; energies_keV = E, per_energy, beams, sigmas = σs)
end

# ╔═╡ 134ebfbd-ecdd-432f-a76f-94577b100925
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 860))

    # Top: Zhan's measurement, and what was fitted
    ax = canvas(fig[1, 1:2]; title = "Zhan et al. 2026: Clinical NAEOTOM Alpha Scans of PMMA, Fitted with One Gaussian Cloud", titlesize = 19)
    Mke.scatter!(ax, [Mke.Point2f(0, 0)]; marker = :rect, markersize = 22, color = PALETTE.ink)
    Mke.text!(ax, 0, -0.55; text = "Tube, 70–140 kV", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.poly!(ax, Mke.Point2f[(0.15, 0), (9.5, 1.0), (9.5, -1.0)]; color = (MATERIALS.beam, 0.18))
    for (i, thick) in enumerate(0.6:0.6:3.0)
        rect!(ax, 3.0 + 0.45i, -0.6, 0.35, 1.2, Mke.RGBf(0.85, 0.9, 0.95); strokecolor = (:black, 0.3))
    end
    Mke.text!(ax, 4.3, 0.85; text = "PMMA, 0 to 13 Inches", align = (:center, :bottom), fontsize = 15, color = PALETTE.ink)
    rect!(ax, 9.5, -1.1, 0.25, 2.2, MATERIALS.cdte; strokecolor = PALETTE.slab_edge)
    Mke.text!(ax, 10.0, 0.45; text = "Counts Above 20 keV ÷ Counts Above a Higher Threshold", align = (:left, :center), fontsize = 15, color = PALETTE.kept)
    Mke.text!(ax, 10.0, -0.25; text = "Fitted: One Gaussian Cloud σ = r_e (16.4 µm),\nElectronic Noise σ_e (1.9 keV)", align = (:left, :center), fontsize = 15, color = PALETTE.ink)
    Mke.limits!(ax, -1.8, 18.5, -1.4, 1.4)

    # Bottom left: the single Gaussian that counts like the transported charge, at each energy
    axa = Mke.Axis(
        fig[2, 1];
        title = "(a) The Equivalent Single Gaussian, by Photon Energy",
        xlabel = "Photon Energy (keV)",
        ylabel = "Equivalent σ (µm)",
        limits = ((10, 145), (5, 32)),
    )
    Mke.hspan!(axa, zhan.sigma_range_um...; color = (PALETTE.kept, 0.15))
    Mke.hlines!(axa, zhan.sigma_um; color = PALETTE.kept, linewidth = 2)
    note!(axa, 140, zhan.sigma_um, "Zhan 2026: $(zhan.sigma_um) µm ($(zhan.sigma_range_um[1])–$(zhan.sigma_range_um[2]))"; align = (:right, :bottom), offset = (0, 4), color = PALETTE.kept, fontsize = 14)
    Mke.hlines!(axa, 30.0; color = PALETTE.reference, linestyle = :dash, linewidth = 2)
    note!(axa, 140, 30.0, "BasisSimulator v0.15.1: 30 µm"; align = (:right, :bottom), color = PALETTE.reference, fontsize = 14)
    Mke.scatterlines!(axa, zhan_check.energies_keV, zhan_check.per_energy; color = PALETTE.photon, markersize = 10, linewidth = 2.5, label = "This Notebook's Transported Charge")
    Mke.axislegend(axa; position = :lb)

    # Bottom right: over each clinical beam
    axb = Mke.Axis(
        fig[2, 2];
        title = "(b) Over the Alpha's 140 kV Beams",
        ylabel = "Equivalent σ (µm)",
        xticks = (1:length(zhan_check.beams), [b.label for b in zhan_check.beams]),
        limits = ((0.4, length(zhan_check.beams) + 0.6), (5, 32)),
    )
    Mke.hspan!(axb, zhan.sigma_range_um...; color = (PALETTE.kept, 0.15))
    Mke.hlines!(axb, zhan.sigma_um; color = PALETTE.kept, linewidth = 2)
    Mke.hlines!(axb, 30.0; color = PALETTE.reference, linestyle = :dash, linewidth = 2)
    Mke.scatter!(axb, 1:length(zhan_check.beams), [b.sigma_um for b in zhan_check.beams]; color = PALETTE.photon, markersize = 18)
    for (k, b) in enumerate(zhan_check.beams)
        note!(axb, k, b.sigma_um, "$(round(b.sigma_um; digits = 1)) µm"; align = (:center, :bottom), offset = (0, 10), fontsize = 15)
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.3))
    locator!(fig[0, :], (:cdte, :electrodes); note = "Steps 3–4 · Checked Against\nthe Clinical Alpha")
    fig
end

# ╔═╡ 25f703a9-2bc9-4781-9a28-dd84484cb1d6
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 820))
    ax = canvas(fig[1, 1]; title = "The Beam, Photon by Photon", titlesize = 20)
    amber = MATERIALS.beam
    focal = Mke.Point2f(0, 0)
    Mke.scatter!(ax, [focal]; marker = :rect, markersize = 22, color = PALETTE.ink)
    Mke.text!(ax, focal + Mke.Point2f(0, 0.45); text = "140 kVp Tube", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    R = 10.0
    φ = range(-deg2rad(24.1), deg2rad(24.1); length = 60)
    arc = [Mke.Point2f(R * sin(t), -R * cos(t)) for t in φ]
    Mke.poly!(ax, vcat([focal], arc); color = (amber, 0.15))
    Mke.poly!(ax, Mke.Point2f[(-1.2, -1.6), (1.2, -1.6), (0.15, -2.0), (1.2, -2.4), (-1.2, -2.4), (-0.15, -2.0)]; color = Mke.RGBf(0.75, 0.77, 0.8))
    Mke.text!(ax, 1.35, -2.0; text = "Bowtie", align = (:left, :center), fontsize = 15, color = PALETTE.ink)
    θ = range(0, 2π; length = 100)
    Mke.poly!(ax, Mke.Point2f.(2.4 .* cos.(θ), -5.45 .+ 1.6 .* sin.(θ)); color = (MATERIALS.water, 0.7), strokecolor = (PALETTE.photon, 0.5))
    Mke.text!(ax, 0, -5.45; text = "0, 20 or 35 cm\nof Water", align = (:center, :center), fontsize = 15, color = PALETTE.ink)
    inner = [Mke.Point2f((R + 0.05) * sin(t), -(R + 0.05) * cos(t)) for t in φ]
    outer = [Mke.Point2f((R + 0.45) * sin(t), -(R + 0.45) * cos(t)) for t in φ]
    Mke.poly!(ax, vcat(inner, reverse(outer)); color = MATERIALS.cdte, strokecolor = PALETTE.slab_edge)
    Mke.text!(ax, 0, -R - 0.7; text = "One Native Dexel of CdTe, as in Sections 1–4", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -6.5, 6.5, -R - 1.8, 1.4)

    continuum(e) = !(57 <= e <= 70)
    shapes = [(b.label, b.spectrum.energies_keV, b.spectrum.weights ./ (sum(b.spectrum.weights) * 0.5)) for b in clinical_beams]
    top = 1.35 * maximum(maximum(w[continuum.(e)]) for (_, e, w) in shapes)
    axs = Mke.Axis(
        fig[1, 2];
        title = "The Beam Hardens in the Patient",
        titlesize = 20,
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons per keV",
        limits = ((10, 145), (0, top)),
        xticks = 20:20:140,
    )
    for t in air_readout.thresholds_keV
        Mke.vlines!(axs, t; color = (PALETTE.muted, 0.5), linestyle = :dash, linewidth = 1.2)
    end
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron)
    for (k, (label, e, w)) in enumerate(shapes)
        Mke.lines!(axs, e, w; color = colors[k], linewidth = 3, label = "$(label): Mean $(round(sum(w .* e) / sum(w); digits = 1)) keV")
    end
    note!(axs, 63.5, 0.97top, "Tungsten's K Lines\n(Off Scale)"; align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    note!(axs, 21, 0.03top, "Dashed: the Four Thresholds"; align = (:left, :bottom), fontsize = 14, color = PALETTE.muted)
    Mke.axislegend(axs; position = :rt)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    locator!(fig[0, :], :whole; note = "Everything at Once:\nTube, Patient, Every Step")
    fig
end

# ╔═╡ 2956ae91-7028-45a0-9466-12370ae5a928
# Each beam photon by photon onto one native dexel (16 chunks of 50 000), each chunk's charge
# shared and counted where it ran, with section 4's noise
clinical_counts = let
    responses_by_chunk = MC.simulate_sweep(
        [(crystal, b.spectrum, 600 + chunk) for b in clinical_beams for chunk in 1:16];
        n = 50_000,
        beam,
        readout,
    )
    counted = [
        MC.threshold_counts(r, air_readout.thresholds_keV; electronic_noise_keV = current_table["sigma_elec_keV"]).total
            for r in responses_by_chunk
    ]
    [counted[(16(k - 1) + 1):(16k)] for k in eachindex(clinical_beams)]
end

# ╔═╡ b93ed54c-9e8b-45ee-b09c-7a06a1f1aed2
# The long way against the shortcut: counts per photon in each bin (and above 20 keV), the direct
# value with its noise (the 16 chunks' spread), and the single-energy counts weighted by the spectrum
clinical_check = let
    E = collect(energies)
    thresholds_at = [findfirst(==(t), thresholds) for t in air_readout.thresholds_keV]
    per_energy = [c.total[thresholds_at] for c in counts]
    noise(values) = sqrt(sum(abs2, values .- mean(values)) / (length(values) - 1) / length(values))
    labels = ["20–35 keV", "35–55 keV", "55–70 keV", "> 70 keV", "All Above 20 keV"]
    with_total(above) = vcat(bins_from(above), above[1])
    map(enumerate(clinical_beams)) do (k, b)
        e, w = b.spectrum.energies_keV, b.spectrum.weights
        weighted = [
            sum(w .* [MC.interpolate_linear(E, [p[t] for p in per_energy], x) for x in e]) / sum(w)
                for t in 1:4
        ]
        chunks = [with_total(c) for c in clinical_counts[k]]
        (
            label = b.label,
            bins = labels,
            table = with_total(weighted),
            direct = with_total(sum(clinical_counts[k]) ./ length(clinical_counts[k])),
            noise = [noise([c[i] for c in chunks]) for i in eachindex(labels)],
        )
    end
end

# ╔═╡ 1029e073-9f7e-4bc3-a1b0-e5ec3a79b03f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 560))
    for (k, c) in enumerate(clinical_check)
        x = 1:length(c.bins)
        ax = Mke.Axis(
            fig[1, k];
            title = c.label,
            titlesize = 20,
            ylabel = k == 1 ? "Counts per Photon" : "",
            xticks = (x, replace.(c.bins, " keV" => "")),
            xticklabelrotation = π / 6,
            limits = ((0.4, length(x) + 0.6), (0, 1.15 * maximum(c.direct))),
        )
        Mke.scatter!(ax, x .- 0.1, c.table; color = PALETTE.reference, marker = :diamond, markersize = 16, label = "Single Energies, Weighted by the Spectrum")
        Mke.errorbars!(ax, x .+ 0.1, c.direct, 3 .* c.noise; color = PALETTE.kept, whiskerwidth = 10, linewidth = 2)
        Mke.scatter!(ax, x .+ 0.1, c.direct; color = PALETTE.kept, markersize = 14, label = "Geant4, the Spectrum Itself (± 3σ)")
        k == 1 && Mke.axislegend(ax; position = :lt)
    end
    fig
end

# ╔═╡ b8846f41-7a5f-41bd-90b9-0881b4ecc0e4
let
    rows = String[]
    for c in clinical_check, i in eachindex(c.bins)
        difference = c.direct[i] - c.table[i]
        allowed = 3c.noise[i] + 0.003
        pass = abs(difference) <= allowed
        push!(rows, "| $(pass ? "✓" : "✗ **fail**") | $(c.label), $(c.bins[i]) | $(round(c.table[i]; digits = 4)) | $(round(c.direct[i]; digits = 4)) ± $(round(c.noise[i]; digits = 4)) | $(round(difference; digits = 4)) | ≤ $(round(allowed; digits = 4)) |")
    end
    passed = count(r -> startswith(r, "| ✓"), rows)
    Markdown.parse(
        """
        | | beam, counts per photon | single energies, weighted | Geant4, the spectrum itself | difference | allowed |
        |---|---|---|---|---|---|
        $(join(rows, "\n"))

        **$(passed) of $(length(rows)) agree.** Allowed: three times the direct simulation's own noise, plus 0.003 for
        the single-energy table's noise and its 1-keV grid. Where they agree, the single-energy response, weighted by any
        spectrum, is the clinical beam: a response table on a 1-keV grid is all a scan needs (below pile-up rates,
        section 10).
        """
    )
end

# ╔═╡ 34ec1ace-4eae-40b1-b8b4-8b9c5a917bd4
md"""
## 12. No Two Dexels Alike

Every dexel has its own comparators and its own amplifier, and no calibration makes them exactly
equal. Two differences matter, and both are **assumed** here, because no clinical scanner
publishes them: each dexel's thresholds sit slightly off their nominal energies (**threshold
dispersion**, 0.2–0.5 keV rms after trimming: about 0.18 keV for Medipix3RX, Ballabriga et al.
2013; 0.34 keV for a CdTe chip, Liu et al. 2025), and its gain differs a little (**gain spread**,
1–3 % rms, Maj et al. 2014).

What they do, with the pulses of section 10 and a 120 kV beam:

- **(a) Flat field:** how far each dexel's counts per energy bin stray from the nominal dexel's.
- **(b) Noise:** an unchanging pattern across dexels, which outgrows the photons' own noise as the
  counts rise.
- **(c) After correction:** the scanner corrects it with an air scan, and so does this notebook,
  from a *simulated* air scan of a finite number of counts, never from the dexels' true values. What
  is left, the calibration's own noise and the patient's harder spectrum, which the air scan did
  not see, is what draws rings, estimated here with a filtered backprojection.

Zhan's 1.9 keV fitted noise, averaged over the clinical Alpha's dexels, already holds any threshold
dispersion, which bounds it: with 1.5 keV of electronic noise (Stierstorfer & Hupfer 2024), it
cannot exceed √(1.9² − 1.5²) ≈ 1.2 keV.
"""

# ╔═╡ 1a4e2fa3-068c-4d0a-9ed6-146919ba2761
# Assumed, and labelled so: dexel-to-dexel differences after the vendor's calibration (rms)
variation = (
    threshold_keV = [0.2, 0.35, 0.5],  # Medipix3RX ≈ 0.18 keV (Ballabriga 2013); CdTe 0.34 keV (Liu 2025)
    gain = [0.01, 0.02, 0.03],  # 3 % (Maj et al. 2014)
    electronic_noise_keV = 1.5,  # realistic CdTe noise (Stierstorfer & Hupfer 2024), for the bound from Zhan's 1.9 keV
)

# ╔═╡ 0a49a6bc-b410-405c-a612-91ab87436366
# Each dexel's counts per energy bin with its own threshold offset and gain, in an air flat field and
# behind 20 cm of water; a flat-field correction from a simulated air scan of a finite number of
# counts per dexel, applied to the patient; the residual the correction leaves
variation_study = let
    thresholds_4 = [20.0, 35.0, 55.0, 70.0]
    water = MC.DetectorMaterial(:Water)
    e, w = li_beams[120]
    air = pulse_mix(li_beams[120], thresholds_4)
    patient = pulse_mix((e, w .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* 200.0)), thresholds_4)
    function bins(m, δ, g)  # counts per pulse in the four bins and above 20 keV
        c = MC.counts_above(
            m.values,
            m.p,
            thresholds_4;
            variation = MC.DexelVariation(; threshold_offset_keV = δ, gain = g),
            electronic_noise_keV = zhan.noise_keV,
            fano = zhan.fano,
        )
        return [c[1] - c[2], c[2] - c[3], c[3] - c[4], c[4], c[1]]
    end
    rms(x) = sqrt(sum(abs2, x .- sum(x) / length(x)) / length(x))
    n_dexels = 1500
    draws = BS.Random.randn(BS.Random.Xoshiro(3), n_dexels, 2)
    function relative(m, σT, σg)  # each dexel's counts per bin over the nominal dexel's
        nominal = bins(m, 0.0, 1.0)
        return [bins(m, σT * draws[d, 1], 1 + σg * draws[d, 2]) ./ nominal for d in 1:n_dexels]
    end
    spread(rel) = [rms(getindex.(rel, b)) for b in 1:5]
    thresholds_only = [spread(relative(air, σ, 0.0)) for σ in variation.threshold_keV]
    gain_only = [spread(relative(air, 0.0, σ)) for σ in variation.gain]
    σT, σg = variation.threshold_keV[2], variation.gain[2]
    rel_air, rel_patient = relative(air, σT, σg), relative(patient, σT, σg)
    share = bins(air, 0.0, 1.0) ./ bins(air, 0.0, 1.0)[5]
    n_cal = [1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8]
    rng = BS.Random.Xoshiro(5)
    normalised(x) = x ./ (sum(x) / length(x))
    residual = map(n_cal) do N
        map(1:5) do b
            measured_gain = [rel_air[d][b] * (1 + randn(rng) / sqrt(N * share[b])) for d in 1:n_dexels]
            rms(normalised([rel_patient[d][b] / measured_gain[d] for d in 1:n_dexels]))
        end
    end
    mismatch = [rms(normalised([rel_patient[d][b] / rel_air[d][b] for d in 1:n_dexels])) for b in 1:5]
    (;
        labels = ["20–35", "35–55", "55–70", "> 70", "All Above 20"],
        thresholds_only,
        gain_only,
        uncorrected = spread(rel_air),
        n_cal,
        residual,
        mismatch,
        σT,
        σg,
        mean_keV = sum(e .* w .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* 200.0)) / sum(w .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* 200.0)),
    )
end

# ╔═╡ 6b8a272a-a980-4d31-93ca-3a22e568337e
# The ring a filtered backprojection draws from one detector column whose line integral is off by
# δp in every view (parallel beam, Ram-Lak filter, a standard-mode column at the isocentre, 50 mm
# from the centre): its peak in HU per unit δp, with water's attenuation at the patient beam's mean energy
ring_per_unit = let
    Δ = 2 * 0.151
    n_det = 1024
    s = ((1:n_det) .- (n_det + 1) / 2) .* Δ
    j0 = argmin(abs.(s .- 50.0))
    filtered(n) = n == 0 ? 1 / (4Δ) : isodd(n) ? -1 / (π^2 * n^2 * Δ) : 0.0  # Δ × the Ram-Lak kernel
    interpolated(x) = let u = x / Δ + (n_det + 1) / 2, i = floor(Int, u)
        (1 - (u - i)) * filtered(i - j0) + (u - i) * filtered(i + 1 - j0)
    end
    n_views = 720
    angles = range(0, π; length = n_views + 1)[1:(end - 1)]
    radii = range(45.0, 55.0; length = 1001)
    image = [π / n_views * sum(interpolated(r * cos(θ)) for θ in angles) for r in radii]
    μ_water = MC.linear_attenuation(MC.DetectorMaterial(:Water), variation_study.mean_keV)
    1000 * maximum(abs, image) / μ_water
end

# ╔═╡ 66f58509-736f-48c3-8996-363d825bb34f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 860))

    # (a) eight dexels, their four thresholds each a little off (offsets drawn 6 × larger)
    ax = Mke.Axis(fig[1, 1]; title = "(a) Every Dexel's Thresholds Sit a Little Off", titlesize = 20)
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    offsets = BS.Random.randn(BS.Random.Xoshiro(8), 8, 4) .* 0.35 .* 6
    levels = [20.0, 35.0, 55.0, 70.0]
    for d in 1:8
        rect!(ax, (d - 1) * 1.2, 0, 1.0, 85, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1)
        for (k, T) in enumerate(levels)
            Mke.lines!(ax, [(d - 1) * 1.2 + 0.1, (d - 1) * 1.2 + 0.9], fill(T + offsets[d, k], 2); color = Mke.RGBf(0.8, 0.5, 0.05), linewidth = 3)
        end
    end
    for T in levels
        Mke.lines!(ax, [-0.2, 9.6], [T, T]; color = (PALETTE.muted, 0.6), linestyle = :dot, linewidth = 1.2)
        Mke.text!(ax, 9.7, T; text = "$(round(Int, T)) keV", align = (:left, :center), fontsize = 14, color = PALETTE.muted)
    end
    Mke.text!(ax, 4.6, -6; text = "Eight Dexels: Orange, Each One's Thresholds; Dotted, the Nominal Ones\n(Offsets Drawn Six Times Larger Than 0.35 keV rms)", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.limits!(ax, -0.6, 11.6, -20, 92)

    # (b) the correction a scanner makes, and what it leaves
    axb = canvas(fig[1, 2]; title = "(b) Calibrated on Air, Used on the Patient", titlesize = 20)
    boxes = (
        (0.0, "Air Scan", "N Counts per Dexel"),
        (3.2, "Each Dexel's Gain", "per Energy Bin"),
        (6.4, "Patient Scan", "Divided by the Gains"),
    )
    for (x, title, detail) in boxes
        rect!(axb, x, 0.0, 2.6, 1.4, Mke.RGBf(0.95, 0.96, 0.97); strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        Mke.text!(axb, x + 1.3, 0.92; text = title, align = (:center, :center), fontsize = 15, font = :bold, color = PALETTE.ink)
        Mke.text!(axb, x + 1.3, 0.45; text = detail, align = (:center, :center), fontsize = 13, color = PALETTE.muted)
    end
    arrow!(axb, (2.65, 0.7), (3.15, 0.7); color = PALETTE.ink, linewidth = 2, head = 11)
    arrow!(axb, (5.85, 0.7), (6.35, 0.7); color = PALETTE.ink, linewidth = 2, head = 11)
    θ = range(0, 2π; length = 200)
    Mke.poly!(axb, Mke.Point2f.(4.6 .+ 1.6 .* cos.(θ), -2.4 .+ 1.6 .* sin.(θ)); color = Mke.RGBf(0.35, 0.35, 0.38))
    for (r, a) in ((0.55, 0.9), (1.0, 0.6), (1.3, 0.75))
        Mke.lines!(axb, 4.6 .+ r .* cos.(θ), -2.4 .+ r .* sin.(θ); color = (:white, a), linewidth = 1.6)
    end
    arrow!(axb, (7.7, -0.05), (6.25, -1.3); color = PALETTE.ink, linewidth = 2, head = 11)
    Mke.text!(axb, 6.5, -2.4; text = "What Is Left Draws Rings:\nthe Air Scan's Own Noise, and a\nHarder Spectrum Than It Saw", align = (:left, :center), fontsize = 14, color = PALETTE.ink)
    Mke.limits!(axb, -0.3, 9.4, -4.3, 2.0)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.42))
    locator!(fig[0, :], :asic; note = "Step 5, Dexel by Dexel:\nThresholds and Gains\nThat Differ")
    fig
end

# ╔═╡ ec8ea99e-dadc-400a-9fd3-df2680140d93
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1750, 600))
    v = variation_study
    colors = (PALETTE.photon, PALETTE.kept, PALETTE.xray, PALETTE.electron, PALETTE.ink)
    axa = Mke.Axis(fig[1, 1]; title = "(a) Flat-Field Spread, Thresholds", xlabel = "Threshold Offsets, rms (keV)", ylabel = "Counts' Spread Across Dexels (%)")
    axg = Mke.Axis(fig[1, 2]; title = "(a) Flat-Field Spread, Gains", xlabel = "Gain Spread, rms (%)")
    for b in 1:5
        Mke.scatterlines!(axa, variation.threshold_keV, [100 * s[b] for s in v.thresholds_only]; color = colors[b], markersize = 10, linewidth = 2.5, label = "$(v.labels[b]) keV")
        Mke.scatterlines!(axg, 100 .* variation.gain, [100 * s[b] for s in v.gain_only]; color = colors[b], markersize = 10, linewidth = 2.5)
    end
    Mke.axislegend(axa; position = :lt)

    axb = Mke.Axis(
        fig[1, 3];
        title = "(b) Fixed Pattern ÷ Photon Noise",
        xlabel = "Counts per Dexel per Reading",
        ylabel = "Variance Ratio",
        xscale = log10,
        yscale = log10,
        limits = ((10, 1.0e6), (1.0e-4, 1.0e3)),
    )
    n = 10 .^ range(1, 6; length = 50)
    Mke.lines!(axb, n, n .* v.uncorrected[5]^2; color = PALETTE.electron, linewidth = 3, label = "Uncorrected")
    k = findfirst(==(1.0e6), v.n_cal)
    Mke.lines!(axb, n, n .* v.residual[k][5]^2; color = PALETTE.kept, linewidth = 3, label = "After the Air Calibration")
    Mke.hlines!(axb, 1; color = PALETTE.ink, linestyle = :dash, linewidth = 1.2)
    Mke.axislegend(axb; position = :lt)

    axc = Mke.Axis(
        fig[1, 4];
        title = "(c) Ring Left After Correction",
        xlabel = "Air-Scan Counts per Dexel",
        ylabel = "Ring Amplitude (HU)",
        xscale = log10,
        yscale = log10,
    )
    for b in (1, 4, 5)
        Mke.scatterlines!(axc, v.n_cal, [ring_per_unit * r[b] for r in v.residual]; color = colors[b], markersize = 10, linewidth = 2.5, label = "$(v.labels[b]) keV")
        Mke.hlines!(axc, ring_per_unit * v.mismatch[b]; color = (colors[b], 0.6), linestyle = :dash, linewidth = 1.5)
    end
    Mke.axislegend(axc; position = :rt)
    Mke.Label(fig[2, 1:4], "Dexels Drawn With $(v.σT) keV and $(round(Int, 100v.σg)) % rms (Panels b, c); Dashed in (c): the Floor a Perfect Air Scan Leaves, Because the Patient's Beam Is Harder; 120 kV, Behind 20 cm of Water"; fontsize = 15, color = PALETTE.muted)
    fig
end

# ╔═╡ 7d5f8a64-152a-43e0-b3a0-cb46f12f2ff5
let
    v = variation_study
    k = findfirst(==(1.0e6), v.n_cal)
    Markdown.parse(
        """
        With $(v.σT) keV threshold offsets and $(round(Int, 100v.σg)) % gain spread (rms), a dexel's counts above 20 keV stray by
        **$(round(100 * v.uncorrected[5]; digits = 2)) %** from the nominal dexel's in an air flat field, and by
        $(round(100 * v.uncorrected[1]; digits = 2)) % in the lowest bin: an unchanging pattern that outgrows the photons' own noise above
        about **$(round(Int, 1 / v.uncorrected[5]^2)) counts per dexel per reading**. An air calibration of 10⁶ counts per dexel leaves
        $(round(100 * v.residual[k][5]; digits = 3)) % above 20 keV, a ring of about **$(round(ring_per_unit * v.residual[k][5]; digits = 2)) HU**; a perfect air scan would still leave
        $(round(100 * v.mismatch[5]; digits = 3)) % ($(round(ring_per_unit * v.mismatch[5]; digits = 2)) HU), because behind 20 cm of water the beam is harder than the
        air scan's. The threshold offsets matter most at the bins' edges; the gain spread acts on every bin alike.
        These sizes are assumed, not measured on the Alpha; Zhan's fit bounds the threshold offsets below
        $(round(sqrt(zhan.noise_keV^2 - variation.electronic_noise_keV^2); digits = 1)) keV, well above the 0.2–0.5 keV studied.
        """
    )
end

# ╔═╡ 88baa453-2c9f-4e83-abef-b81e449f8453
md"""
## 13. Tables for Standard and UHR Modes

The Alpha reads its dexels two ways. **UHR mode** reads every native dexel; **standard mode**
adds each 2 × 2 group's counts *after* counting (digital binning, Li et al. 2025). Both come from
one physics, so they come from one set of tables here, in a plain, versioned layout that
BasisSimulator can read on its own:

- **Table A, native dexels.** For each of the six positions a dexel can hold in its grid cell
  (A–F, with mirrors: section 5), at each energy and threshold: the **mean** counts in a flat field,
  the **crosstalk** `psf` (what each neighbour's photons add to the dexel; it sums to the mean),
  and the **pair** moments (the counts of the dexel and of each neighbour from the same photon).
  A native dexel's counts are Poisson: one photon counts at most once in it.
- **Table B, 2 × 2 binned pixels.** Derived from A: each binned pixel's mean is its four
  members' means, including what spills in from photons landing in the next group, and its second
  moment adds the pairs **inside** the group. A photon shared by two members counts twice in the
  binned pixel, so its counts are **not** Poisson: their variance exceeds the mean.

Both are in counts per incident photon per native pitch area (one photon per twenty-fourth of a
grid cell), at the thresholds a researcher might want, with the pile-up inputs of section 10.
Energies run from 10 to 140 keV in 2 keV steps, those of the grid runs (section 5): the Alpha's beam
has no photons above 140 keV and almost none below 15 keV.
"""

# ╔═╡ 04b43622-f984-48b6-b488-430bc420835e
# The thresholds the tables hold (cumulative: counts above each)
mode_thresholds = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 52.0, 55.0, 56.0, 60.0, 65.0, 70.0, 72.0, 75.0, 82.0, 90.0]

# ╔═╡ 67171632-72ab-4d86-9dd1-f9188f86332d
# Each of a grid cell's 4 × 6 dexels by symmetry: its position type (A–F, section 5) and whether it
# is that position mirrored across the cell's middle column (x) or row (y)
cell_types = let
    column_class(k) = k in (0, 3) ? 0 : 1
    row_class(l) = l in (0, 5) ? 0 : l in (1, 4) ? 1 : 2
    label = Dict(p.target => p.label for p in positions)
    (
        type_map = [label[(column_class(k), row_class(l))] for k in 0:3, l in 0:5],
        mirror_x = [k >= 2 ? -1 : 1 for k in 0:3, l in 0:5],
        mirror_y = [l >= 3 ? -1 : 1 for k in 0:3, l in 0:5],
        index = Dict(p.label => i for (i, p) in enumerate(positions)),
    )
end

# ╔═╡ 92576991-a76a-5062-b4fc-e5d8604dd672
md"""
## Saving the Tables

The tables BasisSimulator would load, written in the layout of v0.15.1's `cdte_response_v4`
(the same keys, dimensions and meaning; `BasisSimulator.load_mc_response` reads them), with the
provenance alongside. Two tables:

- **`naeotom_alpha_cdte_response_geant4.jls`**: one native dexel without a grid, 1–140 keV, the
  nine thresholds, with the charge transported (section 3) and Zhan's 1.9 keV noise: a drop-in
  replacement for `cdte_response_v4`, computed from first principles. The fixed 30 µm cloud with
  1.5 keV noise, v0.15.1's model, is kept beside it under `"fixed_cloud"`.
- **`naeotom_alpha_cdte_response_grid_geant4.jls`**: the six positions in a grid cell, with the
  grid, plus its open fraction: what BasisSimulator would need to model the grid itself; the
  transported charge under `"tables"`, the fixed cloud under `"fixed_cloud_tables"`.
"""

# ╔═╡ 2084043f-e86d-474b-913c-96ae369abe99
md"""
The binned dexels' runs of section 5 again with the charge transported: the same six positions in
a grid cell, the same energies and seeds, and the same check that the fixed 30 µm cloud gives
back exactly section 5's responses. The electrodes at the edge of a grid cell also collect the
nearer half of the gap under the grid's wall, so they get their own, wider weighting potential.
"""

# ╔═╡ 25066dca-f0c1-438c-8846-fb0d4d231b9a
# Section 5's six positions, each energy's deposits brought back and the charge transported
transport_grid = let
    pool = Distributed.WorkerPool(workers)
    # each position's readout with the charge transported
    transported(ro) = MC.DexelReadout(;
        pitch_x_mm = ro.pitch_x_mm,
        pitch_y_mm = ro.pitch_y_mm,
        neighborhood = ro.neighborhood,
        group = ro.group,
        gap_mm = ro.gap_mm,
        target = ro.target,
        transport = charge_transport,
    )
    map(enumerate(grid_readouts)) do (k, ro)
        results = asyncmap(collect(grid_energies); ntasks = length(workers)) do E
            run = Distributed.remotecall_fetch(
                MC.simulate_monoenergetic,
                pool,
                alpha_crystal,
                E;
                n = 100_000,
                beam = MC.dexel_beam(ro),
                seed = 1,
                record_deposits = true,
            )
            e = findfirst(==(E), grid_energies)
            (
                transported = MC.dexel_response(run, transported(ro)),
                reproduces = MC.dexel_response(run, ro).dexel_energy_keV == grid_responses[k][e].dexel_energy_keV,
            )
        end
        (responses = [r.transported for r in results], reproduces = all(r.reproduces for r in results))
    end
end

# ╔═╡ b35eff96-a540-40ee-87d4-056d6ac7490f
# Every photon's counts in the 3 × 3 around the dexel it lands on, for each position type and energy,
# with the charge transported and Zhan's noise (the same noise draws as `grid_table`): the mean counts
# S1[slot, t] and the second moments S2[slot, slot′, t, t′] of the counts from one photon
mode_moments = let
    w = 4.43 / 1000
    nt = length(mode_thresholds)
    map(eachindex(positions)) do k
        map(transport_grid[k].responses) do r
            rng = BS.Random.Xoshiro(hash((1, r.energy_keV)))
            I = zeros(Float32, r.n, 9 * nt)
            for p in 1:r.n, a in 1:9
                energy = Float64(r.dexel_energy_keV[p, mod1(a, 3), cld(a, 3)])
                measured = energy + sqrt(zhan.noise_keV^2 + zhan.fano * w * max(energy, 0.0)) * randn(rng)
                for t in 1:nt
                    measured >= mode_thresholds[t] && (I[p, a + 9 * (t - 1)] = 1.0f0)
                end
            end
            S1 = reshape(Float64.(vec(sum(I; dims = 1))) ./ r.n, 9, nt)
            S2 = permutedims(reshape(Float64.(I' * I) ./ r.n, 9, nt, 9, nt), (1, 3, 2, 4))
            (; S1, S2)
        end
    end
end

# ╔═╡ 63d12259-ec35-41cb-acbb-47500f726483
"""
The moments of the photons landing on dexel `j = (k, l)` (any dexel of the detector, counted from a
grid cell's corner) at energy index `e`, at offsets `g` from it (global axes): `S1` at one offset, `S2`
at two; zero beyond its 3 × 3. Mirrored positions use their type's moments mirrored.
"""
function landing_moments(j, e)
    k, l = mod(j[1], 4) + 1, mod(j[2], 6) + 1
    m = mode_moments[cell_types.index[cell_types.type_map[k, l]]][e]
    mx, my = cell_types.mirror_x[k, l], cell_types.mirror_y[k, l]
    slot(g) = (abs(g[1]) <= 1 && abs(g[2]) <= 1) ? (g[1] * mx + 2) + 3 * (g[2] * my + 1) : 0
    S1(g) = (s = slot(g); s == 0 ? zeros(size(m.S1, 2)) : m.S1[s, :])
    S2(g, h) = (s = slot(g); u = slot(h); (s == 0 || u == 0) ? zeros(size(m.S2, 3), size(m.S2, 4)) : m.S2[s, u, :, :])
    return (; S1, S2)
end

# ╔═╡ fd10aa2e-f7ac-5b15-b095-6f2ead637241
# The machine path stays out of the rendered page
table_dir = joinpath(@__DIR__, "..", "..", "tables");

# ╔═╡ d090515d-ebde-502a-88ff-a1801857dd84
provenance = Dict(
    "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 03_alpha_detector.jl",
    "detector" => "Siemens NAEOTOM Alpha, CdTe 1.6 mm, native dexel 0.275 × 0.322 mm",
    "physics" => string(MC.PhysicsSettings()),
    "photons_per_energy" => 100_000,
    "illumination" => "uniform over the central dexel's electrode, normal incidence",
    "charge_model" => "transported: drift at $(charge_transport.bias_V) V across $(charge_transport.thickness_mm) mm, diffusion and repulsion from each deposit's depth, Shockley–Ramo induction on the dexel electrodes, Hecht trapping (μτ $(charge_transport.mu_tau_electron) and $(charge_transport.mu_tau_hole) cm²/V); electronic noise $(zhan.noise_keV) keV (Zhan et al. 2026); the fixed 30 µm cloud of BasisSimulator v0.15.1 is kept under \"fixed_cloud\"",
)

# ╔═╡ 440836e2-d3b5-4917-9e5d-ad1216731a1e
# Table A: native dexels, one row per position type (A–F), in counts per incident photon per native
# pitch area in a flat field
native_mode_table = let
    open = prod(alpha.group .* alpha.dexel) / prod(cell_pitch)  # landing photons per dexel per unit fluence
    nE, nt = length(grid_energies), length(mode_thresholds)
    receivers = [p.target for p in positions]
    offsets = [(dx, dy) for dx in -1:1, dy in -1:1]
    mean = zeros(6, nE, nt)
    psf = zeros(6, nE, 3, 3, nt)
    pair = zeros(6, nE, 3, 3, nt, nt)
    for (T, r) in enumerate(receivers), e in 1:nE
        for (dx, dy) in offsets
            n = (r[1] + dx, r[2] + dy)
            psf[T, e, dx + 2, dy + 2, :] = open .* landing_moments(n, e).S1((-dx, -dy))
            for (jx, jy) in offsets  # photons landing at r + (jx, jy) reach both r and n
                j = (r[1] + jx, r[2] + jy)
                pair[T, e, dx + 2, dy + 2, :, :] .+= open .* landing_moments(j, e).S2(r .- j, n .- j)
            end
        end
        mean[T, e, :] = vec(sum(psf[T, e, :, :, :]; dims = (1, 2)))
    end
    Dict{String, Any}(
        "schema_version" => "1",
        "readout_unit" => "native dexel (UHR mode reads these)",
        "dexel_mm" => collect(alpha.dexel),
        "pitch_mm" => [cell_pitch[1] / alpha.group[1], cell_pitch[2] / alpha.group[2]],
        "binning_kind" => "none (native)",
        "includes_absorption" => true,
        "multiplicity_allowed" => true,
        "grid" => "2-D, 4 × 6 dexel cells, walls over the gaps, $(alpha.wall_height) mm tall, tungsten (assumed); open fraction $(round(open; digits = 3))",
        "charge_model" => "transported (drift, diffusion, repulsion, Shockley–Ramo induction, Hecht trapping); section 3",
        "electronic_noise_keV" => zhan.noise_keV,
        "fano" => zhan.fano,
        "energies_keV" => collect(grid_energies),
        "thresholds_keV" => mode_thresholds,
        "type_labels" => [p.label for p in positions],
        "type_targets" => [collect(p.target) for p in positions],
        "type_map" => cell_types.type_map,
        "mirror_x" => cell_types.mirror_x,
        "mirror_y" => cell_types.mirror_y,
        "units" => "counts per incident photon per native pitch area (a twenty-fourth of a grid cell), flat field; cumulative above each threshold",
        "offsets" => "psf and pair index 1:3 ↔ offsets −1:1, first along x (columns, across the fan), then y (rows, along z); a mirrored position mirrors its type's offsets",
        "mean" => mean,
        "psf" => psf,
        "pair" => pair,
        "pileup" => pileup_inputs,
        "provenance" => provenance,
    )
end

# ╔═╡ b2c31398-74d1-44a8-80ee-6c49628680ce
# Table B: 2 × 2 binned pixels, added after counting, from Table A's moments; a grid cell holds 2 × 3 of
# them, at the corner (rows 0–1 and 4–5) or in the middle (rows 2–3) of the cell
binned_mode_table = let
    open = prod(alpha.group .* alpha.dexel) / prod(cell_pitch)
    nE, nt = length(grid_energies), length(mode_thresholds)
    members(G) = [(2G[1] + a, 2G[2] + b) for a in 0:1 for b in 0:1]
    groups = (corner = (0, 0), middle = (0, 1))
    near(m, n) = abs(m[1] - n[1]) <= 1 && abs(m[2] - n[2]) <= 1
    landings(G) = unique(vec([(m[1] + dx, m[2] + dy) for m in members(G), dx in -1:1, dy in -1:1]))
    mean = zeros(2, nE, nt)
    spill = zeros(2, nE, nt)
    second = zeros(2, nE, nt, nt)
    psf = zeros(2, nE, 3, 3, nt)
    pair = zeros(2, nE, 3, 3, nt, nt)
    for (g, G) in enumerate(values(groups)), e in 1:nE
        inside = members(G)
        for j in landings(G)
            lm = landing_moments(j, e)
            hit = [m for m in inside if near(m, j)]
            for m in hit
                contribution = open .* lm.S1(m .- j)
                mean[g, e, :] .+= contribution
                j in inside || (spill[g, e, :] .+= contribution)
                for m′ in hit
                    second[g, e, :, :] .+= open .* lm.S2(m .- j, m′ .- j)
                end
            end
            # what this landing photon adds to the neighbouring binned pixels' pairs and crosstalk
            ΔG = (fld(j[1], 2) - G[1], fld(j[2], 2) - G[2])
            for m in hit
                psf[g, e, ΔG[1] + 2, ΔG[2] + 2, :] .+= open .* lm.S1(m .- j)
            end
            for dx in -1:1, dy in -1:1
                (dx, dy) == (0, 0) && continue
                partner = members((G[1] + dx, G[2] + dy))
                for m in hit, n in partner
                    near(n, j) && (pair[g, e, dx + 2, dy + 2, :, :] .+= open .* lm.S2(m .- j, n .- j))
                end
            end
        end
        pair[g, e, 2, 2, :, :] = second[g, e, :, :]
    end
    Dict{String, Any}(
        "schema_version" => "1",
        "readout_unit" => "2 × 2 binned pixel (standard mode), counts of its four native dexels added after counting",
        "dexel_mm" => collect(2 .* alpha.dexel),
        "pitch_mm" => [2cell_pitch[1] / alpha.group[1], 2cell_pitch[2] / alpha.group[2]],
        "binning_kind" => "digital",
        "includes_absorption" => true,
        "multiplicity_allowed" => true,
        "grid" => native_mode_table["grid"],
        "charge_model" => native_mode_table["charge_model"],
        "electronic_noise_keV" => zhan.noise_keV,
        "fano" => zhan.fano,
        "energies_keV" => collect(grid_energies),
        "thresholds_keV" => mode_thresholds,
        "type_labels" => ["corner", "middle"],
        "type_map" => [l in (0, 2) ? "corner" : "middle" for k in 0:1, l in 0:2],
        "mirror_y" => [l == 2 ? -1 : 1 for k in 0:1, l in 0:2],
        "units" => "counts per incident photon per native pitch area on each native dexel (the same unit as Table A), flat field; cumulative above each threshold",
        "offsets" => "psf and pair index 1:3 ↔ offsets −1:1 in binned pixels, x then y; pair[…, 2, 2, …] is the binned pixel's own second moment",
        "mean" => mean,
        "spill_in" => spill,
        "psf" => psf,
        "pair" => pair,
        "pileup" => pileup_inputs,
        "provenance" => provenance,
    )
end

# ╔═╡ 8ba64dd5-6b2c-5c06-8a3b-0e94a0f5941d
alpha_table = let
    table = MC.response_table(
        transport_runs.responses,
        thresholds;
        electronic_noise_keV = zhan.noise_keV,
        metadata = merge(provenance, Dict("grid" => "none")),
    )
    # the width one Gaussian cloud would need to count alike (Zhan's measure), for readers of this key
    table["sigma_cloud_um"] = first(zhan_check.beams).sigma_um
    table["pileup"] = pileup_inputs
    table["variation"] = Dict{String, Any}(
        "model" => "per dexel: every threshold offset by one Gaussian draw (keV), every pulse height scaled by one gain (1 + a Gaussian draw); fixed for the dexel's life",
        "threshold_offset_keV_rms" => variation.threshold_keV[2],
        "gain_rms" => variation.gain[2],
        "studied_ranges" => Dict("threshold_offset_keV_rms" => variation.threshold_keV, "gain_rms" => variation.gain),
        "status" => "assumed: no clinical scanner publishes them; Medipix3RX about 0.18 keV (Ballabriga 2013), CdTe 0.34 keV (Liu 2025), gain 3 % (Maj 2014); Zhan 2026's 1.9 keV fitted noise bounds the offsets below 1.2 keV",
        "correction" => "flat field per dexel and bin from an air scan (calibrate from simulated calibration scans only, never the true offsets)",
    )
    table["fixed_cloud"] = MC.response_table(
        responses,
        thresholds;
        electronic_noise_keV = current_table["sigma_elec_keV"],
        metadata = Dict("charge_model" => "one Gaussian cloud of 30 µm and 1.5 keV noise, as BasisSimulator v0.15.1"),
    )
    table
end

# ╔═╡ 043f5250-d081-56b7-9379-acbcfea44d1a
grid_table = Dict(
    "positions" => [p.label for p in positions],
    "targets" => [p.target for p in positions],
    "tables" => [
        MC.response_table(
                transport_grid[k].responses,
                thresholds;
                electronic_noise_keV = zhan.noise_keV,
            )
            for k in eachindex(positions)
    ],
    "fixed_cloud_tables" => [
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

# ╔═╡ 6e97e2a3-ca00-4b93-ab7f-9c78dd852698
# The tables' checks: the native Poisson identity, the crosstalk adding up to the mean, the binned mean
# and second moment against their members (recomputed from Table A's own arrays), and Table A against
# the response tables already saved (the same photons and noise draws)
mode_checks = let
    A, B = native_mode_table, binned_mode_table
    nE, nt = length(grid_energies), length(mode_thresholds)
    poisson = maximum(abs(A["pair"][T, e, 2, 2, t, u] - A["mean"][T, e, max(t, u)]) for T in 1:6, e in 1:nE, t in 1:nt, u in 1:nt)
    psf_sum = maximum(abs.(dropdims(sum(A["psf"]; dims = (3, 4)); dims = (3, 4)) .- A["mean"]))
    # members in Table A's terms: their types (mirrored as the cell's map says) and their pairs
    type_of(m) = cell_types.index[cell_types.type_map[mod(m[1], 4) + 1, mod(m[2], 6) + 1]]
    flip(m) = (cell_types.mirror_x[mod(m[1], 4) + 1, mod(m[2], 6) + 1], cell_types.mirror_y[mod(m[1], 4) + 1, mod(m[2], 6) + 1])
    pair_of(m, n, e) = let f = flip(m), d = (n[1] - m[1], n[2] - m[2])
        A["pair"][type_of(m), e, d[1] * f[1] + 2, d[2] * f[2] + 2, :, :]
    end
    groups = ((0, 0), (0, 1))
    members(G) = [(2G[1] + a, 2G[2] + b) for a in 0:1 for b in 0:1]
    binned_mean = maximum(
        maximum(abs.(sum(A["mean"][type_of(m), e, :] for m in members(G)) .- B["mean"][g, e, :]) ./ (B["mean"][g, e, :] .+ 1.0e-12))
            for (g, G) in enumerate(groups), e in 1:nE
    )
    binned_second = maximum(
        maximum(abs.(sum(pair_of(m, n, e) for m in members(G), n in members(G)) .- B["pair"][g, e, 2, 2, :, :]) ./ (B["pair"][g, e, 2, 2, :, :] .+ 1.0e-12))
            for (g, G) in enumerate(groups), e in 1:nE
    )
    # Table A's photons against the saved grid table (single precision), at the thresholds both hold
    common = [findfirst(==(t), mode_thresholds) for t in thresholds]
    saved_agree = maximum(
        abs(sum(mode_moments[k][e].S1[:, common[i]]) - grid_table["tables"][k]["R_total"][e, i])
            for k in 1:6, e in 1:nE, i in eachindex(thresholds)
    )
    fano = [B["pair"][g, e, 2, 2, 1, 1] / B["mean"][g, e, 1] for g in 1:2, e in 1:nE]
    spill_share = [B["spill_in"][g, e, 1] / B["mean"][g, e, 1] for g in 1:2, e in 1:nE]
    (; poisson, psf_sum, binned_mean, binned_second, saved_agree, fano, spill_share)
end

# ╔═╡ 0a39985e-dd88-41b9-a201-193f02ba49ab
let
    pass(ok) = ok ? "✓" : "✗ **fail**"
    lo, hi = zhan.sigma_range_um
    rows = String[]
    for b in zhan_check.beams
        push!(rows, "| $(pass(lo <= b.sigma_um <= hi)) | charge spreading, $(b.label): the single Gaussian that counts alike | $(round(b.sigma_um; digits = 1)) µm | $(lo)–$(hi) µm | Zhan et al. 2026, clinical Alpha |")
    end
    push!(rows, "| ✓ | electronic noise used with the transported charge | $(zhan.noise_keV) keV | $(zhan.noise_keV) keV | Zhan et al. 2026 |")
    push!(rows, "| $(pass(transport_runs.reproduces)) | the same photons with the fixed 30 µm cloud give section 3 exactly | $(transport_runs.reproduces ? "identical" : "different") | identical | bookkeeping |")
    clinical_ok = all(
        abs(c.direct[i] - c.table[i]) <= 3c.noise[i] + 0.003
            for c in clinical_check for i in eachindex(c.bins)
    )
    push!(rows, "| $(pass(clinical_ok)) | the single-energy counts, weighted by the clinical spectra, are the clinical beams (section 11) | $(clinical_ok ? "all agree" : "some differ") | 3σ + 0.003 | Geant4, photon by photon |")
    for kV in (120, 140)
        curve = pileup_fit.curves[kV]
        inside = all(
            li_loss(kV, I; τ_ns = li2025.tau_ci_ns[1]) - 0.01 <= L <= li_loss(kV, I; τ_ns = li2025.tau_ci_ns[2]) + 0.01
                for (I, L) in zip(curve.mA, curve.loss)
        )
        worst = maximum(abs(L - li_loss(kV, I)) for (I, L) in zip(curve.mA, curve.loss))
        push!(rows, "| $(pass(inside)) | pile-up, $(kV) kV in air: counts lost against tube current, shaped pulses | largest difference $(round(worst; digits = 3)) | inside the 95 % band (5.6–7.1 ns) ± 0.01 | Li et al. 2025, clinical Alpha, correction off |")
    end
    c = mode_checks
    push!(rows, "| $(pass(c.poisson <= 1.0e-12)) | Table A: a native dexel's counts are Poisson (second moment = mean above the higher threshold) | $(round(c.poisson; sigdigits = 2)) | ≤ 10⁻¹² | identity |")
    push!(rows, "| $(pass(c.psf_sum <= 1.0e-12)) | Table A: the crosstalk from every neighbour adds up to the mean | $(round(c.psf_sum; sigdigits = 2)) | ≤ 10⁻¹² | identity |")
    push!(rows, "| $(pass(c.binned_mean <= 1.0e-10)) | Table B: a binned pixel's mean is its members' means, spill-in included (relative) | $(round(c.binned_mean; sigdigits = 2)) | ≤ 10⁻¹⁰ | Table A |")
    push!(rows, "| $(pass(c.binned_second <= 1.0e-10)) | Table B: its second moment is its members' moments plus twice the pairs inside the group (relative) | $(round(c.binned_second; sigdigits = 2)) | ≤ 10⁻¹⁰ | Table A |")
    push!(rows, "| $(pass(c.saved_agree <= 1.0e-6)) | Table A: the same photons and noise as the saved response table, at its 9 thresholds | $(round(c.saved_agree; sigdigits = 2)) | ≤ 10⁻⁶ (single precision) | the grid table |")
    passed = count(r -> startswith(r, "| ✓"), rows)
    Markdown.parse(
        """
        | | check | measured | target | source |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        **$(passed) of $(length(rows)) checks pass.** Over the clinical beams the transported charge spreads like a single
        Gaussian of $(round(minimum(b.sigma_um for b in zhan_check.beams); digits = 1))–$(round(maximum(b.sigma_um for b in zhan_check.beams); digits = 1)) µm, against Zhan's 16.4 µm fitted to the clinical Alpha, with no parameter of the
        transport fitted to it; BasisSimulator v0.15.1's 30 µm is about twice too wide. Energy by energy the median is
        $(round(sort(filter(!isnan, zhan_check.per_energy))[cld(length(filter(!isnan, zhan_check.per_energy)), 2)]; digits = 1)) µm; single energies just above a threshold (15 and 55 keV) pin the width down poorly and scatter more.
        """
    )
end

# ╔═╡ a6fff8e2-cad7-41eb-93b3-3abb18fb2154
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 1000))
    shared = Mke.RGBf(0.85, 0.55, 0.1)

    # (a) native: a shared photon counts once in each of two dexels
    axa = canvas(fig[1, 1]; title = "(a) UHR: Every Native Dexel Read on Its Own", titlesize = 19)
    for i in 0:2, j in 0:2
        rect!(axa, i * 1.1, j * 1.25, 1.0, 1.15, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1.2)
    end
    Mke.scatter!(axa, [Mke.Point2f(1.08, 1.85)]; marker = :star5, markersize = 26, color = shared)
    for (x, y) in ((0.5, 1.83), (1.6, 1.83))
        Mke.textlabel!(axa, Mke.Point2f(x, y + 0.35); text = "+1", text_align = (:center, :center), fontsize = 14, font = :bold, text_color = :white, background_color = PALETTE.kept, strokewidth = 0, cornerradius = 7)
    end
    Mke.text!(axa, 1.6, -0.3; text = "A Photon on the Edge Counts Once\nin Each Dexel: Each Dexel Is Poisson", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.limits!(axa, -0.4, 3.6, -1.6, 4.0)

    # (b) standard: the 2 × 2 sum counts it twice, and photons next door spill in
    axb = canvas(fig[1, 2]; title = "(b) Standard: 2 × 2 Added After Counting", titlesize = 19)
    for i in 0:2, j in 0:2
        inside = i <= 1 && j >= 1
        rect!(axb, i * 1.1, j * 1.25, 1.0, 1.15, inside ? Mke.RGBf(0.8, 0.9, 0.8) : MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1.2)
    end
    Mke.lines!(axb, Mke.Point2f[(-0.08, 1.17), (2.18, 1.17), (2.18, 3.73), (-0.08, 3.73), (-0.08, 1.17)]; color = PALETTE.kept, linewidth = 3)
    Mke.scatter!(axb, [Mke.Point2f(1.08, 1.85)]; marker = :star5, markersize = 26, color = shared)
    Mke.textlabel!(axb, Mke.Point2f(1.08, 3.95); text = "+2 From One Photon", text_align = (:center, :bottom), fontsize = 14, font = :bold, text_color = :white, background_color = PALETTE.kept, strokewidth = 0, cornerradius = 7)
    Mke.scatter!(axb, [Mke.Point2f(2.75, 1.85)]; marker = :star5, markersize = 22, color = (shared, 0.7))
    arrow!(axb, (2.65, 1.85), (2.1, 1.85); color = shared, linewidth = 2.5, head = 11)
    Mke.text!(axb, 1.6, -0.3; text = "Shared Inside the Group: Counted Twice\n(Variance > Mean); Next Door: Spills In", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.limits!(axb, -0.4, 3.6, -1.6, 4.6)

    # (c) what each table holds
    axc = canvas(fig[1, 3]; title = "(c) What the Tables Hold", titlesize = 19)
    lines = (
        ("Table A, Native", "Mean, Crosstalk From Each Neighbour,\nPair Moments; Six Positions in the Grid Cell"),
        ("Table B, 2 × 2 Binned", "The Same, per Binned Pixel; Corner\nand Middle Pixels of the Cell"),
        ("Both", "16 Thresholds, 10–140 keV, the Grid Inside;\nPile-Up Inputs (Section 10)"),
    )
    for (k, (title, detail)) in enumerate(lines)
        y = 3.4 - 1.5 * (k - 1)
        Mke.text!(axc, 0.0, y; text = title, align = (:left, :bottom), fontsize = 16, font = :bold, color = PALETTE.ink)
        Mke.text!(axc, 0.0, y - 0.1; text = detail, align = (:left, :top), fontsize = 14, color = PALETTE.muted)
    end
    Mke.limits!(axc, -0.2, 5.0, -0.4, 4.2)

    # the binned pixels' counts are not Poisson
    ax = Mke.Axis(
        fig[2, 1:3];
        title = "Variance ÷ Mean of the Counts Above 20 keV, Flat Field",
        xlabel = "Photon Energy (keV)",
        ylabel = "Variance ÷ Mean",
        limits = ((10, 140), (0.95, 1.35)),
    )
    Mke.hlines!(ax, 1; color = PALETTE.photon, linewidth = 3, label = "Native Dexel (Exactly Poisson)")
    for (g, (label, color)) in enumerate((("2 × 2 Binned, Corner of the Cell", PALETTE.kept), ("2 × 2 Binned, Middle of the Cell", PALETTE.xray)))
        Mke.lines!(ax, collect(grid_energies), mode_checks.fano[g, :]; color, linewidth = 3, label)
    end
    Mke.axislegend(ax; position = (0.98, 0.35))
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.5))
    locator!(fig[0, :], (:electrodes, :asic); note = "Steps 4–5: Which Dexels\nAre Read, and How\nTheir Counts Are Added")
    fig
end

# ╔═╡ be6bc69c-d64d-4b6e-b3a6-ac57b43a08ed
let
    k60 = findfirst(==(60.0), collect(grid_energies))
    Markdown.parse(
        """
        Around 60 keV a 2 × 2 binned pixel's counts above 20 keV have a variance **$(round(mode_checks.fano[1, k60]; digits = 3))** times their mean
        at the corner of a cell and **$(round(mode_checks.fano[2, k60]; digits = 3))** in its middle: photons shared inside the group count twice. Of the
        counts, $(round(100 * mode_checks.spill_share[2, k60]; digits = 1)) % come from photons that landed in a neighbouring group. A native dexel is
        Poisson to round-off. Both tables are saved beside the response tables (below), and the scorecard checks them.
        """
    )
end

# ╔═╡ ba9df479-0755-53ad-83b1-635caea0f8ec
saved = let
    mkpath(table_dir)
    paths = (
        joinpath(table_dir, "naeotom_alpha_cdte_response_geant4.jls"),
        joinpath(table_dir, "naeotom_alpha_cdte_response_grid_geant4.jls"),
        joinpath(table_dir, "naeotom_alpha_cdte_native_v1.jls"),
        joinpath(table_dir, "naeotom_alpha_cdte_binned2x2_v1.jls"),
    )
    Serialization.serialize(paths[1], alpha_table)
    Serialization.serialize(paths[2], grid_table)
    Serialization.serialize(paths[3], native_mode_table)
    Serialization.serialize(paths[4], binned_mode_table)
    basename.(paths)
end;

# ╔═╡ b5b66b9b-9dd8-58c6-967a-3571f17ffd31
# BasisSimulator reads the new table with the same loader it uses for cdte_response_v4
reloaded = (saved; BS.load_mc_response(joinpath(table_dir, "naeotom_alpha_cdte_response_geant4.jls")))

# ╔═╡ 288a1e92-09f9-55d2-b6e3-06b2ff4adc96
let
    largest = maximum(abs, reloaded.R_total .- current_table["R_total"])
    Markdown.parse(
        """
        BasisSimulator loads the Geant4 table: $(length(reloaded.energies_keV)) energies × $(length(reloaded.thresholds_keV))
        thresholds. With the charge transported its counts differ from `cdte_response_v4`'s by up to
        **$(round(largest; digits = 3))** counts per photon: v0.15.1's 30 µm cloud shares about twice as much charge as the
        clinical Alpha does (section 3).
        """
    )
end

# ╔═╡ 6cd7d71c-20c3-568f-9386-3ee1047ed406
md"""
## Where This Leaves the Alpha

- **One dexel**, simulated in Geant4 from first principles, reproduces BasisSimulator v0.15.1's
  MCNP table (`cdte_response_v4`) to about 0.01 counts per photon at every threshold, with the
  same 30 µm charge cloud.
- **The charge, transported** (drift, diffusion, repulsion, induction, trapping), spreads like a
  single 15–16 µm Gaussian, matching Zhan et al.'s fit to the clinical Alpha (16.4 µm), with
  nothing fitted to it: v0.15.1's 30 µm is about twice too wide. The saved tables use it.
- **The anti-scatter grid**, absent from BasisSimulator, changes the share of the detector that
  sees photons from 0.95 to about 0.72, without making dexels unequal in an air scan.
- **The whole detector**, built one to one in Geant4, behaves as its six cell positions: a
  full response table needs six dexels per row, not 792 576.
- **Pile-up**, simulated from shaped pulses at the crystal's true pulse rate (charge sharing makes
  about 1.44 pulses per photon above 20 keV at 140 kV) and summed digitally 2 × 2, fits Li et al.'s
  count loss on two clinical Alphas with a retriggering comparator, a 10 ns pulse and a 27 ns
  dead time per subpixel (section 10). Bhattarai et al.'s noise rise at high current is larger than
  the model's; their scans had Siemens' own pile-up correction on, so it is shown as context.
- **No two dexels alike** (section 12): at an assumed 0.35 keV threshold spread and 2 % gain spread,
  a flat-field calibration from a simulated air scan leaves about a 4 HU ring; Zhan's measured noise
  bounds the threshold spread at about 1.2 keV. The spreads are assumed, not published.
- **A clinical beam**, photon by photon, gives the single-energy response weighted by its spectrum
  (section 11): the table on a 1-keV grid is all a scan needs below pile-up rates.
- **Standard and UHR modes** (section 13): one native table (the six grid positions, 16 thresholds,
  each photon's spread to its neighbours and the paired counts between them, the pile-up inputs) and
  the 2 × 2 binned table derived from it, with the counts that spill in from neighbouring groups.
  Binned counts are not Poisson: their variance is 1.17 × the mean at 60 keV and 1.28 × at 140 keV.

What replaces BasisSimulator v0.15.1's tables: the native and 2 × 2 binned tables of section 13,
with the transported charge, the pile-up inputs and the assumed dexel variation, all saved by this
notebook as plain data BasisSimulator reads without BasisMonteCarlo.
"""

# ╔═╡ Cell order:
# ╟─6e30fdb2-879c-52f4-99ab-666bbb68cd4f
# ╟─97e89e01-c1b8-4d6e-892f-b17751b636d1
# ╟─df200390-005d-5328-870a-9a821744961d
# ╟─e71ceeb9-0d88-5608-8588-ee48a312de90
# ╟─f6b40d89-2b55-4c5e-b8e7-881636d05f6e
# ╟─8326156c-5e6a-5f16-89ab-5e8b268b2e0a
# ╟─debadf41-b238-45ca-a2f2-eeecff9a20f5
# ╟─22a88d29-5771-4e17-ab70-f33d13c288a8
# ╟─20aa77fa-7c82-4c2a-b72d-58c13baead11
# ╟─f7ad4569-b274-5714-83ab-bc7ad0497b53
# ╟─943d5189-ca8c-5e1b-860d-dad9ed192843
# ╠═ed5d0968-55b8-54bb-be08-b8b3e5231445
# ╠═73911ef9-afec-5057-9406-29f4217f5d15
# ╠═aa57fb94-4f51-5853-8cf5-c0598ee6b717
# ╠═9215c3a9-8e71-5f66-b12a-2779e5350eda
# ╠═1ff7c097-d985-5412-9686-8e062018d0b2
# ╠═bc2a467c-d990-556b-b955-dd91539122ab
# ╠═fdaa0020-4804-5c58-8453-668f6c70ebf3
# ╠═5285d69f-7fb4-488a-9954-5e2d36bf1204
# ╠═0df22b09-66d5-59e9-9f68-79a2b1886e38
# ╠═40986dbf-a0d8-51ea-aa7b-75b21dcc7eaa
# ╠═a944bba9-6761-5648-b028-0d90a7b56e29
# ╠═c5fa8b52-766b-5c4a-9794-a056c8e00dc2
# ╠═eb7fb66a-4e78-5a01-ada0-e323524fde34
# ╟─f3001712-cc0a-48db-8c7e-27e06c5aa3fd
# ╠═2444710c-f16a-47aa-9ea5-d3cdaa2c6d19
# ╠═18913250-92bd-415e-9559-55e73934e2eb
# ╠═de754d25-442e-4d20-8565-4e0b9a502db7
# ╠═dc842efe-90d4-435c-a6de-fc2375005697
# ╠═72cb803c-129a-4891-ac42-8308cc4ce50a
# ╠═ee6c802f-d1f0-432e-bae6-166cf16320d3
# ╠═792826ce-b8c6-4fbd-9919-8c719d716f49
# ╠═eff17814-a2f6-4d61-a9cd-df7458f5b5b0
# ╠═896c9200-6700-48f4-beb2-cffee0c1ec0f
# ╠═e6249cb2-6672-4f2c-956d-a4e947fa0d07
# ╠═ed2d4d8c-1329-47a4-841f-7affbc4d204f
# ╠═1d5e84c1-714b-47d6-a120-71068b37aebc
# ╠═dd76ac67-b4fa-4c94-bdf6-4b431f3d14a4
# ╠═fe495bf4-f87f-45c4-b1ab-16c586bc4df5
# ╠═a05ae43e-08a8-4696-b931-100fe6023d33
# ╠═a2be090c-6854-4fa7-90ce-5494da5bd57c
# ╠═cf02ddf8-fcef-413d-a30d-d51071fdc761
# ╠═2a420206-70e3-4019-945c-f628bd6ee0df
# ╠═290c44b9-29d5-4d71-ae2b-b14fa5fc02d5
# ╠═c7ef0930-55ad-4a4f-8296-db3571c54a88
# ╠═06101f43-57fd-44a5-b7aa-cd22bfaa5eb1
# ╠═246bff85-ebee-4478-a199-b432c91f57ad
# ╠═7b539e6e-ee97-4464-a2c5-f09abe2a32f7
# ╠═7cefafb2-4fa6-43f2-9cc7-c64d016df12b
# ╠═9893a0c1-2162-4290-8273-19740f076747
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
# ╟─8b38185f-e423-49ca-af97-536138ddc8be
# ╟─6a162bd9-0b36-5cf6-b16e-6b6d095b5ee8
# ╟─043f6510-e50d-4d4c-b343-8fee4e5918f3
# ╠═9950b293-c447-457b-bd7b-5fa3d4a8eddd
# ╠═8e06edcd-ec4f-4332-a291-921130d544c2
# ╠═29622b50-54f1-4b0c-a988-724a4732970d
# ╟─8ea66d84-4cce-4ff9-881f-1ea60174ef27
# ╟─88e2e00a-8c74-4c9e-ac31-88663466235e
# ╟─44c8451f-2fdf-4a30-b09c-0a0bf51f7ad2
# ╟─6eaa219f-a038-4754-a84a-692db7826816
# ╠═d771059c-df37-4346-a774-fa326eb07309
# ╠═1c1f1e65-29b9-4154-b499-ad21f8a26d31
# ╟─5667adb0-4579-4f57-bee7-4b081cdb4f9c
# ╟─5eb2d62c-5e6b-48f3-b451-cb69665bebc1
# ╟─7ce20715-234a-4095-a100-0206821b7cfd
# ╠═788bd9cd-7e98-4bba-a04a-97b4f96f1e2e
# ╟─134ebfbd-ecdd-432f-a76f-94577b100925
# ╟─0a39985e-dd88-41b9-a201-193f02ba49ab
# ╟─65f1babc-4849-5e3a-8404-738562c88837
# ╠═0eff78e6-df72-522a-b5db-d70421a6cdec
# ╟─0c8a2fcc-ca65-4040-9b13-973725659b2e
# ╠═4e507ffb-04ca-5f56-a577-e2715de205a4
# ╠═124978ab-3b89-5c61-996d-64d7f222d405
# ╟─cb866af1-b9f8-467d-bab4-77815e589bd9
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
# ╟─313586e2-12f3-49c0-8418-027e1f21533c
# ╟─52e68256-0954-5993-b7ea-a0ef64dbde97
# ╟─9133ec80-9f4b-5fbd-8d36-626ca9c12719
# ╠═7081fb53-7350-5ef5-82fc-c3866ca8dc5a
# ╠═bfee1bcb-7273-5f67-b684-42e8be79191d
# ╟─c0f06895-b97e-5bc3-81dd-bdd5b1a8a0c3
# ╟─8a8a6147-7140-4c47-8268-4d0002045acf
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
# ╟─c81376fc-7b62-4e44-af3b-ec6cba4a8a9c
# ╟─877c4989-7b59-5741-a50a-433cf4a4a14b
# ╟─848bfaff-246d-4084-989c-c17ef4008e41
# ╠═648db0cd-57fc-46a7-894e-06ab540140a2
# ╟─01ccb19c-6a12-4958-b5c0-acb5ea96acfa
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
# ╟─b3df600c-9c49-48f5-89b1-760870751d97
# ╟─5138ed44-8263-5dbf-9d6e-d78c085c4a85
# ╟─f47124cb-a324-5c43-b885-8868f266f9e8
# ╟─fe7057a7-e676-5a62-899c-da11603bb9ed
# ╠═b42aaddf-0087-5766-91d5-7a5d1fd04261
# ╠═4d962a9a-3fd6-5fab-a132-c9c9907d4abc
# ╠═98710092-8d7f-56d4-964f-36441ac6613a
# ╟─dc89fc6d-57b5-42e2-97ba-04cb2fb1614d
# ╟─4c43a065-f125-5640-8df8-a2b16fdbdd4e
# ╟─455fbb1d-4de4-588e-98b7-7d0159a29d63
# ╟─399f9fe4-230f-450c-80b4-669326efdbf2
# ╠═67a5c658-bcfe-41b5-a154-f5f04a8cfd9f
# ╠═cf22256e-4caa-4a14-86f2-6bda4c461ff8
# ╠═e4f9fe55-8ec8-44df-9d56-d70389f7015c
# ╠═7a122efb-af3a-4f8f-862d-8196f6246c53
# ╠═0cb750de-25a1-4668-9665-42b7a8fca2f7
# ╠═c4a1eb26-d693-427b-9b8f-95d586a55a88
# ╟─8b62777b-7fe3-4637-aa92-73ba515ddb45
# ╟─7baa1504-7942-48ec-93f2-a03d48d3430f
# ╟─ccc7fe92-998c-48d6-8ff5-bf063506827e
# ╠═e9d0ef22-c3ce-4023-a86d-963a234c51d6
# ╟─2fe8db09-89e3-43fb-99a9-2470d61844eb
# ╠═a1549f34-7dfe-431b-9d68-43c201ed3f6c
# ╟─905e2665-4c7e-4686-baed-5ec32a2cf16d
# ╟─18d124be-b21d-4496-b644-c00fe5abae68
# ╠═e9a4425a-7892-48c8-8065-9733bc99b607
# ╟─88117c52-14b8-4302-a0e5-2231d12504e2
# ╟─25f703a9-2bc9-4781-9a28-dd84484cb1d6
# ╠═888c0650-ebc4-4bfa-8880-335e82f146c0
# ╠═2956ae91-7028-45a0-9466-12370ae5a928
# ╠═b93ed54c-9e8b-45ee-b09c-7a06a1f1aed2
# ╟─1029e073-9f7e-4bc3-a1b0-e5ec3a79b03f
# ╟─b8846f41-7a5f-41bd-90b9-0881b4ecc0e4
# ╟─34ec1ace-4eae-40b1-b8b4-8b9c5a917bd4
# ╠═1a4e2fa3-068c-4d0a-9ed6-146919ba2761
# ╠═0a49a6bc-b410-405c-a612-91ab87436366
# ╠═6b8a272a-a980-4d31-93ca-3a22e568337e
# ╟─66f58509-736f-48c3-8996-363d825bb34f
# ╟─ec8ea99e-dadc-400a-9fd3-df2680140d93
# ╟─7d5f8a64-152a-43e0-b3a0-cb46f12f2ff5
# ╟─88baa453-2c9f-4e83-abef-b81e449f8453
# ╠═04b43622-f984-48b6-b488-430bc420835e
# ╠═67171632-72ab-4d86-9dd1-f9188f86332d
# ╠═b35eff96-a540-40ee-87d4-056d6ac7490f
# ╠═63d12259-ec35-41cb-acbb-47500f726483
# ╠═440836e2-d3b5-4917-9e5d-ad1216731a1e
# ╠═b2c31398-74d1-44a8-80ee-6c49628680ce
# ╠═6e97e2a3-ca00-4b93-ab7f-9c78dd852698
# ╟─a6fff8e2-cad7-41eb-93b3-3abb18fb2154
# ╟─be6bc69c-d64d-4b6e-b3a6-ac57b43a08ed
# ╟─92576991-a76a-5062-b4fc-e5d8604dd672
# ╟─2084043f-e86d-474b-913c-96ae369abe99
# ╠═25066dca-f0c1-438c-8846-fb0d4d231b9a
# ╠═fd10aa2e-f7ac-5b15-b095-6f2ead637241
# ╠═d090515d-ebde-502a-88ff-a1801857dd84
# ╠═8ba64dd5-6b2c-5c06-8a3b-0e94a0f5941d
# ╠═043f5250-d081-56b7-9379-acbcfea44d1a
# ╠═ba9df479-0755-53ad-83b1-635caea0f8ec
# ╠═b5b66b9b-9dd8-58c6-967a-3571f17ffd31
# ╟─288a1e92-09f9-55d2-b6e3-06b2ff4adc96
# ╟─6cd7d71c-20c3-568f-9386-3ee1047ed406
