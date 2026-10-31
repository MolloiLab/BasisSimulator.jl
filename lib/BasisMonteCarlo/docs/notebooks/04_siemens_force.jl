### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ 5db002c4-197d-59f8-b950-3eb200999f19
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 72bdb567-8286-5781-9549-ed9f1896c3ce
using Markdown: @md_str, Markdown

# ╔═╡ 4c28b150-8255-5c6f-9ffa-004001712d76
using Statistics: mean

# ╔═╡ 27f245eb-99f2-5e57-b186-46b2371b7d17
md"""
# The SOMATOM Force Detector in Geant4

The Siemens SOMATOM Force is a dual-source, energy-integrating CT scanner. Each of its two
detectors (the Stellar Infinity) is an array of **UFC**, Siemens' "Ultra Fast Ceramic":
gadolinium oxysulfide (Gd₂O₂S) sintered to a dense ceramic and sawn into **dexels** (detector
elements), with a reflector in the cuts and a photodiode under every dexel. In front of the
array stands a two-dimensional anti-scatter grid.

Up to and including **v0.15.1**, BasisSimulator describes how efficiently this detector turns
x rays into signal with one table, η(E) from 1 to 140 keV (`UFC_MC_EFFICIENCY_LUT`), computed
with MCNP by Hamidreza Khodajou-Chokami; the inputs to that simulation were never committed.
This notebook builds the detector in Geant4, one piece at a time, and

1. **checks every piece against the documentation**: Siemens' data sheets and white paper, the
   FDA clearance and the literature;
2. **reproduces the v0.15.1 table**, which shows how it was made, and where it parts from the
   documented detector;
3. **builds a better table** from the documented detector, with its noise (the Swank factor),
   and saves it: the table BasisSimulator takes over from v0.16.0.

!!! note "Which BasisSimulator this notebook compares against"
    Every comparison with "the v0.15.1 table", and every "BasisSimulator v0.15.1" value (the
    density 7.44 g/cm³, the documented depth 1.4 mm, the fill factor), is with **BasisSimulator
    v0.15.1**, the last release before this notebook's table. BasisMonteCarlo keeps that table
    (`BASISSIMULATOR_V0_15_1_EFFICIENCY`, copied verbatim), so the comparisons stay the same
    after BasisSimulator moves to the new one. Only the x-ray spectra and the bowtie
    (sections 8–10) come from the **installed** BasisSimulator, whatever its version; nothing
    here pins BasisSimulator, or BasisMonteCarlo, to a version.
"""

# ╔═╡ bc5428b3-9cde-5d59-ad60-0b21685419fd
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a UFC dexel does not count at all. Every
x ray it absorbs leaves energy in the ceramic, which glows; the reflector keeps the light inside
the dexel, and the photodiode under it adds up the light of every photon over the reading. The
signal is therefore proportional to the **energy** left in the dexel, and the efficiency that
matters is the energy-absorption efficiency: the mean energy a dexel collects per photon,
divided by the photon's energy.
"""

# ╔═╡ 0128955f-7935-42d8-93e3-b27320625add
md"""
### Which Beam, and Why

How the photons arrive decides what a simulation measures, so each section says which beam it
uses. Five are used here, from the simplest to the most realistic:

| beam | what it measures | where |
|---|---|---|
| **pencil beam** into the centre of one dexel | how the v0.15.1 table was made (it is reproduced exactly) | section 5 |
| **uniform on one dexel's ceramic** | the efficiency per photon landing on the ceramic; reading every dexel around it gives the flat field (below) | sections 3–5 |
| **uniform on one cell** (ceramic and its reflector), under the grid | what the grid costs, straight and at an angle | section 6 |
| **flat field**: every dexel lit | what each dexel collects in a scan: what it loses to its neighbours it gets back from them. By symmetry this equals all the energy a uniform beam on one dexel leaves anywhere, which is how it is simulated | sections 4, 5 |
| **from the focal spot**, over the whole arc, with **clinical spectra** | everything at once: real angles, every dexel, the grid, the Force's own 120 kV beam (also after 20 and 35 cm of water) | sections 7, 10 |

The efficiency table is built from single energies, because BasisSimulator weights it by whatever
spectrum a scan uses. Section 10 checks that this shortcut is exact: the clinical beams, simulated
photon by photon, give the same signal and noise as the table weighted by their spectra.
"""

# ╔═╡ c8bd4fe5-c38c-5d02-ab51-8d1653cd5d74
md"""
## The SOMATOM Force, as Documented

Every hardware number this notebook uses, checked against Siemens' documents, the FDA 510(k)
summary and the literature. Where BasisSimulator v0.15.1 and a source disagree the row is marked
**⚠ discrepancy**; where no source gives a number the row says so, and the notebook states its
assumption.

| quantity | BasisSimulator v0.15.1 | documented | status | source |
|---|---|---|---|---|
| scintillator | UFC, Gd₂O₂S | UFC: gadolinium oxysulfide ceramic; dopants proprietary, below 0.1 % | ✓ | Siemens UFC data sheet; UFC safety data sheet §3 |
| density | 7.44 g/cm³ (Geant4's and NIST's stock Gd₂O₂S) | 7.29–7.33 g/cm³ (99.95 % of theoretical) | **⚠ discrepancy** (+2 %); 7.31 used here | UFC data sheet §1.4.1 |
| crystal depth | 1.4 mm | 1.4 mm is the only UFC thickness Siemens quotes; never tied to a scanner | not confirmed (plausible) | UFC data sheet §1.1.1; Detectors & Crystals p. 19 |
| its efficiency table | η(140 keV) = 0.816 | 1.4 mm of UFC can absorb at most 0.68 of 140 keV photons | **⚠ discrepancy** (section 5) | this notebook |
| rows × row width | 96 × 0.6 mm | 96 × 0.6 mm, 57.6 mm | ✓ | Force white paper §1.2; K133589 |
| columns | 920 | 920 (1840 samples with the flying focal spot) | ✓ | Force white paper §1.2; K133589 |
| column width at the isocentre | 0.561 mm | not published | not confirmed | – |
| source–isocentre, source–detector | 595, 1085.6 mm | not published | not confirmed | – |
| detector shape | cylindrical arc | spherical | **⚠ discrepancy** (geometry only) | Force white paper p. 12 |
| reflector between dexels | not modelled (fill factor 0.9 × 0.9) | "a special reflecting polymer"; width not published | assumed: 0.1 mm of TiO₂-loaded epoxy, which gives the same fill factor | Detectors & Crystals p. 39 |
| anti-scatter grid | not modelled | two-dimensional (the "3D collimator grid"); material and dimensions not published | built here: tungsten, 80 µm walls (Siemens' patent example), 10 mm tall (assumed) | Force white paper pp. 11–13; US 2011/0019801 |
| beam quality | – | half-value layer 7.8 mm Al at 120 kV, isocentre | a check in section 8 | Fong et al. 2024, from Siemens' technical manual |
| FDA clearance | – | K133589 (2014) | ✓ | FDA |

Sources: Siemens Healthineers, [UFC data sheet](https://www.siemens-healthineers.com/computed-tomography/ufc),
[Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf),
[UFC safety data sheet](https://marketing.webassets.siemens-healthineers.com/1800000004195975/23a630dc20e0/DI_CT_datasheet_Safety-UFC_en_1800000004195975.pdf);
Flohr et al., [SOMATOM Force white paper](https://ctm2020.shpci.org/data/CT_SOMATOM-Force_Whitepaper_DualSource_HOOD05162002908728_152104378.pdf);
FDA [K133589](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133589.pdf);
Fong et al., Med. Phys. 2024, [doi:10.1002/mp.16836](https://doi.org/10.1002/mp.16836);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612);
Siemens patent [US 2011/0019801](https://patents.justia.com/patent/20110019801).
"""

# ╔═╡ 5ac2b273-dfac-5d7e-b717-60bb42ac1686
md"""
## Notebook Setup
"""

# ╔═╡ edc8017e-fd91-559c-b8fa-30240d4f641a
import PlutoUI

# ╔═╡ ab6d595c-d31b-595b-8c53-edd04f3f12dc
import BasisMonteCarlo as MC

# ╔═╡ 398c3256-d0b6-5fb2-a912-4a20e1a5c480
import CairoMakie as Mke

# ╔═╡ 015aea26-390f-5615-b59d-a5ff35887bad
import BasisSimulator as BS

# ╔═╡ bfdc3a20-afc8-551d-bc8a-579adc960b43
import Serialization

# ╔═╡ 254434ab-51b5-5f29-9c83-d2bb1aa7bb94
import DelimitedFiles

# ╔═╡ e28b3797-8ff8-5a0a-a57c-d9e2d995032a
PlutoUI.TableOfContents()

# ╔═╡ 36170b1a-5d31-5c5d-839b-769f9b4dd828
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ f41ccaf7-dea7-53b9-a121-831b7efd7e8f
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
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator v0.15.1's table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ d3d0e23d-9e6b-59c5-a623-8ab89195054a
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

# ╔═╡ 981bfa07-0f0d-5879-a3b7-c7cc0991b59c
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

# ╔═╡ 73add7d1-3bf8-5ed8-8ae5-42694781870c
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

# ╔═╡ d38babb6-4cd9-4f26-89ff-477edf8efbbe
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses (the pitch, the reflector,
the crystal's depth, the grid), with a small kit: blocks in 3-D, seen from any direction, shaded
by one light and drawn back to front. The colours are the materials': pale blue UFC ceramic, white
TiO₂ reflector, grey tungsten, dark-blue silicon photodiodes, amber x rays.
"""

# ╔═╡ ac3eb9fc-ef3e-4bc3-9d39-de31b283a49b
MATERIALS = (
    ufc = Mke.RGBf(0.74, 0.84, 0.93),  # the UFC ceramic
    ufc_lit = Mke.RGBf(0.55, 0.9, 0.62),  # ceramic glowing
    reflector = Mke.RGBf(0.97, 0.97, 0.95),  # TiO₂ epoxy: white
    tungsten = Mke.RGBf(0.42, 0.44, 0.5),
    photodiode = Mke.RGBf(0.24, 0.27, 0.55),  # back-illuminated silicon
    cmos = Mke.RGBf(0.2, 0.22, 0.27),  # the CMOS wafer with the ADC
    gold = Mke.RGBf(0.86, 0.66, 0.2),  # vias and stud bumps
    substrate = Mke.RGBf(0.86, 0.83, 0.76),  # ceramics substrate
    board = Mke.RGBf(0.13, 0.4, 0.29),  # circuit board
    water = Mke.RGBf(0.62, 0.78, 0.92),
    beam = Mke.RGBf(0.98, 0.7, 0.25),  # x rays, as in the lab's banner
    light = Mke.RGBf(0.45, 0.9, 0.45),  # UFC's light (it glows green)
    night = Mke.RGBf(0.05, 0.06, 0.08),  # the dark backdrop of the lab's banner
    night_ring = Mke.RGBf(0.11, 0.13, 0.16),
)

# ╔═╡ 67d71709-d954-445c-bec4-a70a8e6674af
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

# ╔═╡ df05ae3e-7c49-4f28-8ced-7f393b17e84e
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

# ╔═╡ a801cc4b-a4ed-40d3-a9dc-e0384d9df317
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ e08d1685-09ba-4e1e-8d49-e4e0bc587c7d
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ a3583f6c-07a9-4209-9b74-1e80267e6a3a
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

# ╔═╡ 0c307760-1a4c-444e-9054-8783e59a1f99
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ b44d674a-9303-4ed0-8ba6-ce98b6540914
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

# ╔═╡ 61fe6fa1-21ad-4f73-82af-5bcb53e2e9fd
"""A dexel array's blocks: ceramic in each cell, reflector walls between (in pieces), from z0 to z1."""
function dexel_blocks(nx, ny, pitch, gap, z0, z1; ceramic = MATERIALS.ufc, lit = nothing)
    blocks = Block[]
    p, s = pitch, gap
    for i in 0:(nx - 1), j in 0:(ny - 1)
        x, y = i * p[1], j * p[2]
        color = lit !== nothing && (i, j) in lit ? MATERIALS.ufc_lit : ceramic
        push!(blocks, Block((x + s / 2, y + s / 2, z0), (x + p[1] - s / 2, y + p[2] - s / 2, z1), color))
        # the reflector on this cell's low-x and low-y sides, and the corner post
        push!(blocks, Block((x - s / 2, y + s / 2, z0), (x + s / 2, y + p[2] - s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((x + s / 2, y - s / 2, z0), (x + p[1] - s / 2, y + s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((x - s / 2, y - s / 2, z0), (x + s / 2, y + s / 2, z1), MATERIALS.reflector))
    end
    # the far walls
    X, Y = nx * p[1], ny * p[2]
    for j in 0:(ny - 1)
        y = j * p[2]
        push!(blocks, Block((X - s / 2, y + s / 2, z0), (X + s / 2, y + p[2] - s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((X - s / 2, y - s / 2, z0), (X + s / 2, y + s / 2, z1), MATERIALS.reflector))
    end
    for i in 0:(nx - 1)
        x = i * p[1]
        push!(blocks, Block((x + s / 2, Y - s / 2, z0), (x + p[1] - s / 2, Y + s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((x - s / 2, Y - s / 2, z0), (x + s / 2, Y + s / 2, z1), MATERIALS.reflector))
    end
    push!(blocks, Block((X - s / 2, Y - s / 2, z0), (X + s / 2, Y + s / 2, z1), MATERIALS.reflector))
    return blocks
end

# ╔═╡ 853df9a6-649b-43b4-9985-c22ab54fbf7b
"""The walls of a 2-D grid over an nx × ny array, in pieces one cell long, from z0 to z1."""
function grid_blocks(nx, ny, pitch, wall, z0, z1; color = MATERIALS.tungsten)
    blocks = Block[]
    p, w = pitch, wall
    for i in 0:nx, j in 0:(ny - 1)
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x - w / 2, y + w / 2, z0), (x + w / 2, y + p[2] - w / 2, z1), color))
    end
    for i in 0:(nx - 1), j in 0:ny
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x + w / 2, y - w / 2, z0), (x + p[1] - w / 2, y + w / 2, z1), color))
    end
    for i in 0:nx, j in 0:ny
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x - w / 2, y - w / 2, z0), (x + w / 2, y + w / 2, z1), color))
    end
    return blocks
end

# ╔═╡ b32e7f50-09f9-4bfa-b83d-db839728ebf1
"""A layer cut into one tile per cell (so the painter's order holds), from z0 to z1."""
function layer_blocks(nx, ny, pitch, z0, z1, color; inset = 0.0)
    return [
        Block(
                (i * pitch[1] + inset, j * pitch[2] + inset, z0),
                ((i + 1) * pitch[1] - inset, (j + 1) * pitch[2] - inset, z1),
                color;
                stroke = inset > 0 ? 0.6 : 0.0,
            )
            for i in 0:(nx - 1) for j in 0:(ny - 1)
    ]
end

# ╔═╡ c97c8bf2-7c91-499d-9666-029a6cfc9b88
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

# ╔═╡ 19b80dcc-92d0-4603-9f0f-7046f2309772
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

# ╔═╡ 610bf587-f0c7-40ad-b75c-b7e26f5f6852
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

# ╔═╡ e47f6b0c-b1ef-4df3-a14e-6bb9d4d60265
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ a2e15617-ca67-4e0e-a785-877170fdad02
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ 7eeebb8c-8e88-4be3-8a3e-06de190d9caa
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

# ╔═╡ f86c4c2f-bcdf-419c-937a-44f624c147c0
"""A vertical bracket at `x` from `y0` to `y1`, with `text` to its left."""
function side_bracket!(ax, x, y0, y1, text; color, fontsize = 16, tick = 0.08)
    Mke.lines!(ax, Mke.Point2f[(x + tick, y0), (x, y0), (x, y1), (x + tick, y1)]; color, linewidth = 2)
    return Mke.text!(ax, x - tick, (y0 + y1) / 2; text, color, fontsize, font = :bold, align = (:right, :center), justification = :right)
end

# ╔═╡ cc8677b7-4293-49ce-a3ed-684f1681cad4
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

# ╔═╡ 0b8077c2-d5ef-4ac5-8f1b-f2fdb10de967
TAGS = (geant4 = Mke.RGBf(0.2, 0.55, 0.32), not_simulated = Mke.RGBf(0.52, 0.52, 0.57))

# ╔═╡ e09df38a-a838-43e6-9fde-b1d8690fc7ed
"""A block drawn faded toward the background (`amount` 0: as is, 1: gone)."""
function faded(b::Block, amount; to = Mke.RGBf(1, 1, 1))
    mix(a, c) = a + amount * (c - a)
    color = Mke.RGBf(mix(b.color.r, to.r), mix(b.color.g, to.g), mix(b.color.b, to.b))
    return Block(; lo = b.lo, hi = b.hi, color, alpha = b.alpha, stroke = b.stroke)
end

# ╔═╡ 6ee68941-6f2d-44f4-a7bb-93a56150ac54
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

# ╔═╡ 3b9b3237-0eb0-4005-9525-8852acfb4254
"""Outline a box (lo, hi) as seen by `cam`: the edge of its silhouette."""
function outline!(ax, cam, lo, hi; color = MATERIALS.beam, linewidth = 3, linestyle = :solid)
    corners = [(x, y, z) for x in (lo[1], hi[1]) for y in (lo[2], hi[2]) for z in (lo[3], hi[3])]
    return Mke.lines!(ax, hull([project(cam, c) for c in corners]); color, linewidth, linestyle)
end

# ╔═╡ 49ad6340-31e5-4ee3-b218-d065efb7dfda
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

# ╔═╡ 410ac026-8a00-4b5a-b890-e8615c11d0a1
"""A Gd atom: nucleus and two shells, at `c` with outer radius `r`."""
function atom!(ax, c, r; color = Mke.RGBf(0.45, 0.47, 0.55))
    θ = range(0, 2π; length = 100)
    for (k, rr) in enumerate((0.55r, r))
        Mke.lines!(ax, c[1] .+ rr .* cos.(θ), c[2] .+ rr .* sin.(θ); color = (color, 0.55), linewidth = 1.5)
    end
    Mke.scatter!(ax, [Mke.Point2f(c)]; color, markersize = 18)
    return nothing
end

# ╔═╡ bb6b4817-3d18-5fc6-8870-fcf6d067c1b7
md"""
## 1. The Scintillator

UFC is gadolinium oxysulfide, Gd₂O₂S, with dopants (praseodymium, cerium) below 0.1 % by mass,
far too little to matter for x rays. Two numbers decide how it stops x rays: its composition and
its density. Siemens' data sheet gives **7.29–7.33 g/cm³**; BasisSimulator v0.15.1 uses 7.44 g/cm³,
which is Geant4's and NIST's stock Gd₂O₂S rather than Siemens' ceramic (⚠ *discrepancy* above).
This notebook builds the detector at **7.31 g/cm³**, and keeps 7.44 beside it to reproduce
the v0.15.1 table.
"""

# ╔═╡ 5b0821f5-5ede-5654-bf41-aae557bf750f
# UFC at the density Siemens documents (7.31 g/cm³)
ufc = MC.DetectorMaterial(:UFC)

# ╔═╡ 26cd0fa9-e926-5418-b39e-828104ebe7c0
# Gd₂O₂S at the density BasisSimulator v0.15.1 uses (7.44 g/cm³)
ufc_basis = MC.DetectorMaterial(:Gd2O2S)

# ╔═╡ 47ba3d48-19e8-5503-8148-1f1ef9940bc9
# Gadolinium's K-shell: its edge and the energy of the x ray it emits (Kα)
gadolinium = only(MC.k_edges(ufc))

# ╔═╡ 2d6fc883-fbf8-5a9e-bec4-bcdffd596d03
md"""
Gadolinium's **K-edge, 50.24 keV**, sits in the middle of every CT spectrum. Below it a photon
cannot free a K-shell electron; just above it, it can, and absorption jumps sevenfold. Every
such absorption leaves the atom to emit, nine times in ten, a **Gd K x ray of about 43 keV**,
which is just below the edge and so is absorbed only weakly: it can travel a millimetre or more
before stopping, which is the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ 599f755d-68f8-559f-ae1a-44e3fec49c79
# The ways a photon can interact, as the figures name them
processes = (
    (kind = :total, label = "All Processes Together (Total)", color = PALETTE.ink),
    (
        kind = :photoelectric,
        label = "Photon Absorbed (Photoelectric)",
        color = PALETTE.electron,
    ),
    (
        kind = :compton,
        label = "Bounces Off, Loses Energy (Compton)",
        color = PALETTE.photon,
    ),
    (
        kind = :rayleigh,
        label = "Bounces Off, Keeps Energy (Rayleigh)",
        color = PALETTE.muted,
    ),
)

# ╔═╡ de91f61c-0c20-5dc1-9fa1-b786127b99ba
mu_energies = 10.0:0.25:150.0

# ╔═╡ 13711387-8deb-5b18-84c1-1fb6dcc5fc07
# Geant4's attenuation of UFC, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(ufc, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ e158b4ca-de06-5340-bf46-ed84dd3f5168
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(ufc, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 57dc760e-1303-51dc-b34e-a7fba0133b0c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How Strongly UFC Stops X Rays (7.31 g/cm³)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Chance of Interacting per mm, μ (1/mm)",
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
        "Gd K-Edge, $(round(gadolinium.edge_keV; digits = 2)) keV";
        offset = (8, 0),
        color = PALETTE.xray,
    )
    note!(
        ax,
        148,
        250,
        "Solid: Geant4 (Used)    Dashed: NIST XCOM";
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

# ╔═╡ e8bfee76-4803-5f45-81ff-2b3f273e3c90
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

# ╔═╡ 7364ef77-5601-548b-ae52-f2587cfc86d9
md"""
### A First Check: Siemens' Own Curve

Siemens publishes the fraction of x rays that **1.4 mm of UFC** absorbs, against energy
(*Detectors & Crystals*, p. 19). Its steps at the K-edge go *up*: it is the probability that a
photon interacts at all, `1 − exp(−μd)`, with no fluorescence escape. That depends only on the
material, its density and its depth, so it checks all three before any simulation runs. The
values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ b6077caf-5639-5dd5-810f-47dbc8afa43b
# Siemens' curve, read from the vector figure (reference_data/siemens_ufc_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_ufc_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ 6e6efe87-7cfc-56af-ae92-eb629fb123ca
# The crystal depth: BasisSimulator v0.15.1's, and the only UFC thickness Siemens quotes
depth_mm = 1.4

# ╔═╡ 2e82befa-95ec-5581-9c0e-a893ef543fc6
# The probability a photon interacts in 1.4 mm of UFC, at Siemens' density and at BasisSimulator v0.15.1's
interaction_curves = (
    documented = [1 - exp(-MC.geant4_attenuation(ufc, E) * depth_mm) for E in mu_energies],
    basis = [1 - exp(-MC.geant4_attenuation(ufc_basis, E) * depth_mm) for E in mu_energies],
)

# ╔═╡ 33144218-4d14-4143-8255-1ee267347282
let
    μ_per_mm(E) = MC.interpolate_linear(collect(mu_energies), mu_geant4[:total], E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 500))
    N = 30
    for (k, E) in enumerate((45.0, 90.0, 140.0))
        μ = μ_per_mm(E)
        stopped = 1 - exp(-μ * depth_mm)
        ax = canvas(fig[1, k]; title = "$(Int(E)) keV: $(round(Int, 100stopped)) % Stopped", titlesize = 21)
        rect!(ax, 0, 0, 1.6, depth_mm, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        # N photons; photon i stops at the depth that the i-th of N quantiles of exp(−μx) gives
        for i in 1:N
            x = 1.6 * (i - 0.5) / N
            depth = -log(1 - (mod(i * 0.618034, 1.0))) / μ
            if depth < depth_mm
                Mke.lines!(ax, [x, x], [depth_mm + 0.55, depth_mm - depth]; color = MATERIALS.beam, linewidth = 2)
                Mke.scatter!(ax, [x], [depth_mm - depth]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (x, depth_mm + 0.55), (x, -0.5); color = (MATERIALS.beam, 0.6), linewidth = 2, head = 10)
            end
        end
        Mke.text!(ax, 1.65, depth_mm / 2; text = "1.4 mm\nof UFC", align = (:left, :center), fontsize = 15, color = PALETTE.slab_edge)
        Mke.limits!(ax, -0.05, 2.15, -0.62, depth_mm + 0.65)
    end
    Mke.Label(
        fig[2, 1:3],
        "Each Line Is a Photon from Above: Red Dots Are Where It Interacts; Arrows Pass Straight Through. The Fraction Stopped Is 1 − exp(−μ · 1.4 mm).";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 070fc321-f721-51dc-bbb7-aebc18a576bf
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Fraction of Photons 1.4 mm of UFC Absorbs",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons Absorbed",
        xticks = 20:20:150,
        limits = ((15, 152), (0.45, 1.04)),
    )
    Mke.lines!(
        ax,
        mu_energies,
        interaction_curves.documented;
        color = PALETTE.photon,
        linewidth = 3.5,
        label = "Geant4, 7.31 g/cm³ (Siemens' Density)",
    )
    Mke.lines!(
        ax,
        mu_energies,
        interaction_curves.basis;
        color = PALETTE.reference,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 7.44 g/cm³ (BasisSimulator v0.15.1)",
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
        label = "Siemens' Figure (± 0.025, Its Reading Accuracy)",
    )
    note!(
        ax,
        gadolinium.edge_keV,
        0.88,
        "Gd K-Edge:\nAbsorption Jumps";
        align = (:right, :center),
        offset = (-12, 0),
        color = PALETTE.xray,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 984509ba-dcb0-5de2-af45-34c1ee30eb3f
let
    ours = [MC.interpolate_linear(collect(mu_energies), interaction_curves.documented, E) for E in siemens_curve.energy_keV]
    difference = siemens_curve.absorption .- ours
    low = siemens_curve.energy_keV .<= 90
    high = .!low

    # The depth that would give Siemens' value at each energy above 90 keV
    implied_depth = [
        -log(1 - a) / MC.geant4_attenuation(ufc, E)
            for (E, a) in zip(siemens_curve.energy_keV[high], siemens_curve.absorption[high])
    ]
    Markdown.parse(
        """
        Up to 90 keV Siemens' figure and Geant4 at 1.4 mm and 7.31 g/cm³ agree to within
        **$(round(maximum(abs, difference[low]); digits = 3))**, inside the figure's reading accuracy: the material and
        density are right. Above 90 keV the figure falls below the 1.4 mm curve, by up to
        **$(round(maximum(abs, difference[high]); digits = 3))** at $(Int(siemens_curve.energy_keV[high][argmax(abs.(difference[high]))])) keV; read
        literally, it corresponds to a layer $(round(minimum(implied_depth); digits = 2))–$(round(maximum(implied_depth); digits = 2)) mm deep. Either way the
        published 1.4 mm is, if anything, generous: Siemens' own figure gives no support to a deeper crystal.
        """
    )
end

# ╔═╡ 624c50f1-16e9-537a-80a2-88b0ec6d5bf1
md"""
## 2. One Dexel and Its Neighbours

The array is sawn from one ceramic wafer in two directions and the cuts are filled with
reflector, so a dexel is one block of ceramic with reflector on its four sides. Its pitch at the
detector face is the Force's dexel at the isocentre, 0.561 mm across the fan by 0.6 mm along z,
magnified by the source-to-detector over the source-to-isocentre distance, 1085.6 / 595. The
reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy, the usual CT reflector, is
assumed.
"""

# ╔═╡ b87d28f5-35b8-59c9-9eaa-56929df2e1a5
# The SOMATOM Force's geometry, as BasisSimulator v0.15.1 has it
force = (
    source_to_isocenter_mm = 595.0,
    source_to_detector_mm = 1085.6,
    columns = 920,
    rows = 96,
    column_at_isocenter_mm = 0.561,
    row_at_isocenter_mm = 0.6,
)

# ╔═╡ 0d218946-d0ed-50b8-a347-b76f53bd8a9f
# The dexel pitch at the detector face (across the fan, along z), in mm
pitch_mm = (force.column_at_isocenter_mm, force.row_at_isocenter_mm) .*
    (force.source_to_detector_mm / force.source_to_isocenter_mm)

# ╔═╡ 86d8a811-3df1-5cec-90db-e3d1161f674a
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ 7de77732-a5e4-4d99-959a-1110568c6307
"""Side view of three dexels (x across, depth down as negative y), at x offsets; returns nothing."""
function side_dexels!(ax; n = 3, lit = nothing, depth = depth_mm, ceramic = MATERIALS.ufc)
    p, s = pitch_mm[1], septum_mm
    half = (n - 1) / 2
    rect!(ax, -(half + 0.5) * p, -depth, n * p, depth, MATERIALS.reflector; strokecolor = (:black, 0.3), strokewidth = 1)
    for i in 0:(n - 1)
        x = (i - half) * p
        rect!(ax, x - (p - s) / 2, -depth, p - s, depth, lit !== nothing && i == lit ? Mke.RGBf(0.82, 0.95, 0.84) : ceramic)
    end
    return nothing
end

# ╔═╡ 2597aa0b-31a5-5e2f-acea-8a08b89e189f
# The block of ceramic each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ 68bb39d1-7bd1-5732-8d9e-3143dbd4e31b
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 8068a25b-4729-577e-8b7a-8198ea882d7e
let
    fill = crystal_mm ./ pitch_mm
    Markdown.parse(
        """
        The dexel pitch at the face is **$(round(pitch_mm[1]; digits = 3)) × $(round(pitch_mm[2]; digits = 3)) mm**, and
        the ceramic block inside it $(round(crystal_mm[1]; digits = 3)) × $(round(crystal_mm[2]; digits = 3)) mm: the crystal
        covers $(round(fill[1]; digits = 3)) × $(round(fill[2]; digits = 3)) of the face, the fill factor of 0.9 × 0.9 that BasisSimulator v0.15.1 uses.
        The assumed reflector and that fill factor are the same detector.
        """
    )
end

# ╔═╡ a7e2f1ed-a64b-5798-962b-977f40251001
# The Geant4 detector: a slab of ceramic `depth` deep, cut into 13 × 13 dexels by reflector
# septa (ample for the x rays that leave a dexel), optionally under an anti-scatter grid
function dexel_array(depth; material = ufc, grid = nothing)
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

# ╔═╡ 8f7fd8fb-bad6-5f1b-a112-9244837b96b4
# Photons landing uniformly on one dexel's ceramic, and on its whole cell (ceramic and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ 135302df-8c55-5616-b4b0-c7a92e8bd37f
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 1d0e1508-6800-5211-9822-7bcaefdc76f6
# Each dexel collects the energy left in its own block of ceramic (the reflector keeps the light
# in); the dexel the photons land on and its eight neighbours are recorded
readout = MC.DexelEnergyReadout(
    MC.DexelReadout(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        neighborhood = 3,
        cloud_sigma_um = 0.01,
        transport = nothing,  # a scintillator: each dexel keeps the energy left in its own ceramic
    ),
)

# ╔═╡ 00b453d7-b568-5e07-97f4-b5e7fea4b6eb
md"""
### What a Photon Does in the Ceramic

`record_tracks` follows a few hundred 60 keV photons landing on the central dexel, every step of
every particle. Two of them are drawn: one whose Gd x ray is absorbed **in the next dexel**, so
its energy is split between two photodiodes, and one whose Gd x ray **leaves through the front
face**, so its 43 keV are lost to every dexel. Both happen often, because the Gd x ray is so
weakly absorbed.
"""

# ╔═╡ 3d3d9f03-1438-5fc6-b252-f6a016530cd6
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    60.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ 5a19b142-fa62-5a20-8c3e-fd1b2caba4ea
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

# ╔═╡ 47825cfd-bde9-5b63-82f1-846000c79a0c
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
        xlabel = "X, Across the Fan (mm)",
        ylabel = "Depth (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.55p[1], 1.55p[1]), (top, d + 0.55)),
    )
    axa = Mke.Axis(fig[1, 1]; title = "(a) The Gd X Ray Stops in the Next Dexel", common...)
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Gd X Ray Leaves Through the Front",
        common...,
    )
    panel!(axa, followed.next_door)
    panel!(axb, followed.front)
    Mke.Label(
        fig[2, 1:2],
        "Blue: the 60 keV Photon     Orange: Its Gd X Ray     Red Dots: Where They Are Absorbed     Green: the Energy Each Dexel Keeps";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 207da70b-5e07-4fc4-8dfe-de974aec8542
md"""
### Why Don't the Grid's Walls Go Down Through the Ceramic?

Two things can leak from one dexel into the next, and they are stopped (or not) by different parts:

- **Light.** The ceramic glows where the x ray stopped; light reaching the next dexel's photodiode
  would blur the image (*optical crosstalk*). The **white TiO₂ reflector** in the saw cuts stops
  it: light bounces back into its own dexel. Geant4 does not follow the light; the reflector is
  assumed perfect, which is what it is for.
- **X rays.** A Gd x ray (43 keV) or a Compton-scattered photon crosses 0.1 mm of light epoxy
  almost freely and can stop in the neighbour (*x-ray crosstalk*, the cousin of charge sharing in
  photon-counting detectors). That is the "next door" photon above, and Geant4 follows it.

Tungsten walls through the ceramic would stop the x rays, but they would widen the cuts (less
ceramic per dexel, so fewer photons stopped for the same dose), add tungsten's own x rays (59 and
67 keV) inside the ceramic, and have to be built into a sawn ceramic wafer. And in a scan the
x-ray crosstalk mostly cancels: a dexel gets back from its neighbours what it loses to them
(section 4). What it does cost, a little sharpness and a little noise (the Swank factor), is
measured here. So the grid stands in front, against the scatter from the patient, and only the
reflector goes between the dexels.
"""

# ╔═╡ 9136f582-c77e-5133-9eb9-1c5c62370702
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is not one number but a
spread: the full photon energy when everything stays in the dexel, the energy minus 43 keV when
the Gd x ray leaves, a little when the photon only Compton-scatters once and leaves. Three
energies show the three regimes: 45 keV (below the K-edge), 60 keV (just above it) and 120 keV.
"""

# ╔═╡ e5779ca2-7b7b-5cac-b9ea-f5246af64ac3
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

# ╔═╡ e81d468a-4e95-51a8-b3f8-417d0783c6d9
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ 086a9000-05cf-54f7-bd40-160ea222808b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 520))
    kα = gadolinium.kalpha_keV
    for (k, E) in enumerate((45.0, 60.0, 120.0))
        h = histogram(own_signal[E], E + 4)
        ax = Mke.Axis(
            fig[1, k];
            title = "$(Int(E)) keV Photons",
            xlabel = "Energy the Dexel Collects (keV)",
            ylabel = k == 1 ? "Probability per keV" : "",
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
            "All of It";
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
                "Gd X Ray Lost:\nE − $(round(kα; digits = 1)) keV";
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
                "Compton Scatters\nThat Left";
                align = (:center, :center),
                color = PALETTE.photon,
                fontsize = 14,
            )
        end
    end
    fig
end

# ╔═╡ d8c74acc-320a-5454-8c8e-4fc993879dca
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the ceramic or
leaves it: as the Gd x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of ceramic (no reflector), so that whatever
leaves the ceramic leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ fae36d57-1538-5767-8a86-89f4d674ee92
budget_energies = 15.0:5.0:140.0

# ╔═╡ fbd77aa1-7f9a-5f31-8fa5-fe8bb28c1b61
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(ufc, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ b03a2cc1-a6fa-495b-99aa-fdbed493bd66
let
    Mke.set_theme!(THEME)
    d = depth_mm
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1])
    rect!(ax, -6, -d, 12, d, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 0, -d - 0.12; text = "One Continuous Slab of UFC, $(d) mm Deep (No Reflector, So What Leaves Is Gone)", fontsize = 15, color = PALETTE.slab_edge, align = (:center, :top))
    xs = (-4.5, -1.5, 1.5, 4.5)
    for x in xs
        photon!(ax, (x, 1.6), (x, x == 4.5 ? -d - 1.4 : -0.6); cycles = 5, amplitude = 0.012, color = MATERIALS.beam)
    end
    for x in xs[1:3]
        Mke.scatter!(ax, [Mke.Point2f(x, -0.6)]; marker = :star5, markersize = 24, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)
    end
    photon!(ax, (-1.5, -0.6), (-2.6, 1.3); cycles = 3, amplitude = 0.04, color = PALETTE.xray)
    photon!(ax, (1.5, -0.6), (2.7, -d - 1.3); cycles = 3, amplitude = 0.04, color = PALETTE.photon)
    labels = (
        (-4.5, "Stays in\nthe Ceramic", PALETTE.kept),
        (-1.5, "Leaves as\na Gd X Ray", PALETTE.xray),
        (1.5, "Leaves as a\nScattered Photon", PALETTE.photon),
        (4.5, "Passes Straight\nThrough", PALETTE.muted),
    )
    for (x, text, color) in labels
        Mke.text!(ax, x, 1.75; text, color, fontsize = 18, font = :bold, align = (:center, :bottom))
    end
    Mke.limits!(ax, -6.3, 6.3, -d - 1.6, 2.75)
    fig
end

# ╔═╡ 709be125-f9e1-5fd9-ada9-0ee542722e10
let
    Mke.set_theme!(THEME)
    E = collect(budget_energies)
    parts = (
        (
            values = [b.absorbed for b in budgets],
            label = "Stays in the Ceramic (the Signal)",
            color = PALETTE.kept,
        ),
        (
            values = [b.fluorescence for b in budgets],
            label = "Leaves as a Gd X Ray",
            color = PALETTE.xray,
        ),
        (
            values = [b.scattered for b in budgets],
            label = "Leaves as a Scattered Photon",
            color = PALETTE.photon,
        ),
        (
            values = [b.transmitted for b in budgets],
            label = "Passes Straight Through",
            color = PALETTE.muted,
        ),
    )
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Where the Photon's Energy Goes: $(depth_mm) mm of UFC",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of the Photon's Energy",
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

# ╔═╡ 96411072-2914-50b3-90a8-9a155e951b38
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

# ╔═╡ 1aff4f4a-5c08-5668-a3a9-408158142c6a
md"""
## 4. The Efficiency and Noise of One Dexel

The full simulation: at every whole keV from 1 to 140 (the grid of BasisSimulator's tables),
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

# ╔═╡ c2344ff9-4d61-521d-bde9-fe85b774d1bd
energies = 1.0:1.0:140.0

# ╔═╡ 473d2874-dedb-57bc-a509-60c0203d1761
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 0a89ea01-6e3a-53c1-b61a-cb5387ed2854
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ a5da9355-cf65-5dbf-98e7-d6610831b32c
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1400, 600))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Efficiency, $(depth_mm) mm of UFC",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
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
        label = "Photon Interacts at All",
    )
    Mke.lines!(
        axa,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3.5,
        label = "Flat Field (What a Scan Sees)",
    )
    Mke.lines!(
        axa,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2.5,
        linestyle = :dash,
        label = "Kept by the Dexel It Landed On",
    )
    note!(
        axa,
        gadolinium.edge_keV,
        0.75,
        "Gd K-Edge";
        align = (:right, :center),
        offset = (-8, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.vlines!(axa, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axa; position = :lb, margin = (10, 10, 40, 10))

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Noise: the Swank Factor and DQE(0)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Swank Factor I,  DQE(0)",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        axb,
        E,
        efficiency.swank;
        color = PALETTE.electron,
        linewidth = 3,
        label = "Swank Factor I (Spread of the Signal)",
    )
    Mke.lines!(
        axb,
        E,
        efficiency.dqe;
        color = PALETTE.photon,
        linewidth = 3,
        label = "DQE(0) = Interaction × I",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :lb)
    fig
end

# ╔═╡ 2783d5ea-67a1-5806-9769-6ba444eac9e6
let
    at(E) = findfirst(==(E), collect(energies))
    k60, k140 = at(60.0), at(140.0)
    beer = 1 - exp(-MC.geant4_attenuation(ufc, 140.0) * depth_mm)
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

# ╔═╡ 1e7c8688-b175-523d-8b2b-c3db2965e66b
md"""
### A Check Against Published Monte Carlo

Boone et al. (Med. Phys. 1999) followed x-ray fluorescence in thin Gd₂O₂S screens with the TART 98
code: a pencil beam at normal incidence on a screen of Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S, the binder not counted), the energy tallied in a circle 0.1 mm across the
beam ("absorbed as primary") and in rings out to 4.5 mm ("lateral"), and ideal sensors in front of
the screen and behind it. At 55 keV they report (p. 908) how the incident energy divides. The same
screen and the same tallies in Geant4:
"""

# ╔═╡ ccd1b408-75f5-55b1-a644-2249e8800179
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

# ╔═╡ 6b9cb578-7aef-48d4-a9b3-60077533fc6d
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1]; title = "Side View (Not to Scale)", titlesize = 20)
    rect!(ax, -3, -0.12, 6, 0.24, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 3.05, 0; text = "Gd₂O₂S Screen,\n90 mg/cm²", align = (:left, :center), fontsize = 15, color = PALETTE.slab_edge)
    Mke.lines!(ax, [-3, 3], [1.1, 1.1]; color = PALETTE.photon, linewidth = 3, linestyle = :dash)
    Mke.lines!(ax, [-3, 3], [-1.1, -1.1]; color = PALETTE.photon, linewidth = 3, linestyle = :dash)
    Mke.text!(ax, -3, 1.2; text = "Ideal Sensor: Energy Carried Back Toward the Tube", fontsize = 15, color = PALETTE.photon, align = (:left, :bottom))
    Mke.text!(ax, -3, -1.2; text = "Ideal Sensor: Energy Carried Forward", fontsize = 15, color = PALETTE.photon, align = (:left, :top))
    photon!(ax, (0, 2.0), (0, 0.0); cycles = 6, amplitude = 0.03, color = MATERIALS.beam)
    Mke.text!(ax, 0.1, 1.75; text = "55 keV Pencil Beam", fontsize = 15, color = Mke.RGBf(0.8, 0.5, 0.05), align = (:left, :center))
    photon!(ax, (0, 0), (-1.3, 1.05); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 10)
    photon!(ax, (0, 0), (1.4, -1.05); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 10)
    photon!(ax, (0, 0), (0.9, 0.0); cycles = 2, amplitude = 0.08, color = PALETTE.xray, head = 10)
    Mke.limits!(ax, -3.2, 4.6, -1.7, 2.3)

    ax2 = canvas(fig[1, 2]; title = "Seen from the Beam: the Tallies", titlesize = 20)
    θ = range(0, 2π; length = 120)
    Mke.poly!(ax2, Mke.Point2f.(4.5 .* cos.(θ), 4.5 .* sin.(θ)); color = (MATERIALS.ufc, 0.6), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.poly!(ax2, Mke.Point2f.(0.45 .* cos.(θ), 0.45 .* sin.(θ)); color = PALETTE.kept)
    Mke.text!(ax2, 0, 0.6; text = "Within 0.1 mm:\n\"Absorbed as Primary\"", align = (:center, :bottom), fontsize = 15, color = PALETTE.kept)
    Mke.text!(ax2, 0, -3.3; text = "0.1 to 4.5 mm: \"Lateral\"\n(Reabsorbed Fluorescence)", align = (:center, :center), fontsize = 15, color = PALETTE.slab_edge)
    Mke.text!(ax2, 4.6, 3.6; text = "(Circle Enlarged)", align = (:left, :center), fontsize = 13, color = PALETTE.muted)
    Mke.limits!(ax2, -5, 7, -5, 5)
    fig
end

# ╔═╡ faa33039-ae00-56d9-ae15-e2b2e31fa9a3
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

# ╔═╡ d04a5179-567e-5af2-8d05-57a4f1934c55
md"""
## 5. Compared with BasisSimulator v0.15.1's Table

The v0.15.1 table is an energy-absorption efficiency (its K-edge step goes down, as a
fluorescence escape makes it), and BasisSimulator v0.15.1 documents the crystal as 1.4 mm deep. But at
high energy the table is above what 1.4 mm of UFC can absorb at all: a photon that never
interacts leaves no energy, so the interaction probability is a ceiling on the efficiency.
"""

# ╔═╡ 8aba61c1-6b2d-5111-83b8-4a85a8054edc
# BasisSimulator v0.15.1's table (MCNP, Hamidreza Khodajou-Chokami), as BasisMonteCarlo keeps it
# for comparison: later BasisSimulator releases use this notebook's table instead
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc

# ╔═╡ 68de6046-7b40-4d32-96af-a2432fbdf672
# Is BasisMonteCarlo's copy the table the installed BasisSimulator uses? Up to v0.15.1 it must be,
# exactly; from v0.16.0 BasisSimulator uses this notebook's table, and the copy stands alone
let
    installed = pkgversion(BS)
    same = isdefined(BS, :UFC_MC_EFFICIENCY_LUT) &&
        BS.UFC_MC_EFFICIENCY_LUT.energies == table.energies &&
        BS.UFC_MC_EFFICIENCY_LUT.efficiency == table.efficiency
    message = if installed <= v"0.15.1"
        same ?
            "✓ The installed BasisSimulator, v$(installed), uses exactly this table." :
            "✗ **The installed BasisSimulator, v$(installed), has a different table from BasisMonteCarlo's copy of v0.15.1's.** Check `BASISSIMULATOR_V0_15_1_EFFICIENCY` against BasisSimulator v0.15.1."
    else
        "The installed BasisSimulator is v$(installed), which no longer uses this table. The comparisons below are with BasisSimulator v0.15.1's table, which BasisMonteCarlo keeps (`BASISSIMULATOR_V0_15_1_EFFICIENCY`): they do not change with BasisSimulator's version."
    end
    Markdown.parse(message)
end

# ╔═╡ 5cd42bd3-b86a-5ee8-90de-ffdf30d6b107
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ 251a2c3a-47e5-5256-9c78-ad2d15c53226
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(ufc_basis, e) * depth_mm) for e in E]
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "BasisSimulator v0.15.1's Table Against 1.4 mm of UFC",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.06)),
    )
    Mke.band!(ax, E, ceiling, fill(1.06, length(E)); color = (PALETTE.electron, 0.08))
    Mke.lines!(
        ax,
        E,
        ceiling;
        color = PALETTE.electron,
        linewidth = 2,
        linestyle = :dot,
        label = "Ceiling: 1.4 mm Interacts at All (7.44 g/cm³)",
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 3.5,
        label = "BasisSimulator v0.15.1 (MCNP)",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, 1.4 mm, Flat Field",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 1.4 mm, One Dexel",
    )
    note!(
        ax,
        100,
        1.0,
        "Impossible for 1.4 mm";
        align = (:center, :center),
        color = PALETTE.electron,
        fontsize = 15,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 8f182220-5099-5d7d-b26d-ec7650a032b6
let
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(ufc_basis, e) * depth_mm) for e in E]
    over = E[table_at.(E) .> ceiling .+ 0.01]
    Markdown.parse(
        """
        **⚠ Discrepancy.** From $(Int(first(over))) keV up, the v0.15.1 table is above the fraction of photons 1.4 mm
        of Gd₂O₂S stops at all, even at its own density (at 140 keV: $(round(table_at(140.0); digits = 3)) against a ceiling of
        $(round(last(ceiling); digits = 3))). The table cannot describe a 1.4 mm crystal. Which crystal does it describe?
        """
    )
end

# ╔═╡ d4d258d8-a0ff-5e91-bb06-e18dde132184
md"""
### Which Crystal Did the Table Describe?

The table's shape fixes how it was made. A pencil beam into the centre of one dexel, tallied in
that dexel, gives its deep step at the K-edge (with a uniform beam or a flat field the step is
shallower); the fall-off at high energy then fixes the depth. So: pencil beams at 60–140 keV into
dexels 1.0 to 3.4 mm deep, at BasisSimulator v0.15.1's density, compared with the table, with one
constant factor for the table's flat 1 % loss at low energy.
"""

# ╔═╡ b71c3c63-f028-530c-b506-7c2f44e4e55c
scan_depths = 1.0:0.2:3.4

# ╔═╡ 9f38d967-6625-5bdb-bf78-a28648932329
scan_energies = 60.0:5.0:140.0

# ╔═╡ 81fb4130-1d0e-5c57-b642-866f51f38a19
# One column per depth, one row per energy
pencil_scan = reshape(
    MC.simulate_sweep(
        [
            (dexel_array(depth; material = ufc_basis), E)
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

# ╔═╡ 42a87db7-35b4-5bd0-a3ad-8fafe9a0fa49
# How far one depth's pencil-beam efficiencies are from the table, after the best constant factor
function misfit(runs, target)
    ours = MC.dexel_efficiency.(runs)
    scale = sum(ours .* target) / sum(ours .^ 2)
    rms = sqrt(sum(abs2, scale .* ours .- target) / length(target))
    return (scale = scale, rms = rms)
end

# ╔═╡ 4a8900c3-6ec6-558c-b0a3-d8c9b26dc0fe
fit = let
    target = table_at.(scan_energies)
    misfits = [misfit(pencil_scan[:, k], target) for k in eachindex(scan_depths)]
    best = argmin([m.rms for m in misfits])
    (depth_mm = scan_depths[best], scale = misfits[best].scale, misfits = misfits)
end

# ╔═╡ 38c8c86b-d27a-5ef5-87ee-05ef7abc55ee
let
    Mke.set_theme!(THEME)
    rms = [m.rms for m in fit.misfits]
    fig = Mke.Figure(size = (1200, 500))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Which Crystal Depth Reproduces the v0.15.1 Table?",
        xlabel = "Crystal Depth in the Simulation (mm)",
        ylabel = "RMS Difference from the Table\n(60–140 keV)",
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
        "Best: $(fit.depth_mm) mm";
        align = (:center, :top),
        offset = (0, -12),
        color = PALETTE.photon,
        fontsize = 16,
    )
    note!(
        ax,
        depth_mm,
        0.3,
        "BasisSimulator v0.15.1 Documents 1.4 mm";
        align = (:left, :center),
        offset = (8, 0),
        color = PALETTE.reference,
        fontsize = 15,
    )
    fig
end

# ╔═╡ d36811c9-01e7-5107-8359-46bc751cad2c
# The table's recipe at the depth that fits, at every energy
reproduction = MC.simulate_sweep(
    [(dexel_array(fit.depth_mm; material = ufc_basis), E) for E in energies];
    n = 100_000,
    beam = MC.PencilBeam(; start_mm = 12.0),
    seed = 15,
    readout,
)

# ╔═╡ b7f6dca9-d404-5a4c-a45b-32ba0b461002
reproduced = fit.scale .* MC.dexel_efficiency.(reproduction)

# ╔═╡ ea31ed85-c69c-5e35-9d41-5927c317bee3
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Table, and Geant4's Pencil Beam into One Dexel at $(fit.depth_mm) mm (× $(round(fit.scale; digits = 3)))",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 5,
        label = "BasisSimulator v0.15.1 (MCNP)",
    )
    Mke.lines!(
        ax,
        E,
        reproduced;
        color = PALETTE.photon,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, Reproduced",
    )
    Mke.axislegend(ax; position = :lb)
    axr = Mke.Axis(
        fig[2, 1];
        xlabel = "Photon Energy (keV)",
        ylabel = "Geant4 − Table",
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

# ╔═╡ b8473053-0084-5d1c-b467-2e3ca8edaa07
let
    E = collect(energies)
    difference = reproduced .- table_at.(E)
    typical = sum(abs, difference) / length(difference)
    xcist = fit.depth_mm == 3.0 ?
        " 3.0 mm is XCIST's default detector depth, the source of several of BasisSimulator's scanner defaults." : ""
    Markdown.parse(
        """
        The table is reproduced to **$(round(typical; digits = 4))** on average (largest difference
        $(round(maximum(abs, difference); digits = 3)), at $(Int(E[argmax(abs.(difference))])) keV) by a pencil beam into the centre of one
        dexel, tallied in that dexel, in Gd₂O₂S at 7.44 g/cm³ **$(fit.depth_mm) mm deep**, times $(round(fit.scale; digits = 3)).
        **⚠ Discrepancy:** BasisSimulator v0.15.1 labels this table as the Force's 1.4 mm detector.$(xcist) The table also
        misses what a CT scan sees in two ways: one dexel under a pencil beam, rather than the flat field (section 4),
        and no fall-off from photons landing near the dexel's edge.
        """
    )
end

# ╔═╡ 71a9bf29-a1f9-57eb-ab25-5be4792b7998
md"""
## 6. The Anti-Scatter Grid

The Force's detectors are the first Siemens detectors with a **two-dimensional** anti-scatter
grid: walls between columns and between rows, all pointing at the focal spot (with the spherical
detector, Siemens' "3D collimator grid"), which the white paper credits with 2–3 times less
scatter than a one-directional grid. Its material and dimensions are not published. Built here:
**tungsten walls 80 µm thick** (the preferred value of Siemens' patent for laser-sintered grids,
US 2011/0019801), **10 mm tall** (assumed), one cell per dexel, the walls standing on the
reflector. 80 µm walls on 100 µm of reflector shade no ceramic.
"""

# ╔═╡ 76bf4fed-52bd-57c9-a3a1-c7b001617d3f
grid = MC.Collimator(;
    height_mm = 10.0,
    wall_mm = 0.08,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :both,
    cells = 13,
)

# ╔═╡ e2e24ca7-420b-42c3-b46d-f7c4d94a2191
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1400, 760))
    ax = canvas(fig[1, 1])

    # Layers, side view (x across, y up); heights in mm, the grid cut short
    y_board, y_sub, y_cmos, y_pd = -1.55, -1.2, -0.62, -0.16
    rect!(ax, -1.5p, y_board, 3p, 0.3, MATERIALS.board)
    rect!(ax, -1.5p, y_sub, 3p, 0.34, MATERIALS.substrate)
    rect!(ax, -1.5p, y_cmos, 3p, 0.42, MATERIALS.cmos)
    for i in -1:1
        x = i * p
        rect!(ax, x - (p - s) / 2, y_pd, p - s, 0.14, MATERIALS.photodiode)
        rect!(ax, x - 0.05, y_sub + 0.34, 0.1, 0.24, MATERIALS.gold)  # the via and stud bump
        rect!(ax, x - 0.22, y_cmos + 0.1, 0.44, 0.2, Mke.RGBf(0.32, 0.35, 0.42))  # the ADC's circuit
        Mke.text!(ax, x, y_cmos + 0.2; text = "ADC", color = :white, fontsize = 13, align = (:center, :center), font = :bold)
    end
    # ceramic and reflector
    rect!(ax, -1.5p, 0, 3p, d, MATERIALS.reflector; strokecolor = (:black, 0.25), strokewidth = 1)
    for i in -1:1
        rect!(ax, i * p - (p - s) / 2, 0, p - s, d, i == 0 ? Mke.RGBf(0.8, 0.95, 0.82) : MATERIALS.ufc)
    end
    # grid walls, cut short (wavy break)
    for i in -1:2
        x = (i - 0.5) * p
        rect!(ax, x - grid.wall_mm / 2, d, grid.wall_mm, 1.2, MATERIALS.tungsten)
    end

    # (1) the x ray, (2) stops in the middle dexel, (3) light, (4) photodiode, (5) the ADC
    hit = Mke.Point2f(0.12, 0.62)
    photon!(ax, (0.12, d + 2.6), hit; cycles = 9, amplitude = 0.02, color = MATERIALS.beam, linewidth = 3.5)
    Mke.scatter!(ax, [hit]; marker = :star5, markersize = 30, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.5)
    # light rays: straight down, and bouncing off the reflector walls
    half = (p - s) / 2
    for (dx, bounce) in ((0.0, false), (-1.0, true), (1.0, true), (-0.35, false), (0.4, false))
        if bounce
            wall = Mke.Point2f(dx < 0 ? -half : half, hit[2] + 0.28)
            bottom = Mke.Point2f(dx < 0 ? -0.15 : 0.32, 0.0)
            Mke.lines!(ax, Mke.Point2f[hit, wall, bottom]; color = (MATERIALS.light, 0.95), linewidth = 2.2)
        else
            Mke.lines!(ax, Mke.Point2f[hit, Mke.Point2f(hit[1] + dx * 0.5, 0.0)]; color = (MATERIALS.light, 0.95), linewidth = 2.2)
        end
    end
    # the number out
    arrow!(ax, (0.0, y_cmos), (0.0, y_board - 0.35); color = PALETTE.ink, linewidth = 2, head = 12)
    Mke.text!(ax, 0.0, y_board - 0.45; text = "One Number per Dexel per Reading", align = (:center, :top), fontsize = 16, color = PALETTE.ink)

    # the five steps, labelled on the right
    entries = [
        (anchor = (0.12, d + 1.9), title = "1  An X Ray Arrives", detail = "Down Its Channel in the Grid", color = Mke.RGBf(0.85, 0.5, 0.05), tag = nothing),
        (anchor = hit, title = "2  The Ceramic Stops It", detail = "Its Energy Is Left in the Dexel", color = Mke.RGBf(0.7, 0.55, 0.05), tag = nothing),
        (anchor = (half - 0.02, hit[2] + 0.28), title = "3  The Ceramic Glows", detail = "Light ∝ Energy Left; the White Reflector Keeps It In", color = Mke.RGBf(0.2, 0.6, 0.25), tag = nothing),
        (anchor = (0.4, y_pd + 0.07), title = "4  The Photodiode Collects the Light", detail = "Light → Charge, Summed over the Reading", color = MATERIALS.photodiode, tag = nothing),
        (anchor = (0.22, y_cmos + 0.2), title = "5  The ADC Digitizes the Charge", detail = "In the Same Chip, Right Underneath", color = PALETTE.ink, tag = nothing),
    ]
    xr = 1.5p + 0.75
    labels_right!(ax, entries, xr; spacing = 0.62)
    Mke.text!(ax, 1.5p + 0.05, d + 1.2; text = "Grid Walls\n(Cut Short)", fontsize = 14, color = PALETTE.muted, align = (:left, :top))

    # what Geant4 simulates, and what is assumed
    xl = -1.5p - 0.15
    side_bracket!(ax, xl, 0.0, d + 1.2, "Geant4\nSimulates"; color = TAGS.geant4)
    side_bracket!(ax, xl, y_board, -0.04, "Not Simulated:\nSignal ∝ Energy\nLeft in the Ceramic"; color = TAGS.not_simulated)
    Mke.limits!(ax, -1.5p - 1.75, xr + 3.3, y_board - 0.75, d + 2.75)
    fig
end

# ╔═╡ 9b48b777-a6c9-4ca9-94a6-dfbd06e96922
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1700, 560))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        ("Pencil Beam", "Into the Centre of One Dexel:\nHow the v0.15.1 Table Was Made", :pencil),
        ("Uniform on One Dexel", "Over Its Ceramic; Every Dexel\nRead Out (Sections 3–5)", :ceramic),
        ("Uniform on One Cell", "Ceramic and Its Reflector,\nUnder the Grid (Section 6)", :cell),
        ("Flat Field", "Every Dexel Lit, as in a Scan:\nWhat Each Dexel Collects", :flat),
        ("From the Focal Spot", "The Whole Arc, Real Angles;\nClinical Spectra (Sections 7, 10)", :focal),
    )
    for (k, (title, detail, kind)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        blocks = dexel_blocks(3, 3, p, s, 0.0, d)
        if kind === :cell
            append!(blocks, grid_blocks(3, 3, p, grid.wall_mm, d, d + 1.6))
        end
        top = kind === :cell ? d + 1.6 : d
        extras = Any[]
        # the beam: amber columns (translucent boxes) or lines, drawn after the blocks
        if kind === :pencil
            a, b = project(cam, (1.5p[1], 1.5p[2], d + 4.5)), project(cam, (1.5p[1], 1.5p[2], d))
            push!(extras, (1.0e3, ax -> arrow!(ax, a, b; color = amber, linewidth = 4, head = 14)))
        elseif kind in (:ceramic, :cell, :flat)
            lo = kind === :ceramic ? (p[1] + s / 2, p[2] + s / 2) : kind === :cell ? (p[1], p[2]) : (0.0, 0.0)
            hi = kind === :ceramic ? (2p[1] - s / 2, 2p[2] - s / 2) : kind === :cell ? (2p[1], 2p[2]) : (3p[1], 3p[2])
            beam = Block((lo..., top), (hi..., top + 4.0), amber; alpha = 0.28, stroke = 0.0)
            push!(extras, (1.0e3, ax -> draw_blocks!(ax, cam, [beam]; edge_alpha = 0.0)))
        else
            focal = (1.5p[1], 1.5p[2], d + 9.0)
            for (x, y) in ((0.2, 0.2), (2.8, 0.2), (0.2, 2.8), (2.8, 2.8), (1.5, 1.5), (0.9, 2.1), (2.2, 0.8))
                from, to = project(cam, focal), project(cam, (x * p[1], y * p[2], d))
                push!(extras, (1.0e3, ax -> arrow!(ax, from, to; color = amber, linewidth = 2.2, head = 10)))
            end
            push!(extras, (1.0e3 + 1, ax -> Mke.scatter!(ax, [project(cam, focal)]; color = PALETTE.ink, markersize = 12)))
        end
        draw_blocks!(ax, cam, blocks; extras)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    fig
end

# ╔═╡ 0d5a87a3-1bbb-42a7-b5de-39d9c1cb7f35
"""
The blocks of a patch of the Force's detector, pulled apart layer by layer: board, ceramic
substrate, the CMOS wafer (with a gold via under every dexel), the photodiodes on it, the UFC
dexels in their reflector, the grid (drawn `grid_drawn` mm tall). Returns each layer's blocks
(`layers[key]`), its (bottom, top) (`levels[key]`), and all the blocks together (`blocks`).
"""
function exploded_detector(nx, ny; gap = 1.0, grid_drawn = 2.4)
    p, s, d = pitch_mm, septum_mm, depth_mm
    stack = (
        (key = :board, h = 0.3, gap = 0.5),
        (key = :substrate, h = 0.35, gap = 0.75),
        (key = :cmos, h = 0.4, gap = 0.0),
        (key = :photodiode, h = 0.14, gap = gap),
        (key = :ceramic, h = d, gap = gap),
        (key = :grid, h = grid_drawn, gap = 0.0),
    )
    layers = Dict{Symbol, Vector{Block}}()
    levels = Dict{Symbol, Tuple{Float64, Float64}}()
    z = 0.0
    for layer in stack
        z0, z1 = z, z + layer.h
        levels[layer.key] = (z0, z1)
        layers[layer.key] = if layer.key === :board
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.board)
        elseif layer.key === :substrate
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.substrate)
        elseif layer.key === :cmos
            vcat(
                layer_blocks(nx, ny, p, z0, z1, MATERIALS.cmos),
                [
                    Block((c[1] - 0.07, c[2] - 0.07, z0 - 0.75), (c[1] + 0.07, c[2] + 0.07, z0), MATERIALS.gold)
                        for c in (((i + 0.5) * p[1], (j + 0.5) * p[2]) for i in 0:(nx - 1) for j in 0:(ny - 1))
                ],
            )
        elseif layer.key === :photodiode
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.photodiode; inset = s / 2)
        elseif layer.key === :ceramic
            dexel_blocks(nx, ny, p, s, z0, z1)
        else
            grid_blocks(nx, ny, p, grid.wall_mm, z0, z1)
        end
        z = z1 + layer.gap
    end
    return (; layers, levels, blocks = reduce(vcat, values(layers)))
end

# ╔═╡ 726d09a1-345c-46aa-a33e-dca44249ed7d
let
    Mke.set_theme!(THEME)
    night, ink, sub = MATERIALS.night, Mke.RGBf(0.93, 0.94, 0.96), Mke.RGBf(0.6, 0.63, 0.68)
    fig = Mke.Figure(size = (1500, 600), backgroundcolor = night, figure_padding = (10, 10, 10, 10))

    # (1) The gantry, to scale: two tubes 95° apart, each with its detector arc
    ax1 = canvas(fig[1, 1]; background = night)
    R_iso, R_det = force.source_to_isocenter_mm, force.source_to_detector_mm
    θ = range(0, 2π; length = 200)
    Mke.poly!(ax1, Mke.Point2f.(820 .* cos.(θ), 820 .* sin.(θ)); color = MATERIALS.night_ring)
    Mke.poly!(ax1, Mke.Point2f.(390 .* cos.(θ), 390 .* sin.(θ)); color = night)
    Mke.lines!(ax1, 640 .* cos.(θ), 640 .* sin.(θ); color = (:white, 0.06), linewidth = 1)
    # a patient: a water ellipse with a spine and two lungs
    Mke.poly!(ax1, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = Mke.RGBf(0.42, 0.45, 0.5))
    for x in (-75, 75)
        Mke.poly!(ax1, Mke.Point2f.(x .+ 50 .* cos.(θ), 10 .+ 60 .* sin.(θ)); color = Mke.RGBf(0.16, 0.18, 0.21))
    end
    Mke.poly!(ax1, Mke.Point2f.(22 .* cos.(θ), -105 .+ 20 .* sin.(θ)); color = Mke.RGBf(0.85, 0.85, 0.82))
    function tube!(angle_deg, half_fan_deg, alpha, label)
        a = deg2rad(angle_deg)
        src = Mke.Point2f(R_iso * cos(a), R_iso * sin(a))
        toward = -Mke.Point2f(cos(a), sin(a))
        φ = range(-deg2rad(half_fan_deg), deg2rad(half_fan_deg); length = 80)
        rot(v, t) = Mke.Point2f(v[1] * cos(t) - v[2] * sin(t), v[1] * sin(t) + v[2] * cos(t))
        arc = [src + R_det * rot(toward, t) for t in φ]
        Mke.poly!(ax1, Mke.Point2f.(vcat([src], arc)); color = (MATERIALS.beam, alpha))
        inner = [src + (R_det + 8) * rot(toward, t) for t in φ]
        outer = [src + (R_det + 50) * rot(toward, t) for t in φ]
        Mke.poly!(ax1, Mke.Point2f.(vcat(inner, reverse(outer))); color = Mke.RGBf(0.78, 0.82, 0.88))
        Mke.poly!(ax1, Mke.Rect2f(src[1] - 45, src[2] - 45, 90, 90); color = Mke.RGBf(0.8, 0.83, 0.88))
        Mke.scatter!(ax1, [src]; color = MATERIALS.beam, markersize = 12)
        Mke.text!(ax1, src + Mke.Point2f(0, 85); text = label, color = ink, fontsize = 17, font = :bold, align = (:center, :center))
        return nothing
    end
    tube!(90, 24.9, 0.22, "Tube A")
    tube!(90 + 95, rad2deg(asin(177.5 / R_iso)), 0.14, "Tube B")
    Mke.limits!(ax1, -1010, 1010, -1010, 1010)

    # (2) A patch of the detector under its grid, to scale, the near corner cut away
    ax2 = canvas(fig[1, 2]; background = night)
    p, s, d = pitch_mm, septum_mm, depth_mm
    nx, ny = 7, 5
    cam = camera(-52, 28)
    blocks = dexel_blocks(nx, ny, p, s, 0.0, d)
    walls = filter(grid_blocks(nx, ny, p, grid.wall_mm, d, d + grid.height_mm)) do b
        !(b.lo[1] > 2.4p[1] && b.lo[2] < 3.4p[2])
    end
    append!(blocks, walls)
    rays = map([(3.5, 0.5), (5.5, 1.5), (4.5, 2.5)]) do (i, j)
        c = (i * p[1], j * p[2])
        a, b = project(cam, (c..., d + 12.5)), project(cam, (c..., d + 0.15))
        (dot3((c..., d), cam.toward) + 1, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 12))
    end
    draw_blocks!(ax2, cam, blocks; extras = rays, edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    # (3) One dexel's stack, taken apart
    ax3 = canvas(fig[1, 3]; background = night)
    draw_blocks!(ax3, camera(-56, 30), exploded_detector(1, 1; gap = 0.75).blocks; edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    for (k, (title, subtitle)) in enumerate(
            (
                ("Two Tubes, Two Detectors", "The SOMATOM Force, to Scale: Tubes 95° Apart"),
                ("Dexels Under a 2-D Grid", "UFC Ceramic, White Reflector, Tungsten Walls"),
                ("One Dexel, Taken Apart", "Grid, Ceramic, Photodiode, ADC, Board"),
            )
        )
        Mke.Label(fig[2, k], title; color = ink, fontsize = 26, font = :bold, tellwidth = false)
        Mke.Label(fig[3, k], subtitle; color = sub, fontsize = 18, tellwidth = false)
    end
    Mke.rowgap!(fig.layout, 1, 8)
    Mke.rowgap!(fig.layout, 2, 4)
    fig
end

# ╔═╡ cce76b53-ce59-4546-b518-ff1b5ba68111
let
    Mke.set_theme!(THEME)
    p = pitch_mm
    nx, ny = 4, 3
    cam = camera(-56, 30)
    fig = Mke.Figure(size = (1400, 920))
    ax = canvas(fig[1, 1])
    ex = exploded_detector(nx, ny)
    blocks, levels = ex.blocks, ex.levels

    # X rays from the focal spot, far above, each down the middle of a grid channel
    top = levels[:grid][2]
    rays = map([(0, 0), (1, 0), (2, 0), (3, 0)]) do (i, j)
        c = ((i + 0.5) * p[1], (j + 0.5) * p[2])
        a, b = project(cam, (c..., top + 3.0)), project(cam, (c..., top + 0.35))
        (dot3((c..., top), cam.toward) + 50, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3.5, head = 15))
    end
    draw_blocks!(ax, cam, blocks; extras = rays)

    # One dexel: the front-right column, the same cell in every layer
    i, j = nx - 1, 0
    for (key, (z0, z1)) in levels
        key in (:board, :substrate) && continue
        outline!(ax, cam, (i * p[1], j * p[2], z0), ((i + 1) * p[1], (j + 1) * p[2], z1); linewidth = 3.5)
    end
    Mke.Label(
        fig[2, 1],
        "One Dexel, Outlined in Amber: One Block of Ceramic, Its Photodiode and Its ADC Channel, Under One Grid Cell";
        font = :bold,
        fontsize = 17,
        color = Mke.RGBf(0.8, 0.5, 0.05),
        halign = :left,
        tellwidth = false,
    )

    rows = (
        (:grid, "Anti-Scatter Grid", "2-D Tungsten Walls, One Cell per Dexel\n(10 mm Tall; Drawn Short Here)", true),
        (:ceramic, "UFC Ceramic, Sawn into Dexels", "One Block of Gd₂O₂S per Dexel, 1.4 mm Deep;\nWhite TiO₂ Reflector Fills the Saw Cuts", true),
        (:photodiode, "Photodiode, One per Dexel", "Back-Illuminated Silicon: Light → Charge", false),
        (:cmos, "CMOS Wafer with the ADC", "Digitizes Each Dexel (20 Bit); Gold Vias\nCarry the Number Through the Wafer", false),
        (:substrate, "Ceramic Substrate", "Holds the Chips", false),
        (:board, "Circuit Board", "Carries Every Dexel's Number Out to the Gantry\n(No ADCs on It: Those Are in the Chip Above)", false),
    )
    x_text = maximum(project(cam, (nx * p[1], y, 0))[1] for y in (0, ny * p[2])) + 2.4
    entries = [
        (
                anchor = project(cam, (nx * p[1], ny * p[2] * 0.62, sum(levels[key]) / 2)),
                title = name,
                detail,
                color = PALETTE.ink,
                tag = simulated ? ("In Geant4", TAGS.geant4) : ("Not Simulated", TAGS.not_simulated),
            )
            for (key, name, detail, simulated) in rows
    ]
    labels_right!(ax, entries, x_text; spacing = 1.45, tag_width = 122)
    Mke.xlims!(ax, nothing, x_text + 8.6)
    fig
end

# ╔═╡ c55ace1d-63db-4f39-a0c8-385dbbd2dff2
"""
The master map, as a band across the top of a section's first illustration: the gantry, one
dexel's stack with the part under study (`focus`) lit and outlined, and the dexel's five steps with
this one's lit. `focus` is one of `:tube`, `:arc`, `:grid`, `:ceramic`, `:photodiode`, `:cmos`,
`:whole`. Place it with `locator!(fig[0, :], focus; note)`.
"""
function locator!(position, focus; note = "")
    amber, dim = MATERIALS.beam, Mke.RGBf(0.82, 0.83, 0.86)
    Mke.Box(position; color = Mke.RGBf(0.965, 0.965, 0.975), strokevisible = false, cornerradius = 14)
    layout = Mke.GridLayout(position; halign = :center, tellwidth = false)

    # The gantry, small: tube A, its fan, the detector arc, the one dexel below
    ax = canvas(layout[1, 1]; width = 230, height = 230, backgroundcolor = :transparent)
    θ = range(0, 2π; length = 120)
    R_iso, R_det = force.source_to_isocenter_mm, force.source_to_detector_mm
    lit(k) = focus === k || focus === :whole
    Mke.lines!(ax, 800 .* cos.(θ), 800 .* sin.(θ); color = dim, linewidth = 1.5)
    Mke.poly!(ax, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = (PALETTE.muted, 0.25))
    src = Mke.Point2f(0, R_iso)
    φ = range(-deg2rad(24.9), deg2rad(24.9); length = 60)
    arc = [src + R_det * Mke.Point2f(sin(t), -cos(t)) for t in φ]
    Mke.poly!(ax, Mke.Point2f.(vcat([src], arc)); color = (amber, lit(:tube) ? 0.45 : 0.12))
    Mke.lines!(ax, Mke.Point2f.(arc); color = lit(:arc) ? PALETTE.ink : dim, linewidth = lit(:arc) ? 7 : 4)
    Mke.scatter!(ax, [src]; marker = :rect, markersize = 16, color = lit(:tube) ? PALETTE.ink : dim)
    Mke.scatter!(ax, [arc[30]]; marker = :rect, markersize = 9, color = amber, strokecolor = PALETTE.ink, strokewidth = 1)
    Mke.limits!(ax, -880, 880, -720, 880)

    # One dexel's stack, exploded; the focus lit and outlined, the rest faded
    axs = canvas(layout[1, 2]; width = 170, height = 250, backgroundcolor = :transparent)
    ex = exploded_detector(1, 1; gap = 0.8)
    lit_layers = Dict(
        :grid => (:grid,),
        :ceramic => (:ceramic,),
        :photodiode => (:photodiode,),
        :cmos => (:cmos,),
        :whole => Tuple(keys(ex.layers)),
    )
    on = get(lit_layers, focus, ())
    blocks = Block[]
    for (key, bs) in ex.layers
        append!(blocks, key in on ? bs : [faded(b, 0.86) for b in bs])
    end
    cam = camera(-56, 30)
    draw_blocks!(axs, cam, blocks; edge_alpha = 0.15)
    p = pitch_mm
    for key in on
        z0, z1 = ex.levels[key]
        outline!(axs, cam, (-0.06, -0.06, z0 - 0.06), (p[1] + 0.06, p[2] + 0.06, z1 + 0.06); linewidth = 2.5)
    end

    # Where we are, in words, and the dexel's five steps with this one's lit
    text = Mke.GridLayout(layout[1, 3]; valign = :center)
    Mke.Label(text[1, 1], "Where We Are"; font = :bold, fontsize = 20, color = PALETTE.ink, halign = :left)
    Mke.Label(text[2, 1], note; fontsize = 17, color = PALETTE.ink, halign = :left, justification = :left)
    steps = (
        (:grid, "1  The Grid"),
        (:ceramic, "2  The Ceramic Stops It"),
        (:ceramic, "3  The Ceramic Glows"),
        (:photodiode, "4  The Photodiode"),
        (:cmos, "5  The ADC"),
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

# ╔═╡ 3af4e48b-73ae-4a03-bdc6-e0b3cad7ef2b
let
    μ_per_mm(E) = MC.interpolate_linear(collect(mu_energies), mu_geant4[:total], E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1700, 790))

    red, blue, grey = PALETTE.electron, PALETTE.photon, PALETTE.muted
    panels = (
        ("Photon Absorbed", "Photoelectric: It Vanishes; an Electron\nand Often a 43 keV Gd X Ray Leave", red),
        ("Bounces Off, Loses Energy", "Compton: a Weaker Photon Leaves,\nan Electron Recoils", blue),
        ("Bounces Off, Keeps Energy", "Rayleigh: Same Energy,\na Small Change of Direction", grey),
    )
    for (k, (title, detail, color)) in enumerate(panels)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 21)
        Mke.limits!(ax, -1.25, 1.25, -1.05, 0.95)
        atom!(ax, (0, 0), 0.32)
        photon!(ax, (-1.2, 0.0), (-0.36, 0.0); cycles = 6, amplitude = 0.07, color = MATERIALS.beam)
        if k == 1
            arrow!(ax, (0.12, 0.1), (0.62, 0.55); color = red, linewidth = 3)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = red, fontsize = 16, align = (:left, :bottom))
            photon!(ax, (0.2, -0.24), (1.0, -0.62); cycles = 4, amplitude = 0.07, color = PALETTE.xray)
            Mke.text!(ax, 0.98, -0.75; text = "Gd X Ray", color = PALETTE.xray, fontsize = 16, align = (:right, :top))
        elseif k == 2
            photon!(ax, (0.3, -0.18), (1.1, -0.72); cycles = 3.5, amplitude = 0.07, color = blue)
            Mke.text!(ax, 1.05, -0.86; text = "Weaker Photon", color = blue, fontsize = 16, align = (:right, :top))
            arrow!(ax, (0.2, 0.2), (0.62, 0.55); color = red, linewidth = 2.5)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = red, fontsize = 16, align = (:left, :bottom))
        else
            photon!(ax, (0.35, 0.05), (1.2, 0.28); cycles = 6, amplitude = 0.07, color = MATERIALS.beam)
            Mke.text!(ax, 1.2, 0.42; text = "Same Energy", color = grey, fontsize = 16, align = (:right, :bottom))
        end
        Mke.text!(ax, 0, -0.42; text = "Gd Atom", color = Mke.RGBf(0.4, 0.42, 0.5), fontsize = 15, align = (:center, :top))
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end

    # How far a photon goes, on average, in UFC: 1/μ, against the crystal's 1.4 mm
    ax = Mke.Axis(
        fig[1:2, 4];
        title = "How Far a Photon Goes in UFC",
        xlabel = "Mean Distance Before Interacting (mm)",
        yticks = (1:4, ["40 keV", "60 keV", "100 keV", "140 keV"]),
        yreversed = true,
        limits = ((0, 2.1), (0.4, 4.6)),
        titlesize = 19,
        xlabelsize = 16,
    )
    Mke.vspan!(ax, 0, depth_mm; color = (MATERIALS.ufc, 0.6))
    Mke.text!(ax, depth_mm / 2, 4.5; text = "The 1.4 mm Crystal", align = (:center, :bottom), fontsize = 15, color = PALETTE.slab_edge)
    for (k, E) in enumerate((40.0, 60.0, 100.0, 140.0))
        L = 1 / μ_per_mm(E)
        Mke.barplot!(ax, [k], [L]; direction = :x, color = MATERIALS.beam, width = 0.5)
        Mke.text!(ax, L, k; text = " $(round(L; digits = 2)) mm", align = (:left, :center), fontsize = 15)
    end
    Mke.colsize!(fig.layout, 4, Mke.Relative(0.24))
    locator!(fig[0, :], :ceramic; note = "Step 2 · The Ceramic\nStops the X Ray")
    fig
end

# ╔═╡ f8d455e6-a0c3-585c-a5de-e67e7495e0e1
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1600, 930))


    # (a) the Geant4 model: 13 × 13 dexels, the middle one and its eight neighbours read out
    ax = canvas(fig[1, 1]; title = "(a) What Geant4 Builds: 13 × 13 Dexels, to Scale", titlesize = 20)
    cam = camera(-50, 30)
    blocks = dexel_blocks(13, 13, p, s, 0.0, d; lit = Set([(6, 6)]))
    beam = Block((6p[1] + s / 2, 6p[2] + s / 2, d), (7p[1] - s / 2, 7p[2] - s / 2, d + 5.0), MATERIALS.beam; alpha = 0.3, stroke = 0.0)
    draw_blocks!(ax, cam, blocks; extras = [(1.0e3, a -> draw_blocks!(a, cam, [beam]; edge_alpha = 0.0))], edge_alpha = 0.25)
    outline!(ax, cam, (5p[1], 5p[2], 0), (8p[1], 8p[2], d); color = PALETTE.photon, linewidth = 2.5, linestyle = :dash)
    callout!(ax, project(cam, (6.5p[1], 6.5p[2], d + 5.0)), project(cam, (6.5p[1], 6.5p[2], d + 5.0)) + Mke.Point2f(2.6, 0.4), "Photons Land Uniformly\non the Middle Dexel"; fontsize = 15)
    callout!(ax, project(cam, (8p[1], 5p[2], 0)), project(cam, (8p[1], 5p[2], 0)) + Mke.Point2f(2.2, -1.6), "Read Out: the Middle\nDexel and Its 8\nNeighbours (Dashed)"; fontsize = 15)

    # (b) seen from the beam, 3 × 3, with dimensions
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the Beam",
        xlabel = "Across the Fan (mm)",
        ylabel = "Along Z (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 2.3p[1]), (-2.3p[2], 1.9p[2])),
    )
    rect!(axb, -1.5p[1], -1.5p[2], 3p[1], 3p[2], Mke.RGBf(0.93, 0.93, 0.9); strokecolor = (:black, 0.25), strokewidth = 1)
    for i in -1:1, j in -1:1
        rect!(axb, i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, p[1] - s, p[2] - s, i == 0 && j == 0 ? Mke.RGBf(0.82, 0.95, 0.84) : MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1)
    end
    Mke.bracket!(axb, -0.5p[1], -1.6p[2], 0.5p[1], -1.6p[2]; text = "Pitch $(round(p[1]; digits = 3)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.bracket!(axb, 1.6p[1], -0.5p[2], 1.6p[1], 0.5p[2]; text = "$(round(p[2]; digits = 3)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    note!(axb, 0, 0, "One Dexel"; align = (:center, :center), color = Mke.RGBf(0.15, 0.45, 0.2), fontsize = 15)
    note!(axb, 1.5p[1], 1.5p[2], "White Reflector, $(round(Int, 1000s)) µm"; align = (:right, :bottom), offset = (0, 6), fontsize = 14, color = PALETTE.muted)

    # (c) side view with the depth
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side View",
        xlabel = "Across the Fan (mm)",
        ylabel = "Depth (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 2.0p[1]), (-d - 0.3, 0.7)),
        yticks = ([-d, 0], ["$(d)", "0"]),
    )
    side_dexels!(axc; lit = 1)
    Mke.lines!(axc, [0.1, 0.1], [0.7, -0.5]; color = MATERIALS.beam, linewidth = 3)
    Mke.bracket!(axc, 1.6p[1], -d, 1.6p[1], 0; text = "$(d) mm", orientation = :up, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.38))
    locator!(fig[0, :], :ceramic; note = "Steps 2–3 · The Ceramic,\nOne Dexel and Its\nNeighbours")
    fig
end

# ╔═╡ 84ba9e1d-40c0-4b7e-b634-401e025e07c1
let
    Mke.set_theme!(THEME)
    p, d = pitch_mm[1], depth_mm
    fig = Mke.Figure(size = (1600, 810))

    kα = 43.0
    cases = (
        ("Everything Stays", "The Gd X Ray Is Reabsorbed\nin the Same Dexel → a Peak at E", PALETTE.kept),
        ("The Gd X Ray Escapes", "43 keV Leave Through the Front\n→ a Peak at E − 43 keV", PALETTE.xray),
        ("It Lands Next Door", "The Energy Is Split Between\nTwo Dexels (Both See Part)", Mke.RGBf(0.45, 0.3, 0.6)),
        ("A Compton Photon Leaves", "Only the Electron's Energy\nStays → the Low Tail", PALETTE.photon),
    )
    for (k, (title, detail, color)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 20)
        side_dexels!(ax; lit = 1)
        hit = Mke.Point2f(0.05, -0.45)
        photon!(ax, (0.05, 1.3), hit; cycles = 5, amplitude = 0.03, color = MATERIALS.beam)
        Mke.scatter!(ax, [hit]; marker = :star5, markersize = 22, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)
        if k == 1
            photon!(ax, hit, (0.33, -1.05); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 10)
            Mke.scatter!(ax, [Mke.Point2f(0.33, -1.05)]; color = PALETTE.electron, markersize = 10)
        elseif k == 2
            photon!(ax, hit, (-0.55, 1.2); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 12)
        elseif k == 3
            photon!(ax, hit, (0.85, -0.95); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 10)
            Mke.scatter!(ax, [Mke.Point2f(0.98, -0.95)]; color = PALETTE.electron, markersize = 10)
        else
            photon!(ax, hit, (1.6, 0.6); cycles = 3, amplitude = 0.05, color = PALETTE.photon, head = 12)
        end
        Mke.limits!(ax, -1.75, 1.75, -d - 0.25, 1.45)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :ceramic; note = "Step 2 · Where One\nPhoton's Energy Goes")
    fig
end

# ╔═╡ 8b14052c-6b34-4f30-9a33-429fecf79713
let
    Mke.set_theme!(THEME)
    p, s = pitch_mm, septum_mm
    fig = Mke.Figure(size = (1600, 850))

    lit_colour = Mke.RGBf(0.82, 0.95, 0.84)
    function top!(ax, lit)
        rect!(ax, -2.5p[1], -2.5p[2], 5p[1], 5p[2], Mke.RGBf(0.93, 0.93, 0.9))
        for i in -2:2, j in -2:2
            rect!(ax, i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, p[1] - s, p[2] - s, lit(i, j) ? lit_colour : MATERIALS.ufc; strokecolor = (PALETTE.slab_edge, 0.5), strokewidth = 1)
        end
        return
    end
    ax1 = canvas(fig[1, 1]; title = "One Dexel Lit", titlesize = 20)
    top!(ax1, (i, j) -> i == 0 && j == 0)
    for (dx, dy) in ((1, 0), (0, 1), (-1, 0), (0, -1))
        photon!(ax1, (0.15dx * p[1], 0.15dy * p[2]), (0.95dx * p[1], 0.95dy * p[2]); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 11)
    end
    Mke.Label(fig[2, 1], "Its Gd X Rays Land in the Neighbours:\nLost to This Dexel (the Dashed Curve Below)"; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    ax2 = canvas(fig[1, 2]; title = "Every Dexel Lit (Flat Field)", titlesize = 20)
    top!(ax2, (i, j) -> true)
    for (dx, dy) in ((1, 0), (0, 1), (-1, 0), (0, -1))
        photon!(ax2, (0.15dx * p[1], 0.15dy * p[2] + 0.12), (0.95dx * p[1], 0.95dy * p[2] + 0.12); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 10)
        photon!(ax2, (0.95dx * p[1], 0.95dy * p[2] - 0.12), (0.15dx * p[1], 0.15dy * p[2] - 0.12); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 10)
    end
    Mke.Label(fig[2, 2], "What It Loses, It Gets Back from Them:\nWhat a Scan Sees (the Green Curve)"; fontsize = 16, color = PALETTE.muted, tellwidth = false)

    # Swank: the same photon energy, different signals
    ax3 = Mke.Axis(
        fig[1, 3];
        title = "Same Energy, Different Signals",
        titlesize = 20,
        xlabel = "Ten 60 keV Photons",
        ylabel = "Energy Collected (keV)",
        xticks = 1:10,
        limits = ((0.3, 10.7), (0, 66)),
    )
    signals = [60, 60, 17, 60, 60, 60, 17, 60, 9, 60]
    Mke.barplot!(ax3, 1:10, signals; color = [v == 60 ? PALETTE.kept : v == 17 ? PALETTE.xray : PALETTE.photon for v in signals])
    Mke.Label(fig[2, 3], "The Spread Adds Noise: the Swank Factor\nI = 1 When Every Photon Gives the Same"; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    locator!(fig[0, :], :ceramic; note = "Steps 2–3 · The Whole\nCeramic Layer, Many\nDexels at Once")
    fig
end

# ╔═╡ 619811db-f256-4809-999c-9d7ca805b284
let
    μ_per_mm(E) = MC.geant4_attenuation(ufc_basis, E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 850))

    N = 30
    E = 140.0
    μ = μ_per_mm(E)
    for (k, (depth, title)) in enumerate(((1.4, "1.4 mm: BasisSimulator v0.15.1's Documented Depth"), (3.0, "3.0 mm: the Depth the v0.15.1 Table Fits")))
        stopped = 1 - exp(-μ * depth)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        rect!(ax, 0, 0, 1.6, depth, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        for i in 1:N
            x = 1.6 * (i - 0.5) / N
            z = -log(1 - mod(i * 0.618034, 1.0)) / μ
            if z < depth
                Mke.lines!(ax, [x, x], [depth + 0.55, depth - z]; color = MATERIALS.beam, linewidth = 2)
                Mke.scatter!(ax, [x], [depth - z]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (x, depth + 0.55), (x, -0.5); color = (MATERIALS.beam, 0.6), linewidth = 2, head = 10)
            end
        end
        Mke.text!(ax, 1.68, depth / 2; text = "$(round(Int, 100stopped)) % of\n140 keV Photons\nInteract", align = (:left, :center), fontsize = 16, color = PALETTE.ink)
        Mke.limits!(ax, -0.05, 3.3, -0.62, 3.0 + 0.65)
    end
    Mke.Label(fig[2, 1:2], "At 140 keV the v0.15.1 Table Says 0.816: More Than 1.4 mm Can Absorb at All, Since a Photon That Never Interacts Leaves No Energy."; fontsize = 16, color = PALETTE.muted)
    locator!(fig[0, :], :ceramic; note = "Step 2 · How Deep\nIs the Ceramic?")
    fig
end

# ╔═╡ 9f07fe25-f60a-4976-9252-b06ccba96386
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1600, 600))
    function base!(ax; septum_color = MATERIALS.reflector, septum = s, walls_through = false)
        rect!(ax, -1.5p, -d, 3p, d, septum_color; strokecolor = (:black, 0.3), strokewidth = 1)
        for i in -1:1
            rect!(ax, i * p - (p - septum) / 2, -d, p - septum, d, i == 0 ? Mke.RGBf(0.82, 0.95, 0.84) : MATERIALS.ufc)
        end
        for i in -1:2
            x = (i - 0.5) * p
            rect!(ax, x - grid.wall_mm / 2, 0, grid.wall_mm, 1.0, MATERIALS.tungsten)
            walls_through && rect!(ax, x - septum / 2, -d, septum, d, MATERIALS.tungsten)
        end
        for i in -1:1
            rect!(ax, i * p - (p - septum) / 2, -d - 0.16, p - septum, 0.12, MATERIALS.photodiode)
        end
        return
    end
    hit = Mke.Point2f(0.18, -0.5)
    star!(ax) = Mke.scatter!(ax, [hit]; marker = :star5, markersize = 24, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)

    ax1 = canvas(fig[1, 1]; title = "The Force: Grid Above, White Reflector in the Saw Cuts", titlesize = 20)
    base!(ax1)
    photon!(ax1, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    half = (p - s) / 2
    Mke.lines!(ax1, Mke.Point2f[hit, (half, -0.15), (0.3, -d)]; color = MATERIALS.light, linewidth = 2.5)
    Mke.lines!(ax1, Mke.Point2f[hit, (-half, -0.3), (-0.25, -d)]; color = MATERIALS.light, linewidth = 2.5)
    photon!(ax1, hit, (1.15p, -1.0); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 11)
    star!(ax1)
    Mke.Label(fig[2, 1], "Light (Green): Bounces Off the Reflector, Stays in Its Dexel\nGd X Ray (Orange, 43 keV): Passes Through the Thin, Light Reflector"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax1, -1.65p, 1.65p, -d - 0.3, 1.75)

    ax2 = canvas(fig[1, 2]; title = "Hypothetical: Tungsten Walls Down Through the Ceramic", titlesize = 20)
    base!(ax2; septum = 0.16, walls_through = true)
    photon!(ax2, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    photon!(ax2, hit, (half + 0.02, -0.82); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 11)
    Mke.scatter!(ax2, [Mke.Point2f(half + 0.06, -0.85)]; marker = :xcross, markersize = 18, color = PALETTE.xray)
    star!(ax2)
    Mke.Label(fig[2, 2], "Stops the Gd X Ray, but: Wider Cuts Leave Less Ceramic per Dexel,\nTungsten Adds Its Own X Rays (59 and 67 keV), and It Must Be Built In"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax2, -1.65p, 1.65p, -d - 0.3, 1.75)
    fig
end

# ╔═╡ 6b445b19-7204-5976-8445-5845677718c6
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    H, w = grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1600, 1050))


    # (a) to scale, the near corner cut away
    ax = canvas(fig[1, 1]; title = "(a) The 2-D Grid over the Dexels, to Scale", titlesize = 20)
    cam = camera(-52, 24)
    nx, ny = 6, 5
    blocks = dexel_blocks(nx, ny, p, s, 0.0, d)
    append!(blocks, filter(b -> !(b.lo[1] > 2.4p[1] && b.lo[2] < 2.4p[2]), grid_blocks(nx, ny, p, w, d, d + H)))
    straight = ((4.5p[1], 0.5p[2], d + H + 3.5), (4.5p[1], 0.5p[2], d))
    tilt = tand(20)
    stop_z = d + H - (0.5p[1] - w / 2) / tilt
    angled = ((3.0p[1] + 0.02, 1.5p[2], d + H + 2.0), (4.0p[1] - w / 2, 1.5p[2], stop_z))
    extras = [
        (1.0e3, a -> arrow!(a, project(cam, straight[1]), project(cam, straight[2]); color = MATERIALS.beam, linewidth = 3.5)),
        (1.0e3, a -> Mke.lines!(a, [project(cam, angled[1]), project(cam, angled[2])]; color = PALETTE.xray, linewidth = 3.5)),
        (1.0e3, a -> Mke.scatter!(a, [project(cam, angled[2])]; marker = :xcross, markersize = 20, color = PALETTE.xray)),
    ]
    draw_blocks!(ax, cam, blocks; extras)
    callout!(ax, project(cam, (4.5p[1], 0.5p[2], d + H + 2.5)), project(cam, (4.5p[1], 0.5p[2], d + H + 2.5)) + Mke.Point2f(3.0, 1.2), "From the Focal Spot:\nStraight Down a Channel"; color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 15)
    callout!(ax, project(cam, angled[2]), project(cam, angled[2]) + Mke.Point2f(3.6, -2.2), "Scattered in the Patient:\nStopped by a Wall"; color = PALETTE.xray, fontsize = 15)
    callout!(ax, project(cam, (nx * p[1], ny * p[2], d + H / 2)), project(cam, (nx * p[1], ny * p[2], d + H / 2)) + Mke.Point2f(1.4, 4.2), "Tungsten Walls\n$(round(Int, 1000w)) µm Thick,\n$(round(Int, H)) mm Tall"; fontsize = 15)
    Mke.xlims!(ax, nothing, project(cam, (nx * p[1], 0, 0))[1] + 9.5)

    # (b) side view, to scale, with the angle and the shadow a wall casts
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) One Channel, Side View (Width Stretched)",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Height Above the Ceramic (mm)",
        limits = ((-0.75p[1], 1.05p[1]), (-d - 0.2, H + 1.4)),
    )
    rect!(axb, -0.5p[1], -d, p[1], d, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
    for x in (-0.5p[1], 0.5p[1])
        rect!(axb, x - w / 2, 0, w, H, MATERIALS.tungsten)
    end
    θ = 4.0
    # photons at θ enter the opening evenly; those entering within H tan θ of the far wall hit it
    xl, xr = -0.5p[1] + w / 2, 0.5p[1] - w / 2
    reach = H * tand(θ)
    Mke.poly!(axb, Mke.Point2f[(xr - reach, H), (xr, H), (xr, 0)]; color = (PALETTE.xray, 0.2))
    for x_in in range(xl + 0.03, xr - 0.03; length = 7)
        x_out = x_in + reach
        if x_out <= xr
            Mke.lines!(axb, [x_in, x_out], [H, 0]; color = MATERIALS.beam, linewidth = 2)
        else
            y_hit = H - (xr - x_in) / tand(θ)
            Mke.lines!(axb, [x_in, xr], [H, y_hit]; color = PALETTE.xray, linewidth = 2)
            Mke.scatter!(axb, [xr], [y_hit]; marker = :xcross, markersize = 12, color = PALETTE.xray)
        end
    end
    Mke.bracket!(axb, xr - reach, H + 0.1, xr, H + 0.1; text = "H tan θ: These Hit the Wall", orientation = :up, fontsize = 14, color = PALETTE.xray, textcolor = PALETTE.xray)
    Mke.bracket!(axb, 0.5p[1] + 0.12, 0, 0.5p[1] + 0.12, H; text = "H = $(round(Int, H)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.Label(fig[2, 2], "Photons Arriving θ = $(round(Int, θ))° off the Channel's Axis:\nThe Fraction 1 − H tan θ / Opening Reaches the Ceramic"; fontsize = 15, color = PALETTE.ink, tellwidth = false)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.32))
    locator!(fig[0, :], :grid; note = "Step 1 · The Grid\nIn Front of the Ceramic")
    fig
end

# ╔═╡ af1722c8-f3ff-5d4f-9025-ee5b13e6be0d
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (ceramic and reflector), with and without the grid: the grid
should cost almost nothing, apart from its own tungsten x rays (59 and 67 keV) reaching the
ceramic above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ cb12b286-ac74-5e3b-bc12-8c34c185c197
grid_energies = 20.0:5.0:140.0

# ╔═╡ 57591bb0-24bc-53ea-b228-25b0571db01f
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

# ╔═╡ 13d20760-ef7e-5049-b1e9-f732c752aa18
angles_deg = 0.0:1.0:20.0

# ╔═╡ 146cfad9-9898-589e-aad6-611c73f14d5e
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

# ╔═╡ e57ccb84-f6c5-595d-bf5b-e001a1bd0226
let
    Mke.set_theme!(THEME)
    E = collect(grid_energies)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Straight from the Focal Spot",
        xlabel = "Photon Energy (keV)",
        ylabel = "Energy Collected per Photon on the Cell, ÷ E",
        xticks = 20:20:140,
        limits = ((20, 140), (0.4, 1.0)),
    )
    Mke.lines!(
        axa,
        E,
        MC.energy_absorption_efficiency.(grid_runs.without);
        color = PALETTE.muted,
        linewidth = 3,
        label = "No Grid",
    )
    Mke.lines!(
        axa,
        E,
        MC.energy_absorption_efficiency.(grid_runs.with);
        color = PALETTE.tungsten,
        linewidth = 2.5,
        linestyle = :dash,
        label = "With the 2-D Grid",
    )
    Mke.vlines!(axa, 69.5; color = (PALETTE.xray, 0.6), linestyle = :dot)
    note!(
        axa,
        69.5,
        0.45,
        "W K-Edge";
        align = (:left, :bottom),
        offset = (6, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.axislegend(axa; position = :rt)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) 60 keV Photons Arriving at an Angle",
        xlabel = "Angle Across the Fan (Degrees)",
        ylabel = "Energy Collected, Relative to\nStraight Down Without a Grid",
        limits = ((0, 20), (0, 1.35)),
    )
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    Mke.lines!(
        axb,
        collect(angles_deg),
        MC.energy_absorption_efficiency.(angle_runs.without) ./ reference;
        color = PALETTE.muted,
        linewidth = 3,
        label = "No Grid",
    )
    Mke.lines!(
        axb,
        collect(angles_deg),
        MC.energy_absorption_efficiency.(angle_runs.with) ./ reference;
        color = PALETTE.tungsten,
        linewidth = 3,
        label = "With the 2-D Grid",
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
        label = "Geometry Alone: 1 − H tan θ / Opening",
    )
    Mke.axislegend(axb; position = :rc)
    fig
end

# ╔═╡ 92c3a865-32c1-59f6-ba69-836443f3e8fe
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

# ╔═╡ b802d7eb-f757-5496-a7cc-0653fc3df872
md"""
## 7. The Whole Detector

The whole detector, as BasisSimulator has it: 920 columns × 96 rows on an arc of radius
1085.6 mm around the focal spot, every dexel a block of ceramic in reflector, under the 2-D grid,
all of it focused on the focal spot. (The real detector is spherical, ⚠ *discrepancy* above;
across the fan and along z every dexel faces the focal spot either way.) An air scan fires
photons uniformly from the focal spot over the whole detector, and every dexel's signal is
summed. Two questions: is every dexel the same, and does the whole detector collect what the
single dexel of sections 4 and 6 predicts?
"""

# ╔═╡ 8c5a6f70-f994-54e7-9d70-ebca20449ebd
detector = MC.ArcDetector(;
    columns = force.columns,
    rows = force.rows,
    radius_mm = force.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :both,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = ufc,
)

# ╔═╡ 1bbe17d8-933e-51a9-a6cd-b7babce22957
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ 1ee36aa2-5d50-5b55-94da-2b5234f8c818
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

# ╔═╡ 03ab3733-cb26-4085-80df-a344298a26fb
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 990))

    ax = canvas(fig[1, 1]; title = "The Whole Detector, from the Focal Spot (Depth Exaggerated)", titlesize = 20)
    cam = camera(-62, 26)
    R = force.source_to_detector_mm
    half = deg2rad(24.9)
    half_z = force.rows * force.row_at_isocenter_mm * R / force.source_to_isocenter_mm / 2
    amber = MATERIALS.beam
    items = arc_quads(cam, R, half, half_z; thickness = 25.0)
    # the beam: fan × cone, translucent, from the focal spot
    focal = (0.0, 0.0, 0.0)
    corners = [(R * sin(a), -R * cos(a), z) for (a, z) in ((-half, -half_z), (half, -half_z), (half, half_z), (-half, half_z))]
    push!(
        items, (
            1.0e4, a -> begin
                for (c1, c2) in zip(corners, circshift(corners, -1))
                    Mke.poly!(a, Mke.Point2f[project(cam, focal), project(cam, c1), project(cam, c2)]; color = (amber, 0.1))
                end
                Mke.lines!(a, [project(cam, c) for c in vcat([focal], corners[1:1])]; color = (amber, 0.7), linewidth = 1.5)
                for c in corners
                    Mke.lines!(a, [project(cam, focal), project(cam, c)]; color = (amber, 0.7), linewidth = 1.5)
                end
                Mke.scatter!(a, [project(cam, focal)]; color = PALETTE.ink, markersize = 14)
                iso = (0.0, -force.source_to_isocenter_mm, 0.0)
                Mke.scatter!(a, [project(cam, iso)]; marker = :cross, markersize = 18, color = PALETTE.ink)
            end,
        )
    )
    sort!(items; by = first)
    for (_, f) in items
        f(ax)
    end
    iso = (0.0, -force.source_to_isocenter_mm, 0.0)
    callout!(ax, project(cam, focal), project(cam, focal) + Mke.Point2f(150, 80), "Focal Spot"; fontsize = 16)
    callout!(ax, project(cam, iso), project(cam, iso) + Mke.Point2f(260, 60), "Isocentre, $(round(Int, force.source_to_isocenter_mm)) mm Away"; fontsize = 16)
    edge = (R * sin(half), -R * cos(half), half_z)
    callout!(ax, project(cam, edge), project(cam, edge) + Mke.Point2f(120, 120), "$(force.columns) Columns Across the Fan ($(round(2rad2deg(half); digits = 1))°)\n× $(force.rows) Rows Along Z, at $(round(R; digits = 1)) mm"; fontsize = 16)
    low = (R * sin(-half / 2), -R * cos(-half / 2), -half_z)
    callout!(ax, project(cam, low), project(cam, low) + Mke.Point2f(-120, -160), "The Outer Rows See the Beam\nat the Cone Angle"; fontsize = 16, align = (:right, :center))
    xs = [project(cam, c)[1] for c in vcat(corners, [focal])]
    ys = [project(cam, c)[2] for c in vcat(corners, [focal])]
    Mke.limits!(ax, minimum(xs) - 520, maximum(xs) + 620, minimum(ys) - 240, maximum(ys) + 200)
    locator!(fig[0, :], :arc; note = "The Whole Arc:\n920 × 96 Dexels")
    fig
end

# ╔═╡ 39f15a81-ff31-5ef6-acd6-23a32fa8b0b0
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))

    # (b) The signal per photon arriving, across the fan, in groups of 40 columns
    axb = Mke.Axis(
        fig[1, 1];
        title = "(a) Energy Collected per Photon, Across the Fan",
        xlabel = "Column (Across the Fan)",
        ylabel = "Energy Collected per Photon\non the Cell, ÷ E",
        limits = ((1, force.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:force.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / force.columns
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
        force.columns / 2,
        0.42,
        "Dashed: One Dexel's Cell, Section 6 (with the Grid)";
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
        fig[1, 2];
        title = "(b) Along Z",
        xlabel = "Row (Along Z)",
        limits = ((1, force.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:force.rows, max(1, force.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / force.rows
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
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.3))
    fig
end

# ╔═╡ b13ffa4b-aa10-5428-ab58-f989dc4015db
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
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / force.source_to_detector_mm))
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

# ╔═╡ 0827c6a3-5316-565c-8c37-0cf97d73a0ef
md"""
## 8. The Force's X-Ray Beam

The photons come from the installed BasisSimulator (these are not what v0.16.0 replaces): its IPEM Report 78 tungsten spectra with the
Force's 8° anode, through the scanner's 3 mm of aluminium and the protocol's filters, scaled to
the detector at 1085.6 mm, exactly as `simulate!` uses them for the Force. The Force runs from
70 to 150 kV in 10 kV steps, with or without 0.6 mm of tin; BasisSimulator's spectra are tabulated
at 80, 100, 120 and 140 kV, and its Force protocols also add 0.9 mm of titanium.

The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ 7cecf8ed-e83b-5b37-8b05-cd529f4575e4
# BasisSimulator's Force, as on its scanner page
force_scanner = BS.EICTScanner(
    source_to_isocenter = force.source_to_isocenter_mm,
    source_to_detector = force.source_to_detector_mm,
    detector_rows = force.rows,
    detector_cols = force.columns,
    detector_row_size = force.row_at_isocenter_mm,
    detector_col_size = force.column_at_isocenter_mm,
    focal_spot_width = 0.8,
    focal_spot_length = 1.2,
    target_angle = 8.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 3.0,
    bowtie_filter = :large_body,
    detector_material = :ufc,
    detector_depth = depth_mm,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 0,
    detection_gain = 10.0,
)

# ╔═╡ 08f77ac1-c4b3-51bb-bb31-0a12c45749d7
"""
The photons per mAs per mm² reaching the detector, per bin, for a Force protocol at `kVp` with
`titanium_mm` of titanium (BasisSimulator's protocols: 0.9) and `tin_mm` of tin (0.6 for "Sn").
"""
function force_spectrum(kVp; titanium_mm = 0.9, tin_mm = 0.0)
    filters = Tuple{String, Float64}[]
    titanium_mm > 0 && push!(filters, ("Ti", titanium_mm))
    tin_mm > 0 && push!(filters, ("Sn", tin_mm))
    protocol = BS.CTProtocol(;
        kVp,
        mA = 200.0,
        views = 1160,
        rotation_time = 0.5,
        collimation_mm = 4.8,
        anode_angle = 8,
        additional_filters = filters,
    )
    return BS.resolve_source_spectrum_without_bowtie(
        BS.SimOptions(),
        protocol;
        scanner = force_scanner,
    )
end

# ╔═╡ 2a8e6eb5-7861-52a8-ba63-00aa89219d07
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

# ╔═╡ 117159e7-d9ee-4ebc-be5f-3e24793241d3
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))

    ax = canvas(fig[1, 1])
    amber = MATERIALS.beam
    # tube: envelope, cathode, anode at 8°
    Mke.poly!(ax, Mke.Rect2f(-0.6, -1.5, 2.4, 3.0); color = Mke.RGBf(0.93, 0.94, 0.96), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    rect!(ax, -0.45, -0.25, 0.3, 0.5, PALETTE.slab_edge)
    Mke.text!(ax, -0.3, -0.45; text = "Cathode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.poly!(ax, Mke.Point2f[(1.2, -0.9), (1.55, -0.9), (1.55, 0.9), (1.35, 0.9)]; color = Mke.RGBf(0.6, 0.45, 0.3))
    Mke.text!(ax, 1.42, -1.0; text = "Tungsten Anode, 8°", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    arrow!(ax, (-0.1, 0.0), (1.22, 0.0); color = PALETTE.photon, linewidth = 2.5, head = 11)
    Mke.text!(ax, 0.55, 0.12; text = "Electrons", align = (:center, :bottom), fontsize = 14, color = PALETTE.photon)
    focal = Mke.Point2f(1.28, 0.0)
    Mke.text!(ax, 0.6, 1.6; text = "X-Ray Tube", align = (:center, :bottom), fontsize = 18, font = :bold, color = PALETTE.ink)
    # the fan, rightwards
    x_end = 13.5
    Mke.poly!(ax, Mke.Point2f[focal, (x_end, 2.6), (x_end, -2.6)]; color = (amber, 0.13))
    for y in (-2.6, 0.0, 2.6)
        Mke.lines!(ax, [focal, Mke.Point2f(x_end, y)]; color = (amber, 0.5), linewidth = 1)
    end
    filters = (
        (2.4, 0.18, Mke.RGBf(0.75, 0.77, 0.8), "3 mm Al", "Always"),
        (3.7, 0.12, Mke.RGBf(0.6, 0.62, 0.68), "0.9 mm Ti", "BasisSimulator's\nProtocols"),
        (5.0, 0.1, Mke.RGBf(0.55, 0.6, 0.5), "0.6 mm Sn", "\"Sn\" Protocols\nOnly"),
    )
    for (x, w, c, name, when) in filters
        h = 0.25 + (x - focal[1]) * 2.6 / (x_end - focal[1])
        rect!(ax, x, -h, w, 2h, c; strokecolor = (:black, 0.3))
        Mke.text!(ax, x + w / 2, h + 0.12; text = name, align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
        Mke.text!(ax, x + w / 2, -h - 0.12; text = when, align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    end
    # bowtie: thin in the middle, thick at the edges
    xb = 6.3
    hb = 0.25 + (xb - focal[1]) * 2.6 / (x_end - focal[1])
    Mke.poly!(ax, Mke.Point2f[(xb, -hb), (xb + 0.55, -hb), (xb + 0.08, 0), (xb + 0.55, hb), (xb, hb)]; color = Mke.RGBf(0.75, 0.77, 0.8), strokecolor = (:black, 0.3))
    Mke.text!(ax, xb + 0.25, hb + 0.12; text = "Bowtie", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.text!(ax, xb + 0.25, -hb - 0.12; text = "2.5 mm Al\nat the Centre", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    # patient and isocentre
    θ = range(0, 2π; length = 100)
    xi = 9.6
    Mke.poly!(ax, Mke.Point2f.(xi .+ 1.1 .* cos.(θ), 0.8 .* sin.(θ)); color = (MATERIALS.water, 0.6), strokecolor = (PALETTE.photon, 0.5))
    Mke.scatter!(ax, [Mke.Point2f(xi, 0)]; marker = :cross, markersize = 16, color = PALETTE.ink)
    Mke.text!(ax, xi, 0.95; text = "Patient (Isocentre, 595 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    # detector with grid
    rect!(ax, x_end, -2.7, 0.15, 5.4, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
    for y in range(-2.6, 2.6; length = 14)
        rect!(ax, x_end - 0.35, y - 0.01, 0.35, 0.02, MATERIALS.tungsten)
    end
    Mke.text!(ax, x_end + 0.1, 2.85; text = "Grid + Detector\n(1085.6 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -0.8, 14.6, -3.4, 3.9)
    locator!(fig[0, :], :tube; note = "The X-Ray Tube and\nWhat the Beam Passes")
    fig
end

# ╔═╡ b2e013dd-2bc7-586d-99f3-0a91839cc767
@bind kvp PlutoUI.Slider([80, 100, 120, 140]; default = 120, show_value = true)

# ╔═╡ 112e15a2-75dd-53d6-92b8-511aad50fc7a
let
    Mke.set_theme!(THEME)
    plain = force_spectrum(kvp)
    tin = force_spectrum(kvp; tin_mm = 0.6)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The Photons Reaching the Detector, $(kvp) kV",
        xlabel = "Photon Energy (keV)",
        ylabel = "Photons per mAs per mm²\nper keV",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    for (spectrum, color, label) in ((plain, PALETTE.photon, "3 mm Al + 0.9 mm Ti"), (tin, PALETTE.xray, "… + 0.6 mm Tin"))
        s = signal_per_kev(spectrum, efficiency.flat_field)
        Mke.lines!(axa, s.energy_keV, s.photons; color, linewidth = 3, label)
    end
    Mke.axislegend(axa; position = :rt)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Signal They Make (3 mm Al + 0.9 mm Ti)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Signal per keV\n(Photons × E × η, Arbitrary Units)",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    ours = signal_per_kev(plain, efficiency.flat_field)
    theirs = signal_per_kev(plain, table_at.(collect(energies)))
    Mke.lines!(
        axb,
        theirs.energy_keV,
        theirs.signal;
        color = PALETTE.reference,
        linewidth = 3,
        label = "BasisSimulator v0.15.1's Table",
    )
    Mke.lines!(
        axb,
        ours.energy_keV,
        ours.signal;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, flat field, 1.4 mm",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ ee9b4076-8344-5e14-a36c-b00985bdcdb6
md"""
**The beam's quality, against the scanner.** Siemens' technical manual gives the Force's
half-value layer at 120 kV at the isocentre as **7.8 mm of aluminium** (as used by Fong et al.,
Med. Phys. 2024). At the isocentre the beam has passed the bowtie's thinnest part, 2.5 mm of
aluminium in BasisSimulator's large-body bowtie.
"""

# ╔═╡ 9061fc53-ea2f-5cdf-9b1e-8bdf1959e27e
# The half-value layer at the isocentre: BasisSimulator's 120 kV beam through the bowtie's centre,
# with and without the protocol's 0.9 mm of titanium
hvl_check = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium = MC.DetectorMaterial(:Al)

    function hvl(titanium_mm)
        e, w = force_spectrum(120; titanium_mm)
        μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
        return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
    end

    [
        (label = "BasisSimulator:\n3 mm Al + 0.9 mm Ti", hvl = hvl(0.9)),
        (label = "3 mm Al + 0.6 mm Ti\n(Siemens' UFC Test Beam)", hvl = hvl(0.6)),
        (label = "3 mm Al,\nNo Titanium", hvl = hvl(0.0)),
    ]
end

# ╔═╡ 6ed7c91d-53b6-48e0-b71d-113621a4576d
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 380))
    for (k, (al, reading, title)) in enumerate(((0, "100 %", "No Aluminium in the Beam"), (7.8, "50 %", "7.8 mm of Aluminium: Half Is Left")))
        ax = canvas(fig[1, k]; title, titlesize = 20)
        Mke.scatter!(ax, [Mke.Point2f(0, 0)]; marker = :rect, markersize = 22, color = PALETTE.ink)
        Mke.text!(ax, 0, -0.45; text = "Tube\n(120 kV, Through\nthe Bowtie's Centre)", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
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

# ╔═╡ e3e24f7d-3d7c-50fe-8569-340354b4e66c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-Value Layer at the Isocentre, 120 kV",
        xlabel = "Half-Value Layer (mm of Aluminium)",
        yticks = (1:3, [c.label for c in hvl_check]),
        yreversed = true,
        limits = ((0, 12), (0.4, 4.2)),
    )
    Mke.vspan!(ax, 7.75, 7.85; color = (PALETTE.kept, 0.45))
    note!(
        ax,
        7.8,
        3.85,
        "Siemens' Technical Manual: 7.8 mm";
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

# ╔═╡ 8afa396b-e510-5410-ab20-58f29489aefe
let
    basis, test_beam, bare = (c.hvl for c in hvl_check)
    closest = hvl_check[argmin([abs(c.hvl - 7.8) for c in hvl_check])]
    verdict = abs(basis - 7.8) <= 0.4 ?
        "BasisSimulator's beam matches Siemens' value." :
        "**⚠ Discrepancy:** BasisSimulator's beam is $(basis > 7.8 ? "harder" : "softer") than Siemens' value; the closest of the three is $(replace(closest.label, "\n" => " "))."
    Markdown.parse(
        """
        At 120 kV, the installed BasisSimulator's (v$(pkgversion(BS))) Force beam has a half-value layer of **$(round(basis; digits = 1)) mm** at the isocentre,
        against Siemens' 7.8 mm. $(verdict) The rest of this notebook keeps BasisSimulator's beam, so the two stay
        comparable.
        """
    )
end

# ╔═╡ 807fa3c3-543c-55b6-b4cf-17f8a7415a58
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. Every Force tube voltage BasisSimulator can make, with
its titanium, and the two tin-filtered ones:
"""

# ╔═╡ 104874e1-f75a-5c48-8023-b03c1104a754
scan_protocols = vcat(
    [(label = "$(k) kV", spectrum = force_spectrum(k)) for k in (80, 100, 120, 140)],
    [
        (label = "Sn100", spectrum = force_spectrum(100; tin_mm = 0.6)),
        (label = "Sn140", spectrum = force_spectrum(140; tin_mm = 0.6)),
    ],
)

# ╔═╡ 1770b852-19ac-5728-86cb-7419fbd8af32
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

# ╔═╡ c6e2851e-23f2-4ea1-86c6-9d9f959b794f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 760))

    photons = ((40, 0.6), (70, 1.6), (110, 2.6))
    for (k, (title, detail)) in enumerate(
            (
                ("A Photon-Counting Dexel", "Each Photon Counts 1:\nSignal = 3"),
                ("This Energy-Integrating Dexel", "Each Photon Adds Its Energy × η(E):\nSignal ∝ 40 η(40) + 70 η(70) + 110 η(110)"),
            )
        )
        ax = canvas(fig[1, k]; title, titlesize = 20)
        rect!(ax, 0, -0.6, 3.2, 0.6, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
        for (E, x) in photons
            photon!(ax, (x, 2.2), (x, -0.25); cycles = E / 12, amplitude = 0.02, color = MATERIALS.beam, linewidth = 1 + E / 30)
            Mke.text!(ax, x, 2.3; text = "$(E) keV", align = (:center, :bottom), fontsize = 15, color = PALETTE.ink)
            if k == 1
                Mke.textlabel!(ax, Mke.Point2f(x, -1.2); text = "+1", text_align = (:center, :center), fontsize = 16, font = :bold, text_color = :white, background_color = PALETTE.photon, strokewidth = 0, cornerradius = 8)
            else
                Mke.poly!(ax, Mke.Rect2f(x - 0.18, -1.65, 0.36, 0.28 * E / 40); color = PALETTE.kept)
            end
        end
        Mke.limits!(ax, -0.4, 3.6, -1.8, 2.8)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :photodiode; note = "Steps 4–5 · Adding Up\nthe Light of Many Photons")
    fig
end

# ╔═╡ 69f8b2ab-1204-588b-a121-408d60b26374
let
    Mke.set_theme!(THEME)
    x = 1:length(scan_metrics)
    labels = [m.label for m in scan_metrics]
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Mean Efficiency over Each Protocol's Spectrum",
        ylabel = "Energy-Weighted Efficiency",
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
        label = "BasisSimulator v0.15.1's Table",
    )
    Mke.scatterlines!(
        axa,
        x,
        [m.geant4 for m in scan_metrics];
        color = PALETTE.kept,
        markersize = 13,
        linewidth = 2.5,
        label = "Geant4, 1.4 mm, Flat Field",
    )
    Mke.axislegend(axa; position = :lb)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) DQE(0) over Each Spectrum (Geant4)",
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

# ╔═╡ f0dfceab-4c51-5307-8cf8-48720f5babef
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | v0.15.1 table | Geant4, 1.4 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's ceramic; BasisSimulator's fill factor (0.9 × 0.9) applies to both
        alike.
        """
    )
end

# ╔═╡ fde8bf46-2e00-4a50-91f1-a212e816640f
md"""
## 10. A Clinical Beam, for Real

Every efficiency so far came from photons of **one energy**, landing straight on **one dexel**.
A scan is neither: the Force's tube sends a spectrum, hardened by the bowtie and by the patient,
from a focal spot over the whole arc, so the dexels at the edges of the fan and the outer rows
see it at an angle, through the grid. BasisSimulator never simulates that photon by photon; it
weights the single-energy table by the spectrum. This section checks that the shortcut is exact,
by doing it the long way:

- the Force's **120 kV beam** (BasisSimulator's spectrum, 3 mm Al + 0.9 mm Ti, through the
  bowtie's centre), **bare and after 20 and 35 cm of water** (the patient's primary photons; the
  patient's scatter is notebook 10's);
- each one **drawn photon by photon from its spectrum**, onto one dexel's ceramic (1.6 million
  photons) and from the focal spot over **the whole arc** with its grid (3.2 million);
- against the table **weighted by the same spectrum**: `Σ φ E η / Σ φ E` for the efficiency, and
  `(Σ φ m₁)² / (Σ φ · Σ φ m₂)` for DQE(0), with `m₁`, `m₂` the mean and mean square of the energy
  a photon leaves (section 4).
"""

# ╔═╡ a98d98ef-b4d5-452c-9f38-07e9e885c3d8
# The Force's 120 kV beam on the central ray: BasisSimulator's spectrum through the bowtie's
# centre, then through 0, 20 and 35 cm of water (the primary photons), on its 0.5 keV grid
clinical_beams = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium, water = MC.DetectorMaterial(:Al), MC.DetectorMaterial(:Water)
    e, w = force_spectrum(120)
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

# ╔═╡ fa6ed8d8-c9bc-4c81-874b-7096b8840239
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))


    # The path: tube, bowtie, water, the arc with its grid
    ax = canvas(fig[1, 1]; title = "The Beam, Photon by Photon", titlesize = 20)
    amber = MATERIALS.beam
    focal = Mke.Point2f(0, 0)
    Mke.scatter!(ax, [focal]; marker = :rect, markersize = 22, color = PALETTE.ink)
    Mke.text!(ax, focal + Mke.Point2f(0, 0.45); text = "120 kV Tube", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    φ = range(-deg2rad(24.9), deg2rad(24.9); length = 60)
    R = 10.0
    arc = [Mke.Point2f(R * sin(t), -R * cos(t)) for t in φ]
    Mke.poly!(ax, vcat([focal], arc); color = (amber, 0.15))
    Mke.poly!(ax, Mke.Point2f[(-1.2, -1.6), (1.2, -1.6), (0.15, -2.0), (1.2, -2.4), (-1.2, -2.4), (-0.15, -2.0)]; color = Mke.RGBf(0.75, 0.77, 0.8))
    Mke.text!(ax, 1.35, -2.0; text = "Bowtie", align = (:left, :center), fontsize = 15, color = PALETTE.ink)
    θ = range(0, 2π; length = 100)
    Mke.poly!(ax, Mke.Point2f.(2.4 .* cos.(θ), -5.45 .+ 1.6 .* sin.(θ)); color = (MATERIALS.water, 0.7), strokecolor = (PALETTE.photon, 0.5))
    Mke.text!(ax, 0, -5.45; text = "0, 20 or 35 cm\nof Water", align = (:center, :center), fontsize = 15, color = PALETTE.ink)
    inner = [Mke.Point2f((R + 0.05) * sin(t), -(R + 0.05) * cos(t)) for t in φ]
    outer = [Mke.Point2f((R + 0.45) * sin(t), -(R + 0.45) * cos(t)) for t in φ]
    Mke.poly!(ax, vcat(inner, reverse(outer)); color = MATERIALS.ufc, strokecolor = PALETTE.slab_edge)
    for t in range(φ[1], φ[end]; length = 31)
        Mke.lines!(ax, Mke.Point2f[((R - 0.5) * sin(t), -(R - 0.5) * cos(t)), (R * sin(t), -R * cos(t))]; color = MATERIALS.tungsten, linewidth = 1)
    end
    Mke.text!(ax, 0, -R - 0.7; text = "The Whole Arc and Its Grid (920 × 96 Dexels)", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -6.5, 6.5, -R - 1.8, 1.4)

    # The three spectra, each as a fraction of its photons per keV: the water hardens the beam
    continuum(e) = !(57 <= e <= 70)
    spectra = [
        (b.label, b.spectrum.energies_keV, b.spectrum.weights ./ (sum(b.spectrum.weights) * 0.5))
            for b in clinical_beams
    ]
    top = 1.35 * maximum(maximum(w[continuum.(e)]) for (_, e, w) in spectra)
    axs = Mke.Axis(
        fig[1, 2];
        title = "The Beam Hardens in the Patient",
        titlesize = 20,
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons per keV",
        limits = ((10, 125), (0, top)),
        xticks = 20:20:120,
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron)
    for (k, (label, e, w)) in enumerate(spectra)
        mean_E = sum(w .* e) / sum(w)
        Mke.lines!(axs, e, w; color = colors[k], linewidth = 3, label = "$(label): Mean $(round(mean_E; digits = 1)) keV")
    end
    note!(axs, 63.5, 0.97top, "Tungsten's K Lines\n(Off Scale)"; align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.axislegend(axs; position = :rt)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.32))
    locator!(fig[0, :], :whole; note = "Everything at Once:\nTube, Patient, Grid,\nEvery Dexel")
    fig
end

# ╔═╡ 448c2bf9-de8a-4f88-9e56-464711960b56
# Each beam photon by photon: onto one dexel's ceramic (16 chunks of 100 000), and from the focal
# spot over the whole arc and its grid (16 chunks of 200 000)
clinical_runs = let
    dexel = MC.simulate_sweep(
        [(dexel_array(depth_mm), b.spectrum, 300 + chunk) for b in clinical_beams for chunk in 1:16];
        n = 100_000,
        beam = crystal_beam,
        readout,
    )
    arc = MC.simulate_sweep(
        [(detector, b.spectrum, 400 + chunk) for b in clinical_beams for chunk in 1:16];
        n = 200_000,
        beam = MC.focal_spot(detector),
        readout = MC.ArcEnergyReadout(),
    )
    (
        dexel = [dexel[(16(k - 1) + 1):(16k)] for k in eachindex(clinical_beams)],
        arc = [arc[(16(k - 1) + 1):(16k)] for k in eachindex(clinical_beams)],
    )
end

# ╔═╡ 83fb054a-b24c-4cba-849d-b3c4b9aca19c
# The long way against the shortcut, for each beam: efficiency and DQE(0) on one dexel, and the
# energy the whole arc collects per photon; each direct value with its noise (the 16 chunks' spread)
clinical_check = let
    E = collect(energies)
    m1 = [d.crystal_keV / d.n for d in one_dexel]
    m2 = [d.crystal2_keV2 / d.n for d in one_dexel]
    cell = MC.energy_absorption_efficiency.(grid_runs.with)
    # the whole arc against one cell, from section 7's own air scans: the fan and cone angles (and the
    # model's arc, not tilted along z) change it a little; interpolated between the scans' energies
    cell_at(x) = MC.interpolate_linear(collect(grid_energies), cell, clamp(x, 20.0, 140.0))
    arc_ratio = [sum(s.energy_keV) / s.photon_keV / cell_at(E) for (E, s) in zip(air_energies, air_scans)]
    ratio_at(x) = MC.interpolate_linear(air_energies, arc_ratio, clamp(x, first(air_energies), last(air_energies)))
    at(table, x) = MC.interpolate_linear(E, table, x)
    noise(values) = sqrt(sum(abs2, values .- mean(values)) / (length(values) - 1) / length(values))
    map(enumerate(clinical_beams)) do (k, b)
        e, w = b.spectrum.energies_keV, b.spectrum.weights
        mean_E = sum(w .* e) / sum(w)
        chunks, arcs = clinical_runs.dexel[k], clinical_runs.arc[k]
        efficiency_of(d) = d.crystal_keV / (d.n * mean_E)
        dqe_of(cs) = sum(c.crystal_keV for c in cs)^2 / (sum(c.n for c in cs) * sum(c.crystal2_keV2 for c in cs))
        arc_of(a) = sum(a.energy_keV) / a.photon_keV
        (
            label = b.label,
            mean_keV = mean_E,
            efficiency = (
                table = sum(w .* e .* at.(Ref(efficiency.flat_field), e)) / sum(w .* e),
                direct = sum(c.crystal_keV for c in chunks) / (sum(c.n for c in chunks) * mean_E),
                noise = noise(efficiency_of.(chunks)),
            ),
            dqe = (
                table = sum(w .* at.(Ref(m1), e))^2 / (sum(w) * sum(w .* at.(Ref(m2), e))),
                direct = dqe_of(chunks),
                noise = noise([dqe_of([c]) for c in chunks]),
            ),
            arc = (
                table = sum(w .* e .* [cell_at(x) * ratio_at(x) for x in e]) / sum(w .* e),
                direct = arc_of(sum(arcs)),
                noise = noise(arc_of.(arcs)),
            ),
        )
    end
end

# ╔═╡ aeef74d7-f3eb-4913-ba02-67b330f63b66
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 560))
    x = 1:length(clinical_check)
    labels = [c.label for c in clinical_check]
    panels = (
        (key = :efficiency, title = "(a) Efficiency, One Dexel", ylabel = "Energy-Weighted Efficiency"),
        (key = :dqe, title = "(b) DQE(0), One Dexel", ylabel = "DQE(0)"),
        (key = :arc, title = "(c) The Whole Arc and Its Grid", ylabel = "Energy Collected per Photon ÷ E"),
    )
    for (k, panel) in enumerate(panels)
        values = [getfield(c, panel.key) for c in clinical_check]
        lo = minimum(min(v.table, v.direct) for v in values)
        hi = maximum(max(v.table, v.direct) for v in values)
        pad = max(0.02, 0.6 * (hi - lo))
        ax = Mke.Axis(
            fig[1, k];
            title = panel.title,
            ylabel = panel.ylabel,
            xticks = (x, labels),
            xticklabelrotation = π / 8,
            limits = ((0.5, length(x) + 0.5), (lo - pad, hi + pad)),
        )
        Mke.scatter!(ax, x .- 0.08, [v.table for v in values]; color = PALETTE.reference, markersize = 16, marker = :diamond, label = "Table, Weighted by the Spectrum")
        Mke.errorbars!(ax, x .+ 0.08, [v.direct for v in values], [3v.noise for v in values]; color = PALETTE.kept, whiskerwidth = 10, linewidth = 2)
        Mke.scatter!(ax, x .+ 0.08, [v.direct for v in values]; color = PALETTE.kept, markersize = 14, label = "Geant4, the Spectrum Itself (± 3σ)")
        k == 1 && Mke.axislegend(ax; position = :lb)
    end
    fig
end

# ╔═╡ 8e4a387c-9e9f-4726-a4c5-9691b23a53c6
let
    rows = String[]
    worst = 0.0
    for c in clinical_check, (name, v, floor) in (("efficiency, one dexel", c.efficiency, 0.002), ("DQE(0), one dexel", c.dqe, 0.003), ("the whole arc", c.arc, 0.01))
        difference = v.direct - v.table
        allowed = 3v.noise + floor
        pass = abs(difference) <= allowed
        worst = max(worst, abs(difference) / allowed)
        push!(rows, "| $(pass ? "✓" : "✗ **fail**") | $(c.label), $(name) | $(round(v.table; digits = 4)) | $(round(v.direct; digits = 4)) ± $(round(v.noise; digits = 4)) | $(round(difference; digits = 4)) | ≤ $(round(allowed; digits = 4)) |")
    end
    passed = count(r -> startswith(r, "| ✓"), rows)
    Markdown.parse(
        """
        | | beam, quantity | table, weighted | Geant4, the spectrum itself | difference | allowed |
        |---|---|---|---|---|---|
        $(join(rows, "\n"))

        **$(passed) of $(length(rows)) agree.** Allowed: three times the direct simulation's own noise, plus 0.002 for the
        efficiency and 0.003 for DQE(0) (the table's noise and its 1-keV grid), and 0.01 for the whole arc, whose prediction is one cell's efficiency
        times the arc's own excess over it, measured in section 7's air scans and interpolated between their energies: so
        this checks the spectrum's weighting, not the arc's geometry, which section 7 reports. Where they agree, the
        single-energy table, weighted by any spectrum, is the clinical beam: what BasisSimulator does with it is exact.
        """
    )
end

# ╔═╡ c7624274-0f6d-5fb6-84d0-3017ee9fc029
md"""
## The Table to Use

The table to replace the v0.15.1 table with (BasisSimulator v0.16.0), and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **UFC at 7.31 g/cm³**, Siemens' measured density, not Geant4's stock 7.44 (section 1).
- **1.4 mm deep.** It is the only UFC thickness Siemens quotes (data sheet, *Detectors &
  Crystals*), BasisSimulator v0.15.1 already documents it, and Siemens' own absorption figure fits it (if
  anything, a slightly thinner layer). The v0.15.1 table's 3.0 mm is XCIST's default depth for a
  different, GE material, with no Siemens source (section 5).
- **BasisSimulator's fill factor, 0.9 × 0.9, unchanged.** The assumed reflector reproduces it, and
  the grid's walls stand on the reflector and shade no ceramic (section 6).

Every other depth from 1.0 to 3.0 mm is saved beside it, so a better source for the depth needs
no new simulation.
"""

# ╔═╡ b4d14e55-550f-5e29-9cf2-a8bec9e358bd
md"""
## Saving the Table

One file in `lib/BasisMonteCarlo/tables/`, in the layout of BasisSimulator's
`UFC_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of the documented
detector (UFC at 7.31 g/cm³, 1.4 mm), beside the one-dexel efficiency, the interaction
probability, the Swank factor and DQE(0). Because the crystal depth is the one number no source
pins to the Force, the flat-field efficiency is also saved for depths from 1.0 to 3.0 mm: a
better-documented depth needs no new simulation. The reproduction of the v0.15.1 table is kept
with it, as the record of how that table was made.
"""

# ╔═╡ d2020133-bce6-5dbc-a2fb-a4f4939c6551
table_depths = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]

# ╔═╡ 455be759-8cc7-5071-a2e3-128159b547be
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

# ╔═╡ 970dda9d-fd51-5930-a052-107a1b262d3e
# The machine path stays out of the rendered page
table_dir = joinpath(@__DIR__, "..", "..", "tables");

# ╔═╡ d27ebbe8-bb50-510d-9fa6-c530fea36d3e
saved = let
    path = joinpath(table_dir, "siemens_force_ufc_efficiency_geant4.jls")
    contents = Dict{String, Any}(
        "energies" => collect(energies),
        "efficiency" => efficiency.flat_field,
        "efficiency_one_dexel" => efficiency.one_dexel,
        "interaction_probability" => efficiency.interaction,
        "swank_factor" => efficiency.swank,
        "dqe0" => efficiency.dqe,
        "depths_mm" => table_depths,
        "efficiency_by_depth" => MC.energy_absorption_efficiency.(by_depth),
        "reproduced_table" => reproduced,
        "reproduced_depth_mm" => fit.depth_mm,
        "reproduced_scale" => fit.scale,
        "compared_with" => "BasisSimulator v0.15.1, UFC_MC_EFFICIENCY_LUT (BasisMonteCarlo.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc)",
        "decision" => "use `efficiency`: flat field, UFC at 7.31 g/cm3, 1.4 mm deep (the only UFC thickness Siemens quotes); BasisSimulator's fill factor 0.9 x 0.9 unchanged",
        "definition" => "mean energy collected per photon landing on a dexel's ceramic with every dexel illuminated (flat field), divided by the photon energy; BasisSimulator's fill factor applies separately",
        "detector" => Dict(
            "scanner" => "Siemens SOMATOM Force (Stellar Infinity)",
            "material" => "UFC, Gd2O2S, 7.31 g/cm3",
            "depth_mm" => depth_mm,
            "pitch_mm" => pitch_mm,
            "reflector" => "0.1 mm TiO2-loaded epoxy (assumed)",
            "grid" => "2-D tungsten, 0.08 mm walls, 10 mm tall (assumed); no effect on the flat field",
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 04_siemens_force.jl",
    )
    Serialization.serialize(path, contents)
    path
end;

# ╔═╡ c98272cf-d431-5930-b4d9-4821b38d9bcc
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

# ╔═╡ 6081b331-acd9-59db-ab45-7b26d0e79e7a
md"""
## Where This Leaves the Force
"""

# ╔═╡ 9765af71-bc8c-5467-8629-c348274f1331
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    Markdown.parse(
        """
        - **The material is right, the density 2 % high.** UFC is Gd₂O₂S at 7.31 g/cm³ (Siemens' data sheet);
          BasisSimulator v0.15.1 uses Geant4's stock 7.44. Geant4 at 7.31 g/cm³ and 1.4 mm lands on Siemens' own absorption
          curve up to 90 keV.
        - **BasisSimulator v0.15.1's table is not a 1.4 mm detector.** Above about $(Int(first(collect(energies)[table_at.(collect(energies)) .> [1 - exp(-MC.geant4_attenuation(ufc_basis, e) * depth_mm) for e in energies] .+ 0.01]))) keV it exceeds what 1.4 mm
          can absorb at all. It is reproduced, to about 0.01, by a pencil beam into one dexel of Gd₂O₂S
          **$(fit.depth_mm) mm deep**, tallied in that dexel.
        - **What a scan sees** is the flat-field efficiency: each dexel gets back from its neighbours the Gd x rays it
          loses to them. For the documented 1.4 mm detector it is $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, against the
          table's $(round(m120.table; digits = 3)) ($(round(100(m120.geant4 / m120.table - 1); digits = 1)) %), with DQE(0) $(round(m120.dqe; digits = 3)).
        - **The grid** (2-D, tungsten) costs the photons from the focal spot almost nothing and stops most scatter;
          **the whole detector** collects what one dexel's cell does.
        - **The beam:** the installed BasisSimulator's 120 kV Force beam has a half-value layer of $(round(hvl_check[1].hvl; digits = 1)) mm against Siemens' 7.8 mm.
        - **Saved:** `$(basename(saved))`, with the efficiency for depths 1.0–3.0 mm: the table for BasisSimulator v0.16.0.
        """
    )
end

# ╔═╡ Cell order:
# ╟─27f245eb-99f2-5e57-b186-46b2371b7d17
# ╟─726d09a1-345c-46aa-a33e-dca44249ed7d
# ╟─bc5428b3-9cde-5d59-ad60-0b21685419fd
# ╟─cce76b53-ce59-4546-b518-ff1b5ba68111
# ╟─e2e24ca7-420b-42c3-b46d-f7c4d94a2191
# ╟─0128955f-7935-42d8-93e3-b27320625add
# ╟─9b48b777-a6c9-4ca9-94a6-dfbd06e96922
# ╟─c8bd4fe5-c38c-5d02-ab51-8d1653cd5d74
# ╟─5ac2b273-dfac-5d7e-b717-60bb42ac1686
# ╠═5db002c4-197d-59f8-b950-3eb200999f19
# ╠═72bdb567-8286-5781-9549-ed9f1896c3ce
# ╠═edc8017e-fd91-559c-b8fa-30240d4f641a
# ╠═ab6d595c-d31b-595b-8c53-edd04f3f12dc
# ╠═398c3256-d0b6-5fb2-a912-4a20e1a5c480
# ╠═015aea26-390f-5615-b59d-a5ff35887bad
# ╠═bfdc3a20-afc8-551d-bc8a-579adc960b43
# ╠═254434ab-51b5-5f29-9c83-d2bb1aa7bb94
# ╠═4c28b150-8255-5c6f-9ffa-004001712d76
# ╠═e28b3797-8ff8-5a0a-a57c-d9e2d995032a
# ╠═36170b1a-5d31-5c5d-839b-769f9b4dd828
# ╠═f41ccaf7-dea7-53b9-a121-831b7efd7e8f
# ╠═d3d0e23d-9e6b-59c5-a623-8ab89195054a
# ╠═981bfa07-0f0d-5879-a3b7-c7cc0991b59c
# ╠═73add7d1-3bf8-5ed8-8ae5-42694781870c
# ╟─d38babb6-4cd9-4f26-89ff-477edf8efbbe
# ╠═ac3eb9fc-ef3e-4bc3-9d39-de31b283a49b
# ╠═67d71709-d954-445c-bec4-a70a8e6674af
# ╠═df05ae3e-7c49-4f28-8ced-7f393b17e84e
# ╠═a801cc4b-a4ed-40d3-a9dc-e0384d9df317
# ╠═e08d1685-09ba-4e1e-8d49-e4e0bc587c7d
# ╠═a3583f6c-07a9-4209-9b74-1e80267e6a3a
# ╠═0c307760-1a4c-444e-9054-8783e59a1f99
# ╠═b44d674a-9303-4ed0-8ba6-ce98b6540914
# ╠═61fe6fa1-21ad-4f73-82af-5bcb53e2e9fd
# ╠═853df9a6-649b-43b4-9985-c22ab54fbf7b
# ╠═b32e7f50-09f9-4bfa-b83d-db839728ebf1
# ╠═c97c8bf2-7c91-499d-9666-029a6cfc9b88
# ╠═19b80dcc-92d0-4603-9f0f-7046f2309772
# ╠═610bf587-f0c7-40ad-b75c-b7e26f5f6852
# ╠═e47f6b0c-b1ef-4df3-a14e-6bb9d4d60265
# ╠═a2e15617-ca67-4e0e-a785-877170fdad02
# ╠═7eeebb8c-8e88-4be3-8a3e-06de190d9caa
# ╠═f86c4c2f-bcdf-419c-937a-44f624c147c0
# ╠═cc8677b7-4293-49ce-a3ed-684f1681cad4
# ╠═0b8077c2-d5ef-4ac5-8f1b-f2fdb10de967
# ╠═0d5a87a3-1bbb-42a7-b5de-39d9c1cb7f35
# ╠═e09df38a-a838-43e6-9fde-b1d8690fc7ed
# ╠═6ee68941-6f2d-44f4-a7bb-93a56150ac54
# ╠═3b9b3237-0eb0-4005-9525-8852acfb4254
# ╠═c55ace1d-63db-4f39-a0c8-385dbbd2dff2
# ╠═7de77732-a5e4-4d99-959a-1110568c6307
# ╠═49ad6340-31e5-4ee3-b218-d065efb7dfda
# ╠═410ac026-8a00-4b5a-b890-e8615c11d0a1
# ╟─bb6b4817-3d18-5fc6-8870-fcf6d067c1b7
# ╠═5b0821f5-5ede-5654-bf41-aae557bf750f
# ╠═26cd0fa9-e926-5418-b39e-828104ebe7c0
# ╠═47ba3d48-19e8-5503-8148-1f1ef9940bc9
# ╟─2d6fc883-fbf8-5a9e-bec4-bcdffd596d03
# ╠═599f755d-68f8-559f-ae1a-44e3fec49c79
# ╠═de91f61c-0c20-5dc1-9fa1-b786127b99ba
# ╠═13711387-8deb-5b18-84c1-1fb6dcc5fc07
# ╠═e158b4ca-de06-5340-bf46-ed84dd3f5168
# ╟─3af4e48b-73ae-4a03-bdc6-e0b3cad7ef2b
# ╟─57dc760e-1303-51dc-b34e-a7fba0133b0c
# ╟─e8bfee76-4803-5f45-81ff-2b3f273e3c90
# ╟─7364ef77-5601-548b-ae52-f2587cfc86d9
# ╠═b6077caf-5639-5dd5-810f-47dbc8afa43b
# ╠═6e6efe87-7cfc-56af-ae92-eb629fb123ca
# ╠═2e82befa-95ec-5581-9c0e-a893ef543fc6
# ╟─33144218-4d14-4143-8255-1ee267347282
# ╟─070fc321-f721-51dc-bbb7-aebc18a576bf
# ╟─984509ba-dcb0-5de2-af45-34c1ee30eb3f
# ╟─624c50f1-16e9-537a-80a2-88b0ec6d5bf1
# ╠═b87d28f5-35b8-59c9-9eaa-56929df2e1a5
# ╠═0d218946-d0ed-50b8-a347-b76f53bd8a9f
# ╠═86d8a811-3df1-5cec-90db-e3d1161f674a
# ╠═2597aa0b-31a5-5e2f-acea-8a08b89e189f
# ╠═68bb39d1-7bd1-5732-8d9e-3143dbd4e31b
# ╟─8068a25b-4729-577e-8b7a-8198ea882d7e
# ╠═a7e2f1ed-a64b-5798-962b-977f40251001
# ╠═8f7fd8fb-bad6-5f1b-a112-9244837b96b4
# ╠═135302df-8c55-5616-b4b0-c7a92e8bd37f
# ╠═1d0e1508-6800-5211-9822-7bcaefdc76f6
# ╟─f8d455e6-a0c3-585c-a5de-e67e7495e0e1
# ╟─00b453d7-b568-5e07-97f4-b5e7fea4b6eb
# ╠═3d3d9f03-1438-5fc6-b252-f6a016530cd6
# ╠═5a19b142-fa62-5a20-8c3e-fd1b2caba4ea
# ╟─47825cfd-bde9-5b63-82f1-846000c79a0c
# ╟─207da70b-5e07-4fc4-8dfe-de974aec8542
# ╟─9f07fe25-f60a-4976-9252-b06ccba96386
# ╟─9136f582-c77e-5133-9eb9-1c5c62370702
# ╠═e5779ca2-7b7b-5cac-b9ea-f5246af64ac3
# ╠═e81d468a-4e95-51a8-b3f8-417d0783c6d9
# ╟─84ba9e1d-40c0-4b7e-b634-401e025e07c1
# ╟─086a9000-05cf-54f7-bd40-160ea222808b
# ╟─d8c74acc-320a-5454-8c8e-4fc993879dca
# ╠═fae36d57-1538-5767-8a86-89f4d674ee92
# ╠═fbd77aa1-7f9a-5f31-8fa5-fe8bb28c1b61
# ╟─b03a2cc1-a6fa-495b-99aa-fdbed493bd66
# ╟─709be125-f9e1-5fd9-ada9-0ee542722e10
# ╟─96411072-2914-50b3-90a8-9a155e951b38
# ╟─1aff4f4a-5c08-5668-a3a9-408158142c6a
# ╠═c2344ff9-4d61-521d-bde9-fe85b774d1bd
# ╠═473d2874-dedb-57bc-a509-60c0203d1761
# ╠═0a89ea01-6e3a-53c1-b61a-cb5387ed2854
# ╟─8b14052c-6b34-4f30-9a33-429fecf79713
# ╟─a5da9355-cf65-5dbf-98e7-d6610831b32c
# ╟─2783d5ea-67a1-5806-9769-6ba444eac9e6
# ╟─1e7c8688-b175-523d-8b2b-c3db2965e66b
# ╠═ccd1b408-75f5-55b1-a644-2249e8800179
# ╟─6b9cb578-7aef-48d4-a9b3-60077533fc6d
# ╟─faa33039-ae00-56d9-ae15-e2b2e31fa9a3
# ╟─d04a5179-567e-5af2-8d05-57a4f1934c55
# ╠═8aba61c1-6b2d-5111-83b8-4a85a8054edc
# ╟─68de6046-7b40-4d32-96af-a2432fbdf672
# ╠═5cd42bd3-b86a-5ee8-90de-ffdf30d6b107
# ╟─619811db-f256-4809-999c-9d7ca805b284
# ╟─251a2c3a-47e5-5256-9c78-ad2d15c53226
# ╟─8f182220-5099-5d7d-b26d-ec7650a032b6
# ╟─d4d258d8-a0ff-5e91-bb06-e18dde132184
# ╠═b71c3c63-f028-530c-b506-7c2f44e4e55c
# ╠═9f38d967-6625-5bdb-bf78-a28648932329
# ╠═81fb4130-1d0e-5c57-b642-866f51f38a19
# ╠═42a87db7-35b4-5bd0-a3ad-8fafe9a0fa49
# ╠═4a8900c3-6ec6-558c-b0a3-d8c9b26dc0fe
# ╟─38c8c86b-d27a-5ef5-87ee-05ef7abc55ee
# ╠═d36811c9-01e7-5107-8359-46bc751cad2c
# ╠═b7f6dca9-d404-5a4c-a45b-32ba0b461002
# ╟─ea31ed85-c69c-5e35-9d41-5927c317bee3
# ╟─b8473053-0084-5d1c-b467-2e3ca8edaa07
# ╟─71a9bf29-a1f9-57eb-ab25-5be4792b7998
# ╠═76bf4fed-52bd-57c9-a3a1-c7b001617d3f
# ╟─6b445b19-7204-5976-8445-5845677718c6
# ╟─af1722c8-f3ff-5d4f-9025-ee5b13e6be0d
# ╠═cb12b286-ac74-5e3b-bc12-8c34c185c197
# ╠═57591bb0-24bc-53ea-b228-25b0571db01f
# ╠═13d20760-ef7e-5049-b1e9-f732c752aa18
# ╠═146cfad9-9898-589e-aad6-611c73f14d5e
# ╟─e57ccb84-f6c5-595d-bf5b-e001a1bd0226
# ╟─92c3a865-32c1-59f6-ba69-836443f3e8fe
# ╟─b802d7eb-f757-5496-a7cc-0653fc3df872
# ╠═8c5a6f70-f994-54e7-9d70-ebca20449ebd
# ╠═1bbe17d8-933e-51a9-a6cd-b7babce22957
# ╠═1ee36aa2-5d50-5b55-94da-2b5234f8c818
# ╟─03ab3733-cb26-4085-80df-a344298a26fb
# ╟─39f15a81-ff31-5ef6-acd6-23a32fa8b0b0
# ╟─b13ffa4b-aa10-5428-ab58-f989dc4015db
# ╟─0827c6a3-5316-565c-8c37-0cf97d73a0ef
# ╠═7cecf8ed-e83b-5b37-8b05-cd529f4575e4
# ╠═08f77ac1-c4b3-51bb-bb31-0a12c45749d7
# ╠═2a8e6eb5-7861-52a8-ba63-00aa89219d07
# ╟─117159e7-d9ee-4ebc-be5f-3e24793241d3
# ╟─b2e013dd-2bc7-586d-99f3-0a91839cc767
# ╟─112e15a2-75dd-53d6-92b8-511aad50fc7a
# ╟─ee9b4076-8344-5e14-a36c-b00985bdcdb6
# ╠═9061fc53-ea2f-5cdf-9b1e-8bdf1959e27e
# ╟─6ed7c91d-53b6-48e0-b71d-113621a4576d
# ╟─e3e24f7d-3d7c-50fe-8569-340354b4e66c
# ╟─8afa396b-e510-5410-ab20-58f29489aefe
# ╟─807fa3c3-543c-55b6-b4cf-17f8a7415a58
# ╠═104874e1-f75a-5c48-8023-b03c1104a754
# ╠═1770b852-19ac-5728-86cb-7419fbd8af32
# ╟─c6e2851e-23f2-4ea1-86c6-9d9f959b794f
# ╟─69f8b2ab-1204-588b-a121-408d60b26374
# ╟─f0dfceab-4c51-5307-8cf8-48720f5babef
# ╟─fde8bf46-2e00-4a50-91f1-a212e816640f
# ╟─fa6ed8d8-c9bc-4c81-874b-7096b8840239
# ╠═a98d98ef-b4d5-452c-9f38-07e9e885c3d8
# ╠═448c2bf9-de8a-4f88-9e56-464711960b56
# ╠═83fb054a-b24c-4cba-849d-b3c4b9aca19c
# ╟─aeef74d7-f3eb-4913-ba02-67b330f63b66
# ╟─8e4a387c-9e9f-4726-a4c5-9691b23a53c6
# ╟─c7624274-0f6d-5fb6-84d0-3017ee9fc029
# ╟─b4d14e55-550f-5e29-9cf2-a8bec9e358bd
# ╠═d2020133-bce6-5dbc-a2fb-a4f4939c6551
# ╠═455be759-8cc7-5071-a2e3-128159b547be
# ╠═970dda9d-fd51-5930-a052-107a1b262d3e
# ╠═d27ebbe8-bb50-510d-9fa6-c530fea36d3e
# ╠═c98272cf-d431-5930-b4d9-4821b38d9bcc
# ╟─6081b331-acd9-59db-ab45-7b26d0e79e7a
# ╟─9765af71-bc8c-5467-8629-c348274f1331
