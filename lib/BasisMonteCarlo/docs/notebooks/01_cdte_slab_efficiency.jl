### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 09d7b527-61f3-5c01-a16a-d4178d4d985a
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ db1d8de8-d6dc-5361-bce7-7948e59da8bc
using Markdown: @md_str, Markdown

# ╔═╡ 8296a665-0099-56bd-85fd-89eba76fb807
md"""
# Detector Efficiency from First Principles: a CdTe Slab in Geant4

A CT simulator has to know, for every photon energy, how much of each photon the detector
actually records. BasisSimulator describes this with an efficiency curve η(E). The simplest
version, Beer-Lambert, says a photon counts if it interacts in the crystal at all:
`η = 1 − exp(−μ(E)·d)`. A real detector loses some of what it stops. Just above a K-edge the
absorbed photon knocks an electron out of the innermost (K) shell of an atom; the atom refills
that shell by emitting an x ray of a fixed, characteristic energy, and that x ray can fly
straight back out of the crystal, taking a fixed piece of the energy with it.

Monte Carlo measures this without building the detector: fire photons one at a time, follow
every particle they create through the real physics, and add up what stays. This notebook does
that with Geant4 (through Geant4.jl) on the simplest detector there is, a single slab of
cadmium telluride (CdTe), and checks each step against something known.

1. **The detector and the beam.** What is simulated, drawn to scale.
2. **The physics.** How strongly CdTe stops x rays, process by process, as Geant4 computes it.
3. **Following photons.** Every step of a few dozen photons: where they stop, where the
   characteristic x rays go.
4. **Check 1: the Monte Carlo reproduces Beer-Lambert.** The fraction of photons that interact
   must equal `1 − exp(−μd)` with Geant4's own μ, to within counting noise.
5. **Where the energy goes.** The energy budget per photon (it must add up exactly) and the
   spectrum of energy left in the crystal, with its escape peaks.
6. **The efficiency curves.** What the detector keeps, against what Beer-Lambert predicts,
   across the CT energy range.
"""

# ╔═╡ 217c5b00-469a-5a92-bd6e-da4508b34921
md"""
## How the Notebook Works

Every number in the notebook comes from Geant4. NIST XCOM (through XrayAttenuation.jl, the
attenuation data BasisSimulator uses) appears only as a comparison, so the size of the
difference between the two standard data sets is visible wherever it matters. The figure below
is the whole experiment: what happens to the photons, what Geant4 follows, and what it records.
"""

# ╔═╡ 41e976eb-d5a6-5afd-9c35-4670601ce1ec
md"""
## Notebook Setup
"""

# ╔═╡ 3895df79-1afe-5296-b04c-a0995fb14057
import PlutoUI

# ╔═╡ de195e86-864c-51a4-9372-162e52f23278
import BasisMonteCarlo as MC

# ╔═╡ f0468c43-be4c-5dc1-816e-66eb7e24e1b0
import CairoMakie as Mke

# ╔═╡ 61babc92-4726-5e99-aef4-dde0339aa83b
PlutoUI.TableOfContents()

# ╔═╡ 85de25c3-beb0-590e-b45f-e3e76305c57b
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 74dc1d9e-d900-5f7c-8825-1d332f6a89b3
# One colour per kind of particle, used by every figure
PALETTE = (
    photon = Mke.RGBf(0.13, 0.36, 0.66),  # the incident photon
    xray = Mke.RGBf(0.9, 0.45, 0.05),  # characteristic (fluorescence) x rays
    electron = Mke.RGBf(0.7, 0.12, 0.17),  # electrons set free in the crystal
    kept = Mke.RGBf(0.27, 0.55, 0.34),  # energy that stays in the crystal
    slab = Mke.RGBAf(0.45, 0.52, 0.62, 0.3),
    slab_edge = Mke.RGBf(0.3, 0.36, 0.44),
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ 3b6e9621-cbce-56e9-949f-dd3562147bff
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

# ╔═╡ 5254847b-5c45-5e0c-ab5f-2b6872c13c47
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

# ╔═╡ 3af2ff68-c9d8-4cf6-bfc5-bf391296da68
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses, with a small kit: blocks
in 3-D seen from any direction (shaded by one light, drawn back to front), photons as waves, and
labels on translucent backgrounds. Their colours are the plots' colours: blue for the incoming
photon, orange for characteristic x rays, dark red for electrons, green for the energy that stays.
"""

# ╔═╡ 822095ae-24c5-40b5-a2f6-e25dfb90a015
# The crystal and the beam, as the illustrations draw them: the same colours as the plots
MATERIALS = (
    cdte = Mke.RGBf(0.8, 0.83, 0.89),  # the CdTe crystal
    beam = PALETTE.photon,  # the incoming photons
    star = Mke.RGBf(0.98, 0.95, 0.6),  # where a photon is absorbed
)

# ╔═╡ e9cc638b-caf5-48a4-bd2c-b0fe0e1e2195
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

# ╔═╡ 87e524c6-a58c-4c90-918d-1e929e0e5f6a
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

# ╔═╡ 09afc8c3-67f0-4506-8242-a569104d1515
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ 443b26b9-5365-4eb1-b73d-4228fd3ceb37
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ 49342efd-5bbd-41e6-b871-e992c910404a
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

# ╔═╡ cfba4ed7-6c5e-429d-8361-aa2acb343b1d
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ bdb6e526-af68-4675-b232-2d2d75de03c8
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

# ╔═╡ 3bcdc764-6d39-4d03-9095-029419302112
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

# ╔═╡ baaf379b-4d53-49a5-87d5-28b7cb166e85
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

# ╔═╡ 50a0f3b5-d39f-4e9c-8161-01961a98fee3
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ 7c861316-ce0b-4f55-93a1-e41b8980cd71
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ 3333fc70-8b77-4c2b-b8e8-ee6b8baff691
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

# ╔═╡ cdbf1522-f34f-4563-80b8-ce2cae20c67f
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

# ╔═╡ c7b30aa6-8a09-4b4d-ac6b-1ea4bce5a6f1
TAGS = (geant4 = Mke.RGBf(0.2, 0.55, 0.32), not_simulated = Mke.RGBf(0.52, 0.52, 0.57))

# ╔═╡ dfc666d5-5ff2-502e-aeb0-49439685ac86
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 600))

    # (a) The slab from the side (not to scale): four photons, four fates
    ax = canvas(fig[1, 1]; title = "What Geant4 Follows: Every Photon, X Ray and Electron", titlesize = 20)
    rect!(ax, 0, -2.2, 2.0, 4.4, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 2)
    Mke.text!(ax, 1.0, 2.3; text = "CdTe", align = (:center, :bottom), fontsize = 19, font = :bold, color = PALETTE.slab_edge)
    star!(p) = Mke.scatter!(ax, [Mke.Point2f(p)]; marker = :star5, markersize = 24, color = MATERIALS.star, strokecolor = PALETTE.photon, strokewidth = 1.2)

    # 1: absorbed; its x ray is absorbed again inside: all the energy stays
    photon!(ax, (-2.4, 1.5), (0.55, 1.5); cycles = 7, amplitude = 0.018)
    photon!(ax, (0.55, 1.5), (1.35, 0.95); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 10)
    Mke.scatter!(ax, [Mke.Point2f(1.38, 0.93)]; color = PALETTE.electron, markersize = 10)
    star!((0.55, 1.5))
    # 2: absorbed near the front; its x ray escapes back out
    photon!(ax, (-2.4, 0.5), (0.3, 0.5); cycles = 7, amplitude = 0.018)
    photon!(ax, (0.3, 0.5), (-1.6, 1.0); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 11)
    star!((0.3, 0.5))
    # 3: Compton scatter: the photon leaves through the back, weaker, and frees an electron
    photon!(ax, (-2.4, -0.5), (0.9, -0.5); cycles = 7, amplitude = 0.018)
    photon!(ax, (0.9, -0.5), (3.7, -1.25); cycles = 4, amplitude = 0.04, head = 11)
    Mke.lines!(ax, Mke.Point2f[(0.9, -0.5), (1.0, -0.33)]; color = PALETTE.electron, linewidth = 3)
    Mke.scatter!(ax, [Mke.Point2f(0.9, -0.5)]; marker = :diamond, markersize = 14, color = PALETTE.photon, strokecolor = :white, strokewidth = 1.5)
    # 4: no interaction at all
    photon!(ax, (-2.4, -1.6), (3.7, -1.6); cycles = 12, amplitude = 0.012)

    labels = (
        ((-2.45, 1.5), "Absorbed; Its X Ray Too: All of It Stays", PALETTE.kept),
        ((-2.45, 0.5), "Its X Ray Escapes Back Out", PALETTE.xray),
        ((-2.45, -0.5), "Bounces Off, Leaves Through the Back", PALETTE.photon),
        ((-2.45, -1.6), "Passes Straight Through", PALETTE.muted),
    )
    for (at, text, color) in labels
        Mke.text!(ax, at; text, color, fontsize = 16, font = :bold, align = (:right, :center))
    end
    Mke.limits!(ax, -7.4, 4.0, -2.6, 2.9)

    # (b) What the simulation records for every photon, and what it leaves out
    axb = canvas(fig[1, 2]; title = "What Is Recorded, Photon by Photon", titlesize = 20)
    rows = (
        ("Energy Left in the Crystal", "The Signal: What a Detector Can Read Out", true),
        ("Every Particle That Leaves", "What It Is, Its Energy, Front or Back Face", true),
        ("Whether It Interacted at All", "For the Check Against Beer–Lambert", true),
        ("Pixels, Charge, Thresholds", "A Bare Crystal Here: the Electronics\nCome in Notebook 03", false),
    )
    for (k, (title, detail, simulated)) in enumerate(rows)
        y = 10 - 2.4k
        tag!(axb, Mke.Point2f(0, y), simulated ? "Recorded" : "Not Simulated", simulated ? TAGS.geant4 : TAGS.not_simulated)
        Mke.text!(axb, 2.6, y; text = title, font = :bold, fontsize = 18, color = PALETTE.ink, align = (:left, :bottom), offset = (0, 1))
        Mke.text!(axb, 2.6, y; text = detail, fontsize = 15.5, color = PALETTE.muted, align = (:left, :top), offset = (0, -3))
    end
    Mke.limits!(axb, -0.2, 11, -0.6, 9.4)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.6))
    fig
end

# ╔═╡ cd987c45-a0c4-47d9-ae2b-d263299c8a93
"""A Gd atom: nucleus and two shells, at `c` with outer radius `r`."""
function atom!(ax, c, r; color = Mke.RGBf(0.45, 0.47, 0.55))
    θ = range(0, 2π; length = 100)
    for (k, rr) in enumerate((0.55r, r))
        Mke.lines!(ax, c[1] .+ rr .* cos.(θ), c[2] .+ rr .* sin.(θ); color = (color, 0.55), linewidth = 1.5)
    end
    Mke.scatter!(ax, [Mke.Point2f(c)]; color, markersize = 18)
    return nothing
end

# ╔═╡ 9930db7b-c02f-5cce-8043-57033d7852f6
md"""
## 1. The Detector and the Beam

The detector is one block of cadmium telluride, the crystal photon-counting CT scanners use,
1 mm deep along the beam and 50 mm square across it, in vacuum. A narrow ("pencil") beam of
photons, all of the same energy, hits the centre of its front face head on.

**Which beam, and why.** A pencil beam is enough here, and exact: the slab is uniform and 50 mm
wide, and nothing travels more than a few millimetres in CdTe, so a photon landing anywhere on
the face sees the same crystal around it. A pencil beam into the centre gives what a uniform beam
over the whole face would, for the slab as one detector. Pixels, their edges and their neighbours
are not part of this notebook; notebook 03 builds them, and there the beam matters.

The coordinates used by every figure and every number below: **z** runs along the beam and
photons travel toward +z; the front (entrance) face is at **z = 0** and the back face at
**z = d**; **x** and **y** run across the face. The 50 mm face is far wider than anything
travels in CdTe, so nothing leaks out of the sides and the only ways out are back through the
front face or on through the back face.
"""

# ╔═╡ 1778e101-f9ac-579b-9782-dc12f803a8d9
detector = MC.SlabDetector(:CdTe, 1.0)

# ╔═╡ 9fb6be41-9274-55b0-a8dc-193e709361fc
beam = MC.PencilBeam()

# ╔═╡ fea6f5fe-ec6d-514d-973f-1c6f6cbd2393
let
    m = detector.material
    row(e) = "| $(e.element) | $(round(e.edge_keV; digits = 2)) keV | $(round(e.kalpha_keV; digits = 2)) keV | $(round(Int, 100 * e.yield)) % |"
    rows = [row(e) for e in MC.k_edges(m)]

    Markdown.parse(
        """
        **Material:** $(m.name), density $(m.density) g/cm³, by mass $(MC.composition_string(m)).

        | element | K-edge: photons above this can knock out a K electron | energy of the x ray the atom then emits (Kα) | chance it emits that x ray rather than an electron |
        |---|---|---|---|
        $(join(rows, "\n"))
        """
    )
end

# ╔═╡ e5aaa948-30a7-5bd6-82e0-f1ad5b3a275a
let
    Mke.set_theme!(THEME)

    d = detector.thickness_mm
    w = detector.width_mm
    fig = Mke.Figure(size = (1300, 640))

    # (a) The whole slab, to scale, the beam arriving along z from the left
    ax3 = canvas(fig[1, 1]; title = "(a) The Whole Detector, to Scale", titlesize = 21)
    cam = camera(-148, 20)
    slab = Block((0, -w / 2, -w / 2), (d, w / 2, w / 2), MATERIALS.cdte)
    start, face = (-45.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    draw_blocks!(ax3, cam, [slab]; extras = [(1.0e4, a -> arrow!(a, project(cam, start), project(cam, face); color = PALETTE.photon, linewidth = 4, head = 16))])
    callout!(ax3, project(cam, (d, w / 2, w / 2)), project(cam, (d, w / 2, w / 2)) + Mke.Point2f(6, 8), "CdTe Crystal: $(round(Int, w)) × $(round(Int, w)) mm,\n$(d) mm Deep, in Vacuum"; color = PALETTE.slab_edge, fontsize = 16)
    callout!(ax3, project(cam, (-30.0, 0.0, 0.0)), project(cam, (-30.0, 0.0, 0.0)) + Mke.Point2f(10, -12), "Pencil Beam: Every Photon\nHas the Same Energy"; color = PALETTE.photon, fontsize = 16, align = (:left, :center))

    # (b) A cut through the beam axis, zoomed onto the crystal's depth
    ax = Mke.Axis(
        fig[1, 2];
        title = "(b) Side View Through the Beam, Zoomed In",
        xlabel = "Depth Along the Beam, z (mm)",
        ylabel = "Position Across the Face, x (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.7, d + 1.7), (-1.6, 1.5)),
    )
    Mke.poly!(ax, Mke.Rect2f(0, -1.6, d, 3.1); color = MATERIALS.cdte, strokecolor = PALETTE.slab_edge, strokewidth = 2)
    Mke.arrows2d!(ax, [Mke.Point2f(-1.6, 0)], [Mke.Vec2f(1.56, 0)]; color = PALETTE.photon, shaftwidth = 5, tipwidth = 18, tiplength = 18)
    Mke.bracket!(ax, 0, -1.0, d, -1.0; text = "d = $(d) mm", orientation = :down, fontsize = 17, color = PALETTE.ink, textcolor = PALETTE.ink)
    note!(ax, -0.82, 0.4, "Pencil Beam"; align = (:center, :center), color = PALETTE.photon)
    note!(ax, d / 2, 1.15, "CdTe"; align = (:center, :center), fontsize = 20, color = PALETTE.slab_edge, alpha = 0.0)
    note!(ax, 0, -0.35, "Front Face\nz = 0"; align = (:right, :center), offset = (-8, 0), fontsize = 15)
    note!(ax, d, -0.35, "Back Face\nz = d"; align = (:left, :center), offset = (8, 0), fontsize = 15)
    note!(ax, -1.65, -1.55, "Vacuum"; align = (:left, :bottom), color = PALETTE.muted, fontsize = 15)
    note!(ax, d + 1.65, -1.55, "Vacuum"; align = (:right, :bottom), color = PALETTE.muted, fontsize = 15)
    note!(ax, d / 2, 0.55, "The Crystal Goes On\nto x = ±$(round(Int, w / 2)) mm"; align = (:center, :center), color = PALETTE.muted, fontsize = 14, alpha = 0.0)

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.5))
    Mke.colgap!(fig.layout, 40)
    fig
end

# ╔═╡ bcf70aa4-58b9-522c-880f-e1888896c935
md"""
## 2. The Physics

Three things can happen to an x-ray photon in CdTe at CT energies. Their strengths, written as
attenuation coefficients μ (the chance of the interaction per millimetre travelled), add up to
the total μ that Beer-Lambert uses:

- **Photoelectric absorption: the photon is absorbed.** An electron of an atom takes all its
  energy and is set free; the atom then refills the empty shell by emitting electrons (which
  stop within micrometres) or a characteristic x ray (which can travel a millimetre). This
  dominates in CdTe at every CT energy, and it jumps up at each **K-edge**, the energy above
  which the photon can free an electron from the innermost (K) shell.
- **Compton scattering: the photon bounces off an electron and loses part of its energy**, then
  carries on in a new direction and can leave the crystal.
- **Rayleigh scattering: the photon bounces off a whole atom and keeps its energy.**

The solid curves are Geant4's own coefficients, from the models the simulation samples. The
dashed curves are NIST XCOM, for comparison; the lower panel shows how far apart the two data
sets are. They agree to about 1 % for absorption and Compton scattering. Rayleigh scattering
differs by up to about 25 %: Geant4's Livermore model includes the drop in coherent scattering
just below and at each K-edge (the atom's K electrons stop scattering in step with the photon),
which XCOM's tables leave out. Rayleigh scattering is a small part of the total, so the total
moves by less than 2 %, and only right at the edges.
"""

# ╔═╡ 4807e458-f107-5647-bc1a-1bf891280843
fine_energies = 10.0:0.25:150.0

# ╔═╡ befb271a-8286-565b-8512-c80c1aef4eed
mu_geant4 = Dict(
    kind => [MC.geant4_attenuation(detector.material, E; kind) for E in fine_energies]
        for kind in (:total, :photoelectric, :compton, :rayleigh)
)

# ╔═╡ cd2de766-2eac-5fc7-9b9a-8764e0e80e07
mu_xcom = Dict(
    kind => [MC.linear_attenuation(detector.material, E; kind) for E in fine_energies]
        for kind in (:total, :photoelectric, :compton, :rayleigh)
)

# ╔═╡ 96b43d9c-786f-4379-a7fc-18bcca4238ea
let
    Mke.set_theme!(THEME)
    μ_per_mm(E) = MC.interpolate_linear(collect(fine_energies), mu_geant4[:total], E)
    edges = MC.k_edges(detector.material)
    lines = join(["$(round(e.kalpha_keV; digits = 1))" for e in edges], " and ")
    fig = Mke.Figure(size = (1700, 500))
    panels = (
        ("Photon Absorbed", "Photoelectric: It Vanishes; an Electron Leaves,\nand Often a Cd or Te X Ray ($(lines) keV)", PALETTE.electron),
        ("Bounces Off, Loses Energy", "Compton: a Weaker Photon Leaves,\nan Electron Recoils", PALETTE.photon),
        ("Bounces Off, Keeps Energy", "Rayleigh: Same Energy,\na Small Change of Direction", PALETTE.muted),
    )
    for (k, (title, detail, color)) in enumerate(panels)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 21)
        Mke.limits!(ax, -1.25, 1.25, -1.05, 0.95)
        atom!(ax, (0, 0), 0.32)
        photon!(ax, (-1.2, 0.0), (-0.36, 0.0); cycles = 6, amplitude = 0.07)
        if k == 1
            arrow!(ax, (0.12, 0.1), (0.62, 0.55); color = PALETTE.electron, linewidth = 3)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = PALETTE.electron, fontsize = 16, align = (:left, :bottom))
            photon!(ax, (0.2, -0.24), (1.0, -0.62); cycles = 4, amplitude = 0.07, color = PALETTE.xray)
            Mke.text!(ax, 0.98, -0.75; text = "Cd or Te X Ray", color = PALETTE.xray, fontsize = 16, align = (:right, :top))
        elseif k == 2
            photon!(ax, (0.3, -0.18), (1.1, -0.72); cycles = 3.5, amplitude = 0.07)
            Mke.text!(ax, 1.05, -0.86; text = "Weaker Photon", color = PALETTE.photon, fontsize = 16, align = (:right, :top))
            arrow!(ax, (0.2, 0.2), (0.62, 0.55); color = PALETTE.electron, linewidth = 2.5)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = PALETTE.electron, fontsize = 16, align = (:left, :bottom))
        else
            photon!(ax, (0.35, 0.05), (1.2, 0.28); cycles = 6, amplitude = 0.07)
            Mke.text!(ax, 1.2, 0.42; text = "Same Energy", color = PALETTE.muted, fontsize = 16, align = (:right, :bottom))
        end
        Mke.text!(ax, 0, -0.42; text = "Cd or Te Atom", color = Mke.RGBf(0.4, 0.42, 0.5), fontsize = 15, align = (:center, :top))
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end

    # How far a photon goes, on average, before it interacts: 1/μ, against the 1 mm crystal
    d = detector.thickness_mm
    ax = Mke.Axis(
        fig[1:2, 4];
        title = "How Far a Photon Goes in CdTe",
        xlabel = "Mean Distance Before Interacting (mm)",
        yticks = (1:4, ["30 keV", "60 keV", "100 keV", "140 keV"]),
        yreversed = true,
        limits = ((0, 1.3 * maximum(1 / μ_per_mm(E) for E in (30.0, 60.0, 100.0, 140.0))), (0.4, 4.6)),
        titlesize = 19,
        xlabelsize = 16,
    )
    Mke.vspan!(ax, 0, d; color = (MATERIALS.cdte, 0.8))
    Mke.text!(ax, 0.03, 4.5; text = "The $(d) mm Crystal", align = (:left, :bottom), fontsize = 15, color = PALETTE.slab_edge)
    for (k, E) in enumerate((30.0, 60.0, 100.0, 140.0))
        L = 1 / μ_per_mm(E)
        Mke.barplot!(ax, [k], [L]; direction = :x, color = PALETTE.photon, width = 0.5)
        Mke.text!(ax, L, k; text = " $(round(L; digits = 2)) mm", align = (:left, :center), fontsize = 15)
    end
    Mke.colsize!(fig.layout, 4, Mke.Relative(0.24))
    fig
end

# ╔═╡ 9d277927-0f8a-52c6-82e9-e41826865c35
let
    Mke.set_theme!(THEME)

    E = collect(fine_energies)
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
    fig = Mke.Figure(size = (1200, 820))

    # Top: μ for each process, Geant4 solid, XCOM dashed
    ax = Mke.Axis(
        fig[1, 1];
        yscale = log10,
        ylabel = "Chance of Interacting per mm, μ (1/mm)",
        title = "How Strongly CdTe Stops X Rays",
        xticks = 10:10:150,
        limits = ((10, 150), (8.0e-3, 150)),
    )
    for p in processes
        Mke.lines!(
            ax, E, mu_geant4[p.kind];
            color = p.color,
            label = p.label,
            linewidth = p.kind === :total ? 3.5 : 2.5,
        )
        Mke.lines!(
            ax, E, mu_xcom[p.kind];
            color = (p.color, 0.7),
            linestyle = :dash,
            linewidth = 1.6,
        )
    end
    for (i, e) in enumerate(MC.k_edges(detector.material))
        Mke.vlines!(
            ax,
            e.edge_keV;
            color = (PALETTE.xray, 0.8),
            linestyle = :dot,
            linewidth = 2,
        )
        note!(
            ax, e.edge_keV, i == 1 ? 0.035 : 0.014,
            "$(e.element) K-Edge, $(round(e.edge_keV; digits = 1)) keV";
            offset = (8, 0),
            color = PALETTE.xray,
        )
    end
    Mke.Legend(
        fig[1, 2],
        vcat(
            [Mke.LineElement(; color = p.color, linewidth = 3) for p in processes],
            [
                Mke.LineElement(; color = PALETTE.ink, linewidth = 2.5),
                Mke.LineElement(; color = PALETTE.ink, linestyle = :dash, linewidth = 1.6),
            ],
        ),
        vcat(
            [p.label for p in processes],
            ["Solid: Geant4 (What the Simulation Uses)", "Dashed: NIST XCOM (Comparison)"],
        );
        valign = :top,
    )

    # Bottom: how far apart the two data sets are
    axr = Mke.Axis(
        fig[2, 1];
        xlabel = "Photon Energy (keV)",
        ylabel = "Geant4 ÷ XCOM",
        xticks = 10:10:150,
        limits = ((10, 150), (0.7, 1.15)),
        yticks = 0.7:0.1:1.1,
    )
    Mke.hlines!(axr, 1; color = PALETTE.muted, linewidth = 1.5)
    for p in processes
        Mke.lines!(
            axr, E, mu_geant4[p.kind] ./ mu_xcom[p.kind];
            color = p.color,
            linewidth = p.kind === :total ? 3.5 : 2.5,
        )
    end

    Mke.linkxaxes!(ax, axr)
    Mke.hidexdecorations!(ax; grid = false)
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.28))
    Mke.rowgap!(fig.layout, 14)
    fig
end

# ╔═╡ 26891d21-f1b5-5660-8c3c-b4547fcd0771
md"""
## 3. Following Photons

`record_tracks` fires a few dozen photons and keeps every step of every particle they set
moving. Panel (a) is the side view of figure 1(b): the beam comes in from the left and the grey
band is the crystal. Panel (b) magnifies one absorption about a hundred times.

What to look for: almost every photon is **absorbed** (red dot) in the first few hundred
micrometres, because μ is large. Many absorptions send out a **characteristic x ray** (orange)
in a random direction, which travels up to a millimetre before it is absorbed in turn. The ones
that start near the front face and head backward get out of the crystal, and their energy is
lost to the detector: this is **K-escape**. The **electron** the photon frees (dark red,
panel b) stops within about a micrometre, so its energy stays where it was made.
"""

# ╔═╡ c3606d49-a795-57c0-aab2-898e83adf9ea
tracks = MC.record_tracks(
    detector, 60.0;
    n = 40,
    beam,
    seed = 7,
)

# ╔═╡ 238ced87-5f00-531b-99fe-ac01d8799042
let
    Mke.set_theme!(THEME)

    t = tracks
    d = detector.thickness_mm
    nphotons = length(unique(s.event for s in t.steps))
    fig = Mke.Figure(size = (1200, 1000))

    # (a) Every photon and x ray, side view
    ax = Mke.Axis(
        fig[1, 1:2];
        title = "(a) $(nphotons) Photons of $(round(Int, t.energy_keV)) keV Entering $(d) mm of CdTe (Side View)",
        xlabel = "Depth Along the Beam, z (mm)",
        ylabel = "x (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.0, d + 1.0), (-0.6, 0.6)),
    )
    Mke.poly!(
        ax, Mke.Rect2f(0, -0.6, d, 1.2);
        color = PALETTE.slab,
        strokecolor = PALETTE.slab_edge,
        strokewidth = 2,
    )
    for s in t.steps
        s.particle in (:primary, :fluorescence) || continue
        color = s.particle === :primary ? PALETTE.photon : PALETTE.xray
        Mke.lines!(
            ax, [s.start[3], s.stop[3]], [s.start[1], s.stop[1]];
            color = (color, 0.85),
            linewidth = s.particle === :primary ? 1.8 : 1.6,
        )
    end
    function step_ends(process)
        ends = filter(
            s -> s.particle in (:primary, :fluorescence) && s.process == process,
            t.steps,
        )
        return [Mke.Point2f(s.stop[3], s.stop[1]) for s in ends]
    end
    Mke.scatter!(
        ax,
        step_ends("phot");
        marker = :circle,
        markersize = 9,
        color = PALETTE.electron,
    )
    Mke.scatter!(
        ax, step_ends("compt");
        marker = :diamond,
        markersize = 13,
        color = PALETTE.photon,
        strokecolor = :white,
        strokewidth = 1.5,
    )
    Mke.scatter!(
        ax,
        step_ends("Rayl");
        marker = :xcross,
        markersize = 13,
        color = PALETTE.photon,
    )
    note!(
        ax,
        -0.97,
        0.56,
        "← Escapes Out the Front";
        align = (:left, :top),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        ax,
        d + 0.97,
        0.56,
        "Escapes Out the Back →";
        align = (:right, :top),
        color = PALETTE.muted,
        fontsize = 15,
    )
    Mke.Legend(
        fig[1, 3],
        [
            Mke.LineElement(; color = PALETTE.photon, linewidth = 3),
            Mke.LineElement(; color = PALETTE.xray, linewidth = 3),
            Mke.LineElement(; color = PALETTE.electron, linewidth = 3),
            Mke.MarkerElement(;
                marker = :circle,
                color = PALETTE.electron,
                markersize = 12,
            ),
            Mke.MarkerElement(; marker = :diamond, color = PALETTE.photon, markersize = 14),
            Mke.MarkerElement(; marker = :xcross, color = PALETTE.photon, markersize = 14),
        ],
        [
            "Incoming Photon",
            "Characteristic X Ray",
            "Electron (Panel b)",
            "Photon Absorbed",
            "Bounced, Lost Energy\n(Compton)",
            "Bounced, Kept Energy\n(Rayleigh)",
        ];
        valign = :center,
    )

    # (b) Zoom onto the first photon absorbed by the photoelectric effect
    absorbed = first(s for s in t.steps if s.particle === :primary && s.process == "phot")
    c = absorbed.stop
    steps = [s for s in t.steps if s.event == absorbed.event]
    electrons = [s for s in steps if s.particle === :electron]
    half = 1000 * 1.4 * maximum(
        max(abs(s.stop[3] - c[3]), abs(s.stop[1] - c[1])) for s in electrons;
        init = 1.0e-3,
    )
    µm(v, i) = 1000 * (v[i] - c[i])
    axz = Mke.Axis(
        fig[2, 1];
        title = "(b) One Absorption, Magnified",
        xlabel = "z − z₀ (µm)",
        ylabel = "x − x₀ (µm)",
        aspect = Mke.DataAspect(),
        limits = ((-half, half), (-half, half)),
    )
    for s in steps
        color = s.particle === :electron ? PALETTE.electron :
            s.particle === :primary ? PALETTE.photon : PALETTE.xray
        Mke.lines!(
            axz, [µm(s.start, 3), µm(s.stop, 3)], [µm(s.start, 1), µm(s.stop, 1)];
            color,
            linewidth = s.particle === :electron ? 1.6 : 2.5,
        )
    end
    Mke.scatter!(axz, [Mke.Point2f(0, 0)]; color = PALETTE.electron, markersize = 14)
    ekin = maximum(s.energy_keV for s in electrons)
    Mke.Label(
        fig[2, 2:3],
        """
        The Photon Is Absorbed at the Dot,
        $(round(Int, 1000 * c[3])) µm Inside the Crystal.

        The Electron It Frees ($(round(ekin; digits = 1)) keV) Stops
        Within About a Micrometre: Its Energy Is
        Deposited on the Spot.

        When the Atom Emits a Characteristic X Ray
        Instead (Orange in Panel a), That X Ray Travels
        Hundreds of µm Before It Is Absorbed, or It
        Escapes the Crystal.
        """;
        justification = :left,
        halign = :left,
        valign = :center,
        fontsize = 17,
        tellheight = false,
        tellwidth = false,
        color = PALETTE.ink,
    )

    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.48))
    Mke.rowgap!(fig.layout, 24)
    fig
end

# ╔═╡ abb7bdf7-7414-5dd7-963e-365da87b7db1
let
    t = tracks
    n = length(unique(s.event for s in t.steps))
    absorbed = count(s -> s.particle === :primary && s.process == "phot", t.steps)

    # The first step of every characteristic x ray, to read its energy
    first_step(event, track) = first(s for s in t.steps if (s.event, s.track) == (event, track))
    xray_tracks = unique((s.event, s.track) for s in t.steps if s.particle === :fluorescence)
    xrays = [first_step(event, track) for (event, track) in xray_tracks]
    k_xrays = count(s -> s.energy_keV > 15, xrays)

    depths = sort([s.stop[3] for s in t.steps if s.particle === :primary && s.process == "phot"])
    median_depth_um = round(Int, 1000 * depths[cld(length(depths), 2)])

    Markdown.parse(
        """
        Of these $(n) photons, $(absorbed) were absorbed, half of them within the first
        $(median_depth_um) µm. The atoms emitted $(k_xrays) K x rays (above 15 keV, able to
        travel) and $(length(xrays) - k_xrays) L x rays (a few keV, absorbed within micrometres).
        """
    )
end

# ╔═╡ ea3431db-2042-5556-9c93-4a01ffc37172
md"""
## 4. Check 1: the Monte Carlo Reproduces Beer-Lambert

Before any detector physics is trusted, the simplest number must come out right. The fraction of
photons that interact at all (absorbed or scattered) is exactly `1 − exp(−μd)`. The Monte Carlo
gets it by firing photons and counting; Beer-Lambert gets it from a formula. With Geant4's own μ
in the formula, the two must agree to within counting noise at every energy. If the geometry,
the density, the composition or the beam were wrong, they would not.

The sweep fires 100 000 photons at every whole keV from 10 to 140 keV, BasisSimulator's energy
grid, spread over the eight worker processes started in the setup. Every photon's record is kept, so sections 5 and 6 reuse these runs.
"""

# ╔═╡ ced08648-4d8b-5071-9cb1-43add187e918
energies = 10.0:1.0:140.0

# ╔═╡ 3baf0b3a-d671-5561-83ae-9691038f4481
sweep = MC.simulate_monoenergetic(
    detector, energies;
    n = 100_000,
    beam,
    seed = 1,
)

# ╔═╡ c7c8e6e4-5dbc-5325-bbe6-b215ed7c8c3f
beer_lambert_geant4 = [
    MC.beer_lambert_efficiency(
            detector.material,
            detector.thickness_mm,
            E;
            cross_sections = :geant4,
        )
        for E in energies
]

# ╔═╡ c13e2c0a-19ec-53e0-ba89-094251e99e2c
beer_lambert_xcom = [
    MC.beer_lambert_efficiency(
            detector.material,
            detector.thickness_mm,
            E;
            cross_sections = :xcom,
        )
        for E in energies
]

# ╔═╡ 99963866-c3af-4456-b196-7bf6bfa8a488
let
    Mke.set_theme!(THEME)
    μ_per_mm(E) = MC.interpolate_linear(collect(fine_energies), mu_geant4[:total], E)
    d = detector.thickness_mm
    fig = Mke.Figure(size = (1500, 500))
    N = 24
    for (k, E) in enumerate((25.0, 60.0, 140.0))
        μ = μ_per_mm(E)
        interacts = 1 - exp(-μ * d)
        ax = canvas(fig[1, k]; title = "$(Int(E)) keV: $(round(Int, 100interacts)) % Interact", titlesize = 21)
        rect!(ax, 0, 0, d, 1.6, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        # photon i interacts at the depth the i-th of N spread-out quantiles of exp(−μz) gives
        for i in 1:N
            y = 1.6 * (i - 0.5) / N
            z = -log(1 - mod(i * 0.618034, 1.0)) / μ
            if z < d
                Mke.lines!(ax, [-0.55, z], [y, y]; color = PALETTE.photon, linewidth = 2)
                Mke.scatter!(ax, [z], [y]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (-0.55, y), (d + 0.5, y); color = (PALETTE.photon, 0.55), linewidth = 2, head = 10)
            end
        end
        Mke.text!(ax, d / 2, -0.05; text = "$(d) mm of CdTe", align = (:center, :top), fontsize = 15, color = PALETTE.slab_edge)
        Mke.limits!(ax, -0.65, d + 0.6, -0.35, 1.7)
    end
    Mke.Label(
        fig[2, 1:3],
        "Each Line Is a Photon from the Left: It Interacts at the Red Dot, or Passes Straight Through.\nThe Fraction That Interacts Is 1 − exp(−μ d): the Black Line Below, Which the Blue Dots (Counted) Must Follow.";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 7bfbd9e9-b203-5781-a28b-d3dcf06d765d
let
    Mke.set_theme!(THEME)

    E = collect(energies)
    p = MC.interaction_probability.(sweep)

    # Counting uncertainty of the expected fraction; at least one photon's worth where it is ≈ 1
    σ = [sqrt(max(q * (1 - q), 1 / r.n) / r.n) for (q, r) in zip(beer_lambert_geant4, sweep)]
    fig = Mke.Figure(size = (1200, 900))

    # Top: the fractions themselves
    ax = Mke.Axis(
        fig[1, 1];
        title = "Monte Carlo Against Beer–Lambert, $(detector.material.name) $(detector.thickness_mm) mm",
        ylabel = "Fraction of Photons\nThat Interact",
        xticks = 10:10:140,
        limits = ((8, 142), nothing),
    )
    Mke.lines!(
        ax, E, beer_lambert_geant4;
        color = PALETTE.ink,
        linewidth = 3,
        label = "Beer–Lambert, Geant4's μ",
    )
    Mke.lines!(
        ax, E, beer_lambert_xcom;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Beer–Lambert, XCOM's μ (Comparison)",
    )
    Mke.scatter!(
        ax, E, p;
        color = PALETTE.photon,
        markersize = 7,
        label = "Monte Carlo: Photons That Interacted",
    )
    Mke.axislegend(ax; position = :lb)

    # Middle: the difference in units of counting noise
    axr = Mke.Axis(
        fig[2, 1];
        ylabel = "Difference ÷\nCounting Noise",
        xticks = 10:10:140,
        limits = ((8, 142), (-4.5, 4.5)),
        yticks = -4:2:4,
    )
    Mke.hspan!(axr, -2, 2; color = (PALETTE.kept, 0.15))
    Mke.hlines!(axr, 0; color = PALETTE.ink, linewidth = 1.5)
    Mke.scatter!(
        axr,
        E,
        (p .- beer_lambert_geant4) ./ σ;
        color = PALETTE.photon,
        markersize = 7,
    )
    note!(
        axr, 140, 2.0, "Green Band: Within ±2 × the Counting Noise";
        align = (:right, :bottom),
        color = PALETTE.kept,
        fontsize = 15,
    )

    # Bottom: how far BasisSimulator's XCOM Beer-Lambert sits from Geant4's
    xcom_minus_geant4 = 100 .* (beer_lambert_xcom .- beer_lambert_geant4)
    axx = Mke.Axis(
        fig[3, 1];
        xlabel = "Photon Energy (keV)",
        ylabel = "XCOM − Geant4\n(Percentage Points)",
        xticks = 10:10:140,
        limits = ((8, 142), nothing),
    )
    Mke.hlines!(axx, 0; color = PALETTE.ink, linewidth = 1.5)
    Mke.lines!(axx, E, xcom_minus_geant4; color = PALETTE.muted, linewidth = 2.5)
    note!(
        axx, 10, maximum(xcom_minus_geant4),
        "How Much Beer–Lambert with XCOM (BasisSimulator's Data) Differs";
        align = (:left, :top),
        color = PALETTE.muted,
        fontsize = 15,
    )

    for a in (ax, axr)
        Mke.hidexdecorations!(a; grid = false)
    end
    Mke.linkxaxes!(ax, axr, axx)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.5))
    Mke.rowgap!(fig.layout, 12)
    fig
end

# ╔═╡ 2dcf44e4-8d50-50e0-8e0e-3fefca54e825
let
    p = MC.interaction_probability.(sweep)
    σ = [sqrt(max(q * (1 - q), 1 / r.n) / r.n) for (q, r) in zip(beer_lambert_geant4, sweep)]
    z = (p .- beer_lambert_geant4) ./ σ
    inside = count(x -> abs(x) <= 2, z)
    percent_inside = round(Int, 100 * inside / length(z))
    xcom_gap = 100 * maximum(abs, beer_lambert_xcom .- beer_lambert_geant4)

    Markdown.parse(
        """
        $(inside) of the $(length(z)) energies, $(percent_inside) %, agree with
        Geant4's Beer-Lambert to within twice the counting noise (about 95 % would by pure chance),
        and the largest disagreement is $(round(maximum(abs, z); digits = 1)) × the noise. Against
        XCOM's Beer-Lambert the largest difference is $(round(xcom_gap; digits = 2)) percentage
        points: the size of the disagreement between the two standard cross-section data sets.
        """
    )
end

# ╔═╡ 0d9b0e7c-7ff4-5632-a7ed-9231f0b0f09a
md"""
## 5. Where the Energy Goes

Energy is conserved photon by photon: everything the incoming photon brought is either left in
the crystal or carried out by something that escaped. The budget below adds those up over all
photons at each energy. The top of the stack must be exactly 1.
"""

# ╔═╡ 2cdb7c64-a9f9-43e6-a637-512f70bb88b0
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 520))
    ax = canvas(fig[1, 1])
    rect!(ax, 0, -3.0, 2.0, 6.0, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 1.0, -3.15; text = "The CdTe Slab (Depth Exaggerated)", align = (:center, :top), fontsize = 15, color = PALETTE.slab_edge)
    star!(p) = Mke.scatter!(ax, [Mke.Point2f(p)]; marker = :star5, markersize = 24, color = MATERIALS.star, strokecolor = PALETTE.photon, strokewidth = 1.2)
    ys = (2.2, 1.0, -0.2, -1.4, -2.6)
    for (k, y) in enumerate(ys)
        photon!(ax, (-3.0, y), k == 5 ? (5.6, y) : (k == 4 ? 1.7 : 0.6, y); cycles = k == 5 ? 13 : 6, amplitude = 0.015)
    end
    star!((0.6, 2.2))
    star!((0.6, 1.0))
    photon!(ax, (0.6, 1.0), (-1.9, 1.75); cycles = 3, amplitude = 0.05, color = PALETTE.xray)
    star!((0.6, -0.2))
    photon!(ax, (0.6, -0.2), (5.3, -0.75); cycles = 4, amplitude = 0.04)
    Mke.scatter!(ax, [Mke.Point2f(1.7, -1.4)]; marker = :star5, markersize = 24, color = MATERIALS.star, strokecolor = PALETTE.photon, strokewidth = 1.2)
    arrow!(ax, (1.7, -1.4), (2.45, -1.3); color = PALETTE.electron, linewidth = 3, head = 12)
    labels = (
        (2.2, "Stays in the Crystal (the Signal)", PALETTE.kept),
        (1.0, "Carried Out by a Characteristic X Ray", PALETTE.xray),
        (-0.2, "Carried Out by a Scattered Photon", PALETTE.photon),
        (-1.4, "Carried Out by an Electron (Freed Right at the Back Face)", PALETTE.electron),
        (-2.6, "Passes Straight Through", Mke.RGBf(0.55, 0.55, 0.6)),
    )
    for (y, text, color) in labels
        Mke.text!(ax, -3.1, y; text, color, fontsize = 17, font = :bold, align = (:right, :center))
    end
    Mke.limits!(ax, -12.5, 6.0, -3.6, 2.7)
    fig
end

# ╔═╡ 21575ae6-eade-5f3f-af5e-d8b536251282
let
    Mke.set_theme!(THEME)

    E = collect(energies)
    budgets = MC.energy_budget.(sweep)
    layers = (
        (
            part = :absorbed,
            label = "Stays in the Crystal (the Signal)",
            color = PALETTE.kept,
        ),
        (
            part = :fluorescence,
            label = "Carried Out by a Characteristic X Ray",
            color = PALETTE.xray,
        ),
        (
            part = :scattered,
            label = "Carried Out by a Scattered Photon",
            color = PALETTE.photon,
        ),
        (part = :electron, label = "Carried Out by an Electron", color = PALETTE.electron),
        (
            part = :transmitted,
            label = "Passes Straight Through",
            color = Mke.RGBf(0.85, 0.85, 0.87),
        ),
    )
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Where Each Photon's Energy Goes, $(detector.material.name) $(detector.thickness_mm) mm",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of the Photon's Energy",
        xticks = 10:10:140,
        limits = ((10, 140), (0, 1.0)),
    )

    # Stack the layers from the bottom up
    lower = zeros(length(E))
    for layer in layers
        v = [getproperty(b, layer.part) for b in budgets]
        Mke.band!(ax, E, lower, lower .+ v; color = layer.color, label = layer.label)
        lower .+= v
    end
    for e in MC.k_edges(detector.material)
        Mke.vlines!(ax, e.edge_keV; color = (:white, 0.9), linestyle = :dash, linewidth = 2)
    end
    escape_peak = round(Int, 100 * maximum(b.fluorescence for b in budgets))
    note!(
        ax, 33, 0.7,
        "Just Above the K-Edges,\nUp to $(escape_peak) % of the Energy\nLeaves as X Rays";
        align = (:left, :center),
        fontsize = 15,
    )

    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 265a57e2-edba-5765-b933-976cc027eb82
let
    worst = maximum(abs(b.total - 1) for b in MC.energy_budget.(sweep))

    Markdown.parse(
        """
        Largest departure of the total from 1 over all energies: **$(round(worst; sigdigits = 2))**.
        No energy is created or lost by the simulation's bookkeeping.
        """
    )
end

# ╔═╡ beafa265-f6ce-55b2-abfd-b194220ceed5
md"""
### The spectrum of energy left in the crystal

At one photon energy, the histogram of the energy each photon left behind shows every way of
losing energy as its own feature. The top panel is what escaped as characteristic x rays: sharp
lines, the fingerprints of Cd and Te. The bottom panel is what stayed: a **full-energy peak**
at the photon energy, and for every line above an **escape peak** at the photon energy minus
that line, made by the photons whose x ray got out. The numbers match each escaped line to its
escape peak. (Kα and Kβ are the two strongest characteristic lines of each element; Kβ is the
rarer, slightly more energetic one.)
"""

# ╔═╡ 7b6c3fe9-caa4-5d4b-a1b7-6038524e2b43
spectrum_energy = 60.0

# ╔═╡ 184a9cef-ecb7-5563-bcc9-dbc8af750c2d
spectrum_run = sweep[findfirst(r -> r.energy_keV == spectrum_energy, sweep)]

# ╔═╡ 8fce8f9a-8b98-5a61-9459-6ab5c3b9fb74
escaped_lines = MC.fluorescence_lines(
    spectrum_run;
    min_fraction = 0.05,
)

# ╔═╡ 718fd5fc-af35-4451-b25e-75571d21f3ec
let
    Mke.set_theme!(THEME)
    E0 = round(Int, spectrum_run.energy_keV)
    fig = Mke.Figure(size = (1500, 470))
    cases = (
        ("All of It Stays", "The Photon and Its X Ray Are Both\nAbsorbed: the Full-Energy Peak at $(E0) keV", PALETTE.kept),
        ("Its X Ray Escapes", "A Cd or Te X Ray Leaves Through the Front:\nan Escape Peak at $(E0) keV Minus the X Ray", PALETTE.xray),
        ("A Scattered Photon Leaves", "Only the Electron's Energy Stays:\nthe Low Tail", PALETTE.photon),
    )
    star!(ax, p) = Mke.scatter!(ax, [Mke.Point2f(p)]; marker = :star5, markersize = 24, color = MATERIALS.star, strokecolor = PALETTE.photon, strokewidth = 1.2)
    for (k, (title, detail, color)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 21)
        rect!(ax, 0, -1.2, 1.6, 2.4, MATERIALS.cdte; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        hit = Mke.Point2f(0.45, 0.0)
        photon!(ax, (-1.6, 0.0), hit; cycles = 6, amplitude = 0.03)
        if k == 1
            photon!(ax, hit, (1.2, -0.7); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 10)
            Mke.scatter!(ax, [Mke.Point2f(1.23, -0.72)]; color = PALETTE.electron, markersize = 10)
        elseif k == 2
            photon!(ax, hit, (-1.3, 0.95); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 12)
        else
            photon!(ax, hit, (2.9, -0.9); cycles = 3, amplitude = 0.05, head = 12)
        end
        star!(ax, hit)
        Mke.limits!(ax, -1.7, 3.0, -1.3, 1.2)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    fig
end

# ╔═╡ 2cd36cc8-7384-5e22-8d2d-b4acf3adbb0d
let
    Mke.set_theme!(THEME)

    r = spectrum_run
    E0 = r.energy_keV
    fig = Mke.Figure(size = (1200, 880))

    # Top: the characteristic x rays that escaped
    axe = Mke.Axis(
        fig[1, 1];
        title = "What Escaped: Characteristic X Rays Leaving the Crystal ($(round(Int, E0)) keV Photons)",
        yscale = log10,
        ylabel = "Per keV, per\nIncoming Photon",
        limits = ((0, E0 + 3), (1.0e-4, 3)),
    )
    escaped = MC.escape_spectrum(r; kind = :fluorescence, bin_keV = 0.25)
    Mke.stairs!(
        axe, escaped.energy_keV, max.(escaped.density, 1.0e-9);
        step = :center,
        color = PALETTE.xray,
        linewidth = 2,
    )
    for (i, line) in enumerate(escaped_lines)
        peak = maximum(escaped.density[abs.(escaped.energy_keV .- line.energy_keV) .< 0.5])
        note!(
            axe,
            line.energy_keV,
            1.8 * peak,
            string(i);
            align = (:center, :bottom),
            color = PALETTE.xray,
            fontsize = 17,
        )
    end

    # Bottom: the energy each photon left in the crystal
    ax = Mke.Axis(
        fig[2, 1];
        title = "What Stayed: Energy Left in the Crystal by Each Photon",
        yscale = log10,
        xlabel = "Energy (keV)",
        ylabel = "Per keV, per\nIncoming Photon",
        limits = ((0, E0 + 3), (1.0e-4, 30)),
    )
    kept = MC.absorbed_energy_spectrum(r; bin_keV = 0.25)
    Mke.stairs!(
        ax, kept.energy_keV, max.(kept.density, 1.0e-9);
        step = :center,
        color = PALETTE.kept,
        linewidth = 2,
    )
    note!(
        ax, E0, 8, "Full-Energy Peak:\nAll $(round(Int, E0)) keV Stayed";
        align = (:right, :center),
        offset = (-14, 0),
        fontsize = 15,
    )
    for (i, line) in enumerate(escaped_lines)
        x = E0 - line.energy_keV
        peak = maximum(kept.density[abs.(kept.energy_keV .- x) .< 0.5])
        note!(
            ax,
            x,
            1.8 * peak,
            string(i);
            align = (:center, :bottom),
            color = PALETTE.xray,
            fontsize = 17,
        )
    end
    note!(
        ax,
        1,
        3.0e-4,
        "Scattered Photons That Escaped";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )

    # The key that matches each escaped line to its escape peak
    key_row(i, line) = string(
        "$(i)   $(line.element) $(line.line) X Ray, $(round(line.energy_keV; digits = 1)) keV\n",
        "     Escape Peak at $(round(E0 - line.energy_keV; digits = 1)) keV",
    )
    key = join([key_row(i, line) for (i, line) in enumerate(escaped_lines)], "\n")
    Mke.Label(
        fig[1:2, 2], key;
        justification = :left,
        halign = :left,
        valign = :center,
        fontsize = 16,
        tellheight = false,
        color = PALETTE.ink,
    )

    Mke.colsize!(fig.layout, 2, Mke.Fixed(300))
    Mke.linkxaxes!(axe, ax)
    Mke.hidexdecorations!(axe; grid = false)
    Mke.rowgap!(fig.layout, 16)
    fig
end

# ╔═╡ a2f74850-388c-5e6c-a526-757f0dd79798
md"""
## 6. The Efficiency Curves

Three numbers summarise a detector at each energy:

- **Quantum efficiency:** the fraction of photons that leave *any* energy in the crystal. A
  photon-counting detector with a low threshold counts these.
- **Energy-absorption efficiency:** the average energy a photon leaves, as a fraction of its
  energy. An energy-integrating (scintillator) detector's signal is proportional to it; it is
  the quantity in the Monte Carlo tables BasisSimulator v0.15.1 used, and in the Geant4 tables
  of notebooks 04–08 that replace them.
- **Swank factor:** how alike the energies left by the detected photons are. It is 1 when every
  detected photon leaves the same energy, and it falls as they spread out (escape peaks,
  scatter). The zero-frequency DQE of an energy-integrating detector is the quantum efficiency
  times the Swank factor.

Beer-Lambert counts every interaction, so it sits slightly above the quantum efficiency: a
photon that only bounces out of the crystal (Rayleigh) interacted but left nothing. It cannot
predict the energy-absorption efficiency at all: just above each K-edge absorption gets *more*
likely, yet the energy kept *drops*, because every characteristic x ray that escapes takes its
energy with it.
"""

# ╔═╡ 717fa436-6ead-47d3-85aa-b3b64c2c0bbd
let
    Mke.set_theme!(THEME)
    # Ten photons of 60 keV, and the energy each leaves in the crystal (an illustration, not data):
    # most leave all of it, two lose a Te or Cd x ray, one only bounces off (Rayleigh), one Compton
    E0 = 60.0
    kept = [60.0, 60.0, 32.5, 60.0, 36.8, 60.0, 0.0, 60.0, 60.0, 4.0]
    colors = [v == E0 ? PALETTE.kept : v > 20 ? PALETTE.xray : v > 0 ? PALETTE.photon : Mke.RGBf(0.8, 0.8, 0.83) for v in kept]
    detected = count(>(0), kept)
    M0, M1, M2 = detected / 10, sum(kept) / 10, sum(abs2, kept) / 10
    fig = Mke.Figure(size = (1500, 480))
    panels = (
        ("Quantum Efficiency", "Did It Leave Any Energy?\n$(detected) of 10 Did: $(detected / 10)", PALETTE.photon),
        ("Energy-Absorption Efficiency", "What Share of the Energy Stayed?\n$(round(Int, sum(kept))) of $(round(Int, 10 * E0)) keV: $(round(sum(kept) / (10 * E0); digits = 2))", PALETTE.xray),
        ("Swank Factor", "How Alike Are the Amounts?\nI = M₁² / (M₀ M₂) = $(round(M1^2 / (M0 * M2); digits = 2))", PALETTE.electron),
    )
    for (k, (title, detail, color)) in enumerate(panels)
        ax = Mke.Axis(
            fig[1, k];
            title,
            titlecolor = color,
            xlabel = "Ten $(round(Int, E0)) keV Photons",
            ylabel = k == 1 ? "Energy Left (keV)" : "",
            xticks = 1:10,
            limits = ((0.3, 10.7), (0, 68)),
        )
        Mke.barplot!(ax, 1:10, kept; color = colors)
        k == 2 && Mke.hlines!(ax, sum(kept) / 10; color = PALETTE.xray, linewidth = 2.5, linestyle = :dash)
        k == 2 && note!(ax, 10.6, sum(kept) / 10, "Average Kept"; align = (:right, :bottom), color = PALETTE.xray, fontsize = 14, offset = (0, 4))
        k == 1 && note!(ax, 7, 3, "Left Nothing"; align = (:center, :bottom), color = PALETTE.muted, fontsize = 14)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    end
    fig
end

# ╔═╡ 6c94d133-46f8-5406-bc79-1cc8106a1c4b
let
    Mke.set_theme!(THEME)

    E = collect(energies)
    edges = MC.k_edges(detector.material)
    swank = MC.swank_factor.(sweep)
    fig = Mke.Figure(size = (1200, 860))

    # Top: Beer-Lambert against the two Monte Carlo efficiencies
    ax = Mke.Axis(
        fig[1, 1];
        title = "What Beer–Lambert Predicts and What the Crystal Keeps, $(detector.material.name) $(detector.thickness_mm) mm",
        ylabel = "Efficiency",
        xticks = 10:10:140,
        limits = ((8, 142), (0.25, 1.04)),
    )
    Mke.lines!(
        ax, E, beer_lambert_geant4;
        color = PALETTE.ink,
        linewidth = 3,
        label = "Beer–Lambert: Photons That Interact",
    )
    Mke.lines!(
        ax, E, MC.quantum_efficiency.(sweep);
        color = PALETTE.photon,
        linewidth = 3,
        label = "Quantum Efficiency: Photons That Leave Any Energy",
    )
    Mke.lines!(
        ax, E, MC.energy_absorption_efficiency.(sweep);
        color = PALETTE.xray,
        linewidth = 3,
        label = "Energy-Absorption Efficiency: Share of Energy Kept",
    )
    Mke.axislegend(ax; position = :lb)

    # Bottom: the Swank factor
    axs = Mke.Axis(
        fig[2, 1];
        xlabel = "Photon Energy (keV)",
        ylabel = "Swank Factor",
        xticks = 10:10:140,
        limits = ((8, 142), nothing),
    )
    Mke.lines!(axs, E, swank; color = PALETTE.electron, linewidth = 3)

    for a in (ax, axs), e in edges
        Mke.vlines!(
            a,
            e.edge_keV;
            color = (PALETTE.muted, 0.6),
            linestyle = :dash,
            linewidth = 1.5,
        )
    end
    note!(
        ax,
        edges[end].edge_keV,
        0.7,
        "K-Edges of Cd and Te";
        offset = (10, 0),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        axs, 141, minimum(swank), "1 = Every Detected Photon Leaves the Same Energy";
        align = (:right, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )

    Mke.linkxaxes!(ax, axs)
    Mke.hidexdecorations!(ax; grid = false)
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.3))
    Mke.rowgap!(fig.layout, 14)
    fig
end

# ╔═╡ 68872016-40db-5305-ad2a-8ca120226c43
md"""
## Physics Settings

Every run in this notebook uses the settings below, fixed for the Julia session (Geant4 cannot
change its physics once it has started).
"""

# ╔═╡ b27b88bd-16e8-5270-acc0-16c74c129393
MC.PhysicsSettings()

# ╔═╡ e16fc829-1d5c-545a-ac23-c1a36c562230
md"""
## Next

This notebook establishes that the machinery is right on the simplest detector: the Monte
Carlo reproduces Beer-Lambert with Geant4's own cross sections, conserves energy photon by
photon, and shows the K-escape that Beer-Lambert cannot.

- **Notebook 02** compares it with published Monte Carlo work: the Swank factors of four
  detector materials from Hajdok, Battista & Cunningham (Med Phys 2008, EGSnrc).
- **Notebook 03** builds the Siemens NAEOTOM Alpha's CdTe detector dexel by dexel and compares
  its response with the one BasisSimulator v0.15.1 used.
"""

# ╔═╡ Cell order:
# ╟─8296a665-0099-56bd-85fd-89eba76fb807
# ╟─217c5b00-469a-5a92-bd6e-da4508b34921
# ╟─dfc666d5-5ff2-502e-aeb0-49439685ac86
# ╟─41e976eb-d5a6-5afd-9c35-4670601ce1ec
# ╠═09d7b527-61f3-5c01-a16a-d4178d4d985a
# ╠═db1d8de8-d6dc-5361-bce7-7948e59da8bc
# ╠═3895df79-1afe-5296-b04c-a0995fb14057
# ╠═de195e86-864c-51a4-9372-162e52f23278
# ╠═f0468c43-be4c-5dc1-816e-66eb7e24e1b0
# ╠═61babc92-4726-5e99-aef4-dde0339aa83b
# ╠═85de25c3-beb0-590e-b45f-e3e76305c57b
# ╠═74dc1d9e-d900-5f7c-8825-1d332f6a89b3
# ╠═3b6e9621-cbce-56e9-949f-dd3562147bff
# ╠═5254847b-5c45-5e0c-ab5f-2b6872c13c47
# ╟─3af2ff68-c9d8-4cf6-bfc5-bf391296da68
# ╠═822095ae-24c5-40b5-a2f6-e25dfb90a015
# ╠═e9cc638b-caf5-48a4-bd2c-b0fe0e1e2195
# ╠═87e524c6-a58c-4c90-918d-1e929e0e5f6a
# ╠═09afc8c3-67f0-4506-8242-a569104d1515
# ╠═443b26b9-5365-4eb1-b73d-4228fd3ceb37
# ╠═49342efd-5bbd-41e6-b871-e992c910404a
# ╠═cfba4ed7-6c5e-429d-8361-aa2acb343b1d
# ╠═bdb6e526-af68-4675-b232-2d2d75de03c8
# ╠═3bcdc764-6d39-4d03-9095-029419302112
# ╠═baaf379b-4d53-49a5-87d5-28b7cb166e85
# ╠═50a0f3b5-d39f-4e9c-8161-01961a98fee3
# ╠═7c861316-ce0b-4f55-93a1-e41b8980cd71
# ╠═3333fc70-8b77-4c2b-b8e8-ee6b8baff691
# ╠═cdbf1522-f34f-4563-80b8-ce2cae20c67f
# ╠═c7b30aa6-8a09-4b4d-ac6b-1ea4bce5a6f1
# ╠═cd987c45-a0c4-47d9-ae2b-d263299c8a93
# ╟─9930db7b-c02f-5cce-8043-57033d7852f6
# ╠═1778e101-f9ac-579b-9782-dc12f803a8d9
# ╠═9fb6be41-9274-55b0-a8dc-193e709361fc
# ╟─fea6f5fe-ec6d-514d-973f-1c6f6cbd2393
# ╟─e5aaa948-30a7-5bd6-82e0-f1ad5b3a275a
# ╟─bcf70aa4-58b9-522c-880f-e1888896c935
# ╠═4807e458-f107-5647-bc1a-1bf891280843
# ╠═befb271a-8286-565b-8512-c80c1aef4eed
# ╠═cd2de766-2eac-5fc7-9b9a-8764e0e80e07
# ╟─96b43d9c-786f-4379-a7fc-18bcca4238ea
# ╟─9d277927-0f8a-52c6-82e9-e41826865c35
# ╟─26891d21-f1b5-5660-8c3c-b4547fcd0771
# ╠═c3606d49-a795-57c0-aab2-898e83adf9ea
# ╟─238ced87-5f00-531b-99fe-ac01d8799042
# ╟─abb7bdf7-7414-5dd7-963e-365da87b7db1
# ╟─ea3431db-2042-5556-9c93-4a01ffc37172
# ╠═ced08648-4d8b-5071-9cb1-43add187e918
# ╠═3baf0b3a-d671-5561-83ae-9691038f4481
# ╠═c7c8e6e4-5dbc-5325-bbe6-b215ed7c8c3f
# ╠═c13e2c0a-19ec-53e0-ba89-094251e99e2c
# ╟─99963866-c3af-4456-b196-7bf6bfa8a488
# ╟─7bfbd9e9-b203-5781-a28b-d3dcf06d765d
# ╟─2dcf44e4-8d50-50e0-8e0e-3fefca54e825
# ╟─0d9b0e7c-7ff4-5632-a7ed-9231f0b0f09a
# ╟─2cdb7c64-a9f9-43e6-a637-512f70bb88b0
# ╟─21575ae6-eade-5f3f-af5e-d8b536251282
# ╟─265a57e2-edba-5765-b933-976cc027eb82
# ╟─beafa265-f6ce-55b2-abfd-b194220ceed5
# ╠═7b6c3fe9-caa4-5d4b-a1b7-6038524e2b43
# ╠═184a9cef-ecb7-5563-bcc9-dbc8af750c2d
# ╠═8fce8f9a-8b98-5a61-9459-6ab5c3b9fb74
# ╟─718fd5fc-af35-4451-b25e-75571d21f3ec
# ╟─2cd36cc8-7384-5e22-8d2d-b4acf3adbb0d
# ╟─a2f74850-388c-5e6c-a526-757f0dd79798
# ╟─717fa436-6ead-47d3-85aa-b3b64c2c0bbd
# ╟─6c94d133-46f8-5406-bc79-1cc8106a1c4b
# ╟─68872016-40db-5305-ad2a-8ca120226c43
# ╠═b27b88bd-16e8-5270-acc0-16c74c129393
# ╟─e16fc829-1d5c-545a-ac23-c1a36c562230
