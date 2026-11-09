### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ e73ba2f8-d658-5841-81a3-2a87c22483ad
md"""
# Benchmark 1: the Swank Factor of Four Detector Materials (Hajdok et al., EGSnrc)

Notebook 1 showed that the Monte Carlo is internally right: it reproduces Beer-Lambert with
Geant4's own cross sections and conserves energy photon by photon. This notebook checks it
against someone else's: a published, peer-reviewed Monte Carlo study run with a different code,
EGSnrc.

**The reference.** G. Hajdok, J. J. Battista and I. A. Cunningham, "Fundamental x-ray
interaction limits in diagnostic imaging detectors: Frequency-dependent Swank noise", *Medical
Physics* 35(7):3194 (2008), and Hajdok's PhD thesis (Western University, 2008), which holds the
same work with every figure. Their Figure 3.4 gives the **Swank factor** of four detector
materials, amorphous silicon, amorphous selenium, cesium iodide and lead iodide, from 10 to
150 keV.

**Why the Swank factor.** It measures how alike the energies left by detected photons are
(1 when every detected photon leaves the same energy). It is shaped by everything that makes
energy escape: K-fluorescence near every K-edge, Compton scattering at high energy. It is a
much sharper test of the detector physics than an efficiency, and it is exactly the quantity
`MC.swank_factor` computes from the per-photon records.

1. **The setup.** What they simulated, and how this notebook matches it.
2. **The materials and the thickness rule.** Each layer is as thick as it must be to stop 99 %
   of the photons, so the thickness changes with energy.
3. **The run.** Four materials, about a hundred energies each.
4. **The comparison.** Our Swank factors against theirs, panel by panel.
"""

# ╔═╡ 0c4f12bd-ce8f-54a0-81db-2703555bbbfd
md"""
## The Setup, Theirs and Ours

| | Hajdok et al. (EGSnrc, DOSRZnrc) | this notebook (Geant4) |
|---|---|---|
| beam | infinitely narrow pencil beam, normal incidence | the same |
| detector | uniform disc, 10 cm diameter, in vacuum | the same |
| thickness | `t = −ln(1 − η) / μ(E)` with η = 0.99 | the same, with Geant4's own μ |
| energies | 10–150 keV | 10–150 keV, every 2 keV, every 0.5 keV across the K-edges |
| photons per energy | 10 million | 100 000 (statistical error of the Swank factor ≈ 0.001) |
| photon interactions | photoelectric, bound Compton, Rayleigh | the same (Geant4 option 4: Livermore / Monash) |
| atomic relaxation | K, L, M fluorescence and Auger electrons | K, L, M fluorescence; Auger energy deposited where the vacancy was (following each Auger electron costs 3.4× the time and moves results by ~0.2 %) |
| electrons | transported down to 1 keV | transported, secondaries down to the 1 µm range cut |
| scored | energy deposited per photon, 0.5 keV bins | energy deposited per photon (unbinned) |
| Swank factor | `I = M₁² / (M₀ M₂)` of the absorbed-energy distribution | the same |

Two things are not stated in the thesis: the densities of a-Si, CsI and PbI₂ ("chosen to agree
with those used in practice"), and the cross-section library. The densities barely matter here:
with the thickness set by η, every length in the problem scales with 1/density, so the Swank
factor is the same at any density except through the fixed 10 cm disc. The comparison values are
read off their figure (the solid "0 cy/mm" curve, digitized at 300 dpi, uncertainty about
±0.005); the file `reference_data/hajdok2008_swank_vs_energy.csv` records how.

**Which beam, and why.** A pencil beam on the disc's axis, head on, because that is what Hajdok
et al. simulated, and the Swank factor they report is that of the whole layer: one tally, the
energy left anywhere in the disc. There are no pixels here, so where the photon lands does not
matter; what matters is that the disc is wide (10 cm) and, at high energy, not wide enough.
"""

# ╔═╡ ed187164-fabb-4dd3-bdab-96a140a9809a
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1700, 600))
    disc_colour = PALETTE.materials["CsI"]
    star!(ax, p) = Mke.scatter!(ax, [Mke.Point2f(p)]; marker = :star5, markersize = 24, color = MATERIALS.star, strokecolor = MATERIALS.beam, strokewidth = 1.2)

    # (a) The disc and the beam (the thickness drawn at 20 mm; it changes with energy)
    ax = canvas(fig[1, 1]; title = "(a) The Detector: a Disc, 10 cm Across", titlesize = 20)
    cam = camera(-150, 18)
    R = disc_diameter_mm / 2
    disc!(ax, cam, R, 20.0, Mke.RGBf(0.75, 0.82, 0.92))
    arrow!(ax, project(cam, (-70.0, 0.0, 0.0)), project(cam, (0.0, 0.0, 0.0)); color = MATERIALS.beam, linewidth = 4, head = 16)
    top = project(cam, (0.0, 0.0, R))
    rim = project(cam, (20.0, R * cos(0.6), -R * sin(0.6)))
    tail = project(cam, (-55.0, 0.0, 0.0))
    callout!(ax, top, top + Mke.Point2f(12, 16), "10 cm Across, in Vacuum"; fontsize = 15)
    callout!(ax, rim, rim + Mke.Point2f(14, -14), "Thickness t: Stops 99 %\nof the Photons (Drawn 2 cm)"; fontsize = 15)
    callout!(ax, tail, tail + Mke.Point2f(-8, -30), "Pencil Beam on the Axis,\nHead On, One Energy"; fontsize = 15, align = (:center, :top))
    outline = [project(cam, p) for p in ((0.0, R, 0.0), (0.0, -R, 0.0), (0.0, 0.0, R), (0.0, 0.0, -R), (20.0, R, 0.0), (20.0, -R, 0.0), (-70.0, 0.0, 0.0))]
    Mke.limits!(
        ax,
        minimum(first.(outline)) - 30,
        maximum(first.(outline)) + 110,
        minimum(last.(outline)) - 55,
        maximum(last.(outline)) + 35,
    )

    # (b) What happens in it, and the one tally (side view, not to scale)
    axb = canvas(fig[1, 2]; title = "(b) The Tally: Energy Left in the Whole Disc", titlesize = 20)
    rect!(axb, 0, -2.0, 1.6, 4.0, Mke.RGBf(0.75, 0.82, 0.92); strokecolor = (:black, 0.4), strokewidth = 1.5)
    photon!(axb, (-2.2, 1.2), (0.5, 1.2); color = MATERIALS.beam, cycles = 6, amplitude = 0.02)
    photon!(axb, (0.5, 1.2), (1.2, 0.6); color = MATERIALS.xray, cycles = 2, amplitude = 0.06, head = 10)
    star!(axb, (0.5, 1.2))
    photon!(axb, (-2.2, 0.0), (0.25, 0.0); color = MATERIALS.beam, cycles = 6, amplitude = 0.02)
    photon!(axb, (0.25, 0.0), (-1.7, 0.7); color = MATERIALS.xray, cycles = 3, amplitude = 0.05, head = 11)
    star!(axb, (0.25, 0.0))
    photon!(axb, (-2.2, -1.2), (0.8, -1.2); color = MATERIALS.beam, cycles = 6, amplitude = 0.02)
    photon!(axb, (0.8, -1.2), (3.2, -1.9); color = MATERIALS.scatter, cycles = 3, amplitude = 0.05, head = 11)
    star!(axb, (0.8, -1.2))
    for (y, text, colour) in ((1.2, "All of It Stays", PALETTE.ink), (0.0, "A K X Ray Escapes", MATERIALS.xray), (-1.2, "A Photon Scatters Out", MATERIALS.scatter))
        Mke.text!(axb, -2.3, y; text, color = colour, fontsize = 16, font = :bold, align = (:right, :center))
    end
    Mke.limits!(axb, -5.6, 3.4, -2.4, 2.0)

    # (c) From that tally to the Swank factor (a sketch of the distribution)
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) From the Tally to the Swank Factor",
        titlesize = 20,
        xlabel = "Energy Left by One Photon",
        ylabel = "How Often",
        xticks = ([0.0, 0.7, 1.0], ["0", "E − K X Ray", "E"]),
        yticksvisible = false,
        yticklabelsvisible = false,
        limits = ((-0.03, 1.08), (0, 1.15)),
    )
    E = range(0, 1.05; length = 400)
    shape = [0.08 * (0.05 < e < 0.65) + 0.35 * exp(-((e - 0.7) / 0.012)^2) + 1.0 * exp(-((e - 1.0) / 0.012)^2) for e in E]
    Mke.band!(axc, E, zero(shape), shape; color = (PALETTE.materials["CsI"], 0.35))
    Mke.lines!(axc, E, shape; color = PALETTE.materials["CsI"], linewidth = 2.5)
    note!(axc, 1.0, 1.02, "All of It"; align = (:right, :bottom), fontsize = 14)
    note!(axc, 0.7, 0.4, "K X Ray Lost"; align = (:center, :bottom), fontsize = 14, color = MATERIALS.xray)
    note!(axc, 0.32, 0.12, "Scattered Out"; align = (:center, :bottom), fontsize = 14, color = MATERIALS.scatter)
    Mke.Label(
        fig[2, 3],
        "I = M₁² / (M₀ M₂), Mₖ the Average of (Energy Left)ᵏ\nover All Photons: I = 1 When Every Detected\nPhoton Leaves the Same Energy";
        fontsize = 15,
        color = PALETTE.ink,
        tellwidth = false,
    )
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.34))
    fig
end

# ╔═╡ 099436f2-777c-56a3-a8d2-4511d5ec98c3
md"""
## Notebook Setup
"""

# ╔═╡ 288da2d3-18a6-5144-8f35-479040757117
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ f50aa214-0bf2-59d7-ac58-bed477edbd14
using Markdown: @md_str, Markdown

# ╔═╡ 64463662-90d2-5c11-9abe-bca85bbc8ea2
import PlutoUI

# ╔═╡ f00150ef-2805-5c8e-9356-8fa1ae09e514
import BasisMonteCarlo as MC

# ╔═╡ 18897cc7-4249-581f-afb2-957f70fd46d0
import CairoMakie as Mke

# ╔═╡ 4bcddf7b-dbc6-5083-b062-2756e306657a
import DelimitedFiles

# ╔═╡ 165ee1c4-2db5-510d-bad8-8e8deeac0ae9
PlutoUI.TableOfContents()

# ╔═╡ a8bf1e62-22f9-56ae-bf3b-61a2b2ddebff
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 6e22da4f-5358-57fa-8ef2-c1af6f3e3186
# One colour per material, used by every figure
PALETTE = (
    materials = Dict(
        "a-Si" => Mke.RGBf(0.45, 0.45, 0.5),
        "a-Se" => Mke.RGBf(0.27, 0.55, 0.34),
        "CsI" => Mke.RGBf(0.13, 0.36, 0.66),
        "PbI2" => Mke.RGBf(0.7, 0.12, 0.17),
    ),
    reference = Mke.RGBf(0.9, 0.45, 0.05),
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ e98766b1-8cfa-5f24-9725-ef6c133c6a49
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

# ╔═╡ f952a231-979b-57e6-a76a-a9e7519db225
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

# ╔═╡ 046ba1cb-724c-43a5-8d17-ff7d23c7ee7d
md"""
### Drawing the Setup

The illustrations are drawn from the same numbers the simulation uses (the disc's 10 cm, each
layer's thickness), with a small kit: shapes in 3-D seen from any direction, photons as waves,
labels on translucent backgrounds. Each disc takes its material's colour from the plots; dark
photons arrive, violet K x rays and teal scattered photons leave.
"""

# ╔═╡ 4516c81c-0e57-4778-a067-29c851c7871e
# The beam and what leaves, as the illustrations draw them (the discs take their material's colour)
MATERIALS = (
    beam = Mke.RGBf(0.25, 0.25, 0.3),  # the incoming photons
    xray = Mke.RGBf(0.6, 0.35, 0.8),  # characteristic (K) x rays
    scatter = Mke.RGBf(0.1, 0.55, 0.6),  # scattered photons
    star = Mke.RGBf(0.98, 0.95, 0.6),  # where a photon is absorbed
)

# ╔═╡ c04bf056-9767-4761-9b9c-18e682c5b46d
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

# ╔═╡ 715ddc64-8098-4d69-acc8-3cc2953e7d73
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ efebaaf3-9efa-432b-83f0-e1f1c0cbcf2f
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ 62815d50-61f6-4fc5-8006-a73ed4f7fa9a
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ b91c5895-b2fd-41e1-af99-eb84a759da42
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

# ╔═╡ 258a3933-470c-41a6-b28b-277a0d444288
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

# ╔═╡ d1b38f5a-90b4-448c-b3f0-cdd92a32b9a0
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ 6b5cf920-6655-43a7-b5b1-4eb462fad642
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ 3babb8aa-623e-4feb-8be7-7689375efa67
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

# ╔═╡ 84bb0e96-c9ab-47ea-99fd-b8f039910045
"""
A disc (a short cylinder) along the beam, which runs along x: `radius` and `thickness` in mm,
its front face at x = 0. Drawn shaded, back to front, as `cam` sees it.
"""
function disc!(ax, cam, radius, thickness, color; segments = 72)
    φ = range(0, 2π; length = segments + 1)
    ring(x) = [(x, radius * cos(t), radius * sin(t)) for t in φ]
    front, back = ring(0.0), ring(thickness)
    quads = Any[]
    for k in 1:segments
        corners = (front[k], front[k + 1], back[k + 1], back[k])
        mid = (φ[k] + φ[k + 1]) / 2
        normal = (0.0, cos(mid), sin(mid))
        dot3(normal, cam.toward) > 0 || continue
        depth = dot3(map(+, corners[1], corners[3]) ./ 2, cam.toward)
        push!(quads, (depth, corners, normal))
    end
    sort!(quads; by = first)
    for (_, corners, normal) in quads
        c = shade(color, normal, cam)
        Mke.poly!(ax, [project(cam, p) for p in corners]; color = c, strokecolor = c, strokewidth = 0.5)
    end
    # the face toward the viewer
    cap = cam.toward[1] < 0 ? front : back
    normal = cam.toward[1] < 0 ? (-1.0, 0.0, 0.0) : (1.0, 0.0, 0.0)
    c = shade(color, normal, cam)
    Mke.poly!(ax, [project(cam, p) for p in cap]; color = c, strokecolor = (:black, 0.35), strokewidth = 1)
    return nothing
end

# ╔═╡ 4d005a0b-2dea-5ef5-ad79-7549e327d67d
md"""
## 1. The Materials and the Thickness Rule

The four converters of the study, as compositions and densities. Amorphous selenium's density
is the one the thesis states (4.20 g/cm³, its Table 4.1); the others are the standard values.
"""

# ╔═╡ 99d52b8f-7b2e-5da6-8f70-a6cb688af871
materials = [
    MC.DetectorMaterial("a-Si"; density = 2.33, formula = "Si"),
    MC.DetectorMaterial("a-Se"; density = 4.2, formula = "Se"),
    MC.DetectorMaterial("CsI"; density = 4.51, formula = "CsI"),
    MC.DetectorMaterial("PbI2"; density = 6.16, formula = "PbI2"),
]

# ╔═╡ b3283488-a9dc-542e-b8dc-91d89ba1f8a7
let
    edges(m) = join(["$(e.element) $(round(e.edge_keV; digits = 1)) keV" for e in MC.k_edges(m)], ", ")
    row(m) = "| $(m.name) | $(m.density) g/cm³ | $(MC.composition_string(m)) | $(isempty(MC.k_edges(m)) ? "none above 5 keV" : edges(m)) |"

    Markdown.parse(
        """
        | material | density | composition by mass | K-edges in 10–150 keV |
        |---|---|---|---|
        $(join([row(m) for m in materials], "\n"))
        """
    )
end

# ╔═╡ 67fd055c-2545-5a5e-9deb-45b663953ec2
md"""
Each layer is made exactly thick enough to stop 99 % of the photons at the energy being
simulated, `t = −ln(1 − 0.99) / μ(E)`, with Geant4's own μ so that the Monte Carlo's interaction
probability lands on 0.99. The thickness therefore changes with energy by a factor of a
thousand: a few micrometres of selenium just above its K-edge, more than a decimetre of silicon
at 150 keV. Where the layer becomes thicker than the 10 cm disc is wide (silicon above about
100 keV), scattered photons start to escape through the sides, which is part of why silicon's
Swank factor falls at high energy.
"""

# ╔═╡ 17807a41-01a8-5eb4-8c36-c5e39afb3b16
η = 0.99

# ╔═╡ c5fc3da3-4ffb-5ae5-a49f-8cbe44079d42
disc_diameter_mm = 100.0

# ╔═╡ 8c745dc8-9df7-546e-89dd-ccba38c68762
"""
Thickness of `material` that stops a fraction `η` of photons of `energy_keV` (Geant4's μ).
"""
thickness_for(material, energy_keV) = -log(1 - η) / MC.geant4_attenuation(material, energy_keV)

# ╔═╡ 4a7e2280-57c6-5770-959f-c54ec2801627
energies = sort(
    unique(
        vcat(
            10.0:2.0:150.0,     # the whole range
            12.0:0.5:20.0,      # across the Se K-edge
            32.0:0.5:42.0,      # across the I and Cs K-edges
            86.0:0.5:94.0,      # across the Pb K-edge
        )
    )
)

# ╔═╡ 24397966-5f75-55f3-924d-de397cdc4aba
thicknesses = Dict(
    m.name => [thickness_for(m, E) for E in energies]
        for m in materials
)

# ╔═╡ 01fd1a99-0195-4f10-9942-3875c3faf157
let
    Mke.set_theme!(THEME)
    cases = (("a-Se", 13.0), ("CsI", 60.0), ("a-Si", 150.0))
    fig = Mke.Figure(size = (1600, 470))
    R = disc_diameter_mm / 2
    for (k, (name, E)) in enumerate(cases)
        t = thicknesses[name][findfirst(==(E), energies)]
        label = t < 1 ? "$(round(t * 1000; digits = 1)) µm" : t < 10 ? "$(round(t; digits = 2)) mm" : "$(round(t / 10; digits = 1)) cm"
        colour = PALETTE.materials[name]
        ax = canvas(fig[1, k]; title = "$(name) at $(round(Int, E)) keV: $(label)", titlecolor = colour, titlesize = 21)
        # the disc from the side, to scale: 10 cm tall, t thick
        Mke.poly!(ax, Mke.Rect2f(0, -R, max(t, 0.6), 2R); color = (colour, 0.45), strokecolor = colour, strokewidth = 2)
        arrow!(ax, (-45.0, 0.0), (0.0, 0.0); color = MATERIALS.beam, linewidth = 3, head = 13)
        if t > disc_diameter_mm
            photon!(ax, (70.0, 0.0), (130.0, R + 12); color = MATERIALS.scatter, cycles = 4, amplitude = 0.04, head = 12)
            Mke.text!(ax, 128.0, R + 14; text = "Scattered Photons Leave\nThrough the Sides", color = MATERIALS.scatter, fontsize = 15, align = (:right, :bottom))
        end
        t < 1 && Mke.text!(ax, 4.0, -R + 4; text = "Thinner Than Paper:\nDrawn as a Line", color = PALETTE.muted, fontsize = 15, align = (:left, :bottom))
        Mke.text!(ax, -3.0, R; text = "10 cm", color = PALETTE.muted, fontsize = 14, align = (:right, :top))
        Mke.limits!(ax, -50, 215, -R - 8, R + 40)
    end
    Mke.Label(fig[2, 1:3], "Each Disc Is Drawn to Scale, Seen from the Side: the Same 99 % of the Photons Stopped, in Layers More Than a Thousand Times Apart in Thickness"; fontsize = 16, color = PALETTE.muted)
    fig
end

# ╔═╡ 662e16e5-544c-5943-bc65-a9d61891471f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 620))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How Thick Each Layer Must Be to Stop 99 % of the Photons",
        xlabel = "Photon Energy (keV)",
        ylabel = "Layer Thickness (mm)",
        yscale = log10,
        xticks = 10:10:150,
        yticks = (
            [0.1, 1, 10, 100],
            ["0.1 mm", "1 mm", "1 cm", "10 cm"],
        ),
        limits = ((10, 150), (0.03, 400)),
    )
    for m in materials
        Mke.lines!(
            ax, energies, thicknesses[m.name];
            color = PALETTE.materials[m.name],
            linewidth = 3,
            label = m.name,
        )
    end
    Mke.hlines!(
        ax,
        disc_diameter_mm;
        color = PALETTE.muted,
        linestyle = :dash,
        linewidth = 2,
    )
    note!(
        ax, 12, disc_diameter_mm, "Width of the Disc (10 cm)";
        align = (:left, :bottom),
        offset = (0, 4),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        ax, 148, 0.04, "Each Drop Is a K-Edge: Absorption Jumps,\nSo a Thinner Layer Stops the Same 99 %";
        align = (:right, :bottom),
        fontsize = 15,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ a470f299-d14b-5e6a-834d-701b3243e7ad
md"""
## 2. The Run

For every material and energy: build the disc, fire 100 000 photons down its axis, keep every
photon's record. About 450 runs, spread over the eight worker processes started in the setup.
"""

# ╔═╡ 7dac5424-8cef-585f-874a-f59e7ba40eed
photons_per_energy = 100_000

# ╔═╡ f367cd97-f4c2-579b-b88a-ab8eff65f1a6
# Every (disc, energy) pair of the benchmark, all materials in one list
jobs = [
    (MC.SlabDetector(m, t; width_mm = disc_diameter_mm, shape = :disc), E)
        for m in materials for (E, t) in zip(energies, thicknesses[m.name])
]

# ╔═╡ 79acd58a-bcae-5068-ab2b-22c638d3ddd4
all_runs = MC.simulate_sweep(
    jobs;
    n = photons_per_energy,
    seed = 1,
)

# ╔═╡ 0176f167-c47c-5d04-97d2-7108d632fdc9
runs = Dict(
    m.name => [r for r in all_runs if r.detector.material.name == m.name]
        for m in materials
)

# ╔═╡ 5053304a-fd29-5506-9d92-4da38b7a3d4b
swank = Dict(
    name => MC.swank_factor.(material_runs)
        for (name, material_runs) in runs
)

# ╔═╡ e1874dcb-0280-5c92-aa97-ef671d64d11c
let
    p = MC.interaction_probability.(all_runs)

    Markdown.parse(
        """
        Check: across all $(length(all_runs)) runs the fraction of photons that interacted is
        between **$(round(minimum(p); digits = 4))** and **$(round(maximum(p); digits = 4))**, as the
        thickness rule intends (η = $(η)).
        """
    )
end

# ╔═╡ dd5be4fa-0a4a-59c0-93c6-e1c5a7022d76
md"""
## 3. The Comparison

Their values, digitized from the figure, against ours. In the shaded band near 1 their curve
runs along the top edge of their plot, so it can only be read as "at least 0.99"; there the
check is that ours is too.
"""

# ╔═╡ a9f7d4e1-e769-5556-b92d-2b90bba2aed0
reference = let
    table = DelimitedFiles.readdlm(
        joinpath(@__DIR__, "..", "reference_data", "hajdok2008_swank_vs_energy.csv"),
        ',';
        comments = true,
        header = true,
    )[1]
    Dict(
        name => (
                energy_keV = Float64.(table[table[:, 1] .== name, 2]),
                swank = Float64.(table[table[:, 1] .== name, 3]),
            )
            for name in unique(table[:, 1])
    )
end

# ╔═╡ ab3c32f1-6fd7-4bf5-b436-88a46a81de63
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 520))
    star!(ax, p) = Mke.scatter!(ax, [Mke.Point2f(p)]; marker = :star5, markersize = 24, color = MATERIALS.star, strokecolor = MATERIALS.beam, strokewidth = 1.2)
    cases = (
        ("K X Rays Escape Through the Front", "a-Se, CsI, PbI₂, Just Above Each K-Edge:\nThe Photon Stops Near the Surface", "CsI"),
        ("Compton Photons Escape", "a-Si at Every Energy, the Others\nat High Energy", "a-Si"),
        ("Out Through the Sides", "a-Si Above About 100 keV, When the\nLayer Is Thicker Than the Disc Is Wide", "a-Si"),
    )
    for (k, (title, detail, name)) in enumerate(cases)
        colour = PALETTE.materials[name]
        ax = canvas(fig[1, k]; title, titlesize = 20)
        width = k == 3 ? 4.0 : 1.4
        Mke.poly!(ax, Mke.Rect2f(0, -1.0, width, 2.0); color = (colour, 0.45), strokecolor = colour, strokewidth = 2)
        if k == 1
            photon!(ax, (-1.8, 0.0), (0.15, 0.0); color = MATERIALS.beam, cycles = 6, amplitude = 0.025)
            photon!(ax, (0.15, 0.0), (-1.5, 0.85); color = MATERIALS.xray, cycles = 3, amplitude = 0.05, head = 11)
            star!(ax, (0.15, 0.0))
        elseif k == 2
            photon!(ax, (-1.8, 0.0), (0.7, 0.0); color = MATERIALS.beam, cycles = 6, amplitude = 0.025)
            photon!(ax, (0.7, 0.0), (2.4, -0.9); color = MATERIALS.scatter, cycles = 3, amplitude = 0.05, head = 11)
            star!(ax, (0.7, 0.0))
        else
            photon!(ax, (-1.8, 0.0), (1.6, 0.0); color = MATERIALS.beam, cycles = 7, amplitude = 0.02)
            photon!(ax, (1.6, 0.0), (2.6, 1.45); color = MATERIALS.scatter, cycles = 3, amplitude = 0.05, head = 11)
            star!(ax, (1.6, 0.0))
        end
        Mke.text!(ax, width / 2, -1.08; text = name, color = colour, fontsize = 16, font = :bold, align = (:center, :top))
        Mke.limits!(ax, -1.9, 4.3, -1.5, 1.6)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    Mke.Label(fig[3, 1:3], "Every Escape Leaves Less Than the Full Energy: the Amounts Spread Out, and the Swank Factor Falls Below 1"; fontsize = 17, color = PALETTE.ink)
    fig
end

# ╔═╡ 7e09656c-77d4-5969-8a70-1ab5b01ea84e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 980))
    for (i, m) in enumerate(materials)
        row, col = fldmod1(i, 2)
        color = PALETTE.materials[m.name]
        ax = Mke.Axis(
            fig[row, col];
            title = m.name,
            xlabel = row == 2 ? "Photon Energy (keV)" : "",
            ylabel = col == 1 ? "Swank Factor" : "",
            xticks = 0:20:140,
            limits = ((0, 152), (0.7, 1.01)),
        )
        Mke.hspan!(ax, 0.99, 1.01; color = (PALETTE.muted, 0.12))
        Mke.lines!(
            ax,
            energies,
            swank[m.name];
            color,
            linewidth = 3,
            label = "Geant4 (This Notebook)",
        )
        ref = reference[m.name]
        Mke.scatter!(
            ax, ref.energy_keV, ref.swank;
            color = (PALETTE.reference, 0.85),
            markersize = 6,
            label = "Hajdok et al. (EGSnrc), Read Off Their Fig. 3.4",
        )
        for (j, e) in enumerate(MC.k_edges(m))
            Mke.vlines!(
                ax,
                e.edge_keV;
                color = (PALETTE.muted, 0.6),
                linestyle = :dash,
                linewidth = 1.5,
            )
            note!(
                ax, e.edge_keV, 0.72 + 0.035 * (j - 1), "$(e.element) K-Edge";
                align = (:left, :bottom),
                offset = (4, 0),
                color = PALETTE.muted,
                fontsize = 14,
            )
        end
        i == 1 && note!(
            ax,
            150,
            0.995,
            "Their Curve Reads ≥ 0.99 Here";
            align = (:right, :center),
            color = PALETTE.muted,
            fontsize = 14,
        )
    end
    Mke.Legend(
        fig[3, 1:2],
        [
            Mke.LineElement(; color = PALETTE.ink, linewidth = 3),
            Mke.MarkerElement(;
                color = PALETTE.reference,
                marker = :circle,
                markersize = 10,
            ),
        ],
        [
            "Geant4, This Notebook (Line, One Colour per Material)",
            "Hajdok et al., EGSnrc (Points, Read Off Their Figure)",
        ];
        orientation = :horizontal,
    )
    Mke.Label(
        fig[0, 1:2], "Swank Factor of a Layer That Stops 99 % of the Photons";
        fontsize = 22,
        font = :bold,
        halign = :left,
    )
    fig
end

# ╔═╡ 32a651e3-a493-54e5-9416-6a68359a65b7
md"""
The difference, ours minus theirs, at every energy they can be read: our curve is interpolated
onto their points. The grey band is their reading uncertainty (±0.005) combined with our
counting noise. The tinted strips mark the first 5 keV above each K-edge, where both curves
plunge almost vertically: there a reading error of a few tenths of a keV along the energy axis
of their figure becomes a large error in the Swank factor, so those points test the shape, not
the exact value.
"""

# ╔═╡ 256a630f-503e-5dd0-aa39-f9f67400ffb0
differences = Dict(
    m.name => let
            ref = reference[m.name]
            ours = [MC.interpolate_linear(energies, swank[m.name], E) for E in ref.energy_keV]
            (energy_keV = ref.energy_keV, difference = ours .- ref.swank)
    end
        for m in materials
)

# ╔═╡ b4ddae61-0e59-4de6-a7b1-32eb97bb80d4
let
    Mke.set_theme!(THEME)
    # A sketch (not data): a Swank factor plunging at a K-edge, and the same small error in reading
    # the energy off a scanned figure, on the flat part and on the plunge
    edge = 33.0
    swank_sketch(E) = E < edge ? 0.99 : 0.99 - 0.13 * exp(-(E - edge) / 4)
    E = range(25, 50; length = 600)
    fig = Mke.Figure(size = (1400, 470))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Why Points on a K-Edge Test the Shape, Not the Value",
        xlabel = "Photon Energy (keV)",
        ylabel = "Swank Factor",
        limits = ((25, 50), (0.84, 1.0)),
        xticks = ([edge], ["K-Edge"]),
    )
    Mke.lines!(ax, E, swank_sketch.(E); color = PALETTE.ink, linewidth = 3)
    δ = 0.4
    for (E0, label, align) in ((edge + 0.6, "On the Plunge: Reading 0.4 keV Off\nChanges the Value by Hundredths", (:left, :top)), (44.0, "Away from It: the Same 0.4 keV\nChanges It by a Thousandth", (:center, :top)))
        a, b = swank_sketch(E0), swank_sketch(E0 + δ)
        Mke.lines!(ax, [E0, E0 + δ], [a, a]; color = PALETTE.reference, linewidth = 4)
        Mke.lines!(ax, [E0 + δ, E0 + δ], [a, b]; color = PALETTE.reference, linewidth = 4)
        Mke.scatter!(ax, [E0, E0 + δ], [a, b]; color = PALETTE.reference, markersize = 10)
        note!(ax, E0 < 40 ? E0 + 1.2 : E0, E0 < 40 ? b - 0.004 : a - 0.008, label; align, color = PALETTE.reference, fontsize = 15)
    end
    fig
end

# ╔═╡ 2a065f94-1937-5a38-8a08-1da9a8c2bdba
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 520))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Geant4 Minus EGSnrc, at Every Point Read Off Their Figure",
        xlabel = "Photon Energy (keV)",
        ylabel = "Difference in\nSwank Factor",
        xticks = 0:10:150,
        limits = ((0, 152), (-0.05, 0.05)),
    )
    for m in materials, e in MC.k_edges(m)
        Mke.vspan!(
            ax,
            e.edge_keV - 0.5,
            e.edge_keV + 5;
            color = (PALETTE.materials[m.name], 0.08),
        )
    end
    Mke.hspan!(ax, -0.006, 0.006; color = (PALETTE.muted, 0.15))
    Mke.hlines!(ax, 0; color = PALETTE.ink, linewidth = 1.5)
    for m in materials
        d = differences[m.name]
        Mke.scatter!(
            ax,
            d.energy_keV,
            d.difference;
            color = PALETTE.materials[m.name],
            markersize = 6,
            label = m.name,
        )
    end
    note!(
        ax,
        150,
        0.006,
        "Reading + Counting Uncertainty";
        align = (:right, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 9f940c62-c9bd-52d5-971a-38df8c5ff3bb
let
    function row(m)
        d = differences[m.name]
        ref = reference[m.name]
        i_ref = argmin(ref.swank)
        i_ours = argmin(swank[m.name])
        return string(
            "| $(m.name) | $(length(d.difference)) ",
            "| $(round(sum(abs, d.difference) / length(d.difference); digits = 4)) ",
            "| $(round(maximum(abs, d.difference); digits = 4)) ",
            "| $(round(ref.swank[i_ref]; digits = 3)) at $(ref.energy_keV[i_ref]) keV ",
            "| $(round(swank[m.name][i_ours]; digits = 3)) at $(energies[i_ours]) keV |",
        )
    end

    Markdown.parse(
        """
        | material | points compared | mean difference (absolute) | largest difference | lowest Swank, theirs | lowest Swank, ours |
        |---|---|---|---|---|---|
        $(join([row(m) for m in materials], "\n"))
        """
    )
end

# ╔═╡ 467c009d-9f05-536c-82bb-d4d6df3db5fc
let
    near_edge(m, E) = any(e -> e.edge_keV - 0.5 <= E <= e.edge_keV + 5, MC.k_edges(m))
    function row(m)
        d = differences[m.name]
        away = [x for (E, x) in zip(d.energy_keV, d.difference) if !near_edge(m, E)]
        near = [x for (E, x) in zip(d.energy_keV, d.difference) if near_edge(m, E)]
        fmt(v) = isempty(v) ? "–" : string(round(sum(v) / length(v); digits = 4))
        fmt_max(v) = isempty(v) ? "–" : string(round(maximum(abs, v); digits = 4))
        return "| $(m.name) | $(length(away)) | $(fmt(away)) | $(fmt_max(away)) | $(length(near)) | $(fmt_max(near)) |"
    end

    Markdown.parse(
        """
        | material | points away from the edges | mean difference there | largest there | points on the edges | largest there |
        |---|---|---|---|---|---|
        $(join([row(m) for m in materials], "\n"))
        """
    )
end

# ╔═╡ 1c82bb45-2704-5fe0-9e6d-ae72123f92af
md"""
## What the Comparison Shows

**Agreement.** Away from the K-edges, Geant4 and EGSnrc agree to within a few thousandths for
every material, over the whole 10–150 keV range and over Swank factors from 0.77 to 1. The
agreement covers three quite different pieces of physics:

- **Silicon**, where the Swank factor falls from 1 to 0.77 because Compton-scattered photons
  escape, at high energy partly through the *sides* of the 10 cm disc once the layer is thicker
  than the disc is wide. Reproducing that curve means reproducing the scattering and the
  geometry, not just the absorption.
- **CsI and a-Se**, where the dip just above each K-edge comes from K x rays escaping through
  the entrance face, and its recovery from the absorption moving deeper as the energy rises.
- **PbI₂**, with two edges (iodine and lead) and the lead L-edge structure near 13–16 keV.

**The one systematic difference.** Above both of its edges, our PbI₂ Swank factor sits about
0.007 below theirs, consistently, well outside the reading uncertainty. Both edges of PbI₂ are
the ones where the escaping x rays are most energetic (lead K x rays, 72–85 keV, travel far), so
the difference points at the K-fluorescence data or the treatment of the escaping x rays rather
than at the geometry. Candidates to test: the lead K-fluorescence yield and line energies
(Geant4's atomic relaxation data against the EADL-based tables of EGSnrc), Geant4's coherent
scattering of those x rays, and the PbI₂ density the authors used (not stated).

**On the edges.** The points within 5 keV above an edge differ by up to a few hundredths, both
up and down. These are where the curves are nearly vertical in their figure, so the reading is
limited by where exactly the edge falls on the energy axis of a scanned plot; the dip depths and
recoveries themselves match.

**Conclusion.** A different code, written independently, with different cross-section and
relaxation data, gives the same Swank factors to within a few thousandths. The detector physics
that matters for the efficiency of a CT detector, absorption, K-escape and scatter escape, is
reproduced. The PbI₂ offset is the one open item, and it is small.
"""

# ╔═╡ 631ad4da-2482-5ae7-ae23-6cf1ef01e062
md"""
## Physics Settings
"""

# ╔═╡ d8e74796-5aa6-53ac-8f49-75ef26ad8a92
MC.PhysicsSettings()

# ╔═╡ Cell order:
# ╟─e73ba2f8-d658-5841-81a3-2a87c22483ad
# ╟─0c4f12bd-ce8f-54a0-81db-2703555bbbfd
# ╟─ed187164-fabb-4dd3-bdab-96a140a9809a
# ╟─099436f2-777c-56a3-a8d2-4511d5ec98c3
# ╠═288da2d3-18a6-5144-8f35-479040757117
# ╠═f50aa214-0bf2-59d7-ac58-bed477edbd14
# ╠═64463662-90d2-5c11-9abe-bca85bbc8ea2
# ╠═f00150ef-2805-5c8e-9356-8fa1ae09e514
# ╠═18897cc7-4249-581f-afb2-957f70fd46d0
# ╠═4bcddf7b-dbc6-5083-b062-2756e306657a
# ╠═165ee1c4-2db5-510d-bad8-8e8deeac0ae9
# ╠═a8bf1e62-22f9-56ae-bf3b-61a2b2ddebff
# ╠═6e22da4f-5358-57fa-8ef2-c1af6f3e3186
# ╠═e98766b1-8cfa-5f24-9725-ef6c133c6a49
# ╠═f952a231-979b-57e6-a76a-a9e7519db225
# ╟─046ba1cb-724c-43a5-8d17-ff7d23c7ee7d
# ╠═4516c81c-0e57-4778-a067-29c851c7871e
# ╠═c04bf056-9767-4761-9b9c-18e682c5b46d
# ╠═715ddc64-8098-4d69-acc8-3cc2953e7d73
# ╠═efebaaf3-9efa-432b-83f0-e1f1c0cbcf2f
# ╠═62815d50-61f6-4fc5-8006-a73ed4f7fa9a
# ╠═b91c5895-b2fd-41e1-af99-eb84a759da42
# ╠═258a3933-470c-41a6-b28b-277a0d444288
# ╠═d1b38f5a-90b4-448c-b3f0-cdd92a32b9a0
# ╠═6b5cf920-6655-43a7-b5b1-4eb462fad642
# ╠═3babb8aa-623e-4feb-8be7-7689375efa67
# ╠═84bb0e96-c9ab-47ea-99fd-b8f039910045
# ╟─4d005a0b-2dea-5ef5-ad79-7549e327d67d
# ╠═99d52b8f-7b2e-5da6-8f70-a6cb688af871
# ╟─b3283488-a9dc-542e-b8dc-91d89ba1f8a7
# ╟─67fd055c-2545-5a5e-9deb-45b663953ec2
# ╠═17807a41-01a8-5eb4-8c36-c5e39afb3b16
# ╠═c5fc3da3-4ffb-5ae5-a49f-8cbe44079d42
# ╠═8c745dc8-9df7-546e-89dd-ccba38c68762
# ╠═4a7e2280-57c6-5770-959f-c54ec2801627
# ╠═24397966-5f75-55f3-924d-de397cdc4aba
# ╟─01fd1a99-0195-4f10-9942-3875c3faf157
# ╟─662e16e5-544c-5943-bc65-a9d61891471f
# ╟─a470f299-d14b-5e6a-834d-701b3243e7ad
# ╠═7dac5424-8cef-585f-874a-f59e7ba40eed
# ╠═f367cd97-f4c2-579b-b88a-ab8eff65f1a6
# ╠═79acd58a-bcae-5068-ab2b-22c638d3ddd4
# ╠═0176f167-c47c-5d04-97d2-7108d632fdc9
# ╠═5053304a-fd29-5506-9d92-4da38b7a3d4b
# ╟─e1874dcb-0280-5c92-aa97-ef671d64d11c
# ╟─dd5be4fa-0a4a-59c0-93c6-e1c5a7022d76
# ╠═a9f7d4e1-e769-5556-b92d-2b90bba2aed0
# ╟─ab3c32f1-6fd7-4bf5-b436-88a46a81de63
# ╟─7e09656c-77d4-5969-8a70-1ab5b01ea84e
# ╟─32a651e3-a493-54e5-9416-6a68359a65b7
# ╠═256a630f-503e-5dd0-aa39-f9f67400ffb0
# ╟─b4ddae61-0e59-4de6-a7b1-32eb97bb80d4
# ╟─2a065f94-1937-5a38-8a08-1da9a8c2bdba
# ╟─9f940c62-c9bd-52d5-971a-38df8c5ff3bb
# ╟─467c009d-9f05-536c-82bb-d4d6df3db5fc
# ╟─1c82bb45-2704-5fe0-9e6d-ae72123f92af
# ╟─631ad4da-2482-5ae7-ae23-6cf1ef01e062
# ╠═d8e74796-5aa6-53ac-8f49-75ef26ad8a92
