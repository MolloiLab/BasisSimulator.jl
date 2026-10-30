### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

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
difference between the two standard data sets is visible wherever it matters.
"""

# ╔═╡ dfc666d5-5ff2-502e-aeb0-49439685ac86
let
    Mke.set_theme!(THEME)

    fig = Mke.Figure(size = (1200, 400))
    ax = Mke.Axis(
        fig[1, 1];
        limits = ((0, 104), (0, 40)),
    )
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)

    # A rounded box, drawn as one polygon
    function box!(x, y, w, h; fill, stroke, dashed = false)
        r = 1.6
        θ = range(0, π / 2; length = 10)
        corners = vcat(
            [Mke.Point2f(x + w - r + r * cos(t), y + h - r + r * sin(t)) for t in θ],
            [Mke.Point2f(x + r - r * sin(t), y + h - r + r * cos(t)) for t in θ],
            [Mke.Point2f(x + r - r * cos(t), y + r - r * sin(t)) for t in θ],
            [Mke.Point2f(x + w - r + r * sin(t), y + r - r * cos(t)) for t in θ],
        )
        return Mke.poly!(
            ax, corners;
            color = fill,
            strokecolor = stroke,
            strokewidth = 2,
            linestyle = dashed ? :dash : nothing,
        )
    end

    # The four steps, left to right
    steps = (
        (
            title = "1  Describe",
            body = "the crystal (material,\ndensity, size) and the\nbeam (energy, position)",
            api = "SlabDetector\nPencilBeam",
            color = PALETTE.slab_edge,
        ),
        (
            title = "2  Simulate",
            body = "Geant4 follows each photon\nand every x ray and electron\nit creates, one at a time",
            api = "simulate_monoenergetic",
            color = PALETTE.photon,
        ),
        (
            title = "3  Record",
            body = "for every photon: the energy\nleft in the crystal, whether\nit interacted, what escaped",
            api = "MonoenergeticRun",
            color = PALETTE.xray,
        ),
        (
            title = "4  Measure",
            body = "efficiencies, spectra and\nthe energy budget, checked\nagainst Beer-Lambert",
            api = "quantum_efficiency\nswank_factor",
            color = PALETTE.kept,
        ),
    )
    w, h, gap, y = 23.0, 23.0, 3.0, 15.5
    for (i, step) in enumerate(steps)
        x = 1 + (i - 1) * (w + gap)
        box!(x, y, w, h; fill = (step.color, 0.1), stroke = step.color)
        Mke.text!(
            ax, x + 1.6, y + h - 2.0;
            text = step.title,
            font = :bold,
            fontsize = 21,
            color = step.color,
            align = (:left, :top),
        )
        Mke.text!(
            ax, x + 1.6, y + h - 7.0;
            text = step.body,
            fontsize = 16,
            color = PALETTE.ink,
            align = (:left, :top),
        )
        Mke.text!(
            ax, x + 1.6, y + 2.0;
            text = step.api,
            fontsize = 14,
            font = "DejaVu Sans Mono",
            color = step.color,
            align = (:left, :bottom),
        )
        if i < length(steps)
            Mke.arrows2d!(
                ax,
                [Mke.Point2f(x + w + 0.3, y + h / 2)],
                [Mke.Vec2f(gap - 0.6, 0)];
                color = PALETTE.muted,
                shaftwidth = 3,
                tipwidth = 12,
                tiplength = 10,
            )
        end
    end

    # The comparison data, off to the side of the main path
    box!(
        1,
        1.5,
        49,
        10;
        fill = (PALETTE.muted, 0.06),
        stroke = PALETTE.muted,
        dashed = true,
    )
    Mke.text!(
        ax, 2.6, 9.8;
        text = "Comparison only:  NIST XCOM attenuation",
        font = :bold,
        fontsize = 17,
        color = PALETTE.muted,
        align = (:left, :top),
    )
    Mke.text!(
        ax, 2.6, 3.0;
        text = "XrayAttenuation.jl, the data BasisSimulator uses",
        fontsize = 15,
        color = PALETTE.muted,
        align = (:left, :bottom),
    )
    Mke.lines!(
        ax, [50.5, 92.0, 92.0], [6.5, 6.5, 15.0];
        color = PALETTE.muted,
        linestyle = :dash,
        linewidth = 2,
    )
    Mke.arrows2d!(
        ax,
        [Mke.Point2f(92.0, 13.0)],
        [Mke.Vec2f(0, 2.2)];
        color = PALETTE.muted,
        shaftwidth = 2,
        tipwidth = 11,
        tiplength = 9,
    )

    fig
end

# ╔═╡ 41e976eb-d5a6-5afd-9c35-4670601ce1ec
md"""
## Notebook Setup
"""

# ╔═╡ 09d7b527-61f3-5c01-a16a-d4178d4d985a
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ db1d8de8-d6dc-5361-bce7-7948e59da8bc
using Markdown: @md_str, Markdown

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

# ╔═╡ 9930db7b-c02f-5cce-8043-57033d7852f6
md"""
## 1. The Detector and the Beam

The detector is one block of cadmium telluride, the crystal photon-counting CT scanners use,
1 mm deep along the beam and 50 mm square across it, in vacuum. A narrow ("pencil") beam of
photons, all of the same energy, hits the centre of its front face head on.

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
    fig = Mke.Figure(size = (1200, 640))

    # (a) The whole slab, to scale, with the beam axis drawn left to right
    ax3 = Mke.Axis3(
        fig[1, 1];
        title = "(a) The whole detector, to scale",
        aspect = :data,
        xlabel = "z (mm)",
        ylabel = "x (mm)",
        zlabel = "y (mm)",
        azimuth = 1.65π,
        elevation = 0.12π,
        protrusions = (40, 40, 40, 70),
        xticks = [-40, -20, 0],
        yticks = [-25, 0, 25],
        zticks = [-25, 0, 25],
    )
    slab = Mke.Rect3f(Mke.Vec3f(0, -w / 2, -w / 2), Mke.Vec3f(d, w, w))
    Mke.mesh!(ax3, slab; color = PALETTE.slab)
    Mke.wireframe!(ax3, slab; color = PALETTE.slab_edge, linewidth = 1.2)
    Mke.arrows3d!(
        ax3,
        [Mke.Point3f(-40, 0, 0)],
        [Mke.Vec3f(40, 0, 0)];
        color = PALETTE.photon,
        markerscale = 0.3,
    )
    Mke.text!(
        ax3, Mke.Point3f(-40, 0, 0);
        text = "photons",
        align = (:center, :bottom),
        offset = (0, 12),
        color = PALETTE.photon,
        fontsize = 18,
        font = :bold,
    )
    Mke.text!(
        ax3, Mke.Point3f(d, 0, w / 2);
        text = "CdTe crystal: $(round(Int, w)) × $(round(Int, w)) mm, $(d) mm deep",
        align = (:center, :bottom),
        offset = (0, 12),
        fontsize = 16,
        color = PALETTE.slab_edge,
    )

    # (b) A cut through the beam axis, zoomed onto the crystal's depth
    ax = Mke.Axis(
        fig[1, 2];
        title = "(b) Side view through the beam, zoomed in",
        xlabel = "depth along the beam, z (mm)",
        ylabel = "position across the face, x (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.7, d + 1.7), (-1.6, 1.5)),
    )
    Mke.poly!(
        ax, Mke.Rect2f(0, -1.6, d, 3.1);
        color = PALETTE.slab,
        strokecolor = PALETTE.slab_edge,
        strokewidth = 2,
    )
    Mke.arrows2d!(
        ax,
        [Mke.Point2f(-1.6, 0)],
        [Mke.Vec2f(1.56, 0)];
        color = PALETTE.photon,
        shaftwidth = 5,
        tipwidth = 18,
        tiplength = 18,
    )
    Mke.bracket!(
        ax, 0, -1.0, d, -1.0;
        text = "d = $(d) mm",
        orientation = :down,
        fontsize = 17,
        color = PALETTE.ink,
        textcolor = PALETTE.ink,
    )
    note!(
        ax,
        -0.82,
        0.4,
        "pencil beam\nevery photon\nhas the same energy";
        align = (:center, :center),
        color = PALETTE.photon,
    )
    note!(
        ax,
        d / 2,
        1.15,
        "CdTe";
        align = (:center, :center),
        fontsize = 20,
        color = PALETTE.slab_edge,
        alpha = 0.0,
    )
    note!(
        ax,
        0,
        -0.35,
        "front face\nz = 0";
        align = (:right, :center),
        offset = (-8, 0),
        fontsize = 15,
    )
    note!(
        ax,
        d,
        -0.35,
        "back face\nz = d";
        align = (:left, :center),
        offset = (8, 0),
        fontsize = 15,
    )
    note!(
        ax,
        -1.65,
        -1.55,
        "vacuum";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        ax,
        d + 1.65,
        -1.55,
        "vacuum";
        align = (:right, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        ax, d / 2, 0.55, "crystal goes on\nto x = ±$(round(Int, w / 2)) mm";
        align = (:center, :center),
        color = PALETTE.muted,
        fontsize = 14,
        alpha = 0.0,
    )

    Mke.colsize!(fig.layout, 1, Mke.Relative(0.46))
    Mke.colgap!(fig.layout, 50)
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

# ╔═╡ 9d277927-0f8a-52c6-82e9-e41826865c35
let
    Mke.set_theme!(THEME)

    E = collect(fine_energies)
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
    fig = Mke.Figure(size = (1200, 820))

    # Top: μ for each process, Geant4 solid, XCOM dashed
    ax = Mke.Axis(
        fig[1, 1];
        yscale = log10,
        ylabel = "chance of interacting per mm, μ (1/mm)",
        title = "How strongly CdTe stops x rays",
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
            "$(e.element) K-edge, $(round(e.edge_keV; digits = 1)) keV";
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
            ["solid: Geant4 (what the simulation uses)", "dashed: NIST XCOM (comparison)"],
        );
        valign = :top,
    )

    # Bottom: how far apart the two data sets are
    axr = Mke.Axis(
        fig[2, 1];
        xlabel = "photon energy (keV)",
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
        title = "(a) $(nphotons) photons of $(round(Int, t.energy_keV)) keV entering $(d) mm of CdTe (side view)",
        xlabel = "depth along the beam, z (mm)",
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
        "← escapes out the front";
        align = (:left, :top),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        ax,
        d + 0.97,
        0.56,
        "escapes out the back →";
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
            "incoming photon",
            "characteristic x ray",
            "electron (panel b)",
            "photon absorbed",
            "bounced, lost energy\n(Compton)",
            "bounced, kept energy\n(Rayleigh)",
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
        title = "(b) One absorption, magnified",
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
        The photon is absorbed at the dot,
        $(round(Int, 1000 * c[3])) µm inside the crystal.

        The electron it frees ($(round(ekin; digits = 1)) keV) stops
        within about a micrometre: its energy is
        deposited on the spot.

        When the atom emits a characteristic x ray
        instead (orange in panel a), that x ray travels
        hundreds of µm before it is absorbed, or it
        escapes the crystal.
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
        title = "Monte Carlo against Beer-Lambert, $(detector.material.name) $(detector.thickness_mm) mm",
        ylabel = "fraction of photons\nthat interact",
        xticks = 10:10:140,
        limits = ((8, 142), nothing),
    )
    Mke.lines!(
        ax, E, beer_lambert_geant4;
        color = PALETTE.ink,
        linewidth = 3,
        label = "Beer-Lambert, Geant4's μ",
    )
    Mke.lines!(
        ax, E, beer_lambert_xcom;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Beer-Lambert, XCOM's μ (comparison)",
    )
    Mke.scatter!(
        ax, E, p;
        color = PALETTE.photon,
        markersize = 7,
        label = "Monte Carlo: photons that interacted",
    )
    Mke.axislegend(ax; position = :lb)

    # Middle: the difference in units of counting noise
    axr = Mke.Axis(
        fig[2, 1];
        ylabel = "difference ÷\ncounting noise",
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
        axr, 140, 2.0, "green band: within ±2 × the counting noise";
        align = (:right, :bottom),
        color = PALETTE.kept,
        fontsize = 15,
    )

    # Bottom: how far BasisSimulator's XCOM Beer-Lambert sits from Geant4's
    xcom_minus_geant4 = 100 .* (beer_lambert_xcom .- beer_lambert_geant4)
    axx = Mke.Axis(
        fig[3, 1];
        xlabel = "photon energy (keV)",
        ylabel = "XCOM − Geant4\n(percentage points)",
        xticks = 10:10:140,
        limits = ((8, 142), nothing),
    )
    Mke.hlines!(axx, 0; color = PALETTE.ink, linewidth = 1.5)
    Mke.lines!(axx, E, xcom_minus_geant4; color = PALETTE.muted, linewidth = 2.5)
    note!(
        axx, 10, maximum(xcom_minus_geant4),
        "how much BasisSimulator's Beer-Lambert (XCOM) differs";
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

# ╔═╡ 21575ae6-eade-5f3f-af5e-d8b536251282
let
    Mke.set_theme!(THEME)

    E = collect(energies)
    budgets = MC.energy_budget.(sweep)
    layers = (
        (
            part = :absorbed,
            label = "stays in the crystal (the signal)",
            color = PALETTE.kept,
        ),
        (
            part = :fluorescence,
            label = "carried out by a characteristic x ray",
            color = PALETTE.xray,
        ),
        (
            part = :scattered,
            label = "carried out by a scattered photon",
            color = PALETTE.photon,
        ),
        (part = :electron, label = "carried out by an electron", color = PALETTE.electron),
        (
            part = :transmitted,
            label = "passes straight through",
            color = Mke.RGBf(0.85, 0.85, 0.87),
        ),
    )
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Where each photon's energy goes, $(detector.material.name) $(detector.thickness_mm) mm",
        xlabel = "photon energy (keV)",
        ylabel = "fraction of the photon's energy",
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
        "just above the K-edges,\nup to $(escape_peak) % of the energy\nleaves as x rays";
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

# ╔═╡ 2cd36cc8-7384-5e22-8d2d-b4acf3adbb0d
let
    Mke.set_theme!(THEME)

    r = spectrum_run
    E0 = r.energy_keV
    fig = Mke.Figure(size = (1200, 880))

    # Top: the characteristic x rays that escaped
    axe = Mke.Axis(
        fig[1, 1];
        title = "What escaped: characteristic x rays leaving the crystal ($(round(Int, E0)) keV photons)",
        yscale = log10,
        ylabel = "per keV, per\nincoming photon",
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
        title = "What stayed: energy left in the crystal by each photon",
        yscale = log10,
        xlabel = "energy (keV)",
        ylabel = "per keV, per\nincoming photon",
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
        ax, E0, 8, "full-energy peak:\nall $(round(Int, E0)) keV stayed";
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
        "scattered photons that escaped";
        align = (:left, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )

    # The key that matches each escaped line to its escape peak
    key_row(i, line) = string(
        "$(i)   $(line.element) $(line.line) x ray, $(round(line.energy_keV; digits = 1)) keV\n",
        "     escape peak at $(round(E0 - line.energy_keV; digits = 1)) keV",
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
  the quantity in BasisSimulator's Monte Carlo lookup tables.
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
        title = "What Beer-Lambert predicts and what the crystal keeps, $(detector.material.name) $(detector.thickness_mm) mm",
        ylabel = "efficiency",
        xticks = 10:10:140,
        limits = ((8, 142), (0.25, 1.04)),
    )
    Mke.lines!(
        ax, E, beer_lambert_geant4;
        color = PALETTE.ink,
        linewidth = 3,
        label = "Beer-Lambert: photons that interact",
    )
    Mke.lines!(
        ax, E, MC.quantum_efficiency.(sweep);
        color = PALETTE.photon,
        linewidth = 3,
        label = "quantum efficiency: photons that leave any energy",
    )
    Mke.lines!(
        ax, E, MC.energy_absorption_efficiency.(sweep);
        color = PALETTE.xray,
        linewidth = 3,
        label = "energy-absorption efficiency: share of energy kept",
    )
    Mke.axislegend(ax; position = :lb)

    # Bottom: the Swank factor
    axs = Mke.Axis(
        fig[2, 1];
        xlabel = "photon energy (keV)",
        ylabel = "Swank factor",
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
        "K-edges of Cd and Te";
        offset = (10, 0),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        axs, 141, minimum(swank), "1 = every detected photon leaves the same energy";
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
  its response with the Monte Carlo table BasisSimulator uses today.
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
# ╟─9930db7b-c02f-5cce-8043-57033d7852f6
# ╠═1778e101-f9ac-579b-9782-dc12f803a8d9
# ╠═9fb6be41-9274-55b0-a8dc-193e709361fc
# ╟─fea6f5fe-ec6d-514d-973f-1c6f6cbd2393
# ╟─e5aaa948-30a7-5bd6-82e0-f1ad5b3a275a
# ╟─bcf70aa4-58b9-522c-880f-e1888896c935
# ╠═4807e458-f107-5647-bc1a-1bf891280843
# ╠═befb271a-8286-565b-8512-c80c1aef4eed
# ╠═cd2de766-2eac-5fc7-9b9a-8764e0e80e07
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
# ╟─7bfbd9e9-b203-5781-a28b-d3dcf06d765d
# ╟─2dcf44e4-8d50-50e0-8e0e-3fefca54e825
# ╟─0d9b0e7c-7ff4-5632-a7ed-9231f0b0f09a
# ╟─21575ae6-eade-5f3f-af5e-d8b536251282
# ╟─265a57e2-edba-5765-b933-976cc027eb82
# ╟─beafa265-f6ce-55b2-abfd-b194220ceed5
# ╠═7b6c3fe9-caa4-5d4b-a1b7-6038524e2b43
# ╠═184a9cef-ecb7-5563-bcc9-dbc8af750c2d
# ╠═8fce8f9a-8b98-5a61-9459-6ab5c3b9fb74
# ╟─2cd36cc8-7384-5e22-8d2d-b4acf3adbb0d
# ╟─a2f74850-388c-5e6c-a526-757f0dd79798
# ╟─6c94d133-46f8-5406-bc79-1cc8106a1c4b
# ╟─68872016-40db-5305-ad2a-8ca120226c43
# ╠═b27b88bd-16e8-5270-acc0-16c74c129393
# ╟─e16fc829-1d5c-545a-ac23-c1a36c562230
