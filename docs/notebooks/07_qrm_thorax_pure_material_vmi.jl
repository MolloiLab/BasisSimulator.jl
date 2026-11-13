### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 07010001-0000-4000-8000-000000000001
md"""
# QRM-Thorax Pure-Material VMI: Dual-kVp Reference

**A GE Revolution Apex Elite rapid-kVp-switching scan (80 + 140 kVp) of a body-sized QRM-Thorax
phantom with four pure-material rods, through the package's VMI chain, every rod checked against
a noise-free scan and against theory at 50, 70, 100 and 140 keV.**

The rods (water, lipid, collagen and 5 mg/mL iodine) sit in the phantom's cardiac insert. The
scanner, acquisition and chain are those of the basis-spectral-denoising paper, as in notebook 03,
which runs the same chain on the smaller Gammex 472; notebook 08 scans this phantom with a
photon-counting CT.
"""

# ╔═╡ 8975ea65-20d3-41a6-9893-982aa954dbb6
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scan", "The Chain"],
        [
            "QRM-Thorax, Four Pure-Material Rods",
            "GE Apex Elite, 80 / 140 kVp Switching",
            "Two kVp → Two Materials → VMIs",
        ];
        limits = [(-160, 160, -112, 112), (-7.5, 7.5, -7.5, 7.5), (-0.6, 10.6, -3.8, 3.8)],
    )
    qrm_sketch!(axes[1]; dark = true)
    gantry!(
        axes[2];
        labels = ("80 ⇄ 140 kVp",),
        object = false,
        detector_colors = (COLORS.cyan,),
    )
    Mke.poly!(
        axes[2],
        Mke.Circle(Mke.Point2f(0, 0), 2.0f0);
        color = (COLORS.blue, 0.55),
        strokecolor = COLORS.night_ink,
        strokewidth = 1.5,
    )
    steps = (
        ("80 + 140 kVp Sinograms", COLORS.violet),
        ("Water + Iodine", COLORS.cyan),
        ("VMIs, 50–140 keV", COLORS.green),
    )
    for (i, (name, accent)) in enumerate(steps)
        y = 2.5 - 2.5 * (i - 1)
        node!(
            axes[3],
            (5.0, y),
            name;
            w = 8.6,
            h = 1.35,
            accent,
            fill = COLORS.night_card,
            textcolor = COLORS.night_ink,
        )
        i < 3 &&
            arrow!(axes[3], (5.0, y - 0.72), (5.0, y - 1.78); color = COLORS.night_muted)
    end
    fig
end

# ╔═╡ 27c06485-bdf2-4288-9efb-22e9b11ee3a0
md"""
| | |
|---|---|
| **Phantom** | 1600 × 1100 × 20 voxels at 0.2 mm isotropic (320 × 220 × 4 mm) |
| **Reconstruction** | 512 × 512 × 3 at 0.625 mm isotropic, a 32 cm field, 1.875 mm in z |
| **Scanner** | GE Revolution Apex Elite: 834 columns × 256 rows of 0.6 × 0.625 mm dexels |
| **Protocol** | rapid kVp switching, 80 + 140 kVp, duty cycle 0.65 / 0.35, 984 views in 0.5 s, 5 mm collimation, dose-matched to 10.07 mGy CTDIvol |

The only economy is in **z**: three saved slices and a phantom deep enough for every ray those
slices use (the workspace adds the detector guard rows the reconstruction cylinder needs).

!!! note "Why pure end-members?"
    `XA.Materials.basis_fat` is ICRU-44 adipose tissue (about 83 % triolein, 17 % water and trace
    electrolytes), not a pure lipid. `XA.Materials.basis_lipid` (H/C/O at 0.92 g/cm³) and
    `basis_collagen` (H/C/N/O at 1.26 g/cm³) are mathematically pure end members, made for clean
    water/lipid/collagen decomposition checks.
"""

# ╔═╡ 0f80b460-20be-43bd-aef4-7374aaa15a2b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 470))
    nodes = Dict(
        :phantom => (0.0, 2.6, "QRM-Thorax", "Analytic Slice, Tiled in z"),
        :low => (4.6, 3.8, "simulate!, 80 kVp", "Duty Cycle 0.65", COLORS.cyan),
        :high => (4.6, 1.4, "simulate!, 140 kVp", "Duty Cycle 0.35", COLORS.cyan),
        :basis => (9.2, 2.6, "Spectral Basis", "Each Ray's Response", COLORS.cyan),
        :calib => (
            13.8,
            0.6,
            "Calibration Draw",
            "pair_basis, composite_energy",
            COLORS.beam,
        ),
        :vmi => (13.8, 2.6, "vmi_pipeline", "Measured and Noise-Free", COLORS.green),
        :rods => (18.4, 2.6, "Per-Rod HU", "vs Noise-Free and Theory", COLORS.green),
    )
    edges = [
        :phantom => :low,
        :phantom => :high,
        :low => :basis,
        :high => :basis,
        :basis => :vmi,
        :basis => :calib,
        :calib => :vmi,
        :vmi => :rods,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-2.2, 20.6, -0.2, 4.6), w = 4.0, h = 1.2)
    fig
end

# ╔═╡ 07010002-0000-4000-8000-000000000001
md"""
## Notebook Setup
"""

# ╔═╡ 07010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 07010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 07010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

# ╔═╡ 07010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 07010003-0000-4000-8000-000000000011
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 07010003-0000-4000-8000-000000000012
import PlutoUI

# ╔═╡ 07010003-0000-4000-8000-000000000013
PlutoUI.TableOfContents()

# ╔═╡ 07010003-0000-4000-8000-000000000040
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 07010003-0000-4000-8000-000000000050
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ c0a15200-4f8c-4a6d-afdc-f069970dcf0c
md"""
### Figure Style

BasisSimulator's colours and type, and the drawing helpers every figure uses (the same cells in
every example notebook), then this notebook's own: one colour per rod and a drawing of the
phantom.
"""

# ╔═╡ da68bdcc-b970-41e2-8d8c-a8baf8869d89
# BasisSimulator's colours: the docs site's warm paper and ink, its violet and cyan accents
COLORS = (
    paper = Mke.RGBf(0.973, 0.969, 0.957), # #f8f7f4, every figure's background
    card = Mke.RGBf(0.941, 0.925, 0.894),  # #f0ece4, boxes in diagrams
    rule = Mke.RGBf(0.831, 0.816, 0.784),  # #d4d0c8, outlines and grid
    ink = Mke.RGBf(0.165, 0.145, 0.125),   # #2a2520, text
    muted = Mke.RGBf(0.42, 0.396, 0.376),  # #6b6560, secondary text
    night = Mke.RGBf(0.067, 0.067, 0.063), # #111110, the locator band
    night_card = Mke.RGBf(0.137, 0.129, 0.122),
    night_ink = Mke.RGBf(0.94, 0.925, 0.894),
    night_muted = Mke.RGBf(0.62, 0.6, 0.57),
    violet = Mke.RGBf(0.486, 0.227, 0.929),      # #7c3aed, BasisSimulator itself
    violet_soft = Mke.RGBf(0.867, 0.839, 0.996), # #ddd6fe
    cyan = Mke.RGBf(0.031, 0.569, 0.698),        # #0891b2, measurements and data
    cyan_soft = Mke.RGBf(0.647, 0.953, 0.988),   # #a5f3fc
    beam = Mke.RGBf(0.96, 0.62, 0.14),           # x rays
    red = Mke.RGBf(0.796, 0.235, 0.2),           # Julia red: calcium, failures
    green = Mke.RGBf(0.22, 0.596, 0.149),        # Julia green: iodine, passes
    blue = Mke.RGBf(0.251, 0.388, 0.847),        # Julia blue: water, references
    purple = Mke.RGBf(0.584, 0.345, 0.698),      # Julia purple
);

# ╔═╡ a1347d46-0163-4ca4-9247-4cd901bfade1
# Large, uncluttered type on warm paper for every figure
THEME = Mke.Theme(
    fontsize = 18,
    backgroundcolor = COLORS.paper,
    textcolor = COLORS.ink,
    figure_padding = (12, 24, 12, 12),
    palette = (color = [
        COLORS.violet,
        COLORS.cyan,
        COLORS.beam,
        COLORS.red,
        COLORS.green,
        COLORS.blue,
        COLORS.purple,
    ],),
    Axis = (
        backgroundcolor = COLORS.paper,
        titlesize = 22,
        titlealign = :left,
        titlegap = 12,
        subtitlesize = 17,
        subtitlecolor = COLORS.muted,
        xlabelsize = 19,
        ylabelsize = 19,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xgridcolor = (COLORS.ink, 0.07),
        ygridcolor = (COLORS.ink, 0.07),
        leftspinecolor = COLORS.rule,
        bottomspinecolor = COLORS.rule,
        xtickcolor = COLORS.rule,
        ytickcolor = COLORS.rule,
        topspinevisible = false,
        rightspinevisible = false,
        xlabelpadding = 8,
        ylabelpadding = 12,
    ),
    Legend = (
        labelsize = 17,
        framevisible = false,
        backgroundcolor = COLORS.paper,
        patchsize = (30, 14),
        rowgap = 8,
    ),
    Colorbar = (labelsize = 18, ticklabelsize = 15),
    Lines = (linewidth = 2.5,),
);

# ╔═╡ 6b51fcdc-9a60-4c4f-9031-39214c0cd354
begin
    """A blank drawing area at `position`: equal scales, no axes, limits `(x0, x1, y0, y1)`."""
    function canvas(position, limits; background = COLORS.paper)
        ax = Mke.Axis(position; aspect = Mke.DataAspect(), backgroundcolor = background)
        Mke.hidedecorations!(ax)
        Mke.hidespines!(ax)
        Mke.limits!(ax, limits...)
        return ax
    end

    """The corners of a rectangle centred at `c`, `w` × `h`, its corners rounded by `r`."""
    function rounded_rect(c, w, h, r)
        x0, y0, x1, y1 = c[1] - w / 2, c[2] - h / 2, c[1] + w / 2, c[2] + h / 2
        r = min(r, w / 2, h / 2)
        arc(cx, cy, θ0) = [
            Mke.Point2f(cx + r * cos(θ), cy + r * sin(θ))
            for θ in range(θ0, θ0 + π / 2; length = 8)
        ]
        return vcat(
            arc(x1 - r, y0 + r, -π / 2),
            arc(x1 - r, y1 - r, 0.0),
            arc(x0 + r, y1 - r, π / 2),
            arc(x0 + r, y0 + r, π),
        )
    end

    """A straight arrow from `a` to `b`; `head` is the arrowhead's size in pixels."""
    function arrow!(
        ax,
        a,
        b;
        color = COLORS.muted,
        linewidth = 2.5,
        head = 15,
        linestyle = :solid,
    )
        a, b = Mke.Point2f(a), Mke.Point2f(b)
        Mke.lines!(ax, [a, b]; color, linewidth, linestyle)
        t = b - a
        Mke.scatter!(
            ax,
            [b];
            marker = :utriangle,
            markersize = head,
            rotation = atan(t[2], t[1]) - π / 2,
            color,
            markerspace = :pixel,
        )
        return nothing
    end

    """
    A box in a flow chart: centred at `c`, `w` × `h`, a bold `title` and an optional smaller
    `subtitle`. `accent` colours its outline (and a strip down its left edge).
    """
    function node!(
        ax,
        c,
        title;
        subtitle = "",
        w = 3.0,
        h = 1.0,
        accent = COLORS.violet,
        fill = COLORS.card,
        textcolor = COLORS.ink,
        subcolor = COLORS.muted,
        titlesize = 19,
        subtitlesize = 15,
        code = false,
    )
        Mke.poly!(
            ax,
            rounded_rect(c, w, h, 0.14);
            color = fill,
            strokecolor = accent,
            strokewidth = 2,
        )
        strip = rounded_rect((c[1] - w / 2 + 0.06, c[2]), 0.12, h - 0.1, 0.05)
        Mke.poly!(ax, strip; color = accent)
        font = code ? "DejaVu Sans Mono Bold" : :bold
        dy = isempty(subtitle) ? 0.0 : 0.17 * h
        Mke.text!(
            ax,
            c[1] + 0.05,
            c[2] + dy;
            text = title,
            align = (:center, :center),
            fontsize = titlesize,
            font,
            color = textcolor,
        )
        isempty(subtitle) ||
            Mke.text!(
                ax,
                c[1] + 0.05,
                c[2] - 0.22 * h;
                text = subtitle,
                align = (:center, :center),
                fontsize = subtitlesize,
                color = subcolor,
            )
        return nothing
    end

    """
    A flow chart: `nodes` maps a key to `(x, y, title, subtitle, accent)` (subtitle and accent
    optional), `edges` lists `from => to` pairs, drawn as arrows between the boxes' edges, and
    `labels` (optional) puts a note on an edge. Returns the canvas.
    """
    function flow_chart!(position, nodes, edges; limits, w = 3.0, h = 1.0, labels = Dict())
        ax = canvas(position, limits)
        box(key) = nodes[key]
        function port(from, to)
            a, b = box(from), box(to)
            if abs(b[2] - a[2]) < 1.0e-6
                # side by side
                s = sign(b[1] - a[1])
                return (a[1] + s * w / 2, a[2]), (b[1] - s * w / 2, b[2])
            elseif abs(b[1] - a[1]) < 1.0e-6
                # one above the other
                s = sign(b[2] - a[2])
                return (a[1], a[2] + s * h / 2), (b[1], b[2] - s * h / 2)
            else
                # leave sideways, arrive on the facing side
                s = sign(b[1] - a[1])
                return (a[1] + s * w / 2, a[2]), (b[1] - s * w / 2, b[2])
            end
        end
        for (from, to) in edges
            p, q = port(from, to)
            if abs(p[1] - q[1]) < 1.0e-6 || abs(p[2] - q[2]) < 1.0e-6
                # straight across or straight down
                arrow!(ax, p, q)
            else
                # a right-angled connector: across to the gap's middle, along it, then in
                mx = (p[1] + q[1]) / 2
                Mke.lines!(
                    ax,
                    Mke.Point2f[p, (mx, p[2]), (mx, q[2])];
                    color = COLORS.muted,
                    linewidth = 2.5,
                )
                arrow!(ax, (mx, q[2]), q)
            end
            haskey(labels, from => to) &&
                Mke.text!(
                    ax,
                    (p[1] + q[1]) / 2,
                    q[2] + 0.12;
                    text = labels[from => to],
                    align = (:center, :bottom),
                    fontsize = 15,
                    color = COLORS.muted,
                )
        end
        for (_, n) in nodes
            node!(
                ax,
                (n[1], n[2]),
                n[3];
                subtitle = length(n) >= 4 ? n[4] : "",
                accent = length(n) >= 5 ? n[5] : COLORS.violet,
                w,
                h,
            )
        end
        return ax
    end

    """
    A CT gantry seen along z, centred at the origin: the source(s) at radius `sid` (one per entry
    of `angles`, degrees), the fan, the detector arc at `sdd` from each source, and a round object
    of radius `r_object`. Draws on `ax` in the colours of `dark` (the locator band) or paper.
    """
    function gantry!(
        ax;
        sid = 5.4,
        sdd = 9.5,
        angles = (90.0,),
        fan_deg = 50.0,
        r_object = 2.0,
        detector_colors = (COLORS.cyan,),
        labels = ("Tube",),
        dark = true,
        object = true,
    )
        ink = dark ? COLORS.night_ink : COLORS.ink
        ring = dark ? COLORS.night_card : COLORS.card
        Mke.poly!(ax, Mke.Circle(Mke.Point2f(0, 0), sid + 0.9); color = ring)
        Mke.poly!(
            ax,
            Mke.Circle(Mke.Point2f(0, 0), sid - 1.6);
            color = dark ? COLORS.night : COLORS.paper,
        )
        for (k, θ) in enumerate(angles)
            θ = deg2rad(θ)
            s = Mke.Point2f(sid * cos(θ), sid * sin(θ))
            half = deg2rad(fan_deg / 2)
            # the fan, from the focal spot to the detector arc (a circle about the source)
            edge = Mke.Point2f[
                s + sdd * Mke.Point2f(cos(θ + π + φ), sin(θ + π + φ))
                for φ in range(-half, half; length = 40)
            ]
            Mke.poly!(
                ax,
                Mke.Point2f.(vcat([s], edge));
                color = (COLORS.beam, 0.22),
                strokewidth = 0,
            )
            Mke.lines!(ax, edge; color = detector_colors[mod1(k, end)], linewidth = 9)
            Mke.scatter!(ax, [s]; marker = :rect, markersize = 26, color = ink)
            Mke.scatter!(ax, [s]; marker = :circle, markersize = 9, color = COLORS.beam)
            Mke.text!(
                ax,
                s * 1.18;
                text = labels[mod1(k, end)],
                align = (:center, :center),
                fontsize = 17,
                font = :bold,
                color = ink,
            )
        end
        object &&
            Mke.poly!(
                ax,
                Mke.Circle(Mke.Point2f(0, 0), r_object);
                color = (COLORS.blue, 0.55),
                strokecolor = ink,
                strokewidth = 1.5,
            )
        return ax
    end

    """
    The dark locator band that opens a notebook: `n` panels side by side, each a canvas with
    `limits`, its `titles[i]` and `subtitles[i]` underneath. Returns the figure and the canvases;
    draw into them, then show the figure.
    """
    function locator_band(titles, subtitles; limits, size = (1500, 560))
        fig = Mke.Figure(;
            size,
            backgroundcolor = COLORS.night,
            figure_padding = (10, 10, 18, 10),
        )
        axes = [
            canvas(fig[1, i], limits[i]; background = COLORS.night)
            for i in eachindex(titles)
        ]
        for i in eachindex(titles)
            Mke.Label(
                fig[2, i],
                titles[i];
                fontsize = 25,
                font = :bold,
                color = COLORS.night_ink,
                tellwidth = false,
            )
            Mke.Label(
                fig[3, i],
                subtitles[i];
                fontsize = 17,
                color = COLORS.night_muted,
                tellwidth = false,
            )
        end
        Mke.rowgap!(fig.layout, 1, 8)
        Mke.rowgap!(fig.layout, 2, 2)
        return fig, axes
    end
end;

# ╔═╡ 127012b1-df1a-4c12-b7e5-6eddcf183923
# one colour per heart rod, in the order of `ROD_LABELS`: water, lipid, collagen, iodine
ROD_COLORS = (COLORS.blue, COLORS.beam, COLORS.purple, COLORS.green);

# ╔═╡ 018a50f5-996d-447b-974e-ab87a2e181e8
"""
A drawing of the QRM-Thorax mid-slice from `QRM_GEOMETRY` (mm, sternum up): body, lungs,
mediastinum and heart, the bones, the two lung rods and the four heart rods in `ROD_COLORS`.
`rois` outlines extra circles `(center, r, color)` (mm); `names` writes each heart rod's
material beside it; `dark` draws in the locator band's colours.
"""
function qrm_sketch!(ax; g = QRM_GEOMETRY, dark = false, rois = (), names = false)
    tissue = dark ? Mke.RGBf(0.42, 0.37, 0.32) : Mke.RGBf(0.88, 0.8, 0.7)
    lung = dark ? COLORS.night : COLORS.paper
    bone = dark ? COLORS.night_ink : Mke.RGBf(0.99, 0.97, 0.93)
    edge = dark ? COLORS.night_muted : COLORS.muted
    ink = dark ? COLORS.night_ink : COLORS.ink
    P = Mke.Point2f

    # a superellipse cut flat at its y range, as a polygon
    function superellipse(s)
        pts = P[]
        for t in range(0, 2π; length = 400)
            c, si = cos(t), sin(t)
            x = s.a * sign(c) * abs(c)^(2 / s.n)
            y = s.yc + s.b * sign(si) * abs(si)^(2 / s.n)
            push!(pts, P(x, clamp(y, s.y[1], s.y[2])))
        end
        return pts
    end
    disc(c, r) = Mke.Circle(P(c...), Float32(r))
    function ellipse(c, axes, θ)
        s, co = sincosd(θ)
        return [
            P(
                c[1] + co * axes[1] * cos(t) - s * axes[2] * sin(t),
                c[2] + s * axes[1] * cos(t) + co * axes[2] * sin(t),
            )
            for t in range(0, 2π; length = 60)
        ]
    end
    bone_style = (color = bone, strokecolor = edge, strokewidth = 0.8)

    Mke.poly!(
        ax,
        superellipse(g.body);
        color = tissue,
        strokecolor = edge,
        strokewidth = 1.5,
    )
    Mke.poly!(ax, superellipse(g.lungs); color = lung)
    # the mediastinum above the heart, and the heart, are soft tissue
    m = g.mediastinum
    Mke.poly!(
        ax,
        Mke.Rect2f(
            -m.half_width,
            g.heart.center[2],
            2 * m.half_width,
            g.lungs.y[2] - g.heart.center[2] + 0.5,
        );
        color = tissue,
    )
    Mke.poly!(
        ax,
        disc(g.heart.center, g.heart.r);
        color = tissue,
        strokecolor = edge,
        strokewidth = 1,
    )
    for (rx, ry, θ) in g.ribs
        Mke.poly!(ax, ellipse((rx, ry), g.rib_axes, θ); bone_style...)
    end
    Mke.poly!(ax, ellipse(g.sternum.center, g.sternum.axes, 0.0); bone_style...)
    for t in (g.laminae, g.spinous)
        Mke.poly!(ax, P[t.corners...]; bone_style...)
    end
    Mke.poly!(ax, disc(g.vertebra.center, g.vertebra.r); bone_style...)
    Mke.poly!(ax, disc(g.canal.center, g.canal.r); color = tissue)
    for rod in g.lung_rods
        Mke.poly!(
            ax,
            disc(rod.center, rod.r);
            color = rod.label == 6 ? (COLORS.blue, 0.6) : lung,
            strokecolor = edge,
            strokewidth = 1,
        )
    end
    for (k, rod) in enumerate(g.heart_rods)
        Mke.poly!(
            ax,
            disc(rod.center, rod.r);
            color = ROD_COLORS[k],
            strokecolor = edge,
            strokewidth = 1,
        )
    end
    if names
        offsets = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))
        aligns = ((:center, :bottom), (:left, :center), (:center, :top), (:right, :center))
        for (k, rod) in enumerate(g.heart_rods)
            o = offsets[k]
            Mke.text!(
                ax,
                rod.center[1] + o[1] * (rod.r + 2.5),
                rod.center[2] + o[2] * (rod.r + 2.5);
                text = ROD_NAMES[k],
                align = aligns[k],
                fontsize = 15,
                font = :bold,
                color = ink,
            )
        end
    end
    for (c, r, color) in rois
        Mke.lines!(ax, disc(c, r); color, linewidth = 3.5)
    end
    return ax
end

# ╔═╡ 07020001-0000-4000-8000-000000000001
md"""
## 1. The Phantom: QRM-Thorax With Four Pure-Material Rods

The QRM-Thorax is built analytically from 2-D shapes: a flat-fronted superellipse body, the two
lungs, the cardiac insert and the mediastinum above it, a water rod and an air rod in the lungs,
six ribs and the sternum, and the vertebra with its canal, laminae and spinous process. Every
dimension was measured on the labelled 1600 × 1100 (0.2 mm) mid-slice of the scanned phantom, in
CT display orientation (spine at the bottom). The slice is tiled in z to **1600 × 1100 × 20 at
0.2 mm**, finer than the demagnified dexel pitch, so the forward projector samples a
high-resolution object.

| Label | Material | Label | Material |
|:--|:--|:--|:--|
| 1 | air | 7 | air rod |
| 2 | lung | 8 | (unused) |
| 3 | soft tissue | **9** | rod → water |
| 4 | cortical bone | **10** | rod → lipid |
| 5 | bone marrow | **11** | rod → collagen |
| 6 | water rod (lung) | **12** | rod → 5 mg/mL iodine |

As in the reference slice, the cardiac insert is soft tissue (label 3, like the body) with the
four rods bored into it, so label 8 is unused. `qrm_thorax_slice` draws labels 1–7; the rods are
bored below, at the centres `QRM_GEOMETRY.heart_rods` records. Every material is a prebuilt
`XA.Materials` entry.
"""

# ╔═╡ 07020001-0000-4000-8000-000000000010
begin
    # The QRM-Thorax mid-slice as 2-D shapes, measured on the labelled 1600 × 1100 (0.2 mm)
    # reference slice. Millimetres from the slice centre, +x to the image right and +y up:
    # the CT display orientation, sternum at the top and spine at the bottom.
    const QRM_GEOMETRY = (
        # |x/a|^n + |(y - yc)/b|^n ≤ 1, cut flat at the front and back (y range)
        body = (a = 150.2, b = 106.75, n = 2.3, yc = 0.0, y = (-100.2, 101.0)),
        lungs = (a = 127.6, b = 92.2, n = 2.1, yc = 0.4, y = (-76.8, 76.8)),
        # soft tissue between the lungs above the heart; its corners at the lung apex are rounded
        mediastinum = (half_width = 16.8, corner_r = 2.0),
        # the cardiac insert, soft tissue (label 3)
        heart = (center = (0.0, 18.4), r = 55.2),
        # the four pure-material rods in the heart, as the reference slice has them (labels 9–12)
        heart_rods = (
            (center = (0.0, 45.8), r = 7.5, label = 9),    # north: water
            (center = (24.9, 20.7), r = 7.5, label = 10),  # east: lipid
            (center = (0.0, -4.3), r = 7.5, label = 11),   # south: collagen
            (center = (-25.1, 20.7), r = 7.5, label = 12), # west: 5 mg/mL iodine
        ),
        lung_rods = (
            (center = (-91.4, 0.0), r = 7.5, label = 6), # water rod
            (center = (90.2, 0.0), r = 7.5, label = 7),  # air rod
        ),
        # the spine; `cortex` is the thickness of the cortical-bone shell, mm
        vertebra = (center = (0.0, -53.3), r = 17.4, cortex = 0.6),
        canal = (center = (0.0, -69.6), r = 8.0, cortex = 0.6), # the spinal canal
        laminae = (corners = ((-35.4, -77.2), (35.4, -77.2), (0.0, -62.3)), cortex = 1.0),
        spinous = (corners = ((-2.6, -76.8), (2.6, -76.8), (0.0, -91.6)), cortex = 0.6),
        # (x, y, long-axis angle in °)
        ribs = (
            (-112.4, 64.9, 40.0),
            (108.9, 67.9, -40.0),
            (-138.1, 1.5, 90.0),
            (139.0, 6.1, 90.0),
            (-109.2, -65.9, -40.0),
            (109.2, -67.5, 40.0),
        ),
        rib_axes = (8.2, 3.0), # semi-axes, mm
        sternum = (center = (-2.7, 89.4), axes = (7.6, 3.2)),
        # rib and sternum marrow = their cortical ellipse scaled by this
        marrow_scale = 0.85,
    )

    """
        qrm_thorax_slice(g = QRM_GEOMETRY; n = (1600, 1100), voxel_mm = 0.2, inserts = ())

    Label image of the QRM-Thorax mid-slice (1 air, 2 lung, 3 soft tissue, including the
    cardiac insert, 4 cortical bone, 5 marrow, 6 water rod, 7 air rod). `inserts` are extra
    discs `(center, r, label)` in mm, painted last: the heart rods of `g.heart_rods`.
    """
    function qrm_thorax_slice(
        g = QRM_GEOMETRY;
        n = (1600, 1100),
        voxel_mm = 0.2,
        inserts = (),
    )
        superellipse(x, y, s) = s.y[1] ≤ y ≤ s.y[2] &&
            abs(x / s.a)^s.n + abs((y - s.yc) / s.b)^s.n ≤ 1
        disc(x, y, c, r) = (x - c[1])^2 + (y - c[2])^2 ≤ r^2
        function ellipse(x, y, c, axes, θ)
            s, co = sincosd(θ)
            u, v = co * (x - c[1]) + s * (y - c[2]), -s * (x - c[1]) + co * (y - c[2])
            return (u / axes[1])^2 + (v / axes[2])^2 ≤ 1
        end
        # inside all three edges, each moved `inset` mm toward the interior
        function triangle(x, y, t, inset = 0.0)
            for k in 1:3
                p, q, r = t[k], t[mod1(k + 1, 3)], t[mod1(k + 2, 3)]
                ex, ey = q[1] - p[1], q[2] - p[2]
                side(px, py) = (ex * (py - p[2]) - ey * (px - p[1])) / hypot(ex, ey)
                side(x, y) * sign(side(r...)) ≥ inset || return false
            end
            return true
        end
        function lung(x, y)
            superellipse(x, y, g.lungs) || return false
            disc(x, y, g.heart.center, g.heart.r) && return false
            m = g.mediastinum
            abs(x) < m.half_width && y > g.heart.center[2] && return false
            # round the apex corner where the lung top meets the mediastinum
            cx, cy = m.half_width + m.corner_r, g.lungs.y[2] - m.corner_r
            abs(x) < cx && y > cy && hypot(abs(x) - cx, y - cy) > m.corner_r && return false
            return true
        end

        bones = (
            ((c = (rx, ry), axes = g.rib_axes, θ = θ) for (rx, ry, θ) in g.ribs)...,
            (c = g.sternum.center, axes = g.sternum.axes, θ = 0.0),
        )
        img = ones(UInt8, n)
        for j in 1:n[2], i in 1:n[1]
            x, y = (i - (n[1] + 1) / 2) * voxel_mm, (j - (n[2] + 1) / 2) * voxel_mm
            superellipse(x, y, g.body) || continue
            label = lung(x, y) ? 2 : 3 # the heart is not lung, so it is soft tissue
            for rod in g.lung_rods
                disc(x, y, rod.center, rod.r) && (label = rod.label)
            end
            for b in bones
                ellipse(x, y, b.c, b.axes, b.θ) &&
                    (label = ellipse(x, y, b.c, b.axes .* g.marrow_scale, b.θ) ? 5 : 4)
            end
            for t in (g.laminae, g.spinous)
                triangle(x, y, t.corners) &&
                    (label = triangle(x, y, t.corners, t.cortex) ? 5 : 4)
            end
            v, cn = g.vertebra, g.canal
            disc(x, y, v.center, v.r) &&
                (label = disc(x, y, v.center, v.r - v.cortex) ? 5 : 4)
            disc(x, y, cn.center, cn.r) &&
                (label = disc(x, y, cn.center, cn.r - cn.cortex) ? 3 : 4)
            for (c, r, l) in inserts
                disc(x, y, c, r) && (label = l)
            end
            img[i, j] = label
        end
        return img
    end
end;

# ╔═╡ 07020001-0000-4000-8000-000000000013
const QRM_NZ = 20; # 20 × 0.2 mm = 4 mm: a short z-invariant phantom

# ╔═╡ 07020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02); # (x, y, z) cm: 0.2 mm isotropic

# ╔═╡ 07020001-0000-4000-8000-000000000025
md"""
**Bore four rods** into the heart where the reference slice has them
(`QRM_GEOMETRY.heart_rods`: 15 mm rods, 25 mm north, east, south and west of a point 20.7 mm
above the slice centre), as labels 9–12 (matching `materials_dict`):

| Direction | Label | Material |
|---|---|---|
| North | 9 | `basis_water` |
| East | 10 | `basis_lipid` |
| South | 11 | `basis_collagen` |
| West | 12 | `gammex_472_i5_0` |
"""

# ╔═╡ 07020001-0000-4000-8000-000000000026
# (center in mm, radius in mm, label) of each heart rod
HEART_RODS = [(rod.center, rod.r, rod.label) for rod in QRM_GEOMETRY.heart_rods];

# ╔═╡ 07020001-0000-4000-8000-000000000029
mask_3d = let
    slice = qrm_thorax_slice(; voxel_mm = QRM_VOXEL_SIZE_CM[1] * 10, inserts = HEART_RODS)
    repeat(slice; outer = (1, 1, QRM_NZ))
end;

# ╔═╡ 07020001-0000-4000-8000-000000000040
materials_dict = Dict{Int, BS.XA.Material}(
    # Anatomy
    1 => BS.XA.Materials.air,
    2 => BS.XA.Materials.lung,
    3 => BS.XA.Materials.muscle, # soft tissue: ICRU-44 muscle
    4 => BS.XA.Materials.corticalbone,
    5 => BS.XA.Materials.marrow_red,
    6 => BS.XA.Materials.water, # water rod in the lung
    7 => BS.XA.Materials.air,   # air rod
    # unused: the heart insert is soft tissue (label 3)
    8 => BS.XA.Materials.muscle,
    # Pure-material rod inserts
    9 => BS.XA.Materials.basis_water,
    10 => BS.XA.Materials.basis_lipid,
    11 => BS.XA.Materials.basis_collagen,
    12 => BS.XA.Materials.gammex_472_i5_0, # Gammex 472 5 mg/mL iodine insert
);

# ╔═╡ 07020001-0000-4000-8000-000000000050
phantom_cpu = BS.create_phantom_from_mask(
    Array{Int, 3}(mask_3d),
    materials_dict,
    QRM_VOXEL_SIZE_CM,
);

# ╔═╡ 07020001-0000-4000-8000-000000000051
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 07020001-0000-4000-8000-000000000060
QRM_LABEL_NAMES = Dict{Int, String}(
    1 => "1 Air",
    2 => "2 Lung",
    3 => "3 Soft Tissue",
    4 => "4 Cortical Bone",
    5 => "5 Marrow",
    6 => "6 Water Rod",
    7 => "7 Air Rod",
    8 => "8 (Unused)",
    9 => "9 Water",
    10 => "10 Lipid",
    11 => "11 Collagen",
    12 => "12 Iodine 5 mg/mL",
);

# ╔═╡ 07020001-0000-4000-8000-000000000061
let
    Mke.set_theme!(THEME)
    slice = mask_3d[:, :, QRM_NZ ÷ 2 + 1]
    nx, ny = size(slice)
    px_mm = QRM_VOXEL_SIZE_CM[1] * 10
    xs = ((1:nx) .- (nx + 1) / 2) .* px_mm
    ys = ((1:ny) .- (ny + 1) / 2) .* px_mm
    fig = Mke.Figure(size = (1500, 640))

    # the z tiling, drawn: one slice repeated QRM_NZ times
    ax = canvas(fig[1, 1], (-0.6, 9.4, -1.6, 7.4))
    ax.title = "One Slice, Tiled in z"
    for s in 7:-1:0
        Mke.poly!(
            ax,
            Mke.Point2f[
                (0.4 * s, 0.45 * s),
                (4.6 + 0.4 * s, 0.45 * s),
                (5.6 + 0.4 * s, 2.2 + 0.45 * s),
                (1.0 + 0.4 * s, 2.2 + 0.45 * s),
            ];
            color = s == 0 ? Mke.RGBf(0.88, 0.8, 0.7) : COLORS.card,
            strokecolor = COLORS.muted,
            strokewidth = 1,
        )
    end
    Mke.text!(
        ax,
        3.2,
        -0.3;
        text = "$(QRM_NZ) Slices × $(px_mm) mm = $(QRM_NZ * px_mm) mm",
        align = (:center, :top),
        fontsize = 17,
        color = COLORS.ink,
    )
    arrow!(ax, (6.0, 0.0), (8.3, 2.6); color = COLORS.muted)
    Mke.text!(ax, 8.45, 2.7; text = "z", fontsize = 20, font = :bold, color = COLORS.muted)

    # the data: the labelled mid-slice
    ax = Mke.Axis(
        fig[1, 2];
        title = "QRM-Thorax Mid-Slice, Labelled",
        subtitle = "$(nx) × $(ny) Voxels of $(px_mm) mm · $(length(unique(slice))) Labels",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
    )
    hm = Mke.heatmap!(
        ax,
        xs,
        ys,
        Float32.(slice);
        colormap = Mke.cgrad(:Paired_12; categorical = true),
        colorrange = (0.5, 12.5),
    )
    Mke.Colorbar(
        fig[1, 3],
        hm;
        ticks = (1:12, [QRM_LABEL_NAMES[k] for k in 1:12]),
        width = 16,
        ticklabelsize = 15,
    )
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.27))
    fig
end

# ╔═╡ 07020001-0000-4000-8000-000000000062
let
    slice = mask_3d[:, :, QRM_NZ ÷ 2 + 1]
    px_mm = QRM_VOXEL_SIZE_CM[1] * 10
    body = findall(!=(UInt8(1)), slice)
    i_lo, i_hi = extrema(c -> c[1], body)
    j_lo, j_hi = extrema(c -> c[2], body)
    w, h = i_hi - i_lo + 1, j_hi - j_lo + 1
    counts = [count(==(UInt8(k)), slice) for k in 1:12]
    rows = join(
        [
            "| $(QRM_LABEL_NAMES[k]) | $(counts[k]) | $(round(counts[k] * px_mm^2 / 100, digits = 2)) |"
            for k in 1:12
            if counts[k] > 0
        ],
        "\n",
    )
    Markdown.parse(
        """
    The body measures $(round(w * px_mm / 10, digits = 1)) × $(round(h * px_mm / 10, digits = 1)) cm
    (the QRM-Thorax-small envelope without the fat ring, ≈ 30 × 20 cm).

    | label | voxels in the mid-slice | area (cm²) |
    |-------|------:|------:|
    $(rows)
    """,
    )
end

# ╔═╡ 07030001-0000-4000-8000-000000000001
md"""
## 2. The Scanner and the Dual-kVp Protocol

### `EICTScanner`: GE Revolution Apex Elite

The documented Apex Elite, `REVOLUTION_APEX`, without electronic noise, as in notebook 03: 834
columns × 0.6 mm on an arc with the quarter-detector offset, 256 rows × 0.625 mm, the Gemstone
scintillator (`:lumex`) and the GE large-body bowtie.
"""

# ╔═╡ 07030001-0000-4000-8000-000000000010
scanner = BS.EICTScanner(BS.REVOLUTION_APEX; electronic_noise = 0.0)

# ╔═╡ 524abd9e-0288-5a3b-b911-28cb8ee1ec73
# Every keyword of `scanner`: the documented constant `REVOLUTION_APEX`, with any field this notebook changes marked
let s = scanner, constant = BS.REVOLUTION_APEX
    shown(x) = replace(repr(x; context = :compact => true), "|" => "\\|")
    function row(f)
        value, standard = getproperty(s, f), getproperty(constant, f)
        note = isequal(value, standard) ?
            "" :
            "changed here (the constant has `$(shown(standard))`)"
        return "| `$(f)` | `$(shown(value))` | $(note) |"
    end

    rows = [row(f) for f in propertynames(s)]

    Markdown.parse(
        """
        **`scanner` is `$(nameof(typeof(s)))`, from the documented constant `REVOLUTION_APEX`.** Its keywords
        and their values (units and sources: the constant's docstring, `?BS.REVOLUTION_APEX`, and the Scanners
        page). A variant is the same struct with keywords replaced,
        `$(nameof(typeof(s)))(BS.REVOLUTION_APEX; field = value)`.

        | Keyword | Value | |
        |---|---|---|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 07030002-0000-4000-8000-000000000001
md"""
### Rapid kVp Switching

The tube alternates between 80 and 140 kVp within every view period, on one tube and one
detector, so both channels see the same rays. Each energy is read for its **duty cycle** of the
view period, 0.65 at 80 kVp and 0.35 at 140 kVp: its tube current is the instantaneous current
(407 and 405 mA on the physical scan) times its duty cycle, and each of its views integrates over
that fraction of the arc the gantry turns per view (`view_arc`). The acquisition is two
co-registered acquisitions on one geometry, each with 984 views over a 0.5 s rotation, 5 mm
collimation and 4.5 mm of added aluminium.

The simulator's tube output per mA is generic, not GE's, so both currents are scaled by one
factor, `DOSE_SCALE`, that makes the simulated CTDIvol (32 cm body phantom, summed over the
exposures) the 10.07 mGy of the physical GE scan basis-spectral-denoising modelled.
"""

# ╔═╡ 07030002-0000-4000-8000-000000000010
# The two exposures of one rotation; each has its own noise seed, so the two channels carry
# independent quantum noise.
exposures = (
    low = (
        label = "80 kVp",
        kVp = 80,
        mA = 407 * 0.65,
        view_arc = 0.65,
        seed = 1234,
        filters = [("Al", 4.5)],
    ),
    high = (
        label = "140 kVp",
        kVp = 140,
        mA = 405 * 0.35,
        view_arc = 0.35,
        seed = 4321,
        filters = [("Al", 4.5)],
    ),
);

# ╔═╡ 07030002-0000-4000-8000-000000000011
begin
    "The `CTProtocol` of exposure `e`, its tube current scaled by `mA_scale`."
    exposure_protocol(e; mA_scale = 1.0) = BS.CTProtocol(
        kVp = e.kVp,
        mA = e.mA * mA_scale,
        views = 984,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        additional_filters = e.filters,
    )
    "The CTDIvol (mGy, 32 cm body phantom) of `protocol` on `scanner`."
    ctdi_vol(protocol) = BS.compute_dose(
        BS.dose_source(scanner, protocol),
        protocol,
    ).ctdi_vol_mGy
end;

# ╔═╡ 07030002-0000-4000-8000-000000000012
# One factor on both tube currents, so the simulated CTDIvol is the physical scan's 10.07 mGy.
DOSE_SCALE = 10.07 / sum(e -> ctdi_vol(exposure_protocol(e)), exposures)

# ╔═╡ 07030002-0000-4000-8000-000000000020
protocols = map(e -> exposure_protocol(e; mA_scale = DOSE_SCALE), exposures);

# ╔═╡ 07030002-0000-4000-8000-000000000021
let
    function row(k)
        e, p = exposures[k], protocols[k]
        return "| $(e.label) | $(round(e.mA / e.view_arc, digits = 1)) | $(e.view_arc) | " *
            "$(round(p.mA, digits = 1)) | $(round(ctdi_vol(p), digits = 2)) |"
    end
    rows = [row(k) for k in keys(exposures)]
    Markdown.parse(
        """
    `DOSE_SCALE` = $(round(DOSE_SCALE, digits = 4)). The simulated exposures:

    | exposure | physical mA | duty cycle, `view_arc` | simulated mA | CTDIvol (mGy) |
    |---|---:|---:|---:|---:|
    $(join(rows, "\n"))
    """,
    )
end

# ╔═╡ 07030003-0000-4000-8000-000000000001
md"""
## 3. Physics, Grid and Simulation

The physics switches of the basis-spectral-denoising GE model: every effect of the
energy-integrating path on except optical crosstalk, and the detector efficiency from the Monte
Carlo table of the Lumex scintillator (the scanner's `detector_efficiency_mode = :mc_lut`, its
default).

**View integration.** A detector integrates while the gantry turns, so each view records the
transmitted intensity averaged over the arc it sweeps, `r · view_arc · Δθ` at radius `r`: a blur
of the object, largest far from the isocentre, that the noise, counted once per view, does not
share. `view_samples = 5` samples that arc with five sub-views; without it the simulated signal
is sharper, relative to its noise, than a physical scanner's.

**Draws.** Each exposure is simulated three times, differing only in the noise: the noise-free
expectation (draw 0, `use_noise = false`), the measured scan (draw 1) and a calibration scan
(draw 9) of its own seeds. The reconstruction grid is 512 × 512 × 3 at 0.625 mm, independent of
the 0.2 mm phantom grid; both share the isocentre.
"""

# ╔═╡ 07030003-0000-4000-8000-000000000010
begin
    VIEW_SAMPLES = 5
    ge_opts = BS.SimOptions(projector = :dd_fast, view_samples = VIEW_SAMPLES)
    "The same physics switches with one field replaced."
    with_option(opts, field, value) = BS.SimOptions(;
        (f => getfield(opts, f) for f in fieldnames(BS.SimOptions) if f !== field)...,
        (field => value,)...,
    )
end;

# ╔═╡ 07030003-0000-4000-8000-000000000011
begin
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each exposure the seed
    # `seed + 1000 (r − 1)`, so the exposures stay independent within a draw and the draws of
    # each other.
    DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    "The options of draw `r` of exposure `e`: its seed, its noise, its arc."
    draw_opts(e, r) = with_option(
        with_option(with_option(ge_opts, :seed, realization_seed(e, r)), :use_noise, r > 0),
        :view_arc,
        e.view_arc,
    )
end;

# ╔═╡ 07030003-0000-4000-8000-000000000020
recon_opts = BS.ReconOptions(matrix_size = (512, 512, 3), fov_cm = 32.0, z_cm = 0.1875);

# ╔═╡ 07030004-0000-4000-8000-000000000001
md"""
### Forward Project: `simulate!`

Each exposure is projected once. The noise-free draw runs first with `keep_projection = true`,
which returns everything before the noise; the measured and calibration draws pass it back
(`projection`) and only draw their noise, bit-identically to simulating each afresh. Each
acquisition keeps its corrected log sinograms and its one-channel spectral basis,
`spectral_basis(ws, protocol)`: each ray's detected spectrum (source × filtration × bowtie × the
detector's signal per photon, `m₁(E) = E·efficiency(E)`), the model the forward projector
applied, scaled to the ray's noise-equivalent quanta in air, `I₀·ff·S₁²/S₂` (an energy
integrator's signal is not Poisson: its variance is `S₂/S₁` times its mean, which rises as the
beam hardens).
"""

# ╔═╡ ecbcace5-f22f-4b66-8847-b7aad4b23554
"""
    simulate_exposure(e, protocol) -> NamedTuple

The noise-free, measured and calibration draws of exposure `e` (one projection, three noise
draws), its geometry and its one-channel spectral basis.
"""
function simulate_exposure(e, protocol)
    @info "Simulating dual-kVp channel: $(e.label)"
    channels = Dict{Int, Array{Float32, 3}}()
    projection = nothing
    model = nothing
    elapsed_s = @elapsed for r in (DRAWS.reference, DRAWS.measured, DRAWS.calibration)
        opts = draw_opts(e, r)
        ws = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom)
        try
            result = BS.simulate!(
                ws,
                phantom,
                protocol,
                opts;
                report_dose = false,
                keep_projection = projection === nothing,
                projection,
            )
            projection === nothing && (projection = result.projection)
            channels[r] = Float32.(Array(ws.sinogram))
            if model === nothing
                # the acquisition's one-channel basis: the model the projector applied
                model = (geom = ws.geom, basis = BS.spectral_basis(ws, protocol))
            end
        finally
            BS.release_backend!(ws)
        end
    end
    return (; label = e.label, channels, elapsed_s, model...)
end

# ╔═╡ 07030004-0000-4000-8000-000000000010
dual_scans = [simulate_exposure(exposures[k], protocols[k]) for k in keys(exposures)];

# ╔═╡ 07030004-0000-4000-8000-000000000011
Markdown.parse(
    """
The three draws of the two exposures took
$(join(["$(round(s.elapsed_s, digits = 1)) s for $(s.label)" for s in dual_scans], " and ")),
one projection each. Each sinogram is $(join(size(first(dual_scans).channels[DRAWS.measured]), " × "))
(columns × rows × views).
""",
)

# ╔═╡ 07030004-0000-4000-8000-000000000025
# The phantom labels resampled onto the reconstruction grid (nearest neighbour); every ROI
# below is built from these labels in recon-voxel coordinates.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu,
    first(dual_scans).geom,
    recon_opts.matrix_size;
    method = :nearest,
);

# ╔═╡ 07030004-0000-4000-8000-000000000030
let
    Mke.set_theme!(THEME)
    sinos = [s.channels[DRAWS.measured] for s in dual_scans]
    mid_r = size(first(sinos), 2) ÷ 2 + 1
    slices = [permutedims(s[:, mid_r, :], (2, 1)) for s in sinos]
    sino_window = Tuple(Float64.(quantile(vcat(vec.(slices)...), (0.01, 0.99))))
    kvp_colors = (COLORS.violet, COLORS.cyan)
    fig = Mke.Figure(size = (1500, 1080))

    # the tube switches within every view period
    top = fig[1, 1:3] = Mke.GridLayout()
    ax = canvas(top[1, 1], (-7.2, 7.2, -7.2, 7.2))
    ax.title = "One Tube, One Detector"
    gantry!(ax; labels = ("80 ⇄ 140 kVp",), object = false, dark = false)
    Mke.poly!(
        ax,
        Mke.Circle(Mke.Point2f(0, 0), 2.0f0);
        color = (COLORS.blue, 0.3),
        strokecolor = COLORS.ink,
        strokewidth = 1.5,
    )
    ax = Mke.Axis(
        top[1, 2];
        title = "The Tube Voltage Within Each View",
        subtitle = "80 kVp for 65 % of the View Period, 140 kVp for 35 %",
        xlabel = "Time (View Periods)",
        ylabel = "Tube Voltage (kVp)",
        yticks = [80, 140],
        xticks = 0:3,
    )
    for p in 0:2
        for (k, (t0, t1, kv)) in enumerate(((p, p + 0.65, 80), (p + 0.65, p + 1.0, 140)))
            Mke.band!(ax, [t0, t1], [0, 0], [kv, kv]; color = (kvp_colors[k], 0.35))
            Mke.lines!(ax, [t0, t1], [kv, kv]; color = kvp_colors[k], linewidth = 4)
        end
    end
    Mke.limits!(ax, 0, 3, 0, 165)
    Mke.colsize!(top, 1, Mke.Relative(0.32))

    # the data: the measured sinograms of the central detector row
    for (c, (s, slice)) in enumerate(zip(dual_scans, slices))
        ax = Mke.Axis(
            fig[2, c];
            title = "$(s.label) Sinogram",
            subtitle = "Central Detector Row, Measured Draw",
            titlecolor = kvp_colors[c],
            xlabel = "View",
            ylabel = "Dexel Column",
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = sino_window)
    end
    Mke.Colorbar(
        fig[2, 3];
        colormap = :viridis,
        colorrange = sino_window,
        label = "Log Line Integral",
        width = 16,
    )
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.42))
    fig
end

# ╔═╡ 07030000-0000-4000-8000-0000000000f1
md"""
## 4. The VMI Chain

### Spectral Basis From the Two Acquisitions

`spectral_basis_from_acquisitions` merges the two kVp energy grids onto their union and scales
each acquisition's per-ray spectrum by its own noise-equivalent quanta, so the likelihood sees
the absolute response ``\Phi_k(E)`` of every ray and channel: source, filtration, bowtie and
detector, the model that generated the data. It does not depend on the noise, so every draw
shares it.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000010
basis = BS.spectral_basis_from_acquisitions(acquisitions = [s.basis for s in dual_scans]);

# ╔═╡ 07030006-0000-4000-8000-000000000011
Markdown.parse(
    """
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid
for $(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays, spanning
$(round(Int, minimum(basis.E))) to $(round(Int, maximum(basis.E))) keV; the response sums
to the air counts to within $(round(basis.I0_relerr, sigdigits = 2)) (relative).
""",
)

# ╔═╡ 07030006-0000-4000-8000-000000000001
md"""
### The Chain's Settings

`vmi_pipeline` runs six stages, drawn below. **The denoiser** is generalized HYPR-LR in both
domains, `BS.SpectralHYPR()`: before the decomposition, a 3 × 3 (column × view) window on the
counts of each detector row, pooling views of one parity so the odd- and even-view halves stay
independent, with each channel's dispersion measured on the air rays; after the reconstruction,
the complement of the basis pair pooled within each slice over the window with the least
estimated risk. Neither pools across detector rows or slices. The decomposition is the
K-channel maximum-likelihood estimator with its published controls; T-LBF (a photon-counting
filter) is off; the view-direction antialias is on; ACNR is on (`vmi_pipeline`'s default).

**A pair of FDK windows.** A VMI is ``A(E)\,M + B(E)\,I_\perp``, with ``M`` the minimum-noise VMI
(the composite) and ``I_\perp`` its complement, whose noise is uncorrelated with ``M``'s.
`BS.PairFilter` gives each its own window; one pair serves every energy. A window per basis image
instead would stop the anti-correlated noise of the water and iodine images cancelling wherever
the two windows differ. The windows are the GE pair basis-spectral-denoising fitted to the
physical Gammex scan, ``W(f) = \exp(-(f/f_c)^p)``.

<details><summary><strong>Why the windows are rescaled to this grid</strong></summary>

A window is defined on the frequency axis of the grid it is applied on, `[0, grid_bandlimit]` of
the detector's Nyquist. The pair was fitted on the GE's clinical grid, 512² over 35 cm; this
grid, 512² over 32 cm, has finer pixels and carries higher frequencies, so the windows are the
same functions of physical frequency rescaled onto its axis, ``\exp(-(f\,r/f_c)^p)`` with ``r``
the ratio of the two grids' bandlimits, as basis-spectral-denoising's `RECON_BY` does for its
XCAT grid.
</details>
"""

# ╔═╡ 0ade8600-199d-440a-bfa9-bda793eaeedc
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 220))
    nodes = Dict(
        :hypr => (0.0, 1.0, "Projection HYPR", "3 × 3 on the Counts"),
        :decomp => (3.9, 1.0, "Decompose", "K = 2: Water + Iodine", COLORS.cyan),
        :fdk => (7.8, 1.0, "FDK × 2", "Composite, Complement", COLORS.cyan),
        :acnr => (11.7, 1.0, "ACNR", "Noise Reduction"),
        :image => (15.6, 1.0, "Image HYPR", "On the Complement"),
        :vmi => (19.5, 1.0, "VMIs", "50 / 70 / 100 / 140 keV", COLORS.green),
    )
    edges = [
        :hypr => :decomp,
        :decomp => :fdk,
        :fdk => :acnr,
        :acnr => :image,
        :image => :vmi,
    ]
    flow_chart!(
        fig[1, 1],
        nodes,
        edges;
        limits = (-1.95, 21.45, 0.15, 1.85),
        w = 3.5,
        h = 1.2,
    )
    fig
end

# ╔═╡ 07030006-0000-4000-8000-000000000012
# the published denoiser: ProjectionHYPR(3 × 3 box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 07030006-0000-4000-8000-000000000013
VMI_CHAIN = (
    method = :nchannel,
    controls = BS.NChannelControls(),
    use_tlbf = false,
    antialias = true,
);

# ╔═╡ 07030006-0000-4000-8000-000000000014
# The GE pair of basis-spectral-denoising (`RECON.ge`): W(f) = exp(-(f / f_c)^p) at 11 knots.
GE_KERNELS = (composite = (fc = 0.55, p = 1.5), complement = (fc = 1.05, p = 1.0));

# ╔═╡ 07030006-0000-4000-8000-000000000015
# The ratio of this grid's bandlimit to that of the grid the pair was fitted on: the GE's
# clinical series, 512² × 8 slices over 35 cm.
BANDLIMIT_RATIO = let
    fitted_geom = BS.CTGeometry(
        scanner;
        n_angles = 984,
        fov_cm = 35.0,
        z_cm = 0.5,
        collimation_mm = 5.0,
    )
    BS.grid_bandlimit(first(dual_scans).geom, recon_opts.matrix_size) /
        BS.grid_bandlimit(fitted_geom, (512, 512, 8))
end

# ╔═╡ 07030006-0000-4000-8000-000000000016
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11)), r = BANDLIMIT_RATIO
    window(q) = BS.CustomFilter(
        knots,
        Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots)
    )
    BS.PairFilter(window(GE_KERNELS.composite), window(GE_KERNELS.complement))
end;

# ╔═╡ 07030009-0000-4000-8000-000000000010
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 07030006-0000-4000-8000-000000000020
md"""
### The Calibration Draw: `pair_basis` and `composite_energy`

The pair the windows act on, ``(E^\star, \beta)``, and the energy of the composite that ACNR and
the image-domain HYPR act on are properties of the scanner and protocol, measured once on the
calibration draw and fixed for every other draw, as basis-spectral-denoising's `PAIR_BASIS`
does: the calibration draw is decomposed as measured and after the projection-domain HYPR alone;
`BS.spectral_pair` of the first with a standard soft-tissue window gives ``E^\star`` and
``\beta`` (`pair_basis`); `BS.spectral_pair` of the second on that basis gives the minimum-noise
energy after the projection-domain HYPR (`composite_energy`). Near its minimum the VMI noise
hardly changes with energy, so the argmin of one acquisition is itself noise; measured on a draw
of its own it cannot tune the chain to the draw being scored, and the noise-free reference is
reconstructed with the same pair.
"""

# ╔═╡ 07030006-0000-4000-8000-000000000021
calibration = let
    geom = first(dual_scans).geom
    matrix_size = recon_opts.matrix_size
    channels = [s.channels[DRAWS.calibration] for s in dual_scans]
    decompose(denoiser) = BS.vmi_pipeline(;
        channels,
        basis,
        geom,
        to_backend = to_gpu,
        matrix_size,
        denoiser,
        keep_sinograms = true,
        use_acnr = false,
        VMI_CHAIN...,
    ).sinograms
    pair(d; kw...) = BS.spectral_pair(
        d.water,
        d.iodine,
        geom,
        matrix_size;
        filter = BS.SoftFilter(),
        antialias = VMI_CHAIN.antialias,
        to_backend = to_gpu,
        kw...,
    )
    x = pair(decompose(nothing))
    c = pair(
        decompose(BS.SpectralHYPR(image = nothing));
        basis = (Estar = x.Estar, β = x.β),
    )
    (pair_basis = (Estar = x.Estar, β = x.β), composite_energy = c.Estar)
end

# ╔═╡ 07030007-0000-4000-8000-000000000001
md"""
### `vmi_pipeline`

One call, from the two measured sinograms to the VMI stack, reconstructed on the notebook's grid
with every detector row kept. `keep_sinograms = true` also returns the decomposed basis
sinograms, shown below. The noise-free reference is the noise-free draw through the same
reconstruction, windows and pair, with nothing to denoise (no HYPR, no ACNR).
"""

# ╔═╡ 07030007-0000-4000-8000-000000000010
dual_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.measured] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 07030007-0000-4000-8000-000000000012
reference_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.reference] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 07030007-0000-4000-8000-000000000011
let
    q = dual_vmi.quality
    d = dual_vmi.settings.denoiser
    b = calibration.pair_basis
    pct(x) = round(100x, digits = 3)
    Markdown.parse(
        """
    The windows were rescaled by r = $(round(BANDLIMIT_RATIO, digits = 3)). The calibration draw fixed
    their pair at E* = $(round(Int, b.Estar)) keV, β = $(round(b.β, sigdigits = 3)), and the composite at
    $(round(Int, calibration.composite_energy)) keV. On the measured draw the decomposition solved
    $(q.n_rays) rays in $(round(dual_vmi.elapsed_s, digits = 1)) s, $(round(q.outer_mean, digits = 1)) outer
    iterations on average; $(pct(q.frac_not_converged))% did not converge and
    $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions, the variance-to-mean ratio of the counts on the air rays, of
    $(join(round.(d.dispersion, digits = 2), " and ")) for 80 and 140 kVp. The image HYPR pooled the
    complement over a $(d.image_estimates.window) × $(d.image_estimates.window) window.
    """,
    )
end

# ╔═╡ 07030006-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    mid_r = size(dual_vmi.sinograms.iodine, 2) ÷ 2 + 1
    fig = Mke.Figure(size = (1500, 1000))

    # what the decomposition does to every ray
    nodes = Dict(
        :ray => (0.0, 1.6, "One Ray", "Through the Patient"),
        :low => (4.3, 2.6, "80 kVp Reading", "", COLORS.violet),
        :high => (4.3, 0.6, "140 kVp Reading", "", COLORS.cyan),
        :ml => (8.6, 1.6, "Maximum Likelihood", "Spectral Basis Φ", COLORS.beam),
        :water => (12.9, 2.6, "Water", "g/cm² Along the Ray", COLORS.blue),
        :iodine => (12.9, 0.6, "Iodine", "g/cm² Along the Ray", COLORS.green),
    )
    edges = [
        :ray => :low,
        :ray => :high,
        :low => :ml,
        :high => :ml,
        :ml => :water,
        :ml => :iodine,
    ]
    flow_chart!(
        fig[1, 1:4],
        nodes,
        edges;
        limits = (-2.1, 15.0, -0.1, 3.3),
        w = 3.7,
        h = 1.15,
    )

    # the data: both basis sinograms of the central detector row
    panels = (
        ("Iodine Basis Sinogram", dual_vmi.sinograms.iodine),
        ("Water Basis Sinogram", dual_vmi.sinograms.water),
    )
    for (c, (ttl, sino)) in enumerate(panels)
        slice = permutedims(sino[:, mid_r, :], (2, 1))
        range = Tuple(Float64.(quantile(vec(slice), (0.01, 0.99))))
        ax = Mke.Axis(
            fig[2, 2c - 1];
            title = ttl,
            subtitle = "Central Detector Row",
            xlabel = "View",
            ylabel = "Dexel Column",
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.Colorbar(
            fig[2, 2c];
            colormap = :viridis,
            colorrange = range,
            label = "Line Integral (g/cm²)",
            width = 16,
        )
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.36))
    fig
end

# ╔═╡ 07030008-0000-4000-8000-000000000001
md"""
## 5. Basis Maps and VMIs

The water and iodine basis pair after ACNR and image HYPR (mid slice), beside a drawing of where
the rods sit: the iodine rod should light up in the iodine map and nothing else should. The
VMIs are synthesized from that pair,
``\mu(E) = c_\mathrm{water}\,(\mu/\rho)_\mathrm{water}(E) + c_\mathrm{iodine}\,(\mu/\rho)_\mathrm{iodine}(E)``,
in HU relative to water at the same energy; every VMI comes from the same basis pair.
"""

# ╔═╡ 07030008-0000-4000-8000-000000000030
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 560))
    mid = size(dual_vmi.images.water, 3) ÷ 2 + 1

    ax = canvas(fig[1, 1], (-160, 160, -112, 112))
    ax.title = "Where the Rods Are"
    qrm_sketch!(ax; names = true)

    # iodine in mg/mL on a fixed window around the 5 mg/mL rod (a percentile window of the
    # whole slice sits below the rod, which covers under 1 % of it); water on its percentiles
    iodine_mg = 1000 .* dual_vmi.images.iodine[:, :, mid]
    water = dual_vmi.images.water[:, :, mid]
    panels = (
        ("Iodine Basis Map", iodine_mg, (-2.0, 8.0), "Iodine (mg/mL)"),
        (
            "Water Basis Map",
            water,
            Tuple(Float64.(quantile(vec(water), (0.01, 0.99)))),
            "Water (g/mL)",
        ),
    )
    for (c, (ttl, slice, range, unit)) in enumerate(panels)
        ax = Mke.Axis(
            fig[1, 2c];
            title = ttl,
            subtitle = "Mid Slice, After ACNR and Image HYPR",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(
            fig[1, 2c + 1];
            colormap = :viridis,
            colorrange = range,
            label = unit,
            width = 16,
        )
    end
    fig
end

# ╔═╡ 07030009-0000-4000-8000-000000000020
# The VMI stacks keyed by energy, (512, 512, 3) HU volumes: measured and noise-free.
begin
    vmi_HU_final = Dict(
        Float64(E) => dual_vmi.vmis[:, :, :, k]
        for (k, E) in enumerate(dual_vmi.energies)
    )
    vmi_HU_reference = Dict(
        Float64(E) => reference_vmi.vmis[:, :, :, k]
        for (k, E) in enumerate(reference_vmi.energies)
    )
end;

# ╔═╡ 07030009-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    HU_window = (-200, 500)
    fig = Mke.Figure(size = (1300, 1420))
    mid = size(dual_vmi.vmis, 3) ÷ 2 + 1

    # the synthesis, drawn: two images, two weights, one VMI per energy
    ax = canvas(fig[1, 1:3], (-0.4, 20.4, -0.8, 0.8))
    terms = (
        (1.4, "Water Map", COLORS.blue),
        (5.5, "(μ/ρ) Water at E", COLORS.blue),
        (10.1, "Iodine Map", COLORS.green),
        (14.2, "(μ/ρ) Iodine at E", COLORS.green),
    )
    for (x, name, accent) in terms
        node!(ax, (x, 0.0), name; w = 3.1, h = 1.1, accent, titlesize = 17)
    end
    node!(
        ax,
        (18.6, 0.0),
        "VMI at E";
        w = 2.6,
        h = 1.1,
        accent = COLORS.violet,
        titlesize = 17,
    )
    for (x, op) in ((3.45, "×"), (7.8, "+"), (12.15, "×"), (16.5, "="))
        Mke.text!(
            ax,
            x,
            0.0;
            text = op,
            align = (:center, :center),
            fontsize = 30,
            font = :bold,
            color = COLORS.ink,
        )
    end

    for (k, E) in enumerate(de_vmi_energies)
        ax = Mke.Axis(
            fig[(k - 1) ÷ 2 + 2, (k - 1) % 2 + 1];
            title = "$(Int(E)) keV VMI",
            subtitle = "Mid Slice, Measured Draw",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(
            ax,
            vmi_HU_final[E][:, :, mid];
            colormap = :grays,
            colorrange = HU_window,
        )
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(
        fig[2:3, 3];
        colormap = :grays,
        colorrange = HU_window,
        label = "HU",
        width = 16,
    )
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.09))
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000001
md"""
## 6. Results

Per-rod measured, noise-free and theoretical HU at 50 / 70 / 100 / 140 keV, the water rod's mean
HU, and the noise in the heart:

| quantity | how it is measured |
|---|---|
| **Measured HU** | the mean over a circular ROI at each rod's centroid, over every slice, of the measured VMIs |
| **Noise-free HU** | the same mean on the noise-free reference |
| **Theoretical HU** | ``1000\,(\mu_\mathrm{rod}(E) - \mu_\mathrm{water}(E)) / \mu_\mathrm{water}(E)`` from `BS.compute_μ_at_energy(material, E)`; pure physics, no fitting |
| **Noise** | the standard deviation of the measured VMI minus the noise-free reference, the noise alone |
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000010
const ROD_LABELS = (UInt8(9), UInt8(10), UInt8(11), UInt8(12));

# ╔═╡ 070b0001-0000-4000-8000-000000000011
const ROD_NAMES = ("Water", "Lipid", "Collagen", "Iodine 5 mg/mL");

# ╔═╡ 070b0001-0000-4000-8000-000000000012
const ROD_MATERIALS = (
    BS.XA.Materials.basis_water,
    BS.XA.Materials.basis_lipid,
    BS.XA.Materials.basis_collagen,
    BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 070b0001-0000-4000-8000-000000000013
const ROD_ROI_RADIUS_PX = 8; # 5 mm at 0.625 mm/voxel, inside the 7.5 mm rods

# ╔═╡ 070b0001-0000-4000-8000-000000000014
const HEART_NOISE_ROI_RADIUS_PX = 12; # 7.5 mm: fits between the four rods

# ╔═╡ 070b0001-0000-4000-8000-000000000015
begin
    "Every value of `vol` inside the in-plane `roi`, over all slices."
    roi_values(vol, roi) = Float64[vol[c, z] for z in axes(vol, 3) for c in roi]

    # Rod centroids from the phantom labels on the recon grid; the noise ROI sits at the
    # centre of the rod cross, the mean of the four rod centroids.
    rod_rois = let
        mask_2d = phantom_in_recon[:, :, size(phantom_in_recon, 3) ÷ 2 + 1]
        nx, ny = size(mask_2d)
        function centroid(label)
            idx = findall(==(label), mask_2d)
            isempty(idx) && error("no recon voxel carries label $(label)")
            (sum(c -> c[1], idx) / length(idx), sum(c -> c[2], idx) / length(idx))
        end
        disc(c, r) = [
            CartesianIndex(i, j)
            for j in 1:ny
            for i in 1:nx
            if (i - c[1])^2 + (j - c[2])^2 ≤ r^2
        ]
        centers = Dict(lab => centroid(lab) for lab in ROD_LABELS)
        heart_center = (
            mean(first(centers[l]) for l in ROD_LABELS),
            mean(last(centers[l]) for l in ROD_LABELS)
        )
        (
            centers = centers,
            rods = Dict(lab => disc(centers[lab], ROD_ROI_RADIUS_PX) for lab in ROD_LABELS),
            heart_center = heart_center,
            heart = disc(heart_center, HEART_NOISE_ROI_RADIUS_PX),
        )
    end
end;

# ╔═╡ 4287d0d6-2dda-4b59-aedf-cfffed63d6cb
"""
    recon_mm(c) -> (x, y)

The world position (mm) of in-plane recon-pixel index `c = (i, j)`, as `recon_to_world_affine`
places it: the drawings put each ROI where the image has it.
"""
recon_mm(c) = Tuple(
    10 *
        (
            -recon_opts.fov_cm / 2 +
                (c[k] - 0.5) * recon_opts.fov_cm / recon_opts.matrix_size[k]
        )
    for k in 1:2
);

# ╔═╡ 070b0001-0000-4000-8000-000000000000
md"""
### Phantom-to-Recon Alignment

Before trusting any ROI built on `phantom_in_recon`: the drawing shows what nearest-neighbour
resampling does at a rod edge, to scale (each 0.625 mm pixel takes the label of the 0.2 mm
voxel under its centre). Then the 70 keV VMI and the resampled labels side by side on the same
grid, and overlaid, so the rod, heart and bone edges can be checked against the image.
"""

# ╔═╡ 31a99a3f-b833-4375-bfdf-5f2447ce008c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 600))
    rod = QRM_GEOMETRY.heart_rods[1] # the water rod
    c0 = (
        rod.center[1] + rod.r * cosd(45),
        rod.center[2] + rod.r * sind(45),
    )          # on its edge, 45° up-right
    half = 2.0 # mm, half the zoom window

    ax = canvas(fig[1, 1], (-160, 160, -112, 112))
    ax.title = "Zoom on the Water Rod's Edge"
    qrm_sketch!(ax)
    Mke.lines!(
        ax,
        Mke.Rect2f(c0[1] - 8, c0[2] - 8, 16, 16);
        color = COLORS.red,
        linewidth = 3,
    )

    ax = Mke.Axis(
        fig[1, 2];
        title = "Nearest Neighbour at the Rod Edge, to Scale",
        subtitle = "Recon Pixels Whose Centre Falls in the Rod Take Its Label",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
    )
    vp = 10 * QRM_VOXEL_SIZE_CM[1]
    vr = 10 * recon_opts.fov_cm / recon_opts.matrix_size[1]
    xlo, xhi, ylo, yhi = c0[1] - half, c0[1] + half, c0[2] - half, c0[2] + half
    n_ph = size(mask_3d)
    # phantom voxel centres and recon pixel centres (mm), each grid centred on the isocentre
    ph_x(i) = (i - (n_ph[1] + 1) / 2) * vp
    ph_y(j) = (j - (n_ph[2] + 1) / 2) * vp
    in_rod(x, y) = hypot(x - rod.center[1], y - rod.center[2]) <= rod.r
    nearest_in_rod(x, y) = in_rod(
        ph_x(round(Int, x / vp + (n_ph[1] + 1) / 2)),
        ph_y(round(Int, y / vp + (n_ph[2] + 1) / 2)),
    )
    for i in 1:recon_opts.matrix_size[1], j in 1:recon_opts.matrix_size[2]
        x, y = recon_mm((i, j))
        (xlo - vr <= x <= xhi + vr && ylo - vr <= y <= yhi + vr) || continue
        Mke.poly!(
            ax,
            Mke.Rect2f(x - vr / 2, y - vr / 2, vr, vr);
            color = nearest_in_rod(x, y) ? (ROD_COLORS[1], 0.55) : (COLORS.card, 0.9),
            strokecolor = COLORS.cyan,
            strokewidth = 2,
        )
    end
    edges = [(k - 0.5) * vp for k in floor(Int, xlo / vp):ceil(Int, xhi / vp)]
    Mke.vlines!(ax, edges; color = (COLORS.violet, 0.45), linewidth = 1)
    yedges = [
        ph_y(j) + vp / 2
        for j in 1:n_ph[2]
        if ylo - vp <= ph_y(j) + vp / 2 <= yhi + vp
    ]
    Mke.hlines!(ax, yedges; color = (COLORS.violet, 0.45), linewidth = 1)
    θ = range(0.2, 1.4; length = 100)
    Mke.lines!(
        ax,
        rod.center[1] .+ rod.r .* cos.(θ),
        rod.center[2] .+ rod.r .* sin.(θ);
        color = COLORS.ink,
        linewidth = 3.5,
    )
    Mke.limits!(ax, xlo, xhi, ylo, yhi)
    Mke.Legend(
        fig[1, 3],
        [
            Mke.LineElement(color = COLORS.ink, linewidth = 3.5),
            Mke.LineElement(color = (COLORS.violet, 0.6), linewidth = 2),
            Mke.PolyElement(
                color = (COLORS.card, 0.9),
                strokecolor = COLORS.cyan,
                strokewidth = 2,
            ),
            Mke.PolyElement(
                color = (ROD_COLORS[1], 0.55),
                strokecolor = COLORS.cyan,
                strokewidth = 2,
            ),
        ],
        [
            "Rod Edge (r = $(rod.r) mm)",
            "Phantom Voxels ($(vp) mm)",
            "Recon Pixel ($(vr) mm)",
            "Pixel Labelled Water",
        ],
    )
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-00000000000a
let
    Mke.set_theme!(THEME)
    z = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    hu_slice = vmi_HU_final[70.0][:, :, z]
    pir_slice = phantom_in_recon[:, :, clamp(z, 1, size(phantom_in_recon, 3))]
    rod_overlay = Float32[v in ROD_LABELS ? Float32(v) : NaN32 for v in pir_slice]
    # label 0 = outside the phantom grid (the 32 cm FOV is taller than the 22 cm phantom)
    all_labels = Float32[v == 0 ? NaN32 : Float32(v) for v in pir_slice]
    full_overlay = Float32[v ≤ 1 ? NaN32 : Float32(v) for v in pir_slice]
    labels_cmap = Mke.cgrad(:Paired_12; categorical = true)

    fig = Mke.Figure(size = (1400, 1340))
    hu_kwargs = (colormap = :grays, colorrange = (-200, 500))
    square = (aspect = Mke.DataAspect(),)

    ax = Mke.Axis(
        fig[1, 1];
        title = "70 keV VMI",
        subtitle = "Slice $(z) of $(size(vmi_HU_final[70.0], 3))",
        square...,
    )
    Mke.heatmap!(ax, hu_slice; hu_kwargs...)
    Mke.hidedecorations!(ax)

    ax = Mke.Axis(
        fig[1, 2];
        title = "phantom_in_recon, :nearest",
        subtitle = "$(count(>(0), unique(pir_slice))) Labels on the Recon Grid; White: Outside the Phantom Grid",
        square...,
    )
    Mke.heatmap!(
        ax,
        all_labels;
        colormap = labels_cmap,
        colorrange = (0.5, 12.5),
        nan_color = :white,
    )
    Mke.hidedecorations!(ax)

    ax = Mke.Axis(
        fig[2, 1];
        title = "VMI + Rod Labels (9–12)",
        subtitle = "Rod Edges Should Sit on the Image's Rod Edges",
        square...,
    )
    Mke.heatmap!(ax, hu_slice; hu_kwargs...)
    Mke.heatmap!(
        ax,
        rod_overlay;
        colormap = labels_cmap,
        colorrange = (0.5, 12.5),
        alpha = 0.6,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(ax)

    ax = Mke.Axis(
        fig[2, 2];
        title = "VMI + Every Non-Air Label",
        subtitle = "Bone, Heart and Lung Edges",
        square...,
    )
    Mke.heatmap!(ax, hu_slice; hu_kwargs...)
    Mke.heatmap!(
        ax,
        full_overlay;
        colormap = labels_cmap,
        colorrange = (0.5, 12.5),
        alpha = 0.4,
        nan_color = (:white, 0.0),
    )
    Mke.hidedecorations!(ax)
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000002
md"""
### Water ROI

The water-rod ROI (label 9, `basis_water`), drawn on the phantom and laid on the 70 keV VMI, and
its mean HU at every VMI energy, measured and noise-free. Water is the reference of the HU scale,
so the bars should sit at 0 HU: a constant offset would be a basis-decomposition bias, a trend
with energy a spectral-model error, and a difference between the two bars what the noise and the
denoising add.
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000004
begin
    water_rod_hu = [
        mean(roi_values(vmi_HU_final[E], rod_rois.rods[UInt8(9)]))
        for E in de_vmi_energies
    ]
    water_rod_hu_reference = [
        mean(roi_values(vmi_HU_reference[E], rod_rois.rods[UInt8(9)]))
        for E in de_vmi_energies
    ]
end;

# ╔═╡ 070b0001-0000-4000-8000-000000000003
let
    Mke.set_theme!(THEME)
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    bg = vmi_HU_final[70.0][:, :, mid]
    overlay = fill(NaN32, size(bg))
    overlay[rod_rois.rods[UInt8(9)]] .= 1.0f0
    px_mm = 10 * recon_opts.fov_cm / recon_opts.matrix_size[1]
    fig = Mke.Figure(size = (1400, 1180))

    ax = canvas(fig[1, 1], (-160, 160, -112, 112))
    ax.title = "The Water-Rod ROI, Drawn"
    ax.subtitle = "$(round(ROD_ROI_RADIUS_PX * px_mm; digits = 1)) mm Radius, Every Slice"
    qrm_sketch!(
        ax;
        names = true,
        rois = (
            (recon_mm(rod_rois.centers[UInt8(9)]), ROD_ROI_RADIUS_PX * px_mm, COLORS.red),
        ),
    )
    ax1 = Mke.Axis(
        fig[1, 2];
        title = "The Water-Rod ROI, Measured",
        subtitle = "On the 70 keV VMI",
        aspect = Mke.DataAspect(),
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(ax1, overlay; colormap = :reds, alpha = 0.6, nan_color = (:white, 0.0))
    Mke.hidedecorations!(ax1)

    n_E = length(de_vmi_energies)
    ax2 = Mke.Axis(
        fig[2, 1:2];
        title = "Water-Rod Mean HU at Every VMI Energy",
        subtitle = "Theory: 0 HU",
        xlabel = "VMI Energy (keV)",
        ylabel = "Mean HU",
        xticks = (collect(1:n_E), ["$(Int(E))" for E in de_vmi_energies]),
    )
    values = vcat(water_rod_hu, water_rod_hu_reference)
    group = vcat(fill(1, n_E), fill(2, n_E))
    bar_colors = (ROD_COLORS[1], COLORS.rule)
    Mke.barplot!(
        ax2,
        vcat(1:n_E, 1:n_E),
        values;
        dodge = group,
        color = [bar_colors[g] for g in group],
    )
    Mke.hlines!(ax2, [0.0]; color = COLORS.ink, linewidth = 1.5, linestyle = :dash)
    for (k, h) in pairs(water_rod_hu)
        Mke.text!(
            ax2,
            k - 0.2,
            h;
            text = "$(round(h, digits = 1))",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 17,
            offset = (0, h ≥ 0 ? 4 : -4),
        )
    end
    Mke.axislegend(
        ax2,
        [Mke.PolyElement(color = bar_colors[1]), Mke.PolyElement(color = bar_colors[2])],
        ["Measured", "Noise-Free Reference"];
        position = :lb,
    )
    y_max = max(15.0, 1.2 * maximum(abs, values))
    Mke.ylims!(ax2, -y_max, y_max)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.5))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_vmi_water_roi_check.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 070d0001-0000-4000-8000-000000000001
md"""
### Heart-Centre Noise ROI

The noise in a 7.5 mm ROI in the soft-tissue heart, at the centre of the four rods, at every VMI
energy: the standard deviation of the measured VMI about the noise-free reference, which is how
the chain carries noise from the basis pair into each VMI.
"""

# ╔═╡ 070d0001-0000-4000-8000-000000000020
vmi_noise_by_keV = Dict(
    E =>
        let v = roi_values(vmi_HU_final[E], rod_rois.heart),
            v0 = roi_values(vmi_HU_reference[E], rod_rois.heart)
            (mean = mean(v), std = std(v .- v0), n = length(v))
        end
    for E in de_vmi_energies
);

# ╔═╡ 070d0001-0000-4000-8000-000000000015
# The basis-pair noise covariance in the heart ROI (measured minus noise-free), which sets the
# noise of every VMI:
#   σ_HU(E)² = 10⁶ V_w + α(E)² V_i + 2·10³ α(E) C_iw,  α(E) = (μ/ρ)_I(E) / (μ/ρ)_w(E),
# with water in g/mL and iodine in mg/mL; lowest at α* = -10³ C_iw / V_i.
basis_covariance = let
    w = roi_values(dual_vmi.images.water, rod_rois.heart) .-
        roi_values(reference_vmi.images.water, rod_rois.heart)
    i = (
        roi_values(dual_vmi.images.iodine, rod_rois.heart) .-
            roi_values(reference_vmi.images.iodine, rod_rois.heart)
    ) .*
        1000.0 # mg/mL
    Vw, Vi = mean(abs2, w .- mean(w)), mean(abs2, i .- mean(i))
    Ciw = mean((w .- mean(w)) .* (i .- mean(i)))
    α(E) = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E) /
        BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)
    α_star = -1000.0 * Ciw / Vi
    E_star = let grid = 40.0:1.0:140.0
        grid[argmin(abs.(α.(grid) .- α_star))]
    end
    (
        σ_water = sqrt(Vw),
        σ_iodine = sqrt(Vi),
        ρ = Ciw / sqrt(Vw * Vi),
        α_star,
        E_star,
        α_range = (α(140.0), α(40.0)),
    )
end;

# ╔═╡ 070d0001-0000-4000-8000-000000000016
let c = basis_covariance
    inside = c.α_range[1] ≤ c.α_star ≤ c.α_range[2]
    Markdown.parse(
        """
    In the heart ROI the noise of the water basis is σ = $(round(1000 * c.σ_water, digits = 1)) mg/mL, that
    of the iodine basis σ = $(round(c.σ_iodine, digits = 2)) mg/mL, and their correlation is
    ρ = $(round(c.ρ, digits = 2)). The noise-optimal mixing ratio is α* = $(round(c.α_star, digits = 1)) HU per mg/mL""" *
            (
                inside ?
                    ", reached at about $(Int(c.E_star)) keV; the chain's composite is at $(round(Int, calibration.composite_energy)) keV." :
                    ", outside the 40–140 keV range, so the noise falls monotonically toward $(c.α_star < c.α_range[1] ? "high" : "low") energy."
            ),
    )
end

# ╔═╡ 070d0001-0000-4000-8000-000000000030
let
    Mke.set_theme!(THEME)
    HU_window = (-200, 500)
    mid = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    bg = vmi_HU_final[70.0][:, :, mid]
    overlay = fill(NaN32, size(bg))
    overlay[rod_rois.heart] .= 1.0f0
    px_mm = 10 * recon_opts.fov_cm / recon_opts.matrix_size[1]
    fig = Mke.Figure(size = (1400, 1180))

    ax = canvas(fig[1, 1], (-160, 160, -112, 112))
    ax.title = "The Heart-Centre ROI, Drawn"
    ax.subtitle = "$(round(HEART_NOISE_ROI_RADIUS_PX * px_mm; digits = 1)) mm Radius, Between the Rods"
    qrm_sketch!(
        ax;
        rois = (
            (
                recon_mm(rod_rois.heart_center),
                HEART_NOISE_ROI_RADIUS_PX * px_mm,
                COLORS.red,
            ),
        ),
    )
    ax1 = Mke.Axis(
        fig[1, 2];
        title = "The Heart-Centre ROI, Measured",
        subtitle = "On the 70 keV VMI",
        aspect = Mke.DataAspect(),
    )
    Mke.heatmap!(ax1, bg; colormap = :grays, colorrange = HU_window)
    Mke.heatmap!(ax1, overlay; colormap = :reds, alpha = 0.6, nan_color = (:white, 0.0))
    Mke.hidedecorations!(ax1)

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σs = [vmi_noise_by_keV[E].std for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]
    ax2 = Mke.Axis(
        fig[2, 1:2];
        title = "Heart-Centre Noise vs VMI Energy",
        subtitle = "$(length(rod_rois.heart)) Pixels × $(size(vmi_HU_final[70.0], 3)) Slices, About the Noise-Free Reference",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        xticks = Es,
    )
    Mke.scatterlines!(ax2, Es, σs; color = COLORS.violet, markersize = 18, linewidth = 3)
    for (E, σ, μ) in zip(Es, σs, μs)
        Mke.text!(
            ax2,
            E,
            σ;
            text = "σ = $(round(σ; digits = 1)) HU\nMean $(round(μ; digits = 1)) HU",
            align = (:center, :bottom),
            fontsize = 16,
            offset = (0, 10),
        )
    end
    Mke.ylims!(ax2, 0, maximum(σs) * 1.5)
    Mke.xlims!(ax2, first(Es) - 12, last(Es) + 12)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.5))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_vmi_noise_roi.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000008
md"""
### Basis Values in the Rods

The mean basis pair in each rod ROI. The water rod should read ``c_\mathrm{water} ≈ 1`` g/mL and
``c_\mathrm{iodine} ≈ 0``; the iodine rod carries its 5 mg/mL on a water-equivalent base. Lipid
and collagen are not made of water and iodine, so the chain represents them as the water/iodine
mixture that attenuates the same way (lipid with a negative iodine value).
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000009
let
    function row(lab, name, mat)
        cw = mean(roi_values(dual_vmi.images.water, rod_rois.rods[lab]))
        ci = 1000 * mean(roi_values(dual_vmi.images.iodine, rod_rois.rods[lab]))
        return "| $(name) | $(round(BS.XA.val(mat.density), digits = 3)) | $(round(cw, digits = 3)) | $(round(ci, digits = 2)) |"
    end
    rows = [
        row(lab, name, mat)
        for (lab, name, mat) in zip(ROD_LABELS, ROD_NAMES, ROD_MATERIALS)
    ]
    Markdown.parse(
        """
    | rod | density (g/cm³) | water (g/mL) | iodine (mg/mL) |
    |-----|------:|------:|------:|
    $(join(rows, "\n"))
    """,
    )
end

# ╔═╡ 070b0001-0000-4000-8000-000000000020
rod_data = let
    μ_water(E) = BS.compute_μ_at_energy(BS.XA.Materials.water, E)
    theoretical_hu(mat, E) = 1000.0 *
        (BS.compute_μ_at_energy(mat, E) - μ_water(E)) /
        μ_water(E)
    meas = [
        mean(roi_values(vmi_HU_final[E], rod_rois.rods[lab]))
        for lab in ROD_LABELS, E in de_vmi_energies
    ]
    ref = [
        mean(roi_values(vmi_HU_reference[E], rod_rois.rods[lab]))
        for lab in ROD_LABELS, E in de_vmi_energies
    ]
    theo = [theoretical_hu(mat, E) for mat in ROD_MATERIALS, E in de_vmi_energies]
    rmse(a) = [sqrt(mean(abs2, a[i, :] .- theo[i, :])) for i in eachindex(ROD_LABELS)]
    (
        labels = ROD_LABELS,
        names = ROD_NAMES,
        measured = meas,
        reference = ref,
        theoretical = theo,
        rmse = rmse(meas),
        rmse_reference = rmse(ref),
    )
end;

# ╔═╡ 070b0001-0000-4000-8000-000000000030
md"""
### Per-Rod Measured vs Theoretical HU

Each rod in its own colour, as drawn: solid lines with dots are the measured HU, crosses the
noise-free reference and dashed lines the theoretical HU.
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000031
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 1150))
    px_mm = 10 * recon_opts.fov_cm / recon_opts.matrix_size[1]

    ax = canvas(fig[1, 1:2], (-160, 160, -112, 112))
    ax.title = "The Four Rod ROIs"
    qrm_sketch!(
        ax;
        names = true,
        rois = [
            (recon_mm(rod_rois.centers[lab]), ROD_ROI_RADIUS_PX * px_mm, COLORS.ink)
            for lab in ROD_LABELS
        ],
    )

    ax = Mke.Axis(
        fig[2, 1];
        title = "Pure-Material Rods, Dual-kVp VMIs",
        subtitle = "Mean HU Over Each Rod ROI, Every Slice",
        xlabel = "VMI Energy (keV)",
        ylabel = "HU",
        xticks = de_vmi_energies,
    )
    rod_lines = Any[]
    rod_labels_str = String[]
    for i in eachindex(rod_data.names)
        c = ROD_COLORS[i]
        Mke.scatterlines!(
            ax,
            de_vmi_energies,
            rod_data.measured[i, :];
            color = c,
            linewidth = 3,
            markersize = 12,
        )
        Mke.scatter!(
            ax,
            de_vmi_energies,
            rod_data.reference[i, :];
            color = COLORS.ink,
            marker = :xcross,
            markersize = 13,
        )
        Mke.lines!(
            ax,
            de_vmi_energies,
            rod_data.theoretical[i, :];
            color = c,
            linewidth = 2,
            linestyle = :dash,
        )
        push!(rod_lines, Mke.LineElement(color = c, linewidth = 3))
        push!(
            rod_labels_str,
            "$(rod_data.names[i]) (RMSE $(round(rod_data.rmse[i], digits = 1)) HU)",
        )
    end
    Mke.Legend(
        fig[2, 2],
        vcat(
            [
                Mke.MarkerElement(color = COLORS.ink, marker = :circle, markersize = 12),
                Mke.MarkerElement(color = COLORS.ink, marker = :xcross, markersize = 13),
                Mke.LineElement(color = COLORS.ink, linewidth = 2, linestyle = :dash),
            ],
            rod_lines,
        ),
        vcat(["Measured", "Noise-Free Reference", "Theory"], rod_labels_str),
    )
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.48))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_vmi_vs_theoretical.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 070b0001-0000-4000-8000-000000000032
let
    header = "| rod | " *
        join(["$(Int(E)) keV" for E in de_vmi_energies], " | ") *
        " | RMSE vs theory | RMSE vs theory, noise-free |"
    rule = "|---|" * repeat("---:|", length(de_vmi_energies) + 2)
    r1(x) = round(x, digits = 1)
    function row(i)
        cells = [
            "$(r1(rod_data.measured[i, j])) / $(r1(rod_data.reference[i, j])) / $(r1(rod_data.theoretical[i, j]))"
            for j in eachindex(de_vmi_energies)
        ]
        return "| $(rod_data.names[i]) | " *
            join(cells, " | ") *
            " | $(r1(rod_data.rmse[i])) | $(r1(rod_data.rmse_reference[i])) |"
    end
    rows = [row(i) for i in eachindex(rod_data.names)]
    Markdown.parse(
        "Measured / noise-free / theoretical HU per rod and energy, in HU:\n\n" *
            join([header, rule, rows...], "\n"),
    )
end

# ╔═╡ 841e8099-387c-4cf4-9c3a-31de285ce0cf
md"""
## 7. Verification
"""

# ╔═╡ 070b0001-0000-4000-8000-000000000090
verification = let
    vmi_finite = all(isfinite, dual_vmi.vmis) && all(isfinite, reference_vmi.vmis)
    checks = [
        (
            name = "all channels enter the decomposition",
            value = basis.n_channels,
            pass = basis.n_channels == 2,
        ),
        (
            name = "Σ response matches I0 (max relative error)",
            value = round(basis.I0_relerr, sigdigits = 2),
            pass = basis.I0_relerr < 5.0e-5,
        ),
        (
            name = "chain: SpectralHYPR, ACNR, pair of windows",
            value = (
                dual_vmi.settings.denoiser !== nothing,
                dual_vmi.settings.acnr !== nothing,
                dual_vmi.settings.recon.filter isa BS.PairFilter,
            ),
            pass = dual_vmi.settings.denoiser !== nothing &&
                dual_vmi.settings.acnr !== nothing &&
                dual_vmi.settings.recon.filter isa BS.PairFilter,
        ),
        (
            name = "measured and reference share the pair",
            value = dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis,
            pass = dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis,
        ),
        (
            name = "VMI energies",
            value = de_vmi_energies,
            pass = de_vmi_energies == [50.0, 70.0, 100.0, 140.0],
        ),
        (name = "VMI values finite", value = vmi_finite, pass = vmi_finite),
        (
            name = "water rod within ±10 HU at every energy",
            value = round.(water_rod_hu, digits = 1),
            pass = all(abs.(water_rod_hu) .≤ 10),
        ),
    ]
    passed = count(check -> check.pass, checks)
    rows = join(
        [
            "| $(c.name) | $(c.value) | $(c.pass ? "✅ pass" : "❌ CHECK (failed)") |"
            for c in checks
        ],
        "\n",
    )
    Markdown.parse(
        """
    #### $(passed == length(checks) ? "✅ Verification: PASS ($(passed)/$(length(checks)))" : "❌ Verification: FAIL ($(passed)/$(length(checks)))")

    | check | value | status |
    |---|---:|:---|
    $rows
    """,
    )
end

# ╔═╡ 070c0001-0000-4000-8000-000000000001
let
    r = rod_data
    fmt(v) = join(round.(v, digits = 1), " / ")
    σ = [vmi_noise_by_keV[E].std for E in de_vmi_energies]
    rmse = join(
        ["$(n) $(round(e, digits = 1)) HU" for (n, e) in zip(r.names, r.rmse)],
        ", ",
    )
    Markdown.parse(
        """
    ## Summary

    The analytic QRM-Thorax (1600 × 1100 × 20 at 0.2 mm, four pure-material rods in the heart)
    was scanned at 80 and 140 kVp by rapid switching on the GE Apex Elite (duty cycle
    0.65 / 0.35, `view_samples = 5`), its two sinograms turned into a spectral basis, the
    calibration draw fixed the pair (E* = $(round(Int, calibration.pair_basis.Estar)) keV) and the
    composite ($(round(Int, calibration.composite_energy)) keV), and `vmi_pipeline` made the VMIs.
    At 50 / 70 / 100 / 140 keV:

    - the water rod reads $(fmt(water_rod_hu)) HU, and $(fmt(water_rod_hu_reference)) HU noise-free (theory: 0 HU);
    - measured vs theoretical HU agree to an RMSE of $(rmse);
    - the noise in the heart is σ = $(fmt(σ)) HU.

    Notebook 08 runs the same phantom, rods, grid, energies and ROIs through a photon-counting
    scanner, so the two notebooks compare directly.
    """,
    )
end

# ╔═╡ Cell order:
# ╟─07010001-0000-4000-8000-000000000001
# ╟─8975ea65-20d3-41a6-9893-982aa954dbb6
# ╟─27c06485-bdf2-4288-9efb-22e9b11ee3a0
# ╟─0f80b460-20be-43bd-aef4-7374aaa15a2b
# ╟─07010002-0000-4000-8000-000000000001
# ╠═07010003-0000-4000-8000-000000000001
# ╠═07010003-0000-4000-8000-000000000005
# ╠═07010003-0000-4000-8000-000000000006
# ╠═07010003-0000-4000-8000-000000000010
# ╠═07010003-0000-4000-8000-000000000011
# ╠═07010003-0000-4000-8000-000000000012
# ╠═07010003-0000-4000-8000-000000000013
# ╠═07010003-0000-4000-8000-000000000040
# ╟─07010003-0000-4000-8000-000000000050
# ╟─c0a15200-4f8c-4a6d-afdc-f069970dcf0c
# ╠═da68bdcc-b970-41e2-8d8c-a8baf8869d89
# ╠═a1347d46-0163-4ca4-9247-4cd901bfade1
# ╠═6b51fcdc-9a60-4c4f-9031-39214c0cd354
# ╠═127012b1-df1a-4c12-b7e5-6eddcf183923
# ╠═018a50f5-996d-447b-974e-ab87a2e181e8
# ╟─07020001-0000-4000-8000-000000000001
# ╠═07020001-0000-4000-8000-000000000010
# ╠═07020001-0000-4000-8000-000000000013
# ╠═07020001-0000-4000-8000-000000000014
# ╟─07020001-0000-4000-8000-000000000025
# ╠═07020001-0000-4000-8000-000000000026
# ╠═07020001-0000-4000-8000-000000000029
# ╠═07020001-0000-4000-8000-000000000040
# ╠═07020001-0000-4000-8000-000000000050
# ╠═07020001-0000-4000-8000-000000000051
# ╠═07020001-0000-4000-8000-000000000060
# ╟─07020001-0000-4000-8000-000000000061
# ╟─07020001-0000-4000-8000-000000000062
# ╟─07030001-0000-4000-8000-000000000001
# ╠═07030001-0000-4000-8000-000000000010
# ╟─524abd9e-0288-5a3b-b911-28cb8ee1ec73
# ╟─07030002-0000-4000-8000-000000000001
# ╠═07030002-0000-4000-8000-000000000010
# ╠═07030002-0000-4000-8000-000000000011
# ╠═07030002-0000-4000-8000-000000000012
# ╠═07030002-0000-4000-8000-000000000020
# ╟─07030002-0000-4000-8000-000000000021
# ╟─07030003-0000-4000-8000-000000000001
# ╠═07030003-0000-4000-8000-000000000010
# ╠═07030003-0000-4000-8000-000000000011
# ╠═07030003-0000-4000-8000-000000000020
# ╟─07030004-0000-4000-8000-000000000001
# ╠═ecbcace5-f22f-4b66-8847-b7aad4b23554
# ╠═07030004-0000-4000-8000-000000000010
# ╟─07030004-0000-4000-8000-000000000011
# ╠═07030004-0000-4000-8000-000000000025
# ╟─07030004-0000-4000-8000-000000000030
# ╟─07030000-0000-4000-8000-0000000000f1
# ╠═07030006-0000-4000-8000-000000000010
# ╟─07030006-0000-4000-8000-000000000011
# ╟─07030006-0000-4000-8000-000000000001
# ╟─0ade8600-199d-440a-bfa9-bda793eaeedc
# ╠═07030006-0000-4000-8000-000000000012
# ╠═07030006-0000-4000-8000-000000000013
# ╠═07030006-0000-4000-8000-000000000014
# ╠═07030006-0000-4000-8000-000000000015
# ╠═07030006-0000-4000-8000-000000000016
# ╠═07030009-0000-4000-8000-000000000010
# ╟─07030006-0000-4000-8000-000000000020
# ╠═07030006-0000-4000-8000-000000000021
# ╟─07030007-0000-4000-8000-000000000001
# ╠═07030007-0000-4000-8000-000000000010
# ╠═07030007-0000-4000-8000-000000000012
# ╟─07030007-0000-4000-8000-000000000011
# ╟─07030006-0000-4000-8000-000000000040
# ╟─07030008-0000-4000-8000-000000000001
# ╟─07030008-0000-4000-8000-000000000030
# ╠═07030009-0000-4000-8000-000000000020
# ╟─07030009-0000-4000-8000-000000000040
# ╟─070b0001-0000-4000-8000-000000000001
# ╠═070b0001-0000-4000-8000-000000000010
# ╠═070b0001-0000-4000-8000-000000000011
# ╠═070b0001-0000-4000-8000-000000000012
# ╠═070b0001-0000-4000-8000-000000000013
# ╠═070b0001-0000-4000-8000-000000000014
# ╠═070b0001-0000-4000-8000-000000000015
# ╠═4287d0d6-2dda-4b59-aedf-cfffed63d6cb
# ╟─070b0001-0000-4000-8000-000000000000
# ╟─31a99a3f-b833-4375-bfdf-5f2447ce008c
# ╟─070b0001-0000-4000-8000-00000000000a
# ╟─070b0001-0000-4000-8000-000000000002
# ╠═070b0001-0000-4000-8000-000000000004
# ╟─070b0001-0000-4000-8000-000000000003
# ╟─070d0001-0000-4000-8000-000000000001
# ╠═070d0001-0000-4000-8000-000000000020
# ╠═070d0001-0000-4000-8000-000000000015
# ╟─070d0001-0000-4000-8000-000000000016
# ╟─070d0001-0000-4000-8000-000000000030
# ╟─070b0001-0000-4000-8000-000000000008
# ╟─070b0001-0000-4000-8000-000000000009
# ╠═070b0001-0000-4000-8000-000000000020
# ╟─070b0001-0000-4000-8000-000000000030
# ╟─070b0001-0000-4000-8000-000000000031
# ╟─070b0001-0000-4000-8000-000000000032
# ╟─841e8099-387c-4cf4-9c3a-31de285ce0cf
# ╟─070b0001-0000-4000-8000-000000000090
# ╟─070c0001-0000-4000-8000-000000000001
