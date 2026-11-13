### A Pluto.jl notebook ###
# v0.3.0

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

# ╔═╡ 05000001-0000-4000-8000-000000000010
md"""
# XCAT to CT: Phantom Grids and the Affine Round Trip

**Scan the heart of an XCAT chest, then lay its ground-truth labels exactly onto the
reconstructed image, axial and helical.**

A simulation has two voxel grids: the **phantom grid** (the ground truth, here an XCAT 3.0 chest
read at 0.4 mm) and the **reconstruction grid** (a centred stack of axial slices). Organ ROI
statistics, segmentation scores and partial-volume fractions all need the map between them. This
notebook builds that map with two affines and one resampler, and proves it exact.
"""

# ╔═╡ 5ec9c15c-c610-4636-8a15-ee5215cd87e0
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scan", "The Map"],
        [
            "XCAT Chest, Cropped to the Heart",
            "Axial and Helical, Centred on the Isocentre",
            "Two Grids, One World Frame",
        ];
        limits = [(-18.5, 18.5, -13, 13), (-7.5, 7.5, -7.5, 7.5), (-6.4, 6.4, -6.4, 6.4)],
    )

    # the chest and the cardiac crop box
    chest_sketch!(axes[1]; dark = true)
    Mke.poly!(
        axes[1],
        rounded_rect((2.2, 1.4), 15.0, 12.6, 0.3);
        color = :transparent,
        strokecolor = COLORS.cyan,
        strokewidth = 3,
        linestyle = :dash,
    )
    Mke.text!(
        axes[1],
        10.1,
        7.4;
        text = "Crop",
        color = COLORS.cyan,
        fontsize = 19,
        font = :bold,
        align = (:left, :top),
    )

    # the cropped block alone in the gantry, inside the 14 cm field
    gantry!(axes[2]; labels = ("X-Ray Tube",), object = false)
    Mke.poly!(
        axes[2],
        rounded_rect((0.0, 0.0), 2.6, 2.3, 0.1);
        color = (COLORS.red, 0.8),
        strokecolor = COLORS.night_ink,
        strokewidth = 1.5,
    )
    Mke.lines!(
        axes[2],
        Mke.Circle(Mke.Point2f(0, 0), 1.9f0);
        color = COLORS.cyan,
        linewidth = 2.5,
        linestyle = :dash,
    )

    # the phantom grid and the reconstruction grid, both centred on the isocentre
    grid!(axes[3], (0.0, 0.0), (9, 7), 1.0; color = COLORS.violet, linewidth = 1.8)
    grid!(axes[3], (0.0, 0.0), (5, 5), 1.55; color = COLORS.cyan, linewidth = 2.6)
    Mke.scatter!(
        axes[3],
        [Mke.Point2f(0, 0)];
        marker = :cross,
        markersize = 26,
        color = COLORS.beam,
    )
    Mke.text!(
        axes[3],
        -4.5,
        -3.9;
        text = "Phantom",
        color = COLORS.violet,
        fontsize = 18,
        font = :bold,
        align = (:left, :top),
    )
    Mke.text!(
        axes[3],
        3.9,
        4.1;
        text = "Recon",
        color = COLORS.cyan,
        fontsize = 18,
        font = :bold,
        align = (:right, :bottom),
    )
    fig
end

# ╔═╡ 972e9b27-0fa9-4850-96d8-dd1745391886
md"""
**The pipeline.** Read the export, find and crop the heart, scan the crop twice, and resample the
labels onto each reconstruction. Three facts carry the notebook:

1. The reconstruction grid is **always centred on the isocentre** (`ReconOptions` has no
   off-centre field). To scan a sub-region, crop the **phantom** so the region sits there.
2. `BS.phantom_to_world_affine(phantom)` and `BS.recon_to_world_affine(geom, matrix_size)` are the
   two grids' 4 × 4 voxel → world (cm) matrices.
3. `BS.resample_to_recon(phantom, geom, matrix_size; method = :nearest | :linear)` carries the
   ground truth onto the reconstruction grid.
"""

# ╔═╡ a7131b46-c595-4264-9b65-4bf37ef4848c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 360))
    nodes = Dict(
        :xcat => (0.0, 2.0, "XCAT Export", "0.2 mm Labels"),
        :read => (4.0, 2.0, "Read at 0.4 mm", "Nearest, Step 2"),
        :heart => (8.0, 2.0, "Find the Heart", "Labels by Name"),
        :crop => (12.0, 2.0, "Crop + 1 cm", "Centred on the Isocentre"),
        :axial => (16.0, 3.3, "Scan A: Axial", "BHC → FDK → HU", COLORS.cyan),
        :helical => (16.0, 0.7, "Scan B: Helical", "BHC → WFBP → HU", COLORS.cyan),
        :resample => (
            20.0,
            2.0,
            "resample_to_recon",
            "Labels on the HU Image",
            COLORS.green,
        ),
    )
    edges = [
        :xcat => :read,
        :read => :heart,
        :heart => :crop,
        :crop => :axial,
        :crop => :helical,
        :axial => :resample,
        :helical => :resample,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.9, 21.9, 0.0, 4.0), w = 3.4, h = 1.2)
    fig
end

# ╔═╡ 05000001-0000-4000-8000-000000000020
md"""
## Notebook Setup
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 05000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 05000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 05000001-0000-4000-8000-000000000004
using Unitful: @u_str

# ╔═╡ 05000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 05000001-0000-4000-8000-000000000031
# ╠═╡ show_logs = false
import CairoMakie as Mke

# ╔═╡ 05000001-0000-4000-8000-000000000060
import PlutoUI

# ╔═╡ 05000001-0000-4000-8000-000000000040
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 05000001-0000-4000-8000-000000000050
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ 05000001-0000-4000-8000-000000000070
PlutoUI.TableOfContents()

# ╔═╡ 9ffb859c-b946-4d36-b221-0c8e6c827b12
md"""
### Figure Style

BasisSimulator's colours and type, and the drawing helpers every figure uses (the same cells in
every example notebook), then this notebook's own: a sketch of the chest and a voxel grid.
"""

# ╔═╡ 1da9cf11-165f-4c4d-a06e-4d2533dc40bb
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

# ╔═╡ f0f47d18-a4a7-4845-a8d0-6a5fae3a3d02
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

# ╔═╡ d77fcdce-0d62-4555-849a-523f39c191c4
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

# ╔═╡ 0f569ee4-d4d2-4c27-9f52-aa48118ad858
"""
A simplified axial chest, centred at `c`, `s` units per cm (about 34 × 23 cm across): the body,
the two lungs, the heart (to the patient's left, the viewer's right), the sternum and the spine.
`dark` draws it in the locator band's colours.
"""
function chest_sketch!(ax; c = (0.0, 0.0), s = 1.0, dark = false)
    p(x, y) = Mke.Point2f(c[1] + s * x, c[2] + s * y)
    oval(cx, cy, rx, ry) = [
        p(cx + rx * cos(t), cy + ry * sin(t))
        for t in range(0, 2π; length = 90)
    ]
    body = dark ? Mke.RGBAf(0.85, 0.74, 0.62, 0.45) : Mke.RGBAf(0.88, 0.8, 0.7, 1.0)
    lung = dark ? COLORS.night : COLORS.paper
    bone = dark ? COLORS.night_ink : Mke.RGBf(0.99, 0.97, 0.93)
    edge = dark ? COLORS.night_muted : COLORS.muted

    Mke.poly!(ax, oval(0, 0, 17, 11.5); color = body, strokecolor = edge, strokewidth = 1.5)
    for side in (-1, 1)
        Mke.poly!(ax, oval(side * 9.0, 0.6, 5.8, 8.4); color = lung)
    end
    Mke.poly!(
        ax,
        oval(2.0, 1.2, 5.0, 4.2);
        color = (COLORS.red, dark ? 0.85 : 0.75),
        strokecolor = edge,
        strokewidth = 1,
    )
    Mke.poly!(ax, oval(0, 9.7, 2.4, 0.9); color = bone, strokecolor = edge, strokewidth = 1)
    Mke.poly!(
        ax,
        oval(0, -8.0, 2.2, 2.0);
        color = bone,
        strokecolor = edge,
        strokewidth = 1,
    )
    return ax
end

# ╔═╡ 809f415d-0a41-4856-ab1f-309db255eac7
"""A regular grid of `n = (nx, ny)` cells of `pitch`, centred at `c`, drawn in `color`."""
function grid!(ax, c, n, pitch; color = COLORS.violet, linewidth = 1.2)
    x0 = c[1] - n[1] * pitch / 2
    y0 = c[2] - n[2] * pitch / 2
    for i in 0:n[1]
        x = x0 + i * pitch
        Mke.lines!(ax, [x, x], [y0, y0 + n[2] * pitch]; color, linewidth)
    end
    for j in 0:n[2]
        y = y0 + j * pitch
        Mke.lines!(ax, [x0, x0 + n[1] * pitch], [y, y]; color, linewidth)
    end
    return ax
end

# ╔═╡ 05000002-0000-4000-8000-000000000000
md"""
## 1. Load the XCAT Phantom
"""

# ╔═╡ 05000002-0000-4000-8000-000000000001
md"""
### Locate the Export

An **XCAT 3.0 chest export** (adult male, 50th percentile, with coronary plaque): one 8-bit
activity volume, `*_act_1.raw`, its dimensions in its file name, beside a
`Material_Spreadsheets/` folder with one material table per contrast state. The data is licensed
and not in the repository: the notebook looks in `BASISSIM_XCAT_DIR` (or
`docs/notebooks/data/xcat/`), and without it every compute cell skips and says so.
"""

# ╔═╡ 05000002-0000-4000-8000-000000000010
const XCAT_DIR = get(ENV, "BASISSIM_XCAT_DIR", joinpath(@__DIR__, "data", "xcat"));

# ╔═╡ 05000002-0000-4000-8000-000000000014
const CONTRAST_STATE = "high_contrast"; # the sheets: non_contrast, low_contrast, high_contrast

# ╔═╡ 05000002-0000-4000-8000-000000000011
"""
    find_xcat_export(dir, state) -> NamedTuple or String

The one `*act_1.raw` in `dir`, its dimensions read off its name (`…_1600x1400x867_…`), its
anatomy (`vmale_50`) and the material sheet of contrast `state`, or a String saying what is
missing.
"""
function find_xcat_export(dir, state)
    isdir(dir) || return "no XCAT directory (set `BASISSIM_XCAT_DIR`)"
    raws = [f for f in readdir(dir) if endswith(f, "act_1.raw")]
    length(raws) ==
        1 || return "expected one *act_1.raw in the XCAT directory, found $(length(raws))"
    name = only(raws)
    m = match(r"(\d{3,4})[x_](\d{3,4})[x_](\d{3,4})", name)
    a = match(r"^(v(?:fe)?male_\d+)", name)
    (
        m === nothing || a === nothing
    ) && return "cannot read the dimensions or anatomy from $(name)"
    dims = Tuple(parse.(Int, m.captures))
    raw = joinpath(dir, name)
    filesize(raw) == prod(dims) || return "$(name) is not $(join(dims, " × ")) bytes"
    sheet_name = "$(a.captures[1])_materials_heart_$(state).xlsx"
    sheet = joinpath(dir, "Material_Spreadsheets", sheet_name)
    isfile(sheet) || return "no Material_Spreadsheets/$(sheet_name)"
    return (; raw, name, dims, anatomy = a.captures[1], sheet, sheet_name)
end

# ╔═╡ 05000002-0000-4000-8000-000000000012
xcat = find_xcat_export(XCAT_DIR, CONTRAST_STATE);

# ╔═╡ 05000002-0000-4000-8000-000000000015
const HAS_XCAT = xcat isa NamedTuple;

# ╔═╡ 05000002-0000-4000-8000-000000000013
HAS_XCAT ?
    Markdown.parse(
        """
    **XCAT export:** `$(xcat.name)` ($(join(xcat.dims, " × ")) voxels,
    $(round(prod(xcat.dims) / 1024^3; digits = 2)) GiB) · materials: `$(xcat.sheet_name)`
    """,
    ) :
    Markdown.parse(
        """
    !!! warning "XCAT export not found: compute cells skipped"
        $(xcat). Set `BASISSIM_XCAT_DIR` to a folder holding one `*act_1.raw` and its
        `Material_Spreadsheets/`, or place them in `docs/notebooks/data/xcat/`, and re-run.
    """,
    )

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
### Read the Mask at 0.4 mm

The export is 0.2 mm isotropic, label 0 is air, and the file runs anterior → posterior along its
second axis. Reversing that axis makes `y` point to the patient's front, so every image drawn with
`y` up has the sternum at the top and the patient's left on the viewer's right. A
label-preserving nearest-neighbour step of 2 then reads it at **0.4 mm**: still about as fine as
the reconstruction pixels below.
"""

# ╔═╡ 05000003-0000-4000-8000-000000000010
"""The activity volume, `y` towards anterior (the file runs posterior along axis 2)."""
read_xcat_mask(raw, dims) = reverse(read!(raw, Array{UInt8}(undef, dims...)); dims = 2)

# ╔═╡ 05000003-0000-4000-8000-000000000011
"""Nearest-neighbour 3D downsample by an integer factor: keeps labels intact."""
function downsample_labeled(phantom::AbstractArray{T, 3}, factor::Int) where {T}
    factor == 1 && return phantom
    nx, ny, nz = size(phantom) .÷ factor
    out = similar(phantom, (nx, ny, nz))
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        out[i, j, k] = phantom[
            (i - 1) * factor + factor ÷ 2 + 1,
            (j - 1) * factor + factor ÷ 2 + 1,
            (k - 1) * factor + factor ÷ 2 + 1,
        ]
    end
    return out
end

# ╔═╡ 05000003-0000-4000-8000-000000000020
const DOWNSAMPLE_FACTOR = 2;

# ╔═╡ 05000003-0000-4000-8000-000000000021
const VOXEL_SIZE_CM = ntuple(_ -> 0.02 * DOWNSAMPLE_FACTOR, 3); # 0.2 mm export × factor

# ╔═╡ 05000003-0000-4000-8000-000000000030
phantom_full_uhr = HAS_XCAT ?
    downsample_labeled(read_xcat_mask(xcat.raw, xcat.dims), DOWNSAMPLE_FACTOR) :
    nothing;

# ╔═╡ 05000003-0000-4000-8000-000000000040
if phantom_full_uhr === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let (nx, ny, nz) = size(phantom_full_uhr)
        Markdown.parse(
            """
        **Phantom read:** $(nx) × $(ny) × $(nz) voxels of $(round(VOXEL_SIZE_CM[1] * 10; digits = 2)) mm
        ($(round(sizeof(phantom_full_uhr) / 1024^2; digits = 1)) MiB as UInt8),
        $(join(round.((nx, ny, nz) .* VOXEL_SIZE_CM; digits = 1), " × ")) cm,
        $(length(unique(phantom_full_uhr))) labels present.
        """,
        )
    end
end

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
### Materials from the XCAT Sheet

Each sheet row is one organ: a name, its elemental mass fractions (one column per atomic number),
its density and its label (`Organ ID`). Every row becomes an `XrayAttenuation.Material` keyed by
its label; a label the sheet does not name stays **air**, as `Phantom` assigns it. The
`high_contrast` sheet puts iodinated blood in the cardiac chambers, the aorta and the coronaries.
The table is read before cropping because the organ **names** are how the heart is found.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000002
import XLSX

# ╔═╡ 05000004-0000-4000-8000-000000000004
const _ATOMIC_MASSES = Dict(
    1 => 1.008,
    6 => 12.011,
    7 => 14.007,
    8 => 15.999,
    11 => 22.99,
    12 => 24.305,
    15 => 30.974,
    16 => 32.06,
    17 => 35.45,
    19 => 39.098,
    20 => 40.078,
    26 => 55.845,
    53 => 126.904,
);

# ╔═╡ 05000004-0000-4000-8000-000000000005
const _I_VALUES_EV = Dict(
    1 => 19.2,
    6 => 81.0,
    7 => 82.0,
    8 => 95.0,
    11 => 149.0,
    12 => 156.0,
    15 => 173.0,
    16 => 180.0,
    17 => 174.0,
    19 => 190.0,
    20 => 191.0,
    26 => 286.0,
    53 => 491.0,
);

# ╔═╡ 05000004-0000-4000-8000-000000000006
"""⟨Z/A⟩ of a composition (mass fractions)."""
compute_ZA_ratio(comp::Dict{Int, Float64}) = sum(
    w * Z / _ATOMIC_MASSES[Z]
    for (Z, w) in comp
) /
    sum(values(comp))

# ╔═╡ 05000004-0000-4000-8000-000000000007
"""Bragg additivity for the mean excitation energy of a composition."""
function compute_mean_excitation_energy(comp::Dict{Int, Float64})
    num = sum(w * Z / _ATOMIC_MASSES[Z] * log(_I_VALUES_EV[Z]) for (Z, w) in comp)
    den = sum(w * Z / _ATOMIC_MASSES[Z] for (Z, w) in comp)
    return exp(num / den) * u"eV"
end

# ╔═╡ 05000004-0000-4000-8000-000000000008
"""
    load_xcat_materials(sheet) -> Dict{Int, XA.Material}

`Name | <Z> … | Density | Organ ID` → one material per label.
"""
function load_xcat_materials(sheet)
    wb = XLSX.readxlsx(sheet)
    data = wb[first(XLSX.sheetnames(wb))][:]
    ncol = size(data, 2)
    (string(data[1, ncol - 1]) == "Density" && string(data[1, ncol]) == "Organ ID") ||
        error("unexpected sheet layout: $(join(string.(data[1, :]), " | "))")
    zcols = [(c, parse(Int, string(data[1, c]))) for c in 2:(ncol - 2)]
    out = Dict{Int, BS.XA.Material}()
    for r in 2:size(data, 1)
        data[r, ncol] isa Number || continue
        comp = Dict{Int, Float64}(
            Z => Float64(data[r, c])
            for (c, Z) in zcols
            if data[r, c] isa Number && data[r, c] > 0
        )
        isapprox(sum(values(comp)), 1; atol = 1.0e-3) ||
            error("mass fractions of $(data[r, 1]) do not sum to 1")
        out[Int(data[r, ncol])] = BS.XA.Material(
            string(data[r, 1]),
            compute_ZA_ratio(comp),
            compute_mean_excitation_energy(comp),
            Float64(data[r, ncol - 1]) * u"g/cm^3",
            comp,
        )
    end
    return out
end

# ╔═╡ 05000004-0000-4000-8000-000000000010
materials_full = HAS_XCAT ? load_xcat_materials(xcat.sheet) : nothing;

# ╔═╡ 05000004-0000-4000-8000-000000000012
# one colour per label ID in every map
const LABEL_RANGE = (0, materials_full === nothing ? 1 : maximum(keys(materials_full)));

# ╔═╡ 05000004-0000-4000-8000-000000000011
if materials_full === nothing || phantom_full_uhr === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let present = sort(Int.(unique(phantom_full_uhr)))
        unnamed = [l for l in present if l != 0 && !haskey(materials_full, l)]
        Markdown.parse(
            """
        **$(length(materials_full)) materials** in `$(xcat.sheet_name)`; $(length(present)) labels
        present in the volume; labels present but not named by the sheet (read as air):
        $(isempty(unnamed) ? "none" : join(unnamed, ", ")).
        """,
        )
    end
end

# ╔═╡ 05000005-0000-4000-8000-000000000001
md"""
## 2. Find and Crop the Heart

The heart is found by **name**: the myocardium (`myo…`) and blood pools (`bldpl…`) of the four
chambers, the pericardium, the coronary arteries and veins with their walls, and the plaque
components (`rca…`, `lcx…`, `lad…`). The aorta is left out: it runs the length of the chest and
would stretch the box to the whole export. The bounding box of those labels, padded by 1 cm, is
the region scanned. (For a phantom named differently, list the label IDs instead.)
"""

# ╔═╡ 05000005-0000-4000-8000-000000000010
heart_label_ids = materials_full === nothing ?
    nothing :
    let
        pattern = r"^(myo|bldpl)|pericardium|coronary|^(rca|lcx|lad)\d"i
        sort(
            UInt8[UInt8(id) for (id, mat) in materials_full if occursin(pattern, mat.name)]
        )
    end;

# ╔═╡ 05000005-0000-4000-8000-000000000011
heart_label_ids === nothing ?
    md"" :
    Markdown.parse(
        "**$(length(heart_label_ids)) cardiac labels:** " *
            join(
                ["$(Int(id)) `$(materials_full[Int(id)].name)`" for id in heart_label_ids],
                " · ",
            ),
    )

# ╔═╡ 05000005-0000-4000-8000-000000000020
heart_bbox = (phantom_full_uhr === nothing || heart_label_ids === nothing) ?
    nothing :
    let
        is_heart = falses(256)
        for id in heart_label_ids
            is_heart[Int(id) + 1] = true
        end
        nx, ny, nz = size(phantom_full_uhr)
        lo, hi = [nx + 1, ny + 1, nz + 1], [0, 0, 0]
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            if is_heart[Int(phantom_full_uhr[i, j, k]) + 1]
                lo .= min.(lo, (i, j, k))
                hi .= max.(hi, (i, j, k))
            end
        end
        hi[1] == 0 && error("no cardiac voxels in the phantom")
        pad = round.(Int, 1.0 ./ VOXEL_SIZE_CM) # 1 cm
        lo .= max.(1, lo .- pad)
        hi .= min.((nx, ny, nz), hi .+ pad)
        (i_lo = lo[1], i_hi = hi[1], j_lo = lo[2], j_hi = hi[2], k_lo = lo[3], k_hi = hi[3])
    end;

# ╔═╡ 05000006-0000-4000-8000-000000000001
md"""
### Crop the Phantom

An index operation on the mask. The cropped block is all `simulate!` sees, so nothing outside it
is ray-traced or held in device memory. It also changes the object: the rest of the chest is no
longer in the beam, so this is a scan of the cardiac block, not of a chest with a small display
field. (For that, keep the whole phantom and set a small `ReconOptions(fov_cm = …)`.)
"""

# ╔═╡ 05000006-0000-4000-8000-000000000010
phantom_cropped = heart_bbox === nothing ?
    nothing :
    let b = heart_bbox
        phantom_full_uhr[b.i_lo:b.i_hi, b.j_lo:b.j_hi, b.k_lo:b.k_hi]
    end;

# ╔═╡ 05000006-0000-4000-8000-000000000020
phantom_cropped === nothing ?
    md"" :
    let
        n_full, n_crop = length(phantom_full_uhr), length(phantom_cropped)
        Markdown.parse(
            """
    **Crop:** $(join(size(phantom_cropped), " × ")) voxels,
    $(join(round.(size(phantom_cropped) .* VOXEL_SIZE_CM; digits = 1), " × ")) cm ·
    $(round(100 * n_crop / n_full; digits = 1)) % of the $(round(n_full / 1.0e6; digits = 1)) M-voxel phantom
    """,
        )
    end

# ╔═╡ 05000007-0000-4000-8000-000000000010
if phantom_cropped === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let b = heart_bbox
        Mke.set_theme!(THEME)
        z_full = (b.k_lo + b.k_hi) ÷ 2
        z_crop = z_full - b.k_lo + 1
        vmm = round(VOXEL_SIZE_CM[1] * 10; digits = 2)
        fig = Mke.Figure(size = (1500, 1240))

        # the physical picture: the box in the chest, then the box alone at the isocentre
        ax = canvas(fig[1, 1], (-18.5, 18.5, -12.5, 12.5))
        ax.title = "Cut a Box Around the Heart"
        chest_sketch!(ax)
        Mke.poly!(
            ax,
            rounded_rect((2.2, 1.4), 15.0, 12.6, 0.3);
            color = :transparent,
            strokecolor = COLORS.ink,
            strokewidth = 3,
            linestyle = :dash,
        )
        ax = canvas(fig[1, 2], (-7.2, 7.2, -7.2, 7.2))
        ax.title = "Only the Box Is Scanned, Centred"
        gantry!(ax; labels = ("X-Ray Tube",), object = false, dark = false)
        Mke.poly!(
            ax,
            rounded_rect((0.0, 0.0), 2.6, 2.3, 0.1);
            color = (COLORS.red, 0.75),
            strokecolor = COLORS.ink,
            strokewidth = 1.5,
        )
        Mke.scatter!(
            ax,
            [Mke.Point2f(0, 0)];
            marker = :cross,
            markersize = 22,
            color = COLORS.ink,
        )
        Mke.text!(
            ax,
            0.0,
            -1.4;
            text = "Isocentre",
            align = (:center, :top),
            fontsize = 16,
            color = COLORS.ink,
        )

        # the data: the label maps of the full slice and of the crop
        ax = Mke.Axis(
            fig[2, 1];
            title = "Full Phantom, Slice $(z_full)",
            subtitle = "$(size(phantom_full_uhr, 1)) × $(size(phantom_full_uhr, 2)) Voxels of $(vmm) mm · Crop Box Dashed",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(
            ax,
            Float32.(phantom_full_uhr[:, :, z_full]);
            colormap = :tab20,
            colorrange = LABEL_RANGE,
        )
        Mke.lines!(
            ax,
            Mke.Point2f[
                (b.i_lo, b.j_lo),
                (b.i_hi, b.j_lo),
                (b.i_hi, b.j_hi),
                (b.i_lo, b.j_hi),
                (b.i_lo, b.j_lo),
            ];
            color = COLORS.ink,
            linewidth = 3.5,
            linestyle = :dash,
        )
        Mke.hidedecorations!(ax)
        ax = Mke.Axis(
            fig[2, 2];
            title = "Cropped Cardiac Block, Slice $(z_crop)",
            subtitle = "$(size(phantom_cropped, 1)) × $(size(phantom_cropped, 2)) Voxels of $(vmm) mm: Same Voxels, Smaller Extent",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(
            ax,
            Float32.(phantom_cropped[:, :, z_crop]);
            colormap = :tab20,
            colorrange = LABEL_RANGE,
        )
        Mke.hidedecorations!(ax)
        Mke.Label(
            fig[3, 1:2],
            "Label Maps (Colour = Label ID) · Anterior at the Top, the Patient's Left on the Right";
            fontsize = 17,
            color = COLORS.muted,
            tellwidth = false,
        )
        Mke.rowsize!(fig.layout, 1, Mke.Relative(0.36))
        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_grid_crop.png"),
            fig;
            px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ a9ef1c11-fd34-49f5-8468-b7c28abacc2d
md"""
## 3. Two Grids, Two Affines

### The Phantom and Its Affine

Without an `origin`, `Phantom` centres the block on the isocentre
(`origin = -extent/2 + voxel/2`), exactly where the centred reconstruction grid will look. Two
phantoms are built from the same crop: `phantom_cpu` keeps the XCAT labels, for the resampling,
and the simulation phantom is `compact_materials(phantom_cpu)` on the device, the same
attenuation with the labels renumbered densely (the sheet's IDs are sparse).
"""

# ╔═╡ 05000008-0000-4000-8000-000000000011
phantom_cpu = phantom_cropped === nothing ?
    nothing :
    BS.Phantom(phantom_cropped, materials_full, VOXEL_SIZE_CM);

# ╔═╡ 05000008-0000-4000-8000-000000000010
phantom = phantom_cpu === nothing ?
    nothing :
    let c = BS.compact_materials(phantom_cpu)
        BS.Phantom(to_gpu(c.mask), c.materials, c.voxel_size, c.origin, c.extent)
    end;

# ╔═╡ 05000009-0000-4000-8000-000000000001
md"""
`phantom_to_world_affine` maps a 0-indexed phantom voxel ``(i, j, k)`` to world coordinates in
cm, with ``v`` the voxel size and ``o`` the world position of voxel ``(0, 0, 0)``:

```math
\begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix} =
\begin{pmatrix} v_x & 0 & 0 & o_x \\ 0 & v_y & 0 & o_y \\ 0 & 0 & v_z & o_z \\ 0 & 0 & 0 & 1 \end{pmatrix}
\begin{pmatrix} i \\ j \\ k \\ 1 \end{pmatrix}
```
"""

# ╔═╡ 05000009-0000-4000-8000-000000000010
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let A = BS.phantom_to_world_affine(phantom_cpu)
        rows = ["| " * join(round.(A[i, :]; digits = 4), " | ") * " |" for i in 1:4]
        centre = phantom_cpu.origin .+
            phantom_cpu.extent ./ 2 .-
            phantom_cpu.voxel_size ./ 2
        Markdown.parse(
            """
        **`A_phantom = phantom_to_world_affine(phantom_cpu)`** (cm)

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        Voxel $(round.(phantom_cpu.voxel_size .* 10; digits = 2)) mm, extent
        $(round.(phantom_cpu.extent; digits = 3)) cm, origin $(round.(phantom_cpu.origin; digits = 3)) cm;
        the centre of the block is $(round.(centre; digits = 6)) cm, the isocentre.
        """,
        )
    end
end

# ╔═╡ 0500000a-0000-4000-8000-000000000001
md"""
### Scanner, Protocol and a Tight Reconstruction Grid

The GE Revolution Apex Elite of notebook 01, `REVOLUTION_APEX` (256 × 0.625 mm rows, a curved
Gemstone detector, the large-body bowtie). The grid mapping does not depend on the detector, so
any scanner would do. The protocol is an axial cardiac CTA: 120 kVp, 250 mA, a 1 s rotation of
500 views and 5 mm of collimation, each view integrated while the gantry turns
(`view_samples = 5`, notebook 01).
"""

# ╔═╡ 0500000a-0000-4000-8000-000000000010
scanner = BS.REVOLUTION_APEX

# ╔═╡ 99ccd184-5dcd-5bc5-9ca2-586e5ff69f7c
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

# ╔═╡ 0500000a-0000-4000-8000-000000000030
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 0500000a-0000-4000-8000-000000000040
sim_opts = BS.SimOptions(seed = 1234, projector = :dd_fast, view_samples = 5);

# ╔═╡ 0500000b-0000-4000-8000-000000000001
md"""
The reconstruction grid is a **14 cm × 14 cm** field on 384 × 384 pixels (0.365 mm), eight 0.625
mm slices over the 5 mm beam. It is centred on the isocentre, where the block sits, and it is
deliberately smaller than the block in x: a reconstruction grid need not contain the object.
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000010
recon_opts = BS.ReconOptions(
    matrix_size = (384, 384, round(Int, protocol.collimation_mm / 0.625)),
    fov_cm = 14.0,
    z_cm = protocol.collimation_mm / 10,
);

# ╔═╡ 0500000b-0000-4000-8000-000000000020
md"""
### The Reconstruction Grid's Affine

The reconstruction grid can be inspected before anything is simulated: `CTGeometry` is built from
the scanner, the protocol and the grid, as the workspace builds it. Its affine has the phantom's
form, with the pixel size `fov / matrix_size` and the centred origin `-fov/2 + pixel/2`.
"""

# ╔═╡ 0500000b-0000-4000-8000-000000000030
geom_inspect = BS.CTGeometry(
    scanner;
    n_angles = protocol.views,
    fov_cm = recon_opts.fov_cm,
    z_cm = recon_opts.z_cm,
    collimation_mm = protocol.collimation_mm,
);

# ╔═╡ 0500000b-0000-4000-8000-000000000040
let A = BS.recon_to_world_affine(geom_inspect, recon_opts.matrix_size)
    rows = ["| " * join(round.(A[i, :]; digits = 4), " | ") * " |" for i in 1:4]
    n = recon_opts.matrix_size
    fov = geom_inspect.fov
    Markdown.parse(
        """
    **`A_recon = recon_to_world_affine(geom, matrix_size)`** (cm)

    | col 1 | col 2 | col 3 | col 4 |
    |---|---|---|---|
    $(join(rows, "\n"))

    Matrix $(join(n, " × ")), pixel $(join(round.(fov ./ n .* 10; digits = 3), " × ")) mm, field
    $(join(round.(fov; digits = 3), " × ")) cm, origin
    $(join(round.(-1 .* fov ./ 2 .+ fov ./ n ./ 2; digits = 4), ", ")) cm: centred on the isocentre.
    """,
    )
end

# ╔═╡ 0500000c-0000-4000-8000-000000000010
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let np = size(phantom_cpu.mask), nr = recon_opts.matrix_size, fov = geom_inspect.fov
        vp = round.(phantom_cpu.voxel_size .* 10; digits = 3)
        vr = round.(fov ./ nr .* 10; digits = 3)
        Markdown.parse(
            """
        The two grids share world coordinates but little else:

        | | phantom (cropped) | reconstruction |
        |---|---|---|
        | voxels | $(join(np, " × ")) | $(join(nr, " × ")) |
        | voxel size (mm) | $(join(vp, " × ")) | $(join(vr, " × ")) |
        | extent (cm) | $(join(round.(phantom_cpu.extent; digits = 2), " × ")) | $(join(round.(fov; digits = 2), " × ")) |
        | origin (cm) | $(join(round.(phantom_cpu.origin; digits = 3), ", ")) | $(join(round.(-1 .* fov ./ 2 .+ fov ./ nr ./ 2; digits = 3), ", ")) |
        """,
        )
    end
end

# ╔═╡ 230ebca6-6372-42b6-b300-18e11c8f001c
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        Mke.set_theme!(THEME)
        vp, op, ext = phantom_cpu.voxel_size, phantom_cpu.origin, phantom_cpu.extent
        n = recon_opts.matrix_size
        fov = geom_inspect.fov
        vr = fov ./ n
        or = -1 .* fov ./ 2 .+ vr ./ 2
        fig = Mke.Figure(size = (1500, 760))

        # the whole field, to scale (cm)
        ax = Mke.Axis(
            fig[1, 1];
            title = "The Whole Field, to Scale",
            subtitle = "Phantom Block and Reconstruction Field, World cm",
            xlabel = "x (cm)",
            ylabel = "y (cm)",
            aspect = Mke.DataAspect(),
        )
        lo = op .- vp ./ 2
        Mke.poly!(
            ax,
            Mke.Rect2f(lo[1], lo[2], ext[1], ext[2]);
            color = (COLORS.violet, 0.12),
            strokecolor = COLORS.violet,
            strokewidth = 3,
        )
        Mke.poly!(
            ax,
            Mke.Rect2f(-fov[1] / 2, -fov[2] / 2, fov[1], fov[2]);
            color = :transparent,
            strokecolor = COLORS.cyan,
            strokewidth = 3,
        )
        Mke.lines!(
            ax,
            Mke.Circle(Mke.Point2f(0, 0), Float32(fov[1] / 2));
            color = COLORS.cyan,
            linestyle = :dash,
            linewidth = 2,
        )
        Mke.scatter!(
            ax,
            [Mke.Point2f(op[1], op[2]), Mke.Point2f(or[1], or[2])];
            color = [COLORS.violet, COLORS.cyan],
            markersize = 16,
        )
        Mke.scatter!(
            ax,
            [Mke.Point2f(0, 0)];
            marker = :cross,
            markersize = 26,
            color = COLORS.beam,
        )
        Mke.text!(
            ax,
            op[1],
            min(op[2], or[2]) - 1.25;
            text = "● Phantom Voxel (0, 0)",
            align = (:left, :top),
            fontsize = 16,
            color = COLORS.violet,
        )
        Mke.text!(
            ax,
            op[1],
            min(op[2], or[2]) - 0.3;
            text = "● Recon Pixel (0, 0)",
            align = (:left, :top),
            fontsize = 16,
            color = COLORS.cyan,
        )
        Mke.text!(
            ax,
            lo[1] + 0.3,
            lo[2] + ext[2] - 0.3;
            text = "Phantom Block, $(round(ext[1]; digits = 1)) × $(round(ext[2]; digits = 1)) cm",
            align = (:left, :top),
            fontsize = 16,
            color = COLORS.violet,
        )
        Mke.text!(
            ax,
            fov[1] / 2,
            fov[2] / 2 + 0.25;
            text = "Recon Field, $(round(fov[1]; digits = 1)) cm",
            align = (:right, :bottom),
            fontsize = 16,
            color = COLORS.cyan,
        )
        Mke.text!(
            ax,
            0.3,
            0.3;
            text = "Isocentre",
            align = (:left, :bottom),
            fontsize = 16,
            color = COLORS.ink,
        )
        half = max(ext[1], ext[2], fov[1]) / 2 + 2.6
        Mke.limits!(ax, -half, half, -half, half)

        # near the isocentre, to scale (mm): the two grids do not line up
        ax = Mke.Axis(
            fig[1, 2];
            title = "Near the Isocentre, to Scale",
            subtitle = "Voxel Edges (Lines) and Centres (Dots)",
            xlabel = "x (mm)",
            ylabel = "y (mm)",
            aspect = Mke.DataAspect(),
        )
        w = 1.25 # mm, half the window
        edges(o, v, m) = [
            10 * (o - v / 2 + v * i)
            for i in 0:m
            if abs(10 * (o - v / 2 + v * i)) <= w
        ]
        centres(o, v, m) = [
            10 * (o + v * i)
            for i in 0:(m - 1)
            if abs(10 * (o + v * i)) <= w
        ]
        for (o, v, m, color, ls) in (
                (op, vp, size(phantom_cpu.mask), COLORS.violet, :solid),
                (or, vr, n, COLORS.cyan, :dash),
            )
            Mke.vlines!(ax, edges(o[1], v[1], m[1]); color, linewidth = 2.5, linestyle = ls)
            Mke.hlines!(ax, edges(o[2], v[2], m[2]); color, linewidth = 2.5, linestyle = ls)
            Mke.scatter!(
                ax,
                [
                    Mke.Point2f(x, y)
                    for x in centres(o[1], v[1], m[1]), y in centres(o[2], v[2], m[2])
                ][:];
                color,
                markersize = 11,
            )
        end
        Mke.scatter!(
            ax,
            [Mke.Point2f(0, 0)];
            marker = :cross,
            markersize = 26,
            color = COLORS.beam,
        )
        Mke.limits!(ax, -w, w, -w, w)
        Mke.Legend(
            fig[2, 1:2],
            [
                Mke.LineElement(color = COLORS.violet, linewidth = 3),
                Mke.LineElement(color = COLORS.cyan, linewidth = 3, linestyle = :dash),
                Mke.MarkerElement(color = COLORS.beam, marker = :cross, markersize = 18),
            ],
            [
                "Phantom Voxels ($(round(10 * vp[1]; digits = 3)) mm)",
                "Recon Pixels ($(round(10 * vr[1]; digits = 3)) mm)",
                "Isocentre",
            ];
            orientation = :horizontal,
            tellheight = true,
        )
        fig
    end
end

# ╔═╡ 0500000d-0000-4000-8000-000000000001
md"""
## 4. Scan A: Axial

The chain of notebook 01: `create_workspace` → `simulate!`, then the knobless water
beam-hardening correction → FDK → HU with the correction's own μ_water.
"""

# ╔═╡ 0500000d-0000-4000-8000-000000000010
sim = phantom === nothing ?
    nothing :
    let
        ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
        t = @elapsed result = BS.simulate!(ws, phantom, protocol, sim_opts)
        out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose, t = t)
        ws = nothing
        GC.gc(true)
        out
    end;

# ╔═╡ 0500000d-0000-4000-8000-000000000020
bhc = sim === nothing ?
    nothing :
    BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = sim.geom);

# ╔═╡ 0500000d-0000-4000-8000-000000000030
recon_HU = sim === nothing ?
    nothing :
    let
        sino = BS.apply_bhc_water(to_gpu(sim.sino), bhc)
        ws = BS.create_fdk_recon_workspace(
            sino,
            sim.geom,
            recon_opts.matrix_size;
            filter = :standard,
        )
        μ = Array(BS.reconstruct!(ws, sino, sim.geom))
        ws = nothing
        sino = nothing
        GC.gc(true)
        Float32.(BS.to_hounsfield(μ; μ_water = bhc.μ_water_ref))
    end;

# ╔═╡ 0500000d-0000-4000-8000-000000000040
sim === nothing ?
    md"" :
    Markdown.parse(
        """
**Scan A:** $(join(size(sim.sino), " × ")) sinogram (columns × rows × views), simulated in
$(round(sim.t; digits = 1)) s (first call, compilation included) · CTDIvol =
$(round(sim.dose.ctdi_vol_mGy; digits = 2)) mGy · water BHC at
$(round(bhc.reference_energy_keV; digits = 1)) keV
""",
    )

# ╔═╡ 0500000e-0000-4000-8000-000000000001
md"""
## 5. Labels on the Reconstruction

`resample_to_recon` takes each reconstruction pixel's world coordinate from `A_recon`, maps it to
a continuous phantom index through `inv(A_phantom)`, and samples the phantom there:

| `method` | output | use it for |
|---|---|---|
| `:nearest` | `UInt8` labels | ROI extraction, segmentation scoring |
| `:linear` | `Float32` (trilinear) | continuous fields, and the coverage fraction of one binary mask |

Trilinear interpolation of a *multi-label* mask averages label IDs, which means nothing; for a
partial-volume fraction, resample a 0/1 mask of the structure instead.
"""

# ╔═╡ b17dd975-086f-4472-8446-0e62de4c4706
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 560))
    titles = [
        "One Recon Pixel Over the Phantom",
        "Nearest: Copy One Label",
        "Linear on a 0/1 Mask: A Fraction",
    ]
    subtitles = [
        "Phantom Voxels Inside the Heart Shaded",
        "The Closest Phantom Voxel Wins",
        "Bilinear Weights Here, Trilinear in 3-D",
    ]
    in_heart(x, y) = x + 0.8 * y < 3.6 # a boundary through the patch
    p = Mke.Point2f(1.82, 2.3)         # the recon pixel's centre, in phantom voxel units
    for k in 1:3
        ax = canvas(fig[1, k], (-0.3, 4.3, -0.3, 4.3))
        Mke.Label(fig[2, k], titles[k]; fontsize = 20, font = :bold, tellwidth = false)
        Mke.Label(
            fig[3, k],
            subtitles[k];
            fontsize = 16,
            color = COLORS.muted,
            tellwidth = false,
        )
        for i in 0:3, j in 0:3
            Mke.poly!(
                ax,
                rounded_rect((i + 0.5, j + 0.5), 0.94, 0.94, 0.05);
                color = in_heart(i + 0.5, j + 0.5) ? (COLORS.red, 0.5) : COLORS.card,
                strokecolor = COLORS.rule,
                strokewidth = 1,
            )
            Mke.scatter!(
                ax,
                [Mke.Point2f(i + 0.5, j + 0.5)];
                color = COLORS.violet,
                markersize = 7,
            )
        end
        if k == 1
            Mke.poly!(
                ax,
                rounded_rect(p, 0.9, 0.9, 0.02);
                color = (COLORS.cyan, 0.15),
                strokecolor = COLORS.cyan,
                strokewidth = 3.5,
            )
        elseif k == 2
            Mke.poly!(
                ax,
                rounded_rect((1.5, 2.5), 0.98, 0.98, 0.05);
                color = :transparent,
                strokecolor = COLORS.violet,
                strokewidth = 4,
            )
            arrow!(ax, p, (1.56, 2.46); color = COLORS.violet, head = 14)
            Mke.text!(
                ax,
                1.5,
                3.1;
                text = "Label: Heart",
                align = (:center, :bottom),
                fontsize = 16,
                color = COLORS.ink,
            )
        else
            fraction = 0.0
            for (cx, cy) in ((1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5))
                wgt = (1 - abs(p[1] - cx)) * (1 - abs(p[2] - cy))
                fraction += wgt * in_heart(cx, cy)
                Mke.lines!(
                    ax,
                    [p, Mke.Point2f(cx, cy)];
                    color = COLORS.violet,
                    linewidth = 1 + 7wgt,
                )
                Mke.text!(
                    ax,
                    cx + (cx < p[1] ? -0.1 : 0.1),
                    cy + (cy < p[2] ? -0.12 : 0.12);
                    text = string(round(wgt; digits = 2)),
                    fontsize = 16,
                    font = :bold,
                    color = COLORS.ink,
                    align = (cx < p[1] ? :right : :left, cy < p[2] ? :top : :bottom),
                )
            end
            Mke.text!(
                ax,
                2.0,
                4.05;
                text = "Fraction = $(round(fraction; digits = 2))",
                align = (:center, :bottom),
                fontsize = 17,
                font = :bold,
                color = COLORS.ink,
            )
        end
        Mke.scatter!(ax, [p]; marker = :cross, markersize = 24, color = COLORS.cyan)
    end
    Mke.rowgap!(fig.layout, 1, 8)
    Mke.rowgap!(fig.layout, 2, 2)
    fig
end

# ╔═╡ 0500000e-0000-4000-8000-000000000010
gt_resampled_nn = sim === nothing ?
    nothing :
    BS.resample_to_recon(phantom_cpu, sim.geom, recon_opts.matrix_size; method = :nearest);

# ╔═╡ 0500000e-0000-4000-8000-000000000012
"""Fraction of each reconstruction voxel covered by the cardiac labels (trilinear on a 0/1 mask)."""
function cardiac_coverage(mask, geom, matrix_size)
    binary = zeros(UInt8, size(mask))
    for id in heart_label_ids
        binary[mask .== id] .= 0x01
    end
    ph = BS.Phantom(
        binary,
        Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water),
        VOXEL_SIZE_CM,
    )
    return BS.resample_to_recon(ph, geom, matrix_size; method = :linear)
end

# ╔═╡ 0500000e-0000-4000-8000-000000000013
cardiac_coverage_lin = sim === nothing ?
    nothing :
    cardiac_coverage(phantom_cropped, sim.geom, recon_opts.matrix_size);

# ╔═╡ 0500000e-0000-4000-8000-000000000020
sim === nothing ?
    md"" :
    Markdown.parse(
        """
`gt_resampled_nn` is $(eltype(gt_resampled_nn)) $(join(size(gt_resampled_nn), " × ")), `cardiac_coverage_lin`
is $(eltype(cardiac_coverage_lin)) $(join(size(cardiac_coverage_lin), " × ")), `recon_HU` is
$(eltype(recon_HU)) $(join(size(recon_HU), " × ")): index `(i, j, k)` is the same physical voxel in all three.
""",
    )

# ╔═╡ 05000010-0000-4000-8000-000000000001
md"""
### The Overlay

The central slice, all four panels on the reconstruction grid: the HU image, every label
(`:nearest`), the cardiac labels over the image, and the cardiac coverage fraction (`:linear` on
a 0/1 mask) over the image.
"""

# ╔═╡ 05000010-0000-4000-8000-000000000010
if recon_HU === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        Mke.set_theme!(THEME)
        is_cardiac = falses(256)
        for id in heart_label_ids
            is_cardiac[Int(id) + 1] = true
        end
        z = size(recon_HU, 3) ÷ 2 + 1
        hu = recon_HU[:, :, z]
        lab = gt_resampled_nn[:, :, z]
        nn_overlay = [is_cardiac[Int(l) + 1] ? Float32(l) : NaN32 for l in lab]
        cov = cardiac_coverage_lin[:, :, z]
        lin_overlay = [c < 0.05f0 ? NaN32 : Float32(c) for c in cov]

        fig = Mke.Figure(size = (1400, 1340))
        hu_kw = (colormap = :grays, colorrange = (-300, 700))
        square = (aspect = Mke.DataAspect(),)

        ax = Mke.Axis(
            fig[1, 2];
            title = "HU Reconstruction",
            subtitle = "Slice $(z) of $(size(recon_HU, 3)) · $(recon_opts.fov_cm) cm Field, Centred",
            square...,
        )
        hm = Mke.heatmap!(ax, hu; hu_kw...)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[1, 3], hm; label = "HU", width = 16)

        ax = Mke.Axis(
            fig[1, 1];
            title = "Every Label, :nearest",
            subtitle = "$(length(unique(gt_resampled_nn))) Labels on the Reconstruction Grid",
            square...,
        )
        Mke.heatmap!(ax, Float32.(lab); colormap = :tab20, colorrange = LABEL_RANGE)
        Mke.hidedecorations!(ax)

        ax = Mke.Axis(
            fig[2, 1];
            title = "HU + Cardiac Labels",
            subtitle = ":nearest, 60 % Opaque",
            square...,
        )
        Mke.heatmap!(ax, hu; hu_kw...)
        Mke.heatmap!(
            ax,
            nn_overlay;
            colormap = :tab20,
            colorrange = LABEL_RANGE,
            alpha = 0.6,
            nan_color = (:white, 0.0),
        )
        Mke.hidedecorations!(ax)

        ax = Mke.Axis(
            fig[2, 2];
            title = "HU + Cardiac Coverage",
            subtitle = ":linear on a 0/1 Mask, Shown ≥ 0.05",
            square...,
        )
        Mke.heatmap!(ax, hu; hu_kw...)
        hc = Mke.heatmap!(
            ax,
            lin_overlay;
            colormap = :viridis,
            colorrange = (0, 1),
            alpha = 0.7,
            nan_color = (:white, 0.0),
        )
        Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[2, 3], hc; label = "Cardiac Fraction", width = 16)
        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_grid_overlay.png"),
            fig;
            px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 05000018-0000-4000-8000-000000000001
md"""
### Organ ROI Statistics

With the labels on the reconstruction grid, an organ ROI is one comparison,
`recon_HU[gt_resampled_nn .== label]`. Each label with enough pixels on the central slice is
eroded in-plane by one pixel (dropping the partial-volume rim), and its mean HU is compared with
the monoenergetic HU of its sheet material at the BHC reference energy. Soft tissue, fat, lung
and blood land within a few tens of HU; bone reads low (the single-kVp beam-hardening residual
of notebook 01), and thin structures pick up partial volume across the 0.625 mm slice.
"""

# ╔═╡ 7e808dde-5645-4a48-b10e-39e75f1cffd7
roi_stats = recon_HU === nothing ?
    nothing :
    let lab = gt_resampled_nn, k = size(recon_HU, 3) ÷ 2 + 1
        nx, ny = size(lab, 1), size(lab, 2)
        refE, μw = bhc.reference_energy_keV, bhc.μ_water_ref
        # pixels inside the reconstruction circle only
        inside(i, j) = (i - (nx + 1) / 2)^2 + (j - (ny + 1) / 2)^2 <=
            (min(nx, ny) / 2 - 2)^2
        interior(i, j, id) = inside(i, j) &&
            all(lab[i + di, j + dj, k] == id for di in -1:1, dj in -1:1)
        stats = NamedTuple[]
        for id in sort(unique(lab[:, :, k]))
            haskey(materials_full, Int(id)) || continue
            vox = [
                recon_HU[i, j, k]
                for j in 2:(ny - 1), i in 2:(nx - 1)
                if interior(i, j, id)
            ]
            length(vox) < 200 && continue
            mat = materials_full[Int(id)]
            theory = 1000 * (BS.compute_μ_at_energy(mat, refE) - μw) / μw
            push!(
                stats,
                (
                    id = Int(id),
                    name = mat.name,
                    n = length(vox),
                    mean = mean(vox),
                    std = std(vox),
                    theory = theory,
                ),
            )
        end
        (slice = k, energy = refE, rows = stats)
    end;

# ╔═╡ 05000018-0000-4000-8000-000000000002
if roi_stats === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        r1(x) = round(x; digits = 1)
        rows = [
            "| $(s.id) | `$(s.name)` | $(s.n) | $(r1(s.mean)) | $(r1(s.std)) | $(r1(s.theory)) | $(r1(s.mean - s.theory)) |"
            for s in roi_stats.rows
        ]
        Markdown.parse(
            """
        **Scan A, slice $(roi_stats.slice): per-label ROI statistics** (theory = monoenergetic HU at $(r1(roi_stats.energy)) keV)

        | label | material | pixels | mean HU | σ (HU) | theory HU | mean − theory |
        |--:|:--|--:|--:|--:|--:|--:|
        $(join(rows, "\n"))
        """,
        )
    end
end

# ╔═╡ 0500000f-0000-4000-8000-000000000001
md"""
<details><summary><strong>Bring your own interpolator</strong></summary>

For a B-spline, a sinc kernel or a learned upsampler, compose the two affines yourself:
`M = inv(A_phantom) * A_recon` takes a 0-indexed reconstruction voxel `(i, j, k, 1)` to a
**continuous** phantom index, which any interpolator (Interpolations.jl,
ImageTransformations.jl, your own kernel) can sample:

```julia
for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
    p = M * [i, j, k, 1.0] # continuous phantom index
    out[i + 1, j + 1, k + 1] = my_interpolator(phantom.mask, p[1], p[2], p[3])
end
```

The matrix for this scan is below.
</details>
"""

# ╔═╡ 0500000f-0000-4000-8000-000000000010
if phantom_cpu === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let M = inv(BS.phantom_to_world_affine(phantom_cpu)) *
        BS.recon_to_world_affine(geom_inspect, recon_opts.matrix_size)
        rows = ["| " * join(round.(M[i, :]; digits = 4), " | ") * " |" for i in 1:4]
        n = recon_opts.matrix_size
        vc = M * [(n[1] - 1) / 2, (n[2] - 1) / 2, (n[3] - 1) / 2, 1.0]
        pc = (size(phantom_cpu.mask) .- 1) ./ 2
        Markdown.parse(
            """
        **`M = inv(A_phantom) * A_recon`**

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        The centre of the reconstruction grid maps to phantom index $(round.((vc[1], vc[2], vc[3]); digits = 2)),
        the centre of the cropped block ($(round.(pc; digits = 2))).
        """,
        )
    end
end

# ╔═╡ 05000012-0000-4000-8000-000000000000
md"""
## 6. Scan B: Helical

The same phantom and the same three calls, now with a **taller crop** scanned by a **helical**
acquisition and reconstructed into 64 axial slices to scroll through. The helical grid has its
own affine (a longer z extent than Scan A), and `resample_to_recon` lands the labels on every
slice unchanged.

The crop is Scan A's in-plane box, extended 4 cm beyond the cardiac extent at each end in z
(where the phantom allows), so the helix has anatomy to sweep through.
"""

# ╔═╡ 05000013-0000-4000-8000-000000000010
heart_bbox_tall = heart_bbox === nothing ?
    nothing :
    let b = heart_bbox
        extra = round(Int, 4.0 / VOXEL_SIZE_CM[3])
        merge(
            b,
            (
                k_lo = max(1, b.k_lo - extra),
                k_hi = min(size(phantom_full_uhr, 3), b.k_hi + extra),
            ),
        )
    end;

# ╔═╡ 05000013-0000-4000-8000-000000000020
phantom_cropped_tall = heart_bbox_tall === nothing ?
    nothing :
    let b = heart_bbox_tall
        phantom_full_uhr[b.i_lo:b.i_hi, b.j_lo:b.j_hi, b.k_lo:b.k_hi]
    end;

# ╔═╡ 05000013-0000-4000-8000-000000000031
phantom_helical_cpu = phantom_cropped_tall === nothing ?
    nothing :
    BS.Phantom(phantom_cropped_tall, materials_full, VOXEL_SIZE_CM);

# ╔═╡ 05000013-0000-4000-8000-000000000030
phantom_helical = phantom_helical_cpu === nothing ?
    nothing :
    let c = BS.compact_materials(phantom_helical_cpu)
        BS.Phantom(to_gpu(c.mask), c.materials, c.voxel_size, c.origin, c.extent)
    end;

# ╔═╡ 05000013-0000-4000-8000-000000000040
recon_opts_helical = BS.ReconOptions(
    matrix_size = (384, 384, 64), # 64 slices of 0.625 mm
    fov_cm = 14.0,
    z_cm = 4.0,
);

# ╔═╡ 05000012-0000-4000-8000-000000000001
md"""
### The Helical Acquisition

`pitch` and `n_rotations` make the protocol helical: 10 mm of collimation at pitch 1.0 over 8
rotations is 8 cm of table travel, centred on the isocentre. The geometry becomes a z-ramped
trajectory that `:dd_fast` projects unchanged (each sub-view of the view integration advanced
along z by its share of the table feed), and `reconstruct!` recognises the helical geometry and
runs rebinned weighted FBP. The reconstruction grid is still a centred stack of axial slices, so
the affines apply as before.
"""

# ╔═╡ 05000012-0000-4000-8000-000000000005
protocol_helical = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 10.0,
    additional_filters = [("Al", 4.5)],
    pitch = 1.0,
    n_rotations = 8,
);

# ╔═╡ 05000012-0000-4000-8000-000000000010
sim_helical = phantom_helical === nothing ?
    nothing :
    let
        ws = BS.create_workspace(
            scanner,
            protocol_helical,
            sim_opts,
            recon_opts_helical,
            phantom_helical,
        )
        t = @elapsed result = BS.simulate!(ws, phantom_helical, protocol_helical, sim_opts)
        out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose, t = t)
        ws = nothing
        GC.gc(true)
        out
    end;

# ╔═╡ 05000012-0000-4000-8000-000000000020
recon_HU_helical = sim_helical === nothing ?
    nothing :
    let
        bhc_h = BS.calibrate_bhc_water(
            sim_opts,
            protocol_helical;
            scanner,
            geom = sim_helical.geom,
        )
        sino = BS.apply_bhc_water(to_gpu(sim_helical.sino), bhc_h)
        ws = BS.create_fdk_recon_workspace(
            sino,
            sim_helical.geom,
            recon_opts_helical.matrix_size;
            filter = :standard,
        )
        # helical geometry → rebinned WFBP
        μ = Array(BS.reconstruct!(ws, sino, sim_helical.geom))
        ws = nothing
        sino = nothing
        GC.gc(true)
        Float32.(BS.to_hounsfield(μ; μ_water = bhc_h.μ_water_ref))
    end;

# ╔═╡ 05000014-0000-4000-8000-000000000010
if sim_helical === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let A = BS.recon_to_world_affine(sim_helical.geom, recon_opts_helical.matrix_size)
        rows = ["| " * join(round.(A[i, :]; digits = 4), " | ") * " |" for i in 1:4]
        d = sim_helical.dose
        Markdown.parse(
            """
        **Scan B:** $(join(size(sim_helical.sino), " × ")) sinogram, simulated in
        $(round(sim_helical.t; digits = 1)) s · CTDIvol = $(round(d.ctdi_vol_mGy; digits = 2)) mGy,
        DLP = $(round(d.dlp_mGy_cm; digits = 1)) mGy·cm over $(round(d.scan_length_cm; digits = 1)) cm (pitch $(d.pitch))

        **`A_recon` of the helical grid** (cm)

        | col 1 | col 2 | col 3 | col 4 |
        |---|---|---|---|
        $(join(rows, "\n"))

        $(recon_opts_helical.matrix_size[3]) slices over $(recon_opts_helical.z_cm) cm, against Scan A's
        $(recon_opts.matrix_size[3]) over $(recon_opts.z_cm) cm; the same centred origin rule.
        """,
        )
    end
end

# ╔═╡ 05000012-0000-4000-8000-000000000030
gt_helical_nn = sim_helical === nothing ?
    nothing :
    BS.resample_to_recon(
        phantom_helical_cpu,
        sim_helical.geom,
        recon_opts_helical.matrix_size;
        method = :nearest,
    );

# ╔═╡ 05000016-0000-4000-8000-000000000010
cardiac_coverage_lin_helical = sim_helical === nothing ?
    nothing :
    cardiac_coverage(
        phantom_cropped_tall,
        sim_helical.geom,
        recon_opts_helical.matrix_size,
    );

# ╔═╡ 05000015-0000-4000-8000-000000000001
md"""
### Scroll Through z

The drawing on top places the chosen slice in the scan: the taller crop, the tube's helical path
and the 4 cm reconstructed slab. Below, left: the WFBP reconstruction. Middle: the cardiac labels
resampled with `:nearest`, which snaps every pixel to the single closest phantom voxel, so
boundaries stair-step onto the grid. Right: the cardiac coverage from `:linear` on the 0/1 mask, a
fully 3-D partial-volume fraction that also catches pixels straddling a surface in z.

Trilinear sampling at the voxel centre equals the true covered fraction when the two grids are of
similar resolution, as here (0.4 mm phantom, 0.365 × 0.625 mm reconstruction); for a phantom
much finer than the reconstruction, a volume fraction needs box averaging instead.
"""

# ╔═╡ 05000015-0000-4000-8000-000000000010
@bind z_helical PlutoUI.Slider(
    1:recon_opts_helical.matrix_size[3];
    default = recon_opts_helical.matrix_size[3] ÷ 2,
    show_value = true,
)

# ╔═╡ 05000016-0000-4000-8000-000000000020
if sim_helical === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let nz = size(recon_HU_helical, 3)
        Mke.set_theme!(THEME)
        z = clamp(z_helical, 1, nz)
        recon = recon_HU_helical[:, :, z]
        lab = gt_helical_nn[:, :, z]
        cov = cardiac_coverage_lin_helical[:, :, z]
        is_cardiac = falses(256)
        for id in heart_label_ids
            is_cardiac[Int(id) + 1] = true
        end
        nx, ny = size(lab)
        inside(i, j) = (i - (nx + 1) / 2)^2 + (j - (ny + 1) / 2)^2 <= (min(nx, ny) / 2)^2
        nn_over = [
            inside(i, j) && is_cardiac[Int(lab[i, j]) + 1] ? Float32(lab[i, j]) : NaN32
            for i in 1:nx, j in 1:ny
        ]
        lin_over = [
            inside(i, j) && cov[i, j] > 0.01f0 ? Float32(cov[i, j]) : NaN32
            for i in 1:nx, j in 1:ny
        ]
        z_slab = recon_opts_helical.z_cm
        z_cm = -z_slab / 2 + (z - 0.5) * z_slab / nz
        fig = Mke.Figure(size = (1500, 980))

        # the scan seen from the side: z runs left to right
        z_crop = size(phantom_cropped_tall, 3) * VOXEL_SIZE_CM[3]
        p = protocol_helical
        travel = p.n_rotations * p.pitch * p.collimation_mm / 10
        half = z_crop / 2 + 0.4
        ax = canvas(fig[1, 1:4], (-half, half, -2.15, 1.95))
        Mke.poly!(
            ax,
            rounded_rect((0.0, 0.0), z_crop, 1.6, 0.1);
            color = (COLORS.red, 0.22),
            strokecolor = COLORS.rule,
            strokewidth = 1.5,
        )
        zs = range(-travel / 2, travel / 2; length = 800)
        turn = travel / p.n_rotations
        Mke.lines!(
            ax,
            collect(zs),
            [0.62 * cos(2π * (zz + travel / 2) / turn) for zz in zs];
            color = (COLORS.beam, 0.9),
            linewidth = 2.5,
        )
        Mke.poly!(
            ax,
            rounded_rect((0.0, 0.0), z_slab, 1.6, 0.04);
            color = (COLORS.cyan, 0.22),
            strokecolor = COLORS.cyan,
            strokewidth = 2.5,
        )
        Mke.lines!(ax, [z_cm, z_cm], [-1.0, 0.8]; color = COLORS.violet, linewidth = 5)
        Mke.text!(
            ax,
            z_cm,
            -1.08;
            text = "Slice $(z) of $(nz)",
            align = (:center, :top),
            fontsize = 18,
            font = :bold,
            color = COLORS.violet,
        )
        Mke.text!(
            ax,
            -z_crop / 2 + 0.15,
            0.62;
            text = "Taller Crop: $(round(z_crop; digits = 1)) cm",
            align = (:left, :top),
            fontsize = 17,
            color = COLORS.ink,
        )
        Mke.text!(
            ax,
            -travel / 2,
            0.92;
            text = "Tube Path: $(round(Int, p.n_rotations)) Rotations at Pitch $(p.pitch), $(round(travel; digits = 1)) cm",
            align = (:left, :bottom),
            fontsize = 17,
            color = COLORS.beam,
            font = :bold,
        )
        Mke.text!(
            ax,
            z_slab / 2,
            0.92;
            text = "Reconstructed: $(z_slab) cm, $(nz) Slices",
            align = (:left, :bottom),
            fontsize = 17,
            color = COLORS.cyan,
            font = :bold,
        )
        # a ruler in cm along z
        ticks = collect(-2 * floor(z_crop / 4):2:(2 * floor(z_crop / 4)))
        Mke.lines!(ax, [ticks[1], ticks[end]], [-1.55, -1.55]; color = COLORS.muted)
        for t in ticks
            Mke.lines!(ax, [t, t], [-1.55, -1.45]; color = COLORS.muted)
            Mke.text!(
                ax,
                t,
                -1.62;
                text = "$(round(Int, t))",
                align = (:center, :top),
                fontsize = 15,
                color = COLORS.muted,
            )
        end
        Mke.text!(
            ax,
            ticks[end] + 0.3,
            -1.55;
            text = "z (cm)",
            align = (:left, :center),
            fontsize = 16,
            color = COLORS.muted,
        )

        hu_kw = (colormap = :grays, colorrange = (-200, 600))
        square = (aspect = Mke.DataAspect(),)
        ax1 = Mke.Axis(
            fig[2, 1];
            title = "HU Reconstruction",
            subtitle = "Helical WFBP, z = $(round(10 * z_cm; digits = 2)) mm",
            square...,
        )
        Mke.heatmap!(ax1, recon; hu_kw...)
        Mke.hidedecorations!(ax1)
        ax2 = Mke.Axis(
            fig[2, 2];
            title = "Cardiac Labels",
            subtitle = ":nearest",
            square...,
        )
        Mke.heatmap!(ax2, recon; hu_kw...)
        Mke.heatmap!(
            ax2,
            nn_over;
            colormap = :tab20,
            colorrange = LABEL_RANGE,
            alpha = 0.65,
            nan_color = (:white, 0.0),
        )
        Mke.hidedecorations!(ax2)
        ax3 = Mke.Axis(
            fig[2, 3];
            title = "Cardiac Coverage",
            subtitle = ":linear on a 0/1 Mask",
            square...,
        )
        Mke.heatmap!(ax3, recon; hu_kw...)
        hc = Mke.heatmap!(
            ax3,
            lin_over;
            colormap = :viridis,
            colorrange = (0, 1),
            alpha = 0.75,
            nan_color = (:white, 0.0),
        )
        Mke.hidedecorations!(ax3)
        Mke.Colorbar(fig[2, 4], hc; label = "Cardiac Fraction", width = 16)
        Mke.rowsize!(fig.layout, 1, Mke.Relative(0.36))
        fig
    end
end

# ╔═╡ 05000017-0000-4000-8000-000000000001
md"""
## 7. The Mapping Is Exact

Three checks, for the axial and the helical geometry:

1. **The affine is the reconstructor's grid.** The FDK and WFBP back-projectors place
   reconstruction voxel `idx` (1-indexed) at world `-fov/2 + (idx − ½)·fov/n`, the rule
   `recon_to_world_affine` encodes; the table gives the largest difference.
2. **The round trip is the identity.** `A⁻¹ · A · v = v` to floating-point precision.
3. **The image is registered to the map.** Edges of the reconstruction are correlated with the
   label boundaries over integer in-plane shifts; the best shift should be `(0, 0)`. A
   half-voxel or rebinning offset between the forward projector and the back-projector would
   show here and nowhere else. (A systematic shift below half a pixel also reads `(0, 0)`.)

`is_helical(geom)` switches the back-projection algorithm, never the grid, so the helical
mapping is as exact as the axial one. Softness at a boundary is `:nearest` quantisation or the
reconstruction's point-spread function, not the mapping.
"""

# ╔═╡ 31a6b64b-6059-4946-810b-1ec91e518c35
affine_audit = sim === nothing ?
    nothing :
    let
        # the affine's world coordinate vs the back-projector's voxel-centre rule, and A⁻¹·A·v − v
        function audit(name, geom, ms)
            A = BS.recon_to_world_affine(geom, ms)
            Ainv = inv(A)
            grid_err = 0.0
            rt_err = 0.0
            for ax in 1:3, idx in (1, (ms[ax] + 1) ÷ 2, ms[ax])
                v = zeros(4)
                v[4] = 1.0
                v[ax] = idx - 1
                world_bp = -geom.fov[ax] / 2 + (idx - 0.5) * geom.fov[ax] / ms[ax]
                grid_err = max(grid_err, abs((A * v)[ax] - world_bp))
                rt_err = max(rt_err, abs((Ainv * (A * v))[ax] - (idx - 1)))
            end
            return (name = name, grid_um = grid_err * 1.0e4, roundtrip = rt_err)
        end
        out = [audit("axial", sim.geom, recon_opts.matrix_size)]
        sim_helical === nothing ||
            push!(out, audit("helical", sim_helical.geom, recon_opts_helical.matrix_size))
        out
    end;

# ╔═╡ 05000012-0000-4000-8000-000000000050
if affine_audit === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        sig(x) = round(x; sigdigits = 2)
        rows = [
            "| $(a.name) | $(sig(a.grid_um)) | $(sig(a.roundtrip)) |"
            for a in affine_audit
        ]
        Markdown.parse(
            """
        | geometry | affine vs reconstructor grid (max, µm) | round trip `A⁻¹·A·v − v` (max, voxels) |
        |---|--:|--:|
        $(join(rows, "\n"))
        """,
        )
    end
end

# ╔═╡ ef016027-652e-4344-a080-85392889ed75
registration = sim === nothing ?
    nothing :
    let
        # Correlate reconstruction edge strength with the label-boundary map over integer
        # (dx, dy) shifts on the central slice; the best shift is the real in-plane offset.
        function reg_offset(name, recon, gt; R = 5)
            mid = size(recon, 3) ÷ 2
            hu = Float32.(recon[:, :, mid])
            lab = gt[:, :, mid]
            nx, ny = size(hu)
            gr = zeros(Float32, nx, ny)
            lb = zeros(Float32, nx, ny)
            for j in 2:(ny - 1), i in 2:(nx - 1)
                gr[i, j] = hypot(hu[i + 1, j] - hu[i - 1, j], hu[i, j + 1] - hu[i, j - 1])
                # a symmetric boundary marker (both sides of an edge): no half-voxel bias
                boundary = lab[i, j] != lab[i + 1, j] ||
                    lab[i, j] != lab[i - 1, j] ||
                    lab[i, j] != lab[i, j + 1] ||
                    lab[i, j] != lab[i, j - 1]
                boundary && (lb[i, j] = 1.0f0)
            end
            scores = Dict(
                (dx, dy) => sum(
                    gr[i, j] * lb[i + dx, j + dy]
                    for j in (1 + R):(ny - R), i in (1 + R):(nx - R)
                )
                for dx in -R:R, dy in -R:R
            )
            best = argmax(scores)
            return (name = name, best = best, ratio = scores[(0, 0)] / scores[best])
        end
        out = [reg_offset("axial", recon_HU, gt_resampled_nn)]
        sim_helical === nothing ||
            push!(out, reg_offset("helical", recon_HU_helical, gt_helical_nn))
        out
    end;

# ╔═╡ 05000099-0000-4000-8000-000000000001
if registration === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        rows = [
            "| $(r.name) | $(r.best) | $(round(r.ratio; digits = 3)) |"
            for r in registration
        ]
        Markdown.parse(
            """
        **Registration of the image to the labels** (central slice, shifts in pixels)

        | geometry | best (dx, dy) | score at (0, 0) / best |
        |---|---|--:|
        $(join(rows, "\n"))
        """,
        )
    end
end

# ╔═╡ 9a83292b-a2b4-4310-a921-0e3371e858e6
verification = affine_audit === nothing ?
    nothing :
    let
        checks = NamedTuple[]
        addcheck(name, value, lo, hi) = push!(
            checks,
            (name = name, value = value, lo = lo, hi = hi, pass = lo <= value <= hi),
        )
        for a in affine_audit
            addcheck(
                "$(a.name): affine vs back-projector grid (µm)",
                a.grid_um,
                0.0,
                1.0e-3,
            )
            addcheck("$(a.name): round trip A⁻¹·A·v − v (voxels)", a.roundtrip, 0.0, 1.0e-9)
        end
        for r in registration
            addcheck(
                "$(r.name): best in-plane shift (pixels, largest component)",
                maximum(abs, r.best),
                0,
                0,
            )
        end
        checks
    end;

# ╔═╡ 2a3d1846-2ee9-4224-b762-e4f134513d6e
if verification === nothing
    md"""!!! warning "Skipped: no XCAT export (see 1)" """
else
    let
        n_ok = count(c -> c.pass, verification)
        verdict = n_ok == length(verification) ?
            "✅ Verification: PASS ($(n_ok)/$(length(verification)))" :
            "❌ Verification: FAIL ($(n_ok)/$(length(verification)))"
        rows = [
            "| $(c.name) | $(round(Float64(c.value); sigdigits = 2)) | [$(c.lo), $(c.hi)] | $(c.pass ? "✅ pass" : "❌ CHECK (outside the bounds)") |"
            for c in verification
        ]
        Markdown.parse(
            """
        #### $(verdict)

        | check | value | expected | status |
        |:--|--:|:--|:--|
        $(join(rows, "\n"))
        """,
        )
    end
end

# ╔═╡ 05000011-0000-4000-8000-000000000020
if verification === nothing
    md"""
    ## Summary

    The XCAT export was not found, so nothing was simulated (see 1).
    """
else
    let
        crop_pct = round(
            100 * length(phantom_cropped) / length(phantom_full_uhr);
            digits = 1,
        )
        sig(x) = round(x; sigdigits = 2)
        worst_grid = maximum(a.grid_um for a in affine_audit)
        worst_rt = maximum(a.roundtrip for a in affine_audit)
        shifts = join(["$(r.name) $(r.best)" for r in registration], ", ")
        n_roi = length(roi_stats.rows)
        n_close = count(s -> abs(s.mean - s.theory) <= 50, roi_stats.rows)
        n_ok = count(c -> c.pass, verification)
        Markdown.parse(
            """
        ## Summary

        - **Crop the phantom, not the reconstruction.** The cardiac block is $(crop_pct) % of the
          $(join(size(phantom_full_uhr), " × ")) phantom; only it was ray-traced, centred on the
          isocentre where the reconstruction grid always sits.
        - **Two affines and one resampler.** `phantom_to_world_affine` and `recon_to_world_affine` map
          each grid to world centimetres; `resample_to_recon` composes them (`:nearest` for labels,
          `:linear` for fields and coverage fractions), and `inv(A_phantom) * A_recon` hands the map
          to any other interpolator.
        - **Exact, axial and helical.** The affine matches the back-projectors' grid to
          $(sig(worst_grid)) µm, the round trip is the identity to $(sig(worst_rt)) voxels, and the
          best image-to-label shift is $(shifts). Verification: $(n_ok) of $(length(verification)) checks pass.
        - **What it buys you.** Organ ROIs are one indexing expression: $(n_close) of the $(n_roi)
          labels measured on Scan A's central slice read within 50 HU of their monoenergetic theory.
          Scan A delivered $(round(sim.dose.ctdi_vol_mGy; digits = 2)) mGy CTDIvol, Scan B
          $(round(sim_helical.dose.ctdi_vol_mGy; digits = 2)) mGy over $(round(sim_helical.dose.scan_length_cm; digits = 1)) cm.
        """,
        )
    end
end

# ╔═╡ Cell order:
# ╟─05000001-0000-4000-8000-000000000010
# ╟─5ec9c15c-c610-4636-8a15-ee5215cd87e0
# ╟─972e9b27-0fa9-4850-96d8-dd1745391886
# ╟─a7131b46-c595-4264-9b65-4bf37ef4848c
# ╟─05000001-0000-4000-8000-000000000020
# ╠═05000001-0000-4000-8000-000000000001
# ╠═05000001-0000-4000-8000-000000000002
# ╠═05000001-0000-4000-8000-000000000003
# ╠═05000001-0000-4000-8000-000000000004
# ╠═05000001-0000-4000-8000-000000000030
# ╠═05000001-0000-4000-8000-000000000031
# ╠═05000001-0000-4000-8000-000000000060
# ╠═05000001-0000-4000-8000-000000000040
# ╟─05000001-0000-4000-8000-000000000050
# ╠═05000001-0000-4000-8000-000000000070
# ╟─9ffb859c-b946-4d36-b221-0c8e6c827b12
# ╠═1da9cf11-165f-4c4d-a06e-4d2533dc40bb
# ╠═f0f47d18-a4a7-4845-a8d0-6a5fae3a3d02
# ╠═d77fcdce-0d62-4555-849a-523f39c191c4
# ╠═0f569ee4-d4d2-4c27-9f52-aa48118ad858
# ╠═809f415d-0a41-4856-ab1f-309db255eac7
# ╟─05000002-0000-4000-8000-000000000000
# ╟─05000002-0000-4000-8000-000000000001
# ╠═05000002-0000-4000-8000-000000000010
# ╠═05000002-0000-4000-8000-000000000014
# ╠═05000002-0000-4000-8000-000000000011
# ╠═05000002-0000-4000-8000-000000000012
# ╠═05000002-0000-4000-8000-000000000015
# ╟─05000002-0000-4000-8000-000000000013
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000010
# ╠═05000003-0000-4000-8000-000000000011
# ╠═05000003-0000-4000-8000-000000000020
# ╠═05000003-0000-4000-8000-000000000021
# ╠═05000003-0000-4000-8000-000000000030
# ╟─05000003-0000-4000-8000-000000000040
# ╟─05000004-0000-4000-8000-000000000001
# ╠═05000004-0000-4000-8000-000000000002
# ╠═05000004-0000-4000-8000-000000000004
# ╠═05000004-0000-4000-8000-000000000005
# ╠═05000004-0000-4000-8000-000000000006
# ╠═05000004-0000-4000-8000-000000000007
# ╠═05000004-0000-4000-8000-000000000008
# ╠═05000004-0000-4000-8000-000000000010
# ╠═05000004-0000-4000-8000-000000000012
# ╟─05000004-0000-4000-8000-000000000011
# ╟─05000005-0000-4000-8000-000000000001
# ╠═05000005-0000-4000-8000-000000000010
# ╟─05000005-0000-4000-8000-000000000011
# ╠═05000005-0000-4000-8000-000000000020
# ╟─05000006-0000-4000-8000-000000000001
# ╠═05000006-0000-4000-8000-000000000010
# ╟─05000006-0000-4000-8000-000000000020
# ╟─05000007-0000-4000-8000-000000000010
# ╟─a9ef1c11-fd34-49f5-8468-b7c28abacc2d
# ╠═05000008-0000-4000-8000-000000000011
# ╠═05000008-0000-4000-8000-000000000010
# ╟─05000009-0000-4000-8000-000000000001
# ╟─05000009-0000-4000-8000-000000000010
# ╟─0500000a-0000-4000-8000-000000000001
# ╠═0500000a-0000-4000-8000-000000000010
# ╟─99ccd184-5dcd-5bc5-9ca2-586e5ff69f7c
# ╠═0500000a-0000-4000-8000-000000000030
# ╠═0500000a-0000-4000-8000-000000000040
# ╟─0500000b-0000-4000-8000-000000000001
# ╠═0500000b-0000-4000-8000-000000000010
# ╟─0500000b-0000-4000-8000-000000000020
# ╠═0500000b-0000-4000-8000-000000000030
# ╟─0500000b-0000-4000-8000-000000000040
# ╟─0500000c-0000-4000-8000-000000000010
# ╟─230ebca6-6372-42b6-b300-18e11c8f001c
# ╟─0500000d-0000-4000-8000-000000000001
# ╠═0500000d-0000-4000-8000-000000000010
# ╠═0500000d-0000-4000-8000-000000000020
# ╠═0500000d-0000-4000-8000-000000000030
# ╟─0500000d-0000-4000-8000-000000000040
# ╟─0500000e-0000-4000-8000-000000000001
# ╟─b17dd975-086f-4472-8446-0e62de4c4706
# ╠═0500000e-0000-4000-8000-000000000010
# ╠═0500000e-0000-4000-8000-000000000012
# ╠═0500000e-0000-4000-8000-000000000013
# ╟─0500000e-0000-4000-8000-000000000020
# ╟─05000010-0000-4000-8000-000000000001
# ╟─05000010-0000-4000-8000-000000000010
# ╟─05000018-0000-4000-8000-000000000001
# ╠═7e808dde-5645-4a48-b10e-39e75f1cffd7
# ╟─05000018-0000-4000-8000-000000000002
# ╟─0500000f-0000-4000-8000-000000000001
# ╟─0500000f-0000-4000-8000-000000000010
# ╟─05000012-0000-4000-8000-000000000000
# ╠═05000013-0000-4000-8000-000000000010
# ╠═05000013-0000-4000-8000-000000000020
# ╠═05000013-0000-4000-8000-000000000031
# ╠═05000013-0000-4000-8000-000000000030
# ╠═05000013-0000-4000-8000-000000000040
# ╟─05000012-0000-4000-8000-000000000001
# ╠═05000012-0000-4000-8000-000000000005
# ╠═05000012-0000-4000-8000-000000000010
# ╠═05000012-0000-4000-8000-000000000020
# ╟─05000014-0000-4000-8000-000000000010
# ╠═05000012-0000-4000-8000-000000000030
# ╠═05000016-0000-4000-8000-000000000010
# ╟─05000015-0000-4000-8000-000000000001
# ╠═05000015-0000-4000-8000-000000000010
# ╟─05000016-0000-4000-8000-000000000020
# ╟─05000017-0000-4000-8000-000000000001
# ╠═31a6b64b-6059-4946-810b-1ec91e518c35
# ╟─05000012-0000-4000-8000-000000000050
# ╠═ef016027-652e-4344-a080-85392889ed75
# ╟─05000099-0000-4000-8000-000000000001
# ╠═9a83292b-a2b4-4310-a921-0e3371e858e6
# ╟─2a3d1846-2ee9-4224-b762-e4f134513d6e
# ╟─05000011-0000-4000-8000-000000000020
