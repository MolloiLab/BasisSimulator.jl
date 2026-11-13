### A Pluto.jl notebook ###
# v0.7.0

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

# ╔═╡ 01000001-0000-4000-8000-000000000001
md"""
# The Five-Struct API

**Describe a CT scan with five structs, simulate it, reconstruct it and read off its dose.**

This notebook scans a Gammex 472 calibration phantom on a model of the GE Revolution Apex Elite
twice, at 200 mA and at 50 mA, reconstructs both with filtered back-projection (FBP) and with
Hybrid IR, and checks every number against physics: water at 0 HU, noise that scales with dose,
and each calcium and iodine rod at its theoretical HU.
"""

# ╔═╡ cb84df00-b571-498a-87ae-f805fa7c2e6b
let
    Mke.set_theme!(THEME)
    k = water_roi.kmid
    fig, axes = locator_band(
        ["The Phantom", "The Scanner", "The Image"],
        [
            "Gammex 472: 14 Rods in Solid Water",
            "GE Revolution Apex Elite",
            "FBP, 120 kVp, 200 mA",
        ];
        limits = [(-18, 18, -18, 18), (-7.5, 7.5, -7.5, 7.5), (-17.5, 17.5, -17.5, 17.5)],
    )

    mid = size(phantom_cpu.mask, 3) ÷ 2
    draw_gammex!(
        axes[1],
        phantom_cpu.mask[:, :, mid],
        phantom_cpu.materials,
        phantom_cpu.voxel_size[1];
        annotate = false,
    )

    gantry!(axes[2]; labels = ("X-Ray Tube",), fan_deg = 45.8, object = false)
    draw_gammex!(
        axes[2],
        phantom_cpu.mask[:, :, mid],
        phantom_cpu.materials,
        phantom_cpu.voxel_size[1] * 2.2 / 17.5;
        annotate = false,
    )

    Mke.image!(
        axes[3],
        (-17.5, 17.5),
        (-17.5, 17.5),
        hu_fbp_std[:, :, k];
        colormap = :grays,
        colorrange = (-200, 500),
    )
    fig
end

# ╔═╡ 875b5c6f-425e-4e2e-b5ad-8f6842992502
md"""
Every simulation in `BasisSimulator.jl` is specified by five plain Julia structs, one for each
thing a CT scan needs, and run by three calls:

| Struct | What It Describes | Built Here With |
|:--|:--|:--|
| `Phantom` | the object: a labeled voxel mask, one `XrayAttenuation.Material` per label, the voxel size | `create_gammex_472` |
| `EICTScanner` / `PCCTScanner` | the hardware: geometry, detector, focal spot, filtration, bowtie | `REVOLUTION_APEX` |
| `CTProtocol` | the acquisition: kVp, mA, views, rotation time, collimation, pitch | `CTProtocol(; ...)` |
| `SimOptions` | the physics: which effects the forward model includes, the noise seed, the projector | `SimOptions(; ...)` |
| `ReconOptions` | the output grid: matrix size, in-plane field of view, z extent | `ReconOptions(; ...)` |
"""

# ╔═╡ 25c7eadb-a43d-49b5-bf92-bc3a131bad5b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 520))
    nodes = Dict(
        :phantom => (0.0, 4.0, "Phantom", "The Object"),
        :scanner => (0.0, 2.8, "Scanner", "The Hardware"),
        :protocol => (0.0, 1.6, "CTProtocol", "The Acquisition"),
        :sim => (0.0, 0.4, "SimOptions", "The Physics"),
        :recon => (0.0, -0.8, "ReconOptions", "The Image Grid"),
        :ws => (4.6, 1.6, "create_workspace", "Set Up Once", COLORS.cyan),
        :simulate => (8.4, 1.6, "simulate!", "The Sinogram", COLORS.cyan),
        :reconstruct => (12.2, 1.6, "reconstruct!", "FBP or Hybrid IR", COLORS.cyan),
        :hu => (16.0, 1.6, "HU Image", "to_hounsfield", COLORS.green),
        :dose => (12.2, -0.4, "Dose Report", "CTDIvol, DLP", COLORS.beam),
    )
    edges = [
        :phantom => :ws,
        :scanner => :ws,
        :protocol => :ws,
        :sim => :ws,
        :recon => :ws,
        :ws => :simulate,
        :simulate => :reconstruct,
        :reconstruct => :hu,
        :simulate => :dose,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.8, 17.8, -1.5, 4.7))
    fig
end

# ╔═╡ 01000002-0000-4000-8000-000000000001
md"""
## Notebook Setup

The notebook activates the shared `docs/` environment (which pulls `BasisSimulator` from this
source tree) and loads CairoMakie for the figures.
"""

# ╔═╡ 01000003-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 12000001-0000-4000-8000-000000000002
using PlutoUI

# ╔═╡ 01000003-0000-4000-8000-000000000005
using Markdown: @md_str

# ╔═╡ 01000003-0000-4000-8000-000000000004
using Statistics: mean, std

# ╔═╡ 01000003-0000-4000-8000-000000000002
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 01000003-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
import CairoMakie as Mke

# ╔═╡ 886270ac-b0c1-4c77-b218-3bb67c8bee20
TableOfContents()

# ╔═╡ 01000004-0000-4000-8000-000000000001
md"""
### Choosing the Device

`BasisSimulator` writes its kernels once, with
[AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl), and runs them on
whatever array type it is handed: `CuArray` for an NVIDIA GPU (here), `MtlArray` on Apple silicon,
`ROCArray` on AMD, or `Array` for the CPU. Moving the phantom mask to that type is the only thing
that decides where the whole pipeline runs.
"""

# ╔═╡ 01000005-0000-4000-8000-000000000001
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 01000007-0000-4000-8000-000000000001
Markdown.parse(
    """
**Backend detected:** $(GPU_BACKEND.name)
""",
)

# ╔═╡ 2f9a1c3e-0d7b-4c51-9a3e-6b2f8d1e4a70
md"""
## Figure Style

BasisSimulator's colours and type, and the drawing helpers every notebook's figures share.
"""

# ╔═╡ 694771e5-23e0-4450-afcf-881e4e62dc6a
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

# ╔═╡ bc83aa9a-649e-44ec-b079-0d8cad213f39
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

# ╔═╡ 06dbb07a-4db9-41a9-9c32-397cd8977cbb
"""A blank drawing area at `position`: equal scales, no axes, limits `(x0, x1, y0, y1)`."""
function canvas(position, limits; background = COLORS.paper)
    ax = Mke.Axis(position; aspect = Mke.DataAspect(), backgroundcolor = background)
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    Mke.limits!(ax, limits...)
    return ax
end;

# ╔═╡ b852ff0b-a12c-4d33-b0c0-f6c47841df59
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
end;

# ╔═╡ cc7a479c-361d-4f79-b587-b5cb92df08e9
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
end;

# ╔═╡ df70b269-0011-4839-9d9a-6da1db9101a0
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
end;

# ╔═╡ d5877142-69dc-489b-b5a4-829361dbfc2b
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
end;

# ╔═╡ 0502aa42-5f36-4b12-a551-07b2e8dbca67
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
end;

# ╔═╡ 9a37760b-ee4f-4e5c-ae88-bc78008b1944
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
end;

# ╔═╡ 07a2c861-382c-4a40-8334-c620a03115c9
"""`c` mixed toward `base` (the paper): `t = 1` is `c` itself, `t = 0` is `base`."""
function blend(c, t; base = COLORS.paper)
    return Mke.RGBf(
        t * c.r + (1 - t) * base.r,
        t * c.g + (1 - t) * base.g,
        t * c.b + (1 - t) * base.b,
    )
end

# ╔═╡ 98bca9b1-ce42-4b66-abd0-900b979c517b
"""A Gammex 472 material's insert, `(element, mg_per_mL)`, or `nothing` for water and air."""
function gammex_insert(material)
    m = match(r"([\d.]+) mg/ml (Calcium|Iodine)", material.name)
    return m === nothing ? nothing : (element = m[2], mg_per_mL = parse(Float64, m[1]))
end

# ╔═╡ 498ef857-4c4c-4feb-87fa-8873bbbe1806
"""The colour a Gammex 472 material is drawn in: water blue, calcium red, iodine green, deeper for more."""
function gammex_color(material; base = COLORS.paper)
    insert = gammex_insert(material)
    if insert === nothing
        return blend(COLORS.blue, 0.42; base)
    elseif insert.element == "Calcium"
        return blend(COLORS.red, 0.3 + 0.7 * insert.mg_per_mL / 600; base)
    else
        return blend(COLORS.green, 0.3 + 0.7 * insert.mg_per_mL / 20; base)
    end
end

# ╔═╡ 38f6008d-a7da-4024-8a9b-5a776bf528cf
"""
A slice of Gammex 472 `labels` drawn on `ax` in its material colours, `dx` cm per pixel, centred
on the isocentre (air left clear). `annotate` writes each rod's concentration (mg/mL) on it.
"""
function draw_gammex!(ax, labels, materials, dx; annotate = true, fontsize = 14)
    n1, n2 = size(labels)
    clear = Mke.RGBAf(0, 0, 0, 0)
    colour(l) = startswith(materials[l + 1].name, "Air") ?
        clear :
        Mke.RGBAf(gammex_color(materials[l + 1]))
    lut = Dict(l => colour(l) for l in unique(labels))
    Mke.image!(
        ax,
        (-n1 / 2 * dx, n1 / 2 * dx),
        (-n2 / 2 * dx, n2 / 2 * dx),
        [lut[l] for l in labels];
        interpolate = false,
    )
    annotate || return ax

    for l in unique(labels)
        insert = gammex_insert(materials[l + 1])
        insert === nothing && continue
        pixels = findall(==(l), labels)
        x = (sum(p[1] for p in pixels) / length(pixels) - (n1 + 1) / 2) * dx
        y = (sum(p[2] for p in pixels) / length(pixels) - (n2 + 1) / 2) * dx
        deep = insert.element == "Calcium" ?
            insert.mg_per_mL >= 300 :
            insert.mg_per_mL >= 7.5
        value = insert.mg_per_mL >= 10 ? round(Int, insert.mg_per_mL) : insert.mg_per_mL
        Mke.text!(
            ax,
            x,
            y;
            text = string(value),
            align = (:center, :center),
            fontsize,
            font = :bold,
            color = deep ? Mke.RGBf(1, 1, 1) : COLORS.ink,
        )
    end
    return ax
end

# ╔═╡ 471c73a8-5ac6-4a25-b1bf-953dd4eca3c9
"""A dimension line from `a` to `b`, arrowheads at both ends, `text` beside its middle."""
function dimension!(
    ax,
    a,
    b,
    text;
    align = (:left, :center),
    offset = (10, 0),
    fontsize = 16,
)
    m = (Mke.Point2f(a) + Mke.Point2f(b)) / 2
    arrow!(ax, m, a; linewidth = 1.8, head = 11)
    arrow!(ax, m, b; linewidth = 1.8, head = 11)
    Mke.text!(ax, m; text, align, offset, fontsize, color = COLORS.ink)
    return nothing
end

# ╔═╡ 9dc621c9-6668-4d13-a541-6841e46dfa48
"""A label at `at` with a thin leader line to the point `to`."""
function callout!(
    ax,
    at,
    to,
    text;
    align = (:left, :center),
    fontsize = 16,
    color = COLORS.ink,
)
    Mke.lines!(ax, Mke.Point2f[to, at]; color = (COLORS.ink, 0.35), linewidth = 1.2)
    Mke.scatter!(ax, [Mke.Point2f(to)]; color = (COLORS.ink, 0.55), markersize = 7)
    Mke.text!(
        ax,
        at;
        text,
        align,
        fontsize,
        color,
        offset = (align[1] === :left ? 6 : -6, 0),
    )
    return nothing
end

# ╔═╡ 4e23ef56-a955-4b27-a13c-61aa1927e123
"""
`scanner` seen along z, to scale (one unit is 10 cm): the source, its fan, the detector arc, the
scan field and a round object `object_cm` across, with the distances that define the geometry.
"""
function draw_scanner!(ax, scanner; object_cm = 33.0)
    sid = scanner.source_to_isocenter / 100
    sdd = scanner.source_to_detector / 100
    fan = rad2deg(
        scanner.detector_cols * scanner.detector_col_size / scanner.source_to_isocenter,
    )
    gantry!(
        ax;
        sid,
        sdd,
        fan_deg = fan,
        r_object = object_cm / 20,
        labels = ("",),
        dark = false,
    )

    r_field = scanner.scan_diameter / 200
    Mke.lines!(
        ax,
        [
            Mke.Point2f(r_field * cos(t), r_field * sin(t))
            for t in range(0, 2π; length = 120)
        ];
        color = COLORS.violet,
        linestyle = :dash,
        linewidth = 2,
    )

    detector_y = sid - sdd
    mm(x) = isinteger(x) ? "$(Int(x)) mm" : "$(x) mm"
    guide(y, x; from = 0.0) = Mke.lines!(
        ax,
        Mke.Point2f[(from, y), (x, y)];
        color = (COLORS.ink, 0.25),
        linewidth = 1,
    )
    guide(sid, 8.6)
    guide(0.0, 8.6; from = object_cm / 20)
    guide(sid, -8.6)
    guide(detector_y, -8.6)
    dimension!(
        ax,
        (8.4, sid),
        (8.4, 0.0),
        "Source to Isocentre\n$(mm(scanner.source_to_isocenter))",
    )
    dimension!(
        ax,
        (-8.4, sid),
        (-8.4, detector_y),
        "Source to Detector\n$(mm(scanner.source_to_detector))";
        align = (:right, :center),
        offset = (-10, 0),
    )

    label(y, text, color; font = :bold) = Mke.text!(
        ax,
        0,
        y;
        text,
        align = (:center, :center),
        fontsize = 17,
        font,
        color,
    )
    label(sid + 0.62, "X-Ray Tube and Bowtie", COLORS.ink)
    label(sid - 1.25, "Fan $(round(fan; digits = 1))°", COLORS.ink; font = :regular)
    label(0.0, "Phantom\n$(round(Int, object_cm)) cm", COLORS.ink)
    label(
        -r_field - 0.45,
        "Scan Field, $(round(Int, scanner.scan_diameter / 10)) cm",
        COLORS.violet,
    )
    label(
        detector_y - 0.55,
        "Detector: $(scanner.detector_cols) Columns × $(scanner.detector_rows) Rows",
        COLORS.cyan,
    )
    return ax
end

# ╔═╡ dee92749-7bf9-4c42-98e8-2eba17913754
"""
`scanner` seen from the side, z across (stretched `stretch`×, units cm): the full cone of every
detector row, the collimated beam of `collimation_mm`, the phantom `object_z_cm` long, and the
`rows_read` rows the scan reads.
"""
function draw_collimation!(
    ax,
    scanner,
    collimation_mm,
    rows_read;
    stretch = 4.0,
    object_z_cm = 1.0,
)
    sid = scanner.source_to_isocenter / 10
    sdd = scanner.source_to_detector / 10
    magnification = sdd / sid
    detector_y = sid - sdd
    row_cm = scanner.detector_row_size / 10 * magnification * stretch
    half_detector = scanner.detector_rows * row_cm / 2
    half_beam = collimation_mm / 10 * magnification * stretch / 2
    half_read = rows_read * row_cm / 2
    half_object = object_z_cm * stretch / 2
    source = Mke.Point2f(0, sid)

    # the phantom, then the full cone (dashed) and the collimated beam
    Mke.poly!(
        ax,
        Mke.Rect2f(-half_object, -16.5, 2 * half_object, 33.0);
        color = (COLORS.blue, 0.45),
        strokecolor = COLORS.ink,
        strokewidth = 1,
    )
    Mke.poly!(
        ax,
        Mke.Point2f[source, (-half_detector, detector_y), (half_detector, detector_y)];
        color = (COLORS.beam, 0.08),
    )
    for s in (-1, 1)
        Mke.lines!(
            ax,
            Mke.Point2f[source, (s * half_detector, detector_y)];
            color = (COLORS.beam, 0.7),
            linestyle = :dash,
            linewidth = 1.5,
        )
    end
    Mke.poly!(
        ax,
        Mke.Point2f[source, (-half_beam, detector_y), (half_beam, detector_y)];
        color = (COLORS.beam, 0.7),
    )

    # the detector: every row, and the rows read in cyan
    Mke.poly!(
        ax,
        Mke.Rect2f(-half_detector, detector_y - 3.0, 2 * half_detector, 3.0);
        color = COLORS.card,
        strokecolor = COLORS.rule,
        strokewidth = 1,
    )
    Mke.poly!(
        ax,
        Mke.Rect2f(-half_read, detector_y - 3.0, 2 * half_read, 3.0);
        color = COLORS.cyan,
    )
    Mke.scatter!(ax, [source]; marker = :rect, markersize = 26, color = COLORS.ink)
    Mke.scatter!(ax, [source]; marker = :circle, markersize = 9, color = COLORS.beam)

    Mke.text!(
        ax,
        source + Mke.Point2f(0, 4);
        text = "X-Ray Tube",
        align = (:center, :bottom),
        fontsize = 17,
        font = :bold,
    )
    Mke.text!(
        ax,
        0,
        detector_y - 5;
        text = "$(rows_read) of $(scanner.detector_rows) Rows Read",
        align = (:center, :top),
        fontsize = 17,
        font = :bold,
        color = COLORS.cyan,
    )
    Mke.text!(
        ax,
        half_detector,
        detector_y - 5;
        text = "z →",
        align = (:right, :top),
        fontsize = 16,
        color = COLORS.muted,
    )
    full_mm = round(Int, scanner.detector_rows * scanner.detector_row_size)
    callout!(
        ax,
        (half_detector * 0.62, 22.0),
        (half_detector * 0.42, sid - (sid - detector_y) * 0.42),
        "Full Cone: $(scanner.detector_rows) × $(scanner.detector_row_size) mm\n= $(full_mm) mm at the Isocentre",
    )
    callout!(
        ax,
        (-half_detector * 0.5, 30.0),
        (0.0, 30.0),
        "This Scan's Beam:\n$(collimation_mm) mm at the Isocentre";
        align = (:right, :center),
    )
    callout!(
        ax,
        (-half_detector * 0.62, -8.0),
        (-half_object, -8.0),
        "Phantom, $(round(Int, 10 * object_z_cm)) mm Long";
        align = (:right, :center),
        color = COLORS.blue,
    )
    return ax
end

# ╔═╡ f878b0bd-2f42-4cec-81bb-04c1332cb81e
"""
One view of a turning gantry: the source's path (radius `r`), the arc `Δθ_deg` it turns through
while the detector reads the view (drawn `exaggeration`× wider), and `n` sub-views, one at the
middle of each `1/n` of the arc.
"""
function draw_view_arc!(ax, n; Δθ_deg = 0.72, exaggeration = 28, r = 6.0)
    half = deg2rad(Δθ_deg * exaggeration / 2)
    arc(ρ, a, b) = [Mke.Point2f(ρ * cos(t), ρ * sin(t)) for t in range(a, b; length = 60)]
    Mke.lines!(ax, arc(r, π / 2 - 1.1, π / 2 + 1.1); color = COLORS.rule, linewidth = 3)
    Mke.poly!(
        ax,
        vcat([Mke.Point2f(0, 0)], arc(r + 0.6, π / 2 - half, π / 2 + half));
        color = (COLORS.violet, 0.1),
    )
    Mke.lines!(ax, arc(r, π / 2 - half, π / 2 + half); color = COLORS.violet, linewidth = 5)
    for s in (-1, 1)
        t = π / 2 + s * half
        Mke.lines!(
            ax,
            Mke.Point2f[(0, 0), ((r + 0.6) * cos(t), (r + 0.6) * sin(t))];
            color = (COLORS.violet, 0.5),
            linestyle = :dash,
            linewidth = 1.5,
        )
    end

    sources = [
        Mke.Point2f(r * cos(t), r * sin(t))
        for t in (π / 2 + half - (k - 0.5) * 2 * half / n for k in 1:n)
    ]
    for s in sources
        Mke.lines!(ax, [s, Mke.Point2f(0, 0)]; color = (COLORS.beam, 0.55), linewidth = 2)
    end
    Mke.scatter!(
        ax,
        sources;
        color = COLORS.beam,
        markersize = 18,
        strokecolor = COLORS.ink,
        strokewidth = 1.5,
    )
    Mke.poly!(
        ax,
        Mke.Circle(Mke.Point2f(0, 0), 1.65);
        color = (COLORS.blue, 0.5),
        strokecolor = COLORS.ink,
        strokewidth = 1.2,
    )
    Mke.text!(
        ax,
        0,
        r + 0.9;
        text = "One View, Δθ = $(Δθ_deg)°",
        align = (:center, :bottom),
        fontsize = 17,
        font = :bold,
        color = COLORS.violet,
    )
    return ax
end

# ╔═╡ 9c2ca6c0-055c-42ce-a407-97841e4f31d2
"""Where BasisSimulator puts the source at gantry angle `θ` (radians), `sid` from the isocentre: view 1 starts below it."""
source_at(θ, sid) = Mke.Point2f(-sid * sin(θ), -sid * cos(θ))

# ╔═╡ 06ba114f-b575-4b0e-8dd3-e951793faac4
"""
The detector column (1-based, fractional) that the ray from the source at gantry angle `θ`
through the point `p` (cm) reaches. Column numbers grow clockwise across the fan.
"""
function column_through(p, θ, scanner)
    s = source_at(θ, scanner.source_to_isocenter / 10)
    central = -s
    ray = Mke.Point2f(p) - s
    γ = atan(
        central[1] * ray[2] - central[2] * ray[1],
        central[1] * ray[1] + central[2] * ray[2],
    )
    radians_per_column = scanner.detector_col_size / scanner.source_to_isocenter
    return (scanner.detector_cols + 1) / 2 -
        γ / radians_per_column +
        scanner.detector_col_offset
end

# ╔═╡ 680ebee9-4490-47bf-89ee-3272f213d055
"""
One view sketched to scale (cm): the Gammex slice `labels`, the source at gantry angle `θ`, its
fan and detector arc, and the ray through the point `p` with the dexel it lands on.
"""
function draw_view!(ax, θ, p, scanner, labels, materials, dx)
    sid = scanner.source_to_isocenter / 10
    sdd = scanner.source_to_detector / 10
    half = scanner.detector_cols *
        scanner.detector_col_size /
        scanner.source_to_isocenter /
        2
    s = source_at(θ, sid)
    toward = atan(-s[2], -s[1])
    arc = Mke.Point2f[
        s + sdd * Mke.Point2f(cos(toward + φ), sin(toward + φ))
        for φ in range(-half, half; length = 40)
    ]
    Mke.poly!(ax, vcat([s], arc); color = (COLORS.beam, 0.16))
    draw_gammex!(ax, labels, materials, dx; annotate = false)
    Mke.lines!(ax, arc; color = COLORS.cyan, linewidth = 7)

    d = Mke.Point2f(p) - s
    hit = Mke.Point2f(s + sdd * d / sqrt(d[1]^2 + d[2]^2))
    Mke.lines!(ax, [s, hit]; color = COLORS.red, linewidth = 2.5)
    Mke.scatter!(
        ax,
        [hit];
        color = COLORS.red,
        markersize = 13,
        strokecolor = COLORS.ink,
        strokewidth = 1,
    )
    Mke.scatter!(ax, [s]; marker = :rect, markersize = 20, color = COLORS.ink)
    Mke.scatter!(ax, [s]; marker = :circle, markersize = 7, color = COLORS.beam)
    return ax
end

# ╔═╡ 02000000-0000-4000-8000-000000000000
md"""
## 1. The Phantom

A `Phantom` holds three things: a **labeled mask** (a 3-D array of unsigned integers, one region
label per voxel), a **material per label** (an `XrayAttenuation.Material`, which gives the
simulator μ(E) at every energy of the spectrum) and the **voxel size** `(dx, dy, dz)` in cm. It is
centred on the isocentre unless you pass an origin.

`create_gammex_472` builds the Gammex Model 472 multi-energy phantom: a 33 cm solid-water body
with seven calcium rods (50–600 mg/mL) on an inner ring and seven iodine rods (2–20 mg/mL) on an
outer ring, each 28 mm across.
"""

# ╔═╡ 02000002-0000-4000-8000-000000000001
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512, # in-plane voxels across the 35 cm field
    n_slices = 16,  # axial slices
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 02000003-0000-4000-8000-000000000001
Markdown.parse(
    """
The mask is `$(eltype(phantom_cpu.mask))` of size $(join(size(phantom_cpu.mask), " × ")), with
voxels of $(join(round.(phantom_cpu.voxel_size .* 10; digits = 3), " × ")) mm and
$(length(phantom_cpu.materials)) entries in its material table (entry `L + 1` is the material of
label `L`; labels the phantom doesn't use are air). `Phantom` is parameterised on its mask's array
type, so moving the mask to the device once is enough: every forward projection of this phantom
then runs there.
""",
)

# ╔═╡ 02000004-0000-4000-8000-000000000001
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 02000005-0000-4000-8000-000000000001
let
    Mke.set_theme!(THEME)
    mid = size(phantom_cpu.mask, 3) ÷ 2
    half = size(phantom_cpu.mask, 1) * phantom_cpu.voxel_size[1] / 2

    fig = Mke.Figure(size = (1150, 760))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Gammex 472, Central Slice",
        subtitle = "Rod Concentrations in mg/mL",
        aspect = Mke.DataAspect(),
        xlabel = "x (cm)",
        ylabel = "y (cm)",
    )
    draw_gammex!(
        ax,
        phantom_cpu.mask[:, :, mid],
        phantom_cpu.materials,
        phantom_cpu.voxel_size[1],
    )
    Mke.limits!(ax, -half, half, -half, half)

    swatch(color) = Mke.PolyElement(; color, strokecolor = :transparent)
    Mke.Legend(
        fig[1, 2],
        [
            swatch(blend(COLORS.blue, 0.42)),
            swatch(blend(COLORS.red, 0.65)),
            swatch(blend(COLORS.green, 0.65)),
        ],
        [
            "Solid Water Body, 33 cm",
            "Calcium Rods, 50–600 mg/mL",
            "Iodine Rods, 2–20 mg/mL",
        ];
        valign = :center,
    )
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "gammex_472_phantom.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 03000001-0000-4000-8000-000000000001
md"""
## 2. The Scanner

The scanner is the **hardware**: everything that stays the same from one scan to the next. There
are two families, one per detector type, sharing a `ScannerGeometry` (source and detector
distances, detector array, focal spot, gantry, flat filter, bowtie):

- `EICTScanner`: an energy-integrating scintillator detector (material, depth, fill factors,
  detection gain, electronic noise);
- `PCCTScanner`: a photon-counting detector (sensor, energy thresholds, dead time and pile-up,
  binning; its charge sharing and noise are its Geant4 response table's).

Each constructor rejects the other family's keywords, so a photon-counting parameter can't end
up on a scintillator by accident. A clinical scanner is a documented constant: `REVOLUTION_APEX`
models the GE Revolution Apex Elite (256 rows of 0.625 mm, 834 columns, a curved Gemstone
detector, GE's large-body bowtie and a clinical DAS with 3500 e⁻ of electronic noise), and its
docstring gives the source of every value. A variant copies every parameter and replaces the
keywords given: `EICTScanner(REVOLUTION_APEX; detector_rows = 64)`.
"""

# ╔═╡ 03000002-0000-4000-8000-000000000001
scanner = BS.REVOLUTION_APEX

# ╔═╡ 006e51e3-bb52-5e9f-9cde-8d1ec973b9bd
# Every keyword of `scanner`: the documented constant `REVOLUTION_APEX`, with any field this notebook changes marked
let s = scanner, constant = BS.REVOLUTION_APEX
    shown(x) = replace(repr(x; context = :compact => true), "|" => "\\|")

    rows = map(propertynames(s)) do f
        value, standard = getproperty(s, f), getproperty(constant, f)
        note = isequal(value, standard) ?
            "" :
            "changed here (the constant has `$(shown(standard))`)"
        "| `$(f)` | `$(shown(value))` | $(note) |"
    end

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

# ╔═╡ 950dd46f-2d1d-4e4a-bbab-f20de8500648
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1100, 640))
    ax = canvas(fig[1, 1], (-14, 14, -7.3, 7.6))
    draw_scanner!(ax, scanner)
    Mke.Label(
        fig[0, 1],
        "The Revolution Apex Elite, Seen Along z, to Scale";
        fontsize = 22,
        font = :bold,
        halign = :left,
        tellwidth = false,
    )
    fig
end

# ╔═╡ 03000003-0000-4000-8000-000000000001
Markdown.parse(
    """
!!! info "Hardware here, acquisition in the protocol"
    Anything that changes from scan to scan (kVp, mA, views, rotation time, collimation, pitch)
    belongs to `CTProtocol`. That split is what lets one scanner run a sweep of protocols, as the
    two scans below do. This `scanner` has $(scanner.detector_rows) rows × $(scanner.detector_cols)
    columns, a `:$(scanner.detector_shape)` detector and the `:$(scanner.bowtie_filter)` bowtie.
""",
)

# ╔═╡ 04000001-0000-4000-8000-000000000001
md"""
## 3. The Protocol

The **acquisition**. Two protocols share everything except the tube current: a standard-dose scan
at 200 mA and a low-dose scan at 50 mA, both 120 kVp, 500 views in a 1 s rotation, 5 mm of
collimation and 4.5 mm of extra aluminium filtration.
"""

# ╔═╡ 04000002-0000-4000-8000-000000000001
protocol_standard = BS.CTProtocol(
    kVp = 120,
    mA = 200.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,               # nominal beam width N·T
    additional_filters = [("Al", 4.5)], # on top of the scanner's 2.5 mm flat filter
);

# ╔═╡ 04000003-0000-4000-8000-000000000001
protocol_lowdose = BS.CTProtocol(
    kVp = 120,
    mA = 50.0, # a quarter of the dose
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ a3de7eb8-1b94-4a25-8406-d6fc413aaed9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1100, 720))
    ax = canvas(fig[1, 1], (-80, 80, -56, 70))
    draw_collimation!(ax, scanner, protocol_standard.collimation_mm, size(sim_std.sino, 2))
    Mke.Label(
        fig[0, 1],
        "Collimation Selects the Rows";
        fontsize = 22,
        font = :bold,
        halign = :left,
        tellwidth = false,
    )
    Mke.Label(
        fig[1, 1, Mke.Top()],
        "Seen from the Side, z Stretched 4×";
        fontsize = 17,
        color = COLORS.muted,
        halign = :left,
        tellwidth = false,
    )
    fig
end

# ╔═╡ 04000004-0000-4000-8000-000000000001
md"""
The detector is 256 × 0.625 mm = 160 mm wide at the isocentre; `collimation_mm = 5.0` lights only
its central rows. The simulator adds symmetric guard rows where the cone beam needs them to cover
the whole reconstruction cylinder at the edge of the field. Leave `collimation_mm` unset to use the
full detector; add `pitch` and `n_rotations` for a helical scan (notebook 11).
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
## 4. The Physics

`SimOptions` chooses the effects of the forward model. Every effect is on by default except
optical crosstalk: fill factor, energy-dependent detector efficiency, patient scatter, focal-spot
blur, quantum and electronic noise, detector lag and the anode heel effect. The polychromatic
spectrum, the bowtie and beam hardening are always part of the model. Turn an effect off to
measure its contribution. `projector = :dd_fast` (the default) is the anti-aliased
distance-driven projector that walks the volume once for the whole spectrum; `:siddon`, the
point-sampled ray tracer, is kept for comparison.

**The detector integrates while the gantry turns.** A clinical detector reads each view for the
whole view period, while the gantry sweeps Δθ = 360° / views, so each reading is the intensity
averaged over that arc: the object blurs along the rotation, more the farther it sits from the
isocentre, while the noise, counted once per view, stays independent from view to view.
`view_samples` samples the arc at sub-views (the midpoint rule, in intensity, before scatter,
noise and the detector effects). Every clinical scanner in these notebooks uses
`view_samples = 5`, which costs five forward projections per view.
"""

# ╔═╡ 8b606344-e54f-4566-8a43-728872a93ee9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1300, 540))
    panels = [
        (1, "view_samples = 1: A Point View"),
        (5, "view_samples = 5: The Arc, Integrated"),
    ]
    for (i, (n, title)) in enumerate(panels)
        ax = canvas(fig[1, i], (-6, 6, -2.2, 7.8))
        draw_view_arc!(ax, n; Δθ_deg = 360 / protocol_standard.views)
        Mke.Label(fig[2, i], title; fontsize = 19, font = :bold, tellwidth = false)
    end
    Mke.Label(
        fig[3, 1:2],
        "The Arc Is Drawn 28× Wider Than It Is";
        fontsize = 16,
        color = COLORS.muted,
        tellwidth = false,
    )
    fig
end

# ╔═╡ 05000002-0000-4000-8000-000000000001
sim_opts = BS.SimOptions(
    # fixed noise realisation: same seed + same inputs = same sinogram
    seed = 1234,
    projector = :dd_fast,
    # the detector integrates over the arc of each view (5 sub-views)
    view_samples = 5,
    # Switch individual effects off, e.g.:
    # use_scatter     = false,   # no patient scatter
    # use_heel_effect = false,
    # use_focal_spot  = false,
    # use_noise       = false,   # the noiseless expected sinogram
);

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
!!! info "Detector-specific physics lives on the scanner"
    `SimOptions` carries only the effects both detector families share. Patient scatter is the
    scanner's Geant4 scatter kernels (BasisMonteCarlo's library, `scatter_kernels`), superposed on
    the phantom's own rays; every clinical constant names its kernels (a download of about 250 MB
    per scanner, cached after the first scan; `BS.scatter_kernels(entry; path)` reads a local copy),
    and a scanner the library has no entry for (`scatter_kernels = ""`) throws unless
    `use_scatter = false`. The detector then subtracts the scatter it estimates from its own data,
    as a real one does (`scatter_correction = true` on both families). Pile-up and the count-noise
    blend are fields of `PCCTScanner`; the scintillator's fill factor, energy response, optical
    crosstalk and lag (`use_fill_factor`, `use_detector_efficiency`, `detector_efficiency_mode`,
    `use_optical_crosstalk`, `use_lag`) are fields of `EICTScanner`, e.g.
    `EICTScanner(BS.REVOLUTION_APEX; use_lag = false)`.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
## 5. The Image Grid

`ReconOptions` is the **output grid**: matrix size, in-plane field of view and z extent, centred
on the isocentre. Here the 5 mm beam is reconstructed as eight 0.625 mm slices on a 512 × 512
grid over 35 cm (0.68 mm pixels). The algorithm and its filter are chosen where they are used,
when the reconstruction workspace is created.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000001
recon_opts = let
    slice_thickness_mm = 0.625
    n_slices = round(Int, protocol_standard.collimation_mm / slice_thickness_mm)
    BS.ReconOptions(
        matrix_size = (512, 512, n_slices),
        fov_cm = 35.0,
        z_cm = protocol_standard.collimation_mm / 10,
    )
end;

# ╔═╡ 07000000-0000-4000-8000-000000000000
md"""
## 6. Simulate

`create_workspace` does all the set-up once: it builds the scan geometry, resolves the source
spectrum through the filters and bowtie, and allocates every device buffer on the phantom's
backend. `simulate!` then runs the forward model into `ws.sinogram`, the log-transformed line
integrals `-log(I/I₀)` of size (columns, rows, views), noise included. The same constructor serves
both detector families: for a `PCCTScanner` it builds the photon-counting workspace, and
`simulate!` returns one sinogram per energy bin.

`simulate!` also returns the **dose** of the acquisition. The CTDI follows IEC 60601-2-44: the
simulated beam is transported by Monte Carlo through the 32 cm PMMA body phantom, and the air
kerma is integrated over a 100 mm pencil chamber at the centre and at the periphery, so it
follows this scanner's spectrum, filtration and bowtie rather than a lookup table. Each scan runs
inside a `let` block that keeps only CPU copies, so the device buffers are released before the
next scan starts.
"""

# ╔═╡ 07000010-0000-4000-8000-000000000001
sim_std = let
    ws = BS.create_workspace(scanner, protocol_standard, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol_standard, sim_opts)

    out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose)
    ws = nothing
    GC.gc(true) # hand the device memory back now
    out
end;

# ╔═╡ 07000011-0000-4000-8000-000000000001
sim_std.dose

# ╔═╡ 269d13d5-dc38-41c0-a6ed-97c6d01b10f3
md"""
### Reading the Sinogram

Each view is one gantry angle; each detector column is one ray of the fan. A sinogram stacks the
views side by side, so a rod off the isocentre swings across the columns as the gantry turns and
traces a sine. The sketches follow the ray through the 600 mg/mL calcium rod at four views; the
red dots on the sinogram are the columns it reaches.
"""

# ╔═╡ 07000015-0000-4000-8000-000000000001
let
    Mke.set_theme!(THEME)
    sino = sim_std.sino # (n_col, n_row, n_view), −log(I/I₀)
    n_col, n_row, n_view = size(sino)
    mid = size(phantom_cpu.mask, 3) ÷ 2
    labels = phantom_cpu.mask[:, :, mid]
    dx = phantom_cpu.voxel_size[1]
    n = size(labels, 1)

    # the centre of the densest calcium rod, in cm
    densest = argmax(
        l -> something(
            gammex_insert(phantom_cpu.materials[l + 1]),
            (; mg_per_mL = 0.0),
        ).mg_per_mL,
        unique(labels),
    )
    pixels = findall(==(densest), labels)
    centre = Mke.Point2f(
        (sum(p[1] for p in pixels) / length(pixels) - (n + 1) / 2) * dx,
        (sum(p[2] for p in pixels) / length(pixels) - (n + 1) / 2) * dx,
    )
    views = [1, 1 + n_view ÷ 4, 1 + n_view ÷ 2, 1 + 3 * n_view ÷ 4]
    angle(v) = 2π * (v - 1) / n_view
    trace = [column_through(centre, angle(v), scanner) for v in 1:n_view]

    fig = Mke.Figure(size = (1300, 1080))
    top = fig[1, 1:2] = Mke.GridLayout()
    for (i, v) in enumerate(views)
        ax = canvas(top[1, i], (-70, 70, -70, 70))
        draw_view!(ax, angle(v), centre, scanner, labels, phantom_cpu.materials, dx)
        Mke.Label(
            top[2, i],
            "View $(v), $(round(Int, rad2deg(angle(v))))°";
            fontsize = 18,
            font = :bold,
            tellwidth = false,
        )
    end
    Mke.rowgap!(top, 4)

    ax = Mke.Axis(
        fig[2, 1];
        title = "Standard-Dose Sinogram, Central Detector Row",
        subtitle = "120 kVp, 200 mA, $(n_view) Views × $(n_col) Columns; Red: the 600 mg/mL Rod",
        xlabel = "View",
        ylabel = "Detector Column",
    )
    hm = Mke.heatmap!(
        ax,
        1:n_view,
        1:n_col,
        permutedims(sino[:, n_row ÷ 2 + 1, :]);
        colormap = :viridis,
    )
    Mke.lines!(
        ax,
        1:n_view,
        trace;
        color = (COLORS.red, 0.9),
        linewidth = 1.5,
        linestyle = :dash,
    )
    Mke.scatter!(
        ax,
        views,
        trace[views];
        color = COLORS.red,
        markersize = 15,
        strokecolor = :white,
        strokewidth = 1.5,
    )
    Mke.Colorbar(fig[2, 2], hm; label = "Line Integral, −log(I / I₀)")
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.32))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "sinogram_standard.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
### The Low-Dose Scan

The same two calls with `protocol_lowdose`. The phantom, scanner, physics and grid are reused
unchanged, which is the point of keeping them in separate structs.
"""

# ╔═╡ 08000010-0000-4000-8000-000000000001
sim_low = let
    ws = BS.create_workspace(scanner, protocol_lowdose, sim_opts, recon_opts, phantom)
    result = BS.simulate!(ws, phantom, protocol_lowdose, sim_opts)

    out = (sino = Array(ws.sinogram), geom = ws.geom, dose = result.dose)
    ws = nothing
    GC.gc(true)
    out
end;

# ╔═╡ 08000011-0000-4000-8000-000000000001
sim_low.dose

# ╔═╡ 08000012-0000-4000-8000-000000000001
let
    d_s, d_l = sim_std.dose, sim_low.dose
    Markdown.parse(
        """
    | Protocol | mAs / Rotation | CTDIvol (mGy) | DLP (mGy·cm) | Scan Length (cm) |
    |:--|--:|--:|--:|--:|
    | standard dose | $(round(d_s.mAs_per_rotation; digits = 1)) | $(round(d_s.ctdi_vol_mGy; digits = 2)) | $(round(d_s.dlp_mGy_cm; digits = 2)) | $(round(d_s.scan_length_cm; digits = 2)) |
    | low dose | $(round(d_l.mAs_per_rotation; digits = 1)) | $(round(d_l.ctdi_vol_mGy; digits = 2)) | $(round(d_l.dlp_mGy_cm; digits = 2)) | $(round(d_l.scan_length_cm; digits = 2)) |

    The beam is identical, so CTDIvol scales with the tube current alone: the low-dose scan
    delivers $(round(d_l.ctdi_vol_mGy / d_s.ctdi_vol_mGy; digits = 3)) of the standard dose.
    CTDIw of this beam is $(round(d_s.ctdi_w_mGy_per_100mAs; digits = 2)) mGy per 100 mAs. The
    Monte Carlo's statistical uncertainty is ±$(round(100 * d_s.rel_stat_uncertainty; digits = 2)) %.
    Pass `dose_kwargs = (; phantom = :head16)` to `simulate!` for the 16 cm head phantom.
    """,
    )
end

# ╔═╡ 08000020-0000-4000-8000-000000000001
md"""
### Repeated Noise Draws

Everything before the noise (the view-integrated forward projection and the physics that
precedes the noise, scatter included) is the same for every noise draw of one phantom, protocol
and workspace. `simulate!(...; keep_projection = true)` returns it as `result.projection`, and
`simulate!(...; projection)` draws a new realisation from it, with the seed of the `SimOptions` it
is given, without projecting again. A noise study over many seeds, or a noise-free reference
(`use_noise = false`) that shares the noisy scan's projection, pays for the projection once.
"""

# ╔═╡ 08000021-0000-4000-8000-000000000001
noise_draws = let
    ws = BS.create_workspace(scanner, protocol_standard, sim_opts, recon_opts, phantom)
    # one full simulation that keeps everything before the noise …
    t_full = @elapsed (
        kept = BS.simulate!(
            ws,
            phantom,
            protocol_standard,
            sim_opts;
            report_dose = false,
            keep_projection = true,
        )
    )
    # … and a second noise realization drawn from it with another seed
    opts_2 = BS.SimOptions(;
        seed = 5678,
        projector = sim_opts.projector,
        view_samples = sim_opts.view_samples,
    )
    BS.simulate!(
        ws,
        phantom,
        protocol_standard,
        opts_2;
        report_dose = false,
        projection = kept.projection,
    )
    sino_reused = Array(ws.sinogram)
    # the same seed simulated from scratch, for comparison
    BS.simulate!(ws, phantom, protocol_standard, opts_2; report_dose = false)
    identical = Array(ws.sinogram) == sino_reused
    t_reuse = @elapsed BS.simulate!(
        ws,
        phantom,
        protocol_standard,
        opts_2;
        report_dose = false,
        projection = kept.projection,
    )
    ws = nothing
    kept = nothing
    GC.gc(true)
    (t_full = t_full, t_reuse = t_reuse, identical = identical)
end;

# ╔═╡ 08000022-0000-4000-8000-000000000001
Markdown.parse(
    """
A full simulation of the standard scan took $(round(noise_draws.t_full; digits = 2)) s; a noise
draw from its kept projection took $(round(noise_draws.t_reuse; digits = 2)) s,
$(round(noise_draws.t_full / noise_draws.t_reuse; digits = 1))× less. The draw from the
projection $(noise_draws.identical ? "is bit-identical to" : "differs from") simulating the same
seed from scratch.
""",
)

# ╔═╡ 07000000-0000-4000-8000-000000000001
md"""
## 7. Reconstruct

### The Water Beam-Hardening Correction

The simulated sinogram is polychromatic: the spectrum hardens as it crosses the patient, so the
line integrals grow more slowly than the path length, and an uncorrected reconstruction reads
low in the middle of the body (cupping). `calibrate_bhc_water` removes that for water. For every
detector column it resolves the full detected spectrum (tube, filters, bowtie, heel effect and
detector efficiency: exactly the factors `sim_opts` switched on) and fits the polynomial that
maps polychromatic water line integrals to monochromatic ones at the spectrum's mean energy. It
has no tunable parameters, and nothing in it is fitted to the data being corrected;
`apply_bhc_water` applies it to a sinogram. The reference energy and water μ that come with it
are what `to_hounsfield` needs to put water at 0 HU. Both protocols share a beam, so one
calibration serves both.
"""

# ╔═╡ 09000006-0000-4000-8000-000000000001
bhc = BS.calibrate_bhc_water(sim_opts, protocol_standard; scanner, geom = sim_std.geom);

# ╔═╡ 09000007-0000-4000-8000-000000000001
Markdown.parse(
    """
**Water BHC:** $(length(bhc.water_bhc_per_col)) per-column polynomials · reference energy
$(round(bhc.reference_energy_keV; digits = 1)) keV · water μ = $(round(bhc.μ_water_ref; digits = 5)) cm⁻¹
""",
)

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
### FBP and Hybrid IR Through One `reconstruct!`

`reconstruct!` dispatches on the reconstruction workspace, so the two algorithms differ only in
the workspace you create. `create_fdk_recon_workspace` is filtered back-projection
(Feldkamp–Davis–Kress for an axial scan, weighted FBP for a helical one) with the `:standard`
kernel by default. `create_hir_recon_workspace(...; strength)` is **Hybrid IR**: an FBP start
followed by ordered-subsets penalised weighted least squares with an edge-preserving Huber prior.
`strength` is a percentage in steps of 10, read like the vendor dials (GE ASIR-V, Siemens
SAFIRE): `0` is plain FBP, `60` the standard clinical setting. HIR's system matrix uses the same
projector as the simulation (`projector = sim_opts.projector`), so it inverts the operator that
made the data.
"""

# ╔═╡ 7a8a8143-539e-4884-8053-1b0af611dc7b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 420))
    nodes = Dict(
        :sino => (0.0, 0.0, "Sinogram", "From simulate!", COLORS.cyan),
        :cal => (4.0, 1.5, "calibrate_bhc_water", "From the Spectrum", COLORS.violet),
        :apply => (4.0, 0.0, "apply_bhc_water", "Water BHC", COLORS.violet),
        :fdk => (8.0, 0.75, "FBP Workspace", "create_fdk_recon_workspace", COLORS.cyan),
        :hir => (
            8.0,
            -0.75,
            "Hybrid IR Workspace",
            "create_hir_recon_workspace",
            COLORS.cyan,
        ),
        :rec => (12.0, 0.0, "reconstruct!", "μ in 1/cm", COLORS.cyan),
        :hu => (16.0, 0.0, "to_hounsfield", "HU, Water at 0", COLORS.green),
    )
    edges = [
        :sino => :apply,
        :cal => :apply,
        :apply => :fdk,
        :apply => :hir,
        :fdk => :rec,
        :hir => :rec,
        :rec => :hu,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.9, 17.9, -1.4, 2.2), w = 3.3)
    fig
end

# ╔═╡ ecb30d39-380e-478a-9172-048523b36d15
md"""
The function below is that whole chain and returns a CPU volume in HU.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
"""
    reconstruct_hu(sim, bhc; algorithm = :fbp, strength = 60) -> Array{Float32, 3}

Water BHC, then FBP (`algorithm = :fbp`) or Hybrid IR (`algorithm = :hir`), then HU.
"""
function reconstruct_hu(sim, bhc; algorithm::Symbol = :fbp, strength::Integer = 60)
    sino = BS.apply_bhc_water(to_gpu(sim.sino), bhc)
    ws = if algorithm === :fbp
        BS.create_fdk_recon_workspace(sino, sim.geom, recon_opts.matrix_size)
    elseif algorithm === :hir
        BS.create_hir_recon_workspace(
            sino,
            sim.geom,
            recon_opts.matrix_size;
            strength,
            projector = sim_opts.projector,
        )
    else
        throw(ArgumentError("algorithm must be :fbp or :hir"))
    end
    μ = BS.reconstruct!(ws, sino, sim.geom)
    hu = Float32.(BS.to_hounsfield(Array(μ); μ_water = bhc.μ_water_ref))
    ws = nothing
    sino = nothing
    μ = nothing
    GC.gc(true)
    return hu
end;

# ╔═╡ 07000021-0000-4000-8000-000000000001
hu_fbp_std = reconstruct_hu(sim_std, bhc; algorithm = :fbp);

# ╔═╡ 09000011-0000-4000-8000-000000000001
hu_hir_std = reconstruct_hu(sim_std, bhc; algorithm = :hir, strength = 60);

# ╔═╡ 08000013-0000-4000-8000-000000000001
hu_fbp_low = reconstruct_hu(sim_low, bhc; algorithm = :fbp);

# ╔═╡ 09000021-0000-4000-8000-000000000001
hu_hir_low = reconstruct_hu(sim_low, bhc; algorithm = :hir, strength = 60);

# ╔═╡ 10000000-0000-4000-8000-000000000000
md"""
## 8. Results

### Where Each Number Is Measured

Noise is only meaningful inside one material. The phantom labels are carried onto the
reconstruction grid with `resample_to_recon`, and the solid-water label is eroded by 3 pixels so
that only pixels surrounded by water remain: the rods and their partial-volume edges are
excluded. Each rod is measured in its own label eroded by 2 pixels. All statistics use the
central slice; the edge slices of a cone-beam reconstruction see fewer complete rays.
"""

# ╔═╡ 10000005-0000-4000-8000-000000000001
water_roi = let
    labels = BS.resample_to_recon(
        phantom_cpu,
        sim_std.geom,
        recon_opts.matrix_size;
        method = :nearest,
    )
    nx, ny, nz = size(labels)
    kmid = (nz + 1) ÷ 2
    water = UInt8(BS.REGION_SOLID_WATER)
    r = 3 # erosion radius, voxels
    m = falses(nx, ny)
    for j in (1 + r):(ny - r), i in (1 + r):(nx - r)
        m[i, j] = all(labels[i + di, j + dj, kmid] == water for dj in -r:r, di in -r:r)
    end
    (labels = labels, kmid = kmid, mask = m)
end;

# ╔═╡ 10000006-0000-4000-8000-000000000001
noise = let
    k, m = water_roi.kmid, water_roi.mask
    stat(v) = (mean = Float64(mean(v[:, :, k][m])), σ = Float64(std(v[:, :, k][m])))
    (
        fbp_std = stat(hu_fbp_std),
        hir_std = stat(hu_hir_std),
        fbp_low = stat(hu_fbp_low),
        hir_low = stat(hu_hir_low),
    )
end;

# ╔═╡ 0bf8cbc0-978c-49b0-bbe5-453b0ba93e65
let
    Mke.set_theme!(THEME)
    labels = water_roi.labels[:, :, water_roi.kmid]
    half = recon_opts.fov_cm / 2
    dx = recon_opts.fov_cm / size(labels, 1)

    fig = Mke.Figure(size = (1150, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Measurement Regions, Central Slice",
        subtitle = "On the 512 × 512 Reconstruction Grid",
        aspect = Mke.DataAspect(),
    )
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    draw_gammex!(ax, labels, phantom_cpu.materials, dx)
    cyan = Mke.RGBAf(COLORS.cyan.r, COLORS.cyan.g, COLORS.cyan.b, 0.85)
    Mke.image!(
        ax,
        (-half, half),
        (-half, half),
        [m ? cyan : Mke.RGBAf(0, 0, 0, 0) for m in water_roi.mask];
        interpolate = false,
    )
    Mke.limits!(ax, -half, half, -half, half)

    swatch(color) = Mke.PolyElement(; color, strokecolor = :transparent)
    Mke.Legend(
        fig[1, 2],
        [
            swatch(COLORS.cyan),
            swatch(blend(COLORS.red, 0.65)),
            swatch(blend(COLORS.green, 0.65)),
        ],
        [
            "Water: Mean and Noise\n$(count(water_roi.mask)) Pixels",
            "Calcium Rods: Mean HU",
            "Iodine Rods: Mean HU",
        ];
        valign = :center,
    )
    fig
end

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
### FBP vs Hybrid IR at Two Doses

The four reconstructions share physics, geometry and grid. Down the columns, only the tube
current changes; across the rows, only the reconstruction. Quartering the dose should double the
FBP noise (noise ∝ 1/√mAs) and leave every mean HU where it was. Hybrid IR at strength 60 should
lower the noise at both doses without moving the means.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
let
    Mke.set_theme!(THEME)
    k = water_roi.kmid
    window = (-200, 500)
    σ(s) = "σ = $(round(s.σ; digits = 1)) HU in Water"
    panels = [
        (1, 1, hu_fbp_std, "FBP, 200 mA", σ(noise.fbp_std)),
        (1, 2, hu_hir_std, "Hybrid IR 60, 200 mA", σ(noise.hir_std)),
        (2, 1, hu_fbp_low, "FBP, 50 mA", σ(noise.fbp_low)),
        (2, 2, hu_hir_low, "Hybrid IR 60, 50 mA", σ(noise.hir_low)),
    ]

    fig = Mke.Figure(size = (1100, 1040))
    local hm
    for (r, c, volume, title, subtitle) in panels
        ax = Mke.Axis(fig[r, c]; title, subtitle, aspect = Mke.DataAspect())
        hm = Mke.heatmap!(ax, volume[:, :, k]; colormap = :grays, colorrange = window)
        Mke.hidedecorations!(ax)
        Mke.hidespines!(ax)
    end
    Mke.Colorbar(fig[1:2, 3], hm; label = "HU (Window $(window[1]) to $(window[2]))")
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "recon_compare_4panel.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 10000003-0000-4000-8000-000000000001
let
    band = BS.get_hir_params(60).target_noise_reduction
    red(a, b) = round(100 * (1 - b.σ / a.σ); digits = 1)
    Markdown.parse(
        """
    **Water, central slice, $(count(water_roi.mask)) pixels deep inside the solid-water body:**

    | Protocol | FBP Mean (HU) | FBP σ (HU) | HIR Mean (HU) | HIR σ (HU) | HIR Noise Reduction |
    |:--|--:|--:|--:|--:|--:|
    | 200 mA | $(round(noise.fbp_std.mean; digits = 1)) | $(round(noise.fbp_std.σ; digits = 1)) | $(round(noise.hir_std.mean; digits = 1)) | $(round(noise.hir_std.σ; digits = 1)) | $(red(noise.fbp_std, noise.hir_std)) % |
    | 50 mA | $(round(noise.fbp_low.mean; digits = 1)) | $(round(noise.fbp_low.σ; digits = 1)) | $(round(noise.hir_low.mean; digits = 1)) | $(round(noise.hir_low.σ; digits = 1)) | $(red(noise.fbp_low, noise.hir_low)) % |

    FBP noise ratio, 50 mA / 200 mA = **$(round(noise.fbp_low.σ / noise.fbp_std.σ; digits = 2))×**.
    Quantum noise alone predicts 2×; the fixed electronic noise of the detector weighs more at
    low dose and pushes the ratio a little above that. The package's calibration band for HIR
    at strength 60 is $(band[1])–$(band[2]) % noise reduction.
    """,
    )
end

# ╔═╡ 12000003-0000-4000-8000-000000000001
md"""
### Dose and Noise at Other Tube Currents

The two scans anchor a simple rule: CTDIvol is proportional to mAs (same beam, same phantom), and
quantum-limited FBP noise scales as 1/√mAs. The table applies it to this protocol at other tube
currents. The detector's electronic noise makes the real noise somewhat higher than this at low
mA, as the measured 50 mA scan shows.
"""

# ╔═╡ 12000003-0000-4000-8000-000000000002
let
    mA0, ctdi0, σ0 = protocol_standard.mA, sim_std.dose.ctdi_vol_mGy, noise.fbp_std.σ
    rows = [
        "| $(mA) | $(round(ctdi0 * mA / mA0; digits = 2)) | $(round(σ0 * sqrt(mA0 / mA); digits = 1)) |"
        for mA in (25, 50, 100, 200, 300, 400)
    ]
    Markdown.parse(
        """
    | Tube Current (mA) | CTDIvol (mGy) | Expected FBP Water Noise (HU) |
    |--:|--:|--:|
    $(join(rows, "\n"))

    Measured at 50 mA: $(round(noise.fbp_low.σ; digits = 1)) HU.
    """,
    )
end

# ╔═╡ 12000003-0000-4000-8000-000000000003
# The calculator's inputs — this scan's tube current (mA), CTDIvol (mGy) and FBP water noise (HU) —
# written as constants: the browser runs the calculator compiled to WebAssembly, which bakes constants
# in but would have to compile any computation behind them (here, the whole simulation). The cell
# below checks them against the scan, so they cannot go stale.
const DOSE_INPUTS = (200.0, 13.72, 46.5);

# ╔═╡ 12000003-0000-4000-8000-000000000007
let measured = (
    Float64(protocol_standard.mA),
    Float64(sim_std.dose.ctdi_vol_mGy),
    Float64(noise.fbp_std.σ),
)
    all(isapprox.(DOSE_INPUTS, measured; rtol = 0.01)) ||
        error(
            "DOSE_INPUTS = $(DOSE_INPUTS) no longer match this scan, $(measured): update them",
        )
    nothing
end

# ╔═╡ 12000003-0000-4000-8000-000000000004
@bind tube_mA PlutoUI.Slider(10:10:500; default = 200, show_value = true)

# ╔═╡ 12000003-0000-4000-8000-000000000005
# tube current, CTDIvol (mGy), expected FBP noise (HU), noise relative to the standard scan
dose_plan = let r = tube_mA / DOSE_INPUTS[1]
    (
        tube_mA,
        round(DOSE_INPUTS[2] * r; digits = 2),
        round(DOSE_INPUTS[3] / sqrt(r); digits = 1),
        round(1 / sqrt(r); digits = 2),
    )
end;

# ╔═╡ 12000003-0000-4000-8000-000000000006
md"""
**At $(dose_plan[1]) mA:** CTDIvol is $(dose_plan[2]) mGy and the expected quantum-limited FBP water noise is $(dose_plan[3]) HU, $(dose_plan[4]) times that of the 200 mA scan.
"""

# ╔═╡ 12000001-0000-4000-8000-000000000001
md"""
### Scroll Through the Slices

The reconstructed slab is 5 mm thick. The edge slices are reconstructed from more oblique rays
than the central ones, so this is where any z-dependence would show first.
"""

# ╔═╡ 12000001-0000-4000-8000-000000000003
@bind z_slice PlutoUI.Slider(
    1:size(hu_fbp_std, 3);
    default = size(hu_fbp_std, 3) ÷ 2,
    show_value = true,
)

# ╔═╡ 12000001-0000-4000-8000-000000000004
let
    Mke.set_theme!(THEME)
    nz = size(hu_fbp_std, 3)
    fov_z_mm = sim_std.geom.fov[3] * 10
    dz_mm = fov_z_mm / nz
    z_mm = -fov_z_mm / 2 + (z_slice - 0.5) * dz_mm

    fig = Mke.Figure(size = (1300, 600))
    # where the slice sits in the slab: the slab seen from the side
    ax0 = canvas(fig[1, 1], (-1.6, 1.6, -fov_z_mm / 2 - 1.2, fov_z_mm / 2 + 1.0))
    for k in 1:nz
        z0 = -fov_z_mm / 2 + (k - 1) * dz_mm
        Mke.poly!(
            ax0,
            Mke.Rect2f(-1.0, z0, 2.0, dz_mm);
            color = k == z_slice ? COLORS.cyan : COLORS.card,
            strokecolor = COLORS.rule,
            strokewidth = 1,
        )
    end
    Mke.text!(
        ax0,
        0,
        fov_z_mm / 2 + 0.3;
        text = "The Slab",
        align = (:center, :bottom),
        fontsize = 17,
        font = :bold,
    )
    Mke.text!(
        ax0,
        0,
        -fov_z_mm / 2 - 0.3;
        text = "z = $(round(z_mm; digits = 2)) mm",
        align = (:center, :top),
        fontsize = 17,
        font = :bold,
        color = COLORS.cyan,
    )

    images = [(hu_fbp_std, "FBP, 200 mA"), (hu_hir_std, "Hybrid IR 60, 200 mA")]
    local hm
    for (i, (volume, title)) in enumerate(images)
        ax = Mke.Axis(
            fig[1, i + 1];
            title,
            subtitle = "Slice $(z_slice) of $(nz), $(round(dz_mm; digits = 3)) mm Thick",
            aspect = Mke.DataAspect(),
        )
        hm = Mke.heatmap!(
            ax,
            volume[:, :, z_slice];
            colormap = :grays,
            colorrange = (-200, 500),
        )
        Mke.hidedecorations!(ax)
        Mke.hidespines!(ax)
    end
    Mke.Colorbar(fig[1, 4], hm; label = "HU")
    Mke.colsize!(fig.layout, 1, Mke.Fixed(130))
    fig
end

# ╔═╡ 12000002-0000-4000-8000-000000000001
let
    worst(element) = minimum(
        100 * (r.fbp - r.theory) / abs(r.theory)
        for r in rods
        if r.insert.element == element
    )
    Markdown.parse(
        """
    ## 9. Verification

    Pass/fail against physics-derived expectations. The theory for each rod is its monoenergetic
    HU at the BHC reference energy, computed from the same XrayAttenuation data that drove the
    simulation. The tolerance is max(15 HU, 15 %): a water-only BHC maps water exactly, but a
    dense calcium or iodine rod hardens the beam beyond the water curve. That residual grows with
    density (down to $(round(worst("Calcium"); digits = 1)) % for calcium and
    $(round(worst("Iodine"); digits = 1)) % for iodine, below), and a single-energy scan cannot
    resolve it; quantitative high-Z imaging is what the dual-energy and photon-counting VMI
    notebooks (03, 04) are for. Each rod must also read the same at both doses and in both
    reconstructions.
    """,
    )
end

# ╔═╡ 166f5f5a-d6eb-4b4e-b223-aad4c431c9b9
# Every rod's mean HU (central slice, its label eroded by 2 pixels) beside its theoretical HU
rods = let
    labels, k = water_roi.labels, water_roi.kmid
    refE, μw = bhc.reference_energy_keV, bhc.μ_water_ref
    nx, ny, _ = size(labels)

    # in-plane-eroded mask of one label on the central slice
    function eroded(lab; r = 1)
        m = falses(nx, ny)
        for j in (1 + r):(ny - r), i in (1 + r):(nx - r)
            m[i, j] = all(labels[i + di, j + dj, k] == lab for dj in -r:r, di in -r:r)
        end
        return m
    end
    slc(v) = @view v[:, :, k]

    out = NamedTuple[]
    for lab in 1:(length(phantom_cpu.materials) - 1)
        # material entry L + 1 is label L
        lab == Int(BS.REGION_SOLID_WATER) && continue
        m = eroded(UInt8(lab); r = 2)
        count(m) < 20 && continue
        mat = phantom_cpu.materials[lab + 1]
        theory = 1000 * (BS.compute_μ_at_energy(mat, refE) - μw) / μw
        fbp = mean(slc(hu_fbp_std)[m])
        hir = mean(slc(hu_hir_std)[m])
        fbp_low = mean(slc(hu_fbp_low)[m])
        tol = max(15.0, 0.15 * abs(theory))
        ok = abs(fbp - theory) <= tol &&
            abs(fbp - fbp_low) <= max(10.0, 0.03 * abs(theory)) &&
            abs(hir - fbp) <= max(10.0, 0.03 * abs(theory))
        push!(
            out,
            (
                material = mat,
                insert = gammex_insert(mat),
                theory = theory,
                fbp = fbp,
                hir = hir,
                fbp_low = fbp_low,
                tol = tol,
                ok = ok,
            ),
        )
    end
    out
end;

# ╔═╡ 12000002-0000-4000-8000-000000000002
verification = let
    checks = NamedTuple[]
    addcheck(name, val, lo, hi) = push!(
        checks,
        (
            name = name,
            value = round(val; digits = 2),
            lo = lo,
            hi = hi,
            pass = lo <= val <= hi,
        ),
    )

    addcheck("water mean, FBP 200 mA (HU)", noise.fbp_std.mean, -6.0, 6.0)
    addcheck("water mean, HIR 200 mA (HU)", noise.hir_std.mean, -6.0, 6.0)
    addcheck("FBP noise ratio 50 / 200 mA", noise.fbp_low.σ / noise.fbp_std.σ, 1.7, 2.4)
    addcheck(
        "HIR noise < FBP noise, 200 mA (ratio)",
        noise.hir_std.σ / noise.fbp_std.σ,
        0.0,
        0.95,
    )
    addcheck(
        "CTDIvol ratio 50 / 200 mA",
        sim_low.dose.ctdi_vol_mGy / sim_std.dose.ctdi_vol_mGy,
        0.2499,
        0.2501,
    )

    # Radial non-uniformity QA (non-mutating). On a uniform water cylinder both numbers are ≈ 0
    # after the full-spectrum BHC; on the Gammex the radial fit also picks up the hardening
    # around the dense rod rings (about 12 HU), so the gate is wider than a uniform phantom
    # would need.
    cup = BS.measure_radial_cupping(hu_fbp_std; fov_cm = sim_std.geom.fov[1])
    addcheck("radial cupping QA, FBP (HU, worst slice)", cup.cup_hu, 0.0, 15.0)
    addcheck("DC offset QA, FBP (absolute HU, worst slice)", abs(cup.dc_hu), 0.0, 6.0)

    n_rod = length(rods)
    n_pass = count(r -> r.ok, rods)
    addcheck("rods passing (theory, dose- and algorithm-invariant)", n_pass, n_rod, n_rod)

    (checks = checks, n_ok = count(c -> c.pass, checks), n = length(checks))
end;

# ╔═╡ 91b39992-1ec5-4681-bf48-9f9146d272ce
let
    verdict = verification.n_ok == verification.n ?
        "✅ VERIFICATION PASS ($(verification.n_ok)/$(verification.n))" :
        "❌ VERIFICATION FAIL ($(verification.n_ok)/$(verification.n))"
    rows = join(
        [
            "| $(c.name) | $(c.value) | [$(c.lo), $(c.hi)] | $(c.pass ? "✅" : "❌") |"
            for c in verification.checks
        ],
        "\n",
    )
    rod_row(
        r,
    ) = "| $(r.material.name) | $(round(Int, r.theory)) | $(round(Int, r.fbp)) | " *
        "$(round(r.fbp - r.theory; digits = 0)) ($(round(100 * (r.fbp - r.theory) / max(abs(r.theory), 1.0); digits = 1)) %) | " *
        "$(round(Int, r.fbp_low)) | $(round(Int, r.hir)) | $(r.ok ? "✅" : "❌") |"
    Markdown.parse(
        """
    #### $(verdict)

    | Check | Value | Expected | Pass |
    |:--|--:|:--|:--|
    $(rows)

    **Per rod, central slice.** Theory is the monoenergetic HU at $(round(bhc.reference_energy_keV; digits = 1)) keV.

    | Rod | Theory | FBP 200 mA | FBP − Theory | FBP 50 mA | HIR 200 mA | Pass |
    |:--|--:|--:|--:|--:|--:|:--|
    $(join([rod_row(r) for r in rods], "\n"))
    """,
    )
end

# ╔═╡ 183c08aa-31d1-46be-9489-d759e6044d9a
let
    Mke.set_theme!(THEME)
    mid = size(phantom_cpu.mask, 3) ÷ 2
    half = size(phantom_cpu.mask, 1) * phantom_cpu.voxel_size[1] / 2

    fig = Mke.Figure(size = (1400, 620))
    ax0 = Mke.Axis(
        fig[1, 1];
        title = "The Rods",
        subtitle = "Concentrations in mg/mL",
        aspect = Mke.DataAspect(),
    )
    Mke.hidedecorations!(ax0)
    Mke.hidespines!(ax0)
    draw_gammex!(
        ax0,
        phantom_cpu.mask[:, :, mid],
        phantom_cpu.materials,
        phantom_cpu.voxel_size[1],
    )
    Mke.limits!(ax0, -half, half, -half, half)

    ax = Mke.Axis(
        fig[1, 2];
        title = "Each Rod Against Its Theoretical HU",
        subtitle = "Band: the Tolerance, max(15 HU, 15 %)",
        xlabel = "Theoretical HU at $(round(bhc.reference_energy_keV; digits = 1)) keV",
        ylabel = "Measured − Theoretical (HU)",
    )
    order = sortperm([r.theory for r in rods])
    xs = [rods[i].theory for i in order]
    tol = [rods[i].tol for i in order]
    Mke.band!(ax, xs, -tol, tol; color = (COLORS.green, 0.12))
    Mke.hlines!(ax, [0.0]; color = COLORS.rule, linewidth = 1.5)

    colour(r) = r.insert.element == "Calcium" ? COLORS.red : COLORS.green
    series = [
        (r -> r.fbp, :circle, "FBP, 200 mA", true, 11),
        (r -> r.fbp_low, :rect, "FBP, 50 mA", false, 19),
        (r -> r.hir, :diamond, "Hybrid IR 60, 200 mA", false, 27),
    ]
    for (value, marker, _, filled, size) in series
        Mke.scatter!(
            ax,
            [r.theory for r in rods],
            [value(r) - r.theory for r in rods];
            marker,
            markersize = size,
            color = filled ? colour.(rods) : Mke.RGBAf(0, 0, 0, 0),
            strokecolor = colour.(rods),
            strokewidth = filled ? 0 : 2,
        )
    end
    Mke.axislegend(
        ax,
        [
            Mke.MarkerElement(;
                marker,
                color = filled ? COLORS.ink : Mke.RGBAf(0, 0, 0, 0),
                strokecolor = COLORS.ink,
                strokewidth = 2,
                markersize = 14,
            )
            for (_, marker, _, filled, _) in series
        ],
        [label for (_, _, label, _, _) in series];
        position = :lb,
    )
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    fig
end

# ╔═╡ 11000001-0000-4000-8000-000000000001
let
    d_s, d_l = sim_std.dose, sim_low.dose
    reduction = round(100 * (1 - noise.hir_std.σ / noise.fbp_std.σ); digits = 1)
    Markdown.parse(
        """
    ## Summary

    | | 200 mA | 50 mA |
    |:--|--:|--:|
    | CTDIvol (mGy) | $(round(d_s.ctdi_vol_mGy; digits = 2)) | $(round(d_l.ctdi_vol_mGy; digits = 2)) |
    | Water, FBP: mean ± σ (HU) | $(round(noise.fbp_std.mean; digits = 1)) ± $(round(noise.fbp_std.σ; digits = 1)) | $(round(noise.fbp_low.mean; digits = 1)) ± $(round(noise.fbp_low.σ; digits = 1)) |
    | Water, Hybrid IR 60: mean ± σ (HU) | $(round(noise.hir_std.mean; digits = 1)) ± $(round(noise.hir_std.σ; digits = 1)) | $(round(noise.hir_low.mean; digits = 1)) ± $(round(noise.hir_low.σ; digits = 1)) |

    - **Five structs describe a scan.** `Phantom` (the object), `EICTScanner` / `PCCTScanner` (the
      hardware), `CTProtocol` (the acquisition), `SimOptions` (the physics, including the view
      integration of a turning gantry) and `ReconOptions` (the output grid). Changing the dose
      meant changing one `CTProtocol` and nothing else.
    - **Three calls produce an image.** `create_workspace` sets everything up once, `simulate!`
      runs the forward model and reports the dose, and `reconstruct!` runs FBP or Hybrid IR,
      depending on the workspace it is given. Hybrid IR lowered the 200 mA water noise by
      $(reduction) %; quartering the dose raised the FBP noise
      $(round(noise.fbp_low.σ / noise.fbp_std.σ; digits = 2))×.
    - **A noise draw can reuse the projection**, $(round(noise_draws.t_full / noise_draws.t_reuse; digits = 1))×
      faster than a full simulation and $(noise_draws.identical ? "bit-identical to it" : "not identical to it").
    - **Every scan reports its dose**, from a Monte Carlo of the simulated beam in the IEC body
      phantom.
    - **The standard reconstruction chain** is `calibrate_bhc_water` → `apply_bhc_water` →
      `reconstruct!` → `to_hounsfield`; `measure_radial_cupping` is a QA measurement, not a
      correction. $(count(r -> r.ok, rods)) of $(length(rods)) rods sit within their tolerance of theory, and
      $(verification.n_ok) of $(verification.n) checks pass.
    - **Where it runs** is decided by the phantom mask's array type (here `$(GPU_BACKEND.name)`);
      the same notebook runs on CUDA, Metal, ROCm, oneAPI or the CPU.

    Every other notebook reuses this pattern: build the structs, create a workspace, `simulate!`,
    `reconstruct!`.
    """,
    )
end

# ╔═╡ Cell order:
# ╟─01000001-0000-4000-8000-000000000001
# ╟─cb84df00-b571-498a-87ae-f805fa7c2e6b
# ╟─875b5c6f-425e-4e2e-b5ad-8f6842992502
# ╟─25c7eadb-a43d-49b5-bf92-bc3a131bad5b
# ╟─01000002-0000-4000-8000-000000000001
# ╟─01000003-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000002
# ╟─01000003-0000-4000-8000-000000000005
# ╟─01000003-0000-4000-8000-000000000004
# ╠═01000003-0000-4000-8000-000000000002
# ╟─01000003-0000-4000-8000-000000000003
# ╟─886270ac-b0c1-4c77-b218-3bb67c8bee20
# ╟─01000004-0000-4000-8000-000000000001
# ╠═01000005-0000-4000-8000-000000000001
# ╟─01000007-0000-4000-8000-000000000001
# ╟─2f9a1c3e-0d7b-4c51-9a3e-6b2f8d1e4a70
# ╠═694771e5-23e0-4450-afcf-881e4e62dc6a
# ╠═bc83aa9a-649e-44ec-b079-0d8cad213f39
# ╠═06dbb07a-4db9-41a9-9c32-397cd8977cbb
# ╠═b852ff0b-a12c-4d33-b0c0-f6c47841df59
# ╠═cc7a479c-361d-4f79-b587-b5cb92df08e9
# ╠═df70b269-0011-4839-9d9a-6da1db9101a0
# ╠═d5877142-69dc-489b-b5a4-829361dbfc2b
# ╠═0502aa42-5f36-4b12-a551-07b2e8dbca67
# ╠═9a37760b-ee4f-4e5c-ae88-bc78008b1944
# ╠═07a2c861-382c-4a40-8334-c620a03115c9
# ╠═98bca9b1-ce42-4b66-abd0-900b979c517b
# ╠═498ef857-4c4c-4feb-87fa-8873bbbe1806
# ╠═38f6008d-a7da-4024-8a9b-5a776bf528cf
# ╠═471c73a8-5ac6-4a25-b1bf-953dd4eca3c9
# ╠═9dc621c9-6668-4d13-a541-6841e46dfa48
# ╠═4e23ef56-a955-4b27-a13c-61aa1927e123
# ╠═dee92749-7bf9-4c42-98e8-2eba17913754
# ╠═f878b0bd-2f42-4cec-81bb-04c1332cb81e
# ╠═9c2ca6c0-055c-42ce-a407-97841e4f31d2
# ╠═06ba114f-b575-4b0e-8dd3-e951793faac4
# ╠═680ebee9-4490-47bf-89ee-3272f213d055
# ╟─02000000-0000-4000-8000-000000000000
# ╠═02000002-0000-4000-8000-000000000001
# ╟─02000003-0000-4000-8000-000000000001
# ╠═02000004-0000-4000-8000-000000000001
# ╟─02000005-0000-4000-8000-000000000001
# ╟─03000001-0000-4000-8000-000000000001
# ╠═03000002-0000-4000-8000-000000000001
# ╟─006e51e3-bb52-5e9f-9cde-8d1ec973b9bd
# ╟─950dd46f-2d1d-4e4a-bbab-f20de8500648
# ╟─03000003-0000-4000-8000-000000000001
# ╟─04000001-0000-4000-8000-000000000001
# ╠═04000002-0000-4000-8000-000000000001
# ╠═04000003-0000-4000-8000-000000000001
# ╟─a3de7eb8-1b94-4a25-8406-d6fc413aaed9
# ╟─04000004-0000-4000-8000-000000000001
# ╟─05000001-0000-4000-8000-000000000001
# ╟─8b606344-e54f-4566-8a43-728872a93ee9
# ╠═05000002-0000-4000-8000-000000000001
# ╟─05000003-0000-4000-8000-000000000001
# ╟─06000001-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000001
# ╟─07000000-0000-4000-8000-000000000000
# ╠═07000010-0000-4000-8000-000000000001
# ╠═07000011-0000-4000-8000-000000000001
# ╟─269d13d5-dc38-41c0-a6ed-97c6d01b10f3
# ╟─07000015-0000-4000-8000-000000000001
# ╟─08000001-0000-4000-8000-000000000001
# ╠═08000010-0000-4000-8000-000000000001
# ╠═08000011-0000-4000-8000-000000000001
# ╟─08000012-0000-4000-8000-000000000001
# ╟─08000020-0000-4000-8000-000000000001
# ╠═08000021-0000-4000-8000-000000000001
# ╟─08000022-0000-4000-8000-000000000001
# ╟─07000000-0000-4000-8000-000000000001
# ╠═09000006-0000-4000-8000-000000000001
# ╟─09000007-0000-4000-8000-000000000001
# ╟─09000001-0000-4000-8000-000000000001
# ╟─7a8a8143-539e-4884-8053-1b0af611dc7b
# ╟─ecb30d39-380e-478a-9172-048523b36d15
# ╠═09000002-0000-4000-8000-000000000001
# ╠═07000021-0000-4000-8000-000000000001
# ╠═09000011-0000-4000-8000-000000000001
# ╠═08000013-0000-4000-8000-000000000001
# ╠═09000021-0000-4000-8000-000000000001
# ╟─10000000-0000-4000-8000-000000000000
# ╠═10000005-0000-4000-8000-000000000001
# ╠═10000006-0000-4000-8000-000000000001
# ╟─0bf8cbc0-978c-49b0-bbe5-453b0ba93e65
# ╟─10000001-0000-4000-8000-000000000001
# ╟─10000002-0000-4000-8000-000000000001
# ╟─10000003-0000-4000-8000-000000000001
# ╟─12000003-0000-4000-8000-000000000001
# ╟─12000003-0000-4000-8000-000000000002
# ╟─12000003-0000-4000-8000-000000000003
# ╟─12000003-0000-4000-8000-000000000007
# ╠═12000003-0000-4000-8000-000000000004
# ╟─12000003-0000-4000-8000-000000000005
# ╟─12000003-0000-4000-8000-000000000006
# ╟─12000001-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000003
# ╟─12000001-0000-4000-8000-000000000004
# ╟─12000002-0000-4000-8000-000000000001
# ╠═166f5f5a-d6eb-4b4e-b223-aad4c431c9b9
# ╟─12000002-0000-4000-8000-000000000002
# ╟─91b39992-1ec5-4681-bf48-9f9146d272ce
# ╟─183c08aa-31d1-46be-9489-d759e6044d9a
# ╟─11000001-0000-4000-8000-000000000001
