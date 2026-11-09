### A Pluto.jl notebook ###
# v0.4.2

using Markdown
using InteractiveUtils

# ╔═╡ 12000001-0000-4000-8000-000000000010
md"""
# Siemens SOMATOM Definition Flash: Dual Power and Dual Energy

**The second-generation dual-source scanner as a complete simulated system, its two tubes and
two UFC detectors 95° apart, run in both of its signature modes on the Gammex 472: a regular
dual-power scan (120 kVp on both tubes) and a dual-energy scan (80 / Sn140 kVp) read as the mixed
image and as VMIs.**

The scanner, its physics switches, the dual-energy protocol and the VMI chain are the Flash of the
basis-spectral-denoising study, carried over unchanged, with the Flash's own Geant4 detector
table. Every published number comes from the sourced
[SOMATOM Definition Flash entry of the scanners page](../../scanners/#somatom-definition-flash);
an automated verification closes the notebook.
"""

# ╔═╡ 117658a2-649f-4f7a-9441-8cb8bf6e5a64
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scanner", "Two Modes"],
        [
            "Gammex 472: Calcium and Iodine Rods",
            "Definition Flash: Two Tubes, 95° Apart",
            "Dual Power and Dual Energy",
        ];
        limits = [(-19, 19, -19, 19), (-7.6, 7.6, -7.6, 7.6), (-0.5, 10.5, -3.4, 3.4)],
    )
    gammex!(axes[1]; dark = true)

    # the kit labels a tube outward from its focal spot, which at 185° covers the spot, so tube B's
    # label is drawn above it
    gantry!(
        axes[2];
        angles = (90.0, 185.0),
        detector_colors = (COLORS.cyan, COLORS.violet),
        labels = ("Tube A", ""),
        object = false,
    )
    gammex!(axes[2]; scale = 1.9 / GAMMEX.body_r, dark = true)
    Mke.text!(
        axes[2],
        5.4 * cosd(185.0),
        5.4 * sind(185.0) + 0.75;
        text = "Tube B",
        align = (:center, :bottom),
        fontsize = 17,
        font = :bold,
        color = COLORS.night_ink,
    )

    box(c, title, subtitle, accent; w = 4.6) = node!(
        axes[3],
        c,
        title;
        subtitle,
        w,
        h = 1.25,
        accent,
        fill = COLORS.night_card,
        textcolor = COLORS.night_ink,
        subcolor = COLORS.night_muted,
    )
    box((5.0, 2.2), "One Gantry", "Two Tubes, Two UFC Detectors", COLORS.cyan; w = 6.0)
    box((2.4, -1.6), "Dual Power", "120 / 120 kVp", COLORS.green)
    box((7.6, -1.6), "Dual Energy", "80 / Sn140 kVp", COLORS.violet)
    for x in (2.4, 7.6)
        Mke.lines!(
            axes[3],
            Mke.Point2f[(5.0, 1.55), (5.0, 0.3), (x, 0.3), (x, -0.4)];
            color = COLORS.night_muted,
            linewidth = 2.5,
        )
        arrow!(axes[3], (x, -0.4), (x, -0.95); color = COLORS.night_muted)
    end
    fig
end

# ╔═╡ b0517860-9285-4fa2-9032-68d98332bb37
md"""
The regular mode checks accuracy and the independence of the two tube chains (averaging them cuts
the noise by √2); the dual-energy mode checks the mixed image and the full VMI chain. Every result
is scored by the checks at the end:
"""

# ╔═╡ 96e9486c-0785-4c59-a2a4-696981dd9661
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 600))
    nodes = Dict(
        :phantom => (0.0, 2.6, "Gammex 472", "The Phantom"),
        :scanner => (0.0, 1.3, "Definition Flash", "Flash UFC Geant4 Table"),
        :protocols => (0.0, 0.0, "Three Protocols", "120 kVp; 80 / Sn140 kVp"),
        :simulate => (3.9, 1.3, "simulate!", "One Projection per Tube", COLORS.cyan),
        :calibration => (3.9, -1.3, "Calibration Draw", "Fixes the Pair", COLORS.beam),
        :regular => (7.8, 2.6, "BHC + FDK", "Regular, Both Tubes", COLORS.cyan),
        :de => (7.8, 1.3, "BHC + FDK", "Dual Energy, Each Tube", COLORS.cyan),
        :vmi => (7.8, 0.0, "vmi_pipeline", "SpectralHYPR", COLORS.cyan),
        :power => (11.7, 2.6, "Dual Power", "(A + B) / 2", COLORS.green),
        :mixed => (11.7, 1.3, "Mixed Image", "0.5 A + 0.5 B", COLORS.green),
        :vmis => (11.7, 0.0, "VMIs", "50 / 70 / 100 / 140 keV", COLORS.green),
        :checks => (15.6, 1.3, "Verification", "Pass / Fail Gates", COLORS.red),
    )
    edges = [
        :phantom => :simulate,
        :scanner => :simulate,
        :protocols => :simulate,
        :simulate => :regular,
        :simulate => :de,
        :simulate => :vmi,
        :simulate => :calibration,
        :calibration => :vmi,
        :regular => :power,
        :de => :mixed,
        :vmi => :vmis,
        :power => :checks,
        :mixed => :checks,
        :vmis => :checks,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.7, 17.3, -2.0, 3.3))
    fig
end

# ╔═╡ 12000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 12000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 12000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 12000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 12000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 12000001-0000-4000-8000-000000000031
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 12000001-0000-4000-8000-000000000032
import PlutoUI

# ╔═╡ 12000001-0000-4000-8000-000000000033
PlutoUI.TableOfContents()

# ╔═╡ 12000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    # the backend array type, directly: MtlArray / CuArray / ROCArray
    AT = GPUSelect.Storage()
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 12000001-0000-4000-8000-000000000050
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ c6d3297b-beb3-4190-97dc-18747624fa34
md"""
## Figure Style

BasisSimulator's figure kit (colours, theme, flow charts, the gantry and the locator band),
shared by every example notebook, then this notebook's own drawings: the Gammex 472, a region on
it, and one tube of the gantry to scale.
"""

# ╔═╡ 14b7f814-8c4b-454a-aabd-9b3f4044b33c
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

# ╔═╡ 5c4ece81-ba40-40e1-8620-3678e69bd43e
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

# ╔═╡ 66da3212-c888-4531-923f-6c90a2595573
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

# ╔═╡ 37690342-2920-4aac-89eb-e17113140f4c
const HU_WINDOW = (-200, 500); # every HU image's display window: soft tissue

# ╔═╡ 5295bbac-514c-4a8e-9fb8-18b31c28c0ee
"""
An image panel at `position`: `slice` on a square axis without ticks, in `colormap` over
`colorrange`. Returns the axis.
"""
function slice_panel!(
    position,
    slice;
    title = "",
    colormap = :grays,
    colorrange = HU_WINDOW,
)
    ax = Mke.Axis(position; title, aspect = Mke.DataAspect())
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    Mke.heatmap!(ax, slice; colormap, colorrange)
    return ax
end

# ╔═╡ dd8cb654-bc25-45c7-b74f-1024ad950ce3
# The Gammex 472 as the drawings show it, measured from the phantom's own labels (cm from the
# centre): the body, and the centres of the calcium (labels 10–16) and iodine (20–26) rods
GAMMEX = let
    mask = phantom_cpu.mask[:, :, 1]
    a = phantom_cpu.voxel_size[1]
    n = size(mask, 1)
    function centre(label)
        idx = findall(==(UInt8(label)), mask)
        return (
            (sum(c -> c[1], idx) / length(idx) - (n + 1) / 2) * a,
            (sum(c -> c[2], idx) / length(idx) - (n + 1) / 2) * a,
        )
    end
    (
        body_r = sqrt(count(!=(0), mask) * a^2 / π),
        rod_r = sqrt(count(==(UInt8(10)), mask) * a^2 / π),
        calcium = [centre(l) for l in 10:16],
        iodine = [centre(l) for l in 20:26],
    )
end;

# ╔═╡ a41b173b-f928-47e6-ba27-b5bf181124ed
"""
Draw the Gammex 472 seen along z on `ax` (cm): the solid-water body, the seven calcium rods
(inner ring) and the seven iodine rods (outer ring). `calcium` and `iodine` colour the rods (one
colour, or one per rod); `scale` shrinks the drawing (into a gantry, say); `dark` draws it for the
locator band.
"""
function gammex!(
    ax,
    g = GAMMEX;
    calcium = COLORS.red,
    iodine = COLORS.green,
    scale = 1.0,
    dark = false,
)
    ink = dark ? COLORS.night_ink : COLORS.ink
    Mke.poly!(
        ax,
        Mke.Circle(Mke.Point2f(0, 0), scale * g.body_r);
        color = (COLORS.blue, dark ? 0.55 : 0.18),
        strokecolor = (ink, 0.6),
        strokewidth = 1.5,
    )
    for (centres, colors) in ((g.calcium, calcium), (g.iodine, iodine))
        for (k, c) in enumerate(centres)
            color = colors isa AbstractVector ? colors[k] : colors
            Mke.poly!(
                ax,
                Mke.Circle(Mke.Point2f(scale .* c), scale * g.rod_r);
                color,
                strokecolor = (ink, 0.5),
                strokewidth = 1,
            )
        end
    end
    return ax
end

# ╔═╡ fab85396-95ae-4bad-bb1d-74eee0da3059
"""
Shade the in-plane region `mask` (a Bool matrix on a `fov_cm` grid centred on the isocentre) on a
drawing in cm.
"""
function roi!(ax, mask, fov_cm; color = COLORS.violet, alpha = 0.75)
    n = size(mask, 1)
    xs = ((1:n) .- (n + 1) / 2) .* (fov_cm / n)
    Mke.heatmap!(
        ax,
        xs,
        xs,
        [m ? 1.0f0 : NaN32 for m in mask];
        colormap = [color, color],
        alpha,
        nan_color = :transparent,
    )
    return ax
end

# ╔═╡ d1d2e0b0-8f50-4be1-bd48-ca55bcc2efb4
"""
One tube of a CT gantry to scale (cm, the isocentre at the origin): the focal spot at `sid` and
angle `θ` (degrees), the fan that covers a field of view of diameter `fov`, and its detector arc at
`sdd` from the focal spot. (The kit's `gantry!` gives every tube one fan; the Flash's two differ.)
"""
function tube_fan!(
    ax,
    θ;
    fov,
    sid = 59.5,
    sdd = 108.56,
    color = COLORS.cyan,
    label = "Tube",
    dark = false,
)
    ink = dark ? COLORS.night_ink : COLORS.ink
    θ = deg2rad(θ)
    s = Mke.Point2f(sid * cos(θ), sid * sin(θ))
    half = asin(fov / 2 / sid)
    edge = Mke.Point2f[
        s + sdd * Mke.Point2f(cos(θ + π + φ), sin(θ + π + φ))
        for φ in range(-half, half; length = 40)
    ]
    Mke.poly!(ax, vcat([s], edge); color = (COLORS.beam, 0.2), strokewidth = 0)
    Mke.lines!(ax, edge; color, linewidth = 9)
    Mke.scatter!(ax, [s]; marker = :rect, markersize = 26, color = ink)
    Mke.scatter!(ax, [s]; marker = :circle, markersize = 9, color = COLORS.beam)
    Mke.text!(
        ax,
        s * 1.16;
        text = label,
        align = (:center, :center),
        fontsize = 17,
        font = :bold,
        color = ink,
    )
    return ax
end

# ╔═╡ 12000002-0000-4000-8000-000000000001
md"""
## 1. The Flash's UFC Detector Table

The Flash uses the same UFC Gd₂O₂S ceramic as the Force. Its table,
`BS.UFC_FLASH_MC_EFFICIENCY_LUT` (`detector_material = :ufc_flash`), is its own Geant4 run
(BasisMonteCarlo notebook `05_siemens_flash.jl`, the Flash's pitch and 1-D grid, 100 000 photons
per keV) at 1.4 mm, the only UFC thickness Siemens quotes; the Flash's depth is unpublished. The
`efficiency` is the energy a dexel collects per photon divided by the photon's energy; `dqe0`
adds the spread of the collected energy (the Swank factor) to the absorption.

1. **The Gd K-edge, 50.24 keV**: the efficiency falls from 0.936 to 0.791 between 50 and 51 keV,
   as the 43 keV Gd K x ray freed above the edge partly escapes.
2. **Gd L-edge structure near 7–8 keV** (L₃ 7.24, L₂ 7.93, L₁ 8.38 keV).
3. **The high-energy roll-off**, 0.858 at 100 keV to 0.610 at 140 keV, from transmission through
   1.4 mm and Compton escape.

At the same depth the Flash and Force tables differ only by their grids' geometry, by less than
0.5 % at every energy. BasisSimulator v0.15.1's MCNP table implied a thinner crystal (0.588 at
140 keV against the Force's 0.816).
"""

# ╔═╡ 12000002-0000-4000-8000-000000000030
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 980))

    # the three fates of an x ray in 1.4 mm of UFC
    sketch = canvas(fig[1, 1], (-3, 36, -0.4, 6.4))
    scenes = (
        (6.0, "40 keV: Absorbed", :stop),
        (18.0, "51 keV: The Gd K X Ray Escapes", :escape),
        (30.0, "140 keV: Passes Through", :through),
    )
    for (x, title, fate) in scenes
        Mke.poly!(
            sketch,
            Mke.Rect2f(x - 4.5, 1.0, 9.0, 1.6);
            color = COLORS.cyan_soft,
            strokecolor = COLORS.cyan,
            strokewidth = 1.5,
        )
        Mke.text!(
            sketch,
            x,
            6.2;
            text = title,
            align = (:center, :top),
            fontsize = 19,
            font = :bold,
        )
        stop = fate === :through ? 0.0 : 1.9
        arrow!(sketch, (x - 1.0, 4.9), (x - 1.0, stop); color = COLORS.beam, linewidth = 3)
        if fate !== :through
            Mke.scatter!(
                sketch,
                [Mke.Point2f(x - 1.0, 1.9)];
                marker = :star5,
                markersize = 22,
                color = COLORS.beam,
            )
        end
        if fate === :escape
            arrow!(
                sketch,
                (x - 0.8, 2.0),
                (x + 3.2, 4.6);
                color = COLORS.red,
                linewidth = 3,
                linestyle = :dash,
            )
            Mke.text!(
                sketch,
                x + 3.4,
                4.6;
                text = "43 keV",
                align = (:left, :center),
                fontsize = 16,
                color = COLORS.red,
            )
        end
    end
    Mke.text!(
        sketch,
        1.2,
        1.8;
        text = "UFC\n1.4 mm",
        align = (:right, :center),
        fontsize = 15,
        color = COLORS.muted,
    )

    flash = BS.UFC_FLASH_MC_EFFICIENCY_LUT
    force = BS.UFC_MC_EFFICIENCY_LUT
    gemstone = BS.GEMSTONE_MC_EFFICIENCY_LUT
    ax = Mke.Axis(
        fig[2, 1];
        title = "The Energy Collected per Photon, Geant4",
        subtitle = "The Flash's UFC, the Force's UFC (Same Ceramic, Same Depth) and GE's Gemstone",
        xlabel = "Photon Energy (keV)",
        ylabel = "Collected Fraction η(E)",
    )
    Mke.lines!(
        ax,
        flash.energies,
        flash.efficiency;
        color = COLORS.violet,
        linewidth = 3.5,
        label = "Flash UFC (UFC_FLASH_MC_EFFICIENCY_LUT)",
    )
    Mke.lines!(
        ax,
        force.energies,
        force.efficiency;
        color = COLORS.ink,
        linewidth = 2,
        linestyle = :dash,
        label = "Force UFC (Notebook 09)",
    )
    Mke.lines!(
        ax,
        gemstone.energies,
        gemstone.efficiency;
        color = COLORS.blue,
        linewidth = 3,
        label = "Gemstone (GE Revolution Apex)",
    )
    Mke.vlines!(ax, [50.24]; color = COLORS.beam, linestyle = :dot, linewidth = 2)
    Mke.text!(
        ax,
        51.5,
        0.56;
        text = "Gd K-Edge, 50.24 keV",
        align = (:left, :bottom),
        fontsize = 16,
        color = COLORS.beam,
    )
    # the 140 keV values, marked on both UFC curves
    Mke.scatter!(
        ax,
        [140.0, 140.0],
        [flash.efficiency[end], force.efficiency[end]];
        color = [COLORS.violet, COLORS.ink],
        markersize = 12,
    )
    Mke.text!(
        ax,
        138.0,
        flash.efficiency[end] - 0.02;
        text = "140 keV: Flash $(round(flash.efficiency[end]; digits = 3)), Force $(round(force.efficiency[end]; digits = 3)) " *
            "($(round(100 * (flash.efficiency[end] / force.efficiency[end] - 1); digits = 1)) %)",
        align = (:right, :top),
        fontsize = 16,
        color = COLORS.violet,
    )
    Mke.ylims!(ax, 0.5, 1.02)
    Mke.Legend(fig[3, 1], ax; orientation = :horizontal)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.27))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_lut_comparison.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000003-0000-4000-8000-000000000001
md"""
## 2. The Phantom: Gammex 472

A 33 cm solid-water cylinder with seven calcium rods (50–600 mg/mL) on an inner ring and seven
iodine rods (2–20 mg/mL) on an outer ring, voxelized at 512² over 35 cm, 16 slices over 1 cm.
`roi_labels` (section 5) draws the same phantom on the 33 cm reconstruction grid, where every ROI
is measured.
"""

# ╔═╡ 12000003-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 12000003-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 12000004-0000-4000-8000-000000000001
md"""
## 3. The Scanner: SOMATOM Definition Flash

Second-generation dual source (2008, FDA K082220; Stellar detector K113342): two STRATON MX P
tubes and two UFC detectors, 95° apart in one gantry. Detector A covers a 50 cm field, detector B
33 cm. `flash_scanner` is the documented constant `SOMATOM_DEFINITION_FLASH`, tube and detector
A; `SOMATOM_DEFINITION_FLASH_B` is detector B's 480 columns.

<details><summary>Every number and its source</summary>

| Parameter | Value | Source |
|-----------|-------|--------|
| Source–isocenter (SID) | 595.0 mm | AAPM LDCT-PD / DICOM-CT-PD (acquired on a Flash) |
| Source–detector (SDD)  | 1085.6 mm | LDCT-PD |
| Tube A–B angular offset | 95° | Petersilka 2008; technical reviews |
| Detector A channels | 736 (50 cm FOV) | Siemens datasheet 47,104 el / 64 rows; NICE MIB54; LDCT-PD |
| Detector B channels | 480 (33 cm FOV) | Siemens datasheet 30,720 el / 64 rows |
| Rows | 64 × 0.6 mm at iso (38.4 mm z-coverage) | Siemens datasheet; FDA K173630 |
| Column pitch at iso | 0.70473 mm (1.2858 mm at detector / 1.8245× mag) | LDCT-PD |
| Scintillator | UFC Gd₂O₂S:Pr,Ce → **`:ufc_flash` Geant4 table** | Siemens datasheet; BasisMonteCarlo notebook 05 |
| Tube (STRATON MX P) | 2 × 100 kW, 80/100/120/140 kV, ≤800 mA each | Siemens datasheet |
| Focal spots (IEC 60336) | 0.7 × 0.7 mm and 0.9 × 1.1 mm, **anode 7°** | Siemens datasheet |
| Flat filtration | **8.4 mm Al equivalent** (6.8 tube + 1.6 BLD) | Siemens datasheet |
| Tin filter | **0.4 mm Sn** on tube B (Selective Photon Shield, DE mode) | Primak AJR 2010 |
| Rotation | 0.28 s min (0.5 s routine) | Siemens datasheet |
| Projections | 1152 / rotation per focal-spot position (2304 with z-FFS) | Siemens datasheet; AAPM LDCT |
</details>

<details><summary>Modelling assumptions where no public source exists</summary>

- **Anode angle in the spectrum model**: the Flash's anode is a published 7°, but the bundled IPEM
  spectra come only at 8° and 10°, so the spectrum is read at the nearest, **8°**. The scanner's
  `target_angle` keeps the true 7.0° (the heel effect would use it; it is off here).
- **Bowtie**: Siemens' form filter is unpublished, so CatSim's **large-body** profile stands in (as
  in notebooks 04, 08 and 09).
- **Flat-filter composition**: only the **Al equivalent**, 8.4 mm, is published, so it is modelled
  as 8.4 mm of aluminium. Unlike the Force's assumed Al + Ti stack (notebook 09), this figure is a
  measured datasheet value.
- **Scintillator depth 1.4 mm, fill factor 0.9**: proprietary. 1.4 mm is the only UFC thickness
  Siemens quotes and the depth of the Flash's Geant4 table. The fill factor scales the quanta a
  dexel collects, so it sets the noise, and cancels in the air calibration.
- **Electronic noise 1500 e⁻** (about 2.1 photon equivalents at 70 keV with the 10 e⁻/keV gain),
  modelling the Stellar (K113342) build. Siemens publishes no absolute figure for either build, but
  zero would be wrong: the DAS floor enters the counts before the log and dominates in photon
  starvation (low dose, 80 kVp, dense anatomy), so `simulate!` propagates it
  (`add_system_noise_floor!` in `src/api/driver.jl`). It is about 40 % of the conventional-DAS
  figure (about 3500 e⁻ at this gain, the class of the GE Apex's 5000 e⁻ at 15 e⁻/keV), for
  TrueSignal's integrated ASIC; for the pre-2011 Flash (K082220, discrete photodiode and ASIC) set
  `electronic_noise = 3500`. At this notebook's well-exposed protocols it adds 1 % or less to the
  image σ; it matters when the model is reused at low dose.
- **Tube B's z-offset**: unpublished for the Flash (the Force's is about 0.88 mm), so not modelled.
  The Gammex is z-invariant, so an offset would change nothing; notebook 09 carries the mechanism.
- **z flying focal spot not modelled**: 1152 views per rotation, one focal-spot position, half of
  the 2304 interleaved readings.
</details>

<details><summary>Two tubes as two co-registered scans</summary>

As notebook 03 models rapid kVp switching and notebook 09 the Force, the Flash's two tubes are
**two `CTProtocol`s run on one `EICTScanner`**, each with its own protocol (kVp, mA, filtration)
and its own **independent noise seed**: two physically separate tube-detector chains must not share
a noise realization, which is what section 6's √2 check tests.

- **Detector arc**: tube B gets detector A's 736-column arc, so the low- and high-energy sinograms
  are co-registered ray by ray. The 33 cm Gammex fits inside detector B's real 33.4 cm field
  (tighter than the Force's 35.5 cm: the Flash's B fan is the clinical dual-energy field limit).
- **The 95° offset**: both scans cover a full rotation on the same angle grid; a constant angular
  offset does nothing to a full-rotation axial scan.
- **Collimation**: the Flash reads 2 × 128 × 0.6 mm in dual-energy mode; this notebook uses
  4.8 mm (8 × 0.6 mm), the collimation of the basis-spectral-denoising acquisition and the
  thin-collimation equivalent that fits the 1 cm Gammex.
</details>
"""

# ╔═╡ 6cf27931-9dad-4fd6-afad-aa450e45f4e6
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 760))
    ax = canvas(fig[1, 1], (-75, 75, -72, 72))
    ax.title = "The Definition Flash, to Scale"
    Mke.poly!(ax, Mke.Circle(Mke.Point2f(0, 0), 68.0); color = COLORS.card)
    Mke.poly!(ax, Mke.Circle(Mke.Point2f(0, 0), 39.0); color = COLORS.paper)
    tube_fan!(ax, 90.0; fov = 50.0, color = COLORS.cyan, label = "Tube A")
    tube_fan!(ax, 185.0; fov = 33.0, color = COLORS.violet, label = "Tube B")
    gammex!(ax)
    fields = ((25.0, "Field of A, 50 cm", -25.6), (16.5, "Field of B, 33 cm", 19.0))
    for (r, text, y) in fields
        Mke.lines!(
            ax,
            [Mke.Point2f(r * cos(φ), r * sin(φ)) for φ in range(0, 2π; length = 120)];
            color = COLORS.muted,
            linestyle = :dash,
            linewidth = 1.5,
        )
        Mke.text!(ax, 0, y; text, align = (:center, y < 0 ? :top : :bottom), fontsize = 15)
    end

    spec = canvas(fig[1, 2], (0, 10, 0, 10))
    rows = [
        ("Focal Spot to Isocentre", "59.5 cm"),
        ("Focal Spot to Detector", "108.56 cm"),
        ("Tube B Behind Tube A", "95°"),
        ("Detector A", "736 Columns, 50 cm"),
        ("Detector B", "480 Columns, 33 cm"),
        ("Rows", "64 × 0.6 mm"),
        ("Flat Filter", "8.4 mm Al Equivalent"),
        ("Scintillator", "UFC Gd₂O₂S, 1.4 mm"),
    ]
    for (k, (name, value)) in enumerate(rows)
        y = 9.0 - 1.05 * (k - 1)
        Mke.text!(spec, 0.2, y; text = name, align = (:left, :center), color = COLORS.muted)
        Mke.text!(spec, 9.8, y; text = value, align = (:right, :center), font = :bold)
    end
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.38))
    fig
end

# ╔═╡ 12000004-0000-4000-8000-000000000010
# Siemens SOMATOM Definition Flash, tube and detector A. Both tubes run on detector A's arc so the
# two channels stay co-registered ray by ray (see above).
flash_scanner = BS.SOMATOM_DEFINITION_FLASH;

# ╔═╡ a4f39820-16e5-5809-b70a-5b3f64c6fb38
# Every keyword of `flash_scanner`: the documented constant `SOMATOM_DEFINITION_FLASH`, with any field this notebook changes marked
let s = flash_scanner, constant = BS.SOMATOM_DEFINITION_FLASH
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
        **`flash_scanner` is `$(nameof(typeof(s)))`, from the documented constant `SOMATOM_DEFINITION_FLASH`.** Its keywords
        and their values (units and sources: the constant's docstring, `?BS.SOMATOM_DEFINITION_FLASH`, and the Scanners
        page). A variant is the same struct with keywords replaced,
        `$(nameof(typeof(s)))(BS.SOMATOM_DEFINITION_FLASH; field = value)`.

        | Keyword | Value | |
        |---|---|---|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 12000006-0000-4000-8000-000000000001
md"""
## 4. Physics Switches

`flash_opts` are the Flash's switches in the basis-spectral-denoising study, verbatim: every
effect of the EICT forward model on, except the heel effect, and view integration.

- **Detector response**: the scanner reads the Flash's Geant4 table; every energy is weighted by
  its signal per photon ``m_1 = E\,\mathrm{efficiency}(E)``, and the noise is compound Poisson with
  ``m_2 = m_1^2/\mathrm{dqe0}``, so its variance per unit signal rises as the beam hardens.
- **Heel effect off**, which keeps the forward spectral model equal to the response the spectral
  basis inverts (at 4.8 mm collimation the heel is a small row-direction effect).
- **View integration**, `view_samples = 5`: a detector integrates while the gantry turns, so each
  view reads the object averaged over the arc it sweeps (sub-views 0.2 mm apart at the edge of a
  35 cm field, a third of a pixel), a blur the noise, counted once per view, does not share. Each
  Flash tube reads its own detector for the whole view period (`view_arc = 1.0`); GE's rapid kVp
  switching, by contrast, reads each energy for its duty cycle only.

`with_option` replaces one field (a seed, `use_noise`, `view_arc`) and keeps every other switch.
"""

# ╔═╡ 12000006-0000-4000-8000-000000000010
begin
    VIEW_SAMPLES = 5
    flash_opts = BS.SimOptions(
        projector = :dd_fast,
        use_heel_effect = false,
        view_samples = VIEW_SAMPLES,
    )
    "The same physics switches with one field replaced."
    with_option(opts, field, value) = BS.SimOptions(;
        (f => getfield(opts, f) for f in fieldnames(BS.SimOptions) if f !== field)...,
        (field => value,)...,
    )
end;

# ╔═╡ 12000005-0000-4000-8000-000000000001
md"""
## 5. Protocols, Dose and Grid

**Dual energy** is the protocol of the physical Flash scan of the Gammex that
basis-spectral-denoising compares its simulations with: **80 / Sn140 kVp** at 470 / 182 mA, the
0.4 mm Sn Selective Photon Shield on tube B (Primak, AJR 2010), 1152 views in 0.5 s, 4.8 mm
collimation, reconstructed as 4 × 1.0 mm slices over 33 cm (its clinical series: DE
monoenergetic, D40f). `FLASH` holds every value, as `SCANNERS.flash` does there. The IPEM spectra
stop at 140 kVp, the Flash's top kV, so no substitution is needed.

**Dose matching.** A physical tube's output per mA is its own and the simulator's is generic, so
both tube currents are scaled by one factor, `DOSE_SCALE`, keeping the physical 470 : 182 split,
until the simulated CTDIvol (the package's Monte Carlo dose in the 32 cm body phantom, summed over
both tubes) equals the physical scan's.

**Regular (dual power)** is the Flash's everyday dual-source mode: both tubes at the same kVp,
doubling the power. Routine abdomen is 120 kV at 420 mA per tube (210 quality-reference mAs at
0.5 s), scaled by the same `DOSE_SCALE`, which calibrates the simulated tube whatever it runs.

!!! note "The views are not halved per tube"
    Each tube-detector system has its own data acquisition sampling a full rotation (the datasheet:
    "up to 4,608 projections per 360° per data-acquisition unit"). In dual-energy mode tube A
    acquires a complete 1152-view rotation at 80 kV and tube B one at Sn140: full angular sampling
    per energy is dual source's defining advantage over rapid kVp switching. The 2,304 readings per
    rotation are the z flying focal spot (not modelled), and cardiac quarter-rotation segments are a
    reconstruction mode, not an acquisition split.
"""

# ╔═╡ 12000005-0000-4000-8000-000000000005
# The dual-energy acquisition: basis-spectral-denoising's `SCANNERS.flash`, verbatim. Seeds are per
# exposure: a shared seed would correlate the two channels' noise.
FLASH = (
    name = "Siemens SOMATOM Flash",
    mechanism = "dual source",
    kind = :eict,
    scanner = flash_scanner,
    opts = flash_opts,
    views = 1152,
    rotation_time = 0.5,
    collimation_mm = 4.8,
    slice_mm = 1.0,
    n_slices = 4,
    fov_cm = 33.0,
    clinical_recon = "DE monoenergetic, D40f, 1.0 mm, 330 mm",
    ctdi_vol_mGy = 10.01,
    exposures = [
        (
            label = "80 kVp",
            kVp = 80,
            mA = 470.0,
            view_arc = 1.0,
            seed = 1234,
            filters = Tuple{String, Float64}[],
        ),
        (
            label = "Sn140 kVp",
            kVp = 140,
            mA = 182.0,
            view_arc = 1.0,
            seed = 4321,
            filters = [("Sn", 0.4)],
        ),
    ],
);

# ╔═╡ 12000005-0000-4000-8000-000000000010
begin
    "The CTProtocol of one exposure of scanner `s`, its tube current scaled by `mA_scale` (`DOSE_SCALE`)."
    exposure_protocol(s, e; mA_scale = 1.0) = BS.CTProtocol(;
        kVp = e.kVp,
        mA = e.mA * mA_scale,
        views = s.views,
        rotation_time = s.rotation_time,
        collimation_mm = s.collimation_mm,
        additional_filters = e.filters,
    )
    "Reconstruction grid of scanner `s`: its clinical series' slices, field of view and 512²."
    recon_options(s) = BS.ReconOptions(
        matrix_size = (512, 512, s.n_slices),
        fov_cm = s.fov_cm,
        z_cm = s.n_slices * s.slice_mm / 10,
    )
end;

# ╔═╡ 12000005-0000-4000-8000-000000000020
# One factor for both tubes: the simulated CTDIvol (32 cm body phantom, summed over the exposures)
# equals the physical scan's.
DOSE_SCALE = FLASH.ctdi_vol_mGy /
    sum(FLASH.exposures) do e
        p = exposure_protocol(FLASH, e)
        BS.compute_dose(BS.dose_source(FLASH.scanner, p), p).ctdi_vol_mGy
    end

# ╔═╡ 12000005-0000-4000-8000-000000000030
begin
    # dual energy: tube A (low) and tube B (high), at the physical scan's dose
    protocol_low, protocol_high = (
        exposure_protocol(FLASH, e; mA_scale = DOSE_SCALE)
        for e in FLASH.exposures
    )
    # regular (dual power): 120 kVp, 420 mA on each tube, on the same simulated tube
    protocol_reg = BS.CTProtocol(
        kVp = 120,
        mA = 420.0 * DOSE_SCALE,
        views = FLASH.views,
        rotation_time = FLASH.rotation_time,
        collimation_mm = FLASH.collimation_mm,
    )
end;

# ╔═╡ 12000005-0000-4000-8000-000000000040
let
    filt(p) = isempty(p.additional_filters) ?
        "none" :
        join(["$(f[2]) mm $(f[1])" for f in p.additional_filters], ", ")
    rows = [
        "| $(m) | $(t) | $(p.kVp) | $(filt(p)) | $(round(Int, p.mA / DOSE_SCALE)) | $(round(p.mA, digits = 1)) |"
        for (m, t, p) in (
            ("Regular", "A and B", protocol_reg),
            ("Dual energy", "A (low)", protocol_low),
            ("Dual energy", "B (high)", protocol_high),
        )
    ]
    Markdown.parse(
        """
    `DOSE_SCALE` = $(round(DOSE_SCALE, digits = 3)): at the scaled currents the dual-energy scan's
    simulated CTDIvol is the physical scan's $(FLASH.ctdi_vol_mGy) mGy.

    | Acquisition | Tube | kVp | Added filter | Physical mA | Simulated mA |
    |---|---|---:|---|---:|---:|
    $(join(rows, "\n"))

    Every exposure: $(FLASH.views) views in $(FLASH.rotation_time) s, $(FLASH.collimation_mm) mm
    collimation, the 8.4 mm Al flat filter and the large-body bowtie.
    """,
    )
end

# ╔═╡ 12000006-0000-4000-8000-000000000020
# The clinical series' grid (`recon_options`): 4 × 1.0 mm slices over 33 cm, 512².
recon_opts = recon_options(FLASH);

# ╔═╡ 12000006-0000-4000-8000-000000000025
# The Gammex's labels on the reconstruction grid (the phantom itself is voxelized over 35 cm):
# every ROI below is drawn on this map.
roi_labels = BS.create_gammex_472(
    n_voxels = recon_opts.matrix_size[1],
    n_slices = 1,
    fov_cm = recon_opts.fov_cm,
    z_cm = 0.1,
).mask[
    :,
    :,
    1,
];

# ╔═╡ 12000006-0000-4000-8000-000000000030
"""
    flash_detected_spectrum(protocol) -> (e, w_eta)

Display helper: tube spectrum × flat filtration × protocol filters (IPEM, absolute flux) × the
Flash UFC's signal per photon, `E·efficiency(E)`: what the detector integrates (centre ray, no
bowtie).
"""
function flash_detected_spectrum(protocol)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        flash_opts,
        protocol;
        scanner = flash_scanner,
    )
    t = BS.UFC_FLASH_MC_EFFICIENCY_LUT # 1 keV grid: index = energy
    eff(E) = (
        i = clamp(floor(Int, E), 1, 139);
        f = E - i;
        (1 - f) * t.efficiency[i] + f * t.efficiency[i + 1]
    )
    return e, Float64.(w) .* e .* eff.(e)
end;

# ╔═╡ 12000006-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1000))

    # each beam: its filters, then the UFC
    sketch = canvas(fig[1, 1], (0, 40, 0, 6.6))
    beams = (
        (5.4, "Regular, 120 kVp", ["8.4 mm Al"], COLORS.green),
        (3.3, "Tube A, 80 kVp", ["8.4 mm Al"], COLORS.cyan),
        (1.2, "Tube B, 140 kVp", ["8.4 mm Al", "0.4 mm Sn"], COLORS.violet),
    )
    for (y, label, filters, color) in beams
        Mke.text!(sketch, 5.0, y; text = label, align = (:right, :center), font = :bold)
        Mke.scatter!(
            sketch,
            [Mke.Point2f(6.5, y)];
            marker = :rect,
            markersize = 30,
            color = COLORS.ink,
        )
        Mke.scatter!(sketch, [Mke.Point2f(6.5, y)]; markersize = 10, color = COLORS.beam)
        arrow!(sketch, (7.3, y), (34.6, y); color = COLORS.beam, linewidth = 4)
        for (k, f) in enumerate(filters)
            x = 10.5 + 4.2 * (k - 1)
            Mke.poly!(
                sketch,
                Mke.Rect2f(x - 0.25, y - 0.7, 0.5, 1.4);
                color = k == 2 ? COLORS.muted : COLORS.rule,
                strokecolor = COLORS.ink,
                strokewidth = 1,
            )
            Mke.text!(
                sketch,
                x + 0.5,
                y + 0.5;
                text = f,
                align = (:left, :bottom),
                fontsize = 15,
            )
        end
        Mke.poly!(sketch, Mke.Rect2f(35.0, y - 0.8, 0.7, 1.6); color)
        Mke.text!(
            sketch,
            36.2,
            y;
            text = "UFC",
            align = (:left, :center),
            font = :bold,
            color,
        )
    end

    ax = Mke.Axis(
        fig[2, 1];
        title = "The Spectra the Detector Weighs",
        subtitle = "Photons × Energy × Flash UFC Efficiency, Centre Ray, Each Normalized to Its Peak",
        xlabel = "Photon Energy (keV)",
        ylabel = "Relative Detected Signal",
    )
    curves = (
        (protocol_reg, "Regular, 120 kVp (Both Tubes)", COLORS.green),
        (protocol_low, "Tube A, 80 kVp", COLORS.cyan),
        (protocol_high, "Tube B, Sn140 kVp", COLORS.violet),
    )
    for (protocol, label, color) in curves
        e, wη = flash_detected_spectrum(protocol)
        mean_E = sum(e .* wη) / sum(wη)
        Mke.band!(ax, Float64.(e), 0.0, wη ./ maximum(wη); color = (color, 0.12))
        Mke.lines!(
            ax,
            Float64.(e),
            wη ./ maximum(wη);
            color,
            linewidth = 3,
            label = "$(label): Mean $(round(mean_E, digits = 1)) keV",
        )
    end
    Mke.vlines!(ax, [50.24]; color = COLORS.beam, linestyle = :dot, linewidth = 2)
    Mke.text!(
        ax,
        49.0,
        1.0;
        text = "Gd K-Edge",
        align = (:right, :top),
        fontsize = 16,
        color = COLORS.beam,
    )
    Mke.xlims!(ax, 15, 145)
    Mke.ylims!(ax, 0, 1.08)
    Mke.axislegend(ax; position = :rt)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.33))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_detected_spectra.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000007-0000-4000-8000-000000000001
md"""
## 6. The Regular Scan: Dual Power, 120 / 120 kVp

Both tubes at the same kVp. Clinically this buys temporal resolution (cardiac) or a power reserve
(large patients); what can be checked here is that averaging the two **independent** acquisitions
leaves the accuracy untouched and cuts the noise by **√2**:

- water at 0 HU in tube A's image, tube B's, and the combined image (the BHC is right on both
  chains), and
- ``\sigma_\mathrm{combined} ≈ \sigma_\mathrm{single} / \sqrt{2}``, because the two noise
  realizations are independent; this is what tube B's own seed buys.

The two tubes run one protocol through one geometry, so their noise-free expectation is one
projection: tube A keeps it (`keep_projection = true`) and tube B passes it back (`projection`),
drawing only its own noise. Each tube then runs the standard correction stack: a water
beam-hardening correction (BHC) of the sinogram with the Flash's UFC folded in, FDK, then HU.
"""

# ╔═╡ 12000007-0000-4000-8000-000000000005
"""
    flash_bhc_calibration(protocol, geom)

η-aware per-tube BHC: bowtie-hardened per-column spectrum × Flash UFC η(E)
→ per-column two-material polynomial.  Returns `(model, μ_water, ref_E_keV)`.
"""
function flash_bhc_calibration(protocol, geom)
    # resolve_source_spectrum_full folds bowtie AND the src Flash UFC η(E)
    # (via the same build_physics_config the forward model used).
    e, ŵ = BS.resolve_source_spectrum_full(
        flash_opts,
        protocol;
        scanner = flash_scanner,
        geom = geom,
    )
    e2, w_col = BS.bhc_spectrum_per_column(e, ŵ) # [n_E, n_col]

    # Single mono-equivalent target = mean energy of the η-folded mean spectrum
    w_mean = vec(sum(w_col; dims = 2)) ./ size(w_col, 2)
    ref_E = sum(e2 .* w_mean) / sum(w_mean)

    # Parameter-free water BHC from the per-column Flash-η spectrum.
    model = BS.calibrate_bhc_water(e2, w_col; reference_energy_keV = ref_E)
    return (
        model = model,
        μ_water = model.μ_water_ref,
        ref_E_keV = model.reference_energy_keV,
    )
end;

# ╔═╡ 12000007-0000-4000-8000-000000000008
"""
    flash_poly_recon(sino_cpu, geom, bhc) -> Array{Float32, 3}

Doctrine correction stack for one tube: knobless water sino-BHC → FDK → HU.
"""
function flash_poly_recon(sino_cpu, geom, bhc)
    matrix_size = recon_opts.matrix_size

    sino_gpu = to_gpu(sino_cpu)
    # Knobless water BHC: one sinogram-domain pass, no recon round-trip.
    sino_bhc = BS.apply_bhc_water(sino_gpu, bhc.model)
    sino_gpu = sino_bhc

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom)

    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water))

    ws_fdk = nothing
    sino_gpu = nothing
    recon_μ = nothing
    GC.gc(true)
    return hu
end;

# ╔═╡ 12000007-0000-4000-8000-000000000010
sim_reg = let projection = nothing, out = Dict{Symbol, Any}()
    for (tube, seed) in ((:a, 1234), (:b, 4321))
        opts = with_option(flash_opts, :seed, seed)
        ws = BS.create_workspace(flash_scanner, protocol_reg, opts, recon_opts, phantom)
        try
            t = time()
            result = BS.simulate!(
                ws,
                phantom,
                protocol_reg,
                opts;
                report_dose = false,
                keep_projection = projection === nothing,
                projection,
            )
            # tube B reuses tube A's
            projection === nothing && (projection = result.projection)
            out[tube] = (sino = Array(ws.sinogram), geom = ws.geom, seconds = time() - t)
        finally
            BS.release_backend!(ws)
        end
    end
    projection = nothing
    GC.gc(true)
    (a = out[:a], b = out[:b])
end;

# ╔═╡ 12000007-0000-4000-8000-000000000015
bhc_reg = flash_bhc_calibration(protocol_reg, sim_reg.a.geom);

# ╔═╡ 12000007-0000-4000-8000-000000000020
hu_reg = (
    a = flash_poly_recon(sim_reg.a.sino, sim_reg.a.geom, bhc_reg),
    b = flash_poly_recon(sim_reg.b.sino, sim_reg.b.geom, bhc_reg),
);

# ╔═╡ 12000007-0000-4000-8000-000000000025
# Dual power: the two independent same-kV acquisitions average into the
# combined image (image domain — per-ray sinogram averaging is equivalent
# to first order on the log data at these noise levels).
hu_reg_combined = 0.5f0 .* hu_reg.a .+ 0.5f0 .* hu_reg.b;

# ╔═╡ 12000007-0000-4000-8000-000000000030
# Accuracy in the eroded solid water; noise in a 12-pixel disc at the centre (≈ 7.7 mm at 0.645 mm
# per pixel), every slice
dp_stats = let
    ERODE_PX = 12.0
    sw_bool = BS.erode_mask_2d(
        roi_labels .== UInt8(BS.REGION_SOLID_WATER);
        erode_px = ERODE_PX,
    )
    sw_idx = findall(sw_bool)

    nx_r, ny_r, _ = size(hu_reg_combined)
    cx = nx_r ÷ 2 + 1
    cy = ny_r ÷ 2 + 1
    noise_bool = [(i - cx)^2 + (j - cy)^2 ≤ 12.0^2 for i in 1:nx_r, j in 1:ny_r]
    noise_idx = findall(noise_bool)
    n_z = size(hu_reg_combined, 3)

    function _roi(vol, idx)
        vals = Float64[Float64(vol[ci, z]) for z in 1:n_z, ci in idx]
        (mean = mean(vals), std = std(vals), n = length(vals))
    end

    water = (
        a = _roi(hu_reg.a, sw_idx),
        b = _roi(hu_reg.b, sw_idx),
        combined = _roi(hu_reg_combined, sw_idx),
    )
    noise = (
        a = _roi(hu_reg.a, noise_idx),
        b = _roi(hu_reg.b, noise_idx),
        combined = _roi(hu_reg_combined, noise_idx),
    )
    σ_single = 0.5 * (noise.a.std + noise.b.std)

    (
        water = water,
        noise = noise,
        noise_ratio = noise.combined.std / σ_single,
        sw_mask_2d = collect(sw_bool),
        noise_mask_2d = noise_bool,
    )
end;

# ╔═╡ 12000007-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1150))
    mid = size(hu_reg_combined, 3) ÷ 2

    top = fig[1, 1] = Mke.GridLayout()
    operator(col, text) = Mke.Label(
        top[1, col],
        text;
        fontsize = 54,
        font = :bold,
        color = COLORS.muted,
        tellheight = false,
    )
    slice_panel!(top[1, 1], hu_reg.a[:, :, mid]; title = "0.5 × Tube A (120 kVp)")
    operator(2, "+")
    slice_panel!(top[1, 3], hu_reg.b[:, :, mid]; title = "0.5 × Tube B (120 kVp)")
    operator(4, "=")
    slice_panel!(top[1, 5], hu_reg_combined[:, :, mid]; title = "Dual Power")
    Mke.Colorbar(
        top[1, 6];
        colormap = :grays,
        colorrange = HU_WINDOW,
        label = "HU",
        height = Mke.Relative(0.85),
    )
    for col in (2, 4)
        Mke.colsize!(top, col, Mke.Fixed(60))
    end

    bottom = fig[2, 1] = Mke.GridLayout()
    sketch = canvas(bottom[1, 1], (-18, 18, -18, 18))
    sketch.title = "Where: The Centre"
    gammex!(sketch; calcium = COLORS.rule, iodine = COLORS.rule)
    roi!(
        sketch,
        dp_stats.noise_mask_2d,
        recon_opts.fov_cm;
        color = COLORS.cyan,
        alpha = 1.0,
    )

    entries = (
        ("Tube A", dp_stats.noise.a.std, COLORS.cyan),
        ("Tube B", dp_stats.noise.b.std, COLORS.violet),
        ("Dual Power", dp_stats.noise.combined.std, COLORS.green),
    )
    σs = [e[2] for e in entries]
    σ_pred = 0.5 * (σs[1] + σs[2]) / sqrt(2.0)
    ax = Mke.Axis(
        bottom[1, 2];
        title = "Noise in the Centre",
        subtitle = "Two Independent Tubes Average to σ / √2",
        ylabel = "σ (HU)",
        xticks = (1:3, [e[1] for e in entries]),
    )
    Mke.barplot!(ax, 1:3, σs; color = [e[3] for e in entries])
    Mke.hlines!(
        ax,
        [σ_pred];
        color = COLORS.ink,
        linewidth = 2,
        linestyle = :dash,
        label = "Predicted, Single / √2 = $(round(σ_pred, digits = 2)) HU",
    )
    for (k, σ) in enumerate(σs)
        Mke.text!(
            ax,
            k,
            σ;
            text = "σ = $(round(σ, digits = 2)) HU",
            align = (:center, :bottom),
            offset = (0, 6),
            fontsize = 17,
        )
    end
    Mke.ylims!(ax, 0, 1.4 * maximum(σs))
    Mke.axislegend(ax; position = :rt)
    Mke.colsize!(bottom, 1, Mke.Relative(0.33))
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.5))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_dual_power.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000008-0000-4000-8000-000000000001
md"""
## 7. The Dual-Energy Acquisition: One Projection per Tube, Three Draws

`acquire` builds each tube's workspace and **projects it once**: everything before the noise is
the same for every draw, so the first draw keeps the projection (`keep_projection = true`) and the
others pass it back (`projection`) and draw only their noise, exactly as if simulated afresh.

| Draw | Noise | Used for |
|---:|---|---|
| 0 | off | the **noise-free reference** every noisy result is scored against |
| 1 | on | the **measured** acquisition every result comes from |
| 9 | on | the **calibration** draw, which only fixes the reconstruction's pair (section 9) |

The first draw also keeps each tube's one-channel spectral basis, `spectral_basis(ws, protocol)`:
every ray's detected spectrum (source × filtration × bowtie × the Flash's ``m_1``), the model the
projector applied, scaled to the ray's noise-equivalent quanta in air.
"""

# ╔═╡ 12000006-0000-4000-8000-000000000012
begin
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each exposure the seed
    # `seed + 1000 (r − 1)`, so the two tubes stay independent within a draw, and draws of each other.
    DE_DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    opts_for(e, r) = with_option(
        with_option(
            with_option(FLASH.opts, :seed, realization_seed(e, r)),
            :use_noise,
            r > 0,
        ),
        :view_arc,
        e.view_arc,
    )
end;

# ╔═╡ 12000008-0000-4000-8000-000000000005
"""
    acquire(draws) -> Dict{Int, NamedTuple}

The dual-energy acquisition of the Gammex (both tubes of `FLASH`) for every draw in `draws`: draw 0
the noise-free expectation, draw `r ≥ 1` a noisy realisation with the seeds `realization_seed(e, r)`.
Each exposure is projected once, by the first draw, and every other draw re-uses that projection
(`keep_projection`, `projection`), only its noise drawn. Each entry holds the two log-transmission
channels, the spectral basis the decomposition inverts (shared by every draw), the geometry and the
wall time of the draw per tube.
"""
function acquire(draws)
    parts = map(FLASH.exposures) do e
        protocol = exposure_protocol(FLASH, e; mA_scale = DOSE_SCALE)
        projection = nothing
        channels = Dict{Int, Array{Float32, 3}}()
        seconds = Dict{Int, Float64}()
        model = nothing
        for r in draws
            opts = opts_for(e, r)
            t = time()
            ws = BS.create_workspace(FLASH.scanner, protocol, opts, recon_opts, phantom)
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
                channels[r] = Array(ws.sinogram)
                if model === nothing
                    model = (geom = ws.geom, basis = BS.spectral_basis(ws, protocol))
                end
            finally
                BS.release_backend!(ws)
            end
            seconds[r] = time() - t
        end
        projection = nothing
        (; channels, seconds, model...)
    end
    GC.gc(true)
    basis = BS.spectral_basis_from_acquisitions(acquisitions = [p.basis for p in parts])
    Dict(
        r => (
            draw = r,
            channels = [p.channels[r] for p in parts],
            basis = basis,
            geom = first(parts).geom,
            seconds = [p.seconds[r] for p in parts],
        )
        for r in draws
    )
end;

# ╔═╡ 12000008-0000-4000-8000-000000000020
de_acq = acquire((DE_DRAWS.reference, DE_DRAWS.measured, DE_DRAWS.calibration));

# ╔═╡ 12000008-0000-4000-8000-000000000030
let
    role = Dict(
        DE_DRAWS.reference => "noise-free reference",
        DE_DRAWS.measured => "measured",
        DE_DRAWS.calibration => "calibration",
    )
    seeds(r) = r == 0 ?
        "none" :
        join([realization_seed(e, r) for e in FLASH.exposures], ", ")
    secs(v) = join([string(round(s, digits = 1)) for s in v], ", ")
    rows = [
        "| $(r) | $(role[r]) | $(seeds(r)) | $(secs(de_acq[r].seconds)) |"
        for r in sort(collect(keys(de_acq)))
    ]
    Markdown.parse(
        """
    | Draw | Role | Seeds (A, B) | Seconds (A, B) |
    |---:|---|---|---|
    $(join(rows, "\n"))

    Draw 0 projects each tube, $(VIEW_SAMPLES) sub-views per view; draws 1 and 9 re-use that
    projection and only draw their noise. In the regular mode of section 6, tube A took
    $(round(sim_reg.a.seconds, digits = 1)) s and tube B, re-using its projection,
    $(round(sim_reg.b.seconds, digits = 1)) s (wall time on this render, compilation included in the first call).
    """,
    )
end

# ╔═╡ 12000008-0000-4000-8000-000000000035
# The measured draw: every DE result below comes from it.
de_measured = de_acq[DE_DRAWS.measured];

# ╔═╡ 12000008-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 600))

    # one view: each dexel column reads one ray through the phantom
    sketch = canvas(fig[1, 1], (-7.2, 7.2, -7.2, 7.2))
    sketch.title = "One View"
    gantry!(sketch; dark = false, object = false, labels = ("Tube A",))
    gammex!(sketch; scale = 1.9 / GAMMEX.body_r)
    θ = deg2rad(90 + 180 - 9)
    s = Mke.Point2f(0, 5.4)
    d = Mke.Point2f(s + 9.5f0 * Mke.Point2f(cos(θ), sin(θ)))
    Mke.lines!(sketch, [s, d]; color = COLORS.red, linewidth = 3)
    Mke.scatter!(sketch, [d]; color = COLORS.red, markersize = 16)
    Mke.text!(
        sketch,
        d + Mke.Point2f(0.2, -0.4);
        text = "One Dexel Column",
        align = (:left, :top),
        fontsize = 15,
        color = COLORS.red,
    )

    mid_row = size(de_measured.channels[1], 2) ÷ 2 + 1
    slices = [permutedims(c[:, mid_row, :], (2, 1)) for c in de_measured.channels]
    window = Tuple(Float64.(quantile(vcat(vec.(slices)...), (0.01, 0.99))))
    titles = ("Tube A: 80 kVp", "Tube B: Sn140 kVp")
    for (k, (slice, title)) in enumerate(zip(slices, titles))
        ax = Mke.Axis(
            fig[1, k + 1];
            title,
            subtitle = "Central Detector Row, Every View",
            xlabel = "View",
            ylabel = k == 1 ? "Dexel Column" : "",
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = window)
    end
    Mke.Colorbar(
        fig[1, 4];
        colormap = :viridis,
        colorrange = window,
        label = "Line Integral, −ln(I / I₀)",
    )
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.27))
    fig
end

# ╔═╡ 12000009-0000-4000-8000-000000000001
md"""
## 8. The Mixed Image

In dual-energy mode the routine-looking output is the **mixed image**, a linear blend of the two
tubes' reconstructions, ``M = w\,I_\mathrm{low} + (1 - w)\,I_\mathrm{high}`` (Yu et al., Med Phys
2009; Eusemann et al., SPIE 2008). This notebook shows the equal blend, ``w = 0.5``; clinical Flash
blends weight the low-kV image between about 0.3 and 0.6. Each tube runs the same correction
stack as in section 6; if the Flash's table is folded in right, solid water lands at 0 HU in both
tubes' images, and so in any blend.
"""

# ╔═╡ 12000009-0000-4000-8000-000000000010
bhc_low = flash_bhc_calibration(protocol_low, de_measured.geom);

# ╔═╡ 12000009-0000-4000-8000-000000000012
bhc_high = flash_bhc_calibration(protocol_high, de_measured.geom);

# ╔═╡ 12000009-0000-4000-8000-000000000015
Markdown.parse(
    """
**The calibrations, with the Flash's UFC folded in**: tube A's reference energy is
$(round(bhc_low.ref_E_keV, digits = 1)) keV, where water attenuates $(round(bhc_low.μ_water, digits = 5)) cm⁻¹;
tube B's is $(round(bhc_high.ref_E_keV, digits = 1)) keV, $(round(bhc_high.μ_water, digits = 5)) cm⁻¹; the regular
120 kVp scan's is $(round(bhc_reg.ref_E_keV, digits = 1)) keV.
""",
)

# ╔═╡ 12000009-0000-4000-8000-000000000020
hu_tube = (
    low = flash_poly_recon(de_measured.channels[1], de_measured.geom, bhc_low),
    high = flash_poly_recon(de_measured.channels[2], de_measured.geom, bhc_high),
);

# ╔═╡ 12000009-0000-4000-8000-000000000025
# Siemens' linear mixed image, M = w·I_low + (1 − w)·I_high (image domain, Yu 2009): the equal blend
MIX_W_LOW = 0.5f0;

# ╔═╡ 12000009-0000-4000-8000-000000000028
hu_mixed = MIX_W_LOW .* hu_tube.low .+ (1.0f0 - MIX_W_LOW) .* hu_tube.high;

# ╔═╡ 12000009-0000-4000-8000-000000000030
# Solid water, eroded 12 pixels from every edge, in the three images (every slice)
poly_water_stats = let
    ERODE_PX = 12.0
    sw_bool = BS.erode_mask_2d(
        roi_labels .== UInt8(BS.REGION_SOLID_WATER);
        erode_px = ERODE_PX,
    )
    sw_idx = findall(sw_bool)
    n_z = size(hu_mixed, 3)

    function _stats(vol)
        vals = Float64[Float64(vol[ci, z]) for z in 1:n_z, ci in sw_idx]
        (mean = mean(vals), std = std(vals), n = length(vals))
    end
    (
        low = _stats(hu_tube.low),
        high = _stats(hu_tube.high),
        mixed = _stats(hu_mixed),
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 12000009-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 540))
    mid = size(hu_mixed, 3) ÷ 2
    w = round(MIX_W_LOW; digits = 2)
    operator(col, text) = Mke.Label(
        fig[1, col],
        text;
        fontsize = 54,
        font = :bold,
        color = COLORS.muted,
        tellheight = false,
    )
    slice_panel!(fig[1, 1], hu_tube.low[:, :, mid]; title = "$(w) × Tube A (80 kVp)")
    operator(2, "+")
    slice_panel!(
        fig[1, 3],
        hu_tube.high[:, :, mid];
        title = "$(1 - w) × Tube B (Sn140 kVp)",
    )
    operator(4, "=")
    slice_panel!(fig[1, 5], hu_mixed[:, :, mid]; title = "Mixed Image")
    Mke.Colorbar(
        fig[1, 6];
        colormap = :grays,
        colorrange = HU_WINDOW,
        label = "HU",
        height = Mke.Relative(0.85),
    )
    for col in (2, 4)
        Mke.colsize!(fig.layout, col, Mke.Fixed(60))
    end
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_poly_recon.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 12000009-0000-4000-8000-000000000060
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 640))

    sketch = canvas(fig[1, 1], (-18, 18, -18, 18))
    sketch.title = "Where: The Solid Water"
    gammex!(sketch; calcium = COLORS.rule, iodine = COLORS.rule)
    roi!(sketch, poly_water_stats.mask_2d, recon_opts.fov_cm)

    entries = (
        ("Tube A, 80 kVp", poly_water_stats.low, COLORS.cyan),
        ("Tube B, Sn140 kVp", poly_water_stats.high, COLORS.violet),
        ("Mixed", poly_water_stats.mixed, COLORS.green),
    )
    means = [e[2].mean for e in entries]
    ax = Mke.Axis(
        fig[1, 2];
        title = "Solid Water in the Three Images",
        subtitle = "Mean Over the Shaded Region and Every Slice; σ Is the Pixel Noise",
        ylabel = "Mean HU",
        xticks = (1:3, [e[1] for e in entries]),
    )
    y_max = max(5.0, 1.6 * maximum(abs, means))
    Mke.hlines!(ax, [0.0]; color = COLORS.ink, linewidth = 1, linestyle = :dash)
    Mke.scatter!(ax, 1:3, means; color = [e[3] for e in entries], markersize = 26)
    for (k, e) in enumerate(entries)
        Mke.text!(
            ax,
            k,
            means[k];
            text = "$(round(e[2].mean, digits = 1)) HU\nσ = $(round(e[2].std, digits = 1)) HU",
            align = (:center, :bottom),
            offset = (0, 20),
            fontsize = 17,
        )
    end
    Mke.xlims!(ax, 0.4, 3.6)
    Mke.ylims!(ax, -y_max, y_max)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.38))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_poly_water_values.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000a-0000-4000-8000-000000000001
md"""
## 9. The VMI Chain

**The spectral basis.** `acquire` merged the two tubes' bases with
`spectral_basis_from_acquisitions`: their energy grids onto one union grid, each tube's per-ray
detected spectrum scaled by its own noise-equivalent quanta in air (``I_0\,\mathrm{ff}\,S_1^2/S_2``).
The likelihood thus sees the absolute response ``\Phi_k(E)`` of every ray and channel (source ×
flat filter × bowtie × the Flash UFC's ``m_1(E)``), the model the projector applied. No calibration
scan is involved, and all three draws share it.
"""

# ╔═╡ 1200000a-0000-4000-8000-000000000010
basis = de_measured.basis;

# ╔═╡ 1200000a-0000-4000-8000-000000000015
Markdown.parse(
    """
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid for
$(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response sums to the air counts to within
$(round(basis.I0_relerr, sigdigits = 2)) (relative).
""",
)

# ╔═╡ 1200000b-0000-4000-8000-000000000001
md"""
**The chain.** One package call goes from the two corrected sinograms to the VMIs,
`vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter = PAIR_FILTER.filter, pair_basis, composite_energy)`:

1. **Projection HYPR-LR** (`ProjectionHYPR`): a 3 × 3 window (columns × views of one parity,
   `view_stride = 2`, so the odd and even views stay independent) on each detector row's counts,
   each tube's dispersion measured on its own air rays.
2. **The K = 2 decomposition**: per ray, the Poisson maximum-likelihood iodine and water pair
   under the exact polychromatic mean of both tubes, every detector row kept.
3. **The spectral pair** (`spectral_pair`): FDK of the composite ``M``, the minimum-noise VMI, and
   of its complement ``I_\perp``, whose noise is uncorrelated with ``M``'s, each with its own window
   of the fitted `PairFilter`, on the pair fixed from the calibration draw.
4. **ACNR** on the complement only; the composite stays as reconstructed.
5. **Image HYPR** (`ImageHYPR`): the complement pooled within its slice, weighted by the
   composite, over the window with the least estimated risk.
6. **VMI synthesis** at 50, 70, 100 and 140 keV from the one basis pair. Nothing averages detector
   rows or slices.

**The fitted windows (`PAIR_FILTER`).** Each window is an apodized ramp
``W(f) = \exp(-(f / f_c)^p)`` on the grid's Nyquist axis. The Flash's pair (composite
``f_c = 1.05``, ``p = 1.5``; complement ``f_c = 0.8``, ``p = 1.0``) was fitted in
basis-spectral-denoising to the MTF and NPS of the physical Flash scan of the Gammex, on 512² over
33 cm, the grid used here. A window is a function of physical frequency, so on another grid it is
rescaled by the ratio of the two grids' `grid_bandlimit`.

**The pair and the composite energy come from the calibration draw.** Near its minimum the VMI
noise hardly changes with energy, so an argmin measured on the draw being evaluated would itself
be noise. They are measured once on the calibration draw with a standard soft-tissue window
(`SoftFilter`): ``E^*`` and ``\beta`` from its plain decomposition, the composite energy from its
decomposition after the projection HYPR. The measured draw, the same draw without denoising, and
the noise-free reference are then reconstructed alike.
"""

# ╔═╡ 1200000b-0000-4000-8000-000000000005
# basis-spectral-denoising's HYPR_CHAIN: ProjectionHYPR(3 × 3 Box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 1200000b-0000-4000-8000-000000000006
VMI_CHAIN = (
    method = :nchannel,
    controls = BS.NChannelControls(),
    use_tlbf = false,
    antialias = true,
);

# ╔═╡ 1200000b-0000-4000-8000-000000000007
# The Flash's fitted pair of FDK windows (basis-spectral-denoising's `RECON.flash`), fitted on
# 512² over 33 cm and rescaled onto this grid's frequency axis as its `RECON_BY` does.
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11)),
    fitted = (composite = (fc = 1.05, p = 1.5), complement = (fc = 0.8, p = 1.0)),
    g = de_measured.geom,
    r = BS.grid_bandlimit(g, recon_opts.matrix_size) / min(1.0, g.pixel_size / (33.0 / 512))
    window(q) = BS.CustomFilter(
        knots,
        Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots)
    )
    (
        filter = BS.PairFilter(window(fitted.composite), window(fitted.complement)),
        kernels = fitted,
        scale = r,
    )
end;

# ╔═╡ 1200000c-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 1200000b-0000-4000-8000-000000000008
# `pair_basis` and `composite_energy` (basis-spectral-denoising's `PAIR_BASIS`), from the calibration draw
calibration = let a = de_acq[DE_DRAWS.calibration]
    decompose(denoiser) = BS.vmi_pipeline(;
        channels = a.channels,
        basis = a.basis,
        geom = a.geom,
        to_backend = to_gpu,
        matrix_size = recon_opts.matrix_size,
        vmi_energies = Tuple(de_vmi_energies),
        denoiser,
        keep_sinograms = true,
        VMI_CHAIN...,
        use_acnr = false,
    ).sinograms
    sp(d; kw...) = BS.spectral_pair(
        d.water,
        d.iodine,
        a.geom,
        recon_opts.matrix_size;
        filter = BS.SoftFilter(),
        antialias = VMI_CHAIN.antialias,
        to_backend = to_gpu,
        kw...,
    )
    x = sp(decompose(nothing)) # the plain decomposition
    c = sp(decompose(BS.SpectralHYPR(image = nothing)); # after the projection HYPR
        basis = (Estar = x.Estar, β = x.β)
    )
    (pair_basis = (Estar = x.Estar, β = x.β), composite_energy = c.Estar)
end;

# ╔═╡ 1200000b-0000-4000-8000-000000000010
de_vmi = BS.vmi_pipeline(;
    channels = de_measured.channels,
    basis,
    geom = de_measured.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 1200000b-0000-4000-8000-000000000011
# The same draw and reconstruction without the denoising (no HYPR, no ACNR), and the noise-free
# reference (draw 0, the same projection), reconstructed alike.
de_none, de_ref = (
    BS.vmi_pipeline(;
        channels = de_acq[r].channels,
        basis,
        geom = de_measured.geom,
        to_backend = to_gpu,
        matrix_size = recon_opts.matrix_size,
        vmi_energies = Tuple(de_vmi_energies),
        denoiser = nothing,
        use_acnr = false,
        fbp_filter = PAIR_FILTER.filter,
        pair_basis = calibration.pair_basis,
        composite_energy = calibration.composite_energy,
        VMI_CHAIN...,
    )
    for r in (DE_DRAWS.measured, DE_DRAWS.reference)
);

# ╔═╡ 1200000b-0000-4000-8000-000000000015
let
    q = de_vmi.quality
    d = de_vmi.settings.denoiser
    pct(x) = round(100x, digits = 3)
    pb = calibration.pair_basis
    Markdown.parse(
        """
    The decomposition solved $(q.n_rays) rays in $(round(de_vmi.elapsed_s, digits = 1)) s
    with $(round(q.outer_mean, digits = 1)) outer iterations on average; $(pct(q.frac_not_converged))% did not
    converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
    The projection HYPR measured dispersions (variance / mean of the counts on the air rays) of
    $(join(round.(d.dispersion, digits = 2), " and ")) for tube A and tube B.

    From the calibration draw: the windows act on the pair at E* = $(round(pb.Estar, digits = 1)) keV,
    β = $(round(pb.β, sigdigits = 3)), and ACNR and the image HYPR on the composite at
    $(round(calibration.composite_energy, digits = 1)) keV. The image HYPR pooled the complement over a
    $(d.image_estimates.window) × $(d.image_estimates.window) window. The fitted windows were rescaled by
    $(round(PAIR_FILTER.scale, digits = 3)) onto this grid's frequency axis.
    """,
    )
end

# ╔═╡ 1200000a-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1460))

    nodes = Dict(
        :a => (0.0, 0.75, "Tube A Sinogram", "80 kVp", COLORS.cyan),
        :b => (0.0, -0.75, "Tube B Sinogram", "Sn140 kVp", COLORS.violet),
        :fit => (4.4, 0.0, "Per-Ray ML Fit", "K = 2, Every Row"),
        :basis => (8.8, 0.0, "Basis Sinograms", "Iodine + Water"),
        :images => (13.2, 0.0, "Basis Images", "FDK, ACNR, Image HYPR", COLORS.green),
    )
    flow_chart!(
        fig[1, 1:4],
        nodes,
        [:a => :fit, :b => :fit, :fit => :basis, :basis => :images];
        limits = (-1.7, 14.9, -1.4, 1.4),
    )

    mid_row = size(de_vmi.sinograms.iodine, 2) ÷ 2 + 1
    mid = (size(de_vmi.images.iodine, 3) + 1) ÷ 2
    sinogram(x) = permutedims(x[:, mid_row, :], (2, 1))
    window(x) = Tuple(Float64.(quantile(vec(x), (0.01, 0.99))))
    panels = (
        (
            2,
            1,
            "Iodine Basis Sinogram",
            1000 .* sinogram(de_vmi.sinograms.iodine),
            "mg/cm²",
            false,
        ),
        (2, 3, "Water Basis Sinogram", sinogram(de_vmi.sinograms.water), "g/cm²", false),
        (
            3,
            1,
            "Iodine Basis Image",
            1000 .* de_vmi.images.iodine[:, :, mid],
            "mg/mL",
            true,
        ),
        (3, 3, "Water Basis Image", de_vmi.images.water[:, :, mid], "g/mL", true),
    )
    for (row, col, title, data, unit, image) in panels
        range = window(data)
        if image
            slice_panel!(
                fig[row, col],
                data;
                title,
                colormap = :viridis,
                colorrange = range,
            )
        else
            ax = Mke.Axis(fig[row, col]; title, xlabel = "View", ylabel = "Dexel Column")
            Mke.heatmap!(ax, data; colormap = :viridis, colorrange = range)
        end
        Mke.Colorbar(
            fig[row, col + 1];
            colormap = :viridis,
            colorrange = range,
            label = unit,
        )
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.16))
    fig
end

# ╔═╡ 1200000c-0000-4000-8000-000000000001
md"""
## 10. The VMIs

`vmi_pipeline` synthesizes each VMI from the basis pair (McCollough 2015):
``\mu(E) = c_\mathrm{water}\,(\mu/\rho)_\mathrm{water}(E) + c_\mathrm{iodine}\,(\mu/\rho)_\mathrm{iodine}(E)``,
in HU against water at the same energy. The rows are the measured draw without denoising, the
measured draw through the whole chain, and the noise-free reference.
"""

# ╔═╡ 1200000c-0000-4000-8000-000000000010
# The basis pair in the eroded solid water: a perfect decomposition reads water ≈ 1 g/mL (solid
# water is not pure water, so a small offset is expected) and iodine ≈ 0
solid_water_basis = let
    ERODE_PX = 12.0

    sw_bool_raw = (roi_labels .== UInt8(BS.REGION_SOLID_WATER))
    sw_bool = BS.erode_mask_2d(sw_bool_raw; erode_px = ERODE_PX)

    n_raw = count(sw_bool_raw)
    n_eroded = count(sw_bool)
    n_eroded == 0 &&
        error(
            "solid_water_basis: deep erosion (σ = $(ERODE_PX) px) wiped out the SW " *
                "ROI (raw count = $(n_raw)).  Reduce erode_px or check phantom mask.",
        )

    sw_idx = findall(sw_bool)
    _mean(vol) = mean(vol[ci, z] for z in axes(vol, 3) for ci in sw_idx)

    (
        c_water = Float64(_mean(de_vmi.images.water)),
        c_iodine = Float64(_mean(de_vmi.images.iodine)),
        n_voxels = length(sw_idx) * size(de_vmi.images.water, 3),
        mask_2d = collect(sw_bool),
    )
end;

# ╔═╡ 9bca2667-4cba-4aa2-983a-b1141c89ea44
Markdown.parse(
    """
In the eroded solid water the basis pair reads $(round(solid_water_basis.c_water, digits = 4)) g/mL of
water and $(round(1000 * solid_water_basis.c_iodine, digits = 3)) mg/mL of iodine
($(solid_water_basis.n_voxels) voxels).
""",
)

# ╔═╡ 1200000c-0000-4000-8000-000000000020
# The VMI stack as one volume per energy (keV → (nx, ny, nz) HU).
vmi_HU_final = Dict(
    Float64(E) => de_vmi.vmis[:, :, :, k]
    for (k, E) in pairs(de_vmi.energies)
);

# ╔═╡ 1200000c-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1500))

    # what a VMI is: the object as a single energy would show it, from two broad spectra
    ax = Mke.Axis(
        fig[1, 1:6];
        title = "Two Broad Spectra In, One Energy Out",
        xlabel = "Photon Energy (keV)",
        yticksvisible = false,
        yticklabelsvisible = false,
        ygridvisible = false,
    )
    for (protocol, color) in ((protocol_low, COLORS.cyan), (protocol_high, COLORS.violet))
        e, wη = flash_detected_spectrum(protocol)
        Mke.band!(ax, Float64.(e), 0.0, wη ./ maximum(wη); color = (color, 0.25))
    end
    for E in de_vmi_energies
        Mke.lines!(ax, [E, E], [0.0, 1.15]; color = COLORS.green, linewidth = 4)
        Mke.text!(
            ax,
            E,
            1.18;
            text = "$(Int(E)) keV",
            align = (:center, :bottom),
            font = :bold,
            color = COLORS.green,
        )
    end
    Mke.xlims!(ax, 20, 150)
    Mke.ylims!(ax, 0, 1.5)

    mid = (size(de_vmi.vmis, 3) + 1) ÷ 2
    runs = (
        ("No Denoising", de_none),
        ("SpectralHYPR", de_vmi),
        ("Noise-Free Reference", de_ref),
    )
    for (r, (label, run)) in enumerate(runs)
        Mke.Label(
            fig[r + 1, 1],
            label;
            rotation = π / 2,
            fontsize = 22,
            font = :bold,
            tellheight = false,
        )
        for (c, E) in enumerate(de_vmi_energies)
            slice_panel!(
                fig[r + 1, c + 1],
                run.vmis[:, :, mid, c];
                title = r == 1 ? "$(Int(E)) keV" : "",
            )
        end
    end
    Mke.Colorbar(fig[2:4, 6]; colormap = :grays, colorrange = HU_WINDOW, label = "HU")
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.17))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "flash_ufc_vmi_grid.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1200000e-0000-4000-8000-000000000060
md"""
## 11. Water: Accuracy and Noise

The mean and the noise are both measured in the **deeply eroded solid water** (12 pixels from
every edge, as in notebook 03), over every slice: a large region keeps the per-energy statistics
stable where FBP noise is spatially correlated. The noise is measured **against the noise-free
reference**, as the standard deviation of (VMI − reference): the reference shares the projection
and the reconstruction, so the difference is noise alone, with no structure (residual cupping, the
rods' edges) mixed in. The same draw without denoising shows what the chain removes.
"""

# ╔═╡ 1200000e-0000-4000-8000-000000000080
# Mean HU and noise in the eroded solid water at every energy: σ of (VMI − noise-free reference),
# with and without the denoising
vmi_noise_by_keV = let
    roi_idx = findall(solid_water_basis.mask_2d)
    nz_r = size(de_vmi.vmis, 3)
    vals(run, k) = Float64[Float64(run.vmis[ci, z, k]) for z in 1:nz_r, ci in roi_idx]

    out = Dict{Float64, NamedTuple}()
    for (k, E) in enumerate(de_vmi_energies)
        p, n, r = vals(de_vmi, k), vals(de_none, k), vals(de_ref, k)
        out[E] = (
            mean = mean(p),
            reference = mean(r),
            std = std(p .- r),
            std_none = std(n .- r),
            n = length(p),
        )
    end
    out
end;

# ╔═╡ 1200000e-0000-4000-8000-000000000050
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1180))

    sketch = canvas(fig[1:2, 1], (-18, 18, -18, 18))
    sketch.title = "Where: The Solid Water"
    gammex!(sketch; calcium = COLORS.rule, iodine = COLORS.rule)
    roi!(sketch, solid_water_basis.mask_2d, recon_opts.fov_cm)

    Es = de_vmi_energies
    measured = [vmi_noise_by_keV[E].mean for E in Es]
    reference = [vmi_noise_by_keV[E].reference for E in Es]
    n = length(Es)
    ax = Mke.Axis(
        fig[1, 2];
        title = "Solid Water in Every VMI",
        subtitle = "Bars: SpectralHYPR. Diamonds: the Noise-Free Reference",
        xlabel = "VMI Energy (keV)",
        ylabel = "Mean HU",
        xticks = (1:n, ["$(Int(E))" for E in Es]),
    )
    Mke.barplot!(ax, 1:n, measured; color = COLORS.violet)
    Mke.scatter!(ax, 1:n, reference; color = COLORS.ink, marker = :diamond, markersize = 18)
    Mke.hlines!(ax, [0.0]; color = COLORS.ink, linewidth = 1, linestyle = :dash)
    y_max = max(15.0, 1.3 * maximum(abs, vcat(measured, reference)))
    for (k, (m, r)) in enumerate(zip(measured, reference))
        Mke.text!(
            ax,
            k,
            0.62 * y_max;
            text = "$(round(m, digits = 1)) HU\nRef. $(round(r, digits = 1))",
            align = (:center, :center),
            fontsize = 16,
        )
    end
    Mke.ylims!(ax, -y_max, y_max)

    σ_hypr = [vmi_noise_by_keV[E].std for E in Es]
    σ_none = [vmi_noise_by_keV[E].std_none for E in Es]
    ax2 = Mke.Axis(
        fig[2, 2];
        title = "Noise Against Energy",
        subtitle = "σ of (VMI − Noise-Free Reference) in the Same Region",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        xticks = Es,
    )
    curves = (
        (σ_none, "No Denoising", COLORS.muted, :dash, 1),
        (σ_hypr, "SpectralHYPR", COLORS.violet, :solid, -1),
    )
    for (σs, label, color, linestyle, side) in curves
        Mke.scatterlines!(
            ax2,
            Es,
            σs;
            color,
            linestyle,
            markersize = 16,
            linewidth = 3,
            label,
        )
        for (E, σ) in zip(Es, σs)
            Mke.text!(
                ax2,
                E,
                σ;
                text = "$(round(σ; digits = 1))",
                align = (:center, side > 0 ? :bottom : :top),
                offset = (0, 12 * side),
                fontsize = 16,
                color,
            )
        end
    end
    Mke.xlims!(ax2, first(Es) - 15, last(Es) + 15)
    Mke.ylims!(ax2, 0, 1.25 * maximum(σ_none))
    Mke.axislegend(ax2; position = :rt)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    for name in ("flash_ufc_vmi_water_roi.png", "flash_ufc_vmi_water_noise.png")
        Mke.save(joinpath(@__DIR__, "..", "assets", name), fig; px_per_unit = 2)
    end
    fig
end

# ╔═╡ 1200000e-0000-4000-8000-000000000001
md"""
## 12. The Rods Against Theory

Every rod's mean HU in an 8-pixel-radius ROI at its centroid, over every slice, against the
theoretical HU of its material, ``1000\,(\mu_\mathrm{rod}(E) - \mu_\mathrm{water}(E)) / \mu_\mathrm{water}(E)``
from `BS.compute_μ_at_energy`: pure physics, no fitting.
"""

# ╔═╡ 1200000e-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 1200000e-0000-4000-8000-000000000020
ROD_NAMES = (
    Ca = (
        "50 mg/mL",
        "100 mg/mL",
        "200 mg/mL",
        "300 mg/mL",
        "400 mg/mL",
        "500 mg/mL",
        "600 mg/mL",
    ),
    I = (
        "2.0 mg/mL",
        "2.5 mg/mL",
        "5.0 mg/mL",
        "7.5 mg/mL",
        "10.0 mg/mL",
        "15.0 mg/mL",
        "20.0 mg/mL",
    ),
);

# ╔═╡ 1200000e-0000-4000-8000-000000000030
rod_data = let
    materials = phantom_cpu.materials
    mask_2d = roi_labels
    nx, ny = size(mask_2d)
    ROI_RADIUS_PX = 8

    function rod_centroid(label::UInt8)
        idx = findall(==(label), mask_2d)
        isempty(idx) && error("rod_centroid: no voxels with label $label")
        cx = sum(ci -> Float64(ci[1]), idx) / length(idx)
        cy = sum(ci -> Float64(ci[2]), idx) / length(idx)
        return (cx, cy)
    end

    function rod_roi_mask(label::UInt8)
        cx, cy = rod_centroid(label)
        i_lo = max(1, floor(Int, cx - ROI_RADIUS_PX))
        i_hi = min(nx, ceil(Int, cx + ROI_RADIUS_PX))
        j_lo = max(1, floor(Int, cy - ROI_RADIUS_PX))
        j_hi = min(ny, ceil(Int, cy + ROI_RADIUS_PX))
        roi = CartesianIndex{2}[]
        r² = Float64(ROI_RADIUS_PX)^2
        for j in j_lo:j_hi, i in i_lo:i_hi
            ((i - cx)^2 + (j - cy)^2) ≤ r² && push!(roi, CartesianIndex(i, j))
        end
        return roi
    end

    rod_rois = Dict(
        lab => rod_roi_mask(lab)
        for lab in vcat(collect(ROD_LABELS.Ca), collect(ROD_LABELS.I))
    )

    μ_water_E = Dict(
        E => BS.compute_μ_at_energy(BS.XA.Materials.water, E)
        for E in de_vmi_energies
    )

    function theoretical_hu(material, E::Float64)
        μ = BS.compute_μ_at_energy(material, E)
        return 1000.0 * (μ - μ_water_E[E]) / μ_water_E[E]
    end

    function measured_hu(vmi_vol, label::UInt8)
        roi = rod_rois[label]
        return mean(vmi_vol[ci, z] for z in axes(vmi_vol, 3) for ci in roi)
    end

    out = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        labels = ROD_LABELS[group]
        meas = [measured_hu(vmi_HU_final[E], lab) for lab in labels, E in de_vmi_energies]
        theo = [
            theoretical_hu(materials[Int(lab) + 1], E) # material entry L + 1 is label L
            for lab in labels, E in de_vmi_energies
        ]
        out[group] = (
            labels = labels,
            names = ROD_NAMES[group],
            measured = meas,
            theoretical = theo,
        )
    end
    out
end;

# ╔═╡ ab98e7b4-a3e6-418e-8bf6-b33f6d4c323d
# The least-squares line of measured against theoretical HU, per rod group and energy
rod_fits = let
    function fit_lr(x, y)
        x̄, ȳ = mean(x), mean(y)
        β = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        ss_res = sum((y .- ŷ).^2)
        (
            slope = β,
            intercept = α,
            r² = 1 - ss_res / sum((y .- ȳ).^2),
            rmse = sqrt(ss_res / length(y)),
        )
    end
    Dict(
        (group, E) => fit_lr(
            rod_data[group].theoretical[:, j],
            rod_data[group].measured[:, j],
        )
        for group in (:Ca, :I)
        for (j, E) in enumerate(de_vmi_energies)
    )
end;

# ╔═╡ 1200000f-0000-4000-8000-000000000010
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1180))
    shades(c) = [
        Mke.RGBf(Mke.weighted_color_mean(0.35 + 0.65 * (k - 1) / 6, c, COLORS.paper))
        for k in 1:7
    ]
    colors = (Ca = shades(COLORS.red), I = shades(COLORS.green))

    sketch = canvas(fig[1:2, 1], (-19, 19, -19, 19))
    sketch.title = "The Rods, in mg/mL"
    gammex!(sketch; calcium = colors.Ca, iodine = colors.I)
    for (group, centres) in ((:Ca, GAMMEX.calcium), (:I, GAMMEX.iodine))
        for (k, c) in enumerate(centres)
            Mke.text!(
                sketch,
                c[1],
                c[2] - GAMMEX.rod_r - 0.3;
                text = replace(ROD_NAMES[group][k], " mg/mL" => ""),
                align = (:center, :top),
                fontsize = 14,
            )
        end
    end
    Mke.text!(
        sketch,
        0,
        -18.5;
        text = "Calcium (Red) and Iodine (Green)",
        align = (:center, :bottom),
        fontsize = 15,
        color = COLORS.muted,
    )

    for (col, (group, name)) in enumerate(((:Ca, "Calcium Rods"), (:I, "Iodine Rods")))
        d = rod_data[group]
        ax = Mke.Axis(
            fig[1, col + 1];
            title = name,
            subtitle = "Solid: Measured. Dashed: Theory",
            xlabel = "VMI Energy (keV)",
            ylabel = col == 1 ? "HU" : "",
            xticks = de_vmi_energies,
        )
        for i in eachindex(d.names)
            Mke.scatterlines!(
                ax,
                de_vmi_energies,
                d.measured[i, :];
                color = colors[group][i],
                markersize = 10,
            )
            Mke.lines!(
                ax,
                de_vmi_energies,
                d.theoretical[i, :];
                color = colors[group][i],
                linestyle = :dash,
                linewidth = 1.8,
            )
        end

        ax2 = Mke.Axis(
            fig[2, col + 1];
            title = "$(name): Measured Against Theory",
            xlabel = "Theoretical HU",
            ylabel = col == 1 ? "Measured HU" : "",
        )
        hi = 1.05 * max(maximum(d.measured), maximum(d.theoretical))
        Mke.lines!(
            ax2,
            [0.0, hi],
            [0.0, hi];
            color = COLORS.ink,
            linestyle = :dash,
            linewidth = 1.5,
        )
        for (j, E) in enumerate(de_vmi_energies)
            Mke.scatter!(
                ax2,
                d.theoretical[:, j],
                d.measured[:, j];
                color = THEME.palette.color[][j],
                markersize = 13,
                label = "$(Int(E)) keV",
            )
        end
        col == 2 && Mke.axislegend(ax2; position = :rb)
    end
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.34))
    for name in ("flash_ufc_vmi_vs_theoretical.png", "flash_ufc_vmi_regression.png")
        Mke.save(joinpath(@__DIR__, "..", "assets", name), fig; px_per_unit = 2)
    end
    fig
end

# ╔═╡ 1200000f-0000-4000-8000-000000000030
let
    group_name = Dict(:Ca => "Calcium", :I => "Iodine")
    rows = [
        let f = rod_fits[(group, E)]
            "| $(group_name[group]) | $(Int(E)) | $(round(f.slope, digits = 3)) | " *
                "$(round(f.intercept, digits = 1)) | $(round(f.r², digits = 4)) | $(round(f.rmse, digits = 1)) |"
        end
        for group in (:Ca, :I)
        for E in de_vmi_energies
    ]
    Markdown.parse(
        """
    The line through each group's seven rods at each energy (measured = slope × theory + intercept):

    | Rods | VMI (keV) | Slope | Intercept (HU) | R² | RMSE (HU) |
    |---|---:|---:|---:|---:|---:|
    $(join(rows, "\n"))
    """,
    )
end

# ╔═╡ 12000010-0000-4000-8000-000000000001
md"""
## 13. Verification

Automated pass / fail gates over both modes (notebook 01's convention):

1. **Regular water accuracy**: |mean HU| ≤ 5 in tube A, tube B and the combined image.
2. **Dual-power √2**: ``\sigma_\mathrm{combined} / \sigma_\mathrm{single}`` in [0.62, 0.80] (ideal
   0.707; fails if the tube seeds were ever shared).
3. **Dual-energy water accuracy**: |mean HU| ≤ 5 in the low, high and mixed images.
4. **VMI water accuracy**: |mean HU| ≤ 10 at every energy, in the eroded solid water.
5. **VMI bias against the noise-free reference**: the chain's water mean within 2 HU of the
   reference's at every energy; the denoising moves the noise, not the CT number.
6. **VMI noise flat in energy**: max / min σ over 50–140 keV ≤ 1.10.
7. **The denoising lowers the noise**: σ below the undenoised reconstruction's at every energy.
8. **The chain that ran**: both tubes in the likelihood, projection and image HYPR, ACNR on the
   complement, the fitted `PairFilter`, the calibration draw's pair and composite energy, view
   integration (`de_vmi.settings`, `FLASH.opts`).
9. **Per-rod regression**: slope in [0.85, 1.15] and R² ≥ 0.99 at every energy, both rod groups.

<details><summary>Why the noise gate is flatness</summary>

Without denoising the VMI noise is U-shaped: highest at 50 keV, lowest near the minimum-noise
energy ``E^*``, rising again above it. The chain keeps the composite (the VMI at ``E^*``) as
reconstructed and pools only the complement, which carries the noise that grows away from ``E^*``;
what remains in every VMI is mostly the composite's noise, and the curve flattens (the undenoised
curve's max / min is printed beside the gate). The gate this notebook used before, σ strictly
decreasing from 50 to 140 keV, described the earlier chain and no longer holds; the check for this
chain is that the noise stays near its minimum at every energy.
</details>
"""

# ╔═╡ 12000010-0000-4000-8000-000000000010
verification = let
    checks = Tuple{String, Bool, String}[]

    # 1. Regular (dual-power) water accuracy
    for (tag, s) in (
            ("tube A", dp_stats.water.a),
            ("tube B", dp_stats.water.b),
            ("combined", dp_stats.water.combined),
        )
        push!(
            checks,
            (
                "regular water $(tag)",
                abs(s.mean) ≤ 5.0,
                "mean HU = $(round(s.mean, digits = 2)) (gate ±5)",
            ),
        )
    end

    # 2. Dual-power √2 noise reduction
    push!(
        checks,
        (
            "dual-power σ ratio",
            0.62 ≤ dp_stats.noise_ratio ≤ 0.8,
            "`σ_combined / σ_single` = $(round(dp_stats.noise_ratio, digits = 3)) (ideal 0.707, gate [0.62, 0.80])",
        ),
    )

    # 3. DE poly water accuracy
    for (tag, s) in (
            ("80 kVp", poly_water_stats.low),
            ("Sn140 kVp", poly_water_stats.high),
            ("mixed", poly_water_stats.mixed),
        )
        push!(
            checks,
            (
                "DE poly water $(tag)",
                abs(s.mean) ≤ 5.0,
                "mean HU = $(round(s.mean, digits = 2)) (gate ±5)",
            ),
        )
    end

    # 4. VMI water accuracy per keV
    for E in de_vmi_energies
        m = vmi_noise_by_keV[E].mean
        push!(
            checks,
            (
                "VMI water @ $(Int(E)) keV",
                abs(m) ≤ 10.0,
                "mean HU = $(round(m, digits = 2)) (gate ±10)",
            ),
        )
    end

    # 5. Bias against the noise-free reference (same projection, same reconstruction)
    bias = [
        vmi_noise_by_keV[E].mean - vmi_noise_by_keV[E].reference
        for E in de_vmi_energies
    ]
    push!(
        checks,
        (
            "VMI bias vs noise-free reference",
            maximum(abs, bias) ≤ 2.0,
            "mean HU − reference = " *
                join([string(round(b, digits = 2)) for b in bias], ", ") *
                " (gate ±2)",
        ),
    )

    # 6. The noise is flat in energy: the chain carries the composite's noise to every VMI
    σs = [vmi_noise_by_keV[E].std for E in de_vmi_energies]
    σn_all = [vmi_noise_by_keV[E].std_none for E in de_vmi_energies]
    flat = maximum(σs) / minimum(σs)
    push!(
        checks,
        (
            "VMI noise flat in energy",
            flat ≤ 1.1,
            "σ = " *
                join([string(round(σ, digits = 1)) for σ in σs], ", ") *
                " HU, max / min = $(round(flat, digits = 3)) (gate ≤ 1.10; no denoising " *
                "$(round(maximum(σn_all) / minimum(σn_all), digits = 2)))",
        ),
    )

    # 7. The denoising lowers the noise at every keV
    σn = [vmi_noise_by_keV[E].std_none for E in de_vmi_energies]
    push!(
        checks,
        (
            "denoising lowers σ",
            all(σs .< σn),
            "σ / σ_none = " *
                join([string(round(a / b, digits = 2)) for (a, b) in zip(σs, σn)], ", "),
        ),
    )

    # 8. The chain that ran
    chain = de_vmi.settings
    push!(
        checks,
        (
            "VMI chain",
            chain.n_channels == 2 &&
                chain.denoiser !== nothing &&
                chain.denoiser.projection !== nothing &&
                chain.denoiser.image !== nothing &&
                chain.acnr !== nothing &&
                chain.acnr.on === :complement &&
                chain.recon.filter isa BS.PairFilter &&
                chain.pair.basis == calibration.pair_basis &&
                chain.pair.Estar == calibration.composite_energy &&
                FLASH.opts.view_samples == 5 &&
                all(e -> e.view_arc == 1.0, FLASH.exposures),
            "K = $(chain.n_channels), projection + image HYPR, ACNR on the `$(chain.acnr === nothing ? "none" : chain.acnr.on)`, " *
                "`PairFilter`, calibration pair E* = $(round(chain.pair.basis.Estar, digits = 1)) keV, " *
                "composite $(round(chain.pair.Estar, digits = 1)) keV, `view_samples` = $(FLASH.opts.view_samples)",
        ),
    )

    # 9. Per-rod regression gates
    function _fit(x, y)
        x̄ = mean(x)
        ȳ = mean(y)
        β = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
        α = ȳ - β * x̄
        ŷ = α .+ β .* x
        r² = 1 - sum((y .- ŷ).^2) / sum((y .- ȳ).^2)
        (slope = β, r² = r²)
    end
    for group in (:Ca, :I), (j, E) in pairs(de_vmi_energies)
        d = rod_data[group]
        f = _fit(
            Vector{Float64}(vec(d.theoretical[:, j])),
            Vector{Float64}(vec(d.measured[:, j])),
        )
        push!(
            checks,
            (
                "$(group) regression @ $(Int(E)) keV",
                (0.85 ≤ f.slope ≤ 1.15) && (f.r² ≥ 0.99),
                "slope = $(round(f.slope, digits = 3)), R² = $(round(f.r², digits = 4)) (gates slope [0.85, 1.15], R² ≥ 0.99)",
            ),
        )
    end

    n_pass = count(c -> c[2], checks)
    (
        checks = checks,
        n_pass = n_pass,
        n_total = length(checks),
        pass = n_pass == length(checks),
    )
end;

# ╔═╡ 12000010-0000-4000-8000-000000000020
let
    status(ok) = ok ? "✅ pass" : "❌ CHECK (outside bounds)"
    rows = join(
        [
            "| $(name) | $(detail) | $(status(ok)) |"
            for (name, ok, detail) in verification.checks
        ],
        "\n",
    )
    Markdown.parse(
        """
### Verification: $(verification.n_pass) of $(verification.n_total) checks pass

| Check | Value | Status |
|---|---|---|
$rows
""",
    )
end

# ╔═╡ 12000011-0000-4000-8000-000000000001
let
    fmt(v) = join(round.(v, digits = 1), " / ")
    Es = de_vmi_energies
    water = [vmi_noise_by_keV[E].mean for E in Es]
    σ = [vmi_noise_by_keV[E].std for E in Es]
    σ_none = [vmi_noise_by_keV[E].std_none for E in Es]
    slopes(group) = join([round(rod_fits[(group, E)].slope, digits = 3) for E in Es], " / ")
    rmses(group) = fmt([rod_fits[(group, E)].rmse for E in Es])
    w, p = dp_stats.water, poly_water_stats
    Markdown.parse(
        """
    ## Summary

    The Definition Flash modelled as itself, not as a re-badged Force: its own Geant4 detector
    table, its published geometry (595 / 1085.6 mm, 64 × 736 dexels at 0.70473 mm), anode angle
    (7°) and flat filtration (8.4 mm Al equivalent), and the physical Flash scan's dual-energy
    protocol (80 / Sn140 kVp with 0.4 mm Sn, dose-matched to its CTDIvol).

    | Result | Value |
    |---|---|
    | Regular water, tube A / tube B / dual power | $(fmt([w.a.mean, w.b.mean, w.combined.mean])) HU |
    | Dual-power noise ratio (ideal 0.707) | $(round(dp_stats.noise_ratio, digits = 3)) |
    | Dual-energy water, tube A / tube B / mixed | $(fmt([p.low.mean, p.high.mean, p.mixed.mean])) HU |
    | VMI water, 50 / 70 / 100 / 140 keV | $(fmt(water)) HU |
    | VMI noise, SpectralHYPR (no denoising) | $(fmt(σ)) HU ($(fmt(σ_none)) HU) |
    | Calcium rods, slope / RMSE against theory | $(slopes(:Ca)) / $(rmses(:Ca)) HU |
    | Iodine rods, slope / RMSE against theory | $(slopes(:I)) / $(rmses(:I)) HU |
    | Verification | $(verification.n_pass) of $(verification.n_total) checks pass |

    The remaining gaps (bowtie profile, crystal depth, fill factor, electronic noise, tube B's
    z-offset, the z flying focal spot) are the documented assumptions of section 3, the open items
    on the parity checklist of the
    [SOMATOM Definition Flash entry of the scanners page](../../scanners/#somatom-definition-flash).
    The table, `UFC_FLASH_MC_EFFICIENCY_LUT`, is plain data in `src/detector/eid_response_tables.jl`
    and the scanner, `SOMATOM_DEFINITION_FLASH`, lives in `src/geometry/clinical_scanners.jl`; both
    are covered by `test/detector.jl` and `test/eict_signal.jl`.
    """,
    )
end

# ╔═╡ Cell order:
# ╟─12000001-0000-4000-8000-000000000010
# ╟─117658a2-649f-4f7a-9441-8cb8bf6e5a64
# ╟─b0517860-9285-4fa2-9032-68d98332bb37
# ╟─96e9486c-0785-4c59-a2a4-696981dd9661
# ╟─12000001-0000-4000-8000-000000000020
# ╠═12000001-0000-4000-8000-000000000001
# ╠═12000001-0000-4000-8000-000000000002
# ╠═12000001-0000-4000-8000-000000000003
# ╠═12000001-0000-4000-8000-000000000030
# ╠═12000001-0000-4000-8000-000000000031
# ╠═12000001-0000-4000-8000-000000000032
# ╠═12000001-0000-4000-8000-000000000033
# ╠═12000001-0000-4000-8000-000000000040
# ╟─12000001-0000-4000-8000-000000000050
# ╟─c6d3297b-beb3-4190-97dc-18747624fa34
# ╠═14b7f814-8c4b-454a-aabd-9b3f4044b33c
# ╠═5c4ece81-ba40-40e1-8620-3678e69bd43e
# ╠═66da3212-c888-4531-923f-6c90a2595573
# ╠═37690342-2920-4aac-89eb-e17113140f4c
# ╠═5295bbac-514c-4a8e-9fb8-18b31c28c0ee
# ╠═dd8cb654-bc25-45c7-b74f-1024ad950ce3
# ╠═a41b173b-f928-47e6-ba27-b5bf181124ed
# ╠═fab85396-95ae-4bad-bb1d-74eee0da3059
# ╠═d1d2e0b0-8f50-4be1-bd48-ca55bcc2efb4
# ╟─12000002-0000-4000-8000-000000000001
# ╟─12000002-0000-4000-8000-000000000030
# ╟─12000003-0000-4000-8000-000000000001
# ╠═12000003-0000-4000-8000-000000000010
# ╠═12000003-0000-4000-8000-000000000020
# ╟─12000004-0000-4000-8000-000000000001
# ╟─6cf27931-9dad-4fd6-afad-aa450e45f4e6
# ╠═12000004-0000-4000-8000-000000000010
# ╟─a4f39820-16e5-5809-b70a-5b3f64c6fb38
# ╟─12000006-0000-4000-8000-000000000001
# ╠═12000006-0000-4000-8000-000000000010
# ╟─12000005-0000-4000-8000-000000000001
# ╠═12000005-0000-4000-8000-000000000005
# ╠═12000005-0000-4000-8000-000000000010
# ╠═12000005-0000-4000-8000-000000000020
# ╠═12000005-0000-4000-8000-000000000030
# ╟─12000005-0000-4000-8000-000000000040
# ╠═12000006-0000-4000-8000-000000000020
# ╠═12000006-0000-4000-8000-000000000025
# ╠═12000006-0000-4000-8000-000000000030
# ╟─12000006-0000-4000-8000-000000000040
# ╟─12000007-0000-4000-8000-000000000001
# ╠═12000007-0000-4000-8000-000000000005
# ╠═12000007-0000-4000-8000-000000000008
# ╠═12000007-0000-4000-8000-000000000010
# ╠═12000007-0000-4000-8000-000000000015
# ╠═12000007-0000-4000-8000-000000000020
# ╠═12000007-0000-4000-8000-000000000025
# ╠═12000007-0000-4000-8000-000000000030
# ╟─12000007-0000-4000-8000-000000000040
# ╟─12000008-0000-4000-8000-000000000001
# ╠═12000006-0000-4000-8000-000000000012
# ╠═12000008-0000-4000-8000-000000000005
# ╠═12000008-0000-4000-8000-000000000020
# ╟─12000008-0000-4000-8000-000000000030
# ╠═12000008-0000-4000-8000-000000000035
# ╟─12000008-0000-4000-8000-000000000040
# ╟─12000009-0000-4000-8000-000000000001
# ╠═12000009-0000-4000-8000-000000000010
# ╠═12000009-0000-4000-8000-000000000012
# ╟─12000009-0000-4000-8000-000000000015
# ╠═12000009-0000-4000-8000-000000000020
# ╠═12000009-0000-4000-8000-000000000025
# ╠═12000009-0000-4000-8000-000000000028
# ╠═12000009-0000-4000-8000-000000000030
# ╟─12000009-0000-4000-8000-000000000040
# ╟─12000009-0000-4000-8000-000000000060
# ╟─1200000a-0000-4000-8000-000000000001
# ╠═1200000a-0000-4000-8000-000000000010
# ╟─1200000a-0000-4000-8000-000000000015
# ╟─1200000b-0000-4000-8000-000000000001
# ╠═1200000b-0000-4000-8000-000000000005
# ╠═1200000b-0000-4000-8000-000000000006
# ╠═1200000b-0000-4000-8000-000000000007
# ╠═1200000c-0000-4000-8000-000000000015
# ╠═1200000b-0000-4000-8000-000000000008
# ╠═1200000b-0000-4000-8000-000000000010
# ╠═1200000b-0000-4000-8000-000000000011
# ╟─1200000b-0000-4000-8000-000000000015
# ╟─1200000a-0000-4000-8000-000000000040
# ╟─1200000c-0000-4000-8000-000000000001
# ╠═1200000c-0000-4000-8000-000000000010
# ╟─9bca2667-4cba-4aa2-983a-b1141c89ea44
# ╠═1200000c-0000-4000-8000-000000000020
# ╟─1200000c-0000-4000-8000-000000000040
# ╟─1200000e-0000-4000-8000-000000000060
# ╠═1200000e-0000-4000-8000-000000000080
# ╟─1200000e-0000-4000-8000-000000000050
# ╟─1200000e-0000-4000-8000-000000000001
# ╠═1200000e-0000-4000-8000-000000000010
# ╠═1200000e-0000-4000-8000-000000000020
# ╠═1200000e-0000-4000-8000-000000000030
# ╠═ab98e7b4-a3e6-418e-8bf6-b33f6d4c323d
# ╟─1200000f-0000-4000-8000-000000000010
# ╟─1200000f-0000-4000-8000-000000000030
# ╟─12000010-0000-4000-8000-000000000001
# ╠═12000010-0000-4000-8000-000000000010
# ╟─12000010-0000-4000-8000-000000000020
# ╟─12000011-0000-4000-8000-000000000001
