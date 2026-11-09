### A Pluto.jl notebook ###
# v0.2.1

using Markdown
using InteractiveUtils

# ╔═╡ 09000001-0000-4000-8000-000000000010
md"""
# Siemens SOMATOM Force: Dual-Source Dual Energy

**One dual-source scan of the Gammex 472 on the Siemens SOMATOM Force, 100 kVp on tube A and
Sn140 kVp on tube B, read two ways: as the routine mixed image and as virtual monoenergetic
images (VMIs).**

The Force's UFC scintillator enters through BasisMonteCarlo's Geant4 table, and the VMI chain is
the one of the basis-spectral-denoising study: one projection per tube shared by every noise draw,
a noise-free reference, `SpectralHYPR`, and the spectral pair fixed from a calibration draw.
"""

# ╔═╡ ef027fa5-f75b-435c-a792-6ef688dcc445
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scanner", "Two Readouts"],
        [
            "Gammex 472: Calcium and Iodine Rods",
            "SOMATOM Force: Two Tubes, 95° Apart",
            "One Scan, a Mixed Image and VMIs",
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
    box((5.0, 2.2), "100 / Sn140 kVp", "One Dual-Source Scan", COLORS.cyan; w = 6.0)
    box((2.4, -1.6), "Mixed Image", "0.5 A + 0.5 B", COLORS.green)
    box((7.6, -1.6), "VMIs", "50 to 140 keV", COLORS.violet)
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

# ╔═╡ b088bb14-1243-4c2a-a097-3d2264ab01f4
md"""
The Force is not inherently spectral: routine protocols run both tubes at the same kVp, and dual
energy is a mode. In that mode there is no third, plain scan; the routine-looking grey-scale
image is the **mixed image**, a weighted blend of the two tubes' reconstructions. Both readouts
below come from the same acquisition:
"""

# ╔═╡ 98cfe0df-2f67-4e5b-a60d-9d66f85474e8
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 600))
    nodes = Dict(
        :phantom => (0.0, 2.6, "Gammex 472", "The Phantom"),
        :scanner => (0.0, 1.3, "SOMATOM_FORCE", "UFC Geant4 Table"),
        :protocols => (0.0, 0.0, "Two Protocols", "100 kVp A, Sn140 kVp B"),
        :simulate => (3.9, 1.3, "simulate!", "One Projection per Tube", COLORS.cyan),
        :calibration => (3.9, -1.3, "Calibration Draw", "Fixes the Pair", COLORS.beam),
        :bhc => (7.8, 2.6, "BHC + FDK", "Each Tube Alone", COLORS.cyan),
        :mixed => (11.7, 2.6, "Mixed Image", "0.5 A + 0.5 B", COLORS.green),
        :vmi => (7.8, 0.0, "vmi_pipeline", "SpectralHYPR", COLORS.cyan),
        :vmis => (11.7, 0.0, "VMIs", "50 / 70 / 100 / 140 keV", COLORS.green),
    )
    edges = [
        :phantom => :simulate,
        :scanner => :simulate,
        :protocols => :simulate,
        :simulate => :bhc,
        :simulate => :vmi,
        :simulate => :calibration,
        :calibration => :vmi,
        :bhc => :mixed,
        :vmi => :vmis,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.7, 13.4, -2.0, 3.3))
    fig
end

# ╔═╡ 09000001-0000-4000-8000-000000000020
md"""
## Setup
"""

# ╔═╡ 09000001-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 09000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 09000001-0000-4000-8000-000000000003
using Statistics: mean, std, quantile

# ╔═╡ 09000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 09000001-0000-4000-8000-000000000031
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 09000001-0000-4000-8000-000000000032
import PlutoUI

# ╔═╡ 09000001-0000-4000-8000-000000000033
PlutoUI.TableOfContents()

# ╔═╡ 09000001-0000-4000-8000-000000000040
begin
    import GPUSelect
    # the backend array type, directly: MtlArray / CuArray / ROCArray
    AT = GPUSelect.Storage()
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 09000001-0000-4000-8000-000000000050
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ bd293b7a-2073-468e-845e-51cf5f0e5f5d
md"""
## Figure Style

BasisSimulator's figure kit (colours, theme, flow charts, the gantry and the locator band),
shared by every example notebook, then this notebook's own drawings: the Gammex 472, a region on
it, and one tube of the gantry to scale.
"""

# ╔═╡ 002ed623-0e86-454f-ae59-1f1a73913c6d
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

# ╔═╡ 1c5b3e94-e46f-48fd-8454-4e3bfaa3daa6
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

# ╔═╡ 5b42f9a3-9156-4db8-8534-261a7a3d822a
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

# ╔═╡ 4179f27b-105e-4908-9da1-f92aa0c792b4
const HU_WINDOW = (-200, 500); # every HU image's display window: soft tissue

# ╔═╡ 4e0524a6-23fa-4dd3-9f9e-f8613e5fd1ad
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

# ╔═╡ 6833d0e8-0227-4237-8976-7ca008cf3850
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

# ╔═╡ ed4a09d8-8c8b-4c05-a63a-647ad4f4b0cd
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

# ╔═╡ 0b4226a2-1912-4dca-8417-932a0c3f8d90
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

# ╔═╡ d0ae6214-610c-4431-9538-53c963eb16e8
"""
One tube of a CT gantry to scale (cm, the isocentre at the origin): the focal spot at `sid` and
angle `θ` (degrees), the fan that covers a field of view of diameter `fov`, and its detector arc at
`sdd` from the focal spot. (The kit's `gantry!` gives every tube one fan; the Force's two differ.)
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

# ╔═╡ 09000002-0000-4000-8000-000000000001
md"""
## 1. The UFC Detector Table

The Force's detector is UFC, Siemens' Gd₂O₂S ceramic, 1.4 mm deep. Its response is
`BS.UFC_MC_EFFICIENCY_LUT`, a Geant4 flat-field run (BasisMonteCarlo notebook
`04_siemens_force.jl`, 100 000 photons per keV from 1 to 140 keV): the `efficiency` is the energy
a dexel collects per photon divided by the photon's energy. Three features are fluorescence and
transmission, which Beer–Lambert absorption cannot model:

1. **The Gd K-edge, 50.24 keV**: the efficiency falls from 0.935 to 0.791 between 50 and 51 keV.
   Above the edge a photon is absorbed more readily, but the 43 keV Gd K x ray it frees often
   escapes, carrying its energy away.
2. **Gd L-edge structure near 7–8 keV**: the small dip at 8 keV.
3. **The high-energy roll-off**, 0.857 at 100 keV to 0.612 at 140 keV, as photons pass through
   1.4 mm or Compton-scatter out.

<details><summary>How the table enters the simulation</summary>

`SOMATOM_FORCE` has `detector_material = :ufc` and, by default, `use_detector_efficiency = true`.
A photon of energy ``E`` then leaves a mean signal ``m_1 = E\,\mathrm{efficiency}(E)`` and a mean
square ``m_2 = m_1^2 / \mathrm{dqe0}(E)`` (`dqe0`, the zero-frequency DQE of one energy, adds the
spread of the collected energy to the absorption). The detector reads
``\sum_E w(E)\, m_1(E)\, e^{-\int \mu\, dl}`` with variance ``\sum_E w(E)\, m_2(E)\, e^{-\int \mu\, dl}``.
The spectral basis (`spectral_basis(ws, protocol)`) and the beam-hardening calibration
(`resolve_source_spectrum_full`) weight the spectrum by the same ``m_1``, so the forward and
inverse spectral models match exactly.
</details>
"""

# ╔═╡ 09000002-0000-4000-8000-000000000030
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

    ufc, gemstone = BS.UFC_MC_EFFICIENCY_LUT, BS.GEMSTONE_MC_EFFICIENCY_LUT
    ax = Mke.Axis(
        fig[2, 1];
        title = "The Energy Collected per Photon, Geant4",
        subtitle = "UFC (Force) Against GE's Gemstone, Every Dexel Lit",
        xlabel = "Photon Energy (keV)",
        ylabel = "Collected Fraction η(E)",
    )
    Mke.lines!(
        ax,
        ufc.energies,
        ufc.efficiency;
        color = COLORS.violet,
        linewidth = 3,
        label = "UFC, 1.4 mm (UFC_MC_EFFICIENCY_LUT)",
    )
    Mke.lines!(
        ax,
        gemstone.energies,
        gemstone.efficiency;
        color = COLORS.blue,
        linewidth = 3,
        label = "Gemstone (GEMSTONE_MC_EFFICIENCY_LUT)",
    )
    Mke.vlines!(ax, [50.24]; color = COLORS.beam, linestyle = :dot, linewidth = 2)
    Mke.text!(
        ax,
        51.5,
        0.6;
        text = "Gd K-Edge, 50.24 keV",
        align = (:left, :bottom),
        fontsize = 16,
        color = COLORS.beam,
    )
    Mke.ylims!(ax, 0.55, 1.02)
    Mke.Legend(fig[3, 1], ax; orientation = :horizontal)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.27))
    fig
end

# ╔═╡ 09000003-0000-4000-8000-000000000001
md"""
## 2. The Phantom: Gammex 472

A 33 cm solid-water cylinder with seven calcium rods (50–600 mg/mL) on an inner ring and seven
iodine rods (2–20 mg/mL) on an outer ring, voxelized at 512² over 35 cm, 16 slices over 1 cm.
"""

# ╔═╡ 09000003-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 09000003-0000-4000-8000-000000000020
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 09000004-0000-4000-8000-000000000001
md"""
## 3. The Scanner: SOMATOM Force

Third-generation dual source: two Vectron tubes and two Stellar Infinity UFC detectors, 95° apart
in one gantry. Detector A covers a 50 cm field, detector B 35.5 cm. The geometry was measured on a
clinical Force by Wang et al.

<details><summary>Every number and its source</summary>

| Parameter | Value | Source |
|-----------|-------|--------|
| Source–isocenter (SID) | 595.0 mm | Wang et al. 2021, arXiv:2001.09471 §IV |
| Source–detector (SDD)  | 1085.6 mm | Wang et al. 2021 |
| Tube A–B angular offset | 95° (z-offset 0.88 mm) | Siemens Force whitepaper; Wang et al. |
| Detector A channels | 920 (50 cm FOV) | Wang et al.; whitepaper §1.2 |
| Detector B channels | 640 (35.5 cm FOV) | Wang et al.; Flohr/Schmidt DE-DSCT chapter |
| Rows | 96 × 0.6 mm at iso (57.6 mm z-coverage) | whitepaper §1.2 |
| Column pitch at iso | 0.054°/ch ≈ 0.561 mm | Wang et al. (1.023 mm at detector / 1.825× mag) |
| Scintillator | UFC Gd₂O₂S:Pr,Ce, ρ ≈ 7.34 g/cm³ | Siemens UFC page; Rossner & Grabmaier 1991 |
| Tube (Vectron) | 2 × 120 kW, 70–150 kVp / 10 kV steps, ≤1300 mA | whitepaper §1.1; datasheet |
| Focal spots | 0.4×0.5 mm² smallest; ~0.8 / 1.2 mm nominal | whitepaper; Duan AAPM 2014 |
| Tin filter | 0.6 mm Sn on the high-kV tube (SnX modes) | Faby et al. Siemens DECT white paper; PMC12745887 |
| Rotation | 0.25 s min (0.5 s routine) | whitepaper |
| Projections | 1160 / rotation per focal-spot position | Flohr et al. Med Phys 2005 (Siemens family figure) |
</details>

<details><summary>Modelling assumptions where no public source exists</summary>

- **Anode angle**: Vectron's is unpublished, so the IPEM **8°** spectrum (typical CT anodes are
  7–9°).
- **Flat filtration**: unpublished, so **3.0 mm Al + 0.9 mm Ti**, the Vectron-family stack the
  NAEOTOM Alpha model uses (notebook 08). The 0.6 mm Sn is added on tube B only.
- **Bowtie**: Siemens' body bowtie is unpublished, so CatSim's **large-body** profile stands in
  (as in notebooks 04 and 08).
- **Scintillator depth 1.4 mm, fill factor 0.9**: proprietary. 1.4 mm is the only UFC thickness
  Siemens quotes and the depth of the Geant4 table. The fill factor scales the quanta a dexel
  collects, so it sets the noise, and cancels in the air calibration.
- **Electronic noise 0**: the Stellar's TrueSignal ASIC has no published absolute noise figure;
  its design point is "electronic noise negligible against quantum noise" (Duan et al., AJR 2013).
</details>

<details><summary>Two tubes as two co-registered scans</summary>

As notebook 03 models rapid kVp switching as two scans, the Force's two tubes are **two
`CTProtocol`s run on one `EICTScanner`**, each with its own noise seed:

- **Detector arc**: tube B gets detector A's 920-column arc, so the low- and high-energy sinograms
  are co-registered ray by ray for the decomposition. The 33 cm Gammex fits inside detector B's
  real 35.5 cm field, so no ray used here is missing on the real detector B.
- **The 95° offset**: both scans cover a full rotation on the same angle grid, which is what the
  clinical rebinning produces when it aligns B onto A; a constant angular offset does nothing to a
  full-rotation axial scan.
- **Tube B's 0.88 mm z-offset**: modelled by shifting the phantom origin −0.88 mm in z for tube B
  (section 6). For the z-invariant Gammex this is exactly nothing, but z-varying phantoms (XCAT,
  QRM) inherit the real misalignment.
- **Collimation**: the Force reads 128 × 0.6 mm in dual-energy mode; this notebook uses 4.8 mm
  (8 × 0.6 mm), the thin-collimation equivalent that fits the 1 cm Gammex, as notebook 03 uses 5 mm.
</details>
"""

# ╔═╡ 378aa523-1fb9-4e69-a0d4-22d1ea910d1e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 760))
    ax = canvas(fig[1, 1], (-75, 75, -72, 72))
    ax.title = "The SOMATOM Force, to Scale"
    Mke.poly!(ax, Mke.Circle(Mke.Point2f(0, 0), 68.0); color = COLORS.card)
    Mke.poly!(ax, Mke.Circle(Mke.Point2f(0, 0), 39.0); color = COLORS.paper)
    tube_fan!(ax, 90.0; fov = 50.0, color = COLORS.cyan, label = "Tube A")
    tube_fan!(ax, 185.0; fov = 35.5, color = COLORS.violet, label = "Tube B")
    gammex!(ax)
    fields = ((25.0, "Field of A, 50 cm", -25.6), (17.75, "Field of B, 35.5 cm", 19.0))
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
        ("Detector A", "920 Columns, 50 cm"),
        ("Detector B", "640 Columns, 35.5 cm"),
        ("Rows", "96 × 0.6 mm"),
        ("Scintillator", "UFC Gd₂O₂S, 1.4 mm"),
    ]
    for (k, (name, value)) in enumerate(rows)
        y = 8.6 - 1.1 * (k - 1)
        Mke.text!(spec, 0.2, y; text = name, align = (:left, :center), color = COLORS.muted)
        Mke.text!(spec, 9.8, y; text = value, align = (:right, :center), font = :bold)
    end
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.38))
    fig
end

# ╔═╡ 09000004-0000-4000-8000-000000000010
# Tube and detector A's geometry, shared by both modelled tubes (see above)
scanner = BS.SOMATOM_FORCE;

# ╔═╡ a25dda2c-bb62-534b-be16-0f1de34a44f1
# Every keyword of `scanner`: the documented constant `SOMATOM_FORCE`, with any field this notebook changes marked
let s = scanner, constant = BS.SOMATOM_FORCE
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
        **`scanner` is `$(nameof(typeof(s)))`, from the documented constant `SOMATOM_FORCE`.** Its keywords
        and their values (units and sources: the constant's docstring, `?BS.SOMATOM_FORCE`, and the Scanners
        page). A variant is the same struct with keywords replaced,
        `$(nameof(typeof(s)))(BS.SOMATOM_FORCE; field = value)`.

        | Keyword | Value | |
        |---|---|---|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 09000005-0000-4000-8000-000000000001
md"""
## 4. The Protocols: One Dual-Energy Acquisition

The Force's clinical abdomen pairs are x / Sn150 (x from 70 to 100 kVp, 0.6 mm Sn on tube B). The
bundled IPEM spectra stop at 140 kVp, so this notebook runs **100 / Sn140**, the pair the
Definition Flash ran clinically (with 0.4 mm Sn; the Force's 0.6 mm is kept). The mean energies
are within about 2 keV of the 100 / Sn150 target. The currents follow the published Force abdomen
100 / Sn150 reference, 190 and 95 mAs at 0.5 s, a 2 : 1 split.

| Tube | kVp | Filters | mA | Views | Rotation |
|------|-----|---------|----|-------|----------|
| A (low)  | 100 | 3 mm Al + 0.9 mm Ti | 380 | 1160 | 0.5 s |
| B (high) | 140 | 3 mm Al + 0.9 mm Ti + **0.6 mm Sn** | 190 | 1160 | 0.5 s |
"""

# ╔═╡ 09000005-0000-4000-8000-000000000010
protocol_low = BS.CTProtocol(
    kVp = 100,
    mA = 380.0,
    views = 1160,
    rotation_time = 0.5,
    collimation_mm = 4.8, # nominal beam width; axial cone guards are automatic
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 09000005-0000-4000-8000-000000000020
protocol_high = BS.CTProtocol(
    kVp = 140,
    mA = 190.0,
    views = 1160,
    rotation_time = 0.5,
    collimation_mm = 4.8,
    additional_filters = [("Ti", 0.9), ("Sn", 0.6)],
);

# ╔═╡ 09000006-0000-4000-8000-000000000030
"""
    ufc_detected_spectrum(protocol) -> (e, w_eta)

Display helper: tube spectrum × flat filtration × protocol filters (IPEM, absolute flux) × the
UFC's signal per photon, `E·efficiency(E)`: what the detector integrates (centre ray, no bowtie).
"""
function ufc_detected_spectrum(protocol)
    e, w = BS.resolve_source_spectrum_without_bowtie(
        force_opts,
        protocol;
        scanner = scanner,
    )
    t = BS.UFC_MC_EFFICIENCY_LUT # 1 keV grid: index = energy
    eff(E) = (
        i = clamp(floor(Int, E), 1, 139);
        f = E - i;
        (1 - f) * t.efficiency[i] + f * t.efficiency[i + 1]
    )
    return e, Float64.(w) .* e .* eff.(e)
end;

# ╔═╡ 09000006-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 960))

    # each tube's beam: its filters, then the UFC
    sketch = canvas(fig[1, 1], (0, 40, 0, 5.2))
    beams = (
        (3.9, "Tube A, 100 kVp", ["3 mm Al", "0.9 mm Ti"], COLORS.cyan),
        (1.3, "Tube B, 140 kVp", ["3 mm Al", "0.9 mm Ti", "0.6 mm Sn"], COLORS.violet),
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
                Mke.Rect2f(x - 0.25, y - 0.75, 0.5, 1.5);
                color = k == 3 ? COLORS.muted : COLORS.rule,
                strokecolor = COLORS.ink,
                strokewidth = 1,
            )
            Mke.text!(
                sketch,
                x + 0.5,
                y + 0.55;
                text = f,
                align = (:left, :bottom),
                fontsize = 15,
            )
        end
        Mke.poly!(sketch, Mke.Rect2f(35.0, y - 0.9, 0.7, 1.8); color)
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
        subtitle = "Photons × Energy × UFC Efficiency, Centre Ray, Each Normalized to Its Peak",
        xlabel = "Photon Energy (keV)",
        ylabel = "Relative Detected Signal",
    )
    beams = (
        (protocol_low, "Tube A, 100 kVp", COLORS.cyan),
        (protocol_high, "Tube B, Sn140 kVp", COLORS.violet),
    )
    for (protocol, label, color) in beams
        e, wη = ufc_detected_spectrum(protocol)
        mean_E = sum(e .* wη) / sum(wη)
        Mke.band!(ax, Float64.(e), 0.0, wη ./ maximum(wη); color = (color, 0.15))
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
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.3))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_detected_spectra.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 09000006-0000-4000-8000-000000000001
md"""
## 5. Physics Switches and Noise Draws

`force_opts` are the physics switches the basis-spectral-denoising study settled on for its
dual-source scanner, applied to the Force:

- **Detector response**: the scanner reads the UFC Geant4 table; every energy is weighted by its
  signal per photon ``m_1``, and the noise is compound Poisson with ``m_2 = m_1^2/\mathrm{dqe0}``, so
  its variance per unit signal rises as the beam hardens.
- **Heel effect off**, which keeps the forward spectral model equal to the response the spectral
  basis inverts (at 4.8 mm collimation the heel is a small row-direction effect).
- **View integration**, `view_samples = 5`: a detector integrates while the gantry turns, so each
  view reads the object averaged over the arc it sweeps, a blur largest far from the isocentre
  that the noise, counted once per view, does not share. Each Force tube reads its own detector
  for the whole view period (`view_arc = 1.0`).

**Each tube has its own seed**: the two tube-detector chains are physically independent, and two
acquisitions drawn from one seed would carry correlated noise that the decomposition amplifies.
Draw 0 is the noise-free expectation; draw ``r ≥ 1`` gives each tube the seed
``\mathrm{seed} + 1000\,(r - 1)``.
"""

# ╔═╡ 09000006-0000-4000-8000-000000000010
begin
    VIEW_SAMPLES = 5
    force_opts = BS.SimOptions(
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

# ╔═╡ 09000006-0000-4000-8000-000000000012
begin
    # The two exposures: tube A sees the phantom, tube B the phantom through its −0.88 mm z-offset.
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each tube the seed `seed + 1000 (r − 1)`,
    # so the tubes stay independent within a draw, and draws of each other.
    EXPOSURES = [
        (
            label = "100 kVp",
            protocol = protocol_low,
            seed = 1234,
            view_arc = 1.0,
            tube = :a,
        ),
        (
            label = "Sn140 kVp",
            protocol = protocol_high,
            seed = 4321,
            view_arc = 1.0,
            tube = :b,
        ),
    ]
    DE_DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    opts_for(e, r) = with_option(
        with_option(
            with_option(force_opts, :seed, realization_seed(e, r)),
            :use_noise,
            r > 0,
        ),
        :view_arc,
        e.view_arc,
    )
end;

# ╔═╡ 09000006-0000-4000-8000-000000000020
# The centred 5 × 0.6 mm grid. The axial workspace adds symmetric detector guard rows, so the
# peripheral voxels of both end slices keep measured cone-beam support.
recon_opts = BS.ReconOptions(matrix_size = (512, 512, 5), fov_cm = 35.0, z_cm = 0.3);

# ╔═╡ 09000007-0000-4000-8000-000000000001
md"""
## 6. The Acquisition: One Projection per Tube, Three Draws

`acquire` builds each tube's workspace and **projects it once**: everything before the noise is
the same for every draw, so the first draw keeps the projection (`keep_projection = true`) and
the others pass it back (`projection`) and draw only their noise, exactly as if simulated afresh.

| Draw | Noise | Used for |
|---:|---|---|
| 0 | off | the **noise-free reference** every noisy result is scored against |
| 1 | on | the **measured** acquisition every result comes from |
| 9 | on | the **calibration** draw, which only fixes the reconstruction's pair (section 8) |

The first draw also keeps each tube's one-channel spectral basis, `spectral_basis(ws, protocol)`:
every ray's detected spectrum (source × filtration × bowtie × the UFC's ``m_1``), the model the
projector applied, scaled to the ray's noise-equivalent quanta in air. Tube B sees the phantom
through the real −0.88 mm z-offset of detector B (Wang et al. 2021).
"""

# ╔═╡ 09000007-0000-4000-8000-000000000005
# Tube B's view of the phantom: the origin shifted by the real −0.88 mm detector z-offset (nothing
# changes for the z-invariant Gammex)
phantom_b = BS.Phantom(
    phantom.mask,
    phantom.materials,
    phantom.voxel_size,
    (phantom.origin[1], phantom.origin[2], phantom.origin[3] - 0.088),
    phantom.extent,
);

# ╔═╡ 09000007-0000-4000-8000-000000000020
"""
    acquire(draws) -> Dict{Int, NamedTuple}

The DE acquisition of the Gammex (`EXPOSURES`, both tubes) for every draw in `draws`: draw 0 the
noise-free expectation, draw `r ≥ 1` a noisy realisation with the seeds `realization_seed(e, r)`.
Each tube is projected once, by the first draw, and every other draw re-uses that projection
(`keep_projection`, `projection`), only its noise drawn. Each entry holds the two log-transmission
channels, the spectral basis (shared by every draw), the geometry and the wall time per tube.
"""
function acquire(draws)
    parts = map(EXPOSURES) do e
        ph = e.tube === :b ? phantom_b : phantom
        projection = nothing
        channels = Dict{Int, Array{Float32, 3}}()
        seconds = Dict{Int, Float64}()
        model = nothing
        for r in draws
            opts = opts_for(e, r)
            t = time()
            ws = BS.create_workspace(scanner, e.protocol, opts, recon_opts, ph)
            try
                result = BS.simulate!(
                    ws,
                    ph,
                    e.protocol,
                    opts;
                    report_dose = false,
                    keep_projection = projection === nothing,
                    projection,
                )
                projection === nothing && (projection = result.projection)
                channels[r] = Array(ws.sinogram)
                if model === nothing
                    model = (geom = ws.geom, basis = BS.spectral_basis(ws, e.protocol))
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

# ╔═╡ 09000007-0000-4000-8000-000000000030
de_acq = acquire((DE_DRAWS.reference, DE_DRAWS.measured, DE_DRAWS.calibration));

# ╔═╡ 09000007-0000-4000-8000-000000000035
let
    role = Dict(
        DE_DRAWS.reference => "noise-free reference",
        DE_DRAWS.measured => "measured",
        DE_DRAWS.calibration => "calibration",
    )
    seeds(r) = r == 0 ? "none" : join([realization_seed(e, r) for e in EXPOSURES], ", ")
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

    Draw 0 projects each tube, $(VIEW_SAMPLES) sub-views per view; draws 1 and 9 re-use that projection
    and only draw their noise (wall time on this render, compilation included in the first call).
    """,
    )
end

# ╔═╡ 09000007-0000-4000-8000-000000000036
# The measured draw: every result below comes from it.
de_measured = de_acq[DE_DRAWS.measured];

# ╔═╡ 09000007-0000-4000-8000-000000000040
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
    titles = ("Tube A: 100 kVp", "Tube B: Sn140 kVp")
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

# ╔═╡ 09000008-0000-4000-8000-000000000001
md"""
## 7. The Mixed Image

In dual-energy mode the routine-looking output is the **mixed image**, a linear blend of the two
tubes' reconstructions, ``M = w\,I_\mathrm{low} + (1 - w)\,I_\mathrm{high}`` (Yu et al., Med Phys
2009; Eusemann et al., SPIE 2008); on Sn150-class pairs the clinical weight is 0.5–0.6 (Lenga et
al., Br J Radiol 2021). Each tube runs notebook 01's correction stack on its own, a water
beam-hardening correction (BHC) of the sinogram, FDK, then HU, before the blend. If the UFC table
is folded in right, solid water lands at 0 HU in both tubes' images, and so in any blend.

!!! info "The BHC sees the detected spectrum"
    The water polynomial must see the spectrum the forward model used: the bowtie-hardened
    spectrum of every column weighted by the UFC's signal per photon, ``m_1(E) = E\,\eta(E)``
    (`resolve_source_spectrum_full`), fed to the low-level `calibrate_bhc_water(e, w_col)`, the
    per-column fit the high-level API performs, with the UFC included.
"""

# ╔═╡ 09000008-0000-4000-8000-000000000005
"""
    ufc_bhc_calibration(protocol, geom)

η-aware per-tube BHC: bowtie-hardened per-column spectrum × UFC η(E) →
per-column two-material polynomial.  Returns `(model, μ_water, ref_E_keV)`.
"""
function ufc_bhc_calibration(protocol, geom)
    # resolve_source_spectrum_full folds bowtie AND the src UFC η(E)
    # (via the same build_physics_config the forward model used).
    e, ŵ = BS.resolve_source_spectrum_full(
        force_opts,
        protocol;
        scanner = scanner,
        geom = geom,
    )
    e2, w_col = BS.bhc_spectrum_per_column(e, ŵ) # [n_E, n_col]

    # Single mono-equivalent target = mean energy of the η-folded mean spectrum
    w_mean = vec(sum(w_col; dims = 2)) ./ size(w_col, 2)
    ref_E = sum(e2 .* w_mean) / sum(w_mean)

    # Parameter-free water BHC from the custom UFC-η per-column spectrum — zero
    # segmentation thresholds (the two-material bone pass is deprecated:
    # its 450–600 HU window misclassified dense iodine as bone).
    model = BS.calibrate_bhc_water(e2, w_col; reference_energy_keV = ref_E)
    return (
        model = model,
        μ_water = model.μ_water_ref,
        ref_E_keV = model.reference_energy_keV,
    )
end;

# ╔═╡ 09000008-0000-4000-8000-000000000008
"""
    ufc_poly_recon(sino_cpu, geom, bhc) -> Array{Float32, 3}

Doctrine correction stack for one tube: knobless water sino-BHC → FDK → HU.
"""
function ufc_poly_recon(sino_cpu, geom, bhc)
    matrix_size = recon_opts.matrix_size

    sino_gpu = to_gpu(sino_cpu)
    # Knobless water BHC: one sinogram-domain pass, no recon round-trip.
    sino_bhc = BS.apply_bhc_water(sino_gpu, bhc.model)
    sino_gpu = sino_bhc

    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom)

    # (image-domain BHC + applied cupping removed — deprecated; cupping is a
    #  QA metric via measure_radial_cupping)
    hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water))

    ws_fdk = nothing
    sino_gpu = nothing
    recon_μ = nothing
    GC.gc(true)
    return hu
end;

# ╔═╡ 09000008-0000-4000-8000-000000000010
bhc_low = ufc_bhc_calibration(protocol_low, de_measured.geom);

# ╔═╡ 09000008-0000-4000-8000-000000000012
bhc_high = ufc_bhc_calibration(protocol_high, de_measured.geom);

# ╔═╡ 09000008-0000-4000-8000-000000000015
Markdown.parse(
    """
**The calibrations, with the UFC folded in**: tube A's reference energy is
$(round(bhc_low.ref_E_keV, digits = 1)) keV, where water attenuates
$(round(bhc_low.μ_water, digits = 5)) cm⁻¹; tube B's is $(round(bhc_high.ref_E_keV, digits = 1)) keV,
$(round(bhc_high.μ_water, digits = 5)) cm⁻¹.
""",
)

# ╔═╡ 09000008-0000-4000-8000-000000000020
hu_tube = (
    low = ufc_poly_recon(de_measured.channels[1], de_measured.geom, bhc_low),
    high = ufc_poly_recon(de_measured.channels[2], de_measured.geom, bhc_high),
);

# ╔═╡ 09000008-0000-4000-8000-000000000025
# Siemens' linear mixed image, M = w·I_low + (1 − w)·I_high (image domain, Yu 2009); w = 0.5 is the
# common Sn150-pair default (0.3–0.7 clinically)
MIX_W_LOW = 0.5f0;

# ╔═╡ 09000008-0000-4000-8000-000000000028
hu_mixed = MIX_W_LOW .* hu_tube.low .+ (1.0f0 - MIX_W_LOW) .* hu_tube.high;

# ╔═╡ 09000008-0000-4000-8000-000000000030
# Solid water, eroded 12 pixels from every edge, in the three images (every slice)
poly_water_stats = let
    ERODE_PX = 12.0
    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool = BS.erode_mask_2d(
        mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER);
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

# ╔═╡ 09000008-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 540))
    mid = size(hu_mixed, 3) ÷ 2 + 1
    w = round(MIX_W_LOW; digits = 2)
    operator(col, text) = Mke.Label(
        fig[1, col],
        text;
        fontsize = 54,
        font = :bold,
        color = COLORS.muted,
        tellheight = false,
    )
    slice_panel!(fig[1, 1], hu_tube.low[:, :, mid]; title = "$(w) × Tube A (100 kVp)")
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_poly_recon.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 09000008-0000-4000-8000-000000000060
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 640))

    sketch = canvas(fig[1, 1], (-18, 18, -18, 18))
    sketch.title = "Where: The Solid Water"
    gammex!(sketch; calcium = COLORS.rule, iodine = COLORS.rule)
    roi!(
        sketch,
        poly_water_stats.mask_2d,
        phantom_cpu.voxel_size[1] * size(phantom_cpu.mask, 1),
    )

    entries = (
        ("Tube A, 100 kVp", poly_water_stats.low, COLORS.cyan),
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_poly_water_values.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000a-0000-4000-8000-000000000001
md"""
## 8. The VMI Chain

**The spectral basis.** `acquire` merged the two tubes' bases with
`spectral_basis_from_acquisitions`: their energy grids onto one union grid, each tube's per-ray
detected spectrum scaled by its own noise-equivalent quanta in air (``I_0\,\mathrm{ff}\,S_1^2/S_2``).
The likelihood thus sees the absolute response ``\Phi_k(E)`` of every ray and channel (source ×
flat filters × bowtie × the UFC's ``m_1(E)``), the model the projector applied. No calibration
scan is involved, and all three draws share it.
"""

# ╔═╡ 0900000a-0000-4000-8000-000000000010
basis = de_measured.basis;

# ╔═╡ 0900000a-0000-4000-8000-000000000015
Markdown.parse(
    """
The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid for
$(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response sums to the air counts to within
$(round(basis.I0_relerr, sigdigits = 2)) (relative).
""",
)

# ╔═╡ 0900000b-0000-4000-8000-000000000001
md"""
**The chain.** One package call goes from the two corrected sinograms to the VMIs,
`vmi_pipeline(; denoiser = SpectralHYPR(), fbp_filter, pair_basis, composite_energy)`:

1. **Projection HYPR-LR** (`ProjectionHYPR`): a 3 × 3 window (columns × views of one parity,
   `view_stride = 2`, so the odd and even views stay independent) on each detector row's counts,
   each tube's dispersion measured on its own air rays.
2. **The K = 2 decomposition**: per ray, the Poisson maximum-likelihood iodine and water pair
   under the exact polychromatic mean of both tubes, every detector row kept.
3. **The spectral pair** (`spectral_pair`): FDK of the basis pair and of its odd- and even-view
   halves, whose difference measures the noise of everything downstream; the composite ``M`` (the
   minimum-noise VMI) and its complement ``I_\perp``, whose noise is uncorrelated with ``M``'s.
4. **ACNR** on the complement only; the composite stays as reconstructed.
5. **Image HYPR** (`ImageHYPR`): the complement pooled within its slice, weighted by the
   composite, over the window with the least estimated risk.
6. **VMI synthesis** at 50, 70, 100 and 140 keV from the one basis pair.

**The pair and the composite energy come from the calibration draw.** Near its minimum the VMI
noise hardly changes with energy, so an argmin measured on the draw being evaluated would itself
be noise. ``E^*`` and ``\beta`` come from the calibration draw's plain decomposition, the composite
energy from its decomposition after the projection HYPR; the measured draw, the same draw without
denoising, and the noise-free reference are then reconstructed alike.

!!! warning "One window, not a fitted pair: the Force has none"
    The other dual-energy notebooks reconstruct the composite and its complement with a
    `PairFilter` of two windows fitted, in basis-spectral-denoising, to the MTF and NPS of a
    physical Gammex scan on that scanner. No physical Force scan was fitted, so no Force pair
    exists, and the GE and Flash pairs do not transfer: a fitted pair absorbs everything that
    shapes one scanner's resolution and noise texture (its dexel pitch, 0.561 mm on the Force
    against 0.705 mm on the Flash, its focal spot, the vendor kernel of the series it was matched
    to). Both are reconstructed here with the soft-tissue `SoftFilter()`.
"""

# ╔═╡ 0900000b-0000-4000-8000-000000000005
# basis-spectral-denoising's HYPR_CHAIN: ProjectionHYPR(3 × 3 Box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 0900000b-0000-4000-8000-000000000006
VMI_CHAIN = (
    method = :nchannel,
    controls = BS.NChannelControls(),
    use_tlbf = false,
    antialias = true,
);

# ╔═╡ 0900000b-0000-4000-8000-000000000007
# One window for the composite and its complement alike: no fitted Force pair exists (see above).
FORCE_FILTER = BS.SoftFilter();

# ╔═╡ 0900000c-0000-4000-8000-000000000015
de_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 0900000b-0000-4000-8000-000000000008
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

# ╔═╡ 0900000b-0000-4000-8000-000000000010
de_vmi = BS.vmi_pipeline(;
    channels = de_measured.channels,
    basis,
    geom = de_measured.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(de_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = FORCE_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 0900000b-0000-4000-8000-000000000011
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
        fbp_filter = FORCE_FILTER,
        pair_basis = calibration.pair_basis,
        composite_energy = calibration.composite_energy,
        VMI_CHAIN...,
    )
    for r in (DE_DRAWS.measured, DE_DRAWS.reference)
);

# ╔═╡ 0900000b-0000-4000-8000-000000000015
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

    From the calibration draw: the pair at E* = $(round(pb.Estar, digits = 1)) keV,
    β = $(round(pb.β, sigdigits = 3)), and the composite ACNR and the image HYPR act on at
    $(round(calibration.composite_energy, digits = 1)) keV. The image HYPR pooled the complement over a
    $(d.image_estimates.window) × $(d.image_estimates.window) window.
    """,
    )
end

# ╔═╡ 0900000a-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1460))

    nodes = Dict(
        :a => (0.0, 0.75, "Tube A Sinogram", "100 kVp", COLORS.cyan),
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
    mid = size(de_vmi.images.iodine, 3) ÷ 2 + 1
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

# ╔═╡ 0900000c-0000-4000-8000-000000000001
md"""
## 9. The VMIs

`vmi_pipeline` synthesizes each VMI from the basis pair (McCollough 2015):
``\mu(E) = c_\mathrm{water}\,(\mu/\rho)_\mathrm{water}(E) + c_\mathrm{iodine}\,(\mu/\rho)_\mathrm{iodine}(E)``,
in HU against water at the same energy. The rows are the measured draw without denoising, the
measured draw through the whole chain, and the noise-free reference.
"""

# ╔═╡ 0900000c-0000-4000-8000-000000000010
# The basis pair in the eroded solid water: a perfect decomposition reads water ≈ 1 g/mL (solid
# water is not pure water, so a small offset is expected) and iodine ≈ 0
solid_water_basis = let
    ERODE_PX = 12.0

    mask_2d_raw = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    sw_bool_raw = (mask_2d_raw .== UInt8(BS.REGION_SOLID_WATER))
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

# ╔═╡ 5aef6f39-ba03-49f9-8f3b-86f766b6b76c
Markdown.parse(
    """
In the eroded solid water the basis pair reads $(round(solid_water_basis.c_water, digits = 4)) g/mL of
water and $(round(1000 * solid_water_basis.c_iodine, digits = 3)) mg/mL of iodine
($(solid_water_basis.n_voxels) voxels).
""",
)

# ╔═╡ 0900000c-0000-4000-8000-000000000020
# The VMI stack as one volume per energy (keV → (nx, ny, nz) HU).
vmi_HU_final = Dict(
    Float64(E) => de_vmi.vmis[:, :, :, k]
    for (k, E) in pairs(de_vmi.energies)
);

# ╔═╡ 0900000c-0000-4000-8000-000000000040
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
        e, wη = ufc_detected_spectrum(protocol)
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

    mid = size(de_vmi.vmis, 3) ÷ 2 + 1
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
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_grid.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000e-0000-4000-8000-000000000060
md"""
## 10. Water: Accuracy and Noise

Water is the reference of the HU scale, so the eroded solid water should read close to 0 HU in
every VMI (solid water is not pure water). The noise is measured in a central ROI **against the
noise-free reference**, as the standard deviation of (VMI − reference): the reference shares the
projection and the reconstruction, so the difference is noise alone, with no structure mixed in.
The same draw without denoising shows what the chain removes.
"""

# ╔═╡ 0900000e-0000-4000-8000-000000000050
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 640))

    sketch = canvas(fig[1, 1], (-18, 18, -18, 18))
    sketch.title = "Where: The Solid Water"
    gammex!(sketch; calcium = COLORS.rule, iodine = COLORS.rule)
    roi!(
        sketch,
        solid_water_basis.mask_2d,
        phantom_cpu.voxel_size[1] * size(phantom_cpu.mask, 1),
    )

    sw_idx = findall(solid_water_basis.mask_2d)
    mean_hu(vol) = mean(vol[c, z] for z in axes(vol, 3) for c in sw_idx)
    measured = [mean_hu(vmi_HU_final[E]) for E in de_vmi_energies]
    reference = [mean_hu(view(de_ref.vmis, :, :, :, k)) for k in eachindex(de_vmi_energies)]
    n = length(de_vmi_energies)
    ax = Mke.Axis(
        fig[1, 2];
        title = "Solid Water in Every VMI",
        subtitle = "Bars: SpectralHYPR. Diamonds: the Noise-Free Reference",
        xlabel = "VMI Energy (keV)",
        ylabel = "Mean HU",
        xticks = (1:n, ["$(Int(E))" for E in de_vmi_energies]),
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
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.38))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_water_roi.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0900000e-0000-4000-8000-000000000065
const WATER_NOISE_ROI_RADIUS_PX = 12; # ≈ 8.2 mm at 0.683 mm per pixel

# ╔═╡ 0900000e-0000-4000-8000-000000000070
# A disc of WATER_NOISE_ROI_RADIUS_PX pixels at the centre of the reconstruction grid
water_noise_roi = let
    nx_r, ny_r, nz_r = size(de_vmi.images.water)
    cx = nx_r ÷ 2 + 1
    cy = ny_r ÷ 2 + 1
    r² = Float64(WATER_NOISE_ROI_RADIUS_PX)^2
    roi_bool = [(i - cx)^2 + (j - cy)^2 ≤ r² for i in 1:nx_r, j in 1:ny_r]
    n_vox = count(roi_bool)
    (
        center_xy = (Float64(cx), Float64(cy)),
        mask_2d = roi_bool,
        n_voxels = n_vox,
        n_total = n_vox * nz_r,
    )
end;

# ╔═╡ 0900000e-0000-4000-8000-000000000080
# Mean HU and noise in the central ROI at every energy: σ of (VMI − noise-free reference), with and
# without the denoising
vmi_noise_by_keV = let
    roi_idx = findall(water_noise_roi.mask_2d)
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

# ╔═╡ 0900000e-0000-4000-8000-000000000090
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 640))

    sketch = canvas(fig[1, 1], (-18, 18, -18, 18))
    sketch.title = "Where: The Centre"
    gammex!(sketch; calcium = COLORS.rule, iodine = COLORS.rule)
    roi!(
        sketch,
        water_noise_roi.mask_2d,
        recon_opts.fov_cm;
        color = COLORS.cyan,
        alpha = 1.0,
    )
    Mke.text!(
        sketch,
        0,
        1.0;
        text = "Noise ROI",
        align = (:center, :bottom),
        fontsize = 15,
        color = COLORS.cyan,
    )

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σ_hypr = [vmi_noise_by_keV[E].std for E in Es]
    σ_none = [vmi_noise_by_keV[E].std_none for E in Es]
    ax = Mke.Axis(
        fig[1, 2];
        title = "Noise Against Energy",
        subtitle = "σ of (VMI − Noise-Free Reference), $(water_noise_roi.n_voxels) Pixels × $(size(de_vmi.vmis, 3)) Slices",
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
            ax,
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
                ax,
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
    Mke.xlims!(ax, first(Es) - 15, last(Es) + 15)
    Mke.ylims!(ax, 0, 1.25 * maximum(σ_none))
    Mke.axislegend(ax; position = :rt)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.38))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "force_ufc_vmi_water_noise.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ feb6291a-3736-440e-adb1-1feddf091125
let
    rows = [
        "| $(Int(E)) | $(round(v.mean, digits = 2)) | $(round(v.reference, digits = 2)) | $(round(v.std, digits = 2)) | $(round(v.std_none, digits = 2)) |"
        for (E, v) in sort(collect(vmi_noise_by_keV))
    ]
    Markdown.parse(
        """
    | VMI (keV) | Mean HU | Noise-free mean HU | σ, SpectralHYPR (HU) | σ, no denoising (HU) |
    |---:|---:|---:|---:|---:|
    $(join(rows, "\n"))

    The central ROI: $(water_noise_roi.n_voxels) pixels × $(size(de_vmi.vmis, 3)) slices =
    $(water_noise_roi.n_total) values per energy.
    """,
    )
end

# ╔═╡ 0900000e-0000-4000-8000-000000000001
md"""
## 11. The Rods Against Theory

Every rod's mean HU in an 8-pixel-radius ROI at its centroid, over every slice, against the
theoretical HU of its material, ``1000\,(\mu_\mathrm{rod}(E) - \mu_\mathrm{water}(E)) / \mu_\mathrm{water}(E)``
from `BS.compute_μ_at_energy`: pure physics, no fitting.
"""

# ╔═╡ 0900000e-0000-4000-8000-000000000010
ROD_LABELS = (
    Ca = (UInt8(10), UInt8(11), UInt8(12), UInt8(13), UInt8(14), UInt8(15), UInt8(16)),
    I = (UInt8(20), UInt8(21), UInt8(22), UInt8(23), UInt8(24), UInt8(25), UInt8(26)),
);

# ╔═╡ 0900000e-0000-4000-8000-000000000020
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

# ╔═╡ 0900000e-0000-4000-8000-000000000030
rod_data = let
    materials = phantom_cpu.materials
    mask_2d = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
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

# ╔═╡ 5a62c3c8-857d-40d5-a26a-204b079ecd5d
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

# ╔═╡ 0900000f-0000-4000-8000-000000000010
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
    for name in ("force_ufc_vmi_vs_theoretical.png", "force_ufc_vmi_regression.png")
        Mke.save(joinpath(@__DIR__, "..", "assets", name), fig; px_per_unit = 2)
    end
    fig
end

# ╔═╡ 0900000f-0000-4000-8000-000000000030
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

# ╔═╡ 09000010-0000-4000-8000-000000000001
let
    fmt(v) = join(round.(v, digits = 1), " / ")
    Es = de_vmi_energies
    sw = solid_water_basis
    mean_sw(vol) = mean(vol[c, z] for z in axes(vol, 3) for c in findall(sw.mask_2d))
    water = [mean_sw(vmi_HU_final[E]) for E in Es]
    σ = [vmi_noise_by_keV[E].std for E in Es]
    σ_none = [vmi_noise_by_keV[E].std_none for E in Es]
    slopes(group) = fmt([rod_fits[(group, E)].slope for E in Es])
    rmses(group) = fmt([rod_fits[(group, E)].rmse for E in Es])
    p = poly_water_stats
    Markdown.parse(
        """
    ## Summary

    One SOMATOM Force dual-source acquisition, 100 kVp on tube A and Sn140 kVp on tube B with the
    UFC Geant4 table, read two ways:

    | Readout | Result |
    |---|---|
    | Solid water, tube A / tube B / mixed image | $(fmt([p.low.mean, p.high.mean, p.mixed.mean])) HU (σ $(fmt([p.low.std, p.high.std, p.mixed.std])) HU) |
    | Solid water in the VMIs, 50 / 70 / 100 / 140 keV | $(fmt(water)) HU |
    | Central noise, SpectralHYPR | $(fmt(σ)) HU |
    | Central noise, no denoising | $(fmt(σ_none)) HU |
    | Calcium rods, slope / RMSE against theory | $(slopes(:Ca)) / $(rmses(:Ca)) HU |
    | Iodine rods, slope / RMSE against theory | $(slopes(:I)) / $(rmses(:I)) HU |

    Solid water lands near 0 HU in both tubes' images, each under its own BHC with the UFC folded
    in: the table's spectral shape agrees with the detected sinograms on two very different
    spectra. The VMIs carry it further, through per-ray spectral inversion across two detected
    spectra on opposite sides of the Gd K-edge, each result scored against the noise-free draw of
    the same projection.
    """,
    )
end

# ╔═╡ Cell order:
# ╟─09000001-0000-4000-8000-000000000010
# ╟─ef027fa5-f75b-435c-a792-6ef688dcc445
# ╟─b088bb14-1243-4c2a-a097-3d2264ab01f4
# ╟─98cfe0df-2f67-4e5b-a60d-9d66f85474e8
# ╟─09000001-0000-4000-8000-000000000020
# ╠═09000001-0000-4000-8000-000000000001
# ╠═09000001-0000-4000-8000-000000000002
# ╠═09000001-0000-4000-8000-000000000003
# ╠═09000001-0000-4000-8000-000000000030
# ╠═09000001-0000-4000-8000-000000000031
# ╠═09000001-0000-4000-8000-000000000032
# ╠═09000001-0000-4000-8000-000000000033
# ╠═09000001-0000-4000-8000-000000000040
# ╟─09000001-0000-4000-8000-000000000050
# ╟─bd293b7a-2073-468e-845e-51cf5f0e5f5d
# ╠═002ed623-0e86-454f-ae59-1f1a73913c6d
# ╠═1c5b3e94-e46f-48fd-8454-4e3bfaa3daa6
# ╠═5b42f9a3-9156-4db8-8534-261a7a3d822a
# ╠═4179f27b-105e-4908-9da1-f92aa0c792b4
# ╠═4e0524a6-23fa-4dd3-9f9e-f8613e5fd1ad
# ╠═6833d0e8-0227-4237-8976-7ca008cf3850
# ╠═ed4a09d8-8c8b-4c05-a63a-647ad4f4b0cd
# ╠═0b4226a2-1912-4dca-8417-932a0c3f8d90
# ╠═d0ae6214-610c-4431-9538-53c963eb16e8
# ╟─09000002-0000-4000-8000-000000000001
# ╟─09000002-0000-4000-8000-000000000030
# ╟─09000003-0000-4000-8000-000000000001
# ╠═09000003-0000-4000-8000-000000000010
# ╠═09000003-0000-4000-8000-000000000020
# ╟─09000004-0000-4000-8000-000000000001
# ╟─378aa523-1fb9-4e69-a0d4-22d1ea910d1e
# ╠═09000004-0000-4000-8000-000000000010
# ╟─a25dda2c-bb62-534b-be16-0f1de34a44f1
# ╟─09000005-0000-4000-8000-000000000001
# ╠═09000005-0000-4000-8000-000000000010
# ╠═09000005-0000-4000-8000-000000000020
# ╠═09000006-0000-4000-8000-000000000030
# ╟─09000006-0000-4000-8000-000000000040
# ╟─09000006-0000-4000-8000-000000000001
# ╠═09000006-0000-4000-8000-000000000010
# ╠═09000006-0000-4000-8000-000000000012
# ╠═09000006-0000-4000-8000-000000000020
# ╟─09000007-0000-4000-8000-000000000001
# ╠═09000007-0000-4000-8000-000000000005
# ╠═09000007-0000-4000-8000-000000000020
# ╠═09000007-0000-4000-8000-000000000030
# ╟─09000007-0000-4000-8000-000000000035
# ╠═09000007-0000-4000-8000-000000000036
# ╟─09000007-0000-4000-8000-000000000040
# ╟─09000008-0000-4000-8000-000000000001
# ╠═09000008-0000-4000-8000-000000000005
# ╠═09000008-0000-4000-8000-000000000008
# ╠═09000008-0000-4000-8000-000000000010
# ╠═09000008-0000-4000-8000-000000000012
# ╟─09000008-0000-4000-8000-000000000015
# ╠═09000008-0000-4000-8000-000000000020
# ╠═09000008-0000-4000-8000-000000000025
# ╠═09000008-0000-4000-8000-000000000028
# ╠═09000008-0000-4000-8000-000000000030
# ╟─09000008-0000-4000-8000-000000000040
# ╟─09000008-0000-4000-8000-000000000060
# ╟─0900000a-0000-4000-8000-000000000001
# ╠═0900000a-0000-4000-8000-000000000010
# ╟─0900000a-0000-4000-8000-000000000015
# ╟─0900000b-0000-4000-8000-000000000001
# ╠═0900000b-0000-4000-8000-000000000005
# ╠═0900000b-0000-4000-8000-000000000006
# ╠═0900000b-0000-4000-8000-000000000007
# ╠═0900000c-0000-4000-8000-000000000015
# ╠═0900000b-0000-4000-8000-000000000008
# ╠═0900000b-0000-4000-8000-000000000010
# ╠═0900000b-0000-4000-8000-000000000011
# ╟─0900000b-0000-4000-8000-000000000015
# ╟─0900000a-0000-4000-8000-000000000040
# ╟─0900000c-0000-4000-8000-000000000001
# ╠═0900000c-0000-4000-8000-000000000010
# ╟─5aef6f39-ba03-49f9-8f3b-86f766b6b76c
# ╠═0900000c-0000-4000-8000-000000000020
# ╟─0900000c-0000-4000-8000-000000000040
# ╟─0900000e-0000-4000-8000-000000000060
# ╟─0900000e-0000-4000-8000-000000000050
# ╠═0900000e-0000-4000-8000-000000000065
# ╠═0900000e-0000-4000-8000-000000000070
# ╠═0900000e-0000-4000-8000-000000000080
# ╟─0900000e-0000-4000-8000-000000000090
# ╟─feb6291a-3736-440e-adb1-1feddf091125
# ╟─0900000e-0000-4000-8000-000000000001
# ╠═0900000e-0000-4000-8000-000000000010
# ╠═0900000e-0000-4000-8000-000000000020
# ╠═0900000e-0000-4000-8000-000000000030
# ╠═5a62c3c8-857d-40d5-a26a-204b079ecd5d
# ╟─0900000f-0000-4000-8000-000000000010
# ╟─0900000f-0000-4000-8000-000000000030
# ╟─09000010-0000-4000-8000-000000000001
