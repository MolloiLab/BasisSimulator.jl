### A Pluto.jl notebook ###
# v0.2.1

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

# ╔═╡ 11000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 11000001-0000-4000-8000-000000000004
using Statistics: mean, std

# ╔═╡ 11000001-0000-4000-8000-000000000005
using PlutoUI

# ╔═╡ 11000001-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 11000001-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 11000001-0000-4000-8000-000000000006
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 11000002-0000-4000-8000-000000000001
md"""
# Helical Scanning: Narrow Collimation, Long Coverage

**One keyword, `pitch`, turns any protocol into a spiral scan: here a 20 mm helix and three 160 mm
volume-axial stations cover the same 30 cm of a phantom that changes along z.**

Before spiral CT, long coverage meant step-and-shoot: scan a slab, move the table, scan the next.
Each station is a clean axial scan, but every station has cone-beam edges, and the stitched volume
can step at a station boundary. A helical scan instead sweeps a narrow beam past every slice
continuously: each z position is, at some moment of the spiral, at the centre of the beam.
"""

# ╔═╡ 27540bb6-e69c-4c78-8d86-10efe26727d9
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "Helical: 20 mm, 16 Turns", "Volume Axial: 3 Stations"],
        [
            "A Lung Rod Winds Round, a Fat Cone Narrows",
            "The Table Glides at Pitch $(protocol_helical.pitch)",
            "160 mm Cones, 10 cm Slabs, Stitched",
        ];
        limits = [(-19, 19, -13.5, 13.5), (-17.5, 17.5, -11, 11), (-17.5, 17.5, -11, 11)],
    )
    phantom_side!(axes[1]; dark = true)
    helix!(
        axes[2];
        turns = protocol_helical.n_rotations,
        travel = protocol_helical.pitch *
            protocol_helical.collimation_mm *
            protocol_helical.n_rotations /
            10,
        beam = protocol_helical.collimation_mm / 10,
        dark = true,
    )
    stations!(axes[3]; dark = true)
    fig
end

# ╔═╡ 429c6786-c046-4fc3-9828-b1215c61267d
md"""
## The Two Scans at a Glance

Both cover a **30 cm** reconstructed z-slab on a wide-cone 256 × 0.625 mm (16 cm) volume scanner,
with the tube currents chosen so that the two deliver the same total beam width × current × time:

| Scan | Collimation | Rotations | Table |
|:---|:---|:---|:---|
| volume axial | 160 mm (the full detector) | 3 × axial | steps 10 cm between stations |
| **helical** | **20 mm (one eighth)** | 16 turns | glides at pitch 1.0 |

The forward projection is the anti-aliased `:dd_fast` projector, and helical costs it nothing: the
projectors consume per-view source and detector arrays, and a helix is a z-ramp in those arrays.
Both scans integrate each view over the arc the gantry turns while the detector reads it
(`view_samples = 5`); on the helix each sub-view is also advanced along z by its share of the table
feed. Each `simulate!` also reports its dose, so the two can be compared in CTDIvol and DLP as well
as in image quality.
"""

# ╔═╡ 11000002-0000-4000-8000-000000000002
md"""
## Notebook Setup

The shared docs environment, the device (notebook 01), and a table of contents. Nothing is
simulated until section 4.
"""

# ╔═╡ 11000002-0000-4000-8000-000000000003
PlutoUI.TableOfContents()

# ╔═╡ 1d85cd7b-bc6b-4b40-9e0b-2bfcec03e7ac
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ 04505917-64f0-4832-91c8-67aeb83a9c16
md"""
## Figure Style

The colours, the theme and the drawing helpers every BasisSimulator notebook shares, then this
notebook's own drawings: the phantom from the side, the helical source path and the axial stations.
"""

# ╔═╡ e92386a8-e385-45da-87a5-a8fc2e43ab52
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

# ╔═╡ 6fa82e51-0f08-4da2-b395-f56f03d5b131
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

# ╔═╡ a463b2db-f3bb-4428-b5e4-befcadff26c2
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

# ╔═╡ 9258a3c9-6d6e-45d9-abe4-761db7b02828
"""
The phantom seen from the side on `ax`, z (cm) across and x (cm) up: the 24 cm water body, the
lung rod winding round it once per 12 cm (solid where it is in front, faint behind) and the
adipose cone narrowing from 4 cm to nothing. `z0` is the phantom's first z (cm).
"""
function phantom_side!(ax; z0 = -18.0, length_cm = 36.0, dark = false)
    ink = dark ? COLORS.night_ink : COLORS.ink
    z1 = z0 + length_cm
    Mke.poly!(
        ax,
        Mke.Rect2f(z0, -12, length_cm, 24);
        color = (COLORS.blue, dark ? 0.55 : 0.35),
        strokecolor = ink,
        strokewidth = 1.5,
    )
    Mke.poly!(ax, Mke.Point2f[(z0, -4), (z0, 4), (z1, 0)]; color = (COLORS.green, 0.85))
    zs = range(z0, z1; length = 600)
    θ(z) = 2π * (z - z0) / 12
    for front in (false, true)
        points = [
            (sin(θ(z)) >= 0) == front ? Mke.Point2f(z, 7cos(θ(z))) : Mke.Point2f(NaN, NaN)
            for z in zs
        ]
        Mke.lines!(ax, points; color = (COLORS.beam, front ? 1.0 : 0.35), linewidth = 9)
    end
    return ax
end

# ╔═╡ e77961b0-08a4-4207-86fd-eb47141416f2
"""
The helical source path on `ax`, z (cm) across: `turns` turns of radius `radius` round the
patient over `travel` cm of table, solid in front and faint behind, with the narrow beam of
width `beam` at the middle of the scan.
"""
function helix!(ax; turns = 16, travel = 32.0, radius = 7.0, beam = 2.0, dark = false)
    ink = dark ? COLORS.night_ink : COLORS.ink
    Mke.poly!(
        ax,
        Mke.Rect2f(-travel / 2, -radius + 2.2, travel, 2radius - 4.4);
        color = (COLORS.blue, dark ? 0.45 : 0.3),
    )
    t = range(0, 1; length = round(Int, 120 * turns))
    z = @. -travel / 2 + travel * t
    φ = @. 2π * turns * t
    for front in (false, true)
        points = [
            (sin(p) >= 0) == front ?
                Mke.Point2f(zz, radius * cos(p)) :
                Mke.Point2f(NaN, NaN)
            for (zz, p) in zip(z, φ)
        ]
        Mke.lines!(
            ax,
            points;
            color = (COLORS.beam, front ? 1.0 : 0.3),
            linewidth = front ? 2.5 : 1.5,
        )
    end
    # the beam at one moment: a source on top, a fan as wide as the collimation below it
    Mke.poly!(
        ax,
        Mke.Point2f[(0, radius), (-beam / 2, -radius), (beam / 2, -radius)];
        color = (COLORS.beam, 0.45),
    )
    Mke.scatter!(ax, [Mke.Point2f(0, radius)]; marker = :rect, markersize = 18, color = ink)
    Mke.lines!(
        ax,
        [-beam / 2, beam / 2],
        [-radius, -radius];
        color = COLORS.cyan,
        linewidth = 8,
    )
    arrow!(
        ax,
        (-travel / 2, -radius - 1.6),
        (travel / 2, -radius - 1.6);
        color = ink,
        head = 13,
    )
    return ax
end

# ╔═╡ 6c62678d-b885-415f-a6e1-e8ab537c47ae
"""
Step-and-shoot volume axial scanning on `ax`, z (cm) across: at each station of `stations`, a
source above the patient and a cone as wide as the `detector` (cm), and the `slab` (cm) each
station reconstructs.
"""
function stations!(
    ax;
    stations = (-10.0, 0.0, 10.0),
    detector = 16.0,
    slab = 10.0,
    radius = 7.0,
    dark = false,
)
    ink = dark ? COLORS.night_ink : COLORS.ink
    muted = dark ? COLORS.night_muted : COLORS.muted
    Mke.poly!(
        ax,
        Mke.Rect2f(-16, -radius + 2.2, 32, 2radius - 4.4);
        color = (COLORS.blue, dark ? 0.45 : 0.3),
    )
    tints = (COLORS.violet, COLORS.cyan, COLORS.violet)
    for (k, z) in enumerate(stations)
        Mke.poly!(
            ax,
            Mke.Point2f[
                (z, radius),
                (z - detector / 2, -radius),
                (z + detector / 2, -radius),
            ];
            color = (COLORS.beam, 0.16),
            strokecolor = (COLORS.beam, 0.7),
            strokewidth = 1,
        )
        Mke.poly!(
            ax,
            Mke.Rect2f(z - slab / 2, -radius + 2.2, slab, 2radius - 4.4);
            color = (tints[k], 0.3),
            strokecolor = tints[k],
            strokewidth = 2,
        )
        Mke.scatter!(
            ax,
            [Mke.Point2f(z, radius)];
            marker = :rect,
            markersize = 18,
            color = ink,
        )
        Mke.text!(
            ax,
            z,
            radius + 1.0;
            text = "Station $(k)",
            align = (:center, :bottom),
            fontsize = 15,
            font = :bold,
            color = muted,
        )
    end
    return ax
end

# ╔═╡ 11000003-0000-4000-8000-000000000001
md"""
## 1. A Phantom That Changes Along z

Two design rules. A uniform cylinder would look the same at every slice, useless for judging
long-z fidelity, so every slice must differ. And **no high-Z inserts**: beam hardening is its own
topic (notebook 10), and here it would only confound the geometry comparison.

- a **24 cm water body**, 36 cm long (coarse 2.3 mm in-plane and 2 mm z voxels);
- a **lung-density rod** (about −700 HU, radiologically soft) that **winds helically** round the
  body axis, one turn per 12 cm of z, so its angle tags every slice;
- an **adipose cone** (about −90 HU) on the axis, its radius tapering from 4 cm to 0 across the
  slab, so its diameter tags every slice.
"""

# ╔═╡ 11000003-0000-4000-8000-000000000002
phantom_data = let
    n = 128     # in-plane grid
    nz = 180    # 2 mm z-voxels → 36 cm slab
    fov = 30.0  # cm, in-plane phantom extent
    zext = 36.0 # cm, phantom z extent
    vox = fov / n
    voxz = zext / nz
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    r_body = 12.0 / vox # 24 cm water body
    for k in 1:nz, j in 1:n, i in 1:n
        if (i - c)^2 + (j - c)^2 <= r_body^2
            mask[i, j, k] = 1
        end
    end
    # helically winding lung-density rod: r = 7 cm, one turn per 12 cm of z
    for k in 1:nz
        θ = 2π * (k * voxz) / 12.0
        cx = c + (7.0 / vox) * cos(θ)
        cy = c + (7.0 / vox) * sin(θ)
        r = 1.0 / vox # 2 cm diameter rod
        for j in 1:n, i in 1:n
            if (i - cx)^2 + (j - cy)^2 <= r^2 && mask[i, j, k] == 1
                mask[i, j, k] = 2
            end
        end
    end
    # central adipose cone: radius tapers 4 cm → 0 across the slab
    for k in 1:nz
        r = (4.0 * (1 - (k - 1) / (nz - 1))) / vox
        for j in 1:n, i in 1:n
            if (i - c)^2 + (j - c)^2 <= r^2 && mask[i, j, k] == 1
                mask[i, j, k] = 3
            end
        end
    end
    (mask = mask, vox = vox, voxz = voxz, n = n, nz = nz, zext = zext)
end;

# ╔═╡ 11000003-0000-4000-8000-000000000003
phantom_materials = Dict(
    1 => BS.XA.Materials.water,
    2 => BS.XA.Materials.lung,
    3 => BS.XA.Materials.adipose,
);

# ╔═╡ 11000003-0000-4000-8000-000000000004
phantom = BS.Phantom(
    to_gpu(phantom_data.mask),
    phantom_materials,
    (phantom_data.vox, phantom_data.vox, phantom_data.voxz),
);

# ╔═╡ 11000004-0000-4000-8000-000000000001
md"""
## 2. The Helical Scan: `pitch` Is the Whole API

IEC pitch is the table feed per rotation over the active collimation. With **20 mm** collimation
(32 of the scanner's 256 × 0.625 mm rows), pitch 1.0 and 16 rotations, the table travels

```math
\text{travel} = \text{pitch} \times \text{collimation} \times n_\text{rot}
             = 1.0 \times 20\,\text{mm} \times 16 = 32\,\text{cm}.
```

Everything else (`Phantom`, the scanner, `SimOptions`, `ReconOptions`, `simulate!`, the
corrections, the reconstruction call) is the same as in an axial workflow.

The scanner is the GE Revolution Apex (`REVOLUTION_APEX`), whose 256 × 0.625 mm rows are the volume
detector this comparison needs, with the notebook's 512 one-millimetre columns in place of its 834
so the scans cost what they did. Both scans include the patient's scatter (on by default), from the
Geant4 kernels the Apex names, and the DAS's scatter correction; the kernels are made for one
scanner's distances, so the scans run at the Apex's 625.6 and 1100 mm.
"""

# ╔═╡ 11000004-0000-4000-8000-000000000002
scanner = BS.EICTScanner(BS.REVOLUTION_APEX; # 256 × 0.625 mm = 16 cm wide-cone volume scanner
    # the notebook's 512 one-millimetre columns, not the Apex's 834
    detector_cols = 512,
    detector_col_size = 1.0,
);

# ╔═╡ 31a9e8a0-c16a-511e-a0a0-c0610c198163
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

# ╔═╡ 11000004-0000-4000-8000-000000000003
begin
    protocol_helical = BS.CTProtocol(
        kVp = 120.0,
        mA = 200.0,
        views = 360,
        rotation_time = 0.5,
        collimation_mm = 20.0, # NARROW: one eighth of the physical detector
        pitch = 1.0,           # ← the one helical kwarg
        n_rotations = 16,      # 16 × 2 cm = 32 cm table travel
    )
    protocol_axial = BS.CTProtocol(
        # Three 160 mm stations match 16 × 20 mm helical rotations when
        # weighted by tube current: 3 × 160 × 133⅓ = 16 × 20 × 200.
        kVp = 120.0,
        mA = 400.0 / 3.0,
        views = 360,
        rotation_time = 0.5,
        collimation_mm = 160.0, # WIDE: the full 16 cm detector, volume mode
    )
    # use_heel_effect = false: the anode heel is a per-ROW spectral gradient.
    # An axial slice keeps its rows, but a helical scan sweeps every row past
    # every voxel, which a real scanner handles with a per-row water
    # calibration. `calibrate_bhc_water` is per column, so the heel is switched
    # off here and the comparison isolates the geometry.
    # view_samples = 5: each view integrates over the arc the gantry turns while the detector
    # reads it (notebook 01); on a helix every sub-view also advances by its share of the feed.
    sim_opts = BS.SimOptions(
        seed = 42,
        projector = :dd_fast,
        view_samples = 5,
        use_heel_effect = false,
    )
    recon_opts = BS.ReconOptions(matrix_size = (160, 160, 150), fov_cm = 30.0, z_cm = 30.0)
    # A 16 cm *physical* detector cannot reconstruct a full 16 cm axial
    # cylinder at 30 cm transverse FOV: cone magnification requires 338 rows.
    # Ten centimetres requires 212 guarded rows and fits the 256-row scanner.
    recon_opts_station = BS.ReconOptions(
        matrix_size = (160, 160, 50),
        fov_cm = 30.0,
        z_cm = 10.0,
    )
end;

# ╔═╡ 11000005-0000-4000-8000-000000000001
md"""
## 3. The Reconstruction Chain

One water beam-hardening correction (BHC) per protocol, calibrated on that acquisition's geometry,
then per reconstruction: sinogram-domain water BHC, reconstruction, and HU with the BHC's own
``\mu_\text{water}``. `reconstruct!` recognises a helical geometry and runs **rebinned WFBP**
(Stierstorfer *et al.* 2004, the production spiral algorithm family); an axial one runs FDK.
Quantum and electronic noise are already in the simulated counts.
"""

# ╔═╡ baf87a28-d676-44df-a58c-046ea4fbb7d8
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 320))
    nodes = Dict(
        :simulate => (0.0, 1.0, "simulate!", "Counts, Scatter, Dose", COLORS.cyan),
        :bhc => (4.4, 1.0, "Water BHC", "One per Protocol", COLORS.cyan),
        :reconstruct => (8.8, 1.0, "reconstruct!", "Reads the Geometry", COLORS.violet),
        :wfbp => (13.2, 1.8, "Rebinned WFBP", "Helical Geometry", COLORS.beam),
        :fdk => (13.2, 0.2, "FDK", "Axial Geometry", COLORS.blue),
        :hu => (17.6, 1.0, "HU Volume", "to_hounsfield", COLORS.green),
    )
    edges = [
        :simulate => :bhc,
        :bhc => :reconstruct,
        :reconstruct => :wfbp,
        :reconstruct => :fdk,
        :wfbp => :hu,
        :fdk => :hu,
    ]
    flow_chart!(
        fig[1, 1],
        nodes,
        edges;
        limits = (-2.0, 19.6, -0.55, 2.55),
        w = 3.6,
        h = 1.1,
    )
    fig
end

# ╔═╡ 11000005-0000-4000-8000-000000000002
"""
    corrected_recon(sino_gpu, geom, matrix_size, bhc) -> Array{Float32,3} (HU)

Water BHC → reconstruction → HU for one sinogram. A helical `geom` dispatches to WFBP inside
`reconstruct!`.
"""
function corrected_recon(sino_gpu, geom, matrix_size, bhc)
    sino = BS.apply_bhc_water(sino_gpu, bhc)
    ws_fdk = BS.create_fdk_recon_workspace(sino, geom, matrix_size)
    recon_μ = BS.reconstruct!(ws_fdk, sino, geom)
    return Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc.μ_water_ref))
end;

# ╔═╡ 11000006-0000-4000-8000-000000000001
md"""
## 4. Run Both Acquisitions

**Helical**: one `simulate!`, one reconstruction. **Volume axial (step-and-shoot)**: three 10 cm
reconstruction stations. For each, the table (here, the phantom window) moves so that the station
is centred on the isocentre; the three slabs are then stitched at ``z = ±5`` cm. The 16 cm beams of
neighbouring stations overlap by 6 cm, so each station's dose is reported with a 10 cm table
increment.
"""

# ╔═╡ 11000006-0000-4000-8000-000000000002
helical_result = let
    ws = BS.create_workspace(scanner, protocol_helical, sim_opts, recon_opts, phantom)
    t = @elapsed (sim = BS.simulate!(ws, phantom, protocol_helical, sim_opts))
    bhc = BS.calibrate_bhc_water(sim_opts, protocol_helical; scanner, geom = ws.geom)
    t += @elapsed (hu = corrected_recon(ws.sinogram, ws.geom, recon_opts.matrix_size, bhc))
    n_views = ws.geom.n_angles
    ws = nothing
    GC.gc()
    (hu = hu, t = t, dose = sim.dose, n_views = n_views)
end;

# ╔═╡ 11000006-0000-4000-8000-000000000003
sns_result = let
    station_zs = [-10.0, 0.0, 10.0] # 3 stations × 10 cm, back to back
    n_slab = 50                     # 10 cm at 2 mm recon slices
    hu = zeros(Float32, 160, 160, 150)
    t_total = 0.0
    bhc_ax = nothing
    doses = BS.DoseReport[]
    for (s, z0) in enumerate(station_zs)
        # move the "table": phantom window (±10 cm) centred on this station
        k0 = round(Int, 90 + z0 / phantom_data.voxz)
        requested = (k0 - 49):(k0 + 50) # 100 slices = 20 cm (covers the cone)
        available = max(first(requested), 1):min(last(requested), phantom_data.nz)
        # End stations extend beyond the finite source phantom. Keep their
        # isocentres fixed and pad the missing exterior with material 0 (air),
        # rather than clamping the window and shifting anatomy toward isocentre.
        station_mask = zeros(
            eltype(phantom_data.mask),
            size(phantom_data.mask, 1),
            size(phantom_data.mask, 2),
            length(requested),
        )
        destination = (
            (first(available) - first(requested) + 1):(
                    last(available) - first(requested) + 1
                )
        )
        station_mask[:, :, destination] .= phantom_data.mask[:, :, available]
        ph_st = BS.Phantom(
            to_gpu(station_mask),
            phantom_materials,
            (phantom_data.vox, phantom_data.vox, phantom_data.voxz),
        )
        ws = BS.create_workspace(
            scanner,
            protocol_axial,
            sim_opts,
            recon_opts_station,
            ph_st,
        )
        t_total += @elapsed (
            sim = BS.simulate!(
                ws,
                ph_st,
                protocol_axial,
                sim_opts;
                dose_kwargs = (; table_increment_mm = 100.0),
            )
        )
        push!(doses, sim.dose)
        if bhc_ax === nothing
            bhc_ax = BS.calibrate_bhc_water(
                sim_opts,
                protocol_axial;
                scanner,
                geom = ws.geom,
            )
        end
        t_total += @elapsed (
            hu_st = corrected_recon(
                ws.sinogram,
                ws.geom,
                recon_opts_station.matrix_size,
                bhc_ax,
            )
        )
        ws = nothing
        GC.gc()
        # station slab → global z index: each station covers z0 ± 5 cm.
        k_lo = round(Int, (z0 - 5.0 + 15.0) / 0.2) + 1
        valid = max(k_lo, 1):min(k_lo + n_slab - 1, 150)
        hu[:, :, valid] .= hu_st[:, :, (first(valid) - k_lo + 1):(last(valid) - k_lo + 1)]
    end
    (hu = hu, t = t_total, doses = doses)
end;

# ╔═╡ 11000006-0000-4000-8000-000000000004
let
    h = helical_result.dose
    a = sns_result.doses
    dlp_ax = sum(d.dlp_mGy_cm for d in a)
    Markdown.parse(
        """
        **Dose of each acquisition** (CTDI from a Monte Carlo of the simulated beam in the 32 cm body phantom)

        | Acquisition | N·T (mm) | mAs / rotation | Rotations | CTDIvol (mGy) | DLP (mGy·cm) |
        |:--|--:|--:|--:|--:|--:|
        | helical, pitch $(h.pitch) | $(h.nominal_collimation_mm) | $(round(h.mAs_per_rotation; digits = 1)) | $(Int(h.n_rotations)) | $(round(h.ctdi_vol_mGy; digits = 2)) | $(round(h.dlp_mGy_cm; digits = 1)) |
        | volume axial, 3 stations, 10 cm apart | $(a[1].nominal_collimation_mm) | $(round(a[1].mAs_per_rotation; digits = 1)) | 3 × 1 | $(round(a[1].ctdi_vol_mGy; digits = 2)) | $(round(dlp_ax; digits = 1)) |

        The two DLPs are $(round(h.dlp_mGy_cm; digits = 1)) and $(round(dlp_ax; digits = 1)) mGy·cm:
        matching beam width × current × time matches the total exposure. CTDIvol differs because that
        exposure is spread over different lengths: $(round(h.scan_length_cm; digits = 1)) cm of helical
        travel against three $(round(a[1].table_increment_mm / 10; digits = 1)) cm table increments for the
        axial stations, whose overlapping $(a[1].nominal_collimation_mm) mm beams$(a[1].wide_beam_reference_mm === nothing ? "" : " are dosed with the IEC wide-beam rule (reference $(a[1].wide_beam_reference_mm) mm)").
        """,
    )
end

# ╔═╡ 1100000b-0000-4000-8000-000000000001
md"""
## 5. Plan a Helical Scan

The dose above is the pitch-1.0 scan. At a fixed tube current and rotation time, CTDIvol scales as
1 / pitch, and the table feed per rotation is pitch × collimation, which sets how many rotations
(and seconds) a range takes. Move the slider: the numbers below are computed live in your browser
from this scan's measured CTDIvol and its 20 mm collimation.
"""

# ╔═╡ 1100000b-0000-4000-8000-000000000002
# The calculator's inputs — this scan's CTDIvol at pitch 1 (mGy), collimation (mm) and rotation time
# (s) — written as constants: the browser runs the calculator compiled to WebAssembly, which bakes
# constants in but would have to compile any computation behind them (here, the whole simulation).
# The cell below checks them against the scan, so they cannot go stale.
const HELICAL_INPUTS = (10.31, 20.0, 0.5);

# ╔═╡ 1100000b-0000-4000-8000-000000000006
let measured = (
    Float64(helical_result.dose.ctdi_vol_mGy * helical_result.dose.pitch),
    Float64(helical_result.dose.nominal_collimation_mm),
    Float64(protocol_helical.rotation_time),
)
    all(isapprox.(HELICAL_INPUTS, measured; rtol = 0.01)) ||
        error(
            "HELICAL_INPUTS = $(HELICAL_INPUTS) no longer match this scan, $(measured): update them",
        )
    nothing
end

# ╔═╡ 1100000b-0000-4000-8000-000000000003
@bind pitch_x100 PlutoUI.Slider(50:5:150; default = 100, show_value = true)

# ╔═╡ 1100000b-0000-4000-8000-000000000004
# pitch, table feed (mm), CTDIvol (mGy), rotations and seconds for a 30 cm range
helical_plan = let p = pitch_x100 / 100
    feed = HELICAL_INPUTS[2] * p
    rotations = 300.0 / feed
    (
        p,
        round(feed; digits = 1),
        round(HELICAL_INPUTS[1] / p; digits = 2),
        round(rotations; digits = 1),
        round(rotations * HELICAL_INPUTS[3]; digits = 1),
    )
end;

# ╔═╡ 1100000b-0000-4000-8000-000000000005
md"""
**Pitch $(helical_plan[1]):** the table moves $(helical_plan[2]) mm per rotation, CTDIvol is $(helical_plan[3]) mGy, and a 30 cm range takes $(helical_plan[4]) rotations, $(helical_plan[5]) s.
"""

# ╔═╡ 11000007-0000-4000-8000-000000000001
md"""
## 6. Slide Through z: the Phantom against Both Scans

Top: where the slice is, on the phantom seen from the side. Middle: the phantom's labels in that
slice. Bottom: the **helical** scan (20 mm collimation) and the **volume axial** one (the full
16 cm detector, 8× wider). Watch the rod rotate and the cone shrink, and watch what the wide cone
does away from each station's centre; the station boundaries are at ``z = ±5`` cm.
"""

# ╔═╡ 11000007-0000-4000-8000-000000000002
@bind z_idx PlutoUI.Slider(1:150; default = 75, show_value = true)

# ╔═╡ 11000007-0000-4000-8000-000000000003
let
    Mke.set_theme!(THEME)
    z_cm = -15.0 + (z_idx - 0.5) * (30.0 / 150)
    kp = clamp(round(Int, (z_cm + 18.0) / phantom_data.voxz + 0.5), 1, phantom_data.nz)
    fig = Mke.Figure(size = (1100, 1400))

    # where the slice is: the phantom from the side, z across
    side = Mke.Axis(
        fig[1, 1:3];
        title = "z = $(round(z_cm; digits = 1)) cm (Slice $(z_idx) of 150)",
        xlabel = "z (cm)",
        limits = (-18, 18, -13, 13),
        yticksvisible = false,
        yticklabelsvisible = false,
        ygridvisible = false,
        leftspinevisible = false,
    )
    phantom_side!(side)
    Mke.vlines!(side, [-5.0, 5.0]; color = COLORS.muted, linestyle = :dash, linewidth = 2)
    Mke.vlines!(side, [z_cm]; color = COLORS.red, linewidth = 4)

    # the slice's labels: air, water, the lung rod, the fat cone
    label_colors = Mke.cgrad(
        [COLORS.night, COLORS.blue, COLORS.beam, COLORS.green];
        categorical = true,
    )
    truth = Mke.Axis(fig[2, 1:2]; title = "The Phantom's Labels", aspect = Mke.DataAspect())
    labels = Mke.heatmap!(
        truth,
        Float32.(phantom_data.mask[:, :, kp]);
        colormap = label_colors,
        colorrange = (-0.5, 3.5),
    )
    Mke.hidedecorations!(truth)
    Mke.hidespines!(truth)
    Mke.Colorbar(
        fig[2, 3],
        labels;
        ticks = (0:3, ["Air", "Water Body", "Lung Rod", "Adipose Cone"]),
        tellheight = false,
    )

    # the two reconstructions of the slice
    for (column, (title, volume)) in enumerate((
            ("Helical: 20 mm, Pitch 1.0", helical_result.hu),
            ("Volume Axial: 3 × 10 cm", sns_result.hu),
        ))
        image = Mke.Axis(fig[3, column]; title, aspect = Mke.DataAspect())
        Mke.heatmap!(
            image,
            volume[:, :, z_idx];
            colormap = :grays,
            colorrange = (-500, 100),
        )
        Mke.hidedecorations!(image)
        Mke.hidespines!(image)
    end
    Mke.Colorbar(fig[3, 3]; colormap = :grays, colorrange = (-500, 100), label = "HU")
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.2))
    fig
end

# ╔═╡ 11000008-0000-4000-8000-000000000001
md"""
## 7. Water Flatness Along z

The mean HU in a fixed water ROI, clear of the rod and the cone (cyan, top), slice by slice, with
the axial station boundaries marked; then coronal reformats of both volumes. The table under the
plot reduces the profiles to numbers.
"""

# ╔═╡ 71683b27-eb99-42f1-913d-9b1234da5815
# the water ROI of the profiles (9.4 cm left of centre, clear of the rod's ring and the cone), the
# slices' z (cm), and each scan's mean HU in it, slice by slice
water_profiles = let
    xr, yr = 25:36, 73:88
    zs = [-15.0 + (k - 0.5) * 0.2 for k in 1:150]
    pixel = 30.0 / 160
    x_cm(i) = (i - 80.5) * pixel
    profile(v) = [mean(v[xr, yr, k]) for k in 1:150]
    (
        xr,
        yr,
        zs,
        x_range = (x_cm(first(xr)) - pixel / 2, x_cm(last(xr)) + pixel / 2),
        helical = profile(helical_result.hu),
        axial = profile(sns_result.hu),
    )
end;

# ╔═╡ 11000008-0000-4000-8000-000000000002
z_profile_fig = let
    Mke.set_theme!(THEME)
    w = water_profiles
    fig = Mke.Figure(size = (1400, 900))

    # where the ROI is: the phantom from the side, on the plot's own z axis
    side = Mke.Axis(
        fig[1, 1];
        ylabel = "x (cm)",
        limits = (-15, 15, -13, 13),
        xticklabelsvisible = false,
        ygridvisible = false,
    )
    phantom_side!(side)
    Mke.band!(
        side,
        [-15.0, 15.0],
        fill(w.x_range[1], 2),
        fill(w.x_range[2], 2);
        color = (COLORS.cyan, 0.9),
    )
    Mke.text!(
        side,
        -14.6,
        sum(w.x_range) / 2;
        text = "Water ROI",
        align = (:left, :center),
        fontsize = 17,
        font = :bold,
        color = COLORS.paper,
    )

    ax = Mke.Axis(
        fig[2, 1];
        title = "Water HU vs z",
        subtitle = "Helical (20 mm) vs Volume Axial (160 mm), Matched Beam Width × Current",
        xlabel = "z (cm)",
        ylabel = "Mean HU (Water ROI)",
        limits = (-15, 15, -100, 100),
    )
    Mke.lines!(
        ax,
        w.zs,
        w.helical;
        color = COLORS.beam,
        linewidth = 4,
        label = "Helical: 20 mm, Pitch 1.0 × 16 Rotations",
    )
    Mke.lines!(
        ax,
        w.zs,
        w.axial;
        color = COLORS.violet,
        linewidth = 4,
        label = "Volume Axial: 3 × 10 cm Stations",
    )
    Mke.hlines!(ax, [0.0]; color = COLORS.muted, linestyle = :dash, linewidth = 2)
    for axis in (side, ax)
        Mke.vlines!(
            axis,
            [-5.0, 5.0];
            color = COLORS.muted,
            linestyle = :dot,
            linewidth = 3,
        )
    end
    Mke.axislegend(ax; position = :cb)
    Mke.linkxaxes!(side, ax)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.32))
    fig
end

# ╔═╡ 11000008-0000-4000-8000-000000000004
let
    w = water_profiles
    noise(v) = mean(std(v[w.xr, w.yr, k]) for k in 1:150)
    step_at(p, z) = (k = searchsortedfirst(w.zs, z); p[k] - p[k - 1])
    function row(name, p, hu)
        return "| $(name) | $(round(mean(p); digits = 1)) | $(round(minimum(p); digits = 1)) … $(round(maximum(p); digits = 1)) | " *
            "$(round(std(p); digits = 1)) | $(round(step_at(p, -5.0); digits = 1)) / $(round(step_at(p, 5.0); digits = 1)) | $(round(noise(hu); digits = 1)) |"
    end

    rows = [
        row("helical", w.helical, helical_result.hu),
        row("volume axial", w.axial, sns_result.hu),
    ]

    Markdown.parse(
        """
        | Acquisition | Mean water HU over z | Range over z | σ over z | Step at z = −5 / +5 cm | Pixel noise σ (HU) |
        |:--|--:|--:|--:|--:|--:|
        $(join(rows, "\n"))

        "Step" is the change in the ROI mean between the two slices either side of a station boundary;
        "pixel noise" is the within-ROI standard deviation averaged over the slices.
        """,
    )
end

# ╔═╡ 11000008-0000-4000-8000-000000000003
coronal_fig = let
    Mke.set_theme!(THEME)
    yc = 80
    pixel = 30.0 / 160
    xs = [(i - 80.5) * pixel for i in 1:160]
    zs = water_profiles.zs
    fig = Mke.Figure(size = (1400, 700))
    panels = (
        ("Helical: Coronal", helical_result.hu),
        ("Volume Axial (3 × 10 cm): Coronal", sns_result.hu),
    )
    for (column, (title, volume)) in enumerate(panels)
        ax = Mke.Axis(
            fig[1, column];
            title,
            xlabel = "z (cm)",
            ylabel = column == 1 ? "x (cm)" : "",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(
            ax,
            zs,
            xs,
            permutedims(volume[:, yc, :]);
            colormap = :grays,
            colorrange = (-500, 100),
        )
        Mke.vlines!(
            ax,
            [-5.0, 5.0];
            color = (COLORS.beam, 0.8),
            linestyle = :dot,
            linewidth = 2,
        )
    end
    Mke.Colorbar(fig[1, 3]; colormap = :grays, colorrange = (-500, 100), label = "HU")
    Mke.save(joinpath(@__DIR__, "..", "assets", "helical_vs_stepshoot_coronal.png"), fig)
    fig
end

# ╔═╡ 11000009-0000-4000-8000-000000000001
Markdown.parse(
    """
    ## 8. What This Shows, and Its Scope

    - **`:dd_fast` needed zero changes for helical.** The projectors consume per-view geometry arrays
      (the same representation as ASTRA's `cone_vec` and CatSim's internal trajectory); the helix lives
      entirely in those arrays, and so do the sub-views of the view integration, each rotated and
      advanced along ``z`` with the table.
    - **Helical reconstruction is rebinned WFBP** (Stierstorfer *et al.*, Phys Med Biol 49:2209, 2004):
      row-wise fan→parallel rebinning, parallel ramp filtering, and aperture-weighted (cos², plateau
      ``Q = 0.7``) wedge backprojection with per-half-turn redundancy normalisation. This is the
      production spiral-CT algorithm family (Siemens WFBP; UCLA FreeCT), and `reconstruct!` /
      `fdk_reconstruct` dispatch to it whenever `is_helical(geom)`.
    - **Both scans got identical corrections.** Exposure is matched by beam width × current:
      16 × 20 mm × 200 mA against 3 × 160 mm × 133⅓ mA at equal rotation time. That is an
      acquisition-integral match, not equal CTDIvol or equal noise per slice; the dose table in
      section 4 gives both scans' CTDIvol and DLP.
    - Hybrid IR works on helical unchanged: matched forward and backprojection are geometry-general,
      subsets are angular-interleaved, and the HIR FDK initialisation routes through WFBP.
    - Windmill artifacts round sharp ``z`` edges at high pitch are physics (longitudinal sampling),
      not a defect of the reconstruction.
    - Wall clock (this page was rendered on an NVIDIA RTX PRO 6000, CUDA): helical
      ($(helical_result.n_views) views, simulation + corrections + WFBP)
      $(round(helical_result.t; digits = 1)) s; volume axial (3 stations × $(protocol_axial.views) views)
      $(round(sns_result.t; digits = 1)) s. Both include first-call compilation.
    """,
)

# ╔═╡ 1100000a-0000-4000-8000-000000000001
let
    w = water_profiles
    h = helical_result.dose
    a = sns_result.doses
    f(x) = round(x; digits = 1)
    Markdown.parse(
        """
        ## Summary

        `pitch` and `n_rotations` are the only additions that turn the axial acquisition into a centred
        helical trajectory; `:dd_fast` projects it unchanged, and `reconstruct!` sends helical
        geometries to rebinned WFBP and axial stations to FDK.

        | | Helical, 20 mm | Volume axial, 3 × 160 mm |
        |---|--:|--:|
        | CTDIvol (mGy) | $(round(h.ctdi_vol_mGy; digits = 2)) | $(round(a[1].ctdi_vol_mGy; digits = 2)) |
        | DLP (mGy·cm) | $(f(h.dlp_mGy_cm)) | $(f(sum(d.dlp_mGy_cm for d in a))) |
        | water ROI, mean HU over z | $(f(mean(w.helical))) | $(f(mean(w.axial))) |
        | water ROI, range over z (HU) | $(f(minimum(w.helical))) … $(f(maximum(w.helical))) | $(f(minimum(w.axial))) … $(f(maximum(w.axial))) |
        | wall time, simulation + reconstruction (s) | $(f(helical_result.t)) | $(f(sns_result.t)) |
        """,
    )
end

# ╔═╡ Cell order:
# ╟─11000002-0000-4000-8000-000000000001
# ╟─27540bb6-e69c-4c78-8d86-10efe26727d9
# ╟─429c6786-c046-4fc3-9828-b1215c61267d
# ╟─11000002-0000-4000-8000-000000000002
# ╠═11000001-0000-4000-8000-000000000001
# ╠═11000001-0000-4000-8000-000000000004
# ╠═11000001-0000-4000-8000-000000000005
# ╠═11000001-0000-4000-8000-000000000002
# ╠═11000001-0000-4000-8000-000000000003
# ╠═11000001-0000-4000-8000-000000000006
# ╠═11000002-0000-4000-8000-000000000003
# ╟─1d85cd7b-bc6b-4b40-9e0b-2bfcec03e7ac
# ╟─04505917-64f0-4832-91c8-67aeb83a9c16
# ╠═e92386a8-e385-45da-87a5-a8fc2e43ab52
# ╠═6fa82e51-0f08-4da2-b395-f56f03d5b131
# ╠═a463b2db-f3bb-4428-b5e4-befcadff26c2
# ╠═9258a3c9-6d6e-45d9-abe4-761db7b02828
# ╠═e77961b0-08a4-4207-86fd-eb47141416f2
# ╠═6c62678d-b885-415f-a6e1-e8ab537c47ae
# ╟─11000003-0000-4000-8000-000000000001
# ╠═11000003-0000-4000-8000-000000000002
# ╠═11000003-0000-4000-8000-000000000003
# ╠═11000003-0000-4000-8000-000000000004
# ╟─11000004-0000-4000-8000-000000000001
# ╠═11000004-0000-4000-8000-000000000002
# ╟─31a9e8a0-c16a-511e-a0a0-c0610c198163
# ╠═11000004-0000-4000-8000-000000000003
# ╟─11000005-0000-4000-8000-000000000001
# ╟─baf87a28-d676-44df-a58c-046ea4fbb7d8
# ╠═11000005-0000-4000-8000-000000000002
# ╟─11000006-0000-4000-8000-000000000001
# ╠═11000006-0000-4000-8000-000000000002
# ╠═11000006-0000-4000-8000-000000000003
# ╟─11000006-0000-4000-8000-000000000004
# ╟─1100000b-0000-4000-8000-000000000001
# ╠═1100000b-0000-4000-8000-000000000002
# ╠═1100000b-0000-4000-8000-000000000006
# ╠═1100000b-0000-4000-8000-000000000003
# ╠═1100000b-0000-4000-8000-000000000004
# ╟─1100000b-0000-4000-8000-000000000005
# ╟─11000007-0000-4000-8000-000000000001
# ╠═11000007-0000-4000-8000-000000000002
# ╟─11000007-0000-4000-8000-000000000003
# ╟─11000008-0000-4000-8000-000000000001
# ╠═71683b27-eb99-42f1-913d-9b1234da5815
# ╟─11000008-0000-4000-8000-000000000002
# ╟─11000008-0000-4000-8000-000000000004
# ╟─11000008-0000-4000-8000-000000000003
# ╟─11000009-0000-4000-8000-000000000001
# ╟─1100000a-0000-4000-8000-000000000001
