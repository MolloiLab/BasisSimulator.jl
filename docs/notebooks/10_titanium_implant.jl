### A Pluto.jl notebook ###
# v0.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 10000002-0000-4000-8000-000000000001
md"""
# Titanium Implant: Metal Artifacts

**Define a metal yourself, put it in a water phantom, and measure the artifacts it leaves in the
standard reconstruction chain.**

No metal ships in the built-in material table, and that is the point: any implant is one
`XA.Material` call, an elemental composition and a density, with NIST XCOM (through
XrayAttenuation.jl) supplying its cross-sections. Two 1.5 cm titanium rods in a 26 cm water
cylinder are scanned beside an identical cylinder whose rods are water, so their difference is the
artifact alone.
"""

# ╔═╡ a62dc649-c9bb-48a8-950a-d8167e65935e
let
    Mke.set_theme!(THEME)
    k = size(scan_ti.hu, 3) ÷ 2
    half = recon_opts.fov_cm / 2
    fig, axes = locator_band(
        ["The Implant", "The Scanner", "The Artifact"],
        [
            "Two 1.5 cm Titanium Rods in Water",
            "GE Revolution Apex, 16 Rows",
            "Titanium Scan − Metal-Free Scan",
        ];
        limits = [(-15, 15, -15, 15), (-7.5, 7.5, -7.5, 7.5), (-half, half, -half, half)],
    )
    light_metal = Mke.RGBf(0.82, 0.84, 0.87)
    draw_rods!(
        axes[1],
        phantom_mask.mask[:, :, 1],
        phantom_mask.vox;
        rod = light_metal,
        alpha = 0.6,
    )
    gantry!(axes[2]; labels = ("X-Ray Tube",), fan_deg = 45.8, object = false)
    draw_rods!(
        axes[2],
        phantom_mask.mask[:, :, 1],
        phantom_mask.vox * 2.2 / 13;
        rod = light_metal,
        alpha = 0.6,
    )

    artifact = scan_ti.hu[:, :, k] .- scan_ref.hu[:, :, k]
    artifact[
        inside_water(size(artifact, 1), recon_opts.fov_cm / size(artifact, 1)) .== false,
    ] .= NaN
    Mke.image!(
        axes[3],
        (-half, half),
        (-half, half),
        artifact;
        colormap = :RdBu,
        colorrange = (-300, 300),
        nan_color = :transparent,
    )
    fig
end

# ╔═╡ 0667efe5-5cb2-4a24-a21e-874bfce13247
md"""
Metal artifacts come from two physical mechanisms that `BasisSimulator.jl` models by
construction:

1. **Beam hardening.** The polychromatic forward model sums ``I = \sum_E w_E\, e^{-L_E}`` before
   the log, so a dense high-``Z`` object hardens the transmitted spectrum and the line integrals
   stop growing in proportion to the path length. Reconstructed: dark bands, most of all
   *between* two metal objects.
2. **Photon starvation.** Behind thick metal few photons reach the detector, so quantum and
   electronic noise dominate those rays. Reconstructed: noisy streaks along the directions that
   cross the metal.

The reconstruction is the package's standard chain, a water beam-hardening correction and FDK.
The package ships no metal-artifact-reduction (MAR) algorithm, and the water correction is not
designed for metal; the notebook measures how much artifact is left.
"""

# ╔═╡ 1000b39a-4f11-4af1-b5ac-b87b6d95ffc8
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 400))
    nodes = Dict(
        :ti => (0.0, 0.8, "Titanium", "XA.Material, Ti 4.54 g/cm³", METAL),
        :water => (0.0, -0.8, "Water", "XA.Materials.water", COLORS.blue),
        :ph_ti => (4.0, 0.8, "Titanium Phantom", "Two Rods in 26 cm Water"),
        :ph_ref => (4.0, -0.8, "Reference Phantom", "The Rods Made of Water"),
        :scan => (8.0, 0.0, "scan_hu", "simulate!, Water BHC, FDK", COLORS.cyan),
        :images => (12.0, 0.0, "Two HU Images", "Same Noise Seed", COLORS.cyan),
        :diff => (16.0, 0.0, "Difference", "The Artifact Alone", COLORS.green),
    )
    edges = [
        :ti => :ph_ti,
        :water => :ph_ref,
        :ph_ti => :scan,
        :ph_ref => :scan,
        :scan => :images,
        :images => :diff,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.9, 17.9, -1.5, 1.6), w = 3.4)
    fig
end

# ╔═╡ 10000002-0000-4000-8000-000000000002
Markdown.parse(
    """
## Notebook Setup

The shared docs environment and the device choice of notebook 01.

**Backend detected:** $(GPU_BACKEND.name)
""",
)

# ╔═╡ 10000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 10000001-0000-4000-8000-000000000002
import BasisSimulator as BS

# ╔═╡ 10000001-0000-4000-8000-000000000003
# ╠═╡ show_logs = false
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 10000001-0000-4000-8000-000000000007
import PlutoUI

# ╔═╡ 10000001-0000-4000-8000-000000000004
using Statistics: mean, std

# ╔═╡ 10000001-0000-4000-8000-000000000005
using Unitful: @u_str

# ╔═╡ 10000001-0000-4000-8000-000000000006
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 10000002-0000-4000-8000-000000000003
PlutoUI.TableOfContents()

# ╔═╡ e382ad41-4be6-45d7-8b93-383c4817a270
md"""
## Figure Style

BasisSimulator's colours and type, and the drawing helpers every notebook's figures share.
"""

# ╔═╡ 2a39ebed-e4a9-43b9-bc7b-af934a4859a6
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

# ╔═╡ 008737ed-a53e-4517-a391-e00ae54bb545
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

# ╔═╡ ce7451cc-ef38-4300-95c7-5ebe59fbe42a
"""A blank drawing area at `position`: equal scales, no axes, limits `(x0, x1, y0, y1)`."""
function canvas(position, limits; background = COLORS.paper)
    ax = Mke.Axis(position; aspect = Mke.DataAspect(), backgroundcolor = background)
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    Mke.limits!(ax, limits...)
    return ax
end;

# ╔═╡ a3a1767d-a580-4031-bc66-2b81cf9034a7
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

# ╔═╡ 31541e14-f64f-44fa-959b-a1cbd5f2785a
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

# ╔═╡ 72a7326c-0f5a-4a23-a2be-8e4b47b0e3bc
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

# ╔═╡ 48d56533-d1f7-49f9-ab5a-65fec1235fb1
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

# ╔═╡ 44d77da6-14f1-4237-ac17-ab43a5ad0327
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

# ╔═╡ 4a059481-953c-4f36-9f8c-67f3a1bf3f1c
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

# ╔═╡ 379819f6-ec19-4686-b775-b62f1f55edd4
"""The colour titanium is drawn in: a steel grey."""
const METAL = Mke.RGBf(0.45, 0.47, 0.5)

# ╔═╡ d34b7681-4255-43a5-9586-69277b9922c1
"""
The phantom `mask` slice (0 air, 1 water, 2 rod) drawn on `ax` in cm, centred, `vox` cm per
voxel, air left clear; `rod` colours the rods, `alpha` the water.
"""
function draw_rods!(ax, mask, vox; rod = METAL, alpha = 0.45)
    n1, n2 = size(mask)
    colours = Dict(
        0 => Mke.RGBAf(0, 0, 0, 0),
        1 => Mke.RGBAf(COLORS.blue.r, COLORS.blue.g, COLORS.blue.b, alpha),
        2 => Mke.RGBAf(rod),
    )
    Mke.image!(
        ax,
        (-n1 / 2 * vox, n1 / 2 * vox),
        (-n2 / 2 * vox, n2 / 2 * vox),
        [colours[Int(m)] for m in mask];
        interpolate = false,
    )
    return ax
end

# ╔═╡ cdb1aad2-9e72-44d7-82ac-9acb38dfa757
"""The rectangle (cm) that pixels `is × js` cover on an `n`-pixel grid of `dx` cm pixels, centred."""
function pixel_rect(is, js, n, dx)
    x0 = (first(is) - 0.5 - (n + 1) / 2) * dx
    y0 = (first(js) - 0.5 - (n + 1) / 2) * dx
    return Mke.Rect2f(x0, y0, length(is) * dx, length(js) * dx)
end

# ╔═╡ 537a7178-8b1c-460e-9c56-5936f18f8d70
"""Which pixels of an `n`-pixel grid of `dx` cm pixels lie inside the 26 cm water cylinder."""
inside_water(n, dx) = [
    hypot((i - (n + 1) / 2) * dx, (j - (n + 1) / 2) * dx) <= 13.0
    for i in 1:n, j in 1:n
]

# ╔═╡ 0bb6c4bd-0697-4b6a-b35c-3f3b90860f8c
"""A dimension line from `a` to `b`, arrowheads at both ends, `text` beside its middle."""
function dimension!(
    ax,
    a,
    b,
    text;
    align = (:center, :bottom),
    offset = (0, 6),
    fontsize = 16,
)
    m = (Mke.Point2f(a) + Mke.Point2f(b)) / 2
    arrow!(ax, m, a; linewidth = 1.8, head = 11)
    arrow!(ax, m, b; linewidth = 1.8, head = 11)
    Mke.text!(ax, m; text, align, offset, fontsize, color = COLORS.ink)
    return nothing
end

# ╔═╡ 10000003-0000-4000-8000-000000000001
md"""
## 1. Titanium as a Custom `XA.Material`

Titanium is exposed by `XrayAttenuation.jl` as an `Element`, not a prebuilt `Material`, so we
construct the material ourselves: pure Ti (``Z = 22``, mass fraction 1.0) at 4.54 g/cm³. `ZA_ratio`
and the mean excitation energy `I` come from the element data (0.45948, 233 eV).
"""

# ╔═╡ 10000003-0000-4000-8000-000000000002
titanium = BS.XA.Material(
    "Titanium (implant)",
    0.45948,         # ⟨Z/A⟩ for Ti
    233.0u"eV",      # mean excitation energy
    4.54u"g/cm^3",   # bulk density
    Dict(22 => 1.0), # elemental composition: pure Ti
);

# ╔═╡ 82c05b19-7a43-4171-96f9-d17fbf1c1c74
md"""
Why titanium hardens the beam: 1.5 cm of water lets most photons of a 120 kVp spectrum through
at every energy, but 1.5 cm of titanium stops nearly all of them below about 50 keV. What crosses
a rod is a much harder spectrum, and a water correction maps its line integrals as if it were
not.
"""

# ╔═╡ 960697ce-01d5-4059-84fd-ae1e3db2494a
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 860))

    # the physical situation: one beam through each slab
    ax0 = canvas(fig[1, 1], (-1, 23, -0.6, 5.6))
    water = Mke.RGBAf(COLORS.blue.r, COLORS.blue.g, COLORS.blue.b, 0.45)
    slabs = [
        (4.0, "1.5 cm of Water", water, 6, 0.85, "Most of the Beam\nGets Through"),
        (
            1.0,
            "1.5 cm of Titanium",
            Mke.RGBAf(METAL),
            2,
            0.45,
            "Only the Highest\nEnergies Get Through",
        ),
    ]
    for (y, name, colour, width, alpha, outcome) in slabs
        Mke.poly!(
            ax0,
            Mke.Rect2f(8.5, y - 0.8, 3.0, 1.6);
            color = colour,
            strokecolor = COLORS.ink,
            strokewidth = 1,
        )
        arrow!(ax0, (1.0, y), (8.3, y); color = COLORS.beam, linewidth = 6, head = 22)
        arrow!(
            ax0,
            (11.7, y),
            (17.0, y);
            color = (COLORS.beam, alpha),
            linewidth = width,
            head = 22,
        )
        Mke.text!(
            ax0,
            10.0,
            y + 0.95;
            text = name,
            align = (:center, :bottom),
            fontsize = 18,
            font = :bold,
        )
        Mke.text!(
            ax0,
            17.4,
            y;
            text = outcome,
            align = (:left, :center),
            fontsize = 16,
            color = COLORS.muted,
        )
    end
    Mke.text!(
        ax0,
        1.0,
        5.3;
        text = "120 kVp Beam",
        align = (:left, :center),
        fontsize = 16,
        color = COLORS.muted,
    )

    # the measurement: transmission through each, from the materials' own cross-sections
    E_keV = 20.0:1.0:120.0
    transmission(material, L) = [
        exp(-BS.compute_μ_at_energy(material, e) * L)
        for e in E_keV
    ]
    ax = Mke.Axis(
        fig[2, 1];
        title = "What Each Slab Lets Through",
        subtitle = "exp(−μ(E) L), From NIST XCOM via XrayAttenuation.jl",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction Transmitted",
        yscale = log10,
    )
    Mke.lines!(
        ax,
        E_keV,
        transmission(BS.XA.Materials.water, 1.5);
        color = COLORS.blue,
        label = "Water, 1.5 cm",
    )
    Mke.lines!(
        ax,
        E_keV,
        transmission(titanium, 1.5);
        color = METAL,
        label = "Titanium, 1.5 cm (One Rod)",
    )
    Mke.lines!(
        ax,
        E_keV,
        transmission(titanium, 3.0);
        color = METAL,
        linestyle = :dash,
        label = "Titanium, 3 cm (Both Rods)",
    )
    Mke.ylims!(ax, 1.0e-4, 1.5)
    Mke.axislegend(ax; position = :rb)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.3))
    fig
end

# ╔═╡ 10000004-0000-4000-8000-000000000001
md"""
## 2. The Phantom: Water Cylinder and Two Titanium Rods

A 26 cm water cylinder with two 1.5 cm titanium rods, 4 cm either side of the centre (two rods,
so that the classic *between-rod dark band* appears). A second phantom is the same cylinder with
the rods made of water: scanned identically, it is the artifact-free reference every measurement
below is compared with.
"""

# ╔═╡ 10000004-0000-4000-8000-000000000002
phantom_mask = let
    n = 128
    nz = 8
    vox = 30.0 / n # 30 cm field, cm per voxel
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    r_water = 13.0 / vox # 26 cm diameter water cylinder
    r_ti = 0.75 / vox    # 1.5 cm diameter rods
    rods = ((c - 4.0 / vox, c), (c + 4.0 / vox, c))
    for k in 1:nz, j in 1:n, i in 1:n
        (i - c)^2 + (j - c)^2 <= r_water^2 || continue
        mask[i, j, k] = any((i - x)^2 + (j - y)^2 <= r_ti^2 for (x, y) in rods) ? 2 : 1
    end
    (mask = mask, vox = vox)
end;

# ╔═╡ 10000004-0000-4000-8000-000000000003
phantom = BS.Phantom(
    to_gpu(phantom_mask.mask),
    Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water, 2 => titanium),
    ntuple(_ -> phantom_mask.vox, 3),
);

# ╔═╡ 10000004-0000-4000-8000-000000000004
phantom_reference = BS.Phantom( # the same geometry, rods made of water
    to_gpu(phantom_mask.mask),
    Dict(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water, 2 => BS.XA.Materials.water),
    ntuple(_ -> phantom_mask.vox, 3),
);

# ╔═╡ 00d11668-2033-4ebd-a584-d6d8a1624df4
let
    Mke.set_theme!(THEME)
    slice = phantom_mask.mask[:, :, 1]
    vox = phantom_mask.vox
    fig = Mke.Figure(size = (1200, 600))
    panels = [
        ("Titanium Phantom", "Labels 1: Water, 2: Titanium", METAL),
        ("Reference Phantom", "The Same Mask, Label 2 Is Water", blend_water()),
    ]
    for (i, (title, subtitle, rod)) in enumerate(panels)
        ax = Mke.Axis(fig[1, i]; title, subtitle, aspect = Mke.DataAspect())
        Mke.hidedecorations!(ax)
        Mke.hidespines!(ax)
        Mke.limits!(ax, -15, 15, -17, 14)
        draw_rods!(ax, slice, vox; rod)
        dimension!(
            ax,
            (-13, -14.2),
            (13, -14.2),
            "26 cm";
            align = (:center, :top),
            offset = (0, -6),
        )
        dimension!(ax, (-4, 3.2), (4, 3.2), "8 cm Apart")
        dimension!(
            ax,
            (3.25, -1.6),
            (4.75, -1.6),
            "1.5 cm";
            align = (:center, :top),
            offset = (0, -6),
        )
    end
    fig
end

# ╔═╡ 622bca5e-a57f-4cb2-9efc-1ee6f1d1782a
"""The reference phantom's rods: water, the colour `draw_rods!` gives water, without transparency."""
blend_water() = Mke.RGBf(
    0.45 * COLORS.blue.r + 0.55 * COLORS.paper.r,
    0.45 * COLORS.blue.g + 0.55 * COLORS.paper.g,
    0.45 * COLORS.blue.b + 0.55 * COLORS.paper.b,
)

# ╔═╡ 10000005-0000-4000-8000-000000000001
md"""
## 3. Scan and Reconstruct

A 120 kVp / 200 mA axial acquisition (360 views, 0.5 s) on the GE Revolution Apex
(`REVOLUTION_APEX`, its Gemstone detector, bowtie and DAS) narrowed to a compact 16-row detector
of 512 one-millimetre columns, each view integrated over the arc the gantry turns through while
the detector reads it (`view_samples = 5`, notebook 01), then the standard chain:
`calibrate_bhc_water` → `apply_bhc_water` → FDK → HU. The function below runs it for either
phantom and also returns the largest line integral in the sinogram.

The scan includes the patient's scatter (on by default), from the Geant4 scatter kernels the Apex
names, and the DAS's scatter correction. The kernels are made for one scanner's distances, so the
scanner is a variant of the Apex at its own 625.6 and 1100 mm, with only the detector's rows and
columns changed.
"""

# ╔═╡ 10000005-0000-4000-8000-000000000002
scanner = BS.EICTScanner(
    BS.REVOLUTION_APEX;
    detector_rows = 16,
    detector_cols = 512,
    detector_row_size = 1.0,
    detector_col_size = 1.0,
);

# ╔═╡ 6cdbea4a-7a0f-5155-a0da-bf4b499298f6
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

# ╔═╡ 10000005-0000-4000-8000-000000000003
protocol = BS.CTProtocol(kVp = 120.0, mA = 200.0, views = 360, rotation_time = 0.5);

# ╔═╡ 10000005-0000-4000-8000-000000000004
sim_opts = BS.SimOptions(seed = 42, projector = :dd_fast, view_samples = 5);

# ╔═╡ 10000005-0000-4000-8000-000000000005
recon_opts = BS.ReconOptions(matrix_size = (256, 256, 8), fov_cm = 30.0);

# ╔═╡ 10000005-0000-4000-8000-000000000006
"""
    scan_hu(phantom) -> (; hu, max_line_integral)

`create_workspace` → `simulate!` → water BHC → FDK → HU, returning a CPU volume.
"""
function scan_hu(phantom)
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol, sim_opts; report_dose = false)
    bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom)
    sino = BS.apply_bhc_water(ws.sinogram, bhc)
    ws_fdk = BS.create_fdk_recon_workspace(sino, ws.geom, recon_opts.matrix_size)
    μ = BS.reconstruct!(ws_fdk, sino, ws.geom)
    out = (
        hu = Float32.(BS.to_hounsfield(Array(μ); μ_water = bhc.μ_water_ref)),
        max_line_integral = Float64(maximum(ws.sinogram)),
    )
    ws = nothing
    ws_fdk = nothing
    sino = nothing
    μ = nothing
    GC.gc(true)
    return out
end;

# ╔═╡ 10000005-0000-4000-8000-000000000007
scan_ti = scan_hu(phantom);

# ╔═╡ 10000005-0000-4000-8000-000000000008
scan_ref = scan_hu(phantom_reference);

# ╔═╡ 10000006-0000-4000-8000-000000000001
md"""
## 4. The Artifacts

### Where They Are Measured

Every number below compares the titanium scan with the metal-free one on the central slice, in
three regions: the water **between the rods**, where beam hardening leaves its dark band; water
**far from the rods**, 8 cm below their axis; and a **streak region** 3–4.7 cm above the axis,
whose spread is the streaks. The profile runs along the rods' axis.
"""

# ╔═╡ 86b3a0a7-ed32-4c96-8db3-129f57c63b8f
# The measurement regions, as pixel ranges of the central slice (`cn` is the middle pixel)
regions = let
    n = size(scan_ti.hu, 1)
    cn = n ÷ 2
    (
        n = n,
        k = size(scan_ti.hu, 3) ÷ 2,
        dx = recon_opts.fov_cm / n,
        between = ((cn - 8):(cn + 8), (cn - 2):(cn + 2)), # between the rods
        away = ((cn - 8):(cn + 8), (cn - 72):(cn - 64)),  # water, 8 cm below the rod axis
        streaks = (
            (cn - 40):(cn + 40),
            (cn + 30):(cn + 40),
        ), # water band 3–4.7 cm above the axis
        profile_row = cn,
    )
end;

# ╔═╡ 3c72556c-577b-4d97-a79b-4723410a650c
# The mean (or spread) of each region in both scans
artifact = let
    ti = scan_ti.hu[:, :, regions.k]
    ref = scan_ref.hu[:, :, regions.k]
    region_mean(v, r) = Float64(mean(v[r[1], r[2]]))
    region_std(v, r) = Float64(std(v[r[1], r[2]]))
    (
        between = (
            ref = region_mean(ref, regions.between),
            ti = region_mean(ti, regions.between),
        ),
        away = (ref = region_mean(ref, regions.away), ti = region_mean(ti, regions.away)),
        streaks = (
            ref = region_std(ref, regions.streaks),
            ti = region_std(ti, regions.streaks),
        ),
        peak = Float64(maximum(ti)),
    )
end;

# ╔═╡ 5fbefca0-a505-4013-a8c1-53bde291280c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1150, 660))
    ax = canvas(fig[1, 1], (-15, 15, -15, 15))
    ax.title = "The Measurement Regions, Central Slice"
    draw_rods!(ax, phantom_mask.mask[:, :, 1], phantom_mask.vox)
    boxes = [
        (regions.between, COLORS.red, "Between the Rods: Mean"),
        (regions.away, COLORS.green, "Far From the Rods: Mean"),
        (regions.streaks, COLORS.violet, "Streak Region: σ"),
    ]
    for (r, colour, _) in boxes
        Mke.poly!(
            ax,
            pixel_rect(r[1], r[2], regions.n, regions.dx);
            color = (colour, 0.35),
            strokecolor = colour,
            strokewidth = 2.5,
        )
    end
    y = (regions.profile_row - (regions.n + 1) / 2) * regions.dx
    Mke.lines!(
        ax,
        [-14.5, 14.5],
        [y, y];
        color = COLORS.ink,
        linestyle = :dash,
        linewidth = 1.5,
    )

    swatch(colour) = Mke.PolyElement(;
        color = (colour, 0.5),
        strokecolor = colour,
        strokewidth = 2,
    )
    Mke.Legend(
        fig[1, 2],
        [
            [swatch(c) for (_, c, _) in boxes]...,
            Mke.LineElement(; color = COLORS.ink, linestyle = :dash, linewidth = 1.5),
            Mke.PolyElement(; color = METAL, strokecolor = :transparent),
        ],
        [[label for (_, _, label) in boxes]..., "The Profile Below", "Titanium Rods"];
        valign = :center,
    )
    fig
end

# ╔═╡ 35e8e9c9-1cbb-40a3-9a4d-c7d3722ab02a
md"""
### The Images

Top: the metal-free reference and the titanium scan in a soft-tissue window. Bottom: the titanium
scan in a wide window that shows the rods themselves, and the difference titanium − reference,
which is the artifact alone (the anatomy cancels).
"""

# ╔═╡ 10000006-0000-4000-8000-000000000002
fig_artifacts = let
    Mke.set_theme!(THEME)
    k = regions.k
    ti, ref = scan_ti.hu[:, :, k], scan_ref.hu[:, :, k]
    half = recon_opts.fov_cm / 2
    outside = .!inside_water(regions.n, regions.dx)
    difference = ti .- ref
    difference[outside] .= NaN

    fig = Mke.Figure(size = (1200, 1060))
    function image_axis(position, title, subtitle)
        ax = Mke.Axis(position; title, subtitle, aspect = Mke.DataAspect())
        Mke.hidedecorations!(ax)
        Mke.hidespines!(ax)
        return ax
    end
    xy = ((-half, half), (-half, half))

    panels = [
        (1, 1, ref, "Reference, No Metal", (-200, 200), :grays, "HU"),
        (1, 3, ti, "Titanium Rods", (-200, 200), :grays, "HU"),
        (2, 1, ti, "Titanium Rods, Wide Window", (-1000, 10000), :grays, "HU"),
        (
            2,
            3,
            difference,
            "The Artifact: Titanium − Reference",
            (-300, 300),
            :RdBu,
            "ΔHU (Red: Too Dark)",
        ),
    ]
    for (r, c, image, title, window, colormap, label) in panels
        ax = image_axis(fig[r, c], title, "Window $(window[1]) to $(window[2])")
        hm = Mke.heatmap!(
            ax,
            xy...,
            image;
            colormap,
            colorrange = window,
            nan_color = COLORS.paper,
        )
        Mke.Colorbar(fig[r, c + 1], hm; label)
    end
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "titanium_artifacts.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 1577f6f5-947b-49da-8fc5-e5b65f535310
let
    Mke.set_theme!(THEME)
    k = regions.k
    row = regions.profile_row
    x_cm = ((1:regions.n) .- (regions.n + 1) / 2) .* regions.dx

    fig = Mke.Figure(size = (1200, 820))
    # the physical situation: the strip of phantom the profile crosses, on the same x axis
    ax0 = Mke.Axis(fig[1, 1]; title = "The Profile Along the Rods' Axis")
    Mke.hidedecorations!(ax0)
    Mke.hidespines!(ax0)
    draw_rods!(ax0, phantom_mask.mask[:, :, 1], phantom_mask.vox)
    y = (row - (regions.n + 1) / 2) * regions.dx
    Mke.lines!(
        ax0,
        [-15, 15],
        [y, y];
        color = COLORS.ink,
        linestyle = :dash,
        linewidth = 1.5,
    )
    Mke.limits!(ax0, -15, 15, -2.2, 2.2)

    ax = Mke.Axis(
        fig[2, 1];
        xlabel = "x (cm)",
        ylabel = "HU",
        subtitle = "Clipped at 1000 HU; the Rods Read Several Thousand",
    )
    Mke.lines!(
        ax,
        x_cm,
        scan_ref.hu[:, row, k];
        color = COLORS.blue,
        label = "Reference, No Metal",
    )
    Mke.lines!(
        ax,
        x_cm,
        scan_ti.hu[:, row, k];
        color = COLORS.beam,
        label = "Titanium Rods",
    )
    Mke.hlines!(ax, [0.0]; color = COLORS.rule, linewidth = 1.5)
    Mke.ylims!(ax, -400, 1000)
    Mke.xlims!(ax, -15, 15)
    Mke.linkxaxes!(ax0, ax)
    Mke.axislegend(ax; position = :lt)
    Mke.rowsize!(fig.layout, 1, Mke.Fixed(165))
    fig
end

# ╔═╡ 10000006-0000-4000-8000-000000000003
let
    a = artifact
    r1(x) = round(x; digits = 1)
    Markdown.parse(
        """
    | Measurement | Reference (No Metal) | Titanium | Titanium − Reference |
    |:--|--:|--:|--:|
    | between the rods, mean HU | $(r1(a.between.ref)) | $(r1(a.between.ti)) | $(r1(a.between.ti - a.between.ref)) |
    | water far from the rods, mean HU | $(r1(a.away.ref)) | $(r1(a.away.ti)) | $(r1(a.away.ti - a.away.ref)) |
    | streak region beside the rods, σ (HU) | $(r1(a.streaks.ref)) | $(r1(a.streaks.ti)) | |
    | peak HU (rods) | | $(round(Int, a.peak)) | |
    | largest line integral in the sinogram | $(round(scan_ref.max_line_integral; digits = 2)) | $(round(scan_ti.max_line_integral; digits = 2)) | |

    The water BHC holds the metal-free cylinder at water ($(r1(a.away.ref)) HU far from the rods).
    With titanium present, the region between the rods drops by
    $(round(Int, a.between.ref - a.between.ti)) HU: the beam-hardening dark band that a water
    correction cannot remove. The σ beside the rods rises from $(r1(a.streaks.ref)) to
    $(r1(a.streaks.ti)) HU, the streaks. A line integral of
    $(round(scan_ti.max_line_integral; digits = 1)) means the rays through both rods keep about
    $(round(100 * exp(-scan_ti.max_line_integral); sigdigits = 2)) % of their unattenuated signal.
    """,
    )
end

# ╔═╡ 10000007-0000-4000-8000-000000000001
md"""
## 5. Scope

- The water beam-hardening correction is a per-column **water** polynomial. It is exact for
  water and leaves the metal artifacts, as measured above. A MAR method (for example sinogram
  inpainting over the metal trace) would sit on the same reconstruction stack; none ships with
  the package.
- Any other implant works the same way: stainless steel, CoCr or amalgam are one
  `XA.Material(...)` call each, with NIST XCOM supplying the cross-sections.
"""

# ╔═╡ 10000008-0000-4000-8000-000000000001
let
    a = artifact
    Markdown.parse(
        """
    ## Summary

    - A user-defined `XA.Material` enters the same attenuation pipeline as every built-in tissue.
    - Scanning the same geometry with and without the metal isolates the artifact: the water
      between the rods reads $(round(Int, a.between.ref - a.between.ti)) HU too dark, the
      spread beside them rises from $(round(a.streaks.ref; digits = 1)) to
      $(round(a.streaks.ti; digits = 1)) HU, and water far from the rods moves by
      $(round(a.away.ti - a.away.ref; digits = 1)) HU.
    - The rays through both rods keep about $(round(100 * exp(-scan_ti.max_line_integral); sigdigits = 2)) %
      of their signal, which is where the streaks' noise comes from.
    - `:dd_fast`, the default projector, keeps its anti-aliased distance-driven footprint in the
      severe beam hardening around titanium.
    - No MAR claim is made; these numbers are the baseline a future MAR method would be measured
      against.
    """,
    )
end

# ╔═╡ Cell order:
# ╟─10000002-0000-4000-8000-000000000001
# ╟─a62dc649-c9bb-48a8-950a-d8167e65935e
# ╟─0667efe5-5cb2-4a24-a21e-874bfce13247
# ╟─1000b39a-4f11-4af1-b5ac-b87b6d95ffc8
# ╟─10000002-0000-4000-8000-000000000002
# ╟─10000001-0000-4000-8000-000000000001
# ╠═10000001-0000-4000-8000-000000000002
# ╟─10000001-0000-4000-8000-000000000003
# ╟─10000001-0000-4000-8000-000000000007
# ╟─10000001-0000-4000-8000-000000000004
# ╟─10000001-0000-4000-8000-000000000005
# ╠═10000001-0000-4000-8000-000000000006
# ╟─10000002-0000-4000-8000-000000000003
# ╟─e382ad41-4be6-45d7-8b93-383c4817a270
# ╠═2a39ebed-e4a9-43b9-bc7b-af934a4859a6
# ╠═008737ed-a53e-4517-a391-e00ae54bb545
# ╠═ce7451cc-ef38-4300-95c7-5ebe59fbe42a
# ╠═a3a1767d-a580-4031-bc66-2b81cf9034a7
# ╠═31541e14-f64f-44fa-959b-a1cbd5f2785a
# ╠═72a7326c-0f5a-4a23-a2be-8e4b47b0e3bc
# ╠═48d56533-d1f7-49f9-ab5a-65fec1235fb1
# ╠═44d77da6-14f1-4237-ac17-ab43a5ad0327
# ╠═4a059481-953c-4f36-9f8c-67f3a1bf3f1c
# ╠═379819f6-ec19-4686-b775-b62f1f55edd4
# ╠═d34b7681-4255-43a5-9586-69277b9922c1
# ╠═cdb1aad2-9e72-44d7-82ac-9acb38dfa757
# ╠═537a7178-8b1c-460e-9c56-5936f18f8d70
# ╠═0bb6c4bd-0697-4b6a-b35c-3f3b90860f8c
# ╟─10000003-0000-4000-8000-000000000001
# ╠═10000003-0000-4000-8000-000000000002
# ╟─82c05b19-7a43-4171-96f9-d17fbf1c1c74
# ╟─960697ce-01d5-4059-84fd-ae1e3db2494a
# ╟─10000004-0000-4000-8000-000000000001
# ╠═10000004-0000-4000-8000-000000000002
# ╠═10000004-0000-4000-8000-000000000003
# ╠═10000004-0000-4000-8000-000000000004
# ╟─00d11668-2033-4ebd-a584-d6d8a1624df4
# ╠═622bca5e-a57f-4cb2-9efc-1ee6f1d1782a
# ╟─10000005-0000-4000-8000-000000000001
# ╠═10000005-0000-4000-8000-000000000002
# ╟─6cdbea4a-7a0f-5155-a0da-bf4b499298f6
# ╠═10000005-0000-4000-8000-000000000003
# ╠═10000005-0000-4000-8000-000000000004
# ╠═10000005-0000-4000-8000-000000000005
# ╠═10000005-0000-4000-8000-000000000006
# ╠═10000005-0000-4000-8000-000000000007
# ╠═10000005-0000-4000-8000-000000000008
# ╟─10000006-0000-4000-8000-000000000001
# ╠═86b3a0a7-ed32-4c96-8db3-129f57c63b8f
# ╠═3c72556c-577b-4d97-a79b-4723410a650c
# ╟─5fbefca0-a505-4013-a8c1-53bde291280c
# ╟─35e8e9c9-1cbb-40a3-9a4d-c7d3722ab02a
# ╟─10000006-0000-4000-8000-000000000002
# ╟─1577f6f5-947b-49da-8fc5-e5b65f535310
# ╟─10000006-0000-4000-8000-000000000003
# ╟─10000007-0000-4000-8000-000000000001
# ╟─10000008-0000-4000-8000-000000000001
