### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 08010001-0000-4000-8000-000000000001
md"""
# QRM-Thorax Pure-Material Phantom on Photon-Counting CT

**A body-sized phantom with water, lipid, collagen and 5 mg/mL iodine rods in its heart, scanned
by the Siemens NAEOTOM Alpha at 140 kVp with four energy bins, turned into virtual
monoenergetic images (VMIs) by `BS.vmi_pipeline`, every rod checked against theory.**

This is notebook 07's phantom, reconstruction grid, VMI energies and ROIs on a photon-counting
scanner, run through the chain of the basis-spectral-denoising study: its denoiser,
reconstruction windows and spectral pair. Each rod is compared with the HU its material has in
theory and with the same chain applied to a noise-free scan.
"""

# ╔═╡ e234dc6a-7f7a-41da-abb7-a7e387e2a57c
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scanner", "Four Energy Bins"],
        [
            "QRM-Thorax, Four Pure-Material Rods",
            "Siemens NAEOTOM Alpha, 140 kVp",
            "One Beam, Counted in Four Ranges",
        ];
        limits = [(-170, 170, -130, 130), (-7.5, 7.5, -7.5, 7.5), (-0.8, 15.6, -2.4, 7.2)],
    )
    thorax!(axes[1]; voxel_mm = 0.4)

    gantry!(axes[2]; labels = ("X-Ray Tube",), object = false)
    # the thorax to scale inside the 50 cm scan field, its 25 cm radius drawn as 3.2
    thorax!(axes[2]; voxel_mm = 1.0, scale = 3.2 / 250)

    beam = BS.dose_source(scanner, alpha_protocol(174.0))
    bin_spectrum!(
        axes[3],
        beam.E,
        beam.phi_iso;
        thresholds = scanner.energy_thresholds,
        dark = true,
    )
    fig
end

# ╔═╡ 46a1725c-1bfa-4076-9b9f-b06b9cc0cd30
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 330))
    nodes = Dict(
        :phantom => (0.0, 2.0, "Phantom", "QRM-Thorax + 4 Rods"),
        :sim => (4.0, 2.0, "simulate!", "3 Draws, 1 Projection", COLORS.cyan),
        :basis => (8.0, 2.0, "spectral_basis", "Per-Ray Response", COLORS.cyan),
        :pair => (8.0, 0.2, "spectral_pair", "From the Calibration Draw", COLORS.cyan),
        :vmi => (12.0, 2.0, "vmi_pipeline", "Four Bins to VMIs", COLORS.cyan),
        :rois => (16.0, 2.0, "Rod HU", "vs Theory, vs Noise-Free", COLORS.green),
    )
    edges = [
        :phantom => :sim,
        :sim => :basis,
        :basis => :vmi,
        :sim => :pair,
        :pair => :vmi,
        :vmi => :rois,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-1.7, 17.7, -0.5, 2.7), w = 3.2, h = 1.2)
    fig
end

# ╔═╡ 68353060-bff1-4c28-a34a-eee23bbf36f6
md"""
| Stage | Matrix | Voxel | Extent |
|:--|:--|:--|:--|
| Phantom | **1600 × 1100 × 20** | **0.2 mm** isotropic | 320 × 220 × 4 mm |
| Reconstruction | 512 × 512 × **3** | **0.625 mm** isotropic | 32 cm field of view × 1.875 mm in z |
| Collimation | **5.0 mm** nominal | — | cone-guard rows added automatically |
| Scanner | Siemens NAEOTOM Alpha | 0.373 × 0.399 mm dexels (2 × 2 digitally binned) | 144 rows, 50 cm scan field |
| Protocol | PCCT, 4 bins | 140 kVp, 1200 views, 0.5 s rotation, 5 sub-views per view | the physical scan's 10.12 mGy CTDIvol |

!!! tip "When to reach for this notebook"
    Photon-counting VMI accuracy on a body-sized phantom and field of view, at the resolution
    a NAEOTOM Alpha delivers. Notebook 04 runs the same scanner and chain on the Gammex 472;
    notebook 07 scans this phantom with dual-kVp switching.
"""

# ╔═╡ 08010002-0000-4000-8000-000000000001
md"""
## Setup
"""

# ╔═╡ 08010003-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 08010003-0000-4000-8000-000000000005
using Markdown: @md_str, Markdown

# ╔═╡ 08010003-0000-4000-8000-000000000006
using Statistics: mean, std, quantile

# ╔═╡ 08010003-0000-4000-8000-000000000010
import BasisSimulator as BS

# ╔═╡ 08010003-0000-4000-8000-000000000011
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 08010003-0000-4000-8000-000000000012
import PlutoUI

# ╔═╡ 08010003-0000-4000-8000-000000000013
PlutoUI.TableOfContents()

# ╔═╡ 08010003-0000-4000-8000-000000000040
begin
    import GPUSelect
    # the backend array type, directly: MtlArray / CuArray / ROCArray
    AT = GPUSelect.Storage()
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 08010003-0000-4000-8000-000000000050
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ da96616f-0478-45c9-97e5-456589f784bc
md"""
## Figure Style

BasisSimulator's figure kit (colours, theme, flow charts, the gantry and the locator band),
shared by every example notebook, then this notebook's own drawings: the phantom, the heart
with its rods, and the spectrum split into bins.
"""

# ╔═╡ ade0946d-d027-40d6-880e-bb1988bbbc31
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

# ╔═╡ 93358bcb-f683-43e3-a75d-d74036ad199e
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

# ╔═╡ 3a4acf27-0117-4643-b26c-2e72520e861e
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

# ╔═╡ c10956e5-f3de-4782-8122-26edfd58b851
# The colour of each phantom label in the drawings (air, labels 1 and 7, stays transparent)
LABEL_COLORS = Dict(
    2 => Mke.RGBf(0.77, 0.84, 0.9),  # lung
    3 => Mke.RGBf(0.89, 0.73, 0.67), # soft tissue
    4 => Mke.RGBf(0.95, 0.91, 0.79), # cortical bone
    5 => Mke.RGBf(0.76, 0.47, 0.38), # marrow
    6 => COLORS.blue,                # the water rod in the lung
    8 => Mke.RGBf(0.89, 0.73, 0.67), # unused: soft tissue
    9 => COLORS.blue,                # water
    10 => COLORS.beam,               # lipid
    11 => COLORS.purple,             # collagen
    12 => COLORS.green,              # 5 mg/mL iodine
);

# ╔═╡ 2871c6e1-b1d7-4d8a-ab3b-763141cd92e6
# The four rods' colours, in the order of ROD_LABELS (water, lipid, collagen, iodine)
ROD_COLORS = Tuple(LABEL_COLORS[l] for l in 9:12);

# ╔═╡ 77f8e4df-f64e-40f1-902c-1c628442994c
# The four counting bins' colours, lowest energy first
BIN_COLORS = (COLORS.violet, COLORS.cyan, COLORS.green, COLORS.beam);

# ╔═╡ 9a5e2331-b1b5-4201-aaee-fb089974777e
begin
    """
    The QRM-Thorax mid-slice drawn on `ax` at `voxel_mm` (coarser is faster), centred at the
    origin with `scale` drawing units per millimetre: every label in its `LABEL_COLORS` colour,
    air transparent.
    """
    function thorax!(ax; voxel_mm = 1.0, scale = 1.0)
        n = round.(Int, (320, 220) ./ voxel_mm)
        slice = qrm_thorax_slice(; n, voxel_mm, inserts = HEART_RODS)
        clear = Mke.RGBAf(0, 0, 0, 0)
        rgba = [
            haskey(LABEL_COLORS, Int(l)) ? Mke.RGBAf(LABEL_COLORS[Int(l)], 1.0) : clear
            for l in slice
        ]
        half = n .* voxel_mm .* scale ./ 2
        Mke.image!(ax, (-half[1], half[1]), (-half[2], half[2]), rgba; interpolate = false)
        return ax
    end

    # where each rod's name sits: the direction away from the heart's centre, and the alignment
    ROD_NAME_SPOTS = Dict(
        9 => ((0.0, 1.0), (:center, :bottom)),
        10 => ((1.0, 0.0), (:left, :center)),
        11 => ((0.0, -1.0), (:center, :top)),
        12 => ((-1.0, 0.0), (:right, :center)),
    )

    """
    The cardiac insert and its four rods, in millimetres, each rod named and in its colour.
    `rois` are `(center, radius, label)` circles drawn on top: the ROIs a plot measures.
    """
    function heart_sketch!(ax; rois = ())
        h = QRM_GEOMETRY.heart
        Mke.poly!(
            ax,
            Mke.Circle(Mke.Point2f(h.center...), h.r);
            color = LABEL_COLORS[3],
            strokecolor = COLORS.rule,
            strokewidth = 1.5,
        )

        for (k, rod) in enumerate(QRM_GEOMETRY.heart_rods)
            Mke.poly!(
                ax,
                Mke.Circle(Mke.Point2f(rod.center...), rod.r);
                color = ROD_COLORS[k],
            )
            direction, align = ROD_NAME_SPOTS[rod.label]
            Mke.text!(
                ax,
                rod.center[1] + 10.5 * direction[1],
                rod.center[2] + 10.5 * direction[2];
                text = first(split(ROD_NAMES[k])),
                align,
                fontsize = 17,
                font = :bold,
                color = COLORS.ink,
            )
        end

        for (c, r, label) in rois
            Mke.lines!(
                ax,
                Mke.Circle(Mke.Point2f(c...), r);
                color = COLORS.red,
                linewidth = 3,
            )
            isempty(label) ||
                Mke.text!(
                    ax,
                    c[1] + 0.8 * r,
                    c[2] - 0.8 * r - 1.0;
                    text = label,
                    align = (:left, :top),
                    fontsize = 16,
                    font = :bold,
                    color = COLORS.red,
                )
        end
        Mke.limits!(ax, -66, 66, -42, 80)
        return ax
    end

    """
    The tube spectrum `E` (keV), `phi` drawn on a canvas: x = (E − 15) / 8.5, the continuum's
    peak at height 4.5 (the characteristic lines clipped at 6.4), its area split into the
    counting bins at `thresholds`, each in its `BIN_COLORS` colour.
    """
    function bin_spectrum!(ax, E, phi; thresholds, dark = false)
        ink = dark ? COLORS.night_ink : COLORS.ink
        muted = dark ? COLORS.night_muted : COLORS.muted
        xof(e) = (e - 15) / 8.5
        keep = E .≥ 15
        E, phi = E[keep], phi[keep]
        # the continuum's peak, away from tungsten's K lines (57–70 keV)
        continuum = maximum(phi[(E .< 56) .| (E .> 71)])
        y = min.(4.5 .* phi ./ continuum, 6.4)
        x = xof.(E)
        edges = vcat(thresholds, maximum(E))

        for k in eachindex(thresholds)
            inbin = edges[k] .≤ E .≤ edges[k + 1]
            Mke.band!(ax, x[inbin], zero(y[inbin]), y[inbin]; color = (BIN_COLORS[k], 0.85))
            Mke.text!(
                ax,
                xof((edges[k] + edges[k + 1]) / 2),
                -1.0;
                text = "Bin $(k)",
                align = (:center, :top),
                fontsize = 16,
                font = :bold,
                color = BIN_COLORS[k],
            )
        end

        Mke.lines!(ax, x, y; color = ink, linewidth = 2)
        Mke.lines!(ax, [0.0, maximum(x)], [0.0, 0.0]; color = muted, linewidth = 1.5)
        for t in vcat(thresholds, maximum(E))
            Mke.text!(
                ax,
                xof(t),
                -0.2;
                text = string(round(Int, t)),
                align = (:center, :top),
                fontsize = 15,
                color = muted,
            )
        end
        Mke.text!(
            ax,
            maximum(x),
            -1.0;
            text = "keV",
            align = (:right, :top),
            fontsize = 15,
            color = muted,
        )
        return ax
    end
end;

# ╔═╡ 08020001-0000-4000-8000-000000000001
md"""
## 1. The Phantom

The QRM-Thorax is built analytically, from 2-D shapes: a flat-fronted superellipse body, the
two lungs, the cardiac insert with the mediastinum above it, a water rod and an air rod in the
lungs, six ribs and the sternum, and the vertebra with its canal, laminae and spinous process.
Every dimension was measured on the labelled 1600 × 1100 (0.2 mm) mid-slice of the scanned
phantom, in CT display orientation (spine at the bottom). The slice is tiled in z to
**1600 × 1100 × 20** at **0.2 mm isotropic** (320 × 220 × 4 mm), finer than the demagnified dexel
pitch, so the forward projector samples a high-resolution object.

As in the reference slice, the cardiac insert is soft tissue (label 3, like the body) with four
15 mm pure-material rods bored into it where the reference slice has them
(`QRM_GEOMETRY.heart_rods`: 25 mm north, east, south and west of a point 20.7 mm above the slice
centre). `qrm_thorax_slice` draws labels 1–7 and the rods are painted last, as labels 9–12, so
label 8 is unused. Every material is a prebuilt `XA.Materials` entry.

| Label | Material | Label | Material |
|:--|:--|:--|:--|
| 1 | air | 7 | air rod |
| 2 | lung | 8 | (unused) |
| 3 | soft tissue (ICRU-44 muscle) | **9** | rod, north: water (`basis_water`) |
| 4 | cortical bone | **10** | rod, east: lipid (`basis_lipid`) |
| 5 | red marrow | **11** | rod, south: collagen (`basis_collagen`) |
| 6 | water rod (in the lung) | **12** | rod, west: 5 mg/mL iodine (`gammex_472_i5_0`) |

!!! info "Why pure end-members?"
    `XA.Materials.basis_fat` is ICRU-44 adipose tissue (≈ 83 % triolein, 17 % water and trace
    electrolytes), not a pure lipid. `XA.Materials.basis_lipid` (H/C/O at 0.92 g/cm³) and
    `basis_collagen` (H/C/N/O at 1.26 g/cm³) are mathematically pure end members, made for
    clean water/lipid/collagen decomposition checks.
"""

# ╔═╡ 08020001-0000-4000-8000-000000000010
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

# ╔═╡ 08020001-0000-4000-8000-000000000013
const QRM_NZ = 20; # 20 × 0.2 mm = 4 mm: a short z-invariant phantom

# ╔═╡ 08020001-0000-4000-8000-000000000014
const QRM_VOXEL_SIZE_CM = (0.02, 0.02, 0.02); # (x, y, z) cm: 0.2 mm isotropic

# ╔═╡ 08020001-0000-4000-8000-000000000026
# (center in mm, radius in mm, label) of each heart rod
HEART_RODS = [(rod.center, rod.r, rod.label) for rod in QRM_GEOMETRY.heart_rods];

# ╔═╡ 08020001-0000-4000-8000-000000000029
mask_3d = let
    slice = qrm_thorax_slice(; voxel_mm = QRM_VOXEL_SIZE_CM[1] * 10, inserts = HEART_RODS)
    repeat(slice; outer = (1, 1, QRM_NZ))
end;

# ╔═╡ 08020001-0000-4000-8000-000000000040
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

# ╔═╡ 08020001-0000-4000-8000-000000000050
phantom_cpu = BS.create_phantom_from_mask(
    Array{Int, 3}(mask_3d),
    materials_dict,
    QRM_VOXEL_SIZE_CM,
);

# ╔═╡ 08020001-0000-4000-8000-000000000051
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 08020001-0000-4000-8000-000000000060
QRM_LABEL_NAMES = Dict{Int, String}(
    1 => "1 air",
    2 => "2 lung",
    3 => "3 soft tissue",
    4 => "4 cortical bone",
    5 => "5 marrow",
    6 => "6 water rod",
    7 => "7 air rod",
    8 => "8 (unused)",
    9 => "9 water",
    10 => "10 lipid",
    11 => "11 collagen",
    12 => "12 iodine 5 mg/mL",
);

# ╔═╡ 08020001-0000-4000-8000-000000000061
let
    Mke.set_theme!(THEME)
    slice = mask_3d[:, :, QRM_NZ ÷ 2 + 1]
    nx, ny = size(slice)
    px_mm = QRM_VOXEL_SIZE_CM[1] * 10

    fig = Mke.Figure(size = (1400, 760))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The QRM-Thorax Mid-Slice",
        subtitle = "$(nx) × $(ny) Voxels of $(px_mm) mm, $(length(unique(slice))) Labels",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
    )
    thorax!(ax; voxel_mm = 0.4)
    Mke.text!(
        ax,
        QRM_GEOMETRY.lung_rods[2].center[1],
        QRM_GEOMETRY.lung_rods[2].center[2] - 11;
        text = "Air Rod",
        align = (:center, :top),
        fontsize = 16,
        color = COLORS.muted,
    )

    entries = [
        (2, "Lung"),
        (3, "Soft Tissue"),
        (4, "Cortical Bone"),
        (5, "Marrow"),
        (9, "Water (Both Rods)"),
        (10, "Lipid"),
        (11, "Collagen"),
        (12, "Iodine, 5 mg/mL"),
    ]
    Mke.Legend(
        fig[1, 2],
        [
            Mke.PolyElement(
                color = LABEL_COLORS[k],
                strokecolor = COLORS.rule,
                strokewidth = 1,
            )
            for (k, _) in entries
        ],
        [name for (_, name) in entries],
        "Material";
        titlefont = :bold,
        patchsize = (26, 20),
    )
    fig
end

# ╔═╡ 08020001-0000-4000-8000-000000000062
let
    slice = mask_3d[:, :, QRM_NZ ÷ 2 + 1]
    px_mm = QRM_VOXEL_SIZE_CM[1] * 10
    body = findall(!=(UInt8(1)), slice)
    i_lo, i_hi = extrema(c -> c[1], body)
    j_lo, j_hi = extrema(c -> c[2], body)
    w, h = i_hi - i_lo + 1, j_hi - j_lo + 1
    counts = [count(==(UInt8(k)), slice) for k in 1:12]
    area(k) = round(counts[k] * px_mm^2 / 100, digits = 2)
    rows = join(
        [
            "| $(QRM_LABEL_NAMES[k]) | $(counts[k]) | $(area(k)) |"
            for k in 1:12
            if counts[k] > 0
        ],
        "\n",
    )

    Markdown.parse(
        """
        The body measures $(round(w * px_mm / 10, digits = 1)) × $(round(h * px_mm / 10, digits = 1)) cm
        (the QRM-Thorax-small envelope without the fat ring, ≈ 30 × 20 cm).

        | Label | Voxels in the mid-slice | Area (cm²) |
        |-------|------:|------:|
        $(rows)
        """,
    )
end

# ╔═╡ 08030001-0000-4000-8000-000000000001
md"""
## 2. The Scanner: Siemens NAEOTOM Alpha

`NAEOTOM_ALPHA`, the documented constant, in its standard mode: native CdTe dexels at a
0.34 × 0.364 mm pitch summed digitally 2 × 2 after counting (1376 columns of 0.373 mm and 144
rows of 0.399 mm at the isocentre), an arc detector with the quarter-detector offset, a 50 cm
scan field behind the large-body bowtie, and a 1.6 mm CdTe sensor whose response (absorption,
charge sharing, electronic noise, the anti-scatter grid) is BasisMonteCarlo's Geant4 table. The
thresholds `T = [20, 35, 55, 70] keV` make four bins: **20–35, 35–55, 55–70 and above 70 keV**.

It is a realistic unit, defects included:

- each native dexel counts on its own (Poisson), shares photons with its neighbours and piles up
  at its own rate (27 ns dead time);
- the native dexels' gains spread by 2 % and their thresholds by 0.35 keV, a fixed pattern that
  would ring the image uncorrected;
- patient scatter, from the Geant4 library's kernels for the Alpha, is part of every simulation.

`simulate!` corrects all three in a scanner's order (section 5): the pile-up and the flat field
with calibrations from simulated scans, as in notebook 04, then the scatter from the recorded
counts alone. The bowtie makes the air response vary across the fan, so the spectral model is
per ray.
"""

# ╔═╡ 08030001-0000-4000-8000-000000000010
scanner = BS.NAEOTOM_ALPHA;

# ╔═╡ 87a264c8-e748-573c-84b3-99debe660f48
# Every keyword of `scanner`: the documented constant `NAEOTOM_ALPHA`, with any field this notebook changes marked
let s = scanner, constant = BS.NAEOTOM_ALPHA
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
        **`scanner` is `$(nameof(typeof(s)))`, from the documented constant `NAEOTOM_ALPHA`.** Its keywords
        and their values (units and sources: the constant's docstring, `?BS.NAEOTOM_ALPHA`, and the Scanners
        page). A variant is the same struct with keywords replaced,
        `$(nameof(typeof(s)))(BS.NAEOTOM_ALPHA; field = value)`.

        | Keyword | Value | |
        |---|---|---|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 08030002-0000-4000-8000-000000000001
md"""
## 3. The Protocol: 140 kVp at the Physical Scan's Dose

The protocol of the physical NAEOTOM Alpha Gammex scan: 140 kVp, 0.5 s rotation, 1200 views,
5 mm collimation, 0.9 mm titanium added, 174 mA and a reported CTDIvol of 10.12 mGy. A tube's
output per mA is its own and the simulator's source is generic, so the tube current is scaled by
one factor, `DOSE_SCALE`, that makes the simulated CTDIvol (BasisSimulator's Monte Carlo dose in
the 32 cm body phantom) equal the physical scan's.
"""

# ╔═╡ 08030002-0000-4000-8000-000000000011
"The NAEOTOM Alpha's 140 kVp axial protocol at tube current `mA`."
alpha_protocol(mA) = BS.CTProtocol(
    kVp = 140,
    mA = mA,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 08030002-0000-4000-8000-000000000012
# The tube-current scale that makes the simulated CTDIvol equal the physical scan's
DOSE_SCALE = let physical = (mA = 174.0, ctdi_vol_mGy = 10.12)
    p = alpha_protocol(physical.mA)
    simulated = BS.compute_dose(BS.dose_source(scanner, p), p).ctdi_vol_mGy
    physical.ctdi_vol_mGy / simulated
end

# ╔═╡ 08030002-0000-4000-8000-000000000010
protocol = alpha_protocol(174.0 * DOSE_SCALE);

# ╔═╡ 08030003-0000-4000-8000-000000000001
md"""
## 4. Physics and Reconstruction Options

A `PCCTScanner` selects the photon-counting path (per-bin sinograms, the Monte Carlo detector
response, the Geant4 scatter kernels, pile-up); the detector-model settings live on the scanner,
and `SimOptions` carries the physics common to both detector families. These are the switches of
the basis-spectral-denoising Alpha model: noise, scatter and the focal-spot blur on; the
scintillator effects (fill factor, optical crosstalk, the scintillator efficiency table, lag)
off, since the CdTe detector's efficiency, fill and charge sharing are in its Monte Carlo
response; the heel effect off.

<details><summary><strong>View integration: five sub-views per view</strong></summary>

A detector integrates while the gantry turns, so each view reads the transmitted intensity
averaged over the arc it sweeps: a blur of the object, largest far from the isocentre, that the
noise, counted once per view, does not share. `view_samples = 5` samples that arc with five
sub-views per view (the midpoint rule), and `view_arc = 1.0` because a photon-counting detector
reads for the whole view period.

</details>

**Three draws, one projection.** Everything before the noise is the same for every draw, so the
projection (sub-views included) is computed once, on the noise-free draw
(`keep_projection = true`), and the other draws reuse it (`projection`), bit-identically to
simulating them afresh:

| Draw | Noise | Seed | Used for |
|---|---|---:|---|
| `reference` | off | — | the noise-free expectation every result is compared with |
| `calibration` | on | 9234 | fixing the spectral pair, never measured |
| `measured` | on | 1234 | every result below |

The reconstruction is notebook 07's centred 3-slice grid; the workspace adds the symmetric
detector guard rows for full cone support.
"""

# ╔═╡ 08030003-0000-4000-8000-000000000010
"The Alpha model's physics switches for one draw."
draw_options(; seed, use_noise) = BS.SimOptions(
    seed = seed,
    use_noise = use_noise,
    projector = :dd_fast,
    use_focal_spot = true,
    use_heel_effect = false,
    # sub-views per view: the arc the gantry turns while a view is read
    view_samples = 5,
    view_arc = 1.0, # a photon-counting detector integrates for the whole view period
);

# ╔═╡ 08030003-0000-4000-8000-000000000011
# basis-spectral-denoising's seeds: draw r has seed 1234 + 1000 (r − 1); the calibration draw is draw 9
DRAWS = (
    reference = (seed = 1234, use_noise = false),
    calibration = (seed = 1234 + 1000 * (9 - 1), use_noise = true),
    measured = (seed = 1234, use_noise = true),
);

# ╔═╡ 08030003-0000-4000-8000-000000000020
# The saved recon grid of notebook 07: 512 × 512 at 0.625 mm isotropic, 3 slices.
recon_opts = BS.ReconOptions(matrix_size = (512, 512, 3), fov_cm = 32.0, z_cm = 0.1875);

# ╔═╡ 08030004-0000-4000-8000-000000000001
md"""
## 5. Simulate: `simulate!` and `spectral_basis`

`simulate!` runs the whole photon-counting detector model on the GPU: polychromatic forward
projection of every sub-view through the Monte Carlo detector response, the patient scatter (the
Geant4 kernels superposed on the phantom's own rays), the counts of every native dexel (Poisson,
neighbours covarying through the photons they share), its pile-up and the digital 2 × 2 sum.
It then corrects the counts in a scanner's order, with the workspace's calibration
(`calibrate(scanner, protocol, recon_opts, phantom)`, made once from simulated scans, never from
the model's own matrices):

| Step | What it recovers | Calibrated from |
|:--|:--|:--|
| 1. Pile-up | each ray's pile-up-free counts, from its recorded counts in every bin, its rate and its spectrum | scans of air and two water cylinders at rising tube currents |
| 2. Flat field | each binned dexel's gain in every bin, against the nominal detector's, as the patient hardens the beam | low-flux scans of air and of water slabs 5–50 cm thick |
| 3. Scatter | the scatter, estimated from the corrected counts, the patient taken as water | the counts themselves |

The scatter estimate is a nominal dexel's: subtracted before the other two corrections, it
would leave each dexel's gain error on its scatter, as rings.

`simulate!` returns the four corrected sinograms ``h_k = -\log(y_k / I_{0,k})``
(`pcct_sino.bins`, `[col, row, view]`), the air count of every ray in every bin (`I0_bins`,
`[col, row, bin]`), the dose report and, with `keep_projection = true`, the projection the next
draw reuses. `spectral_basis(ws; I0)` is read from the same workspace: the response the
simulation applied, so the decomposition inverts exactly what was simulated and needs no
calibration scan. It checks that the response sums to `I0` in every ray and bin. The model does
not depend on the noise, so one basis serves every draw.
"""

# ╔═╡ 08030004-0000-4000-8000-000000000010
acquisition = let
    projection = nothing
    model = nothing
    channels = Dict{Symbol, Vector{Array{Float32, 3}}}()
    seconds = Dict{Symbol, Float64}()

    # the scanner's calibration, as a scanner is calibrated: the pile-up from air and water scans
    # at rising currents; the flat field from low-flux scans of air and water slabs (each binned
    # dexel's gain per bin, and how it changes as the patient hardens the beam)
    calibration = BS.calibrate(
        scanner,
        protocol,
        recon_opts,
        phantom;
        sim_opts = draw_options(; DRAWS[:reference]...),
    )

    # the noise-free draw first: it projects
    for name in (:reference, :calibration, :measured)
        opts = draw_options(; DRAWS[name]...)
        t0 = time()
        ws = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom; calibration)
        try
            # the dose and the projection do not depend on the noise: both come from the first
            # draw; simulate! corrects in a scanner's order: pile-up, flat field, then the scatter
            sim = BS.simulate!(
                ws,
                phantom,
                protocol,
                opts;
                capture_raw_counts = false,
                report_dose = projection === nothing,
                keep_projection = projection === nothing,
                projection = projection,
            )
            projection === nothing && (projection = sim.projection)

            # the four corrected bins
            channels[name] = [Array(b) for b in sim.pcct_sino.bins]
            if model === nothing
                # [n_cols, n_rows, n_bins]
                I0 = Float32.(Array(sim.I0_bins))
                model = (
                    I0 = I0,
                    # ray-resolved spectral model
                    basis = BS.spectral_basis(ws; I0 = I0),
                    geom = ws.geom,
                    dose = sim.dose,
                )
            end
        finally
            BS.release_backend!(ws)
        end
        seconds[name] = time() - t0
    end
    (; channels, seconds, model...)
end;

# ╔═╡ 08030004-0000-4000-8000-000000000011
let
    a = acquisition
    nc, nr, _ = size(a.I0)
    c, r = (nc + 1) ÷ 2, (nr + 1) ÷ 2
    fmt(x) = string(round(x; sigdigits = 3))
    s(name) = string(round(a.seconds[name]; digits = 1), " s")
    Markdown.parse(
        """
        | Quantity | Value |
        |---|---|
        | sinogram per bin | $(nc) columns × $(nr) rows × $(size(first(a.channels[:measured]), 3)) views |
        | air counts per ray and view, centre ray (bins 1 to 4) | $(join(round.(Int, a.I0[c, r, :]), " / ")) |
        | spectral basis | $(size(a.basis.Φ, 3)) energies, ray-resolved = $(a.basis.ray_resolved) |
        | largest relative mismatch of Σ Φ and I0 | $(fmt(a.basis.I0_relerr)) |
        | tube current | $(round(protocol.mA; digits = 1)) mA (174 mA × DOSE_SCALE = $(round(DOSE_SCALE; digits = 3))) |
        | CTDIvol (32 cm body phantom) | $(round(a.dose.ctdi_vol_mGy; digits = 2)) mGy |
        | wall time: noise-free draw (projects) / calibration / measured | $(s(:reference)) / $(s(:calibration)) / $(s(:measured)) |
        """,
    )
end

# ╔═╡ 08030004-0000-4000-8000-000000000025
# The phantom labels resampled onto the reconstruction grid (nearest neighbour); every ROI
# below is built from these labels in recon-voxel coordinates.
phantom_in_recon = BS.resample_to_recon(
    phantom_cpu,
    acquisition.geom,
    recon_opts.matrix_size;
    method = :nearest,
);

# ╔═╡ 08030004-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    bins = acquisition.channels[:measured]
    mid_r = size(first(bins), 2) ÷ 2 + 1
    slices = [permutedims(b[:, mid_r, :], (2, 1)) for b in bins]
    window = Tuple(Float64.(quantile(vcat(vec.(slices)...), (0.01, 0.99))))
    T = scanner.energy_thresholds
    ranges = [
        k < length(T) ?
            "$(round(Int, T[k]))–$(round(Int, T[k + 1])) keV" :
            "Above $(round(Int, T[k])) keV"
        for k in eachindex(T)
    ]

    fig = Mke.Figure(size = (1500, 860))
    top = canvas(fig[1, 1:5], (-0.6, 15.4, -2.4, 7.2))
    top.aspect = nothing
    top.title = "The 140 kVp Beam, Counted in Four Bins"
    beam = BS.dose_source(scanner, protocol)
    bin_spectrum!(top, beam.E, beam.phi_iso; thresholds = T)

    for k in eachindex(slices)
        ax = Mke.Axis(
            fig[2, k];
            title = "Bin $(k)",
            subtitle = ranges[k],
            titlecolor = BIN_COLORS[k],
            xlabel = "View",
            ylabel = k == 1 ? "Detector Column" : "",
            yticklabelsvisible = k == 1,
        )
        Mke.heatmap!(ax, slices[k]; colormap = :viridis, colorrange = window)
    end
    Mke.Colorbar(
        fig[2, 5];
        colormap = :viridis,
        colorrange = window,
        label = "Line Integral, −ln(y / I₀)",
    )
    Mke.Label(
        fig[3, 1:5],
        "Central detector row of the measured draw, every view; one sinogram per bin.";
        fontsize = 16,
        color = COLORS.muted,
    )
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.3))
    fig
end

# ╔═╡ 08030000-0000-4000-8000-0000000000f1
md"""
## 6. The VMI Chain

The denoiser is generalized HYPR-LR in both domains, `BS.SpectralHYPR()`, the chain of
basis-spectral-denoising:

- **Projection domain:** within each detector row, each ray's split of its total count across
  the four bins is pooled over a 3 × 3 (column × view) window whose view neighbours are two views
  apart (`view_stride = 2`), so the odd and even views stay independent; the count dispersion of
  every bin is measured from the air rays.
- **Image domain:** the reconstructed basis pair is rewritten as the minimum-noise VMI (the
  composite), kept as reconstructed, and a complement whose noise is uncorrelated with it, pooled
  within its slice with likelihood weights on the composite over the window width (3 to 41
  pixels) with the least estimated risk. Every noise level comes from the difference of the odd-
  and even-view reconstructions, and no slices are averaged.

The decomposition is the K-channel maximum-likelihood estimator on all four bins with its
published controls; T-LBF is off; the view-direction antialias is on; ACNR acts on the
complement, before the image HYPR.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000012
HYPR_CHAIN = BS.SpectralHYPR();

# ╔═╡ 08030007-0000-4000-8000-000000000013
# The decomposition and reconstruction settings of the chain; ACNR is added on the measured draw
VMI_CHAIN = (
    method = :nchannel,
    controls = BS.NChannelControls(),
    use_tlbf = false,
    antialias = true,
);

# ╔═╡ 0803000a-0000-4000-8000-000000000010
pcct_vmi_energies = [50.0, 70.0, 100.0, 140.0];

# ╔═╡ 08030007-0000-4000-8000-000000000020
md"""
### The Reconstruction Windows, Rescaled to This Grid

`BS.PairFilter` gives the composite and its complement each their own FDK window (the water and
iodine images, whose noise is anti-correlated, must be filtered alike, or their noise stops
cancelling in the quiet VMIs). Each window is ``W(f) = \exp(-(f/f_c)^p)`` on the grid-Nyquist
axis, sampled at 11 knots as a `CustomFilter`. basis-spectral-denoising fitted the Alpha's pair
to the physical Gammex scan's MTF and NPS shape on 512² over 35 cm: composite ``f_c = 0.5``,
``p = 2``; complement ``f_c = 0.45``, ``p = 6``. This grid is 512² over 32 cm, whose finer pixels
represent higher frequencies, so each window is kept the same function of physical frequency by
rescaling its argument by the ratio `r` of the two grids' bandlimits (`BS.grid_bandlimit`), as
the study does for its other grids.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000021
ALPHA_WINDOWS = (composite = (fc = 0.5, p = 2.0), complement = (fc = 0.45, p = 6.0));

# ╔═╡ 08030007-0000-4000-8000-000000000022
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11))
    # the grid the windows were fitted on: the Alpha's clinical series, 512² over 35 cm
    fitted = BS.CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = 35.0,
        z_cm = 0.48,
        collimation_mm = protocol.collimation_mm,
    )
    r = BS.grid_bandlimit(acquisition.geom, recon_opts.matrix_size) /
        BS.grid_bandlimit(fitted, (512, 512, 12))
    window(q) = BS.CustomFilter(
        knots,
        Tuple(round(exp(-(x * r / q.fc)^q.p), digits = 5) for x in knots)
    )
    (
        filter = BS.PairFilter(
            window(ALPHA_WINDOWS.composite),
            window(ALPHA_WINDOWS.complement),
        ),
        r = r,
    )
end;

# ╔═╡ 08030007-0000-4000-8000-000000000023
md"""
### The Spectral Pair, Fixed from the Calibration Draw

Near its minimum the VMI noise hardly changes with energy, so the minimum-noise energy measured
on one acquisition is itself noise; it is a property of the scanner and protocol, measured once
on the calibration draw and fixed for the measured and the noise-free draws:
`pair_basis = (Estar, β)` from the calibration draw decomposed without denoising and
reconstructed with `SoftFilter` (the pair the windows act on), and `composite_energy` from the
same draw after the projection-domain HYPR (the composite ACNR and the image HYPR act on).
"""

# ╔═╡ 08030007-0000-4000-8000-000000000024
PAIR_BASIS = let ch = acquisition.channels[:calibration], I0 = acquisition.I0
    decompose(channels) = BS.vmi_pipeline(;
        channels,
        basis = acquisition.basis,
        geom = acquisition.geom,
        to_backend = to_gpu,
        matrix_size = recon_opts.matrix_size,
        vmi_energies = Tuple(pcct_vmi_energies),
        keep_sinograms = true,
        use_acnr = false,
        VMI_CHAIN...,
    ).sinograms
    p = HYPR_CHAIN.projection

    d_none = decompose(ch)
    d_proj = decompose(BS.hypr_lr(
        ch,
        I0;
        kernel = p.kernel,
        dispersion = BS.estimate_dispersion(ch, I0),
        view_stride = p.view_stride,
        to_backend = to_gpu,
    ))

    sp(d; kw...) = BS.spectral_pair(
        d.water,
        d.iodine,
        acquisition.geom,
        recon_opts.matrix_size;
        filter = BS.SoftFilter(),
        antialias = VMI_CHAIN.antialias,
        to_backend = to_gpu,
        kw...,
    )
    x = sp(d_none)
    c = sp(d_proj; basis = (Estar = x.Estar, β = x.β))
    (Estar = x.Estar, β = x.β, composite = c.Estar)
end

# ╔═╡ 08030007-0000-4000-8000-000000000001
md"""
### `vmi_pipeline`

One call, from the four corrected bins of the measured draw to the VMI stack, reconstructed on
the notebook's grid with every detector row kept. `keep_sinograms = true` also returns the
decomposed basis sinograms, shown below. The noise-free draw goes through the same call without
the denoiser and ACNR: the reference, reconstructed exactly as its noisy counterpart.
"""

# ╔═╡ 08030007-0000-4000-8000-000000000010
pcct_vmi = BS.vmi_pipeline(;
    channels = acquisition.channels[:measured],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(pcct_vmi_energies),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    keep_sinograms = true,
    VMI_CHAIN...,
);

# ╔═╡ 08030007-0000-4000-8000-000000000014
reference_vmi = BS.vmi_pipeline(;
    channels = acquisition.channels[:reference],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = Tuple(pcct_vmi_energies),
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 08030007-0000-4000-8000-000000000011
let
    q = pcct_vmi.quality
    d = pcct_vmi.settings.denoiser
    pct(x) = round(100x, digits = 3)
    Markdown.parse(
        """
        The decomposition solved $(q.n_rays) rays in $(round(pcct_vmi.elapsed_s, digits = 1)) s
        ($(round(q.outer_mean, digits = 1)) outer iterations on average); $(pct(q.frac_not_converged))% did not
        converge and $(pct(q.frac_bound_iodine))% / $(pct(q.frac_bound_water))% touched the iodine / water bounds.
        The projection HYPR measured dispersions (variance / mean of the counts on the air rays) of
        $(join(round.(d.dispersion, digits = 2), " / ")) for bins 1 to 4. From the calibration draw, the windows
        act on the pair at E* = $(round(Int, PAIR_BASIS.Estar)) keV (β = $(round(PAIR_BASIS.β, sigdigits = 3))), and the
        composite is fixed at $(round(Int, PAIR_BASIS.composite)) keV. The windows' bandlimit ratio on this grid is
        r = $(round(PAIR_FILTER.r, digits = 4)), and the image HYPR chose a $(d.image_estimates.window)-pixel window
        for the complement.
        """,
    )
end

# ╔═╡ 08030007-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1560))

    # the chain inside vmi_pipeline, and where the two rows of panels below come from
    nodes = Dict(
        :bins => (0.0, 0.0, "Four Bins", "Corrected Counts"),
        :hp => (3.4, 0.0, "Projection", "HYPR, 3 × 3", COLORS.cyan),
        :dec => (6.8, 0.0, "Decompose", "K = 4, Max. Likelihood", COLORS.cyan),
        :fdk => (10.2, 0.0, "FDK", "Pair Windows", COLORS.cyan),
        :acnr => (13.6, 0.0, "ACNR", "On the Complement", COLORS.cyan),
        :hi => (17.0, 0.0, "Image", "HYPR", COLORS.cyan),
        :vmi => (20.4, 0.0, "VMIs", "50 to 140 keV", COLORS.green),
    )
    edges = [
        :bins => :hp,
        :hp => :dec,
        :dec => :fdk,
        :fdk => :acnr,
        :acnr => :hi,
        :hi => :vmi,
    ]
    chain = flow_chart!(
        fig[1, 1:4],
        nodes,
        edges;
        limits = (-1.6, 22.0, -1.6, 0.9),
        w = 2.9,
        h = 1.2,
    )
    Mke.text!(
        chain,
        [6.8, 17.0],
        [-0.8, -0.8];
        text = ["Row 2: Basis Sinograms", "Row 3: Basis Maps"],
        align = (:center, :top),
        fontsize = 16,
        color = COLORS.muted,
    )

    # the decomposed sinograms, central detector row
    mid_r = size(pcct_vmi.sinograms.iodine, 2) ÷ 2 + 1
    sinograms = (
        ("Iodine Basis Sinogram", pcct_vmi.sinograms.iodine),
        ("Water Basis Sinogram", pcct_vmi.sinograms.water),
    )
    for (c, (title, sino)) in enumerate(sinograms)
        slice = permutedims(sino[:, mid_r, :], (2, 1))
        window = Tuple(Float64.(quantile(vec(slice), (0.01, 0.99))))
        ax = Mke.Axis(
            fig[2, 2c - 1];
            title,
            subtitle = "Central Detector Row",
            xlabel = "View",
            ylabel = "Detector Column",
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = window)
        Mke.Colorbar(fig[2, 2c]; colormap = :viridis, colorrange = window, label = "g/cm²")
    end

    # the basis maps after ACNR and the image HYPR, mid slice: iodine on a fixed window around
    # the 5 mg/mL rod (a percentile window of the whole slice sits below the rod, which covers
    # under 1 % of it); water on its percentiles
    mid = size(pcct_vmi.images.water, 3) ÷ 2 + 1
    water = pcct_vmi.images.water[:, :, mid]
    maps = (
        (
            "Iodine Basis Map",
            1000 .* pcct_vmi.images.iodine[:, :, mid],
            (-2.0, 8.0),
            "Iodine (mg/mL)",
        ),
        (
            "Water Basis Map",
            water,
            Tuple(Float64.(quantile(vec(water), (0.01, 0.99)))),
            "Water (g/mL)",
        ),
    )
    for (c, (title, slice, window, label)) in enumerate(maps)
        ax = Mke.Axis(
            fig[3, 2c - 1];
            title,
            subtitle = "Mid Slice, After the Image HYPR",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(ax, slice; colormap = :viridis, colorrange = window)
        Mke.hidedecorations!(ax)
        Mke.Colorbar(fig[3, 2c]; colormap = :viridis, colorrange = window, label)
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.12))
    fig
end

# ╔═╡ 08030008-0000-4000-8000-000000000001
md"""
## 7. The VMIs

Every VMI is synthesized from the same water and iodine pair:
``\mu(E) = c_\mathrm{water}\,(\mu/\rho)_\mathrm{water}(E) +
c_\mathrm{iodine}\,(\mu/\rho)_\mathrm{iodine}(E)``, in HU relative to water at the same energy.
The curves above the images are the two mass attenuation coefficients each VMI weighs; iodine's
falls much faster with energy, so the iodine rod brightens toward 50 keV.
"""

# ╔═╡ 0803000a-0000-4000-8000-000000000020
# The VMI stack keyed by energy, (512, 512, 3) HU volumes.
vmi_HU_final = Dict(
    Float64(E) => pcct_vmi.vmis[:, :, :, k]
    for (k, E) in enumerate(pcct_vmi.energies)
);

# ╔═╡ 0803000a-0000-4000-8000-000000000021
# The noise-free reference's VMI stack, keyed alike.
vmi_HU_reference = Dict(
    Float64(E) => reference_vmi.vmis[:, :, :, k]
    for (k, E) in enumerate(reference_vmi.energies)
);

# ╔═╡ 0803000a-0000-4000-8000-000000000040
let
    Mke.set_theme!(THEME)
    HU_window = (-200, 500)
    mid = size(pcct_vmi.vmis, 3) ÷ 2 + 1
    nx, ny = size(pcct_vmi.vmis, 1), size(pcct_vmi.vmis, 2)
    xs = ((1:nx) .- (nx + 1) / 2) .* RECON_PIXEL_MM
    ys = ((1:ny) .- (ny + 1) / 2) .* RECON_PIXEL_MM
    fig = Mke.Figure(size = (1500, 760))

    # the two basis materials' mass attenuation, with the VMI energies marked
    ax = Mke.Axis(
        fig[1, 1:4];
        title = "What Each VMI Weighs",
        subtitle = "Mass Attenuation of the Two Basis Materials",
        xlabel = "Photon Energy (keV)",
        ylabel = "μ/ρ (cm²/g)",
        yscale = log10,
        limits = (nothing, (0.05, 80.0)),
    )
    Es = 30.0:0.5:150.0
    water_μ = [BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E) for E in Es]
    iodine_μ = [BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E) for E in Es]
    Mke.lines!(ax, Es, water_μ; color = COLORS.blue, label = "Water")
    Mke.lines!(ax, Es, iodine_μ; color = COLORS.green, label = "Iodine")
    Mke.vlines!(
        ax,
        pcct_vmi_energies;
        color = COLORS.muted,
        linestyle = :dash,
        linewidth = 1.5,
    )
    Mke.text!(
        ax,
        pcct_vmi_energies .+ 1.5,
        fill(0.06, length(pcct_vmi_energies));
        text = ["$(round(Int, E)) keV" for E in pcct_vmi_energies],
        align = (:left, :bottom),
        fontsize = 16,
        color = COLORS.muted,
    )
    Mke.axislegend(ax; position = :rt)

    for (k, E) in enumerate(pcct_vmi_energies)
        img = Mke.Axis(
            fig[2, k];
            title = "$(round(Int, E)) keV VMI",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(
            img,
            xs,
            ys,
            vmi_HU_final[E][:, :, mid];
            colormap = :grays,
            colorrange = HU_window,
        )
        # the body, cropped from the 32 cm field of view
        Mke.limits!(img, -160, 160, -112, 112)
        Mke.hidedecorations!(img)
    end
    Mke.Colorbar(fig[2, 5]; colormap = :grays, colorrange = HU_window, label = "HU")
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.5))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_grid.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000001
md"""
## 8. Accuracy and Noise in the Rods

Per-rod measured vs theoretical HU at 50 / 70 / 100 / 140 keV, the water rod's mean HU, and the
noise in the heart.

- **Measured HU:** the mean over a circular ROI at each rod's centroid, over every slice.
- **Theoretical HU:** ``1000\,(\mu_\mathrm{rod}(E) - \mu_\mathrm{water}(E)) / \mu_\mathrm{water}(E)``
  from `BS.compute_μ_at_energy(material, E)`; pure physics, no fitting.
- **Noise-free HU:** the same ROI on the noise-free reference, the same decomposition and
  reconstruction without noise (and so without the denoiser). Its error against theory is the
  bias of the iodine–water basis model itself; the measured draw's departure from it is what the
  noise and the denoising add.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000010
const ROD_LABELS = (UInt8(9), UInt8(10), UInt8(11), UInt8(12));

# ╔═╡ 080b0001-0000-4000-8000-000000000011
const ROD_NAMES = ("Water", "Lipid", "Collagen", "Iodine 5 mg/mL");

# ╔═╡ 080b0001-0000-4000-8000-000000000012
const ROD_MATERIALS = (
    BS.XA.Materials.basis_water,
    BS.XA.Materials.basis_lipid,
    BS.XA.Materials.basis_collagen,
    BS.XA.Materials.gammex_472_i5_0,
);

# ╔═╡ 080b0001-0000-4000-8000-000000000013
const ROD_ROI_RADIUS_PX = 8; # 5 mm at 0.625 mm/voxel, inside the 7.5 mm rods

# ╔═╡ 080b0001-0000-4000-8000-000000000014
const HEART_NOISE_ROI_RADIUS_PX = 12; # 7.5 mm: fits between the four rods

# ╔═╡ 13f469c5-7fa6-4079-92a4-2ffd71f6dd6d
# The reconstruction grid's pixel, mm
RECON_PIXEL_MM = 10 * recon_opts.fov_cm / recon_opts.matrix_size[1];

# ╔═╡ 080b0001-0000-4000-8000-000000000015
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
            return (sum(c -> c[1], idx) / length(idx), sum(c -> c[2], idx) / length(idx))
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

# ╔═╡ 080b0001-0000-4000-8000-000000000000
md"""
### Where the ROIs Sit

Before trusting any ROI built on `phantom_in_recon`: the resampled phantom labels laid over the
70 keV VMI on the same grid (rod, heart and bone edges should sit on the image's edges), and
every ROI on the heart.
"""

# ╔═╡ 080b0001-0000-4000-8000-00000000000a
let
    Mke.set_theme!(THEME)
    z = size(vmi_HU_final[70.0], 3) ÷ 2 + 1
    hu = vmi_HU_final[70.0][:, :, z]
    labels = phantom_in_recon[:, :, clamp(z, 1, size(phantom_in_recon, 3))]
    nx, ny = size(hu)
    xs = ((1:nx) .- (nx + 1) / 2) .* RECON_PIXEL_MM
    ys = ((1:ny) .- (ny + 1) / 2) .* RECON_PIXEL_MM
    to_mm(c) = (
        (c[1] - (nx + 1) / 2) * RECON_PIXEL_MM,
        (c[2] - (ny + 1) / 2) * RECON_PIXEL_MM,
    )

    # label 0 is outside the phantom grid (the 32 cm field of view is taller than the phantom)
    clear = Mke.RGBAf(0, 0, 0, 0)
    overlay = [
        haskey(LABEL_COLORS, Int(l)) ? Mke.RGBAf(LABEL_COLORS[Int(l)], 0.5) : clear
        for l in labels
    ]
    n_labels = count(>(0), unique(labels))

    fig = Mke.Figure(size = (1500, 720))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Phantom Labels on the 70 keV VMI",
        subtitle = "$(n_labels) Labels Resampled to the Reconstruction Grid (Nearest)",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
    )
    Mke.heatmap!(ax, xs, ys, hu; colormap = :grays, colorrange = (-200, 500))
    half = RECON_PIXEL_MM / 2
    Mke.image!(
        ax,
        (xs[1] - half, xs[end] + half),
        (ys[1] - half, ys[end] + half),
        overlay;
        interpolate = false,
    )
    Mke.limits!(ax, -165, 165, -118, 118)

    hc = to_mm(rod_rois.heart_center)
    zoom = Mke.Axis(
        fig[1, 2];
        title = "Every ROI, on the Heart",
        subtitle = "Rod ROIs $(ROD_ROI_RADIUS_PX * RECON_PIXEL_MM) mm, Noise ROI (Red) $(HEART_NOISE_ROI_RADIUS_PX * RECON_PIXEL_MM) mm in Radius",
        xlabel = "x (mm)",
        aspect = Mke.DataAspect(),
    )
    Mke.heatmap!(zoom, xs, ys, hu; colormap = :grays, colorrange = (-200, 500))
    for (k, lab) in enumerate(ROD_LABELS)
        c = to_mm(rod_rois.centers[lab])
        Mke.lines!(
            zoom,
            Mke.Circle(Mke.Point2f(c...), ROD_ROI_RADIUS_PX * RECON_PIXEL_MM);
            color = ROD_COLORS[k],
            linewidth = 3.5,
        )
        direction, align = ROD_NAME_SPOTS[Int(lab)]
        Mke.text!(
            zoom,
            c[1] + 10.5 * direction[1],
            c[2] + 10.5 * direction[2];
            text = first(split(ROD_NAMES[k])),
            align,
            fontsize = 18,
            font = :bold,
            color = :white,
            strokecolor = :black,
            strokewidth = 1,
        )
    end
    Mke.lines!(
        zoom,
        Mke.Circle(Mke.Point2f(hc...), HEART_NOISE_ROI_RADIUS_PX * RECON_PIXEL_MM);
        color = COLORS.red,
        linewidth = 3.5,
    )
    Mke.limits!(zoom, hc[1] - 62, hc[1] + 62, hc[2] - 62, hc[2] + 62)
    Mke.Colorbar(fig[1, 3]; colormap = :grays, colorrange = (-200, 500), label = "HU")
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000002
md"""
### The Water Rod

The water rod's mean HU at every VMI energy, measured (bars) and on the noise-free reference
(diamonds). Water is the reference of the HU scale, so the bars should sit at 0 HU: a constant
offset would be a basis-decomposition bias, a trend with energy a spectral-model error.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000004
water_rod_hu = [
    mean(roi_values(vmi_HU_final[E], rod_rois.rods[UInt8(9)]))
    for E in pcct_vmi_energies
];

# ╔═╡ 080b0001-0000-4000-8000-000000000005
water_rod_hu_reference = [
    mean(roi_values(vmi_HU_reference[E], rod_rois.rods[UInt8(9)]))
    for E in pcct_vmi_energies
];

# ╔═╡ 080b0001-0000-4000-8000-000000000003
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))
    sketch = canvas(fig[1, 1], (-66, 66, -42, 80))
    sketch.title = "The Water-Rod ROI"
    water_rod = QRM_GEOMETRY.heart_rods[1]
    heart_sketch!(
        sketch;
        rois = [(water_rod.center, ROD_ROI_RADIUS_PX * RECON_PIXEL_MM, "ROI")],
    )

    n_E = length(pcct_vmi_energies)
    ax = Mke.Axis(
        fig[1, 2];
        title = "Water Rod, Mean HU",
        subtitle = "Theory: 0 HU at Every Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Mean HU",
        xticks = (1:n_E, ["$(round(Int, E))" for E in pcct_vmi_energies]),
    )
    Mke.barplot!(ax, 1:n_E, water_rod_hu; color = (COLORS.blue, 0.75), label = "Measured")
    Mke.scatter!(
        ax,
        1:n_E,
        water_rod_hu_reference;
        color = COLORS.ink,
        marker = :diamond,
        markersize = 18,
        label = "Noise-Free Reference",
    )
    Mke.hlines!(ax, [0.0]; color = COLORS.muted, linewidth = 1.5, linestyle = :dash)
    for (k, h) in pairs(water_rod_hu)
        Mke.text!(
            ax,
            k,
            h;
            text = "$(round(h, digits = 1)) HU",
            align = (:center, h ≥ 0 ? :bottom : :top),
            fontsize = 17,
            offset = (0, h ≥ 0 ? 12 : -12),
        )
    end
    Mke.axislegend(ax; position = :rb)
    y_max = max(15.0, 1.25 * maximum(abs, vcat(water_rod_hu, water_rod_hu_reference)))
    Mke.ylims!(ax, -y_max, y_max)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_water_roi.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 080d0001-0000-4000-8000-000000000001
md"""
### Noise in the Heart

The standard deviation of HU in a 7.5 mm ROI in the soft-tissue heart, at the centre of the four
rods, at every VMI energy: how the chain carries noise from the basis pair into each VMI. It is
measured on the difference of the measured VMI and the noise-free reference, in which any
structure in the ROI (streaks, cupping) cancels; the labels give the measured VMI's mean HU.
"""

# ╔═╡ 080d0001-0000-4000-8000-000000000020
vmi_noise_by_keV = Dict(
    E =>
        let v = roi_values(vmi_HU_final[E], rod_rois.heart),
            d = roi_values(vmi_HU_final[E] .- vmi_HU_reference[E], rod_rois.heart)
            (mean = mean(v), std = std(v), std_vs_reference = std(d), n = length(v))
        end
    for E in pcct_vmi_energies
);

# ╔═╡ 080d0001-0000-4000-8000-000000000015
# The basis-pair covariance in the heart ROI, which sets the noise of every VMI:
#   σ_HU(E)² = 10⁶ V_w + α(E)² V_i + 2·10³ α(E) C_iw,  α(E) = (μ/ρ)_I(E) / (μ/ρ)_w(E),
# with water in g/mL and iodine in mg/mL; lowest at α* = -10³ C_iw / V_i. Anti-correlated
# basis noise (ρ < 0) is what ACNR removes.
basis_covariance = let
    w = roi_values(pcct_vmi.images.water, rod_rois.heart)
    i = roi_values(pcct_vmi.images.iodine, rod_rois.heart) .* 1000.0 # mg/mL
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

# ╔═╡ 080d0001-0000-4000-8000-000000000016
let c = basis_covariance
    inside = c.α_range[1] ≤ c.α_star ≤ c.α_range[2]
    gap = maximum(
        abs(vmi_noise_by_keV[E].std - vmi_noise_by_keV[E].std_vs_reference)
        for E in pcct_vmi_energies
    )
    toward = c.α_star < c.α_range[1] ? "high" : "low"
    location = inside ?
        ", reached at about $(Int(c.E_star)) keV." :
        ", outside the 40–140 keV range, so the noise falls monotonically toward $(toward) energy."
    Markdown.parse(
        """
        The standard deviation of the measured VMI itself differs from that of its difference from the
        noise-free reference by at most $(round(gap, digits = 2)) HU at any energy: the ROI holds noise only.

        In the heart ROI the water basis has σ = $(round(1000 * c.σ_water, digits = 1)) mg/mL, the iodine basis
        σ = $(round(c.σ_iodine, digits = 2)) mg/mL, and their correlation is ρ = $(round(c.ρ, digits = 2)).
        The noise-optimal mixing ratio is α* = $(round(c.α_star, digits = 1)) HU per mg/mL$(location)
        """,
    )
end

# ╔═╡ 080d0001-0000-4000-8000-000000000030
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))
    sketch = canvas(fig[1, 1], (-66, 66, -42, 80))
    sketch.title = "The Noise ROI"
    rods = QRM_GEOMETRY.heart_rods
    centre = (mean(rod.center[1] for rod in rods), mean(rod.center[2] for rod in rods))
    heart_sketch!(
        sketch;
        rois = [(centre, HEART_NOISE_ROI_RADIUS_PX * RECON_PIXEL_MM, "ROI")],
    )

    Es = sort(collect(keys(vmi_noise_by_keV)))
    σd = [vmi_noise_by_keV[E].std_vs_reference for E in Es]
    μs = [vmi_noise_by_keV[E].mean for E in Es]
    ax = Mke.Axis(
        fig[1, 2];
        title = "Noise in the Heart vs Energy",
        subtitle = "σ of Measured − Noise-Free, $(length(rod_rois.heart)) Pixels × $(size(vmi_HU_final[70.0], 3)) Slices",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        xticks = Es,
    )
    Mke.scatterlines!(ax, Es, σd; color = COLORS.red, markersize = 18, linewidth = 3)
    for (E, σ, μ) in zip(Es, σd, μs)
        # the noisiest point is labelled above, the rest below, clear of the line
        up = σ == maximum(σd)
        Mke.text!(
            ax,
            E,
            σ;
            text = "σ $(round(σ; digits = 1)) HU\nMean $(round(μ; digits = 1)) HU",
            align = (:center, up ? :bottom : :top),
            fontsize = 16,
            offset = (0, up ? 14 : -14),
        )
    end
    Mke.ylims!(ax, 0, maximum(σd) * 1.5)
    Mke.xlims!(ax, first(Es) - 12, last(Es) + 12)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_noise_vs_energy.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000008
md"""
### Basis Values in the Rods

The mean basis pair in each rod ROI. The water rod should read ``c_\mathrm{water} ≈ 1`` g/mL and
``c_\mathrm{iodine} ≈ 0``; the iodine rod carries its 5 mg/mL on a water-equivalent base. Lipid
and collagen are not made of water and iodine, so the chain represents them as the water/iodine
mixture that attenuates the same way (lipid with a negative iodine value).
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000009
let
    function row(lab, name, mat)
        cw = mean(roi_values(pcct_vmi.images.water, rod_rois.rods[lab]))
        ci = 1000 * mean(roi_values(pcct_vmi.images.iodine, rod_rois.rods[lab]))
        ρ = round(BS.XA.val(mat.density), digits = 3)
        return "| $(name) | $(ρ) | $(round(cw, digits = 3)) | $(round(ci, digits = 2)) |"
    end
    rows = [row(rod...) for rod in zip(ROD_LABELS, ROD_NAMES, ROD_MATERIALS)]

    Markdown.parse(
        """
        | Rod | Density (g/cm³) | Water (g/mL) | Iodine (mg/mL) |
        |-----|------:|------:|------:|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 080b0001-0000-4000-8000-000000000020
rod_data = let
    μ_water(E) = BS.compute_μ_at_energy(BS.XA.Materials.water, E)
    theoretical_hu(mat, E) = 1000.0 *
        (BS.compute_μ_at_energy(mat, E) - μ_water(E)) /
        μ_water(E)
    meas = [
        mean(roi_values(vmi_HU_final[E], rod_rois.rods[lab]))
        for lab in ROD_LABELS, E in pcct_vmi_energies
    ]
    noise_free = [
        mean(roi_values(vmi_HU_reference[E], rod_rois.rods[lab]))
        for lab in ROD_LABELS, E in pcct_vmi_energies
    ]
    theo = [theoretical_hu(mat, E) for mat in ROD_MATERIALS, E in pcct_vmi_energies]
    rmse = [sqrt(mean(abs2, meas[i, :] .- theo[i, :])) for i in eachindex(ROD_LABELS)]
    rmse_noise_free = [
        sqrt(mean(abs2, noise_free[i, :] .- theo[i, :]))
        for i in eachindex(ROD_LABELS)
    ]
    (
        labels = ROD_LABELS,
        names = ROD_NAMES,
        measured = meas,
        noise_free = noise_free,
        theoretical = theo,
        rmse = rmse,
        rmse_noise_free = rmse_noise_free,
    )
end;

# ╔═╡ 080b0001-0000-4000-8000-000000000030
md"""
### Measured vs Theoretical HU

Every rod at every energy: measured (solid lines with circles), theory (dashed) and the
noise-free reference (open diamonds), each rod in its colour.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000031
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 600))
    sketch = canvas(fig[1, 1], (-66, 66, -42, 80))
    sketch.title = "The Four Rod ROIs"
    rod_roi = ROD_ROI_RADIUS_PX * RECON_PIXEL_MM
    heart_sketch!(
        sketch;
        rois = [(rod.center, rod_roi, "") for rod in QRM_GEOMETRY.heart_rods],
    )

    ax = Mke.Axis(
        fig[1, 2];
        title = "Pure-Material Rods, Measured vs Theory",
        subtitle = "50, 70, 100 and 140 keV VMIs",
        xlabel = "VMI Energy (keV)",
        ylabel = "Mean HU",
        xticks = pcct_vmi_energies,
    )
    for i in eachindex(rod_data.names)
        c = ROD_COLORS[i]
        Mke.scatterlines!(
            ax,
            pcct_vmi_energies,
            rod_data.measured[i, :];
            color = c,
            linewidth = 3,
            markersize = 11,
        )
        Mke.lines!(
            ax,
            pcct_vmi_energies,
            rod_data.theoretical[i, :];
            color = c,
            linewidth = 1.8,
            linestyle = :dash,
        )
        Mke.scatter!(
            ax,
            pcct_vmi_energies,
            rod_data.noise_free[i, :];
            color = (:white, 0.0),
            marker = :diamond,
            markersize = 18,
            strokecolor = c,
            strokewidth = 2,
        )
    end

    styles = [
        Mke.MarkerElement(color = COLORS.ink, marker = :circle, markersize = 11),
        Mke.LineElement(color = COLORS.ink, linewidth = 1.8, linestyle = :dash),
        Mke.MarkerElement(
            color = (:white, 0.0),
            marker = :diamond,
            markersize = 16,
            strokecolor = COLORS.ink,
            strokewidth = 2,
        ),
    ]
    rods = [Mke.LineElement(color = c, linewidth = 3) for c in ROD_COLORS]
    rod_labels = [
        "$(rod_data.names[i]), RMSE $(round(rod_data.rmse[i], digits = 1)) HU"
        for i in eachindex(rod_data.names)
    ]
    Mke.Legend(
        fig[1, 3],
        [styles, rods],
        [["Measured", "Theory", "Noise-Free"], rod_labels],
        ["Marker", "Rod"];
        titlefont = :bold,
    )
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.27))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "qrm_thorax_pcct_vmi_vs_theoretical.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 080b0001-0000-4000-8000-000000000032
let
    f(x) = round(x, digits = 1)
    header = "| Rod | " *
        join(["$(Int(E)) keV" for E in pcct_vmi_energies], " | ") *
        " | RMSE (noise-free) |"
    rule = "|---|" * repeat("---:|", length(pcct_vmi_energies) + 1)
    function row(i)
        cells = [
            "$(f(rod_data.measured[i, j])) ($(f(rod_data.noise_free[i, j]))) / $(f(rod_data.theoretical[i, j]))"
            for j in eachindex(pcct_vmi_energies)
        ]
        return "| $(rod_data.names[i]) | " *
            join(cells, " | ") *
            " | $(f(rod_data.rmse[i])) ($(f(rod_data.rmse_noise_free[i]))) |"
    end
    rows = [row(i) for i in eachindex(rod_data.names)]

    Markdown.parse(
        "Measured (noise-free) / theoretical HU per rod and energy:\n\n" *
            join([header, rule, rows...], "\n"),
    )
end

# ╔═╡ 5ed4d276-afc1-4044-a875-b85e915a7685
md"""
## 9. Verification

The notebook checks itself: every row must pass for the page to publish.
"""

# ╔═╡ 080b0001-0000-4000-8000-000000000090
verification = let
    vmi_finite = all(isfinite, pcct_vmi.vmis) && all(isfinite, reference_vmi.vmis)
    checks = NamedTuple[]
    addcheck(name, value, pass) = push!(checks, (; name, value, pass))

    addcheck(
        "all channels enter the decomposition",
        acquisition.basis.n_channels,
        acquisition.basis.n_channels == 4,
    )
    addcheck(
        "Σ response matches I0 (max relative error)",
        round(acquisition.basis.I0_relerr, sigdigits = 2),
        acquisition.basis.I0_relerr < 5.0e-5,
    )
    addcheck(
        "VMI energies",
        pcct_vmi_energies,
        pcct_vmi_energies == [50.0, 70.0, 100.0, 140.0],
    )
    addcheck(
        "simulated CTDIvol equals the physical scan's 10.12 mGy",
        round(acquisition.dose.ctdi_vol_mGy, digits = 2),
        abs(acquisition.dose.ctdi_vol_mGy - 10.12) < 0.05,
    )
    addcheck("VMI values finite (measured and reference)", vmi_finite, vmi_finite)
    addcheck(
        "water rod within ±10 HU at every energy",
        round.(water_rod_hu, digits = 1),
        all(abs.(water_rod_hu) .≤ 10),
    )
    addcheck(
        "noise-free water rod within ±10 HU at every energy",
        round.(water_rod_hu_reference, digits = 1),
        all(abs.(water_rod_hu_reference) .≤ 10),
    )
    (; checks, passed = count(c -> c.pass, checks))
end;

# ╔═╡ ff4a7010-f828-4580-bfda-903fd3f02269
let v = verification
    status(c) = c.pass ? "✅ pass" : "❌ CHECK (outside bounds)"
    rows = join(["| $(c.name) | $(c.value) | $(status(c)) |" for c in v.checks], "\n")
    Markdown.parse(
        """
        **$(v.passed) of $(length(v.checks)) checks pass.**

        | Check | Value | Status |
        |---|---:|:---|
        $(rows)
        """,
    )
end

# ╔═╡ 080c0001-0000-4000-8000-000000000001
let
    r = rod_data
    fmt(v) = join(round.(v, digits = 1), " / ")
    σ = [vmi_noise_by_keV[E].std_vs_reference for E in pcct_vmi_energies]
    rmse(i) = "$(round(r.rmse[i], digits = 1)) ($(round(r.rmse_noise_free[i], digits = 1)))"
    rmse_rows = join(["| $(r.names[i]) | $(rmse(i)) |" for i in eachindex(r.names)], "\n")
    Markdown.parse(
        """
        ## Summary

        The QRM-Thorax with its four pure-material rods, scanned by the NAEOTOM Alpha at 140 kVp
        and $(round(acquisition.dose.ctdi_vol_mGy, digits = 2)) mGy CTDIvol, four bins through
        `BS.vmi_pipeline` with SpectralHYPR and the study's pair windows; $(verification.passed) of
        $(length(verification.checks)) checks pass. At 50 / 70 / 100 / 140 keV:

        | | HU |
        |:--|:--|
        | Water rod, measured | $(fmt(water_rod_hu)) |
        | Water rod, noise-free (theory: 0) | $(fmt(water_rod_hu_reference)) |
        | Noise in the heart, σ of measured − noise-free | $(fmt(σ)) |

        | Rod | RMSE vs theory, HU: measured (noise-free) |
        |:--|--:|
        $(rmse_rows)

        Notebook 07 runs the same phantom, rods, grid, energies and ROIs through a dual-kVp
        scanner, so the two notebooks compare directly.
        """,
    )
end

# ╔═╡ Cell order:
# ╟─08010001-0000-4000-8000-000000000001
# ╟─e234dc6a-7f7a-41da-abb7-a7e387e2a57c
# ╟─46a1725c-1bfa-4076-9b9f-b06b9cc0cd30
# ╟─68353060-bff1-4c28-a34a-eee23bbf36f6
# ╟─08010002-0000-4000-8000-000000000001
# ╠═08010003-0000-4000-8000-000000000001
# ╠═08010003-0000-4000-8000-000000000005
# ╠═08010003-0000-4000-8000-000000000006
# ╠═08010003-0000-4000-8000-000000000010
# ╠═08010003-0000-4000-8000-000000000011
# ╠═08010003-0000-4000-8000-000000000012
# ╠═08010003-0000-4000-8000-000000000013
# ╠═08010003-0000-4000-8000-000000000040
# ╟─08010003-0000-4000-8000-000000000050
# ╟─da96616f-0478-45c9-97e5-456589f784bc
# ╠═ade0946d-d027-40d6-880e-bb1988bbbc31
# ╠═93358bcb-f683-43e3-a75d-d74036ad199e
# ╠═3a4acf27-0117-4643-b26c-2e72520e861e
# ╠═c10956e5-f3de-4782-8122-26edfd58b851
# ╠═2871c6e1-b1d7-4d8a-ab3b-763141cd92e6
# ╠═77f8e4df-f64e-40f1-902c-1c628442994c
# ╠═9a5e2331-b1b5-4201-aaee-fb089974777e
# ╟─08020001-0000-4000-8000-000000000001
# ╠═08020001-0000-4000-8000-000000000010
# ╠═08020001-0000-4000-8000-000000000013
# ╠═08020001-0000-4000-8000-000000000014
# ╠═08020001-0000-4000-8000-000000000026
# ╠═08020001-0000-4000-8000-000000000029
# ╠═08020001-0000-4000-8000-000000000040
# ╠═08020001-0000-4000-8000-000000000050
# ╠═08020001-0000-4000-8000-000000000051
# ╠═08020001-0000-4000-8000-000000000060
# ╟─08020001-0000-4000-8000-000000000061
# ╟─08020001-0000-4000-8000-000000000062
# ╟─08030001-0000-4000-8000-000000000001
# ╠═08030001-0000-4000-8000-000000000010
# ╟─87a264c8-e748-573c-84b3-99debe660f48
# ╟─08030002-0000-4000-8000-000000000001
# ╠═08030002-0000-4000-8000-000000000011
# ╠═08030002-0000-4000-8000-000000000012
# ╠═08030002-0000-4000-8000-000000000010
# ╟─08030003-0000-4000-8000-000000000001
# ╠═08030003-0000-4000-8000-000000000010
# ╠═08030003-0000-4000-8000-000000000011
# ╠═08030003-0000-4000-8000-000000000020
# ╟─08030004-0000-4000-8000-000000000001
# ╠═08030004-0000-4000-8000-000000000010
# ╟─08030004-0000-4000-8000-000000000011
# ╠═08030004-0000-4000-8000-000000000025
# ╟─08030004-0000-4000-8000-000000000040
# ╟─08030000-0000-4000-8000-0000000000f1
# ╠═08030007-0000-4000-8000-000000000012
# ╠═08030007-0000-4000-8000-000000000013
# ╠═0803000a-0000-4000-8000-000000000010
# ╟─08030007-0000-4000-8000-000000000020
# ╠═08030007-0000-4000-8000-000000000021
# ╠═08030007-0000-4000-8000-000000000022
# ╟─08030007-0000-4000-8000-000000000023
# ╠═08030007-0000-4000-8000-000000000024
# ╟─08030007-0000-4000-8000-000000000001
# ╠═08030007-0000-4000-8000-000000000010
# ╠═08030007-0000-4000-8000-000000000014
# ╟─08030007-0000-4000-8000-000000000011
# ╟─08030007-0000-4000-8000-000000000040
# ╟─08030008-0000-4000-8000-000000000001
# ╠═0803000a-0000-4000-8000-000000000020
# ╠═0803000a-0000-4000-8000-000000000021
# ╟─0803000a-0000-4000-8000-000000000040
# ╟─080b0001-0000-4000-8000-000000000001
# ╠═080b0001-0000-4000-8000-000000000010
# ╠═080b0001-0000-4000-8000-000000000011
# ╠═080b0001-0000-4000-8000-000000000012
# ╠═080b0001-0000-4000-8000-000000000013
# ╠═080b0001-0000-4000-8000-000000000014
# ╠═13f469c5-7fa6-4079-92a4-2ffd71f6dd6d
# ╠═080b0001-0000-4000-8000-000000000015
# ╟─080b0001-0000-4000-8000-000000000000
# ╟─080b0001-0000-4000-8000-00000000000a
# ╟─080b0001-0000-4000-8000-000000000002
# ╠═080b0001-0000-4000-8000-000000000004
# ╠═080b0001-0000-4000-8000-000000000005
# ╟─080b0001-0000-4000-8000-000000000003
# ╟─080d0001-0000-4000-8000-000000000001
# ╠═080d0001-0000-4000-8000-000000000020
# ╠═080d0001-0000-4000-8000-000000000015
# ╟─080d0001-0000-4000-8000-000000000016
# ╟─080d0001-0000-4000-8000-000000000030
# ╟─080b0001-0000-4000-8000-000000000008
# ╟─080b0001-0000-4000-8000-000000000009
# ╠═080b0001-0000-4000-8000-000000000020
# ╟─080b0001-0000-4000-8000-000000000030
# ╟─080b0001-0000-4000-8000-000000000031
# ╟─080b0001-0000-4000-8000-000000000032
# ╟─5ed4d276-afc1-4044-a875-b85e915a7685
# ╠═080b0001-0000-4000-8000-000000000090
# ╟─ff4a7010-f828-4580-bfda-903fd3f02269
# ╟─080c0001-0000-4000-8000-000000000001
