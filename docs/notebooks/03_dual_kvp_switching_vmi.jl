### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ 171294a2-26bd-49e2-ac92-9df48ae5444f
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 69358294-97f2-4782-94d7-c29c747c45f4
using Markdown: @md_str, Markdown

# ╔═╡ 9ae27110-5c47-442b-a98e-d137599570f2
using Statistics: mean, std, quantile

# ╔═╡ d3054785-9e00-4094-a491-088ce63be9dc
md"""
# Dual-kVp Rapid-Switching Virtual Monoenergetic Imaging

**One rotation of a GE Revolution Apex Elite that switches between 80 and 140 kVp becomes
virtual monoenergetic images (VMIs) at 50, 70, 100 and 140 keV, checked rod by rod against
theory.**

The scanner, the acquisition and the chain are those of the basis-spectral-denoising paper: its
GE model and physics switches, the two exposures of a rotation with their duty cycles, view
integration, its fitted pair of FDK windows and a calibration draw that fixes the spectral pair.
"""

# ╔═╡ b9c6ce63-074d-410b-90d6-ab9bd39ea84b
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "One Tube, Two Voltages", "Rapid kVp Switching"],
        [
            "Gammex 472: Calcium and Iodine Rods",
            "GE Revolution Apex Elite",
            "80 and 140 kVp Within Every View",
        ];
        limits = [(-18, 18, -18, 18), (-7.5, 7.5, -7.5, 7.5), (-0.2, 13.4, -2.5, 8.5)],
    )
    gammex!(axes[1], GAMMEX; dark = true)
    gantry!(axes[2]; labels = ("80 / 140 kVp",))
    kvp_switching!(axes[3]; duty = exposures.low.view_arc, dark = true)
    fig
end

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## The Chain at a Glance

Each exposure is projected once and drawn three times: the noise-free expectation, the measured
scan and a calibration scan. The measured counts of both kVp go through one call,
`BS.vmi_pipeline`. Both channels stay separate all the way into the likelihood; nothing is
rebinned into one "effective" spectrum. The noise-free draw goes through the same
reconstruction, without the denoising, and is the reference every result is scored against.
"""

# ╔═╡ 14168d18-e05c-4ed7-b5c9-f2c15491653a
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 330))
    nodes = Dict(
        :low => (0.0, 3.0, "80 kVp Counts", "Measured Draw", KVP_COLORS.low),
        :high => (0.0, 1.4, "140 kVp Counts", "Measured Draw", KVP_COLORS.high),
        :hypr => (4.4, 2.2, "Projection HYPR", "3 × 3 in Each Row", COLORS.cyan),
        :decompose => (8.8, 2.2, "Decomposition", "K = 2, Every Ray", COLORS.cyan),
        :fdk => (13.2, 2.2, "FDK Pair", "A Window Each", COLORS.cyan),
        :acnr => (17.6, 2.2, "ACNR + HYPR", "On the Complement", COLORS.cyan),
        :vmi => (22.0, 2.2, "VMIs", "50 to 140 keV", COLORS.green),
        :basis => (8.8, 0.0, "Spectral Basis", "Φₖ(E) of Every Ray", COLORS.violet),
        :calibration => (13.2, 0.0, "Calibration Draw", "E*, β", COLORS.purple),
    )
    edges = [
        :low => :hypr,
        :high => :hypr,
        :hypr => :decompose,
        :decompose => :fdk,
        :fdk => :acnr,
        :acnr => :vmi,
        :basis => :decompose,
        :calibration => :fdk,
    ]
    flow_chart!(
        fig[1, 1],
        nodes,
        edges;
        limits = (-2.0, 24.0, -0.75, 3.75),
        w = 3.6,
        h = 1.1,
    )
    fig
end

# ╔═╡ 3d515abe-f3d9-4ce5-96c7-bef7da9bf294
md"""
## Notebook Setup
"""

# ╔═╡ 492bb299-678d-4e6f-8c21-1e9178cc2beb
import PlutoUI

# ╔═╡ 9f8d5cd4-147e-4359-95bc-cc096a53f0e7
import BasisSimulator as BS

# ╔═╡ 2ff539c9-a678-403c-b629-8068a332a0e9
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
PlutoUI.TableOfContents()

# ╔═╡ 86c52e9e-7987-4504-93e6-128017f5e703
begin
    import GPUSelect
    # the backend array type, directly: MtlArray / CuArray / ROCArray
    AT = GPUSelect.Storage()
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 551f84fe-d7b4-48f9-a475-0c63178a6ede
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ 796ba102-7406-414c-8994-a8c86f6564e8
md"""
## Figure Style

The colours, the theme and the drawing helpers every BasisSimulator notebook shares, then this
notebook's own drawings: the Gammex 472, drawn from its label mask, and the kVp waveform.
"""

# ╔═╡ eb6ef3f6-324f-4737-a96c-0cbce838156d
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

# ╔═╡ 1b9d059f-6c05-4215-91f3-75a70fc9d5b3
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

# ╔═╡ b3a56cb6-3ec3-4f0f-9004-a94d0d77468a
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

# ╔═╡ cba974a0-17fa-40c6-8e84-28268204a721
# the two exposures' colours: warm for the soft 80 kVp beam, violet for the hard 140 kVp one
KVP_COLORS = (low = COLORS.beam, high = COLORS.violet);

# ╔═╡ 88a25e41-aa75-44ba-8d85-1e57e2ed0255
"""
The Gammex 472 as the drawings show it: its body radius and every rod's centre and radius
(cm, about the centre of the image), measured from the phantom's own label mask.
"""
function gammex_layout(phantom)
    mask = phantom.mask[:, :, (size(phantom.mask, 3) + 1) ÷ 2]
    voxel = phantom.voxel_size[1]
    centre = (size(mask, 1) + 1) / 2
    function rod(label)
        pixels = findall(==(UInt8(label)), mask)
        return (
            x = (mean(p -> p[1], pixels) - centre) * voxel,
            y = (mean(p -> p[2], pixels) - centre) * voxel,
            r = sqrt(length(pixels) / π) * voxel,
        )
    end
    return (
        body = sqrt(count(!=(0x00), mask) / π) * voxel,
        voxel = voxel,
        Ca = [rod(label) for label in 10:16],
        I = [rod(label) for label in 20:26],
        Ca_names = ["50", "100", "200", "300", "400", "500", "600"],
        I_names = ["2", "2.5", "5", "7.5", "10", "15", "20"],
    )
end

# ╔═╡ 9dfe5884-b920-413b-a408-11797ab58dae
"""
Draws the Gammex 472 of `layout` on `ax`: the solid-water body, calcium rods in red and iodine
rods in green. `show` picks what stands out: `:all`, `:water` (with `region`, a mask on the
phantom's image grid, drawn in cyan), `:Ca` or `:I` (those rods, named by concentration, each with
its ROI of radius `roi` cm); the rest is faded.
"""
function gammex!(ax, layout; show = :all, region = nothing, roi = 0.55, dark = false)
    ink = dark ? COLORS.night_ink : COLORS.ink
    faded(group) = show in (:all, group) ? 1.0 : 0.25
    disk(x, y, r) = Mke.Circle(Mke.Point2f(x, y), r)
    Mke.poly!(
        ax,
        disk(0, 0, layout.body);
        color = (COLORS.blue, show == :water ? 0.25 : 0.55),
        strokecolor = ink,
        strokewidth = 1.5,
    )
    if show == :water && region !== nothing
        xs = ((1:size(region, 1)) .- (size(region, 1) + 1) / 2) .* layout.voxel
        ys = ((1:size(region, 2)) .- (size(region, 2) + 1) / 2) .* layout.voxel
        Mke.heatmap!(
            ax,
            xs,
            ys,
            [inside ? 1.0f0 : NaN32 for inside in region];
            colormap = [COLORS.cyan, COLORS.cyan],
            colorrange = (0, 1),
            nan_color = :transparent,
        )
    end
    for (group, color) in ((:Ca, COLORS.red), (:I, COLORS.green))
        rods = getfield(layout, group)
        names = getfield(layout, Symbol(group, :_names))
        offset = group == :Ca ? 2.3 : 2.6
        for (rod, name) in zip(rods, names)
            Mke.poly!(ax, disk(rod.x, rod.y, rod.r); color = (color, faded(group)))
            show == group || continue
            Mke.lines!(
                ax,
                disk(rod.x, rod.y, roi);
                color = COLORS.paper,
                linewidth = 2,
                linestyle = :dash,
            )
            Mke.text!(
                ax,
                rod.x * (1 + offset / hypot(rod.x, rod.y)),
                rod.y * (1 + offset / hypot(rod.x, rod.y));
                text = name,
                align = (:center, :center),
                fontsize = 15,
                font = :bold,
                color = ink,
            )
        end
    end
    return ax
end

# ╔═╡ d6175803-7b56-49f0-8939-8c4e0ac85345
"""
Rapid kVp switching on `ax` (a canvas about `(0, 12.6, -1.6, 7.6)`): the tube voltage over three
view periods, 80 kVp for `duty` of each and 140 kVp for the rest.
"""
function kvp_switching!(ax; duty = 0.65, dark = false)
    ink = dark ? COLORS.night_ink : COLORS.ink
    muted = dark ? COLORS.night_muted : COLORS.muted
    period, low, high = 4.0, 1.8, 5.0
    for v in 0:2
        t0 = 0.6 + v * period
        t1 = t0 + duty * period
        t2 = t0 + period
        Mke.poly!(ax, Mke.Rect2f(t0, 0, t1 - t0, low); color = (KVP_COLORS.low, 0.55))
        Mke.poly!(ax, Mke.Rect2f(t1, 0, t2 - t1, high); color = (KVP_COLORS.high, 0.55))
        Mke.lines!(
            ax,
            Mke.Point2f[(t0, low), (t1, low), (t1, high), (t2, high), (t2, low)];
            color = ink,
            linewidth = 2.5,
        )
        Mke.lines!(ax, [t2, t2], [-0.25, 0.0]; color = muted, linewidth = 2)
    end
    arrow!(ax, (0.6, 0), (12.9, 0); color = muted, head = 12)
    Mke.text!(
        ax,
        12.9,
        -0.35;
        text = "Time",
        align = (:right, :top),
        fontsize = 16,
        color = muted,
    )
    Mke.text!(
        ax,
        0.6 + duty * period / 2,
        low / 2;
        text = "80 kVp",
        align = (:center, :center),
        fontsize = 17,
        font = :bold,
        color = ink,
    )
    Mke.text!(
        ax,
        0.6 + period * (1 + duty) / 2,
        high / 2 + 0.6;
        text = "140\nkVp",
        align = (:center, :center),
        fontsize = 17,
        font = :bold,
        color = ink,
    )
    # one view period, bracketed
    Mke.lines!(
        ax,
        Mke.Point2f[(4.6, 5.6), (4.6, 5.9), (8.6, 5.9), (8.6, 5.6)];
        color = muted,
        linewidth = 2,
    )
    Mke.text!(
        ax,
        6.6,
        6.1;
        text = "One View: $(round(Int, 100duty)) % + $(round(Int, 100(1 - duty))) %",
        align = (:center, :bottom),
        fontsize = 16,
        color = muted,
    )
    return ax
end

# ╔═╡ 040e1000-0000-4000-8000-000000000100
md"""
## 1. The Phantom

The Gammex 472 multi-energy phantom: a solid-water body with seven calcium (50–600 mg/mL) and
seven iodine (2–20 mg/mL) rods, 1 cm thick and the same at every z. The label mask goes to the
GPU; its material table stays on the host.
"""

# ╔═╡ 939dcda3-9be5-46c8-aaa1-ded273e8cf04
phantom_cpu = BS.create_gammex_472(
    n_voxels = 512,
    n_slices = 16,
    fov_cm = 35.0,
    z_cm = 1.0,
);

# ╔═╡ 5248ba55-965a-41f7-845c-99616018b475
phantom = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ c91576b0-73ba-4359-a29f-aae918b6344f
# where the body and every rod are, for the drawings
GAMMEX = gammex_layout(phantom_cpu);

# ╔═╡ 040e1000-0000-4000-8000-000000000101
md"""
## 2. The Scanner and Its kVp Switching

The documented `REVOLUTION_APEX`: source–isocentre 625.6 mm, source–detector 1100 mm, 834
columns × 0.6 mm on an arc with the quarter-detector offset, 256 rows × 0.625 mm, a Gemstone
scintillator (`:lumex`) and the GE large-body bowtie. Here without electronic noise.

Rapid kVp switching alternates one tube between 80 and 140 kVp within every view period, so both
channels see the same rays on one detector. It is modelled as two co-registered acquisitions on
this one geometry, each with its share of the view period, its **duty cycle**: 0.65 at 80 kVp,
0.35 at 140 kVp. A channel's tube current is the instantaneous current times its duty cycle, and
its views integrate over that fraction of the arc the gantry turns per view (`view_arc`).
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
scanner = BS.EICTScanner(BS.REVOLUTION_APEX; electronic_noise = 0.0)

# ╔═╡ eeed376c-8d0d-52ba-b48e-715ba6d3123e
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

# ╔═╡ 040e1000-0000-4000-8000-000000000102
md"""
## 3. Two Exposures and Their Dose

One protocol per exposure: 984 views over a 0.5 s rotation, 5 mm collimation, 4.5 mm of added
aluminium on both. The physical scan ran 407 mA at 80 kVp and 405 mA at 140 kVp, each multiplied
by its duty cycle.

The simulator's tube output per mA is generic, not GE's, so one factor, `DOSE_SCALE`, scales both
currents: it keeps the split between the exposures and makes the simulated CTDIvol (the package's
Monte Carlo dose in the 32 cm body phantom, summed over the exposures) equal the 10.07 mGy the
physical scan reported. The noise is then that of the dose the physical scan delivered.
"""

# ╔═╡ 03c00001-0000-4000-8000-000000000001
# The two exposures of one rotation: each energy is read for its duty cycle of the view
# period, the fraction of the view's arc it integrates over and of its tube current. Each has
# its own noise seed: two acquisitions drawn from one seed would carry the same noise pattern,
# correlated between the channels, which a real scan never has.
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

# ╔═╡ 03c00001-0000-4000-8000-000000000002
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

# ╔═╡ 03c00001-0000-4000-8000-000000000003
# One factor on both tube currents, so the simulated CTDIvol is the physical scan's 10.07 mGy.
DOSE_SCALE = 10.07 / sum(e -> ctdi_vol(exposure_protocol(e)), exposures)

# ╔═╡ 03c00001-0000-4000-8000-000000000004
protocols = map(e -> exposure_protocol(e; mA_scale = DOSE_SCALE), exposures);

# ╔═╡ 03c00001-0000-4000-8000-000000000005
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

        | Exposure | Physical mA | Duty cycle, `view_arc` | Simulated mA | CTDIvol (mGy) |
        |---|---:|---:|---:|---:|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 040e1000-0000-4000-8000-000000000103
md"""
## 4. Physics, Draws and the Image Grid

The physics switches of the basis-spectral-denoising GE model: every effect of the
energy-integrating path on except optical crosstalk, and the detector efficiency from the Monte
Carlo table of the Lumex scintillator (the scanner's default, `detector_efficiency_mode = :mc_lut`).

| Draw | Noise | Used for |
|---|---|---|
| 0, `reference` | off | the noise-free expectation every result is scored against |
| 1, `measured` | on | every result below |
| 9, `calibration` | on | fixing the spectral pair, never scored |

The reconstruction grid is the clinical series': 8 slices of 0.625 mm over the central 5 mm,
512² over 35 cm.

<details><summary><strong>Why <code>view_samples = 5</code>: view integration</strong></summary>

A detector integrates while the gantry turns, so each view records the transmitted intensity
averaged over the arc it sweeps, `r · view_arc · Δθ` at radius `r`: a blur of the object, largest
far from the isocentre, that the noise, counted once per view, does not share. `view_samples = 5`
samples that arc with five sub-views (the midpoint rule), 0.2 mm apart at the edge of a 35 cm
field. Without it the simulated signal is sharper, relative to its noise, than a physical
scanner's, and no reconstruction window can match both the physical resolution and the physical
noise texture.

</details>
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
begin
    VIEW_SAMPLES = 5
    ge_opts = BS.SimOptions(projector = :dd_fast, view_samples = VIEW_SAMPLES)
    "The same physics switches with one field replaced."
    with_option(opts, field, value) = BS.SimOptions(;
        (f => getfield(opts, f) for f in fieldnames(BS.SimOptions) if f !== field)...,
        (field => value,)...,
    )
end;

# ╔═╡ 03c00001-0000-4000-8000-000000000006
begin
    # Draw 0 is the noise-free expectation; draw r ≥ 1 gives each exposure the seed
    # `seed + 1000 (r − 1)`, so the exposures stay independent within a draw and the draws of
    # each other (basis-spectral-denoising's seeds for the Gammex).
    DRAWS = (reference = 0, measured = 1, calibration = 9)
    realization_seed(e, r) = e.seed + 1000 * (max(r, 1) - 1)
    "The options of draw `r` of exposure `e`: its seed, its noise, its arc."
    draw_opts(e, r) = with_option(
        with_option(with_option(ge_opts, :seed, realization_seed(e, r)), :use_noise, r > 0),
        :view_arc,
        e.view_arc,
    )
end;

# ╔═╡ 08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
recon_opts = let
    slice_thickness_mm = 0.625
    n_recon_slices = round(Int, 5.0 / slice_thickness_mm)
    BS.ReconOptions(matrix_size = (512, 512, n_recon_slices), fov_cm = 35.0, z_cm = 0.5)
end;

# ╔═╡ 5ecd97c6-ad47-4558-886d-22ed45eda97d
md"""
## 5. Simulate the Two Exposures: `simulate!`

Each exposure is projected once. The noise-free draw runs first with `keep_projection = true`,
which returns everything before the noise; the measured and calibration draws pass it back
(`projection`) and only draw their noise, bit-identical to simulating each afresh at a fraction
of the cost.

Each acquisition keeps its corrected log sinograms and its one-channel spectral basis,
`spectral_basis(ws, protocol)`: each ray's detected spectrum (source × filtration × bowtie × the
detector's signal per photon, `m₁(E) = E·efficiency(E)`), the model the projector applied, scaled
to the ray's noise-equivalent quanta in air, `I₀·ff·S₁²/S₂`. An energy integrator's signal is not
Poisson: its variance is `S₂/S₁` times its mean, the dispersion, which rises as the beam hardens.
"""

# ╔═╡ 90ce0f86-1464-42a8-b856-cc5487a4e4c1
"""
Simulates the three draws of exposure `e` under `protocol`, projecting once: each draw's
sinogram, the wall time, and the acquisition's geometry and one-channel spectral basis.
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

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
dual_scans = [simulate_exposure(exposures[k], protocols[k]) for k in keys(exposures)];

# ╔═╡ 03c00001-0000-4000-8000-000000000007
Markdown.parse(
    """
    The three draws of the two exposures took
    $(join(["$(round(s.elapsed_s, digits = 1)) s for $(s.label)" for s in dual_scans], " and ")),
    one projection each. Each sinogram is $(join(size(first(dual_scans).channels[DRAWS.measured]), " × "))
    (columns × rows × views).
    """,
)

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## 6. Two Spectra on Every Ray

`spectral_basis_from_acquisitions` merges the two kVp energy grids onto their union and scales
each acquisition's per-ray spectrum by its own noise-equivalent quanta, so the likelihood sees the
absolute response ``\Phi_k(E)`` of every ray and channel: source, filtration, bowtie and detector,
the model that generated the data. It does not depend on the noise, so every draw shares it.
"""

# ╔═╡ 4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
basis = BS.spectral_basis_from_acquisitions(acquisitions = [s.basis for s in dual_scans]);

# ╔═╡ 03a10001-0000-4000-8000-000000000001
let
    Mke.set_theme!(THEME)
    col = size(basis.Φ, 1) ÷ 2 + 1
    row = size(basis.Φ, 2) ÷ 2 + 1
    fig = Mke.Figure(size = (1400, 900))

    # what is measured: one tube switching voltage, and the central ray through the phantom
    waveform = canvas(fig[1, 1], (-0.2, 13.4, -1.8, 7.4))
    kvp_switching!(waveform; duty = exposures.low.view_arc)
    ring = canvas(fig[1, 2], (-7.5, 7.5, -7.6, 7.6))
    gantry!(ring; dark = false, labels = ("80 / 140 kVp",))
    Mke.lines!(ring, [0.0, 0.0], [5.4, -4.1]; color = COLORS.ink, linewidth = 2.5)
    Mke.text!(
        ring,
        0.3,
        3.4;
        text = "Central Ray",
        align = (:left, :center),
        fontsize = 16,
        font = :bold,
        color = COLORS.ink,
    )

    ax = Mke.Axis(
        fig[2, 1:2];
        title = "Two Detected Spectra on the Same Ray",
        subtitle = "The Central Ray's Φₖ(E), Each Normalized to Its Peak",
        xlabel = "Photon Energy (keV)",
        ylabel = "Relative Detected Fluence",
    )
    for k in 1:basis.n_channels
        φ = Float64.(basis.Φ[col, row, :, k])
        mean_E = sum(basis.E .* φ) / sum(φ)
        Mke.lines!(
            ax,
            basis.E,
            φ ./ maximum(φ);
            color = k == 1 ? KVP_COLORS.low : KVP_COLORS.high,
            linewidth = 3.5,
            label = "$(dual_scans[k].label), Mean $(round(mean_E, digits = 1)) keV",
        )
    end
    Mke.axislegend(ax; position = :rt)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.38))
    fig
end

# ╔═╡ 03a10001-0000-4000-8000-000000000002
Markdown.parse(
    """
    The basis holds $(basis.n_channels) channels on a $(length(basis.E))-point energy grid,
    $(round(Int, minimum(basis.E))) to $(round(Int, maximum(basis.E))) keV, for
    $(size(basis.Φ, 1)) × $(size(basis.Φ, 2)) rays; the response sums to the air counts to
    within $(round(basis.I0_relerr, sigdigits = 2)) (relative).
    """,
)

# ╔═╡ 03a10001-0000-4000-8000-000000000003
md"""
## 7. The Chain's Settings

**The denoiser** is generalized HYPR-LR in both domains, `BS.SpectralHYPR()`. Before the
decomposition it pools a 3 × 3 (column × view) window on the counts of each detector row, over
views of one parity so the odd- and even-view halves stay independent, with each channel's
dispersion measured on the air rays. After the reconstruction it pools the complement of the basis
pair within each slice, over the window with the least estimated risk. Neither ever pools across
detector rows or slices. The decomposition is the K-channel estimator with its published
controls; T-LBF (a photon-counting filter) is off; the view-direction antialias is on; ACNR is on
(`vmi_pipeline`'s default).

**A pair of FDK windows.** A VMI is ``A(E)\,M + B(E)\,I_\perp``: ``M`` is the minimum-noise VMI
(the composite, at ``E^\star``) and ``I_\perp`` its complement, whose noise is uncorrelated with
``M``'s. `BS.PairFilter` gives each its own window, and one pair serves every energy. Each window is
an apodized ramp ``W(f) = \exp(-(f/f_c)^p)`` on the grid-Nyquist axis, the GE pair fitted by
basis-spectral-denoising to the physical Gammex scan's MTF at every energy and NPS shape, on this
notebook's grid.

!!! note "Why not one window per basis image?"
    The water and iodine images' noise is strongly anti-correlated, and cancels in a VMI only
    where both were filtered alike. A window per basis image would leave uncancelled noise in
    every VMI wherever the two windows differ; the composite and its complement have no such
    correlation to break.
"""

# ╔═╡ 03a10001-0000-4000-8000-000000000004
# the published denoiser: ProjectionHYPR(3 × 3 box, view_stride = 2) and ImageHYPR()
HYPR_CHAIN = BS.SpectralHYPR()

# ╔═╡ 03a10001-0000-4000-8000-000000000005
VMI_CHAIN = (
    method = :nchannel,
    controls = BS.NChannelControls(),
    use_tlbf = false,
    antialias = true,
);

# ╔═╡ 03a10001-0000-4000-8000-000000000006
# The GE pair of basis-spectral-denoising (`RECON.ge`): W(f) = exp(-(f / f_c)^p) at 11 knots.
GE_KERNELS = (composite = (fc = 0.55, p = 1.5), complement = (fc = 1.05, p = 1.0));

# ╔═╡ 03c00001-0000-4000-8000-000000000008
PAIR_FILTER = let knots = Tuple(range(0.0, 1.0, length = 11))
    window(q) = BS.CustomFilter(
        knots,
        Tuple(round(exp(-(x / q.fc)^q.p), digits = 5) for x in knots)
    )
    BS.PairFilter(window(GE_KERNELS.composite), window(GE_KERNELS.complement))
end;

# ╔═╡ 03c00001-0000-4000-8000-000000000009
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 880))
    window(q) = "Window: fc = $(q.fc), p = $(q.p)"

    # what the windows act on: the basis pair, rewritten as a composite and its complement
    nodes = Dict(
        :pair => (0.0, 0.7, "Water + Iodine", "The Basis Pair", COLORS.blue),
        :composite => (
            5.2,
            1.5,
            "Composite M",
            window(GE_KERNELS.composite),
            COLORS.violet,
        ),
        :complement => (
            5.2,
            -0.1,
            "Complement I⊥",
            window(GE_KERNELS.complement),
            COLORS.cyan,
        ),
        :vmi => (10.4, 0.7, "VMI at Energy E", "A(E) M + B(E) I⊥", COLORS.green),
    )
    edges = [
        :pair => :composite,
        :pair => :complement,
        :composite => :vmi,
        :complement => :vmi,
    ]
    flow_chart!(
        fig[1, 1],
        nodes,
        edges;
        limits = (-2.4, 12.8, -0.95, 2.35),
        w = 4.2,
        h = 1.2,
    )

    f = range(0.0, 1.0, length = 201)
    ax = Mke.Axis(
        fig[2, 1];
        title = "The GE Pair of FDK Windows",
        xlabel = "Frequency (Fraction of the Grid's Nyquist)",
        ylabel = "Window W(f)",
    )
    for (name, w, color) in (
            ("Composite M", PAIR_FILTER.composite, COLORS.violet),
            ("Complement I⊥", PAIR_FILTER.complement, COLORS.cyan),
        )
        Mke.lines!(
            ax,
            f,
            [BS.frequency_window(w, x) for x in f];
            color,
            linewidth = 3.5,
            label = name,
        )
    end
    Mke.axislegend(ax; position = :rt)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.33))
    fig
end

# ╔═╡ 03c00001-0000-4000-8000-00000000000a
md"""
## 8. The Calibration Draw: `pair_basis` and `composite_energy`

The pair the windows act on, ``(E^\star, \beta)``, and the energy of the composite that ACNR and
the image-domain HYPR act on are properties of the scanner and protocol. They are measured once,
on the calibration draw, and fixed for every other draw, as basis-spectral-denoising's
`PAIR_BASIS` does:

1. the calibration draw is decomposed as measured, and after the projection-domain HYPR alone
   (`vmi_pipeline(; keep_sinograms = true)`, with `SpectralHYPR(image = nothing)` for the second);
2. `BS.spectral_pair` of the first, with a standard soft-tissue window, gives the minimum-noise
   energy ``E^\star`` and the complement ``\beta``: the `pair_basis`;
3. `BS.spectral_pair` of the second on that basis gives the minimum-noise energy after the
   projection-domain HYPR, which moves it: the `composite_energy`.

Near its minimum the VMI noise hardly changes with energy, so the argmin of one acquisition is
itself noise and would move from draw to draw. Measured on a draw of its own, it cannot tune the
chain to the draw being scored, and the noise-free reference, which has no noise to measure it
from, is reconstructed with the same pair as the measured scan.
"""

# ╔═╡ 03c00001-0000-4000-8000-00000000000b
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

# ╔═╡ 040e1000-0000-4000-8000-000000000003
md"""
## 9. One Call: `vmi_pipeline`

From the two measured sinograms to the VMI stack, reconstructed on the notebook's grid
(`recon_opts.matrix_size`) with every detector row kept: the published chain, with the GE windows
and the calibration draw's pair.
"""

# ╔═╡ 040e0002-0000-4000-8000-000000000002
dual_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.measured] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = (50.0, 70.0, 100.0, 140.0),
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    VMI_CHAIN...,
);

# ╔═╡ 03c00001-0000-4000-8000-00000000000c
md"""
The **noise-free reference** is the noise-free draw through the same reconstruction, windows and
pair, with nothing to denoise: no HYPR and no ACNR. Scoring the measured VMIs against it measures
what the noise and the denoising do, separately from what the physics the decomposition does not
invert exactly (scatter, and the blurs of the focal spot, the detector and the view integration)
does to both.
"""

# ╔═╡ 03c00001-0000-4000-8000-00000000000d
reference_vmi = BS.vmi_pipeline(;
    channels = [s.channels[DRAWS.reference] for s in dual_scans],
    basis,
    geom = first(dual_scans).geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = (50.0, 70.0, 100.0, 140.0),
    fbp_filter = PAIR_FILTER,
    pair_basis = calibration.pair_basis,
    composite_energy = calibration.composite_energy,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 03a10001-0000-4000-8000-000000000007
let
    q = dual_vmi.quality
    d = dual_vmi.settings.denoiser
    b = calibration.pair_basis
    pct(x) = round(100x, digits = 3)
    Markdown.parse(
        """
        | Quantity | Value |
        |---|---|
        | the windows' pair, from the calibration draw | E* = $(round(Int, b.Estar)) keV, β = $(round(b.β, sigdigits = 3)) |
        | the composite's energy, from the calibration draw | $(round(Int, calibration.composite_energy)) keV |
        | rays decomposed, and the time | $(q.n_rays) in $(round(dual_vmi.elapsed_s, digits = 1)) s |
        | outer iterations, mean | $(round(q.outer_mean, digits = 1)) |
        | rays not converged | $(pct(q.frac_not_converged)) % |
        | rays at the iodine / water bound | $(pct(q.frac_bound_iodine)) % / $(pct(q.frac_bound_water)) % |
        | count dispersion on the air rays, 80 / 140 kVp | $(join(round.(d.dispersion, digits = 2), " / ")) |
        | image HYPR window on the complement | $(d.image_estimates.window) × $(d.image_estimates.window) pixels |
        """,
    )
end

# ╔═╡ 040e0002-0000-4000-8000-000000000003
md"""
## 10. Basis Images and VMIs

The water and iodine basis pair after ACNR and the image HYPR (mid slice), and the VMIs
synthesized from it. Every VMI comes from this one pair; no energy-dependent filtering is applied
to any of them.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000003
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 640))
    mid = size(dual_vmi.images.water, 3) ÷ 2 + 1
    panels = (
        ("Iodine Basis", "Iodine (mg/mL)", 1000 .* dual_vmi.images.iodine),
        ("Water Basis", "Water (g/mL)", dual_vmi.images.water),
    )
    for (column, (title, label, volume)) in pairs(panels)
        image = volume[:, :, mid]
        range = Tuple(quantile(vec(image), (0.01, 0.99)))
        axis = Mke.Axis(fig[1, 2column - 1]; title, aspect = Mke.DataAspect())
        Mke.heatmap!(axis, image; colormap = :viridis, colorrange = range)
        Mke.hidedecorations!(axis)
        Mke.hidespines!(axis)
        Mke.Colorbar(
            fig[1, 2column];
            colormap = :viridis,
            colorrange = range,
            label,
            width = 16,
        )
    end
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000004
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 1180))
    mid = size(dual_vmi.vmis, 3) ÷ 2 + 1
    for (index, energy) in pairs(dual_vmi.energies)
        axis = Mke.Axis(
            fig[(index - 1) ÷ 2 + 1, (index - 1) % 2 + 1];
            title = "$(Int(energy)) keV VMI",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(
            axis,
            dual_vmi.vmis[:, :, mid, index];
            colormap = :grays,
            colorrange = (-200, 500),
        )
        Mke.hidedecorations!(axis)
        Mke.hidespines!(axis)
    end
    Mke.Colorbar(
        fig[1:2, 3];
        colormap = :grays,
        colorrange = (-200, 500),
        label = "HU",
        width = 16,
    )
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_projection_grid.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000008
md"""
## 11. Accuracy and Noise

Per-rod measured, noise-free and theoretical HU at the four VMI energies, the water region's mean
and noise, and the regression of measured on theoretical HU.

| Quantity | How it is measured |
|---|---|
| measured HU | the mean over an 8-pixel-radius disk at each rod's centroid, over every slice, of the measured VMIs |
| noise-free HU | the same mean on the noise-free reference |
| theoretical HU | ``1000\,(\mu_r(E) - \mu_w(E))/\mu_w(E)`` of the rod's material, from `BS.compute_μ_at_energy`, with no fitting |
| water region | the solid water eroded with `erode_mask_2d(…; erode_px = 12)`: its mask blurred by a 12-pixel Gaussian and kept where the blur stays above 0.999, about 3σ (2.5 cm) clear of the body's edge and of every rod |
| noise σ | the standard deviation, over the water region, of the measured VMI minus the noise-free reference: the noise alone, without the phantom's own nonuniformity |
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000002
dual_results = let
    mask = phantom_cpu.mask[:, :, size(phantom_cpu.mask, 3) ÷ 2]
    water_mask = collect(BS.erode_mask_2d(
        mask .== UInt8(BS.REGION_SOLID_WATER);
        erode_px = 12.0,
    ))
    rod_labels = (
        Ca = UInt8.((10, 11, 12, 13, 14, 15, 16)),
        I = UInt8.((20, 21, 22, 23, 24, 25, 26)),
    )
    rod_names = (
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
    )
    function rod_roi(label)
        pixels = findall(==(label), mask)
        cx = mean(pixel -> Float64(pixel[1]), pixels)
        cy = mean(pixel -> Float64(pixel[2]), pixels)
        [
            CartesianIndex(i, j)
            for j in max(1, floor(Int, cy - 8)):min(size(mask, 2), ceil(Int, cy + 8))
            for i in max(1, floor(Int, cx - 8)):min(size(mask, 1), ceil(Int, cx + 8))
            if (i - cx)^2 + (j - cy)^2 ≤ 64
        ]
    end
    energies = Float64.(dual_vmi.energies)
    vmi(stack, index) = view(stack.vmis, :, :, :, index)
    region(volume, pixels) = [volume[p, z] for z in axes(volume, 3) for p in pixels]
    μwater = Dict(E => BS.compute_μ_at_energy(BS.XA.Materials.water, E) for E in energies)
    rods = Dict{Symbol, NamedTuple}()
    for group in (:Ca, :I)
        measured = zeros(length(rod_labels[group]), length(energies))
        reference = similar(measured)
        theoretical = similar(measured)
        for (row, label) in pairs(rod_labels[group])
            roi = rod_roi(label)
            material = phantom_cpu.materials[Int(label) + 1]
            for (column, E) in pairs(energies)
                measured[row, column] = mean(region(vmi(dual_vmi, column), roi))
                reference[row, column] = mean(region(vmi(reference_vmi, column), roi))
                μ = BS.compute_μ_at_energy(material, E)
                theoretical[row, column] = 1000 * (μ - μwater[E]) / μwater[E]
            end
        end
        rods[group] = (names = rod_names[group], measured, reference, theoretical)
    end
    water_pixels = findall(water_mask)
    water(stack, k) = region(vmi(stack, k), water_pixels)
    (
        energies,
        water_mask,
        rods,
        water_mean = [mean(water(dual_vmi, k)) for k in eachindex(energies)],
        water_mean_reference = [mean(water(reference_vmi, k)) for k in eachindex(energies)],
        water_noise = [
            std(water(dual_vmi, k) .- water(reference_vmi, k))
            for k in eachindex(energies)
        ],
        finite = all(isfinite, dual_vmi.vmis) && all(isfinite, reference_vmi.vmis),
    )
end;

# ╔═╡ 040e1000-0000-4000-8000-000000000040
md"""
### The Solid Water

The eroded solid-water region (left) and its mean HU at each energy, measured and noise-free. The
phantom's solid water has the composition of water, so its theoretical HU is 0 at every energy.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000005
let
    Mke.set_theme!(THEME)
    r = dual_results
    n = length(r.energies)
    fig = Mke.Figure(size = (1400, 600))

    sketch = canvas(fig[1, 1], (-17.5, 17.5, -17.5, 17.5))
    gammex!(sketch, GAMMEX; show = :water, region = dual_results.water_mask)
    Mke.Label(
        fig[2, 1],
        "The Eroded Water Region";
        fontsize = 19,
        font = :bold,
        tellwidth = false,
    )

    ax = Mke.Axis(
        fig[1:2, 2];
        title = "Solid-Water Mean HU",
        subtitle = "Theory: 0 HU at Every Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Mean HU",
        xticks = (1:n, string.(Int.(r.energies))),
    )
    Mke.barplot!(
        ax,
        vcat(1:n, 1:n),
        vcat(r.water_mean, r.water_mean_reference);
        dodge = vcat(fill(1, n), fill(2, n)),
        color = vcat(fill(COLORS.cyan, n), fill(COLORS.rule, n)),
    )
    Mke.hlines!(ax, [0.0]; color = COLORS.ink, linestyle = :dash, linewidth = 1.5)
    for (index, value) in pairs(r.water_mean)
        Mke.text!(
            ax,
            index - 0.2,
            value;
            text = "$(round(value, digits = 1))",
            align = (:center, value ≥ 0 ? :bottom : :top),
            offset = (0, value ≥ 0 ? 5 : -5),
            fontsize = 16,
        )
    end
    Mke.axislegend(
        ax,
        [Mke.PolyElement(color = COLORS.cyan), Mke.PolyElement(color = COLORS.rule)],
        ["Measured", "Noise-Free Reference"];
        position = :lb,
    )
    Mke.ylims!(ax, -10, 10)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "vmi_water_roi_check.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000050
md"""
### The Noise

The noise is what is left when the noise-free reference is subtracted from the measured VMI (left,
at 50 keV, in the water region): the phantom's structure and any deterministic artefact cancel.
Its standard deviation is the noise σ at each energy (right).
"""

# ╔═╡ 040e1000-0000-4000-8000-000000000051
let
    Mke.set_theme!(THEME)
    r = dual_results
    n = length(r.energies)
    mid = size(dual_vmi.vmis, 3) ÷ 2 + 1
    difference = dual_vmi.vmis[:, :, mid, 1] .- reference_vmi.vmis[:, :, mid, 1]
    noise_only = [
        r.water_mask[i] ? Float32(difference[i]) : NaN32
        for i in eachindex(difference)
    ]
    noise_only = reshape(noise_only, size(difference))
    limit = 3 * first(r.water_noise)
    fig = Mke.Figure(size = (1400, 600))

    image = Mke.Axis(
        fig[1, 1];
        title = "Measured − Noise-Free",
        subtitle = "$(Int(first(r.energies))) keV, Mid Slice, the Water Region",
        aspect = Mke.DataAspect(),
    )
    Mke.heatmap!(
        image,
        noise_only;
        colormap = :grays,
        colorrange = (-limit, limit),
        nan_color = COLORS.card,
    )
    Mke.hidedecorations!(image)
    Mke.hidespines!(image)
    Mke.Colorbar(
        fig[1, 2];
        colormap = :grays,
        colorrange = (-limit, limit),
        label = "Noise (HU)",
        width = 16,
    )

    ax = Mke.Axis(
        fig[1, 3];
        title = "Water-Region Noise vs Energy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        xticks = (1:n, string.(Int.(r.energies))),
    )
    Mke.barplot!(ax, 1:n, r.water_noise; color = COLORS.cyan)
    for index in 1:n
        Mke.text!(
            ax,
            index,
            r.water_noise[index];
            text = "σ = $(round(r.water_noise[index], digits = 1))",
            align = (:center, :bottom),
            offset = (0, 6),
            fontsize = 17,
        )
    end
    Mke.ylims!(ax, 0, 1.25maximum(r.water_noise))
    Mke.colsize!(fig.layout, 3, Mke.Relative(0.5))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "dual_kvp_vmi_water_noise_vs_energy.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000060
md"""
### The Rods against Theory

Each rod is measured in the dashed disk drawn on it (top). Solid lines are the measured HU,
crosses the noise-free reference and dashed lines the theoretical HU.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000006
let
    Mke.set_theme!(THEME)
    r = dual_results
    roi = 8 * phantom_cpu.voxel_size[1]
    fig = Mke.Figure(size = (1400, 1100))
    panels = (
        (
            group = :Ca,
            title = "Calcium Rods (mg/mL)",
            shades = Mke.cgrad(
                [Mke.RGBf(0.96, 0.65, 0.55), Mke.RGBf(0.45, 0.08, 0.06)],
                7;
                categorical = true,
            ),
            ylim = (0, 3800),
        ),
        (
            group = :I,
            title = "Iodine Rods (mg/mL)",
            shades = Mke.cgrad(
                [Mke.RGBf(0.62, 0.85, 0.52), Mke.RGBf(0.06, 0.32, 0.05)],
                7;
                categorical = true,
            ),
            ylim = (0, 1500),
        ),
    )
    for (column, panel) in pairs(panels)
        sketch = canvas(fig[1, column], (-17.5, 17.5, -17.5, 17.5))
        gammex!(sketch, GAMMEX; show = panel.group, roi)
        Mke.Label(
            fig[2, column],
            panel.title;
            fontsize = 19,
            font = :bold,
            tellwidth = false,
        )

        axis = Mke.Axis(
            fig[3, column];
            title = panel.group == :Ca ? "Calcium: HU vs Energy" : "Iodine: HU vs Energy",
            subtitle = "Solid: Measured · ×: Noise-Free · Dashed: Theory",
            xlabel = "VMI Energy (keV)",
            ylabel = "HU",
            xticks = r.energies,
        )
        Mke.ylims!(axis, panel.ylim...)
        data = r.rods[panel.group]
        for index in eachindex(data.names)
            color = panel.shades[index]
            Mke.scatterlines!(
                axis,
                r.energies,
                vec(data.measured[index, :]);
                color,
                linewidth = 2.5,
                markersize = 10,
                label = data.names[index],
            )
            Mke.scatter!(
                axis,
                r.energies,
                vec(data.reference[index, :]);
                color = COLORS.ink,
                marker = :xcross,
                markersize = 10,
            )
            Mke.lines!(
                axis,
                r.energies,
                vec(data.theoretical[index, :]);
                color,
                linewidth = 1.6,
                linestyle = :dash,
            )
        end
        Mke.axislegend(axis; position = :rt, labelsize = 15, rowgap = 2)
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.38))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "vmi_vs_theoretical.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 03c00001-0000-4000-8000-00000000000e
let
    r = dual_results
    f(x) = round(x, digits = 1)
    bias(g, k) = r.rods[g].measured[:, k] .- r.rods[g].reference[:, k]
    mae(a, b) = mean(abs.(a .- b))
    function row(k)
        ca, io = r.rods[:Ca], r.rods[:I]
        allbias = vcat(bias(:Ca, k), bias(:I, k))
        return "| $(Int(r.energies[k])) | $(f(r.water_mean[k])) / $(f(r.water_mean_reference[k])) | $(f(r.water_noise[k])) | " *
            "$(f(mae(ca.measured[:, k], ca.theoretical[:, k]))) / $(f(mae(ca.reference[:, k], ca.theoretical[:, k]))) | " *
            "$(f(mae(io.measured[:, k], io.theoretical[:, k]))) / $(f(mae(io.reference[:, k], io.theoretical[:, k]))) | " *
            "$(f(mean(allbias))) | $(f(maximum(abs, allbias))) |"
    end

    rows = [row(k) for k in eachindex(r.energies)]

    Markdown.parse(
        """
        **Accuracy against the noise-free reference and against theory.** Water: the mean HU of the
        measured / noise-free VMI, and the noise σ. Rods: the mean absolute error against theory of the
        measured / noise-free VMI, and the mean and the largest absolute bias of the 14 measured rods
        against the noise-free reference.

        | keV | Water HU | Water σ (HU) | Ca rods, error vs theory (HU) | I rods, error vs theory (HU) | Rod bias vs reference, mean (HU) | Rod bias vs reference, max (HU) |
        |---:|---:|---:|---:|---:|---:|---:|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 03c00001-0000-4000-8000-00000000000f
let
    r = dual_results
    f(x) = round(x, digits = 1)
    gap = maximum(abs, r.water_mean .- r.water_mean_reference)
    mae(g, a) = [
        mean(abs.(getfield(r.rods[g], a)[:, k] .- r.rods[g].theoretical[:, k]))
        for k in eachindex(r.energies)
    ]
    worst(a) = maximum(max(x, y) for (x, y) in zip(mae(:Ca, a), mae(:I, a)))
    Markdown.parse(
        """
        The solid water reads $(join(f.(r.water_mean), " / ")) HU at 50 / 70 / 100 / 140 keV, within
        $(f(gap)) HU of the noise-free reference: an offset the reference shares comes from physics the
        decomposition does not invert exactly (scatter, and the blurs of the focal spot, the detector and
        the view integration, taken before the log), not from the noise or the denoising. The noise falls
        from σ = $(f(first(r.water_noise))) HU at 50 keV to $(f(last(r.water_noise))) HU at 140 keV. The
        largest mean rod error against theory, over both rod groups and every energy, is
        $(f(worst(:measured))) HU measured and $(f(worst(:reference))) HU noise-free.

        That largest error is the calcium rods at 50 keV, $(f(first(mae(:Ca, :measured)))) HU measured and
        $(f(first(mae(:Ca, :reference)))) HU noise-free. The noise-free reference has it too, so it is the
        basis, not the noise or the denoiser: the decomposition writes every material as water plus
        iodine, which spans calcium's energy dependence only approximately, and least well at low energy,
        where the photoelectric term dominates. From 70 keV up the calcium error is
        $(f(minimum(mae(:Ca, :measured)[2:end]))) to $(f(maximum(mae(:Ca, :measured)[2:end]))) HU.
        """,
    )
end

# ╔═╡ 040e1000-0000-4000-8000-000000000070
md"""
### Linear Regression

The same rods (drawn above), measured HU against theoretical HU at each energy, with the
least-squares line. The dashed line is identity, perfect agreement.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000007
let
    Mke.set_theme!(THEME)
    colors = Dict(
        50.0 => COLORS.red,
        70.0 => COLORS.beam,
        100.0 => COLORS.cyan,
        140.0 => COLORS.blue,
    )
    fig = Mke.Figure(size = (1400, 640))
    for (column, group) in enumerate((:Ca, :I))
        data = dual_results.rods[group]
        axis = Mke.Axis(
            fig[1, column];
            title = group == :Ca ? "Calcium Rods" : "Iodine Rods",
            xlabel = "Theoretical HU",
            ylabel = "Measured HU",
        )
        low = min(0.0, minimum(data.measured), minimum(data.theoretical))
        high = 1.05max(maximum(data.measured), maximum(data.theoretical))
        Mke.lines!(
            axis,
            [low, high],
            [low, high];
            color = COLORS.ink,
            linestyle = :dash,
            linewidth = 2,
            label = "Identity",
        )
        for (index, energy) in pairs(dual_results.energies)
            x = vec(data.theoretical[:, index])
            y = vec(data.measured[:, index])
            slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum(abs2, x .- mean(x))
            intercept = mean(y) - slope * mean(x)
            prediction = intercept .+ slope .* x
            r2 = 1 - sum(abs2, y .- prediction) / sum(abs2, y .- mean(y))
            endpoints = collect(extrema(x))
            color = colors[energy]
            Mke.scatter!(axis, x, y; color, markersize = 12)
            Mke.lines!(
                axis,
                endpoints,
                intercept .+ slope .* endpoints;
                color,
                linewidth = 2,
                label = "$(Int(energy)) keV: Slope $(round(slope, digits = 3)), " *
                    "R² $(round(r2, digits = 4))",
            )
        end
        Mke.axislegend(axis; position = :rb, labelsize = 15)
    end
    Mke.save(joinpath(@__DIR__, "..", "assets", "vmi_regression.png"), fig; px_per_unit = 2)
    fig
end

# ╔═╡ 040e0002-0000-4000-8000-000000000004
checks = let
    list = NamedTuple[]
    addcheck(name, value, pass) = push!(list, (; name, value, pass))
    addcheck(
        "both kVp channels in the likelihood",
        dual_vmi.settings.n_channels,
        dual_vmi.settings.n_channels == 2,
    )
    addcheck(
        "chain: projection + image HYPR, ACNR, pair of windows",
        (
            dual_vmi.settings.denoiser !== nothing,
            dual_vmi.settings.acnr !== nothing,
            dual_vmi.settings.recon.filter isa BS.PairFilter,
        ),
        dual_vmi.settings.denoiser !== nothing &&
            dual_vmi.settings.acnr !== nothing &&
            dual_vmi.settings.recon.filter isa BS.PairFilter,
    )
    addcheck(
        "measured and reference share the pair",
        dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis,
        dual_vmi.settings.pair.basis == reference_vmi.settings.pair.basis,
    )
    addcheck("all VMI values finite", dual_results.finite, dual_results.finite)
    water_worst = maximum(abs, dual_results.water_mean)
    addcheck(
        "solid-water worst absolute HU",
        round(water_worst, digits = 2),
        water_worst ≤ 10,
    )
    list
end;

# ╔═╡ 12d92a54-471f-4018-9822-610113e19f27
let
    passed = count(check -> check.pass, checks)
    rows = [
        "| $(check.name) | $(check.value) | $(check.pass ? "pass" : "CHECK") |"
        for check in checks
    ]
    Markdown.parse(
        """
        ## 12. Verification: $(passed) of $(length(checks)) Checks Pass

        | Check | Value | Result |
        |---|---:|:---:|
        $(join(rows, "\n"))
        """,
    )
end

# ╔═╡ 040e0002-0000-4000-8000-000000000006
let
    r = dual_results
    f(x) = round(x, digits = 1)
    b = calibration.pair_basis
    rod_error = maximum(
        mean(abs.(r.rods[g].measured[:, k] .- r.rods[g].theoretical[:, k]))
        for g in (:Ca, :I)
        for k in eachindex(r.energies)
    )
    Markdown.parse(
        """
        ## Summary

        Two co-registered exposures, 80 kVp for $(round(Int, 100 * exposures.low.view_arc)) % of every
        view and 140 kVp for the rest, go into one maximum-likelihood decomposition as two separate
        measurements, and every VMI is synthesized from the one water–iodine pair it returns.

        | Result | Value |
        |---|---|
        | CTDIvol, both exposures (dose-matched to the physical scan) | $(round(sum(ctdi_vol(p) for p in protocols), digits = 2)) mGy |
        | spectral pair from the calibration draw | E* = $(round(Int, b.Estar)) keV, β = $(round(b.β, sigdigits = 3)) |
        | solid water at $(join(Int.(r.energies), " / ")) keV (theory 0) | $(join(f.(r.water_mean), " / ")) HU |
        | noise σ at $(join(Int.(r.energies), " / ")) keV | $(join(f.(r.water_noise), " / ")) HU |
        | largest mean rod error against theory (calcium, 50 keV: the basis) | $(f(rod_error)) HU |
        | checks passed | $(count(check -> check.pass, checks)) of $(length(checks)) |
        """,
    )
end

# ╔═╡ Cell order:
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─b9c6ce63-074d-410b-90d6-ab9bd39ea84b
# ╟─f2798d62-3509-4cc4-a24f-39ace8bb5a9e
# ╟─14168d18-e05c-4ed7-b5c9-f2c15491653a
# ╟─3d515abe-f3d9-4ce5-96c7-bef7da9bf294
# ╠═171294a2-26bd-49e2-ac92-9df48ae5444f
# ╠═69358294-97f2-4782-94d7-c29c747c45f4
# ╠═9ae27110-5c47-442b-a98e-d137599570f2
# ╠═492bb299-678d-4e6f-8c21-1e9178cc2beb
# ╠═9f8d5cd4-147e-4359-95bc-cc096a53f0e7
# ╠═2ff539c9-a678-403c-b629-8068a332a0e9
# ╠═320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
# ╠═86c52e9e-7987-4504-93e6-128017f5e703
# ╟─551f84fe-d7b4-48f9-a475-0c63178a6ede
# ╟─796ba102-7406-414c-8994-a8c86f6564e8
# ╠═eb6ef3f6-324f-4737-a96c-0cbce838156d
# ╠═1b9d059f-6c05-4215-91f3-75a70fc9d5b3
# ╠═b3a56cb6-3ec3-4f0f-9004-a94d0d77468a
# ╠═cba974a0-17fa-40c6-8e84-28268204a721
# ╠═88a25e41-aa75-44ba-8d85-1e57e2ed0255
# ╠═9dfe5884-b920-413b-a408-11797ab58dae
# ╠═d6175803-7b56-49f0-8939-8c4e0ac85345
# ╟─040e1000-0000-4000-8000-000000000100
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╠═c91576b0-73ba-4359-a29f-aae918b6344f
# ╟─040e1000-0000-4000-8000-000000000101
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╟─eeed376c-8d0d-52ba-b48e-715ba6d3123e
# ╟─040e1000-0000-4000-8000-000000000102
# ╠═03c00001-0000-4000-8000-000000000001
# ╠═03c00001-0000-4000-8000-000000000002
# ╠═03c00001-0000-4000-8000-000000000003
# ╠═03c00001-0000-4000-8000-000000000004
# ╟─03c00001-0000-4000-8000-000000000005
# ╟─040e1000-0000-4000-8000-000000000103
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═03c00001-0000-4000-8000-000000000006
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═90ce0f86-1464-42a8-b856-cc5487a4e4c1
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─03c00001-0000-4000-8000-000000000007
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═4ca28c64-ee96-47c8-b7c3-f0e0c4c99423
# ╟─03a10001-0000-4000-8000-000000000001
# ╟─03a10001-0000-4000-8000-000000000002
# ╟─03a10001-0000-4000-8000-000000000003
# ╠═03a10001-0000-4000-8000-000000000004
# ╠═03a10001-0000-4000-8000-000000000005
# ╠═03a10001-0000-4000-8000-000000000006
# ╠═03c00001-0000-4000-8000-000000000008
# ╟─03c00001-0000-4000-8000-000000000009
# ╟─03c00001-0000-4000-8000-00000000000a
# ╠═03c00001-0000-4000-8000-00000000000b
# ╟─040e1000-0000-4000-8000-000000000003
# ╠═040e0002-0000-4000-8000-000000000002
# ╟─03c00001-0000-4000-8000-00000000000c
# ╠═03c00001-0000-4000-8000-00000000000d
# ╟─03a10001-0000-4000-8000-000000000007
# ╟─040e0002-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000003
# ╟─040e0001-0000-4000-8000-000000000004
# ╟─040e0001-0000-4000-8000-000000000008
# ╠═040e0001-0000-4000-8000-000000000002
# ╟─040e1000-0000-4000-8000-000000000040
# ╟─040e0001-0000-4000-8000-000000000005
# ╟─040e1000-0000-4000-8000-000000000050
# ╟─040e1000-0000-4000-8000-000000000051
# ╟─040e1000-0000-4000-8000-000000000060
# ╟─040e0001-0000-4000-8000-000000000006
# ╟─03c00001-0000-4000-8000-00000000000e
# ╟─03c00001-0000-4000-8000-00000000000f
# ╟─040e1000-0000-4000-8000-000000000070
# ╟─040e0001-0000-4000-8000-000000000007
# ╠═040e0002-0000-4000-8000-000000000004
# ╟─12d92a54-471f-4018-9822-610113e19f27
# ╟─040e0002-0000-4000-8000-000000000006
