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
using Statistics: mean, std, median

# ╔═╡ d3054785-9e00-4094-a491-088ce63be9dc
md"""
# Photon-Counting CT Virtual Monoenergetic Imaging

**A Siemens NAEOTOM Alpha scan of the Gammex 472, its four CdTe energy bins decomposed together,
becomes virtual monoenergetic images (VMIs) at 40, 70, 100 and 140 keV, checked rod by rod against
theory.**

The scanner is the Alpha model of the basis-spectral-denoising study: a CdTe detector that sorts
every photon into four energy windows (thresholds 20 / 35 / 55 / 70 keV), a 140 kVp, 0.5 s axial
acquisition at the dose of the physical Gammex scan, every view integrated over the arc the gantry
turns while it is read, and that study's denoiser, reconstruction windows and spectral pair.
"""

# ╔═╡ 6b217424-7795-4e52-b0de-056b643a12c2
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scanner", "Counting Into Energy Bins"],
        [
            "Gammex 472: Calcium and Iodine Rods",
            "Siemens NAEOTOM Alpha, CdTe Detector",
            "Four Thresholds: $(join(Int.(scanner.energy_thresholds), ", ")) keV",
        ];
        limits = [(-18, 18, -18, 18), (-7.5, 7.5, -7.5, 7.5), (-0.5, 13.8, -1.6, 11.2)],
    )
    gammex!(axes[1], GAMMEX; dark = true)
    gantry!(axes[2]; labels = ("$(Int(protocol.kVp)) kVp",))
    counting_dexel!(
        axes[3];
        thresholds = Tuple(Int.(scanner.energy_thresholds)),
        pulse = false,
        dark = true,
    )
    fig
end

# ╔═╡ f2798d62-3509-4cc4-a24f-39ace8bb5a9e
md"""
## The Chain at a Glance

The phantom is projected once; the noise-free reference, a calibration draw and the measured draw
all reuse that projection. Each `simulate!` returns the four corrected energy-bin sinograms and the
air count of every ray in every bin, `I0[col, row, bin]`, from which `spectral_basis(ws; I0)` builds
the model the simulation applied. All four bins then enter one decomposition as separate
measurements; nothing is merged into "low" and "high".
"""

# ╔═╡ a0a37858-dd02-485c-91ea-98df6b7d8345
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 330))
    nodes = Dict(
        :bins => (0.0, 2.2, "Four Bin Counts", "Measured Draw", BIN_COLORS[3]),
        :hypr => (4.4, 2.2, "Projection HYPR", "3 × 3 in Each Row", COLORS.cyan),
        :decompose => (8.8, 2.2, "Decomposition", "K = 4, Every Ray", COLORS.cyan),
        :fdk => (13.2, 2.2, "FDK Pair", "A Window Each", COLORS.cyan),
        :acnr => (17.6, 2.2, "ACNR + HYPR", "On the Complement", COLORS.cyan),
        :vmi => (22.0, 2.2, "VMIs", "40 to 140 keV", COLORS.green),
        :basis => (8.8, 0.0, "Spectral Basis", "Per Ray, From I₀", COLORS.violet),
        :calibration => (13.2, 0.0, "Calibration Draw", "E*, β", COLORS.purple),
    )
    edges = [
        :bins => :hypr,
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
import CairoMakie as Mke

# ╔═╡ 320e1b29-4ae3-4757-a2cb-d28b0aa3ec2d
PlutoUI.TableOfContents()

# ╔═╡ 86c52e9e-7987-4504-93e6-128017f5e703
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end

# ╔═╡ 551f84fe-d7b4-48f9-a475-0c63178a6ede
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ 3374b87e-9e38-410f-8cf2-f1a78439f3ef
md"""
## Figure Style

The colours, the theme and the drawing helpers every BasisSimulator notebook shares, then this
notebook's own drawings: the Gammex 472, drawn from its label mask, a photon-counting dexel and the
bowtie's fan.
"""

# ╔═╡ 42f1aeb4-0d26-4c6a-87ff-765a6db5b7a4
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

# ╔═╡ 63410694-a179-4c1e-a15e-e256c89be1b6
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

# ╔═╡ 49d1d4f5-964f-41a2-9a65-1fd1ae7feeac
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

# ╔═╡ ff468aa7-9173-4b37-b4db-6aaba2e762f2
# the four energy bins' colours, from the lowest bin to the highest
BIN_COLORS = [COLORS.violet, COLORS.blue, COLORS.cyan, COLORS.green];

# ╔═╡ 5615a564-a130-48cb-abc2-3f42996da720
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

# ╔═╡ 80ab6ea4-0d34-49b8-9405-1a0c8aefc5b1
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

# ╔═╡ 35f29bd8-7078-4f38-a80e-4eca5b40bfed
"""
A photon-counting dexel at work, on `ax` (a canvas about `(-0.5, 25, -1.6, 10.4)`): a photon of
`energy` keV stops in the CdTe, its charge drifts to the dexel's anode, and the pulse it makes is
compared with every threshold of `thresholds` (keV): the highest it crosses decides the bin
whose counter goes up by one.
"""
function counting_dexel!(
    ax;
    energy = 62.0,
    thresholds = (20, 35, 55, 70),
    pulse = true,
    dark = false,
)
    ink = dark ? COLORS.night_ink : COLORS.ink
    muted = dark ? COLORS.night_muted : COLORS.muted
    card = dark ? COLORS.night_card : COLORS.card
    label(
        x,
        y,
        text;
        align = (:center, :center),
        size = 16,
        bold = false,
        color = ink,
    ) = Mke.text!(
        ax,
        x,
        y;
        text,
        align,
        fontsize = size,
        font = bold ? :bold : :regular,
        color,
    )

    # the dexel: CdTe between the cathode and its pixel anode
    Mke.poly!(
        ax,
        Mke.Rect2f(0.4, 1.8, 4.2, 5.4);
        color = (COLORS.violet_soft, dark ? 0.35 : 1.0),
    )
    Mke.poly!(ax, Mke.Rect2f(0.4, 7.2, 4.2, 0.35); color = ink)
    Mke.poly!(ax, Mke.Rect2f(1.0, 1.45, 3.0, 0.35); color = COLORS.cyan)
    label(1.1, 8.0, "Cathode"; size = 15, color = muted)
    label(2.5, 0.95, "Dexel Anode"; size = 15, color = muted)
    label(1.5, 2.6, "CdTe"; bold = true)

    # the photon, its stop and its charge drifting down
    zig = [
        Mke.Point2f(2.5 + 0.22 * sin(4π * t), 10.2 - 4.6 * t)
        for t in range(0, 1; length = 80)
    ]
    Mke.lines!(ax, zig; color = COLORS.beam, linewidth = 3)
    Mke.scatter!(
        ax,
        [Mke.Point2f(2.5, 5.6)];
        marker = :star5,
        markersize = 24,
        color = COLORS.beam,
    )
    for (dx, dy) in (
            (-0.3, -0.7),
            (0.25, -0.9),
            (-0.05, -1.25),
            (0.35, -1.45),
            (-0.35, -1.6),
        )
        Mke.scatter!(
            ax,
            [Mke.Point2f(2.5 + dx, 5.6 + dy)];
            markersize = 8,
            color = COLORS.cyan,
        )
    end
    arrow!(ax, (3.7, 4.6), (3.7, 2.1); color = COLORS.cyan, linewidth = 2, head = 12)
    label(
        2.5,
        10.25,
        "$(round(Int, energy)) keV Photon";
        align = (:center, :bottom),
        bold = true,
    )

    # the pulse against the thresholds (or, without `pulse`, straight to the counters)
    hit = findlast(t -> energy >= t, collect(thresholds))
    counters = pulse ? 21.6 : 9.4
    pulse ||
        arrow!(ax, (4.9, 4.4), (counters - 2.75, 1.6 + 1.75 * (hit - 1)); color = muted)
    pulse && draw_pulse!(ax, energy, thresholds; ink, muted, label)

    # the counters, one per bin
    for (k, threshold) in enumerate(thresholds)
        upper = k < length(thresholds) ? "–$(thresholds[k + 1])" : "+"
        node!(
            ax,
            (counters, 1.6 + 1.75 * (k - 1)),
            "Bin $(k): $(threshold)$(upper) keV";
            w = 5.2,
            h = 1.4,
            accent = BIN_COLORS[k],
            fill = k == hit ? (BIN_COLORS[k], 0.25) : card,
            textcolor = ink,
            titlesize = 16,
        )
    end
    Mke.text!(
        ax,
        counters + 2.8,
        1.6 + 1.75 * (hit - 1);
        text = "+1",
        align = (:left, :center),
        fontsize = 20,
        font = :bold,
        color = BIN_COLORS[hit],
    )
    return ax
end

# ╔═╡ 3a2e9afe-2f6d-41b1-a24d-81e610177f1d
"The pulse of `counting_dexel!` against its thresholds, and the arrow to the bin it counts in."
function draw_pulse!(ax, energy, thresholds; ink, muted, label)
    hit = findlast(t -> energy >= t, collect(thresholds))
    arrow!(ax, (4.6, 1.6), (6.3, 1.6); color = muted)
    keV(e) = 1.6 + 6.6 * e / 80
    arrow!(ax, (6.6, 1.6), (14.6, 1.6); color = muted, head = 12)
    label(14.6, 1.2, "Time"; align = (:right, :top), size = 15, color = muted)
    pulse = [
        Mke.Point2f(6.8 + 7.4 * t, keV(energy * (t / 0.3) * exp(1 - t / 0.3)))
        for t in range(0, 1; length = 120)
    ]
    for (k, threshold) in enumerate(thresholds)
        crossed = energy >= threshold
        Mke.lines!(
            ax,
            [6.6, 14.6],
            [keV(threshold), keV(threshold)];
            color = BIN_COLORS[k],
            linewidth = 2,
            linestyle = :dash,
        )
        label(
            14.8,
            keV(threshold),
            "$(threshold) keV";
            align = (:left, :center),
            size = 15,
            color = crossed ? BIN_COLORS[k] : muted,
        )
    end
    Mke.lines!(ax, pulse; color = ink, linewidth = 3.5)
    label(
        10.3,
        9.7,
        "Pulse Height ∝ Energy";
        align = (:center, :bottom),
        size = 16,
        color = muted,
    )

    arrow!(ax, (17.4, keV(energy) - 0.3), (18.8, 1.6 + 1.75 * (hit - 1)); color = muted)
    return nothing
end

# ╔═╡ 5bb4711f-b473-4fd6-bc4e-f530cf3d63df
"""
A tube, its bowtie filter and the fan it shapes, on `ax` (a canvas about `(-6.5, 6.5, -4.6,
7.6)`): the central ray crosses the thin middle of the bowtie, the edge ray its thick flank, so
the fan's edge gets fewer and harder photons.
"""
function bowtie_fan!(ax; dark = false)
    ink = dark ? COLORS.night_ink : COLORS.ink
    muted = dark ? COLORS.night_muted : COLORS.muted
    source = Mke.Point2f(0, 6.4)
    half = deg2rad(25)
    edge = [
        source + 10.0f0 * Mke.Point2f(sin(φ), -cos(φ))
        for φ in range(-half, half; length = 50)
    ]
    Mke.poly!(ax, vcat([source], edge); color = (COLORS.beam, 0.18), strokewidth = 0)
    Mke.lines!(ax, edge; color = COLORS.cyan, linewidth = 9)
    # the bowtie: thin at the middle, thick at the flanks
    xs = range(-2.2, 2.2; length = 40)
    top = [Mke.Point2f(x, 5.0) for x in xs]
    bottom = [Mke.Point2f(x, 4.75 - 1.1 * (x / 2.2)^2) for x in reverse(xs)]
    Mke.poly!(
        ax,
        vcat(top, bottom);
        color = (COLORS.muted, 0.55),
        strokecolor = ink,
        strokewidth = 1.5,
    )
    Mke.text!(
        ax,
        2.5,
        4.6;
        text = "Bowtie",
        align = (:left, :center),
        fontsize = 16,
        color = muted,
    )
    Mke.scatter!(ax, [source]; marker = :rect, markersize = 26, color = ink)
    Mke.scatter!(ax, [source]; marker = :circle, markersize = 9, color = COLORS.beam)
    Mke.text!(
        ax,
        0,
        7.2;
        text = "Tube",
        align = (:center, :bottom),
        fontsize = 16,
        font = :bold,
        color = ink,
    )
    for (φ, name, style) in ((0.0, "Centre", :solid), (0.92half, "Fan Edge", :dash))
        tip = source + 10.0f0 * Mke.Point2f(sin(φ), -cos(φ))
        Mke.lines!(ax, [source, tip]; color = ink, linewidth = 2.5, linestyle = style)
        Mke.text!(
            ax,
            tip + Mke.Point2f(0, -0.6);
            text = name,
            align = (:center, :top),
            fontsize = 16,
            font = :bold,
            color = ink,
        )
    end
    return ax
end

# ╔═╡ 040e1000-0000-4000-8000-000000000100
md"""
## 1. The Phantom

The Gammex 472: a 33 cm solid-water body with seven calcium rods (50 to 600 mg/mL) on an inner
ring and seven iodine rods (2 to 20 mg/mL) on an outer ring, 28 mm each. The label mask (512 × 512
at 35 cm, 16 slices over 1 cm) goes to the GPU; its material table stays on the host.
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

# ╔═╡ 203181f5-2f7e-4201-a02e-757682b5f47d
# where the body and every rod are, for the drawings
GAMMEX = gammex_layout(phantom_cpu);

# ╔═╡ 040e1000-0000-4000-8000-000000000101
md"""
## 2. The Scanner

The documented NAEOTOM Alpha, `NAEOTOM_ALPHA`, in its standard mode: native CdTe dexels at a
0.34 × 0.364 mm pitch summed digitally 2 × 2 after counting (1376 columns of 0.373 mm and 144
rows of 0.399 mm at the isocentre), an arc detector with the quarter-detector offset, a 50 cm scan
field behind the large-body bowtie, and a 1.6 mm CdTe sensor whose response (absorption, charge
sharing, electronic noise, the anti-scatter grid) is BasisMonteCarlo's Geant4 table. See the
[scanners page](../../scanners/) for where each value comes from.

<details><summary><strong>A realistic unit: its defects and their corrections</strong></summary>

Each native dexel counts on its own, Poisson, sharing photons with its neighbours, and piles up at
its own rate (27 ns dead time). Its native dexels' gains spread by 2 % and their thresholds by
0.35 keV, a fixed pattern that would ring the image uncorrected. Scatter (the Geant4 library's
kernels for the Alpha) is simulated inside `simulate!` and corrected there from the recorded counts
alone; pile-up and then the flat field are corrected after it, each with a calibration from
simulated air scans.

</details>
"""

# ╔═╡ 2c157064-8567-450b-bc08-c2606084a77f
scanner = BS.NAEOTOM_ALPHA

# ╔═╡ d15334a3-db15-541c-b1a3-ab862db955f4
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

# ╔═╡ 040e1000-0000-4000-8000-000000000102
md"""
## 3. Protocol and Dose

140 kVp, 0.5 s rotation, 1200 views, 5 mm collimation, 0.9 mm titanium added: the physical Gammex
scan, which ran at 174 mA and reported a CTDIvol of 10.12 mGy. A tube's output per mA is its own
and the simulator's source is generic, so one factor, `DOSE_SCALE`, scales the tube current until
the simulated CTDIvol (BasisSimulator's Monte Carlo dose in the 32 cm body phantom) equals the
physical scan's. The noise is then that of the dose the physical scan delivered.
"""

# ╔═╡ 04b00000-0000-4000-8000-000000000001
"The NAEOTOM Alpha's 140 kVp axial protocol at tube current `mA`."
alpha_protocol(mA) = BS.CTProtocol(
    kVp = 140,
    mA = mA,
    views = 1200,
    rotation_time = 0.5,
    collimation_mm = 5.0,
    additional_filters = [("Ti", 0.9)],
);

# ╔═╡ 04b00000-0000-4000-8000-000000000002
# The tube-current scale that makes the simulated CTDIvol equal the physical scan's
DOSE_SCALE = let physical = (mA = 174.0, ctdi_vol_mGy = 10.12)
    p = alpha_protocol(physical.mA)
    physical.ctdi_vol_mGy / BS.compute_dose(BS.dose_source(scanner, p), p).ctdi_vol_mGy
end

# ╔═╡ f9c0af7a-addd-4249-96fb-b9078765fbd1
protocol = alpha_protocol(174.0 * DOSE_SCALE);

# ╔═╡ 040e1000-0000-4000-8000-000000000103
md"""
## 4. Physics, Draws and the Image Grid

The physics switches of the basis-spectral-denoising Alpha model. Quantum noise, patient scatter
(the Alpha's Geant4 kernels, superposed on the phantom's own rays) and the focal-spot blur are on.
The switches that model a scintillator (fill factor, optical crosstalk, the scintillator efficiency
table, lag) are off: the CdTe detector's efficiency, fill and charge sharing are already in its
Monte Carlo response. The heel effect is off.

Everything before the noise (the projection through the phantom, sub-views included) is the same
for every draw, so it is computed once, on the noise-free draw (`keep_projection = true`), and every
other draw reuses it (`projection`) with only its own noise drawn, bit-identical to simulating each
afresh.

| Draw | Noise | Seed | Used for |
|---|---|---:|---|
| `reference` | off | — | the noise-free expectation every result is compared with |
| `calibration` | on | 9234 | fixing the spectral pair, never measured |
| `measured` | on | 1234 | every result below |

The reconstruction grid is the physical scan's clinical series: 512 × 512 over 35 cm, twelve 0.4 mm
slices.

<details><summary><strong>Why <code>view_samples = 5</code> and <code>view_arc = 1.0</code>: view integration</strong></summary>

A detector integrates while the gantry turns, so each view reads the transmitted intensity averaged
over the arc it sweeps: a blur of the object, largest far from the isocentre, that the noise,
counted once per view, does not share. `view_samples = 5` samples that arc with five sub-views per
view (the midpoint rule; at the edge of a 35 cm field they are 0.2 mm apart, a third of a pixel),
and `view_arc = 1.0` because a photon-counting detector reads for the whole view period. Without it
the simulated signal is sharper, relative to its noise, than a physical scanner's.

</details>
"""

# ╔═╡ 2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
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

# ╔═╡ 04b00000-0000-4000-8000-000000000003
# basis-spectral-denoising's seeds: draw r of the Gammex has seed 1234 + 1000 (r − 1); its
# calibration draw is draw 9
DRAWS = (
    reference = (seed = 1234, use_noise = false),
    calibration = (seed = 1234 + 1000 * (9 - 1), use_noise = true),
    measured = (seed = 1234, use_noise = true),
);

# ╔═╡ 08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
recon_opts = BS.ReconOptions(
    matrix_size = (512, 512, 12),
    fov_cm = 35.0,
    z_cm = 12 * 0.04, # twelve 0.4 mm slices
);

# ╔═╡ 5ecd97c6-ad47-4558-886d-22ed45eda97d
md"""
## 5. Simulate: `simulate!` and `spectral_basis`

`simulate!` runs the whole photon-counting detector model on the GPU: polychromatic forward
projection of every sub-view through the Monte Carlo detector response, the patient scatter, the
counts of every native dexel (Poisson, neighbours covarying through the photons they share), its
pile-up and the digital 2 × 2 sum. It then corrects the counts in a scanner's order, with the
workspace's calibration (`calibrate(scanner, protocol, recon_opts, phantom)`, made once from
simulated scans, never from the model's own matrices). It returns

- `pcct_sino.bins`: four corrected sinograms ``h_k = -\log(y_k / I_{0,k})``, `[col, row, view]`;
- `I0_bins`: the air count of every ray in every bin, `[col, row, bin]`;
- `projection`, with `keep_projection = true`: everything before the noise, for the next draw;
- `dose`: the `DoseReport` of the beam.

`spectral_basis(ws; I0)` is read from the same workspace: the response the simulation applied,
passed through each ray's own bowtie transmission, so the decomposition inverts exactly what was
simulated and needs no calibration scan. It checks that the response sums to `I0` in every ray and
bin, and refuses a basis that does not. It does not depend on the noise, so one basis serves every
draw.

<details><summary><strong>The order of the corrections, and why</strong></summary>

First the pile-up: scans of air and two water cylinders at rising tube currents give each ray's
pile-up-free counts from its recorded counts in every bin, its rate and its spectrum. Then the
dexels' gains: low-flux scans of air and of water slabs 5–50 cm thick, each dexel's against the
nominal detector's (its gain in every bin as the patient hardens the beam). Last the scatter,
estimated from the corrected counts, the patient taken as water: the estimate is a nominal
dexel's, and subtracted before the other two corrections it would leave each dexel's gain error on
its scatter, as rings.

</details>
"""

# ╔═╡ 4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
acquisition = let
    projection = nothing
    model = nothing
    channels = Dict{Symbol, Vector{Array{Float32, 3}}}()
    seconds = Dict{Symbol, Float64}()
    # the scanner's calibration, as a scanner is calibrated: the pile-up from air and water scans at
    # rising currents; the flat field from low-flux scans of air and water slabs (each dexel's gain
    # per bin, and how it changes as the patient hardens the beam)
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
            # the dose and the projection do not depend on the noise: both come from the first draw;
            # simulate! corrects in a scanner's order: pile-up, flat field, then the scatter
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

# ╔═╡ 04a10000-0000-4000-8000-000000000001
let
    a = acquisition
    nc, nr, K = size(a.I0)
    c, r = (nc + 1) ÷ 2, (nr + 1) ÷ 2
    centre = round.(Int, a.I0[c, r, :])
    edge = [a.I0[1, r, k] / a.I0[c, r, k] for k in 1:K]
    fmt(x) = string(round(x; sigdigits = 3))
    s(name) = string(round(a.seconds[name]; digits = 1), " s")
    Markdown.parse(
        """
        | Quantity | Value |
        |---|---|
        | sinogram per bin | $(nc) columns × $(nr) rows × $(size(first(a.channels[:measured]), 3)) views |
        | air counts per ray and view, centre ray (bins 1 to 4) | $(join(centre, " / ")) |
        | air counts at the fan edge / centre (bins 1 to 4) | $(join(fmt.(edge), " / ")) |
        | spectral basis | $(size(a.basis.Φ, 3)) energies, ray-resolved = $(a.basis.ray_resolved) |
        | largest relative mismatch of Σ Φ and I0 | $(fmt(a.basis.I0_relerr)) |
        | tube current | $(round(protocol.mA; digits = 1)) mA (174 mA × DOSE_SCALE = $(round(DOSE_SCALE; digits = 3))) |
        | CTDIvol (32 cm body phantom) | $(round(a.dose.ctdi_vol_mGy; digits = 2)) mGy |
        | wall time: noise-free draw (projects) / calibration / measured | $(s(:reference)) / $(s(:calibration)) / $(s(:measured)) |
        """,
    )
end

# ╔═╡ 04a10000-0000-4000-8000-000000000002
md"""
## 6. The Four Energy Windows

A photon stops in a dexel's CdTe and frees a charge that drifts to the dexel's anode; the pulse it
makes is as tall as the photon's energy, and the highest threshold it crosses decides the bin that
counts it (top left). Bottom left: what each bin detects on the central ray, the absolute response
the decomposition uses, against the energy the photon arrived with. The windows overlap because
charge sharing, electronic noise and K-fluorescence escape let a photon register below its true
energy; the spikes are tungsten's characteristic lines.

The bowtie (top right) is thin at the middle of the fan and thick at its edges: it cuts the flux
toward the edge and hardens what is left, so the high bins fall off less than the low ones (bottom
right). One scalar `I0` per bin could not describe this, which is why the basis is resolved per
ray.
"""

# ╔═╡ 04a10000-0000-4000-8000-000000000003
let
    Mke.set_theme!(THEME)
    a = acquisition
    nc, nr, K = size(a.I0)
    c, r = (nc + 1) ÷ 2, (nr + 1) ÷ 2
    E = Float64.(a.basis.E)
    dE = median(diff(E))
    t = Int.(scanner.energy_thresholds)
    labels = [
        k < K ? "Bin $(k): $(t[k])–$(t[k + 1]) keV" : "Bin $(k): Above $(t[k]) keV"
        for k in 1:K
    ]
    fig = Mke.Figure(size = (1400, 1020))

    # how a dexel sorts photons, and what the bowtie does to the fan
    dexel = canvas(fig[1, 1], (-0.5, 26.5, -1.6, 11.2))
    counting_dexel!(dexel; thresholds = Tuple(t))
    fan = canvas(fig[1, 2], (-6.5, 6.5, -5.4, 7.8))
    bowtie_fan!(fan)

    ax1 = Mke.Axis(
        fig[2, 1];
        title = "What Each Bin Counts",
        subtitle = "Central Ray, One View",
        xlabel = "Incident Photon Energy (keV)",
        ylabel = "Counts per keV of Incident Energy",
    )
    for k in 1:K
        Mke.lines!(
            ax1,
            E,
            Float64.(a.basis.Φ[c, r, :, k]) ./ dE;
            color = BIN_COLORS[k],
            linewidth = 2.5,
            label = labels[k],
        )
    end
    Mke.axislegend(ax1; position = :rt)

    γ = ((1:nc) .- (nc + 1) / 2) .* a.geom.pixel_size ./ a.geom.SAD .* (180 / π)
    ax2 = Mke.Axis(
        fig[2, 2];
        title = "Air Count Across the Fan",
        subtitle = "Relative to the Central Ray",
        xlabel = "Fan Angle (Degrees)",
        ylabel = "I₀ / I₀(Centre)",
    )
    for k in 1:K
        Mke.lines!(
            ax2,
            γ,
            Float64.(a.I0[:, r, k]) ./ a.I0[c, r, k];
            color = BIN_COLORS[k],
            linewidth = 2.5,
            label = "Bin $(k)",
        )
    end
    Mke.ylims!(ax2, 0, 1.05)
    Mke.axislegend(ax2; position = :cb, nbanks = 2)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.6))
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.42))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_energy_windows.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ dc8a8352-5598-4cdd-952f-3d77367850e9
md"""
## 7. The Chain's Settings

**The denoiser** is generalized HYPR-LR in both domains, `BS.SpectralHYPR()`, the chain of
basis-spectral-denoising:

- **projection domain** (`ProjectionHYPR`): within each detector row, each ray's split of its total
  count across the four bins is pooled over a 3 × 3 (column × view) window whose view neighbours are
  two views apart (`view_stride = 2`), so the odd and even views stay independent. Neighbours are
  weighted by how likely their total count is under the ray's own, and each ray keeps its own total.
  The count dispersion of every bin is measured from the air rays.
- **image domain** (`ImageHYPR`): the reconstructed basis pair is rewritten as the minimum-noise VMI
  (the composite) and a complement whose noise is uncorrelated with it. The composite is kept as
  reconstructed, so its resolution and noise texture are FDK's; the complement, which carries only
  spectral information, is pooled within its slice with likelihood weights on the composite, over
  the window width (3 to 41 pixels) with the least estimated risk. Every noise level comes from the
  difference of the odd- and even-view reconstructions.

Nothing in the chain averages detector rows or slices.
"""

# ╔═╡ 04a20000-0000-4000-8000-000000000001
HYPR_CHAIN = BS.SpectralHYPR()

# ╔═╡ 04a20000-0000-4000-8000-000000000002
VMI_ENERGIES = (40, 70, 100, 140);

# ╔═╡ 04a20000-0000-4000-8000-000000000006
# The decomposition and reconstruction settings of the chain; ACNR is added on the measured draw
VMI_CHAIN = (
    method = :nchannel,
    controls = BS.NChannelControls(),
    use_tlbf = false,
    antialias = true,
);

# ╔═╡ 04b00000-0000-4000-8000-000000000010
md"""
**The reconstruction windows.** `BS.PairFilter` filters the composite, whose window sets the
resolution and noise texture at the minimum-noise energy, and the complement, whose window sets
how that changes away from it. Each window is an apodized ramp, ``W(f) = \exp(-(f/f_c)^p)`` on the
grid-Nyquist axis, sampled at 11 knots as a `CustomFilter`. basis-spectral-denoising fitted the
Alpha's pair once, to the physical Gammex scan's MTF at every energy and its NPS shape (VMI, Br36f,
QIR 0, 0.4 mm, 350 mm): composite ``f_c = 0.5``, ``p = 2``; complement ``f_c = 0.45``, ``p = 6``.

!!! note "Why the composite and the complement, not water and iodine?"
    The water and iodine images' noise is strongly anti-correlated, and a VMI near the
    minimum-noise energy is quiet only where both were filtered alike. The composite and its
    complement have uncorrelated noise, so each can have its own window.

A window is defined on the frequency axis of the grid it is applied on, so on a grid other than the
fitted one (512² over 35 cm) its argument is rescaled by the ratio `r` of the grids' bandlimits
(`BS.grid_bandlimit`). This notebook uses the fitted grid, so `r = 1`.
"""

# ╔═╡ 04b00000-0000-4000-8000-000000000011
ALPHA_WINDOWS = (composite = (fc = 0.5, p = 2.0), complement = (fc = 0.45, p = 6.0));

# ╔═╡ 04b00000-0000-4000-8000-000000000012
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

# ╔═╡ 04b00000-0000-4000-8000-000000000020
md"""
## 8. The Spectral Pair, Fixed from the Calibration Draw

`spectral_pair` measures, from the odd- and even-view halves of a reconstruction, the
minimum-noise energy ``E^*`` and the complement's coefficient ``\beta``. Near its minimum the VMI
noise hardly changes with energy, so the argmin of one acquisition is itself noise and would move
from draw to draw; it is a property of the scanner and protocol, so it is measured once, on the
calibration draw, and fixed for every other draw, the noise-free one included:

- `pair_basis = (Estar, β)`: from the calibration draw decomposed without denoising and
  reconstructed with a standard soft-tissue window (`SoftFilter`); the pair the windows act on;
- `composite_energy`: the minimum-noise energy of the same draw after the projection-domain HYPR,
  which moves it; the composite ACNR and the image HYPR act on.
"""

# ╔═╡ 04b00000-0000-4000-8000-000000000021
PAIR_BASIS = let ch = acquisition.channels[:calibration], I0 = acquisition.I0
    decompose(channels) = BS.vmi_pipeline(;
        channels,
        basis = acquisition.basis,
        geom = acquisition.geom,
        to_backend = to_gpu,
        matrix_size = recon_opts.matrix_size,
        vmi_energies = VMI_ENERGIES,
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

# ╔═╡ 04a20000-0000-4000-8000-000000000003
md"""
## 9. One Call: `vmi_pipeline`

One call runs the whole chain on the measured draw. Every detector row is kept and reconstructed
onto the twelve-slice grid; the decomposition is the K-channel maximum-likelihood estimator
(`method = :nchannel`) with its published controls; ACNR (on by default) acts on the complement
only, before the image HYPR. The noise-free draw goes through the same call without the denoiser
and ACNR: the reference, reconstructed exactly as its noisy counterpart.
"""

# ╔═╡ 04a20000-0000-4000-8000-000000000004
vmi = BS.vmi_pipeline(;
    channels = acquisition.channels[:measured],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = VMI_ENERGIES,
    denoiser = HYPR_CHAIN,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    VMI_CHAIN...,
);

# ╔═╡ 04b00000-0000-4000-8000-000000000022
reference = BS.vmi_pipeline(;
    channels = acquisition.channels[:reference],
    basis = acquisition.basis,
    geom = acquisition.geom,
    to_backend = to_gpu,
    matrix_size = recon_opts.matrix_size,
    vmi_energies = VMI_ENERGIES,
    fbp_filter = PAIR_FILTER.filter,
    pair_basis = (Estar = PAIR_BASIS.Estar, β = PAIR_BASIS.β),
    composite_energy = PAIR_BASIS.composite,
    use_acnr = false,
    VMI_CHAIN...,
);

# ╔═╡ 04a20000-0000-4000-8000-000000000005
let
    s = vmi.settings
    q = vmi.quality
    pct(x) = string(round(100x; digits = 3), " %")
    est = s.denoiser.image_estimates
    Markdown.parse(
        """
        What the chain fixed and measured:

        | Quantity | Value |
        |---|---|
        | channels decomposed | $(s.n_channels) |
        | count dispersion per bin (variance / mean, from the air rays) | $(join(string.(round.(s.denoiser.dispersion; digits = 3)), " / ")) |
        | pair the windows act on, from the calibration draw: E* and β | $(round(Int, PAIR_BASIS.Estar)) keV, β = $(round(PAIR_BASIS.β; sigdigits = 3)) |
        | composite energy, from the calibration draw after projection HYPR | $(round(Int, PAIR_BASIS.composite)) keV |
        | window bandlimit ratio r (this grid / the fitted grid) | $(round(PAIR_FILTER.r; digits = 4)) |
        | image HYPR window chosen for the complement | $(est.window) pixels |
        | rays hitting an iodine / water bound | $(pct(q.frac_bound_iodine)) / $(pct(q.frac_bound_water)) |
        | rays not converged | $(pct(q.frac_not_converged)) |
        | ACNR | $(s.acnr.passes) passes on the $(s.acnr.on), beta_max = $(s.acnr.beta_max) |
        | VMI stack | $(join(size(vmi.vmis), " × ")) (x, y, slice, energy) |
        """,
    )
end

# ╔═╡ 040e0002-0000-4000-8000-000000000003
md"""
## 10. Basis Images and VMIs

The chain's basis pair: iodine as a mass density (mg/mL) and water as a density (g/mL), the central
slice. Every VMI is synthesized from this one pair with the monoenergetic attenuation of water and
iodine; no energy-dependent filtering is applied.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000003
let
    Mke.set_theme!(THEME)
    z = (size(vmi.images.water, 3) + 1) ÷ 2
    fig = Mke.Figure(size = (1400, 640))
    panels = (
        (
            "Iodine Basis",
            "Iodine (mg/mL)",
            1000 .* vmi.images.iodine[:, :, z],
            (-5.0, 25.0),
        ),
        ("Water Basis", "Water (g/mL)", vmi.images.water[:, :, z], (0.0, 1.5)),
    )
    for (column, (title, label, image, range)) in pairs(panels)
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
    z = (size(vmi.vmis, 3) + 1) ÷ 2
    window = (-200, 500)
    fig = Mke.Figure(size = (1200, 1180))
    for (index, energy) in pairs(vmi.energies)
        axis = Mke.Axis(
            fig[(index - 1) ÷ 2 + 1, (index - 1) % 2 + 1];
            title = "$(energy) keV VMI",
            aspect = Mke.DataAspect(),
        )
        Mke.heatmap!(axis, vmi.vmis[:, :, z, index]; colormap = :grays, colorrange = window)
        Mke.hidedecorations!(axis)
        Mke.hidespines!(axis)
    end
    Mke.Colorbar(
        fig[1:2, 3];
        colormap = :grays,
        colorrange = window,
        label = "HU",
        width = 16,
    )
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_projection_grid.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e0001-0000-4000-8000-000000000008
md"""
## 11. Accuracy and Noise

The phantom's labels are resampled onto the reconstruction grid (`resample_to_recon`).

| Quantity | How it is measured |
|---|---|
| rod HU | the mean over a disk of 60 % of the rod's radius, all twelve slices |
| water region | the solid water eroded with `erode_mask_2d(…; erode_px = 12)` in every slice: its mask blurred by a 12-pixel Gaussian and kept where the blur stays above 0.999, about 3σ (2.5 cm) clear of the body's edge and of every rod |
| theoretical HU | ``1000\,(\mu_E - \mu_{E,\mathrm{water}})/\mu_{E,\mathrm{water}}`` of every material, from its composition |
| noise-free HU | the same quantity on the noise-free reference: its error against theory is the bias of the basis model itself |
| noise σ | the standard deviation of the measured VMI minus the reference within one slice, RMS over the slices: the phantom's structure and any deterministic artefact cancel |
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000002
pcct_results = let
    energies = Float64.(vmi.energies)
    nz = size(vmi.vmis, 3)
    labels = BS.resample_to_recon(phantom_cpu, acquisition.geom, recon_opts.matrix_size)
    water_mask = cat(
        (
            collect(BS.erode_mask_2d(
                labels[:, :, z] .== UInt8(BS.REGION_SOLID_WATER);
                erode_px = 12.0,
            ))
            for z in 1:nz
        )...;
        dims = 3,
    )
    theory_hu(material, E) = let μw = BS.compute_μ_at_energy(BS.XA.Materials.water, E)
        1000 * (BS.compute_μ_at_energy(material, E) - μw) / μw
    end
    material(label) = phantom_cpu.materials[Int(label) + 1]

    # a disk of 60 % of the rod's radius about its centroid, in every slice
    function rod_roi(label)
        pixels = findall(==(UInt8(label)), labels[:, :, (nz + 1) ÷ 2])
        cx, cy = mean(p -> p[1], pixels), mean(p -> p[2], pixels)
        radius = 0.6 * sqrt(length(pixels) / π)
        [
            (i - cx)^2 + (j - cy)^2 <= radius^2
            for i in axes(labels, 1), j in axes(labels, 2), _ in 1:nz
        ]
    end
    # measured against theoretical at one energy: least-squares slope, intercept and R²
    function fit(x, y)
        slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum(abs2, x .- mean(x))
        intercept = mean(y) - slope * mean(x)
        r2 = 1 - sum(abs2, y .- (intercept .+ slope .* x)) / sum(abs2, y .- mean(y))
        return (; slope, intercept, r2)
    end
    groups = ( # the region labels of the rods, BS.REGION_CA_50 … and BS.REGION_I_2_0 …
        Ca = (
            labels = UInt8.(10:16),
            names = ["50", "100", "200", "300", "400", "500", "600"] .* " mg/mL",
        ),
        I = (
            labels = UInt8.(20:26),
            names = ["2.0", "2.5", "5.0", "7.5", "10.0", "15.0", "20.0"] .* " mg/mL",
        ),
    )
    rods = Dict{Symbol, NamedTuple}()
    for (group, g) in pairs(groups)
        rod_labels = g.labels
        measured = zeros(length(rod_labels), length(energies))
        noise_free = similar(measured)
        theoretical = similar(measured)
        for (row, label) in pairs(rod_labels)
            roi = rod_roi(label)
            for (column, E) in pairs(energies)
                measured[row, column] = mean(vmi.vmis[:, :, :, column][roi])
                noise_free[row, column] = mean(reference.vmis[:, :, :, column][roi])
                theoretical[row, column] = theory_hu(material(label), E)
            end
        end
        fits = [
            fit(theoretical[:, column], measured[:, column])
            for column in axes(measured, 2)
        ]
        rods[group] = (names = g.names, measured, noise_free, theoretical, fits)
    end
    # per-slice standard deviation in the water ROI, RMS over the slices
    slice_noise(volume, column) = sqrt(mean(
        abs2,
        [std(volume[:, :, z, column][water_mask[:, :, z]]) for z in 1:nz],
    ))
    difference = vmi.vmis .- reference.vmis
    (
        energies,
        labels,
        water_mask,
        rods,
        water_mean = [mean(vmi.vmis[:, :, :, c][water_mask]) for c in eachindex(energies)],
        water_reference = [
            mean(reference.vmis[:, :, :, c][water_mask])
            for c in eachindex(energies)
        ],
        water_theory = [theory_hu(material(BS.REGION_SOLID_WATER), E) for E in energies],
        water_noise = [slice_noise(vmi.vmis, c) for c in eachindex(energies)],
        water_noise_vs_reference = [
            slice_noise(difference, c)
            for c in eachindex(energies)
        ],
        water_pixels_per_slice = count(water_mask) ÷ nz,
        finite = all(isfinite, vmi.vmis) && all(isfinite, reference.vmis),
    )
end;

# ╔═╡ 040e1000-0000-4000-8000-000000000040
md"""
### The Solid Water

The eroded solid-water region (left) and its mean HU at each energy, measured (bars) and on the
noise-free reference (diamonds). The phantom's solid water is modelled with the composition of
water, so its theoretical HU is 0 at every energy (dashed line).
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000005
let
    Mke.set_theme!(THEME)
    r = pcct_results
    n = length(r.energies)
    fig = Mke.Figure(size = (1400, 600))

    sketch = canvas(fig[1, 1], (-17.5, 17.5, -17.5, 17.5))
    gammex!(
        sketch,
        GAMMEX;
        show = :water,
        region = pcct_results.water_mask[:, :, (size(pcct_results.water_mask, 3) + 1) ÷ 2],
    )
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
    Mke.barplot!(ax, 1:n, r.water_mean; color = COLORS.cyan, label = "Measured")
    Mke.scatter!(
        ax,
        1:n,
        r.water_reference;
        color = COLORS.ink,
        marker = :diamond,
        markersize = 18,
        label = "Noise-Free Reference",
    )
    Mke.hlines!(
        ax,
        r.water_theory[1:1];
        color = COLORS.ink,
        linestyle = :dash,
        linewidth = 1.5,
    )
    for (index, value) in pairs(r.water_mean)
        Mke.text!(
            ax,
            index,
            value;
            text = "$(round(value; digits = 1)) HU",
            align = (:center, value >= 0 ? :bottom : :top),
            offset = (0, value >= 0 ? 12 : -12),
            fontsize = 16,
        )
    end
    Mke.ylims!(ax, -15, 15)
    Mke.axislegend(ax; position = :lb)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.36))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_water_roi_check.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000050
let
    r = pcct_results
    gap = maximum(abs, r.water_noise .- r.water_noise_vs_reference)
    Markdown.parse(
        """
        ### The Noise

        The noise is what is left when the noise-free reference is subtracted from the measured VMI
        (left, at $(Int(first(r.energies))) keV, in the water region of one 0.4 mm slice); its
        per-slice standard deviation is the noise σ at each energy (right). The standard deviation of
        the measured VMI itself differs from it by at most $(round(gap; digits = 2)) HU at any energy,
        so the ROI holds noise only. The noise is highest at 40 keV, where the synthesis weights the
        iodine image most heavily and so amplifies its noise, and changes little from 70 keV up, where
        the VMI is mostly the composite and the water image.
        """,
    )
end

# ╔═╡ 040e1000-0000-4000-8000-000000000051
let
    Mke.set_theme!(THEME)
    r = pcct_results
    n = length(r.energies)
    σ = r.water_noise_vs_reference
    z = (size(vmi.vmis, 3) + 1) ÷ 2
    difference = vmi.vmis[:, :, z, 1] .- reference.vmis[:, :, z, 1]
    inside = r.water_mask[:, :, z]
    noise_only = [inside[i] ? Float32(difference[i]) : NaN32 for i in eachindex(difference)]
    noise_only = reshape(noise_only, size(difference))
    limit = 3 * first(σ)
    fig = Mke.Figure(size = (1400, 600))

    image = Mke.Axis(
        fig[1, 1];
        title = "Measured − Noise-Free",
        subtitle = "$(Int(first(r.energies))) keV, Central Slice, the Water Region",
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
        title = "Solid-Water Noise vs VMI Energy",
        subtitle = "Per 0.4 mm Slice, $(round(acquisition.dose.ctdi_vol_mGy; digits = 2)) mGy",
        xlabel = "VMI Energy (keV)",
        ylabel = "Noise σ (HU)",
        xticks = (1:n, string.(Int.(r.energies))),
    )
    Mke.barplot!(ax, 1:n, σ; color = COLORS.cyan)
    for index in 1:n
        Mke.text!(
            ax,
            index,
            σ[index];
            text = "σ = $(round(σ[index]; digits = 1))",
            align = (:center, :bottom),
            offset = (0, 6),
            fontsize = 17,
        )
    end
    Mke.ylims!(ax, 0, 1.25maximum(σ))
    Mke.colsize!(fig.layout, 3, Mke.Relative(0.5))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_water_noise_vs_energy.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000060
md"""
### The Rods against Theory

Each rod is measured in the dashed disk drawn on it (top). Solid lines are the measured HU of each
rod, dashed lines its theoretical HU.

The decomposition represents every voxel as iodine plus water. The solid water and the iodine rods
are close to that basis; calcium is not exactly a combination of iodine and water at every energy,
so its VMIs carry a small, energy-dependent model error on top of the noise. The per-rod table
below separates the two with the noise-free reference.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000006
let
    Mke.set_theme!(THEME)
    r = pcct_results
    roi = 0.6 * mean(rod.r for rod in vcat(GAMMEX.Ca, GAMMEX.I))
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
        ),
        (
            group = :I,
            title = "Iodine Rods (mg/mL)",
            shades = Mke.cgrad(
                [Mke.RGBf(0.62, 0.85, 0.52), Mke.RGBf(0.06, 0.32, 0.05)],
                7;
                categorical = true,
            ),
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
            subtitle = "Solid: Measured · Dashed: Theory",
            xlabel = "VMI Energy (keV)",
            ylabel = "HU",
            xticks = r.energies,
        )
        data = r.rods[panel.group]
        for index in eachindex(data.names)
            color = panel.shades[index]
            Mke.scatterlines!(
                axis,
                r.energies,
                data.measured[index, :];
                color,
                linewidth = 2.5,
                markersize = 10,
                label = data.names[index],
            )
            Mke.lines!(
                axis,
                r.energies,
                data.theoretical[index, :];
                color,
                linewidth = 1.6,
                linestyle = :dash,
            )
        end
        Mke.ylims!(axis; low = 0)
        Mke.axislegend(axis; position = :rt, labelsize = 15, rowgap = 2)
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.38))
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_vs_theoretical.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 040e1000-0000-4000-8000-000000000070
md"""
### Linear Regression

The same rods (drawn above), measured HU against theoretical HU at each energy, with the
least-squares line. The dashed line is identity.
"""

# ╔═╡ 040e0001-0000-4000-8000-000000000007
let
    Mke.set_theme!(THEME)
    r = pcct_results
    colors = [COLORS.red, COLORS.beam, COLORS.cyan, COLORS.blue]
    fig = Mke.Figure(size = (1400, 640))
    for (column, group) in enumerate((:Ca, :I))
        data = r.rods[group]
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
        for (index, energy) in pairs(r.energies)
            x = data.theoretical[:, index]
            fit = data.fits[index]
            ends = collect(extrema(x))
            Mke.scatter!(
                axis,
                x,
                data.measured[:, index];
                color = colors[index],
                markersize = 12,
            )
            Mke.lines!(
                axis,
                ends,
                fit.intercept .+ fit.slope .* ends;
                color = colors[index],
                linewidth = 2,
                label = "$(Int(energy)) keV: Slope $(round(fit.slope; digits = 3)), " *
                    "R² $(round(fit.r2; digits = 4))",
            )
        end
        Mke.axislegend(axis; position = :lt, labelsize = 15)
    end
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "pcct_vmi_regression.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 04a30000-0000-4000-8000-000000000001
let
    r = pcct_results
    relative(g, field) = maximum(
        abs,
        (getfield(r.rods[g], field) .- r.rods[g].theoretical) ./ r.rods[g].theoretical,
    )
    pct(x) = round(100x; digits = 1)
    Markdown.parse(
        """
        ### Per-Rod Error

        Measured minus theoretical HU for every rod at every energy, and in brackets the same for the
        noise-free reference: the bias of the iodine–water basis model itself, which no amount of dose
        would remove. The difference between the two is what the noise and the denoiser add. The largest
        relative error of the measured draw is $(pct(relative(:I, :measured))) % over the iodine rods
        and $(pct(relative(:Ca, :measured))) % over the calcium rods; of the noise-free reference,
        $(pct(relative(:I, :noise_free))) % and $(pct(relative(:Ca, :noise_free))) %.
        """,
    )
end

# ╔═╡ 04a30000-0000-4000-8000-000000000002
let
    r = pcct_results
    header = "| Rod | " * join(["$(Int(E)) keV" for E in r.energies], " | ") * " |"
    rule = "|---|" * repeat("---:|", length(r.energies))
    cell(e, b) = "$(round(e; digits = 1)) ($(round(b; digits = 1)))"
    function row(group, name, index)
        data = r.rods[group]
        errors = data.measured[index, :] .- data.theoretical[index, :]
        biases = data.noise_free[index, :] .- data.theoretical[index, :]
        return "| $(name) $(data.names[index]) | " *
            join(cell.(errors, biases), " | ") *
            " |"
    end

    rows = [
        row(group, name, index)
        for (group, name) in ((:Ca, "Ca"), (:I, "I"))
        for index in eachindex(r.rods[group].names)
    ]

    Markdown.parse(join([header, rule, rows...], "\n"))
end

# ╔═╡ 040e0002-0000-4000-8000-000000000004
checks = let
    r = pcct_results
    list = NamedTuple[]
    addcheck(name, value, pass) = push!(list, (; name, value, pass))
    addcheck(
        "spectral basis reproduces I0 in every ray and bin (max rel. error)",
        round(acquisition.basis.I0_relerr; sigdigits = 2),
        acquisition.basis.I0_relerr < 5.0e-5,
    )
    addcheck(
        "simulated CTDIvol equals the physical scan's 10.12 mGy",
        round(acquisition.dose.ctdi_vol_mGy; digits = 2),
        abs(acquisition.dose.ctdi_vol_mGy - 10.12) < 0.05,
    )
    addcheck(
        "all four bins decomposed",
        vmi.settings.n_channels,
        vmi.settings.n_channels == 4,
    )
    addcheck("every VMI value finite (measured and reference)", r.finite, r.finite)
    addcheck(
        "decomposition: fraction of rays not converged",
        vmi.quality.frac_not_converged,
        vmi.quality.frac_not_converged < 1.0e-3,
    )
    water_error = maximum(abs, r.water_mean .- r.water_theory)
    addcheck(
        "solid water within 10 HU of theory at every energy (largest error, HU)",
        round(water_error; digits = 1),
        water_error <= 10,
    )
    reference_error = maximum(abs, r.water_reference .- r.water_theory)
    addcheck(
        "noise-free solid water within 10 HU of theory at every energy (largest error, HU)",
        round(reference_error; digits = 1),
        reference_error <= 10,
    )
    fits = [f for g in (:Ca, :I) for f in r.rods[g].fits]
    slopes = [f.slope for f in fits]
    addcheck(
        "rod regression slope within 1 ± 0.1 at every energy (min, max)",
        (round(minimum(slopes); digits = 3), round(maximum(slopes); digits = 3)),
        all(s -> abs(s - 1) <= 0.1, slopes),
    )
    r2 = minimum(f.r2 for f in fits)
    addcheck(
        "rod regression R² at least 0.99 at every energy (smallest)",
        round(r2; digits = 4),
        r2 >= 0.99,
    )
    list
end;

# ╔═╡ f11c86e7-a145-4fab-a652-90e02e173589
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
    r = pcct_results
    fmt(v) = join(round.(v; digits = 1), " / ")
    slopes = [f.slope for g in (:Ca, :I) for f in r.rods[g].fits]
    Markdown.parse(
        """
        ## Summary

        ```julia
        calibration = BS.calibrate(scanner, protocol, recon_opts, phantom)  # once per scanner and protocol
        ws    = BS.create_workspace(scanner, protocol, opts, recon_opts, phantom; calibration)
        sim   = BS.simulate!(ws, phantom, protocol, opts; keep_projection = true)  # then (…; projection)
        basis = BS.spectral_basis(ws; I0 = sim.I0_bins)          # ray-resolved, no calibration scan
        vmi   = BS.vmi_pipeline(; channels = sim.pcct_sino.bins, basis, geom = ws.geom,
                                  to_backend, matrix_size, vmi_energies = (40, 70, 100, 140),
                                  denoiser = BS.SpectralHYPR(), fbp_filter = PAIR_FILTER.filter,
                                  pair_basis, composite_energy)
        ```

        Four photon-counting energy windows go into the decomposition as four measurements, and the
        spectral model is the one the simulation applied, ray by ray through the bowtie, so no
        calibration scan is involved.

        | Result | Value |
        |---|---|
        | CTDIvol (the physical scan's 10.12 mGy) | $(round(acquisition.dose.ctdi_vol_mGy; digits = 2)) mGy |
        | spectral pair from the calibration draw | E* = $(round(Int, PAIR_BASIS.Estar)) keV, β = $(round(PAIR_BASIS.β; sigdigits = 3)) |
        | solid water at $(join(Int.(r.energies), " / ")) keV (theory 0) | $(fmt(r.water_mean)) HU |
        | the same, noise-free | $(fmt(r.water_reference)) HU |
        | noise σ per 0.4 mm slice | $(fmt(r.water_noise_vs_reference)) HU |
        | rod regression slopes, every energy | $(round(minimum(slopes); digits = 3)) to $(round(maximum(slopes); digits = 3)) |
        | checks passed | $(count(check -> check.pass, checks)) of $(length(checks)) |

        The per-rod table separates each rod's error into the basis model's own bias, largest for
        calcium, which the iodine–water basis represents only approximately, and what the noise and the
        denoiser add to it.
        """,
    )
end

# ╔═╡ Cell order:
# ╟─d3054785-9e00-4094-a491-088ce63be9dc
# ╟─6b217424-7795-4e52-b0de-056b643a12c2
# ╟─f2798d62-3509-4cc4-a24f-39ace8bb5a9e
# ╟─a0a37858-dd02-485c-91ea-98df6b7d8345
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
# ╟─3374b87e-9e38-410f-8cf2-f1a78439f3ef
# ╠═42f1aeb4-0d26-4c6a-87ff-765a6db5b7a4
# ╠═63410694-a179-4c1e-a15e-e256c89be1b6
# ╠═49d1d4f5-964f-41a2-9a65-1fd1ae7feeac
# ╠═ff468aa7-9173-4b37-b4db-6aaba2e762f2
# ╠═5615a564-a130-48cb-abc2-3f42996da720
# ╠═80ab6ea4-0d34-49b8-9405-1a0c8aefc5b1
# ╠═35f29bd8-7078-4f38-a80e-4eca5b40bfed
# ╠═3a2e9afe-2f6d-41b1-a24d-81e610177f1d
# ╠═5bb4711f-b473-4fd6-bc4e-f530cf3d63df
# ╟─040e1000-0000-4000-8000-000000000100
# ╠═939dcda3-9be5-46c8-aaa1-ded273e8cf04
# ╠═5248ba55-965a-41f7-845c-99616018b475
# ╠═203181f5-2f7e-4201-a02e-757682b5f47d
# ╟─040e1000-0000-4000-8000-000000000101
# ╠═2c157064-8567-450b-bc08-c2606084a77f
# ╟─d15334a3-db15-541c-b1a3-ab862db955f4
# ╟─040e1000-0000-4000-8000-000000000102
# ╠═04b00000-0000-4000-8000-000000000001
# ╠═04b00000-0000-4000-8000-000000000002
# ╠═f9c0af7a-addd-4249-96fb-b9078765fbd1
# ╟─040e1000-0000-4000-8000-000000000103
# ╠═2d65a0c0-b25d-41ad-9cd3-e7a2d08a2482
# ╠═04b00000-0000-4000-8000-000000000003
# ╠═08cbc6fd-3c7c-432f-99e5-b220f8fe7fde
# ╟─5ecd97c6-ad47-4558-886d-22ed45eda97d
# ╠═4315ef69-aa2f-4ee0-a13b-c65e01fb87ce
# ╟─04a10000-0000-4000-8000-000000000001
# ╟─04a10000-0000-4000-8000-000000000002
# ╟─04a10000-0000-4000-8000-000000000003
# ╟─dc8a8352-5598-4cdd-952f-3d77367850e9
# ╠═04a20000-0000-4000-8000-000000000001
# ╠═04a20000-0000-4000-8000-000000000002
# ╠═04a20000-0000-4000-8000-000000000006
# ╟─04b00000-0000-4000-8000-000000000010
# ╠═04b00000-0000-4000-8000-000000000011
# ╠═04b00000-0000-4000-8000-000000000012
# ╟─04b00000-0000-4000-8000-000000000020
# ╠═04b00000-0000-4000-8000-000000000021
# ╟─04a20000-0000-4000-8000-000000000003
# ╠═04a20000-0000-4000-8000-000000000004
# ╠═04b00000-0000-4000-8000-000000000022
# ╟─04a20000-0000-4000-8000-000000000005
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
# ╟─040e1000-0000-4000-8000-000000000070
# ╟─040e0001-0000-4000-8000-000000000007
# ╟─04a30000-0000-4000-8000-000000000001
# ╟─04a30000-0000-4000-8000-000000000002
# ╠═040e0002-0000-4000-8000-000000000004
# ╟─f11c86e7-a145-4fab-a652-90e02e173589
# ╟─040e0002-0000-4000-8000-000000000006
