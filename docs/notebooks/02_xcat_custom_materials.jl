### A Pluto.jl notebook ###
# v0.2.3

using Markdown
using InteractiveUtils

# ╔═╡ 02000001-0000-4000-8000-000000000001
md"""
# An XCAT Chest With Custom Materials

**Scan a digital anatomy whose materials you define, down to a contrast agent built from its
elements.**

Notebook 01 walked the five-struct API on a calibration phantom. This notebook swaps the phantom
for a voxelized **XCAT** adult-male chest and gives each tissue its own
[`XrayAttenuation.Material`](https://github.com/MolloiLab/XrayAttenuation.jl), with an iodinated
blood pool (a CT angiography bolus) built inline, then compares FBP with Hybrid IR on the same
scan.

| | |
|---|---|
| **Phantom** | XCAT adult-male 50th-percentile chest, one axial slab at full resolution (XCIST `phantoms-voxelized`, downloaded as a Julia artifact) |
| **Materials** | `XA.Materials.ncat_*` tissues, plus an iodinated-blood material constructed inline |
| **Scanner** | GE Revolution Apex Elite (as in notebook 01) |
| **Reconstruction** | FBP and Hybrid IR at strength 60, both after the water beam-hardening correction |
"""

# ╔═╡ e4cbe9e4-4357-4fa8-8959-03c8ab21bc97
let
    Mke.set_theme!(THEME)
    if xcat === nothing || hu_hir === nothing
        md"""
        !!! warning "Skipped: the XCAT phantom could not be loaded (section 1)"
        """
    else
        w = size(xcat.mask, 1) * xcat.voxel_size_cm[1] / 2
        fig, axes = locator_band(
            ["The Anatomy", "The Contrast", "The Image"],
            [
                "XCAT Male Chest, 0.25 mm Voxels",
                "Blood With 5 mg/mL Iodine, Built Inline",
                "Hybrid IR, 120 kVp, 250 mA",
            ];
            limits = [(-w, w, -w, w), (-1, 11, -6, 6), (-17.5, 17.5, -17.5, 17.5)],
        )
        draw_tissues!(axes[1], xcat.mask[:, :, 1], xcat.label_names, xcat.voxel_size_cm[1])

        recipe = [
            ((5.0, 3.6), "ncat_blood", COLORS.red),
            ((5.0, 0.0), "+ Iodine (Z = 53)", COLORS.violet),
            ((5.0, -3.6), "Iodinated Blood", COLORS.green),
        ]
        for (i, (c, title, accent)) in enumerate(recipe)
            node!(
                axes[2],
                c,
                title;
                w = 8.0,
                h = 2.0,
                accent,
                fill = COLORS.night_card,
                textcolor = COLORS.night_ink,
                titlesize = 21,
            )
            i < 3 &&
                arrow!(
                    axes[2],
                    (c[1], c[2] - 1.05),
                    (c[1], c[2] - 2.55);
                    color = COLORS.night_muted,
                )
        end

        mid = size(hu_hir, 3) ÷ 2
        Mke.image!(
            axes[3],
            (-17.5, 17.5),
            (-17.5, 17.5),
            hu_hir[:, :, mid];
            colormap = :grays,
            colorrange = (-300, 550),
        )
        fig
    end
end

# ╔═╡ bf475777-4a27-43bb-971f-b2d78dd97840
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 470))
    nodes = Dict(
        :defaults => (
            0.0,
            1.0,
            "xcat_default_materials",
            "XCIST Name → Material",
            COLORS.violet,
        ),
        :custom => (0.0, -0.6, "build_iodine_blood", "Your Own XA.Material", COLORS.violet),
        :map => (4.2, 0.2, "materials_map", "\"ncat_blood\" Replaced", COLORS.violet),
        :load => (8.4, 0.2, "load_xcat_male_slab", "Mask, Materials, Voxels", COLORS.cyan),
        :phantom => (12.6, 0.2, "Phantom", "Mask on the Device", COLORS.cyan),
        :scan => (16.8, 0.2, "simulate!", "Then FBP and Hybrid IR", COLORS.green),
    )
    edges = [
        :defaults => :map,
        :custom => :map,
        :map => :load,
        :load => :phantom,
        :phantom => :scan,
    ]
    flow_chart!(fig[1, 1], nodes, edges; limits = (-2.1, 18.9, -1.4, 1.9), w = 3.7)
    fig
end

# ╔═╡ 02000002-0000-4000-8000-000000000001
md"""
## Notebook Setup

As in notebook 01: the shared `docs/` environment, then the device. **Unitful** supplies the
`u"eV"` and `u"g/cm^3"` units the `XA.Material` constructor takes.
"""

# ╔═╡ 02000003-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 02000003-0000-4000-8000-000000000002
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 02000003-0000-4000-8000-000000000003
# Use CairoMakie for faithful build-time rendering. Snapshot can still isolate
# and compile independent browser-safe islands without hoisting this import.
import CairoMakie as Mke

# ╔═╡ 02000003-0000-4000-8000-000000000004
using Statistics: std, mean

# ╔═╡ 02000003-0000-4000-8000-000000000005
using Markdown: @md_str

# ╔═╡ 02000003-0000-4000-8000-000000000006
using Unitful: @u_str

# ╔═╡ 05000003-0000-4000-8000-000000000003
import Unitful: ustrip, uconvert

# ╔═╡ b0c50f56-f6fc-4589-b699-c25f51d6b247
using PlutoUI: TableOfContents

# ╔═╡ 1ab8176f-2b8e-4948-a332-f326aed838c9
TableOfContents()

# ╔═╡ 02000004-0000-4000-8000-000000000001
md"""
### Choosing the Device

`GPUSelect.Storage()` returns the array type of the GPU it finds (`CuArray`, `MtlArray`,
`ROCArray`, `oneArray`), or `Array` on a CPU-only host; moving the phantom mask to it decides
where the pipeline runs.
"""

# ╔═╡ 02000005-0000-4000-8000-000000000001
begin
    import GPUSelect
    # the backend array type, directly: MtlArray / CuArray / ROCArray
    AT = GPUSelect.Storage()
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 02000007-0000-4000-8000-000000000001
Markdown.parse(
    """
**Backend detected:** $(GPU_BACKEND.name)
""",
)

# ╔═╡ 05e43a6d-19e9-4bcc-87a0-3614520bec60
md"""
## Figure Style

BasisSimulator's colours and type, and the drawing helpers every notebook's figures share.
"""

# ╔═╡ 991c36be-451d-40fd-a485-0f90f88dc229
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

# ╔═╡ f6194700-35b4-4ded-a8fb-ab9105e7928b
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

# ╔═╡ 19bcf176-1a00-43ec-8d43-ba62e30a2e60
"""A blank drawing area at `position`: equal scales, no axes, limits `(x0, x1, y0, y1)`."""
function canvas(position, limits; background = COLORS.paper)
    ax = Mke.Axis(position; aspect = Mke.DataAspect(), backgroundcolor = background)
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    Mke.limits!(ax, limits...)
    return ax
end;

# ╔═╡ 7f154f20-9267-4f5d-96a8-8752af93c86f
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

# ╔═╡ e736f21e-498a-42ea-a278-62a5a3d61616
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

# ╔═╡ 073d91ea-50cd-4603-af28-2cbf8d2ffff1
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

# ╔═╡ 0298d7c6-ded8-4249-b86c-ed9d0831b09f
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

# ╔═╡ 161c24a8-88b9-4939-9339-6dce2f68307c
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

# ╔═╡ 1a2640db-10c7-471e-8733-804c6415996e
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

# ╔═╡ 88c22cb0-4506-4df0-b66d-60f42e0d6bda
"""`c` mixed toward `base` (the paper): `t = 1` is `c` itself, `t = 0` is `base`."""
function blend(c, t; base = COLORS.paper)
    return Mke.RGBf(
        t * c.r + (1 - t) * base.r,
        t * c.g + (1 - t) * base.g,
        t * c.b + (1 - t) * base.b,
    )
end

# ╔═╡ dba58912-3fd4-48eb-88c2-9ce7771338f0
"""How each XCIST tissue is drawn: its colour and its name in a legend, by XCIST name."""
TISSUES = Dict(
    "ncat_lung" => (blend(COLORS.cyan, 0.18), "Lung"),
    "ncat_adipose" => (blend(COLORS.beam, 0.45), "Fat"),
    "ncat_muscle" => (blend(COLORS.purple, 0.38), "Muscle"),
    "ncat_heart" => (COLORS.purple, "Myocardium"),
    "ncat_blood" => (COLORS.red, "Blood + 5 mg/mL Iodine"),
    "ncat_dry_spine" => (COLORS.ink, "Spine"),
    "ncat_dry_rib" => (COLORS.muted, "Rib"),
    "ncat_cartilage" => (blend(COLORS.cyan, 0.6), "Cartilage"),
    "ncat_water" => (COLORS.blue, "Water"),
)

# ╔═╡ 419e20b0-e45f-467a-bc31-ed264c12bb1e
"""
An XCAT slice of `labels` drawn on `ax` in its tissue colours (`TISSUES`), `dx` cm per voxel,
centred (air left clear; a tissue not in `TISSUES` is drawn grey).
"""
function draw_tissues!(ax, labels, label_names, dx)
    n1, n2 = size(labels)
    colour(l) = haskey(label_names, l) ?
        Mke.RGBAf(get(TISSUES, label_names[l], (COLORS.rule, ""))[1]) :
        Mke.RGBAf(0, 0, 0, 0)
    lut = Dict(l => colour(l) for l in unique(labels))
    Mke.image!(
        ax,
        (-n1 / 2 * dx, n1 / 2 * dx),
        (-n2 / 2 * dx, n2 / 2 * dx),
        [lut[l] for l in labels];
        interpolate = false,
    )
    return ax
end

# ╔═╡ 03000000-0000-4000-8000-000000000000
md"""
## 1. Download the XCAT Phantom

No local files, no manual setup. `BS.load_xcat_male_chest()` fetches the voxelized **XCAT
adult-male 50th-percentile chest** phantom from the open
[`xcist/phantoms-voxelized`](https://github.com/xcist/phantoms-voxelized) repository
(BSD-3-Clause) into Julia's content-addressed artifact store: it downloads once, verifies the
file's SHA-256 against the value pinned in the package, logs the Segars XCAT + XCIST citation,
and reuses the cached copy on every later call.

!!! info "Using the XCAT loaders"
    Four phantoms ship: `:female_slab`, `:female_chest`, `:male_slab`, `:male_chest` (see
    `BS.xcat_phantoms()`). Each `load_xcat_<name>()` returns the **labeled pieces**: `mask`, a
    `materials` dict (`label => XA.Material`), `label_names` and `voxel_size_cm`, which you turn
    into a `Phantom` in one line (section 3). Useful keywords:

    - `downsample = n`: a label-preserving majority-vote shrink (physical extent preserved). This
      notebook uses the **slab** at its full 0.25 mm, so it passes no `downsample`; reach for it
      on the full `:male_chest` volume. It is lossy (a 6× round trip agrees with full resolution
      on about 96 % of voxels), so treat it as a speed knob, not a lossless representation.
    - `materials = Dict("ncat_blood" => …)`: override the tissue → material map by XCIST name
      *before* assembly (section 2).
    - `path = …`: skip the download and read an already-extracted copy from disk.
    - `quiet = true`: silence the citation and progress logging.
"""

# ╔═╡ 05000001-0000-4000-8000-000000000001
md"""
## 2. Custom Materials via `XrayAttenuation`

The loader assigns every label a sensible default tissue, but the point of a digital phantom is
that **you control the material physics**. Each label maps to an
[`XA.Material`](https://github.com/MolloiLab/XrayAttenuation.jl) that gives the simulator its
energy-dependent μ(E). There are two ways to get one: **prebuilt**, `XA.Materials.ncat_blood`,
`XA.Materials.ncat_muscle` and the rest of XrayAttenuation's NCAT/XCAT catalog; or **constructed
inline**, `XA.Material(name, ZA_ratio, I, density, composition)`, for your own contrast bolus,
calibration solution or alloy.

### A Prebuilt NCAT Material
"""

# ╔═╡ 05000002-0000-4000-8000-000000000002
let m = BS.XA.Materials.ncat_blood
    Markdown.parse(
        """
    **`XA.Materials.ncat_blood`**

    | Field         | Value                                    |
    |---------------|------------------------------------------|
    | `name`        | $(m.name)                                |
    | `ZA_ratio`    | $(round(m.ZA_ratio; digits = 4))           |
    | `I`           | $(m.I)                                   |
    | `density`     | $(m.density)                             |
    | `composition` | $(length(m.composition)) elements (Z = $(sort(collect(keys(m.composition))))) |
    """,
    )
end

# ╔═╡ 05000003-0000-4000-8000-000000000001
md"""
### A Custom Contrast Material

A 5 mg/mL iodinated blood mixture, like a coronary CT-angiography bolus: start from
`ncat_blood`'s composition and add elemental iodine. Adding `iodine_mg_per_mL` mg of iodine per mL
of blood at density ρ gives an iodine **mass fraction** of `iodine_mg_per_mL / (1000 ρ)` (ρ in
g/cm³); the other elements are rescaled so the composition still sums to 1.
"""

# ╔═╡ 05000003-0000-4000-8000-000000000002
function build_iodine_blood(iodine_mg_per_mL::Real)
    base = BS.XA.Materials.ncat_blood
    density = base.density # ~1.06 g/cm³
    ρ_g_cm3 = ustrip(uconvert(u"g/cm^3", density))
    f_I = iodine_mg_per_mL / (1000.0 * ρ_g_cm3) # iodine mass fraction
    scale = 1.0 - f_I

    comp = Dict{Int, Float64}()
    for (Z, frac) in base.composition
        comp[Z] = frac * scale
    end
    comp[53] = get(comp, 53, 0.0) + f_I # add iodine

    return BS.XA.Material(
        "Iodine-blood ($(iodine_mg_per_mL) mg/mL)",
        # ZA shifts ~negligibly at clinical dose
        base.ZA_ratio,
        # likewise for excitation energy
        base.I,
        density,
        comp,
    )
end;

# ╔═╡ 05000003-0000-4000-8000-000000000004
iodine_blood = build_iodine_blood(5.0);

# ╔═╡ f6486729-34f3-4680-a748-30df74cbd0c5
md"""
What the iodine does: below its K-edge at 33.2 keV iodine absorbs like any heavy element, and
just above it absorbs far more, right where a 120 kVp spectrum is rich in photons. Five milligrams
per millilitre is a mass fraction under half a percent, yet it lifts the blood's attenuation
visibly at every energy of the spectrum.
"""

# ╔═╡ 75a2b744-6b1a-4a82-be46-9bfd26684031
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 880))

    # the recipe, drawn
    ax0 = canvas(fig[1, 1], (-1, 25, -1.6, 2.4))
    node!(
        ax0,
        (3.0, 0.4),
        "ncat_blood";
        subtitle = "1.06 g/cm³",
        w = 5.0,
        h = 1.8,
        accent = COLORS.red,
    )
    Mke.text!(
        ax0,
        7.0,
        0.4;
        text = "+",
        align = (:center, :center),
        fontsize = 40,
        font = :bold,
    )
    node!(
        ax0,
        (11.0, 0.4),
        "Iodine, Z = 53";
        subtitle = "5 mg per mL",
        w = 5.0,
        h = 1.8,
        accent = COLORS.violet,
    )
    arrow!(ax0, (13.8, 0.4), (16.2, 0.4))
    node!(
        ax0,
        (19.5, 0.4),
        "iodine_blood";
        subtitle = "Mass Fraction of I: $(round(100 * iodine_blood.composition[53]; digits = 2)) %",
        w = 6.4,
        h = 1.8,
        accent = COLORS.green,
    )

    # the measurement: attenuation over the spectrum's energies
    E_keV = 20.0:0.25:120.0
    μ(material) = [BS.compute_μ_at_energy(material, e) for e in E_keV]
    ax = Mke.Axis(
        fig[2, 1];
        title = "Linear Attenuation, From NIST XCOM via XrayAttenuation.jl",
        xlabel = "Photon Energy (keV)",
        ylabel = "μ (1/cm)",
    )
    Mke.lines!(ax, E_keV, μ(BS.XA.Materials.water); color = COLORS.blue, label = "Water")
    Mke.lines!(
        ax,
        E_keV,
        μ(BS.XA.Materials.ncat_blood);
        color = COLORS.red,
        linestyle = :dash,
        label = "ncat_blood",
    )
    Mke.lines!(
        ax,
        E_keV,
        μ(iodine_blood);
        color = COLORS.green,
        label = "iodine_blood, 5 mg/mL",
    )
    Mke.vlines!(ax, [33.17]; color = (COLORS.violet, 0.6), linestyle = :dot, linewidth = 2)
    Mke.text!(
        ax,
        33.9,
        1.0;
        text = "Iodine K-Edge, 33.2 keV",
        align = (:left, :center),
        fontsize = 16,
        color = COLORS.violet,
    )
    if bhc_calibration !== nothing
        E = bhc_calibration.ref_E_keV
        hu(m) = 1000 *
            (BS.compute_μ_at_energy(m, E) - bhc_calibration.μ_water) /
            bhc_calibration.μ_water
        Mke.vlines!(ax, [E]; color = (COLORS.ink, 0.4), linewidth = 1.5)
        Mke.text!(
            ax,
            E + 1.0,
            0.55;
            text = "At $(round(E; digits = 1)) keV, the Scan's Reference:\nBlood $(round(Int, hu(BS.XA.Materials.ncat_blood))) HU, With Iodine $(round(Int, hu(iodine_blood))) HU",
            align = (:left, :center),
            fontsize = 16,
        )
    end
    Mke.axislegend(ax; position = :rt)
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.2))
    fig
end

# ╔═╡ 05000004-0000-4000-8000-000000000001
md"""
### The Name → Material Map

The loader ships the canonical NCAT/XCAT mapping as `BS.xcat_default_materials()`: every XCIST
material name → an `XrayAttenuation` material. This is what replaces hand-parsing a per-organ
spreadsheet: the compositions already live in XrayAttenuation's NCAT catalog. Copy the dict, edit
it by name, and hand it back to the loader with `materials =`.
"""

# ╔═╡ 05000004-0000-4000-8000-0000000000c1
let m = BS.xcat_default_materials()
    rows = ["| `$(k)` | `$(v.name)` |" for (k, v) in sort(collect(m); by = first)]
    Markdown.parse(
        "**`BS.xcat_default_materials()`**\n\n" *
            "| XCIST Name | XrayAttenuation Material |\n|---|---|\n" *
            join(rows, "\n"),
    )
end

# ╔═╡ 05000004-0000-4000-8000-000000000008
md"""
### Override a Material

Copy the default map and reassign by XCIST name: here the blood pool becomes the 5 mg/mL
`iodine_blood` built above, so the heart chambers and great vessels light up like a
contrast-enhanced CTA. The edited map goes to the loader in section 3.
"""

# ╔═╡ 05000004-0000-4000-8000-000000000009
materials_map = let m = BS.xcat_default_materials()
    m["ncat_blood"] = iodine_blood # dope the blood pool with 5 mg/mL iodine contrast
    m
end;

# ╔═╡ 06000001-0000-4000-8000-000000000001
md"""
## 3. Build the `Phantom`

`load_xcat_male_slab` returns the labeled pieces; we assemble the `Phantom` from them. For a GPU
simulation the mask must live on the device (the workspace picks its compute backend from the
mask's array type), so we `to_gpu` it first. The slab is a single axial section at the phantom's
full in-plane resolution: real anatomy at a docs-friendly compute cost, with no downsampling.
"""

# ╔═╡ 04000004-0000-4000-8000-000000000001
xcat = try
    # full-resolution single slab
    BS.load_xcat_male_slab(; materials = materials_map, quiet = true)
catch err
    @warn "XCAT download/load failed; the compute cells will skip" exception = err
    nothing
end;

# ╔═╡ 06000000-0000-4000-8000-0000000000a1
phantom_labeled = xcat === nothing ? nothing : xcat.mask;

# ╔═╡ 06000002-0000-4000-8000-000000000001
VOXEL_SIZE_CM = xcat === nothing ? (0.15, 0.15, 0.15) : xcat.voxel_size_cm;

# ╔═╡ 06000003-0000-4000-8000-000000000001
phantom = xcat === nothing ?
    nothing :
    BS.Phantom(to_gpu(xcat.mask), xcat.materials, xcat.voxel_size_cm);

# ╔═╡ 04000005-0000-4000-8000-000000000001
let
    Mke.set_theme!(THEME)
    if phantom_labeled === nothing
        md"""
        !!! warning "Skipped: see section 1"
        """
    else
        mid = max(1, size(phantom_labeled, 3) ÷ 2)
        slice = phantom_labeled[:, :, mid]
        w = size(slice, 1) * VOXEL_SIZE_CM[1] / 2
        h = size(slice, 2) * VOXEL_SIZE_CM[2] / 2

        fig = Mke.Figure(size = (1250, 640))
        ax = Mke.Axis(
            fig[1, 1];
            title = "XCAT Male Chest Slab, Full Resolution",
            subtitle = "$(size(slice, 1)) × $(size(slice, 2)) Voxels of $(join(round.(VOXEL_SIZE_CM[1:2] .* 10; digits = 3), " × ")) mm",
            aspect = Mke.DataAspect(),
            xlabel = "x (cm)",
            ylabel = "y (cm)",
        )
        draw_tissues!(ax, slice, xcat.label_names, VOXEL_SIZE_CM[1])
        Mke.limits!(ax, -w, w, -h, h)

        present = [
            xcat.label_names[l]
            for l in sort(unique(slice))
            if haskey(xcat.label_names, l)
        ]
        shown = [name for name in present if haskey(TISSUES, name)]
        Mke.Legend(
            fig[1, 2],
            [
                Mke.PolyElement(; color = TISSUES[name][1], strokecolor = :transparent)
                for name in shown
            ],
            [TISSUES[name][2] for name in shown];
            valign = :center,
        )
        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_phantom.png"),
            fig;
            px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 07000000-0000-4000-8000-000000000000
md"""
## 4. Scan

The GE Revolution Apex Elite of notebook 01, `REVOLUTION_APEX`. The protocol is a body CTA:
120 kVp / 250 mA, 5 mm collimation, 500 views in 1 s, each view integrated over the arc the
gantry turns through while the detector reads it (`view_samples = 5`). Reconstruction: 512 × 512
over 35 cm, eight 0.625 mm slices.
"""

# ╔═╡ 07000002-0000-4000-8000-000000000001
# The documented GE Revolution Apex Elite, its clinical DAS included: electronic noise enters the
# counts before the log, as a real DAS does, so iterative reconstruction can act on it.
scanner = BS.REVOLUTION_APEX

# ╔═╡ 0f01cfb1-2f77-583c-a21d-8168bca60aec
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

# ╔═╡ 07000003-0000-4000-8000-000000000001
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 250.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 5.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 07000004-0000-4000-8000-000000000001
# `projector` picks the forward ray tracer: :dd_fast (default) is distance-driven and
# anti-aliased, and walks the volume once for the whole spectrum; :siddon point-samples
# the volume and can alias in strongly beam-hardened regions. The Hybrid-IR cell reads
# `sim_opts.projector`, so its system matrix always matches the operator that made the data.
# `view_samples = 5`: the detector integrates each view while the gantry turns (notebook 01).
sim_opts = BS.SimOptions(seed = 1234, projector = :dd_fast, view_samples = 5);

# ╔═╡ 07000005-0000-4000-8000-000000000001
recon_opts = BS.ReconOptions(matrix_size = (512, 512, 8), fov_cm = 35.0, z_cm = 0.5);

# ╔═╡ 08000001-0000-4000-8000-000000000001
md"""
The notebook 01 pattern: `create_workspace`, `simulate!`, copy the sinogram off the device,
release the device buffers. `simulate!` returns the acquisition's dose report.
"""

# ╔═╡ 08000002-0000-4000-8000-000000000001
sim = phantom === nothing ?
    nothing :
    let
        @info "Simulating XCAT body CTA: 120 kVp / 250 mA…"
        ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
        out = BS.simulate!(ws, phantom, protocol, sim_opts)

        result = (sino = Array(ws.sinogram), geom = ws.geom, dose = out.dose)

        ws = nothing
        GC.gc(true)

        result
    end;

# ╔═╡ 08000003-0000-4000-8000-000000000001
sim === nothing ? md"" : sim.dose

# ╔═╡ 09000000-0000-4000-8000-000000000000
md"""
## 5. Reconstruct

The chain of notebook 01, applied with both algorithms: the water beam-hardening correction,
calibrated per detector column from the full detected spectrum (no tunable parameters), then FBP
or Hybrid IR, then HU with the correction's own water μ.
"""

# ╔═╡ 78a5e414-4f9d-4474-b32a-a8c9e6af9113
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

# ╔═╡ 09000001-0000-4000-8000-000000000001
md"""
!!! info "Noise is already in the counts"
    Quantum noise and the electronic noise of `scanner.electronic_noise` are both added by
    `simulate!` to the detector counts, before the log, where a real detector adds them. The
    reconstruction sees all of it, so iterative reconstruction can reduce all of it. Adding
    noise to the HU volume after reconstruction would be identical for FBP and HIR by
    construction.

### The Water BHC

One calibration for the 120 kVp beam (2.5 mm flat filter + 4.5 mm Al + bowtie). It returns a
`WaterBHC` with one polynomial per detector column and its reference `μ_water_ref`, used by both
reconstructions.
"""

# ╔═╡ 09000004-0000-4000-8000-000000000001
bhc_calibration = sim === nothing ?
    nothing :
    let
        # Parameter-free water BHC — physics only, no tunables, no thresholds:
        # per-column poly→mono polynomials from the FULL detected spectrum
        # (tube × filters × bowtie × heel × η(E) — whatever sim_opts enabled).
        model = BS.calibrate_bhc_water(
            sim_opts,
            protocol;
            scanner = scanner,
            geom = sim.geom,
        )

        (model = model, μ_water = model.μ_water_ref, ref_E_keV = model.reference_energy_keV)
    end;

# ╔═╡ 09000005-0000-4000-8000-000000000001
Markdown.parse(
    """
**Calibrated:** reference energy $(bhc_calibration === nothing ? "—" : round(bhc_calibration.ref_E_keV, digits = 1)) keV,
water μ $(bhc_calibration === nothing ? "—" : round(bhc_calibration.μ_water, digits = 5)) cm⁻¹.
""",
)

# ╔═╡ 09000010-0000-4000-8000-000000000001
md"""
### FBP

Water BHC → FDK with the `:standard` kernel → HU, releasing the device buffers at the end.
"""

# ╔═╡ 09000002-0000-4000-8000-000000000001
hu_fbp = sim === nothing ?
    nothing :
    let
        sino_gpu = BS.apply_bhc_water(to_gpu(sim.sino), bhc_calibration.model)
        ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, sim.geom, recon_opts.matrix_size)
        recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, sim.geom)
        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))
        ws_fdk = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)
        hu
    end;

# ╔═╡ 10000001-0000-4000-8000-000000000001
md"""
### Hybrid IR

The same chain with `create_hir_recon_workspace(...; strength = 60)` in place of the FDK
workspace: an FBP start, then ordered-subsets penalised weighted least squares with an
edge-preserving Huber prior, the open-literature (Fessler) counterpart of the vendor hybrid IR
algorithms (GE ASIR-V, Siemens SAFIRE, Philips iDose⁴, Canon AIDR 3D), whose noise-reduction range
its strength table targets.

!!! tip "One dial: strength"
    A percentage in steps of 10, read like the GE ASIR-V dial: `0` is pure FBP (the iterative
    loop is skipped), `60` the standard clinical setting used here, `100` the maximum noise
    reduction. It moves the regularisation weight, the Huber threshold, the relaxation and the
    number of epochs together along one calibrated trajectory.

!!! warning "Projector consistency"
    Hybrid IR's data term uses a forward projector, which must be the one that generated the
    sinogram. The cell passes `projector = sim_opts.projector`, so changing the simulation's
    projector keeps the reconstruction consistent. FBP has no forward projector and is unaffected.
"""

# ╔═╡ 10000002-0000-4000-8000-000000000001
hu_hir = sim === nothing ?
    nothing :
    let
        sino_gpu = BS.apply_bhc_water(to_gpu(sim.sino), bhc_calibration.model)
        ws_hir = BS.create_hir_recon_workspace(
            sino_gpu,
            sim.geom,
            recon_opts.matrix_size;
            strength = 60,
            projector = sim_opts.projector,
        )
        recon_μ = BS.reconstruct!(ws_hir, sino_gpu, sim.geom)
        hu = Float32.(BS.to_hounsfield(Array(recon_μ); μ_water = bhc_calibration.μ_water))
        ws_hir = nothing
        sino_gpu = nothing
        recon_μ = nothing
        GC.gc(true)
        hu
    end;

# ╔═╡ 11000000-0000-4000-8000-000000000000
md"""
## 6. FBP vs Hybrid IR

### Where the Noise Is Measured

A box near the image centre spans lung to iodinated blood, so its standard deviation is anatomy,
not noise, and would hide what HIR does. The phantom labels are carried onto the reconstruction
grid with `BS.resample_to_recon` and the myocardium label is eroded by 2 pixels, so only pixels
surrounded by myocardium are measured.
"""

# ╔═╡ 270e47b9-b687-45f9-a6de-05abd9e699d6
# The myocardium, eroded: a pixel counts only if every in-plane neighbour within r is myocardium
# too (boundary pixels mix two tissues, and their spread is the edge, not the noise)
myocardium = hu_fbp === nothing || hu_hir === nothing ?
    nothing :
    let
        labels = BS.resample_to_recon(phantom, sim.geom, recon_opts.matrix_size)
        heart = first(k for (k, v) in xcat.label_names if v == "ncat_heart")
        r = 2
        nx, ny, nz = size(labels)
        roi = [
            CartesianIndex(i, j, k)
            for k in 1:nz, j in (1 + r):(ny - r), i in (1 + r):(nx - r)
            if all(labels[i + di, j + dj, k] == heart for dj in -r:r, di in -r:r)
        ]
        (
            labels = labels,
            roi = roi,
            fbp = (mean = Float64(mean(hu_fbp[roi])), σ = Float64(std(hu_fbp[roi]))),
            hir = (mean = Float64(mean(hu_hir[roi])), σ = Float64(std(hu_hir[roi]))),
        )
    end;

# ╔═╡ ad66fcbb-94fb-4716-a3db-62fe61770c73
let
    Mke.set_theme!(THEME)
    if myocardium === nothing
        md"""
        !!! warning "Skipped: see section 1"
        """
    else
        mid = size(hu_fbp, 3) ÷ 2
        labels = myocardium.labels[:, :, mid]
        half = recon_opts.fov_cm / 2
        dx = recon_opts.fov_cm / size(labels, 1)
        in_roi = falses(size(labels))
        for c in myocardium.roi
            c[3] == mid && (in_roi[c[1], c[2]] = true)
        end

        fig = Mke.Figure(size = (1250, 640))
        ax = Mke.Axis(
            fig[1, 1];
            title = "The Myocardium Region, Central Slice",
            subtitle = "On the 512 × 512 Reconstruction Grid",
            aspect = Mke.DataAspect(),
        )
        Mke.hidedecorations!(ax)
        Mke.hidespines!(ax)
        draw_tissues!(ax, labels, xcat.label_names, dx)
        cyan = Mke.RGBAf(COLORS.cyan.r, COLORS.cyan.g, COLORS.cyan.b, 1.0)
        Mke.image!(
            ax,
            (-half, half),
            (-half, half),
            [m ? cyan : Mke.RGBAf(0, 0, 0, 0) for m in in_roi];
            interpolate = false,
        )
        Mke.limits!(ax, -14, 14, -9, 11)
        swatch(color) = Mke.PolyElement(; color, strokecolor = :transparent)
        Mke.Legend(
            fig[1, 2],
            [swatch(COLORS.cyan), swatch(COLORS.purple), swatch(COLORS.red)],
            [
                "Measured: $(length(myocardium.roi)) Pixels,\nAll Slices",
                "Myocardium Edge, Excluded",
                "Iodinated Blood Pool",
            ];
            valign = :center,
        )
        fig
    end
end

# ╔═╡ 11000001-0000-4000-8000-000000000001
md"""
### The Images

Both reconstructions share the whole chain except the algorithm. The soft-tissue window
(W 850 / L 125) shows the iodine-enhanced blood pools of the cardiac chambers and the difference
in noise texture between the two.
"""

# ╔═╡ 11000002-0000-4000-8000-000000000001
let
    Mke.set_theme!(THEME)
    if hu_fbp === nothing || hu_hir === nothing
        md"""
        !!! warning "Skipped: see section 1 for the XCAT setup"
        """
    else
        mid = size(hu_fbp, 3) ÷ 2
        window = (-300, 550) # W 850 / L 125: soft tissue and the iodinated blood pools
        σ(s) = "σ = $(round(s.σ; digits = 1)) HU in the Myocardium"
        panels = [
            (hu_fbp, "FBP", σ(myocardium.fbp)),
            (hu_hir, "Hybrid IR, Strength 60", σ(myocardium.hir)),
        ]

        fig = Mke.Figure(size = (1250, 640))
        local hm
        for (i, (volume, title, subtitle)) in enumerate(panels)
            ax = Mke.Axis(fig[1, i]; title, subtitle, aspect = Mke.DataAspect())
            hm = Mke.heatmap!(ax, volume[:, :, mid]; colormap = :grays, colorrange = window)
            Mke.hidedecorations!(ax)
            Mke.hidespines!(ax)
        end
        Mke.Colorbar(fig[1, 3], hm; label = "HU (W 850 / L 125)")
        Mke.save(
            joinpath(@__DIR__, "..", "assets", "xcat_fbp_vs_hir.png"),
            fig;
            px_per_unit = 2,
        )
        fig
    end
end

# ╔═╡ 11000003-0000-4000-8000-000000000001
let
    if myocardium === nothing
        md""
    else
        band = BS.get_hir_params(60).target_noise_reduction
        m = myocardium
        Markdown.parse(
            """
        **Noise inside the myocardium** ($(length(m.roi)) pixels, all slices)

        | Algorithm | Mean (HU) | σ (HU) |
        |:--|--:|--:|
        | FBP | $(round(m.fbp.mean; digits = 1)) | $(round(m.fbp.σ; digits = 1)) |
        | Hybrid IR, strength 60 | $(round(m.hir.mean; digits = 1)) | $(round(m.hir.σ; digits = 1)) |

        Hybrid IR lowers the noise by **$(round(100 * (1 - m.hir.σ / m.fbp.σ); digits = 1)) %** and leaves the mean
        where it was. The package's calibration band for strength 60 is $(band[1])–$(band[2]) %.
        """,
        )
    end
end

# ╔═╡ 12000001-0000-4000-8000-000000000001
let
    m = myocardium
    numbers = m === nothing ?
        "" :
        """
    On this scan (CTDIvol $(round(sim.dose.ctdi_vol_mGy; digits = 2)) mGy), Hybrid IR at strength 60
    lowered the myocardium noise from $(round(m.fbp.σ; digits = 1)) to $(round(m.hir.σ; digits = 1)) HU
    ($(round(100 * (1 - m.hir.σ / m.fbp.σ); digits = 1)) %) with the mean moving by
    $(round(m.hir.mean - m.fbp.mean; digits = 1)) HU.
    """
    Markdown.parse(
        """
    ## Summary

    Three ideas on top of the five-struct API of notebook 01:

    - **Voxel phantoms from an artifact.** `BS.load_xcat_male_slab()` (and the other
      `load_xcat_*` loaders) download a published XCAT phantom once, verify it, and return the
      labeled mask, a material per label and the voxel size;
      `BS.Phantom(to_gpu(mask), materials, voxel_size)` makes it simulatable. Any other labeled
      mask works the same way.
    - **Materials you define.** `XA.Materials.ncat_*` for the standard tissues, or
      `XA.Material(name, ZA, I, density, composition)` for a contrast bolus, a calibration solution
      or an alloy; the composition is a `Dict` of atomic number → mass fraction, and the loader's
      `materials =` keyword swaps it in by XCIST name. Here 5 mg/mL of iodine is a mass fraction of
      $(round(100 * iodine_blood.composition[53]; digits = 2)) %.
    - **Hybrid IR.** `create_hir_recon_workspace(sino, geom, matrix; strength)` in place of the FDK
      workspace: the same `reconstruct!` call, lower noise at the same mean HU.

    $(numbers)
    """,
    )
end

# ╔═╡ Cell order:
# ╟─02000001-0000-4000-8000-000000000001
# ╟─e4cbe9e4-4357-4fa8-8959-03c8ab21bc97
# ╟─bf475777-4a27-43bb-971f-b2d78dd97840
# ╟─02000002-0000-4000-8000-000000000001
# ╟─02000003-0000-4000-8000-000000000001
# ╠═02000003-0000-4000-8000-000000000002
# ╟─02000003-0000-4000-8000-000000000003
# ╟─02000003-0000-4000-8000-000000000004
# ╟─02000003-0000-4000-8000-000000000005
# ╟─02000003-0000-4000-8000-000000000006
# ╟─05000003-0000-4000-8000-000000000003
# ╟─b0c50f56-f6fc-4589-b699-c25f51d6b247
# ╟─1ab8176f-2b8e-4948-a332-f326aed838c9
# ╟─02000004-0000-4000-8000-000000000001
# ╠═02000005-0000-4000-8000-000000000001
# ╟─02000007-0000-4000-8000-000000000001
# ╟─05e43a6d-19e9-4bcc-87a0-3614520bec60
# ╠═991c36be-451d-40fd-a485-0f90f88dc229
# ╠═f6194700-35b4-4ded-a8fb-ab9105e7928b
# ╠═19bcf176-1a00-43ec-8d43-ba62e30a2e60
# ╠═7f154f20-9267-4f5d-96a8-8752af93c86f
# ╠═e736f21e-498a-42ea-a278-62a5a3d61616
# ╠═073d91ea-50cd-4603-af28-2cbf8d2ffff1
# ╠═0298d7c6-ded8-4249-b86c-ed9d0831b09f
# ╠═161c24a8-88b9-4939-9339-6dce2f68307c
# ╠═1a2640db-10c7-471e-8733-804c6415996e
# ╠═88c22cb0-4506-4df0-b66d-60f42e0d6bda
# ╠═dba58912-3fd4-48eb-88c2-9ce7771338f0
# ╠═419e20b0-e45f-467a-bc31-ed264c12bb1e
# ╟─03000000-0000-4000-8000-000000000000
# ╟─05000001-0000-4000-8000-000000000001
# ╟─05000002-0000-4000-8000-000000000002
# ╟─05000003-0000-4000-8000-000000000001
# ╠═05000003-0000-4000-8000-000000000002
# ╠═05000003-0000-4000-8000-000000000004
# ╟─f6486729-34f3-4680-a748-30df74cbd0c5
# ╟─75a2b744-6b1a-4a82-be46-9bfd26684031
# ╟─05000004-0000-4000-8000-000000000001
# ╟─05000004-0000-4000-8000-0000000000c1
# ╟─05000004-0000-4000-8000-000000000008
# ╠═05000004-0000-4000-8000-000000000009
# ╟─06000001-0000-4000-8000-000000000001
# ╠═04000004-0000-4000-8000-000000000001
# ╠═06000000-0000-4000-8000-0000000000a1
# ╠═06000002-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000001
# ╟─04000005-0000-4000-8000-000000000001
# ╟─07000000-0000-4000-8000-000000000000
# ╠═07000002-0000-4000-8000-000000000001
# ╟─0f01cfb1-2f77-583c-a21d-8168bca60aec
# ╠═07000003-0000-4000-8000-000000000001
# ╠═07000004-0000-4000-8000-000000000001
# ╠═07000005-0000-4000-8000-000000000001
# ╟─08000001-0000-4000-8000-000000000001
# ╠═08000002-0000-4000-8000-000000000001
# ╠═08000003-0000-4000-8000-000000000001
# ╟─09000000-0000-4000-8000-000000000000
# ╟─78a5e414-4f9d-4474-b32a-a8c9e6af9113
# ╟─09000001-0000-4000-8000-000000000001
# ╠═09000004-0000-4000-8000-000000000001
# ╟─09000005-0000-4000-8000-000000000001
# ╟─09000010-0000-4000-8000-000000000001
# ╠═09000002-0000-4000-8000-000000000001
# ╟─10000001-0000-4000-8000-000000000001
# ╠═10000002-0000-4000-8000-000000000001
# ╟─11000000-0000-4000-8000-000000000000
# ╠═270e47b9-b687-45f9-a6de-05abd9e699d6
# ╟─ad66fcbb-94fb-4716-a3db-62fe61770c73
# ╟─11000001-0000-4000-8000-000000000001
# ╟─11000002-0000-4000-8000-000000000001
# ╟─11000003-0000-4000-8000-000000000001
# ╟─12000001-0000-4000-8000-000000000001
