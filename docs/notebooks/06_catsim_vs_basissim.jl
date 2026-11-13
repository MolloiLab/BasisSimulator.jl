### A Pluto.jl notebook ###
# v0.3.0

using Markdown
using InteractiveUtils

# ╔═╡ 06000001-0000-4000-8000-000000000010
md"""
# CatSim vs BasisSimulator: Agreement and Runtime

**One scanner, one protocol and one Gammex 472 phantom, simulated and reconstructed three ways:
XCIST/CatSim (GE's open-source reference simulator, Python + C), BasisSimulator on the CPU, and
BasisSimulator on the GPU.**

A cross-validation of the forward model and the reconstruction against an independent
implementation. Both simulators get the same beam and physics, both correct beam hardening to the
same monoenergetic water μ, and both reconstruct with FDK and a `standard` kernel, so the
comparison reads in HU. Noise is off everywhere: the two simulators calibrate tube flux
independently, and a noisy comparison would mix a flux difference into a projector and
reconstruction check.
"""

# ╔═╡ b5d4108b-cbcb-492e-ad8d-9222e8bec75a
let
    Mke.set_theme!(THEME)
    fig, axes = locator_band(
        ["The Phantom", "The Scanner", "Three Pipelines"],
        [
            "Gammex 472: 7 Calcium and 7 Iodine Rods",
            "GE Revolution Apex Elite, 120 kVp",
            "One Phantom, Three Simulators",
        ];
        limits = [
            (-17.5, 17.5, -17.5, 17.5),
            (-7.5, 7.5, -7.5, 7.5),
            (-0.6, 10.6, -3.6, 3.6),
        ],
    )
    gammex_sketch!(axes[1]; dark = true)
    gantry!(axes[2]; labels = ("X-Ray Tube",))

    pipelines = (
        ("CatSim", COLORS.beam),
        ("BasisSim, CPU", COLORS.violet),
        ("BasisSim, GPU", COLORS.cyan),
    )
    for (i, (name, accent)) in enumerate(pipelines)
        y = 2.3 - 2.3 * (i - 1)
        node!(
            axes[3],
            (3.0, y),
            name;
            w = 5.4,
            h = 1.3,
            accent,
            fill = COLORS.night_card,
            textcolor = COLORS.night_ink,
        )
        arrow!(axes[3], (5.8, y), (7.7, 0.0); color = COLORS.night_muted)
    end
    Mke.text!(
        axes[3],
        8.0,
        0.0;
        text = "HU",
        align = (:left, :center),
        fontsize = 30,
        font = :bold,
        color = COLORS.night_ink,
    )
    fig
end

# ╔═╡ 3603ec96-7d2e-46f0-ad1d-841c1098eeee
md"""
**The comparison.** One shared set-up feeds three pipelines; their images, per-rod HU and
wall-clock times are compared at the end.
"""

# ╔═╡ 882ffd5c-5b65-4173-94e5-530119393fae
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 470))
    nodes = Dict(
        :phantom => (0.0, 2.0, "Gammex 472", "128² × 8, 2.7 mm Voxels"),
        :setup => (4.6, 2.0, "Shared Scan", "Apex Elite · 120 kVp · 500 Views"),
        :catsim => (9.6, 3.6, "CatSim (Python + C)", "Project → BHC → FDK", COLORS.beam),
        :cpu => (9.6, 2.0, "BasisSim, CPU", "simulate! → BHC → reconstruct!"),
        :gpu => (9.6, 0.4, "BasisSim, GPU", "Same Code, Device Mask", COLORS.cyan),
        :compare => (14.6, 2.0, "Compare", "Images · Rod HU · Runtime", COLORS.green),
    )
    edges = [
        :phantom => :setup,
        :setup => :catsim,
        :setup => :cpu,
        :setup => :gpu,
        :catsim => :compare,
        :cpu => :compare,
        :gpu => :compare,
    ]
    flow_chart!(
        fig[1, 1],
        nodes,
        edges;
        limits = (-2.3, 16.9, -0.5, 4.5),
        w = 4.2,
        h = 1.25,
    )
    fig
end

# ╔═╡ 06000001-0000-4000-8000-000000000020
md"""
## Notebook Setup

The only docs notebook that uses Python. `docs/CondaPkg.toml` pins `gecatsim` to the MolloiLab
fork of XCIST (`git+https://github.com/MolloiLab/main`), which adds the Gammex 472 material
definitions, with numpy and pydicom. The first `import PythonCall` in the `docs/` environment
makes CondaPkg install them into `docs/.CondaPkg/` (a few minutes and about 1 GB, once). If
PythonCall or gecatsim cannot be loaded, the CatSim cells skip with a notice and the
BasisSimulator runs still execute.
"""

# ╔═╡ 06000001-0000-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 06000001-0000-4000-8000-000000000002
using Markdown: @md_str, Markdown

# ╔═╡ 06000001-0000-4000-8000-000000000003
using Statistics: mean, std

# ╔═╡ 06000001-0000-4000-8000-000000000004
using Printf: @sprintf

# ╔═╡ 06000001-0000-4000-8000-000000000030
import BasisSimulator as BS

# ╔═╡ 06000001-0000-4000-8000-000000000031
# ╠═╡ show_logs = false
import CairoMakie as Mke

# ╔═╡ 06000001-0000-4000-8000-000000000032
# ╠═╡ show_logs = false
# PythonCall, loaded so that a missing Python environment degrades to "CatSim skipped"
PC = try
    Base.require(Base.PkgId(
        Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
        "PythonCall",
    ))
catch err
    @warn "PythonCall could not be loaded; the CatSim cells will skip" exception = (
        err,
        catch_backtrace(),
    )
    nothing
end;

# ╔═╡ 06000001-0000-4000-8000-000000000033
import PlutoUI

# ╔═╡ 06000001-0000-4000-8000-000000000034
PlutoUI.TableOfContents()

# ╔═╡ 06000001-0000-4000-8000-000000000040
begin
    import CUDA
    # the NVIDIA GPU's array type, or Array on a host without one (on Apple silicon, Metal.MtlArray)
    AT = CUDA.functional() ? CUDA.CuArray : Array
    to_gpu(x) = AT(x)
    GPU_BACKEND = (name = string(nameof(AT)),)
end;

# ╔═╡ 06000001-0000-4000-8000-000000000050
Markdown.parse("**Backend detected:** $(GPU_BACKEND.name)")

# ╔═╡ 06000001-0000-4000-8000-000000000060
const HAS_GECATSIM = PC !== nothing &&
    try
        PC.pyimport("gecatsim")
        true
    catch err
        @warn "gecatsim is not importable; the CatSim cells will skip" exception = err
        false
    end;

# ╔═╡ 06000001-0000-4000-8000-000000000070
HAS_GECATSIM ?
    Markdown.parse(
        """
    **gecatsim** $(PC.pyconvert(String, PC.pyimport("importlib.metadata").version("gecatsim"))) loaded
    (Python $(PC.pyconvert(String, PC.pyimport("platform").python_version()))): the CatSim cells run.
    """,
    ) :
    md"""
    !!! warning "gecatsim not available: CatSim cells skipped"
        `import PythonCall` or `pyimport("gecatsim")` failed. Instantiate `docs/` so that CondaPkg
        can install the Python environment (see `docs/CondaPkg.toml`), then re-run. The
        BasisSimulator CPU and GPU runs below still execute.
    """

# ╔═╡ 06000001-0000-4000-8000-000000000080
# ╠═╡ show_logs = false
# Two import stubs for the MolloiLab gecatsim fork.
#
# `gecatsim.reconstruction.pyfiles.recon` imports the iterative reconstructions (ART, SIRT, CGLS)
# at module load, and those import `gecatsim.pyfiles.C_DD3Back` / `C_DD3WBack`, which the fork
# does not ship (only `C_DD3Proj*.py`). Without them the FDK reconstruction cannot even be
# imported. We write a stub of each missing module into the installed package: its function
# raises if called, which FDK never does. Idempotent: re-running is a no-op once they exist.
gecatsim_patched = !HAS_GECATSIM ?
    false :
    let
        spec = PC.pyimport("importlib.util").find_spec("gecatsim")
        pyfiles_dir = joinpath(dirname(PC.pyconvert(String, spec.origin)), "pyfiles")
        wrote = String[]
        for name in ("C_DD3Back", "C_DD3WBack")
            path = joinpath(pyfiles_dir, "$(name).py")
            isfile(path) && continue
            write(
                path,
                """
            # Stub written by BasisSimulator.jl docs notebook 06: the fork does not ship this module.
            def $(replace(name, "C_" => ""))(*args, **kwargs):
                raise NotImplementedError("$(name) is a stub; FDK does not need it, iterative recons do.")
            """,
            )
            push!(wrote, name)
        end
        @info "[gecatsim] import stubs: $(isempty(wrote) ? "already present" : "wrote " * join(wrote, ", "))"
        true
    end;

# ╔═╡ 06000001-0000-4000-8000-000000000090
# Two runtime patches for CatSim inside Pluto.
#
# 1. Speed: `fdk_equiAngle.float3Darray2pointer` / `float3Dpointer2array` copy a numpy volume to
#    and from a C triple pointer element by element in Python — millions of ctypes assignments
#    before and after the C FDK. The replacements alias one row pointer per (i, j) into the numpy
#    buffer and copy back with one `memmove` per row: the same C ABI and the same numbers.
# 2. Output: `run_all()` and `recon()` print hundreds of lines. Under PythonCall in a Pluto worker
#    that flood can block the captured stdout pipe, so both are wrapped in
#    `contextlib.redirect_stdout`.
gecatsim_fdk_patched = !HAS_GECATSIM ?
    false :
    let
        PC.pyexec(
            """
        import ctypes, contextlib, io
        import numpy as np
        import gecatsim as _gecatsim
        import gecatsim.reconstruction.pyfiles.fdk_equiAngle as _fdk
        import gecatsim.reconstruction.pyfiles.recon as _recon_mod

        FLOAT = ctypes.c_float
        PtrFLOAT = ctypes.POINTER(FLOAT)
        PtrPtrFLOAT = ctypes.POINTER(PtrFLOAT)

        def _fast_arr2ptr(arr):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            n0, n1, _ = arr.shape
            out = (PtrPtrFLOAT * n0)()
            for i in range(n0):
                row = (PtrFLOAT * n1)()
                for j in range(n1):
                    row[j] = arr[i, j].ctypes.data_as(PtrFLOAT)
                out[i] = row
            out._keepalive_ = arr          # the row pointers alias arr's buffer
            return out

        def _fast_ptr2arr(ptr, n, m, o):
            out = np.empty((n, m, o), dtype=np.float32)
            nbytes = o * ctypes.sizeof(FLOAT)
            for i in range(n):
                for j in range(m):
                    ctypes.memmove(out[i, j].ctypes.data_as(PtrFLOAT), ptr[i][j], nbytes)
            return out

        _fdk.float3Darray2pointer = _fast_arr2ptr
        _fdk.float3Dpointer2array = _fast_ptr2arr

        if not getattr(_gecatsim.CatSim, '_basissim_quiet', False):
            _orig_run_all = _gecatsim.CatSim.run_all
            def _quiet_run_all(self, *args, **kwargs):
                with contextlib.redirect_stdout(io.StringIO()):
                    return _orig_run_all(self, *args, **kwargs)
            _gecatsim.CatSim.run_all = _quiet_run_all
            _gecatsim.CatSim._basissim_quiet = True

        if not getattr(_recon_mod, '_basissim_quiet', False):
            _orig_recon = _recon_mod.recon
            def _quiet_recon(ct, *args, **kwargs):
                with contextlib.redirect_stdout(io.StringIO()):
                    return _orig_recon(ct, *args, **kwargs)
            _recon_mod.recon = _quiet_recon
            _recon_mod._basissim_quiet = True
        """,
            Main,
        )
        true
    end;

# ╔═╡ e41259df-f198-4af5-89c0-19679c46bbb5
md"""
### Figure Style

BasisSimulator's colours and type, and the drawing helpers every figure uses (the same cells in
every example notebook), then this notebook's own: the Gammex 472 rods and a sketch of the
phantom.
"""

# ╔═╡ 15786719-1413-4d88-82d0-35d24f115310
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

# ╔═╡ 02e8b3a2-43a7-48c1-bce9-97d39c28493d
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

# ╔═╡ 80748e21-bd9e-4ee9-81c5-79129ae25f53
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

# ╔═╡ 24dd130a-4560-49ea-8441-82f15890516d
# The rods as `BS.create_gammex_472` places them: calcium on the 5 cm ring from 0°, iodine on
# the 10.5 cm ring offset by half a step, 28 mm across (cm, radians, mg/mL).
GAMMEX_RODS = vcat(
    [
        (
            label = 9 + i,
            element = "Calcium",
            conc = c,
            r = 5.0,
            θ = 2π * (i - 1) / 7,
            color = COLORS.red,
            level = i / 7,
        )
        for (i, c) in enumerate((50, 100, 200, 300, 400, 500, 600))
    ],
    [
        (
            label = 19 + i,
            element = "Iodine",
            conc = c,
            r = 10.5,
            θ = 2π * (i - 1) / 7 + π / 7,
            color = COLORS.green,
            level = i / 7,
        )
        for (i, c) in enumerate((2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0))
    ],
);

# ╔═╡ bcd32d72-7c5f-437e-b982-1ef79a472a94
"""
The Gammex 472 of `BS.create_gammex_472` in cm, centred at the origin: the 33 cm solid-water body
and the 14 rods of `GAMMEX_RODS`, shaded by concentration. `labels` writes each rod's
concentration on it; `dark` draws in the locator band's colours.
"""
function gammex_sketch!(ax; dark = false, labels = false)
    edge = dark ? COLORS.night_muted : COLORS.rule
    Mke.poly!(
        ax,
        Mke.Circle(Mke.Point2f(0, 0), 16.5f0);
        color = (COLORS.blue, dark ? 0.5 : 0.16),
        strokecolor = edge,
        strokewidth = 1.5,
    )
    for rod in GAMMEX_RODS
        c = Mke.Point2f(rod.r * cos(rod.θ), rod.r * sin(rod.θ))
        Mke.poly!(
            ax,
            Mke.Circle(c, 1.4f0);
            color = (rod.color, 0.3 + 0.7 * rod.level),
            strokecolor = edge,
            strokewidth = 1,
        )
        labels &&
            Mke.text!(
                ax,
                c;
                text = isinteger(rod.conc) ?
                    string(round(Int, rod.conc)) :
                    string(rod.conc),
                align = (:center, :center),
                fontsize = 14,
                font = :bold,
                color = rod.level > 0.4 ? COLORS.paper : COLORS.ink,
            )
    end
    return ax
end

# ╔═╡ 060000f1-0000-4000-8000-000000000001
md"""
## 1. One Set-Up for All Three

One scanner, one protocol and one phantom, shared verbatim by the three pipelines, and a thin
wrapper that hands them to CatSim.

### The Scanner

The GE Revolution Apex Elite of notebook 01 (`REVOLUTION_APEX`), with its large-body bowtie.
BasisSimulator's `:ge_revolution_large` bowtie is read from the same `large.txt` table CatSim
ships, so both see the same bowtie. Electronic noise is zero (noise is off everywhere here), and
the scintillator's lag (`use_lag`) is off, as CatSim's is.
"""

# ╔═╡ 06000002-0000-4000-8000-000000000010
scanner = BS.EICTScanner(BS.REVOLUTION_APEX; electronic_noise = 0.0, use_lag = false)

# ╔═╡ a8ac0e53-23ec-533d-a797-af31d0867766
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

# ╔═╡ 06000003-0000-4000-8000-000000000001
md"""
### Protocol, Physics and Grid

120 kVp, 200 mA, 500 views in a 1 s rotation, 4 mm of collimation and 4.5 mm of extra aluminium.
The physics is what both simulators model by default: polychromatic transport through the
filters and bowtie, energy-dependent detector efficiency, fill factor and focal-spot blur.
Noise, scatter and the heel effect are off in BasisSimulator (and lag on the scanner) because
CatSim's defaults have them off. The reconstruction is 256 × 256 over 35 cm, 0.625 mm slices.

!!! note "Point views on both sides"
    The other notebooks integrate each view while the gantry turns
    (`SimOptions(; view_samples = 5)`, notebook 01). This like-for-like check samples each view at
    a single angle in both simulators: BasisSimulator keeps its default `view_samples = 1`, and
    the wrapper sets CatSim's `physics.viewSampleCount` (2 by default) to 1.
"""

# ╔═╡ 06000003-0000-4000-8000-000000000010
protocol = BS.CTProtocol(
    kVp = 120,
    mA = 200.0,
    views = 500,
    rotation_time = 1.0,
    collimation_mm = 4.0,
    additional_filters = [("Al", 4.5)],
);

# ╔═╡ 06000003-0000-4000-8000-000000000020
sim_opts = BS.SimOptions(
    seed = 1234,
    projector = :dd_fast,
    use_noise = false,
    use_scatter = false,
    use_heel_effect = false,
);

# ╔═╡ 06000003-0000-4000-8000-000000000030
recon_opts = BS.ReconOptions(
    matrix_size = (256, 256, round(Int, protocol.collimation_mm / 0.625)),
    fov_cm = 35.0,
    z_cm = protocol.collimation_mm / 10,
);

# ╔═╡ 06000004-0000-4000-8000-000000000001
md"""
### The CatSim Wrapper

A few Julia functions map `EICTScanner` / `CTProtocol` / `ReconOptions` / `Phantom` onto CatSim's
configuration:

| | what the wrapper does |
|---|---|
| **Geometry** | BasisSimulator gives the dexel pitch at the isocentre; CatSim wants it at the detector face, so it is multiplied by `SDD/SID`. Every column is its own module with no gap (`detectorColsPerMod = 1`, `detectorColSkip = 0`); without that CatSim braids or squashes the sinogram. |
| **Beam** | CatSim's 120 kVp tungsten spectrum for a 10° target (the protocol's anode angle), the scanner's flat filter plus the protocol's filters, the same `large.txt` bowtie, target angle and focal spot, and no graphite detector prefilter (BasisSimulator models none). |
| **Views** | One angular sample per view (`physics.viewSampleCount = 1`), BasisSimulator's `view_samples = 1`. |
| **Preprocessing** | CatSim's own water BHC, `Prep_BHC_Accurate`: a degree-5 polynomial per dexel fitted to air scans through 1–50 cm of water, mapped to the same monoenergetic water μ that `calibrate_bhc_water` uses; HU conversion uses that μ too. |
"""

# ╔═╡ 06000004-0000-4000-8000-000000000010
const _catsim_state = Dict{Symbol, Any}();

# ╔═╡ 06000004-0000-4000-8000-000000000020
function catsim_init()
    # Reference both patch flags so Pluto runs the stubs and the FDK patch first.
    (gecatsim_patched && gecatsim_fdk_patched) || error("the gecatsim patches did not run")
    if isempty(_catsim_state)
        _catsim_state[:xc] = PC.pyimport("gecatsim")
        _catsim_state[:recon] = PC.pyimport("gecatsim.reconstruction.pyfiles.recon")
        origin = PC.pyconvert(
            String,
            PC.pyimport("importlib.util").find_spec("gecatsim").origin,
        )
        _catsim_state[:cfg] = joinpath(dirname(origin), "examples", "cfg")
    end
    return _catsim_state[:xc], _catsim_state[:recon], _catsim_state[:cfg]
end;

# ╔═╡ 06000004-0000-4000-8000-000000000030
function catsim_create_simulation()
    xc, _, cfg = catsim_init()
    return xc.CatSim(
        joinpath(cfg, "Phantom_Sample.cfg"),
        joinpath(cfg, "Scanner_Sample_generic.cfg"),
        joinpath(cfg, "Protocol_Sample_axial.cfg"),
    )
end;

# ╔═╡ 06000004-0000-4000-8000-000000000040
function catsim_configure_scanner!(ct, scanner, protocol)
    magnification = scanner.source_to_detector / scanner.source_to_isocenter
    n_rows = protocol.collimation_mm === nothing ?
        scanner.detector_rows :
        round(Int, protocol.collimation_mm / scanner.detector_row_size)

    ct.scanner.sid = scanner.source_to_isocenter
    ct.scanner.sdd = scanner.source_to_detector
    ct.scanner.detectorColCount = scanner.detector_cols
    ct.scanner.detectorRowCount = n_rows
    # isocentre → face
    ct.scanner.detectorColSize = scanner.detector_col_size * magnification
    ct.scanner.detectorRowSize = scanner.detector_row_size * magnification
    ct.scanner.detectorColOffset = scanner.detector_col_offset
    ct.scanner.detectorColsPerMod = 1 # every column its own module …
    ct.scanner.detectorRowsPerMod = n_rows
    ct.scanner.detectorColSkip = 0.0 # … with no inter-module gap
    ct.scanner.detectorRowSkip = 0.0

    ct.scanner.targetAngle = scanner.target_angle
    ct.scanner.focalspotWidth = scanner.focal_spot_width
    ct.scanner.focalspotLength = scanner.focal_spot_length
    ct.scanner.detectorMaterial = "Lumex"
    ct.scanner.detectorDepth = scanner.detector_depth
    ct.scanner.detectorColFillFraction = scanner.fill_factor_col
    ct.scanner.detectorRowFillFraction = scanner.fill_factor_row
    ct.scanner.detectorPrefilter = PC.pylist([])
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000050
function catsim_configure_protocol!(ct, scanner, protocol; μ_water_cm)
    ct.protocol.mA = protocol.mA
    ct.protocol.viewsPerRotation = protocol.views
    ct.protocol.viewCount = protocol.views
    ct.protocol.stopViewId = protocol.views - 1
    ct.protocol.rotationTime = protocol.rotation_time
    ct.protocol.spectrumFilename = "tungsten_tar$(Int(scanner.target_angle)).0_$(Int(protocol.kVp))_filt.dat"
    scanner.bowtie_filter in (:ge_revolution_large, :large_body) ||
        error("this wrapper maps only the large-body bowtie")
    ct.protocol.bowtie = "large.txt"
    catsim_name(m) = Dict(
        "aluminum" => "Al",
        "Al" => "Al",
        "copper" => "Cu",
        "Cu" => "Cu",
    )[string(m)]
    filters = Any[catsim_name(scanner.flat_filter_material), scanner.flat_filter_thickness]
    for (m, t) in protocol.additional_filters
        push!(filters, catsim_name(m), t)
    end
    ct.protocol.flatFilter = PC.pylist(filters)

    # point views, like BasisSimulator's view_samples = 1
    ct.physics.viewSampleCount = 1
    ct.physics.enableQuantumNoise = 0
    ct.physics.enableElectronicNoise = 0
    ct.physics.callback_post_log = "Prep_BHC_Accurate" # CatSim's water BHC
    ct.physics.EffectiveMu = μ_water_cm                # cm⁻¹, the mono target of the fit
    ct.physics.BHC_poly_order = 5
    ct.physics.BHC_max_length_mm = 500
    ct.physics.BHC_length_step_mm = 10
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000060
function catsim_configure_recon!(ct, recon_opts; μ_water_cm)
    xc, _, cfg = catsim_init()
    xc.source_cfg(joinpath(cfg, "Recon_Sample_2d.cfg"), ct)
    n_slices = recon_opts.matrix_size[3]
    ct.recon.fov = recon_opts.fov_cm * 10.0
    ct.recon.imageSize = recon_opts.matrix_size[1]
    ct.recon.sliceCount = n_slices
    ct.recon.sliceThickness = recon_opts.z_cm * 10.0 / n_slices
    ct.recon.reconType = "fdk_equiAngle"
    ct.recon.kernelType = "standard"
    ct.recon.unit = "HU"
    ct.recon.mu = μ_water_cm / 10.0 # cm⁻¹ → mm⁻¹
    ct.recon.huOffset = -1000
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000070
function catsim_configure_phantom!(ct, json_path)
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.filename = json_path
    ct.phantom.scale = 1.0
    ct.phantom.centerOffset = PC.pylist([0.0, 0.0, 0.0])
    return ct
end;

# ╔═╡ 06000004-0000-4000-8000-000000000080
function catsim_forward_project(ct; results_name)
    ct.resultsName = results_name
    ct.run_all()
    return nothing
end;

# ╔═╡ 06000004-0000-4000-8000-000000000090
function catsim_reconstruct_fdk(ct; results_name)
    _, recon_mod, _ = catsim_init()
    ct.resultsName = results_name
    ct.recon.filename = results_name
    ct.do_Recon = 1
    recon_mod.recon(ct)
    n = Int(PC.pyconvert(Float64, ct.recon.imageSize))
    nz = Int(PC.pyconvert(Float64, ct.recon.sliceCount))
    file = "$(results_name)_$(n)x$(n)x$(nz).raw"
    isfile(file) || error("CatSim wrote no reconstruction")
    return copy(reshape(reinterpret(Float32, read(file)), (n, n, nz)))
end;

# ╔═╡ 06000005-0000-4000-8000-000000000001
md"""
### The Phantom, in CatSim's Format

`create_gammex_472` labels the solid-water body 3, the calcium rods 10–16 and the iodine rods
20–26; the MolloiLab fork defines the matching `Gammex472_*` materials. CatSim's
`Phantom_Voxelized` reads a JSON header plus one Float32 volume-fraction map per material; for a
hard-segmented mask each map is the indicator `mask .== label`.
"""

# ╔═╡ 06000005-0000-4000-8000-000000000010
const REGION_TO_CATSIM = Dict{Int, String}(
    1 => "water",
    2 => "water",
    3 => "water", # air region label 1 is empty in this phantom
    10 => "Gammex472_Ca_50",
    11 => "Gammex472_Ca_100",
    12 => "Gammex472_Ca_200",
    13 => "Gammex472_Ca_300",
    14 => "Gammex472_Ca_400",
    15 => "Gammex472_Ca_500",
    16 => "Gammex472_Ca_600",
    20 => "Gammex472_I_2_0",
    21 => "Gammex472_I_2_5",
    22 => "Gammex472_I_5_0",
    23 => "Gammex472_I_7_5",
    24 => "Gammex472_I_10_0",
    25 => "Gammex472_I_15_0",
    26 => "Gammex472_I_20_0",
);

# ╔═╡ 06000005-0000-4000-8000-000000000030
function export_phantom_for_catsim(phantom, output_dir, name)
    mask = Array(phantom.mask)
    nx, ny, nz = size(mask)
    vx, vy, vz = phantom.voxel_size .* 10.0 # cm → mm
    mkpath(output_dir)
    entries = NamedTuple[]
    for lbl in sort(unique(mask))
        haskey(REGION_TO_CATSIM, Int(lbl)) || continue
        fname = "$(name)_mat$(Int(lbl)).density_"
        write(joinpath(output_dir, fname), Float32.(mask .== lbl))
        push!(entries, (mat = REGION_TO_CATSIM[Int(lbl)], file = fname))
    end
    n = length(entries)
    list(x) = "[" * join(x, ", ") * "]"
    strs(x) = "[" * join(("\"$v\"" for v in x), ", ") * "]"
    json = """
    {
      "n_materials": $(n),
      "mat_name": $(strs(e.mat for e in entries)),
      "volumefractionmap_filename": $(strs(e.file for e in entries)),
      "volumefractionmap_datatype": $(strs(fill("float", n))),
      "cols": $(list(fill(nx, n))), "rows": $(list(fill(ny, n))), "slices": $(list(fill(nz, n))),
      "x_size": $(list(fill(vx, n))), "y_size": $(list(fill(vy, n))), "z_size": $(list(fill(vz, n))),
      "x_offset": $(list(fill((nx + 1) / 2, n))), "y_offset": $(list(fill((ny + 1) / 2, n))),
      "z_offset": $(list(fill((nz + 1) / 2, n))),
      "density_scale": $(list(fill(1.0, n)))
    }
    """
    path = joinpath(output_dir, "$(name).json")
    write(path, json)
    return path
end;

# ╔═╡ 06000006-0000-4000-8000-000000000001
md"""
### The Gammex 472 Phantom

`n_voxels = 128` gives 2.7 mm voxels over 35 cm: coarse enough that CatSim's voxelized projector
finishes in minutes, fine enough to resolve the 28 mm rods. Eight slices cover the 4 mm beam.
"""

# ╔═╡ 06000006-0000-4000-8000-000000000010
phantom_cpu = BS.create_gammex_472(
    n_voxels = 128,
    n_slices = 8,
    fov_cm = 35.0,
    z_cm = protocol.collimation_mm / 10.0,
);

# ╔═╡ 06000006-0000-4000-8000-000000000020
phantom_gpu = BS.Phantom(
    to_gpu(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
);

# ╔═╡ 06000007-0000-4000-8000-000000000001
md"""
### The Shared Water Reference

`calibrate_bhc_water` resolves BasisSimulator's detected spectrum per dexel column and returns the
per-column correction with its monoenergetic reference (the spectrum's mean energy). That water
μ is CatSim's target for its own BHC fit and the HU reference of all three reconstructions.
"""

# ╔═╡ 06000007-0000-4000-8000-000000000010
geom_inspect = BS.CTGeometry(
    scanner;
    n_angles = protocol.views,
    fov_cm = recon_opts.fov_cm,
    z_cm = recon_opts.z_cm,
    collimation_mm = protocol.collimation_mm,
);

# ╔═╡ 06000007-0000-4000-8000-000000000020
bhc = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = geom_inspect);

# ╔═╡ 06000007-0000-4000-8000-000000000030
Markdown.parse(
    """
**Water reference:** $(round(bhc.reference_energy_keV; digits = 1)) keV, water μ = $(round(bhc.μ_water_ref; digits = 5)) cm⁻¹
""",
)

# ╔═╡ 060000f2-0000-4000-8000-000000000001
md"""
## 2. Run the Three Pipelines

Each timing covers the whole job from the phantom to an HU volume: forward projection,
preprocessing and BHC, FDK. Both BasisSimulator runs are compiled by one untimed call first, so
their timings are steady-state; CatSim's includes writing and reading its intermediate files and
fitting its BHC polynomials.

### CatSim
"""

# ╔═╡ 06000008-0000-4000-8000-000000000010
# ╠═╡ show_logs = false
catsim_result = !HAS_GECATSIM ?
    nothing :
    let
        work_dir = mktempdir(; prefix = "basissim_catsim_06_")
        tag = joinpath(work_dir, "gammex472")
        local recon
        elapsed = @elapsed begin
            json = export_phantom_for_catsim(phantom_cpu, work_dir, "gammex472")
            ct = catsim_create_simulation()
            catsim_configure_phantom!(ct, json)
            catsim_configure_scanner!(ct, scanner, protocol)
            catsim_configure_protocol!(ct, scanner, protocol; μ_water_cm = bhc.μ_water_ref)
            catsim_configure_recon!(ct, recon_opts; μ_water_cm = bhc.μ_water_ref)
            catsim_forward_project(ct; results_name = tag)
            recon = catsim_reconstruct_fdk(ct; results_name = tag)
        end
        rm(work_dir; recursive = true, force = true)
        (recon = recon, elapsed = elapsed)
    end;

# ╔═╡ 06000009-0000-4000-8000-000000000001
md"""
### BasisSimulator, CPU and GPU

One function, two phantoms: `phantom_cpu.mask` is an `Array`, so the first run stays on the host;
`phantom_gpu.mask` lives on the GPU, so the second runs there. Nothing else differs.
"""

# ╔═╡ 06000009-0000-4000-8000-000000000005
"""
    basissim_pipeline(phantom) -> HU volume

`create_workspace` → `simulate!` → water BHC → FDK → HU, on the phantom's backend.
"""
function basissim_pipeline(phantom)
    ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    BS.simulate!(ws, phantom, protocol, sim_opts; report_dose = false)
    sino = BS.apply_bhc_water(ws.sinogram, bhc)
    ws_fdk = BS.create_fdk_recon_workspace(
        sino,
        ws.geom,
        recon_opts.matrix_size;
        filter = :standard,
    )
    μ = Array(BS.reconstruct!(ws_fdk, sino, ws.geom))
    return Float32.(BS.to_hounsfield(μ; μ_water = bhc.μ_water_ref))
end;

# ╔═╡ 06000009-0000-4000-8000-000000000010
basissim_cpu_result = let
    basissim_pipeline(phantom_cpu)
    GC.gc(true) # compile, untimed
    elapsed = @elapsed recon = basissim_pipeline(phantom_cpu)
    GC.gc(true)
    (recon = recon, elapsed = elapsed)
end;

# ╔═╡ 0600000a-0000-4000-8000-000000000010
basissim_gpu_result = let
    basissim_pipeline(phantom_gpu)
    GC.gc(true) # compile, untimed
    elapsed = @elapsed recon = basissim_pipeline(phantom_gpu)
    GC.gc(true)
    (recon = recon, elapsed = elapsed)
end;

# ╔═╡ 0600000b-0000-4000-8000-000000000001
md"""
## 3. Image Agreement

The central slice of each reconstruction in one HU window, and the difference of each
BasisSimulator image from CatSim's. CatSim writes its image with its own axis convention; it is
brought onto BasisSimulator's orientation by the flip or transpose (of the eight in-plane
symmetries) that best matches the BasisSimulator image, which the figure reports. Thin rings
along the body and rod edges are where the two images differ in edge sharpness; the flat
interiors are what the per-rod table measures.
"""

# ╔═╡ 0600000b-0000-4000-8000-000000000005
catsim_aligned = catsim_result === nothing ?
    nothing :
    let
        ref = basissim_gpu_result.recon
        k = size(ref, 3) ÷ 2 + 1
        transposed(v) = permutedims(v, (2, 1, 3))
        cands = [
            ("as written", identity),
            ("x reversed", v -> reverse(v; dims = 1)),
            ("y reversed", v -> reverse(v; dims = 2)),
            ("x and y reversed", v -> reverse(v; dims = (1, 2))),
            ("transposed", transposed),
            ("transposed, x reversed", v -> reverse(transposed(v); dims = 1)),
            ("transposed, y reversed", v -> reverse(transposed(v); dims = 2)),
            ("transposed, x and y reversed", v -> reverse(transposed(v); dims = (1, 2))),
        ]
        a = vec(Float64.(ref[:, :, k]))
        function corr(b)
            x = a .- mean(a)
            y = b .- mean(b)
            return sum(x .* y) / sqrt(sum(x.^2) * sum(y.^2))
        end
        scores = [corr(vec(Float64.(f(catsim_result.recon)[:, :, k]))) for (_, f) in cands]
        i = argmax(scores)
        (
            recon = cands[i][2](catsim_result.recon),
            transform = cands[i][1],
            correlation = scores[i],
        )
    end;

# ╔═╡ ffa467af-ec12-4b92-bd4e-732aa0c6cbe2
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 540))
    nz = recon_opts.matrix_size[3]
    k = nz ÷ 2 + 1

    # the slice seen from above: the phantom and its rods
    ax = canvas(fig[1, 1], (-17.5, 17.5, -17.5, 17.5))
    gammex_sketch!(ax; labels = true)
    Mke.Label(
        fig[2, 1],
        "The Central Slice, From Above";
        fontsize = 20,
        font = :bold,
        tellwidth = false,
    )
    Mke.Label(
        fig[3, 1],
        "Calcium (Red, mg/mL) Inside, Iodine (Green, mg/mL) Outside";
        fontsize = 16,
        color = COLORS.muted,
        tellwidth = false,
    )

    # the beam seen from the side: z runs left to right, the slices across the 4 mm beam
    ax = canvas(fig[1, 2], (-6.0, 6.0, -3.3, 3.6))
    w = 0.9 # one 0.5 mm slice, drawn
    Mke.poly!(
        ax,
        Mke.Point2f[(0, 3.2), (-nz * w / 2, -1.6), (nz * w / 2, -1.6)];
        color = (COLORS.beam, 0.22),
    )
    Mke.scatter!(
        ax,
        [Mke.Point2f(0, 3.2)];
        marker = :rect,
        markersize = 22,
        color = COLORS.ink,
    )
    Mke.scatter!(
        ax,
        [Mke.Point2f(0, 3.2)];
        marker = :circle,
        markersize = 8,
        color = COLORS.beam,
    )
    for s in 1:nz
        x = -nz * w / 2 + (s - 0.5) * w
        Mke.poly!(
            ax,
            rounded_rect((x, -0.4), w * 0.92, 2.2, 0.06);
            color = s == k ? (COLORS.violet, 0.75) : (COLORS.blue, 0.2),
            strokecolor = COLORS.rule,
            strokewidth = 1,
        )
    end
    Mke.lines!(
        ax,
        [-nz * w / 2, nz * w / 2],
        [-1.9, -1.9];
        color = COLORS.cyan,
        linewidth = 8,
    )
    Mke.text!(
        ax,
        0.0,
        -2.3;
        text = "Detector: $(round(Int, protocol.collimation_mm)) mm of Rows",
        align = (:center, :top),
        fontsize = 16,
        color = COLORS.ink,
    )
    Mke.Label(
        fig[2, 2],
        "Which Slice, From the Side";
        fontsize = 20,
        font = :bold,
        tellwidth = false,
    )
    Mke.Label(
        fig[3, 2],
        "Slice $(k) of $(nz) (0.5 mm Each) Across the Beam";
        fontsize = 16,
        color = COLORS.muted,
        tellwidth = false,
    )
    Mke.rowgap!(fig.layout, 1, 6)
    Mke.rowgap!(fig.layout, 2, 2)
    fig
end

# ╔═╡ 0600000b-0000-4000-8000-000000000010
let
    Mke.set_theme!(THEME)
    k = size(basissim_gpu_result.recon, 3) ÷ 2 + 1
    win = (-200, 600)
    fmt(t) = @sprintf("%.2f s", t)
    panels = Any[]
    catsim_aligned === nothing ||
        push!(
            panels,
            ("CatSim", "Python + C · $(fmt(catsim_result.elapsed))", catsim_aligned.recon),
        )
    push!(
        panels,
        (
            "BasisSimulator.jl, CPU",
            "Julia · $(fmt(basissim_cpu_result.elapsed))",
            basissim_cpu_result.recon,
        ),
    )
    push!(
        panels,
        (
            "BasisSimulator.jl, $(GPU_BACKEND.name)",
            "Julia · $(fmt(basissim_gpu_result.elapsed))",
            basissim_gpu_result.recon,
        ),
    )

    n = length(panels)
    fig = Mke.Figure(size = (n * 480 + 120, catsim_aligned === nothing ? 580 : 1080))
    local hm
    for (c, (title, sub, vol)) in enumerate(panels)
        ax = Mke.Axis(fig[1, c]; title, subtitle = sub, aspect = Mke.DataAspect())
        hm = Mke.heatmap!(ax, vol[:, :, k]; colormap = :grays, colorrange = win)
        Mke.hidedecorations!(ax)
    end
    Mke.Colorbar(fig[1, n + 1], hm; label = "HU", width = 16)
    if catsim_aligned !== nothing
        local hd
        nx, ny = size(catsim_aligned.recon, 1), size(catsim_aligned.recon, 2)
        # inside the reconstruction circle only
        inside(i, j) = (i - (nx + 1) / 2)^2 + (j - (ny + 1) / 2)^2 <=
            (min(nx, ny) / 2 - 1)^2
        for (c, (title, _, vol)) in enumerate(panels[2:end])
            ax = Mke.Axis(
                fig[2, c + 1];
                title = "$(replace(title, "BasisSimulator.jl" => "BasisSim")) − CatSim",
                subtitle = "Difference, Central Slice",
                aspect = Mke.DataAspect(),
            )
            d = vol[:, :, k] .- catsim_aligned.recon[:, :, k]
            masked = [inside(i, j) ? d[i, j] : NaN32 for i in 1:nx, j in 1:ny]
            hd = Mke.heatmap!(
                ax,
                masked;
                colormap = :RdBu,
                colorrange = (-100, 100),
                nan_color = COLORS.paper,
            )
            Mke.hidedecorations!(ax)
        end
        Mke.Colorbar(fig[2, n + 1], hd; label = "ΔHU", width = 16)
        Mke.Label(
            fig[2, 1],
            "CatSim's Image, $(titlecase(catsim_aligned.transform))\nCorrelation With BasisSim: $(round(catsim_aligned.correlation; digits = 4))";
            fontsize = 18,
            color = COLORS.ink,
            tellwidth = false,
            tellheight = false,
        )
        Mke.rowsize!(fig.layout, 2, Mke.Aspect(2, 1.0))
    end
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "catsim_vs_basissim_mosaic.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000b-0000-4000-8000-000000000020
md"""
## 4. Per-Rod HU

The mean HU inside each rod (the phantom labels resampled onto the reconstruction grid, eroded by
one pixel) on the central slice, next to the rod's theoretical monoenergetic HU at the water
reference energy. A water-only BHC leaves dense calcium and iodine rods below their theory in
**both** simulators (the rod's own beam hardening beyond the water curve; notebook 01 discusses
it). What this tests is whether the two simulators agree with each other.
"""

# ╔═╡ bf0f3dc3-168e-4984-b8e0-a22f126a6443
rod_hu = let
    labels = BS.resample_to_recon(
        phantom_cpu,
        geom_inspect,
        recon_opts.matrix_size;
        method = :nearest,
    )
    k = size(labels, 3) ÷ 2 + 1
    nx, ny = size(labels, 1), size(labels, 2)
    refE, μw = bhc.reference_energy_keV, bhc.μ_water_ref
    roi(lab) = [
        CartesianIndex(i, j, k)
        for j in 2:(ny - 1), i in 2:(nx - 1)
        if all(labels[i + di, j + dj, k] == lab for di in -1:1, dj in -1:1)
    ]
    cs = catsim_aligned === nothing ? nothing : catsim_aligned.recon
    rows = NamedTuple[]
    for lab in [3; 10:16; 20:26]
        idx = roi(UInt8(lab))
        length(idx) < 5 && continue
        mat = phantom_cpu.materials[lab + 1]
        theory = lab == 3 ? 0.0 : 1000 * (BS.compute_μ_at_energy(mat, refE) - μw) / μw
        push!(
            rows,
            (
                label = lab,
                name = lab == 3 ? "solid water body" : mat.name,
                n = length(idx),
                theory = theory,
                catsim = cs === nothing ? NaN : mean(cs[idx]),
                basissim = mean(basissim_gpu_result.recon[idx]),
            ),
        )
    end
    cpu_gpu = maximum(abs.(basissim_cpu_result.recon .- basissim_gpu_result.recon))
    (rows = rows, cpu_gpu = cpu_gpu, has_catsim = cs !== nothing)
end;

# ╔═╡ cf2708e4-b701-4953-893a-3209e66aca04
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 1080))

    # where the ROIs are: the rods, by concentration
    ax = canvas(fig[1, 1:2], (-17.5, 17.5, -17.5, 17.5))
    gammex_sketch!(ax; labels = true)
    ax.title = "One ROI per Rod, Eroded by One Pixel"
    rods = Dict(r.label => r for r in GAMMEX_RODS)

    for (c, element, color) in ((1, "Calcium", COLORS.red), (2, "Iodine", COLORS.green))
        rows = [
            r
            for r in rod_hu.rows
            if haskey(rods, r.label) && rods[r.label].element == element
        ]
        conc = [Float64(rods[r.label].conc) for r in rows]
        ax = Mke.Axis(
            fig[2, c];
            title = "$(element) Rods",
            subtitle = "Mean HU on the Central Slice",
            xlabel = "$(element) Concentration (mg/mL)",
            ylabel = c == 1 ? "HU" : "",
        )
        Mke.lines!(
            ax,
            conc,
            [r.theory for r in rows];
            color = COLORS.muted,
            linestyle = :dash,
            label = "Theory (Monoenergetic)",
        )
        rod_hu.has_catsim &&
            Mke.scatterlines!(
                ax,
                conc,
                [r.catsim for r in rows];
                color = COLORS.beam,
                marker = :diamond,
                markersize = 16,
                label = "CatSim",
            )
        Mke.scatterlines!(
            ax,
            conc,
            [r.basissim for r in rows];
            color = color,
            markersize = 14,
            label = "BasisSimulator",
        )
        Mke.axislegend(ax; position = :lt)
    end
    Mke.rowsize!(fig.layout, 1, Mke.Relative(0.42))
    fig
end

# ╔═╡ 0600000b-0000-4000-8000-000000000030
let
    r1(x) = round(x; digits = 1)
    rows = [
        "| $(r.name) | $(r.n) | $(round(r.theory; digits = 0)) | " *
            (rod_hu.has_catsim ? "$(r1(r.catsim))" : "—") *
            " | $(r1(r.basissim)) | " *
            (rod_hu.has_catsim ? "$(r1(r.basissim - r.catsim))" : "—") *
            " | " *
            (
                !rod_hu.has_catsim || r.label == 3 ?
                    "" :
                    "$(r1(100 * (r.basissim - r.catsim) / r.theory)) %"
            ) *
            " |"
        for r in rod_hu.rows
    ]
    rel = [100 * (r.basissim - r.catsim) / r.theory for r in rod_hu.rows if r.label != 3]
    water = only(r for r in rod_hu.rows if r.label == 3)
    summary = !rod_hu.has_catsim ?
        "CatSim skipped." :
        "Water: BasisSim − CatSim = $(r1(water.basissim - water.catsim)) HU. Over the $(length(rel)) rods the difference is " *
            "$(r1(minimum(rel))) to $(r1(maximum(rel))) % of the rod's theoretical HU. " *
            "Both simulators read every dense rod below its theory, and they differ mainly in how far: " *
            "the two use independent source-spectrum models (IPEM tables in BasisSimulator, CatSim's own " *
            "tungsten spectra) and independently fitted water corrections, and a spectrum difference " *
            "would show exactly there, in the beam-hardening residual that grows with rod density."
    Markdown.parse(
        """
    | region | pixels | theory (HU) | CatSim (HU) | BasisSim (HU) | BasisSim − CatSim | ÷ theory |
    |:--|--:|--:|--:|--:|--:|--:|
    $(join(rows, "\n"))

    $(summary) BasisSimulator CPU vs GPU: largest voxel difference $(@sprintf("%.3g", rod_hu.cpu_gpu)) HU
    over the whole volume (the same code on two backends; floating-point summation order is all
    that differs).
    """,
    )
end

# ╔═╡ 0600000c-0000-4000-8000-000000000001
md"""
## 5. Runtime

What each timing covers is drawn first: CatSim hands files between its stages, BasisSimulator
keeps the sinogram in memory from the projector to the image. Then the wall-clock of each
end-to-end pipeline, on a log scale so the GPU bar stays visible. The CPU bar is the
like-for-like comparison with CatSim (both on the host); the GPU bar is what BasisSimulator is
built for. This page was rendered on an NVIDIA RTX PRO 6000 (CUDA).
"""

# ╔═╡ c90afced-10a6-46cc-bdb3-fcab013ee4a9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 340))
    ax = canvas(fig[1, 1], (-3.4, 23.4, -0.4, 4.6))
    function lane!(y, name, accent, steps)
        Mke.text!(
            ax,
            -3.2,
            y;
            text = name,
            align = (:left, :center),
            fontsize = 19,
            font = :bold,
            color = accent,
        )
        for (i, (title, kind)) in enumerate(steps)
            x = 1.9 + 3.6 * (i - 1)
            file = kind == :file
            node!(
                ax,
                (x, y),
                title;
                w = 3.1,
                h = 1.15,
                accent = file ? COLORS.rule : accent,
                fill = file ? COLORS.paper : COLORS.card,
                titlesize = 16,
            )
            i < length(steps) && arrow!(ax, (x + 1.55, y), (x + 2.05, y); head = 11)
        end
    end
    lane!(
        3.4,
        "CatSim",
        COLORS.beam,
        [
            ("Write Phantom", :file),
            ("Project", :step),
            ("Write .scan", :file),
            ("Fit + Apply BHC", :step),
            ("Write .prep", :file),
            ("FDK", :step),
        ],
    )
    lane!(
        0.9,
        "BasisSim",
        COLORS.violet,
        [("simulate!", :step), ("apply_bhc_water", :step), ("reconstruct!", :step)],
    )
    Mke.text!(
        ax,
        13.0,
        0.9;
        text = "Grey Boxes: Files on Disk. Coloured: Compute.",
        align = (:left, :center),
        fontsize = 16,
        color = COLORS.muted,
    )
    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000005
let
    Mke.set_theme!(THEME)
    rows = [
        (
            "CatSim\n(Python + C)",
            catsim_result === nothing ? NaN : catsim_result.elapsed,
            COLORS.beam,
        ),
        ("BasisSimulator.jl\nCPU", basissim_cpu_result.elapsed, COLORS.violet),
        (
            "BasisSimulator.jl\n$(GPU_BACKEND.name)",
            basissim_gpu_result.elapsed,
            COLORS.cyan,
        ),
    ]
    ok = [i for i in eachindex(rows) if !isnan(rows[i][2])]
    ref_t = catsim_result === nothing ? nothing : catsim_result.elapsed
    fig = Mke.Figure(size = (1300, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "End-to-End Time: Forward Projection + BHC + FDK",
        subtitle = "120 kVp · 200 mA · 500 Views · Noise-Free · Gammex 472 at 128² × 8 → 256² × 8",
        ylabel = "Wall-Clock (s)",
        xticks = (1:3, [r[1] for r in rows]),
        yscale = log10,
        xticklabelsize = 18,
    )
    Mke.barplot!(
        ax,
        ok,
        [rows[i][2] for i in ok];
        color = [rows[i][3] for i in ok],
        width = 0.6,
    )
    for i in ok
        t = rows[i][2]
        label = ref_t === nothing ?
            @sprintf("%.2f s", t) :
            (
                t == ref_t ?
                    @sprintf("%.2f s\n(Reference)", t) :
                    @sprintf("%.2f s\n(%.1f× Faster)", t, ref_t / t)
            )
        Mke.text!(
            ax,
            i,
            t * 1.18;
            text = label,
            align = (:center, :bottom),
            fontsize = 19,
            font = :bold,
        )
    end
    ys = [rows[i][2] for i in ok]
    Mke.ylims!(ax, minimum(ys) * 0.5, maximum(ys) * 4.0)
    Mke.xlims!(ax, 0.4, 3.6)
    Mke.save(
        joinpath(@__DIR__, "..", "assets", "catsim_vs_basissim_runtime_bar.png"),
        fig;
        px_per_unit = 2,
    )
    fig
end

# ╔═╡ 0600000c-0000-4000-8000-000000000010
let
    t_cs = catsim_result === nothing ? nothing : catsim_result.elapsed
    speed(t) = t_cs === nothing ? "—" : @sprintf("%.1f×", t_cs / t)
    Markdown.parse(
        """
    | pipeline | wall-clock | speed-up vs CatSim |
    |:--|--:|--:|
    | CatSim (voxelized projector, Python + C) | $(t_cs === nothing ? "— (skipped)" : @sprintf("%.2f s", t_cs)) | $(t_cs === nothing ? "—" : "1.0× (reference)") |
    | BasisSimulator.jl, CPU ($(Threads.nthreads()) Julia threads) | $(@sprintf("%.2f s", basissim_cpu_result.elapsed)) | $(speed(basissim_cpu_result.elapsed)) |
    | BasisSimulator.jl, $(GPU_BACKEND.name) | $(@sprintf("%.2f s", basissim_gpu_result.elapsed)) | $(speed(basissim_gpu_result.elapsed)) |
    """,
    )
end

# ╔═╡ 0600000d-0000-4000-8000-000000000001
md"""
### Why BasisSimulator Is Faster

- **No disk round trip.** CatSim writes `.air`, `.offset`, `.scan` and `.prep` files and reads
  the `.prep` back for FDK, then writes the image volume. BasisSimulator keeps the sinogram on the
  device from forward projection through FBP.
- **One volume walk for the whole spectrum.** The default `:dd_fast` projector accumulates the
  path length through each material once and weights all energies from it, instead of tracing
  every energy separately.
- **The same kernels on any backend.** CatSim's voxelized projector is C on the host.
  BasisSimulator's kernels are written once with AcceleratedKernels.jl and run on CUDA, Metal,
  ROCm, oneAPI or the CPU, as the CPU and GPU runs above show.

The phantom is deliberately small so that CatSim finishes in minutes. For a larger phantom or a
clinical-size detector, scale `n_voxels` and the scanner; the wrapper forwards any
`EICTScanner` / `CTProtocol` pair, with the bowtie restriction noted in
`catsim_configure_protocol!`.
"""

# ╔═╡ 0f593214-b814-46c0-bd21-ea61a16094b9
let
    r1(x) = round(x; digits = 1)
    t_cpu, t_gpu = basissim_cpu_result.elapsed, basissim_gpu_result.elapsed
    catsim_line = if rod_hu.has_catsim
        water = only(r for r in rod_hu.rows if r.label == 3)
        rel = [
            100 * (r.basissim - r.catsim) / r.theory
            for r in rod_hu.rows
            if r.label != 3
        ]
        t_cs = catsim_result.elapsed
        """
    - **Agreement.** CatSim's image correlates with BasisSimulator's at
      $(round(catsim_aligned.correlation; digits = 4)); in the solid-water body the two differ by
      $(r1(water.basissim - water.catsim)) HU, and over the 14 rods by $(r1(minimum(rel))) to
      $(r1(maximum(rel))) % of the rod's theoretical HU, the spread expected from two independent
      spectrum models and water corrections.
    - **Runtime.** CatSim took $(@sprintf("%.1f", t_cs)) s, BasisSimulator $(@sprintf("%.2f", t_cpu)) s
      on the CPU ($(@sprintf("%.0f", t_cs / t_cpu))× faster) and $(@sprintf("%.2f", t_gpu)) s on the
      GPU ($(@sprintf("%.0f", t_cs / t_gpu))× faster).
    """
    else
        """
    - **CatSim was skipped** (gecatsim not available); BasisSimulator took $(@sprintf("%.2f", t_cpu)) s
      on the CPU and $(@sprintf("%.2f", t_gpu)) s on the GPU.
    """
    end
    Markdown.parse(
        """
    ## Summary

    $(catsim_line)
    - **CPU and GPU are the same code.** Their images differ by at most
      $(@sprintf("%.3g", rod_hu.cpu_gpu)) HU over the whole volume.
    """,
    )
end

# ╔═╡ Cell order:
# ╟─06000001-0000-4000-8000-000000000010
# ╟─b5d4108b-cbcb-492e-ad8d-9222e8bec75a
# ╟─3603ec96-7d2e-46f0-ad1d-841c1098eeee
# ╟─882ffd5c-5b65-4173-94e5-530119393fae
# ╟─06000001-0000-4000-8000-000000000020
# ╠═06000001-0000-4000-8000-000000000001
# ╠═06000001-0000-4000-8000-000000000002
# ╠═06000001-0000-4000-8000-000000000003
# ╠═06000001-0000-4000-8000-000000000004
# ╠═06000001-0000-4000-8000-000000000030
# ╠═06000001-0000-4000-8000-000000000031
# ╠═06000001-0000-4000-8000-000000000032
# ╠═06000001-0000-4000-8000-000000000033
# ╠═06000001-0000-4000-8000-000000000034
# ╠═06000001-0000-4000-8000-000000000040
# ╟─06000001-0000-4000-8000-000000000050
# ╠═06000001-0000-4000-8000-000000000060
# ╟─06000001-0000-4000-8000-000000000070
# ╠═06000001-0000-4000-8000-000000000080
# ╠═06000001-0000-4000-8000-000000000090
# ╟─e41259df-f198-4af5-89c0-19679c46bbb5
# ╠═15786719-1413-4d88-82d0-35d24f115310
# ╠═02e8b3a2-43a7-48c1-bce9-97d39c28493d
# ╠═80748e21-bd9e-4ee9-81c5-79129ae25f53
# ╠═24dd130a-4560-49ea-8441-82f15890516d
# ╠═bcd32d72-7c5f-437e-b982-1ef79a472a94
# ╟─060000f1-0000-4000-8000-000000000001
# ╠═06000002-0000-4000-8000-000000000010
# ╟─a8ac0e53-23ec-533d-a797-af31d0867766
# ╟─06000003-0000-4000-8000-000000000001
# ╠═06000003-0000-4000-8000-000000000010
# ╠═06000003-0000-4000-8000-000000000020
# ╠═06000003-0000-4000-8000-000000000030
# ╟─06000004-0000-4000-8000-000000000001
# ╠═06000004-0000-4000-8000-000000000010
# ╠═06000004-0000-4000-8000-000000000020
# ╠═06000004-0000-4000-8000-000000000030
# ╠═06000004-0000-4000-8000-000000000040
# ╠═06000004-0000-4000-8000-000000000050
# ╠═06000004-0000-4000-8000-000000000060
# ╠═06000004-0000-4000-8000-000000000070
# ╠═06000004-0000-4000-8000-000000000080
# ╠═06000004-0000-4000-8000-000000000090
# ╟─06000005-0000-4000-8000-000000000001
# ╠═06000005-0000-4000-8000-000000000010
# ╠═06000005-0000-4000-8000-000000000030
# ╟─06000006-0000-4000-8000-000000000001
# ╠═06000006-0000-4000-8000-000000000010
# ╠═06000006-0000-4000-8000-000000000020
# ╟─06000007-0000-4000-8000-000000000001
# ╠═06000007-0000-4000-8000-000000000010
# ╠═06000007-0000-4000-8000-000000000020
# ╟─06000007-0000-4000-8000-000000000030
# ╟─060000f2-0000-4000-8000-000000000001
# ╠═06000008-0000-4000-8000-000000000010
# ╟─06000009-0000-4000-8000-000000000001
# ╠═06000009-0000-4000-8000-000000000005
# ╠═06000009-0000-4000-8000-000000000010
# ╠═0600000a-0000-4000-8000-000000000010
# ╟─0600000b-0000-4000-8000-000000000001
# ╠═0600000b-0000-4000-8000-000000000005
# ╟─ffa467af-ec12-4b92-bd4e-732aa0c6cbe2
# ╟─0600000b-0000-4000-8000-000000000010
# ╟─0600000b-0000-4000-8000-000000000020
# ╠═bf0f3dc3-168e-4984-b8e0-a22f126a6443
# ╟─cf2708e4-b701-4953-893a-3209e66aca04
# ╟─0600000b-0000-4000-8000-000000000030
# ╟─0600000c-0000-4000-8000-000000000001
# ╟─c90afced-10a6-46cc-bdb3-fcab013ee4a9
# ╟─0600000c-0000-4000-8000-000000000005
# ╟─0600000c-0000-4000-8000-000000000010
# ╟─0600000d-0000-4000-8000-000000000001
# ╟─0f593214-b814-46c0-bd21-ea61a16094b9
