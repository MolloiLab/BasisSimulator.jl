### A Pluto.jl notebook ###
# v0.20.4

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


# ╔═╡ 80569823-8008-530d-a0df-34c4e21e0758
md"""
# The Canon Aquilion ONE Detector in Geant4: GENESIS and PRISM

The Canon (formerly Toshiba) Aquilion ONE is an energy-integrating CT scanner with a 320-row,
16 cm wide detector, wide enough to cover an organ in one rotation. Two current editions are modelled
here, the **Aquilion ONE GENESIS** (TSX-305A) and the **Aquilion ONE PRISM** (TSX-306A). They share
the detector, Canon's **PUREViSION**, and the tube: the PRISM's FDA clearance lists "no change in the
hardware" of the detector. A PUREViSION dexel (detector element) is a block of praseodymium-activated
ceramic with a reflector around it and a photodiode under it.

BasisSimulator, up to and including **v0.15.1**, has no Canon scanner: no geometry, no beam and no
detector table. This notebook builds the detector in Geant4, one piece at a time, and

1. **gathers what is documented** (the FDA clearances, Canon's brochures, the literature) and
   states every assumption where nothing is;
2. **fits the x-ray beam** to measured half-value layers, and checks it on Canon's silver filter;
3. **builds the detector's efficiency table**, with its noise (the Swank factor), and saves it for
   BasisSimulator, once for each edition.

!!! note "Which BasisSimulator this notebook compares against"
    BasisSimulator **v0.15.1**, the last release before these Geant4 tables, has no Canon detector;
    the nearest table it has is its SOMATOM Definition Flash table, the one a Canon model would have
    had to borrow, and that is what section 5 compares against. BasisMonteCarlo keeps a verbatim copy
    of it (`BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash`), so the comparison stays the same whatever
    BasisSimulator is installed. Only the x-ray spectra and the bowtie (sections 8–10) come from the
    **installed** BasisSimulator; nothing here pins BasisSimulator, or BasisMonteCarlo, to a version.
"""

# ╔═╡ 5668de3a-22de-42be-8847-10f72fb78120
let
    Mke.set_theme!(THEME)
    night, ink, sub = MATERIALS.night, Mke.RGBf(0.93, 0.94, 0.96), Mke.RGBf(0.6, 0.63, 0.68)
    fig = Mke.Figure(size = (1800, 600), backgroundcolor = night, figure_padding = (10, 10, 10, 10))
    R_iso, R_det = aquilion.source_to_isocenter_mm, aquilion.source_to_detector_mm
    half_fan = asin(aquilion.scan_field_mm / 2 / R_iso)
    θ = range(0, 2π; length = 200)

    # (1) The gantry, to scale: one tube and its detector arc
    ax1 = canvas(fig[1, 1]; background = night)
    Mke.poly!(ax1, Mke.Point2f.(820 .* cos.(θ), 820 .* sin.(θ)); color = MATERIALS.night_ring)
    Mke.poly!(ax1, Mke.Point2f.(390 .* cos.(θ), 390 .* sin.(θ)); color = night)
    Mke.lines!(ax1, 640 .* cos.(θ), 640 .* sin.(θ); color = (:white, 0.06), linewidth = 1)
    Mke.poly!(ax1, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = Mke.RGBf(0.42, 0.45, 0.5))
    for x in (-75, 75)
        Mke.poly!(ax1, Mke.Point2f.(x .+ 50 .* cos.(θ), 10 .+ 60 .* sin.(θ)); color = Mke.RGBf(0.16, 0.18, 0.21))
    end
    Mke.poly!(ax1, Mke.Point2f.(22 .* cos.(θ), -105 .+ 20 .* sin.(θ)); color = Mke.RGBf(0.85, 0.85, 0.82))
    src = Mke.Point2f(0, R_iso)
    φ = range(-half_fan, half_fan; length = 80)
    arc = [src + R_det * Mke.Point2f(sin(t), -cos(t)) for t in φ]
    Mke.poly!(ax1, Mke.Point2f.(vcat([src], arc)); color = (MATERIALS.beam, 0.22))
    inner = [src + (R_det + 8) * Mke.Point2f(sin(t), -cos(t)) for t in φ]
    outer = [src + (R_det + 50) * Mke.Point2f(sin(t), -cos(t)) for t in φ]
    Mke.poly!(ax1, Mke.Point2f.(vcat(inner, reverse(outer))); color = Mke.RGBf(0.78, 0.82, 0.88))
    Mke.poly!(ax1, Mke.Rect2f(src[1] - 45, src[2] - 45, 90, 90); color = Mke.RGBf(0.8, 0.83, 0.88))
    Mke.scatter!(ax1, [src]; color = MATERIALS.beam, markersize = 12)
    Mke.text!(ax1, src + Mke.Point2f(0, 85); text = "Tube", color = ink, fontsize = 17, font = :bold, align = (:center, :center))
    Mke.limits!(ax1, -1010, 1010, -1010, 1010)

    # (2) Side view, to scale: the cone covers 16 cm at the isocentre, a whole heart in one rotation
    ax2 = canvas(fig[1, 2]; background = night)
    half_z = aquilion.rows * aquilion.row_at_isocenter_mm / 2
    detector_y = R_iso - R_det
    detector_half = half_z * R_det / R_iso
    Mke.poly!(ax2, Mke.Rect2f(-330, -150, 660, 290); color = Mke.RGBf(0.42, 0.45, 0.5))
    Mke.poly!(ax2, Mke.Point2f.(60 .* cos.(θ), 10 .+ 50 .* sin.(θ)); color = Mke.RGBf(0.75, 0.3, 0.32))
    Mke.poly!(ax2, Mke.Point2f[(0, R_iso), (detector_half, detector_y), (-detector_half, detector_y)]; color = (MATERIALS.beam, 0.25))
    Mke.poly!(ax2, Mke.Rect2f(-detector_half - 10, detector_y - 45, 2detector_half + 20, 40); color = Mke.RGBf(0.78, 0.82, 0.88))
    Mke.poly!(ax2, Mke.Rect2f(-45, R_iso - 45, 90, 90); color = Mke.RGBf(0.8, 0.83, 0.88))
    Mke.scatter!(ax2, [Mke.Point2f(0, R_iso)]; color = MATERIALS.beam, markersize = 12)
    Mke.lines!(ax2, Mke.Point2f[(-half_z, 0), (half_z, 0)]; color = ink, linewidth = 2.5)
    for z in (-half_z, half_z)
        Mke.lines!(ax2, Mke.Point2f[(z, -18), (z, 18)]; color = ink, linewidth = 2.5)
    end
    Mke.text!(ax2, 0, -30; text = "$(round(Int, 2half_z / 10)) cm", color = ink, fontsize = 18, font = :bold, align = (:center, :top))
    Mke.text!(ax2, 0, R_iso + 85; text = "Tube", color = ink, fontsize = 17, font = :bold, align = (:center, :center))
    Mke.limits!(ax2, -700, 700, detector_y - 120, R_iso + 160)

    # (3) A patch of the detector under its plates, to scale, the near corner cut away
    ax3 = canvas(fig[1, 3]; background = night)
    p, s, d = pitch_mm, septum_mm, depth_mm
    nx, ny = 7, 5
    cam = camera(-52, 28)
    blocks = dexel_blocks(nx, ny, p, s, 0.0, d)
    plates = filter(grid_blocks(nx, ny, p, grid.wall_mm, d, d + grid.height_mm; walls = :x)) do b
        !(b.lo[1] > 2.4p[1] && b.lo[2] < 2.0p[2])
    end
    append!(blocks, plates)
    rays = map([(3.5, 0.5), (5.5, 1.5), (4.5, 0.5)]) do (i, j)
        c = (i * p[1], j * p[2])
        a, b = project(cam, (c..., d + 12.5)), project(cam, (c..., d + 0.15))
        (dot3((c..., d), cam.toward) + 1, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 12))
    end
    draw_blocks!(ax3, cam, blocks; extras = rays, edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    # (4) One dexel's stack, taken apart
    ax4 = canvas(fig[1, 4]; background = night)
    draw_blocks!(ax4, camera(-56, 30), exploded_detector(1, 1; gap = 0.75).blocks; edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    for (k, (title, subtitle)) in enumerate(
            (
                ("One Tube, One Detector", "The Aquilion ONE, to Scale"),
                ("16 cm in One Rotation", "320 Rows of 0.5 mm: a Whole Heart"),
                ("Dexels Under a 1-D Grid", "Ceramic, White Reflector, Tungsten Plates"),
                ("One Dexel, Taken Apart", "Grid, Ceramic, Photodiode, Electronics"),
            )
        )
        Mke.Label(fig[2, k], title; color = ink, fontsize = 26, font = :bold, tellwidth = false)
        Mke.Label(fig[3, k], subtitle; color = sub, fontsize = 18, tellwidth = false)
    end
    Mke.rowgap!(fig.layout, 1, 8)
    Mke.rowgap!(fig.layout, 2, 4)
    fig
end

# ╔═╡ 0e0805a1-0452-5360-8e0e-27c7450b5769
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a PUREViSION dexel does not count at all.
Every x ray it absorbs leaves energy in the ceramic, which glows; the reflector keeps the light
inside the dexel, and the photodiode under it adds up the light of every photon over the reading.
The signal is therefore proportional to the **energy** left in the dexel, and the efficiency that
matters is the energy-absorption efficiency: the mean energy a dexel collects per photon, divided
by the photon's energy.
"""

# ╔═╡ 2c81d3e6-baec-47b5-a67e-63e27c1e674e
let
    Mke.set_theme!(THEME)
    p = pitch_mm
    nx, ny = 4, 3
    cam = camera(-56, 30)
    fig = Mke.Figure(size = (1400, 920))
    ax = canvas(fig[1, 1])
    ex = exploded_detector(nx, ny)
    blocks, levels = ex.blocks, ex.levels

    # X rays from the focal spot, far above, each down the middle of a grid channel
    top = levels[:grid][2]
    rays = map([(0, 0), (1, 0), (2, 0), (3, 0)]) do (i, j)
        c = ((i + 0.5) * p[1], (j + 0.5) * p[2])
        a, b = project(cam, (c..., top + 3.0)), project(cam, (c..., top + 0.35))
        (dot3((c..., top), cam.toward) + 50, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3.5, head = 15))
    end
    draw_blocks!(ax, cam, blocks; extras = rays)

    # One dexel: the front-right column, the same cell in every layer
    i, j = nx - 1, 0
    for (key, (z0, z1)) in levels
        key === :electronics && continue
        outline!(ax, cam, (i * p[1], j * p[2], z0), ((i + 1) * p[1], (j + 1) * p[2], z1); linewidth = 3.5)
    end
    Mke.Label(
        fig[2, 1],
        "One Dexel, Outlined in Amber: One Block of Ceramic and Its Photodiode, Between Two Grid Plates";
        font = :bold,
        fontsize = 17,
        color = Mke.RGBf(0.8, 0.5, 0.05),
        halign = :left,
        tellwidth = false,
    )

    rows = (
        (:grid, "Anti-Scatter Grid (Assumed)", "1-D Tungsten Plates Between the Channels,\nNone Along Z; 10 mm Tall, Drawn Short Here", true),
        (:ceramic, "PUREViSION Ceramic, Sawn into Dexels", "\"Praseodymium-Activated\": Modelled as Gd₂O₂S,\n1.4 mm Deep (Assumed); White Reflector Between", true),
        (:photodiode, "Photodiode, One per Dexel", "Silicon: Light → Charge (Canon Publishes\nNo Details; Drawn Generic)", false),
        (:electronics, "Readout Electronics", "Digitize Every Dexel's Charge and Send It\nto the Gantry (Not Documented; Drawn Generic)", false),
    )
    x_text = maximum(project(cam, (nx * p[1], y, 0))[1] for y in (0, ny * p[2])) + 2.4
    entries = [
        (
                anchor = project(cam, (nx * p[1], ny * p[2] * 0.62, sum(levels[key]) / 2)),
                title = name,
                detail,
                color = PALETTE.ink,
                tag = simulated ? ("In Geant4", TAGS.geant4) : ("Not Simulated", TAGS.not_simulated),
            )
            for (key, name, detail, simulated) in rows
    ]
    labels_right!(ax, entries, x_text; spacing = 1.45, tag_width = 122)
    Mke.xlims!(ax, nothing, x_text + 8.6)
    fig
end

# ╔═╡ cb572a6e-a206-4ee0-a7d8-a07bb0f13cde
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1400, 760))
    ax = canvas(fig[1, 1])

    # Layers, side view (x across, y up); heights in mm, the grid cut short
    y_board, y_pd = -1.55, -0.16
    rect!(ax, -1.5p, y_board, 3p, 0.45, MATERIALS.board)
    rect!(ax, -1.5p, -0.3, 3p, 0.14, MATERIALS.cmos)  # the photodiode array's silicon
    for i in -1:1
        x = i * p
        rect!(ax, x - (p - s) / 2, y_pd, p - s, 0.14, MATERIALS.photodiode)
        rect!(ax, x - 0.04, y_board + 0.45, 0.08, -0.3 - (y_board + 0.45), MATERIALS.gold)  # its connector
    end
    rect!(ax, -0.55, y_board + 0.08, 1.1, 0.29, Mke.RGBf(0.32, 0.35, 0.42))  # a readout chip
    Mke.text!(ax, 0.0, y_board + 0.225; text = "Readout", color = :white, fontsize = 13, align = (:center, :center), font = :bold)
    # ceramic and reflector
    rect!(ax, -1.5p, 0, 3p, d, MATERIALS.reflector; strokecolor = (:black, 0.25), strokewidth = 1)
    for i in -1:1
        rect!(ax, i * p - (p - s) / 2, 0, p - s, d, i == 0 ? Mke.RGBf(0.8, 0.95, 0.82) : MATERIALS.ufc)
    end
    # grid walls, cut short (wavy break)
    for i in -1:2
        x = (i - 0.5) * p
        rect!(ax, x - grid.wall_mm / 2, d, grid.wall_mm, 1.2, MATERIALS.tungsten)
    end

    # (1) the x ray, (2) stops in the middle dexel, (3) light, (4) photodiode, (5) the electronics
    hit = Mke.Point2f(0.12, 0.62)
    photon!(ax, (0.12, d + 2.6), hit; cycles = 9, amplitude = 0.02, color = MATERIALS.beam, linewidth = 3.5)
    Mke.scatter!(ax, [hit]; marker = :star5, markersize = 30, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.5)
    # light rays: straight down, and bouncing off the reflector walls
    half = (p - s) / 2
    for (dx, bounce) in ((0.0, false), (-1.0, true), (1.0, true), (-0.35, false), (0.4, false))
        if bounce
            wall = Mke.Point2f(dx < 0 ? -half : half, hit[2] + 0.28)
            bottom = Mke.Point2f(dx < 0 ? -0.15 : 0.32, 0.0)
            Mke.lines!(ax, Mke.Point2f[hit, wall, bottom]; color = (MATERIALS.light, 0.95), linewidth = 2.2)
        else
            Mke.lines!(ax, Mke.Point2f[hit, Mke.Point2f(hit[1] + dx * 0.5, 0.0)]; color = (MATERIALS.light, 0.95), linewidth = 2.2)
        end
    end
    # the number out
    arrow!(ax, (0.0, y_board + 0.08), (0.0, y_board - 0.35); color = PALETTE.ink, linewidth = 2, head = 12)
    Mke.text!(ax, 0.0, y_board - 0.45; text = "One Number per Dexel per Reading", align = (:center, :top), fontsize = 16, color = PALETTE.ink)

    # the five steps, labelled on the right
    entries = [
        (anchor = (0.12, d + 1.9), title = "1  An X Ray Arrives", detail = "Between Two Plates of the Grid", color = Mke.RGBf(0.85, 0.5, 0.05), tag = nothing),
        (anchor = hit, title = "2  The Ceramic Stops It", detail = "Its Energy Is Left in the Dexel", color = Mke.RGBf(0.7, 0.55, 0.05), tag = nothing),
        (anchor = (half - 0.02, hit[2] + 0.28), title = "3  The Ceramic Glows", detail = "Light ∝ Energy Left; the White Reflector Keeps It In", color = Mke.RGBf(0.2, 0.6, 0.25), tag = nothing),
        (anchor = (0.4, y_pd + 0.07), title = "4  The Photodiode Collects the Light", detail = "Light → Charge, Summed over the Reading", color = MATERIALS.photodiode, tag = nothing),
        (anchor = (0.4, y_board + 0.225), title = "5  The Electronics Digitize the Charge", detail = "On a Readout Board (Not Documented)", color = PALETTE.ink, tag = nothing),
    ]
    xr = 1.5p + 0.75
    labels_right!(ax, entries, xr; spacing = 0.62)
    Mke.text!(ax, 1.5p + 0.05, d + 1.2; text = "Grid Plates\n(Cut Short)", fontsize = 14, color = PALETTE.muted, align = (:left, :top))

    # what Geant4 simulates, and what is assumed
    xl = -1.5p - 0.15
    side_bracket!(ax, xl, 0.0, d + 1.2, "Geant4\nSimulates"; color = TAGS.geant4)
    side_bracket!(ax, xl, y_board, -0.04, "Not Simulated:\nSignal ∝ Energy\nLeft in the Ceramic"; color = TAGS.not_simulated)
    Mke.limits!(ax, -1.5p - 1.75, xr + 3.3, y_board - 0.75, d + 2.75)
    fig
end

# ╔═╡ 5e84a0e1-897c-4189-9364-bd06e6401f17
md"""
### Which Beam, and Why

How the photons arrive decides what a simulation measures, so each section says which beam it
uses. Five are used here, from the simplest to the most realistic:

| beam | what it measures | where |
|---|---|---|
| **pencil beam** into the centre of one dexel | how BasisSimulator v0.15.1's nearest table, the Definition Flash's, was made (notebook 05 reproduces it) | section 5 |
| **uniform on one dexel's ceramic** | the efficiency per photon landing on the ceramic; reading every dexel around it gives the flat field (below) | sections 3–5 |
| **uniform on one cell** (ceramic and its reflector), under the grid's plates | what the grid costs, straight and at an angle | section 6 |
| **flat field**: every dexel lit | what each dexel collects in a scan: what it loses to its neighbours it gets back from them. By symmetry this equals all the energy a uniform beam on one dexel leaves anywhere, which is how it is simulated | sections 4, 5 |
| **from the focal spot**, over the whole arc, with **clinical spectra** | everything at once: real angles, every dexel, the grid, the Aquilion ONE's fitted 120 kV beam (also after 20 and 35 cm of water), over its whole 16 cm | sections 7, 10 |

The efficiency table is built from single energies, because BasisSimulator weights it by whatever
spectrum a scan uses. Section 10 checks that this shortcut is exact: the clinical beams, simulated
photon by photon, give the same signal and noise as the table weighted by their spectra.
"""

# ╔═╡ e3e7a4a7-1dbd-443a-b51d-db0893932643
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1700, 560))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        ("Pencil Beam", "Into the Centre of One Dexel:\nHow the Flash Table Was Made", :pencil),
        ("Uniform on One Dexel", "Over Its Ceramic; Every Dexel\nRead Out (Sections 3–5)", :ceramic),
        ("Uniform on One Cell", "Ceramic and Its Reflector,\nUnder the Grid (Section 6)", :cell),
        ("Flat Field", "Every Dexel Lit, as in a Scan:\nWhat Each Dexel Collects", :flat),
        ("From the Focal Spot", "The Whole Arc, Real Angles;\nClinical Spectra (Sections 7, 10)", :focal),
    )
    for (k, (title, detail, kind)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        blocks = dexel_blocks(3, 3, p, s, 0.0, d)
        if kind === :cell
            append!(blocks, grid_blocks(3, 3, p, grid.wall_mm, d, d + 1.6; walls = :x))
        end
        top = kind === :cell ? d + 1.6 : d
        extras = Any[]
        # the beam: amber columns (translucent boxes) or lines, drawn after the blocks
        if kind === :pencil
            a, b = project(cam, (1.5p[1], 1.5p[2], d + 4.5)), project(cam, (1.5p[1], 1.5p[2], d))
            push!(extras, (1.0e3, ax -> arrow!(ax, a, b; color = amber, linewidth = 4, head = 14)))
        elseif kind in (:ceramic, :cell, :flat)
            lo = kind === :ceramic ? (p[1] + s / 2, p[2] + s / 2) : kind === :cell ? (p[1], p[2]) : (0.0, 0.0)
            hi = kind === :ceramic ? (2p[1] - s / 2, 2p[2] - s / 2) : kind === :cell ? (2p[1], 2p[2]) : (3p[1], 3p[2])
            beam = Block((lo..., top), (hi..., top + 4.0), amber; alpha = 0.28, stroke = 0.0)
            push!(extras, (1.0e3, ax -> draw_blocks!(ax, cam, [beam]; edge_alpha = 0.0)))
        else
            focal = (1.5p[1], 1.5p[2], d + 9.0)
            for (x, y) in ((0.2, 0.2), (2.8, 0.2), (0.2, 2.8), (2.8, 2.8), (1.5, 1.5), (0.9, 2.1), (2.2, 0.8))
                from, to = project(cam, focal), project(cam, (x * p[1], y * p[2], d))
                push!(extras, (1.0e3, ax -> arrow!(ax, from, to; color = amber, linewidth = 2.2, head = 10)))
            end
            push!(extras, (1.0e3 + 1, ax -> Mke.scatter!(ax, [project(cam, focal)]; color = PALETTE.ink, markersize = 12)))
        end
        draw_blocks!(ax, cam, blocks; extras)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    fig
end

# ╔═╡ db0ace6b-e7f9-56bf-850a-db1a6d6e2e06
md"""
## The Aquilion ONE, as Documented

Every hardware number this notebook uses, from the FDA 510(k) summaries, Canon's brochures, Toshiba's
patents and the literature. Where no source gives a number the row says so, and the notebook states
its assumption. BasisSimulator (up to v0.15.1) has no Canon values to compare against. Canon publishes even less about
its detector than Philips does.

| quantity | documented | status | source |
|---|---|---|---|
| detector | PUREViSION (since 2014), the same on GENESIS and PRISM ("no change in the hardware") | ✓ | K133497; K192188 |
| scintillator | "praseodymium-activated" ceramic, cut from a solid ingot, with reflective isolation; host material not stated | assumed here: Gd₂O₂S at 7.31 g/cm³ | Canon PUREViSION brochure |
| crystal depth | not published | assumed here: 1.4 mm; 0.8–2.0 mm saved | – |
| dexels | 320 rows × 0.5 mm (16 cm at the isocentre); 896 channels per row | ✓ | Canon brochures; K160587 |
| source–isocentre | 600 mm (original Aquilion ONE and ONE ViSION) | used; not confirmed for GENESIS/PRISM | Geleijns et al. 2009 |
| source–detector | 1072 mm, from a secondary table only | used; not confirmed | – |
| dexel pitch at the detector | not published | calc: 0.5 mm × 1072/600 along z; across the fan, 896 channels over the 50 cm field | – |
| reflector between dexels | "reflective isolation", width not published | assumed: 0.1 mm of TiO₂-loaded epoxy | Canon PUREViSION brochure |
| anti-scatter grid | not published for any edition; Toshiba's patents describe 1-D and 2-D tungsten or molybdenum collimators | assumed here: 1-D tungsten plates, 0.1 mm, 10 mm tall | US 7,526,070; US 9,320,476 |
| tube | CXB-750F on both; 80, 100, 120, 135 kV | ✓ | K160587; K192188 |
| anode angle | 10° (ONE ViSION); not stated for GENESIS/PRISM | 10° used | K122109 |
| filtration | not published | fitted in section 8 | – |
| beam quality (PRISM) | half-value layer 6.0, 7.4, 8.5, 9.2 mm Al at 80, 100, 120, 135 kV; 13.7 mm Al with SilverBeam (0.5 mm silver) at 120 kV | the target of the beam fit, and its check | Ito et al. 2026 |
| bowtie filters | GENESIS: two, unnamed; PRISM: M and L, plus SilverBeam | – | K160587; K213504 |

Sources: FDA [K133497](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133497.pdf),
[K160587](https://www.accessdata.fda.gov/cdrh_docs/pdf16/K160587.pdf),
[K192188](https://www.accessdata.fda.gov/cdrh_docs/pdf19/K192188.pdf),
[K213504](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213504.pdf),
[K122109](https://www.accessdata.fda.gov/cdrh_docs/pdf12/K122109.pdf);
Canon, [GENESIS brochure](https://us.medical.canon/download/ct-br-aq-one-genesis),
[PRISM brochure](https://us.medical.canon/download/ct-br-aquilion-one-prism),
[PUREViSION detector](https://us.medical.canon/products/computed-tomography/pure-vision-detector);
Toshiba patents [US 7,526,070](https://patents.google.com/patent/US7526070B2/en),
[US 9,320,476](https://patents.google.com/patent/US9320476B2/en);
Geleijns et al., [Phys. Med. Biol. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2948862/);
Ito et al., [J. Appl. Clin. Med. Phys. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13486965/);
Siemens, [Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ 33e46fd5-c542-55e8-bcb5-192a0bac3d5e
md"""
## Notebook Setup
"""

# ╔═╡ ab9db025-f507-5388-bf67-4c0d170d0e28
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ bc041287-f4f5-5ce7-8924-9633ea5dc417
using Markdown: @md_str, Markdown

# ╔═╡ 808a4723-438f-586f-b92c-6e49eb9f89e6
import PlutoUI

# ╔═╡ 8b53812d-c201-51bb-8847-1b66b5f1f09c
import BasisMonteCarlo as MC

# ╔═╡ 429139dd-55bf-52bd-a9d5-9691d14d06a2
import CairoMakie as Mke

# ╔═╡ 1a29df66-ff12-5ff0-acdd-9c0e6b2b13d6
import BasisSimulator as BS

# ╔═╡ 726ef0af-666e-5b87-9ea4-02801135cab1
import Serialization

# ╔═╡ 3accdd63-8b76-5d41-a238-01a65ba33416
import DelimitedFiles

# ╔═╡ 10fb07d1-38ee-5894-b238-df1f1e70be84
using Statistics: mean

# ╔═╡ 558bad13-644e-5e3d-be50-5c5850d10928
PlutoUI.TableOfContents()

# ╔═╡ ed268a94-6b31-5c94-87d3-d11efbaeb1a0
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ c9b96524-0916-5e0a-bd06-366cce225226
# One colour per kind of particle, used by every figure
PALETTE = (
    photon = Mke.RGBf(0.13, 0.36, 0.66),  # the incident photon
    xray = Mke.RGBf(0.9, 0.45, 0.05),  # characteristic x rays
    electron = Mke.RGBf(0.7, 0.12, 0.17),  # electrons, and the light they make
    kept = Mke.RGBf(0.27, 0.55, 0.34),  # the signal
    slab = Mke.RGBAf(0.45, 0.52, 0.62, 0.3),
    slab_edge = Mke.RGBf(0.3, 0.36, 0.44),
    reflector = Mke.RGBf(0.85, 0.83, 0.7),
    ceramic = Mke.RGBf(0.9, 0.92, 0.95),
    ceramic_hit = Mke.RGBf(0.78, 0.85, 0.95),
    tungsten = Mke.RGBf(0.25, 0.25, 0.3),
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator v0.15.1's Flash table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ 57dd0152-ffba-53a8-a5fc-33d79b6ce3ac
# Large, uncluttered type for every figure
THEME = Mke.Theme(
    fontsize = 18,
    figure_padding = (12, 24, 12, 12),
    Axis = (
        titlesize = 21,
        titlealign = :left,
        titlegap = 14,
        xlabelsize = 19,
        ylabelsize = 19,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xgridcolor = (:black, 0.07),
        ygridcolor = (:black, 0.07),
        topspinevisible = false,
        rightspinevisible = false,
        xlabelpadding = 8,
        ylabelpadding = 12,
    ),
    Axis3 = (
        titlesize = 21,
        titlealign = :left,
        xlabelsize = 18,
        ylabelsize = 18,
        zlabelsize = 18,
        xticklabelsize = 15,
        yticklabelsize = 15,
        zticklabelsize = 15,
    ),
    Legend = (
        labelsize = 17,
        framevisible = false,
        patchsize = (30, 14),
        rowgap = 8,
    ),
    Lines = (
        linewidth = 2.5,
    ),
)

# ╔═╡ 1b1c6cc3-ced3-5a5c-affb-f6456a4012f2
"""
An in-figure annotation on a translucent rounded background, so it never fights the data.
"""
function note!(
        ax,
        x,
        y,
        text;
        align = (:left, :center),
        color = PALETTE.ink,
        fontsize = 16,
        offset = (0, 0),
        alpha = 0.85,
    )
    return Mke.textlabel!(
        ax,
        Mke.Point2f(x, y);
        text,
        text_align = align,
        text_color = color,
        fontsize,
        offset,
        background_color = (:white, alpha),
        strokewidth = 0,
        padding = (7, 7, 4, 4),
        cornerradius = 5,
    )
end

# ╔═╡ bfe11ba8-730e-58f7-9f66-212ffb916313
"""Probability per keV per photon of `values` falling in each bin of width `bin_keV` up to `top_keV`."""
function histogram(values, top_keV; bin_keV = 0.5)
    edges = 0:bin_keV:top_keV
    counts = zeros(length(edges) - 1)
    for v in values
        v > 0 || continue
        k = clamp(floor(Int, v / bin_keV) + 1, 1, length(counts))
        counts[k] += 1
    end
    return (
        centres = edges[1:(end - 1)] .+ bin_keV / 2,
        density = counts ./ (length(values) * bin_keV),
    )
end

# ╔═╡ d249bab7-2437-44ed-8739-4479d42c8913
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses (the pitch, the reflector,
the crystal's depth, the grid), with a small kit: blocks in 3-D, seen from any direction, shaded
by one light and drawn back to front. The colours are the materials': pale blue ceramic, white
TiO₂ reflector, grey tungsten, dark-blue silicon photodiodes, amber x rays.
"""

# ╔═╡ a5eec273-7b42-4e5e-8716-1ba9b2ba4bc5
MATERIALS = (
    ufc = Mke.RGBf(0.74, 0.84, 0.93),  # the scintillating ceramic
    ufc_lit = Mke.RGBf(0.55, 0.9, 0.62),  # ceramic glowing
    reflector = Mke.RGBf(0.97, 0.97, 0.95),  # TiO₂ epoxy: white
    tungsten = Mke.RGBf(0.42, 0.44, 0.5),
    photodiode = Mke.RGBf(0.24, 0.27, 0.55),  # back-illuminated silicon
    cmos = Mke.RGBf(0.2, 0.22, 0.27),  # the CMOS wafer with the ADC
    gold = Mke.RGBf(0.86, 0.66, 0.2),  # vias and stud bumps
    substrate = Mke.RGBf(0.86, 0.83, 0.76),  # ceramics substrate
    board = Mke.RGBf(0.13, 0.4, 0.29),  # circuit board
    water = Mke.RGBf(0.62, 0.78, 0.92),
    beam = Mke.RGBf(0.98, 0.7, 0.25),  # x rays, as in the lab's banner
    light = Mke.RGBf(0.45, 0.9, 0.45),  # the ceramic's light (Gd₂O₂S:Pr glows green)
    night = Mke.RGBf(0.05, 0.06, 0.08),  # the dark backdrop of the lab's banner
    night_ring = Mke.RGBf(0.11, 0.13, 0.16),
)

# ╔═╡ 369dc801-1932-4c4c-8208-d6f3231f5c25
begin
    """A block in 3-D (axis-aligned, `lo` to `hi` in mm), and how to draw it."""
    Base.@kwdef struct Block
        lo::NTuple{3, Float64}
        hi::NTuple{3, Float64}
        color::Mke.RGBf
        alpha::Float64 = 1.0
        stroke::Float64 = 0.6
    end

    Block(lo, hi, color; kw...) = Block(; lo = Float64.(Tuple(lo)), hi = Float64.(Tuple(hi)), color, kw...)
end

# ╔═╡ 4ac95dda-4850-4ba6-a34d-12802dd49631
"""
An orthographic camera: seen from `azimuth` (degrees, about the vertical axis z) and `elevation`
(degrees above the horizontal). Returns the screen's right and up and the direction toward the viewer.
"""
function camera(azimuth, elevation)
    θ, φ = deg2rad(azimuth), deg2rad(elevation)
    toward = (cos(φ) * cos(θ), cos(φ) * sin(θ), sin(φ))
    right = (-sin(θ), cos(θ), 0.0)
    up = (
        toward[2] * right[3] - toward[3] * right[2],
        toward[3] * right[1] - toward[1] * right[3],
        toward[1] * right[2] - toward[2] * right[1],
    )
    return (; right, up, toward)
end

# ╔═╡ f3e47edf-5e45-4d71-8b44-89768e87a76a
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ 517c6f56-669d-4d73-9b25-d1cb4d396e81
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ c21aedb3-3ef4-4978-adec-e62119468b9c
"""The six faces of a block: outward normal and corners."""
function faces(b::Block)
    (x0, y0, z0), (x1, y1, z1) = b.lo, b.hi
    return (
        ((0, 0, 1), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0, 0, -1), ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))),
        ((1, 0, 0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1, 0, 0), ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))),
        ((0, 1, 0), ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))),
        ((0, -1, 0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
    )
end

# ╔═╡ 6ff3ad7b-9969-4ed8-93e4-c84c35e30998
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ 0536bb6b-fdb7-4e3d-933c-77318a5c808b
"""
Draw `blocks` back to front (the painter's way), each with its visible faces shaded, and any
`extras` (functions drawing lines or arrows, with their depth) in their place among them.
Blocks should be of similar size (split long walls into pieces) for the order to be right.
"""
function draw_blocks!(ax, cam, blocks; extras = [], edge = Mke.RGBf(0.1, 0.12, 0.16), edge_alpha = 0.35)
    items = Any[(dot3(map(+, b.lo, b.hi) ./ 2, cam.toward), b) for b in blocks]
    append!(items, extras)
    sort!(items; by = first)
    polygons = Vector{Mke.Point2f}[]
    colors = Mke.RGBAf[]
    strokes = Mke.RGBAf[]
    function flush!()
        isempty(polygons) && return
        Mke.poly!(
            ax,
            copy(polygons);
            color = copy(colors),
            strokecolor = copy(strokes),
            strokewidth = 0.7,
        )
        empty!(polygons)
        empty!(colors)
        return empty!(strokes)
    end
    for (_, item) in items
        if item isa Block
            for (normal, corners) in faces(item)
                dot3(normal, cam.toward) > 1.0e-9 || continue
                push!(polygons, [project(cam, c) for c in corners])
                c = shade(item.color, normal, cam)
                push!(colors, Mke.RGBAf(c.r, c.g, c.b, item.alpha))
                # a seam between tiles of one layer is drawn in the face's own colour
                push!(
                    strokes,
                    item.stroke > 0 ? Mke.RGBAf(edge.r, edge.g, edge.b, edge_alpha * item.alpha) :
                        Mke.RGBAf(c.r, c.g, c.b, item.alpha),
                )
            end
        else
            flush!()
            item(ax)
        end
    end
    flush!()
    return ax
end

# ╔═╡ c356dd1c-8044-48a8-b1b7-6d4143ee8826
"""A dexel array's blocks: ceramic in each cell, reflector walls between (in pieces), from z0 to z1."""
function dexel_blocks(nx, ny, pitch, gap, z0, z1; ceramic = MATERIALS.ufc, lit = nothing)
    blocks = Block[]
    p, s = pitch, gap
    for i in 0:(nx - 1), j in 0:(ny - 1)
        x, y = i * p[1], j * p[2]
        color = lit !== nothing && (i, j) in lit ? MATERIALS.ufc_lit : ceramic
        push!(blocks, Block((x + s / 2, y + s / 2, z0), (x + p[1] - s / 2, y + p[2] - s / 2, z1), color))
        # the reflector on this cell's low-x and low-y sides, and the corner post
        push!(blocks, Block((x - s / 2, y + s / 2, z0), (x + s / 2, y + p[2] - s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((x + s / 2, y - s / 2, z0), (x + p[1] - s / 2, y + s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((x - s / 2, y - s / 2, z0), (x + s / 2, y + s / 2, z1), MATERIALS.reflector))
    end
    # the far walls
    X, Y = nx * p[1], ny * p[2]
    for j in 0:(ny - 1)
        y = j * p[2]
        push!(blocks, Block((X - s / 2, y + s / 2, z0), (X + s / 2, y + p[2] - s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((X - s / 2, y - s / 2, z0), (X + s / 2, y + s / 2, z1), MATERIALS.reflector))
    end
    for i in 0:(nx - 1)
        x = i * p[1]
        push!(blocks, Block((x + s / 2, Y - s / 2, z0), (x + p[1] - s / 2, Y + s / 2, z1), MATERIALS.reflector))
        push!(blocks, Block((x - s / 2, Y - s / 2, z0), (x + s / 2, Y + s / 2, z1), MATERIALS.reflector))
    end
    push!(blocks, Block((X - s / 2, Y - s / 2, z0), (X + s / 2, Y + s / 2, z1), MATERIALS.reflector))
    return blocks
end

# ╔═╡ d1aa97ab-cc9c-4bb0-82c9-be7565a26ae6
"""
The walls of a grid over an nx × ny array, in pieces one cell long, from z0 to z1: `walls = :both`
for a 2-D grid, `:x` for a 1-D grid's plates between the columns (running along z).
"""
function grid_blocks(nx, ny, pitch, wall, z0, z1; color = MATERIALS.tungsten, walls = :both)
    blocks = Block[]
    p, w = pitch, wall
    if walls === :x
        for i in 0:nx, j in 0:(ny - 1)
            x, y = i * p[1], j * p[2]
            push!(blocks, Block((x - w / 2, y, z0), (x + w / 2, y + p[2], z1), color))
        end
        return blocks
    end
    for i in 0:nx, j in 0:(ny - 1)
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x - w / 2, y + w / 2, z0), (x + w / 2, y + p[2] - w / 2, z1), color))
    end
    for i in 0:(nx - 1), j in 0:ny
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x + w / 2, y - w / 2, z0), (x + p[1] - w / 2, y + w / 2, z1), color))
    end
    for i in 0:nx, j in 0:ny
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x - w / 2, y - w / 2, z0), (x + w / 2, y + w / 2, z1), color))
    end
    return blocks
end

# ╔═╡ 3126d46a-d155-4d3e-b623-9209d440d84a
"""A layer cut into one tile per cell (so the painter's order holds), from z0 to z1."""
function layer_blocks(nx, ny, pitch, z0, z1, color; inset = 0.0)
    return [
        Block(
                (i * pitch[1] + inset, j * pitch[2] + inset, z0),
                ((i + 1) * pitch[1] - inset, (j + 1) * pitch[2] - inset, z1),
                color;
                stroke = inset > 0 ? 0.6 : 0.0,
            )
            for i in 0:(nx - 1) for j in 0:(ny - 1)
    ]
end

# ╔═╡ 5ba1e6a0-564e-42ea-88d1-3dddd70e0886
"""
A label with a leader line: a dot at `anchor`, a line to `at`, and the text there, on a
translucent rounded background.
"""
function callout!(
        ax,
        anchor,
        at,
        text;
        color = Mke.RGBf(0.13, 0.13, 0.16),
        align = (:left, :center),
        fontsize = 17,
        background = (:white, 0.9),
        line = (Mke.RGBf(0.3, 0.3, 0.35), 0.8),
        font = :regular,
    )
    Mke.lines!(ax, Mke.Point2f[anchor, at]; color = line, linewidth = 1.3)
    Mke.scatter!(ax, [anchor]; color = line, markersize = 7)
    return Mke.textlabel!(
        ax,
        at;
        text,
        text_align = align,
        text_color = color,
        fontsize,
        font,
        background_color = background,
        strokewidth = 0,
        padding = (8, 8, 5, 5),
        cornerradius = 6,
    )
end

# ╔═╡ 4b31992d-307b-45e6-ba53-97844f8e05e4
"""A blank canvas for an illustration: equal scales, no axes."""
function canvas(position; background = :white, title = "", kw...)
    ax = Mke.Axis(
        position;
        aspect = Mke.DataAspect(),
        backgroundcolor = background,
        title,
        kw...,
    )
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    return ax
end

# ╔═╡ f17692d1-0d59-44c4-bdc2-2a6e8caf55a3
"""
A photon drawn as a wave from `a` to `b` (2-D), with `cycles` oscillations (more for a higher
energy) and an arrowhead at `b`.
"""
function photon!(ax, a, b; color = MATERIALS.beam, cycles = 6, amplitude = 0.06, linewidth = 3, head = 14)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    d = b - a
    L = sqrt(d[1]^2 + d[2]^2)
    t = d / L
    n = Mke.Point2f(-t[2], t[1])
    s = range(0, 1; length = 240)
    ramp(u) = min(1, 8u, 8(1 - u))  # the wave starts and ends on the line
    points = Mke.Point2f[a + u * d + amplitude * L * ramp(u) * sin(2π * cycles * u) * n for u in s]
    Mke.lines!(ax, points[1:(end - 6)]; color, linewidth)
    Mke.scatter!(
        ax,
        [b];
        marker = :utriangle,
        markersize = head,
        rotation = atan(t[2], t[1]) - π / 2,
        color,
    )
    return nothing
end

# ╔═╡ 6f910999-861d-4c55-a413-7817cddd2337
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ 49a94f31-0592-4639-bded-a21d8386ada0
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ 43065420-8e14-436c-a9bb-a2c60286508d
"""
Spread labels down the right of a drawing: each `entry` is `(; anchor, title, detail, color, tag)`
(`tag` is `nothing` or `(text, color)`). Labels keep at least `spacing` apart (data units), in the
order of their anchors, and a bent leader line runs from each anchor to its label at `x`.
"""
function labels_right!(ax, entries, x; spacing = 1.0, tag_width = 0.0, title_size = 19, detail_size = 15.5)
    order = sortperm([e.anchor[2] for e in entries]; rev = true)
    ys = Float64[]
    for k in order
        y = entries[k].anchor[2]
        isempty(ys) || (y = min(y, last(ys) - spacing))
        push!(ys, y)
    end
    # centre the column of labels on the anchors
    shift = (sum(e.anchor[2] for e in entries) - sum(ys)) / length(ys)
    ys .+= max(0, shift)
    for (k, y) in zip(order, ys)
        e = entries[k]
        elbow = Mke.Point2f(x - 0.35 * spacing, y)
        Mke.lines!(ax, Mke.Point2f[e.anchor, Mke.Point2f(elbow[1] - 0.3 * spacing, e.anchor[2]), elbow, Mke.Point2f(x, y)]; color = (:black, 0.32), linewidth = 1.2)
        Mke.scatter!(ax, [Mke.Point2f(e.anchor)]; color = (:black, 0.55), markersize = 7)
        at = Mke.Point2f(x, y)
        if e.tag !== nothing
            Mke.textlabel!(ax, at; text = e.tag[1], text_align = (:left, :center), text_color = :white, font = :bold, fontsize = 13, offset = (6, 0), background_color = e.tag[2], strokewidth = 0, padding = (8, 8, 3, 3), cornerradius = 9)
        end
        dx = e.tag === nothing ? 6 : 6 + tag_width
        Mke.text!(ax, at; text = e.title, font = :bold, fontsize = title_size, align = (:left, :bottom), offset = (dx, 1), color = e.color)
        e.detail == "" || Mke.text!(ax, at; text = e.detail, fontsize = detail_size, align = (:left, :top), offset = (dx, -3), color = Mke.RGBf(0.42, 0.42, 0.46))
    end
    return nothing
end

# ╔═╡ 691eb609-3c95-4091-8951-803053559dff
"""A vertical bracket at `x` from `y0` to `y1`, with `text` to its left."""
function side_bracket!(ax, x, y0, y1, text; color, fontsize = 16, tick = 0.08)
    Mke.lines!(ax, Mke.Point2f[(x + tick, y0), (x, y0), (x, y1), (x + tick, y1)]; color, linewidth = 2)
    return Mke.text!(ax, x - tick, (y0 + y1) / 2; text, color, fontsize, font = :bold, align = (:right, :center), justification = :right)
end

# ╔═╡ ad5a2ff3-ec50-4d51-8e2b-2b08382eeb4a
"""A small rounded tag ("In Geant4" / "Not Simulated")."""
function tag!(ax, at, text, color; align = (:left, :center), fontsize = 14, offset = (0, 0))
    return Mke.textlabel!(
        ax,
        at;
        text,
        text_align = align,
        text_color = :white,
        font = :bold,
        fontsize,
        offset,
        background_color = color,
        strokewidth = 0,
        padding = (8, 8, 3, 3),
        cornerradius = 9,
    )
end

# ╔═╡ 81b6e112-d19d-4f34-a160-443efbab8bbb
TAGS = (geant4 = Mke.RGBf(0.2, 0.55, 0.32), not_simulated = Mke.RGBf(0.52, 0.52, 0.57))

# ╔═╡ eab9ebaf-0e72-4d17-9e6b-6d1ba535d9c5
"""
The blocks of a patch of the PUREViSION detector, pulled apart layer by layer: the readout
electronics, the photodiodes, the ceramic dexels in their reflector, the 1-D grid's plates (drawn
`grid_drawn` mm tall). Canon documents none of the layers below the ceramic, so they are drawn
generically. Returns each layer's blocks (`layers[key]`), its (bottom, top) (`levels[key]`), and all
the blocks together (`blocks`).
"""
function exploded_detector(nx, ny; gap = 1.0, grid_drawn = 2.4)
    p, s, d = pitch_mm, septum_mm, depth_mm
    stack = (
        (key = :electronics, h = 0.45, gap = 0.9),
        (key = :photodiode, h = 0.18, gap = gap),
        (key = :ceramic, h = d, gap = gap),
        (key = :grid, h = grid_drawn, gap = 0.0),
    )
    layers = Dict{Symbol, Vector{Block}}()
    levels = Dict{Symbol, Tuple{Float64, Float64}}()
    z = 0.0
    for layer in stack
        z0, z1 = z, z + layer.h
        levels[layer.key] = (z0, z1)
        layers[layer.key] = if layer.key === :electronics
            vcat(
                layer_blocks(nx, ny, p, z0, z1, MATERIALS.board),
                # a connector under every dexel, carrying its charge down to the electronics
                [
                    Block((c[1] - 0.06, c[2] - 0.06, z1), (c[1] + 0.06, c[2] + 0.06, z1 + 0.9), MATERIALS.gold)
                        for c in (((i + 0.5) * p[1], (j + 0.5) * p[2]) for i in 0:(nx - 1) for j in 0:(ny - 1))
                ],
            )
        elseif layer.key === :photodiode
            vcat(
                layer_blocks(nx, ny, p, z0, z0 + 0.08, MATERIALS.cmos),
                layer_blocks(nx, ny, p, z0 + 0.08, z1, MATERIALS.photodiode; inset = s / 2),
            )
        elseif layer.key === :ceramic
            dexel_blocks(nx, ny, p, s, z0, z1)
        else
            grid_blocks(nx, ny, p, grid.wall_mm, z0, z1; walls = :x)
        end
        z = z1 + layer.gap
    end
    return (; layers, levels, blocks = reduce(vcat, values(layers)))
end

# ╔═╡ 640cd996-29e0-42fa-a581-9abf401ec18d
"""A block drawn faded toward the background (`amount` 0: as is, 1: gone)."""
function faded(b::Block, amount; to = Mke.RGBf(1, 1, 1))
    mix(a, c) = a + amount * (c - a)
    color = Mke.RGBf(mix(b.color.r, to.r), mix(b.color.g, to.g), mix(b.color.b, to.b))
    return Block(; lo = b.lo, hi = b.hi, color, alpha = b.alpha, stroke = b.stroke)
end

# ╔═╡ ba771f9a-8ff5-456d-87e0-9e4c145f39e1
"""The convex hull of points (2-D), counter-clockwise, closed."""
function hull(points)
    pts = sort(unique(Mke.Point2f.(points)); by = p -> (p[1], p[2]))
    length(pts) < 3 && return pts
    cross2(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    lower, upper = Mke.Point2f[], Mke.Point2f[]
    for p in pts
        while length(lower) >= 2 && cross2(lower[end - 1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end
    for p in reverse(pts)
        while length(upper) >= 2 && cross2(upper[end - 1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end
    return vcat(lower[1:(end - 1)], upper[1:(end - 1)], lower[1:1])
end

# ╔═╡ a1697bfb-6249-4f64-9e50-e1f2befdbae7
"""Outline a box (lo, hi) as seen by `cam`: the edge of its silhouette."""
function outline!(ax, cam, lo, hi; color = MATERIALS.beam, linewidth = 3, linestyle = :solid)
    corners = [(x, y, z) for x in (lo[1], hi[1]) for y in (lo[2], hi[2]) for z in (lo[3], hi[3])]
    return Mke.lines!(ax, hull([project(cam, c) for c in corners]); color, linewidth, linestyle)
end

# ╔═╡ 1718e123-ab81-465e-9b45-09c7c80b4351
"""
The master map, as a band across the top of a section's first illustration: the gantry, one
dexel's stack with the part under study (`focus`) lit and outlined, and the dexel's five steps with
this one's lit. `focus` is one of `:tube`, `:arc`, `:grid`, `:ceramic`, `:photodiode`, `:electronics`,
`:whole`. Place it with `locator!(fig[0, :], focus; note)`.
"""
function locator!(position, focus; note = "")
    amber, dim = MATERIALS.beam, Mke.RGBf(0.82, 0.83, 0.86)
    Mke.Box(position; color = Mke.RGBf(0.965, 0.965, 0.975), strokevisible = false, cornerradius = 14)
    layout = Mke.GridLayout(position; halign = :center, tellwidth = false)

    # The gantry, small: tube A, its fan, the detector arc, the one dexel below
    ax = canvas(layout[1, 1]; width = 230, height = 230, backgroundcolor = :transparent)
    θ = range(0, 2π; length = 120)
    R_iso, R_det = aquilion.source_to_isocenter_mm, aquilion.source_to_detector_mm
    half_fan = asin(aquilion.scan_field_mm / 2 / R_iso)
    lit(k) = focus === k || focus === :whole
    Mke.lines!(ax, 800 .* cos.(θ), 800 .* sin.(θ); color = dim, linewidth = 1.5)
    Mke.poly!(ax, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = (PALETTE.muted, 0.25))
    src = Mke.Point2f(0, R_iso)
    φ = range(-half_fan, half_fan; length = 60)
    arc = [src + R_det * Mke.Point2f(sin(t), -cos(t)) for t in φ]
    Mke.poly!(ax, Mke.Point2f.(vcat([src], arc)); color = (amber, lit(:tube) ? 0.45 : 0.12))
    Mke.lines!(ax, Mke.Point2f.(arc); color = lit(:arc) ? PALETTE.ink : dim, linewidth = lit(:arc) ? 7 : 4)
    Mke.scatter!(ax, [src]; marker = :rect, markersize = 16, color = lit(:tube) ? PALETTE.ink : dim)
    Mke.scatter!(ax, [arc[30]]; marker = :rect, markersize = 9, color = amber, strokecolor = PALETTE.ink, strokewidth = 1)
    Mke.limits!(ax, -880, 880, -720, 880)

    # One dexel's stack, exploded; the focus lit and outlined, the rest faded
    axs = canvas(layout[1, 2]; width = 170, height = 250, backgroundcolor = :transparent)
    ex = exploded_detector(1, 1; gap = 0.8)
    lit_layers = Dict(
        :grid => (:grid,),
        :ceramic => (:ceramic,),
        :photodiode => (:photodiode,),
        :electronics => (:electronics,),
        :whole => Tuple(keys(ex.layers)),
    )
    on = get(lit_layers, focus, ())
    blocks = Block[]
    for (key, bs) in ex.layers
        append!(blocks, key in on ? bs : [faded(b, 0.86) for b in bs])
    end
    cam = camera(-56, 30)
    draw_blocks!(axs, cam, blocks; edge_alpha = 0.15)
    p = pitch_mm
    for key in on
        z0, z1 = ex.levels[key]
        outline!(axs, cam, (-0.06, -0.06, z0 - 0.06), (p[1] + 0.06, p[2] + 0.06, z1 + 0.06); linewidth = 2.5)
    end

    # Where we are, in words, and the dexel's five steps with this one's lit
    text = Mke.GridLayout(layout[1, 3]; valign = :center)
    Mke.Label(text[1, 1], "Where We Are"; font = :bold, fontsize = 20, color = PALETTE.ink, halign = :left)
    Mke.Label(text[2, 1], note; fontsize = 17, color = PALETTE.ink, halign = :left, justification = :left)
    steps = (
        (:grid, "1  The Grid"),
        (:ceramic, "2  The Ceramic Stops It"),
        (:ceramic, "3  The Ceramic Glows"),
        (:photodiode, "4  The Photodiode"),
        (:electronics, "5  The Electronics Digitize It"),
    )
    for (k, (key, label)) in enumerate(steps)
        active = key in on
        Mke.Label(
            text[k + 2, 1],
            label;
            fontsize = 15,
            font = active ? :bold : :regular,
            color = active ? Mke.RGBf(0.8, 0.5, 0.05) : PALETTE.muted,
            halign = :left,
        )
    end
    Mke.rowgap!(text, 3)
    Mke.rowgap!(text, 1, 6)
    Mke.rowgap!(text, 2, 10)
    Mke.colgap!(layout, 28)
    return layout
end

# ╔═╡ 4f9450a4-2b36-46da-8244-eab441445365
"""Side view of three dexels (x across, depth down as negative y), at x offsets; returns nothing."""
function side_dexels!(ax; n = 3, lit = nothing, depth = depth_mm, ceramic = MATERIALS.ufc)
    p, s = pitch_mm[1], septum_mm
    half = (n - 1) / 2
    rect!(ax, -(half + 0.5) * p, -depth, n * p, depth, MATERIALS.reflector; strokecolor = (:black, 0.3), strokewidth = 1)
    for i in 0:(n - 1)
        x = (i - half) * p
        rect!(ax, x - (p - s) / 2, -depth, p - s, depth, lit !== nothing && i == lit ? Mke.RGBf(0.82, 0.95, 0.84) : ceramic)
    end
    return nothing
end

# ╔═╡ 8516f41d-d952-47fc-92a0-a06b0f359dc0
"""A curved strip (part of a cylinder about the z axis at the focal spot), as quads with depth, for the painter."""
function arc_quads(cam, R, half_angle, half_z; columns = 46, thickness = 18.0, color = MATERIALS.ufc)
    quads = Any[]
    for k in 1:columns
        a0 = -half_angle + (k - 1) * 2half_angle / columns
        a1 = a0 + 2half_angle / columns
        for (r, normal_sign) in ((R, -1), (R + thickness, 1))
            corners = [(r * sin(a), -r * cos(a), z) for (a, z) in ((a0, -half_z), (a1, -half_z), (a1, half_z), (a0, half_z))]
            mid = (a0 + a1) / 2
            n = (normal_sign * sin(mid), -normal_sign * cos(mid), 0.0)
            depth = dot3(corners[1] .+ corners[3], cam.toward) / 2
            push!(
                quads, (
                    depth, ax -> begin
                        dot3(n, cam.toward) > 0 || return
                        c = shade(color, n, cam)
                        Mke.poly!(ax, [project(cam, q) for q in corners]; color = c, strokecolor = (:black, 0.25), strokewidth = 0.5)
                    end,
                )
            )
        end
    end
    return quads
end

# ╔═╡ f632ab48-f213-4dbd-a2cf-dedb3cc00b95
"""A Gd atom: nucleus and two shells, at `c` with outer radius `r`."""
function atom!(ax, c, r; color = Mke.RGBf(0.45, 0.47, 0.55))
    θ = range(0, 2π; length = 100)
    for (k, rr) in enumerate((0.55r, r))
        Mke.lines!(ax, c[1] .+ rr .* cos.(θ), c[2] .+ rr .* sin.(θ); color = (color, 0.55), linewidth = 1.5)
    end
    Mke.scatter!(ax, [Mke.Point2f(c)]; color, markersize = 18)
    return nothing
end

# ╔═╡ ddadd9bd-a2b7-58f8-a970-3bff3a89201e
md"""
## 1. The Scintillator

Canon describes the PUREViSION scintillator only as a "praseodymium-activated" ceramic, cut from a
solid ingot. Praseodymium is the activator of the gadolinium oxysulfide ceramics used across CT,
Gd₂O₂S:Pr, so this notebook takes the PUREViSION ceramic to be Gd₂O₂S at **7.31 g/cm³**, the density
Siemens measures for its own Gd₂O₂S ceramic. That is an assumption: Canon publishes neither the host
material nor the density.
"""

# ╔═╡ 54cfabcd-ec9e-5930-af26-fe46ae2f6df2
# Gd₂O₂S ceramic at 7.31 g/cm³ (assumed: Canon publishes neither the host material nor the density)
gos = MC.DetectorMaterial("GOS ceramic"; density = 7.31, formula = "Gd2O2S")

# ╔═╡ bfd257ef-c874-5d3e-865b-91eca06e30f1
# Gadolinium's K-shell: its edge and the energy of the x ray it emits (Kα)
gadolinium = only(MC.k_edges(gos))

# ╔═╡ ce933054-651a-5933-9566-efb31407e7cd
md"""
Gadolinium's **K-edge, 50.24 keV**, sits in the middle of every CT spectrum. Below it a photon
cannot free a K-shell electron; just above it, it can, and absorption jumps sevenfold. Every
such absorption leaves the atom to emit, nine times in ten, a **Gd K x ray of about 43 keV**,
which is just below the edge and so is absorbed only weakly: it can travel a millimetre or more
before stopping, which is the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ f2132f86-2dcb-5010-bb46-b3216f321502
# The ways a photon can interact, as the figures name them
processes = (
    (kind = :total, label = "All Processes Together (Total)", color = PALETTE.ink),
    (
        kind = :photoelectric,
        label = "Photon Absorbed (Photoelectric)",
        color = PALETTE.electron,
    ),
    (
        kind = :compton,
        label = "Bounces Off, Loses Energy (Compton)",
        color = PALETTE.photon,
    ),
    (
        kind = :rayleigh,
        label = "Bounces Off, Keeps Energy (Rayleigh)",
        color = PALETTE.muted,
    ),
)

# ╔═╡ f7c3b5dc-b20b-5db8-bbc9-8b7b9ad389bb
mu_energies = 10.0:0.25:150.0

# ╔═╡ 4e886e38-891b-5244-b3f7-f4217cb5785c
# Geant4's attenuation of the ceramic, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 919250d4-f19e-5252-bad8-a0191e7f3dd9
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 2cecfe74-4b79-4662-850f-2fb93923e463
let
    μ_per_mm(E) = MC.interpolate_linear(collect(mu_energies), mu_geant4[:total], E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1700, 790))

    red, blue, grey = PALETTE.electron, PALETTE.photon, PALETTE.muted
    panels = (
        ("Photon Absorbed", "Photoelectric: It Vanishes; an Electron\nand Often a 43 keV Gd X Ray Leave", red),
        ("Bounces Off, Loses Energy", "Compton: a Weaker Photon Leaves,\nan Electron Recoils", blue),
        ("Bounces Off, Keeps Energy", "Rayleigh: Same Energy,\na Small Change of Direction", grey),
    )
    for (k, (title, detail, color)) in enumerate(panels)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 21)
        Mke.limits!(ax, -1.25, 1.25, -1.05, 0.95)
        atom!(ax, (0, 0), 0.32)
        photon!(ax, (-1.2, 0.0), (-0.36, 0.0); cycles = 6, amplitude = 0.07, color = MATERIALS.beam)
        if k == 1
            arrow!(ax, (0.12, 0.1), (0.62, 0.55); color = red, linewidth = 3)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = red, fontsize = 16, align = (:left, :bottom))
            photon!(ax, (0.2, -0.24), (1.0, -0.62); cycles = 4, amplitude = 0.07, color = PALETTE.xray)
            Mke.text!(ax, 0.98, -0.75; text = "Gd X Ray", color = PALETTE.xray, fontsize = 16, align = (:right, :top))
        elseif k == 2
            photon!(ax, (0.3, -0.18), (1.1, -0.72); cycles = 3.5, amplitude = 0.07, color = blue)
            Mke.text!(ax, 1.05, -0.86; text = "Weaker Photon", color = blue, fontsize = 16, align = (:right, :top))
            arrow!(ax, (0.2, 0.2), (0.62, 0.55); color = red, linewidth = 2.5)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = red, fontsize = 16, align = (:left, :bottom))
        else
            photon!(ax, (0.35, 0.05), (1.2, 0.28); cycles = 6, amplitude = 0.07, color = MATERIALS.beam)
            Mke.text!(ax, 1.2, 0.42; text = "Same Energy", color = grey, fontsize = 16, align = (:right, :bottom))
        end
        Mke.text!(ax, 0, -0.42; text = "Gd Atom", color = Mke.RGBf(0.4, 0.42, 0.5), fontsize = 15, align = (:center, :top))
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end

    # How far a photon goes, on average, in the ceramic: 1/μ, against the crystal's 1.4 mm
    ax = Mke.Axis(
        fig[1:2, 4];
        title = "How Far a Photon Goes in the Ceramic",
        xlabel = "Mean Distance Before Interacting (mm)",
        yticks = (1:4, ["40 keV", "60 keV", "100 keV", "140 keV"]),
        yreversed = true,
        limits = ((0, 2.1), (0.4, 4.6)),
        titlesize = 19,
        xlabelsize = 16,
    )
    Mke.vspan!(ax, 0, depth_mm; color = (MATERIALS.ufc, 0.6))
    Mke.text!(ax, depth_mm / 2, 4.5; text = "The 1.4 mm Crystal", align = (:center, :bottom), fontsize = 15, color = PALETTE.slab_edge)
    for (k, E) in enumerate((40.0, 60.0, 100.0, 140.0))
        L = 1 / μ_per_mm(E)
        Mke.barplot!(ax, [k], [L]; direction = :x, color = MATERIALS.beam, width = 0.5)
        Mke.text!(ax, L, k; text = " $(round(L; digits = 2)) mm", align = (:left, :center), fontsize = 15)
    end
    Mke.colsize!(fig.layout, 4, Mke.Relative(0.24))
    locator!(fig[0, :], :ceramic; note = "Step 2 · The Ceramic\nStops the X Ray")
    fig
end

# ╔═╡ 94ece8ce-3e4d-53b1-8769-420e883094e7
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How Strongly the Gd₂O₂S Ceramic Stops X Rays (7.31 g/cm³)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Chance of Interacting per mm, μ (1/mm)",
        yscale = log10,
        xticks = 10:20:150,
        limits = ((10, 150), (4.0e-3, 400)),
    )
    for p in processes
        Mke.lines!(
            ax,
            mu_energies,
            mu_geant4[p.kind];
            color = p.color,
            label = p.label,
            linewidth = p.kind === :total ? 3.5 : 2.5,
        )
        Mke.lines!(
            ax,
            mu_energies,
            mu_xcom[p.kind];
            color = (p.color, 0.7),
            linestyle = :dash,
            linewidth = 1.6,
        )
    end
    Mke.vlines!(
        ax,
        gadolinium.edge_keV;
        color = (PALETTE.xray, 0.8),
        linestyle = :dot,
        linewidth = 2,
    )
    note!(
        ax,
        gadolinium.edge_keV,
        0.4,
        "Gd K-Edge, $(round(gadolinium.edge_keV; digits = 2)) keV";
        offset = (8, 0),
        color = PALETTE.xray,
    )
    note!(
        ax,
        148,
        250,
        "Solid: Geant4 (Used)    Dashed: NIST XCOM";
        align = (:right, :top),
        fontsize = 15,
        color = PALETTE.muted,
    )
    Mke.Legend(
        fig[2, 1],
        ax;
        orientation = :horizontal,
        nbanks = 2,
        tellheight = true,
    )
    fig
end

# ╔═╡ 6926962b-931d-56b2-be23-590de184f48e
let
    total = maximum(abs.(mu_geant4[:total] ./ mu_xcom[:total] .- 1))
    rayleigh = maximum(abs.(mu_geant4[:rayleigh] ./ mu_xcom[:rayleigh] .- 1))
    Markdown.parse(
        """
        Geant4 and XCOM agree on the total attenuation to within **$(round(100total; digits = 1)) %** from 10 to
        150 keV. They differ most on Rayleigh scattering (up to $(round(100rayleigh; digits = 0)) %), which Geant4
        computes with anomalous scattering near the K-edge; Rayleigh is a small part of the total.
        """
    )
end

# ╔═╡ fe6f7616-c5c3-5864-b959-5fd0d16bb6be
md"""
### A First Check: A Published Gd₂O₂S Curve

Canon publishes no absorption curve, but Siemens publishes the fraction of x rays that **1.4 mm
of its own Gd₂O₂S ceramic** absorbs (*Detectors & Crystals*, p. 19). It is the probability that a
photon interacts at all, `1 − exp(−μd)`, so it checks the material model and the density assumed
here. The values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ 4a1a5d21-5581-50d2-b02a-9174b8bfcd02
# Siemens' curve, read from the vector figure (reference_data/siemens_ufc_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_ufc_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ 3e588088-f4e1-5470-a7d8-f02f18437d48
# The crystal depth (assumed: not published)
depth_mm = 1.4

# ╔═╡ 17fe60de-9741-5ec5-9555-7afb4820e74b
# The probability a photon interacts in 1.4 mm of the ceramic
interaction_curve = [1 - exp(-MC.geant4_attenuation(gos, E) * 1.4) for E in mu_energies]

# ╔═╡ b59c72be-f16c-480f-946a-978ec616d260
let
    μ_per_mm(E) = MC.interpolate_linear(collect(mu_energies), mu_geant4[:total], E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 500))
    N = 30
    for (k, E) in enumerate((45.0, 90.0, 140.0))
        μ = μ_per_mm(E)
        stopped = 1 - exp(-μ * depth_mm)
        ax = canvas(fig[1, k]; title = "$(Int(E)) keV: $(round(Int, 100stopped)) % Stopped", titlesize = 21)
        rect!(ax, 0, 0, 1.6, depth_mm, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        # N photons; photon i stops at the depth that the i-th of N quantiles of exp(−μx) gives
        for i in 1:N
            x = 1.6 * (i - 0.5) / N
            depth = -log(1 - (mod(i * 0.618034, 1.0))) / μ
            if depth < depth_mm
                Mke.lines!(ax, [x, x], [depth_mm + 0.55, depth_mm - depth]; color = MATERIALS.beam, linewidth = 2)
                Mke.scatter!(ax, [x], [depth_mm - depth]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (x, depth_mm + 0.55), (x, -0.5); color = (MATERIALS.beam, 0.6), linewidth = 2, head = 10)
            end
        end
        Mke.text!(ax, 1.65, depth_mm / 2; text = "1.4 mm of\nCeramic", align = (:left, :center), fontsize = 15, color = PALETTE.slab_edge)
        Mke.limits!(ax, -0.05, 2.15, -0.62, depth_mm + 0.65)
    end
    Mke.Label(
        fig[2, 1:3],
        "Each Line Is a Photon from Above: Red Dots Are Where It Interacts; Arrows Pass Straight Through. The Fraction Stopped Is 1 − exp(−μ · 1.4 mm).";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 3d6ad596-a1da-5c24-b39a-8cdc0a1314da
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Fraction of Photons 1.4 mm of Gd₂O₂S Ceramic Absorbs",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons Absorbed",
        xticks = 20:20:150,
        limits = ((15, 152), (0.45, 1.04)),
    )
    Mke.lines!(
        ax,
        mu_energies,
        interaction_curve;
        color = PALETTE.photon,
        linewidth = 3.5,
        label = "Geant4, Gd₂O₂S at 7.31 g/cm³",
    )
    Mke.rangebars!(
        ax,
        siemens_curve.energy_keV,
        siemens_curve.absorption .- 0.025,
        siemens_curve.absorption .+ 0.025;
        color = (PALETTE.ink, 0.5),
        whiskerwidth = 8,
    )
    Mke.scatter!(
        ax,
        siemens_curve.energy_keV,
        siemens_curve.absorption;
        color = PALETTE.ink,
        markersize = 13,
        label = "Siemens' Figure for Its Gd₂O₂S Ceramic (± 0.025)",
    )
    note!(
        ax,
        gadolinium.edge_keV,
        0.88,
        "Gd K-Edge:\nAbsorption Jumps";
        align = (:right, :center),
        offset = (-12, 0),
        color = PALETTE.xray,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 157eec79-393b-5b59-bb8a-950a2c4f5802
let
    ours = [MC.interpolate_linear(collect(mu_energies), interaction_curve, E) for E in siemens_curve.energy_keV]
    low = siemens_curve.energy_keV .<= 90
    Markdown.parse(
        """
        Up to 90 keV the published curve and Geant4 for 1.4 mm of Gd₂O₂S at 7.31 g/cm³ agree to within
        **$(round(maximum(abs, (siemens_curve.absorption .- ours)[low]); digits = 3))**, inside the figure's reading accuracy: the material model is sound.
        Whether the PUREViSION ceramic is this material at this density, nothing published says.
        """
    )
end

# ╔═╡ fcee931d-a949-5b8c-9bee-d59e556f0fc1
md"""
## 2. One Dexel and Its Neighbours

A dexel is one block of ceramic with reflector on its four sides. Its pitch at the detector face is
not published, so it is worked out from what is: 320 rows of 0.5 mm at the isocentre, magnified by
1072 / 600 along z; and 896 channels across the 50 cm scan field, on an arc of 1072 mm across the
fan. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy is assumed. Neither is the
crystal's depth; 1.4 mm is assumed, and the table is saved for other depths too.
"""

# ╔═╡ e01b99a8-d89e-53cd-9b56-b4dcc5c374db
# The Aquilion ONE's geometry: 600 mm (Geleijns et al. 2009) and 1072 mm (secondary source)
aquilion = (
    source_to_isocenter_mm = 600.0,
    source_to_detector_mm = 1072.0,
    columns = 896,
    rows = 320,
    row_at_isocenter_mm = 0.5,
    scan_field_mm = 500.0,
)

# ╔═╡ 2594a24f-dc08-533c-8b9d-2f99108aab4e
# The dexel pitch at the detector face (across the fan, along z), in mm: the 50 cm scan field's fan
# angle shared among the channels, and the row width magnified to the detector
pitch_mm = let
    fan = 2 * asin(aquilion.scan_field_mm / 2 / aquilion.source_to_isocenter_mm)
    (
        aquilion.source_to_detector_mm * fan / aquilion.columns,
        aquilion.row_at_isocenter_mm * aquilion.source_to_detector_mm / aquilion.source_to_isocenter_mm,
    )
end

# ╔═╡ 219cb299-5acc-5911-9eaf-2e86161e449d
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ 92ec3c2f-5d09-56f1-95fd-e5d2d1450ff2
# The block of ceramic each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ f266e1ff-8e7c-52ae-acfb-8fc1c6f2ca9f
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 498b5b6a-a948-5126-92a3-364216f7b20e
let
    fill = crystal_mm ./ pitch_mm
    Markdown.parse(
        """
        The dexel pitch at the face is **$(round(pitch_mm[1]; digits = 3)) × $(round(pitch_mm[2]; digits = 3)) mm**, and the ceramic block inside it
        $(round(crystal_mm[1]; digits = 3)) × $(round(crystal_mm[2]; digits = 3)) mm: the ceramic covers $(round(fill[1]; digits = 3)) × $(round(fill[2]; digits = 3)) of the face, the fill factor a
        BasisSimulator model of this scanner would use.
        """
    )
end

# ╔═╡ 85053c4d-c787-591e-868e-b0548f93a32f
# The Geant4 detector: a slab of ceramic `depth` deep, cut into 13 × 13 dexels by reflector
# septa (ample for the x rays that leave a dexel), optionally under an anti-scatter grid
function dexel_array(depth; material = gos, grid = nothing)
    septa = MC.Septa(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        width_x_mm = septum_mm,
        width_y_mm = septum_mm,
        material = reflector,
        cells = 13,
    )
    return MC.SlabDetector(
        material,
        depth;
        width_mm = 13 * maximum(pitch_mm) + 1,
        septa,
        collimator = grid,
    )
end

# ╔═╡ f18449da-5fb9-5dd1-8889-b6f1fd902f59
# Photons landing uniformly on one dexel's ceramic, and on its whole cell (ceramic and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ 076c1cfb-65a3-5d2e-b5b0-1d264e756fe8
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 77ad16fd-5153-5a4b-801a-57c22ebbd8b7
# Each dexel collects the energy left in its own block of ceramic (the reflector keeps the light
# in); the dexel the photons land on and its eight neighbours are recorded
readout = MC.DexelEnergyReadout(
    MC.DexelReadout(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        neighborhood = 3,
        cloud_sigma_um = 0.01,
    ),
)

# ╔═╡ 5c906809-30a7-5e8f-b82e-78981da811f3
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1600, 930))


    # (a) the Geant4 model: 13 × 13 dexels, the middle one and its eight neighbours read out
    ax = canvas(fig[1, 1]; title = "(a) What Geant4 Builds: 13 × 13 Dexels, to Scale", titlesize = 20)
    cam = camera(-50, 30)
    blocks = dexel_blocks(13, 13, p, s, 0.0, d; lit = Set([(6, 6)]))
    beam = Block((6p[1] + s / 2, 6p[2] + s / 2, d), (7p[1] - s / 2, 7p[2] - s / 2, d + 5.0), MATERIALS.beam; alpha = 0.3, stroke = 0.0)
    draw_blocks!(ax, cam, blocks; extras = [(1.0e3, a -> draw_blocks!(a, cam, [beam]; edge_alpha = 0.0))], edge_alpha = 0.25)
    outline!(ax, cam, (5p[1], 5p[2], 0), (8p[1], 8p[2], d); color = PALETTE.photon, linewidth = 2.5, linestyle = :dash)
    callout!(ax, project(cam, (6.5p[1], 6.5p[2], d + 5.0)), project(cam, (6.5p[1], 6.5p[2], d + 5.0)) + Mke.Point2f(2.6, 0.4), "Photons Land Uniformly\non the Middle Dexel"; fontsize = 15)
    callout!(ax, project(cam, (8p[1], 5p[2], 0)), project(cam, (8p[1], 5p[2], 0)) + Mke.Point2f(2.2, -1.6), "Read Out: the Middle\nDexel and Its 8\nNeighbours (Dashed)"; fontsize = 15)

    # (b) seen from the beam, 3 × 3, with dimensions
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Seen from the Beam",
        xlabel = "Across the Fan (mm)",
        ylabel = "Along Z (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 2.3p[1]), (-2.3p[2], 1.9p[2])),
    )
    rect!(axb, -1.5p[1], -1.5p[2], 3p[1], 3p[2], Mke.RGBf(0.93, 0.93, 0.9); strokecolor = (:black, 0.25), strokewidth = 1)
    for i in -1:1, j in -1:1
        rect!(axb, i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, p[1] - s, p[2] - s, i == 0 && j == 0 ? Mke.RGBf(0.82, 0.95, 0.84) : MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1)
    end
    Mke.bracket!(axb, -0.5p[1], -1.6p[2], 0.5p[1], -1.6p[2]; text = "Pitch $(round(p[1]; digits = 3)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.bracket!(axb, 1.6p[1], -0.5p[2], 1.6p[1], 0.5p[2]; text = "$(round(p[2]; digits = 3)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    note!(axb, 0, 0, "One Dexel"; align = (:center, :center), color = Mke.RGBf(0.15, 0.45, 0.2), fontsize = 15)
    note!(axb, 1.5p[1], 1.5p[2], "White Reflector, $(round(Int, 1000s)) µm"; align = (:right, :bottom), offset = (0, 6), fontsize = 14, color = PALETTE.muted)

    # (c) side view with the depth
    axc = Mke.Axis(
        fig[1, 3];
        title = "(c) Side View",
        xlabel = "Across the Fan (mm)",
        ylabel = "Depth (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-1.6p[1], 2.0p[1]), (-d - 0.3, 0.7)),
        yticks = ([-d, 0], ["$(d)", "0"]),
    )
    side_dexels!(axc; lit = 1)
    Mke.lines!(axc, [0.1, 0.1], [0.7, -0.5]; color = MATERIALS.beam, linewidth = 3)
    Mke.bracket!(axc, 1.6p[1], -d, 1.6p[1], 0; text = "$(d) mm", orientation = :up, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.38))
    locator!(fig[0, :], :ceramic; note = "Steps 2–3 · The Ceramic,\nOne Dexel and Its\nNeighbours")
    fig
end

# ╔═╡ 4121046c-46f1-52bc-b36f-5e41408abb40
md"""
### What a Photon Does in the Ceramic

`record_tracks` follows a few hundred 60 keV photons landing on the central dexel, every step of
every particle. Two of them are drawn: one whose Gd x ray is absorbed **in the next dexel**, so
its energy is split between two photodiodes, and one whose Gd x ray **leaves through the front
face**, so its 43 keV are lost to every dexel. Both happen often, because the Gd x ray is so
weakly absorbed.
"""

# ╔═╡ 4ee27e1b-2c0f-5ba4-baaa-6e883057b012
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    60.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ ac571320-09f8-5d9b-843c-50d2bef6fc49
# Two photons to follow: one whose Gd x ray stops in the next dexel across the fan (in the same
# row, so the side view shows it), one whose x ray leaves through the front face without stopping
followed = let
    in_ceramic(z) = 0 <= z <= depth_mm
    xray_steps(event) = [s for s in tracks.steps if s.event == event && s.particle === :fluorescence]
    events = unique(s.event for s in tracks.steps if s.particle === :fluorescence)

    across(s) = abs(s.stop[1]) > pitch_mm[1] / 2 && abs(s.stop[2]) < crystal_mm[2] / 2
    next_door = [
        e for e in events
            if any(s.process == "phot" && in_ceramic(s.stop[3]) && across(s) for s in xray_steps(e))
    ]
    front = [
        e for e in events
            if !any(s.process == "phot" for s in xray_steps(e)) && any(s.stop[3] < -0.01 for s in xray_steps(e))
    ]
    (next_door = first(next_door), front = first(front))
end

# ╔═╡ e71c343e-4038-52da-8770-c462750ba0bc
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    top = -0.9
    fig = Mke.Figure(size = (1300, 520))

    # Clip a step to the drawing (above the ceramic it may run off to the world's edge)
    function clipped(a, b)
        a[3] >= top && b[3] >= top && return (a, b)
        t = (top - a[3]) / (b[3] - a[3])
        cut = a .+ t .* (b .- a)
        return a[3] < top ? (cut, b) : (a, cut)
    end

    function panel!(ax, event)
        Mke.poly!(ax, Mke.Rect2f(-1.5p[1], 0, 3p[1], d); color = PALETTE.reflector)
        for i in -1:1
            Mke.poly!(
                ax,
                Mke.Rect2f(i * p[1] - (p[1] - s) / 2, 0, p[1] - s, d);
                color = PALETTE.ceramic,
                strokecolor = PALETTE.slab_edge,
            )
        end
        kept = Dict(-1 => 0.0, 0 => 0.0, 1 => 0.0)
        for st in tracks.steps
            st.event == event || continue
            if st.edep_keV > 0 && 0 <= st.stop[3] <= d && abs(st.stop[2]) < p[2] / 2
                i = round(Int, st.stop[1] / p[1])
                haskey(kept, i) && (kept[i] += st.edep_keV)
            end
            st.particle in (:primary, :fluorescence) || continue
            a, b = clipped(st.start, st.stop)
            Mke.lines!(
                ax,
                [a[1], b[1]],
                [a[3], b[3]];
                color = st.particle === :primary ? PALETTE.photon : PALETTE.xray,
                linewidth = 3,
            )
            if st.process == "phot" && 0 <= st.stop[3] <= d
                Mke.scatter!(
                    ax,
                    [st.stop[1]],
                    [st.stop[3]];
                    color = PALETTE.electron,
                    markersize = 13,
                )
            end
        end
        for (i, e) in kept
            e > 0.5 || continue
            note!(
                ax,
                i * p[1],
                d + 0.12,
                "$(round(e; digits = 1)) keV";
                align = (:center, :top),
                color = PALETTE.kept,
                fontsize = 15,
            )
        end
        return nothing
    end

    common = (
        xlabel = "Across the Fan (mm)",
        ylabel = "Depth (mm)",
        yreversed = true,
        aspect = Mke.DataAspect(),
        limits = ((-1.55p[1], 1.55p[1]), (top, d + 0.55)),
    )
    axa = Mke.Axis(fig[1, 1]; title = "(a) The Gd X Ray Stops in the Next Dexel", common...)
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Gd X Ray Leaves Through the Front",
        common...,
    )
    panel!(axa, followed.next_door)
    panel!(axb, followed.front)
    Mke.Label(
        fig[2, 1:2],
        "Blue: the 60 keV Photon     Orange: Its Gd X Ray     Red Dots: Where They Are Absorbed     Green: the Energy Each Dexel Keeps";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 23479c79-f281-4863-8e11-70e01667ce3e
md"""
### Why Don't the Grid's Plates Go Down Through the Ceramic?

Two things can leak from one dexel into the next, and they are stopped (or not) by different parts:

- **Light.** The ceramic glows where the x ray stopped; light reaching the next dexel's photodiode
  would blur the image (*optical crosstalk*). The **white TiO₂ reflector** in the saw cuts stops
  it: light bounces back into its own dexel. Geant4 does not follow the light; the reflector is
  assumed perfect, which is what it is for.
- **X rays.** A Gd x ray (43 keV) or a Compton-scattered photon crosses 0.1 mm of light epoxy
  almost freely and can stop in the neighbour (*x-ray crosstalk*, the cousin of charge sharing in
  photon-counting detectors). That is the "next door" photon above, and Geant4 follows it.

Tungsten plates through the ceramic would stop the x rays, but they would widen the cuts (less
ceramic per dexel, so fewer photons stopped for the same dose), add tungsten's own x rays (59 and
67 keV) inside the ceramic, and have to be built into a sawn ceramic wafer. And in a scan the
x-ray crosstalk mostly cancels: a dexel gets back from its neighbours what it loses to them
(section 4). What it does cost, a little sharpness and a little noise (the Swank factor), is
measured here. So the grid stands in front, against the scatter from the patient, and only the
reflector goes between the dexels. (Canon documents neither the grid nor the reflector; both
are assumed here, the grid as 1-D plates between the channels, section 6.)
"""

# ╔═╡ 18dfc26f-97bd-4329-81f0-1e88ed75aaf9
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1600, 600))
    function base!(ax; septum_color = MATERIALS.reflector, septum = s, walls_through = false)
        rect!(ax, -1.5p, -d, 3p, d, septum_color; strokecolor = (:black, 0.3), strokewidth = 1)
        for i in -1:1
            rect!(ax, i * p - (p - septum) / 2, -d, p - septum, d, i == 0 ? Mke.RGBf(0.82, 0.95, 0.84) : MATERIALS.ufc)
        end
        for i in -1:2
            x = (i - 0.5) * p
            rect!(ax, x - grid.wall_mm / 2, 0, grid.wall_mm, 1.0, MATERIALS.tungsten)
            walls_through && rect!(ax, x - septum / 2, -d, septum, d, MATERIALS.tungsten)
        end
        for i in -1:1
            rect!(ax, i * p - (p - septum) / 2, -d - 0.16, p - septum, 0.12, MATERIALS.photodiode)
        end
        return
    end
    hit = Mke.Point2f(0.18, -0.5)
    star!(ax) = Mke.scatter!(ax, [hit]; marker = :star5, markersize = 24, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)

    ax1 = canvas(fig[1, 1]; title = "As Modelled: Grid Plates Above, White Reflector in the Saw Cuts", titlesize = 20)
    base!(ax1)
    photon!(ax1, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    half = (p - s) / 2
    Mke.lines!(ax1, Mke.Point2f[hit, (half, -0.15), (0.3, -d)]; color = MATERIALS.light, linewidth = 2.5)
    Mke.lines!(ax1, Mke.Point2f[hit, (-half, -0.3), (-0.25, -d)]; color = MATERIALS.light, linewidth = 2.5)
    photon!(ax1, hit, (1.15p, -1.0); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 11)
    star!(ax1)
    Mke.Label(fig[2, 1], "Light (Green): Bounces Off the Reflector, Stays in Its Dexel\nGd X Ray (Orange, 43 keV): Passes Through the Thin, Light Reflector"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax1, -1.65p, 1.65p, -d - 0.3, 1.75)

    ax2 = canvas(fig[1, 2]; title = "Hypothetical: Tungsten Plates Down Through the Ceramic", titlesize = 20)
    base!(ax2; septum = 0.16, walls_through = true)
    photon!(ax2, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    photon!(ax2, hit, (half + 0.02, -0.82); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 11)
    Mke.scatter!(ax2, [Mke.Point2f(half + 0.06, -0.85)]; marker = :xcross, markersize = 18, color = PALETTE.xray)
    star!(ax2)
    Mke.Label(fig[2, 2], "Stops the Gd X Ray, but: Wider Cuts Leave Less Ceramic per Dexel,\nTungsten Adds Its Own X Rays (59 and 67 keV), and It Must Be Built In"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax2, -1.65p, 1.65p, -d - 0.3, 1.75)
    fig
end

# ╔═╡ 2baedffc-48d3-58a5-9816-c781f3598720
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is not one number but a
spread: the full photon energy when everything stays in the dexel, the energy minus 43 keV when
the Gd x ray leaves, a little when the photon only Compton-scatters once and leaves. Three
energies show the three regimes: 45 keV (below the K-edge), 60 keV (just above it) and 120 keV.
"""

# ╔═╡ 767031ad-4240-58a7-8784-73ed6193049b
# 100 000 photons at each energy, every energy deposit recorded
signal_runs = Dict(
    E => MC.simulate_monoenergetic(
            dexel_array(depth_mm),
            E;
            n = 100_000,
            beam = crystal_beam,
            seed = 5,
            record_deposits = true,
        )
        for E in (45.0, 60.0, 120.0)
)

# ╔═╡ ac0ae21f-6ba3-5b81-b801-b226d34a8645
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ 2c56d6b4-e760-4797-9f7e-222b1a3fdffe
let
    Mke.set_theme!(THEME)
    p, d = pitch_mm[1], depth_mm
    fig = Mke.Figure(size = (1600, 810))

    kα = 43.0
    cases = (
        ("Everything Stays", "The Gd X Ray Is Reabsorbed\nin the Same Dexel → a Peak at E", PALETTE.kept),
        ("The Gd X Ray Escapes", "43 keV Leave Through the Front\n→ a Peak at E − 43 keV", PALETTE.xray),
        ("It Lands Next Door", "The Energy Is Split Between\nTwo Dexels (Both See Part)", Mke.RGBf(0.45, 0.3, 0.6)),
        ("A Compton Photon Leaves", "Only the Electron's Energy\nStays → the Low Tail", PALETTE.photon),
    )
    for (k, (title, detail, color)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlecolor = color, titlesize = 20)
        side_dexels!(ax; lit = 1)
        hit = Mke.Point2f(0.05, -0.45)
        photon!(ax, (0.05, 1.3), hit; cycles = 5, amplitude = 0.03, color = MATERIALS.beam)
        Mke.scatter!(ax, [hit]; marker = :star5, markersize = 22, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)
        if k == 1
            photon!(ax, hit, (0.33, -1.05); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 10)
            Mke.scatter!(ax, [Mke.Point2f(0.33, -1.05)]; color = PALETTE.electron, markersize = 10)
        elseif k == 2
            photon!(ax, hit, (-0.55, 1.2); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 12)
        elseif k == 3
            photon!(ax, hit, (0.85, -0.95); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 10)
            Mke.scatter!(ax, [Mke.Point2f(0.98, -0.95)]; color = PALETTE.electron, markersize = 10)
        else
            photon!(ax, hit, (1.6, 0.6); cycles = 3, amplitude = 0.05, color = PALETTE.photon, head = 12)
        end
        Mke.limits!(ax, -1.75, 1.75, -d - 0.25, 1.45)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :ceramic; note = "Step 2 · Where One\nPhoton's Energy Goes")
    fig
end

# ╔═╡ 3734b13a-da38-504d-bb4a-ef79fccc86b9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 520))
    kα = gadolinium.kalpha_keV
    for (k, E) in enumerate((45.0, 60.0, 120.0))
        h = histogram(own_signal[E], E + 4)
        ax = Mke.Axis(
            fig[1, k];
            title = "$(Int(E)) keV Photons",
            xlabel = "Energy the Dexel Collects (keV)",
            ylabel = k == 1 ? "Probability per keV" : "",
            yscale = log10,
            limits = ((0, E + 4), (1.0e-5, 2)),
        )
        Mke.lines!(
            ax,
            h.centres,
            max.(h.density, 1.0e-6);
            color = PALETTE.kept,
            linewidth = 2.5,
        )
        note!(
            ax,
            E,
            0.9,
            "All of It";
            align = (:right, :center),
            offset = (-6, 0),
            color = PALETTE.kept,
            fontsize = 14,
        )
        if E > gadolinium.edge_keV
            note!(
                ax,
                E - kα,
                0.3,
                "Gd X Ray Lost:\nE − $(round(kα; digits = 1)) keV";
                align = (:left, :center),
                offset = (12, 0),
                color = PALETTE.xray,
                fontsize = 14,
            )
        end
        if E > 100
            note!(
                ax,
                25,
                0.02,
                "Compton Scatters\nThat Left";
                align = (:center, :center),
                color = PALETTE.photon,
                fontsize = 14,
            )
        end
    end
    fig
end

# ╔═╡ 4456dbc6-d2ac-5b8b-b744-cd7109af51ea
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the ceramic or
leaves it: as the Gd x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of ceramic (no reflector), so that whatever
leaves the ceramic leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ b0e5a025-3a3b-5d9d-8ae0-d597c6507d3d
budget_energies = 15.0:5.0:140.0

# ╔═╡ d21e6491-d12e-52ae-b383-bd12d671b4e9
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(gos, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ ac57653b-ccdc-448b-b017-47ce1deb749a
let
    Mke.set_theme!(THEME)
    d = depth_mm
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1])
    rect!(ax, -6, -d, 12, d, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 0, -d - 0.12; text = "One Continuous Slab of the Ceramic, $(d) mm Deep (No Reflector, So What Leaves Is Gone)", fontsize = 15, color = PALETTE.slab_edge, align = (:center, :top))
    xs = (-4.5, -1.5, 1.5, 4.5)
    for x in xs
        photon!(ax, (x, 1.6), (x, x == 4.5 ? -d - 1.4 : -0.6); cycles = 5, amplitude = 0.012, color = MATERIALS.beam)
    end
    for x in xs[1:3]
        Mke.scatter!(ax, [Mke.Point2f(x, -0.6)]; marker = :star5, markersize = 24, color = Mke.RGBf(0.98, 0.95, 0.6), strokecolor = MATERIALS.beam, strokewidth = 1.2)
    end
    photon!(ax, (-1.5, -0.6), (-2.6, 1.3); cycles = 3, amplitude = 0.04, color = PALETTE.xray)
    photon!(ax, (1.5, -0.6), (2.7, -d - 1.3); cycles = 3, amplitude = 0.04, color = PALETTE.photon)
    labels = (
        (-4.5, "Stays in\nthe Ceramic", PALETTE.kept),
        (-1.5, "Leaves as\na Gd X Ray", PALETTE.xray),
        (1.5, "Leaves as a\nScattered Photon", PALETTE.photon),
        (4.5, "Passes Straight\nThrough", PALETTE.muted),
    )
    for (x, text, color) in labels
        Mke.text!(ax, x, 1.75; text, color, fontsize = 18, font = :bold, align = (:center, :bottom))
    end
    Mke.limits!(ax, -6.3, 6.3, -d - 1.6, 2.75)
    fig
end

# ╔═╡ c012a993-2845-5b07-be0a-f019bf4fea80
let
    Mke.set_theme!(THEME)
    E = collect(budget_energies)
    parts = (
        (
            values = [b.absorbed for b in budgets],
            label = "Stays in the Ceramic (the Signal)",
            color = PALETTE.kept,
        ),
        (
            values = [b.fluorescence for b in budgets],
            label = "Leaves as a Gd X Ray",
            color = PALETTE.xray,
        ),
        (
            values = [b.scattered for b in budgets],
            label = "Leaves as a Scattered Photon",
            color = PALETTE.photon,
        ),
        (
            values = [b.transmitted for b in budgets],
            label = "Passes Straight Through",
            color = PALETTE.muted,
        ),
    )
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Where the Photon's Energy Goes: $(depth_mm) mm of the Ceramic",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of the Photon's Energy",
        xticks = 20:20:140,
        limits = ((15, 140), (0, 1)),
    )
    below = zeros(length(E))
    for part in parts
        above = below .+ part.values
        Mke.band!(ax, E, below, above; color = (part.color, 0.75), label = part.label)
        below = above
    end
    Mke.vlines!(
        ax,
        gadolinium.edge_keV;
        color = PALETTE.ink,
        linestyle = :dot,
        linewidth = 1.5,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ b2e83820-62f2-50f2-9aea-a57e382bfc14
let
    k = findfirst(>(gadolinium.edge_keV), collect(budget_energies))
    b = budgets[k]
    closure = maximum(abs(b.total - 1) for b in budgets)
    Markdown.parse(
        """
        Just above the K-edge, at $(Int(budget_energies[k])) keV, **$(round(100b.fluorescence; digits = 1)) %** of the energy leaves as
        Gd x rays and $(round(100b.absorbed; digits = 1)) % stays. The energy balance closes to within
        $(round(100closure; digits = 2)) % at every energy (the rest is binding energy below Geant4's tracking cuts).
        """
    )
end

# ╔═╡ cf122acc-fc8b-5d8d-ad7d-84a701424756
md"""
## 4. The Efficiency and Noise of One Dexel

The full simulation: at every whole keV from 1 to 140 (the grid of BasisSimulator's tables),
100 000 photons land uniformly on one dexel's ceramic, and each run is reduced, where it ran, to
the energy every dexel of the 3 × 3 collected and the energy left anywhere in the ceramic. From
them come four curves:

- **interaction probability**: the photon interacts at all (`1 − exp(−μd)`, the check);
- **one-dexel efficiency**: the energy the dexel the photon landed on keeps, ÷ E;
- **flat-field efficiency**: the energy left anywhere in the ceramic, ÷ E. In a scan every dexel
  is illuminated, and what one dexel loses to its neighbours it gets back from them, so this is
  what each dexel collects per photon landing on it: **the efficiency a CT detector has**;
- **Swank factor** `I = M₁² / (M₀ M₂)` of the flat-field signal: the noise the spread of
  collected energies adds. A detector's zero-frequency DQE is `M₁² / M₂` per incident photon, the
  interaction probability times `I`.
"""

# ╔═╡ e4c398e8-b01d-565f-81ed-8a19750e5e33
energies = 1.0:1.0:140.0

# ╔═╡ 98a5f0ad-bbe1-5aab-8dc8-4e2aecbe102c
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 76f1fe13-108c-5f56-aeb2-5b626860523d
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ 9f728df7-dc6b-4ab6-8e01-6375174c9ccd
let
    Mke.set_theme!(THEME)
    p, s = pitch_mm, septum_mm
    fig = Mke.Figure(size = (1600, 850))

    lit_colour = Mke.RGBf(0.82, 0.95, 0.84)
    function top!(ax, lit)
        rect!(ax, -2.5p[1], -2.5p[2], 5p[1], 5p[2], Mke.RGBf(0.93, 0.93, 0.9))
        for i in -2:2, j in -2:2
            rect!(ax, i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, p[1] - s, p[2] - s, lit(i, j) ? lit_colour : MATERIALS.ufc; strokecolor = (PALETTE.slab_edge, 0.5), strokewidth = 1)
        end
        return
    end
    ax1 = canvas(fig[1, 1]; title = "One Dexel Lit", titlesize = 20)
    top!(ax1, (i, j) -> i == 0 && j == 0)
    for (dx, dy) in ((1, 0), (0, 1), (-1, 0), (0, -1))
        photon!(ax1, (0.15dx * p[1], 0.15dy * p[2]), (0.95dx * p[1], 0.95dy * p[2]); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 11)
    end
    Mke.Label(fig[2, 1], "Its Gd X Rays Land in the Neighbours:\nLost to This Dexel (the Dashed Curve Below)"; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    ax2 = canvas(fig[1, 2]; title = "Every Dexel Lit (Flat Field)", titlesize = 20)
    top!(ax2, (i, j) -> true)
    for (dx, dy) in ((1, 0), (0, 1), (-1, 0), (0, -1))
        photon!(ax2, (0.15dx * p[1], 0.15dy * p[2] + 0.12), (0.95dx * p[1], 0.95dy * p[2] + 0.12); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 10)
        photon!(ax2, (0.95dx * p[1], 0.95dy * p[2] - 0.12), (0.15dx * p[1], 0.15dy * p[2] - 0.12); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 10)
    end
    Mke.Label(fig[2, 2], "What It Loses, It Gets Back from Them:\nWhat a Scan Sees (the Green Curve)"; fontsize = 16, color = PALETTE.muted, tellwidth = false)

    # Swank: the same photon energy, different signals
    ax3 = Mke.Axis(
        fig[1, 3];
        title = "Same Energy, Different Signals",
        titlesize = 20,
        xlabel = "Ten 60 keV Photons",
        ylabel = "Energy Collected (keV)",
        xticks = 1:10,
        limits = ((0.3, 10.7), (0, 66)),
    )
    signals = [60, 60, 17, 60, 60, 60, 17, 60, 9, 60]
    Mke.barplot!(ax3, 1:10, signals; color = [v == 60 ? PALETTE.kept : v == 17 ? PALETTE.xray : PALETTE.photon for v in signals])
    Mke.Label(fig[2, 3], "The Spread Adds Noise: the Swank Factor\nI = 1 When Every Photon Gives the Same"; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    locator!(fig[0, :], :ceramic; note = "Steps 2–3 · The Whole\nCeramic Layer, Many\nDexels at Once")
    fig
end

# ╔═╡ 89682fb9-9156-5081-9815-65ae00e73940
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1400, 600))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Efficiency, $(depth_mm) mm of the Ceramic",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        axa,
        E,
        efficiency.interaction;
        color = PALETTE.ink,
        linestyle = :dot,
        linewidth = 2.5,
        label = "Photon Interacts at All",
    )
    Mke.lines!(
        axa,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3.5,
        label = "Flat Field (What a Scan Sees)",
    )
    Mke.lines!(
        axa,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2.5,
        linestyle = :dash,
        label = "Kept by the Dexel It Landed On",
    )
    note!(
        axa,
        gadolinium.edge_keV,
        0.75,
        "Gd K-Edge";
        align = (:right, :center),
        offset = (-8, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.vlines!(axa, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axa; position = :lb, margin = (10, 10, 40, 10))

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) Noise: the Swank Factor and DQE(0)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Swank Factor I,  DQE(0)",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        axb,
        E,
        efficiency.swank;
        color = PALETTE.electron,
        linewidth = 3,
        label = "Swank Factor I (Spread of the Signal)",
    )
    Mke.lines!(
        axb,
        E,
        efficiency.dqe;
        color = PALETTE.photon,
        linewidth = 3,
        label = "DQE(0) = Interaction × I",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :lb)
    fig
end

# ╔═╡ ff5bcd43-c401-569f-bb51-3b3163ebcde3
let
    at(E) = findfirst(==(E), collect(energies))
    k60, k140 = at(60.0), at(140.0)
    beer = 1 - exp(-MC.geant4_attenuation(gos, 140.0) * depth_mm)
    Markdown.parse(
        """
        At 60 keV, just above the K-edge, the dexel a photon lands on keeps **$(round(efficiency.one_dexel[k60]; digits = 3))**
        of its energy; with every dexel illuminated each collects **$(round(efficiency.flat_field[k60]; digits = 3))**: the Gd
        x rays a dexel loses to its neighbours, it gets back from them. The Swank factor there is
        $(round(efficiency.swank[k60]; digits = 3)): the lost x rays also make the signal noisier. At 140 keV the photon interacts
        with probability $(round(efficiency.interaction[k140]; digits = 3)) (Beer–Lambert: $(round(beer; digits = 3))), and the flat-field
        efficiency is $(round(efficiency.flat_field[k140]; digits = 3)).
        """
    )
end

# ╔═╡ b67523ad-a398-5025-a89e-31bbc644a886
md"""
### A Check Against Published Monte Carlo

Boone et al. (Med. Phys. 1999) followed x-ray fluorescence in thin Gd₂O₂S screens with the TART 98
code: a pencil beam at normal incidence on a screen of Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S, the binder not counted), the energy tallied in a circle 0.1 mm across the
beam ("absorbed as primary") and in rings out to 4.5 mm ("lateral"), and ideal sensors in front of
the screen and behind it. At 55 keV they report (p. 908) how the incident energy divides. The same
screen and the same tallies in Geant4:
"""

# ╔═╡ b807babb-788e-50b5-9857-ded2f7dfff09
boone = let
    # Gd₂O₂S with 5 % binder by weight (Carboset, an acrylic, taken as C₅H₈O₂)
    fractions = Dict(
        "Gd" => 0.95 * 0.83083,
        "O" => 0.95 * 0.08453 + 0.05 * 0.3196,
        "S" => 0.95 * 0.08469,
        "C" => 0.05 * 0.5998,
        "H" => 0.05 * 0.0805,
    )
    material = MC.DetectorMaterial(
        "Gd2O2S screen with binder";
        density = 7.34,
        mass_fractions = fractions,
    )
    screen = MC.SlabDetector(
        material,
        (0.09 / 0.95) / 7.34 * 10;
        width_mm = 30.0,
        shape = :disc,
    )

    n = 400_000
    run = MC.simulate_monoenergetic(
        screen,
        55.0;
        n,
        beam = MC.PencilBeam(),
        seed = 11,
        record_deposits = true,
    )
    incident = n * 55.0
    dep, esc = run.deposits, run.escapes
    radius(k) = hypot(dep.x_mm[k], dep.y_mm[k])
    (
        primary = sum(dep.energy_keV[k] for k in eachindex(dep.event) if radius(k) <= 0.1) / incident,
        lateral = sum(dep.energy_keV[k] for k in eachindex(dep.event) if 0.1 < radius(k) <= 4.5) / incident,
        backward = sum(esc.energy_keV[i] for i in eachindex(esc.kind) if esc.backward[i]) / incident,
        forward = sum(
            esc.energy_keV[i] for i in eachindex(esc.kind) if !esc.backward[i] && esc.kind[i] !== :transmitted
        ) / incident,
        transmitted = sum(esc.energy_keV[i] for i in eachindex(esc.kind) if esc.kind[i] === :transmitted) / incident,
    )
end

# ╔═╡ dfb875ed-6d51-485a-b1ad-fd2525cc623b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1]; title = "Side View (Not to Scale)", titlesize = 20)
    rect!(ax, -3, -0.12, 6, 0.24, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 3.05, 0; text = "Gd₂O₂S Screen,\n90 mg/cm²", align = (:left, :center), fontsize = 15, color = PALETTE.slab_edge)
    Mke.lines!(ax, [-3, 3], [1.1, 1.1]; color = PALETTE.photon, linewidth = 3, linestyle = :dash)
    Mke.lines!(ax, [-3, 3], [-1.1, -1.1]; color = PALETTE.photon, linewidth = 3, linestyle = :dash)
    Mke.text!(ax, -3, 1.2; text = "Ideal Sensor: Energy Carried Back Toward the Tube", fontsize = 15, color = PALETTE.photon, align = (:left, :bottom))
    Mke.text!(ax, -3, -1.2; text = "Ideal Sensor: Energy Carried Forward", fontsize = 15, color = PALETTE.photon, align = (:left, :top))
    photon!(ax, (0, 2.0), (0, 0.0); cycles = 6, amplitude = 0.03, color = MATERIALS.beam)
    Mke.text!(ax, 0.1, 1.75; text = "55 keV Pencil Beam", fontsize = 15, color = Mke.RGBf(0.8, 0.5, 0.05), align = (:left, :center))
    photon!(ax, (0, 0), (-1.3, 1.05); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 10)
    photon!(ax, (0, 0), (1.4, -1.05); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 10)
    photon!(ax, (0, 0), (0.9, 0.0); cycles = 2, amplitude = 0.08, color = PALETTE.xray, head = 10)
    Mke.limits!(ax, -3.2, 4.6, -1.7, 2.3)

    ax2 = canvas(fig[1, 2]; title = "Seen from the Beam: the Tallies", titlesize = 20)
    θ = range(0, 2π; length = 120)
    Mke.poly!(ax2, Mke.Point2f.(4.5 .* cos.(θ), 4.5 .* sin.(θ)); color = (MATERIALS.ufc, 0.6), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.poly!(ax2, Mke.Point2f.(0.45 .* cos.(θ), 0.45 .* sin.(θ)); color = PALETTE.kept)
    Mke.text!(ax2, 0, 0.6; text = "Within 0.1 mm:\n\"Absorbed as Primary\"", align = (:center, :bottom), fontsize = 15, color = PALETTE.kept)
    Mke.text!(ax2, 0, -3.3; text = "0.1 to 4.5 mm: \"Lateral\"\n(Reabsorbed Fluorescence)", align = (:center, :center), fontsize = 15, color = PALETTE.slab_edge)
    Mke.text!(ax2, 4.6, 3.6; text = "(Circle Enlarged)", align = (:left, :center), fontsize = 13, color = PALETTE.muted)
    Mke.limits!(ax2, -5, 7, -5, 5)
    fig
end

# ╔═╡ 8a97e96d-38c0-52f0-a4b4-9c835b6413c8
let
    pct(x) = round(100x; digits = 1)
    rows = [
        ("absorbed within 0.1 mm of the beam (\"primary\")", 27.9, boone.primary),
        ("reabsorbed 0.1–4.5 mm away (\"lateral\")", 7.6, boone.lateral),
        ("carried back toward the tube", 16.5, boone.backward),
        ("carried forward as secondary radiation", 14.8, boone.forward),
        (
            "transmitted primary (Boone: the rest)",
            round(100 - 27.9 - 7.6 - 16.5 - 14.8; digits = 1),
            boone.transmitted,
        ),
    ]
    escaped = boone.backward + boone.forward
    Markdown.parse(
        """
        | 55 keV, 90 mg/cm² Gd₂O₂S screen | Boone et al. 1999 (TART 98) | Geant4 |
        |---|---|---|
        $(join(["| $(r[1]) | $(r[2]) % | $(pct(r[3])) % |" for r in rows], "\n"))
        | **escaped, back and forward** | **31.3 %** | **$(pct(escaped)) %** |

        The photons that pass straight through agree, so the screen and its attenuation are the same, and so does the
        split of what escapes between back and forward ($(round(boone.backward / boone.forward; digits = 2)) against 1.11). But Geant4 lets
        $(round(31.3 - 100escaped; digits = 1)) points less escape and keeps $(round(100boone.primary - 27.9; digits = 1)) more within 0.1 mm of the beam: **⚠ TART 98 and Geant4 differ by
        about $(round(Int, 100 * (1 - 100escaped / 31.3))) % in the energy the screen re-emits.** The same Geant4 setup reproduces EGSnrc's Swank factors
        for Gd₂O₂S to a few thousandths (Hajdok et al., notebook 02), and those depend directly on how much fluorescence
        escapes; the difference most likely lies in the 1999 code's atomic data rather than in this screen, but that is an
        inference, not a demonstration.
        """
    )
end

# ╔═╡ f4b79fc1-aa23-56f6-8bc1-eb2b3db7b47e
md"""
## 5. Compared with BasisSimulator v0.15.1's Nearest Table

BasisSimulator v0.15.1 has no Canon detector. The nearest table it has is its SOMATOM Definition
Flash table, `UFC_FLASH_MC_EFFICIENCY_LUT`: Gd₂O₂S, made (notebook 05 reproduces it) as a pencil beam
into the centre of one dexel 1.4 mm deep at 7.44 g/cm³, tallied in that dexel. It is what a Canon
model in BasisSimulator v0.15.1 would have had to borrow. How it compares with the PUREViSION's flat
field:
"""

# ╔═╡ fdbc6f9a-497d-588a-b8b5-6d7bd3b5d4e7
# BasisSimulator v0.15.1's nearest table: the Definition Flash's Gd₂O₂S (MCNP, Hamidreza Khodajou-Chokami)
# BasisSimulator v0.15.1's table, as BasisMonteCarlo keeps it for comparison
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash

# ╔═╡ afa80bb9-13fb-4c92-b215-16f2b8bf7ff4
# Is BasisMonteCarlo's copy the table the installed BasisSimulator has? Up to v0.15.1 it must be,
# exactly; from v0.16.0 BasisSimulator uses the Geant4 tables, and the copy stands alone
let
    installed = pkgversion(BS)
    same = isdefined(BS, :UFC_FLASH_MC_EFFICIENCY_LUT) &&
        BS.UFC_FLASH_MC_EFFICIENCY_LUT.energies == table.energies &&
        BS.UFC_FLASH_MC_EFFICIENCY_LUT.efficiency == table.efficiency
    message = if installed <= v"0.15.1"
        same ?
            "✓ The installed BasisSimulator, v$(installed), has exactly this table (its `UFC_FLASH_MC_EFFICIENCY_LUT`)." :
            "✗ **The installed BasisSimulator, v$(installed), has a different Flash table from BasisMonteCarlo's copy of v0.15.1's.** Check `BASISSIMULATOR_V0_15_1_EFFICIENCY` against BasisSimulator v0.15.1."
    else
        "The installed BasisSimulator is v$(installed), which no longer uses this table. The comparisons below are with BasisSimulator v0.15.1's Flash table, which BasisMonteCarlo keeps (`BASISSIMULATOR_V0_15_1_EFFICIENCY`): they do not change with BasisSimulator's version."
    end
    Markdown.parse(message)
end

# ╔═╡ 65264960-57e5-5281-b471-39fa9e34f538
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ 319f52f4-4c6d-42f3-931b-f6218dfd6a1e
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1500, 900))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        (
            "The Borrowed Table: the Definition Flash's",
            "A Pencil Beam into the Centre of One Dexel, Tallied\nin That Dexel Only; Gd₂O₂S at 7.44 g/cm³ (BasisSimulator v0.15.1)",
            :pencil,
        ),
        (
            "This Notebook: the PUREViSION's Flat Field",
            "Every Dexel Lit, as in a Scan; Gd₂O₂S Ceramic\nat 7.31 g/cm³, 1.4 mm Deep (Assumed)",
            :flat,
        ),
    )
    for (k, (title, detail, kind)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlesize = 21)
        lit = kind === :pencil ? Set([(1, 1)]) : Set((i, j) for i in 0:2 for j in 0:2)
        blocks = dexel_blocks(3, 3, p, s, 0.0, d; lit)
        extras = Any[]
        if kind === :pencil
            a, b = project(cam, (1.5p[1], 1.5p[2], d + 4.5)), project(cam, (1.5p[1], 1.5p[2], d))
            push!(extras, (1.0e3, ax -> arrow!(ax, a, b; color = amber, linewidth = 4, head = 14)))
        else
            beam = Block((0.0, 0.0, d), (3p[1], 3p[2], d + 4.0), amber; alpha = 0.28, stroke = 0.0)
            push!(extras, (1.0e3, ax -> draw_blocks!(ax, cam, [beam]; edge_alpha = 0.0)))
        end
        draw_blocks!(ax, cam, blocks; extras)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :ceramic; note = "Step 2 · Which Table\nDescribes the Ceramic?")
    fig
end

# ╔═╡ dac0f49a-6dcf-5351-843b-5bf3d2d603df
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "BasisSimulator v0.15.1's Nearest Table Against the PUREViSION at 1.4 mm",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 3.5,
        label = "BasisSimulator v0.15.1's Flash Table (MCNP, One Dexel, Pencil Beam)",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.flat_field;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, 1.4 mm, Flat Field",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 1.4 mm, One Dexel",
    )
    Mke.vlines!(ax, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 89ee2f6a-8d3d-5837-a72b-b16ac6dd781c
let
    at(E) = findfirst(==(E), collect(energies))
    rows = [
        "| $(Int(E)) keV | $(round(table_at(E); digits = 3)) | $(round(efficiency.flat_field[at(E)]; digits = 3)) |"
            for E in (40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
    ]
    Markdown.parse(
        """
        | photons of | BasisSimulator v0.15.1's Flash table | Geant4, PUREViSION flat field |
        |---|---|---|
        $(join(rows, "\n"))

        Just above the K-edge the flat field is higher (each dexel gets back the Gd x rays its neighbours lose); at high
        energy it is lower (7.31 g/cm³ rather than 7.44, and photons landing near a dexel's edge lose more). Borrowing
        the v0.15.1 Flash table would misplace the Canon detector's efficiency by several hundredths at the energies a
        scan uses most.
        """
    )
end

# ╔═╡ eab56392-a661-5fc6-8f95-068d32573a2b
md"""
## 6. The Anti-Scatter Grid

Canon publishes nothing about the Aquilion ONE's anti-scatter grid; Toshiba's patents describe both
one-dimensional collimators (plates between the channels) and two-dimensional ones, of tungsten or
molybdenum. Canon does not advertise a two-dimensional grid, as Siemens, GE and Philips do for
theirs, so a **one-dimensional** grid is assumed: **tungsten plates 0.1 mm thick and 10 mm tall**, one
per channel, standing on the reflector. The grid does not enter the efficiency table; it matters for
scatter.
"""

# ╔═╡ f2759008-f052-5b85-92fb-04c23dfa5842
grid = MC.Collimator(;
    height_mm = 10.0,
    wall_mm = 0.1,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :x,
    cells = 13,
)

# ╔═╡ 9dbcae9a-cb06-5a25-a722-9fb39b6e7114
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    H, w = grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1600, 1050))


    # (a) to scale, the near corner cut away
    ax = canvas(fig[1, 1]; title = "(a) The 1-D Grid's Plates over the Dexels, to Scale", titlesize = 20)
    cam = camera(-52, 24)
    nx, ny = 6, 5
    blocks = dexel_blocks(nx, ny, p, s, 0.0, d)
    append!(blocks, filter(b -> !(b.lo[1] > 2.4p[1] && b.lo[2] < 2.0p[2]), grid_blocks(nx, ny, p, w, d, d + H; walls = :x)))
    straight = ((4.5p[1], 0.5p[2], d + H + 3.5), (4.5p[1], 0.5p[2], d))
    tilt = tand(20)
    stop_z = d + H - (0.5p[1] - w / 2) / tilt
    angled = ((3.0p[1] + 0.02, 1.5p[2], d + H + 2.0), (4.0p[1] - w / 2, 1.5p[2], stop_z))
    # tilted 15° along z, between two plates: nothing in its way
    along_z = ((5.5p[1], 0.3p[2], d + H + 2.0), (5.5p[1], 0.3p[2] + (H + 2.0) * tand(15), d))
    extras = [
        (1.0e3, a -> arrow!(a, project(cam, straight[1]), project(cam, straight[2]); color = MATERIALS.beam, linewidth = 3.5)),
        (1.0e3, a -> Mke.lines!(a, [project(cam, angled[1]), project(cam, angled[2])]; color = PALETTE.xray, linewidth = 3.5)),
        (1.0e3, a -> Mke.scatter!(a, [project(cam, angled[2])]; marker = :xcross, markersize = 20, color = PALETTE.xray)),
        (1.0e3, a -> arrow!(a, project(cam, along_z[1]), project(cam, along_z[2]); color = PALETTE.electron, linewidth = 3.5)),
    ]
    draw_blocks!(ax, cam, blocks; extras)
    callout!(ax, project(cam, (4.5p[1], 0.5p[2], d + H + 2.5)), project(cam, (4.5p[1], 0.5p[2], d + H + 2.5)) + Mke.Point2f(3.0, 1.2), "From the Focal Spot:\nStraight Down a Channel"; color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 15)
    callout!(ax, project(cam, angled[2]), project(cam, angled[2]) + Mke.Point2f(3.6, -2.2), "Scattered Across the Fan:\nStopped by a Plate"; color = PALETTE.xray, fontsize = 15)
    callout!(
        ax,
        project(cam, (nx * p[1], ny * p[2], d + H)),
        project(cam, (nx * p[1], ny * p[2], d + H)) + Mke.Point2f(1.6, 1.2), "Tungsten Plates (Assumed)\n$(round(Int, 1000w)) µm Thick,\n$(round(Int, H)) mm Tall"; fontsize = 15
    )
    callout!(ax, project(cam, along_z[2]), project(cam, along_z[2]) + Mke.Point2f(4.2, -1.0), "Scattered Along Z:\nNo Plate in Its Way"; color = PALETTE.electron, fontsize = 15)
    Mke.xlims!(ax, nothing, project(cam, (nx * p[1], 0, 0))[1] + 9.5)

    # (b) side view across the fan, with the angle and the shadow a plate casts
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) One Channel, Side View (Width Stretched)",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Height Above the Ceramic (mm)",
        limits = ((-0.75p[1], 1.05p[1]), (-d - 0.2, H + 1.4)),
    )
    rect!(axb, -0.5p[1], -d, p[1], d, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
    for x in (-0.5p[1], 0.5p[1])
        rect!(axb, x - w / 2, 0, w, H, MATERIALS.tungsten)
    end
    θ = 4.0
    # photons at θ enter the opening evenly; those entering within H tan θ of the far wall hit it
    xl, xr = -0.5p[1] + w / 2, 0.5p[1] - w / 2
    reach = H * tand(θ)
    Mke.poly!(axb, Mke.Point2f[(xr - reach, H), (xr, H), (xr, 0)]; color = (PALETTE.xray, 0.2))
    for x_in in range(xl + 0.03, xr - 0.03; length = 7)
        x_out = x_in + reach
        if x_out <= xr
            Mke.lines!(axb, [x_in, x_out], [H, 0]; color = MATERIALS.beam, linewidth = 2)
        else
            y_hit = H - (xr - x_in) / tand(θ)
            Mke.lines!(axb, [x_in, xr], [H, y_hit]; color = PALETTE.xray, linewidth = 2)
            Mke.scatter!(axb, [xr], [y_hit]; marker = :xcross, markersize = 12, color = PALETTE.xray)
        end
    end
    Mke.bracket!(axb, xr - reach, H + 0.1, xr, H + 0.1; text = "H tan θ: These Hit the Plate", orientation = :up, fontsize = 14, color = PALETTE.xray, textcolor = PALETTE.xray)
    Mke.bracket!(axb, 0.5p[1] + 0.12, 0, 0.5p[1] + 0.12, H; text = "H = $(round(Int, H)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.Label(fig[2, 2], "Photons Arriving θ = $(round(Int, θ))° off the Channel's Axis, Across the Fan:\nThe Fraction 1 − H tan θ / Opening Reaches the Ceramic"; fontsize = 15, color = PALETTE.ink, tellwidth = false)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.32))
    locator!(fig[0, :], :grid; note = "Step 1 · The Grid's Plates\nIn Front of the Ceramic")
    fig
end

# ╔═╡ 66a50ae8-d09e-57d8-b69c-664f6b362651
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (ceramic and reflector), with and without the grid: the grid
should cost almost nothing, apart from its own tungsten x rays (59 and 67 keV) reaching the
ceramic above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ da786503-a2d6-5880-9361-079b494dabcc
grid_energies = 20.0:5.0:140.0

# ╔═╡ 9ad0c644-b539-5e83-a22e-35fc92391228
# Photons over one dexel's whole cell, straight down, without and with the grid
grid_runs = (
    without = MC.simulate_sweep(
        [(dexel_array(depth_mm), E) for E in grid_energies];
        n = 100_000,
        beam = cell_beam,
        seed = 17,
        readout,
    ),
    with = MC.simulate_sweep(
        [(dexel_array(depth_mm; grid), E) for E in grid_energies];
        n = 100_000,
        beam = cell_beam,
        seed = 17,
        readout,
    ),
)

# ╔═╡ f1873392-5be8-5055-9c30-fdbe75ee33ab
angles_deg = 0.0:1.0:20.0

# ╔═╡ a6d94647-e210-576c-9539-a0ae4159ad3f
# 60 keV photons arriving at an angle across the fan, over a whole cell, without and with the grid
angle_runs = let
    tilted(angle) = MC.AreaBeam(;
        width_mm = pitch_mm[1],
        height_mm = pitch_mm[2],
        angle_x_deg = angle,
        start_mm = 12.0,
    )
    (
        without = MC.simulate_sweep(
            [(dexel_array(depth_mm), 60.0, 21, tilted(a)) for a in angles_deg];
            n = 100_000,
            readout,
        ),
        with = MC.simulate_sweep(
            [(dexel_array(depth_mm; grid), 60.0, 21, tilted(a)) for a in angles_deg];
            n = 100_000,
            readout,
        ),
    )
end

# ╔═╡ 3aea3630-70c7-5d4f-ab87-489a1320cce4
let
    Mke.set_theme!(THEME)
    E = collect(grid_energies)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Straight from the Focal Spot",
        xlabel = "Photon Energy (keV)",
        ylabel = "Energy Collected per Photon on the Cell, ÷ E",
        xticks = 20:20:140,
        limits = ((20, 140), (0.4, 1.0)),
    )
    Mke.lines!(
        axa,
        E,
        MC.energy_absorption_efficiency.(grid_runs.without);
        color = PALETTE.muted,
        linewidth = 3,
        label = "No Grid",
    )
    Mke.lines!(
        axa,
        E,
        MC.energy_absorption_efficiency.(grid_runs.with);
        color = PALETTE.tungsten,
        linewidth = 2.5,
        linestyle = :dash,
        label = "With the 1-D Grid",
    )
    Mke.vlines!(axa, 69.5; color = (PALETTE.xray, 0.6), linestyle = :dot)
    note!(
        axa,
        69.5,
        0.45,
        "W K-Edge";
        align = (:left, :bottom),
        offset = (6, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.axislegend(axa; position = :rt)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) 60 keV Photons Arriving at an Angle Across the Fan",
        xlabel = "Angle Across the Fan (Degrees)",
        ylabel = "Energy Collected, Relative to\nStraight Down Without a Grid",
        limits = ((0, 20), (0, 1.35)),
    )
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    Mke.lines!(
        axb,
        collect(angles_deg),
        MC.energy_absorption_efficiency.(angle_runs.without) ./ reference;
        color = PALETTE.muted,
        linewidth = 3,
        label = "No Grid",
    )
    Mke.lines!(
        axb,
        collect(angles_deg),
        MC.energy_absorption_efficiency.(angle_runs.with) ./ reference;
        color = PALETTE.tungsten,
        linewidth = 3,
        label = "With the 1-D Grid",
    )
    opening = pitch_mm[1] - grid.wall_mm
    geometric = [max(0, 1 - grid.height_mm * tand(a) / opening) for a in angles_deg]
    Mke.lines!(
        axb,
        collect(angles_deg),
        geometric;
        color = PALETTE.xray,
        linewidth = 2,
        linestyle = :dash,
        label = "Geometry Alone: 1 − H tan θ / Opening",
    )
    Mke.axislegend(axb; position = :rc)
    fig
end

# ╔═╡ d8ad9d29-97f4-5325-8ec6-68943312f265
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) % (the walls stand on the reflector; above 69.5 keV the walls' own x rays add a little).
        At an angle across the fan it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %. A
        photon that scatters in the patient arrives from anywhere, so most of it is stopped.
        """
    )
end

# ╔═╡ e259dd7e-12c9-56b0-b77b-93c23dd38226
md"""
## 7. The Whole Detector

The whole detector: 896 channels × 320 rows on an arc of radius 1072 mm around the focal spot,
160 mm along z at the isocentre, every dexel a block of ceramic in reflector, under the plates of the
1-D grid, focused on the focal spot across the fan. An air scan fires photons uniformly from the
focal spot over the whole detector, and every dexel's signal is summed. Two questions: is every dexel
the same, and does the whole detector collect what the single dexel of sections 4 and 6 predicts?
"""

# ╔═╡ 0ffbdeda-bc29-5233-9b92-7f5d1f13573b
detector = MC.ArcDetector(;
    columns = aquilion.columns,
    rows = aquilion.rows,
    radius_mm = aquilion.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :across,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = gos,
)

# ╔═╡ 1c62ec1e-8fd3-52a8-b84f-5610a8257098
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ 53b69f21-a0cd-5ce4-bede-baa7b0519223
# 3.2 million photons per energy from the focal spot, in 16 independent chunks
air_scans = let
    chunks = MC.simulate_sweep(
        [(detector, E, chunk) for E in air_energies for chunk in 1:16];
        n = 200_000,
        beam = MC.focal_spot(detector),
        readout = MC.ArcEnergyReadout(),
    )
    [sum(chunks[(16(k - 1) + 1):(16k)]) for k in eachindex(air_energies)]
end

# ╔═╡ 20f94825-7cc1-4883-987f-c28460713360
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 990))

    ax = canvas(fig[1, 1]; title = "The Whole Detector, from the Focal Spot (Depth Exaggerated)", titlesize = 20)
    cam = camera(-62, 26)
    R = aquilion.source_to_detector_mm
    half = asin(aquilion.scan_field_mm / 2 / aquilion.source_to_isocenter_mm)
    half_z = aquilion.rows * aquilion.row_at_isocenter_mm * R / aquilion.source_to_isocenter_mm / 2
    amber = MATERIALS.beam
    items = arc_quads(cam, R, half, half_z; thickness = 25.0)
    # the beam: fan × cone, translucent, from the focal spot
    focal = (0.0, 0.0, 0.0)
    corners = [(R * sin(a), -R * cos(a), z) for (a, z) in ((-half, -half_z), (half, -half_z), (half, half_z), (-half, half_z))]
    push!(
        items, (
            1.0e4, a -> begin
                for (c1, c2) in zip(corners, circshift(corners, -1))
                    Mke.poly!(a, Mke.Point2f[project(cam, focal), project(cam, c1), project(cam, c2)]; color = (amber, 0.1))
                end
                Mke.lines!(a, [project(cam, c) for c in vcat([focal], corners[1:1])]; color = (amber, 0.7), linewidth = 1.5)
                for c in corners
                    Mke.lines!(a, [project(cam, focal), project(cam, c)]; color = (amber, 0.7), linewidth = 1.5)
                end
                Mke.scatter!(a, [project(cam, focal)]; color = PALETTE.ink, markersize = 14)
                iso = (0.0, -aquilion.source_to_isocenter_mm, 0.0)
                Mke.scatter!(a, [project(cam, iso)]; marker = :cross, markersize = 18, color = PALETTE.ink)
            end,
        )
    )
    sort!(items; by = first)
    for (_, f) in items
        f(ax)
    end
    iso = (0.0, -aquilion.source_to_isocenter_mm, 0.0)
    callout!(ax, project(cam, focal), project(cam, focal) + Mke.Point2f(150, 80), "Focal Spot"; fontsize = 16)
    callout!(ax, project(cam, iso), project(cam, iso) + Mke.Point2f(-260, 120), "Isocentre, $(round(Int, aquilion.source_to_isocenter_mm)) mm Away"; fontsize = 16, align = (:right, :center))
    edge = (R * sin(half), -R * cos(half), half_z)
    callout!(ax, project(cam, edge), project(cam, edge) + Mke.Point2f(120, 120), "$(aquilion.columns) Columns Across the Fan ($(round(2rad2deg(half); digits = 1))°)\n× $(aquilion.rows) Rows Along Z, at $(round(R; digits = 1)) mm"; fontsize = 16)
    low = (R * sin(-half / 2), -R * cos(-half / 2), -half_z)
    callout!(ax, project(cam, low), project(cam, low) + Mke.Point2f(-120, -160), "16 cm at the Isocentre: the Outer Rows\nSee the Beam at the Cone Angle"; fontsize = 16, align = (:right, :center))
    xs = [project(cam, c)[1] for c in vcat(corners, [focal])]
    ys = [project(cam, c)[2] for c in vcat(corners, [focal])]
    Mke.limits!(ax, minimum(xs) - 520, maximum(xs) + 620, minimum(ys) - 240, maximum(ys) + 200)
    locator!(fig[0, :], :arc; note = "The Whole Arc:\n896 × 320 Dexels, 16 cm")
    fig
end

# ╔═╡ 23ab1779-eac6-5e29-af6d-d465174c5654
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))

    # (b) The signal per photon arriving, across the fan, in groups of 40 columns
    axb = Mke.Axis(
        fig[1, 1];
        title = "(a) Energy Collected per Photon, Across the Fan",
        xlabel = "Column (Across the Fan)",
        ylabel = "Energy Collected per Photon\non the Cell, ÷ E",
        limits = ((1, aquilion.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:aquilion.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / aquilion.columns
        centres = [mean(g) for g in groups]
        per_group = [sum(scan.energy_keV[g, :]) / (length(g) * per_photon) for g in groups]
        Mke.scatterlines!(
            axb,
            centres,
            per_group;
            color = colors[k],
            markersize = 9,
            linewidth = 2,
            label = "$(Int(air_energies[k])) keV",
        )
        slab = MC.energy_absorption_efficiency(grid_runs.with[findfirst(==(air_energies[k]), collect(grid_energies))])
        Mke.hlines!(axb, slab; color = colors[k], linestyle = :dash, linewidth = 2)
    end
    note!(
        axb,
        aquilion.columns / 2,
        0.42,
        "Dashed: One Dexel's Cell, Section 6 (with the Grid)";
        align = (:center, :bottom),
        color = PALETTE.muted,
        fontsize = 14,
    )
    Mke.axislegend(
        axb;
        position = :rt,
        orientation = :horizontal,
        backgroundcolor = (:white, 0.85),
        framevisible = false,
    )

    # (c) The same along z, in groups of rows: the outer rows see the beam at the cone angle
    axc = Mke.Axis(
        fig[1, 2];
        title = "(b) Along Z",
        xlabel = "Row (Along Z)",
        limits = ((1, aquilion.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:aquilion.rows, max(1, aquilion.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / aquilion.rows
        Mke.scatterlines!(
            axc,
            [mean(g) for g in row_groups],
            [sum(scan.energy_keV[:, g]) / (length(g) * per_photon) for g in row_groups];
            color = colors[k],
            markersize = 9,
            linewidth = 2,
        )
    end
    Mke.hideydecorations!(axc; grid = false)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.3))
    fig
end

# ╔═╡ d887e3ee-97ca-53ea-8ecd-ee1a7b5b6ea0
let
    lines = String[]
    ratios = Float64[]
    for (k, scan) in enumerate(air_scans)
        whole = sum(scan.energy_keV) / scan.photon_keV
        slab = MC.energy_absorption_efficiency(grid_runs.with[findfirst(==(air_energies[k]), collect(grid_energies))])
        push!(ratios, whole / slab)
        push!(
            lines,
            "| $(Int(air_energies[k])) keV | $(round(whole; digits = 4)) | $(round(slab; digits = 4)) |",
        )
    end

    # Outer rows against central rows, at each energy; the largest difference is reported
    function edge_of(scan)
        per_row = vec(sum(scan.energy_keV; dims = 1))
        n = length(per_row)
        eighth = max(1, n ÷ 8)
        centre = mean(per_row[(n ÷ 2 - eighth + 1):(n ÷ 2 + eighth)])
        outer = mean(vcat(per_row[1:eighth], per_row[(end - eighth + 1):end]))
        return outer / centre - 1
    end
    edges = edge_of.(air_scans)
    k = argmax(abs.(edges))
    edge = edges[k]
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / aquilion.source_to_detector_mm))
    worst = maximum(abs.(ratios .- 1))

    why = edge > 0 ?
        "a photon that lands on the reflector between rows at an angle passes through it into the next ring of crystal, so the outer rows catch photons the reflector would stop face-on" :
        "a photon that goes deep crosses from its ring of crystal into the reflector between rows"
    verdict = worst <= 0.01 || abs(edge) < 0.01 ?
        """The whole detector collects what one dexel's cell does, at every energy, to within $(round(100worst; digits = 1)) %: in an air
        scan each dexel gets back from its neighbours what it loses to them, and no column or row differs from the others
        beyond the statistics of the photons that reached it. One dexel, simulated in detail, stands for them all.""" :
        """Across the fan the whole detector collects what one dexel's cell does; the difference, up to
        $(round(100worst; digits = 1)) %, is along z. At $(Int(air_energies[k])) keV the outer rows collect $(round(100abs(edge); digits = 1)) % $(edge > 0 ? "more" : "less") than the central
        ones: the beam reaches them up to $(round(cone; digits = 1))° off the axis, and in this model the rings of crystal are perpendicular
        to the axis (only the grid's walls, where there are any, lean toward the focal spot), so $(why). Real detectors
        are built from modules turned toward the focal spot along z as well, which this model's single arc is not; the
        efficiency table, from one dexel facing the beam, is not affected."""
    Markdown.parse(
        """
        | photons of | whole detector (air scan) | one dexel's cell, with the grid (section 6) |
        |---|---|---|
        $(join(lines, "\n"))

        $(verdict)
        """
    )
end

# ╔═╡ 2ab149ac-24be-56fc-9c18-fc41227b7113
md"""
## 8. The Aquilion ONE's X-Ray Beam

Canon publishes no filtration, so the beam is fitted. The spectra are the installed BasisSimulator's IPEM Report 78
tungsten spectra at a 10° anode (the ONE ViSION's; not stated for GENESIS or PRISM), through a flat
aluminium filter whose thickness is chosen so that the half-value layer at the isocentre, through
the thinnest part of BasisSimulator's large-body bowtie (2.5 mm of aluminium), matches Ito et al.'s
measurements on an Aquilion ONE PRISM: **6.0, 7.4 and 8.5 mm of aluminium at 80, 100 and 120 kV**
(BasisSimulator has no 135 kV spectrum; the measured 9.2 mm there is left out). The PRISM's
**SilverBeam** filter, 0.5 mm of silver at 120 kV, measured at **13.7 mm** of aluminium, then checks
the fitted beam independently.
"""

# ╔═╡ 10c68b00-24d3-58cd-9f5b-2b25c917fb5f
# Measured half-value layers at the isocentre, Aquilion ONE PRISM (Ito et al., J. Appl. Clin. Med. Phys. 2026)
measured_hvl = [(kVp = 80, hvl = 6.0), (kVp = 100, hvl = 7.4), (kVp = 120, hvl = 8.5)]

# ╔═╡ 28f3d388-5432-5c8b-83dd-17ff0ca8fdd6
aluminium = MC.DetectorMaterial(:Al)

# ╔═╡ 5ae61127-e981-5614-b7cc-49a51e5fa77e
# The large-body bowtie at its thinnest, the ray through the isocentre: aluminium only, in mm
bowtie_al_mm = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
end

# ╔═╡ 77c888f8-546b-55bd-8e6d-76a9c2068b16
"""
The photons per mAs per mm² reaching the detector, per bin, for an Aquilion ONE protocol at `kVp`, with
`flat_mm` of aluminium as the beam's filtration and a 10° anode.
"""
function aquilion_spectrum(kVp; flat_mm)
    scanner = BS.EICTScanner(
        source_to_isocenter = aquilion.source_to_isocenter_mm,
        source_to_detector = aquilion.source_to_detector_mm,
        detector_rows = aquilion.rows,
        detector_cols = aquilion.columns,
        detector_row_size = aquilion.row_at_isocenter_mm,
        detector_col_size = pitch_mm[1] * aquilion.source_to_isocenter_mm / aquilion.source_to_detector_mm,
        detector_shape = :arc,
        target_angle = 10.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = flat_mm,
        bowtie_filter = :large_body,
        detector_material = :ufc,
        detector_depth = depth_mm,
    )
    protocol = BS.CTProtocol(;
        kVp,
        mA = 200.0,
        views = 1200,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        anode_angle = 10,
    )
    return BS.resolve_source_spectrum_without_bowtie(BS.SimOptions(), protocol; scanner)
end

# ╔═╡ 6002b833-7de7-5e06-a753-277aab945c0e
# The half-value layer at the isocentre for `flat_mm` of filtration, through the bowtie's centre
function hvl_at(kVp, flat_mm)
    e, w = aquilion_spectrum(kVp; flat_mm)
    μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
    return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
end

# ╔═╡ 36ead4eb-6aa3-5a38-82f0-7427bcd7c338
# The aluminium filtration whose half-value layers best match the measured ones (least squares,
# golden-section search over 0–20 mm)
filtration = let
    misfit(t) = sum(abs2, [hvl_at(m.kVp, t) - m.hvl for m in measured_hvl])
    a, b = 0.0, 20.0
    φ = (sqrt(5) - 1) / 2
    for _ in 1:40
        c = b - φ * (b - a)
        d = a + φ * (b - a)
        if misfit(c) < misfit(d)
            b = d
        else
            a = c
        end
    end
    (a + b) / 2
end

# ╔═╡ be6ef043-75d4-59cf-b172-491e2200a90a
hvl_check = [(kVp = m.kVp, fitted = hvl_at(m.kVp, filtration), measured = m.hvl) for m in measured_hvl]

# ╔═╡ f367d731-fe78-4f9f-964d-b182faf883a9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 380))
    for (k, (al, reading, title)) in enumerate(((0, "100 %", "No Aluminium in the Beam"), (8.5, "50 %", "8.5 mm of Aluminium: Half Is Left (Measured)")))
        ax = canvas(fig[1, k]; title, titlesize = 20)
        Mke.scatter!(ax, [Mke.Point2f(0, 0)]; marker = :rect, markersize = 22, color = PALETTE.ink)
        Mke.text!(ax, 0, -0.45; text = "Tube\n(120 kV, Through\nthe Bowtie's Centre)", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
        Mke.poly!(ax, Mke.Point2f[(0.1, 0), (6, 0.45), (6, -0.45)]; color = (MATERIALS.beam, 0.25))
        if al > 0
            for i in 0:3
                rect!(ax, 2.8 + 0.18i, -0.35, 0.12, 0.7, Mke.RGBf(0.75, 0.77, 0.8); strokecolor = (:black, 0.3))
            end
            Mke.text!(ax, 3.1, 0.5; text = "Aluminium Sheets", align = (:center, :bottom), fontsize = 14, color = PALETTE.ink)
        end
        Mke.poly!(ax, Mke.Point2f.(6.0 .+ 0.35 .* cos.(range(0, 2π; length = 60)), 0.35 .* sin.(range(0, 2π; length = 60))); color = Mke.RGBf(0.85, 0.87, 0.9), strokecolor = PALETTE.slab_edge)
        Mke.text!(ax, 6.0, -0.5; text = "Ion Chamber\nat the Isocentre", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
        Mke.textlabel!(ax, Mke.Point2f(6.0, 0.75); text = "Air Kerma: $(reading)", text_align = (:center, :bottom), fontsize = 16, font = :bold, text_color = PALETTE.kept, background_color = (:white, 0.9), strokewidth = 0)
        Mke.limits!(ax, -0.8, 7.0, -1.3, 1.3)
    end
    fig
end

# ╔═╡ 81e9027d-a62a-5a0c-a091-f587d7c0a121
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-Value Layer at the Isocentre: the Fitted Beam Against Measurement",
        xlabel = "Tube Voltage",
        ylabel = "Half-Value Layer (mm of Aluminium)",
        xticks = (1:length(hvl_check), ["$(c.kVp) kV" for c in hvl_check]),
        limits = ((0.4, length(hvl_check) + 0.6), (0, 12)),
    )
    x = 1:length(hvl_check)
    Mke.barplot!(
        ax,
        x .- 0.18,
        [c.fitted for c in hvl_check];
        width = 0.34,
        color = PALETTE.photon,
        label = "Fitted Beam: $(round(filtration; digits = 1)) mm Al, 10° Anode",
    )
    Mke.barplot!(
        ax,
        x .+ 0.18,
        [c.measured for c in hvl_check];
        width = 0.34,
        color = PALETTE.kept,
        label = "Measured (Ito et al. 2026)",
    )
    for (k, c) in enumerate(hvl_check)
        for (dx, v) in ((-0.18, c.fitted), (0.18, c.measured))
            note!(
                ax,
                k + dx,
                v,
                "$(round(v; digits = 1))";
                align = (:center, :bottom),
                offset = (0, 4),
                fontsize = 15,
                alpha = 0.0,
            )
        end
    end
    Mke.axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 0f0250c8-45d8-56d7-b1e5-f15bcf7a6714
let
    worst = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        **$(round(filtration; digits = 1)) mm of aluminium** at a 10° anode reproduces the measured half-value layers to within
        **$(round(worst; digits = 2)) mm** at every tube voltage: one filtration fits all three, which is what a real tube's fixed filtration
        does. This fitted beam is the Aquilion ONE beam for the rest of the notebook and is saved with the table. (The fit
        lumps the unstated titanium filter, the tube's own filtration and the anode angle into one aluminium thickness;
        the bowtie the measurement used is not stated.)
        """
    )
end

# ╔═╡ 5c7fff9b-6630-5e79-a499-efc14c14e3bc
md"""
**An independent check: SilverBeam.** The PRISM's SilverBeam adds 0.5 mm of silver to the 120 kV
beam, for low-dose lung screening; Ito et al. measured its half-value layer at 13.7 mm of aluminium.
The fit never saw this number, so the fitted beam through 0.5 mm of silver should land on it if the
fit captured the real spectrum, not just three numbers.
"""

# ╔═╡ 6b66d29d-5dab-579c-a5d8-1233dc4f35b9
silverbeam = let
    silver = MC.DetectorMaterial("Ag"; density = 10.49, formula = "Ag")
    e, w = aquilion_spectrum(120; flat_mm = filtration)
    μ_al = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
    μ_ag = [MC.linear_attenuation(silver, max(E, 1.0)) for E in e]
    filtered = w .* exp.(-μ_ag .* 0.5)
    (
        spectrum = (e, filtered),
        hvl = MC.half_value_layer(e, filtered .* exp.(-μ_al .* bowtie_al_mm)),
        measured = 13.7,
    )
end

# ╔═╡ 1703f60a-9d85-5a57-962d-91c4f8b0bece
Markdown.parse(
    """
    Through 0.5 mm of silver the fitted 120 kV beam has a half-value layer of **$(round(silverbeam.hvl; digits = 1)) mm** of aluminium,
    against the measured 13.7 mm: $(abs(silverbeam.hvl - 13.7) <= 0.7 ? "the fitted beam passes a check it was not fitted to." : "**⚠ the fitted beam misses a check it was not fitted to**; one aluminium thickness does not capture the spectrum's shape well enough for a filter this strong.")
    """
)

# ╔═╡ d10caf51-f7a2-5154-8d24-e1ed46c4d86c
md"""
The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ ff9e2424-574b-5274-9c8f-adbad84450d6
@bind kvp PlutoUI.Slider([80, 100, 120]; default = 120, show_value = true)

# ╔═╡ 45043ecb-063f-5b53-bd1f-5b559c97199a
"""
For a spectrum `(e, w)`: the photons per keV, and the signal they make per keV in a detector of
efficiency `η` on BasisSimulator's 1-keV grid (`Φ(E) · E · η(E)`).
"""
function signal_per_kev(spectrum, η)
    e, w = spectrum
    bin = e[2] - e[1]
    at(E) = MC.interpolate_linear(collect(energies), η, E)
    return (
        energy_keV = e,
        photons = w ./ bin,
        signal = [wi / bin * E * at(E) for (E, wi) in zip(e, w)],
    )
end

# ╔═╡ 74ed96cd-3bb4-4cf5-b1ca-d01c5ee9b0d2
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))

    ax = canvas(fig[1, 1])
    amber = MATERIALS.beam
    # tube: envelope, cathode, anode at 10° (the ONE ViSION's; not stated for GENESIS or PRISM)
    Mke.poly!(ax, Mke.Rect2f(-0.6, -1.5, 2.4, 3.0); color = Mke.RGBf(0.93, 0.94, 0.96), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    rect!(ax, -0.45, -0.25, 0.3, 0.5, PALETTE.slab_edge)
    Mke.text!(ax, -0.3, -0.45; text = "Cathode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.poly!(ax, Mke.Point2f[(1.2, -0.9), (1.55, -0.9), (1.55, 0.9), (1.35, 0.9)]; color = Mke.RGBf(0.6, 0.45, 0.3))
    Mke.text!(ax, 1.42, -1.0; text = "Tungsten Anode, 10°", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    arrow!(ax, (-0.1, 0.0), (1.22, 0.0); color = PALETTE.photon, linewidth = 2.5, head = 11)
    Mke.text!(ax, 0.55, 0.12; text = "Electrons", align = (:center, :bottom), fontsize = 14, color = PALETTE.photon)
    focal = Mke.Point2f(1.28, 0.0)
    Mke.text!(ax, 0.6, 1.6; text = "X-Ray Tube", align = (:center, :bottom), fontsize = 18, font = :bold, color = PALETTE.ink)
    # the fan, rightwards
    x_end = 13.5
    Mke.poly!(ax, Mke.Point2f[focal, (x_end, 2.6), (x_end, -2.6)]; color = (amber, 0.13))
    for y in (-2.6, 0.0, 2.6)
        Mke.lines!(ax, [focal, Mke.Point2f(x_end, y)]; color = (amber, 0.5), linewidth = 1)
    end
    filters = (
        (3.2, 0.22, Mke.RGBf(0.75, 0.77, 0.8), "$(round(filtration; digits = 1)) mm Al", "Fitted to the\nMeasured HVLs"),
        (4.7, 0.08, Mke.RGBf(0.82, 0.83, 0.86), "0.5 mm Ag", "SilverBeam:\nPRISM Only"),
    )
    for (x, w, c, name, when) in filters
        h = 0.25 + (x - focal[1]) * 2.6 / (x_end - focal[1])
        rect!(ax, x, -h, w, 2h, c; strokecolor = (:black, 0.3))
        Mke.text!(ax, x + w / 2, h + 0.12; text = name, align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
        Mke.text!(ax, x + w / 2, -h - 0.12; text = when, align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    end
    # bowtie: thin in the middle, thick at the edges
    xb = 6.3
    hb = 0.25 + (xb - focal[1]) * 2.6 / (x_end - focal[1])
    Mke.poly!(ax, Mke.Point2f[(xb, -hb), (xb + 0.55, -hb), (xb + 0.08, 0), (xb + 0.55, hb), (xb, hb)]; color = Mke.RGBf(0.75, 0.77, 0.8), strokecolor = (:black, 0.3))
    Mke.text!(ax, xb + 0.25, hb + 0.12; text = "Bowtie", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.text!(ax, xb + 0.25, -hb - 0.12; text = "2.5 mm Al\nat the Centre", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    # patient and isocentre
    θ = range(0, 2π; length = 100)
    xi = 9.6
    Mke.poly!(ax, Mke.Point2f.(xi .+ 1.1 .* cos.(θ), 0.8 .* sin.(θ)); color = (MATERIALS.water, 0.6), strokecolor = (PALETTE.photon, 0.5))
    Mke.scatter!(ax, [Mke.Point2f(xi, 0)]; marker = :cross, markersize = 16, color = PALETTE.ink)
    Mke.text!(ax, xi, 0.95; text = "Patient (Isocentre, 600 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    # detector with grid
    rect!(ax, x_end, -2.7, 0.15, 5.4, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
    for y in range(-2.6, 2.6; length = 14)
        rect!(ax, x_end - 0.35, y - 0.01, 0.35, 0.02, MATERIALS.tungsten)
    end
    Mke.text!(ax, x_end + 0.1, 2.85; text = "Grid + Detector\n(1072 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -0.8, 14.6, -3.4, 3.9)
    locator!(fig[0, :], :tube; note = "The X-Ray Tube and\nWhat the Beam Passes")
    fig
end

# ╔═╡ b8e5ce21-e91b-5d3a-a6eb-942d1fb42af8
let
    Mke.set_theme!(THEME)
    spectrum = aquilion_spectrum(kvp; flat_mm = filtration)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The Photons Reaching the Detector, $(kvp) kV",
        xlabel = "Photon Energy (keV)",
        ylabel = "Photons per mAs per mm²\nper keV",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    s = signal_per_kev(spectrum, efficiency.flat_field)
    Mke.lines!(axa, s.energy_keV, s.photons; color = PALETTE.photon, linewidth = 3)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Signal They Make",
        xlabel = "Photon Energy (keV)",
        ylabel = "Signal per keV\n(Photons × E × η, Arbitrary Units)",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    theirs = signal_per_kev(spectrum, table_at.(collect(energies)))
    Mke.lines!(
        axb,
        theirs.energy_keV,
        theirs.signal;
        color = PALETTE.reference,
        linewidth = 3,
        label = "BasisSimulator v0.15.1's Flash Table",
    )
    Mke.lines!(
        axb,
        s.energy_keV,
        s.signal;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, PUREViSION Flat Field",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ 5ca60f2c-a230-5894-a8fa-1a5dcc16b64c
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. The Aquilion ONE's tube voltages BasisSimulator can make,
through the fitted beam, and the PRISM's SilverBeam:
"""

# ╔═╡ 2e1b5a19-af32-5b42-8f00-e79cbdc0fad1
scan_protocols = vcat(
    [(label = "$(k) kV", spectrum = aquilion_spectrum(k; flat_mm = filtration)) for k in (80, 100, 120)],
    [(label = "120 kV SilverBeam", spectrum = silverbeam.spectrum)],
)

# ╔═╡ b247d175-c3ff-5561-98c7-bbc089fbf17a
# Each protocol's spectrum on the 1-keV grid, then its energy-weighted efficiencies and DQE(0)
scan_metrics = let
    E = collect(energies)
    m1 = [d.crystal_keV / d.n for d in one_dexel]
    m2 = [d.crystal2_keV2 / d.n for d in one_dexel]
    function metrics(spectrum)
        e, w = spectrum
        φ = [sum(w[abs.(e .- Ek) .<= 0.5]) for Ek in E]
        weighted(η) = sum(φ .* E .* η) / sum(φ .* E)
        return (
            table = weighted(table_at.(E)),
            geant4 = weighted(efficiency.flat_field),
            dqe = sum(φ .* m1)^2 / (sum(φ) * sum(φ .* m2)),
        )
    end
    [merge((label = p.label,), metrics(p.spectrum)) for p in scan_protocols]
end

# ╔═╡ 74850911-4736-46f0-8270-9c09bbe85f0a
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 760))

    photons = ((40, 0.6), (70, 1.6), (110, 2.6))
    for (k, (title, detail)) in enumerate(
            (
                ("A Photon-Counting Dexel", "Each Photon Counts 1:\nSignal = 3"),
                ("This Energy-Integrating Dexel", "Each Photon Adds Its Energy × η(E):\nSignal ∝ 40 η(40) + 70 η(70) + 110 η(110)"),
            )
        )
        ax = canvas(fig[1, k]; title, titlesize = 20)
        rect!(ax, 0, -0.6, 3.2, 0.6, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
        for (E, x) in photons
            photon!(ax, (x, 2.2), (x, -0.25); cycles = E / 12, amplitude = 0.02, color = MATERIALS.beam, linewidth = 1 + E / 30)
            Mke.text!(ax, x, 2.3; text = "$(E) keV", align = (:center, :bottom), fontsize = 15, color = PALETTE.ink)
            if k == 1
                Mke.textlabel!(ax, Mke.Point2f(x, -1.2); text = "+1", text_align = (:center, :center), fontsize = 16, font = :bold, text_color = :white, background_color = PALETTE.photon, strokewidth = 0, cornerradius = 8)
            else
                Mke.poly!(ax, Mke.Rect2f(x - 0.18, -1.65, 0.36, 0.28 * E / 40); color = PALETTE.kept)
            end
        end
        Mke.limits!(ax, -0.4, 3.6, -1.8, 2.8)
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :photodiode; note = "Steps 4–5 · Adding Up\nthe Light of Many Photons")
    fig
end

# ╔═╡ baaa15f4-add3-5d99-9695-3d6721e6d6f0
let
    Mke.set_theme!(THEME)
    x = 1:length(scan_metrics)
    labels = [m.label for m in scan_metrics]
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Mean Efficiency over Each Protocol's Spectrum",
        ylabel = "Energy-Weighted Efficiency",
        xticks = (x, labels),
        xticklabelrotation = π / 5,
        limits = ((0.4, length(x) + 0.6), (0.5, 1.0)),
    )
    Mke.scatterlines!(
        axa,
        x,
        [m.table for m in scan_metrics];
        color = PALETTE.reference,
        markersize = 13,
        linewidth = 2.5,
        label = "BasisSimulator v0.15.1's Flash Table",
    )
    Mke.scatterlines!(
        axa,
        x,
        [m.geant4 for m in scan_metrics];
        color = PALETTE.kept,
        markersize = 13,
        linewidth = 2.5,
        label = "Geant4, 1.4 mm, Flat Field",
    )
    Mke.axislegend(axa; position = :lb)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) DQE(0) over Each Spectrum (Geant4)",
        ylabel = "DQE(0)",
        xticks = (x, labels),
        xticklabelrotation = π / 5,
        limits = ((0.4, length(x) + 0.6), (0.5, 1.0)),
    )
    Mke.scatterlines!(
        axb,
        x,
        [m.dqe for m in scan_metrics];
        color = PALETTE.photon,
        markersize = 13,
        linewidth = 2.5,
    )
    fig
end

# ╔═╡ 47436151-80ce-52d6-856e-b75b78deee48
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | v0.15.1 Flash table | Geant4, PUREViSION 1.4 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's ceramic; the fill factor (section 2) applies to both alike.
        """
    )
end

# ╔═╡ 26e1e385-4437-4022-9071-4da46fd02fb2
md"""
## 10. A Clinical Beam, for Real

Every efficiency so far came from photons of **one energy**, landing straight on **one dexel**.
A scan is neither: the Aquilion ONE's tube sends a spectrum, hardened by the bowtie and by the patient,
from a focal spot over the whole arc, 16 cm wide, so the dexels at the edges of the fan and the
outer rows see it at an angle, through the grid. BasisSimulator never simulates that photon by photon; it
weights the single-energy table by the spectrum. This section checks that the shortcut is exact,
by doing it the long way:

- the Aquilion ONE's **120 kV beam** as fitted in section 8 (BasisSimulator's spectrum at a 10°
  anode through the fitted aluminium and the bowtie's centre), **bare and after 20 and 35 cm of
  water** (the patient's primary photons; the patient's scatter is notebook 10's), and the
  PRISM's **SilverBeam** (0.5 mm of silver more), bare;
- each one **drawn photon by photon from its spectrum**, onto one dexel's ceramic (1.6 million
  photons) and from the focal spot over **the whole arc** with its grid (3.2 million);
- against the table **weighted by the same spectrum**; for the whole arc, one cell's efficiency
  times the arc's own excess over it at single energies (section 7: about 5 %, a limit of the
  model's single arc across 16 cm), weighted the same way: `Σ φ E η / Σ φ E` for the efficiency, and
  `(Σ φ m₁)² / (Σ φ · Σ φ m₂)` for DQE(0), with `m₁`, `m₂` the mean and mean square of the energy
  a photon leaves (section 4).
"""

# ╔═╡ 33f76d9d-9a91-4c51-b08f-e100114f8181
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))


    # The path: tube, bowtie, water, the arc with its grid
    ax = canvas(fig[1, 1]; title = "The Beam, Photon by Photon", titlesize = 20)
    amber = MATERIALS.beam
    focal = Mke.Point2f(0, 0)
    Mke.scatter!(ax, [focal]; marker = :rect, markersize = 22, color = PALETTE.ink)
    Mke.text!(ax, focal + Mke.Point2f(0, 0.45); text = "120 kV Tube", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    φ = range(-deg2rad(24.9), deg2rad(24.9); length = 60)
    R = 10.0
    arc = [Mke.Point2f(R * sin(t), -R * cos(t)) for t in φ]
    Mke.poly!(ax, vcat([focal], arc); color = (amber, 0.15))
    Mke.poly!(ax, Mke.Point2f[(-1.2, -1.6), (1.2, -1.6), (0.15, -2.0), (1.2, -2.4), (-1.2, -2.4), (-0.15, -2.0)]; color = Mke.RGBf(0.75, 0.77, 0.8))
    Mke.text!(ax, 1.35, -2.0; text = "Bowtie", align = (:left, :center), fontsize = 15, color = PALETTE.ink)
    θ = range(0, 2π; length = 100)
    Mke.poly!(ax, Mke.Point2f.(2.4 .* cos.(θ), -5.45 .+ 1.6 .* sin.(θ)); color = (MATERIALS.water, 0.7), strokecolor = (PALETTE.photon, 0.5))
    Mke.text!(ax, 0, -5.45; text = "0, 20 or 35 cm\nof Water", align = (:center, :center), fontsize = 15, color = PALETTE.ink)
    inner = [Mke.Point2f((R + 0.05) * sin(t), -(R + 0.05) * cos(t)) for t in φ]
    outer = [Mke.Point2f((R + 0.45) * sin(t), -(R + 0.45) * cos(t)) for t in φ]
    Mke.poly!(ax, vcat(inner, reverse(outer)); color = MATERIALS.ufc, strokecolor = PALETTE.slab_edge)
    for t in range(φ[1], φ[end]; length = 31)
        Mke.lines!(ax, Mke.Point2f[((R - 0.5) * sin(t), -(R - 0.5) * cos(t)), (R * sin(t), -R * cos(t))]; color = MATERIALS.tungsten, linewidth = 1)
    end
    Mke.text!(ax, 0, -R - 0.7; text = "The Whole Arc and Its Grid ($(aquilion.columns) × $(aquilion.rows) Dexels)", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -6.5, 6.5, -R - 1.8, 1.4)

    # The three spectra, each as a fraction of its photons per keV: the water hardens the beam
    continuum(e) = !(57 <= e <= 70)
    spectra = [
        (b.label, b.spectrum.energies_keV, b.spectrum.weights ./ (sum(b.spectrum.weights) * 0.5))
            for b in clinical_beams
    ]
    top = 1.35 * maximum(maximum(w[continuum.(e)]) for (_, e, w) in spectra)
    axs = Mke.Axis(
        fig[1, 2];
        title = "The Beam Hardens in the Patient",
        titlesize = 20,
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons per keV",
        limits = ((10, 125), (0, top)),
        xticks = 20:20:120,
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    for (k, (label, e, w)) in enumerate(spectra)
        mean_E = sum(w .* e) / sum(w)
        Mke.lines!(axs, e, w; color = colors[k], linewidth = 3, label = "$(label): Mean $(round(mean_E; digits = 1)) keV")
    end
    note!(axs, 63.5, 0.97top, "Tungsten's K Lines\n(Off Scale)"; align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.axislegend(axs; position = :rt)
    Mke.colsize!(fig.layout, 1, Mke.Relative(0.32))
    locator!(fig[0, :], :whole; note = "Everything at Once:\nTube, Patient, Grid,\nEvery Dexel")
    fig
end

# ╔═╡ da3393ec-da8a-4e7a-9cb1-260090d7a201
# The Aquilion ONE's fitted 120 kV beam on the central ray, through the bowtie's centre, then
# through 0, 20 and 35 cm of water (the primary photons); and the PRISM's SilverBeam, bare
clinical_beams = let
    water = MC.DetectorMaterial(:Water)
    through_bowtie(e, w) = w .* exp.(-[MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e] .* bowtie_al_mm)
    e, w = aquilion_spectrum(120; flat_mm = filtration)
    plain = [
        (
                label = L == 0 ? "No Patient" : "$(round(Int, L / 10)) cm of Water",
                water_mm = L,
                spectrum = MC.Spectrum(
                    collect(Float64, e),
                    through_bowtie(e, w) .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* L),
                ),
            )
            for L in (0.0, 200.0, 350.0)
    ]
    se, sw = silverbeam.spectrum
    vcat(plain, [(label = "SilverBeam", water_mm = 0.0, spectrum = MC.Spectrum(collect(Float64, se), through_bowtie(se, sw)))])
end

# ╔═╡ 9d326916-1069-4e7b-bbdf-3a7814f72778
# Each beam photon by photon: onto one dexel's ceramic (16 chunks of 100 000), and from the focal
# spot over the whole arc and its grid (16 chunks of 200 000)
clinical_runs = let
    dexel = MC.simulate_sweep(
        [(dexel_array(depth_mm), b.spectrum, 300 + chunk) for b in clinical_beams for chunk in 1:16];
        n = 100_000,
        beam = crystal_beam,
        readout,
    )
    arc = MC.simulate_sweep(
        [(detector, b.spectrum, 400 + chunk) for b in clinical_beams for chunk in 1:16];
        n = 200_000,
        beam = MC.focal_spot(detector),
        readout = MC.ArcEnergyReadout(),
    )
    (
        dexel = [dexel[(16(k - 1) + 1):(16k)] for k in eachindex(clinical_beams)],
        arc = [arc[(16(k - 1) + 1):(16k)] for k in eachindex(clinical_beams)],
    )
end

# ╔═╡ 5375ef36-11c4-43ac-8555-895ed05c3263
# The long way against the shortcut, for each beam: efficiency and DQE(0) on one dexel, and the
# energy the whole arc collects per photon (against its own single-energy air scans); each direct value with its noise (the 16 chunks' spread)
clinical_check = let
    E = collect(energies)
    m1 = [d.crystal_keV / d.n for d in one_dexel]
    m2 = [d.crystal2_keV2 / d.n for d in one_dexel]
    cell = MC.energy_absorption_efficiency.(grid_runs.with)
    # the whole arc against one cell, from section 7's own air scans: the fan and cone angles (and the
    # model's arc, not tilted along z) change it a little; interpolated between the scans' energies
    cell_at(x) = MC.interpolate_linear(collect(grid_energies), cell, clamp(x, 20.0, 140.0))
    arc_ratio = [sum(s.energy_keV) / s.photon_keV / cell_at(E) for (E, s) in zip(air_energies, air_scans)]
    ratio_at(x) = MC.interpolate_linear(air_energies, arc_ratio, clamp(x, first(air_energies), last(air_energies)))
    at(table, x) = MC.interpolate_linear(E, table, x)
    noise(values) = sqrt(sum(abs2, values .- mean(values)) / (length(values) - 1) / length(values))
    map(enumerate(clinical_beams)) do (k, b)
        e, w = b.spectrum.energies_keV, b.spectrum.weights
        mean_E = sum(w .* e) / sum(w)
        chunks, arcs = clinical_runs.dexel[k], clinical_runs.arc[k]
        efficiency_of(d) = d.crystal_keV / (d.n * mean_E)
        dqe_of(cs) = sum(c.crystal_keV for c in cs)^2 / (sum(c.n for c in cs) * sum(c.crystal2_keV2 for c in cs))
        arc_of(a) = sum(a.energy_keV) / a.photon_keV
        (
            label = b.label,
            mean_keV = mean_E,
            efficiency = (
                table = sum(w .* e .* at.(Ref(efficiency.flat_field), e)) / sum(w .* e),
                direct = sum(c.crystal_keV for c in chunks) / (sum(c.n for c in chunks) * mean_E),
                noise = noise(efficiency_of.(chunks)),
            ),
            dqe = (
                table = sum(w .* at.(Ref(m1), e))^2 / (sum(w) * sum(w .* at.(Ref(m2), e))),
                direct = dqe_of(chunks),
                noise = noise([dqe_of([c]) for c in chunks]),
            ),
            arc = (
                table = sum(w .* e .* [cell_at(x) * ratio_at(x) for x in e]) / sum(w .* e),
                direct = arc_of(sum(arcs)),
                noise = noise(arc_of.(arcs)),
            ),
        )
    end
end

# ╔═╡ c5265d77-13fe-4887-b27c-3cb2688d54bd
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 560))
    x = 1:length(clinical_check)
    labels = [c.label for c in clinical_check]
    panels = (
        (key = :efficiency, title = "(a) Efficiency, One Dexel", ylabel = "Energy-Weighted Efficiency"),
        (key = :dqe, title = "(b) DQE(0), One Dexel", ylabel = "DQE(0)"),
        (key = :arc, title = "(c) The Whole Arc and Its Grid", ylabel = "Energy Collected per Photon ÷ E"),
    )
    for (k, panel) in enumerate(panels)
        values = [getfield(c, panel.key) for c in clinical_check]
        lo = minimum(min(v.table, v.direct) for v in values)
        hi = maximum(max(v.table, v.direct) for v in values)
        pad = max(0.02, 0.6 * (hi - lo))
        ax = Mke.Axis(
            fig[1, k];
            title = panel.title,
            ylabel = panel.ylabel,
            xticks = (x, labels),
            xticklabelrotation = π / 8,
            limits = ((0.5, length(x) + 0.5), (lo - pad, hi + pad)),
        )
        Mke.scatter!(ax, x .- 0.08, [v.table for v in values]; color = PALETTE.reference, markersize = 16, marker = :diamond, label = "Table, Weighted by the Spectrum")
        Mke.errorbars!(ax, x .+ 0.08, [v.direct for v in values], [3v.noise for v in values]; color = PALETTE.kept, whiskerwidth = 10, linewidth = 2)
        Mke.scatter!(ax, x .+ 0.08, [v.direct for v in values]; color = PALETTE.kept, markersize = 14, label = "Geant4, the Spectrum Itself (± 3σ)")
        k == 1 && Mke.axislegend(ax; position = :lb)
    end
    fig
end

# ╔═╡ e46cc64f-eeac-4161-b327-a23606fea015
let
    rows = String[]
    worst = 0.0
    for c in clinical_check, (name, v, floor) in (("efficiency, one dexel", c.efficiency, 0.002), ("DQE(0), one dexel", c.dqe, 0.003), ("the whole arc", c.arc, 0.01))
        difference = v.direct - v.table
        allowed = 3v.noise + floor
        pass = abs(difference) <= allowed
        worst = max(worst, abs(difference) / allowed)
        push!(rows, "| $(pass ? "✓" : "✗ **fail**") | $(c.label), $(name) | $(round(v.table; digits = 4)) | $(round(v.direct; digits = 4)) ± $(round(v.noise; digits = 4)) | $(round(difference; digits = 4)) | ≤ $(round(allowed; digits = 4)) |")
    end
    passed = count(r -> startswith(r, "| ✓"), rows)
    Markdown.parse(
        """
        | | beam, quantity | table, weighted | Geant4, the spectrum itself | difference | allowed |
        |---|---|---|---|---|---|
        $(join(rows, "\n"))

        **$(passed) of $(length(rows)) agree.** Allowed: three times the direct simulation's own noise, plus 0.002 for the
        efficiency and 0.003 for DQE(0) (the table's noise and its 1-keV grid), and 0.01 for the whole arc, whose prediction is one cell's efficiency
        times the arc's own excess over it, measured in section 7's air scans and interpolated between their energies: so
        this checks the spectrum's weighting, not the arc's geometry, which section 7 reports. Where they agree, the
        single-energy table, weighted by any spectrum, is the clinical beam: what BasisSimulator does with it is exact.
        """
    )
end

# ╔═╡ c48615a2-707b-5ccc-b824-acf128726517
md"""
## The Table to Use

The table for an Aquilion ONE in BasisSimulator (from v0.16.0), and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **Gd₂O₂S ceramic at 7.31 g/cm³, 1.4 mm deep** (assumed). Canon publishes only "praseodymium-
  activated ceramic"; 1.4 mm is the one Gd₂O₂S CT thickness a vendor (Siemens) quotes. Every depth from
  0.8 to 2.0 mm is saved beside it.
- **The fill factor of the assumed reflector** (section 2); the 1-D grid's plates stand on the
  reflector and take nothing more (section 6).
- **The fitted beam**: BasisSimulator's IPEM spectra at a 10° anode through the fitted aluminium,
  which reproduces the measured half-value layers and is checked on SilverBeam (section 8).

The GENESIS and the PRISM share the detector and the tube, so both files hold the same efficiency and
beam; the PRISM's adds SilverBeam.
"""

# ╔═╡ 8b77445b-bc28-5204-89ef-89814e715f0d
md"""
## Saving the Tables

Two files in `lib/BasisMonteCarlo/tables/`, one per edition, in the layout of BasisSimulator's
`*_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of the PUREViSION as
modelled here, beside the one-dexel efficiency, the interaction probability, the Swank factor,
DQE(0), the efficiency for depths from 0.8 to 2.0 mm, and the fitted beam.
"""

# ╔═╡ 14c7a701-5529-58e5-bd40-1d6d46907e66
table_depths = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# ╔═╡ fd3f9361-3413-50e2-8dca-dadab6f783b7
# One column per depth, one row per energy
by_depth = reshape(
    MC.simulate_sweep(
        [(dexel_array(depth), E) for depth in table_depths for E in energies];
        n = 50_000,
        beam = crystal_beam,
        seed = 19,
        readout,
    ),
    length(energies),
    length(table_depths),
)

# ╔═╡ 1549c982-b5aa-54ba-80c0-000deeceece7
# The machine path stays out of the rendered page
table_dir = joinpath(@__DIR__, "..", "..", "tables");

# ╔═╡ 95459b06-583d-5fe0-bf96-830980415543
saved = let
    common = Dict{String, Any}(
        "energies" => collect(energies),
        "efficiency" => efficiency.flat_field,
        "efficiency_one_dexel" => efficiency.one_dexel,
        "interaction_probability" => efficiency.interaction,
        "swank_factor" => efficiency.swank,
        "dqe0" => efficiency.dqe,
        "depths_mm" => table_depths,
        "efficiency_by_depth" => MC.energy_absorption_efficiency.(by_depth),
        "decision" => "use `efficiency`: flat field, Gd2O2S ceramic at 7.31 g/cm3 and 1.4 mm deep (both assumed; Canon publishes only 'praseodymium-activated ceramic'); fill factor of the assumed 0.1 mm reflector",
        "definition" => "mean energy collected per photon landing on a dexel's ceramic with every dexel illuminated (flat field), divided by the photon energy; the fill factor applies separately",
        "beam" => Dict(
            "flat_filter_al_mm" => filtration,
            "anode_angle_deg" => 10,
            "bowtie" => "BasisSimulator :large_body",
            "fitted_to" => "measured HVL 6.0, 7.4, 8.5 mm Al at 80, 100, 120 kV, Aquilion ONE PRISM (Ito et al. 2026)",
            "silverbeam_hvl_mm_al" => silverbeam.hvl,
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 08_canon_aquilion_one.jl",
    )
    paths = String[]
    for (file, scanner, rows, grid) in (
            (
                "canon_aquilion_one_genesis_efficiency_geant4.jls",
                "Canon Aquilion ONE GENESIS (TSX-305A)",
                aquilion.rows,
                "1-D (assumed; not published)",
            ),
            (
                "canon_aquilion_one_prism_efficiency_geant4.jls",
                "Canon Aquilion ONE PRISM (TSX-306A), with SilverBeam",
                aquilion.rows,
                "1-D (assumed; not published)",
            ),
        )
        contents = merge(
            common,
            Dict{String, Any}(
                "detector" => Dict(
                    "scanner" => scanner,
                    "material" => "praseodymium-activated ceramic, modelled as Gd2O2S at 7.31 g/cm3 (assumed)",
                    "depth_mm" => depth_mm,
                    "pitch_mm" => pitch_mm,
                    "columns" => aquilion.columns,
                    "rows" => rows,
                    "reflector" => "0.1 mm TiO2-loaded epoxy (assumed)",
                    "grid" => grid,
                ),
            ),
        )
        path = joinpath(table_dir, file)
        Serialization.serialize(path, contents)
        push!(paths, path)
    end
    paths
end;

# ╔═╡ 09fd9bf9-da85-5488-ac38-c1bd9bf15da8
# Read them back in BasisSimulator's layout
[
    let t = Serialization.deserialize(path)
            (
                file = basename(path),
                rows = t["detector"]["rows"],
                at_60keV = t["efficiency"][60],
                at_120keV = t["efficiency"][120],
            )
    end
        for path in saved
]

# ╔═╡ db084ca4-290d-5555-bba5-06a94a2b6348
md"""
## Where This Leaves the Aquilion ONE
"""

# ╔═╡ 8c1258f9-0411-5633-807d-fa9809e266d0
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    worst_hvl = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        - **Little is documented**: 320 × 0.5 mm rows, 896 channels, the PUREViSION detector shared by GENESIS and PRISM.
          The distances (600 / 1072 mm), the pitch at the detector, the scintillator (Gd₂O₂S at 7.31 g/cm³, 1.4 mm), the
          reflector and the 1-D grid are worked out or assumed, and said so.
        - **The beam is fitted**: $(round(filtration; digits = 1)) mm of aluminium at a 10° anode reproduces the PRISM's measured half-value
          layers to within $(round(worst_hvl; digits = 2)) mm, and gives $(round(silverbeam.hvl; digits = 1)) mm through SilverBeam against the measured 13.7 mm.
        - **What a scan sees** is the flat-field efficiency: $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, with DQE(0)
          $(round(m120.dqe; digits = 3)); borrowing BasisSimulator v0.15.1's Flash table would give $(round(m120.table; digits = 3)).
        - **The whole detector**: 16 cm wide, its outer rows see the beam up to $(round(rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / aquilion.source_to_detector_mm)); digits = 1))° off the axis (section 7).
        - **Saved:** $(join(["`$(basename(p))`" for p in saved], " and ")), for BasisSimulator.
        """
    )
end

# ╔═╡ Cell order:
# ╟─80569823-8008-530d-a0df-34c4e21e0758
# ╟─5668de3a-22de-42be-8847-10f72fb78120
# ╟─0e0805a1-0452-5360-8e0e-27c7450b5769
# ╟─2c81d3e6-baec-47b5-a67e-63e27c1e674e
# ╟─cb572a6e-a206-4ee0-a7d8-a07bb0f13cde
# ╟─5e84a0e1-897c-4189-9364-bd06e6401f17
# ╟─e3e7a4a7-1dbd-443a-b51d-db0893932643
# ╟─db0ace6b-e7f9-56bf-850a-db1a6d6e2e06
# ╟─33e46fd5-c542-55e8-bcb5-192a0bac3d5e
# ╠═ab9db025-f507-5388-bf67-4c0d170d0e28
# ╠═bc041287-f4f5-5ce7-8924-9633ea5dc417
# ╠═808a4723-438f-586f-b92c-6e49eb9f89e6
# ╠═8b53812d-c201-51bb-8847-1b66b5f1f09c
# ╠═429139dd-55bf-52bd-a9d5-9691d14d06a2
# ╠═1a29df66-ff12-5ff0-acdd-9c0e6b2b13d6
# ╠═726ef0af-666e-5b87-9ea4-02801135cab1
# ╠═3accdd63-8b76-5d41-a238-01a65ba33416
# ╠═10fb07d1-38ee-5894-b238-df1f1e70be84
# ╠═558bad13-644e-5e3d-be50-5c5850d10928
# ╠═ed268a94-6b31-5c94-87d3-d11efbaeb1a0
# ╠═c9b96524-0916-5e0a-bd06-366cce225226
# ╠═57dd0152-ffba-53a8-a5fc-33d79b6ce3ac
# ╠═1b1c6cc3-ced3-5a5c-affb-f6456a4012f2
# ╠═bfe11ba8-730e-58f7-9f66-212ffb916313
# ╟─d249bab7-2437-44ed-8739-4479d42c8913
# ╠═a5eec273-7b42-4e5e-8716-1ba9b2ba4bc5
# ╠═369dc801-1932-4c4c-8208-d6f3231f5c25
# ╠═4ac95dda-4850-4ba6-a34d-12802dd49631
# ╠═f3e47edf-5e45-4d71-8b44-89768e87a76a
# ╠═517c6f56-669d-4d73-9b25-d1cb4d396e81
# ╠═c21aedb3-3ef4-4978-adec-e62119468b9c
# ╠═6ff3ad7b-9969-4ed8-93e4-c84c35e30998
# ╠═0536bb6b-fdb7-4e3d-933c-77318a5c808b
# ╠═c356dd1c-8044-48a8-b1b7-6d4143ee8826
# ╠═d1aa97ab-cc9c-4bb0-82c9-be7565a26ae6
# ╠═3126d46a-d155-4d3e-b623-9209d440d84a
# ╠═5ba1e6a0-564e-42ea-88d1-3dddd70e0886
# ╠═4b31992d-307b-45e6-ba53-97844f8e05e4
# ╠═f17692d1-0d59-44c4-bdc2-2a6e8caf55a3
# ╠═6f910999-861d-4c55-a413-7817cddd2337
# ╠═49a94f31-0592-4639-bded-a21d8386ada0
# ╠═43065420-8e14-436c-a9bb-a2c60286508d
# ╠═691eb609-3c95-4091-8951-803053559dff
# ╠═ad5a2ff3-ec50-4d51-8e2b-2b08382eeb4a
# ╠═81b6e112-d19d-4f34-a160-443efbab8bbb
# ╠═eab9ebaf-0e72-4d17-9e6b-6d1ba535d9c5
# ╠═640cd996-29e0-42fa-a581-9abf401ec18d
# ╠═ba771f9a-8ff5-456d-87e0-9e4c145f39e1
# ╠═a1697bfb-6249-4f64-9e50-e1f2befdbae7
# ╠═1718e123-ab81-465e-9b45-09c7c80b4351
# ╠═4f9450a4-2b36-46da-8244-eab441445365
# ╠═8516f41d-d952-47fc-92a0-a06b0f359dc0
# ╠═f632ab48-f213-4dbd-a2cf-dedb3cc00b95
# ╟─ddadd9bd-a2b7-58f8-a970-3bff3a89201e
# ╠═54cfabcd-ec9e-5930-af26-fe46ae2f6df2
# ╠═bfd257ef-c874-5d3e-865b-91eca06e30f1
# ╟─ce933054-651a-5933-9566-efb31407e7cd
# ╠═f2132f86-2dcb-5010-bb46-b3216f321502
# ╠═f7c3b5dc-b20b-5db8-bbc9-8b7b9ad389bb
# ╠═4e886e38-891b-5244-b3f7-f4217cb5785c
# ╠═919250d4-f19e-5252-bad8-a0191e7f3dd9
# ╟─2cecfe74-4b79-4662-850f-2fb93923e463
# ╟─94ece8ce-3e4d-53b1-8769-420e883094e7
# ╟─6926962b-931d-56b2-be23-590de184f48e
# ╟─fe6f7616-c5c3-5864-b959-5fd0d16bb6be
# ╠═4a1a5d21-5581-50d2-b02a-9174b8bfcd02
# ╠═3e588088-f4e1-5470-a7d8-f02f18437d48
# ╠═17fe60de-9741-5ec5-9555-7afb4820e74b
# ╟─b59c72be-f16c-480f-946a-978ec616d260
# ╟─3d6ad596-a1da-5c24-b39a-8cdc0a1314da
# ╟─157eec79-393b-5b59-bb8a-950a2c4f5802
# ╟─fcee931d-a949-5b8c-9bee-d59e556f0fc1
# ╠═e01b99a8-d89e-53cd-9b56-b4dcc5c374db
# ╠═2594a24f-dc08-533c-8b9d-2f99108aab4e
# ╠═219cb299-5acc-5911-9eaf-2e86161e449d
# ╠═92ec3c2f-5d09-56f1-95fd-e5d2d1450ff2
# ╠═f266e1ff-8e7c-52ae-acfb-8fc1c6f2ca9f
# ╟─498b5b6a-a948-5126-92a3-364216f7b20e
# ╠═85053c4d-c787-591e-868e-b0548f93a32f
# ╠═f18449da-5fb9-5dd1-8889-b6f1fd902f59
# ╠═076c1cfb-65a3-5d2e-b5b0-1d264e756fe8
# ╠═77ad16fd-5153-5a4b-801a-57c22ebbd8b7
# ╟─5c906809-30a7-5e8f-b82e-78981da811f3
# ╟─4121046c-46f1-52bc-b36f-5e41408abb40
# ╠═4ee27e1b-2c0f-5ba4-baaa-6e883057b012
# ╠═ac571320-09f8-5d9b-843c-50d2bef6fc49
# ╟─e71c343e-4038-52da-8770-c462750ba0bc
# ╟─23479c79-f281-4863-8e11-70e01667ce3e
# ╟─18dfc26f-97bd-4329-81f0-1e88ed75aaf9
# ╟─2baedffc-48d3-58a5-9816-c781f3598720
# ╠═767031ad-4240-58a7-8784-73ed6193049b
# ╠═ac0ae21f-6ba3-5b81-b801-b226d34a8645
# ╟─2c56d6b4-e760-4797-9f7e-222b1a3fdffe
# ╟─3734b13a-da38-504d-bb4a-ef79fccc86b9
# ╟─4456dbc6-d2ac-5b8b-b744-cd7109af51ea
# ╠═b0e5a025-3a3b-5d9d-8ae0-d597c6507d3d
# ╠═d21e6491-d12e-52ae-b383-bd12d671b4e9
# ╟─ac57653b-ccdc-448b-b017-47ce1deb749a
# ╟─c012a993-2845-5b07-be0a-f019bf4fea80
# ╟─b2e83820-62f2-50f2-9aea-a57e382bfc14
# ╟─cf122acc-fc8b-5d8d-ad7d-84a701424756
# ╠═e4c398e8-b01d-565f-81ed-8a19750e5e33
# ╠═98a5f0ad-bbe1-5aab-8dc8-4e2aecbe102c
# ╠═76f1fe13-108c-5f56-aeb2-5b626860523d
# ╟─9f728df7-dc6b-4ab6-8e01-6375174c9ccd
# ╟─89682fb9-9156-5081-9815-65ae00e73940
# ╟─ff5bcd43-c401-569f-bb51-3b3163ebcde3
# ╟─b67523ad-a398-5025-a89e-31bbc644a886
# ╠═b807babb-788e-50b5-9857-ded2f7dfff09
# ╟─dfb875ed-6d51-485a-b1ad-fd2525cc623b
# ╟─8a97e96d-38c0-52f0-a4b4-9c835b6413c8
# ╟─f4b79fc1-aa23-56f6-8bc1-eb2b3db7b47e
# ╠═fdbc6f9a-497d-588a-b8b5-6d7bd3b5d4e7
# ╟─afa80bb9-13fb-4c92-b215-16f2b8bf7ff4
# ╠═65264960-57e5-5281-b471-39fa9e34f538
# ╟─319f52f4-4c6d-42f3-931b-f6218dfd6a1e
# ╟─dac0f49a-6dcf-5351-843b-5bf3d2d603df
# ╟─89ee2f6a-8d3d-5837-a72b-b16ac6dd781c
# ╟─eab56392-a661-5fc6-8f95-068d32573a2b
# ╠═f2759008-f052-5b85-92fb-04c23dfa5842
# ╟─9dbcae9a-cb06-5a25-a722-9fb39b6e7114
# ╟─66a50ae8-d09e-57d8-b69c-664f6b362651
# ╠═da786503-a2d6-5880-9361-079b494dabcc
# ╠═9ad0c644-b539-5e83-a22e-35fc92391228
# ╠═f1873392-5be8-5055-9c30-fdbe75ee33ab
# ╠═a6d94647-e210-576c-9539-a0ae4159ad3f
# ╟─3aea3630-70c7-5d4f-ab87-489a1320cce4
# ╟─d8ad9d29-97f4-5325-8ec6-68943312f265
# ╟─e259dd7e-12c9-56b0-b77b-93c23dd38226
# ╠═0ffbdeda-bc29-5233-9b92-7f5d1f13573b
# ╠═1c62ec1e-8fd3-52a8-b84f-5610a8257098
# ╠═53b69f21-a0cd-5ce4-bede-baa7b0519223
# ╟─20f94825-7cc1-4883-987f-c28460713360
# ╟─23ab1779-eac6-5e29-af6d-d465174c5654
# ╟─d887e3ee-97ca-53ea-8ecd-ee1a7b5b6ea0
# ╟─2ab149ac-24be-56fc-9c18-fc41227b7113
# ╠═10c68b00-24d3-58cd-9f5b-2b25c917fb5f
# ╠═28f3d388-5432-5c8b-83dd-17ff0ca8fdd6
# ╠═5ae61127-e981-5614-b7cc-49a51e5fa77e
# ╠═77c888f8-546b-55bd-8e6d-76a9c2068b16
# ╠═6002b833-7de7-5e06-a753-277aab945c0e
# ╠═36ead4eb-6aa3-5a38-82f0-7427bcd7c338
# ╠═be6ef043-75d4-59cf-b172-491e2200a90a
# ╟─f367d731-fe78-4f9f-964d-b182faf883a9
# ╟─81e9027d-a62a-5a0c-a091-f587d7c0a121
# ╟─0f0250c8-45d8-56d7-b1e5-f15bcf7a6714
# ╟─5c7fff9b-6630-5e79-a499-efc14c14e3bc
# ╠═6b66d29d-5dab-579c-a5d8-1233dc4f35b9
# ╟─1703f60a-9d85-5a57-962d-91c4f8b0bece
# ╟─d10caf51-f7a2-5154-8d24-e1ed46c4d86c
# ╠═ff9e2424-574b-5274-9c8f-adbad84450d6
# ╠═45043ecb-063f-5b53-bd1f-5b559c97199a
# ╟─74ed96cd-3bb4-4cf5-b1ca-d01c5ee9b0d2
# ╟─b8e5ce21-e91b-5d3a-a6eb-942d1fb42af8
# ╟─5ca60f2c-a230-5894-a8fa-1a5dcc16b64c
# ╠═2e1b5a19-af32-5b42-8f00-e79cbdc0fad1
# ╠═b247d175-c3ff-5561-98c7-bbc089fbf17a
# ╟─74850911-4736-46f0-8270-9c09bbe85f0a
# ╟─baaa15f4-add3-5d99-9695-3d6721e6d6f0
# ╟─47436151-80ce-52d6-856e-b75b78deee48
# ╟─26e1e385-4437-4022-9071-4da46fd02fb2
# ╟─33f76d9d-9a91-4c51-b08f-e100114f8181
# ╠═da3393ec-da8a-4e7a-9cb1-260090d7a201
# ╠═9d326916-1069-4e7b-bbdf-3a7814f72778
# ╠═5375ef36-11c4-43ac-8555-895ed05c3263
# ╟─c5265d77-13fe-4887-b27c-3cb2688d54bd
# ╟─e46cc64f-eeac-4161-b327-a23606fea015
# ╟─c48615a2-707b-5ccc-b824-acf128726517
# ╟─8b77445b-bc28-5204-89ef-89814e715f0d
# ╠═14c7a701-5529-58e5-bd40-1d6d46907e66
# ╠═fd3f9361-3413-50e2-8dca-dadab6f783b7
# ╠═1549c982-b5aa-54ba-80c0-000deeceece7
# ╠═95459b06-583d-5fe0-bf96-830980415543
# ╠═09fd9bf9-da85-5488-ac38-c1bd9bf15da8
# ╟─db084ca4-290d-5555-bba5-06a94a2b6348
# ╟─8c1258f9-0411-5633-807d-fa9809e266d0
