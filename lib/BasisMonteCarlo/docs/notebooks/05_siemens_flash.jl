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


# ╔═╡ 43087614-5ea9-557c-946e-81714d62f3b3
md"""
# The SOMATOM Definition Flash Detector in Geant4

The Siemens SOMATOM Definition Flash is a dual-source, energy-integrating CT scanner, cleared in
2008 and fitted since 2011 with Siemens' Stellar detector. Each of its two detectors is an array
of **UFC**, Siemens' "Ultra Fast Ceramic": gadolinium oxysulfide (Gd₂O₂S) sintered to a dense
ceramic and sawn into **dexels** (detector elements), with a reflector in the cuts and a
photodiode under every dexel. In front of the array stands an anti-scatter grid of plates
between the columns.

Up to and including **v0.15.1**, BasisSimulator describes how efficiently this detector turns
x rays into signal with one table, η(E) from 1 to 140 keV (`UFC_FLASH_MC_EFFICIENCY_LUT`),
computed with MCNP by Hamidreza Khodajou-Chokami; the inputs to that simulation were never
committed. This notebook builds the detector in Geant4, one piece at a time, and

1. **checks every piece against the documentation**: Siemens' data sheets, the FDA clearances
   and the literature;
2. **reproduces the v0.15.1 table**, which shows how it was made, and where it parts from what
   BasisSimulator v0.15.1 says the detector is;
3. **builds a better table** with its noise (the Swank factor), and saves it: the table
   BasisSimulator takes over from v0.16.0.

!!! note "Which BasisSimulator this notebook compares against"
    Every comparison with "the v0.15.1 table", and every "BasisSimulator v0.15.1" value (the
    density 7.44 g/cm³, the documented depth 1.0 mm, the fill factor), is with **BasisSimulator
    v0.15.1**, the last release before this notebook's table. BasisMonteCarlo keeps that table
    (`BASISSIMULATOR_V0_15_1_EFFICIENCY`, copied verbatim), so the comparisons stay the same
    after BasisSimulator moves to the new one. Only the x-ray spectra and the bowtie
    (sections 8–10) come from the **installed** BasisSimulator, whatever its version; nothing
    here pins BasisSimulator, or BasisMonteCarlo, to a version.
"""

# ╔═╡ 522577f8-9416-4099-ac48-93e0abeae0cc
let
    Mke.set_theme!(THEME)
    night, ink, sub = MATERIALS.night, Mke.RGBf(0.93, 0.94, 0.96), Mke.RGBf(0.6, 0.63, 0.68)
    fig = Mke.Figure(size = (1500, 600), backgroundcolor = night, figure_padding = (10, 10, 10, 10))

    # (1) The gantry, to scale: two tubes 95° apart, each with its detector arc
    ax1 = canvas(fig[1, 1]; background = night)
    R_iso, R_det = flash.source_to_isocenter_mm, flash.source_to_detector_mm
    θ = range(0, 2π; length = 200)
    Mke.poly!(ax1, Mke.Point2f.(820 .* cos.(θ), 820 .* sin.(θ)); color = MATERIALS.night_ring)
    Mke.poly!(ax1, Mke.Point2f.(390 .* cos.(θ), 390 .* sin.(θ)); color = night)
    Mke.lines!(ax1, 640 .* cos.(θ), 640 .* sin.(θ); color = (:white, 0.06), linewidth = 1)
    # a patient: a water ellipse with a spine and two lungs
    Mke.poly!(ax1, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = Mke.RGBf(0.42, 0.45, 0.5))
    for x in (-75, 75)
        Mke.poly!(ax1, Mke.Point2f.(x .+ 50 .* cos.(θ), 10 .+ 60 .* sin.(θ)); color = Mke.RGBf(0.16, 0.18, 0.21))
    end
    Mke.poly!(ax1, Mke.Point2f.(22 .* cos.(θ), -105 .+ 20 .* sin.(θ)); color = Mke.RGBf(0.85, 0.85, 0.82))
    function tube!(angle_deg, half_fan_deg, alpha, label)
        a = deg2rad(angle_deg)
        src = Mke.Point2f(R_iso * cos(a), R_iso * sin(a))
        toward = -Mke.Point2f(cos(a), sin(a))
        φ = range(-deg2rad(half_fan_deg), deg2rad(half_fan_deg); length = 80)
        rot(v, t) = Mke.Point2f(v[1] * cos(t) - v[2] * sin(t), v[1] * sin(t) + v[2] * cos(t))
        arc = [src + R_det * rot(toward, t) for t in φ]
        Mke.poly!(ax1, Mke.Point2f.(vcat([src], arc)); color = (MATERIALS.beam, alpha))
        inner = [src + (R_det + 8) * rot(toward, t) for t in φ]
        outer = [src + (R_det + 50) * rot(toward, t) for t in φ]
        Mke.poly!(ax1, Mke.Point2f.(vcat(inner, reverse(outer))); color = Mke.RGBf(0.78, 0.82, 0.88))
        Mke.poly!(ax1, Mke.Rect2f(src[1] - 45, src[2] - 45, 90, 90); color = Mke.RGBf(0.8, 0.83, 0.88))
        Mke.scatter!(ax1, [src]; color = MATERIALS.beam, markersize = 12)
        Mke.text!(ax1, src + Mke.Point2f(0, 85); text = label, color = ink, fontsize = 17, font = :bold, align = (:center, :center))
        return nothing
    end
    tube!(90, half_fan_deg, 0.22, "Tube A")
    tube!(90 + 95, rad2deg(asin(165.0 / R_iso)), 0.14, "Tube B")
    Mke.limits!(ax1, -1010, 1010, -1010, 1010)

    # (2) A patch of the detector under its grid, to scale, the near corner cut away
    ax2 = canvas(fig[1, 2]; background = night)
    p, s, d = pitch_mm, septum_mm, depth_mm
    nx, ny = 7, 5
    cam = camera(-52, 28)
    blocks = dexel_blocks(nx, ny, p, s, 0.0, d)
    walls = filter(grid_blocks(nx, ny, p, grid.wall_mm, d, d + grid.height_mm)) do b
        !(b.lo[1] > 2.4p[1] && b.lo[2] < 3.4p[2])
    end
    append!(blocks, walls)
    rays = map([(3.5, 0.5), (5.5, 1.5), (4.5, 2.5)]) do (i, j)
        c = (i * p[1], j * p[2])
        a, b = project(cam, (c..., d + 12.5)), project(cam, (c..., d + 0.15))
        (dot3((c..., d), cam.toward) + 1, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 12))
    end
    draw_blocks!(ax2, cam, blocks; extras = rays, edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    # (3) One dexel's stack, taken apart
    ax3 = canvas(fig[1, 3]; background = night)
    draw_blocks!(ax3, camera(-56, 30), exploded_detector(1, 1; gap = 0.75).blocks; edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    for (k, (title, subtitle)) in enumerate(
            (
                ("Two Tubes, Two Detectors", "The SOMATOM Definition Flash, to Scale: Tubes 95° Apart"),
                ("Dexels Under a 1-D Grid", "UFC Ceramic, White Reflector, Tungsten Plates"),
                ("One Dexel, Taken Apart", "Grid, Ceramic, Photodiode, ADC, Board"),
            )
        )
        Mke.Label(fig[2, k], title; color = ink, fontsize = 26, font = :bold, tellwidth = false)
        Mke.Label(fig[3, k], subtitle; color = sub, fontsize = 18, tellwidth = false)
    end
    Mke.rowgap!(fig.layout, 1, 8)
    Mke.rowgap!(fig.layout, 2, 4)
    fig
end

# ╔═╡ 4c99acaf-d4fa-59d5-819d-dc4b474284b5
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a UFC dexel does not count at all. Every
x ray it absorbs leaves energy in the ceramic, which glows; the reflector keeps the light inside
the dexel, and the photodiode under it adds up the light of every photon over the reading. The
signal is therefore proportional to the **energy** left in the dexel, and the efficiency that
matters is the energy-absorption efficiency: the mean energy a dexel collects per photon,
divided by the photon's energy.

Since 2012 the Flash has carried Siemens' **Stellar** detector, the first with the photodiode and
the analogue-to-digital converter (ADC) in one chip (*"the second generation Dual Source (SOMATOM
Definition Flash) had the first integrated detector, which was called the Stellar detector"*,
Force white paper p. 11; the chip itself: Siemens' Stellar detector white paper), as drawn below.
Flashes built before then had the conventional design: the photodiodes, then the ADCs on a
separate electronics board. Neither is simulated (the signal is taken to be proportional to the
energy left in the ceramic), so the difference does not touch the efficiency; the Stellar
generation's lower electronic noise is outside this notebook.
"""

# ╔═╡ 060ceab7-f004-4019-a677-fbe33b15df82
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
        key in (:board, :substrate) && continue
        outline!(ax, cam, (i * p[1], j * p[2], z0), ((i + 1) * p[1], (j + 1) * p[2], z1); linewidth = 3.5)
    end
    Mke.Label(
        fig[2, 1],
        "One Dexel, Outlined in Amber: One Block of Ceramic, Its Photodiode and Its ADC Channel, Between Two Grid Plates";
        font = :bold,
        fontsize = 17,
        color = Mke.RGBf(0.8, 0.5, 0.05),
        halign = :left,
        tellwidth = false,
    )

    rows = (
        (:grid, "Anti-Scatter Grid", "1-D: Tungsten Plates Between the Columns\n(10 mm Tall; Drawn Short Here)", true),
        (:ceramic, "UFC Ceramic, Sawn into Dexels", "One Block of Gd₂O₂S per Dexel, 1.4 mm Deep;\nWhite TiO₂ Reflector Fills the Saw Cuts", true),
        (:photodiode, "Photodiode, One per Dexel", "Back-Illuminated Silicon: Light → Charge", false),
        (:cmos, "CMOS Wafer with the ADC (Stellar, 2012 On)", "Digitizes Each Dexel (20 Bit); Gold Vias\nCarry the Number Through the Wafer", false),
        (:substrate, "Ceramic Substrate", "Holds the Chips", false),
        (:board, "Circuit Board", "Carries Every Dexel's Number Out to the Gantry\n(No ADCs on It: Those Are in the Chip Above)", false),
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

# ╔═╡ 17331933-7602-4de6-bd2a-ed721d18cd59
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1400, 760))
    ax = canvas(fig[1, 1])

    # Layers, side view (x across, y up); heights in mm, the grid cut short
    y_board, y_sub, y_cmos, y_pd = -1.55, -1.2, -0.62, -0.16
    rect!(ax, -1.5p, y_board, 3p, 0.3, MATERIALS.board)
    rect!(ax, -1.5p, y_sub, 3p, 0.34, MATERIALS.substrate)
    rect!(ax, -1.5p, y_cmos, 3p, 0.42, MATERIALS.cmos)
    for i in -1:1
        x = i * p
        rect!(ax, x - (p - s) / 2, y_pd, p - s, 0.14, MATERIALS.photodiode)
        rect!(ax, x - 0.05, y_sub + 0.34, 0.1, 0.24, MATERIALS.gold)  # the via and stud bump
        rect!(ax, x - 0.22, y_cmos + 0.1, 0.44, 0.2, Mke.RGBf(0.32, 0.35, 0.42))  # the ADC's circuit
        Mke.text!(ax, x, y_cmos + 0.2; text = "ADC", color = :white, fontsize = 13, align = (:center, :center), font = :bold)
    end
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

    # (1) the x ray, (2) stops in the middle dexel, (3) light, (4) photodiode, (5) the ADC
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
    arrow!(ax, (0.0, y_cmos), (0.0, y_board - 0.35); color = PALETTE.ink, linewidth = 2, head = 12)
    Mke.text!(ax, 0.0, y_board - 0.45; text = "One Number per Dexel per Reading", align = (:center, :top), fontsize = 16, color = PALETTE.ink)

    # the five steps, labelled on the right
    entries = [
        (anchor = (0.12, d + 1.9), title = "1  An X Ray Arrives", detail = "Between Two of the Grid's Plates", color = Mke.RGBf(0.85, 0.5, 0.05), tag = nothing),
        (anchor = hit, title = "2  The Ceramic Stops It", detail = "Its Energy Is Left in the Dexel", color = Mke.RGBf(0.7, 0.55, 0.05), tag = nothing),
        (anchor = (half - 0.02, hit[2] + 0.28), title = "3  The Ceramic Glows", detail = "Light ∝ Energy Left; the White Reflector Keeps It In", color = Mke.RGBf(0.2, 0.6, 0.25), tag = nothing),
        (anchor = (0.4, y_pd + 0.07), title = "4  The Photodiode Collects the Light", detail = "Light → Charge, Summed over the Reading", color = MATERIALS.photodiode, tag = nothing),
        (anchor = (0.22, y_cmos + 0.2), title = "5  The ADC Digitizes the Charge", detail = "In the Same Chip, Right Underneath", color = PALETTE.ink, tag = nothing),
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

# ╔═╡ 157a12e0-b5b4-4548-bf56-1429d6a241d1
md"""
### Which Beam, and Why

How the photons arrive decides what a simulation measures, so each section says which beam it
uses. Five are used here, from the simplest to the most realistic:

| beam | what it measures | where |
|---|---|---|
| **pencil beam** into the centre of one dexel | how the v0.15.1 table was made (it is reproduced exactly) | section 5 |
| **uniform on one dexel's ceramic** | the efficiency per photon landing on the ceramic; reading every dexel around it gives the flat field (below) | sections 3–5 |
| **uniform on one cell** (ceramic and its reflector), under the grid | what the grid costs, straight and at an angle | section 6 |
| **flat field**: every dexel lit | what each dexel collects in a scan: what it loses to its neighbours it gets back from them. By symmetry this equals all the energy a uniform beam on one dexel leaves anywhere, which is how it is simulated | sections 4, 5 |
| **from the focal spot**, over the whole arc, with **clinical spectra** | everything at once: real angles, every dexel, the grid, the Flash's own 120 kV beam (also after 20 and 35 cm of water) | sections 7, 10 |

The efficiency table is built from single energies, because BasisSimulator weights it by whatever
spectrum a scan uses. Section 10 checks that this shortcut is exact: the clinical beams, simulated
photon by photon, give the same signal and noise as the table weighted by their spectra.
"""

# ╔═╡ 321e4165-9171-4b97-86ff-19171fe3eb71
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1700, 560))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        ("Pencil Beam", "Into the Centre of One Dexel:\nHow the v0.15.1 Table Was Made", :pencil),
        ("Uniform on One Dexel", "Over Its Ceramic; Every Dexel\nRead Out (Sections 3–5)", :ceramic),
        ("Uniform on One Cell", "Ceramic and Its Reflector,\nUnder the Grid (Section 6)", :cell),
        ("Flat Field", "Every Dexel Lit, as in a Scan:\nWhat Each Dexel Collects", :flat),
        ("From the Focal Spot", "The Whole Arc, Real Angles;\nClinical Spectra (Sections 7, 10)", :focal),
    )
    for (k, (title, detail, kind)) in enumerate(cases)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        blocks = dexel_blocks(3, 3, p, s, 0.0, d)
        if kind === :cell
            append!(blocks, grid_blocks(3, 3, p, grid.wall_mm, d, d + 1.6))
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

# ╔═╡ 3ae5f27e-e5cb-5c6b-bf30-814e26acb31e
md"""
## The SOMATOM Definition Flash, as Documented

Every hardware number this notebook uses, checked against Siemens' documents, the FDA 510(k)
summaries and the literature. Where BasisSimulator v0.15.1 and a source disagree the row is marked
**⚠ discrepancy**; where no source gives a number the row says so, and the notebook states its
assumption. Far less is published about the Flash's detector than about the Force's.

| quantity | BasisSimulator v0.15.1 | documented | status | source |
|---|---|---|---|---|
| scintillator | UFC, Gd₂O₂S | UFC, "available across all Siemens CT scanners"; dopants proprietary, below 0.1 % | ✓ | trade press quoting Siemens (2012); UFC safety data sheet §3 |
| density | 7.44 g/cm³ (its documentation: 7.34) | 7.29–7.33 g/cm³ (99.95 % of theoretical) | **⚠ discrepancy**; 7.31 used here | UFC data sheet §1.4.1 |
| crystal depth | 1.0 mm | not published; 1.4 mm is the only UFC thickness Siemens quotes | not confirmed (no source at all) | UFC data sheet §1.1.1; Detectors & Crystals p. 19 |
| its efficiency table | η(140 keV) = 0.588 | 1.0 mm of UFC can absorb at most about 0.56 of 140 keV photons | **⚠ discrepancy** (section 5) | this notebook |
| rows × row width | 64 × 0.6 mm | 64 rows, 38.4 mm | ✓ | K133589 (comparison table); K190578 Table 5 |
| columns | 736 | 736 (1472 measuring channels with the flying focal spot) | ✓ | K133589 |
| column width at the isocentre | 0.70473 mm | not published | not confirmed | – |
| source–isocentre, source–detector | 595, 1085.6 mm | not published | not confirmed | – |
| reflector between dexels | not modelled (fill factor 0.9 × 0.9) | "a special reflecting polymer"; width not published | assumed: 0.1 mm of TiO₂-loaded epoxy, which gives the same fill factor | Detectors & Crystals p. 39 |
| anti-scatter grid | not modelled | one-directional (the two-dimensional grid arrived with the Force); material and dimensions not published | built here: tungsten plates between columns, 80 µm thick, 10 mm tall (assumed) | Force white paper p. 11 |
| beam quality | – | no published half-value layer found | – | – |
| FDA clearances | – | K082220 (2008); K113342, with the Stellar detector (2011) | ✓ | FDA |

Sources: Siemens Healthineers, [UFC data sheet](https://www.siemens-healthineers.com/computed-tomography/ufc),
[Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf),
[UFC safety data sheet](https://marketing.webassets.siemens-healthineers.com/1800000004195975/23a630dc20e0/DI_CT_datasheet_Safety-UFC_en_1800000004195975.pdf);
Flohr et al., [SOMATOM Force white paper](https://ctm2020.shpci.org/data/CT_SOMATOM-Force_Whitepaper_DualSource_HOOD05162002908728_152104378.pdf) (on one- and two-directional grids);
FDA [K082220](https://www.accessdata.fda.gov/cdrh_docs/pdf8/K082220.pdf), [K113342](https://www.accessdata.fda.gov/cdrh_docs/pdf11/K113342.pdf),
[K133589](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133589.pdf);
[DAIC 2012](http://www.dicardiology.com/product/somatom-definition-flash-ct-now-equipped-fda-cleared-stellar-detector);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ dc8fc5c4-937e-55d4-98bd-2f48387c7da0
md"""
## Notebook Setup
"""

# ╔═╡ aae12472-9dce-5e25-ad8a-8e4e7c67f16d
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 5fe29898-f393-5a55-9bb6-c6c073ee7066
using Markdown: @md_str, Markdown

# ╔═╡ ee624c58-008c-5e31-95d8-872303dc3a3a
import PlutoUI

# ╔═╡ 1d95c3a9-6ecd-5c3c-94a0-da9bdc1b345e
import BasisMonteCarlo as MC

# ╔═╡ 4eff9487-c35d-5f43-a940-3732dd6c7333
import CairoMakie as Mke

# ╔═╡ 2cd1697f-e9e4-5073-bc81-89a8d6fe1d59
import BasisSimulator as BS

# ╔═╡ 32170bf5-5efc-55fe-a925-b5a05d93fcdc
import Serialization

# ╔═╡ c13b2b79-71d3-5d5a-b407-8e7cdb518d5b
import DelimitedFiles

# ╔═╡ be0ade99-9563-5db7-bf09-e72d3fa36cf2
using Statistics: mean

# ╔═╡ 9e62282b-bb3d-58e1-b2cd-1f485feee4df
PlutoUI.TableOfContents()

# ╔═╡ 5ff14064-4c4f-5400-929d-91426e2ee31f
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ fa94fde5-ae10-5f3a-ae6a-b470df2fa380
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
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator v0.15.1's table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ 6604eb4b-4571-5cf6-9522-7749a6b5018a
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

# ╔═╡ da837658-54a1-5f7f-b36b-df754e2c5081
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

# ╔═╡ c2f6f29a-dc0e-5a6a-a27e-be4d8a15a61e
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

# ╔═╡ 9df857f1-00f6-4ea8-b572-ef8bb1d72f62
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses (the pitch, the reflector,
the crystal's depth, the grid), with a small kit: blocks in 3-D, seen from any direction, shaded
by one light and drawn back to front. The colours are the materials': pale blue UFC ceramic, white
TiO₂ reflector, grey tungsten, dark-blue silicon photodiodes, amber x rays.
"""

# ╔═╡ 70e9e722-217b-4aa8-a54c-0ee8f2e37c3f
MATERIALS = (
    ufc = Mke.RGBf(0.74, 0.84, 0.93),  # the UFC ceramic
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
    light = Mke.RGBf(0.45, 0.9, 0.45),  # UFC's light (it glows green)
    night = Mke.RGBf(0.05, 0.06, 0.08),  # the dark backdrop of the lab's banner
    night_ring = Mke.RGBf(0.11, 0.13, 0.16),
)

# ╔═╡ 8aced4bd-0255-4e09-bba0-b6a684c2cf7b
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

# ╔═╡ fb112cc1-82b1-4c65-9613-469c2a06462c
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

# ╔═╡ 2af0f891-185b-4537-9480-bf9feea3443e
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ 6b97c87e-9c2d-4c05-b335-e8444197f473
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ 6e25f6ed-1671-4479-85b2-150dce07afaf
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

# ╔═╡ 9bdfaf44-367b-419f-bc91-11e62627b792
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ 2420f6e7-9bdb-4f66-85a8-58dac88967c2
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

# ╔═╡ e60344bc-3308-4b38-83c8-ad7899fcd81b
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

# ╔═╡ 2efaaa99-1dc5-4b41-8a19-a58a814936b5
"""
The plates of the Flash's 1-D grid over an nx × ny array (`walls = :across`: between the columns
only), or a 2-D grid's walls (`walls = :both`), in pieces one cell long, from z0 to z1.
"""
function grid_blocks(nx, ny, pitch, wall, z0, z1; color = MATERIALS.tungsten, walls = :across)
    blocks = Block[]
    p, w = pitch, wall
    for i in 0:nx, j in 0:(ny - 1)
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x - w / 2, y + w / 2, z0), (x + w / 2, y + p[2] - w / 2, z1), color))
    end
    if walls === :both
        for i in 0:(nx - 1), j in 0:ny
            x, y = i * p[1], j * p[2]
            push!(blocks, Block((x + w / 2, y - w / 2, z0), (x + p[1] - w / 2, y + w / 2, z1), color))
        end
    end
    for i in 0:nx, j in 0:ny
        x, y = i * p[1], j * p[2]
        push!(blocks, Block((x - w / 2, y - w / 2, z0), (x + w / 2, y + w / 2, z1), color))
    end
    return blocks
end

# ╔═╡ 7d0641a4-9883-4e69-bc28-3838dde0a7fb
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

# ╔═╡ 317d1245-687c-439d-91e7-547fcc844313
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

# ╔═╡ 4f1eecf3-543a-43dd-a6b4-5c843b72e155
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

# ╔═╡ 3fb6787c-7210-4df8-843a-4edb2a74e7fe
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

# ╔═╡ f436da99-757f-4e92-8cb0-3a590224ee61
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ aab4d99a-83fa-456a-b52b-35f6f8e47d58
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ a01538e6-4fe6-432a-bc20-c8f9b368f065
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

# ╔═╡ 8ce5f74e-69d5-467e-939e-267a1ffcc1a0
"""A vertical bracket at `x` from `y0` to `y1`, with `text` to its left."""
function side_bracket!(ax, x, y0, y1, text; color, fontsize = 16, tick = 0.08)
    Mke.lines!(ax, Mke.Point2f[(x + tick, y0), (x, y0), (x, y1), (x + tick, y1)]; color, linewidth = 2)
    return Mke.text!(ax, x - tick, (y0 + y1) / 2; text, color, fontsize, font = :bold, align = (:right, :center), justification = :right)
end

# ╔═╡ 42387328-1bee-47fa-bcf8-8d28b1f310c8
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

# ╔═╡ 4630e698-1d3a-4936-bc81-c9fecdff60d4
TAGS = (geant4 = Mke.RGBf(0.2, 0.55, 0.32), not_simulated = Mke.RGBf(0.52, 0.52, 0.57))

# ╔═╡ f906a50e-31b6-4808-8f7f-3e1e6d845ce6
"""
The blocks of a patch of the Flash's detector, pulled apart layer by layer: board, ceramic
substrate, the CMOS wafer (with a gold via under every dexel), the photodiodes on it, the UFC
dexels in their reflector, the grid (drawn `grid_drawn` mm tall). Returns each layer's blocks
(`layers[key]`), its (bottom, top) (`levels[key]`), and all the blocks together (`blocks`).
"""
function exploded_detector(nx, ny; gap = 1.0, grid_drawn = 2.4)
    p, s, d = pitch_mm, septum_mm, depth_mm
    stack = (
        (key = :board, h = 0.3, gap = 0.5),
        (key = :substrate, h = 0.35, gap = 0.75),
        (key = :cmos, h = 0.4, gap = 0.0),
        (key = :photodiode, h = 0.14, gap = gap),
        (key = :ceramic, h = d, gap = gap),
        (key = :grid, h = grid_drawn, gap = 0.0),
    )
    layers = Dict{Symbol, Vector{Block}}()
    levels = Dict{Symbol, Tuple{Float64, Float64}}()
    z = 0.0
    for layer in stack
        z0, z1 = z, z + layer.h
        levels[layer.key] = (z0, z1)
        layers[layer.key] = if layer.key === :board
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.board)
        elseif layer.key === :substrate
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.substrate)
        elseif layer.key === :cmos
            vcat(
                layer_blocks(nx, ny, p, z0, z1, MATERIALS.cmos),
                [
                    Block((c[1] - 0.07, c[2] - 0.07, z0 - 0.75), (c[1] + 0.07, c[2] + 0.07, z0), MATERIALS.gold)
                        for c in (((i + 0.5) * p[1], (j + 0.5) * p[2]) for i in 0:(nx - 1) for j in 0:(ny - 1))
                ],
            )
        elseif layer.key === :photodiode
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.photodiode; inset = s / 2)
        elseif layer.key === :ceramic
            dexel_blocks(nx, ny, p, s, z0, z1)
        else
            grid_blocks(nx, ny, p, grid.wall_mm, z0, z1)
        end
        z = z1 + layer.gap
    end
    return (; layers, levels, blocks = reduce(vcat, values(layers)))
end

# ╔═╡ 136d5223-aefc-4bd6-8d9f-f7da9a90533f
"""A block drawn faded toward the background (`amount` 0: as is, 1: gone)."""
function faded(b::Block, amount; to = Mke.RGBf(1, 1, 1))
    mix(a, c) = a + amount * (c - a)
    color = Mke.RGBf(mix(b.color.r, to.r), mix(b.color.g, to.g), mix(b.color.b, to.b))
    return Block(; lo = b.lo, hi = b.hi, color, alpha = b.alpha, stroke = b.stroke)
end

# ╔═╡ 293b6771-66a9-4f4b-956a-2a48f426d21f
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

# ╔═╡ 6eddd832-8dab-44d7-ad21-e574b0ad732f
"""Outline a box (lo, hi) as seen by `cam`: the edge of its silhouette."""
function outline!(ax, cam, lo, hi; color = MATERIALS.beam, linewidth = 3, linestyle = :solid)
    corners = [(x, y, z) for x in (lo[1], hi[1]) for y in (lo[2], hi[2]) for z in (lo[3], hi[3])]
    return Mke.lines!(ax, hull([project(cam, c) for c in corners]); color, linewidth, linestyle)
end

# ╔═╡ 04985f30-4003-4e13-953d-cf478f93caf3
"""
The master map, as a band across the top of a section's first illustration: the gantry, one
dexel's stack with the part under study (`focus`) lit and outlined, and the dexel's five steps with
this one's lit. `focus` is one of `:tube`, `:arc`, `:grid`, `:ceramic`, `:photodiode`, `:cmos`,
`:whole`. Place it with `locator!(fig[0, :], focus; note)`.
"""
function locator!(position, focus; note = "")
    amber, dim = MATERIALS.beam, Mke.RGBf(0.82, 0.83, 0.86)
    Mke.Box(position; color = Mke.RGBf(0.965, 0.965, 0.975), strokevisible = false, cornerradius = 14)
    layout = Mke.GridLayout(position; halign = :center, tellwidth = false)

    # The gantry, small: tube A, its fan, the detector arc, the one dexel below
    ax = canvas(layout[1, 1]; width = 230, height = 230, backgroundcolor = :transparent)
    θ = range(0, 2π; length = 120)
    R_iso, R_det = flash.source_to_isocenter_mm, flash.source_to_detector_mm
    lit(k) = focus === k || focus === :whole
    Mke.lines!(ax, 800 .* cos.(θ), 800 .* sin.(θ); color = dim, linewidth = 1.5)
    Mke.poly!(ax, Mke.Point2f.(170 .* cos.(θ), 120 .* sin.(θ) .- 20); color = (PALETTE.muted, 0.25))
    src = Mke.Point2f(0, R_iso)
    φ = range(-deg2rad(half_fan_deg), deg2rad(half_fan_deg); length = 60)
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
        :cmos => (:cmos,),
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
        (:cmos, "5  The ADC"),
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

# ╔═╡ 695af4b7-fd89-45a9-a35a-b01a80b5ed46
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

# ╔═╡ 5ada2e54-299f-4da9-bcbb-1a65c5f6a2a0
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

# ╔═╡ cc96f529-3e5f-4a88-a35a-757d5adf5b53
"""A Gd atom: nucleus and two shells, at `c` with outer radius `r`."""
function atom!(ax, c, r; color = Mke.RGBf(0.45, 0.47, 0.55))
    θ = range(0, 2π; length = 100)
    for (k, rr) in enumerate((0.55r, r))
        Mke.lines!(ax, c[1] .+ rr .* cos.(θ), c[2] .+ rr .* sin.(θ); color = (color, 0.55), linewidth = 1.5)
    end
    Mke.scatter!(ax, [Mke.Point2f(c)]; color, markersize = 18)
    return nothing
end

# ╔═╡ 9331e7c6-ad6c-5ca1-8a96-2d239fcd68f7
md"""
## 1. The Scintillator

UFC is gadolinium oxysulfide, Gd₂O₂S, with dopants (praseodymium, cerium) below 0.1 % by mass,
far too little to matter for x rays. Two numbers decide how it stops x rays: its composition and
its density. Siemens' data sheet gives **7.29–7.33 g/cm³**; BasisSimulator v0.15.1 uses 7.44 g/cm³ (its
documentation says 7.34), which is Geant4's and NIST's stock Gd₂O₂S rather than Siemens'
ceramic (⚠ *discrepancy* above).
This notebook builds the detector at **7.31 g/cm³**, and keeps 7.44 beside it to reproduce
the v0.15.1 table.
"""

# ╔═╡ 108ca11c-e55d-5f47-9770-60f3e5b802cf
# UFC at the density Siemens documents (7.31 g/cm³)
ufc = MC.DetectorMaterial(:UFC)

# ╔═╡ 8718ba53-cf7b-5a52-ad08-bce13c35b85d
# Gd₂O₂S at the density BasisSimulator v0.15.1 uses (7.44 g/cm³)
ufc_basis = MC.DetectorMaterial(:Gd2O2S)

# ╔═╡ 0c12fe24-c4b0-5bdd-b1da-ca6cda0649fd
# Gadolinium's K-shell: its edge and the energy of the x ray it emits (Kα)
gadolinium = only(MC.k_edges(ufc))

# ╔═╡ 2e6bd82e-2963-56c0-9d24-59952bf496b9
md"""
Gadolinium's **K-edge, 50.24 keV**, sits in the middle of every CT spectrum. Below it a photon
cannot free a K-shell electron; just above it, it can, and absorption jumps sevenfold. Every
such absorption leaves the atom to emit, nine times in ten, a **Gd K x ray of about 43 keV**,
which is just below the edge and so is absorbed only weakly: it can travel a millimetre or more
before stopping, which is the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ 002a21be-8ec7-596b-94f7-d3205e56673d
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

# ╔═╡ d5333d07-5574-5dba-834c-587102c2762a
mu_energies = 10.0:0.25:150.0

# ╔═╡ 5844ef71-4f68-5515-9b80-9d22e3360553
# Geant4's attenuation of UFC, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(ufc, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ b3faa435-5787-52c9-941b-6fcf7f4dad4f
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(ufc, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ a8708a37-3d1c-43b8-b1d1-e18dcbe9c409
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

    # How far a photon goes, on average, in UFC: 1/μ, against the crystal's 1.4 mm
    ax = Mke.Axis(
        fig[1:2, 4];
        title = "How Far a Photon Goes in UFC",
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

# ╔═╡ 9ef53632-c492-54e8-bbe0-c1ffcff7312d
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How Strongly UFC Stops X Rays (7.31 g/cm³)",
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

# ╔═╡ db66a77b-19df-5600-bd03-af9e287bfc9a
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

# ╔═╡ 61af1c5e-8856-567a-90ea-f767bf5fc11e
md"""
### A First Check: Siemens' Own Curve

Siemens publishes the fraction of x rays that **1.4 mm of UFC** absorbs, against energy
(*Detectors & Crystals*, p. 19). Its steps at the K-edge go *up*: it is the probability that a
photon interacts at all, `1 − exp(−μd)`, with no fluorescence escape. That depends only on the
material, its density and its depth, so at 1.4 mm it checks the material and density before any
simulation runs (the Flash's own depth comes next). The values were read from the vector figure,
a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ 3c0cabe0-2286-5d68-8d18-f9538ed090d1
# Siemens' curve, read from the vector figure (reference_data/siemens_ufc_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_ufc_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ 82e5e0b6-7679-55b0-b4b6-96db5d8ea994
# The depth of Siemens' figure
siemens_depth_mm = 1.4

# ╔═╡ 2b0afce5-3ccd-528c-b7ae-a8044d2d5683
# The probability a photon interacts in 1.4 mm of UFC, at Siemens' density and at BasisSimulator v0.15.1's
interaction_curves = (
    documented = [1 - exp(-MC.geant4_attenuation(ufc, E) * siemens_depth_mm) for E in mu_energies],
    basis = [1 - exp(-MC.geant4_attenuation(ufc_basis, E) * siemens_depth_mm) for E in mu_energies],
)

# ╔═╡ 693ba223-ad33-44f9-8ca2-ae9f51b12821
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
        Mke.text!(ax, 1.65, depth_mm / 2; text = "1.4 mm\nof UFC", align = (:left, :center), fontsize = 15, color = PALETTE.slab_edge)
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

# ╔═╡ 5781d012-225a-5557-afe3-620a360dd3ba
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Fraction of Photons 1.4 mm of UFC Absorbs",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons Absorbed",
        xticks = 20:20:150,
        limits = ((15, 152), (0.45, 1.04)),
    )
    Mke.lines!(
        ax,
        mu_energies,
        interaction_curves.documented;
        color = PALETTE.photon,
        linewidth = 3.5,
        label = "Geant4, 7.31 g/cm³ (Siemens' Density)",
    )
    Mke.lines!(
        ax,
        mu_energies,
        interaction_curves.basis;
        color = PALETTE.reference,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 7.44 g/cm³ (BasisSimulator v0.15.1)",
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
        label = "Siemens' Figure (± 0.025, Its Reading Accuracy)",
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

# ╔═╡ 73b50137-be3c-5db5-ab49-da03668bb584
let
    ours = [MC.interpolate_linear(collect(mu_energies), interaction_curves.documented, E) for E in siemens_curve.energy_keV]
    difference = siemens_curve.absorption .- ours
    low = siemens_curve.energy_keV .<= 90
    high = .!low

    # The depth that would give Siemens' value at each energy above 90 keV
    implied_depth = [
        -log(1 - a) / MC.geant4_attenuation(ufc, E)
            for (E, a) in zip(siemens_curve.energy_keV[high], siemens_curve.absorption[high])
    ]
    Markdown.parse(
        """
        Up to 90 keV Siemens' figure and Geant4 at 1.4 mm and 7.31 g/cm³ agree to within
        **$(round(maximum(abs, difference[low]); digits = 3))**, inside the figure's reading accuracy: the material and
        density are right. Above 90 keV the figure falls below the 1.4 mm curve, by up to
        **$(round(maximum(abs, difference[high]); digits = 3))** at $(Int(siemens_curve.energy_keV[high][argmax(abs.(difference[high]))])) keV; read
        literally, it corresponds to a layer $(round(minimum(implied_depth); digits = 2))–$(round(maximum(implied_depth); digits = 2)) mm deep. No source gives the
        Flash's own depth.
        """
    )
end

# ╔═╡ aa61390f-493b-564b-a4b5-b93a87b2737f
md"""
## 2. One Dexel and Its Neighbours

The array is sawn from one ceramic wafer in two directions and the cuts are filled with
reflector, so a dexel is one block of ceramic with reflector on its four sides. Its pitch at the
detector face is the Flash's dexel at the isocentre as BasisSimulator v0.15.1 has it, 0.70473 mm across
the fan by 0.6 mm along z, magnified by the source-to-detector over the source-to-isocentre
distance, 1085.6 / 595. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy, the
usual CT reflector, is assumed. The crystal's depth is not published either: BasisSimulator v0.15.1 says
1.0 mm, with no source; this notebook uses **1.4 mm**, the only UFC thickness Siemens quotes (the
reasons are under "The Table to Use"), and saves the table for other depths too.
"""

# ╔═╡ 05d18f32-ecb0-5777-940f-83aea74c9585
# The SOMATOM Definition Flash's geometry, as BasisSimulator v0.15.1 has it
flash = (
    source_to_isocenter_mm = 595.0,
    source_to_detector_mm = 1085.6,
    columns = 736,
    rows = 64,
    column_at_isocenter_mm = 0.70473,
    row_at_isocenter_mm = 0.6,
)

# ╔═╡ 5cbc9df5-f1aa-43ed-87e0-1d08fa008dee
# Half the fan, in degrees: the columns' width at the isocentre, seen from the focal spot
half_fan_deg = rad2deg(flash.columns * flash.column_at_isocenter_mm / flash.source_to_isocenter_mm / 2)

# ╔═╡ 818d4df5-9882-5fe8-8a97-22162cc76fde
# The crystal depth: 1.4 mm, the only UFC thickness Siemens quotes (see "The Table to Use")
depth_mm = 1.4

# ╔═╡ e0743f95-30f6-504b-9ee3-fa8bc89b64f3
# The depth BasisSimulator v0.15.1 documents for the Flash (no source gives it)
basis_depth_mm = 1.0

# ╔═╡ 0636b82f-c4db-5d58-94aa-81afc15b2bc5
# The dexel pitch at the detector face (across the fan, along z), in mm
pitch_mm = (flash.column_at_isocenter_mm, flash.row_at_isocenter_mm) .*
    (flash.source_to_detector_mm / flash.source_to_isocenter_mm)

# ╔═╡ 10b7f044-9ef7-5c94-b774-fed3968906f0
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ ab9d1d88-c53b-5c91-8027-2cada0a314c3
# The block of ceramic each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ a0d86cb6-f528-5523-8afb-9530c768d695
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 3fd947f2-6838-53ea-a46a-78685348e52b
let
    fill = crystal_mm ./ pitch_mm
    Markdown.parse(
        """
        The dexel pitch at the face is **$(round(pitch_mm[1]; digits = 3)) × $(round(pitch_mm[2]; digits = 3)) mm**, and
        the ceramic block inside it $(round(crystal_mm[1]; digits = 3)) × $(round(crystal_mm[2]; digits = 3)) mm: the crystal
        covers $(round(fill[1]; digits = 3)) × $(round(fill[2]; digits = 3)) of the face, the fill factor of 0.9 × 0.9 that BasisSimulator v0.15.1 uses.
        The assumed reflector and that fill factor are the same detector.
        """
    )
end

# ╔═╡ daa5074e-b919-5cb6-b45e-b1cea0c1258b
# The Geant4 detector: a slab of ceramic `depth` deep, cut into 13 × 13 dexels by reflector
# septa (ample for the x rays that leave a dexel), optionally under an anti-scatter grid
function dexel_array(depth; material = ufc, grid = nothing)
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

# ╔═╡ d5234118-fc2c-585a-a56f-567d72908a9c
# Photons landing uniformly on one dexel's ceramic, and on its whole cell (ceramic and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ a0a407f6-6b5c-5b35-9e1a-362d05bb0ca2
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ aaf055ae-9730-591e-b076-2b1a87fdda17
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

# ╔═╡ 6655624b-23e3-5403-8220-dbd09038bfbd
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

# ╔═╡ a5e2bfee-aac8-57ee-b273-54fb5ecc15f2
md"""
### What a Photon Does in the Ceramic

`record_tracks` follows a few hundred 60 keV photons landing on the central dexel, every step of
every particle. Two of them are drawn: one whose Gd x ray is absorbed **in the next dexel**, so
its energy is split between two photodiodes, and one whose Gd x ray **leaves through the front
face**, so its 43 keV are lost to every dexel. Both happen often, because the Gd x ray is so
weakly absorbed.
"""

# ╔═╡ c1346985-186d-58ed-8d05-8a664a33ef4f
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    60.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ f8589d48-c9c7-571d-8b8d-9c3ea330604f
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

# ╔═╡ 22714c6e-01d7-5eb4-be91-8b8e68d7f036
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
        xlabel = "X, Across the Fan (mm)",
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

# ╔═╡ be0d59f2-e19b-44b0-af6d-bdb7dd2e355e
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

Tungsten plates through the ceramic would stop the x rays (across the fan; the Flash's grid has none along z), but they would widen the cuts (less
ceramic per dexel, so fewer photons stopped for the same dose), add tungsten's own x rays (59 and
67 keV) inside the ceramic, and have to be built into a sawn ceramic wafer. And in a scan the
x-ray crosstalk mostly cancels: a dexel gets back from its neighbours what it loses to them
(section 4). What it does cost, a little sharpness and a little noise (the Swank factor), is
measured here. So the grid stands in front, against the scatter from the patient, and only the
reflector goes between the dexels.
"""

# ╔═╡ c5490cd0-fe64-44ee-8306-93f33b5b4131
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

    ax1 = canvas(fig[1, 1]; title = "The Flash: Grid Plates Above, White Reflector in the Saw Cuts", titlesize = 20)
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

# ╔═╡ f4f8ffd2-4418-5389-9c2f-098bfcc8d0b5
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is not one number but a
spread: the full photon energy when everything stays in the dexel, the energy minus 43 keV when
the Gd x ray leaves, a little when the photon only Compton-scatters once and leaves. Three
energies show the three regimes: 45 keV (below the K-edge), 60 keV (just above it) and 120 keV.
"""

# ╔═╡ a1c71e56-24db-5d77-913f-6326f04cace7
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

# ╔═╡ d31b0627-dba4-5469-b818-f3b3566c9c4d
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ 419873a9-c9de-4f49-9671-9c28b39f9f28
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

# ╔═╡ cd446a63-4adb-5f61-8de9-5d0e0adcf000
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

# ╔═╡ 02c6d1e4-6da3-591a-be8f-57f184c64cf2
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the ceramic or
leaves it: as the Gd x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of ceramic (no reflector), so that whatever
leaves the ceramic leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ 99c71afa-fb06-5c6f-ba03-32ed332b2712
budget_energies = 15.0:5.0:140.0

# ╔═╡ eddc2f24-22fb-5b53-9eb4-55d6099d4ea2
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(ufc, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ 3fe0e4c3-34d5-445d-96a8-a1df49943e96
let
    Mke.set_theme!(THEME)
    d = depth_mm
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1])
    rect!(ax, -6, -d, 12, d, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 0, -d - 0.12; text = "One Continuous Slab of UFC, $(d) mm Deep (No Reflector, So What Leaves Is Gone)", fontsize = 15, color = PALETTE.slab_edge, align = (:center, :top))
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

# ╔═╡ 19bfc4a0-f377-58c9-b348-c107b5956377
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
        title = "Where the Photon's Energy Goes: $(depth_mm) mm of UFC",
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

# ╔═╡ c5c4e668-d102-53d0-8083-3fea5ac2a72e
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

# ╔═╡ 0439842c-d207-5998-992b-a805548de5e4
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

# ╔═╡ d5e73ebf-d120-5bbb-81a5-bbace6b0672e
energies = 1.0:1.0:140.0

# ╔═╡ 9289c700-43d7-5aef-b9d2-f725563323e8
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ e3d686de-c192-5a51-9ba3-fc75ca707bf9
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ 30053ee8-f64f-43aa-ac23-17f29150dab5
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

# ╔═╡ 7327c893-ff25-50f0-912f-022e5486075d
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1400, 600))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Efficiency, $(depth_mm) mm of UFC",
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

# ╔═╡ 081b6107-7fab-5d8f-ab08-2b313210305a
let
    at(E) = findfirst(==(E), collect(energies))
    k60, k140 = at(60.0), at(140.0)
    beer = 1 - exp(-MC.geant4_attenuation(ufc, 140.0) * depth_mm)
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

# ╔═╡ 7833105e-c989-5a22-a1be-f90a30e20236
md"""
### A Check Against Published Monte Carlo

Boone et al. (Med. Phys. 1999) followed x-ray fluorescence in thin Gd₂O₂S screens with the TART 98
code: a pencil beam at normal incidence on a screen of Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S, the binder not counted), the energy tallied in a circle 0.1 mm across the
beam ("absorbed as primary") and in rings out to 4.5 mm ("lateral"), and ideal sensors in front of
the screen and behind it. At 55 keV they report (p. 908) how the incident energy divides. The same
screen and the same tallies in Geant4:
"""

# ╔═╡ 32c72144-66d9-5162-a408-c872b6e67442
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

# ╔═╡ 136d7bd4-9e91-4313-83b5-143c9554bcaa
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

# ╔═╡ 477b6af3-e9b6-50c2-ba44-771359ec9420
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

# ╔═╡ 350b821b-48cd-57be-bb2d-252054214409
md"""
## 5. Compared with BasisSimulator v0.15.1's Table

The v0.15.1 table is an energy-absorption efficiency (its K-edge step goes down, as a
fluorescence escape makes it), and BasisSimulator v0.15.1 documents the crystal as 1.0 mm deep. A
photon that never interacts leaves no energy, so the probability of interacting at all is a
ceiling on the efficiency, and at high energy the table reaches it.
"""

# ╔═╡ 8a971118-c3fc-5397-bc27-7f56b074b34e
# BasisSimulator v0.15.1's table (MCNP, Hamidreza Khodajou-Chokami), as BasisMonteCarlo keeps it
# for comparison: later BasisSimulator releases use this notebook's table instead
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash

# ╔═╡ a4b69209-db1a-47bc-8435-58654e47a56d
# Is BasisMonteCarlo's copy the table the installed BasisSimulator uses? Up to v0.15.1 it must be,
# exactly; from v0.16.0 BasisSimulator uses this notebook's table, and the copy stands alone
let
    installed = pkgversion(BS)
    same = isdefined(BS, :UFC_FLASH_MC_EFFICIENCY_LUT) &&
        BS.UFC_FLASH_MC_EFFICIENCY_LUT.energies == table.energies &&
        BS.UFC_FLASH_MC_EFFICIENCY_LUT.efficiency == table.efficiency
    message = if installed <= v"0.15.1"
        same ?
            "✓ The installed BasisSimulator, v$(installed), uses exactly this table." :
            "✗ **The installed BasisSimulator, v$(installed), has a different table from BasisMonteCarlo's copy of v0.15.1's.** Check `BASISSIMULATOR_V0_15_1_EFFICIENCY` against BasisSimulator v0.15.1."
    else
        "The installed BasisSimulator is v$(installed), which no longer uses this table. The comparisons below are with BasisSimulator v0.15.1's table, which BasisMonteCarlo keeps (`BASISSIMULATOR_V0_15_1_EFFICIENCY`): they do not change with BasisSimulator's version."
    end
    Markdown.parse(message)
end

# ╔═╡ dd7879b9-406a-5760-84b8-73eb1b6a367e
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ 46d5bbd9-c8de-4ebe-8d36-77792b7308e4
let
    μ_per_mm(E) = MC.geant4_attenuation(ufc_basis, E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 850))

    N = 30
    E = 140.0
    μ = μ_per_mm(E)
    for (k, (depth, title)) in enumerate(((basis_depth_mm, "$(basis_depth_mm) mm: BasisSimulator v0.15.1's Documented Depth"), (fit.depth_mm, "$(fit.depth_mm) mm: the Depth the v0.15.1 Table Fits")))
        stopped = 1 - exp(-μ * depth)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        rect!(ax, 0, 0, 1.6, depth, MATERIALS.ufc; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        for i in 1:N
            x = 1.6 * (i - 0.5) / N
            z = -log(1 - mod(i * 0.618034, 1.0)) / μ
            if z < depth
                Mke.lines!(ax, [x, x], [depth + 0.55, depth - z]; color = MATERIALS.beam, linewidth = 2)
                Mke.scatter!(ax, [x], [depth - z]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (x, depth + 0.55), (x, -0.5); color = (MATERIALS.beam, 0.6), linewidth = 2, head = 10)
            end
        end
        Mke.text!(ax, 1.68, depth / 2; text = "$(round(Int, 100stopped)) % of\n140 keV Photons\nInteract", align = (:left, :center), fontsize = 16, color = PALETTE.ink)
        Mke.limits!(ax, -0.05, 3.3, -0.62, max(basis_depth_mm, fit.depth_mm) + 0.65)
    end
    Mke.Label(fig[2, 1:2], "At 140 keV the v0.15.1 Table Says $(round(table_at(140.0); digits = 3)): More Than $(basis_depth_mm) mm Can Absorb at All, Since a Photon That Never Interacts Leaves No Energy."; fontsize = 16, color = PALETTE.muted)
    locator!(fig[0, :], :ceramic; note = "Step 2 · How Deep\nIs the Ceramic?")
    fig
end

# ╔═╡ 4b513a3a-e153-5474-b80f-ea0a83aaadd6
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(ufc_basis, e) * basis_depth_mm) for e in E]
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "BasisSimulator v0.15.1's Table Against 1.0 mm of UFC",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.06)),
    )
    Mke.band!(ax, E, ceiling, fill(1.06, length(E)); color = (PALETTE.electron, 0.08))
    Mke.lines!(
        ax,
        E,
        ceiling;
        color = PALETTE.electron,
        linewidth = 2,
        linestyle = :dot,
        label = "Ceiling: 1.0 mm Interacts at All (7.44 g/cm³)",
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 3.5,
        label = "BasisSimulator v0.15.1 (MCNP)",
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
    note!(
        ax,
        100,
        1.0,
        "Impossible for 1.0 mm";
        align = (:center, :center),
        color = PALETTE.electron,
        fontsize = 15,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 0a61ea56-fdd2-57d4-8bde-f2490f63401b
let
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(ufc_basis, e) * basis_depth_mm) for e in E]
    over = E[table_at.(E) .> ceiling]

    # The energies above the ceiling, as runs of consecutive keV
    runs = String[]
    first_in_run = first(over)
    for k in eachindex(over)
        if k == lastindex(over) || over[k + 1] != over[k] + 1
            push!(
                runs,
                first_in_run == over[k] ? "$(Int(over[k]))" : "$(Int(first_in_run))–$(Int(over[k]))",
            )
            k < lastindex(over) && (first_in_run = over[k + 1])
        end
    end
    Markdown.parse(
        """
        **⚠ Discrepancy.** At $(join(runs, " and ")) keV, the v0.15.1 table is above the fraction of photons 1.0 mm
        of Gd₂O₂S stops at all, even at its own density (at 140 keV: $(round(table_at(140.0); digits = 3)) against a ceiling of
        $(round(last(ceiling); digits = 3))). The excess is small, but no 1.0 mm crystal can give it. Which crystal does the
        table describe?
        """
    )
end

# ╔═╡ fca7e15c-bf53-5b4e-b12b-76610eeffcc8
md"""
### Which Crystal Did the Table Describe?

The table's shape fixes how it was made. A pencil beam into the centre of one dexel, tallied in
that dexel, gives its deep step at the K-edge (with a uniform beam or a flat field the step is
shallower); the fall-off at high energy then fixes the depth. So: pencil beams at 60–140 keV into
dexels 0.6 to 2.4 mm deep, at BasisSimulator v0.15.1's density, compared with the table, with one
constant factor for the table's flat 1 % loss at low energy.
"""

# ╔═╡ 5383f381-53af-59aa-9564-bc9cc599215d
scan_depths = 0.6:0.1:2.4

# ╔═╡ e5093aeb-4ea9-5061-8652-853408cd9871
scan_energies = 60.0:5.0:140.0

# ╔═╡ 03dbc810-873f-5ccd-880c-0457715d5780
# One column per depth, one row per energy
pencil_scan = reshape(
    MC.simulate_sweep(
        [
            (dexel_array(depth; material = ufc_basis), E)
                for depth in scan_depths for E in scan_energies
        ];
        n = 50_000,
        beam = MC.PencilBeam(; start_mm = 12.0),
        seed = 13,
        readout,
    ),
    length(scan_energies),
    length(scan_depths),
)

# ╔═╡ 96c4845c-15f3-54a4-ae96-654e74c5b90f
# How far one depth's pencil-beam efficiencies are from the table, after the best constant factor
function misfit(runs, target)
    ours = MC.dexel_efficiency.(runs)
    scale = sum(ours .* target) / sum(ours .^ 2)
    rms = sqrt(sum(abs2, scale .* ours .- target) / length(target))
    return (scale = scale, rms = rms)
end

# ╔═╡ 69df791a-942d-56f6-b730-3cdddbfb7477
fit = let
    target = table_at.(scan_energies)
    misfits = [misfit(pencil_scan[:, k], target) for k in eachindex(scan_depths)]
    best = argmin([m.rms for m in misfits])
    (depth_mm = scan_depths[best], scale = misfits[best].scale, misfits = misfits)
end

# ╔═╡ dbda913a-b1dd-5500-8568-98e38bcca95c
let
    Mke.set_theme!(THEME)
    rms = [m.rms for m in fit.misfits]
    fig = Mke.Figure(size = (1200, 500))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Which Crystal Depth Reproduces the v0.15.1 Table?",
        xlabel = "Crystal Depth in the Simulation (mm)",
        ylabel = "RMS Difference from the Table\n(60–140 keV)",
        yscale = log10,
        limits = (nothing, (1.0e-3, 0.5)),
    )
    Mke.vlines!(
        ax,
        basis_depth_mm;
        color = PALETTE.reference,
        linestyle = :dash,
        linewidth = 2,
    )
    Mke.scatterlines!(
        ax,
        collect(scan_depths),
        rms;
        color = PALETTE.photon,
        markersize = 11,
    )
    note!(
        ax,
        fit.depth_mm,
        minimum(rms),
        "Best: $(fit.depth_mm) mm";
        align = (:center, :top),
        offset = (0, -12),
        color = PALETTE.photon,
        fontsize = 16,
    )
    note!(
        ax,
        basis_depth_mm,
        0.3,
        "BasisSimulator v0.15.1 Documents 1.0 mm";
        align = (:left, :center),
        offset = (8, 0),
        color = PALETTE.reference,
        fontsize = 15,
    )
    fig
end

# ╔═╡ b2bd0487-6f90-512e-807a-030292a2b6a4
# The table's recipe at the depth that fits, at every energy
reproduction = MC.simulate_sweep(
    [(dexel_array(fit.depth_mm; material = ufc_basis), E) for E in energies];
    n = 100_000,
    beam = MC.PencilBeam(; start_mm = 12.0),
    seed = 15,
    readout,
)

# ╔═╡ 8f1937c0-4cc6-5aec-8d82-b941b761f267
reproduced = fit.scale .* MC.dexel_efficiency.(reproduction)

# ╔═╡ 14c6dbf6-b4c3-5ddf-8ce8-95f748137025
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 640))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Table, and Geant4's Pencil Beam into One Dexel at $(fit.depth_mm) mm (× $(round(fit.scale; digits = 3)))",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.4, 1.03)),
    )
    Mke.lines!(
        ax,
        E,
        table_at.(E);
        color = PALETTE.reference,
        linewidth = 5,
        label = "BasisSimulator v0.15.1 (MCNP)",
    )
    Mke.lines!(
        ax,
        E,
        reproduced;
        color = PALETTE.photon,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, Reproduced",
    )
    Mke.axislegend(ax; position = :lb)
    axr = Mke.Axis(
        fig[2, 1];
        xlabel = "Photon Energy (keV)",
        ylabel = "Geant4 − Table",
        xticks = 0:20:140,
        limits = ((0, 141), (-0.04, 0.04)),
    )
    Mke.hspan!(axr, -0.01, 0.01; color = (:black, 0.06))
    Mke.hlines!(axr, 0; color = PALETTE.ink, linewidth = 1)
    Mke.lines!(axr, E, reproduced .- table_at.(E); color = PALETTE.photon, linewidth = 2)
    Mke.linkxaxes!(ax, axr)
    Mke.hidexdecorations!(ax; grid = false)
    Mke.rowsize!(fig.layout, 2, Mke.Relative(0.25))
    fig
end

# ╔═╡ c1e171af-68b9-58ca-919f-af9a2efe31ba
let
    E = collect(energies)
    difference = reproduced .- table_at.(E)
    typical = sum(abs, difference) / length(difference)
    siemens = abs(fit.depth_mm - 1.4) <= 0.1 ?
        " 1.4 mm is the one UFC thickness Siemens quotes." : ""
    Markdown.parse(
        """
        The table is reproduced to **$(round(typical; digits = 4))** on average (largest difference
        $(round(maximum(abs, difference); digits = 3)), at $(Int(E[argmax(abs.(difference))])) keV) by a pencil beam into the centre of one
        dexel, tallied in that dexel, in Gd₂O₂S at 7.44 g/cm³ **$(fit.depth_mm) mm deep**, times $(round(fit.scale; digits = 3)).
        **⚠ Discrepancy:** BasisSimulator v0.15.1 labels this table as the Flash's 1.0 mm detector.$(siemens) The table also
        misses what a CT scan sees in two ways: one dexel under a pencil beam, rather than the flat field (section 4),
        and no fall-off from photons landing near the dexel's edge.
        """
    )
end

# ╔═╡ 980f18e5-1050-580f-9439-39127dd9654b
md"""
## 6. The Anti-Scatter Grid

The Flash's grid is **one-directional**: plates between the columns, pointing at the focal spot,
which stop photons arriving at an angle across the fan but not along z. Siemens' Force white
paper says as much (*"most medical CT scanners use one-directional collimation"*), presenting the
Force's two-directional grid as new. Its material and dimensions are not published. Built here:
**tungsten plates 80 µm thick** (the preferred value of Siemens' patent for grid walls,
US 2011/0019801) and **10 mm tall** (assumed), one per column, standing on the reflector. 80 µm
plates on 100 µm of reflector shade no ceramic.
"""

# ╔═╡ 722c2410-b3a2-5325-a3bd-83a737169149
grid = MC.Collimator(;
    height_mm = 10.0,
    wall_mm = 0.08,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :x,
    cells = 13,
)

# ╔═╡ a5980443-b05b-5f82-ae79-3c535ee866b2
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    H, w = grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1600, 1050))


    # (a) to scale, the near corner cut away
    ax = canvas(fig[1, 1]; title = "(a) The 1-D Grid over the Dexels, to Scale", titlesize = 20)
    cam = camera(-52, 24)
    nx, ny = 6, 5
    blocks = dexel_blocks(nx, ny, p, s, 0.0, d)
    append!(blocks, filter(b -> !(b.lo[1] > 2.4p[1] && b.lo[2] < 2.4p[2]), grid_blocks(nx, ny, p, w, d, d + H)))
    straight = ((4.5p[1], 0.5p[2], d + H + 3.5), (4.5p[1], 0.5p[2], d))
    tilt = tand(20)
    stop_z = d + H - (0.5p[1] - w / 2) / tilt
    angled = ((3.0p[1] + 0.02, 1.5p[2], d + H + 2.0), (4.0p[1] - w / 2, 1.5p[2], stop_z))
    extras = [
        (1.0e3, a -> arrow!(a, project(cam, straight[1]), project(cam, straight[2]); color = MATERIALS.beam, linewidth = 3.5)),
        (1.0e3, a -> Mke.lines!(a, [project(cam, angled[1]), project(cam, angled[2])]; color = PALETTE.xray, linewidth = 3.5)),
        (1.0e3, a -> Mke.scatter!(a, [project(cam, angled[2])]; marker = :xcross, markersize = 20, color = PALETTE.xray)),
    ]
    draw_blocks!(ax, cam, blocks; extras)
    callout!(ax, project(cam, (4.5p[1], 0.5p[2], d + H + 2.5)), project(cam, (4.5p[1], 0.5p[2], d + H + 2.5)) + Mke.Point2f(3.0, 1.2), "From the Focal Spot:\nStraight Down a Channel"; color = Mke.RGBf(0.8, 0.5, 0.05), fontsize = 15)
    callout!(ax, project(cam, angled[2]), project(cam, angled[2]) + Mke.Point2f(3.6, -2.2), "Scattered in the Patient:\nStopped by a Plate"; color = PALETTE.xray, fontsize = 15)
    callout!(ax, project(cam, (nx * p[1], ny * p[2], d + H / 2)), project(cam, (nx * p[1], ny * p[2], d + H / 2)) + Mke.Point2f(1.4, 4.2), "Tungsten Plates\n$(round(Int, 1000w)) µm Thick,\n$(round(Int, H)) mm Tall"; fontsize = 15)
    Mke.xlims!(ax, nothing, project(cam, (nx * p[1], 0, 0))[1] + 9.5)

    # (b) side view, to scale, with the angle and the shadow a wall casts
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
    Mke.Label(fig[2, 2], "Photons Arriving θ = $(round(Int, θ))° Across the Fan:\nthe Fraction 1 − H tan θ / Opening Reaches the Ceramic.\nAlong Z There Are No Plates, So Nothing Is Stopped."; fontsize = 15, color = PALETTE.ink, tellwidth = false)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.32))
    locator!(fig[0, :], :grid; note = "Step 1 · The Grid\nIn Front of the Ceramic")
    fig
end

# ╔═╡ cf2841a8-b9aa-5fdb-8bc5-e10e24c3977d
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (ceramic and reflector), with and without the grid: the grid
should cost almost nothing, apart from its own tungsten x rays (59 and 67 keV) reaching the
ceramic above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ 69aee410-efa6-592d-8180-453b2de0fe49
grid_energies = 20.0:5.0:140.0

# ╔═╡ b1ec83d8-ac56-50c9-a3f6-571b398f8961
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

# ╔═╡ ff87450c-b7f3-527b-b80b-ed5836d3ebff
angles_deg = 0.0:1.0:20.0

# ╔═╡ f852ba38-b574-5557-9476-f736a5f5a0b8
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

# ╔═╡ ac409d87-d8df-5a56-b68b-00663f03200f
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
        title = "(b) 60 keV Photons Arriving at an Angle",
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

# ╔═╡ 2fff787e-5e88-5a8f-a0c4-2743126df964
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) % (the walls stand on the reflector; above 69.5 keV the walls' own x rays add a little).
        At an angle across the fan it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %.
        Along z it does nothing: a photon scattered toward the next row passes between the plates, which is why
        the Force's grid gained walls between rows.
        """
    )
end

# ╔═╡ 158d286e-48dd-5953-bb06-8dd6737b56d5
md"""
## 7. The Whole Detector

The whole detector, as BasisSimulator v0.15.1 has it: 736 columns × 64 rows on an arc of radius
1085.6 mm around the focal spot, every dexel a block of ceramic in reflector, under the plates
of the 1-D grid, all of it focused on the focal spot. An air scan fires
photons uniformly from the focal spot over the whole detector, and every dexel's signal is
summed. Two questions: is every dexel the same, and does the whole detector collect what the
single dexel of sections 4 and 6 predicts?
"""

# ╔═╡ 8262c7db-be74-5a8f-a9bd-323208a1072f
detector = MC.ArcDetector(;
    columns = flash.columns,
    rows = flash.rows,
    radius_mm = flash.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :across,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = ufc,
)

# ╔═╡ dae73e52-5156-586b-8a3d-39f034e4fd37
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ 3e41a1fe-825e-5e89-addd-5fd4a7422e3d
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

# ╔═╡ cb6fd61a-e2de-4359-8500-9a01a4ae4a6a
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 990))

    ax = canvas(fig[1, 1]; title = "The Whole Detector, from the Focal Spot (Depth Exaggerated)", titlesize = 20)
    cam = camera(-62, 26)
    R = flash.source_to_detector_mm
    half = deg2rad(half_fan_deg)
    half_z = flash.rows * flash.row_at_isocenter_mm * R / flash.source_to_isocenter_mm / 2
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
                iso = (0.0, -flash.source_to_isocenter_mm, 0.0)
                Mke.scatter!(a, [project(cam, iso)]; marker = :cross, markersize = 18, color = PALETTE.ink)
            end,
        )
    )
    sort!(items; by = first)
    for (_, f) in items
        f(ax)
    end
    iso = (0.0, -flash.source_to_isocenter_mm, 0.0)
    callout!(ax, project(cam, focal), project(cam, focal) + Mke.Point2f(150, 80), "Focal Spot"; fontsize = 16)
    callout!(ax, project(cam, iso), project(cam, iso) + Mke.Point2f(260, 60), "Isocentre, $(round(Int, flash.source_to_isocenter_mm)) mm Away"; fontsize = 16)
    edge = (R * sin(half), -R * cos(half), half_z)
    callout!(ax, project(cam, edge), project(cam, edge) + Mke.Point2f(120, 120), "$(flash.columns) Columns Across the Fan ($(round(2rad2deg(half); digits = 1))°)\n× $(flash.rows) Rows Along Z, at $(round(R; digits = 1)) mm"; fontsize = 16)
    low = (R * sin(-half / 2), -R * cos(-half / 2), -half_z)
    callout!(ax, project(cam, low), project(cam, low) + Mke.Point2f(-120, -160), "The Outer Rows See the Beam\nat the Cone Angle"; fontsize = 16, align = (:right, :center))
    xs = [project(cam, c)[1] for c in vcat(corners, [focal])]
    ys = [project(cam, c)[2] for c in vcat(corners, [focal])]
    Mke.limits!(ax, minimum(xs) - 520, maximum(xs) + 620, minimum(ys) - 240, maximum(ys) + 200)
    locator!(fig[0, :], :arc; note = "The Whole Arc:\n736 × 64 Dexels")
    fig
end

# ╔═╡ 7a8c5280-1636-52d9-9d00-753cb9d45901
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))

    # (b) The signal per photon arriving, across the fan, in groups of 40 columns
    axb = Mke.Axis(
        fig[1, 1];
        title = "(a) Energy Collected per Photon, Across the Fan",
        xlabel = "Column (Across the Fan)",
        ylabel = "Energy Collected per Photon\non the Cell, ÷ E",
        limits = ((1, flash.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:flash.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / flash.columns
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
        flash.columns / 2,
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
        limits = ((1, flash.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:flash.rows, max(1, flash.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / flash.rows
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

# ╔═╡ 2d128f1c-5a85-55e8-b79c-61441007204c
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
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / flash.source_to_detector_mm))
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

# ╔═╡ 98e10d09-f31e-53fc-a4bc-d485b5d074e5
md"""
## 8. The Flash's X-Ray Beam

The photons come from the installed BasisSimulator (these are not what v0.16.0 replaces): its IPEM Report 78 tungsten spectra, through the
scanner's flat filter and the protocol's filters, scaled to the detector at 1085.6 mm, exactly
as `simulate!` uses them for the Flash. BasisSimulator gives the Flash 8.4 mm of aluminium as its
flat filter (the Force has 3 mm), and its Flash protocols add 0.4 mm of tin at Sn140 and nothing
otherwise. BasisSimulator's spectra are tabulated at 80, 100, 120 and 140 kV.

The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ 051c955e-404e-5b0c-8a31-ea1f242941af
# BasisSimulator's Definition Flash, as on its scanner page
flash_scanner = BS.EICTScanner(
    source_to_isocenter = flash.source_to_isocenter_mm,
    source_to_detector = flash.source_to_detector_mm,
    detector_rows = flash.rows,
    detector_cols = flash.columns,
    detector_row_size = flash.row_at_isocenter_mm,
    detector_col_size = flash.column_at_isocenter_mm,
    detector_shape = :arc,
    focal_spot_width = 0.7,
    focal_spot_length = 0.7,
    target_angle = 7.0,
    gantry_rotation_time = 0.5,
    scan_diameter = 500.0,
    gantry_aperture = 780.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 8.4,
    bowtie_filter = :large_body,
    detector_material = :ufc_flash,
    detector_depth = basis_depth_mm,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 1500,
    detection_gain = 10.0,
)

# ╔═╡ 31e2e2b8-cfd3-565a-bb30-b483fd94a72e
"""
The photons per mAs per mm² reaching the detector, per bin, for a Flash protocol at `kVp` with
`tin_mm` of tin (0.4 for Sn140), through `flat_mm` of aluminium (BasisSimulator's Flash: 8.4).
"""
function flash_spectrum(kVp; tin_mm = 0.0, flat_mm = 8.4)
    filters = tin_mm > 0 ? [("Sn", tin_mm)] : Tuple{String, Float64}[]
    scanner = flat_mm == flash_scanner.flat_filter_thickness ? flash_scanner :
        BS.EICTScanner(
            source_to_isocenter = flash.source_to_isocenter_mm,
            source_to_detector = flash.source_to_detector_mm,
            detector_rows = flash.rows,
            detector_cols = flash.columns,
            detector_row_size = flash.row_at_isocenter_mm,
            detector_col_size = flash.column_at_isocenter_mm,
            detector_shape = :arc,
            target_angle = 7.0,
            flat_filter_material = :aluminum,
            flat_filter_thickness = flat_mm,
            bowtie_filter = :large_body,
            detector_material = :ufc_flash,
            detector_depth = basis_depth_mm,
        )
    protocol = BS.CTProtocol(;
        kVp,
        mA = 200.0,
        views = 1160,
        rotation_time = 0.5,
        collimation_mm = 4.8,
        anode_angle = 8,
        additional_filters = filters,
    )
    return BS.resolve_source_spectrum_without_bowtie(BS.SimOptions(), protocol; scanner)
end

# ╔═╡ dfb62e52-ecc8-533b-aae0-68c10ab542ba
@bind kvp PlutoUI.Slider([80, 100, 120, 140]; default = 120, show_value = true)

# ╔═╡ 515dd708-8847-539f-92d6-2a017a2e1fa9
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

# ╔═╡ caeb64bd-1f65-4592-b791-bb82e4aa8a4d
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))

    ax = canvas(fig[1, 1])
    amber = MATERIALS.beam
    # tube: envelope, cathode, anode at 8°
    Mke.poly!(ax, Mke.Rect2f(-0.6, -1.5, 2.4, 3.0); color = Mke.RGBf(0.93, 0.94, 0.96), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    rect!(ax, -0.45, -0.25, 0.3, 0.5, PALETTE.slab_edge)
    Mke.text!(ax, -0.3, -0.45; text = "Cathode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.poly!(ax, Mke.Point2f[(1.2, -0.9), (1.55, -0.9), (1.55, 0.9), (1.35, 0.9)]; color = Mke.RGBf(0.6, 0.45, 0.3))
    Mke.text!(ax, 1.42, -1.0; text = "Tungsten Anode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
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
        (3.0, 0.3, Mke.RGBf(0.75, 0.77, 0.8), "8.4 mm Al", "BasisSimulator's\nFlash, Always"),
        (4.6, 0.1, Mke.RGBf(0.55, 0.6, 0.5), "0.4 mm Sn", "Sn140 Only"),
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
    Mke.text!(ax, xi, 0.95; text = "Patient (Isocentre, 595 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    # detector with grid
    rect!(ax, x_end, -2.7, 0.15, 5.4, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
    for y in range(-2.6, 2.6; length = 14)
        rect!(ax, x_end - 0.35, y - 0.01, 0.35, 0.02, MATERIALS.tungsten)
    end
    Mke.text!(ax, x_end + 0.1, 2.85; text = "Grid + Detector\n(1085.6 mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -0.8, 14.6, -3.4, 3.9)
    locator!(fig[0, :], :tube; note = "The X-Ray Tube and\nWhat the Beam Passes")
    fig
end

# ╔═╡ a6557a74-3be6-558a-8f69-04983f9cafee
let
    Mke.set_theme!(THEME)
    plain = flash_spectrum(kvp)
    tin = flash_spectrum(kvp; tin_mm = 0.4)
    fig = Mke.Figure(size = (1400, 560))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) The Photons Reaching the Detector, $(kvp) kV",
        xlabel = "Photon Energy (keV)",
        ylabel = "Photons per mAs per mm²\nper keV",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    for (spectrum, color, label) in ((plain, PALETTE.photon, "8.4 mm Al"), (tin, PALETTE.xray, "… + 0.4 mm Tin"))
        s = signal_per_kev(spectrum, efficiency.flat_field)
        Mke.lines!(axa, s.energy_keV, s.photons; color, linewidth = 3, label)
    end
    Mke.axislegend(axa; position = :rt)

    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) The Signal They Make (8.4 mm Al)",
        xlabel = "Photon Energy (keV)",
        ylabel = "Signal per keV\n(Photons × E × η, Arbitrary Units)",
        xticks = 0:20:140,
        limits = ((10, 142), (0, nothing)),
    )
    ours = signal_per_kev(plain, efficiency.flat_field)
    theirs = signal_per_kev(plain, table_at.(collect(energies)))
    Mke.lines!(
        axb,
        theirs.energy_keV,
        theirs.signal;
        color = PALETTE.reference,
        linewidth = 3,
        label = "BasisSimulator v0.15.1's Table",
    )
    Mke.lines!(
        axb,
        ours.energy_keV,
        ours.signal;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, Flat Field, 1.4 mm",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ cc09b7bb-18b0-52e7-8a8d-c50fa47b6afc
md"""
**The beam's quality.** No published half-value layer for the Definition Flash was found, so the
beam cannot be checked the way the Alpha's and the Force's are. For the record, BasisSimulator's
Flash beam at 120 kV, at the isocentre (through the bowtie's thinnest part, 2.5 mm of aluminium),
beside the same beam with the Force's 3 mm flat filter, and Siemens' documented value for the
Force (a different scanner, shown only for scale):
"""

# ╔═╡ 0e3cf276-6097-5886-ad43-ffd792b2918b
hvl_check = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium = MC.DetectorMaterial(:Al)

    function hvl(flat_mm)
        e, w = flash_spectrum(120; flat_mm)
        μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
        return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
    end

    [
        (label = "BasisSimulator's Flash:\n8.4 mm Al", hvl = hvl(8.4)),
        (label = "With the Force's\n3 mm Al", hvl = hvl(3.0)),
    ]
end

# ╔═╡ c132195a-dc8e-49f3-9ecd-3bc8591563c6
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 380))
    for (k, (al, reading, title)) in enumerate(((0, "100 %", "No Aluminium in the Beam"), (hvl_check[1].hvl, "50 %", "$(round(hvl_check[1].hvl; digits = 1)) mm of Aluminium: Half Is Left")))
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

# ╔═╡ ff5e1b8b-42e7-598c-bd69-1de078ce21e5
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 400))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-Value Layer at the Isocentre, 120 kV",
        xlabel = "Half-Value Layer (mm of Aluminium)",
        yticks = (1:2, [c.label for c in hvl_check]),
        yreversed = true,
        limits = ((0, 14), (0.4, 3.2)),
    )
    Mke.vspan!(ax, 7.75, 7.85; color = (PALETTE.muted, 0.45))
    note!(
        ax,
        7.8,
        2.85,
        "The Force (Siemens' Manual): 7.8 mm";
        align = (:center, :center),
        color = PALETTE.muted,
        fontsize = 16,
    )
    Mke.barplot!(
        ax,
        1:2,
        [c.hvl for c in hvl_check];
        direction = :x,
        width = 0.6,
        color = [PALETTE.reference, PALETTE.photon],
    )
    for (k, c) in enumerate(hvl_check)
        note!(
            ax,
            c.hvl,
            k,
            "$(round(c.hvl; digits = 1)) mm";
            align = (:left, :center),
            offset = (8, 0),
            fontsize = 16,
            alpha = 0.0,
        )
    end
    fig
end

# ╔═╡ 649da741-6f87-557c-a158-98404b3b12a4
Markdown.parse(
    """
    The installed BasisSimulator's (v$(pkgversion(BS))) Flash beam has a half-value layer of **$(round(hvl_check[1].hvl; digits = 1)) mm** at 120 kV, much harder than the
    Force's documented 7.8 mm; with the Force's 3 mm flat filter it would be $(round(hvl_check[2].hvl; digits = 1)) mm. An 8.4 mm aluminium flat
    filter is unusual for a Siemens scanner. Without a published Flash value this stays an open question for
    BasisSimulator, not a confirmed discrepancy; the rest of this notebook keeps BasisSimulator's beam.
    """
)

# ╔═╡ 8ea16ba1-c4f8-51cd-a910-e43a4cca0a04
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. Every Flash tube voltage BasisSimulator can make, and its
tin-filtered Sn140:
"""

# ╔═╡ b9eab93c-d2d0-5ea9-9fa8-2ddfdc9c8b00
scan_protocols = vcat(
    [(label = "$(k) kV", spectrum = flash_spectrum(k)) for k in (80, 100, 120, 140)],
    [(label = "Sn140", spectrum = flash_spectrum(140; tin_mm = 0.4))],
)

# ╔═╡ 2282240e-cc35-5cb7-b730-ee5272eb5d39
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

# ╔═╡ cb8dc9ea-201c-4d51-be5b-1efb3a0db47a
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

# ╔═╡ ce9ffa92-e606-5229-9c4a-0b06ce03ae6a
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
        label = "BasisSimulator v0.15.1's Table",
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

# ╔═╡ f9d6d80b-d1c2-5741-baf9-1ad81966113d
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | v0.15.1 table | Geant4, 1.4 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's ceramic; BasisSimulator's fill factor (0.9 × 0.9) applies to both
        alike.
        """
    )
end

# ╔═╡ 3e4c7423-d5d0-45c6-95b7-3c3493f1cdd9
md"""
## 10. A Clinical Beam, for Real

Every efficiency so far came from photons of **one energy**, landing straight on **one dexel**.
A scan is neither: the Flash's tube sends a spectrum, hardened by the bowtie and by the patient,
from a focal spot over the whole arc, so the dexels at the edges of the fan and the outer rows
see it at an angle, through the grid. BasisSimulator never simulates that photon by photon; it
weights the single-energy table by the spectrum. This section checks that the shortcut is exact,
by doing it the long way:

- the Flash's **120 kV beam** (the installed BasisSimulator's spectrum, 8.4 mm Al, through the
  bowtie's centre), **bare and after 20 and 35 cm of water** (the patient's primary photons; the
  patient's scatter is notebook 10's);
- each one **drawn photon by photon from its spectrum**, onto one dexel's ceramic (1.6 million
  photons) and from the focal spot over **the whole arc** with its grid (3.2 million);
- against the table **weighted by the same spectrum**: `Σ φ E η / Σ φ E` for the efficiency, and
  `(Σ φ m₁)² / (Σ φ · Σ φ m₂)` for DQE(0), with `m₁`, `m₂` the mean and mean square of the energy
  a photon leaves (section 4).
"""

# ╔═╡ 263af7a6-ecbf-4699-bcce-d901f339f5da
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))


    # The path: tube, bowtie, water, the arc with its grid
    ax = canvas(fig[1, 1]; title = "The Beam, Photon by Photon", titlesize = 20)
    amber = MATERIALS.beam
    focal = Mke.Point2f(0, 0)
    Mke.scatter!(ax, [focal]; marker = :rect, markersize = 22, color = PALETTE.ink)
    Mke.text!(ax, focal + Mke.Point2f(0, 0.45); text = "120 kV Tube", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    φ = range(-deg2rad(half_fan_deg), deg2rad(half_fan_deg); length = 60)
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
    Mke.text!(ax, 0, -R - 0.7; text = "The Whole Arc and Its Grid (736 × 64 Dexels)", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
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
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron)
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

# ╔═╡ 0bdd2a90-d2e4-415d-bc0a-5422e01dd4bb
# The Flash's 120 kV beam on the central ray: BasisSimulator's spectrum through the bowtie's
# centre, then through 0, 20 and 35 cm of water (the primary photons), on its 0.5 keV grid
clinical_beams = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium, water = MC.DetectorMaterial(:Al), MC.DetectorMaterial(:Water)
    e, w = flash_spectrum(120)
    through_bowtie = w .* exp.(-[MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e] .* bowtie_al_mm)
    [
        (
                label = L == 0 ? "No Patient" : "$(round(Int, L / 10)) cm of Water",
                water_mm = L,
                spectrum = MC.Spectrum(
                    collect(Float64, e),
                    through_bowtie .* exp.(-[MC.linear_attenuation(water, max(E, 1.0)) for E in e] .* L),
                ),
            )
            for L in (0.0, 200.0, 350.0)
    ]
end

# ╔═╡ eeccaba7-d822-4809-958b-f3b8ad95f64c
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

# ╔═╡ 5a588a4d-0e14-427b-9be0-78939b813762
# The long way against the shortcut, for each beam: efficiency and DQE(0) on one dexel, and the
# energy the whole arc collects per photon; each direct value with its noise (the 16 chunks' spread)
clinical_check = let
    E = collect(energies)
    m1 = [d.crystal_keV / d.n for d in one_dexel]
    m2 = [d.crystal2_keV2 / d.n for d in one_dexel]
    cell = MC.energy_absorption_efficiency.(grid_runs.with)
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
                table = sum(w .* e .* [MC.interpolate_linear(collect(grid_energies), cell, clamp(x, 20.0, 140.0)) for x in e]) / sum(w .* e),
                direct = arc_of(sum(arcs)),
                noise = noise(arc_of.(arcs)),
            ),
        )
    end
end

# ╔═╡ 7bc14eac-f0e5-4a2b-a17a-a19829b1e90c
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

# ╔═╡ c9bc4aef-d984-45dd-a6a2-05a3e05636e0
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
        efficiency and 0.003 for DQE(0) (the table's noise and its 1-keV grid), and 0.01 for the whole arc, whose dexels
        see the beam at the fan and cone angles that one straight dexel does not (section 7). Where they agree, the
        single-energy table, weighted by any spectrum, is the clinical beam: what BasisSimulator does with it is exact.
        """
    )
end

# ╔═╡ 61d8130f-8798-5ba8-bb82-fdc6f8dc674d
md"""
## The Table to Use

The table to replace the v0.15.1 table with (BasisSimulator v0.16.0), and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **UFC at 7.31 g/cm³**, Siemens' measured density, not Geant4's stock 7.44 (section 1).
- **1.4 mm deep, not BasisSimulator v0.15.1's 1.0 mm.** No source gives the Flash's depth. 1.0 mm has no
  source at all; 1.4 mm is the only UFC thickness Siemens quotes for the ceramic it uses across its
  scanners, and the v0.15.1 table itself was made at 1.4 mm (section 5).
- **BasisSimulator's fill factor, 0.9 × 0.9, unchanged.** The assumed reflector reproduces it, and
  the grid's plates stand on the reflector and shade no ceramic (section 6).
- **The beam stays BasisSimulator's** until a measured half-value layer for the Flash turns up
  (section 8).

The efficiency for depths from 0.8 to 2.0 mm, 1.0 mm among them, is saved beside it.
"""

# ╔═╡ 8362641f-6ebc-5ec4-9db7-7f9ce102e3f0
md"""
## Saving the Table

One file in `lib/BasisMonteCarlo/tables/`, in the layout of BasisSimulator's
`UFC_FLASH_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of UFC at its
documented 7.31 g/cm³ and 1.4 mm, beside the one-dexel efficiency, the interaction probability,
the Swank factor and DQE(0). Because no source gives the Flash's depth,
the flat-field efficiency is also saved for depths from 0.8 to 2.0 mm: a
better-documented depth needs no new simulation. The reproduction of the v0.15.1 table is kept
with it, as the record of how that table was made.
"""

# ╔═╡ a3507583-779d-5995-9793-f5ee76cd2770
table_depths = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# ╔═╡ 591e6770-c8ed-5def-ac63-994d0cf5f6ed
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

# ╔═╡ 1394fdc2-d531-5449-9aa0-1264ec197dcc
# The machine path stays out of the rendered page
table_dir = joinpath(@__DIR__, "..", "..", "tables");

# ╔═╡ 096f1101-d468-50fd-b476-a105db66af80
saved = let
    path = joinpath(table_dir, "siemens_flash_ufc_efficiency_geant4.jls")
    contents = Dict{String, Any}(
        "energies" => collect(energies),
        "efficiency" => efficiency.flat_field,
        "efficiency_one_dexel" => efficiency.one_dexel,
        "interaction_probability" => efficiency.interaction,
        "swank_factor" => efficiency.swank,
        "dqe0" => efficiency.dqe,
        "depths_mm" => table_depths,
        "efficiency_by_depth" => MC.energy_absorption_efficiency.(by_depth),
        "reproduced_table" => reproduced,
        "reproduced_depth_mm" => fit.depth_mm,
        "reproduced_scale" => fit.scale,
        "compared_with" => "BasisSimulator v0.15.1, UFC_FLASH_MC_EFFICIENCY_LUT (BasisMonteCarlo.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash)",
        "decision" => "use `efficiency`: flat field, UFC at 7.31 g/cm3, 1.4 mm deep (no source gives the Flash's depth; 1.4 mm is the only UFC thickness Siemens quotes, and the current table was made at it); BasisSimulator's fill factor 0.9 x 0.9 unchanged",
        "definition" => "mean energy collected per photon landing on a dexel's ceramic with every dexel illuminated (flat field), divided by the photon energy; BasisSimulator's fill factor applies separately",
        "detector" => Dict(
            "scanner" => "Siemens SOMATOM Definition Flash (Stellar)",
            "material" => "UFC, Gd2O2S, 7.31 g/cm3",
            "depth_mm" => depth_mm,
            "pitch_mm" => pitch_mm,
            "reflector" => "0.1 mm TiO2-loaded epoxy (assumed)",
            "grid" => "1-D tungsten plates, 0.08 mm, 10 mm tall (assumed); no effect on the flat field",
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 05_siemens_flash.jl",
    )
    Serialization.serialize(path, contents)
    path
end;

# ╔═╡ ccd89431-c3da-5885-bd96-2e1c6ddcbbff
# Read it back in BasisSimulator's layout
let
    t = Serialization.deserialize(saved)
    lut = (energies = t["energies"], efficiency = t["efficiency"])
    (
        file = basename(saved),
        energies = extrema(lut.energies),
        at_60keV = lut.efficiency[60],
        at_120keV = lut.efficiency[120],
    )
end

# ╔═╡ 51418200-0d83-5422-88a3-407a22fbf450
md"""
## Where This Leaves the Flash
"""

# ╔═╡ 701f42dd-3aee-56f7-afa1-79db4ce240e3
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    Markdown.parse(
        """
        - **The material is right, the density 2 % high.** UFC is Gd₂O₂S at 7.31 g/cm³ (Siemens' data sheet);
          BasisSimulator v0.15.1 uses Geant4's stock 7.44. Geant4 lands on Siemens' own 1.4 mm absorption curve up to 90 keV.
        - **The depth: 1.4 mm.** No source gives the Flash's crystal depth; BasisSimulator v0.15.1 says 1.0 mm with no source,
          Siemens quotes only 1.4 mm for UFC, and the v0.15.1 table is reproduced, to about 0.01, by a pencil
          beam into one dexel of Gd₂O₂S **$(fit.depth_mm) mm deep**, tallied in that dexel.
        - **What a scan sees** is the flat-field efficiency: each dexel gets back from its neighbours the Gd x rays it
          loses to them. At 1.4 mm it is $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, against the table's
          $(round(m120.table; digits = 3)) ($(round(100(m120.geant4 / m120.table - 1); digits = 1)) %), with DQE(0) $(round(m120.dqe; digits = 3)).
        - **The grid** (1-D, tungsten) costs the photons from the focal spot almost nothing and stops scatter across the
          fan, not along z; **the whole detector** collects what one dexel's cell does.
        - **The beam:** the installed BasisSimulator's Flash has an 8.4 mm aluminium flat filter and a 120 kV half-value layer of
          $(round(hvl_check[1].hvl; digits = 1)) mm; no published value exists to check it against.
        - **Saved:** `$(basename(saved))`, with the efficiency for depths 0.8–2.0 mm: the table for BasisSimulator v0.16.0.
        """
    )
end

# ╔═╡ Cell order:
# ╟─43087614-5ea9-557c-946e-81714d62f3b3
# ╟─522577f8-9416-4099-ac48-93e0abeae0cc
# ╟─4c99acaf-d4fa-59d5-819d-dc4b474284b5
# ╟─060ceab7-f004-4019-a677-fbe33b15df82
# ╟─17331933-7602-4de6-bd2a-ed721d18cd59
# ╟─157a12e0-b5b4-4548-bf56-1429d6a241d1
# ╟─321e4165-9171-4b97-86ff-19171fe3eb71
# ╟─3ae5f27e-e5cb-5c6b-bf30-814e26acb31e
# ╟─dc8fc5c4-937e-55d4-98bd-2f48387c7da0
# ╠═aae12472-9dce-5e25-ad8a-8e4e7c67f16d
# ╠═5fe29898-f393-5a55-9bb6-c6c073ee7066
# ╠═ee624c58-008c-5e31-95d8-872303dc3a3a
# ╠═1d95c3a9-6ecd-5c3c-94a0-da9bdc1b345e
# ╠═4eff9487-c35d-5f43-a940-3732dd6c7333
# ╠═2cd1697f-e9e4-5073-bc81-89a8d6fe1d59
# ╠═32170bf5-5efc-55fe-a925-b5a05d93fcdc
# ╠═c13b2b79-71d3-5d5a-b407-8e7cdb518d5b
# ╠═be0ade99-9563-5db7-bf09-e72d3fa36cf2
# ╠═9e62282b-bb3d-58e1-b2cd-1f485feee4df
# ╠═5ff14064-4c4f-5400-929d-91426e2ee31f
# ╠═fa94fde5-ae10-5f3a-ae6a-b470df2fa380
# ╠═6604eb4b-4571-5cf6-9522-7749a6b5018a
# ╠═da837658-54a1-5f7f-b36b-df754e2c5081
# ╠═c2f6f29a-dc0e-5a6a-a27e-be4d8a15a61e
# ╟─9df857f1-00f6-4ea8-b572-ef8bb1d72f62
# ╠═70e9e722-217b-4aa8-a54c-0ee8f2e37c3f
# ╠═8aced4bd-0255-4e09-bba0-b6a684c2cf7b
# ╠═fb112cc1-82b1-4c65-9613-469c2a06462c
# ╠═2af0f891-185b-4537-9480-bf9feea3443e
# ╠═6b97c87e-9c2d-4c05-b335-e8444197f473
# ╠═6e25f6ed-1671-4479-85b2-150dce07afaf
# ╠═9bdfaf44-367b-419f-bc91-11e62627b792
# ╠═2420f6e7-9bdb-4f66-85a8-58dac88967c2
# ╠═e60344bc-3308-4b38-83c8-ad7899fcd81b
# ╠═2efaaa99-1dc5-4b41-8a19-a58a814936b5
# ╠═7d0641a4-9883-4e69-bc28-3838dde0a7fb
# ╠═317d1245-687c-439d-91e7-547fcc844313
# ╠═4f1eecf3-543a-43dd-a6b4-5c843b72e155
# ╠═3fb6787c-7210-4df8-843a-4edb2a74e7fe
# ╠═f436da99-757f-4e92-8cb0-3a590224ee61
# ╠═aab4d99a-83fa-456a-b52b-35f6f8e47d58
# ╠═a01538e6-4fe6-432a-bc20-c8f9b368f065
# ╠═8ce5f74e-69d5-467e-939e-267a1ffcc1a0
# ╠═42387328-1bee-47fa-bcf8-8d28b1f310c8
# ╠═4630e698-1d3a-4936-bc81-c9fecdff60d4
# ╠═f906a50e-31b6-4808-8f7f-3e1e6d845ce6
# ╠═136d5223-aefc-4bd6-8d9f-f7da9a90533f
# ╠═293b6771-66a9-4f4b-956a-2a48f426d21f
# ╠═6eddd832-8dab-44d7-ad21-e574b0ad732f
# ╠═04985f30-4003-4e13-953d-cf478f93caf3
# ╠═695af4b7-fd89-45a9-a35a-b01a80b5ed46
# ╠═5ada2e54-299f-4da9-bcbb-1a65c5f6a2a0
# ╠═cc96f529-3e5f-4a88-a35a-757d5adf5b53
# ╟─9331e7c6-ad6c-5ca1-8a96-2d239fcd68f7
# ╠═108ca11c-e55d-5f47-9770-60f3e5b802cf
# ╠═8718ba53-cf7b-5a52-ad08-bce13c35b85d
# ╠═0c12fe24-c4b0-5bdd-b1da-ca6cda0649fd
# ╟─2e6bd82e-2963-56c0-9d24-59952bf496b9
# ╠═002a21be-8ec7-596b-94f7-d3205e56673d
# ╠═d5333d07-5574-5dba-834c-587102c2762a
# ╠═5844ef71-4f68-5515-9b80-9d22e3360553
# ╠═b3faa435-5787-52c9-941b-6fcf7f4dad4f
# ╟─a8708a37-3d1c-43b8-b1d1-e18dcbe9c409
# ╟─9ef53632-c492-54e8-bbe0-c1ffcff7312d
# ╟─db66a77b-19df-5600-bd03-af9e287bfc9a
# ╟─61af1c5e-8856-567a-90ea-f767bf5fc11e
# ╠═3c0cabe0-2286-5d68-8d18-f9538ed090d1
# ╠═82e5e0b6-7679-55b0-b4b6-96db5d8ea994
# ╠═2b0afce5-3ccd-528c-b7ae-a8044d2d5683
# ╟─693ba223-ad33-44f9-8ca2-ae9f51b12821
# ╟─5781d012-225a-5557-afe3-620a360dd3ba
# ╟─73b50137-be3c-5db5-ab49-da03668bb584
# ╟─aa61390f-493b-564b-a4b5-b93a87b2737f
# ╠═05d18f32-ecb0-5777-940f-83aea74c9585
# ╠═5cbc9df5-f1aa-43ed-87e0-1d08fa008dee
# ╠═818d4df5-9882-5fe8-8a97-22162cc76fde
# ╠═e0743f95-30f6-504b-9ee3-fa8bc89b64f3
# ╠═0636b82f-c4db-5d58-94aa-81afc15b2bc5
# ╠═10b7f044-9ef7-5c94-b774-fed3968906f0
# ╠═ab9d1d88-c53b-5c91-8027-2cada0a314c3
# ╠═a0d86cb6-f528-5523-8afb-9530c768d695
# ╟─3fd947f2-6838-53ea-a46a-78685348e52b
# ╠═daa5074e-b919-5cb6-b45e-b1cea0c1258b
# ╠═d5234118-fc2c-585a-a56f-567d72908a9c
# ╠═a0a407f6-6b5c-5b35-9e1a-362d05bb0ca2
# ╠═aaf055ae-9730-591e-b076-2b1a87fdda17
# ╟─6655624b-23e3-5403-8220-dbd09038bfbd
# ╟─a5e2bfee-aac8-57ee-b273-54fb5ecc15f2
# ╠═c1346985-186d-58ed-8d05-8a664a33ef4f
# ╠═f8589d48-c9c7-571d-8b8d-9c3ea330604f
# ╟─22714c6e-01d7-5eb4-be91-8b8e68d7f036
# ╟─be0d59f2-e19b-44b0-af6d-bdb7dd2e355e
# ╟─c5490cd0-fe64-44ee-8306-93f33b5b4131
# ╟─f4f8ffd2-4418-5389-9c2f-098bfcc8d0b5
# ╠═a1c71e56-24db-5d77-913f-6326f04cace7
# ╠═d31b0627-dba4-5469-b818-f3b3566c9c4d
# ╟─419873a9-c9de-4f49-9671-9c28b39f9f28
# ╟─cd446a63-4adb-5f61-8de9-5d0e0adcf000
# ╟─02c6d1e4-6da3-591a-be8f-57f184c64cf2
# ╠═99c71afa-fb06-5c6f-ba03-32ed332b2712
# ╠═eddc2f24-22fb-5b53-9eb4-55d6099d4ea2
# ╟─3fe0e4c3-34d5-445d-96a8-a1df49943e96
# ╟─19bfc4a0-f377-58c9-b348-c107b5956377
# ╟─c5c4e668-d102-53d0-8083-3fea5ac2a72e
# ╟─0439842c-d207-5998-992b-a805548de5e4
# ╠═d5e73ebf-d120-5bbb-81a5-bbace6b0672e
# ╠═9289c700-43d7-5aef-b9d2-f725563323e8
# ╠═e3d686de-c192-5a51-9ba3-fc75ca707bf9
# ╟─30053ee8-f64f-43aa-ac23-17f29150dab5
# ╟─7327c893-ff25-50f0-912f-022e5486075d
# ╟─081b6107-7fab-5d8f-ab08-2b313210305a
# ╟─7833105e-c989-5a22-a1be-f90a30e20236
# ╠═32c72144-66d9-5162-a408-c872b6e67442
# ╟─136d7bd4-9e91-4313-83b5-143c9554bcaa
# ╟─477b6af3-e9b6-50c2-ba44-771359ec9420
# ╟─350b821b-48cd-57be-bb2d-252054214409
# ╠═8a971118-c3fc-5397-bc27-7f56b074b34e
# ╟─a4b69209-db1a-47bc-8435-58654e47a56d
# ╠═dd7879b9-406a-5760-84b8-73eb1b6a367e
# ╟─46d5bbd9-c8de-4ebe-8d36-77792b7308e4
# ╟─4b513a3a-e153-5474-b80f-ea0a83aaadd6
# ╟─0a61ea56-fdd2-57d4-8bde-f2490f63401b
# ╟─fca7e15c-bf53-5b4e-b12b-76610eeffcc8
# ╠═5383f381-53af-59aa-9564-bc9cc599215d
# ╠═e5093aeb-4ea9-5061-8652-853408cd9871
# ╠═03dbc810-873f-5ccd-880c-0457715d5780
# ╠═96c4845c-15f3-54a4-ae96-654e74c5b90f
# ╠═69df791a-942d-56f6-b730-3cdddbfb7477
# ╟─dbda913a-b1dd-5500-8568-98e38bcca95c
# ╠═b2bd0487-6f90-512e-807a-030292a2b6a4
# ╠═8f1937c0-4cc6-5aec-8d82-b941b761f267
# ╟─14c6dbf6-b4c3-5ddf-8ce8-95f748137025
# ╟─c1e171af-68b9-58ca-919f-af9a2efe31ba
# ╟─980f18e5-1050-580f-9439-39127dd9654b
# ╠═722c2410-b3a2-5325-a3bd-83a737169149
# ╟─a5980443-b05b-5f82-ae79-3c535ee866b2
# ╟─cf2841a8-b9aa-5fdb-8bc5-e10e24c3977d
# ╠═69aee410-efa6-592d-8180-453b2de0fe49
# ╠═b1ec83d8-ac56-50c9-a3f6-571b398f8961
# ╠═ff87450c-b7f3-527b-b80b-ed5836d3ebff
# ╠═f852ba38-b574-5557-9476-f736a5f5a0b8
# ╟─ac409d87-d8df-5a56-b68b-00663f03200f
# ╟─2fff787e-5e88-5a8f-a0c4-2743126df964
# ╟─158d286e-48dd-5953-bb06-8dd6737b56d5
# ╠═8262c7db-be74-5a8f-a9bd-323208a1072f
# ╠═dae73e52-5156-586b-8a3d-39f034e4fd37
# ╠═3e41a1fe-825e-5e89-addd-5fd4a7422e3d
# ╟─cb6fd61a-e2de-4359-8500-9a01a4ae4a6a
# ╟─7a8c5280-1636-52d9-9d00-753cb9d45901
# ╟─2d128f1c-5a85-55e8-b79c-61441007204c
# ╟─98e10d09-f31e-53fc-a4bc-d485b5d074e5
# ╠═051c955e-404e-5b0c-8a31-ea1f242941af
# ╠═31e2e2b8-cfd3-565a-bb30-b483fd94a72e
# ╠═dfb62e52-ecc8-533b-aae0-68c10ab542ba
# ╠═515dd708-8847-539f-92d6-2a017a2e1fa9
# ╟─caeb64bd-1f65-4592-b791-bb82e4aa8a4d
# ╟─a6557a74-3be6-558a-8f69-04983f9cafee
# ╟─cc09b7bb-18b0-52e7-8a8d-c50fa47b6afc
# ╠═0e3cf276-6097-5886-ad43-ffd792b2918b
# ╟─c132195a-dc8e-49f3-9ecd-3bc8591563c6
# ╟─ff5e1b8b-42e7-598c-bd69-1de078ce21e5
# ╟─649da741-6f87-557c-a158-98404b3b12a4
# ╟─8ea16ba1-c4f8-51cd-a910-e43a4cca0a04
# ╠═b9eab93c-d2d0-5ea9-9fa8-2ddfdc9c8b00
# ╠═2282240e-cc35-5cb7-b730-ee5272eb5d39
# ╟─cb8dc9ea-201c-4d51-be5b-1efb3a0db47a
# ╟─ce9ffa92-e606-5229-9c4a-0b06ce03ae6a
# ╟─f9d6d80b-d1c2-5741-baf9-1ad81966113d
# ╟─3e4c7423-d5d0-45c6-95b7-3c3493f1cdd9
# ╟─263af7a6-ecbf-4699-bcce-d901f339f5da
# ╠═0bdd2a90-d2e4-415d-bc0a-5422e01dd4bb
# ╠═eeccaba7-d822-4809-958b-f3b8ad95f64c
# ╠═5a588a4d-0e14-427b-9be0-78939b813762
# ╟─7bc14eac-f0e5-4a2b-a17a-a19829b1e90c
# ╟─c9bc4aef-d984-45dd-a6a2-05a3e05636e0
# ╟─61d8130f-8798-5ba8-bb82-fdc6f8dc674d
# ╟─8362641f-6ebc-5ec4-9db7-7f9ce102e3f0
# ╠═a3507583-779d-5995-9793-f5ee76cd2770
# ╠═591e6770-c8ed-5def-ac63-994d0cf5f6ed
# ╠═1394fdc2-d531-5449-9aa0-1264ec197dcc
# ╠═096f1101-d468-50fd-b476-a105db66af80
# ╠═ccd89431-c3da-5885-bd96-2e1c6ddcbbff
# ╟─51418200-0d83-5422-88a3-407a22fbf450
# ╟─701f42dd-3aee-56f7-afa1-79db4ce240e3
