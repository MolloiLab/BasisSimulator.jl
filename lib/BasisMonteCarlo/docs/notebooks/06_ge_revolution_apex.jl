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


# ╔═╡ 982eeab8-0545-51f1-b14c-f4984bc5ce21
md"""
# The GE Gemstone Detector in Geant4: Revolution Apex

The GE Revolution Apex is an energy-integrating CT scanner with a 256-row, 160 mm wide detector
(GE's "Gemstone Clarity"). Its scintillator is **Gemstone**, a cerium-activated garnet of
terbium, lutetium and aluminium, (Tb,Lu)₃Al₅O₁₂:Ce, made as a transparent polycrystalline
ceramic and cut into **dexels** (detector elements), with a reflector between them and a
backlit photodiode under every dexel. In front of it stands GE's "3D collimator", a
two-dimensional anti-scatter grid.

Up to and including **v0.15.1**, BasisSimulator describes how efficiently this detector turns
x rays into signal with one table, η(E) from 1 to 140 keV (`GEMSTONE_MC_EFFICIENCY_LUT`),
computed with MCNP; the inputs to that simulation were never committed. This notebook builds the
detector in Geant4, one piece at a time, and

1. **checks every piece against the documentation**: the FDA clearances, GE's patents and
   brochures, and the literature;
2. **reproduces the v0.15.1 table**, which shows how it was made;
3. **builds a better table** for a garnet that can exist (BasisSimulator v0.15.1's density
   cannot), with its noise (the Swank factor), and saves it: the table BasisSimulator takes over
   from v0.16.0.

!!! note "Which BasisSimulator this notebook compares against"
    Every comparison with "the v0.15.1 table", and every "BasisSimulator v0.15.1" value (the
    garnet's composition and 7.0 g/cm³, the 3.0 mm depth, the fill factor, the geometry), is with
    **BasisSimulator v0.15.1**, the last release before this notebook's table. BasisMonteCarlo
    keeps that table (`BASISSIMULATOR_V0_15_1_EFFICIENCY`, copied verbatim), so the comparisons
    stay the same after BasisSimulator moves to the new one. Only the x-ray spectra and the bowtie
    (sections 8–10) come from the **installed** BasisSimulator, whatever its version; nothing here
    pins BasisSimulator, or BasisMonteCarlo, to a version.
"""

# ╔═╡ f47a2a75-13c1-4576-b10c-da26be745122
let
    Mke.set_theme!(THEME)
    night, ink, sub = MATERIALS.night, Mke.RGBf(0.93, 0.94, 0.96), Mke.RGBf(0.6, 0.63, 0.68)
    fig = Mke.Figure(size = (1500, 600), backgroundcolor = night, figure_padding = (10, 10, 10, 10))

    # (1) The gantry, to scale: one tube and its wide detector arc, the 80 cm bore
    ax1 = canvas(fig[1, 1]; background = night)
    R_iso, R_det = apex.source_to_isocenter_mm, apex.source_to_detector_mm
    half_fan_deg = rad2deg(apex.columns * apex.column_at_isocenter_mm / apex.source_to_isocenter_mm / 2)
    θ = range(0, 2π; length = 200)
    Mke.poly!(ax1, Mke.Point2f.(820 .* cos.(θ), 820 .* sin.(θ)); color = MATERIALS.night_ring)
    Mke.poly!(ax1, Mke.Point2f.(400 .* cos.(θ), 400 .* sin.(θ)); color = night)
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
    tube!(90, half_fan_deg, 0.22, "X-Ray Tube")
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
        a, b = project(cam, (c..., d + grid.height_mm + 2.5)), project(cam, (c..., d + 0.15))
        (dot3((c..., d), cam.toward) + 1, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 12))
    end
    draw_blocks!(ax2, cam, blocks; extras = rays, edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    # (3) One dexel's stack, taken apart
    ax3 = canvas(fig[1, 3]; background = night)
    draw_blocks!(ax3, camera(-56, 30), exploded_detector(1, 1; gap = 0.75).blocks; edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    for (k, (title, subtitle)) in enumerate(
            (
                ("One Tube, a 160 mm Wide Detector", "The Revolution Apex, to Scale: 80 cm Bore"),
                ("Dexels Under GE's 3D Collimator", "Gemstone Garnet, White Reflector, Tungsten Walls"),
                ("One Dexel, Taken Apart", "Grid, Garnet, Backlit Photodiode, Electronics"),
            )
        )
        Mke.Label(fig[2, k], title; color = ink, fontsize = 26, font = :bold, tellwidth = false)
        Mke.Label(fig[3, k], subtitle; color = sub, fontsize = 18, tellwidth = false)
    end
    Mke.rowgap!(fig.layout, 1, 8)
    Mke.rowgap!(fig.layout, 2, 4)
    fig
end

# ╔═╡ 7d76168f-6aaf-5cac-ad93-06b3d605dce4
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a Gemstone dexel does not count at all.
Every x ray it absorbs leaves energy in the garnet, which glows; the reflector keeps the light
inside the dexel, and the photodiode under it adds up the light of every photon over the reading.
The signal is therefore proportional to the **energy** left in the dexel, and the efficiency
that matters is the energy-absorption efficiency: the mean energy a dexel collects per photon,
divided by the photon's energy.
"""

# ╔═╡ 5b30322f-8454-4b47-8acf-88c1e1bc53a5
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
        key === :board && continue
        outline!(ax, cam, (i * p[1], j * p[2], z0), ((i + 1) * p[1], (j + 1) * p[2], z1); linewidth = 3.5)
    end
    Mke.Label(
        fig[2, 1],
        "One Dexel, Outlined in Amber: One Block of Garnet, Its Photodiode and Its Readout Channel, Under One Grid Cell";
        font = :bold,
        fontsize = 17,
        color = Mke.RGBf(0.8, 0.5, 0.05),
        halign = :left,
        tellwidth = false,
    )

    rows = (
        (:grid, "GE's 3D Collimator", "2-D Tungsten Walls, One Cell per Dexel\n(8 mm Tall in GE's Patent; Drawn Short Here)", true),
        (:ceramic, "Gemstone Garnet, Cut into Dexels", "One Block of (Tb,Lu)₃Al₅O₁₂:Ce per Dexel, 3.0 mm Deep;\nWhite Reflector Between (TiO₂ Assumed)", true),
        (:photodiode, "Backlit Photodiode, One per Dexel", "\"Low Capacitance Backlit Photodiode\" (K213715):\nLight → Charge", false),
        (:cmos, "Readout Electronics", "Digitizes Each Dexel; GE Does Not Publish\nHow or Where (Drawn Generic)", false),
        (:board, "Detector Module", "Focally Aligned Sub-Modules (K133705),\nCooled by the Detector Thermal System", false),
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

# ╔═╡ 1637020f-80bb-43db-9357-909efdb66156
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1500, 860))
    ax = canvas(fig[1, 1])

    # Layers, side view (x across, y up); heights in mm, the grid cut short
    y_board, y_cmos, y_pd = -1.35, -0.8, -0.16
    rect!(ax, -1.5p, y_board, 3p, 0.3, MATERIALS.board)
    rect!(ax, -1.5p, y_cmos, 3p, 0.42, MATERIALS.cmos)
    Mke.text!(ax, 0, y_cmos + 0.21; text = "Readout Electronics (Not Published)", color = :white, fontsize = 13, align = (:center, :center), font = :bold)
    for i in -1:1
        rect!(ax, i * p - (p - s) / 2, y_pd, p - s, 0.14, MATERIALS.photodiode)
    end
    # garnet and reflector
    rect!(ax, -1.5p, 0, 3p, d, MATERIALS.reflector; strokecolor = (:black, 0.25), strokewidth = 1)
    for i in -1:1
        rect!(ax, i * p - (p - s) / 2, 0, p - s, d, i == 0 ? Mke.RGBf(0.8, 0.95, 0.82) : MATERIALS.scintillator)
    end
    # grid walls, cut short (wavy break)
    for i in -1:2
        x = (i - 0.5) * p
        rect!(ax, x - grid.wall_mm / 2, d, grid.wall_mm, 1.2, MATERIALS.tungsten)
    end

    # (1) the x ray, (2) stops in the middle dexel, (3) light, (4) photodiode, (5) the electronics
    hit = Mke.Point2f(0.12, d - 1.1)
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
        (anchor = (0.12, d + 1.9), title = "1  An X Ray Arrives", detail = "Down Its Channel in the Grid", color = Mke.RGBf(0.85, 0.5, 0.05), tag = nothing),
        (anchor = hit, title = "2  The Garnet Stops It", detail = "Its Energy Is Left in the Dexel", color = Mke.RGBf(0.7, 0.55, 0.05), tag = nothing),
        (anchor = (half - 0.02, hit[2] + 0.28), title = "3  The Garnet Glows", detail = "Light ∝ Energy Left; the White Reflector Keeps It In", color = Mke.RGBf(0.2, 0.6, 0.25), tag = nothing),
        (anchor = (0.4, y_pd + 0.07), title = "4  The Backlit Photodiode Collects the Light", detail = "Light → Charge, Summed over the Reading", color = MATERIALS.photodiode, tag = nothing),
        (anchor = (0.5, y_cmos + 0.21), title = "5  The Electronics Digitize the Charge", detail = "Where and How: Not Published by GE", color = PALETTE.ink, tag = nothing),
    ]
    xr = 1.5p + 0.75
    labels_right!(ax, entries, xr; spacing = 0.62)
    Mke.text!(ax, 1.5p + 0.15, d + 1.2; text = "Grid Walls\n(Cut Short)", fontsize = 14, color = PALETTE.muted, align = (:left, :top))

    # what Geant4 simulates, and what is assumed
    xl = -1.5p - 0.15
    side_bracket!(ax, xl, 0.0, d + 1.2, "Geant4\nSimulates"; color = TAGS.geant4)
    side_bracket!(ax, xl, y_board, -0.04, "Not Simulated:\nSignal ∝ Energy\nLeft in the Garnet"; color = TAGS.not_simulated)
    Mke.limits!(ax, -1.5p - 1.75, xr + 5.2, y_board - 0.75, d + 2.75)
    fig
end

# ╔═╡ 2e433940-1ca3-4ed7-a00d-2410fb565f03
md"""
### Which Beam, and Why

How the photons arrive decides what a simulation measures, so each section says which beam it
uses. Five are used here, from the simplest to the most realistic:

| beam | what it measures | where |
|---|---|---|
| **pencil beam** into the centre of one dexel | how the v0.15.1 table was made (it is reproduced exactly) | section 5 |
| **uniform on one dexel's garnet** | the efficiency per photon landing on the garnet; reading every dexel around it gives the flat field (below) | sections 3–5 |
| **uniform on one cell** (garnet and its reflector), under the grid | what the grid costs, straight and at an angle | section 6 |
| **flat field**: every dexel lit | what each dexel collects in a scan: what it loses to its neighbours it gets back from them. By symmetry this equals all the energy a uniform beam on one dexel leaves anywhere, which is how it is simulated | sections 4, 5 |
| **from the focal spot**, over the whole arc, with **clinical spectra** | everything at once: real angles, every dexel, the grid, the Apex's own 120 kV beam (also after 20 and 35 cm of water) | sections 7, 10 |

The efficiency table is built from single energies, because BasisSimulator weights it by whatever
spectrum a scan uses. Section 10 checks that this shortcut is exact: the clinical beams, simulated
photon by photon, give the same signal and noise as the table weighted by their spectra.
"""

# ╔═╡ 81e47492-f46a-4e97-a3a2-d04bd5d8e70d
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1700, 560))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        ("Pencil Beam", "Into the Centre of One Dexel:\nHow the v0.15.1 Table Was Made", :pencil),
        ("Uniform on One Dexel", "Over Its Garnet; Every Dexel\nRead Out (Sections 3–5)", :ceramic),
        ("Uniform on One Cell", "Garnet and Its Reflector,\nUnder the Grid (Section 6)", :cell),
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

# ╔═╡ 0d89ac30-a1cd-555c-a60a-934112484f93
md"""
## The Revolution Apex, as Documented

Every hardware number this notebook uses, checked against the FDA 510(k) summaries, GE's patents
and brochures, and the literature. Where BasisSimulator v0.15.1 and a source disagree the row is marked
**⚠ discrepancy**; where no source gives a number the row says so, and the notebook states its
assumption. GE publishes very little about the detector itself.

| quantity | BasisSimulator v0.15.1 | documented | status | source |
|---|---|---|---|---|
| scintillator | Gemstone, (Tb,Lu)₃Al₅O₁₂:Ce | a Ce-activated rare-earth garnet; "(Tb,Lu)" named in the literature; GE's patents claim (Tb₁₋ₓLuₓ)₃Al₅O₁₂:Ce with x from 0 to about 0.3 | ✓ (family) | LLNL-TR-587512 p. 7; US 6,793,848; You et al. 2023 |
| composition | Tb 0.287, Lu 0.316, Al 0.163, O 0.231, Ce 0.002 by mass (an even Tb/Lu split) | not published; GE's patent example Lu₀.₈Tb₂.₂Al₅O₁₂ is Tb-rich (Tb 0.428, Lu 0.171) | not confirmed; bracketed in section 4 | US 6,793,848 |
| citation for the composition | You et al. 2023 (PMC10179960) | that paper names Gemstone only, with no composition | **⚠ discrepancy** (miscitation) | You et al., Materials 2023 |
| density | 7.0 g/cm³ | any (Tb,Lu)₃Al₅O₁₂ lies between 6.07 (Tb₃Al₅O₁₂) and 6.72 (Lu₃Al₅O₁₂) | **⚠ discrepancy** (impossible); 6.40 used here | Konoshima; Zhu et al. 2022 |
| crystal depth | 3.0 mm | not published; 3.0 mm is XCIST's default detector depth (for its "Lumex", a different material); GE's reflector patent gives cells about 1.5–3 mm tall | not confirmed | XCIST `Scanner_Default.cfg`; US 8,385,499 |
| the `:lumex` name | `:lumex` = Gemstone | XCIST's "Lumex" is (Y,Gd)₂O₃:Eu, GE's older HiLight ceramic | **⚠ discrepancy** (naming) | XCIST `material/Lumex` |
| rows × row width | 256 × 0.625 mm | 256 × 0.625 mm, 160 mm | ✓ | K133705; K213715 |
| columns, column width, source–isocentre, source–detector | 834; 0.6 mm; 625.6, 1100 mm | not published | not confirmed | – |
| reflector between dexels | not modelled (fill factor 0.9 × 0.9) | not published for the product; GE's patent lists TiO₂ among reflector materials | assumed: 0.1 mm of TiO₂-loaded epoxy, which gives the same fill factor | US 8,385,499 |
| anti-scatter grid | not modelled | "3D collimator (post patient)", two-dimensional; product dimensions not published; GE's patent: walls 0.10–0.20 mm, about 8 mm tall, one channel per dexel | built here: tungsten, 0.15 mm walls, 8 mm tall | K133705; US 8,385,499 |
| photodiode | – | "low capacitance backlit photodiode" | – | K213715 |
| beam quality (Revolution CT, large bowtie) | – | half-value layer 7.08, 9.20, 10.0 mm Al at 80, 120, 140 kV, isocentre | a check in section 8 | Kawahara et al. 2023 |

Sources: FDA [K133705](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K133705.pdf),
[K213715](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf);
GE patents [US 6,793,848](https://patents.google.com/patent/US6793848B2/en),
[US 6,630,077](https://patents.google.com/patent/US6630077B2/en),
[US 8,385,499](https://patents.google.com/patent/US8385499B2/en);
Lanier, [LLNL-TR-587512](https://www.osti.gov/servlets/purl/1053658);
You et al., [Materials 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10179960/);
Zhu et al., [J. Adv. Ceram. 2022](https://d-nb.info/1280575069/34);
Konoshima Chemical, [TAG ceramics](https://www.konoshima.co.jp/eng/ceramics/resdev2.html);
XCIST, [github.com/xcist/main](https://github.com/xcist/main);
Kawahara et al., [Pol. J. Radiol. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10317008/);
Siemens, [Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ b5d65082-f8d7-5b19-a30a-117dcd788835
md"""
## Notebook Setup
"""

# ╔═╡ c7b8ce23-df15-5445-8f9c-1d533b22b055
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ b9afe761-a41a-56bd-89b3-aa4d1f850542
using Markdown: @md_str, Markdown

# ╔═╡ 1114b922-2909-5382-ad68-7870b9f17a49
import PlutoUI

# ╔═╡ 3c7da9cf-d44f-5117-a6cd-ab70462594ba
import BasisMonteCarlo as MC

# ╔═╡ d7980232-34d1-599f-89a1-18c144d7468b
import CairoMakie as Mke

# ╔═╡ 2e1263bc-6a15-5dad-9da8-0c5ea9d9a1ec
import BasisSimulator as BS

# ╔═╡ 6f0a1390-4e16-5007-aff0-205efb24710a
import Serialization

# ╔═╡ 6650624a-9201-5026-9af0-2852cb32ff32
import DelimitedFiles

# ╔═╡ 065c2cd0-3724-5ae0-a368-5c8eb5540a1a
using Statistics: mean

# ╔═╡ 0f1f7711-29fa-5149-b284-b47dcdc295c0
PlutoUI.TableOfContents()

# ╔═╡ e3e9c212-4583-5b56-8c63-55a23aceccc6
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 3b5f881c-54d6-5378-ad04-eefcacc3c8ea
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

# ╔═╡ 425bf6f0-bb29-56ec-9883-45afcf680eda
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

# ╔═╡ 4551bbf4-0c6f-558d-8dd3-19061afbeee1
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

# ╔═╡ d852372b-0f59-5689-a6f2-a96f8756249b
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

# ╔═╡ c8a64c0f-8f57-41f2-9e46-d50e428c1036
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses (the pitch, the reflector,
the crystal's depth, the grid), with a small kit: blocks in 3-D, seen from any direction, shaded
by one light and drawn back to front. The colours are the materials': pale blue Gemstone garnet, white
TiO₂ reflector, grey tungsten, dark-blue silicon photodiodes, amber x rays.
"""

# ╔═╡ 0ed04217-4dc3-4912-ad84-aff29fa419f3
MATERIALS = (
    scintillator = Mke.RGBf(0.74, 0.84, 0.93),  # the Gemstone garnet
    scintillator_lit = Mke.RGBf(0.55, 0.9, 0.62),  # the garnet glowing
    reflector = Mke.RGBf(0.97, 0.97, 0.95),  # TiO₂ epoxy: white
    tungsten = Mke.RGBf(0.42, 0.44, 0.5),
    photodiode = Mke.RGBf(0.24, 0.27, 0.55),  # back-illuminated silicon
    cmos = Mke.RGBf(0.2, 0.22, 0.27),  # the readout electronics
    gold = Mke.RGBf(0.86, 0.66, 0.2),  # vias and stud bumps
    substrate = Mke.RGBf(0.86, 0.83, 0.76),  # ceramics substrate
    board = Mke.RGBf(0.13, 0.4, 0.29),  # circuit board
    water = Mke.RGBf(0.62, 0.78, 0.92),
    beam = Mke.RGBf(0.98, 0.7, 0.25),  # x rays, as in the lab's banner
    light = Mke.RGBf(0.45, 0.9, 0.45),  # the garnet's light (yellow-green)
    night = Mke.RGBf(0.05, 0.06, 0.08),  # the dark backdrop of the lab's banner
    night_ring = Mke.RGBf(0.11, 0.13, 0.16),
)

# ╔═╡ bb748ca7-2f2c-4159-b359-76d52f2f3609
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

# ╔═╡ a9830f78-b3ee-499b-a95a-4bbb68c74dd4
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

# ╔═╡ 511ddd8d-3459-47c6-9b6d-560f6e960f9f
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ c705d51c-e973-48fa-bbd0-d84ad220fc4d
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ a3361458-4c30-464b-8636-bad7c9d5da58
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

# ╔═╡ 61264840-6153-4a9f-b0ae-ce8277b5010b
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ 7bde14bc-c0f6-4574-8629-950a29447b64
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

# ╔═╡ 6359c896-d676-492c-a544-425666580d1d
"""A dexel array's blocks: ceramic in each cell, reflector walls between (in pieces), from z0 to z1."""
function dexel_blocks(nx, ny, pitch, gap, z0, z1; ceramic = MATERIALS.scintillator, lit = nothing)
    blocks = Block[]
    p, s = pitch, gap
    for i in 0:(nx - 1), j in 0:(ny - 1)
        x, y = i * p[1], j * p[2]
        color = lit !== nothing && (i, j) in lit ? MATERIALS.scintillator_lit : ceramic
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

# ╔═╡ 686a8f39-4bb3-4656-8f53-91fb0a7d2254
"""The walls of a 2-D grid over an nx × ny array, in pieces one cell long, from z0 to z1."""
function grid_blocks(nx, ny, pitch, wall, z0, z1; color = MATERIALS.tungsten)
    blocks = Block[]
    p, w = pitch, wall
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

# ╔═╡ 02dcd805-ca95-47a9-9fb2-b0b71942aae5
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

# ╔═╡ a8569152-9721-40e8-9f46-c16279e3fa42
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

# ╔═╡ 3136d195-a95b-4ddb-a868-542dfb8ac793
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

# ╔═╡ af481ff1-3a3f-4045-8d9f-7ee63312785b
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

# ╔═╡ 32fe7f10-9fa0-4cd7-9cb3-51751c3e6f5a
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ a83bfa57-44d8-4b63-aca3-37e5ea4fa1c3
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ 99873a48-bd7a-4996-8e3f-ea3fea7859a6
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

# ╔═╡ c6d7d774-ef13-4f99-83cd-6465fc1f6888
"""A vertical bracket at `x` from `y0` to `y1`, with `text` to its left."""
function side_bracket!(ax, x, y0, y1, text; color, fontsize = 16, tick = 0.08)
    Mke.lines!(ax, Mke.Point2f[(x + tick, y0), (x, y0), (x, y1), (x + tick, y1)]; color, linewidth = 2)
    return Mke.text!(ax, x - tick, (y0 + y1) / 2; text, color, fontsize, font = :bold, align = (:right, :center), justification = :right)
end

# ╔═╡ 08ed4a29-4109-490b-b791-5987c138c173
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

# ╔═╡ b04c0a12-4601-4267-9293-98d2023dafd8
TAGS = (geant4 = Mke.RGBf(0.2, 0.55, 0.32), not_simulated = Mke.RGBf(0.52, 0.52, 0.57))

# ╔═╡ 0085388d-2916-4ebe-9217-8dd1f1d568f2
"""
The blocks of a patch of the Apex's detector, pulled apart layer by layer: the module's board,
its readout electronics (GE does not publish them: drawn generic), the backlit photodiodes, the
Gemstone dexels in their reflector, the grid (drawn `grid_drawn` mm tall). Returns each layer's blocks
(`layers[key]`), its (bottom, top) (`levels[key]`), and all the blocks together (`blocks`).
"""
function exploded_detector(nx, ny; gap = 1.0, grid_drawn = 2.4)
    p, s, d = pitch_mm, septum_mm, depth_mm
    stack = (
        (key = :board, h = 0.3, gap = 0.5),
        (key = :cmos, h = 0.4, gap = 0.55),
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
        elseif layer.key === :cmos
            layer_blocks(nx, ny, p, z0, z1, MATERIALS.cmos)
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

# ╔═╡ a4a8981a-9496-40dd-bfb1-3e0648de4ad0
"""A block drawn faded toward the background (`amount` 0: as is, 1: gone)."""
function faded(b::Block, amount; to = Mke.RGBf(1, 1, 1))
    mix(a, c) = a + amount * (c - a)
    color = Mke.RGBf(mix(b.color.r, to.r), mix(b.color.g, to.g), mix(b.color.b, to.b))
    return Block(; lo = b.lo, hi = b.hi, color, alpha = b.alpha, stroke = b.stroke)
end

# ╔═╡ ee4d93d9-757c-460e-8173-7f28a04f7372
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

# ╔═╡ 71976bd4-43a7-4cca-8bd7-9e41c3071263
"""Outline a box (lo, hi) as seen by `cam`: the edge of its silhouette."""
function outline!(ax, cam, lo, hi; color = MATERIALS.beam, linewidth = 3, linestyle = :solid)
    corners = [(x, y, z) for x in (lo[1], hi[1]) for y in (lo[2], hi[2]) for z in (lo[3], hi[3])]
    return Mke.lines!(ax, hull([project(cam, c) for c in corners]); color, linewidth, linestyle)
end

# ╔═╡ 6d118872-a994-4622-87a2-2a4d2c452d93
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
    R_iso, R_det = apex.source_to_isocenter_mm, apex.source_to_detector_mm
    half_fan = apex.columns * apex.column_at_isocenter_mm / apex.source_to_isocenter_mm / 2
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
        (:ceramic, "2  The Garnet Stops It"),
        (:ceramic, "3  The Garnet Glows"),
        (:photodiode, "4  The Photodiode"),
        (:cmos, "5  The Electronics"),
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

# ╔═╡ 8cec9efb-82c7-4a73-a60c-6a2f4e2d48b9
"""Side view of three dexels (x across, depth down as negative y), at x offsets; returns nothing."""
function side_dexels!(ax; n = 3, lit = nothing, depth = depth_mm, ceramic = MATERIALS.scintillator)
    p, s = pitch_mm[1], septum_mm
    half = (n - 1) / 2
    rect!(ax, -(half + 0.5) * p, -depth, n * p, depth, MATERIALS.reflector; strokecolor = (:black, 0.3), strokewidth = 1)
    for i in 0:(n - 1)
        x = (i - half) * p
        rect!(ax, x - (p - s) / 2, -depth, p - s, depth, lit !== nothing && i == lit ? Mke.RGBf(0.82, 0.95, 0.84) : ceramic)
    end
    return nothing
end

# ╔═╡ a2d50813-d2e4-4a26-8004-7cfbbd80c8bc
"""A curved strip (part of a cylinder about the z axis at the focal spot), as quads with depth, for the painter."""
function arc_quads(cam, R, half_angle, half_z; columns = 46, thickness = 18.0, color = MATERIALS.scintillator)
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

# ╔═╡ 2fc965a3-f475-4d25-b639-e556396e249a
"""An atom (Tb or Lu): nucleus and two shells, at `c` with outer radius `r`."""
function atom!(ax, c, r; color = Mke.RGBf(0.45, 0.47, 0.55))
    θ = range(0, 2π; length = 100)
    for (k, rr) in enumerate((0.55r, r))
        Mke.lines!(ax, c[1] .+ rr .* cos.(θ), c[2] .+ rr .* sin.(θ); color = (color, 0.55), linewidth = 1.5)
    end
    Mke.scatter!(ax, [Mke.Point2f(c)]; color, markersize = 18)
    return nothing
end

# ╔═╡ 9ed4e154-0c35-5252-b2dc-388f22a1c47f
md"""
## 1. The Scintillator

Gemstone is a garnet, (Tb,Lu)₃Al₅O₁₂ activated with cerium: terbium and lutetium share the large
sites of the crystal in a proportion GE does not publish. BasisSimulator v0.15.1's composition splits
them evenly (by atoms) at **7.0 g/cm³**, but no such garnet can be that dense: pure
Tb₃Al₅O₁₂ is 6.07 g/cm³ and pure Lu₃Al₅O₁₂ 6.72 g/cm³, and a mixture lies between (⚠
*discrepancy* above). The even split, interpolated between the two, is **6.40 g/cm³**: this
notebook builds the detector from BasisSimulator's composition at that density, keeps 7.0 to
reproduce the v0.15.1 table, and brackets the unknown composition with the Tb-rich garnet of
GE's patent (6.24 g/cm³).
"""

# ╔═╡ 0565c39d-9720-5863-a0b6-c3e104a475d1
# BasisSimulator v0.15.1's composition (an even Tb/Lu split) at a density such a garnet can have
gemstone = MC.DetectorMaterial(
    "Gemstone 6.40";
    density = 6.4,
    mass_fractions = Dict(
        "Tb" => 0.287,
        "Lu" => 0.316,
        "Al" => 0.163,
        "O" => 0.231,
        "Ce" => 0.002,
    ),
)

# ╔═╡ b5c946ee-f16f-5cca-82cf-b5040fe481b8
# BasisSimulator v0.15.1's Gemstone: the same composition at 7.0 g/cm³
gemstone_basis = MC.DetectorMaterial(:Gemstone)

# ╔═╡ 8fba75c0-6a9f-5bbe-9a10-9d52e2296292
# The Tb-rich garnet of GE's patent, Lu₀.₈Tb₂.₂Al₅O₁₂ (6.24 g/cm³ by the same interpolation)
gemstone_tb_rich = MC.DetectorMaterial(
    "Gemstone Tb-rich";
    density = 6.24,
    mass_fractions = Dict("Tb" => 0.428, "Lu" => 0.171, "Al" => 0.165, "O" => 0.235),
)

# ╔═╡ b349f0be-6494-5c81-80ea-82a481df3b90
# Terbium's and lutetium's K-shells: their edges and the energies of the x rays they emit (Kα)
terbium, lutetium = sort(MC.k_edges(gemstone); by = e -> e.edge_keV)

# ╔═╡ 3e134eba-e90e-5fef-845b-b81b61397dca
md"""
Gemstone has **two K-edges** in the CT spectrum: terbium's at 52.0 keV and lutetium's at
63.3 keV. Above each, absorption jumps, and every absorption in that element's K shell leaves the
atom to emit, most of the time, its K x ray: **44.5 keV for terbium, 54.1 keV for lutetium**.
Each is just below its own edge, so weakly absorbed by its own element, and can travel a
millimetre or more before stopping, about the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ f88f3daf-cbd3-55ec-9361-63ae45a21b33
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

# ╔═╡ d6256293-43fa-5626-bba6-6b1c01af6dda
mu_energies = 10.0:0.25:150.0

# ╔═╡ 2bcabe1a-a3fd-5fa7-8e2a-2a88d9126859
# Geant4's attenuation of Gemstone, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(gemstone, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 7b39f748-54d2-5a67-977a-b81bf47b6b27
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(gemstone, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 74cea183-7970-4071-ae47-c9a7a199c514
let
    μ_per_mm(E) = MC.interpolate_linear(collect(mu_energies), mu_geant4[:total], E)

    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1700, 790))

    red, blue, grey = PALETTE.electron, PALETTE.photon, PALETTE.muted
    panels = (
        ("Photon Absorbed", "Photoelectric: It Vanishes; an Electron and Often\na Tb (44.5 keV) or Lu (54.1 keV) X Ray Leave", red),
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
            Mke.text!(ax, 0.98, -0.75; text = "Tb or Lu X Ray", color = PALETTE.xray, fontsize = 16, align = (:right, :top))
        elseif k == 2
            photon!(ax, (0.3, -0.18), (1.1, -0.72); cycles = 3.5, amplitude = 0.07, color = blue)
            Mke.text!(ax, 1.05, -0.86; text = "Weaker Photon", color = blue, fontsize = 16, align = (:right, :top))
            arrow!(ax, (0.2, 0.2), (0.62, 0.55); color = red, linewidth = 2.5)
            Mke.text!(ax, 0.66, 0.6; text = "Electron", color = red, fontsize = 16, align = (:left, :bottom))
        else
            photon!(ax, (0.35, 0.05), (1.2, 0.28); cycles = 6, amplitude = 0.07, color = MATERIALS.beam)
            Mke.text!(ax, 1.2, 0.42; text = "Same Energy", color = grey, fontsize = 16, align = (:right, :bottom))
        end
        Mke.text!(ax, 0, -0.42; text = "Tb or Lu Atom", color = Mke.RGBf(0.4, 0.42, 0.5), fontsize = 15, align = (:center, :top))
        Mke.Label(fig[2, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end

    # How far a photon goes, on average, in Gemstone: 1/μ, against the crystal's depth
    ax = Mke.Axis(
        fig[1:2, 4];
        title = "How Far a Photon Goes in Gemstone",
        xlabel = "Mean Distance Before Interacting (mm)",
        yticks = (1:4, ["40 keV", "60 keV", "100 keV", "140 keV"]),
        yreversed = true,
        limits = ((0, 1.15depth_mm), (0.4, 4.6)),
        titlesize = 19,
        xlabelsize = 16,
    )
    Mke.vspan!(ax, 0, depth_mm; color = (MATERIALS.scintillator, 0.6))
    Mke.text!(ax, depth_mm / 2, 4.5; text = "The $(depth_mm) mm Crystal", align = (:center, :bottom), fontsize = 15, color = PALETTE.slab_edge)
    for (k, E) in enumerate((40.0, 60.0, 100.0, 140.0))
        L = 1 / μ_per_mm(E)
        Mke.barplot!(ax, [k], [L]; direction = :x, color = MATERIALS.beam, width = 0.5)
        Mke.text!(ax, L, k; text = " $(round(L; digits = 2)) mm", align = (:left, :center), fontsize = 15)
    end
    Mke.colsize!(fig.layout, 4, Mke.Relative(0.24))
    locator!(fig[0, :], :ceramic; note = "Step 2 · The Garnet\nStops the X Ray")
    fig
end

# ╔═╡ 79bf52e2-7776-5cc1-8d0b-4e264aa8530e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How Strongly Gemstone Stops X Rays (6.40 g/cm³)",
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
    for (k, edge) in enumerate((terbium, lutetium))
        Mke.vlines!(
            ax,
            edge.edge_keV;
            color = (PALETTE.xray, 0.8),
            linestyle = :dot,
            linewidth = 2,
        )
        note!(
            ax,
            edge.edge_keV,
            k == 1 ? 0.03 : 0.012,
            "$(edge.element) K-Edge, $(round(edge.edge_keV; digits = 1)) keV";
            offset = (8, 0),
            color = PALETTE.xray,
        )
    end
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

# ╔═╡ d0f1d5ee-5eb0-5fa2-bcd6-74a957ed0358
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

# ╔═╡ ea752deb-d5f8-530f-a2bc-d8f70ac7f9d6
md"""
### A First Check: Siemens' Curve for LuTAG

No absorption curve for Gemstone is published, but Siemens, comparing its own ceramic with its
rivals', plots the fraction of x rays that **1.4 mm of LuTAG** (lutetium terbium aluminium
garnet, Gemstone's family) absorbs, against energy (*Detectors & Crystals*, p. 19). Its steps at
the K-edges go *up*: it is the probability that a photon interacts at all, `1 − exp(−μd)`, so it
depends only on composition, density and depth. Siemens gives neither LuTAG's composition nor
its density, so this is a loose check; but it can tell a garnet that exists from one that cannot.
The values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ ad4b77f0-e633-5a1f-8474-144d368eb858
# Siemens' LuTAG curve, read from the vector figure (reference_data/siemens_lutag_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_lutag_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ 7fea9a31-343e-5084-b697-c614d31d86cc
# The depth of Siemens' figure
siemens_depth_mm = 1.4

# ╔═╡ c6bf2e2b-147e-5c10-a14e-53a1f4421e56
# The probability a photon interacts in 1.4 mm of each candidate garnet
interaction_curves = [
    (
            label = label,
            color = color,
            curve = [1 - exp(-MC.geant4_attenuation(material, E) * siemens_depth_mm) for E in mu_energies],
        )
        for (label, color, material) in (
            (
                "BasisSimulator v0.15.1: Even Split, 7.0 g/cm³ (Impossible)",
                PALETTE.reference,
                gemstone_basis,
            ),
            ("Even Split, 6.40 g/cm³ (Used Here)", PALETTE.photon, gemstone),
            ("Tb-Rich (GE's Patent), 6.24 g/cm³", PALETTE.kept, gemstone_tb_rich),
        )
]

# ╔═╡ 24709fa8-0823-4f19-bcd6-ca2794d35249
let
    μ_per_mm(E) = MC.interpolate_linear(collect(mu_energies), mu_geant4[:total], E)

    d = siemens_depth_mm  # the depth of Siemens' figure
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 500))
    N = 30
    for (k, E) in enumerate((45.0, 90.0, 140.0))
        μ = μ_per_mm(E)
        stopped = 1 - exp(-μ * d)
        ax = canvas(fig[1, k]; title = "$(Int(E)) keV: $(round(Int, 100stopped)) % Stopped", titlesize = 21)
        rect!(ax, 0, 0, 1.6, d, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        # N photons; photon i stops at the depth that the i-th of N quantiles of exp(−μx) gives
        for i in 1:N
            x = 1.6 * (i - 0.5) / N
            depth = -log(1 - (mod(i * 0.618034, 1.0))) / μ
            if depth < d
                Mke.lines!(ax, [x, x], [d + 0.55, d - depth]; color = MATERIALS.beam, linewidth = 2)
                Mke.scatter!(ax, [x], [d - depth]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (x, d + 0.55), (x, -0.5); color = (MATERIALS.beam, 0.6), linewidth = 2, head = 10)
            end
        end
        Mke.text!(ax, 1.65, d / 2; text = "1.4 mm of\nGemstone,\n6.40 g/cm³", align = (:left, :center), fontsize = 15, color = PALETTE.slab_edge)
        Mke.limits!(ax, -0.05, 2.45, -0.62, d + 0.65)
    end
    Mke.Label(
        fig[2, 1:3],
        "Each Line Is a Photon from Above: Red Dots Are Where It Interacts; Arrows Pass Straight Through. The Fraction Stopped Is 1 − exp(−μ · 1.4 mm).";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 7e771ff9-6f1a-5c5b-8469-30e1a7ef0a6e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "The Fraction of Photons 1.4 mm of Garnet Absorbs",
        xlabel = "Photon Energy (keV)",
        ylabel = "Fraction of Photons Absorbed",
        xticks = 20:20:150,
        limits = ((15, 152), (0.3, 1.04)),
    )
    for c in interaction_curves
        Mke.lines!(
            ax,
            mu_energies,
            c.curve;
            color = c.color,
            linewidth = 3,
            label = c.label,
        )
    end
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
        label = "Siemens' LuTAG Figure (± 0.025)",
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ ba7bd8e3-3fe8-51fb-a149-651b7415cecf
let
    high = siemens_curve.energy_keV .>= 80
    rms(c) = sqrt(
        sum(
            abs2,
            [MC.interpolate_linear(collect(mu_energies), c.curve, E) for E in siemens_curve.energy_keV[high]] .-
                siemens_curve.absorption[high],
        ) / count(high),
    )
    rows = ["| $(c.label) | $(round(rms(c); digits = 3)) |" for c in interaction_curves]
    Markdown.parse(
        """
        | garnet, 1.4 mm | rms difference from Siemens' LuTAG figure, 80–150 keV |
        |---|---|
        $(join(rows, "\n"))

        Above 80 keV, where density matters most, every candidate absorbs more than Siemens' LuTAG does, and the
        lighter the garnet, the closer it comes: BasisSimulator v0.15.1's 7.0 g/cm³ is furthest. LuTAG's own composition and
        depth are not published, so this does not pin Gemstone down, but it points the same way as the density limits:
        a lutetium-terbium garnet absorbs less than BasisSimulator v0.15.1's Gemstone.
        """
    )
end

# ╔═╡ 76d9fe06-65d2-50ec-a57f-9d13c0424438
md"""
## 2. One Dexel and Its Neighbours

A dexel is one block of garnet with reflector on its four sides. Its pitch at the detector face
is the Apex's dexel at the isocentre as BasisSimulator v0.15.1 has it, 0.6 mm across the fan by 0.625 mm
along z, magnified by the source-to-detector over the source-to-isocentre distance,
1100 / 625.6. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy is assumed
(TiO₂ is among the reflector materials of GE's patent). The crystal is BasisSimulator v0.15.1's 3.0 mm
deep, inherited from XCIST, and within the 1.5–3 mm cells of GE's reflector patent.
"""

# ╔═╡ 59b35b57-f457-5c5a-89fd-f215ac91ec5f
# The Revolution Apex's geometry, as BasisSimulator v0.15.1 has it
apex = (
    source_to_isocenter_mm = 625.6,
    source_to_detector_mm = 1100.0,
    columns = 834,
    rows = 256,
    column_at_isocenter_mm = 0.6,
    row_at_isocenter_mm = 0.625,
)

# ╔═╡ 87c01464-7920-5151-aaf1-668655c36c98
# The crystal depth: BasisSimulator v0.15.1's (no source gives the Apex's)
depth_mm = 3.0

# ╔═╡ 66875788-7213-5a45-93df-f02a465feab5
# The dexel pitch at the detector face (across the fan, along z), in mm
pitch_mm = (apex.column_at_isocenter_mm, apex.row_at_isocenter_mm) .*
    (apex.source_to_detector_mm / apex.source_to_isocenter_mm)

# ╔═╡ 85332bcb-a653-5577-986e-55557180a527
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ 43a2476d-f0b3-5c4e-ad60-a79093af88bc
# The block of garnet each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ b937f150-98dd-52ad-a1a9-39606e1cc17d
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 69f88251-23f2-5b2f-8e30-f8037e72aabd
let
    fill = crystal_mm ./ pitch_mm
    Markdown.parse(
        """
        The dexel pitch at the face is **$(round(pitch_mm[1]; digits = 3)) × $(round(pitch_mm[2]; digits = 3)) mm**, and
        the garnet block inside it $(round(crystal_mm[1]; digits = 3)) × $(round(crystal_mm[2]; digits = 3)) mm: the crystal
        covers $(round(fill[1]; digits = 3)) × $(round(fill[2]; digits = 3)) of the face, the fill factor of 0.9 × 0.9 that BasisSimulator v0.15.1 uses.
        The assumed reflector and that fill factor are the same detector.
        """
    )
end

# ╔═╡ 3cd39c08-6d7d-5852-911d-70b71149b9ca
# The Geant4 detector: a slab of garnet `depth` deep, cut into 13 × 13 dexels by reflector
# septa (ample for the x rays that leave a dexel), optionally under an anti-scatter grid
function dexel_array(depth; material = gemstone, grid = nothing)
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

# ╔═╡ 13a363b3-f112-59a0-9192-be6a9bef1026
# Photons landing uniformly on one dexel's garnet, and on its whole cell (garnet and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ e340add3-b5f6-5242-93ea-81ef5ba335a2
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 0df4a950-2c66-5f98-827e-d647b3db7d5c
# Each dexel collects the energy left in its own block of garnet (the reflector keeps the light
# in); the dexel the photons land on and its eight neighbours are recorded
readout = MC.DexelEnergyReadout(
    MC.DexelReadout(;
        pitch_x_mm = pitch_mm[1],
        pitch_y_mm = pitch_mm[2],
        neighborhood = 3,
        cloud_sigma_um = 0.01,
        transport = nothing,  # a scintillator: each dexel keeps the energy left in its own ceramic
    ),
)

# ╔═╡ 3a51adf8-1bd6-5027-a06e-b288b5ea7b6e
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
        rect!(axb, i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, p[1] - s, p[2] - s, i == 0 && j == 0 ? Mke.RGBf(0.82, 0.95, 0.84) : MATERIALS.scintillator; strokecolor = PALETTE.slab_edge, strokewidth = 1)
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
    locator!(fig[0, :], :ceramic; note = "Steps 2–3 · The Garnet,\nOne Dexel and Its\nNeighbours")
    fig
end

# ╔═╡ 3d4f918b-7df5-5eed-becb-a6dd61ba726c
md"""
### What a Photon Does in the Garnet

`record_tracks` follows a few hundred 70 keV photons (above both K-edges) landing on the
central dexel, every step of every particle. Two of them are drawn: one whose Tb or Lu x ray is
absorbed **in the next dexel**, so its energy is split between two photodiodes, and one whose
x ray **leaves through the front face**, so its energy is lost to every dexel.
"""

# ╔═╡ bfafebed-31d8-53f5-a471-25993d2ed385
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    70.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ 32ca1abd-3897-5936-a735-a52773f47bf5
# Two photons to follow: one whose Tb or Lu x ray stops in the next dexel across the fan (in the same
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

# ╔═╡ 76876cdf-0539-5b54-a239-98f3e02df836
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    top = -0.9
    fig = Mke.Figure(size = (1300, 520))

    # Clip a step to the drawing (above the garnet it may run off to the world's edge)
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
    axa = Mke.Axis(fig[1, 1]; title = "(a) Its X Ray Stops in the Next Dexel", common...)
    axb = Mke.Axis(fig[1, 2]; title = "(b) Its X Ray Leaves Through the Front", common...)
    panel!(axa, followed.next_door)
    panel!(axb, followed.front)
    Mke.Label(
        fig[2, 1:2],
        "Blue: the 70 keV Photon     Orange: Its Tb or Lu X Ray     Red Dots: Where They Are Absorbed     Green: the Energy Each Dexel Keeps";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ da6b53d3-7e98-4a7d-a74c-20aa703c9633
md"""
### Why Don't the Grid's Walls Go Down Through the Garnet?

Two things can leak from one dexel into the next, and they are stopped (or not) by different parts:

- **Light.** The garnet glows where the x ray stopped; light reaching the next dexel's photodiode
  would blur the image (*optical crosstalk*). The **white TiO₂ reflector** in the saw cuts stops
  it: light bounces back into its own dexel. Geant4 does not follow the light; the reflector is
  assumed perfect, which is what it is for.
- **X rays.** A Tb x ray (44.5 keV), a Lu x ray (54.1 keV) or a Compton-scattered photon crosses 0.1 mm of light epoxy
  almost freely and can stop in the neighbour (*x-ray crosstalk*, the cousin of charge sharing in
  photon-counting detectors). That is the "next door" photon above, and Geant4 follows it.

Tungsten walls through the garnet would stop the x rays, but they would widen the cuts (less
garnet per dexel, so fewer photons stopped for the same dose), add tungsten's own x rays (59 and
67 keV) inside the garnet, and have to be built into the ceramic. GE has patented a combined
reflector and collimator (US 8,385,499), one structure doing both jobs; whether the Apex's walls
reach down between its dexels is not published, so this notebook keeps them in front, standing
on the reflector. And in a scan the
x-ray crosstalk mostly cancels: a dexel gets back from its neighbours what it loses to them
(section 4). What it does cost, a little sharpness and a little noise (the Swank factor), is
measured here. So the grid stands in front, against the scatter from the patient, and only the
reflector goes between the dexels.
"""

# ╔═╡ e76baa84-2c08-4f91-bdd6-f2c94d64fa06
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm[1], septum_mm, depth_mm
    fig = Mke.Figure(size = (1600, 600))
    function base!(ax; septum_color = MATERIALS.reflector, septum = s, walls_through = false)
        rect!(ax, -1.5p, -d, 3p, d, septum_color; strokecolor = (:black, 0.3), strokewidth = 1)
        for i in -1:1
            rect!(ax, i * p - (p - septum) / 2, -d, p - septum, d, i == 0 ? Mke.RGBf(0.82, 0.95, 0.84) : MATERIALS.scintillator)
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

    ax1 = canvas(fig[1, 1]; title = "As Built Here: Grid Above, White Reflector Between", titlesize = 20)
    base!(ax1)
    photon!(ax1, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    half = (p - s) / 2
    Mke.lines!(ax1, Mke.Point2f[hit, (half, -0.15), (0.3, -d)]; color = MATERIALS.light, linewidth = 2.5)
    Mke.lines!(ax1, Mke.Point2f[hit, (-half, -0.3), (-0.25, -d)]; color = MATERIALS.light, linewidth = 2.5)
    photon!(ax1, hit, (1.15p, -1.0); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 11)
    star!(ax1)
    Mke.Label(fig[2, 1], "Light (Green): Bounces Off the Reflector, Stays in Its Dexel\nTb or Lu X Ray (Orange, 44.5 or 54.1 keV): Passes Through the Thin, Light Reflector"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax1, -1.65p, 1.65p, -d - 0.3, 1.75)

    ax2 = canvas(fig[1, 2]; title = "Hypothetical: Tungsten Walls Down Through the Garnet", titlesize = 20)
    base!(ax2; septum = 0.16, walls_through = true)
    photon!(ax2, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    photon!(ax2, hit, (half + 0.02, -0.82); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 11)
    Mke.scatter!(ax2, [Mke.Point2f(half + 0.06, -0.85)]; marker = :xcross, markersize = 18, color = PALETTE.xray)
    star!(ax2)
    Mke.Label(fig[2, 2], "Stops the Tb or Lu X Ray, but: Wider Cuts Leave Less Garnet per Dexel,\nTungsten Adds Its Own X Rays (59 and 67 keV), and It Must Be Built In"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax2, -1.65p, 1.65p, -d - 0.3, 1.75)
    fig
end

# ╔═╡ fdb32583-deac-5102-80c8-d5c90e587dc2
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is a spread: the full
photon energy when everything stays in the dexel, the energy minus 44.5 keV when a terbium x ray
leaves, minus 54.1 keV when a lutetium x ray leaves, a little when the photon only
Compton-scatters once and leaves. Three energies show the regimes: 50 keV (below both K-edges),
70 keV (above both) and 120 keV.
"""

# ╔═╡ a2a74ae1-42e5-5360-b821-fa1c5835af23
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
        for E in (50.0, 70.0, 120.0)
)

# ╔═╡ d465f476-4b7a-5a26-8485-e167450cf6e3
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ 2c48c3e8-a7a4-496c-85c1-674afac1c9d1
let
    Mke.set_theme!(THEME)
    p, d = pitch_mm[1], depth_mm
    fig = Mke.Figure(size = (1600, 810))

    kα = 43.0
    cases = (
        ("Everything Stays", "The Tb or Lu X Ray Is Reabsorbed\nin the Same Dexel → a Peak at E", PALETTE.kept),
        ("Its X Ray Escapes", "44.5 (Tb) or 54.1 (Lu) keV Leave → Peaks\nat E − 44.5 and E − 54.1 keV", PALETTE.xray),
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

# ╔═╡ be0eef36-6795-5094-814c-7838d52f5e92
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 520))
    for (k, E) in enumerate((50.0, 70.0, 120.0))
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
        for (edge, y) in ((terbium, 0.3), (lutetium, 0.05))
            E > edge.edge_keV || continue
            note!(
                ax,
                E - edge.kalpha_keV,
                y,
                "$(edge.element) X Ray Lost:\nE − $(round(edge.kalpha_keV; digits = 1)) keV";
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
                0.002,
                "Compton Scatters\nThat Left";
                align = (:center, :center),
                color = PALETTE.photon,
                fontsize = 14,
            )
        end
    end
    fig
end

# ╔═╡ fb4ffcd0-bac4-5ce3-858c-c45c2085c83b
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the garnet or
leaves it: as a Tb or Lu x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of garnet (no reflector), so that whatever
leaves the garnet leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ 71ce3fe1-447b-54b5-8ded-a897f1c8636e
budget_energies = 15.0:5.0:140.0

# ╔═╡ c1642d2c-c743-50d8-8fa0-b307a93526da
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(gemstone, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ d6c44cff-e7b9-4790-b287-f784595829ce
let
    Mke.set_theme!(THEME)
    d = depth_mm
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1])
    rect!(ax, -6, -d, 12, d, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.text!(ax, 0, -d - 0.12; text = "One Continuous Slab of Gemstone, $(d) mm Deep (No Reflector, So What Leaves Is Gone)", fontsize = 15, color = PALETTE.slab_edge, align = (:center, :top))
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
        (-4.5, "Stays in\nthe Garnet", PALETTE.kept),
        (-1.5, "Leaves as a Tb\nor Lu X Ray", PALETTE.xray),
        (1.5, "Leaves as a\nScattered Photon", PALETTE.photon),
        (4.5, "Passes Straight\nThrough", PALETTE.muted),
    )
    for (x, text, color) in labels
        Mke.text!(ax, x, 1.75; text, color, fontsize = 18, font = :bold, align = (:center, :bottom))
    end
    Mke.limits!(ax, -6.3, 6.3, -d - 1.6, 2.75)
    fig
end

# ╔═╡ 590bc8d2-76b3-5e54-9bc8-d8aa7104552e
let
    Mke.set_theme!(THEME)
    E = collect(budget_energies)
    parts = (
        (
            values = [b.absorbed for b in budgets],
            label = "Stays in the Garnet (the Signal)",
            color = PALETTE.kept,
        ),
        (
            values = [b.fluorescence for b in budgets],
            label = "Leaves as a Tb or Lu X Ray",
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
        title = "Where the Photon's Energy Goes: $(depth_mm) mm of Gemstone",
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
        [terbium.edge_keV, lutetium.edge_keV];
        color = PALETTE.ink,
        linestyle = :dot,
        linewidth = 1.5,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 90bbeea9-e955-5350-9c91-055bb731efb2
let
    k = findfirst(>(lutetium.edge_keV), collect(budget_energies))
    b = budgets[k]
    closure = maximum(abs(b.total - 1) for b in budgets)
    Markdown.parse(
        """
        Just above the Lu K-edge, at $(Int(budget_energies[k])) keV, **$(round(100b.fluorescence; digits = 1)) %** of the energy leaves
        as Tb or Lu x rays and $(round(100b.absorbed; digits = 1)) % stays. Gemstone is $(depth_mm) mm deep, so far fewer photons pass
        straight through than in the thinner Siemens ceramics. The energy balance closes to within
        $(round(100closure; digits = 2)) % at every energy.
        """
    )
end

# ╔═╡ 76eecb6b-bcfa-5b64-820a-0ab315fad43f
md"""
## 4. The Efficiency and Noise of One Dexel

The full simulation: at every whole keV from 1 to 140 (the grid of BasisSimulator's tables),
100 000 photons land uniformly on one dexel's garnet, and each run is reduced, where it ran, to
the energy every dexel of the 3 × 3 collected and the energy left anywhere in the garnet. From
them come four curves:

- **interaction probability**: the photon interacts at all (`1 − exp(−μd)`, the check);
- **one-dexel efficiency**: the energy the dexel the photon landed on keeps, ÷ E;
- **flat-field efficiency**: the energy left anywhere in the garnet, ÷ E. In a scan every dexel
  is illuminated, and what one dexel loses to its neighbours it gets back from them, so this is
  what each dexel collects per photon landing on it: **the efficiency a CT detector has**;
- **Swank factor** `I = M₁² / (M₀ M₂)` of the flat-field signal: the noise the spread of
  collected energies adds. A detector's zero-frequency DQE is `M₁² / M₂` per incident photon, the
  interaction probability times `I`.
"""

# ╔═╡ 27dc6d6f-6037-5f6a-b642-1c5af6a9bc4d
energies = 1.0:1.0:140.0

# ╔═╡ 05c86c98-403d-55b7-ae80-f23a4cbbfcd5
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 651673f4-ac81-5aa2-9b6c-4d3a1f232562
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ 7dd51815-ccc2-492a-81f2-80a0e1229257
let
    Mke.set_theme!(THEME)
    p, s = pitch_mm, septum_mm
    fig = Mke.Figure(size = (1600, 850))

    lit_colour = Mke.RGBf(0.82, 0.95, 0.84)
    function top!(ax, lit)
        rect!(ax, -2.5p[1], -2.5p[2], 5p[1], 5p[2], Mke.RGBf(0.93, 0.93, 0.9))
        for i in -2:2, j in -2:2
            rect!(ax, i * p[1] - (p[1] - s) / 2, j * p[2] - (p[2] - s) / 2, p[1] - s, p[2] - s, lit(i, j) ? lit_colour : MATERIALS.scintillator; strokecolor = (PALETTE.slab_edge, 0.5), strokewidth = 1)
        end
        return
    end
    ax1 = canvas(fig[1, 1]; title = "One Dexel Lit", titlesize = 20)
    top!(ax1, (i, j) -> i == 0 && j == 0)
    for (dx, dy) in ((1, 0), (0, 1), (-1, 0), (0, -1))
        photon!(ax1, (0.15dx * p[1], 0.15dy * p[2]), (0.95dx * p[1], 0.95dy * p[2]); cycles = 2, amplitude = 0.06, color = PALETTE.xray, head = 11)
    end
    Mke.Label(fig[2, 1], "Its Tb and Lu X Rays Land in the Neighbours:\nLost to This Dexel (the Dashed Curve Below)"; fontsize = 16, color = PALETTE.muted, tellwidth = false)
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
    locator!(fig[0, :], :ceramic; note = "Steps 2–3 · The Whole\nGarnet Layer, Many\nDexels at Once")
    fig
end

# ╔═╡ fdb38e4c-3f74-5de4-be98-cdda136f1a56
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1400, 600))
    axa = Mke.Axis(
        fig[1, 1];
        title = "(a) Efficiency, $(depth_mm) mm of Gemstone",
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
        terbium.edge_keV,
        0.75,
        "Tb and Lu\nK-Edges";
        align = (:right, :center),
        offset = (-8, 0),
        color = PALETTE.xray,
        fontsize = 15,
    )
    Mke.vlines!(
        axa,
        [terbium.edge_keV, lutetium.edge_keV];
        color = (PALETTE.xray, 0.6),
        linestyle = :dot,
    )
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
    Mke.vlines!(
        axb,
        [terbium.edge_keV, lutetium.edge_keV];
        color = (PALETTE.xray, 0.6),
        linestyle = :dot,
    )
    Mke.axislegend(axb; position = :lb)
    fig
end

# ╔═╡ f192b404-d795-5a67-8bf5-87aa47aeda2e
let
    at(E) = findfirst(==(E), collect(energies))
    k70, k140 = at(70.0), at(140.0)
    beer = 1 - exp(-MC.geant4_attenuation(gemstone, 140.0) * depth_mm)
    Markdown.parse(
        """
        At 70 keV, above both K-edges, the dexel a photon lands on keeps **$(round(efficiency.one_dexel[k70]; digits = 3))** of its
        energy; with every dexel illuminated each collects **$(round(efficiency.flat_field[k70]; digits = 3))**: the Tb and Lu x rays a
        dexel loses to its neighbours, it gets back from them. The Swank factor there is $(round(efficiency.swank[k70]; digits = 3)).
        At 140 keV the photon interacts with probability $(round(efficiency.interaction[k140]; digits = 3)) (Beer–Lambert:
        $(round(beer; digits = 3))), and the flat-field efficiency is $(round(efficiency.flat_field[k140]; digits = 3)).
        """
    )
end

# ╔═╡ 243b6cd3-d4a7-5e9a-a7d0-39a424919164
md"""
### How Much the Unknown Composition Matters

The garnet's Tb/Lu split is the one material number nobody publishes. The flat-field efficiency
at 3.0 mm for the even split (6.40 g/cm³, used here), for the Tb-rich garnet of GE's patent
(6.24 g/cm³), and for BasisSimulator v0.15.1's impossible 7.0 g/cm³:
"""

# ╔═╡ ee6081a3-7bb9-5320-a338-909cf1251027
composition_runs = Dict(
    label => MC.energy_absorption_efficiency.(
            MC.simulate_sweep(
                [(dexel_array(depth_mm; material), E) for E in energies];
                n = 50_000,
                beam = crystal_beam,
                seed = 23,
                readout,
            ),
        )
        for (label, material) in (("Tb-rich, 6.24", gemstone_tb_rich), ("even split, 7.0", gemstone_basis))
)

# ╔═╡ 739e2f06-3ac1-41ba-b800-c95f4274a083
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 430))
    elements = (("Tb", 65), ("Lu", 71), ("Al", 13), ("O", 8), ("Ce", 58))  # mass fractions are keyed by Z
    colors = (PALETTE.xray, PALETTE.electron, Mke.RGBf(0.75, 0.77, 0.8), PALETTE.photon, PALETTE.kept)
    garnets = (
        ("Even Split, 6.40 g/cm³", gemstone, "Used Here"),
        ("Tb-Rich (GE's Patent), 6.24 g/cm³", gemstone_tb_rich, "The Bracket"),
        ("Even Split, 7.0 g/cm³", gemstone_basis, "BasisSimulator v0.15.1 (Impossible)"),
    )
    ax = Mke.Axis(
        fig[1, 1];
        title = "Three Garnets: What Each Is Made of, by Mass",
        xlabel = "Fraction of the Garnet's Mass",
        yticks = (1:3, [g[1] for g in garnets]),
        yreversed = true,
        limits = ((0, 1), (0.4, 3.6)),
    )
    for (k, (_, material, note)) in enumerate(garnets)
        left = 0.0
        for ((element, Z), color) in zip(elements, colors)
            f = get(material.mass_fractions, Z, 0.0)
            f > 0 || continue
            Mke.poly!(ax, Mke.Rect2f(left, k - 0.3, f, 0.6); color, strokecolor = :white, strokewidth = 1.5)
            f > 0.06 && Mke.text!(ax, left + f / 2, k; text = element, align = (:center, :center), fontsize = 16, font = :bold, color = :white)
            left += f
        end
        note!(ax, 0.99, k - 0.42, note; align = (:right, :center), fontsize = 14, color = PALETTE.muted)
    end
    Mke.Label(
        fig[2, 1],
        "Tb and Lu Do the Stopping (Their K-Edges at 52.0 and 63.3 keV); GE Does Not Publish Its Split, So Two Are Simulated.";
        fontsize = 16,
        color = PALETTE.muted,
    )
    fig
end

# ╔═╡ 7f4daa38-7a47-5d7f-81c6-2aa821f3715e
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 520))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Flat-Field Efficiency at 3.0 mm, Three Garnets",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.6, 1.02)),
    )
    Mke.lines!(
        ax,
        E,
        composition_runs["even split, 7.0"];
        color = PALETTE.reference,
        linewidth = 2.5,
        linestyle = :dash,
        label = "Even Split, 7.0 g/cm³ (BasisSimulator v0.15.1, Impossible)",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.flat_field;
        color = PALETTE.photon,
        linewidth = 3.5,
        label = "Even Split, 6.40 g/cm³ (Used Here)",
    )
    Mke.lines!(
        ax,
        E,
        composition_runs["Tb-rich, 6.24"];
        color = PALETTE.kept,
        linewidth = 2.5,
        label = "Tb-Rich (GE's Patent), 6.24 g/cm³",
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 208a44d2-987f-5887-b584-1f976fcc90a5
let
    E = collect(energies)
    spread = maximum(abs.(composition_runs["Tb-rich, 6.24"] .- efficiency.flat_field))
    density = maximum(abs.(composition_runs["even split, 7.0"] .- efficiency.flat_field))
    Markdown.parse(
        """
        Between the two possible garnets the flat-field efficiency differs by at most **$(round(spread; digits = 3))**
        (largest where the Tb and Lu edges sit); BasisSimulator v0.15.1's impossible density moves it by up to
        $(round(density; digits = 3)). Both possible garnets are saved with the table.
        """
    )
end

# ╔═╡ 56b952ce-233b-5076-a0ad-fa6756455914
md"""
### A Check Against Published Monte Carlo

The efficiency above the K-edges is shaped by fluorescence escape. No benchmark exists for a
garnet, but Boone et al. (Med. Phys. 1999) followed the same physics in thin Gd₂O₂S screens with
the TART 98 code: a pencil beam at normal incidence on Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S), the energy tallied in a circle 0.1 mm across the beam and in rings out to
4.5 mm, and ideal sensors in front of the screen and behind it. At 55 keV they report (p. 908)
how the incident energy divides. The same screen and the same tallies in Geant4:
"""

# ╔═╡ 1d5ea56f-89a5-535f-a2ab-dce36fcb20a7
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

# ╔═╡ 322cca7c-08ea-4928-90bf-2ef4628f918b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 470))
    ax = canvas(fig[1, 1]; title = "Side View (Not to Scale)", titlesize = 20)
    rect!(ax, -3, -0.12, 6, 0.24, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
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
    Mke.poly!(ax2, Mke.Point2f.(4.5 .* cos.(θ), 4.5 .* sin.(θ)); color = (MATERIALS.scintillator, 0.6), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    Mke.poly!(ax2, Mke.Point2f.(0.45 .* cos.(θ), 0.45 .* sin.(θ)); color = PALETTE.kept)
    Mke.text!(ax2, 0, 0.6; text = "Within 0.1 mm:\n\"Absorbed as Primary\"", align = (:center, :bottom), fontsize = 15, color = PALETTE.kept)
    Mke.text!(ax2, 0, -3.3; text = "0.1 to 4.5 mm: \"Lateral\"\n(Reabsorbed Fluorescence)", align = (:center, :center), fontsize = 15, color = PALETTE.slab_edge)
    Mke.text!(ax2, 4.6, 3.6; text = "(Circle Enlarged)", align = (:left, :center), fontsize = 13, color = PALETTE.muted)
    Mke.limits!(ax2, -5, 7, -5, 5)
    fig
end

# ╔═╡ 8ec58a1a-ddc9-51e5-a0e4-6ced725ac8fe
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

# ╔═╡ 4ace8ef9-4637-5260-9f14-5f7685e76537
md"""
## 5. Compared with BasisSimulator v0.15.1's Table

The v0.15.1 table is an energy-absorption efficiency (its K-edge steps go down, as
fluorescence escape makes them), for a crystal BasisSimulator v0.15.1 documents as 3.0 mm deep. A photon
that never interacts leaves no energy, so the probability of interacting at all is a ceiling on
the efficiency.
"""

# ╔═╡ 797de5bb-99b6-5094-a316-d8ca2bfb284f
# BasisSimulator v0.15.1's table (MCNP), as BasisMonteCarlo keeps it for comparison: later
# BasisSimulator releases use this notebook's table instead
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.gemstone

# ╔═╡ e165adb1-2895-48f3-b063-326d1ef88539
# Is BasisMonteCarlo's copy the table the installed BasisSimulator uses? Up to v0.15.1 it must be,
# exactly; from v0.16.0 BasisSimulator uses this notebook's table, and the copy stands alone
let
    installed = pkgversion(BS)
    same = isdefined(BS, :GEMSTONE_MC_EFFICIENCY_LUT) &&
        BS.GEMSTONE_MC_EFFICIENCY_LUT.energies == table.energies &&
        BS.GEMSTONE_MC_EFFICIENCY_LUT.efficiency == table.efficiency
    message = if installed <= v"0.15.1"
        same ?
            "✓ The installed BasisSimulator, v$(installed), uses exactly this table." :
            "✗ **The installed BasisSimulator, v$(installed), has a different table from BasisMonteCarlo's copy of v0.15.1's.** Check `BASISSIMULATOR_V0_15_1_EFFICIENCY` against BasisSimulator v0.15.1."
    else
        "The installed BasisSimulator is v$(installed), which no longer uses this table. The comparisons below are with BasisSimulator v0.15.1's table, which BasisMonteCarlo keeps (`BASISSIMULATOR_V0_15_1_EFFICIENCY`): they do not change with BasisSimulator's version."
    end
    Markdown.parse(message)
end

# ╔═╡ 6830b6a0-8ae0-5e43-80e9-da1cc552508c
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ ac337b7c-b33c-4a34-9917-e0d030638c52
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 850))
    N = 30
    E = 140.0
    cases = (
        (gemstone_basis, "7.0 g/cm³: BasisSimulator v0.15.1's Garnet", "Denser Than Any\n(Tb,Lu)₃Al₅O₁₂ Can Be"),
        (gemstone, "6.40 g/cm³: the Same Garnet, a Density It Can Have", "Used Here"),
    )
    for (k, (material, title, note)) in enumerate(cases)
        μ = MC.geant4_attenuation(material, E)
        stopped = 1 - exp(-μ * depth_mm)
        ax = canvas(fig[1, k]; title, titlesize = 20)
        rect!(ax, 0, 0, 1.6, depth_mm, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
        for i in 1:N
            x = 1.6 * (i - 0.5) / N
            z = -log(1 - mod(i * 0.618034, 1.0)) / μ
            if z < depth_mm
                Mke.lines!(ax, [x, x], [depth_mm + 0.55, depth_mm - z]; color = MATERIALS.beam, linewidth = 2)
                Mke.scatter!(ax, [x], [depth_mm - z]; color = PALETTE.electron, markersize = 9)
            else
                arrow!(ax, (x, depth_mm + 0.55), (x, -0.5); color = (MATERIALS.beam, 0.6), linewidth = 2, head = 10)
            end
        end
        Mke.text!(
            ax,
            1.68,
            depth_mm / 2;
            text = "$(round(Int, 100stopped)) % of\n140 keV Photons\nInteract in\n$(depth_mm) mm\n\n$(note)",
            align = (:left, :center),
            fontsize = 16,
            color = PALETTE.ink,
        )
        Mke.limits!(ax, -0.05, 3.3, -0.62, depth_mm + 0.65)
    end
    Mke.Label(
        fig[2, 1:2],
        "Any (Tb,Lu)₃Al₅O₁₂ Lies Between 6.07 g/cm³ (Pure Tb) and 6.72 g/cm³ (Pure Lu): the v0.15.1 Table's Garnet Stops More Than the Real One Can.";
        fontsize = 16,
        color = PALETTE.muted,
    )
    locator!(fig[0, :], :ceramic; note = "Step 2 · How Dense\nIs the Garnet?")
    fig
end

# ╔═╡ f27dd4ed-4167-5bd4-ac84-a8f98e88754b
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(gemstone_basis, e) * depth_mm) for e in E]
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "BasisSimulator v0.15.1's Table Against 3.0 mm of Gemstone",
        xlabel = "Photon Energy (keV)",
        ylabel = "Efficiency",
        xticks = 0:20:140,
        limits = ((0, 141), (0.55, 1.06)),
    )
    Mke.band!(ax, E, ceiling, fill(1.06, length(E)); color = (PALETTE.electron, 0.08))
    Mke.lines!(
        ax,
        E,
        ceiling;
        color = PALETTE.electron,
        linewidth = 2,
        linestyle = :dot,
        label = "Ceiling: 3.0 mm Interacts at All (7.0 g/cm³)",
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
        label = "Geant4, 3.0 mm, Flat Field",
    )
    Mke.lines!(
        ax,
        E,
        efficiency.one_dexel;
        color = PALETTE.muted,
        linewidth = 2,
        linestyle = :dash,
        label = "Geant4, 3.0 mm, One Dexel",
    )
    note!(
        ax,
        120,
        1.02,
        "Impossible for 3.0 mm";
        align = (:center, :center),
        color = PALETTE.electron,
        fontsize = 15,
    )
    Mke.axislegend(ax; position = :lb)
    fig
end

# ╔═╡ 2ad68ded-ca19-504d-a805-f0caf48d398b
let
    E = collect(energies)
    ceiling = [1 - exp(-MC.geant4_attenuation(gemstone_basis, e) * depth_mm) for e in E]
    margin = minimum(ceiling .- table_at.(E))
    Markdown.parse(
        """
        Unlike the Siemens tables, this one stays under its ceiling at every energy (closest at $(Int(E[argmin(ceiling .- table_at.(E))])) keV,
        $(round(margin; digits = 3)) below it): it can describe a 3.0 mm crystal of its material. But it sits below Geant4's
        flat field above the K-edges, the signature of one dexel under a pencil beam. Which crystal does it describe?
        """
    )
end

# ╔═╡ 23cc9c86-3ad7-5182-a4e2-9857fcf1c6f2
md"""
### Which Crystal Did the Table Describe?

The table's shape fixes how it was made. A pencil beam into the centre of one dexel, tallied in
that dexel, gives its deep step at the K-edge (with a uniform beam or a flat field the step is
shallower); the fall-off at high energy then fixes the depth. So: pencil beams at 60–140 keV into
dexels 1.5 to 4.0 mm deep, in BasisSimulator v0.15.1's garnet at its 7.0 g/cm³, compared with the
table, with one constant factor for the table's flat 1 % loss at low energy.
"""

# ╔═╡ 2f3c0378-a2dd-5a8a-aac2-8ecba127fdc3
scan_depths = 1.5:0.25:4.0

# ╔═╡ 1ae1a7b5-befc-57dd-8506-a3904c6e56e2
scan_energies = 60.0:5.0:140.0

# ╔═╡ 876a93ce-7ca7-5f15-a44d-80a8c18bd676
# One column per depth, one row per energy
pencil_scan = reshape(
    MC.simulate_sweep(
        [
            (dexel_array(depth; material = gemstone_basis), E)
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

# ╔═╡ 24568e75-698d-5a5e-9401-f553959c2742
# How far one depth's pencil-beam efficiencies are from the table, after the best constant factor
function misfit(runs, target)
    ours = MC.dexel_efficiency.(runs)
    scale = sum(ours .* target) / sum(ours .^ 2)
    rms = sqrt(sum(abs2, scale .* ours .- target) / length(target))
    return (scale = scale, rms = rms)
end

# ╔═╡ e41cfdd8-b032-5c29-abef-a5c856dec82f
fit = let
    target = table_at.(scan_energies)
    misfits = [misfit(pencil_scan[:, k], target) for k in eachindex(scan_depths)]
    best = argmin([m.rms for m in misfits])
    (depth_mm = scan_depths[best], scale = misfits[best].scale, misfits = misfits)
end

# ╔═╡ ba31b322-bf2a-57ca-8068-9858e8ac5d00
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
    Mke.vlines!(ax, depth_mm; color = PALETTE.reference, linestyle = :dash, linewidth = 2)
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
        depth_mm,
        0.3,
        "BasisSimulator v0.15.1 Documents 3.0 mm";
        align = (:left, :center),
        offset = (8, 0),
        color = PALETTE.reference,
        fontsize = 15,
    )
    fig
end

# ╔═╡ 05b60d54-a668-5497-9fc5-dcedf626515e
# The table's recipe at the depth that fits, at every energy
reproduction = MC.simulate_sweep(
    [(dexel_array(fit.depth_mm; material = gemstone_basis), E) for E in energies];
    n = 100_000,
    beam = MC.PencilBeam(; start_mm = 12.0),
    seed = 15,
    readout,
)

# ╔═╡ d067a0b4-d46d-5a21-a351-d3c7e347f9b7
reproduced = fit.scale .* MC.dexel_efficiency.(reproduction)

# ╔═╡ 54179647-5aa9-520a-a0ce-933b5e136cbf
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

# ╔═╡ 25b60608-28d7-5442-89df-fb1e7ac471c8
let
    E = collect(energies)
    difference = reproduced .- table_at.(E)
    away_from_edge = abs.(E .- terbium.edge_keV) .> 0.5
    typical = sum(abs, difference) / length(difference)
    Markdown.parse(
        """
        The table is reproduced to **$(round(typical; digits = 4))** on average by a pencil beam into the centre of one dexel,
        tallied in that dexel, in BasisSimulator v0.15.1's garnet at 7.0 g/cm³ **$(fit.depth_mm) mm deep**, times $(round(fit.scale; digits = 3))$(abs(fit.depth_mm - depth_mm) <= 0.25 ? ": within one step of the scan of the 3.0 mm BasisSimulator v0.15.1 documents" : ", not the 3.0 mm BasisSimulator v0.15.1 documents").
        Away from 52 keV the largest difference is $(round(maximum(abs, difference[away_from_edge]); digits = 3)).
        At 52 keV, on the terbium K-edge (51.996 keV), Geant4 places the photon 4 eV above the edge and the table
        below it: one point, a few eV of edge energy or a shifted energy grid in the MCNP run.

        So the v0.15.1 table is BasisSimulator v0.15.1's documented detector, with two problems: **⚠ its density is impossible**
        (section 1), and it is one dexel under a pencil beam rather than the flat field a scan sees (section 4).
        """
    )
end

# ╔═╡ 55dad12b-a818-55c3-b5ba-aea9be0cd98c
md"""
## 6. The Anti-Scatter Grid

GE calls it a **"3D collimator"**: a two-dimensional grid behind the patient, walls between
columns and between rows, focused on the focal spot. Its product dimensions are not published;
GE's patent for a combined reflector and collimator (US 8,385,499) gives walls **0.10–0.20 mm**
thick, **about 8 mm** tall, with "a pitch of channels identical to a pitch of detector
elements", made of tungsten (or lead, tantalum) powder in a binder. Built here: **solid
tungsten walls 0.15 mm thick and 8 mm tall**, one cell per dexel, standing on the reflector.
0.15 mm walls on 0.1 mm of reflector shade a 25 µm strip of garnet on each side.
"""

# ╔═╡ d89950fe-ea70-592b-998b-3e954534f5c9
grid = MC.Collimator(;
    height_mm = 8.0,
    wall_mm = 0.15,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :both,
    cells = 13,
)

# ╔═╡ 6b89ebe5-d2ca-5d77-93e1-ec26b393c271
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    H, w = grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1600, 1050))


    # (a) to scale, the near corner cut away
    ax = canvas(fig[1, 1]; title = "(a) The 3D Collimator over the Dexels, to Scale", titlesize = 20)
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
    callout!(ax, project(cam, angled[2]), project(cam, angled[2]) + Mke.Point2f(3.6, -2.2), "Scattered in the Patient:\nStopped by a Wall"; color = PALETTE.xray, fontsize = 15)
    callout!(ax, project(cam, (nx * p[1], ny * p[2], d + H / 2)), project(cam, (nx * p[1], ny * p[2], d + H / 2)) + Mke.Point2f(1.4, 4.2), "Tungsten Walls\n$(round(Int, 1000w)) µm Thick,\n$(round(Int, H)) mm Tall"; fontsize = 15)
    Mke.xlims!(ax, nothing, project(cam, (nx * p[1], 0, 0))[1] + 9.5)

    # (b) side view, to scale, with the angle and the shadow a wall casts
    axb = Mke.Axis(
        fig[1, 2];
        title = "(b) One Channel, Side View (Width Stretched)",
        titlesize = 20,
        xlabel = "Across the Fan (mm)",
        ylabel = "Height Above the Garnet (mm)",
        limits = ((-0.75p[1], 1.05p[1]), (-d - 0.2, H + 1.4)),
    )
    rect!(axb, -0.5p[1], -d, p[1], d, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge)
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
    Mke.bracket!(axb, xr - reach, H + 0.1, xr, H + 0.1; text = "H tan θ: These Hit the Wall", orientation = :up, fontsize = 14, color = PALETTE.xray, textcolor = PALETTE.xray)
    Mke.bracket!(axb, 0.5p[1] + 0.12, 0, 0.5p[1] + 0.12, H; text = "H = $(round(Int, H)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.Label(fig[2, 2], "Photons Arriving θ = $(round(Int, θ))° off the Channel's Axis:\nThe Fraction 1 − H tan θ / Opening Reaches the Garnet"; fontsize = 15, color = PALETTE.ink, tellwidth = false)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.32))
    locator!(fig[0, :], :grid; note = "Step 1 · GE's 3D Collimator\nIn Front of the Garnet")
    fig
end

# ╔═╡ c599896a-d8dc-5d31-8aa6-d04f7eb6bf7e
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (garnet and reflector), with and without the grid: the grid
should cost only the strip of garnet its walls shade, and its own tungsten x rays (59 and
67 keV) reach the garnet above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ 367261c6-ffb8-5661-970e-6494b0a10197
grid_energies = 20.0:5.0:140.0

# ╔═╡ 8c9ab3eb-23b7-57a5-9c13-0aa983dae5e0
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

# ╔═╡ 387074f5-5299-5768-b50a-e1a315621ab8
angles_deg = 0.0:1.0:20.0

# ╔═╡ 173503d6-98e5-51d2-b783-7e5885c408eb
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

# ╔═╡ 8c9677c3-aac1-593c-a53c-ccc5218c3b3e
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
        label = "With the 2-D Grid",
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
        label = "With the 2-D Grid",
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

# ╔═╡ 9af6dffc-d99a-54bf-97cf-dd20156bbe48
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) %: the walls are 50 µm wider than the reflector they stand on, so the face open to the
        beam is $(round(prod((pitch_mm .- grid.wall_mm) ./ pitch_mm); digits = 3)) of the cell, against $(round(prod(crystal_mm ./ pitch_mm); digits = 3)) of garnet without the grid. If
        GE's walls are as thick as its patent's, BasisSimulator v0.15.1's fill factor (0.81) overstates the collecting face.
        (Above 69.5 keV the walls' own x rays add a little.)
        At an angle it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %. A
        photon that scatters in the patient arrives from anywhere, so most of it is stopped.
        """
    )
end

# ╔═╡ 89f83a80-82e6-5d12-8616-36560dba03a1
md"""
## 7. The Whole Detector

The whole detector, as BasisSimulator v0.15.1 has it: 834 columns × 256 rows on an arc of radius 1100 mm
around the focal spot, 160 mm along z, every dexel a block of garnet in reflector, under the 2-D
grid, all of it focused on the focal spot (GE's "focally aligned" sub-modules). An air scan fires
photons uniformly from the focal spot over the whole detector, and every dexel's signal is
summed. Two questions: is every dexel the same, and does the whole detector collect what the
single dexel of sections 4 and 6 predicts?
"""

# ╔═╡ 8074b52c-eaab-5eef-ab41-b3482d729940
detector = MC.ArcDetector(;
    columns = apex.columns,
    rows = apex.rows,
    radius_mm = apex.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :both,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = gemstone,
)

# ╔═╡ bd408664-f831-56e0-971a-4d7c4f14fa33
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ 92105c08-3d1a-533c-8939-3c30460be4b4
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

# ╔═╡ 7389b76e-a842-4021-bbca-ff56f1e40825
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 990))

    ax = canvas(fig[1, 1]; title = "The Whole Detector, from the Focal Spot (Depth Exaggerated)", titlesize = 20)
    cam = camera(-62, 26)
    R = apex.source_to_detector_mm
    half = apex.columns * apex.column_at_isocenter_mm / apex.source_to_isocenter_mm / 2
    half_z = apex.rows * apex.row_at_isocenter_mm * R / apex.source_to_isocenter_mm / 2
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
                iso = (0.0, -apex.source_to_isocenter_mm, 0.0)
                Mke.scatter!(a, [project(cam, iso)]; marker = :cross, markersize = 18, color = PALETTE.ink)
            end,
        )
    )
    sort!(items; by = first)
    for (_, f) in items
        f(ax)
    end
    iso = (0.0, -apex.source_to_isocenter_mm, 0.0)
    callout!(ax, project(cam, focal), project(cam, focal) + Mke.Point2f(150, 80), "Focal Spot"; fontsize = 16)
    callout!(ax, project(cam, iso), project(cam, iso) + Mke.Point2f(260, 60), "Isocentre, $(round(Int, apex.source_to_isocenter_mm)) mm Away"; fontsize = 16)
    edge = (R * sin(half), -R * cos(half), half_z)
    callout!(ax, project(cam, edge), project(cam, edge) + Mke.Point2f(120, 120), "$(apex.columns) Columns Across the Fan ($(round(2rad2deg(half); digits = 1))°)\n× $(apex.rows) Rows Along Z, at $(round(R; digits = 1)) mm"; fontsize = 16)
    low = (R * sin(-half / 2), -R * cos(-half / 2), -half_z)
    callout!(ax, project(cam, low), project(cam, low) + Mke.Point2f(-120, -160), "The Outer Rows See the Beam\nat the Cone Angle"; fontsize = 16, align = (:right, :center))
    xs = [project(cam, c)[1] for c in vcat(corners, [focal])]
    ys = [project(cam, c)[2] for c in vcat(corners, [focal])]
    Mke.limits!(ax, minimum(xs) - 520, maximum(xs) + 620, minimum(ys) - 240, maximum(ys) + 200)
    locator!(fig[0, :], :arc; note = "The Whole Arc:\n$(apex.columns) × $(apex.rows) Dexels, 160 mm Along Z")
    fig
end

# ╔═╡ 59e3353e-13f5-53f2-be17-2062c579f044
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))

    # (b) The signal per photon arriving, across the fan, in groups of 40 columns
    axb = Mke.Axis(
        fig[1, 1];
        title = "(a) Energy Collected per Photon, Across the Fan",
        xlabel = "Column (Across the Fan)",
        ylabel = "Energy Collected per Photon\non the Cell, ÷ E",
        limits = ((1, apex.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:apex.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / apex.columns
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
        apex.columns / 2,
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
        limits = ((1, apex.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:apex.rows, max(1, apex.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / apex.rows
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

# ╔═╡ ace5c787-13e0-57f3-93f1-345861f9c001
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
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / apex.source_to_detector_mm))
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

# ╔═╡ 96693605-bc02-50f2-a3ea-da32946c6c79
md"""
## 8. The Apex's X-Ray Beam

The photons come from the installed BasisSimulator (these are not what v0.16.0 replaces): its IPEM Report 78 tungsten spectra (10° anode),
through the scanner's 2.5 mm aluminium flat filter and the protocol's 4.5 mm of aluminium,
scaled to the detector at 1100 mm, exactly as `simulate!` uses them for the Apex.
BasisSimulator's spectra are tabulated at 80, 100, 120 and 140 kV; its Apex runs 120 kV, and
80/140 kV with rapid kV switching.

The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ 4575a19b-f864-53fc-a050-40b57b16fc8f
# BasisSimulator's Revolution Apex, as on its scanner page
apex_scanner = BS.EICTScanner(
    source_to_isocenter = apex.source_to_isocenter_mm,
    source_to_detector = apex.source_to_detector_mm,
    detector_rows = apex.rows,
    detector_cols = apex.columns,
    detector_row_size = apex.row_at_isocenter_mm,
    detector_col_size = apex.column_at_isocenter_mm,
    detector_col_offset = 0.25,
    detector_shape = :arc,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex,
    detector_depth = depth_mm,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 3500.0,
    detection_gain = 10.0,
)

# ╔═╡ 9f4fa6af-f684-525c-9389-5081654e6d71
"""
The photons per mAs per mm² reaching the detector, per bin, for an Apex protocol at `kVp`, with
the protocol's 4.5 mm of aluminium.
"""
function apex_spectrum(kVp)
    protocol = BS.CTProtocol(;
        kVp,
        mA = 200.0,
        views = 984,
        rotation_time = 0.5,
        collimation_mm = 5.0,
        additional_filters = [("Al", 4.5)],
    )
    return BS.resolve_source_spectrum_without_bowtie(
        BS.SimOptions(),
        protocol;
        scanner = apex_scanner,
    )
end

# ╔═╡ cf2e67e6-4692-5f99-bd79-a5fbc8fb9abf
@bind kvp PlutoUI.Slider([80, 100, 120, 140]; default = 120, show_value = true)

# ╔═╡ 1bae766b-aa3d-5389-937c-13cb98623d47
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

# ╔═╡ fb9bdb08-269b-439f-9880-891fda6f08f3
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))

    ax = canvas(fig[1, 1])
    amber = MATERIALS.beam
    # tube: envelope, cathode, anode at 10°
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
        (2.4, 0.16, Mke.RGBf(0.75, 0.77, 0.8), "2.5 mm Al", "The Scanner's\nFlat Filter"),
        (4.0, 0.22, Mke.RGBf(0.72, 0.74, 0.78), "4.5 mm Al", "BasisSimulator's\nProtocols"),
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
    Mke.text!(ax, xb + 0.25, hb + 0.12; text = "GE Large Bowtie", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.text!(ax, xb + 0.25, -hb - 0.12; text = "2.5 mm Al\nat the Centre", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    # patient and isocentre
    θ = range(0, 2π; length = 100)
    xi = 9.6
    Mke.poly!(ax, Mke.Point2f.(xi .+ 1.1 .* cos.(θ), 0.8 .* sin.(θ)); color = (MATERIALS.water, 0.6), strokecolor = (PALETTE.photon, 0.5))
    Mke.scatter!(ax, [Mke.Point2f(xi, 0)]; marker = :cross, markersize = 16, color = PALETTE.ink)
    Mke.text!(ax, xi, 0.95; text = "Patient (Isocentre, $(apex.source_to_isocenter_mm) mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    # detector with grid
    rect!(ax, x_end, -2.7, 0.15, 5.4, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge)
    for y in range(-2.6, 2.6; length = 14)
        rect!(ax, x_end - 0.35, y - 0.01, 0.35, 0.02, MATERIALS.tungsten)
    end
    Mke.text!(ax, x_end - 0.1, -2.85; text = "3D Collimator\n+ Detector\n($(round(Int, apex.source_to_detector_mm)) mm)", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -0.8, 14.6, -4.4, 3.4)
    locator!(fig[0, :], :tube; note = "The X-Ray Tube and\nWhat the Beam Passes")
    fig
end

# ╔═╡ 7f435d0f-3376-5aba-9422-da95aeee9fd4
let
    Mke.set_theme!(THEME)
    spectrum = apex_spectrum(kvp)
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
        label = "BasisSimulator v0.15.1's Table",
    )
    Mke.lines!(
        axb,
        s.energy_keV,
        s.signal;
        color = PALETTE.kept,
        linewidth = 3,
        label = "Geant4, Flat Field, 3.0 mm",
    )
    Mke.vlines!(
        axb,
        [terbium.edge_keV, lutetium.edge_keV];
        color = (PALETTE.xray, 0.6),
        linestyle = :dot,
    )
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ e8fa69d7-5554-59ca-9c0b-ab33f82e52dc
md"""
**The beam's quality, against measurement.** Kawahara et al. (Pol. J. Radiol. 2023) measured the
half-value layer of a Revolution CT (the Apex's predecessor, with the same Gemstone Clarity
detector) at the isocentre with a stationary tube and the large bowtie: **7.08, 9.20 and
10.0 mm of aluminium at 80, 120 and 140 kV**. At the isocentre the beam has passed the bowtie's
thinnest part, 2.5 mm of aluminium in BasisSimulator's GE large bowtie.
"""

# ╔═╡ a5ab816b-0816-53c4-a84d-40a2992a1de7
hvl_check = let
    bowtie = BS.resolve_bowtie_filter(:ge_revolution_large)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium = MC.DetectorMaterial(:Al)

    function hvl(kVp)
        e, w = apex_spectrum(kVp)
        μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
        return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
    end

    [(kVp = k, basis = hvl(k), measured = m) for (k, m) in ((80, 7.08), (120, 9.2), (140, 10.0))]
end

# ╔═╡ 064bfa0d-1a2d-40a9-9f1d-4bf769a5965a
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 380))
    for (k, (al, reading, title)) in enumerate(((0, "100 %", "No Aluminium in the Beam"), (9.2, "50 %", "9.2 mm of Aluminium at 120 kV: Half Is Left")))
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

# ╔═╡ 8eae3d73-e14e-5d93-9420-74e73deb379c
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Half-Value Layer at the Isocentre",
        xlabel = "Tube Voltage",
        ylabel = "Half-Value Layer (mm of Aluminium)",
        xticks = (1:3, ["$(c.kVp) kV" for c in hvl_check]),
        limits = ((0.4, 3.6), (0, 13)),
    )
    x = 1:3
    Mke.barplot!(
        ax,
        x .- 0.18,
        [c.basis for c in hvl_check];
        width = 0.34,
        color = PALETTE.reference,
        label = "Installed BasisSimulator's Beam",
    )
    Mke.barplot!(
        ax,
        x .+ 0.18,
        [c.measured for c in hvl_check];
        width = 0.34,
        color = PALETTE.kept,
        label = "Measured (Kawahara et al. 2023)",
    )
    for (k, c) in enumerate(hvl_check)
        for (dx, v) in ((-0.18, c.basis), (0.18, c.measured))
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

# ╔═╡ 5ab9e1a0-492b-54b0-bbdb-f272e3fa8bd1
let
    worst = maximum(abs(c.basis - c.measured) for c in hvl_check)
    verdict = worst <= 0.5 ?
        "The installed BasisSimulator's (v$(pkgversion(BS))) Apex beam matches the measured beam quality at every tube voltage." :
        "**⚠ Discrepancy:** the installed BasisSimulator's (v$(pkgversion(BS))) Apex beam differs from the measured one by up to $(round(worst; digits = 1)) mm of aluminium."
    Markdown.parse(
        """
        $(verdict) (The measurement is on a Revolution CT with its large bowtie; the Apex shares its tube filtration
        and bowtie family.) The rest of this notebook keeps BasisSimulator's beam, so the two stay comparable.
        """
    )
end

# ╔═╡ ba5b5a00-0e75-5e2c-9bae-727aec37b499
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. Every Apex tube voltage BasisSimulator can make:
"""

# ╔═╡ f775068c-2324-5cf4-8811-cd75f968a57d
scan_protocols = [(label = "$(k) kV", spectrum = apex_spectrum(k)) for k in (80, 100, 120, 140)]

# ╔═╡ 354a335b-59d0-565d-a29c-4a3242eac17c
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

# ╔═╡ 8c7a8cd0-ad15-4bc3-80c0-2e49d3e11b8e
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
        rect!(ax, 0, -0.6, 3.2, 0.6, MATERIALS.scintillator; strokecolor = PALETTE.slab_edge)
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

# ╔═╡ ffeaa78c-e964-5348-9a8d-7e17230dbfc1
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
        label = "Geant4, 3.0 mm, Flat Field",
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

# ╔═╡ 2077865c-68c8-5b80-8b56-275858a50a24
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | v0.15.1 table | Geant4, 3.0 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's garnet; BasisSimulator's fill factor (0.9 × 0.9) applies to both
        alike.
        """
    )
end

# ╔═╡ 8030fd72-30b1-4fc0-bd93-cfbaa57fe8ae
md"""
## 10. A Clinical Beam, for Real

Every efficiency so far came from photons of **one energy**, landing straight on **one dexel**.
A scan is neither: the Apex's tube sends a spectrum, hardened by the bowtie and by the patient,
from a focal spot over the whole arc, so the dexels at the edges of the fan and the outer rows
see it at an angle, through the grid. BasisSimulator never simulates that photon by photon; it
weights the single-energy table by the spectrum. This section checks that the shortcut is exact,
by doing it the long way:

- the Apex's **120 kV beam** (BasisSimulator's spectrum, 2.5 mm Al + the protocol's 4.5 mm Al,
  through the centre of GE's large bowtie), **bare and after 20 and 35 cm of water** (the patient's primary photons; the
  patient's scatter is notebook 10's);
- each one **drawn photon by photon from its spectrum**, onto one dexel's garnet (1.6 million
  photons) and from the focal spot over **the whole arc** with its grid (3.2 million);
- against the table **weighted by the same spectrum**: `Σ φ E η / Σ φ E` for the efficiency, and
  `(Σ φ m₁)² / (Σ φ · Σ φ m₂)` for DQE(0), with `m₁`, `m₂` the mean and mean square of the energy
  a photon leaves (section 4).
"""

# ╔═╡ 172113d0-e39e-4374-a519-71d82112d240
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))


    # The path: tube, bowtie, water, the arc with its grid
    ax = canvas(fig[1, 1]; title = "The Beam, Photon by Photon", titlesize = 20)
    amber = MATERIALS.beam
    focal = Mke.Point2f(0, 0)
    Mke.scatter!(ax, [focal]; marker = :rect, markersize = 22, color = PALETTE.ink)
    Mke.text!(ax, focal + Mke.Point2f(0, 0.45); text = "120 kV Tube", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    φ = range(-(apex.columns * apex.column_at_isocenter_mm / apex.source_to_isocenter_mm / 2), apex.columns * apex.column_at_isocenter_mm / apex.source_to_isocenter_mm / 2; length = 60)
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
    Mke.poly!(ax, vcat(inner, reverse(outer)); color = MATERIALS.scintillator, strokecolor = PALETTE.slab_edge)
    for t in range(φ[1], φ[end]; length = 31)
        Mke.lines!(ax, Mke.Point2f[((R - 0.5) * sin(t), -(R - 0.5) * cos(t)), (R * sin(t), -R * cos(t))]; color = MATERIALS.tungsten, linewidth = 1)
    end
    Mke.text!(ax, 0, -R - 0.7; text = "The Whole Arc and Its 3D Collimator ($(apex.columns) × $(apex.rows) Dexels)", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
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

# ╔═╡ f8780d12-0cc9-47fc-a81b-67fd0b043079
# The Apex's 120 kV beam on the central ray: BasisSimulator's spectrum through the bowtie's
# centre, then through 0, 20 and 35 cm of water (the primary photons), on its 0.5 keV grid
clinical_beams = let
    bowtie = BS.resolve_bowtie_filter(:ge_revolution_large)
    centre = argmin(abs.(bowtie.angles))
    bowtie_al_mm = 10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
    aluminium, water = MC.DetectorMaterial(:Al), MC.DetectorMaterial(:Water)
    e, w = apex_spectrum(120)
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

# ╔═╡ 337897b9-f2c9-438d-a33a-75ebe6fa164a
# Each beam photon by photon: onto one dexel's garnet (16 chunks of 100 000), and from the focal
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

# ╔═╡ eab5f822-0a9c-4c69-a7ac-74ed040acc8d
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

# ╔═╡ 04b4a062-43be-4fb9-b161-e6c11d51adc4
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

# ╔═╡ 9dfb0a5f-71d7-4aaa-ae86-e84b1105f97c
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

# ╔═╡ c582ddd0-5d0e-5091-bd69-4e30c9217079
md"""
## The Table to Use

The table to replace the v0.15.1 table with (BasisSimulator v0.16.0), and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **BasisSimulator v0.15.1's composition at 6.40 g/cm³**, a density such a garnet can have, instead of the
  impossible 7.0 (section 1). The Tb-rich garnet of GE's patent is saved beside it; between the two
  the efficiency moves by at most a few hundredths (section 4).
- **3.0 mm deep.** BasisSimulator v0.15.1 documents it, the v0.15.1 table was made at it, and it is
  within the 1.5–3 mm reflector cells of GE's patent (sections 2 and 5).
- **BasisSimulator's fill factor, 0.9 × 0.9, unchanged.** The assumed reflector reproduces it.
  With the 0.15 mm grid walls of GE's patent example it would be about 0.74 of the face
  (section 6), but that is a patent example, not the product, so it is not adopted.

The efficiency for depths from 1.5 to 4.0 mm is saved beside it.
"""

# ╔═╡ 93ff99c5-0fce-54a2-bb23-da7eb0316397
md"""
## Saving the Table

One file in `lib/BasisMonteCarlo/tables/`, in the layout of BasisSimulator's
`GEMSTONE_MC_EFFICIENCY_LUT` (`energies`, `efficiency`): the flat-field efficiency of 3.0 mm of
Gemstone at a density a garnet can have (BasisSimulator's composition at 6.40 g/cm³), beside the
one-dexel efficiency, the interaction probability, the Swank factor and DQE(0), the Tb-rich
garnet of GE's patent, and the flat-field efficiency for depths from 1.5 to 4.0 mm, since no
source gives the Apex's depth. The reproduction of the v0.15.1 table is kept with it, as the
record of how that table was made.
"""

# ╔═╡ 7b873d56-4442-5532-9a8c-1d466d90360f
table_depths = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# ╔═╡ 82ad04aa-e3c4-51e2-a29b-5e9290db49e4
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

# ╔═╡ 3e1c3140-a3e4-5297-9faa-a403a0c3482d
# The machine path stays out of the rendered page
table_dir = joinpath(@__DIR__, "..", "..", "tables");

# ╔═╡ 2ba11eeb-f02c-5028-8c7b-3ec4edc3219d
saved = let
    path = joinpath(table_dir, "ge_revolution_apex_gemstone_efficiency_geant4.jls")
    contents = Dict{String, Any}(
        "energies" => collect(energies),
        "efficiency" => efficiency.flat_field,
        "efficiency_one_dexel" => efficiency.one_dexel,
        "interaction_probability" => efficiency.interaction,
        "swank_factor" => efficiency.swank,
        "dqe0" => efficiency.dqe,
        "depths_mm" => table_depths,
        "efficiency_by_depth" => MC.energy_absorption_efficiency.(by_depth),
        "efficiency_tb_rich" => composition_runs["Tb-rich, 6.24"],
        "efficiency_at_7g_cm3" => composition_runs["even split, 7.0"],
        "reproduced_table" => reproduced,
        "reproduced_depth_mm" => fit.depth_mm,
        "reproduced_scale" => fit.scale,
        "compared_with" => "BasisSimulator v0.15.1, GEMSTONE_MC_EFFICIENCY_LUT (BasisMonteCarlo.BASISSIMULATOR_V0_15_1_EFFICIENCY.gemstone)",
        "decision" => "use `efficiency`: flat field, BasisSimulator's Gemstone composition at 6.40 g/cm3 (7.0 is impossible for any (Tb,Lu)3Al5O12), 3.0 mm deep; BasisSimulator's fill factor 0.9 x 0.9 unchanged",
        "definition" => "mean energy collected per photon landing on a dexel's garnet with every dexel illuminated (flat field), divided by the photon energy; BasisSimulator's fill factor applies separately",
        "detector" => Dict(
            "scanner" => "GE Revolution Apex (Gemstone Clarity)",
            "material" => "Gemstone, (Tb,Lu)3Al5O12:Ce, BasisSimulator's mass fractions at 6.40 g/cm3",
            "depth_mm" => depth_mm,
            "pitch_mm" => pitch_mm,
            "reflector" => "0.1 mm TiO2-loaded epoxy (assumed)",
            "grid" => "2-D tungsten, 0.15 mm walls, 8 mm tall (GE patent example); not in the flat field",
        ),
        "photons_per_energy" => 100_000,
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 06_ge_revolution_apex.jl",
    )
    Serialization.serialize(path, contents)
    path
end;

# ╔═╡ fe7f4f17-2218-5691-8c95-15a0e171c1ce
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

# ╔═╡ 1eb7490b-a247-5aff-829e-55401af71db1
md"""
## Where This Leaves the Apex
"""

# ╔═╡ f0fd4be7-bd7e-5437-b754-3370abcdbd27
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    worst_hvl = maximum(abs(c.basis - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        - **The density is impossible.** BasisSimulator v0.15.1's Gemstone is 7.0 g/cm³; any (Tb,Lu)₃Al₅O₁₂ lies between 6.07 and
          6.72. This notebook uses BasisSimulator v0.15.1's composition at 6.40 g/cm³ and brackets the unpublished Tb/Lu split
          with GE's Tb-rich patent garnet; Siemens' own LuTAG curve sides with the lighter garnets.
        - **BasisSimulator v0.15.1's table is its documented detector**, at the impossible density: a pencil beam into one dexel,
          $(fit.depth_mm) mm deep, tallied in that dexel, reproduces it to about 0.01 (one point on the Tb K-edge apart).
        - **What a scan sees** is the flat-field efficiency: each dexel gets back from its neighbours the Tb and Lu
          x rays it loses to them. At 3.0 mm and 6.40 g/cm³ it is $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, against the
          table's $(round(m120.table; digits = 3)) ($(round(100(m120.geant4 / m120.table - 1); digits = 1)) %), with DQE(0) $(round(m120.dqe; digits = 3)).
        - **The grid** (2-D, tungsten, GE's patent dimensions) shades a thin strip of garnet and stops most scatter;
          **the whole detector** collects what one dexel's cell does.
        - **The beam:** the installed BasisSimulator's Apex beam $(worst_hvl <= 0.5 ? "agrees with" : "differs from") the measured Revolution CT half-value layers $(worst_hvl <= 0.5 ? "to within" : "by up to") $(round(worst_hvl; digits = 1)) mm of aluminium.
        - **Saved:** `$(basename(saved))`, with both possible garnets and depths 1.5–4.0 mm: the table for BasisSimulator v0.16.0.
        """
    )
end

# ╔═╡ Cell order:
# ╟─982eeab8-0545-51f1-b14c-f4984bc5ce21
# ╟─f47a2a75-13c1-4576-b10c-da26be745122
# ╟─7d76168f-6aaf-5cac-ad93-06b3d605dce4
# ╟─5b30322f-8454-4b47-8acf-88c1e1bc53a5
# ╟─1637020f-80bb-43db-9357-909efdb66156
# ╟─2e433940-1ca3-4ed7-a00d-2410fb565f03
# ╟─81e47492-f46a-4e97-a3a2-d04bd5d8e70d
# ╟─0d89ac30-a1cd-555c-a60a-934112484f93
# ╟─b5d65082-f8d7-5b19-a30a-117dcd788835
# ╠═c7b8ce23-df15-5445-8f9c-1d533b22b055
# ╠═b9afe761-a41a-56bd-89b3-aa4d1f850542
# ╠═1114b922-2909-5382-ad68-7870b9f17a49
# ╠═3c7da9cf-d44f-5117-a6cd-ab70462594ba
# ╠═d7980232-34d1-599f-89a1-18c144d7468b
# ╠═2e1263bc-6a15-5dad-9da8-0c5ea9d9a1ec
# ╠═6f0a1390-4e16-5007-aff0-205efb24710a
# ╠═6650624a-9201-5026-9af0-2852cb32ff32
# ╠═065c2cd0-3724-5ae0-a368-5c8eb5540a1a
# ╠═0f1f7711-29fa-5149-b284-b47dcdc295c0
# ╠═e3e9c212-4583-5b56-8c63-55a23aceccc6
# ╠═3b5f881c-54d6-5378-ad04-eefcacc3c8ea
# ╠═425bf6f0-bb29-56ec-9883-45afcf680eda
# ╠═4551bbf4-0c6f-558d-8dd3-19061afbeee1
# ╠═d852372b-0f59-5689-a6f2-a96f8756249b
# ╟─c8a64c0f-8f57-41f2-9e46-d50e428c1036
# ╠═0ed04217-4dc3-4912-ad84-aff29fa419f3
# ╠═bb748ca7-2f2c-4159-b359-76d52f2f3609
# ╠═a9830f78-b3ee-499b-a95a-4bbb68c74dd4
# ╠═511ddd8d-3459-47c6-9b6d-560f6e960f9f
# ╠═c705d51c-e973-48fa-bbd0-d84ad220fc4d
# ╠═a3361458-4c30-464b-8636-bad7c9d5da58
# ╠═61264840-6153-4a9f-b0ae-ce8277b5010b
# ╠═7bde14bc-c0f6-4574-8629-950a29447b64
# ╠═6359c896-d676-492c-a544-425666580d1d
# ╠═686a8f39-4bb3-4656-8f53-91fb0a7d2254
# ╠═02dcd805-ca95-47a9-9fb2-b0b71942aae5
# ╠═a8569152-9721-40e8-9f46-c16279e3fa42
# ╠═3136d195-a95b-4ddb-a868-542dfb8ac793
# ╠═af481ff1-3a3f-4045-8d9f-7ee63312785b
# ╠═32fe7f10-9fa0-4cd7-9cb3-51751c3e6f5a
# ╠═a83bfa57-44d8-4b63-aca3-37e5ea4fa1c3
# ╠═99873a48-bd7a-4996-8e3f-ea3fea7859a6
# ╠═c6d7d774-ef13-4f99-83cd-6465fc1f6888
# ╠═08ed4a29-4109-490b-b791-5987c138c173
# ╠═b04c0a12-4601-4267-9293-98d2023dafd8
# ╠═0085388d-2916-4ebe-9217-8dd1f1d568f2
# ╠═a4a8981a-9496-40dd-bfb1-3e0648de4ad0
# ╠═ee4d93d9-757c-460e-8173-7f28a04f7372
# ╠═71976bd4-43a7-4cca-8bd7-9e41c3071263
# ╠═6d118872-a994-4622-87a2-2a4d2c452d93
# ╠═8cec9efb-82c7-4a73-a60c-6a2f4e2d48b9
# ╠═a2d50813-d2e4-4a26-8004-7cfbbd80c8bc
# ╠═2fc965a3-f475-4d25-b639-e556396e249a
# ╟─9ed4e154-0c35-5252-b2dc-388f22a1c47f
# ╠═0565c39d-9720-5863-a0b6-c3e104a475d1
# ╠═b5c946ee-f16f-5cca-82cf-b5040fe481b8
# ╠═8fba75c0-6a9f-5bbe-9a10-9d52e2296292
# ╠═b349f0be-6494-5c81-80ea-82a481df3b90
# ╟─3e134eba-e90e-5fef-845b-b81b61397dca
# ╠═f88f3daf-cbd3-55ec-9361-63ae45a21b33
# ╠═d6256293-43fa-5626-bba6-6b1c01af6dda
# ╠═2bcabe1a-a3fd-5fa7-8e2a-2a88d9126859
# ╠═7b39f748-54d2-5a67-977a-b81bf47b6b27
# ╟─74cea183-7970-4071-ae47-c9a7a199c514
# ╟─79bf52e2-7776-5cc1-8d0b-4e264aa8530e
# ╟─d0f1d5ee-5eb0-5fa2-bcd6-74a957ed0358
# ╟─ea752deb-d5f8-530f-a2bc-d8f70ac7f9d6
# ╠═ad4b77f0-e633-5a1f-8474-144d368eb858
# ╠═7fea9a31-343e-5084-b697-c614d31d86cc
# ╠═c6bf2e2b-147e-5c10-a14e-53a1f4421e56
# ╟─24709fa8-0823-4f19-bcd6-ca2794d35249
# ╟─7e771ff9-6f1a-5c5b-8469-30e1a7ef0a6e
# ╟─ba7bd8e3-3fe8-51fb-a149-651b7415cecf
# ╟─76d9fe06-65d2-50ec-a57f-9d13c0424438
# ╠═59b35b57-f457-5c5a-89fd-f215ac91ec5f
# ╠═87c01464-7920-5151-aaf1-668655c36c98
# ╠═66875788-7213-5a45-93df-f02a465feab5
# ╠═85332bcb-a653-5577-986e-55557180a527
# ╠═43a2476d-f0b3-5c4e-ad60-a79093af88bc
# ╠═b937f150-98dd-52ad-a1a9-39606e1cc17d
# ╟─69f88251-23f2-5b2f-8e30-f8037e72aabd
# ╠═3cd39c08-6d7d-5852-911d-70b71149b9ca
# ╠═13a363b3-f112-59a0-9192-be6a9bef1026
# ╠═e340add3-b5f6-5242-93ea-81ef5ba335a2
# ╠═0df4a950-2c66-5f98-827e-d647b3db7d5c
# ╟─3a51adf8-1bd6-5027-a06e-b288b5ea7b6e
# ╟─3d4f918b-7df5-5eed-becb-a6dd61ba726c
# ╠═bfafebed-31d8-53f5-a471-25993d2ed385
# ╠═32ca1abd-3897-5936-a735-a52773f47bf5
# ╟─76876cdf-0539-5b54-a239-98f3e02df836
# ╟─da6b53d3-7e98-4a7d-a74c-20aa703c9633
# ╟─e76baa84-2c08-4f91-bdd6-f2c94d64fa06
# ╟─fdb32583-deac-5102-80c8-d5c90e587dc2
# ╠═a2a74ae1-42e5-5360-b821-fa1c5835af23
# ╠═d465f476-4b7a-5a26-8485-e167450cf6e3
# ╟─2c48c3e8-a7a4-496c-85c1-674afac1c9d1
# ╟─be0eef36-6795-5094-814c-7838d52f5e92
# ╟─fb4ffcd0-bac4-5ce3-858c-c45c2085c83b
# ╠═71ce3fe1-447b-54b5-8ded-a897f1c8636e
# ╠═c1642d2c-c743-50d8-8fa0-b307a93526da
# ╟─d6c44cff-e7b9-4790-b287-f784595829ce
# ╟─590bc8d2-76b3-5e54-9bc8-d8aa7104552e
# ╟─90bbeea9-e955-5350-9c91-055bb731efb2
# ╟─76eecb6b-bcfa-5b64-820a-0ab315fad43f
# ╠═27dc6d6f-6037-5f6a-b642-1c5af6a9bc4d
# ╠═05c86c98-403d-55b7-ae80-f23a4cbbfcd5
# ╠═651673f4-ac81-5aa2-9b6c-4d3a1f232562
# ╟─7dd51815-ccc2-492a-81f2-80a0e1229257
# ╟─fdb38e4c-3f74-5de4-be98-cdda136f1a56
# ╟─f192b404-d795-5a67-8bf5-87aa47aeda2e
# ╟─243b6cd3-d4a7-5e9a-a7d0-39a424919164
# ╠═ee6081a3-7bb9-5320-a338-909cf1251027
# ╟─739e2f06-3ac1-41ba-b800-c95f4274a083
# ╟─7f4daa38-7a47-5d7f-81c6-2aa821f3715e
# ╟─208a44d2-987f-5887-b584-1f976fcc90a5
# ╟─56b952ce-233b-5076-a0ad-fa6756455914
# ╠═1d5ea56f-89a5-535f-a2ab-dce36fcb20a7
# ╟─322cca7c-08ea-4928-90bf-2ef4628f918b
# ╟─8ec58a1a-ddc9-51e5-a0e4-6ced725ac8fe
# ╟─4ace8ef9-4637-5260-9f14-5f7685e76537
# ╠═797de5bb-99b6-5094-a316-d8ca2bfb284f
# ╟─e165adb1-2895-48f3-b063-326d1ef88539
# ╠═6830b6a0-8ae0-5e43-80e9-da1cc552508c
# ╟─ac337b7c-b33c-4a34-9917-e0d030638c52
# ╟─f27dd4ed-4167-5bd4-ac84-a8f98e88754b
# ╟─2ad68ded-ca19-504d-a805-f0caf48d398b
# ╟─23cc9c86-3ad7-5182-a4e2-9857fcf1c6f2
# ╠═2f3c0378-a2dd-5a8a-aac2-8ecba127fdc3
# ╠═1ae1a7b5-befc-57dd-8506-a3904c6e56e2
# ╠═876a93ce-7ca7-5f15-a44d-80a8c18bd676
# ╠═24568e75-698d-5a5e-9401-f553959c2742
# ╠═e41cfdd8-b032-5c29-abef-a5c856dec82f
# ╟─ba31b322-bf2a-57ca-8068-9858e8ac5d00
# ╠═05b60d54-a668-5497-9fc5-dcedf626515e
# ╠═d067a0b4-d46d-5a21-a351-d3c7e347f9b7
# ╟─54179647-5aa9-520a-a0ce-933b5e136cbf
# ╟─25b60608-28d7-5442-89df-fb1e7ac471c8
# ╟─55dad12b-a818-55c3-b5ba-aea9be0cd98c
# ╠═d89950fe-ea70-592b-998b-3e954534f5c9
# ╟─6b89ebe5-d2ca-5d77-93e1-ec26b393c271
# ╟─c599896a-d8dc-5d31-8aa6-d04f7eb6bf7e
# ╠═367261c6-ffb8-5661-970e-6494b0a10197
# ╠═8c9ab3eb-23b7-57a5-9c13-0aa983dae5e0
# ╠═387074f5-5299-5768-b50a-e1a315621ab8
# ╠═173503d6-98e5-51d2-b783-7e5885c408eb
# ╟─8c9677c3-aac1-593c-a53c-ccc5218c3b3e
# ╟─9af6dffc-d99a-54bf-97cf-dd20156bbe48
# ╟─89f83a80-82e6-5d12-8616-36560dba03a1
# ╠═8074b52c-eaab-5eef-ab41-b3482d729940
# ╠═bd408664-f831-56e0-971a-4d7c4f14fa33
# ╠═92105c08-3d1a-533c-8939-3c30460be4b4
# ╟─7389b76e-a842-4021-bbca-ff56f1e40825
# ╟─59e3353e-13f5-53f2-be17-2062c579f044
# ╟─ace5c787-13e0-57f3-93f1-345861f9c001
# ╟─96693605-bc02-50f2-a3ea-da32946c6c79
# ╠═4575a19b-f864-53fc-a050-40b57b16fc8f
# ╠═9f4fa6af-f684-525c-9389-5081654e6d71
# ╠═cf2e67e6-4692-5f99-bd79-a5fbc8fb9abf
# ╠═1bae766b-aa3d-5389-937c-13cb98623d47
# ╟─fb9bdb08-269b-439f-9880-891fda6f08f3
# ╟─7f435d0f-3376-5aba-9422-da95aeee9fd4
# ╟─e8fa69d7-5554-59ca-9c0b-ab33f82e52dc
# ╠═a5ab816b-0816-53c4-a84d-40a2992a1de7
# ╟─064bfa0d-1a2d-40a9-9f1d-4bf769a5965a
# ╟─8eae3d73-e14e-5d93-9420-74e73deb379c
# ╟─5ab9e1a0-492b-54b0-bbdb-f272e3fa8bd1
# ╟─ba5b5a00-0e75-5e2c-9bae-727aec37b499
# ╠═f775068c-2324-5cf4-8811-cd75f968a57d
# ╠═354a335b-59d0-565d-a29c-4a3242eac17c
# ╟─8c7a8cd0-ad15-4bc3-80c0-2e49d3e11b8e
# ╟─ffeaa78c-e964-5348-9a8d-7e17230dbfc1
# ╟─2077865c-68c8-5b80-8b56-275858a50a24
# ╟─8030fd72-30b1-4fc0-bd93-cfbaa57fe8ae
# ╟─172113d0-e39e-4374-a519-71d82112d240
# ╠═f8780d12-0cc9-47fc-a81b-67fd0b043079
# ╠═337897b9-f2c9-438d-a33a-75ebe6fa164a
# ╠═eab5f822-0a9c-4c69-a7ac-74ed040acc8d
# ╟─04b4a062-43be-4fb9-b161-e6c11d51adc4
# ╟─9dfb0a5f-71d7-4aaa-ae86-e84b1105f97c
# ╟─c582ddd0-5d0e-5091-bd69-4e30c9217079
# ╟─93ff99c5-0fce-54a2-bb23-da7eb0316397
# ╠═7b873d56-4442-5532-9a8c-1d466d90360f
# ╠═82ad04aa-e3c4-51e2-a29b-5e9290db49e4
# ╠═3e1c3140-a3e4-5297-9faa-a403a0c3482d
# ╠═2ba11eeb-f02c-5028-8c7b-3ec4edc3219d
# ╠═fe7f4f17-2218-5691-8c95-15a0e171c1ce
# ╟─1eb7490b-a247-5aff-829e-55401af71db1
# ╟─f0fd4be7-bd7e-5437-b754-3370abcdbd27
