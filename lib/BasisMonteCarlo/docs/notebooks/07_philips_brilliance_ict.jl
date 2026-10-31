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


# ╔═╡ 1e8019db-b5b1-5ad1-a89a-ffa9fb0e8341
md"""
# The Philips Brilliance iCT Detector in Geant4

The Philips Brilliance iCT is an energy-integrating CT scanner, made in two widths: the
**iCT 256** (128 detector rows, 8 cm) and the **iCT SP**, sold as the iCT 128 (64 rows, 4 cm). Both
are at UCI Health, and both use the same detector, Philips' **NanoPanel**: a ceramic scintillator
cut into **dexels** (detector elements), with a reflector between them and a photodiode under every
dexel, curved in two directions (a spherical detector) behind a two-dimensional anti-scatter grid.

BasisSimulator, up to and including v0.15.1, has no Philips scanner: no geometry, no beam and
no detector table. This notebook builds the detector in Geant4, one piece at a time, and

1. **gathers what is documented** (the FDA clearances, Philips' Dunlee detector datasheet, the
   literature) and states every assumption where nothing is;
2. **fits the x-ray beam** to measured half-value layers, since Philips publishes no filtration;
3. **builds the detector's efficiency table**, with its noise (the Swank factor), and saves it for
   BasisSimulator, once for each of the two scanners.

!!! note "Which BasisSimulator this notebook compares against"
    BasisSimulator up to and including **v0.15.1** has no Philips detector. Where a table is
    needed for comparison, the nearest it had is used: its SOMATOM Definition Flash table
    (Gd₂O₂S, MCNP), which BasisMonteCarlo keeps verbatim as
    `BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash`, so the comparison stays the same after
    BasisSimulator moves on. The x-ray spectra and the bowtie (sections 8–10) come from the
    **installed** BasisSimulator, whatever its version; nothing here pins BasisSimulator, or
    BasisMonteCarlo, to a version.
"""

# ╔═╡ 83592696-cd7f-4682-99b3-1a0e047aaff4
let
    Mke.set_theme!(THEME)
    night, ink, sub = MATERIALS.night, Mke.RGBf(0.93, 0.94, 0.96), Mke.RGBf(0.6, 0.63, 0.68)
    fig = Mke.Figure(size = (1500, 600), backgroundcolor = night, figure_padding = (10, 10, 10, 10))

    # (1) The gantry, to scale: one tube and its detector arc
    ax1 = canvas(fig[1, 1]; background = night)
    R_iso, R_det = ict.source_to_isocenter_mm, ict.source_to_detector_mm
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
        a, b = project(cam, (c..., d + 12.5)), project(cam, (c..., d + 0.15))
        (dot3((c..., d), cam.toward) + 1, ax -> arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 12))
    end
    draw_blocks!(ax2, cam, blocks; extras = rays, edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    # (3) One dexel's stack, taken apart
    ax3 = canvas(fig[1, 3]; background = night)
    draw_blocks!(ax3, camera(-56, 30), exploded_detector(1, 1; gap = 0.75).blocks; edge = Mke.RGBf(0.05, 0.06, 0.08), edge_alpha = 0.5)

    for (k, (title, subtitle)) in enumerate(
            (
                ("One Tube, One Detector", "The Brilliance iCT, to Scale: a 50 cm Field of View"),
                ("Dexels Under a 2-D Grid", "Gd₂O₂S Ceramic, Reflector, Tungsten Walls (Assumed)"),
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

# ╔═╡ c34c0bd0-aa5c-5e4a-8bda-740e8ca07147
md"""
## How a Dexel Measures

A photon-counting dexel counts photons one at a time; a NanoPanel dexel does not count at all.
Every x ray it absorbs leaves energy in the ceramic, which glows; the reflector keeps the light
inside the dexel, and the photodiode under it adds up the light of every photon over the reading.
The signal is therefore proportional to the **energy** left in the dexel, and the efficiency that
matters is the energy-absorption efficiency: the mean energy a dexel collects per photon, divided
by the photon's energy.
"""

# ╔═╡ f84d9387-4b88-489e-9ffc-9614cc74dced
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
        "One Dexel, Outlined in Amber: One Block of Ceramic and Its Photodiode, Under One Grid Cell (1.408 × 1.14 mm)";
        font = :bold,
        fontsize = 17,
        color = Mke.RGBf(0.8, 0.5, 0.05),
        halign = :left,
        tellwidth = false,
    )

    rows = (
        (:grid, "Anti-Scatter Grid (2-D on the iCT 256)", "Documented 2-D, Pre-Focused; Tungsten, 0.1 mm\nWalls, 10 mm Tall Assumed (Drawn Short Here)", true),
        (:ceramic, "Ceramic Scintillator, Sawn into Dexels", "Documented \"Ceramic\"; Gd₂O₂S, 7.31 g/cm³, 1.4 mm Deep\nand a 0.1 mm Reflector Are Assumed", true),
        (:photodiode, "Photodiode, One per Dexel", "Documented (FDA K162838): Light → Charge", false),
        (:cmos, "Readout Electronics", "Philips' \"Tile\" Detector Integrates Them; Their\nLayout Is Not Published, So Drawn Generic", false),
        (:substrate, "Tile Substrate", "Holds the Chips (Generic)", false),
        (:board, "Circuit Board", "Carries Every Dexel's Number Out\nto the Gantry (Generic)", false),
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

# ╔═╡ a967d445-3fe2-4149-859b-4e544f9376b7
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
        (anchor = (0.12, d + 1.9), title = "1  An X Ray Arrives", detail = "Down Its Channel in the Grid", color = Mke.RGBf(0.85, 0.5, 0.05), tag = nothing),
        (anchor = hit, title = "2  The Ceramic Stops It", detail = "Its Energy Is Left in the Dexel", color = Mke.RGBf(0.7, 0.55, 0.05), tag = nothing),
        (anchor = (half - 0.02, hit[2] + 0.28), title = "3  The Ceramic Glows", detail = "Light ∝ Energy Left; the White Reflector Keeps It In", color = Mke.RGBf(0.2, 0.6, 0.25), tag = nothing),
        (anchor = (0.4, y_pd + 0.07), title = "4  The Photodiode Collects the Light", detail = "Light → Charge, Summed over the Reading", color = MATERIALS.photodiode, tag = nothing),
        (anchor = (0.22, y_cmos + 0.2), title = "5  The ADC Digitizes the Charge", detail = "In the Tile's Electronics (Layout Not Published)", color = PALETTE.ink, tag = nothing),
    ]
    xr = 1.5p + 0.75
    labels_right!(ax, entries, xr; spacing = 0.62)
    Mke.text!(ax, 1.5p + 0.05, d + 1.2; text = "Grid Walls\n(Cut Short)", fontsize = 14, color = PALETTE.muted, align = (:left, :top))

    # what Geant4 simulates, and what is assumed
    xl = -1.5p - 0.15
    side_bracket!(ax, xl, 0.0, d + 1.2, "Geant4\nSimulates"; color = TAGS.geant4)
    side_bracket!(ax, xl, y_board, -0.04, "Not Simulated:\nSignal ∝ Energy\nLeft in the Ceramic"; color = TAGS.not_simulated)
    Mke.limits!(ax, -1.5p - 1.75, xr + 3.3, y_board - 0.75, d + 2.75)
    fig
end

# ╔═╡ b79e0b22-d18c-415b-a7f3-bf3e3f04d397
md"""
### Which Beam, and Why

How the photons arrive decides what a simulation measures, so each section says which beam it
uses. Five are used here, from the simplest to the most realistic:

| beam | what it measures | where |
|---|---|---|
| **pencil beam** into the centre of one dexel | how BasisSimulator v0.15.1's nearest table, the Definition Flash's, was made (notebook 05 reproduces it) | section 5 |
| **uniform on one dexel's ceramic** | the efficiency per photon landing on the ceramic; reading every dexel around it gives the flat field (below) | sections 3–5 |
| **uniform on one cell** (ceramic and its reflector), under the grid | what the grid costs, straight and at an angle | section 6 |
| **flat field**: every dexel lit | what each dexel collects in a scan: what it loses to its neighbours it gets back from them. By symmetry this equals all the energy a uniform beam on one dexel leaves anywhere, which is how it is simulated | sections 4, 5 |
| **from the focal spot**, over the whole arc, with **clinical spectra** | everything at once: real angles, every dexel, the grid, the iCT's fitted 120 kV beam (also after 20 and 35 cm of water) | sections 7, 10 |

The efficiency table is built from single energies, because BasisSimulator weights it by whatever
spectrum a scan uses. Section 10 checks that this shortcut is exact: the clinical beams, simulated
photon by photon, give the same signal and noise as the table weighted by their spectra.
"""

# ╔═╡ 4ad6e0a4-6bbe-48da-a921-12055b764503
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1700, 560))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    cases = (
        ("Pencil Beam", "Into the Centre of One Dexel: How\nv0.15.1's Flash Table Was Made", :pencil),
        ("Uniform on One Dexel", "Over Its Ceramic; Every Dexel\nRead Out (Sections 3–5)", :ceramic),
        ("Uniform on One Cell", "Ceramic and Its Reflector,\nUnder the Grid (Section 6)", :cell),
        ("Flat Field", "Every Dexel Lit, as in a Scan:\nWhat Each Dexel Collects", :flat),
        ("From the Focal Spot", "The Whole Arc, Real Angles;\nClinical Spectra (Sections 7, 10)", :focal),
    )
    for (k, (title, detail, kind)) in enumerate(cases)
        Mke.Label(fig[1, k], title; fontsize = 20, font = :bold, halign = :left, tellwidth = false)
        ax = canvas(fig[2, k])
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
        Mke.Label(fig[3, k], detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    fig
end

# ╔═╡ e1a742dd-c09a-5ada-87f2-df7c195ed69b
md"""
## The Brilliance iCT, as Documented

Every hardware number this notebook uses, from the FDA 510(k) summaries, Philips' Dunlee OEM
detector datasheet (which matches the iCT's dexel count and distances exactly), Philips' patents and
the literature. Where no source gives a number the row says so, and the notebook states its
assumption. BasisSimulator, up to and including v0.15.1, has no Philips values to compare against.

| quantity | documented | status | source |
|---|---|---|---|
| scintillator | "NanoPanel: ceramic scintillator + photodiode"; composition not stated (Philips' GOS patents describe Gd₂O₂S:Pr,Ce) | assumed here: Gd₂O₂S at 7.31 g/cm³ | K162838; US 8,492,724 |
| crystal depth | not published; Philips' GOS-array patent: "about 2 mm or less" | assumed here: 1.4 mm; 0.8–2.0 mm saved | US 7,728,302 |
| dexels | 86 016 (iCT 256) and 43 008 (iCT SP): 672 columns × 128 or 64 rows | ✓ | K162838 |
| dexel pitch at the detector | 1.408 mm across the fan × 1.14 mm along z (0.772 × 0.625 mm at the isocentre, calc) | ✓ (OEM datasheet) | Dunlee CD500 |
| z coverage | 8 cm (iCT 256), 4 cm (iCT SP) | ✓ | K162838 |
| source–isocentre, source–detector | 570, 1040 mm | ✓ | K162838 |
| detector shape | spherical | built here as a cylindrical arc (geometry only) | Dunlee CD500 |
| reflector between dexels | not published | assumed: 0.1 mm of TiO₂-loaded epoxy | – |
| anti-scatter grid | two-dimensional, pre-focused (iCT 256); the iCT SP's may be one-dimensional; material and dimensions not published | built here: tungsten, 0.1 mm walls, 10 mm tall (assumed) | Dunlee CD500; EP 2,433,156 |
| tube | 80, 100, 120, 140 kV; focal spots 0.6 × 0.7 and 1.1 × 1.2 mm, flying in x and z | ✓ | K162838 |
| anode angle, filtration | not published (a titanium filter of unstated thickness) | anode 10° assumed; filtration fitted in section 8 | – |
| beam quality | half-value layer 6.4, 7.7, 8.8 mm Al at 80, 100, 120 kV | the target of the beam fit | Trattner et al. 2017 |

Sources: FDA [K162838](https://www.accessdata.fda.gov/cdrh_docs/pdf16/K162838.pdf),
[K131773](https://www.accessdata.fda.gov/cdrh_docs/pdf13/K131773.pdf);
Dunlee (Philips), [CD500 128-row detector datasheet](https://www.philips.com/c-dam/dunlee/downloads/5218142_Dunlee_Datasheet_Detector_CD500_128Rows.pdf),
[CD300 64-row detector datasheet](https://www.philips.com/c-dam/dunlee/downloads/5218150_Dunlee_Datasheet_Detector_CD300_64rows.pdf);
Philips patents [US 7,728,302](https://patents.google.com/patent/US7728302B2/en),
[US 8,492,724](https://patents.google.com/patent/US8492724),
[EP 2,433,156](https://patents.google.com/patent/EP2433156);
Trattner et al., [Med. Phys. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5734630/);
Siemens, [Detectors & Crystals](https://marketing.webassets.siemens-healthineers.com/8c32068491abf95e/b35c7f193990/siemens-healthineers_DI-CT_CT-Detectors-and-Crystals-Center.pdf);
Boone et al., Med. Phys. 1999, [doi:10.1118/1.598612](https://doi.org/10.1118/1.598612).
"""

# ╔═╡ ec086b05-5dc0-5983-bb6f-4f7ebd3781af
md"""
## Notebook Setup
"""

# ╔═╡ 03b0eb03-c3fe-59cb-acd8-5a14b0d80207
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ e457121e-c315-501e-aa97-d71761eb2100
using Markdown: @md_str, Markdown

# ╔═╡ cf074860-7eb8-551a-9545-d52b492f3d22
import PlutoUI

# ╔═╡ a281979d-cff0-5fa9-87ca-baf1abec600c
import BasisMonteCarlo as MC

# ╔═╡ 6ded5641-4694-5be0-8dd3-61dbc4c47466
import CairoMakie as Mke

# ╔═╡ 6399c96e-ad68-5802-b2fa-ed8629257ff3
import BasisSimulator as BS

# ╔═╡ 5fab2bcc-a52b-5a40-bfd2-4c3ab57a916f
import Serialization

# ╔═╡ 98b6fe03-3bc8-58a3-a3d2-51eb0a66db65
import DelimitedFiles

# ╔═╡ 673ac0c5-46e5-5961-9fcd-014c44cf2edd
using Statistics: mean

# ╔═╡ 092f835e-0e32-57d5-ba25-6a5783c55235
PlutoUI.TableOfContents()

# ╔═╡ 293d3699-48da-599a-bef5-fdd75fda7e1b
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 88fd7a1a-b153-5163-82fe-94476836d0c4
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
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator v0.15.1's nearest table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ 43f718d9-bdcd-55dc-8181-8ec3ca691693
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

# ╔═╡ 68f8c99b-e252-5cf4-a55b-ad5517bc4017
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

# ╔═╡ fe7cf193-49ce-5b3e-8d57-69460f810f9a
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

# ╔═╡ 16035000-554f-4376-8013-68b105841b6f
md"""
### Drawing the Detector

The illustrations are drawn from the same numbers the simulation uses (the pitch, the reflector,
the crystal's depth, the grid), with a small kit: blocks in 3-D, seen from any direction, shaded
by one light and drawn back to front. The colours are the materials': pale blue Gd₂O₂S ceramic, white
TiO₂ reflector, grey tungsten, dark-blue silicon photodiodes, amber x rays.
"""

# ╔═╡ 7500fcfc-2a78-44ab-8a2e-7e29a5f1efdf
MATERIALS = (
    ufc = Mke.RGBf(0.74, 0.84, 0.93),  # the Gd₂O₂S ceramic
    ufc_lit = Mke.RGBf(0.55, 0.9, 0.62),  # ceramic glowing
    reflector = Mke.RGBf(0.97, 0.97, 0.95),  # TiO₂ epoxy: white
    tungsten = Mke.RGBf(0.42, 0.44, 0.5),
    photodiode = Mke.RGBf(0.24, 0.27, 0.55),  # back-illuminated silicon
    cmos = Mke.RGBf(0.2, 0.22, 0.27),  # the readout electronics
    gold = Mke.RGBf(0.86, 0.66, 0.2),  # vias and stud bumps
    substrate = Mke.RGBf(0.86, 0.83, 0.76),  # ceramics substrate
    board = Mke.RGBf(0.13, 0.4, 0.29),  # circuit board
    water = Mke.RGBf(0.62, 0.78, 0.92),
    beam = Mke.RGBf(0.98, 0.7, 0.25),  # x rays, as in the lab's banner
    light = Mke.RGBf(0.45, 0.9, 0.45),  # the ceramic's light (Gd₂O₂S glows green)
    night = Mke.RGBf(0.05, 0.06, 0.08),  # the dark backdrop of the lab's banner
    night_ring = Mke.RGBf(0.11, 0.13, 0.16),
)

# ╔═╡ dd438c02-b58e-48d6-8247-361bd6056134
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

# ╔═╡ 9c9da424-c175-45f9-908e-795b262fcea5
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

# ╔═╡ 8256033e-bcf6-4488-8854-e7b85ae889aa
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# ╔═╡ e6df8949-80b5-487d-88b1-1207921c731c
"""Where a point in 3-D lands on the drawing."""
project(cam, p) = Mke.Point2f(dot3(p, cam.right), dot3(p, cam.up))

# ╔═╡ f0163adb-f394-49bb-9286-378c2b94ea24
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

# ╔═╡ 80807927-64db-454a-afe6-9f09dd5a9198
"""Shade a face by one light from the upper left, front, as illustrators do."""
function shade(color, normal, cam; light = (-0.35, -0.45, 0.82))
    n = sqrt(dot3(light, light))
    lambert = max(0.0, dot3(normal, light) / n)
    k = 0.62 + 0.42 * lambert
    return Mke.RGBf(min(1, color.r * k), min(1, color.g * k), min(1, color.b * k))
end

# ╔═╡ e4ab74a3-7550-473d-8709-077a862eddd0
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

# ╔═╡ 5f91bc2a-b0e7-49ac-b3b3-66bebdba1b3f
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

# ╔═╡ af6ca42d-91cc-4d7b-b0ff-3a442d8d2ee1
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

# ╔═╡ 859ab6ec-720b-44c9-bd78-25c36161d305
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

# ╔═╡ 87186f98-63f9-437f-8884-98053321491d
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

# ╔═╡ 563445e3-21b2-4257-8fba-319c00be722d
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

# ╔═╡ 6fe48af4-2106-4f8e-a0f6-7f8f60fb6cc6
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

# ╔═╡ 99253ec5-d67a-40f6-a762-aae13b20d4e2
"""A straight arrow from `a` to `b` (2-D)."""
function arrow!(ax, a, b; color = MATERIALS.beam, linewidth = 3, head = 14, linestyle = :solid)
    a, b = Mke.Point2f(a), Mke.Point2f(b)
    t = (b - a) / sqrt(sum(abs2, b - a))
    Mke.lines!(ax, Mke.Point2f[a, b - 0.5 * head / 100 * t]; color, linewidth, linestyle)
    Mke.scatter!(ax, [b]; marker = :utriangle, markersize = head, rotation = atan(t[2], t[1]) - π / 2, color)
    return nothing
end

# ╔═╡ 7a3cca40-1b73-4ab6-9cda-d85264fdd77e
"""A rectangle (2-D) from its lower-left corner, width and height."""
rect!(ax, x, y, w, h, color; kw...) = Mke.poly!(ax, Mke.Rect2f(x, y, w, h); color, kw...)

# ╔═╡ 2119d161-8464-48b4-983c-132fb972d6f6
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

# ╔═╡ 7739a18c-3e77-484b-8c0c-465dae125abd
"""A vertical bracket at `x` from `y0` to `y1`, with `text` to its left."""
function side_bracket!(ax, x, y0, y1, text; color, fontsize = 16, tick = 0.08)
    Mke.lines!(ax, Mke.Point2f[(x + tick, y0), (x, y0), (x, y1), (x + tick, y1)]; color, linewidth = 2)
    return Mke.text!(ax, x - tick, (y0 + y1) / 2; text, color, fontsize, font = :bold, align = (:right, :center), justification = :right)
end

# ╔═╡ 127c8c6f-e757-4731-b2fb-fdbd41314595
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

# ╔═╡ 44d44093-8ce4-467b-968e-0fcdef6ba91f
TAGS = (geant4 = Mke.RGBf(0.2, 0.55, 0.32), not_simulated = Mke.RGBf(0.52, 0.52, 0.57))

# ╔═╡ a98648ee-d676-4adf-b452-3b233838b7a0
"""
The blocks of a patch of the iCT's detector, pulled apart layer by layer: board, substrate, the
readout electronics (a contact under every dexel; their layout is not published, so drawn
generic), the photodiodes on them, the Gd₂O₂S
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

# ╔═╡ 0ed8bccd-ee73-4904-a6d4-6494a90d0b06
"""A block drawn faded toward the background (`amount` 0: as is, 1: gone)."""
function faded(b::Block, amount; to = Mke.RGBf(1, 1, 1))
    mix(a, c) = a + amount * (c - a)
    color = Mke.RGBf(mix(b.color.r, to.r), mix(b.color.g, to.g), mix(b.color.b, to.b))
    return Block(; lo = b.lo, hi = b.hi, color, alpha = b.alpha, stroke = b.stroke)
end

# ╔═╡ 77d395a5-c25d-4949-8bc6-fdef12b43a1d
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

# ╔═╡ df621723-f28e-4d3c-8b88-59f6bc8ab8ee
"""Outline a box (lo, hi) as seen by `cam`: the edge of its silhouette."""
function outline!(ax, cam, lo, hi; color = MATERIALS.beam, linewidth = 3, linestyle = :solid)
    corners = [(x, y, z) for x in (lo[1], hi[1]) for y in (lo[2], hi[2]) for z in (lo[3], hi[3])]
    return Mke.lines!(ax, hull([project(cam, c) for c in corners]); color, linewidth, linestyle)
end

# ╔═╡ d2f10ba6-b309-44cb-b2d7-c03dac409842
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
    R_iso, R_det = ict.source_to_isocenter_mm, ict.source_to_detector_mm
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

# ╔═╡ 6010bf53-d718-4985-8f85-1660ac419be2
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

# ╔═╡ 28956147-e8a6-4fc0-aa03-51873205222c
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

# ╔═╡ a1e828ac-535c-4ae6-a651-f37cf18bcdc2
"""A Gd atom: nucleus and two shells, at `c` with outer radius `r`."""
function atom!(ax, c, r; color = Mke.RGBf(0.45, 0.47, 0.55))
    θ = range(0, 2π; length = 100)
    for (k, rr) in enumerate((0.55r, r))
        Mke.lines!(ax, c[1] .+ rr .* cos.(θ), c[2] .+ rr .* sin.(θ); color = (color, 0.55), linewidth = 1.5)
    end
    Mke.scatter!(ax, [Mke.Point2f(c)]; color, markersize = 18)
    return nothing
end

# ╔═╡ 4a2ba4d6-c8ee-536d-b050-8d8e63896fc4
md"""
## 1. The Scintillator

Philips' FDA filings say only "ceramic scintillator"; its patents for CT detector arrays describe
gadolinium oxysulfide, Gd₂O₂S, activated with praseodymium and cerium at levels far too small to
matter for x rays. This notebook therefore takes the NanoPanel to be a dense Gd₂O₂S ceramic, at
**7.31 g/cm³**, the density Siemens measures for its own Gd₂O₂S ceramic (99.95 % of theoretical).
That is an assumption: Philips publishes neither composition nor density.
"""

# ╔═╡ 3c064df0-e2d1-5119-8221-98b267523e6c
# Gd₂O₂S ceramic at 7.31 g/cm³ (assumed: Philips publishes neither composition nor density)
gos = MC.DetectorMaterial("GOS ceramic"; density = 7.31, formula = "Gd2O2S")

# ╔═╡ f9c206fd-83ce-5049-a55f-7e6f67e51826
# Gadolinium's K-shell: its edge and the energy of the x ray it emits (Kα)
gadolinium = only(MC.k_edges(gos))

# ╔═╡ 02efe6e9-af41-50a5-ab20-dd5e3ea56823
md"""
Gadolinium's **K-edge, 50.24 keV**, sits in the middle of every CT spectrum. Below it a photon
cannot free a K-shell electron; just above it, it can, and absorption jumps sevenfold. Every
such absorption leaves the atom to emit, nine times in ten, a **Gd K x ray of about 43 keV**,
which is just below the edge and so is absorbed only weakly: it can travel a millimetre or more
before stopping, which is the size of a dexel.

Geant4's own cross sections are used throughout (it samples from them); NIST XCOM's are shown
beside them as the comparison.
"""

# ╔═╡ 9adff7b9-1f90-5e35-bfba-1fb3b0ce2850
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

# ╔═╡ 8bef983f-486a-5add-af56-509a943fc213
mu_energies = 10.0:0.25:150.0

# ╔═╡ ce839469-d815-502c-a9a8-b239ef98e6e4
# Geant4's attenuation of the ceramic, per process, in 1/mm
mu_geant4 = Dict(
    p.kind => [MC.geant4_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ f54d6623-1392-523e-9669-cda9e27df0e6
# NIST XCOM's, for comparison
mu_xcom = Dict(
    p.kind => [MC.linear_attenuation(gos, E; kind = p.kind) for E in mu_energies]
        for p in processes
)

# ╔═╡ 1ccba758-e480-4c29-ae8a-c59afe1fc58c
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

# ╔═╡ 58cb196e-00e0-5355-9388-1c235999f0e2
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 680))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How Strongly Gd₂O₂S Ceramic Stops X Rays (7.31 g/cm³)",
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

# ╔═╡ b702ba31-bdd4-535a-8bb5-82d4998424ad
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

# ╔═╡ 9de8c04a-d7fa-563a-9037-a24a4874949f
md"""
### A First Check: A Published Gd₂O₂S Curve

Philips publishes no absorption curve, but Siemens publishes the fraction of x rays that **1.4 mm
of its own Gd₂O₂S ceramic** absorbs (*Detectors & Crystals*, p. 19). It is the probability that a
photon interacts at all, `1 − exp(−μd)`, so it checks the material model and the density assumed
here. The values were read from the vector figure, a smoothed schematic good to about ±0.02–0.03.
"""

# ╔═╡ 4dbb12b4-8b84-55f4-971d-834621179fd2
# Siemens' curve, read from the vector figure (reference_data/siemens_ufc_1p4mm_absorption.csv)
siemens_curve = let
    path = joinpath(@__DIR__, "..", "reference_data", "siemens_ufc_1p4mm_absorption.csv")
    data, header = DelimitedFiles.readdlm(path, ','; comments = true, header = true)
    (energy_keV = Float64.(data[:, 1]), absorption = Float64.(data[:, 2]))
end

# ╔═╡ e3e5fb2d-8839-5c6d-bc55-6ab33f4fbdc0
# The crystal depth (assumed: not published; Philips' patent says "about 2 mm or less")
depth_mm = 1.4

# ╔═╡ 1c6c8e35-e26f-5c80-af70-8886225448fd
# The probability a photon interacts in 1.4 mm of the ceramic
interaction_curve = [1 - exp(-MC.geant4_attenuation(gos, E) * 1.4) for E in mu_energies]

# ╔═╡ d088def7-cf27-4b01-8822-9924767c1716
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

# ╔═╡ 9494b99e-6ee9-5e7c-bfc6-4107cf482f79
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

# ╔═╡ c250a1e3-30e5-53ce-91ca-7e6d01da00c9
let
    ours = [MC.interpolate_linear(collect(mu_energies), interaction_curve, E) for E in siemens_curve.energy_keV]
    low = siemens_curve.energy_keV .<= 90
    Markdown.parse(
        """
        Up to 90 keV the published curve and Geant4 for 1.4 mm of Gd₂O₂S at 7.31 g/cm³ agree to within
        **$(round(maximum(abs, (siemens_curve.absorption .- ours)[low]); digits = 3))**, inside the figure's reading accuracy: the material model is sound.
        Whether the NanoPanel is this material at this density, nothing published says.
        """
    )
end

# ╔═╡ 9d862dad-bf55-5fdf-a8aa-42c15cea2306
md"""
## 2. One Dexel and Its Neighbours

A dexel is one block of ceramic with reflector on its four sides. Its pitch at the detector face
is documented in Philips' Dunlee datasheet for the 128-row detector: **1.408 mm across the fan by
1.14 mm along z**. The reflector's width is not published; 0.1 mm of TiO₂-loaded epoxy, the usual CT
reflector, is assumed. The crystal's depth is not published either; 1.4 mm is assumed (within the
"about 2 mm or less" of Philips' patent), and the table is saved for other depths too.
"""

# ╔═╡ 394d3e3f-93d1-56fe-bcdb-6c13f6cb7849
# The Brilliance iCT's geometry: FDA K162838 and the Dunlee CD500 datasheet
ict = (
    source_to_isocenter_mm = 570.0,
    source_to_detector_mm = 1040.0,
    columns = 672,
    rows = 128,          # iCT 256; the iCT SP has 64
    rows_sp = 64,
    pitch_at_detector_mm = (1.408, 1.14),
)

# ╔═╡ a754a356-83f4-547f-9d84-1cdda3dbe6b2
# The dexel pitch at the detector face (across the fan, along z), in mm
pitch_mm = ict.pitch_at_detector_mm

# ╔═╡ 428864a1-bd51-40c3-a69b-6101eb1c1eac
# Half the fan angle (degrees): the detector's width at its face over its radius
half_fan_deg = rad2deg(ict.columns * pitch_mm[1] / ict.source_to_detector_mm / 2)

# ╔═╡ 6ab628e2-352e-5a0a-b3ea-dbbc5afeacdd
# The reflector between dexels (assumed: no width is published)
septum_mm = 0.1

# ╔═╡ bb3446dc-7de0-5105-905b-874f3cf8b572
# The block of ceramic each dexel keeps
crystal_mm = pitch_mm .- septum_mm

# ╔═╡ c6349ec8-c5a9-58ff-a25f-2246c548381a
# TiO₂-loaded epoxy (assumed: 40 % TiO₂ by mass, 1.8 g/cm³)
reflector = MC.DetectorMaterial(:TiO2_epoxy)

# ╔═╡ 8670cbca-682f-56ab-b865-7b3a01ff1c2d
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

# ╔═╡ de7908ce-c3e4-5a45-89cd-759a322c3d82
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

# ╔═╡ 6f1cc53f-bdcd-544a-a60c-d85e6ad99214
# Photons landing uniformly on one dexel's ceramic, and on its whole cell (ceramic and its share
# of the reflector); both start 12 mm above the face, above the grid of section 6
crystal_beam = MC.AreaBeam(;
    width_mm = crystal_mm[1],
    height_mm = crystal_mm[2],
    start_mm = 12.0,
)

# ╔═╡ 8cac5e6c-cede-53b4-ab8a-d56c561ebadd
cell_beam = MC.AreaBeam(; width_mm = pitch_mm[1], height_mm = pitch_mm[2], start_mm = 12.0)

# ╔═╡ 5d77a48b-c23d-5aad-a0c9-ad585a95cd1f
# Each dexel collects the energy left in its own block of ceramic (the reflector keeps the light
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

# ╔═╡ 894ed03e-aca5-5497-a05e-e250aa7186a7
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    fig = Mke.Figure(size = (1600, 760))


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

# ╔═╡ 5e60b6ee-9554-5f5c-bc2f-ab22c98206f3
md"""
### What a Photon Does in the Ceramic

`record_tracks` follows a few hundred 60 keV photons landing on the central dexel, every step of
every particle. Two of them are drawn: one whose Gd x ray is absorbed **in the next dexel**, so
its energy is split between two photodiodes, and one whose Gd x ray **leaves through the front
face**, so its 43 keV are lost to every dexel. Both happen often, because the Gd x ray is so
weakly absorbed.
"""

# ╔═╡ 6a1403cc-ac6c-5bdb-9833-3abe703e67ce
tracks = MC.record_tracks(
    dexel_array(depth_mm),
    60.0;
    n = 400,
    beam = crystal_beam,
    seed = 3,
)

# ╔═╡ 9777254d-9999-50b0-833f-4545c6b84cfb
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

# ╔═╡ 7cf5e41e-9f00-5457-aa85-61be46632fe0
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

# ╔═╡ fb1f5699-46f6-455f-b15c-0d9ce1fe58c1
md"""
### Why Don't the Grid's Walls Go Down Through the Ceramic?

Two things can leak from one dexel into the next, and they are stopped (or not) by different parts:

- **Light.** The ceramic glows where the x ray stopped; light reaching the next dexel's photodiode
  would blur the image (*optical crosstalk*). The **white TiO₂ reflector** in the saw cuts stops
  it: light bounces back into its own dexel (its material and width are assumed here; Philips
  publishes neither). Geant4 does not follow the light; the reflector is taken to be perfect.
- **X rays.** A Gd x ray (43 keV) or a Compton-scattered photon crosses 0.1 mm of light epoxy
  almost freely and can stop in the neighbour (*x-ray crosstalk*, the cousin of charge sharing in
  photon-counting detectors). That is the "next door" photon above, and Geant4 follows it.

Tungsten walls through the ceramic would stop the x rays, but they would widen the cuts (less
ceramic per dexel, so fewer photons stopped for the same dose), add tungsten's own x rays (59 and
67 keV) inside the ceramic, and have to be built into a sawn ceramic wafer. And in a scan the
x-ray crosstalk mostly cancels: a dexel gets back from its neighbours what it loses to them
(section 4). What it does cost, a little sharpness and a little noise (the Swank factor), is
measured here. So the grid stands in front, against the scatter from the patient, and only the
reflector goes between the dexels.
"""

# ╔═╡ 9b6b15ba-e18d-4b5c-8687-a36f59bf7f19
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

    ax1 = canvas(fig[1, 1]; title = "The iCT: Grid Above, Reflector in the Saw Cuts", titlesize = 20)
    base!(ax1)
    photon!(ax1, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    half = (p - s) / 2
    Mke.lines!(ax1, Mke.Point2f[hit, (half, -0.15), (0.3, -d)]; color = MATERIALS.light, linewidth = 2.5)
    Mke.lines!(ax1, Mke.Point2f[hit, (-half, -0.3), (-0.25, -d)]; color = MATERIALS.light, linewidth = 2.5)
    photon!(ax1, hit, (1.15p, -1.0); cycles = 3, amplitude = 0.05, color = PALETTE.xray, head = 11)
    star!(ax1)
    Mke.Label(fig[2, 1], "Light (Green): Bounces Off the Reflector, Stays in Its Dexel\nGd X Ray (Orange, 43 keV): Passes Through the Thin, Light Reflector"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax1, -1.65p, 1.65p, -d - 0.3, 1.75)

    ax2 = canvas(fig[1, 2]; title = "Hypothetical: Tungsten Walls Down Through the Ceramic", titlesize = 20)
    base!(ax2; septum = 0.16, walls_through = true)
    photon!(ax2, (0.18, 1.6), hit; cycles = 6, amplitude = 0.02)
    photon!(ax2, hit, (half + 0.02, -0.82); cycles = 2, amplitude = 0.05, color = PALETTE.xray, head = 11)
    Mke.scatter!(ax2, [Mke.Point2f(half + 0.06, -0.85)]; marker = :xcross, markersize = 18, color = PALETTE.xray)
    star!(ax2)
    Mke.Label(fig[2, 2], "Stops the Gd X Ray, but: Wider Cuts Leave Less Ceramic per Dexel,\nTungsten Adds Its Own X Rays (59 and 67 keV), and It Must Be Built In"; fontsize = 16, color = PALETTE.ink, tellwidth = false)
    Mke.limits!(ax2, -1.65p, 1.65p, -d - 0.3, 1.75)
    fig
end

# ╔═╡ 59507a74-4442-5ca0-a716-8ddd907568c7
md"""
## 3. From a Photon to a Signal

The photodiode sees the light of the energy left in its dexel, so what one photon contributes to
the signal is the energy its dexel collects. At a single energy that is not one number but a
spread: the full photon energy when everything stays in the dexel, the energy minus 43 keV when
the Gd x ray leaves, a little when the photon only Compton-scatters once and leaves. Three
energies show the three regimes: 45 keV (below the K-edge), 60 keV (just above it) and 120 keV.
"""

# ╔═╡ 9948f898-1d1d-5a61-856c-15b2704e1306
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

# ╔═╡ 0773a247-588d-565c-8752-5d4e21bb843f
# The energy the dexel they landed on collected from each photon
own_signal = Dict(
    E => vec(MC.dexel_response(run, readout.readout).dexel_energy_keV[:, 2, 2])
        for (E, run) in signal_runs
)

# ╔═╡ baad2b5e-35f8-4132-bea4-0553b9e18cc2
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

# ╔═╡ 39a75aab-0cbc-5ab9-82fa-77f14140b2e0
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

# ╔═╡ d52dcc27-2572-524f-9683-7ba09b3e847d
md"""
### Where a Photon's Energy Goes

Over all the dexels together, every bit of a photon's energy either stays in the ceramic or
leaves it: as the Gd x ray, as a scattered photon, or as the photon itself passing straight
through. The bookkeeping uses one continuous slab of ceramic (no reflector), so that whatever
leaves the ceramic leaves for good; the energy balance closes to within Geant4's tracking cuts.
"""

# ╔═╡ 11d971ce-2bac-52bc-87d5-f5d096b8a656
budget_energies = 15.0:5.0:140.0

# ╔═╡ 7295db49-8d5d-59de-a43e-113ed6a28a2e
budgets = [
    MC.energy_budget(run) for run in MC.simulate_sweep(
            [(MC.SlabDetector(gos, depth_mm; width_mm = 16.0), E) for E in budget_energies];
            n = 50_000,
            beam = crystal_beam,
            seed = 9,
        )
]

# ╔═╡ 790f4caf-4854-40c0-ad00-7550809ff3cc
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

# ╔═╡ 9169c0cc-f78b-5697-ab12-33ad5ca5654c
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

# ╔═╡ 3741ebf6-6078-592b-afc7-ea3706eb3386
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

# ╔═╡ 728d0bdf-38e7-5044-adfa-594b5bd87bdf
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

# ╔═╡ 3f1986f0-e3f2-51c7-925c-25ca42d42836
energies = 1.0:1.0:140.0

# ╔═╡ 099118be-25b9-5d87-a5f1-00931a2c71f7
one_dexel = MC.simulate_sweep(
    [(dexel_array(depth_mm), E) for E in energies];
    n = 100_000,
    beam = crystal_beam,
    seed = 7,
    readout,
)

# ╔═╡ 3b4afd60-19f1-52ec-9355-73a9faa66245
efficiency = (
    interaction = MC.interaction_probability.(one_dexel),
    one_dexel = MC.dexel_efficiency.(one_dexel),
    flat_field = MC.energy_absorption_efficiency.(one_dexel),
    swank = MC.swank_factor.(one_dexel),
    dqe = [d.crystal_keV^2 / (d.n * d.crystal2_keV2) for d in one_dexel],
)

# ╔═╡ ffbbc2e2-8b9c-4353-9704-229251799600
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

# ╔═╡ db494983-9552-582e-857d-9dffb7afdb00
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

# ╔═╡ 62aad628-ba2a-50a0-9040-c5742f0c9d13
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

# ╔═╡ 0c5f1902-7ad2-5f34-9e0d-4a9dd16e3262
md"""
### A Check Against Published Monte Carlo

Boone et al. (Med. Phys. 1999) followed x-ray fluorescence in thin Gd₂O₂S screens with the TART 98
code: a pencil beam at normal incidence on a screen of Gd₂O₂S at 7.34 g/cm³ with 5 % binder
(90 mg/cm² of Gd₂O₂S, the binder not counted), the energy tallied in a circle 0.1 mm across the
beam ("absorbed as primary") and in rings out to 4.5 mm ("lateral"), and ideal sensors in front of
the screen and behind it. At 55 keV they report (p. 908) how the incident energy divides. The same
screen and the same tallies in Geant4:
"""

# ╔═╡ b8d0877a-78e8-5adf-8850-d22dbfe2e2bf
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

# ╔═╡ 27c2a4d2-62ca-413e-8276-6bb62e3850e9
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

# ╔═╡ fcd5bd5f-321e-5322-abcd-872d1486dc05
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

# ╔═╡ 4343e931-d6ea-5aa8-86b7-a3f9ad3e149a
md"""
## 5. Compared with BasisSimulator v0.15.1's Nearest Table

BasisSimulator v0.15.1 has no Philips detector. The nearest table it had is its SOMATOM
Definition Flash table, `UFC_FLASH_MC_EFFICIENCY_LUT`: Gd₂O₂S, and, judging by its shape, made as a pencil beam into
the centre of one dexel 1.4 mm deep at 7.44 g/cm³, tallied in that dexel. It is what a Philips model
would use today if it borrowed a Siemens table. How it compares with the NanoPanel's flat field:
"""

# ╔═╡ 97418e01-601a-5e85-bdb4-0c291a14f494
# BasisSimulator v0.15.1's nearest table: the Definition Flash's Gd₂O₂S (MCNP, Hamidreza
# Khodajou-Chokami), as BasisMonteCarlo keeps it for comparison
table = MC.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash

# ╔═╡ 42d3b3e6-8b64-4cba-a0b9-9f6fa8cf270f
# Is BasisMonteCarlo's copy the table the installed BasisSimulator ships? Up to v0.15.1 it must be,
# exactly; later releases replace it, and the copy stands alone
let
    installed = pkgversion(BS)
    same = isdefined(BS, :UFC_FLASH_MC_EFFICIENCY_LUT) &&
        BS.UFC_FLASH_MC_EFFICIENCY_LUT.energies == table.energies &&
        BS.UFC_FLASH_MC_EFFICIENCY_LUT.efficiency == table.efficiency
    message = if installed <= v"0.15.1"
        same ?
            "✓ The installed BasisSimulator, v$(installed), ships exactly this table (as its Definition Flash's)." :
            "✗ **The installed BasisSimulator, v$(installed), has a different Flash table from BasisMonteCarlo's copy of v0.15.1's.** Check `BASISSIMULATOR_V0_15_1_EFFICIENCY` against BasisSimulator v0.15.1."
    else
        "The installed BasisSimulator is v$(installed). The comparison below is with BasisSimulator v0.15.1's Flash table, which BasisMonteCarlo keeps (`BASISSIMULATOR_V0_15_1_EFFICIENCY`): it does not change with BasisSimulator's version."
    end
    Markdown.parse(message)
end

# ╔═╡ 7cb3e6da-71fa-5220-bc3b-46f8c9659eaa
table_at(E) = table.efficiency[findfirst(==(E), table.energies)]

# ╔═╡ 25c7ab9e-ccc6-4bbc-8b3e-71bc51610a40
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 870))
    cam = camera(-56, 32)
    amber = MATERIALS.beam
    s, d = septum_mm, depth_mm
    cases = (
        (
            title = "BasisSimulator v0.15.1's Flash Table",
            detail = "Siemens Flash Dexels, UFC at 7.44 g/cm³: a Pencil Beam\ninto One Dexel, Only That Dexel Tallied (Notebook 05)",
            pitch = (0.95, 1.05),
            flat = false,
        ),
        (
            title = "This Notebook: the NanoPanel",
            detail = "Dexels of 1.408 × 1.14 mm, Gd₂O₂S at 7.31 g/cm³ (Assumed):\nEvery Dexel Lit, What Each One Collects in a Scan",
            pitch = pitch_mm,
            flat = true,
        ),
    )
    for (k, c) in enumerate(cases)
        ax = canvas(fig[1, k]; title = c.title, titlesize = 21)
        p = c.pitch
        blocks = dexel_blocks(3, 3, p, s, 0.0, d; lit = c.flat ? Set((i, j) for i in 0:2 for j in 0:2) : Set([(1, 1)]))
        extras = Any[]
        if c.flat
            beam = Block((0.0, 0.0, d), (3p[1], 3p[2], d + 3.0), amber; alpha = 0.25, stroke = 0.0)
            push!(extras, (1.0e3, a -> draw_blocks!(a, cam, [beam]; edge_alpha = 0.0)))
        else
            from, to = project(cam, (1.5p[1], 1.5p[2], d + 4.0)), project(cam, (1.5p[1], 1.5p[2], d))
            push!(extras, (1.0e3, a -> arrow!(a, from, to; color = amber, linewidth = 4, head = 14)))
        end
        draw_blocks!(ax, cam, blocks; extras)
        Mke.Label(fig[2, k], c.detail; fontsize = 16, color = PALETTE.muted, tellwidth = false)
    end
    locator!(fig[0, :], :ceramic; note = "Step 2 · Which Table\nDescribes the NanoPanel?")
    fig
end

# ╔═╡ 5bafab62-de83-566e-8809-4542f9986240
let
    Mke.set_theme!(THEME)
    E = collect(energies)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "BasisSimulator v0.15.1's Nearest Table Against the NanoPanel at 1.4 mm",
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

# ╔═╡ 4a9d192f-853c-5c30-b135-b9c654181bd1
let
    at(E) = findfirst(==(E), collect(energies))
    rows = [
        "| $(Int(E)) keV | $(round(table_at(E); digits = 3)) | $(round(efficiency.flat_field[at(E)]; digits = 3)) |"
            for E in (40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
    ]
    Markdown.parse(
        """
        | photons of | BasisSimulator v0.15.1's Flash table | Geant4, NanoPanel flat field |
        |---|---|---|
        $(join(rows, "\n"))

        Just above the K-edge the flat field is higher (each dexel gets back the Gd x rays its neighbours lose); at high
        energy it is lower (7.31 g/cm³ rather than 7.44, and photons landing near a dexel's edge lose more). Borrowing
        the Siemens table would misplace the Philips detector's efficiency by several hundredths at the energies a
        scan uses most.
        """
    )
end

# ╔═╡ f62ede7a-50a6-528f-9fe9-1eee52766a58
md"""
## 6. The Anti-Scatter Grid

The iCT 256's detector has a **two-dimensional**, pre-focused anti-scatter grid (Dunlee's datasheet
for the 128-row detector; Philips' patent EP 2,433,156); the iCT SP's 64-row sibling may have a
one-dimensional one, which nothing published confirms. Material and dimensions are not published.
Built here: **tungsten walls 0.1 mm thick and 10 mm tall** (assumed), one cell per dexel, standing on
the reflector, so they shade no ceramic.
"""

# ╔═╡ 39bf1600-bece-5065-8028-4cbdb0f46cfd
grid = MC.Collimator(;
    height_mm = 10.0,
    wall_mm = 0.1,
    pitch_x_mm = pitch_mm[1],
    pitch_y_mm = pitch_mm[2],
    walls = :both,
    cells = 13,
)

# ╔═╡ ab633fe6-4662-5646-a9bf-e1b2cb27eea4
let
    Mke.set_theme!(THEME)
    p, s, d = pitch_mm, septum_mm, depth_mm
    H, w = grid.height_mm, grid.wall_mm
    fig = Mke.Figure(size = (1600, 1050))


    # (a) to scale, the near corner cut away
    ax = canvas(fig[1, 1]; title = "(a) The 2-D Grid over the Dexels, to Scale", titlesize = 20)
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
    Mke.bracket!(axb, xr - reach, H + 0.1, xr, H + 0.1; text = "H tan θ: These Hit the Wall", orientation = :up, fontsize = 14, color = PALETTE.xray, textcolor = PALETTE.xray)
    Mke.bracket!(axb, 0.5p[1] + 0.12, 0, 0.5p[1] + 0.12, H; text = "H = $(round(Int, H)) mm", orientation = :down, fontsize = 15, color = PALETTE.ink, textcolor = PALETTE.ink)
    Mke.Label(fig[2, 2], "Photons Arriving θ = $(round(Int, θ))° off the Channel's Axis:\nThe Fraction 1 − H tan θ / Opening Reaches the Ceramic"; fontsize = 15, color = PALETTE.ink, tellwidth = false)
    Mke.colsize!(fig.layout, 2, Mke.Relative(0.32))
    locator!(fig[0, :], :grid; note = "Step 1 · The Grid\nIn Front of the Ceramic")
    fig
end

# ╔═╡ 5e8374e9-8118-5d45-a15d-37291ee50143
md"""
Two simulations show what the grid does. Photons arriving straight from the focal spot, landing
uniformly over a dexel's whole cell (ceramic and reflector), with and without the grid: the grid
should cost almost nothing, apart from its own tungsten x rays (59 and 67 keV) reaching the
ceramic above tungsten's K-edge at 69.5 keV. And photons arriving at an angle across the fan, as
scattered photons do: the grid should stop more of them the steeper they come.
"""

# ╔═╡ 2c65e6be-5d25-5bbe-9f42-1fb5f613e03e
grid_energies = 20.0:5.0:140.0

# ╔═╡ 4c4fce07-e9dc-5385-9787-cdc0dbfed941
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

# ╔═╡ 476ceedb-0ddf-50a6-93ee-0f2c6f0d2440
angles_deg = 0.0:1.0:20.0

# ╔═╡ 81032d50-5b45-54e9-98dc-7a3b9d343c87
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

# ╔═╡ 1b7c92ca-1c9f-5bf2-af18-b8f3b682798f
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

# ╔═╡ 1fa6a519-6041-510f-a5c2-8a4ccdfcc902
let
    straight = MC.energy_absorption_efficiency.(grid_runs.with) ./ MC.energy_absorption_efficiency.(grid_runs.without)
    reference = MC.energy_absorption_efficiency(first(angle_runs.without))
    at(angle) = MC.energy_absorption_efficiency(angle_runs.with[findfirst(==(angle), collect(angles_deg))]) / reference
    Markdown.parse(
        """
        Straight from the focal spot the grid changes the collected energy by $(round(100(minimum(straight) - 1); digits = 1)) to
        $(round(100(maximum(straight) - 1); digits = 1)) % (the walls stand on the reflector; above 69.5 keV the walls' own x rays add a little).
        At an angle it is a filter: at 3° it passes $(round(100at(3.0); digits = 0)) % of what arrives, at 6° $(round(100at(6.0); digits = 0)) %. A
        photon that scatters in the patient arrives from anywhere, so most of it is stopped.
        """
    )
end

# ╔═╡ 44ac8c50-3613-5fa6-91fb-68206beb55d4
md"""
## 7. The Whole Detector

The iCT 256's whole detector: 672 columns × 128 rows on an arc of radius 1040 mm around the focal
spot, every dexel a block of ceramic in reflector, under the 2-D grid, all of it focused on the
focal spot. (The real detector is spherical; across the fan and along z every dexel faces the focal
spot either way.) The iCT SP's is the same detector, half as wide. An air scan fires photons
uniformly from the focal spot over the whole detector, and every dexel's signal is summed. Two
questions: is every dexel the same, and does the whole detector collect what the single dexel of
sections 4 and 6 predicts?
"""

# ╔═╡ 1811f34e-14b5-59d0-9953-667d0f42d13c
detector = MC.ArcDetector(;
    columns = ict.columns,
    rows = ict.rows,
    radius_mm = ict.source_to_detector_mm,
    thickness_mm = depth_mm,
    dexel_mm = crystal_mm,
    group = (1, 1),
    gap_mm = (septum_mm, septum_mm),
    septum_material = reflector,
    grid_height_mm = grid.height_mm,
    grid_walls = :both,
    grid_wall_mm = (grid.wall_mm, grid.wall_mm),
    material = gos,
)

# ╔═╡ c1a130ac-f15e-571a-af8a-01ca4badfbcf
air_energies = [40.0, 60.0, 80.0, 120.0]

# ╔═╡ a2b31647-666c-54be-a436-657c08249224
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

# ╔═╡ 00676805-5189-4660-80da-d93cce64ebd7
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 990))

    ax = canvas(fig[1, 1]; title = "The Whole Detector, from the Focal Spot (Depth Exaggerated)", titlesize = 20)
    cam = camera(-62, 26)
    R = ict.source_to_detector_mm
    half = deg2rad(half_fan_deg)
    half_z = ict.rows * pitch_mm[2] / 2
    half_z_sp = ict.rows_sp * pitch_mm[2] / 2
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
                iso = (0.0, -ict.source_to_isocenter_mm, 0.0)
                Mke.scatter!(a, [project(cam, iso)]; marker = :cross, markersize = 18, color = PALETTE.ink)
            end,
        )
    )
    # the iCT SP: the same detector, its central 64 rows
    band = vcat(
        [(R * sin(a), -R * cos(a), -half_z_sp) for a in range(-half, half; length = 40)],
        [(R * sin(a), -R * cos(a), half_z_sp) for a in range(half, -half; length = 40)],
    )
    push!(items, (2.0e4, a -> Mke.lines!(a, [project(cam, q) for q in vcat(band, band[1:1])]; color = PALETTE.reference, linewidth = 3, linestyle = :dash)))
    sort!(items; by = first)
    for (_, f) in items
        f(ax)
    end
    iso = (0.0, -ict.source_to_isocenter_mm, 0.0)
    callout!(ax, project(cam, focal), project(cam, focal) + Mke.Point2f(150, 80), "Focal Spot"; fontsize = 16)
    callout!(ax, project(cam, iso), project(cam, iso) + Mke.Point2f(260, 60), "Isocentre, $(round(Int, ict.source_to_isocenter_mm)) mm Away"; fontsize = 16)
    edge = (R * sin(half), -R * cos(half), half_z)
    callout!(ax, project(cam, edge), project(cam, edge) + Mke.Point2f(120, 120), "iCT 256: $(ict.columns) Columns Across the Fan ($(round(2rad2deg(half); digits = 1))°)\n× $(ict.rows) Rows Along Z (8 cm), at $(round(R; digits = 1)) mm"; fontsize = 16)
    sp = (R * sin(half * 0.6), -R * cos(half * 0.6), half_z_sp)
    callout!(ax, project(cam, sp), project(cam, sp) + Mke.Point2f(160, -170), "iCT SP: the Same Detector,\nIts Central 64 Rows (4 cm, Dashed)"; fontsize = 16, color = PALETTE.reference)
    low = (R * sin(-half / 2), -R * cos(-half / 2), -half_z)
    callout!(ax, project(cam, low), project(cam, low) + Mke.Point2f(-120, -160), "The Outer Rows See the Beam\nat the Cone Angle"; fontsize = 16, align = (:right, :center))
    xs = [project(cam, c)[1] for c in vcat(corners, [focal])]
    ys = [project(cam, c)[2] for c in vcat(corners, [focal])]
    Mke.limits!(ax, minimum(xs) - 520, maximum(xs) + 620, minimum(ys) - 240, maximum(ys) + 200)
    locator!(fig[0, :], :arc; note = "The Whole Arc:\n672 × 128 Dexels (iCT 256)\n672 × 64 (iCT SP)")
    fig
end

# ╔═╡ eb74df12-b344-5362-87f4-55d783c39c99
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))

    # (b) The signal per photon arriving, across the fan, in groups of 40 columns
    axb = Mke.Axis(
        fig[1, 1];
        title = "(a) Energy Collected per Photon, Across the Fan",
        xlabel = "Column (Across the Fan)",
        ylabel = "Energy Collected per Photon\non the Cell, ÷ E",
        limits = ((1, ict.columns), (0.4, 1.0)),
    )
    colors = (PALETTE.photon, PALETTE.xray, PALETTE.electron, PALETTE.kept)
    groups = Iterators.partition(1:ict.columns, 40)
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / ict.columns
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
        ict.columns / 2,
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
        limits = ((1, ict.rows), (0.4, 1.0)),
    )
    row_groups = Iterators.partition(1:ict.rows, max(1, ict.rows ÷ 16))
    for (k, scan) in enumerate(air_scans)
        per_photon = scan.photon_keV / ict.rows
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

# ╔═╡ 69491fa0-423d-5973-9fe6-6aa45682ee37
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
    cone = rad2deg(atan(MC.face_size_mm(detector)[2] / 2 / ict.source_to_detector_mm))
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

# ╔═╡ 50184044-477a-5299-90e6-416d67bf057a
md"""
## 8. The iCT's X-Ray Beam

Philips publishes neither the anode angle nor the filtration (beyond "a titanium filter" of no
stated thickness), so the beam is fitted. The spectra are the installed BasisSimulator's IPEM Report 78 tungsten
spectra at a 10° anode (assumed), through a flat aluminium filter whose thickness is chosen so that
the half-value layer at the isocentre, through the thinnest part of BasisSimulator's large-body
bowtie (2.5 mm of aluminium), matches Trattner et al.'s measurements on a Brilliance iCT:
**6.4, 7.7 and 8.8 mm of aluminium at 80, 100 and 120 kV**.
"""

# ╔═╡ 8c632429-70dc-57e1-a968-7f7f0c831027
# Measured half-value layers at the isocentre, Brilliance iCT (Trattner et al., Med. Phys. 2017)
measured_hvl = [(kVp = 80, hvl = 6.4), (kVp = 100, hvl = 7.7), (kVp = 120, hvl = 8.8)]

# ╔═╡ ad0ee5e7-30f3-5144-9618-99680c8b58c0
aluminium = MC.DetectorMaterial(:Al)

# ╔═╡ 2b4ddb85-5b0e-5572-a884-08c034f20cbe
# The large-body bowtie at its thinnest, the ray through the isocentre: aluminium only, in mm
bowtie_al_mm = let
    bowtie = BS.resolve_bowtie_filter(:large_body)
    centre = argmin(abs.(bowtie.angles))
    10 * bowtie.thickness[centre, findfirst(==("Al"), bowtie.materials)]
end

# ╔═╡ ed171294-6af7-58ed-bceb-efff28a6b4d8
"""
The photons per mAs per mm² reaching the detector, per bin, for an iCT protocol at `kVp`, with
`flat_mm` of aluminium as the beam's filtration and a 10° anode.
"""
function ict_spectrum(kVp; flat_mm)
    scanner = BS.EICTScanner(
        source_to_isocenter = ict.source_to_isocenter_mm,
        source_to_detector = ict.source_to_detector_mm,
        detector_rows = ict.rows,
        detector_cols = ict.columns,
        detector_row_size = pitch_mm[2] * ict.source_to_isocenter_mm / ict.source_to_detector_mm,
        detector_col_size = pitch_mm[1] * ict.source_to_isocenter_mm / ict.source_to_detector_mm,
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

# ╔═╡ e676d602-7cac-54de-999e-a7465e957763
# The half-value layer at the isocentre for `flat_mm` of filtration, through the bowtie's centre
function hvl_at(kVp, flat_mm)
    e, w = ict_spectrum(kVp; flat_mm)
    μ = [MC.linear_attenuation(aluminium, max(E, 1.0)) for E in e]
    return MC.half_value_layer(e, w .* exp.(-μ .* bowtie_al_mm))
end

# ╔═╡ 0fcfa999-406e-5a3f-9e9c-4494ebe0d683
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

# ╔═╡ 659f32de-5a65-54da-a298-5515dc4eb347
hvl_check = [(kVp = m.kVp, fitted = hvl_at(m.kVp, filtration), measured = m.hvl) for m in measured_hvl]

# ╔═╡ 898b3f4c-dbec-4bd0-9e8c-c8bf709d3425
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 380))
    for (k, (al, reading, title)) in enumerate(((0, "100 %", "No Aluminium in the Beam"), (8.8, "50 %", "8.8 mm of Aluminium at 120 kV: Half Is Left")))
        ax = canvas(fig[1, k]; title, titlesize = 20)
        Mke.scatter!(ax, [Mke.Point2f(0, 0)]; marker = :rect, markersize = 22, color = PALETTE.ink)
        Mke.text!(ax, 0, -0.45; text = "Tube\n(Through the\nBowtie's Centre)", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
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

# ╔═╡ ffcbc0d1-307e-598b-8a88-9ae6a0955a96
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
        label = "Measured (Trattner et al. 2017)",
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

# ╔═╡ de610460-4367-5ed4-8880-f975f3c423c0
let
    worst = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        **$(round(filtration; digits = 1)) mm of aluminium** at a 10° anode reproduces the measured half-value layers to within
        **$(round(worst; digits = 2)) mm** at every tube voltage: one filtration fits all three, which is what a real tube's fixed filtration
        does. This fitted beam is the iCT beam for the rest of the notebook and is saved with the table. (The fit
        lumps the unstated titanium filter, the tube's own filtration and the anode angle into one aluminium thickness;
        the bowtie the measurement used is not stated.)
        """
    )
end

# ╔═╡ 4d4fe7bd-017b-56cd-9fb4-71993782ef42
md"""
The tube voltage decides everything that follows, so choose it here:
"""

# ╔═╡ 3d965bcf-74b8-502a-8ea1-896161f71704
@bind kvp PlutoUI.Slider([80, 100, 120, 140]; default = 120, show_value = true)

# ╔═╡ 458f37c2-1cda-54e0-805f-21366ba9fb57
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

# ╔═╡ 8686060d-4ede-482c-9bb9-108b4f29f4ba
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 810))

    ax = canvas(fig[1, 1])
    amber = MATERIALS.beam
    # tube: envelope, cathode, anode at 10° (assumed)
    Mke.poly!(ax, Mke.Rect2f(-0.6, -1.5, 2.4, 3.0); color = Mke.RGBf(0.93, 0.94, 0.96), strokecolor = PALETTE.slab_edge, strokewidth = 1.5)
    rect!(ax, -0.45, -0.25, 0.3, 0.5, PALETTE.slab_edge)
    Mke.text!(ax, -0.3, -0.45; text = "Cathode", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
    Mke.poly!(ax, Mke.Point2f[(1.2, -0.9), (1.55, -0.9), (1.55, 0.9), (1.35, 0.9)]; color = Mke.RGBf(0.6, 0.45, 0.3))
    Mke.text!(ax, 1.42, -1.0; text = "Tungsten Anode, 10° (Assumed)", align = (:center, :top), fontsize = 14, color = PALETTE.muted)
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
        (3.9, 0.22, Mke.RGBf(0.75, 0.77, 0.8), "$(round(filtration; digits = 1)) mm Al (Fitted)", "Fitted to the\nMeasured HVLs"),
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
    Mke.text!(ax, xb + 0.25, -hb - 0.12; text = "2.5 mm Al at the Centre\n(BasisSimulator's Large-Body)", align = (:center, :top), fontsize = 13, color = PALETTE.muted)
    # patient and isocentre
    θ = range(0, 2π; length = 100)
    xi = 9.6
    Mke.poly!(ax, Mke.Point2f.(xi .+ 1.1 .* cos.(θ), 0.8 .* sin.(θ)); color = (MATERIALS.water, 0.6), strokecolor = (PALETTE.photon, 0.5))
    Mke.scatter!(ax, [Mke.Point2f(xi, 0)]; marker = :cross, markersize = 16, color = PALETTE.ink)
    Mke.text!(ax, xi, 0.95; text = "Patient (Isocentre, $(round(Int, ict.source_to_isocenter_mm)) mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    # detector with grid
    rect!(ax, x_end, -2.7, 0.15, 5.4, MATERIALS.ufc; strokecolor = PALETTE.slab_edge)
    for y in range(-2.6, 2.6; length = 14)
        rect!(ax, x_end - 0.35, y - 0.01, 0.35, 0.02, MATERIALS.tungsten)
    end
    Mke.text!(ax, x_end + 0.1, 2.85; text = "Grid + Detector\n($(round(Int, ict.source_to_detector_mm)) mm)", align = (:center, :bottom), fontsize = 15, font = :bold, color = PALETTE.ink)
    Mke.limits!(ax, -0.8, 14.6, -3.4, 3.9)
    locator!(fig[0, :], :tube; note = "The X-Ray Tube and\nWhat the Beam Passes")
    fig
end

# ╔═╡ 75de37dc-82d1-5ea1-a8fc-f86b81c2f954
let
    Mke.set_theme!(THEME)
    spectrum = ict_spectrum(kvp; flat_mm = filtration)
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
        label = "Geant4, NanoPanel Flat Field",
    )
    Mke.vlines!(axb, gadolinium.edge_keV; color = (PALETTE.xray, 0.6), linestyle = :dot)
    Mke.axislegend(axb; position = :rt)
    fig
end

# ╔═╡ 4070778b-47ba-561c-86c5-99008cb991ba
md"""
## 9. Signal and Noise in a Scan

An energy-integrating detector weighs every photon by the energy it leaves, so what a protocol
sees is the efficiency averaged over its spectrum, weighted by `Φ(E) · E`; its zero-frequency
DQE is `(Σ Φ m₁)² / (Σ Φ · Σ Φ m₂)`, with `m₁(E)` and `m₂(E)` the mean and mean square of the
energy left per photon at each energy. Every iCT tube voltage, through the fitted beam:
"""

# ╔═╡ 566d74f6-a104-5697-b4cd-15e7cd351eed
scan_protocols = [(label = "$(k) kV", spectrum = ict_spectrum(k; flat_mm = filtration)) for k in (80, 100, 120, 140)]

# ╔═╡ e73cb88d-3684-5b40-b2b1-549d4394a785
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

# ╔═╡ 9beb0f0e-4989-41b9-817f-b8c7aebdba99
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

# ╔═╡ 90dc1e82-8b70-5ec7-99a3-af34e29179aa
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

# ╔═╡ c65ca608-7514-50d2-ace5-31a7f8abbfce
let
    rows = [
        "| $(m.label) | $(round(m.table; digits = 3)) | $(round(m.geant4; digits = 3)) | $(round(100(m.geant4 / m.table - 1); digits = 1)) % | $(round(m.dqe; digits = 3)) |"
            for m in scan_metrics
    ]
    Markdown.parse(
        """
        | protocol | BasisSimulator v0.15.1's Flash table | Geant4, NanoPanel 1.4 mm | change | DQE(0) |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Both are per photon landing on a dexel's ceramic; the fill factor applies to both alike.
        """
    )
end

# ╔═╡ 9bfd259e-64a8-4107-8ab8-0e1d3ca765df
md"""
## 10. A Clinical Beam, for Real

Every efficiency so far came from photons of **one energy**, landing straight on **one dexel**.
A scan is neither: the iCT's tube sends a spectrum, hardened by the bowtie and by the patient,
from a focal spot over the whole arc, so the dexels at the edges of the fan and the outer rows
see it at an angle, through the grid. BasisSimulator never simulates that photon by photon; it
weights the single-energy table by the spectrum. This section checks that the shortcut is exact,
by doing it the long way:

- the iCT's **120 kV beam** (BasisSimulator's spectrum through the aluminium fitted in section 8,
  through the bowtie's centre), **bare and after 20 and 35 cm of water** (the patient's primary photons; the
  patient's scatter is notebook 10's);
- each one **drawn photon by photon from its spectrum**, onto one dexel's ceramic (1.6 million
  photons) and from the focal spot over **the whole arc** with its grid (3.2 million);
- against the table **weighted by the same spectrum**: `Σ φ E η / Σ φ E` for the efficiency, and
  `(Σ φ m₁)² / (Σ φ · Σ φ m₂)` for DQE(0), with `m₁`, `m₂` the mean and mean square of the energy
  a photon leaves (section 4).
"""

# ╔═╡ 2f70ff11-7608-4dfa-bf9d-337a2272ef52
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
    Mke.text!(ax, 0, -R - 0.7; text = "The Whole Arc and Its Grid (672 × 128 Dexels)", align = (:center, :top), fontsize = 15, font = :bold, color = PALETTE.ink)
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

# ╔═╡ 7d660806-420b-4512-a6b6-7b5d67e43bff
# The iCT's 120 kV beam on the central ray: BasisSimulator's spectrum through the fitted aluminium
# and the bowtie's centre, then through 0, 20 and 35 cm of water (the primary photons), on its
# 0.5 keV grid
clinical_beams = let
    water = MC.DetectorMaterial(:Water)
    e, w = ict_spectrum(120; flat_mm = filtration)
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

# ╔═╡ e037db3c-0776-4422-9c93-13e0c5bb7797
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

# ╔═╡ 28e91d48-b64b-4e0a-8b56-bbdde8415380
# The long way against the shortcut, for each beam: efficiency and DQE(0) on one dexel, and the
# energy the whole arc collects per photon; each direct value with its noise (the 16 chunks' spread)
clinical_check = let
    E = collect(energies)
    m1 = [d.crystal_keV / d.n for d in one_dexel]
    m2 = [d.crystal2_keV2 / d.n for d in one_dexel]
    cell = MC.energy_absorption_efficiency.(grid_runs.with)
    at(table, x) = MC.interpolate_linear(E, table, x)
    # the whole arc against one cell, from section 7's own air scans: the fan and cone angles (and the
    # model's arc, not tilted along z) change it a little; interpolated between the scans' energies
    cell_at(x) = MC.interpolate_linear(collect(grid_energies), cell, clamp(x, 20.0, 140.0))
    arc_ratio = [sum(s.energy_keV) / s.photon_keV / cell_at(E) for (E, s) in zip(air_energies, air_scans)]
    ratio_at(x) = MC.interpolate_linear(air_energies, arc_ratio, clamp(x, first(air_energies), last(air_energies)))
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

# ╔═╡ 959bd17e-739c-4542-8b56-60214b3dc0fc
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

# ╔═╡ 3d563e0f-a685-4c2f-9d56-74ed979fe816
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

# ╔═╡ 899d1884-4c2a-51ff-ab93-a5b7af30bc97
md"""
## The Table to Use

When BasisSimulator gains a Philips Brilliance iCT, this is the table to use, and why:

- **The flat field, not one dexel under a pencil beam.** In a scan every dexel is illuminated, and
  the flat field is what each one collects (section 4).
- **Gd₂O₂S ceramic at 7.31 g/cm³, 1.4 mm deep** (assumed). Philips publishes neither; its patents
  describe Gd₂O₂S arrays "about 2 mm or less" deep, and 1.4 mm is the one Gd₂O₂S CT thickness a
  vendor (Siemens) quotes. Every depth from 0.8 to 2.0 mm is saved beside it.
- **The fill factor of the assumed reflector**, about 0.93 across × 0.91 along z; the grid's walls
  stand on the reflector and take nothing more (section 6).
- **The fitted beam**: BasisSimulator's IPEM spectra at a 10° anode through the fitted aluminium,
  which reproduces the measured half-value layers (section 8).

The iCT 256 and the iCT SP share the detector, so both files hold the same efficiency; they differ in
rows, coverage and grid.
"""

# ╔═╡ 753b7d0c-b3e0-59e2-a8b8-a3ce8d806c00
md"""
## Saving the Tables

Two files in `lib/BasisMonteCarlo/tables/`, one per scanner, in the layout of BasisSimulator's
tables (`energies`, `efficiency`): the flat-field efficiency of the NanoPanel as
modelled here, beside the one-dexel efficiency, the interaction probability, the Swank factor,
DQE(0), the efficiency for depths from 0.8 to 2.0 mm, and the fitted beam.
"""

# ╔═╡ c7de753a-cca5-5efe-97f6-71973736c4e4
table_depths = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# ╔═╡ 9c331ef8-e137-562d-a92c-5035f6921af6
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

# ╔═╡ 55a7667e-c0bc-5685-8ac3-dd8dc1077898
# The machine path stays out of the rendered page
table_dir = joinpath(@__DIR__, "..", "..", "tables");

# ╔═╡ ff228af5-db4f-5f96-bdf7-fd9de37b4fea
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
        "decision" => "use `efficiency`: flat field, Gd2O2S ceramic at 7.31 g/cm3 and 1.4 mm deep (both assumed; Philips publishes neither); fill factor of the assumed 0.1 mm reflector",
        "definition" => "mean energy collected per photon landing on a dexel's ceramic with every dexel illuminated (flat field), divided by the photon energy; the fill factor applies separately",
        "beam" => Dict(
            "flat_filter_al_mm" => filtration,
            "anode_angle_deg" => 10,
            "bowtie" => "BasisSimulator :large_body",
            "fitted_to" => "measured HVL 6.4, 7.7, 8.8 mm Al at 80, 100, 120 kV (Trattner et al. 2017)",
        ),
        "photons_per_energy" => 100_000,
        "compared_with" => "BasisSimulator v0.15.1's nearest table, the Definition Flash's UFC_FLASH_MC_EFFICIENCY_LUT (BasisMonteCarlo.BASISSIMULATOR_V0_15_1_EFFICIENCY.ufc_flash)",
        "physics" => string(MC.PhysicsSettings()),
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 07_philips_brilliance_ict.jl",
    )
    paths = String[]
    for (file, scanner, rows, grid) in (
            (
                "philips_brilliance_ict256_efficiency_geant4.jls",
                "Philips Brilliance iCT 256",
                ict.rows,
                "2-D, pre-focused (Dunlee CD500)",
            ),
            (
                "philips_brilliance_ict_sp_efficiency_geant4.jls",
                "Philips Brilliance iCT SP (iCT 128)",
                ict.rows_sp,
                "1-D or 2-D: not confirmed",
            ),
        )
        contents = merge(
            common,
            Dict{String, Any}(
                "detector" => Dict(
                    "scanner" => scanner,
                    "material" => "Gd2O2S ceramic, 7.31 g/cm3 (assumed)",
                    "depth_mm" => depth_mm,
                    "pitch_mm" => pitch_mm,
                    "columns" => ict.columns,
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

# ╔═╡ c57c863f-8b8d-514b-9a96-a1dac0cd9a41
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

# ╔═╡ b47feb77-eb44-530c-9eb8-8993d09a5760
md"""
## Where This Leaves the Brilliance iCT
"""

# ╔═╡ 41a509c6-8f7f-5833-b89a-224880f73922
let
    m120 = scan_metrics[findfirst(m -> m.label == "120 kV", scan_metrics)]
    worst_hvl = maximum(abs(c.fitted - c.measured) for c in hvl_check)
    Markdown.parse(
        """
        - **The geometry is documented**: 570 / 1040 mm, 672 × 128 dexels (64 on the iCT SP) of 1.408 × 1.14 mm at the
          detector, a spherical detector behind a 2-D grid. **The scintillator is not**: Gd₂O₂S ceramic at 7.31 g/cm³ and
          1.4 mm deep are assumptions, with other depths saved.
        - **The beam is fitted**: $(round(filtration; digits = 1)) mm of aluminium at a 10° anode reproduces the measured half-value layers to
          within $(round(worst_hvl; digits = 2)) mm.
        - **What a scan sees** is the flat-field efficiency: $(round(m120.geant4; digits = 3)) over the 120 kV spectrum, with DQE(0)
          $(round(m120.dqe; digits = 3)); borrowing BasisSimulator v0.15.1's Flash table would give $(round(m120.table; digits = 3)).
        - **The whole detector** collects what one dexel's cell does.
        - **Saved:** $(join(["`$(basename(p))`" for p in saved], " and ")), for BasisSimulator's Philips Brilliance iCT.
        """
    )
end

# ╔═╡ Cell order:
# ╟─1e8019db-b5b1-5ad1-a89a-ffa9fb0e8341
# ╟─83592696-cd7f-4682-99b3-1a0e047aaff4
# ╟─c34c0bd0-aa5c-5e4a-8bda-740e8ca07147
# ╟─f84d9387-4b88-489e-9ffc-9614cc74dced
# ╟─a967d445-3fe2-4149-859b-4e544f9376b7
# ╟─b79e0b22-d18c-415b-a7f3-bf3e3f04d397
# ╟─4ad6e0a4-6bbe-48da-a921-12055b764503
# ╟─e1a742dd-c09a-5ada-87f2-df7c195ed69b
# ╟─ec086b05-5dc0-5983-bb6f-4f7ebd3781af
# ╠═03b0eb03-c3fe-59cb-acd8-5a14b0d80207
# ╠═e457121e-c315-501e-aa97-d71761eb2100
# ╠═cf074860-7eb8-551a-9545-d52b492f3d22
# ╠═a281979d-cff0-5fa9-87ca-baf1abec600c
# ╠═6ded5641-4694-5be0-8dd3-61dbc4c47466
# ╠═6399c96e-ad68-5802-b2fa-ed8629257ff3
# ╠═5fab2bcc-a52b-5a40-bfd2-4c3ab57a916f
# ╠═98b6fe03-3bc8-58a3-a3d2-51eb0a66db65
# ╠═673ac0c5-46e5-5961-9fcd-014c44cf2edd
# ╠═092f835e-0e32-57d5-ba25-6a5783c55235
# ╠═293d3699-48da-599a-bef5-fdd75fda7e1b
# ╠═88fd7a1a-b153-5163-82fe-94476836d0c4
# ╠═43f718d9-bdcd-55dc-8181-8ec3ca691693
# ╠═68f8c99b-e252-5cf4-a55b-ad5517bc4017
# ╠═fe7cf193-49ce-5b3e-8d57-69460f810f9a
# ╟─16035000-554f-4376-8013-68b105841b6f
# ╠═7500fcfc-2a78-44ab-8a2e-7e29a5f1efdf
# ╠═dd438c02-b58e-48d6-8247-361bd6056134
# ╠═9c9da424-c175-45f9-908e-795b262fcea5
# ╠═8256033e-bcf6-4488-8854-e7b85ae889aa
# ╠═e6df8949-80b5-487d-88b1-1207921c731c
# ╠═f0163adb-f394-49bb-9286-378c2b94ea24
# ╠═80807927-64db-454a-afe6-9f09dd5a9198
# ╠═e4ab74a3-7550-473d-8709-077a862eddd0
# ╠═5f91bc2a-b0e7-49ac-b3b3-66bebdba1b3f
# ╠═af6ca42d-91cc-4d7b-b0ff-3a442d8d2ee1
# ╠═859ab6ec-720b-44c9-bd78-25c36161d305
# ╠═87186f98-63f9-437f-8884-98053321491d
# ╠═563445e3-21b2-4257-8fba-319c00be722d
# ╠═6fe48af4-2106-4f8e-a0f6-7f8f60fb6cc6
# ╠═99253ec5-d67a-40f6-a762-aae13b20d4e2
# ╠═7a3cca40-1b73-4ab6-9cda-d85264fdd77e
# ╠═2119d161-8464-48b4-983c-132fb972d6f6
# ╠═7739a18c-3e77-484b-8c0c-465dae125abd
# ╠═127c8c6f-e757-4731-b2fb-fdbd41314595
# ╠═44d44093-8ce4-467b-968e-0fcdef6ba91f
# ╠═a98648ee-d676-4adf-b452-3b233838b7a0
# ╠═0ed8bccd-ee73-4904-a6d4-6494a90d0b06
# ╠═77d395a5-c25d-4949-8bc6-fdef12b43a1d
# ╠═df621723-f28e-4d3c-8b88-59f6bc8ab8ee
# ╠═d2f10ba6-b309-44cb-b2d7-c03dac409842
# ╠═6010bf53-d718-4985-8f85-1660ac419be2
# ╠═28956147-e8a6-4fc0-aa03-51873205222c
# ╠═a1e828ac-535c-4ae6-a651-f37cf18bcdc2
# ╟─4a2ba4d6-c8ee-536d-b050-8d8e63896fc4
# ╠═3c064df0-e2d1-5119-8221-98b267523e6c
# ╠═f9c206fd-83ce-5049-a55f-7e6f67e51826
# ╟─02efe6e9-af41-50a5-ab20-dd5e3ea56823
# ╠═9adff7b9-1f90-5e35-bfba-1fb3b0ce2850
# ╠═8bef983f-486a-5add-af56-509a943fc213
# ╠═ce839469-d815-502c-a9a8-b239ef98e6e4
# ╠═f54d6623-1392-523e-9669-cda9e27df0e6
# ╟─1ccba758-e480-4c29-ae8a-c59afe1fc58c
# ╟─58cb196e-00e0-5355-9388-1c235999f0e2
# ╟─b702ba31-bdd4-535a-8bb5-82d4998424ad
# ╟─9de8c04a-d7fa-563a-9037-a24a4874949f
# ╠═4dbb12b4-8b84-55f4-971d-834621179fd2
# ╠═e3e5fb2d-8839-5c6d-bc55-6ab33f4fbdc0
# ╠═1c6c8e35-e26f-5c80-af70-8886225448fd
# ╟─d088def7-cf27-4b01-8822-9924767c1716
# ╟─9494b99e-6ee9-5e7c-bfc6-4107cf482f79
# ╟─c250a1e3-30e5-53ce-91ca-7e6d01da00c9
# ╟─9d862dad-bf55-5fdf-a8aa-42c15cea2306
# ╠═394d3e3f-93d1-56fe-bcdb-6c13f6cb7849
# ╠═a754a356-83f4-547f-9d84-1cdda3dbe6b2
# ╠═428864a1-bd51-40c3-a69b-6101eb1c1eac
# ╠═6ab628e2-352e-5a0a-b3ea-dbbc5afeacdd
# ╠═bb3446dc-7de0-5105-905b-874f3cf8b572
# ╠═c6349ec8-c5a9-58ff-a25f-2246c548381a
# ╟─8670cbca-682f-56ab-b865-7b3a01ff1c2d
# ╠═de7908ce-c3e4-5a45-89cd-759a322c3d82
# ╠═6f1cc53f-bdcd-544a-a60c-d85e6ad99214
# ╠═8cac5e6c-cede-53b4-ab8a-d56c561ebadd
# ╠═5d77a48b-c23d-5aad-a0c9-ad585a95cd1f
# ╟─894ed03e-aca5-5497-a05e-e250aa7186a7
# ╟─5e60b6ee-9554-5f5c-bc2f-ab22c98206f3
# ╠═6a1403cc-ac6c-5bdb-9833-3abe703e67ce
# ╠═9777254d-9999-50b0-833f-4545c6b84cfb
# ╟─7cf5e41e-9f00-5457-aa85-61be46632fe0
# ╟─fb1f5699-46f6-455f-b15c-0d9ce1fe58c1
# ╟─9b6b15ba-e18d-4b5c-8687-a36f59bf7f19
# ╟─59507a74-4442-5ca0-a716-8ddd907568c7
# ╠═9948f898-1d1d-5a61-856c-15b2704e1306
# ╠═0773a247-588d-565c-8752-5d4e21bb843f
# ╟─baad2b5e-35f8-4132-bea4-0553b9e18cc2
# ╟─39a75aab-0cbc-5ab9-82fa-77f14140b2e0
# ╟─d52dcc27-2572-524f-9683-7ba09b3e847d
# ╠═11d971ce-2bac-52bc-87d5-f5d096b8a656
# ╠═7295db49-8d5d-59de-a43e-113ed6a28a2e
# ╟─790f4caf-4854-40c0-ad00-7550809ff3cc
# ╟─9169c0cc-f78b-5697-ab12-33ad5ca5654c
# ╟─3741ebf6-6078-592b-afc7-ea3706eb3386
# ╟─728d0bdf-38e7-5044-adfa-594b5bd87bdf
# ╠═3f1986f0-e3f2-51c7-925c-25ca42d42836
# ╠═099118be-25b9-5d87-a5f1-00931a2c71f7
# ╠═3b4afd60-19f1-52ec-9355-73a9faa66245
# ╟─ffbbc2e2-8b9c-4353-9704-229251799600
# ╟─db494983-9552-582e-857d-9dffb7afdb00
# ╟─62aad628-ba2a-50a0-9040-c5742f0c9d13
# ╟─0c5f1902-7ad2-5f34-9e0d-4a9dd16e3262
# ╠═b8d0877a-78e8-5adf-8850-d22dbfe2e2bf
# ╟─27c2a4d2-62ca-413e-8276-6bb62e3850e9
# ╟─fcd5bd5f-321e-5322-abcd-872d1486dc05
# ╟─4343e931-d6ea-5aa8-86b7-a3f9ad3e149a
# ╠═97418e01-601a-5e85-bdb4-0c291a14f494
# ╟─42d3b3e6-8b64-4cba-a0b9-9f6fa8cf270f
# ╠═7cb3e6da-71fa-5220-bc3b-46f8c9659eaa
# ╟─25c7ab9e-ccc6-4bbc-8b3e-71bc51610a40
# ╟─5bafab62-de83-566e-8809-4542f9986240
# ╟─4a9d192f-853c-5c30-b135-b9c654181bd1
# ╟─f62ede7a-50a6-528f-9fe9-1eee52766a58
# ╠═39bf1600-bece-5065-8028-4cbdb0f46cfd
# ╟─ab633fe6-4662-5646-a9bf-e1b2cb27eea4
# ╟─5e8374e9-8118-5d45-a15d-37291ee50143
# ╠═2c65e6be-5d25-5bbe-9f42-1fb5f613e03e
# ╠═4c4fce07-e9dc-5385-9787-cdc0dbfed941
# ╠═476ceedb-0ddf-50a6-93ee-0f2c6f0d2440
# ╠═81032d50-5b45-54e9-98dc-7a3b9d343c87
# ╟─1b7c92ca-1c9f-5bf2-af18-b8f3b682798f
# ╟─1fa6a519-6041-510f-a5c2-8a4ccdfcc902
# ╟─44ac8c50-3613-5fa6-91fb-68206beb55d4
# ╠═1811f34e-14b5-59d0-9953-667d0f42d13c
# ╠═c1a130ac-f15e-571a-af8a-01ca4badfbcf
# ╠═a2b31647-666c-54be-a436-657c08249224
# ╟─00676805-5189-4660-80da-d93cce64ebd7
# ╟─eb74df12-b344-5362-87f4-55d783c39c99
# ╟─69491fa0-423d-5973-9fe6-6aa45682ee37
# ╟─50184044-477a-5299-90e6-416d67bf057a
# ╠═8c632429-70dc-57e1-a968-7f7f0c831027
# ╠═ad0ee5e7-30f3-5144-9618-99680c8b58c0
# ╠═2b4ddb85-5b0e-5572-a884-08c034f20cbe
# ╠═ed171294-6af7-58ed-bceb-efff28a6b4d8
# ╠═e676d602-7cac-54de-999e-a7465e957763
# ╠═0fcfa999-406e-5a3f-9e9c-4494ebe0d683
# ╠═659f32de-5a65-54da-a298-5515dc4eb347
# ╟─898b3f4c-dbec-4bd0-9e8c-c8bf709d3425
# ╟─ffcbc0d1-307e-598b-8a88-9ae6a0955a96
# ╟─de610460-4367-5ed4-8880-f975f3c423c0
# ╟─4d4fe7bd-017b-56cd-9fb4-71993782ef42
# ╠═3d965bcf-74b8-502a-8ea1-896161f71704
# ╠═458f37c2-1cda-54e0-805f-21366ba9fb57
# ╟─8686060d-4ede-482c-9bb9-108b4f29f4ba
# ╟─75de37dc-82d1-5ea1-a8fc-f86b81c2f954
# ╟─4070778b-47ba-561c-86c5-99008cb991ba
# ╠═566d74f6-a104-5697-b4cd-15e7cd351eed
# ╠═e73cb88d-3684-5b40-b2b1-549d4394a785
# ╟─9beb0f0e-4989-41b9-817f-b8c7aebdba99
# ╟─90dc1e82-8b70-5ec7-99a3-af34e29179aa
# ╟─c65ca608-7514-50d2-ace5-31a7f8abbfce
# ╟─9bfd259e-64a8-4107-8ab8-0e1d3ca765df
# ╟─2f70ff11-7608-4dfa-bf9d-337a2272ef52
# ╠═7d660806-420b-4512-a6b6-7b5d67e43bff
# ╠═e037db3c-0776-4422-9c93-13e0c5bb7797
# ╠═28e91d48-b64b-4e0a-8b56-bbdde8415380
# ╟─959bd17e-739c-4542-8b56-60214b3dc0fc
# ╟─3d563e0f-a685-4c2f-9d56-74ed979fe816
# ╟─899d1884-4c2a-51ff-ab93-a5b7af30bc97
# ╟─753b7d0c-b3e0-59e2-a8b8-a3ce8d806c00
# ╠═c7de753a-cca5-5efe-97f6-71973736c4e4
# ╠═9c331ef8-e137-562d-a92c-5035f6921af6
# ╠═55a7667e-c0bc-5685-8ac3-dd8dc1077898
# ╠═ff228af5-db4f-5f96-bdf7-fd9de37b4fea
# ╠═c57c863f-8b8d-514b-9a96-a1dac0cd9a41
# ╟─b47feb77-eb44-530c-9eb8-8993d09a5760
# ╟─41a509c6-8f7f-5833-b89a-224880f73922
