### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils


# ╔═╡ d2de6a58-bec8-56b2-8689-bf65472c34ac
md"""
# Scatter in Water Cylinders

Before any scatter model can be trusted it has to reproduce what has been measured. This notebook
fires a CT fan beam through water cylinders from 20 to 50 cm in Geant4, every photon followed
through the patient, and scores every scattering on the detector through its anti-scatter grid by
forced detection (notebook 10 explains it), and asks four things:

1. **How does scatter grow with patient size**, with no grid, a 1-D grid and a 2-D grid?
2. **Does it match the literature**: the wide-cone measurements of Endo et al. on a Toshiba
   256-row prototype, the scatter fractions of Kyriakou and Kalender on a Siemens Definition, and
   the 2-D against 1-D grid reductions of Melnyk et al.?
3. **Is the grid model right**: the probability that a photon at an angle gets through a grid,
   against Geant4 transporting it through the tungsten walls?
4. **How far is BasisSimulator's current model** (one convolution scaled by one diameter) from all
   of this?

Every answer is a number with a target; the notebook ends with its scorecard.
"""

# ╔═╡ 1d3fc646-ad0c-5906-8543-a2bca385909f
md"""
## How Scatter Reaches the Detector

A photon from the focal spot either crosses the patient untouched (the **primary**, which carries
the image) or is deflected (Compton or Rayleigh) and, if it still leaves the patient, may reach the
detector somewhere else (the **scatter**). The anti-scatter grid's walls point at the focal spot: a
primary passes between them, a scattered photon arriving at an angle mostly hits one. The
**scatter-to-primary ratio** (SPR) at a detector element is the measure of how much the image is
contaminated.
"""

# ╔═╡ 7231e728-0bfd-57ce-81c6-d146e84839db
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1100, 640))
    ax = Mke.Axis(fig[1, 1]; aspect = Mke.DataAspect(), limits = ((-560, 560), (-60, 1150)))
    Mke.hidedecorations!(ax)
    Mke.hidespines!(ax)
    sid, sdd, R, fan = 600.0, 1072.0, 160.0, 50.0
    θ = range(0, 2π; length = 200)
    Mke.poly!(
        ax,
        [Mke.Point2f(R * cos(t), sid + R * sin(t)) for t in θ];
        color = (PALETTE.photon, 0.12),
        strokecolor = PALETTE.photon,
        strokewidth = 2,
    )
    for a in (-fan / 2, fan / 2)
        Mke.lines!(
            ax,
            [Mke.Point2f(0, 0), Mke.Point2f(sdd * sind(a), sdd * cosd(a))];
            color = (PALETTE.photon, 0.35),
        )
    end
    arc = [Mke.Point2f(sdd * sind(a), sdd * cosd(a)) for a in range(-fan / 2, fan / 2; length = 100)]
    Mke.lines!(ax, arc; color = PALETTE.kept, linewidth = 7)
    Mke.scatter!(ax, [Mke.Point2f(0, 0)]; color = PALETTE.photon, markersize = 16)
    Mke.lines!(
        ax,
        [Mke.Point2f(0, 0), Mke.Point2f(0, sdd)];
        color = PALETTE.photon,
        linewidth = 3,
    )
    Mke.lines!(
        ax,
        [Mke.Point2f(-60, sid - 40), Mke.Point2f(260, sdd * cosd(14))];
        color = PALETTE.xray,
        linewidth = 3,
        linestyle = :dash,
    )
    Mke.lines!(
        ax,
        [Mke.Point2f(0, 0), Mke.Point2f(-60, sid - 40)];
        color = PALETTE.photon,
        linewidth = 2,
    )
    note!(
        ax,
        0,
        0,
        "focal spot";
        align = (:center, :top),
        offset = (0, -12),
        color = PALETTE.photon,
    )
    note!(ax, 0, sid, "patient"; align = (:center, :center), color = PALETTE.photon)
    note!(
        ax,
        0,
        sdd,
        "primary";
        align = (:right, :bottom),
        offset = (-8, 10),
        color = PALETTE.photon,
    )
    note!(
        ax,
        260,
        sdd * cosd(14),
        "scattered photon";
        align = (:left, :bottom),
        offset = (8, 6),
        color = PALETTE.xray,
    )
    note!(
        ax,
        sdd * sind(-24),
        sdd * cosd(-24),
        "detector, behind its grid";
        align = (:right, :bottom),
        offset = (-10, 4),
        color = PALETTE.kept,
    )
    fig
end

# ╔═╡ 796d3afd-1bd9-5ac0-a24c-8cc7f5d665a4
md"""
## Notebook Setup
"""

# ╔═╡ dbb9e0a8-5a0b-53de-950e-359b14ac8932
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 81e8de79-0027-5593-ae75-cdd9e0b34298
using Markdown: @md_str, Markdown

# ╔═╡ 4cc4966f-57f8-5d35-afa1-185ee9047008
import PlutoUI

# ╔═╡ ea80c519-3b37-56fb-8022-024d4b2d9a00
import BasisMonteCarlo as MC

# ╔═╡ 3ae5a319-af60-5f17-a828-950bcf814985
import CairoMakie as Mke

# ╔═╡ c2b679e9-661e-5ff4-88da-05e87e76eb65
import BasisSimulator as BS

# ╔═╡ f1c2c0be-d0d4-5770-bc14-6a0152db3ed2
import Serialization

# ╔═╡ d0bc19dd-b31c-5812-bf12-e9f48c55fcdf
using Statistics: mean, median

# ╔═╡ be5b480e-57c1-5786-bd8f-21531b085217
PlutoUI.TableOfContents()

# ╔═╡ 05efa783-8a75-5a1e-89d3-4dacf6b592a0
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 519f33ab-549a-501a-8d7b-fa7943551970
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
    reference = Mke.RGBf(0.55, 0.2, 0.6),  # BasisSimulator's current table
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ f1fb99f9-4de4-5ebd-8f97-4c22159cd301
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

# ╔═╡ 5047a5c0-8771-5586-afb8-108e0931ea6a
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

# ╔═╡ 2e482794-9336-5a70-ba6b-cc3725b80d6d
# For scatter only the photons matter: electrons stop within a millimetre of where they are made
physics = MC.PhysicsSettings(; production_cut_um = 1000.0, electron_step_um = 1000.0)

# ╔═╡ d8f06a15-d98e-59f1-9c92-f480807db948
md"""
## 1. The Scanners and the Beam

Four geometries, each scored three ways where it matters (no grid, a 1-D grid, a 2-D grid), with the
detector efficiencies notebooks 04–08 saved: the SOMATOM Force (595 / 1085.6 mm), the Definition
Flash (the same distances), the Revolution Apex (625.6 / 1100 mm) and the Aquilion ONE
(600 / 1072 mm, the descendant of Endo's prototype). The grids are as built in those notebooks:
tungsten walls 10 mm tall (8 mm on the Apex), one cell per dexel. Detector bins are 1° across the
fan and 10 mm along z; the scatter-to-primary ratio is read in the central bin.

The beam: BasisSimulator's IPEM spectrum at 120 kV through 9.2 mm of aluminium (the Aquilion ONE
beam fitted to measured half-value layers in notebook 08), no bowtie. Neither Endo, Kyriakou nor
Melnyk states the full filtration; this beam's half-value layer is a typical clinical 8.5 mm.
"""

# ╔═╡ 13a73b8b-3056-5fc8-82f1-8e41bd653cd3
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ 427593d5-fe51-570c-bcde-6992223e3900
# The detectors' flat-field efficiencies, 1–140 keV, as notebooks 04–08 saved them
detector_efficiency = let
    load(file) = Serialization.deserialize(joinpath(table_dir, file))["efficiency"]
    extend(η) = vcat(η, fill(η[end], 10))
    Dict(
        "force" => extend(load("siemens_force_ufc_efficiency_geant4.jls")),
        "flash" => extend(load("siemens_flash_ufc_efficiency_geant4.jls")),
        "apex" => extend(load("ge_revolution_apex_gemstone_efficiency_geant4.jls")),
        "aquilion" => extend(load("canon_aquilion_one_genesis_efficiency_geant4.jls")),
    )
end

# ╔═╡ c8f766db-b02f-55e7-b642-e03de02ebc52
# Each scanner: distances, fan, detector half-coverage at the detector, the grid's walls and openings
geometries = Dict(
    "force" => (
        sid = 595.0,
        sdd = 1085.6,
        fan = 49.7,
        z_half = 105.0,
        height = 10.0,
        opening = (1.0236 - 0.08, 1.0947 - 0.08),
    ),
    "flash" => (
        sid = 595.0,
        sdd = 1085.6,
        fan = 50.0,
        z_half = 105.0,
        height = 10.0,
        opening = (1.2858 - 0.08, 1.0947 - 0.08),
    ),
    "apex" => (
        sid = 625.6,
        sdd = 1100.0,
        fan = 45.8,
        z_half = 290.0,
        height = 8.0,
        opening = (1.055 - 0.15, 1.099 - 0.15),
    ),
    "aquilion" => (
        sid = 600.0,
        sdd = 1072.0,
        fan = 49.2,
        z_half = 290.0,
        height = 10.0,
        opening = (1.0285 - 0.1, 0.893 - 0.1),
    ),
)

# ╔═╡ a5e50c51-c35c-59b1-8591-519e7110b2b1
"""One scanner's detector, with `grid` (`:none`, `:across` or `:both`)."""
function scanner_with(key, grid)
    g = geometries[key]
    return MC.ScatterScanner(;
        name = "$(key) $(grid)",
        source_to_isocenter_mm = g.sid,
        source_to_detector_mm = g.sdd,
        fan_deg = g.fan,
        z_half_mm = g.z_half,
        column_bin_deg = 1.0,
        row_bin_mm = 10.0,
        grid,
        grid_height_mm = g.height,
        grid_opening_mm = grid === :across ? (g.opening[1], Inf) : g.opening,
        efficiency = detector_efficiency[key],
    )
end

# ╔═╡ 7f733707-d2e5-5732-bbbd-7c874530f298
spectrum = let
    scanner = BS.EICTScanner(
        source_to_isocenter = 600.0,
        source_to_detector = 1072.0,
        detector_rows = 320,
        detector_cols = 896,
        detector_row_size = 0.5,
        detector_col_size = 0.5756,
        detector_shape = :arc,
        target_angle = 10.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 9.2,
        bowtie_filter = :large_body,
        detector_material = :ufc,
        detector_depth = 1.4,
    )
    protocol = BS.CTProtocol(;
        kVp = 120,
        mA = 200.0,
        views = 1200,
        rotation_time = 0.5,
        collimation_mm = 40.0,
        anode_angle = 10,
    )
    BS.resolve_source_spectrum_without_bowtie(BS.SimOptions(), protocol; scanner)
end

# ╔═╡ 8bb9f076-b85b-59ca-8451-b5f576b67b46
# Geant4's attenuation (1/mm), 1–150 keV, in all and by Compton and Rayleigh scattering: for the
# exact primary, and for forced detection's way out of the patient
attenuation = MC.forced_attenuation([MC.DetectorMaterial(:Water), MC.DetectorMaterial(:PMMA)])


# ╔═╡ 2dcf8255-0e2b-4cf7-849d-97a9eed8a6ad
geant4_mu = Dict(name => attenuation[MC.DetectorMaterial(name).name].total for name in (:Water, :PMMA))

# ╔═╡ 2d378e76-156a-5016-8397-79e007b0622e
"""
The primary per photon in the central detector bin of `s`, exactly: the bin's share of the fan's
photons, of the beam's rows, and Beer–Lambert through `length_mm` of material with attenuation `μ`.
"""
function primary_in_bin(s, length_mm, μ, collimation_mm)
    e, w = spectrum
    weights = w ./ sum(w)
    width = collimation_mm * s.source_to_detector_mm / s.source_to_isocenter_mm
    row_share = min(s.row_bin_mm, width) / width
    column_share = s.column_bin_deg / s.fan_deg
    detected = sum(
        wk * exp(-μ[clamp(round(Int, Ek), 1, 150)] * length_mm) * s.efficiency[clamp(round(Int, Ek), 1, 150)] * Ek
            for (Ek, wk) in zip(e, weights)
    )
    return column_share * row_share * detected
end

# ╔═╡ f037945b-bdc6-47e4-b64e-52c141e38505
"""
The exact primary per photon in column `ci` of one row bin wholly inside the beam, through a
cylinder of `diameter_mm` with attenuation `μ`: Beer–Lambert averaged over the column's angles (the
chord changes across a bin), times the bin's share of the fan's photons and of the beam's rows.
"""
function primary_in_column(s, ci, diameter_mm, μ, collimation_mm; steps = 50)
    e, w = spectrum
    weights = w ./ sum(w)
    cols = MC.column_edges(s)
    width = collimation_mm * s.source_to_detector_mm / s.source_to_isocenter_mm
    R = diameter_mm / 2
    detected(L) = sum(
        wk * exp(-μ[clamp(round(Int, Ek), 1, 150)] * L) * s.efficiency[clamp(round(Int, Ek), 1, 150)] * Ek
            for (Ek, wk) in zip(e, weights)
    )
    function at(γ)
        abs(γ) <= s.fan_deg / 2 || return 0.0
        b = s.source_to_isocenter_mm * sind(γ)
        return detected(abs(b) < R ? 2 * sqrt(R^2 - b^2) : 0.0)
    end
    Δ = cols[ci + 1] - cols[ci]
    mean_detected = sum(at(cols[ci] + (k - 0.5) * Δ / steps) for k in 1:steps) / steps
    return Δ / s.fan_deg * s.row_bin_mm / width * mean_detected
end

# ╔═╡ e6bdc677-04f8-53f0-98a3-0d8b213df331
"""
A fan beam of `collimation_mm` at the isocentre through a cylinder of `material` and `diameter_mm`,
every photon followed, its scatterings scored by forced detection on the detectors `scanners_` (all
of one geometry): 16 independent chunks of `n` photons; the merged tallies, one per scanner.
"""
function fan_scan(scanners_, diameter_mm, material, collimation_mm; n = 1_000_000, seed = 1)
    s = first(scanners_)
    phantom = MC.CylinderPhantom(
        diameter_mm;
        material,
        centre_mm = (0.0, s.source_to_isocenter_mm),
    )
    beam = MC.FocalSpot(;
        distance_mm = s.source_to_isocenter_mm,
        fan_deg = s.fan_deg,
        z_half_mm = collimation_mm / 2,
    )
    chunks = MC.simulate_sweep(
        [(phantom, MC.Spectrum(spectrum...), 1000 * seed + chunk, beam) for chunk in 1:16];
        n,
        physics,
        readout = MC.ForcedScatterReadout(scanners_; attenuation),
    )
    return MC.merge_tallies(chunks)
end

# ╔═╡ fe49b770-1d9e-5f1f-b1c0-1f4fa95764da
"""The scatter per photon in the central bin of a tally."""
central_scatter(t) = t.scatter[(size(t.scatter, 1) + 1) ÷ 2, (size(t.scatter, 2) + 1) ÷ 2] / t.n

# ╔═╡ ad0535ed-e20c-5d0f-b356-602e50d01d73
md"""
## 2. Scatter Against Patient Size

The SOMATOM Force's geometry, a 40 mm beam, water cylinders from 20 to 50 cm in 5 cm steps; the same
photons scored with no grid, with the Definition's kind of 1-D grid and with the Force's 2-D grid.
"""

# ╔═╡ 9a470377-5e0f-5d86-9c6b-120b4bff696e
size_diameters = collect(200.0:50.0:500.0)

# ╔═╡ 0fb143e2-13d2-5681-b1bb-31c0965d8216
force_variants = [scanner_with("force", grid) for grid in (:none, :across, :both)]

# ╔═╡ b35396c7-f7e6-5b79-9744-27766d9a96a0
size_scans = [fan_scan(force_variants, D, MC.DetectorMaterial(:Water), 40.0; seed = round(Int, D)) for D in size_diameters]

# ╔═╡ eee182c1-089c-5624-8d26-439b81c9bdd2
# SPR in the central bin, one row per diameter, one column per grid (none, 1-D, 2-D)
size_spr = [
    central_scatter(size_scans[i][k]) / primary_in_bin(force_variants[k], D, geant4_mu[:Water], 40.0)
        for (i, D) in enumerate(size_diameters), k in 1:3
]

# ╔═╡ ca7f3555-1bed-586a-b69f-e4ff6e8144b3
# The simulated primary against the exact one, 20–30 cm, pooled over the bins the beam covers whole:
# every row inside the 40 mm beam and the five central columns, each bin against its own exact
# primary. (One bin behind 30 cm of water receives about a hundred primary photons: a 9 % noise,
# too much to judge 5 % by.)
primary_check = [
    let t = size_scans[i][1], s = force_variants[1]
            rows = MC.row_edges(s)
            half = 40.0 * s.source_to_detector_mm / s.source_to_isocenter_mm / 2
            covered = [r for r in 1:(length(rows) - 1) if -half <= rows[r] && rows[r + 1] <= half]
            c = (size(t.primary, 1) + 1) ÷ 2
            columns = (c - 2):(c + 2)
            simulated = sum(t.primary[columns, covered]) / t.n
            exact = length(covered) * sum(primary_in_column(s, ci, D, geant4_mu[:Water], 40.0) for ci in columns)
            abs(simulated / exact - 1)
    end
        for (i, D) in enumerate(size_diameters) if D <= 300
]

# ╔═╡ 859c7739-942a-5cc1-a90a-064f8a3b1ae4
md"""
**BasisSimulator today.** Its model, `geometry_aware_scatter_model` with `estimate_scatter_field!`,
on the same cylinders: the projection of each cylinder through the same beam, the Force as
BasisSimulator defines it, 40 mm wide, and the model's own size scaling from the diameter. Its SPR
in the central bin is its scatter field, weighted by its energy model, over the primary.
"""

# ╔═╡ 8e6da242-e150-5b6f-9a89-0281bd4aa933
bs_force = BS.EICTScanner(
    source_to_isocenter = 595.0,
    source_to_detector = 1085.6,
    detector_rows = 64,
    detector_cols = 920,
    detector_row_size = 0.6,
    detector_col_size = 0.561,
    focal_spot_width = 0.8,
    focal_spot_length = 1.2,
    target_angle = 8.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 3.0,
    bowtie_filter = :large_body,
    detector_material = :ufc,
    detector_depth = 1.4,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = 0,
    detection_gain = 10.0,
)

# ╔═╡ 5323db56-e1e0-53d0-92ce-ab689486fc8f
"""
BasisSimulator's SPR at the central detector element, for a water cylinder of `diameter_mm` in
`scanner`'s beam: its scatter field from the polychromatic projection, its energy weighting, over
the primary.
"""
function bs_spr(scanner, diameter_mm)
    e, w = spectrum
    μ = geant4_mu[:Water]
    cols, rows = scanner.detector_cols, scanner.detector_rows
    Δγ = scanner.detector_col_size / scanner.source_to_isocenter
    R = diameter_mm / 2
    sinogram = zeros(Float64, cols, rows, 1)
    for i in 1:cols
        b = scanner.source_to_isocenter * sin((i - (cols + 1) / 2) * Δγ)
        L = abs(b) < R ? 2 * sqrt(R^2 - b^2) : 0.0
        transmitted = sum(wk * exp(-μ[clamp(round(Int, Ek), 1, 150)] * L) for (Ek, wk) in zip(e, w)) / sum(w)
        sinogram[i, :, 1] .= -log(transmitted)
    end
    model = BS.geometry_aware_scatter_model(scanner; phantom_diameter_cm = diameter_mm / 10)
    field = similar(sinogram)
    BS.estimate_scatter_field!(field, sinogram, model)
    weight = sum(w .* BS.compute_scatter_energy_weights(collect(Float64, e))) / sum(w)
    c, r = (cols + 1) ÷ 2, (rows + 1) ÷ 2
    return field[c, r, 1] * weight / exp(-sinogram[c, r, 1])
end

# ╔═╡ 555af2c9-409e-5c2b-9c34-90573793dd46
bs_size_spr = [bs_spr(bs_force, D) for D in size_diameters]

# ╔═╡ cbe8bd23-e221-50ef-af17-f43e94dfc1ef
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 600))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Scatter-to-primary ratio at the centre, SOMATOM Force geometry, 40 mm, 120 kV",
        xlabel = "water cylinder diameter (cm)",
        ylabel = "scatter-to-primary ratio",
        yscale = log10,
    )
    labels = ("Geant4, no grid", "Geant4, 1-D grid", "Geant4, 2-D grid (the Force's)")
    colors = (PALETTE.muted, PALETTE.xray, PALETTE.kept)
    for k in 1:3
        Mke.scatterlines!(
            ax,
            size_diameters ./ 10,
            size_spr[:, k];
            color = colors[k],
            markersize = 12,
            linewidth = 3,
            label = labels[k],
        )
    end
    Mke.scatterlines!(
        ax,
        size_diameters ./ 10,
        bs_size_spr;
        color = PALETTE.reference,
        markersize = 12,
        linewidth = 3,
        linestyle = :dash,
        label = "BasisSimulator today",
    )
    Mke.axislegend(ax; position = :lt)
    fig
end

# ╔═╡ e30f1805-d2fa-526c-bb96-afbbbfcdd749
let
    rows = [
        "| $(Int(D / 10)) cm | $(round(size_spr[i, 1]; digits = 3)) | $(round(size_spr[i, 2]; digits = 3)) | $(round(size_spr[i, 3]; digits = 3)) | $(round(bs_size_spr[i]; digits = 3)) |"
            for (i, D) in enumerate(size_diameters)
    ]
    Markdown.parse(
        """
        | water cylinder | no grid | 1-D grid | 2-D grid | BasisSimulator today |
        |---|---|---|---|---|
        $(join(rows, "\n"))

        Per centimetre of water the SPR with a grid grows by about $(round(100 * (exp(log(size_spr[end, 3] / size_spr[1, 3]) / 30) - 1); digits = 1)) % (2-D) and
        $(round(100 * (exp(log(size_spr[end, 2] / size_spr[1, 2]) / 30) - 1); digits = 1)) % (1-D): this is the rate a library over water-equivalent diameter must follow.
        """
    )
end

# ╔═╡ 08a0e736-c6a0-59ad-bfec-33a6e6f19434
md"""
## 3. Against the Literature

- **Endo et al. 2006** (Med. Phys. 33:3359): a Toshiba 256-row prototype, the Aquilion ONE's
  ancestor, with a 138 mm wide beam, measured an SPR of 49.7 % in a 20 cm water cylinder and 78.5 %
  in 32 cm, **without** its anti-scatter collimator. Simulated here in the Aquilion ONE's geometry,
  scored with no grid. **Shown as context, not as a check**: only the abstract is available to us,
  and its two numbers do not behave like a scatter-to-primary ratio at the centre of this beam (see
  below); until the full paper confirms the beam, the bowtie and how the SPR was measured, there is
  nothing fair to check against.
- **Kyriakou and Kalender 2007** (Phys. Med. Biol. 52:6969): on a Siemens SOMATOM Definition at
  19.2 mm, the scatter fraction `S / (S + P)` rose from 0.02 in an 8 cm PMMA cylinder to 0.11 in a
  40 cm one. Simulated in the Definition Flash's geometry with its 1-D grid. A clinical scanner
  shapes its beam with a bowtie filter, which this fan beam does not have; it matters for the large
  phantom, so its effect is taken from the scatter library of notebook 10 (below).
- **Melnyk et al. 2014**: at 160 mm coverage in 35 cm of water, 2-D grids cut the SPR by 59–75 %
  against a 1-D grid. Simulated in the Revolution Apex's geometry, scored with both grids.

None of the three states its full beam; the tolerances below are set for that.
"""

# ╔═╡ 3a16aba1-1864-5675-9a36-bd468a5c70b1
endo = let
    variants = [scanner_with("aquilion", :none)]
    [
        (
                diameter_mm = D,
                spr = central_scatter(only(fan_scan(variants, D, MC.DetectorMaterial(:Water), 138.0; seed = 3000 + round(Int, D)))) /
                primary_in_bin(only(variants), D, geant4_mu[:Water], 138.0),
                measured = measured,
            )
            for (D, measured) in ((200.0, 0.497), (320.0, 0.785))
    ]
end

# ╔═╡ 12c8dc45-23b7-571d-a6ef-afb2cc01f530
kyriakou = let
    variants = [scanner_with("flash", :across)]
    s = only(variants)
    map_sf(D) = let spr = central_scatter(only(fan_scan(variants, D, MC.DetectorMaterial(:PMMA), 19.2; seed = 4000 + round(Int, D)))) /
            primary_in_bin(s, D, geant4_mu[:PMMA], 19.2)
        spr / (1 + spr)
    end
    [(diameter_mm = D, fraction = map_sf(D)) for D in (80.0, 160.0, 240.0, 320.0, 400.0)]
end

# ╔═╡ 76692a1d-befe-45d6-acf2-72418d43cefb
# The scatter library of notebook 10, for the bowtie's effect on Kyriakou's large phantom
library = Serialization.deserialize(joinpath(table_dir, "scatter_library_geant4.jls"))

# ╔═╡ 0b011d62-d42a-4784-8356-b8df9cb4f78f
# BasisSimulator's large-body bowtie: its aluminium at each fan angle (the other materials are zero)
bowtie = let
    b = BS.resolve_bowtie_filter(:large_body)
    al = findfirst(==("Al"), b.materials)
    @assert all(iszero, b.thickness[:, setdiff(eachindex(b.materials), [al])])
    (angles_deg = rad2deg.(b.angles), aluminium_mm = 10 .* b.thickness[:, al])
end

# ╔═╡ 99cde5cd-b1d4-47b7-87e3-3a840ae7613c
"""
The scatter-to-primary ratio at the centre of `name`'s detector, from the library, for a water
cylinder of `diameter_mm` in a beam of `collimation_mm`; with `bowtie`, every ray's photons are
first filtered by the bowtie's aluminium at its angle (fewer, and harder, toward the edges).
"""
function library_central_spr(name, diameter_mm, collimation_mm; bowtie_on)
    s = library.scanners[findfirst(x -> x.name == name, library.scanners)]
    e, w = spectrum
    μ_al = [MC.linear_attenuation(MC.DetectorMaterial(:Al), Float64(E)) for E in 1:150]
    cols = MC.column_edges(s)
    R = diameter_mm / 2
    S, P = 0.0, 0.0
    for γ in (cols[1] + 0.05):0.1:cols[end]
        abs(γ) <= s.fan_deg / 2 || continue
        b = s.source_to_isocenter_mm * sind(γ)
        abs(b) < R || continue
        t = bowtie_on ? bowtie.aluminium_mm[argmin(abs.(bowtie.angles_deg .- γ))] : 0.0
        filtered = [wk * exp(-μ_al[clamp(round(Int, Ek), 1, 150)] * t) for (Ek, wk) in zip(e, w)]
        ray = (angle_deg = γ, wet_mm = 2 * sqrt(R^2 - b^2), weight = 0.1 / s.fan_deg * sum(filtered) / sum(w))
        scan = MC.library_scan(library, name, diameter_mm, [ray]; collimation_mm, spectrum = (e, filtered))
        c, r = (size(scan.scatter, 1) + 1) ÷ 2, (size(scan.scatter, 2) + 1) ÷ 2
        S += scan.scatter[c, r]
        P += scan.primary[c, r]
    end
    return S / P
end

# ╔═╡ 5d69e5b7-3f07-4cb4-9ef0-15583848d0c5
# The bowtie's effect on the SPR at the centre of the 40 cm PMMA phantom (as its water-equivalent
# cylinder), and how much aluminium it adds within the 8 cm phantom's rays
kyriakou_bowtie = let
    name = "SOMATOM Definition Flash"
    pmma, water = MC.DetectorMaterial(:PMMA), MC.DetectorMaterial(:Water)
    wed = 400.0 * MC.linear_attenuation(pmma, 70.0) / MC.linear_attenuation(water, 70.0)
    without = library_central_spr(name, wed, 19.2; bowtie_on = false)
    with = library_central_spr(name, wed, 19.2; bowtie_on = true)
    factor = with / without
    spr = kyriakou[end].fraction / (1 - kyriakou[end].fraction)
    small = abs.(bowtie.angles_deg) .<= asind(40.0 / 595.0)
    (
        wed_mm = wed,
        library_without = without / (1 + without),
        library_with = with / (1 + with),
        factor = factor,
        fraction = factor * spr / (1 + factor * spr),
        small_extra_aluminium_mm = maximum(bowtie.aluminium_mm[small]) - minimum(bowtie.aluminium_mm),
    )
end

# ╔═╡ 8332cbcd-5987-5d5b-becc-218c442f4c40
melnyk = let
    variants = [scanner_with("apex", :across), scanner_with("apex", :both)]
    tallies = fan_scan(variants, 350.0, MC.DetectorMaterial(:Water), 160.0; seed = 5000)
    spr = [central_scatter(tallies[k]) / primary_in_bin(variants[k], 350.0, geant4_mu[:Water], 160.0) for k in 1:2]
    (one_d = spr[1], two_d = spr[2], reduction = 1 - spr[2] / spr[1])
end

# ╔═╡ 5ee5818d-4cf2-59ae-a2d6-cf3c5ba78198
let
    e, w = spectrum
    μ = geant4_mu[:Water]
    primary(L) = sum(wk * Ek * exp(-μ[clamp(round(Int, Ek), 1, 150)] * L) for (Ek, wk) in zip(e, w))
    fall = primary(320.0) / primary(200.0)
    Markdown.parse(
        """
        | case | Geant4 | published |
        |---|---|---|
        | Endo, 20 cm water, 138 mm, no grid: SPR (context) | $(round(endo[1].spr; digits = 3)) | 0.497 |
        | Endo, 32 cm water, 138 mm, no grid: SPR (context) | $(round(endo[2].spr; digits = 3)) | 0.785 |
        | Kyriakou, 8 cm PMMA, 19.2 mm, 1-D grid: scatter fraction | $(round(kyriakou[1].fraction; digits = 3)) | 0.02 |
        | Kyriakou, 40 cm PMMA, 19.2 mm, 1-D grid: scatter fraction, no bowtie | $(round(kyriakou[end].fraction; digits = 3)) | 0.11 |
        | Kyriakou, 40 cm PMMA, 19.2 mm, 1-D grid: scatter fraction, with the bowtie | $(round(kyriakou_bowtie.fraction; digits = 3)) | 0.11 |
        | Melnyk, 35 cm water, 160 mm: 2-D grid's reduction of SPR against 1-D | $(round(100melnyk.reduction; digits = 1)) % | 59–75 % |

        **Why Endo is context, not a check.** From 20 to 32 cm of water the primary at the centre falls
        to $(round(fall; digits = 3)) of itself. Endo's SPR rises only $(round(0.785 / 0.497; digits = 2))-fold, so the scatter at the centre would have to
        fall $(round(1 / (0.785 / 0.497 * fall); digits = 1))-fold; here it falls $(round(1 / (endo[2].spr / endo[1].spr * fall); digits = 1))-fold, in line with the 1-D and 2-D grids and with
        the library. The published numbers are therefore not a central SPR of a bowtie-free beam like this
        one; what they are (beam, bowtie, how the primary was measured) needs the full paper.

        **Kyriakou's bowtie.** The library, which knows nothing of this run, gives the 40 cm phantom (as
        $(round(kyriakou_bowtie.wed_mm / 10; digits = 1)) cm of water) a scatter fraction of $(round(kyriakou_bowtie.library_without; digits = 3)) without a bowtie, close to the direct
        $(round(kyriakou[end].fraction; digits = 3)); BasisSimulator's large-body bowtie multiplies its central SPR by $(round(kyriakou_bowtie.factor; digits = 3)). The check
        applies that factor to the direct simulation. The 8 cm phantom's rays stay within the bowtie's
        flat centre (at most $(round(kyriakou_bowtie.small_extra_aluminium_mm; digits = 2)) mm more aluminium), so it needs none. The Definition's own bowtie
        and grid are not published; BasisSimulator's stand in for them.
        """
    )
end

# ╔═╡ 034fbda5-aa70-5474-a209-680587e3e823
md"""
## 4. The Grid Model Against Geant4

The scatter scoring lets a photon through a grid with probability `1 − H tan α / opening` per
direction of walls. Geant4 checks it: photons at an angle α across the fan land uniformly over one
cell of the Force's 2-D grid (10 mm tall, 80 µm of tungsten), on a 2 mm tungsten absorber below it;
the fraction absorbed, relative to straight-down, is the grid's transmission.
"""

# ╔═╡ 133e1015-067c-5090-8e85-194c0cb2b5f8
grid_check = let
    g = geometries["force"]
    pitch = g.opening .+ 0.08
    collimator = MC.Collimator(;
        height_mm = g.height,
        wall_mm = 0.08,
        pitch_x_mm = pitch[1],
        pitch_y_mm = pitch[2],
        walls = :both,
        cells = 13,
    )
    absorber = MC.SlabDetector(MC.DetectorMaterial(:W), 2.0; width_mm = 20.0, collimator)
    angles = collect(0.0:1.0:6.0)
    beams = [
        MC.AreaBeam(;
                width_mm = pitch[1],
                height_mm = pitch[2],
                angle_x_deg = a,
                start_mm = 12.0,
            )
            for a in angles
    ]
    runs = MC.simulate_sweep(
        [(absorber, 60.0, 7, beam) for beam in beams];
        n = 200_000,
        physics,
    )
    transmitted = [MC.interaction_probability(r) for r in runs]
    geant4 = transmitted ./ first(transmitted)
    model = [MC.grid_acceptance(deg2rad(a), g.height, g.opening[1]) for a in angles]
    (angles = angles, geant4 = geant4, model = model)
end

# ╔═╡ 3f4a1d7d-bea2-5b98-b2ad-86ab1cd40fc9
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1100, 480))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Getting through the Force's grid at an angle (60 keV)",
        xlabel = "angle across the fan (°)",
        ylabel = "transmission, relative to straight down",
        limits = (nothing, (0, 1.05)),
    )
    Mke.scatterlines!(
        ax,
        grid_check.angles,
        grid_check.geant4;
        color = PALETTE.ink,
        markersize = 12,
        linewidth = 3,
        label = "Geant4 through the tungsten walls",
    )
    Mke.lines!(
        ax,
        grid_check.angles,
        grid_check.model;
        color = PALETTE.kept,
        linewidth = 3,
        linestyle = :dash,
        label = "model: 1 − H tan α / opening",
    )
    Mke.axislegend(ax; position = :rt)
    fig
end

# ╔═╡ 5169e8a7-f563-5abe-b12c-71cf777b37c7
md"""
## 5. The Scorecard
"""

# ╔═╡ 9e34049b-1f60-5d97-b8c6-4287dba18678
checks = [
    MC.check_below(
        "primary: simulated against exact, 20–30 cm, the bins the beam covers whole (largest relative difference)",
        maximum(primary_check),
        0.05;
        source = "Beer–Lambert with Geant4's attenuation",
    ),
    MC.check_below(
        "grid model against Geant4's transport, 0–6° (largest absolute difference)",
        maximum(abs.(grid_check.geant4 .- grid_check.model)),
        0.05;
        source = "Geant4 through the tungsten walls",
    ),
    MC.check_range(
        "SPR grows with size, every grid (smallest ratio of successive diameters)",
        minimum(size_spr[2:end, :] ./ size_spr[1:(end - 1), :]),
        1.0,
        Inf;
        source = "physics",
    ),
    MC.check_relative(
        "Kyriakou: scatter fraction, 8 cm PMMA, 1-D grid",
        kyriakou[1].fraction,
        0.02,
        0.5;
        source = "Kyriakou & Kalender 2007",
    ),
    MC.check_relative(
        "Kyriakou: scatter fraction, 40 cm PMMA, 1-D grid, with BasisSimulator's bowtie (from the library)",
        kyriakou_bowtie.fraction,
        0.11,
        0.35;
        source = "Kyriakou & Kalender 2007",
    ),
    MC.check_range(
        "Melnyk: SPR reduction, 2-D against 1-D grid, 35 cm, 160 mm",
        melnyk.reduction,
        0.5,
        0.8;
        source = "Melnyk et al. 2014 (59–75 %)",
    ),
]

# ╔═╡ 19c010f9-8fac-5910-ba73-7f0ababc696e
Markdown.parse(MC.scorecard(checks))

# ╔═╡ 52df07a4-8d96-5285-9126-3005a235d6a7
md"""
## Saving the Results

One file in `lib/BasisMonteCarlo/tables/`: the SPR against diameter for the three grids and for
BasisSimulator's current model, the literature comparisons, the grid check and the scorecard.
"""

# ╔═╡ a79a77e5-a407-5b78-ac96-f9ceabb4144d
saved = let
    path = joinpath(table_dir, "scatter_water_cylinders_geant4.jls")
    Serialization.serialize(
        path,
        Dict{String, Any}(
            "diameters_mm" => size_diameters,
            "spr" => size_spr,
            "spr_columns" => ["no grid", "1-D grid", "2-D grid"],
            "spr_basissimulator" => bs_size_spr,
            "endo" => endo,
            "kyriakou" => kyriakou,
            "kyriakou_bowtie" => kyriakou_bowtie,
            "melnyk" => melnyk,
            "grid_check" => grid_check,
            "scorecard" => MC.scorecard(checks),
            "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 09_scatter_water_cylinders.jl",
        ),
    )
    path
end

# ╔═╡ Cell order:
# ╟─d2de6a58-bec8-56b2-8689-bf65472c34ac
# ╟─1d3fc646-ad0c-5906-8543-a2bca385909f
# ╟─7231e728-0bfd-57ce-81c6-d146e84839db
# ╟─796d3afd-1bd9-5ac0-a24c-8cc7f5d665a4
# ╠═dbb9e0a8-5a0b-53de-950e-359b14ac8932
# ╠═81e8de79-0027-5593-ae75-cdd9e0b34298
# ╠═4cc4966f-57f8-5d35-afa1-185ee9047008
# ╠═ea80c519-3b37-56fb-8022-024d4b2d9a00
# ╠═3ae5a319-af60-5f17-a828-950bcf814985
# ╠═c2b679e9-661e-5ff4-88da-05e87e76eb65
# ╠═f1c2c0be-d0d4-5770-bc14-6a0152db3ed2
# ╠═d0bc19dd-b31c-5812-bf12-e9f48c55fcdf
# ╠═be5b480e-57c1-5786-bd8f-21531b085217
# ╠═05efa783-8a75-5a1e-89d3-4dacf6b592a0
# ╠═519f33ab-549a-501a-8d7b-fa7943551970
# ╠═f1fb99f9-4de4-5ebd-8f97-4c22159cd301
# ╠═5047a5c0-8771-5586-afb8-108e0931ea6a
# ╠═2e482794-9336-5a70-ba6b-cc3725b80d6d
# ╟─d8f06a15-d98e-59f1-9c92-f480807db948
# ╠═13a73b8b-3056-5fc8-82f1-8e41bd653cd3
# ╠═427593d5-fe51-570c-bcde-6992223e3900
# ╠═c8f766db-b02f-55e7-b642-e03de02ebc52
# ╠═a5e50c51-c35c-59b1-8591-519e7110b2b1
# ╠═7f733707-d2e5-5732-bbbd-7c874530f298
# ╠═8bb9f076-b85b-59ca-8451-b5f576b67b46
# ╠═2dcf8255-0e2b-4cf7-849d-97a9eed8a6ad
# ╠═2d378e76-156a-5016-8397-79e007b0622e
# ╠═f037945b-bdc6-47e4-b64e-52c141e38505
# ╠═e6bdc677-04f8-53f0-98a3-0d8b213df331
# ╠═fe49b770-1d9e-5f1f-b1c0-1f4fa95764da
# ╟─ad0535ed-e20c-5d0f-b356-602e50d01d73
# ╠═9a470377-5e0f-5d86-9c6b-120b4bff696e
# ╠═0fb143e2-13d2-5681-b1bb-31c0965d8216
# ╠═b35396c7-f7e6-5b79-9744-27766d9a96a0
# ╠═eee182c1-089c-5624-8d26-439b81c9bdd2
# ╠═ca7f3555-1bed-586a-b69f-e4ff6e8144b3
# ╟─859c7739-942a-5cc1-a90a-064f8a3b1ae4
# ╠═8e6da242-e150-5b6f-9a89-0281bd4aa933
# ╠═5323db56-e1e0-53d0-92ce-ab689486fc8f
# ╠═555af2c9-409e-5c2b-9c34-90573793dd46
# ╟─cbe8bd23-e221-50ef-af17-f43e94dfc1ef
# ╟─e30f1805-d2fa-526c-bb96-afbbbfcdd749
# ╟─08a0e736-c6a0-59ad-bfec-33a6e6f19434
# ╠═3a16aba1-1864-5675-9a36-bd468a5c70b1
# ╠═12c8dc45-23b7-571d-a6ef-afb2cc01f530
# ╠═76692a1d-befe-45d6-acf2-72418d43cefb
# ╠═0b011d62-d42a-4784-8356-b8df9cb4f78f
# ╠═99cde5cd-b1d4-47b7-87e3-3a840ae7613c
# ╠═5d69e5b7-3f07-4cb4-9ef0-15583848d0c5
# ╠═8332cbcd-5987-5d5b-becc-218c442f4c40
# ╟─5ee5818d-4cf2-59ae-a2d6-cf3c5ba78198
# ╟─034fbda5-aa70-5474-a209-680587e3e823
# ╠═133e1015-067c-5090-8e85-194c0cb2b5f8
# ╟─3f4a1d7d-bea2-5b98-b2ad-86ab1cd40fc9
# ╟─5169e8a7-f563-5abe-b12c-71cf777b37c7
# ╠═9e34049b-1f60-5d97-b8c6-4287dba18678
# ╟─19c010f9-8fac-5910-ba73-7f0ababc696e
# ╟─52df07a4-8d96-5285-9126-3005a235d6a7
# ╠═a79a77e5-a407-5b78-ac96-f9ceabb4144d
