### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils


# ╔═╡ be56f13c-5abb-5ce4-9011-3964bb0223d3
md"""
# A Scatter Library over Water-Equivalent Diameter

Scatter is the x rays a patient deflects onto the detector: it adds a low, smooth signal under
every projection and, uncorrected, cups and streaks the image. BasisSimulator models it today with
one analytic convolution, a single 50 mm Gaussian scaled by a single patient diameter, with no
anti-scatter grid, no collimation width and no scatter spectrum.

This notebook builds what replaces it: a library of **pencil-beam scatter kernels**, simulated in
Geant4, for water cylinders of every water-equivalent diameter from **20 to 50 cm in 1 cm steps**,
for every ray through them and every photon energy, scored at once on the detectors of six
scanners (with their own grids and efficiencies, from notebooks 03–08). DukeSim reruns its Monte
Carlo for every patient; a library computed once serves any patient in this size range in seconds,
and keeps the energy of every scattered photon, which a photon-counting detector needs.

Every claim here is checked: the notebook ends with a scorecard of quantitative pass/fail checks
(the library's own noise, symmetry, forced detection against counting photon by photon, held-out
diameters, and the library against direct Monte Carlo of whole fan beams), saved with the library.
"""

# ╔═╡ 9e90d4ca-5f06-5847-bb8d-aabdf1f2c9b2
md"""
## How One Kernel Serves Every Scanner

A ray through a cylinder crosses it along the same path whatever direction it comes from: only its
distance from the axis matters, the **impact** `b`. So the Monte Carlo fires one ray at impact `b`
through a water cylinder and logs every photon that leaves the cylinder, where and in which
direction. Outside the patient a photon travels in a straight line, so for each scanner the focal
spot is put back on the ray's line, at that scanner's source-to-isocentre distance, and every
photon is followed to that scanner's detector arc, through its anti-scatter grid, into its
detector. One simulation, six scanners.
"""


# ╔═╡ e0e565df-c3ed-4b6d-b27f-66226d58c2ce
md"""
## Counting Scatter by Forced Detection

Counting only the photons that happen to land on a detector wastes nearly all of them: of a million
photons along one ray through 45 cm of water, about fifty end up detected as scatter behind a 2-D
anti-scatter grid. So the library scores every Compton and Rayleigh scattering Geant4 makes with the
*chance* that the photon's next flight reaches each detector: the scattering's angular distribution
(from Geant4's own scattering functions and form factors for hydrogen and oxygen), the attenuation
of the water on the way out (exact, in a cylinder), the grid and the detector. Geant4 still follows
every photon to its end, so all the multiple scattering is there. The first scattering of each
photon along the ray is not even sampled: its chance per millimetre of depth is known, so it is
integrated over the ray.

Rayleigh scattering's narrow forward cone is found by also drawing directions from the cone itself,
combined with points spread over the detector by multiple importance sampling; scatterings that
can add little (deep, with every way out long) are played by Russian roulette. None of this biases the result, and it
is checked: every kernel also carries the photon-by-photon count of the same scatter, and the two
are compared over the whole library in the scorecard. Section 2 shows one kernel counted both ways.
"""

# ╔═╡ 260664e3-1968-5c97-8dfd-017ae8516352
md"""
## Notebook Setup
"""

# ╔═╡ 47fa26ea-721e-5b66-922a-52973af46f34
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 57bc93bd-06f3-5fd9-ac5e-fb4bf1cd0552
using Markdown: @md_str, Markdown

# ╔═╡ 0abf2ee9-00ed-56be-a6d2-aa80e0ab5ff3
import PlutoUI

# ╔═╡ af0399b5-424a-5185-8615-b98ab53fd637
import BasisMonteCarlo as MC

# ╔═╡ 92592abd-21b6-52ee-b4ea-fc480e994c21
import CairoMakie as Mke

# ╔═╡ 25ba20a9-568c-5267-bb9b-2fe80bd08476
import BasisSimulator as BS

# ╔═╡ dc10d567-286d-577d-ae4e-dfb80d3c8f3b
import Serialization

# ╔═╡ 1b94b105-16a2-52ed-8367-157464da5eea
using Statistics: mean, median, std

# ╔═╡ 9c017251-6dcc-5d5e-bf01-07818d561ec5
PlutoUI.TableOfContents()

# ╔═╡ 927f4d8a-337c-5169-93dc-b740520aaa90
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ fa7d87e4-de51-5da1-8211-9182959161dc
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

# ╔═╡ da800d9e-8bc7-59c6-ad19-e5365d75c7cc
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

# ╔═╡ 115c241b-2b41-5a68-8983-2135a55af1a8
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

# ╔═╡ 7fbd21af-2717-503b-af34-b57421e69d09
# For scatter only the photons matter: electrons stop within a millimetre of where they are made,
# so they are not followed further than that, which makes a patient several times quicker to cross
physics = MC.PhysicsSettings(; production_cut_um = 1000.0, electron_step_um = 1000.0)

# ╔═╡ 0f82726d-b35c-5dae-a6e1-ba68689f076d
md"""
## 1. The Six Detectors

Each scanner as notebooks 03–08 built it: its source-to-isocentre and source-to-detector distances,
fan, coverage along z, anti-scatter grid (2-D on the Force, NAEOTOM Alpha, Revolution Apex and
Brilliance iCT; 1-D on the Definition Flash and Aquilion ONE), and its detector's efficiency at
every energy, read from the tables those notebooks saved. A photon arriving at an angle to the
grid's focused walls gets through with probability `1 − H tan α / opening` in each direction the
grid has walls; a photon crossing the crystal obliquely has a longer path in it. The tally's bins
are 2° across the fan and 10–20 mm along z, covering the detector's full width on either side of a
ray.
"""

# ╔═╡ 7848689c-0a69-53f7-b7fd-88f3413a4176
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ bdcc54a3-c0f5-5800-8c85-896244e8d07d
# The energy-integrating detectors' flat-field efficiencies, 1–140 keV, as notebooks 04–08 saved them
detector_efficiency = let
    load(file) = Serialization.deserialize(joinpath(table_dir, file))["efficiency"]
    Dict(
        "force" => load("siemens_force_ufc_efficiency_geant4.jls"),
        "flash" => load("siemens_flash_ufc_efficiency_geant4.jls"),
        "apex" => load("ge_revolution_apex_gemstone_efficiency_geant4.jls"),
        "ict" => load("philips_brilliance_ict256_efficiency_geant4.jls"),
        "aquilion" => load("canon_aquilion_one_genesis_efficiency_geant4.jls"),
    )
end

# ╔═╡ 7d685d9d-aaed-5ffe-9b04-08a35096c634
scanners = let
    mu(material) = [MC.linear_attenuation(material, Float64(E)) for E in 1:150]
    extend(η) = vcat(η, fill(η[end], 150 - length(η)))
    gos = MC.DetectorMaterial(:UFC)
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
    cdte = MC.DetectorMaterial(:CdTe)
    cdte_counting = [1 - exp(-μ * 1.6) for μ in mu(cdte)]
    [
        MC.ScatterScanner(;
            name = "SOMATOM Force",
            source_to_isocenter_mm = 595.0,
            source_to_detector_mm = 1085.6,
            fan_deg = 49.7,
            z_half_mm = 105.0,
            row_bin_mm = 10.0,
            grid = :both,
            grid_height_mm = 10.0,
            grid_opening_mm = (1.0236 - 0.08, 1.0947 - 0.08),
            efficiency = extend(detector_efficiency["force"]),
            mu_per_mm = mu(gos),
            depth_mm = 1.4,
        ),
        MC.ScatterScanner(;
            name = "SOMATOM Definition Flash",
            source_to_isocenter_mm = 595.0,
            source_to_detector_mm = 1085.6,
            fan_deg = 50.0,
            z_half_mm = 70.0,
            row_bin_mm = 10.0,
            grid = :across,
            grid_height_mm = 10.0,
            grid_opening_mm = (1.2858 - 0.08, Inf),
            efficiency = extend(detector_efficiency["flash"]),
            mu_per_mm = mu(gos),
            depth_mm = 1.4,
        ),
        MC.ScatterScanner(;
            name = "NAEOTOM Alpha",
            source_to_isocenter_mm = 610.0,
            source_to_detector_mm = 1113.0,
            fan_deg = 48.4,
            z_half_mm = 105.0,
            row_bin_mm = 10.0,
            grid = :both,
            grid_height_mm = 10.0,
            grid_opening_mm = (4 * 0.275, 6 * 0.322),
            efficiency = cdte_counting,
            mu_per_mm = mu(cdte),
            depth_mm = 1.6,
            energy_weighted = false,
        ),
        MC.ScatterScanner(;
            name = "Revolution Apex",
            source_to_isocenter_mm = 625.6,
            source_to_detector_mm = 1100.0,
            fan_deg = 45.8,
            z_half_mm = 281.0,
            row_bin_mm = 20.0,
            grid = :both,
            grid_height_mm = 8.0,
            grid_opening_mm = (1.055 - 0.15, 1.099 - 0.15),
            efficiency = extend(detector_efficiency["apex"]),
            mu_per_mm = mu(gemstone),
            depth_mm = 3.0,
        ),
        MC.ScatterScanner(;
            name = "Brilliance iCT",
            source_to_isocenter_mm = 570.0,
            source_to_detector_mm = 1040.0,
            fan_deg = 52.1,
            z_half_mm = 146.0,
            row_bin_mm = 15.0,
            grid = :both,
            grid_height_mm = 10.0,
            grid_opening_mm = (1.408 - 0.1, 1.14 - 0.1),
            efficiency = extend(detector_efficiency["ict"]),
            mu_per_mm = mu(gos),
            depth_mm = 1.4,
        ),
        MC.ScatterScanner(;
            name = "Aquilion ONE",
            source_to_isocenter_mm = 600.0,
            source_to_detector_mm = 1072.0,
            fan_deg = 49.2,
            z_half_mm = 286.0,
            row_bin_mm = 20.0,
            grid = :across,
            grid_height_mm = 10.0,
            grid_opening_mm = (1.0285 - 0.1, Inf),
            efficiency = extend(detector_efficiency["aquilion"]),
            mu_per_mm = mu(gos),
            depth_mm = 1.4,
        ),
    ]
end

# ╔═╡ 1db19aed-4a60-54a1-913e-398750a7997e
# Forced detection on all six detectors, the first scatterings integrated along each ray
readout = MC.ForcedScatterReadout(
    scanners;
    water_mu_per_mm = water_mu,
    water_compton_per_mm = water_compton,
    water_rayleigh_per_mm = water_rayleigh,
)

# ╔═╡ e4a18eea-cb39-5575-9230-e0ff552875a3
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1300, 760))
    ax = Mke.Axis(
        fig[1, 1];
        title = "One ray through a 30 cm cylinder, at half its radius, seen by two scanners",
        xlabel = "x (mm)",
        ylabel = "y (mm)",
        aspect = Mke.DataAspect(),
        limits = ((-700, 900), (-680, 560)),
    )
    R, u = 150.0, 0.5
    b = u * R
    θ = range(0, 2π; length = 200)
    Mke.poly!(
        ax,
        [Mke.Point2f(R * cos(t), R * sin(t)) for t in θ];
        color = (PALETTE.photon, 0.12),
        strokecolor = PALETTE.photon,
        strokewidth = 2,
    )
    note!(
        ax,
        0,
        0,
        "water";
        align = (:center, :center),
        color = PALETTE.photon,
        fontsize = 15,
    )
    for (s, color, side) in ((scanners[1], PALETTE.reference, -1), (scanners[5], PALETTE.kept, 1))
        L = sqrt(s.source_to_isocenter_mm^2 - b^2)
        f = (b, -L)
        c = (-f[1], -f[2]) ./ s.source_to_isocenter_mm
        φ0 = atan(c[2], c[1])
        half = deg2rad(s.fan_deg / 2)
        arc = [
            Mke.Point2f(
                    f[1] + s.source_to_detector_mm * cos(φ0 + a),
                    f[2] + s.source_to_detector_mm * sin(φ0 + a),
                )
                for a in range(-half, half; length = 100)
        ]
        Mke.lines!(ax, arc; color, linewidth = 5)
        Mke.scatter!(ax, [Mke.Point2f(f...)]; color, markersize = 14)
        note!(
            ax,
            f[1],
            f[2],
            "focal spot, $(s.name)";
            align = side < 0 ? (:right, :center) : (:left, :center),
            offset = (side * 14, 0),
            color,
            fontsize = 14,
        )
        note!(
            ax,
            last(arc)[1],
            last(arc)[2],
            "its detector";
            align = (:left, :bottom),
            offset = (6, 4),
            color,
            fontsize = 14,
        )
    end
    Mke.lines!(
        ax,
        [Mke.Point2f(b, -640), Mke.Point2f(b, 540)];
        color = PALETTE.photon,
        linewidth = 3,
    )
    Mke.lines!(
        ax,
        [Mke.Point2f(b, -60), Mke.Point2f(b + 330, 470)];
        color = PALETTE.xray,
        linewidth = 2.5,
        linestyle = :dash,
    )
    note!(
        ax,
        b + 170,
        200,
        "a scattered photon";
        align = (:left, :center),
        offset = (10, 0),
        color = PALETTE.xray,
        fontsize = 14,
    )
    note!(
        ax,
        b,
        -400,
        "the ray, impact b";
        align = (:left, :center),
        offset = (10, 0),
        color = PALETTE.photon,
        fontsize = 14,
    )
    fig
end

# ╔═╡ 49b6c40d-5008-54e4-bee4-c49738a1f800
md"""
## 2. The Kernels

Water cylinders from 20 to 50 cm in 1 cm steps; rays at 21 impacts from the axis to the edge
(0 to 0.95 of the radius in steps of 0.05, and 0.99); photons of 20 to 140 keV in 10 keV steps:
8463 kernels. A larger patient lets less through, so the photons per kernel grow with the square of
its diameter: about 160,000 at 20 cm, 500,000 at 35 cm and a million at 50 cm (about four hours on
eight cores). Each kernel is simulated in two independent halves with their own seeds: the library
is their sum, and the difference between the halves measures the library's noise (section 3). Any
one kernel can be rerun alone, and more halves added, without changing the rest.
"""


# ╔═╡ aa74500e-69bb-45dc-9b84-b97d02d12150
md"""
**One kernel, counted both ways.** A ray at 0.3 of the radius through 35 cm of water, 70 keV, 200,000
photons: the scatter on two detectors across the fan (summed along z), counted photon by photon and
by forced detection, from the very same photons.
"""

# ╔═╡ a285f955-c394-41f8-a94b-65861e8a9bee
demo = only(
    MC.simulate_sweep(
        [(MC.CylinderPhantom(350.0), 70.0, 42, MC.kernel_ray(350.0, 0.3))];
        n = 200_000,
        physics,
        readout,
    ),
)

# ╔═╡ ecd65b8c-d0f6-4502-80f0-7c417e498baa
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 560))
    for (j, name) in enumerate(("SOMATOM Force", "Aquilion ONE"))
        t = demo[findfirst(x -> x.scanner == name, demo)]
        centres = (t.column_edges_deg[1:(end - 1)] .+ t.column_edges_deg[2:end]) ./ 2
        ax = Mke.Axis(
            fig[1, j];
            title = "$(name), $(j == 1 ? "2-D" : "1-D") grid",
            xlabel = "angle across the fan (°)",
            ylabel = j == 1 ? "scatter per photon (keV)" : "",
        )
        Mke.stairs!(
            ax,
            centres,
            vec(sum(t.analog_scatter; dims = 2)) ./ t.n;
            step = :center,
            color = PALETTE.muted,
            linewidth = 2,
            label = "photon by photon",
        )
        Mke.stairs!(
            ax,
            centres,
            vec(sum(t.scatter; dims = 2)) ./ t.n;
            step = :center,
            color = PALETTE.kept,
            linewidth = 4,
            label = "forced detection",
        )
        j == 1 && Mke.axislegend(ax; position = :lt, framevisible = false)
    end
    Mke.Label(
        fig[0, :],
        "One ray through 35 cm of water, 200,000 photons, scored two ways";
        fontsize = 22,
        font = :bold,
    )
    fig
end
# ╔═╡ 420ca537-a700-5d72-8901-ab9c5fdcb42f
diameters_mm = collect(200.0:10.0:500.0)

# ╔═╡ 0191eb88-a4aa-593b-8d18-32f5fab725cb
impact_fractions = vcat(collect(0.0:0.05:0.95), 0.99)

# ╔═╡ 1ca0b7d9-8065-5e30-9fad-7e035e483d95
energies_keV = collect(20.0:10.0:140.0)

# ╔═╡ 490c427d-a22a-5ea9-87af-256758fd46cf
# Photons per kernel in each of its two halves: more for a larger patient, which lets less through
photons_per_half(D) = 10_000 * cld(round(Int, 250_000 * (D / 350)^2), 10_000)

# ╔═╡ 9fd96be0-2511-59f0-873c-440d74415297
# Each kernel half's own seed, from its cylinder, ray and half
kernel_seed(D, u, half = 1) = 100_000 + round(Int, 1000 * D + 100 * u) + 1_000_000 * (half - 1)


# ╔═╡ df41e785-c04b-4d8a-b2f1-ab5d1e49a130
# Every kernel, in the order the library stores them
kernel_grid = [(D, u, E) for D in diameters_mm for u in impact_fractions for E in energies_keV]
# ╔═╡ 254033b1-90b6-5df7-8582-b5aba5fe9e46
# Every kernel in two independent halves (one after the other)
kernel_halves = MC.simulate_sweep(
    [
        (MC.CylinderPhantom(D), E, kernel_seed(D, u, half), MC.kernel_ray(D, u), photons_per_half(D))
            for (D, u, E) in kernel_grid for half in 1:2
    ];
    physics,
    readout,
)


# ╔═╡ 673d1dfc-4b31-4673-a44e-e5255186ea3b
# Each kernel: its two halves added
kernel_tallies = [MC.merge_tallies(kernel_halves[(2j - 1):(2j)]) for j in eachindex(kernel_grid)]
# ╔═╡ ac791f75-a62f-424a-a893-c12e5492ed7e
# Geant4's attenuation of water at every keV, in all and by Compton and Rayleigh scattering: the
# primary of any ray is then Beer–Lambert, exact, and so is forced detection's way out of the patient
water_mu, water_compton, water_rayleigh = [
    [MC.geant4_attenuation(MC.DetectorMaterial(:Water), Float64(E); kind) for E in 1:150]
        for kind in (:total, :compton, :rayleigh)
]

# ╔═╡ b680104b-e544-5afd-94b3-e6a92fc42960
library = MC.scatter_library(
    kernel_tallies,
    scanners,
    diameters_mm,
    impact_fractions,
    energies_keV;
    water_mu_per_mm = water_mu,
    metadata = Dict{String, Any}(
        "photons_per_kernel" => Dict(D => 2 * photons_per_half(D) for D in diameters_mm),
        "physics" => string(physics),
        "scoring" => "forced detection, first scattering integrated along the ray",
        "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 10_scatter_library.jl",
    ),
)


# ╔═╡ c36e2624-afa7-44a0-a0b2-bbb1a66211a1
# The two halves as libraries of their own, to measure the library's noise
library_halves = [
    MC.scatter_library(
            kernel_halves[half:2:end],
            scanners,
            diameters_mm,
            impact_fractions,
            energies_keV;
            water_mu_per_mm = water_mu,
        )
        for half in 1:2
]
# ╔═╡ 77f080ef-da2b-5ce5-81a0-0ee4ae8f5310
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 620))
    for (k, (name, u)) in enumerate((("SOMATOM Force", 0.0), ("SOMATOM Force", 0.8), ("Aquilion ONE", 0.8)))
        s = scanners[findfirst(x -> x.name == name, scanners)]
        kernel, _ = MC.library_kernel(library, name, 300.0, u, 70.0)
        cols, rows = MC.column_edges(s), MC.row_edges(s)
        ax = Mke.Axis(
            fig[1, k];
            title = "$(name), ray at $(u) R",
            xlabel = "angle across the fan (°)",
            ylabel = k == 1 ? "z at the detector (mm)" : "",
        )
        hm = Mke.heatmap!(
            ax,
            (cols[1:(end - 1)] .+ cols[2:end]) ./ 2,
            (rows[1:(end - 1)] .+ rows[2:end]) ./ 2,
            1000 .* kernel;
            colormap = :viridis,
        )
        Mke.Colorbar(
            fig[2, k],
            hm;
            vertical = false,
            label = "scatter signal per photon (eV)",
            flipaxis = false,
        )
    end
    Mke.Label(
        fig[0, :],
        "Scatter from one 70 keV ray through 30 cm of water, on the detector";
        fontsize = 22,
        font = :bold,
    )
    fig
end

# ╔═╡ 687935da-9ec8-55cf-8491-b95da499e2ba
md"""
How the scatter one ray sends to each detector grows with the patient: through the axis (impact 0)
of cylinders from 20 to 50 cm, at 70 keV. The 2-D grids (Force, Alpha, Apex, iCT) let through a
fraction of what the 1-D grids (Flash, Aquilion ONE) do.
"""

# ╔═╡ 2108da2a-f36e-5b98-944b-bd92b09fd20b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 560))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Scatter from one 70 keV ray through the axis, by cylinder diameter",
        xlabel = "water-equivalent diameter (cm)",
        ylabel = "scatter signal reaching the detector,\nper photon (fraction of 70 keV)",
        yscale = log10,
    )
    colors = (
        PALETTE.reference,
        PALETTE.muted,
        PALETTE.electron,
        PALETTE.xray,
        PALETTE.kept,
        PALETTE.photon,
    )
    for (k, s) in enumerate(scanners)
        total = [sum(MC.library_kernel(library, s.name, D, 0.0, 70.0)[1]) for D in diameters_mm]
        per = s.energy_weighted ? total ./ 70.0 : total
        Mke.scatterlines!(
            ax,
            diameters_mm ./ 10,
            per;
            color = colors[k],
            markersize = 7,
            label = s.name,
        )
    end
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 12c91dff-71b8-54ef-b649-c9e389568616
md"""
## 3. The Checks

Five kinds, each a number with a target:

- **The library's noise.** Each kernel's two halves are independent estimates; half their difference
  is the noise of their sum. It is measured where the library is used: on single rays of a 120 kV
  beam (each ray's kernels over the beam's energies) and on whole scans predicted from each half of
  the library.
- **Symmetry.** A ray through the axis scatters equally to both sides of the fan: summed over
  every diameter and energy, the two sides must agree to well within what noise allows.
- **Forced detection against counting.** Over every kernel, the scatter by forced detection and the
  scatter counted photon by photon, from the same photons, must agree.
- **Held-out diameters and energies.** Kernels simulated at diameters and an energy *between* the
  library's grid points (23.5, 31.5 and 44.5 cm; 75 keV), each with eight times the library's
  photons, must match the library's interpolation: this is what decides whether 1 cm steps in
  water-equivalent diameter are fine enough.
- **Whole scans.** A fan beam from a 120 kV tube through cylinders of 27.5 and 43.5 cm (between
  grid points) is simulated directly, every photon, and its scatter on the Force's and the
  Aquilion ONE's detectors compared with the library's prediction from its pencil kernels: at the
  centre, across the fan and along z.
"""

# ╔═╡ 494c2134-fd68-51de-bef2-eea2ca1d5b91
# Rays at three sizes: through the axis, halfway out and near the edge
noise_cases = [(D, u) for D in (200.0, 350.0, 500.0) for u in (0.0, 0.5, 0.9)]

# ╔═╡ b3cf4ace-0f65-5679-b861-da072652ac64
"""Relative difference of two kernels: of their totals, and rms over the bins above 10 % of the peak."""
function kernel_difference(a, b)
    total = abs(sum(a) / sum(b) - 1)
    strong = findall(>(0.1 * maximum(b)), b)
    bins = sqrt(sum(abs2, a[strong] ./ b[strong] .- 1) / length(strong))
    return (total = total, bins = bins)
end


# ╔═╡ 0bdb1f8f-17d8-45ac-a652-e509f89d422b
"""The kernel of one ray of the 120 kV beam: each energy's kernel, weighted by the beam's photons."""
function beam_kernel(lib, name, D, u)
    energies, weights = scan_spectrum
    kernel = zero(MC.library_kernel(lib, name, D, u, 70.0)[1])
    for (E, w) in zip(energies, weights)
        w > 0 || continue
        kernel .+= w / sum(weights) .* MC.library_kernel(lib, name, D, u, E)[1]
    end
    return kernel
end
# ╔═╡ f32a0349-3698-55f3-ab6f-966791eea928
# The noise of each ray's kernel: half the difference of the library's two halves
noise_differences = [
    let a = beam_kernel(library_halves[1], s.name, case...),
            b = beam_kernel(library_halves[2], s.name, case...),
            d = kernel_difference(a, b)
            (total = d.total / 2, bins = d.bins / 2)
    end
        for case in noise_cases for s in scanners
]


# ╔═╡ ee197700-86f5-4eea-8631-389819370c97
# The same at single energies, through the axis of the largest patients
energy_noise = [
    (
            scanner = s.name,
            D = D,
            E = E,
            total = kernel_difference(
                MC.library_kernel(library_halves[1], s.name, D, 0.0, E)[1],
                MC.library_kernel(library_halves[2], s.name, D, 0.0, E)[1],
            ).total / 2,
        )
        for D in (400.0, 450.0, 500.0) for E in (50.0, 70.0, 90.0) for s in scanners
]

# ╔═╡ f09afb2f-b189-44ff-b632-9b230b731275
let
    worst = energy_noise[argmax([d.total for d in energy_noise])]
    at_70 = maximum(d.total for d in energy_noise if d.E == 70.0)
    Markdown.parse(
        """
        **The noise of single energies.** One ray at one energy through the axis of the largest
        patients is the noisiest kernel in the library: at worst $(round(100 * worst.total; digits = 1)) %
        in its total ($(worst.scanner), $(round(worst.D / 10; digits = 1)) cm, $(round(Int, worst.E)) keV),
        and up to $(round(100 * at_70; digits = 1)) % at 70 keV. Almost all of it is multiple scattering,
        photons that diffused through 40–50 cm of water before reaching the detector, which forced
        detection cannot make less rare. No scan uses such a kernel alone: each ray carries the beam's
        whole spectrum and each scan hundreds of rays, so the checks judge the noise there.
        """
    )
end
# ╔═╡ fc4b31b5-83a1-5f18-8570-4be1f5c7985a
held_out_cases = vcat(
    [(D, u, 70.0) for D in (235.0, 315.0, 445.0) for u in (0.0, 0.5)],
    [(300.0, 0.5, 75.0)],
)

# ╔═╡ 9bc53a5c-830c-5637-b25a-ee344fedfe95
# Each held-out kernel from eight independent runs, each with the library's photons per kernel
held_out_runs = let
    runs = MC.simulate_sweep(
        [
            (
                    MC.CylinderPhantom(D),
                    E,
                    kernel_seed(D, u) + 11 + 1000 * chunk,
                    MC.kernel_ray(D, u),
                    2 * photons_per_half(D),
                )
                for (D, u, E) in held_out_cases for chunk in 1:8
        ];
        physics,
        readout,
    )
    [MC.merge_tallies(runs[(8j - 7):(8j)]) for j in eachindex(held_out_cases)]
end

# ╔═╡ e170737f-fc41-5ecb-bcd4-9ec7bd4938b8
held_out_differences = [
    kernel_difference(
            MC.library_kernel(library, s.name, held_out_cases[j]...)[1],
            held_out_runs[j][k].scatter ./ held_out_runs[j][k].n,
        )
        for j in eachindex(held_out_cases) for (k, s) in enumerate(scanners)
]

# ╔═╡ 61cc2a1d-bcb8-446b-91bf-099cee6f17ba
# The kernels' Monte Carlo primaries against Beer–Lambert, where enough primaries got through to
# measure (at least a thousand): a check of the transport and of the scoring together
primary_differences = let
    s = scanners[1]
    cases = [(D, u, E) for D in (200.0, 300.0) for u in (0.0, 0.5) for E in (60.0, 80.0, 100.0, 120.0)]
    differences = Float64[]
    for (D, u, E) in cases
        iD, iu, iE = findfirst(==(D), diameters_mm), findfirst(==(u), impact_fractions), findfirst(==(E), energies_keV)
        chord = D * sqrt(1 - u^2)
        k = round(Int, E)
        transmission = exp(-water_mu[k] * chord)
        transmission * 2 * photons_per_half(D) >= 1000 || continue
        expected = transmission * s.efficiency[k] * E
        push!(differences, abs(library.primary[s.name][iu, iD, iE] / expected - 1))
    end
    differences
end

# ╔═╡ db13c422-a859-5bc4-a784-ce63d23bbf34
# Rays through the axis: the scatter on the two halves of the fan, summed over every diameter and
# energy (each kernel then counts by its signal, the precise ones most)
symmetry = [
    let sides = [
                let kernel = MC.library_kernel(library, s.name, D, 0.0, E)[1]
                    half = size(kernel, 1) ÷ 2
                    (sum(kernel[1:half, :]), sum(kernel[(end - half + 1):end, :]))
            end
                for D in diameters_mm for E in energies_keV
            ]
            abs(log(sum(first, sides) / sum(last, sides)))
    end
        for s in scanners
]

# ╔═╡ 96dc3864-3035-4923-9f2d-ffd51cd56032
# Forced detection against the photon-by-photon count of the same photons: each scanner's total over
# every kernel, for the patients up to 35 cm and those above
forced_vs_analog = [
    let kernels = [j for (j, (D, u, E)) in enumerate(kernel_grid) if (D <= 350) == small]
            forced = sum(sum(kernel_tallies[j][k].scatter) / kernel_tallies[j][k].n for j in kernels)
            analog = sum(sum(kernel_tallies[j][k].analog_scatter) / kernel_tallies[j][k].n for j in kernels)
            (scanner = s.name, sizes = small ? "20–35 cm" : "36–50 cm", difference = abs(forced / analog - 1))
    end
        for (k, s) in enumerate(scanners) for small in (true, false)
]
# ╔═╡ bcab3e26-0dd7-5534-ac6b-33e841a2109b
md"""
**Whole scans.** The tube: BasisSimulator's IPEM spectrum at 120 kV through 9.2 mm of aluminium (the
Aquilion ONE beam fitted in notebook 08), a 40 mm beam at the isocentre, no bowtie. The direct
simulation fires the fan beam from the focal spot through the cylinder at the isocentre, 96 million
photons per case in 64 independent chunks (whose spread gives its own noise), scored on the one
scanner by forced detection. The library sums its pencil kernels for rays every 0.1° across the fan,
each weighted by its share of the beam's photons and energies. Each kernel reaches a full
detector width on either side of its own row, so along z the library covers whole the rows within
that reach of every ray in the beam (`complete_rows`): the whole detector and more, where the two
are compared.
"""

# ╔═╡ 1a500ca1-cd75-531b-89ee-5498b1477e6b
scan_spectrum = let
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

# ╔═╡ 3e855498-7f7a-51c1-a26a-89a2469ad324
scan_cases = [(name = name, D = D) for name in ("SOMATOM Force", "Aquilion ONE") for D in (275.0, 435.0)]

# ╔═╡ 048f71d8-7293-522d-acec-51433e592c41
# Every photon of the fan beam, 64 independent chunks of 1.5 million per case
direct_scans = [
    let s = scanners[findfirst(x -> x.name == case.name, scanners)]
            phantom = MC.CylinderPhantom(
                case.D;
                centre_mm = (0.0, s.source_to_isocenter_mm),
            )
            beam = MC.FocalSpot(;
                distance_mm = s.source_to_isocenter_mm,
                fan_deg = s.fan_deg,
                z_half_mm = 20.0,
            )
            chunks = MC.simulate_sweep(
                [(phantom, MC.Spectrum(scan_spectrum...), 500 + chunk, beam) for chunk in 1:64];
                n = 1_500_000,
                physics,
                readout = MC.ForcedScatterReadout([s]; water_mu_per_mm = water_mu),
            )
            (all = only(MC.merge_tallies(chunks)), chunks = [only(c) for c in chunks])
    end
        for case in scan_cases
]

# ╔═╡ aded6c40-7771-5e42-ad5c-3ab591c1a356
# The library's prediction from rays every 0.1° across the fan, each crossing its chord of the
# centred cylinder; from the whole library and from each of its halves, and the rows every ray's
# kernel reaches whole
library_scans = [
    let s = scanners[findfirst(x -> x.name == case.name, scanners)]
            cols = MC.column_edges(s)
            R = case.D / 2
            rays = [
                (
                    angle_deg = γ,
                    wet_mm = let b = s.source_to_isocenter_mm * sind(γ)
                        abs(b) < R ? 2 * sqrt(R^2 - b^2) : 0.0
                end,
                    weight = abs(γ) <= s.fan_deg / 2 ? 0.1 / s.fan_deg : 0.0,
                )
                for γ in (cols[1] + 0.05):0.1:cols[end]
            ]
            predict(lib) = MC.library_scan(
                lib,
                s.name,
                case.D,
                rays;
                collimation_mm = 40.0,
                spectrum = scan_spectrum,
            )
            whole = predict(library)
            (
                all = whole.scatter,
                halves = [predict(half).scatter for half in library_halves],
                complete = whole.complete_rows,
            )
    end
        for case in scan_cases
]

# ╔═╡ d22bc4c0-a510-5b73-8db1-7f159323027b
# The direct simulation against the library, per photon: at the centre (the three middle columns
# of the beam's rows), across the fan (each column, summed over the beam's rows) and along z (each
# row the library covers whole, summed over the columns the patient covers); and the noise of both
scan_differences = [
    let t = direct_scans[j].all
            direct = t.scatter ./ t.n
            predicted = library_scans[j].all
            halves = library_scans[j].halves
            rows = (t.row_edges_mm[1:(end - 1)] .+ t.row_edges_mm[2:end]) ./ 2
            s = scanners[findfirst(x -> x.name == case.name, scanners)]
            beam = findall(z -> abs(z) < 20.0 * s.source_to_detector_mm / s.source_to_isocenter_mm, rows)
            c = (size(direct, 1) + 1) ÷ 2
            middle = (c - 1):(c + 1)
            centre(m) = sum(m[middle, beam])
            across(m) = vec(sum(m[:, beam]; dims = 2))
            inside = findall(>(0.2 * maximum(across(direct))), across(direct))
            along(m) = vec(sum(m[inside, :]; dims = 1))
            profile = along(direct)
            covered = findall(
                r -> library_scans[j].complete[r] && profile[r] > 0.05 * maximum(profile),
                eachindex(profile),
            )
            rms(a, b) = sqrt(sum(abs2, a ./ b .- 1) / length(a))
            chunk_centres = [centre(k.scatter ./ k.n) for k in direct_scans[j].chunks]
            (
                centre = abs(centre(predicted) / centre(direct) - 1),
                across = rms(across(predicted)[inside], across(direct)[inside]),
                along = rms(along(predicted)[covered], along(direct)[covered]),
                library_noise_centre = abs(centre(halves[1]) / centre(halves[2]) - 1) / 2,
                library_noise_across = rms(across(halves[1])[inside], across(halves[2])[inside]) / 2,
                direct_noise_centre = std(chunk_centres) / sqrt(length(chunk_centres)) / mean(chunk_centres),
            )
    end
        for (j, case) in enumerate(scan_cases)
]

# ╔═╡ 3859e670-2f6e-5ed3-b726-b46b803a977b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1400, 900))
    for (j, case) in enumerate(scan_cases)
        t = direct_scans[j].all
        s = scanners[findfirst(x -> x.name == case.name, scanners)]
        direct = t.scatter ./ t.n
        predicted = library_scans[j].all
        cols, rows = t.column_edges_deg, t.row_edges_mm
        col_centres = (cols[1:(end - 1)] .+ cols[2:end]) ./ 2
        row_centres = (rows[1:(end - 1)] .+ rows[2:end]) ./ 2
        beam = findall(z -> abs(z) < 20.0 * s.source_to_detector_mm / s.source_to_isocenter_mm, row_centres)
        ax = Mke.Axis(
            fig[1, j];
            title = "$(case.name), $(round(case.D / 10; digits = 2)) cm",
            xlabel = "angle across the fan (°)",
            ylabel = j == 1 ? "scatter in the beam's rows,\nper photon (keV)" : "",
        )
        Mke.lines!(
            ax,
            col_centres,
            vec(sum(direct[:, beam]; dims = 2));
            color = PALETTE.ink,
            linewidth = 3,
            label = "direct Monte Carlo",
        )
        Mke.lines!(
            ax,
            col_centres,
            vec(sum(predicted[:, beam]; dims = 2));
            color = PALETTE.kept,
            linewidth = 3,
            linestyle = :dash,
            label = "library",
        )
        j == 1 && Mke.axislegend(ax; position = :ct, framevisible = false)
        ax2 = Mke.Axis(
            fig[2, j];
            xlabel = "z at the detector (mm)",
            ylabel = j == 1 ? "scatter summed across the fan,\nper photon (keV)" : "",
        )
        complete = library_scans[j].complete
        Mke.lines!(
            ax2,
            row_centres[complete],
            vec(sum(direct; dims = 1))[complete];
            color = PALETTE.ink,
            linewidth = 3,
        )
        Mke.lines!(
            ax2,
            row_centres[complete],
            vec(sum(predicted; dims = 1))[complete];
            color = PALETTE.kept,
            linewidth = 3,
            linestyle = :dash,
        )
    end
    fig
end

# ╔═╡ a9e8a8e9-d7bb-5cb7-9a06-c8d469c74476
md"""
## 4. The Scorecard
"""

# ╔═╡ 48f835c4-861a-5c9d-8f59-dfcd686d0aa4
checks = vcat(
    MC.check_below(
        "primaries: Monte Carlo against Beer–Lambert (largest relative difference)",
        isempty(primary_differences) ? NaN : maximum(primary_differences),
        0.05;
        source = "exact attenuation in water",
    ),
    MC.check_below(
        "noise of single rays of the 120 kV beam: totals (largest)",
        maximum(d.total for d in noise_differences),
        0.03;
        source = "the kernels' two halves",
    ),
    MC.check_below(
        "noise of single rays of the 120 kV beam: strong bins (median rms)",
        median([d.bins for d in noise_differences]),
        0.1;
        source = "the kernels' two halves",
    ),
    MC.check_below(
        "noise of the library's whole scans: at the centre (largest)",
        maximum(d.library_noise_centre for d in scan_differences),
        0.015;
        source = "the library's two halves",
    ),
    MC.check_below(
        "noise of the library's whole scans: across the fan (largest rms)",
        maximum(d.library_noise_across for d in scan_differences),
        0.02;
        source = "the library's two halves",
    ),
    [
        MC.check_below(
                "symmetry, $(s.name): rays through the axis, left against right (log ratio)",
                symmetry[k],
                0.005;
                source = "geometry, every diameter and energy",
            )
            for (k, s) in enumerate(scanners)
    ],
    [
        MC.check_below(
                "forced detection against counting, $(d.scanner), $(d.sizes)",
                d.difference,
                0.03;
                source = "the same photons, counted one by one",
            )
            for d in forced_vs_analog
    ],
    MC.check_below(
        "held-out diameters and energy: kernel totals (largest relative error)",
        maximum(d.total for d in held_out_differences),
        0.03;
        source = "kernels simulated between grid points, eight times the photons",
    ),
    MC.check_below(
        "held-out diameters and energy: strong bins (median rms relative error)",
        median([d.bins for d in held_out_differences]),
        0.12;
        source = "kernels simulated between grid points, eight times the photons",
    ),
    MC.check_below(
        "whole scans: the direct simulation's own noise at the centre (largest)",
        maximum(d.direct_noise_centre for d in scan_differences),
        0.02;
        source = "64 independent chunks",
    ),
    [
        MC.check_below(
                "whole scan, $(case.name), $(case.D / 10) cm: scatter at the centre",
                scan_differences[j].centre,
                0.05;
                source = "direct Monte Carlo of the fan beam",
            )
            for (j, case) in enumerate(scan_cases)
    ],
    [
        MC.check_below(
                "whole scan, $(case.name), $(case.D / 10) cm: across the fan (rms)",
                scan_differences[j].across,
                0.06;
                source = "direct Monte Carlo of the fan beam",
            )
            for (j, case) in enumerate(scan_cases)
    ],
    [
        MC.check_below(
                "whole scan, $(case.name), $(case.D / 10) cm: along z, the rows covered whole (rms)",
                scan_differences[j].along,
                0.08;
                source = "direct Monte Carlo of the fan beam",
            )
            for (j, case) in enumerate(scan_cases)
    ],
)

# ╔═╡ 702a6c05-758c-59d0-b017-9c3da299400a
Markdown.parse(MC.scorecard(checks))

# ╔═╡ ff194c83-d65b-5feb-9d05-bf4af427736e
md"""
## Saving the Library

One file in `lib/BasisMonteCarlo/tables/`: the kernels of all six scanners, their grids of
diameters, impacts and energies, the scanners as modelled, the photons per kernel, and the
scorecard above, so that whoever uses the library sees what it passed.
"""

# ╔═╡ 657db79e-fdde-5587-9858-798b66066db7
saved = let
    path = joinpath(table_dir, "scatter_library_geant4.jls")
    library.metadata["scorecard"] = MC.scorecard(checks)
    library.metadata["checks_passed"] = count(c -> c.passed, checks)
    library.metadata["checks_total"] = length(checks)
    Serialization.serialize(path, library)
    path
end

# ╔═╡ 84c11691-4e90-5bb2-92d8-e22f90fcf4ab
let
    reloaded = Serialization.deserialize(saved)
    (
        file = basename(saved),
        megabytes = round(filesize(saved) / 1.0e6; digits = 1),
        scanners = [s.name for s in reloaded.scanners],
        kernels = length(reloaded.diameters_mm) * length(reloaded.impact_fractions) * length(reloaded.energies_keV),
    )
end

# ╔═╡ Cell order:
# ╟─be56f13c-5abb-5ce4-9011-3964bb0223d3
# ╟─9e90d4ca-5f06-5847-bb8d-aabdf1f2c9b2
# ╟─e0e565df-c3ed-4b6d-b27f-66226d58c2ce
# ╟─260664e3-1968-5c97-8dfd-017ae8516352
# ╠═47fa26ea-721e-5b66-922a-52973af46f34
# ╠═57bc93bd-06f3-5fd9-ac5e-fb4bf1cd0552
# ╠═0abf2ee9-00ed-56be-a6d2-aa80e0ab5ff3
# ╠═af0399b5-424a-5185-8615-b98ab53fd637
# ╠═92592abd-21b6-52ee-b4ea-fc480e994c21
# ╠═25ba20a9-568c-5267-bb9b-2fe80bd08476
# ╠═dc10d567-286d-577d-ae4e-dfb80d3c8f3b
# ╠═1b94b105-16a2-52ed-8367-157464da5eea
# ╠═9c017251-6dcc-5d5e-bf01-07818d561ec5
# ╠═927f4d8a-337c-5169-93dc-b740520aaa90
# ╠═fa7d87e4-de51-5da1-8211-9182959161dc
# ╠═da800d9e-8bc7-59c6-ad19-e5365d75c7cc
# ╠═115c241b-2b41-5a68-8983-2135a55af1a8
# ╠═7fbd21af-2717-503b-af34-b57421e69d09
# ╟─0f82726d-b35c-5dae-a6e1-ba68689f076d
# ╠═7848689c-0a69-53f7-b7fd-88f3413a4176
# ╠═bdcc54a3-c0f5-5800-8c85-896244e8d07d
# ╠═7d685d9d-aaed-5ffe-9b04-08a35096c634
# ╠═1db19aed-4a60-54a1-913e-398750a7997e
# ╟─e4a18eea-cb39-5575-9230-e0ff552875a3
# ╟─49b6c40d-5008-54e4-bee4-c49738a1f800
# ╟─aa74500e-69bb-45dc-9b84-b97d02d12150
# ╠═a285f955-c394-41f8-a94b-65861e8a9bee
# ╟─ecd65b8c-d0f6-4502-80f0-7c417e498baa
# ╠═420ca537-a700-5d72-8901-ab9c5fdcb42f
# ╠═0191eb88-a4aa-593b-8d18-32f5fab725cb
# ╠═1ca0b7d9-8065-5e30-9fad-7e035e483d95
# ╠═490c427d-a22a-5ea9-87af-256758fd46cf
# ╠═9fd96be0-2511-59f0-873c-440d74415297
# ╠═df41e785-c04b-4d8a-b2f1-ab5d1e49a130
# ╠═254033b1-90b6-5df7-8582-b5aba5fe9e46
# ╠═673d1dfc-4b31-4673-a44e-e5255186ea3b
# ╠═ac791f75-a62f-424a-a893-c12e5492ed7e
# ╠═b680104b-e544-5afd-94b3-e6a92fc42960
# ╠═c36e2624-afa7-44a0-a0b2-bbb1a66211a1
# ╟─77f080ef-da2b-5ce5-81a0-0ee4ae8f5310
# ╟─687935da-9ec8-55cf-8491-b95da499e2ba
# ╟─2108da2a-f36e-5b98-944b-bd92b09fd20b
# ╟─12c91dff-71b8-54ef-b649-c9e389568616
# ╠═494c2134-fd68-51de-bef2-eea2ca1d5b91
# ╠═b3cf4ace-0f65-5679-b861-da072652ac64
# ╠═0bdb1f8f-17d8-45ac-a652-e509f89d422b
# ╠═f32a0349-3698-55f3-ab6f-966791eea928
# ╠═ee197700-86f5-4eea-8631-389819370c97
# ╟─f09afb2f-b189-44ff-b632-9b230b731275
# ╠═fc4b31b5-83a1-5f18-8570-4be1f5c7985a
# ╠═9bc53a5c-830c-5637-b25a-ee344fedfe95
# ╠═e170737f-fc41-5ecb-bcd4-9ec7bd4938b8
# ╠═61cc2a1d-bcb8-446b-91bf-099cee6f17ba
# ╠═db13c422-a859-5bc4-a784-ce63d23bbf34
# ╠═96dc3864-3035-4923-9f2d-ffd51cd56032
# ╟─bcab3e26-0dd7-5534-ac6b-33e841a2109b
# ╠═1a500ca1-cd75-531b-89ee-5498b1477e6b
# ╠═3e855498-7f7a-51c1-a26a-89a2469ad324
# ╠═048f71d8-7293-522d-acec-51433e592c41
# ╠═aded6c40-7771-5e42-ad5c-3ab591c1a356
# ╠═d22bc4c0-a510-5b73-8db1-7f159323027b
# ╟─3859e670-2f6e-5ed3-b726-b46b803a977b
# ╟─a9e8a8e9-d7bb-5cb7-9a06-c8d469c74476
# ╠═48f835c4-861a-5c9d-8f59-dfcd686d0aa4
# ╠═702a6c05-758c-59d0-b017-9c3da299400a
# ╟─ff194c83-d65b-5feb-9d05-bf4af427736e
# ╠═657db79e-fdde-5587-9858-798b66066db7
# ╠═84c11691-4e90-5bb2-92d8-e22f90fcf4ab
