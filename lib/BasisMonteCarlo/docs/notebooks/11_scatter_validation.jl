### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils


# ╔═╡ 9e0038f2-df07-5bc8-b36f-25eae12e74eb
md"""
# Validating the Scatter Library Beyond Water Cylinders

The scatter library of notebook 10 was built from water cylinders. Patients are not water cylinders:
they are elliptical, off-centre, and made of lung, bone and soft tissue. The library treats every
patient, ray by ray, as the water cylinder of the patient's **water-equivalent diameter** (WED,
AAPM Report 220) that the ray crosses for the same water-equivalent length. This notebook measures
how well that holds: five phantoms that are not water cylinders, each simulated directly in Geant4
(every photon of the fan beam, every scattering scored by forced detection), against the library's
prediction and against BasisSimulator's current model, on a scanner with a 2-D grid (the SOMATOM
Force) and one with a 1-D grid (the Aquilion ONE). Every comparison is a number with a target; the notebook ends with its scorecard.
"""

# ╔═╡ 6c441e5d-e5e6-59ee-ab46-e7bb118e71b5
md"""
## Notebook Setup
"""

# ╔═╡ ddfb77a8-1acd-5fc1-90a3-81b04bd7216e
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 3ac59922-7da3-5b41-8448-c49ea6db6ffa
using Markdown: @md_str, Markdown

# ╔═╡ 482f53df-95cc-5eff-8b49-9f50f561f1f2
import PlutoUI

# ╔═╡ d8487463-34bf-5901-8cd4-869baad6d729
import BasisMonteCarlo as MC

# ╔═╡ 5e0d4372-98f1-5e7d-aaf0-4275aceb5d04
import CairoMakie as Mke

# ╔═╡ 3a5cb03a-bff7-51fe-b9d0-04030865ec7d
import BasisSimulator as BS

# ╔═╡ b42b632b-ed78-52a9-8661-c959957e4f4c
import Serialization

# ╔═╡ 10023f1f-dc34-5825-9fe8-618da7e842c5
using Statistics: mean, median, std

# ╔═╡ 7d3fd247-14e0-5f8c-9ed6-868ba1e749a1
PlutoUI.TableOfContents()

# ╔═╡ b9beeb33-1e85-564b-8c42-f8ad1243ad1c
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 62442d0b-ef4d-5cbf-8d1a-e99984d29c0c
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

# ╔═╡ a6224573-fec9-5281-8cc4-3620e6868dcc
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

# ╔═╡ 5cfb96aa-3d6a-509a-bc47-89f3c7dafa85
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

# ╔═╡ bdde2ceb-4db8-572a-82a9-f8fb295ba9e0
# For scatter only the photons matter: electrons stop within a millimetre of where they are made
physics = MC.PhysicsSettings(; production_cut_um = 1000.0, electron_step_um = 1000.0)

# ╔═╡ 17d722cd-bfe7-55b7-9543-462bec58e049
table_dir = joinpath(@__DIR__, "..", "..", "tables")

# ╔═╡ 997fd0c7-f40b-5436-87a1-6b85407c0c61
# The library notebook 10 saved, with its own scorecard
library = Serialization.deserialize(joinpath(table_dir, "scatter_library_geant4.jls"))

# ╔═╡ 9ae13765-5bf9-5316-9464-18955d2d91df
Markdown.parse("**The library's own checks (notebook 10):**\n\n" * library.metadata["scorecard"])

# ╔═╡ df809b21-81f0-5d9e-971f-942df142a15f
md"""
## 1. The Phantoms

Each 80 cm long along the patient's axis, centred at the isocentre unless stated:

- **wide ellipse**: water, 36 × 26 cm, its long axis across the fan;
- **tall ellipse**: the same ellipse turned, its long axis along the beam;
- **off-centre cylinder**: water, 30 cm, moved 6 cm across the fan;
- **Gammex 472**: BasisSimulator's phantom of that name, a 33 cm solid-water body with seven calcium
  and seven iodine rods 2.8 cm across, in BasisSimulator's own materials;
- **thorax**: a 34 × 24 cm soft-tissue (water) ellipse with two lungs (ICRP lung, 0.26 g/cm³) and a
  spine (cortical bone).
"""

# ╔═╡ f5ff809e-9476-59fa-9718-dd4859116c80
phantoms = let
    water = MC.DetectorMaterial(:Water)
    gammex_rods = vcat(
        [
            MC.PhantomInsert(;
                    material = MC.DetectorMaterial(m),
                    x_mm = 50.0 * cos(2π * i / 7),
                    y_mm = 50.0 * sin(2π * i / 7),
                    semi_axes_mm = (14.0, 14.0),
                )
                for (i, m) in zip(0:6, (BS.Ca_50, BS.Ca_100, BS.Ca_200, BS.Ca_300, BS.Ca_400, BS.Ca_500, BS.Ca_600))
        ],
        [
            MC.PhantomInsert(;
                    material = MC.DetectorMaterial(m),
                    x_mm = 105.0 * cos(2π * i / 7 + π / 7),
                    y_mm = 105.0 * sin(2π * i / 7 + π / 7),
                    semi_axes_mm = (14.0, 14.0),
                )
                for (i, m) in zip(0:6, (BS.I_2_0, BS.I_2_5, BS.I_5_0, BS.I_7_5, BS.I_10_0, BS.I_15_0, BS.I_20_0))
        ],
    )
    [
        (
            name = "wide ellipse",
            phantom = MC.CylinderPhantom(; semi_axes_mm = (180.0, 130.0)),
        ),
        (
            name = "tall ellipse",
            phantom = MC.CylinderPhantom(; semi_axes_mm = (130.0, 180.0)),
        ),
        (
            name = "off-centre cylinder",
            phantom = MC.CylinderPhantom(300.0; centre_mm = (60.0, 0.0)),
        ),
        (
            name = "Gammex 472",
            phantom = MC.CylinderPhantom(;
                material = MC.DetectorMaterial(BS.solid_water),
                semi_axes_mm = (165.0, 165.0),
                inserts = gammex_rods,
            ),
        ),
        (
            name = "thorax",
            phantom = MC.CylinderPhantom(;
                material = water,
                semi_axes_mm = (170.0, 120.0),
                inserts = [
                    MC.PhantomInsert(;
                        material = MC.DetectorMaterial(:Lung),
                        x_mm = -75.0,
                        y_mm = -5.0,
                        semi_axes_mm = (55.0, 75.0),
                    ),
                    MC.PhantomInsert(;
                        material = MC.DetectorMaterial(:Lung),
                        x_mm = 75.0,
                        y_mm = -5.0,
                        semi_axes_mm = (55.0, 75.0),
                    ),
                    MC.PhantomInsert(;
                        material = MC.DetectorMaterial(:CorticalBone),
                        x_mm = 0.0,
                        y_mm = 90.0,
                        semi_axes_mm = (18.0, 15.0),
                    ),
                ],
            ),
        ),
    ]
end

# ╔═╡ 0ec9cb9c-ca21-5509-88c6-45802583bb6b
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1500, 380))
    θ = range(0, 2π; length = 160)
    for (k, case) in enumerate(phantoms)
        p = case.phantom
        ax = Mke.Axis(
            fig[1, k];
            title = case.name,
            aspect = Mke.DataAspect(),
            limits = ((-230, 230), (-230, 230)),
        )
        Mke.hidedecorations!(ax)
        ellipse(c, a) = [Mke.Point2f(c[1] + a[1] * cos(t), c[2] + a[2] * sin(t)) for t in θ]
        Mke.poly!(
            ax,
            ellipse(p.centre_mm, p.semi_axes_mm);
            color = (PALETTE.photon, 0.15),
            strokecolor = PALETTE.photon,
            strokewidth = 2,
        )
        for insert in p.inserts
            color = insert.material.density < 0.5 ? (PALETTE.muted, 0.25) : (PALETTE.xray, 0.6)
            Mke.poly!(
                ax,
                ellipse(p.centre_mm .+ (insert.x_mm, insert.y_mm), insert.semi_axes_mm);
                color,
                strokecolor = PALETTE.ink,
                strokewidth = 1,
            )
        end
        Mke.scatter!(
            ax,
            [Mke.Point2f(0, 0)];
            marker = :cross,
            color = PALETTE.ink,
            markersize = 12,
        )
        note!(
            ax,
            0,
            -225,
            "beam enters from below";
            align = (:center, :bottom),
            fontsize = 12,
            color = PALETTE.muted,
        )
    end
    fig
end

# ╔═╡ af12453d-0101-5c03-94b3-11b34f6c2190
md"""
## 2. The Scans

The beam: BasisSimulator's IPEM spectrum at 120 kV through 9.2 mm of aluminium, 40 mm wide at the
isocentre, no bowtie, one view (the beam along +y). The direct simulation fires the fan beam through
each phantom placed at the scanner's isocentre, 48 million photons per case in 32 independent chunks
(whose spread gives its own noise), and scores every scattering on the scanner's detector by forced
detection (notebook 10), in whatever material it happened: water, lung, bone or a rod. The
library's prediction takes rays every 0.1° across the fan, each ray's water-equivalent length
through the phantom and the phantom's water-equivalent diameter from the same projection, and sums
the kernels. A ray longer than the WED (along the tall ellipse's long axis, through the Gammex
phantom's rods, through the thorax's spine) cannot cross a cylinder of that diameter for its own
length, so it is given its own cylinder, as wide as it is long, and crosses it through the axis. BasisSimulator's current model is run on the same projection, one ray per column of
the detector, with its size scaling at the same diameter.
"""

# ╔═╡ 3ddaaa30-9111-50c0-9836-f278f92809fa
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

# ╔═╡ 73a026f2-074f-5260-823f-6b36d1a3db31
scanner_names = ["SOMATOM Force", "Aquilion ONE"]

# ╔═╡ f980cfd3-0d1d-51dd-915c-88ca48712a60
"""The phantom moved to the scanner's isocentre (the focal spot at the origin, the beam along +y)."""
at_isocenter(p, s) = MC.CylinderPhantom(;
    material = p.material,
    semi_axes_mm = p.semi_axes_mm,
    length_mm = p.length_mm,
    centre_mm = p.centre_mm .+ (0.0, s.source_to_isocenter_mm),
    inserts = p.inserts,
)

# ╔═╡ f7a2b1b9-2d25-5025-97ff-e9d379d4788a
cases = [(phantom = case, scanner = name) for case in phantoms for name in scanner_names]


# ╔═╡ f67c80e6-ec4d-43e1-aaa7-b674a8ee268d
# Geant4's attenuation of every material in the phantoms, for forced detection
attenuation = MC.forced_attenuation(
    unique(
        m -> m.name,
        [m for case in phantoms for m in vcat(case.phantom.material, [i.material for i in case.phantom.inserts])],
    ),
)

# ╔═╡ 56dcfda7-4d44-5201-8424-15fbdc7859cc
# Every photon of the fan beam: 32 independent chunks of 1.5 million per case
direct_chunks = [
    let s = library.scanners[findfirst(x -> x.name == case.scanner, library.scanners)]
            beam = MC.FocalSpot(;
                distance_mm = s.source_to_isocenter_mm,
                fan_deg = s.fan_deg,
                z_half_mm = 20.0,
            )
            chunks = MC.simulate_sweep(
                [(at_isocenter(case.phantom.phantom, s), MC.Spectrum(spectrum...), 9000 + 100j + chunk, beam) for chunk in 1:32];
                n = 1_500_000,
                physics,
                readout = MC.ForcedScatterReadout([s]; attenuation),
            )
            [only(chunk) for chunk in chunks]
    end
        for (j, case) in enumerate(cases)
]


# ╔═╡ 26def491-0406-41b8-94a9-a816b529c240
direct = [reduce(+, chunks) for chunks in direct_chunks]

# ╔═╡ 8a34c98b-99fc-5892-a874-9edad32772de
# For each case: the ray of each column of the detector and rays every 0.1° across the fan, each
# with its water-equivalent length; and the phantom's WED
projections = [
    let s = library.scanners[findfirst(x -> x.name == case.scanner, library.scanners)]
            p = at_isocenter(case.phantom.phantom, s)
            cols = MC.column_edges(s)
            ray(γ, width_deg) = (
                angle_deg = γ,
                wet_mm = MC.water_equivalent_length(p, (0.0, 0.0), (sind(γ), cosd(γ))),
                weight = abs(γ) <= s.fan_deg / 2 ? width_deg / s.fan_deg : 0.0,
            )
            (
                rays = [ray(γ, s.column_bin_deg) for γ in (cols[1:(end - 1)] .+ cols[2:end]) ./ 2],
                fine_rays = [ray(γ, 0.1) for γ in (cols[1] + 0.05):0.1:cols[end]],
                wed_mm = MC.water_equivalent_diameter(p; fan_deg = s.fan_deg),
            )
    end
        for case in cases
]

# ╔═╡ a3a2c951-b97d-5e30-aae2-29f934c02655
predicted = [
    MC.library_scan(
            library,
            case.scanner,
            projections[j].wed_mm,
            projections[j].fine_rays;
            collimation_mm = 40.0,
            spectrum,
        )
        for (j, case) in enumerate(cases)
]

# ╔═╡ 241b138e-6fce-5b2a-a217-71902cdf0f62
md"""
**BasisSimulator today**, on each case: its scatter field (`estimate_scatter_field!`, the scanner's
geometry and the phantom's water-equivalent diameter in `geometry_aware_scatter_model`) from the
polychromatic projection of the water-equivalent lengths, times its energy weighting. It works in
intensity relative to the unattenuated beam, so it is compared as a scatter-to-primary ratio; the
direct simulation and the library are put on the same footing with the exact primary of each
column.
"""

# ╔═╡ f20ef946-92af-51a4-b778-2a29e23e4804
water_mu = [MC.geant4_attenuation(MC.DetectorMaterial(:Water), Float64(E)) for E in 1:150]

# ╔═╡ 6d28368c-4aae-566f-bdfc-140f80a10b98
"""The fraction of the beam's photons a water-equivalent length lets through, unscattered."""
function transmission(wet_mm)
    e, w = spectrum
    return sum(wk * exp(-water_mu[clamp(round(Int, Ek), 1, 150)] * wet_mm) for (Ek, wk) in zip(e, w)) / sum(w)
end

# ╔═╡ a569223d-d467-56fa-9122-3b32a5ca8d8f
# BasisSimulator's scatter-to-primary ratio on the central row, column by column
bs_profiles = [
    let s = library.scanners[findfirst(x -> x.name == case.scanner, library.scanners)]
            rays = projections[j].rays
            bs_scanner = BS.EICTScanner(
                source_to_isocenter = s.source_to_isocenter_mm,
                source_to_detector = s.source_to_detector_mm,
                detector_rows = 64,
                detector_cols = length(rays),
                detector_row_size = 40.0 / 64,
                detector_col_size = s.source_to_isocenter_mm * deg2rad(s.column_bin_deg),
                detector_material = :ufc,
                detector_depth = 1.4,
            )
            sinogram = zeros(Float64, length(rays), 64, 1)
            for (i, ray) in enumerate(rays)
                sinogram[i, :, 1] .= -log(transmission(ray.wet_mm))
        end
            model = BS.geometry_aware_scatter_model(
                bs_scanner;
                phantom_diameter_cm = projections[j].wed_mm / 10,
            )
            field = similar(sinogram)
            BS.estimate_scatter_field!(field, sinogram, model)
            e, w = spectrum
            weight = sum(w .* BS.compute_scatter_energy_weights(collect(Float64, e))) / sum(w)
            field[:, 32, 1] .* weight ./ exp.(-sinogram[:, 32, 1])
    end
        for (j, case) in enumerate(cases)
]

# ╔═╡ 3855003e-b0ba-5371-b6a8-dd1c9a31da0e
"""The exact primary per photon in each column of the central row: Beer–Lambert through every material."""
function exact_primary(case, s, row_share)
    p = at_isocenter(case.phantom.phantom, s)
    e, w = spectrum
    weights = w ./ sum(w)
    cols = MC.column_edges(s)
    centres = (cols[1:(end - 1)] .+ cols[2:end]) ./ 2
    μ(m) = [MC.linear_attenuation(m, Float64(E)) for E in 1:150]
    cache = Dict{String, Vector{Float64}}()
    function column_primary(γ)
        abs(γ) <= s.fan_deg / 2 || return 0.0
        lengths = MC.material_lengths(p, (0.0, 0.0), (sind(γ), cosd(γ)))
        detected = sum(
            wk * exp(-sum(get!(() -> μ(m), cache, m.name)[clamp(round(Int, Ek), 1, 150)] * L for (m, L) in lengths)) *
                s.efficiency[clamp(round(Int, Ek), 1, 150)] * Ek
                for (Ek, wk) in zip(e, weights)
        )
        return s.column_bin_deg / s.fan_deg * row_share * detected
    end
    return [column_primary(γ) for γ in centres]
end

# ╔═╡ 6c7a3173-cca3-5472-9586-44db7ccbdd14
# Central-row scatter: library against direct (absolute, per photon, at the centre: the three middle
# columns), every model as SPR over every column inside the phantom, and the direct simulation's
# own noise at the centre
comparison = [
    let s = library.scanners[findfirst(x -> x.name == case.scanner, library.scanners)]
            d = direct[j].scatter ./ direct[j].n
            l = predicted[j].scatter
            r = (size(d, 2) + 1) ÷ 2
            width = 40.0 * s.source_to_detector_mm / s.source_to_isocenter_mm
            primary = exact_primary(case, s, min(s.row_bin_mm, width) / width)
            # every column whose ray crosses the phantom: the most attenuated rays included
            inside = findall(
                i -> projections[j].rays[i].wet_mm > 0 && primary[i] > 0 && d[i, r] > 0,
                1:size(d, 1),
            )
            middle = ((size(d, 1) + 1) ÷ 2 - 1):((size(d, 1) + 1) ÷ 2 + 1)
            chunk_centres = [sum(k.scatter[middle, r]) / k.n for k in direct_chunks[j]]
            spr_direct = d[inside, r] ./ primary[inside]
            spr_library = l[inside, r] ./ primary[inside]
            spr_bs = bs_profiles[j][inside]
            rms(a, b) = sqrt(sum(abs2, a ./ b .- 1) / length(b))
            (
                centre = abs(sum(l[middle, r]) / sum(d[middle, r]) - 1),
                direct_noise = std(chunk_centres) / sqrt(length(chunk_centres)) / mean(chunk_centres),
                library_rms = rms(spr_library, spr_direct),
                bs_rms = rms(spr_bs, spr_direct),
                inside = inside,
                angles = ((direct[j].column_edges_deg[1:(end - 1)] .+ direct[j].column_edges_deg[2:end]) ./ 2)[inside],
                spr_direct = spr_direct,
                spr_library = spr_library,
                spr_bs = spr_bs,
            )
    end
        for (j, case) in enumerate(cases)
]

# ╔═╡ 3777185c-2648-5803-abbd-12248ba7efc6
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1600, 760))
    for (j, case) in enumerate(cases)
        row, col = case.scanner == "SOMATOM Force" ? 1 : 2, (j + 1) ÷ 2
        cmp = comparison[j]
        ax = Mke.Axis(
            fig[row, col];
            title = "$(case.phantom.name)\n$(case.scanner)",
            titlesize = 16,
            xlabel = row == 2 ? "angle across the fan (°)" : "",
            ylabel = col == 1 ? "scatter-to-primary ratio" : "",
            yscale = log10,
        )
        Mke.lines!(
            ax,
            cmp.angles,
            cmp.spr_direct;
            color = PALETTE.ink,
            linewidth = 3,
            label = "direct Monte Carlo",
        )
        Mke.lines!(
            ax,
            cmp.angles,
            cmp.spr_library;
            color = PALETTE.kept,
            linewidth = 3,
            linestyle = :dash,
            label = "library",
        )
        Mke.lines!(
            ax,
            cmp.angles,
            cmp.spr_bs;
            color = PALETTE.reference,
            linewidth = 2,
            linestyle = :dot,
            label = "BasisSimulator today",
        )
        j == 1 && Mke.axislegend(ax; position = :lt, labelsize = 13)
    end
    fig
end

# ╔═╡ 674b51a1-4b66-5f67-9593-94903a24c841
let
    rows = [
        "| $(case.phantom.name) | $(case.scanner) | $(round(projections[j].wed_mm / 10; digits = 1)) cm | $(round(100comparison[j].centre; digits = 1)) % | $(round(100comparison[j].library_rms; digits = 1)) % | $(round(100comparison[j].bs_rms; digits = 1)) % |"
            for (j, case) in enumerate(cases)
    ]
    Markdown.parse(
        """
        | phantom | scanner | WED | library, centre | library, across the fan (rms) | BasisSimulator today, across the fan (rms) |
        |---|---|---|---|---|---|
        $(join(rows, "\n"))

        Differences from the direct simulation: the scatter at the centre, and the scatter-to-primary ratio across the fan.
        """
    )
end

# ╔═╡ e67c62ed-d243-4525-ba94-b49b4c3a2b5d
md"""
### The Thorax: Which WED?

The thorax's lungs lower its water-equivalent diameter well below that of its outline, while the
central ray crosses soft tissue and spine. A comparison, **not** the method the library uses: the
same prediction with the WED of the thorax's outer contour (the lungs taken as water), against the
same direct simulation.
"""

# ╔═╡ 06281871-fe7c-4cf1-b442-c871693bf24e
# The library's prediction for the thorax with the WED of its outline, every insert taken as the body
thorax_outline = [
    let s = library.scanners[findfirst(x -> x.name == case.scanner, library.scanners)]
            p = at_isocenter(case.phantom.phantom, s)
            outline = MC.CylinderPhantom(;
                material = p.material,
                semi_axes_mm = p.semi_axes_mm,
                length_mm = p.length_mm,
                centre_mm = p.centre_mm,
            )
            wed = MC.water_equivalent_diameter(outline; fan_deg = s.fan_deg)
            l = MC.library_scan(library, case.scanner, wed, projections[j].fine_rays; collimation_mm = 40.0, spectrum).scatter
            d = direct[j].scatter ./ direct[j].n
            r = (size(d, 2) + 1) ÷ 2
            middle = ((size(d, 1) + 1) ÷ 2 - 1):((size(d, 1) + 1) ÷ 2 + 1)
            inside = comparison[j].inside
            (
                scanner = case.scanner,
                wed_mm = wed,
                centre = abs(sum(l[middle, r]) / sum(d[middle, r]) - 1),
                library_rms = sqrt(sum(abs2, l[inside, r] ./ d[inside, r] .- 1) / length(inside)),
            )
    end
        for (j, case) in enumerate(cases) if case.phantom.name == "thorax"
]

# ╔═╡ 10db6197-a1f6-43cb-bcbf-7e1f33290858
let
    rows = [
        let j = findfirst(c -> c.phantom.name == "thorax" && c.scanner == o.scanner, cases)
                "| $(o.scanner) | $(round(projections[j].wed_mm / 10; digits = 1)) cm: $(round(100comparison[j].centre; digits = 1)) %, $(round(100comparison[j].library_rms; digits = 1)) % | $(round(o.wed_mm / 10; digits = 1)) cm: $(round(100o.centre; digits = 1)) %, $(round(100o.library_rms; digits = 1)) % |"
        end
            for o in thorax_outline
    ]
    Markdown.parse(
        """
        | thorax | the patient's WED (the method): centre, across the fan | the outline's WED (comparison): centre, across the fan |
        |---|---|---|
        $(join(rows, "\n"))

        **Not settled.** One phantom favouring the outline is a hint, not a rule: lungs are the case where
        the two definitions part most, and a rule for them needs more than one thorax (a range of lung
        sizes, positions and densities) before the library's method changes.
        """
    )
end

# ╔═╡ df233fa9-bedc-4605-9a7a-bb0fa36b193d
md"""
### The Gammex Phantom: Its Solid Water Is Not Water

Water-equivalence matches attenuation, not every kind of scattering. Gammex's solid water is mostly
carbon, water mostly oxygen: at the same total attenuation, its Rayleigh scattering (the gentle,
forward kind that a grid lets through most) is weaker. The plain body of the phantom, a 33 cm
cylinder with no rods, settles how much of the Gammex error that explains: the same cylinder in
water and in solid water, simulated directly and predicted by the library, with forced detection
checked against counting photon by photon on the same photons.
"""

# ╔═╡ 46a9849d-76ec-4238-8d63-8fe8db2118cc
# A 33 cm cylinder of water and of Gammex solid water, no rods: direct (16 chunks) against the library
gammex_body = let
    s = library.scanners[findfirst(x -> x.name == "SOMATOM Force", library.scanners)]
    solid = MC.DetectorMaterial(BS.solid_water)
    water = MC.DetectorMaterial(:Water)
    beam = MC.FocalSpot(; distance_mm = s.source_to_isocenter_mm, fan_deg = s.fan_deg, z_half_mm = 20.0)
    cols = MC.column_edges(s)
    [
        let p = at_isocenter(MC.CylinderPhantom(; material, semi_axes_mm = (165.0, 165.0)), s)
                chunks = [
                    only(c) for c in MC.simulate_sweep(
                        [(p, MC.Spectrum(spectrum...), 9900 + chunk, beam) for chunk in 1:16];
                        n = 1_500_000,
                        physics,
                        readout = MC.ForcedScatterReadout([s]; attenuation),
                    )
                ]
                t = reduce(+, chunks)
                r = (size(t.scatter, 2) + 1) ÷ 2
                middle = ((size(t.scatter, 1) + 1) ÷ 2 - 1):((size(t.scatter, 1) + 1) ÷ 2 + 1)
                wed = MC.water_equivalent_diameter(p; fan_deg = s.fan_deg)
                rays = [
                    (
                        angle_deg = γ,
                        wet_mm = MC.water_equivalent_length(p, (0.0, 0.0), (sind(γ), cosd(γ))),
                        weight = abs(γ) <= s.fan_deg / 2 ? 0.1 / s.fan_deg : 0.0,
                    )
                    for γ in (cols[1] + 0.05):0.1:cols[end]
                ]
                l = MC.library_scan(library, s.name, wed, rays; collimation_mm = 40.0, spectrum).scatter
                forced_over_analog = [sum(c.scatter) / sum(c.analog_scatter) for c in chunks]
                (
                    material = material.name,
                    wed_mm = wed,
                    library_over_direct = sum(l[middle, r]) / (sum(t.scatter[middle, r]) / t.n),
                    centre_per_photon = sum(t.scatter[middle, r]) / t.n,
                    forced_over_analog = mean(forced_over_analog),
                    forced_over_analog_error = std(forced_over_analog) / sqrt(length(forced_over_analog)),
                )
        end
            for material in (water, solid)
    ]
end

# ╔═╡ 8c16e306-3a3a-46b4-bc4a-6d08a6a7bab8
let
    w, sw = gammex_body
    solid, water = MC.DetectorMaterial(BS.solid_water).name, MC.DetectorMaterial(:Water).name
    ratio(kind, E) = getproperty(attenuation[solid], kind)[E] / getproperty(attenuation[water], kind)[E]
    Markdown.parse(
        """
        | 33 cm cylinder, no rods | WED | library over direct, at the centre | forced detection over counting (whole detector) |
        |---|---|---|---|
        | water | $(round(w.wed_mm / 10; digits = 1)) cm | $(round(w.library_over_direct; digits = 3)) | $(round(w.forced_over_analog; digits = 4)) ± $(round(w.forced_over_analog_error; digits = 4)) |
        | Gammex solid water | $(round(sw.wed_mm / 10; digits = 1)) cm | $(round(sw.library_over_direct; digits = 3)) | $(round(sw.forced_over_analog; digits = 4)) ± $(round(sw.forced_over_analog_error; digits = 4)) |

        The two cylinders have the same WED and the same total and Compton attenuation within half a
        percent (at 70 keV: $(round(ratio(:total, 70); digits = 3)) and $(round(ratio(:compton, 70); digits = 3)) of water's), but solid water's Rayleigh
        scattering is $(round(100 * (1 - ratio(:rayleigh, 70)); digits = 0)) % weaker, and the scatter reaching the centre of the detector is
        $(round(100 * (1 - sw.centre_per_photon / w.centre_per_photon); digits = 0)) % lower. Forced detection agrees with counting photon by photon in both, so the
        difference is physics, not the scoring. The library, built in water, cannot see it: **for this
        phantom it is limited by its body material, and its Gammex checks stay failing** (the target is
        not loosened). A material correction (Rayleigh and Compton kernels kept apart, each scaled by the
        material's own cross-section) would be the fix; it is not built.
        """
    )
end

# ╔═╡ 0050e067-6aae-5e84-b4eb-1a6248de33ec
md"""
## 3. The Scorecard

For every phantom and scanner: the library's scatter at the centre within 10 % of the direct
simulation, its scatter-to-primary ratio across the fan within 15 % (rms, over every column whose
ray crosses the phantom), and closer to the direct simulation than BasisSimulator's current model;
and the direct simulation precise enough to judge that (its own noise at the centre within 2 %).

The SPR across the fan used to be judged only where the scatter was above a fifth of its peak. That
left out the most attenuated rays, the very ones the library got wrong (rays longer than the WED,
which it shortened to fit), so the tall ellipse passed across the fan while its centre was off by a
factor of two. Every column inside the phantom now counts.
"""

# ╔═╡ 9c1da4a7-b08d-5601-950d-b06473a0e538
checks = vcat(
    MC.check_below(
        "the direct simulation's own noise at the centre (largest)",
        maximum(cmp.direct_noise for cmp in comparison),
        0.02;
        source = "32 independent chunks",
    ),
    [
        MC.check_below(
                "$(case.phantom.name), $(case.scanner): library, scatter at the centre",
                comparison[j].centre,
                0.1;
                source = "direct Monte Carlo",
            )
            for (j, case) in enumerate(cases)
    ],
    [
        MC.check_below(
                "$(case.phantom.name), $(case.scanner): library, SPR across the fan (rms)",
                comparison[j].library_rms,
                0.15;
                source = "direct Monte Carlo",
            )
            for (j, case) in enumerate(cases)
    ],
    [
        MC.check_below(
                "$(case.phantom.name), $(case.scanner): library error over BasisSimulator's",
                comparison[j].library_rms / comparison[j].bs_rms,
                1.0;
                source = "direct Monte Carlo",
            )
            for (j, case) in enumerate(cases)
    ],
)

# ╔═╡ e4d90fed-ae00-5719-8d9f-f4ad795e5bf4
Markdown.parse(MC.scorecard(checks))

# ╔═╡ f30d12f3-dc05-5b43-b022-6454cc112cdd
md"""
## Saving the Results

One file in `lib/BasisMonteCarlo/tables/`: every case's water-equivalent diameter, the three
scatter-to-primary profiles, the errors and the scorecard.
"""

# ╔═╡ 921f3ab8-b2ce-5f87-817b-12f511e52128
saved = let
    path = joinpath(table_dir, "scatter_validation_geant4.jls")
    Serialization.serialize(
        path,
        Dict{String, Any}(
            "cases" => [(phantom = case.phantom.name, scanner = case.scanner, wed_mm = projections[j].wed_mm) for (j, case) in enumerate(cases)],
            "comparison" => comparison,
            "thorax_outline_wed" => thorax_outline,
            "gammex_body" => gammex_body,
            "scorecard" => MC.scorecard(checks),
            "source" => "BasisMonteCarlo.jl (Geant4 via Geant4.jl), notebook 11_scatter_validation.jl",
        ),
    )
    path
end

# ╔═╡ Cell order:
# ╟─9e0038f2-df07-5bc8-b36f-25eae12e74eb
# ╟─6c441e5d-e5e6-59ee-ab46-e7bb118e71b5
# ╠═ddfb77a8-1acd-5fc1-90a3-81b04bd7216e
# ╠═3ac59922-7da3-5b41-8448-c49ea6db6ffa
# ╠═482f53df-95cc-5eff-8b49-9f50f561f1f2
# ╠═d8487463-34bf-5901-8cd4-869baad6d729
# ╠═5e0d4372-98f1-5e7d-aaf0-4275aceb5d04
# ╠═3a5cb03a-bff7-51fe-b9d0-04030865ec7d
# ╠═b42b632b-ed78-52a9-8661-c959957e4f4c
# ╠═10023f1f-dc34-5825-9fe8-618da7e842c5
# ╠═7d3fd247-14e0-5f8c-9ed6-868ba1e749a1
# ╠═b9beeb33-1e85-564b-8c42-f8ad1243ad1c
# ╠═62442d0b-ef4d-5cbf-8d1a-e99984d29c0c
# ╠═a6224573-fec9-5281-8cc4-3620e6868dcc
# ╠═5cfb96aa-3d6a-509a-bc47-89f3c7dafa85
# ╠═bdde2ceb-4db8-572a-82a9-f8fb295ba9e0
# ╠═17d722cd-bfe7-55b7-9543-462bec58e049
# ╠═997fd0c7-f40b-5436-87a1-6b85407c0c61
# ╠═9ae13765-5bf9-5316-9464-18955d2d91df
# ╟─df809b21-81f0-5d9e-971f-942df142a15f
# ╠═f5ff809e-9476-59fa-9718-dd4859116c80
# ╟─0ec9cb9c-ca21-5509-88c6-45802583bb6b
# ╟─af12453d-0101-5c03-94b3-11b34f6c2190
# ╠═3ddaaa30-9111-50c0-9836-f278f92809fa
# ╠═73a026f2-074f-5260-823f-6b36d1a3db31
# ╠═f980cfd3-0d1d-51dd-915c-88ca48712a60
# ╠═f7a2b1b9-2d25-5025-97ff-e9d379d4788a
# ╠═f67c80e6-ec4d-43e1-aaa7-b674a8ee268d
# ╠═56dcfda7-4d44-5201-8424-15fbdc7859cc
# ╠═26def491-0406-41b8-94a9-a816b529c240
# ╠═8a34c98b-99fc-5892-a874-9edad32772de
# ╠═a3a2c951-b97d-5e30-aae2-29f934c02655
# ╟─241b138e-6fce-5b2a-a217-71902cdf0f62
# ╠═f20ef946-92af-51a4-b778-2a29e23e4804
# ╠═6d28368c-4aae-566f-bdfc-140f80a10b98
# ╠═a569223d-d467-56fa-9122-3b32a5ca8d8f
# ╠═3855003e-b0ba-5371-b6a8-dd1c9a31da0e
# ╠═6c7a3173-cca3-5472-9586-44db7ccbdd14
# ╟─3777185c-2648-5803-abbd-12248ba7efc6
# ╟─674b51a1-4b66-5f67-9593-94903a24c841
# ╟─e67c62ed-d243-4525-ba94-b49b4c3a2b5d
# ╠═06281871-fe7c-4cf1-b442-c871693bf24e
# ╟─10db6197-a1f6-43cb-bcbf-7e1f33290858
# ╟─df233fa9-bedc-4605-9a7a-bb0fa36b193d
# ╠═46a9849d-76ec-4238-8d63-8fe8db2118cc
# ╟─8c16e306-3a3a-46b4-bc4a-6d08a6a7bab8
# ╟─0050e067-6aae-5e84-b4eb-1a6248de33ec
# ╠═9c1da4a7-b08d-5601-950d-b06473a0e538
# ╟─e4d90fed-ae00-5719-8d9f-f4ad795e5bf4
# ╟─f30d12f3-dc05-5b43-b022-6454cc112cdd
# ╠═921f3ab8-b2ce-5f87-817b-12f511e52128
