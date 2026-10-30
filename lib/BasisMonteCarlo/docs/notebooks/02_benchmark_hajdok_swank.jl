### A Pluto.jl notebook ###
# v0.2.6

using Markdown
using InteractiveUtils

# ╔═╡ e73ba2f8-d658-5841-81a3-2a87c22483ad
md"""
# Benchmark 1: the Swank Factor of Four Detector Materials (Hajdok et al., EGSnrc)

Notebook 1 showed that the Monte Carlo is internally right: it reproduces Beer-Lambert with
Geant4's own cross sections and conserves energy photon by photon. This notebook checks it
against someone else's: a published, peer-reviewed Monte Carlo study run with a different code,
EGSnrc.

**The reference.** G. Hajdok, J. J. Battista and I. A. Cunningham, "Fundamental x-ray
interaction limits in diagnostic imaging detectors: Frequency-dependent Swank noise", *Medical
Physics* 35(7):3194 (2008), and Hajdok's PhD thesis (Western University, 2008), which holds the
same work with every figure. Their Figure 3.4 gives the **Swank factor** of four detector
materials, amorphous silicon, amorphous selenium, cesium iodide and lead iodide, from 10 to
150 keV.

**Why the Swank factor.** It measures how alike the energies left by detected photons are
(1 when every detected photon leaves the same energy). It is shaped by everything that makes
energy escape: K-fluorescence near every K-edge, Compton scattering at high energy. It is a
much sharper test of the detector physics than an efficiency, and it is exactly the quantity
`MC.swank_factor` computes from the per-photon records.

1. **The setup.** What they simulated, and how this notebook matches it.
2. **The materials and the thickness rule.** Each layer is as thick as it must be to stop 99 %
   of the photons, so the thickness changes with energy.
3. **The run.** Four materials, about a hundred energies each.
4. **The comparison.** Our Swank factors against theirs, panel by panel.
"""

# ╔═╡ 0c4f12bd-ce8f-54a0-81db-2703555bbbfd
md"""
## The Setup, Theirs and Ours

| | Hajdok et al. (EGSnrc, DOSRZnrc) | this notebook (Geant4) |
|---|---|---|
| beam | infinitely narrow pencil beam, normal incidence | the same |
| detector | uniform disc, 10 cm diameter, in vacuum | the same |
| thickness | `t = −ln(1 − η) / μ(E)` with η = 0.99 | the same, with Geant4's own μ |
| energies | 10–150 keV | 10–150 keV, every 2 keV, every 0.5 keV across the K-edges |
| photons per energy | 10 million | 100 000 (statistical error of the Swank factor ≈ 0.001) |
| photon interactions | photoelectric, bound Compton, Rayleigh | the same (Geant4 option 4: Livermore / Monash) |
| atomic relaxation | K, L, M fluorescence and Auger electrons | K, L, M fluorescence; Auger energy deposited where the vacancy was (following each Auger electron costs 3.4× the time and moves results by ~0.2 %) |
| electrons | transported down to 1 keV | transported, secondaries down to the 1 µm range cut |
| scored | energy deposited per photon, 0.5 keV bins | energy deposited per photon (unbinned) |
| Swank factor | `I = M₁² / (M₀ M₂)` of the absorbed-energy distribution | the same |

Two things are not stated in the thesis: the densities of a-Si, CsI and PbI₂ ("chosen to agree
with those used in practice"), and the cross-section library. The densities barely matter here:
with the thickness set by η, every length in the problem scales with 1/density, so the Swank
factor is the same at any density except through the fixed 10 cm disc. The comparison values are
read off their figure (the solid "0 cy/mm" curve, digitized at 300 dpi, uncertainty about
±0.005); the file `reference_data/hajdok2008_swank_vs_energy.csv` records how.
"""

# ╔═╡ 099436f2-777c-56a3-a8d2-4511d5ec98c3
md"""
## Notebook Setup
"""

# ╔═╡ 288da2d3-18a6-5144-8f35-479040757117
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ f50aa214-0bf2-59d7-ac58-bed477edbd14
using Markdown: @md_str, Markdown

# ╔═╡ 64463662-90d2-5c11-9abe-bca85bbc8ea2
import PlutoUI

# ╔═╡ f00150ef-2805-5c8e-9356-8fa1ae09e514
import BasisMonteCarlo as MC

# ╔═╡ 18897cc7-4249-581f-afb2-957f70fd46d0
import CairoMakie as Mke

# ╔═╡ 4bcddf7b-dbc6-5083-b062-2756e306657a
import DelimitedFiles

# ╔═╡ 165ee1c4-2db5-510d-bad8-8e8deeac0ae9
PlutoUI.TableOfContents()

# ╔═╡ a8bf1e62-22f9-56ae-bf3b-61a2b2ddebff
# Eight Geant4 processes run the sweeps in parallel; the rest of the machine stays free
workers = MC.start_workers(8)

# ╔═╡ 6e22da4f-5358-57fa-8ef2-c1af6f3e3186
# One colour per material, used by every figure
PALETTE = (
    materials = Dict(
        "a-Si" => Mke.RGBf(0.45, 0.45, 0.5),
        "a-Se" => Mke.RGBf(0.27, 0.55, 0.34),
        "CsI" => Mke.RGBf(0.13, 0.36, 0.66),
        "PbI2" => Mke.RGBf(0.7, 0.12, 0.17),
    ),
    reference = Mke.RGBf(0.9, 0.45, 0.05),
    ink = Mke.RGBf(0.13, 0.13, 0.16),
    muted = Mke.RGBf(0.42, 0.42, 0.46),
)

# ╔═╡ e98766b1-8cfa-5f24-9725-ef6c133c6a49
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

# ╔═╡ f952a231-979b-57e6-a76a-a9e7519db225
"""
An in-figure annotation on a translucent rounded background, so it never fights the data.
"""
function note!(
        ax, x, y, text;
        align = (:left, :center),
        color = PALETTE.ink,
        fontsize = 16,
        offset = (0, 0),
        alpha = 0.85,
    )
    return Mke.textlabel!(
        ax, Mke.Point2f(x, y);
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

# ╔═╡ 4d005a0b-2dea-5ef5-ad79-7549e327d67d
md"""
## 1. The Materials and the Thickness Rule

The four converters of the study, as compositions and densities. Amorphous selenium's density
is the one the thesis states (4.20 g/cm³, its Table 4.1); the others are the standard values.
"""

# ╔═╡ 99d52b8f-7b2e-5da6-8f70-a6cb688af871
materials = [
    MC.DetectorMaterial("a-Si"; density = 2.33, formula = "Si"),
    MC.DetectorMaterial("a-Se"; density = 4.2, formula = "Se"),
    MC.DetectorMaterial("CsI"; density = 4.51, formula = "CsI"),
    MC.DetectorMaterial("PbI2"; density = 6.16, formula = "PbI2"),
]

# ╔═╡ b3283488-a9dc-542e-b8dc-91d89ba1f8a7
let
    edges(m) = join(["$(e.element) $(round(e.edge_keV; digits = 1)) keV" for e in MC.k_edges(m)], ", ")
    row(m) = "| $(m.name) | $(m.density) g/cm³ | $(MC.composition_string(m)) | $(isempty(MC.k_edges(m)) ? "none above 5 keV" : edges(m)) |"

    Markdown.parse(
        """
        | material | density | composition by mass | K-edges in 10–150 keV |
        |---|---|---|---|
        $(join([row(m) for m in materials], "\n"))
        """
    )
end

# ╔═╡ 67fd055c-2545-5a5e-9deb-45b663953ec2
md"""
Each layer is made exactly thick enough to stop 99 % of the photons at the energy being
simulated, `t = −ln(1 − 0.99) / μ(E)`, with Geant4's own μ so that the Monte Carlo's interaction
probability lands on 0.99. The thickness therefore changes with energy by a factor of a
thousand: a few micrometres of selenium just above its K-edge, more than a decimetre of silicon
at 150 keV. Where the layer becomes thicker than the 10 cm disc is wide (silicon above about
100 keV), scattered photons start to escape through the sides, which is part of why silicon's
Swank factor falls at high energy.
"""

# ╔═╡ 17807a41-01a8-5eb4-8c36-c5e39afb3b16
η = 0.99

# ╔═╡ c5fc3da3-4ffb-5ae5-a49f-8cbe44079d42
disc_diameter_mm = 100.0

# ╔═╡ 8c745dc8-9df7-546e-89dd-ccba38c68762
"""
Thickness of `material` that stops a fraction `η` of photons of `energy_keV` (Geant4's μ).
"""
thickness_for(material, energy_keV) = -log(1 - η) / MC.geant4_attenuation(material, energy_keV)

# ╔═╡ 4a7e2280-57c6-5770-959f-c54ec2801627
energies = sort(
    unique(
        vcat(
            10.0:2.0:150.0,     # the whole range
            12.0:0.5:20.0,      # across the Se K-edge
            32.0:0.5:42.0,      # across the I and Cs K-edges
            86.0:0.5:94.0,      # across the Pb K-edge
        )
    )
)

# ╔═╡ 24397966-5f75-55f3-924d-de397cdc4aba
thicknesses = Dict(
    m.name => [thickness_for(m, E) for E in energies]
        for m in materials
)

# ╔═╡ 662e16e5-544c-5943-bc65-a9d61891471f
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 620))
    ax = Mke.Axis(
        fig[1, 1];
        title = "How thick each layer must be to stop 99 % of the photons",
        xlabel = "photon energy (keV)",
        ylabel = "layer thickness (mm)",
        yscale = log10,
        xticks = 10:10:150,
        yticks = (
            [0.1, 1, 10, 100],
            ["0.1 mm", "1 mm", "1 cm", "10 cm"],
        ),
        limits = ((10, 150), (0.03, 400)),
    )
    for m in materials
        Mke.lines!(
            ax, energies, thicknesses[m.name];
            color = PALETTE.materials[m.name],
            linewidth = 3,
            label = m.name,
        )
    end
    Mke.hlines!(
        ax,
        disc_diameter_mm;
        color = PALETTE.muted,
        linestyle = :dash,
        linewidth = 2,
    )
    note!(
        ax, 12, disc_diameter_mm, "width of the disc (10 cm)";
        align = (:left, :bottom),
        offset = (0, 4),
        color = PALETTE.muted,
        fontsize = 15,
    )
    note!(
        ax, 148, 0.04, "each drop is a K-edge: absorption jumps,\nso a thinner layer stops the same 99 %";
        align = (:right, :bottom),
        fontsize = 15,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ a470f299-d14b-5e6a-834d-701b3243e7ad
md"""
## 2. The Run

For every material and energy: build the disc, fire 100 000 photons down its axis, keep every
photon's record. About 450 runs, spread over the eight worker processes started in the setup.
"""

# ╔═╡ 7dac5424-8cef-585f-874a-f59e7ba40eed
photons_per_energy = 100_000

# ╔═╡ f367cd97-f4c2-579b-b88a-ab8eff65f1a6
# Every (disc, energy) pair of the benchmark, all materials in one list
jobs = [
    (MC.SlabDetector(m, t; width_mm = disc_diameter_mm, shape = :disc), E)
        for m in materials for (E, t) in zip(energies, thicknesses[m.name])
]

# ╔═╡ 79acd58a-bcae-5068-ab2b-22c638d3ddd4
all_runs = MC.simulate_sweep(
    jobs;
    n = photons_per_energy,
    seed = 1,
)

# ╔═╡ 0176f167-c47c-5d04-97d2-7108d632fdc9
runs = Dict(
    m.name => [r for r in all_runs if r.detector.material.name == m.name]
        for m in materials
)

# ╔═╡ 5053304a-fd29-5506-9d92-4da38b7a3d4b
swank = Dict(
    name => MC.swank_factor.(material_runs)
        for (name, material_runs) in runs
)

# ╔═╡ e1874dcb-0280-5c92-aa97-ef671d64d11c
let
    p = MC.interaction_probability.(all_runs)

    Markdown.parse(
        """
        Check: across all $(length(all_runs)) runs the fraction of photons that interacted is
        between **$(round(minimum(p); digits = 4))** and **$(round(maximum(p); digits = 4))**, as the
        thickness rule intends (η = $(η)).
        """
    )
end

# ╔═╡ dd5be4fa-0a4a-59c0-93c6-e1c5a7022d76
md"""
## 3. The Comparison

Their values, digitized from the figure, against ours. In the shaded band near 1 their curve
runs along the top edge of their plot, so it can only be read as "at least 0.99"; there the
check is that ours is too.
"""

# ╔═╡ a9f7d4e1-e769-5556-b92d-2b90bba2aed0
reference = let
    table = DelimitedFiles.readdlm(
        joinpath(@__DIR__, "..", "reference_data", "hajdok2008_swank_vs_energy.csv"),
        ',';
        comments = true,
        header = true,
    )[1]
    Dict(
        name => (
                energy_keV = Float64.(table[table[:, 1] .== name, 2]),
                swank = Float64.(table[table[:, 1] .== name, 3]),
            )
            for name in unique(table[:, 1])
    )
end

# ╔═╡ 7e09656c-77d4-5969-8a70-1ab5b01ea84e
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 980))
    for (i, m) in enumerate(materials)
        row, col = fldmod1(i, 2)
        color = PALETTE.materials[m.name]
        ax = Mke.Axis(
            fig[row, col];
            title = m.name,
            xlabel = row == 2 ? "photon energy (keV)" : "",
            ylabel = col == 1 ? "Swank factor" : "",
            xticks = 0:20:140,
            limits = ((0, 152), (0.7, 1.01)),
        )
        Mke.hspan!(ax, 0.99, 1.01; color = (PALETTE.muted, 0.12))
        Mke.lines!(
            ax,
            energies,
            swank[m.name];
            color,
            linewidth = 3,
            label = "Geant4 (this notebook)",
        )
        ref = reference[m.name]
        Mke.scatter!(
            ax, ref.energy_keV, ref.swank;
            color = (PALETTE.reference, 0.85),
            markersize = 6,
            label = "Hajdok et al. (EGSnrc), read off their Fig. 3.4",
        )
        for (j, e) in enumerate(MC.k_edges(m))
            Mke.vlines!(
                ax,
                e.edge_keV;
                color = (PALETTE.muted, 0.6),
                linestyle = :dash,
                linewidth = 1.5,
            )
            note!(
                ax, e.edge_keV, 0.72 + 0.035 * (j - 1), "$(e.element) K-edge";
                align = (:left, :bottom),
                offset = (4, 0),
                color = PALETTE.muted,
                fontsize = 14,
            )
        end
        i == 1 && note!(
            ax,
            150,
            0.995,
            "their curve reads ≥ 0.99 here";
            align = (:right, :center),
            color = PALETTE.muted,
            fontsize = 14,
        )
    end
    Mke.Legend(
        fig[3, 1:2],
        [
            Mke.LineElement(; color = PALETTE.ink, linewidth = 3),
            Mke.MarkerElement(;
                color = PALETTE.reference,
                marker = :circle,
                markersize = 10,
            ),
        ],
        [
            "Geant4, this notebook (line, one colour per material)",
            "Hajdok et al., EGSnrc (points, read off their figure)",
        ];
        orientation = :horizontal,
    )
    Mke.Label(
        fig[0, 1:2], "Swank factor of a layer that stops 99 % of the photons";
        fontsize = 22,
        font = :bold,
        halign = :left,
    )
    fig
end

# ╔═╡ 32a651e3-a493-54e5-9416-6a68359a65b7
md"""
The difference, ours minus theirs, at every energy they can be read: our curve is interpolated
onto their points. The grey band is their reading uncertainty (±0.005) combined with our
counting noise. The tinted strips mark the first 5 keV above each K-edge, where both curves
plunge almost vertically: there a reading error of a few tenths of a keV along the energy axis
of their figure becomes a large error in the Swank factor, so those points test the shape, not
the exact value.
"""

# ╔═╡ 256a630f-503e-5dd0-aa39-f9f67400ffb0
differences = Dict(
    m.name => let
            ref = reference[m.name]
            ours = [MC.interpolate_linear(energies, swank[m.name], E) for E in ref.energy_keV]
            (energy_keV = ref.energy_keV, difference = ours .- ref.swank)
    end
        for m in materials
)

# ╔═╡ 2a065f94-1937-5a38-8a08-1da9a8c2bdba
let
    Mke.set_theme!(THEME)
    fig = Mke.Figure(size = (1200, 520))
    ax = Mke.Axis(
        fig[1, 1];
        title = "Geant4 minus EGSnrc, at every point read off their figure",
        xlabel = "photon energy (keV)",
        ylabel = "difference in\nSwank factor",
        xticks = 0:10:150,
        limits = ((0, 152), (-0.05, 0.05)),
    )
    for m in materials, e in MC.k_edges(m)
        Mke.vspan!(
            ax,
            e.edge_keV - 0.5,
            e.edge_keV + 5;
            color = (PALETTE.materials[m.name], 0.08),
        )
    end
    Mke.hspan!(ax, -0.006, 0.006; color = (PALETTE.muted, 0.15))
    Mke.hlines!(ax, 0; color = PALETTE.ink, linewidth = 1.5)
    for m in materials
        d = differences[m.name]
        Mke.scatter!(
            ax,
            d.energy_keV,
            d.difference;
            color = PALETTE.materials[m.name],
            markersize = 6,
            label = m.name,
        )
    end
    note!(
        ax,
        150,
        0.006,
        "reading + counting uncertainty";
        align = (:right, :bottom),
        color = PALETTE.muted,
        fontsize = 15,
    )
    Mke.Legend(fig[1, 2], ax; valign = :center)
    fig
end

# ╔═╡ 9f940c62-c9bd-52d5-971a-38df8c5ff3bb
let
    function row(m)
        d = differences[m.name]
        ref = reference[m.name]
        i_ref = argmin(ref.swank)
        i_ours = argmin(swank[m.name])
        return string(
            "| $(m.name) | $(length(d.difference)) ",
            "| $(round(sum(abs, d.difference) / length(d.difference); digits = 4)) ",
            "| $(round(maximum(abs, d.difference); digits = 4)) ",
            "| $(round(ref.swank[i_ref]; digits = 3)) at $(ref.energy_keV[i_ref]) keV ",
            "| $(round(swank[m.name][i_ours]; digits = 3)) at $(energies[i_ours]) keV |",
        )
    end

    Markdown.parse(
        """
        | material | points compared | mean difference (absolute) | largest difference | lowest Swank, theirs | lowest Swank, ours |
        |---|---|---|---|---|---|
        $(join([row(m) for m in materials], "\n"))
        """
    )
end

# ╔═╡ 467c009d-9f05-536c-82bb-d4d6df3db5fc
let
    near_edge(m, E) = any(e -> e.edge_keV - 0.5 <= E <= e.edge_keV + 5, MC.k_edges(m))
    function row(m)
        d = differences[m.name]
        away = [x for (E, x) in zip(d.energy_keV, d.difference) if !near_edge(m, E)]
        near = [x for (E, x) in zip(d.energy_keV, d.difference) if near_edge(m, E)]
        fmt(v) = isempty(v) ? "–" : string(round(sum(v) / length(v); digits = 4))
        fmt_max(v) = isempty(v) ? "–" : string(round(maximum(abs, v); digits = 4))
        return "| $(m.name) | $(length(away)) | $(fmt(away)) | $(fmt_max(away)) | $(length(near)) | $(fmt_max(near)) |"
    end

    Markdown.parse(
        """
        | material | points away from the edges | mean difference there | largest there | points on the edges | largest there |
        |---|---|---|---|---|---|
        $(join([row(m) for m in materials], "\n"))
        """
    )
end

# ╔═╡ 1c82bb45-2704-5fe0-9e6d-ae72123f92af
md"""
## What the Comparison Shows

**Agreement.** Away from the K-edges, Geant4 and EGSnrc agree to within a few thousandths for
every material, over the whole 10–150 keV range and over Swank factors from 0.77 to 1. The
agreement covers three quite different pieces of physics:

- **Silicon**, where the Swank factor falls from 1 to 0.77 because Compton-scattered photons
  escape, at high energy partly through the *sides* of the 10 cm disc once the layer is thicker
  than the disc is wide. Reproducing that curve means reproducing the scattering and the
  geometry, not just the absorption.
- **CsI and a-Se**, where the dip just above each K-edge comes from K x rays escaping through
  the entrance face, and its recovery from the absorption moving deeper as the energy rises.
- **PbI₂**, with two edges (iodine and lead) and the lead L-edge structure near 13–16 keV.

**The one systematic difference.** Above both of its edges, our PbI₂ Swank factor sits about
0.007 below theirs, consistently, well outside the reading uncertainty. Both edges of PbI₂ are
the ones where the escaping x rays are most energetic (lead K x rays, 72–85 keV, travel far), so
the difference points at the K-fluorescence data or the treatment of the escaping x rays rather
than at the geometry. Candidates to test: the lead K-fluorescence yield and line energies
(Geant4's atomic relaxation data against the EADL-based tables of EGSnrc), Geant4's coherent
scattering of those x rays, and the PbI₂ density the authors used (not stated).

**On the edges.** The points within 5 keV above an edge differ by up to a few hundredths, both
up and down. These are where the curves are nearly vertical in their figure, so the reading is
limited by where exactly the edge falls on the energy axis of a scanned plot; the dip depths and
recoveries themselves match.

**Conclusion.** A different code, written independently, with different cross-section and
relaxation data, gives the same Swank factors to within a few thousandths. The detector physics
that matters for the efficiency of a CT detector, absorption, K-escape and scatter escape, is
reproduced. The PbI₂ offset is the one open item, and it is small.
"""

# ╔═╡ 631ad4da-2482-5ae7-ae23-6cf1ef01e062
md"""
## Physics Settings
"""

# ╔═╡ d8e74796-5aa6-53ac-8f49-75ef26ad8a92
MC.PhysicsSettings()

# ╔═╡ Cell order:
# ╟─e73ba2f8-d658-5841-81a3-2a87c22483ad
# ╟─0c4f12bd-ce8f-54a0-81db-2703555bbbfd
# ╟─099436f2-777c-56a3-a8d2-4511d5ec98c3
# ╠═288da2d3-18a6-5144-8f35-479040757117
# ╠═f50aa214-0bf2-59d7-ac58-bed477edbd14
# ╠═64463662-90d2-5c11-9abe-bca85bbc8ea2
# ╠═f00150ef-2805-5c8e-9356-8fa1ae09e514
# ╠═18897cc7-4249-581f-afb2-957f70fd46d0
# ╠═4bcddf7b-dbc6-5083-b062-2756e306657a
# ╠═165ee1c4-2db5-510d-bad8-8e8deeac0ae9
# ╠═a8bf1e62-22f9-56ae-bf3b-61a2b2ddebff
# ╠═6e22da4f-5358-57fa-8ef2-c1af6f3e3186
# ╠═e98766b1-8cfa-5f24-9725-ef6c133c6a49
# ╠═f952a231-979b-57e6-a76a-a9e7519db225
# ╟─4d005a0b-2dea-5ef5-ad79-7549e327d67d
# ╠═99d52b8f-7b2e-5da6-8f70-a6cb688af871
# ╟─b3283488-a9dc-542e-b8dc-91d89ba1f8a7
# ╟─67fd055c-2545-5a5e-9deb-45b663953ec2
# ╠═17807a41-01a8-5eb4-8c36-c5e39afb3b16
# ╠═c5fc3da3-4ffb-5ae5-a49f-8cbe44079d42
# ╠═8c745dc8-9df7-546e-89dd-ccba38c68762
# ╠═4a7e2280-57c6-5770-959f-c54ec2801627
# ╠═24397966-5f75-55f3-924d-de397cdc4aba
# ╟─662e16e5-544c-5943-bc65-a9d61891471f
# ╟─a470f299-d14b-5e6a-834d-701b3243e7ad
# ╠═7dac5424-8cef-585f-874a-f59e7ba40eed
# ╠═f367cd97-f4c2-579b-b88a-ab8eff65f1a6
# ╠═79acd58a-bcae-5068-ab2b-22c638d3ddd4
# ╠═0176f167-c47c-5d04-97d2-7108d632fdc9
# ╠═5053304a-fd29-5506-9d92-4da38b7a3d4b
# ╟─e1874dcb-0280-5c92-aa97-ef671d64d11c
# ╟─dd5be4fa-0a4a-59c0-93c6-e1c5a7022d76
# ╠═a9f7d4e1-e769-5556-b92d-2b90bba2aed0
# ╟─7e09656c-77d4-5969-8a70-1ab5b01ea84e
# ╟─32a651e3-a493-54e5-9416-6a68359a65b7
# ╠═256a630f-503e-5dd0-aa39-f9f67400ffb0
# ╟─2a065f94-1937-5a38-8a08-1da9a8c2bdba
# ╟─9f940c62-c9bd-52d5-971a-38df8c5ff3bb
# ╟─467c009d-9f05-536c-82bb-d4d6df3db5fc
# ╟─1c82bb45-2704-5fe0-9e6d-ae72123f92af
# ╟─631ad4da-2482-5ae7-ae23-6cf1ef01e062
# ╠═d8e74796-5aa6-53ac-8f49-75ef26ad8a92
