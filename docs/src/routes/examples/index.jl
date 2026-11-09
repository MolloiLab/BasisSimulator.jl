# Examples gallery — one card per Pluto notebook in docs/notebooks/, grouped by topic, and one per
# BasisMonteCarlo.jl notebook in lib/BasisMonteCarlo/docs/notebooks/ (slug `mc-<stem>`, under
# "Detector Physics"). Each card links to /examples/<slug>/, registered programmatically in
# docs/app.jl and served by NotebookPage(slug). A notebook without an entry in NOTEBOOK_META still
# gets a card (under "More"), so a new notebook never 404s; give it an entry and a group for a real
# description. A BasisMonteCarlo notebook not rendered yet (no export) gets an unlinked card, and its
# card image is the first figure of its export (docs/assets/mc-<stem>.webp, written by
# docs/extract_montecarlo.jl), a plain panel until then.

let BASE = get(ENV, "BASISSIM_BASE", ""),
    notebooks_dir = joinpath(@__DIR__, "..", "..", "..", "notebooks"),
    mc_notebooks_dir = joinpath(
        @__DIR__,
        "..",
        "..",
        "..",
        "..",
        "lib",
        "BasisMonteCarlo",
        "docs",
        "notebooks",
    ),
    static_dir = joinpath(@__DIR__, "..", "..", "..", "notebooks-static"),
    assets_dir = joinpath(@__DIR__, "..", "..", "..", "assets"),
    GROUPS = [
        (
            "getting-started",
            "Getting Started",
            "The API end to end, on one clinical scanner.",
            ["01_five_struct_api"],
        ),
        (
            "phantoms",
            "Phantoms & Anatomy",
            "Digital anatomy, custom materials, and mapping ground truth onto the reconstruction.",
            ["02_xcat_custom_materials", "05_xcat_grid_to_recon"],
        ),
        (
            "spectral",
            "Spectral & VMI",
            "Dual-kVp and photon-counting acquisitions through BS.vmi_pipeline, checked rod by rod against theory.",
            [
                "03_dual_kvp_switching_vmi",
                "04_pcct_vmi",
                "07_qrm_thorax_pure_material_vmi",
                "08_qrm_thorax_pure_material_pcct",
            ],
        ),
        (
            "scanners",
            "Scanner Models",
            "Dual-source Siemens systems with their own Monte Carlo detector tables, read out as mixed images and VMIs.",
            ["09_siemens_force_ufc_dual_source_vmi", "12_siemens_flash_ufc"],
        ),
        (
            "acquisition",
            "Acquisition, Artifacts & Validation",
            "Helical coverage, metal artifacts, and a head-to-head with CatSim.",
            ["11_helical_scanning", "10_titanium_implant", "06_catsim_vs_basissim"],
        ),
        (
            "detector-physics",
            "Detector Physics",
            "Geant4 Monte Carlo of the detectors and the scatter behind BasisSimulator's tables, each piece checked against documentation and the literature.",
            [
                "mc-01_cdte_slab_efficiency",
                "mc-02_benchmark_hajdok_swank",
                "mc-03_alpha_detector",
                "mc-04_siemens_force",
                "mc-05_siemens_flash",
                "mc-06_ge_revolution_apex",
                "mc-07_philips_brilliance_ict",
                "mc-08_canon_aquilion_one",
                "mc-09_scatter_water_cylinders",
                "mc-10_scatter_library",
                "mc-11_scatter_validation",
            ],
        ),
    ],
    NOTEBOOK_META = Dict(
        "01_five_struct_api" => (
            title = "The Five-Struct API",
            summary = "Five structs, then create_workspace, simulate! and reconstruct!: a Gammex 472 " *
                "scan on a GE Revolution Apex Elite model at 200 and 50 mA, the dose report of each, " *
                "FBP against hybrid IR, and checks of water, noise scaling and every rod against theory.",
            thumbnail = "recon_compare_4panel.png",
            tags = ["EICT", "FBP", "Hybrid IR", "Dose"],
        ),
        "02_xcat_custom_materials" => (
            title = "XCAT Anatomy + Custom Materials",
            summary = "An XCAT adult-male chest with an XrayAttenuation material for every organ, including an " *
                "iodinated blood mixture defined inline, scanned on the Apex Elite and reconstructed with " *
                "FBP and hybrid IR at strength 60 after the water beam-hardening correction.",
            thumbnail = "xcat_fbp_vs_hir.png",
            tags = ["XCAT", "Custom materials", "Hybrid IR"],
        ),
        "05_xcat_grid_to_recon" => (
            title = "XCAT Grids and the Affine Round-Trip",
            summary = "An XCAT 3.0 male chest export cropped to the heart and scanned axially and helically; " *
                "phantom_to_world_affine, recon_to_world_affine and resample_to_recon register the " *
                "ground-truth labels onto both reconstructions, with organ ROI statistics and an exactness audit.",
            thumbnail = "xcat_grid_overlay.png",
            tags = ["XCAT", "Affine", "Helical"],
        ),
        "03_dual_kvp_switching_vmi" => (
            title = "Dual-kVp Switching VMI",
            summary = "GE Apex Elite rapid kVp switching (80/140 kVp) on Gammex 472 through BS.vmi_pipeline: " *
                "n-channel profile-likelihood decomposition, SpectralHYPR in the projection and image " *
                "domains, ACNR, and VMIs at 50, 70, 100 and 140 keV checked rod by rod against theory.",
            thumbnail = "dual_kvp_vmi_projection_grid.png",
            tags = ["EICT", "Dual-kVp", "VMI", "SpectralHYPR"],
        ),
        "04_pcct_vmi" => (
            title = "Photon-Counting VMI",
            summary = "A Siemens NAEOTOM Alpha model at 140 kVp with four counting bins on Gammex 472: the per-ray " *
                "spectral_basis(ws; I0) of the simulation, one BS.vmi_pipeline call, and VMIs at 40, 70, 100 " *
                "and 140 keV verified rod by rod against theory.",
            thumbnail = "pcct_vmi_projection_grid.png",
            tags = ["PCCT", "VMI", "NAEOTOM Alpha"],
        ),
        "07_qrm_thorax_pure_material_vmi" => (
            title = "QRM Thorax: Pure-Material Dual-kVp VMI",
            summary = "A body-sized QRM thorax phantom, built analytically in the notebook, with four pure-material " *
                "rods (water, lipid, collagen, iodine), scanned by Apex Elite rapid kVp switching at " *
                "clinical resolution; BS.vmi_pipeline and a per-rod measured-versus-theoretical regression.",
            thumbnail = "qrm_thorax_vmi_vs_theoretical.png",
            tags = ["EICT", "Dual-kVp", "VMI", "QRM thorax"],
        ),
        "08_qrm_thorax_pure_material_pcct" => (
            title = "QRM Thorax: Pure-Material Photon-Counting VMI",
            summary = "The photon-counting counterpart of notebook 07: the same analytic thorax phantom and rods, " *
                "one 140 kVp NAEOTOM Alpha acquisition with pile-up and scatter correction in the detector " *
                "model, BS.vmi_pipeline, and the same per-rod regression.",
            thumbnail = "qrm_thorax_pcct_vmi_vs_theoretical.png",
            tags = ["PCCT", "VMI", "QRM thorax"],
        ),
        "09_siemens_force_ufc_dual_source_vmi" => (
            title = "Siemens SOMATOM Force: Dual-Source VMI",
            summary = "The third-generation dual-source Force with its UFC Gd₂O₂S Monte Carlo efficiency table. " *
                "One 100/Sn140 kV acquisition read out two ways: per-tube reconstructions with a " *
                "Siemens-style mixed image, and BS.vmi_pipeline VMIs checked per rod.",
            thumbnail = "force_ufc_detected_spectra.png",
            tags = ["Dual-source", "UFC", "Sn filter", "VMI"],
        ),
        "12_siemens_flash_ufc" => (
            title = "Siemens SOMATOM Definition Flash",
            summary = "The second-generation dual-source Flash with its own, thinner UFC crystal table. Regular " *
                "dual-power scanning at 120 kV (water accuracy and the √2 noise gain of two tubes) and " *
                "the clinical 80/Sn140 kV pair to a mixed image and BS.vmi_pipeline VMIs, closed by a " *
                "21-check verification gate.",
            thumbnail = "flash_ufc_lut_comparison.png",
            tags = ["Dual-source", "UFC Flash", "VMI"],
        ),
        "11_helical_scanning" => (
            title = "Helical Scanning",
            summary = "One keyword, pitch, turns an axial protocol into a spiral. A 30 cm slab from a 20 mm helix " *
                "at pitch 1.0 against three volume-axial stations (16 cm beam, 10 cm apart) at matched beam-width × current: " *
                "rebinned WFBP, CTDIvol and DLP of each, coronal reformats and z-profiles.",
            thumbnail = "helical_vs_stepshoot_coronal.png",
            tags = ["Helical", "WFBP", "Dose"],
        ),
        "10_titanium_implant" => (
            title = "Titanium Implant Artifacts",
            summary = "Titanium defined as a user material and two rods scanned in water at 120 kVp: the dark band " *
                "of beam hardening and the streaks of photon starvation through the standard water BHC and " *
                "FDK chain, measured against the same scan without the metal.",
            thumbnail = "titanium_artifacts.png",
            tags = ["Metal", "Custom material", "Beam hardening"],
        ),
        "06_catsim_vs_basissim" => (
            title = "CatSim vs BasisSimulator",
            summary = "The same Gammex 472 scan through CatSim/XCIST and BasisSimulator on the CPU and the GPU, " *
                "configured to the same beam and water BHC: images, differences, per-rod HU agreement " *
                "against theory, and the timing of each pipeline.",
            thumbnail = "catsim_vs_basissim_mosaic.png",
            tags = ["CatSim", "Validation", "Benchmark"],
        ),
        # BasisMonteCarlo.jl: the card image is the notebook's first figure (see the top of this file).
        "mc-01_cdte_slab_efficiency" => (
            title = "Detector Efficiency from First Principles: a CdTe Slab",
            summary = "Geant4 on the simplest detector, a slab of CdTe: photons followed step by step, a check " *
                "against Beer-Lambert, the energy budget photon by photon, and the efficiency η(E) the " *
                "escaping characteristic x rays take from Beer-Lambert.",
            thumbnail = "mc-01_cdte_slab_efficiency.webp",
            tags = ["Geant4", "PCCT", "CdTe", "Efficiency"],
        ),
        "mc-02_benchmark_hajdok_swank" => (
            title = "Benchmark: the Swank Factor of Four Detector Materials",
            summary = "The Monte Carlo against a published EGSnrc study (Hajdok, Battista and Cunningham, " *
                "Med. Phys. 2008): the Swank factor of amorphous silicon and selenium, cesium iodide " *
                "and lead iodide from 10 to 150 keV, panel by panel.",
            thumbnail = "mc-02_benchmark_hajdok_swank.webp",
            tags = ["Geant4", "Benchmark", "Swank factor"],
        ),
        "mc-03_alpha_detector" => (
            title = "The NAEOTOM Alpha Detector",
            summary = "The photon-counting detector built in Geant4 from one dexel up: grid, oblique photons, " *
                "2 × 2 binning, charge sharing, thresholds and pile-up, compared point by point with " *
                "BasisSimulator v0.15.1's MCNP response table.",
            thumbnail = "mc-03_alpha_detector.webp",
            tags = ["PCCT", "Geant4", "NAEOTOM Alpha"],
        ),
        "mc-04_siemens_force" => (
            title = "The SOMATOM Force Detector",
            summary = "The Force's UFC (Gd₂O₂S) dexels and 2-D grid in Geant4, each piece checked against " *
                "Siemens' documentation; the v0.15.1 efficiency table reproduced, and the new table, " *
                "with its Swank factor, built from the documented detector.",
            thumbnail = "mc-04_siemens_force.webp",
            tags = ["EID", "Geant4", "UFC", "Dual-source"],
        ),
        "mc-05_siemens_flash" => (
            title = "The SOMATOM Definition Flash Detector",
            summary = "The Flash's UFC detector and grid in Geant4, checked against Siemens' data sheets and " *
                "the FDA clearances; BasisSimulator v0.15.1's table reproduced, where it parts from the documented " *
                "detector, and a better one with its Swank factor.",
            thumbnail = "mc-05_siemens_flash.webp",
            tags = ["EID", "Geant4", "UFC", "Dual-source"],
        ),
        "mc-06_ge_revolution_apex" => (
            title = "The GE Gemstone Detector: Revolution Apex",
            summary = "GE's Gemstone garnet scintillator and 3D collimator in Geant4, checked against the FDA " *
                "clearances, GE's patents and the literature; BasisSimulator v0.15.1's table reproduced, and a new " *
                "one for a garnet that can exist.",
            thumbnail = "mc-06_ge_revolution_apex.webp",
            tags = ["EID", "Geant4", "Gemstone"],
        ),
        "mc-07_philips_brilliance_ict" => (
            title = "The Philips Brilliance iCT Detector",
            summary = "A scanner BasisSimulator does not have yet: the NanoPanel detector in Geant4 from what " *
                "is documented and stated assumptions, the beam fitted to measured half-value layers, " *
                "and an efficiency table for the iCT 256 and the iCT SP.",
            thumbnail = "mc-07_philips_brilliance_ict.webp",
            tags = ["EID", "Geant4", "Philips", "HVL fit"],
        ),
        "mc-08_canon_aquilion_one" => (
            title = "The Canon Aquilion ONE Detector: GENESIS and PRISM",
            summary = "A scanner BasisSimulator does not have yet: the PUREViSION detector in Geant4, the beam " *
                "fitted to measured half-value layers and checked on Canon's silver filter, and an " *
                "efficiency table for each edition.",
            thumbnail = "mc-08_canon_aquilion_one.webp",
            tags = ["EID", "Geant4", "Canon", "HVL fit"],
        ),
        "mc-09_scatter_water_cylinders" => (
            title = "Scatter in Water Cylinders",
            summary = "A fan beam through 20 to 50 cm water cylinders in Geant4, scored by forced detection " *
                "with no grid, a 1-D and a 2-D grid, against published measurements and " *
                "BasisSimulator v0.15.1's analytic model, closed by a scorecard.",
            thumbnail = "mc-09_scatter_water_cylinders.webp",
            tags = ["Scatter", "Geant4", "Anti-scatter grid", "Scorecard"],
        ),
        "mc-10_scatter_library" => (
            title = "A Scatter Library over Patient Shape",
            summary = "Geant4 pencil-beam scatter kernels in water ellipses 14 to 50 cm thick and 0.6 to 2.5 " *
                "times as wide, every ray and energy, scored on six scanners' detectors at once and shipped " *
                "to BasisSimulator per scanner; closed by a 52-check scorecard against direct Monte Carlo.",
            thumbnail = "mc-10_scatter_library.webp",
            tags = ["Scatter", "Geant4", "Ellipses", "Scorecard"],
        ),
        "mc-11_scatter_validation" => (
            title = "Validating the Scatter Library Beyond Water",
            summary = "Ellipses, an off-centre cylinder, Gammex 472 and two thoraxes (one held out) simulated " *
                "directly in Geant4, against the library and BasisSimulator v0.15.1's analytic model, on " *
                "a 2-D-grid and a 1-D-grid scanner; 37 of 37 checks pass.",
            thumbnail = "mc-11_scatter_validation.webp",
            tags = ["Scatter", "Geant4", "Thorax", "Validation"],
        ),
    )
    number(slug) = startswith(slug, "mc-") ?
        "MC " * first(split(chopprefix(slug, "mc-"), "_")) :
        first(split(slug, "_"))
    rendered(slug) = !startswith(slug, "mc-") ||
        isfile(joinpath(static_dir, "$(slug).fragment.html"))

    # Card builder — defined inside the `let` so Therapy's file-based router does not register it. A group's only
    # card is `wide`: it spans the row, its figure beside its text.
    notebook_card = function (slug::AbstractString, k::Integer; wide::Bool = false)
        meta = get(
            NOTEBOOK_META,
            slug,
            (
                title = replace(slug, r"^\d+_" => "", "_" => " "),
                summary = "",
                thumbnail = nothing,
                tags = String[],
            ),
        )
        live = rendered(slug)
        has_image = meta.thumbnail !== nothing &&
            isfile(joinpath(assets_dir, meta.thumbnail))
        children = (
            Div(
                :class =>
                    "bs-plate" *
                        (has_image ? "" : " bs-plate-empty") *
                        (wide ? " md:w-[56%] md:shrink-0 md:mb-2.5" : ""),
                has_image ?
                    Img(
                        :src => "$(BASE)/assets/$(meta.thumbnail)",
                        :alt => meta.title,
                        :loading => "lazy",
                        :class => "w-full aspect-[16/10] object-contain",
                    ) :
                    Div(:class => "w-full aspect-[16/10]"),
            ),
            Div(
                :class =>
                    "px-6 sm:px-7 pt-5 pb-6 flex-1 flex flex-col gap-3" *
                        (wide ? " md:justify-center md:px-9 md:py-8" : ""),
                Div(
                    :class => "flex items-baseline gap-3",
                    Span(
                        :class => "bs-numeral text-[1.75rem] leading-none shrink-0",
                        number(slug),
                    ),
                    H3(:class => "bs-tile-title bs-title text-[1.625rem]", meta.title),
                ),
                P(
                    :class => "bs-body text-[1.0625rem]" * (wide ? "" : " flex-1"),
                    meta.summary,
                ),
                Div(
                    :class => "flex flex-wrap items-center gap-1.5 pt-1",
                    [
                        Span(:class => "bs-pill bs-pill-xs bs-pill-static", tag)
                        for tag in meta.tags
                    ]...,
                ),
                Div(
                    :class => "bs-tile-more pt-1",
                    live ? "Open Notebook →" : "Not Rendered Yet",
                ),
            ),
        )
        style = "animation-delay:$(min(60 * (k - 1), 360))ms"
        # not rendered yet: no page to open, so the card is not a link
        return live ?
            A(
                :href => "$(BASE)/examples/$(slug)/",
                :class =>
                    "bs-tile bs-enter h-full" * (wide ? " md:col-span-2 md:flex-row" : ""),
                :style => style,
                children...,
            ) :
            Div(
                :class =>
                    "bs-tile bs-tile-off bs-enter h-full" *
                        (wide ? " md:col-span-2 md:flex-row" : ""),
                :style => style,
                children...,
            )
    end

    () -> begin
        # Honor BASISSIM_SKIP_NOTEBOOKS (app.jl registers no route for a skipped notebook); a
        # trailing `*` matches a prefix, as in app.jl.
        patterns = strip.(split(
            get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""),
            ",";
            keepempty = false,
        ))
        skipped = s -> any(
            p -> endswith(p, "*") ? startswith(s, chop(p)) : s == p,
            patterns,
        )
        present = isdir(notebooks_dir) ?
            Set(
                splitext(f)[1]
                for f in readdir(notebooks_dir)
                if endswith(f, ".jl") && !skipped(splitext(f)[1])
            ) :
            Set{String}()
        isdir(mc_notebooks_dir) &&
            union!(
                present,
                (
                    "mc-" * splitext(f)[1]
                    for f in readdir(mc_notebooks_dir)
                    if endswith(f, ".jl") && !skipped("mc-" * splitext(f)[1])
                )
            )
        grouped = Set(s for g in GROUPS for s in g[4])
        extra = sort([s for s in present if !(s in grouped)])
        groups = [(id, t, d, filter(in(present), slugs)) for (id, t, d, slugs) in GROUPS]
        isempty(extra) || push!(groups, ("more", "More", "", extra))
        groups = filter(g -> !isempty(g[4]), groups)

        Div(
            :class => "space-y-16 sm:space-y-20",
            Div(
                :class => "space-y-7",
                PageHeader(
                    "$(length(present)) Pluto notebooks",
                    ("Examples, ", Span(:class => "bs-flourish", "worked in full.")),
                    "Complete studies, each run end to end (on a GPU; the detector-physics notebooks on CPU workers): ",
                    "the code, the prose and every figure, rendered in place. Each page links to the notebook's source, ",
                    "which opens in Pluto.",
                ),
                # One row of section links per package: BasisSimulator.jl's notebooks, then BasisMonteCarlo.jl's
                Div(
                    :class => "space-y-3 pt-2",
                    [
                        Div(
                            :class => "flex flex-wrap items-center gap-2",
                            Span(
                                :class =>
                                    "bs-eyebrow text-[0.6875rem] w-full sm:w-44 shrink-0",
                                package,
                            ),
                            [
                                A(
                                    :href => "#$(id)",
                                    :class => "bs-pill bs-pill-sm",
                                    t,
                                    Span(:class => "bs-numeral", string(length(slugs))),
                                )
                                for (id, t, _, slugs) in rows
                            ]...,
                        )
                        for (package, rows) in (
                            (
                                "BasisSimulator.jl",
                                [g for g in groups if !all(startswith("mc-"), g[4])],
                            ),
                            (
                                "BasisMonteCarlo.jl",
                                [g for g in groups if all(startswith("mc-"), g[4])],
                            ),
                        )
                        if !isempty(rows)
                    ]...,
                ),
            ),
            if isempty(groups)
                Div(
                    :class => "bs-island py-16 text-center",
                    P(
                        :class => "bs-body",
                        "No notebooks yet. Drop a .jl in ",
                        Code(:class => "bs-code", "docs/notebooks/"),
                        " and rebuild.",
                    ),
                )
            else
                Div(
                    :class => "space-y-20 sm:space-y-24",
                    [
                        Section(
                            :id => id,
                            :class => "space-y-8 scroll-mt-28",
                            Div(
                                :class => "max-w-3xl space-y-3",
                                P(
                                    :class => "bs-eyebrow text-[0.6875rem]",
                                    all(startswith("mc-"), slugs) ?
                                        "BasisMonteCarlo.jl" :
                                        "BasisSimulator.jl",
                                ),
                                H2(:class => "bs-display text-4xl sm:text-5xl", t),
                                isempty(d) ?
                                    Fragment() :
                                    P(:class => "bs-body text-[1.1875rem]", d),
                            ),
                            Div(
                                :class => "grid grid-cols-1 md:grid-cols-2 gap-5 sm:gap-6",
                                [
                                    notebook_card(s, k; wide = length(slugs) == 1)
                                    for (k, s) in enumerate(slugs)
                                ]...,
                            ),
                        )
                        for (id, t, d, slugs) in groups
                    ]...,
                )
            end,
        )
    end
end
