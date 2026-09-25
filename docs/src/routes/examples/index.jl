# Examples gallery — one card per Pluto notebook in docs/notebooks/, grouped by topic.
# Each card links to /examples/<slug>/, registered programmatically in docs/app.jl and served
# by NotebookPage(slug). A notebook without an entry in NOTEBOOK_META still gets a card (under
# "More"), so a new notebook never 404s; give it an entry and a group for a real description.

let BASE          = get(ENV, "BASISSIM_BASE", ""),
    notebooks_dir = joinpath(@__DIR__, "..", "..", "..", "notebooks"),
    GROUPS = [
        ("getting-started", "Getting started", "The API end to end, on one clinical scanner.",
            ["01_five_struct_api"]),
        ("phantoms", "Phantoms & anatomy", "Digital anatomy, custom materials, and mapping ground truth onto the reconstruction.",
            ["02_xcat_custom_materials", "05_xcat_grid_to_recon"]),
        ("spectral", "Spectral & VMI", "Dual-kVp and photon-counting acquisitions through BS.vmi_pipeline, checked rod by rod against theory.",
            ["03_dual_kvp_switching_vmi", "04_pcct_vmi", "07_qrm_thorax_pure_material_vmi", "08_qrm_thorax_pure_material_pcct"]),
        ("scanners", "Scanner models", "Dual-source Siemens systems with their own Monte Carlo detector tables, read out as mixed images and VMIs.",
            ["09_siemens_force_ufc_dual_source_vmi", "12_siemens_flash_ufc"]),
        ("acquisition", "Acquisition, artifacts & validation", "Helical coverage, metal artifacts, and a head-to-head with CatSim.",
            ["11_helical_scanning", "10_titanium_implant", "06_catsim_vs_basissim"]),
    ],
    NOTEBOOK_META = Dict(
        "01_five_struct_api" => (
            title     = "The Five-Struct API",
            summary   = "Five structs, then create_eict_workspace, simulate! and reconstruct!: a Gammex 472 " *
                        "scan on a GE Revolution Apex Elite model at 200 and 50 mA, the dose report of each, " *
                        "FBP against hybrid IR, and checks of water, noise scaling and every rod against theory.",
            thumbnail = "recon_compare_4panel.png",
            tags      = ["EICT", "FBP", "Hybrid IR", "Dose"],
        ),
        "02_xcat_custom_materials" => (
            title     = "XCAT Anatomy + Custom Materials",
            summary   = "An XCAT adult-male chest with an XrayAttenuation material for every organ, including an " *
                        "iodinated blood mixture defined inline, scanned on the Apex Elite and reconstructed with " *
                        "FBP and hybrid IR at strength 60 after the water beam-hardening correction.",
            thumbnail = "xcat_fbp_vs_hir.png",
            tags      = ["XCAT", "Custom materials", "Hybrid IR"],
        ),
        "05_xcat_grid_to_recon" => (
            title     = "XCAT Grids and the Affine Round-Trip",
            summary   = "An XCAT 3.0 male chest export cropped to the heart and scanned axially and helically; " *
                        "phantom_to_world_affine, recon_to_world_affine and resample_to_recon register the " *
                        "ground-truth labels onto both reconstructions, with organ ROI statistics and an exactness audit.",
            thumbnail = "xcat_grid_overlay.png",
            tags      = ["XCAT", "Affine", "Helical"],
        ),
        "03_dual_kvp_switching_vmi" => (
            title     = "Dual-kVp Switching VMI",
            summary   = "GE Apex Elite rapid kVp switching (80/140 kVp) on Gammex 472 through BS.vmi_pipeline: " *
                        "n-channel profile-likelihood decomposition, SpectralHYPR in the projection and image " *
                        "domains, ACNR, and VMIs at 50, 70, 100 and 140 keV checked rod by rod against theory.",
            thumbnail = "dual_kvp_vmi_projection_grid.png",
            tags      = ["EICT", "Dual-kVp", "VMI", "SpectralHYPR"],
        ),
        "04_pcct_vmi" => (
            title     = "Photon-Counting VMI",
            summary   = "A Siemens NAEOTOM Alpha model at 140 kVp with four counting bins on Gammex 472: the per-ray " *
                        "spectral_basis(ws; I0) of the simulation, one BS.vmi_pipeline call, and VMIs at 40, 70, 100 " *
                        "and 140 keV verified rod by rod against theory.",
            thumbnail = "pcct_vmi_projection_grid.png",
            tags      = ["PCCT", "VMI", "NAEOTOM Alpha"],
        ),
        "07_qrm_thorax_pure_material_vmi" => (
            title     = "QRM Thorax: Pure-Material Dual-kVp VMI",
            summary   = "A body-sized QRM thorax phantom, built analytically in the notebook, with four pure-material " *
                        "rods (water, lipid, collagen, iodine), scanned by Apex Elite rapid kVp switching at " *
                        "clinical resolution; BS.vmi_pipeline and a per-rod measured-versus-theoretical regression.",
            thumbnail = "qrm_thorax_vmi_vs_theoretical.png",
            tags      = ["EICT", "Dual-kVp", "VMI", "QRM thorax"],
        ),
        "08_qrm_thorax_pure_material_pcct" => (
            title     = "QRM Thorax: Pure-Material Photon-Counting VMI",
            summary   = "The photon-counting counterpart of notebook 07: the same analytic thorax phantom and rods, " *
                        "one 140 kVp NAEOTOM Alpha acquisition with pile-up and scatter correction in the detector " *
                        "model, BS.vmi_pipeline, and the same per-rod regression.",
            thumbnail = "qrm_thorax_pcct_vmi_vs_theoretical.png",
            tags      = ["PCCT", "VMI", "QRM thorax"],
        ),
        "09_siemens_force_ufc_dual_source_vmi" => (
            title     = "Siemens SOMATOM Force: Dual-Source VMI",
            summary   = "The third-generation dual-source Force with its UFC Gd₂O₂S Monte Carlo efficiency table. " *
                        "One 100/Sn140 kV acquisition read out two ways: per-tube reconstructions with a " *
                        "Siemens-style mixed image, and BS.vmi_pipeline VMIs checked per rod.",
            thumbnail = "force_ufc_detected_spectra.png",
            tags      = ["Dual-source", "UFC", "Sn filter", "VMI"],
        ),
        "12_siemens_flash_ufc" => (
            title     = "Siemens SOMATOM Definition Flash",
            summary   = "The second-generation dual-source Flash with its own, thinner UFC crystal table. Regular " *
                        "dual-power scanning at 120 kV (water accuracy and the √2 noise gain of two tubes) and " *
                        "the clinical 100/Sn140 kV pair to a mixed image and BS.vmi_pipeline VMIs, closed by a " *
                        "21-check verification gate.",
            thumbnail = "flash_ufc_lut_comparison.png",
            tags      = ["Dual-source", "UFC Flash", "VMI"],
        ),
        "11_helical_scanning" => (
            title     = "Helical Scanning",
            summary   = "One keyword, pitch, turns an axial protocol into a spiral. A 30 cm slab from a 20 mm helix " *
                        "at pitch 1.0 against three 16 cm volume-axial stations at matched beam-width × current: " *
                        "rebinned WFBP, CTDIvol and DLP of each, coronal reformats and z-profiles.",
            thumbnail = "helical_vs_stepshoot_coronal.png",
            tags      = ["Helical", "WFBP", "Dose"],
        ),
        "10_titanium_implant" => (
            title     = "Titanium Implant Artifacts",
            summary   = "Titanium defined as a user material and two rods scanned in water at 120 kVp: the dark band " *
                        "of beam hardening and the streaks of photon starvation through the standard water BHC and " *
                        "FDK chain, measured against the same scan without the metal.",
            thumbnail = "titanium_artifacts.png",
            tags      = ["Metal", "Custom material", "Beam hardening"],
        ),
        "06_catsim_vs_basissim" => (
            title     = "CatSim vs BasisSimulator",
            summary   = "The same Gammex 472 scan through CatSim/XCIST and BasisSimulator on the CPU and the GPU, " *
                        "configured to the same beam and water BHC: images, differences, per-rod HU agreement " *
                        "against theory, and the timing of each pipeline.",
            thumbnail = "catsim_vs_basissim_mosaic.png",
            tags      = ["CatSim", "Validation", "Benchmark"],
        ),
    )

    number(slug) = first(split(slug, "_"))

    # Card builder — defined inside the `let` so Therapy's file-based router does not register it.
    notebook_card = function (slug::AbstractString)
        meta = get(NOTEBOOK_META, slug, (
            title     = replace(slug, r"^\d+_" => "", "_" => " "),
            summary   = "",
            thumbnail = nothing,
            tags      = String[],
        ))
        A(:href => "$(BASE)/examples/$(slug)/", :class => "block group no-underline",
            Div(:class => "h-full flex flex-col border border-warm-200 dark:border-warm-800 rounded-xl overflow-hidden bg-warm-50 dark:bg-warm-900/40 hover:bg-white dark:hover:bg-warm-900/70 hover:border-accent-400 dark:hover:border-accent-600 transition-colors",
                if meta.thumbnail !== nothing
                    Div(:class => "bg-white border-b border-warm-200 dark:border-warm-800",
                        Img(:src => "$(BASE)/assets/$(meta.thumbnail)", :alt => meta.title, :loading => "lazy",
                            :class => "w-full aspect-[16/10] object-contain"))
                else
                    Div(:class => "w-full aspect-[16/10] bg-accent-100/50 dark:bg-accent-900/30")
                end,
                Div(:class => "p-6 space-y-3 flex-1 flex flex-col",
                    Div(:class => "flex flex-wrap items-center gap-2 text-[10px] tracking-[0.15em] uppercase font-mono",
                        Span(:class => "px-2 py-0.5 rounded bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300", number(slug)),
                        [Span(:class => "px-2 py-0.5 rounded border border-warm-300 dark:border-warm-700 text-warm-600 dark:text-warm-400", tag)
                         for tag in meta.tags]...),
                    H3(:class => "no-rule font-serif font-semibold text-xl text-warm-900 dark:text-warm-100 leading-snug group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors",
                        meta.title),
                    P(:class => "text-sm text-warm-600 dark:text-warm-400 leading-relaxed flex-1", meta.summary),
                    Div(:class => "pt-2 text-xs font-mono text-warm-500 group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors",
                        "Open notebook →"))))
    end

    () -> begin
        # Honor BASISSIM_SKIP_NOTEBOOKS (app.jl registers no route for a skipped notebook).
        skip = Set(strip.(split(get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""), ","; keepempty = false)))
        present = isdir(notebooks_dir) ?
            Set(splitext(f)[1] for f in readdir(notebooks_dir)
                if endswith(f, ".jl") && !(splitext(f)[1] in skip)) :
            Set{String}()
        grouped = Set(s for g in GROUPS for s in g[4])
        extra = sort([s for s in present if !(s in grouped)])
        groups = [(id, t, d, filter(in(present), slugs)) for (id, t, d, slugs) in GROUPS]
        isempty(extra) || push!(groups, ("more", "More", "", extra))
        groups = filter(g -> !isempty(g[4]), groups)

        Div(:class => "max-w-5xl mx-auto space-y-14",
            Div(:class => "space-y-4",
                Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500",
                    RawHtml("""<span style="color:var(--color-accent-500)">●</span>&nbsp; $(length(present)) Pluto notebooks""")),
                H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl leading-[1.05] tracking-tight",
                    "Examples"),
                P(:class => "max-w-2xl text-warm-600 dark:text-warm-400 leading-relaxed text-base",
                    "Complete studies, each run end to end on a GPU: the code, the prose and every figure, rendered ",
                    "in place. Each page links to the notebook's source, which opens in Pluto."),
                Div(:class => "flex flex-wrap gap-2 pt-2",
                    [A(:href => "#$(id)", :class => "px-3 py-1 rounded-full border border-warm-300 dark:border-warm-700 text-xs text-warm-700 dark:text-warm-300 hover:border-accent-400 hover:text-accent-600 dark:hover:text-accent-400 no-underline transition-colors",
                        "$(t) · $(length(slugs))") for (id, t, _, slugs) in groups]...)
            ),
            if isempty(groups)
                Div(:class => "py-16 border border-dashed border-warm-300 dark:border-warm-700 rounded-xl text-center",
                    P(:class => "text-warm-500 text-sm", "No notebooks yet. Drop a .jl in ",
                        Code(:class => "font-mono text-accent-600 dark:text-accent-400", "docs/notebooks/"), " and rebuild."))
            else
                Div(:class => "space-y-16",
                    [Section(:id => id, :class => "space-y-6 scroll-mt-24",
                        Div(:class => "flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-warm-200 dark:border-warm-800 pb-3",
                            H2(:class => "no-rule font-serif text-2xl md:text-3xl text-warm-900 dark:text-warm-100", t),
                            P(:class => "text-sm text-warm-500 dark:text-warm-400", d)),
                        Div(:class => "grid grid-cols-1 md:grid-cols-2 gap-6", [notebook_card(s) for s in slugs]...))
                     for (id, t, d, slugs) in groups]...)
            end
        )
    end
end
