# Examples gallery — auto-discovers Pluto notebooks from docs/notebooks/.
# Each card links to /examples/<slug>/, which is registered programmatically
# in docs/app.jl and served via NotebookPage(slug).
#
# Notebook metadata (title, summary, thumbnail) is hardcoded in this file for
# now.  When a new notebook is dropped in docs/notebooks/, add an entry to
# `NOTEBOOK_META` below and restart — no other code changes needed.

let BASE          = get(ENV, "BASISSIM_BASE", ""),
    notebooks_dir = joinpath(@__DIR__, "..", "..", "..", "notebooks"),
    NOTEBOOK_META = Dict(
        "01_five_struct_api" => (
            index     = "01",
            title     = "The Five-Struct API",
            summary   = "Walk the entire BasisSimulator surface — Phantom, Scanner, " *
                        "CTProtocol, SimOptions, ReconOptions — on the GE Revolution Apex Elite, " *
                        "with detected-spectrum water BHC, counts-domain detector noise, and cupping QA.",
            thumbnail = "recon_compare_4panel.png",
            tags      = ["EICT", "FBP", "GE Apex Elite"],
        ),
        "02_xcat_custom_materials" => (
            index     = "02",
            title     = "XCAT Phantom + Custom Materials",
            summary   = "Load a high-resolution XCAT voxel phantom and assign each region a " *
                        "tissue-specific XrayAttenuation.Material — including a custom iodinated " *
                        "blood mixture built inline. FBP vs Hybrid IR side-by-side.",
            thumbnail = "xcat_fbp_vs_hir.png",
            tags      = ["EICT", "FBP", "Hybrid IR", "XCAT"],
        ),
        "03_dual_kvp_switching_vmi" => (
            index     = "03",
            title     = "Dual-kVp Switching VMI on Gammex 472",
            summary   = "Fully projection-domain dual-energy pipeline on a GE Apex Elite GSI — " *
                        "joint sinogram SVD denoiser → bowtie-aware Cong material decomposition → " *
                        "FBP → z-median → Mono+ VMI at 40/70/100/140 keV, verified per-rod against " *
                        "XrayAttenuation theory.",
            thumbnail = "vmi_regression.png",
            tags      = ["EICT", "Dual-kVp", "VMI", "Mono+"],
        ),
        "04_pcct_vmi" => (
            index     = "04",
            title     = "PCCT VMI on Gammex 472",
            summary   = "Fully projection-domain photon-counting CT pipeline on a Siemens Naeotom " *
                        "Alpha — 4-bin sim → 4-channel SVD denoiser → bin combine → Cong univariate " *
                        "material decomposition (PCCT-generalized via the effective-spectral-response " *
                        "Φ_k(ε)) → FBP → z-median → Mono+ VMI at 40/70/100/140 keV, verified per-rod " *
                        "against XrayAttenuation theory.",
            thumbnail = "pcct_vmi_vs_theoretical.png",
            tags      = ["PCCT", "Naeotom Alpha", "VMI", "Mono+"],
        ),
        "05_xcat_grid_to_recon" => (
            index     = "05",
            title     = "XCAT UHR → CT Scan: Phantom Grids and the Affine Round-Trip",
            summary   = "Crop a 0.4 mm UHR XCAT down to a cardiac sub-region (the simulator's " *
                        "memory-efficient equivalent of scanner SFOV), then use phantom_to_world_affine + " *
                        "recon_to_world_affine + resample_to_recon to overlay ground-truth labels onto " *
                        "the HU recon — pixel-perfect, with :nearest / :linear / bring-your-own interpolation.",
            thumbnail = "xcat_grid_overlay.png",
            tags      = ["EICT", "XCAT", "Affine", "Resampling"],
        ),
        "06_catsim_vs_basissim" => (
            index     = "06",
            title     = "CatSim vs BasisSimulator (CPU + GPU) — qualitative + runtime",
            summary   = "Same Gammex 472 phantom (heavily downsampled), same GE Apex Elite scanner, three " *
                        "forward-projection + FDK pipelines: XCIST/CatSim (Python reference), BasisSim CPU, " *
                        "BasisSim GPU.  Side-by-side mid-slice mosaic + wallclock table — " *
                        "BasisSim matches the physics and lands well ahead on time.",
            thumbnail = "catsim_vs_basissim_mosaic.png",
            tags      = ["EICT", "CatSim", "GPU", "Benchmark"],
        ),
        "07_qrm_thorax_pure_material_vmi" => (
            index     = "07",
            title     = "QRM-Thorax Pure-Material VMI — Full-Resolution True-Scan Reference",
            summary   = "Canonical full-fidelity reference: clinical GE Apex Elite acquisition on a " *
                        "body-sized QRM-Thorax phantom (truly 0.2 mm isotropic ground truth, 0.625 mm " *
                        "isotropic recon, 2.5 mm DE collimation, 32 cm FOV) with four pure-material " *
                        "rod inserts — water, lipid, collagen, iodine 5 mg/mL.  Z trimmed to the " *
                        "cone-beam usable budget (3 slices) so the high-res forward projector runs " *
                        "in reasonable time.  Same dual-kVp pipeline as notebook 03: SVD denoiser → " *
                        "bowtie-aware Cong decomposition → FBP → z-median → Mono+ VMI at 40/70/100/140 " *
                        "keV, verified per-rod against XrayAttenuation theoretical curves.",
            thumbnail = "qrm_thorax_vmi_vs_theoretical.png",
            tags      = ["EICT", "Dual-kVp", "VMI", "Mono+", "QRM-Thorax"],
        ),
        "08_qrm_thorax_pure_material_pcct" => (
            index     = "08",
            title     = "QRM-Thorax Pure-Material PCCT — Full-Resolution True-Scan Reference",
            summary   = "The photon-counting mirror of notebook 07: Siemens Naeotom Alpha PCCT " *
                        "acquisition on the body-sized QRM-Thorax with the same four pure-material " *
                        "rods.  Full PCCT physics inside simulate!() (MC-LUT detector response, " *
                        "MC pile-up + correction, scatter + correction) → 4→2 count-domain bin " *
                        "combine → Cong-Φ_k decomposition → FBP → data-adaptive cov-ACNR (the " *
                        "VMI-noise-U killer) → z-median → 2-basis VMI at 40/70/100/140 keV, " *
                        "verified per-rod against XrayAttenuation theory.",
            thumbnail = "qrm_thorax_pcct_vmi_vs_theoretical.png",
            tags      = ["PCCT", "Naeotom Alpha", "VMI", "cov-ACNR", "QRM-Thorax"],
        ),
        "11_helical_scanning" => (
            index     = "11",
            title     = "Helical Scanning — Narrow Collimation, Long Coverage",
            summary   = "One new kwarg — pitch — turns any protocol into a spiral scan.  A 32 cm " *
                        "z-slab captured with a NARROW 20 mm collimation (pitch 1.0 × 16 rotations, " *
                        ":dd_fast projector, rebinned-WFBP helical recon) vs classic step-and-shoot " *
                        "(8 axial stations at the scanner-max 40 mm collimation), matched exposure, " *
                        "full nb01 correction stack.  A z-varying low-Z phantom (helically winding " *
                        "lung rod + tapering adipose cone) and a PlutoUI z-slider show phantom truth " *
                        "against both recons slice by slice — helical holds water flat across the " *
                        "whole slab while the stitched stations show their seams.",
            thumbnail = "helical_vs_stepshoot_coronal.png",
            tags      = ["EICT", "Helical", "Pitch", "dd_fast", "FDK"],
        ),
        "10_titanium_implant" => (
            index     = "10",
            title     = "Titanium Implant — Metal Artifacts from a User-Defined Material",
            summary   = "Register titanium as a custom XA.Material (pure Ti, 4.54 g/cm³, NIST " *
                        "XCOM cross-sections) and scan two 1.5 cm rods in a water cylinder at " *
                        "120 kVp.  The polychromatic forward model and count-domain noise chain " *
                        "produce the classic metal artifacts by construction — a between-rod " *
                        "beam-hardening dark band and photon-starvation streaks — reconstructed " *
                        "uncorrected with FDK.  States the no-MAR scope explicitly.",
            thumbnail = "titanium_artifacts.png",
            tags      = ["EICT", "Metal artifacts", "Custom material", "Beam hardening", "FDK"],
        ),
        "09_siemens_force_ufc_dual_source_vmi" => (
            index     = "09",
            title     = "Siemens SOMATOM Force — UFC MC LUT + Dual-Source VMI",
            summary   = "First outing of the second EICT MC-LUT detector: the Siemens UFC " *
                        "(Gd₂O₂S:Pr,Ce) scintillator on the third-generation dual-source Force, " *
                        "with the Gd K-edge fluorescence-escape cliff baked into η(E).  One " *
                        "100 kVp / Sn140 (0.6 mm tin) DE acquisition feeds both readouts — " *
                        "per-tube η-aware BHC recons + Siemens-style mixed image, and a " *
                        "projection-domain Cong → cov-ACNR → VMI chain at 50/70/100/140 keV, " *
                        "verified per-rod against XrayAttenuation theory.",
            thumbnail = "force_ufc_detected_spectra.png",
            tags      = ["EICT", "Dual-Source", "UFC MC LUT", "Sn filter", "VMI"],
        ),
    )

    # Card builder — defined inside the `let` so Therapy's file-based router
    # doesn't try to register it as a `/examples/_NotebookCard/` route.
    notebook_card = function(slug::AbstractString)
        meta = get(NOTEBOOK_META, slug, (
            index     = "—",
            title     = replace(slug, "_" => " "),
            summary   = "",
            thumbnail = nothing,
            tags      = String[],
        ))

        A(:href => "$(BASE)/examples/$(slug)/",
            :class => "block group no-underline",
            Div(:class => "h-full border border-warm-200 dark:border-warm-800 rounded-xl overflow-hidden bg-warm-50 dark:bg-warm-900/40 hover:bg-white dark:hover:bg-warm-900/70 hover:border-accent-400 dark:hover:border-accent-600 transition-colors",

                # Thumbnail (asset PNG) — falls back to a tinted accent block
                if meta.thumbnail !== nothing
                    Img(:src => "$(BASE)/assets/$(meta.thumbnail)",
                        :alt => meta.title,
                        :class => "w-full aspect-[16/10] object-cover bg-warm-200 dark:bg-warm-800")
                else
                    Div(:class => "w-full aspect-[16/10] bg-accent-100/50 dark:bg-accent-900/30")
                end,

                # Body
                Div(:class => "p-6 space-y-3",
                    Div(:class => "flex items-center gap-2 text-[10px] tracking-[0.2em] uppercase font-mono",
                        Span(:class => "px-2 py-0.5 rounded bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300",
                            meta.index),
                        [Span(:class => "px-2 py-0.5 rounded border border-warm-300 dark:border-warm-700 text-warm-600 dark:text-warm-400",
                            tag) for tag in meta.tags]...
                    ),
                    H3(:class => "no-rule font-serif font-semibold text-xl text-warm-900 dark:text-warm-100 leading-snug group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors",
                        meta.title),
                    P(:class => "text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                        meta.summary),
                    Div(:class => "pt-2 text-xs font-mono text-warm-500 dark:text-warm-500 group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors",
                        "Open notebook →")
                )
            )
        )
    end

    () -> begin
        # Discover slugs at render time so a fresh notebook drop shows up
        # without code changes here (other than its NOTEBOOK_META entry).
        # Honor BASISSIM_SKIP_NOTEBOOKS so cards don't 404 when a notebook
        # was skipped during render.
        skip = Set(strip.(split(get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""), ","; keepempty = false)))
        slugs = isdir(notebooks_dir) ?
            sort([
                splitext(f)[1] for f in readdir(notebooks_dir)
                if endswith(f, ".jl") && !endswith(f, ".sessions.toml") && !(splitext(f)[1] in skip)
            ]) :
            String[]

        Div(:class => "max-w-5xl mx-auto space-y-12",

            # Page header
            Div(:class => "space-y-4",
                Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500",
                    RawHtml("""<span style="color:var(--color-accent-500)">●</span>&nbsp; Pluto notebooks · auto-rendered""")
                ),
                H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl leading-[1.05] tracking-tight",
                    "Examples"
                ),
                P(:class => "max-w-2xl text-warm-600 dark:text-warm-400 leading-relaxed text-base",
                    "Worked examples covering the full BasisSimulator.jl pipeline. ",
                    "Each notebook runs end-to-end against a simulated phantom and renders ",
                    "publication-quality figures. Pluto-rendered, statically embedded — open one ",
                    "to see the full code, prose, and outputs in place."
                )
            ),

            # Notebook grid
            if isempty(slugs)
                Div(:class => "py-16 border border-dashed border-warm-300 dark:border-warm-700 rounded-xl text-center",
                    P(:class => "text-warm-500 dark:text-warm-500 text-sm",
                        "No notebooks yet. Drop a .jl in ",
                        Code(:class => "font-mono text-accent-600 dark:text-accent-400", "docs/notebooks/"),
                        " and rebuild.")
                )
            else
                Div(:class => "grid grid-cols-1 md:grid-cols-2 gap-6",
                    [notebook_card(slug) for slug in slugs]...
                )
            end
        )
    end
end
