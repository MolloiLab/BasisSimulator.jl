# Landing page — what BasisSimulator.jl is, what it does, a runnable quick start, and where to go.
# Figures are the ones the example notebooks write to docs/assets/.

let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
    eyebrow  = "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 mb-8"
    h2_cls   = "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl lg:text-6xl leading-[1.08] tracking-tight"
    lead_cls = "text-warm-600 dark:text-warm-400 leading-relaxed mt-6 text-base"
    card_cls = "border border-warm-200 dark:border-warm-800 rounded-xl p-6 bg-warm-50 dark:bg-warm-900/40 hover:bg-white dark:hover:bg-warm-900/70 transition-colors"
    inline   = "font-mono text-[0.85em] text-accent-700 dark:text-accent-300"

    section_head(n, label, title, accent, lead) = Div(:class => "max-w-3xl mb-14",
        Div(:class => eyebrow,
            RawHtml("""<span style="color:var(--color-accent-500)">§ $(n)</span>&nbsp;&nbsp;$(label)""")),
        H2(:class => h2_cls, title, Br(),
            RawHtml("""<span style="color:var(--color-accent-500)" class="italic">$(accent)</span>""")),
        P(:class => lead_cls, lead))

    feature(tag, title, body, apis) = Div(:class => card_cls,
        Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-accent-600 dark:text-accent-400 mb-3", tag),
        H3(:class => "no-rule font-serif text-xl mb-2 text-warm-900 dark:text-warm-100 leading-snug", title),
        P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed", body),
        Div(:class => "mt-4 flex flex-wrap gap-1.5",
            [Code(:class => "px-1.5 py-0.5 rounded bg-warm-200/60 dark:bg-warm-800/60 text-[11px] font-mono text-warm-700 dark:text-warm-300", a) for a in apis]...))

    stat(n, a, b) = Div(
        Div(:class => "font-serif text-5xl md:text-6xl text-warm-900 dark:text-warm-100 leading-none", n),
        Div(:class => "mt-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 leading-tight", a, Br(), b))

    figure(img, alt, caption, slug) = A(:href => "$(BASE)/examples/$(slug)/", :class => "group block no-underline",
        Div(:class => "rounded-xl overflow-hidden border border-warm-200 dark:border-warm-800 bg-white shadow-sm group-hover:border-accent-400 dark:group-hover:border-accent-600 transition-colors",
            Img(:src => "$(BASE)/assets/$(img)", :alt => alt, :loading => "lazy", :class => "w-full aspect-[16/10] object-contain bg-white")),
        P(:class => "mt-3 text-sm text-warm-600 dark:text-warm-400 leading-relaxed group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors", caption))

    Div(
        # ═══════════════════════════════════════════════════════════════════
        # HERO + STATS — one viewport-fit block on md+ screens: 4rem nav + 6rem
        # of MainEl padding are subtracted; the hero takes `flex-1 min-h-0` so the
        # stats row sits at the bottom, and the collage is clipped vertically but
        # may bleed horizontally past the page width.
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "relative md:h-[calc(100vh-10rem)] flex flex-col overflow-y-clip overflow-x-visible",
        Div(:class => "flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 pt-4 lg:pt-8 overflow-y-clip overflow-x-visible",

            # ─── Left: copy (z above the cards, blurred backdrop) ─────────
            Div(:class => "lg:col-span-7 space-y-8 relative z-[35] backdrop-blur-md",
                Div(:class => "flex flex-wrap items-center gap-3 text-[10px] tracking-[0.2em] uppercase font-mono",
                    Span(:class => "px-3 py-1 rounded-full bg-accent-600 text-white",
                        RawHtml("""<span class="opacity-80">●</span>&nbsp; Open Source · MIT""")),
                    Span(:class => "px-3 py-1 rounded-full border border-warm-300 dark:border-warm-700 text-warm-600 dark:text-warm-400",
                        RawHtml("""v0.15&nbsp;·&nbsp;Julia&nbsp;1.12"""))
                ),
                H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-5xl md:text-6xl lg:text-7xl leading-[1.05] tracking-tight",
                    "Polychromatic CT, ",
                    RawHtml("""<span style="color:var(--color-accent-500)" class="italic font-semibold">end-to-end</span>"""),
                    ", on any ",
                    RawHtml("""<span class="relative inline-block">GPU.<span class="absolute left-0 -bottom-1 h-[3px] w-full" style="background:var(--color-accent-500)"></span></span>""")
                ),
                P(:class => "max-w-xl text-warm-600 dark:text-warm-400 leading-relaxed text-base",
                    "BasisSimulator.jl simulates energy-integrating and photon-counting CT scanners from the source ",
                    "spectrum to the detector counts, and reconstructs what they measure: ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "FDK"), ", helical ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "WFBP"), ", hybrid ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "IR"), ", and ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "virtual monoenergetic images"),
                    " from dual-kVp, dual-source and photon-counting data, with the dose of every scan. ",
                    "One source tree runs on CUDA, Metal, ROCm, oneAPI and the CPU."
                ),
                Div(:class => "flex flex-wrap gap-3 pt-2",
                    A(:href => "$(BASE)/getting-started/",
                        :class => "px-5 py-2.5 bg-accent-600 hover:bg-accent-700 text-white rounded-md text-sm font-medium tracking-wide transition-colors no-underline",
                        "Get started →"),
                    A(:href => "$(BASE)/examples/",
                        :class => "px-5 py-2.5 border border-warm-300 dark:border-warm-700 rounded-md text-sm font-medium text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-900 transition-colors no-underline",
                        "Examples"),
                    A(:href => "https://github.com/MolloiLab/BasisSimulator.jl", :target => "_blank",
                        :class => "px-5 py-2.5 border border-warm-300 dark:border-warm-700 rounded-md text-sm font-medium text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-900 transition-colors no-underline",
                        "GitHub")
                )
            ),

            # ─── Right: image-card collage (z-10/20/30, under the nav's z-40) ─
            Div(:class => "lg:col-span-5 relative h-[520px] lg:h-[640px] mt-4 lg:mt-0 overflow-visible",
                Div(:class => "absolute top-2 right-2 z-[35] w-16 h-16 rounded-full flex items-center justify-center text-[9px] tracking-[0.15em] uppercase font-mono text-white text-center leading-tight shadow-lg",
                    :style => "background:radial-gradient(circle at 30% 30%, var(--color-accent-400), var(--color-accent-600) 70%, var(--color-accent-700));",
                    RawHtml("Open<br/>Source")),
                Img(:src   => "$(BASE)/assets/gammex_472_phantom.png",
                    :alt   => "Gammex Model 472 phantom label map",
                    :class => "absolute top-[-3%] left-[-4%] w-[82%] h-auto block rounded-xl border border-warm-300 dark:border-warm-800 shadow-2xl -rotate-3 z-10"),
                Img(:src   => "$(BASE)/assets/xcat_fbp_vs_hir.png",
                    :alt   => "XCAT chest reconstructed with FDK and with hybrid IR",
                    :class => "absolute top-[30%] right-[-12%] w-[80%] h-auto block rounded-xl border border-warm-300 dark:border-warm-800 shadow-2xl rotate-[3deg] z-20"),
                Img(:src   => "$(BASE)/assets/recon_compare_4panel.png",
                    :alt   => "Gammex 472 reconstructed with FBP and with hybrid IR at 200 mA and at 50 mA",
                    :class => "absolute bottom-[-4%] left-[-3%] w-[72%] h-auto block rounded-xl border border-warm-300 dark:border-warm-800 shadow-2xl -rotate-[5deg] z-30")
            )
        ),

        # ─── Stats — pinned to the bottom of the viewport-fit block ───────
        Div(:class => "shrink-0 mt-16 lg:mt-0 pt-8 lg:pt-10 pb-2 border-t border-warm-300 dark:border-warm-800 relative z-[35] backdrop-blur-md",
            Div(:class => "grid grid-cols-2 md:grid-cols-4 gap-y-8 gap-x-6",
                stat("02", "Detector", "families"),
                stat("05", "Structs to", "a scan"),
                stat("04", "Clinical scanner", "models"),
                stat("12", "Worked", "notebooks"))
        ),
        ),

        # ═══════════════════════════════════════════════════════════════════
        # §02 — QUICK START
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-32 lg:mt-48",
            section_head("02", "Quick start", "Phantom → sinogram → image,", "in five structs.",
                "A phantom, a scanner, a protocol, the physics and the output grid. The workspace allocates once and is reused on every call; the dose report comes with the scan."),
            CodeBlock("""import BasisSimulator as BS, GPUSelect
to_gpu = GPUSelect.Storage()           # CuArray, MtlArray, ROCArray, oneArray, or Array

cpu      = BS.create_gammex_472(n_voxels = 256, n_slices = 4, z_cm = 1.0)
phantom  = BS.Phantom(to_gpu(cpu.mask), cpu.materials, cpu.voxel_size, cpu.origin, cpu.extent)
scanner  = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0,
                          detector_rows = 32, detector_cols = 834,
                          detector_row_size = 0.625, detector_col_size = 0.6)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = 500, collimation_mm = 5.0)
sim_opts = BS.SimOptions()
rec_opts = BS.ReconOptions(matrix_size = (512, 512, 4), fov_cm = 35.0, z_cm = 0.5)

ws   = BS.create_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
dose = BS.simulate!(ws, phantom, protocol, sim_opts).dose        # CTDIvol, DLP
bhc  = BS.calibrate_bhc_water(sim_opts, protocol; scanner, geom = ws.geom)
sino = to_gpu(BS.apply_bhc_water(ws.sinogram, bhc))
fdk  = BS.create_fdk_recon_workspace(sino, ws.geom, rec_opts.matrix_size)
hu   = BS.to_hounsfield(Array(BS.reconstruct!(fdk, sino, ws.geom)); μ_water = bhc.μ_water_ref)"""),
            Div(:class => "mt-6 flex justify-end",
                A(:href => "$(BASE)/getting-started/",
                    :class => "text-sm text-warm-600 dark:text-warm-400 hover:text-accent-600 dark:hover:text-accent-400 no-underline",
                    "Full walkthrough, with photon counting and VMI →"))
        ),

        # ═══════════════════════════════════════════════════════════════════
        # §03 — CAPABILITIES
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-32 lg:mt-48",
            section_head("03", "What's inside", "Simulation, reconstruction,", "spectral imaging.",
                "Every effect a clinical acquisition applies is modelled where the scanner applies it, and every reconstruction reads the same geometry the simulation wrote."),
            Div(:class => "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5",
                feature("Detectors", "Energy-integrating and photon-counting",
                    "Scintillators weighted by Monte Carlo efficiency tables for GE Gemstone and the Siemens Force and Definition Flash UFC crystals. CdTe, CZT or Si counting through a Monte Carlo detector response, with charge sharing, K-fluorescence, Monte Carlo pile-up and model-based pile-up and scatter correction.",
                    ["EICTScanner", "PCCTScanner"]),
                feature("Physics", "Polychromatic from the spectrum up",
                    "IPEM tungsten spectra, flat and added filters (Al, Ti, Sn, Cu), CatSim bowties, heel effect, focal-spot blur, fill factor, lag and scatter, then quantum and electronic noise drawn in the counts, before the log.",
                    ["SimOptions", "CTProtocol"]),
                feature("Projectors", "One volume walk, the whole spectrum",
                    "The distance-driven :dd_fast projector integrates every energy in a single pass over arc or flat detectors with the quarter-detector offset; Siddon is kept for comparison. Path lengths can be cached and reused across tube voltages.",
                    [":dd_fast", "material_paths"]),
                feature("Reconstruction", "FDK, helical WFBP, hybrid IR",
                    "Cone-beam FDK, rebinned WFBP chosen automatically when the protocol has a pitch, and hybrid iterative reconstruction with one strength dial from 0 (FBP) to 100. Water beam hardening is calibrated from the detected spectrum itself.",
                    ["reconstruct!", "calibrate_bhc_water"]),
                feature("Spectral", "VMI from any spectral acquisition",
                    "Photon-counting bins, rapid kVp switching and dual-source pairs all go through one n-channel profile-likelihood water–iodine decomposition (or Cong's estimator), per-basis reconstruction, ACNR and virtual monoenergetic synthesis in HU.",
                    ["vmi_pipeline", "spectral_basis"]),
                feature("Denoising", "SpectralHYPR",
                    "Generalized HYPR-LR in two places: on the counts of each detector row before the decomposition, and on the reconstructed basis pair. Also T-LBF for photon counting and ACNR.",
                    ["SpectralHYPR", "hypr_lr", "image_hypr"]),
                feature("Dose", "CTDIvol and DLP of every scan",
                    "Monte Carlo of the simulated beam in the 16 and 32 cm CTDI phantoms: CTDI100 at the centre and periphery, CTDIw, CTDIvol and DLP, attached to every simulate! result.",
                    ["result.dose", "compute_dose"]),
                feature("Phantoms", "Calibration phantoms and anatomy",
                    "Gammex 472, XCAT anatomies downloaded on demand, any label map with any XrayAttenuation material, and affine maps between the phantom, world and reconstruction grids for ground-truth overlays.",
                    ["create_gammex_472", "load_xcat_male_chest", "resample_to_recon"]),
                feature("Portability", "Written once, for every backend",
                    "Kernels are written against AcceleratedKernels.jl; GPUSelect picks the device. Workspaces allocate every buffer once, so repeated scans do not allocate.",
                    ["GPUSelect.Storage", "create_workspace"]),
            )
        ),

        # ═══════════════════════════════════════════════════════════════════
        # §04 — FROM THE NOTEBOOKS
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-32 lg:mt-48",
            section_head("04", "From the notebooks", "Figures the examples", "draw themselves.",
                "Each figure below is written by an example notebook when it runs. Open one to read the code that made it."),
            Div(:class => "grid grid-cols-1 md:grid-cols-2 gap-8",
                figure("xcat_fbp_vs_hir.png", "XCAT chest, FDK beside hybrid IR at strength 60",
                    "An XCAT chest with a material per organ, FDK beside hybrid IR at strength 60.", "02_xcat_custom_materials"),
                figure("flash_ufc_vmi_grid.png", "Virtual monoenergetic images of Gammex 472 at 50, 70, 100 and 140 keV",
                    "Dual-source 100/Sn140 kV virtual monoenergetic images on the SOMATOM Definition Flash.", "12_siemens_flash_ufc"),
                figure("helical_vs_stepshoot_coronal.png", "Coronal reformats of a helical scan and three volume-axial stations",
                    "30 cm of coverage: a 20 mm helix at pitch 1.0 against three 16 cm-beam axial stations 10 cm apart.", "11_helical_scanning"),
                figure("flash_ufc_lut_comparison.png", "Monte Carlo detector efficiency of three scintillators",
                    "Monte Carlo detector efficiency: two UFC crystals and the Gemstone garnet.", "12_siemens_flash_ufc"),
                figure("titanium_artifacts.png", "Titanium rods in water, soft-tissue and wide windows",
                    "Beam hardening and photon starvation from a user-defined titanium implant.", "10_titanium_implant"),
                figure("catsim_vs_basissim_mosaic.png", "CatSim, BasisSimulator CPU and GPU reconstructions of Gammex 472",
                    "The same scan through CatSim/XCIST and through BasisSimulator on CPU and GPU.", "06_catsim_vs_basissim"),
            )
        ),

        # ═══════════════════════════════════════════════════════════════════
        # §05 — WHERE NEXT
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-32 lg:mt-48 mb-16",
            section_head("05", "Where next", "Read, run,", "build on it.",
                "The walkthrough runs on a laptop in minutes; the notebooks are the full studies; the scanner page has the hardware parameter sets to copy."),
            Div(:class => "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5",
                [A(:href => href, :class => "block group no-underline " * card_cls * " hover:border-accent-400 dark:hover:border-accent-600",
                    Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-accent-600 dark:text-accent-400 mb-3", tag),
                    Div(:class => "font-serif text-2xl text-warm-900 dark:text-warm-100 group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors", t),
                    P(:class => "text-sm text-warm-600 dark:text-warm-400 mt-2 leading-relaxed", d))
                 for (href, tag, t, d) in [
                    ("$(BASE)/getting-started/", "Start", "Getting started →", "Install, the five structs, FDK and HIR, dose, photon counting and VMI."),
                    ("$(BASE)/examples/", "Learn", "Examples →", "Twelve Pluto notebooks with their code, prose and figures."),
                    ("$(BASE)/scanners/", "Hardware", "Scanners →", "GE Apex Elite, Siemens NAEOTOM Alpha, Force and Definition Flash."),
                    ("$(BASE)/api/", "Reference", "API →", "Every exported function and type, from the docstrings."),
                 ]]...),
            Div(:class => "mt-16 rounded-xl border border-warm-200 dark:border-warm-800 px-6 py-5 text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                "If BasisSimulator.jl is useful in your work, please cite: Black D, Khodajou-Chokami H, Molloi S. ",
                Em("BasisSimulator.jl: Open-source polychromatic CT simulation with a GPU-portable reconstruction stack."),
                " SoftwareX 2026;35:102910. ",
                A(:href => "https://doi.org/10.1016/j.softx.2026.102910", :target => "_blank",
                    :class => "text-accent-600 dark:text-accent-400 hover:underline", "doi:10.1016/j.softx.2026.102910"))
        )
    )
    end
end
