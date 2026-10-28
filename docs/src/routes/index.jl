# Landing page — what BasisSimulator.jl is in one screen, two showcase figures, and the four steps of a
# scan, each linking into the Getting Started guide. Figures are the ones the example notebooks write to
# docs/assets/. The version chip is read from Project.toml, so it never goes stale.

let BASE = get(ENV, "BASISSIM_BASE", ""),
    PROJECT = read(joinpath(@__DIR__, "..", "..", "..", "Project.toml"), String),
    VERSION_CHIP = let v = VersionNumber(match(r"^version\s*=\s*\"(.+?)\""m, PROJECT)[1]),
                       j = match(r"^julia\s*=\s*\"(.+?)\""m, PROJECT)[1]
        "v$(v.major).$(v.minor)&nbsp;·&nbsp;Julia&nbsp;$(j)+"
    end,
    SHOWCASE = [
        ("xcat_fbp_vs_hir.png", "Anatomy", "An XCAT chest with a material for every organ, FBP beside hybrid IR",
            "02_xcat_custom_materials"),
        ("flash_ufc_vmi_grid.png", "Spectral", "Dual-source virtual monoenergetic images at 50 to 140 keV, raw and denoised",
            "12_siemens_flash_ufc"),
    ],
    STEPS = [
        (n = "1", title = "Phantom", image = "gammex_472_phantom.png", anchor = "phantom",
            alt = "Gammex 472 calibration phantom label map",
            text = "What is scanned: a label map and one material per label. A calibration phantom, an anatomical model, or your own segmentation."),
        (n = "2", title = "Scan", image = "sinogram_standard.png", anchor = "scan",
            alt = "Simulated sinogram of the central detector row",
            text = "A model of a real scanner takes the spectrum through the patient to the detector, with its noise, and reports the dose."),
        (n = "3", title = "Reconstruct", image = "recon_compare_4panel.png", anchor = "reconstruct",
            alt = "Gammex 472 reconstructed with FBP and with hybrid IR at two dose levels",
            text = "Filtered back-projection, helical reconstruction or hybrid iterative reconstruction, in Hounsfield units."),
        (n = "4", title = "Spectral", image = "pcct_vmi_projection_grid.png", anchor = "spectral",
            alt = "Photon-counting virtual monoenergetic images",
            text = "Photon-counting, dual-kVp and dual-source data become water and iodine maps and virtual monoenergetic images."),
    ]

    () -> begin
    tile = "group block no-underline rounded-2xl overflow-hidden border border-warm-200 dark:border-warm-800 bg-warm-50 dark:bg-warm-900/40 hover:border-accent-400 dark:hover:border-accent-600 transition-colors"
    pill = "px-5 py-2.5 border border-warm-300 dark:border-warm-700 rounded-md text-sm font-medium text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-900 transition-colors no-underline"

    Div(
        # ─── Hero ───────────────────────────────────────────────────────────
        Div(:class => "max-w-3xl space-y-8 pt-4 lg:pt-10",
            Div(:class => "flex flex-wrap items-center gap-3 text-[10px] tracking-[0.2em] uppercase font-mono",
                Span(:class => "px-3 py-1 rounded-full bg-accent-600 text-white",
                    RawHtml("""<span class="opacity-80">●</span>&nbsp; Open Source · MIT""")),
                Span(:class => "px-3 py-1 rounded-full border border-warm-300 dark:border-warm-700 text-warm-600 dark:text-warm-400",
                    RawHtml(VERSION_CHIP))),
            H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-5xl md:text-6xl lg:text-7xl leading-[1.05] tracking-tight",
                "Polychromatic CT, ",
                RawHtml("""<span style="color:var(--color-accent-500)" class="italic font-semibold">end-to-end</span>"""),
                ", on any ",
                RawHtml("""<span class="relative inline-block">GPU.<span class="absolute left-0 -bottom-1 h-[3px] w-full" style="background:var(--color-accent-500)"></span></span>""")),
            P(:class => "max-w-2xl text-warm-600 dark:text-warm-400 leading-relaxed text-lg",
                "Put a phantom in a model of a clinical CT scanner and get back what that scanner would give you: ",
                "the images, in Hounsfield units, with realistic noise and artifacts, and the dose of the scan. ",
                "Energy-integrating and photon-counting detectors, spectral and virtual monoenergetic imaging, ",
                "on an NVIDIA or Apple GPU, or on the CPU."),
            Div(:class => "flex flex-wrap gap-3 pt-1",
                A(:href => "$(BASE)/getting-started/",
                    :class => "px-5 py-2.5 bg-accent-600 hover:bg-accent-700 text-white rounded-md text-sm font-medium tracking-wide transition-colors no-underline",
                    "Get started →"),
                A(:href => "$(BASE)/examples/", :class => pill, "Examples"),
                A(:href => "https://github.com/MolloiLab/BasisSimulator.jl", :target => "_blank", :class => pill, "GitHub"))),

        # ─── Two showcase figures ───────────────────────────────────────────
        Div(:class => "mt-16 lg:mt-20 grid grid-cols-1 md:grid-cols-2 gap-6",
            [A(:href => "$(BASE)/examples/$(slug)/", :class => tile,
                Div(:class => "bg-white", Img(:src => "$(BASE)/assets/$(img)", :alt => caption, :loading => "lazy",
                    :class => "w-full aspect-[16/10] object-contain")),
                P(:class => "px-5 py-4 text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                    Span(:class => "font-mono text-[10px] tracking-[0.2em] uppercase text-accent-600 dark:text-accent-400 mr-2", label),
                    caption))
             for (img, label, caption, slug) in SHOWCASE]...),

        # ─── The four steps ─────────────────────────────────────────────────
        Div(:class => "mt-28 lg:mt-36",
            Div(:class => "max-w-2xl mb-10 space-y-4",
                H2(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl leading-[1.08] tracking-tight",
                    "Four steps, ", RawHtml("""<span style="color:var(--color-accent-500)" class="italic">one scan.</span>""")),
                P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed text-lg",
                    "Every study is the same four steps. The guide walks through each one, from a first image in a few ",
                    "minutes to the settings a physicist will want to check.")),
            Div(:class => "grid grid-cols-1 md:grid-cols-2 gap-6",
                [A(:href => "$(BASE)/getting-started/#$(s.anchor)", :class => tile,
                    Div(:class => "bg-white", Img(:src => "$(BASE)/assets/$(s.image)", :alt => s.alt, :loading => "lazy",
                        :class => "w-full aspect-[16/10] object-contain p-2")),
                    Div(:class => "p-6 space-y-2",
                        Div(:class => "flex items-baseline gap-3",
                            Span(:class => "font-serif text-2xl text-accent-500 leading-none", s.n),
                            H3(:class => "no-rule font-serif text-2xl text-warm-900 dark:text-warm-100 group-hover:text-accent-600 dark:group-hover:text-accent-400 transition-colors", s.title)),
                        P(:class => "text-[15px] text-warm-600 dark:text-warm-400 leading-relaxed", s.text)))
                 for s in STEPS]...),
            Div(:class => "mt-10 flex justify-center",
                A(:href => "$(BASE)/getting-started/", :class => pill, "Read the guide →"))),

        # ─── Citation ───────────────────────────────────────────────────────
        Div(:class => "mt-24 mb-16 max-w-3xl rounded-xl border border-warm-200 dark:border-warm-800 px-6 py-5 text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
            "If BasisSimulator.jl is useful in your work, please cite: Black D, Khodajou-Chokami H, Molloi S. ",
            Em("BasisSimulator.jl: Open-source polychromatic CT simulation with a GPU-portable reconstruction stack."),
            " SoftwareX 2026;35:102910. ",
            A(:href => "https://doi.org/10.1016/j.softx.2026.102910", :target => "_blank",
                :class => "text-accent-600 dark:text-accent-400 hover:underline", "doi:10.1016/j.softx.2026.102910"))
    )
    end
end
