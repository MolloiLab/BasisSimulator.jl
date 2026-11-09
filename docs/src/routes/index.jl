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
        (
            "xcat_fbp_vs_hir.png",
            "Anatomy",
            "An XCAT chest with a material for every organ, FBP beside hybrid IR",
            "02_xcat_custom_materials",
        ),
        (
            "flash_ufc_vmi_grid.png",
            "Spectral",
            "Dual-source virtual monoenergetic images at 50 to 140 keV, raw and denoised",
            "12_siemens_flash_ufc",
        ),
    ],
    STEPS = [
        (
            n = "1",
            title = "Phantom",
            image = "gammex_472_phantom.png",
            anchor = "phantom",
            alt = "Gammex 472 calibration phantom label map",
            text = "What is scanned: a label map and one material per label. A calibration phantom, an anatomical model, or your own segmentation.",
        ),
        (
            n = "2",
            title = "Scan",
            image = "sinogram_standard.png",
            anchor = "scan",
            alt = "Simulated sinogram of the central detector row",
            text = "A model of a real scanner takes the spectrum through the patient to the detector, with its noise, and reports the dose.",
        ),
        (
            n = "3",
            title = "Reconstruct",
            image = "recon_compare_4panel.png",
            anchor = "reconstruct",
            alt = "Gammex 472 reconstructed with FBP and with hybrid IR at two dose levels",
            text = "Filtered back-projection, helical reconstruction or hybrid iterative reconstruction, in Hounsfield units.",
        ),
        (
            n = "4",
            title = "Spectral",
            image = "pcct_vmi_projection_grid.png",
            anchor = "spectral",
            alt = "Photon-counting virtual monoenergetic images",
            text = "Photon-counting, dual-kVp and dual-source data become water and iodine maps and virtual monoenergetic images.",
        ),
    ]
    () -> begin
        Div(
            # ─── Hero: calm serif type over the violet and cyan wash ───────────
            Div(
                :class => "relative",
                Div(:class => "bs-glow"),
                Div(
                    :class => "relative z-10 max-w-4xl space-y-8 pt-4 sm:pt-10",
                    Div(
                        :class => "flex flex-wrap items-center gap-2",
                        Span(
                            :class => "bs-pill bs-pill-sm bs-pill-static",
                            RawHtml(
                                """<span class="inline-block w-1.5 h-1.5 rounded-full" style="background:var(--accent)"></span>""",
                            ),
                            "Open Source · MIT",
                        ),
                        Span(
                            :class => "bs-pill bs-pill-sm bs-pill-static",
                            RawHtml(VERSION_CHIP),
                        ),
                    ),
                    H1(
                        :class => "bs-display text-[3.25rem] sm:text-7xl lg:text-[5.75rem]",
                        "Polychromatic CT, ",
                        Span(:class => "bs-flourish", "end-to-end"),
                        ", on any GPU.",
                    ),
                    P(
                        :class => "bs-lede max-w-2xl",
                        "Put a phantom in a model of a clinical CT scanner and get back what that scanner would give you: ",
                        "the images, in Hounsfield units, with realistic noise and artifacts, and the dose of the scan. ",
                        "Energy-integrating and photon-counting detectors, spectral and virtual monoenergetic imaging, ",
                        "on an NVIDIA or Apple GPU, or on the CPU.",
                    ),
                    Div(
                        :class => "flex flex-wrap gap-3 pt-1",
                        A(
                            :href => "$(BASE)/getting-started/",
                            :class => "bs-pill bs-pill-on",
                            "Get Started →",
                        ),
                        A(:href => "$(BASE)/examples/", :class => "bs-pill", "Examples"),
                        A(
                            :href => "https://github.com/MolloiLab/BasisSimulator.jl",
                            :target => "_blank",
                            :class => "bs-pill",
                            RawHtml(GITHUB_ICON),
                            "GitHub",
                        ),
                    ),
                ),
            ),
            # ─── Two showcase figures ───────────────────────────────────────────
            Div(
                :class =>
                    "relative z-10 mt-16 sm:mt-24 grid grid-cols-1 md:grid-cols-2 gap-5 sm:gap-6",
                [
                    A(
                        :href => "$(BASE)/examples/$(slug)/",
                        :class => "bs-tile bs-enter",
                        :style => "animation-delay:$(90 * (k - 1))ms",
                        Div(
                            :class => "bs-plate",
                            Img(
                                :src => "$(BASE)/assets/$(img)",
                                :alt => caption,
                                :loading => "lazy",
                                :class => "w-full aspect-[16/10] object-contain",
                            ),
                        ),
                        Div(
                            :class =>
                                "px-6 pt-4 pb-5 flex items-baseline justify-between gap-4",
                            P(
                                :class => "bs-body text-[1.0625rem]",
                                Span(:class => "bs-numeral mr-2", label),
                                caption,
                            ),
                            Span(:class => "bs-tile-more shrink-0", "→"),
                        ),
                    )
                    for (k, (img, label, caption, slug)) in enumerate(SHOWCASE)
                ]...,
            ),
            # ─── The four steps ─────────────────────────────────────────────────
            Div(
                :class => "mt-24 sm:mt-32",
                Div(
                    :class => "bs-fleuron mb-16 sm:mb-20",
                    Span(Symbol("aria-hidden") => "true", "❦"),
                ),
                Div(
                    :class => "max-w-2xl mb-10 space-y-4",
                    P(:class => "bs-eyebrow", "Phantom · Scan · Reconstruct · Spectral"),
                    H2(
                        :class => "bs-display text-5xl sm:text-6xl",
                        "Four steps, ",
                        Span(:class => "bs-flourish", "one scan."),
                    ),
                    P(
                        :class => "bs-body text-[1.1875rem]",
                        "Every study is the same four steps. The guide walks through each one, from a first image in a few ",
                        "minutes to the settings a physicist will want to check.",
                    ),
                ),
                Div(
                    :class => "grid grid-cols-1 md:grid-cols-2 gap-5 sm:gap-6",
                    [
                        A(
                            :href => "$(BASE)/getting-started/#$(s.anchor)",
                            :class => "bs-tile",
                            Div(
                                :class => "bs-plate",
                                Img(
                                    :src => "$(BASE)/assets/$(s.image)",
                                    :alt => s.alt,
                                    :loading => "lazy",
                                    :class => "w-full aspect-[16/10] object-contain p-2",
                                ),
                            ),
                            Div(
                                :class => "px-6 sm:px-7 pt-5 pb-6 space-y-2",
                                Div(
                                    :class => "flex items-baseline gap-3",
                                    Span(:class => "bs-numeral text-3xl leading-none", s.n),
                                    H3(
                                        :class => "bs-tile-title bs-title text-[1.875rem]",
                                        s.title,
                                    ),
                                ),
                                P(:class => "bs-body text-[1.0625rem]", s.text),
                            ),
                        )
                        for s in STEPS
                    ]...,
                ),
                Div(
                    :class => "mt-10 flex justify-center",
                    A(
                        :href => "$(BASE)/getting-started/",
                        :class => "bs-pill",
                        "Read the Guide →",
                    ),
                ),
            ),
            # ─── Citation ───────────────────────────────────────────────────────
            Div(
                :class =>
                    "bs-island mt-24 sm:mt-28 max-w-3xl mx-auto px-6 sm:px-10 py-7 sm:py-8 text-center space-y-3",
                P(:class => "bs-eyebrow", "Cite"),
                P(
                    :class => "bs-body text-[1.0625rem]",
                    "If BasisSimulator.jl is useful in your work, please cite: Black D, Khodajou-Chokami H, Molloi S. ",
                    Em(
                        :class => "bs-ink",
                        "BasisSimulator.jl: Open-source polychromatic CT simulation with a GPU-portable reconstruction stack.",
                    ),
                    " SoftwareX 2026;35:102910. ",
                    A(
                        :href => "https://doi.org/10.1016/j.softx.2026.102910",
                        :target => "_blank",
                        :class => "bs-link",
                        "doi:10.1016/j.softx.2026.102910",
                    ),
                ),
            ),
        )
    end
end
