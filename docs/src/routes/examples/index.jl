() -> begin
    Div(:class => "max-w-3xl mx-auto space-y-8",
        H1("Examples"),
        P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed",
            "Worked examples that double as the figure sources for the SoftwareX paper. Each example is a ",
            "Pluto notebook that runs end-to-end against a simulated phantom and renders publication-quality figures. ",
            "Notebooks will be embedded as static HTML once rendered."
        ),

        Div(:class => "grid grid-cols-1 gap-4",
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "flex items-baseline gap-3 mb-2",
                    Span(:class => "text-xs font-mono px-2 py-0.5 rounded bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300", "01"),
                    H3(:class => "no-rule font-semibold m-0", "Single-kVp verification against CatSim/XCIST")
                ),
                P(:class => "text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                    "Head-to-head comparison on identical Gammex 472 configurations at 120 kVp. Validates ",
                    "the full pipeline — forward projection, FDK reconstruction, HU accuracy, NPS, MTF, CNR — against the established CatSim framework."
                )
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "flex items-baseline gap-3 mb-2",
                    Span(:class => "text-xs font-mono px-2 py-0.5 rounded bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300", "02"),
                    H3(:class => "no-rule font-semibold m-0", "Multi-protocol dose & Hybrid IR")
                ),
                P(:class => "text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                    "Three protocols (80 / 120 / 140 kVp at proportional mA) demonstrate spectral sensitivity and ",
                    "dose-proportional noise behavior. Hybrid IR at strengths 1, 3, 5 then maps the noise–resolution tradeoff."
                )
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "flex items-baseline gap-3 mb-2",
                    Span(:class => "text-xs font-mono px-2 py-0.5 rounded bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300", "03"),
                    H3(:class => "no-rule font-semibold m-0", "Dual-kVp VMI vs NIST XCOM")
                ),
                P(:class => "text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                    "Complete dual-energy VMI pipeline: 80/140 kVp acquisition, sinogram-domain photo/Compton ",
                    "decomposition, VMI synthesis at 40–140 keV, validated against NIST XCOM reference attenuation."
                )
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "flex items-baseline gap-3 mb-2",
                    Span(:class => "text-xs font-mono px-2 py-0.5 rounded bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300", "04"),
                    H3(:class => "no-rule font-semibold m-0", "PCCT detector physics")
                ),
                P(:class => "text-sm text-warm-600 dark:text-warm-400 leading-relaxed",
                    "Physics-based CdTe simulation: charge cloud transport (Koch-Mehrin), K-fluorescence cascade, ",
                    "Hecht charge collection efficiency, pileup, and unified detector response matrix. Reconstructs ",
                    "4-bin energy-resolved data with FDK and Hybrid IR, then synthesizes VMI and K-edge images."
                )
            ),
        ),

        P(:class => "text-sm text-warm-500 dark:text-warm-500 pt-4",
            "Verification notebooks against CatSim and clinical-data benchmarks live in ",
            A(:href => "https://github.com/MolloiLab/basis-verification", :target => "_blank",
              :class => "text-accent-500 hover:text-accent-600 underline no-underline", "MolloiLab/basis-verification"),
            "."
        )
    )
end
