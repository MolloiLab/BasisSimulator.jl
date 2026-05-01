let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
    Div(
        # ═══════════════════════════════════════════════════════════════════
        # HERO — two columns, natural height (no flex-stretch, no min-h)
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 pt-4 lg:pt-8",

            # ─── Left: copy ─────────────────────────────────────────────────
            Div(:class => "lg:col-span-7 space-y-8",
                Div(:class => "flex flex-wrap items-center gap-3 text-[10px] tracking-[0.2em] uppercase font-mono",
                    Span(:class => "px-3 py-1 rounded-full bg-accent-600 text-white",
                        RawHtml("""<span class="opacity-80">●</span>&nbsp; Open Source · MIT""")
                    ),
                    Span(:class => "px-3 py-1 rounded-full border border-warm-300 dark:border-warm-700 text-warm-600 dark:text-warm-400",
                        RawHtml("""Simulation&nbsp;+&nbsp;Reconstruction""")
                    )
                ),
                H1(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-5xl md:text-6xl lg:text-7xl leading-[1.05] tracking-tight",
                    "Polychromatic CT, ",
                    RawHtml("""<span style="color:var(--color-accent-500)" class="italic font-semibold">end-to-end</span>"""),
                    ", on every ",
                    RawHtml("""<span class="relative inline-block">GPU.<span class="absolute left-0 -bottom-1 h-[3px] w-full" style="background:var(--color-accent-500)"></span></span>""")
                ),
                P(:class => "max-w-xl text-warm-600 dark:text-warm-400 leading-relaxed text-base",
                    "BasisSimulator.jl is an open-source polychromatic CT simulator and reconstructor. ",
                    "From source spectrum through Beer-Lambert forward projection to ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "FBP"), ", ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "Hybrid IR"), ", and material-basis ",
                    Span(:class => "text-warm-900 dark:text-warm-100 font-semibold", "VMI"),
                    " — covering both energy-integrating and CdTe photon-counting detectors. The same source tree runs on Metal, CUDA, ROCm, and CPU."
                ),
                Div(:class => "flex flex-wrap gap-3 pt-2",
                    A(:href => "$(BASE)/getting-started/",
                        :class => "px-5 py-2.5 bg-accent-600 hover:bg-accent-700 text-white rounded-md text-sm font-medium tracking-wide transition-colors no-underline",
                        "Get Started →"
                    ),
                    A(:href => "https://github.com/MolloiLab/BasisSimulator.jl", :target => "_blank",
                        :class => "px-5 py-2.5 border border-warm-300 dark:border-warm-700 rounded-md text-sm font-medium text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-900 transition-colors no-underline",
                        "GitHub"
                    )
                )
            ),

            # ─── Right: bounded image-card collage ──────────────────────────
            Div(:class => "lg:col-span-5 relative h-[480px] lg:h-[560px] mt-4 lg:mt-0",
                # OPEN SOURCE orb badge — z-30 keeps it above the floating
                # image cards (z-10/20/30) but BELOW the sticky nav (z-40 in
                # Layout.jl), so scrolling doesn't push it over the header.
                Div(:class => "absolute top-2 right-2 z-30 w-16 h-16 rounded-full flex items-center justify-center text-[9px] tracking-[0.15em] uppercase font-mono text-white text-center leading-tight shadow-lg",
                    :style => "background:radial-gradient(circle at 30% 30%, var(--color-accent-400), var(--color-accent-600) 70%, var(--color-accent-700));",
                    RawHtml("Open<br/>Source")
                ),
                # Card 1 — top, slight CCW (Gammex 472 phantom)
                Div(:class => "absolute top-4 left-2 lg:left-0 w-[78%] aspect-[16/10] rounded-xl border border-warm-300 dark:border-warm-800 bg-warm-50 dark:bg-warm-900 shadow-2xl overflow-hidden -rotate-3 z-10",
                    Img(:src => "$(BASE)/assets/gammex_472_phantom.png",
                        :alt => "Gammex Model 472 phantom",
                        :class => "w-full h-full object-contain")
                ),
                # Card 2 — middle-right, light CW (standard-dose sinogram)
                Div(:class => "absolute top-[34%] right-2 lg:right-[-6%] w-[64%] aspect-[4/5] rounded-xl border border-warm-300 dark:border-warm-800 bg-warm-50 dark:bg-warm-900 shadow-2xl overflow-hidden rotate-[4deg] z-20",
                    Img(:src => "$(BASE)/assets/sinogram_standard.png",
                        :alt => "Standard-dose sinogram (central detector row)",
                        :class => "w-full h-full object-contain")
                ),
                # Card 3 — bottom-left, larger CCW (recon comparison: dose × correction)
                Div(:class => "absolute bottom-2 left-4 lg:left-[-2%] w-[62%] aspect-[6/5] rounded-xl border border-warm-300 dark:border-warm-800 bg-warm-50 dark:bg-warm-900 shadow-2xl overflow-hidden -rotate-[6deg] z-30",
                    Img(:src => "$(BASE)/assets/recon_compare_4panel.png",
                        :alt => "Standard vs low-dose reconstruction, raw vs corrected",
                        :class => "w-full h-full object-contain")
                )
            )
        ),

        # ═══════════════════════════════════════════════════════════════════
        # STATS — full-width row, own block, clear separation from hero
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-24 lg:mt-32 pt-10 border-t border-warm-300 dark:border-warm-800",
            Div(:class => "grid grid-cols-2 md:grid-cols-4 gap-y-8 gap-x-6",
                Div(
                    Div(:class => "font-serif text-5xl md:text-6xl text-warm-900 dark:text-warm-100 leading-none", "02"),
                    Div(:class => "mt-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 leading-tight",
                        "Detector", Br(), "Physics")
                ),
                Div(
                    Div(:class => "font-serif text-5xl md:text-6xl text-warm-900 dark:text-warm-100 leading-none", "03"),
                    Div(:class => "mt-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 leading-tight",
                        "Recon", Br(), "Pipelines")
                ),
                Div(
                    Div(:class => "font-serif text-5xl md:text-6xl text-warm-900 dark:text-warm-100 leading-none", "04"),
                    Div(:class => "mt-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 leading-tight",
                        "GPU", Br(), "Backends")
                ),
                Div(
                    Div(:class => "font-serif text-5xl md:text-6xl text-warm-900 dark:text-warm-100 leading-none", "05"),
                    Div(:class => "mt-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 leading-tight",
                        "Composable", Br(), "Structs")
                )
            )
        ),

        # ═══════════════════════════════════════════════════════════════════
        # §02 — THE WORKFLOW (code preview)
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-32 lg:mt-48",
            # Section header
            Div(:class => "max-w-3xl mb-14",
                Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 mb-8",
                    RawHtml("""<span style="color:var(--color-accent-500)">§ 02</span>&nbsp;&nbsp;The Workflow""")
                ),
                H2(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl lg:text-6xl leading-[1.08] tracking-tight",
                    "Phantom → sinogram → image,",
                    Br(),
                    RawHtml("""<span style="color:var(--color-accent-500)" class="italic">in five structs.</span>""")
                ),
                P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed mt-6 text-base",
                    "Allocate a workspace once. Reuse it on every subsequent call. Forward projection, noise model, and reconstruction are all GPU-resident and zero-allocation in steady state."
                )
            ),

            # Code block
            Pre(:class => "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 md:p-8 rounded-xl overflow-x-auto border border-warm-800 shadow-sm",
                Code(:class => "language-julia text-sm font-mono leading-relaxed", """import BasisSimulator as BS
using Metal  # or CUDA / AMDGPU; omit for CPU

# 1. Phantom — labeled mask + materials dict + voxel size
phantom_cpu = BS.create_gammex_472(n_voxels=256)
phantom = BS.Phantom(MtlArray(phantom_cpu.mask),
                     phantom_cpu.materials,
                     phantom_cpu.voxel_size)

# 2-5. Scanner / Protocol / SimOptions / ReconOptions
scanner  = BS.Scanner(source_to_isocenter=626.0, source_to_detector=1097.0)
protocol = BS.CTProtocol(kVp=120.0, mA=200.0, views=984)
sim_opts = BS.SimOptions(fidelity=:eict)              # or :pcct
rec_opts = BS.ReconOptions(matrix_size=(512, 512, 64), fov_cm=35.0)

# Allocate workspace once, reuse on subsequent calls (zero-alloc steady state)
ws = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
BS.simulate!(ws, phantom, scanner, protocol, sim_opts, rec_opts)

# Reconstruct → Hounsfield units
ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, rec_opts.matrix_size)
hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, rec_opts.matrix_size));
    μ_water = BS.get_reference_μ_water(70.0),
)""")
            ),

            Div(:class => "mt-6 flex justify-end",
                A(:href => "$(BASE)/getting-started/",
                    :class => "text-sm text-warm-600 dark:text-warm-400 hover:text-accent-600 dark:hover:text-accent-400 no-underline",
                    "Full walkthrough →"
                )
            )
        ),

        # ═══════════════════════════════════════════════════════════════════
        # §03 — WHAT'S INSIDE (three pillars)
        # ═══════════════════════════════════════════════════════════════════
        Div(:class => "mt-32 lg:mt-48 mb-16",
            Div(:class => "max-w-3xl mb-14",
                Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500 mb-8",
                    RawHtml("""<span style="color:var(--color-accent-500)">§ 03</span>&nbsp;&nbsp;What's inside""")
                ),
                H2(:class => "no-rule font-serif font-medium text-warm-900 dark:text-warm-100 text-4xl md:text-5xl lg:text-6xl leading-[1.08] tracking-tight",
                    "Three pillars,",
                    Br(),
                    RawHtml("""<span style="color:var(--color-accent-500)" class="italic">one package.</span>""")
                ),
                P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed mt-6 text-base",
                    "End-to-end means simulation physics, reconstruction algorithms, and hardware backends — all under one roof, all GPU-portable."
                )
            ),

            Div(:class => "grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8",
                Div(:class => "border border-warm-200 dark:border-warm-800 rounded-xl p-8 bg-warm-100/40 dark:bg-warm-900/40 hover:bg-warm-100 dark:hover:bg-warm-900/70 transition-colors",
                    Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-accent-600 dark:text-accent-400 mb-4", "Physics"),
                    H3(:class => "no-rule font-serif text-2xl mb-3 text-warm-900 dark:text-warm-100 leading-snug", "Polychromatic & Spectral"),
                    P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed",
                        "Beer-Lambert across the full source spectrum. Energy-integrating detection or CdTe photon-counting with Koch-Mehrin charge transport, K-fluorescence, pileup, and an MC-derived DRM."
                    )
                ),
                Div(:class => "border border-warm-200 dark:border-warm-800 rounded-xl p-8 bg-warm-100/40 dark:bg-warm-900/40 hover:bg-warm-100 dark:hover:bg-warm-900/70 transition-colors",
                    Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-accent-600 dark:text-accent-400 mb-4", "Reconstruction"),
                    H3(:class => "no-rule font-serif text-2xl mb-3 text-warm-900 dark:text-warm-100 leading-snug", "FBP · IR · VMI"),
                    P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed",
                        "FBP (FDK), Hybrid Iterative Reconstruction (PWLS + Huber), and material-basis Virtual Monoenergetic Imaging — all GPU-resident with a zero-allocation workspace pattern."
                    )
                ),
                Div(:class => "border border-warm-200 dark:border-warm-800 rounded-xl p-8 bg-warm-100/40 dark:bg-warm-900/40 hover:bg-warm-100 dark:hover:bg-warm-900/70 transition-colors",
                    Div(:class => "text-[10px] tracking-[0.2em] uppercase font-mono text-accent-600 dark:text-accent-400 mb-4", "Portability"),
                    H3(:class => "no-rule font-serif text-2xl mb-3 text-warm-900 dark:text-warm-100 leading-snug", "Backend-Agnostic GPU"),
                    P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed",
                        "Backend-agnostic via AcceleratedKernels.jl — Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD), or CPU from the same Julia source. No vendor-specific kernels to maintain."
                    )
                )
            )
        )
    )
    end
end
