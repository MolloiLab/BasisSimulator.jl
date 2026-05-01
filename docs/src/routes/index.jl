() -> begin
    Div(:class => "space-y-16",
        # Hero
        Div(:class => "text-center space-y-6 pt-8",
            H1(:class => "no-rule text-5xl md:text-6xl font-serif font-bold text-warm-900 dark:text-warm-100",
                "Polychromatic CT Simulation"
            ),
            H1(:class => "no-rule text-5xl md:text-6xl font-serif font-bold text-accent-500",
                "in Julia"
            ),
            P(:class => "text-lg text-warm-600 dark:text-warm-400 max-w-2xl mx-auto leading-relaxed",
                "GPU-portable CT simulator with end-to-end physics: source spectra, polychromatic forward projection, ",
                "energy-integrating and photon-counting detectors, and a full reconstruction pipeline (FBP, Hybrid IR, VMI). ",
                "Backend-agnostic via ",
                A(:href => "https://github.com/JuliaGPU/AcceleratedKernels.jl", :target => "_blank",
                  :class => "text-accent-500 hover:text-accent-600 underline", "AcceleratedKernels.jl"),
                " — runs on Metal, CUDA, ROCm, or CPU from the same source. Core ray tracing ported from ",
                A(:href => "https://github.com/CERN/TIGRE", :target => "_blank",
                  :class => "text-accent-500 hover:text-accent-600 underline", "TIGRE"),
                "; calibration workflow follows ",
                A(:href => "https://github.com/xcist/main", :target => "_blank",
                  :class => "text-accent-500 hover:text-accent-600 underline", "CatSim/XCIST"),
                "."
            ),
            Div(:class => "flex gap-4 justify-center pt-4",
                A(:href => "/BasisSimulator.jl/getting-started/",
                    :class => "px-6 py-3 bg-accent-600 hover:bg-accent-700 text-white rounded-lg font-medium transition-colors",
                    "Get Started"
                ),
                A(:href => "https://github.com/MolloiLab/BasisSimulator.jl", :target => "_blank",
                    :class => "px-6 py-3 border border-warm-300 dark:border-warm-700 rounded-lg font-medium text-warm-700 dark:text-warm-300 hover:bg-warm-100 dark:hover:bg-warm-900 transition-colors",
                    "View on GitHub"
                )
            )
        ),
        # Code preview
        Div(:class => "flex flex-col items-center gap-6",
            Div(:class => "w-full max-w-3xl",
                Pre(:class => "bg-warm-900 dark:bg-warm-950 text-warm-200 p-6 rounded-lg overflow-x-auto border border-warm-800",
                    Code(:class => "language-julia text-sm font-mono", """import BasisSimulator as BS
using Metal  # or CUDA / AMDGPU; omit for CPU

phantom_cpu = BS.create_gammex_472(n_voxels=256)
phantom = BS.Phantom(MtlArray(phantom_cpu.mask),
                     phantom_cpu.materials,
                     phantom_cpu.voxel_size)

scanner  = BS.Scanner(source_to_isocenter=626.0, source_to_detector=1097.0)
protocol = BS.CTProtocol(kVp=120.0, mA=200.0, views=984)
sim_opts = BS.SimOptions(fidelity=:eict)
rec_opts = BS.ReconOptions(matrix_size=(512, 512, 64), fov_cm=35.0)

ws = BS.create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)
BS.simulate!(ws, phantom, scanner, protocol, sim_opts, rec_opts)

ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, rec_opts.matrix_size)
hu = BS.to_hounsfield(
    Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, rec_opts.matrix_size));
    μ_water = BS.get_reference_μ_water(70.0),
)""")
                )
            )
        ),
        # Feature cards
        Div(:class => "grid grid-cols-1 md:grid-cols-3 gap-6",
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "w-10 h-10 rounded-lg bg-accent-100 dark:bg-accent-900/50 flex items-center justify-center mb-4",
                    RawHtml("""<svg class="w-5 h-5 text-accent-600 dark:text-accent-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>""")
                ),
                H3(:class => "font-semibold mb-2 text-warm-900 dark:text-warm-100", "GPU-Portable"),
                P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed",
                    "One source tree runs on Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD), or CPU. No vendor-specific kernels."
                )
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "w-10 h-10 rounded-lg bg-accent-secondary-100 dark:bg-accent-secondary-900/50 flex items-center justify-center mb-4",
                    RawHtml("""<svg class="w-5 h-5 text-accent-secondary-600 dark:text-accent-secondary-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>""")
                ),
                H3(:class => "font-semibold mb-2 text-warm-900 dark:text-warm-100", "Polychromatic + Spectral"),
                P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed",
                    "Beer-Lambert across the full source spectrum, dual-kVp acquisition, and CdTe photon-counting physics (charge transport, K-fluorescence, pileup, DRM)."
                )
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-6 bg-warm-100/50 dark:bg-warm-900/50",
                Div(:class => "w-10 h-10 rounded-lg bg-accent-100 dark:bg-accent-900/50 flex items-center justify-center mb-4",
                    RawHtml("""<svg class="w-5 h-5 text-accent-600 dark:text-accent-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 6-6"/></svg>""")
                ),
                H3(:class => "font-semibold mb-2 text-warm-900 dark:text-warm-100", "Full Reconstruction Pipeline"),
                P(:class => "text-warm-600 dark:text-warm-400 text-sm leading-relaxed",
                    "FBP (FDK), Hybrid IR (PWLS + Huber), and material-basis VMI from sinograms — all GPU-resident with a zero-allocation workspace pattern."
                )
            )
        )
    )
end
