() -> begin
    Div(:class => "max-w-3xl mx-auto space-y-8",
        H1("API Reference"),
        P(:class => "text-warm-600 dark:text-warm-400 leading-relaxed",
            "Generated API reference is forthcoming. In the meantime, every public function and struct ",
            "carries a docstring — query it from the Julia REPL with ",
            Code(:class => "text-accent-500 font-mono", "?Function"), "."
        ),

        H2("Core types"),
        P("The five structs every simulation specifies:"),
        Div(:class => "grid grid-cols-1 sm:grid-cols-2 gap-4 not-prose",
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-4 bg-warm-50 dark:bg-warm-900/50",
                H3(:class => "no-rule font-mono text-base mb-1", "Phantom"),
                P(:class => "text-sm text-warm-600 dark:text-warm-400", "Labeled 3D mask + materials dictionary + voxel size.")
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-4 bg-warm-50 dark:bg-warm-900/50",
                H3(:class => "no-rule font-mono text-base mb-1", "Scanner"),
                P(:class => "text-sm text-warm-600 dark:text-warm-400", "Geometry, source, detector array, filtration. Switches between EICT and PCCT via ", Code(:class => "text-xs", "detector_type"), ".")
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-4 bg-warm-50 dark:bg-warm-900/50",
                H3(:class => "no-rule font-mono text-base mb-1", "CTProtocol"),
                P(:class => "text-sm text-warm-600 dark:text-warm-400", "kVp, mA, views, rotation time, dual-energy switches.")
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-4 bg-warm-50 dark:bg-warm-900/50",
                H3(:class => "no-rule font-mono text-base mb-1", "SimOptions"),
                P(:class => "text-sm text-warm-600 dark:text-warm-400", "Fidelity preset (", Code(:class => "text-xs", ":eict"), " | ", Code(:class => "text-xs", ":pcct"), ") and per-effect toggles.")
            ),
            Div(:class => "border border-warm-200 dark:border-warm-800 rounded-lg p-4 bg-warm-50 dark:bg-warm-900/50",
                H3(:class => "no-rule font-mono text-base mb-1", "ReconOptions"),
                P(:class => "text-sm text-warm-600 dark:text-warm-400", "Output matrix, FOV, algorithm (", Code(:class => "text-xs", ":fdk"), "), VMI energies and basis materials.")
            ),
        ),

        H2("Driver functions"),
        Ul(:class => "list-disc list-inside space-y-1 text-warm-700 dark:text-warm-300",
            Li(Code(:class => "font-mono text-accent-500", "create_eict_workspace(scanner, protocol, sim_opts, rec_opts, phantom)"), " — pre-allocate buffers for an EICT acquisition."),
            Li(Code(:class => "font-mono text-accent-500", "create_workspace(scanner, protocol, sim_opts, rec_opts, phantom)"), " — auto-detected EICT/PCCT dispatch."),
            Li(Code(:class => "font-mono text-accent-500", "simulate!(ws, phantom, scanner, protocol, sim_opts, rec_opts)"), " — zero-alloc forward simulation."),
            Li(Code(:class => "font-mono text-accent-500", "create_fdk_recon_workspace(sino, geom, size)"), " — pre-allocate FDK recon buffers."),
            Li(Code(:class => "font-mono text-accent-500", "create_hir_recon_workspace(sino, geom, size; strength=3)"), " — Hybrid IR (PWLS + Huber)."),
            Li(Code(:class => "font-mono text-accent-500", "reconstruct!(ws_recon, sino, geom, size)"), " — run the chosen reconstruction in-place."),
            Li(Code(:class => "font-mono text-accent-500", "to_hounsfield(volume; μ_water)"), " — convert μ-volume to HU."),
        ),
    )
end
