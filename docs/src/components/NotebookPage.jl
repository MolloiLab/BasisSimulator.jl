# NotebookPage — wraps a static-rendered Pluto notebook in an iframe with
# Therapy chrome (top breadcrumb back-link).  Used by per-slug routes
# registered in docs/app.jl.
#
# The iframe points to /notebooks-static/<slug>.html, which is written by
# `export_notebooks()` (docs/extract_all.jl) into docs/dist/notebooks-static/.
# Pluto's CSS/JS stays sandboxed inside the iframe — no style collisions with
# the Tailwind layout chrome.

"""Pretty-print a slug like `01_five_struct_api` → `01 · The Five-Struct API`.
Falls back to the raw slug if no override is registered."""
function _notebook_display_title(slug::AbstractString)
    overrides = Dict(
        "01_five_struct_api"        => "01 · The Five-Struct API",
        "02_xcat_custom_materials"  => "02 · XCAT Phantom + Custom Materials",
        "03_dual_kvp_switching_vmi" => "03 · Dual-kVp Switching VMI on Gammex 472",
        "04_pcct_vmi"               => "04 · PCCT VMI on Gammex 472",
        "05_xcat_grid_to_recon"     => "05 · XCAT UHR → CT Scan: Affine Round-Trip",
        "06_catsim_vs_basissim"     => "06 · CatSim vs BasisSimulator (CPU + GPU)",
    )
    get(overrides, slug, replace(slug, "_" => " "))
end

const _NOTEBOOK_GITHUB_BASE =
    "https://github.com/MolloiLab/BasisSimulator.jl/blob/main/docs/notebooks"

function NotebookPage(slug::AbstractString)
    BASE = get(ENV, "BASISSIM_BASE", "")
    title = _notebook_display_title(slug)
    github_src_url = "$(_NOTEBOOK_GITHUB_BASE)/$(slug).jl"

    Div(:class => "max-w-6xl mx-auto px-6 py-6 space-y-4",
        # Breadcrumb (left) + GitHub source link (right) — single flex row.
        Div(:class => "flex items-center justify-between gap-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500",
            # Left: breadcrumb back-link
            Div(:class => "flex items-center gap-3",
                A(:href => "$(BASE)/examples/",
                    :class => "hover:text-accent-600 dark:hover:text-accent-400 no-underline transition-colors",
                    "← Examples"
                ),
                Span(:class => "text-warm-300 dark:text-warm-700", "/"),
                Span(:class => "text-warm-700 dark:text-warm-300", title),
            ),
            # Right: direct link to the .jl source on GitHub
            A(:href => github_src_url, :target => "_blank", :rel => "noopener",
                :class => "flex items-center gap-2 px-3 py-1.5 rounded-full border border-warm-300 dark:border-warm-700 hover:border-accent-400 dark:hover:border-accent-600 hover:text-accent-600 dark:hover:text-accent-400 transition-colors no-underline",
                RawHtml("""<svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>"""),
                Span("View source on GitHub"),
            ),
        ),

        # Iframe — Pluto static HTML, sandboxed CSS/JS.  Force a white
        # background regardless of our docs site's dark mode: Pluto's
        # standalone HTML uses light-mode colors (dark text on white), so
        # rendering it on top of `dark:bg-warm-950` makes it invisible.
        Iframe(
            :src    => "$(BASE)/notebooks-static/$(slug).html",
            :class  => "w-full h-[calc(100vh-9rem)] border border-warm-200 dark:border-warm-800 rounded-lg",
            :style  => "background-color: #ffffff;",
            :loading => "eager",
            :title  => title,
        ),
    )
end
