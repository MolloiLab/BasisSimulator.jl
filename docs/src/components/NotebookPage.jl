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
    )
    get(overrides, slug, replace(slug, "_" => " "))
end

function NotebookPage(slug::AbstractString)
    BASE = get(ENV, "BASISSIM_BASE", "")
    title = _notebook_display_title(slug)

    Div(:class => "max-w-6xl mx-auto px-6 py-6 space-y-4",
        # Breadcrumb / back-link
        Div(:class => "flex items-center gap-3 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-500 dark:text-warm-500",
            A(:href => "$(BASE)/examples/",
                :class => "hover:text-accent-600 dark:hover:text-accent-400 no-underline transition-colors",
                "← Examples"
            ),
            Span(:class => "text-warm-300 dark:text-warm-700", "/"),
            Span(:class => "text-warm-700 dark:text-warm-300", title),
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
