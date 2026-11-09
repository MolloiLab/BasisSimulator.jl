# NotebookPage — embeds Snapshot's scoped Pluto fragment directly inside the
# Therapy route. Used by per-slug routes registered in docs/app.jl.
#
# `export_notebooks()` writes both a standalone `<slug>.html` and a native
# `<slug>.fragment.html`. The fragment wraps its CSS in `@scope
# (.snap-notebook)`, so Pluto styling cannot leak into the docs shell and the
# notebook reads as ordinary page content instead of a nested iframe viewport.

"""Pretty-print a slug like `01_five_struct_api` → `01 · The Five-Struct API`.
Falls back to the raw slug if no override is registered."""
function _notebook_display_title(slug::AbstractString)
    overrides = Dict(
        "01_five_struct_api" => "01 · The Five-Struct API",
        "02_xcat_custom_materials" => "02 · XCAT Anatomy + Custom Materials",
        "03_dual_kvp_switching_vmi" => "03 · Dual-kVp Switching VMI",
        "04_pcct_vmi" => "04 · Photon-Counting VMI",
        "05_xcat_grid_to_recon" => "05 · XCAT Grids and the Affine Round-Trip",
        "06_catsim_vs_basissim" => "06 · CatSim vs BasisSimulator",
        "07_qrm_thorax_pure_material_vmi" => "07 · QRM Thorax: Dual-kVp VMI",
        "08_qrm_thorax_pure_material_pcct" => "08 · QRM Thorax: Photon-Counting VMI",
        "09_siemens_force_ufc_dual_source_vmi" => "09 · Siemens SOMATOM Force",
        "10_titanium_implant" => "10 · Titanium Implant Artifacts",
        "11_helical_scanning" => "11 · Helical Scanning",
        "12_siemens_flash_ufc" => "12 · Siemens SOMATOM Definition Flash",
        # BasisMonteCarlo.jl (lib/BasisMonteCarlo/docs/notebooks, rendered by docs/extract_montecarlo.jl)
        "mc-01_cdte_slab_efficiency" => "MC 01 · A CdTe Slab in Geant4",
        "mc-02_benchmark_hajdok_swank" => "MC 02 · Benchmark: the Swank Factor",
        "mc-03_alpha_detector" => "MC 03 · The NAEOTOM Alpha Detector",
        "mc-04_siemens_force" => "MC 04 · The SOMATOM Force Detector",
        "mc-05_siemens_flash" => "MC 05 · The SOMATOM Definition Flash Detector",
        "mc-06_ge_revolution_apex" => "MC 06 · The GE Gemstone Detector",
        "mc-07_philips_brilliance_ict" => "MC 07 · The Philips Brilliance iCT Detector",
        "mc-08_canon_aquilion_one" => "MC 08 · The Canon Aquilion ONE Detector",
        "mc-09_scatter_water_cylinders" => "MC 09 · Scatter in Water Cylinders",
        "mc-10_scatter_library" => "MC 10 · A Scatter Library",
        "mc-11_scatter_validation" => "MC 11 · Validating the Scatter Library",
    )
    get(overrides, slug, replace(slug, "_" => " "))
end

"""The notebook's sections, `(id, label)` for every second-level heading, for the page's table of contents."""
function _notebook_sections(html::AbstractString)
    unescape(s) = replace(
        s,
        r"&#(\d+);" => m -> string(Char(parse(Int, m[3:end - 1]))),
        "&amp;" => "&",
        "&lt;" => "<",
        "&gt;" => ">",
        "&quot;" => "\"",
    )
    return [
        (unescape(m[1]), unescape(strip(replace(m[2], r"<[^>]+>" => ""))))
        for m in eachmatch(r"<h2\b[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</h2>"s, html)
    ]
end

const _NOTEBOOK_GITHUB_BASE = "https://github.com/MolloiLab/BasisSimulator.jl/blob/main/docs/notebooks"
const _MC_NOTEBOOK_GITHUB_BASE = "https://github.com/MolloiLab/BasisSimulator.jl/blob/main/lib/BasisMonteCarlo/docs/notebooks"

"""The notebooks that have a page, in gallery order (BasisSimulator's, then BasisMonteCarlo's `mc-` ones): every
rendered fragment with its source, less those `BASISSIM_SKIP_NOTEBOOKS` leaves out (a trailing `*` matches a prefix, as in app.jl)."""
function _notebook_neighbours(static_dir::AbstractString)
    patterns = strip.(split(
        get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""),
        ",";
        keepempty = false,
    ))
    skipped(s) = any(p -> endswith(p, "*") ? startswith(s, chop(p)) : s == p, patterns)
    docs_dir = joinpath(static_dir, "..")
    source(s) = startswith(s, "mc-") ?
        joinpath(
            docs_dir,
            "..",
            "lib",
            "BasisMonteCarlo",
            "docs",
            "notebooks",
            "$(chopprefix(s, "mc-")).jl",
        ) :
        joinpath(docs_dir, "notebooks", "$(s).jl")
    slugs = [
        chopsuffix(f, ".fragment.html")
        for f in readdir(static_dir)
        if endswith(f, ".fragment.html")
    ]
    return sort(filter(s -> !skipped(s) && isfile(source(s)), slugs))
end

"""A display title split at its first ` · ` into the numeral (`01`, `MC 04`) and the name."""
function _title_parts(title::AbstractString)
    parts = split(title, " · "; limit = 2)
    return length(parts) == 2 ? (String(parts[1]), String(parts[2])) : ("", String(title))
end

"""A glass tile to the previous or next notebook, at the foot of a notebook page."""
function _notebook_neighbour_tile(BASE, slug, direction::Symbol)
    numeral, name = _title_parts(_notebook_display_title(slug))
    previous = direction === :previous
    return A(
        :href => "$(BASE)/examples/$(slug)/",
        :class =>
            "bs-tile p-5 sm:p-6 gap-1.5 " *
                (previous ? "items-start text-left" : "items-end text-right"),
        Span(:class => "bs-eyebrow text-[0.6875rem]", previous ? "← Previous" : "Next →"),
        Span(
            :class => "bs-tile-title bs-title text-xl sm:text-2xl",
            Span(:class => "bs-numeral", numeral),
            isempty(numeral) ? "" : " ",
            name,
        ),
    )
end

function NotebookPage(slug::AbstractString)
    BASE = get(ENV, "BASISSIM_BASE", "")
    numeral, name = _title_parts(_notebook_display_title(slug))
    is_mc = startswith(slug, "mc-")
    # A BasisMonteCarlo notebook's slug is `mc-` and the stem of its file in lib/BasisMonteCarlo.
    github_src_url = is_mc ?
        "$(_MC_NOTEBOOK_GITHUB_BASE)/$(chopprefix(slug, "mc-")).jl" :
        "$(_NOTEBOOK_GITHUB_BASE)/$(slug).jl"
    static_dir = normpath(joinpath(@__DIR__, "..", "..", "notebooks-static"))
    fragment_path = joinpath(static_dir, "$(slug).fragment.html")
    isfile(fragment_path) ||
        error(
            "Missing notebook fragment $(fragment_path). Run `docs/render_notebooks.sh $(slug)`.",
        )
    notebook_html = replace(
        read(fragment_path, String),
        "__BASISSIM_NOTEBOOKS_BASE__" => "$(BASE)/notebooks-static",
    )
    order = _notebook_neighbours(static_dir)
    k = findfirst(==(slug), order)
    previous = k === nothing || k == 1 ? nothing : order[k - 1]
    next = k === nothing || k == length(order) ? nothing : order[k + 1]

    # The notebook takes the page's width beside its table of contents (see `PageWithTOC`), floating in a large
    # glass island; its prose keeps a readable measure while figures and code use the width (input.css,
    # "Notebook pages").
    return PageWithTOC(
        _notebook_sections(notebook_html),
        Div(
            :class => "space-y-8 sm:space-y-10",
            # Back pill (left) + source pill (right)
            Div(
                :class => "flex items-center justify-between gap-3",
                A(
                    :href => "$(BASE)/examples/",
                    :class => "bs-pill bs-pill-sm",
                    "← Examples",
                ),
                A(
                    :href => github_src_url,
                    :target => "_blank",
                    :rel => "noopener",
                    :class => "bs-pill bs-pill-sm",
                    RawHtml(GITHUB_ICON),
                    Span(:class => "hidden sm:inline", "View Source"),
                ),
            ),
            # The title: the numeral in violet italic, then the name
            Div(
                :class => "relative space-y-4",
                Div(:class => "bs-glow"),
                P(
                    :class => "relative bs-eyebrow",
                    is_mc ?
                        "BasisMonteCarlo.jl · Pluto notebook" :
                        "BasisSimulator.jl · Pluto notebook",
                ),
                H1(
                    :class => "relative bs-display text-[2.5rem] sm:text-6xl",
                    isempty(numeral) ? Fragment() : Span(:class => "bs-numeral", numeral),
                    isempty(numeral) ?
                        Fragment() :
                        Span(:class => "bs-muted font-normal", " · "),
                    name,
                ),
            ),
            # Native inline notebook — scoped Pluto/Snapshot styles, one page scroll, inherited Therapy theme, no
            # nested viewport or iframe chrome. The fragment stays real DOM and theme-aware inside the island.
            Div(
                :class => "bs-island px-4 sm:px-10 lg:px-14 py-7 sm:py-12",
                RawHtml(notebook_html),
            ),
            # Onward: the neighbouring notebooks, and the way back to the gallery
            Div(
                :class => "grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5 pt-2",
                previous === nothing ?
                    Div(:class => "hidden sm:block") :
                    _notebook_neighbour_tile(BASE, previous, :previous),
                next === nothing ?
                    Div(:class => "hidden sm:block") :
                    _notebook_neighbour_tile(BASE, next, :next),
            ),
            Div(
                :class => "flex justify-center",
                A(:href => "$(BASE)/examples/", :class => "bs-pill", "← All Examples"),
            ),
        ),
    )
end
