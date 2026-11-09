# Base path read from app.jl (empty in dev, "/BasisSimulator.jl" in build).
# Hrefs are built as `$(BASE)/path/` so the same Layout works in both modes.
const BASE = get(ENV, "BASISSIM_BASE", "")

# Primary navigation, in reading order: (path under BASE, label).
const NAV_ITEMS = [
    ("/getting-started/", "Getting Started"),
    ("/examples/", "Examples"),
    ("/scanners/", "Scanners"),
    ("/api/", "API"),
]

const GITHUB_ICON = """<svg class="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>"""

const _JL_SUFFIX = """<span style="color:var(--jl-dot)">.</span><span style="color:var(--jl-j)">j</span><span style="color:var(--jl-l)">l</span>"""

"""BasisSimulator.jl wordmark: the name in EB Garamond, the `.jl` in Julia's colours."""
function BasisSimulatorWordmark()
    return NavLink(
        "$(BASE)/",
        RawHtml("""BasisSimulator$(_JL_SUFFIX)""");
        class = "font-serif text-[1.375rem] font-semibold tracking-[-0.01em] text-[var(--ink)] no-underline hover:opacity-75 transition-opacity whitespace-nowrap",
        active_class = "",
    )
end

# The "On This Page" rail marks the section being read. Runs on the first load and after every client-side
# navigation (Therapy's router swaps #page-content and fires `therapy:router:loaded`).
const _TOC_SPY_JS = """<script>(function(){var io=null;function run(){if(io){io.disconnect();io=null;}var links=[].slice.call(document.querySelectorAll('[data-toc-link]'));if(!links.length||!('IntersectionObserver' in window))return;var byId={};links.forEach(function(a){byId[decodeURIComponent(a.getAttribute('href').slice(1))]=a;});var heads=Object.keys(byId).map(function(id){return document.getElementById(id);}).filter(Boolean);var vis=new Set();function mark(){var top=null;heads.forEach(function(h){if(vis.has(h)&&(!top||h.getBoundingClientRect().top<top.getBoundingClientRect().top))top=h;});if(!top){var y=window.innerHeight*0.3;heads.forEach(function(h){if(h.getBoundingClientRect().top<y)top=h;});}links.forEach(function(a){a.removeAttribute('data-active');});if(top&&byId[top.id])byId[top.id].setAttribute('data-active','');}io=new IntersectionObserver(function(es){es.forEach(function(e){e.isIntersecting?vis.add(e.target):vis.delete(e.target);});mark();},{rootMargin:'-80px 0px -55% 0px'});heads.forEach(function(h){io.observe(h);});mark();}window.addEventListener('therapy:router:loaded',run);document.readyState==='loading'?document.addEventListener('DOMContentLoaded',run):run();})();</script>"""

function Layout(content)
    return Div(
        :class =>
            "min-h-screen flex flex-col overflow-x-clip text-[var(--ink)] transition-colors",
        RawHtml("""<style>html,body{overflow-x:clip}</style>"""),
        # EB Garamond's italics and heavier weights (Therapy's own link loads only the roman and one italic), and
        # JetBrains Mono for code where JuliaMono is not installed.
        RawHtml(
            """<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&family=JetBrains+Mono:wght@400;500&display=swap">""",
        ),
        # Therapy owns the light/dark preference through the `dark` class. Keep Snapshot's embedded notebook token
        # theme on the same state before the notebook fragment is painted.
        RawHtml(
            """<script>(function(){var h=document.documentElement;h.setAttribute('data-theme',h.classList.contains('dark')?'fun-dark':'fun-light');})();</script>""",
        ),
        # The nav: a glass capsule floating over the page, wordmark left, pills right.
        Header(
            :class => "bs-header sticky top-0 z-40 px-3 sm:px-6 pt-3 sm:pt-4 pb-2",
            Div(
                :class =>
                    "bs-navbar max-w-7xl mx-auto h-14 pl-5 pr-2 flex items-center justify-between gap-3",
                BasisSimulatorWordmark(),
                Nav(
                    Symbol("aria-label") => "Primary navigation",
                    :class => "flex items-center gap-1",
                    Div(
                        :class => "hidden md:flex items-center gap-1",
                        [
                            NavLink(
                                "$(BASE)$(path)",
                                label;
                                class = "bs-pill",
                                active_class = "bs-pill-on",
                                inactive_class = "",
                            )
                            for (path, label) in NAV_ITEMS
                        ]...,
                    ),
                    Details(
                        Symbol("data-mobile-nav") => "",
                        :class => "relative md:hidden",
                        Summary(
                            Symbol("aria-label") => "Open navigation menu",
                            :class => "bs-pill bs-pill-icon",
                            RawHtml(
                                """<svg class="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>""",
                            ),
                        ),
                        Div(
                            :class =>
                                "bs-menu absolute right-0 top-full mt-3 w-56 p-2 flex flex-col gap-1",
                            [
                                NavLink(
                                    "$(BASE)$(path)",
                                    label;
                                    class = "bs-pill justify-start w-full",
                                    active_class = "bs-pill-on",
                                    inactive_class = "",
                                )
                                for (path, label) in NAV_ITEMS
                            ]...,
                        ),
                    ),
                    A(
                        :href => "https://github.com/MolloiLab/BasisSimulator.jl",
                        :target => "_blank",
                        :rel => "noopener",
                        Symbol("aria-label") => "BasisSimulator.jl on GitHub",
                        :class => "bs-pill bs-pill-icon",
                        RawHtml(GITHUB_ICON),
                    ),
                    DarkModeToggle(),
                ),
            ),
        ),
        # Main content — id="page-content" enables SPA navigation (router swaps this)
        MainEl(
            :id => "page-content",
            :class => "flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-20 sm:pt-16",
            content,
        ),
        Footer(
            :class => "px-4 sm:px-6 pb-12",
            Div(
                :class => "max-w-7xl mx-auto space-y-6",
                Div(:class => "bs-fleuron", Span(Symbol("aria-hidden") => "true", "❦")),
                Div(
                    :class =>
                        "flex flex-wrap gap-x-7 gap-y-2 items-center justify-center text-[0.9375rem] bs-muted",
                    A(
                        :href => "https://github.com/MolloiLab",
                        :target => "_blank",
                        :class =>
                            "no-underline hover:text-[var(--accent)] transition-colors",
                        "MolloiLab",
                    ),
                    A(
                        :href =>
                            "https://github.com/MolloiLab/BasisSimulator.jl/blob/main/LICENSE",
                        :target => "_blank",
                        :class =>
                            "no-underline hover:text-[var(--accent)] transition-colors",
                        "MIT License",
                    ),
                    A(
                        :href => "https://github.com/GroupTherapyOrg/Therapy.jl",
                        :target => "_blank",
                        :class =>
                            "no-underline hover:text-[var(--accent)] transition-colors",
                        Em("Built with "),
                        RawHtml("""Therapy$(_JL_SUFFIX)"""),
                    ),
                ),
            ),
        ),
        # Close the compact menu after choosing a destination, clicking away, or pressing Escape. Native <details>
        # keeps keyboard toggling without a hydration dependency.
        RawHtml(
            """<script>(function(){var menu=document.querySelector('[data-mobile-nav]');if(!menu)return;document.addEventListener('click',function(e){if(menu.open&&(!menu.contains(e.target)||e.target.closest('[data-mobile-nav] a')))menu.removeAttribute('open');});document.addEventListener('keydown',function(e){if(e.key==='Escape'&&menu.open){menu.removeAttribute('open');var s=menu.querySelector('summary');if(s)s.focus();}});})();</script>""",
        ),
        RawHtml(_TOC_SPY_JS),
    )
end

"""
    PageHeader(eyebrow, title, lede...)

The head of a page: a small uppercase eyebrow, the display title, and an optional lede. `title` may be a string or
a tuple of pieces (e.g. a word set in the violet italic flourish).
"""
function PageHeader(eyebrow, title, lede...)
    pieces = title isa Tuple ? title : (title,)
    return Div(
        :class => "relative space-y-5",
        Div(:class => "bs-glow"),
        P(:class => "relative bs-eyebrow", eyebrow),
        H1(:class => "relative bs-display text-5xl sm:text-6xl lg:text-7xl", pieces...),
        isempty(lede) ? Fragment() : P(:class => "relative bs-lede max-w-2xl", lede...),
    )
end
