# Base path read from app.jl (empty in dev, "/BasisSimulator.jl" in build).
# Hrefs are built as `$(BASE)/path/` so the same Layout works in both modes.
const BASE = get(ENV, "BASISSIM_BASE", "")

"""BasisSimulator.jl wordmark with colored .jl suffix"""
function BasisSimulatorWordmark()
    NavLink("$(BASE)/",
        RawHtml("""BasisSimulator<span style="color:var(--jl-dot)">.</span><span style="color:var(--jl-j)">j</span><span style="color:var(--jl-l)">l</span>""");
        class = "text-xl font-serif font-bold text-warm-900 dark:text-warm-100 hover:opacity-80 transition-opacity no-underline",
        active_class = ""
    )
end

function Layout(content)
    Div(:class => "min-h-screen flex flex-col overflow-x-clip bg-warm-100 dark:bg-warm-950 text-warm-800 dark:text-warm-200 transition-colors",
        # Therapy owns the light/dark preference through the `dark` class. Keep
        # Snapshot's embedded notebook token theme on the same authoritative
        # state before the notebook fragment is painted.
        RawHtml("""<style>html,body{overflow-x:clip}</style>"""),
        RawHtml("""<script>(function(){var h=document.documentElement;h.setAttribute('data-theme',h.classList.contains('dark')?'fun-dark':'fun-light');})();</script>"""),
        # Sticky top nav — always visible while content scrolls underneath.
        Header(:class => "sticky top-0 z-40 border-b border-warm-200 dark:border-warm-800 h-16 px-6 bg-warm-100/80 dark:bg-warm-950/80 backdrop-blur supports-[backdrop-filter]:bg-warm-100/60 supports-[backdrop-filter]:dark:bg-warm-950/60",
            Div(:class => "max-w-5xl mx-auto h-full flex items-center justify-between",
                BasisSimulatorWordmark(),
                Nav(Symbol("aria-label") => "Primary navigation", :class => "flex items-center gap-3 sm:gap-6",
                    Div(:class => "hidden sm:flex items-center gap-6",
                        NavLink("$(BASE)/getting-started/", "Getting Started";
                            class = "text-sm transition-colors no-underline",
                            active_class = "text-accent-600 dark:text-accent-400 font-medium",
                            inactive_class = "text-warm-600 dark:text-warm-400 hover:text-accent-600 dark:hover:text-accent-400"
                        ),
                        NavLink("$(BASE)/api/", "API";
                            class = "text-sm transition-colors no-underline",
                            active_class = "text-accent-600 dark:text-accent-400 font-medium",
                            inactive_class = "text-warm-600 dark:text-warm-400 hover:text-accent-600 dark:hover:text-accent-400"
                        ),
                        NavLink("$(BASE)/examples/", "Examples";
                            class = "text-sm transition-colors no-underline",
                            active_class = "text-accent-600 dark:text-accent-400 font-medium",
                            inactive_class = "text-warm-600 dark:text-warm-400 hover:text-accent-600 dark:hover:text-accent-400"
                        )
                    ),
                    Details(Symbol("data-mobile-nav") => "", :class => "relative sm:hidden",
                        Summary(Symbol("aria-label") => "Open navigation menu", :class => "mobile-nav-trigger text-warm-600 dark:text-warm-400 hover:text-warm-700 dark:hover:text-warm-300 transition-colors cursor-pointer rounded-md p-1",
                            RawHtml("""<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>""")
                        ),
                        Div(:class => "absolute right-0 top-full mt-3 w-44 overflow-hidden rounded-lg border border-warm-300 dark:border-warm-700 bg-warm-50 dark:bg-warm-900 shadow-lg p-1.5",
                            NavLink("$(BASE)/getting-started/", "Getting Started";
                                class = "block rounded-md px-3 py-2 text-sm transition-colors no-underline",
                                active_class = "bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300 font-medium",
                                inactive_class = "text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-800"
                            ),
                            NavLink("$(BASE)/api/", "API";
                                class = "block rounded-md px-3 py-2 text-sm transition-colors no-underline",
                                active_class = "bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300 font-medium",
                                inactive_class = "text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-800"
                            ),
                            NavLink("$(BASE)/examples/", "Examples";
                                class = "block rounded-md px-3 py-2 text-sm transition-colors no-underline",
                                active_class = "bg-accent-100 dark:bg-accent-900/50 text-accent-700 dark:text-accent-300 font-medium",
                                inactive_class = "text-warm-700 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-800"
                            )
                        )
                    ),
                    A(:href => "https://github.com/MolloiLab/BasisSimulator.jl", :target => "_blank",
                        Symbol("aria-label") => "BasisSimulator.jl on GitHub",
                        :class => "text-warm-600 dark:text-warm-400 hover:text-warm-700 dark:hover:text-warm-300 transition-colors",
                        RawHtml("""<svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>""")
                    ),
                    DarkModeToggle()
                )
            )
        ),
        # Main content — id="page-content" enables SPA navigation (router swaps this)
        MainEl(:id => "page-content", :class => "flex-1 w-full max-w-5xl mx-auto px-3 sm:px-6 py-8 sm:py-12",
            content
        ),
        # Footer — MolloiLab | MIT | Built with Therapy.jl
        Footer(:class => "border-t border-warm-200 dark:border-warm-800 px-6 py-6",
            Div(:class => "max-w-5xl mx-auto flex flex-wrap gap-x-6 gap-y-2 items-center justify-center sm:justify-between",
                A(:href => "https://github.com/MolloiLab", :target => "_blank",
                    :class => "text-sm text-warm-600 dark:text-warm-400 hover:text-warm-700 dark:hover:text-warm-300 transition-colors no-underline",
                    "MolloiLab"
                ),
                A(:href => "https://github.com/MolloiLab/BasisSimulator.jl/blob/main/LICENSE", :target => "_blank",
                    :class => "text-sm text-warm-500 dark:text-warm-500 hover:text-warm-700 dark:hover:text-warm-300 transition-colors no-underline",
                    "MIT License"
                ),
                P(:class => "text-sm text-warm-500 dark:text-warm-500",
                    "Built with ",
                    A(:href => "https://github.com/GroupTherapyOrg/Therapy.jl", :target => "_blank",
                        :class => "font-serif hover:text-warm-600 dark:hover:text-warm-300 transition-colors no-underline",
                        RawHtml("""Therapy<span style="color:var(--jl-dot)">.</span><span style="color:var(--jl-j)">j</span><span style="color:var(--jl-l)">l</span>""")
                    )
                )
            )
        ),
        # Match Snapshot's proven compact-menu behavior: close after choosing a
        # destination, clicking away, or pressing Escape. Native <details>
        # preserves keyboard toggling without a hydration dependency.
        RawHtml("""<script>(function(){var menu=document.querySelector('[data-mobile-nav]');if(!menu)return;document.addEventListener('click',function(e){if(menu.open&&(!menu.contains(e.target)||e.target.closest('[data-mobile-nav] a')))menu.removeAttribute('open');});document.addEventListener('keydown',function(e){if(e.key==='Escape'&&menu.open){menu.removeAttribute('open');var s=menu.querySelector('summary');if(s)s.focus();}});})();</script>""")
    )
end
