# TableOfContents.jl — right-hand TOC beside the page's content.
#
# A column of its own at xl+ breakpoints, sticky under the nav bar, so it stays next to the text at every width
# instead of pinned to the window's edge; hidden on smaller viewports. Each entry is an anchor link to a section `id`
# within the current page.

function TableOfContents(sections::Vector{Tuple{String,String}})
    Nav(:class => "hidden xl:block sticky top-24 self-start max-h-[calc(100vh-7rem)] overflow-y-auto",
        Div(:class => "space-y-1.5 border-l border-warm-200 dark:border-warm-800 pl-3",
            P(:class => "text-[11px] font-semibold text-warm-400 dark:text-warm-500 uppercase tracking-wider mb-3", "On this page"),
            For(sections) do (id, label)
                A(:href => "#$id",
                  :class => "block text-[12px] leading-relaxed text-warm-500 dark:text-warm-400 hover:text-accent-500 dark:hover:text-accent-400 transition-colors",
                  label)
            end
        )
    )
end

function PageWithTOC(sections::Vector{Tuple{String,String}}, content)
    Div(:class => "xl:grid xl:grid-cols-[minmax(0,1fr)_11rem] xl:gap-12",
        Div(:class => "min-w-0", content),
        TableOfContents(sections),
    )
end
