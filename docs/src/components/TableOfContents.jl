# TableOfContents.jl — floating right-hand TOC, same pattern as Therapy.jl docs.
#
# Fixed to the right side at xl+ breakpoints; hidden on smaller viewports.
# Each entry is an anchor link to a section `id` within the current page.

function TableOfContents(sections::Vector{Tuple{String,String}})
    Nav(:class => "hidden xl:block fixed right-8 top-24 w-44",
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
    Fragment(
        content,
        TableOfContents(sections),
    )
end
