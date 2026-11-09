# TableOfContents.jl — the "On This Page" rail beside a page's content.
#
# A column of its own at xl+ breakpoints: a small glass panel, sticky under the floating nav, so it stays next to the
# text at every width instead of floating over it; hidden on smaller viewports. Each entry is an anchor link to a
# section `id` within the current page; Layout's scroll-spy marks the one being read (`data-toc-link`).

function TableOfContents(sections::Vector{Tuple{String, String}})
    return Nav(
        Symbol("aria-label") => "On this page",
        :class => "hidden xl:block sticky top-28 self-start",
        Div(
            :class =>
                "bs-glass rounded-[1.5rem] p-2.5 max-h-[calc(100vh-8.5rem)] overflow-y-auto",
            P(:class => "bs-eyebrow text-[0.6875rem] px-3 pt-2 pb-2", "On This Page"),
            For(sections) do (id, label)
                A(
                    :href => "#$id",
                    Symbol("data-toc-link") => "",
                    :class => "bs-toc-link",
                    label,
                )
            end,
        ),
    )
end

function PageWithTOC(sections::Vector{Tuple{String, String}}, content)
    return Div(
        :class => "xl:grid xl:grid-cols-[minmax(0,1fr)_14rem] xl:gap-10",
        Div(:class => "min-w-0", content),
        TableOfContents(sections),
    )
end
