# CodeBlock — a dark Julia listing with a copy button. Therapy's page shell loads Prism
# (prism-julia) and highlights every `code.language-julia`, so the source goes in as text.

const _COPY_ICON = """<svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>"""

const _COPY_BUTTON = """<button type="button" aria-label="Copy code" onclick="(function(b){var c=b.closest('[data-codeblock]').querySelector('code');navigator.clipboard.writeText(c.innerText).then(function(){b.setAttribute('data-copied','');setTimeout(function(){b.removeAttribute('data-copied')},1400)})})(this)" class="group/copy absolute top-2.5 right-2.5 z-10 flex items-center gap-1.5 rounded-md border border-warm-700 bg-warm-800/90 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.15em] text-warm-400 opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-warm-100 transition"><span class="group-data-[copied]/copy:hidden">$(_COPY_ICON)</span><span class="hidden group-data-[copied]/copy:inline">copied</span></button>"""

"""
    CodeBlock(code; title = nothing)

A dark Julia listing with a copy button; `title` adds a small caption bar (e.g. the notebooks a
parameter set comes from).
"""
function CodeBlock(code::AbstractString; title = nothing)
    Div(Symbol("data-codeblock") => "", :class => "group relative rounded-xl border border-warm-800 bg-warm-900 dark:bg-warm-950 shadow-sm overflow-hidden",
        title === nothing ? Fragment() :
            Div(:class => "px-5 py-2 border-b border-warm-800 text-[10px] tracking-[0.2em] uppercase font-mono text-warm-400", title),
        RawHtml(_COPY_BUTTON),
        Pre(:class => "p-5 md:p-6 overflow-x-auto text-warm-200 text-[12.5px] leading-relaxed",
            Code(:class => "language-julia font-mono", code))
    )
end
