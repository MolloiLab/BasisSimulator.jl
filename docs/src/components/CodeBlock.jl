# CodeBlock — a dark Julia listing with a copy pill. Therapy's page shell loads Prism
# (prism-julia) and highlights every `code.language-julia`, so the source goes in as text.
# The pill copies the listing and marks itself `data-copied` for a moment; CSS (`.bs-copy`) swaps its label.

const _COPY_ICON = """<svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>"""

const _COPY_BUTTON = """<button type="button" aria-label="Copy code" onclick="(function(b){var c=b.closest('[data-codeblock]').querySelector('code');navigator.clipboard.writeText(c.innerText).then(function(){b.setAttribute('data-copied','');setTimeout(function(){b.removeAttribute('data-copied')},1400)})})(this)" class="bs-copy"><span class="bs-copy-idle inline-flex items-center gap-1.5">$(_COPY_ICON)Copy</span><span class="bs-copy-done">Copied</span></button>"""

"""
    CodeBlock(code; title = nothing, language = "julia")

A dark listing with a copy pill; `title` adds a small caption bar (e.g. the notebooks a
parameter set comes from). `language = "bash"` for shell commands, `"text"` for plain text (wrapped).
"""
function CodeBlock(code::AbstractString; title = nothing, language = "julia")
    return Div(
        Symbol("data-codeblock") => "",
        :class => "bs-codeblock",
        title === nothing ? Fragment() : Div(:class => "bs-codeblock-title", title),
        RawHtml(_COPY_BUTTON),
        Pre(
            :class => "bs-pre" * (language == "text" ? " whitespace-pre-wrap" : ""),
            Code(:class => "language-$(language)", code),
        ),
    )
end
