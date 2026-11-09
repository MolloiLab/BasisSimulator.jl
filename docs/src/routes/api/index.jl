# /api/ — the API reference, GENERATED at build time from BasisSimulator's docstrings.
#
# Nothing on this page is hand-written about the API: every exported name is listed with its
# signature(s), its rendered docstring and a link to its source on GitHub. The section a name
# appears in comes from docs/src/api_sections.jl (a small, dependency-free map shared with
# test/docs.jl); a name the map does not place lands in "Other" and fails the test suite.
#
# Therapy includes this file into its `TherapyApp` module; its last expression is the route.

import Markdown
import BasisSimulator

include(joinpath(@__DIR__, "..", "..", "api_sections.jl"))

# The site documents a release (it is built from the release tag), so source links point at that
# tag's code; their line anchors stay right however far `main` moves on.
const _API_GITHUB_SRC = "https://github.com/MolloiLab/BasisSimulator.jl/blob/v$(pkgversion(BasisSimulator))/src"

# ── classes ───────────────────────────────────────────────────────────────────────────────────
# A docstring's elements are styled by their container (`.bs-api-doc`, `.bs-api-intro` in input.css), so
# most tags get no class of their own; signatures and code listings get the dark slab.
const _API_CLS = Dict(
    "p" => "",
    "code" => "",
    "pre" => "",
    "sig" => "bs-api-sig",
    "ul" => "",
    "ol" => "",
    "h" => "",
    "table" => "",
    "th" => "",
    "td" => "",
    "blockquote" => "",
    "a" => "",
    "admonition" => "",
)

_api_esc(s) = replace(
    string(s),
    '&' => "&amp;",
    '<' => "&lt;",
    '>' => "&gt;",
    '"' => "&quot;",
)
_api_unesc(s) = replace(
    replace(s, r"&#(\d+);" => m -> string(Char(parse(Int, m[3:end - 1])))),
    "&lt;" => "<",
    "&gt;" => ">",
    "&quot;" => "\"",
    "&amp;" => "&",
)

# `[`name`](@ref)` → an in-page link when `name` is on this page, plain code otherwise.
function _api_ref_target(txt, onpage)
    t = replace(_api_unesc(txt), r"^BasisSimulator\." => "")
    t = strip(first(split(t, r"[\(\{\s]"; limit = 2)))
    t in onpage ? String(t) : nothing
end

function _api_style_html(html::AbstractString, onpage::Set{String})
    C = _API_CLS
    h = replace(
        html,
        r"<a href=\"@ref[^\"]*\"><code>(.*?)</code></a>" =>
            function (m)
                inner = match(r"<code>(.*?)</code>", m).captures[1]
                tgt = _api_ref_target(inner, onpage)
                tgt === nothing ?
                    "<code>$inner</code>" :
                    "<a href=\"#$(_api_esc(tgt))\" data-api-ref><code>$inner</code></a>"
            end,
    )
    h = replace(h, r"<a href=\"@ref[^\"]*\">(.*?)</a>" => s"\1")
    h = replace(
        h,
        "<pre><code class=\"language-" =>
            "<pre class=\"$(C["pre"])\"><code class=\"language-",
        "<pre><code>" => "<pre class=\"$(C["pre"])\"><code class=\"language-text\">",
    )
    h = replace(h, "<code>" => "<code class=\"$(C["code"])\">")
    h = replace(
        h,
        r"<a href=\"(?!#)" =>
            "<a class=\"$(C["a"])\" target=\"_blank\" rel=\"noopener\" href=\"",
    )
    h = replace(h, " data-api-ref>" => " class=\"$(C["a"])\">")
    h = replace(h, r"<h[1-6]>" => "<h4 class=\"$(C["h"])\">", r"</h[1-6]>" => "</h4>")
    for t in ("p", "ul", "ol", "table", "th", "td", "blockquote")
        h = replace(h, "<$t>" => "<$t class=\"$(C[t])\">")
    end
    h = replace(
        h,
        "<div class=\"admonition" => "<div class=\"$(C["admonition"]) admonition",
    )
    # admonition titles are plain text in Julia Markdown: render their `code` words as code
    replace(
        h,
        r"<p class=\"admonition-title\">(.*?)</p>" =>
            t -> replace(t, r"`([^`]+)`" => s"<code>\1</code>"),
    )
end

_api_md_html(md, onpage) = replace(
    _api_style_html(Markdown.html(md), onpage),
    r" class=\"\s*\"" => "",
    "class=\" admonition" => "class=\"admonition",
)

# ── collecting the docstrings ─────────────────────────────────────────────────────────────────

function _api_kind(M::Module, n::Symbol)
    isdefined(M, n) || return "binding"
    v = getfield(M, n)
    v isa Module && return "module"
    if v isa Type
        t = Base.unwrap_unionall(v)
        t isa DataType || return "type"
        t <: Enum && return "enum"
        isabstracttype(t) && return "abstract type"
        return ismutabletype(t) ? "mutable struct" : "struct"
    end
    v isa Function && return startswith(string(n), "@") ? "macro" : "function"
    return "constant"
end

# Julia Markdown reads an underscore inside a word (`large_body`, `μ_water`, `n_cols`) as the start of
# emphasis, so docstring prose written without backticks would lose its underscores to italics.
# Re-parse the raw text with every intra-word underscore escaped — outside fenced blocks, indented
# code and inline code spans, which Markdown leaves alone anyway.
function _api_escape_underscores(text::AbstractString)
    out = IOBuffer()
    fenced = false
    for line in split(text, '\n'; keepempty = true)
        if startswith(lstrip(line), "```")
            fenced = !fenced
            println(out, line)
            continue
        end
        if fenced || startswith(line, "    ") || startswith(line, "\t")
            println(out, line)
            continue
        end
        # split into code spans and prose; escape only the prose
        parts = split(line, r"(`+)"; keepempty = true)
        ticks = [m.match for m in eachmatch(r"`+", line)]
        incode = false
        opener = ""
        for (i, part) in enumerate(parts)
            print(
                out,
                incode ?
                    part :
                    replace(part, r"(?<=[\p{L}\p{N}])_(?=[\p{L}\p{N}])" => "\\_"),
            )
            i <= length(ticks) || break
            t = ticks[i]
            if !incode
                incode = true
                opener = t
            elseif t == opener
                incode = false
            end
            print(out, t)
        end
        println(out)
    end
    return String(chomp(String(take!(out))))
end

_api_parse(d) = isempty(d.text) ?
    d.object :
    Markdown.parse(_api_escape_underscores(join(string.(d.text))))

# Flatten `MD(MD(...))` nesting to a block list.
_api_blocks(md) = length(md.content) == 1 && md.content[1] isa Markdown.MD ?
    _api_blocks(md.content[1]) :
    md.content

function _api_fallback_signature(M, n, kind)
    kind == "module" && return "module $(nameof(getfield(M, n)))"
    kind in
        (
            "struct",
            "mutable struct",
            "abstract type",
            "enum",
            "type",
        ) && return "$(kind == "enum" ? "@enum" : kind) $(n)"
    kind == "constant" && return "$(n)::$(typeof(getfield(M, n)))"
    if kind == "function"
        ms = [m for m in methods(getfield(M, n)) if parentmodule(m) === M]
        sigs = unique([
            replace(sprint(show, m), r"\s+@\s.*$"s => "", "BasisSimulator." => "")
            for m in ms
        ])
        isempty(sigs) || return join(sigs, "\n")
    end
    return string(n)
end

"""Every exported name: its section, kind, and one `(sig, body, path, line)` per docstring."""
function _api_collect(M::Module)
    meta = Base.Docs.meta(M)
    names_ = sort!(filter(n -> n !== nameof(M), names(M)); by = n -> lowercase(string(n)))
    onpage = Set(string.(names_))
    entries = []
    for n in names_
        kind = _api_kind(M, n)
        b = Base.Docs.Binding(M, n)
        docs = []
        if haskey(meta, b)
            md = meta[b]
            for sig in md.order
                d = md.docs[sig]
                blocks = copy(_api_blocks(_api_parse(d)))
                signature = nothing
                if !isempty(blocks) &&
                        blocks[1] isa Markdown.Code &&
                        blocks[1].language in ("", "julia") &&
                        occursin(replace(string(n), "!" => ""), blocks[1].code)
                    signature = blocks[1].code
                    popfirst!(blocks)
                end
                path, line = api_doc_location(d.data)
                push!(
                    docs,
                    (
                        sig = signature,
                        body = _api_md_html(Markdown.MD(blocks), onpage),
                        path = path,
                        line = line,
                    ),
                )
            end
        end
        path = isempty(docs) ? nothing : docs[1].path
        push!(
            entries,
            (
                name = string(n),
                kind = kind,
                section = api_section(string(n), path),
                docs = docs,
                path = path,
                line = isempty(docs) ? 0 : docs[1].line,
                fallback = _api_fallback_signature(M, n, kind),
            ),
        )
    end
    return entries, onpage
end

# ── rendering ─────────────────────────────────────────────────────────────────────────────────

_api_srcurl(path, line) = "$(_API_GITHUB_SRC)/$(path)" * (line > 0 ? "#L$(line)" : "")

function _api_source_link(path, line)
    path === nothing && return ""
    """<a href="$(_api_esc(_api_srcurl(path, line)))" target="_blank" rel="noopener" title="src/$(_api_esc(path))$(line > 0 ? ":$line" : "") on GitHub" class="bs-api-src">Source ↗</a>"""
end

_api_sig_html(
    code,
) = """<pre class="$(_API_CLS["sig"])"><code class="language-julia">$(_api_esc(code))</code></pre>"""

function _api_entry_html(e)
    io = IOBuffer()
    id = _api_esc(e.name)
    print(
        io,
        """<article id="$id" data-api-entry="$(_api_esc(lowercase(e.name)))" class="bs-api-entry">""",
    )
    print(
        io,
        """<header class="bs-api-head">""",
        """<a href="#$id" class="bs-api-name">$id</a>""",
        """<span class="bs-api-kind">$(_api_esc(e.kind))</span>""",
        """<span class="flex-1"></span>""",
    )
    length(e.docs) == 1 && print(io, _api_source_link(e.path, e.line))
    print(io, "</header>")
    if isempty(e.docs)
        print(
            io,
            _api_sig_html(e.fallback),
            """<p class="bs-api-missing">No docstring yet.</p>""",
        )
    end
    for (i, d) in enumerate(e.docs)
        i > 1 && print(io, """<hr class="bs-api-rule">""")
        length(e.docs) > 1 &&
            print(
                io,
                """<div class="flex justify-end">""",
                _api_source_link(d.path, d.line),
                "</div>",
            )
        print(io, _api_sig_html(something(d.sig, e.fallback)))
        print(io, """<div class="bs-api-doc">""", d.body, "</div>")
    end
    print(io, "</article>")
    String(take!(io))
end

function _api_section_html(sec, entries, onpage)
    io = IOBuffer()
    print(
        io,
        """<div class="space-y-4"><div class="bs-api-intro">""",
        _api_md_html(Markdown.parse(sec.intro), onpage),
        "</div>",
    )
    if !isempty(entries)
        print(io, """<div class="flex flex-wrap gap-1.5 pt-1">""")
        for e in entries
            print(
                io,
                """<a href="#$(_api_esc(e.name))" data-api-chip="$(_api_esc(lowercase(e.name)))" class="bs-api-chip">$(_api_esc(e.name))</a>""",
            )
        end
        print(io, "</div>")
        lastpath = missing
        for e in entries
            if !isequal(e.path, lastpath)
                lastpath = e.path
                label = e.path === nothing ? "undocumented" : "src/" * e.path
                print(io, """<p data-api-file class="bs-api-file">$(_api_esc(label))</p>""")
            end
            print(io, _api_entry_html(e))
        end
    end
    print(io, "</div>")
    String(take!(io))
end

# Tiny, dependency-free name filter. `data-therapy-rerun` makes Therapy's client router run it
# again after an SPA navigation swaps this page in.
const _API_FILTER_JS = """
<script data-therapy-rerun>(function(){var i=document.getElementById('api-filter');if(!i||i.dataset.bound)return;i.dataset.bound='1';
var es=[].slice.call(document.querySelectorAll('[data-api-entry]')),cs=[].slice.call(document.querySelectorAll('[data-api-chip]')),
ss=[].slice.call(document.querySelectorAll('[data-api-section]')),fs=[].slice.call(document.querySelectorAll('[data-api-file]')),c=document.getElementById('api-filter-count');
function run(){var q=i.value.trim().toLowerCase(),n=0;
es.forEach(function(e){var m=!q||e.getAttribute('data-api-entry').indexOf(q)>=0;e.style.display=m?'':'none';if(m)n++;});
cs.forEach(function(e){e.style.display=(!q||e.getAttribute('data-api-chip').indexOf(q)>=0)?'':'none';});
fs.forEach(function(f){f.style.display=q?'none':'';});
ss.forEach(function(s){var any=false;s.querySelectorAll('[data-api-entry]').forEach(function(e){if(e.style.display!=='none')any=true;});s.style.display=(q&&!any)?'none':'';});
if(c)c.textContent=q?(n+' of '+es.length+' names match'):(es.length+' exported names');}
i.addEventListener('input',run);
document.addEventListener('keydown',function(ev){var a=document.activeElement;if(ev.key==='/'&&a!==i&&!/^(input|textarea|select)\$/i.test((a&&a.tagName)||'')){ev.preventDefault();i.focus();}});
run();})();</script>
"""

let BASE = get(ENV, "BASISSIM_BASE", "")
    () -> begin
        entries, onpage = _api_collect(BasisSimulator)
        by_section = Dict{String, Vector{Any}}()
        for e in entries
            push!(get!(by_section, e.section, Any[]), e)
        end
        lead_rank(n) = (i = findfirst(==(n), API_LEADS); i === nothing ? typemax(Int) : i)
        for v in values(by_section)
            sort!(v; by = e -> (lead_rank(e.name), something(e.path, "~"), e.line, e.name))
        end
        shown = [
            s
            for s in API_SECTIONS
            if s.id == "overview" || !isempty(get(by_section, s.id, []))
        ]
        toc = [(s.id, s.title) for s in shown]

        PageWithTOC(
            toc,
            Div(
                :class => "space-y-10 sm:space-y-12",
                PageHeader(
                    "Generated from the docstrings",
                    ("API ", Span(:class => "bs-flourish", "Reference")),
                    "Every exported name of BasisSimulator.jl. The core surface is five structs, two entry points (",
                    Code(:class => "bs-code", "simulate!"),
                    " and ",
                    Code(:class => "bs-code", "reconstruct!"),
                    "), and helpers grouped by workflow stage. The ",
                    A(:href => "$(BASE)/examples/", :class => "bs-link", "worked examples"),
                    " show them in real pipelines.",
                ),
                # The filter bar is a direct child of the page column so it stays pinned under the nav for the whole
                # page (a sticky element only sticks within its parent).
                Div(
                    :class => "bs-search-bar max-w-4xl",
                    Div(
                        :class => "relative",
                        RawHtml(
                            """<svg class="bs-search-icon w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>""",
                        ),
                        Input(
                            :id => "api-filter",
                            :type => "search",
                            :placeholder => "Filter names…  (press / to focus)",
                            :autocomplete => "off",
                            :spellcheck => "false",
                            Symbol("aria-label") => "Filter the API reference by name",
                            :class => "bs-search",
                        ),
                        Span(
                            :id => "api-filter-count",
                            :class => "bs-search-count",
                            "$(length(entries)) exported names",
                        ),
                    ),
                ),
                [
                    Section(
                        :id => s.id,
                        :class => "max-w-4xl space-y-5 scroll-mt-40",
                        Symbol("data-api-section") => s.id,
                        H2(:class => "bs-h2", s.title),
                        RawHtml(_api_section_html(s, get(by_section, s.id, []), onpage)),
                    )
                    for s in shown
                ]...,
                RawHtml(_API_FILTER_JS),
            ),
        )
    end
end
