# Fatou for the Julia that `fatou format` cannot see as code: the cells of the Pluto notebooks,
# and the snippets that live inside other text.
#
#     julia docs/fatou.jl                  # check: show what would change, exit 1 if anything would
#     julia docs/fatou.jl --write          # format in place
#     julia docs/fatou.jl --write <files or folders>
#
# - Pluto notebooks (`docs/notebooks`): Pluto rewrites its
#   header (the `@bind` mock), its `# ╔═╡` markers, the package cells and the Cell order block on
#   every save, so `fatou.toml` keeps Fatou's own walk out of these folders and each cell's code is
#   formatted here, alone. Disabled cells and the package cells are left as they are.
# - Snippets: the docs site's listings (`julia(raw"""…""")`, `CodeBlock(raw"""…""")` in
#   `docs/src`), the ```julia fences of `README.md` and `AGENTS.md`, and the ```julia examples in
#   the docstrings of `src/` and `lib/BasisMonteCarlo/src/`. REPL transcripts (`julia>`), shell
#   and text listings, and interpolated strings are left alone.
#
# Every piece goes through `fatou format -` with the repository's `fatou.toml`, so it gets
# Fatou's defaults like every other file; a piece Fatou cannot parse fails the check. Lint the
# notebooks by name: `fatou lint docs/notebooks/*.jl`.
# Needs only Julia and `fatou` on the PATH.

const ROOT = dirname(@__DIR__)

# lib/BasisMonteCarlo/docs/notebooks joins once its notebooks re-render without re-running
# Geant4 (a render must never rewrite the tables they make): until then they are left as they are.
const NOTEBOOK_DIRS = ["docs/notebooks"]
const SNIPPET_PATHS = [
    "docs/src",
    "docs/app.jl",
    "README.md",
    "AGENTS.md",
    "src",
    "lib/BasisMonteCarlo/src",
]

const MARKER = r"^# ╔═╡ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
const PACKAGE_CELLS = (
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
)
# A docs listing: `julia(` or `CodeBlock(`, a (raw) triple-quoted string, then the call's other
# arguments up to its closing parenthesis (where a `language` keyword would be)
const LISTING = r"\b(?:julia|CodeBlock)\(\s*(raw)?\"\"\"(.*?)\"\"\"([^)]*\))"s
# A ```julia fence, in Markdown or in a docstring, closed at the same indentation
const FENCE = r"^([ \t]*)```julia[ \t]*\n(.*?)^\1```[ \t]*$"ms

"""Fatou's formatting of `code`, or `nothing` and its message when it cannot parse it."""
function fatou(code::AbstractString)
    out = IOBuffer()
    err = IOBuffer()
    proc = run(pipeline(
        ignorestatus(Cmd(`fatou format -`; dir = ROOT));
        stdin = IOBuffer(code),
        stdout = out,
        stderr = err,
    ))
    success(proc) || return nothing, String(take!(err))
    return String(take!(out)), ""
end

"""Bytes `a:b` of `s` as a String (regex offsets are byte offsets)."""
bytes(s, a, b) = String(codeunits(s)[a:b])

"""
`(new_text, changed, failures)` for a notebook: each cell's code formatted alone.
`changed` lists `(label, old, new)`; `failures` lists `(label, error)`.
"""
function format_notebook(text)
    lines = split(text, '\n')
    out = String[]
    changed = Tuple{String, String, String}[]
    failures = Tuple{String, String}[]
    i = 1
    while i <= length(lines)
        m = match(MARKER, lines[i])
        if m === nothing
            push!(out, lines[i])
            i += 1
            continue
        end
        cell = m.captures[1]
        push!(out, lines[i])
        i += 1
        meta = String[]
        while i <= length(lines) && startswith(lines[i], "# ╠═╡ ")
            push!(meta, lines[i])
            i += 1
        end
        append!(out, meta)
        j = i
        while j <= length(lines) && !startswith(lines[j], "# ╔═╡ ")
            j += 1
        end
        body = lines[i:(j - 1)]
        # the code, then the blank line(s) Pluto writes before the next marker
        trailing = 0
        while !isempty(body) && body[end] == ""
            pop!(body)
            trailing += 1
        end
        code = join(body, '\n')
        skip = cell in PACKAGE_CELLS ||
            any(x -> occursin("disabled = true", x), meta) ||
            startswith(code, "#=╠═╡")
        new = code
        if !skip && !isempty(strip(code))
            formatted, err = fatou(code * "\n")
            if formatted === nothing
                push!(failures, ("cell $(cell)", err))
            else
                new = rstrip(formatted, '\n')
                new == code || push!(changed, ("cell $(cell)", code, new))
            end
        end
        isempty(new) || append!(out, split(new, '\n'))
        append!(out, fill("", trailing))
        i = j
    end
    return join(out, '\n'), changed, failures
end

"""
Every snippet of a file as `(first byte, last byte, code, rebuild)`, where `rebuild(new)` is
the text that replaces those bytes.
"""
function snippets(rel, text)
    found = Tuple{Int, Int, String, Function}[]
    if endswith(rel, ".jl") && startswith(rel, "docs/")
        for m in eachmatch(LISTING, text)
            raw, body, tail = m.captures
            occursin(r"language\s*=\s*\"(bash|text|toml|shell)\"", tail) && continue
            raw === nothing &&
                occursin('$', body) && continue # interpolated: not literal code
            occursin("julia>", body) && continue
            first_byte = m.offsets[2]
            last_byte = first_byte + ncodeunits(body) - 1
            push!(found, (first_byte, last_byte, String(body), identity))
        end
    end
    for m in eachmatch(FENCE, text)
        _, body = m.captures
        occursin("julia>", body) && continue
        lines = split(rstrip(body, '\n'), '\n')
        width = minimum(
            (length(x) - length(lstrip(x)) for x in lines if !isempty(strip(x)));
            init = typemax(Int),
        )
        width = width == typemax(Int) ? 0 : width
        code = join(
            (isempty(strip(x)) ? "" : x[(nextind(x, 0, width + 1)):end] for x in lines),
            '\n',
        )
        pad = " "^width
        rebuild(new) = join(
            (isempty(strip(x)) ? "" : pad * x for x in split(new, '\n')),
            '\n',
        ) *
            "\n"
        first_byte = m.offsets[2]
        last_byte = first_byte + ncodeunits(body) - 1
        push!(found, (first_byte, last_byte, code, rebuild))
    end
    return sort!(found; by = first)
end

"""`(new_text, changed, failures)` for a file's snippets, as `format_notebook` returns them."""
function format_snippets(rel, text)
    changed = Tuple{String, String, String}[]
    failures = Tuple{String, String}[]
    out = IOBuffer()
    last = 1
    for (a, b, code, rebuild) in snippets(rel, text)
        isempty(strip(code)) && continue
        label = "snippet `$(first(split(strip(code), '\n')))`"
        formatted, err = fatou(rstrip(code, '\n') * "\n")
        if formatted === nothing
            push!(failures, (label, err))
            continue
        end
        old = rstrip(code, '\n')
        new = rstrip(formatted, '\n')
        new == old && continue
        push!(changed, (label, old, new))
        write(out, bytes(text, last, a - 1))
        write(out, rebuild(new))
        last = b + 1
    end
    write(out, bytes(text, last, ncodeunits(text)))
    return String(take!(out)), changed, failures
end

is_notebook(text) = startswith(text, "### A Pluto.jl notebook ###")

"""Every `.jl` / `.md` file under `path` (or `path` itself)."""
function files_under(path)
    isfile(path) && return [path]
    isdir(path) || error("no such file or folder: $(path)")
    return sort!([
        joinpath(dir, f)
        for (dir, _, fs) in walkdir(path)
        for f in fs
        if endswith(f, ".jl") || endswith(f, ".md")
    ])
end

"""A unified diff of `old` and `new` (through `diff`, if it is installed)."""
function show_diff(label, old, new)
    Sys.which("diff") === nothing && return println("  ", label)
    mktempdir() do dir
        a = joinpath(dir, "before.jl")
        b = joinpath(dir, "after.jl")
        write(a, old * "\n")
        write(b, new * "\n")
        diff = read(
            ignorestatus(`diff -u --label $(label) --label "$(label) (fatou)" $a $b`),
            String,
        )
        print(diff)
    end
    return nothing
end

function main(args)
    write_mode = "--write" in args
    given = [abspath(a) for a in args if !startswith(a, "--")]
    roots = isempty(given) ?
        [joinpath(ROOT, p) for p in vcat(NOTEBOOK_DIRS, SNIPPET_PATHS)] :
        given
    paths = unique(reduce(
        vcat,
        (files_under(p) for p in roots if ispath(p));
        init = String[],
    ))
    n_files = 0
    n_pieces = 0
    n_failed = 0
    for path in paths
        rel = relpath(path, ROOT)
        text = read(path, String)
        new, changed, failures = is_notebook(text) ?
            format_notebook(text) :
            format_snippets(rel, text)
        for (label, err) in failures
            n_failed += 1
            println("$(rel): Fatou cannot parse $(label): $(first(split(err, '\n')))")
        end
        isempty(changed) && continue
        n_files += 1
        n_pieces += length(changed)
        if write_mode
            write(path, new)
            println("formatted $(length(changed)) piece(s) of $(rel)")
        else
            println("would reformat $(length(changed)) piece(s) of $(rel)")
            for (label, old, fmt) in changed
                show_diff("$(rel) $(label)", old, fmt)
            end
        end
    end
    verb = write_mode ? "formatted" : "would reformat"
    failed = n_failed > 0 ? "; $(n_failed) piece(s) Fatou cannot parse" : ""
    println(
        "$(n_pieces) piece(s) in $(n_files) of $(length(paths)) file(s) $(verb)$(failed)",
    )
    return n_failed > 0 || (n_pieces > 0 && !write_mode) ? 1 : 0
end

exit(main(ARGS))
