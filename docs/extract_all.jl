#!/usr/bin/env julia
# Render every Pluto notebook in docs/notebooks/*.jl to a self-contained
# HTML at docs/notebooks-static/<slug>.html — a full standalone Pluto
# export, identical to File → Export → HTML inside the Pluto IDE.
#
# Cache is keyed on the SHA-256 of the source .jl content (NOT mtime, so
# `git checkout` doesn't trigger spurious rebuilds).  The hash is embedded
# as a `<meta>` tag in the rendered HTML; on each run we compare current
# source hash to the cached hash and only re-render on mismatch.
#
# Workflow:
#   1. Render locally on whatever GPU you have (Metal / CUDA / ROCm) —
#      this populates docs/notebooks-static/<slug>.html with the hash baked in.
#   2. `git add docs/notebooks-static/*.html` and commit.
#   3. CI inherits the rendered HTML, sees a hash match, skips re-render,
#      deploys what you committed.  Re-render only happens when somebody
#      actually edits a .jl source.
#
# Force a full rebuild with: BASISSIM_FORCE_NB_REBUILD=1 julia ...
#
# Usage:
#   julia --project=docs/build_env docs/extract_all.jl

import Pkg
let build_env = joinpath(@__DIR__, "build_env")
    if Base.active_project() != joinpath(build_env, "Project.toml")
        Pkg.activate(build_env; io = devnull)
    end
end

using Pluto
using SHA: bytes2hex, sha256

const NOTEBOOKS_DIR = joinpath(@__DIR__, "notebooks")
const STATIC_OUT    = joinpath(@__DIR__, "notebooks-static")
const HASH_META     = "basissim-source-hash"

"""Discover every Pluto notebook under `docs/notebooks/`."""
function notebook_slugs()
    isdir(NOTEBOOKS_DIR) || return String[]
    [
        splitext(f)[1]
        for f in sort(readdir(NOTEBOOKS_DIR))
        if endswith(f, ".jl") && !endswith(f, ".sessions.toml")
    ]
end

"""SHA-256 of the source .jl content."""
source_hash(slug::AbstractString) =
    bytes2hex(sha256(read(joinpath(NOTEBOOKS_DIR, "$(slug).jl"))))

"""Hash recorded inside the cached HTML, or `nothing` if the cache is
missing or unreadable."""
function cached_hash(slug::AbstractString)::Union{String, Nothing}
    out = joinpath(STATIC_OUT, "$(slug).html")
    isfile(out) || return nothing
    html = read(out, String)
    m = match(Regex("<meta\\s+name=\"$(HASH_META)\"\\s+content=\"([0-9a-f]+)\""),
              html)
    m === nothing ? nothing : m.captures[1]
end

needs_rebuild(slug::AbstractString)::Bool = cached_hash(slug) != source_hash(slug)

"""Render one notebook → standalone HTML, with the source hash embedded
in a meta tag so subsequent runs can detect cache hits."""
function render_notebook!(session::Pluto.ServerSession,
                          src_path::AbstractString,
                          dst_path::AbstractString,
                          hash::AbstractString)
    nb = Pluto.SessionActions.open(session, src_path; run_async = false)
    try
        # `header_html` is spliced into the <head> of the exported HTML.
        # Using a meta tag keeps the hash machine-readable + invisible.
        header = """<meta name="$(HASH_META)" content="$(hash)">"""
        html   = Pluto.generate_html(nb; header_html = header)
        write(dst_path, html)
    finally
        Pluto.SessionActions.shutdown(session, nb)
    end
end

"""Walk every notebook, rendering only those whose source hash doesn't
match the cached hash.  Returns the full list of slugs (cached + freshly
rendered) so callers can register routes for all of them."""
function export_notebooks()
    slugs = notebook_slugs()
    isempty(slugs) && (println("[notebooks] no .jl files in $(NOTEBOOKS_DIR)"); return slugs)

    isdir(STATIC_OUT) || mkpath(STATIC_OUT)

    force     = haskey(ENV, "BASISSIM_FORCE_NB_REBUILD")
    to_render = force ? slugs : filter(needs_rebuild, slugs)

    if isempty(to_render)
        println("[notebooks] all $(length(slugs)) cached (source hash matches) — no rebuild")
        return slugs
    end

    println("[notebooks] rendering $(length(to_render))/$(length(slugs)) " *
            (force ? "(forced)" : "(source changed)") * "…")

    session = Pluto.ServerSession()
    for slug in to_render
        src  = joinpath(NOTEBOOKS_DIR, "$(slug).jl")
        dst  = joinpath(STATIC_OUT,    "$(slug).html")
        hash = source_hash(slug)
        println("  ▸ $(slug)  (hash $(first(hash, 12))…)")
        t0 = time()
        render_notebook!(session, src, dst, hash)
        kb = round(filesize(dst) / 1024; digits = 1)
        println("    ✓ $(round(time() - t0; digits = 1))s  →  $(kb) KB")
    end

    return slugs
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_notebooks()
end
