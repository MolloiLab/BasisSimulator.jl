#!/usr/bin/env julia
# Render the BasisMonteCarlo.jl notebooks (lib/BasisMonteCarlo/docs/notebooks/*.jl) for the
# Examples page's "Detector physics (BasisMonteCarlo.jl)" category. Each notebook `<stem>.jl` is
# exported as the slug `mc-<stem>` into docs/notebooks-static/ (the prefix keeps its slug and files
# apart from the BasisSimulator notebooks), and its first figure becomes its gallery card,
# docs/assets/mc-<stem>.webp.
#
# The exporter is docs/extract_all.jl's: that file is included below for its helpers and is NOT
# edited for the Monte Carlo notebooks, because its own bytes are part of every BasisSimulator
# export's fingerprint. What differs here:
#   • Environment: each notebook activates lib/BasisMonteCarlo/docs itself (BasisMonteCarlo and
#     BasisSimulator by path) and starts its own Distributed workers. They run on the CPU.
#   • Fingerprint: the notebook; lib/BasisMonteCarlo/src/**; lib/BasisMonteCarlo/Project.toml (not
#     its `version` line); lib/BasisMonteCarlo/docs/Project.toml and its Manifest-v*.toml lockfiles; the committed
#     reference data in lib/BasisMonteCarlo/docs/reference_data/; and the exporter (Snapshot's
#     locked tree, docs/build_env/Manifest.toml, docs/extract_all.jl, this file and
#     MC_EXPORT_CONTRACT). Not docs/Manifest.toml, which these notebooks never load, and not the
#     root src/ (see AGENTS.md, "The documentation site"). Mirrored in
#     docs/verify_notebook_exports.py.
#   • Names: Snapshot names an export after its notebook, so each one is exported into a scratch
#     directory and moved into docs/notebooks-static/ under its `mc-` slug.
#
# Runtimes on 8 workers: 5–15 min each; mc-10_scatter_library about 3.6 h. Render them by slug:
#   docs/render_notebooks.sh mc-04_siemens_force mc-05_siemens_flash
#
# Usage:
#   julia --project=docs/build_env docs/extract_montecarlo.jl                       # every stale one
#   julia --project=docs/build_env docs/extract_montecarlo.jl mc-04_siemens_force   # these (if stale)
#   julia --project=docs/build_env docs/extract_montecarlo.jl --list-stale          # the stale slugs
#   julia --project=docs/build_env docs/extract_montecarlo.jl --queue DIR           # render DIR/<slug>.todo
#
# Force a full rebuild with BASISSIM_FORCE_NB_REBUILD=1.

include(joinpath(@__DIR__, "extract_all.jl"))   # build_env, Pluto, Snapshot and the shared helpers

const MC_ROOT = normpath(joinpath(@__DIR__, "..", "lib", "BasisMonteCarlo"))
const MC_DOCS = joinpath(MC_ROOT, "docs")
const MC_NOTEBOOKS_DIR = joinpath(MC_DOCS, "notebooks")
const MC_PREFIX = "mc-"
const CARD_DIR = joinpath(@__DIR__, "assets")

# The kV sliders redraw CairoMakie figures from BasisSimulator spectra, outside WasmTarget's scope:
# publish them as honest static fallbacks (docs/extract_all.jl, FORCE_FALLBACK_BONDS).
const MC_FORCE_FALLBACK_BONDS = Dict(
    "mc-04_siemens_force" => (:kvp,),
    "mc-05_siemens_flash" => (:kvp,),
    "mc-06_ge_revolution_apex" => (:kvp,),
    "mc-07_philips_brilliance_ict" => (:kvp,),
    "mc-08_canon_aquilion_one" => (:kvp,),
)
const MC_EXPORT_CONTRACT = "basismc-lean-v1|therapy=true|fragment=true|islands=true|verify=true|optimize=size|cpu|forced-fallback=04,05,06,07,08:kvp|card=first-figure-webp"

"""Every BasisMonteCarlo notebook, as its `mc-` slug."""
function mc_notebook_slugs()
    isdir(MC_NOTEBOOKS_DIR) || return String[]
    return [
        MC_PREFIX * splitext(f)[1]
            for f in sort(readdir(MC_NOTEBOOKS_DIR))
            if endswith(f, ".jl")
    ]
end

mc_stem(slug::AbstractString) = chopprefix(slug, MC_PREFIX)

mc_source(slug::AbstractString) = joinpath(MC_NOTEBOOKS_DIR, "$(mc_stem(slug)).jl")

mc_source_hash(slug::AbstractString) = bytes2hex(sha256(read(mc_source(slug))))

"Digest the BasisMonteCarlo code, its locked docs environment and its committed reference data."
const _MC_INPUT_HASH = Ref{Union{Nothing, String}}(nothing)
function mc_input_hash()
    _MC_INPUT_HASH[] === nothing || return _MC_INPUT_HASH[]::String
    root = dirname(@__DIR__)
    # every lockfile: one per Julia minor version (Manifest-v1.12.toml, Manifest-v1.13.toml)
    files = vcat(
        [joinpath(MC_DOCS, "Project.toml")],
        sort([joinpath(MC_DOCS, name) for name in readdir(MC_DOCS) if occursin(r"^Manifest(-v[0-9.]+)?\.toml$", name)]),
    )
    for dir in (joinpath(MC_ROOT, "src"), joinpath(MC_DOCS, "reference_data"))
        isdir(dir) || continue
        for (base, _, names) in walkdir(dir), name in names
            path = joinpath(base, name)
            isfile(path) && push!(files, path)
        end
    end
    entries = [
        string(relpath(path, root), ":", bytes2hex(sha256(read(path))))
            for path in files
    ]
    project = joinpath(MC_ROOT, "Project.toml")
    push!(entries, string(relpath(project, root), ":", project_input_hash(project)))
    sort!(entries)
    _MC_INPUT_HASH[] = bytes2hex(sha256(join(entries, '\0')))
    return _MC_INPUT_HASH[]::String
end

"Fingerprint the notebook, the BasisMonteCarlo inputs and the exporter."
function mc_export_fingerprint(slug::AbstractString)
    snapshot_tree = string(Pkg.dependencies()[SNAPSHOT_UUID].tree_hash)
    build_lock = bytes2hex(sha256(read(joinpath(@__DIR__, "build_env", "Manifest.toml"))))
    exporter = bytes2hex(sha256(read(joinpath(@__DIR__, "extract_all.jl"))))
    driver = bytes2hex(sha256(read(@__FILE__)))
    payload = join(
        (
            mc_source_hash(slug),
            mc_input_hash(),
            snapshot_tree,
            build_lock,
            exporter,
            driver,
            MC_EXPORT_CONTRACT,
        ),
        '\0',
    )
    return bytes2hex(sha256(payload))
end

mc_needs_rebuild(slug::AbstractString) = cached_hash(slug) != mc_export_fingerprint(slug)

"""`BASISSIM_SKIP_NOTEBOOKS`, where a trailing `*` matches a prefix (`mc-*` skips every one)."""
function mc_skipped(slug::AbstractString)
    patterns = strip.(split(get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""), ","; keepempty = false))
    return any(
        p -> endswith(p, "*") ? startswith(slug, chop(p)) : slug == p,
        patterns,
    )
end

"""Export one notebook into a scratch directory, move it into docs/notebooks-static/ as
`mc-<stem>` (`move_export!`), then embed the fingerprint, shrink the figures and save the card
image."""
function render_mc_notebook!(session::Pluto.ServerSession, slug::AbstractString)
    src = mc_source(slug)
    isfile(src) || error("no notebook $(src)")
    stage = mktempdir()
    try
        export_notebook(
            src;
            output_dir = stage,
            therapy = true,
            fragment = true,
            assets_base = "__BASISSIM_NOTEBOOKS_BASE__",
            islands = true,
            verify = true,
            optimize = :size,
            force_fallback_bonds = get(MC_FORCE_FALLBACK_BONDS, slug, ()),
            env_dir = MC_DOCS,
            session = session,
        )
        move_export!(stage, STATIC_OUT, mc_stem(slug), slug)
    finally
        rm(stage; recursive = true, force = true)
    end

    fingerprint = mc_export_fingerprint(slug)
    html_path = joinpath(STATIC_OUT, "$(slug).html")
    meta = """<meta name="$(HASH_META)" content="$(fingerprint)">"""
    html = read(html_path, String)
    html = occursin("<head>", html) ?
        replace(html, "<head>" => "<head>" * meta; count = 1) :
        html * "\n" * meta * "\n"
    write(html_path, html)
    compress_page_images!(html_path)

    fragment_path = joinpath(STATIC_OUT, "$(slug).fragment.html")
    marker = "<!-- $(HASH_META): $(fingerprint) -->"
    write(fragment_path, marker * "\n" * read(fragment_path, String))
    compress_page_images!(fragment_path)

    save_card_image!(fragment_path, joinpath(CARD_DIR, "$(slug).webp"))
    return nothing
end

"""Move Snapshot's export of `<stem>.jl` from `stage` into `out` as `slug`: the standalone page,
the fragment and, when the notebook has a `@bind`, its `.islands/` directory, renaming every
reference to that directory."""
function move_export!(stage, out, stem, slug)
    old_islands = "$(stem).islands"
    new_islands = "$(slug).islands"
    missed = Regex("(?<!$(MC_PREFIX))\\Q$(old_islands)/\\E")
    for suffix in (".html", ".fragment.html")
        from = joinpath(stage, "$(stem)$(suffix)")
        isfile(from) || error("Snapshot did not emit $(basename(from))")
        html = replace(
            read(from, String),
            "<title>$(stem)</title>" => "<title>$(slug)</title>",
            "\"$(old_islands)/" => "\"$(new_islands)/",
            "/$(old_islands)/" => "/$(new_islands)/",
        )
        occursin(missed, html) &&
            error("$(basename(from)) still refers to $(old_islands)/ after the rename")
        write(joinpath(out, "$(slug)$(suffix)"), html)
    end
    islands_dst = joinpath(out, new_islands)
    rm(islands_dst; recursive = true, force = true)
    islands_src = joinpath(stage, old_islands)
    isdir(islands_src) && cp(islands_src, islands_dst)   # the scratch directory may be on another disk
    return nothing
end

"""Save the first figure of a rendered page as its gallery card (WebP, at most 1200 px), the way
the BasisSimulator notebooks save theirs into docs/assets/. A page without a figure gets no card
(the gallery shows a plain panel), and an old card is removed."""
function save_card_image!(page_path::AbstractString, card_path::AbstractString)
    py = raw"""
    import re, base64, io, os, sys
    from PIL import Image
    page, card = sys.argv[1], sys.argv[2]
    h = open(page, encoding='utf-8', errors='surrogateescape').read()
    m = re.search(r'<img\b[^>]*?\bsrc="data:image/(?:png|webp|jpeg);base64,\s*([A-Za-z0-9+/=]+)"', h)
    if m is None:
        if os.path.exists(card):
            os.remove(card)
        print("card: no figure on the page; no card")
        sys.exit(0)
    im = Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert('RGB')
    im.thumbnail((1200, 1200))
    im.save(card, 'WEBP', quality=88, method=6)
    print(f"card: {im.size[0]}x{im.size[1]} -> {os.path.basename(card)}")
    """
    run(`python3 -c $py $page_path $card_path`)
    return nothing
end

function render_mc_slug!(session, slug)
    pre_hash = mc_source_hash(slug)
    println("  ▸ $(slug)  (hash $(first(pre_hash, 12))…)")
    t0 = time()
    render_mc_notebook!(session, slug)
    post_hash = mc_source_hash(slug)
    if post_hash != pre_hash
        println("    ⚠ the notebook changed while it rendered: render $(slug) again")
    end
    kb = round(filesize(joinpath(STATIC_OUT, "$(slug).html")) / 1024; digits = 1)
    println("    ✓ $(round(time() - t0; digits = 1))s  →  $(kb) KB")
    flush(stdout)
    return nothing
end

"""Render every stale notebook (or every one of `only` that is stale)."""
function mc_export_notebooks(only = nothing)
    slugs = mc_notebook_slugs()
    if only !== nothing
        unknown = setdiff(only, slugs)
        isempty(unknown) ||
            error("no notebook named $(join(unknown, ", ")) in $(MC_NOTEBOOKS_DIR)")
        slugs = filter(in(only), slugs)
    end
    slugs = filter(!mc_skipped, slugs)
    isempty(slugs) && return slugs
    isdir(STATIC_OUT) || mkpath(STATIC_OUT)
    force = haskey(ENV, "BASISSIM_FORCE_NB_REBUILD")
    to_render = force ? slugs : filter(mc_needs_rebuild, slugs)
    if isempty(to_render)
        println("[notebooks] all $(length(slugs)) BasisMonteCarlo exports current — no rebuild")
        return slugs
    end
    println("[notebooks] rendering $(length(to_render))/$(length(slugs)) BasisMonteCarlo notebook(s)…")
    session = render_session()
    foreach(slug -> render_mc_slug!(session, slug), to_render)
    return slugs
end

"""Render `<queue>/<slug>.todo` entries one at a time until none is left (docs/extract_all.jl,
`render_queue`)."""
function mc_render_queue(queue::AbstractString)
    isdir(STATIC_OUT) || mkpath(STATIC_OUT)
    session = render_session()
    while true
        slug = claim_next(queue)
        slug === nothing && return
        render_mc_slug!(session, slug)
        rm(joinpath(queue, "$(slug).claimed.$(getpid())"))
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        mc_export_notebooks()
    elseif ARGS == ["--list-stale"]
        force = haskey(ENV, "BASISSIM_FORCE_NB_REBUILD")
        slugs = filter(!mc_skipped, mc_notebook_slugs())
        foreach(println, force ? slugs : filter(mc_needs_rebuild, slugs))
    elseif length(ARGS) == 2 && ARGS[1] == "--queue"
        mc_render_queue(ARGS[2])
    else
        mc_export_notebooks(ARGS)
    end
end
