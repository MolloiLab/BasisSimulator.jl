#!/usr/bin/env julia
# Render every Pluto notebook in docs/notebooks/*.jl to a LEAN standalone
# HTML at docs/notebooks-static/<slug>.html via Snapshot.jl's Therapy emit
# (therapy=true): rendered cell outputs as plain HTML + WASM islands where a
# @bind group compiles — and NO embedded Pluto statefile. The old
# Pluto.generate_html path baked the full notebook state into every page
# (17–42 MB each, 141 MB total); the lean pages are ~100× smaller, which is
# what lets the site fit through Snapshot's publish pipeline.
#
# Cache is keyed on a SHA-256 fingerprint of the source notebook, its data declarations, the docs
# environment (Snapshot's locked tree, the build and docs lockfiles) and the export contract (NOT
# mtime, so `git checkout` doesn't trigger spurious rebuilds), embedded in both exported forms.
# The package's src/ is not in it: a code change does not make every page stale. Re-render the
# notebooks a change affects by name, or all with BASISSIM_FORCE_NB_REBUILD=1.
#
# Workflow (RELEASING.md, step 6):
#   1. Render on a GPU machine: `docs/render_notebooks.sh` runs one of these processes per GPU,
#      each taking notebooks from a shared queue. The notebooks pick the backend themselves
#      (CUDA when it is functional, the CPU otherwise), after docs/gpu_agreement.jl passes.
#   2. `git add docs/notebooks-static/` and commit.
#   3. CI re-computes the fingerprint (docs/verify_notebook_exports.py) and refuses stale exports;
#      it never renders.
#
# Force a full rebuild with BASISSIM_FORCE_NB_REBUILD=1.
#
# Usage:
#   julia --project=docs/build_env docs/extract_all.jl                 # every stale notebook
#   julia --project=docs/build_env docs/extract_all.jl 04_pcct_vmi     # these slugs (if stale)
#   julia --project=docs/build_env docs/extract_all.jl --list-stale    # print the stale slugs
#   julia --project=docs/build_env docs/extract_all.jl --queue DIR     # render DIR/<slug>.todo, one
#                                                                      #   at a time, until none is left

import Pkg
let build_env = joinpath(@__DIR__, "build_env")
    if Base.active_project() != joinpath(build_env, "Project.toml")
        Pkg.activate(build_env; io = devnull)
    end
end

using Pluto
using Snapshot: export_notebook
using SHA: bytes2hex, sha256
using UUIDs: UUID

const NOTEBOOKS_DIR = joinpath(@__DIR__, "notebooks")
const STATIC_OUT = joinpath(@__DIR__, "notebooks-static")
const DATA_DIR = joinpath(NOTEBOOKS_DIR, "data")
const DATA_PROVENANCE = joinpath(NOTEBOOKS_DIR, "DATA_PROVENANCE.sha256")
const HASH_META = "basissim-export-fingerprint"
const SNAPSHOT_UUID = UUID("1e64ef43-5f79-4f6b-8e97-159df4e27032")
const FORCE_FALLBACK_BONDS = Dict(
    "01_five_struct_api" => (:z_slice,),
    "05_xcat_grid_to_recon" => (:z_helical,),
    "11_helical_scanning" => (:z_idx,),
)
const EXPORT_CONTRACT = "basissim-lean-v4|therapy=true|fragment=true|islands=true|verify=true|optimize=size|forced-fallback=01:z_slice,05:z_helical,11:z_idx"

"""Discover every Pluto notebook under `docs/notebooks/`."""
function notebook_slugs()
    isdir(NOTEBOOKS_DIR) || return String[]
    return [
        splitext(f)[1]
        for f in sort(readdir(NOTEBOOKS_DIR))
        if endswith(f, ".jl") && !endswith(f, ".sessions.toml")
    ]
end

"""SHA-256 of the source .jl content."""
source_hash(slug::AbstractString) = bytes2hex(sha256(read(joinpath(
    NOTEBOOKS_DIR,
    "$(slug).jl",
))))

# The repository's own packages, whose `version` lines a lockfile records (BasisSimulator and
# BasisMonteCarlo by path): a release bumps them without changing what a notebook renders.
const OWN_PACKAGES = ("[[deps.BasisSimulator]]", "[[deps.BasisMonteCarlo]]")

"A lockfile's hash, ignoring the `version` lines of this repository's own packages."
function lock_input_hash(path::AbstractString)
    kept = String[]
    own = false
    for line in split(read(path, String), '\n')
        if startswith(line, "[[")
            own = strip(line) in OWN_PACKAGES
        elseif own && occursin(r"^\s*version\s*=", line)
            continue
        end
        push!(kept, line)
    end
    return bytes2hex(sha256(join(kept, '\n')))
end

"The committed declarations of the notebooks' optional data."
data_input_hash() = bytes2hex(sha256(read(DATA_PROVENANCE)))

"""Verify any locally available, ignored datasets against the committed
provenance declaration. Missing files are allowed: CI consumes the committed
exports and does not need private/large rendering inputs."""
function verify_local_data!()
    isfile(DATA_PROVENANCE) ||
        error("missing committed data provenance: $(DATA_PROVENANCE)")
    configured_roots = Dict("xcat" => get(
        ENV,
        "BASISSIM_XCAT_DIR",
        joinpath(DATA_DIR, "xcat"),
    ))
    for raw_line in eachline(DATA_PROVENANCE)
        line = strip(raw_line)
        (isempty(line) || startswith(line, '#')) && continue
        parts = split(line; limit = 2)
        length(parts) == 2 || error("malformed data provenance line: $(raw_line)")
        expected, relative = parts
        relative = strip(relative)
        parts = splitpath(relative)
        path = if !isempty(parts) && haskey(configured_roots, first(parts))
            joinpath(configured_roots[first(parts)], parts[2:end]...)
        else
            joinpath(DATA_DIR, relative)
        end
        isfile(path) || continue
        actual = bytes2hex(sha256(read(path)))
        actual == expected || error("dataset checksum mismatch: $(relative)")
    end
    return nothing
end

"""Fingerprint what decides a notebook's export: the notebook, its data declarations, the docs
environment (Snapshot's locked tree, docs/build_env/Manifest.toml, the docs lockfiles without this
repository's own version lines) and `EXPORT_CONTRACT`. Not the package's `src/`, `Project.toml` or
the exporter scripts' own bytes: a code change re-renders the notebooks it affects when asked
(`docs/render_notebooks.sh <slug>` or `BASISSIM_FORCE_NB_REBUILD=1`), not every page on every
commit; a change to how pages render bumps `EXPORT_CONTRACT`."""
function export_fingerprint(slug::AbstractString)
    snapshot_tree = string(Pkg.dependencies()[SNAPSHOT_UUID].tree_hash)
    build_lock = bytes2hex(sha256(read(joinpath(@__DIR__, "build_env", "Manifest.toml"))))
    # every docs lockfile: one per Julia minor version (Manifest-v1.12.toml, Manifest-v1.13.toml)
    docs_locks = sort([
        name
        for name in readdir(@__DIR__)
        if occursin(r"^Manifest(-v[0-9.]+)?\.toml$", name)
    ])
    docs_lock = bytes2hex(sha256(join(
        ["$(name):$(lock_input_hash(joinpath(@__DIR__, name)))" for name in docs_locks],
        '\0',
    )))
    payload = join(
        (
            source_hash(slug),
            data_input_hash(),
            snapshot_tree,
            build_lock,
            docs_lock,
            EXPORT_CONTRACT,
        ),
        '\0',
    )
    return bytes2hex(sha256(payload))
end

"""Fingerprint recorded inside the cached HTML, or `nothing` if the cache is
missing or unreadable."""
function cached_hash(slug::AbstractString)::Union{String, Nothing}
    out = joinpath(STATIC_OUT, "$(slug).html")
    isfile(out) || return nothing
    html = read(out, String)
    m = match(Regex("<meta\\s+name=\"$(HASH_META)\"\\s+content=\"([0-9a-f]+)\""), html)
    return m === nothing ? nothing : m.captures[1]
end

needs_rebuild(slug::AbstractString)::Bool = cached_hash(slug) != export_fingerprint(slug)

"""Render one notebook → LEAN standalone HTML (Snapshot.jl therapy emit).

`export_notebook` opens the notebook in `session`, runs every cell (the
notebooks self-activate docs/Project.toml, so Metal/CairoMakie resolve as
always), SSRs the rendered outputs, attempts a WASM island per `@bind`
group (a group that can't compile falls back to its static output with a
visible note — never a blank), writes `<slug>.html` + `<slug>.islands/`,
and shuts the notebook down.

Returns the SHA-256 hash embedded in the HTML so the caller can diff
against the pre-render hash and report when Pluto's normalization shifted
the source. The hash is recomputed AFTER the run (the session disables
notebook writes, but belt-and-suspenders: whatever lands on disk is what
we key the cache on).
"""
function render_notebook!(
    session::Pluto.ServerSession,
    src_path::AbstractString,
    dst_path::AbstractString,
)
    slug = splitext(basename(src_path))[1]
    html_path = export_notebook(src_path;
        output_dir = dirname(dst_path),
        therapy = true, # lean page: SSR cells + islands, NO statefile
        fragment = true, # native inline component for Therapy docs (no iframe)
        assets_base = "__BASISSIM_NOTEBOOKS_BASE__",
        islands = true, # compile @bind groups to wasm where possible
        verify = true, # oracle-check compiled islands (node)
        optimize = :size,
        # These controls drive native Cairo/GPU simulation volumes whose
        # closures are intentionally outside WasmTarget's current scope.
        # Keep Pluto interactive; publish the rendered result with an honest,
        # disabled control instead of spending unbounded time in inference.
        force_fallback_bonds = get(FORCE_FALLBACK_BONDS, slug, ()),
        env_dir = @__DIR__, # notebook imports resolve from docs/Project.toml
        session = session
    )
    post_hash = source_hash(slug)
    fingerprint = export_fingerprint(slug)
    # Embed the source+exporter fingerprint for the rebuild cache.
    meta = """<meta name="$(HASH_META)" content="$(fingerprint)">"""
    html = read(html_path, String)
    html = occursin("<head>", html) ?
        replace(html, "<head>" => "<head>" * meta; count = 1) :
        html * "\n" * meta * "\n"
    write(html_path, html)
    # export_notebook names the file after the notebook stem == our slug, so
    # html_path and dst_path coincide; assert so a future rename can't desync.
    abspath(html_path) == abspath(dst_path) ||
        error("export wrote $(html_path) but the gallery expects $(dst_path)")
    compress_page_images!(html_path) # embedded figure PNGs → WebP q90 (~90% smaller)
    fragment_path = replace(html_path, r"\.html$" => ".fragment.html")
    isfile(fragment_path) ||
        error("Snapshot did not emit expected fragment: $(fragment_path)")
    fragment = read(fragment_path, String)
    marker = "<!-- $(HASH_META): $(fingerprint) -->"
    write(fragment_path, marker * "\n" * fragment)
    compress_page_images!(fragment_path)
    return post_hash
end

"""Set of slugs the user wants to skip entirely (no render, not in the route
list).  Comma-separated env var.  Example:
    BASISSIM_SKIP_NOTEBOOKS=06_catsim_vs_basissim julia --project=docs docs/app.jl build
"""
function skipped_slugs()
    return Set(
        strip.(split(get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""), ","; keepempty = false)),
    )
end

"""Walk every notebook (or `only`), rendering only those whose export fingerprint doesn't
match the cached fingerprint. Returns the full list of slugs (cached + freshly
rendered) so callers can register routes for all of them."""
function export_notebooks(only = nothing)
    verify_local_data!()
    slugs = notebook_slugs()
    if only !== nothing
        unknown = setdiff(only, slugs)
        isempty(unknown) ||
            error("no notebook named $(join(unknown, ", ")) in $(NOTEBOOKS_DIR)")
        slugs = filter(in(only), slugs)
    end
    skip = skipped_slugs()
    if !isempty(skip)
        kept = filter(s -> !(s in skip), slugs)
        dropped = filter(s -> s in skip, slugs)
        isempty(dropped) ||
            println("[notebooks] BASISSIM_SKIP_NOTEBOOKS → skipping $(join(dropped, ", "))")
        slugs = kept
    end
    isempty(slugs) &&
        (println("[notebooks] no .jl files in $(NOTEBOOKS_DIR)"); return slugs)

    isdir(STATIC_OUT) || mkpath(STATIC_OUT)

    force = haskey(ENV, "BASISSIM_FORCE_NB_REBUILD")
    to_render = force ? slugs : filter(needs_rebuild, slugs)

    if isempty(to_render)
        println(
            "[notebooks] all $(length(slugs)) cached (export fingerprint matches) — no rebuild",
        )
        return slugs
    end

    println(
        "[notebooks] rendering $(length(to_render))/$(length(slugs)) " *
            (force ? "(forced)" : "(source changed)") *
            "…",
    )

    session = render_session()
    foreach(slug -> render_slug!(session, slug), to_render)
    return slugs
end

# disable_writing_notebook_files=true: keep Pluto from re-saving the
# source .jl during open() — that's the root of the "every run re-renders"
# loop (Pluto normalizes the source bytes between hash-pre-render and
# hash-post-render).  See render_notebook! for the second safeguard.
render_session() = Pluto.ServerSession(; options = Pluto.Configuration.from_flat_kwargs(
    disable_writing_notebook_files = true,
))

function render_slug!(session, slug)
    src = joinpath(NOTEBOOKS_DIR, "$(slug).jl")
    dst = joinpath(STATIC_OUT, "$(slug).html")
    pre_hash = source_hash(slug)
    println("  ▸ $(slug)  (hash $(first(pre_hash, 12))…)")
    t0 = time()
    post_hash = render_notebook!(session, src, dst)
    if post_hash != pre_hash
        println(
            "    ⚠ Pluto normalized source: hash $(first(pre_hash, 12))… → $(first(post_hash, 12))… (embedding post-render hash so cache stays consistent)",
        )
    end
    kb = round(filesize(dst) / 1024; digits = 1)
    println("    ✓ $(round(time() - t0; digits = 1))s  →  $(kb) KB")
    return flush(stdout)
end

"""Render `<queue>/<slug>.todo` entries one at a time until none is left. A process claims an
entry by renaming it (atomic on one filesystem), so several processes — one per GPU — share one
queue without rendering a notebook twice; the claimed entry is removed once its export is written."""
function render_queue(queue::AbstractString)
    verify_local_data!()
    isdir(STATIC_OUT) || mkpath(STATIC_OUT)
    session = render_session()
    while true
        slug = claim_next(queue)
        slug === nothing && return
        render_slug!(session, slug)
        rm(joinpath(queue, "$(slug).claimed.$(getpid())"))
    end
end

function claim_next(queue::AbstractString)
    for f in sort(readdir(queue))
        endswith(f, ".todo") || continue
        slug = f[1:(end - 5)]
        try
            mv(joinpath(queue, f), joinpath(queue, "$(slug).claimed.$(getpid())"))
            return slug
        catch
            # another process claimed it first
        end
    end
    return nothing
end

"""Shrink the figures embedded in a rendered page: every `data:image/png` URI
over ~200 KB is transcoded to WebP q90 (visually indistinguishable for the CT
figures here; ~90% smaller — a 2360×2360 retina CairoMakie PNG is ~4.7 MB as
PNG, ~0.3 MB as WebP). Runs python3+Pillow — fine, this script only ever runs
on the local render machine. Idempotent: already-webp URIs are untouched."""
function compress_page_images!(html_path::AbstractString)
    py = raw"""
    import re, base64, io, sys
    from PIL import Image
    p = sys.argv[1]
    h = open(p, encoding='utf-8', errors='surrogateescape').read()
    saved = [0, 0]
    def sub(m):
        raw = base64.b64decode(m.group(1))
        if len(raw) < 200_000: return m.group(0)
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        out = io.BytesIO(); im.save(out, 'WEBP', quality=90, method=6)
        if out.tell() >= len(raw): return m.group(0)
        saved[0] += len(raw); saved[1] += out.tell()
        return 'data:image/webp;base64,' + base64.b64encode(out.getvalue()).decode()
    h2 = re.sub(r'data:image/png;base64,\s*([A-Za-z0-9+/=]{1000,})', sub, h)
    if h2 != h:
        open(p, 'w', encoding='utf-8', errors='surrogateescape').write(h2)
    print(f"png->webp: {saved[0]/1e6:.1f} MB -> {saved[1]/1e6:.1f} MB")
    """
    return run(`python3 -c $py $html_path`)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        export_notebooks()
    elseif ARGS == ["--list-stale"]
        force = haskey(ENV, "BASISSIM_FORCE_NB_REBUILD")
        foreach(println, force ? notebook_slugs() : filter(needs_rebuild, notebook_slugs()))
    elseif length(ARGS) == 2 && ARGS[1] == "--queue"
        render_queue(ARGS[2])
    else
        export_notebooks(ARGS)
    end
end
