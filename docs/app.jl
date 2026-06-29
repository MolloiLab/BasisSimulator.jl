#!/usr/bin/env julia
# BasisSimulator.jl Documentation Site
#
# Usage (from BasisSimulator.jl root directory):
#   julia --project=docs docs/app.jl dev    # Development server with HMR
#   julia --project=docs docs/app.jl build  # Build static site to docs/dist
#
# Built with Therapy.jl (https://github.com/GroupTherapyOrg/Therapy.jl):
# - File-based routing from src/routes/
# - Automatic component loading from src/components/
# - Pluto notebooks under docs/notebooks/ are rendered to standalone HTML
#   via PlutoStaticHTML and iframe-embedded into the /examples/<slug>/ pages.

if !haskey(ENV, "JULIA_PROJECT")
    using Pkg
    Pkg.activate(@__DIR__)
end

using Therapy

cd(@__DIR__)

# =============================================================================
# Base path — empty in dev mode (so http://localhost:8080/getting-started/ works),
# "/BasisSimulator.jl" in build mode (so GH Pages at
# https://molloilab.github.io/BasisSimulator.jl/getting-started/ works). Layout
# and routes read BASISSIM_BASE from ENV to construct hrefs accordingly.
# =============================================================================

const IS_BUILD = length(ARGS) > 0 && ARGS[1] == "build"
# Snapshot hosts this app under /app/<owner>/<repo>/ and exports SNAPSHOT_BASE_PATH to
# that path — honor it so every href + asset resolves there. Otherwise fall back to
# "/BasisSimulator.jl" for GH Pages (molloilab.github.io/BasisSimulator.jl) in build
# mode, or "" for the local dev server.
ENV["BASISSIM_BASE"] = get(ENV, "SNAPSHOT_BASE_PATH", IS_BUILD ? "/BasisSimulator.jl" : "")

# =============================================================================
# Pluto notebook export — runs PlutoStaticHTML on docs/notebooks/*.jl, writes
# self-contained HTML to docs/dist/notebooks-static/<slug>.html.  mtime-cached:
# unchanged notebooks are skipped, so iterating on Layout/CSS doesn't trigger
# a re-render.  The export runs as a SUBPROCESS against the sidecar
# `docs/build_env/Project.toml` (PlutoStaticHTML conflicts with CairoMakie in
# the main docs env via tectonic_jll/HarfBuzz_jll), then we discover slugs
# from disk to register per-notebook routes below.
# =============================================================================

let build_env = joinpath(@__DIR__, "build_env"),
    extract   = joinpath(@__DIR__, "extract_all.jl")
    if !haskey(ENV, "BASISSIM_SKIP_NB_EXPORT")
        Base.run(`$(Base.julia_cmd()) --project=$(build_env) $(extract)`)
    else
        println("[notebooks] BASISSIM_SKIP_NB_EXPORT set — skipping export")
    end
end

const NOTEBOOK_SLUGS = let dir = joinpath(@__DIR__, "notebooks"),
    skip = Set(strip.(split(get(ENV, "BASISSIM_SKIP_NOTEBOOKS", ""), ","; keepempty = false)))

    all_slugs = isdir(dir) ?
        sort([
            splitext(f)[1] for f in readdir(dir)
            if endswith(f, ".jl") && !endswith(f, ".sessions.toml")
        ]) :
        String[]
    kept = filter(s -> !(s in skip), all_slugs)
    if !isempty(skip)
        dropped = filter(s -> s in skip, all_slugs)
        isempty(dropped) || println("[app] BASISSIM_SKIP_NOTEBOOKS → skipping $(join(dropped, ", "))")
    end
    kept
end

# =============================================================================
# App Configuration
# =============================================================================

app = App(
    routes_dir = "src/routes",
    components_dir = "src/components",
    title = "BasisSimulator.jl",
    output_dir = "dist",
    base_path = ENV["BASISSIM_BASE"],
    layout = :Layout
)

# Load file-based routes + components first so per-slug route handlers
# below can reference NotebookPage from src/components/.
Therapy.load_app!(app)

# =============================================================================
# Static asset mounts — Pluto HTML + image assets
# =============================================================================
# Therapy's `staticfiles` mounts a folder under a URL path: dev server
# serves it on-the-fly, build mode copies it into dist/.  Both folders
# live OUTSIDE dist/ because Therapy wipes its output_dir before build.

let nb_static = joinpath(@__DIR__, "notebooks-static"),
    assets    = joinpath(@__DIR__, "assets")
    if isdir(nb_static)
        Therapy.staticfiles(app, nb_static, "notebooks-static")
        println("  Mounted: /notebooks-static/  → $(nb_static)")
    end
    if isdir(assets)
        Therapy.staticfiles(app, assets, "assets")
        println("  Mounted: /assets/  → $(assets)")
    end
end

# =============================================================================
# Per-notebook routes — /examples/<slug>/ → iframe of the Pluto static HTML
# =============================================================================
# Each slug discovered above gets its own Therapy route that renders
# `NotebookPage(slug)`.  Following Sessions.jl's loader-order dance: the
# component reaches its final world AFTER `Therapy.load_app!` returns,
# so we resolve via `invokelatest` to satisfy Julia 1.12's strict-binding
# warning.  Drop a new .jl in docs/notebooks/, restart, and it shows up
# automatically — no other code changes required.

let host = isdefined(Main, :TherapyApp) ? getfield(Main, :TherapyApp) : Main
    NotebookPage = isdefined(host, :NotebookPage) ?
        getfield(host, :NotebookPage) : nothing

    if NotebookPage === nothing
        @warn "[docs] NotebookPage component not found — skipping per-notebook route registration"
    else
        for slug in NOTEBOOK_SLUGS
            route = "/examples/$(slug)/"
            push!(app.routes, route => let s = slug, np = NotebookPage
                () -> Base.invokelatest(np, s)
            end)
            println("  Registered notebook route: $(route)")
        end
    end
end

# =============================================================================
# Favicon — mounted at root URL so browsers auto-discover /favicon.ico in dev,
# and copied to dist/favicon.ico in build.  The post-build pass below also
# injects an explicit <link rel="icon"> into every emitted HTML file so the
# favicon resolves under GH Pages' /BasisSimulator.jl/ subpath (where the
# browser's automatic /favicon.ico probe hits the wrong host root).
# =============================================================================

let favicon = joinpath(@__DIR__, "assets", "favicon.ico")
    if isfile(favicon)
        hdrs = Pair{String,String}["Cache-Control" => "public, max-age=86400"]
        push!(app.static_mounts, Therapy.StaticMount(
            "/favicon.ico", favicon, hdrs, nothing, true,
            Therapy.file(favicon; headers=hdrs)
        ))
        println("  Mounted: /favicon.ico  → $(favicon)")
    end
end

# =============================================================================
# Run - dev or build based on args
# =============================================================================

Therapy.run(app)

# =============================================================================
# Post-build: inject favicon <link> into every dist/**/*.html so GH Pages
# (served under /BasisSimulator.jl/) actually finds the icon.  No-op in dev.
# =============================================================================

if IS_BUILD && isdir(app.output_dir)
    favicon_link = "<link rel=\"icon\" type=\"image/x-icon\" href=\"$(ENV["BASISSIM_BASE"])/favicon.ico\">"
    for (root, _, files) in walkdir(app.output_dir)
        for f in files
            endswith(f, ".html") || continue
            path = joinpath(root, f)
            content = read(path, String)
            (occursin(favicon_link, content) || !occursin("</head>", content)) && continue
            write(path, replace(content, "</head>" => "    $(favicon_link)\n</head>"; count=1))
        end
    end
end
