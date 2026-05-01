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
ENV["BASISSIM_BASE"] = IS_BUILD ? "/BasisSimulator.jl" : ""

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

# =============================================================================
# Run - dev or build based on args
# =============================================================================

Therapy.run(app)
