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
# - Interactive components (DarkModeToggle) compiled to WebAssembly via WasmTarget.jl

if !haskey(ENV, "JULIA_PROJECT")
    using Pkg
    Pkg.activate(@__DIR__)
end

using Therapy

cd(@__DIR__)

# =============================================================================
# App Configuration
# =============================================================================

app = App(
    routes_dir = "src/routes",
    components_dir = "src/components",
    title = "BasisSimulator.jl",
    output_dir = "dist",
    base_path = "/BasisSimulator.jl",
    layout = :Layout
)

# =============================================================================
# Run - dev or build based on args
# =============================================================================

Therapy.run(app)
