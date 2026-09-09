# -----------------------------------------------------------------------------
# One entry point behind the five structs: dispatch on the scanner family
# -----------------------------------------------------------------------------

"""
    pipeline(phantom, scanner, protocol, sim_opts, recon_opts; kwargs...)

The functional pipeline for the scanner family: [`eict_pipeline`](@ref) for an
[`EICTScanner`](@ref), [`pcct_pipeline`](@ref) for a [`PCCTScanner`](@ref)
(keyword arguments are forwarded).  Run it with [`forward`](@ref).
"""
pipeline(phantom::BS.Phantom, scanner::BS.EICTScanner, protocol, sim_opts, recon_opts; kwargs...) =
    eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; kwargs...)
pipeline(phantom::BS.Phantom, scanner::BS.PCCTScanner, protocol, sim_opts, recon_opts; kwargs...) =
    pcct_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; kwargs...)

"""
    forward(fractions, pipe, noise...) 

The pure forward model of the pipeline: HU image for an [`EICTPipeline`](@ref)
([`eict_forward`](@ref)), per-channel μ volumes for a [`PCCTPipeline`](@ref)
([`pcct_forward`](@ref)).  Compile with `Reactant.@compile`, differentiate with
`Enzyme.gradient(Reverse, …)`.
"""
forward(input::PipelineInput, pipe::EICTPipeline, ε = nothing, ε_e = nothing) = eict_forward(input, pipe, ε, ε_e)
forward(input::PipelineInput, pipe::PCCTPipeline, N_input = nothing) = pcct_forward(input, pipe, N_input)
