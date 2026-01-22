"""
    Simulation/Driver.jl

High-level driver for running end-to-end CT simulations.
"""

export simulate, SimulationResult

"""
    SimulationResult

Container for simulation outputs.
"""
struct SimulationResult{T, G, P}
    sinogram_ideal::AbstractArray{T, 3}
    sinogram_noisy::AbstractArray{T, 3}
    reconstruction::AbstractArray{T, 3}
    geometry::G
    physics_config::P
end

"""
    simulate(phantom, scanner, protocol, sim_opts, recon_opts; energies, weights)

Run a full end-to-end CT simulation.

# Arguments
- `phantom`: Struct containing `.mask` (UInt8) and material definitions.
- `scanner`: `Scanner` hardware definition.
- `protocol`: `CTProtocol` acquisition settings.
- `sim_opts`: `SimOptions` for physics fidelity.
- `recon_opts`: `ReconOptions` for image formation.

# Keyword Arguments
- `energies`: Optional Vector{Float64} of energy bin centers (keV). Overrides internal lookup.
- `weights`: Optional Vector{Float64} of photon weights. Overrides internal lookup.

# Returns
`SimulationResult` containing sinograms and reconstruction.
"""
function simulate(
    phantom, 
    scanner::Scanner, 
    protocol::CTProtocol, 
    sim_opts::SimOptions = SimOptions(), 
    recon_opts::ReconOptions = ReconOptions();
    energies::Union{Vector{Float64}, Nothing} = nothing,
    weights::Union{Vector{Float64}, Nothing} = nothing
)
    # 1. Build Geometry
    geom = CTGeometry(
        scanner; 
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing # Auto-calc from detector
    )

    # 2. Build Physics Config (The Bridge)
    
    # Initialize these to ensure they are defined for the physics config later
    current_energies = nothing
    current_weights = nothing
    materials = get_region_materials()

    # LOGIC UPDATE: Prioritize injected kwargs, then fall back to sim_opts settings
    if !isnothing(energies) && !isnothing(weights)
        # CASE A: User explicitly provided spectra (e.g. from the notebook loop)
        current_energies = energies
        current_weights = weights
    elseif sim_opts.use_beam_hardening
        # CASE B: Internal Polychromatic Lookup
        e_full, w_full = load_spectrum(Int(protocol.kVp))
        # Downsample for speed (30 bins is standard accuracy/speed tradeoff)
        current_energies, current_weights = downsample_spectrum(e_full, w_full, 30)
    else
        # CASE C: Monochromatic Fallback
        # Default to ~half kVp for effective energy
        current_energies = [Float64(protocol.kVp) * 0.5] 
        current_weights = [1.0]
    end
    
    # Physics pipeline construction
    physics_kwargs = Dict{Symbol, Any}()
    
    # Common settings
    physics_kwargs[:energy_keV] = sum(current_energies .* current_weights) / sum(current_weights)
    physics_kwargs[:noise_seed] = sim_opts.seed
    
    # Explicitly DISABLE internal noise in PhysicsConfig (handled via sim_detect)
    physics_kwargs[:noise] = nothing 
    
    # Scatter
    if sim_opts.use_scatter
        physics_kwargs[:scatter] = default_scatter_model()
        physics_kwargs[:scatter_correction] = default_scatter_correction()
    end
    
    # Focal Spot
    if sim_opts.use_focal_spot
        physics_kwargs[:focal_spot] = focal_spot_medium()
    end
    
    # Crosstalk
    if sim_opts.use_crosstalk
        physics_kwargs[:crosstalk] = crosstalk_medium()
    end
    
    # Assemble Config
    config = default_physics_config(; physics_kwargs...)
    
    # 3. Forward Project (Ideal/Physics-only, No Noise)
    sino_ideal = forward_project(
        phantom.mask, geom;
        energies=current_energies,
        weights=current_weights,
        materials=materials,
        physics=config
    )
    
    # 4. Apply Detector Noise (Protocol-driven)
    sino_final = if sim_opts.use_noise
        sim_detect(sino_ideal, geom, protocol)
    else
        copy(sino_ideal)
    end
    
    # 5. Reconstruction
    recon_vol = if recon_opts.algorithm == :fdk
        fdk_reconstruct(sino_final, geom, recon_opts.matrix_size)
    elseif recon_opts.algorithm == :sirt
        sirt_reconstruct(sino_final, geom, recon_opts.matrix_size; niter=recon_opts.iterations)
    elseif recon_opts.algorithm == :cgls
        cgls_reconstruct(sino_final, geom, recon_opts.matrix_size; niter=recon_opts.iterations)
    else
        error("Unknown reconstruction algorithm: $(recon_opts.algorithm)")
    end
    
    return SimulationResult(
        sino_ideal,
        sino_final,
        recon_vol,
        geom,
        config
    )
end

# -- HELPERS -- #
"""
    get_spectrum(protocol::CTProtocol)

Helper to extract the exact energy/weight vectors the simulator will use 
for a given protocol. Useful for verification and plotting.
"""
function get_spectrum(protocol::CTProtocol)
    # 1. Priority: Explicit Path
    if !isnothing(protocol.spectrum_path) && isfile(protocol.spectrum_path)
        data = readdlm(protocol.spectrum_path)
        return data[:,1], data[:,2] # energies, weights
    end

    # 2. Priority: Auto-Lookup via kVp
    # This confirms the simulator can find the library file
    return load_spectrum(Int(protocol.kVp))
end

export get_spectrum
