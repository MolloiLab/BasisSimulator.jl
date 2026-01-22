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
    simulate(phantom, scanner, protocol, sim_opts, recon_opts; energies, weights, physics_config)

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
- `physics_config`: Optional `PhysicsConfig` for full control over all physics effects.
  When provided, bypasses SimOptions-driven physics construction and passes this config
  directly to `forward_project()`. Use `full_physics_config()`, `realistic_physics_config()`,
  or build a custom `PhysicsConfig` to control all 13 effects.

# Returns
`SimulationResult` containing sinograms and reconstruction.

# Example
```julia
# Simple preset-based usage:
result = simulate(phantom, scanner, protocol, SimOptions(fidelity=:high), recon_opts)

# Full physics control:
config = full_physics_config(energy_keV=65.0, noise_seed=42)
result = simulate(phantom, scanner, protocol, SimOptions(), recon_opts; physics_config=config)
```
"""
function simulate(
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions();
    energies::Union{Vector{Float64}, Nothing} = nothing,
    weights::Union{Vector{Float64}, Nothing} = nothing,
    physics_config::Union{PhysicsConfig, Nothing} = nothing
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
    elseif needs_polychromatic(sim_opts)
        # CASE B: Internal Polychromatic Lookup (auto-detected from enabled effects)
        e_full, w_full = load_spectrum(Int(protocol.kVp))
        current_energies, current_weights = downsample_spectrum(e_full, w_full, sim_opts.n_energy_bins)
    else
        # CASE C: Monochromatic Fallback
        current_energies = [Float64(protocol.kVp) * 0.5]
        current_weights = [1.0]
    end

    # Physics config: use explicit config if provided, otherwise build from SimOptions
    config = if !isnothing(physics_config)
        # User provided full PhysicsConfig - use directly
        physics_config
    else
        # Build from SimOptions presets
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
        default_physics_config(; physics_kwargs...)
    end
    
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

"""
    needs_polychromatic(sim_opts::SimOptions) -> Bool

Determine if polychromatic spectrum is needed based on enabled effects.
Returns true if any energy-dependent effect is ON: flat_filter, bowtie_filter,
detector_efficiency, or bhc.
"""
function needs_polychromatic(sim_opts::SimOptions)::Bool
    return sim_opts.use_flat_filter ||
           sim_opts.use_bowtie_filter ||
           sim_opts.use_detector_efficiency ||
           sim_opts.use_bhc
end

"""
    build_physics_config(scanner::Scanner, sim_opts::SimOptions, energies::Vector{Float64}, weights::Vector{Float64}) -> PhysicsConfig

Build a complete PhysicsConfig from Scanner hardware fields and SimOptions toggles.

For effects with Scanner fields (focal_spot, flat_filter, fill_factor, detector_efficiency,
heel_effect), the Scanner hardware parameters are used to construct the effect structs.
For effects without Scanner fields (scatter, scatter_correction, crosstalk, optical_crosstalk,
bowtie, lag, bhc), factory function defaults are used.

Noise is ALWAYS `nothing` in the returned PhysicsConfig — noise is applied externally
via `sim_detect()` when `sim_opts.use_noise == true`.

# Arguments
- `scanner`: Scanner hardware definition (provides physical parameters)
- `sim_opts`: SimOptions with resolved boolean toggles
- `energies`: Energy bin centers (keV) for the current spectrum
- `weights`: Photon weights for each energy bin

# Returns
A `PhysicsConfig` ready for `forward_project()`.
"""
function build_physics_config(
    scanner::Scanner,
    sim_opts::SimOptions,
    energies::Vector{Float64},
    weights::Vector{Float64}
)
    kwargs = Dict{Symbol, Any}()

    # --- Common settings ---
    kwargs[:energy_keV] = sum(energies .* weights) / sum(weights)
    kwargs[:noise_seed] = sim_opts.seed
    kwargs[:noise] = nothing  # Noise is ALWAYS handled by sim_detect, never in PhysicsConfig

    # --- Physics Pipeline effects (from Scanner fields where available) ---

    # Fill factor: use Scanner's row/col fill factors
    if sim_opts.use_fill_factor
        row_fill = scanner.fill_factor_row
        col_fill = scanner.fill_factor_col
        if row_fill > 0 && col_fill > 0
            kwargs[:fill_factor] = FillFactorModel(row_fill, col_fill, row_fill ≈ col_fill)
        else
            kwargs[:fill_factor] = fill_factor_standard()
        end
    end

    # Flat filter: use Scanner's material and thickness
    if sim_opts.use_flat_filter
        thickness = scanner.flat_filter_thickness
        material = scanner.flat_filter_material
        if thickness > 0
            kwargs[:flat_filter] = FlatFilter([String(material)], [thickness], "scanner_$(material)_$(thickness)mm")
        else
            kwargs[:flat_filter] = flat_filter_al()
        end
    end

    # Bowtie filter: no Scanner field, use factory default
    if sim_opts.use_bowtie_filter
        kwargs[:bowtie_filter] = bowtie_filter_large_body()
    end

    # Detector efficiency: use Scanner's material and depth
    if sim_opts.use_detector_efficiency
        depth = scanner.detector_depth
        material = scanner.detector_material
        if depth > 0
            kwargs[:detector_efficiency] = DetectorEfficiency(String(material), depth, 1.0)
        else
            kwargs[:detector_efficiency] = detector_efficiency_gos()
        end
    end

    # Scatter: no Scanner field, use factory default
    if sim_opts.use_scatter
        kwargs[:scatter] = default_scatter_model()
    end

    # Scatter correction: no Scanner field, use factory default
    if sim_opts.use_scatter_correction
        kwargs[:scatter_correction] = default_scatter_correction()
    end

    # Crosstalk (electronic): no Scanner field, use factory default
    if sim_opts.use_crosstalk
        kwargs[:crosstalk] = crosstalk_medium()
    end

    # Optical crosstalk: no Scanner field, use factory default
    if sim_opts.use_optical_crosstalk
        kwargs[:optical_crosstalk] = optical_crosstalk_typical()
    end

    # Focal spot: use Scanner's width and length
    if sim_opts.use_focal_spot
        width = scanner.focal_spot_width
        length = scanner.focal_spot_length
        if width > 0 && length > 0
            kwargs[:focal_spot] = FocalSpot(width, length, :gaussian, 5)
        else
            kwargs[:focal_spot] = focal_spot_medium()
        end
    end

    # Lag: no Scanner field, use factory default
    if sim_opts.use_lag
        kwargs[:lag] = lag_gadox()
    end

    # --- Signal Chain effects ---

    # Heel effect: use Scanner's target angle
    if sim_opts.use_heel_effect
        angle = scanner.target_angle
        if angle > 0
            kwargs[:heel_effect] = HeelEffect(angle, :tungsten, 0.01, true)
        else
            kwargs[:heel_effect] = default_heel_effect()
        end
    end

    # DAS: BROKEN — never enable even if toggle is true (safety guard)
    # sim_opts.use_das defaults to false at all fidelity levels
    if sim_opts.use_das
        @warn "DAS model is BROKEN. Ignoring use_das=true."
    end

    # BHC: no Scanner field, use factory default with reference energy from spectrum
    if sim_opts.use_bhc
        ref_energy = sum(energies .* weights) / sum(weights)
        kwargs[:bhc] = bhc_water_default(; reference_energy_keV=ref_energy)
    end

    return default_physics_config(; kwargs...)
end

export build_physics_config
