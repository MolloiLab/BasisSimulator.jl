"""
    Simulation/Driver.jl

High-level driver for running end-to-end CT simulations.
"""

export simulate, SimulationResult

"""
    SimulationResult

Container for simulation outputs. Supports single and multi-reconstruction,
dual-energy sinograms, material maps, and VMI volumes.

# Core Fields
- `sinogram_ideal`: Noise-free sinogram (or high-kVp sinogram for DE)
- `sinogram_noisy`: Noisy sinogram after detector simulation
- `reconstructions`: Vector of (algorithm_name, volume) pairs
- `geometry`: CTGeometry or HelicalGeometry used for simulation
- `physics_config`: PhysicsConfig with all enabled effects

# Dual-Energy Fields
- `de_sinogram`: DualEnergySinogram (low/high kVp pair), nothing if single-kVp
- `material_maps`: MaterialMap from decomposition, nothing if single-kVp
- `vmi_volumes`: Dict{Float64, Array} of VMI reconstructions by energy

# Backward Compatibility
- `result.reconstruction` returns the first reconstruction volume
- Single-recon calls populate `reconstructions` with one entry
"""
struct SimulationResult{T, G, P}
    sinogram_ideal::AbstractArray{T, 3}
    sinogram_noisy::AbstractArray{T, 3}
    reconstructions::Vector{Pair{Symbol, AbstractArray{T, 3}}}
    geometry::G
    physics_config::P
    # Dual-energy fields
    de_sinogram::Union{Nothing, DualEnergySinogram}
    material_maps::Union{Nothing, MaterialMap}
    vmi_volumes::Dict{Float64, AbstractArray{T, 3}}
end

# Backward-compatible property accessor: result.reconstruction → first volume
function Base.getproperty(r::SimulationResult, s::Symbol)
    if s === :reconstruction
        recons = getfield(r, :reconstructions)
        isempty(recons) && error("No reconstructions available")
        return recons[1].second
    else
        return getfield(r, s)
    end
end

function Base.propertynames(::SimulationResult, private::Bool=false)
    return (:sinogram_ideal, :sinogram_noisy, :reconstruction, :reconstructions,
            :geometry, :physics_config, :de_sinogram, :material_maps, :vmi_volumes)
end

"""
    simulate(phantom, scanner, protocol, sim_opts, recon_opts)

Run a full end-to-end CT simulation.

The 4-struct API: Scanner provides hardware parameters, CTProtocol provides acquisition
settings, SimOptions controls which physics effects are enabled, and ReconOptions
controls image reconstruction. No additional kwargs needed.

# Arguments
- `phantom`: Struct containing `.mask` (UInt8) and material definitions.
- `scanner`: `Scanner` hardware definition.
- `protocol`: `CTProtocol` acquisition settings.
- `sim_opts`: `SimOptions` for physics fidelity (controls all 14 effects).
- `recon_opts`: `ReconOptions` for image formation.

# Returns
`SimulationResult` containing sinograms and reconstruction.

# Example
```julia
result = simulate(phantom, scanner, protocol, SimOptions(fidelity=:high), ReconOptions())
result = simulate(phantom, scanner, protocol, SimOptions(fidelity=:medium, use_scatter=false), ReconOptions())
```
"""
function simulate(
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions()
)
    # 1. Build Geometry
    geom = CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing  # Auto-calc from detector
    )

    # 2. Resolve spectrum (polychromatic vs monochromatic)
    energies, weights = resolve_spectrum(sim_opts, protocol)

    # 3. Build PhysicsConfig from Scanner + SimOptions
    config = build_physics_config(scanner, sim_opts, energies, weights)

    # 4. Forward Project (physics-only, no noise)
    materials = get_region_materials()
    sino_ideal = forward_project(
        phantom.mask, geom;
        energies=energies,
        weights=weights,
        materials=materials,
        physics=config
    )

    # 5. Apply Detector Noise (protocol-driven I0)
    sino_final = if sim_opts.use_noise
        sim_detect(sino_ideal, geom, protocol)
    else
        copy(sino_ideal)
    end

    # 6. Reconstruction
    recon_vol = if recon_opts.algorithm == :fdk
        fdk_reconstruct(sino_final, geom, recon_opts.matrix_size)
    elseif recon_opts.algorithm == :sirt
        sirt_reconstruct(sino_final, geom, recon_opts.matrix_size; niter=recon_opts.iterations)
    elseif recon_opts.algorithm == :cgls
        cgls_reconstruct(sino_final, geom, recon_opts.matrix_size; niter=recon_opts.iterations)
    else
        error("Unknown reconstruction algorithm: $(recon_opts.algorithm)")
    end

    T = eltype(recon_vol)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()

    return SimulationResult(
        sino_ideal,
        sino_final,
        recons,
        geom,
        config,
        nothing,              # de_sinogram
        nothing,              # material_maps
        vmi_dict              # vmi_volumes
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
    resolve_spectrum(sim_opts::SimOptions, protocol::CTProtocol) -> (energies, weights)

Determine the energy spectrum based on SimOptions effect toggles.
If any energy-dependent effect is enabled (flat_filter, bowtie_filter, detector_efficiency,
bhc), loads the full polychromatic spectrum and downsamples to `sim_opts.n_energy_bins`.
Otherwise, uses monochromatic approximation at `kVp * 0.5` keV.
"""
function resolve_spectrum(sim_opts::SimOptions, protocol::CTProtocol)
    if needs_polychromatic(sim_opts)
        e_full, w_full = load_spectrum(Int(protocol.kVp))
        return downsample_spectrum(e_full, w_full, sim_opts.n_energy_bins)
    else
        return [Float64(protocol.kVp) * 0.5], [1.0]
    end
end

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
