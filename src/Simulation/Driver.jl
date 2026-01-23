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

# PCCT Fields
- `pcct_sinogram`: EnergyResolvedSinogram from PCCT, nothing if not PCCT
- `pcct_material_maps`: PCCTMaterialMap from N-material decomposition, nothing if not PCCT
- `pcct_vmi_volumes`: Dict{Float64, Array} of VMI from material synthesis

# Property Aliases
- `result.reconstruction` returns the first reconstruction volume
- `result.bin_sinograms` is an alias for `pcct_sinogram` (PRD naming convention)
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
    # PCCT fields
    pcct_sinogram::Union{Nothing, EnergyResolvedSinogram}
    pcct_material_maps::Union{Nothing, PCCTMaterialMap}
    pcct_vmi_volumes::Dict{Float64, AbstractArray{T, 3}}
end

# Property accessors for backward compatibility and PRD naming conventions
function Base.getproperty(r::SimulationResult, s::Symbol)
    if s === :reconstruction
        recons = getfield(r, :reconstructions)
        isempty(recons) && error("No reconstructions available")
        return recons[1].second
    elseif s === :bin_sinograms
        return getfield(r, :pcct_sinogram)
    else
        return getfield(r, s)
    end
end

function Base.propertynames(::SimulationResult, private::Bool=false)
    return (:sinogram_ideal, :sinogram_noisy, :reconstruction, :reconstructions,
            :geometry, :physics_config, :de_sinogram, :material_maps, :vmi_volumes,
            :pcct_sinogram, :pcct_material_maps, :pcct_vmi_volumes, :bin_sinograms)
end

"""
    simulate(phantom, scanner, protocol, sim_opts, recon_opts)

Run a full end-to-end CT simulation with automatic mode routing.

The 4-struct API: Scanner provides hardware parameters, CTProtocol provides acquisition
settings, SimOptions controls which physics effects are enabled, and ReconOptions
controls image reconstruction. The driver automatically routes between 5 scan modes:

1. **Axial single-kVp** (default): Standard CT acquisition
2. **Axial dual-kVp**: Dual-energy with VMI pipeline
3. **Helical single-kVp**: Spiral CT with helical reconstruction
4. **Helical dual-kVp**: Combined helical + dual-energy
5. **Axial PCCT**: Photon-counting CT with energy-resolved sinograms, N-material
   decomposition, and material-based VMI synthesis (auto-detected via Scanner)

# Arguments
- `phantom`: Struct containing `.mask` (UInt8) and material definitions.
- `scanner`: `Scanner` hardware definition.
- `protocol`: `CTProtocol` acquisition settings (scan_mode, dual_energy select mode).
- `sim_opts`: `SimOptions` for physics fidelity (controls all 14 effects).
- `recon_opts`: `ReconOptions` or `Vector{ReconOptions}` for multi-recon.

# Returns
`SimulationResult` containing sinograms, reconstructions, and optional DE outputs.

# Examples
```julia
# Axial single-kVp (unchanged from before)
result = simulate(phantom, scanner, CTProtocol(kVp=120, mA=200), SimOptions(), ReconOptions())

# Helical single-kVp
result = simulate(phantom, scanner,
    CTProtocol(scan_mode=:helical, kVp=120, mA=200, pitch=0.984, n_rotations=5.0),
    SimOptions(), ReconOptions(algorithm=:helical_fdk))

# Dual-energy axial with VMI
result = simulate(phantom, scanner,
    CTProtocol(dual_energy=true, kVp=140, mA=200, kVp_low=80, mA_low=350),
    SimOptions(), ReconOptions(vmi_energies=[50.0, 70.0, 100.0]))

# Multi-recon from one scan
recon_list = [ReconOptions(algorithm=:fdk), ReconOptions(algorithm=:sirt, iterations=50)]
result = simulate(phantom, scanner, protocol, sim_opts, recon_list)
```
"""
function simulate(
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions()
)
    # Route based on scan_mode, dual_energy, and PCCT
    is_helical = protocol.scan_mode == :helical
    is_dual = protocol.dual_energy
    _is_pcct = is_pcct(scanner)

    if _is_pcct && is_dual
        error("PCCT scanners cannot use dual_energy mode — spectral info comes from detector energy bins, not dual kVp. Set dual_energy=false.")
    end

    if _is_pcct && is_helical
        error("Helical PCCT is not yet implemented. Use axial scan_mode with PCCT scanner.")
    end

    if _is_pcct
        # PCCT mode: photon-counting scanner detected
        return _simulate_axial_pcct(phantom, scanner, protocol, sim_opts, recon_opts)
    elseif !is_helical && !is_dual
        return _simulate_axial_single(phantom, scanner, protocol, sim_opts, recon_opts)
    elseif !is_helical && is_dual
        return _simulate_axial_dual(phantom, scanner, protocol, sim_opts, recon_opts)
    elseif is_helical && !is_dual
        return _simulate_helical_single(phantom, scanner, protocol, sim_opts, recon_opts)
    else  # is_helical && is_dual
        return _simulate_helical_dual(phantom, scanner, protocol, sim_opts, recon_opts)
    end
end

# Multi-recon dispatch: simulate() with Vector{ReconOptions}
function simulate(
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions,
    recon_opts_list::Vector{ReconOptions}
)
    # Run simulation with first recon option to get sinograms
    first_result = simulate(phantom, scanner, protocol, sim_opts, recon_opts_list[1])

    # Reconstruct with additional options from the same sinogram
    T = eltype(first_result.sinogram_noisy)
    recons = copy(first_result.reconstructions)
    geom = first_result.geometry
    is_helical = geom isa HelicalGeometry

    for i in 2:length(recon_opts_list)
        opts = recon_opts_list[i]

        # Cascading warm start: use previous reconstruction as init (opt-in)
        # Priority: explicit warm_start > cascade_warm_start > algorithm default
        effective_opts = if opts.cascade_warm_start && isnothing(opts.warm_start) && !isempty(recons)
            # Create new ReconOptions with previous result as warm_start
            prev_vol = recons[end].second
            ReconOptions(
                opts.algorithm, opts.matrix_size, opts.fov_cm, opts.filter, opts.iterations,
                opts.lambda, opts.tv_weight, opts.n_subsets,
                opts.penalty, opts.penalty_delta, opts.use_edge_weights, opts.blend_percent,
                opts.interpolation, opts.vmi_energies, opts.vmi_basis,
                prev_vol, opts.cascade_warm_start
            )
        else
            opts
        end

        vol = if is_helical
            _run_helical_reconstruction(first_result.sinogram_noisy, geom, effective_opts)
        else
            _run_reconstruction(first_result.sinogram_noisy, geom, effective_opts)
        end
        push!(recons, opts.algorithm => vol)
    end

    return SimulationResult(
        first_result.sinogram_ideal,
        first_result.sinogram_noisy,
        recons,
        geom,
        first_result.physics_config,
        first_result.de_sinogram,
        first_result.material_maps,
        first_result.vmi_volumes,
        first_result.pcct_sinogram,
        first_result.pcct_material_maps,
        first_result.pcct_vmi_volumes
    )
end

# =============================================================================
# Mode 1: Axial Single-kVp (original behavior preserved)
# =============================================================================

function _simulate_axial_single(phantom, scanner, protocol, sim_opts, recon_opts)
    # 1. Build Geometry
    geom = CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing
    )

    # 2. Resolve spectrum
    energies, weights = resolve_spectrum(sim_opts, protocol)

    # 3. Build PhysicsConfig
    config = build_physics_config(scanner, sim_opts, energies, weights)

    # 4. Forward Project
    materials = get_region_materials()
    sino_ideal = forward_project(
        phantom.mask, geom;
        energies=energies, weights=weights,
        materials=materials, physics=config
    )

    # 5. Apply Detector Noise
    sino_final = if sim_opts.use_noise
        sim_detect(sino_ideal, geom, protocol)
    else
        copy(sino_ideal)
    end

    # 6. Reconstruction
    recon_vol = _run_reconstruction(sino_final, geom, recon_opts)

    T = eltype(recon_vol)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()

    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    return SimulationResult(
        sino_ideal, sino_final, recons, geom, config,
        nothing, nothing, vmi_dict,
        nothing, nothing, pcct_vmi_dict
    )
end

# =============================================================================
# Mode 2: Axial Dual-kVp
# =============================================================================

function _simulate_axial_dual(phantom, scanner, protocol, sim_opts, recon_opts)
    # 1. Build Geometry
    geom = CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing
    )

    # 2. Build PhysicsConfig (using high-kVp spectrum)
    energies_high, weights_high = resolve_spectrum(sim_opts, protocol)
    config = build_physics_config(scanner, sim_opts, energies_high, weights_high)

    # 3. Build GSIProtocol from CTProtocol fields
    gsi = _build_gsi_protocol(protocol)

    # 4. Dual-energy forward projection
    materials = get_region_materials()
    de_sino = forward_project_dual_energy(
        phantom.mask, geom, gsi;
        materials=materials,
        physics=config,
        scanner=nothing  # Use GSI protocol's I0 directly
    )

    # 5. Use high-kVp sinogram as primary (for noise and standard recon)
    sino_ideal = de_sino.high
    sino_final = if sim_opts.use_noise
        sim_detect(sino_ideal, geom, protocol)
    else
        copy(sino_ideal)
    end

    # 6. Material decomposition
    mat_map = decompose_materials(de_sino; basis=Tuple(recon_opts.vmi_basis[1:2]))

    # 7. VMI reconstruction (if energies specified)
    T = eltype(sino_final)
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    if !isempty(recon_opts.vmi_energies)
        for E in recon_opts.vmi_energies
            vmi_vol = reconstruct_vmi(mat_map, E, geom, recon_opts.matrix_size;
                                      method=recon_opts.algorithm == :sirt ? :sirt : :fdk,
                                      to_hu=false)
            vmi_dict[E] = T.(vmi_vol)
        end
    end

    # 8. Standard reconstruction from high-kVp sinogram
    recon_vol = _run_reconstruction(sino_final, geom, recon_opts)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]

    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    return SimulationResult(
        sino_ideal, sino_final, recons, geom, config,
        de_sino, mat_map, vmi_dict,
        nothing, nothing, pcct_vmi_dict
    )
end

# =============================================================================
# Mode 3: Helical Single-kVp
# =============================================================================

function _simulate_helical_single(phantom, scanner, protocol, sim_opts, recon_opts)
    # 1. Build axial geometry first (for beam parameters)
    n_angles_total = round(Int, protocol.views * protocol.n_rotations)
    base_geom = CTGeometry(
        scanner;
        n_angles = n_angles_total,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing
    )

    # 2. Create helical geometry (modifies source/detector z-positions for helical motion)
    helical_geom = create_helical_geometry(
        base_geom;
        pitch = protocol.pitch,
        rotation_time = protocol.rotation_time,
        z_start = 0.0
    )

    # 3. Resolve spectrum
    energies, weights = resolve_spectrum(sim_opts, protocol)

    # 4. Build PhysicsConfig
    config = build_physics_config(scanner, sim_opts, energies, weights)

    # 5. Helical forward projection (use helical geometry with z-varying positions)
    materials = get_region_materials()
    sino_ideal = forward_project(
        phantom.mask, helical_geom.base_geom;
        energies=energies, weights=weights,
        materials=materials, physics=config
    )

    # 6. Apply detector noise
    sino_final = if sim_opts.use_noise
        sim_detect(sino_ideal, helical_geom.base_geom, protocol)
    else
        copy(sino_ideal)
    end

    # 7. Helical reconstruction
    recon_vol = _run_helical_reconstruction(sino_final, helical_geom, recon_opts)

    T = eltype(recon_vol)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()

    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    return SimulationResult(
        sino_ideal, sino_final, recons, helical_geom, config,
        nothing, nothing, vmi_dict,
        nothing, nothing, pcct_vmi_dict
    )
end

# =============================================================================
# Mode 4: Helical Dual-kVp
# =============================================================================

function _simulate_helical_dual(phantom, scanner, protocol, sim_opts, recon_opts)
    # 1. Build helical geometry
    n_angles_total = round(Int, protocol.views * protocol.n_rotations)
    base_geom = CTGeometry(
        scanner;
        n_angles = n_angles_total,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing
    )

    helical_geom = create_helical_geometry(
        base_geom;
        pitch = protocol.pitch,
        rotation_time = protocol.rotation_time,
        z_start = 0.0
    )

    # 2. Build PhysicsConfig (high-kVp)
    energies_high, weights_high = resolve_spectrum(sim_opts, protocol)
    config = build_physics_config(scanner, sim_opts, energies_high, weights_high)

    # 3. Build GSIProtocol
    gsi = _build_gsi_protocol(protocol)

    # 4. Dual-energy forward projection (uses helical geometry with z-varying positions)
    materials = get_region_materials()
    de_sino = forward_project_dual_energy(
        phantom.mask, helical_geom.base_geom, gsi;
        materials=materials,
        physics=config,
        scanner=nothing
    )

    # 5. High-kVp sinogram as primary
    sino_ideal = de_sino.high
    sino_final = if sim_opts.use_noise
        sim_detect(sino_ideal, helical_geom.base_geom, protocol)
    else
        copy(sino_ideal)
    end

    # 6. Material decomposition
    mat_map = decompose_materials(de_sino; basis=Tuple(recon_opts.vmi_basis[1:2]))

    # 7. VMI reconstruction (helical)
    T = eltype(sino_final)
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    if !isempty(recon_opts.vmi_energies)
        for E in recon_opts.vmi_energies
            # Generate VMI sinogram, then helical reconstruct
            vmi_sino = virtual_monoenergetic(mat_map, E)
            vmi_vol = helical_fdk_reconstruct_volume(
                T.(vmi_sino), helical_geom, recon_opts.matrix_size;
                interpolation = recon_opts.interpolation == :li_360 ? :li360 : :li180
            )
            vmi_dict[E] = vmi_vol
        end
    end

    # 8. Helical reconstruction from high-kVp sinogram
    recon_vol = _run_helical_reconstruction(sino_final, helical_geom, recon_opts)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]

    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    return SimulationResult(
        sino_ideal, sino_final, recons, helical_geom, config,
        de_sino, mat_map, vmi_dict,
        nothing, nothing, pcct_vmi_dict
    )
end

# =============================================================================
# Mode 5: Axial PCCT (Photon-Counting CT)
# =============================================================================

function _simulate_axial_pcct(phantom, scanner, protocol, sim_opts, recon_opts)
    # 1. Build Geometry
    geom = CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = nothing
    )

    # 2. Resolve spectrum — PCCT ALWAYS needs polychromatic spectrum
    # (energy-resolved detection is meaningless with monochromatic input)
    e_full, w_full = load_spectrum(Int(protocol.kVp))
    energies, weights = downsample_spectrum(e_full, w_full, sim_opts.n_energy_bins)

    # 3. Build PhysicsConfig
    config = build_physics_config(scanner, sim_opts, energies, weights)

    # 4. Build PCCT detector from Scanner
    pcct_detector = _build_pcct_detector(scanner)

    # 5. PCCT forward projection (mask+materials → energy-resolved sinogram)
    materials = get_region_materials()
    pcct_sino = pcct_forward_project(
        phantom.mask, geom, pcct_detector;
        energies=energies, weights=weights,
        materials=materials,
        apply_spectral_response=true
    )

    # 6. Also produce conventional sinogram (sum of all bins → single channel)
    # This is used for standard reconstruction — transfer to CPU for recon
    T = Float32
    sino_ideal_gpu = similar(pcct_sino.bins[1])
    fill!(sino_ideal_gpu, zero(T))
    for bin in pcct_sino.bins
        sino_ideal_gpu .+= bin
    end
    sino_ideal_gpu ./= T(length(pcct_sino.bins))
    sino_ideal = Array(sino_ideal_gpu)

    # 7. Apply PCCT noise (per-bin Poisson, no electronic noise)
    pcct_sino_noisy = if sim_opts.use_noise
        noisy_bins = [copy(b) for b in pcct_sino.bins]
        noisy_sino = EnergyResolvedSinogram(noisy_bins, copy(pcct_sino.thresholds_keV))
        apply_pcct_noise!(noisy_sino, pcct_detector, protocol;
                          seed=sim_opts.seed, I0=1e6,
                          energies=energies, weights=weights)
        noisy_sino
    else
        pcct_sino
    end

    # 8. Conventional noisy sinogram (for standard recon — CPU for reconstruction)
    sino_noisy_gpu = similar(pcct_sino_noisy.bins[1])
    fill!(sino_noisy_gpu, zero(T))
    for bin in pcct_sino_noisy.bins
        sino_noisy_gpu .+= bin
    end
    sino_noisy_gpu ./= T(length(pcct_sino_noisy.bins))
    sino_noisy = Array(sino_noisy_gpu)

    # 9. N-material decomposition (if vmi_basis specified with 2+ materials)
    # Decomposition is a per-pixel CPU operation — transfer bins to CPU if on GPU
    pcct_mat_map = if length(recon_opts.vmi_basis) >= 2
        basis_tuple = Tuple(recon_opts.vmi_basis)
        cpu_bins = [Array(b) for b in pcct_sino_noisy.bins]
        cpu_sino = EnergyResolvedSinogram(cpu_bins, pcct_sino_noisy.thresholds_keV)
        pcct_material_decomposition(cpu_sino; basis=basis_tuple)
    else
        nothing
    end

    # 10. VMI synthesis (if energies specified and decomposition succeeded)
    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    if !isempty(recon_opts.vmi_energies) && !isnothing(pcct_mat_map)
        for E in recon_opts.vmi_energies
            vmi_sino = synthesize_vmi(pcct_mat_map, E)
            # Reconstruct VMI volume
            vmi_vol = _run_reconstruction(vmi_sino, geom, recon_opts)
            pcct_vmi_dict[E] = vmi_vol
        end
    end

    # 11. Standard reconstruction from combined sinogram
    recon_vol = _run_reconstruction(sino_noisy, geom, recon_opts)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()

    return SimulationResult(
        sino_ideal, sino_noisy, recons, geom, config,
        nothing, nothing, vmi_dict,
        pcct_sino_noisy, pcct_mat_map, pcct_vmi_dict
    )
end

# =============================================================================
# Reconstruction Dispatcher (axial)
# =============================================================================

"""
    _run_reconstruction(sinogram, geom, recon_opts) -> Array{T,3}

Dispatch to the correct reconstruction algorithm based on ReconOptions.
"""
function _run_reconstruction(
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    recon_opts::ReconOptions
) where T
    alg = recon_opts.algorithm
    ms = recon_opts.matrix_size
    ws = recon_opts.warm_start

    # Resolve init: warm_start array takes priority, otherwise use algorithm default
    # SIRT/CGLS/TV-SIRT/TV-CGLS default to :zeros; MBIR defaults to :fdk
    init_sirt = isnothing(ws) ? :zeros : ws
    init_mbir = isnothing(ws) ? :fdk : ws

    if alg == :fdk
        return fdk_reconstruct(sinogram, geom, ms)
    elseif alg == :sirt
        return sirt_reconstruct(sinogram, geom, ms;
            niter=recon_opts.iterations, lambda=recon_opts.lambda,
            init=init_sirt)
    elseif alg == :cgls
        return cgls_reconstruct(sinogram, geom, ms;
            niter=recon_opts.iterations,
            init=init_sirt)
    elseif alg == :tv_sirt
        return tv_sirt_reconstruct(sinogram, geom, ms;
            niter=recon_opts.iterations, lambda_sirt=recon_opts.lambda,
            lambda_tv=recon_opts.tv_weight,
            init=init_sirt)
    elseif alg == :tv_cgls
        return tv_cgls_reconstruct(sinogram, geom, ms;
            niter=recon_opts.iterations, lambda_tv=recon_opts.tv_weight,
            init=init_sirt)
    elseif alg == :asir
        return asir_style_reconstruct(sinogram, geom, ms;
            niter=recon_opts.iterations, lambda=recon_opts.lambda,
            blend_percent=recon_opts.blend_percent)
    elseif alg == :mbir
        penalty_type = _resolve_penalty(recon_opts.penalty, recon_opts.penalty_delta)
        return mbir_reconstruct(sinogram, geom, ms;
            niter=recon_opts.iterations, n_subsets=recon_opts.n_subsets,
            lambda=recon_opts.lambda, penalty=penalty_type,
            use_edge_weights=recon_opts.use_edge_weights,
            init=init_mbir)
    elseif alg == :helical_fdk || alg == :helical_sirt
        # If user requests helical algo on axial data, fall back to axial equivalent
        if alg == :helical_fdk
            return fdk_reconstruct(sinogram, geom, ms)
        else
            return sirt_reconstruct(sinogram, geom, ms;
                niter=recon_opts.iterations, lambda=recon_opts.lambda,
                init=init_sirt)
        end
    else
        error("Unknown reconstruction algorithm: $alg. " *
              "Supported: :fdk, :sirt, :cgls, :tv_sirt, :tv_cgls, :asir, :mbir, :helical_fdk, :helical_sirt")
    end
end

# =============================================================================
# Helical Reconstruction Dispatcher
# =============================================================================

"""
    _run_helical_reconstruction(sinogram, helical_geom, recon_opts) -> Array{T,3}

Dispatch to helical reconstruction algorithms.
"""
function _run_helical_reconstruction(
    sinogram::AbstractArray{T, 3},
    helical_geom::HelicalGeometry,
    recon_opts::ReconOptions
) where T
    alg = recon_opts.algorithm
    ms = recon_opts.matrix_size
    ws = recon_opts.warm_start
    interp = recon_opts.interpolation == :li_360 ? :li360 : :li180

    # Resolve init: warm_start array takes priority, otherwise use algorithm default
    init_helical = isnothing(ws) ? :zeros : ws

    if alg ∈ (:fdk, :helical_fdk)
        return helical_fdk_reconstruct_volume(sinogram, helical_geom, ms;
            interpolation=interp)
    elseif alg ∈ (:sirt, :helical_sirt)
        return helical_sirt_reconstruct(sinogram, helical_geom, ms;
            niter=recon_opts.iterations, lambda=recon_opts.lambda,
            init=init_helical)
    elseif alg == :cgls
        # No helical CGLS — fall back to helical SIRT
        return helical_sirt_reconstruct(sinogram, helical_geom, ms;
            niter=recon_opts.iterations, lambda=recon_opts.lambda,
            init=init_helical)
    else
        # For other algorithms, use helical FDK as fallback
        return helical_fdk_reconstruct_volume(sinogram, helical_geom, ms;
            interpolation=interp)
    end
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _build_gsi_protocol(protocol::CTProtocol) -> GSIProtocol

Build a GSIProtocol from CTProtocol dual-energy fields.
"""
function _build_gsi_protocol(protocol::CTProtocol)
    return GSIProtocol(
        Int(protocol.kVp_low),       # low_kvp
        Int(protocol.kVp),           # high_kvp
        protocol.mA_low > 0 ? protocol.mA_low : protocol.mA,  # low_mA
        protocol.mA,                 # high_mA
        protocol.integration_fraction,
        protocol.rotation_time,
        protocol.views
    )
end

"""
    _resolve_penalty(penalty::Symbol, delta::Float64) -> PenaltyType

Convert penalty symbol to PenaltyType instance.
"""
function _resolve_penalty(penalty::Symbol, delta::Float64)
    if penalty == :quadratic
        return QuadraticPenalty()
    elseif penalty == :huber
        return HuberPenalty(delta)
    elseif penalty == :hyperbola
        return HyperbolaPenalty(delta)
    else
        return QuadraticPenalty()  # Default fallback
    end
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
