"""
    Simulation/Driver.jl

High-level driver for running end-to-end CT simulations.
"""

export simulate, simulate!, SimulationResult

# =============================================================================
# GPU Array Handling
# =============================================================================

"""
    _resolve_materials(phantom, materials_kwarg) -> Vector{XA.Material}

Resolve materials with priority: (1) explicit kwarg, (2) phantom.materials, (3) fallback.

This is the v20.0 unified materials resolution logic used by all simulate() functions.
"""
function _resolve_materials(phantom, materials_kwarg::Union{Nothing, Vector})
    if !isnothing(materials_kwarg)
        # Priority 1: Explicit materials kwarg override
        return materials_kwarg
    elseif hasproperty(phantom, :materials) && !isnothing(phantom.materials)
        # Priority 2: Materials stored in phantom (v20.0 unified API)
        return phantom.materials
    else
        # Priority 3: Fallback to Gammex 472 region materials (backwards compat)
        return get_region_materials()
    end
end

"""
    _to_gpu(arr::AbstractArray)

Move array to GPU if a GPU backend is available.
Automatically detects Metal, CUDA, or AMDGPU and uses the appropriate array type.
Falls back to CPU if no GPU backend is loaded.
"""
function _to_gpu(arr::AbstractArray)
    # Check for Metal (Apple Silicon)
    if isdefined(Main, :Metal) && isdefined(Main.Metal, :MtlArray)
        return Main.Metal.MtlArray(arr)
    end
    # Check for CUDA (NVIDIA)
    if isdefined(Main, :CUDA) && isdefined(Main.CUDA, :CuArray)
        return Main.CUDA.CuArray(arr)
    end
    # Check for AMDGPU (AMD)
    if isdefined(Main, :AMDGPU) && isdefined(Main.AMDGPU, :ROCArray)
        return Main.AMDGPU.ROCArray(arr)
    end
    # No GPU backend - return as-is
    return arr
end

"""
    SimulationResult

Container for simulation outputs. Supports single and multi-reconstruction,
dual-energy sinograms, material maps, and VMI volumes.

# Core Fields
- `sinogram_ideal`: Noise-free sinogram (or high-kVp sinogram for DE)
- `sinogram_noisy`: Noisy sinogram after detector simulation
- `reconstructions`: Vector of (algorithm_name, volume) pairs
- `geometry`: CTGeometry used for simulation
- `physics_config`: PhysicsConfig with all enabled effects

# Dual-Energy Fields
- `de_sinogram`: DualEnergySinogram (low/high kVp pair), nothing if single-kVp
- `material_maps`: MaterialMap from decomposition, nothing if single-kVp
- `vmi_volumes`: Dict{Float64, Array} of VMI reconstructions by energy

# PCCT Fields
- `pcct_sinogram`: EnergyResolvedSinogram from PCCT, nothing if not PCCT
- `pcct_material_maps`: Legacy field (always nothing — use `material_maps` for PCCT decomposition)
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
    simulate(phantom, scanner, protocol, sim_opts, recon_opts; materials=nothing)

Run a full end-to-end CT simulation with automatic mode routing.

The 4-struct API: Scanner provides hardware parameters, CTProtocol provides acquisition
settings, SimOptions controls which physics effects are enabled, and ReconOptions
controls image reconstruction. The driver automatically routes between 3 scan modes:

1. **Axial single-kVp** (default): Standard CT acquisition
2. **Axial dual-kVp**: Dual-energy with VMI pipeline
3. **Axial PCCT**: Photon-counting CT with energy-resolved sinograms, N-material
   decomposition, and material-based VMI synthesis (auto-detected via Scanner)

# Arguments
- `phantom`: Struct containing `.mask` (UInt8) and material definitions.
- `scanner`: `Scanner` hardware definition.
- `protocol`: `CTProtocol` acquisition settings.
- `sim_opts`: `SimOptions` for physics fidelity (controls all 14 effects).
- `recon_opts`: `ReconOptions` or `Vector{ReconOptions}` for multi-recon.

# Keyword Arguments
- `materials::Union{Nothing, Vector}=nothing`: Custom materials vector for arbitrary phantoms.
  If `nothing`, uses `get_region_materials()` (Gammex 472 materials, backwards compatible).
  For custom phantoms, provide a `Vector{XA.Material}` where `materials[mask_value + 1]`
  returns the material for each voxel. Use `build_materials_vector(materials_dict)` to
  create this from a Dict{Int, XA.Material}.

# Returns
`SimulationResult` containing sinograms, reconstructions, and optional DE outputs.

# Examples
```julia
# Axial single-kVp (unchanged from before)
result = simulate(phantom, scanner, CTProtocol(kVp=120, mA=200), SimOptions(), ReconOptions())

# Dual-energy axial with VMI
result = simulate(phantom, scanner,
    CTProtocol(dual_energy=true, kVp=140, mA=200, kVp_low=80, mA_low=350),
    SimOptions(), ReconOptions(vmi_energies=[50.0, 70.0, 100.0]))

# Multi-recon from one scan
recon_list = [ReconOptions(algorithm=:fdk), ReconOptions(algorithm=:sirt, iterations=50)]
result = simulate(phantom, scanner, protocol, sim_opts, recon_list)

# Custom phantom with arbitrary materials (XCAT, custom segmentation, etc.)
import XrayAttenuation as XA
materials_dict = Dict(0 => XA.Materials.air, 1 => XA.Materials.water, 2 => XA.Materials.corticalbone)
phantom = create_phantom_from_mask(labeled_array, materials_dict, (0.1, 0.1, 0.1))
materials_vec = build_materials_vector(materials_dict)
result = simulate(phantom, scanner, protocol, sim_opts, recon_opts; materials=materials_vec)
```
"""
function simulate(
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions();
    materials::Union{Nothing, Vector} = nothing
)
    # Route based on dual_energy and PCCT
    is_dual = protocol.dual_energy
    _is_pcct = is_pcct(scanner)

    if _is_pcct && is_dual
        error("PCCT scanners cannot use dual_energy mode — spectral info comes from detector energy bins, not dual kVp. Set dual_energy=false.")
    end

    if _is_pcct
        # PCCT mode: photon-counting scanner detected
        return _simulate_axial_pcct(phantom, scanner, protocol, sim_opts, recon_opts; materials=materials)
    elseif !is_dual
        return _simulate_axial_single(phantom, scanner, protocol, sim_opts, recon_opts; materials=materials)
    else
        return _simulate_axial_dual(phantom, scanner, protocol, sim_opts, recon_opts; materials=materials)
    end
end

# Multi-recon dispatch: simulate() with Vector{ReconOptions}
function simulate(
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions,
    recon_opts_list::Vector{ReconOptions};
    materials::Union{Nothing, Vector} = nothing
)
    # Run simulation with first recon option to get sinograms
    first_result = simulate(phantom, scanner, protocol, sim_opts, recon_opts_list[1]; materials=materials)

    # Reconstruct with additional options from the same sinogram
    T = eltype(first_result.sinogram_noisy)
    recons = copy(first_result.reconstructions)
    geom = first_result.geometry

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
                opts.vmi_energies, opts.vmi_basis,
                prev_vol, opts.cascade_warm_start
            )
        else
            opts
        end

        vol = _run_reconstruction(first_result.sinogram_noisy, geom, effective_opts)
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
# Mode 1: Axial Single-kVp (DEPRECATED — use workspace-based simulate!() instead)
# =============================================================================

function _simulate_axial_single(phantom, scanner, protocol, sim_opts, recon_opts;
                                materials::Union{Nothing, Vector} = nothing)
    # 1. Build Geometry
    geom = CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = recon_opts.z_cm,
        collimation_mm = protocol.collimation_mm
    )

    # 2. Resolve spectrum
    energies, weights = resolve_spectrum(sim_opts, protocol)

    # 3. Build PhysicsConfig (with phantom for size-aware scatter)
    config = build_physics_config(scanner, sim_opts, energies, weights; phantom=phantom)

    # 4. Forward Project (move mask to GPU if available)
    # Use custom materials if provided, otherwise default to Gammex region materials
    mats = _resolve_materials(phantom, materials)
    mask_gpu = _to_gpu(phantom.mask)
    sino_ideal = forward_project(
        mask_gpu, geom;
        energies=energies, weights=weights,
        materials=mats, physics=config,
        volume_extent=phantom.extent
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
# Mode 2: Axial Dual-kVp (WRAPPER AROUND SINGLE-KVP)
# =============================================================================
#
# v11.0 REFACTOR: Dual-energy now runs single-kVp pipeline TWICE.
#
# Previous implementation had wave artifacts because:
# - Scatter was ADDED with energy-dependent coefficients (different for 80/140 kVp)
# - Scatter was CORRECTED with joint estimate at average energy
# - The coefficient mismatch created different residuals that amplified in decomposition
#
# New implementation:
# - Run single-kVp for 80 kVp → scatter added AND corrected at 50 keV (matched!)
# - Run single-kVp for 140 kVp → scatter added AND corrected at 70 keV (matched!)
# - Each sinogram is fully corrected before decomposition
# - No special dual-energy scatter handling needed
#
# See DESIGN-DE-WRAPPER in progress.md for full rationale.
# =============================================================================

"""
    _forward_single_pass(phantom, scanner, protocol, sim_opts, geom; materials=nothing) -> (sinogram, config)

Run forward projection for a single kVp with all physics effects.
Returns fully-corrected sinogram (scatter added AND corrected at same energy).

This is the core building block for both single-kVp and dual-kVp simulations.
By using this for dual-energy, we ensure scatter is properly matched.
"""
function _forward_single_pass(phantom, scanner, protocol, sim_opts, geom;
                              materials::Union{Nothing, Vector} = nothing)
    # Resolve spectrum for this kVp
    energies, weights = resolve_spectrum(sim_opts, protocol)

    # Build PhysicsConfig (scatter add + scatter correct at SAME energy)
    config = build_physics_config(scanner, sim_opts, energies, weights; phantom=phantom)

    # Forward project with all physics
    # Use custom materials if provided, otherwise default to Gammex region materials
    mats = _resolve_materials(phantom, materials)
    mask_gpu = _to_gpu(phantom.mask)
    sinogram = forward_project(
        mask_gpu, geom;
        energies=energies, weights=weights,
        materials=mats, physics=config,
        volume_extent=phantom.extent
    )

    return sinogram, config
end

function _simulate_axial_dual(phantom, scanner, protocol, sim_opts, recon_opts;
                              materials::Union{Nothing, Vector} = nothing)
    # 1. Build Geometry (shared for both kVp)
    # DEPRECATED: prefer workspace-based simulate!() path
    geom = CTGeometry(
        scanner;
        n_angles = protocol.views,
        fov_cm = recon_opts.fov_cm,
        z_cm = recon_opts.z_cm,
        collimation_mm = protocol.collimation_mm
    )

    # 2. Create single-kVp protocols for each energy level
    # These are single-kVp protocols (dual_energy=false) so that
    # _forward_single_pass uses the standard scatter add/correct pipeline
    protocol_low = CTProtocol(
        mA = protocol.mA_low > 0 ? protocol.mA_low : protocol.mA,
        kVp = protocol.kVp_low,       # 80 kVp
        views = protocol.views,
        rotation_time = protocol.rotation_time,
        flux_density = protocol.flux_density,
        spectrum_path = nothing,
        dual_energy = false           # Single-kVp mode for clean scatter handling
    )

    protocol_high = CTProtocol(
        mA = protocol.mA,
        kVp = protocol.kVp,           # 140 kVp
        views = protocol.views,
        rotation_time = protocol.rotation_time,
        flux_density = protocol.flux_density,
        spectrum_path = nothing,
        dual_energy = false           # Single-kVp mode for clean scatter handling
    )

    # 3. Run single-kVp pipeline for LOW kVp (80 kVp)
    # Scatter is added AND corrected at ~50 keV (matched coefficients!)
    sino_low, config_low = _forward_single_pass(phantom, scanner, protocol_low, sim_opts, geom; materials=materials)

    # 4. Run single-kVp pipeline for HIGH kVp (140 kVp)
    # Scatter is added AND corrected at ~70 keV (matched coefficients!)
    sino_high, config_high = _forward_single_pass(phantom, scanner, protocol_high, sim_opts, geom; materials=materials)

    # 5. Apply detector noise to BOTH sinograms for realistic downstream processing
    # sim_detect returns a copy — originals remain clean for sino_ideal
    sino_ideal = sino_high
    if sim_opts.use_noise
        sino_low_noisy = sim_detect(sino_low, geom, protocol_low)
        sino_high_noisy = sim_detect(sino_high, geom, protocol_high)
    else
        sino_low_noisy = sino_low
        sino_high_noisy = sino_high
    end
    sino_final = sino_high_noisy

    # 6. Create DualEnergySinogram from (potentially noisy) sinograms
    # With use_noise=true: material decomposition sees realistic noise
    # With use_noise=false: identical to previous behavior (clean sinograms)
    de_sino = DualEnergySinogram(sino_low_noisy, sino_high_noisy;
        low_kvp = Int(protocol.kVp_low),
        high_kvp = Int(protocol.kVp)
    )

    # 7. Material decomposition (now operates on noisy sinograms when use_noise=true)
    mat_map = decompose_materials(de_sino; basis=Tuple(recon_opts.vmi_basis[1:2]))

    # 8. VMI reconstruction (if energies specified)
    T = eltype(sino_final)
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    if !isempty(recon_opts.vmi_energies)
        for E in recon_opts.vmi_energies
            vmi_sino = virtual_monoenergetic(mat_map, E)
            vmi_vol = _run_reconstruction(vmi_sino, geom, recon_opts)
            vmi_dict[E] = T.(vmi_vol)
        end
    end

    # 9. Standard reconstruction from high-kVp sinogram
    recon_vol = _run_reconstruction(sino_final, geom, recon_opts)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]

    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    return SimulationResult(
        sino_ideal, sino_final, recons, geom, config_high,
        de_sino, mat_map, vmi_dict,
        nothing, nothing, pcct_vmi_dict
    )
end

# =============================================================================
# PCCT Combined Sinogram Helper
# =============================================================================

"""
    _combine_pcct_bins(pcct_sino, detector, energies, weights, kVp; I0=1e6,
                        apply_detector_effects=false, apply_corrections=false, flux_rate=1e8)

Combine energy-resolved PCCT sinogram bins into a single conventional-equivalent
sinogram using correct physics.

The bins are in line-integral domain: sino_bin = -log(N_bin / I0_bin).
To combine correctly, the SAME I0 used for normalization must be used here:
1. Convert back to counts: N_bin = I0_bin × exp(-sino_bin)
2. Sum counts: N_total = Σ N_bin
3. Sum reference: I0_total = Σ I0_bin
4. Combined: sino = -log(N_total / I0_total)

When `apply_detector_effects=true` and `apply_corrections=false`, uses degraded I0
(consistent with `pcct_forward_project` normalization when detector effects are applied
without corrections). When corrections are applied, uses theoretical I0.

Returns a GPU array (same device as input bins).
"""
function _combine_pcct_bins(pcct_sino::EnergyResolvedSinogram, detector::PhotonCountingDetector,
                             energies, weights, kVp; I0=1e6,
                             apply_detector_effects::Bool=false, apply_corrections::Bool=false,
                             flux_rate::Real=1e8,
                             output=nothing,
                             ws_I0_bins=nothing)
    T = Float32
    n_bins = length(pcct_sino.bins)
    thresholds = detector.energy_thresholds_keV

    # Compute per-bin I0 values — MUST match what pcct_forward_project used for normalization
    I0_bins = if ws_I0_bins !== nothing
        ws_I0_bins  # Pre-computed by caller (zero-alloc path)
    elseif apply_detector_effects && !apply_corrections
        # Effects without corrections: use degraded I0
        η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)
        R = compute_spectral_response_matrix(
            detector.material, detector.thickness_mm, thresholds, kVp;
            energy_resolution_keV=detector.energy_resolution_keV,
            pixel_size_mm=detector.pixel_size_mm,
            include_fluorescence=true,
            include_tailing=true,
            n_energy_points=length(energies)
        )
        _compute_degraded_I0(detector, energies, weights, η, thresholds, kVp, I0, flux_rate; R=R)
    else
        # No effects OR effects+corrections: use theoretical I0
        η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)
        R = compute_spectral_response_matrix(
            detector.material, detector.thickness_mm, thresholds, kVp;
            energy_resolution_keV=detector.energy_resolution_keV,
            pixel_size_mm=detector.pixel_size_mm,
            include_fluorescence=true,
            include_tailing=true,
            n_energy_points=length(energies)
        )
        [_compute_bin_I0(detector, energies, weights, η, thresholds, b,
                          Float64(kVp), Float64(I0); R=R) for b in 1:n_bins]
    end
    I0_total = T(sum(I0_bins))

    # Accumulate total photon counts: N_total = Σ I0_bin × exp(-sino_bin)
    # Reuse pre-allocated output buffer if provided
    N_total_gpu = output === nothing ? similar(pcct_sino.bins[1]) : output
    fill!(N_total_gpu, zero(T))

    eps_val = T(1e-10)
    for (b, bin_sino) in enumerate(pcct_sino.bins)
        let I0b = T(I0_bins[b]), bs = bin_sino, nt = N_total_gpu
            AK.foreachindex(bs) do idx
                nt[idx] += I0b * exp(-bs[idx])
            end
        end
    end

    # Combined sinogram = -log(N_total / I0_total)
    let nt = N_total_gpu, I0t = I0_total, eps = eps_val
        AK.foreachindex(nt) do idx
            nt[idx] = -log(max(nt[idx], eps) / I0t)
        end
    end

    return N_total_gpu
end

# =============================================================================
# simulate!() — Zero-allocation PCCT simulation hot path
# =============================================================================

"""
    simulate!(ws::PCCTWorkspace, phantom, scanner, protocol, sim_opts, recon_opts; materials=nothing)

Run PCCT simulation using pre-allocated workspace buffers for zero allocations.

The first call may allocate due to JIT compilation. The second call with the same
workspace achieves `@allocated == 0`.

Create the workspace with `create_workspace(scanner, protocol, sim_opts, recon_opts, phantom)`.

All setup data (geometry, spectrum, physics config, detector, spectral response matrices)
is pre-computed in the workspace by `create_workspace()`. Reconstruction is NOT included —
it is handled by the `simulate()` wrapper or `_simulate_axial_pcct`.

# Returns
`SimulationResult` with sinograms, material maps, and VMI sinograms from workspace.
Reconstruction fields are empty (filled by wrapper).
"""
function simulate!(
    ws::PCCTWorkspace{T},
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions();
    materials::Union{Nothing, Vector} = nothing
) where {T}
    # All setup data comes from workspace (pre-computed in create_workspace)
    geom = ws.geom
    energies = ws.energies
    weights = ws.weights
    config = ws.config
    pcct_detector = ws.pcct_detector
    mats = ws.mats
    use_detector_fx = ws.use_detector_fx
    use_corrections = ws.use_corrections
    kVp = ws.kVp

    # Forward projection with ALL workspace buffers
    pcct_sino = pcct_forward_project(
        phantom.mask, geom, pcct_detector;
        energies=energies, weights=weights,
        materials=mats,
        apply_spectral_response=true,
        apply_detector_effects=use_detector_fx,
        apply_corrections=use_corrections,
        ws_bins=ws.bins, ws_μ_volume=ws.μ_volume, ws_sino_buf=ws.sino_buf,
        ws_scratch=ws.scratch, ws_total_counts=ws.total_counts,
        ws_thresholds_T=ws.thresholds_T,
        ws_η=ws.η, ws_R=ws.R, ws_R_energies=ws.R_energies,
        ws_I0_bins_norm=ws.I0_bins_norm,
        ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
        ws_μ_table=ws.μ_table,
        ws_pileup_counts=ws.pileup_counts,
        ws_pileup_migration=ws.pileup_migration,
        ws_correction_counts=ws.correction_pileup_counts,
        ws_correction_migration=ws.correction_migration,
        ws_pileup_S=ws.pileup_S,
        ws_pileup_thresh=ws.pileup_thresh,
        ws_pileup_E_low=ws.pileup_E_low,
        ws_pileup_E_high=ws.pileup_E_high,
        ws_pileup_E_centers=ws.pileup_E_centers,
        ws_pileup_w=ws.pileup_w,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_charge_probs=ws.charge_sharing_probs,
        volume_extent=phantom.extent
    )

    # Combine ideal (workspace buffer + pre-computed I0_bins)
    sino_ideal_gpu = _combine_pcct_bins(pcct_sino, pcct_detector, energies, weights, kVp;
                                         apply_detector_effects=use_detector_fx,
                                         apply_corrections=use_corrections,
                                         output=ws.combined,
                                         ws_I0_bins=ws.I0_bins)

    # BHC on ideal
    if config.bhc !== nothing
        apply_bhc!(sino_ideal_gpu, config.bhc; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
    end

    # Save ideal to CPU workspace buffer
    copyto!(ws.sino_ideal_out, sino_ideal_gpu)

    # Noise (in-place on pcct_sino.bins)
    if sim_opts.use_noise
        I0_physics = compute_detector_I0(geom, protocol)
        apply_pcct_noise!(pcct_sino, pcct_detector, protocol;
                          seed=sim_opts.seed, I0=I0_physics,
                          energies=energies, weights=weights,
                          ws_noise_staging=ws.noise_staging,
                          ws_noise_buf=ws.noise_buf,
                          ws_rng=ws.rng,
                          ws_noise_I0=ws.noise_I0,
                          ws_η=ws.η,
                          noise_reduction=sim_opts.pcct_noise_reduction)
    end

    # Combine noisy (reuse workspace buffer + pre-computed I0_bins)
    sino_noisy_gpu = _combine_pcct_bins(pcct_sino, pcct_detector, energies, weights, kVp;
                                         apply_detector_effects=use_detector_fx,
                                         apply_corrections=use_corrections,
                                         output=ws.combined,
                                         ws_I0_bins=ws.I0_bins)

    # BHC on noisy
    if config.bhc !== nothing
        apply_bhc!(sino_noisy_gpu, config.bhc; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
    end

    # Save noisy to CPU workspace buffer (uncorrected, for inspection)
    copyto!(ws.sino_noisy_out, sino_noisy_gpu)

    # Combine PCCT bins into pseudo-dual-energy sinograms (GPU)
    combine_pcct_bins!(ws.vmi_sino_low, ws.vmi_sino_high, pcct_sino.bins; split_bin=2)

    # 2-material decomposition (same as dual-kVp, GPU)
    spectral_decompose!(ws.vmi_material1, ws.vmi_material2,
                        ws.vmi_sino_low, ws.vmi_sino_high,
                        ws.vmi_inv_a11, ws.vmi_inv_a12, ws.vmi_inv_a21, ws.vmi_inv_a22)

    mat_map = if length(ws.basis_tuple) >= 2
        MaterialMap(ws.vmi_material1, ws.vmi_material2;
            material1_name=ws.basis_tuple[1], material2_name=ws.basis_tuple[2],
            domain=:projection)
    else
        nothing
    end

    # Return intermediate results — reconstruction and VMI are done by the wrapper
    # (they inherently allocate new volumes, outside zero-alloc scope)
    return (pcct_sino=pcct_sino, mat_map=mat_map)
end

# =============================================================================
# simulate!() — Zero-allocation EICT single-kVp simulation hot path
# =============================================================================

"""
    simulate!(ws::EICTWorkspace, phantom, scanner, protocol, sim_opts, recon_opts; materials=nothing)

Run EICT single-kVp simulation using pre-allocated workspace buffers.

Create the workspace with `create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)`.
Reconstruction is NOT included — handled by the wrapper.
"""
function simulate!(
    ws::EICTWorkspace{T},
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions();
    materials::Union{Nothing, Vector} = nothing
) where {T}
    geom = ws.geom
    energies = ws.energies
    config = ws.config
    # Re-resolve materials from the incoming phantom each call.
    # This supports dynamic phantoms (e.g. time-varying iodine contrast) where
    # phantom.materials changes between simulate! calls on the same workspace.
    # When materials change, skip the pre-sized LUT workspace buffers so
    # create_μ_volume! allocates fresh ones of the correct length.
    mats = _resolve_materials(phantom, materials)
    lut_cpu, lut_gpu, μ_table = if length(mats) == length(ws.mats)
        ws.μ_lut_cpu, ws.μ_lut_gpu, ws.μ_table
    else
        nothing, nothing, nothing
    end

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Polychromatic forward projection (Beer-Lambert)
    # ═══════════════════════════════════════════════════════════════════════
    fill!(ws.sinogram, zero(T))
    _forward_project_poly!(ws.sinogram, phantom.mask, geom, energies, ws.weights, mats;
                            ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
                            ws_I_transmitted=ws.I_transmitted,
                            ws_weights_norm=ws.weights_norm,
                            ws_μ_lut_cpu=lut_cpu, ws_μ_lut_gpu=lut_gpu,
                            ws_μ_table=μ_table,
                            ws_source_positions=ws.geom_source_positions,
                            ws_detector_centers=ws.geom_detector_centers,
                            ws_detector_u=ws.geom_detector_u,
                            ws_detector_v=ws.geom_detector_v,
                            volume_extent=phantom.extent)

    if ws.has_signal_chain
        # ═══════════════════════════════════════════════════════════════════
        # CatSim signal chain — inlined from _forward_project_with_signal_chain!
        # Uses workspace buffers to avoid allocations
        # ═══════════════════════════════════════════════════════════════════

        heel_effect = ws.heel_effect
        das_model = ws.das_model
        bhc_eff = ws.bhc

        # STEP 2: Apply physics pipeline (sinogram domain, no noise)
        _apply_physics_no_noise!(ws.sinogram, geom, config;
            ws_output=ws.physics_output,
            ws_scatter_kernel=ws.scatter_kernel,
            ws_scatter_correct_kernel=ws.scatter_correct_kernel,
            ws_crosstalk_kernel=ws.crosstalk_kernel,
            ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
            ws_focal_spot_kernel=ws.focal_spot_kernel,
            ws_flat_filter_projection=ws.flat_filter_projection,
            ws_bowtie_projection=ws.bowtie_projection,
            ws_lag_output=ws.physics_output,
            ws_lag_intensity=ws.lag_intensity,
            ws_lag_coeffs=ws.lag_coeffs)

        # STEP 3: Convert to intensity domain
        eps = T(1e-10)
        let sino = ws.sinogram
            AK.foreachindex(sino) do idx
                sino[idx] = exp(-clamp(sino[idx], T(-1), T(15)))
            end
        end

        # STEP 4: Apply heel effect to phantom intensity
        if heel_effect !== nothing
            apply_heel_effect!(ws.sinogram, heel_effect, geom)
        end

        # STEP 5: Apply DAS model (gain + noise) to phantom
        if das_model !== nothing
            apply_das_model!(ws.sinogram, das_model; seed=config.noise_seed)
        end

        # STEP 6: Create noise-free air scan (workspace buffer)
        fill!(ws.air_scan, one(T))
        if heel_effect !== nothing
            apply_heel_effect!(ws.air_scan, heel_effect, geom)
        end
        if das_model !== nothing
            gain = T(das_model.gain)
            let air = ws.air_scan
                AK.foreachindex(air) do idx
                    air[idx] *= gain
                end
            end
        end

        # STEP 7: Calibration (prep = phantom / air)
        let sino = ws.sinogram, air = ws.air_scan
            AK.foreachindex(sino) do idx
                air_val = max(air[idx], eps)
                sino[idx] = sino[idx] / air_val
            end
        end

        # STEP 8: Low signal correction
        low_signal_correction_gpu!(ws.sinogram)

        # STEP 9: Log transform
        let sino = ws.sinogram
            AK.foreachindex(sino) do idx
                sino[idx] = -log(max(sino[idx], eps))
            end
        end

        # STEP 10: Beam hardening correction
        if bhc_eff !== nothing
            apply_bhc!(ws.sinogram, bhc_eff; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
        end
    else
        # Standard path (no signal chain) — apply physics + BHC separately
        if config !== nothing
            apply_physics_effects!(ws.sinogram, geom, config;
                ws_output=ws.physics_output,
                ws_scatter_kernel=ws.scatter_kernel,
                ws_scatter_correct_kernel=ws.scatter_correct_kernel,
                ws_crosstalk_kernel=ws.crosstalk_kernel,
                ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
                ws_focal_spot_kernel=ws.focal_spot_kernel,
                ws_flat_filter_projection=ws.flat_filter_projection,
                ws_bowtie_projection=ws.bowtie_projection,
                ws_lag_output=ws.physics_output,
                ws_lag_intensity=ws.lag_intensity,
                ws_lag_coeffs=ws.lag_coeffs,
                ws_bhc_coeffs_gpu=ws.bhc_coeffs_gpu)
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Save ideal sinogram to CPU
    # ═══════════════════════════════════════════════════════════════════════
    copyto!(ws.sino_ideal_out, ws.sinogram)

    # ═══════════════════════════════════════════════════════════════════════
    # Apply quantum noise (in-place using workspace buffers)
    # ═══════════════════════════════════════════════════════════════════════
    if sim_opts.use_noise
        I0_T = T(compute_detector_I0(geom, protocol))

        # Use default RNG (matches sim_detect behavior: seed=nothing)
        randn!(ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)

        let sino = ws.sinogram, rg = ws.noise_rand_gpu, I0v = I0_T
            AK.foreachindex(sino) do idx
                λ = I0v * exp(-sino[idx])
                λ_noisy = λ + sqrt(max(λ, T(1))) * rg[idx]
                λ_noisy = max(λ_noisy, T(1))
                sino[idx] = -log(λ_noisy / I0v)
            end
        end
    end

    # Save noisy sinogram to CPU
    copyto!(ws.sino_noisy_out, ws.sinogram)

    return (sino_ideal=ws.sino_ideal_out, sino_noisy=ws.sino_noisy_out)
end

# =============================================================================
# simulate!() — Zero-allocation EICT dual-kVp simulation hot path
# =============================================================================

"""
    simulate!(ws::EICTDualWorkspace, phantom, scanner, protocol, sim_opts, recon_opts; materials=nothing)

Run EICT dual-kVp simulation using pre-allocated workspace buffers.

Runs single-kVp pipeline twice (low + high kVp) using shared scratch buffers,
then performs material decomposition into workspace buffers.

Create the workspace with `create_eict_dual_workspace(scanner, protocol, sim_opts, recon_opts, phantom)`.
Reconstruction and VMI reconstruction are NOT included — handled by the wrapper.
"""
function simulate!(
    ws::EICTDualWorkspace{T},
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions();
    materials::Union{Nothing, Vector} = nothing
) where {T}
    geom = ws.geom
    mats = ws.mats

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1: Low kVp forward projection + physics → ws.sino_low
    # ═══════════════════════════════════════════════════════════════════════
    _eict_dual_forward_pass!(ws, ws.sino_low, phantom, geom, mats,
        ws.energies_low, ws.weights_low, ws.weights_norm_low,
        ws.μ_table_low, ws.config_low,
        ws.flat_filter_projection_low, ws.bowtie_projection_low,
        ws.bhc_coeffs_gpu_low, ws.bhc_low,
        sim_opts)

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2: High kVp forward projection + physics → ws.sino_high
    # ═══════════════════════════════════════════════════════════════════════
    _eict_dual_forward_pass!(ws, ws.sino_high, phantom, geom, mats,
        ws.energies_high, ws.weights_high, ws.weights_norm_high,
        ws.μ_table_high, ws.config_high,
        ws.flat_filter_projection_high, ws.bowtie_projection_high,
        ws.bhc_coeffs_gpu_high, ws.bhc_high,
        sim_opts)

    # ═══════════════════════════════════════════════════════════════════════
    # Save ideal sinograms (both kVps) to CPU
    # ═══════════════════════════════════════════════════════════════════════
    copyto!(ws.sino_ideal_out_low, ws.sino_low)
    copyto!(ws.sino_ideal_out_high, ws.sino_high)

    # ═══════════════════════════════════════════════════════════════════════
    # Apply quantum noise to BOTH sinograms (in-place, sequential)
    # ═══════════════════════════════════════════════════════════════════════
    if sim_opts.use_noise
        # Noise for low kVp
        I0_low = T(compute_detector_I0(geom, CTProtocol(
            mA = protocol.mA_low > 0 ? protocol.mA_low : protocol.mA,
            kVp = protocol.kVp_low, views = protocol.views,
            rotation_time = protocol.rotation_time,
            flux_density = protocol.flux_density)))

        randn!(ws.rng, ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)
        let sino = ws.sino_low, rg = ws.noise_rand_gpu, I0v = I0_low
            AK.foreachindex(sino) do idx
                λ = I0v * exp(-sino[idx])
                λ_noisy = λ + sqrt(max(λ, T(1))) * rg[idx]
                λ_noisy = max(λ_noisy, T(1))
                sino[idx] = -log(λ_noisy / I0v)
            end
        end

        # Noise for high kVp
        I0_high = T(compute_detector_I0(geom, CTProtocol(
            mA = protocol.mA, kVp = protocol.kVp, views = protocol.views,
            rotation_time = protocol.rotation_time,
            flux_density = protocol.flux_density)))

        randn!(ws.rng, ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)
        let sino = ws.sino_high, rg = ws.noise_rand_gpu, I0v = I0_high
            AK.foreachindex(sino) do idx
                λ = I0v * exp(-sino[idx])
                λ_noisy = λ + sqrt(max(λ, T(1))) * rg[idx]
                λ_noisy = max(λ_noisy, T(1))
                sino[idx] = -log(λ_noisy / I0v)
            end
        end
    end

    # Save noisy sinograms (both kVps) to CPU
    copyto!(ws.sino_noisy_out_low, ws.sino_low)
    copyto!(ws.sino_noisy_out_high, ws.sino_high)

    # ═══════════════════════════════════════════════════════════════════════
    # Material decomposition (in-place into workspace buffers)
    # ═══════════════════════════════════════════════════════════════════════
    de_sino = DualEnergySinogram(ws.sino_low, ws.sino_high;
        low_kvp = Int(protocol.kVp_low),
        high_kvp = Int(protocol.kVp))

    spectral_decompose!(ws.material1, ws.material2, ws.sino_low, ws.sino_high,
                        ws.inv_a11, ws.inv_a12, ws.inv_a21, ws.inv_a22)
    mat_map = MaterialMap(ws.material1, ws.material2;
        material1_name=ws.basis[1], material2_name=ws.basis[2],
        domain=:projection)

    return (sino_ideal_low=ws.sino_ideal_out_low, sino_ideal_high=ws.sino_ideal_out_high,
            sino_noisy_low=ws.sino_noisy_out_low, sino_noisy_high=ws.sino_noisy_out_high,
            de_sino=de_sino, mat_map=mat_map)
end

"""
    _eict_dual_forward_pass!(ws, target_sino, phantom, geom, mats,
        energies, weights, weights_norm, μ_table, config,
        flat_filter_proj, bowtie_proj, bhc_coeffs_gpu, bhc_effect,
        sim_opts)

Run one forward projection pass for dual-kVp simulation.
Uses shared scratch buffers from the dual workspace (μ_volume, sino_mono, I_transmitted).
Writes result into `target_sino`.
"""
function _eict_dual_forward_pass!(
    ws::EICTDualWorkspace{T}, target_sino, phantom, geom, mats,
    energies, weights, weights_norm, μ_table, config,
    flat_filter_proj, bowtie_proj, bhc_coeffs_gpu, bhc_effect,
    sim_opts
) where {T}
    # Forward projection (Beer-Lambert polychromatic)
    fill!(target_sino, zero(T))
    _forward_project_poly!(target_sino, phantom.mask, geom, energies, weights, mats;
                            ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
                            ws_I_transmitted=ws.I_transmitted,
                            ws_weights_norm=weights_norm,
                            ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
                            ws_μ_table=μ_table,
                            ws_source_positions=ws.geom_source_positions,
                            ws_detector_centers=ws.geom_detector_centers,
                            ws_detector_u=ws.geom_detector_u,
                            ws_detector_v=ws.geom_detector_v,
                            volume_extent=phantom.extent)

    if ws.has_signal_chain
        # CatSim signal chain
        heel_effect = ws.heel_effect
        das_model = ws.das_model

        # Apply physics pipeline (no noise)
        _apply_physics_no_noise!(target_sino, geom, config;
            ws_output=ws.physics_output,
            ws_scatter_kernel=ws.scatter_kernel,
            ws_scatter_correct_kernel=ws.scatter_correct_kernel,
            ws_crosstalk_kernel=ws.crosstalk_kernel,
            ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
            ws_focal_spot_kernel=ws.focal_spot_kernel,
            ws_flat_filter_projection=flat_filter_proj,
            ws_bowtie_projection=bowtie_proj,
            ws_lag_output=ws.physics_output,
            ws_lag_intensity=ws.lag_intensity,
            ws_lag_coeffs=ws.lag_coeffs)

        # Convert to intensity domain
        eps = T(1e-10)
        AK.foreachindex(target_sino) do idx
            target_sino[idx] = exp(-clamp(target_sino[idx], T(-1), T(15)))
        end

        # Heel effect
        if heel_effect !== nothing
            apply_heel_effect!(target_sino, heel_effect, geom)
        end

        # DAS model
        if das_model !== nothing
            apply_das_model!(target_sino, das_model; seed=config.noise_seed)
        end

        # Air scan (workspace buffer)
        fill!(ws.air_scan, one(T))
        if heel_effect !== nothing
            apply_heel_effect!(ws.air_scan, heel_effect, geom)
        end
        if das_model !== nothing
            gain = T(das_model.gain)
            let air = ws.air_scan
                AK.foreachindex(air) do idx
                    air[idx] *= gain
                end
            end
        end

        # Calibration
        let sino = target_sino, air = ws.air_scan
            AK.foreachindex(sino) do idx
                air_val = max(air[idx], eps)
                sino[idx] = sino[idx] / air_val
            end
        end

        # Low signal correction
        low_signal_correction_gpu!(target_sino)

        # Log transform
        AK.foreachindex(target_sino) do idx
            target_sino[idx] = -log(max(target_sino[idx], eps))
        end

        # BHC
        if bhc_effect !== nothing
            apply_bhc!(target_sino, bhc_effect; ws_coeffs_gpu=bhc_coeffs_gpu)
        end
    else
        # Standard path (no signal chain)
        if config !== nothing
            apply_physics_effects!(target_sino, geom, config;
                ws_output=ws.physics_output,
                ws_scatter_kernel=ws.scatter_kernel,
                ws_scatter_correct_kernel=ws.scatter_correct_kernel,
                ws_crosstalk_kernel=ws.crosstalk_kernel,
                ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
                ws_focal_spot_kernel=ws.focal_spot_kernel,
                ws_flat_filter_projection=flat_filter_proj,
                ws_bowtie_projection=bowtie_proj,
                ws_lag_output=ws.physics_output,
                ws_lag_intensity=ws.lag_intensity,
                ws_lag_coeffs=ws.lag_coeffs,
                ws_bhc_coeffs_gpu=bhc_coeffs_gpu)
        end
    end
end

# =============================================================================
# Mode 5: Axial PCCT (Photon-Counting CT)
# =============================================================================

function _simulate_axial_pcct(phantom, scanner, protocol, sim_opts, recon_opts;
                              materials::Union{Nothing, Vector} = nothing)
    T = Float32
    # Convert mask to GPU before creating workspace so buffers match GPU backend
    mask_gpu = _to_gpu(phantom.mask)
    # Create a phantom wrapper with the GPU mask for consistent backend
    gpu_phantom = Phantom(mask_gpu, phantom.materials, phantom.voxel_size,
                          phantom.origin, phantom.extent)
    ws = create_workspace(scanner, protocol, sim_opts, recon_opts, gpu_phantom;
                          materials=materials)

    # Run zero-alloc PCCT pipeline
    result = simulate!(ws, gpu_phantom, scanner, protocol, sim_opts, recon_opts;
                        materials=materials)
    pcct_sino = result.pcct_sino
    mat_map = result.mat_map

    # --- Post-processing (allocates — outside zero-alloc scope) ---
    geom = ws.geom

    # VMI synthesis + reconstruction (unified with dual-kVp path)
    pcct_vmi_dict = Dict{Float64, AbstractArray{T, 3}}()
    if !isempty(recon_opts.vmi_energies) && !isnothing(mat_map)
        for E in recon_opts.vmi_energies
            vmi_sino = virtual_monoenergetic(mat_map, E; ws_output=ws.vmi_sino)
            vmi_vol = _run_reconstruction(vmi_sino, geom, recon_opts)
            pcct_vmi_dict[E] = vmi_vol
        end
    end

    # Main reconstruction
    recon_vol = _run_reconstruction(ws.sino_noisy_out, geom, recon_opts)
    recons = Pair{Symbol, AbstractArray{T, 3}}[recon_opts.algorithm => recon_vol]
    vmi_dict = Dict{Float64, AbstractArray{T, 3}}()

    return SimulationResult(
        ws.sino_ideal_out, ws.sino_noisy_out, recons, geom, ws.config,
        nothing, mat_map, vmi_dict,
        pcct_sino, nothing, pcct_vmi_dict
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
    else
        error("Unknown reconstruction algorithm: $alg. " *
              "Supported: :fdk, :sirt, :cgls, :tv_sirt, :tv_cgls, :asir, :mbir")
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
    build_physics_config(scanner::Scanner, sim_opts::SimOptions, energies::Vector{Float64}, weights::Vector{Float64}; phantom=nothing) -> PhysicsConfig

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

# Keyword Arguments
- `phantom::Union{Nothing, Phantom} = nothing`: Phantom for automatic scatter scaling.
  If provided and scatter is enabled, the phantom diameter is estimated from the mask
  and used to scale the scatter coefficient appropriately.

# Returns
A `PhysicsConfig` ready for `forward_project()`.
"""
function build_physics_config(
    scanner::Scanner,
    sim_opts::SimOptions,
    energies::Vector{Float64},
    weights::Vector{Float64};
    phantom::Union{Nothing, Phantom} = nothing
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

    # Scatter: use geometry-aware model scaled for this scanner and phantom size
    # If phantom is provided, estimate diameter from mask for size-aware scatter scaling
    phantom_diameter_cm = if phantom !== nothing && (sim_opts.use_scatter || sim_opts.use_scatter_correction)
        # Convert voxel_size from cm to mm for estimate_phantom_diameter_cm
        voxel_size_mm = phantom.voxel_size .* 10.0
        estimate_phantom_diameter_cm(phantom.mask, voxel_size_mm)
    else
        nothing  # Use default reference diameter (30 cm)
    end

    if sim_opts.use_scatter
        kwargs[:scatter] = geometry_aware_scatter_model(scanner; phantom_diameter_cm=phantom_diameter_cm)
    end

    # Scatter correction: use geometry-aware correction scaled for this scanner
    if sim_opts.use_scatter_correction
        kwargs[:scatter_correction] = geometry_aware_scatter_correction(scanner; phantom_diameter_cm=phantom_diameter_cm)
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

    # BHC: calibrate polynomial from actual spectrum (not hardcoded defaults)
    # The calibration generates a water-based BHC that properly maps polychromatic
    # line integrals to monochromatic-equivalent values at the reference energy.
    if sim_opts.use_bhc
        ref_energy = sum(energies .* weights) / sum(weights)
        kwargs[:bhc] = calibrate_bhc(energies, weights;
                                      order=5, reference_energy_keV=ref_energy)
    end

    return default_physics_config(; kwargs...)
end

export build_physics_config

# =============================================================================
# reconstruct!() — Zero-allocation FDK reconstruction hot path
# =============================================================================

export reconstruct!

"""
    reconstruct!(ws::FDKReconWorkspace, sinogram, geom, volume_size; filter=StandardFilter(), cutoff=1.0)

Zero-allocation FDK reconstruction using pre-allocated workspace buffers.

Create the workspace with `create_fdk_recon_workspace(sinogram, geom, volume_size)`.
"""
function reconstruct!(
    ws::FDKReconWorkspace{T},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    filter::FilterType = StandardFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    # Step 1: Copy sinogram into filtering scratch buffer
    copyto!(ws.filtered, sinogram)

    # Step 2: Filter in-place (cosine weighting + ramp convolution)
    # Uses pre-allocated convolution scratch and filter kernel
    filter_sinogram!(ws.filtered, geom; filter=filter, cutoff=cutoff,
                     ws_conv_scratch=ws.conv_scratch,
                     ws_filter_kernel=ws.filter_kernel)

    # Step 3: Backproject into pre-allocated volume
    fill!(ws.volume, zero(T))
    backproject!(ws.volume, ws.filtered, geom;
                 weighted=true,
                 ws_source_positions=ws.bp_source_positions,
                 ws_detector_centers=ws.bp_detector_centers,
                 ws_detector_u=ws.bp_detector_u,
                 ws_detector_v=ws.bp_detector_v)

    return ws.volume
end

# =============================================================================
# reconstruct!() — Zero-allocation Hybrid IR reconstruction hot path
# =============================================================================

"""
    _copy_subset_into_buffer!(buf, full, angle_indices, n_sub)

Copy subset of angle data from `full` into pre-allocated `buf`.
`buf[:,:,1:n_sub] = full[:,:,angle_indices]` — zero-allocation via views.
"""
function _copy_subset_into_buffer!(
    buf::AbstractArray{T, 3},
    full::AbstractArray{T, 3},
    angle_indices::Vector{Int},
    n_sub::Int
) where T
    for (i, aidx) in enumerate(angle_indices)
        copyto!(view(buf, :, :, i), view(full, :, :, aidx))
    end
end

"""
    reconstruct!(ws::HIRReconWorkspace, sinogram, geom, volume_size; filter=StandardFilter(), cutoff=1.0)

Zero-allocation Hybrid IR reconstruction using pre-allocated workspace buffers.

Implements TRUE Hybrid IR = FDK initialization + PWLS refinement with Huber regularization.
When `n_subsets > 0`, uses Ordered Subsets PWLS (OS-PWLS) for ~10-25x speedup.
All iteration buffers are pre-allocated in the workspace.

Create the workspace with `create_hir_recon_workspace(sinogram, geom, volume_size; strength=3)`.
"""
function reconstruct!(
    ws::HIRReconWorkspace{T},
    sinogram::AbstractArray{T, 3},
    geom::CTGeometry,
    volume_size::NTuple{3, Int};
    filter::FilterType = StandardFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat

    # ─── Step 1: FDK initialization (same as FDKReconWorkspace) ───
    copyto!(ws.filtered, sinogram)
    filter_sinogram!(ws.filtered, geom; filter=filter, cutoff=cutoff,
                     ws_conv_scratch=ws.conv_scratch,
                     ws_filter_kernel=ws.filter_kernel)
    fill!(ws.volume, zero(T))
    backproject!(ws.volume, ws.filtered, geom;
                 weighted=true,
                 ws_source_positions=ws.geom_source_positions,
                 ws_detector_centers=ws.geom_detector_centers,
                 ws_detector_u=ws.geom_detector_u,
                 ws_detector_v=ws.geom_detector_v)

    # ─── Step 2: PWLS refinement with Huber regularization ───
    params = ws.params
    λ = T(params.lambda)
    λ_relax = T(params.relaxation)
    δ = T(params.huber_delta)
    backend = AK.get_backend(ws.volume)

    if params.n_subsets > 0
        # ═══════════════════════════════════════════════════════════════
        # OS-PWLS path: Ordered Subsets for ~10-25x speedup
        # ═══════════════════════════════════════════════════════════════
        nepochs = params.nepochs
        n_subsets = params.n_subsets
        subset_scale = T(n_subsets)

        # Initialize statistical weights once from sinogram: w ≈ exp(-y)
        let sw = ws.stat_weights, ε = T(1e-6)
            AK.foreachindex(sw, backend) do idx
                y_val = sinogram[idx]
                y_clipped = clamp(y_val, T(-10), T(10))
                sw[idx] = exp(-y_clipped) + ε
            end
        end

        for epoch in 1:nepochs
            # Compute Huber gradient ONCE per epoch (not per sub-iteration)
            compute_huber_gradient!(ws.reg_grad, ws.volume, δ)

            for (s, angle_indices) in enumerate(ws.subsets)
                n_sub = length(angle_indices)
                geom_s = ws.subset_geometries[s]

                # Copy subset data into pre-allocated buffers
                _copy_subset_into_buffer!(ws.subset_sino_buf, sinogram, angle_indices, n_sub)
                _copy_subset_into_buffer!(ws.subset_W_proj_buf, ws.W_proj, angle_indices, n_sub)
                _copy_subset_into_buffer!(ws.subset_stat_weights_buf, ws.stat_weights, angle_indices, n_sub)

                # Forward project with subset geometry → subset_Ax_buf
                ax_view = view(ws.subset_Ax_buf, :, :, 1:n_sub)
                fill!(ax_view, zero(T))
                siddon_forward_project!(ax_view, ws.volume, geom_s;
                    ws_source_positions=ws.subset_geom_source_positions[s],
                    ws_detector_centers=ws.subset_geom_detector_centers[s],
                    ws_detector_u=ws.subset_geom_detector_u[s],
                    ws_detector_v=ws.subset_geom_detector_v[s])

                # Compute weighted residual in-place:
                # Ax_s = W_s ⊙ stat_w_s ⊙ (sino_s - Ax_s)
                let ax = ax_view,
                    sino_s = view(ws.subset_sino_buf, :, :, 1:n_sub),
                    wp_s = view(ws.subset_W_proj_buf, :, :, 1:n_sub),
                    sw_s = view(ws.subset_stat_weights_buf, :, :, 1:n_sub)
                    AK.foreachindex(ax, backend) do idx
                        residual = sino_s[idx] - ax[idx]
                        ax[idx] = wp_s[idx] * sw_s[idx] * residual
                    end
                end

                # Backproject weighted residual → correction
                fill!(ws.correction, zero(T))
                backproject!(ws.correction, ax_view, geom_s;
                             weighted=false,
                             ws_source_positions=ws.subset_geom_source_positions[s],
                             ws_detector_centers=ws.subset_geom_detector_centers[s],
                             ws_detector_u=ws.subset_geom_detector_u[s],
                             ws_detector_v=ws.subset_geom_detector_v[s])

                # SIRT-style update with subset scaling:
                # x += λ_relax * V_inv * (n_subsets * correction) - λ * V_inv * reg_grad
                let vol = ws.volume, vinv = ws.V_inv, corr = ws.correction,
                    rg = ws.reg_grad, ss = subset_scale
                    AK.foreachindex(vol, backend) do idx
                        data_update = λ_relax * vinv[idx] * ss * corr[idx]
                        reg_update = λ * vinv[idx] * rg[idx]
                        vol[idx] += data_update - reg_update
                    end
                end
            end
        end
    else
        # ═══════════════════════════════════════════════════════════════
        # Legacy full-data PWLS path (n_subsets == 0)
        # ═══════════════════════════════════════════════════════════════
        niter = params.niter

        # Initialize simple statistical weights from sinogram: w ≈ exp(-y)
        let sw = ws.stat_weights, ε = T(1e-6)
            AK.foreachindex(sw, backend) do idx
                y_val = sinogram[idx]
                y_clipped = clamp(y_val, T(-10), T(10))
                sw[idx] = exp(-y_clipped) + ε
            end
        end

        for iter in 1:niter
            # Update statistical weights periodically (Poisson noise model)
            if iter == 1 || iter % 10 == 0
                fill!(ws.Ax, zero(T))
                siddon_forward_project!(ws.Ax, ws.volume, geom;
                    ws_source_positions=ws.geom_source_positions,
                    ws_detector_centers=ws.geom_detector_centers,
                    ws_detector_u=ws.geom_detector_u,
                    ws_detector_v=ws.geom_detector_v)

                let sw = ws.stat_weights, ax = ws.Ax
                    AK.foreachindex(sw, backend) do idx
                        ax_val = ax[idx]
                        ax_clipped = clamp(ax_val, T(-10), T(10))
                        sw[idx] = exp(-ax_clipped)
                    end
                end
                max_w = maximum(ws.stat_weights)
                if max_w > zero(T)
                    let sw = ws.stat_weights, mw = max_w
                        AK.foreachindex(sw, backend) do idx
                            sw[idx] = sw[idx] / mw
                        end
                    end
                end
            end

            # Compute Huber regularization gradient
            compute_huber_gradient!(ws.reg_grad, ws.volume, δ)

            # Forward project: Ax = A * x
            fill!(ws.Ax, zero(T))
            siddon_forward_project!(ws.Ax, ws.volume, geom;
                ws_source_positions=ws.geom_source_positions,
                ws_detector_centers=ws.geom_detector_centers,
                ws_detector_u=ws.geom_detector_u,
                ws_detector_v=ws.geom_detector_v)

            # Compute combined-weighted residual in-place: Ax = W_proj ⊙ stat_weights ⊙ (y - Ax)
            let ax = ws.Ax, wp = ws.W_proj, sw = ws.stat_weights
                AK.foreachindex(ax, backend) do idx
                    residual = sinogram[idx] - ax[idx]
                    ax[idx] = wp[idx] * sw[idx] * residual
                end
            end

            # Backproject weighted residual into ws.correction (unweighted for SIRT)
            fill!(ws.correction, zero(T))
            backproject!(ws.correction, ws.Ax, geom;
                         weighted=false,
                         ws_source_positions=ws.geom_source_positions,
                         ws_detector_centers=ws.geom_detector_centers,
                         ws_detector_u=ws.geom_detector_u,
                         ws_detector_v=ws.geom_detector_v)

            # Apply SIRT-style update with regularization:
            # x = x + λ_relax * V_inv * correction - λ_reg * V_inv * reg_grad
            let vol = ws.volume, vinv = ws.V_inv, corr = ws.correction, rg = ws.reg_grad
                AK.foreachindex(vol, backend) do idx
                    data_update = λ_relax * vinv[idx] * corr[idx]
                    reg_update = λ * vinv[idx] * rg[idx]
                    vol[idx] += data_update - reg_update
                end
            end
        end
    end

    return ws.volume
end
