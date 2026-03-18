"""
    Simulation/Driver.jl

High-level driver for running end-to-end CT simulations.
"""

export simulate!, add_system_noise_floor!

# =============================================================================
# Materials Resolution
# =============================================================================

"""
    _resolve_materials(phantom, materials_kwarg) -> Vector{XA.Material}

Resolve materials with priority: (1) explicit kwarg, (2) phantom.materials, (3) fallback.
"""
function _resolve_materials(phantom, materials_kwarg::Union{Nothing,Vector})
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
        R = compute_drm(detector, kVp)
        _compute_degraded_I0(detector, energies, weights, η, thresholds, kVp, I0, flux_rate; R=R)
    else
        # No effects OR effects+corrections: use theoretical I0
        η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)
        R = compute_drm(detector, kVp)
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
# PCCT Tube Physics (applied to combined sinogram at binned resolution)
# =============================================================================

"""
    _apply_pcct_tube_physics!(sinogram, geom, config; ws_scratch=nothing)

Apply tube/geometry effects to the combined PCCT sinogram.

These are tube-side effects that are skipped by the PCCT per-bin pipeline but
affect the measured signal: heel effect, scatter, and focal spot blur.
Applied to the **combined** (single) sinogram at binned resolution.

The effects are a subset of the physics pipeline — only those that are
tube/geometry effects (not detector-specific like noise, crosstalk, lag).
"""
function _apply_pcct_tube_physics!(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    config::PhysicsConfig;
    ws_scratch::Union{Nothing, AbstractArray{T,3}}=nothing,
    ws_scatter_temp=nothing,
    ws_scatter_kernel_1d=nothing,
    ws_scatter_correct_kernel_1d=nothing
) where T
    _scratch = ws_scratch

    # 1. Heel effect (intensity domain)
    if config.heel_effect !== nothing
        eps = T(1e-10)
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = exp(-sinogram[idx])
        end
        apply_heel_effect!(sinogram, config.heel_effect, geom)
        AK.foreachindex(sinogram) do idx
            sinogram[idx] = -log(max(sinogram[idx], eps))
        end
    end

    # 2. Scatter add + correct (sinogram domain)
    if config.scatter !== nothing && _scratch !== nothing
        add_scatter!(sinogram, config.scatter; ws_output=_scratch,
                     ws_scatter_temp=ws_scatter_temp, ws_kernel_1d=ws_scatter_kernel_1d)
    end
    if config.scatter_correction !== nothing && _scratch !== nothing
        correct_scatter!(sinogram, config.scatter_correction; ws_output=_scratch,
                         ws_scatter_temp=ws_scatter_temp, ws_kernel_1d=ws_scatter_correct_kernel_1d)
    end

    # 3. Focal spot blur (sinogram domain)
    if config.focal_spot !== nothing && _scratch !== nothing
        apply_focal_spot_blur!(sinogram, config.focal_spot, geom; ws_output=_scratch)
    end

    return sinogram
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
is pre-computed in the workspace by `create_workspace()`. Reconstruction is handled
separately via `reconstruct!()`.

# Returns
Named tuple with `pcct_sino` and `mat_map` fields.
"""
function simulate!(
    ws::PCCTWorkspace{T},
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions=SimOptions(),
    recon_opts::ReconOptions=ReconOptions();
    materials::Union{Nothing,Vector}=nothing
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

    # Forward projection with ALL workspace buffers (including native-res path)
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
        volume_extent=phantom.extent,
        # Native-resolution forward projection path
        native_geom=ws.native_geom,
        ws_native_bins=ws.native_bins,
        ws_native_sino_buf=ws.native_sino_buf,
        ws_native_scratch=ws.native_scratch,
        ws_native_total_counts=ws.native_total_counts,
        ws_native_source_positions=ws.native_geom_source_positions,
        ws_native_detector_centers=ws.native_geom_detector_centers,
        ws_native_detector_u=ws.native_geom_detector_u,
        ws_native_detector_v=ws.native_geom_detector_v
    )

    # Combine ideal (workspace buffer + pre-computed I0_bins)
    sino_ideal_gpu = _combine_pcct_bins(pcct_sino, pcct_detector, energies, weights, kVp;
        apply_detector_effects=use_detector_fx,
        apply_corrections=use_corrections,
        output=ws.combined,
        ws_I0_bins=ws.I0_bins)

    # Tube physics on ideal combined sinogram (scatter, focal spot, heel)
    _apply_pcct_tube_physics!(sino_ideal_gpu, geom, config;
        ws_scratch=ws.tube_physics_scratch)

    # BHC on ideal
    if config.bhc !== nothing
        apply_bhc!(sino_ideal_gpu, config.bhc; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
    end

    # Save ideal to CPU workspace buffer
    copyto!(ws.sino_ideal_out, sino_ideal_gpu)

    # Noise (in-place on pcct_sino.bins — operates at binned resolution)
    if sim_opts.use_noise
        I0_physics = compute_detector_I0(geom, protocol, sum(ws.weights))
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

    # Tube physics on noisy combined sinogram
    _apply_pcct_tube_physics!(sino_noisy_gpu, geom, config;
        ws_scratch=ws.tube_physics_scratch)

    # BHC on noisy
    if config.bhc !== nothing
        apply_bhc!(sino_noisy_gpu, config.bhc; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
    end

    # Save noisy to CPU workspace buffer (uncorrected, for inspection)
    copyto!(ws.sino_noisy_out, sino_noisy_gpu)

    # Combine PCCT bins into pseudo-dual-energy sinograms (GPU)
    # split_bin = floor(n_bins/2): bin 1 for 2-bin, bins 1-2 for 4-bin
    combine_pcct_bins!(ws.vmi_sino_low, ws.vmi_sino_high, pcct_sino.bins;
        split_bin=length(pcct_sino.bins) ÷ 2)

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
    sim_opts::SimOptions=SimOptions(),
    recon_opts::ReconOptions=ReconOptions();
    materials::Union{Nothing,Vector}=nothing
) where {T}
    geom = ws.geom
    energies = ws.energies
    mats = ws.mats
    config = ws.config

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Polychromatic forward projection (Beer-Lambert)
    # ═══════════════════════════════════════════════════════════════════════
    fill!(ws.sinogram, zero(T))
    _forward_project_poly!(ws.sinogram, phantom.mask, geom, energies, ws.weights, mats;
        ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
        ws_I_transmitted=ws.I_transmitted,
        ws_weights_norm=ws.weights_norm,
        ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
        ws_μ_table=ws.μ_table,
        ws_μ_table_gpu=ws.μ_table_gpu,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom.extent,
        ws_η=ws.η_vec,
        ws_bowtie_spectral=ws.bowtie_spectral,
        ws_wη_gpu=ws.wη_gpu)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Signal chain (always active)
    # ═══════════════════════════════════════════════════════════════════════

    heel_effect = ws.heel_effect
    bhc_eff = ws.bhc

    # Apply physics pipeline (sinogram domain, no noise)
    _apply_physics_no_noise!(ws.sinogram, geom, config;
        ws_output=ws.physics_output,
        ws_scatter_kernel=ws.scatter_kernel,
        ws_scatter_correct_kernel=ws.scatter_correct_kernel,
        ws_scatter_temp=ws.scatter_temp,
        ws_scatter_kernel_1d=ws.scatter_kernel_1d,
        ws_scatter_correct_kernel_1d=ws.scatter_correct_kernel_1d,
        ws_crosstalk_kernel=ws.crosstalk_kernel,
        ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
        ws_focal_spot_kernel=ws.focal_spot_kernel,
        ws_flat_filter_projection=ws.flat_filter_projection,
        ws_bowtie_projection=nothing,
        ws_lag_output=ws.physics_output,
        ws_lag_intensity=ws.lag_intensity,
        ws_lag_coeffs=ws.lag_coeffs)

    # Convert to intensity domain
    eps = T(1e-10)
    let sino = ws.sinogram
        AK.foreachindex(sino) do idx
            sino[idx] = exp(-clamp(sino[idx], T(-1), T(15)))
        end
    end

    # Apply heel effect to phantom intensity
    if heel_effect !== nothing
        apply_heel_effect!(ws.sinogram, heel_effect, geom)
    end

    # Create noise-free air scan (workspace buffer)
    # Clinical scanners calibrate with bowtie in beam, so air scan includes bowtie
    # Air scan uses spectral bowtie air reference: I₀(col,row) = Σ w(E) × T_bt(E,col,row) × η(E)
    fill!(ws.air_scan, one(T))
    if ws.bowtie_air_reference !== nothing
        let air = ws.air_scan, ref = ws.bowtie_air_reference, nc = size(air, 1), nr = size(air, 2)
            AK.foreachindex(air) do idx
                idx_0 = Int32(idx - 1)
                col = (idx_0 % Int32(nc)) + Int32(1)
                row = ((idx_0 ÷ Int32(nc)) % Int32(nr)) + Int32(1)
                ref_idx = col + (row - 1) * nc
                air[idx] *= ref[ref_idx]
            end
        end
    end
    if heel_effect !== nothing
        apply_heel_effect!(ws.air_scan, heel_effect, geom)
    end

    # Calibration (prep = phantom / air)
    let sino = ws.sinogram, air = ws.air_scan
        AK.foreachindex(sino) do idx
            air_val = max(air[idx], eps)
            sino[idx] = sino[idx] / air_val
        end
    end

    # Low signal correction
    low_signal_correction_gpu!(ws.sinogram)

    # Log transform
    let sino = ws.sinogram
        AK.foreachindex(sino) do idx
            sino[idx] = -log(max(sino[idx], eps))
        end
    end

    # Beam hardening correction
    if bhc_eff !== nothing
        apply_bhc!(ws.sinogram, bhc_eff; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Save ideal sinogram to CPU
    # ═══════════════════════════════════════════════════════════════════════
    copyto!(ws.sino_ideal_out, ws.sinogram)

    # ═══════════════════════════════════════════════════════════════════════
    # Apply quantum noise (in-place using workspace buffers)
    # ═══════════════════════════════════════════════════════════════════════
    if sim_opts.use_noise
        I0_raw = compute_detector_I0(geom, protocol, sum(ws.weights))
        # Scale by spectrum-weighted average efficiency: η_eff = Σ wₑ × η(E)
        η_eff = sum(ws.weights_norm[i] * ws.η_vec[i] for i in 1:length(ws.η_vec))
        I0_T = T(I0_raw * η_eff)

        # Quantum noise random numbers
        randn!(ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)

        # Electronic noise: convert from electron domain to photon-equivalent
        # σ_e_photon = σ_e_electrons / (mean_E_keV × gain_e_per_keV)
        mean_E_keV = sum(ws.weights_norm[i] * ws.energies[i] for i in 1:length(ws.energies))
        σ_e_photon = T(scanner.electronic_noise / (mean_E_keV * scanner.detection_gain))

        if σ_e_photon > T(0)
            randn!(ws.enoise_rand_cpu)
            copyto!(ws.enoise_rand_gpu, ws.enoise_rand_cpu)

            let sino = ws.sinogram, rg = ws.noise_rand_gpu, eg = ws.enoise_rand_gpu,
                    I0v = I0_T, σ_e = σ_e_photon
                AK.foreachindex(sino) do idx
                    λ = I0v * exp(-sino[idx])
                    λ_noisy = λ + sqrt(max(λ, T(1))) * rg[idx]  # Quantum (Poisson)
                    λ_noisy += σ_e * eg[idx]                      # Electronic (Gaussian)
                    λ_noisy = max(λ_noisy, T(1))
                    sino[idx] = -log(λ_noisy / I0v)
                end
            end
        else
            # No electronic noise (e.g., PCCT with thresholding) — quantum only
            let sino = ws.sinogram, rg = ws.noise_rand_gpu, I0v = I0_T
                AK.foreachindex(sino) do idx
                    λ = I0v * exp(-sino[idx])
                    λ_noisy = λ + sqrt(max(λ, T(1))) * rg[idx]
                    λ_noisy = max(λ_noisy, T(1))
                    sino[idx] = -log(λ_noisy / I0v)
                end
            end
        end
    end

    # Save noisy sinogram to CPU
    copyto!(ws.sino_noisy_out, ws.sinogram)

    return (sino_ideal=ws.sino_ideal_out, sino_noisy=ws.sino_noisy_out)
end

# =============================================================================
# System Noise Floor (dose-independent)
# =============================================================================

"""
    add_system_noise_floor!(vol, sigma_hu; seed=nothing)

Add dose-independent Gaussian noise to a reconstruction volume (in-place).

Models the irreducible scanner noise floor from imperfect scatter correction,
electronic noise, calibration residuals, etc. Added in quadrature with existing
noise: σ_total = √(σ_quantum² + σ_floor²).

# Arguments
- `vol::AbstractArray{T}`: Reconstruction volume in HU
- `sigma_hu::Real`: Noise floor standard deviation in HU

# Keyword Arguments
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility
"""
function add_system_noise_floor!(vol::AbstractArray{T}, sigma_hu::Real; seed::Union{Int,Nothing}=nothing) where T
    sigma_hu <= 0 && return vol
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed + 7919)
    vol .+= T(sigma_hu) .* randn(rng, T, size(vol))
    return vol
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
        return data[:, 1], data[:, 2] # energies, weights
    end

    # 2. Priority: Auto-Lookup via kVp
    # This confirms the simulator can find the library file
    return load_spectrum(Int(protocol.kVp))
end

export get_spectrum, resolve_spectrum

"""
    resolve_spectrum(sim_opts, protocol; scanner=nothing) -> (energies, weights)

Determine the energy spectrum for simulation.

Loads a raw IPEM Anode spectrum and applies Beer-Lambert filtering
(scanner flat filter + protocol additional_filters) plus inverse-square-law
distance scaling. Full resolution (~160-280 bins), no downsampling.

Pass `scanner` so the pipeline can read `flat_filter_material`,
`flat_filter_thickness`, and `source_to_detector` from the hardware spec.
"""
function resolve_spectrum(sim_opts::SimOptions, protocol::CTProtocol; scanner=nothing)
    # Physics-based: raw IPEM spectrum + Beer-Lambert filtering
    e, w = load_spectrum_unfiltered(Int(protocol.kVp); anode_angle=protocol.anode_angle)

    # Build filter list: scanner's built-in flat filter + protocol extras
    filters = Tuple{String, Float64}[]
    if scanner !== nothing && scanner.flat_filter_thickness > 0
        push!(filters, (String(scanner.flat_filter_material), Float64(scanner.flat_filter_thickness)))
    end
    append!(filters, protocol.additional_filters)

    # Apply Beer-Lambert filtering + inverse-square-law distance scaling
    sdd_mm = scanner !== nothing ? Float64(scanner.source_to_detector) : 750.0
    e, w = filter_spectrum(e, w; filters=filters, sdd_mm=sdd_mm)

    return e, w
end

"""
    build_physics_config(scanner::Scanner, sim_opts::SimOptions, energies::Vector{Float64}, weights::Vector{Float64}; phantom=nothing) -> PhysicsConfig

Build a complete PhysicsConfig from Scanner hardware fields and SimOptions toggles.

For effects with Scanner fields (focal_spot, flat_filter, fill_factor, detector_efficiency,
heel_effect), the Scanner hardware parameters are used to construct the effect structs.
For effects without Scanner fields (scatter, scatter_correction, crosstalk, optical_crosstalk,
bowtie, lag, bhc), factory function defaults are used.

Noise is ALWAYS `nothing` in the returned PhysicsConfig — noise is applied externally
via `compute_detector_I0()` + `add_quantum_noise!()` when `sim_opts.use_noise == true`.

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
    phantom::Union{Nothing,Phantom}=nothing
)
    kwargs = Dict{Symbol,Any}()

    # --- Common settings ---
    kwargs[:energy_keV] = sum(energies .* weights) / sum(weights)
    kwargs[:noise_seed] = sim_opts.seed
    kwargs[:noise] = nothing  # Noise is handled externally via compute_detector_I0 + add_quantum_noise!

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

    # Bowtie filter: resolve from Scanner's bowtie_filter symbol
    if sim_opts.use_bowtie_filter
        kwargs[:bowtie_filter] = resolve_bowtie_filter(scanner.bowtie_filter)
    end

    # Detector efficiency: use Scanner's material and depth
    if sim_opts.use_detector_efficiency
        depth = scanner.detector_depth
        material = scanner.detector_material
        de_mode = sim_opts.detector_efficiency_mode   # :auto, :mc_lut, :beer_lambert
        if material == :lumex || material == :Lumex || material == :LUMEX
            # GE Gemstone Ce:(Tb,Lu)₃Al₅O₁₂
            gem_mode = de_mode == :beer_lambert ? :beer_lambert : :mc_lut  # :auto defaults to :mc_lut
            kwargs[:detector_efficiency] = detector_efficiency_gemstone(
                mode=gem_mode,
                thickness_mm=depth > 0 ? depth : 3.0,
                fill_factor=scanner.fill_factor_row > 0 ? scanner.fill_factor_row : 0.90)
        elseif depth > 0
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
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    filter::FilterType=StandardFilter(),
    cutoff::Float64=1.0
) where T<:AbstractFloat

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

    # Step 4: Mask outside FOV (clinical convention)
    apply_fov_mask!(ws.volume, geom)

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
    buf::AbstractArray{T,3},
    full::AbstractArray{T,3},
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
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    filter::FilterType=StandardFilter(),
    cutoff::Float64=1.0
) where T<:AbstractFloat

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
