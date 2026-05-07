"""
    Simulation/Driver.jl

High-level driver for running end-to-end CT simulations.
"""

export simulate!,
       reconstruct!,
       add_system_noise_floor!,
       compute_detector_I0,
       build_physics_config,
       resolve_source_spectrum_without_bowtie,
       resolve_source_spectrum_with_bowtie,
       apply_bowtie_to_spectrum

# =============================================================================
# Detector I0 Computation
# =============================================================================

"""
    compute_detector_I0(geom, protocol, spectrum_flux_sum) -> Float64

Expected primary photon count per detector pixel per view (no detector
quantum efficiency applied — combine with `quantum_efficiency_vector` at the
call site if needed).

    I₀ = spectrum_flux_sum × mA × time_per_view × pixel_area_at_detector_mm²

where `time_per_view = protocol.rotation_time / protocol.views` and
`pixel_area_at_detector_mm² = (pixel_size · M) × (pixel_row_size · M) · 100`
with `M = SDD/SAD` and pixel sizes in cm at isocenter (the `· 100` folds in
the cm→mm conversion of both edges).

`spectrum_flux_sum = sum(weights)` from `resolve_source_spectrum_without_bowtie`.
That spectrum has units of photons/mAs/mm² **at the scanner SDD** — the
`(750 mm / SDD)²` inverse-square-law correction is already baked into the
upstream `filter_spectrum` step, so no distance factor appears here.

# Reference (port from XCIST/CatSim, GE Research)
- `gecatsim/pyfiles/Spectrum.py`        — defines `viewTime = rotationTime/views`
                                          and `Ivec *= mA × viewTime`.
- `gecatsim/pyfiles/Detection_Flux.py`  — multiplies by detector active area
                                          and the inverse-square distance
                                          factor (which BasisSimulator folds
                                          into `filter_spectrum` instead).
"""
function compute_detector_I0(geom::CTGeometry, protocol::CTProtocol, spectrum_flux_sum::Float64)
    SDD_mm = geom.SDD * 10.0
    SAD_mm = geom.SAD * 10.0
    magnification = SDD_mm / SAD_mm
    pixel_col_det_mm = (geom.pixel_size * 10.0) * magnification
    pixel_row_det_mm = (geom.pixel_row_size * 10.0) * magnification
    pixel_area_mm2 = pixel_col_det_mm * pixel_row_det_mm
    time_per_view = protocol.rotation_time / protocol.views
    return spectrum_flux_sum * protocol.mA * time_per_view * pixel_area_mm2
end

# =============================================================================
# simulate!() — Zero-allocation PCCT simulation hot path
# =============================================================================

"""
    simulate!(ws::PCCTWorkspace, phantom, protocol, sim_opts) -> (pcct_sino, I0_bins)

Run PCCT simulation using pre-allocated workspace buffers for zero allocations
on the second and later calls (the first call may JIT-allocate).

All setup data (geometry, spectrum, physics config, detector, spectral response,
materials) is baked into the workspace at `create_workspace()` time.  The PCCT
detector model **always** applies the MC-LUT detector response matrix
(`compute_mc_drm`); the analytical / ideal-binning fallback inside
`pcct_forward_project` is deprecated and only reachable by callers that bypass
this driver.

Pulse pileup is on by default and toggleable via `sim_opts.use_pcct_pileup`.
Bin combination, scatter correction, and reconstruction are all decoupled —
do them at the notebook level using the returned per-bin sinograms and the
ground-truth `I0_bins`.

# Returns
- `pcct_sino` — `EnergyResolvedSinogram` (per-bin log line integrals).
- `I0_bins`   — per-bin reference photon count vector (for `-log(N/I0)` undo).
"""
function simulate!(
    ws::PCCTWorkspace{T},
    phantom,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
) where {T}
    geom          = ws.geom
    energies      = ws.energies
    weights       = ws.weights
    config        = ws.config
    pcct_detector = ws.pcct_detector
    mats          = ws.mats

    # Forward projection with workspace buffers (including native-res path + tiled spectral)
    pcct_sino = pcct_forward_project(
        phantom.mask, geom, pcct_detector;
        energies=energies, weights=weights,
        materials=mats,
        ws_bins=ws.bins, ws_μ_volume=ws.μ_volume, ws_sino_buf=ws.sino_buf,
        ws_scratch=ws.scratch,
        ws_thresholds_T=ws.thresholds_T,
        ws_η=ws.η, ws_R=ws.R, ws_R_energies=ws.R_energies,
        ws_I0_bins_norm=ws.I0_bins_norm,
        ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
        ws_μ_table=ws.μ_table,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom.extent,
        # Native-resolution forward projection path
        native_geom=ws.native_geom,
        ws_native_bins=ws.native_bins,
        ws_native_sino_buf=ws.native_sino_buf,
        ws_native_scratch=ws.native_scratch,
        ws_native_source_positions=ws.native_geom_source_positions,
        ws_native_detector_centers=ws.native_geom_detector_centers,
        ws_native_detector_u=ws.native_geom_detector_u,
        ws_native_detector_v=ws.native_geom_detector_v,
        # Tiled spectral projection buffers (fused PCCT path)
        ws_μ_table_gpu=ws.μ_table_gpu,
        ws_W_matrix_gpu=ws.W_matrix_gpu,
        ws_outputs_flat=ws.outputs_flat,
        ws_native_outputs_flat=ws.native_outputs_flat,
        ws_source_spectral=ws.source_spectral_gpu
    )

    # ─── Energy-resolved scatter injection (BEFORE noise) ───
    # Unified per-energy scatter model (shared with EICT):
    # 1. Spatial distribution: Ohnesorge convolution on combined sinogram
    # 2. Per-energy weights: Compton fraction 1/(1+(20/E)³) (NIST XCOM)
    # 3. Detector response: weights convolved through DRM → per-bin scatter
    # Must be added before noise so Poisson statistics are on total counts.
    #
    # References:
    # - Ohnesorge B et al., Eur Radiol 1999 (spatial scatter model)
    # - NIST XCOM (per-energy Compton fractions)
    I0_bins = ws.I0_bins
    I0_total = T(sum(I0_bins))
    eps_combine = T(1e-10)

    if config.scatter !== nothing
        # Step 1: Combine primary bins → combined_primary (for scatter spatial estimation)
        combined_primary = ws.combined
        fill!(combined_primary, zero(T))
        for (b, bin_sino) in enumerate(pcct_sino.bins)
            let I0b = T(I0_bins[b]), bs = bin_sino, comb = combined_primary
                AK.foreachindex(bs) do idx
                    comb[idx] += I0b * exp(-bs[idx])
                end
            end
        end
        let comb = combined_primary, I0t = I0_total, eps = eps_combine
            AK.foreachindex(comb) do idx
                comb[idx] = -log(max(comb[idx], eps) / I0t)
            end
        end

        # Step 2: Spatial scatter field (Ohnesorge convolution model)
        scatter_field = ws.tube_physics_scratch
        estimate_scatter_field!(scatter_field, combined_primary, config.scatter;
            ws_scatter_temp=ws.scratch)

        # Step 3: Per-energy scatter weights → per-bin via DRM
        ew = compute_scatter_energy_weights(Float64.(energies))
        bin_weights = compute_scatter_bin_weights(
            Float64.(energies), Float64.(weights),
            ew, Float64.(ws.η), ws.R, ws.kVp)

        # Step 4: Inject scatter into each bin
        inject_scatter_bins!(pcct_sino.bins, scatter_field, I0_bins, I0_total, bin_weights)
    end

    # ─── Noise (in-place on pcct_sino.bins — now includes scatter in counts) ───
    I0_physics = compute_detector_I0(geom, protocol, sum(ws.weights))
    if sim_opts.use_noise
        apply_pcct_noise!(pcct_sino, pcct_detector, protocol;
            seed=sim_opts.seed, I0=I0_physics,
            energies=energies, weights=weights,
            ws_noise_staging=ws.noise_staging,
            ws_noise_buf=ws.noise_buf,
            ws_rng=ws.rng,
            ws_noise_I0=ws.noise_I0,
            ws_η=ws.η,
            ws_R=ws.R,
            noise_reduction=sim_opts.pcct_noise_reduction)
    end

    # ─── MC-LUT pulse pileup — full spectral-migration matrix S ───
    # S[i,j] = P(true-bin-j count is recorded in bin-i), pre-computed at
    # workspace creation by `compute_mc_pileup_matrix` (Monte Carlo of
    # `simulate_pulse_train` with per-event trigger-bin tracking).  Column
    # sums ≤ 1 — the deficit is the count loss, so a single S × counts
    # multiply captures inter-bin migration AND count loss in one pass.
    # No analytical Taguchi / semi-non-paralyzable count factor.
    #
    # Per-pixel pipeline (one fused kernel):
    #   counts[j]   = I0_truth[j] · exp(-bins[j])
    #   recorded[i] = Σ_j S[i,j] · counts[j]
    #   bins[i]     = -log(recorded[i] / I0_truth[i])      ←  truth-basis (KEY!)
    #
    # **Truth-basis normalization** (NOT recorded-basis): the post-pileup
    # bin is `-log(recorded / I0_truth)` so the I0 returned to the caller
    # remains the truth I0 — i.e. `I0_b · exp(-bin) = recorded_count`,
    # the count-domain math notebooks rely on for scatter correction stays
    # valid.  Bins for an air ray are no longer exactly 0 but a small
    # per-bin offset `log(I0_truth / I0_recorded)` reflecting how much of
    # the bin's air baseline pile-up shifted into other bins — physically
    # the right thing.
    #
    # Why this matters: an earlier draft normalized against I0_recorded
    # which made an air ray give bins=0 but broke nb04's scatter
    # subtraction (`N_measured - N_scatter`) by mis-scaling per-bin I0
    # vs the I0_total used inside scatter — over-corrected bins 1–3,
    # under-corrected bin 4 → bin-2-only streaks.
    if ws.use_pcct_pileup && ws.pileup_S !== nothing
        S = ws.pileup_S
        n_bins = length(pcct_sino.bins)
        n_bins == 4 || error("MC pile-up application is currently specialized to 4 bins; got $(n_bins).")

        eps_pileup = T(1e-10)
        let b1 = pcct_sino.bins[1], b2 = pcct_sino.bins[2],
            b3 = pcct_sino.bins[3], b4 = pcct_sino.bins[4],
            I0_t1 = T(ws.I0_bins[1]), I0_t2 = T(ws.I0_bins[2]),
            I0_t3 = T(ws.I0_bins[3]), I0_t4 = T(ws.I0_bins[4]),
            S11 = T(S[1, 1]),
            S21 = T(S[2, 1]), S22 = T(S[2, 2]),
            S31 = T(S[3, 1]), S32 = T(S[3, 2]), S33 = T(S[3, 3]),
            S41 = T(S[4, 1]), S42 = T(S[4, 2]), S43 = T(S[4, 3]), S44 = T(S[4, 4]),
            eps = eps_pileup
            AK.foreachindex(b1) do idx
                # 1. truth counts per bin (Float32 registers, no scratch sinos)
                c1 = I0_t1 * exp(-b1[idx])
                c2 = I0_t2 * exp(-b2[idx])
                c3 = I0_t3 * exp(-b3[idx])
                c4 = I0_t4 * exp(-b4[idx])
                # 2. recorded counts via lower-triangular S × counts
                r1 = S11 * c1
                r2 = S21 * c1 + S22 * c2
                r3 = S31 * c1 + S32 * c2 + S33 * c3
                r4 = S41 * c1 + S42 * c2 + S43 * c3 + S44 * c4
                # 3. back to log-line-integral, normalized against TRUTH I0
                #    so I0_b · exp(-bin) = recorded count for downstream math.
                b1[idx] = -log(max(r1, eps) / I0_t1)
                b2[idx] = -log(max(r2, eps) / I0_t2)
                b3[idx] = -log(max(r3, eps) / I0_t3)
                b4[idx] = -log(max(r4, eps) / I0_t4)
            end
        end
    end

    # Bin-combine, scatter correction, BHC, and pile-up correction are all
    # decoupled — done at the notebook level (see docs/notebooks/04_pcct_vmi.jl
    # for the canonical combine + correct pattern).
    #
    # Returned fields:
    # - `pcct_sino`  : per-bin log-line-integral sinograms.  When pile-up is
    #                  on these are `-log(recorded / I0_truth)`; running
    #                  `apply_pcct_pileup_correction!` with `pileup_S` recovers
    #                  truth-like bins for downstream calibration / decomp.
    # - `I0_bins`    : truth per-bin air baseline (DRM-weighted, pre-pileup).
    #                  `I0_b · exp(-bin) = recorded count` round-trip identity.
    # - `pileup_S`   : MC pile-up migration matrix (`nothing` when pile-up off).
    #                  Pass into `apply_pcct_pileup_correction!` to invert
    #                  the pile-up degradation in the sinogram domain.
    return (
        pcct_sino = pcct_sino,
        I0_bins   = ws.I0_bins,
        pileup_S  = ws.pileup_S,
    )
end

# =============================================================================
# simulate!() — Zero-allocation EICT single-kVp simulation hot path
# =============================================================================

"""
    simulate!(ws::EICTWorkspace, phantom, scanner, protocol, sim_opts, recon_opts)

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
    recon_opts::ReconOptions=ReconOptions(),
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

    # Apply physics pipeline (sinogram domain, no noise, no scatter)
    # Note: scatter is now applied separately below (unified per-energy model)
    _apply_physics_no_noise!(ws.sinogram, geom, config;
        ws_output=ws.physics_output,
        ws_scatter_kernel=ws.scatter_kernel,
        ws_scatter_temp=ws.scatter_temp,
        ws_scatter_kernel_1d=ws.scatter_kernel_1d,
        ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
        ws_focal_spot_kernel=ws.focal_spot_kernel,
        ws_lag_output=ws.physics_output,
        ws_lag_intensity=ws.lag_intensity,
        ws_lag_coeffs=ws.lag_coeffs)

    # ─── Energy-resolved scatter injection (unified with PCCT) ───
    # Same per-energy scatter model as PCCT, integrated over the full spectrum
    # since an energy-integrating detector sums all energies.
    # 1. Spatial field: Ohnesorge convolution (Ohnesorge et al., Eur Radiol 1999)
    # 2. Per-energy weights: Compton fraction 1/(1+(20/E)³) (NIST XCOM)
    # 3. Detector response: spectrum-weighted integration
    #
    # Scatter field + weight are saved for decoupled correction (notebook-level).
    scatter_field_cpu = nothing
    scatter_total_weight = 0.0
    if config.scatter !== nothing
        scatter_field = ws.physics_output  # reuse scratch buffer
        estimate_scatter_field!(scatter_field, ws.sinogram, config.scatter;
            ws_scatter_temp=ws.scatter_temp, ws_kernel_1d=ws.scatter_kernel_1d)
        ew = compute_scatter_energy_weights(Float64.(ws.energies))
        wn = Float64.(ws.weights_norm)
        η = ws.η_vec
        scatter_total_weight = sum(wn[i] * ew[i] * η[i] for i in eachindex(wn)) /
                               max(sum(wn[i] * η[i] for i in eachindex(wn)), 1e-30)
        # Save scatter artifacts for notebook-level correction
        scatter_field_cpu = Array(scatter_field)
        inject_scatter!(ws.sinogram, scatter_field, scatter_total_weight)
    end

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Fused noise + scatter subtraction (counts domain)
    #
    # Physical reality (DAS signal chain):
    #   1. Detector sees primary + scatter → total counts
    #   2. Poisson noise on total counts
    #   3. DAS subtracts estimated scatter counts
    #   4. Clamp at 1 count (DAS hardware minimum)
    #
    # Fused into one kernel to avoid intermediate log/exp round-trips.
    # Scatter_field + weight still returned for notebook custom processing.
    # ═══════════════════════════════════════════════════════════════════════
    I0_raw = compute_detector_I0(geom, protocol, sum(ws.weights))
    η_eff = sum(ws.weights_norm[i] * ws.η_vec[i] for i in 1:length(ws.η_vec))
    I0_T = T(I0_raw * η_eff)

    has_scatter = config.scatter !== nothing
    scatter_field_gpu = if has_scatter
        sf_gpu = similar(ws.sinogram)
        copyto!(sf_gpu, scatter_field_cpu)
        sf_gpu
    else
        nothing
    end

    if sim_opts.use_noise
        randn!(ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)

        mean_E_keV = sum(ws.weights_norm[i] * ws.energies[i] for i in 1:length(ws.energies))
        σ_e_photon = T(scanner.electronic_noise / (mean_E_keV * scanner.detection_gain))

        if σ_e_photon > T(0)
            randn!(ws.enoise_rand_cpu)
            copyto!(ws.enoise_rand_gpu, ws.enoise_rand_cpu)

            let sino = ws.sinogram, rg = ws.noise_rand_gpu, eg = ws.enoise_rand_gpu,
                    I0v = I0_T, σ_e = σ_e_photon,
                    sf = scatter_field_gpu, sw = T(scatter_total_weight), do_sc = has_scatter
                AK.foreachindex(sino) do idx
                    λ_total = I0v * exp(-sino[idx])
                    λ_noisy = λ_total + sqrt(max(λ_total, one(T))) * rg[idx]
                    λ_noisy += σ_e * eg[idx]
                    λ_primary = if do_sc
                        max(λ_noisy - I0v * sf[idx] * sw, one(T))
                    else
                        max(λ_noisy, one(T))
                    end
                    sino[idx] = -log(λ_primary / I0v)
                end
            end
        else
            let sino = ws.sinogram, rg = ws.noise_rand_gpu, I0v = I0_T,
                    sf = scatter_field_gpu, sw = T(scatter_total_weight), do_sc = has_scatter
                AK.foreachindex(sino) do idx
                    λ_total = I0v * exp(-sino[idx])
                    λ_noisy = λ_total + sqrt(max(λ_total, one(T))) * rg[idx]
                    λ_primary = if do_sc
                        max(λ_noisy - I0v * sf[idx] * sw, one(T))
                    else
                        max(λ_noisy, one(T))
                    end
                    sino[idx] = -log(λ_primary / I0v)
                end
            end
        end
    elseif has_scatter
        # No noise but scatter subtraction still needed (noise-free sim)
        let sino = ws.sinogram, sf = scatter_field_gpu, sw = T(scatter_total_weight), I0v = I0_T
            AK.foreachindex(sino) do idx
                λ_total = I0v * exp(-sino[idx])
                λ_primary = max(λ_total - I0v * sf[idx] * sw, one(T))
                sino[idx] = -log(λ_primary / I0v)
            end
        end
    end
    scatter_field_gpu = nothing

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Calibration (intensity → air scan → log)
    # ═══════════════════════════════════════════════════════════════════════
    eps = T(1e-10)

    # Convert to intensity domain
    let sino = ws.sinogram
        AK.foreachindex(sino) do idx
            sino[idx] = exp(-clamp(sino[idx], T(-1), T(15)))
        end
    end

    # Air scan calibration
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

    # BHC is decoupled — applied at notebook level

    # ═══════════════════════════════════════════════════════════════════════
    # Save sinograms
    # ═══════════════════════════════════════════════════════════════════════
    copyto!(ws.sino_noisy_out, ws.sinogram)
    copyto!(ws.sino_ideal_out, ws.sinogram)

    return (sino_ideal=ws.sino_ideal_out, sino_noisy=ws.sino_noisy_out,
            scatter_field=scatter_field_cpu, scatter_weight=scatter_total_weight)
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

"""
    resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner=nothing) -> (energies, weights)

**1D centered spectrum** — tube × flat filter × additional filter, NO bowtie.

Returns the source-side spectrum along the centered reference ray.  This is
what leaves the tube and passes through the fixed filtration (flat filter +
any `additional_filters` in the protocol) at the source-to-detector distance.
It does NOT include:
- **Bowtie filter** — rays at the fan edge see more filtration than center.
  Use `resolve_source_spectrum_with_bowtie` when bowtie matters (basis
  decomposition, beam hardening correction, etc.).
- **Detector response** (QE, energy-resolving DRM) — callers apply these
  separately when needed (see PCCT basis construction in `src/scanners/`).

Suitable for: quick mean-energy diagnostics, global BHC calibration (when
bowtie is ignored), scatter estimation, and Monte-Carlo source spectra.
Not suitable for: per-ray polychromatic forward modelling when the scanner
has a bowtie — use the `_with_bowtie` variant.

Pass `scanner` so the pipeline can read `flat_filter_material`,
`flat_filter_thickness`, and `source_to_detector` from the hardware spec.
"""
function resolve_source_spectrum_without_bowtie(sim_opts::SimOptions, protocol::CTProtocol; scanner=nothing)
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
    apply_bowtie_to_spectrum(w_1d, e, scanner, geom, protocol; include_bowtie=true, label="") -> ŵ

Inject the bowtie transmission into a 1D source spectrum → per-ray 3D `ŵ`.

Given a 1D centered spectrum `w_1d` (from
`resolve_source_spectrum_without_bowtie`) and its energy grid `e`, multiplies
by the bowtie attenuation `B[col, row, e]` pixel-by-pixel and renormalizes so
`Σ_k ŵ[col, row, k] = 1` per ray.  Returns:
- `ŵ::Array{Float32,3}` of shape `[n_col, n_row, n_E]` when bowtie is present.
- `Float32` vector (normalized `w_1d`) unchanged when `include_bowtie == false`
  OR the scanner has no bowtie configured (`bowtie_filter` property absent,
  `:none`, or `nothing`).

This is the low-level primitive.  Most callers should use
`resolve_source_spectrum_with_bowtie(...)`, which composes the two steps.

# Arguments
- `w_1d`, `e` — 1D spectrum + its energy grid (from `_without_bowtie`).
- `scanner`, `geom`, `protocol` — passed to `resolve_bowtie_filter` and
  `compute_bowtie_attenuation_spectral` to assemble `B[col, row, e]`.

# Keyword Arguments
- `include_bowtie::Bool = true` — set `false` to bypass and return a 1D
  normalized spectrum, useful as a runtime toggle.
- `label::String = ""` — prefix for the `@info` diagnostic line (e.g. `"low"`,
  `"high"`) so multi-bin pipelines label their logs.
"""
function apply_bowtie_to_spectrum(
        w_1d::AbstractVector,
        e::AbstractVector{<:Real},
        scanner,
        geom,
        protocol;
        include_bowtie::Bool = true,
        label::String = "",
    )
    n_E = length(e)
    bowtie_present = include_bowtie &&
                     hasproperty(scanner, :bowtie_filter) &&
                     scanner.bowtie_filter !== :none &&
                     scanner.bowtie_filter !== nothing
    if !bowtie_present
        w_norm = Float32.(Float64.(w_1d) ./ sum(Float64.(w_1d)))
        @info "[apply_bowtie_to_spectrum$(isempty(label) ? "" : " ("*label*")")] bowtie OFF → 1D centered spectrum"
        return w_norm
    end
    bowtie = resolve_bowtie_filter(scanner.bowtie_filter; kVp = Int(protocol.kVp))
    B = compute_bowtie_attenuation_spectral(bowtie, geom, Float64.(e))
    n_col, n_row = size(B, 1), size(B, 2)
    w_pr = Array{Float32, 3}(undef, n_col, n_row, n_E)
    @inbounds for k in 1:n_E
        wk = Float32(w_1d[k])
        for row in 1:n_row, col in 1:n_col
            w_pr[col, row, k] = wk * Float32(B[col, row, k])
        end
    end
    # Per-ray normalization so Σ_k ŵ = 1 per (col, row).
    @inbounds for row in 1:n_row, col in 1:n_col
        tot = 0f0
        for k in 1:n_E
            tot += w_pr[col, row, k]
        end
        inv_tot = 1f0 / max(tot, 1f-20)
        for k in 1:n_E
            w_pr[col, row, k] *= inv_tot
        end
    end
    mid_c = n_col ÷ 2 + 1
    mid_r = n_row ÷ 2 + 1
    mean_E_center = sum(Float64(e[k]) * w_pr[mid_c, mid_r, k] for k in 1:n_E)
    mean_E_edge1  = sum(Float64(e[k]) * w_pr[1,     mid_r, k] for k in 1:n_E)
    @info "[apply_bowtie_to_spectrum$(isempty(label) ? "" : " ("*label*")")] bowtie ON → per-ray 3D ŵ  [$(n_col) × $(n_row) × $(n_E)]"
    @info "  center-ray mean E = $(round(mean_E_center, digits = 1)) keV"
    @info "  edge-ray   mean E = $(round(mean_E_edge1,  digits = 1)) keV   (Δ = $(round(mean_E_edge1 - mean_E_center, digits = 1)) keV ← bowtie hardening)"
    return w_pr
end

"""
    resolve_source_spectrum_with_bowtie(sim_opts, protocol; scanner, geom, include_bowtie=true, label="") -> (e, ŵ)

**Bowtie-aware spectrum** — `resolve_source_spectrum_without_bowtie` composed
with `apply_bowtie_to_spectrum`.

Convenience wrapper: returns the 1D energy grid `e` and a per-ray spectrum
`ŵ`.  When bowtie is present AND `include_bowtie == true`, `ŵ` is 3D
`[n_col, n_row, n_E]` with the bowtie transmission baked in and each ray
normalized to `Σ_k ŵ = 1`.  When bowtie is absent or disabled, `ŵ` collapses
to a 1D `[n_E]` vector (the same as `_without_bowtie` output, just
normalized).  Downstream code should dispatch on `ndims(ŵ)`.

Use this instead of calling the two primitives yourself whenever you want
basis decomposition, per-ray BHC, or any other forward-model step that
needs the exact spectrum each ray saw.
"""
function resolve_source_spectrum_with_bowtie(
        sim_opts::SimOptions,
        protocol::CTProtocol;
        scanner,
        geom,
        include_bowtie::Bool = true,
        label::String = "",
    )
    e, w_1d = resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)
    ŵ = apply_bowtie_to_spectrum(w_1d, e, scanner, geom, protocol;
                                  include_bowtie = include_bowtie, label = label)
    return e, ŵ
end

"""
    build_physics_config(scanner::Scanner, sim_opts::SimOptions, energies::Vector{Float64}, weights::Vector{Float64}; phantom=nothing) -> PhysicsConfig

Build a complete PhysicsConfig from Scanner hardware fields and SimOptions toggles.

For effects with Scanner fields (focal_spot, fill_factor, detector_efficiency,
heel_effect), the Scanner hardware parameters are used to construct the effect structs.
For effects without Scanner fields (scatter, optical_crosstalk, lag),
factory function defaults are used.

Noise is not part of PhysicsConfig — it is applied externally via
`compute_detector_I0()` + quantum noise when `sim_opts.use_noise == true`.

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
    phantom_diameter_cm = if phantom !== nothing && sim_opts.use_scatter
        voxel_size_mm = phantom.voxel_size .* 10.0
        estimate_phantom_diameter_cm(phantom.mask, voxel_size_mm)
    else
        nothing
    end

    if sim_opts.use_scatter
        kwargs[:scatter] = geometry_aware_scatter_model(scanner; phantom_diameter_cm=phantom_diameter_cm)
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

    # BHC is decoupled — applied at notebook level using calibrate_bhc() + apply_bhc_*()

    return default_physics_config(; kwargs...)
end

# =============================================================================
# reconstruct!() — Zero-allocation FDK reconstruction hot path
# =============================================================================

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
    reconstruct!(ws::HIRReconWorkspace, sinogram, geom, volume_size; filter=StandardFilter(), cutoff=1.0, air_reference=nothing)

Zero-allocation Hybrid IR reconstruction using pre-allocated workspace buffers.

Implements TRUE Hybrid IR = FDK initialization + PWLS refinement with Huber regularization.
When `n_subsets > 0`, uses Ordered Subsets PWLS (OS-PWLS) for ~10-25x speedup.
All iteration buffers are pre-allocated in the workspace.

Create the workspace with `create_hir_recon_workspace(sinogram, geom, volume_size; strength=3)`.

# Keyword Arguments
- `air_reference`: Optional 2D array [n_cols, n_rows] of bowtie air reference values.
  When provided, stat_weights are scaled by air_ref to account for position-dependent
  noise from the bowtie filter (edge pixels have fewer counts → more noise → lower weight).
"""
function reconstruct!(
    ws::HIRReconWorkspace{T},
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    filter::FilterType=StandardFilter(),
    cutoff::Float64=1.0,
    init_volume::Union{Nothing, AbstractArray{T,3}}=nothing,
    air_reference::Union{Nothing, AbstractArray}=nothing
) where T<:AbstractFloat

    # ─── Step 1: FDK initialization (or warm-start from provided volume) ───
    if init_volume === nothing
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
    else
        copyto!(ws.volume, init_volume)
    end

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

        # Initialize statistical weights: w = air_ref(col,row) × exp(-y)
        # air_ref accounts for bowtie-modulated I0 (edge pixels → fewer counts → lower weight)
        if air_reference !== nothing
            let sw = ws.stat_weights, ε = T(1e-6), aref = air_reference,
                    nc = Int32(size(sinogram, 1)), nr = Int32(size(sinogram, 2))
                AK.foreachindex(sw, backend) do idx
                    idx_0 = Int32(idx - 1)
                    col = (idx_0 % nc) + Int32(1)
                    row = ((idx_0 ÷ nc) % nr) + Int32(1)
                    ref_idx = col + (row - Int32(1)) * nc
                    y_val = sinogram[idx]
                    y_clipped = clamp(y_val, T(-10), T(10))
                    sw[idx] = T(aref[ref_idx]) * exp(-y_clipped) + ε
                end
            end
        else
            let sw = ws.stat_weights, ε = T(1e-6)
                AK.foreachindex(sw, backend) do idx
                    y_val = sinogram[idx]
                    y_clipped = clamp(y_val, T(-10), T(10))
                    sw[idx] = exp(-y_clipped) + ε
                end
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

        # Initialize statistical weights: w = air_ref(col,row) × exp(-y)
        if air_reference !== nothing
            let sw = ws.stat_weights, ε = T(1e-6), aref = air_reference,
                    nc = Int32(size(sinogram, 1)), nr = Int32(size(sinogram, 2))
                AK.foreachindex(sw, backend) do idx
                    idx_0 = Int32(idx - 1)
                    col = (idx_0 % nc) + Int32(1)
                    row = ((idx_0 ÷ nc) % nr) + Int32(1)
                    ref_idx = col + (row - Int32(1)) * nc
                    y_val = sinogram[idx]
                    y_clipped = clamp(y_val, T(-10), T(10))
                    sw[idx] = T(aref[ref_idx]) * exp(-y_clipped) + ε
                end
            end
        else
            let sw = ws.stat_weights, ε = T(1e-6)
                AK.foreachindex(sw, backend) do idx
                    y_val = sinogram[idx]
                    y_clipped = clamp(y_val, T(-10), T(10))
                    sw[idx] = exp(-y_clipped) + ε
                end
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

                if air_reference !== nothing
                    let sw = ws.stat_weights, ax = ws.Ax, aref = air_reference,
                            nc = Int32(size(sinogram, 1)), nr = Int32(size(sinogram, 2))
                        AK.foreachindex(sw, backend) do idx
                            idx_0 = Int32(idx - 1)
                            col = (idx_0 % nc) + Int32(1)
                            row = ((idx_0 ÷ nc) % nr) + Int32(1)
                            ref_idx = col + (row - Int32(1)) * nc
                            ax_val = ax[idx]
                            ax_clipped = clamp(ax_val, T(-10), T(10))
                            sw[idx] = T(aref[ref_idx]) * exp(-ax_clipped)
                        end
                    end
                else
                    let sw = ws.stat_weights, ax = ws.Ax
                        AK.foreachindex(sw, backend) do idx
                            ax_val = ax[idx]
                            ax_clipped = clamp(ax_val, T(-10), T(10))
                            sw[idx] = exp(-ax_clipped)
                        end
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
