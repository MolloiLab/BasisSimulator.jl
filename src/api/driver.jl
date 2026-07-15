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
    resolve_source_spectrum_full,
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
    geom = ws.geom
    energies = ws.energies
    weights = ws.weights
    config = ws.config
    pcct_detector = ws.pcct_detector
    mats = ws.mats

    # Forward projection with workspace buffers (including native-res path + tiled spectral)
    pcct_sino = pcct_forward_project(
        phantom.mask, geom, pcct_detector;
        energies = energies, weights = weights,
        materials = mats,
        ws_bins = ws.bins, ws_μ_volume = ws.μ_volume, ws_sino_buf = ws.sino_buf,
        ws_scratch = ws.scratch,
        ws_thresholds_T = ws.thresholds_T,
        ws_η = ws.η, ws_R = ws.R, ws_R_energies = ws.R_energies,
        ws_I0_bins_norm = ws.I0_bins_norm,
        ws_μ_lut_cpu = ws.μ_lut_cpu, ws_μ_lut_gpu = ws.μ_lut_gpu,
        ws_μ_table = ws.μ_table,
        ws_source_positions = ws.geom_source_positions,
        ws_detector_centers = ws.geom_detector_centers,
        ws_detector_u = ws.geom_detector_u,
        ws_detector_v = ws.geom_detector_v,
        volume_extent = phantom.extent,
        # Native-resolution forward projection path (used when bf > 1)
        native_geom = ws.native_geom,
        ws_native_bins = ws.native_bins,
        ws_native_sino_buf = ws.native_sino_buf,
        ws_native_source_positions = ws.native_geom_source_positions,
        ws_native_detector_centers = ws.native_geom_detector_centers,
        ws_native_detector_u = ws.native_geom_detector_u,
        ws_native_detector_v = ws.native_geom_detector_v,
        # Tiled spectral projection buffers (fused PCCT path)
        ws_μ_table_gpu = ws.μ_table_gpu,
        ws_W_matrix_gpu = ws.W_matrix_gpu,
        ws_outputs_flat = ws.outputs_flat,
        ws_native_outputs_flat = ws.native_outputs_flat,
        projector = sim_opts.projector,
    )

    # ─── Tube-side focal-spot blur (per bin, BEFORE scatter/noise/pile-up) ───
    # The focal-spot penumbra is a tube-side effect common to both detector
    # chains; here it mirrors the EICT placement in `_apply_physics_no_noise!`
    # (a detector-plane convolution of the log line integrals). Placed before
    # the rate-dependent pile-up step so blurred local count rates feed it.
    # Note the blur acts at binned (not native-dexel) resolution.
    # Off by default for the :pcct fidelity preset — opt in with
    # `use_focal_spot = true`. Detector lag is intentionally NOT applied on
    # this path: the shipped lag model is scintillator (Gd₂O₂S) afterglow,
    # which direct-conversion PCCT detectors do not exhibit.
    if config.focal_spot !== nothing
        for bin_sino in pcct_sino.bins
            apply_focal_spot_blur!(
                bin_sino, config.focal_spot, geom;
                ws_output = ws.tube_physics_scratch,
                ws_kernel = ws.focal_spot_kernel
            )
        end
    end

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
    eps_combine = T(1.0e-10)

    if config.scatter !== nothing && sim_opts.use_pcct_scatter
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
        estimate_scatter_field!(
            scatter_field, combined_primary, config.scatter;
            ws_scatter_temp = ws.scratch
        )

        # Step 3: Per-energy scatter weights → per-bin via DRM
        ew = compute_scatter_energy_weights(Float64.(energies))
        bin_weights = compute_scatter_bin_weights(
            Float64.(energies), Float64.(weights),
            ew, Float64.(ws.η), ws.R, ws.kVp
        )

        # Step 4: Inject scatter into each bin
        inject_scatter_bins!(pcct_sino.bins, scatter_field, I0_bins, I0_total, bin_weights)
    end

    # ─── Noise (in-place on pcct_sino.bins — now includes scatter in counts) ───
    I0_physics = compute_detector_I0(geom, protocol, sum(ws.weights))
    if sim_opts.use_noise
        apply_pcct_noise!(
            pcct_sino, pcct_detector, protocol;
            seed = sim_opts.seed, I0 = I0_physics,
            energies = energies, weights = weights,
            ws_noise_staging = ws.noise_staging,
            ws_noise_buf = ws.noise_buf,
            ws_rng = ws.rng,
            ws_noise_I0 = ws.noise_I0,
            ws_η = ws.η,
            ws_R = ws.R,
            noise_reduction = sim_opts.pcct_noise_reduction
        )
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

        eps_pileup = T(1.0e-10)
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

    # --- PCCT pileup correction (optional; use_pcct_pileup_correction) ---
    # Inverts the MC pileup matrix S applied above via apply_pcct_pileup_correction! —
    # the same model-based un-pileup a clinical recon performs before downstream
    # processing.  Logically identical to the validated nb08 inline sim_pileup cell.
    # Runs before scatter correction so the latter re-estimates from un-piled counts.
    if ws.use_pcct_pileup && ws.pileup_S !== nothing && sim_opts.use_pcct_pileup_correction
        apply_pcct_pileup_correction!(pcct_sino.bins, ws.I0_bins, ws.pileup_S)
    end

    # --- PCCT scatter correction (optional; use_pcct_scatter_correction) ---
    # Model-based re-estimate-and-subtract, logically identical to the validated
    # nb08 inline scatter-correction cell: re-combine the CURRENT bins (now
    # primary+scatter+noise+pileup), re-estimate the Ohnesorge field, and subtract
    # the per-bin scatter.  Scalar I0 here equals the per-pixel reference for a
    # bowtie-free scanner.  Runs after pileup so it sees the recorded counts.
    if config.scatter !== nothing && sim_opts.use_pcct_scatter_correction
        combined_corr = ws.combined
        fill!(combined_corr, zero(T))
        for (b, bin_sino) in enumerate(pcct_sino.bins)
            let I0b = T(I0_bins[b]), bs = bin_sino, comb = combined_corr
                AK.foreachindex(bs) do idx
                    comb[idx] += I0b * exp(-bs[idx])
                end
            end
        end
        let comb = combined_corr, I0t = I0_total, eps = eps_combine
            AK.foreachindex(comb) do idx
                comb[idx] = -log(max(comb[idx], eps) / I0t)
            end
        end
        scatter_field_corr = ws.tube_physics_scratch
        estimate_scatter_field!(
            scatter_field_corr, combined_corr, config.scatter;
            ws_scatter_temp = ws.scratch
        )
        ew_corr = compute_scatter_energy_weights(Float64.(energies))
        bin_weights_corr = compute_scatter_bin_weights(
            Float64.(energies), Float64.(weights),
            ew_corr, Float64.(ws.η), ws.R, ws.kVp
        )
        inject_scatter_bins!(
            pcct_sino.bins, scatter_field_corr, I0_bins, I0_total,
            bin_weights_corr; subtract = true
        )
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
        I0_bins = ws.I0_bins,
        pileup_S = ws.pileup_S,
    )
end

# =============================================================================
# simulate!() — Zero-allocation EICT single-kVp simulation hot path
# =============================================================================

"""
    simulate!(ws::EICTWorkspace, phantom, protocol, sim_opts)

Run EICT single-kVp simulation using pre-allocated workspace buffers.

Mutates `ws.sinogram` in place (the final log line-integral sinogram).
Returns `nothing` — read `ws.sinogram` (and `ws.geom`) off the workspace.

Create the workspace with `create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)`;
scanner-derived noise constants (`η_eff`, `σ_e_photon`) are baked into `ws` at
that point, so `simulate!` no longer needs `scanner`.

Reconstruction is NOT included — handled by the caller via `reconstruct!`.
"""
function simulate!(
        ws::EICTWorkspace{T},
        phantom,
        protocol::CTProtocol,
        sim_opts::SimOptions = SimOptions(),
    ) where {T}
    geom = ws.geom
    energies = ws.energies
    mats = ws.mats
    config = ws.config

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Polychromatic forward projection (Beer-Lambert)
    # ═══════════════════════════════════════════════════════════════════════
    fill!(ws.sinogram, zero(T))
    _forward_project_poly!(
        ws.sinogram, phantom.mask, geom, energies, ws.weights, mats;
        ws_μ_volume = ws.μ_volume, ws_sino_mono = ws.sino_mono,
        ws_I_transmitted = ws.I_transmitted,
        ws_weights_norm = ws.weights_norm,
        ws_μ_lut_cpu = ws.μ_lut_cpu, ws_μ_lut_gpu = ws.μ_lut_gpu,
        ws_μ_table = ws.μ_table,
        ws_μ_table_gpu = ws.μ_table_gpu,
        ws_source_positions = ws.geom_source_positions,
        ws_detector_centers = ws.geom_detector_centers,
        ws_detector_u = ws.geom_detector_u,
        ws_detector_v = ws.geom_detector_v,
        volume_extent = phantom.extent,
        ws_η = ws.η_vec,
        ws_bowtie_spectral = ws.bowtie_spectral,
        ws_wη_gpu = ws.wη_gpu,
        projector = sim_opts.projector
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Signal chain (always active)
    # ═══════════════════════════════════════════════════════════════════════

    # Apply physics pipeline (sinogram domain, no noise, no scatter)
    # Note: scatter is now applied separately below (unified per-energy model)
    _apply_physics_no_noise!(
        ws.sinogram, geom, config;
        ws_output = ws.physics_output,
        ws_scatter_kernel = ws.scatter_kernel,
        ws_scatter_temp = ws.scatter_temp,
        ws_scatter_kernel_1d = ws.scatter_kernel_1d,
        ws_optical_crosstalk_kernel = ws.optical_crosstalk_kernel,
        ws_focal_spot_kernel = ws.focal_spot_kernel,
        ws_lag_output = ws.physics_output,
        ws_lag_intensity = ws.lag_intensity,
        ws_lag_coeffs = ws.lag_coeffs
    )

    # ─── Energy-resolved scatter injection (unified with PCCT) ───
    # Same per-energy scatter model as PCCT, integrated over the full spectrum
    # since an energy-integrating detector sums all energies.
    # 1. Spatial field: Ohnesorge convolution (Ohnesorge et al., Eur Radiol 1999)
    # 2. Per-energy weights: Compton fraction 1/(1+(20/E)³) (NIST XCOM)
    # 3. Detector response: spectrum-weighted integration
    has_scatter = config.scatter !== nothing
    scatter_total_weight = 0.0
    scatter_field_gpu = nothing  # set below if has_scatter; reused in step 3
    if has_scatter
        scatter_field_gpu = ws.physics_output  # GPU buffer; live until step 3
        estimate_scatter_field!(
            scatter_field_gpu, ws.sinogram, config.scatter;
            ws_scatter_temp = ws.scatter_temp, ws_kernel_1d = ws.scatter_kernel_1d
        )
        ew = compute_scatter_energy_weights(Float64.(ws.energies))
        wn = Float64.(ws.weights_norm)
        η = ws.η_vec
        scatter_total_weight = sum(wn[i] * ew[i] * η[i] for i in eachindex(wn)) /
            max(sum(wn[i] * η[i] for i in eachindex(wn)), 1.0e-30)
        inject_scatter!(ws.sinogram, scatter_field_gpu, scatter_total_weight)
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
    # Scatter field is the same GPU buffer estimated in step 2 (no round-trip).
    # ═══════════════════════════════════════════════════════════════════════
    I0_raw = compute_detector_I0(geom, protocol, sum(ws.weights))
    I0_T = T(I0_raw) * ws.η_eff   # ws.η_eff already T-typed, baked at create time

    # Capture `sf` as a real GPU buffer regardless of scatter on/off.  When
    # scatter is off, `do_sc=false` zeroes out the contribution AND we point
    # `sf` at `ws.physics_output` (already-allocated GPU-side buffer) so the
    # `sf[idx]` reference type-checks during Metal kernel compilation — even
    # though the dead branch never reads it.  Capturing `sf::Nothing` here
    # tripped GPUCompiler's box_int64 path on Metal.
    sf_kernel = has_scatter ? scatter_field_gpu : ws.physics_output

    if sim_opts.use_noise
        # Reproducibility: seed `ws.rng` from sim_opts.seed each call.  Without
        # this the inline randn! pulls from the global RNG and same-seed reruns
        # disagree.  PCCT path goes through apply_pcct_noise!(seed=...) which
        # already does this.
        if sim_opts.seed !== nothing
            Random.seed!(ws.rng, sim_opts.seed)
        end
        randn!(ws.rng, ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)

        σ_e_photon = ws.σ_e_photon

        if σ_e_photon > T(0)
            randn!(ws.rng, ws.enoise_rand_cpu)
            copyto!(ws.enoise_rand_gpu, ws.enoise_rand_cpu)

            let sino = ws.sinogram, rg = ws.noise_rand_gpu, eg = ws.enoise_rand_gpu,
                    I0v = I0_T, σ_e = σ_e_photon,
                    sf = sf_kernel, sw = T(scatter_total_weight), do_sc = has_scatter
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
                    sf = sf_kernel, sw = T(scatter_total_weight), do_sc = has_scatter
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

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Calibration (intensity → air scan → log)
    # ═══════════════════════════════════════════════════════════════════════
    eps = T(1.0e-10)

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

    # ─── Fill-factor calibration (auto-correct, matching the clinical air-scan)
    # ── apply_fill_factor! injected +(−log(ff_eff)) ≥ 0 to every log-line-
    # integral up in STEP 2; the air-scan calibration above does NOT roll
    # ff_eff into its reference (real scanners absorb detector gain into the
    # air-scan reference, BS does not yet do that at workspace-construction
    # time).  Subtract the offset here so the returned sinogram is the same
    # post-calibration log-line-integral a real scanner would output.
    # No-op when use_fill_factor = false (then config.fill_factor === nothing)
    # or ff_eff ≈ 1.
    if config.fill_factor !== nothing
        ff_eff = T(effective_fill_factor(config.fill_factor))
        if !(ff_eff ≈ one(T))
            ff_log = T(log(ff_eff))
            let sino = ws.sinogram, ff_log = ff_log
                AK.foreachindex(sino) do idx
                    sino[idx] += ff_log
                end
            end
        end
    end

    # BHC is decoupled — applied at notebook level
    return nothing
end

# =============================================================================
# System Noise Floor (dose-independent)
# =============================================================================

"""
    add_system_noise_floor!(vol, sigma_hu; seed=nothing) -> vol

Add IID zero-mean Gaussian noise of standard deviation `sigma_hu` to a
reconstruction volume in the HU domain, in place.  Returns `vol`.

This is an **empirical, post-reconstruction noise-floor model** — *not* a port
of CatSim, TIGRE, or any other simulator.  CatSim and TIGRE both stop at the
sinogram-domain pipeline (Poisson + electronic noise on counts); neither
adds an HU-domain noise floor.

# Why a separate dose-independent floor?

Real scanners exhibit a noise component that *does not* scale as 1/√dose:
imperfect scatter correction residuals, detector calibration drift, and
ring-correction residuals.  In low-dose regimes this floor can rival the
quantum (Poisson) noise that the sinogram-domain simulator already produces.

Combining quadratically: σ_total = √(σ_quantum² + σ_floor²).  Adding the
floor in image space is correct as long as the floor is approximately white
in HU after reconstruction — empirically true for clinical scanners at the
spatial frequencies that matter for soft-tissue contrast.

!!! danger "Never use this to compare reconstruction algorithms"
    Noise added here bypasses the reconstruction operator, so it is *identical*
    in an FBP and an iterative recon of the same sinogram.  Any FBP-vs-IR noise
    comparison that includes this floor understates the iterative reduction —
    at σ = 28 HU it reported Hybrid IR at 1.6 % noise reduction where the true
    figure was ~26 %.  Reserve this helper for effects that genuinely survive
    reconstruction.

!!! danger "DAS / electronic noise does NOT belong here"
    A/D and DAS electronic noise enters the **counts**, before the log
    transform, and reconstruction therefore acts on it.  `simulate!` already
    models it from `scanner.electronic_noise` (electrons) — see the
    `λ_noisy += σ_e · randn()` term in the EICT noise kernel.  Set that field
    rather than adding an equivalent HU-domain floor here.  (At standard dose
    its contribution is small: 5000 e⁻ moves a soft-tissue σ from 22.2 to
    22.5 HU at 120 kVp / 250 mA.  It matters at low dose and behind dense
    anatomy, which is exactly where it must be allowed to propagate through
    the reconstruction.)

!!! danger "Do not apply to PCCT"
    A photon-counting detector discards charge pulses below its lowest energy
    threshold, which is precisely the mechanism that removes electronic noise —
    the headline PCCT advantage.  Its residual spectral distortion (charge
    sharing, K-escape, electronic smearing) is already encoded in the
    Monte-Carlo detector response matrix on the forward path, and pile-up /
    count-loss are modeled explicitly.  Adding a white HU floor to a PCCT recon
    would erase the advantage the simulation exists to demonstrate.

For physical background see Kalender, *Computed Tomography*, 3rd ed.,
Wiley 2011, Ch. 4 (noise sources in CT) and Hsieh, *Computed Tomography*,
3rd ed., SPIE Press 2015 (electronic-noise floor at low dose).  For a
PCCT-specific characterization see Leng et al., *IEEE TMI* 37(11), 2018.

# Choosing `sigma_hu`

Empirical, scanner-specific.  Notebooks 01/05/11 use **σ ≈ 28 HU**, which
matches a soft-tissue ROI std measured on a clinical GE Apex Elite (120 kVp,
standard-dose abdominal protocol) after FBP reconstruction at 5 mm slice
thickness.  Re-measure for other scanners / kernels / slice thicknesses.

Note that σ ≈ 28 HU is a *total* soft-tissue σ, not a residual-only floor, so
using it here double-counts the quantum noise `simulate!` has already added.
Notebook 02 instead leaves this helper off and lets `scanner.electronic_noise`
carry the DAS term through the reconstruction.

# Arguments
- `vol::AbstractArray{T}`: reconstruction volume in HU (mutated in place).
- `sigma_hu::Real`: noise-floor standard deviation in HU.
  `sigma_hu ≤ 0` → no-op; `vol` is returned unchanged.

# Keyword Arguments
- `seed::Union{Int,Nothing}=nothing`: when an `Int` is given a private
  `MersenneTwister(seed + 7919)` is used so the per-volume realization is
  bit-reproducible across runs without disturbing `Random.default_rng()`.
  When `nothing`, draws from the global RNG.

# Returns
The mutated `vol` (same object — `===` to the input).
"""
function add_system_noise_floor!(vol::AbstractArray{T}, sigma_hu::Real; seed::Union{Int, Nothing} = nothing) where {T}
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
function resolve_source_spectrum_without_bowtie(sim_opts::SimOptions, protocol::CTProtocol; scanner = nothing)
    # Physics-based: raw IPEM spectrum + Beer-Lambert filtering
    e, w = load_spectrum_unfiltered(Int(protocol.kVp); anode_angle = protocol.anode_angle)

    # Build filter list: scanner's built-in flat filter + protocol extras
    filters = Tuple{String, Float64}[]
    if scanner !== nothing && scanner.flat_filter_thickness > 0
        push!(filters, (String(scanner.flat_filter_material), Float64(scanner.flat_filter_thickness)))
    end
    append!(filters, protocol.additional_filters)

    # Apply Beer-Lambert filtering + inverse-square-law distance scaling
    sdd_mm = scanner !== nothing ? Float64(scanner.source_to_detector) : 750.0
    e, w = filter_spectrum(e, w; filters = filters, sdd_mm = sdd_mm)

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

# Reference (algorithm port from XCIST / CatSim, GE Research)
- `gecatsim/pyfiles/Xray_Filter.py:bowtie_filter()` — per-ray attenuation for
  the 4-material `{Al, graphite, Cu, Ti}` stack:
  `B(α, E) = exp(−Σ_m μ_m(E) · t_m(α))` with `t_m` interpolated from the
  vendor bowtie thickness file by fan angle and divided by `cos(alpha)` for
  the cone-angle path correction.  BasisSim's `compute_bowtie_attenuation_spectral`
  in `src/source/bowtie_filter.jl` implements the same Beer-Lambert sum for
  the same four materials.
- `gecatsim/pyfiles/Resample_Spectrum_Bowtie_FlatFilter.py` — the per-pixel
  spectrum × bowtie × flat-filter product `spec.netIvec = spec.Ivec ·
  bowtie.transVec · FiltrationTransVec`.  We do the spectrum × bowtie step
  here and the flat-filter step in `resolve_source_spectrum_without_bowtie`
  (so this primitive only handles the bowtie multiplication).

The Julia code is original; the algorithm and the `{Al, graphite, Cu, Ti}`
material composition are inherited from CatSim.

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
        @info "[apply_bowtie_to_spectrum$(isempty(label) ? "" : " (" * label * ")")] bowtie OFF → 1D centered spectrum"
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
        tot = 0.0f0
        for k in 1:n_E
            tot += w_pr[col, row, k]
        end
        inv_tot = 1.0f0 / max(tot, 1.0f-20)
        for k in 1:n_E
            w_pr[col, row, k] *= inv_tot
        end
    end
    mid_c = n_col ÷ 2 + 1
    mid_r = n_row ÷ 2 + 1
    mean_E_center = sum(Float64(e[k]) * w_pr[mid_c, mid_r, k] for k in 1:n_E)
    mean_E_edge1 = sum(Float64(e[k]) * w_pr[1, mid_r, k] for k in 1:n_E)
    @info "[apply_bowtie_to_spectrum$(isempty(label) ? "" : " (" * label * ")")] bowtie ON → per-ray 3D ŵ  [$(n_col) × $(n_row) × $(n_E)]"
    @info "  center-ray mean E = $(round(mean_E_center, digits = 1)) keV"
    @info "  edge-ray   mean E = $(round(mean_E_edge1, digits = 1)) keV   (Δ = $(round(mean_E_edge1 - mean_E_center, digits = 1)) keV ← bowtie hardening)"
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
    ŵ = apply_bowtie_to_spectrum(
        w_1d, e, scanner, geom, protocol;
        include_bowtie = include_bowtie, label = label
    )
    return e, ŵ
end

"""
    resolve_source_spectrum_full(
        sim_opts, protocol;
        scanner, geom, phantom = nothing,
        diagnostic = false, label = "",
    ) -> (e, ŵ)

**Full-physics effective spectrum** — the spectrum each detector ray actually
sees after every spectrum-shaping effect simulate! applies, normalized per
ray so `Σ_E ŵ[col, row, E] = 1`.

This composes `resolve_source_spectrum_without_bowtie` (tube spectrum × flat
filter × additional filters × inverse-square SDD scaling) with the same
`PhysicsConfig` simulate! builds (via `build_physics_config`), then multiplies
in the bowtie, the heel-effect spectrum, and the energy-dependent detector
efficiency `η(E)` — each conditionally on its `sim_opts.use_*` flag.

The output `ŵ` is what to feed to per-ray polychromatic inverters (Cong,
beam-hardening correction, etc.) when you want the inversion's forward model
to match the forward model `simulate!` actually applied.  Use this instead of
chaining `resolve_source_spectrum_with_bowtie` + manual heel + manual η.

# Composition
```
w_source(E) = resolve_source_spectrum_without_bowtie(sim_opts, protocol)
B(col,row,E) = compute_bowtie_attenuation_spectral(scanner.bowtie_filter, geom, E)
heel(col,row,E) = compute_heel_spectral(config.heel_effect, geom, E)         [if heel enabled]
η(E)        = compute_eid_efficiency_vector(config.detector_efficiency, E)   [if η enabled]

ŵ_raw(col,row,E) = w_source(E) · η(E) · B(col,row,E) · heel(col,row,E)
ŵ(col,row,E)     = ŵ_raw / Σ_E ŵ_raw
```

When `sim_opts.use_heel_effect = false` or `use_detector_efficiency = false`,
the corresponding factor collapses to 1 (via the `config.heel_effect ===
nothing` branch), so the function automatically follows whatever the
SimOptions config says — guaranteeing the inversion sees exactly what
simulate! applied.

# Arguments
- `sim_opts::SimOptions` — same struct passed to `simulate!`; its `use_*`
  flags gate which spectrum effects are baked in.
- `protocol::CTProtocol` — kVp / anode angle / additional filters.

# Keyword Arguments
- `scanner` — Scanner hardware definition.
- `geom` — Workspace geometry (`ws.geom` from a forward-projected sim);
  needed to assemble the per-(col, row) bowtie / heel maps.
- `phantom = nothing` — Forwarded to `build_physics_config` (only matters
  for scatter scaling, which is sinogram-domain and irrelevant here).
- `diagnostic = false` — When `true`, log the center-vs-edge mean energy
  to help spot bowtie / heel calibration drift.
- `label = ""` — Prepended to the diagnostic log line for multi-kVp runs.

# Returns
- `e::Vector` — Energy grid (keV).
- `ŵ::Array{Float32, 3}` — Per-ray effective spectrum, shape `[n_col, n_row, n_E]`,
  normalized per ray.

# Example
```julia
e_L, ŵ_L = BS.resolve_source_spectrum_full(
    sim_opts, protocol_low;
    scanner = scanner, geom = sim_low.geom, phantom = phantom,
)
e_H, ŵ_H = BS.resolve_source_spectrum_full(
    sim_opts, protocol_high;
    scanner = scanner, geom = sim_high.geom, phantom = phantom,
)
material_basis = (ŵ_L = ŵ_L, p_L = …, q_L = …, ŵ_H = ŵ_H, p_H = …, q_H = …)
```
"""
function resolve_source_spectrum_full(
        sim_opts::SimOptions,
        protocol::CTProtocol;
        scanner,
        geom,
        phantom::Union{Nothing, Phantom} = nothing,
        diagnostic::Bool = false,
        label::String = "",
    )
    # Step 1: 1D source spectrum (tube × flat filter × additional filters × SDD)
    e, w_1d = resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)

    # Step 2: identical PhysicsConfig that simulate!'s workspace ctor uses,
    # so `config.heel_effect` / `config.detector_efficiency` are populated
    # iff the corresponding sim_opts flag is true.
    config = build_physics_config(
        scanner, sim_opts, Float64.(e), Float64.(w_1d); phantom = phantom,
    )

    # Step 3: bowtie B[col, row, E]
    bowtie = resolve_bowtie_filter(scanner.bowtie_filter)
    B = compute_bowtie_attenuation_spectral(bowtie, geom, Float64.(e))   # [col, row, E]

    # Step 4: conditional heel(col, row, E) and η(E)
    heel = config.heel_effect !== nothing ?
        compute_heel_spectral(config.heel_effect, geom, Float64.(e)) :
        ones(Float64, size(B)...)
    η = config.detector_efficiency !== nothing ?
        compute_eid_efficiency_vector(config.detector_efficiency, Float64.(e)) :
        ones(Float64, length(e))

    # Step 5: w_source · η · B · heel, then per-pixel normalize
    w_norm = Float64.(w_1d) ./ sum(Float64.(w_1d))
    ŵ_raw = similar(B)
    @inbounds for k in 1:length(e)
        wη = w_norm[k] * η[k]
        for row in 1:size(B, 2), col in 1:size(B, 1)
            ŵ_raw[col, row, k] = wη * B[col, row, k] * heel[col, row, k]
        end
    end
    ŵ = Float32.(ŵ_raw ./ sum(ŵ_raw; dims = 3))

    if diagnostic
        mid_c = size(ŵ, 1) ÷ 2 + 1
        mid_r = size(ŵ, 2) ÷ 2 + 1
        mE_center = sum(Float64(e[k]) * ŵ[mid_c, mid_r, k] for k in eachindex(e))
        mE_edge   = sum(Float64(e[k]) * ŵ[1,     mid_r, k] for k in eachindex(e))
        tag = isempty(label) ? "" : " ($(label))"
        @info "effective spectrum$(tag) @ $(Int(protocol.kVp)) kVp — " *
              "center mean E = $(round(mE_center, digits = 1)) keV, " *
              "edge mean E = $(round(mE_edge, digits = 1)) keV " *
              "(Δ = $(round(mE_edge - mE_center, digits = 1)) keV)"
    end

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
A `PhysicsConfig` consumed by `simulate!` via the workspace pathway
(`create_workspace` / `create_eict_workspace` cache it on the workspace at
construction time).  Original BasisSim glue — not a CatSim port.  CatSim's
equivalent is the per-effect callback registry on `cfg.physics.*Callback`.
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

    # Detector efficiency (EICT only): two MC-LUT scintillators are supported —
    # GE Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ (:lumex) and Siemens UFC Gd₂O₂S:Pr,Ce
    # (:ufc). `detector_efficiency_mode` (:auto, :mc_lut, :beer_lambert)
    # toggles between the MC LUT and the analytical fallback. PCCT scanners go
    # through `pcct_forward_project` which encodes all detector physics
    # (charge sharing, fluorescence escape, pileup) in the MC DRM — they
    # don't consume this `PhysicsConfig.detector_efficiency` field, so we
    # skip it.
    if sim_opts.use_detector_efficiency && scanner.detector_type != :photon_counting
        material = scanner.detector_material
        de_mode = sim_opts.detector_efficiency_mode   # :auto, :mc_lut, :beer_lambert
        eff_mode = de_mode == :beer_lambert ? :beer_lambert : :mc_lut
        depth = scanner.detector_depth
        fill = scanner.fill_factor_row > 0 ? scanner.fill_factor_row : 0.9
        kwargs[:detector_efficiency] = if material in (:lumex, :Lumex, :LUMEX)
            detector_efficiency_gemstone(
                mode = eff_mode,
                thickness_mm = depth > 0 ? depth : 3.0,
                fill_factor = fill
            )
        elseif material in (:ufc, :UFC, :gd2o2s, :Gd2O2S)
            detector_efficiency_ufc(
                mode = eff_mode,
                thickness_mm = depth > 0 ? depth : 1.4,
                fill_factor = fill
            )
        else
            error(
                "Unsupported EICT detector material: $material — supported: " *
                    ":lumex (GE Gemstone MC LUT), :ufc (Siemens UFC Gd₂O₂S MC LUT)"
            )
        end
    end

    # Scatter: use geometry-aware model scaled for this scanner and phantom size
    # If phantom is provided, estimate diameter from mask for size-aware scatter scaling
    phantom_diameter_cm = if phantom !== nothing && (sim_opts.use_scatter || sim_opts.use_pcct_scatter)
        voxel_size_mm = phantom.voxel_size .* 10.0
        estimate_phantom_diameter_cm(phantom.mask, voxel_size_mm)
    else
        nothing
    end

    if sim_opts.use_scatter || sim_opts.use_pcct_scatter
        kwargs[:scatter] = geometry_aware_scatter_model(scanner; phantom_diameter_cm = phantom_diameter_cm)
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
    reconstruct!(ws::FDKReconWorkspace, sinogram, geom) -> ws.volume

Zero-allocation FDK reconstruction using pre-allocated workspace buffers.

The four-step pipeline (copy → filter → backproject → FOV-mask) and the
supported filter set (`:ram_lak`, `:shepp_logan`, `:cosine`, `:hamming`,
`:hann`, plus the BasisSim-specific `:standard`/`:soft`/`:bone` clinical
kernels) follow Feldkamp-Davis-Kress (1984) and the TIGRE Toolbox
implementation.

Each step is a separate exported primitive (`filter_sinogram!`,
`backproject!`, `apply_fov_mask!`) so callers can rebuild the same pipeline
piecewise (the hybrid-IR path reuses all three).

The filter kernel, kernel size (driven by `cutoff`), and output volume size
are all locked at workspace creation time (`create_fdk_recon_workspace`).
There are no runtime overrides on this hot path — re-create the workspace
to change them.

# Reference (algorithm port from TIGRE Toolbox)
- Feldkamp LA, Davis LC, Kress JW. "Practical cone-beam algorithm." J Opt
  Soc Am A 1(6):612–619 (1984).  doi:10.1364/JOSAA.1.000612
- TIGRE: `MATLAB/Algorithms/FDK.m` — same `filtering → backprojection`
  flow, same filter options.  Cone-beam backprojection is in
  `Common/CUDA/voxel_backprojection.cu` ("CUDA function for backprojection
  using FDK weights for CBCT") and BasisSim's
  `src/reconstruction/core/backprojection.jl` is the Julia/Metal/CUDA port
  of the same voxel-driven cone-beam scheme.
- Note: this path does NOT use the Siddon (1985) ray-tracing algorithm.
  Siddon is forward-projection only — see `siddon_forward_project!`.  FDK
  reconstruction is voxel-driven backprojection, a different algorithm.

# Arguments
- `ws::FDKReconWorkspace{T}` — pre-allocated buffers from
  `create_fdk_recon_workspace(sinogram, geom, volume_size; filter, cutoff)`.
- `sinogram::AbstractArray{T,3}` — `(n_col, n_row, n_view)` log line-integral
  sinogram, on the same backend as `ws` (CPU/Metal/CUDA/AMDGPU).
- `geom::CTGeometry` — cone-beam geometry.

# Returns
The mutated `ws.volume` (same object — `===` to the workspace field).
"""
function reconstruct!(
        ws::FDKReconWorkspace{T},
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry,
    ) where {T <: AbstractFloat}

    if is_helical(geom)
        # Helical → rebinned WFBP chain (Stierstorfer family), reusing the
        # workspace buffers: ws.filtered holds the rebinned/filtered data.
        # ws.filter_kernel was built with spacing geom.pixel_size == Δt, so it
        # is the correct parallel ramp kernel as-is.
        _, Δt = _wfbp_rebin!(ws.filtered, sinogram, geom)
        filter_sinogram!(
            ws.filtered, geom;
            ws_conv_scratch = ws.conv_scratch,
            ws_filter_kernel = ws.filter_kernel,
            apply_cosine = false, ray_spacing = Δt
        )
        fill!(ws.volume, zero(T))
        _wfbp_backproject!(ws.volume, ws.filtered, geom, T(Δt))
        apply_fov_mask!(ws.volume, geom)
        return ws.volume
    end

    # Step 1: Copy sinogram into filtering scratch buffer
    copyto!(ws.filtered, sinogram)

    # Step 2: Filter in-place (cosine weighting + ramp convolution).
    # Filter and kernel size are baked into ws.filter_kernel at workspace
    # creation time; filter_sinogram! short-circuits to ws_filter_kernel.
    filter_sinogram!(
        ws.filtered, geom;
        ws_conv_scratch = ws.conv_scratch,
        ws_filter_kernel = ws.filter_kernel
    )

    # Step 3: Backproject into pre-allocated volume
    fill!(ws.volume, zero(T))
    backproject!(
        ws.volume, ws.filtered, geom;
        weighted = true,
        ws_source_positions = ws.bp_source_positions,
        ws_detector_centers = ws.bp_detector_centers,
        ws_detector_u = ws.bp_detector_u,
        ws_detector_v = ws.bp_detector_v
    )

    # Step 4: Mask outside FOV (clinical convention)
    apply_fov_mask!(ws.volume, geom)

    return ws.volume
end

# =============================================================================
# reconstruct!() — Zero-allocation Hybrid IR reconstruction hot path
# =============================================================================

function _hir_seed_work!(work::AbstractArray{T, 3}, init::AbstractArray{T, 3}, output_z) where {T}
    work === init && return work
    nx = Int32(size(work, 1))
    ny = Int32(size(work, 2))
    nz = Int32(size(init, 3))
    z0 = Int32(first(output_z))
    backend = AK.get_backend(work)
    AK.foreachindex(work, backend) do idx
        idx0 = Int32(idx - 1)
        i = (idx0 % nx) + Int32(1)
        j = ((idx0 ÷ nx) % ny) + Int32(1)
        kw = (idx0 ÷ (nx * ny)) + Int32(1)
        ko = clamp(kw - z0 + Int32(1), Int32(1), nz)
        work[idx] = init[i, j, ko]
    end
    return work
end

function _hir_extract_output!(output::AbstractArray{T, 3}, work::AbstractArray{T, 3}, output_z) where {T}
    output === work && return output
    nx = Int32(size(output, 1))
    ny = Int32(size(output, 2))
    z0 = Int32(first(output_z))
    backend = AK.get_backend(output)
    AK.foreachindex(output, backend) do idx
        idx0 = Int32(idx - 1)
        i = (idx0 % nx) + Int32(1)
        j = ((idx0 ÷ nx) % ny) + Int32(1)
        k = (idx0 ÷ (nx * ny)) + Int32(1)
        output[idx] = work[i, j, z0 + k - Int32(1)]
    end
    return output
end

"""
    reconstruct!(ws::HIRReconWorkspace, sinogram, geom; init_volume=nothing, air_reference=nothing) -> ws.volume

Zero-allocation Hybrid IR reconstruction using pre-allocated workspace buffers.

Pipeline: FDK initialization → Ordered-Subsets Penalized Weighted Least Squares
(OS-PWLS) with Huber edge-preserving regularization on a 6-connected neighborhood.

# Reference (algorithm — Fessler / U-Michigan school, NOT a TIGRE port)

TIGRE has SART, OS-SART, MLEM, OSEM, CGLS, FISTA, ASD-POCS — but no PWLS and
no Huber prior.  The HIR algorithm is from a different family entirely.  The
canonical reference implementation is Jeff Fessler's MIRT (Michigan Image
Reconstruction Toolbox); each component below is matched 1-to-1 with a MIRT
source file.

- **PWLS framework for transmission CT**:
  Sauer K, Bouman C. "A local update strategy for iterative reconstruction
  from projections." IEEE Trans Signal Process 41(2):534–548 (1993).
  Fessler JA. "Penalized weighted least-squares image reconstruction for
  positron emission tomography." IEEE Trans Med Imaging 13(2):290–300
  (1994).  doi:10.1109/42.293921
- **Ordered-Subsets PWLS** (the OS path used here):
  Erdoğan H, Fessler JA. "Ordered subsets algorithms for transmission
  tomography." Phys Med Biol 44(11):2835–2851 (1999).
  doi:10.1088/0031-9155/44/11/311
  → MIRT: `transmission/tpl_os_sps.m` (T-PL-OS-SPS — "Transmission
  Penalized-Likelihood Ordered-Subsets Separable Paraboloidal Surrogates").
  Our OS loop structure (stat-weight init → outer epoch → inner subset
  loop → forward + backproject + V_inv-preconditioned update) mirrors
  `tpl_os_sps`'s outer iteration; we use a SIRT-style `V_inv` preconditioner
  in place of MIRT's per-pixel SPS curvatures (`denom`) — same algorithm
  class, fixed-curvature variant.
- **Huber edge-preserving penalty + gradient**:
  Huber PJ. "Robust estimation of a location parameter." Ann Math Stat
  35(1):73–101 (1964).
  → MIRT: `penalty/huber_pot.m` (potential function) and `huber_dpot.m`
  (potential derivative).  `compute_huber_gradient!` in
  `src/reconstruction/ir/utils.jl` is a direct GPU port of `huber_dpot`
  applied to a 6-connected finite-difference stencil (`penalty/Cdiffs.m`).
- **Volume update rule**: SIRT-style multiplicative update with `V_inv` image-
  domain weights — Andersen & Kak SART/SIRT framework, generalized inside
  the same MIRT family (`general/qpwls_*`, `transmission/tpl_*`).

The `strength`-keyed `HIRParams` lookup table targets the noise-reduction
behavior of vendor IR (GE ASIR-V, Siemens SAFIRE, Philips iDose-4, Canon
AIDR 3D) as reported in the clinical validation literature (Geyer et al.
*Radiology* 2015; Willemink & Noël *Eur Radiol* 2019; Ghetti et al.
PMC5714520).  Those papers describe the *target performance*; they do not
provide vendor algorithms (which are proprietary).  Our underlying PWLS+OS
solver is the open-literature substitute.

# Implementation
The workspace pre-allocates everything the OS-PWLS loop needs:
ordered subsets (`ws.subsets`, `ws.subset_geometries`), per-subset GPU
geometry arrays and angle maps, the folded data-weight buffer
(`data_weights` = `W_proj` ⊙ statistical weights), image-domain weights
(`V_inv`), regularization gradient (`reg_grad`), and the forward-projection
buffer.  Each subset gathers its rays out of the full sinogram through
`ws.subset_angle_idx[s]` rather than staging them into a copy.

At `strength = 0` (`nepochs == 0`) the PWLS loop is skipped entirely and the
FDK initialization is returned — i.e. plain FBP.

The filter kernel, kernel size, and output volume size are all locked at
workspace creation time — no runtime overrides on this hot path.

# Keyword Arguments
- `init_volume`: Optional warm-start volume (skip FDK init step).  If
  provided, must be the same shape as `ws.volume`. For axial HIR, its terminal
  slices are replicated only into the private computational halo; the returned
  grid and caller-owned input are unchanged.
- `air_reference`: Optional 2D array `[n_cols, n_rows]` of bowtie air
  reference values.  When provided, statistical weights are scaled by the
  air reference to account for position-dependent noise from the bowtie
  filter (edge pixels have fewer counts → more noise → lower weight).
"""
function reconstruct!(
        ws::HIRReconWorkspace{T},
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry;
        init_volume::Union{Nothing, AbstractArray{T, 3}} = nothing,
        air_reference::Union{Nothing, AbstractArray} = nothing
    ) where {T <: AbstractFloat}

    params = ws.params
    model_volume = ws.work_volume
    model_geom = ws.work_geom

    # ─── Step 1: FDK initialization (or warm-start from provided volume) ───
    if init_volume === nothing
        if is_helical(model_geom)
            # Helical FDK init → rebinned WFBP (see FDKReconWorkspace path)
            _, Δt_h = _wfbp_rebin!(ws.filtered, sinogram, model_geom)
            filter_sinogram!(
                ws.filtered, model_geom;
                ws_conv_scratch = ws.conv_scratch,
                ws_filter_kernel = ws.filter_kernel,
                apply_cosine = false, ray_spacing = Δt_h
            )
            fill!(model_volume, zero(T))
            _wfbp_backproject!(model_volume, ws.filtered, model_geom, T(Δt_h))
        else
        copyto!(ws.filtered, sinogram)
        filter_sinogram!(
            ws.filtered, model_geom;
            ws_conv_scratch = ws.conv_scratch,
            ws_filter_kernel = ws.filter_kernel
        )
        fill!(model_volume, zero(T))
        backproject!(
            model_volume, ws.filtered, model_geom;
            weighted = true,
            ws_source_positions = ws.geom_source_positions,
            ws_detector_centers = ws.geom_detector_centers,
            ws_detector_u = ws.geom_detector_u,
            ws_detector_v = ws.geom_detector_v
        )
        end
    else
        size(init_volume) == size(ws.volume) || throw(DimensionMismatch(
            "init_volume has size $(size(init_volume)); expected $(size(ws.volume))"))
        copyto!(ws.volume, init_volume)
        _hir_seed_work!(model_volume, ws.volume, ws.output_z)
    end

    # ─── Step 2: OS-PWLS refinement with Huber regularization ───
    # Erdoğan & Fessler 1999 ordered-subsets PWLS for transmission CT.
    λ = T(params.lambda)
    λ_relax = T(params.relaxation)
    δ = T(params.huber_delta)
    backend = AK.get_backend(model_volume)

    nepochs = params.nepochs
    n_subsets = params.n_subsets
    subset_scale = T(n_subsets)

    # strength = 0 ⇒ no PWLS refinement at all: the FDK init IS the result.
    # Bail out before the weight kernels so a 0 % request costs a plain FBP.
    if nepochs == 0
        _hir_extract_output!(ws.volume, model_volume, ws.output_z)
        apply_fov_mask!(ws.volume, geom)
        return ws.volume
    end

    # The iterative system matrix represents the circular reconstruction FOV,
    # not the enclosing square array. Keep that support constraint active
    # throughout HIR; masking only at return lets exterior corner estimates
    # feed back through A*x and leaves a circular boundary ring.
    apply_fov_mask!(model_volume, model_geom; sentinel_μ = zero(T))

    # Initialize statistical weights: w = air_ref(col,row) × exp(-y)
    # air_ref accounts for bowtie-modulated I0 (edge pixels → fewer counts → lower weight)
    if air_reference !== nothing
        let sw = ws.data_weights, ε = T(1.0e-6), aref = air_reference,
                nc = Int32(size(sinogram, 1)), nr = Int32(size(sinogram, 2))
            AK.foreachindex(sw, backend) do idx
                idx_0 = Int32(idx - 1)
                col = (idx_0 % nc) + Int32(1)
                row = ((idx_0 ÷ nc) % nr) + Int32(1)
                ref_idx = col + (row - Int32(1)) * nc
                y_val = sinogram[idx]
                # clamp below at 0: a noisy negative-y air ray would get weight
                # exp(+|y|) and dominate the subset update (audit A4)
                y_clipped = clamp(y_val, T(0), T(10))
                sw[idx] = T(aref[ref_idx]) * exp(-y_clipped) + ε
            end
        end
    else
        let sw = ws.data_weights, ε = T(1.0e-6)
            AK.foreachindex(sw, backend) do idx
                y_val = sinogram[idx]
                # clamp below at 0: a noisy negative-y air ray would get weight
                # exp(+|y|) and dominate the subset update (audit A4)
                y_clipped = clamp(y_val, T(0), T(10))
                sw[idx] = exp(-y_clipped) + ε
            end
        end
    end

    # Fold the projection weights into the statistical weights once.  The
    # residual is `W_proj ⊙ stat_w ⊙ (y - Ax)` and `a * b * c` associates left,
    # so pre-multiplying `W_proj ⊙ stat_w` reproduces the same roundings while
    # letting each subset read a single array.
    let dw = ws.data_weights, wp = ws.W_proj
        AK.foreachindex(dw, backend) do idx
            dw[idx] = wp[idx] * dw[idx]
        end
    end

    for epoch in 1:nepochs
        # Compute Huber gradient ONCE per epoch (not per sub-iteration)
        compute_huber_gradient!(ws.reg_grad, model_volume, δ)

        for (s, angle_indices) in enumerate(ws.subsets)
            n_sub = length(angle_indices)
            geom_s = ws.subset_geometries[s]

            # Forward project with subset geometry → subset_Ax_buf.
            # Uses ws.projector so A·x matches the projector that made the data.
            ax_view = view(ws.subset_Ax_buf, :, :, 1:n_sub)
            fill!(ax_view, zero(T))
            _project_mono!(
                ws.projector,
                ax_view, model_volume, geom_s;
                ws_source_positions = ws.subset_geom_source_positions[s],
                ws_detector_centers = ws.subset_geom_detector_centers[s],
                ws_detector_u = ws.subset_geom_detector_u[s],
                ws_detector_v = ws.subset_geom_detector_v[s]
            )

            # Compute weighted residual in-place:
            # Ax_s = (W ⊙ stat_w)_s ⊙ (sino_s - Ax_s)
            # The subset's rays are gathered straight out of the full arrays via
            # the angle map, so no staging copy is needed.
            let ax = ax_view, sino = sinogram, dw = ws.data_weights,
                    aidx = ws.subset_angle_idx[s],
                    nc = Int32(size(sinogram, 1)), nr = Int32(size(sinogram, 2))

                AK.foreachindex(ax, backend) do idx
                    idx_0 = Int32(idx - 1)
                    col = (idx_0 % nc) + Int32(1)
                    row = ((idx_0 ÷ nc) % nr) + Int32(1)
                    k = (idx_0 ÷ (nc * nr)) + Int32(1)   # slot within the subset
                    a = aidx[k]                          # global view index
                    residual = sino[col, row, a] - ax[idx]
                    ax[idx] = dw[col, row, a] * residual
                end
            end

            # Backproject weighted residual → correction
            fill!(ws.correction, zero(T))
            backproject!(
                ws.correction, ax_view, geom_s;
                weighted = false,
                ws_source_positions = ws.subset_geom_source_positions[s],
                ws_detector_centers = ws.subset_geom_detector_centers[s],
                ws_detector_u = ws.subset_geom_detector_u[s],
                ws_detector_v = ws.subset_geom_detector_v[s]
            )

            # SIRT-style update with subset scaling:
            # x += λ_relax * V_inv * (n_subsets * correction) - λ * V_inv * reg_grad
            # Audit A1: V_inv ≈ 1/n_views makes the raw reg step scale as
            # λ/n_views while the data step is n_views-invariant (subset_scale
            # cancels) — so "strength" would mean different smoothing per
            # protocol.  Scale the reg term by n_views/1000 to make it
            # protocol-invariant (bands anchored at ~1000-view protocols).
            reg_views_scale = T(length(geom.angles)) / T(1000)
            let vol = model_volume, vinv = ws.V_inv, corr = ws.correction,
                    rg = ws.reg_grad, ss = subset_scale, rvs = reg_views_scale,
                    nx = Int32(size(model_volume, 1)), ny = Int32(size(model_volume, 2)),
                    dx = T(model_geom.fov[1]) / T(size(model_volume, 1)),
                    dy = T(model_geom.fov[2]) / T(size(model_volume, 2)),
                    radius_sq = T(min(model_geom.fov[1], model_geom.fov[2]) / 2)^2,
                    half = T(0.5)

                AK.foreachindex(vol, backend) do idx
                    idx0 = Int32(idx - 1)
                    ix = (idx0 % nx) + Int32(1)
                    iy = ((idx0 ÷ nx) % ny) + Int32(1)
                    x = (T(ix) - half - T(nx) / T(2)) * dx
                    y = (T(iy) - half - T(ny) / T(2)) * dy
                    if x * x + y * y > radius_sq
                        vol[idx] = zero(T)
                    else
                        data_update = λ_relax * vinv[idx] * ss * corr[idx]
                        reg_update = λ * rvs * vinv[idx] * rg[idx]
                        vol[idx] += data_update - reg_update
                    end
                end
            end
        end
    end

    # Audit A6: match the FDK path's clinical convention (outside-FOV corners
    # otherwise retain FDK-init ringing and skew volume statistics).
    _hir_extract_output!(ws.volume, model_volume, ws.output_z)
    apply_fov_mask!(ws.volume, geom)

    return ws.volume
end
