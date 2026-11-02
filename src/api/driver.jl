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

Expected primary photon count per detector pixel per view (incident; no
detector response applied: the detector's response per incident photon multiplies it at the
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

function _capture_pcct_raw_counts(
        bins::Vector{<:AbstractArray{T, 3}},
        I0::AbstractArray{T, 3},
    ) where {T <: AbstractFloat}
    length(bins) == size(I0, 3) ||
        throw(DimensionMismatch("one I0 plane is required per PCCT bin"))
    captured = [similar(bin) for bin in bins]
    n_rays = Int32(size(I0, 1) * size(I0, 2))
    for (b, bin_sino) in enumerate(bins)
        let out = captured[b], input = bin_sino, i0 = I0, m = n_rays, off = Int32(b - 1) * n_rays
            AK.foreachindex(input) do idx
                out[idx] = i0[(Int32(idx - 1) % m) + Int32(1) + off] * exp(-input[idx])
            end
        end
    end
    return captured
end


# The photon-counting detection of `simulate!`: the expected counts of every native dexel (the
# projection's native counts scaled, group by group, to the binned bins, which carry the focal-spot
# blur and the scatter), their counting noise (`pcct_count!`), their pile-up at each dexel's own
# rate, and the digital sum to pixels. Writes the bins as `-log(max(N, 1) / I0)` and, when given,
# the summed counts before that floor into `raw`.
function _pcct_detect!(ws::PCCTWorkspace, bins, use_noise::Bool, seed, raw)
    nb = length(bins)
    bf = ws.native_geom === nothing ? 1 : ws.native_geom.n_cols ÷ ws.geom.n_cols
    I0 = ws.I0_cpu
    expected_binned = [I0[:, :, b] .* exp.(-Float64.(Array(bins[b]))) for b in 1:nb]
    expected = if bf == 1
        expected_binned
    else
        map(1:nb) do b
            native = Float64.(Array(ws.native_bins[b]))
            target = expected_binned[b]
            for v in axes(native, 3), r in axes(target, 2), c in axes(target, 1)
                group = view(native, ((c - 1) * bf + 1):(c * bf), ((r - 1) * bf + 1):(r * bf), v)
                total = sum(group)
                total > 0 ? (group .*= target[c, r, v] / total) : (group .= target[c, r, v] / bf^2)
            end
            native
        end
    end
    if ws.defects !== nothing                       # each dexel's gain and threshold offsets
        _apply_pcct_defects!(expected, bf == 1 ? I0 : Array{Float64}(ws.I0_native), ws.defects)
    end
    counts = use_noise ? pcct_count!([similar(e) for e in expected], expected, ws.noise_classes; seed) : expected
    if ws.pileup && ws.pileup_S !== nothing
        pileup_counts!(counts, bf == 1 ? I0 : Array{Float64}(ws.I0_native), ws.pileup_S, ws.pileup_rates, ws.pileup_rate_air)
    end
    nr = ws.noise_reduction
    for b in 1:nb
        N = bf == 1 ? counts[b] : _digital_sum(counts[b], bf)
        use_noise && nr > 0 && (N .= expected_binned[b] .+ (1 - nr) .* (N .- expected_binned[b]))
        raw === nothing || copyto!(raw[b], eltype(raw[b]).(N))
        copyto!(bins[b], eltype(bins[b]).(.-log.(max.(N, 1.0) ./ I0[:, :, b])))
    end
    return bins
end

# A 2-D digital sum of `bf × bf` native dexels per pixel, view by view.
function _digital_sum(native::AbstractArray{<:Real, 3}, bf::Integer)
    nc, nr = size(native, 1) ÷ bf, size(native, 2) ÷ bf
    out = zeros(Float64, nc, nr, size(native, 3))
    for v in axes(native, 3), r in 1:nr, c in 1:nc
        out[c, r, v] = sum(@view native[((c - 1) * bf + 1):(c * bf), ((r - 1) * bf + 1):(r * bf), v])
    end
    return out
end

"""
    simulate!(ws::PCCTWorkspace, phantom, protocol, sim_opts;
              capture_raw_counts=true, report_dose=true, dose_kwargs=(;))

Run PCCT simulation using pre-allocated workspace buffers for zero allocations
on the second and later calls (the first call may JIT-allocate).

All setup data (geometry, spectrum, physics config, detector, spectral response,
materials) is baked into the workspace at `create_workspace()` time.  The PCCT
detector model **always** applies the MC-LUT detector response matrix
(`compute_mc_drm`); `pcct_forward_project` builds it when no precomputed matrix
is passed, and there is no ideal-binning path.

Pulse pileup is on by default and toggleable via `PCCTScanner(; pileup)`; it is corrected
like a scanner corrects it, outside: [`calibrate_pileup`](@ref) → [`apply_pileup!`](@ref), as the
dexels' defects are by [`calibrate_flat_field`](@ref) → [`apply_flat_field!`](@ref). With
`SimOptions(use_scatter = true)` the scanner's Geant4 scatter kernels ([`scatter_kernels`](@ref))
add the patient's scatter to every bin before the noise; the scatter correction
(`PCCTScanner(; scatter_correction)`) subtracts the scatter estimated from the recorded counts
alone (the patient taken as water), inside `simulate!`. Bin combination,
water BHC and reconstruction are the caller's: the returned per-bin sinograms are
log-transmissions against the per-ray air response `I0_bins` (`[n_cols, n_rows, n_bins]`), which
is what [`spectral_basis`](@ref) and [`vmi_pipeline`](@ref) take.

# Returns
- `pcct_sino` — `EnergyResolvedSinogram` (per-bin log line integrals).
- `I0_bins`   — the air response of every ray and bin, `[n_cols, n_rows, n_bins]` (per ray, not
  a per-bin scalar); `I0_bins[:, :, b] .* exp.(-bin_b)` recovers the recorded counts.
- `dose`      — the acquisition's [`DoseReport`](@ref), or `nothing` with `report_dose = false`
  (`dose_kwargs` reaches [`compute_dose`](@ref)).
- `raw_counts` — captured by default; independent per-bin arrays of each pixel's counts before
  the pile-up and scatter corrections: the digital sum of its native dexels' counts. With noise
  on and pile-up off they are integers, true zeros kept (the floor at 1 applies only to the
  log-domain sinograms); pile-up (`S × counts` per native dexel) makes them fractional; with noise
  off they are the expected counts.

The noise is drawn per native dexel, photon class by photon class (`pcct_count!`): each
dexel Poisson, neighbours covarying through the photons they share, so a 2 × 2 binned pixel's
variance exceeds its mean. Views are drawn in parallel from streams seeded by `(seed, view)`.

Capturing raw counts allocates one additional full-size array per energy bin;
pass `capture_raw_counts=false` only when memory-constrained.
"""
function simulate!(
        ws::PCCTWorkspace{T},
        phantom,
        protocol::CTProtocol,
        sim_opts::SimOptions = SimOptions(),
        ;
        capture_raw_counts::Bool = true,
        report_dose::Bool = true,
        dose_kwargs = (;),
        projection = nothing,
        keep_projection::Bool = false,
    ) where {T}
    geom = ws.geom
    energies = ws.energies
    weights = ws.weights
    config = ws.config
    pcct_detector = ws.pcct_detector
    mats = ws.mats

    I0 = ws.I0
    # Everything before the noise is deterministic (see the EICT `simulate!`): `keep_projection`
    # returns it and `projection` re-uses it for another noise draw, bit-identically.
    pcct_sino = nothing
    if projection === nothing
        # Forward projection with workspace buffers (including native-res path + tiled spectral),
        # integrated over each view's arc (`_integrate_views!`, the same pathway as the EICT chain)
        function load_pcct!(g)
            _load_view_geometry!(
                ws.geom_source_positions, ws.geom_detector_centers,
                ws.geom_detector_u, ws.geom_detector_v, g
            )
            ws.native_geom === nothing && return nothing
            ng = g === geom ? ws.native_geom : rotate_geometry(ws.native_geom, g.angles[1] - geom.angles[1])
            _load_view_geometry!(
                ws.native_geom_source_positions, ws.native_geom_detector_centers,
                ws.native_geom_detector_u, ws.native_geom_detector_v, ng
            )
            return nothing
        end
        projected = Ref{Any}(nothing)
        _integrate_views!(geom, sim_opts, load_pcct!) do s_view, g_view
            ng_view = ws.native_geom === nothing || g_view === geom ? ws.native_geom :
                rotate_geometry(ws.native_geom, g_view.angles[1] - geom.angles[1])
            projected[] = pcct_forward_project(
                phantom.mask, g_view, pcct_detector;
                energies = energies, weights = weights,
                materials = mats,
                ws_bins = ws.bins, ws_μ_volume = ws.μ_volume, ws_sino_buf = ws.sino_buf,
                ws_scratch = ws.scratch,
                ws_thresholds_T = ws.thresholds_T,
                ws_R = ws.R, ws_R_energies = ws.R_energies,
                ws_I0 = ws.I0,
                ws_μ_lut_cpu = ws.μ_lut_cpu, ws_μ_lut_gpu = ws.μ_lut_gpu,
                ws_μ_table = ws.μ_table,
                ws_source_positions = ws.geom_source_positions,
                ws_detector_centers = ws.geom_detector_centers,
                ws_detector_u = ws.geom_detector_u,
                ws_detector_v = ws.geom_detector_v,
                volume_extent = phantom.extent,
                # Native-resolution forward projection path (used when bf > 1)
                native_geom = ng_view,
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
                # the source transmission per ray (bowtie × heel) at the projection's resolution
                ws_source_spectral = ws.native_geom !== nothing ? ws.native_bowtie_spectral : ws.bowtie_spectral,
                projector = sim_opts.projector,
            )
            Tuple(projected[].bins)
        end
        pcct_sino = projected[]

        # ─── Tube-side focal-spot blur (per bin, BEFORE scatter/noise/pile-up) ───
        # The focal-spot penumbra is a tube-side effect common to both detector
        # chains; here it mirrors the EICT placement in `_apply_physics_no_noise!`
        # (a detector-plane convolution of the log line integrals). Placed before
        # the rate-dependent pile-up step so blurred local count rates feed it.
        # Note the blur acts at binned (not native-dexel) resolution. It runs when
        # `use_focal_spot = true` (the `SimOptions` default); pass `use_focal_spot = false`
        # to disable it. Detector lag is intentionally NOT applied on
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

        # ─── Patient scatter (before the noise): the Geant4 library on the patient's own rays,
        # each bin's expected counts (src/detector/scatter.jl) ───
        if ws.scatter !== nothing
            _scatter_forward!(ws.scatter, phantom)
            _pcct_scatter!(pcct_sino.bins, I0, ws.scatter, compute_detector_I0(geom, protocol, sum(weights)), 1)
        end
    else
        length(projection.bins) == length(ws.bins) && size(first(projection.bins)) == size(first(ws.bins)) ||
            throw(DimensionMismatch("the projection does not match this workspace's bins"))
        pcct_sino = EnergyResolvedSinogram([copy(b) for b in projection.bins], projection.thresholds_keV)
        if ws.native_bins !== nothing
            for (dst, src) in zip(ws.native_bins, projection.native_bins)
                copyto!(dst, src)
            end
        end
    end
    kept = keep_projection ? (
            bins = [copy(b) for b in pcct_sino.bins],
            native_bins = ws.native_bins === nothing ? nothing : [copy(b) for b in ws.native_bins],
            thresholds_keV = pcct_sino.thresholds_keV,
        ) : nothing

    # ─── Detection: every native dexel counts, piles up, and the counts are summed ───
    # The expected counts of each native dexel (primary, focal-spot blur and scatter included),
    # drawn photon class by photon class (`pcct_count!`: each dexel Poisson, neighbours covarying
    # through the photons they share), piled up at the dexel's own rate, then summed 2 × 2 after
    # counting (digital binning), so a binned pixel's variance exceeds its mean. `raw_counts` are
    # the summed counts before the log-domain floor at one and before any correction.
    raw_counts = capture_raw_counts ? [similar(b) for b in pcct_sino.bins] : nothing
    _pcct_detect!(ws, pcct_sino.bins, sim_opts.use_noise, sim_opts.seed, raw_counts)

    # ─── Scatter correction (`scatter_correction`): the scatter estimated from the recorded
    # counts alone, the patient taken as water, subtracted from each bin (src/detector/scatter.jl) ───
    if ws.scatter !== nothing && ws.scatter_correction
        plan = ws.scatter
        for b in eachindex(pcct_sino.bins)
            _reduce_footprints!(plan.measured, plan, pcct_sino.bins[b]; i0 = I0, plane = b, accumulate = b > 1)
        end
        photons = compute_detector_I0(geom, protocol, sum(weights))
        _scatter_estimate!(plan, Array(plan.measured) ./ photons)
        _pcct_scatter!(pcct_sino.bins, I0, plan, photons, -1)
    end

    # Bin combination, BHC and reconstruction are the caller's (spectral_basis / vmi_pipeline
    # take the per-bin sinograms and the per-ray I0 returned here).
    #
    # Returned fields:
    # - `pcct_sino`  : per-bin log-line-integral sinograms.  When pile-up is
    #                  on these are `-log(recorded / I0_truth)`; `apply_pileup!` with a
    #                  `calibrate_pileup` calibration recovers pile-up-free bins.
    # - `I0_bins`    : truth per-bin air baseline (DRM-weighted, pre-pileup).
    #                  `I0_b · exp(-bin) = recorded count` round-trip identity.
    # - `dose`       : `DoseReport` of this acquisition (CTDIvol, DLP, …) from the simulated
    #                  beam; `nothing` with `report_dose = false`.  Costs a cached Monte Carlo
    #                  (about a second) the first time a given beam is seen, then nothing.
    #
    # `dose_kwargs` reaches `compute_dose`'s keywords: the defaults are one tube and the 32 cm
    # body phantom, so a head protocol wants `(; phantom = :head16)` and a dual-source scanner
    # `(; n_tubes = 2)`.

    result = (
        pcct_sino = pcct_sino,
        I0_bins = ws.I0,            # per ray: [n_cols, n_rows, n_bins], the air response of each ray and bin
        dose = report_dose ? dose_report(ws, protocol; dose_kwargs...) : nothing,
    )
    result = capture_raw_counts ? merge(result, (; raw_counts)) : result
    return keep_projection ? merge(result, (; projection = kept)) : result
end

# =============================================================================
# simulate!() — Zero-allocation EICT single-kVp simulation hot path
# =============================================================================

"""
    simulate!(ws::EICTWorkspace, phantom, protocol, sim_opts)

Run EICT single-kVp simulation using pre-allocated workspace buffers.

Mutates `ws.sinogram` in place (the final log line-integral sinogram); read it (and `ws.geom`)
off the workspace. Returns `(; dose)`, the [`DoseReport`](@ref) of this acquisition (CTDIvol, DLP,
…) computed from the simulated beam, or `(; dose = nothing)` with `report_dose = false`.

`report_dose` attaches this acquisition's [`DoseReport`](@ref); it costs a cached Monte Carlo
(about a second) the first time a beam is seen and nothing afterwards, so set it to `false` in a
tight loop over noise realisations if that second matters. `dose_kwargs` reaches the keywords of
[`compute_dose`](@ref) — the defaults are one tube and the 32 cm body phantom, so a head
protocol wants `dose_kwargs = (; phantom = :head16)` and a dual-source scanner
`(; n_tubes = 2)`.

Pass `paths` from [`material_paths`](@ref) to skip the volume walk (point views only).

`keep_projection = true` adds `projection` to the result: everything before the noise, which every
noise draw of this phantom and protocol shares. Pass it back as `projection` — to a workspace of the
same scanner, protocol and grid — to draw another realization without projecting again,
bit-identically to a full simulation with that call's seed. The walk depends on the
geometry and the material map alone, so one cache serves every spectrum measured through that
geometry, and a several-kVp study of one phantom stops paying for it repeatedly (four tube
voltages of a 1024² × 60 phantom: 4.81 s of forward projection down to 2.16 s, bit-identical).

Create the workspace with `create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)`;
the detector's response (`m1`, `m2`) and electronic noise (`σ_electronic`) are baked into `ws` at
that point, so `simulate!` no longer needs `scanner`. The signal model is in
[`EICTWorkspace`](@ref).

Reconstruction is NOT included — handled by the caller via `reconstruct!`.
"""
function simulate!(
        ws::EICTWorkspace{T},
        phantom,
        protocol::CTProtocol,
        sim_opts::SimOptions = SimOptions();
        report_dose::Bool = true,
        dose_kwargs = (;),
        paths = nothing,
        projection = nothing,
        keep_projection::Bool = false,
    ) where {T}
    geom = ws.geom
    energies = ws.energies
    mats = ws.mats
    config = ws.config

    # Everything before the noise is deterministic: the same for every noise draw of this phantom,
    # protocol and workspace. `keep_projection` returns it (`projection`), and passing it back as
    # `projection` skips the forward projection and the physics before the noise — bit-identical to
    # simulating again, the noise drawn from this call's seed.
    if projection === nothing
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Polychromatic forward projection. One walk forms both spectral
        # sums of every ray, S₁ = Σ w̄ m₁′ B e^{−L} (the mean signal, into
        # ws.sinogram) and S₂ = Σ w̄ m₂′ B e^{−L} (its variance, into
        # ws.dispersion), each as −log S.
        # ═══════════════════════════════════════════════════════════════════════
        load_eict!(g) = _load_view_geometry!(
            ws.geom_source_positions,
            ws.geom_detector_centers,
            ws.geom_detector_u,
            ws.geom_detector_v,
            g,
        )
        n_sub = length(view_sample_offsets(geom, sim_opts))
        _integrate_views!(geom, sim_opts, load_eict!) do s_view, g_view
            fill!(ws.sinogram, zero(T))
            _forward_project_poly!(
                ws.sinogram, phantom.mask, g_view, energies, ws.weights, mats;
                ws_μ_volume = ws.μ_volume,
                ws_sino_mono = ws.sino_mono,
                ws_I_transmitted = ws.I_transmitted,
                ws_weights_norm = ws.weights_norm,
                ws_μ_lut_cpu = ws.μ_lut_cpu,
                ws_μ_lut_gpu = ws.μ_lut_gpu,
                ws_μ_table = ws.μ_table,
                ws_μ_table_gpu = ws.μ_table_gpu,
                ws_source_positions = ws.geom_source_positions,
                ws_detector_centers = ws.geom_detector_centers,
                ws_detector_u = ws.geom_detector_u,
                ws_detector_v = ws.geom_detector_v,
                volume_extent = phantom.extent,
                ws_bowtie_spectral = ws.bowtie_spectral,
                ws_w1_gpu = ws.wm1_gpu,
                ws_w2_gpu = ws.wm2_gpu,
                ws_out2 = ws.dispersion,
                paths = _view_paths(paths, n_sub),
                projector = sim_opts.projector
            )
            (ws.sinogram, ws.dispersion)
        end

        # The dispersion of each ray, S₂/S₁: its signal variance per unit mean signal
        let d = ws.dispersion, s = ws.sinogram
            AK.foreachindex(d) do idx
                d[idx] = exp(s[idx] - d[idx])
            end
        end

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Signal chain (always active)
        # ═══════════════════════════════════════════════════════════════════════
        _apply_physics_no_noise!(
            ws.sinogram, geom, config;
            ws_output = ws.physics_output,
            ws_optical_crosstalk_kernel = ws.optical_crosstalk_kernel,
            ws_focal_spot_kernel = ws.focal_spot_kernel,
            ws_lag_output = ws.physics_output,
            ws_lag_intensity = ws.lag_intensity,
            ws_lag_coeffs = ws.lag_coeffs
        )

        # ─── Patient scatter: the Geant4 library on the patient's own rays, added to S₁ and S₂
        # (src/detector/scatter.jl) ───
        if ws.scatter !== nothing
            _scatter_forward!(ws.scatter, phantom)
            _add_eict_scatter!(ws.sinogram, ws.dispersion, ws.scatter)
        end
    else
        size(projection.sinogram) == size(ws.sinogram) || throw(
            DimensionMismatch(
                "the projection is $(size(projection.sinogram)); this workspace's sinogram is $(size(ws.sinogram))"
            )
        )
        copyto!(ws.sinogram, projection.sinogram)
        copyto!(ws.dispersion, projection.dispersion)
    end
    kept = keep_projection ? (
            sinogram = copy(ws.sinogram),
            dispersion = copy(ws.dispersion),
        ) : nothing

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Noise and scatter subtraction, in signal units (E_ref):
    #   1. The detector reads primary + scatter, λ = I₀·ff·S₁ — the fill factor
    #      once, as the fraction of the quanta a dexel collects
    #   2. Compound-Poisson quantum noise, variance D·λ (D = S₂/S₁), plus the
    #      electronic noise, variance σ_electronic²
    #   3. The DAS subtracts the scatter it estimates from the measured signal
    #      (the patient taken as water) and holds the signal at ≥ 1
    # ═══════════════════════════════════════════════════════════════════════
    ff = config.fill_factor === nothing ? 1.0 : effective_fill_factor(config.fill_factor)
    I0v = T(compute_detector_I0(geom, protocol, sum(ws.weights)) * ff)
    σ = sim_opts.use_noise ? ws.σ_electronic : zero(T)
    has_scatter = ws.scatter !== nothing

    if sim_opts.use_noise
        # Seeded from sim_opts.seed every call, so same-seed reruns agree.
        if sim_opts.seed !== nothing
            Random.seed!(ws.rng, sim_opts.seed)
        end
        randn!(ws.rng, ws.noise_rand_cpu)
        copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)
        if σ > zero(T)
            randn!(ws.rng, ws.enoise_rand_cpu)
            copyto!(ws.enoise_rand_gpu, ws.enoise_rand_cpu)
        end
    end
    let sino = ws.sinogram, disp = ws.dispersion, rg = ws.noise_rand_gpu,
            gain = ws.dexel_gain, offset = ws.dexel_offset, ncnr = length(ws.dexel_gain),
            eg = σ > zero(T) ? ws.enoise_rand_gpu : ws.noise_rand_gpu,
            I0v = I0v, σ = σ, noisy = sim_opts.use_noise, linear = has_scatter
        AK.foreachindex(sino) do idx
            λ = I0v * exp(-sino[idx])
            if noisy
                λ += sqrt(max(disp[idx] * λ, zero(T))) * rg[idx] + σ * eg[idx]
            end
            ray = (idx - 1) % ncnr + 1
            λ = gain[ray] * λ + offset[ray]          # the dexel's own gain and offset
            # with scatter, the measured signal (relative to I₀) waits for the DAS's estimate
            sino[idx] = linear ? λ / I0v : -log(max(λ, one(T)) / I0v)
        end
    end
    if has_scatter
        _reduce_footprints!(ws.scatter.measured, ws.scatter, ws.sinogram)
        _scatter_estimate!(ws.scatter, Array(ws.scatter.measured))
        _subtract_eict_scatter!(ws.sinogram, ws.scatter, I0v)
    end

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Calibration: the air scan (each dexel's air signal), the low-signal
    # correction, the log
    # ═══════════════════════════════════════════════════════════════════════
    eps = T(1.0e-10)
    let sino = ws.sinogram, air = ws.air_reference, ncnr = length(ws.air_reference)
        AK.foreachindex(sino) do idx
            a = max(air[(idx - 1) % ncnr + 1], eps)
            sino[idx] = exp(-clamp(sino[idx], T(-1), T(15))) / a
        end
    end
    low_signal_correction_gpu!(ws.sinogram)
    let sino = ws.sinogram
        AK.foreachindex(sino) do idx
            sino[idx] = -log(max(sino[idx], eps))
        end
    end

    # BHC is decoupled — applied at notebook level
    dose = report_dose ? dose_report(ws, protocol; dose_kwargs...) : nothing
    return keep_projection ? (; dose, projection = kept) : (; dose)
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
function resolve_source_spectrum_without_bowtie(sim_opts::SimOptions, protocol::CTProtocol; scanner::Scanner)
    # Physics-based: raw IPEM spectrum + Beer-Lambert filtering
    e, w = load_spectrum_unfiltered(Int(protocol.kVp); anode_angle = ipem_anode_angle(scanner.target_angle))

    # Build filter list: scanner's built-in flat filter + protocol extras
    filters = Tuple{String, Float64}[]
    if scanner.flat_filter_thickness > 0
        push!(filters, (String(scanner.flat_filter_material), Float64(scanner.flat_filter_thickness)))
    end
    append!(filters, protocol.additional_filters)

    # Apply Beer-Lambert filtering + inverse-square-law distance scaling
    sdd_mm = Float64(scanner.source_to_detector)
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
        scanner, geom,
        diagnostic = false, label = "",
    ) -> (e, ŵ)

**Full-physics effective spectrum** — the spectrum each detector ray actually
sees after every spectrum-shaping effect simulate! applies, normalized per
ray so `Σ_E ŵ[col, row, E] = 1`.

This composes `resolve_source_spectrum_without_bowtie` (tube spectrum × flat
filter × additional filters × inverse-square SDD scaling) with the same
`PhysicsConfig` simulate! builds (via `build_physics_config`), then multiplies
in the bowtie, the heel-effect spectrum, and the detector's signal per photon
`m₁(E)` — each conditionally on its `sim_opts.use_*` flag.

The output `ŵ` is what to feed to per-ray polychromatic inverters (Cong,
beam-hardening correction, etc.) when you want the inversion's forward model
to match the forward model `simulate!` actually applied.  Use this instead of
chaining `resolve_source_spectrum_with_bowtie` + manual heel + manual m₁.

# Composition
```
w_source(E) = resolve_source_spectrum_without_bowtie(sim_opts, protocol)
B(col,row,E) = compute_bowtie_attenuation_spectral(scanner.bowtie_filter, geom, E)
heel(col,row,E) = compute_heel_spectral(config.heel_effect, geom, E)         [if heel enabled]
m₁(E)       = first(eid_signal_moments(config.detector_efficiency, E))     [EICT; 1 for PCCT]

ŵ_raw(col,row,E) = w_source(E) · m₁(E) · B(col,row,E) · heel(col,row,E)
ŵ(col,row,E)     = ŵ_raw / Σ_E ŵ_raw
```

With `sim_opts.use_heel_effect = false` the heel factor is 1; with
`use_detector_efficiency = false` an energy integrator's `m₁(E)` is `E` (every
photon absorbed whole), so the function automatically follows whatever the
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
    scanner = scanner, geom = sim_low.geom,
)
e_H, ŵ_H = BS.resolve_source_spectrum_full(
    sim_opts, protocol_high;
    scanner = scanner, geom = sim_high.geom,
)
material_basis = (ŵ_L = ŵ_L, p_L = …, q_L = …, ŵ_H = ŵ_H, p_H = …, q_H = …)
```
"""
function resolve_source_spectrum_full(
        sim_opts::SimOptions,
        protocol::CTProtocol;
        scanner,
        geom,
        diagnostic::Bool = false,
        label::String = "",
    )
    # Step 1: 1D source spectrum (tube × flat filter × additional filters × SDD)
    e, w_1d = resolve_source_spectrum_without_bowtie(sim_opts, protocol; scanner = scanner)

    # Step 2: identical PhysicsConfig that simulate!'s workspace ctor uses,
    # so `config.heel_effect` / `config.detector_efficiency` are populated
    # iff the corresponding sim_opts flag is true.
    # (scatter does not shape the spectrum, so its kernels are not loaded here)
    config = build_physics_config(
        scanner,
        SimOptions(sim_opts; use_scatter = false),
        Float64.(e),
        Float64.(w_1d),
    )

    # Step 3: bowtie B[col, row, E]
    bowtie = resolve_bowtie_filter(scanner.bowtie_filter)
    B = compute_bowtie_attenuation_spectral(bowtie, geom, Float64.(e))   # [col, row, E]

    # Step 4: conditional heel(col, row, E) and η(E)
    heel = config.heel_effect !== nothing ?
        compute_heel_spectral(config.heel_effect, geom, Float64.(e)) :
        ones(Float64, size(B)...)
    # an energy integrator weights each photon by its signal m₁; a counter counts it once
    m1 = scanner isa EICTScanner ?
        first(eid_signal_moments(config.detector_efficiency, Float64.(e))) :
        ones(Float64, length(e))

    # Step 5: w_source · m₁ · B · heel, then per-pixel normalize
    w_norm = Float64.(w_1d) ./ sum(Float64.(w_1d))
    ŵ_raw = similar(B)
    @inbounds for k in 1:length(e)
        wη = w_norm[k] * m1[k]
        for row in 1:size(B, 2), col in 1:size(B, 1)
            ŵ_raw[col, row, k] = wη * B[col, row, k] * heel[col, row, k]
        end
    end
    ŵ = Float32.(ŵ_raw ./ sum(ŵ_raw; dims = 3))

    if diagnostic
        mid_c = size(ŵ, 1) ÷ 2 + 1
        mid_r = size(ŵ, 2) ÷ 2 + 1
        mE_center = sum(Float64(e[k]) * ŵ[mid_c, mid_r, k] for k in eachindex(e))
        mE_edge = sum(Float64(e[k]) * ŵ[1, mid_r, k] for k in eachindex(e))
        tag = isempty(label) ? "" : " ($(label))"
        @info "effective spectrum$(tag) @ $(Int(protocol.kVp)) kVp — " *
            "center mean E = $(round(mE_center, digits = 1)) keV, " *
            "edge mean E = $(round(mE_edge, digits = 1)) keV " *
            "(Δ = $(round(mE_edge - mE_center, digits = 1)) keV)"
    end

    return e, ŵ
end

"""
    build_physics_config(scanner::Scanner, sim_opts::SimOptions, energies::Vector{Float64}, weights::Vector{Float64}) -> PhysicsConfig

Build a complete PhysicsConfig from Scanner hardware fields and SimOptions toggles.

For effects with Scanner fields (focal_spot, fill_factor, detector_efficiency,
heel_effect, scatter), the Scanner hardware parameters are used to construct the effect structs:
the scatter is the scanner's [`scatter_kernels`](@ref), loaded (and on first use downloaded) when
`use_scatter` is on. For effects without Scanner fields (optical_crosstalk, lag), factory function
defaults are used.

Noise is not part of PhysicsConfig — it is applied externally via
`compute_detector_I0()` + quantum noise when `sim_opts.use_noise == true`.

# Arguments
- `scanner`: Scanner hardware definition (provides physical parameters)
- `sim_opts`: SimOptions with resolved boolean toggles
- `energies`: Energy bin centers (keV) for the current spectrum
- `weights`: Photon weights for each energy bin

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
        weights::Vector{Float64},
    )
    kwargs = Dict{Symbol, Any}()

    # --- Common settings ---
    kwargs[:energy_keV] = sum(energies .* weights) / sum(weights)
    kwargs[:noise_seed] = sim_opts.seed

    # --- Physics Pipeline effects (from Scanner fields where available) ---

    # Fill factor: use Scanner's row/col fill factors
    if sim_opts.use_fill_factor && scanner isa EICTScanner
        kwargs[:fill_factor] = FillFactorModel(
            scanner.fill_factor_row,
            scanner.fill_factor_col,
            scanner.fill_factor_row ≈ scanner.fill_factor_col,
        )
    end

    # Detector response (EICT only): the scintillator's Geant4 table, or analytic absorption with
    # `detector_efficiency_mode = :beer_lambert`. A photon-counting detector's response is its own
    # (the PCCT response matrix), so a PCCTScanner never reads this field.
    if sim_opts.use_detector_efficiency && scanner isa EICTScanner
        kwargs[:detector_efficiency] = DetectorEfficiency(
            scanner.detector_material,
            scanner.detector_depth,
            sim_opts.detector_efficiency_mode == :beer_lambert ? BEER_LAMBERT : MC_LUT,
        )
    end

    # Scatter: the scanner's Geant4 kernels (throws when it names none)
    if sim_opts.use_scatter
        kwargs[:scatter] = scatter_kernels(scanner)
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
            kwargs[:heel_effect] = default_heel_effect(anode_angle_deg = angle)
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

`helical_q` and `coverage` reach the helical chain and are refused on an axial geometry; see
[`wfbp_helical_reconstruct`](@ref). `coverage` must share the volume's element type and backend,
and is masked outside the reconstruction circle along with the volume.

# Returns
The mutated `ws.volume` (same object — `===` to the workspace field).
"""
function reconstruct!(
        ws::FDKReconWorkspace{T},
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry;
        helical_q::Real = 0.7,
        coverage::Union{Nothing, AbstractArray{T, 3}} = nothing,
    ) where {T <: AbstractFloat}

    if !is_helical(geom) && coverage !== nothing
        throw(ArgumentError("coverage is only defined for a helical geometry; this one is axial"))
    end
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
        _wfbp_backproject!(ws.volume, ws.filtered, geom, T(Δt); helical_q, coverage)
        apply_fov_mask!(ws.volume, geom)
        # the mask is part of the answer, so a voxel it discards has no coverage either
        coverage === nothing || apply_fov_mask!(coverage, geom; sentinel_μ = zero(T))
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

"""
    _hir_continue_z!(work, output_z)

Fill the axial halo planes of `work` (those outside `output_z`) by continuation from the nearest
plane inside it.  The halo is a boundary model for the cone, not a reconstruction: FBP would give
it unsupported terminal estimates from the narrower acquisition, so it starts as the boundary
and the exact iterative operator updates only what the detector observes there.
"""
function _hir_continue_z!(work::AbstractArray{T, 3}, output_z) where {T}
    nz_work = size(work, 3)
    length(output_z) == nz_work && return work
    nx = Int32(size(work, 1))
    ny = Int32(size(work, 2))
    zlo = Int32(first(output_z))
    zhi = Int32(last(output_z))
    backend = AK.get_backend(work)
    _foreachindex(work, backend) do idx
        idx0 = Int32(idx - 1)
        i = (idx0 % nx) + Int32(1)
        j = ((idx0 ÷ nx) % ny) + Int32(1)
        kw = (idx0 ÷ (nx * ny)) + Int32(1)
        if kw < zlo
            work[idx] = work[i, j, zlo]
        elseif kw > zhi
            work[idx] = work[i, j, zhi]
        end
    end
    return work
end

"Place a caller's initial volume (the requested grid) into its window of the work grid."
function _hir_seed_work!(work::AbstractArray{T, 3}, init::AbstractArray{T, 3}, output_x, output_y, output_z) where {T}
    work === init && return work
    nx = Int32(size(init, 1))
    ny = Int32(size(init, 2))
    x0 = Int32(first(output_x)); y0 = Int32(first(output_y)); z0 = Int32(first(output_z))
    backend = AK.get_backend(work)
    _foreachindex(init, backend) do idx
        idx0 = Int32(idx - 1)
        i = (idx0 % nx) + Int32(1)
        j = ((idx0 ÷ nx) % ny) + Int32(1)
        k = (idx0 ÷ (nx * ny)) + Int32(1)
        work[x0 + i - Int32(1), y0 + j - Int32(1), z0 + k - Int32(1)] = init[idx]
    end
    return _hir_continue_z!(work, output_z)
end

"The requested grid, read out of its window of the work grid."
function _hir_extract_output!(output::AbstractArray{T, 3}, work::AbstractArray{T, 3}, output_x, output_y, output_z) where {T}
    output === work && return output
    nx = Int32(size(output, 1))
    ny = Int32(size(output, 2))
    x0 = Int32(first(output_x)); y0 = Int32(first(output_y)); z0 = Int32(first(output_z))
    backend = AK.get_backend(output)
    _foreachindex(output, backend) do idx
        idx0 = Int32(idx - 1)
        i = (idx0 % nx) + Int32(1)
        j = ((idx0 ÷ nx) % ny) + Int32(1)
        k = (idx0 ÷ (nx * ny)) + Int32(1)
        output[idx] = work[x0 + i - Int32(1), y0 + j - Int32(1), z0 + k - Int32(1)]
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
- `init_volume`: Optional warm-start volume on the requested grid (same shape as `ws.volume`).
  It replaces the FDK seed inside the requested window only: the rest of HIR's support (the
  patient outside the requested circle, the cone halo along z) is still seeded by FDK from the
  sinogram, so the model holds everything the detector saw. The caller's array is unchanged.
- `air_reference`: Optional 2D array `[n_cols, n_rows]` of bowtie air
  reference values.  When provided, statistical weights are scaled by the
  air reference to account for position-dependent noise from the bowtie
  filter (edge pixels have fewer counts → more noise → lower weight).

`weights` selects the statistical weighting of the data term, and the right choice depends on
what the sinogram IS, not on which scanner made it. `:transmission` (default) is `exp(-y)`
scaled by `air_reference` when given: right for any log-transmission sinogram — an
energy-integrating acquisition, a photon-counting acquisition's summed bins, either tube voltage
of a dual-energy pair — because a ray through more material carried fewer photons. `:uniform`
weights every ray equally, leaving only the geometric ray-length normalisation. An array of the
sinogram's shape is used as the per-ray statistical weight directly: for a material-basis
sinogram from a projection-domain decomposition (photon-counting bins or a dual-kVp pair through
[`decompose_nchannel`](@ref)), whose values are g/cm² and not log-transmissions, pass the
decomposition's inverse-variance map (see [`vmi_pipeline`](@ref)) or `:uniform`; `exp(-y)` has
no meaning for it.
"""
function reconstruct!(
        ws::HIRReconWorkspace{T},
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry;
        init_volume::Union{Nothing, AbstractArray{T, 3}} = nothing,
        air_reference::Union{Nothing, AbstractArray} = nothing,
        weights::Union{Symbol, AbstractArray} = :transmission,
    ) where {T <: AbstractFloat}
    # `weights` is how the data term knows what it is fitting. `:transmission` is the Poisson
    # heuristic for a log-transmission sinogram, w = exp(-y): a ray through more material carried
    # fewer photons and is trusted less. That is only right when `y` IS a log-transmission. A
    # basis sinogram from a material decomposition is not — it is g/cm² of one material, can be
    # negative, and its variance comes from the decomposition's Fisher information, not from
    # exp(-y) — so for it the caller passes `:uniform` or the inverse-variance map itself.
    weights isa Symbol && (
        weights in (:transmission, :uniform) || throw(
            ArgumentError("weights must be :transmission, :uniform or an array; got :$(weights)")
        )
    )
    if weights isa AbstractArray
        size(weights) == size(sinogram) || throw(
            DimensionMismatch(
                "weights has size $(size(weights)); the sinogram is $(size(sinogram))"
            )
        )
        air_reference === nothing || throw(
            ArgumentError(
                "air_reference is the transmission weighting's bowtie map; it has no meaning " *
                    "alongside explicit weights"
            )
        )
    end

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
                ws.filtered, geom;
                ws_conv_scratch = ws.conv_scratch,
                ws_filter_kernel = ws.filter_kernel
            )
            # Seed the whole in-plane support by FDK — the fan covers the scan circle, so the
            # data support it — and the axial halo by continuation: FDK from the narrower
            # acquisition would give the halo unsupported terminal estimates, so it starts as
            # the boundary and the exact iterative operator updates only the detector-observable
            # components there (V_inv is zero on its nullspace).
            fill!(model_volume, zero(T))
            backproject!(
                model_volume, ws.filtered, model_geom;
                weighted = true,
                ws_source_positions = ws.geom_source_positions,
                ws_detector_centers = ws.geom_detector_centers,
                ws_detector_u = ws.geom_detector_u,
                ws_detector_v = ws.geom_detector_v
            )
            _hir_continue_z!(model_volume, ws.output_z)
        end
    else
        size(init_volume) == size(ws.volume) || throw(
            DimensionMismatch(
                "init_volume has size $(size(init_volume)); expected $(size(ws.volume))"
            )
        )
        # The caller's volume is the requested grid; the support beyond it (the in-plane ring
        # of patient outside the requested circle, the cone halo) must still hold the object,
        # or the data term piles that attenuation onto the circle's edge exactly as a
        # truncated model does — and a reused workspace would otherwise keep the previous
        # object in the ring. Seed the whole support by FDK, then place the caller's volume
        # in its window.
        if size(model_volume) != size(ws.volume)
            copyto!(ws.filtered, sinogram)
            filter_sinogram!(
                ws.filtered, geom;
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
        copyto!(ws.volume, init_volume)
        _hir_seed_work!(model_volume, ws.volume, ws.output_x, ws.output_y, ws.output_z)
    end

    # ─── Step 2: OS-PWLS refinement with Huber regularization ───
    # Erdoğan & Fessler 1999 ordered-subsets PWLS for transmission CT.
    # The public table is historically calibrated to the legacy Siddon/BP
    # normalization. Exact DDᵀ has a roughly 10× different regularizer
    # sensitivity; convert that scale here without changing the strength API or
    # the supported Siddon reconstruction semantics.
    λ = T(params.lambda) * (ws.projector === :siddon ? one(T) : T(0.1))
    λ_relax = T(params.relaxation)
    δ = T(params.huber_delta)
    backend = AK.get_backend(model_volume)

    nepochs = params.nepochs
    n_subsets = params.n_subsets
    subset_scale = T(n_subsets)

    # strength = 0 ⇒ no PWLS refinement at all: the FDK init IS the result.
    # Bail out before the weight kernels so a 0 % request costs a plain FBP.
    if nepochs == 0
        _hir_extract_output!(ws.volume, model_volume, ws.output_x, ws.output_y, ws.output_z)
        apply_fov_mask!(ws.volume, geom)
        return ws.volume
    end

    # The iterative system matrix represents the circular reconstruction FOV,
    # not the enclosing square array. Keep that support constraint active
    # throughout HIR; masking only at return lets exterior corner estimates
    # feed back through A*x and leaves a circular boundary ring.
    apply_fov_mask!(model_volume, model_geom; sentinel_μ = zero(T))

    # Initialize statistical weights.
    #   :transmission — w = air_ref(col,row) × exp(-y); air_ref accounts for bowtie-modulated I0
    #                   (edge pixels → fewer counts → lower weight)
    #   :uniform      — w = 1: the data term is the geometric ray-length normalisation alone
    #   array         — the caller's statistical weights, used as given (floored at ε)
    if weights isa AbstractArray
        let sw = ws.data_weights, ε = T(1.0e-6)
            copyto!(sw, weights)
            _foreachindex(sw, backend) do idx
                w = sw[idx]
                sw[idx] = (isfinite(w) && w > zero(T)) ? w + ε : ε
            end
        end
    elseif weights === :uniform
        fill!(ws.data_weights, one(T))
    elseif air_reference !== nothing
        let sw = ws.data_weights, ε = T(1.0e-6), aref = air_reference,
                nc = Int32(size(sinogram, 1)), nr = Int32(size(sinogram, 2))
            _foreachindex(sw, backend) do idx
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
            _foreachindex(sw, backend) do idx
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
        _foreachindex(dw, backend) do idx
            dw[idx] = wp[idx] * dw[idx]
        end
    end

    for epoch in 1:nepochs
        # Preserve the historically calibrated Siddon cadence. Exact DD uses
        # the current-iterate gradient below to avoid stale-Laplacian gain.
        ws.projector === :siddon && compute_huber_gradient!(ws.reg_grad, model_volume, δ)
        for (s, angle_indices) in enumerate(ws.subsets)
            # The penalty gradient must describe the CURRENT ordered-subset
            # iterate. Reusing the epoch-start gradient for all 12 updates
            # repeatedly applies one stale Laplacian step; on high-frequency
            # modes its cumulative gain can cross zero and amplify the very
            # texture the Huber prior is meant to suppress.
            ws.projector === :siddon || compute_huber_gradient!(ws.reg_grad, model_volume, δ)

            n_sub = length(angle_indices)
            geom_s = ws.subset_geometries[s]

            # Forward project with subset geometry → subset_Ax_buf.
            # Uses ws.projector so A·x matches the projector that made the data.
            # Avoid wrapping the full buffer in a SubArray. Metal dispatch on
            # that otherwise-equivalent view is ~2.3× slower for both A and Aᵀ.
            # Uneven final subsets retain the bounded view fallback.
            ax_view = n_sub == size(ws.subset_Ax_buf, 3) ?
                ws.subset_Ax_buf : view(ws.subset_Ax_buf, :, :, 1:n_sub)
            fill!(ax_view, zero(T))
            _project_mono_hir!(
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

                _foreachindex(ax, backend) do idx
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
            if ws.projector === :siddon
                _backproject_mono!(
                    ws.projector, ws.correction, ax_view, geom_s;
                    ws_source_positions = ws.subset_geom_source_positions[s],
                    ws_detector_centers = ws.subset_geom_detector_centers[s],
                    ws_detector_u = ws.subset_geom_detector_u[s],
                    ws_detector_v = ws.subset_geom_detector_v[s]
                )
            else
                _backproject_mono!(
                    ws.projector, ws.correction, ax_view, geom_s;
                    active_z = 1:size(ws.work_volume, 3), circular_support = true,
                    ws_source_positions = ws.subset_geom_source_positions[s],
                    ws_detector_centers = ws.subset_geom_detector_centers[s],
                    ws_detector_u = ws.subset_geom_detector_u[s],
                    ws_detector_v = ws.subset_geom_detector_v[s]
                )
            end

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

                _foreachindex(vol, backend) do idx
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
    _hir_extract_output!(ws.volume, model_volume, ws.output_x, ws.output_y, ws.output_z)
    apply_fov_mask!(ws.volume, geom)

    return ws.volume
end

# =============================================================================
# Cached per-material path lengths
# =============================================================================

"""
    material_paths(ws::EICTWorkspace, phantom; volume_extent=nothing) -> Array
    material_paths!(paths, ws::EICTWorkspace, phantom; volume_extent=nothing) -> paths

Walk `phantom` once through the workspace's geometry and return the per-material path length of
every detector element, in cm — `(n_materials, n_cols, n_rows, n_views)` on the phantom's
backend.

The walk is the expensive part of a polychromatic forward projection (about 95 % of it) and
depends only on the geometry and the material map: not on the tube voltage, the filtration, the
bowtie or the detector. So one cache serves every spectrum measured through that geometry — pass
it to [`simulate!`](@ref) as `paths` and each further acquisition costs the spectral conversion
alone, bit-identically.

```julia
paths = material_paths(ws_120, phantom)              # one walk
for (kvp, ws) in workspaces                          # 80, 100, 120, 140 kVp
    simulate!(ws, phantom, protocols[kvp], sim_opts; paths)
end
```

!!! warning
    The cache belongs to the phantom and geometry it was walked for, and nothing downstream can
    tell that it does not: reusing it with a different phantom silently projects the old one.
    Its size is also the cost — `n_materials × n_cols × n_rows × n_views` floats, 1.7 GiB for
    16 materials on an 834 × 34 × 1000 sinogram — so `compact_materials(phantom)` first.

The cache is of point views: with view integration (`SimOptions(; view_samples > 1)`) every
sub-view is a geometry of its own, and the noise draws of one acquisition share their projection
instead (`keep_projection`, `projection`).
"""
function material_paths(ws::EICTWorkspace{T}, phantom; kwargs...) where {T}
    paths = similar(ws.sinogram, T, size(ws.μ_table_gpu, 1), size(ws.sinogram)...)
    return material_paths!(paths, ws, phantom; kwargs...)
end

@doc (@doc material_paths)
function material_paths!(
        paths::AbstractArray{T, 4}, ws::EICTWorkspace{T}, phantom;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ) where {T}
    extent = volume_extent !== nothing ? volume_extent :
        (
            hasproperty(phantom, :extent) && phantom.extent !== nothing ?
            Tuple(Float64.(phantom.extent)) : nothing
        )
    return dd_fast_material_paths!(
        paths, phantom.mask, ws.geom;
        volume_extent = extent,
        ws_source_positions = ws.geom_source_positions,
        ws_detector_centers = ws.geom_detector_centers,
        ws_detector_u = ws.geom_detector_u,
        ws_detector_v = ws.geom_detector_v,
    )
end

export material_paths, material_paths!
