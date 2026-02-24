# =============================================================================
# GPU Performance Profiler for BasisSimulator.jl
# =============================================================================
#
# Lightweight wall-time profiler that instruments the simulate!() hot path
# to reveal exactly where time is spent across GPU kernel stages.
#
# Key stages tracked:
#   μ_table_diff        — CPU diff of changed material rows (skips NIST for unchanged)
#   create_μ_volume     — GPU scatter kernel: mask → μ_volume (×n_energies)
#   siddon_raytrace     — GPU ray tracing: μ_volume → sino_mono (×n_energies, MAIN COST)
#   beer_lambert_accum  — GPU Beer-Lambert accumulation: I += w·exp(-sino_mono) (×n_energies)
#   beer_lambert_final  — GPU final -log(I) pass
#   physics_pipeline    — GPU physics effects (scatter, crosstalk, etc.)
#   noise               — GPU quantum noise injection
#   cpu_gpu_copy        — GPU↔CPU memcpy (sinogram readback + noise RNG)
#
# GPU Synchronization
# -------------------
# Accurate GPU timing requires inserting a synchronization point before reading
# the wall clock — otherwise the kernel may still be queued.
# We call Metal.synchronize() / CUDA.synchronize() if available; otherwise we
# accept that timings include queuing overhead only (still useful for relative ratios).
#
# Usage
# -----
#   prof = GPUProfiler()
#   profile_simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts; profiler=prof)
#   print_profile(prof)
#
# =============================================================================

using Printf

export GPUProfiler, print_profile, reset_profile!, profile_simulate!

# =============================================================================
# Stage constants (symbols → readable names)
# =============================================================================

const _PROFILE_STAGES = [
    :μ_table_diff        => "μ-table diff (CPU)",
    :create_μ_volume     => "create_μ_volume! ×N_E (GPU scatter)",
    :siddon_raytrace     => "siddon_forward_project! ×N_E (GPU rays)",
    :beer_lambert_accum  => "Beer-Lambert accum ×N_E (GPU)",
    :beer_lambert_final  => "Beer-Lambert final -log (GPU)",
    :physics_pipeline    => "physics pipeline (GPU)",
    :noise               => "quantum noise (GPU)",
    :cpu_gpu_copy        => "CPU↔GPU copies",
    :other               => "other (fill!, overhead)",
]

# =============================================================================
# GPUProfiler struct
# =============================================================================

"""
    GPUProfiler

Mutable accumulator for wall-time profiling of `simulate!()` GPU stages.

Fields contain cumulative seconds per named stage across all `profile_simulate!`
calls. Divide by `n_calls` to get per-call averages.

# Usage
```julia
prof = GPUProfiler()
profile_simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts; profiler=prof)
print_profile(prof)
```
"""
mutable struct GPUProfiler
    # Cumulative time (seconds) per stage
    times::Dict{Symbol, Float64}
    # Number of energy bins observed (filled on first call)
    n_energies::Int
    # Number of profile_simulate! calls
    n_calls::Int

    function GPUProfiler()
        times = Dict{Symbol, Float64}(k => 0.0 for (k, _) in _PROFILE_STAGES)
        new(times, 0, 0)
    end
end

"""
    reset_profile!(prof::GPUProfiler)

Zero all accumulated timing data.
"""
function reset_profile!(prof::GPUProfiler)
    for k in keys(prof.times)
        prof.times[k] = 0.0
    end
    prof.n_energies = 0
    prof.n_calls = 0
    return prof
end

# =============================================================================
# GPU synchronization helper
# =============================================================================

"""
    _gpu_sync(arr)

Synchronize the GPU device associated with `arr` before taking a wall-clock
reading. Falls back gracefully to no-op on CPU arrays.
"""
function _gpu_sync(arr::AbstractArray)
    if isdefined(Main, :Metal) && isdefined(Main.Metal, :synchronize)
        Main.Metal.synchronize()
    elseif isdefined(Main, :CUDA) && isdefined(Main.CUDA, :synchronize)
        Main.CUDA.synchronize()
    elseif isdefined(Main, :AMDGPU) && isdefined(Main.AMDGPU, :synchronize)
        Main.AMDGPU.synchronize()
    end
    # CPU arrays: no-op
    return nothing
end

# =============================================================================
# Timed record helper
# =============================================================================

"""
    _timed(f, prof, stage, sync_arr)

Execute `f()`, synchronize GPU, measure wall time, and add to `prof.times[stage]`.
Returns the return value of `f()`.
"""
function _timed(f::F, prof::GPUProfiler, stage::Symbol, sync_arr=nothing) where F
    t_start = time_ns()
    result = f()
    if sync_arr !== nothing
        _gpu_sync(sync_arr)
    end
    elapsed = (time_ns() - t_start) / 1e9
    prof.times[stage] = get(prof.times, stage, 0.0) + elapsed
    return result
end

# =============================================================================
# Instrumented _forward_project_poly! with profiler support
# =============================================================================

"""
    _forward_project_poly_profiled!(sinogram, mask, geom, energies, weights, materials, prof; kwargs...)

Instrumented version of `_forward_project_poly!` that records per-stage timing
into `prof::GPUProfiler`.

The per-energy loop is timed in three sub-stages:
  1. `create_μ_volume`    — GPU scatter: mask → μ_volume
  2. `siddon_raytrace`    — GPU Siddon ray tracing
  3. `beer_lambert_accum` — GPU Beer-Lambert accumulation

The final -log pass is recorded as `beer_lambert_final`.
"""
function _forward_project_poly_profiled!(
    sinogram::AbstractArray{T, 3},
    mask::AbstractArray{<:Unsigned, 3},
    geom::CTGeometry,
    energies::Vector,
    weights::Vector,
    materials::Vector,
    prof::GPUProfiler;
    ws_μ_volume=nothing,
    ws_sino_mono=nothing,
    ws_I_transmitted=nothing,
    ws_weights_norm::Union{Nothing, Vector{T}}=nothing,
    ws_μ_lut_cpu::Union{Nothing, Vector{T}}=nothing,
    ws_μ_lut_gpu=nothing,
    ws_μ_table=nothing,
    ws_source_positions=nothing,
    ws_detector_centers=nothing,
    ws_detector_u=nothing,
    ws_detector_v=nothing,
    volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing
) where T <: AbstractFloat

    n_energies = length(energies)
    prof.n_energies = n_energies

    weights_norm = ws_weights_norm !== nothing ? ws_weights_norm : T.(weights ./ sum(weights))
    μ_volume     = ws_μ_volume     !== nothing ? ws_μ_volume     : similar(sinogram, T, size(mask))
    sino_mono    = ws_sino_mono    !== nothing ? ws_sino_mono     : similar(sinogram)
    I_transmitted = ws_I_transmitted !== nothing ? ws_I_transmitted : similar(sinogram)

    fill!(I_transmitted, zero(T))

    for e_idx in 1:n_energies

        # Stage 1: create_μ_volume — GPU scatter (mask → μ_volume)
        _timed(prof, :create_μ_volume, sinogram) do
            create_μ_volume!(μ_volume, mask, materials, energies[e_idx];
                             ws_μ_lut_cpu=ws_μ_lut_cpu, ws_μ_lut_gpu=ws_μ_lut_gpu,
                             ws_μ_table=ws_μ_table, energy_idx=e_idx)
        end

        # Stage 2: siddon_forward_project — GPU ray tracing
        _timed(prof, :siddon_raytrace, sinogram) do
            fill!(sino_mono, zero(T))
            siddon_forward_project!(sino_mono, μ_volume, geom;
                ws_source_positions=ws_source_positions,
                ws_detector_centers=ws_detector_centers,
                ws_detector_u=ws_detector_u,
                ws_detector_v=ws_detector_v,
                volume_extent=volume_extent)
        end

        # Stage 3: Beer-Lambert accumulation
        w = weights_norm[e_idx]
        _timed(prof, :beer_lambert_accum, sinogram) do
            let it = I_transmitted, sm = sino_mono
                AK.foreachindex(it) do idx
                    it[idx] += w * exp(-sm[idx])
                end
            end
        end
    end

    # Stage 4: Final -log pass
    eps_val = T(1e-10)
    _timed(prof, :beer_lambert_final, sinogram) do
        let sg = sinogram, it = I_transmitted
            AK.foreachindex(sg) do idx
                sg[idx] = -log(max(it[idx], eps_val))
            end
        end
    end

    return sinogram
end

# =============================================================================
# profile_simulate! — drop-in for simulate! with profiling enabled
# =============================================================================

"""
    profile_simulate!(ws, phantom, scanner, protocol[, sim_opts, recon_opts]; profiler, materials)

Instrumented version of `simulate!()` that records wall-time per GPU stage.

All arguments are identical to `simulate!(ws::EICTWorkspace, ...)`. Returns the
same `(sino_ideal, sino_noisy)` NamedTuple as `simulate!`.

After the call, inspect timings with `print_profile(profiler)`.

# Example
```julia
prof = GPUProfiler()
result = profile_simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts; profiler=prof)
print_profile(prof)
```
"""
function profile_simulate!(
    ws::EICTWorkspace{T},
    phantom,
    scanner::Scanner,
    protocol::CTProtocol,
    sim_opts::SimOptions = SimOptions(),
    recon_opts::ReconOptions = ReconOptions();
    profiler::GPUProfiler = GPUProfiler(),
    materials::Union{Nothing, Vector} = nothing
) where {T}

    prof = profiler
    prof.n_calls += 1

    geom     = ws.geom
    energies = ws.energies
    config   = ws.config
    mats     = _resolve_materials(phantom, materials)

    # ── μ-table diff (CPU) ────────────────────────────────────────────────────
    lut_cpu, lut_gpu, μ_table = _timed(prof, :μ_table_diff) do
        if length(mats) == length(ws.mats)
            for r in 1:length(mats)
                ws.mats[r].name === mats[r].name && continue
                for (e_idx, E) in enumerate(energies)
                    ws.μ_table[r, e_idx] = T(compute_μ_at_energy(mats[r], Float64(E)))
                end
                ws.mats[r] = mats[r]
            end
            (ws.μ_lut_cpu, ws.μ_lut_gpu, ws.μ_table)
        else
            (nothing, nothing, nothing)
        end
    end

    # ── Polychromatic forward projection ─────────────────────────────────────
    fill!(ws.sinogram, zero(T))
    _forward_project_poly_profiled!(
        ws.sinogram, phantom.mask, geom, energies, ws.weights, mats, prof;
        ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
        ws_I_transmitted=ws.I_transmitted,
        ws_weights_norm=ws.weights_norm,
        ws_μ_lut_cpu=lut_cpu, ws_μ_lut_gpu=lut_gpu,
        ws_μ_table=μ_table,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom.extent
    )

    # ── Signal chain / physics pipeline ──────────────────────────────────────
    if ws.has_signal_chain
        heel_effect = ws.heel_effect
        das_model   = ws.das_model
        bhc_eff     = ws.bhc

        _timed(prof, :physics_pipeline, ws.sinogram) do
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
        end

        eps = T(1e-10)
        _timed(prof, :other, ws.sinogram) do
            let sino = ws.sinogram
                AK.foreachindex(sino) do idx
                    sino[idx] = exp(-clamp(sino[idx], T(-1), T(15)))
                end
            end
            if heel_effect !== nothing
                apply_heel_effect!(ws.sinogram, heel_effect, geom)
            end
            if das_model !== nothing
                apply_das_model!(ws.sinogram, das_model; seed=config.noise_seed)
            end

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

            let sino = ws.sinogram, air = ws.air_scan
                AK.foreachindex(sino) do idx
                    air_val = max(air[idx], eps)
                    sino[idx] = sino[idx] / air_val
                end
            end

            low_signal_correction_gpu!(ws.sinogram)

            let sino = ws.sinogram
                AK.foreachindex(sino) do idx
                    sino[idx] = -log(max(sino[idx], eps))
                end
            end

            if bhc_eff !== nothing
                apply_bhc!(ws.sinogram, bhc_eff; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
            end
        end
    else
        if config !== nothing
            _timed(prof, :physics_pipeline, ws.sinogram) do
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
    end

    # ── CPU readback (ideal sinogram) ─────────────────────────────────────────
    _timed(prof, :cpu_gpu_copy, ws.sinogram) do
        copyto!(ws.sino_ideal_out, ws.sinogram)
    end

    # ── Quantum noise ─────────────────────────────────────────────────────────
    if sim_opts.use_noise
        I0_T = T(compute_detector_I0(geom, protocol))

        _timed(prof, :cpu_gpu_copy) do
            Random.randn!(ws.noise_rand_gpu)
        end
        _timed(prof, :noise, ws.sinogram) do
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

    # ── CPU readback (noisy sinogram) ─────────────────────────────────────────
    _timed(prof, :cpu_gpu_copy, ws.sinogram) do
        copyto!(ws.sino_noisy_out, ws.sinogram)
    end

    return (sino_ideal=ws.sino_ideal_out, sino_noisy=ws.sino_noisy_out)
end

# =============================================================================
# print_profile — formatted summary table
# =============================================================================

"""
    print_profile(prof::GPUProfiler; per_call=true)

Print a formatted breakdown of wall-time per GPU stage.

If `per_call=true` (default), shows per-call averages (total / n_calls).
Set `per_call=false` to show raw cumulative totals.

# Output example
```
┌─────────────────────────────────────────────────────────────────┐
│ BasisSimulator GPU Profile  (1 call, 30 energies)               │
├────────────────────────────────────────────┬────────┬───────────┤
│ Stage                                      │   Time │       %   │
├────────────────────────────────────────────┼────────┼───────────┤
│ siddon_forward_project! ×N_E (GPU rays)    │  98.2s │    83.4%  │
│ create_μ_volume! ×N_E (GPU scatter)        │   8.1s │     6.9%  │
│ Beer-Lambert accum ×N_E (GPU)              │   3.4s │     2.9%  │
│ Beer-Lambert final -log (GPU)              │   0.2s │     0.2%  │
│ physics pipeline (GPU)                     │   0.0s │     0.0%  │
│ quantum noise (GPU)                        │   0.3s │     0.3%  │
│ CPU↔GPU copies                             │   2.1s │     1.8%  │
│ μ-table diff (CPU)                         │   0.1s │     0.1%  │
│ other (fill!, overhead)                    │   4.9s │     4.2%  │
├────────────────────────────────────────────┼────────┼───────────┤
│ TOTAL                                      │ 117.3s │   100.0%  │
└────────────────────────────────────────────┴────────┴───────────┘

  Bottleneck: siddon_forward_project! accounts for 83.4% of simulation time.
  Optimization target: fuse 30-energy loop into single GPU kernel to eliminate
  serial kernel dispatch overhead (~90 separate GPU launches per time point).
```
"""
function print_profile(prof::GPUProfiler; per_call::Bool=true)
    nc = max(prof.n_calls, 1)
    scale = per_call ? 1.0 / nc : 1.0

    # Stage label map
    stage_labels = Dict(k => v for (k, v) in _PROFILE_STAGES)

    # Gather (label, time) pairs in canonical order
    rows = [(stage_labels[k], prof.times[k] * scale) for (k, _) in _PROFILE_STAGES]

    total = sum(t for (_, t) in rows)
    total = max(total, 1e-9)  # avoid division by zero

    # Column widths
    lw = max(maximum(length(l) for (l, _) in rows), 40)
    tw = 8   # time column
    pw = 9   # percent column

    bar = "─"
    sep_top   = "┌" * bar^(lw+2) * "┬" * bar^(tw+2) * "┬" * bar^(pw+2) * "┐"
    sep_head  = "├" * bar^(lw+2) * "┼" * bar^(tw+2) * "┼" * bar^(pw+2) * "┤"
    sep_bot   = "└" * bar^(lw+2) * "┴" * bar^(tw+2) * "┴" * bar^(pw+2) * "┘"
    hdr_title = "│ " * rpad("BasisSimulator GPU Profile  ($(nc) call$(nc==1 ? "" : "s"), $(prof.n_energies) energies)", lw+tw+pw+5) * " │"

    println(sep_top)
    println(hdr_title)
    println(sep_head)
    println("│ " * rpad("Stage", lw) * " │ " * lpad("Time", tw) * " │ " * lpad("%", pw) * " │")
    println(sep_head)

    # Sort by descending time for easier reading
    sorted_rows = sort(rows; by=((_, t),) -> -t)

    for (label, t) in sorted_rows
        pct = 100.0 * t / total
        t_str = t >= 100 ? @sprintf("%6.1fs", t) :
                t >= 10  ? @sprintf("%6.2fs", t) :
                t >= 1   ? @sprintf("%6.3fs", t) :
                           @sprintf("%5.1fms", t * 1000)
        p_str = @sprintf("%6.1f%%", pct)
        println("│ " * rpad(label, lw) * " │ " * lpad(t_str, tw) * " │ " * lpad(p_str, pw) * " │")
    end

    println(sep_head)
    total_str = total >= 100 ? @sprintf("%6.1fs", total) :
                total >= 10  ? @sprintf("%6.2fs", total) :
                total >= 1   ? @sprintf("%6.3fs", total) :
                               @sprintf("%5.1fms", total * 1000)
    println("│ " * rpad("TOTAL", lw) * " │ " * lpad(total_str, tw) * " │ " * lpad("100.0%", pw) * " │")
    println(sep_bot)

    # Bottleneck annotation
    top_label, top_t = sorted_rows[1]
    top_pct = 100.0 * top_t / total
    if top_pct > 50
        println()
        println("  Bottleneck: \"$(top_label)\" accounts for $(@sprintf("%.1f%%", top_pct)) of simulation time.")
    end

    # Per-energy breakdown for the energy loop stages
    if prof.n_energies > 0
        ne = prof.n_energies
        println()
        println("  Per-energy breakdown (÷ $(ne) bins):")
        for stage in (:create_μ_volume, :siddon_raytrace, :beer_lambert_accum)
            t = prof.times[stage] * scale / ne
            t_str = t >= 1   ? @sprintf("%.3fs", t) :
                               @sprintf("%.1fms", t * 1000)
            println("    $(rpad(stage_labels[stage], lw))  $(lpad(t_str, 8)) / bin")
        end
    end

    println()
    return nothing
end
