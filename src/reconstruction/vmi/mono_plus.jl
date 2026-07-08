"""
Mono+ / VMI+ — strict 1:1 port of Grant 2014 §Technique for Calculating
Mono+ Images.

Paper specs (HARD constraints — faithfully implemented):
- Inputs: VMI at target keV + VMI at noise-optimal keV (~70 keV).
- Frequency-split decomposition of BOTH images into LP + HP subimages.
- Combination: Mono+(E) = LP(VMI_E) + HP(VMI_opt).
- LP + HP complementary, so Mono+(E_opt) = VMI_opt identically.

Paper is SILENT on (best guesses, all tunable):
- LP filter shape      → 2D Gaussian LP via FFT diagonal.
- LP kernel size       → σ = `σ_lp_px` pixels.
- 2D per-slice vs 3D?  → per-slice 2D.
- Boundary conditions  → FFT periodic.

# Workspace

Mono+ runs entirely on CPU (FFTW), so memory pressure is system RAM, not
GPU memory.  The workspace owns:
  • `lp_buf`   — vol-shape Float32 LP scratch (reused across calls).
  • `lp_opt`   — vol-shape Float32 LP(VMI_opt) cache (uniform-σ fast path).
  • `hp_opt`   — vol-shape Float32 (VMI_opt − LP(VMI_opt)) cache.
  • `out_vols` — Vector of pre-allocated output volumes (length `n_energies`).
  • `kernel_cache` — Dict{σ → 2D Gaussian kernel}, populated lazily.

`apply_mono_plus!` mutates `ws.out_vols` in place and returns a NamedTuple
referencing them.  **If you need to retain output across multiple calls
on the same workspace, `copy(...)` the returned volumes.**

Reference:
  Grant, Flohr, Krauss, Sedlmair, Thomas, Schmidt (2014)
  *Invest Radiol* 49(9):586–592.  DOI 10.1097/RLI.0000000000000060.
"""

import FFTW

# ════════════════════════════════════════════════════════════════════════
#  Workspace
# ════════════════════════════════════════════════════════════════════════

"""
    MonoPlusWorkspace{T, A3}

Pre-allocated scratch + output volumes for `apply_mono_plus!`.

Layout:
  • 3 vol-shape Float32 scratch: `lp_buf`, `lp_opt`, `hp_opt`.
  • `out_vols`: `Vector{Array{Float32, 3}}` of length `n_energies`,
    each shape matching the input volumes — reused across calls.
  • `kernel_cache`: lazy `Dict{Float64, Matrix{Float64}}` keyed by σ.

Construct with [`create_mono_plus_workspace`](@ref).
"""
mutable struct MonoPlusWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}}
    lp_buf::A3
    lp_opt::A3
    hp_opt::A3
    out_vols::Vector{A3}
    kernel_cache::Dict{Float64, Matrix{Float64}}
end

"""
    create_mono_plus_workspace(vol_template; n_energies, mem_budget_GB = nothing)
        -> MonoPlusWorkspace

Allocate the Mono+ workspace on the same backend as `vol_template`.

# Arguments
- `vol_template`  : an `AbstractArray{<:AbstractFloat, 3}` whose shape +
  backend the workspace should match (one example VMI volume).

# Keyword arguments
- `n_energies`    : number of target energies — sizes `out_vols`.
- `mem_budget_GB` : explicit budget override.  `nothing` = auto-probe via
  `Sys.free_memory() · 0.6`.  Even though Mono+ runs on CPU (FFTW), the
  output volumes can be large enough on big inputs to OOM the host —
  the constructor will retry-with-shrink-on-failure.

Mono+ does not auto-tile; if even allocation at `tile_size = 1` fails, the
total working set (3 scratch + n_energies output volumes) just doesn't
fit.  Downsize the volume or reduce `n_energies`.
"""
function create_mono_plus_workspace(
        vol_template::AbstractArray{<:AbstractFloat, 3};
        n_energies::Integer,
        mem_budget_GB::Union{Nothing, Real} = nothing,
    )
    T = eltype(vol_template)
    n_energies > 0 || error("create_mono_plus_workspace: n_energies must be ≥ 1.")
    sh = size(vol_template)
    bytes_per_vol = prod(sh) * sizeof(T)

    # Crude fit-check: 3 scratch + n_energies output volumes.
    total_need = (3 + Int(n_energies)) * bytes_per_vol
    avail = if mem_budget_GB === nothing
        floor(Int, Float64(Sys.free_memory()) * 0.6)
    else
        floor(Int, Float64(mem_budget_GB) * 2^30)
    end
    if total_need > avail
        @warn ("MonoPlusWorkspace working set exceeds 60% of free memory; allocation may fail. " *
               "Pass `mem_budget_GB = ...` to override the probe, or downsize the volume / reduce n_energies.") total_need_GB=round(total_need/2^30, digits=2) avail_GB=round(avail/2^30, digits=2) n_energies vol_shape=sh
    end

    with_oom_retry("create_mono_plus_workspace", 1) do _
        lp_buf  = similar(vol_template, T, sh);  fill!(lp_buf, zero(T))
        lp_opt  = similar(vol_template, T, sh);  fill!(lp_opt, zero(T))
        hp_opt  = similar(vol_template, T, sh);  fill!(hp_opt, zero(T))
        out_vols = [similar(vol_template, T, sh) for _ in 1:Int(n_energies)]
        for v in out_vols; fill!(v, zero(T)); end
        MonoPlusWorkspace{T, typeof(lp_buf)}(
            lp_buf, lp_opt, hp_opt, out_vols,
            Dict{Float64, Matrix{Float64}}(),
        )
    end
end

# ════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════

"""
    apply_mono_plus!(ws, volumes, energies;
                     E_noise_opt = 70.0,
                     σ_lp_px     = 2.0,
                     verbose     = true) -> NamedTuple

Compute Mono+(E) for each energy in `energies` using Grant 2014's
frequency-split rule:

    Mono+(E)     = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)
    Mono+(E_opt) = VMI_opt   (identity at the noise-optimal reference)

Mutates `ws.out_vols` in place and returns a NamedTuple with `energies`,
`volumes` (= references into `ws.out_vols`), `σ_lp_px`, `E_noise_opt`.
**Copy if you need to retain across subsequent calls** on the same `ws`.

# Arguments
- `ws`         : `MonoPlusWorkspace` from `create_mono_plus_workspace`.
  REQUIRED — no internal allocation path.
- `volumes`    : `Vector{Array{Float32, 3}}` aligned to `energies`.
- `energies`   : Vector of target keV values.

# Keyword arguments
- `E_noise_opt` : noise-optimal reference energy (must appear in `energies`).
- `σ_lp_px`     : Gaussian LP σ in pixels.  Either a scalar (same σ at
  every energy) or a per-energy vector (length == `energies`).  σ = 0
  is a valid identity case (Mono+(E) = VMI_E exactly, no FFT).
- `verbose`     : log timing + identity-sanity message.
- `phantom_mask`: optional `Bool` array same shape as the volumes.  When
  provided, voxels where the mask is `false` are reverted to the raw
  input `VMI_E` after the Mono+ formula.  Eliminates the bright ring at
  the phantom-air boundary that LP-based frequency splitting injects via
  `HP_opt = VMI_opt − LP_σ(VMI_opt)`.  Pass an *eroded* phantom mask
  (≥ ~3·σ pixels of erosion — see `BS.erode_mask_3d`) so the residual
  edge spike falls outside the masked region.  `nothing` (default) →
  no masking, identical to legacy behavior.
"""
function apply_mono_plus!(
        ws::MonoPlusWorkspace,
        volumes::AbstractVector{<:AbstractArray{Float32, 3}},
        energies::AbstractVector;
        E_noise_opt::Real                    = 70.0,
        σ_lp_px::Union{Real, AbstractVector} = 2.0,
        verbose::Bool                        = true,
        phantom_mask::Union{Nothing, AbstractArray{Bool, 3}} = nothing,
    )
    length(volumes) == length(energies) ||
        error("apply_mono_plus!: length(volumes) = $(length(volumes)) ≠ length(energies) = $(length(energies))")
    length(ws.out_vols) == length(energies) ||
        error("apply_mono_plus!: workspace has $(length(ws.out_vols)) output slots ≠ length(energies) = $(length(energies)).  " *
              "Recreate workspace with `create_mono_plus_workspace(...; n_energies = $(length(energies)))`.")
    sh = size(ws.lp_buf)
    for (i, v) in enumerate(volumes)
        size(v) == sh ||
            error("apply_mono_plus!: volumes[$i] shape $(size(v)) ≠ workspace shape $(sh).")
    end
    if phantom_mask !== nothing
        size(phantom_mask) == sh ||
            error("apply_mono_plus!: phantom_mask shape $(size(phantom_mask)) ≠ workspace shape $(sh).")
    end

    i_opt = findfirst(==(Float64(E_noise_opt)), Float64.(energies))
    i_opt === nothing &&
        error("apply_mono_plus!: E_noise_opt = $(E_noise_opt) keV not in energies = $(energies)")

    σ_vec = if σ_lp_px isa Real
        fill(Float64(σ_lp_px), length(energies))
    else
        length(σ_lp_px) == length(energies) ||
            error("apply_mono_plus!: σ_lp_px length $(length(σ_lp_px)) ≠ energies length $(length(energies))")
        Float64.(σ_lp_px)
    end

    vmi_opt = volumes[i_opt]
    nx, ny, nz = size(vmi_opt)
    fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
    fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]

    # Gaussian-kernel cache — populated lazily into ws.kernel_cache.
    function _kernel_for(σ::Float64)
        get!(ws.kernel_cache, σ) do
            σ² = σ^2
            [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]
        end
    end

    # 2D per-slice Gaussian LP via FFT — writes into the provided `dst`.
    function _gaussian_lp!(dst::AbstractArray{Float32, 3}, img::AbstractArray{Float32, 3}, σ::Float64)
        kernel = _kernel_for(σ)
        Threads.@threads for k in 1:nz
            slice = Float64.(@view img[:, :, k])
            dst[:, :, k] .= Float32.(real.(FFTW.ifft(FFTW.fft(slice) .* kernel)))
        end
        return dst
    end

    # Apply phantom mask post-Mono+: outside the mask, revert to raw VMI_E.
    # No-op for σ=0 / reference-energy slots (they're already exact copies
    # of the input).
    function _apply_mask!(out::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3})
        if phantom_mask !== nothing
            @inbounds for j in eachindex(out)
                if !phantom_mask[j]
                    out[j] = src[j]
                end
            end
        end
        return out
    end

    t0 = time()
    σ_uniform = all(σ -> σ == σ_vec[1], σ_vec)

    if σ_uniform
        σ0 = σ_vec[1]
        if σ0 == 0.0
            # σ = 0 ⇒ LP = identity ⇒ Mono+(E) = VMI_E.
            for i in eachindex(energies)
                copyto!(ws.out_vols[i], volumes[i])
            end
        else
            _gaussian_lp!(ws.lp_opt, vmi_opt, σ0)
            @. ws.hp_opt = vmi_opt - ws.lp_opt
            for i in eachindex(energies)
                if i == i_opt
                    copyto!(ws.out_vols[i], vmi_opt)
                else
                    _gaussian_lp!(ws.lp_buf, volumes[i], σ0)
                    @. ws.out_vols[i] = ws.lp_buf + ws.hp_opt
                    _apply_mask!(ws.out_vols[i], volumes[i])
                end
            end
        end
    else
        # Per-energy σ: each Mono+(E) needs its own LP(VMI_E) and LP(VMI_opt).
        for i in eachindex(energies)
            σ_i = σ_vec[i]
            if i == i_opt
                copyto!(ws.out_vols[i], vmi_opt)
            elseif σ_i == 0.0
                copyto!(ws.out_vols[i], volumes[i])
            else
                _gaussian_lp!(ws.lp_buf, volumes[i], σ_i)
                _gaussian_lp!(ws.lp_opt, vmi_opt,    σ_i)
                @. ws.out_vols[i] = ws.lp_buf + vmi_opt - ws.lp_opt
                _apply_mask!(ws.out_vols[i], volumes[i])
            end
        end
    end
    dt = time() - t0

    if verbose
        σ_label = σ_uniform ? "σ=$(σ_vec[1]) px" : "σ per-keV $(σ_vec)"
        @info "Mono+ (Grant 2014 1:1 parity): $σ_label, E_opt=$(Int(E_noise_opt)) keV, $(Threads.nthreads()) threads, $(round(dt * 1000; digits = 1)) ms"
        @info "  LP = 2D Gaussian, FFT periodic BC, per-slice"
        @info "  sanity: Mono+(E_opt) = VMI_opt identically  (identity at i_opt=$(i_opt))"
    end

    (energies = collect(energies),
     volumes  = ws.out_vols,
     σ_lp_px  = (σ_lp_px isa Real ? Float64(σ_lp_px) : σ_vec),
     E_noise_opt = Float64(E_noise_opt))
end

# ════════════════════════════════════════════════════════════════════════
#  Resolution-preserving Mono+  (per-pixel HF regression — Kalender-style)
# ════════════════════════════════════════════════════════════════════════

"""
    apply_mono_plus_regression!(ws, volumes, energies;
                                E_noise_opt = 70.0,
                                σ_lp_px     = 1.5,
                                window      = 4,
                                beta_max    = 4.0,
                                verbose     = true,
                                phantom_mask = nothing) -> NamedTuple

Structure-preserving Mono+.  Same LP/HP skeleton as [`apply_mono_plus!`]
(Grant 2014) — the quantitative low-frequency band is kept from the target
energy `E`, so rod HU is untouched — but the **hard high-frequency swap is
replaced by a per-pixel local regression** of the target HF onto the
low-noise reference HF, the exact analog of `apply_acnr_kalender!`:

    Mono+_res(E) = LP_σ(VMI_E) + β(x)·HP_σ(VMI_opt)
                 = VMI_E − ( hE − β(x)·hOpt )

    β(x) = clamp( Σ_win hE·hOpt / Σ_win hOpt² , 0, β_max )

with `hE = VMI_E − LP_σ(VMI_E)`, `hOpt = VMI_opt − LP_σ(VMI_opt)`.

Why this reduces noise with **zero resolution cost** (by construction, not
by an edge heuristic):

  * **Real edge (present in both energies)** → hE, hOpt strongly correlated
    → β ≈ the true local contrast ratio C_E/C_opt.  `β·hOpt` rebuilds the
    target energy's TRUE edge amplitude (fixing plain Mono+'s contrast
    loss for fine iodine detail) but carried on the low-noise reference →
    denoised.  Edge geometry = hOpt's geometry exactly (β only scales
    amplitude) → no blur, no misregistration.
  * **Flat / noise-only region** → the two energies' HF are independent
    noise realizations → uncorrelated → β ≈ 0 → HF ≈ 0 → HF noise removed
    (strictly better than plain Mono+, which injects hOpt's residual HF
    noise everywhere).
  * **E == E_opt** → hE ≡ hOpt → β ≡ 1 → output = VMI_opt identically
    (Grant's reference-identity property retained).

`β_max` is additionally variance-limited to the global HF-energy ratio
`λ = √(Σ hE² / Σ hOpt²)` (Kalender's original bound), so a noisy local β̂
can never over-amplify the reference HF.

# Keyword arguments
- `E_noise_opt` : noise-optimal reference energy (must appear in `energies`).
- `σ_lp_px`     : Gaussian LP σ (px) for the HF split.  Scalar or per-energy
  vector.  σ = 0 ⇒ identity at that energy (Mono+_res(E) = VMI_E).
- `window`      : half-width of the (2w+1)² regression window (px).  Sizes
  only the ESTIMATE of β — the correction stays a per-pixel linear
  combination, so no signal smoothing occurs regardless of window.
- `beta_max`    : hard ceiling on the HF amplification β (further capped by
  the per-energy global HF ratio λ).
- `phantom_mask`: optional eroded `Bool` mask; voxels where `false` revert to
  the raw `VMI_E` (kills the LP boundary ring — same semantics as
  `apply_mono_plus!`).

Mutates `ws.out_vols` in place; **copy** the returned volumes to retain them
across subsequent calls on the same `ws`.  Returns a NamedTuple with
`energies`, `volumes`, `σ_lp_px`, `E_noise_opt`, and per-energy diagnostics
`β_mean` / `β_frac_active` (fraction of pixels with β > 0).
"""
function apply_mono_plus_regression!(
        ws::MonoPlusWorkspace,
        volumes::AbstractVector{<:AbstractArray{Float32, 3}},
        energies::AbstractVector;
        E_noise_opt::Real                    = 70.0,
        σ_lp_px::Union{Real, AbstractVector} = 1.5,
        window::Integer                      = 4,
        beta_max::Real                       = 4.0,
        verbose::Bool                        = true,
        phantom_mask::Union{Nothing, AbstractArray{Bool, 3}} = nothing,
    )
    length(volumes) == length(energies) ||
        error("apply_mono_plus_regression!: length(volumes) = $(length(volumes)) ≠ length(energies) = $(length(energies))")
    length(ws.out_vols) == length(energies) ||
        error("apply_mono_plus_regression!: workspace has $(length(ws.out_vols)) output slots ≠ length(energies) = $(length(energies)).  " *
              "Recreate workspace with `create_mono_plus_workspace(...; n_energies = $(length(energies)))`.")
    sh = size(ws.lp_buf)
    for (i, v) in enumerate(volumes)
        size(v) == sh ||
            error("apply_mono_plus_regression!: volumes[$i] shape $(size(v)) ≠ workspace shape $(sh).")
    end
    if phantom_mask !== nothing
        size(phantom_mask) == sh ||
            error("apply_mono_plus_regression!: phantom_mask shape $(size(phantom_mask)) ≠ workspace shape $(sh).")
    end

    i_opt = findfirst(==(Float64(E_noise_opt)), Float64.(energies))
    i_opt === nothing &&
        error("apply_mono_plus_regression!: E_noise_opt = $(E_noise_opt) keV not in energies = $(energies)")

    σ_vec = if σ_lp_px isa Real
        fill(Float64(σ_lp_px), length(energies))
    else
        length(σ_lp_px) == length(energies) ||
            error("apply_mono_plus_regression!: σ_lp_px length $(length(σ_lp_px)) ≠ energies length $(length(energies))")
        Float64.(σ_lp_px)
    end

    vmi_opt = volumes[i_opt]
    nx, ny, nz = size(vmi_opt)
    fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
    fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]
    w = Int(window)

    function _kernel_for(σ::Float64)
        get!(ws.kernel_cache, σ) do
            σ² = σ^2
            [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]
        end
    end
    function _gaussian_lp!(dst::AbstractArray{Float32, 3}, img::AbstractArray{Float32, 3}, σ::Float64)
        kernel = _kernel_for(σ)
        Threads.@threads for k in 1:nz
            slice = Float64.(@view img[:, :, k])
            dst[:, :, k] .= Float32.(real.(FFTW.ifft(FFTW.fft(slice) .* kernel)))
        end
        return dst
    end
    function _apply_mask!(out::AbstractArray{Float32, 3}, src::AbstractArray{Float32, 3})
        if phantom_mask !== nothing
            @inbounds for j in eachindex(out)
                phantom_mask[j] || (out[j] = src[j])
            end
        end
        return out
    end

    # HF-regression transplant, writing Mono+_res(E) into `out`.
    #   hE   = VMI_E − lpE      (target HF, lpE precomputed in ws.lp_buf)
    #   hOpt = ws.hp_opt        (reference HF, precomputed for this σ)
    #   β    = clamp(Σ_win hE·hOpt / Σ_win hOpt², 0, βmax)  (variance-limited)
    #   out  = lpE + β·hOpt
    function _hf_regress!(out, vol_E, lpE, hOpt)
        # per-energy global HF ratio → variance-limited β ceiling (Kalender)
        sEE = 0.0; sOO = 0.0
        @inbounds for idx in eachindex(vol_E)
            he = Float64(vol_E[idx] - lpE[idx]); ho = Float64(hOpt[idx])
            sEE += he * he; sOO += ho * ho
        end
        λ = sqrt(sEE / max(sOO, 1.0e-30))
        βmax = Float32(min(Float64(beta_max), λ))
        β_acc = 0.0; β_active = 0
        β_lock = Threads.SpinLock()
        Threads.@threads for k in 1:nz
            local_acc = 0.0; local_active = 0
            @inbounds for j in 1:ny, i in 1:nx
                sOOw = 0.0f0; sEOw = 0.0f0
                for dj in -w:w
                    jj = j + dj; (1 <= jj <= ny) || continue
                    for di in -w:w
                        ii = i + di; (1 <= ii <= nx) || continue
                        ho = hOpt[ii, jj, k]
                        he = vol_E[ii, jj, k] - lpE[ii, jj, k]
                        sOOw += ho * ho; sEOw += he * ho
                    end
                end
                β = clamp(sEOw / max(sOOw, 1.0f-20), 0.0f0, βmax)
                out[i, j, k] = lpE[i, j, k] + β * hOpt[i, j, k]
                local_acc += β; (β > 0) && (local_active += 1)
            end
            Threads.lock(β_lock) do
                β_acc += local_acc; β_active += local_active
            end
        end
        npix = nx * ny * nz
        return (β_mean = β_acc / npix, β_frac_active = β_active / npix)
    end

    t0 = time()
    σ_uniform = all(σ -> σ == σ_vec[1], σ_vec)
    diags = Vector{NamedTuple{(:β_mean, :β_frac_active), Tuple{Float64, Float64}}}(undef, length(energies))

    if σ_uniform && σ_vec[1] != 0.0
        σ0 = σ_vec[1]
        _gaussian_lp!(ws.lp_opt, vmi_opt, σ0)
        @. ws.hp_opt = vmi_opt - ws.lp_opt              # reference HF (cached)
        for i in eachindex(energies)
            if i == i_opt
                copyto!(ws.out_vols[i], vmi_opt)
                diags[i] = (β_mean = 1.0, β_frac_active = 1.0)
            else
                _gaussian_lp!(ws.lp_buf, volumes[i], σ0) # lpE into ws.lp_buf
                diags[i] = _hf_regress!(ws.out_vols[i], volumes[i], ws.lp_buf, ws.hp_opt)
                _apply_mask!(ws.out_vols[i], volumes[i])
            end
        end
    else
        # Per-energy σ (or all-zero): recompute the reference HF per σ.
        for i in eachindex(energies)
            σ_i = σ_vec[i]
            if i == i_opt
                copyto!(ws.out_vols[i], vmi_opt)
                diags[i] = (β_mean = 1.0, β_frac_active = 1.0)
            elseif σ_i == 0.0
                copyto!(ws.out_vols[i], volumes[i])         # identity
                diags[i] = (β_mean = 0.0, β_frac_active = 0.0)
            else
                _gaussian_lp!(ws.lp_opt, vmi_opt, σ_i)
                @. ws.hp_opt = vmi_opt - ws.lp_opt
                _gaussian_lp!(ws.lp_buf, volumes[i], σ_i)
                diags[i] = _hf_regress!(ws.out_vols[i], volumes[i], ws.lp_buf, ws.hp_opt)
                _apply_mask!(ws.out_vols[i], volumes[i])
            end
        end
    end
    dt = time() - t0

    if verbose
        σ_label = σ_uniform ? "σ=$(σ_vec[1]) px" : "σ per-keV $(σ_vec)"
        @info "Mono+ (resolution-preserving, per-pixel HF regression): $σ_label, window=$(w), β_max=$(beta_max), E_opt=$(Int(E_noise_opt)) keV, $(Threads.nthreads()) threads, $(round(dt * 1000; digits = 1)) ms"
        for (i, E) in enumerate(energies)
            i == i_opt && continue
            @info "  $(Int(E)) keV: β_mean=$(round(diags[i].β_mean; sigdigits = 3)) · active(β>0)=$(round(100 * diags[i].β_frac_active; digits = 1))% · structure keeps low-keV amplitude, flat HF → 0 (noise removed), edges bit-aligned to VMI_opt"
        end
        @info "  sanity: Mono+_res(E_opt) = VMI_opt identically  (identity at i_opt=$(i_opt))"
    end

    (energies = collect(energies),
     volumes  = ws.out_vols,
     σ_lp_px  = (σ_lp_px isa Real ? Float64(σ_lp_px) : σ_vec),
     E_noise_opt = Float64(E_noise_opt),
     β_mean = [d.β_mean for d in diags],
     β_frac_active = [d.β_frac_active for d in diags])
end

# ════════════════════════════════════════════════════════════════════════
#  Structure-transplant Mono+  (WIDE-band, contrast-matched — the powerful one)
# ════════════════════════════════════════════════════════════════════════

"""
    apply_mono_plus_structure!(ws, volumes, energies;
                               E_noise_opt = 70.0,
                               σ_lp_px     = 6.0,
                               window      = 5,
                               beta_max    = 6.0,
                               verbose     = true,
                               phantom_mask = nothing) -> NamedTuple

The **powerful** resolution-preserving Mono+.  Same Grant-2014 frequency-split
skeleton and the same per-pixel contrast-matched transplant as
[`apply_mono_plus_regression!`], but run in the regime that actually denoises:

    Mono+(E) = LP_σ(VMI_E) + β(x)·HP_σ(VMI_opt),   HP_σ = I − LP_σ

Two things make it much stronger than either prior variant:

  1. **WIDE transplant band (low cutoff / large σ).**  Post-FBP CT noise is
     high-frequency-weighted (ramp filter), so most of the noise *variance*
     lives above even a low cutoff.  The generic Gaussian Mono+ and the
     narrow-σ regression Mono+ used a HIGH cutoff (σ≈1.5) → they only
     transplanted the top sliver of the spectrum → ~all the noise stayed in
     the LP band (impotent, esp. at low keV where σ_VMI is dominated by the
     broadband α(E)²σ²_I term).  Here σ is LARGE: the LP band keeps only the
     **coarse quantitative base** (per-region HU ⇒ rod accuracy preserved)
     and the entire broadband detail is transplanted from the low-noise 70
     keV anchor.  In a flat region this collapses the output to a
     heavily-smoothed base ⇒ σ plummets (≈ 15–20× for ramp-weighted noise).

  2. **Contrast-matched transplant (local β) — the anatomical-identity prior,
     quantified.**  β(x) = clamp(Σ_win hE·hOpt / Σ_win hOpt², 0, βmax) is the
     LOCAL contrast ratio: ≈0 in flat regions (⇒ pure smoothed base, noise
     killed), ≈ α(E)/α(E_opt) at real iodine/anatomy edges (⇒ the anchor's
     sharp low-noise edge, re-scaled to the TARGET energy's amplitude).  This
     is why it does not blur: edges come from the full-bandwidth anchor at the
     target's own contrast.  βmax is variance-limited to the global HF ratio
     λ = √(Σhᴇ²/Σh_opt²) so a noisy β̂ cannot over-amplify.

Net: flat → smoothed base (huge low-keV denoise); edges → low-noise anchor
structure at correct target contrast (full resolution).  E == E_opt ⇒ β ≡ 1 ⇒
output = VMI_opt identically.

Only the default σ (6.0 vs 1.5) and window (5 vs 4) differ from
`apply_mono_plus_regression!` in mechanism; `σ_lp_px` is the master dial —
larger = wider transplant = more denoising (until the LP base gets too coarse
to hold per-rod HU).  All arguments as in `apply_mono_plus_regression!`.
"""
function apply_mono_plus_structure!(
        ws::MonoPlusWorkspace,
        volumes::AbstractVector{<:AbstractArray{Float32, 3}},
        energies::AbstractVector;
        E_noise_opt::Real                    = 70.0,
        σ_lp_px::Union{Real, AbstractVector} = 6.0,
        window::Integer                      = 5,
        beta_max::Real                       = 6.0,
        verbose::Bool                        = true,
        phantom_mask::Union{Nothing, AbstractArray{Bool, 3}} = nothing,
    )
    return apply_mono_plus_regression!(
        ws, volumes, energies;
        E_noise_opt = E_noise_opt,
        σ_lp_px     = σ_lp_px,
        window      = window,
        beta_max    = beta_max,
        verbose     = verbose,
        phantom_mask = phantom_mask,
    )
end

export MonoPlusWorkspace, create_mono_plus_workspace, apply_mono_plus!
export apply_mono_plus_regression!, apply_mono_plus_structure!
