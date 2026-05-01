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

export MonoPlusWorkspace, create_mono_plus_workspace, apply_mono_plus!
