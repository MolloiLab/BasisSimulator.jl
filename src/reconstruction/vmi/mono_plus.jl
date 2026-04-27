"""
Mono+ / VMI+ — strict 1:1 port of Grant 2014 §Technique for Calculating
Mono+ Images.

Paper specs (HARD constraints — faithfully implemented):
- Inputs: VMI at target keV + VMI at noise-optimal keV (~70 keV)
- Frequency-split decomposition of BOTH images into LP + HP subimages
- Combination: Mono+(E) = LP(VMI_E) + HP(VMI_opt)
- LP + HP complementary, so Mono+(E_opt) = VMI_opt identically

Paper is SILENT on (best guesses, all tunable):
- LP filter shape      → [BEST GUESS] 2D Gaussian LP via FFT diagonal
- LP kernel size       → [BEST GUESS] σ = `σ_lp_px` pixels
- Same LP for both?    → YES (strongly implied by the paper wording)
- Pre-denoise VMI_opt? → NO (strict parity ⇒ use as-is)
- 2D per-slice vs 3D?  → [BEST GUESS] per-slice 2D
- Boundary conditions  → [BEST GUESS] FFT periodic

Reference:
  Grant, Flohr, Krauss, Sedlmair, Thomas, Schmidt (2014)
  *Invest Radiol* 49(9):586–592.  DOI 10.1097/RLI.0000000000000060.
"""

import FFTW

"""
    apply_mono_plus(volumes, energies;
                    E_noise_opt = 70.0,
                    σ_lp_px     = 2.0,
                    verbose     = true)

Compute Mono+(E) for each energy in `energies` using Grant 2014's
frequency-split rule:

    Mono+(E)     = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt)
    Mono+(E_opt) = VMI_opt   (identity at the noise-optimal reference)

Returns a NamedTuple with `energies`, `volumes` (same shape as input),
`σ_lp_px`, `E_noise_opt`.

# Arguments
- `volumes`    : Vector{Array{Float32,3}} aligned to `energies`
- `energies`   : Vector of target keV values

# Keyword arguments
- `E_noise_opt` : noise-optimal reference energy (must appear in `energies`)
- `σ_lp_px`     : Gaussian LP σ in pixels.  Either:
    • a scalar (`Real`) → same σ at every energy (legacy behavior), or
    • a per-energy vector (`AbstractVector`, length == `energies`) → each
      Mono+(E) uses its own σ for both `LP(VMI_E)` and `LP(VMI_opt)`.
      σ = 0 is a valid identity case (Mono+(E) = VMI_E exactly, no FFT).
      σ at `E_noise_opt` is irrelevant — Mono+(E_opt) = VMI_opt regardless.

Larger σ → more denoising (more HF content borrowed from VMI_opt); smaller
σ → preserve more own-VMI detail.  Effective HP cutoff ≈ 1/(2π·σ) cycles/px.
"""
function apply_mono_plus(
        volumes::AbstractVector{<:AbstractArray{Float32, 3}},
        energies::AbstractVector;
        E_noise_opt::Real                    = 70.0,
        σ_lp_px::Union{Real, AbstractVector} = 2.0,
        verbose::Bool                        = true,
    )
    length(volumes) == length(energies) ||
        error("apply_mono_plus: length(volumes) = $(length(volumes)) ≠ length(energies) = $(length(energies))")

    i_opt = findfirst(==(Float64(E_noise_opt)), Float64.(energies))
    i_opt === nothing &&
        error("apply_mono_plus: E_noise_opt = $(E_noise_opt) keV not in energies = $(energies)")

    σ_vec = if σ_lp_px isa Real
        fill(Float64(σ_lp_px), length(energies))
    else
        length(σ_lp_px) == length(energies) ||
            error("apply_mono_plus: σ_lp_px length $(length(σ_lp_px)) ≠ energies length $(length(energies))")
        Float64.(σ_lp_px)
    end

    vmi_opt = volumes[i_opt]
    nx, ny, _ = size(vmi_opt)

    # Frequency-grid coordinates for FFT-domain Gaussian (shared across σ).
    fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
    fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]

    # Cache one Gaussian kernel per distinct σ value to avoid recomputation.
    kernel_cache = Dict{Float64, Matrix{Float64}}()
    function _kernel_for(σ::Float64)
        get!(kernel_cache, σ) do
            σ² = σ^2
            [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]
        end
    end

    # 2D per-slice Gaussian LP via FFT.  `lp_buf` is named distinctly from
    # the outer `out` Vector so Julia's same-scope unification doesn't fold
    # them — see the cautionary note in the original notebook port.
    function _gaussian_lp_2d_fft(img::AbstractArray{Float32, 3}, σ::Float64)
        kernel = _kernel_for(σ)
        nxl, nyl, nzl = size(img)
        lp_buf = similar(img)
        Threads.@threads for k in 1:nzl
            slice = Float64.(@view img[:, :, k])
            lp_buf[:, :, k] .= Float32.(real.(FFTW.ifft(FFTW.fft(slice) .* kernel)))
        end
        lp_buf
    end

    t0 = time()
    out = Vector{Array{Float32, 3}}(undef, length(energies))

    σ_uniform = all(σ -> σ == σ_vec[1], σ_vec)
    if σ_uniform
        # Fast path: same σ at every energy — compute LP(VMI_opt) once.
        σ0 = σ_vec[1]
        if σ0 == 0.0
            # σ = 0 ⇒ LP = identity ⇒ Mono+(E) = VMI_E + (VMI_opt − VMI_opt) = VMI_E.
            for i in eachindex(energies)
                out[i] = copy(volumes[i])
            end
        else
            lp_opt = _gaussian_lp_2d_fft(vmi_opt, σ0)
            hp_opt = vmi_opt .- lp_opt
            for i in eachindex(energies)
                if i == i_opt
                    out[i] = copy(vmi_opt)
                else
                    lp_E = _gaussian_lp_2d_fft(volumes[i], σ0)
                    out[i] = lp_E .+ hp_opt
                end
            end
        end
    else
        # Per-energy σ: each Mono+(E) uses its own LP(VMI_E) and LP(VMI_opt).
        for i in eachindex(energies)
            σ_i = σ_vec[i]
            if i == i_opt
                out[i] = copy(vmi_opt)
            elseif σ_i == 0.0
                out[i] = copy(volumes[i])
            else
                lp_E     = _gaussian_lp_2d_fft(volumes[i], σ_i)
                lp_opt_i = _gaussian_lp_2d_fft(vmi_opt,    σ_i)
                out[i]   = lp_E .+ vmi_opt .- lp_opt_i
            end
        end
    end
    dt = time() - t0

    if verbose
        σ_label = σ_uniform ? "σ=$(σ_vec[1]) px" : "σ per-keV $(σ_vec)"
        @info "Mono+ (Grant 2014 1:1 parity): $σ_label, E_opt=$(Int(E_noise_opt)) keV, $(Threads.nthreads()) threads, $(round(dt * 1000; digits = 1)) ms"
        @info "  LP = 2D Gaussian, FFT periodic BC, per-slice  [BEST GUESS — paper unspecified]"
        @info "  HP = (VMI_opt − LP(VMI_opt)), σ matched per energy when vector"
        @info "  sanity: Mono+(E_opt) = VMI_opt identically  (identity at i_opt=$(i_opt))"
    end

    (energies = collect(energies), volumes = out,
     σ_lp_px = (σ_lp_px isa Real ? Float64(σ_lp_px) : σ_vec),
     E_noise_opt = Float64(E_noise_opt))
end

export apply_mono_plus
