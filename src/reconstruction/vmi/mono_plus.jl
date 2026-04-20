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

# Per-slice 2D Gaussian LP via FFT (exact, one-shot).  Top-level helper
# so there's no inner-closure capture of the outer function's scope.
#
#   img     : Array{Float32, 3}  — input volume (3D)
#   kernel  : Matrix{Float64}    — Fourier-space Gaussian kernel,
#                                  size (nx, ny) matching img[:,:,k].
#   → returns a fresh Array{Float32, 3} with the LP-filtered result.
function _gaussian_lp_2d_fft(img::AbstractArray{Float32, 3},
                             kernel::AbstractMatrix{Float64})
    nx, ny, nz = size(img)
    result = Array{Float32, 3}(undef, nx, ny, nz)
    Threads.@threads for k in 1:nz
        slice_f64 = Float64.(view(img, :, :, k))
        lp_slice  = Float32.(real.(FFTW.ifft(FFTW.fft(slice_f64) .* kernel)))
        @inbounds @views result[:, :, k] .= lp_slice
    end
    return result
end

"""
    apply_mono_plus(volumes, energies;
                    E_noise_opt=70.0, σ_lp_px=2.0, verbose=true)

Compute Mono+(E) for each energy in `energies` using Grant 2014's
frequency-split rule:

    Mono+(E)   = LP(VMI_E) + VMI_opt − LP(VMI_opt)
    Mono+(E_opt) = VMI_opt  (identity sanity check at the reference)

Returns a NamedTuple with `energies`, `volumes` (same shape as input),
`σ_lp_px`, `E_noise_opt`.

# Arguments
- `volumes`    : Vector{Array{Float32,3}} aligned to `energies`
- `energies`   : Vector of target keV values

# Keyword arguments
- `E_noise_opt` : noise-optimal reference energy (must be in `energies`)
- `σ_lp_px`     : Gaussian LP σ in pixels (effective smoothing scale)
"""
function apply_mono_plus(
        volumes::AbstractVector{<:AbstractArray{Float32, 3}},
        energies::AbstractVector;
        E_noise_opt::Real = 70.0,
        σ_lp_px::Real     = 2.0,
        verbose::Bool     = true,
    )
    i_opt = findfirst(==(Float64(E_noise_opt)), Float64.(energies))
    i_opt === nothing &&
        error("E_noise_opt=$(E_noise_opt) keV not in energies=$(energies)")

    σ_lp = Float64(σ_lp_px)
    vmi_opt = volumes[i_opt]
    nx, ny, _ = size(vmi_opt)

    # Fourier periodic Gaussian kernel (same math as 00 §6.2).
    σ² = σ_lp^2
    fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
    fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]
    kernel = [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]

    t0 = time()
    lp_opt = _gaussian_lp_2d_fft(vmi_opt, kernel)
    hp_opt = vmi_opt .- lp_opt

    result_vols = Vector{Array{Float32, 3}}(undef, length(energies))
    @inbounds for i in 1:length(energies)
        if i == i_opt
            result_vols[i] = copy(vmi_opt)
        else
            lp_E = _gaussian_lp_2d_fft(volumes[i], kernel)
            result_vols[i] = Array{Float32, 3}(lp_E .+ hp_opt)
        end
    end
    dt = time() - t0

    if verbose
        @info "Mono+ (Grant 2014 1:1 parity): σ_LP=$(σ_lp) px (radius ≈ $(round(sqrt(σ_lp); digits = 2)) px), E_opt=$(Int(E_noise_opt)) keV, $(Threads.nthreads()) threads, $(round(dt * 1000; digits = 1)) ms"
        @info "  LP = 2D Gaussian, FFT periodic BC, per-slice  [BEST GUESS — paper unspecified]"
        @info "  HP = (VMI_opt − LP(VMI_opt)), SAME LP applied to each VMI_E, identical σ."
        @info "  sanity: Mono+(E_opt) = VMI_opt identically  (identity at i_opt=$(i_opt))"
    end

    (energies = collect(energies), volumes = result_vols,
     σ_lp_px = σ_lp, E_noise_opt = Float64(E_noise_opt))
end

export apply_mono_plus
