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
    nx, ny, nz = size(vmi_opt)

    # Fourier periodic Gaussian kernel (same math as 00 §6.2).
    σ² = σ_lp^2
    fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
    fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]
    kernel = [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]

    # 2D Gaussian LP via FFT (exact, one-shot per slice).
    # NOTE: the inner buffer is named `lp_buf` (not `out`) because the
    # outer loop below also uses `out` for the result vector.  Inside a
    # top-level `function ... end`, Julia's scope analysis unifies
    # same-named variables between an anonymous closure and its
    # enclosing function, so using `out` in both places causes the
    # outer `out[i] = ...` to try setindex! on the inner 3D array
    # instead of the outer Vector.  (The notebook's `let ... end`
    # version uses a different outer name (`volumes`) so the clash
    # never surfaced there.)
    gaussian_lp_2d_fft = function (img::AbstractArray{Float32, 3})
        nxl, nyl, nzl = size(img)
        lp_buf = similar(img)
        Threads.@threads for k in 1:nzl
            slice = Float64.(@view img[:, :, k])
            lp_buf[:, :, k] .= Float32.(real.(FFTW.ifft(FFTW.fft(slice) .* kernel)))
        end
        lp_buf
    end

    t0 = time()
    lp_opt = gaussian_lp_2d_fft(vmi_opt)
    hp_opt = vmi_opt .- lp_opt

    out = Vector{Array{Float32, 3}}(undef, length(energies))
    for (i, E) in enumerate(energies)
        if i == i_opt
            out[i] = copy(vmi_opt)
        else
            lp_E = gaussian_lp_2d_fft(volumes[i])
            out[i] = lp_E .+ hp_opt
        end
    end
    dt = time() - t0

    if verbose
        @info "Mono+ (Grant 2014 1:1 parity): σ_LP=$(σ_lp) px (radius ≈ $(round(sqrt(σ_lp); digits = 2)) px), E_opt=$(Int(E_noise_opt)) keV, $(Threads.nthreads()) threads, $(round(dt * 1000; digits = 1)) ms"
        @info "  LP = 2D Gaussian, FFT periodic BC, per-slice  [BEST GUESS — paper unspecified]"
        @info "  HP = (VMI_opt − LP(VMI_opt)), SAME LP applied to each VMI_E, identical σ."
        @info "  sanity: Mono+(E_opt) = VMI_opt identically  (identity at i_opt=$(i_opt))"
    end

    (energies = collect(energies), volumes = out,
     σ_lp_px = σ_lp, E_noise_opt = Float64(E_noise_opt))
end

export apply_mono_plus
