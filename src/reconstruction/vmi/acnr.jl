"""
Sinogram-domain Anti-Correlated Noise Reduction (ACNR).

Projects noise onto the direction orthogonal to the signal direction
`(c_a, c_b)` in basis-sinogram space, smooths it, and projects back into
the basis pair with opposite signs.  By construction, `c_a·Δa + c_b·Δb
= 0`, so `μ(E_ref)` at every pixel is pixel-perfect preserved regardless
of the smoother — zero resolution loss at `E_ref` is a structural
guarantee.

Two smoother families are supported (pick one):
  • `σ`  — 2D Gaussian LP via FFT (kernel = `exp(-2π²σ²(f_x²+f_y²))`).
           Radius ≈ σ px.  Matches notebook 06/07 current pattern.
  • `λ`  — FFT Tikhonov (denominator `1 + λ·μ_{p,q}`).
           Effective radius ≈ √λ px.

The function is **basis-agnostic**: pass in the pair of coefficients at
the reference energy.  The two canonical DE bases that show up in the
codebase:

  * photo/Compton (Cong):    `c_a = p_photoelectric(E_ref)`,  `c_b = q_compton(E_ref)`
  * material (water,iodine): `c_a = μρ_water(E_ref)`,         `c_b = μρ_iodine(E_ref)`

As long as the (a, b) arrays match the basis convention the coefficients
describe, the math is identical.

Reference:
  Kalender, Klotz, Kostaridou (1988) IEEE TMI 7(3):218–224.
"""

import FFTW

"""
    apply_acnr!(sino_a, sino_b;
                c_a, c_b,
                σ = nothing,
                λ = nothing,
                γ = 1.0,
                verbose = true) -> NamedTuple

In-place ACNR on a basis-sinogram pair.  The signal direction is
`(c_a, c_b)`; ACNR attenuates only the orthogonal component.

Exactly one of `σ` (Gaussian LP) or `λ` (Tikhonov) must be specified.

# Arguments
- `sino_a`, `sino_b` : Float32 material sinograms (mutated in place)

# Keyword arguments
- `c_a`, `c_b`  : signal-direction coefficients at the reference energy
- `σ`           : Gaussian LP kernel σ in pixels (if using Gaussian)
- `λ`           : FFT Tikhonov strength (if using Tikhonov)
- `γ`           : correction strength ∈ [0, 1]; γ=0 ⇒ identity

Returns a NamedTuple of diagnostics (smoother choice, γ, σ_n_orth,
σ_s_orth, σ_s_smooth).
"""
function apply_acnr!(
        sino_a::AbstractArray{Float32, 3},
        sino_b::AbstractArray{Float32, 3};
        c_a::Real,
        c_b::Real,
        σ::Union{Real, Nothing} = nothing,
        λ::Union{Real, Nothing} = nothing,
        γ::Real = 1.0,
        verbose::Bool = true,
    )
    (σ === nothing) ⊻ (λ === nothing) ||
        error("apply_acnr!: specify exactly one of σ (Gaussian LP) or λ (Tikhonov).")

    c_a_f = Float64(c_a)
    c_b_f = Float64(c_b)
    c_sq  = c_a_f^2 + c_b_f^2
    c_sq > 0 || error("apply_acnr!: (c_a, c_b) both zero; signal direction undefined.")

    # s_⊥ = signal-orthogonal channel  (noise-only when c_a, c_b are the
    # μ coefficients at E_ref).
    s_orth = @. Float64(-c_b_f * sino_a + c_a_f * sino_b)
    n1, n2, n3 = size(s_orth)

    # Build the smoother transfer function in Fourier space.
    smoother_name, transfer = if σ !== nothing
        σ_f = Float64(σ)
        fx = [min(i - 1, n1 - (i - 1)) / n1 for i in 1:n1]
        fy = [min(j - 1, n2 - (j - 1)) / n2 for j in 1:n2]
        kernel = [exp(-2π^2 * σ_f^2 * (fx[i]^2 + fy[j]^2)) for i in 1:n1, j in 1:n2]
        ("Gaussian LP (σ=$(σ_f) px)", kernel)
    else
        λ_f = Float64(λ)
        kx_sq = [4 * sin(π * (i - 1) / n1)^2 for i in 1:n1]
        ky_sq = [4 * sin(π * (j - 1) / n2)^2 for j in 1:n2]
        ("Tikhonov (λ=$(λ_f), radius≈$(round(sqrt(λ_f); digits=2)) px)",
         [1.0 / (1.0 + λ_f * (kx_sq[i] + ky_sq[j])) for i in 1:n1, j in 1:n2])
    end

    t0 = time()
    s_smooth = similar(s_orth)
    Threads.@threads for k in 1:n3
        slice = Float64.(@view s_orth[:, :, k])
        s_smooth[:, :, k] .= real.(FFTW.ifft(FFTW.fft(slice) .* transfer))
    end
    dt = time() - t0

    # Residual = high-freq component (the "noise"); project back with
    # opposite sign contributions into (a, b).
    n_orth = Float32.(s_orth .- s_smooth)
    γ_f = Float32(γ)
    α_a = Float32(γ_f * c_b_f / c_sq)
    α_b = Float32(γ_f * c_a_f / c_sq)
    @. sino_a = sino_a + α_a * n_orth
    @. sino_b = sino_b - α_b * n_orth

    σ_sorth  = std(s_orth)
    σ_smooth = std(s_smooth)
    σ_n      = std(n_orth)

    if verbose
        @info "ACNR smoother=$(smoother_name), γ=$(γ_f), c_a=$(round(c_a_f; sigdigits=3)), c_b=$(round(c_b_f; sigdigits=3)), $(round(dt * 1000; digits = 1)) ms"
        @info "  kept (signal): std(s_smooth)=$(round(σ_smooth; sigdigits=3))   ($(round(100 * σ_smooth / max(σ_sorth, eps()); digits=1))% of s_⊥ retained)"
        @info "  removed (noise): std(n_⊥)=$(round(σ_n; sigdigits=3))             ($(round(100 * σ_n / max(σ_sorth, eps()); digits=1))% of s_⊥ extracted as noise)"
    end

    (σ_n_orth   = σ_n,
     σ_s_orth   = σ_sorth,
     σ_s_smooth = σ_smooth,
     smoother   = smoother_name,
     γ          = Float64(γ_f))
end

"""
    apply_acnr(sino_a, sino_b; kwargs...) -> (sino_a_corr, sino_b_corr, info)

Allocating wrapper around `apply_acnr!`.
"""
function apply_acnr(
        sino_a::AbstractArray{Float32, 3},
        sino_b::AbstractArray{Float32, 3};
        kwargs...,
    )
    a_out = copy(sino_a)
    b_out = copy(sino_b)
    info = apply_acnr!(a_out, b_out; kwargs...)
    (a_out, b_out, info)
end

export apply_acnr!, apply_acnr
