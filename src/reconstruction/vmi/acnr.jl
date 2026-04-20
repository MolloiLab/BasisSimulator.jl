"""
Sinogram-domain Anti-Correlated Noise Reduction (ACNR).

Projects noise onto the direction orthogonal to the signal direction
`(p(E_ref), q(E_ref))` in basis-sinogram space, smooths it, and projects
back into the basis pair with opposite signs.  By construction,
`p·Δa + q·Δc = 0`, so `μ(E_ref)` at every pixel is pixel-perfect
preserved regardless of the smoother — zero resolution loss at `E_ref`
is a structural guarantee.

Smoother = FFT-based Tikhonov (exact one-shot solve):

    ŝ_smooth[p,q] = ŝ_⊥[p,q] / (1 + λ · μ_{p,q})
    μ_{p,q}       = 4·(sin²(πp/nx) + sin²(πq/nv))

Effective smoothing radius ≈ √λ px.  λ has monotonic effect at every
magnitude (no iterative-solver saturation).

Reference:
  Kalender, Klotz, Kostaridou (1988) *"An algorithm for noise suppression
  in dual energy CT material density images."*
  IEEE Trans Med Imaging 7(3):218–224.  DOI 10.1109/42.7784.
"""

import FFTW

"""
    apply_acnr!(a, c; E_ref=70.0, λ=4.0, γ=1.0, verbose=true)

FFT-Tikhonov ACNR on a basis-sinogram pair.  Mutates `a` and `c` in place.

# Keyword arguments
- `E_ref` : reference energy (keV) at which `μ` is preserved
- `λ`     : Tikhonov strength — effective radius ≈ √λ px
- `γ`     : projection strength ∈ [0, 1]

Returns a NamedTuple with diagnostic `σ_n_orth`, `σ_s_orth`,
`σ_s_smooth`, plus echoed `E_ref`, `λ`, `γ`.
"""
function apply_acnr!(
        a::AbstractArray{Float32, 3},
        c::AbstractArray{Float32, 3};
        E_ref::Real = 70.0,
        λ::Real     = 4.0,
        γ::Real     = 1.0,
        verbose::Bool = true,
    )
    p_E = Float32(p_photoelectric(Float64(E_ref)))
    q_E = Float32(q_compton(Float64(E_ref)))
    u_sq = p_E^2 + q_E^2   # |u_sig|²

    # s_⊥ = anti-correlated noise channel (Float64 for FFT precision).
    s_orth = @. Float64(-q_E * a + p_E * c)
    nx, nv, nr = size(s_orth)

    # Fourier Tikhonov denominator: 1 + λ·μ_{p,q}. Built once, reused.
    λf = Float64(λ)
    kx_sq = [4 * sin(π * (i - 1) / nx)^2 for i in 1:nx]
    ky_sq = [4 * sin(π * (j - 1) / nv)^2 for j in 1:nv]
    denom = [1.0 + λf * (kx_sq[i] + ky_sq[j]) for i in 1:nx, j in 1:nv]

    t0 = time()
    s_smooth = similar(s_orth)
    Threads.@threads for k in 1:nr
        s_k = Float64.(@view s_orth[:, :, k])
        s_smooth[:, :, k] .= real.(FFTW.ifft(FFTW.fft(s_k) ./ denom))
    end
    dt = time() - t0

    # Residual = noise removed; project back into basis pair.
    n_orth = Float32.(s_orth .- s_smooth)
    γf = Float32(γ)
    @. a = a + γf * q_E / u_sq * n_orth
    @. c = c - γf * p_E / u_sq * n_orth

    σ_sorth  = std(s_orth)
    σ_smooth = std(s_smooth)
    σ_n      = std(n_orth)

    if verbose
        @info "ACNR (FFT Tikhonov smoother): E_ref=$(Int(E_ref)) keV, λ=$(λf) (radius ≈ $(round(sqrt(λf); digits = 2)) px), γ=$(γf), $(Threads.nthreads()) threads, $(round(dt * 1000; digits = 1)) ms"
        @info "  spectral weights @E_ref:  p=$(round(Float64(p_E); sigdigits = 3))  q=$(round(Float64(q_E); sigdigits = 3))  |u|²=$(round(Float64(u_sq); sigdigits = 3))"
        @info "  smoother kept (signal):   std(s_smooth)=$(round(σ_smooth; sigdigits = 3))    ($(round(100 * σ_smooth / max(σ_sorth, eps()); digits = 1))% of s_⊥ retained)"
        @info "  smoother removed (noise): std(n_⊥)=$(round(σ_n; sigdigits = 3))              ($(round(100 * σ_n / max(σ_sorth, eps()); digits = 1))% of s_⊥ extracted as noise)"
        @info "  μ(E_ref) preserved per-pixel by construction (correction ⊥ signal direction)."
    end

    (σ_n_orth  = σ_n,
     σ_s_orth  = σ_sorth,
     σ_s_smooth = σ_smooth,
     E_ref     = Float64(E_ref),
     λ         = λf,
     γ         = Float64(γ))
end

export apply_acnr!
