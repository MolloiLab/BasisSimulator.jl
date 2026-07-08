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
    c_sq = c_a_f^2 + c_b_f^2
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
        (
            "Tikhonov (λ=$(λ_f), radius≈$(round(sqrt(λ_f); digits = 2)) px)",
            [1.0 / (1.0 + λ_f * (kx_sq[i] + ky_sq[j])) for i in 1:n1, j in 1:n2],
        )
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

    σ_sorth = std(s_orth)
    σ_smooth = std(s_smooth)
    σ_n = std(n_orth)

    if verbose
        @info "ACNR smoother=$(smoother_name), γ=$(γ_f), c_a=$(round(c_a_f; sigdigits = 3)), c_b=$(round(c_b_f; sigdigits = 3)), $(round(dt * 1000; digits = 1)) ms"
        @info "  kept (signal): std(s_smooth)=$(round(σ_smooth; sigdigits = 3))   ($(round(100 * σ_smooth / max(σ_sorth, eps()); digits = 1))% of s_⊥ retained)"
        @info "  removed (noise): std(n_⊥)=$(round(σ_n; sigdigits = 3))             ($(round(100 * σ_n / max(σ_sorth, eps()); digits = 1))% of s_⊥ extracted as noise)"
    end

    return (
        σ_n_orth = σ_n,
        σ_s_orth = σ_sorth,
        σ_s_smooth = σ_smooth,
        smoother = smoother_name,
        γ = Float64(γ_f),
    )
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
    return (a_out, b_out, info)
end

"""
    apply_image_acnr!(W, I; gamma=1.0, bilat_radius=3, bilat_sigma_s=2.0,
                      bilat_range_k=2.5) -> info

Image-domain, **data-adaptive** Anti-Correlated Noise Reduction for a basis-map
pair — water `W` and iodine `I` (3-D volumes), modified IN PLACE.  The
image-domain counterpart of [`apply_acnr!`] (which is sinogram-domain).

Material decomposition stamps strongly anti-correlated noise on the basis maps
(`ρ_basis < 0`); that anti-correlation *is* the VMI-noise "U".  Rather than
assuming the signal/noise axes, this LEARNS them from the joint W–I covariance
via a closed-form 2×2 eigen-rotation:

* **e1** (large-variance axis = correlated structure) is kept **pixel-perfect**;
* **e2** (small-variance axis = the anti-correlated noise) is denoised with a
  joint bilateral filter guided by BOTH basis maps, so real water/iodine edges
  (Δ ≳ `bilat_range_k`·σ_noise) survive — only locally-flat anti-correlated
  noise is removed.  Resolution is preserved two ways (pixel-perfect e1 +
  edge-aware e2).

# Keyword arguments (all knobs)
- `gamma::Real = 1.0`         : strength ∈ [0,1]; 0 ⇒ identity (no-op).
- `bilat_radius::Integer = 3` : bilateral spatial window radius (px).
- `bilat_sigma_s::Real = 2.0` : bilateral spatial Gaussian σ (px).
- `bilat_range_k::Real = 2.5` : range σ = k · per-basis noise std (edge threshold).

# Returns
`info::NamedTuple` with `θ_deg`, `ρ_struct`, `σ_W`, `σ_I`, `γ` (diagnostics).
`W` and `I` are updated in place.
"""
function apply_image_acnr!(
        W::AbstractArray{T, 3},
        I::AbstractArray{T, 3};
        gamma::Real = 1.0,
        bilat_radius::Integer = 3,
        bilat_sigma_s::Real = 2.0,
        bilat_range_k::Real = 2.5,
    ) where {T}
    size(I) == size(W) ||
        error("apply_image_acnr!: W $(size(W)) and I $(size(I)) must match shape.")
    γ = T(gamma)
    nx, ny, nz = size(W)

    # robust per-basis noise std (adjacent-column difference / √2)
    _nstd = function (V)
        s = 0.0; m = 0
        @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
            d = V[i, j, k] - V[i - 1, j, k]; s += d * d; m += 1
        end
        sqrt(s / max(m, 1)) / sqrt(2)
    end
    σW = max(_nstd(W), 1.0e-8)   # Float64 (matches the inline cell's arithmetic)
    σI = max(_nstd(I), 1.0e-8)

    # covariance of the joint (W, I) field → eigen-rotation angle θ
    mW = T(sum(W) / length(W))
    mI = T(sum(I) / length(I))
    a = 0.0; bb = 0.0; c = 0.0
    @inbounds for idx in eachindex(W)
        dw = Float64(W[idx] - mW); di = Float64(I[idx] - mI)
        a += dw * dw; bb += dw * di; c += di * di
    end
    ρ_struct = bb / sqrt(max(a, 1.0e-30) * max(c, 1.0e-30))

    if γ <= 0   # identity
        return (θ_deg = 0.0, ρ_struct = ρ_struct,
                σ_W = Float64(σW), σ_I = Float64(σI), γ = Float64(γ))
    end

    θ  = 0.5 * atan(2bb, a - c)               # e1 = (cosθ, sinθ) → larger eigenvalue
    ct = T(cos(θ)); st = T(sin(θ))

    # rotate centered (W, I) into (signal, noise) axes
    p_sig   = Array{T}(undef, nx, ny, nz)
    p_noise = Array{T}(undef, nx, ny, nz)
    @inbounds for idx in eachindex(W)
        dw = W[idx] - mW; di = I[idx] - mI
        p_sig[idx]   =  ct * dw + st * di
        p_noise[idx] = -st * dw + ct * di
    end

    # joint-bilateral denoise of the noise axis, guided by BOTH basis maps
    denW = T(2 * (bilat_range_k * σW)^2)
    denI = T(2 * (bilat_range_k * σI)^2)
    r    = Int(bilat_radius)
    σs2  = T(2 * bilat_sigma_s^2)
    sw   = T[exp(-(di * di + dj * dj) / σs2) for di in -r:r, dj in -r:r]

    p_noise_s = Array{T}(undef, nx, ny, nz)
    @inbounds for k in 1:nz
        for j in 1:ny, i in 1:nx
            Wc = W[i, j, k]; Ic = I[i, j, k]
            acc = zero(T); wsum = zero(T)
            for dj in -r:r
                jj = j + dj; (1 ≤ jj ≤ ny) || continue
                for di in -r:r
                    ii = i + di; (1 ≤ ii ≤ nx) || continue
                    dW = W[ii, jj, k] - Wc; dI = I[ii, jj, k] - Ic
                    wr = exp(-(dW * dW / denW + dI * dI / denI))
                    w  = sw[di + r + 1, dj + r + 1] * wr
                    acc += w * p_noise[ii, jj, k]; wsum += w
                end
            end
            p_noise_s[i, j, k] = acc / wsum
        end
    end

    # reconstitute: keep p_sig (e1) pixel-perfect, γ-blend the denoised noise
    # axis, rotate back (Rᵀ), restore means.
    @inbounds for idx in eachindex(W)
        pn = p_noise[idx] + γ * (p_noise_s[idx] - p_noise[idx])
        ps = p_sig[idx]
        W[idx] = mW + ct * ps - st * pn
        I[idx] = mI + st * ps + ct * pn
    end

    return (θ_deg = rad2deg(θ), ρ_struct = ρ_struct,
            σ_W = Float64(σW), σ_I = Float64(σI), γ = Float64(γ))
end

"""
    apply_image_acnr(W, I; kwargs...) -> (W_out, I_out, info)

Allocating wrapper around [`apply_image_acnr!`].
"""
function apply_image_acnr(
        W::AbstractArray{T, 3},
        I::AbstractArray{T, 3};
        kwargs...,
    ) where {T}
    W_out = copy(W); I_out = copy(I)
    info = apply_image_acnr!(W_out, I_out; kwargs...)
    return (W_out, I_out, info)
end

export apply_acnr!, apply_acnr
export apply_image_acnr!, apply_image_acnr

# =============================================================================
# Kalender-1988 TRUE ACNR — per-pixel regression, zero structural blur
# =============================================================================

"""
    apply_acnr_kalender!(W, I; hp_sigma_px=1.5, window=2, beta_max=8.0) -> info

TRUE anti-correlated noise reduction (Kalender, Klotz & Kostaridou 1988,
the original dual-energy ACNR): per-pixel LOCAL linear regression between
the two basis maps' high-frequency channels.

Physics: dual-energy basis noise is ANTI-correlated (`cov(n_W, n_I) < 0`)
while anatomical structure is positively correlated.  For each pixel take
`hW = W − G(W)`, `hI = I − G(I)` (high-pass, σ = `hp_sigma_px`), estimate
the local regression slope `β = cov(hI, hW)/var(hW)` in a `(2w+1)²`
window (the window sizes only the ESTIMATE of β — the correction itself
stays a per-pixel linear combination, so no signal smoothing occurs;
larger windows reduce β̂ variance, whose noise otherwise re-introduces a
small positive residual correlation that flips the VMI noise-vs-keV
tail upward), and CLAMP it to `[−beta_max, 0]`:

  * noise-dominated pixels → β ≈ −σ_I/σ_W → `I − β·hW` cancels the
    predictable anti-correlated component EXACTLY (pixelwise subtraction,
    no smoothing operator ever multiplies the signal);
  * structure-dominated pixels → local cov > 0 → β clamps to 0 → the
    pixel is BIT-UNTOUCHED.

Resolution preservation is therefore by construction, not by an
edge-stopping heuristic.  Symmetric correction is applied to `W` from
`hI`.  Returns `(ρ_hp, σ_hW, σ_hI)` global HF statistics.
"""
function apply_acnr_kalender!(
        W::AbstractArray{T, 3},
        I::AbstractArray{T, 3};
        hp_sigma_px::Real = 1.5,
        window::Int = 4,
        beta_max::Real = 8.0,
        passes::Int = 2,
    ) where {T <: AbstractFloat}
    # Multi-pass: the one-sided clamp on a NOISY local β̂ under-corrects on
    # average (β̂ > 0 pixels keep their full anti-correlated noise), leaving
    # residual C_WI < 0 that puts the VMI noise-vs-keV minimum inside the
    # clinical range (tail flips up at 140 keV).  Each pass shrinks the
    # residual geometrically; the correction stays a per-pixel linear
    # combination — zero signal smoothing regardless of pass count.
    if passes > 1
        info = nothing
        for _ in 1:passes
            info = apply_acnr_kalender!(W, I; hp_sigma_px = hp_sigma_px,
                window = window, beta_max = beta_max, passes = 1)
        end
        return info
    end

    nx, ny, nz = size(W)
    # high-pass via small separable Gaussian (CPU FFT fine — done once)
    G(vol) = begin
        out = similar(vol)
        σ = Float64(hp_sigma_px)
        r = max(2, ceil(Int, 3σ))
        k = T.(exp.(-(collect(-r:r) .^ 2) ./ (2σ^2)))
        k ./= sum(k)
        tmp = similar(vol)
        # x
        for z in 1:nz, j in 1:ny, i in 1:nx
            acc = zero(T)
            for t in -r:r
                ii = clamp(i + t, 1, nx)
                acc += k[t + r + 1] * vol[ii, j, z]
            end
            tmp[i, j, z] = acc
        end
        # y
        for z in 1:nz, j in 1:ny, i in 1:nx
            acc = zero(T)
            for t in -r:r
                jj = clamp(j + t, 1, ny)
                acc += k[t + r + 1] * tmp[i, jj, z]
            end
            out[i, j, z] = acc
        end
        out
    end

    Wc = Array(W); Ic = Array(I)
    hW = Wc .- G(Wc)
    hI = Ic .- G(Ic)

    w = window
    # Variance-limited regression bound (Kalender's original): |β| may not
    # exceed the GLOBAL HF noise ratio — an unbounded local β̂ overshoots on
    # noisy covariance estimates, over-subtracting until the residual pair
    # is positively correlated (the VMI noise-vs-keV tail flips upward).
    λ_I = sqrt(sum(abs2, hI) / max(sum(abs2, hW), T(1.0e-30)))
    λ_W = sqrt(sum(abs2, hW) / max(sum(abs2, hI), T(1.0e-30)))
    βmaxI = min(T(beta_max), λ_I)
    βmaxW = min(T(beta_max), λ_W)
    outW = copy(Wc); outI = copy(Ic)
    for z in 1:nz
        for j in 1:ny, i in 1:nx
            sWW = zero(T); sII = zero(T); sWI = zero(T)
            n = 0
            for dj in -w:w, di in -w:w
                ii = i + di; jj = j + dj
                (1 <= ii <= nx && 1 <= jj <= ny) || continue
                a = hW[ii, jj, z]; b = hI[ii, jj, z]
                sWW += a * a; sII += b * b; sWI += a * b
                n += 1
            end
            # regression of hI on hW (for I) and hW on hI (for W), clamped ≤ 0
            βI = clamp(sWI / max(sWW, T(1.0e-20)), -βmaxI, zero(T))
            βW = clamp(sWI / max(sII, T(1.0e-20)), -βmaxW, zero(T))
            outI[i, j, z] = Ic[i, j, z] - βI * hW[i, j, z]
            outW[i, j, z] = Wc[i, j, z] - βW * hI[i, j, z]
        end
    end
    copyto!(W, outW); copyto!(I, outI)

    ρ = sum(hW .* hI) / sqrt(max(sum(hW .^ 2) * sum(hI .^ 2), T(1.0e-30)))
    return (ρ_hp = ρ, σ_hW = sqrt(sum(hW .^ 2) / length(hW)), σ_hI = sqrt(sum(hI .^ 2) / length(hI)))
end

export apply_acnr_kalender!
