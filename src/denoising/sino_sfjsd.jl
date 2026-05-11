"""
Subspace–Frequency Joint Sinogram Denoiser (SF-JSD).

Two-channel projection-domain denoiser that operates on co-registered
log-line-integral pairs (e.g. dual-kVp DECT, split-filter, PCCT bin
pairs, dual-layer detectors) before any decomposition or
reconstruction.

# Algorithm (per detector row, paper §2.1–2.6)

1. **Per-pixel Poisson whitening** (paper Eq 4).  Subtract a heavy
   low-pass reference `p̄ₖ⁽⁰⁾` (10-px Gaussian, fixed) and scale by
   `√Nₖ`, with per-pixel photon counts `Nₖ = I0,k · exp(-pₖ)`.
   Whitened residuals `ξₖ` have unit per-pixel noise variance.

2. **Per-row SVD** of the whitened (col, view) matrix
   `M_v = [vec(ξ_lo)  vec(ξ_hi)]`.

3. **Joint bilateral filter on both subspaces** with rank-sparse
   bandwidth `σ_e = σ₀ · √(Σ₁/Σ_e)` (paper-fixed exponent γ = ½),
   product-of-channels range kernel (paper Eq 12), 5×5 locally-averaged
   squared diff (Eq 13), MAD-based per-component range scale (Eq 14),
   and stride `s ∈ {1, 2, 3}` derived from the noise correlation length.

4. **Iterative refinement** with `σ₀⁽ᵗ⁾ = 0.7ᵗ · σ₀★`.  Iteration count
   `n_iter ∈ {1, 2}` from `min(N) ≥ 100` rule.

5. **Inverse-whiten** to recover log-line-integrals.

# The single user knob — σ₀

The principal scale.  Set automatically by Stein's unbiased risk
estimator (SURE) with Hutchinson MC divergence and golden-section
search on a representative mid-row.  Pass `σ₀ = 0.0` (default) to
auto-select; pass any positive value to override.

Every other quantity (γ = ½, h = 1 absorbed into σ_e^rng, 5×5 local
averaging window, α = 0.7 iteration decay, σ_ref = 10 px reference
scale) is fixed by RSKR (Clark, Badea 2023) and absorbed into the
operator definition; per-component range scale (MAD), stride (noise
correlation length), and iteration count (min photon count) are
derived from the photon-count map at filter time.

# Reference

Black (in prep.), *Joint Sinogram Denoising via Subspace–Frequency
Reduction for Two-Channel Spectral CT*.

Inspirations:
- Cong, De Man, Wang (2026) — projection-domain Cong solver and the
  Φ_k(ε) effective-spectral-response unification of dual-kVp / PCCT /
  dual-layer / split-filter acquisitions.
- Clark, Badea (2023) — image-domain RSKR (rank-sparse bandwidth,
  product-of-channels range, locally-averaged range, stride).
- Grant et al. (2014) — Mono+ frequency-split rule, applied here to a
  single SVD residual rather than between two reconstructed image stacks.
"""

# =============================================================================
#  Paper-fixed internal constants (§2.4 / §2.6) — not exposed
# =============================================================================

const _SFJSD_α = 0.7f0    # iteration decay
const _SFJSD_lavg = 5        # locally-averaged range window (5×5)
const _SFJSD_σ_ref_px = 10.0f0   # heavy low-pass reference (Gaussian)
const _SFJSD_σ_cap = 12.0f0   # internal compute safety on σ_e (paper §2.9)


# =============================================================================
#  Internal helpers
# =============================================================================

"""
    _sfjsd_sep_gauss_3d(sino, σ_px) -> Array{Float32, 3}

Per-row 2D separable Gaussian smoothing on a (n_col, n_row, n_view)
sinogram.  Each detector row is smoothed independently in (col, view).
Used for the heavy low-pass reference `p̄ₖ⁽⁰⁾`.
"""
function _sfjsd_sep_gauss_3d(sino::Array{Float32, 3}, σ_px::Real)
    σ = Float64(σ_px)
    σ ≤ 0 && return copy(sino)
    radius = max(1, ceil(Int, 3σ))
    ks = Float32[exp(-(k^2) / (2σ^2)) for k in -radius:radius]
    ks ./= sum(ks)
    n_col, n_row, n_view = size(sino)
    out = similar(sino)
    Threads.@threads for r in 1:n_row
        slice = Float32.(@view sino[:, r, :])
        out[:, r, :] .= _sfjsd_gauss_2d(slice, ks, radius)
    end
    return out
end

"""
    _sfjsd_gauss_2d(slice2d, ks, radius) -> Matrix{Float32}

Pure 2D separable Gaussian on a (n_col, n_view) slice — col-pass then
view-pass.  Internal helper for `_sfjsd_sep_gauss_3d`.
"""
function _sfjsd_gauss_2d(
        slice2d::AbstractMatrix{Float32},
        ks::AbstractVector{Float32}, radius::Int
    )
    nc, nv = size(slice2d)
    tmp = Matrix{Float32}(undef, nc, nv)
    out = Matrix{Float32}(undef, nc, nv)
    @inbounds for v in 1:nv, c in 1:nc
        s = 0.0f0; w = 0.0f0
        for (i, dk) in enumerate(-radius:radius)
            c2 = c + dk
            (1 ≤ c2 ≤ nc) || continue
            s += ks[i] * slice2d[c2, v]; w += ks[i]
        end
        tmp[c, v] = s / w
    end
    @inbounds for v in 1:nv, c in 1:nc
        s = 0.0f0; w = 0.0f0
        for (i, dk) in enumerate(-radius:radius)
            v2 = v + dk
            (1 ≤ v2 ≤ nv) || continue
            s += ks[i] * tmp[c, v2]; w += ks[i]
        end
        out[c, v] = s / w
    end
    return out
end

"""
    _sfjsd_mad_scale(arr2d) -> Float32

Robust median-absolute-deviation scale estimate of a 2D residual,
× 1.4826 to match Gaussian σ.  Used as the per-component range
scale `σ_e^rng` (paper Eq 14).
"""
function _sfjsd_mad_scale(arr2d::AbstractMatrix{Float32})
    med = median(arr2d)
    mad = median(abs.(arr2d .- med))
    return Float32(1.4826 * mad)
end

"""
    _sfjsd_pass!(out, target, σ_sp, Λ1, Λ2, σ1_rng, σ2_rng, stride) -> nothing

Joint bilateral pass on a (n_col, n_view) slice.  Range kernel is the
product-of-channels form (paper Eq 12) with 5×5 locally-averaged
squared diff (Eq 13).  Range strength h = 1 is absorbed into the
MAD-derived σ_e^rng; stride is `s` (paper §2.4(iii)).

Spatial neighborhood is a square box of half-radius `ceil(3·σ_sp)`,
sub-sampled at stride `s`.  Each pixel's output is the bilateral
weighted mean over its neighborhood.
"""
function _sfjsd_pass!(
        out::AbstractMatrix{Float32},
        target::AbstractMatrix{Float32},
        σ_sp::Float32,
        Λ1::AbstractMatrix{Float32}, Λ2::AbstractMatrix{Float32},
        σ1_rng::Float32, σ2_rng::Float32, stride::Int
    )
    nc, nv = size(target)
    radius = max(1, ceil(Int, 3 * Float64(σ_sp)))
    inv_2σ²_sp = 1.0f0 / (2.0f0 * σ_sp * σ_sp + 1.0f-30)
    inv_2σ²_r1 = 1.0f0 / (2.0f0 * σ1_rng * σ1_rng + 1.0f-30)
    inv_2σ²_r2 = 1.0f0 / (2.0f0 * σ2_rng * σ2_rng + 1.0f-30)
    half_lavg = _SFJSD_lavg ÷ 2

    Threads.@threads for v in 1:nv
        @inbounds for c in 1:nc
            sum_v = 0.0f0; wtot = 0.0f0
            for dv in -radius:stride:radius
                v2 = v + dv
                (1 ≤ v2 ≤ nv) || continue
                for dc in -radius:stride:radius
                    c2 = c + dc
                    (1 ≤ c2 ≤ nc) || continue

                    # 5×5 locally-averaged squared diff
                    Δ1² = 0.0f0; Δ2² = 0.0f0; cnt = 0
                    for dav in -half_lavg:half_lavg, dac in -half_lavg:half_lavg
                        ca = c + dac;  va = v + dav
                        cb = c2 + dac; vb = v2 + dav
                        (1 ≤ ca ≤ nc && 1 ≤ va ≤ nv) || continue
                        (1 ≤ cb ≤ nc && 1 ≤ vb ≤ nv) || continue
                        d1 = Λ1[cb, vb] - Λ1[ca, va]
                        d2 = Λ2[cb, vb] - Λ2[ca, va]
                        Δ1² += d1 * d1; Δ2² += d2 * d2; cnt += 1
                    end
                    if cnt > 0
                        Δ1² /= cnt; Δ2² /= cnt
                    end

                    spatial_d² = Float32(dc * dc + dv * dv)
                    log_w = -spatial_d² * inv_2σ²_sp -
                        Δ1² * inv_2σ²_r1 -
                        Δ2² * inv_2σ²_r2
                    w = exp(log_w)
                    sum_v += w * target[c2, v2]
                    wtot += w
                end
            end
            out[c, v] = sum_v / max(wtot, 1.0f-30)
        end
    end
    return nothing
end

"""
    _sfjsd_apply_D(M, σ0, n_col, n_view, stride) -> Matrix{Float32}

Single-row forward pass of the full operator D (paper Eq 15).  Used
internally by both the SURE optimization and the main per-row loop.
Returns the denoised (n_col·n_view, 2) matrix.
"""
function _sfjsd_apply_D(
        M::AbstractMatrix{Float32}, σ0::Float32,
        n_col::Int, n_view::Int, stride::Int
    )
    F = svd(M; full = false)
    U, Σ, V = F.U, F.S, F.V

    σ1 = min(σ0, _SFJSD_σ_cap)
    σ2 = min(σ0 * sqrt(Σ[1] / max(Σ[2], 1.0f-12)), _SFJSD_σ_cap)

    Λ1 = reshape(copy(@view U[:, 1]), n_col, n_view)
    Λ2 = reshape(copy(@view U[:, 2]), n_col, n_view)
    σ1_rng = max(_sfjsd_mad_scale(Λ1), eps(Float32))
    σ2_rng = max(_sfjsd_mad_scale(Λ2), eps(Float32))

    Λ1_d = similar(Λ1); Λ2_d = similar(Λ2)
    _sfjsd_pass!(Λ1_d, Λ1, σ1, Λ1, Λ2, σ1_rng, σ2_rng, stride)
    _sfjsd_pass!(Λ2_d, Λ2, σ2, Λ1, Λ2, σ1_rng, σ2_rng, stride)

    U_d = hcat(vec(Λ1_d), vec(Λ2_d))
    return U_d * Diagonal(Σ) * V'
end

"""
    _sfjsd_sure(M, σ0, n_col, n_view, stride) -> Float32

Stein's unbiased risk estimator with Hutchinson MC divergence
(paper Eq 16+17) in whitened identity-covariance coordinates.
"""
function _sfjsd_sure(
        M::AbstractMatrix{Float32}, σ0::Float32,
        n_col::Int, n_view::Int, stride::Int
    )
    Md = _sfjsd_apply_D(M, σ0, n_col, n_view, stride)

    rng = MersenneTwister(42)
    b = randn(rng, Float32, size(M))
    δ = max(Float32(1.0f-3) * Float32(std(M)), Float32(1.0f-8))
    Md_p = _sfjsd_apply_D(M .+ δ .* b, σ0, n_col, n_view, stride)
    div_est = sum(b .* (Md_p .- Md)) / δ

    n = length(M)
    return Float32(sum((Md .- M) .^ 2)) - Float32(n) + 2.0f0 * div_est
end

"""
    _sfjsd_sure_optimize(M, n_col, n_view, stride; σ_lo, σ_hi, tol) -> Float32

Golden-section search for σ_0★ minimizing SURE on a representative
detector row.  Default search range [0.5, 5.0] px covers the typical
clinical optimum across all four hardware classes (paper §2.6).
"""
function _sfjsd_sure_optimize(
        M::AbstractMatrix{Float32},
        n_col::Int, n_view::Int, stride::Int;
        σ_lo::Real = 0.5, σ_hi::Real = 5.0,
        tol::Real = 0.15,
        verbose::Bool = true
    )
    φ = Float32((sqrt(5) - 1) / 2)
    a, b = Float32(σ_lo), Float32(σ_hi)
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fc = _sfjsd_sure(M, c, n_col, n_view, stride)
    fd = _sfjsd_sure(M, d, n_col, n_view, stride)
    n_evals = 2
    while abs(b - a) > tol
        if fc < fd
            b = d; d = c; fd = fc
            c = b - φ * (b - a)
            fc = _sfjsd_sure(M, c, n_col, n_view, stride)
        else
            a = c; c = d; fc = fd
            d = a + φ * (b - a)
            fd = _sfjsd_sure(M, d, n_col, n_view, stride)
        end
        n_evals += 1
    end
    σ_star = (a + b) / 2.0f0
    verbose && @info "[SF-JSD SURE] converged: σ₀★ = $(round(σ_star, digits = 2)) px after $(n_evals) evals"
    return σ_star
end

"""
    _sfjsd_corr_length(p) -> Float64

Noise correlation length (FWHM in detector pixels of the autocovariance
of high-pass log residuals on a flat sinogram patch — typically the
center detector column ± 5 cols at the mid detector row).  Used to
pick the spatial-kernel resampling stride.
"""
function _sfjsd_corr_length(p::Array{Float32, 3})
    n_col, n_row, n_view = size(p)
    c0 = max(1, n_col ÷ 2 - 5); c1 = min(n_col, n_col ÷ 2 + 4)
    mid_r = n_row ÷ 2 + 1
    patch = Float32.(p[c0:c1, mid_r, :])

    σ = 10.0
    radius = ceil(Int, 3σ)
    ks = Float32[exp(-(k^2) / (2σ^2)) for k in -radius:radius]
    ks ./= sum(ks)
    smoothed = similar(patch)
    @inbounds for c_i in axes(patch, 1), v in axes(patch, 2)
        s = 0.0f0; w = 0.0f0
        for (i, dk) in enumerate(-radius:radius)
            v2 = v + dk
            (1 ≤ v2 ≤ size(patch, 2)) || continue
            s += ks[i] * patch[c_i, v2]; w += ks[i]
        end
        smoothed[c_i, v] = s / w
    end
    resid = patch .- smoothed

    n_lags = 5
    ac = zeros(Float32, n_lags)
    for lag in 0:(n_lags - 1)
        s = 0.0; n = 0
        for c_i in 1:(size(resid, 1) - lag), v in axes(resid, 2)
            s += Float64(resid[c_i, v]) * Float64(resid[c_i + lag, v])
            n += 1
        end
        ac[lag + 1] = Float32(s / n)
    end

    ac0 = ac[1]
    ac0 ≤ 0 && return 1.0
    for lag in 1:(n_lags - 1)
        r1 = ac[lag] / ac0
        r2 = ac[lag + 1] / ac0
        if r2 ≤ 0.5
            return Float64((lag - 1) + (r1 - 0.5) / max(r1 - r2, 1.0f-6))
        end
    end
    return Float64(n_lags - 1)
end

"""
    _sfjsd_pick_stride(corr_len) -> Int

Stride from the noise correlation length: <1.5 px → 1, 1.5–2.5 → 2,
≥2.5 → 3 (paper §2.4(iii)).
"""
_sfjsd_pick_stride(corr_len::Real) =
    corr_len < 1.5 ? 1 : (corr_len < 2.5 ? 2 : 3)

"""
    _sfjsd_pick_n_iter(min_N) -> Int

Outer iteration count from the minimum photon count: `min(N) ≥ 100`
gives 1 iter (high-flux dual-kVp regime), else 2 (photon-starved PCCT
bin pairs and dual-layer detectors) (paper §2.5).
"""
_sfjsd_pick_n_iter(min_N::Real) = min_N ≥ 100 ? 1 : 2


# =============================================================================
#  Public API — apply_sino_sfjsd_denoise
# =============================================================================

"""
    apply_sino_sfjsd_denoise(channels, I0; σ₀ = 0.0, verbose = true)
        -> Vector{Array{Float32, 3}}

Two-channel projection-domain SF-JSD joint denoiser.  Operates on a
co-registered pair of log-line-integral sinograms (`channels[1]`,
`channels[2]`) with per-channel scalar reference photon flux
(`I0[1]`, `I0[2]`) — typically `BS.compute_detector_I0(geom, protocol,
sum(spectrum_without_bowtie))` for the corresponding tube /
energy-bin partition.

# Arguments
- `channels::Vector{<:AbstractArray{Float32, 3}}` — the two log-line-
  integral sinograms `(n_col, n_row, n_view)`.  Length must be 2.
- `I0::Vector{<:Real}` — per-channel scalar I₀ (photons / pixel /
  view, before patient).  Length must be 2.

# Keyword arguments
- `σ₀::Real = 0.0` — principal smoothing scale in detector pixels.
  `0.0` ⇒ SURE auto-selects on the mid detector row (recommended).
  Any positive value is used directly.
- `verbose::Bool = true` — log auto-derived stride / n_iter / σ₀★.

# Returns
A `Vector{Array{Float32, 3}}` of denoised log-line-integral sinograms,
one per channel, same shapes as the inputs.

# Example — dual-kVp pair
```julia
e_lo, w_lo = BS.resolve_source_spectrum_without_bowtie(
    sim_opts, protocol_low; scanner = scanner,
)
e_hi, w_hi = BS.resolve_source_spectrum_without_bowtie(
    sim_opts, protocol_high; scanner = scanner,
)
I0_lo = BS.compute_detector_I0(sim_low.geom,  protocol_low,  Float64(sum(w_lo)))
I0_hi = BS.compute_detector_I0(sim_high.geom, protocol_high, Float64(sum(w_hi)))

out = BS.apply_sino_sfjsd_denoise(
    [Float32.(sim_low.sino), Float32.(sim_high.sino)],
    [I0_lo, I0_hi],
)
sino_low_d, sino_high_d = out[1], out[2]
```

# Example — PCCT bin-combined pair
```julia
I0_lo = sum(Float64.(sim_bins.I0_bins[1:2]))     # bins 1+2 → low channel
I0_hi = sum(Float64.(sim_bins.I0_bins[3:4]))     # bins 3+4 → high channel
out = BS.apply_sino_sfjsd_denoise(
    [Float32.(sino_lo_combined), Float32.(sino_hi_combined)],
    [I0_lo, I0_hi],
)
```

# Reference
Black (in prep.), *Joint Sinogram Denoising via Subspace–Frequency
Reduction for Two-Channel Spectral CT*.
"""
function apply_sino_sfjsd_denoise(
        channels::AbstractVector{<:AbstractArray{Float32, 3}},
        I0::AbstractVector{<:Real};
        σ₀::Real = 0.0,
        verbose::Bool = true,
    )
    length(channels) == 2 || error(
        "apply_sino_sfjsd_denoise: requires exactly 2 channels (got $(length(channels))). " *
            "N>2 is paper §2.9 future work."
    )
    length(I0) == 2 || error(
        "apply_sino_sfjsd_denoise: I0 must have length 2 (got $(length(I0)))."
    )
    sz = size(channels[1])
    size(channels[2]) == sz || error(
        "apply_sino_sfjsd_denoise: channel shapes differ — $(size(channels[1])) vs $(size(channels[2]))."
    )

    p_lo = Float32.(channels[1])
    p_hi = Float32.(channels[2])
    n_col, n_row, n_view = size(p_lo)

    I0_lo = Float32(I0[1])
    I0_hi = Float32(I0[2])
    N_lo = I0_lo .* exp.(-p_lo)
    N_hi = I0_hi .* exp.(-p_hi)
    min_N = Float64(min(minimum(N_lo), minimum(N_hi)))

    # ─── Auto-derive stride and n_iter (paper §2.4(iii) / §2.5) ──────────
    corr_len = max(_sfjsd_corr_length(p_lo), _sfjsd_corr_length(p_hi))
    stride = _sfjsd_pick_stride(corr_len)
    n_iter = _sfjsd_pick_n_iter(min_N)

    if verbose
        @info "[SF-JSD auto] I0_lo = $(round(Int, I0_lo)) ph, " *
            "I0_hi = $(round(Int, I0_hi)) ph, " *
            "min(N) = $(round(Int, min_N)) ph"
        @info "[SF-JSD auto] noise corr length = $(round(corr_len, digits = 2)) px → stride = $(stride)"
        @info "[SF-JSD auto] n_iter = $(n_iter) (rule: 1 if min(N) ≥ 100 everywhere)"
    end

    # ─── Heavy low-pass reference (paper §2.2 fixed σ_ref = 10 px) ───────
    p_ref_lo = _sfjsd_sep_gauss_3d(p_lo, _SFJSD_σ_ref_px)
    p_ref_hi = _sfjsd_sep_gauss_3d(p_hi, _SFJSD_σ_ref_px)

    # ─── Whitened residuals ξ_k = √N_k · (p_k − p̄_k⁽⁰⁾) ─────────────────
    w_lo = sqrt.(max.(N_lo, 1.0f0))
    w_hi = sqrt.(max.(N_hi, 1.0f0))
    ξ_lo = w_lo .* (p_lo .- p_ref_lo)
    ξ_hi = w_hi .* (p_hi .- p_ref_hi)

    p_lo = nothing; p_hi = nothing
    N_lo = nothing; N_hi = nothing
    GC.gc(true)

    # ─── Auto-derive σ_0★ via SURE on the mid-row, or use user's value. ─
    σ₀_user = Float32(σ₀)
    σ₀_star = if σ₀_user > 0
        verbose && @info "[SF-JSD] σ₀ from user knob: $(σ₀_user) px (SURE skipped)"
        σ₀_user
    else
        mid_r = n_row ÷ 2 + 1
        slice_lo = Float32.(@view ξ_lo[:, mid_r, :])
        slice_hi = Float32.(@view ξ_hi[:, mid_r, :])
        M_mid = hcat(vec(slice_lo), vec(slice_hi))
        verbose && @info "[SF-JSD SURE] running on mid-row r=$(mid_r), σ ∈ [0.5, 5.0] px..."
        _sfjsd_sure_optimize(M_mid, n_col, n_view, stride; verbose = verbose)
    end

    # ─── Run the operator with σ_0^(t) = α^t · σ_0★ ──────────────────────
    Σ_ratio_log = Vector{Float32}(undef, n_row)
    σ0 = σ₀_star
    for t in 0:(n_iter - 1)
        Threads.@threads for r in 1:n_row
            slice_lo = Float32.(@view ξ_lo[:, r, :])
            slice_hi = Float32.(@view ξ_hi[:, r, :])
            M = hcat(vec(slice_lo), vec(slice_hi))

            F = svd(M; full = false)
            U, Σ, V = F.U, F.S, F.V

            σ1 = min(σ0, _SFJSD_σ_cap)
            σ2 = min(σ0 * sqrt(Σ[1] / max(Σ[2], 1.0f-12)), _SFJSD_σ_cap)
            Σ_ratio_log[r] = Float32(Σ[1] / max(Σ[2], 1.0f-12))

            Λ1 = reshape(copy(@view U[:, 1]), n_col, n_view)
            Λ2 = reshape(copy(@view U[:, 2]), n_col, n_view)
            σ1_rng = max(_sfjsd_mad_scale(Λ1), eps(Float32))
            σ2_rng = max(_sfjsd_mad_scale(Λ2), eps(Float32))

            Λ1_d = similar(Λ1); Λ2_d = similar(Λ2)
            _sfjsd_pass!(Λ1_d, Λ1, σ1, Λ1, Λ2, σ1_rng, σ2_rng, stride)
            _sfjsd_pass!(Λ2_d, Λ2, σ2, Λ1, Λ2, σ1_rng, σ2_rng, stride)

            U_d = hcat(vec(Λ1_d), vec(Λ2_d))
            M_d = U_d * Diagonal(Σ) * V'
            ξ_lo[:, r, :] .= reshape(view(M_d, :, 1), n_col, n_view)
            ξ_hi[:, r, :] .= reshape(view(M_d, :, 2), n_col, n_view)
        end
        verbose && @info "[SF-JSD iter $(t + 1)/$(n_iter)] σ₀ = $(round(σ0, digits = 2)) px, " *
            "median Σ₁/Σ₂ = $(round(median(Σ_ratio_log), sigdigits = 3)), " *
            "σ_cap = $(_SFJSD_σ_cap) px"
        σ0 *= _SFJSD_α
    end

    # ─── Inverse-whiten: p = ξ/√N + p̄⁽⁰⁾ ─────────────────────────────────
    p_lo_d = ξ_lo ./ w_lo .+ p_ref_lo
    p_hi_d = ξ_hi ./ w_hi .+ p_ref_hi

    return [p_lo_d, p_hi_d]
end


export apply_sino_sfjsd_denoise
