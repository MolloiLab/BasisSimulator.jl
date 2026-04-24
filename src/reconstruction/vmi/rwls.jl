"""
RWLS-GN (Reweighted Least Squares — Gauss-Newton) for material basis
decomposition.

Ducros 2017 style count-domain Gauss-Newton inversion of the polychromatic
forward model, with an optional per-slice FFT proximal quadratic prior.
Warm-started from a cheap decomp (CMV polynomial, Cong, …).

**Unified N-bin form.** One function covers both:
  • PCCT: N bins (e.g. A=1+2, B=3, C=4) with explicit per-bin I0 photon counts.
  • Classical DE: N = 2 bins (low/high kVp) — pass `bins = [counts_low,
    counts_high]` and I0 as the measured (or assumed) photon counts per bin.

Per-pixel forward model:

    F_m(s_I, s_W) = I0_m · Σ_k ŵ_m[k] · exp(−p[k]·s_I − q[k]·s_W)

Per-pixel 2×2 Gauss-Newton step on the N-equation / 2-unknown system with
Poisson inverse-variance weights in count space:

    r_m   = s_m − F_m
    w_m   = 1 / max(F_m, 1)                          (one-count floor)
    H     = Σ_m w_m · J_mᵀ J_m      (2×2)
    g     = Σ_m w_m · J_mᵀ r_m      (2-vector)
    Δs    = clamp.(H⁻¹ · g, ±step_lim),  applied with relaxation

followed by a per-slice FFT quadratic proximal prior:

    ŝ_smooth[p,q] = ŝ[p,q] / (1 + α·β · μ_{p,q})

with `μ_{p,q}` the 2D lattice Laplacian eigenvalues.

Reference:
  Ducros, Bussod, Sixou, Trouvé, Peyrin, Nuyts (2017)
  "Regularization of nonlinear decomposition of spectral X-ray projection
  images."  *Med. Phys.* 44(9):e174–e187.
"""

"""
    apply_rwls!(sino_iodine, sino_water, bins, I0;
                basis,
                n_iter = 3,
                α = 0.3, β_iodine = 1.0, β_water = 1.0,
                step_lim_iodine = 0.75, step_lim_water = 5.0,
                relax = 0.5,
                verbose = true) -> NamedTuple

In-place N-bin count-domain RWLS-GN.  Mutates `sino_iodine` and `sino_water`
(warm-started by the caller — typically from a CMV polynomial init).

# Arguments
- `sino_iodine`, `sino_water` : Float32 material sinograms (g/cm²),
  shape `[n_col, n_row, n_view]`.  Mutated in place.
- `bins`  : `Vector` of Float32 count arrays, one per bin group.  Each
  shares shape with `sino_iodine`.  Counts
  `s_m = Σ_{b ∈ group_m} I0_b · exp(-h_b)` (scatter-correct beforehand).
  For classical DE use `bins = [counts_low, counts_high]`.
- `I0`    : `Vector` of per-bin `I0_m` scalars, same length as `bins`.

# Keyword arguments
- `basis` : NamedTuple with
    - `ŵ_bins::Vector{<:AbstractVector{Float32}}` — N spectral weight
      vectors, each length n_E.  Internally renormalized so each sums to 1.
    - `p::AbstractVector{Float32}` — iodine μρ at each spectrum energy.
    - `q::AbstractVector{Float32}` — water  μρ at each spectrum energy.
- `n_iter`            : outer GN iterations (default 3).
- `α`                 : FFT proximal weight (0 disables smoothing).
- `β_iodine`, `β_water` : per-basis smoothing scales.
- `step_lim_iodine`, `step_lim_water` : per-iter |Δ| clamp (g/cm²).
- `relax`             : GN relaxation factor (default 0.5).

Returns `(n_iter, α, β_iodine, β_water, n_bins)`.
"""
function apply_rwls!(
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        bins::AbstractVector,
        I0::AbstractVector{<:Real};
        basis,
        n_iter::Integer = 3,
        α::Real           = 0.3,
        β_iodine::Real    = 1.0,
        β_water::Real     = 1.0,
        step_lim_iodine::Real = 0.75,
        step_lim_water::Real  = 5.0,
        relax::Real       = 0.5,
        verbose::Bool     = true,
    )
    n_bins = length(bins)
    length(I0) == n_bins ||
        error("apply_rwls!: length(I0) = $(length(I0)) must equal length(bins) = $n_bins.")
    length(basis.ŵ_bins) == n_bins ||
        error("apply_rwls!: length(basis.ŵ_bins) = $(length(basis.ŵ_bins)) must equal length(bins) = $n_bins.")

    eltype(sino_iodine) === Float32 && eltype(sino_water) === Float32 ||
        error("apply_rwls!: sino_iodine / sino_water must be Float32.")
    for (b, bin_arr) in enumerate(bins)
        eltype(bin_arr) === Float32 ||
            error("apply_rwls!: bins[$b] has eltype $(eltype(bin_arr)); Float32 required.")
        size(bin_arr) == size(sino_iodine) ||
            error("apply_rwls!: bins[$b] shape $(size(bin_arr)) ≠ sino shape $(size(sino_iodine)).")
    end
    eltype(basis.p) === Float32 && eltype(basis.q) === Float32 ||
        error("apply_rwls!: basis.p and basis.q must be Float32 vectors.")
    for (b, ŵb) in enumerate(basis.ŵ_bins)
        eltype(ŵb) === Float32 ||
            error("apply_rwls!: basis.ŵ_bins[$b] has eltype $(eltype(ŵb)); Float32 required.")
        length(ŵb) == length(basis.p) ||
            error("apply_rwls!: basis.ŵ_bins[$b] length = $(length(ŵb)) must match length(basis.p) = $(length(basis.p)).")
    end

    n_E = length(basis.p)

    # Per-bin normalized spectrum on CPU.
    ŵ_bins_norm = Vector{Vector{Float32}}(undef, n_bins)
    for b in 1:n_bins
        ŵb64 = Float64.(basis.ŵ_bins[b])
        s = sum(ŵb64)
        s > 0 || error("apply_rwls!: basis.ŵ_bins[$b] sums to 0; cannot normalize.")
        ŵ_bins_norm[b] = Float32.(ŵb64 ./ s)
    end

    p_vec = Float32.(Array(basis.p))
    q_vec = Float32.(Array(basis.q))
    I0_f32 = Float32.(I0)

    # FFT proximal reg eigenvalues.
    n_col, n_row, n_view = size(sino_iodine)
    freq2 = [Float64(2 - 2cos(2π * (i - 1) / n_col) + 2 - 2cos(2π * (j - 1) / n_row))
             for i in 1:n_col, j in 1:n_row]
    α_bw = 2.0 * Float64(α) * Float64(β_water)
    α_bI = 2.0 * Float64(α) * Float64(β_iodine)

    step_I  = Float32(step_lim_iodine)
    step_W  = Float32(step_lim_water)
    relax_f = Float32(relax)

    # Scratch buffers on the sinogram backend.
    a_I = sino_iodine
    a_W = sino_water
    et  = similar(a_I)
    F   = [similar(a_I) for _ in 1:n_bins]
    J_I = [similar(a_I) for _ in 1:n_bins]
    J_W = [similar(a_I) for _ in 1:n_bins]
    r   = [similar(a_I) for _ in 1:n_bins]
    wt  = [similar(a_I) for _ in 1:n_bins]
    H11 = similar(a_I);  H12 = similar(a_I);  H22 = similar(a_I)
    g1  = similar(a_I);  g2  = similar(a_I)
    det_H = similar(a_I)
    δ_Ig  = similar(a_I)
    δ_Wg  = similar(a_I)

    t0 = time()
    for iter in 1:Int(n_iter)
        # ── Forward model + Jacobian (energy-outer, pixel-broadcast) ──
        for b in 1:n_bins
            fill!(F[b],   0f0)
            fill!(J_I[b], 0f0)
            fill!(J_W[b], 0f0)
        end
        for k in 1:n_E
            tI = p_vec[k]
            tw = q_vec[k]
            @. et = exp(-a_I * tI - a_W * tw)
            for b in 1:n_bins
                wb = ŵ_bins_norm[b][k]
                @. F[b]   += wb * et
                @. J_I[b] -= wb * tI * et
                @. J_W[b] -= wb * tw * et
            end
        end
        # Apply per-bin I0 so F is in count-domain (counts = I0 · Σ ŵ exp).
        for b in 1:n_bins
            I0b = I0_f32[b]
            @. F[b]   *= I0b
            @. J_I[b] *= I0b
            @. J_W[b] *= I0b
        end

        # ── Poisson-weighted normal equations (count domain) ──
        for b in 1:n_bins
            @. r[b]  = bins[b] - F[b]
            @. wt[b] = 1f0 / max(F[b], 1f0)
        end
        fill!(H11, 0f0);  fill!(H12, 0f0);  fill!(H22, 0f0)
        fill!(g1,  0f0);  fill!(g2,  0f0)
        for b in 1:n_bins
            @. H11 += J_W[b] * J_W[b] * wt[b]
            @. H12 += J_W[b] * J_I[b] * wt[b]
            @. H22 += J_I[b] * J_I[b] * wt[b]
            @. g1  += J_W[b] * wt[b] * r[b]
            @. g2  += J_I[b] * wt[b] * r[b]
        end

        # ── 2×2 GN step (clamped, non-negativity projected) ──
        @. det_H = max(abs(H11 * H22 - H12 * H12), 1f-30)
        @. δ_Wg  = clamp((H22 * g1 - H12 * g2) / det_H, -step_W, step_W)
        @. δ_Ig  = clamp((H11 * g2 - H12 * g1) / det_H, -step_I, step_I)
        @. a_W = max(a_W + relax_f * δ_Wg, 0f0)
        @. a_I = max(a_I + relax_f * δ_Ig, 0f0)

        # ── FFT proximal prior (per slice, CPU) ──
        aw_cpu = Array(a_W)
        aI_cpu = Array(a_I)
        denom_W = 1.0 .+ α_bw .* freq2
        denom_I = 1.0 .+ α_bI .* freq2
        for k in 1:n_view
            sw = Float64.(aw_cpu[:, :, k])
            sI = Float64.(aI_cpu[:, :, k])
            aw_cpu[:, :, k] .= Float32.(real.(FFTW.ifft(FFTW.fft(sw) ./ denom_W)))
            aI_cpu[:, :, k] .= Float32.(real.(FFTW.ifft(FFTW.fft(sI) ./ denom_I)))
        end
        copyto!(a_W, aw_cpu)
        copyto!(a_I, aI_cpu)

        verbose && @info "[apply_rwls!] iter $iter done"
    end
    dt = time() - t0

    if verbose
        @info "[apply_rwls! (n_bins=$n_bins)] $(n_iter) iters in $(round(dt, digits = 1)) s"
        @info "  α=$(α), β_I=$(β_iodine), β_W=$(β_water), relax=$(relax), step_lim=(I=$(step_lim_iodine), W=$(step_lim_water))"
    end

    (n_iter = Int(n_iter),
     α = Float64(α), β_iodine = Float64(β_iodine), β_water = Float64(β_water),
     n_bins = n_bins)
end

"""
    apply_rwls(bins, I0, sino_iodine_init, sino_water_init; kwargs...)
        -> (sino_iodine, sino_water, info)

Allocating wrapper around `apply_rwls!`.  Deep-copies the warm start on the
same backend as `bins[1]` and forwards all kwargs.
"""
function apply_rwls(
        bins::AbstractVector,
        I0::AbstractVector{<:Real},
        sino_iodine_init::AbstractArray,
        sino_water_init::AbstractArray;
        kwargs...,
    )
    sino_iodine = similar(bins[1], Float32, size(bins[1]))
    sino_water  = similar(bins[1], Float32, size(bins[1]))
    copyto!(sino_iodine, sino_iodine_init)
    copyto!(sino_water,  sino_water_init)
    info = apply_rwls!(sino_iodine, sino_water, bins, I0; kwargs...)
    (sino_iodine, sino_water, info)
end

export apply_rwls!, apply_rwls
