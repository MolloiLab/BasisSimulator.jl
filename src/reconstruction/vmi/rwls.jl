"""
RWLS-GN (Reweighted Least Squares — Gauss-Newton) for material basis
decomposition.

Ducros 2017 style count-domain Gauss-Newton inversion of the polychromatic
forward model, with a CG-on-spatial-Laplacian quadratic proximal prior.
Warm-started from a cheap decomp (Cong, CMV, …).

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

followed by a per-slice CG quadratic proximal prior solving the linear
system  `(I + αβ·L) ŝ = s`  via vendor-agnostic spatial-Laplacian CG
(5-point stencil, periodic BC).  Replaces the legacy FFT prior with no
padding / power-of-2 / ndim restrictions.

**Performance:** for `n_bins == 3` (typical PCCT bin grouping), a fused
forward-model + Hessian + gradient kernel collapses the n_E×9 inner-loop
broadcasts into ONE AK kernel launch per outer iteration — ~8× faster
on Metal vs the energy-outer broadcast loop.  Other `n_bins` fall back to
the broadcast loop (still GPU-resident, just slower).

Reference:
  Ducros, Bussod, Sixou, Trouvé, Peyrin, Nuyts (2017)
  "Regularization of nonlinear decomposition of spectral X-ray projection
  images."  *Med. Phys.* 44(9):e174–e187.
"""

# ────────────────────────────────────────────────────────────────────────────
# CG-on-spatial-Laplacian proximal prior (vendor-agnostic, GPU-resident).
#
# Solves `(I + αβ·L) x = b` where L is the negative 2D Laplacian with
# periodic BC on dims 1, 2 (dim 3 = view = independent batch).
#
#   • One coalesced 5-point stencil kernel + a few broadcasts/reductions.
#   • No FFT, no bit-reversal, no transposes, no padding.
#   • (I + αβ·L) is SPD with κ ≈ 1 + 8·α·β → CG converges in 12–20 iters.
#   • Runs natively on CPU / Metal / CUDA / ROCm via AK.foreachindex.
# ────────────────────────────────────────────────────────────────────────────

@inline function _apply_I_plus_αβL!(
        y::AbstractArray{Float32, 3},
        x::AbstractArray{Float32, 3},
        αβ::Float32,
    )
    N1, N2, _ = size(x)
    let N1_ = N1, N2_ = N2, αβ_ = αβ
        AK.foreachindex(x) do idx
            lin = idx - 1
            k   = lin ÷ (N1_ * N2_)
            rem = lin - k * (N1_ * N2_)
            j   = rem ÷ N1_
            i   = rem - j * N1_
            i_plus  = i + 1 == N1_ ? 0       : i + 1
            i_minus = i == 0       ? N1_ - 1 : i - 1
            j_plus  = j + 1 == N2_ ? 0       : j + 1
            j_minus = j == 0       ? N2_ - 1 : j - 1
            center = x[i + 1, j + 1, k + 1]
            lap = 4f0 * center -
                  x[i_plus  + 1, j + 1, k + 1] -
                  x[i_minus + 1, j + 1, k + 1] -
                  x[i + 1, j_plus  + 1, k + 1] -
                  x[i + 1, j_minus + 1, k + 1]
            y[idx] = center + αβ_ * lap
        end
    end
    return
end

_rwls_cg_prior_alloc(a_W::AbstractArray{Float32, 3}) =
    (b  = similar(a_W),
     r  = similar(a_W),
     p  = similar(a_W),
     Ap = similar(a_W))

function _rwls_cg_solve!(
        x::AbstractArray{Float32, 3},
        b::AbstractArray{Float32, 3},
        αβ::Real, buf;
        max_iter::Int = 20, rel_tol::Real = 1f-4,
    )
    αβ_f = Float32(αβ)
    r  = buf.r;  p = buf.p;  Ap = buf.Ap

    # x₀ = b is a cheap, close initial guess (true x ≈ b for small αβ).
    copyto!(x, b)
    _apply_I_plus_αβL!(Ap, x, αβ_f)
    @. r = b - Ap
    copyto!(p, r)

    rr        = Float32(sum(r .* r))
    b_norm_sq = max(Float32(sum(b .* b)), 1f-30)
    tol_sq    = (Float32(rel_tol)^2) * b_norm_sq

    k_final = max_iter
    for k in 1:max_iter
        _apply_I_plus_αβL!(Ap, p, αβ_f)
        pAp = max(Float32(sum(p .* Ap)), 1f-30)
        α_k = rr / pAp
        @. x += α_k * p
        @. r -= α_k * Ap
        rr_new = Float32(sum(r .* r))
        if rr_new < tol_sq
            k_final = k;  rr = rr_new
            break
        end
        β_k = rr_new / rr
        @. p = r + β_k * p
        rr = rr_new
    end
    (k_iter = k_final, final_rel_res = sqrt(max(rr, 0f0) / b_norm_sq))
end

function _rwls_cg_prior!(
        a_W::AbstractArray{Float32, 3},
        a_I::AbstractArray{Float32, 3},
        α_bw::Real, α_bI::Real, buf;
        max_iter::Int = 20, rel_tol::Real = 1f-4,
        verbose::Bool = false,
    )
    copyto!(buf.b, a_W)
    info_W = _rwls_cg_solve!(a_W, buf.b, α_bw, buf;
                             max_iter = max_iter, rel_tol = rel_tol)
    copyto!(buf.b, a_I)
    info_I = _rwls_cg_solve!(a_I, buf.b, α_bI, buf;
                             max_iter = max_iter, rel_tol = rel_tol)
    if verbose
        @info "[RWLS CG prior] W: $(info_W.k_iter) iters (rel_res $(round(info_W.final_rel_res, sigdigits=3)))   I: $(info_I.k_iter) iters (rel_res $(round(info_I.final_rel_res, sigdigits=3)))"
    end
    (water = info_W, iodine = info_I)
end

# ────────────────────────────────────────────────────────────────────────────
# Fused forward + Hessian + gradient kernel, n_bins == 3 specialization.
#
# Hot path for PCCT 3-bin grouping (e.g. [[1,2], [3], [4]]).  Replaces the
# n_E × 9 broadcast launches in the energy-outer-loop fallback with ONE AK
# kernel.  Each thread:
#   1. Reads a_I, a_W once.
#   2. Loops over energies in registers (ŵ / p / q tables fit in L1).
#   3. Accumulates F, J_I, J_W per bin in registers.
#   4. Scales by I0, computes Poisson residual + weight.
#   5. Writes H11 / H12 / H22 / g1 / g2 once.
# ────────────────────────────────────────────────────────────────────────────

function _rwls_fused_step_n3!(
        H11::AbstractArray{Float32, 3}, H12::AbstractArray{Float32, 3}, H22::AbstractArray{Float32, 3},
        g1::AbstractArray{Float32, 3},  g2::AbstractArray{Float32, 3},
        a_I::AbstractArray{Float32, 3}, a_W::AbstractArray{Float32, 3},
        bins_1::AbstractArray{Float32, 3}, bins_2::AbstractArray{Float32, 3},
        bins_3::AbstractArray{Float32, 3},
        p_vec::AbstractVector{Float32},
        q_vec::AbstractVector{Float32},
        ŵ_1::AbstractVector{Float32}, ŵ_2::AbstractVector{Float32},
        ŵ_3::AbstractVector{Float32},
        I0_1::Float32, I0_2::Float32, I0_3::Float32,
    )
    nE = length(p_vec)
    let nE = nE, I0_1_ = I0_1, I0_2_ = I0_2, I0_3_ = I0_3
        AK.foreachindex(a_I) do idx
            a_I_v = a_I[idx]
            a_W_v = a_W[idx]
            F_1  = 0f0; F_2  = 0f0; F_3  = 0f0
            JI_1 = 0f0; JI_2 = 0f0; JI_3 = 0f0
            JW_1 = 0f0; JW_2 = 0f0; JW_3 = 0f0
            @inbounds for k in 1:nE
                tI = p_vec[k]
                tw = q_vec[k]
                et = exp(-a_I_v * tI - a_W_v * tw)
                w1 = ŵ_1[k];  w2 = ŵ_2[k];  w3 = ŵ_3[k]
                F_1  += w1 * et;         F_2  += w2 * et;         F_3  += w3 * et
                JI_1 -= w1 * tI * et;    JI_2 -= w2 * tI * et;    JI_3 -= w3 * tI * et
                JW_1 -= w1 * tw * et;    JW_2 -= w2 * tw * et;    JW_3 -= w3 * tw * et
            end
            F_1  *= I0_1_;    F_2  *= I0_2_;    F_3  *= I0_3_
            JI_1 *= I0_1_;    JI_2 *= I0_2_;    JI_3 *= I0_3_
            JW_1 *= I0_1_;    JW_2 *= I0_2_;    JW_3 *= I0_3_
            r_1 = bins_1[idx] - F_1
            r_2 = bins_2[idx] - F_2
            r_3 = bins_3[idx] - F_3
            wt_1 = 1f0 / max(F_1, 1f0)
            wt_2 = 1f0 / max(F_2, 1f0)
            wt_3 = 1f0 / max(F_3, 1f0)
            H11[idx] = JW_1*JW_1*wt_1 + JW_2*JW_2*wt_2 + JW_3*JW_3*wt_3
            H12[idx] = JW_1*JI_1*wt_1 + JW_2*JI_2*wt_2 + JW_3*JI_3*wt_3
            H22[idx] = JI_1*JI_1*wt_1 + JI_2*JI_2*wt_2 + JI_3*JI_3*wt_3
            g1[idx]  = JW_1*wt_1*r_1  + JW_2*wt_2*r_2  + JW_3*wt_3*r_3
            g2[idx]  = JI_1*wt_1*r_1  + JI_2*wt_2*r_2  + JI_3*wt_3*r_3
        end
    end
    return
end

# Generic N-bin fallback — energy-outer broadcast loop.  Slower than the
# n_bins == 3 fused kernel (2,740-launch hot path on Metal) but keeps the
# library functional for arbitrary bin counts.
function _rwls_step_generic!(
        H11, H12, H22, g1, g2,
        a_I, a_W, bins, ŵ_bins_norm, p_vec, q_vec, I0_f32,
        scratch,
    )
    n_bins = length(bins)
    n_E    = length(p_vec)
    et = scratch.et
    F   = scratch.F
    J_I = scratch.J_I
    J_W = scratch.J_W
    r   = scratch.r
    wt  = scratch.wt

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
    for b in 1:n_bins
        I0b = I0_f32[b]
        @. F[b]   *= I0b
        @. J_I[b] *= I0b
        @. J_W[b] *= I0b
    end
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
    return
end

"""
    apply_rwls!(sino_iodine, sino_water, bins, I0;
                basis,
                n_iter = 3,
                α = 0.3, β_iodine = 1.0, β_water = 1.0,
                step_lim_iodine = 0.75, step_lim_water = 5.0,
                relax = 0.5,
                cg_max_iter = 20, cg_rel_tol = 1f-4,
                verbose = true) -> NamedTuple

In-place N-bin count-domain RWLS-GN.  Mutates `sino_iodine` and `sino_water`
(warm-started by the caller — typically from a Cong / CMV polynomial init).

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
- `α`                 : CG proximal weight (0 disables smoothing).
- `β_iodine`, `β_water` : per-basis smoothing scales.
- `step_lim_iodine`, `step_lim_water` : per-iter |Δ| clamp (g/cm²).
- `relax`             : GN relaxation factor (default 0.5).
- `cg_max_iter`       : inner CG iterations for the proximal solve.
- `cg_rel_tol`        : CG relative-residual stopping tolerance.

Returns `(n_iter, α, β_iodine, β_water, n_bins)`.
"""
function apply_rwls!(
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        bins::AbstractVector,
        I0::AbstractVector{<:Real};
        basis,
        n_iter::Integer       = 3,
        α::Real               = 0.3,
        β_iodine::Real        = 1.0,
        β_water::Real         = 1.0,
        step_lim_iodine::Real = 0.75,
        step_lim_water::Real  = 5.0,
        relax::Real           = 0.5,
        cg_max_iter::Int      = 20,
        cg_rel_tol::Real      = 1f-4,
        verbose::Bool         = true,
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

    # Per-bin normalized spectrum (Σ_k ŵ = 1).
    ŵ_bins_norm = Vector{Vector{Float32}}(undef, n_bins)
    for b in 1:n_bins
        ŵb64 = Float64.(basis.ŵ_bins[b])
        s = sum(ŵb64)
        s > 0 || error("apply_rwls!: basis.ŵ_bins[$b] sums to 0; cannot normalize.")
        ŵ_bins_norm[b] = Float32.(ŵb64 ./ s)
    end

    p_vec  = Float32.(Array(basis.p))
    q_vec  = Float32.(Array(basis.q))
    I0_f32 = Float32.(I0)

    α_bw = 2.0 * Float64(α) * Float64(β_water)
    α_bI = 2.0 * Float64(α) * Float64(β_iodine)

    step_I  = Float32(step_lim_iodine)
    step_W  = Float32(step_lim_water)
    relax_f = Float32(relax)

    a_I = sino_iodine
    a_W = sino_water

    # 2×2 GN scratch on the sinogram backend.
    H11 = similar(a_I);  H12 = similar(a_I);  H22 = similar(a_I)
    g1  = similar(a_I);  g2  = similar(a_I)
    det_H = similar(a_I)
    δ_Ig  = similar(a_I)
    δ_Wg  = similar(a_I)

    # CG proximal-prior buffers.
    cg_prior_buf = _rwls_cg_prior_alloc(a_W)

    use_fused_n3 = (n_bins == 3)

    if use_fused_n3
        # Stage spectrum tables onto GPU once.
        p_vec_gpu = similar(a_I, Float32, n_E);  copyto!(p_vec_gpu, p_vec)
        q_vec_gpu = similar(a_I, Float32, n_E);  copyto!(q_vec_gpu, q_vec)
        ŵ_1_gpu   = similar(a_I, Float32, n_E);  copyto!(ŵ_1_gpu, ŵ_bins_norm[1])
        ŵ_2_gpu   = similar(a_I, Float32, n_E);  copyto!(ŵ_2_gpu, ŵ_bins_norm[2])
        ŵ_3_gpu   = similar(a_I, Float32, n_E);  copyto!(ŵ_3_gpu, ŵ_bins_norm[3])
        I0_1 = I0_f32[1];  I0_2 = I0_f32[2];  I0_3 = I0_f32[3]
        bins_1 = bins[1];  bins_2 = bins[2];  bins_3 = bins[3]
    else
        # Generic-fallback scratch — N copies of the sinogram-shape buffers.
        scratch = (et  = similar(a_I),
                   F   = [similar(a_I) for _ in 1:n_bins],
                   J_I = [similar(a_I) for _ in 1:n_bins],
                   J_W = [similar(a_I) for _ in 1:n_bins],
                   r   = [similar(a_I) for _ in 1:n_bins],
                   wt  = [similar(a_I) for _ in 1:n_bins])
    end

    t0 = time()
    for iter in 1:Int(n_iter)
        # ── Forward + Hessian + gradient ──
        if use_fused_n3
            _rwls_fused_step_n3!(
                H11, H12, H22, g1, g2,
                a_I, a_W,
                bins_1, bins_2, bins_3,
                p_vec_gpu, q_vec_gpu,
                ŵ_1_gpu, ŵ_2_gpu, ŵ_3_gpu,
                I0_1, I0_2, I0_3,
            )
        else
            _rwls_step_generic!(
                H11, H12, H22, g1, g2,
                a_I, a_W, bins, ŵ_bins_norm, p_vec, q_vec, I0_f32,
                scratch,
            )
        end

        # ── 2×2 GN step (clamped + non-negativity projected) ──
        @. det_H = max(abs(H11 * H22 - H12 * H12), 1f-30)
        @. δ_Wg  = clamp((H22 * g1 - H12 * g2) / det_H, -step_W, step_W)
        @. δ_Ig  = clamp((H11 * g2 - H12 * g1) / det_H, -step_I, step_I)
        @. a_W = max(a_W + relax_f * δ_Wg, 0f0)
        @. a_I = max(a_I + relax_f * δ_Ig, 0f0)

        # ── CG proximal prior on the periodic 2D Laplacian ──
        if α_bw > 0.0 || α_bI > 0.0
            _rwls_cg_prior!(a_W, a_I, α_bw, α_bI, cg_prior_buf;
                            max_iter = cg_max_iter, rel_tol = cg_rel_tol,
                            verbose  = verbose && iter == 1)
        end

        verbose && @info "[apply_rwls!] iter $iter done"
    end
    dt = time() - t0

    if verbose
        path_label = use_fused_n3 ? "fused n_bins=3" : "generic n_bins=$n_bins"
        @info "[apply_rwls! ($path_label)] $(n_iter) iters in $(round(dt, digits = 1)) s"
        @info "  α=$(α), β_I=$(β_iodine), β_W=$(β_water), relax=$(relax), step_lim=(I=$(step_lim_iodine), W=$(step_lim_water))"
        @info "  CG prior: max_iter=$(cg_max_iter), rel_tol=$(cg_rel_tol)"
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
