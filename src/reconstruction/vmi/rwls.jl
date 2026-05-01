"""
RWLS-GN (Reweighted Least Squares — Gauss-Newton) for material basis
decomposition — PCCT 3-bin specialization.

Ducros 2017 style count-domain Gauss-Newton inversion of the polychromatic
forward model, with a CG-on-spatial-Laplacian quadratic proximal prior.
Warm-started from a cheap decomp (Cong, …).

**Hard-coded for `n_bins = 3`** — matches the PCCT bin-grouping
`[[1,2], [3], [4]]` used throughout this codebase.  Classical DE-CT
(2-bin, low/high kVp) goes through PWLS, not RWLS.

Per-pixel forward model:

    F_m(s_I, s_W) = I0_m · Σ_k ŵ_m[k] · exp(−p[k]·s_I − q[k]·s_W)

Per-pixel 2×2 Gauss-Newton step on the 3-equation / 2-unknown system
with Poisson inverse-variance weights in count space:

    r_m   = s_m − F_m
    w_m   = 1 / max(F_m, 1)                          (one-count floor)
    H     = Σ_m w_m · J_mᵀ J_m      (2×2)
    g     = Σ_m w_m · J_mᵀ r_m      (2-vector)
    Δs    = clamp.(H⁻¹ · g, ±step_lim),  applied with relaxation

followed by a per-slice CG quadratic proximal prior solving
`(I + αβ·L) ŝ = s` via vendor-agnostic spatial-Laplacian CG (5-point
stencil, periodic BC).

# Workspace + auto-tiling

Workspace is **required**.  The factory `create_rwls_workspace` probes
available memory and chooses a tile size such that the workspace's nine
sino-shape parents (3 H + 2 g + 4 CG) fit.  `apply_rwls!` then iterates
the GN + CG-prior pipeline tile-by-tile along dim 3 (view axis), so a
volume that exceeds device memory still completes — only the tile size
shrinks transparently.

Reference:
  Ducros, Bussod, Sixou, Trouvé, Peyrin, Nuyts (2017)
  *"Regularization of nonlinear decomposition of spectral X-ray projection
  images."*  *Med. Phys.* 44(9):e174–e187.
"""

# ════════════════════════════════════════════════════════════════════════
#  Workspace
# ════════════════════════════════════════════════════════════════════════

"""
    RwlsWorkspace{T, A3, A1}

Pre-allocated scratch + spectrum tables for `apply_rwls!`.  Each
sino-shape parent is sized `(n_col, n_row, tile_size)`; per call the
inner kernels operate on contiguous views over `1:k` of dim 3 (where
`k ≤ tile_size` is the actual tile length).

Layout:
  • 3 GN Hessian unique components: `H11`, `H12`, `H22`
  • 2 GN gradient components: `g1`, `g2`
  • 4 CG-prior buffers: `cg_b`, `cg_r`, `cg_p`, `cg_Ap`
  • 2 shared spectrum tables: `p_vec` (iodine μρ), `q_vec` (water μρ)
  • 3 per-bin spectrum weights: `ŵ_1`, `ŵ_2`, `ŵ_3` (already normalized)

Total **9 sino-shape parents** at `tile_size` views — down from the
12-parent layout we had pre-workspace (det_H, δ_Wg, δ_Ig now computed
inline, no storage).

Construct with [`create_rwls_workspace`](@ref); thread-unsafe (one
workspace per task).
"""
mutable struct RwlsWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}, A1 <: AbstractArray{T, 1}}
    H11::A3;   H12::A3;   H22::A3
    g1::A3;    g2::A3
    cg_b::A3;  cg_r::A3;  cg_p::A3;  cg_Ap::A3
    p_vec::A1; q_vec::A1
    ŵ_1::A1;   ŵ_2::A1;   ŵ_3::A1
    tile_size::Int
end

"""
    create_rwls_workspace(sino_template; n_E, mem_budget_GB = nothing)
        -> RwlsWorkspace

Allocate the RWLS workspace on the same backend as `sino_template` (its
type and element type are the witnesses; we never name `MtlArray` /
`CuArray` directly).

# Arguments
- `sino_template`  : an `AbstractArray{<:AbstractFloat, 3}` whose shape
  and backend the workspace should match (the actual sinogram you'll
  pass to `apply_rwls!`).

# Keyword arguments
- `n_E`            : number of energies in the spectrum (length of
  `basis.p` / `basis.q` / each `basis.ŵ_bins[i]`).
- `mem_budget_GB`  : explicit memory-budget override.  `nothing` =
  auto-probe via `Sys.free_memory() · 0.6`.  **Pass explicitly on
  discrete-GPU systems** (CUDA / ROCm / Intel Arc) where
  `Sys.free_memory` doesn't reflect VRAM.

# Failure
On `OutOfMemoryError` / `OutOfGPUMemoryError`, the constructor halves
`tile_size` and retries up to 4 times.  If even `tile_size = 1` fails,
raises a clear error with the byte-count gap.
"""
function create_rwls_workspace(
        sino_template::AbstractArray{<:AbstractFloat, 3};
        n_E::Integer,
        mem_budget_GB::Union{Nothing, Real} = nothing,
    )
    T = eltype(sino_template)
    n_col, n_row, n_view = size(sino_template)

    # 9 sino-shape parents × tile_size views = per_view × tile_size bytes.
    per_view_bytes = 9 * n_col * n_row * sizeof(T)
    initial_tile = suggest_tile_size(per_view_bytes, n_view; mem_budget_GB)

    with_oom_retry("create_rwls_workspace", initial_tile) do tile_size
        sh = (n_col, n_row, Int(tile_size))
        H11   = similar(sino_template, T, sh);  fill!(H11, zero(T))
        H12   = similar(sino_template, T, sh);  fill!(H12, zero(T))
        H22   = similar(sino_template, T, sh);  fill!(H22, zero(T))
        g1    = similar(sino_template, T, sh);  fill!(g1,  zero(T))
        g2    = similar(sino_template, T, sh);  fill!(g2,  zero(T))
        cg_b  = similar(sino_template, T, sh);  fill!(cg_b,  zero(T))
        cg_r  = similar(sino_template, T, sh);  fill!(cg_r,  zero(T))
        cg_p  = similar(sino_template, T, sh);  fill!(cg_p,  zero(T))
        cg_Ap = similar(sino_template, T, sh);  fill!(cg_Ap, zero(T))
        p_vec = similar(sino_template, T, Int(n_E));  fill!(p_vec, zero(T))
        q_vec = similar(sino_template, T, Int(n_E));  fill!(q_vec, zero(T))
        ŵ_1   = similar(sino_template, T, Int(n_E));  fill!(ŵ_1, zero(T))
        ŵ_2   = similar(sino_template, T, Int(n_E));  fill!(ŵ_2, zero(T))
        ŵ_3   = similar(sino_template, T, Int(n_E));  fill!(ŵ_3, zero(T))
        RwlsWorkspace{T, typeof(H11), typeof(p_vec)}(
            H11, H12, H22, g1, g2,
            cg_b, cg_r, cg_p, cg_Ap,
            p_vec, q_vec, ŵ_1, ŵ_2, ŵ_3,
            Int(tile_size),
        )
    end
end

# ════════════════════════════════════════════════════════════════════════
#  CG-on-spatial-Laplacian proximal prior (vendor-agnostic, GPU-resident)
# ════════════════════════════════════════════════════════════════════════
# Solves `(I + αβ·L) x = b` where L is the negative 2D Laplacian with
# periodic BC on dims 1, 2 (dim 3 = view = independent batch).
#
#   • One coalesced 5-point stencil kernel + a few broadcasts/reductions.
#   • No FFT, no bit-reversal, no transposes, no padding.
#   • (I + αβ·L) is SPD with κ ≈ 1 + 8·α·β → CG converges in 12–20 iters.
#   • Runs natively on CPU / Metal / CUDA / ROCm via AK.foreachindex.

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

# CG solve `(I + αβ·L) x = b`, x₀ = b initial guess (close for small αβ).
# `r`, `p`, `Ap` are buffers passed in (workspace.cg_r/cg_p/cg_Ap).
function _rwls_cg_solve!(
        x::AbstractArray{Float32, 3},
        b::AbstractArray{Float32, 3},
        αβ::Real,
        r::AbstractArray{Float32, 3},
        p::AbstractArray{Float32, 3},
        Ap::AbstractArray{Float32, 3};
        max_iter::Int = 20, rel_tol::Real = 1f-4,
    )
    αβ_f = Float32(αβ)
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

# Sequential water + iodine CG solves, reusing buffers.
function _rwls_cg_prior!(
        a_W::AbstractArray{Float32, 3},
        a_I::AbstractArray{Float32, 3},
        α_bw::Real, α_bI::Real,
        b::AbstractArray{Float32, 3},
        r::AbstractArray{Float32, 3},
        p::AbstractArray{Float32, 3},
        Ap::AbstractArray{Float32, 3};
        max_iter::Int = 20, rel_tol::Real = 1f-4,
        verbose::Bool = false,
    )
    copyto!(b, a_W)
    info_W = _rwls_cg_solve!(a_W, b, α_bw, r, p, Ap; max_iter, rel_tol)
    copyto!(b, a_I)
    info_I = _rwls_cg_solve!(a_I, b, α_bI, r, p, Ap; max_iter, rel_tol)
    if verbose
        @info "[RWLS CG prior] W: $(info_W.k_iter) iters (rel_res $(round(info_W.final_rel_res, sigdigits=3)))   I: $(info_I.k_iter) iters (rel_res $(round(info_I.final_rel_res, sigdigits=3)))"
    end
    (water = info_W, iodine = info_I)
end

# ════════════════════════════════════════════════════════════════════════
#  Fused forward + Hessian + gradient kernel (n_bins == 3 hardcoded)
# ════════════════════════════════════════════════════════════════════════
# Replaces n_E × 9 broadcast launches with ONE AK kernel.  Each thread:
#   1. Reads a_I, a_W once.
#   2. Loops energies in registers (ŵ / p / q tables fit in L1).
#   3. Accumulates F, J_I, J_W per bin in registers.
#   4. Scales by I0, computes Poisson residual + weight.
#   5. Writes H11 / H12 / H22 / g1 / g2 once.

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

# ════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════

"""
    apply_rwls!(ws, sino_iodine, sino_water, bins, I0;
                basis,
                n_iter           = 3,
                α                = 0.3,
                β_iodine         = 1.0,
                β_water          = 1.0,
                step_lim_iodine  = 0.75,
                step_lim_water   = 5.0,
                relax            = 0.5,
                cg_max_iter      = 20,
                cg_rel_tol       = 1f-4,
                verbose          = true) -> NamedTuple

In-place 3-bin count-domain RWLS-GN.  Mutates `sino_iodine` and
`sino_water` (warm-started by the caller — typically from Cong).

# Arguments
- `ws`               : `RwlsWorkspace` from `create_rwls_workspace`.
  REQUIRED — no internal allocation path.
- `sino_iodine`,
  `sino_water`       : Float32 material sinograms (g/cm²), shape
  `[n_col, n_row, n_view]`.  Mutated in place.
- `bins`             : 3-tuple or 3-element vector of Float32 count
  arrays (PCCT bin groups), each shaped like `sino_iodine`.  Counts
  `s_m = Σ_{b ∈ group_m} I0_b · exp(-h_b)` (scatter-correct beforehand).
- `I0`               : 3 per-bin I0 scalars.

# Keyword arguments
- `basis`            : NamedTuple with
    - `ŵ_bins::Vector{<:AbstractVector{Float32}}` of length 3 — spectral
      weights per bin group, length `n_E`.  Internally renormalized to
      sum to 1.
    - `p::AbstractVector{Float32}` — iodine μρ at each spectrum energy.
    - `q::AbstractVector{Float32}` — water μρ at each spectrum energy.
- `n_iter`            : outer GN iterations (default 3).
- `α`                 : CG proximal weight (0 ⇒ skip prior).
- `β_iodine`,
  `β_water`           : per-basis smoothing scales.
- `step_lim_iodine`,
  `step_lim_water`    : per-iter |Δ| clamp (g/cm²).
- `relax`             : GN relaxation factor.
- `cg_max_iter`       : inner CG iterations for the proximal solve.
- `cg_rel_tol`        : CG relative-residual stopping tolerance.

Returns `(n_iter, α, β_iodine, β_water, n_bins, tile_size)`.
"""
function apply_rwls!(
        ws::RwlsWorkspace,
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        bins,
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
    # ── Validate hard contract: 3 bins, Float32 throughout, shapes match.
    length(bins) == 3 ||
        error("apply_rwls!: requires exactly n_bins = 3 (got $(length(bins))).  " *
              "Classical DE-CT (2 bins) goes through PWLS, not RWLS.")
    length(I0) == 3 ||
        error("apply_rwls!: length(I0) = $(length(I0)) must equal 3.")
    length(basis.ŵ_bins) == 3 ||
        error("apply_rwls!: length(basis.ŵ_bins) = $(length(basis.ŵ_bins)) must equal 3.")
    eltype(sino_iodine) === Float32 && eltype(sino_water) === Float32 ||
        error("apply_rwls!: sino_iodine / sino_water must be Float32.")
    for b in 1:3
        eltype(bins[b]) === Float32 ||
            error("apply_rwls!: bins[$b] has eltype $(eltype(bins[b])); Float32 required.")
        size(bins[b]) == size(sino_iodine) ||
            error("apply_rwls!: bins[$b] shape $(size(bins[b])) ≠ sino shape $(size(sino_iodine)).")
    end
    eltype(basis.p) === Float32 && eltype(basis.q) === Float32 ||
        error("apply_rwls!: basis.p / basis.q must be Float32 vectors.")
    for b in 1:3
        eltype(basis.ŵ_bins[b]) === Float32 ||
            error("apply_rwls!: basis.ŵ_bins[$b] has eltype $(eltype(basis.ŵ_bins[b])); Float32 required.")
        length(basis.ŵ_bins[b]) == length(basis.p) ||
            error("apply_rwls!: basis.ŵ_bins[$b] length = $(length(basis.ŵ_bins[b])) must match length(basis.p) = $(length(basis.p)).")
    end
    n_E = length(basis.p)
    length(ws.p_vec) == n_E ||
        error("apply_rwls!: workspace.p_vec length $(length(ws.p_vec)) ≠ basis n_E = $n_E.  " *
              "Recreate workspace with `create_rwls_workspace(...; n_E = $n_E)`.")
    sino_shape = size(sino_iodine)
    sino_shape[1:2] == size(ws.H11)[1:2] ||
        error("apply_rwls!: sino (col,row) = $(sino_shape[1:2]) ≠ workspace (col,row) = $(size(ws.H11)[1:2]).")
    size(ws.H11, 3) == ws.tile_size ||
        error("apply_rwls!: workspace.tile_size mismatch with H11's dim-3 — corrupted workspace.")

    # ── Stage spectrum tables onto workspace once (CPU → device copy).
    # Renormalize ŵ on CPU so each bin sums to 1; downstream kernels read
    # workspace's pre-normalized values.
    for (b, ŵ_dst) in enumerate((ws.ŵ_1, ws.ŵ_2, ws.ŵ_3))
        ŵ64 = Float64.(basis.ŵ_bins[b])
        s = sum(ŵ64)
        s > 0 || error("apply_rwls!: basis.ŵ_bins[$b] sums to 0; cannot normalize.")
        copyto!(ŵ_dst, Float32.(ŵ64 ./ s))
    end
    copyto!(ws.p_vec, Float32.(Array(basis.p)))
    copyto!(ws.q_vec, Float32.(Array(basis.q)))

    α_bw = 2.0 * Float64(α) * Float64(β_water)
    α_bI = 2.0 * Float64(α) * Float64(β_iodine)
    step_I  = Float32(step_lim_iodine)
    step_W  = Float32(step_lim_water)
    relax_f = Float32(relax)

    I0_1 = Float32(I0[1]);  I0_2 = Float32(I0[2]);  I0_3 = Float32(I0[3])

    n_view = sino_shape[3]
    n_tiles = cld(n_view, ws.tile_size)
    do_prior = (α_bw > 0.0) || (α_bI > 0.0)

    t0 = time()
    for tile_range in tile_ranges(n_view, ws.tile_size)
        k = length(tile_range)
        # Slice inputs and workspace parents to this tile's actual length.
        # Single-tile case (full volume fits) — these are identity views.
        sI_t = view(sino_iodine, :, :, tile_range)
        sW_t = view(sino_water,  :, :, tile_range)
        b1_t = view(bins[1],     :, :, tile_range)
        b2_t = view(bins[2],     :, :, tile_range)
        b3_t = view(bins[3],     :, :, tile_range)
        H11_t = view(ws.H11, :, :, 1:k);  H12_t = view(ws.H12, :, :, 1:k);  H22_t = view(ws.H22, :, :, 1:k)
        g1_t  = view(ws.g1,  :, :, 1:k);  g2_t  = view(ws.g2,  :, :, 1:k)
        cg_b_t  = view(ws.cg_b,  :, :, 1:k)
        cg_r_t  = view(ws.cg_r,  :, :, 1:k)
        cg_p_t  = view(ws.cg_p,  :, :, 1:k)
        cg_Ap_t = view(ws.cg_Ap, :, :, 1:k)

        for iter in 1:Int(n_iter)
            # Forward + Hessian + gradient (one fused kernel launch).
            _rwls_fused_step_n3!(
                H11_t, H12_t, H22_t, g1_t, g2_t,
                sI_t, sW_t,
                b1_t, b2_t, b3_t,
                ws.p_vec, ws.q_vec,
                ws.ŵ_1, ws.ŵ_2, ws.ŵ_3,
                I0_1, I0_2, I0_3,
            )

            # 2×2 GN step (clamped + non-negativity projected).  det_H,
            # δ_Wg, δ_Ig computed inline — no scratch needed beyond the
            # 5 H+g views above.
            @. begin
                # Reuse cg_b_t / cg_r_t as transient det_H / δ_Wg / δ_Ig
                # storage — the CG prior overwrites them after the GN step.
                cg_b_t = max(abs(H11_t * H22_t - H12_t * H12_t), 1f-30)        # det_H
                cg_r_t = clamp((H22_t * g1_t - H12_t * g2_t) / cg_b_t, -step_W, step_W)  # δ_Wg
                cg_p_t = clamp((H11_t * g2_t - H12_t * g1_t) / cg_b_t, -step_I, step_I)  # δ_Ig
                sW_t = max(sW_t + relax_f * cg_r_t, 0f0)
                sI_t = max(sI_t + relax_f * cg_p_t, 0f0)
            end

            # CG proximal prior on the periodic 2D Laplacian (per slice).
            if do_prior
                _rwls_cg_prior!(sW_t, sI_t, α_bw, α_bI,
                                cg_b_t, cg_r_t, cg_p_t, cg_Ap_t;
                                max_iter = cg_max_iter, rel_tol = cg_rel_tol,
                                verbose  = verbose && iter == 1 && first(tile_range) == 1)
            end

            verbose && @info "[apply_rwls!] tile $(first(tile_range)):$(last(tile_range))  iter $iter done"
        end
    end
    dt = time() - t0

    if verbose
        tile_msg = n_tiles == 1 ? "single tile (full volume fits)" :
                                    "$n_tiles tiles of ≤$(ws.tile_size) views"
        @info "[apply_rwls!] $(n_iter) iters × $tile_msg in $(round(dt, digits = 1)) s"
        @info "  α=$(α), β_I=$(β_iodine), β_W=$(β_water), relax=$(relax), step_lim=(I=$(step_lim_iodine), W=$(step_lim_water))"
        @info "  CG prior: max_iter=$(cg_max_iter), rel_tol=$(cg_rel_tol)"
    end

    (n_iter = Int(n_iter),
     α = Float64(α), β_iodine = Float64(β_iodine), β_water = Float64(β_water),
     n_bins = 3, tile_size = ws.tile_size)
end

export RwlsWorkspace, create_rwls_workspace, apply_rwls!
