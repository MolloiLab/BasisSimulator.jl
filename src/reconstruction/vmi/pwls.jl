"""
PWLS-L₂ sinogram restoration for dual-energy basis sinograms.

1:1 port of Long & Fessler (2014) §IV-B 2×2 matrix-curvature SQS on the
Noh-Fessler-Kinahan (2009) cost, with an optional bowtie-aware per-ray
spectrum.  Dispatches on `ndims(basis.ŵ_L)`:

  • `ŵ_L[n_E]`                 → centered-spectrum (legacy, uniform rays)
  • `ŵ_L[n_col, n_row, n_E]`   → per-ray bowtie spectrum (3D)

Cost (Noh 2009 Eq 12):

    Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))² + ½·Σ_l κ_l·‖C·s_l‖²

where
  - `s = (s_I, s_W)` — iodine + water basis-sinogram state (g/cm²)
  - `h_{mi}`         — measured −log(y_{mi}/I_m) at kVp m
  - `f_m(s_i)`       — polychromatic forward model  (−log Σ_k ŵ_m[k]·exp(…))
  - `w_{mi} ≈ y_{mi}` — Poisson-approximate inverse-variance weights
  - `‖C·s_l‖²`       — 2D 2nd-order difference (col + row axes, Neumann BC)

Per-ray update (Long/Fessler 2014 §IV-B GN form):

    Δs = M⁻¹ · g,    M = data_curv(2×2) + diag(κ_I, κ_W),
                      g = data_grad + κ_l · C·s

The 2×2 M captures the iodine↔water anti-correlation that diagonal-SQS
majorants miss, which is why this form dominates diagonal PWLS in DE.

Runs on whatever backend the sinograms live on (CPU `Array`, Metal
`MtlArray`, CUDA `CuArray`, …) via `AK.foreachindex`.

Reference:
  Noh, Fessler, Kinahan (2009) IEEE TMI 28(11):1688–1702 (cost).
  Long, Fessler       (2014)  IEEE TMI 33(8):1614–1626 (2×2 curvature).
"""

"""
    apply_pwls!(sino_iodine, sino_water, h_low, h_high;
                basis,
                κ_iodine = 32.0f0, κ_water = 32.0f0,
                n_iter = 20, relax = 1.0f0,
                verbose = true) -> NamedTuple

2×2 matrix-curvature PWLS-L₂ on the DE basis-sinogram pair.  Mutates
`sino_iodine` and `sino_water` in place — call with a Cong/CMV/RWLS warm
start.

# Arguments
- `sino_iodine`, `sino_water` : Float32 sinograms (g/cm²), shape
  `[n_col, n_row, n_view]`.  Mutated in place.
- `h_low`, `h_high`           : Float32 measured log line integrals at
  low/high kVp, same shape.

# Keyword arguments
- `basis`       : NamedTuple with `ŵ_bins::Vector` of length 2 (low + high
  bin-group spectral weights, each `[n_E]` or `[n_col, n_row, n_E]`) plus
  `p::Vector{Float32}` (iodine μρ) and `q::Vector{Float32}` (water μρ) shared
  across bins.  Same structure `apply_rwls!` uses.  Build with
  `BS.pcct_pwls_basis(scanner, protocol; sim_opts, low_bins, high_bins)`.
- `κ_iodine`, `κ_water` : De Pierro row-sum bounds on the regularizer
  curvature (per-basis).  Larger → more smoothing per iter.
- `n_iter`      : SQS iterations.
- `relax`       : SQS relaxation (1.0 = unrelaxed; 0.5–1.0 typical).
- `verbose`     : log cost decrease per iter summary.

Returns a NamedTuple `(cost_history, n_iter, κ_iodine, κ_water, relax)`.
"""
function apply_pwls!(
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        h_low::AbstractArray{Float32, 3},
        h_high::AbstractArray{Float32, 3};
        basis,
        κ_iodine::Real = 32.0f0,
        κ_water::Real  = 32.0f0,
        n_iter::Integer = 20,
        relax::Real     = 1.0f0,
        verbose::Bool   = true,
    )
    # Unified basis form shared with `apply_rwls!`: (ŵ_bins, p, q).  Exactly
    # two bin-groups expected (low/high).  All bins share the energy grid and
    # therefore the p(E), q(E) tables.
    length(basis.ŵ_bins) == 2 ||
        error("apply_pwls!: 2×2 matrix curvature requires exactly 2 bins; got $(length(basis.ŵ_bins)).")
    for (name, arr) in ((:p, basis.p), (:q, basis.q),
                        (Symbol("ŵ_bins[1]"), basis.ŵ_bins[1]),
                        (Symbol("ŵ_bins[2]"), basis.ŵ_bins[2]))
        eltype(arr) === Float32 || error(
            "apply_pwls!: basis.$(name) has eltype $(eltype(arr)); Float32 required."
        )
    end

    # 1D vs 3D dispatch flag (fails loud if mixed).
    per_ray = ndims(basis.ŵ_bins[1]) == 3
    per_ray == (ndims(basis.ŵ_bins[2]) == 3) ||
        error("apply_pwls!: basis.ŵ_bins[1] and basis.ŵ_bins[2] must share ndims (both 1D or both 3D).")

    # Normalize each bin's spectrum so Σ_k ŵ = 1 (per-ray when 3D bowtie).
    # Forward model assumes this: at an air ray f_m = −log(Σ ŵ_m) = 0.
    # `pcct_pwls_basis` already normalizes 1D; this is a defensive check.
    function _normalize_ŵ(ŵ_raw)
        ŵ = Float32.(Array(ŵ_raw))
        if ndims(ŵ) == 1
            ŵ ./= sum(ŵ)
        else  # 3D [n_col, n_row, n_E] — renormalize per-ray
            nc, nr, nE = size(ŵ)
            @inbounds for r in 1:nr, c in 1:nc
                s = 0f0
                for k in 1:nE; s += ŵ[c, r, k]; end
                if s > 0f0
                    inv_s = 1f0 / s
                    for k in 1:nE; ŵ[c, r, k] *= inv_s; end
                end
            end
        end
        ŵ
    end
    ŵ_L_cpu = _normalize_ŵ(basis.ŵ_bins[1])
    ŵ_H_cpu = _normalize_ŵ(basis.ŵ_bins[2])

    # Stage spectral tables onto the sinogram's backend.
    ŵ_L = _match_backend(ŵ_L_cpu, sino_iodine)
    ŵ_H = _match_backend(ŵ_H_cpu, sino_iodine)
    p_L = _match_backend(basis.p, sino_iodine)
    q_L = _match_backend(basis.q, sino_iodine)
    p_H = p_L
    q_H = q_L

    nE_L = per_ray ? size(ŵ_L, 3) : length(ŵ_L)
    nE_H = per_ray ? size(ŵ_H, 3) : length(ŵ_H)

    n_col  = Int(size(sino_iodine, 1))
    n_row  = Int(size(sino_iodine, 2))

    κ_I_f   = Float32(κ_iodine)
    κ_W_f   = Float32(κ_water)
    relax_f = Float32(relax)
    n_it    = Int(n_iter)

    # Scratch buffers on the sinogram's backend (single alloc, reused per iter).
    reg_I     = similar(sino_iodine)
    reg_W     = similar(sino_iodine)
    tmp_buf   = similar(sino_iodine)
    cost_data = similar(sino_iodine)

    # ── 2D 2nd-order difference (C = Cx ⊕ Cy, Neumann BC) — GPU-friendly
    # 4-pass formulation (two for the x-axis, two for the y-axis).
    lapl_x! = function (out, s, nc, nr)
        AK.foreachindex(out) do idx
            i0 = idx - 1
            c = (i0 % nc) + 1
            r = ((i0 ÷ nc) % nr) + 1
            v = ((i0 ÷ (nc * nr))) + 1
            cl = c == 1  ? c : c - 1
            cr = c == nc ? c : c + 1
            out[c, r, v] = s[cl, r, v] - 2f0 * s[c, r, v] + s[cr, r, v]
        end
        return
    end
    lapl_y! = function (out, s, nc, nr, accum::Bool)
        AK.foreachindex(out) do idx
            i0 = idx - 1
            c = (i0 % nc) + 1
            r = ((i0 ÷ nc) % nr) + 1
            v = ((i0 ÷ (nc * nr))) + 1
            ru = r == 1  ? r : r - 1
            rd = r == nr ? r : r + 1
            val = s[c, ru, v] - 2f0 * s[c, r, v] + s[c, rd, v]
            out[c, r, v] = accum ? out[c, r, v] + val : val
        end
        return
    end
    apply_CtC! = function (out, s, tmp)
        lapl_x!(tmp, s,   n_col, n_row)           # tmp = Cx·s
        lapl_x!(out, tmp, n_col, n_row)           # out = Cx·tmp = Cx²·s
        lapl_y!(tmp, s,   n_col, n_row, false)    # tmp = Cy·s
        lapl_y!(out, tmp, n_col, n_row, true)     # out += Cy·tmp = Cx²·s + Cy²·s
        return
    end

    cost_history = Float64[]

    t0 = time()
    for iter in 1:n_it
        # Jacobi SQS: snapshot regularizer gradients at the current iterate.
        apply_CtC!(reg_I, sino_iodine, tmp_buf)
        apply_CtC!(reg_W, sino_water,  tmp_buf)

        AK.foreachindex(sino_iodine) do idx
            Iv = sino_iodine[idx]
            Wv = sino_water[idx]

            # Decode (col, row) for per-ray spectrum lookup.
            i0    = idx - 1
            col_k = (i0 % n_col) + 1
            row_k = ((i0 ÷ n_col) % n_row) + 1

            # ── Low-kVp spectral Beer moments.
            Z_L = 0f0;  Z_Lp = 0f0;  Z_Lq = 0f0
            if per_ray
                for k in 1:nE_L
                    wk = ŵ_L[col_k, row_k, k] * exp(-p_L[k] * Iv - q_L[k] * Wv)
                    Z_L += wk;  Z_Lp += p_L[k] * wk;  Z_Lq += q_L[k] * wk
                end
            else
                for k in 1:nE_L
                    wk = ŵ_L[k] * exp(-p_L[k] * Iv - q_L[k] * Wv)
                    Z_L += wk;  Z_Lp += p_L[k] * wk;  Z_Lq += q_L[k] * wk
                end
            end
            invZ_L = 1f0 / max(Z_L, 1f-20)
            P_L = Z_Lp * invZ_L;  Q_L = Z_Lq * invZ_L
            f_L = -log(max(Z_L, 1f-20))

            # ── High-kVp spectral Beer moments.
            Z_H = 0f0;  Z_Hp = 0f0;  Z_Hq = 0f0
            if per_ray
                for k in 1:nE_H
                    wk = ŵ_H[col_k, row_k, k] * exp(-p_H[k] * Iv - q_H[k] * Wv)
                    Z_H += wk;  Z_Hp += p_H[k] * wk;  Z_Hq += q_H[k] * wk
                end
            else
                for k in 1:nE_H
                    wk = ŵ_H[k] * exp(-p_H[k] * Iv - q_H[k] * Wv)
                    Z_H += wk;  Z_Hp += p_H[k] * wk;  Z_Hq += q_H[k] * wk
                end
            end
            invZ_H = 1f0 / max(Z_H, 1f-20)
            P_H = Z_Hp * invZ_H;  Q_H = Z_Hq * invZ_H
            f_H = -log(max(Z_H, 1f-20))

            h_Lv = h_low[idx]
            h_Hv = h_high[idx]
            res_L = f_L - h_Lv
            res_H = f_H - h_Hv
            wL = exp(-h_Lv)
            wH = exp(-h_Hv)

            cost_data[idx] = 0.5f0 * (wL * res_L * res_L + wH * res_H * res_H)

            # Data gradient + 2×2 data curvature (Long/Fessler eq 27).
            g_d_I = wL * res_L * P_L + wH * res_H * P_H
            g_d_W = wL * res_L * Q_L + wH * res_H * Q_H
            cd_II = wL * P_L * P_L   + wH * P_H * P_H
            cd_IW = wL * P_L * Q_L   + wH * P_H * Q_H
            cd_WW = wL * Q_L * Q_L   + wH * Q_H * Q_H

            rg_I = reg_I[idx]
            rg_W = reg_W[idx]
            gI = g_d_I + rg_I
            gW = g_d_W + rg_W
            m_II = cd_II + κ_I_f
            m_IW = cd_IW
            m_WW = cd_WW + κ_W_f

            det_m   = m_II * m_WW - m_IW * m_IW
            inv_det = 1f0 / max(det_m, 1f-20)
            ΔI = inv_det * (m_WW * gI - m_IW * gW)
            ΔW = inv_det * (m_II * gW - m_IW * gI)

            sino_iodine[idx] = max(Iv - relax_f * ΔI, 0f0)
            sino_water[idx]  = max(Wv - relax_f * ΔW, 0f0)
        end

        # Backend-native reductions for cost tracking.
        Φ_data = Float64(sum(cost_data))
        Φ_reg  = 0.5 * Float64(sum(sino_iodine .* reg_I) + sum(sino_water .* reg_W))
        push!(cost_history, Φ_data + Φ_reg)

        if iter > 1 && cost_history[iter] > cost_history[iter-1]
            @warn "apply_pwls!: cost increased iter $(iter-1)→$iter  ($(cost_history[iter-1]) → $(cost_history[iter])).  relax too large?"
        end
    end
    dt = time() - t0

    if verbose
        basis_mode = per_ray ? "per-ray bowtie" : "centered (1D)"
        @info "[apply_pwls! ($basis_mode ŵ)] $(n_it) iters, $(round(dt, digits = 1)) s, $(round(1000 * dt / max(n_it, 1), digits = 0)) ms/iter"
        @info "  κ_I=$(κ_I_f), κ_W=$(κ_W_f), relax=$(relax_f)"
        @info "  Φ: $(round(cost_history[1], sigdigits = 5)) → $(round(cost_history[end], sigdigits = 5))   ($(round(100 * (cost_history[1] - cost_history[end]) / abs(cost_history[1] + eps()), digits = 2))% decrease)"
    end

    (cost_history = cost_history,
     n_iter       = n_it,
     κ_iodine     = κ_I_f,
     κ_water      = κ_W_f,
     relax        = relax_f)
end

"""
    apply_pwls(h_low, h_high, sino_iodine_init, sino_water_init; kwargs...)
        -> (sino_iodine, sino_water, info)

Allocating wrapper: deep-copies the warm-start pair on the measurements'
backend and runs `apply_pwls!` in place.  Returns the refined
`(sino_iodine, sino_water)` along with the diagnostic NamedTuple from
`apply_pwls!`.
"""
function apply_pwls(
        h_low::AbstractArray,
        h_high::AbstractArray,
        sino_iodine_init::AbstractArray,
        sino_water_init::AbstractArray;
        kwargs...,
    )
    sino_iodine = similar(h_low, Float32)
    sino_water  = similar(h_low, Float32)
    copyto!(sino_iodine, sino_iodine_init)
    copyto!(sino_water,  sino_water_init)
    h_L = eltype(h_low)  === Float32 ? h_low  : Float32.(h_low)
    h_H = eltype(h_high) === Float32 ? h_high : Float32.(h_high)
    info = apply_pwls!(sino_iodine, sino_water, h_L, h_H; kwargs...)
    (sino_iodine, sino_water, info)
end

export apply_pwls!, apply_pwls
