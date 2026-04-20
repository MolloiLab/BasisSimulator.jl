"""
PWLS-SQS sinogram restoration for dual-energy basis sinograms.

1:1 port of MIRT's `ct/de_wls_dercurv.m` (per-ray WLS gradient +
Fessler-Erdogan precomputed separable curvature) and
`wls/pwls_sqs_os.m` (SQS update loop).

Cost (Noh 2009 Eq 12):

    Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))² + Σ_l ½·γ_l·‖C·s_l‖²

where
- `s = (s_y, s_C)`  — basis-sinogram state (photo + Compton line integrals)
- `h_{mi}`          — measured polychromatic transmissions (−log y_{mi}/I_m)
- `f_m(s_i)`        — polychromatic forward model (same as Cong)
- `w_{mi} ≈ y_{mi}` — Poisson inverse-variance weights (Eq 13)
- `‖C·s_l‖²`        — 2D 2nd-order difference (detector-col + view axes,
                       Neumann BC) — MIRT Cdiffs pattern

Fessler-Erdogan precomputed curvature (MIRT de_wls_dercurv.m 131–133):

    M[m, l]      = Σ_k ŵ_m[k]·c_l[k]
    curv_geom[l] = Σ_m |M[m, l]| · Σ_{l'} |M[m, l']|
    curv_data_l[i] = max_m(w_{mi}) · curv_geom[l]
    curv_R_l       = γ_l · 32
    curv_l[i]      = curv_data_l[i] + curv_R_l

SQS guarantees monotonic decrease of Φ(s) (Fessler 2000).

Reference:
  Noh, Fessler, Kinahan (2009) *"Statistical Sinogram Restoration in
  Dual-Energy CT for PET Attenuation Correction."*
  IEEE Trans Med Imaging 28(11):1688–1702.  DOI 10.1109/TMI.2009.2023988.
"""

"""
    apply_pwls!(sino_y, sino_c;
                h_low, h_high, basis,
                γ_photo=2.0^-8, γ_compton=2.0^-8,
                n_iter=20, relax=1.0,
                verbose=true)

PWLS-SQS restoration of dual-energy basis sinograms.  Mutates `sino_y`
and `sino_c` in place — call with Cong-warm-started arrays.

Returns a NamedTuple with diagnostic info:
- `cost_history` : Φ(s) at each iteration (length = n_iter + 1)
- `n_iter`       : iterations run
- `γ_photo`, `γ_compton`, `relax` : echoed inputs

Emits a `@warn` if Φ ever increases — monotonic descent is guaranteed
by the De Pierro row-sum bound built into `curv_l[i]`, so any increase
indicates a numerical bug or a curvature-bound violation.
"""
function apply_pwls!(
        sino_y::AbstractArray{Float32, 3},
        sino_c::AbstractArray{Float32, 3};
        h_low::AbstractArray,
        h_high::AbstractArray,
        basis,
        γ_photo::Real  = 2.0^(-8),
        γ_compton::Real = 2.0^(-8),
        n_iter::Integer = 20,
        relax::Real    = 1.0,
        verbose::Bool  = true,
    )
    ŵ_L, p_L_bin, q_L_bin = basis.ŵ_L, basis.p_L, basis.q_L
    ŵ_H, p_H_bin, q_H_bin = basis.ŵ_H, basis.p_H, basis.q_H
    nE_L, nE_H = length(ŵ_L), length(ŵ_H)

    # ── Precomputed separable curvature (MIRT de_wls_dercurv.m) ──
    # Spectral-averaged MAC matrix M[m, l] = Σ_k ŵ_m[k]·c_l[k]
    M_Ly = sum(ŵ_L .* p_L_bin);  M_LC = sum(ŵ_L .* q_L_bin)
    M_Hy = sum(ŵ_H .* p_H_bin);  M_HC = sum(ŵ_H .* q_H_bin)
    row_sum_L = abs(M_Ly) + abs(M_LC)
    row_sum_H = abs(M_Hy) + abs(M_HC)
    curv_geom_y = abs(M_Ly) * row_sum_L + abs(M_Hy) * row_sum_H
    curv_geom_C = abs(M_LC) * row_sum_L + abs(M_HC) * row_sum_H

    y_all = Float64.(sino_y)
    C_all = Float64.(sino_c)
    nx, nv, nr = size(y_all)

    # Measurements (log line integrals) + Poisson-approximate weights.
    # w_{mi} ∝ y_{mi} = I_m·exp(−h_{mi}); I_m cancels in num/den of the
    # SQS update (assumes I_L ≈ I_H — good for GE fast-kVp switching).
    h_L_all = Float64.(h_low)
    h_H_all = Float64.(h_high)
    w_L_all = @. exp(-h_L_all)
    w_H_all = @. exp(-h_H_all)

    γy    = Float64(γ_photo)
    γC    = Float64(γ_compton)
    relax_f = Float64(relax)

    # Per-pixel curvatures (constant across iterations).
    # γ·32 is the De Pierro row-sum majorant for |Cx'Cx + Cy'Cy| (2D
    # 2nd-order diff, 16 per axis × 2 axes).
    w_max_all  = @. max(w_L_all, w_H_all)
    curv_y_all = @. w_max_all * curv_geom_y + γy * 32.0
    curv_C_all = @. w_max_all * curv_geom_C + γC * 32.0

    # Per-ray polychromatic forward + Jacobian.
    # MIRT's fit.fmfun + fit.fgrad play the same role.
    @inline ray_fj = function (y_r::Float64, C_r::Float64)
        T_L = 0.0; nLy = 0.0; nLC = 0.0
        @inbounds for i in 1:nE_L
            e = ŵ_L[i] * exp(-p_L_bin[i] * y_r - q_L_bin[i] * C_r)
            T_L += e; nLy += e * p_L_bin[i]; nLC += e * q_L_bin[i]
        end
        T_H = 0.0; nHy = 0.0; nHC = 0.0
        @inbounds for i in 1:nE_H
            e = ŵ_H[i] * exp(-p_H_bin[i] * y_r - q_H_bin[i] * C_r)
            T_H += e; nHy += e * p_H_bin[i]; nHC += e * q_H_bin[i]
        end
        T_L = max(T_L, 1e-30); T_H = max(T_H, 1e-30)
        (-log(T_L), -log(T_H), nLy / T_L, nLC / T_L, nHy / T_H, nHC / T_H)
    end

    # 2D 2nd-order difference (MIRT Cdiffs pattern — Cx stacked with Cy).
    # Both axes use Neumann BC (zero at boundary rows/cols).  Output:
    # `out = Cx'Cx·z + Cy'Cy·z`; `cx_buf = Cx·z`, `cy_buf = Cy·z`.
    apply_CtC! = function (
            out::AbstractMatrix{Float64},
            cx_buf::AbstractMatrix{Float64},
            cy_buf::AbstractMatrix{Float64},
            z::AbstractMatrix{Float64},
        )
        nxl, nvl = size(z)
        @inbounds for j in 1:nvl
            cx_buf[1, j]   = 0.0
            cx_buf[nxl, j] = 0.0
            for i in 2:nxl-1
                cx_buf[i, j] = z[i-1, j] - 2.0 * z[i, j] + z[i+1, j]
            end
        end
        @inbounds for i in 1:nxl
            cy_buf[i, 1]   = 0.0
            cy_buf[i, nvl] = 0.0
        end
        @inbounds for j in 2:nvl-1, i in 1:nxl
            cy_buf[i, j] = z[i, j-1] - 2.0 * z[i, j] + z[i, j+1]
        end
        @inbounds for j in 1:nvl, i in 1:nxl
            left  = (i > 1)   ? cx_buf[i-1, j] : 0.0
            right = (i < nxl) ? cx_buf[i+1, j] : 0.0
            below = (j > 1)   ? cy_buf[i, j-1] : 0.0
            above = (j < nvl) ? cy_buf[i, j+1] : 0.0
            out[i, j] = (left  - 2.0 * cx_buf[i, j] + right) +
                        (below - 2.0 * cy_buf[i, j] + above)
        end
    end

    # Per-slice buffers (allocated once, reused across all SQS iterations).
    grad_y_bufs = [zeros(Float64, nx, nv) for _ in 1:nr]
    grad_C_bufs = [zeros(Float64, nx, nv) for _ in 1:nr]
    reg_y_bufs  = [zeros(Float64, nx, nv) for _ in 1:nr]
    reg_C_bufs  = [zeros(Float64, nx, nv) for _ in 1:nr]
    cx_y_bufs   = [zeros(Float64, nx, nv) for _ in 1:nr]
    cy_y_bufs   = [zeros(Float64, nx, nv) for _ in 1:nr]
    cx_C_bufs   = [zeros(Float64, nx, nv) for _ in 1:nr]
    cy_C_bufs   = [zeros(Float64, nx, nv) for _ in 1:nr]

    cost_history = zeros(Float64, n_iter + 1)

    t0 = time()
    for k_iter in 1:(n_iter + 1)
        is_final_eval = (k_iter == n_iter + 1)
        cost_total = Threads.Atomic{Float64}(0.0)

        Threads.@threads for k_sl in 1:nr
            y_sl = @view y_all[:, :, k_sl]
            C_sl = @view C_all[:, :, k_sl]
            hLs  = @view h_L_all[:, :, k_sl]
            hHs  = @view h_H_all[:, :, k_sl]
            wLs  = @view w_L_all[:, :, k_sl]
            wHs  = @view w_H_all[:, :, k_sl]
            cvy  = @view curv_y_all[:, :, k_sl]
            cvC  = @view curv_C_all[:, :, k_sl]

            grad_y = grad_y_bufs[k_sl]
            grad_C = grad_C_bufs[k_sl]
            reg_y  = reg_y_bufs[k_sl]
            reg_C  = reg_C_bufs[k_sl]
            cx_y   = cx_y_bufs[k_sl]
            cy_y   = cy_y_bufs[k_sl]
            cx_C   = cx_C_bufs[k_sl]
            cy_C   = cy_C_bufs[k_sl]

            # Data term: cost + gradient at current iterate.
            cost_sl = 0.0
            @inbounds for idx in eachindex(y_sl)
                pL, pH, jLy, jLC, jHy, jHC = ray_fj(y_sl[idx], C_sl[idx])
                errL = pL - hLs[idx]
                errH = pH - hHs[idx]
                wL   = wLs[idx]
                wH   = wHs[idx]
                cost_sl += 0.5 * (wL * errL * errL + wH * errH * errH)
                grad_y[idx] = wL * errL * jLy + wH * errH * jHy
                grad_C[idx] = wL * errL * jLC + wH * errH * jHC
            end

            # Regularizer: 2D penalty cost + (Cx'Cx + Cy'Cy)·s.
            apply_CtC!(reg_y, cx_y, cy_y, y_sl)
            apply_CtC!(reg_C, cx_C, cy_C, C_sl)
            @inbounds for idx in eachindex(cx_y)
                cost_sl += 0.5 * γy * (cx_y[idx]*cx_y[idx] + cy_y[idx]*cy_y[idx])
                cost_sl += 0.5 * γC * (cx_C[idx]*cx_C[idx] + cy_C[idx]*cy_C[idx])
            end

            Threads.atomic_add!(cost_total, cost_sl)

            # SQS update (skipped on the final cost-only pass).
            if !is_final_eval
                @inbounds for idx in eachindex(y_sl)
                    dy = (grad_y[idx] + γy * reg_y[idx]) / cvy[idx]
                    dC = (grad_C[idx] + γC * reg_C[idx]) / cvC[idx]
                    y_sl[idx] = max(y_sl[idx] - relax_f * dy, 0.0)
                    C_sl[idx] = max(C_sl[idx] - relax_f * dC, 0.0)
                end
            end
        end

        cost_history[k_iter] = cost_total[]
        if k_iter > 1 && cost_history[k_iter] > cost_history[k_iter-1] * (1 + 1e-9)
            @warn "PWLS cost INCREASED iter $(k_iter-1)→$(k_iter): $(cost_history[k_iter-1]) → $(cost_history[k_iter]).  (sign bug? relax too large?)"
        end
    end
    dt = time() - t0

    # Copy refined values back into caller's Float32 arrays.
    sino_y .= Float32.(y_all)
    sino_c .= Float32.(C_all)

    if verbose
        @info "PWLS-SQS restoration: $(n_iter) iters, $(Threads.nthreads()) threads, $(round(dt, digits = 1)) s"
        @info "  γ_photo=$(γy), γ_compton=$(γC), relax=$(relax_f)"
        @info "  Φ(s): initial=$(round(cost_history[1]; sigdigits = 5)), final=$(round(cost_history[end]; sigdigits = 5))   [SPS monotonic descent]"
        @info "  Total decrease: $(round((cost_history[1] - cost_history[end]) / abs(cost_history[1] + eps()) * 100, digits = 2))%"
    end

    (cost_history = cost_history,
     n_iter       = n_iter,
     γ_photo      = γy,
     γ_compton    = γC,
     relax        = relax_f)
end

export apply_pwls!
