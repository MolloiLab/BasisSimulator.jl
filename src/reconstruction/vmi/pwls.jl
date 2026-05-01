"""
PWLS-L₂ sinogram restoration for dual-energy basis sinograms.

1:1 port of Long & Fessler (2014) §IV-B 2×2 matrix-curvature SQS on the
Noh-Fessler-Kinahan (2009) cost, with optional bowtie-aware per-ray
spectrum.  Dispatches on `ndims(basis.ŵ_bins[1])`:

  • `ŵ_bins[i] :: Vector{Float32}`           → centered-spectrum (legacy)
  • `ŵ_bins[i] :: Array{Float32, 3}`         → per-ray bowtie spectrum

Cost (Noh 2009 Eq 12):

    Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))² + ½·Σ_l κ_l·‖C·s_l‖²

Per-ray update (Long/Fessler 2014 §IV-B GN form):

    Δs = M⁻¹ · g,  M = data_curv(2×2) + diag(κ_I, κ_W),
                     g = data_grad + κ_l · C·s

The 2×2 M captures the iodine↔water anti-correlation that diagonal-SQS
majorants miss.

# Workspace + auto-tiling

Workspace is **required**.  `create_pwls_workspace` probes available
memory and chooses a tile size such that the workspace's four sino-shape
parents fit.  `apply_pwls!` iterates the SQS pipeline tile-by-tile along
dim 3 (view axis), so a volume that exceeds device memory still completes
— only the tile size shrinks transparently.  The Laplacian operates on
dims 1, 2 only (per-slice), so tiling is bit-identical.

Reference:
  Noh, Fessler, Kinahan (2009) IEEE TMI 28(11):1688–1702 (cost).
  Long, Fessler       (2014)  IEEE TMI 33(8):1614–1626 (2×2 curvature).
"""

# ════════════════════════════════════════════════════════════════════════
#  Workspace
# ════════════════════════════════════════════════════════════════════════

"""
    PwlsWorkspace{T, A3}

Pre-allocated scratch for `apply_pwls!`.  Each sino-shape parent is sized
`(n_col, n_row, tile_size)`; per-call inner kernels operate on contiguous
views over `1:k` of dim 3.

Layout:
  • 2 regularizer-gradient buffers (per-iter snapshot): `reg_I`, `reg_W`
  • 1 Laplacian intermediate: `tmp_buf`
  • 1 per-pixel data-cost buffer (used in cost reduction): `cost_data`

Total **4 sino-shape parents** at `tile_size` views.

Construct with [`create_pwls_workspace`](@ref); thread-unsafe (one
workspace per task).
"""
mutable struct PwlsWorkspace{T <: AbstractFloat, A3 <: AbstractArray{T, 3}}
    reg_I::A3
    reg_W::A3
    tmp_buf::A3
    cost_data::A3
    tile_size::Int
end

"""
    create_pwls_workspace(sino_template; mem_budget_GB = nothing) -> PwlsWorkspace

Allocate the PWLS workspace on the same backend as `sino_template`.

# Arguments
- `sino_template` : `AbstractArray{<:AbstractFloat, 3}` whose shape and
  backend the workspace should match.

# Keyword arguments
- `mem_budget_GB` : explicit memory-budget override.  `nothing` =
  auto-probe via `Sys.free_memory() · 0.6`.  **Pass explicitly on
  discrete-GPU systems** (CUDA / ROCm / Intel Arc).

# Failure
On `OutOfMemoryError` / `OutOfGPUMemoryError`, halves `tile_size` and
retries up to 4 times.  Hard error if even `tile_size = 1` fails.
"""
function create_pwls_workspace(
        sino_template::AbstractArray{<:AbstractFloat, 3};
        mem_budget_GB::Union{Nothing, Real} = nothing,
    )
    T = eltype(sino_template)
    n_col, n_row, n_view = size(sino_template)

    # 4 sino-shape parents × tile_size views.
    per_view_bytes = 4 * n_col * n_row * sizeof(T)
    initial_tile = suggest_tile_size(per_view_bytes, n_view; mem_budget_GB)

    with_oom_retry("create_pwls_workspace", initial_tile) do tile_size
        sh = (n_col, n_row, Int(tile_size))
        reg_I     = similar(sino_template, T, sh);  fill!(reg_I,     zero(T))
        reg_W     = similar(sino_template, T, sh);  fill!(reg_W,     zero(T))
        tmp_buf   = similar(sino_template, T, sh);  fill!(tmp_buf,   zero(T))
        cost_data = similar(sino_template, T, sh);  fill!(cost_data, zero(T))
        PwlsWorkspace{T, typeof(reg_I)}(reg_I, reg_W, tmp_buf, cost_data, Int(tile_size))
    end
end

# ════════════════════════════════════════════════════════════════════════
#  Laplacian helpers (Neumann BC, per-slice on dims 1, 2)
# ════════════════════════════════════════════════════════════════════════
# CᵀC·s = Cx²·s + Cy²·s — 4-pass formulation, two for each axis.
# Operate on the tile dim transparently because dim 3 is a batch axis.

@inline function _pwls_lapl_x!(out, s, n_col::Int, n_row::Int)
    AK.foreachindex(out) do idx
        i0 = idx - 1
        c = (i0 % n_col) + 1
        r = ((i0 ÷ n_col) % n_row) + 1
        v = ((i0 ÷ (n_col * n_row))) + 1
        cl = c == 1     ? c : c - 1
        cr = c == n_col ? c : c + 1
        out[c, r, v] = s[cl, r, v] - 2f0 * s[c, r, v] + s[cr, r, v]
    end
    return
end

@inline function _pwls_lapl_y!(out, s, n_col::Int, n_row::Int, accum::Bool)
    AK.foreachindex(out) do idx
        i0 = idx - 1
        c = (i0 % n_col) + 1
        r = ((i0 ÷ n_col) % n_row) + 1
        v = ((i0 ÷ (n_col * n_row))) + 1
        ru = r == 1     ? r : r - 1
        rd = r == n_row ? r : r + 1
        val = s[c, ru, v] - 2f0 * s[c, r, v] + s[c, rd, v]
        out[c, r, v] = accum ? out[c, r, v] + val : val
    end
    return
end

@inline function _pwls_apply_CtC!(out, s, tmp, n_col::Int, n_row::Int)
    _pwls_lapl_x!(tmp, s,   n_col, n_row)
    _pwls_lapl_x!(out, tmp, n_col, n_row)
    _pwls_lapl_y!(tmp, s,   n_col, n_row, false)
    _pwls_lapl_y!(out, tmp, n_col, n_row, true)
    return
end

# ════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════

"""
    apply_pwls!(ws, sino_iodine, sino_water, h_low, h_high;
                basis,
                κ_iodine = 32.0f0, κ_water = 32.0f0,
                n_iter   = 20, relax = 1.0f0,
                verbose  = true) -> NamedTuple

2×2 matrix-curvature PWLS-L₂ on the DE basis-sinogram pair.  Mutates
`sino_iodine` and `sino_water` in place — call with a Cong / RWLS / CMV
warm start.

# Arguments
- `ws`            : `PwlsWorkspace` from `create_pwls_workspace`.
  REQUIRED — no internal allocation path.
- `sino_iodine`,
  `sino_water`    : Float32 sinograms (g/cm²), shape
  `[n_col, n_row, n_view]`.  Mutated in place.
- `h_low`, `h_high` : Float32 measured log line integrals at low / high kVp.

# Keyword arguments
- `basis`           : NamedTuple with
    - `ŵ_bins::Vector` of length 2 — low + high spectral weights, each
      `[n_E]` or `[n_col, n_row, n_E]` for per-ray bowtie.
    - `p_bins::Vector{Vector{Float32}}` of length 2 — iodine μρ aligned
      with each bin's energy grid (`p_bins[1]` ↔ `ŵ_bins[1]`).
    - `q_bins::Vector{Vector{Float32}}` of length 2 — water μρ analogous.
- `I0_L`, `I0_H`     : Per-bin Poisson photon-count weights.  The optimal
  weight on the data-fidelity term in line-integral space is
  `w(h) = I0 · exp(-h) = I0 · T`; PWLS uses `exp(-h)` and multiplies by
  `I0_L` / `I0_H` to capture the per-bin Poisson scaling.
  - **DE-CT (typical use)**: `I0_L ≈ I0_H` (mAs is tuned per-kVp to balance
    dose), so the default `(1.0, 1.0)` is approximately correct.
  - **PCCT bin-grouped**: `I0` differs significantly between bin groups
    (e.g., `[[1,2], [3,4]]` at 140 kVp gives `I0_low ≈ 0.4·I0_high`).
    **Pass actual** `I0_low`, `I0_high` (sum-of-bins-in-group for grouped
    PCCT) to avoid over-trusting the lower-count bin.
  Internally normalized by `mean(I0_L, I0_H)` so passing `(1.0, 1.0)` or
  `(actual_I0_L, actual_I0_H)` keeps the κ scale comparable.
- `κ_iodine`, `κ_water` : De Pierro row-sum bounds on the regularizer
  curvature (per-basis).  Larger → more smoothing per iter.
- `n_iter`           : SQS iterations.
- `relax`            : SQS relaxation (1.0 = unrelaxed; 0.5–1.0 typical).
- `verbose`          : log per-iter cost decrease.

Returns `(cost_history, n_iter, κ_iodine, κ_water, relax, tile_size)`.
"""
function apply_pwls!(
        ws::PwlsWorkspace,
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        h_low::AbstractArray{Float32, 3},
        h_high::AbstractArray{Float32, 3};
        basis,
        I0_L::Real     = 1.0,
        I0_H::Real     = 1.0,
        κ_iodine::Real = 32.0f0,
        κ_water::Real  = 32.0f0,
        n_iter::Integer = 20,
        relax::Real     = 1.0f0,
        verbose::Bool   = true,
    )
    # ── Validate basis (per-bin tables, length 2).
    length(basis.ŵ_bins) == 2 ||
        error("apply_pwls!: 2×2 matrix curvature requires exactly 2 bins; got $(length(basis.ŵ_bins)).")
    (hasproperty(basis, :p_bins) && hasproperty(basis, :q_bins)) ||
        error("apply_pwls!: basis must carry `p_bins::Vector{Vector{Float32}}` and `q_bins::Vector{Vector{Float32}}` of length 2.")
    length(basis.p_bins) == 2 && length(basis.q_bins) == 2 ||
        error("apply_pwls!: basis.p_bins / basis.q_bins must each have length 2.")
    for (name, arr) in ((Symbol("p_bins[1]"), basis.p_bins[1]),
                        (Symbol("p_bins[2]"), basis.p_bins[2]),
                        (Symbol("q_bins[1]"), basis.q_bins[1]),
                        (Symbol("q_bins[2]"), basis.q_bins[2]),
                        (Symbol("ŵ_bins[1]"), basis.ŵ_bins[1]),
                        (Symbol("ŵ_bins[2]"), basis.ŵ_bins[2]))
        eltype(arr) === Float32 ||
            error("apply_pwls!: basis.$(name) has eltype $(eltype(arr)); Float32 required.")
    end

    per_ray = ndims(basis.ŵ_bins[1]) == 3
    per_ray == (ndims(basis.ŵ_bins[2]) == 3) ||
        error("apply_pwls!: basis.ŵ_bins[1] and basis.ŵ_bins[2] must share ndims (both 1D or both 3D).")

    # ── Validate workspace shape compatibility.
    sino_shape = size(sino_iodine)
    sino_shape == size(sino_water) == size(h_low) == size(h_high) ||
        error("apply_pwls!: sino_iodine / sino_water / h_low / h_high must share shape.")
    sino_shape[1:2] == size(ws.reg_I)[1:2] ||
        error("apply_pwls!: sino (col,row) = $(sino_shape[1:2]) ≠ workspace (col,row) = $(size(ws.reg_I)[1:2]).")
    size(ws.reg_I, 3) == ws.tile_size ||
        error("apply_pwls!: workspace.tile_size mismatch with reg_I's dim-3 — corrupted workspace.")

    # ── Normalize ŵ on CPU (each bin sums to 1, per-ray when 3D) and
    # stage onto sinogram backend.
    function _normalize_ŵ(ŵ_raw)
        ŵ = Float32.(Array(ŵ_raw))
        if ndims(ŵ) == 1
            ŵ ./= sum(ŵ)
        else
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
    ŵ_L = _match_backend(ŵ_L_cpu, sino_iodine)
    ŵ_H = _match_backend(ŵ_H_cpu, sino_iodine)
    p_L = _match_backend(basis.p_bins[1], sino_iodine)
    q_L = _match_backend(basis.q_bins[1], sino_iodine)
    p_H = _match_backend(basis.p_bins[2], sino_iodine)
    q_H = _match_backend(basis.q_bins[2], sino_iodine)

    nE_L = per_ray ? size(ŵ_L, 3) : length(ŵ_L)
    nE_H = per_ray ? size(ŵ_H, 3) : length(ŵ_H)
    length(p_L) == nE_L && length(q_L) == nE_L ||
        error("apply_pwls!: low-bin (p, q) length ($(length(p_L)), $(length(q_L))) must equal nE_L = $nE_L.")
    length(p_H) == nE_H && length(q_H) == nE_H ||
        error("apply_pwls!: high-bin (p, q) length ($(length(p_H)), $(length(q_H))) must equal nE_H = $nE_H.")

    n_col = Int(sino_shape[1])
    n_row = Int(sino_shape[2])
    κ_I_f   = Float32(κ_iodine)
    κ_W_f   = Float32(κ_water)
    relax_f = Float32(relax)
    n_it    = Int(n_iter)

    # Per-bin Poisson weight scales — normalize by mean so default (1.0, 1.0)
    # reproduces the legacy weighting and the κ scale stays comparable when
    # users pass actual I0 values (typical PCCT use).
    I0_L_raw = Float32(I0_L)
    I0_H_raw = Float32(I0_H)
    I0_L_raw > 0 && I0_H_raw > 0 ||
        error("apply_pwls!: I0_L = $(I0_L_raw), I0_H = $(I0_H_raw) — must both be positive.")
    I0_avg   = 0.5f0 * (I0_L_raw + I0_H_raw)
    wL_scale = I0_L_raw / I0_avg
    wH_scale = I0_H_raw / I0_avg

    n_view = sino_shape[3]
    n_tiles = cld(n_view, ws.tile_size)
    cost_history = Float64[]

    t0 = time()
    for iter in 1:n_it
        # Cost accumulators across tiles (sum reductions are linear; safe to
        # accumulate per tile and combine).
        Φ_data = 0.0
        Φ_reg  = 0.0

        for tile_range in tile_ranges(n_view, ws.tile_size)
            k = length(tile_range)
            sI_t = view(sino_iodine, :, :, tile_range)
            sW_t = view(sino_water,  :, :, tile_range)
            hL_t = view(h_low,       :, :, tile_range)
            hH_t = view(h_high,      :, :, tile_range)
            reg_I_t   = view(ws.reg_I,     :, :, 1:k)
            reg_W_t   = view(ws.reg_W,     :, :, 1:k)
            tmp_t     = view(ws.tmp_buf,   :, :, 1:k)
            cost_t    = view(ws.cost_data, :, :, 1:k)

            # Jacobi SQS: snapshot regularizer gradients at iterate.
            _pwls_apply_CtC!(reg_I_t, sI_t, tmp_t, n_col, n_row)
            _pwls_apply_CtC!(reg_W_t, sW_t, tmp_t, n_col, n_row)

            AK.foreachindex(sI_t) do idx
                Iv = sI_t[idx]
                Wv = sW_t[idx]

                i0    = idx - 1
                col_k = (i0 % n_col) + 1
                row_k = ((i0 ÷ n_col) % n_row) + 1

                # Low-kVp Beer moments
                Z_L = 0f0;  Z_Lp = 0f0;  Z_Lq = 0f0
                if per_ray
                    for kk in 1:nE_L
                        wk = ŵ_L[col_k, row_k, kk] * exp(-p_L[kk] * Iv - q_L[kk] * Wv)
                        Z_L += wk;  Z_Lp += p_L[kk] * wk;  Z_Lq += q_L[kk] * wk
                    end
                else
                    for kk in 1:nE_L
                        wk = ŵ_L[kk] * exp(-p_L[kk] * Iv - q_L[kk] * Wv)
                        Z_L += wk;  Z_Lp += p_L[kk] * wk;  Z_Lq += q_L[kk] * wk
                    end
                end
                invZ_L = 1f0 / max(Z_L, 1f-20)
                P_L = Z_Lp * invZ_L;  Q_L = Z_Lq * invZ_L
                f_L = -log(max(Z_L, 1f-20))

                # High-kVp Beer moments
                Z_H = 0f0;  Z_Hp = 0f0;  Z_Hq = 0f0
                if per_ray
                    for kk in 1:nE_H
                        wk = ŵ_H[col_k, row_k, kk] * exp(-p_H[kk] * Iv - q_H[kk] * Wv)
                        Z_H += wk;  Z_Hp += p_H[kk] * wk;  Z_Hq += q_H[kk] * wk
                    end
                else
                    for kk in 1:nE_H
                        wk = ŵ_H[kk] * exp(-p_H[kk] * Iv - q_H[kk] * Wv)
                        Z_H += wk;  Z_Hp += p_H[kk] * wk;  Z_Hq += q_H[kk] * wk
                    end
                end
                invZ_H = 1f0 / max(Z_H, 1f-20)
                P_H = Z_Hp * invZ_H;  Q_H = Z_Hq * invZ_H
                f_H = -log(max(Z_H, 1f-20))

                h_Lv = hL_t[idx]
                h_Hv = hH_t[idx]
                res_L = f_L - h_Lv
                res_H = f_H - h_Hv
                # Poisson weight in line-integral space: w = I0·T = I0·exp(-h).
                # Per-bin I0 scale captures the count-statistics asymmetry that
                # matters for PCCT bin-grouped data (mostly ~1 for DE-CT).
                wL = wL_scale * exp(-h_Lv)
                wH = wH_scale * exp(-h_Hv)

                cost_t[idx] = 0.5f0 * (wL * res_L * res_L + wH * res_H * res_H)

                g_d_I = wL * res_L * P_L + wH * res_H * P_H
                g_d_W = wL * res_L * Q_L + wH * res_H * Q_H
                cd_II = wL * P_L * P_L   + wH * P_H * P_H
                cd_IW = wL * P_L * Q_L   + wH * P_H * Q_H
                cd_WW = wL * Q_L * Q_L   + wH * Q_H * Q_H

                rg_I = reg_I_t[idx]
                rg_W = reg_W_t[idx]
                gI = g_d_I + rg_I
                gW = g_d_W + rg_W
                m_II = cd_II + κ_I_f
                m_IW = cd_IW
                m_WW = cd_WW + κ_W_f

                det_m   = m_II * m_WW - m_IW * m_IW
                inv_det = 1f0 / max(det_m, 1f-20)
                ΔI = inv_det * (m_WW * gI - m_IW * gW)
                ΔW = inv_det * (m_II * gW - m_IW * gI)

                sI_t[idx] = max(Iv - relax_f * ΔI, 0f0)
                sW_t[idx] = max(Wv - relax_f * ΔW, 0f0)
            end

            Φ_data += Float64(sum(cost_t))
            Φ_reg  += 0.5 * Float64(sum(sI_t .* reg_I_t) + sum(sW_t .* reg_W_t))
        end  # tile loop

        push!(cost_history, Φ_data + Φ_reg)
        if iter > 1 && cost_history[iter] > cost_history[iter-1]
            @warn "apply_pwls!: cost increased iter $(iter-1)→$iter  ($(cost_history[iter-1]) → $(cost_history[iter])).  relax too large?"
        end
    end
    dt = time() - t0

    if verbose
        basis_mode = per_ray ? "per-ray bowtie" : "centered (1D)"
        tile_msg = n_tiles == 1 ? "single tile" : "$n_tiles tiles of ≤$(ws.tile_size) views"
        @info "[apply_pwls! ($basis_mode ŵ, $tile_msg)] $(n_it) iters, $(round(dt, digits = 1)) s, $(round(1000 * dt / max(n_it, 1), digits = 0)) ms/iter"
        @info "  κ_I=$(κ_I_f), κ_W=$(κ_W_f), relax=$(relax_f),  I0_scale (L, H) = ($(round(wL_scale, digits = 3)), $(round(wH_scale, digits = 3)))"
        @info "  Φ: $(round(cost_history[1], sigdigits = 5)) → $(round(cost_history[end], sigdigits = 5))   ($(round(100 * (cost_history[1] - cost_history[end]) / abs(cost_history[1] + eps()), digits = 2))% decrease)"
    end

    (cost_history = cost_history,
     n_iter       = n_it,
     κ_iodine     = κ_I_f,
     κ_water      = κ_W_f,
     relax        = relax_f,
     tile_size    = ws.tile_size)
end

export PwlsWorkspace, create_pwls_workspace, apply_pwls!
