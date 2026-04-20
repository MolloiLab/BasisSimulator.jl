"""
Cong 2022 per-ray analytic dual-energy basis decomposition.

Given measured line integrals at low and high kVp, invert the
polychromatic forward model **ray by ray** to recover the photoelectric
(`y = ∫a(r)dr`) and Compton (`C = ∫c(r)dr`) basis sinograms.  Zero
calibration required — only the resolved x-ray spectra
(`resolve_spectrum`) and the physical `p(ε), q(ε)` basis (see `basis.jl`).

Algorithm (Cong Eqs 6–10):
  1. **Outer Brent on water-equivalent path `L`**
     solve ∫Ŝ_L(ε)·exp(−(p(ε)·a_w + q(ε)·c_w)·L) dε = T_L_meas
     → `c̄ = c_w · L`
  2. **Inner Newton on the Eq 8 quintic** for `x = h(y)`
     (strictly convex ⇒ unique real root, Newton from x=0 ≤ 12 iters)
  3. **Outer Brent on `y` via Eq 10**
     G(y) = T_H_pred(y, c̄ + h(y)) − T_H_meas
     G is monotonic on `[0, τ_L/min(p_L))`

Parallelized via `AK.foreachindex` — same kernel body compiles for CPU
/ Metal / CUDA automatically.  Each root-find uses `brent_solve`
(see `roots_kernels.jl`), a 1:1 line-for-line port of Roots.jl's
`Brent()` that runs inside GPU kernels without allocations.  Parity vs
Roots.jl is continuously verified by `test/vmi_brent_parity.jl`.

Reference:
  Cong, De Man, Wang (2022) *"Projection decomposition via univariate
  optimization for dual-energy CT."* *J X-Ray Sci Technol* 30:725–736.
  DOI 10.3233/XST-221153.
"""

# ─────────────────────────────────────────────────────────────────────
# Backend helper — move a CPU Vector{Float64} onto whatever device
# `ref` lives on.  Used to stage the small spectral tables onto the
# sinogram's backend before we launch the kernel.
# ─────────────────────────────────────────────────────────────────────

@inline function _match_backend(arr::AbstractArray, ref::AbstractArray)
    ref isa Array && return arr
    out = similar(ref, eltype(arr), size(arr))
    copyto!(out, arr)
    out
end

"""
    apply_cong!(sino_y, sino_c, sino_low, sino_high; basis, water_basis)

Per-ray Cong 2022 decomposition, writing results into `sino_y` and
`sino_c`.  All four arrays share the same shape (nx, nv, nr) and element
type `Float32`.

Runs on whatever backend the sinograms live on (CPU `Array`, Metal
`MtlArray`, CUDA `CuArray`, …) via `AK.foreachindex`.  The spectral
tables from `basis` are automatically staged onto the matching backend
before launch.

# Arguments
- `sino_y`, `sino_c`  : output photoelectric and Compton line integrals
- `sino_low`          : measured low-kVp sinogram (log line integrals)
- `sino_high`         : measured high-kVp sinogram (log line integrals)

# Keyword arguments
- `basis`       : NamedTuple from `compute_photo_compton_basis`
- `water_basis` : NamedTuple from `water_basis_constants()`
"""
function apply_cong!(
        sino_y::AbstractArray{Float32, 3},
        sino_c::AbstractArray{Float32, 3},
        sino_low::AbstractArray,
        sino_high::AbstractArray;
        basis,
        water_basis,
    )
    # Stage spectral tables (Float32) onto the sinogram's backend.  All
    # internal kernel math runs in Float32 so Metal / CUDA execute
    # natively without a Float64 fallback.  Float32 ULP ≈ 1.2e-7 around
    # 1.0 — plenty for the physical line-integral range Cong operates on
    # (0–10 cm·g/cm²).
    ŵ_L = _match_backend(basis.ŵ_L, sino_low)
    p_L = _match_backend(basis.p_L, sino_low)
    q_L = _match_backend(basis.q_L, sino_low)
    ŵ_H = _match_backend(basis.ŵ_H, sino_low)
    p_H = _match_backend(basis.p_H, sino_low)
    q_H = _match_backend(basis.q_H, sino_low)
    nE_L = length(ŵ_L)
    nE_H = length(ŵ_H)

    a_w     = Float32(water_basis.a)
    c_w     = Float32(water_basis.c)
    p_L_min = Float32(minimum(basis.p_L))

    AK.foreachindex(sino_low) do idx
        p_L_meas = Float32(sino_low[idx])
        p_H_meas = Float32(sino_high[idx])

        # Skip air rays.
        if p_L_meas < 1f-6 && p_H_meas < 1f-6
            sino_y[idx] = 0f0
            sino_c[idx] = 0f0
            return
        end
        T_L_meas = exp(-p_L_meas)
        T_H_meas = exp(-p_H_meas)

        # ── Step 1 — Brent on water-equivalent path L (Cong Eq 7) ──
        water_T_L = function (L::Float32)
            T = 0f0
            @inbounds for i in 1:nE_L
                T += ŵ_L[i] * exp(-(p_L[i] * a_w + q_L[i] * c_w) * L)
            end
            T - T_L_meas
        end

        L_water, ok_L = brent_solve(water_T_L, 0f0, 60f0)
        if !ok_L
            sino_y[idx] = 0f0
            sino_c[idx] = 0f0
            return
        end
        c̄ = c_w * L_water

        y_max = min(0.99f0 * p_L_meas / max(p_L_min, eps(Float32)), 1f7)
        if y_max <= 0f0
            sino_y[idx] = 0f0
            sino_c[idx] = c̄
            return
        end

        # ── Step 2 — Newton on Eq 8 quintic for x = h(y) ──
        # Newton tolerance ≈ eps(Float32) ≈ 1.2e-7 (was 1e-14 for Float64).
        solve_quintic = function (y::Float32, c̄_::Float32)
            P0 = 0f0; P1 = 0f0; P2 = 0f0
            P3 = 0f0; P4 = 0f0; P5 = 0f0
            @inbounds for i in 1:nE_L
                q_i  = q_L[i]
                q2   = q_i * q_i;  q3 = q2 * q_i
                q4   = q3 * q_i;   q5 = q4 * q_i
                wexp = ŵ_L[i] * exp(-p_L[i] * y - q_i * c̄_)
                P0 += wexp
                P1 -= wexp * q_i
                P2 += wexp * q2 * 0.5f0
                P3 -= wexp * q3 * (1f0 / 6f0)
                P4 += wexp * q4 * (1f0 / 24f0)
                P5 -= wexp * q5 * (1f0 / 120f0)
            end
            x = 0f0
            for _ in 1:12
                F  = (P0 - T_L_meas) +
                     x * (P1 + x * (P2 + x * (P3 + x * (P4 + x * P5))))
                dF = P1 + x * (2f0*P2 + x * (3f0*P3 + x * (4f0*P4 + x * 5f0*P5)))
                abs(dF) < 1f-30 && break
                Δ = F / dF
                x -= Δ
                abs(Δ) < eps(Float32) && break
            end
            x
        end

        # ── Step 3 — Outer Brent on y via G(y) = T_H_pred − T_H_meas ──
        T_H_pred = function (y::Float32, C::Float32)
            T = 0f0
            @inbounds for i in 1:nE_H
                T += ŵ_H[i] * exp(-p_H[i] * y - q_H[i] * C)
            end
            T
        end

        G = function (y::Float32)
            x = solve_quintic(y, c̄)
            T_H_pred(y, c̄ + x) - T_H_meas
        end

        y_opt, ok_y = brent_solve(G, 0f0, y_max)
        if !ok_y
            # Fall back to water-only estimate (matches CPU reference).
            sino_y[idx] = a_w * L_water
            sino_c[idx] = c_w * L_water
            return
        end
        x_final = solve_quintic(y_opt, c̄)
        sino_y[idx] = max(y_opt, 0f0)
        sino_c[idx] = max(c̄ + x_final, 0f0)
    end

    (sino_y, sino_c)
end

"""
    apply_cong(sino_low, sino_high; basis, water_basis) -> (sino_y, sino_c)

Allocating wrapper around `apply_cong!`.  Returns a fresh `(sino_y,
sino_c)` pair on the same backend as the inputs.
"""
function apply_cong(
        sino_low::AbstractArray,
        sino_high::AbstractArray;
        basis,
        water_basis,
    )
    sino_y = similar(sino_low, Float32)
    sino_c = similar(sino_low, Float32)
    apply_cong!(sino_y, sino_c, sino_low, sino_high;
                basis = basis, water_basis = water_basis)
    (sino_y, sino_c)
end

export apply_cong!, apply_cong
