"""
Cong 2022 per-ray analytic dual-energy basis decomposition.

Given measured line integrals at low and high kVp, invert the
polychromatic forward model **ray by ray** to recover the photoelectric
(`y = ∫a(r)dr`) and Compton (`C = ∫c(r)dr`) basis sinograms.  Zero
calibration required — only the resolved x-ray spectra (`resolve_spectrum`)
and the physical `p(ε), q(ε)` basis (see `basis.jl`).

Algorithm (Cong Eqs 6–10):
  1. **Outer Brent on water-equivalent path `L`**
     solve ∫Ŝ_L(ε)·exp(−(p(ε)·a_w + q(ε)·c_w)·L) dε = T_L_meas
     → `c̄ = c_w · L`
  2. **Inner Newton on the Eq 8 quintic** for `x = h(y)`
     (strictly convex ⇒ unique real root, Newton from x=0 ≤ 8 iters)
  3. **Outer Brent on `y` via Eq 10**
     G(y) = T_H_pred(y, c̄ + h(y)) − T_H_meas
     G is monotonic on `[0, τ_L/min(p_L))`

Parallelized over pixels via `Threads.@threads` (CPU).  A future
`apply_cong_gpu!` can be swapped in without API change.

Reference:
  Cong, De Man, Wang (2022) *"Projection decomposition via univariate
  optimization for dual-energy CT."* *J X-Ray Sci Technol* 30:725–736.
  DOI 10.3233/XST-221153.
"""

import Roots

"""
    apply_cong!(sino_y, sino_c, sino_low, sino_high; basis, water_basis)

Per-ray Cong 2022 decomposition, writing results into `sino_y` and
`sino_c`.  All four arrays share the same shape (nx, nv, nr) and element
type `Float32`.

# Arguments
- `sino_y`, `sino_c`  : output photoelectric and Compton line integrals
- `sino_low`          : measured low-kVp sinogram (log line integrals)
- `sino_high`         : measured high-kVp sinogram (log line integrals)

# Keyword arguments
- `basis`       : NamedTuple from `compute_photo_compton_basis`
- `water_basis` : NamedTuple from `water_basis_constants()`

# Returns
The function returns the tuple `(sino_y, sino_c)` unchanged — callers
wanting a new allocation should use `apply_cong` (no bang).
"""
function apply_cong!(
        sino_y::AbstractArray{Float32, 3},
        sino_c::AbstractArray{Float32, 3},
        sino_low::AbstractArray,
        sino_high::AbstractArray;
        basis,
        water_basis,
    )
    ŵ_L, p_L_bin, q_L_bin = basis.ŵ_L, basis.p_L, basis.q_L
    ŵ_H, p_H_bin, q_H_bin = basis.ŵ_H, basis.p_H, basis.q_H
    nE_L, nE_H = length(ŵ_L), length(ŵ_H)
    a_w, c_w   = water_basis.a, water_basis.c
    p_L_min    = minimum(p_L_bin)

    # Water-only predicted transmission at low-kVp for path L.
    water_T_L = function (L::Float64)
        T = 0.0
        @inbounds for i in 1:nE_L
            T += ŵ_L[i] * exp(-(p_L_bin[i] * a_w + q_L_bin[i] * c_w) * L)
        end
        T
    end

    # Eq 8 quintic for x = h(y) given (y, c̄) against T_L_meas. Newton from x=0.
    solve_quintic = function (y::Float64, c̄::Float64, T_L_meas::Float64)
        P0 = 0.0; P1 = 0.0; P2 = 0.0
        P3 = 0.0; P4 = 0.0; P5 = 0.0
        @inbounds for i in 1:nE_L
            q_i  = q_L_bin[i]
            q2   = q_i * q_i; q3 = q2 * q_i
            q4   = q3 * q_i;  q5 = q4 * q_i
            wexp = ŵ_L[i] * exp(-p_L_bin[i] * y - q_i * c̄)
            P0 += wexp
            P1 -= wexp * q_i
            P2 += wexp * q2 * 0.5
            P3 -= wexp * q3 * (1.0 / 6.0)
            P4 += wexp * q4 * (1.0 / 24.0)
            P5 -= wexp * q5 * (1.0 / 120.0)
        end
        x = 0.0
        for _ in 1:12
            F  = (P0 - T_L_meas) +
                 x * (P1 + x * (P2 + x * (P3 + x * (P4 + x * P5))))
            dF = P1 + x * (2P2 + x * (3P3 + x * (4P4 + x * 5P5)))
            abs(dF) < 1e-30 && break
            Δ = F / dF
            x -= Δ
            abs(Δ) < 1e-14 && break
        end
        x
    end

    # Exact high-kVp transmission prediction (Eq 9).
    T_H_pred = function (y::Float64, C::Float64)
        T = 0.0
        @inbounds for i in 1:nE_H
            T += ŵ_H[i] * exp(-p_H_bin[i] * y - q_H_bin[i] * C)
        end
        T
    end

    decompose_ray = function (p_L_meas::Float64, p_H_meas::Float64)
        # Skip air rays.
        if p_L_meas < 1.0e-6 && p_H_meas < 1.0e-6
            return (0.0, 0.0)
        end
        T_L_meas = exp(-p_L_meas)
        T_H_meas = exp(-p_H_meas)

        # Step 1 — water-equivalent path L via Brent.
        L_water = try
            Roots.find_zero(L -> water_T_L(L) - T_L_meas, (0.0, 60.0), Roots.Brent())
        catch
            return (0.0, 0.0)
        end
        c̄ = c_w * L_water

        # Step 3 — outer Brent on y.  Physical upper bound: C ≥ 0 ⇒ p_L_meas ≥ p_L_min·y.
        y_max = min(0.99 * p_L_meas / max(p_L_min, eps()), 1.0e7)
        if y_max <= 0.0
            return (0.0, c̄)
        end

        G(y) = T_H_pred(y, c̄ + solve_quintic(y, c̄, T_L_meas)) - T_H_meas

        y_opt = try
            Roots.find_zero(G, (0.0, y_max), Roots.Brent())
        catch
            # Fall back to water-only estimate if bracketing/convergence fails.
            return (a_w * L_water, c_w * L_water)
        end
        x_final = solve_quintic(y_opt, c̄, T_L_meas)
        (max(y_opt, 0.0), max(c̄ + x_final, 0.0))
    end

    @inbounds Threads.@threads for idx in eachindex(sino_low)
        p_L = Float64(sino_low[idx])
        p_H = Float64(sino_high[idx])
        (y, C)            = decompose_ray(p_L, p_H)
        sino_y[idx]       = Float32(y)
        sino_c[idx]       = Float32(C)
    end
    (sino_y, sino_c)
end

"""
    apply_cong(sino_low, sino_high; basis, water_basis) -> (sino_y, sino_c)

Allocating wrapper around `apply_cong!`.
"""
function apply_cong(
        sino_low::AbstractArray,
        sino_high::AbstractArray;
        basis,
        water_basis,
    )
    sino_y = similar(sino_low, Float32)
    sino_c = similar(sino_low, Float32)
    apply_cong!(sino_y, sino_c, sino_low, sino_high; basis = basis, water_basis = water_basis)
    (sino_y, sino_c)
end

export apply_cong!, apply_cong
