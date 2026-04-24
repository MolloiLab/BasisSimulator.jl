"""
Cong 2022 per-ray analytic dual-energy basis decomposition.

Given measured line integrals at low and high kVp, invert the
polychromatic forward model **ray by ray** to recover the photoelectric
(`y = ∫a(r)dr`) and Compton (`C = ∫c(r)dr`) basis sinograms.  Zero
calibration required — only the resolved x-ray spectra
(`resolve_source_spectrum_without_bowtie`) and the physical `p(ε), q(ε)` basis (see `basis.jl`).

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
    apply_cong!(sino_y, sino_c, sino_low, sino_high; basis, water_basis, ...)

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
- `basis`       : NamedTuple from `compute_photo_compton_basis` (or a
  direct material basis).  `basis.ŵ_L` / `basis.ŵ_H` may be 1D
  `[n_E]` (centered spectrum, legacy) or 3D `[n_col, n_row, n_E]`
  for per-ray bowtie-aware inversion — both paths run the same
  algorithm, only the spectral lookup differs.
- `water_basis` : NamedTuple from `water_basis_constants()`

# Tuning knobs (per-ray solver tolerances; exposed for residual beam-hardening tuning)
- `newton_max_iter::Int = 12` — inner Newton iterations on the Eq 8
  quintic.  Raise for dense iodine rays where Newton bails before the
  root is tight (e.g. try 20 if residual 40 keV iodine HU slope > 1.03).
- `newton_tol::Real = eps(Float32) ≈ 1.2e-7` — |Δ| break tolerance for
  the Newton inner loop.  Loosen (e.g. 5e-7) for noise-limited rays if
  Newton is spinning; tighten for precision tuning.
- `y_max_factor::Real = 0.99` — safety factor on the outer Brent upper
  bound `y_max = factor · p_L_meas / min(p_L)`.  K-edge basis functions
  have very low `min(p_L)` near the iodine K-edge → widening this
  factor (e.g. 2.0) helps dense-iodine rays converge.
- `y_max_cap::Real = 1e7` — hard ceiling on `y_max`; only relevant if
  the ratio above overflows for air-adjacent rays.
"""
function apply_cong!(
        sino_y::AbstractArray{Float32, 3},
        sino_c::AbstractArray{Float32, 3},
        sino_low::AbstractArray{Float32, 3},
        sino_high::AbstractArray{Float32, 3};
        basis,
        water_basis,
        newton_max_iter::Int = 12,
        newton_tol::Real     = eps(Float32),
        y_max_factor::Real   = 0.99,
        y_max_cap::Real      = 1f7,
    )
    # Enforce Float32 basis + water_basis up-front.  The kernel runs
    # entirely in Float32 (Metal can't allocate Float64 arrays), so any
    # Float64 input would silently upcast inside the closure and crash
    # on GPU with a confusing "Metal does not support Float64" error
    # deep in the similar() call chain.  Fail loud here with the fix.
    for (name, arr) in ((:ŵ_L, basis.ŵ_L), (:p_L, basis.p_L), (:q_L, basis.q_L),
                        (:ŵ_H, basis.ŵ_H), (:p_H, basis.p_H), (:q_H, basis.q_H))
        eltype(arr) === Float32 || error(
            "apply_cong!: basis.$(name) has eltype $(eltype(arr)); Float32 required. " *
            "Call BS.compute_photo_compton_basis(...) to rebuild the basis " *
            "(it returns Float32 vectors), or convert your basis vectors manually."
        )
    end
    (water_basis.a isa Float32 && water_basis.c isa Float32) || error(
        "apply_cong!: water_basis has $(typeof(water_basis.a))/$(typeof(water_basis.c)); " *
        "Float32/Float32 required.  Call BS.water_basis_constants() (returns Float32)."
    )

    # `basis.ŵ_*` is either 1D (centered spectrum) or 3D
    # [n_col, n_row, n_E] for per-ray bowtie-aware inversion.  Both ŵ_L
    # and ŵ_H must share dimensionality.
    per_ray = ndims(basis.ŵ_L) == 3
    per_ray == (ndims(basis.ŵ_H) == 3) ||
        error("apply_cong!: basis.ŵ_L and basis.ŵ_H must share ndims (both 1D or both 3D).")

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
    nE_L = per_ray ? size(ŵ_L, 3) : length(ŵ_L)
    nE_H = per_ray ? size(ŵ_H, 3) : length(ŵ_H)

    a_w     = Float32(water_basis.a)
    c_w     = Float32(water_basis.c)
    p_L_min = Float32(minimum(basis.p_L))

    # Capture tuning knobs as plain locals so the AK closure can see them
    # on every backend (kwargs aren't visible inside the do-block).
    nm_iter = newton_max_iter
    n_tol   = Float32(newton_tol)
    y_fac   = Float32(y_max_factor)
    y_cap   = Float32(y_max_cap)

    # BS sinogram layout [n_col, n_row, n_view]; bowtie ŵ only varies in
    # (col, row) so we decode (col, row) from the linear kernel index.
    n_col = Int(size(sino_low, 1))
    n_row = Int(size(sino_low, 2))

    AK.foreachindex(sino_low) do idx
        i0  = idx - 1
        col = (i0 % n_col) + 1
        row = ((i0 ÷ n_col) % n_row) + 1

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
            if per_ray
                @inbounds for i in 1:nE_L
                    T += ŵ_L[col, row, i] * exp(-(p_L[i] * a_w + q_L[i] * c_w) * L)
                end
            else
                @inbounds for i in 1:nE_L
                    T += ŵ_L[i] * exp(-(p_L[i] * a_w + q_L[i] * c_w) * L)
                end
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

        y_max = min(y_fac * p_L_meas / max(p_L_min, eps(Float32)), y_cap)
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
            if per_ray
                @inbounds for i in 1:nE_L
                    q_i  = q_L[i]
                    q2   = q_i * q_i;  q3 = q2 * q_i
                    q4   = q3 * q_i;   q5 = q4 * q_i
                    wexp = ŵ_L[col, row, i] * exp(-p_L[i] * y - q_i * c̄_)
                    P0 += wexp
                    P1 -= wexp * q_i
                    P2 += wexp * q2 * 0.5f0
                    P3 -= wexp * q3 * (1f0 / 6f0)
                    P4 += wexp * q4 * (1f0 / 24f0)
                    P5 -= wexp * q5 * (1f0 / 120f0)
                end
            else
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
            end
            x = 0f0
            for _ in 1:nm_iter
                F  = (P0 - T_L_meas) +
                     x * (P1 + x * (P2 + x * (P3 + x * (P4 + x * P5))))
                dF = P1 + x * (2f0*P2 + x * (3f0*P3 + x * (4f0*P4 + x * 5f0*P5)))
                abs(dF) < 1f-30 && break
                Δ = F / dF
                x -= Δ
                abs(Δ) < n_tol && break
            end
            x
        end

        # ── Step 3 — Outer Brent on y via G(y) = T_H_pred − T_H_meas ──
        T_H_pred = function (y::Float32, C::Float32)
            T = 0f0
            if per_ray
                @inbounds for i in 1:nE_H
                    T += ŵ_H[col, row, i] * exp(-p_H[i] * y - q_H[i] * C)
                end
            else
                @inbounds for i in 1:nE_H
                    T += ŵ_H[i] * exp(-p_H[i] * y - q_H[i] * C)
                end
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
    apply_cong(sino_low, sino_high; basis, water_basis, ...) -> (sino_y, sino_c)

Allocating wrapper around `apply_cong!`.  Returns a fresh `(sino_y,
sino_c)` pair on the same backend as the inputs.  Forwards all
`apply_cong!` tuning kwargs (`newton_max_iter`, `newton_tol`,
`y_max_factor`, `y_max_cap`).
"""
function apply_cong(
        sino_low::AbstractArray,
        sino_high::AbstractArray;
        basis,
        water_basis,
        kwargs...,
    )
    sino_y = similar(sino_low, Float32)
    sino_c = similar(sino_low, Float32)
    apply_cong!(sino_y, sino_c, sino_low, sino_high;
                basis = basis, water_basis = water_basis, kwargs...)
    (sino_y, sino_c)
end

export apply_cong!, apply_cong
