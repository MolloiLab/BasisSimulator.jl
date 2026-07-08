"""
Cong 2022 per-ray analytic dual-energy basis decomposition.

Given measured line integrals at low and high kVp, invert the
polychromatic forward model **ray by ray** to recover the photoelectric
(`y = ∫a(r)dr`) and Compton (`C = ∫c(r)dr`) basis sinograms.  Zero
calibration required — only the resolved x-ray spectra and a physical
`p(ε), q(ε)` basis (see `basis.jl`).

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
`Brent()` that runs inside GPU kernels without allocations.

# Workspace

The kernel writes per-pixel directly to the user-provided `sino_y` and
`sino_c` outputs — there is no per-tile sino-shape scratch.  The
workspace owns the **staged spectral basis** (six backend arrays
matching the sinogram's backend) so repeated calls don't re-stage CPU
basis tables to GPU each time.  No `tile_size`, no auto-tiling.

Reference:
  Cong, De Man, Wang (2022) *"Projection decomposition via univariate
  optimization for dual-energy CT."*  *J X-Ray Sci Technol* 30:725–736.
  DOI 10.3233/XST-221153.
"""

# ─────────────────────────────────────────────────────────────────────
# Backend helper — copy a CPU array onto whatever device `ref` lives on.
# Already used by other VMI algorithms; defined once here, reused.

@inline function _match_backend(arr::AbstractArray, ref::AbstractArray)
    ref isa Array && return arr
    out = similar(ref, eltype(arr), size(arr))
    copyto!(out, arr)
    out
end

# ════════════════════════════════════════════════════════════════════════
#  Workspace
# ════════════════════════════════════════════════════════════════════════

"""
    CongWorkspace{T, AŴ_L, AŴ_H, A1}

Pre-staged spectral basis on the sinogram's backend.  Cong is per-pixel
parallel and writes directly to the user's outputs, so the workspace has
no sino-shape scratch — only the six basis tables.

Type parameters:
  • `T`     — element type (Float32 enforced).
  • `AŴ_L`  — type of staged `ŵ_L` (1D `Vector{Float32}` for centered
              spectrum, 3D `[n_col,n_row,n_E]` for per-ray bowtie).
  • `AŴ_H`  — same for `ŵ_H` (must share dimensionality with `AŴ_L`).
  • `A1`    — type of staged `p_L`/`q_L`/`p_H`/`q_H` (always 1D Vector).

Construct with [`create_cong_workspace`](@ref).
"""
struct CongWorkspace{T <: AbstractFloat,
                      AŴ_L <: AbstractArray{T},
                      AŴ_H <: AbstractArray{T},
                      A1   <: AbstractVector{T}}
    ŵ_L::AŴ_L; p_L::A1; q_L::A1
    ŵ_H::AŴ_H; p_H::A1; q_H::A1
end

"""
    create_cong_workspace(sino_template, basis) -> CongWorkspace

Stage the Cong basis (`ŵ_L`, `p_L`, `q_L`, `ŵ_H`, `p_H`, `q_H`) onto the
same backend as `sino_template`.

Cong is per-pixel parallel with no per-tile scratch, so this workspace's
sole purpose is amortizing the basis-staging cost across repeated
`apply_cong!` calls (e.g., hyperparameter sweeps, multiple scans sharing
the same basis).

# Arguments
- `sino_template` : witness for the target backend.
- `basis`         : NamedTuple with `ŵ_L`, `p_L`, `q_L`, `ŵ_H`, `p_H`,
  `q_H`, all `Float32` — built by your basis-construction cell (see
  notebook 06 / 07 for the canonical setup).

`ŵ_L` / `ŵ_H` may be 1D `[n_E]` (centered spectrum) or 3D
`[n_col, n_row, n_E]` (per-ray bowtie).  Both must share dimensionality.
"""
function create_cong_workspace(
        sino_template::AbstractArray{<:AbstractFloat, 3},
        basis,
    )
    for (name, arr) in ((:ŵ_L, basis.ŵ_L), (:p_L, basis.p_L), (:q_L, basis.q_L),
                        (:ŵ_H, basis.ŵ_H), (:p_H, basis.p_H), (:q_H, basis.q_H))
        eltype(arr) === Float32 ||
            error("create_cong_workspace: basis.$(name) has eltype $(eltype(arr)); Float32 required.")
    end
    ndims(basis.ŵ_L) == ndims(basis.ŵ_H) ||
        error("create_cong_workspace: basis.ŵ_L and basis.ŵ_H must share ndims.")

    ŵ_L = _match_backend(basis.ŵ_L, sino_template)
    p_L = _match_backend(basis.p_L, sino_template)
    q_L = _match_backend(basis.q_L, sino_template)
    ŵ_H = _match_backend(basis.ŵ_H, sino_template)
    p_H = _match_backend(basis.p_H, sino_template)
    q_H = _match_backend(basis.q_H, sino_template)

    CongWorkspace{Float32, typeof(ŵ_L), typeof(ŵ_H), typeof(p_L)}(
        ŵ_L, p_L, q_L, ŵ_H, p_H, q_H,
    )
end

# ════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════

"""
    apply_cong!(ws, sino_y, sino_c, sino_low, sino_high;
                water_basis,
                newton_max_iter = 12,
                newton_tol      = eps(Float32),
                y_max_factor    = 0.99,
                y_max_cap       = 1f7) -> (sino_y, sino_c)

Per-ray Cong 2022 decomposition, writing results into `sino_y` and
`sino_c`.  All four sino arrays share shape `(nx, nv, nr)` and `Float32`.

# Arguments
- `ws`         : `CongWorkspace` from `create_cong_workspace(...)`.
  REQUIRED — no internal allocation path.
- `sino_y`,
  `sino_c`     : output photoelectric and Compton line integrals,
  mutated in place.
- `sino_low`,
  `sino_high`  : measured low- / high-kVp sinograms (log line integrals).

# Keyword arguments
- `water_basis`        : NamedTuple from `water_basis_constants()`.
- `newton_max_iter`    : inner Newton iterations on the Eq 8 quintic.
- `newton_tol`         : `|Δ|` break tolerance for the Newton inner loop.
- `y_max_factor`       : safety factor on the outer Brent upper bound
  `y_max = factor · p_L_meas / min(p_L)`.
- `y_max_cap`          : hard ceiling on `y_max`.
"""
function apply_cong!(
        ws::CongWorkspace,
        sino_y::AbstractArray{Float32, 3},
        sino_c::AbstractArray{Float32, 3},
        sino_low::AbstractArray{Float32, 3},
        sino_high::AbstractArray{Float32, 3};
        water_basis,
        newton_max_iter::Int = 12,
        newton_tol::Real     = eps(Float32),
        y_max_factor::Real   = 0.99,
        y_max_cap::Real      = 1f7,
    )
    (water_basis.a isa Float32 && water_basis.c isa Float32) || error(
        "apply_cong!: water_basis has $(typeof(water_basis.a))/$(typeof(water_basis.c)); " *
        "Float32/Float32 required.  Call BS.water_basis_constants() (returns Float32)."
    )
    size(sino_y) == size(sino_c) == size(sino_low) == size(sino_high) ||
        error("apply_cong!: sino_y / sino_c / sino_low / sino_high must share shape.")

    ŵ_L = ws.ŵ_L;  p_L = ws.p_L;  q_L = ws.q_L
    ŵ_H = ws.ŵ_H;  p_H = ws.p_H;  q_H = ws.q_H
    per_ray = ndims(ŵ_L) == 3
    nE_L = per_ray ? size(ŵ_L, 3) : length(ŵ_L)
    nE_H = per_ray ? size(ŵ_H, 3) : length(ŵ_H)

    a_w     = Float32(water_basis.a)
    c_w     = Float32(water_basis.c)
    p_L_min = Float32(minimum(p_L))
    # softest per-cm water attenuation over the low spectrum — used for the
    # adaptive Brent bracket (audit B3)
    μ_w_min = Float32(minimum(Float32.(Array(p_L)) .* a_w .+ Float32.(Array(q_L)) .* c_w))

    # Capture tuning knobs as locals so the AK closure sees them on every backend.
    nm_iter = newton_max_iter
    n_tol   = Float32(newton_tol)
    y_fac   = Float32(y_max_factor)
    y_cap   = Float32(y_max_cap)

    n_col = Int(size(sino_low, 1))
    n_row = Int(size(sino_low, 2))

    AK.foreachindex(sino_low) do idx
        i0  = idx - 1
        col = (i0 % n_col) + 1
        row = ((i0 ÷ n_col) % n_row) + 1

        p_L_meas = Float32(sino_low[idx])
        p_H_meas = Float32(sino_high[idx])

        # Symmetric air gate: |p| small in BOTH channels → air.  (The old
        # one-sided `p < 1e-6` gate rectified zero-mean air noise into a
        # positive-mean basis bias — audit finding B4.)
        if abs(p_L_meas) < 5f-3 && abs(p_H_meas) < 5f-3
            sino_y[idx] = 0f0
            sino_c[idx] = 0f0
            return
        end
        T_L_meas = exp(-p_L_meas)
        T_H_meas = exp(-p_H_meas)

        # Step 1 — Brent on water-equivalent path L (Cong Eq 7).
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

        # Adaptive upper bracket (audit B3): the old hard 60 cm cap zeroed
        # BOTH bases on dense rays (p_L/μ_min > 60) → paired dropout streaks.
        # Bound L by the measured low-channel integral over the softest μ in
        # the spectrum, with margin; floor at 60 cm for the water-only regime.
        L_hi = max(60f0, 1.5f0 * p_L_meas / max(μ_w_min, 1f-4))
        L_water, ok_L = brent_solve(water_T_L, -1f0, L_hi)
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

        # Step 2 — Newton on Eq 8 quintic for x = h(y).
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

        # Step 3 — Outer Brent on y via G(y) = T_H_pred − T_H_meas.
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

        # Root-bracket selection.  G(y) is INCREASING in y (the iodine-for-
        # water substitution rate p̄_L/q̄_L exceeds p̄_H/q̄_H), and the quintic
        # x(y) is a LOCAL Taylor approximation — far from the anchor its
        # Newton diverges and G returns garbage/NaN (the legacy y_max ≈
        # p_L/p_L_min is an absurd ~3 g/cm² of iodine).  So: expand the upper
        # bracket geometrically from a small start, never evaluating G
        # outside its validity; open only a NOISE-scale negative window when
        # G(0) > 0 proves the root sits below zero.
        G0 = G(0f0)
        y_opt, ok_y = if G0 == 0f0
            (0f0, true)
        elseif G0 < 0f0
            # positive-iodine root: double y_hi until G flips sign (finite)
            y_hi = min(0.01f0, y_max)
            G_hi = G(y_hi)
            n_exp = Int32(0)
            while isfinite(G_hi) && G_hi < 0f0 && y_hi < y_max && n_exp < Int32(24)
                y_hi = min(y_hi * 2f0, y_max)
                G_hi = G(y_hi)
                n_exp += Int32(1)
            end
            if isfinite(G_hi) && G_hi >= 0f0
                brent_solve(G, 0f0, y_hi)
            else
                (0f0, false)                     # no valid root → water-only fallback
            end
        else
            # G0 > 0: zero-mean noise excursion → root slightly below 0.
            # Window at the noise scale only — far-negative y is outside the
            # quintic's validity and hosts spurious roots.
            Glo = G(-0.1f0)
            if isfinite(Glo) && Glo < 0f0
                brent_solve(G, -0.1f0, 0f0)
            else
                (0f0, true)
            end
        end
        if !ok_y
            sino_y[idx] = a_w * L_water
            sino_c[idx] = c_w * L_water
            return
        end
        x_final = solve_quintic(y_opt, c̄)
        # Audit B4: do NOT rectify noise — small negative basis line
        # integrals are legitimate zero-mean noise excursions; clamping them
        # to 0 creates a positive-mean bias that ramp-filtering spreads into
        # the body as a keV-dependent HU offset (previously masked by
        # pcct_noise_reduction).  Clamp only at absurd magnitudes.
        sino_y[idx] = clamp(y_opt, -5f0, 1f4)
        sino_c[idx] = clamp(c̄ + x_final, -5f0, 1f4)
    end

    (sino_y, sino_c)
end

export CongWorkspace, create_cong_workspace, apply_cong!
