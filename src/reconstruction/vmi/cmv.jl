"""
CMV (Constant Material Value / linear DE) decomposition.

Spectrum-effective 2×2 mass-atten matrix inversion per ray.  Zero
iteration — a closed-form linear solve on the effective matrix

    [μ̄_W_L  μ̄_I_L]     with  μ̄_m_k = ⟨ŵ_k(ε), μρ_m(ε)⟩
    [μ̄_W_H  μ̄_I_H]

for each `(s_L, s_H)` pair.  Ignores beam hardening (the polychromatic
spread collapses into a single scalar per kVp), so expect residual
cupping — but it's cheap, reliable, and makes an excellent warm start
for Cong/PWLS/RWLS.

When `basis.ŵ_*` is 3D (per-ray bowtie), the effective matrix is
computed from the centered ray only; per-ray bowtie awareness is
properly handled by Cong.
"""

"""
    apply_cmv!(sino_iodine, sino_water, sino_low, sino_high; basis) -> (sino_iodine, sino_water)

In-place CMV linear DE decomposition.  All four arrays share shape
`(n_col, n_row, n_view)` and `Float32` element type.  Runs on whatever
backend the sinograms live on via `AK.foreachindex`.

# Arguments
- `sino_iodine`, `sino_water`  : output material line integrals (g/cm²)
- `sino_low`, `sino_high`      : measured low-/high-kVp log line integrals

# Keyword arguments
- `basis` : NamedTuple exposing `ŵ_L`, `ŵ_H` (1D or 3D Float32 spectral
  weights), `p_L`, `p_H` (iodine μρ, cm²/g), `q_L`, `q_H` (water μρ).
  3D `ŵ` is collapsed to its centered ray for the effective matrix.
"""
function apply_cmv!(
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        sino_low::AbstractArray{Float32, 3},
        sino_high::AbstractArray{Float32, 3};
        basis,
    )
    # Collapse ŵ to a centered 1D spectrum if bowtie is on; the 2×2
    # effective-matrix inversion has no room for per-ray spectra anyway.
    ŵ_L_raw = basis.ŵ_L
    ŵ_H_raw = basis.ŵ_H
    ŵ_L_1d, ŵ_H_1d = if ndims(ŵ_L_raw) == 3
        nc = size(ŵ_L_raw, 1); nr = size(ŵ_L_raw, 2)
        mc = nc ÷ 2 + 1;       mr = nr ÷ 2 + 1
        (Float64.(Array(ŵ_L_raw)[mc, mr, :]),
         Float64.(Array(ŵ_H_raw)[mc, mr, :]))
    else
        (Float64.(Array(ŵ_L_raw)), Float64.(Array(ŵ_H_raw)))
    end
    ŵ_L_1d ./= sum(ŵ_L_1d)
    ŵ_H_1d ./= sum(ŵ_H_1d)

    p_L_vec = Float64.(Array(basis.p_L))
    q_L_vec = Float64.(Array(basis.q_L))
    p_H_vec = Float64.(Array(basis.p_H))
    q_H_vec = Float64.(Array(basis.q_H))

    μ̄_I_L = sum(ŵ_L_1d .* p_L_vec)
    μ̄_W_L = sum(ŵ_L_1d .* q_L_vec)
    μ̄_I_H = sum(ŵ_H_1d .* p_H_vec)
    μ̄_W_H = sum(ŵ_H_1d .* q_H_vec)

    det_M = μ̄_W_L * μ̄_I_H - μ̄_I_L * μ̄_W_H
    abs(det_M) < eps(Float64) * 1e3 && error(
        "apply_cmv!: 2×2 effective matrix is singular (det = $det_M). " *
        "Check that 80/140 kVp spectra differ and de_basis is built correctly."
    )

    inv_det  = Float32(1.0 / det_M)
    μ̄_I_L32 = Float32(μ̄_I_L)
    μ̄_I_H32 = Float32(μ̄_I_H)
    μ̄_W_L32 = Float32(μ̄_W_L)
    μ̄_W_H32 = Float32(μ̄_W_H)

    AK.foreachindex(sino_low) do idx
        l = Float32(sino_low[idx])
        h = Float32(sino_high[idx])
        a_w = inv_det * ( μ̄_I_H32 * l - μ̄_I_L32 * h)
        a_i = inv_det * (-μ̄_W_H32 * l + μ̄_W_L32 * h)
        sino_water[idx]  = ifelse(a_w > 0f0, a_w, 0f0)
        sino_iodine[idx] = ifelse(a_i > 0f0, a_i, 0f0)
    end

    (sino_iodine, sino_water)
end

"""
    apply_cmv(sino_low, sino_high; basis) -> (sino_iodine, sino_water)

Allocating wrapper around `apply_cmv!`.  Returns a fresh
`(sino_iodine, sino_water)` pair on the same backend as the inputs.
"""
function apply_cmv(
        sino_low::AbstractArray,
        sino_high::AbstractArray;
        basis,
    )
    sino_iodine = similar(sino_low, Float32)
    sino_water  = similar(sino_low, Float32)
    apply_cmv!(sino_iodine, sino_water, sino_low, sino_high; basis = basis)
    (sino_iodine, sino_water)
end

export apply_cmv!, apply_cmv
