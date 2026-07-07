"""
Photoelectric + Compton physical basis tables (Cong et al. 2022).

    μ(r, ε) = p(ε)·a(r) + q(ε)·c(r)

where `a = ρ·⟨Z⁴/A⟩` and `c = ρ·⟨Z/A⟩`.  `p(ε)` is the photoelectric
cross-section per electron (ε = E / m_e c²) and `q(ε)` is the
Klein-Nishina Compton cross-section per electron.  Both are universal
physical constants — no calibration.

Reference:
  Cong, De Man, Wang (2022) *J X-Ray Sci Technol* 30:725–736.
  DOI 10.3233/XST-221153, Eqs 3a–3e, 4.
"""

# Physical constants
const _COMPTON_N_A        = 6.02214076e23      # Avogadro's, 1/mol
const _COMPTON_α_FS       = 7.2973525693e-3    # fine-structure constant
const _COMPTON_R_E_CM     = 2.8179403262e-13   # classical electron radius, cm
const _COMPTON_M_E_C2_KEV = 510.99895          # electron rest energy, keV

"""
    p_photoelectric(E_keV)

Photoelectric cross-section per electron at energy `E_keV` (Cong Eq 3c).
"""
function p_photoelectric(E_keV::Real)
    ε = E_keV / _COMPTON_M_E_C2_KEV
    _COMPTON_N_A * _COMPTON_α_FS^4 * (8 / 3) * π *
        _COMPTON_R_E_CM^2 * sqrt(32 / ε^7)
end

"""
    q_compton(E_keV)

Klein-Nishina Compton cross-section per electron at energy `E_keV`
(Cong Eq 3d; f_kn from Eq 3e).
"""
function q_compton(E_keV::Real)
    ε = E_keV / _COMPTON_M_E_C2_KEV
    A_ = (1 + ε) / ε^2
    B_ = 2 * (1 + ε) / (1 + 2ε)
    C_ = (1 / ε) * log(1 + 2ε)
    D_ = (1 / (2ε)) * log(1 + 2ε)
    E_ = (1 + 3ε) / (1 + 2ε)^2
    _COMPTON_N_A * 2π * _COMPTON_R_E_CM^2 * (A_ * (B_ - C_) + D_ - E_)
end

"""
    water_basis_constants()

Water basis constants `(a_water, c_water)` for Cong's initial L estimate.
Derived from the H₂O composition (ρ = 1 g/cm³).  This is NOT a calibration
— it's a physical constant, identical across all scanners.

Returned as `Float32` scalars so the downstream kernels (GPU-compatible)
run natively without Float64 fallback on Metal / CUDA.
"""
function water_basis_constants()
    ρ_w = 1.0
    Z_H, A_H = 1.0, 1.008
    Z_O, A_O = 8.0, 15.999
    M_H2O = 2 * A_H + A_O
    m_H = 2 * A_H / M_H2O
    m_O = A_O / M_H2O

    a_w = ρ_w * (m_H * Z_H^4 / A_H + m_O * Z_O^4 / A_O)
    c_w = ρ_w * (m_H * Z_H / A_H + m_O * Z_O / A_O)
    (a = Float32(a_w), c = Float32(c_w))
end

"""
    compute_photo_compton_basis(prot_low, prot_high; sim_opts, scanner)

Build the per-spectral-bin photoelectric + Compton basis tables for a
dual-kVp protocol pair.

Returns a NamedTuple of **Float32** vectors so the downstream `apply_cong!`
kernel runs natively on Metal / CUDA without a Float64 fallback:

- `ŵ_L, ŵ_H`  : normalized spectral weights (sum to 1) at low / high kVp
- `p_L, p_H`  : `p(ε)` at each bin of the resolved spectrum
- `q_L, q_H`  : `q(ε)` at each bin of the resolved spectrum

These tables are the only inputs Cong's per-ray decomposition and the
downstream PWLS / ACNR / VMI stages need to know about the physics of
the scan.  Float32 precision (~7 decimal digits) is more than enough for
the physical line-integral ranges (0–10 cm·g/cm²) that Cong operates on.
"""
function compute_photo_compton_basis(prot_low, prot_high; sim_opts, scanner)
    Base.depwarn("photo/Compton basis is DEPRECATED: the LSQ fit fails on the iodine K-edge (body-envelope noise). Use material-direct (murho_iodine, murho_water) with water_basis=(0,1).", :compute_photo_compton_basis)
    e_L, w_L = resolve_source_spectrum_without_bowtie(sim_opts, prot_low;  scanner = scanner)
    e_H, w_H = resolve_source_spectrum_without_bowtie(sim_opts, prot_high; scanner = scanner)

    ŵ_L = Float32.(Float64.(w_L) ./ sum(Float64.(w_L)))
    ŵ_H = Float32.(Float64.(w_H) ./ sum(Float64.(w_H)))

    p_L = Float32[Float32(p_photoelectric(Float64(e))) for e in e_L]
    q_L = Float32[Float32(q_compton(Float64(e)))       for e in e_L]
    p_H = Float32[Float32(p_photoelectric(Float64(e))) for e in e_H]
    q_H = Float32[Float32(q_compton(Float64(e)))       for e in e_H]

    (ŵ_L = ŵ_L, p_L = p_L, q_L = q_L,
     ŵ_H = ŵ_H, p_H = p_H, q_H = q_H)
end

export p_photoelectric, q_compton, water_basis_constants,
       compute_photo_compton_basis
