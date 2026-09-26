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

export p_photoelectric, q_compton
