"""
    Forward/DetectorEfficiency.jl

Detector absorption efficiency for CT simulation.

# What's here

Two paths, both routed by `DetectorEfficiencyMode`:

1. **`MC_LUT`** — Monte Carlo-derived per-energy efficiency LUTs for the
   supported EICT scintillators:
   - GE Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ (`GEMSTONE_MC_EFFICIENCY_LUT`) —
     fluorescence escape at the Tb (52 keV) and Lu (63 keV) K-edges.
   - Siemens UFC Gd₂O₂S:Pr,Ce (`UFC_MC_EFFICIENCY_LUT`, SOMATOM Force
     StellarInfinity) — fluorescence escape at the Gd K-edge (50.24 keV).
   At those K-edges real η DROPS rather than rising as photons in the
   fluorescent shell escape the crystal — Beer-Lambert cannot model this.

2. **`BEER_LAMBERT`** — analytical fallback `η(E) = 1 − exp(−μ(E) × d / cos θ)`
   using `get_scintillator_mu`. Reachable via `sim_opts.detector_efficiency_mode
   = :beer_lambert` for verification against the MC LUT (PR #5 introduced this
   toggle precisely for that comparison).

The PCCT path (`photon_counting.jl`) also consumes `get_scintillator_mu` to
look up CdTe μ(E) for `quantum_efficiency(material, thickness, E)` — so the
`SCINTILLATOR_MU_DATA` dictionary keeps its `"CdTe"` and `"Gemstone"` entries.

# References

1. PR #5 — Gemstone MC LUT + `detector_efficiency_mode` toggle
2. NIST XCOM database for scintillator μ(E)
"""

import AcceleratedKernels as AK

# =============================================================================
# Scintillator μ(E) data (NIST XCOM)
# =============================================================================
# Only the two materials that are actually used live in the package:
# - "CdTe"     → PCCT crystal η(E) lookup via get_detector_material_attenuation
# - "Gemstone" → EICT Beer-Lambert fallback when mode=:beer_lambert (PR #5)
#
# The MC path for Gemstone bypasses this dictionary entirely — it uses
# GEMSTONE_MC_EFFICIENCY_LUT below.
# =============================================================================

const SCINTILLATOR_MU_DATA = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}(
    # Cadmium Telluride (CdTe) — direct conversion PCCT (NAEOTOM Alpha)
    # ρ = 5.85 g/cm³, K-edges: Cd at 26.7 keV, Te at 31.8 keV
    "CdTe" => (
        [20.0, 26.7, 31.8, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [95.0, 180.0, 200.0, 100.0, 55.0, 33.0, 14.5, 7.8, 4.9, 2.8]
    ),

    # Cadmium Zinc Telluride (CZT) — PCCT alt sensor (no scanner uses it live,
    # but the CZT_MATERIAL enum + Scanner :czt vocab is public; the table is
    # kept so quantum_efficiency(CZT_MATERIAL, ...) returns the right μ).
    # ρ ≈ 5.78 g/cm³, K-edges: Cd 26.7 keV, Te 31.8 keV (similar to CdTe).
    "CZT" => (
        [20.0, 26.7, 31.8, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [90.0, 170.0, 190.0, 95.0, 52.0, 31.0, 13.5, 7.3, 4.6, 2.6]
    ),

    # Silicon (Si) — PCCT alt sensor (low-Z, very transparent at CT energies).
    # Kept for symmetry with CZT_MATERIAL / SI_MATERIAL enum support.
    # ρ = 2.33 g/cm³
    "Si" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [2.8, 0.95, 0.58, 0.48, 0.46, 0.43, 0.40, 0.39, 0.37]
    ),

    # Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ — GE proprietary garnet scintillator
    # ρ = 7.0 g/cm³, K-edges: Tb at 52.0 keV, Lu at 63.31 keV
    # μ(E) = (μ/ρ)(E) × ρ, with (μ/ρ) from NIST XCOM (mixture, coherent incl.)
    "Gemstone" => (
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
            51.0, 51.99, 52.01, 53.0, 55.0, 57.0, 59.0, 60.0,
            61.0, 63.0, 63.30, 63.32, 65.0, 67.0, 69.0,
            71.0, 73.0, 75.0, 77.0, 80.0, 85.0,
            91.0, 95.0, 100.0, 105.0, 110.0, 115.0, 121.0,
            125.0, 131.0, 135.0, 141.0, 150.0],
        [1098.3, 486.2, 228.3, 126.4, 78.3, 52.1, 36.8, 27.3, 20.8,
            19.8, 18.9, 47.2, 44.9, 40.8, 37.2, 34.1, 32.7,
            31.3, 28.8, 28.5, 51.0, 47.7, 44.2, 41.0,
            38.1, 35.5, 33.2, 31.0, 28.1, 24.1,
            20.3, 18.2, 16.0, 14.1, 12.3, 11.2, 9.9,
            9.1, 8.1, 7.6, 6.8, 5.9]
    )
)

# Gemstone aliases
SCINTILLATOR_MU_DATA["LUMEX"]      = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["lumex"]      = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["Garnet"]     = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["TbLuAG"]     = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["TbLuAG:Ce"]  = SCINTILLATOR_MU_DATA["Gemstone"]

# UFC Gd₂O₂S:Pr,Ce — Siemens "Ultra-Fast Ceramic" scintillator
# (SOMATOM Force StellarInfinity).  μ(E) = (μ/ρ)(E) × ρ evaluated from the
# module's GD2O2S compound (ρ = 7.44 g/cm³, NIST XCOM via XrayAttenuation);
# Gd K-edge discontinuity bracketed at 50.23 / 50.25 keV (K-edge 50.24 keV).
# Used only by the :beer_lambert verification mode — the live UFC path is
# UFC_MC_EFFICIENCY_LUT below.
SCINTILLATOR_MU_DATA["UFC"] = (
    [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
        50.23, 50.25, 52.0, 55.0, 60.0, 65.0, 70.0, 75.0,
        80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    [1700.9, 588.5, 274.3, 151.3, 93.2, 61.9, 43.6, 32.0, 24.4,
        24.1, 115.6, 105.8, 91.5, 73.0, 59.4, 49.0, 41.0,
        34.7, 29.7, 25.6, 19.4, 15.2, 12.2, 9.9, 8.3, 7.0]
)

# UFC aliases
SCINTILLATOR_MU_DATA["ufc"]    = SCINTILLATOR_MU_DATA["UFC"]
SCINTILLATOR_MU_DATA["Gd2O2S"] = SCINTILLATOR_MU_DATA["UFC"]
SCINTILLATOR_MU_DATA["GOS"]    = SCINTILLATOR_MU_DATA["UFC"]

# =============================================================================
# Monte Carlo-Derived Detector Efficiency LUT
# =============================================================================

"""
    GEMSTONE_MC_EFFICIENCY_LUT

Monte Carlo-derived detector efficiency lookup table for Ce:(Tb,Lu)₃Al₅O₁₂
(Gemstone-type garnet scintillator) as a function of photon energy.

Computed from a full MCNP simulation of the GE Revolution Apex detector with
composition Tb(0.287) / Lu(0.316) / Al(0.163) / O(0.231) / Ce(0.002).

# Key features captured by MC (not in Beer-Lambert)

1. **Fluorescence escape at Tb K-edge (52 keV)**: η drops 0.956 → 0.807 due
   to Tb Kα fluorescence photons (~44 keV) escaping the detector volume.
2. **Fluorescence escape at Lu K-edge (63 keV)**: η drops 0.846 → 0.762 due
   to Lu Kα fluorescence photons (~54 keV) escaping.
3. **L-edge effects** at 8-12 keV from Tb/Lu L-shells.
4. **Compton scattering escape** at higher energies.

Beer-Lambert predicts η INCREASES at K-edges (more absorption); MC correctly
shows DECREASES because fluorescent photons escape, reducing deposited energy.

# Reference
- Material: PMC10179960 (Ce:(Tb,Lu)₃Al₅O₁₂ garnet scintillator)
- Simulation: MCNP with ENDF photon cross-sections
- Origin: PR #5
"""
const GEMSTONE_MC_EFFICIENCY_LUT = (
    energies=Float64[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138, 139, 140
    ],
    efficiency=Float64[
        0.991, 0.990, 0.990, 0.990, 0.990, 0.989, 0.988, 0.989, 0.979, 0.983,
        0.973, 0.976, 0.979, 0.981, 0.982, 0.983, 0.984, 0.984, 0.985, 0.985,
        0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.984, 0.984, 0.984,
        0.983, 0.983, 0.982, 0.982, 0.981, 0.981, 0.980, 0.979, 0.978, 0.977,
        0.975, 0.975, 0.974, 0.972, 0.971, 0.970, 0.969, 0.967, 0.965, 0.962,
        0.960, 0.956, 0.807, 0.813, 0.818, 0.823, 0.827, 0.831, 0.834, 0.838,
        0.841, 0.844, 0.846, 0.762, 0.767, 0.773, 0.777, 0.783, 0.787, 0.791,
        0.796, 0.800, 0.803, 0.809, 0.813, 0.817, 0.820, 0.823, 0.826, 0.829,
        0.832, 0.835, 0.837, 0.840, 0.842, 0.844, 0.846, 0.848, 0.850, 0.851,
        0.853, 0.854, 0.855, 0.856, 0.857, 0.858, 0.859, 0.859, 0.859, 0.859,
        0.859, 0.859, 0.858, 0.858, 0.857, 0.856, 0.855, 0.854, 0.853, 0.852,
        0.850, 0.848, 0.847, 0.845, 0.842, 0.840, 0.839, 0.836, 0.833, 0.830,
        0.828, 0.824, 0.821, 0.819, 0.815, 0.812, 0.808, 0.805, 0.801, 0.797,
        0.793, 0.789, 0.785, 0.781, 0.777, 0.773, 0.768, 0.763, 0.758, 0.754
    ]
)

"""
    UFC_MC_EFFICIENCY_LUT

Monte Carlo-derived detector efficiency lookup table for the Siemens UFC
(Ultra-Fast Ceramic, Gd₂O₂S:Pr,Ce) scintillator as a function of photon
energy — the Siemens SOMATOM Force sister of [`GEMSTONE_MC_EFFICIENCY_LUT`].

Computed from a full Monte-Carlo transport simulation of the SOMATOM Force
StellarInfinity detector by Hamidreza Khodajou-Chokami, PhD (UC Irvine
Medical Imaging Laboratory), `efficiency_results.csv`, 2026-06-08.  Values
are verbatim from that dataset on a 1-keV grid (1–140 keV).

# Key features captured by MC (not in Beer-Lambert)

1. **Fluorescence escape at the Gd K-edge (50.24 keV)**: η drops
   0.969 → 0.741 between 50 and 51 keV as Gd Kα fluorescence (~43 keV)
   escapes the crystal.  Beer-Lambert predicts the opposite jump.
2. **Gd L-edge structure** near 7–8 keV (L₃ 7.24 / L₂ 7.93 / L₁ 8.38 keV).
3. **High-energy roll-off** from primary transmission + Compton escape
   (0.897 at 100 keV → 0.816 at 140 keV).

First validated end-to-end (poly + dual-source VMI) in
`docs/notebooks/09_siemens_force_ufc_dual_source_vmi.jl`.
"""
const UFC_MC_EFFICIENCY_LUT = (
    energies=Float64[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138, 139, 140
    ],
    efficiency=Float64[
        9.90863305e-01, 9.90798712e-01, 9.90207366e-01, 9.89968109e-01, 9.89353993e-01,
        9.88877621e-01, 9.88764364e-01, 9.69360328e-01, 9.72729277e-01, 9.76089134e-01,
        9.78741846e-01, 9.80889490e-01, 9.82528622e-01, 9.83559147e-01, 9.84355716e-01,
        9.84992948e-01, 9.85387625e-01, 9.85860336e-01, 9.86036647e-01, 9.86192298e-01,
        9.86036424e-01, 9.86016013e-01, 9.85866718e-01, 9.85760036e-01, 9.85619041e-01,
        9.85427871e-01, 9.85144176e-01, 9.84997524e-01, 9.84686433e-01, 9.84226060e-01,
        9.83886432e-01, 9.83888151e-01, 9.83530746e-01, 9.82826388e-01, 9.82299763e-01,
        9.81754579e-01, 9.81145416e-01, 9.80406222e-01, 9.79366721e-01, 9.78673909e-01,
        9.78040413e-01, 9.77222704e-01, 9.75291071e-01, 9.74675907e-01, 9.73959516e-01,
        9.72894908e-01, 9.71301325e-01, 9.70168735e-01, 9.69081131e-01, 9.69026725e-01,
        7.41154644e-01, 7.47765810e-01, 7.54996954e-01, 7.61970239e-01, 7.68339996e-01,
        7.74502998e-01, 7.80637525e-01, 7.85478337e-01, 7.91350826e-01, 7.97260988e-01,
        8.01716927e-01, 8.06821449e-01, 8.12187403e-01, 8.16816937e-01, 8.21093164e-01,
        8.25750637e-01, 8.29601348e-01, 8.33439150e-01, 8.36683997e-01, 8.40730609e-01,
        8.44150931e-01, 8.47429296e-01, 8.50435158e-01, 8.53887225e-01, 8.56827475e-01,
        8.59963243e-01, 8.62840135e-01, 8.65738358e-01, 8.68314163e-01, 8.70655876e-01,
        8.72908866e-01, 8.74932562e-01, 8.77096478e-01, 8.78888967e-01, 8.80993084e-01,
        8.82948845e-01, 8.84599046e-01, 8.86221492e-01, 8.87676325e-01, 8.88783948e-01,
        8.90449254e-01, 8.91856769e-01, 8.92648690e-01, 8.93756963e-01, 8.94433775e-01,
        8.95239544e-01, 8.95655380e-01, 8.96253439e-01, 8.96718581e-01, 8.96705367e-01,
        8.96467303e-01, 8.96790377e-01, 8.96711472e-01, 8.96650987e-01, 8.96839385e-01,
        8.96598287e-01, 8.96387397e-01, 8.95719919e-01, 8.94503447e-01, 8.93412046e-01,
        8.92210714e-01, 8.90899489e-01, 8.89779601e-01, 8.88385073e-01, 8.87015853e-01,
        8.85130923e-01, 8.83793476e-01, 8.82173648e-01, 8.80395144e-01, 8.78167359e-01,
        8.76309315e-01, 8.73851763e-01, 8.71613318e-01, 8.69143288e-01, 8.66828531e-01,
        8.64327296e-01, 8.61391429e-01, 8.58218989e-01, 8.55394561e-01, 8.52652873e-01,
        8.49222638e-01, 8.45710682e-01, 8.41841634e-01, 8.37917905e-01, 8.34416213e-01,
        8.30333026e-01, 8.26984885e-01, 8.23573340e-01, 8.20082743e-01, 8.15971281e-01
    ]
)

# =============================================================================
# Detector Efficiency Types
# =============================================================================

"""
    DetectorEfficiencyMode

Selector for how detector efficiency is computed.

- `BEER_LAMBERT` — Analytical Beer-Lambert absorption law using μ(E) from XCOM
- `MC_LUT` — Monte Carlo-derived lookup table (captures fluorescence escape)
"""
@enum DetectorEfficiencyMode begin
    BEER_LAMBERT
    MC_LUT
end

"""
    DetectorEfficiency

Detector efficiency model specification.

# Fields
- `material`: Scintillator material name (Gemstone aliases, or "CdTe" via PCCT)
- `thickness_mm`: Scintillator thickness in mm
- `fill_factor`: Fraction of detector area that is active (0-1) — physical
  property stored here but applied separately via `FillFactorModel`
- `mode`: `BEER_LAMBERT` or `MC_LUT`

`compute_eid_efficiency_vector` computes scintillator absorption η(E) only;
fill factor is applied in the physics pipeline.
"""
struct DetectorEfficiency
    material::String
    thickness_mm::Float64
    fill_factor::Float64
    mode::DetectorEfficiencyMode
end

"""
    detector_efficiency_gemstone(; mode::Symbol=:mc_lut, thickness_mm=3.0,
                                  fill_factor=0.90)

GE Gemstone Clarity Ce:(Tb,Lu)₃Al₅O₁₂ garnet scintillator detector.

# Modes
- `:mc_lut` (default) — Monte Carlo-derived efficiency LUT from MCNP simulation.
  Captures fluorescence escape at Tb/Lu K-edges that Beer-Lambert cannot model.
- `:beer_lambert` — Analytical Beer-Lambert using XCOM-derived μ(E) data.
"""
function detector_efficiency_gemstone(; mode::Symbol=:mc_lut,
    thickness_mm::Float64=3.0,
    fill_factor::Float64=0.90)
    eff_mode = mode == :mc_lut ? MC_LUT : BEER_LAMBERT
    return DetectorEfficiency("Gemstone", thickness_mm, fill_factor, eff_mode)
end

"""
    detector_efficiency_ufc(; mode::Symbol=:mc_lut, thickness_mm=1.4,
                             fill_factor=0.90)

Siemens UFC (Ultra-Fast Ceramic) Gd₂O₂S:Pr,Ce scintillator detector
(SOMATOM Force StellarInfinity) — sister factory of
[`detector_efficiency_gemstone`](@ref).

# Modes
- `:mc_lut` (default) — Monte Carlo-derived efficiency LUT
  ([`UFC_MC_EFFICIENCY_LUT`]).  Captures Gd K-edge fluorescence escape
  that Beer-Lambert cannot model.
- `:beer_lambert` — Analytical Beer-Lambert using the Gd₂O₂S μ(E) table.

The 1.4 mm default thickness is a documented assumption — Siemens does not
publish the UFC layer depth (the value is inert in `:mc_lut` mode; with
μ ≈ 46 mm⁻¹ at 120 kVp mean energies, ≥1 mm already absorbs >99%).
"""
function detector_efficiency_ufc(; mode::Symbol=:mc_lut,
    thickness_mm::Float64=1.4,
    fill_factor::Float64=0.90)
    eff_mode = mode == :mc_lut ? MC_LUT : BEER_LAMBERT
    return DetectorEfficiency("UFC", thickness_mm, fill_factor, eff_mode)
end

# =============================================================================
# Scintillator Attenuation Lookup
# =============================================================================

"""
    get_scintillator_mu(material::String, energy_keV::Float64) -> Float64

Get linear attenuation coefficient (cm⁻¹) for scintillator material at given
energy.  Log-linear interpolation of tabulated NIST XCOM data;
K-edge discontinuities are captured by including data points at the edges.

Live callers:
- EICT Beer-Lambert efficiency via `compute_eid_efficiency_vector`
- PCCT crystal η(E) via `photon_counting.jl::get_detector_material_attenuation`
"""
function get_scintillator_mu(material::String, energy_keV::Float64)
    if !haskey(SCINTILLATOR_MU_DATA, material)
        @warn "Unknown scintillator material: $material, falling back to Gemstone"
        material = "Gemstone"
    end

    energies, mus = SCINTILLATOR_MU_DATA[material]
    E = clamp(energy_keV, energies[1], energies[end])

    log_E = log(E)
    log_energies = log.(energies)
    log_mus = log.(mus)

    idx = 1
    for i in 1:(length(energies)-1)
        if E >= energies[i] && E <= energies[i+1]
            idx = i
            break
        end
    end

    t = (log_E - log_energies[idx]) / (log_energies[idx+1] - log_energies[idx])
    log_mu = log_mus[idx] + t * (log_mus[idx+1] - log_mus[idx])

    return exp(log_mu)
end

"""
    get_gemstone_mc_efficiency(energy_keV::Float64) -> Float64

Look up Monte Carlo-derived detector efficiency for Ce:(Tb,Lu)₃Al₅O₁₂
Gemstone scintillator at the given photon energy.

Linear interpolation of the 1-140 keV MC efficiency table.
Energies outside the range are clamped.

# Key physics
- Tb K-edge fluorescence escape at 52-53 keV (η: 0.956 → 0.807)
- Lu K-edge fluorescence escape at 63-64 keV (η: 0.846 → 0.762)
- Peak efficiency ~0.985 at 20-27 keV
- Efficiency ~0.859 at 100 keV, declining to 0.754 at 140 keV
"""
function get_gemstone_mc_efficiency(energy_keV::Float64)
    lut = GEMSTONE_MC_EFFICIENCY_LUT
    E = clamp(energy_keV, lut.energies[1], lut.energies[end])

    if E == floor(E) && 1.0 <= E <= 140.0
        return lut.efficiency[Int(E)]
    end

    idx = 1
    for i in 1:(length(lut.energies)-1)
        if E >= lut.energies[i] && E <= lut.energies[i+1]
            idx = i
            break
        end
    end

    t = (E - lut.energies[idx]) / (lut.energies[idx+1] - lut.energies[idx])
    return lut.efficiency[idx] + t * (lut.efficiency[idx+1] - lut.efficiency[idx])
end

"""
    get_ufc_mc_efficiency(energy_keV::Float64) -> Float64

Look up Monte Carlo-derived detector efficiency for the Siemens UFC
Gd₂O₂S:Pr,Ce scintillator at the given photon energy.

Linear interpolation of the 1–140 keV MC efficiency table
([`UFC_MC_EFFICIENCY_LUT`]).  Energies outside the range are clamped.

# Key physics
- Gd K-edge fluorescence escape at 50–51 keV (η: 0.969 → 0.741)
- Gd L-edge dip near 8 keV
- Peak efficiency ~0.986 at ~20 keV
- η ≈ 0.897 at 100 keV, declining to 0.816 at 140 keV
"""
function get_ufc_mc_efficiency(energy_keV::Float64)
    lut = UFC_MC_EFFICIENCY_LUT
    E = clamp(energy_keV, lut.energies[1], lut.energies[end])

    if E == floor(E) && 1.0 <= E <= 140.0
        return lut.efficiency[Int(E)]
    end

    idx = clamp(floor(Int, E), 1, 139)
    t = E - lut.energies[idx]
    return lut.efficiency[idx] + t * (lut.efficiency[idx+1] - lut.efficiency[idx])
end

# =============================================================================
# EID Pipeline Efficiency Vector
# =============================================================================

"""
    compute_eid_efficiency_vector(model::DetectorEfficiency, energies) -> Vector{Float64}

Compute detector efficiency η(E) for each energy in the spectrum.

Routes through the Monte Carlo LUT for Gemstone-type scintillators (captures
fluorescence escape at K-edges) or Beer-Lambert for other materials. This
vector weights the polychromatic Beer-Lambert sum in the EID pipeline:
`I = Σ wₑ × η(E) × exp(-∫μₑ dl)`.
"""
function compute_eid_efficiency_vector(model::DetectorEfficiency, energies::AbstractVector)
    if model.mode == MC_LUT && model.material in ("Gemstone", "LUMEX", "lumex", "Garnet", "TbLuAG", "TbLuAG:Ce")
        return [get_gemstone_mc_efficiency(Float64(E)) for E in energies]
    end
    if model.mode == MC_LUT && model.material in ("UFC", "ufc", "Gd2O2S", "GOS")
        return [get_ufc_mc_efficiency(Float64(E)) for E in energies]
    end
    d_cm = model.thickness_mm / 10.0
    return [1.0 - exp(-get_scintillator_mu(model.material, Float64(E)) * d_cm) for E in energies]
end

# =============================================================================
# Exports
# =============================================================================

export DetectorEfficiency, DetectorEfficiencyMode, BEER_LAMBERT, MC_LUT
export detector_efficiency_gemstone, detector_efficiency_ufc
export get_scintillator_mu, get_gemstone_mc_efficiency, get_ufc_mc_efficiency
export compute_eid_efficiency_vector
export GEMSTONE_MC_EFFICIENCY_LUT, UFC_MC_EFFICIENCY_LUT
