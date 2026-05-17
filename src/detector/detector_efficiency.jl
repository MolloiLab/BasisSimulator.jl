"""
    Forward/DetectorEfficiency.jl

Detector absorption efficiency for CT simulation.

# What's here

Two paths, both routed by `DetectorEfficiencyMode`:

1. **`MC_LUT`** — Monte Carlo-derived per-energy efficiency LUT for the GE
   Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ scintillator (`GEMSTONE_MC_EFFICIENCY_LUT`).
   Captures fluorescence escape at the Tb (52 keV) and Lu (63 keV) K-edges,
   which Beer-Lambert cannot model — at those energies, real η DROPS rather
   than rising as photons in the fluorescent shell escape the crystal.

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
    d_cm = model.thickness_mm / 10.0
    return [1.0 - exp(-get_scintillator_mu(model.material, Float64(E)) * d_cm) for E in energies]
end

# =============================================================================
# Exports
# =============================================================================

export DetectorEfficiency, DetectorEfficiencyMode, BEER_LAMBERT, MC_LUT
export detector_efficiency_gemstone
export get_scintillator_mu, get_gemstone_mc_efficiency, compute_eid_efficiency_vector
export GEMSTONE_MC_EFFICIENCY_LUT
