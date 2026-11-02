# The energy-integrating detector's response: the first two moments of the signal one incident
# photon leaves in a dexel (`eid_signal_moments`), from each scintillator's Geant4 flat-field table
# (`MC_LUT`, eid_response_tables.jl) or analytic absorption (`BEER_LAMBERT`).
#
# The PCCT path (`photon_counting.jl`) also reads `get_scintillator_mu` for CdTe μ(E), so the
# `SCINTILLATOR_MU_DATA` dictionary keeps its `"CdTe"` entry.

import AcceleratedKernels as AK

# =============================================================================
# Scintillator μ(E) data (NIST XCOM)
# =============================================================================
# - "CdTe", "CZT", "Si" → PCCT sensor η(E) via get_detector_material_attenuation
# - "Gemstone", "UFC"   → the EICT BEER_LAMBERT mode, at the densities of the Geant4 tables
# =============================================================================

const SCINTILLATOR_MU_DATA = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}(
    # Cadmium Telluride (CdTe) — direct conversion PCCT (NAEOTOM Alpha)
    # ρ = 5.85 g/cm³, K-edges: Cd at 26.7 keV, Te at 31.8 keV
    "CdTe" => (
        [20.0, 26.7, 31.8, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [95.0, 180.0, 200.0, 100.0, 55.0, 33.0, 14.5, 7.8, 4.9, 2.8],
    ),

    # Cadmium Zinc Telluride (CZT) — PCCT alt sensor (no scanner uses it live,
    # but the CZT_MATERIAL enum + Scanner :czt vocab is public; the table is
    # kept so quantum_efficiency(CZT_MATERIAL, ...) returns the right μ).
    # ρ ≈ 5.78 g/cm³, K-edges: Cd 26.7 keV, Te 31.8 keV (similar to CdTe).
    "CZT" => (
        [20.0, 26.7, 31.8, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [90.0, 170.0, 190.0, 95.0, 52.0, 31.0, 13.5, 7.3, 4.6, 2.6],
    ),

    # Silicon (Si) — PCCT alt sensor (low-Z, very transparent at CT energies).
    # Kept for symmetry with CZT_MATERIAL / SI_MATERIAL enum support.
    # ρ = 2.33 g/cm³
    "Si" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [2.8, 0.95, 0.58, 0.48, 0.46, 0.43, 0.4, 0.39, 0.37],
    ),

    # Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ — GE proprietary garnet scintillator
    # K-edges: Tb at 52.0 keV, Lu at 63.31 keV. μ(E) = (μ/ρ)(E) × ρ, (μ/ρ) from NIST XCOM
    # (mixture, coherent incl.), tabulated at 7.0 g/cm³ and scaled to the 6.40 g/cm³ of
    # GEMSTONE_MC_EFFICIENCY_LUT (7.0 is impossible for any (Tb,Lu)₃Al₅O₁₂).
    "Gemstone" => (
        [
            10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
            51.0, 51.99, 52.01, 53.0, 55.0, 57.0, 59.0, 60.0,
            61.0, 63.0, 63.3, 63.32, 65.0, 67.0, 69.0,
            71.0, 73.0, 75.0, 77.0, 80.0, 85.0,
            91.0, 95.0, 100.0, 105.0, 110.0, 115.0, 121.0,
            125.0, 131.0, 135.0, 141.0, 150.0,
        ],
        [
            1098.3, 486.2, 228.3, 126.4, 78.3, 52.1, 36.8, 27.3, 20.8,
            19.8, 18.9, 47.2, 44.9, 40.8, 37.2, 34.1, 32.7,
            31.3, 28.8, 28.5, 51.0, 47.7, 44.2, 41.0,
            38.1, 35.5, 33.2, 31.0, 28.1, 24.1,
            20.3, 18.2, 16.0, 14.1, 12.3, 11.2, 9.9,
            9.1, 8.1, 7.6, 6.8, 5.9,
        ] .* (6.4 / 7.0),
    ),

    # UFC Gd₂O₂S:Pr,Ce — Siemens "Ultra-Fast Ceramic" scintillator (Force and Flash).
    # μ(E) = (μ/ρ)(E) × ρ from the module's GD2O2S compound (NIST XCOM via XrayAttenuation),
    # tabulated at 7.44 g/cm³ and scaled to the 7.31 g/cm³ of UFC_MC_EFFICIENCY_LUT. Gd K-edge
    # (50.24 keV) bracketed at 50.23 / 50.25 keV.
    "UFC" => (
        [
            10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
            50.23, 50.25, 52.0, 55.0, 60.0, 65.0, 70.0, 75.0,
            80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
        ],
        [
            1700.9, 588.5, 274.3, 151.3, 93.2, 61.9, 43.6, 32.0, 24.4,
            24.1, 115.6, 105.8, 91.5, 73.0, 59.4, 49.0, 41.0,
            34.7, 29.7, 25.6, 19.4, 15.2, 12.2, 9.9, 8.3, 7.0,
        ] .* (7.31 / 7.44),
    ),
)

include("eid_response_tables.jl")

# =============================================================================
# The scintillator model
# =============================================================================

"""
    DetectorEfficiencyMode

How [`eid_signal_moments`](@ref) computes an energy-integrating detector's response:
[`MC_LUT`](@ref) (its Geant4 table) or [`BEER_LAMBERT`](@ref) (analytic absorption).
"""
@enum DetectorEfficiencyMode begin
    BEER_LAMBERT
    MC_LUT
end

@doc """
    BEER_LAMBERT::DetectorEfficiencyMode

Analytic scintillator absorption: a photon interacts with probability `q = 1 − exp(−μ(E)·d)`,
`μ(E)` from [`get_scintillator_mu`](@ref) and `d` the crystal depth, and leaves all its energy:
`m₁ = E·q`, `m₂ = E²·q`. It has no fluorescence escape and no Swank spread. Select it with
`SimOptions(detector_efficiency_mode = :beer_lambert)`, e.g. to compare against the table.
""" BEER_LAMBERT

@doc """
    MC_LUT::DetectorEfficiencyMode

The Geant4 flat-field response of the scanner's scintillator:
[`GEMSTONE_MC_EFFICIENCY_LUT`](@ref) (`:lumex`), [`UFC_MC_EFFICIENCY_LUT`](@ref) (`:ufc`) and
[`UFC_FLASH_MC_EFFICIENCY_LUT`](@ref) (`:ufc_flash`). It carries the fluorescence escape at the
K-edges and the Swank spread of the collected energy.
""" MC_LUT

# Each scintillator's table, and its attenuation in SCINTILLATOR_MU_DATA.
const _EID_TABLES = (
    lumex = GEMSTONE_MC_EFFICIENCY_LUT,
    ufc = UFC_MC_EFFICIENCY_LUT,
    ufc_flash = UFC_FLASH_MC_EFFICIENCY_LUT,
)
const _EID_MU_KEYS = (lumex = "Gemstone", ufc = "UFC", ufc_flash = "UFC")

"""
    DetectorEfficiency(material::Symbol, thickness_mm, mode::DetectorEfficiencyMode)

The scintillator of an energy-integrating detector: the scanner's `detector_material`
(`:lumex`, `:ufc` or `:ufc_flash`), its depth in mm and the [`DetectorEfficiencyMode`](@ref).
A table is made at one depth, so [`MC_LUT`](@ref) requires the table's; another depth needs
[`BEER_LAMBERT`](@ref) or a table of its own. The fill factor is the scanner's, applied once in
the noise and never here.
"""
struct DetectorEfficiency
    material::Symbol
    thickness_mm::Float64
    mode::DetectorEfficiencyMode

    function DetectorEfficiency(material::Symbol, thickness_mm::Real, mode::DetectorEfficiencyMode)
        haskey(_EID_TABLES, material) || throw(
            ArgumentError(
                "no energy-integrating detector model for :$(material); the scintillators are " *
                    "$(join(":" .* string.(keys(_EID_TABLES)), ", "))"
            )
        )
        thickness_mm > 0 || throw(ArgumentError("thickness_mm must be positive (got $(thickness_mm))"))
        table_mm = _EID_TABLES[material].thickness_mm
        (mode == BEER_LAMBERT || thickness_mm ≈ table_mm) || throw(
            ArgumentError(
                "the :$(material) table was made at $(table_mm) mm, the detector is $(thickness_mm) mm " *
                    "deep; use detector_efficiency_mode = :beer_lambert or a table made at that depth"
            )
        )
        return new(material, Float64(thickness_mm), mode)
    end
end

"""
    eid_signal_moments(model::DetectorEfficiency, energies) -> (m₁, m₂)

The mean signal one incident photon of each energy leaves in a dexel, `m₁` (keV), and its mean
square, `m₂` (keV²). [`MC_LUT`](@ref): `m₁ = E·efficiency`, `m₂ = m₁² / dqe0`, the table
interpolated linearly in energy. [`BEER_LAMBERT`](@ref): `m₁ = E·q`, `m₂ = E²·q`.

`m₁` weights the mean signal of every energy and `m₂` its variance: a ray of incident spectrum
`N(E)` reads `Σ N m₁` with variance `Σ N m₂`, so its NEQ is `(Σ N m₁)² / Σ N m₂`.
"""
function eid_signal_moments(model::DetectorEfficiency, energies::AbstractVector)
    E = Float64.(energies)
    if model.mode == MC_LUT
        t = _EID_TABLES[model.material]
        m1 = E .* _interp_table.(Ref(t.energies), Ref(t.efficiency), E)
        return m1, m1 .^ 2 ./ _interp_table.(Ref(t.energies), Ref(t.dqe0), E)
    end
    key = _EID_MU_KEYS[model.material]
    q = [1.0 - exp(-get_scintillator_mu(key, e) * model.thickness_mm / 10) for e in E]
    return E .* q, E .^ 2 .* q
end

# No detector model (`use_detector_efficiency = false`): every photon leaves all its energy.
eid_signal_moments(::Nothing, energies::AbstractVector) = (Float64.(energies), Float64.(energies) .^ 2)

"""
    get_scintillator_mu(material::String, energy_keV::Float64) -> Float64

Linear attenuation coefficient (cm⁻¹) of a scintillator or sensor at the given energy:
log-linear interpolation of the NIST XCOM table, K-edges bracketed by points on either side.
"""
function get_scintillator_mu(material::String, energy_keV::Float64)
    haskey(SCINTILLATOR_MU_DATA, material) || throw(
        ArgumentError(
            "no attenuation table for \"$(material)\"; the materials are " *
                "$(join(sort(collect(keys(SCINTILLATOR_MU_DATA))), ", "))"
        )
    )
    energies, mus = SCINTILLATOR_MU_DATA[material]
    E = clamp(energy_keV, energies[1], energies[end])
    i = min(searchsortedlast(energies, E), length(energies) - 1)
    t = (log(E) - log(energies[i])) / (log(energies[i + 1]) - log(energies[i]))
    return exp(log(mus[i]) + t * (log(mus[i + 1]) - log(mus[i])))
end

# Linear interpolation in a table, held at its ends.
function _interp_table(x::AbstractVector, y::AbstractVector, xq::Real)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    return y[i] + (xq - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i])
end

# =============================================================================
# Exports
# =============================================================================

export DetectorEfficiency, DetectorEfficiencyMode, BEER_LAMBERT, MC_LUT
export get_scintillator_mu, eid_signal_moments
export GEMSTONE_MC_EFFICIENCY_LUT, UFC_MC_EFFICIENCY_LUT, UFC_FLASH_MC_EFFICIENCY_LUT
