# =============================================================================
# CdTe Material Constants for PCCT Detector Physics
# =============================================================================
#
# Centralized CdTe/CZT/Si material properties for physics-based PCCT simulation.
# All values sourced from published literature with citations.
#
# Primary references:
#   - Koch-Mehrin et al. 2020, NIM-A 976:164241 (charge cloud, fluorescence, CCE)
#   - Konrad et al. 2025, PMB 70:065004 (NAEOTOM Alpha detector parameters)
#   - Owens 2012, "Compound Semiconductor Radiation Detectors" (carrier transport)
#   - Redus et al. 2009, IEEE TNS 56:2524 (Fano factor)
#   - Bambynek et al. 1972, Rev Mod Phys 44:716 (fluorescence yields)
#
# DESIGN: Constants are organized as immutable structs for type safety and
# GPU compatibility (structs can be passed to kernels). Functions that depend
# on these constants import them from here rather than hardcoding values.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Physical Constants
# ─────────────────────────────────────────────────────────────────────────────

"""Physical constants used in charge transport calculations."""
const BOLTZMANN_J_PER_K = 1.380649e-23       # J/K — CODATA 2018 exact
const ELEMENTARY_CHARGE_C = 1.602176634e-19  # C — CODATA 2018 exact
const VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12  # F/m — CODATA 2018

# ─────────────────────────────────────────────────────────────────────────────
# CdTe Carrier Transport
# ─────────────────────────────────────────────────────────────────────────────

"""
    CdTeTransport

Carrier transport properties for CdTe at operating temperature.

All values at T = 301.15 K (28°C) unless noted.

# References
- Electron mobility: Owens 2012, Table 3.1
- Hole mobility: Owens 2012, Table 3.1
- Lifetimes: Koch-Mehrin 2020, Section 4.1 (HEXITEC measurements)
- μτ products: Koch-Mehrin 2020 (μₑτₑ ≈ 3.3e-3, μₕτₕ ≈ 2.0e-4 cm²/V)
"""
struct CdTeTransport
    # Individual carrier properties
    electron_mobility_cm2_per_Vs::Float64  # μ_e
    hole_mobility_cm2_per_Vs::Float64      # μ_h
    electron_lifetime_s::Float64           # τ_e
    hole_lifetime_s::Float64               # τ_h

    # Mobility-lifetime products (used in Hecht equation)
    mu_e_tau_e_cm2_per_V::Float64          # μₑτₑ = μ_e × τ_e
    mu_h_tau_h_cm2_per_V::Float64          # μₕτₕ = μ_h × τ_h

    # Material properties
    relative_permittivity::Float64         # ε_r (CdTe ≈ 10.2)
    permittivity_F_per_m::Float64          # ε = ε_r × ε₀
    density_g_per_cm3::Float64             # ρ
    temperature_K::Float64                 # Operating temperature

    # Pair creation
    pair_creation_energy_eV::Float64       # ω (energy per e-h pair)
    fano_factor::Float64                   # F (intrinsic variance reduction)
end

"""Default CdTe transport properties from Koch-Mehrin 2020 and Owens 2012."""
const CDTE_TRANSPORT = CdTeTransport(
    1100.0,    # μ_e = 1100 cm²/V·s — Owens 2012
    100.0,     # μ_h = 100 cm²/V·s — Owens 2012
    3.0e-6,    # τ_e = 3 μs — Koch-Mehrin 2020 Section 4.1
    2.0e-6,    # τ_h = 2 μs — Koch-Mehrin 2020 Section 4.1
    3.3e-3,    # μₑτₑ = 1100 × 3e-6 = 3.3e-3 cm²/V
    2.0e-4,    # μₕτₕ = 100 × 2e-6 = 2.0e-4 cm²/V
    10.2,      # ε_r — CdTe relative permittivity
    10.2 * VACUUM_PERMITTIVITY_F_PER_M,  # ε = 9.03e-11 F/m
    5.85,      # ρ = 5.85 g/cm³ — Koch-Mehrin 2020
    301.15,    # T = 28°C = 301.15 K — Koch-Mehrin 2020 Section 4.1
    4.3,       # ω = 4.3 eV — Koch-Mehrin 2020
    0.1        # F = 0.1 — Redus et al. 2009
)

# ─────────────────────────────────────────────────────────────────────────────
# K-Shell Fluorescence Transitions
# ─────────────────────────────────────────────────────────────────────────────

"""
    KShellTransition

A single K-shell fluorescence transition line.

# Fields
- `label`: Transition label (e.g., "Kα1", "Kβ1")
- `shell_notation`: Siegbahn notation (e.g., "KL3", "KM3")
- `energy_keV`: Fluorescence photon energy
- `relative_rate`: Emission rate relative to other transitions of same element
"""
struct KShellTransition
    label::String
    shell_notation::String
    energy_keV::Float64
    relative_rate::Float64
end

"""
    ElementFluorescence

Complete fluorescence data for one element.

# Fields
- `symbol`: Element symbol (:Cd or :Te)
- `atomic_number`: Z
- `k_edge_keV`: K-shell absorption edge energy
- `k_yield`: K-shell fluorescence yield ω_K
- `k_transitions`: All K-shell transition lines
- `mean_path_mm`: Mean attenuation length of Kα in CdTe crystal [mm]
"""
struct ElementFluorescence
    symbol::Symbol
    atomic_number::Int
    k_edge_keV::Float64
    k_yield::Float64
    k_transitions::Vector{KShellTransition}
    mean_path_mm::Float64
end

"""
Cadmium K-shell fluorescence data.

K-edge: 26.711 keV (NIST XCOM)
Fluorescence yield: ω_K ≈ 0.84 (Bambynek et al. 1972)
Mean Kα path in CdTe: ~100 μm (Koch-Mehrin 2020)

Transition table from Koch-Mehrin 2020, Table 1.
"""
const CD_FLUORESCENCE = ElementFluorescence(
    :Cd, 48,
    26.711,  # K-edge [keV] — NIST
    0.84,    # ω_K — Bambynek et al. 1972
    [
        KShellTransition("Kα1", "KL3", 23.174, 0.55),
        KShellTransition("Kα2", "KL2", 22.984, 0.29),
        KShellTransition("Kβ1", "KM3", 26.095, 0.09),
        KShellTransition("Kβ2", "KM2", 26.654, 0.02),
        KShellTransition("Kβ3", "KM2", 26.061, 0.05),
    ],
    0.100  # ~100 μm mean path for Cd Kα in CdTe
)

"""
Tellurium K-shell fluorescence data.

K-edge: 31.814 keV (NIST XCOM)
Fluorescence yield: ω_K ≈ 0.88 (Bambynek et al. 1972)
Mean Kα path in CdTe: ~60 μm (Koch-Mehrin 2020)

CRITICAL: Te Kα (27.473 keV) is ABOVE Cd K-edge (26.711 keV).
This means Te fluorescence can trigger SECONDARY Cd fluorescence (cascade).

Transition table from Koch-Mehrin 2020, Table 1.
"""
const TE_FLUORESCENCE = ElementFluorescence(
    :Te, 52,
    31.814,  # K-edge [keV] — NIST
    0.88,    # ω_K — Bambynek et al. 1972
    [
        KShellTransition("Kα1", "KL3", 27.473, 0.53),
        KShellTransition("Kα2", "KL2", 27.202, 0.29),
        KShellTransition("Kβ1", "KM3", 30.996, 0.10),
        KShellTransition("Kβ2", "KM2", 31.700, 0.03),
        KShellTransition("Kβ3", "KM2", 30.945, 0.05),
    ],
    0.060  # ~60 μm mean path for Te Kα in CdTe
)

# ─────────────────────────────────────────────────────────────────────────────
# Detector Geometry Presets
# ─────────────────────────────────────────────────────────────────────────────

"""
    PCCTDetectorGeometry

Physical dimensions of a PCCT detector pixel.

# Fields
- `name`: Detector identifier
- `pixel_pitch_mm`: Pixel pitch (channel, slice) in mm
- `thickness_mm`: Sensor thickness in mm
- `bias_voltage_V`: Applied bias voltage
- `effective_voltage_V`: Effective internal voltage (after contact resistance)
- `electronic_noise_keV`: Electronic noise σ in keV
- `noise_threshold_keV`: Energy threshold below which events are discarded
"""
struct PCCTDetectorGeometry
    name::String
    pixel_pitch_mm::Tuple{Float64, Float64}
    thickness_mm::Float64
    bias_voltage_V::Float64
    effective_voltage_V::Float64
    electronic_noise_keV::Float64
    noise_threshold_keV::Float64
end

"""
Siemens NAEOTOM Alpha detector geometry (Konrad et al. 2025, Table 1).

- CdTe sensor, 1.6 mm thick
- 275 × 322 μm pixel pitch (non-square: channel × slice)
- Bias voltage ~800 V (effective voltage estimated)
- Electronic noise 0.6 keV (σ)
- Clinical thresholds: [20, 35, 55, 70] keV
- Measurement thresholds (Konrad): [20, 35, 65, 80] keV
"""
const NAEOTOM_ALPHA = PCCTDetectorGeometry(
    "NAEOTOM Alpha",
    (0.275, 0.322),   # pixel pitch (channel, slice) [mm]
    1.6,              # thickness [mm]
    800.0,            # bias voltage [V]
    680.0,            # effective voltage [V] (estimated: ~85% of applied)
    0.6,              # electronic noise σ [keV] — Konrad 2025 Table 1
    3.0               # noise threshold [keV] — Koch-Mehrin 2020 Section 4
)

"""
HEXITEC detector geometry (Koch-Mehrin 2020, Section 3).

- CdTe sensor, 1.0 mm thick
- 250 × 250 μm pixel pitch (square)
- Bias voltage -500 V (effective -425 V per Cola et al. 2006)
- Electronic noise 60 electrons rms ≈ 0.26 keV (σ) at ω=4.3 eV
- Used for validation against Koch-Mehrin published data
"""
const HEXITEC = PCCTDetectorGeometry(
    "HEXITEC",
    (0.250, 0.250),   # pixel pitch [mm]
    1.0,              # thickness [mm]
    500.0,            # bias voltage [V] (applied)
    425.0,            # effective voltage [V] — Cola et al. 2006
    0.26,             # electronic noise σ [keV] (60 electrons × 4.3 eV / 1000)
    3.0               # noise threshold [keV] — Koch-Mehrin 2020 Section 4
)

# ─────────────────────────────────────────────────────────────────────────────
# Initial Charge Cloud Size
# ─────────────────────────────────────────────────────────────────────────────

"""
    initial_cloud_sigma_um(E_keV::Real) -> Float64

Compute initial charge cloud size from photoelectron excitation.

Second-order polynomial fit to GEANT simulations (Blevis & Levinson 2005,
via Koch-Mehrin 2020 Section 2.3):
    σ_i(E) = a₁E + a₂E²  [μm, keV]

Fitted to: (0 keV, 0 μm), (35 keV, 1 μm), (100 keV, 4 μm).
Solving: 35a₁ + 1225a₂ = 1, 100a₁ + 10000a₂ = 4
    → a₁ = 51/2275 ≈ 0.02242, a₂ = 1/5688 ≈ 0.0001758

For diagnostic CT energies (30-140 keV), the initial cloud is small compared
to drift growth (σ_i ≈ 1-4 μm vs σ_drift ≈ 10-13 μm). Above ~100 keV,
initial cloud starts to dominate.

# Arguments
- `E_keV`: Photoelectron energy in keV

# Returns
- Initial cloud σ in μm
"""
function initial_cloud_sigma_um(E_keV::Real)
    E = Float64(E_keV)
    # Exact rational coefficients: a₁ = 51/2275, a₂ = 1/5688
    return max(0.0, (51.0 / 2275.0) * E + (1.0 / 5688.0) * E^2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    cdte_diffusion_coefficient_cm2_per_s(; transport::CdTeTransport=CDTE_TRANSPORT) -> Float64

Compute electron diffusion coefficient D from Einstein relation (Koch-Mehrin 2020, Eq. 11):
    D = μ_e × k_B × T / q

# Returns
- D in cm²/s
"""
function cdte_diffusion_coefficient_cm2_per_s(; transport::CdTeTransport=CDTE_TRANSPORT)
    return transport.electron_mobility_cm2_per_Vs * BOLTZMANN_J_PER_K *
           transport.temperature_K / ELEMENTARY_CHARGE_C
end

"""
    electron_drift_time_s(drift_length_cm::Real, sensor_thickness_cm::Real,
                          bias_voltage_V::Real;
                          transport::CdTeTransport=CDTE_TRANSPORT) -> Float64

Compute electron drift time to anode (Koch-Mehrin 2020, Eq. 12):
    t_d = d × L / (μ_e × V)

where d = drift distance, L = sensor thickness.

# Arguments
- `drift_length_cm`: Distance from interaction point to anode [cm]
- `sensor_thickness_cm`: Full sensor thickness [cm]
- `bias_voltage_V`: Applied bias voltage [V]
"""
function electron_drift_time_s(drift_length_cm::Real, sensor_thickness_cm::Real,
                                bias_voltage_V::Real;
                                transport::CdTeTransport=CDTE_TRANSPORT)
    return Float64(drift_length_cm) * Float64(sensor_thickness_cm) /
           (transport.electron_mobility_cm2_per_Vs * Float64(bias_voltage_V))
end

"""
    num_electron_hole_pairs(E_keV::Real; transport::CdTeTransport=CDTE_TRANSPORT) -> Float64

Mean number of electron-hole pairs generated by a photon of energy E.
    N = E / ω

Standard deviation from Fano statistics:
    σ_N = √(F × N)

# Arguments
- `E_keV`: Photon energy in keV

# Returns
- Mean number of e-h pairs (Float64)
"""
function num_electron_hole_pairs(E_keV::Real; transport::CdTeTransport=CDTE_TRANSPORT)
    E_eV = Float64(E_keV) * 1000.0
    return E_eV / transport.pair_creation_energy_eV
end

"""
    weighted_mean_fluorescence_energy(element::ElementFluorescence) -> Float64

Compute the probability-weighted mean fluorescence energy for an element.
Useful for quick analytical escape peak estimates.

# Returns
- Weighted mean K-fluorescence energy in keV
"""
function weighted_mean_fluorescence_energy(element::ElementFluorescence)
    total_rate = 0.0
    weighted_E = 0.0
    for t in element.k_transitions
        weighted_E += t.energy_keV * t.relative_rate
        total_rate += t.relative_rate
    end
    return weighted_E / total_rate
end

"""
    pixel_to_thickness_ratio(geom::PCCTDetectorGeometry) -> Float64

Compute w/L ratio (pixel pitch / sensor thickness) for small-pixel effect assessment.

- w/L < 0.3: strong small-pixel effect (NAEOTOM: ~0.17)
- w/L ≈ 0.5: moderate small-pixel effect
- w/L > 1.0: negligible small-pixel effect (planar geometry)

# Returns
- w/L ratio using the smaller pixel pitch dimension
"""
function pixel_to_thickness_ratio(geom::PCCTDetectorGeometry)
    w = min(geom.pixel_pitch_mm...)
    return w / geom.thickness_mm
end
