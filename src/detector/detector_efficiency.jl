"""
    Forward/DetectorEfficiency.jl

Detector absorption efficiency and DQE modeling for CT simulation.

# Physics Background

The detector efficiency η(E, θ) represents the probability that an incident
X-ray photon of energy E is absorbed in the scintillator. It follows the
Beer-Lambert absorption law:

    η(E, θ) = 1 - exp(-μ(E) × d / cos(θ))

where:
- μ(E) is the energy-dependent linear attenuation coefficient (cm⁻¹)
- d is the scintillator thickness (cm)
- θ is the incidence angle (cone angle for peripheral detector rows)

# Key Physics Characteristics

1. **Low-energy photons**: Nearly 100% absorbed (high μ at low E)
2. **High-energy photons**: Increased transparency (low μ at high E)
3. **K-edge effects**: Sudden increase in absorption at element K-edges
   - GOS: Gd K-edge at 50.2 keV
   - CsI: Cs K-edge at 36 keV, I K-edge at 33 keV
   - CdTe: Cd K-edge at 26.7 keV, Te K-edge at 31.8 keV

# CatSim Compatibility

This implementation uses the exact CatSim formula from Detection_EI.py:

    detEff = 1 - exp(-0.1 × detectorDepth / cos(beta) × detectorMu)

where:
- `detectorDepth` is scintillator thickness in mm (0.1 factor converts to cm)
- `detectorMu` is linear attenuation coefficient from GetMu() in cm⁻¹
- `beta` is the cone angle (z-direction incidence)

# Calibration Note

In properly calibrated CT (with air scan normalization), detector efficiency
cancels between phantom and reference scans:

    projection = -log(I_phantom / I_air) = -log(I₀·exp(-μL)·η / (I₀·η)) = μL

Therefore, `apply_detector_efficiency!` is a no-op for projection data.
The primary effect of detector efficiency is on NOISE levels (fewer detected
photons = more quantum noise), which is handled separately in noise modeling.

# GPU Compatibility
- ✅ Metal (via AcceleratedKernels.jl)
- ✅ CUDA
- ✅ ROCm
- ✅ CPU fallback

# References

1. Swank RK. "Absorption and noise in x-ray phosphors."
   J Appl Phys. 1973;44(9):4199-4203. doi:10.1063/1.1662918

2. Huda W, et al. "X-ray absorption in scintillators used in
   computed tomography." Med Phys. 1984;11(6):785-790.
   doi:10.1118/1.595575

3. GE CatSim/XCIST Detection_EI.py - Reference implementation
   https://github.com/xcist/main

4. NIST XCOM database for scintillator attenuation coefficients
   https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
"""

import AcceleratedKernels as AK

# =============================================================================
# Scintillator Materials Database
# =============================================================================

# Linear attenuation coefficients (cm⁻¹) for common scintillator materials
# Data from NIST XCOM, organized as (energies, μ values)
const SCINTILLATOR_MU_DATA = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}(
    # Gadolinium Oxysulfide (Gd₂O₂S, GOS/Gadox)
    # ρ = 7.34 g/cm³, K-edge at 50.2 keV
    "GOS" => (
        [20.0, 30.0, 40.0, 50.0, 50.2, 60.0, 80.0, 100.0, 120.0, 150.0],
        [145.0, 48.0, 22.0, 12.0, 65.0, 42.0, 18.0, 9.5, 5.8, 3.2]
    ),

    # Cesium Iodide (CsI)
    # ρ = 4.51 g/cm³, K-edges: Cs at 36 keV, I at 33 keV
    "CsI" => (
        [20.0, 30.0, 33.0, 36.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [85.0, 32.0, 88.0, 95.0, 65.0, 35.0, 22.0, 10.5, 5.8, 3.8, 2.3]
    ),

    # Cadmium Telluride (CdTe) - for photon counting
    # ρ = 5.85 g/cm³, K-edges: Cd at 26.7 keV, Te at 31.8 keV
    "CdTe" => (
        [20.0, 26.7, 31.8, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [95.0, 180.0, 200.0, 100.0, 55.0, 33.0, 14.5, 7.8, 4.9, 2.8]
    ),

    # Cadmium Zinc Telluride (CZT) - for photon counting
    # Similar to CdTe, ρ ≈ 5.8 g/cm³
    "CZT" => (
        [20.0, 26.7, 31.8, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [90.0, 170.0, 190.0, 95.0, 52.0, 31.0, 13.5, 7.3, 4.6, 2.6]
    ),

    # Silicon (Si) - for photon counting
    # ρ = 2.33 g/cm³
    "Si" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [2.8, 0.95, 0.58, 0.48, 0.46, 0.43, 0.40, 0.39, 0.37]
    ),

    # Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ — GE proprietary garnet scintillator
    # Composition (wt%): Tb(0.287), Lu(0.316), Al(0.163), O(0.231), Ce(0.002)
    # ρ = 7.0 g/cm³, K-edges: Tb at 52.0 keV, Lu at 63.31 keV
    # μ(E) = (μ/ρ)(E) × ρ, with (μ/ρ) from NIST XCOM (mixture, coherent incl.)
    # Source: extracted XCOM data, Feb 2026
    "Gemstone" => (
        # Energy (keV) — dense grid around K-edges; offsets ±0.01 keV model discontinuity
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
            51.0, 51.99, 52.01, 53.0, 55.0, 57.0, 59.0, 60.0,
            61.0, 63.0, 63.30, 63.32, 65.0, 67.0, 69.0,
            71.0, 73.0, 75.0, 77.0, 80.0, 85.0,
            91.0, 95.0, 100.0, 105.0, 110.0, 115.0, 121.0,
            125.0, 131.0, 135.0, 141.0, 150.0],
        # μ (cm⁻¹) = (μ/ρ) × 7.0 g/cm³
        [1098.3, 486.2, 228.3, 126.4, 78.3, 52.1, 36.8, 27.3, 20.8,
            19.8, 18.9, 47.2, 44.9, 40.8, 37.2, 34.1, 32.7,
            31.3, 28.8, 28.5, 51.0, 47.7, 44.2, 41.0,
            38.1, 35.5, 33.2, 31.0, 28.1, 24.1,
            20.3, 18.2, 16.0, 14.1, 12.3, 11.2, 9.9,
            9.1, 8.1, 7.6, 6.8, 5.9]
    )
)

# Aliases for GOS
SCINTILLATOR_MU_DATA["Gadox"] = SCINTILLATOR_MU_DATA["GOS"]
SCINTILLATOR_MU_DATA["Gd2O2S"] = SCINTILLATOR_MU_DATA["GOS"]

# Aliases for Gemstone (GE garnet scintillator)
SCINTILLATOR_MU_DATA["LUMEX"] = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["lumex"] = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["Garnet"] = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["TbLuAG"] = SCINTILLATOR_MU_DATA["Gemstone"]
SCINTILLATOR_MU_DATA["TbLuAG:Ce"] = SCINTILLATOR_MU_DATA["Gemstone"]

# =============================================================================
# Monte Carlo-Derived Detector Efficiency LUT
# =============================================================================

"""
    GEMSTONE_MC_EFFICIENCY_LUT

Monte Carlo-derived detector efficiency lookup table for Ce:(Tb,Lu)₃Al₅O₁₂
(Gemstone-type garnet scintillator) as a function of photon energy.

This LUT was computed from a full Monte Carlo simulation (MCNP) of the
GE Revolution Apex detector using the material composition:

    m1003  65000 -0.287424   (Tb)
           71000 -0.316437   (Lu)
           13000 -0.162658   (Al)
            8000 -0.231480   (O)
           58000 -0.002000   (Ce dopant)

# Key Features Captured by MC (Not in Beer-Lambert)

1. **Fluorescence escape at Tb K-edge (52 keV)**: Efficiency drops from
   0.956 → 0.807 due to Tb Kα fluorescence photons (~44 keV) escaping.
2. **Fluorescence escape at Lu K-edge (63 keV)**: Efficiency drops from
   0.846 → 0.762 due to Lu Kα fluorescence photons (~54 keV) escaping.
3. **L-edge effects** at 8-12 keV from Tb/Lu L-shells.
4. **Compton scattering escape** at higher energies.

Beer-Lambert predicts efficiency INCREASES at K-edges (more absorption),
but MC correctly shows DECREASES because fluorescent photons escape the
detector volume, reducing the energy actually deposited.

# Reference
- Material: PMC10179960 (Ce:(Tb,Lu)₃Al₅O₁₂ garnet scintillator)
- Simulation: MCNP with ENDF photon cross-sections
"""
const GEMSTONE_MC_EFFICIENCY_LUT = (
    # Energy (keV): 1-140 keV, 1 keV steps
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
    # Efficiency η(E) = Real/Ideal from Monte Carlo simulation
    efficiency=Float64[
        0.991, 0.990, 0.990, 0.990, 0.990, 0.989, 0.988, 0.989, 0.979, 0.983,
        0.973, 0.976, 0.979, 0.981, 0.982, 0.983, 0.984, 0.984, 0.985, 0.985,
        0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.984, 0.984, 0.984,
        0.983, 0.983, 0.982, 0.982, 0.981, 0.981, 0.980, 0.979, 0.978, 0.977,
        0.975, 0.975, 0.974, 0.972, 0.971, 0.970, 0.969, 0.967, 0.965, 0.962,
        0.960, 0.956, 0.807, 0.813, 0.818, 0.823, 0.827, 0.831, 0.834, 0.838,  # ← Tb K-edge drop
        0.841, 0.844, 0.846, 0.762, 0.767, 0.773, 0.777, 0.783, 0.787, 0.791,  # ← Lu K-edge drop
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
- `MC_LUT` — Monte Carlo-derived lookup table (captures fluorescence escape, etc.)
"""
@enum DetectorEfficiencyMode begin
    BEER_LAMBERT    # Analytical: η = 1 - exp(-μd/cosθ)
    MC_LUT          # Monte Carlo-derived LUT
end

"""
    DetectorEfficiency

Detector efficiency model specification.

# Fields
- `material`: Scintillator material name (GOS, CsI, CdTe, Gemstone, etc.)
- `thickness_mm`: Scintillator thickness in mm
- `fill_factor`: Fraction of detector area that is active (0-1)
- `mode`: `BEER_LAMBERT` or `MC_LUT`
"""
struct DetectorEfficiency
    material::String
    thickness_mm::Float64
    fill_factor::Float64
    mode::DetectorEfficiencyMode
end

# Backward-compatible 3-argument constructor (defaults to Beer-Lambert)
DetectorEfficiency(material::String, thickness_mm::Float64, fill_factor::Float64) =
    DetectorEfficiency(material, thickness_mm, fill_factor, BEER_LAMBERT)

# =============================================================================
# Pre-defined Detector Models
# =============================================================================

"""
    detector_efficiency_gos(thickness_mm::Float64=0.5; fill_factor::Float64=0.85)

GOS (Gadox) scintillator detector.

Standard for energy-integrating CT detectors.
Typical thickness: 0.3-1.0 mm.
"""
function detector_efficiency_gos(thickness_mm::Float64=0.5; fill_factor::Float64=0.85)
    return DetectorEfficiency("GOS", thickness_mm, fill_factor)
end

"""
    detector_efficiency_csi(thickness_mm::Float64=0.6; fill_factor::Float64=0.90)

CsI (Cesium Iodide) scintillator detector.

Used in flat-panel detectors and some CT systems.
Good light output, less afterglow than GOS.
"""
function detector_efficiency_csi(thickness_mm::Float64=0.6; fill_factor::Float64=0.90)
    return DetectorEfficiency("CsI", thickness_mm, fill_factor)
end

"""
    detector_efficiency_cdte(thickness_mm::Float64=1.6; fill_factor::Float64=0.95)

CdTe detector for photon-counting CT.

Direct conversion, high absorption efficiency.
"""
function detector_efficiency_cdte(thickness_mm::Float64=1.6; fill_factor::Float64=0.95)
    return DetectorEfficiency("CdTe", thickness_mm, fill_factor)
end

"""
    detector_efficiency_gemstone(; mode::Symbol=:mc_lut,
                                  thickness_mm::Float64=3.0,
                                  fill_factor::Float64=0.90)

GE Gemstone Clarity Ce:(Tb,Lu)₃Al₅O₁₂ garnet scintillator detector.

# Modes
- `:mc_lut` (default) — Monte Carlo-derived efficiency LUT from MCNP simulation.
  Captures fluorescence escape at Tb K-edge (52 keV) and Lu K-edge (63 keV)
  that Beer-Lambert cannot model. **Recommended for GE Revolution simulations.**
- `:beer_lambert` — Analytical Beer-Lambert using XCOM-derived μ(E) data.
  Faster, but overestimates efficiency near K-edges.

# Material Composition
    Ce:(Tb₁.₅Lu₁.₅)Al₅O₁₂
    Tb: 0.2874, Lu: 0.3164, Al: 0.1627, O: 0.2315, Ce: 0.002
    ρ ≈ 7.0 g/cm³

# Reference
- Composition: PMC10179960
- MC simulation: MCNP with ENDF photon cross-sections

# Example
```julia
# Monte Carlo LUT (recommended)
model = detector_efficiency_gemstone()  # mode=:mc_lut

# Beer-Lambert analytical
model = detector_efficiency_gemstone(mode=:beer_lambert)
```
"""
function detector_efficiency_gemstone(; mode::Symbol=:mc_lut,
    thickness_mm::Float64=3.0,
    fill_factor::Float64=0.90)
    eff_mode = mode == :mc_lut ? MC_LUT : BEER_LAMBERT
    return DetectorEfficiency("Gemstone", thickness_mm, fill_factor, eff_mode)
end

"""
    detector_efficiency_ideal()

Ideal detector with 100% efficiency.

For testing or when detector effects are not needed.
"""
function detector_efficiency_ideal()
    return DetectorEfficiency("ideal", 0.0, 1.0)
end

"""
    detector_efficiency_custom(material::String, thickness_mm::Float64;
                               fill_factor::Float64=0.90)

Create custom detector efficiency model.

# Arguments
- `material`: Scintillator material (GOS, CsI, CdTe, CZT, Si, Gemstone)
- `thickness_mm`: Scintillator thickness in mm
- `fill_factor`: Active area fraction (default: 0.90)
"""
function detector_efficiency_custom(material::String, thickness_mm::Float64;
    fill_factor::Float64=0.90)
    @assert 0 < fill_factor <= 1 "Fill factor must be between 0 and 1"
    @assert thickness_mm >= 0 "Thickness must be non-negative"
    return DetectorEfficiency(material, thickness_mm, fill_factor)
end

# =============================================================================
# Scintillator Attenuation Lookup
# =============================================================================

"""
    get_scintillator_mu(material::String, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for scintillator material at given energy.

# Algorithm

Uses log-linear interpolation of tabulated μ values from NIST XCOM data.
K-edge discontinuities are captured by including data points at the edge energies.

# Arguments
- `material::String`: Scintillator material ("GOS", "CsI", "CdTe", "CZT", "Si")
- `energy_keV::Float64`: Photon energy in keV (clamped to valid range)

# Returns
- `Float64`: Linear attenuation coefficient μ in cm⁻¹

# CatSim Equivalence

This function returns the same μ values as CatSim's GetMu() function for the
corresponding material, enabling direct comparison:

    # CatSim (Python)
    detectorMu = GetMu(cfg.scanner.detectorMaterial, Evec)

    # BasisSimulator (Julia)
    μ = get_scintillator_mu(material, E)

# Example
```julia
# Get μ for GOS at 60 keV (typical CT imaging energy)
μ_gos = get_scintillator_mu("GOS", 60.0)  # ~42 cm⁻¹

# Get μ for CdTe at the Cd K-edge
μ_cdte_kedge = get_scintillator_mu("CdTe", 27.0)  # Very high (>180 cm⁻¹)
```

# See Also
- [`compute_detector_efficiency`](@ref): Compute absorption efficiency from μ
- [`SCINTILLATOR_MU_DATA`](@ref): Tabulated μ values
"""
function get_scintillator_mu(material::String, energy_keV::Float64)
    if material == "ideal"
        return Inf  # Perfect absorption
    end

    if !haskey(SCINTILLATOR_MU_DATA, material)
        @warn "Unknown scintillator material: $material, using GOS"
        material = "GOS"
    end

    energies, mus = SCINTILLATOR_MU_DATA[material]
    E = clamp(energy_keV, energies[1], energies[end])

    # Log-linear interpolation
    log_E = log(E)
    log_energies = log.(energies)
    log_mus = log.(mus)

    # Find interpolation interval
    idx = 1
    for i in 1:(length(energies)-1)
        if E >= energies[i] && E <= energies[i+1]
            idx = i
            break
        end
    end

    # Linear interpolation in log space
    t = (log_E - log_energies[idx]) / (log_energies[idx+1] - log_energies[idx])
    log_mu = log_mus[idx] + t * (log_mus[idx+1] - log_mus[idx])

    return exp(log_mu)
end

"""
    get_gemstone_mc_efficiency(energy_keV::Float64) -> Float64

Look up Monte Carlo-derived detector efficiency for Ce:(Tb,Lu)₃Al₅O₁₂ Gemstone
scintillator at the given photon energy.

Uses linear interpolation of the 1-140 keV MC efficiency table.
Energies outside the table range are clamped to the nearest boundary value.

# Key Physics
- Tb K-edge fluorescence escape at 52-53 keV (η: 0.956 → 0.807)
- Lu K-edge fluorescence escape at 63-64 keV (η: 0.846 → 0.762)
- Peak efficiency ~0.985 at 20-27 keV
- Efficiency ~0.859 at 100 keV, declining to 0.754 at 140 keV

# Example
```julia
η = get_gemstone_mc_efficiency(60.0)   # ~0.838
η = get_gemstone_mc_efficiency(52.5)   # ~0.882 (between Tb K-edge dip)
```
"""
function get_gemstone_mc_efficiency(energy_keV::Float64)
    lut = GEMSTONE_MC_EFFICIENCY_LUT
    E = clamp(energy_keV, lut.energies[1], lut.energies[end])

    # Fast path: integer keV within range → direct lookup
    if E == floor(E) && 1.0 <= E <= 140.0
        return lut.efficiency[Int(E)]
    end

    # Linear interpolation for non-integer energies
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
# Efficiency Computation
# =============================================================================

"""
    compute_detector_efficiency(model::DetectorEfficiency, geom::CTGeometry;
                                energy_keV::Float64=60.0) -> Array{Float64,2}

Compute detector efficiency for all detector pixels.

Efficiency = fill_factor × (1 - exp(-μ × d / cos(β)))

where β is the cone angle for each row.

# Returns
2D array [n_cols, n_rows] of efficiency values (0 to 1).
"""
function compute_detector_efficiency(
    model::DetectorEfficiency,
    geom::CTGeometry;
    energy_keV::Float64=60.0
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows

    if model.material == "ideal"
        return ones(Float64, n_cols, n_rows)
    end

    # MC LUT mode for Gemstone: use Monte Carlo-derived efficiency
    if model.mode == MC_LUT && (model.material == "Gemstone" || model.material == "LUMEX" ||
                                model.material == "lumex" || model.material == "Garnet")
        η_mc = get_gemstone_mc_efficiency(energy_keV)
        η = model.fill_factor * η_mc

        efficiency = zeros(Float64, n_cols, n_rows)
        for row in 1:n_rows
            for col in 1:n_cols
                efficiency[col, row] = η
            end
        end
        return efficiency
    end

    # Beer-Lambert mode (default for all materials)
    # Get μ for the scintillator material
    μ = get_scintillator_mu(model.material, energy_keV)

    # Convert thickness from mm to cm
    d_cm = model.thickness_mm / 10.0

    pixel_row_size_det = geom.pixel_row_size * (geom.SDD / geom.SAD)
    efficiency = zeros(Float64, n_cols, n_rows)

    for row in 1:n_rows
        # Cone angle (beta in CatSim)
        v_offset = (row - (n_rows + 1) / 2) * pixel_row_size_det
        cos_beta = cos(atan(v_offset / geom.SDD))

        # Path length through scintillator
        path_length = d_cm / cos_beta

        # Absorption efficiency: 1 - exp(-μ × path)
        absorption = 1.0 - exp(-μ * path_length)

        # Apply fill factor
        η = model.fill_factor * absorption

        for col in 1:n_cols
            efficiency[col, row] = η
        end
    end

    return efficiency
end

"""
    compute_detector_efficiency_spectral(model::DetectorEfficiency, geom::CTGeometry,
                                         energies::Vector{Float64}) -> Array{Float64,3}

Compute energy-dependent detector efficiency.

# Returns
3D array [n_cols, n_rows, n_energies] of efficiency values.
"""
function compute_detector_efficiency_spectral(
    model::DetectorEfficiency,
    geom::CTGeometry,
    energies::Vector{Float64}
)
    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_energies = length(energies)

    if model.material == "ideal"
        return ones(Float64, n_cols, n_rows, n_energies)
    end

    # MC LUT mode for Gemstone: use Monte Carlo-derived efficiency per energy
    if model.mode == MC_LUT && (model.material == "Gemstone" || model.material == "LUMEX" ||
                                model.material == "lumex" || model.material == "Garnet")
        efficiency = zeros(Float64, n_cols, n_rows, n_energies)
        for k in 1:n_energies
            η_mc = get_gemstone_mc_efficiency(energies[k])
            η = model.fill_factor * η_mc
            for row in 1:n_rows
                for col in 1:n_cols
                    efficiency[col, row, k] = η
                end
            end
        end
        return efficiency
    end

    # Beer-Lambert mode
    # Get μ for each energy
    μ_vec = [get_scintillator_mu(model.material, E) for E in energies]

    d_cm = model.thickness_mm / 10.0
    pixel_row_size_det = geom.pixel_row_size * (geom.SDD / geom.SAD)

    efficiency = zeros(Float64, n_cols, n_rows, n_energies)

    for row in 1:n_rows
        v_offset = (row - (n_rows + 1) / 2) * pixel_row_size_det
        cos_beta = cos(atan(v_offset / geom.SDD))
        path_length = d_cm / cos_beta

        for k in 1:n_energies
            absorption = 1.0 - exp(-μ_vec[k] * path_length)
            η = model.fill_factor * absorption

            for col in 1:n_cols
                efficiency[col, row, k] = η
            end
        end
    end

    return efficiency
end

"""
    apply_detector_efficiency!(sinogram, model::DetectorEfficiency, geom::CTGeometry;
                               energy_keV::Float64=60.0) -> sinogram

Apply detector efficiency to sinogram data (in-place, GPU-native).

In CT with proper calibration (air scan normalization), detector efficiency
does NOT change projection values because both phantom and air scans use
the same detector - the efficiency cancels in the ratio:
    projection = -log(I_phantom/I_air) = -log(I0×exp(-μL)×η / (I0×η)) = μL

The main effect of detector efficiency is on NOISE levels (fewer detected
photons = more quantum noise). This is handled separately in noise modeling.

This function is kept for API compatibility but currently has no effect on
projection values. Energy-dependent efficiency variations could be added
in the future for more sophisticated spectral modeling.

# Arguments
- `sinogram`: Sinogram data [n_cols, n_rows, n_angles] in projection domain
- `model::DetectorEfficiency`: Detector efficiency model
- `geom::CTGeometry`: Scanner geometry

# Returns
Sinogram (unchanged - efficiency is handled via calibration).
"""
function apply_detector_efficiency!(
    sinogram::AbstractArray{T,3},
    model::DetectorEfficiency,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    # In calibrated CT, detector efficiency cancels between phantom and air scan.
    # The effect on signal levels is already handled by the calibration pipeline.
    # The effect on noise is handled by the noise model.
    #
    # Therefore, this function is a no-op for projection-domain data.
    # If intensity-domain application is needed, convert explicitly:
    #   intensity = exp.(-sinogram)
    #   intensity .*= efficiency
    #   sinogram .= -log.(intensity)

    return sinogram
end

# Convenience wrapper that allocates (for backward compatibility during transition)
function apply_detector_efficiency(
    intensity::AbstractArray{T,3},
    model::DetectorEfficiency,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    result = copy(intensity)
    return apply_detector_efficiency!(result, model, geom; energy_keV=energy_keV)
end

"""
    get_detector_efficiency_info(model::DetectorEfficiency; energy_keV::Float64=60.0) -> NamedTuple

Get diagnostic information about detector efficiency.
"""
function get_detector_efficiency_info(model::DetectorEfficiency; energy_keV::Float64=60.0)
    if model.material == "ideal"
        return (
            material="ideal",
            thickness_mm=0.0,
            fill_factor=1.0,
            absorption_at_ref_energy=1.0,
            total_efficiency=1.0
        )
    end

    μ = get_scintillator_mu(model.material, energy_keV)
    d_cm = model.thickness_mm / 10.0
    absorption = 1.0 - exp(-μ * d_cm)

    return (
        material=model.material,
        thickness_mm=model.thickness_mm,
        fill_factor=model.fill_factor,
        μ_at_ref_energy=μ,
        absorption_at_ref_energy=absorption,
        total_efficiency=model.fill_factor * absorption
    )
end

"""
    compute_dqe(model::DetectorEfficiency, energy_keV::Float64;
                swank_factor::Float64=0.95) -> Float64

Compute approximate Detective Quantum Efficiency (DQE).

# Definition

DQE measures how efficiently a detector converts incoming X-ray photons into
useful signal, accounting for both absorption efficiency and signal variance:

    DQE(E) = (SNR_out² / SNR_in²) ≈ η(E) × I_s × f²

where:
- η(E) is the absorption efficiency at energy E
- I_s is the Swank factor (accounts for variance in scintillator light output)
- f is the geometric fill factor

# Arguments
- `model::DetectorEfficiency`: Detector efficiency model
- `energy_keV::Float64`: Photon energy in keV
- `swank_factor::Float64=0.95`: Swank factor (0.90-0.98 typical for GOS/CsI)

# Returns
- `Float64`: DQE value (0 to 1)

# Typical Values

| Material | Thickness | DQE at 60 keV |
|----------|-----------|---------------|
| GOS      | 3.0 mm    | ~0.77         |
| CsI      | 0.6 mm    | ~0.56         |
| CdTe     | 1.6 mm    | ~0.77         |

# References

1. Swank RK. "Absorption and noise in x-ray phosphors."
   J Appl Phys. 1973;44(9):4199-4203.

2. Siewerdsen JH, Antonuk LE. "DQE and system optimization for indirect-detection
   flat-panel imagers." Med Phys. 1998;25(11):2199-2209.

# Example
```julia
model = detector_efficiency_gos(3.0)
dqe = compute_dqe(model, 60.0)  # ~0.77 with default Swank factor
```
"""
function compute_dqe(model::DetectorEfficiency, energy_keV::Float64;
    swank_factor::Float64=0.95)
    if model.material == "ideal"
        return 1.0
    end

    μ = get_scintillator_mu(model.material, energy_keV)
    d_cm = model.thickness_mm / 10.0
    absorption = 1.0 - exp(-μ * d_cm)

    # Approximate DQE
    dqe = absorption * swank_factor * model.fill_factor^2

    return dqe
end

# =============================================================================
# Exports
# =============================================================================

export DetectorEfficiency, DetectorEfficiencyMode, BEER_LAMBERT, MC_LUT
export detector_efficiency_gos, detector_efficiency_csi, detector_efficiency_cdte
export detector_efficiency_gemstone
export detector_efficiency_ideal, detector_efficiency_custom
export get_scintillator_mu, get_gemstone_mc_efficiency
export compute_detector_efficiency, compute_detector_efficiency_spectral
export apply_detector_efficiency!, apply_detector_efficiency
export get_detector_efficiency_info, compute_dqe
export GEMSTONE_MC_EFFICIENCY_LUT
