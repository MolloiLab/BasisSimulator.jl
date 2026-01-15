"""
    Forward/DetectorEfficiency.jl

Detector absorption efficiency and DQE modeling for CT simulation.

The detector efficiency depends on:
1. Scintillator material (GOS, CsI, CdTe, etc.)
2. Scintillator thickness
3. Photon energy
4. Incidence angle (oblique rays travel longer paths)

Implementation follows CatSim/XCIST approach:
    η(E, θ) = 1 - exp(-μ(E) × d / cos(θ))

where:
- μ(E) is the energy-dependent linear attenuation coefficient
- d is the scintillator thickness
- θ is the incidence angle

GPU-native implementation using AcceleratedKernels.jl.
"""

import AcceleratedKernels as AK

# =============================================================================
# Scintillator Materials Database
# =============================================================================

# Linear attenuation coefficients (cm⁻¹) for common scintillator materials
# Data from NIST XCOM, organized as (energies, μ values)
const SCINTILLATOR_MU_DATA = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}(
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
    )
)

# Aliases
SCINTILLATOR_MU_DATA["Gadox"] = SCINTILLATOR_MU_DATA["GOS"]
SCINTILLATOR_MU_DATA["Gd2O2S"] = SCINTILLATOR_MU_DATA["GOS"]

# =============================================================================
# Detector Efficiency Types
# =============================================================================

"""
    DetectorEfficiency

Detector efficiency model specification.

# Fields
- `material`: Scintillator material name (GOS, CsI, CdTe, etc.)
- `thickness_mm`: Scintillator thickness in mm
- `fill_factor`: Fraction of detector area that is active (0-1)
"""
struct DetectorEfficiency
    material::String
    thickness_mm::Float64
    fill_factor::Float64
end

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
- `material`: Scintillator material (GOS, CsI, CdTe, CZT, Si)
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

Returns μ in cm⁻¹.
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

    # Get μ for the scintillator material
    μ = get_scintillator_mu(model.material, energy_keV)

    # Convert thickness from mm to cm
    d_cm = model.thickness_mm / 10.0

    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)
    efficiency = zeros(Float64, n_cols, n_rows)

    for row in 1:n_rows
        # Cone angle (beta in CatSim)
        v_offset = (row - (n_rows + 1) / 2) * pixel_size_det
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

    # Get μ for each energy
    μ_vec = [get_scintillator_mu(model.material, E) for E in energies]

    d_cm = model.thickness_mm / 10.0
    pixel_size_det = geom.pixel_size * (geom.SDD / geom.SAD)

    efficiency = zeros(Float64, n_cols, n_rows, n_energies)

    for row in 1:n_rows
        v_offset = (row - (n_rows + 1) / 2) * pixel_size_det
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
    apply_detector_efficiency!(intensity, model::DetectorEfficiency, geom::CTGeometry;
                               energy_keV::Float64=60.0) -> intensity

Apply detector efficiency to intensity-domain data (in-place, GPU-native).

This should be applied BEFORE adding noise, as it affects the number
of detected photons.

# Arguments
- `intensity`: Intensity data [n_cols, n_rows, n_angles]
- `model::DetectorEfficiency`: Detector efficiency model
- `geom::CTGeometry`: Scanner geometry

# Returns
Modified intensity scaled by detector efficiency.
"""
function apply_detector_efficiency!(
    intensity::AbstractArray{T,3},
    model::DetectorEfficiency,
    geom::CTGeometry;
    energy_keV::Float64=60.0
) where T
    if model.material == "ideal"
        return intensity
    end

    n_cols = size(intensity, 1)
    n_rows = size(intensity, 2)

    # Compute efficiency on CPU (done once)
    η_cpu = T.(compute_detector_efficiency(model, geom; energy_keV=energy_keV))

    # Transfer to GPU (same type as intensity)
    η = similar(intensity, n_cols, n_rows)
    copyto!(η, η_cpu)

    # GPU-native element-wise operation
    AK.foreachindex(intensity) do idx
        ci = CartesianIndices(intensity)[idx]
        col, row, _ = Tuple(ci)
        eff_idx = col + (row - 1) * n_cols
        intensity[idx] *= η[eff_idx]
    end

    return intensity
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
            material = "ideal",
            thickness_mm = 0.0,
            fill_factor = 1.0,
            absorption_at_ref_energy = 1.0,
            total_efficiency = 1.0
        )
    end

    μ = get_scintillator_mu(model.material, energy_keV)
    d_cm = model.thickness_mm / 10.0
    absorption = 1.0 - exp(-μ * d_cm)

    return (
        material = model.material,
        thickness_mm = model.thickness_mm,
        fill_factor = model.fill_factor,
        μ_at_ref_energy = μ,
        absorption_at_ref_energy = absorption,
        total_efficiency = model.fill_factor * absorption
    )
end

"""
    compute_dqe(model::DetectorEfficiency, energy_keV::Float64;
                swank_factor::Float64=0.95) -> Float64

Compute approximate Detective Quantum Efficiency (DQE).

DQE ≈ η × Swank factor × fill_factor²

The Swank factor accounts for variance in light output per absorbed X-ray.
Typical values: 0.90-0.98 for GOS/CsI.

# Returns
DQE value (0 to 1).
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

export DetectorEfficiency
export detector_efficiency_gos, detector_efficiency_csi, detector_efficiency_cdte
export detector_efficiency_ideal, detector_efficiency_custom
export get_scintillator_mu
export compute_detector_efficiency, compute_detector_efficiency_spectral
export apply_detector_efficiency!, apply_detector_efficiency
export get_detector_efficiency_info, compute_dqe
