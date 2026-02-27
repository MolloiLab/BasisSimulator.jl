"""
    Materials/MaterialAttenuation.jl

NIST XCOM linear attenuation coefficients μ(E) for common CT filter materials.

Provides energy-dependent linear attenuation coefficients (cm⁻¹) for:
- Aluminum (Al, ρ = 2.70 g/cm³)
- Copper (Cu, ρ = 8.96 g/cm³) — K-edge at 8.98 keV
- Tin (Sn, ρ = 7.287 g/cm³) — K-edge at 29.2 keV

Data source: NIST XCOM Photon Cross Sections Database
    https://www.nist.gov/pml/xcom-photon-cross-sections-database

Values are linear attenuation coefficients μ = (μ/ρ) × ρ in cm⁻¹,
computed from XCOM mass attenuation coefficients multiplied by material density.

# Usage
```julia
# Get μ for aluminum at 60 keV
μ = get_filter_mu("Al", 60.0)  # returns cm⁻¹

# Apply Beer-Lambert filter transmission
T = exp(-μ * thickness_cm)
```

# References
1. Hubbell JH, Seltzer SM. "Tables of X-Ray Mass Attenuation Coefficients."
   NIST Standard Reference Database 126.
   https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
"""

# =============================================================================
# Material Densities (g/cm³)
# =============================================================================

const MATERIAL_DENSITY = Dict{String,Float64}(
    "Al" => 2.70,
    "Cu" => 8.96,
    "Sn" => 7.287,
    "Ti" => 4.51,
    "graphite" => 1.70
)

# =============================================================================
# NIST XCOM Linear Attenuation Coefficients
# =============================================================================
#
# Format: (energies_keV, μ_cm⁻¹)
# Energy values converted from MeV to keV for consistency with spectrum data.
# μ values are linear attenuation coefficients in cm⁻¹ (already density-corrected).

const FILTER_MU_DATA = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}(
    # =========================================================================
    # Aluminum (Al), ρ = 2.70 g/cm³
    # =========================================================================
    "Al" => (
        # Energy (keV)
        [1.56, 2.0, 4.0, 6.0, 8.0, 8.98, 10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 29.2, 30.0, 32.0, 34.0,
            36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0,
            56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0,
            76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0,
            96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0,
            116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 134.0,
            136.0, 138.0, 140.0, 142.0, 144.0, 146.0, 148.0, 150.0],
        # μ (cm⁻¹)
        [1.067850e+04, 6.110100e+03, 9.733500e+02, 3.113100e+02, 1.358640e+02,
            9.698400e+01, 7.076700e+01, 4.141800e+01, 2.630880e+01, 1.777410e+01,
            1.260090e+01, 9.293400e+00, 7.082100e+00, 5.548500e+00, 4.452300e+00,
            3.647700e+00, 3.267000e+00, 3.045600e+00, 2.585520e+00, 2.227500e+00,
            1.944540e+00, 1.718280e+00, 1.534680e+00, 1.384560e+00, 1.260090e+00,
            1.155870e+00, 1.068390e+00, 9.938700e-01, 9.304200e-01, 8.753400e-01,
            8.278200e-01, 7.865100e-01, 7.500600e-01, 7.179300e-01, 6.895800e-01,
            6.642000e-01, 6.415200e-01, 6.212700e-01, 6.029100e-01, 5.864400e-01,
            5.713200e-01, 5.572800e-01, 5.448600e-01, 5.332500e-01, 5.224500e-01,
            5.127300e-01, 5.035500e-01, 4.949100e-01, 4.870800e-01, 4.797900e-01,
            4.727700e-01, 4.662900e-01, 4.600800e-01, 4.544100e-01, 4.490100e-01,
            4.438800e-01, 4.387500e-01, 4.341600e-01, 4.298400e-01, 4.255200e-01,
            4.214700e-01, 4.176900e-01, 4.139100e-01, 4.104000e-01, 4.071600e-01,
            4.039200e-01, 4.006800e-01, 3.977100e-01, 3.947400e-01, 3.917700e-01,
            3.890700e-01, 3.863700e-01, 3.839400e-01, 3.815100e-01, 3.790800e-01,
            3.766500e-01, 3.744900e-01, 3.720600e-01]
    ),

    # =========================================================================
    # Copper (Cu), ρ = 8.96 g/cm³ — K-edge at 8.98 keV
    # =========================================================================
    "Cu" => (
        # Energy (keV)
        [1.56, 2.0, 4.0, 6.0, 8.0, 8.98, 10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 29.2, 30.0, 32.0, 34.0,
            36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0,
            56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0,
            76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0,
            96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0,
            116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 134.0,
            136.0, 138.0, 140.0, 142.0, 144.0, 146.0, 148.0, 150.0],
        # μ (cm⁻¹)
        [3.589e+04, 1.930e+04, 3.112e+03, 1.036e+03, 4.708e+02,
            2.494e+03, 1.935e+03, 1.216e+03, 8.013e+02, 5.565e+02,
            4.039e+02, 3.028e+02, 2.328e+02, 1.828e+02, 1.461e+02,
            1.187e+02, 1.055e+02, 9.775e+01, 8.154e+01, 6.875e+01,
            5.857e+01, 5.031e+01, 4.357e+01, 3.800e+01, 3.336e+01,
            2.948e+01, 2.621e+01, 2.341e+01, 2.102e+01, 1.896e+01,
            1.718e+01, 1.563e+01, 1.427e+01, 1.308e+01, 1.203e+01,
            1.110e+01, 1.027e+01, 9.533e+00, 8.877e+00, 8.284e+00,
            7.749e+00, 7.270e+00, 6.836e+00, 6.441e+00, 6.080e+00,
            5.753e+00, 5.454e+00, 5.180e+00, 4.929e+00, 4.699e+00,
            4.486e+00, 4.290e+00, 4.109e+00, 3.939e+00, 3.784e+00,
            3.639e+00, 3.505e+00, 3.378e+00, 3.261e+00, 3.151e+00,
            3.049e+00, 2.953e+00, 2.863e+00, 2.778e+00, 2.698e+00,
            2.623e+00, 2.553e+00, 2.486e+00, 2.424e+00, 2.363e+00,
            2.307e+00, 2.255e+00, 2.204e+00, 2.156e+00, 2.110e+00,
            2.067e+00, 2.026e+00, 1.986e+00]
    ),

    # =========================================================================
    # Tin (Sn), ρ = 7.287 g/cm³ — K-edge at 29.2 keV
    # =========================================================================
    "Sn" => (
        # Energy (keV)
        [1.56, 2.0, 4.0, 6.0, 8.0, 8.98, 10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 29.2, 30.0, 32.0, 34.0,
            36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0,
            56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0,
            76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0,
            96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0,
            116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 134.0,
            136.0, 138.0, 140.0, 142.0, 144.0, 146.0, 148.0, 150.0],
        # μ (cm⁻¹)
        [2.189e+04, 1.213e+04, 6.846e+03, 3.858e+03, 1.822e+03,
            1.340e+03, 1.009e+03, 6.173e+02, 4.086e+02, 2.853e+02,
            2.075e+02, 1.564e+02, 1.207e+02, 9.547e+01, 7.701e+01,
            6.319e+01, 5.654e+01, 3.003e+02, 2.591e+02, 2.218e+02,
            1.899e+02, 1.633e+02, 1.416e+02, 1.239e+02, 1.094e+02,
            9.721e+01, 8.688e+01, 7.799e+01, 7.025e+01, 6.351e+01,
            5.761e+01, 5.242e+01, 4.786e+01, 4.382e+01, 4.021e+01,
            3.700e+01, 3.415e+01, 3.159e+01, 2.928e+01, 2.721e+01,
            2.532e+01, 2.361e+01, 2.207e+01, 2.066e+01, 1.937e+01,
            1.820e+01, 1.712e+01, 1.612e+01, 1.521e+01, 1.438e+01,
            1.360e+01, 1.288e+01, 1.222e+01, 1.161e+01, 1.103e+01,
            1.050e+01, 9.999e+00, 9.541e+00, 9.107e+00, 8.704e+00,
            8.330e+00, 7.973e+00, 7.644e+00, 7.331e+00, 7.041e+00,
            6.767e+00, 6.507e+00, 6.262e+00, 6.028e+00, 5.814e+00,
            5.607e+00, 5.411e+00, 5.227e+00, 5.052e+00, 4.885e+00,
            4.729e+00, 4.582e+00, 4.437e+00]
    ),

    # =========================================================================
    # Titanium (Ti), ρ = 4.51 g/cm³ — K-edge at 4.97 keV
    # (retained from existing BOWTIE_MU_DATA, coarser grid)
    # =========================================================================
    "Ti" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [17.0, 5.75, 2.74, 1.56, 1.02, 0.54, 0.37, 0.30, 0.25]
    ),

    # =========================================================================
    # Graphite/Carbon (ρ = 1.70 g/cm³)
    # (retained from existing BOWTIE_MU_DATA, coarser grid)
    # =========================================================================
    "graphite" => (
        [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0],
        [0.56, 0.31, 0.27, 0.26, 0.26, 0.25, 0.24, 0.24, 0.23]
    )
)

# Alias for carbon
FILTER_MU_DATA["C"] = FILTER_MU_DATA["graphite"]

# Long-form name aliases (Scanner symbols like :aluminum → String → "aluminum")
FILTER_MU_DATA["aluminum"] = FILTER_MU_DATA["Al"]
FILTER_MU_DATA["copper"] = FILTER_MU_DATA["Cu"]
FILTER_MU_DATA["tin"] = FILTER_MU_DATA["Sn"]
FILTER_MU_DATA["titanium"] = FILTER_MU_DATA["Ti"]
FILTER_MU_DATA["carbon"] = FILTER_MU_DATA["graphite"]

# =============================================================================
# Interpolation Functions
# =============================================================================

"""
    get_filter_mu(material::String, energy_keV::Float64) -> Float64

Get linear attenuation coefficient for a filter material at given energy.

Uses log-log interpolation for accurate results across the wide dynamic range
of attenuation coefficients. Handles K-edge discontinuities via piecewise
interpolation between tabulated data points.

# Arguments
- `material`: Material name ("Al", "Cu", "Sn", "Ti", "graphite", "C")
- `energy_keV`: Photon energy in keV (clamped to data range)

# Returns
Linear attenuation coefficient μ in cm⁻¹.

# Example
```julia
μ_al_60 = get_filter_mu("Al", 60.0)   # ~0.75 cm⁻¹
μ_cu_60 = get_filter_mu("Cu", 60.0)   # ~14.3 cm⁻¹
μ_sn_60 = get_filter_mu("Sn", 60.0)   # ~47.9 cm⁻¹
```
"""
function get_filter_mu(material::String, energy_keV::Float64)
    if !haskey(FILTER_MU_DATA, material)
        error("Unknown filter material: '$material'. Available: $(join(sort(collect(keys(FILTER_MU_DATA))), ", "))")
    end

    energies, mus = FILTER_MU_DATA[material]
    E = clamp(energy_keV, energies[1], energies[end])

    # Log-log interpolation (μ varies as power law between K-edges)
    log_E = log(E)
    log_energies = log.(energies)
    log_mus = log.(mus)

    # Find interpolation interval
    idx = 1
    for i in 1:(length(energies)-1)
        if log_E >= log_energies[i] && log_E <= log_energies[i+1]
            idx = i
            break
        end
    end
    # Handle case where E is at the last point
    if log_E >= log_energies[end]
        idx = length(energies) - 1
    end

    # Linear interpolation in log-log space
    t = (log_E - log_energies[idx]) / (log_energies[idx+1] - log_energies[idx])
    log_mu = log_mus[idx] + t * (log_mus[idx+1] - log_mus[idx])

    return exp(log_mu)
end

"""
    get_filter_mu(material::String, energies::Vector{Float64}) -> Vector{Float64}

Vectorized version: get μ at multiple energies at once.

# Example
```julia
E = [40.0, 60.0, 80.0, 100.0, 120.0]
μ_al = get_filter_mu("Al", E)
```
"""
function get_filter_mu(material::String, energies::Vector{Float64})
    return [get_filter_mu(material, E) for E in energies]
end

# =============================================================================
# Note: get_bowtie_mu / get_bowtie_mu_reference are defined in bowtie_filter.jl
# (using bowtie_filter.jl's own NIST data). Do NOT alias here to avoid
# 'Method overwriting' precompilation errors.
# =============================================================================

# =============================================================================
# Utility Functions
# =============================================================================

"""
    list_filter_materials() -> Vector{String}

List all available filter materials.
"""
function list_filter_materials()
    return sort(collect(keys(FILTER_MU_DATA)))
end

"""
    get_material_density(material::String) -> Float64

Get material density in g/cm³.
"""
function get_material_density(material::String)
    if !haskey(MATERIAL_DENSITY, material)
        error("Unknown material: '$material'")
    end
    return MATERIAL_DENSITY[material]
end

# =============================================================================
# Exports
# =============================================================================

export FILTER_MU_DATA, MATERIAL_DENSITY
export get_filter_mu
export list_filter_materials, get_material_density
