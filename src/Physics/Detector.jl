"""
    Physics/Detector.jl

Detector response modeling for CT systems.

Includes:
- Quantum detection efficiency (QDE) calculation
- Energy-dependent absorption in scintillator
"""

import XrayAttenuation as XA

"""
    compute_detector_efficiency(
        energies::Vector{Float64},
        material::XA.Material,
        thickness_mm::Float64
    )::Vector{Float64}

Compute quantum detection efficiency for each energy.

QDE is the probability that a photon of energy E interacts with the detector:
```
η(E) = 1 - exp(-μ(E) × t)
```

# Arguments
- `energies::Vector{Float64}` - X-ray energies in keV
- `material::XA.Material` - Scintillator material (e.g., CsI, Gd2O2S)
- `thickness_mm::Float64` - Detector thickness in mm

# Returns
- `Vector{Float64}` - Efficiency values [0, 1] for each energy

# Example
```julia
using XrayAttenuation
energies = collect(1.0:120.0)
mat = XA.Materials.csi
efficiency = compute_detector_efficiency(energies, mat, 0.5)  # 0.5mm CsI
```

# References
- Hsieh (2009) Section 4.2 - Detector physics
- Samei & Flynn (2003) Med Phys - Detector performance metrics
"""
function compute_detector_efficiency(
        energies::Vector{Float64},
        material::XA.Material,
        thickness_mm::Float64
    )::Vector{Float64}

    thickness_cm = thickness_mm / 10.0

    # Get energy-dependent mass attenuation coefficients
    efficiency = zeros(Float64, length(energies))
    for (i, E_keV) in enumerate(energies)
        # Get linear attenuation coefficient at this energy
        μ_cm = get_linear_attenuation(material, E_keV)

        # QDE: probability of interaction
        # η(E) = 1 - exp(-μ × t)
        efficiency[i] = 1.0 - exp(-μ_cm * thickness_cm)
    end

    return efficiency
end

# Export
export compute_detector_efficiency
