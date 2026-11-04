# =============================================================================
# The x-ray beam, as a dosimeter sees it
# =============================================================================
# A beam's quality is quoted as its half-value layer (HVL): the thickness of aluminium that
# halves its air kerma. Air kerma weights each photon by its energy times air's mass
# energy-absorption coefficient μ_en/ρ, tabulated by NIST (Hubbell & Seltzer, NIST Standard
# Reference Database 126, Table 4, "Air, Dry (near sea level)"). XrayAttenuation (XCOM) gives
# the aluminium's attenuation.
# =============================================================================

const AIR_ENERGIES_KEV = [
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    80.0,
    100.0,
    150.0,
    200.0,
]
const AIR_MU_EN_CM2_G = [
    39.31, 22.7, 9.446, 4.742, 1.334, 0.5389, 0.1537, 0.06833, 0.04098, 0.03041, 0.02407,
    0.02325, 0.02496, 0.02672,
]

"""
    air_mu_en(energy_keV) -> μ_en/ρ of dry air in cm²/g

NIST's mass energy-absorption coefficient of dry air, interpolated log–log (5–200 keV).
"""
function air_mu_en(energy_keV::Real)
    x = log(clamp(energy_keV, first(AIR_ENERGIES_KEV), last(AIR_ENERGIES_KEV)))
    xs = log.(AIR_ENERGIES_KEV)
    k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    f = (x - xs[k]) / (xs[k + 1] - xs[k])
    return exp((1 - f) * log(AIR_MU_EN_CM2_G[k]) + f * log(AIR_MU_EN_CM2_G[k + 1]))
end

"""
    air_kerma(energies_keV, weights; filter_mm = 0.0, material = DetectorMaterial(:Al))

The air kerma of a beam with `weights` photons at `energies_keV` (arbitrary units), after
`filter_mm` of `material`.
"""
function air_kerma(
        energies_keV::AbstractVector,
        weights::AbstractVector;
        filter_mm::Real = 0.0,
        material::DetectorMaterial = DetectorMaterial(:Al),
    )
    return sum(
        w * E * air_mu_en(E) * exp(-linear_attenuation(material, E) * filter_mm)
            for (E, w) in zip(energies_keV, weights) if w > 0 && E >= 1
    )
end

"""
    half_value_layer(energies_keV, weights; material = DetectorMaterial(:Al)) -> mm

The thickness of `material` (aluminium by default) that halves the beam's air kerma.
"""
function half_value_layer(
        energies_keV::AbstractVector,
        weights::AbstractVector;
        material::DetectorMaterial = DetectorMaterial(:Al),
    )
    target = air_kerma(energies_keV, weights; material) / 2
    low, high = 0.0, 50.0
    for _ in 1:60
        mid = (low + high) / 2
        if air_kerma(energies_keV, weights; filter_mm = mid, material) > target
            low = mid
        else
            high = mid
        end
    end
    return (low + high) / 2
end
