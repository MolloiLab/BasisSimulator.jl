# =============================================================================
# GROUND TRUTH HU VALUES via XrayAttenuation.jl
# =============================================================================
#
# PURPOSE: Establish physics-based ground truth HU values for all phantom
#          materials at relevant kVp settings (80/100/120/140).
#
# SOURCE: XrayAttenuation.jl wraps NIST XCOM database and Geant4 validated data.
#         These are PHYSICS GROUND TRUTH, not empirical measurements.
#
# METHODOLOGY:
#   1. Load polychromatic spectrum at each kVp
#   2. Compute spectrum-weighted effective μ for each material:
#      μ_eff = Σ(w_E × μ(E)) / Σ(w_E)
#   3. Convert to HU: HU = 1000 × (μ_eff - μ_water) / μ_water
#
# VALIDATION:
#   - Water = 0 HU (by definition)
#   - Air ≈ -1000 HU
#   - Iodine series: monotonically increasing with concentration
#   - Calcium series: monotonically increasing with concentration
#
# USAGE:
#   Both CatSim and BasisSimulator MUST match these ground truth values
#   within tolerance. If CatSim disagrees with ground truth, investigate
#   CatSim - ground truth is the tiebreaker.
#
# UNCERTAINTY:
#   - NIST XCOM data uncertainty: ~2-3% for energies 10-100 keV
#   - Gammex material composition uncertainty: ~1-2%
#   - Spectrum uncertainty: depends on tube/filter calibration
#   - Combined HU uncertainty: ~±10-20 HU for tissue-like materials
#
# REFERENCES:
#   1. NIST XCOM: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
#   2. Berger MJ, et al. "XCOM: Photon Cross Sections Database" NIST 2010
#   3. XrayAttenuation.jl: https://github.com/JuliaHealth/XrayAttenuation.jl
#
# =============================================================================

using BasisSimulator
import XrayAttenuation as XA
using Statistics
using Printf

# =============================================================================
# MATERIAL DEFINITIONS
# =============================================================================

"""
Ground truth material list with region indices matching Phantom.jl
"""
const GROUND_TRUTH_MATERIALS = [
    # (region_id, symbol, description, concentration_mg_cc)
    (UInt8(0),  :air,         "Air",                           0.0),
    (UInt8(2),  :water,       "Water",                         0.0),
    (UInt8(3),  :solid_water, "Solid Water (phantom body)",    0.0),
    # Calcium series (ascending concentration)
    (UInt8(10), :Ca_50,       "Calcium 50 mg/cc",              50.0),
    (UInt8(11), :Ca_100,      "Calcium 100 mg/cc",             100.0),
    (UInt8(12), :Ca_200,      "Calcium 200 mg/cc",             200.0),
    (UInt8(13), :Ca_300,      "Calcium 300 mg/cc",             300.0),
    (UInt8(14), :Ca_400,      "Calcium 400 mg/cc",             400.0),
    (UInt8(15), :Ca_500,      "Calcium 500 mg/cc",             500.0),
    (UInt8(16), :Ca_600,      "Calcium 600 mg/cc",             600.0),
    # Iodine series (ascending concentration)
    (UInt8(20), :I_2_0,       "Iodine 2.0 mg/cc",              2.0),
    (UInt8(21), :I_2_5,       "Iodine 2.5 mg/cc",              2.5),
    (UInt8(22), :I_5_0,       "Iodine 5.0 mg/cc",              5.0),
    (UInt8(23), :I_7_5,       "Iodine 7.5 mg/cc",              7.5),
    (UInt8(24), :I_10_0,      "Iodine 10.0 mg/cc",             10.0),
    (UInt8(25), :I_15_0,      "Iodine 15.0 mg/cc",             15.0),
    (UInt8(26), :I_20_0,      "Iodine 20.0 mg/cc",             20.0),
]

# =============================================================================
# EXPECTED HU DATA STRUCTURE
# =============================================================================

"""
    ExpectedHU

Ground truth HU value for a single material at a single kVp.

# Fields
- `material_symbol::Symbol`: Material identifier (e.g., :Ca_100)
- `region_id::UInt8`: Region ID in phantom mask
- `kVp::Int`: Tube voltage
- `expected_hu::Float64`: Physics-computed expected HU
- `μ_eff::Float64`: Effective attenuation coefficient (cm⁻¹)
- `μ_water::Float64`: Water effective attenuation at same kVp (cm⁻¹)
- `uncertainty_hu::Float64`: Estimated uncertainty in HU (±)
- `description::String`: Human-readable description
"""
struct ExpectedHU
    material_symbol::Symbol
    region_id::UInt8
    kVp::Int
    expected_hu::Float64
    μ_eff::Float64
    μ_water::Float64
    uncertainty_hu::Float64
    description::String
end

# =============================================================================
# GROUND TRUTH COMPUTATION
# =============================================================================

"""
    compute_effective_mu(material, energies::Vector{Float64}, weights::Vector{Float64}) -> Float64

Compute spectrum-weighted effective linear attenuation coefficient.

μ_eff = Σ(w_E × μ(E)) / Σ(w_E)

This is the "thin sample" approximation - valid for HU prediction.
"""
function compute_effective_mu(material, energies::Vector{Float64}, weights::Vector{Float64})
    total_weight = sum(weights)
    μ_weighted_sum = 0.0

    for (E, w) in zip(energies, weights)
        μ = compute_μ_at_energy(material, E)
        μ_weighted_sum += w * μ
    end

    return μ_weighted_sum / total_weight
end

"""
    estimate_uncertainty(material_symbol::Symbol, expected_hu::Float64) -> Float64

Estimate HU uncertainty based on material type and expected value.

Sources of uncertainty:
- NIST XCOM data: ~2-3%
- Material composition: ~1-2%
- Spectrum calibration: ~1%
- Combined: ~3-5% relative, minimum ±10 HU

For high-Z materials (iodine), additional uncertainty from K-edge effects.
"""
function estimate_uncertainty(material_symbol::Symbol, expected_hu::Float64)
    # Base uncertainty (NIST + composition + spectrum)
    relative_uncertainty = 0.03  # 3%

    # Minimum absolute uncertainty
    min_uncertainty = 10.0  # HU

    # Higher uncertainty for materials with K-edges in diagnostic range
    if startswith(string(material_symbol), "I_")
        relative_uncertainty = 0.05  # 5% for iodine (K-edge at 33 keV)
    end

    # Compute uncertainty
    abs_uncertainty = abs(expected_hu) * relative_uncertainty
    return max(abs_uncertainty, min_uncertainty)
end

"""
    compute_ground_truth_hu(kVp::Int; n_bins::Int=60) -> Vector{ExpectedHU}

Compute physics-based ground truth HU values for all materials at specified kVp.

# Arguments
- `kVp::Int`: Tube voltage (80, 100, 120, or 140)
- `n_bins::Int`: Number of energy bins for spectrum integration (default 60)

# Returns
- `Vector{ExpectedHU}`: Ground truth HU for each material

# Note
Uses 60 bins by default for accuracy. Can be reduced for speed.
"""
function compute_ground_truth_hu(kVp::Int; n_bins::Int=60)
    # Load and downsample spectrum
    energies, weights = load_spectrum(kVp)
    energies, weights = downsample_spectrum(energies, weights, n_bins)

    # Compute water reference
    μ_water = compute_effective_mu(XA.Materials.water, energies, weights)

    results = ExpectedHU[]

    for (region_id, symbol, description, _) in GROUND_TRUTH_MATERIALS
        material = get_material(symbol)
        μ_eff = compute_effective_mu(material, energies, weights)

        # HU = 1000 × (μ - μ_water) / μ_water
        expected_hu = 1000.0 * (μ_eff - μ_water) / μ_water
        uncertainty = estimate_uncertainty(symbol, expected_hu)

        push!(results, ExpectedHU(
            symbol,
            region_id,
            kVp,
            expected_hu,
            μ_eff,
            μ_water,
            uncertainty,
            description
        ))
    end

    return results
end

# =============================================================================
# EXPECTED_HU DICTIONARY - THE MAIN EXPORT
# =============================================================================

"""
    EXPECTED_HU

Ground truth HU values for all materials at all kVp settings.

Structure: Dict{Int, Dict{Symbol, ExpectedHU}}
- First key: kVp (80, 100, 120, 140)
- Second key: material symbol (:water, :Ca_100, :I_10_0, etc.)

# Example
```julia
# Get expected HU for Ca-100 at 120 kVp
hu = EXPECTED_HU[120][:Ca_100].expected_hu

# Get all expected HU at 120 kVp
for (sym, entry) in EXPECTED_HU[120]
    println("\$sym: \$(round(entry.expected_hu, digits=1)) ± \$(round(entry.uncertainty_hu, digits=1)) HU")
end
```

# Physics Ground Truth
These values are computed from NIST/Geant4 data via XrayAttenuation.jl.
Both BasisSimulator and CatSim MUST match these values within tolerance.
If CatSim disagrees with ground truth, ground truth is the tiebreaker.
"""
const EXPECTED_HU = let
    result = Dict{Int, Dict{Symbol, ExpectedHU}}()

    for kVp in [80, 100, 120, 140]
        ground_truth = compute_ground_truth_hu(kVp)
        result[kVp] = Dict{Symbol, ExpectedHU}()
        for entry in ground_truth
            result[kVp][entry.material_symbol] = entry
        end
    end

    result
end

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

"""
    validate_water_hu() -> Bool

Verify that water = 0 HU at all kVp (sanity check).
"""
function validate_water_hu()
    for kVp in [80, 100, 120, 140]
        water_hu = EXPECTED_HU[kVp][:water].expected_hu
        if abs(water_hu) > 1e-10
            @warn "Water HU at $kVp kVp is $water_hu, expected exactly 0"
            return false
        end
    end
    return true
end

"""
    validate_air_hu() -> Bool

Verify that air ≈ -1000 HU at all kVp (within ±20 HU tolerance).
"""
function validate_air_hu()
    for kVp in [80, 100, 120, 140]
        air_hu = EXPECTED_HU[kVp][:air].expected_hu
        if abs(air_hu + 1000.0) > 20.0
            @warn "Air HU at $kVp kVp is $air_hu, expected -1000 ± 20"
            return false
        end
    end
    return true
end

"""
    validate_calcium_ordering() -> Bool

Verify that calcium HU is monotonically increasing with concentration.
Ca_50 < Ca_100 < Ca_200 < Ca_300 < Ca_400 < Ca_500 < Ca_600
"""
function validate_calcium_ordering()
    ca_symbols = [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]

    for kVp in [80, 100, 120, 140]
        hu_values = [EXPECTED_HU[kVp][sym].expected_hu for sym in ca_symbols]
        if !issorted(hu_values)
            @warn "Calcium HU not monotonic at $kVp kVp: $hu_values"
            return false
        end
    end
    return true
end

"""
    validate_iodine_ordering() -> Bool

Verify that iodine HU is monotonically increasing with concentration.
I_2_0 < I_2_5 < I_5_0 < I_7_5 < I_10_0 < I_15_0 < I_20_0
"""
function validate_iodine_ordering()
    i_symbols = [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]

    for kVp in [80, 100, 120, 140]
        hu_values = [EXPECTED_HU[kVp][sym].expected_hu for sym in i_symbols]
        if !issorted(hu_values)
            @warn "Iodine HU not monotonic at $kVp kVp: $hu_values"
            return false
        end
    end
    return true
end

"""
    validate_all_ground_truth() -> Bool

Run all validation checks on ground truth values.
"""
function validate_all_ground_truth()
    water_ok = validate_water_hu()
    air_ok = validate_air_hu()
    ca_ok = validate_calcium_ordering()
    i_ok = validate_iodine_ordering()

    all_ok = water_ok && air_ok && ca_ok && i_ok

    if all_ok
        println("✓ All ground truth validations passed")
    else
        println("✗ Ground truth validation FAILED")
    end

    return all_ok
end

# =============================================================================
# PRETTY PRINTING
# =============================================================================

"""
    print_expected_hu_table(kVp::Int)

Print formatted table of expected HU values at specified kVp.
"""
function print_expected_hu_table(kVp::Int)
    println()
    println("=" ^ 80)
    println("GROUND TRUTH HU VALUES ($kVp kVp)")
    println("Source: XrayAttenuation.jl (NIST XCOM / Geant4 validated data)")
    println("=" ^ 80)
    println()
    println("Material        | Region | Expected HU | Uncertainty | μ_eff (cm⁻¹)")
    println("-" ^ 80)

    # Order: air, water, solid_water, calcium series, iodine series
    order = [:air, :water, :solid_water,
             :Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600,
             :I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]

    for sym in order
        entry = EXPECTED_HU[kVp][sym]
        name = rpad(string(sym), 14)
        region = lpad(string(Int(entry.region_id)), 4)
        hu = lpad(@sprintf("%.1f", entry.expected_hu), 10)
        unc = lpad(@sprintf("±%.1f", entry.uncertainty_hu), 10)
        mu = @sprintf("%.4f", entry.μ_eff)
        println("  $name |   $region | $hu | $unc |   $mu")
    end

    println("-" ^ 80)
    println("μ_water = $(@sprintf("%.4f", EXPECTED_HU[kVp][:water].μ_water)) cm⁻¹")
    println()
end

"""
    print_all_kvp_summary()

Print summary table comparing HU values across all kVp settings.
"""
function print_all_kvp_summary()
    println()
    println("=" ^ 100)
    println("GROUND TRUTH HU SUMMARY - ALL kVp SETTINGS")
    println("=" ^ 100)
    println()
    println("Material        |    80 kVp    |   100 kVp    |   120 kVp    |   140 kVp")
    println("-" ^ 100)

    order = [:air, :water, :solid_water,
             :Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600,
             :I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]

    for sym in order
        name = rpad(string(sym), 14)
        values = []
        for kVp in [80, 100, 120, 140]
            hu = EXPECTED_HU[kVp][sym].expected_hu
            push!(values, lpad(@sprintf("%.1f", hu), 10))
        end
        println("  $name |   $(values[1]) |   $(values[2]) |   $(values[3]) |   $(values[4])")
    end

    println("-" ^ 100)
    println()
    println("Key observations:")
    println("  - HU values generally decrease with increasing kVp (beam hardening effect)")
    println("  - Iodine shows larger kVp dependence (K-edge at 33 keV)")
    println("  - Calcium shows moderate kVp dependence")
    println()
end

# =============================================================================
# RUN VALIDATION ON LOAD
# =============================================================================

# Run validation when this file is included
println("Computing ground truth HU values from XrayAttenuation.jl...")
println()

if !validate_all_ground_truth()
    error("Ground truth validation failed! Check material compositions.")
end

# Print summary
print_all_kvp_summary()

# =============================================================================
# EXPORTS
# =============================================================================

# Main export
export EXPECTED_HU, ExpectedHU

# Validation functions
export validate_water_hu, validate_air_hu
export validate_calcium_ordering, validate_iodine_ordering
export validate_all_ground_truth

# Printing functions
export print_expected_hu_table, print_all_kvp_summary

# Computation functions
export compute_ground_truth_hu, GROUND_TRUTH_MATERIALS
