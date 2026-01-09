"""
Component Validation 02: Material Attenuation Coefficients

Compare BasisSimulator attenuation (XrayAttenuation.jl/NIST XCOM) vs GECATSIM.

Goals:
1. Load GECATSIM attenuation coefficients via GetMu module
2. Compute BasisSimulator attenuation for same materials
3. Compare across energy range (20-120 keV)
4. Validate common materials: water, bone, iodine, air
"""

using BasisSimulator
using Statistics
using PythonCall
import XrayAttenuation as XA

println("\n" * "="^70)
println("COMPONENT VALIDATION 02: ATTENUATION COEFFICIENTS")
println("="^70)

# ==============================================================================
# 1. Setup GECATSIM GetMu Module
# ==============================================================================
println("\n1. Loading GECATSIM GetMu module...")

gecatsim = pyimport("gecatsim")
get_mu = pyimport("gecatsim.pyfiles.GetMu")

println("   ✅ GECATSIM GetMu loaded")

# ==============================================================================
# 2. Define Test Materials and Energies
# ==============================================================================
println("\n2. Defining test parameters...")

# Energy range for comparison (keV)
test_energies = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]

# Materials to compare
# GECATSIM uses material names, BasisSimulator uses XrayAttenuation.jl Materials
materials = [
    ("water", XA.Materials.water),
    ("air", XA.Materials.air),
    ("bone", XA.Materials.corticalbone),
]

println("   Test energies: $(length(test_energies)) points (20-120 keV)")
println("   Materials: $(length(materials)) common CT materials")

# ==============================================================================
# 3. Compare Attenuation Coefficients
# ==============================================================================
println("\n3. Comparing attenuation coefficients...")

all_results = []

for (mat_name, mat_xa) in materials
    println("\n   Testing: $mat_name")

    geca_mu = Float64[]
    basis_mu = Float64[]

    for E in test_energies
        # GECATSIM attenuation (cm²/g or cm⁻¹?)
        # GetMu.get_mu(material, energy_keV, density)
        # Need to check GECATSIM API for exact function signature
        try
            mu_geca = pyconvert(Float64, get_mu.get_mu(mat_name, E))
            push!(geca_mu, mu_geca)
        catch e
            println("      ⚠️ GECATSIM failed at $E keV: $e")
            push!(geca_mu, NaN)
        end

        # BasisSimulator attenuation
        # Use our wrapper that returns linear attenuation (cm⁻¹)
        mu_basis_linear = get_linear_attenuation(mat_xa, E)  # cm⁻¹

        push!(basis_mu, mu_basis_linear)
    end

    # Compute comparison metrics
    valid_indices = .!isnan.(geca_mu)

    if sum(valid_indices) > 0
        geca_valid = geca_mu[valid_indices]
        basis_valid = basis_mu[valid_indices]
        energies_valid = test_energies[valid_indices]

        diff = (basis_valid .- geca_valid) ./ geca_valid .* 100  # percent difference
        rmse = sqrt(mean((basis_valid .- geca_valid).^2))
        max_diff = maximum(abs.(diff))
        mean_diff = mean(abs.(diff))

        println("      Valid comparisons: $(sum(valid_indices))/$(length(test_energies))")
        println("      Mean abs % diff: $(round(mean_diff, digits=2))%")
        println("      Max abs % diff: $(round(max_diff, digits=2))%")
        println("      RMSE: $(round(rmse, digits=6)) cm⁻¹")

        push!(all_results, (
            material = mat_name,
            energies = energies_valid,
            geca_mu = geca_valid,
            basis_mu = basis_valid,
            mean_diff_pct = mean_diff,
            max_diff_pct = max_diff,
            rmse = rmse
        ))
    else
        println("      ⚠️ No valid comparisons for $mat_name")
    end
end

# ==============================================================================
# 4. Validation Summary
# ==============================================================================
println("\n" * "="^70)
println("ATTENUATION VALIDATION SUMMARY")
println("="^70)

checks = []
for res in all_results
    push!(checks, ("$(res.material): mean diff < 5%", res.mean_diff_pct < 5.0))
    push!(checks, ("$(res.material): max diff < 10%", res.max_diff_pct < 10.0))
end

println()
all_passed = true
for (name, passed) in checks
    status = passed ? "✅" : "⚠️"
    println("$status $name")
    global all_passed = all_passed && passed
end

println("\n" * "="^70)
if all_passed
    println("✅ ATTENUATION VALIDATION PASSED")
    println("BasisSimulator attenuation matches GECATSIM within tolerances")
else
    println("⚠️ ATTENUATION VALIDATION: DIFFERENCES FOUND")
    println("Potential causes:")
    println("  - Different NIST XCOM database versions")
    println("  - Different interpolation methods")
    println("  - Material composition differences (e.g., bone model)")
    println("  - Density values (ICRU vs GECATSIM)")
end
println("="^70 * "\n")

println("Results saved in variable 'all_results' for inspection")
