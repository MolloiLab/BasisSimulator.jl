"""
Debug script to check attenuation calculations
"""

using BasisSimulator
import XrayAttenuation as XA

# Check attenuation coefficients for water at 60 keV
E = 60.0  # keV

# Water material from XA
water_mat = XA.Materials.water

# Get both mass and linear attenuation
μ_mass = get_mass_attenuation(water_mat, E)
μ_linear = get_linear_attenuation(water_mat, E)

println("="^70)
println("ATTENUATION COEFFICIENT CHECK")
println("="^70)
println("Energy: $E keV")
println("Material: Water")
println()
println("Mass attenuation μ/ρ: $μ_mass cm²/g")
println("Linear attenuation μ: $μ_linear cm^-1")
println()
println("Water density (from XA): $(ustrip(u"g/cm^3", water_mat.density)) g/cm³")
println()
println("Relationship: μ_linear = μ_mass × ρ")
println("Check: $(μ_mass) × $(ustrip(u"g/cm^3", water_mat.density)) = $(μ_mass * ustrip(u"g/cm^3", water_mat.density))")
println("Should equal μ_linear = $μ_linear")
println()

# Expected values for 33 cm of water
path_geom_cm = 33.0
path_rad_g_per_cm2 = path_geom_cm * 1.0  # density ≈ 1.0 g/cm³

println("="^70)
println("EXPECTED VALUES FOR 33 CM WATER")
println("="^70)
println("Geometric path: $path_geom_cm cm")
println("Radiological path (ρ×L): $path_rad_g_per_cm2 g/cm²")
println()
println("Using LINEAR attenuation (WRONG for density-weighted paths):")
println("  μ_linear × L_geom = $μ_linear × $path_geom_cm = $(μ_linear * path_geom_cm)")
println()
println("Using MASS attenuation (CORRECT for density-weighted paths):")
println("  μ_mass × L_rad = $μ_mass × $path_rad_g_per_cm2 = $(μ_mass * path_rad_g_per_cm2)")
println()
println("Expected from literature: ~6.8 cm^-1 equivalent")
println("="^70)
