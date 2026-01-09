"""
Baseline working simulation extracted from old ct_simulator_final.jl
This verifies the old approach works, then we'll port to module structure
"""

using BasisSimulator
import XrayAttenuation as XA
using CairoMakie
using Statistics

println("="^70)
println("BASELINE TEST: Old Working Simulation Approach")
println("="^70)

# Use BasisSimulator's existing infrastructure
println("\n1. Creating phantom...")
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=160.0)

println("\n2. Setting up scanner...")
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=180)
geometry = create_aquilion_one(protocol=protocol)

println("\n3. Generating spectrum...")
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

println("\n4. Running simulation...")
# The KEY DIFFERENCE: Old notebook used photon counts with detector efficiency
# and applied -log transform INSIDE simulation to get line integrals directly

# For now, let's test with current simulate_ct_scan and see what we get
sinogram_raw = simulate_ct_scan(
    phantom = phantom,
    geometry = geometry,
    spectrum = spectrum,
    verbose = true
)

println("\n5. Checking sinogram...")
println("  Shape: $(size(sinogram_raw))")
println("  Range: $(minimum(sinogram_raw)) to $(maximum(sinogram_raw))")
println("  Unique values: $(length(unique(sinogram_raw)))")

# Convert to line integrals
I0 = estimate_air_scan(spectrum)
println("  I0: $I0")

sinogram = convert_to_attenuation(sinogram_raw, I0)
println("  After -log transform: $(minimum(sinogram)) to $(maximum(sinogram))")

# Expected values for Gammex phantom (33cm diameter, water + inserts):
# Max path through water: ~33 cm
# μ_water ≈ 0.206 cm^-1 at 60 keV
# Expected max: ~6.8 cm^-1
println("  Expected max (33cm water): ~6.8 cm^-1")

if maximum(sinogram) > 20.0
    println("  ❌ SINOGRAM TOO HIGH! Something wrong in simulation.")
else
    println("  ✅ Sinogram in reasonable range")
end

println("\n6. Reconstructing...")
recon_fov_cm = 35.0
recon_size = 256
z_thickness_cm = 16.0

x_coords = collect(range(-recon_fov_cm/2, recon_fov_cm/2, length=recon_size))
y_coords = collect(range(-recon_fov_cm/2, recon_fov_cm/2, length=recon_size))
z_coords = collect(range(-z_thickness_cm/2, z_thickness_cm/2, length=64))

reconstruction = reconstruct_fdk(
    sinogram,
    geometry.SAD_cm,
    geometry.SDD_cm,
    geometry.pixel_width_cm,
    geometry.pixel_height_cm,
    rad2deg.(geometry.angles),
    x_coords,
    y_coords,
    z_coords
)

println("  Reconstruction μ: $(minimum(reconstruction)) to $(maximum(reconstruction)) cm^-1")
println("  Expected μ_water: ~0.2 cm^-1 (with 2x FDK error: ~0.4)")

# Convert to HU
μ_water = get_linear_attenuation(XA.Materials.water, 60.0)
reconstruction_hu = convert_to_hounsfield_units(reconstruction, μ_water)

println("  HU range: $(minimum(reconstruction_hu)) to $(maximum(reconstruction_hu))")
println("  Expected HU: -1000 to +3000")

# Assess results
if abs(median(reconstruction) - μ_water) / μ_water < 2.0  # Within 2x
    println("\n✅ RECONSTRUCTION REASONABLE!")
else
    println("\n❌ RECONSTRUCTION BROKEN!")
    println("   Median μ: $(median(reconstruction)) cm^-1")
    println("   vs expected: $μ_water cm^-1")
    println("   Factor off: $(median(reconstruction) / μ_water)x")
end

println("="^70)
