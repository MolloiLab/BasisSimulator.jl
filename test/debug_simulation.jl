"""
Debug simulation - check if materials produce contrast in sinogram
"""

using BasisSimulator
import XrayAttenuation as XA
using Statistics

println("="^70)
println("Simulation Debug: Testing Material Contrast")
println("="^70)

# Create small phantom for fast testing
println("\n1. Creating Gammex 472 phantom...")
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=160.0)  # Match scanner Z coverage!

println("  Phantom created: $(phantom.name)")
println("  Matrix: $(phantom.grid.nx) × $(phantom.grid.ny) × $(phantom.grid.nz)")

# Count materials
counts = count_materials(phantom)
println("\n2. Material distribution:")
for (mat, cnt) in sort(collect(counts), by=x->string(x[1]))
    println("  $mat: $cnt voxels")
end

# Scanner setup
println("\n3. Setting up scanner...")
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=60)  # Fewer projections
geometry = create_aquilion_one(protocol=protocol)

# Spectrum
println("\n4. Generating spectrum...")
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)
println("  Spectrum energies: $(minimum(spectrum.energies)) - $(maximum(spectrum.energies)) keV")

# Run simulation
println("\n5. Running forward simulation...")
println("  (This may take a minute with Gammex phantom...)")
sinogram = simulate_ct_scan(
    phantom = phantom,
    geometry = geometry,
    spectrum = spectrum,
    verbose = false  # Suppress progress messages
)

# Analyze sinogram
println("\n6. Sinogram Analysis:")
println("  Shape: $(size(sinogram))")
println("  Min: $(minimum(sinogram))")
println("  Max: $(maximum(sinogram))")
println("  Mean: $(mean(sinogram))")
println("  Std: $(std(sinogram))")
println("  Unique values: $(length(unique(sinogram)))")

# Check for contrast
has_contrast = std(sinogram) > 1e-6 && length(unique(sinogram)) > 100

println("\n7. Result:")
if has_contrast
    println("  ✅ SUCCESS: Sinogram shows contrast!")
    println("  Standard deviation: $(std(sinogram))")
    println("  Unique values: $(length(unique(sinogram)))")
else
    println("  ❌ PROBLEM: Sinogram has no contrast")
    println("  All values identical or nearly identical")
end

println("\n" * "="^70)
