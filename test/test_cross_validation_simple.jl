"""
Simple cross-validation between BasisSimulator and GECATSIM.

Compares sinogram outputs from both simulators for a water cylinder phantom.
"""

using BasisSimulator
using Statistics
using PythonCall

println("\n" * "="^70)
println("BASISSIMULATOR vs GECATSIM CROSS-VALIDATION")
println("="^70)

# ==============================================================================
# 1. Read GECATSIM Output
# ==============================================================================
println("\n1. Reading GECATSIM sinogram data...")

gecatsim = pyimport("gecatsim")
gecatsim_file = joinpath(@__DIR__, "cfg_gecatsim", "test_output.scan")

if !isfile(gecatsim_file)
    error("GECATSIM output not found: $gecatsim_file\nRun test_gecatsim_simple_run.jl first")
end

# GECATSIM output dimensions: n_rows × n_cols × n_angles
# Based on our configuration: 320 × 800 × 360
n_rows = 320
n_cols = 800
n_angles = 360

# Read raw data
gecatsim_data = gecatsim.rawread(
    gecatsim_file,
    pylist([n_angles, n_rows, n_cols]),
    "float"
)

# Convert to Julia array and reorder dimensions
gecatsim_sino = pyconvert(Array{Float64}, gecatsim_data)
# Transpose to match BasisSimulator format: rows × cols × angles
gecatsim_sino = permutedims(gecatsim_sino, (2, 3, 1))

println("   ✅ GECATSIM data loaded")
println("      Shape: $(size(gecatsim_sino))")
println("      Range: $(round(minimum(gecatsim_sino), sigdigits=3)) - $(round(maximum(gecatsim_sino), sigdigits=3))")
println("      Mean: $(round(mean(gecatsim_sino), sigdigits=3))")

# ==============================================================================
# 2. Run BasisSimulator
# ==============================================================================
println("\n2. Running BasisSimulator simulation...")

# Create water cylinder phantom (scaled to match GECATSIM W20)
phantom = create_water_cylinder(
    diameter_mm=100.0,  # ~10cm diameter (W20 scaled 0.5)
    height_mm=100.0,    # taller to ensure coverage
    resolution_mm=2.0
)

# Generate spectrum (120 kVp, 200 mA)
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

# Create geometry matching GECATSIM
protocol = ScanProtocol(
    kVp=120.0,
    mAs=200.0,
    scan_fov_mm=200.0,
    num_projections=360
)

geometry = create_aquilion_one(protocol=protocol)

println("   Running forward simulation...")
detector_signal = simulate_ct_scan(
    phantom=phantom,
    geometry=geometry,
    spectrum=spectrum
)

println("   ✅ BasisSimulator simulation complete")
println("      Shape: $(size(detector_signal))")
println("      Range: $(round(minimum(detector_signal), sigdigits=3)) - $(round(maximum(detector_signal), sigdigits=3))")
println("      Mean: $(round(mean(detector_signal), sigdigits=3))")

# ==============================================================================
# 3. Normalize and Compare
# ==============================================================================
println("\n3. Comparing sinograms...")

# GECATSIM and BasisSimulator may have different detector sizes
# Extract overlapping region for comparison
basis_rows, basis_cols, basis_angles = size(detector_signal)
geca_rows, geca_cols, geca_angles = size(gecatsim_sino)

# Use smaller dimensions for comparison
compare_rows = min(basis_rows, geca_rows)
compare_cols = min(basis_cols, geca_cols)
compare_angles = min(basis_angles, geca_angles)

println("   Comparison region: $compare_rows × $compare_cols × $compare_angles")

# Extract central regions
basis_crop = detector_signal[1:compare_rows, 1:compare_cols, 1:compare_angles]
geca_crop = gecatsim_sino[1:compare_rows, 1:compare_cols, 1:compare_angles]

# Normalize both to [0,1] for comparison
basis_norm = (basis_crop .- minimum(basis_crop)) ./ (maximum(basis_crop) - minimum(basis_crop))
geca_norm = (geca_crop .- minimum(geca_crop)) ./ (maximum(geca_crop) - minimum(geca_crop))

# Compute metrics
diff = basis_norm .- geca_norm
rmse = sqrt(mean(diff.^2))
mae = mean(abs.(diff))
max_diff = maximum(abs.(diff))
correlation = cor(vec(basis_norm), vec(geca_norm))

println("\n   Comparison Metrics:")
println("      RMSE (normalized): $(round(rmse, digits=4))")
println("      MAE (normalized):  $(round(mae, digits=4))")
println("      Max diff:          $(round(max_diff, digits=4))")
println("      Correlation:       $(round(correlation, digits=4))")

# ==============================================================================
# 4. Profile Comparison
# ==============================================================================
println("\n4. Comparing central profiles...")

# Get central slice
center_angle = div(compare_angles, 2)
basis_profile = basis_norm[:, div(compare_cols, 2), center_angle]
geca_profile = geca_norm[:, div(compare_cols, 2), center_angle]

profile_diff = basis_profile .- geca_profile
profile_rmse = sqrt(mean(profile_diff.^2))
profile_corr = cor(basis_profile, geca_profile)

println("   Central profile (row direction):")
println("      RMSE: $(round(profile_rmse, digits=4))")
println("      Correlation: $(round(profile_corr, digits=4))")

# ==============================================================================
# 5. Validation Summary
# ==============================================================================
println("\n" * "="^70)
println("CROSS-VALIDATION SUMMARY")
println("="^70)

checks = [
    ("GECATSIM data loaded", true),
    ("BasisSimulator completed", true),
    ("Correlation > 0.9", correlation > 0.9),
    ("RMSE < 0.1", rmse < 0.1),
    ("Profile match good", profile_corr > 0.9)
]

println()
all_passed = true
for (name, passed) in checks
    status = passed ? "✅" : "⚠️"
    println("$status $name")
    global all_passed = all_passed && passed
end

println("\n" * "="^70)
if all_passed
    println("✅ CROSS-VALIDATION SUCCESSFUL")
    println("BasisSimulator and GECATSIM show good agreement")
else
    println("⚠️  PARTIAL AGREEMENT")
    println("Differences may be due to:")
    println("  - Different phantom geometries (GECATSIM W20 vs custom)")
    println("  - Spectrum differences (filtration, energy binning)")
    println("  - Detector response models")
    println("  - Numerical precision")
end
println("="^70 * "\n")

# Save results
results = (
    gecatsim_sino = gecatsim_sino,
    basis_signal = detector_signal,
    rmse = rmse,
    mae = mae,
    correlation = correlation,
    profile_correlation = profile_corr
)

println("Results saved in variable 'results' for inspection")
