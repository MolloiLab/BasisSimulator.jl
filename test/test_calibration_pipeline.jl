"""
Test Calibration Pipeline (Phase 1)

Tests calibration functions: air scan, offset scan, gain/offset correction, log transform.
Validates on both CPU and GPU (Metal).
"""

using BasisSimulator
using Test
using Statistics

# Check if Metal is available
HAS_METAL = try
    using Metal
    Metal.functional()
catch
    false
end

println("Metal available: $HAS_METAL")

# Create test geometry
geom = create_aquilion_one(n_angles=90, n_rows=16, n_cols=128, fov_cm=35.0)

# Create test phantom
phantom = create_gammex_472(n_voxels=64, fov_cm=35.0, z_cm=4.0)

# Create test data on CPU
println("\n=== Forward Projection ===")
sinogram_cpu = siddon_forward_project(Float32.(phantom.μ), geom)
println("Sinogram size: $(size(sinogram_cpu))")
println("Sinogram range: [$(minimum(sinogram_cpu)), $(maximum(sinogram_cpu))]")

# =============================================================================
# Test 1: Air Scan Simulation
# =============================================================================
println("\n=== Test 1: Air Scan Simulation ===")

air = simulate_air_scan(geom)
println("Air scan size: $(size(air))")
println("Air scan range: [$(minimum(air)), $(maximum(air))]")

# Air scan with no physics should be all ones
@test isapprox(mean(air), 1.0, rtol=1e-6)
@test size(air) == (geom.n_cols, geom.n_rows, geom.n_angles)
println("Air scan (no physics): PASS")

# =============================================================================
# Test 2: Offset Scan Simulation
# =============================================================================
println("\n=== Test 2: Offset Scan Simulation ===")

# Offset with no noise
offset_zero = simulate_offset_scan(geom; electronic_noise_sigma=0.0)
@test isapprox(mean(offset_zero), 0.0, atol=1e-6)
println("Offset scan (no noise): PASS")

# Offset with noise
offset_noisy = simulate_offset_scan(geom; electronic_noise_sigma=10.0)
@test abs(mean(offset_noisy)) < 2.0  # Mean should be close to 0
@test std(offset_noisy) > 5.0  # Should have significant variance
println("Offset scan (with noise): std=$(round(std(offset_noisy), digits=2)), PASS")

# =============================================================================
# Test 3: Calibration Correction
# =============================================================================
println("\n=== Test 3: Calibration Correction ===")

# Create intensity data (exp of negative sinogram)
intensity = exp.(-sinogram_cpu)
println("Intensity range: [$(minimum(intensity)), $(maximum(intensity))]")

# Apply calibration with scalar offset
calibrated = apply_calibration(intensity, air, 0.0f0)
println("Calibrated range: [$(minimum(calibrated)), $(maximum(calibrated))]")

# Calibrated should be same as intensity when air=1, offset=0
@test isapprox(mean(calibrated), mean(intensity), rtol=1e-6)
println("Calibration (scalar offset): PASS")

# Apply calibration with array offset
offset_array = zeros(Float32, size(air))
calibrated2 = apply_calibration(intensity, air, offset_array)
@test isapprox(mean(calibrated2), mean(intensity), rtol=1e-6)
println("Calibration (array offset): PASS")

# =============================================================================
# Test 4: Log Transform
# =============================================================================
println("\n=== Test 4: Log Transform ===")

# Apply log transform to calibrated intensity
sino_from_log = apply_log_transform(copy(intensity))
println("Log-transformed range: [$(minimum(sino_from_log)), $(maximum(sino_from_log))]")

# Should match original sinogram (since intensity = exp(-sinogram))
@test isapprox(mean(sino_from_log), mean(sinogram_cpu), rtol=0.01)
println("Log transform: PASS")

# =============================================================================
# Test 5: Full Calibration Pipeline
# =============================================================================
println("\n=== Test 5: Full Calibration Pipeline ===")

# Start from intensity and apply full pipeline
intensity_copy = copy(intensity)
sinogram_calibrated = calibrate_sinogram!(intensity_copy, air, 0.0f0)
println("Calibrated sinogram range: [$(minimum(sinogram_calibrated)), $(maximum(sinogram_calibrated))]")

# Should match original sinogram
@test isapprox(mean(sinogram_calibrated), mean(sinogram_cpu), rtol=0.01)
println("Full pipeline: PASS")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    intensity_gpu = MtlArray(exp.(-sinogram_cpu))
    air_gpu = MtlArray(air)
    offset_gpu = MtlArray(zeros(Float32, size(air)))
    println("GPU array type: $(typeof(intensity_gpu))")

    # Test calibration on GPU
    println("\nCalibration on GPU...")
    calibrated_gpu = apply_calibration(copy(intensity_gpu), air_gpu, 0.0f0)
    calibrated_gpu_result = Array(calibrated_gpu)

    cpu_mean = mean(calibrated)
    gpu_mean = mean(calibrated_gpu_result)
    println("  CPU mean: $(cpu_mean)")
    println("  GPU mean: $(gpu_mean)")
    @test isapprox(cpu_mean, gpu_mean, rtol=0.05)
    println("  Calibration GPU: PASS")

    # Test log transform on GPU
    println("\nLog transform on GPU...")
    sino_gpu = apply_log_transform(copy(intensity_gpu))
    sino_gpu_result = Array(sino_gpu)

    cpu_log_mean = mean(sino_from_log)
    gpu_log_mean = mean(sino_gpu_result)
    println("  CPU mean: $(cpu_log_mean)")
    println("  GPU mean: $(gpu_log_mean)")
    @test isapprox(cpu_log_mean, gpu_log_mean, rtol=0.05)
    println("  Log transform GPU: PASS")

    # Test full pipeline on GPU
    println("\nFull pipeline on GPU...")
    intensity_gpu_copy = copy(intensity_gpu)
    sino_pipeline_gpu = calibrate_sinogram!(intensity_gpu_copy, air_gpu, 0.0f0)
    sino_pipeline_gpu_result = Array(sino_pipeline_gpu)

    cpu_pipeline_mean = mean(sinogram_calibrated)
    gpu_pipeline_mean = mean(sino_pipeline_gpu_result)
    println("  CPU mean: $(cpu_pipeline_mean)")
    println("  GPU mean: $(gpu_pipeline_mean)")
    @test isapprox(cpu_pipeline_mean, gpu_pipeline_mean, rtol=0.05)
    println("  Full pipeline GPU: PASS")

    println("\n=== All GPU Calibration tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Test Round-Trip: Intensity -> Calibration -> Sinogram -> Recon
# =============================================================================
println("\n=== Round-Trip Test ===")

# Forward project -> intensity -> calibrate -> reconstruct
recon_from_calibrated = fdk_reconstruct(sinogram_calibrated, geom, size(phantom.μ))
recon_direct = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))

println("Direct recon range: [$(minimum(recon_direct)), $(maximum(recon_direct))]")
println("Calibrated recon range: [$(minimum(recon_from_calibrated)), $(maximum(recon_from_calibrated))]")

# Should be very similar
@test isapprox(mean(recon_from_calibrated), mean(recon_direct), rtol=0.05)
println("Round-trip: PASS")

println("\n=== Calibration Pipeline Tests Complete ===")
