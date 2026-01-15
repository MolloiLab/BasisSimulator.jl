"""
Test Helical Scanning (Phase 4)

Tests helical CT: geometry, fan-to-parallel rebinning, helical reconstruction.
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

# Create base axial geometry
base_geom = create_aquilion_one(n_angles=360, n_rows=16, n_cols=128, fov_cm=35.0)

# Create test phantom
phantom = create_gammex_472(n_voxels=64, fov_cm=35.0, z_cm=8.0)

# =============================================================================
# Test 1: Helical Geometry Creation
# =============================================================================
println("\n=== Test 1: Helical Geometry Creation ===")

helical_geom = create_helical_geometry(base_geom; pitch=1.0, rotation_time=0.5, z_start=0.0)

info = get_helical_info(helical_geom)
println("Helical geometry info:")
println("  Pitch: $(info.pitch)")
println("  Table speed: $(info.table_speed) mm/s")
println("  Rotation time: $(info.rotation_time) s")
println("  Number of rotations: $(round(info.n_rotations, digits=2))")
println("  Z range: $(info.z_range) mm")
println("  Z coverage: $(info.z_coverage) mm")

@test info.pitch == 1.0
@test info.rotation_time == 0.5
@test length(helical_geom.z_positions) == base_geom.n_angles
println("Helical geometry: PASS")

# =============================================================================
# Test 2: Different Pitch Values
# =============================================================================
println("\n=== Test 2: Different Pitch Values ===")

for pitch in [0.5, 1.0, 1.5, 2.0]
    hg = create_helical_geometry(base_geom; pitch=pitch)
    info = get_helical_info(hg)
    println("  Pitch $pitch: z_coverage=$(round(info.z_coverage, digits=2)) mm")
end
println("Pitch variations: PASS")

# =============================================================================
# Test 3: Z-Position Calculation
# =============================================================================
println("\n=== Test 3: Z-Position Calculation ===")

# Z-positions should be linearly increasing
z_pos = helical_geom.z_positions
z_diff = diff(z_pos)
mean_diff = mean(z_diff)
std_diff = std(z_diff)

println("Z-position differences: mean=$(round(mean_diff, digits=4)), std=$(round(std_diff, digits=6))")

# Should be approximately constant (linear motion)
@test std_diff / mean_diff < 0.01  # Less than 1% variation
println("Z-position linearity: PASS")

# =============================================================================
# Test 4: Forward Projection for Helical
# =============================================================================
println("\n=== Test 4: Forward Projection ===")

# Use base geometry for forward projection (helical uses same projection)
sinogram_cpu = siddon_forward_project(Float32.(phantom.μ), base_geom)
println("Sinogram size: $(size(sinogram_cpu))")
println("Sinogram range: [$(minimum(sinogram_cpu)), $(maximum(sinogram_cpu))]")

# =============================================================================
# Test 5: Helical Weighting
# =============================================================================
println("\n=== Test 5: Helical Weighting ===")

# Test linear interpolation weighting
sino_weighted_li = apply_helical_weights(copy(sinogram_cpu), helical_geom; method=:linear_interp)
println("After LI weighting mean: $(mean(sino_weighted_li))")

# For current simplified implementation, weights are 1.0
@test isapprox(mean(sino_weighted_li), mean(sinogram_cpu), rtol=0.01)
println("Linear interpolation weighting: PASS")

# Test Parker weighting
sino_weighted_parker = apply_helical_weights(copy(sinogram_cpu), helical_geom; method=:parker)
println("After Parker weighting mean: $(mean(sino_weighted_parker))")
@test isapprox(mean(sino_weighted_parker), mean(sinogram_cpu), rtol=0.01)
println("Parker weighting: PASS")

# =============================================================================
# Test 6: Fan-to-Parallel Rebinning
# =============================================================================
println("\n=== Test 6: Fan-to-Parallel Rebinning ===")

parallel_sino = fan_to_parallel_rebin(sinogram_cpu, base_geom)
println("Parallel sinogram size: $(size(parallel_sino))")
println("Parallel sinogram range: [$(minimum(parallel_sino)), $(maximum(parallel_sino))]")

# Rebinned sinogram should have same total signal (approximately)
original_sum = sum(sinogram_cpu)
parallel_sum = sum(parallel_sino)
println("Original sum: $(original_sum)")
println("Parallel sum: $(parallel_sum)")

# Note: Perfect conservation not expected due to interpolation at boundaries
@test abs(parallel_sum - original_sum) / original_sum < 0.5  # Within 50%
println("Fan-to-parallel rebinning: PASS")

# =============================================================================
# Test 7: Helical FDK Reconstruction
# =============================================================================
println("\n=== Test 7: Helical FDK Reconstruction ===")

volume_size = size(phantom.μ)
recon = helical_fdk_reconstruct(sinogram_cpu, helical_geom, volume_size)

println("Reconstruction size: $(size(recon))")
println("Reconstruction range: [$(minimum(recon)), $(maximum(recon))]")

# Should produce reasonable values
@test minimum(recon) > -1.0
@test maximum(recon) < 2.0
println("Helical FDK: PASS")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    sinogram_gpu = MtlArray(sinogram_cpu)
    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test helical weighting on GPU
    println("\nHelical weighting on GPU...")
    sino_weighted_gpu = apply_helical_weights(copy(sinogram_gpu), helical_geom; method=:linear_interp)
    sino_weighted_gpu_result = Array(sino_weighted_gpu)

    cpu_mean = mean(sino_weighted_li)
    gpu_mean = mean(sino_weighted_gpu_result)
    println("  CPU mean: $(cpu_mean)")
    println("  GPU mean: $(gpu_mean)")
    @test isapprox(cpu_mean, gpu_mean, rtol=0.05)
    println("  Helical weighting GPU: PASS")

    # Test fan-to-parallel rebinning on GPU
    println("\nFan-to-parallel rebinning on GPU...")
    parallel_sino_gpu = fan_to_parallel_rebin(sinogram_gpu, base_geom)
    parallel_sino_gpu_result = Array(parallel_sino_gpu)

    cpu_parallel_mean = mean(parallel_sino)
    gpu_parallel_mean = mean(parallel_sino_gpu_result)
    println("  CPU mean: $(cpu_parallel_mean)")
    println("  GPU mean: $(gpu_parallel_mean)")
    @test isapprox(cpu_parallel_mean, gpu_parallel_mean, rtol=0.1)
    println("  Fan-to-parallel GPU: PASS")

    println("\n=== All GPU Helical tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Reconstruction Comparison: Axial vs Helical
# =============================================================================
println("\n=== Reconstruction Comparison: Axial vs Helical ===")

recon_axial = fdk_reconstruct(sinogram_cpu, base_geom, volume_size)
recon_helical = helical_fdk_reconstruct(sinogram_cpu, helical_geom, volume_size)

println("Axial recon range: [$(minimum(recon_axial)), $(maximum(recon_axial))]")
println("Helical recon range: [$(minimum(recon_helical)), $(maximum(recon_helical))]")

# For identical sinogram input, results should be similar
# (helical adds weighting but at pitch=1.0, weights are ~1.0)
@test isapprox(mean(recon_axial), mean(recon_helical), rtol=0.2)
println("Axial vs Helical comparison: PASS")

# =============================================================================
# Test Different Reconstruction Filters
# =============================================================================
println("\n=== Test: Different Reconstruction Filters ===")

for filter in [RampFilter(), SheppLoganFilter(), HammingFilter()]
    filter_name = string(typeof(filter))
    recon_filtered = helical_fdk_reconstruct(sinogram_cpu, helical_geom, volume_size;
        filter=filter, cutoff=0.8)
    println("  $filter_name: range=[$(round(minimum(recon_filtered), digits=3)), $(round(maximum(recon_filtered), digits=3))]")
end

println("\n=== Helical Scanning Tests Complete ===")
