"""
Test Phase 4 GPU Physics Effects - Complex Effects

Tests that Scatter and DetectorLag work correctly
on both CPU and GPU (Metal).
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
# Test 1: Scatter (Spatial Convolution with Large Kernel)
# =============================================================================
println("\n=== Test 1: Scatter ===")

scatter_model = default_scatter_model(scale_factor=1.0, kernel_fwhm=30.0)
println("Scatter model: coefficient=$(scatter_model.scatter_coefficient), scale=$(scatter_model.scale_factor)")
println("Kernel FWHM: $(scatter_model.kernel_fwhm) pixels")

# Test kernel creation
kernel = create_scatter_kernel_spatial(scatter_model)
println("Scatter kernel size: $(size(kernel))")
println("Kernel sum: $(sum(kernel))")
@test isapprox(sum(kernel), 1.0, rtol=1e-6)  # Kernel should be normalized

# Test scatter application on CPU
sino_scatter_cpu = add_scatter(copy(sinogram_cpu), scatter_model)
println("\nScatter effect (CPU):")
println("  Original mean: $(mean(sinogram_cpu))")
println("  With scatter mean: $(mean(sino_scatter_cpu))")

# Scatter should generally reduce projection values (add intensity, reduce line integral)
# The effect depends on the specific phantom and geometry
@test !isapprox(mean(sino_scatter_cpu), mean(sinogram_cpu), rtol=0.001)  # Should change
println("  Relative change: $(100 * (mean(sino_scatter_cpu) - mean(sinogram_cpu)) / mean(sinogram_cpu))%")

# =============================================================================
# Test 2: Detector Lag (Temporal Weighted Sum)
# =============================================================================
println("\n=== Test 2: Detector Lag ===")

lag_model = lag_gadox(frame_time=0.5)
println("Lag model: $(length(lag_model.amplitudes)) components")
println("  Amplitudes: $(lag_model.amplitudes)")
println("  Time constants: $(lag_model.time_constants) ms")
println("  Frame time: $(lag_model.frame_time) ms")

# Get lag info
lag_info = get_lag_info(lag_model)
println("Total lag fraction: $(lag_info.total_lag_fraction)")

# Test lag coefficients
coeffs = compute_lag_coefficients(lag_model, 10)
println("\nLag coefficients (first 10):")
println("  $(coeffs)")
@test isapprox(sum(coeffs), 1.0, rtol=1e-6)  # Coefficients should sum to 1

# Test lag application on CPU
sino_lag_cpu = apply_lag(copy(sinogram_cpu), lag_model; n_history=20)
println("\nLag effect (CPU):")
println("  Original mean: $(mean(sinogram_cpu))")
println("  With lag mean: $(mean(sino_lag_cpu))")

# Lag should preserve mean (signal is redistributed temporally)
@test isapprox(mean(sino_lag_cpu), mean(sinogram_cpu), rtol=0.05)

# Test recursive version
sino_lag_rec_cpu = apply_lag_recursive(copy(sinogram_cpu), lag_model)
println("  With lag (recursive) mean: $(mean(sino_lag_rec_cpu))")

# Both methods should produce similar results
@test isapprox(mean(sino_lag_rec_cpu), mean(sino_lag_cpu), rtol=0.1)

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    sinogram_gpu = MtlArray(sinogram_cpu)
    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test Scatter on GPU
    println("\nScatter on GPU...")
    sino_scatter_gpu = add_scatter(copy(sinogram_gpu), scatter_model)
    sino_scatter_gpu_result = Array(sino_scatter_gpu)

    cpu_scatter_mean = mean(sino_scatter_cpu)
    gpu_scatter_mean = mean(sino_scatter_gpu_result)
    println("  CPU mean: $(cpu_scatter_mean)")
    println("  GPU mean: $(gpu_scatter_mean)")
    @test isapprox(cpu_scatter_mean, gpu_scatter_mean, rtol=0.05)
    println("  Scatter GPU: PASS")

    # Test Lag on GPU
    println("\nLag on GPU...")
    sino_lag_gpu = apply_lag(copy(sinogram_gpu), lag_model; n_history=20)
    sino_lag_gpu_result = Array(sino_lag_gpu)

    cpu_lag_mean = mean(sino_lag_cpu)
    gpu_lag_mean = mean(sino_lag_gpu_result)
    println("  CPU mean: $(cpu_lag_mean)")
    println("  GPU mean: $(gpu_lag_mean)")
    @test isapprox(cpu_lag_mean, gpu_lag_mean, rtol=0.05)
    println("  Lag GPU: PASS")

    # Test recursive lag on GPU
    println("\nRecursive Lag on GPU...")
    sino_lag_rec_gpu = apply_lag_recursive(copy(sinogram_gpu), lag_model)
    sino_lag_rec_gpu_result = Array(sino_lag_rec_gpu)

    cpu_lag_rec_mean = mean(sino_lag_rec_cpu)
    gpu_lag_rec_mean = mean(sino_lag_rec_gpu_result)
    println("  CPU mean: $(cpu_lag_rec_mean)")
    println("  GPU mean: $(gpu_lag_rec_mean)")
    @test isapprox(cpu_lag_rec_mean, gpu_lag_rec_mean, rtol=0.05)
    println("  Recursive Lag GPU: PASS")

    println("\n=== All GPU Phase 4 tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Full Pipeline Test: Combined Phase 4 Effects
# =============================================================================
println("\n=== Full Pipeline Test: Combined Phase 4 Effects ===")

sino_combined = copy(sinogram_cpu)

# Apply effects in sequence
add_scatter!(sino_combined, scatter_model)
apply_lag!(sino_combined, lag_model)

println("Combined Phase 4 effects:")
println("  Original mean: $(mean(sinogram_cpu))")
println("  Combined mean: $(mean(sino_combined))")
println("  Original std: $(std(sinogram_cpu))")
println("  Combined std: $(std(sino_combined))")

# Reconstruct and compare
recon_original = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))
recon_combined = fdk_reconstruct(sino_combined, geom, size(phantom.μ))

println("\nReconstruction comparison:")
println("  Original recon range: [$(minimum(recon_original)), $(maximum(recon_original))]")
println("  Combined recon range: [$(minimum(recon_combined)), $(maximum(recon_combined))]")

# =============================================================================
# Test Different Scatter Scales
# =============================================================================
println("\n=== Test: Different Scatter Scale Factors ===")

for scale in [0.5, 1.0, 2.0]
    model = default_scatter_model(scale_factor=scale)
    sino_scatter = add_scatter(copy(sinogram_cpu), model)
    change = 100 * (mean(sino_scatter) - mean(sinogram_cpu)) / mean(sinogram_cpu)
    println("  scale_factor=$scale: $(round(change, digits=2))% change")
end

# =============================================================================
# Test Different Lag Models
# =============================================================================
println("\n=== Test: Different Lag Models ===")

for (name, model) in [("none", lag_none()), ("gadox", lag_gadox()), ("csi", lag_csi()), ("high", lag_high())]
    sino_lag = apply_lag(copy(sinogram_cpu), model)
    info = get_lag_info(model)
    println("  $name: total_lag=$(round(info.total_lag_fraction * 100, digits=2))%, mean=$(round(mean(sino_lag), digits=4))")
end

println("\n=== Phase 4 Complex Effects Tests Complete ===")
