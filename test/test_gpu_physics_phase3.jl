"""
Test Phase 3 GPU Physics Effects - Convolution

Tests that Crosstalk and FocalSpot blur work correctly
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

# Convert to intensity domain for testing
intensity_cpu = exp.(-sinogram_cpu)

# =============================================================================
# Test 1: Crosstalk (3x3 kernel)
# =============================================================================
println("\n=== Test 1: Crosstalk ===")

ct_model = crosstalk_medium()  # 10% crosstalk

# Test kernel creation
kernel_3x3 = create_crosstalk_kernel_3x3(ct_model)
println("Crosstalk kernel (3x3):")
println("  Center: $(kernel_3x3[2,2])")
println("  Direct neighbors: $(kernel_3x3[2,1]), $(kernel_3x3[2,3]), $(kernel_3x3[1,2]), $(kernel_3x3[3,2])")
println("  Kernel sum: $(sum(kernel_3x3))")

@test isapprox(sum(kernel_3x3), 1.0, rtol=1e-10)  # Kernel should be normalized

# Test crosstalk application
sino_ct_cpu = apply_crosstalk(copy(sinogram_cpu), ct_model)
println("\nCrosstalk effect:")
println("  Original mean: $(mean(sinogram_cpu))")
println("  With crosstalk mean: $(mean(sino_ct_cpu))")

# Crosstalk is applied in intensity domain, should slightly change values
# but mean should remain similar (signal is redistributed, not added/removed)
@test isapprox(mean(sino_ct_cpu), mean(sinogram_cpu), rtol=0.1)

# Test intensity domain crosstalk
int_ct_cpu = apply_crosstalk_intensity(copy(intensity_cpu), ct_model)
println("  Intensity domain ratio: $(mean(int_ct_cpu) / mean(intensity_cpu))")

# =============================================================================
# Test 2: Optical Crosstalk (separable kernel)
# =============================================================================
println("\n=== Test 2: Optical Crosstalk ===")

opt_ct_model = optical_crosstalk_typical()

# Test kernel creation
opt_kernel = create_optical_crosstalk_kernel(opt_ct_model)
println("Optical crosstalk kernel:")
println("  Size: $(size(opt_kernel))")
println("  Sum: $(sum(opt_kernel))")

@test isapprox(sum(opt_kernel), 1.0, rtol=1e-10)

# Test application
sino_opt_ct_cpu = apply_optical_crosstalk(copy(sinogram_cpu), opt_ct_model)
println("  With optical crosstalk mean: $(mean(sino_opt_ct_cpu))")

# =============================================================================
# Test 3: Focal Spot Blur
# =============================================================================
println("\n=== Test 3: Focal Spot Blur ===")

fs_model = focal_spot_medium()  # 0.8mm focal spot

# Check blur FWHM
blur_fwhm = compute_focal_spot_blur_fwhm(fs_model, geom, geom.SAD)
println("Focal spot blur FWHM at isocenter:")
println("  Width (pixels): $(blur_fwhm[1])")
println("  Length (pixels): $(blur_fwhm[2])")

# Test kernel creation
kernel_fs = create_focal_spot_kernel_spatial(fs_model, blur_fwhm)
println("Focal spot kernel:")
println("  Size: $(size(kernel_fs))")
println("  Sum: $(sum(kernel_fs))")

@test isapprox(sum(kernel_fs), 1.0, rtol=1e-10)

# Test blur application
sino_fs_cpu = apply_focal_spot_blur(copy(sinogram_cpu), fs_model, geom)
println("\nFocal spot blur effect:")
println("  Original mean: $(mean(sinogram_cpu))")
println("  With blur mean: $(mean(sino_fs_cpu))")

# Blur should preserve mean (normalized kernel)
@test isapprox(mean(sino_fs_cpu), mean(sinogram_cpu), rtol=0.05)

# Blur should reduce variance (smoothing effect)
original_std = std(sinogram_cpu)
blurred_std = std(sino_fs_cpu)
println("  Original std: $(original_std)")
println("  Blurred std: $(blurred_std)")
@test blurred_std <= original_std  # Blur reduces variance

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    sinogram_gpu = MtlArray(sinogram_cpu)
    intensity_gpu = MtlArray(intensity_cpu)
    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test Crosstalk on GPU
    println("\nCrosstalk on GPU...")
    sino_ct_gpu = apply_crosstalk(copy(sinogram_gpu), ct_model)
    sino_ct_gpu_result = Array(sino_ct_gpu)

    # Compare means (should be similar)
    cpu_mean = mean(sino_ct_cpu)
    gpu_mean = mean(sino_ct_gpu_result)
    println("  CPU mean: $(cpu_mean)")
    println("  GPU mean: $(gpu_mean)")
    @test isapprox(cpu_mean, gpu_mean, rtol=0.05)
    println("  Crosstalk GPU: PASS")

    # Test Optical Crosstalk on GPU
    println("\nOptical Crosstalk on GPU...")
    sino_opt_ct_gpu = apply_optical_crosstalk(copy(sinogram_gpu), opt_ct_model)
    sino_opt_ct_gpu_result = Array(sino_opt_ct_gpu)

    cpu_opt_mean = mean(sino_opt_ct_cpu)
    gpu_opt_mean = mean(sino_opt_ct_gpu_result)
    println("  CPU mean: $(cpu_opt_mean)")
    println("  GPU mean: $(gpu_opt_mean)")
    @test isapprox(cpu_opt_mean, gpu_opt_mean, rtol=0.05)
    println("  Optical Crosstalk GPU: PASS")

    # Test Focal Spot Blur on GPU
    println("\nFocal Spot Blur on GPU...")
    sino_fs_gpu = apply_focal_spot_blur(copy(sinogram_gpu), fs_model, geom)
    sino_fs_gpu_result = Array(sino_fs_gpu)

    cpu_fs_mean = mean(sino_fs_cpu)
    gpu_fs_mean = mean(sino_fs_gpu_result)
    println("  CPU mean: $(cpu_fs_mean)")
    println("  GPU mean: $(gpu_fs_mean)")
    @test isapprox(cpu_fs_mean, gpu_fs_mean, rtol=0.05)
    println("  Focal Spot Blur GPU: PASS")

    # Check GPU std reduction (blur should smooth)
    gpu_blurred_std = std(sino_fs_gpu_result)
    println("  GPU blurred std: $(gpu_blurred_std)")
    @test gpu_blurred_std <= original_std

    println("\n=== All GPU convolution tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Full Pipeline Test: Combined Convolution Effects
# =============================================================================
println("\n=== Full Pipeline Test: Combined Convolution Effects ===")

sino_combined = copy(sinogram_cpu)

# Apply effects in sequence (in projection domain for simplicity)
apply_crosstalk!(sino_combined, ct_model)
apply_focal_spot_blur!(sino_combined, fs_model, geom)

println("Combined convolution effects:")
println("  Original mean: $(mean(sinogram_cpu))")
println("  Combined mean: $(mean(sino_combined))")
println("  Original std: $(std(sinogram_cpu))")
println("  Combined std: $(std(sino_combined))")

# Reconstruct
recon_original = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))
recon_combined = fdk_reconstruct(sino_combined, geom, size(phantom.μ))

println("\nReconstruction comparison:")
println("  Original recon range: [$(minimum(recon_original)), $(maximum(recon_original))]")
println("  Combined recon range: [$(minimum(recon_combined)), $(maximum(recon_combined))]")

println("\n=== Phase 3 Convolution Tests Complete ===")
