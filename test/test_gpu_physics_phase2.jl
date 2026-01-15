"""
Test Phase 2 GPU Physics Effects - Noise

Tests that quantum (Poisson) and electronic (Gaussian) noise
produce correct statistics on both CPU and GPU (Metal).
"""

using BasisSimulator
using Test
using Statistics
using Random

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
println("Sinogram mean: $(mean(sinogram_cpu))")

# =============================================================================
# Test 1: Electronic Noise
# =============================================================================
println("\n=== Test 1: Electronic Noise ===")

# Create detector model with only electronic noise
det_model_elec = DetectorModel(0.0, 1e5, 50.0, 42)  # blur=0, I0=1e5, σ=50, seed=42

# Test on CPU
sino_elec_cpu = add_electronic_noise(copy(sinogram_cpu), det_model_elec)

# Compute noise statistics
noise_diff = sino_elec_cpu .- sinogram_cpu
noise_std_measured = std(noise_diff)
noise_mean_measured = mean(noise_diff)

# Expected noise std = electronic_noise_std / I0 = 50 / 1e5 = 5e-4
expected_noise_std = det_model_elec.electronic_noise_std / det_model_elec.I0

println("Electronic Noise:")
println("  Expected noise std: $(expected_noise_std)")
println("  Measured noise std: $(noise_std_measured)")
println("  Measured noise mean: $(noise_mean_measured) (should be ~0)")
println("  Ratio measured/expected: $(noise_std_measured / expected_noise_std)")

@test isapprox(noise_std_measured, expected_noise_std, rtol=0.1)
@test abs(noise_mean_measured) < expected_noise_std * 0.1  # Mean should be near zero

# =============================================================================
# Test 2: Quantum Noise (Poisson with Gaussian approximation)
# =============================================================================
println("\n=== Test 2: Quantum Noise ===")

# Create detector model with only quantum noise
det_model_quant = DetectorModel(0.0, 1e5, 0.0, 42)  # blur=0, I0=1e5, σ=0, seed=42

# Test on CPU
sino_quant_cpu = add_quantum_noise(copy(sinogram_cpu), det_model_quant)

# For Poisson noise: variance = mean (in photon count domain)
# In attenuation domain, the noise is more complex but should scale with I0

# Compute noise statistics
noise_diff_quant = sino_quant_cpu .- sinogram_cpu
noise_std_quant = std(noise_diff_quant)

println("Quantum Noise:")
println("  Noisy sinogram mean: $(mean(sino_quant_cpu))")
println("  Clean sinogram mean: $(mean(sinogram_cpu))")
println("  Measured noise std: $(noise_std_quant)")

# The noise std should be roughly 1/sqrt(I0) in projection space
# For I0=1e5, sqrt(I0)≈316, so noise std ≈ 1/316 ≈ 0.003 in relative terms
@test noise_std_quant > 0.001  # Should have measurable noise
@test noise_std_quant < 0.1   # But not excessive noise

# =============================================================================
# Test 3: Combined Noise
# =============================================================================
println("\n=== Test 3: Combined Noise ===")

# Create detector model with both noise types
det_model_both = DetectorModel(0.0, 1e5, 50.0, 42)

sino_both_cpu = copy(sinogram_cpu)
add_quantum_noise!(sino_both_cpu, det_model_both)
add_electronic_noise!(sino_both_cpu, det_model_both)

noise_stats = compute_noise_level(sinogram_cpu, sino_both_cpu)
println("Combined Noise Statistics:")
println("  SNR: $(noise_stats.snr)")
println("  Noise std: $(noise_stats.std_diff)")
println("  Max diff: $(noise_stats.max_diff)")

@test noise_stats.snr > 10  # Should still have reasonable SNR

# =============================================================================
# Test 4: Reproducibility with seed
# =============================================================================
println("\n=== Test 4: Reproducibility ===")

det_model_seed = DetectorModel(0.0, 1e5, 50.0, 123)

sino_run1 = add_electronic_noise(copy(sinogram_cpu), det_model_seed)
sino_run2 = add_electronic_noise(copy(sinogram_cpu), det_model_seed)

@test sino_run1 == sino_run2
println("Reproducibility: PASS (same seed gives same results)")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU array
    sinogram_gpu = MtlArray(sinogram_cpu)
    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test Electronic Noise on GPU
    println("\nElectronic Noise on GPU...")
    det_model_elec_gpu = DetectorModel(0.0, 1e5, 50.0, 42)
    sino_elec_gpu = add_electronic_noise(copy(sinogram_gpu), det_model_elec_gpu)
    sino_elec_gpu_result = Array(sino_elec_gpu)

    # Note: Results won't match exactly due to different array indexing on GPU
    # but noise statistics should be similar
    noise_diff_gpu = sino_elec_gpu_result .- Array(sinogram_cpu)
    noise_std_gpu = std(noise_diff_gpu)
    println("  GPU noise std: $(noise_std_gpu)")
    println("  CPU noise std: $(noise_std_measured)")
    @test isapprox(noise_std_gpu, expected_noise_std, rtol=0.2)
    println("  Electronic Noise GPU statistics: PASS")

    # Test Quantum Noise on GPU
    println("\nQuantum Noise on GPU...")
    det_model_quant_gpu = DetectorModel(0.0, 1e5, 0.0, 42)
    sino_quant_gpu = add_quantum_noise(copy(sinogram_gpu), det_model_quant_gpu)
    sino_quant_gpu_result = Array(sino_quant_gpu)

    noise_diff_quant_gpu = sino_quant_gpu_result .- Array(sinogram_cpu)
    noise_std_quant_gpu = std(noise_diff_quant_gpu)
    println("  GPU quantum noise std: $(noise_std_quant_gpu)")
    println("  CPU quantum noise std: $(noise_std_quant)")
    @test noise_std_quant_gpu > 0.001
    @test noise_std_quant_gpu < 0.1
    println("  Quantum Noise GPU statistics: PASS")

    # Test Combined pipeline on GPU
    println("\nCombined Noise Pipeline on GPU...")
    det_model_full = DetectorModel(0.0, 1e5, 50.0, 42)
    sino_full_gpu = copy(sinogram_gpu)
    add_quantum_noise!(sino_full_gpu, det_model_full)
    add_electronic_noise!(sino_full_gpu, det_model_full)

    sino_full_gpu_result = Array(sino_full_gpu)
    noise_stats_gpu = compute_noise_level(Array(sinogram_cpu), sino_full_gpu_result)
    println("  GPU Combined SNR: $(noise_stats_gpu.snr)")
    println("  CPU Combined SNR: $(noise_stats.snr)")

    # SNR should be in similar range
    @test noise_stats_gpu.snr > 5
    println("  Combined Noise Pipeline GPU: PASS")

    println("\n=== All GPU noise tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Full Pipeline Test: Noise + Reconstruction
# =============================================================================
println("\n=== Full Pipeline Test: Noisy Reconstruction ===")

det_model_full = DetectorModel(0.0, 1e4, 100.0, 42)  # More noise for visible effect

sino_noisy = copy(sinogram_cpu)
add_quantum_noise!(sino_noisy, det_model_full)
add_electronic_noise!(sino_noisy, det_model_full)

# Reconstruct
recon_noisy = fdk_reconstruct(sino_noisy, geom, size(phantom.μ))
recon_clean = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))

# Compare
recon_diff = recon_noisy .- recon_clean
recon_noise_std = std(recon_diff)

println("Reconstruction noise:")
println("  Clean recon range: [$(minimum(recon_clean)), $(maximum(recon_clean))]")
println("  Noisy recon range: [$(minimum(recon_noisy)), $(maximum(recon_noisy))]")
println("  Recon noise std: $(recon_noise_std)")

# Noise should be visible in reconstruction
@test recon_noise_std > 1e-4

println("\n=== Phase 2 Noise Tests Complete ===")
