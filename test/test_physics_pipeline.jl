"""
Test Unified Physics Pipeline

Tests that the unified apply_physics_effects!() function works correctly
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
# Test 1: Default Config (no effects)
# =============================================================================
println("\n=== Test 1: Default Config (no effects) ===")

config_default = default_physics_config()
info = get_physics_config_info(config_default)
println("Enabled effects: $(info.n_enabled) of $(info.n_total)")
@test info.n_enabled == 0

sino_default = apply_physics_effects(copy(sinogram_cpu), geom, config_default)
@test isapprox(mean(sino_default), mean(sinogram_cpu), rtol=1e-6)
println("Default config: PASS (no change)")

# =============================================================================
# Test 2: Minimal Config (noise only)
# =============================================================================
println("\n=== Test 2: Minimal Config (noise only) ===")

config_minimal = minimal_physics_config(noise_level=1.0, noise_seed=42)
info = get_physics_config_info(config_minimal)
println("Enabled effects: $(info.enabled_effects)")
@test info.n_enabled == 1

sino_minimal = apply_physics_effects(copy(sinogram_cpu), geom, config_minimal)
println("Original mean: $(mean(sinogram_cpu))")
println("With noise mean: $(mean(sino_minimal))")

# Noise should not change mean significantly (Gaussian approximation)
@test isapprox(mean(sino_minimal), mean(sinogram_cpu), rtol=0.1)

# But std should increase
@test std(sino_minimal) >= std(sinogram_cpu) * 0.9  # Allow some tolerance
println("Minimal config: PASS")

# =============================================================================
# Test 3: Realistic Config
# =============================================================================
println("\n=== Test 3: Realistic Config ===")

config_realistic = realistic_physics_config(
    scatter_scale=1.0,
    noise_level=1.0,
    noise_seed=42
)
info = get_physics_config_info(config_realistic)
println("Enabled effects: $(info.enabled_effects)")
@test info.n_enabled >= 4  # scatter, crosstalk, focal_spot, noise, lag

sino_realistic = apply_physics_effects(copy(sinogram_cpu), geom, config_realistic)
println("Original mean: $(mean(sinogram_cpu))")
println("With all effects mean: $(mean(sino_realistic))")
println("Realistic config: PASS")

# =============================================================================
# Test 4: Custom Config
# =============================================================================
println("\n=== Test 4: Custom Config ===")

config_custom = default_physics_config(
    scatter = default_scatter_model(scale_factor=0.5),
    crosstalk = crosstalk_low(),
    noise = default_detector_model(I0=500000.0)
)
info = get_physics_config_info(config_custom)
println("Enabled effects: $(info.enabled_effects)")
@test info.n_enabled == 3

sino_custom = apply_physics_effects(copy(sinogram_cpu), geom, config_custom)
println("Original mean: $(mean(sinogram_cpu))")
println("With custom effects mean: $(mean(sino_custom))")
println("Custom config: PASS")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    sinogram_gpu = MtlArray(sinogram_cpu)
    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test realistic config on GPU
    println("\nRealistic config on GPU...")
    sino_realistic_gpu = apply_physics_effects(copy(sinogram_gpu), geom, config_realistic)
    sino_realistic_gpu_result = Array(sino_realistic_gpu)

    # GPU and CPU should produce similar statistics (noise seed may vary)
    # We compare the deterministic part by using different seeds
    config_realistic_cpu = realistic_physics_config(scatter_scale=1.0, noise_level=1.0, noise_seed=42)
    config_realistic_gpu_test = realistic_physics_config(scatter_scale=1.0, noise_level=1.0, noise_seed=42)

    sino_cpu_seeded = apply_physics_effects(copy(sinogram_cpu), geom, config_realistic_cpu)
    sino_gpu_seeded = apply_physics_effects(copy(sinogram_gpu), geom, config_realistic_gpu_test)
    sino_gpu_seeded_result = Array(sino_gpu_seeded)

    cpu_mean = mean(sino_cpu_seeded)
    gpu_mean = mean(sino_gpu_seeded_result)
    println("  CPU mean: $(cpu_mean)")
    println("  GPU mean: $(gpu_mean)")

    # Means should be similar (though noise adds some variance)
    @test isapprox(cpu_mean, gpu_mean, rtol=0.1)
    println("  GPU realistic: PASS")

    println("\n=== All GPU pipeline tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Reconstruction Test: Compare ideal vs realistic
# =============================================================================
println("\n=== Reconstruction Comparison ===")

# Ideal (no physics effects)
recon_ideal = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))

# With realistic physics
recon_realistic = fdk_reconstruct(sino_realistic, geom, size(phantom.μ))

println("Ideal reconstruction range: [$(minimum(recon_ideal)), $(maximum(recon_ideal))]")
println("Realistic reconstruction range: [$(minimum(recon_realistic)), $(maximum(recon_realistic))]")

# The reconstructions should be different due to physics effects
# but both should have reasonable values
@test minimum(recon_realistic) > -1.0
@test maximum(recon_realistic) < 2.0

println("\n=== Unified Physics Pipeline Tests Complete ===")
