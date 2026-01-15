"""
Test DAS Model (Phase 3)

Tests Data Acquisition System model: signal conversion, quantization, mA scaling.
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

# Create test data on CPU (intensity mode)
println("\n=== Forward Projection (Intensity) ===")
sinogram_cpu = siddon_forward_project(Float32.(phantom.μ), geom)
# Convert to intensity (simulating pre-log signal)
intensity_cpu = exp.(-sinogram_cpu) .* 10000  # Scale to reasonable photon counts
println("Intensity size: $(size(intensity_cpu))")
println("Intensity range: [$(minimum(intensity_cpu)), $(maximum(intensity_cpu))]")

# =============================================================================
# Test 1: DAS Ideal (No Effects)
# =============================================================================
println("\n=== Test 1: DAS Ideal (No Effects) ===")

das_none = das_ideal()
info = get_das_info(das_none)
println("DAS ideal: gain=$(info.gain), noise=$(info.electronic_noise_sigma)")
@test info.has_noise == false
@test info.has_quantization == false

# Apply ideal DAS (just gain scaling)
signal_ideal = apply_das_model(copy(intensity_cpu), das_none)
println("Original mean: $(mean(intensity_cpu))")
println("After ideal DAS mean: $(mean(signal_ideal))")

# Should be scaled by gain (1.0 for ideal)
@test isapprox(mean(signal_ideal), mean(intensity_cpu) * das_none.gain, rtol=1e-6)
println("DAS ideal: PASS")

# =============================================================================
# Test 2: DAS Default (With Noise)
# =============================================================================
println("\n=== Test 2: DAS Default (With Noise) ===")

das_default = default_das_model(
    gain=20.0,
    electronic_noise_sigma=100.0,
    lsb=0.0
)
info = get_das_info(das_default)
println("DAS default: gain=$(info.gain), noise=$(info.electronic_noise_sigma)")
@test info.has_noise == true

# Apply default DAS
signal_noisy = apply_das_model(copy(intensity_cpu), das_default; seed=42)
println("Original mean × gain: $(mean(intensity_cpu) * das_default.gain)")
println("After noisy DAS mean: $(mean(signal_noisy))")

# Mean should be approximately preserved (noise is Gaussian)
@test isapprox(mean(signal_noisy), mean(intensity_cpu) * das_default.gain, rtol=0.05)

# Std should increase due to noise
original_std = std(intensity_cpu) * das_default.gain
noisy_std = std(signal_noisy)
println("Original std × gain: $(original_std)")
println("Noisy std: $(noisy_std)")
@test noisy_std > original_std * 0.9  # Should have added noise
println("DAS default: PASS")

# =============================================================================
# Test 3: DAS Clinical
# =============================================================================
println("\n=== Test 3: DAS Clinical ===")

das_clinical_model = das_clinical(noise_level=1.0)
info = get_das_info(das_clinical_model)
println("DAS clinical: gain=$(info.gain), noise=$(info.electronic_noise_sigma)")
println("  max_value=$(info.max_value), min_value=$(info.min_value)")

# Apply clinical DAS
signal_clinical = apply_das_model(copy(intensity_cpu), das_clinical_model; seed=42)
println("After clinical DAS mean: $(mean(signal_clinical))")
println("After clinical DAS range: [$(minimum(signal_clinical)), $(maximum(signal_clinical))]")

# Should be clamped to valid range
@test minimum(signal_clinical) >= info.min_value
@test maximum(signal_clinical) <= info.max_value
println("DAS clinical: PASS")

# =============================================================================
# Test 4: Quantization
# =============================================================================
println("\n=== Test 4: Quantization ===")

das_quantized = default_das_model(
    gain=1.0,
    electronic_noise_sigma=0.0,
    lsb=100.0  # Quantization step
)
info = get_das_info(das_quantized)
@test info.has_quantization == true

# Apply quantization
signal_quantized = apply_das_model(copy(intensity_cpu), das_quantized)

# Values should be multiples of lsb
unique_vals = unique(round.(signal_quantized, digits=-2))
println("Number of unique quantized values: $(length(unique_vals))")
println("Sample quantized values: $(unique_vals[1:min(5, length(unique_vals))])")

# Check that values are approximately multiples of lsb
for val in signal_quantized[1:100]
    @test abs(val % 100.0) < 1.0 || abs(100.0 - (val % 100.0)) < 1.0
end
println("Quantization: PASS")

# =============================================================================
# Test 5: Tube Current Modulation
# =============================================================================
println("\n=== Test 5: Tube Current Modulation ===")

# Constant tube current
signal_100mA = apply_tube_current(copy(intensity_cpu), 100.0)
signal_200mA = apply_tube_current(copy(intensity_cpu), 200.0)

# 200 mA should give 2× signal
ratio = mean(signal_200mA) / mean(signal_100mA)
println("200mA / 100mA ratio: $(ratio)")
@test isapprox(ratio, 2.0, rtol=0.01)
println("Constant mA: PASS")

# Per-view tube current modulation
n_angles = geom.n_angles
mA_per_view = [100.0 + 50.0 * cos(2π * i / n_angles) for i in 1:n_angles]
signal_modulated = apply_tube_current(copy(intensity_cpu), mA_per_view)

# Mean should be scaled appropriately
mean_mA = mean(mA_per_view) / 100.0
println("Mean mA scale factor: $(mean_mA)")
println("Signal scale: $(mean(signal_modulated) / mean(intensity_cpu))")
@test isapprox(mean(signal_modulated) / mean(intensity_cpu), mean_mA, rtol=0.1)
println("Per-view mA modulation: PASS")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    intensity_gpu = MtlArray(intensity_cpu)
    println("GPU array type: $(typeof(intensity_gpu))")

    # Test DAS ideal on GPU
    println("\nDAS ideal on GPU...")
    signal_ideal_gpu = apply_das_model(copy(intensity_gpu), das_none)
    signal_ideal_gpu_result = Array(signal_ideal_gpu)

    cpu_mean = mean(signal_ideal)
    gpu_mean = mean(signal_ideal_gpu_result)
    println("  CPU mean: $(cpu_mean)")
    println("  GPU mean: $(gpu_mean)")
    @test isapprox(cpu_mean, gpu_mean, rtol=0.05)
    println("  DAS ideal GPU: PASS")

    # Test DAS with noise on GPU (seeded)
    println("\nDAS with noise on GPU...")
    signal_noisy_gpu = apply_das_model(copy(intensity_gpu), das_default; seed=42)
    signal_noisy_gpu_result = Array(signal_noisy_gpu)

    # Note: Noise is generated on CPU and transferred, so results should be similar
    cpu_noisy_mean = mean(signal_noisy)
    gpu_noisy_mean = mean(signal_noisy_gpu_result)
    println("  CPU mean: $(cpu_noisy_mean)")
    println("  GPU mean: $(gpu_noisy_mean)")
    @test isapprox(cpu_noisy_mean, gpu_noisy_mean, rtol=0.1)
    println("  DAS noisy GPU: PASS")

    # Test tube current on GPU
    println("\nTube current on GPU...")
    signal_mA_gpu = apply_tube_current(copy(intensity_gpu), 200.0)
    signal_mA_gpu_result = Array(signal_mA_gpu)

    cpu_mA_mean = mean(signal_200mA)
    gpu_mA_mean = mean(signal_mA_gpu_result)
    println("  CPU mean: $(cpu_mA_mean)")
    println("  GPU mean: $(gpu_mA_mean)")
    @test isapprox(cpu_mA_mean, gpu_mA_mean, rtol=0.05)
    println("  Tube current GPU: PASS")

    println("\n=== All GPU DAS tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Test Different Noise Levels
# =============================================================================
println("\n=== Test: Different Noise Levels ===")

for noise_level in [0.5, 1.0, 2.0, 5.0]
    das = das_clinical(noise_level=noise_level)
    signal = apply_das_model(copy(intensity_cpu), das; seed=42)
    signal_std = std(signal)
    println("  noise_level=$noise_level: std=$(round(signal_std, digits=2))")
end

# =============================================================================
# Test Value Clamping
# =============================================================================
println("\n=== Test: Value Clamping ===")

das_clamped = DASModel(1.0, 0.0, 0.0, 100.0, 5000.0, 0.0)  # min=100, max=5000
signal_clamped = apply_das_model(copy(intensity_cpu), das_clamped)
println("Clamped range: [$(minimum(signal_clamped)), $(maximum(signal_clamped))]")
@test minimum(signal_clamped) >= 100.0
@test maximum(signal_clamped) <= 5000.0
println("Clamping: PASS")

println("\n=== DAS Model Tests Complete ===")
