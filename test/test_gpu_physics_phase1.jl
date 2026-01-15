"""
Test Phase 1 GPU Physics Effects

Tests that FillFactor, FlatFilter, BowtieFilter, and DetectorEfficiency
produce correct results on both CPU and GPU (Metal).
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

# Convert sinogram to intensity domain for testing intensity-based effects
intensity_cpu = exp.(-sinogram_cpu)

# =============================================================================
# Test 1: FillFactor
# =============================================================================
println("\n=== Test 1: FillFactor ===")

ff_model = fill_factor_standard()  # 90% fill factor

# Test projection domain
sino_ff_cpu = apply_fill_factor(copy(sinogram_cpu), ff_model)
println("FillFactor (projection domain):")
println("  Original mean: $(mean(sinogram_cpu))")
println("  With fill factor mean: $(mean(sino_ff_cpu))")
println("  Expected increase: $(round(-log(0.9), digits=4))")

# Test intensity domain
int_ff_cpu = apply_fill_factor_intensity(copy(intensity_cpu), ff_model)
println("FillFactor (intensity domain):")
println("  Original mean: $(mean(intensity_cpu))")
println("  With fill factor mean: $(mean(int_ff_cpu))")
println("  Ratio: $(mean(int_ff_cpu) / mean(intensity_cpu)) (expected 0.9)")

@test mean(sino_ff_cpu) > mean(sinogram_cpu)  # Fill factor increases projection
@test isapprox(mean(int_ff_cpu) / mean(intensity_cpu), 0.9, rtol=0.01)

# =============================================================================
# Test 2: FlatFilter
# =============================================================================
println("\n=== Test 2: FlatFilter ===")

flat_model = flat_filter_al(2.5)  # 2.5mm aluminum

# Test intensity domain
int_flat_cpu = apply_flat_filter_to_intensity(copy(intensity_cpu), flat_model, geom)
println("FlatFilter (2.5mm Al):")
println("  Original mean: $(mean(intensity_cpu))")
println("  With filter mean: $(mean(int_flat_cpu))")
println("  Transmission: $(mean(int_flat_cpu) / mean(intensity_cpu))")

@test mean(int_flat_cpu) < mean(intensity_cpu)  # Filter reduces intensity

# =============================================================================
# Test 3: BowtieFilter
# =============================================================================
println("\n=== Test 3: BowtieFilter ===")

bowtie_model = bowtie_filter_medium_body()

# Test intensity domain
int_bowtie_cpu = apply_bowtie_to_intensity(copy(intensity_cpu), bowtie_model, geom)
println("BowtieFilter (medium body):")
println("  Original mean: $(mean(intensity_cpu))")
println("  With bowtie mean: $(mean(int_bowtie_cpu))")
println("  Center col transmission: $(int_bowtie_cpu[64, 8, 1] / intensity_cpu[64, 8, 1])")
println("  Edge col transmission: $(int_bowtie_cpu[1, 8, 1] / intensity_cpu[1, 8, 1])")

@test mean(int_bowtie_cpu) < mean(intensity_cpu)  # Bowtie reduces intensity

# =============================================================================
# Test 4: DetectorEfficiency
# =============================================================================
println("\n=== Test 4: DetectorEfficiency ===")

det_model = detector_efficiency_gos(0.5)  # 0.5mm GOS

# Test intensity domain
int_det_cpu = apply_detector_efficiency(copy(intensity_cpu), det_model, geom)
println("DetectorEfficiency (0.5mm GOS):")
println("  Original mean: $(mean(intensity_cpu))")
println("  With detector efficiency mean: $(mean(int_det_cpu))")
println("  Efficiency: $(mean(int_det_cpu) / mean(intensity_cpu))")

# Get expected efficiency
det_info = get_detector_efficiency_info(det_model)
println("  Expected efficiency: $(det_info.total_efficiency)")

@test mean(int_det_cpu) < mean(intensity_cpu)  # Detector reduces intensity

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    sinogram_gpu = MtlArray(sinogram_cpu)
    intensity_gpu = MtlArray(intensity_cpu)

    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test FillFactor on GPU
    println("\nFillFactor on GPU...")
    sino_ff_gpu = apply_fill_factor(copy(sinogram_gpu), ff_model)
    sino_ff_result = Array(sino_ff_gpu)
    @test isapprox(sino_ff_result, sino_ff_cpu, rtol=1e-5)
    println("  FillFactor GPU vs CPU: PASS")

    # Test FlatFilter on GPU
    println("\nFlatFilter on GPU...")
    int_flat_gpu = apply_flat_filter_to_intensity(copy(intensity_gpu), flat_model, geom)
    int_flat_result = Array(int_flat_gpu)
    @test isapprox(int_flat_result, int_flat_cpu, rtol=1e-5)
    println("  FlatFilter GPU vs CPU: PASS")

    # Test BowtieFilter on GPU
    println("\nBowtieFilter on GPU...")
    int_bowtie_gpu = apply_bowtie_to_intensity(copy(intensity_gpu), bowtie_model, geom)
    int_bowtie_result = Array(int_bowtie_gpu)
    @test isapprox(int_bowtie_result, int_bowtie_cpu, rtol=1e-5)
    println("  BowtieFilter GPU vs CPU: PASS")

    # Test DetectorEfficiency on GPU
    println("\nDetectorEfficiency on GPU...")
    int_det_gpu = apply_detector_efficiency(copy(intensity_gpu), det_model, geom)
    int_det_result = Array(int_det_gpu)
    @test isapprox(int_det_result, int_det_cpu, rtol=1e-5)
    println("  DetectorEfficiency GPU vs CPU: PASS")

    println("\n=== All GPU tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Full Pipeline Test: HU Accuracy
# =============================================================================
println("\n=== Full Pipeline Test: HU Accuracy ===")

# Forward project with physics effects (CPU)
sino_physics = copy(sinogram_cpu)

# Apply physics effects in correct order (intensity domain)
intensity = exp.(-sino_physics)  # Convert to intensity

# 1. Flat filter attenuation
apply_flat_filter_to_intensity!(intensity, flat_model, geom)

# 2. Bowtie filter attenuation
apply_bowtie_to_intensity!(intensity, bowtie_model, geom)

# 3. Detector efficiency
apply_detector_efficiency!(intensity, det_model, geom)

# 4. Fill factor
apply_fill_factor_intensity!(intensity, ff_model)

# Convert back to projection domain
sino_physics = -log.(intensity)

println("Sinogram with physics effects:")
println("  Mean: $(mean(sino_physics))")
println("  Range: [$(minimum(sino_physics)), $(maximum(sino_physics))]")

# Reconstruct
recon_physics = fdk_reconstruct(sino_physics, geom, size(phantom.μ))

# Convert to HU (assuming water = 0.02 cm⁻¹)
μ_water = 0.02f0
hu_physics = @. (recon_physics - μ_water) / μ_water * 1000

# Check air regions (should be ~ -1000 HU)
# Check water regions (should be ~ 0 HU)
println("\nReconstruction with physics effects:")
println("  Recon range: [$(minimum(recon_physics)), $(maximum(recon_physics))]")
println("  HU range: [$(minimum(hu_physics)), $(maximum(hu_physics))]")

# Compare to reconstruction without physics
recon_no_physics = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))
hu_no_physics = @. (recon_no_physics - μ_water) / μ_water * 1000

println("\nReconstruction without physics effects:")
println("  HU range: [$(minimum(hu_no_physics)), $(maximum(hu_no_physics))]")

println("\n=== Phase 1 Physics Tests Complete ===")
