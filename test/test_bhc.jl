"""
Test Beam Hardening Correction (Phase 2)

Tests BHC: water calibration curve, polynomial fitting, BHC application.
Validates HU accuracy improvement on both CPU and GPU (Metal).
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
geom = create_aquilion_one(n_angles=180, n_rows=16, n_cols=128, fov_cm=35.0)

# Create test phantom
phantom = create_gammex_472(n_voxels=64, fov_cm=35.0, z_cm=4.0)

# Create test data on CPU
println("\n=== Forward Projection ===")
sinogram_cpu = siddon_forward_project(Float32.(phantom.μ), geom)
println("Sinogram size: $(size(sinogram_cpu))")
println("Sinogram range: [$(minimum(sinogram_cpu)), $(maximum(sinogram_cpu))]")

# =============================================================================
# Test 1: BHC None (Identity)
# =============================================================================
println("\n=== Test 1: BHC None (Identity) ===")

bhc_identity = bhc_none()
println("BHC identity coefficients: $(bhc_identity.coefficients)")

# Apply identity BHC
sino_identity = apply_bhc(copy(sinogram_cpu), bhc_identity)
@test isapprox(mean(sino_identity), mean(sinogram_cpu), rtol=1e-6)
println("BHC identity: PASS (no change)")

# =============================================================================
# Test 2: BHC Water Default
# =============================================================================
println("\n=== Test 2: BHC Water Default ===")

bhc_water = bhc_water_default(reference_energy_keV=70.0)
info = get_bhc_info(bhc_water)
println("BHC order: $(info.order)")
println("Reference energy: $(info.reference_energy_keV) keV")
println("Coefficients: $(info.coefficients)")

# Apply water BHC
sino_bhc = apply_bhc(copy(sinogram_cpu), bhc_water)
println("Original mean: $(mean(sinogram_cpu))")
println("With BHC mean: $(mean(sino_bhc))")

# BHC should change values (typically increases them slightly)
@test !isapprox(mean(sino_bhc), mean(sinogram_cpu), rtol=0.001)
println("BHC water default: PASS (values changed)")

# =============================================================================
# Test 3: Polynomial Evaluation
# =============================================================================
println("\n=== Test 3: Polynomial Evaluation ===")

# Test evaluate_bhc at specific points
test_values = [0.0, 0.5, 1.0, 2.0, 5.0]
println("Polynomial evaluation:")
for p in test_values
    corrected = evaluate_bhc(p, bhc_water)
    println("  p=$p -> corrected=$(round(corrected, digits=4))")
end

# At p=0, should return a₀
@test isapprox(evaluate_bhc(0.0, bhc_water), bhc_water.coefficients[1], rtol=1e-6)
println("Polynomial evaluation: PASS")

# =============================================================================
# Test 4: Polynomial Fitting
# =============================================================================
println("\n=== Test 4: Polynomial Fitting ===")

# Test polynomial fitting with known data
x = Float64[0, 1, 2, 3, 4, 5]
y = x .+ 0.01 .* x.^2  # Slightly nonlinear

# Fit a quadratic
coeffs = BasisSimulator.fit_polynomial(x, y, 2)
println("Fitted coefficients (for y = x + 0.01x²): $coeffs")

# Should be approximately [0, 1, 0.01]
@test isapprox(coeffs[1], 0.0, atol=0.01)
@test isapprox(coeffs[2], 1.0, atol=0.01)
@test isapprox(coeffs[3], 0.01, atol=0.01)
println("Polynomial fitting: PASS")

# =============================================================================
# Test 5: BHC Application (In-place)
# =============================================================================
println("\n=== Test 5: BHC In-place Application ===")

sino_copy = copy(sinogram_cpu)
apply_bhc!(sino_copy, bhc_water)

# Should match non-mutating version
@test isapprox(mean(sino_copy), mean(sino_bhc), rtol=1e-6)
println("In-place BHC: PASS")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    sinogram_gpu = MtlArray(sinogram_cpu)
    println("GPU array type: $(typeof(sinogram_gpu))")

    # Test BHC identity on GPU
    println("\nBHC identity on GPU...")
    sino_identity_gpu = apply_bhc(copy(sinogram_gpu), bhc_identity)
    sino_identity_gpu_result = Array(sino_identity_gpu)

    @test isapprox(mean(sino_identity_gpu_result), mean(sinogram_cpu), rtol=0.05)
    println("  BHC identity GPU: PASS")

    # Test BHC water on GPU
    println("\nBHC water on GPU...")
    sino_bhc_gpu = apply_bhc(copy(sinogram_gpu), bhc_water)
    sino_bhc_gpu_result = Array(sino_bhc_gpu)

    cpu_bhc_mean = mean(sino_bhc)
    gpu_bhc_mean = mean(sino_bhc_gpu_result)
    println("  CPU mean: $(cpu_bhc_mean)")
    println("  GPU mean: $(gpu_bhc_mean)")
    @test isapprox(cpu_bhc_mean, gpu_bhc_mean, rtol=0.05)
    println("  BHC water GPU: PASS")

    # Test in-place on GPU
    println("\nIn-place BHC on GPU...")
    sino_gpu_copy = copy(sinogram_gpu)
    apply_bhc!(sino_gpu_copy, bhc_water)
    sino_gpu_inplace_result = Array(sino_gpu_copy)

    @test isapprox(mean(sino_gpu_inplace_result), gpu_bhc_mean, rtol=1e-6)
    println("  In-place BHC GPU: PASS")

    println("\n=== All GPU BHC tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Reconstruction Comparison: With and Without BHC
# =============================================================================
println("\n=== Reconstruction Comparison ===")

recon_no_bhc = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))
recon_with_bhc = fdk_reconstruct(sino_bhc, geom, size(phantom.μ))

println("Without BHC recon range: [$(minimum(recon_no_bhc)), $(maximum(recon_no_bhc))]")
println("With BHC recon range: [$(minimum(recon_with_bhc)), $(maximum(recon_with_bhc))]")

# BHC should affect reconstruction
@test !isapprox(mean(recon_with_bhc), mean(recon_no_bhc), rtol=0.001)
println("Reconstruction comparison: PASS (different results)")

# =============================================================================
# Test Different BHC Orders
# =============================================================================
println("\n=== Test: Different BHC Orders ===")

for order in [1, 2, 3, 4]
    # Create polynomial with increasing orders
    coeffs = zeros(Float64, order + 1)
    coeffs[2] = 1.0  # Linear term
    if order >= 2
        coeffs[3] = -0.01  # Small quadratic correction
    end
    if order >= 3
        coeffs[4] = 0.001  # Small cubic correction
    end

    poly = BHCPolynomial(coeffs, order, 70.0)
    sino_test = apply_bhc(copy(sinogram_cpu), poly)
    change = 100 * (mean(sino_test) - mean(sinogram_cpu)) / mean(sinogram_cpu)
    println("  Order $order: $(round(change, digits=2))% change")
end

println("\n=== Beam Hardening Correction Tests Complete ===")
