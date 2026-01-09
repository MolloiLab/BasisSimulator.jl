"""
BasisSimulator.jl Test Suite

Comprehensive tests for publication-grade CT simulator.

# Test Organization

- `test_spectrum.jl` - X-ray spectrum generation (Boone & Seibert 1997)
- `test_fdk.jl` - FDK reconstruction (Feldkamp et al. 1984)
- `test_physics_validation.jl` - **ACTIVE** Physics validation (NIST, HU, kVp)
- `test_gecatsim_validation.jl` - **ACTIVE** GECATSIM comparison (optional)
- `test_ray_tracing.jl` - Ray tracing (Amanatides & Woo 1987) [TODO]
- `test_gradients.jl` - Enzyme autodiff validation [TODO]

# Running Tests

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run specific test file
julia --project=. test/test_fdk.jl

# Run with multiple threads
julia --project=. -t 8 test/runtests.jl
```

# Coverage Target

- Unit tests: >95% coverage
- Integration tests: Full pipeline validation
- GECATSIM validation: RMSE <5%, SSIM >0.95
"""

using Test
using BasisSimulator

# Print test environment info
println("="^70)
println("BasisSimulator.jl Test Suite")
println("="^70)
println("Julia version: $(VERSION)")
println("BasisSimulator version: $(BasisSimulator.version())")
println("Number of threads: $(Threads.nthreads())")
println("="^70)
println()

# Track test timing
test_start = time()

# ============================================================================
# Test Modules
# ============================================================================

@testset "BasisSimulator.jl" begin

    # ------------------------------------------------------------------------
    # Physics Module Tests
    # ------------------------------------------------------------------------
    @testset "Physics" begin
        println("\n📡 Testing Physics/Spectrum.jl...")
        include("test_spectrum.jl")

        # TODO: Uncomment as modules are implemented
        # println("\n⚛️  Testing Physics/Attenuation.jl...")
        # include("test_attenuation.jl")

        # println("\n🌊 Testing Physics/Scatter.jl...")
        # include("test_scatter.jl")

        # println("\n📟 Testing Physics/Detector.jl...")
        # include("test_detector.jl")

        # println("\n📊 Testing Physics/Noise.jl...")
        # include("test_noise.jl")
    end

    # ------------------------------------------------------------------------
    # Geometry Module Tests
    # ------------------------------------------------------------------------
    @testset "Geometry" begin
        # TODO: Uncomment as modules are implemented
        # println("\n📐 Testing Geometry/ScannerGeometry.jl...")
        # include("test_scanner_geometry.jl")

        # println("\n🔦 Testing Geometry/RayTracing.jl...")
        # include("test_ray_tracing.jl")

        # println("\n🎭 Testing Geometry/Phantoms.jl...")
        # include("test_phantoms.jl")
    end

    # ------------------------------------------------------------------------
    # Reconstruction Module Tests
    # ------------------------------------------------------------------------
    @testset "Reconstruction" begin
        println("\n🔄 Testing Reconstruction/FDK.jl...")
        include("test_fdk.jl")

        # TODO: Uncomment as modules are implemented
        # println("\n🔁 Testing Reconstruction/Iterative.jl...")
        # include("test_iterative.jl")

        # println("\n⚙️  Testing Reconstruction/Corrections.jl...")
        # include("test_corrections.jl")
    end

    # ------------------------------------------------------------------------
    # Simulation Module Tests
    # ------------------------------------------------------------------------
    # @testset "Simulation" begin
    #     println("\n🚀 Testing Simulation.jl...")
    #     include("test_simulation.jl")
    # end

    # ------------------------------------------------------------------------
    # Validation Module Tests (CRITICAL FOR PUBLICATION)
    # ------------------------------------------------------------------------
    @testset "Validation" begin
        println("\n⚛️  Testing physics validation (NIST, HU, kVp effects)...")
        include("test_physics_validation.jl")

        println("\n🎯 Testing GECATSIM comparison (CRITICAL)...")
        include("test_gecatsim_validation.jl")

        # TODO: Uncomment as modules are implemented
        # println("\n✅ Testing Validation/Metrics.jl...")
        # include("test_metrics.jl")
    end

    # ------------------------------------------------------------------------
    # Gradient/Autodiff Tests (CRITICAL FOR PUBLICATION)
    # ------------------------------------------------------------------------
    # @testset "Autodifferentiation" begin
    #     println("\n∇ Testing Enzyme.jl gradients...")
    #     include("test_gradients.jl")
    # end

    # ------------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------------
    # @testset "Integration" begin
    #     println("\n🔗 Testing end-to-end pipeline...")
    #     include("test_end_to_end.jl")
    # end

end

# ============================================================================
# Test Summary
# ============================================================================

test_duration = time() - test_start

println()
println("="^70)
println("Test Suite Complete")
println("="^70)
println("Total time: $(round(test_duration, digits=2)) seconds")
println("="^70)
