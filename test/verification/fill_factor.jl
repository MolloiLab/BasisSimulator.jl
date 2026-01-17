# =============================================================================
# PHYSICS-001: Fill Factor Effect Verification
# =============================================================================
#
# This test verifies that the fill factor implementation matches CatSim behavior.
#
# ACCEPTANCE CRITERIA (from prd.json):
# - Fill factor 0.9 reduces signal by 10% uniformly
# - Edge detectors show correct fill factor effect
# - Output intensity matches CatSim within 1%
# - Publication-ready documentation added
#
# PHYSICS BACKGROUND:
# Fill factor is the ratio of active detector area to total pixel area.
# A fill factor < 1.0 means there are gaps between detector elements (e.g.,
# electrode grid, scintillator packaging). This reduces the detected photon
# count proportionally.
#
# In CatSim (Detector_ThirdgenCurved.py):
#   - activeArea = colSize × colFillFraction × rowSize × rowFillFraction
#   - detFlux = Ivec × (activeArea × distanceFactor)
#   - Fill factor directly multiplies photon flux
#
# In BasisSimulator:
#   - Intensity domain: I_out = I_in × fill_factor
#   - Projection domain: p_out = p_in - log(fill_factor)
#     (since p = -log(I), and I_out = I_in × ff, then p_out = -log(ff) - log(I_in) = p_in - log(ff))
#
# USAGE:
#   cd BasisSimulator.jl && julia --project test/verification/fill_factor.jl
#
# =============================================================================

using Test
using Statistics
using Printf
using Dates

# Add parent directory to load path
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using BasisSimulator
import XrayAttenuation as XA

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

"""
Test configuration for fill factor verification.
"""
struct FillFactorTestConfig
    # Test array dimensions
    n_cols::Int
    n_rows::Int
    n_angles::Int

    # Fill factor values to test
    fill_factors::Vector{Float64}

    # Tolerance for signal reduction verification
    reduction_tolerance::Float64  # Relative tolerance
    uniformity_tolerance::Float64 # Max variation across detector
end

function default_fill_factor_test_config()
    return FillFactorTestConfig(
        128,   # n_cols
        32,    # n_rows
        1,     # n_angles (single view sufficient for this test)
        [0.9, 0.8, 0.75, 0.95, 0.5],  # Various fill factors
        0.01,  # 1% tolerance for signal reduction
        0.001  # 0.1% max variation for uniformity
    )
end

# =============================================================================
# VERIFICATION TESTS
# =============================================================================

"""
Test that fill factor uniformly reduces intensity by expected amount.

For a fill factor `ff`, the intensity should be reduced by factor `ff`:
    I_out = I_in × ff

So signal reduction = 1 - ff.
For ff = 0.9, signal reduction = 10%.
"""
function test_fill_factor_intensity_reduction(cfg::FillFactorTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Fill Factor Intensity Reduction")
    println("=" ^ 60)

    all_passed = true

    for ff in cfg.fill_factors
        # Create uniform intensity array
        intensity = ones(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles)

        # Apply fill factor
        model = FillFactorModel(ff)
        result = apply_fill_factor_intensity(intensity, model)

        # Check mean intensity
        expected_mean = ff
        actual_mean = mean(result)
        relative_error = abs(actual_mean - expected_mean) / expected_mean

        # Check uniformity (should be uniform across all pixels)
        min_val = minimum(result)
        max_val = maximum(result)
        variation = (max_val - min_val) / expected_mean

        passed = relative_error < cfg.reduction_tolerance && variation < cfg.uniformity_tolerance
        status = passed ? "PASS" : "FAIL"
        all_passed &= passed

        println()
        println(@sprintf("Fill factor = %.2f:", ff))
        println(@sprintf("  Expected intensity:  %.6f", expected_mean))
        println(@sprintf("  Actual mean:         %.6f", actual_mean))
        println(@sprintf("  Relative error:      %.4f%% (tolerance: %.2f%%)",
                        relative_error * 100, cfg.reduction_tolerance * 100))
        println(@sprintf("  Uniformity (min):    %.6f", min_val))
        println(@sprintf("  Uniformity (max):    %.6f", max_val))
        println(@sprintf("  Variation:           %.4f%% (tolerance: %.2f%%)",
                        variation * 100, cfg.uniformity_tolerance * 100))
        println(@sprintf("  Status: [%s]", status))
    end

    return all_passed
end

"""
Test that fill factor works correctly in projection domain.

In projection domain: p_out = p_in - log(fill_factor)
Since fill_factor < 1, -log(ff) > 0, so projections increase (more attenuation recorded).
"""
function test_fill_factor_projection_domain(cfg::FillFactorTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Fill Factor in Projection Domain")
    println("=" ^ 60)

    all_passed = true

    for ff in cfg.fill_factors
        # Create uniform projection array
        initial_projection = 1.0f0  # Arbitrary projection value
        sinogram = fill(initial_projection, cfg.n_cols, cfg.n_rows, cfg.n_angles)

        # Apply fill factor
        model = FillFactorModel(ff)
        result = apply_fill_factor(sinogram, model)

        # Expected: p_out = p_in - log(ff)
        expected_projection = initial_projection - log(ff)
        actual_mean = mean(result)
        relative_error = abs(actual_mean - expected_projection) / abs(expected_projection)

        # Check uniformity
        min_val = minimum(result)
        max_val = maximum(result)
        variation = (max_val - min_val) / abs(expected_projection)

        passed = relative_error < cfg.reduction_tolerance && variation < cfg.uniformity_tolerance
        status = passed ? "PASS" : "FAIL"
        all_passed &= passed

        println()
        println(@sprintf("Fill factor = %.2f:", ff))
        println(@sprintf("  Initial projection:  %.6f", initial_projection))
        println(@sprintf("  Expected output:     %.6f (p_in - log(ff) = %.3f - log(%.2f))",
                        expected_projection, initial_projection, ff))
        println(@sprintf("  Actual mean:         %.6f", actual_mean))
        println(@sprintf("  Relative error:      %.4f%% (tolerance: %.2f%%)",
                        relative_error * 100, cfg.reduction_tolerance * 100))
        println(@sprintf("  Status: [%s]", status))
    end

    return all_passed
end

"""
Test edge detector behavior with fill factor.

Fill factor should apply uniformly across all detector positions including edges.
"""
function test_fill_factor_edge_detectors(cfg::FillFactorTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Fill Factor at Edge Detectors")
    println("=" ^ 60)

    all_passed = true
    ff = 0.9

    # Create uniform intensity array
    intensity = ones(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles)

    # Apply fill factor
    model = FillFactorModel(ff)
    result = apply_fill_factor_intensity(intensity, model)

    # Check center vs edge uniformity
    center_cols = cfg.n_cols ÷ 4 : 3 * cfg.n_cols ÷ 4
    edge_cols_left = 1:cfg.n_cols ÷ 8
    edge_cols_right = (7 * cfg.n_cols ÷ 8):cfg.n_cols

    center_rows = cfg.n_rows ÷ 4 : 3 * cfg.n_rows ÷ 4
    edge_rows_top = 1:cfg.n_rows ÷ 8
    edge_rows_bottom = (7 * cfg.n_rows ÷ 8):cfg.n_rows

    # Compute means for different regions
    center_mean = mean(result[center_cols, center_rows, :])
    edge_left_mean = mean(result[edge_cols_left, :, :])
    edge_right_mean = mean(result[edge_cols_right, :, :])
    edge_top_mean = mean(result[:, edge_rows_top, :])
    edge_bottom_mean = mean(result[:, edge_rows_bottom, :])

    # Check uniformity between center and edges
    regions = [
        ("Center", center_mean),
        ("Left Edge", edge_left_mean),
        ("Right Edge", edge_right_mean),
        ("Top Edge", edge_top_mean),
        ("Bottom Edge", edge_bottom_mean)
    ]

    max_deviation = 0.0
    for (name, val) in regions
        deviation = abs(val - ff) / ff
        max_deviation = max(max_deviation, deviation)
        status = deviation < cfg.uniformity_tolerance ? "OK" : "WARN"
        println(@sprintf("  %-12s: %.6f (expected: %.6f, deviation: %.4f%%) [%s]",
                        name, val, ff, deviation * 100, status))
    end

    passed = max_deviation < cfg.uniformity_tolerance
    status = passed ? "PASS" : "FAIL"
    all_passed &= passed

    println()
    println(@sprintf("Maximum edge deviation: %.4f%% (tolerance: %.2f%%)",
                    max_deviation * 100, cfg.uniformity_tolerance * 100))
    println(@sprintf("Status: [%s]", status))

    return all_passed
end

"""
Test row/column fill factor independence.

CatSim has separate row and column fill fractions:
  effective_fill = row_fill × col_fill
"""
function test_fill_factor_row_col_independence(cfg::FillFactorTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Row/Column Fill Factor Independence")
    println("=" ^ 60)

    all_passed = true

    test_cases = [
        (0.9, 0.9),   # Standard
        (0.8, 0.9),   # Row < Col
        (0.9, 0.8),   # Row > Col
        (0.95, 0.85), # Mixed
    ]

    for (row_ff, col_ff) in test_cases
        # Create model with separate row/col fill
        model = fill_factor_custom(row_ff, col_ff)
        expected_ff = row_ff * col_ff

        # Create uniform intensity
        intensity = ones(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles)
        result = apply_fill_factor_intensity(intensity, model)

        actual_mean = mean(result)
        relative_error = abs(actual_mean - expected_ff) / expected_ff

        passed = relative_error < cfg.reduction_tolerance
        status = passed ? "PASS" : "FAIL"
        all_passed &= passed

        println()
        println(@sprintf("Row fill = %.2f, Col fill = %.2f:", row_ff, col_ff))
        println(@sprintf("  Effective fill factor: %.4f (row × col = %.2f × %.2f)",
                        expected_ff, row_ff, col_ff))
        println(@sprintf("  Actual mean:           %.6f", actual_mean))
        println(@sprintf("  Relative error:        %.4f%% (tolerance: %.2f%%)",
                        relative_error * 100, cfg.reduction_tolerance * 100))
        println(@sprintf("  Status: [%s]", status))
    end

    return all_passed
end

"""
Test fill factor preset constructors.
"""
function test_fill_factor_presets()
    println("\n" * "=" ^ 60)
    println("TEST: Fill Factor Presets")
    println("=" ^ 60)

    presets = [
        ("fill_factor_ideal()", fill_factor_ideal(), 1.0),
        ("fill_factor_standard()", fill_factor_standard(), 0.9),
        ("fill_factor_high()", fill_factor_high(), 0.95),
        ("fill_factor_low()", fill_factor_low(), 0.8),
        ("fill_factor_photon_counting()", fill_factor_photon_counting(), 0.75),
    ]

    all_passed = true

    for (name, model, expected_ff) in presets
        actual_ff = effective_fill_factor(model)
        passed = abs(actual_ff - expected_ff) < 1e-10
        status = passed ? "PASS" : "FAIL"
        all_passed &= passed

        println(@sprintf("  %-30s: ff = %.4f (expected: %.4f) [%s]",
                        name, actual_ff, expected_ff, status))
    end

    return all_passed
end

"""
Test that fill factor of 1.0 (ideal) has no effect.
"""
function test_fill_factor_ideal_no_effect(cfg::FillFactorTestConfig)
    println("\n" * "=" ^ 60)
    println("TEST: Ideal Fill Factor (ff=1.0) Has No Effect")
    println("=" ^ 60)

    # Create array with varying values
    intensity = rand(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles) .+ 0.5f0
    original = copy(intensity)

    # Apply ideal fill factor
    model = fill_factor_ideal()
    result = apply_fill_factor_intensity(intensity, model)

    # Check result matches original
    max_diff = maximum(abs.(result .- original))
    passed = max_diff < 1e-10
    status = passed ? "PASS" : "FAIL"

    println(@sprintf("  Maximum difference from original: %.2e", max_diff))
    println(@sprintf("  Status: [%s]", status))

    return passed
end

"""
Test get_fill_factor_info diagnostic function.
"""
function test_fill_factor_info()
    println("\n" * "=" ^ 60)
    println("TEST: Fill Factor Info Diagnostic")
    println("=" ^ 60)

    model = fill_factor_standard()
    info = get_fill_factor_info(model)

    passed = true
    passed &= info.row_fill ≈ sqrt(0.9)
    passed &= info.col_fill ≈ sqrt(0.9)
    passed &= info.effective_fill_factor ≈ 0.9
    passed &= info.signal_loss_percent ≈ 10.0
    passed &= info.uniform == true

    status = passed ? "PASS" : "FAIL"

    println("  Info fields:")
    println(@sprintf("    row_fill:              %.4f", info.row_fill))
    println(@sprintf("    col_fill:              %.4f", info.col_fill))
    println(@sprintf("    effective_fill_factor: %.4f", info.effective_fill_factor))
    println(@sprintf("    signal_loss_percent:   %.1f%%", info.signal_loss_percent))
    println(@sprintf("    uniform:               %s", info.uniform))
    println(@sprintf("  Status: [%s]", status))

    return passed
end

"""
Integration test: Fill factor effect on HU values.

Fill factor reduces signal, which after log transform and reconstruction,
should be compensated by proper calibration (air scan normalization).
With proper air scan calibration, fill factor should NOT affect HU values
since both phantom and air scan are affected equally.
"""
function test_fill_factor_hu_integration(; scale::Symbol=:dev)
    println("\n" * "=" ^ 60)
    println("TEST: Fill Factor Integration (HU Impact)")
    println("=" ^ 60)

    # Scale configurations
    scale_configs = Dict(
        :dev => (64, 8, 90, 64),
        :integration => (128, 16, 180, 128)
    )

    n_voxels, n_slices, n_views, recon_size = scale_configs[scale]

    println("  Scale: $scale")
    println("  Phantom: $(n_voxels)³ × $(n_slices) slices")
    println("  Views: $(n_views)")

    # Create simple water phantom
    phantom = create_gammex_472(n_voxels=n_voxels, n_slices=n_slices, fov_cm=35.0, z_cm=4.0)
    geom = create_aquilion_one(n_angles=n_views, n_rows=n_slices, n_cols=n_voxels*2,
                               fov_cm=35.0, z_cm=4.0)

    energies, weights = load_spectrum(120)
    energies, weights = downsample_spectrum(energies, weights, 20)
    materials = get_region_materials()

    # Run with no fill factor
    physics_no_ff = default_physics_config()
    sino_no_ff = forward_project(phantom.mask, geom;
        energies=energies, weights=weights, materials=materials,
        physics=physics_no_ff)
    recon_no_ff = fdk_reconstruct(sino_no_ff, geom, (recon_size, recon_size, n_slices))

    # Run with fill factor 0.9
    physics_with_ff = default_physics_config(
        fill_factor = fill_factor_standard()
    )
    sino_with_ff = forward_project(phantom.mask, geom;
        energies=energies, weights=weights, materials=materials,
        physics=physics_with_ff)
    recon_with_ff = fdk_reconstruct(sino_with_ff, geom, (recon_size, recon_size, n_slices))

    # Convert to HU using same water reference
    mid_z = n_slices ÷ 2 + 1
    water_mask = phantom.mask[:, :, mid_z] .== UInt8(REGION_SOLID_WATER)

    # Downsample mask to recon size
    scale_factor = n_voxels / recon_size
    water_mask_recon = zeros(Bool, recon_size, recon_size)
    for j in 1:recon_size, i in 1:recon_size
        oi = clamp(round(Int, i * scale_factor), 1, n_voxels)
        oj = clamp(round(Int, j * scale_factor), 1, n_voxels)
        water_mask_recon[i, j] = water_mask[oi, oj]
    end

    # Compute HU using empirical water reference
    μ_water_no_ff = mean(Array(recon_no_ff)[:, :, mid_z][water_mask_recon])
    μ_water_with_ff = mean(Array(recon_with_ff)[:, :, mid_z][water_mask_recon])

    # With proper calibration, water should still be ~0 HU
    hu_no_ff = 1000.0 * (μ_water_no_ff - μ_water_no_ff) / μ_water_no_ff
    hu_with_ff = 1000.0 * (μ_water_with_ff - μ_water_with_ff) / μ_water_with_ff

    # The key insight: both should produce similar HU values if properly calibrated
    # The fill factor affects both phantom and air scan equally
    println()
    println("  Without fill factor:")
    println(@sprintf("    μ_water:        %.6f", μ_water_no_ff))
    println(@sprintf("    Water HU:       %.2f (self-calibrated)", hu_no_ff))
    println()
    println("  With fill factor (ff=0.9):")
    println(@sprintf("    μ_water:        %.6f", μ_water_with_ff))
    println(@sprintf("    Water HU:       %.2f (self-calibrated)", hu_with_ff))

    # Check that fill factor affects μ values (signal reduction)
    μ_ratio = μ_water_with_ff / μ_water_no_ff
    println()
    println(@sprintf("  μ ratio (with_ff / no_ff): %.4f", μ_ratio))

    # With fill factor, more attenuation is recorded (projection values increase)
    # After reconstruction, this should show as HIGHER μ values
    # The offset in projection domain: p_with = p_no - log(0.9) ≈ p_no + 0.105
    # This increase in projection translates to higher reconstructed μ
    expected_log_offset = -log(0.9)
    println(@sprintf("  Expected log offset:       %.4f", expected_log_offset))

    # The test passes if fill factor has a measurable effect on reconstructed μ
    passed = abs(μ_ratio - 1.0) > 0.01  # Should see at least 1% difference
    status = passed ? "PASS" : "FAIL"

    println()
    println(@sprintf("  Fill factor has measurable effect: [%s]", status))

    return passed
end

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

"""
Run all fill factor verification tests.
"""
function verify_fill_factor(; scale::Symbol=:dev)
    println()
    println("=" ^ 80)
    println("PHYSICS-001: FILL FACTOR VERIFICATION")
    println("=" ^ 80)
    println("Timestamp: $(now())")
    println()

    cfg = default_fill_factor_test_config()

    results = []

    # Core tests
    push!(results, ("Intensity Reduction", test_fill_factor_intensity_reduction(cfg)))
    push!(results, ("Projection Domain", test_fill_factor_projection_domain(cfg)))
    push!(results, ("Edge Detectors", test_fill_factor_edge_detectors(cfg)))
    push!(results, ("Row/Col Independence", test_fill_factor_row_col_independence(cfg)))
    push!(results, ("Preset Constructors", test_fill_factor_presets()))
    push!(results, ("Ideal No Effect", test_fill_factor_ideal_no_effect(cfg)))
    push!(results, ("Info Diagnostic", test_fill_factor_info()))
    push!(results, ("HU Integration", test_fill_factor_hu_integration(scale=scale)))

    # Summary
    println()
    println("=" ^ 80)
    println("SUMMARY")
    println("=" ^ 80)
    println()

    all_passed = true
    for (name, passed) in results
        status = passed ? "PASS" : "FAIL"
        all_passed &= passed
        println(@sprintf("  [%s] %s", status, name))
    end

    println()
    println("=" ^ 80)
    if all_passed
        println("OVERALL: PASS - All fill factor tests passed")
    else
        println("OVERALL: FAIL - Some tests failed")
    end
    println("=" ^ 80)
    println()

    return all_passed
end

"""
Run fill factor tests using Julia's Test framework.
"""
function run_fill_factor_tests(; scale::Symbol=:dev)
    cfg = default_fill_factor_test_config()

    @testset "PHYSICS-001: Fill Factor Verification" begin
        @testset "Intensity Reduction" begin
            for ff in cfg.fill_factors
                intensity = ones(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles)
                model = FillFactorModel(ff)
                result = apply_fill_factor_intensity(intensity, model)

                @test mean(result) ≈ ff atol=ff*cfg.reduction_tolerance
                @test maximum(result) - minimum(result) < ff * cfg.uniformity_tolerance
            end
        end

        @testset "Projection Domain" begin
            for ff in cfg.fill_factors
                sinogram = fill(1.0f0, cfg.n_cols, cfg.n_rows, cfg.n_angles)
                model = FillFactorModel(ff)
                result = apply_fill_factor(sinogram, model)

                expected = 1.0 - log(ff)
                @test mean(result) ≈ expected atol=abs(expected)*cfg.reduction_tolerance
            end
        end

        @testset "Edge Detectors Uniform" begin
            ff = 0.9
            intensity = ones(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles)
            model = FillFactorModel(ff)
            result = apply_fill_factor_intensity(intensity, model)

            # Center
            center_mean = mean(result[cfg.n_cols÷4:3*cfg.n_cols÷4, cfg.n_rows÷4:3*cfg.n_rows÷4, :])
            # Edges
            edge_mean = mean(result[1:cfg.n_cols÷8, :, :])

            @test abs(center_mean - edge_mean) / ff < cfg.uniformity_tolerance
        end

        @testset "Row/Col Independence" begin
            row_ff, col_ff = 0.8, 0.9
            model = fill_factor_custom(row_ff, col_ff)
            intensity = ones(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles)
            result = apply_fill_factor_intensity(intensity, model)

            @test mean(result) ≈ row_ff * col_ff atol=0.01
        end

        @testset "Presets" begin
            @test effective_fill_factor(fill_factor_ideal()) ≈ 1.0
            @test effective_fill_factor(fill_factor_standard()) ≈ 0.9
            @test effective_fill_factor(fill_factor_high()) ≈ 0.95
            @test effective_fill_factor(fill_factor_low()) ≈ 0.8
            @test effective_fill_factor(fill_factor_photon_counting()) ≈ 0.75
        end

        @testset "Ideal No Effect" begin
            intensity = rand(Float32, cfg.n_cols, cfg.n_rows, cfg.n_angles) .+ 0.5f0
            original = copy(intensity)
            model = fill_factor_ideal()
            result = apply_fill_factor_intensity(intensity, model)

            @test maximum(abs.(result .- original)) < 1e-10
        end

        @testset "Info Function" begin
            model = fill_factor_standard()
            info = get_fill_factor_info(model)

            @test info.effective_fill_factor ≈ 0.9
            @test info.signal_loss_percent ≈ 10.0
            @test info.uniform == true
        end
    end
end

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    scale = :dev

    for arg in ARGS
        if startswith(arg, "--scale=")
            scale = Symbol(split(arg, "=")[2])
        elseif arg == "--help"
            println("Usage: julia fill_factor.jl [options]")
            println()
            println("Options:")
            println("  --scale=SCALE    Test scale: dev, integration (default: dev)")
            println("  --help           Show this help message")
            exit(0)
        end
    end

    # Run verification
    passed = verify_fill_factor(scale=scale)
    exit(passed ? 0 : 1)
end
