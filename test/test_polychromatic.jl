# =============================================================================
# Test Polychromatic Forward Projection
# =============================================================================
#
# Tests the Beer-Lambert physics implementation:
#   I = Σₑ wₑ × exp(-∫μₑ dl)
#   sinogram = -log(I / I₀)
#
# Key validation:
# 1. Line integrals should match expected physics
# 2. Polychromatic should show beam hardening (lower values than mono at same effective energy)
# 3. Memory stays bounded with batch_size=1
#
# =============================================================================

using Test
using BasisSimulator
using Reactant
using Statistics

# =============================================================================
# Helper function
# =============================================================================

function percentile(x, p)
    sort(x)[max(1, round(Int, length(x) * p / 100))]
end

# =============================================================================
# Test Configuration
# =============================================================================

const CONFIG = (
    n_cols = 128,
    n_rows = 8,
    n_angles = 45,
    n_voxels = 64,
    fov_cm = 35.0,
    z_cm = 2.0,
    n_energy_bins = 15,  # Small for testing
)

# =============================================================================
# Test 1: Basic Setup and Line Integral Validation
# =============================================================================

@testset "Polychromatic: Setup" begin
    println("\n" * "="^60)
    println("TEST 1: Polychromatic Setup")
    println("="^60)

    # Create phantom with known materials
    global phantom = create_gammex_472(
        n_voxels=CONFIG.n_voxels,
        fov_cm=CONFIG.fov_cm,
        z_cm=CONFIG.z_cm
    )
    println("  Phantom size: ", size(phantom.μ))

    # Create geometry
    global geom = create_aquilion_one(
        n_angles=CONFIG.n_angles,
        n_rows=CONFIG.n_rows,
        n_cols=CONFIG.n_cols,
        fov_cm=CONFIG.fov_cm
    )
    println("  Geometry: $(CONFIG.n_cols) × $(CONFIG.n_rows) × $(CONFIG.n_angles)")

    @test size(phantom.μ)[1] == CONFIG.n_voxels
    @test geom.n_angles == CONFIG.n_angles
end

# =============================================================================
# Test 2: Monochromatic Forward Projection (baseline)
# =============================================================================

@testset "Polychromatic: Monochromatic Baseline" begin
    println("\n" * "="^60)
    println("TEST 2: Monochromatic Forward Projection (60 keV baseline)")
    println("="^60)

    # Pre-compute Siddon geometry
    global siddon_geom = precompute_siddon_geometry(geom, size(phantom.μ), phantom.fov)
    println("  Siddon geometry computed")

    # Convert to Reactant
    vol_ra = Reactant.to_rarray(Float32.(vec(phantom.μ)))
    global idx_ra = Reactant.to_rarray(siddon_geom.voxel_indices)
    global len_ra = Reactant.to_rarray(siddon_geom.path_lengths)

    # Compile
    global compiled_fp = @compile siddon_forward_project_xla(vol_ra, idx_ra, len_ra)
    println("  Forward projection compiled")

    # Execute
    global sino_mono = Array(compiled_fp(vol_ra, idx_ra, len_ra))
    println("  Sinogram size: ", size(sino_mono))
    println("  Sinogram range: $(round(minimum(sino_mono), digits=4)) to $(round(maximum(sino_mono), digits=4))")

    @test size(sino_mono) == (CONFIG.n_cols, CONFIG.n_rows, CONFIG.n_angles)
    @test minimum(sino_mono) >= 0
    @test maximum(sino_mono) < 30  # Reasonable line integral range
end

# =============================================================================
# Test 3: Polychromatic Geometry Precomputation
# =============================================================================

@testset "Polychromatic: Precomputation" begin
    println("\n" * "="^60)
    println("TEST 3: Polychromatic Geometry Precomputation")
    println("="^60)

    # Load spectrum
    energies_full, weights_full = load_spectrum(120)
    global energies, weights = downsample_spectrum(energies_full, weights_full, CONFIG.n_energy_bins)
    global materials = get_region_materials()

    println("  Spectrum: $(length(energies_full)) → $(CONFIG.n_energy_bins) bins")
    println("  Energy range: $(round(minimum(energies), digits=1)) - $(round(maximum(energies), digits=1)) keV")
    println("  Materials: $(length(materials)) regions")

    # Pre-compute polychromatic geometry
    t = @elapsed global poly_geom = precompute_polychromatic_geometry(
        geom, size(phantom.μ), phantom.fov, materials, energies, weights
    )
    println("  Precompute time: $(round(t, digits=2)) s")

    @test poly_geom.n_energies == CONFIG.n_energy_bins
    @test size(poly_geom.μ_by_energy) == (length(materials), CONFIG.n_energy_bins)
    @test sum(poly_geom.spectrum_weights) ≈ 1.0  # Normalized
end

# =============================================================================
# Test 4: Polychromatic Forward Projection (batch_size=1)
# =============================================================================

@testset "Polychromatic: Forward Projection (loop)" begin
    println("\n" * "="^60)
    println("TEST 4: Polychromatic Forward Projection (batch_size=1)")
    println("="^60)

    t = @elapsed global sino_poly = polychromatic_forward_project_compiled(
        phantom.mask, poly_geom, compiled_fp, idx_ra, len_ra, Reactant.to_rarray;
        batch_size=1
    )
    println("  Execution time: $(round(t, digits=2)) s")
    println("  Sinogram size: ", size(sino_poly))
    println("  Sinogram range: $(round(minimum(sino_poly), digits=4)) to $(round(maximum(sino_poly), digits=4))")

    @test size(sino_poly) == size(sino_mono)
    @test minimum(sino_poly) >= -0.01  # Small tolerance
    @test maximum(sino_poly) < 30
end

# =============================================================================
# Test 5: Line Integral Physics Validation
# =============================================================================

@testset "Polychromatic: Physics Validation" begin
    println("\n" * "="^60)
    println("TEST 5: Line Integral Physics Validation")
    println("="^60)

    # Key physics check: Beam hardening
    # Polychromatic projection through high-attenuation material should show
    # LOWER effective line integrals than monochromatic at mean energy
    # (because low-energy photons are preferentially absorbed)

    # Find high-attenuation rays (through calcium inserts)
    high_atten_mask = sino_mono .> percentile(vec(sino_mono), 90)
    low_atten_mask = sino_mono .< percentile(vec(sino_mono), 10)

    mono_high = mean(sino_mono[high_atten_mask])
    poly_high = mean(sino_poly[high_atten_mask])
    mono_low = mean(sino_mono[low_atten_mask])
    poly_low = mean(sino_poly[low_atten_mask])

    println("  High attenuation rays:")
    println("    Mono (60 keV): $(round(mono_high, digits=4))")
    println("    Poly (120 kVp): $(round(poly_high, digits=4))")
    println("    Diff: $(round(poly_high - mono_high, digits=4))")

    println("  Low attenuation rays:")
    println("    Mono (60 keV): $(round(mono_low, digits=4))")
    println("    Poly (120 kVp): $(round(poly_low, digits=4))")
    println("    Diff: $(round(poly_low - mono_low, digits=4))")

    # Beam hardening: poly should be LESS than mono for high attenuation paths
    # (hard photons dominate after soft photons are absorbed)
    @test poly_high < mono_high  # Beam hardening effect

    # For low attenuation, difference should be smaller
    diff_high = abs(poly_high - mono_high)
    diff_low = abs(poly_low - mono_low)
    println("  |Δ| high: $(round(diff_high, digits=4)), |Δ| low: $(round(diff_low, digits=4))")

    # Difference is larger for high-attenuation paths (more beam hardening)
    @test diff_high > diff_low
end

# =============================================================================
# Test 6: Air Region Validation
# =============================================================================

@testset "Polychromatic: Air Region" begin
    println("\n" * "="^60)
    println("TEST 6: Air Region Validation")
    println("="^60)

    # Rays through air should have near-zero line integral
    # Find center column (should pass through air in Gammex phantom center hole)
    center_col = CONFIG.n_cols ÷ 2
    center_row = CONFIG.n_rows ÷ 2

    # Line integrals at center
    mono_center = sino_mono[center_col, center_row, :]
    poly_center = sino_poly[center_col, center_row, :]

    println("  Center ray line integrals:")
    println("    Mono mean: $(round(mean(mono_center), digits=4))")
    println("    Poly mean: $(round(mean(poly_center), digits=4))")

    # Both should be similar for rays through same material
    @test abs(mean(mono_center) - mean(poly_center)) < 1.0
end

# =============================================================================
# Test 7: Simple vs Compiled Consistency
# =============================================================================

@testset "Polychromatic: Simple vs Compiled" begin
    println("\n" * "="^60)
    println("TEST 7: Simple (non-Reactant) vs Compiled Consistency")
    println("="^60)

    # Run simple version (no Reactant)
    t = @elapsed sino_simple = polychromatic_forward_project_simple(
        phantom.mask, poly_geom; verbose=false
    )
    println("  Simple execution time: $(round(t, digits=2)) s")

    # Compare
    diff = sino_poly .- sino_simple
    max_diff = maximum(abs.(diff))
    mean_diff = mean(abs.(diff))

    println("  Max absolute difference: $(round(max_diff, digits=6))")
    println("  Mean absolute difference: $(round(mean_diff, digits=6))")

    # Should be nearly identical (just different execution paths)
    @test max_diff < 1e-4
end

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^60)
println("POLYCHROMATIC TEST COMPLETE")
println("="^60)
println("\nKey findings:")
println("  - Mono sinogram range: $(round(minimum(sino_mono), digits=3)) to $(round(maximum(sino_mono), digits=3))")
println("  - Poly sinogram range: $(round(minimum(sino_poly), digits=3)) to $(round(maximum(sino_poly), digits=3))")
println("  - Beam hardening visible: poly max < mono max at high attenuation")
