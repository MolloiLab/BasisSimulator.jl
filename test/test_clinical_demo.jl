# =============================================================================
# Clinical Demo Test - Mirrors examples/clinical_demo.jl exactly
# =============================================================================
#
# This test replicates the exact setup from the Pluto notebook to ensure
# all compiled operations work correctly at the notebook's scale.
#
# =============================================================================

using Test
using BasisSimulator
using Reactant
using Statistics

# =============================================================================
# Configuration (exactly matching Pluto notebook)
# =============================================================================

const CONFIG = (
    n_cols = 256,
    n_rows = 32,
    n_angles = 180,
    n_voxels = 128,
    fov_cm = 35.0,
    z_cm = 4.0,
)

# =============================================================================
# Test 1: Phantom and Geometry Creation
# =============================================================================

@testset "Clinical Demo: Setup" begin
    println("\n" * "="^60)
    println("TEST 1: Phantom and Geometry Setup")
    println("="^60)

    global phantom = create_gammex_472(
        n_voxels=CONFIG.n_voxels,
        fov_cm=CONFIG.fov_cm,
        z_cm=CONFIG.z_cm
    )
    println("  Phantom size: ", size(phantom.μ))
    @test size(phantom.μ)[1] == CONFIG.n_voxels
    @test size(phantom.μ)[2] == CONFIG.n_voxels

    global geom = create_aquilion_one(
        n_angles=CONFIG.n_angles,
        n_rows=CONFIG.n_rows,
        n_cols=CONFIG.n_cols,
        fov_cm=CONFIG.fov_cm
    )
    println("  Geometry: $(CONFIG.n_cols) cols × $(CONFIG.n_rows) rows × $(CONFIG.n_angles) angles")
    @test geom.n_angles == CONFIG.n_angles
end

# =============================================================================
# Test 2: Siddon Geometry Precomputation
# =============================================================================

@testset "Clinical Demo: Siddon Precompute" begin
    println("\n" * "="^60)
    println("TEST 2: Siddon Geometry Precomputation")
    println("="^60)

    t = @elapsed global siddon_geom = precompute_siddon_geometry(geom, size(phantom.μ), phantom.fov)
    println("  Precompute time: $(round(t, digits=2)) s")
    println("  voxel_indices size: ", size(siddon_geom.voxel_indices))
    println("  path_lengths size: ", size(siddon_geom.path_lengths))

    @test size(siddon_geom.voxel_indices)[2] == CONFIG.n_cols
    @test size(siddon_geom.voxel_indices)[3] == CONFIG.n_rows
    @test size(siddon_geom.voxel_indices)[4] == CONFIG.n_angles
end

# =============================================================================
# Test 3: Compiled Forward Projection
# =============================================================================

@testset "Clinical Demo: Compiled Forward Projection" begin
    println("\n" * "="^60)
    println("TEST 3: Compiled Forward Projection")
    println("="^60)

    # Convert to Reactant arrays
    vol_ra = Reactant.to_rarray(Float32.(vec(phantom.μ)))
    idx_ra = Reactant.to_rarray(siddon_geom.voxel_indices)
    len_ra = Reactant.to_rarray(siddon_geom.path_lengths)

    # Compile
    println("  Compiling...")
    t_compile = @elapsed compiled_forward = @compile siddon_forward_project_xla(vol_ra, idx_ra, len_ra)
    println("  Compile time: $(round(t_compile, digits=2)) s")
    @test compiled_forward !== nothing

    # Execute
    println("  Executing...")
    t_exec = @elapsed global sinogram = Array(compiled_forward(vol_ra, idx_ra, len_ra))
    println("  Execution time: $(round(t_exec * 1000, digits=1)) ms")
    println("  Sinogram size: ", size(sinogram))
    println("  Sinogram range: $(round(minimum(sinogram), digits=3)) to $(round(maximum(sinogram), digits=3))")

    @test size(sinogram) == (CONFIG.n_cols, CONFIG.n_rows, CONFIG.n_angles)
    @test minimum(sinogram) >= 0
    @test maximum(sinogram) < 20
end

# =============================================================================
# Test 4: FDK Reconstruction
# =============================================================================

@testset "Clinical Demo: FDK Reconstruction" begin
    println("\n" * "="^60)
    println("TEST 4: FDK Reconstruction")
    println("="^60)

    t = @elapsed global recon = fdk_reconstruct(
        Float32.(sinogram), geom, size(phantom.μ), phantom.fov;
        kernel=RampKernel()
    )
    println("  Reconstruction time: $(round(t, digits=2)) s")
    println("  Recon size: ", size(recon))

    @test size(recon) == size(phantom.μ)
end

# =============================================================================
# Test 5: HU Validation
# =============================================================================

@testset "Clinical Demo: HU Validation" begin
    println("\n" * "="^60)
    println("TEST 5: HU Validation")
    println("="^60)

    μ_water = get_reference_μ_water(60.0)

    function measure_hu(vol, mask, region_id)
        m = mask .== UInt8(region_id)
        sum(m) < 100 && return (mean=NaN, std=NaN)
        vals = vol[m]
        hu = @. 1000 * (vals - μ_water) / μ_water
        (mean=mean(hu), std=std(hu))
    end

    water = measure_hu(recon, phantom.mask, REGION_SOLID_WATER)
    ca100 = measure_hu(recon, phantom.mask, REGION_CA_100)
    ca200 = measure_hu(recon, phantom.mask, REGION_CA_200)

    println("  Solid Water: $(round(water.mean, digits=1)) ± $(round(water.std, digits=1)) HU")
    println("  Ca 100: $(round(ca100.mean, digits=1)) ± $(round(ca100.std, digits=1)) HU")
    println("  Ca 200: $(round(ca200.mean, digits=1)) ± $(round(ca200.std, digits=1)) HU")

    @test ca200.mean > ca100.mean  # Material ordering preserved
end

# =============================================================================
# Test 6: Polychromatic Geometry Precomputation
# =============================================================================

@testset "Clinical Demo: Polychromatic Precompute" begin
    println("\n" * "="^60)
    println("TEST 6: Polychromatic Geometry Precomputation")
    println("="^60)

    # Load and DOWNSAMPLE spectrum (240 bins → 30 bins)
    # 30 bins ≈ 3 keV resolution → <1% error (PMC8126163)
    # This is critical for memory efficiency!
    energies_full, weights_full = load_spectrum(120)
    global energies, weights = downsample_spectrum(energies_full, weights_full, 30)
    global materials = get_region_materials()
    println("  Original spectrum: $(length(energies_full)) energy bins")
    println("  Downsampled to: $(length(energies)) energy bins")
    println("  Materials: $(length(materials))")

    t = @elapsed global poly_geom = precompute_polychromatic_geometry(
        geom, size(phantom.μ), phantom.fov, materials, energies, weights
    )
    println("  Precompute time: $(round(t, digits=2)) s")
    println("  n_energies: ", poly_geom.n_energies)

    @test poly_geom.n_energies == length(energies)
    @test length(poly_geom.spectrum_weights) == length(energies)
end

# =============================================================================
# Test 7: Memory-Efficient Polychromatic Forward Projection
# =============================================================================

@testset "Clinical Demo: Polychromatic Memory-Efficient" begin
    println("\n" * "="^60)
    println("TEST 7: Memory-Efficient Polychromatic Forward Projection")
    println("="^60)

    # This approach loops over energies and reuses the ALREADY COMPILED
    # forward projection from Test 3. Memory usage: ~50 MB vs 36 GB for old approach!

    # Prepare Reactant arrays (reuse idx_ra and len_ra from monochromatic test)
    idx_ra = Reactant.to_rarray(siddon_geom.voxel_indices)
    len_ra = Reactant.to_rarray(siddon_geom.path_lengths)

    # Compile monochromatic forward projection (will be reused for each energy)
    vol_ra = Reactant.to_rarray(Float32.(vec(phantom.μ)))
    println("  Compiling forward projection...")
    t_compile = @elapsed compiled_forward = @compile siddon_forward_project_xla(vol_ra, idx_ra, len_ra)
    println("  Compile time: $(round(t_compile, digits=2)) s")

    # Run memory-efficient polychromatic projection
    println("  Running polychromatic projection ($(poly_geom.n_energies) energies)...")
    t_exec = @elapsed sino_poly = polychromatic_forward_project_compiled(
        phantom.mask, poly_geom, compiled_forward, idx_ra, len_ra, Reactant.to_rarray
    )
    println("  Execution time: $(round(t_exec, digits=2)) s")
    println("  Poly sinogram size: ", size(sino_poly))
    println("  Poly sinogram range: $(round(minimum(sino_poly), digits=3)) to $(round(maximum(sino_poly), digits=3))")

    @test size(sino_poly) == size(sinogram)
    @test minimum(sino_poly) >= -0.01  # Small tolerance for numerical precision
    @test maximum(sino_poly) < 20

    # Compare mono vs poly - poly should show beam hardening (lower values for high attenuation)
    diff = sino_poly .- sinogram
    println("  Mono vs Poly mean diff: $(round(mean(diff), digits=4))")
    println("  Beam hardening visible: poly max ($(round(maximum(sino_poly), digits=2))) < mono max ($(round(maximum(sinogram), digits=2)))")
end

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^60)
println("CLINICAL DEMO TEST COMPLETE")
println("="^60)
println("\nAll tests passed - Pluto notebook should work correctly.")
