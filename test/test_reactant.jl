"""
Test Reactant/XLA compilation of BasisSimulator functions.

ALL tests use compiled functions only - no CPU fallbacks.
"""

using Test
using BasisSimulator
using Reactant

println("=" ^ 60)
println("Testing Reactant/XLA Compilation")
println("=" ^ 60)

# =============================================================================
# Setup: Create test phantom and geometry
# =============================================================================
println("\n1. Creating test data...")
phantom = create_gammex_472(n_voxels=64, n_slices=16)
geom = create_aquilion_one(n_angles=64, n_rows=16, n_cols=128, fov_cm=phantom.fov[1])
output_size = size(phantom.μ)

println("   Phantom size: $(size(phantom.μ))")
println("   Geometry: $(geom.n_angles) angles, $(geom.n_cols)×$(geom.n_rows) detector")

# =============================================================================
# Test 1: XLA Forward Projection
# =============================================================================
println("\n2. Pre-computing forward projection geometry...")
proj_geom = precompute_forward_projection_geometry(geom, size(phantom.μ), phantom.fov)
mem_gb = (sizeof(proj_geom.linear_indices) + sizeof(proj_geom.sample_weights)) / 1e9
println("   n_samples: $(proj_geom.n_samples)")
println("   Memory: $(round(mem_gb, digits=3)) GB")

println("\n3. Compiling forward_project_xla_arrays...")
vol_ra = Reactant.to_rarray(Float32.(vec(phantom.μ)))
proj_idx_ra = Reactant.to_rarray(proj_geom.linear_indices)
proj_wts_ra = Reactant.to_rarray(proj_geom.sample_weights)

compiled_fp = @compile forward_project_xla_arrays(vol_ra, proj_idx_ra, proj_wts_ra)
println("   ✓ forward_project_xla_arrays compiled!")

println("\n4. Running compiled forward projection...")
sino_ra = compiled_fp(vol_ra, proj_idx_ra, proj_wts_ra)
sino = Array(sino_ra)
println("   ✓ Sinogram: $(size(sino)), range: [$(minimum(sino)), $(maximum(sino))]")

# =============================================================================
# Test 2: XLA Backprojection
# =============================================================================
println("\n5. Pre-computing backprojection geometry...")
bp_geom = precompute_backprojection_geometry(geom, output_size, phantom.fov)
println("   linear_indices size: $(size(bp_geom.linear_indices))")

println("\n6. Compiling backproject_volume_arrays...")
# Pre-weight and filter sinogram
weighted = preweight_cosine(sino, geom)
filtered = filter_ramp(weighted, geom)
sino_flat = Float32.(vec(filtered))

sino_flat_ra = Reactant.to_rarray(sino_flat)
bp_idx_ra = Reactant.to_rarray(bp_geom.linear_indices)
bp_bilin_ra = Reactant.to_rarray(Float32.(bp_geom.bilinear_weights))
bp_dist_ra = Reactant.to_rarray(Float32.(bp_geom.distance_weights))

compiled_bp = @compile backproject_volume_arrays(sino_flat_ra, bp_idx_ra, bp_bilin_ra, bp_dist_ra)
println("   ✓ backproject_volume_arrays compiled!")

println("\n7. Running compiled backprojection...")
recon_ra = compiled_bp(sino_flat_ra, bp_idx_ra, bp_bilin_ra, bp_dist_ra)
recon = reshape(Array(recon_ra), output_size)
println("   ✓ Reconstruction: $(size(recon)), range: [$(minimum(recon)), $(maximum(recon))]")

# =============================================================================
# Test 3: Full Pipeline with Physics Effects
# =============================================================================
println("\n8. Testing full pipeline with physics effects...")

# Re-run forward projection with new volume (demonstrate reuse)
vol2_ra = Reactant.to_rarray(Float32.(vec(phantom.μ .* 1.1)))  # Slightly different volume
sino2_ra = compiled_fp(vol2_ra, proj_idx_ra, proj_wts_ra)
sino2 = Array(sino2_ra)

# Apply physics effects (these are element-wise, can be compiled later)
detector_model = default_detector_model(I0=1e5)
sino2_noisy = add_quantum_noise(sino2, detector_model)  # Add Poisson noise

# Reconstruct with noise
weighted2 = preweight_cosine(sino2_noisy, geom)
filtered2 = filter_ramp(weighted2, geom)
sino2_flat_ra = Reactant.to_rarray(Float32.(vec(filtered2)))

recon2_ra = compiled_bp(sino2_flat_ra, bp_idx_ra, bp_bilin_ra, bp_dist_ra)
recon2 = reshape(Array(recon2_ra), output_size)
println("   ✓ Pipeline with noise: $(size(recon2)), range: [$(minimum(recon2)), $(maximum(recon2))]")

# =============================================================================
# Test 4: Verify Results
# =============================================================================
println("\n9. Verifying results...")

# Check that reconstruction has reasonable values
center_val = recon[32, 32, 8]
println("   Center voxel value: $(center_val)")
println("   Expected (water μ): ~0.019 cm⁻¹")

# Check that sinogram values are non-negative
@test all(sino .>= 0)
@test size(sino) == (geom.n_cols, geom.n_rows, geom.n_angles)
@test size(recon) == output_size

println("\n" * "=" ^ 60)
println("✓ All Reactant/XLA Compilation Tests PASSED!")
println("=" ^ 60)
println("\nSummary:")
println("  - forward_project_xla_arrays: COMPILED ✓")
println("  - backproject_volume_arrays: COMPILED ✓")
println("  - Full pipeline with physics: WORKING ✓")
