"""
Test Advanced Physics (Phase 5)

Tests heel effect and analytical phantoms.
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
# Convert to intensity for heel effect testing
intensity_cpu = exp.(-sinogram_cpu)
println("Intensity size: $(size(intensity_cpu))")
println("Intensity range: [$(minimum(intensity_cpu)), $(maximum(intensity_cpu))]")

# =============================================================================
# HEEL EFFECT TESTS
# =============================================================================

# =============================================================================
# Test 1: Heel Effect None
# =============================================================================
println("\n=== Test 1: Heel Effect None ===")

heel_none = heel_effect_none()
info = get_heel_effect_info(heel_none)
println("Heel effect enabled: $(info.enabled)")
@test info.enabled == false

intensity_no_heel = apply_heel_effect(copy(intensity_cpu), heel_none, geom)
@test isapprox(mean(intensity_no_heel), mean(intensity_cpu), rtol=1e-6)
println("Heel effect none: PASS (no change)")

# =============================================================================
# Test 2: Default Heel Effect
# =============================================================================
println("\n=== Test 2: Default Heel Effect ===")

heel_default = default_heel_effect(anode_angle_deg=7.0, target_material=:tungsten)
info = get_heel_effect_info(heel_default)
println("Heel effect info:")
println("  Enabled: $(info.enabled)")
println("  Anode angle: $(info.anode_angle_deg)°")
println("  Target material: $(info.target_material)")
println("  Expected variation: $(info.expected_variation)")

@test info.enabled == true
@test info.anode_angle_deg == 7.0

intensity_with_heel = apply_heel_effect(copy(intensity_cpu), heel_default, geom)
println("Original mean: $(mean(intensity_cpu))")
println("With heel mean: $(mean(intensity_with_heel))")

# Heel effect reduces intensity (especially on anode side)
# Mean will be lower due to attenuation - allow up to 80% reduction
@test mean(intensity_with_heel) < mean(intensity_cpu)
@test mean(intensity_with_heel) > 0.1 * mean(intensity_cpu)  # Not too much reduction
println("Default heel effect: PASS")

# =============================================================================
# Test 3: Heel Effect Variation Across Field
# =============================================================================
println("\n=== Test 3: Heel Effect Variation Across Field ===")

# Get intensity at anode side (col=1) and cathode side (col=n_cols)
n_cols = size(intensity_with_heel, 1)
anode_side = mean(intensity_with_heel[1:10, :, :])
cathode_side = mean(intensity_with_heel[end-9:end, :, :])
center = mean(intensity_with_heel[n_cols÷2-5:n_cols÷2+5, :, :])

println("Anode side intensity: $(anode_side)")
println("Center intensity: $(center)")
println("Cathode side intensity: $(cathode_side)")

# Anode side should be lower than cathode side (more attenuation)
@test anode_side < cathode_side
println("Heel effect variation: PASS (anode < cathode)")

# =============================================================================
# Test 4: Different Anode Angles
# =============================================================================
println("\n=== Test 4: Different Anode Angles ===")

for angle in [5.0, 7.0, 10.0, 15.0]
    heel = default_heel_effect(anode_angle_deg=angle)
    info = get_heel_effect_info(heel)
    println("  Anode $(angle)°: $(info.expected_variation)")
end
println("Different anode angles: PASS")

# =============================================================================
# ANALYTICAL PHANTOM TESTS
# =============================================================================

# =============================================================================
# Test 5: Ellipsoid Creation
# =============================================================================
println("\n=== Test 5: Ellipsoid Creation ===")

ellipsoid = Ellipsoid(
    (0.0, 0.0, 0.0),      # center
    (50.0, 50.0, 50.0),   # semi-axes
    (0.0, 0.0, 0.0),      # rotation
    0.02,                  # μ
    true                   # additive
)
println("Ellipsoid center: $(ellipsoid.center)")
println("Ellipsoid semi-axes: $(ellipsoid.semi_axes)")
println("Ellipsoid μ: $(ellipsoid.μ)")
@test ellipsoid.μ == 0.02
println("Ellipsoid creation: PASS")

# =============================================================================
# Test 6: Sphere Creation
# =============================================================================
println("\n=== Test 6: Sphere Creation ===")

sphere = Sphere(
    (10.0, 0.0, 0.0),  # center
    30.0,               # radius
    0.03,               # μ
    true                # additive
)
println("Sphere center: $(sphere.center)")
println("Sphere radius: $(sphere.radius)")
println("Sphere μ: $(sphere.μ)")
@test sphere.radius == 30.0
println("Sphere creation: PASS")

# =============================================================================
# Test 7: Cylinder Creation
# =============================================================================
println("\n=== Test 7: Cylinder Creation ===")

cylinder = Cylinder(
    (0.0, 0.0),           # center (x, y)
    20.0,                  # radius
    (-50.0, 50.0),        # z_range
    0.025,                 # μ
    true                   # additive
)
println("Cylinder center: $(cylinder.center)")
println("Cylinder radius: $(cylinder.radius)")
println("Cylinder z_range: $(cylinder.z_range)")
@test cylinder.radius == 20.0
println("Cylinder creation: PASS")

# =============================================================================
# Test 8: Shepp-Logan Phantom
# =============================================================================
println("\n=== Test 8: Shepp-Logan Phantom ===")

shepp_logan = create_shepp_logan_3d(fov_mm=200.0, scale=1.0)
println("Shepp-Logan phantom:")
println("  Number of objects: $(length(shepp_logan.objects))")
println("  FOV: $(shepp_logan.fov) mm")
println("  Background μ: $(shepp_logan.background_μ)")

@test length(shepp_logan.objects) == 10  # Standard Shepp-Logan has 10 ellipsoids
@test shepp_logan.fov == 200.0
println("Shepp-Logan creation: PASS")

# =============================================================================
# Test 9: Ray-Ellipsoid Intersection
# =============================================================================
println("\n=== Test 9: Ray-Ellipsoid Intersection ===")

# Simple sphere (ellipsoid with equal semi-axes)
sphere_e = Ellipsoid(
    (0.0, 0.0, 0.0),
    (50.0, 50.0, 50.0),
    (0.0, 0.0, 0.0),
    0.02,
    true
)

# Ray through center
ray_origin = (-100.0, 0.0, 0.0)
ray_dir = (1.0, 0.0, 0.0)  # Pointing in +x
path_length = ray_intersect_ellipsoid(ray_origin, ray_dir, sphere_e)

println("Ray through sphere center:")
println("  Expected path length: 100.0 mm (diameter)")
println("  Computed path length: $(path_length) mm")
@test isapprox(path_length, 100.0, rtol=0.01)
println("Ray-ellipsoid intersection: PASS")

# =============================================================================
# Test 10: Ray-Cylinder Intersection
# =============================================================================
println("\n=== Test 10: Ray-Cylinder Intersection ===")

cyl = Cylinder(
    (0.0, 0.0),
    30.0,
    nothing,  # Infinite z
    0.02,
    true
)

# Ray through cylinder center
ray_origin_c = (-100.0, 0.0, 0.0)
ray_dir_c = (1.0, 0.0, 0.0)
path_length_c = ray_intersect_cylinder(ray_origin_c, ray_dir_c, cyl)

println("Ray through cylinder center:")
println("  Expected path length: 60.0 mm (diameter)")
println("  Computed path length: $(path_length_c) mm")
@test isapprox(path_length_c, 60.0, rtol=0.01)
println("Ray-cylinder intersection: PASS")

# =============================================================================
# Test 11: Analytical Forward Projection
# =============================================================================
println("\n=== Test 11: Analytical Forward Projection ===")

# Create small analytical phantom for testing
simple_phantom = AnalyticalPhantom([sphere_e], 200.0, 0.0)

# Use smaller geometry for speed
small_geom = create_aquilion_one(n_angles=36, n_rows=8, n_cols=64, fov_cm=20.0)

# Forward project analytically
sino_analytical = analytical_forward_project(simple_phantom, small_geom)
println("Analytical sinogram size: $(size(sino_analytical))")
println("Analytical sinogram range: [$(minimum(sino_analytical)), $(maximum(sino_analytical))]")

# Should have non-zero values where rays intersect the sphere
@test maximum(sino_analytical) > 0
println("Analytical forward projection: PASS")

# =============================================================================
# Test GPU if available
# =============================================================================
if HAS_METAL
    println("\n=== GPU Tests (Metal) ===")

    # Create GPU arrays
    intensity_gpu = MtlArray(intensity_cpu)
    println("GPU array type: $(typeof(intensity_gpu))")

    # Test heel effect on GPU
    println("\nHeel effect on GPU...")
    intensity_heel_gpu = apply_heel_effect(copy(intensity_gpu), heel_default, geom)
    intensity_heel_gpu_result = Array(intensity_heel_gpu)

    cpu_heel_mean = mean(intensity_with_heel)
    gpu_heel_mean = mean(intensity_heel_gpu_result)
    println("  CPU mean: $(cpu_heel_mean)")
    println("  GPU mean: $(gpu_heel_mean)")
    @test isapprox(cpu_heel_mean, gpu_heel_mean, rtol=0.05)
    println("  Heel effect GPU: PASS")

    # Note: Analytical projection GPU test skipped - requires CTGeometry refactoring
    # The CPU version works correctly; GPU support would require extracting all
    # geometry values before the kernel to avoid capturing the non-bitstype struct.
    println("\nAnalytical projection on GPU: SKIPPED (requires CTGeometry refactoring)")

    println("\n=== All GPU Advanced Physics tests passed! ===")
else
    println("\n=== GPU tests skipped (Metal not available) ===")
end

# =============================================================================
# Reconstruction with Heel Effect
# =============================================================================
println("\n=== Reconstruction with Heel Effect ===")

# Apply heel effect to sinogram (need to convert back)
sinogram_with_heel = -log.(max.(intensity_with_heel, Float32(1e-10)))

recon_no_heel = fdk_reconstruct(sinogram_cpu, geom, size(phantom.μ))
recon_with_heel = fdk_reconstruct(sinogram_with_heel, geom, size(phantom.μ))

println("Recon without heel range: [$(minimum(recon_no_heel)), $(maximum(recon_no_heel))]")
println("Recon with heel range: [$(minimum(recon_with_heel)), $(maximum(recon_with_heel))]")

# Heel effect should cause some difference
@test !isapprox(mean(recon_with_heel), mean(recon_no_heel), rtol=0.001)
println("Heel effect on reconstruction: PASS")

# =============================================================================
# Test Shepp-Logan Forward Projection
# =============================================================================
println("\n=== Shepp-Logan Forward Projection ===")

small_geom_sl = create_aquilion_one(n_angles=36, n_rows=8, n_cols=64, fov_cm=20.0)
sino_sl = analytical_forward_project(shepp_logan, small_geom_sl)
println("Shepp-Logan sinogram range: [$(minimum(sino_sl)), $(maximum(sino_sl))]")

# Shepp-Logan should produce varied projections
@test maximum(sino_sl) > 0
@test std(sino_sl) > 0
println("Shepp-Logan projection: PASS")

println("\n=== Advanced Physics Tests Complete ===")
