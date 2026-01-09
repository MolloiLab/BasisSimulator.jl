"""
Debug Geometry - Check phantom vs scanner alignment
"""

using BasisSimulator

println("="^70)
println("Geometry Debug: Phantom vs Scanner Alignment")
println("="^70)

# Create phantom and scanner
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=20.0)
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=2)
geometry = create_aquilion_one(protocol=protocol)

println("\n1. Phantom Extents:")
println("   X: $(minimum(phantom.grid.x)) to $(maximum(phantom.grid.x)) cm")
println("   Y: $(minimum(phantom.grid.y)) to $(maximum(phantom.grid.y)) cm")
println("   Z: $(minimum(phantom.grid.z)) to $(maximum(phantom.grid.z)) cm")
println("   FOV XY: $(phantom.grid.fov_xy_cm) cm")
println("   FOV Z: $(phantom.grid.fov_z_cm) cm")

println("\n2. Scanner Geometry:")
println("   SAD: $(geometry.SAD_cm) cm")
println("   SDD: $(geometry.SDD_cm) cm")
println("   Detector rows: $(geometry.n_rows)")
println("   Detector cols: $(geometry.n_cols)")
println("   Pixel width: $(geometry.pixel_width_cm) cm")
println("   Pixel height: $(geometry.pixel_height_cm) cm")

println("\n3. Source position (angle 1):")
src = geometry.source_positions[:, 1]
println("   ($(round(src[1], digits=2)), $(round(src[2], digits=2)), $(round(src[3], digits=2))) cm")

println("\n4. Detector center (angle 1):")
det = geometry.det_centers[:, 1]
println("   ($(round(det[1], digits=2)), $(round(det[2], digits=2)), $(round(det[3], digits=2))) cm")

println("\n5. Insert Positions (expected):")
println("   Calcium ring: 5.0 cm radius")
println("   Iodine ring: 10.5 cm radius")
println("   Body radius: 16.5 cm")

# Check if insert radii are within detector field of view at isocenter
det_width_at_iso = (geometry.n_cols * geometry.pixel_width_cm) / (geometry.SDD_cm / geometry.SAD_cm)
println("\n6. Detector FOV at isocenter:")
println("   Width: $(round(det_width_at_iso, digits=2)) cm")
println("   Height: $(round((geometry.n_rows * geometry.pixel_height_cm) / (geometry.SDD_cm / geometry.SAD_cm), digits=2)) cm")

# Check Z alignment
println("\n7. Z-axis alignment:")
println("   Phantom Z range: $(minimum(phantom.grid.z)) to $(maximum(phantom.grid.z)) cm")
println("   Scanner Z coverage: $(round(geometry.n_rows * geometry.pixel_height_cm / (geometry.SDD_cm / geometry.SAD_cm), digits=2)) cm")
println("   Central detector row should hit Z=0")

# Sample a ray through center
println("\n8. Testing ray through center...")
grid_meta = GridMeta(
    nx=phantom.grid.nx, ny=phantom.grid.ny, nz=phantom.grid.nz,
    fov_xy=phantom.grid.fov_xy_cm, fov_z=phantom.grid.fov_z_cm
)

# Ray from source to detector center (should pass through isocenter)
println("   Ray: Source$(round.(src, digits=2)) → Detector$(round.(det, digits=2))")
println("   This ray should pass through (0, 0, 0) - the phantom center")

# Check if it passes through phantom bounds
ray_passes_through_origin = (src[1] * det[1] < 0) || (src[2] * det[2] < 0)
println("   Does ray cross origin XY plane? $(ray_passes_through_origin)")

println("\n" * "="^70)
