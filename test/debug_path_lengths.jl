"""
Debug script to check ray tracer path lengths
"""

using BasisSimulator
import XrayAttenuation as XA

println("="^70)
println("RAY TRACER PATH LENGTH CHECK")
println("="^70)

# Create simple water phantom
phantom = create_water_cylinder(
    diameter_mm=330.0,  # Same as Gammex body
    height_mm=160.0,
    resolution_mm=4.0
)

# Setup simple geometry (use 2 projections, test first one)
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=2)
geometry = create_aquilion_one(protocol=protocol)

# Grid metadata
grid_meta = GridMeta(
    nx = phantom.grid.nx,
    ny = phantom.grid.ny,
    nz = phantom.grid.nz,
    fov_xy = phantom.grid.fov_xy_cm,
    fov_z = phantom.grid.fov_z_cm
)

# Material lookup
unique_mat_ids = sort(unique(phantom.material_ids))
n_materials = length(unique_mat_ids)
id_lut = zeros(Int, 256)
for (idx, mat_id) in enumerate(unique_mat_ids)
    id_lut[Int(mat_id) + 1] = idx
end

# Test ray through center of phantom (should go through ~33 cm of water)
src_pos = geometry.source_positions[:, 1]  # First angle (0°)
det_center = geometry.det_centers[:, 1]

println("Source position: $src_pos")
println("Detector center: $det_center")
println()

# Trace ray
path_lengths = trace_ray_material_paths(
    grid_meta,
    phantom.material_ids,
    phantom.densities,
    id_lut,
    n_materials,
    src_pos[1], src_pos[2], src_pos[3],
    det_center[1], det_center[2], det_center[3]
)

println("Path lengths returned by ray tracer:")
for (idx, path_len) in enumerate(path_lengths)
    if path_len > 0
        mat_id = unique_mat_ids[idx]
        mat_symbol = phantom.id_to_material[mat_id]
        println("  Material $idx ($mat_symbol): $path_len")

        # Check units
        if mat_symbol == :water
            println("    Expected geometric path: ~33 cm")
            println("    If this is radiological path (ρ×L): ~33 g/cm² (ρ≈1 for water)")
            println("    Actual value: $path_len")
            println("    Factor off: $(path_len / 33.0)x")
        end
    end
end

println()
println("Total path (sum): $(sum(path_lengths))")
println("Ray length (source to detector): $(sqrt(sum((det_center .- src_pos).^2)))")
println("="^70)
