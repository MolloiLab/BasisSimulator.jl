"""
Debug Direct Ray - Trace a ray directly through a known insert position
"""

using BasisSimulator

println("="^70)
println("Direct Ray Test: Tracing through known insert")
println("="^70)

# Create phantom
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=160.0)

# Build material lookup (same as simulation)
unique_mat_ids = sort(unique(phantom.material_ids))
n_materials = length(unique_mat_ids)

id_to_idx = Dict{UInt8, Int}()
idx_to_id = Dict{Int, UInt8}()
for (idx, mat_id) in enumerate(unique_mat_ids)
    id_to_idx[mat_id] = idx
    idx_to_id[idx] = mat_id
end

id_lut = zeros(Int, 256)
for (mat_id, idx) in id_to_idx
    id_lut[Int(mat_id) + 1] = idx
end

# GridMeta
grid_meta = GridMeta(
    nx=phantom.grid.nx, ny=phantom.grid.ny, nz=phantom.grid.nz,
    fov_xy=phantom.grid.fov_xy_cm, fov_z=phantom.grid.fov_z_cm
)

println("\n1. Finding insert positions...")
ca_indices = findall(x -> phantom.id_to_material[x] == :Ca_50, phantom.material_ids)
ca_pos = (phantom.grid.x[ca_indices[1][1]], phantom.grid.y[ca_indices[1][2]], 0.0)
println("   Ca_50 at: $(round.(ca_pos, digits=2)) cm")

println("\n2. Creating ray that passes through Ca_50 insert...")
# Ray from source at (-60, 0, 0) through insert to detector
source_pos = (-60.0, 0.0, 0.0)
target_pos = (60.0, 0.0, 0.0)  # Opposite side, ray passes through X-axis

println("   Source: $source_pos cm")
println("   Target: $target_pos cm")
println("   Ray should pass through X-axis where inserts are")

println("\n3. Tracing ray...")
path_lengths = trace_ray_material_paths(
    grid_meta,
    phantom.material_ids,
    phantom.densities,
    id_lut,
    n_materials,
    source_pos[1], source_pos[2], source_pos[3],
    target_pos[1], target_pos[2], target_pos[3]
)

println("\n4. Ray tracing results:")
total_path = sum(path_lengths)
println("   Total path length: $(round(total_path, digits=2)) cm")
println("   Expected ~120 cm (source to detector distance)")

println("\n5. Materials encountered:")
found_inserts = false
for m_idx in 1:n_materials
    if path_lengths[m_idx] > 0.01  # Significant path length
        mat_id = idx_to_id[m_idx]
        mat_symbol = phantom.id_to_material[mat_id]
        println("   $mat_symbol: $(round(path_lengths[m_idx], digits=2)) cm")

        if startswith(string(mat_symbol), "Ca_") || startswith(string(mat_symbol), "I_")
            found_inserts = true
        end
    end
end

println("\n6. Result:")
if found_inserts
    println("   ✅ SUCCESS: Ray tracer detected insert materials!")
    println("      Ray tracing is working correctly.")
else
    println("   ✗ PROBLEM: Ray tracer did NOT detect inserts")
    println("      Even though inserts exist in phantom at correct positions.")
    println("      Issue must be in trace_ray_material_paths function.")
end

println("\n" * "="^70)
