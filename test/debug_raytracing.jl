"""
Debug Ray Tracing - Check if rays intersect with Gammex inserts
"""

using BasisSimulator
import XrayAttenuation as XA
using Statistics

println("="^70)
println("Ray Tracing Debug: Testing Insert Intersection")
println("="^70)

# Create phantom
println("\n1. Creating Gammex 472 phantom...")
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=160.0)  # Match scanner!

# Count materials
counts = count_materials(phantom)
println("\n2. Material distribution:")
for (mat, cnt) in sort(collect(counts), by=x->string(x[1]))
    println("  $mat: $cnt voxels")
end

# Scanner setup
println("\n3. Setting up scanner...")
protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=400.0, num_projections=2)  # Minimum 2 projections
geometry = create_aquilion_one(protocol=protocol)

# GridMeta for ray tracing
grid_meta = GridMeta(
    nx=phantom.grid.nx, ny=phantom.grid.ny, nz=phantom.grid.nz,
    fov_xy=phantom.grid.fov_xy_cm, fov_z=phantom.grid.fov_z_cm
)

println("\n4. Building material lookup...")

# Build ID to material mapping (same as simulation)
unique_mat_ids = sort(unique(phantom.material_ids))
n_materials = length(unique_mat_ids)

id_to_idx = Dict{UInt8, Int}()
idx_to_id = Dict{Int, UInt8}()
for (idx, mat_id) in enumerate(unique_mat_ids)
    id_to_idx[mat_id] = idx
    idx_to_id[idx] = mat_id
end

# Create lookup table
id_lut = zeros(Int, 256)
for (mat_id, idx) in id_to_idx
    id_lut[Int(mat_id) + 1] = idx
end

println("   Number of materials: $n_materials")

println("\n5. Testing ray paths through phantom center...")

# Test a single ray that should pass through an insert
angle_idx = 1
src_pos = geometry.source_positions[:, angle_idx]
det_center = geometry.det_centers[:, angle_idx]
u_vec = geometry.det_u_vecs[:, angle_idx]
v_vec = geometry.det_v_vecs[:, angle_idx]

# Try multiple detector positions
println("\n6. Sampling rays across detector:")
println("   (Looking for rays that hit calcium or iodine inserts)")

n_hits_per_material = Dict{Symbol, Int}()
global n_samples = 0

for row in [1, geometry.n_rows÷4, geometry.n_rows÷2, 3*geometry.n_rows÷4, geometry.n_rows]
    for col in [1, geometry.n_cols÷4, geometry.n_cols÷2, 3*geometry.n_cols÷4, geometry.n_cols]
        global n_samples

        # Detector pixel position
        u_offset = (col - geometry.n_cols/2 - 0.5) * geometry.pixel_width_cm
        v_offset = (row - geometry.n_rows/2 - 0.5) * geometry.pixel_height_cm

        detector_pos = det_center .+ (u_offset .* u_vec) .+ (v_offset .* v_vec)

        # Trace ray - returns path lengths per material
        path_lengths = trace_ray_material_paths(
            grid_meta,
            phantom.material_ids,
            phantom.densities,
            id_lut,
            n_materials,
            src_pos[1], src_pos[2], src_pos[3],
            detector_pos[1], detector_pos[2], detector_pos[3]
        )

        n_samples += 1

        # Check which materials were hit (path_length > 0)
        materials_hit = Set{Symbol}()
        for m_idx in 1:n_materials
            if path_lengths[m_idx] > 0
                mat_id = idx_to_id[m_idx]
                mat_symbol = phantom.id_to_material[mat_id]
                push!(materials_hit, mat_symbol)

                # Count hits
                n_hits_per_material[mat_symbol] = get(n_hits_per_material, mat_symbol, 0) + 1
            end
        end

        # Print interesting rays (those that hit calcium or iodine)
        has_insert = any(m -> startswith(string(m), "Ca_") || startswith(string(m), "I_"), materials_hit)
        if has_insert
            # Also print path lengths for debugging
            insert_materials = filter(m -> startswith(string(m), "Ca_") || startswith(string(m), "I_"), materials_hit)
            println("   Ray [row=$row, col=$col]: $(join(insert_materials, ", "))")
        end
    end
end

println("\n7. Material Hit Statistics (out of $n_samples rays):")
for (mat, hits) in sort(collect(n_hits_per_material), by=x->x[2], rev=true)
    pct = round(100 * hits / n_samples, digits=1)
    println("  $mat: $hits hits ($pct%)")
end

# Check if ANY rays hit calcium or iodine
has_ca = any(m -> startswith(string(m), "Ca_"), keys(n_hits_per_material))
has_i = any(m -> startswith(string(m), "I_"), keys(n_hits_per_material))

println("\n8. Result:")
if has_ca && has_i
    println("  ✅ SUCCESS: Rays are hitting both calcium and iodine inserts!")
    println("     Ray tracer is working correctly.")
else
    println("  ❌ PROBLEM: Rays are NOT hitting insert materials")
    println("     Calcium hit: $has_ca")
    println("     Iodine hit: $has_i")
    println("     This explains why sinogram has no contrast!")
end

println("\n" * "="^70)
