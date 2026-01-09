"""
Debug Insert Positions - Verify inserts are where we expect them
"""

using BasisSimulator

println("="^70)
println("Insert Position Debug")
println("="^70)

# Create phantom
phantom = create_gammex_472(resolution_mm=4.0, z_coverage_mm=160.0)

println("\n1. Looking for calcium insert (Ca_50)...")
ca_indices = findall(x -> phantom.id_to_material[x] == :Ca_50, phantom.material_ids)
if !isempty(ca_indices)
    idx = ca_indices[1]
    x_pos = phantom.grid.x[idx[1]]
    y_pos = phantom.grid.y[idx[2]]
    z_pos = phantom.grid.z[idx[3]]
    radius = sqrt(x_pos^2 + y_pos^2)
    println("  ✓ Ca_50 insert found!")
    println("    Voxel index: $idx")
    println("    Position: ($(round(x_pos, digits=2)), $(round(y_pos, digits=2)), $(round(z_pos, digits=2))) cm")
    println("    Radius from center: $(round(radius, digits=2)) cm")
    println("    Expected radius: 5.0 cm")
    println("    Match: $(abs(radius - 5.0) < 1.0 ? "✓" : "✗ MISMATCH!")")
else
    println("  ✗ ERROR: No Ca_50 voxels found in phantom!")
end

println("\n2. Looking for iodine insert (I_2_0)...")
i_indices = findall(x -> phantom.id_to_material[x] == :I_2_0, phantom.material_ids)
if !isempty(i_indices)
    idx = i_indices[1]
    x_pos = phantom.grid.x[idx[1]]
    y_pos = phantom.grid.y[idx[2]]
    z_pos = phantom.grid.z[idx[3]]
    radius = sqrt(x_pos^2 + y_pos^2)
    println("  ✓ I_2_0 insert found!")
    println("    Voxel index: $idx")
    println("    Position: ($(round(x_pos, digits=2)), $(round(y_pos, digits=2)), $(round(z_pos, digits=2))) cm")
    println("    Radius from center: $(round(radius, digits=2)) cm")
    println("    Expected radius: 10.5 cm")
    println("    Match: $(abs(radius - 10.5) < 1.0 ? "✓" : "✗ MISMATCH!")")
else
    println("  ✗ ERROR: No I_2_0 voxels found in phantom!")
end

println("\n3. Checking water body...")
water_indices = findall(x -> phantom.id_to_material[x] == :water, phantom.material_ids)
if !isempty(water_indices)
    # Sample a water voxel near center
    center_idx = findfirst(x -> phantom.id_to_material[x] == :water &&
                                abs(phantom.grid.x[x[1]]) < 2.0 &&
                                abs(phantom.grid.y[x[2]]) < 2.0,
                          phantom.material_ids)
    if !isnothing(center_idx)
        x_pos = phantom.grid.x[center_idx[1]]
        y_pos = phantom.grid.y[center_idx[2]]
        println("  ✓ Water body found near center")
        println("    Sample position: ($(round(x_pos, digits=2)), $(round(y_pos, digits=2))) cm")
    end
end

println("\n" * "="^70)
