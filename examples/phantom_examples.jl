"""
# Phantom Loading and Material System - Usage Examples

This file demonstrates the phantom loading and material system capabilities.
"""

# =============================================================================
# Example 1: Load P2 Phantom with Semantic Classification
# =============================================================================

"""
    example_p2_phantom_with_classification()

Demonstrates loading a P2 XCAT phantom with automatic semantic classification.

# Usage:
```julia
using BasisSimulator
phantom = example_p2_phantom_with_classification()
```
"""
function example_p2_phantom_with_classification()
    using BasisSimulator.SemanticClassification
    
    # Example: Create a simulated P2 phantom structure
    # In practice, you would load from actual P2_*.raw files
    
    # Define dimensions for P2 phantom
    dims = (400, 400, 200)  # Example P2 dimensions
    
    # Create a labeled array (simulating P2 structure IDs)
    # P2 has IDs: 1-1102 with different structures
    labeled = zeros(UInt16, dims)
    
    # Simulate some structures by ID ranges
    # IDs 0: background/air
    # IDs 1-22: basic tissues (P1 style)
    # IDs 468-702: veins
    # IDs 703-1102: arteries
    # IDs 252-329: gray matter
    # IDs 330-454: white matter
    
    # Set up material mapping
    materials_dict = Dict{Int, XA.Material}()
    
    # Use semantic classification to get materials
    for id in 0:1102
        category = classify_structure("structure_$id", id)
        mat_symbol = get_category_material(category)
        materials_dict[id] = get_material(mat_symbol)
    end
    
    # Create phantom with 2mm voxels (typical for P2)
    voxel_size = (0.2, 0.2, 0.2)  # 2mm = 0.2cm
    
    phantom = Phantom(labeled, materials_dict, voxel_size)
    
    println("P2 Phantom created:")
    println("  Dimensions: ", size(phantom.mask))
    println("  FOV: ", phantom.fov)
    println("  Materials: ", length(phantom.materials))
    
    return phantom
end

# =============================================================================
# Example 2: Apply Dynamic Iodine Contrast at Different Time Points
# =============================================================================

"""
    example_iodine_contrast_perfusion()

Demonstrates applying dynamic iodine contrast for perfusion simulation.

# Usage:
```julia
using BasisSimulator
result = example_iodine_contrast_perfusion()
```
"""
function example_iodine_contrast_perfusion()
    import XrayAttenuation as XA
    
    # Base blood material
    blood = get_material(:blood)
    
    # Time points in seconds (simulating contrast arrival)
    timepoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    
    # Simulated iodine concentrations (mg/g) at each time point
    # Typical perfusion: peaks around 30-40 seconds
    concentrations = [0.0, 1.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
    
    println("Iodine Contrast Perfusion:")
    println("  Time (s)\tConc (mg/g)\tμ at 60keV")
    println("  " * "-"^40)
    
    for (t, c) in zip(timepoints, concentrations)
        if c > 0
            contrast_blood = create_iodine_blood_mixture(blood, c)
            μ = calculate_mixture_attenuation(contrast_blood, 60.0)
            println("  $t\t\t$c\t\t", round(μ, digits=4))
        else
            μ = calculate_mixture_attenuation(blood, 60.0)
            println("  $t\t\t0\t\t", round(μ, digits=4))
        end
    end
    
    return (timepoints=timepoints, concentrations=concentrations)
end

# =============================================================================
# Example 3: Update Material Composition for Specific Regions
# =============================================================================

"""
    example_update_region_materials()

Demonstrates updating material composition for specific anatomical regions.

# Usage:
```julia
using BasisSimulator
phantom = example_update_region_materials()
```
"""
function example_update_region_materials()
    import XrayAttenuation as XA
    
    # Create a simple 3-region phantom
    dims = (100, 100, 50)
    labeled = zeros(UInt8, dims)
    
    # Region 0: Air (background)
    # Region 1: Soft tissue
    # Region 2: Bone
    
    labeled[1:50, :, :] .= 1  # Left half: soft tissue
    labeled[51:100, :, :] .= 2  # Right half: bone
    
    materials_dict = Dict{Int, XA.Material}(
        0 => XA.Materials.air,
        1 => XA.Materials.water,  # Soft tissue ~ water
        2 => get_material(:bone)
    )
    
    phantom = Phantom(labeled, materials_dict, (0.1, 0.1, 0.1))
    
    # Now demonstrate updating a region with contrast
    println("Original materials:")
    μ = compute_μ(phantom, 60.0)
    println("  Soft tissue μ: ", round(mean(μ[labeled .== 1]), digits=4))
    println("  Bone μ: ", round(mean(μ[labeled .== 2]), digits=4))
    
    # Update soft tissue region with iodine contrast
    # (In practice, you'd use a mask to identify the region)
    contrast_tissue = create_iodine_blood_mixture(
        XA.Materials.water,
        5.0  # 5 mg/g iodine
    )
    
    # Create new materials dict with contrast
    updated_materials = Dict{Int, XA.Material}(
        0 => XA.Materials.air,
        1 => contrast_tissue,  # Contrast-enhanced tissue
        2 => get_material(:bone)
    )
    
    contrast_phantom = Phantom(labeled, updated_materials, (0.1, 0.1, 0.1))
    
    println("\nWith iodine contrast (5 mg/g):")
    μ_contrast = compute_μ(contrast_phantom, 60.0)
    println("  Enhanced tissue μ: ", round(mean(μ_contrast[labeled .== 1]), digits=4))
    println("  Bone μ: ", round(mean(μ_contrast[labeled .== 2]), digits=4))
    
    return contrast_phantom
end

# =============================================================================
# Example 4: Load PVAT Phantom (Backward Compatibility)
# =============================================================================

"""
    example_pvat_backward_compatibility()

Demonstrates backward compatibility with existing PVAT workflow.

# Usage:
```julia
using BasisSimulator
phantom = example_pvat_backward_compatibility()
```
"""
function example_pvat_backward_compatibility()
    import XrayAttenuation as XA
    
    # PVAT dimensions: 1600 x 1400 x 500
    # Typical PVAT has 33 organ IDs (0-32)
    
    # Simulate loading a PVAT-style phantom
    dims = (1600, 1400, 500)
    
    # In practice, you'd load from actual file:
    # labeled = load_raw_volume("path/to/PVAT.raw"; cols=1600, rows=1400, slices=500)
    
    labeled = rand(0:32, dims)
    
    # Define materials for each organ ID (typical PVAT mapping)
    # This mimics the original PVAT_main_script_Shu.jl workflow
    materials_dict = Dict{Int, XA.Material}()
    
    # Default to air
    for i in 0:32
        materials_dict[i] = XA.Materials.air
    end
    
    # Common PVAT organ mappings
    materials_dict[1] = XA.Materials.water  # Soft tissue
    materials_dict[2] = get_material(:bone)
    materials_dict[3] = get_material(:muscle)
    materials_dict[4] = get_material(:brain)
    materials_dict[5] = get_material(:blood)
    
    # Create phantom with 1mm voxels
    voxel_size = (0.1, 0.1, 0.1)  # 1mm = 0.1cm
    
    phantom = Phantom(labeled, materials_dict, voxel_size)
    
    println("PVAT-style Phantom (Backward Compatible):")
    println("  Dimensions: ", size(phantom.mask))
    println("  FOV: ", phantom.fov)
    println("  Voxel size: ", phantom.voxel_size)
    println("  Materials: ", length(phantom.materials))
    
    # Compute attenuation
    μ = compute_μ(phantom, 60.0)
    println("  μ range: ", round(minimum(μ), digits=4), " to ", round(maximum(μ), digits=4))
    
    return phantom
end

# =============================================================================
# Run Examples
# =============================================================================

if false  # Set to true to run examples
    println("="^60)
    println("Example 1: P2 Phantom with Semantic Classification")
    println("="^60)
    example_p2_phantom_with_classification()
    
    println("\n" * "="^60)
    println("Example 2: Iodine Contrast Perfusion")
    println("="^60)
    example_iodine_contrast_perfusion()
    
    println("\n" * "="^60)
    println("Example 3: Update Region Materials")
    println("="^60)
    example_update_region_materials()
    
    println("\n" * "="^60)
    println("Example 4: PVAT Backward Compatibility")
    println("="^60)
    example_pvat_backward_compatibility()
end
