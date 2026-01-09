"""
    Geometry/Phantoms.jl

Digital phantom generation for CT simulation validation and testing.

# Overview

This module provides functions for creating computational phantoms used in CT
simulation. Phantoms are digital representations of physical objects with known
material compositions and geometries.

**Key Applications:**
- **Validation**: Compare simulator output against known ground truth
- **Calibration**: Measure scanner performance (HU accuracy, MTF, noise)
- **Algorithm Testing**: Evaluate reconstruction and correction algorithms
- **Training**: Generate synthetic datasets for machine learning

# Phantom Types

## 1. Calibration Phantoms
**Gammex 472** - Industry-standard CT calibration phantom:
- 33 cm diameter water-equivalent body
- 7 calcium inserts (50-600 mg/ml) at 5 cm radius
- 7 iodine inserts (2-20 mg/ml) at 10.5 cm radius
- Used for HU calibration and material decomposition validation

## 2. Anatomical Phantoms (Future)
**XCAT** - 4D cardiac and respiratory motion phantom:
- Realistic organ geometries
- Time-varying cardiac and respiratory motion
- Patient-specific scaling

## 3. Geometric Phantoms (Future)
**Resolution phantoms** - MTF and spatial resolution testing
**Contrast phantoms** - Low-contrast detectability

# Data Structures

## VoxelGrid
Defines the spatial coordinate system and voxel spacing.
All units in centimeters (cm).

## PhantomData
Contains material IDs and density variations:
- `material_ids::Array{UInt8, 3}` - Material index per voxel (0-255)
- `densities::Array{Float32, 3}` - Relative density per voxel
- `id_to_material::Dict{UInt8, Symbol}` - Mapping to material names

Memory-optimized: 5 bytes/voxel vs 16 bytes for Symbol + Float64

# Material Integration

Uses `XrayAttenuation.jl` for material properties:
- Pre-defined materials: `XA.Materials.water`, `XA.Materials.corticalbone`, etc.
- Custom mixtures: Create iodine and calcium solutions dynamically

# References

**Phantoms:**
- Gammex Inc. (2018). "Model 472 Phantom Specifications"
- Segars, W. P., et al. (2010). "4D XCAT phantom." Medical physics, 37(9)

**Calibration:**
- Schneider, U., et al. (2000). "Correlation between CT numbers and tissue."
  Physics in Medicine & Biology, 45(2), 459.

# Author

Dale Black, MolloiLab
Created: January 2026
"""

import XrayAttenuation as XA
using Random: randn!

# ==============================================================================
# Voxel Grid Definition
# ==============================================================================

"""
    VoxelGrid

Defines the spatial coordinate system for phantom generation.

# Fields

- `nx, ny, nz::Int` - Number of voxels in each dimension
- `fov_xy_cm::Float64` - Field-of-view in XY plane (cm)
- `fov_z_cm::Float64` - Field-of-view in Z direction (cm)
- `x, y, z::Vector{Float64}` - Voxel center coordinates (cm)
- `x_planes, y_planes, z_planes::Vector{Float64}` - Voxel boundaries (cm)

# Constructor

```julia
grid = VoxelGrid(
    size_mm = (340.0, 340.0, 40.0),  # FOV in mm
    matrix = (680, 680, 80)           # Number of voxels
)
```

# Coordinate System

- Origin at (0, 0, 0) (isocenter)
- X-axis: Left (-) to Right (+)
- Y-axis: Posterior (-) to Anterior (+)
- Z-axis: Inferior (-) to Superior (+)

# Memory Layout

Voxel arrays use **column-major** indexing (Julia default):
- `array[i, j, k]` accesses voxel at (x[i], y[j], z[k])
- Fastest varying index: i (X)
- Slowest varying index: k (Z)
"""
struct VoxelGrid
    nx::Int
    ny::Int
    nz::Int
    fov_xy_cm::Float64
    fov_z_cm::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    x_planes::Vector{Float64}
    y_planes::Vector{Float64}
    z_planes::Vector{Float64}

    function VoxelGrid(; size_mm::Tuple{Real, Real, Real}, matrix::Tuple{Int, Int, Int})
        fov_x_mm, fov_y_mm, fov_z_mm = size_mm
        nx, ny, nz = matrix

        # Validation
        @assert all(size_mm .> 0) "FOV sizes must be positive"
        @assert all(matrix .> 0) "Matrix sizes must be positive"
        @assert fov_x_mm == fov_y_mm "FOV must be square in XY plane"

        # Convert to cm
        fov_xy_cm = fov_x_mm / 10.0
        fov_z_cm = fov_z_mm / 10.0

        # Voxel spacing
        dx = fov_xy_cm / nx
        dy = fov_xy_cm / ny
        dz = fov_z_cm / nz

        # Voxel centers (origin at 0,0,0)
        x = collect(range(-fov_xy_cm/2 + dx/2, length=nx, step=dx))
        y = collect(range(-fov_xy_cm/2 + dy/2, length=ny, step=dy))
        z = collect(range(-fov_z_cm/2 + dz/2, length=nz, step=dz))

        # Voxel boundaries (for ray tracing)
        x_planes = collect(range(-fov_xy_cm/2, length=nx+1, step=dx))
        y_planes = collect(range(-fov_xy_cm/2, length=ny+1, step=dy))
        z_planes = collect(range(-fov_z_cm/2, length=nz+1, step=dz))

        new(nx, ny, nz, fov_xy_cm, fov_z_cm, x, y, z, x_planes, y_planes, z_planes)
    end
end

"""
    get_voxel_size(grid::VoxelGrid)::Tuple{Float64, Float64, Float64}

Get voxel dimensions in cm (dx, dy, dz).
"""
function get_voxel_size(grid::VoxelGrid)::Tuple{Float64, Float64, Float64}
    dx = grid.fov_xy_cm / grid.nx
    dy = grid.fov_xy_cm / grid.ny
    dz = grid.fov_z_cm / grid.nz
    return (dx, dy, dz)
end

"""
    get_voxel_volume(grid::VoxelGrid)::Float64

Get voxel volume in cm³.
"""
function get_voxel_volume(grid::VoxelGrid)::Float64
    dx, dy, dz = get_voxel_size(grid)
    return dx * dy * dz
end

# ==============================================================================
# Phantom Data Structure
# ==============================================================================

"""
    PhantomData

Memory-optimized phantom representation with material IDs and densities.

# Fields

- `name::String` - Phantom identifier
- `grid::VoxelGrid` - Spatial coordinate system
- `material_ids::Array{UInt8, 3}` - Material index per voxel (0=air, 1-255=materials)
- `densities::Array{Float32, 3}` - Relative density per voxel (1.0 = nominal density)
- `id_to_material::Dict{UInt8, Symbol}` - Material name lookup

# Memory Optimization

Uses compact types to reduce memory footprint:
- `UInt8` material IDs (1 byte) vs `Symbol` (8+ bytes)
- `Float32` densities (4 bytes) vs `Float64` (8 bytes)
- **Savings:** 16 → 5 bytes per voxel (68% reduction)

For 512³ phantom: 2.1 GB → 0.67 GB

# Material Lookup

```julia
# Get material at voxel (i, j, k)
material_id = phantom.material_ids[i, j, k]
material_symbol = phantom.id_to_material[material_id]
material = XA.Materials[material_symbol]  # Get XA material

# Get density-adjusted attenuation
density_factor = phantom.densities[i, j, k]
μ_nominal = get_linear_attenuation(material, energy_keV)
μ_actual = μ_nominal * density_factor
```

# Texture/Noise

The `densities` array can include small random variations to simulate:
- Material heterogeneity
- Reconstruction noise
- Partial volume effects

Typical: `densities[i,j,k] = 1.0 + randn() * 0.01` (1% variation)
"""
struct PhantomData
    name::String
    grid::VoxelGrid
    material_ids::Array{UInt8, 3}
    densities::Array{Float32, 3}
    id_to_material::Dict{UInt8, Symbol}

    function PhantomData(;
            name::String = "Unknown",
            grid::VoxelGrid,
            material_ids::Array{UInt8, 3},
            densities::Array{Float32, 3},
            id_to_material::Dict{UInt8, Symbol}
        )

        # Validation
        @assert size(material_ids) == (grid.nx, grid.ny, grid.nz) "Material array size mismatch"
        @assert size(densities) == (grid.nx, grid.ny, grid.nz) "Density array size mismatch"
        @assert haskey(id_to_material, UInt8(0)) "Must include air (ID=0) in material dict"

        new(name, grid, material_ids, densities, id_to_material)
    end
end

"""
    get_memory_usage(phantom::PhantomData)::Float64

Estimate phantom memory usage in GB.
"""
function get_memory_usage(phantom::PhantomData)::Float64
    n_voxels = phantom.grid.nx * phantom.grid.ny * phantom.grid.nz
    bytes_per_voxel = sizeof(UInt8) + sizeof(Float32)  # 5 bytes
    total_bytes = n_voxels * bytes_per_voxel
    return total_bytes / 1024^3
end

# ==============================================================================
# Gammex 472 Phantom Generation
# ==============================================================================

"""
    create_gammex_472(;
        resolution_mm::Float64 = 0.5,
        z_coverage_mm::Float64 = 40.0,
        add_texture::Bool = true,
        texture_std::Float64 = 0.01
    )

Create Gammex Model 472 calibration phantom with iodine and calcium inserts.

# Phantom Specifications

**Body:**
- Material: Solid water (tissue-equivalent)
- Diameter: 330 mm
- Composition: H₂O-equivalent

**Insert Rods:**
- Diameter: 28 mm (27.9 mm rod + 0.1 mm air gap)
- Length: Full phantom height

**Inner Ring (5 cm radius):** 7 Calcium Inserts
- 50, 100, 200, 300, 400, 500, 600 mg/ml

**Outer Ring (10.5 cm radius):** 7 Iodine Inserts
- 2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0 mg/ml

# Arguments

- `resolution_mm` - Voxel size (default: 0.5 mm, clinical ~1-2 mm)
- `z_coverage_mm` - Phantom height (default: 40 mm)
- `add_texture` - Add density variations (default: true)
- `texture_std` - Std dev of density noise (default: 0.01 = 1%)

# Returns

`PhantomData` with materials mapped to XrayAttenuation.jl

# Example

```julia
# Clinical resolution
phantom = create_gammex_472(resolution_mm=1.0, z_coverage_mm=40.0)

# High resolution for validation
phantom_hires = create_gammex_472(resolution_mm=0.5, z_coverage_mm=20.0)

# Memory estimate
mem_gb = get_memory_usage(phantom)
println("Memory: \$mem_gb GB")
```

# References

- Gammex Inc. (2018). "Model 472 Tissue Characterization Phantom"
- Schneider, U., et al. (2000). "Correlation between CT numbers and tissue"
"""
function create_gammex_472(;
        resolution_mm::Float64 = 0.5,
        z_coverage_mm::Float64 = 40.0,
        add_texture::Bool = true,
        texture_std::Float64 = 0.01
    )

    # =========================================================================
    # 1. Physical Dimensions (Gammex 472 Specifications)
    # =========================================================================

    body_diameter_mm = 330.0  # 33 cm outer diameter
    rod_diameter_mm = 28.0    # 2.8 cm rods
    air_gap_mm = 0.05         # 50 micron manufacturing tolerance

    # Simulation margins
    sim_margin_mm = 10.0
    fov_total_mm = body_diameter_mm + sim_margin_mm

    # Calculate matrix size
    nx = round(Int, fov_total_mm / resolution_mm)
    ny = nx
    nz = round(Int, z_coverage_mm / resolution_mm)

    # =========================================================================
    # 2. Logging & Memory Estimate
    # =========================================================================

    total_voxels = nx * ny * nz
    mem_est_gb = (total_voxels * 5) / 1024^3  # 5 bytes/voxel

    @info """
    🏭 Manufacturing Gammex 472 Phantom
    ====================================
    Body Diameter    : $(body_diameter_mm) mm
    Rod Diameter     : $(rod_diameter_mm) mm
    Inner Ring       : 7 calcium inserts at 50 mm radius
    Outer Ring       : 7 iodine inserts at 105 mm radius
    Resolution       : $(resolution_mm) mm
    Matrix Size      : $(nx) × $(ny) × $(nz)
    Total Voxels     : $(total_voxels)
    Memory Estimate  : $(round(mem_est_gb, digits=2)) GB
    Add Texture      : $(add_texture) (σ=$(texture_std))
    ====================================
    """

    if mem_est_gb > 16.0
        @warn "⚠️  High memory usage (>16 GB). Consider coarser resolution."
    end

    # Create grid
    grid = VoxelGrid(
        size_mm = (fov_total_mm, fov_total_mm, z_coverage_mm),
        matrix = (nx, ny, nz)
    )

    # =========================================================================
    # 3. Material Definitions (XrayAttenuation.jl Integration)
    # =========================================================================

    # Use pre-defined materials from XA
    # For Gammex inserts, we approximate with standard materials
    # In reality, these are proprietary tissue-equivalent mixtures

    materials_list = [
        :water,              # ID 1: Solid water (body)
        :Ca_50,             # ID 2-8: Calcium inserts
        :Ca_100,
        :Ca_200,
        :Ca_300,
        :Ca_400,
        :Ca_500,
        :Ca_600,
        :I_2_0,             # ID 9-15: Iodine inserts
        :I_2_5,
        :I_5_0,
        :I_7_5,
        :I_10_0,
        :I_15_0,
        :I_20_0
    ]

    # Build ID mappings
    id_to_material = Dict{UInt8, Symbol}(UInt8(0) => :air)
    material_to_id = Dict{Symbol, UInt8}(:air => UInt8(0))

    for (i, mat_symbol) in enumerate(materials_list)
        id = UInt8(i)
        id_to_material[id] = mat_symbol
        material_to_id[mat_symbol] = id
    end

    # =========================================================================
    # 4. Geometry Definitions
    # =========================================================================

    # Helper function: check if point (x_cm, y_cm) is inside circle at (cx_mm, cy_mm)
    function in_circle(x_cm::Float64, y_cm::Float64, cx_mm::Float64, cy_mm::Float64, r_mm::Float64)::Bool
        # Convert x_cm, y_cm to mm for comparison
        return (x_cm * 10.0 - cx_mm)^2 + (y_cm * 10.0 - cy_mm)^2 <= r_mm^2
    end

    # Rod geometry
    hole_radius_mm = rod_diameter_mm / 2.0  # Drilled hole
    rod_radius_mm = hole_radius_mm - air_gap_mm  # Actual rod (with gap)

    # Define insert positions
    inserts = []

    # Body (solid water background)
    push!(inserts, (0.0, 0.0, body_diameter_mm/2, :water))

    # Inner ring: 7 calcium inserts at 50 mm radius
    ca_materials = [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
    for (i, mat) in enumerate(ca_materials)
        angle = (i - 1) * (2π / 7)  # Evenly spaced
        cx_mm = 50.0 * cos(angle)
        cy_mm = 50.0 * sin(angle)
        push!(inserts, (cx_mm, cy_mm, rod_radius_mm, mat))
    end

    # Outer ring: 7 iodine inserts at 105 mm radius (offset by 30°)
    i_materials = [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]
    for (i, mat) in enumerate(i_materials)
        angle = (i - 1) * (2π / 7) + deg2rad(30)  # Offset from calcium ring
        cx_mm = 105.0 * cos(angle)
        cy_mm = 105.0 * sin(angle)
        push!(inserts, (cx_mm, cy_mm, rod_radius_mm, mat))
    end

    # =========================================================================
    # 5. Voxelization (Render Phantom)
    # =========================================================================

    material_ids = fill(UInt8(0), nx, ny, nz)  # Start with air
    densities = zeros(Float32, nx, ny, nz)

    @info "🎨 Rendering phantom geometry..."

    # Parallelize over Z slices
    Threads.@threads for k in 1:nz
        z_cm = grid.z[k]

        # Only render slices within phantom bounds
        if abs(z_cm) > (z_coverage_mm / 20.0)
            continue
        end

        for j in 1:ny
            for i in 1:nx
                px_cm = grid.x[i]
                py_cm = grid.y[j]

                current_id = UInt8(0)  # Start with air

                # 1. Check if inside body
                if in_circle(px_cm, py_cm, 0.0, 0.0, body_diameter_mm / 2)
                    current_id = material_to_id[:water]
                end

                # 2. Check inserts (drill holes and place rods)
                for (cx_mm, cy_mm, r_rod_mm, mat) in inserts[2:end]  # Skip body definition
                    # Check if in drilled hole
                    if in_circle(px_cm, py_cm, cx_mm, cy_mm, hole_radius_mm)
                        # Check if in actual rod (vs air gap)
                        if in_circle(px_cm, py_cm, cx_mm, cy_mm, r_rod_mm)
                            current_id = material_to_id[mat]
                        else
                            current_id = UInt8(0)  # Air gap
                        end
                    end
                end

                material_ids[i, j, k] = current_id

                # Set density
                if current_id != 0
                    if add_texture
                        # Add small random variations (1% std by default)
                        densities[i, j, k] = 1.0f0 + Float32(randn() * texture_std)
                    else
                        densities[i, j, k] = 1.0f0
                    end
                end
            end
        end
    end

    @info "✅ Phantom rendering complete"

    # =========================================================================
    # 6. Create Phantom Data Structure
    # =========================================================================

    phantom = PhantomData(
        name = "Gammex 472 Calibration Phantom",
        grid = grid,
        material_ids = material_ids,
        densities = densities,
        id_to_material = id_to_material
    )

    # Report statistics
    unique_ids = sort(unique(material_ids))
    unique_materials = [id_to_material[id] for id in unique_ids]

    @info """
    📊 Phantom Statistics
    =====================
    Unique Materials : $(length(unique_materials))
    Materials        : $(unique_materials)
    Memory Usage     : $(round(get_memory_usage(phantom), digits=2)) GB
    =====================
    """

    return phantom
end

# ==============================================================================
# Simple Geometric Phantoms
# ==============================================================================

"""
    create_water_cylinder(;
        diameter_mm::Float64 = 200.0,
        height_mm::Float64 = 40.0,
        resolution_mm::Float64 = 1.0
    )

Create simple water cylinder phantom for basic testing.

# Example

```julia
phantom = create_water_cylinder(diameter_mm=300.0, height_mm=40.0)
```
"""
function create_water_cylinder(;
        diameter_mm::Float64 = 200.0,
        height_mm::Float64 = 40.0,
        resolution_mm::Float64 = 1.0
    )

    # FOV with margin
    fov_mm = diameter_mm + 20.0

    nx = round(Int, fov_mm / resolution_mm)
    ny = nx
    nz = round(Int, height_mm / resolution_mm)

    grid = VoxelGrid(
        size_mm = (fov_mm, fov_mm, height_mm),
        matrix = (nx, ny, nz)
    )

    # Material mapping
    id_to_material = Dict{UInt8, Symbol}(
        UInt8(0) => :air,
        UInt8(1) => :water
    )

    material_ids = fill(UInt8(0), nx, ny, nz)
    densities = zeros(Float32, nx, ny, nz)

    radius_cm = diameter_mm / 20.0

    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                r = sqrt(grid.x[i]^2 + grid.y[j]^2)
                if r <= radius_cm
                    material_ids[i, j, k] = UInt8(1)
                    densities[i, j, k] = 1.0f0
                end
            end
        end
    end

    return PhantomData(
        name = "Water Cylinder",
        grid = grid,
        material_ids = material_ids,
        densities = densities,
        id_to_material = id_to_material
    )
end

# ==============================================================================
# Utility Functions
# ==============================================================================

"""
    get_material_at_voxel(phantom::PhantomData, i::Int, j::Int, k::Int)::Symbol

Get material symbol at voxel (i, j, k).
"""
function get_material_at_voxel(phantom::PhantomData, i::Int, j::Int, k::Int)::Symbol
    id = phantom.material_ids[i, j, k]
    return phantom.id_to_material[id]
end

"""
    count_materials(phantom::PhantomData)::Dict{Symbol, Int}

Count number of voxels per material.
"""
function count_materials(phantom::PhantomData)::Dict{Symbol, Int}
    counts = Dict{Symbol, Int}()

    for id in unique(phantom.material_ids)
        mat = phantom.id_to_material[id]
        n_voxels = sum(phantom.material_ids .== id)
        counts[mat] = n_voxels
    end

    return counts
end

# ==============================================================================
# Exports
# ==============================================================================

export VoxelGrid, PhantomData
export create_gammex_472, create_water_cylinder
export get_voxel_size, get_voxel_volume, get_memory_usage
export get_material_at_voxel, count_materials
