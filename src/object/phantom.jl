"""
    Geometry/Phantom.jl

Phantom generation with semantic masks for validation.

Every phantom comes with a mask that identifies regions (air, water, inserts, etc.)
for automated testing and validation.
"""

# =============================================================================
# Region Labels
# =============================================================================

"""
Region labels for phantom masks.

Each voxel in the mask is assigned a label indicating its material type.
Used for automated validation after reconstruction.

All labels are prefixed with REGION_ to avoid conflicts with material exports.
"""
@enum RegionLabel::UInt8 begin
    REGION_BACKGROUND = 0
    REGION_AIR = 1
    REGION_WATER = 2
    REGION_SOLID_WATER = 3
    # Calcium inserts (Gammex 472)
    REGION_CA_50 = 10
    REGION_CA_100 = 11
    REGION_CA_200 = 12
    REGION_CA_300 = 13
    REGION_CA_400 = 14
    REGION_CA_500 = 15
    REGION_CA_600 = 16
    # Iodine inserts (Gammex 472)
    REGION_I_2_0 = 20
    REGION_I_2_5 = 21
    REGION_I_5_0 = 22
    REGION_I_7_5 = 23
    REGION_I_10_0 = 24
    REGION_I_15_0 = 25
    REGION_I_20_0 = 26
end

# Map from RegionLabel to material symbol
const REGION_TO_MATERIAL = Dict{RegionLabel, Symbol}(
    REGION_BACKGROUND => :air,
    REGION_AIR => :air,
    REGION_WATER => :water,
    REGION_SOLID_WATER => :solid_water,
    REGION_CA_50 => :Ca_50,
    REGION_CA_100 => :Ca_100,
    REGION_CA_200 => :Ca_200,
    REGION_CA_300 => :Ca_300,
    REGION_CA_400 => :Ca_400,
    REGION_CA_500 => :Ca_500,
    REGION_CA_600 => :Ca_600,
    REGION_I_2_0 => :I_2_0,
    REGION_I_2_5 => :I_2_5,
    REGION_I_5_0 => :I_5_0,
    REGION_I_7_5 => :I_7_5,
    REGION_I_10_0 => :I_10_0,
    REGION_I_15_0 => :I_15_0,
    REGION_I_20_0 => :I_20_0,
)

# =============================================================================
# Phantom Struct
# =============================================================================

"""
    Phantom

Digital phantom with semantic mask and materials for polychromatic simulation.

# Fields (v20.0-pivot: simplified, no μ field)
- `mask::AbstractArray{<:Unsigned,3}`: Region labels (see `RegionLabel` enum)
- `materials::Vector{XA.Material}`: Materials for each region (indexed by mask_value + 1)
- `voxel_size::NTuple{3,Float64}`: Voxel dimensions (cm) as (dx, dy, dz)
- `origin::NTuple{3,Float64}`: Origin coordinates (cm) - center of first voxel
- `extent::NTuple{3,Float64}`: Physical extent (cm) as (x, y, z)

# Coordinate System
- X: left-right (increasing right)
- Y: anterior-posterior (increasing posterior)
- Z: inferior-superior (increasing superior)
- Origin at isocenter (0, 0, 0)

# Design (v20.0-pivot)
The μ field was removed because polychromatic simulation computes μ(E) on-demand
at each spectrum energy via `create_μ_volume!()`. The pre-computed μ at arbitrary
60 keV was redundant and confusing.

Use `compute_μ(phantom, energy_keV)` to get attenuation coefficients at any energy.

# Usage
```julia
# Create phantom (no energy_keV needed!)
materials_dict = Dict(0 => XA.Materials.air, 1 => XA.Materials.water)
phantom = Phantom(labeled_array, materials_dict, (0.1, 0.1, 0.1))

# Get μ at any energy when needed
μ_60keV = compute_μ(phantom, 60.0)
μ_120keV = compute_μ(phantom, 120.0)

# Simulate - just works (uses mask + materials internally)
result = simulate(phantom, scanner, protocol)

# GPU workflow: mask on GPU, materials stay on CPU
using Metal
phantom_gpu = Phantom(
    MtlArray(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent
)
```

See also: [`compute_μ`](@ref)
"""
struct Phantom{T<:Unsigned, M<:AbstractArray{T,3}, Mat}
    mask::M
    materials::Mat  # Vector{XA.Material}
    voxel_size::NTuple{3,Float64}
    origin::NTuple{3,Float64}
    extent::NTuple{3,Float64}
end

# =============================================================================
# Compute μ On-Demand
# =============================================================================

"""
    compute_μ(phantom::Phantom, energy_keV::Real) -> Array{Float32,3}

Compute linear attenuation coefficient volume at specified energy.

This function efficiently computes μ by:
1. Computing μ for each unique material once (O(n_materials))
2. Broadcasting via mask indexing (O(n_voxels), but just integer lookups)

This is exactly what `create_μ_volume!()` does internally in polychromatic
forward projection, so there's no performance penalty vs the old μ field.

# Arguments
- `phantom::Phantom`: Phantom with mask and materials
- `energy_keV::Real`: Energy in keV for attenuation computation

# Returns
- `Array{Float32,3}`: Linear attenuation coefficients (cm⁻¹) at specified energy

# Example
```julia
phantom = create_gammex_472(n_voxels=128)

# Get μ at different energies
μ_60 = compute_μ(phantom, 60.0)   # ~0.207 cm⁻¹ for water
μ_120 = compute_μ(phantom, 120.0) # ~0.165 cm⁻¹ for water
```
"""
function compute_μ(phantom::Phantom, energy_keV::Real)
    # Compute μ for each material at this energy (O(n_materials))
    μ_lookup = Float32[compute_μ_at_energy(mat, Float64(energy_keV))
                       for mat in phantom.materials]
    # Broadcast via mask indexing (mask is 0-based, vector is 1-based)
    return μ_lookup[phantom.mask .+ 1]
end

# =============================================================================
# Unified Phantom Constructor (v20.0)
# =============================================================================

"""
    Phantom(labeled_array, materials_dict, voxel_size_cm; kwargs...) -> Phantom

Create a Phantom from a labeled array with materials stored internally.

This is the **unified v20.0-pivot API**: the returned Phantom contains everything needed
for polychromatic simulation, so `simulate(phantom, scanner, protocol)` just works
without a separate `materials` kwarg.

**No energy_keV parameter needed!** The μ field was removed in v20.0-pivot. Use
`compute_μ(phantom, energy_keV)` to get attenuation coefficients at any energy.

# Arguments
- `labeled_array::AbstractArray{<:Integer, 3}`: Integer array where each voxel
  contains a region label (0-255 supported via UInt8 conversion)
- `materials_dict::Dict{Int, <:Any}`: Mapping from label values to materials.
  Materials can be:
  - `XA.Material`: Direct XrayAttenuation.jl material
  - `Symbol`: Material name to look up (e.g., `:water`, `:Ca_100`)
- `voxel_size_cm::NTuple{3, Real}`: Physical voxel dimensions in cm as (dx, dy, dz)

# Keyword Arguments
- `origin::Union{Nothing, NTuple{3, Real}}=nothing`: Origin coordinates (cm).
  If `nothing`, phantom is centered at isocenter.

# Returns
A `Phantom` with:
- `mask`: UInt8 mask with original label values
- `materials`: Vector{XA.Material} for polychromatic simulation
- `voxel_size`, `origin`, `extent`: Geometry parameters

# Example

```julia
using BasisSimulator, XrayAttenuation
import XrayAttenuation as XA

# Define materials
materials_dict = Dict{Int, XA.Material}(
    0 => XA.Materials.air,
    1 => XA.Materials.water,
    2 => XA.Materials.cortical_bone
)

# Create phantom (1mm voxels)
phantom = Phantom(labeled_array, materials_dict, (0.1, 0.1, 0.1))

# Get μ at any energy when needed
μ_60keV = compute_μ(phantom, 60.0)

# Simulate - no materials kwarg needed!
result = simulate(phantom, scanner, protocol, SimOptions(), ReconOptions())
```

See also: [`compute_μ`](@ref), [`create_phantom_from_mask`](@ref), [`create_gammex_472`](@ref)
"""
function Phantom(
    labeled_array::AbstractArray{<:Integer, 3},
    materials_dict::Dict{Int, M},
    voxel_size_cm::NTuple{3, Real};
    origin::Union{Nothing, NTuple{3, Real}} = nothing
) where M
    # Get dimensions
    nx, ny, nz = size(labeled_array)
    dx, dy, dz = Float64.(voxel_size_cm)

    # Compute physical extent
    ext_x = dx * nx
    ext_y = dy * ny
    ext_z = dz * nz

    # Compute origin (center at isocenter if not specified)
    if origin === nothing
        origin_x = -ext_x/2 + dx/2
        origin_y = -ext_y/2 + dy/2
        origin_z = -ext_z/2 + dz/2
        computed_origin = (origin_x, origin_y, origin_z)
    else
        computed_origin = Float64.(origin)
    end

    # Convert labeled array to UInt8 or UInt16 mask (auto-promote based on max label)
    max_label_val = maximum(labeled_array)
    mask = max_label_val > typemax(UInt8) ? UInt16.(labeled_array) : UInt8.(labeled_array)

    # Build materials vector (indexed by mask_value + 1)
    materials_vec = build_materials_vector(materials_dict)

    return Phantom(
        mask,
        materials_vec,
        (dx, dy, dz),
        computed_origin,
        (ext_x, ext_y, ext_z)
    )
end

# =============================================================================
# Gammex 472 Phantom
# =============================================================================

"""
    create_gammex_472(; n_voxels=64, n_slices=nothing, fov_cm=35.0, z_cm=4.0)

Create a Gammex 472 calibration phantom with semantic mask.

# Arguments
- `n_voxels::Int`: Voxels per side in x/y (default 64 for fast iteration)
- `n_slices::Union{Int,Nothing}`: Number of z slices (if specified, overrides z_cm calculation)
- `fov_cm::Float64`: Field of view in x/y (cm), default 35.0
- `z_cm::Float64`: Height in z (cm), default 4.0 (used if n_slices not specified)

# Returns
`Phantom` with:
- 330mm diameter solid water body
- 7 calcium inserts (50-600 mg/ml) in inner ring (5cm radius)
- 7 iodine inserts (2-20 mg/ml) in outer ring (10.5cm radius)
- 28mm diameter rods
- Semantic mask labeling each region
- Materials vector for polychromatic simulation

Use `compute_μ(phantom, energy_keV)` to get attenuation coefficients at any energy.

# Gammex 472 Specifications
- Body: 330mm diameter solid water cylinder
- Insert rods: 28mm diameter
- Inner ring radius: 50mm (calcium inserts)
- Outer ring radius: 105mm (iodine inserts)
- Insert spacing: ~51.4° (7 inserts per ring)

# Example
```julia
phantom = create_gammex_472(n_voxels=128)

# Get μ at any energy
μ_60 = compute_μ(phantom, 60.0)
μ_120 = compute_μ(phantom, 120.0)

# Simulate - just works
result = simulate(phantom, scanner, protocol)
```
"""
function create_gammex_472(;
    n_voxels::Int=64,
    n_slices::Union{Int,Nothing}=nothing,
    fov_cm::Float64=35.0,
    z_cm::Float64=4.0
)
    # Grid setup - use n_slices if specified, otherwise compute from z_cm
    n_z = if n_slices !== nothing
        n_slices
    else
        max(1, round(Int, n_voxels * z_cm / fov_cm))
    end
    dx = fov_cm / n_voxels
    dy = fov_cm / n_voxels
    dz = z_cm / n_z

    # Coordinate arrays (centered at isocenter)
    x = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)
    y = range(-fov_cm/2 + dy/2, fov_cm/2 - dy/2, length=n_voxels)

    # Initialize mask array only (no μ array - computed on demand)
    mask = zeros(UInt8, n_voxels, n_voxels, n_z)

    # Gammex 472 dimensions (cm)
    body_radius = 16.5    # 330mm diameter
    rod_radius = 1.4      # 28mm diameter
    inner_ring_radius = 5.0   # 50mm - calcium inserts
    outer_ring_radius = 10.5  # 105mm - iodine inserts

    # Region labels for inserts
    ca_labels = [REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600]
    i_labels = [REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0]

    # Insert angular positions (evenly spaced, starting at 0°)
    n_inserts = 7
    angles_ca = [2π * i / n_inserts for i in 0:(n_inserts-1)]
    angles_i = [2π * i / n_inserts + π/n_inserts for i in 0:(n_inserts-1)]  # Offset by half spacing

    # Fill mask voxel by voxel (geometry only, no μ computation)
    for k in 1:n_z
        for j in 1:n_voxels
            for i in 1:n_voxels
                xi = x[i]
                yj = y[j]
                r = sqrt(xi^2 + yj^2)

                # Default: background (air)
                mask[i, j, k] = UInt8(REGION_BACKGROUND)

                # Check if inside body cylinder
                if r <= body_radius
                    # Default body is solid water
                    mask[i, j, k] = UInt8(REGION_SOLID_WATER)

                    # Check calcium inserts (inner ring)
                    for (idx, angle) in enumerate(angles_ca)
                        cx = inner_ring_radius * cos(angle)
                        cy = inner_ring_radius * sin(angle)
                        dist = sqrt((xi - cx)^2 + (yj - cy)^2)
                        if dist <= rod_radius
                            mask[i, j, k] = UInt8(ca_labels[idx])
                            break
                        end
                    end

                    # Check iodine inserts (outer ring) - only if not already assigned
                    if mask[i, j, k] == UInt8(REGION_SOLID_WATER)
                        for (idx, angle) in enumerate(angles_i)
                            cx = outer_ring_radius * cos(angle)
                            cy = outer_ring_radius * sin(angle)
                            dist = sqrt((xi - cx)^2 + (yj - cy)^2)
                            if dist <= rod_radius
                                mask[i, j, k] = UInt8(i_labels[idx])
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    # Build materials vector for all region labels used in Gammex 472
    # Index = label_value + 1 (since Julia is 1-indexed)
    materials_dict = Dict{Int, XA.Material}(
        Int(REGION_BACKGROUND) => XA.Materials.air,
        Int(REGION_AIR) => XA.Materials.air,
        Int(REGION_WATER) => XA.Materials.water,
        Int(REGION_SOLID_WATER) => XA.Materials.water,  # Solid water approximated as water
        Int(REGION_CA_50) => get_material(:Ca_50),
        Int(REGION_CA_100) => get_material(:Ca_100),
        Int(REGION_CA_200) => get_material(:Ca_200),
        Int(REGION_CA_300) => get_material(:Ca_300),
        Int(REGION_CA_400) => get_material(:Ca_400),
        Int(REGION_CA_500) => get_material(:Ca_500),
        Int(REGION_CA_600) => get_material(:Ca_600),
        Int(REGION_I_2_0) => get_material(:I_2_0),
        Int(REGION_I_2_5) => get_material(:I_2_5),
        Int(REGION_I_5_0) => get_material(:I_5_0),
        Int(REGION_I_7_5) => get_material(:I_7_5),
        Int(REGION_I_10_0) => get_material(:I_10_0),
        Int(REGION_I_15_0) => get_material(:I_15_0),
        Int(REGION_I_20_0) => get_material(:I_20_0),
    )
    materials_vec = build_materials_vector(materials_dict)

    return Phantom(
        mask,
        materials_vec,
        (dx, dy, dz),
        (-fov_cm/2 + dx/2, -fov_cm/2 + dy/2, -z_cm/2 + dz/2),
        (Float64(fov_cm), Float64(fov_cm), Float64(z_cm))
    )
end

# =============================================================================
# Create Phantom from Arbitrary Labeled Array
# =============================================================================

"""
    create_phantom_from_mask(labeled_array, materials, voxel_size_cm; kwargs...) -> Phantom

Create a Phantom from an arbitrary labeled array with custom material mapping.

This function enables loading arbitrary phantoms (XCAT, custom segmentations, etc.)
by providing a mapping from integer labels to materials.

**Note (v20.0-pivot):** This function is now equivalent to the `Phantom()` constructor.
Consider using `Phantom(labeled_array, materials_dict, voxel_size)` directly.

# Arguments
- `labeled_array::AbstractArray{<:Integer, 3}`: Integer array where each voxel
  contains a region label (0-255 supported via UInt8 conversion)
- `materials::Dict{Int, <:Any}`: Mapping from label values to materials. Materials
  can be:
  - `XA.Material`: Direct XrayAttenuation.jl material
  - `Symbol`: Material name to look up in MATERIALS_REGISTRY (e.g., `:water`, `:Ca_100`)
- `voxel_size_cm::NTuple{3, Real}`: Physical voxel dimensions in cm as (dx, dy, dz)

# Keyword Arguments
- `origin::Union{Nothing, NTuple{3, Real}}=nothing`: Origin coordinates (cm).
  If `nothing`, phantom is centered at isocenter.

# Returns
A `Phantom` struct with:
- `mask`: UInt8 mask with original label values
- `materials`: Vector{XA.Material} for polychromatic simulation
- `voxel_size`: Physical voxel dimensions (cm)
- `origin`: Origin coordinates (cm)
- `extent`: Physical extent (cm)

Use `compute_μ(phantom, energy_keV)` to get attenuation coefficients at any energy.

# Example

```julia
using BasisSimulator, XrayAttenuation
import XrayAttenuation as XA

# Load XCAT phantom (hypothetical)
xcat_mask = load_phantom_bin("xcat.bin"; cols=400, rows=400, slices=200)

# Define materials for each label
materials_dict = Dict{Int, XA.Material}(
    0 => XA.Materials.air,
    1 => XA.Materials.water,  # soft tissue approximation
    2 => XA.Materials.cortical_bone,
    3 => XA.Materials.lung,
)

# Create phantom (0.1 cm = 1mm voxels)
phantom = create_phantom_from_mask(xcat_mask, materials_dict, (0.1, 0.1, 0.1))

# Get μ at any energy when needed
μ_70 = compute_μ(phantom, 70.0)

# Simulate - just works (materials stored in phantom)
result = simulate(phantom, scanner, protocol)
```

See also: [`Phantom`](@ref), [`compute_μ`](@ref), [`create_gammex_472`](@ref)
"""
function create_phantom_from_mask(
    labeled_array::AbstractArray{<:Integer, 3},
    materials::Dict{Int, M},
    voxel_size_cm::NTuple{3, Real};
    origin::Union{Nothing, NTuple{3, Real}} = nothing
) where M
    # Delegate to the unified Phantom constructor
    return Phantom(labeled_array, materials, voxel_size_cm; origin=origin)
end

"""
    build_materials_vector(materials_dict::Dict{Int, <:Any}) -> Vector{XA.Material}

Build a materials vector from a materials dictionary for use with `simulate()`.

The returned vector is indexed by `mask_value + 1`, so `materials_vec[1]` corresponds
to label 0, `materials_vec[2]` to label 1, etc.

# Arguments
- `materials_dict::Dict{Int, <:Any}`: Mapping from label values to materials.
  Materials can be `XA.Material` or `Symbol`.

# Returns
`Vector{XA.Material}` with length `max_label + 1`.

# Example
```julia
materials_dict = Dict(
    0 => XA.Materials.air,
    1 => :water,
    2 => XA.Materials.cortical_bone
)
materials_vec = build_materials_vector(materials_dict)
# materials_vec has 3 elements: [air, water, bone]
```
"""
function build_materials_vector(materials_dict::Dict{Int, M}) where M
    max_label = maximum(keys(materials_dict))
    materials_vec = Vector{XA.Material}(undef, max_label + 1)

    # Fill with air by default
    for i in 1:(max_label + 1)
        materials_vec[i] = XA.Materials.air
    end

    # Fill from dict
    for (label, mat) in materials_dict
        if mat isa Symbol
            materials_vec[label + 1] = get_material(mat)
        else
            materials_vec[label + 1] = mat
        end
    end

    return materials_vec
end

# =============================================================================
# Validation Functions
# =============================================================================

"""
    get_region_mask(phantom::Phantom, label::RegionLabel)

Get a boolean mask for a specific region.

# Returns
`BitArray{3}` where `true` indicates voxels belonging to the region.
"""
function get_region_mask(phantom::Phantom, label::RegionLabel)
    return phantom.mask .== UInt8(label)
end


"""
    get_xcat_material(symbol::Symbol) -> XA.Material

Get XrayAttenuation material for XCAT semantic categories.
"""
function get_xcat_material(symbol::Symbol)::XA.Material
    return get_xcat_material_dict(symbol)
end

const XCAT_MATERIALS = Dict{Symbol, XA.Material}()

function get_xcat_material_dict(symbol::Symbol)::XA.Material
    if isempty(XCAT_MATERIALS)
        initialize_xcat_materials()
    end
    return get(XCAT_MATERIALS, symbol, XA.Materials.water)
end

function initialize_xcat_materials()
    XCAT_MATERIALS[:air] = XA.Materials.air
    XCAT_MATERIALS[:soft_tissue] = XA.Materials.water
    XCAT_MATERIALS[:bone] = XA.Materials.cortical_bone
    XCAT_MATERIALS[:brain] = XA.Materials.brain
    XCAT_MATERIALS[:blood] = XA.Materials.blood
    XCAT_MATERIALS[:csf] = XA.Materials.csf
    XCAT_MATERIALS[:cartilage] = XA.Materials.cortical_bone
end

"""
    create_phantom_from_xcat(
        mask::Array{UInt16,3},
        material_mapping::Dict{Int, Symbol},
        voxel_size_cm::NTuple{3, Float64}
    ) -> Phantom

Create a BasisSimulator Phantom from XCAT classification data.

# Arguments
- `mask::Array{UInt16,3}`: Raw XCAT mask with structure IDs
- `material_mapping::Dict{Int, Symbol}`: Mapping from structure ID to material symbol
- `voxel_size_cm::NTuple{3, Float64}`: Voxel dimensions in cm

# Returns
Phantom ready for simulation
"""
function create_phantom_from_xcat(
    mask::Array{UInt16,3},
    material_mapping::Dict{Int, Symbol},
    voxel_size_cm::NTuple{3, Float64}
)::Phantom
    initialize_xcat_materials()
    
    unique_ids = unique(mask)
    
    materials_dict = Dict{Int, XA.Material}()
    for id in unique_ids
        if haskey(material_mapping, Int(id))
            mat_symbol = material_mapping[Int(id)]
            materials_dict[Int(id)] = get_xcat_material(mat_symbol)
        else
            materials_dict[Int(id)] = XA.Materials.air
        end
    end
    
    return Phantom(mask, materials_dict, voxel_size_cm)
end

# =============================================================================
# Phantom Utilities
# =============================================================================

"""
    relabel_zero_islands_2d!(arr::Array{Int,3}; newlabel::Int=10)

For each z-slice of `arr`, find connected components of zero-valued voxels that do
not touch the slice border (interior "islands") and relabel them to `newlabel`.
Border-connected zeros (true background / air outside the head) are left as 0.

This corrects XCAT P1 raw files where interior voxels (e.g. inside the skull) are
stored as 0 rather than a valid tissue ID.
"""
function relabel_zero_islands_2d!(arr::Array{Int,3}; newlabel::Int=10)
    nx, ny, nz = size(arr)
    for z in 1:nz
        slice   = view(arr, :, :, z)
        visited = falses(nx, ny)
        queue   = Tuple{Int,Int}[]

        # Seed BFS from every border pixel that is zero
        for x in 1:nx
            for y in (1, ny)
                if slice[x, y] == 0 && !visited[x, y]
                    visited[x, y] = true
                    push!(queue, (x, y))
                end
            end
        end
        for y in 2:ny-1
            for x in (1, nx)
                if slice[x, y] == 0 && !visited[x, y]
                    visited[x, y] = true
                    push!(queue, (x, y))
                end
            end
        end

        # BFS: flood-fill all border-connected zeros
        while !isempty(queue)
            cx, cy = popfirst!(queue)
            for (dx, dy) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                nx2, ny2 = cx + dx, cy + dy
                (nx2 < 1 || nx2 > nx || ny2 < 1 || ny2 > ny) && continue
                if slice[nx2, ny2] == 0 && !visited[nx2, ny2]
                    visited[nx2, ny2] = true
                    push!(queue, (nx2, ny2))
                end
            end
        end

        # Any zero voxel not reached from the border is an interior island
        for x in 1:nx, y in 1:ny
            if slice[x, y] == 0 && !visited[x, y]
                arr[x, y, z] = newlabel
            end
        end
    end
    return arr
end

"""
    load_structure_map(path::String) -> Dict{Int, String}

Load an XCAT voxelize table (tab-separated `name\\tID` per line, no header).
Returns a `Dict` mapping integer segment ID → segment name.
"""
function load_structure_map(path::String)::Dict{Int, String}
    result = Dict{Int, String}()
    open(path, "r") do io
        for line in eachline(io)
            parts = split(strip(line), '\t')
            length(parts) == 2 || continue
            name = strip(parts[1])
            id   = tryparse(Int, strip(parts[2]))
            (id === nothing || isempty(name)) && continue
            result[id] = name
        end
    end
    return result
end


# =============================================================================
# XCAT I/O Helpers
# =============================================================================

"""
    update_structures!(new_phantom_shift, structure_map, tissue_prefix,
                       raw_file, base_sym, base_mat, info_table,
                       iodine_matrix, t_contrast) -> Dict{Int, XA.Material}

Stamp XCAT segment IDs from a reference phantom (`raw_file`) into `new_phantom_shift`,
and build per-segment iodine-doped materials for each segment whose name starts with
`tissue_prefix`.

# Arguments
- `new_phantom_shift::Array{Int,3}`: Target array (modified in-place).
- `structure_map::Dict{Int,String}`: ID → name mapping from `load_structure_map`.
- `tissue_prefix::String`: Only segments whose name *starts with* this string are
  processed (e.g. `"5"`, `"4"`, `"3"`, `"2"` for XCAT brain prefix codes).
- `raw_file::Array{Int,3}`: Reference subject's raw phantom (label source).
- `base_sym::Symbol`: Key into `MATERIALS_REGISTRY` for the base tissue
  (e.g. `:gray_matter`, `:white_matter`, `:blood`).
- `base_mat::XA.Material`: Pre-resolved material for `base_sym`.
- `info_table::Dict{String,Any}`: Must have `"name"::Vector{String}` and
  `"volume"::Vector{Float64}` (cm³ for `:gray_matter`/`:white_matter`,
  mm³ otherwise — automatic unit conversion applied).
- `iodine_matrix::Matrix{Float64}`: rows = segments, cols = time points; values in **mg**.
- `t_contrast::Int`: Column index (1-based) for the desired contrast time point.

# Returns
`Dict{Int, XA.Material}` mapping each stamped segment ID to its iodine-doped material.
IDs with no corresponding row in `info_table` are silently skipped.
# Notes
For `:gray_matter` / `:white_matter` the P2 segment name strings do not match the
`info_table["name"]` entries, so a **positional index** is used instead: segments are
sorted by ascending P2 ID and paired with `info_table` rows by position (row 1, 2, 3…).
This matches the reference `Dynamic_Contrast.jl` behaviour.
For arteries/veins the stripped segment names do match `info_table["name"]`, so a
name-based lookup is used.
"""
function update_structures!(
    new_phantom_shift::Array{Int,3},
    structure_map::Dict{Int,String},
    tissue_prefix::String,
    raw_file::Array{Int,3},
    base_sym::Symbol,
    base_mat::XA.Material,
    info_table::Dict{String,Any},
    iodine_matrix::Matrix{Float64},
    t_contrast::Int,
)::Dict{Int, XA.Material}
    # --- 1. Filter segments by prefix and sort by ascending ID ---
    entries = filter(kv -> startswith(kv[2], tissue_prefix), structure_map)
    sorted_pairs = sort(collect(entries), by = kv -> kv[1])   # ascending ID
    ids   = [kv[1] for kv in sorted_pairs]
    names = [replace(kv[2], r"^\d{4}_" => "") for kv in sorted_pairs]
    # --- 2. Stamp IDs from reference raw_file into target new_phantom_shift ---
    for id in ids
        idxs = findall(==(id), raw_file)
        isempty(idxs) && continue
        new_phantom_shift[idxs] .= id
    end
    # --- 3. Build iodine-doped material per segment ---
    seg_materials = Dict{Int, XA.Material}()
    density = ustrip(u"g/cm^3", base_mat.density)
    use_positional = (base_sym == :gray_matter || base_sym == :white_matter)
    n_rows = size(iodine_matrix, 1)

    for (i, (id, name)) in enumerate(zip(ids, names))
        # Determine row in info_table
        if use_positional
            # Reference behavior: i-th P2 segment (sorted by ID) → i-th info_table row
            i > n_rows && continue
            row = i
            segment_volume_mL = info_table["volume"][row]           # cm³ for GM/WM
        else
            # Arteries/veins: match by stripped name
            row = findfirst(==(name), info_table["name"])
            row === nothing && continue
            segment_volume_mL = info_table["volume"][row] / 1000.0  # mm³ → cm³
        end

        segment_volume_mL <= 0 && continue
        mass_I_mg = iodine_matrix[row, t_contrast]  # mg
        conc_mg_per_mL = mass_I_mg / segment_volume_mL
        seg_materials[id] = iodine_contrast_material(base_mat, conc_mg_per_mL;
            density_g_per_mL=density)
    end
    return seg_materials
end

# =============================================================================
# Exports
# =============================================================================
export RegionLabel
export REGION_BACKGROUND, REGION_AIR, REGION_WATER, REGION_SOLID_WATER
export REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600
export REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0
export REGION_TO_MATERIAL
export get_region_mask
export create_phantom_from_xcat
export relabel_zero_islands_2d!, load_structure_map
export update_structures!