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

Digital phantom with semantic mask for validation and optional materials for polychromatic simulation.

# Fields
- `μ::Array{Float32,3}`: Linear attenuation coefficients (cm⁻¹) at effective energy
- `mask::Array{UInt8,3}`: Region labels (see `RegionLabel` enum)
- `materials::Union{Vector{XA.Material}, Nothing}`: Materials for polychromatic simulation (v20.0+)
- `voxel_size::NTuple{3,Float64}`: Voxel dimensions (cm) as (dx, dy, dz)
- `origin::NTuple{3,Float64}`: Origin coordinates (cm) - center of first voxel
- `fov::NTuple{3,Float64}`: Field of view (cm) as (x, y, z)

# Coordinate System
- X: left-right (increasing right)
- Y: anterior-posterior (increasing posterior)
- Z: inferior-superior (increasing superior)
- Origin at isocenter (0, 0, 0)

# Materials Field (v20.0)
When `materials` is populated, `simulate(phantom, ...)` automatically uses it for
polychromatic physics without needing a separate `materials` kwarg.

Materials vector is indexed by `mask_value + 1`, so `materials[1]` corresponds to
region label 0, `materials[2]` to label 1, etc.

# Usage
```julia
# Create phantom with materials (v20.0 unified API)
materials_dict = Dict(0 => XA.Materials.air, 1 => XA.Materials.water)
phantom = Phantom(labeled_array, materials_dict, (0.1, 0.1, 0.1))
result = simulate(phantom, scanner, protocol)  # Just works!

# Legacy: GPU phantom (Metal) - materials stay on CPU
using Metal
phantom_gpu = Phantom(
    MtlArray(phantom_cpu.μ),
    MtlArray(phantom_cpu.mask),
    phantom_cpu.materials,  # CPU reference, not transferred to GPU
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.fov
)
```
"""
struct Phantom{T<:AbstractArray{Float32,3}, M<:AbstractArray{UInt8,3}, Mat}
    μ::T
    mask::M
    materials::Mat  # Vector{XA.Material} or Nothing
    voxel_size::NTuple{3,Float64}
    origin::NTuple{3,Float64}
    fov::NTuple{3,Float64}
end

# Backwards-compatible constructor (5 args, no materials)
function Phantom(
    μ::T,
    mask::M,
    voxel_size::NTuple{3,Float64},
    origin::NTuple{3,Float64},
    fov::NTuple{3,Float64}
) where {T<:AbstractArray{Float32,3}, M<:AbstractArray{UInt8,3}}
    return Phantom{T, M, Nothing}(μ, mask, nothing, voxel_size, origin, fov)
end

# =============================================================================
# Unified Phantom Constructor (v20.0)
# =============================================================================

"""
    Phantom(labeled_array, materials_dict, voxel_size_cm; kwargs...) -> Phantom

Create a Phantom from a labeled array with materials stored internally.

This is the **unified v20.0 API**: the returned Phantom contains everything needed
for polychromatic simulation, so `simulate(phantom, scanner, protocol)` just works
without a separate `materials` kwarg.

# Arguments
- `labeled_array::AbstractArray{<:Integer, 3}`: Integer array where each voxel
  contains a region label (0-255 supported via UInt8 conversion)
- `materials_dict::Dict{Int, <:Any}`: Mapping from label values to materials.
  Materials can be:
  - `XA.Material`: Direct XrayAttenuation.jl material
  - `Symbol`: Material name to look up (e.g., `:water`, `:Ca_100`)
- `voxel_size_cm::NTuple{3, Real}`: Physical voxel dimensions in cm as (dx, dy, dz)

# Keyword Arguments
- `energy_keV::Real=60.0`: Energy for computing reference μ values (keV)
- `origin::Union{Nothing, NTuple{3, Real}}=nothing`: Origin coordinates (cm).
  If `nothing`, phantom is centered at isocenter.

# Returns
A `Phantom` with:
- `μ`: Linear attenuation coefficients (cm⁻¹) at specified energy
- `mask`: UInt8 mask with original label values
- `materials`: Vector{XA.Material} for polychromatic simulation
- `voxel_size`, `origin`, `fov`: Geometry parameters

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

# Simulate - no materials kwarg needed!
result = simulate(phantom, scanner, protocol, SimOptions(), ReconOptions())
```

See also: [`create_phantom_from_mask`](@ref), [`create_gammex_472`](@ref)
"""
function Phantom(
    labeled_array::AbstractArray{<:Integer, 3},
    materials_dict::Dict{Int, M},
    voxel_size_cm::NTuple{3, Real};
    energy_keV::Real = 60.0,
    origin::Union{Nothing, NTuple{3, Real}} = nothing
) where M
    # Get dimensions
    nx, ny, nz = size(labeled_array)
    dx, dy, dz = Float64.(voxel_size_cm)

    # Compute FOV
    fov_x = dx * nx
    fov_y = dy * ny
    fov_z = dz * nz

    # Compute origin (center at isocenter if not specified)
    if origin === nothing
        origin_x = -fov_x/2 + dx/2
        origin_y = -fov_y/2 + dy/2
        origin_z = -fov_z/2 + dz/2
        computed_origin = (origin_x, origin_y, origin_z)
    else
        computed_origin = Float64.(origin)
    end

    # Convert labeled array to UInt8 mask
    mask = UInt8.(labeled_array)

    # Build materials vector (indexed by mask_value + 1)
    materials_vec = build_materials_vector(materials_dict)

    # Build μ array from materials
    μ = zeros(Float32, nx, ny, nz)

    # Pre-compute μ for each unique label
    unique_labels = unique(labeled_array)
    μ_lookup = Dict{Int, Float32}()

    for label in unique_labels
        if !haskey(materials_dict, label)
            @warn "Label $label not found in materials dict, using air"
            mat = XA.Materials.air
        else
            mat = materials_dict[label]
            # Handle Symbol lookup
            if mat isa Symbol
                mat = get_material(mat)
            end
        end
        μ_lookup[label] = Float32(compute_μ_at_energy(mat, Float64(energy_keV)))
    end

    # Fill μ array
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                label = labeled_array[i, j, k]
                μ[i, j, k] = μ_lookup[label]
            end
        end
    end

    return Phantom(
        μ,
        mask,
        materials_vec,
        (dx, dy, dz),
        computed_origin,
        (fov_x, fov_y, fov_z)
    )
end

# =============================================================================
# Gammex 472 Phantom
# =============================================================================

"""
    create_gammex_472(; n_voxels=64, n_slices=nothing, fov_cm=35.0, z_cm=4.0, μ_effective_energy_keV=60.0)

Create a Gammex 472 calibration phantom with semantic mask.

# Arguments
- `n_voxels::Int`: Voxels per side in x/y (default 64 for fast iteration)
- `n_slices::Union{Int,Nothing}`: Number of z slices (if specified, overrides z_cm calculation)
- `fov_cm::Float64`: Field of view in x/y (cm), default 35.0
- `z_cm::Float64`: Height in z (cm), default 4.0 (used if n_slices not specified)
- `μ_effective_energy_keV::Float64`: Energy for μ values (keV), default 60.0

# Returns
`Phantom` with:
- 330mm diameter solid water body
- 7 calcium inserts (50-600 mg/ml) in inner ring (5cm radius)
- 7 iodine inserts (2-20 mg/ml) in outer ring (10.5cm radius)
- 28mm diameter rods
- Semantic mask labeling each region

# Gammex 472 Specifications
- Body: 330mm diameter solid water cylinder
- Insert rods: 28mm diameter
- Inner ring radius: 50mm (calcium inserts)
- Outer ring radius: 105mm (iodine inserts)
- Insert spacing: ~51.4° (7 inserts per ring)
"""
function create_gammex_472(;
    n_voxels::Int=64,
    n_slices::Union{Int,Nothing}=nothing,
    fov_cm::Float64=35.0,
    z_cm::Float64=4.0,
    μ_effective_energy_keV::Float64=60.0
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
    z = range(-z_cm/2 + dz/2, z_cm/2 - dz/2, length=n_z)

    # Initialize arrays
    μ = zeros(Float32, n_voxels, n_voxels, n_z)
    mask = zeros(UInt8, n_voxels, n_voxels, n_z)

    # Gammex 472 dimensions (cm)
    body_radius = 16.5    # 330mm diameter
    rod_radius = 1.4      # 28mm diameter
    inner_ring_radius = 5.0   # 50mm - calcium inserts
    outer_ring_radius = 10.5  # 105mm - iodine inserts

    # Get materials and compute μ values
    # Use pure water for the phantom body (water-equivalent)
    solid_water_mat = XA.Materials.water
    air_mat = XA.Materials.air

    μ_solid_water = Float32(compute_μ_at_energy(solid_water_mat, μ_effective_energy_keV))
    μ_air = Float32(compute_μ_at_energy(air_mat, μ_effective_energy_keV))

    # Calcium inserts (inner ring) - 7 inserts evenly spaced
    ca_materials = [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
    ca_labels = [REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600]
    ca_μ = [Float32(compute_μ_at_energy(get_material(m), μ_effective_energy_keV)) for m in ca_materials]

    # Iodine inserts (outer ring) - 7 inserts evenly spaced
    i_materials = [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]
    i_labels = [REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0]
    i_μ = [Float32(compute_μ_at_energy(get_material(m), μ_effective_energy_keV)) for m in i_materials]

    # Insert angular positions (evenly spaced, starting at 0°)
    n_inserts = 7
    angles_ca = [2π * i / n_inserts for i in 0:(n_inserts-1)]
    angles_i = [2π * i / n_inserts + π/n_inserts for i in 0:(n_inserts-1)]  # Offset by half spacing

    # Fill phantom voxel by voxel
    for k in 1:n_z
        for j in 1:n_voxels
            for i in 1:n_voxels
                xi = x[i]
                yj = y[j]
                r = sqrt(xi^2 + yj^2)

                # Default: background (air)
                μ[i, j, k] = μ_air
                mask[i, j, k] = UInt8(REGION_BACKGROUND)

                # Check if inside body cylinder
                if r <= body_radius
                    # Default body is solid water
                    μ[i, j, k] = μ_solid_water
                    mask[i, j, k] = UInt8(REGION_SOLID_WATER)

                    # Check calcium inserts (inner ring)
                    for (idx, angle) in enumerate(angles_ca)
                        cx = inner_ring_radius * cos(angle)
                        cy = inner_ring_radius * sin(angle)
                        dist = sqrt((xi - cx)^2 + (yj - cy)^2)
                        if dist <= rod_radius
                            μ[i, j, k] = ca_μ[idx]
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
                                μ[i, j, k] = i_μ[idx]
                                mask[i, j, k] = UInt8(i_labels[idx])
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    return Phantom(
        μ,
        mask,
        (dx, dy, dz),
        (-fov_cm/2 + dx/2, -fov_cm/2 + dy/2, -z_cm/2 + dz/2),
        (fov_cm, fov_cm, z_cm)
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

# Arguments
- `labeled_array::AbstractArray{<:Integer, 3}`: Integer array where each voxel
  contains a region label (0-255 supported via UInt8 conversion)
- `materials::Dict{Int, <:Any}`: Mapping from label values to materials. Materials
  can be:
  - `XA.Material`: Direct XrayAttenuation.jl material
  - `Symbol`: Material name to look up in MATERIALS_REGISTRY (e.g., `:water`, `:Ca_100`)
- `voxel_size_cm::NTuple{3, Real}`: Physical voxel dimensions in cm as (dx, dy, dz)

# Keyword Arguments
- `energy_keV::Real=60.0`: Energy for computing μ values in the Phantom (keV)
- `origin::Union{Nothing, NTuple{3, Real}}=nothing`: Origin coordinates (cm).
  If `nothing`, phantom is centered at isocenter.

# Returns
A `Phantom` struct with:
- `μ`: Linear attenuation coefficients (cm⁻¹) at specified energy
- `mask`: UInt8 mask with original label values
- `voxel_size`: Physical voxel dimensions (cm)
- `origin`: Origin coordinates (cm)
- `fov`: Field of view (cm)

# Material Lookup
For polychromatic simulation, the returned `Phantom.mask` values are used as indices
into a materials vector. To use this phantom with `simulate()`, you must also
provide a `materials` vector via the `materials` keyword argument (see IMPL-CUSTOM-MATERIALS).

The materials vector should be constructed such that `materials[mask_value + 1]`
returns the correct material for each voxel. Use `build_materials_vector()` to
create this vector from the materials dict.

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
    # ... more materials
)

# Create phantom (0.1 cm = 1mm voxels)
phantom = create_phantom_from_mask(
    xcat_mask,
    materials_dict,
    (0.1, 0.1, 0.1);
    energy_keV=70.0
)

# Build materials vector for simulation
materials_vec = build_materials_vector(materials_dict)

# Simulate with custom materials
result = simulate(phantom, scanner, protocol, sim_opts, recon_opts;
                  materials=materials_vec)
```

See also: [`build_materials_vector`](@ref), [`create_gammex_472`](@ref)
"""
function create_phantom_from_mask(
    labeled_array::AbstractArray{<:Integer, 3},
    materials::Dict{Int, M},
    voxel_size_cm::NTuple{3, Real};
    energy_keV::Real = 60.0,
    origin::Union{Nothing, NTuple{3, Real}} = nothing
) where M
    # Get dimensions
    nx, ny, nz = size(labeled_array)
    dx, dy, dz = Float64.(voxel_size_cm)

    # Compute FOV
    fov_x = dx * nx
    fov_y = dy * ny
    fov_z = dz * nz

    # Compute origin (center at isocenter if not specified)
    if origin === nothing
        origin_x = -fov_x/2 + dx/2
        origin_y = -fov_y/2 + dy/2
        origin_z = -fov_z/2 + dz/2
        computed_origin = (origin_x, origin_y, origin_z)
    else
        computed_origin = Float64.(origin)
    end

    # Convert labeled array to UInt8 mask
    # Note: labels > 255 will wrap around, but this is documented limitation
    mask = UInt8.(labeled_array)

    # Build μ array from materials
    μ = zeros(Float32, nx, ny, nz)

    # Pre-compute μ for each unique label
    unique_labels = unique(labeled_array)
    μ_lookup = Dict{Int, Float32}()

    for label in unique_labels
        if !haskey(materials, label)
            @warn "Label $label not found in materials dict, using air"
            mat = XA.Materials.air
        else
            mat = materials[label]
            # Handle Symbol lookup
            if mat isa Symbol
                mat = get_material(mat)
            end
        end
        μ_lookup[label] = Float32(compute_μ_at_energy(mat, Float64(energy_keV)))
    end

    # Fill μ array
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                label = labeled_array[i, j, k]
                μ[i, j, k] = μ_lookup[label]
            end
        end
    end

    return Phantom(
        μ,
        mask,
        (dx, dy, dz),
        computed_origin,
        (fov_x, fov_y, fov_z)
    )
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


# =============================================================================
# Exports
# =============================================================================

export RegionLabel
export REGION_BACKGROUND, REGION_AIR, REGION_WATER, REGION_SOLID_WATER
export REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600
export REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0
export REGION_TO_MATERIAL

export Phantom, create_gammex_472, create_phantom_from_mask, build_materials_vector
export get_region_mask
