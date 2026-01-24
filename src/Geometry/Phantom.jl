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

Digital phantom with semantic mask for validation.

# Fields
- `μ::Array{Float32,3}`: Linear attenuation coefficients (cm⁻¹) at effective energy
- `mask::Array{UInt8,3}`: Region labels (see `RegionLabel` enum)
- `voxel_size::NTuple{3,Float64}`: Voxel dimensions (cm) as (dx, dy, dz)
- `origin::NTuple{3,Float64}`: Origin coordinates (cm) - center of first voxel
- `fov::NTuple{3,Float64}`: Field of view (cm) as (x, y, z)

# Coordinate System
- X: left-right (increasing right)
- Y: anterior-posterior (increasing posterior)
- Z: inferior-superior (increasing superior)
- Origin at isocenter (0, 0, 0)

# Usage
```julia
phantom = create_gammex_472(; n_voxels=64)
mean_μ = mean(phantom.μ[phantom.mask .== UInt8(REGION_CA_100)])
```
"""
struct Phantom
    μ::Array{Float32,3}
    mask::Array{UInt8,3}
    voxel_size::NTuple{3,Float64}
    origin::NTuple{3,Float64}
    fov::NTuple{3,Float64}
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

export Phantom, create_gammex_472
export get_region_mask
