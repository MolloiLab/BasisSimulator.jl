"""
    src/object/phantom.jl

Phantom struct + factories for the scanned object (mask + materials +
voxel size + origin).  Every phantom carries a semantic region mask
that identifies voxel category (air / water / Gammex insert / …) so
downstream ROI analysis can compute per-region statistics without
re-segmenting the reconstructed volume.

Two built-in factories:
  * `create_gammex_472`   — canonical 16-rod Gammex 472 phantom
    (7 calcium + 7 iodine inserts in a solid-water background).
  * `create_phantom_from_mask` — thin wrapper to wrap a user-supplied
    labeled array + materials dict into a `Phantom`.

μ at any energy is computed on demand via `compute_μ(phantom, E)` /
`create_μ_volume!` to keep memory bounded — the phantom only stores
the mask, not a pre-computed μ volume.

BasisSim-original — no upstream port.
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
- `mask::AbstractArray{<:Unsigned,3}`: Region labels (UInt8 for ≤255, UInt16 for >255)
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

# Simulate via workspace API
ws = create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)

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
struct Phantom{T <: Unsigned, M <: AbstractArray{T, 3}, Mat}
    mask::M
    materials::Mat  # Vector{XA.Material}
    voxel_size::NTuple{3, Float64}
    origin::NTuple{3, Float64}
    extent::NTuple{3, Float64}
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
    μ_lookup = Float32[
        compute_μ_at_energy(mat, Float64(energy_keV))
            for mat in phantom.materials
    ]
    # Broadcast via mask indexing (mask is 0-based, vector is 1-based)
    return μ_lookup[phantom.mask .+ 1]
end

# =============================================================================
# Unified Phantom Constructor (v20.0)
# =============================================================================

"""
    Phantom(labeled_array, materials_dict, voxel_size_cm; kwargs...) -> Phantom

Create a Phantom from a labeled array with materials stored internally.

This is the **unified API**: the returned Phantom contains everything needed
for polychromatic simulation via the workspace-based `simulate!()` pipeline.

**No energy_keV parameter needed!** The μ field was removed in v20.0-pivot. Use
`compute_μ(phantom, energy_keV)` to get attenuation coefficients at any energy.

# Arguments
- `labeled_array::AbstractArray{<:Integer, 3}`: Integer array where each voxel
  contains a region label. Auto-promotes to UInt16 when >255 labels.
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
- `mask`: UInt8 mask (≤255 labels) or UInt16 mask (>255 labels)
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

# Simulate via workspace API
ws = create_eict_workspace(scanner, protocol, SimOptions(), ReconOptions(), phantom)
simulate!(ws, phantom, scanner, protocol)
```

See also: [`compute_μ`](@ref), [`create_phantom_from_mask`](@ref), [`create_gammex_472`](@ref)
"""
function Phantom(
        labeled_array::AbstractArray{<:Integer, 3},
        materials_dict::Dict{Int, M},
        voxel_size_cm::NTuple{3, Real};
        origin::Union{Nothing, NTuple{3, Real}} = nothing
    ) where {M}
    # Get dimensions
    nx, ny, nz = size(labeled_array)
    dx, dy, dz = Float64.(voxel_size_cm)

    # Compute physical extent
    ext_x = dx * nx
    ext_y = dy * ny
    ext_z = dz * nz

    # Compute origin (center at isocenter if not specified)
    if origin === nothing
        origin_x = -ext_x / 2 + dx / 2
        origin_y = -ext_y / 2 + dy / 2
        origin_z = -ext_z / 2 + dz / 2
        computed_origin = (origin_x, origin_y, origin_z)
    else
        computed_origin = Float64.(origin)
    end

    # Convert labeled array to smallest unsigned type that fits.  Reuse the array
    # as-is when it already has that type: `UInt8.(x)` on a UInt8 GPU array is a
    # pointless copy, and on a device it is a KERNEL whose result the caller never
    # synchronizes.  Handing back that unsynchronized buffer is a real race — a
    # phantom built in one Pluto cell then projected from another intermittently
    # reads all zeros, and the reconstruction comes out empty.  `maximum` above
    # already forces a device→host readback, so `labeled_array` itself is settled.
    max_label = maximum(labeled_array)
    U = max_label > typemax(UInt8) ? UInt16 : UInt8
    mask = eltype(labeled_array) === U ? labeled_array : U.(labeled_array)

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

# Forward project
sino = forward_project(phantom.mask, geom; energies=energies, weights=weights, materials=phantom.materials)
```
"""
function create_gammex_472(;
        n_voxels::Int = 64,
        n_slices::Union{Int, Nothing} = nothing,
        fov_cm::Float64 = 35.0,
        z_cm::Float64 = 4.0
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
    x = range(-fov_cm / 2 + dx / 2, fov_cm / 2 - dx / 2, length = n_voxels)
    y = range(-fov_cm / 2 + dy / 2, fov_cm / 2 - dy / 2, length = n_voxels)

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
    angles_ca = [2π * i / n_inserts for i in 0:(n_inserts - 1)]
    angles_i = [2π * i / n_inserts + π / n_inserts for i in 0:(n_inserts - 1)]  # Offset by half spacing

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
        (-fov_cm / 2 + dx / 2, -fov_cm / 2 + dy / 2, -z_cm / 2 + dz / 2),
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
  contains a region label. Auto-promotes to UInt16 when >255 labels.
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
- `mask`: UInt8 mask (≤255 labels) or UInt16 mask (>255 labels)
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

# Simulate via workspace API
ws = create_eict_workspace(scanner, protocol, SimOptions(), ReconOptions(), phantom)
simulate!(ws, phantom, scanner, protocol)
```

See also: [`Phantom`](@ref), [`compute_μ`](@ref), [`create_gammex_472`](@ref)
"""
function create_phantom_from_mask(
        labeled_array::AbstractArray{<:Integer, 3},
        materials::Dict{Int, M},
        voxel_size_cm::NTuple{3, Real};
        origin::Union{Nothing, NTuple{3, Real}} = nothing
    ) where {M}
    # Delegate to the unified Phantom constructor
    return Phantom(labeled_array, materials, voxel_size_cm; origin = origin)
end

# Internal helper used by the `Phantom` ctor + `create_gammex_472` to build
# an air-padded `Vector{XA.Material}` indexed by `label + 1`.  Not exported
# — callers that need it should use the high-level `Phantom` ctor.
function build_materials_vector(materials_dict::Dict{Int, M}) where {M}
    max_label = maximum(keys(materials_dict))
    materials_vec = Vector{XA.Material}(undef, max_label + 1)
    for i in 1:(max_label + 1)
        materials_vec[i] = XA.Materials.air
    end
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
# Exports
# =============================================================================
#
# `REGION_TO_MATERIAL` is kept module-internal (no export) — notebooks use
# `get_material(:Ca_100)` directly instead of round-tripping through the
# enum table.

export RegionLabel
export REGION_BACKGROUND, REGION_AIR, REGION_WATER, REGION_SOLID_WATER
export REGION_CA_50, REGION_CA_100, REGION_CA_200, REGION_CA_300, REGION_CA_400, REGION_CA_500, REGION_CA_600
export REGION_I_2_0, REGION_I_2_5, REGION_I_5_0, REGION_I_7_5, REGION_I_10_0, REGION_I_15_0, REGION_I_20_0

export Phantom, compute_μ, create_gammex_472, create_phantom_from_mask
