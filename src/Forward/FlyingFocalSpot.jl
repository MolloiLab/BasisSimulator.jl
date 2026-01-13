"""
    Forward/FlyingFocalSpot.jl

Flying focal spot (FFS) modeling for CT simulation.

A flying focal spot is a technique where the X-ray focal spot position is
electromagnetically deflected between views to improve spatial sampling.

Benefits:
- Doubles effective in-plane sampling (2-position FFS)
- Reduces aliasing artifacts
- Improves spatial resolution without smaller detector pixels
- Can also be used in z-direction for helical (z-FFS)

Implementation:
The source position is shifted slightly between views according to the FFS pattern.
This effectively interleaves two (or more) sets of rays per gantry angle.

Common configurations:
- 2-position in-plane FFS: alternates ±Δu (most common)
- 2-position z-FFS: alternates ±Δz
- 4-position FFS: combines both in-plane and z deflection
"""

# =============================================================================
# Flying Focal Spot Types
# =============================================================================

"""
    FlyingFocalSpotModel

Flying focal spot configuration.

# Fields
- `enabled`: Whether FFS is active
- `deflection_u`: In-plane (u) deflection magnitude in cm
- `deflection_z`: Z-axis deflection magnitude in cm
- `pattern`: Deflection pattern (:in_plane, :z_axis, :combined)
- `n_positions`: Number of focal spot positions (2 or 4)
"""
struct FlyingFocalSpotModel
    enabled::Bool
    deflection_u::Float64
    deflection_z::Float64
    pattern::Symbol
    n_positions::Int
end

# =============================================================================
# Pre-defined FFS Models
# =============================================================================

"""
    ffs_none()

No flying focal spot (standard fixed focal spot).
"""
ffs_none() = FlyingFocalSpotModel(false, 0.0, 0.0, :none, 1)

"""
    ffs_in_plane(deflection_cm=0.03)

2-position in-plane flying focal spot.

Alternates the focal spot position in the in-plane (u) direction.
This is the most common FFS configuration.

# Arguments
- `deflection_cm`: Half the total deflection distance (default: 0.3mm = 0.03cm)

# Typical Values
- Quarter detector pitch: gives optimal sampling improvement
- GE: ~0.25mm, Siemens: ~0.3mm, Canon: ~0.2mm
"""
ffs_in_plane(deflection_cm::Float64=0.03) = FlyingFocalSpotModel(true, deflection_cm, 0.0, :in_plane, 2)

"""
    ffs_z_axis(deflection_cm=0.03)

2-position z-axis flying focal spot.

Alternates the focal spot position in the z (slice) direction.
Used in helical scanning to improve z-resolution.

# Arguments
- `deflection_cm`: Half the total deflection distance (default: 0.3mm)
"""
ffs_z_axis(deflection_cm::Float64=0.03) = FlyingFocalSpotModel(true, 0.0, deflection_cm, :z_axis, 2)

"""
    ffs_combined(deflection_u_cm=0.03, deflection_z_cm=0.03)

4-position combined flying focal spot.

Combines in-plane and z-axis deflection for maximum sampling improvement.
Pattern: (+u,+z), (-u,-z), (+u,-z), (-u,+z) or similar.

# Arguments
- `deflection_u_cm`: In-plane deflection (default: 0.3mm)
- `deflection_z_cm`: Z-axis deflection (default: 0.3mm)
"""
function ffs_combined(deflection_u_cm::Float64=0.03, deflection_z_cm::Float64=0.03)
    return FlyingFocalSpotModel(true, deflection_u_cm, deflection_z_cm, :combined, 4)
end

"""
    ffs_custom(deflection_u, deflection_z, n_positions)

Custom flying focal spot configuration.

# Arguments
- `deflection_u`: In-plane deflection in cm
- `deflection_z`: Z-axis deflection in cm
- `n_positions`: Number of positions (2 or 4)
"""
function ffs_custom(deflection_u::Float64, deflection_z::Float64, n_positions::Int)
    @assert n_positions in [2, 4] "n_positions must be 2 or 4"

    if deflection_u > 0 && deflection_z > 0
        pattern = :combined
    elseif deflection_u > 0
        pattern = :in_plane
    elseif deflection_z > 0
        pattern = :z_axis
    else
        pattern = :none
    end

    return FlyingFocalSpotModel(deflection_u > 0 || deflection_z > 0,
                                 deflection_u, deflection_z, pattern, n_positions)
end

# =============================================================================
# FFS Position Calculation
# =============================================================================

"""
    get_ffs_offset(model::FlyingFocalSpotModel, view_idx::Int) -> Tuple{Float64, Float64}

Get the focal spot offset for a given view index.

Returns (u_offset, z_offset) in cm.
"""
function get_ffs_offset(model::FlyingFocalSpotModel, view_idx::Int)
    if !model.enabled
        return (0.0, 0.0)
    end

    # Determine position in cycle
    pos = mod(view_idx - 1, model.n_positions)

    if model.n_positions == 2
        # 2-position: alternates between +/- deflection
        sign = pos == 0 ? 1.0 : -1.0
        return (sign * model.deflection_u, sign * model.deflection_z)
    else  # 4-position
        # Pattern: (++), (--), (+-), (-+)
        u_sign = (pos == 0 || pos == 2) ? 1.0 : -1.0
        z_sign = (pos == 0 || pos == 3) ? 1.0 : -1.0
        return (u_sign * model.deflection_u, z_sign * model.deflection_z)
    end
end

"""
    get_ffs_offsets(model::FlyingFocalSpotModel, n_views::Int) -> Tuple{Vector, Vector}

Get all focal spot offsets for a sequence of views.

Returns (u_offsets, z_offsets) vectors in cm.
"""
function get_ffs_offsets(model::FlyingFocalSpotModel, n_views::Int)
    u_offsets = zeros(Float64, n_views)
    z_offsets = zeros(Float64, n_views)

    for i in 1:n_views
        u_offsets[i], z_offsets[i] = get_ffs_offset(model, i)
    end

    return (u_offsets, z_offsets)
end

"""
    apply_ffs_to_geometry!(geom::CTGeometry, model::FlyingFocalSpotModel)

Modify geometry source positions to include FFS deflection.

This modifies the geometry in-place, shifting source positions according
to the FFS pattern.

# Warning
This permanently modifies the geometry. Create a copy if you need the original.
"""
function apply_ffs_to_geometry!(geom, model::FlyingFocalSpotModel)
    if !model.enabled
        return geom
    end

    for i in 1:geom.n_angles
        u_off, z_off = get_ffs_offset(model, i)

        # Get the in-plane direction (perpendicular to source-detector axis)
        # For circular geometry, this is approximately the detector u-axis
        ux = geom.detector_u[1, i]
        uy = geom.detector_u[2, i]
        uz = geom.detector_u[3, i]

        # Apply in-plane offset
        geom.source_positions[1, i] += u_off * ux
        geom.source_positions[2, i] += u_off * uy
        geom.source_positions[3, i] += u_off * uz

        # Apply z-axis offset
        geom.source_positions[3, i] += z_off
    end

    return geom
end

"""
    create_geometry_with_ffs(base_geom, model::FlyingFocalSpotModel) -> CTGeometry

Create a new geometry with FFS offsets applied.

This returns a copy of the geometry with modified source positions.
"""
function create_geometry_with_ffs(base_geom, model::FlyingFocalSpotModel)
    if !model.enabled
        return base_geom
    end

    # Create deep copy of geometry
    new_source_pos = copy(base_geom.source_positions)
    new_det_centers = copy(base_geom.detector_centers)
    new_det_u = copy(base_geom.detector_u)
    new_det_v = copy(base_geom.detector_v)
    new_angles = copy(base_geom.angles)

    # Apply FFS offsets
    for i in 1:base_geom.n_angles
        u_off, z_off = get_ffs_offset(model, i)

        ux = base_geom.detector_u[1, i]
        uy = base_geom.detector_u[2, i]
        uz = base_geom.detector_u[3, i]

        new_source_pos[1, i] += u_off * ux
        new_source_pos[2, i] += u_off * uy
        new_source_pos[3, i] += u_off * uz
        new_source_pos[3, i] += z_off
    end

    # Return new geometry (assuming CTGeometry constructor exists)
    return CTGeometry(
        base_geom.SAD, base_geom.SDD,
        base_geom.n_angles, base_geom.n_rows, base_geom.n_cols,
        base_geom.pixel_size,
        new_angles, new_source_pos, new_det_centers, new_det_u, new_det_v
    )
end

"""
    get_ffs_info(model::FlyingFocalSpotModel) -> NamedTuple

Get diagnostic information about FFS model.
"""
function get_ffs_info(model::FlyingFocalSpotModel)
    sampling_improvement = model.enabled ? model.n_positions : 1

    return (
        enabled = model.enabled,
        pattern = model.pattern,
        n_positions = model.n_positions,
        deflection_u_mm = model.deflection_u * 10,  # cm to mm
        deflection_z_mm = model.deflection_z * 10,
        sampling_improvement = sampling_improvement
    )
end

# =============================================================================
# Exports
# =============================================================================

export FlyingFocalSpotModel
export ffs_none, ffs_in_plane, ffs_z_axis, ffs_combined, ffs_custom
export get_ffs_offset, get_ffs_offsets
export apply_ffs_to_geometry!, create_geometry_with_ffs
export get_ffs_info
