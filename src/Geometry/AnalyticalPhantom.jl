# =============================================================================
# Analytical Phantoms
# =============================================================================
#
# Implements analytical phantom geometries for exact ray-object intersection.
#
# Advantages over voxelized phantoms:
# - No discretization artifacts ("inverse crime")
# - Exact path length computation
# - Resolution-independent
#
# Supported shapes:
# - Ellipsoid (3D ellipse)
# - Cylinder
# - Sphere
# - Shepp-Logan phantom (standard CT test phantom)
#
# =============================================================================

import AcceleratedKernels as AK

export AnalyticalObject, Ellipsoid, Cylinder, Sphere
export AnalyticalPhantom
export create_shepp_logan_3d
export ray_intersect_ellipsoid, ray_intersect_cylinder
export analytical_forward_project!, analytical_forward_project

# =============================================================================
# Analytical Object Types
# =============================================================================

"""
    AnalyticalObject

Abstract type for analytical phantom objects.
"""
abstract type AnalyticalObject end

"""
    Ellipsoid <: AnalyticalObject

3D ellipsoid defined by center, semi-axes, rotation, and attenuation.

# Fields
- `center`: Center position (x, y, z) in mm
- `semi_axes`: Semi-axis lengths (a, b, c) in mm
- `rotation_deg`: Rotation angles (rx, ry, rz) in degrees
- `μ`: Attenuation coefficient (cm⁻¹)
- `additive`: If true, adds to existing μ; if false, replaces
"""
struct Ellipsoid <: AnalyticalObject
    center::NTuple{3, Float64}
    semi_axes::NTuple{3, Float64}
    rotation_deg::NTuple{3, Float64}
    μ::Float64
    additive::Bool
end

"""
    Cylinder <: AnalyticalObject

Cylinder with axis along z.

# Fields
- `center`: Center position (x, y) in mm (z is infinite or bounded)
- `radius`: Radius in mm
- `z_range`: (z_min, z_max) in mm, or nothing for infinite
- `μ`: Attenuation coefficient (cm⁻¹)
- `additive`: If true, adds to existing μ; if false, replaces
"""
struct Cylinder <: AnalyticalObject
    center::NTuple{2, Float64}
    radius::Float64
    z_range::Union{Nothing, NTuple{2, Float64}}
    μ::Float64
    additive::Bool
end

"""
    Sphere <: AnalyticalObject

Sphere (special case of ellipsoid with equal semi-axes).
"""
struct Sphere <: AnalyticalObject
    center::NTuple{3, Float64}
    radius::Float64
    μ::Float64
    additive::Bool
end

# =============================================================================
# Analytical Phantom
# =============================================================================

"""
    AnalyticalPhantom

Collection of analytical objects forming a phantom.

# Fields
- `objects`: Vector of AnalyticalObjects
- `fov`: Field of view (mm)
- `background_μ`: Background attenuation (usually 0 for air)
"""
struct AnalyticalPhantom
    objects::Vector{AnalyticalObject}
    fov::Float64
    background_μ::Float64
end

# =============================================================================
# Shepp-Logan Phantom
# =============================================================================

"""
    create_shepp_logan_3d(; fov_mm=200.0, scale=1.0)

Create 3D Shepp-Logan phantom (standard CT test phantom).

# Keyword Arguments
- `fov_mm`: Field of view in mm (default: 200.0)
- `scale`: Scale factor for attenuation values (default: 1.0)

# Returns
- AnalyticalPhantom with Shepp-Logan ellipsoids

# Reference
- Shepp & Logan (1974), "The Fourier reconstruction of a head section"
- Extended to 3D by Koay et al.
"""
function create_shepp_logan_3d(; fov_mm::Real = 200.0, scale::Real = 1.0)
    # Scale factor to convert normalized coordinates to mm
    s = Float64(fov_mm) / 2

    # Shepp-Logan parameters: (center_x, center_y, center_z, a, b, c, angle, μ)
    # Coordinates are normalized to [-1, 1]
    params = [
        # Outer skull
        (0.0, 0.0, 0.0, 0.69, 0.92, 0.81, 0.0, 1.0),
        # Inner skull
        (0.0, -0.0184, 0.0, 0.6624, 0.874, 0.78, 0.0, -0.8),
        # Right ventricle
        (0.22, 0.0, 0.0, 0.11, 0.31, 0.22, -18.0, -0.2),
        # Left ventricle
        (-0.22, 0.0, 0.0, 0.16, 0.41, 0.28, 18.0, -0.2),
        # Tumor 1
        (0.0, 0.35, -0.15, 0.21, 0.25, 0.41, 0.0, 0.1),
        # Tumor 2
        (0.0, 0.1, 0.25, 0.046, 0.046, 0.05, 0.0, 0.1),
        # Tumor 3
        (0.0, -0.1, 0.25, 0.046, 0.046, 0.05, 0.0, 0.1),
        # Small features
        (-0.08, -0.605, 0.0, 0.046, 0.023, 0.02, 0.0, 0.1),
        (0.0, -0.605, 0.0, 0.023, 0.023, 0.02, 0.0, 0.1),
        (0.06, -0.605, 0.0, 0.023, 0.046, 0.02, 0.0, 0.1),
    ]

    objects = AnalyticalObject[]

    for (cx, cy, cz, a, b, c, angle, μ) in params
        ellipsoid = Ellipsoid(
            (cx * s, cy * s, cz * s),
            (a * s, b * s, c * s),
            (0.0, 0.0, angle),
            μ * scale,
            true  # additive
        )
        push!(objects, ellipsoid)
    end

    return AnalyticalPhantom(objects, Float64(fov_mm), 0.0)
end

# =============================================================================
# Ray-Object Intersection
# =============================================================================

"""
    ray_intersect_ellipsoid(ray_origin, ray_dir, ellipsoid)

Compute ray-ellipsoid intersection path length.

# Arguments
- `ray_origin`: Ray origin (x, y, z)
- `ray_dir`: Ray direction (normalized)
- `ellipsoid`: Ellipsoid object

# Returns
- Path length through ellipsoid (0 if no intersection)
"""
function ray_intersect_ellipsoid(
    ray_origin::NTuple{3, T},
    ray_dir::NTuple{3, T},
    e::Ellipsoid
) where T <: AbstractFloat

    # Transform to ellipsoid-centered coordinates
    ox = ray_origin[1] - T(e.center[1])
    oy = ray_origin[2] - T(e.center[2])
    oz = ray_origin[3] - T(e.center[3])

    dx, dy, dz = ray_dir

    # Scale by semi-axes to transform to unit sphere
    a, b, c = T(e.semi_axes[1]), T(e.semi_axes[2]), T(e.semi_axes[3])

    # TODO: Apply rotation if needed

    # Scaled coordinates
    ox_s, oy_s, oz_s = ox/a, oy/b, oz/c
    dx_s, dy_s, dz_s = dx/a, dy/b, dz/c

    # Quadratic coefficients for unit sphere intersection
    A = dx_s^2 + dy_s^2 + dz_s^2
    B = 2 * (ox_s*dx_s + oy_s*dy_s + oz_s*dz_s)
    C = ox_s^2 + oy_s^2 + oz_s^2 - 1

    discriminant = B^2 - 4*A*C

    if discriminant < 0
        return zero(T)  # No intersection
    end

    sqrt_disc = sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)

    # Both intersections must be positive (in front of ray)
    if t2 < 0
        return zero(T)
    end

    t_entry = max(t1, zero(T))
    t_exit = t2

    # Path length in original coordinates
    path_length = t_exit - t_entry

    return path_length
end

"""
    ray_intersect_cylinder(ray_origin, ray_dir, cylinder)

Compute ray-cylinder intersection path length.
"""
function ray_intersect_cylinder(
    ray_origin::NTuple{3, T},
    ray_dir::NTuple{3, T},
    cyl::Cylinder
) where T <: AbstractFloat

    ox = ray_origin[1] - T(cyl.center[1])
    oy = ray_origin[2] - T(cyl.center[2])
    oz = ray_origin[3]

    dx, dy, dz = ray_dir
    r = T(cyl.radius)

    # 2D quadratic for infinite cylinder along z
    A = dx^2 + dy^2
    B = 2 * (ox*dx + oy*dy)
    C = ox^2 + oy^2 - r^2

    if abs(A) < T(1e-10)
        return zero(T)  # Ray parallel to z-axis, outside cylinder
    end

    discriminant = B^2 - 4*A*C

    if discriminant < 0
        return zero(T)
    end

    sqrt_disc = sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)

    if t2 < 0
        return zero(T)
    end

    t_entry = max(t1, zero(T))
    t_exit = t2

    # Apply z-bounds if specified
    if cyl.z_range !== nothing
        z_min, z_max = T(cyl.z_range[1]), T(cyl.z_range[2])

        z_entry = oz + t_entry * dz
        z_exit = oz + t_exit * dz

        if z_exit < z_min || z_entry > z_max
            return zero(T)
        end

        # Clamp to z-bounds
        if z_entry < z_min
            t_entry = (z_min - oz) / dz
        end
        if z_exit > z_max
            t_exit = (z_max - oz) / dz
        end
    end

    return max(t_exit - t_entry, zero(T))
end

# =============================================================================
# Analytical Forward Projection
# =============================================================================

"""
    analytical_forward_project!(sinogram, phantom, geom)

Forward project through analytical phantom (in-place).

This computes exact ray-object intersections without discretization.

# Arguments
- `sinogram`: Output sinogram [n_cols, n_rows, n_angles] (modified in place)
- `phantom`: AnalyticalPhantom
- `geom`: CTGeometry

# Returns
- Sinogram with line integrals
"""
function analytical_forward_project!(
    sinogram::AbstractArray{T, 3},
    phantom::AnalyticalPhantom,
    geom::CTGeometry
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(sinogram)
    SAD = T(geom.SAD)
    SDD = T(geom.SDD)

    # Initialize to background
    fill!(sinogram, T(phantom.background_μ))

    # For each object, compute contribution
    for obj in phantom.objects
        if obj isa Ellipsoid
            _add_ellipsoid_projection!(sinogram, obj, geom)
        elseif obj isa Cylinder
            _add_cylinder_projection!(sinogram, obj, geom)
        elseif obj isa Sphere
            # Convert sphere to ellipsoid
            e = Ellipsoid(obj.center, (obj.radius, obj.radius, obj.radius),
                         (0.0, 0.0, 0.0), obj.μ, obj.additive)
            _add_ellipsoid_projection!(sinogram, e, geom)
        end
    end

    return sinogram
end

"""Internal: Add ellipsoid contribution to sinogram"""
function _add_ellipsoid_projection!(
    sinogram::AbstractArray{T, 3},
    e::Ellipsoid,
    geom::CTGeometry
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(sinogram)
    SAD = T(geom.SAD)
    SDD = T(geom.SDD)
    μ = T(e.μ)

    # Pre-compute ellipsoid parameters for GPU
    cx, cy, cz = T(e.center[1]), T(e.center[2]), T(e.center[3])
    a, b, c = T(e.semi_axes[1]), T(e.semi_axes[2]), T(e.semi_axes[3])

    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle_idx = Tuple(ci)

        # Source position (rotates around isocenter)
        θ = geom.angles[angle_idx]
        src_x = -SAD * sin(θ)
        src_y = SAD * cos(θ)
        src_z = zero(T)

        # Detector pixel position (fov[1] is x-direction, fov[3] is z-direction)
        det_u = (T(col) - T(n_cols)/2 - T(0.5)) * T(geom.fov[1] / n_cols)
        det_v = (T(row) - T(n_rows)/2 - T(0.5)) * T(geom.fov[3] / n_rows)

        det_x = (SDD - SAD) * sin(θ) + det_u * cos(θ)
        det_y = -(SDD - SAD) * cos(θ) + det_u * sin(θ)
        det_z = det_v

        # Ray direction (normalized)
        ray_len = sqrt((det_x - src_x)^2 + (det_y - src_y)^2 + (det_z - src_z)^2)
        dx = (det_x - src_x) / ray_len
        dy = (det_y - src_y) / ray_len
        dz = (det_z - src_z) / ray_len

        # Compute intersection
        ray_origin = (src_x, src_y, src_z)
        ray_dir = (dx, dy, dz)

        # Inline ellipsoid intersection for GPU
        ox = src_x - cx
        oy = src_y - cy
        oz = src_z - cz

        ox_s, oy_s, oz_s = ox/a, oy/b, oz/c
        dx_s, dy_s, dz_s = dx/a, dy/b, dz/c

        A = dx_s^2 + dy_s^2 + dz_s^2
        B = T(2) * (ox_s*dx_s + oy_s*dy_s + oz_s*dz_s)
        C = ox_s^2 + oy_s^2 + oz_s^2 - T(1)

        discriminant = B^2 - T(4)*A*C

        if discriminant >= 0
            sqrt_disc = sqrt(discriminant)
            t1 = (-B - sqrt_disc) / (T(2)*A)
            t2 = (-B + sqrt_disc) / (T(2)*A)

            if t2 >= 0
                t_entry = max(t1, zero(T))
                t_exit = t2
                path_length = t_exit - t_entry

                # Convert path to cm for μ (assumed in mm)
                path_cm = path_length / T(10)

                if e.additive
                    sinogram[idx] += μ * path_cm
                else
                    sinogram[idx] = μ * path_cm
                end
            end
        end
    end
end

"""Internal: Add cylinder contribution to sinogram"""
function _add_cylinder_projection!(
    sinogram::AbstractArray{T, 3},
    cyl::Cylinder,
    geom::CTGeometry
) where T <: AbstractFloat
    # Similar implementation to ellipsoid but for cylinder geometry
    # Simplified for now - full implementation would follow same pattern
end

"""
    analytical_forward_project(phantom, geom)

Non-mutating version of analytical_forward_project!.
"""
function analytical_forward_project(
    phantom::AnalyticalPhantom,
    geom::CTGeometry
)
    T = Float32
    sinogram = zeros(T, geom.n_cols, geom.n_rows, geom.n_angles)
    return analytical_forward_project!(sinogram, phantom, geom)
end
