# =============================================================================
# Patients for scatter: cylinders of tissue, and what is put in them
# =============================================================================
# The frame is the arc's (arc.jl): the focal spot at the origin, the central ray along +y, the
# patient's axis along z. To the engine a phantom is what a crystal is to a detector, the volume
# whose escapes are logged: every photon that leaves the patient is recorded with where it left,
# where it was going and its energy. Reaching a scanner's detector from there is straight-line
# geometry, done in Julia (scatter.jl), for any scanner, from the same photons.
# =============================================================================

"""
    PhantomInsert(; material, x_mm = 0.0, y_mm = 0.0, semi_axes_mm)

A rod along the patient's axis inside a [`CylinderPhantom`](@ref): an elliptical cylinder of
`material` with semi-axes `semi_axes_mm` (across x, along y; equal for a circular rod), centred at
(`x_mm`, `y_mm`) relative to the phantom's centre, as long as the phantom.
"""
Base.@kwdef struct PhantomInsert
    material::DetectorMaterial
    x_mm::Float64 = 0.0
    y_mm::Float64 = 0.0
    semi_axes_mm::Tuple{Float64, Float64}
end

"""
    CylinderPhantom(; material = DetectorMaterial(:Water), semi_axes_mm, length_mm = 800.0,
                      centre_mm = (0.0, 0.0), inserts = PhantomInsert[])
    CylinderPhantom(diameter_mm; kw...)

A patient for scatter: an elliptical cylinder of `material` with semi-axes `semi_axes_mm` (x, y),
`length_mm` long along z (the patient's axis), centred at `centre_mm` (x, y), with `inserts` inside.
The second form is the circular cylinder of `diameter_mm`.
"""
Base.@kwdef struct CylinderPhantom <: AbstractDetector
    material::DetectorMaterial = DetectorMaterial(:Water)
    semi_axes_mm::Tuple{Float64, Float64}
    length_mm::Float64 = 800.0
    centre_mm::Tuple{Float64, Float64} = (0.0, 0.0)
    inserts::Vector{PhantomInsert} = PhantomInsert[]
end

CylinderPhantom(diameter_mm::Real; kw...) =
    CylinderPhantom(; semi_axes_mm = (diameter_mm / 2, diameter_mm / 2), kw...)

geometry_key(i::PhantomInsert) = (i.material.name, i.x_mm, i.y_mm, i.semi_axes_mm)
geometry_key(p::CylinderPhantom) =
    (
    p.material.name,
    p.semi_axes_mm,
    p.length_mm,
    p.centre_mm,
    map(geometry_key, p.inserts),
)

Base.:(==)(a::CylinderPhantom, b::CylinderPhantom) = geometry_key(a) == geometry_key(b)
same_geometry(a::CylinderPhantom, b::CylinderPhantom) = a == b

world_half_mm(p::CylinderPhantom) =
    max(hypot(p.centre_mm...) + maximum(p.semi_axes_mm), p.length_mm / 2) + 50

"""An elliptical (or circular) cylinder along z, `length_mm` long."""
function elliptical_tube(name, semi_axes_mm, length_mm)
    a, b = semi_axes_mm
    return a == b ?
        G4Tubs(name, 0, a * mm, length_mm * mm / 2, 0, 2π) :
        G4EllipticalTube(name, a * mm, b * mm, length_mm * mm / 2)
end

function build_detector!(world, p::CylinderPhantom)
    body = G4LogicalVolume(
        elliptical_tube("Phantom", p.semi_axes_mm, p.length_mm),
        g4_material(p.material),
        "Phantom",
    )
    G4PVPlacement(
        nothing,
        G4ThreeVector(p.centre_mm[1] * mm, p.centre_mm[2] * mm, 0),
        body,
        "Phantom",
        world,
        false,
        CRYSTAL_COPY,
    )

    # Inserts are part of the patient too (the same copy number): crossing into one is no escape
    for (k, insert) in enumerate(p.inserts)
        rod = G4LogicalVolume(
            elliptical_tube("Insert$k", insert.semi_axes_mm, p.length_mm),
            g4_material(insert.material),
            "Insert$k",
        )
        G4PVPlacement(
            nothing,
            G4ThreeVector(insert.x_mm * mm, insert.y_mm * mm, 0),
            rod,
            "Insert$k",
            body,
            false,
            CRYSTAL_COPY,
        )
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Rays through a phantom: lengths per material, and the water-equivalent diameter
# -----------------------------------------------------------------------------

"""Length (mm) of the ray from `origin` along `direction` (the x–y plane) inside an ellipse."""
function ellipse_chord(origin, direction, centre, semi_axes)
    d = direction ./ hypot(direction[1], direction[2])
    ox, oy = (origin[1] - centre[1]) / semi_axes[1], (origin[2] - centre[2]) / semi_axes[2]
    ux, uy = d[1] / semi_axes[1], d[2] / semi_axes[2]
    a = ux^2 + uy^2
    b = 2 * (ox * ux + oy * uy)
    c = ox^2 + oy^2 - 1
    disc = b^2 - 4a * c
    disc <= 0 && return 0.0
    t1, t2 = (-b - sqrt(disc)) / (2a), (-b + sqrt(disc)) / (2a)
    return max(0.0, t2) - max(0.0, t1)
end

"""
    material_lengths(phantom, origin, direction) -> Vector of (material, length_mm)

How far the ray from `origin` along `direction` (in the x–y plane) travels through each material of
`phantom`: the body, less the inserts it crosses, and each insert.
"""
function material_lengths(p::CylinderPhantom, origin, direction)
    lengths = Tuple{DetectorMaterial, Float64}[]
    body = ellipse_chord(origin, direction, p.centre_mm, p.semi_axes_mm)
    for insert in p.inserts
        L = ellipse_chord(origin, direction, p.centre_mm .+ (insert.x_mm, insert.y_mm), insert.semi_axes_mm)
        L > 0 || continue
        push!(lengths, (insert.material, L))
        body -= L
    end
    pushfirst!(lengths, (p.material, max(body, 0.0)))
    return lengths
end

"""
    water_equivalent_length(phantom, origin, direction; energy_keV = 70.0) -> mm

The ray's water-equivalent length: each material's length times its attenuation over water's at
`energy_keV`.
"""
function water_equivalent_length(p::CylinderPhantom, origin, direction; energy_keV::Real = 70.0)
    μ_water = linear_attenuation(DetectorMaterial(:Water), energy_keV)
    return sum(L * linear_attenuation(m, energy_keV) / μ_water for (m, L) in material_lengths(p, origin, direction))
end

"""
    water_equivalent_diameter(phantom; origin = (0.0, 0.0), fan_deg = 60.0, rays = 1201,
                              energy_keV = 70.0) -> mm

The phantom's water-equivalent diameter as AAPM Report 220 defines it from a projection: the
water-equivalent area `A_w`, the sum over rays of water-equivalent length times ray spacing at the
phantom's centre, and `2 √(A_w / π)`. The rays fan out from `origin` over `fan_deg` around the
direction to the phantom's centre.
"""
function water_equivalent_diameter(
        p::CylinderPhantom;
        origin = (0.0, 0.0),
        fan_deg::Real = 60.0,
        rays::Integer = 1201,
        energy_keV::Real = 70.0,
    )
    to_centre = p.centre_mm .- origin
    distance = hypot(to_centre...)
    θ0 = atan(to_centre[1], to_centre[2])
    angles = range(-deg2rad(fan_deg) / 2, deg2rad(fan_deg) / 2; length = rays)
    spacing = distance * step(angles)
    area = sum(
        water_equivalent_length(p, origin, (sin(θ0 + γ), cos(θ0 + γ)); energy_keV) * spacing for γ in angles
    )
    return 2 * sqrt(area / π)
end
