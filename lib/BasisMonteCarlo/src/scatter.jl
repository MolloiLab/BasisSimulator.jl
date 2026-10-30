# =============================================================================
# Scatter: from the photons leaving a patient to a scanner's detector
# =============================================================================
# Geant4 follows every photon through the patient (a CylinderPhantom) and logs each one that leaves
# it: where, in which direction, with what energy, and whether it is the primary photon unscattered
# (`:transmitted`) or a scattered or secondary one. From there to the detector a photon travels in a
# straight line through air, so any scanner's detector can be reached in Julia, from the same
# photons: intersect the line with the scanner's detector arc (radius = source-to-detector distance,
# centred on the focal spot), let the anti-scatter grid accept or reject it by its angle, and weigh
# it by the detector's efficiency at its energy. One simulation serves every scanner.
#
# Two kinds of runs:
#   - a fan beam from the focal spot ([`FocalSpot`](@ref)) through the patient at the isocentre:
#     the scatter a scan sees, directly;
#   - one ray ([`RayBeam`](@ref)) through a patient centred at the origin: a pencil-beam scatter
#     kernel. The scanner's focal spot is put back on the ray's line, at the scanner's
#     source-to-isocentre distance from the patient's centre, so one kernel serves any scanner.
# =============================================================================

"""
    ScatterScanner(; name, source_to_isocenter_mm, source_to_detector_mm, fan_deg, z_half_mm,
                     column_bin_deg = 2.0, row_bin_mm = 20.0, grid = :none, grid_height_mm = 0.0,
                     grid_opening_mm = (Inf, Inf), efficiency = ones(150), mu_per_mm = zeros(150),
                     depth_mm = 0.0, energy_weighted = true)

How one scanner sees the photons leaving a patient:

- geometry: the source-to-isocentre and source-to-detector distances; the detector's fan
  (`fan_deg`, centred on the central ray) and its half-extent along z at the detector (`z_half_mm`);
- the tally's bins: `column_bin_deg` across the fan, `row_bin_mm` along z at the detector;
- the anti-scatter grid: `:none`, `:across` (1-D: plates between columns) or `:both` (2-D), its
  walls `grid_height_mm` tall with `grid_opening_mm` open between them (across, along z), all
  focused on the focal spot. A photon arriving at an angle α to the focused direction gets through
  with probability `max(0, 1 − H tan α / opening)` in each direction the grid has walls;
- the detector: its flat-field `efficiency` η(E) on a 1-keV grid from 1 keV (normal incidence), and
  the crystal's `mu_per_mm` and `depth_mm` to lengthen the path of photons arriving obliquely
  (`depth_mm = 0` leaves η as it is);
- `energy_weighted`: an energy-integrating detector's signal (each photon counts E·η) or a
  photon-counting one's (each counts η).
"""
Base.@kwdef struct ScatterScanner
    name::String
    source_to_isocenter_mm::Float64
    source_to_detector_mm::Float64
    fan_deg::Float64
    z_half_mm::Float64
    column_bin_deg::Float64 = 2.0
    row_bin_mm::Float64 = 20.0
    grid::Symbol = :none
    grid_height_mm::Float64 = 0.0
    grid_opening_mm::Tuple{Float64, Float64} = (Inf, Inf)
    efficiency::Vector{Float64} = ones(150)
    mu_per_mm::Vector{Float64} = zeros(150)
    depth_mm::Float64 = 0.0
    energy_weighted::Bool = true
end

"""Bin edges across the fan (degrees from the central ray) and along z (mm at the detector): an odd
number of bins each way, so the central ray and z = 0 fall in the middle of a bin."""
function column_edges(s::ScatterScanner)
    n = round(Int, s.fan_deg / s.column_bin_deg)
    n += iseven(n)
    return collect(range(-n * s.column_bin_deg / 2, n * s.column_bin_deg / 2; length = n + 1))
end

function row_edges(s::ScatterScanner)
    n = round(Int, 2 * s.z_half_mm / s.row_bin_mm)
    n += iseven(n)
    return collect(range(-n * s.row_bin_mm / 2, n * s.row_bin_mm / 2; length = n + 1))
end

"""
    ScatterReadout(scanners)

Passed to [`simulate_sweep`](@ref), it reduces each run on a [`CylinderPhantom`](@ref), where it
ran, to one [`ScatterTally`](@ref) per scanner.
"""
struct ScatterReadout
    scanners::Vector{ScatterScanner}
end

"""
    ScatterTally

What one scanner's detector recorded from a run (`n` photons, `photon_keV` their summed energy),
on its grid of bins (`column_edges_deg` across the fan from the central ray, `row_edges_mm` along z
at the detector): `scatter[column, row]` and `primary[column, row]`, the detected signal of
scattered and of unscattered photons (keV for an energy-integrating detector, photons for a
counting one), and `scatter_spectrum[k]`, the detected scattered photons by energy (1-keV bins).
`analog_scatter` is the scatter counted photon by photon: the same array as `scatter` from a
[`ScatterReadout`](@ref); from a [`ForcedScatterReadout`](@ref), an independent estimate of what
`scatter` estimates by forced detection, to check it against.
"""
struct ScatterTally
    scanner::String
    n::Int
    photon_keV::Float64
    column_edges_deg::Vector{Float64}
    row_edges_mm::Vector{Float64}
    scatter::Matrix{Float64}
    primary::Matrix{Float64}
    scatter_spectrum::Vector{Float64}
    analog_scatter::Matrix{Float64}
end

Base.:+(a::ScatterTally, b::ScatterTally) = ScatterTally(
    a.scanner,
    a.n + b.n,
    a.photon_keV + b.photon_keV,
    a.column_edges_deg,
    a.row_edges_mm,
    a.scatter .+ b.scatter,
    a.primary .+ b.primary,
    a.scatter_spectrum .+ b.scatter_spectrum,
    a.analog_scatter .+ b.analog_scatter,
)

"""
The focal spot and the central ray (unit vector) of `scanner` for a run: the beam's own origin for a
fan beam; for one ray, the point on its line at the source-to-isocentre distance from the patient's
centre, on the side it comes from.
"""
function focal_spot_of(beam::FocalSpot, phantom::CylinderPhantom, scanner::ScatterScanner)
    return (0.0, 0.0), (0.0, 1.0)
end

function focal_spot_of(beam::RayBeam, phantom::CylinderPhantom, scanner::ScatterScanner)
    o = (beam.origin_mm[1], beam.origin_mm[2])
    d = (beam.direction[1], beam.direction[2]) ./ hypot(beam.direction[1], beam.direction[2])
    c = phantom.centre_mm
    q = o .- c
    # |q + s d| = SID, the root behind the patient (smaller s)
    b2 = 2 * (q[1] * d[1] + q[2] * d[2])
    c0 = q[1]^2 + q[2]^2 - scanner.source_to_isocenter_mm^2
    s = (-b2 - sqrt(b2^2 - 4c0)) / 2
    f = o .+ s .* d
    central = (c .- f) ./ hypot((c .- f)...)
    return f, central
end

"""Probability of getting through one direction of a focused grid at angle `α` (rad)."""
grid_acceptance(α, height_mm, opening_mm) = isinf(opening_mm) ? 1.0 : max(0.0, 1 - height_mm * abs(tan(α)) / opening_mm)

"""
    scatter_tally(run, readout::ScatterReadout) -> Vector{ScatterTally}

Take every photon that left the run's [`CylinderPhantom`](@ref) to each scanner's detector.
"""
function scatter_tally(run::MonoenergeticRun, readout::ScatterReadout)
    phantom = run.detector
    phantom isa CylinderPhantom || throw(ArgumentError("scatter_tally needs a run on a CylinderPhantom"))
    esc = run.escapes
    drawn = run.photon_energy_keV
    photon_keV = drawn === nothing || isempty(drawn) ? run.n * run.energy_keV : sum(Float64, drawn)

    return [tally_for(s, run, phantom, photon_keV) for s in readout.scanners]
end

"""One scanner's [`ScatterTally`](@ref) of a run."""
function tally_for(
        s::ScatterScanner,
        run::MonoenergeticRun,
        phantom::CylinderPhantom,
        photon_keV,
    )
    esc = run.escapes
    f, c = focal_spot_of(run.beam, phantom, s)
    cols, rows = column_edges(s), row_edges(s)
    scatter = zeros(length(cols) - 1, length(rows) - 1)
    primary = zeros(length(cols) - 1, length(rows) - 1)
    spectrum = zeros(150)
    R = s.source_to_detector_mm
    for i in eachindex(esc.kind)
        kind = esc.kind[i]
        kind === :electron && continue
        p, d = esc.position_mm[i], esc.direction[i]

        # Where the straight line meets the detector arc (centred on the focal spot)
        q = (p[1] - f[1], p[2] - f[2])
        a = d[1]^2 + d[2]^2
        a > 1.0e-12 || continue
        b2 = 2 * (q[1] * d[1] + q[2] * d[2])
        c0 = q[1]^2 + q[2]^2 - R^2
        c0 < 0 || continue
        t = (-b2 + sqrt(b2^2 - 4a * c0)) / (2a)
        h = (q[1] + t * d[1], q[2] + t * d[2])
        z = p[3] + t * d[3]
        γ = rad2deg(atan(h[1] * c[2] - h[2] * c[1], c[1] * h[1] + c[2] * h[2]))   # positive toward +x
        ci = searchsortedlast(cols, γ)
        ri = searchsortedlast(rows, z)
        (1 <= ci < length(cols) && 1 <= ri < length(rows)) || continue

        # Through the grid, into the detector
        E = Float64(esc.energy_keV[i])
        gη, k = detection_weight(s, h ./ R, z, d, E)
        gη > 0 || continue
        w = gη * (s.energy_weighted ? E : 1.0)
        if kind === :transmitted
            primary[ci, ri] += w
        else
            scatter[ci, ri] += w
            spectrum[k] += gη
        end
    end
    return ScatterTally(s.name, run.n, photon_keV, cols, rows, scatter, primary, spectrum, scatter)
end

"""
    detection_weight(s, n_xy, z, d, E) -> (g η, k)

The chance a photon of `E` keV travelling along `d` gets through `s`'s grid and is detected where
it meets the detector arc at height `z`, the arc's direction from the focal spot there being the
unit vector `n_xy`; and the 1-keV bin `k` of its energy.
"""
function detection_weight(s::ScatterScanner, n_xy, z, d, E)
    R = s.source_to_detector_mm

    # The photon's angles to the grid's focused direction
    α_fan = atan(n_xy[1] * d[2] - n_xy[2] * d[1], n_xy[1] * d[1] + n_xy[2] * d[2])
    α_z = atan(d[3], hypot(d[1], d[2])) - atan(z, R)
    g = s.grid === :none ? 1.0 :
        s.grid === :across ? grid_acceptance(α_fan, s.grid_height_mm, s.grid_opening_mm[1]) :
        grid_acceptance(α_fan, s.grid_height_mm, s.grid_opening_mm[1]) *
        grid_acceptance(α_z, s.grid_height_mm, s.grid_opening_mm[2])
    k = clamp(round(Int, E), 1, length(s.efficiency))
    g > 0 || return (0.0, k)

    # The detector's efficiency at this energy, the path lengthened by the obliquity
    η = s.efficiency[k]
    if s.depth_mm > 0 && s.mu_per_mm[k] > 0
        n3 = (n_xy[1], n_xy[2], z / R) ./ sqrt(1 + (z / R)^2)
        cosθ = max(1.0e-3, abs(n3[1] * d[1] + n3[2] * d[2] + n3[3] * d[3]))
        μd = s.mu_per_mm[k] * s.depth_mm
        η *= (1 - exp(-μd / cosθ)) / (1 - exp(-μd))
    end
    return (g * η, k)
end

reduce_run(run, readout::ScatterReadout) = scatter_tally(run, readout)
needs_deposits(::ScatterReadout) = false

"""Add up per-scanner tallies from several runs (chunks of one simulation)."""
merge_tallies(runs) = [reduce(+, (r[k] for r in runs)) for k in eachindex(first(runs))]

# -----------------------------------------------------------------------------
# Pencil-beam kernels
# -----------------------------------------------------------------------------

"""
    kernel_ray(diameter_mm, impact_fraction) -> RayBeam

The ray through a water cylinder of `diameter_mm` centred at the origin, passing its axis at
`impact_fraction` of its radius, travelling along +y (starting 10 mm before the cylinder).
"""
function kernel_ray(diameter_mm::Real, impact_fraction::Real)
    radius = diameter_mm / 2
    return RayBeam(;
        origin_mm = (impact_fraction * radius, -(radius + 10.0), 0.0),
        direction = (0.0, 1.0, 0.0),
    )
end

"""
    ScatterLibrary

Pencil-beam scatter kernels in water cylinders, for each scanner: `scatter[scanner][column, row,
impact, diameter, energy]` and `primary[scanner][impact, diameter, energy]`, the detected signal
per incident photon (see [`ScatterTally`](@ref)), for cylinders of `diameters_mm`, rays passing the
axis at `impact_fractions` of the radius, photons of `energies_keV`; Geant4's attenuation of water
on a 1-keV grid (`water_mu_per_mm`, for primaries computed exactly by Beer–Lambert rather than
interpolated); with each scanner's bins and how the library was made (`metadata`).
"""
struct ScatterLibrary
    scanners::Vector{ScatterScanner}
    diameters_mm::Vector{Float64}
    impact_fractions::Vector{Float64}
    energies_keV::Vector{Float64}
    scatter::Dict{String, Array{Float32, 5}}
    primary::Dict{String, Array{Float32, 3}}
    water_mu_per_mm::Vector{Float64}
    metadata::Dict{String, Any}
end

"""
    scatter_library(tallies, scanners, diameters_mm, impact_fractions, energies_keV; water_mu_per_mm, metadata)

Assemble the per-photon kernels from the tallies of the jobs `(diameter, impact, energy)`, in the
order `for D in diameters, u in impact_fractions, E in energies` (each job's tallies one vector per
scanner, as [`scatter_tally`](@ref) gives them).
"""
function scatter_library(
        tallies,
        scanners::Vector{ScatterScanner},
        diameters_mm,
        impact_fractions,
        energies_keV;
        water_mu_per_mm::Vector{Float64},
        metadata = Dict{String, Any}(),
    )
    nD, nu, nE = length(diameters_mm), length(impact_fractions), length(energies_keV)
    scatter = Dict{String, Array{Float32, 5}}()
    primary = Dict{String, Array{Float32, 3}}()
    for (k, s) in enumerate(scanners)
        nc, nr = length(column_edges(s)) - 1, length(row_edges(s)) - 1
        S = zeros(Float32, nc, nr, nu, nD, nE)
        P = zeros(Float32, nu, nD, nE)
        j = 0
        for iD in 1:nD, iu in 1:nu, iE in 1:nE
            j += 1
            t = tallies[j][k]
            S[:, :, iu, iD, iE] .= t.scatter ./ t.n
            P[iu, iD, iE] = sum(t.primary) / t.n
        end
        scatter[s.name] = S
        primary[s.name] = P
    end
    return ScatterLibrary(
        scanners,
        collect(Float64, diameters_mm),
        collect(Float64, impact_fractions),
        collect(Float64, energies_keV),
        scatter,
        primary,
        water_mu_per_mm,
        metadata,
    )
end

"""Linear interpolation weights of `x` on the sorted grid `xs` (clamped): (i, i + 1, weight of i + 1)."""
function bracket(xs, x)
    x <= first(xs) && return (1, 1, 0.0)
    x >= last(xs) && return (length(xs), length(xs), 0.0)
    i = searchsortedlast(xs, x)
    return (i, i + 1, (x - xs[i]) / (xs[i + 1] - xs[i]))
end

"""
    library_kernel(library, scanner, diameter_mm, impact_fraction, energy_keV) -> (scatter, primary)

The kernel of one ray, interpolated in diameter, impact and energy: the scatter linearly (it changes
gently), the primary linearly in its logarithm (it falls exponentially with diameter and with
attenuation, and a straight line between grid points would overestimate it). A ray grazing the
cylinder beyond the last impact has the last impact's scatter scaled by its chord, which falls to
zero at the edge.
"""
function library_kernel(lib::ScatterLibrary, name::String, D, u, E)
    S, P = lib.scatter[name], lib.primary[name]
    iD = bracket(lib.diameters_mm, D)
    iu = bracket(lib.impact_fractions, u)
    iE = bracket(lib.energies_keV, E)
    scatter = zeros(size(S, 1), size(S, 2))
    log_primary = 0.0
    for (a, wa) in ((iD[1], 1 - iD[3]), (iD[2], iD[3])), (b, wb) in ((iu[1], 1 - iu[3]), (iu[2], iu[3])),
            (e, we) in ((iE[1], 1 - iE[3]), (iE[2], iE[3]))
        w = wa * wb * we
        w == 0 && continue
        scatter .+= w .* view(S, :, :, b, a, e)
        log_primary += w * log(max(Float64(P[b, a, e]), 1.0e-30))
    end
    u_last = last(lib.impact_fractions)
    u > u_last && (scatter .*= sqrt(max(0.0, 1 - min(u, 1.0)^2)) / sqrt(1 - u_last^2))
    return scatter, exp(log_primary)
end

"""
    library_scan(library, scanner, wed_mm, rays; collimation_mm, spectrum)
        -> (; scatter, primary, column_edges_deg, row_edges_mm, complete_rows)

The scatter and primary a fan beam makes on `scanner`'s detector, built from the library's pencil
kernels. `rays` is one entry per column bin of the scanner, `(; angle_deg, wet_mm, weight)`: the ray
through that column's centre, the water-equivalent length it crosses in the patient (0 for air), and
its share of the photons. The patient is taken, ray by ray, as the water cylinder of diameter
`wed_mm` (its water-equivalent diameter) that the ray crosses for the same length; its kernel is
moved to the ray's own column. `spectrum` is `(energies_keV, weights)`; `collimation_mm` is the
beam's width along z at the isocentre (the kernels are summed over it).

A ray longer than the patient's WED (through the long axis of an ellipse, or through bone) cannot
cross a cylinder of that diameter for its own length; it is given its own cylinder instead, of
diameter `max(wed_mm, ray.wet_mm)`, which it crosses through the axis. Shortening such a ray to the
WED would put too little water in its path and, for a tall ellipse, double the scatter at the
centre. A cylinder's rays are never longer than its diameter, so for cylinders nothing changes.

A kernel reaches the tally's z-range on either side of its own row, no further, so a ray at the
edge of the beam misses the rows beyond that: `complete_rows` marks the rows every ray reaches
whole (with the library's tally a full detector wide on either side, the whole detector for any
collimation up to the detector's own).
"""
function library_scan(
        lib::ScatterLibrary,
        name::String,
        wed_mm,
        rays;
        collimation_mm,
        spectrum,
    )
    s = lib.scanners[findfirst(x -> x.name == name, lib.scanners)]
    cols, rows = column_edges(s), row_edges(s)
    nc, nr = length(cols) - 1, length(rows) - 1
    centres = (cols[1:(end - 1)] .+ cols[2:end]) ./ 2
    row_centres = (rows[1:(end - 1)] .+ rows[2:end]) ./ 2
    sid = s.source_to_isocenter_mm

    # The beam's rows at the detector: each row bin's share of the collimated beam
    width = collimation_mm * s.source_to_detector_mm / sid
    overlap = [max(0.0, min(rows[r + 1], width / 2) - max(rows[r], -width / 2)) for r in 1:nr]
    beam_share = overlap ./ sum(overlap)
    beam_rows = findall(>(0), beam_share)

    e, w = spectrum
    weights = w ./ sum(w)
    scatter = zeros(nc, nr)
    primary = zeros(nc, nr)
    for (j, ray) in enumerate(rays)
        ray.wet_mm > 0 || continue
        # The ray's own cylinder: the patient's WED, or the ray's length if that is longer
        D = max(wed_mm, ray.wet_mm)
        R = D / 2
        u = sqrt(max(0.0, 1 - (ray.wet_mm / D)^2)) * sign(ray.angle_deg)
        # The kernel's own column, and the shift that puts it under this ray
        kernel_angle = rad2deg(asin(clamp(abs(u) * R / sid, -1, 1))) * sign(ray.angle_deg)
        shift = (ray.angle_deg - kernel_angle) / s.column_bin_deg
        for (Ek, wk) in zip(e, weights)
            wk > 0 || continue
            k, _ = library_kernel(lib, name, D, abs(u), Ek)
            k = u < 0 ? reverse(k; dims = 1) : k
            # Place the kernel's row offsets on the beam's rows (translation along z)
            for r0 in beam_rows, ci in 1:nc
                src = ci - shift
                i0 = floor(Int, src)
                fr = src - i0
                for (ii, wi) in ((i0, 1 - fr), (i0 + 1, fr))
                    1 <= ii <= nc || continue
                    for ri in 1:nr
                        dr = ri - r0 + (nr + 1) ÷ 2
                        1 <= dr <= nr || continue
                        scatter[ci, ri] += ray.weight * wk * wi * beam_share[r0] * k[ii, dr]
                    end
                end
            end
            # The primary, exactly: Beer–Lambert in water, through the detector's efficiency
            ci = searchsortedlast(cols, ray.angle_deg)
            if 1 <= ci <= nc
                ke = clamp(round(Int, Ek), 1, length(lib.water_mu_per_mm))
                p = exp(-lib.water_mu_per_mm[ke] * ray.wet_mm) * s.efficiency[ke] * (s.energy_weighted ? Ek : 1.0)
                for r0 in beam_rows
                    primary[ci, r0] += ray.weight * wk * beam_share[r0] * p
                end
            end
        end
    end
    mid = (nr + 1) ÷ 2
    complete_rows = [all(1 <= ri - r0 + mid <= nr for r0 in beam_rows) for ri in 1:nr]
    return (
        scatter = scatter,
        primary = primary,
        column_edges_deg = cols,
        row_edges_mm = rows,
        complete_rows = complete_rows,
    )
end
