# =============================================================================
# The scatter library, v2: water ellipses, and Compton and Rayleigh scatter apart
# =============================================================================
# A view of a patient is not a cylinder: through a chest from the front the patient is thin and
# wide, from the side thick and narrow. Version 1 stood every ray in a water cylinder of the
# patient's water-equivalent diameter, which puts the wrong amount of water beside a ray of a
# non-round patient, and the scatter that leaves sideways with it. Version 2 stands each view in a
# water ELLIPSE: as thick along the view and as wide across it as the patient's own water,
# measured from the view's rays (their water-equivalent lengths). Its kernels are pencil beams in
# water elliptical cylinders, indexed by the ellipse's thickness along the ray, its aspect
# (width ÷ thickness), the ray's offset as a fraction of the half-width, and the energy.
#
# The kernels also keep apart the scatter whose last scattering was a Rayleigh scattering from
# the scatter whose last was a Compton scattering. A patient that is not water (solid water's
# Rayleigh cross-section is 15 % below water's; bone's photoelectric absorption inflates its
# water-equivalent length) then has each part scaled, ray by ray, by how much more or less of that
# scattering its materials make along the ray than the water the kernel stands for. The Rayleigh
# scatter sent less than 10° is kept apart again (most of what reaches the middle of the detector,
# and where a material's form factors differ most from water's), and so are the lengths the detected
# photons travelled at each energy: a material that attenuates more than water at the lower
# energies of scattered photons (calcium, iodine) lets less of them out than its water-equivalent
# length, taken at 70 keV, says.
# =============================================================================

"""
    kernel_phantom(thickness_mm, aspect; length_mm = 800.0) -> CylinderPhantom

The water elliptical cylinder of a v2 kernel, centred at the origin: `thickness_mm` along the ray
(+y), `aspect × thickness_mm` across it (x), but never wider than `2 KERNEL_MAX_HALF_WIDTH_MM`
([`kernel_half_width`](@ref)).
"""
kernel_phantom(thickness_mm::Real, aspect::Real; length_mm::Real = 800.0) =
    CylinderPhantom(; semi_axes_mm = (kernel_half_width(thickness_mm, aspect), thickness_mm / 2), length_mm = Float64(length_mm))

"""
The widest a v2 kernel's ellipse is made (half-width, mm): 70 cm across, wider than any patient. A
grid of thicknesses and aspects reaches ellipses that are no one's (2.5 times 50 cm), and one wider
than the focal spot is far from its axis has no ray at its outer impacts; those are made 70 cm wide.
"""
const KERNEL_MAX_HALF_WIDTH_MM = 350.0

"""The half-width (mm) of the kernel ellipse of `thickness_mm` and `aspect`, at most `KERNEL_MAX_HALF_WIDTH_MM`."""
kernel_half_width(thickness_mm, aspect) = min(aspect * thickness_mm / 2, KERNEL_MAX_HALF_WIDTH_MM)

"""The distance (mm) from a v2 kernel's focal spot to its ellipse's axis."""
const KERNEL_SOURCE_MM = 600.0

"""
    kernel_ray(thickness_mm, aspect, impact_fraction) -> RayBeam

The ray of a v2 kernel, as a scanner fires it: from a focal spot `KERNEL_SOURCE_MM` from the axis
of [`kernel_phantom`](@ref)`(thickness_mm, aspect)`, on the side its thickness faces, passing the
axis at `impact_fraction` of the ellipse's half-width as that ray sees it ([`kernel_offset`](@ref)):
0 through the axis, 1 grazing the edge, for every shape. It is tilted toward that side, by
`asin(offset / KERNEL_SOURCE_MM)`, as the rays of a fan are: an ellipse, unlike a cylinder, looks
different from a tilted ray, and the kernel must see it as the scanner does. It starts 10 mm
outside the ellipse.
"""
function kernel_ray(thickness_mm::Real, aspect::Real, impact_fraction::Real)
    a, b = thickness_mm / 2, kernel_half_width(thickness_mm, aspect)
    d = kernel_direction(a, b, impact_fraction)
    t0 = KERNEL_SOURCE_MM - max(a, b) - 10.0
    return RayBeam(; origin_mm = (t0 * d[1], -KERNEL_SOURCE_MM + t0 * d[2], 0.0), direction = (d[1], d[2], 0.0))
end

"""
The distance (mm) from the axis of the ellipse of semi-axes (b across, a along the view) at which
the kernel ray of `impact_fraction` passes it. A ray tilted by `t` sees the ellipse as wide as
`√(b² cos²t + a² sin²t)` on either side of the axis (its extent perpendicular to the ray), not `b`:
less for an ellipse wider than thick, more for one thicker than wide. The offset is that fraction
of what its own ray sees, the tilt following from the offset (a few fixed-point steps), so that 1
grazes the edge whatever the shape.
"""
function kernel_offset(a, b, impact_fraction)
    offset = impact_fraction * b
    for _ in 1:30
        s = clamp(offset / KERNEL_SOURCE_MM, -1.0, 1.0)
        offset = impact_fraction * sqrt(b^2 * (1 - s^2) + a^2 * s^2)
    end
    return offset
end

"""The direction of the kernel ray of `impact_fraction` through the ellipse of semi-axes (b, a)."""
function kernel_direction(a, b, impact_fraction)
    s = kernel_offset(a, b, impact_fraction) / KERNEL_SOURCE_MM
    return (s, sqrt(1 - s^2))
end

"""The chord (mm) of the kernel ray of `impact_fraction` through the ellipse of semi-axes (b, a)."""
kernel_chord(a, b, impact_fraction) =
    ellipse_chord((0.0, -KERNEL_SOURCE_MM), kernel_direction(a, b, impact_fraction), (0.0, 0.0), (b, a))

"""The impact fraction whose kernel ray passes the axis of the ellipse (b, a) `offset_mm` away (bisection)."""
function kernel_impact_at(a, b, offset_mm)
    lo, hi = 0.0, 1.0
    for _ in 1:40
        mid = (lo + hi) / 2
        kernel_offset(a, b, mid) < offset_mm ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
The impact fraction in (0, 1) whose kernel ray crosses the ellipse (b, a) for `length_mm`
(bisection; 0 if the ray through the axis is no longer than that).
"""
function kernel_impact(a, b, length_mm)
    kernel_chord(a, b, 0.0) <= length_mm && return 0.0
    lo, hi = 0.0, 1.0
    for _ in 1:40
        mid = (lo + hi) / 2
        kernel_chord(a, b, mid) > length_mm ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    ShapeScatterLibrary

Pencil-beam scatter kernels in water elliptical cylinders, for each scanner:
`scatter[scanner][column, row, impact, aspect, thickness, energy]`, the part of it sent to the
detector by a Rayleigh scattering `rayleigh[scanner][…]` (same axes) and of that, by a Rayleigh
scattering of less than `FORWARD_RAYLEIGH_DEG`, `forward[scanner][…]`, and
`primary[scanner][impact, aspect, thickness, energy]`, all per incident photon;
`paths[scanner][process, band, impact, aspect, thickness, energy]`, the mean length (mm) the
detected photons of each process travelled in the water with an energy in each band of
`PATH_ENERGY_EDGES_KEV`; `offsets[scanner][column, impact, aspect, thickness, energy]` and
`spreads[…]`, the mean and the standard deviation (degrees) of where in each column the scatter
landed, from the column's centre; and
`angles[scanner][process, bin, energy]`, the detected scatter by its last scattering's process and
angle (bins of `SCATTER_ANGLE_EDGES_DEG`), summed over every kernel of each energy; for ellipses
`thicknesses_mm` thick along the ray and `aspects` × that wide, rays at `impact_fractions` of the
half-width, photons of `energies_keV`. Geant4's attenuation of water on a 1-keV grid
(`water_mu_per_mm`), each scanner's bins and how the library was made (`metadata`).
"""
struct ShapeScatterLibrary
    scanners::Vector{ScatterScanner}
    thicknesses_mm::Vector{Float64}
    aspects::Vector{Float64}
    impact_fractions::Vector{Float64}
    energies_keV::Vector{Float64}
    scatter::Dict{String, Array{Float32, 6}}
    rayleigh::Dict{String, Array{Float32, 6}}
    forward::Dict{String, Array{Float32, 6}}
    primary::Dict{String, Array{Float32, 4}}
    paths::Dict{String, Array{Float32, 6}}
    offsets::Dict{String, Array{Float32, 5}}
    spreads::Dict{String, Array{Float32, 5}}
    angles::Dict{String, Array{Float32, 3}}
    water_mu_per_mm::Vector{Float64}
    metadata::Dict{String, Any}
end

"""
    shape_scatter_library(tallies, scanners, thicknesses_mm, aspects, impact_fractions, energies_keV;
                          water_mu_per_mm, metadata) -> ShapeScatterLibrary

Assemble the per-photon kernels from the tallies of the jobs, in the order `for T in thicknesses,
r in aspects, u in impact_fractions, E in energies` (each job's tallies one per scanner).
"""
function shape_scatter_library(
        tallies,
        scanners::Vector{ScatterScanner},
        thicknesses_mm,
        aspects,
        impact_fractions,
        energies_keV;
        water_mu_per_mm::Vector{Float64},
        metadata = Dict{String, Any}(),
    )
    nT, nr, nu, nE = length(thicknesses_mm), length(aspects), length(impact_fractions), length(energies_keV)
    scatter = Dict{String, Array{Float32, 6}}()
    rayleigh = Dict{String, Array{Float32, 6}}()
    forward = Dict{String, Array{Float32, 6}}()
    primary = Dict{String, Array{Float32, 4}}()
    paths = Dict{String, Array{Float32, 6}}()
    offsets = Dict{String, Array{Float32, 5}}()
    spreads = Dict{String, Array{Float32, 5}}()
    angles = Dict{String, Array{Float32, 3}}()
    for (k, s) in enumerate(scanners)
        nc, nrow = length(column_edges(s)) - 1, length(row_edges(s)) - 1
        S = zeros(Float32, nc, nrow, nu, nr, nT, nE)
        Ry = zeros(Float32, nc, nrow, nu, nr, nT, nE)
        F = zeros(Float32, nc, nrow, nu, nr, nT, nE)
        P = zeros(Float32, nu, nr, nT, nE)
        L = zeros(Float32, 2, length(PATH_ENERGY_EDGES_KEV) - 1, nu, nr, nT, nE)
        # a uniform spread over the column, where nothing landed
        O = zeros(Float32, nc, nu, nr, nT, nE)
        Sd = fill(Float32(s.column_bin_deg / sqrt(12)), nc, nu, nr, nT, nE)
        A = zeros(Float32, 2, length(SCATTER_ANGLE_EDGES_DEG) - 1, nE)
        j = 0
        for iT in 1:nT, ir in 1:nr, iu in 1:nu, iE in 1:nE
            j += 1
            t = tallies[j][k]
            S[:, :, iu, ir, iT, iE] .= t.scatter ./ t.n
            Ry[:, :, iu, ir, iT, iE] .= t.rayleigh ./ t.n
            F[:, :, iu, ir, iT, iE] .= t.forward ./ t.n
            P[iu, ir, iT, iE] = sum(t.primary) / t.n
            mass = vec(sum(t.scatter; dims = 2))
            for c in 1:nc
                mass[c] > 0 || continue
                mean_offset = t.column_moments[1, c] / mass[c]
                O[c, iu, ir, iT, iE] = mean_offset
                Sd[c, iu, ir, iT, iE] = sqrt(max(t.column_moments[2, c] / mass[c] - mean_offset^2, 0.0))
            end
            by_process = (sum(t.scatter) - sum(t.rayleigh), sum(t.rayleigh))
            for process in 1:2
                by_process[process] > 0 && (L[process, :, iu, ir, iT, iE] .= t.paths[process, :] ./ by_process[process])
            end
            A[:, :, iE] .+= t.angles ./ t.n
        end
        scatter[s.name] = S
        rayleigh[s.name] = Ry
        forward[s.name] = F
        primary[s.name] = P
        paths[s.name] = L
        offsets[s.name] = O
        spreads[s.name] = Sd
        angles[s.name] = A
    end
    return ShapeScatterLibrary(
        scanners,
        collect(Float64, thicknesses_mm),
        collect(Float64, aspects),
        collect(Float64, impact_fractions),
        collect(Float64, energies_keV),
        scatter,
        rayleigh,
        forward,
        primary,
        paths,
        offsets,
        spreads,
        angles,
        water_mu_per_mm,
        metadata,
    )
end

"""
    library_kernel(library::ShapeScatterLibrary, scanner, thickness_mm, aspect, impact_fraction, energy_keV)
        -> (scatter, rayleigh, primary, forward)

One ray's kernel, interpolated linearly in thickness, in the logarithm of the aspect (the aspects
are spaced by ratios), in impact and in energy; the primary linearly in its logarithm. Outside the
grids the nearest kernels are used, and a ray grazing the ellipse beyond the last impact has the
last impact's scatter scaled by its chord, as in version 1.
"""
function library_kernel(lib::ShapeScatterLibrary, name::String, T, r, u, E)
    S, Ry, F, P = lib.scatter[name], lib.rayleigh[name], lib.forward[name], lib.primary[name]
    iT = bracket(lib.thicknesses_mm, T)
    ir = bracket(log.(lib.aspects), log(r))
    iu = bracket(lib.impact_fractions, u)
    iE = bracket(lib.energies_keV, E)
    scatter = zeros(size(S, 1), size(S, 2))
    rayleigh = zeros(size(S, 1), size(S, 2))
    forward = zeros(size(S, 1), size(S, 2))
    log_primary = 0.0
    for (t, wt) in ((iT[1], 1 - iT[3]), (iT[2], iT[3])), (a, wa) in ((ir[1], 1 - ir[3]), (ir[2], ir[3])),
            (b, wb) in ((iu[1], 1 - iu[3]), (iu[2], iu[3])), (e, we) in ((iE[1], 1 - iE[3]), (iE[2], iE[3]))
        w = wt * wa * wb * we
        w == 0 && continue
        scatter .+= w .* view(S, :, :, b, a, t, e)
        rayleigh .+= w .* view(Ry, :, :, b, a, t, e)
        forward .+= w .* view(F, :, :, b, a, t, e)
        log_primary += w * log(max(Float64(P[b, a, t, e]), 1.0e-30))
    end
    u_last = last(lib.impact_fractions)
    if u > u_last
        f = sqrt(max(0.0, 1 - min(u, 1.0)^2)) / sqrt(1 - u_last^2)
        scatter .*= f
        rayleigh .*= f
        forward .*= f
    end
    return scatter, rayleigh, exp(log_primary), forward
end

"""
    view_ellipse(rays, source_to_isocenter_mm; length = :wet_mm) -> (; thickness_mm, width_mm, centre_mm)

The water ellipse that stands for one view of a patient: its chords across the view have the same
water (the zeroth moment of the rays' water-equivalent lengths across the fan), the same centre
(the first) and the same spread (the second) as the patient's. `rays` are `(; angle_deg, …)` with
the length named by `length`, every 0.1° or so across the fan; a ray's place across the view is
where it passes the isocentre, `source_to_isocenter_mm × sin(angle)`. For an ellipse of water the
moments give it back exactly: its area π a b, its centre, and b²/4 for the spread of its chords.
"""
function view_ellipse(rays, sid; length::Symbol = :wet_mm)
    xs = [sid * sind(r.angle_deg) for r in rays]
    ls = [getfield(r, length) for r in rays]
    dx = [i == 1 ? xs[2] - xs[1] : i == lastindex(xs) ? xs[end] - xs[end - 1] : (xs[i + 1] - xs[i - 1]) / 2 for i in eachindex(xs)]
    m0 = sum(ls .* dx)
    m0 > 0 || return (thickness_mm = 0.0, width_mm = 0.0, centre_mm = 0.0)
    x0 = sum(xs .* ls .* dx) / m0
    spread = sum((xs .- x0) .^ 2 .* ls .* dx) / m0
    b = 2 * sqrt(spread)
    a = m0 / (π * b)
    return (thickness_mm = 2a, width_mm = 2b, centre_mm = x0)
end

"""
    angular_factors(materials, profile, energies_keV)
        -> Dict(name => (; compton, rayleigh, rayleigh_forward, rayleigh_wide))

How much more (or less) each material scatters, by each process, into the angles a scanner
detects than its total cross-section says, against water's, on a 1-keV grid. `profile[process,
bin, energy]` is the detected scatter by its last scattering's process (1 Compton, 2 Rayleigh) and
angle (bins of `SCATTER_ANGLE_EDGES_DEG`), at `energies_keV` (from forced detection: a
[`ScatterTally`](@ref)'s `angles`, summed over kernels). For a material m the factor is
`Σ_bin f(bin) P_m(bin) / P_water(bin)`, with `f` the profile's share of each bin and `P` the share
of each bin in the material's angular distribution for that process (Geant4's scattering functions
and form factors): 1 for water, below 1 for a material whose scattering is less forward.
`rayleigh_forward` and `rayleigh_wide` are Rayleigh's factor over the bins below and above
`FORWARD_RAYLEIGH_DEG` alone, for the library's `forward` scatter and the rest.
"""
function angular_factors(materials, profile, energies_keV)
    water = cached_scattering_tables(DETECTOR_MATERIALS[:Water])
    cells = water.cells
    nbins = length(SCATTER_ANGLE_EDGES_DEG) - 1
    # each bin's share of a material's angular distribution for one process, at one keV
    function shares(tables, process, E)
        mass = zeros(nbins)
        for k in 1:cells
            θ = acosd(clamp(-1 + (k - 0.5) * 2 / cells, -1.0, 1.0))
            b = clamp(searchsortedlast(SCATTER_ANGLE_EDGES_DEG, θ), 1, nbins)
            mass[b] += tables.pdf[k, E, process]
        end
        return mass ./ sum(mass)
    end
    # the profile's share of each bin, at every keV (interpolated between the library's energies)
    f = zeros(2, nbins, 150)
    for process in 1:2, E in 1:150
        i1, i2, w = bracket(energies_keV, Float64(E))
        v = (1 - w) .* profile[process, :, i1] .+ w .* profile[process, :, i2]
        f[process, :, E] .= sum(v) > 0 ? v ./ sum(v) : fill(1 / nbins, nbins)
    end
    forward_bins = findall(<=(FORWARD_RAYLEIGH_DEG), SCATTER_ANGLE_EDGES_DEG[2:end])
    wide_bins = setdiff(1:nbins, forward_bins)
    result = Dict{String, NamedTuple{(:compton, :rayleigh, :rayleigh_forward, :rayleigh_wide), NTuple{4, Vector{Float64}}}}()
    for m in materials
        tables = cached_scattering_tables(m)
        # the profile's average of the material's share over water's, over some of the bins
        function factor(process, bins)
            return map(1:150) do E
                pw, pm = shares(water, process, E), shares(tables, process, E)
                weight = sum(f[process, b, E] for b in bins)
                weight > 0 || return 1.0
                return sum(f[process, b, E] * (pw[b] > 0 ? pm[b] / pw[b] : 1.0) for b in bins) / weight
            end
        end
        result[m.name] = (
            compton = factor(1, 1:nbins),
            rayleigh = factor(2, 1:nbins),
            rayleigh_forward = factor(2, forward_bins),
            rayleigh_wide = factor(2, wide_bins),
        )
    end
    return result
end

"""
    process_ratios(phantom, origin, direction, attenuation, wet_mm; angular = nothing)
        -> (; compton, rayleigh, rayleigh_forward, delta_mu_per_mm)

How much Compton and Rayleigh scattering a ray makes in `phantom`, against the water its
water-equivalent length `wet_mm` stands for, on a 1-keV grid: `Σ_m L_m μ_process,m(E) /
(wet_mm × μ_process,water(E))`, `L_m` the ray's length in material `m`. `attenuation` holds every
material's and water's tables ([`forced_attenuation`](@ref)). With `angular`
([`angular_factors`](@ref)), each material's cross-sections are those into the angles the scanner
detects, and `rayleigh` is for the Rayleigh scatter sent wider than `FORWARD_RAYLEIGH_DEG`,
`rayleigh_forward` for the rest. 1 for water.

`delta_mu_per_mm[band]` is how much more the ray's materials attenuate than its water-equivalent
length of water, per mm of that water, at the energies of each band of `PATH_ENERGY_EDGES_KEV`:
`Σ_m (L_m / wet_mm) (μ_m(E) − μ_water(E) μ_m(70) / μ_water(70))`, zero at 70 keV, where the
water-equivalent length is taken, and at every energy for a material that attenuates as water
does. A detected photon that travelled `ℓ_b` mm of the kernel's water in each band is then
`exp(−Σ_b ℓ_b delta_mu_per_mm[b])` as likely to get out of the patient.
"""
function process_ratios(p::CylinderPhantom, origin, direction, attenuation, wet_mm; angular = nothing)
    water = attenuation[DETECTOR_MATERIALS[:Water].name]
    lengths = material_lengths(p, origin, direction)
    n = length(water.compton)
    nbands = length(PATH_ENERGY_EDGES_KEV) - 1
    wet_mm > 0 || return (compton = ones(n), rayleigh = ones(n), rayleigh_forward = ones(n), delta_mu_per_mm = zeros(nbands))
    g(m, key, k) = angular === nothing ? 1.0 : getfield(angular[m.name], key)[k]
    ratio(key, angle_key) = [
        sum(L * getfield(attenuation[m.name], key)[k] * g(m, angle_key, k) for (m, L) in lengths) /
            (wet_mm * getfield(water, key)[k]) for k in 1:n
    ]
    # each band's mean attenuation, over its whole keV
    band_keV(b) = max(1, 5(b - 1)):min(n, 5b - 1)
    band_mean(μ, b) = sum(μ[k] for k in band_keV(b)) / length(band_keV(b))
    delta = [
        sum(
                (L / wet_mm) * (band_mean(attenuation[m.name].total, b) - band_mean(water.total, b) * attenuation[m.name].total[70] / water.total[70])
                for (m, L) in lengths
            )
            for b in 1:nbands
    ]
    return (
        compton = ratio(:compton, :compton),
        rayleigh = ratio(:rayleigh, angular === nothing ? :rayleigh : :rayleigh_wide),
        rayleigh_forward = ratio(:rayleigh, angular === nothing ? :rayleigh : :rayleigh_forward),
        delta_mu_per_mm = delta,
    )
end

"""
    library_scan(library::ShapeScatterLibrary, scanner, rays; collimation_mm, spectrum,
                 view = :wet) -> (; scatter, primary, column_edges_deg, row_edges_mm,
                                       complete_rows, ellipse)

The scatter and primary a fan beam makes on `scanner`'s detector in one view, from the v2
kernels. `rays` has one entry every 0.1° or so across the fan, `(; angle_deg, wet_mm, weight)`,
and optionally `outline_mm` (the ray's geometric length through the patient's outline),
`compton_ratio`, `rayleigh_ratio`, `rayleigh_forward_ratio` (1-keV grids; 1 if absent) and
`delta_mu_per_mm` (by band; 0 if absent), from [`process_ratios`](@ref).

The view is stood in its water ellipse ([`view_ellipse`](@ref), from the water-equivalent lengths,
with `view = :wet`; from the outline with `view = :outline`; `:mean` takes the geometric mean of
the two scatters, bin by bin). Each ray takes the kernel of that ellipse at the
offset whose chord is the ray's own water-equivalent length; a ray longer than the ellipse is
thick gets an ellipse of its own, as thick as the ray and as wide as the view's (version 1's
protection, in its ellipse form). Its kernel's Compton, wide Rayleigh and forward Rayleigh parts are
scaled by the ray's ratios and by how much less of each gets out of the ray's materials than of
water (the kernel's `paths` against the ray's `delta_mu_per_mm`), and moved to the ray's own
column.
"""
function library_scan(
        lib::ShapeScatterLibrary,
        name::String,
        rays;
        collimation_mm,
        spectrum,
        view::Symbol = :wet,
        impact::Symbol = :matched,
        ellipse = nothing,
    )
    if view === :mean
        # Low-density tissue, between its two readings: the geometric mean, bin by bin
        w = library_scan(lib, name, rays; collimation_mm, spectrum, view = :wet, impact, ellipse)
        o = library_scan(lib, name, rays; collimation_mm, spectrum, view = :outline, impact, ellipse)
        return merge(w, (scatter = sqrt.(w.scatter .* o.scatter), ellipse = (wet = w.ellipse, outline = o.ellipse)))
    end
    view in (:wet, :outline) || throw(ArgumentError("view must be :wet, :outline or :mean"))
    s = lib.scanners[findfirst(x -> x.name == name, lib.scanners)]
    cols, rows = column_edges(s), row_edges(s)
    nc, nr = length(cols) - 1, length(rows) - 1
    sid = s.source_to_isocenter_mm
    ellipse = ellipse === nothing ? view_ellipse(rays, sid; length = view === :outline ? :outline_mm : :wet_mm) : ellipse
    a, b, x0 = ellipse.thickness_mm / 2, ellipse.width_mm / 2, ellipse.centre_mm

    width = collimation_mm * s.source_to_detector_mm / sid
    overlap = [max(0.0, min(rows[r + 1], width / 2) - max(rows[r], -width / 2)) for r in 1:nr]
    beam_share = overlap ./ sum(overlap)
    beam_rows = findall(>(0), beam_share)

    e, w = spectrum
    weights = w ./ sum(w)
    nE = length(lib.energies_keV)
    # Each spectral energy's share of the library's energies (linear interpolation in energy)
    shares = [bracket(lib.energies_keV, Ek) for Ek in e]
    scatter = zeros(nc, nr)
    primary = zeros(nc, nr)
    compton_weight, rayleigh_weight, forward_weight = zeros(nE), zeros(nE), zeros(nE)
    for ray in rays
        ray.wet_mm > 0 || continue
        # The ray's offset from the ellipse's axis, perpendicular to it
        offset = sid * sind(ray.angle_deg) - x0 * cosd(ray.angle_deg)
        side = offset >= 0 ? 1.0 : -1.0
        if impact === :geometric
            # The ray where it really passes across the view, its kernel scaled by the water it
            # crosses against the ellipse's chord there
            a_ray = a
            u = clamp(kernel_impact_at(a, b, abs(offset)), 0.0, 0.99) * side
            amount = min(3.0, ray.wet_mm / max(kernel_chord(a, b, abs(u)), 1.0))
        else
            # The offset whose kernel ray crosses the ray's own water-equivalent length; a ray
            # longer than the ellipse is thick gets an ellipse as thick as itself
            a_ray = max(a, ray.wet_mm / 2)
            u = kernel_impact(a_ray, b, ray.wet_mm) * side
            amount = 1.0
        end
        aspect = b / a_ray
        cr = hasproperty(ray, :compton_ratio) ? ray.compton_ratio : nothing
        rr = hasproperty(ray, :rayleigh_ratio) ? ray.rayleigh_ratio : nothing
        fr_ratio = hasproperty(ray, :rayleigh_forward_ratio) ? ray.rayleigh_forward_ratio : rr
        δμ = hasproperty(ray, :delta_mu_per_mm) ? ray.delta_mu_per_mm : nothing

        # The spectrum, and the ray's material ratios at each of its energies, gathered onto the
        # library's energies: then each grid energy's kernel is interpolated once
        fill!(compton_weight, 0.0)
        fill!(rayleigh_weight, 0.0)
        fill!(forward_weight, 0.0)
        ci = searchsortedlast(cols, ray.angle_deg)
        for (k, (Ek, wk)) in enumerate(zip(e, weights))
            wk > 0 || continue
            ke = clamp(round(Int, Ek), 1, length(lib.water_mu_per_mm))
            fc = cr === nothing ? 1.0 : cr[ke]
            fr = rr === nothing ? 1.0 : rr[ke]
            ff = fr_ratio === nothing ? 1.0 : fr_ratio[ke]
            g1, g2, t = shares[k]
            compton_weight[g1] += wk * (1 - t) * fc
            compton_weight[g2] += wk * t * fc
            rayleigh_weight[g1] += wk * (1 - t) * fr
            rayleigh_weight[g2] += wk * t * fr
            forward_weight[g1] += wk * (1 - t) * ff
            forward_weight[g2] += wk * t * ff
            if 1 <= ci <= nc
                p = exp(-lib.water_mu_per_mm[ke] * ray.wet_mm) * s.efficiency[ke] * (s.energy_weighted ? Ek : 1.0)
                for r0 in beam_rows
                    primary[ci, r0] += ray.weight * wk * beam_share[r0] * p
                end
            end
        end
        # The grid's kernels around the ray (thickness, aspect, impact), each moved so that its own
        # ray lies under this one before they are blended: neighbouring kernels' rays pass at
        # different offsets, so blending them in place would smear the scatter across columns
        iT = bracket(lib.thicknesses_mm, 2a_ray)
        ir = bracket(log.(lib.aspects), log(aspect))
        iu = bracket(lib.impact_fractions, abs(u))
        u_last = last(lib.impact_fractions)
        graze = abs(u) > u_last ? sqrt(max(0.0, 1 - min(abs(u), 1.0)^2)) / sqrt(1 - u_last^2) : 1.0
        S, Ry, F, Lp = lib.scatter[name], lib.rayleigh[name], lib.forward[name], lib.paths[name]
        Om, Sd = lib.offsets[name], lib.spreads[name]
        for (tT, wT) in ((iT[1], 1 - iT[3]), (iT[2], iT[3])), (tr, wr) in ((ir[1], 1 - ir[3]), (ir[2], ir[3])),
                (tu, wu) in ((iu[1], 1 - iu[3]), (iu[2], iu[3]))
            wg = wT * wr * wu
            wg == 0 && continue
            kernel = zeros(nc, nr)
            # each column's scatter, and its first and second moments about the column's centre
            column_mass, first, second = zeros(nc), zeros(nc), zeros(nc)
            for g in 1:nE
                compton_weight[g] == 0 && rayleigh_weight[g] == 0 && continue
                ks, kr, kf = Base.view(S, :, :, tu, tr, tT, g), Base.view(Ry, :, :, tu, tr, tT, g), Base.view(F, :, :, tu, tr, tT, g)
                # how much less of each process's scatter gets out of the ray's materials
                out_c, out_r = if δμ === nothing
                    1.0, 1.0
                else
                    exp(-sum(Lp[1, band, tu, tr, tT, g] * δμ[band] for band in eachindex(δμ))),
                        exp(-sum(Lp[2, band, tu, tr, tT, g] * δμ[band] for band in eachindex(δμ)))
                end
                part = (compton_weight[g] * out_c) .* (ks .- kr) .+ (rayleigh_weight[g] * out_r) .* (kr .- kf) .+
                    (forward_weight[g] * out_r) .* kf
                kernel .+= part
                for c in 1:nc
                    m = sum(Base.view(part, c, :))
                    o, sd = Om[c, tu, tr, tT, g], Sd[c, tu, tr, tT, g]
                    column_mass[c] += m
                    first[c] += m * o
                    second[c] += m * (sd^2 + o^2)
                end
            end
            if u < 0
                # the mirror image: columns reversed, offsets within them negated
                kernel = reverse(kernel; dims = 1)
                column_mass, first, second = reverse(column_mass), -reverse(first), reverse(second)
            end
            # this kernel's own ray: its impact of its own half-width, seen from this scanner
            T_node = lib.thicknesses_mm[tT]
            offset_node = kernel_offset(T_node / 2, kernel_half_width(T_node, lib.aspects[tr]), lib.impact_fractions[tu])
            node_angle = rad2deg(asin(clamp(offset_node / sid, -1, 1))) * side
            factor = ray.weight * wg * amount * graze
            # Each column of the kernel moved by the ray's angle from its own, as what it holds: the
            # scatter centred where it landed in the column and spread as it landed, a box of that
            # width laid over this scan's columns. A sharp forward peak lands whole; scatter spread
            # over its column is shared as linear interpolation would share it.
            w = s.column_bin_deg
            for ii in 1:nc
                column_mass[ii] > 0 || continue
                centre = first[ii] / column_mass[ii]
                spread = sqrt(max(second[ii] / column_mass[ii] - centre^2, 0.0))
                x = cols[ii] + w / 2 + centre + (ray.angle_deg - node_angle)
                half = min(sqrt(3) * spread, w / 2)
                lo, hi = x - half, x + half
                for ci in max(1, searchsortedlast(cols, lo)):min(nc, searchsortedlast(cols, hi))
                    share = hi > lo ? (min(hi, cols[ci + 1]) - max(lo, cols[ci])) / (hi - lo) : 1.0
                    share > 0 || continue
                    for r0 in beam_rows, ri in 1:nr
                        dr = ri - r0 + (nr + 1) ÷ 2
                        1 <= dr <= nr || continue
                        scatter[ci, ri] += factor * share * beam_share[r0] * kernel[ii, dr]
                    end
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
        ellipse = ellipse,
    )
end
