# =============================================================================
# Forced detection: every scattering in the patient scored toward every detector
# =============================================================================
# Counting the scattered photons that happen to reach a detector wastes almost all of them: behind a
# 2-D grid, one photon in ten to a hundred thousand of a pencil beam through 45 cm of water is
# detected as scatter. Forced detection (next-event estimation) instead scores, at every Compton
# and Rayleigh scattering Geant4 makes, the chance that the photon's next flight reaches the
# detector: the integral, over the directions that meet the detector, of the scattering's angular
# distribution times the attenuation along the way out (exact, through the phantom's cylinder and
# its inserts), the grid and the detector. Geant4 still follows every photon to the end, so multiple scattering is all
# there; only the last flight is computed rather than sampled.
#
# The integral is estimated from points spread uniformly over the detector. Compton scattering's
# angular distribution is broad, and these points alone estimate it well. Rayleigh scattering's is a
# narrow forward cone: a uniform point that happens to fall in it would carry a huge weight. So at a
# Rayleigh scattering, directions are also drawn from the cone itself, and the two kinds of draws
# are combined by multiple importance sampling (the balance heuristic), which keeps every draw's
# weight below 1/(the number of draws from the cone).
#
# For a pencil-beam kernel (one ray, one energy) the incident photon's first scattering needs no
# sampling at all: along the ray its chance per mm of scattering at depth t by each process is
# μ_process e^(−μ t), so the first scattering is integrated over the ray's depth (stratified, a few
# hundred depths, many points on the detector from each), and Geant4's own first scatterings are
# left out of the sum. Only the multiple scattering is then left to the photons' random paths.
#
# The estimate is unbiased for the physics it scores with: Geant4's own scattering functions
# (Compton) and form factors (Rayleigh) of each element of the material where the photon scatters,
# from its data files (G4LEDATA), and Geant4's own attenuation of each material. The Doppler
# broadening of Compton energies is left out, the one difference from Geant4's transport, and each
# run carries the photon-by-photon count of the same scatter to check it against.
# =============================================================================

"""
    ForcedScatterReadout(scanners; attenuation, points = 4, directions = 4, roulette_below = 0.01,
                         depths = 256, seed = 0)
    ForcedScatterReadout(scanners; water_mu_per_mm, water_compton_per_mm = Float64[],
                         water_rayleigh_per_mm = Float64[], kw...)

Score a run on a [`CylinderPhantom`](@ref) by forced detection: at every Compton and Rayleigh
scattering, the chance of the photon's next flight reaching each scanner's detector, estimated from
`points` points drawn uniformly over the detector, and at a Rayleigh scattering also `directions`
directions drawn from its angular distribution. `attenuation` holds, for every material of the
phantom (by name), Geant4's attenuation in all and by Compton and Rayleigh scattering on a 1-keV
grid from 1 keV ([`forced_attenuation`](@ref) makes it); the second form is the same for a phantom
of water alone.

A monoenergetic run of one ray ([`RayBeam`](@ref)) through a phantom of one material, given its
Compton and Rayleigh attenuation, has its first scatterings integrated along the ray at `depths`
depths rather than sampled.

A scattering that can add little, its weight `w` (1 for a photon Geant4 followed) times the
attenuation of its shortest way out toward the detector (its centre or either end of the fan) below
`roulette_below`, is scored only with probability `w a / roulette_below`, and then with its weight
raised to match (Russian roulette): the estimate stays unbiased, and the time goes to the
scatterings that matter. The first scatterings integrated along a ray, each standing for many
photons, are scored at every depth.

The points are drawn from the run's seed and energy and `seed`, so a run's tally is repeatable.
Each [`ScatterTally`](@ref) also carries the photon-by-photon count (`analog_scatter`) and the
primary.
"""
struct ForcedScatterReadout
    scanners::Vector{ScatterScanner}
    attenuation::Dict{String, NamedTuple{(:total, :compton, :rayleigh), NTuple{3, Vector{Float64}}}}
    points::Int
    directions::Int
    roulette_below::Float64
    depths::Int
    seed::Int
end

function ForcedScatterReadout(
        scanners;
        attenuation = nothing,
        water_mu_per_mm = nothing,
        water_compton_per_mm = Float64[],
        water_rayleigh_per_mm = Float64[],
        points = 4,
        directions = 4,
        roulette_below = 0.01,
        depths = 256,
        seed = 0,
    )
    (attenuation === nothing) == (water_mu_per_mm === nothing) &&
        throw(ArgumentError("give exactly one of `attenuation` or `water_mu_per_mm`"))
    tables = if attenuation === nothing
        Dict(
            DETECTOR_MATERIALS[:Water].name => (
                total = collect(Float64, water_mu_per_mm),
                compton = collect(Float64, water_compton_per_mm),
                rayleigh = collect(Float64, water_rayleigh_per_mm),
            ),
        )
    else
        Dict(
            String(name) => (
                    total = collect(Float64, a.total),
                    compton = collect(Float64, a.compton),
                    rayleigh = collect(Float64, a.rayleigh),
                )
                for (name, a) in attenuation
        )
    end
    return ForcedScatterReadout(scanners, tables, points, directions, roulette_below, depths, seed)
end

"""
    forced_attenuation(materials; physics = PhysicsSettings()) -> Dict

Geant4's attenuation of each of `materials` ([`DetectorMaterial`](@ref)s) at every keV from 1 to
150, in all and by Compton and Rayleigh scattering, by name: what [`ForcedScatterReadout`](@ref)
needs.
"""
forced_attenuation(materials; physics::PhysicsSettings = PhysicsSettings()) = Dict(
    m.name => (
            total = [geant4_attenuation(m, Float64(E); kind = :total, physics) for E in 1:150],
            compton = [geant4_attenuation(m, Float64(E); kind = :compton, physics) for E in 1:150],
            rayleigh = [geant4_attenuation(m, Float64(E); kind = :rayleigh, physics) for E in 1:150],
        )
        for m in unique(m -> m.name, materials)
)

needs_deposits(::ForcedScatterReadout) = false
needs_vertices(::ForcedScatterReadout) = true

# -----------------------------------------------------------------------------
# The angular distributions of scattering in a material
# -----------------------------------------------------------------------------

const ELECTRON_REST_KEV = 510.99895
const HC_KEV_CM = 1.239841984e-7   # h c: the wavelength of a 1-keV photon in cm

"""Two-column data (x, y) from a Geant4 data file, until its `-1` terminator."""
function read_g4_pairs(path)
    xs, ys = Float64[], Float64[]
    for line in eachline(path)
        fields = split(line)
        length(fields) == 2 || continue
        x, y = parse(Float64, fields[1]), parse(Float64, fields[2])
        x < 0 && break
        push!(xs, x)
        push!(ys, y)
    end
    return xs, ys
end

"""Log-log interpolation of tabulated (x, y), linear below the first non-zero point."""
function loglog(xs, ys, x)
    x <= 0 && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    if xs[i] <= 0 || ys[i] <= 0 || ys[i + 1] <= 0
        return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
    end
    t = log(x / xs[i]) / log(xs[i + 1] / xs[i])
    return ys[i] * (ys[i + 1] / ys[i])^t
end

"""
    ScatteringTables

The angular distributions of Compton and Rayleigh scattering in one material, per steradian, for
incident energies of 1–150 keV (1 keV apart), constant within each of `cells` equal intervals of the cosine
of the scattering angle from −1 to 1: `pdf[cell, energy, process]`, each normalised to 1 over the
sphere; `cdf[cell, energy, process]`, its cumulative sum, to draw directions from, and
`guide[m, energy, process]`, the first cell whose cumulative sum reaches (m − 1)/`cells`, so that a
draw starts its search there.
"""
struct ScatteringTables
    cells::Int
    pdf::Array{Float64, 3}
    cdf::Array{Float64, 3}
    guide::Array{Int32, 3}
end

"""
    scattering_tables(material::DetectorMaterial; data = ENV["G4LEDATA"], cells = 4000) -> ScatteringTables

Build the angular distributions from Geant4's data: Klein–Nishina times the incoherent scattering
function S(x) (Compton), Thomson times the squared form factor F(x)² (Rayleigh), summed over the
material's elements by their atoms per gram (mass fraction over atomic mass), x = sin(θ/2)/λ in
cm⁻¹ as in the files; each cell takes the mean of its two ends.
"""
function scattering_tables(material::DetectorMaterial; data = ENV["G4LEDATA"], cells::Integer = 4000)
    Zs = sort!(collect(keys(material.mass_fractions)))
    atoms = Dict(Z => material.mass_fractions[Z] / XA.ATOMIC_MASS[element_symbol(Z)] for Z in Zs)
    S = Dict(Z => read_g4_pairs(joinpath(data, "comp", "ce-sf-$Z.dat")) for Z in Zs)
    F = Dict(Z => read_g4_pairs(joinpath(data, "livermore", "rayl", "re-ff-$Z.dat")) for Z in Zs)
    μs = range(-1.0, 1.0; length = cells + 1)
    Δμ = step(μs)
    ends = zeros(cells + 1, 2)
    pdf = zeros(cells, 150, 2)
    cdf = zeros(cells, 150, 2)
    for E in 1:150
        for (j, μ) in enumerate(μs)
            x = sqrt((1 - μ) / 2) * E / HC_KEV_CM
            ratio = 1 / (1 + E / ELECTRON_REST_KEV * (1 - μ))
            klein_nishina = ratio^2 * (ratio + 1 / ratio - (1 - μ^2)) / 2
            ends[j, VERTEX_COMPTON] = klein_nishina * sum(atoms[Z] * loglog(S[Z]..., x) for Z in Zs)
            ends[j, VERTEX_RAYLEIGH] = (1 + μ^2) / 2 * sum(atoms[Z] * loglog(F[Z]..., x)^2 for Z in Zs)
        end
        for p in (VERTEX_COMPTON, VERTEX_RAYLEIGH)
            cell = (ends[1:(end - 1), p] .+ ends[2:end, p]) ./ 2
            total = 2π * Δμ * sum(cell)
            pdf[:, E, p] .= cell ./ total
            cdf[:, E, p] .= cumsum(cell) ./ sum(cell)
        end
    end
    guide = Int32[
        min(searchsortedfirst(view(cdf, :, E, p), (m - 1) / cells), cells)
            for m in 1:cells, E in 1:150, p in 1:2
    ]
    return ScatteringTables(Int(cells), pdf, cdf, guide)
end

const SCATTERING_TABLES = Dict{String, ScatteringTables}()

"""A material's scattering tables, built on first use in this process."""
cached_scattering_tables(m::DetectorMaterial) = get!(() -> scattering_tables(m), SCATTERING_TABLES, m.name)

"""The energy bins either side of `E` keV and the weight of the upper one."""
function energy_bracket(E)
    e = clamp(E, 1.0, 150.0)
    i = min(floor(Int, e), 149)
    return i, e - i
end

"""The chance per steradian of scattering by `process` at cosine `μ`, for a photon of `E` keV."""
function scattering_pdf(w::ScatteringTables, process, E, μ)
    i, te = energy_bracket(E)
    j = clamp(floor(Int, (μ + 1) / 2 * w.cells) + 1, 1, w.cells)
    return (1 - te) * w.pdf[j, i, process] + te * w.pdf[j, i + 1, process]
end

"""
A cosine of the scattering angle drawn from [`scattering_pdf`](@ref) exactly: one of the two energy
bins by its weight, a cell by that bin's distribution, a point uniformly within the cell.
"""
function sample_cosine(w::ScatteringTables, process, E, rng)
    i, te = energy_bracket(E)
    k = rand(rng) < te ? i + 1 : i
    u = rand(rng)
    j = Int(w.guide[min(floor(Int, u * w.cells) + 1, w.cells), k, process])
    while j < w.cells && w.cdf[j, k, process] < u
        j += 1
    end
    return -1 + 2 * (j - 1 + rand(rng)) / w.cells
end

"""A unit vector at cosine `μ` to the unit vector `d`, at a uniformly drawn azimuth."""
function rotate_by(d, μ, rng)
    sinθ = sqrt(max(0.0, 1 - μ^2))
    sφ, cφ = sincos(2π * rand(rng))
    # Two unit vectors perpendicular to d
    a = abs(d[3]) < 0.9 ? (0.0, 0.0, 1.0) : (1.0, 0.0, 0.0)
    e1 = (a[2] * d[3] - a[3] * d[2], a[3] * d[1] - a[1] * d[3], a[1] * d[2] - a[2] * d[1])
    e1 = e1 ./ sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2)
    e2 = (d[2] * e1[3] - d[3] * e1[2], d[3] * e1[1] - d[1] * e1[3], d[1] * e1[2] - d[2] * e1[1])
    return (
        μ * d[1] + sinθ * (cφ * e1[1] + sφ * e2[1]),
        μ * d[2] + sinθ * (cφ * e1[2] + sφ * e2[2]),
        μ * d[3] + sinθ * (cφ * e1[3] + sφ * e2[3]),
    )
end

"""Linear interpolation of a table on the 1-keV grid from 1 keV."""
function on_kev_grid(table, E)
    e = clamp(E, 1.0, length(table) - 1.0e-9)
    i = floor(Int, e)
    return table[i] + (e - i) * (table[i + 1] - table[i])
end

# -----------------------------------------------------------------------------
# The way out of the patient
# -----------------------------------------------------------------------------

"""The path from the point `r` inside `p` along the unit vector `ω` to where it leaves `p` (mm)."""
function exit_length(p::CylinderPhantom, r, ω)
    a, b = p.semi_axes_mm
    qx, qy = (r[1] - p.centre_mm[1]) / a, (r[2] - p.centre_mm[2]) / b
    ux, uy = ω[1] / a, ω[2] / b
    A = ux^2 + uy^2
    side = if A > 1.0e-18
        B = qx * ux + qy * uy
        C = qx^2 + qy^2 - 1
        (-B + sqrt(max(B^2 - A * C, 0.0))) / A
    else
        Inf
    end
    half = p.length_mm / 2
    cap = ω[3] > 0 ? (half - r[3]) / ω[3] : ω[3] < 0 ? (-half - r[3]) / ω[3] : Inf
    return max(min(side, cap), 0.0)
end

"""
The media of `p`, its body then each of its inserts: each one's scattering tables and attenuation
(from `readout`).
"""
function phantom_media(p::CylinderPhantom, readout::ForcedScatterReadout)
    materials = vcat(p.material, [insert.material for insert in p.inserts])
    for m in materials
        haskey(readout.attenuation, m.name) ||
            throw(ArgumentError("forced detection needs the attenuation of $(m.name) (forced_attenuation)"))
    end
    return [(tables = cached_scattering_tables(m), attenuation = readout.attenuation[m.name]) for m in materials]
end

"""Which medium of `p` the point `r` is in: 1 for the body, 1 + k inside its k-th insert."""
function medium_at(p::CylinderPhantom, r)
    for (k, insert) in enumerate(p.inserts)
        x = (r[1] - p.centre_mm[1] - insert.x_mm) / insert.semi_axes_mm[1]
        y = (r[2] - p.centre_mm[2] - insert.y_mm) / insert.semi_axes_mm[2]
        x^2 + y^2 <= 1 && return 1 + k
    end
    return 1
end

"""The length of the path from `r` along the unit vector `ω`, up to `t_max`, inside `insert` of `p`."""
function insert_length(p::CylinderPhantom, insert::PhantomInsert, r, ω, t_max)
    a, b = insert.semi_axes_mm
    qx = (r[1] - p.centre_mm[1] - insert.x_mm) / a
    qy = (r[2] - p.centre_mm[2] - insert.y_mm) / b
    ux, uy = ω[1] / a, ω[2] / b
    A = ux^2 + uy^2
    A > 1.0e-18 || return qx^2 + qy^2 <= 1 ? t_max : 0.0
    B = qx * ux + qy * uy
    disc = B^2 - A * (qx^2 + qy^2 - 1)
    disc > 0 || return 0.0
    return max(0.0, min((-B + sqrt(disc)) / A, t_max) - max((-B - sqrt(disc)) / A, 0.0))
end

"""
The fraction of photons of `E` keV that get out of `p` from `r` along the unit vector `ω` without
interacting: through the body, and through each insert on the way.
"""
function escape_attenuation(p::CylinderPhantom, media, r, ω, E)
    t_exit = exit_length(p, r, ω)
    isempty(p.inserts) && return exp(-on_kev_grid(media[1].attenuation.total, E) * t_exit)
    μL = 0.0
    body = t_exit
    for (k, insert) in enumerate(p.inserts)
        len = insert_length(p, insert, r, ω, t_exit)
        len > 0 || continue
        μL += on_kev_grid(media[1 + k].attenuation.total, E) * len
        body -= len
    end
    return exp(-μL - on_kev_grid(media[1].attenuation.total, E) * max(body, 0.0))
end

"""The path from the point `r` inside `p` out toward the point `l` (x, y) at the height z = 0 (mm)."""
function path_toward(p::CylinderPhantom, r, l)
    w = (l[1] - r[1], l[2] - r[2], -r[3])
    return exit_length(p, r, w ./ sqrt(w[1]^2 + w[2]^2 + w[3]^2))
end

# -----------------------------------------------------------------------------
# The tally
# -----------------------------------------------------------------------------

"""
    forced_scatter_tally(run, readout::ForcedScatterReadout) -> Vector{ScatterTally}

The scatter each scanner's detector would record from the run, by forced detection, with the
photon-by-photon count of the same and the primary alongside.
"""
function forced_scatter_tally(run::MonoenergeticRun, readout::ForcedScatterReadout)
    phantom = run.detector
    phantom isa CylinderPhantom || throw(ArgumentError("forced detection needs a run on a CylinderPhantom"))
    run.vertices === nothing && throw(ArgumentError("simulate with record_vertices = true"))

    analog = scatter_tally(run, ScatterReadout(readout.scanners))
    rng = Xoshiro(hash((run.seed, run.energy_keV, readout.seed, :forced)))
    media = phantom_media(phantom, readout)
    frames = [detector_frame(s, run, phantom, analog[k]) for (k, s) in enumerate(readout.scanners)]

    # One ray at one energy through one material: its first scatterings integrated along it, the
    # sampled ones left out
    integrate_first = run.beam isa RayBeam && run.photon_energy_keV === nothing && isempty(phantom.inserts) &&
        !isempty(media[1].attenuation.compton) && !isempty(media[1].attenuation.rayleigh)
    if integrate_first
        forced_detection!(frames, subset(run.vertices, .!run.vertices.first), phantom, media, readout, rng)
        first_scatter!(frames, run, phantom, media, readout, rng)
    else
        forced_detection!(frames, run.vertices, phantom, media, readout, rng)
    end
    return [
        ScatterTally(
                s.name,
                run.n,
                analog[k].photon_keV,
                frames[k].cols,
                frames[k].rows,
                frames[k].scatter,
                analog[k].primary,
                frames[k].spectrum,
                analog[k].scatter,
            )
            for (k, s) in enumerate(readout.scanners)
    ]
end

"""
One scanner's detector as a run sees it: the focal spot `f`, the central ray `c` and the direction
of +γ across the fan, the arc's radius, its bins (uniform: the first edge and the width), the
binned area on the arc (mm²), three points on it to judge a scattering's way out (the centre and the
ends of the fan), and the tally being built.
"""
function detector_frame(s::ScatterScanner, run, phantom, analog::ScatterTally)
    f, c = focal_spot_of(run.beam, phantom, s)
    across = (c[2], -c[1])   # +γ, toward +x when the central ray is +y
    cols, rows = analog.column_edges_deg, analog.row_edges_mm
    R = s.source_to_detector_mm
    on_arc(γ) = (f[1] + R * (cosd(γ) * c[1] + sind(γ) * across[1]), f[2] + R * (cosd(γ) * c[2] + sind(γ) * across[2]))
    return (
        scanner = s,
        f = f,
        c = c,
        across = across,
        R = R,
        cols = cols,
        rows = rows,
        γ_lo = cols[1],
        γ_step = cols[2] - cols[1],
        z_lo = rows[1],
        z_step = rows[2] - rows[1],
        area = R * deg2rad(cols[end] - cols[1]) * (rows[end] - rows[1]),
        landmarks = (on_arc(0.0), on_arc(s.fan_deg / 2), on_arc(-s.fan_deg / 2)),
        scatter = zeros(length(cols) - 1, length(rows) - 1),
        spectrum = zeros(length(analog.scatter_spectrum)),
    )
end

"""
Score one direction `ω` from a scattering at `r` that meets the detector of `frame` at `h`, `dist2`
away: the integrand (the scattering's chance per steradian `p_scatter` toward `ω`, the way out
`escape`, the grid and the detector) over the density of the draws: `K` points uniform over the
detector and `L` directions from the scattering.
"""
function score!(frame, ω, h, dist2, E′, p_scatter, escape, K, L, boost)
    n_xy = ((h[1] - frame.f[1]) / frame.R, (h[2] - frame.f[2]) / frame.R)
    c = frame.c
    γ = rad2deg(atan(n_xy[1] * c[2] - n_xy[2] * c[1], c[1] * n_xy[1] + c[2] * n_xy[2]))
    ci = floor(Int, (γ - frame.γ_lo) / frame.γ_step) + 1
    ri = floor(Int, (h[3] - frame.z_lo) / frame.z_step) + 1
    (1 <= ci <= size(frame.scatter, 1) && 1 <= ri <= size(frame.scatter, 2)) || return nothing
    s = frame.scanner
    gη, k = detection_weight(s, n_xy, h[3], ω, E′)
    gη > 0 || return nothing
    p_uniform = dist2 / (frame.area * abs(n_xy[1] * ω[1] + n_xy[2] * ω[2]))
    reach = p_scatter * escape / (K * p_uniform + L * p_scatter) * boost
    frame.scatter[ci, ri] += reach * gη * (s.energy_weighted ? E′ : 1.0)
    frame.spectrum[k] += reach * gη
    return nothing
end

"""The scatterings of `v` where `keep` is true."""
subset(v::VertexLog, keep) = VertexLog(
    v.event[keep],
    v.process[keep],
    v.first[keep],
    v.position_mm[keep],
    v.energy_keV[keep],
    v.direction[keep],
)

"""
The incident photons' first scatterings, integrated along the run's ray: at `readout.depths` depths
t (one drawn in each equal stretch of the chord), a scattering by each process weighted by its
chance μ_process e^(−μ t) Δt (times the run's photons, as the tallies are sums over photons), each
scored with enough points that the integral costs about an eighth of the sampled scatterings.
"""
function first_scatter!(frames, run, phantom, media, readout, rng)
    beam = run.beam
    o = beam.origin_mm
    d = beam.direction ./ sqrt(sum(abs2, beam.direction))
    a, b = phantom.semi_axes_mm
    qx, qy = (o[1] - phantom.centre_mm[1]) / a, (o[2] - phantom.centre_mm[2]) / b
    ux, uy = d[1] / a, d[2] / b
    A, B, C = ux^2 + uy^2, qx * ux + qy * uy, qx^2 + qy^2 - 1
    B^2 - A * C > 0 || return frames
    t_in = max((-B - sqrt(B^2 - A * C)) / A, 0.0)
    chord = (-B + sqrt(B^2 - A * C)) / A - t_in
    chord > 0 || return frames

    E = run.energy_keV
    attenuation = media[1].attenuation
    μ = on_kev_grid(attenuation.total, E)
    M = readout.depths
    K = max(16, cld(readout.points * run.n, 8 * M))
    for (process, table) in ((VERTEX_COMPTON, attenuation.compton), (VERTEX_RAYLEIGH, attenuation.rayleigh))
        μ_process = on_kev_grid(table, E)
        depths = [(m - 1 + rand(rng)) * chord / M for m in 1:M]
        virtual = VertexLog(
            fill(Int32(0), M),
            fill(process, M),
            trues(M),
            [Float32.(o .+ (t_in + t) .* d) for t in depths],
            fill(Float32(E), M),
            fill(Float32.(d), M),
        )
        weights = [run.n * μ_process * exp(-μ * t) * chord / M for t in depths]
        forced_detection!(frames, virtual, phantom, media, readout, rng; weights, points = K, directions = K)
    end
    return frames
end

"""
Forced detection of every scattering in `vertices` onto every detector in `frames`, each weighted by
`weights` (1 if `nothing`), with `points` uniform points per detector and, at a Rayleigh
scattering, `directions` directions from its cone; each scattering by the tables of the medium it
happened in. The directions serve all the detectors; the points drawn on each detector, that one.
"""
function forced_detection!(
        frames,
        vertices::VertexLog,
        phantom,
        media,
        readout,
        rng;
        weights = nothing,
        points = readout.points,
        directions = readout.directions,
    )
    K = points
    μ_body = media[1].attenuation.total
    boosts = zeros(length(frames))
    for i in eachindex(vertices.event)
        r = vertices.position_mm[i]
        d0 = vertices.direction[i]
        E = Float64(vertices.energy_keV[i])
        process = vertices.process[i]
        tables = media[medium_at(phantom, r)].tables

        # Russian roulette, detector by detector, for a scattering that can add little: its weight (in
        # photons of the run) times its shortest way out (judged as though it were all body)
        μE = on_kev_grid(μ_body, E)
        weight = weights === nothing ? 1.0 : weights[i]
        for (j, frame) in enumerate(frames)
            worth = weight * maximum(exp(-μE * path_toward(phantom, r, l)) for l in frame.landmarks)
            boosts[j] = weight * (
                worth >= readout.roulette_below ? 1.0 :
                    rand(rng) < worth / readout.roulette_below ? readout.roulette_below / worth : 0.0
            )
        end
        any(>(0), boosts) || continue

        # At a Rayleigh scattering, directions drawn from its cone, followed to each detector arc
        L = process == VERTEX_RAYLEIGH ? directions : 0
        for _ in 1:L
            μ = sample_cosine(tables, process, E, rng)
            ω = rotate_by(d0, μ, rng)
            a = ω[1]^2 + ω[2]^2
            a > 1.0e-12 || continue
            E′ = process == VERTEX_COMPTON ? E / (1 + E / ELECTRON_REST_KEV * (1 - μ)) : E
            p_scatter = scattering_pdf(tables, process, E, μ)
            escape = escape_attenuation(phantom, media, r, ω, E′)
            for (j, frame) in enumerate(frames)
                boosts[j] > 0 || continue
                q = (r[1] - frame.f[1], r[2] - frame.f[2])
                b = q[1] * ω[1] + q[2] * ω[2]
                t = (-b + sqrt(b^2 - a * (q[1]^2 + q[2]^2 - frame.R^2))) / a
                h = (r[1] + t * ω[1], r[2] + t * ω[2], r[3] + t * ω[3])
                score!(frame, ω, h, t^2, E′, p_scatter, escape, K, L, boosts[j])
            end
        end

        # Points spread uniformly over each detector
        for (j, frame) in enumerate(frames)
            boosts[j] > 0 || continue
            for _ in 1:K
                γ = frame.γ_lo + rand(rng) * (frame.cols[end] - frame.cols[1])
                z = frame.z_lo + rand(rng) * (frame.rows[end] - frame.rows[1])
                sγ, cγ = sincosd(γ)
                h = (
                    frame.f[1] + frame.R * (cγ * frame.c[1] + sγ * frame.across[1]),
                    frame.f[2] + frame.R * (cγ * frame.c[2] + sγ * frame.across[2]),
                    z,
                )
                w = (h[1] - r[1], h[2] - r[2], h[3] - r[3])
                dist2 = w[1]^2 + w[2]^2 + w[3]^2
                ω = w ./ sqrt(dist2)
                μ = d0[1] * ω[1] + d0[2] * ω[2] + d0[3] * ω[3]
                E′ = process == VERTEX_COMPTON ? E / (1 + E / ELECTRON_REST_KEV * (1 - μ)) : E
                p_scatter = scattering_pdf(tables, process, E, μ)
                escape = escape_attenuation(phantom, media, r, ω, E′)
                score!(frame, ω, h, dist2, E′, p_scatter, escape, K, L, boosts[j])
            end
        end
    end
    return frames
end

reduce_run(run, readout::ForcedScatterReadout) = forced_scatter_tally(run, readout)
