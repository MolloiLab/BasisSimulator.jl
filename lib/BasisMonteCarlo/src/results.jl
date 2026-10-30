# =============================================================================
# Runs and the quantities measured from them
# =============================================================================
# A run keeps every photon's record, not just averages: the energy it left in the detector,
# whether it interacted, and everything that escaped. Every metric below is then a one-line
# statistic of those records, and any new one can be added later without simulating again.
# =============================================================================

"""
    MonoenergeticRun

The record of `n` photons of `energy_keV` fired into `detector`
([`simulate_monoenergetic`](@ref)).

- `edep_keV[i]`: energy photon `i` left in the detector (0 if it passed straight through or
  everything it set moving escaped);
- `interacted[i]`: whether photon `i` interacted at all (photoelectric, Compton or Rayleigh);
- `escapes`: one row per particle that left the detector: `event`, `kind`
  (`:transmitted`, `:scattered`, `:fluorescence`, `:bremsstrahlung`, `:electron`), `energy_keV`,
  `backward` (left through the entrance face's side, travelling against the beam), `position_mm`
  (where it left) and `direction` (where it was going, a unit vector);
- `deposits`: with `record_deposits = true`, a [`DepositLog`](@ref) of where every bit of energy
  was left (photon, position in mm, energy in keV), which a dexel readout needs; otherwise
  `nothing`;
- `vertices`: with `record_vertices = true`, a [`VertexLog`](@ref) of every Compton and Rayleigh
  scattering of a photon; otherwise `nothing`;
- `photon_energy_keV`: each photon's energy when they were drawn from a [`Spectrum`](@ref)
  ([`simulate_spectrum`](@ref); `energy_keV` is then the spectrum's mean); otherwise `nothing`.
"""
struct MonoenergeticRun
    energy_keV::Float64
    n::Int
    detector::AbstractDetector
    beam::Beam
    physics::PhysicsSettings
    seed::Int
    edep_keV::Vector{Float32}
    interacted::BitVector
    escapes::NamedTuple{
        (:event, :kind, :energy_keV, :backward, :position_mm, :direction),
        Tuple{
            Vector{Int32},
            Vector{Symbol},
            Vector{Float32},
            BitVector,
            Vector{NTuple{3, Float32}},
            Vector{NTuple{3, Float32}},
        },
    }
    deposits::Union{Nothing, DepositLog}
    vertices::Union{Nothing, VertexLog}
    photon_energy_keV::Union{Nothing, Vector{Float32}}
end

function Base.show(io::IO, r::MonoenergeticRun)
    qe = round(quantum_efficiency(r); digits = 4)
    eae = round(energy_absorption_efficiency(r); digits = 4)
    return print(
        io,
        "MonoenergeticRun(",
        r.energy_keV,
        " keV × ",
        r.n,
        " photons → ",
        r.detector,
        ": QE = ",
        qe,
        ", EAE = ",
        eae,
        ")",
    )
end

"""
    simulate_monoenergetic(detector, energy_keV; n = 100_000, beam = PencilBeam(),
                           physics = PhysicsSettings(), seed = 1, record_deposits = false,
                           record_vertices = false)
        -> MonoenergeticRun
    simulate_monoenergetic(detector, energies_keV::AbstractVector; kw...) -> Vector{MonoenergeticRun}

Fire `n` photons of each energy into `detector` and keep every photon's record. The same
`seed` and energy always give the same photons. `beam` is a [`PencilBeam`](@ref) or an
[`AreaBeam`](@ref). With `record_deposits = true` the run also keeps where every bit of energy
was left (for a dexel readout; about 20 bytes per deposit, a few dozen deposits per photon); with
`record_vertices = true`, every photon scattering (for forced detection).
The first call in a Julia session starts Geant4 (a few seconds) and fixes `physics` for the
session.
"""
function simulate_monoenergetic(
        detector::AbstractDetector,
        energy_keV::Real;
        n::Integer = 100_000,
        beam::Beam = PencilBeam(),
        physics::PhysicsSettings = PhysicsSettings(),
        seed::Integer = 1,
        record_deposits::Bool = false,
        record_vertices::Bool = false,
    )
    e = engine(detector, physics)
    set_beam!(e, energy_keV, beam, seed)
    return run_photons(
        e,
        detector,
        Float64(energy_keV),
        n,
        beam,
        physics,
        seed,
        record_deposits,
        record_vertices,
    )
end

"""Fire `n` photons with the beam already set, and collect the run's records."""
function run_photons(e, detector, energy_keV, n, beam, physics, seed, record_deposits, record_vertices = false)
    d = reset!(getSIMdata(e.app); recording_deposits = record_deposits, recording_vertices = record_vertices)
    beamOn(e.app, Int(n))

    escapes = (
        event = copy(d.escape_event),
        kind = Symbol[ESCAPE_KINDS[k] for k in d.escape_kind],
        energy_keV = copy(d.escape_energy),
        backward = copy(d.escape_backward),
        position_mm = copy(d.escape_position),
        direction = copy(d.escape_direction),
    )
    deposits = record_deposits ? copy(d.deposits) : nothing
    d.recording_deposits = false
    empty!(d.deposits)
    vertices = record_vertices ? copy(d.vertices) : nothing
    d.recording_vertices = false
    empty!(d.vertices)
    photon_energies = isempty(d.photon_energies) ? nothing : copy(d.photon_energies)

    return MonoenergeticRun(
        Float64(energy_keV),
        Int(n),
        detector,
        beam,
        physics,
        Int(seed),
        copy(d.edeps),
        copy(d.interacted_flags),
        escapes,
        deposits,
        vertices,
        photon_energies,
    )
end

# Several energies: one run each, spread over the workers of `start_workers` when there are any
simulate_monoenergetic(detector::AbstractDetector, energies::AbstractVector; kw...) =
    simulate_sweep([(detector, E) for E in energies]; kw...)

"""
    TrackRecord

Every step of every particle in a handful of photons ([`record_tracks`](@ref)), for drawing:
`steps` (a vector of [`TrackStep`](@ref)) and `edep_keV` per photon.
"""
struct TrackRecord
    energy_keV::Float64
    detector::AbstractDetector
    beam::Beam
    steps::Vector{TrackStep}
    edep_keV::Vector{Float32}
end

"""
    record_tracks(detector, energy_keV; n = 30, beam = PencilBeam(),
                  physics = PhysicsSettings(), seed = 1) -> TrackRecord

Fire `n` photons and keep every step of every particle, to see what the Monte Carlo is
doing: where photons interact, which way they scatter, where the fluorescence goes. Slow per
photon; meant for tens of photons, not thousands.
"""
function record_tracks(
        detector::AbstractDetector, energy_keV::Real; n::Integer = 30,
        beam::Beam = PencilBeam(),
        physics::PhysicsSettings = PhysicsSettings(), seed::Integer = 1
    )
    e = engine(detector, physics)
    set_beam!(e, energy_keV, beam, seed)
    d = reset!(getSIMdata(e.app); recording = true)
    beamOn(e.app, Int(n))
    d.recording = false
    return TrackRecord(Float64(energy_keV), detector, beam, copy(d.steps), copy(d.edeps))
end

# -----------------------------------------------------------------------------
# Metrics. All are per incident photon unless stated.
# -----------------------------------------------------------------------------

"""
    interaction_probability(run)

Fraction of photons that interacted in the detector at all. Physically this is
`1 − exp(−μd)`, so it must match [`beer_lambert_efficiency`](@ref) to within the
statistical error and the difference between Geant4's and XCOM's cross sections: the first
check that the geometry, material and beam are what they should be.
"""
interaction_probability(r::MonoenergeticRun) = count(r.interacted) / r.n

"""
    quantum_efficiency(run; threshold_keV = 0)

Fraction of photons that left more than `threshold_keV` in the detector: a counting
detector's detection probability with that threshold. With the default threshold it is
slightly below [`interaction_probability`](@ref): a photon that only Rayleigh-scatters out
deposits nothing.
"""
quantum_efficiency(r::MonoenergeticRun; threshold_keV::Real = 0) =
    count(>(threshold_keV), r.edep_keV) / r.n

"""
    energy_absorption_efficiency(run)

Mean energy left in the detector divided by the photon energy: what an energy-integrating
(scintillator) detector's signal is proportional to. The quantity in BasisSimulator's
efficiency LUTs. Below a K-edge it is close to the interaction probability; above one it drops,
because every Kα photon that escapes carries its energy away.
"""
energy_absorption_efficiency(r::MonoenergeticRun) = mean(Float64, r.edep_keV) / r.energy_keV

"""
    swank_factor(run)

The Swank factor `I = M₁² / (M₀ M₂)`, with `Mₙ` the n-th moment of the absorbed-energy
distribution of the photons that deposited anything. It is 1 when every detected photon
deposits the same energy, and falls as the absorbed energies spread out (K-escape, Compton
escape); an energy-integrating detector's DQE(0) is the quantum efficiency times `I`.
"""
function swank_factor(r::MonoenergeticRun)
    M0 = count(>(0), r.edep_keV) / r.n
    M1 = mean(Float64, r.edep_keV)
    M2 = mean(e -> Float64(e)^2, r.edep_keV)
    return M1^2 / (M0 * M2)
end

"""
    photopeak_fraction(run; tolerance_keV = 0.5)

Fraction of photons that left their full energy (within `tolerance_keV`) in the detector.
"""
photopeak_fraction(r::MonoenergeticRun; tolerance_keV::Real = 0.5) =
    count(>=(r.energy_keV - tolerance_keV), r.edep_keV) / r.n

"""
    absorbed_energy_spectrum(run; bin_keV = 0.5) -> (; energy_keV, density)

Histogram of the energy left per photon, over the photons that left any, normalised to
probability per keV per incident photon (so it integrates to the quantum efficiency). The
full-energy peak, the K-escape peaks (at `E − Kα`) and the Compton continuum are read off it.
"""
function absorbed_energy_spectrum(r::MonoenergeticRun; bin_keV::Real = 0.5)
    nbins = ceil(Int, r.energy_keV / bin_keV) + 1
    counts = zeros(Int, nbins)
    for e in r.edep_keV
        e > 0 || continue
        counts[min(nbins, floor(Int, e / bin_keV) + 1)] += 1
    end
    centers = ((1:nbins) .- 0.5) .* bin_keV
    return (energy_keV = collect(centers), density = counts ./ (r.n * bin_keV))
end

"""
    escape_spectrum(run; kind = :fluorescence, backward = nothing, bin_keV = 0.5)

Histogram of the energies of escaping particles of `kind` (see [`MonoenergeticRun`](@ref)),
per incident photon per keV. `backward = true` keeps only those leaving against the beam,
`false` only those leaving with it, `nothing` both.
"""
function escape_spectrum(
        r::MonoenergeticRun; kind::Symbol = :fluorescence, backward = nothing,
        bin_keV::Real = 0.5
    )
    nbins = ceil(Int, r.energy_keV / bin_keV) + 1
    counts = zeros(Int, nbins)
    for i in eachindex(r.escapes.kind)
        r.escapes.kind[i] === kind || continue
        backward === nothing || r.escapes.backward[i] == backward || continue
        counts[min(nbins, floor(Int, r.escapes.energy_keV[i] / bin_keV) + 1)] += 1
    end
    centers = ((1:nbins) .- 0.5) .* bin_keV
    return (energy_keV = collect(centers), density = counts ./ (r.n * bin_keV))
end

"""
    fluorescence_lines(run; min_fraction = 0.02, width_keV = 0.4) -> Vector of (energy_keV, element, line, fraction)

The characteristic x-ray lines that escaped in `run`, read off the escaping fluorescence
photons themselves. The most common energy is taken as a line and every escaped photon within
`width_keV` of it is assigned to it (so Kα₁ and Kα₂ make one line, at their mean); this repeats
until the next line would carry less than `min_fraction` of the escaped fluorescence. Each line
is named from the material's K-shells: within 0.6 keV of an element's Kα energy it is that
element's `"Kα"`, between its Kα energy and its K-edge it is its `"Kβ"`. Every escape peak in
the absorbed-energy spectrum sits at the photon energy minus one of these lines. Sorted by
energy.
"""
function fluorescence_lines(
        r::MonoenergeticRun;
        min_fraction::Real = 0.02,
        width_keV::Real = 0.4,
    )
    T = NamedTuple{
        (:energy_keV, :element, :line, :fraction),
        Tuple{Float64, String, String, Float64},
    }
    remaining = Float64[
        r.escapes.energy_keV[i] for i in eachindex(r.escapes.kind)
            if r.escapes.kind[i] === :fluorescence
    ]
    total = length(remaining)
    edges = k_edges(r.detector.material)
    lines = T[]
    while !isempty(remaining)
        bins = Dict{Int, Int}()
        for e in remaining
            b = round(Int, e / 0.1)
            bins[b] = get(bins, b, 0) + 1
        end
        peak = 0.1 * first(argmax(last, collect(bins)))
        members = filter(e -> abs(e - peak) <= width_keV, remaining)
        fraction = length(members) / total
        fraction >= min_fraction || break
        filter!(e -> abs(e - peak) > width_keV, remaining)
        energy = mean(members)
        name, line = "?", "L or other"
        for k in edges
            if abs(energy - k.kalpha_keV) <= 0.6
                name, line = k.element, "Kα"
            elseif k.kalpha_keV < energy < k.edge_keV
                name, line = k.element, "Kβ"
            end
        end
        push!(lines, T((energy, name, line, fraction)))
    end
    return sort!(lines; by = l -> l.energy_keV)
end

"""
    energy_budget(run) -> NamedTuple

Where the incident energy went, as fractions of `n × E`: `absorbed`, and for each escape kind
the energy that left (`transmitted`, `scattered`, `fluorescence`, `bremsstrahlung`,
`electron`), plus `total`. Energy is conserved, so `total` is 1 up to the energy below Geant4's
tracking cuts (binding energies of outer shells left without a particle): a bookkeeping check
on the whole simulation.
"""
function energy_budget(r::MonoenergeticRun)
    incident = r.n * r.energy_keV
    absorbed = sum(Float64, r.edep_keV) / incident
    escaped = Dict(k => 0.0 for k in ESCAPE_KINDS)
    for i in eachindex(r.escapes.kind)
        escaped[r.escapes.kind[i]] += r.escapes.energy_keV[i]
    end
    parts = NamedTuple{ESCAPE_KINDS}(Tuple(escaped[k] / incident for k in ESCAPE_KINDS))
    return merge((absorbed = absorbed,), parts, (total = absorbed + sum(parts),))
end

"""
    standard_error(run, metric)

One-sigma statistical uncertainty of `metric(run)` from the finite number of photons, for
`metric` one of [`interaction_probability`](@ref), [`quantum_efficiency`](@ref),
[`photopeak_fraction`](@ref) (binomial) or [`energy_absorption_efficiency`](@ref) (standard
error of the mean).
"""
function standard_error(r::MonoenergeticRun, metric)
    if metric === energy_absorption_efficiency
        return sqrt(var(Float64.(r.edep_keV)) / r.n) / r.energy_keV
    end
    p = metric(r)
    return sqrt(p * (1 - p) / r.n)
end

"""
    interpolate_linear(x, y, xq)

Linear interpolation of the curve `(x, y)` (with `x` sorted ascending) at `xq`, clamped to the
end values outside the range. Used to compare a simulated curve with reference points read at
other energies.
"""
function interpolate_linear(x::AbstractVector, y::AbstractVector, xq::Real)
    xq <= first(x) && return float(first(y))
    xq >= last(x) && return float(last(y))
    i = searchsortedlast(x, xq)
    w = (xq - x[i]) / (x[i + 1] - x[i])
    return (1 - w) * y[i] + w * y[i + 1]
end
