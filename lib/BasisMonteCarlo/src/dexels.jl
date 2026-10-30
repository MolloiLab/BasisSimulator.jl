# =============================================================================
# Dexel readout of a photon-counting detector (dexel = detector element)
# =============================================================================
# The crystal is one continuous block; the dexels are the electrode pattern on its back face.
# Geant4 records where every bit of energy is left (a DepositLog). Everything after that is the
# detector's electronics, modelled here in Julia, so it can be changed without simulating again:
#
#   1. Each deposit becomes a cloud of electrons that drifts to the segmented electrode and
#      spreads as it goes (a Gaussian of width σ in x and y).
#   2. Each dexel collects the part of every cloud that lands on it: its share of the energy.
#   3. Each dexel's energy is blurred by the statistics of charge creation (Fano) and by the
#      electronic noise, then compared with the thresholds: one count per threshold crossed.
#
# The dexel grid: dexels come in groups (the cells of an anti-scatter grid; 1 × 1 when there is
# none), separated by a gap under the grid's walls. Along each axis dexel m (any integer) belongs
# to group c = fld(m, G) at position k = mod(m, G) in it; group 0 is centred on the origin. A
# dexel collects the charge that arrives over its own electrode and, at the edge of a group, over
# the near half of the gap. The readout records the `target` dexel of group 0 and its neighbours
# within ±h dexels (a 3 × 3 neighbourhood has h = 1).
# =============================================================================

"""
    DexelReadout(; pitch_x_mm, pitch_y_mm, neighborhood = 3, cloud_sigma_um = 30.0,
                   group = (1, 1), gap_mm = (0.0, 0.0), target = (0, 0))

The dexel grid and charge spreading of a photon-counting detector:

- `pitch_x_mm`, `pitch_y_mm`: the dexel size (the electrode pitch at the crystal);
- `neighborhood`: the side of the square of dexels recorded around the target one (odd; 3
  gives the target dexel and its 8 neighbours);
- `cloud_sigma_um`: the width σ of the Gaussian charge cloud each deposit becomes by the time it
  reaches the electrodes (the same σ for every deposit);
- `group`, `gap_mm`: dexels come in groups of `group[1] × group[2]` (x, y), separated by
  `gap_mm` (the cells of an anti-scatter grid, whose walls stand over the gaps). The default,
  one dexel per group and no gap, is a plain uniform grid;
- `target`: which dexel of the central group is the one photons are aimed at, counted from 0
  at the group's −x, −y corner. In a uniform grid it is the dexel centred on the origin.

[`dexel_beam`](@ref) gives the beam that illuminates the target dexel uniformly. See
[`dexel_response`](@ref) for how a run becomes dexel signals, and [`threshold_counts`](@ref) for
the noise and thresholds.
"""
Base.@kwdef struct DexelReadout
    pitch_x_mm::Float64
    pitch_y_mm::Float64
    neighborhood::Int = 3
    cloud_sigma_um::Float64 = 30.0
    group::Tuple{Int, Int} = (1, 1)
    gap_mm::Tuple{Float64, Float64} = (0.0, 0.0)
    target::Tuple{Int, Int} = (0, 0)
end

"""Centre of dexel `m` along an axis with dexel pitch `p`, groups of `G` and gaps `g`."""
function dexel_centre(m::Integer, p, G, g)
    group, k = fld(m, G), mod(m, G)
    return group * (G * p + g) + (k - (G - 1) / 2) * p
end

"""The stretch of the axis whose charge dexel `m` collects (its electrode, plus half a gap at a group edge)."""
function dexel_edges(m::Integer, p, G, g)
    centre, k = dexel_centre(m, p, G, g), mod(m, G)
    low = centre - p / 2 - (k == 0 ? g / 2 : 0.0)
    high = centre + p / 2 + (k == G - 1 ? g / 2 : 0.0)
    return low, high
end

"""Global dexel indices recorded along x and along y."""
function recorded_dexels(r::DexelReadout)
    h = r.neighborhood ÷ 2
    return (r.target[1] - h):(r.target[1] + h), (r.target[2] - h):(r.target[2] + h)
end

"""
    dexel_beam(readout; include_gap = false, kw...) -> AreaBeam

A beam whose photons land uniformly over the target dexel's electrode: the photons an air scan
delivers to it. With `include_gap = true` it also covers the half gap beside a dexel at the
edge of a group (where a grid wall stands, so those photons are stopped). Extra keywords
(tilts, `start_mm`) are passed to [`AreaBeam`](@ref).
"""
function dexel_beam(r::DexelReadout; include_gap::Bool = false, kw...)
    if include_gap
        x0, x1 = dexel_edges(r.target[1], r.pitch_x_mm, r.group[1], r.gap_mm[1])
        y0, y1 = dexel_edges(r.target[2], r.pitch_y_mm, r.group[2], r.gap_mm[2])
    else
        cx = dexel_centre(r.target[1], r.pitch_x_mm, r.group[1], r.gap_mm[1])
        cy = dexel_centre(r.target[2], r.pitch_y_mm, r.group[2], r.gap_mm[2])
        x0, x1 = cx - r.pitch_x_mm / 2, cx + r.pitch_x_mm / 2
        y0, y1 = cy - r.pitch_y_mm / 2, cy + r.pitch_y_mm / 2
    end
    return AreaBeam(;
        width_mm = x1 - x0,
        height_mm = y1 - y0,
        x_mm = (x0 + x1) / 2,
        y_mm = (y0 + y1) / 2,
        kw...,
    )
end

function Base.show(io::IO, r::DexelReadout)
    return print(
        io,
        "DexelReadout(", r.pitch_x_mm, " × ", r.pitch_y_mm, " mm dexels, ",
        r.neighborhood, " × ", r.neighborhood, " neighbourhood, σ_cloud = ", r.cloud_sigma_um, " µm)",
    )
end

"""
    DexelResponse

The signal of every dexel of a [`DexelReadout`](@ref)'s neighbourhood for every photon of a
run ([`dexel_response`](@ref)): `energy_keV[p, i, j]` is the energy photon `p` delivered to dexel
`(i, j)`, before noise, with `i` along x and `j` along y and the target dexel in the middle.
"""
struct DexelResponse
    energy_keV::Float64
    n::Int
    readout::DexelReadout
    detector::AbstractDetector
    dexel_energy_keV::Array{Float32, 3}
end

function Base.show(io::IO, r::DexelResponse)
    return print(
        io,
        "DexelResponse(",
        r.energy_keV,
        " keV × ",
        r.n,
        " photons, ",
        r.readout,
        ")",
    )
end

"""
Share of a Gaussian cloud centred at `x` (width `σ`) that lands between `a` and `b`.
"""
cloud_share(x, a, b, σ) = 0.5 * (erf((b - x) / (sqrt(2) * σ)) - erf((a - x) / (sqrt(2) * σ)))

"""
    dexel_response(run, readout) -> DexelResponse

Turn a run recorded with `record_deposits = true` into the energy each dexel of `readout`'s
neighbourhood collected from each photon: every deposit spreads into a Gaussian cloud of width
`readout.cloud_sigma_um`, and every dexel collects the share of each cloud that falls on it.
Charge that lands outside the neighbourhood is not recorded.
"""
function dexel_response(run::MonoenergeticRun, readout::DexelReadout)
    run.deposits === nothing && throw(
        ArgumentError("the run has no deposits: simulate it with record_deposits = true")
    )
    isodd(readout.neighborhood) || throw(ArgumentError("neighborhood must be odd"))

    n = readout.neighborhood
    σ = readout.cloud_sigma_um / 1000
    reach = 6σ                                  # beyond this a cloud's share is < 1e-9
    mx, my = recorded_dexels(readout)
    edges_x = [dexel_edges(m, readout.pitch_x_mm, readout.group[1], readout.gap_mm[1]) for m in mx]
    edges_y = [dexel_edges(m, readout.pitch_y_mm, readout.group[2], readout.gap_mm[2]) for m in my]
    x_min, x_max = first(edges_x)[1] - reach, last(edges_x)[2] + reach
    y_min, y_max = first(edges_y)[1] - reach, last(edges_y)[2] + reach

    energies = zeros(Float32, run.n, n, n)
    share_x = zeros(n)
    share_y = zeros(n)

    dep = run.deposits
    for k in eachindex(dep.event)
        x, y, e = dep.x_mm[k], dep.y_mm[k], dep.energy_keV[k]

        # Skip deposits whose cloud cannot reach the recorded dexels
        (x_min <= x <= x_max && y_min <= y <= y_max) || continue

        # The share of the cloud over each column (x) and each row (y) of dexels
        for i in 1:n
            share_x[i] = cloud_share(x, edges_x[i]..., σ)
            share_y[i] = cloud_share(y, edges_y[i]..., σ)
        end

        p = dep.event[k]
        for j in 1:n, i in 1:n
            energies[p, i, j] += e * share_x[i] * share_y[j]
        end
    end

    return DexelResponse(run.energy_keV, run.n, readout, run.detector, energies)
end

"""
    threshold_counts(response, thresholds_keV; electronic_noise_keV = 1.5, fano = 0.1,
                     pair_energy_eV = 4.43, seed = 1)
        -> (; per_dexel, total)

Counts per incident photon above each threshold. Each dexel's energy is first blurred by a
Gaussian of variance `electronic_noise_keV² + fano × pair_energy × E` (electronic noise, and the
statistics of the electron-hole pairs a deposit of energy E creates in CdTe), then every
threshold it exceeds adds one count.

- `per_dexel[i, j, t]`: mean counts in dexel `(i, j)` above threshold `t`;
- `total[t]`: the sum over the whole neighbourhood, which exceeds 1 when charge sharing makes
  one photon count in two dexels.

The noise is drawn from a generator seeded with `seed`, so results are reproducible.
"""
function threshold_counts(
        r::DexelResponse,
        thresholds_keV::AbstractVector;
        electronic_noise_keV::Real = 1.5,
        fano::Real = 0.1,
        pair_energy_eV::Real = 4.43,
        seed::Integer = 1,
    )
    rng = Xoshiro(seed)
    w = pair_energy_eV / 1000
    n, nx, ny = size(r.dexel_energy_keV)
    counts = zeros(nx, ny, length(thresholds_keV))

    for j in 1:ny, i in 1:nx, p in 1:n
        e = Float64(r.dexel_energy_keV[p, i, j])
        σ = sqrt(electronic_noise_keV^2 + fano * w * max(e, 0.0))
        measured = e + σ * randn(rng)
        for (t, threshold) in enumerate(thresholds_keV)
            measured >= threshold && (counts[i, j, t] += 1)
        end
    end

    per_dexel = counts ./ n
    total = vec(sum(per_dexel; dims = (1, 2)))
    return (; per_dexel, total)
end

"""
    dexel_spectrum(response, dexel = (0, 0); bin_keV = 0.5) -> (; energy_keV, density)

Histogram of the energy dexel `dexel` (offset from the target dexel: `(0, 0)` is the target,
`(1, 0)` its neighbour along +x) collected per incident photon, before noise, over the photons
that gave it anything; normalised per keV per incident photon.
"""
function dexel_spectrum(
        r::DexelResponse,
        dexel::Tuple{Int, Int} = (0, 0);
        bin_keV::Real = 0.5,
    )
    h = r.readout.neighborhood ÷ 2
    i, j = dexel[1] + h + 1, dexel[2] + h + 1
    nbins = ceil(Int, r.energy_keV / bin_keV) + 1
    counts = zeros(Int, nbins)
    for p in 1:r.n
        e = r.dexel_energy_keV[p, i, j]
        e > 0.05 || continue
        counts[min(nbins, floor(Int, e / bin_keV) + 1)] += 1
    end
    centers = ((1:nbins) .- 0.5) .* bin_keV
    return (energy_keV = collect(centers), density = counts ./ (r.n * bin_keV))
end

"""
    response_table(responses, thresholds_keV; electronic_noise_keV = 1.5, fano = 0.1,
                   pair_energy_eV = 4.43, seed = 1, metadata = Dict()) -> Dict{String, Any}

A detector response table from one [`DexelResponse`](@ref) per energy (a 3 × 3 neighbourhood),
in the layout of BasisSimulator's `cdte_response_v4` so it can be loaded the same way
(`BasisSimulator.load_mc_response`):

- `"energies_keV"` (n_E), `"thresholds_keV"` (n_t);
- `"R_total"[E, t]`: mean counts per photon above threshold `t` over the 3 × 3 dexels;
- `"R_perpixel"[E, i, j, t]`: the same per dexel (key name kept from the original table);
- `"Var_total"[E, t]`, `"Cov_total"[E, t, t′]`: variance and covariance of the 3 × 3 counts;
- `"Cov_pixpair"[E, a, b, t, t′]`: covariance between dexel `a`'s count above `t` and dexel `b`'s
  above `t′` (dexels numbered column-major over the 3 × 3);
- `"sigma_cloud_um"`, `"sigma_elec_keV"`, plus everything in `metadata` (provenance).

Noise is drawn as in [`threshold_counts`](@ref), from a generator seeded with `seed`.
"""
function response_table(
        responses::AbstractVector,
        thresholds_keV::AbstractVector;
        electronic_noise_keV::Real = 1.5,
        fano::Real = 0.1,
        pair_energy_eV::Real = 4.43,
        seed::Integer = 1,
        metadata::AbstractDict = Dict{String, Any}(),
    )
    n_E, n_t = length(responses), length(thresholds_keV)
    first(responses).readout.neighborhood == 3 || throw(ArgumentError("the table layout needs a 3 × 3 neighbourhood"))
    w = pair_energy_eV / 1000

    R_total = zeros(n_E, n_t)
    R_perpixel = zeros(n_E, 3, 3, n_t)
    Var_total = zeros(n_E, n_t)
    Cov_total = zeros(n_E, n_t, n_t)
    Cov_pixpair = zeros(n_E, 9, 9, n_t, n_t)

    for (e, r) in enumerate(responses)
        rng = Xoshiro(hash((seed, r.energy_keV)))

        # One row per photon: a 0/1 indicator for every (dexel, threshold)
        indicators = zeros(Float64, r.n, 9, n_t)
        for p in 1:r.n, a in 1:9
            i, j = mod1(a, 3), cld(a, 3)
            energy = Float64(r.dexel_energy_keV[p, i, j])
            spread = sqrt(electronic_noise_keV^2 + fano * w * max(energy, 0.0))
            measured = energy + spread * randn(rng)
            for t in 1:n_t
                indicators[p, a, t] = measured >= thresholds_keV[t] ? 1.0 : 0.0
            end
        end

        flat = reshape(indicators, r.n, 9 * n_t)
        μ = vec(sum(flat; dims = 1)) ./ r.n
        C = (flat' * flat) ./ r.n .- μ * μ'
        Cov_pixpair[e, :, :, :, :] = permutedims(reshape(C, 9, n_t, 9, n_t), (1, 3, 2, 4))

        totals = dropdims(sum(indicators; dims = 2); dims = 2)          # photons × thresholds
        R_total[e, :] = vec(sum(totals; dims = 1)) ./ r.n
        μ_total = R_total[e, :]
        Cov_total[e, :, :] = (totals' * totals) ./ r.n .- μ_total * μ_total'
        Var_total[e, :] = [Cov_total[e, t, t] for t in 1:n_t]
        for a in 1:9, t in 1:n_t
            R_perpixel[e, mod1(a, 3), cld(a, 3), t] = μ[a + 9 * (t - 1)]
        end
    end

    table = Dict{String, Any}(
        "energies_keV" => Float64[r.energy_keV for r in responses],
        "thresholds_keV" => Float64.(thresholds_keV),
        "R_total" => R_total,
        "R_perpixel" => R_perpixel,
        "Var_total" => Var_total,
        "Cov_total" => Cov_total,
        "Cov_pixpair" => Cov_pixpair,
        "sigma_cloud_um" => first(responses).readout.cloud_sigma_um,
        "sigma_elec_keV" => Float64(electronic_noise_keV),
    )
    merge!(table, Dict{String, Any}(string(k) => v for (k, v) in metadata))
    return table
end

# -----------------------------------------------------------------------------
# Energy-integrating dexels: sums only
# -----------------------------------------------------------------------------

"""
    DexelEnergyReadout(readout::DexelReadout)

An energy-integrating readout of `readout`'s neighbourhood (a scintillator array, whose light
stays in its dexel) that keeps only sums. Passed to [`simulate_sweep`](@ref) it reduces each run,
where it ran, to a [`DexelEnergy`](@ref) of a few numbers instead of every photon's signal.
"""
struct DexelEnergyReadout
    readout::DexelReadout
end

"""
    DexelEnergy

What an energy-integrating detector collected from `n` photons of `energy_keV`:

- per dexel of the neighbourhood (target in the middle): `dexel_keV[i, j]`, the energy collected
  summed over photons; `dexel2_keV2[i, j]`, the sum of its square; `dexel_hit[i, j]`, the
  photons that left any energy in it;
- over all the crystal: `crystal_keV`, `crystal2_keV2` and `crystal_hit`, the same sums for the
  energy left anywhere in the crystal;
- `interacted`: the photons that interacted in the crystal at all.

The metrics: [`interaction_probability`](@ref), [`energy_absorption_efficiency`](@ref) and
[`swank_factor`](@ref) (all the crystal), [`dexel_efficiency`](@ref) and [`dexel_swank`](@ref)
(one dexel).
"""
struct DexelEnergy
    energy_keV::Float64
    n::Int
    dexel_keV::Matrix{Float64}
    dexel2_keV2::Matrix{Float64}
    dexel_hit::Matrix{Int}
    crystal_keV::Float64
    crystal2_keV2::Float64
    crystal_hit::Int
    interacted::Int
end

"""
    dexel_energy(run, readout::DexelEnergyReadout) -> DexelEnergy

Reduce a run recorded with `record_deposits = true` to a [`DexelEnergy`](@ref).
"""
function dexel_energy(run::MonoenergeticRun, r::DexelEnergyReadout)
    signal = dexel_response(run, r.readout).dexel_energy_keV
    edep = run.edep_keV
    return DexelEnergy(
        run.energy_keV,
        run.n,
        dropdims(sum(Float64, signal; dims = 1); dims = 1),
        dropdims(sum(e -> Float64(e)^2, signal; dims = 1); dims = 1),
        dropdims(count(>(0), signal; dims = 1); dims = 1),
        sum(Float64, edep),
        sum(e -> Float64(e)^2, edep),
        count(>(0), edep),
        count(run.interacted),
    )
end

"""Mean energy one dexel (default: the target) collects per photon, divided by the photon energy."""
function dexel_efficiency(
        d::DexelEnergy,
        i::Integer = (size(d.dexel_keV, 1) + 1) ÷ 2,
        j::Integer = i,
    )
    return d.dexel_keV[i, j] / (d.n * d.energy_keV)
end

"""The Swank factor of one dexel's signal (default: the target)."""
function dexel_swank(
        d::DexelEnergy,
        i::Integer = (size(d.dexel_keV, 1) + 1) ÷ 2,
        j::Integer = i,
    )
    M0, M1, M2 = d.dexel_hit[i, j] / d.n, d.dexel_keV[i, j] / d.n, d.dexel2_keV2[i, j] / d.n
    return M1^2 / (M0 * M2)
end

interaction_probability(d::DexelEnergy) = d.interacted / d.n
energy_absorption_efficiency(d::DexelEnergy) = d.crystal_keV / (d.n * d.energy_keV)

function swank_factor(d::DexelEnergy)
    M0, M1, M2 = d.crystal_hit / d.n, d.crystal_keV / d.n, d.crystal2_keV2 / d.n
    return M1^2 / (M0 * M2)
end
