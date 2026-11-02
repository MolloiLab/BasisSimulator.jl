# The photon-counting detector, read at its native dexels: BasisMonteCarlo's Geant4 table of the
# NAEOTOM Alpha's CdTe (notebook 03, section 13), copied in as plain data so that BasisSimulator
# never loads BasisMonteCarlo. Per incident photon per native pitch area, flat field, cumulative
# above each of 16 thresholds, at each of the six positions a dexel can hold in its 4 × 6 grid
# cell (the anti-scatter grid, absorption, transported charge and electronic noise included):
#
# - `mean[type, E, threshold]`: the counts;
# - `pair[type, E, x, y, ti, tj]`: the counts of the dexel above `ti` times those of its neighbour
#   at offset `(x, y) − 2` above `tj`, from the same photon (offsets −1:1, across the fan then
#   along z; a mirrored position mirrors its type's offsets);
# - the pile-up inputs: the pulses each photon makes, their heights, the shaper and dead time.
#
# Three things are read from it: the response (`load_mc_response`, the cell's mean), the gain of
# each position in the cell (`pcct_position_gain`, the grid's shadow), and the event classes of
# the counting noise (`pcct_event_classes`, the photons two dexels share).

"""
    PCCT_NATIVE_TABLE_PATH

The native-dexel response of the NAEOTOM Alpha's CdTe detector (Geant4, BasisMonteCarlo notebook
03): the mean counts, the counts two neighbours share and the pile-up inputs of each of the six
positions in a grid cell. BasisSimulator's photon-counting response.
"""
const PCCT_NATIVE_TABLE_PATH = joinpath(@__DIR__, "naeotom_alpha_cdte_native_v1.jls")

const _PCCT_NATIVE_TABLE = Ref{Any}(nothing)
const _PCCT_NATIVE_TABLE_LOCK = ReentrantLock()

"""
    pcct_native_table() -> NamedTuple

The native table at [`PCCT_NATIVE_TABLE_PATH`](@ref), read once: `energies`, `thresholds`,
`mean[type, E, t]`, `pair[type, E, x, y, ti, tj]`, `type_index`, `mirror_x`, `mirror_y` (4 × 6,
one per dexel of a grid cell), `pitch_mm`, `electronic_noise_keV` and `pileup` (`energies`,
`pulses_per_photon`, `heights[E, h]`, `edges`, `dead_time_ns`, `shaper_fwhm_ns`).
"""
function pcct_native_table()
    return lock(_PCCT_NATIVE_TABLE_LOCK) do
        _PCCT_NATIVE_TABLE[] === nothing || return _PCCT_NATIVE_TABLE[]
        d = open(deserialize, PCCT_NATIVE_TABLE_PATH)
        index = Dict(l => i for (i, l) in enumerate(String.(d["type_labels"])))
        p = d["pileup"]
        _PCCT_NATIVE_TABLE[] = (
            energies = Float64.(d["energies_keV"]),
            thresholds = Float64.(d["thresholds_keV"]),
            mean = Float64.(d["mean"]),
            pair = Float64.(d["pair"]),
            type_index = [index[s] for s in d["type_map"]],
            mirror_x = Int.(d["mirror_x"]),
            mirror_y = Int.(d["mirror_y"]),
            pitch_mm = Float64.(d["pitch_mm"]),
            electronic_noise_keV = Float64(d["electronic_noise_keV"]),
            pileup = (
                energies = Float64.(p["energies_keV"]),
                pulses_per_photon = Float64.(p["pulses_per_photon"]),
                heights = Float64.(p["pulse_heights_per_photon"]),
                edges = Float64.(p["pulse_height_edges_keV"]),
                dead_time_ns = Float64(p["dead_time_ns"]),
                shaper_fwhm_ns = Float64(p["shaper_fwhm_ns"]),
            ),
        )
    end
end

# The response as `load_mc_response` returns it: the cell's mean dexel (the grid included) and its
# own second moment, the native dexel's counts being Poisson.
function _native_table_response(t)
    nE, nT = length(t.energies), length(t.thresholds)
    R = dropdims(sum(t.mean[t.type_index[:], :, :]; dims = 1); dims = 1) ./ length(t.type_index)
    second = dropdims(sum(t.pair[t.type_index[:], :, 2, 2, :, :]; dims = 1); dims = 1) ./ length(t.type_index)
    Cov = second .- reshape(R, nE, nT, 1) .* reshape(R, nE, 1, nT)
    return MCResponseData(round.(Int, t.energies), round.(Int, t.thresholds), R, Cov)
end

# The scanner's thresholds among the table's: the table is cumulative at its own 16.
function _table_threshold_index(t, thresholds)
    idx = [findfirst(==(Float64(x)), t.thresholds) for x in thresholds]
    any(isnothing, idx) && throw(
        ArgumentError(
            "the photon-counting response is tabulated at the thresholds $(t.thresholds) keV; " *
                "$(thresholds) are not all among them"
        )
    )
    return Int.(idx)
end

# The spectrum's weights spread linearly onto the table's energy grid.
function _table_weights(grid::AbstractVector, energies::AbstractVector, weights::AbstractVector)
    w = zeros(length(grid))
    for (E, x) in zip(Float64.(energies), Float64.(weights))
        x == 0 && continue
        if E <= grid[1]
            w[1] += x
        elseif E >= grid[end]
            w[end] += x
        else
            i = searchsortedlast(grid, E)
            f = (E - grid[i]) / (grid[i + 1] - grid[i])
            w[i] += (1 - f) * x
            w[i + 1] += f * x
        end
    end
    return w ./ max(sum(w), eps())
end

# Linear interpolation of `A[type, E, …]` along E at one energy.
function _interp_energy(t, E::Real)
    grid = t.energies
    E <= grid[1] && return 1, 1, 0.0
    E >= grid[end] && return length(grid), length(grid), 0.0
    i = searchsortedlast(grid, E)
    return i, i + 1, (E - grid[i]) / (grid[i + 1] - grid[i])
end

"""
    pcct_position_gain(energies) -> Array{Float64, 3}

The counts above the lowest table threshold of each dexel of a 4 × 6 grid cell relative to the
cell's mean, at each energy: `gain[k, l, e]` for the dexel `k` columns and `l` rows into its cell.
The grid shadows the dexels by their position; the response is the cell's mean, so this map
carries the difference, multiplied into each native dexel's source transmission.
"""
function pcct_position_gain(energies::AbstractVector)
    t = pcct_native_table()
    gain = ones(4, 6, length(energies))
    for (e, E) in pairs(Float64.(energies))
        i, j, f = _interp_energy(t, E)
        m = (1 - f) .* t.mean[:, i, 1] .+ f .* t.mean[:, j, 1]
        cell = sum(m[t.type_index]) / length(t.type_index)
        cell > 1.0e-9 || continue
        gain[:, :, e] .= m[t.type_index] ./ cell
    end
    return gain
end

"""
    pcct_event_classes(thresholds, energies, weights) -> NamedTuple

The photons two neighbouring native dexels share, as Poisson event classes for the counting
noise, from the table's pair moments averaged over a grid cell and weighted by the spectrum:

- `pairs`: one class per neighbour offset `(dx, dy)` (across the fan, along z: the four forward
  ones, `(1, 0)`, `(0, 1)`, `(1, 1)`, `(1, −1)`) and pair of bins `(b, b′)`, with `c1` and `c2`, the
  shared counts per count of the first dexel in bin `b` and of the second in bin `b′`;
- `single`: per bin, the share of a dexel's counts no neighbour shares.

A photon of a class counts once in each dexel it is shared by, so each dexel's counts stay
Poisson while two neighbours covary by the class's rate, and a 2 × 2 digital sum counts a shared
photon twice: its variance exceeds its mean.
"""
function pcct_event_classes(thresholds::AbstractVector, energies::AbstractVector, weights::AbstractVector)
    t = pcct_native_table()
    ti = _table_threshold_index(t, thresholds)
    nb = length(ti)
    D = zeros(nb, length(t.thresholds))                  # cumulative → differential
    for b in 1:nb
        D[b, ti[b]] = 1
        b < nb && (D[b, ti[b + 1]] = -1)
    end
    w = _table_weights(t.energies, energies, weights)
    M = zeros(nb)
    offsets = [(dx, dy) for dx in -1:1 for dy in -1:1 if (dx, dy) != (0, 0)]
    P = Dict(o => zeros(nb, nb) for o in offsets)
    for k in 1:4, l in 1:6
        ty, mx, my = t.type_index[k, l], t.mirror_x[k, l], t.mirror_y[k, l]
        for (e, x) in pairs(w)
            x == 0 && continue
            M .+= x .* (D * @view(t.mean[ty, e, :]))
            for (dx, dy) in offsets
                P[(dx, dy)] .+= x .* (D * @view(t.pair[ty, e, mx * dx + 2, my * dy + 2, :, :]) * transpose(D))
            end
        end
    end
    shared_pairs = NamedTuple{(:dx, :dy, :b, :b2, :c1, :c2), Tuple{Int, Int, Int, Int, Float64, Float64}}[]
    shared = zeros(nb)
    for (dx, dy) in ((1, 0), (0, 1), (1, 1), (1, -1))
        S = (P[(dx, dy)] .+ transpose(P[(-dx, -dy)])) ./ 2   # the same photons, seen from either side
        for b in 1:nb, b2 in 1:nb
            (M[b] > 0 && M[b2] > 0 && S[b, b2] > 1.0e-7 * max(M[b], M[b2])) || continue
            push!(shared_pairs, (dx = dx, dy = dy, b = b, b2 = b2, c1 = S[b, b2] / M[b], c2 = S[b, b2] / M[b2]))
            shared[b] += S[b, b2] / M[b]
            shared[b2] += S[b, b2] / M[b2]
        end
    end
    all(shared .< 1) || error("pcct_event_classes: the shared counts exceed the counts ($(shared))")
    return (pairs = shared_pairs, single = 1 .- shared)
end

"""
    pcct_count!(counts, expected, classes; seed = nothing)

Draw the counts of every native dexel from its expected counts `expected[b][col, row, view]`
(bins `b`), photon by photon class: each dexel's unshared counts, and for every pair of
neighbours and class of [`pcct_event_classes`](@ref) one Poisson count added to both. A
neighbour beyond the detector's edge takes no share: the edge dexel keeps its own. Views are
drawn in parallel, each from its own stream seeded by `(seed, view)`, so a seed fixes the
realisation whatever the thread count; `seed = nothing` draws a fresh one.
"""
function pcct_count!(counts::AbstractVector, expected::AbstractVector, classes; seed = nothing)
    nb = length(expected)
    nc, nr, nv = size(expected[1])
    base = seed === nothing ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    Threads.@threads for v in 1:nv
        rng = Random.Xoshiro(hash((base, v)))
        for b in 1:nb
            λ, out, s = expected[b], counts[b], classes.single[b]
            @inbounds for r in 1:nr, c in 1:nc
                out[c, r, v] = _poisson_sample(rng, Float64(λ[c, r, v]) * s)
            end
        end
        for p in classes.pairs
            λ1, λ2, o1, o2 = expected[p.b], expected[p.b2], counts[p.b], counts[p.b2]
            @inbounds for r in 1:nr, c in 1:nc
                cn, rn = c + p.dx, r + p.dy
                if 1 <= cn <= nc && 1 <= rn <= nr
                    x = _poisson_sample(rng, (λ1[c, r, v] * p.c1 + λ2[cn, rn, v] * p.c2) / 2)
                    o1[c, r, v] += x
                    o2[cn, rn, v] += x
                else
                    o1[c, r, v] += _poisson_sample(rng, λ1[c, r, v] * p.c1)
                end
                # the second dexel of a pair whose first lies beyond the edge
                cp, rp = c - p.dx, r - p.dy
                (1 <= cp <= nc && 1 <= rp <= nr) ||
                    (o2[c, r, v] += _poisson_sample(rng, λ2[c, r, v] * p.c2))
            end
        end
    end
    return counts
end

export PCCT_NATIVE_TABLE_PATH, pcct_native_table, pcct_position_gain, pcct_event_classes, pcct_count!
