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
- `single`: per bin, the share of a dexel's counts no neighbour shares, in a flat field of this
  spectrum.

A photon of a class counts once in each dexel it is shared by, so each dexel's counts stay
Poisson while two neighbours covary by the class's rate, and a 2 × 2 digital sum counts a shared
photon twice: its variance exceeds its mean. The shares are this spectrum's; [`pcct_count!`](@ref)
keeps every dexel's mean whatever its own spectrum.
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
    pcct_count!(counts, expected, classes; seed = nothing, binning = 1, pileup = nothing) -> counts

Count every native dexel from its expected counts `expected[b][col, row, view]` (bins `b`) and
sum them digitally, `binning × binning` dexels per pixel, into `counts[b]`; the arrays may live
on any backend, and one kernel does it all there.

- The noise, photon class by photon class (`classes` from [`pcct_event_classes`](@ref);
  `nothing` counts the expected counts): for every pair of neighbours and class one Poisson count
  added to both, at the mean of the two dexels' estimates of its rate (at most either dexel's
  share), and each dexel's unshared counts, the rest of its expected counts. Each dexel's mean is
  its expected counts in every bin, whatever the bowtie or the patient has made of its spectrum.
  A neighbour beyond the detector's edge takes no share: the edge dexel keeps its own.
- The pile-up of each dexel at its own rate and spectrum, before the sum (`pileup`, or
  `nothing`): `(; S, rates, hardness, time_per_view)`, the family of [`pileup_family`](@ref),
  `S[:, :, k, m]` at counted rates `rates[k]` (evenly spaced from 0) and mean bins `hardness[m]`.
  A dexel's counted rate is its counts over `time_per_view`, its hardness the mean bin of its
  counts; its matrix is bilinear in the two (held at the family's ends).

Every draw has its own counter-based stream, keyed by `(seed, class, dexel, view)`: a seed fixes
the realisation on a backend whatever the thread count or launch order (`seed = nothing` draws a
fresh one), and the two dexels of a shared photon draw it alike.
"""
function pcct_count!(
        counts::AbstractVector,
        expected::AbstractVector,
        classes;
        seed = nothing,
        binning::Integer = 1,
        pileup = nothing,
    )
    nb = length(expected)
    nc, nr, nv = size(expected[1])
    length(counts) == nb || throw(DimensionMismatch("one counts array per bin"))
    size(counts[1]) == (nc ÷ binning, nr ÷ binning, nv) ||
        throw(DimensionMismatch("the counts must be the expected counts summed $binning × $binning"))
    T = eltype(counts[1])
    device(x) = copyto!(similar(counts[1], eltype(x), size(x)), x)
    base = seed === nothing ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    pairs = if classes === nothing
        nothing
    else
        (
            dx = device(Int32[p.dx for p in classes.pairs]),
            dy = device(Int32[p.dy for p in classes.pairs]),
            b = device(Int32[p.b for p in classes.pairs]),
            b2 = device(Int32[p.b2 for p in classes.pairs]),
            c1 = device(T[p.c1 for p in classes.pairs]),
            c2 = device(T[p.c2 for p in classes.pairs]),
            # a class's share of each dexel's shared counts in its bin: the rate's cap
            k1 = device(T[p.c1 / (1 - classes.single[p.b]) for p in classes.pairs]),
            k2 = device(T[p.c2 / (1 - classes.single[p.b2]) for p in classes.pairs]),
        )
    end
    pile = if pileup === nothing
        nothing
    else
        size(pileup.S, 3) == length(pileup.rates) && size(pileup.S, 4) == length(pileup.hardness) ||
            throw(DimensionMismatch("pcct_count!: one matrix per rate and hardness"))
        length(pileup.rates) >= 2 && length(pileup.hardness) >= 2 && issorted(pileup.hardness) ||
            throw(ArgumentError("pcct_count!: at least two rates and two increasing hardness nodes"))
        (
            S = device(T.(pileup.S)),
            dr = T(pileup.rates[2] - pileup.rates[1]),
            K = Int32(length(pileup.rates)),
            q = device(T.(pileup.hardness)),
            M = Int32(length(pileup.hardness)),
            scale = T(1 / pileup.time_per_view),
        )
    end
    _pcct_count!(Tuple(counts), Tuple(expected), pairs, pile, base, Int32(binning))
    return counts
end

function _pcct_count!(out::NTuple{NB}, λ::NTuple{NB}, pairs, pile, base, bf) where {NB}
    nc, nr = Int32.(size(λ[1])[1:2])
    pixels = CartesianIndices(size(out[1]))
    AK.foreachindex(out[1]) do idx
        cb, rb, v = Tuple(pixels[idx])
        acc = ntuple(_ -> zero(eltype(out[1])), Val(NB))
        for j in Int32(0):(bf - Int32(1)), i in Int32(0):(bf - Int32(1))
            c, r = (Int32(cb) - Int32(1)) * bf + Int32(1) + i, (Int32(rb) - Int32(1)) * bf + Int32(1) + j
            n = pairs === nothing ? ntuple(b -> @inbounds(λ[b][c, r, v]), Val(NB)) :
                _dexel_counts(λ, pairs, base, c, r, v, nc, nr)
            pile === nothing || (n = _pile_up(n, pile))
            acc = acc .+ n
        end
        map((o, a) -> (@inbounds o[cb, rb, v] = a; nothing), out, acc)
    end
    return out
end

@inline _add_at(n::NTuple{NB}, b, x) where {NB} = ntuple(i -> i == b ? n[i] + x : n[i], Val(NB))

# One dexel's counts in every bin: each class it shares with a forward neighbour (`+ (dx, dy)`,
# counted in bin `b`) and with a backward one (`− (dx, dy)`, in `b2`), then its unshared counts, the
# rest of its expected counts, so that its mean is its expected counts in every bin whatever its
# spectrum and its neighbours'. A shared photon is the draw of the pair's first dexel, so both
# dexels draw it alike; a dexel at the detector's edge keeps its missing neighbour's share.
@inline function _dexel_counts(λ::NTuple{NB}, pairs, base, c, r, v, nc, nr) where {NB}
    own = ntuple(b -> @inbounds(λ[b][c, r, v]), Val(NB))
    n = ntuple(_ -> zero(eltype(own)), Val(NB))
    shared = n
    for p in Int32(1):Int32(length(pairs.dx))
        dx, dy, b, b2 = @inbounds pairs.dx[p], pairs.dy[p], pairs.b[p], pairs.b2[p]
        cn, rn = c + dx, r + dy
        if 1 <= cn <= nc && 1 <= rn <= nr
            rate = _shared_rate(own[b], @inbounds(λ[b2][cn, rn, v]), pairs, p)
            n = _add_at(n, b, first(_poisson_draw(_DrawStream(base, NB + p, c, r, v), rate)))
            shared = _add_at(shared, b, rate)
        end
        cp, rp = c - dx, r - dy
        if 1 <= cp <= nc && 1 <= rp <= nr
            rate = _shared_rate(@inbounds(λ[b][cp, rp, v]), own[b2], pairs, p)
            n = _add_at(n, b2, first(_poisson_draw(_DrawStream(base, NB + p, cp, rp, v), rate)))
            shared = _add_at(shared, b2, rate)
        end
    end
    return _add_unshared(n, shared, own, base, c, r, v)
end

# A dexel's counts `n` from the photons it shares, plus its unshared counts in every bin: the rest
# of its expected counts `own`, the classes' rates `shared` taken off.
@inline _add_unshared(n::NTuple{NB}, shared, own, base, c, r, v) where {NB} = ntuple(
    b -> n[b] + first(_poisson_draw(_DrawStream(base, b, c, r, v), max(own[b] - shared[b], zero(own[b])))),
    Val(NB),
)

# The rate of class `p`'s photons shared by a dexel counting `λ1` in the class's first bin and its
# neighbour counting `λ2` in the second: the mean of the two dexels' estimates, `λ1 c1` and `λ2 c2`
# (equal in a flat field of the classes' spectrum), capped at each dexel's `λ k`, the class's share of
# its shared counts, so that no dexel's shared counts exceed its counts.
@inline _shared_rate(λ1, λ2, pairs, p) =
    @inbounds min(fma(λ1, pairs.c1[p], λ2 * pairs.c2[p]) / 2, λ1 * pairs.k1[p], λ2 * pairs.k2[p])

# A dexel's counts piled up at its own counted rate and spectrum: `S` bilinear in the rate and the
# mean bin between the family's matrices.
@inline function _pile_up(n::NTuple{NB, T}, pile) where {NB, T}
    tot = sum(n)
    tot > 0 || return n
    x = clamp(pile.scale * tot / pile.dr, zero(T), T(pile.K - 1))
    k = min(unsafe_trunc(Int32, x), pile.K - Int32(2))
    f = x - k
    q = _mean_bin(n)
    m = Int32(1)
    while m < pile.M - Int32(1) && q >= @inbounds(pile.q[m + 1])
        m += Int32(1)
    end
    g = @inbounds clamp((q - pile.q[m]) / (pile.q[m + 1] - pile.q[m]), zero(T), one(T))
    return _migrate(pile.S, n, k, m, f, g)
end

# `S` bilinear between rate nodes `k + 1, k + 2` (fraction `f`) and hardness nodes `m, m + 1`
# (fraction `g`), times the counts `n` (no reassigned variable is captured, so it infers on the GPU).
@inline function _migrate(S, n::NTuple{NB, T}, k, m, f, g) where {NB, T}
    return ntuple(i -> _migrate_row(S, n, i, k, m, f, g), Val(NB))
end
@inline _migrate_row(S, n::NTuple{NB, T}, i, k, m, f, g) where {NB, T} = sum(
    ntuple(
        j -> @inbounds(
            (
                (1 - g) * ((1 - f) * S[i, j, k + 1, m] + f * S[i, j, k + 2, m]) +
                    g * ((1 - f) * S[i, j, k + 1, m + 1] + f * S[i, j, k + 2, m + 1])
            ) * n[j]
        ),
        Val(NB),
    )
)

# Internal: the simulator's plumbing, reached as `BasisSimulator.pcct_count!` etc. (not exported).
