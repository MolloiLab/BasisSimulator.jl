# =============================================================================
# Dexel-to-dexel defects and the corrections calibrated for them
# =============================================================================
#
# No two dexels are alike. An energy-integrating dexel has its own gain and offset; a
# photon-counting dexel its own gain and threshold offsets. Each scanner carries the spreads
# (`gain_spread`, `offset_spread_keV`, `threshold_spread_keV`) and the seed of its fixed pattern
# (`defect_seed`); the ideal detector is a copy with the spreads at zero. Every defect has its
# correction, calibrated from simulated calibration scans, never from the truth:
#
# - `calibrate_flat_field` → `apply_flat_field!`: air scans (two currents for an energy
#   integrator, so its offset separates from its gain);
# - `calibrate_pileup` → `apply_pileup!`: air scans at rising tube currents, the pile-up-free
#   counts extrapolated linearly from the lowest.
# =============================================================================

# A detector's fixed pattern: `n` independent standard-normal maps of `nc × nr` dexels, the same
# for every scan of the scanner (its `defect_seed`).
function _dexel_pattern(seed::Integer, n::Integer, nc::Integer, nr::Integer)
    rng = Random.Xoshiro(hash((:dexel_pattern, seed)))
    return randn(rng, nc, nr, n)
end

_has_defects(s::EICTScanner) = s.gain_spread > 0 || s.offset_spread_keV > 0
_has_defects(s::PCCTScanner) = s.gain_spread > 0 || s.threshold_spread_keV > 0

# The energy integrator's gain and offset (in units of E_ref) of each dexel.
function _eict_dexel_defects(s::EICTScanner, nc, nr, E_ref)
    z = _dexel_pattern(s.defect_seed, 2, nc, nr)
    return 1 .+ s.gain_spread .* z[:, :, 1], (s.offset_spread_keV / E_ref) .* z[:, :, 2]
end

# A photon-counting dexel's counts in each bin relative to the ideal dexel's, for one spectrum:
# its thresholds `(t + δ) / g` read off the table's cumulative counts, interpolated in threshold.
function _pcct_defect_factors(s::PCCTScanner, nc, nr, energies, weights)
    t = pcct_native_table()
    thr = Float64.(s.energy_thresholds)
    nb = length(thr)
    w = _table_weights(t.energies, energies, weights)
    cell = dropdims(sum(t.mean[t.type_index[:], :, :]; dims = 1); dims = 1) ./ length(t.type_index)
    C = transpose(w) * cell                                     # spectrum-weighted cumulative, per table threshold
    cum(x) = begin                                              # linear in threshold, extrapolated at the ends
        i = clamp(searchsortedlast(t.thresholds, x), 1, length(t.thresholds) - 1)
        f = (x - t.thresholds[i]) / (t.thresholds[i + 1] - t.thresholds[i])
        max((1 - f) * C[i] + f * C[i + 1], 0.0)
    end
    bin(c, b) = c[b] - (b < nb ? c[b + 1] : 0.0)
    nominal = [cum(x) for x in thr]
    z = _dexel_pattern(s.defect_seed, nb + 1, nc, nr)
    out = ones(nc, nr, nb)
    for r in 1:nr, c in 1:nc
        g = 1 + s.gain_spread * z[c, r, nb + 1]
        shifted = [cum((thr[k] + s.threshold_spread_keV * z[c, r, k]) / g) for k in 1:nb]
        for b in 1:nb
            n0 = bin(nominal, b)
            out[c, r, b] = n0 > 0 ? bin(shifted, b) / n0 : 1.0
        end
    end
    return out
end

# The photon-counting defects at the projection's resolution: the factors in air and behind
# 20 cm of water, and that water's line integral; a ray's factors are interpolated between them
# by its own line integral (a dexel's threshold offsets weigh a hardened spectrum differently).
function _pcct_dexel_defects(s::PCCTScanner, nc, nr, energies, weights)
    _has_defects(s) || return nothing
    μ = [compute_μ_at_energy(XA.Materials.water, Float64(E)) for E in energies]
    hard = Float64.(weights) .* exp.(-20 .* μ)
    p_hard = -log(sum(hard) / sum(weights))
    return (
        air = _pcct_defect_factors(s, nc, nr, energies, weights),
        hard = _pcct_defect_factors(s, nc, nr, energies, hard),
        p_hard = p_hard,
    )
end

# Apply them to expected counts `[b][c, r, v]` against the air counts `I0[c, r, b]`.
function _apply_pcct_defects!(expected, I0, d)
    nb = length(expected)
    for v in axes(expected[1], 3), r in axes(expected[1], 2), c in axes(expected[1], 1)
        p = -log(max(sum(expected[b][c, r, v] for b in 1:nb), 1.0e-30) / max(sum(@view I0[c, r, :]), 1.0e-30))
        x = clamp(p / d.p_hard, 0.0, 2.0)
        for b in 1:nb
            expected[b][c, r, v] *= d.air[c, r, b] + x * (d.hard[c, r, b] - d.air[c, r, b])
        end
    end
    return expected
end

_air_phantom() = Phantom(zeros(UInt8, 4, 4, 2), Dict{Int, Any}(0 => XA.Materials.air), (0.05, 0.05, 0.05))
_calibration_options(o::SimOptions) = SimOptions(;
    merge(NamedTuple{fieldnames(SimOptions)}(getfield.(Ref(o), fieldnames(SimOptions))), (use_noise = false, use_scatter = false))...,
)

"""
    FlatFieldCalibration

A detector's flat field, from simulated air scans: `gain` (`[cols, rows]` for an energy
integrator, `[cols, rows, bins]` for a photon counter) and `offset` (an energy integrator's,
relative to its air signal at the calibration's own protocol; `nothing` for a photon counter).
"""
struct FlatFieldCalibration{G <: AbstractArray, O}
    gain::G
    offset::O
end

"""
    calibrate_flat_field(scanner, protocol, recon_opts; sim_opts = SimOptions()) -> FlatFieldCalibration

Calibrate the flat field of `scanner` the way a scanner does it, from simulated air scans
through the protocol's beam (noise-free: a calibration averages far more views than a scan).
An energy integrator is scanned at the protocol's current and at a quarter of it, which separates
each dexel's gain from its offset; a photon counter is scanned once, without pile-up (a
low-flux calibration), and each bin's air counts give its gain. Use it with
[`apply_flat_field!`](@ref) on scans of the same scanner and protocol.
"""
function calibrate_flat_field(scanner::EICTScanner, protocol::CTProtocol, recon_opts; sim_opts::SimOptions = SimOptions())
    opts, air = _calibration_options(sim_opts), _air_phantom()
    y = map((protocol, CTProtocol(protocol; mA = protocol.mA / 4))) do p
        ws = create_eict_workspace(scanner, p, opts, recon_opts, air)
        try
            simulate!(ws, air, p, opts; report_dose = false)
            dropdims(sum(exp.(-Float64.(Array(ws.sinogram))); dims = 3); dims = 3) ./ size(ws.sinogram, 3)
        finally
            release_backend!(ws)
        end
    end
    # y₁ = g + o, y₂ = g + 4o (the offset relative to the full current's air signal)
    return FlatFieldCalibration((4 .* y[1] .- y[2]) ./ 3, (y[2] .- y[1]) ./ 3)
end

function calibrate_flat_field(scanner::PCCTScanner, protocol::CTProtocol, recon_opts; sim_opts::SimOptions = SimOptions())
    opts, air = _calibration_options(sim_opts), _air_phantom()
    s = PCCTScanner(scanner; pileup = false)
    ws = create_workspace(s, protocol, opts, recon_opts, air)
    try
        bins = simulate!(ws, air, protocol, opts; report_dose = false, capture_raw_counts = false).pcct_sino.bins
        nv = size(bins[1], 3)
        gain = cat((dropdims(sum(exp.(-Float64.(Array(b))); dims = 3); dims = 3) ./ nv for b in bins)...; dims = 3)
        return FlatFieldCalibration(gain, nothing)
    finally
        release_backend!(ws)
    end
end

"""
    apply_flat_field!(sinogram, cal::FlatFieldCalibration) -> sinogram
    apply_flat_field!(bins::Vector, cal::FlatFieldCalibration) -> bins

Correct a scan with the flat field [`calibrate_flat_field`](@ref) measured: an energy
integrator's air-normalised signal `y = e^{−p}` becomes `(y − offset) / gain`; a photon counter's
counts in each bin are divided by the bin's gain.
"""
function apply_flat_field!(sinogram::AbstractArray{T, 3}, cal::FlatFieldCalibration{<:AbstractMatrix}) where {T}
    g = copyto!(similar(sinogram, T, size(cal.gain)), T.(cal.gain))
    o = copyto!(similar(sinogram, T, size(cal.offset)), T.(cal.offset))
    sinogram .= .-log.(max.((exp.(.-sinogram) .- o) ./ g, T(1.0e-10)))
    return sinogram
end

function apply_flat_field!(bins::AbstractVector, cal::FlatFieldCalibration{<:AbstractArray{<:Real, 3}, Nothing})
    length(bins) == size(cal.gain, 3) || throw(DimensionMismatch("one gain plane per bin"))
    for (b, bin) in enumerate(bins)
        T = eltype(bin)
        bin .+= copyto!(similar(bin, T, size(cal.gain)[1:2]), T.(log.(cal.gain[:, :, b])))
    end
    return bins
end

"""
    PileupCalibration

A photon counter's pile-up, from simulated air scans at rising tube currents: for each bin, the
pile-up-free counts per recorded count (`ratio[b, k]`) at each recorded count of a pixel
(`recorded[k]`, all bins summed, per view).
"""
struct PileupCalibration
    recorded::Vector{Float64}
    ratio::Matrix{Float64}
end

"""
    calibrate_pileup(scanner::PCCTScanner, protocol, recon_opts; sim_opts = SimOptions(),
                     currents = [1/16, 1/8, 1/4, 1/2, 1, 2]) -> PileupCalibration

Calibrate pile-up the way a scanner can, from simulated air scans at the protocol's tube current
times each of `currents`: the pile-up-free counts of every pixel are the lowest current's,
scaled linearly with the current, and each bin's ratio of those to the recorded counts is
tabulated against the pixel's recorded counts (every pixel and current, binned in log counts).
Use it with [`apply_pileup!`](@ref).
"""
function calibrate_pileup(
        scanner::PCCTScanner, protocol::CTProtocol, recon_opts;
        sim_opts::SimOptions = SimOptions(), currents = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2],
    )
    opts, air = _calibration_options(sim_opts), _air_phantom()
    levels = sort(Float64.(collect(currents)))
    counts = map(levels) do x
        p = CTProtocol(protocol; mA = protocol.mA * x)
        ws = create_workspace(PCCTScanner(scanner; pileup = true), p, opts, recon_opts, air)
        try
            r = simulate!(ws, air, p, opts; report_dose = false, capture_raw_counts = true)
            [dropdims(sum(Float64.(Array(c)); dims = 3); dims = 3) ./ size(c, 3) for c in r.raw_counts]
        finally
            release_backend!(ws)
        end
    end
    nb = length(counts[1])
    xs, ys = Float64[], Vector{Float64}[]
    for (k, x) in pairs(levels)
        for i in eachindex(counts[k][1])
            rec = [counts[k][b][i] for b in 1:nb]
            free = [counts[1][b][i] * x / levels[1] for b in 1:nb]
            sum(rec) > 0 || continue
            push!(xs, sum(rec))
            push!(ys, [rec[b] > 0 ? free[b] / rec[b] : 1.0 for b in 1:nb])
        end
    end
    edges = exp.(range(log(minimum(xs)), log(maximum(xs)) * (1 + 1.0e-9); length = 41))
    recorded, ratio = Float64[], Vector{Float64}[]
    for k in 1:40
        sel = findall(x -> edges[k] <= x < edges[k + 1], xs)
        isempty(sel) && continue
        push!(recorded, sum(xs[sel]) / length(sel))
        push!(ratio, sum(ys[sel]) ./ length(sel))
    end
    return PileupCalibration(recorded, reduce(hcat, ratio))
end

"""
    apply_pileup!(bins, I0, cal::PileupCalibration) -> bins

Undo pile-up with the curves [`calibrate_pileup`](@ref) measured: each ray's recorded counts
`I0[col, row, b]·e^{−bin_b}`, all bins summed, look up each bin's ratio of pile-up-free to
recorded counts (linear between the calibration's points, held at its ends), which multiplies
that bin's counts.
"""
function apply_pileup!(bins::AbstractVector, I0::AbstractArray{<:Real, 3}, cal::PileupCalibration)
    nb = length(bins)
    size(cal.ratio, 1) == nb == size(I0, 3) || throw(DimensionMismatch("one calibration curve and I0 plane per bin"))
    host = [Float64.(Array(b)) for b in bins]
    I0h = Float64.(Array(I0))
    x, κ = cal.recorded, cal.ratio
    for v in axes(host[1], 3), r in axes(host[1], 2), c in axes(host[1], 1)
        n = sum(I0h[c, r, b] * exp(-host[b][c, r, v]) for b in 1:nb)
        if n <= x[1]
            i, f = 1, 0.0
        elseif n >= x[end]
            i, f = length(x), 0.0
        else
            i = searchsortedlast(x, n)
            f = (n - x[i]) / (x[i + 1] - x[i])
        end
        j = min(i + 1, length(x))
        for b in 1:nb
            host[b][c, r, v] -= log((1 - f) * κ[b, i] + f * κ[b, j])
        end
    end
    for (b, bin) in enumerate(bins)
        copyto!(bin, eltype(bin).(host[b]))
    end
    return bins
end

export FlatFieldCalibration, calibrate_flat_field, apply_flat_field!
export PileupCalibration, calibrate_pileup, apply_pileup!
