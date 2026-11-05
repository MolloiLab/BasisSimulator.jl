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

# Apply them to expected counts `[b][c, r, v]` against the air counts `I0[c, r, b]`, in one kernel
# on the arrays' backend (the factors `d.air`, `d.hard` live there too).
function _apply_pcct_defects!(expected::AbstractVector, I0, d)
    _apply_pcct_defects!(Tuple(expected), I0, d.air, d.hard, eltype(expected[1])(d.p_hard))
    return expected
end

function _apply_pcct_defects!(λ::NTuple{NB}, I0, air, hard, p_hard) where {NB}
    dexels = CartesianIndices(size(λ[1]))
    AK.foreachindex(λ[1]) do idx
        c, r, v = Tuple(dexels[idx])
        e = ntuple(b -> @inbounds(λ[b][c, r, v]), Val(NB))
        a = sum(ntuple(b -> @inbounds(I0[c, r, b]), Val(NB)))
        tiny = oftype(p_hard, 1.0e-30)
        x = clamp(-log(max(sum(e), tiny) / max(a, tiny)) / p_hard, zero(p_hard), oftype(p_hard, 2))
        for b in 1:NB
            @inbounds λ[b][c, r, v] = e[b] * (air[c, r, b] + x * (hard[c, r, b] - air[c, r, b]))
        end
    end
    return λ
end

# The calibrations' air scans run on the scan's phantom's device (`simulate!` runs where the
# phantom's mask lives); they scan air, never the patient.
_air_phantom(phantom::Phantom) = Phantom(
    fill!(similar(phantom.mask, UInt8, 4, 4, 2), 0x00),
    Dict{Int, Any}(0 => XA.Materials.air),
    (0.05, 0.05, 0.05),
)
# The mean over the views of a sinogram, `[cols, rows]`, on the host in Float64.
_view_mean(x::AbstractArray{<:Real, 3}) = Float64.(Array(dropdims(sum(x; dims = 3); dims = 3))) ./ size(x, 3)

# An air scan does not change as the gantry turns, so the calibrations scan a few views, each read
# at one angle (`view_samples = 1`), noise-free and without scatter. Each keeps the protocol's time
# per view, so its counts per view, its offsets relative to them and its count rates are the
# scan's. Two views (the fewest a geometry has) match the protocol's full views (1200 × 5 sub-views,
# measured on notebook 04's NAEOTOM Alpha) to 1.9e-5 in the flat field and 1.5e-5 in a scan the
# pile-up curves correct; 4 and 8 views do no better.
const _CALIBRATION_VIEWS = 2
_calibration_protocol(p::CTProtocol) = CTProtocol(
    p;
    views = _CALIBRATION_VIEWS,
    rotation_time = p.rotation_time * _CALIBRATION_VIEWS / p.views,
)
_calibration_options(o::SimOptions) = SimOptions(
    o;
    use_noise = false,
    use_scatter = false,
    view_samples = 1,
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
    calibrate_flat_field(scanner, protocol, recon_opts, phantom; sim_opts = SimOptions())
        -> FlatFieldCalibration

Calibrate the flat field of `scanner` the way a scanner does it, from simulated air scans
through the protocol's beam (noise-free: a calibration averages far more views than a scan).
An energy integrator is scanned at the protocol's current and at a quarter of it, which separates
each dexel's gain from its offset; a photon counter is scanned once, without pile-up (a
low-flux calibration), and each bin's air counts give its gain. Use it with
[`apply_flat_field!`](@ref) on scans of the same scanner and protocol.

An air scan does not change as the gantry turns, so each scan is two views at the protocol's time
per view, each read at one angle (`view_samples = 1`, whatever `sim_opts` says). The gains and
offsets differ from those of the protocol's full views by at most 2e-5 (measured against
1200 views × 5 sub-views: 1.9e-5 in the NAEOTOM Alpha's gains, 6e-6 and 1.5e-6 in the gains and
offsets of a SOMATOM Force with gain and offset spreads), and the calibration takes 1.0 s on an
NVIDIA L40S for notebook 04's Alpha protocol (2.6 s with its 1200 × 5 views).

The scans run on the device of `phantom`, the scan's own (as for [`create_workspace`](@ref)): a
phantom on the GPU calibrates on the GPU, one on the CPU on the CPU. They scan air, never the
phantom itself; the calibration's arrays are on the host, in Float64.
"""
function calibrate_flat_field(
        scanner::Scanner,
        protocol::CTProtocol,
        recon_opts,
        phantom::Phantom;
        sim_opts::SimOptions = SimOptions(),
    )
    return _flat_field(
        scanner,
        _calibration_protocol(protocol),
        recon_opts,
        _calibration_options(sim_opts),
        phantom,
    )
end

# The flat field from air scans of exactly `protocol` and `opts`.
function _flat_field(
        scanner::EICTScanner,
        protocol::CTProtocol,
        recon_opts,
        opts::SimOptions,
        phantom::Phantom,
    )
    air = _air_phantom(phantom)
    y = map((protocol, CTProtocol(protocol; mA = protocol.mA / 4))) do p
        ws = create_eict_workspace(scanner, p, opts, recon_opts, air)
        try
            simulate!(ws, air, p, opts; report_dose = false)
            _view_mean(exp.(.-ws.sinogram))
        finally
            release_backend!(ws)
        end
    end
    # y₁ = g + o, y₂ = g + 4o (the offset relative to the full current's air signal)
    return FlatFieldCalibration((4 .* y[1] .- y[2]) ./ 3, (y[2] .- y[1]) ./ 3)
end

function _flat_field(
        scanner::PCCTScanner,
        protocol::CTProtocol,
        recon_opts,
        opts::SimOptions,
        phantom::Phantom,
    )
    air = _air_phantom(phantom)
    s = PCCTScanner(scanner; pileup = false)
    ws = create_workspace(s, protocol, opts, recon_opts, air)
    try
        bins = simulate!(ws, air, protocol, opts; report_dose = false, capture_raw_counts = false).pcct_sino.bins
        gain = cat((_view_mean(exp.(.-b)) for b in bins)...; dims = 3)
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
pile-up-free counts per recorded count (`ratio[b, k]`) at each recorded count rate of a pixel
(`recorded[k]`, all bins summed, counts per second).
"""
struct PileupCalibration
    recorded::Vector{Float64}       # counts per second
    ratio::Matrix{Float64}
end

"""
    calibrate_pileup(scanner::PCCTScanner, protocol, recon_opts, phantom; sim_opts = SimOptions(),
                     currents = [1/16, 1/8, 1/4, 1/2, 1, 2]) -> PileupCalibration

Calibrate pile-up the way a scanner can, from simulated air scans at the protocol's tube current
times each of `currents`: the pile-up-free counts of every pixel are the lowest current's,
scaled linearly with the current, and each bin's ratio of those to the recorded counts is
tabulated against the pixel's recorded counts (every pixel and current, binned in log counts).
Tabulated against the recorded count rate, it serves scans of any view count. Pass the scan's own
protocol: each air scan is two views at its time per view, each read at one angle, as for
[`calibrate_flat_field`](@ref). A scan it corrects differs from one corrected with the curves of
the protocol's full views (1200 × 5 sub-views) by at most 1.5e-5 in log counts (measured on
notebook 04's NAEOTOM Alpha, 6.2 s on an NVIDIA L40S). Use it with [`apply_pileup!`](@ref).
The scans run on the device of `phantom`, the scan's own, as for [`calibrate_flat_field`](@ref).
"""
function calibrate_pileup(
        scanner::PCCTScanner,
        protocol::CTProtocol,
        recon_opts,
        phantom::Phantom;
        sim_opts::SimOptions = SimOptions(),
        currents = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2],
    )
    return _pileup(
        scanner,
        _calibration_protocol(protocol),
        recon_opts,
        _calibration_options(sim_opts),
        phantom,
        currents,
    )
end

# The pile-up curves from air scans of exactly `protocol` and `opts`.
function _pileup(
        scanner::PCCTScanner,
        protocol::CTProtocol,
        recon_opts,
        opts::SimOptions,
        phantom::Phantom,
        currents,
    )
    air = _air_phantom(phantom)
    levels = sort(Float64.(collect(currents)))
    counts = map(levels) do x
        p = CTProtocol(protocol; mA = protocol.mA * x)
        ws = create_workspace(PCCTScanner(scanner; pileup = true), p, opts, recon_opts, air)
        try
            r = simulate!(ws, air, p, opts; report_dose = false, capture_raw_counts = true)
            [_view_mean(c) for c in r.raw_counts]
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
            push!(xs, sum(rec) / (protocol.rotation_time / protocol.views))   # a rate: views need not match the scan's
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
    apply_pileup!(bins, I0, cal::PileupCalibration, protocol) -> bins

Undo pile-up with the curves [`calibrate_pileup`](@ref) measured: each ray's recorded count rate
(`I0[col, row, b]·e^{−bin_b}`, all bins summed, over the scan's time per view), looks up each bin's ratio of pile-up-free to
recorded counts (linear between the calibration's points, held at its ends), which multiplies
that bin's counts.
"""
function apply_pileup!(bins::AbstractVector, I0::AbstractArray{<:Real, 3}, cal::PileupCalibration, protocol::CTProtocol)
    nb = length(bins)
    size(cal.ratio, 1) == nb == size(I0, 3) || throw(DimensionMismatch("one calibration curve and I0 plane per bin"))
    T = eltype(bins[1])
    device(x) = copyto!(similar(bins[1], T, size(x)), T.(Array(x)))
    _apply_pileup!(Tuple(bins), device(I0), device(cal.recorded), device(cal.ratio), T(protocol.rotation_time / protocol.views))
    return bins
end

function _apply_pileup!(h::NTuple{NB}, I0, x, κ, time_per_view) where {NB}
    rays = CartesianIndices(size(h[1]))
    K = length(x)
    AK.foreachindex(h[1]) do idx
        c, r, v = Tuple(rays[idx])
        n = sum(ntuple(b -> @inbounds(I0[c, r, b] * exp(-h[b][c, r, v])), Val(NB))) / time_per_view
        i, f = 1, zero(time_per_view)
        if n >= @inbounds x[K]
            i = K
        elseif n > @inbounds x[1]
            while @inbounds x[i + 1] <= n
                i += 1
            end
            f = @inbounds (n - x[i]) / (x[i + 1] - x[i])
        end
        j = min(i + 1, K)
        for b in 1:NB
            @inbounds h[b][c, r, v] -= log((1 - f) * κ[b, i] + f * κ[b, j])
        end
    end
    return h
end

export calibrate_flat_field, apply_flat_field!, calibrate_pileup, apply_pileup!
