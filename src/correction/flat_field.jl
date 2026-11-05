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

# A calibration scan of air or of a centred cylinder does not change as the gantry turns, so the
# calibrations scan a few views, each read at one angle (`view_samples = 1`), noise-free and without
# scatter. Each keeps the protocol's time per view, so its counts per view, its offsets relative to
# them and its count rates are the scan's. Two views (the fewest a geometry has) match the
# protocol's full views (1200 × 5 sub-views, measured on notebook 04's NAEOTOM Alpha) to 1.9e-5 in
# the flat field; the pile-up correction's water cylinders are voxelised, and two angles change the
# scan it corrects by about 2e-5. Four and eight views do no better.
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

A photon counter's pile-up correction, from simulated calibration scans ([`calibrate_pileup`](@ref)):
the matrices `C[:, :, k, m]` that take a pixel's recorded counts in every bin to its pile-up-free
counts, `n = C · r`, at recorded count rates `rates[k]` (counts per second per native dexel, all
bins) and recorded mean bins `hardness[m]`; a ray takes the matrix bilinear in its own two
(held at the table's ends). `binning` is the scanner's, which turns a pixel's rate into a dexel's.
"""
struct PileupCalibration
    rates::Vector{Float64}
    hardness::Vector{Float64}
    C::Array{Float64, 4}
    binning::Int
end

# The water cylinders the pile-up is calibrated through, besides air (diameters in cm): their chords
# give every pixel rays from air down to 35 cm of water, so the calibration sees each rate with the
# spectra a patient gives it.
const _PILEUP_CYLINDERS_CM = (20.0, 35.0)

# A centred water cylinder of diameter `d_cm` (10 cm long) on the device of `phantom`.
function _water_cylinder(phantom::Phantom, d_cm::Real)
    n, voxel = 192, 0.25                                   # a 48 cm grid of 2.5 mm voxels
    mask = zeros(UInt8, n, n, 2)
    for j in 1:n, i in 1:n
        hypot((i - 0.5 - n / 2) * voxel, (j - 0.5 - n / 2) * voxel) <= d_cm / 2 && (mask[i, j, :] .= 0x01)
    end
    return Phantom(
        copyto!(similar(phantom.mask, UInt8, size(mask)), mask),
        Dict{Int, Any}(0 => XA.Materials.air, 1 => XA.Materials.water),
        (voxel, voxel, 5.0),
    )
end

"""
    calibrate_pileup(scanner::PCCTScanner, protocol, recon_opts, phantom; sim_opts = SimOptions(),
                     currents = [1/16, 1/8, 1/4, 1/2, 1/√2, 1, √2]) -> PileupCalibration

Calibrate pile-up the way a scanner can, from simulated scans with pile-up as the detector has it:
air and two water cylinders (20 and 35 cm), each at the protocol's tube current times each of
`currents`, two views at the protocol's time per view (each at one angle).

- **The pile-up-free counts** of every pixel and bin are its counts per unit current extrapolated to
  zero current, quadratic through the three lowest currents (at 1/16 of a clinical current the
  NAEOTOM Alpha still loses 16 % of its lowest bin in air, so the lowest scan is not pile-up free).
- **The correction** takes a pixel's recorded counts in every bin to its pile-up-free counts with a
  matrix, `n = C · r`, that depends on the pixel's recorded count rate and the mean bin of its
  recorded counts: pile-up migrates counts between bins as well as losing them, and how it does
  depends on the spectrum as well as the rate (a ray hardened by the patient piles up differently
  from an air ray at the same rate). `C` is fitted to every pixel, current and scan at once
  (least squares in relative counts, smooth in rate and hardness), on a 33 × 6 grid.

Pixels at a cylinder's edge, whose dexels see different rates, are left out. The scans run on the
device of `phantom`, the scan's own, as for [`calibrate_flat_field`](@ref); the scans are of air and
water, never the phantom. Use it with [`apply_pileup!`](@ref).
"""
function calibrate_pileup(
        scanner::PCCTScanner,
        protocol::CTProtocol,
        recon_opts,
        phantom::Phantom;
        sim_opts::SimOptions = SimOptions(),
        currents = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1 / sqrt(2), 1, sqrt(2)],
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

# The pile-up correction from scans of exactly `protocol` and `opts`.
function _pileup(
        scanner::PCCTScanner,
        protocol::CTProtocol,
        recon_opts,
        opts::SimOptions,
        phantom::Phantom,
        currents;
        n_rates::Integer = 33,
        n_hardness::Integer = 6,
    )
    levels = sort(Float64.(collect(currents)))
    length(levels) >= 3 || throw(ArgumentError("calibrate_pileup needs at least three currents"))
    sc = PCCTScanner(scanner; pileup = true)
    bf = sc.binning_factor
    tpv = protocol.rotation_time / protocol.views
    objects = vcat([_air_phantom(phantom)], [_water_cylinder(phantom, d) for d in _PILEUP_CYLINDERS_CM])
    # the recorded counts of every scan, [object][level][bin] → [col, row]
    recorded = map(objects) do obj
        map(levels) do x
            p = CTProtocol(protocol; mA = protocol.mA * x)
            ws = create_workspace(sc, p, opts, recon_opts, obj)
            try
                r = simulate!(ws, obj, p, opts; report_dose = false, capture_raw_counts = true)
                [_view_mean(c) for c in r.raw_counts]
            finally
                release_backend!(ws)
            end
        end
    end
    nb = length(recorded[1][1])
    # the pile-up-free counts per unit current: r(x) = recorded / x, quadratic to x = 0
    x1, x2, x3 = levels[1], levels[2], levels[3]
    L = (x2 * x3 / ((x1 - x2) * (x1 - x3)), x1 * x3 / ((x2 - x1) * (x2 - x3)), x1 * x2 / ((x3 - x1) * (x3 - x2)))
    Rs, Ns = Vector{Float64}[], Vector{Float64}[]        # each sample's recorded and free counts, per bin
    for (o, rec) in pairs(recorded)
        free = [L[1] .* rec[1][b] ./ x1 .+ L[2] .* rec[2][b] ./ x2 .+ L[3] .* rec[3][b] ./ x3 for b in 1:nb]
        total = sum(free)
        nc, nr = size(total)
        for r in 1:nr, c in 1:nc
            all(b -> free[b][c, r] > 0, 1:nb) || continue
            if o > 1 && 1 < c < nc      # a cylinder's edge: the pixel's dexels see different rates
                abs(total[c + 1, r] - total[c - 1, r]) <= 0.05 * total[c, r] || continue
            end
            for (k, x) in pairs(levels)
                rv = [rec[k][b][c, r] for b in 1:nb]
                all(>(0), rv) || continue
                push!(Rs, rv)
                push!(Ns, [free[b][c, r] * x for b in 1:nb])
            end
        end
    end
    isempty(Rs) && throw(ArgumentError("calibrate_pileup: the scans recorded no counts"))
    R, N = reduce(hcat, Rs), reduce(hcat, Ns)              # [bin, sample]
    ρs = vec(sum(R; dims = 1)) ./ (bf * bf * tpv)          # recorded counts per second per dexel
    qs = vec(sum((1:nb) .* R; dims = 1)) ./ vec(sum(R; dims = 1))
    rates = collect(range(0.0, maximum(ρs) * (1 + 1.0e-9); length = n_rates))
    hardness = nb > 1 && maximum(qs) > minimum(qs) ?
        collect(range(minimum(qs), maximum(qs); length = n_hardness)) : [1.0, 2.0]
    C = _fit_inverse_migration(R, N, ρs, qs, rates, hardness)
    return PileupCalibration(rates, hardness, C, bf)
end

# The matrices `C[:, :, k, m]` (bilinear in rate and hardness) with `N ≈ C(ρ, q) · R`, sample by sample:
# least squares in relative counts, `C = I` at zero rate, smooth (second differences) in both.
function _fit_inverse_migration(R::Matrix{Float64}, N::Matrix{Float64}, ρs, qs, rates, hardness)
    nb, ns = size(R)
    K, M = length(rates), length(hardness)
    P = nb * K * M
    idx(j, k, m) = j + nb * ((k - 1) + K * (m - 1))
    corners = map(1:ns) do t
        k, f = _node(rates, ρs[t])
        m, g = _node(hardness, qs[t])
        ((k, m, (1 - f) * (1 - g)), (k + 1, m, f * (1 - g)), (k, m + 1, (1 - f) * g), (k + 1, m + 1, f * g))
    end
    C = zeros(nb, nb, K, M)
    Threads.@threads for i in 1:nb
        A = zeros(P, P)
        y = zeros(P)
        cols = zeros(Int, 4nb)
        vals = zeros(4nb)
        for t in 1:ns
            s = 1 / N[i, t]                                  # relative counts
            q = 0
            for (kk, mm, w) in corners[t], j in 1:nb
                q += 1
                cols[q] = idx(j, kk, mm)
                vals[q] = w * R[j, t] * s
            end
            for a in 1:q
                ca, va = cols[a], vals[a]
                y[ca] += va
                for b in 1:q
                    A[ca, cols[b]] += va * vals[b]
                end
            end
        end
        λ0, λ = 1.0e6, 0.1 * ns / (K * M)
        for m in 1:M, j in 1:nb
            a = idx(j, 1, m)
            A[a, a] += λ0
            y[a] += λ0 * (i == j)
        end
        for j in 1:nb, m in 1:M, k in 2:(K - 1)
            _penalise!(A, (idx(j, k - 1, m), idx(j, k, m), idx(j, k + 1, m)), λ)
        end
        for j in 1:nb, k in 1:K, m in 2:(M - 1)
            _penalise!(A, (idx(j, k, m - 1), idx(j, k, m), idx(j, k, m + 1)), λ)
        end
        θ = Symmetric(A) \ y
        for j in 1:nb, k in 1:K, m in 1:M
            C[i, j, k, m] = θ[idx(j, k, m)]
        end
    end
    return C
end

# The node below `v` and the fraction of the way to the next (held at the ends).
function _node(nodes, v)
    n = length(nodes)
    v <= nodes[1] && return 1, 0.0
    v >= nodes[n] && return n - 1, 1.0
    k = clamp(searchsortedlast(nodes, v), 1, n - 1)
    return k, (v - nodes[k]) / (nodes[k + 1] - nodes[k])
end

# λ · (a − 2b + c)² added to the normal equations
function _penalise!(A, (a, b, c), λ)
    d = ((a, 1.0), (b, -2.0), (c, 1.0))
    for (p, u) in d, (q, v) in d
        A[p, q] += λ * u * v
    end
    return A
end

"""
    apply_pileup!(bins, I0, cal::PileupCalibration, protocol) -> bins

Undo pile-up with the correction [`calibrate_pileup`](@ref) measured: each ray's recorded counts in
every bin (`I0[col, row, b]·e^{−bin_b}`), its recorded count rate per dexel over the scan's time per
view and the mean bin of its counts pick the matrix that takes them to pile-up-free counts, which
replace them: `bin_b = −log(n_b / I0[col, row, b])`. On the device of `bins`.
"""
function apply_pileup!(bins::AbstractVector, I0::AbstractArray{<:Real, 3}, cal::PileupCalibration, protocol::CTProtocol)
    nb = length(bins)
    size(cal.C, 1) == size(cal.C, 2) == nb == size(I0, 3) || throw(DimensionMismatch("one correction row and I0 plane per bin"))
    T = eltype(bins[1])
    device(x) = copyto!(similar(bins[1], T, size(x)), T.(Array(x)))
    time_per_view = protocol.rotation_time / protocol.views
    _apply_pileup!(
        Tuple(bins), device(I0), device(cal.C), device(cal.hardness), Int32(length(cal.rates)),
        T(cal.rates[2] - cal.rates[1]), T(1 / (cal.binning^2 * time_per_view)),
    )
    return bins
end

function _apply_pileup!(h::NTuple{NB}, I0, C, hardness, K, dr, scale) where {NB}
    rays = CartesianIndices(size(h[1]))
    M = Int32(length(hardness))
    AK.foreachindex(h[1]) do idx
        c, r, v = Tuple(rays[idx])
        y = ntuple(b -> @inbounds(I0[c, r, b] * exp(-h[b][c, r, v])), Val(NB))
        x = clamp(scale * sum(y) / dr, zero(dr), oftype(dr, K - 1))
        k = min(unsafe_trunc(Int32, x), K - Int32(2))
        f = x - k
        q = _mean_bin(y)
        m = Int32(1)
        while m < M - Int32(1) && q >= @inbounds(hardness[m + 1])
            m += Int32(1)
        end
        g = @inbounds clamp((q - hardness[m]) / (hardness[m + 1] - hardness[m]), zero(dr), one(dr))
        n = _migrate(C, y, k, m, f, g)
        for b in 1:NB
            @inbounds h[b][c, r, v] = -log(max(n[b], oftype(dr, 1.0e-30)) / I0[c, r, b])
        end
    end
    return h
end

export calibrate_flat_field, apply_flat_field!, calibrate_pileup, apply_pileup!
