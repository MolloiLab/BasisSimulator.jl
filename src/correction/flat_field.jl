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
# - `_calibrate_flat_field`: air scans (two currents for an energy
#   integrator, so its offset separates from its gain); for a photon counter also water slabs,
#   each pixel's scan of them against the nominal detector's, since its threshold offsets weigh a
#   hardened beam differently;
# - `_calibrate_pileup`: scans of air and water cylinders at rising tube
#   currents, the pile-up-free counts extrapolated to zero current.
#
# `calibrate` makes both (what a scanner needs of them), the workspace holds the `Calibration`, and
# `simulate!` applies it in a scanner's order: pile-up, flat field, then the scatter correction on the
# corrected signal (`_correct_pileup!`, `_correct_flat_field!`).
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
    Threads.@threads for r in 1:nr              # rows in parallel, each its own slice of `out`
        for c in 1:nc
            g = 1 + s.gain_spread * z[c, r, nb + 1]
            shifted = [cum((thr[k] + s.threshold_spread_keV * z[c, r, k]) / g) for k in 1:nb]
            for b in 1:nb
                n0 = bin(nominal, b)
                out[c, r, b] = n0 > 0 ? bin(shifted, b) / n0 : 1.0
            end
        end
    end
    return out
end

# The water thicknesses (cm) at which a photon-counting dexel's defects are evaluated: a dexel's
# threshold offsets weigh a hardened spectrum differently, so its factors change with the ray's
# hardening, smoothly up to the thickest (held beyond it).
const _PCCT_DEFECT_WATER_CM = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)

# The photon-counting defects at the projection's resolution: each dexel's factors in every bin for
# the beam through each of `_PCCT_DEFECT_WATER_CM` of water (`factors[col, row, bin, m]`), and each
# beam's line integral (`p[m]`); a ray takes the factors linear in its own line integral between them.
# Memoised: a detector's factors depend on its defects, its grid and the spectrum's shape (they are
# ratios of counts, so the tube current cancels), the same for every workspace of a scanner and beam
# — a calibration's dozens of scans, every scan of a cohort.
const _PCCT_DEFECTS_CACHE = Dict{UInt64, Any}()
const _PCCT_DEFECTS_LOCK = ReentrantLock()

function _pcct_dexel_defects(s::PCCTScanner, nc, nr, energies, weights)
    _has_defects(s) || return nothing
    w = Float64.(weights)
    key = hash((s.energy_thresholds, s.gain_spread, s.threshold_spread_keV, s.defect_seed, nc, nr, Float64.(energies), w ./ sum(w)))
    cached = lock(() -> get(_PCCT_DEFECTS_CACHE, key, nothing), _PCCT_DEFECTS_LOCK)
    cached === nothing || return cached
    μ = [compute_μ_at_energy(XA.Materials.water, Float64(E)) for E in energies]
    spectra = [w .* exp.(-t .* μ) for t in _PCCT_DEFECT_WATER_CM]
    d = (
        factors = cat((_pcct_defect_factors(s, nc, nr, energies, x) for x in spectra)...; dims = 4),
        p = [-log(sum(x) / sum(w)) for x in spectra],
    )
    lock(() -> (_PCCT_DEFECTS_CACHE[key] = d), _PCCT_DEFECTS_LOCK)
    return d
end

# Apply them to expected counts `[b][c, r, v]` against the air counts `I0[c, r, b]`, in one kernel
# on the arrays' backend (the factors live there too).
function _apply_pcct_defects!(expected::AbstractVector, I0, d)
    _apply_pcct_defects!(Tuple(expected), I0, d.factors, d.p, Int32(length(d.p)))
    return expected
end

function _apply_pcct_defects!(λ::NTuple{NB}, I0, F, pn, M) where {NB}
    dexels = CartesianIndices(size(λ[1]))
    AK.foreachindex(λ[1]) do idx
        c, r, v = Tuple(dexels[idx])
        e = ntuple(b -> @inbounds(λ[b][c, r, v]), Val(NB))
        a = sum(ntuple(b -> @inbounds(I0[c, r, b]), Val(NB)))
        tiny = oftype(@inbounds(pn[1]), 1.0e-30)
        p = -log(max(sum(e), tiny) / max(a, tiny))
        m = Int32(1)
        while m < M - Int32(1) && p >= @inbounds(pn[m + 1])
            m += Int32(1)
        end
        g = @inbounds clamp((p - pn[m]) / (pn[m + 1] - pn[m]), zero(p), one(p))
        for b in 1:NB
            @inbounds λ[b][c, r, v] = e[b] * ((1 - g) * F[c, r, b, m] + g * F[c, r, b, m + 1])
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

A detector's flat field, from simulated calibration scans: `gain` (`[cols, rows]` for an energy
integrator, `[cols, rows, bins]` for a photon counter: each dexel's gain in air) and `offset` (an
energy integrator's, relative to its air signal at the calibration's own protocol; `nothing` for a
photon counter). A photon counter's `hardening` is how each pixel's gain in every bin changes as
the patient hardens its beam (a threshold offset weighs a hardened spectrum differently): at line
integrals `p[col, row, k]` (the calibration's water slabs; `Inf` where a pixel has none), its gain relative to air
`r[col, row, bin, k]`, linear in the line integral between them (1 at zero, held beyond the last),
and `w[col, row, bin]`, each bin's share of the pixel's air counts, which gives a ray's line
integral from its bins. An energy integrator's gain does not depend on the spectrum (`nothing`).
"""
struct FlatFieldCalibration{G <: AbstractArray, O, H}
    gain::G
    offset::O
    hardening::H
end
FlatFieldCalibration(gain, offset) = FlatFieldCalibration(gain, offset, nothing)

"""
    _calibrate_flat_field(scanner, protocol, recon_opts, phantom; sim_opts = SimOptions())
        -> FlatFieldCalibration

Calibrate the flat field of `scanner` the way a scanner does it, from simulated air scans
through the protocol's beam (noise-free: a calibration averages far more views than a scan).
An energy integrator is scanned at the protocol's current and at a quarter of it, which separates
each dexel's gain from its offset (both energy-blind). A photon counter is scanned without pile-up
(a low-flux calibration): in air, where each bin's counts give its gain, and through water slabs
every 5 cm from 5 to 50 cm thick. Its dexels' threshold offsets weigh a hardened beam differently, so
a pixel's gain in a bin changes with the patient's hardening; each pixel's scan of a slab is divided
by the nominal detector's scan of it (the scanner's design response to its own calibration
phantom, as a scanner's per-pixel spectral calibration does), which gives its gain at that line
integral with no smoothing across pixels. [`calibrate`](@ref) makes it for a detector with defects. A detector without defects needs no slabs (`hardening` is `nothing`).

On notebook 04's NAEOTOM Alpha (Gammex 472, noise-free, pile-up corrected), the corrected images of
every bin are within 0.6 HU of the ideal detector's in their rings (ring RMS 0.09 HU); with the air
gains alone they ring by up to 170 HU.

An air scan does not change as the gantry turns, so each scan is two views at the protocol's time
per view, each read at one angle (`view_samples = 1`, whatever `sim_opts` says). The gains and
offsets differ from those of the protocol's full views by at most 2e-5 (measured against
1200 views × 5 sub-views: 1.9e-5 in the NAEOTOM Alpha's gains, 6e-6 and 1.5e-6 in the gains and
offsets of a SOMATOM Force with gain and offset spreads), and the calibration takes 1.0 s on an
NVIDIA L40S for notebook 04's Alpha protocol (2.6 s with its 1200 × 5 views).

The scans run on the device of `phantom`, the scan's own (as for [`create_workspace`](@ref)): a
phantom on the GPU calibrates on the GPU, one on the CPU on the CPU. They scan air and water,
never the phantom itself; the calibration's arrays are on the host, in Float64.
"""
function _calibrate_flat_field(
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
        ws = create_workspace(scanner, p, opts, recon_opts, air)
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

# The water slabs a photon counter's flat field is calibrated through, besides air (thicknesses in
# cm): every pixel sees its gain at ten hardenings, every 5 cm from 5 to 50 cm of water (a little
# more at the fan's edge, which crosses a slab obliquely). Each pixel's scan of a slab is divided by the nominal
# detector's scan of it, the scanner's design response to its own calibration phantom: what is left is
# that pixel's own gain at that hardening, with no smoothing across pixels and nothing to leak.
const _FLAT_FIELD_SLABS_CM = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)

# A water slab `t_cm` thick, 70 cm wide and 10 cm long, centred, its faces square to the calibration's
# two views (at 0 and 180 degrees, whose rays run along y), on the device of `phantom`: one voxel
# through its thickness, so every ray crosses the same flat faces.
function _water_slab(phantom::Phantom, t_cm::Real)
    mask = ones(UInt8, 14, 1, 2)
    return Phantom(
        copyto!(similar(phantom.mask, UInt8, size(mask)), mask),
        Dict{Int, Any}(0 => XA.Materials.air, 1 => XA.Materials.water),
        (5.0, Float64(t_cm), 5.0),
    )
end

function _flat_field(
        scanner::PCCTScanner,
        protocol::CTProtocol,
        recon_opts,
        opts::SimOptions,
        phantom::Phantom,
    )
    unit = PCCTScanner(scanner; pileup = false)
    nominal = PCCTScanner(unit; gain_spread = 0.0, threshold_spread_keV = 0.0)
    # each bin's counts over its air counts, view-averaged, and the air counts of every pixel
    function transmission(s, obj)
        ws = create_workspace(s, protocol, opts, recon_opts, obj)
        try
            bins = simulate!(ws, obj, protocol, opts; report_dose = false, capture_raw_counts = false).pcct_sino.bins
            return [_view_mean(exp.(.-b)) for b in bins], copy(ws.I0_cpu)
        finally
            release_backend!(ws)
        end
    end
    air, I0 = transmission(unit, _air_phantom(phantom))
    gain = cat(air...; dims = 3)
    nc, nr, nb = size(gain)
    w = I0 ./ max.(sum(I0; dims = 3), 1.0e-300)
    _has_defects(unit) || return FlatFieldCalibration(gain, nothing, nothing)
    K = length(_FLAT_FIELD_SLABS_CM)
    P = fill(Inf, nc, nr, K)
    R = ones(nc, nr, nb, K)
    lit = dropdims(sum(I0; dims = 3); dims = 3) .> 0           # the pixels the beam reaches
    for (k, d) in pairs(_FLAT_FIELD_SLABS_CM)
        slab = _water_slab(phantom, d)
        t, _ = transmission(unit, slab)
        t0, _ = transmission(nominal, slab)
        p = -log.(max.(sum(w[:, :, b] .* t0[b] for b in 1:nb), 1.0e-300))  # the slab's line integral
        for r in 1:nr, c in 1:nc
            lit[c, r] || continue
            P[c, r, k] = p[c, r]
            for b in 1:nb
                R[c, r, b, k] = t0[b][c, r] > 0 ? t[b][c, r] / gain[c, r, b] / t0[b][c, r] : 1.0
            end
        end
    end
    return FlatFieldCalibration(gain, nothing, (p = P, r = R, w = w))
end

# A scan corrected with the flat field `_calibrate_flat_field` measured (`simulate!`'s step): an energy
# integrator's air-normalised signal `y = e^{−p}` becomes `(y − offset) / gain`; a photon counter's
# counts in each bin are divided by the bin's gain in air, then by its gain at the ray's own line
# integral (from its bins), linear between the calibration's water slabs. On the device of the bins.
function _correct_flat_field!(sinogram::AbstractArray{T, 3}, cal::FlatFieldCalibration{<:AbstractMatrix}) where {T}
    g = copyto!(similar(sinogram, T, size(cal.gain)), T.(cal.gain))
    o = copyto!(similar(sinogram, T, size(cal.offset)), T.(cal.offset))
    sinogram .= .-log.(max.((exp.(.-sinogram) .- o) ./ g, T(1.0e-10)))
    return sinogram
end

function _correct_flat_field!(bins::AbstractVector, cal::FlatFieldCalibration{<:AbstractArray{<:Real, 3}, Nothing})
    length(bins) == size(cal.gain, 3) || throw(DimensionMismatch("one gain plane per bin"))
    for (b, bin) in enumerate(bins)
        T = eltype(bin)
        bin .+= copyto!(similar(bin, T, size(cal.gain)[1:2]), T.(log.(cal.gain[:, :, b])))
    end
    cal.hardening === nothing && return bins
    T = eltype(bins[1])
    device(x) = copyto!(similar(bins[1], T, size(x)), T.(x))
    h = cal.hardening
    _apply_hardening!(Tuple(bins), device(h.p), device(h.r), device(h.w), Int32(size(h.p, 3)))
    return bins
end

# Each ray's gain in every bin at its own line integral (from its air-flat-fielded bins), linear
# between its pixel's calibrated hardenings: the bins are divided by it.
function _apply_hardening!(h::NTuple{NB}, P, R, w, K) where {NB}
    rays = CartesianIndices(size(h[1]))
    AK.foreachindex(h[1]) do idx
        c, r, v = Tuple(rays[idx])
        y = ntuple(b -> @inbounds(w[c, r, b] * exp(-h[b][c, r, v])), Val(NB))
        tiny = oftype(@inbounds(P[c, r, 1]), 1.0e-30)
        p = -log(max(sum(y), tiny))
        for b in 1:NB
            g = _hardening_gain(P, R, c, r, b, K, p)
            @inbounds h[b][c, r, v] += log(max(g, tiny))
        end
    end
    return h
end

# A pixel's gain in bin `b` at line integral `p`: 1 at zero, linear between its hardenings, held
# beyond the last it has.
@inline function _hardening_gain(P, R, c, r, b, K, p)
    p0, r0 = zero(p), one(p)
    for k in Int32(1):K
        pk = @inbounds P[c, r, k]
        isfinite(pk) || return r0
        rk = @inbounds R[c, r, b, k]
        p <= pk && return r0 + (rk - r0) * (p - p0) / (pk - p0)
        p0, r0 = pk, rk
    end
    return r0
end

"""
    PileupCalibration

A photon counter's pile-up correction, from simulated calibration scans ([`calibrate`](@ref)):
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
    _calibrate_pileup(scanner::PCCTScanner, protocol, recon_opts, phantom; sim_opts = SimOptions(),
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
device of `phantom`, the scan's own, as for [`calibrate`](@ref); the scans are of air and
water, never the phantom. [`calibrate`](@ref) makes it for a photon counter that piles up.
"""
function _calibrate_pileup(
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
        n_rates::Integer = 65,
        n_hardness::Integer = 12,
    )
    levels = sort(Float64.(collect(currents)))
    length(levels) >= 3 || throw(ArgumentError("_calibrate_pileup needs at least three currents"))
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
    isempty(Rs) && throw(ArgumentError("_calibrate_pileup: the scans recorded no counts"))
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

# Pile-up undone with the correction `_calibrate_pileup` measured (`simulate!`'s first step): each ray's
# recorded counts in every bin (`I0[col, row, b]·e^{−bin_b}`), its recorded count rate per dexel over
# the scan's time per view and the mean bin of its counts pick the matrix that takes them to
# pile-up-free counts, which replace them: `bin_b = −log(n_b / I0[col, row, b])`. On the device of `bins`.
function _correct_pileup!(bins::AbstractVector, I0::AbstractArray{<:Real, 3}, cal::PileupCalibration, protocol::CTProtocol)
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

# =============================================================================
# A scanner's calibration: everything its corrections need, from one call
# =============================================================================

"""
    Calibration

A scanner's calibration for one protocol, from simulated calibration scans ([`calibrate`](@ref)):
`pileup`, the pile-up correction (a `PileupCalibration`, or `nothing` for a detector that does not
pile up), and `flat_field`, the dexels' gains (a `FlatFieldCalibration`, or `nothing` for a
detector without defects); `scanner` and `protocol` are the ones it was made for. Pass it to
[`create_workspace`](@ref) (`calibration`): [`simulate!`](@ref) then corrects in a scanner's order,
pile-up, flat field, then the scatter.
"""
struct Calibration{S <: Scanner, P, F}
    scanner::S
    protocol::CTProtocol
    pileup::P
    flat_field::F
end

"""
    calibrate(scanner, protocol, recon_opts, phantom; sim_opts = SimOptions()) -> Calibration

Calibrate `scanner` for `protocol` the way a scanner is calibrated, from simulated scans of air and
water (never of the phantom, never from the model's truth), on the device of `phantom`: whatever this
scanner's corrections need, `nothing` where it needs none.

- **Flat field**, for a detector whose dexels have defects (`gain_spread`, `offset_spread_keV`,
  `threshold_spread_keV`). An energy integrator is scanned in air at the protocol's current and at a
  quarter of it, which separates each dexel's gain from its offset. A photon counter is scanned
  without pile-up in air and through water slabs every 5 cm from 5 to 50 cm, each pixel's scan of a
  slab divided by the nominal detector's: its gain in every bin as the patient hardens the beam (its
  threshold offsets weigh a hardened beam differently).
- **Pile-up**, for a photon counter that piles up (`pileup` and a dead time): scans of air and of 20
  and 35 cm water cylinders at 1/16 … √2 of the protocol's current; the pile-up-free counts,
  extrapolated to zero current, fit the inverse migration as a function of a ray's recorded rate
  and mean bin.

Each calibration scan is two views at the protocol's time per view, noise-free and without scatter:
an air or a centred scan does not change as the gantry turns (within 2e-5 of the protocol's full
views). On notebook 04's NAEOTOM Alpha (Gammex 472, noise-free) the corrected images are within
0.6 HU of the ideal detector's in their rings; with the air gains alone they ring by up to 170 HU.
The calibration depends on the scanner and the beam (kVp, filtration, collimation), not on the
phantom or the tube current's exact value: calibrate once and reuse it across phantoms and scans.
"""
function calibrate(scanner::PCCTScanner, protocol::CTProtocol, recon_opts, phantom::Phantom; sim_opts::SimOptions = SimOptions())
    pileup = _piles_up(scanner) ? _calibrate_pileup(scanner, protocol, recon_opts, phantom; sim_opts) : nothing
    flat_field = _has_defects(scanner) ? _calibrate_flat_field(scanner, protocol, recon_opts, phantom; sim_opts) : nothing
    return Calibration(scanner, protocol, pileup, flat_field)
end
function calibrate(scanner::EICTScanner, protocol::CTProtocol, recon_opts, phantom::Phantom; sim_opts::SimOptions = SimOptions())
    flat_field = _has_defects(scanner) ? _calibrate_flat_field(scanner, protocol, recon_opts, phantom; sim_opts) : nothing
    return Calibration(scanner, protocol, nothing, flat_field)
end

_piles_up(s::PCCTScanner) = s.pileup && s.dead_time_ns > 0
_piles_up(::EICTScanner) = false

# A scanner's detector, for matching a calibration: every field but the scatter's, which the
# calibration (scanned without scatter) does not depend on.
_detector_fields(s::Scanner) = [getfield(s, f) for f in fieldnames(typeof(s)) if f ∉ (:scatter_kernels, :scatter_correction)]

# The calibration a workspace for `scanner`, `protocol` and `sim_opts` takes: checked against the
# scanner and the beam it was made for, and required whenever the scatter correction is on and the
# detector piles up or has defects (the scatter estimate is a nominal dexel's: subtracted from
# uncorrected counts it leaves each dexel S·(1 − 1/g) of its scatter, as rings).
function _check_calibration(scanner::Scanner, protocol::CTProtocol, sim_opts::SimOptions, cal)
    needs = (_piles_up(scanner) || _has_defects(scanner)) && scanner.scatter_correction && sim_opts.use_scatter
    if cal === nothing
        needs && throw(
            ArgumentError(
                "this detector $(_piles_up(scanner) ? "piles up" : "has dexel defects"), and the scatter correction " *
                    "subtracts its estimate from corrected counts, as a scanner does: pass " *
                    "`calibration = calibrate(scanner, protocol, recon_opts, phantom)` to create_workspace " *
                    "(or turn the scatter correction off on the scanner)"
            )
        )
        return nothing
    end
    cal isa Calibration || throw(ArgumentError("calibration must come from calibrate(scanner, protocol, recon_opts, phantom)"))
    typeof(cal.scanner) == typeof(scanner) && _detector_fields(cal.scanner) == _detector_fields(scanner) ||
        throw(ArgumentError("the calibration was made for another detector: calibrate this scanner"))
    p, q = cal.protocol, protocol
    (p.kVp == q.kVp && p.additional_filters == q.additional_filters && p.collimation_mm == q.collimation_mm) ||
        throw(ArgumentError("the calibration was made for another beam (kVp, filtration or collimation): calibrate this protocol"))
    (_piles_up(scanner) && cal.pileup === nothing) && throw(ArgumentError("the calibration has no pile-up correction"))
    (_has_defects(scanner) && cal.flat_field === nothing) && throw(ArgumentError("the calibration has no flat field"))
    return cal
end

export Calibration, calibrate
