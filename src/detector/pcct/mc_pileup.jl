# =============================================================================
# Pulse pile-up of a photon-counting dexel
# =============================================================================
#
# The pile-up model fitted in BasisMonteCarlo notebook 03 (section 10) to the count loss of two
# clinical NAEOTOM Alphas (Li et al. 2025, Med Phys 52:e70045), re-implemented from its saved
# inputs (`pcct_native_table().pileup`): each photon makes `pulses_per_photon` pulses in a dexel
# (charge sharing gives a dexel pulses from its neighbours' photons too) with heights
# `heights[E, h]`; every pulse is a first-order CR-RC shape of FWHM `shaper_fwhm_ns`; pulses that
# overlap add; one retriggering comparator per threshold counts every peak of the summed signal
# above its threshold at least `dead_time_ns` after its own last count. Each dexel piles up on its
# own: a 2 × 2 binned pixel is the digital sum of four counters, each at a quarter of its rate.
# =============================================================================

using Random

const _PILEUP_MATRIX_CACHE = Dict{UInt64, Matrix{Float64}}()
const _PILEUP_MATRIX_LOCK = ReentrantLock()

# A first-order CR-RC pulse of unit height: (x/τ)·e^{1 − x/τ}, whose FWHM is 2.4464 τ.
const _CRRC_FWHM_PER_TAU = 2.446386
@inline _crrc(x::Float64, τ::Float64) = x <= 0 ? 0.0 : (x / τ) * exp(1 - x / τ)

"""
    compute_mc_pileup_matrix(thresholds_keV, spectrum_weights, energies, count_rate,
                             dead_time_ns; n_pulses = 200_000, seed = 42) -> Matrix

The pile-up of one dexel at `count_rate` incident photons per second, as the migration matrix of
its energy bins: `S[i, j]`, the counts recorded in bin `i` per count of bin `j` at zero rate. A
pulse train of `n_pulses` pulses from the spectrum (`spectrum_weights` on `energies`) runs
through the shaper and one retriggering comparator per threshold with dead time
`dead_time_ns`; each count is charged to the bin of the pulse that dominates its peak. Columns
sum to less than one: piled-up pulses are lost, or counted in a higher bin. Memoised per
argument set ([`empty_pileup_cache!`](@ref)).
"""
function compute_mc_pileup_matrix(
        thresholds_keV::AbstractVector{<:Real},
        spectrum_weights::AbstractVector{<:Real},
        energies::AbstractVector{<:Real},
        count_rate::Real,
        dead_time_ns::Real;
        n_pulses::Integer = 200_000,
        seed::Integer = 42,
    )
    key = hash(
        (
            Float64.(thresholds_keV), Float64.(spectrum_weights), Float64.(energies),
            Float64(count_rate), Float64(dead_time_ns), n_pulses, seed,
        )
    )
    cached = lock(() -> get(_PILEUP_MATRIX_CACHE, key, nothing), _PILEUP_MATRIX_LOCK)
    cached === nothing || return copy(cached)

    pu = pcct_native_table().pileup
    thr = Float64.(thresholds_keV)
    nb = length(thr)
    w = _table_weights(pu.energies, energies, spectrum_weights)
    H = vec(sum(w .* pu.heights; dims = 1))                 # pulses per photon, per height bin
    per_photon = sum(H)
    rate_ns = Float64(count_rate) * per_photon * 1.0e-9     # pulses per ns
    S = Matrix{Float64}(I, nb, nb)
    if rate_ns > 0 && per_photon > 0
        S = _comparator_migration(
            thr, H ./ per_photon, pu.edges, rate_ns, Float64(dead_time_ns),
            pu.shaper_fwhm_ns / _CRRC_FWHM_PER_TAU, Int(n_pulses), MersenneTwister(seed),
        )
    end
    lock(() -> (_PILEUP_MATRIX_CACHE[key] = copy(S)), _PILEUP_MATRIX_LOCK)
    return S
end

# The pulse train: arrival times, heights and their bins; the summed signal's peaks, each
# charged to its dominant pulse; one comparator per threshold.
function _comparator_migration(thr, p_height, edges, rate_ns, dead_ns, τ, n, rng)
    nb = length(thr)
    cdf = cumsum(p_height)
    centres = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    t = cumsum(randexp(rng, n) ./ rate_ns)
    h = [centres[min(searchsortedfirst(cdf, rand(rng) * cdf[end]), length(centres))] for _ in 1:n]
    bin_of(x) = searchsortedlast(thr, x)                    # 0 below the lowest threshold
    truth = zeros(nb)
    for x in h
        b = bin_of(x)
        b > 0 && (truth[b] += 1)
    end

    reach = 10τ                                             # a pulse is spent after 10 τ
    dt = τ / 16
    peaks_t, peaks_s, peaks_bin = Float64[], Float64[], Int[]
    i = 1
    while i <= n
        j = i                                               # the cluster of overlapping pulses
        while j < n && t[j + 1] - t[j] < reach
            j += 1
        end
        if j == i                                           # an isolated pulse peaks at its height, τ after it
            push!(peaks_t, t[i] + τ)
            push!(peaks_s, h[i])
            push!(peaks_bin, bin_of(h[i]))
            i += 1
            continue
        end
        lo = i
        s_prev2, s_prev, peak_dom = -Inf, -Inf, 0
        for x in t[i]:dt:(t[j] + reach)
            while lo <= j && x - t[lo] > reach
                lo += 1
            end
            s, best, dom = 0.0, -1.0, 0
            for k in lo:j
                t[k] > x && break
                c = h[k] * _crrc(x - t[k], τ)
                s += c
                c > best && (best = c; dom = k)
            end
            if s_prev > s_prev2 && s_prev >= s && s_prev > 0
                push!(peaks_t, x - dt)
                push!(peaks_s, s_prev)
                push!(peaks_bin, peak_dom)
            end
            s_prev2, s_prev = s_prev, s
            peak_dom = dom > 0 ? bin_of(h[dom]) : 0
        end
        i = j + 1
    end

    C = zeros(nb, nb)                                       # counts at comparator k, by bin
    for k in 1:nb
        last = -Inf
        for (x, s, b) in zip(peaks_t, peaks_s, peaks_bin)
            (s > thr[k] && x - last >= dead_ns) || continue
            last = x
            b > 0 && (C[k, b] += 1)
        end
    end
    S = zeros(nb, nb)
    for j in 1:nb
        truth[j] > 0 || (S[j, j] = 1; continue)
        for i in 1:nb
            S[i, j] = (C[i, j] - (i < nb ? C[i + 1, j] : 0.0)) / truth[j]
        end
    end
    return S
end

"""
    counted_per_photon(thresholds_keV, spectrum_weights, energies) -> Float64

The pulses a dexel counts (above its lowest threshold) per photon of the spectrum at zero rate,
from the pile-up model's pulse heights: the factor between a dexel's counted rate and the photon
rate [`compute_mc_pileup_matrix`](@ref) takes.
"""
function counted_per_photon(thresholds_keV, spectrum_weights, energies)
    pu = pcct_native_table().pileup
    w = _table_weights(pu.energies, energies, spectrum_weights)
    H = vec(sum(w .* pu.heights; dims = 1)) ./ sum(w)
    centres = (pu.edges[1:(end - 1)] .+ pu.edges[2:end]) ./ 2
    return sum(H[centres .>= minimum(thresholds_keV)])
end

# The mean bin (1 … nb) of a dexel's counts: the hardness index of its spectrum.
@inline function _mean_bin(n::NTuple{NB, T}) where {NB, T}
    tot = sum(n)
    return tot > 0 ? sum(ntuple(b -> T(b) * n[b], Val(NB))) / tot : one(T)
end

# The family's spectra: the source spectrum through 0, 1, 2, 4, 8, 16 and 32 cm of water. A
# dexel's spectrum (the bowtie's and the patient's hardening) is placed in it by its mean bin.
const PILEUP_HARDENING_CM = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

# The family's counted-rate step in units of rate × dead time (2e6 counts per second at the
# NAEOTOM Alpha's 27 ns): the same absolute rates in every workspace of a detector, so a calibration
# at another tube current sees the same pile-up. A family needing more than 64 rates (a toy protocol
# far beyond a scanner's rates) spreads 64 over its range instead.
const PILEUP_RATE_STEP_TAU = 0.054
const PILEUP_MAX_RATES = 64

"""
    pileup_family(thresholds_keV, spectrum_weights, energies, rate_max, dead_time_ns)
        -> (; S, rates, hardness)

The pile-up of one dexel as a function of its counted rate and its spectrum: the migration
matrices of [`compute_mc_pileup_matrix`](@ref), `S[:, :, k, m]`, at counted rates `rates[k]` (counts per
second per dexel above the lowest threshold, evenly spaced from 0 to `rate_max` by 0.054 / τ —
2e6 at the NAEOTOM Alpha's 27 ns — the same absolute rates in every workspace of a detector) and at the spectrum hardened through `PILEUP_HARDENING_CM[m]` of water,
whose mean bin is `hardness[m]`. Every matrix draws the same pulse train (one seed, common random
numbers), so the family is smooth in both. A dexel takes the matrix bilinear in its counted rate and
its mean bin. With one threshold the spectrum cannot be told apart, and the family has the source
spectrum only (twice, so that it still has two hardness nodes).
"""
function pileup_family(thresholds_keV, spectrum_weights, energies, rate_max::Real, dead_time_ns::Real)
    thr = Float64.(thresholds_keV)
    nb = length(thr)
    e = Float64.(energies)
    w0 = Float64.(spectrum_weights) ./ sum(spectrum_weights)
    μ = [ustrip(u"cm^-1", XA.linear_attenuation_coeff(XA.Materials.water, x * u"keV")) for x in e]
    cms = nb > 1 ? collect(PILEUP_HARDENING_CM) : [0.0, 0.0]
    spectra = map(cms) do t
        w = w0 .* exp.(.-μ .* t)
        w ./ sum(w)
    end
    step = max(PILEUP_RATE_STEP_TAU / (Float64(dead_time_ns) * 1.0e-9), Float64(rate_max) / (PILEUP_MAX_RATES - 1))
    K = max(2, ceil(Int, Float64(rate_max) / step) + 1)
    rates = collect(0:(K - 1)) .* step
    # the mean bin of each spectrum at zero rate, from the pulse heights the matrices count
    pu = pcct_native_table().pileup
    centres = (pu.edges[1:(end - 1)] .+ pu.edges[2:end]) ./ 2
    hardness = map(spectra) do w
        H = vec(sum(_table_weights(pu.energies, e, w) .* pu.heights; dims = 1))
        counts = [sum(H[(centres .>= thr[b]) .& (b < nb ? centres .< thr[b + 1] : trues(length(centres)))]) for b in 1:nb]
        sum((1:nb) .* counts) / sum(counts)
    end
    nb == 1 && (hardness = [1.0, 2.0])
    per_photon = [counted_per_photon(thr, w, e) for w in spectra]
    S = zeros(nb, nb, K, length(spectra))
    jobs = [(k, m) for k in 1:K for m in eachindex(spectra)]
    Threads.@threads for (k, m) in jobs
        S[:, :, k, m] = k == 1 ? Matrix{Float64}(I, nb, nb) :
            compute_mc_pileup_matrix(thr, spectra[m], e, rates[k] / per_photon[m], dead_time_ns; seed = 42)
    end
    return (; S, rates, hardness)
end

"Forget every memoised pile-up migration matrix (see [`compute_mc_pileup_matrix`](@ref))."
empty_pileup_cache!() = lock(() -> (empty!(_PILEUP_MATRIX_CACHE); nothing), _PILEUP_MATRIX_LOCK)

export compute_mc_pileup_matrix, empty_pileup_cache!, pileup_family, counted_per_photon
