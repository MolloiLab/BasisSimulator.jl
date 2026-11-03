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

"Forget every memoised pile-up migration matrix (see [`compute_mc_pileup_matrix`](@ref))."
empty_pileup_cache!() = lock(() -> (empty!(_PILEUP_MATRIX_CACHE); nothing), _PILEUP_MATRIX_LOCK)

export compute_mc_pileup_matrix, empty_pileup_cache!
