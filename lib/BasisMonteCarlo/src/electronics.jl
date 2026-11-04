# =============================================================================
# electronics.jl: what a photon-counting ASIC does with each dexel's charge
#
# Every dexel that records charge from a photon makes one pulse. The shaper turns each pulse into a
# CR-RC^n pulse; pulses arriving close together add (pile-up). One comparator per threshold counts
# them, blind for a dead time after each count. Each dexel's thresholds and gain sit a little off
# the nominal (dexel-to-dexel variation).
# =============================================================================

"""
    PulseShaper(; fwhm_ns, dead_ns, order = 1, retrigger = false)

A photon-counting ASIC's shaper and comparators, for [`pulse_counts`](@ref):

- `fwhm_ns`: the width of a shaped pulse at half its height; the pulse is CR-RC^`order`,
  h (t/(nτ))ⁿ exp(n − t/τ), peaking at height h after nτ;
- `dead_ns`: how long a comparator ignores its threshold after it counts;
- `retrigger`: `false` counts every upward crossing of a threshold; `true` counts every peak
  above it, so pulses that pile up without dropping back below the threshold still count once each.
"""
Base.@kwdef struct PulseShaper
    fwhm_ns::Float64
    dead_ns::Float64
    order::Int = 1
    retrigger::Bool = false
end

"""A CR-RC^`n` pulse of unit height and time constant `τ` at time `t` after it starts."""
shaper_pulse(t, τ, n) = t <= 0 ? 0.0 : (t / (n * τ))^n * exp(n - t / τ)

"""The full width at half maximum of a CR-RC^`n` pulse whose time constant is 1."""
function shaper_fwhm(n)
    f(x) = shaper_pulse(x, 1.0, n) - 0.5
    function bisect(a, b)
        for _ in 1:80
            m = (a + b) / 2
            f(a) * f(m) <= 0 ? (b = m) : (a = m)
        end
        return (a + b) / 2
    end
    return bisect(Float64(n), 20.0n) - bisect(1.0e-9, Float64(n))
end

"""
    pulse_counts(rng, shaper::PulseShaper, rate_per_ns, n_pulses, values, cdf, thresholds_keV;
                 noise = h -> 0.0, dt_ns = 0.2)

Counts above each of `thresholds_keV` from `n_pulses` pulses arriving at random at `rate_per_ns` on
one dexel: heights drawn from the distribution `values` with cumulative `cdf`, plus Gaussian noise of
width `noise(h)`, shaped and counted by `shaper`. Pulses farther apart than a pulse's length are
processed one group at a time, so the cost grows with the pulses, not the time they span.
"""
function pulse_counts(
        rng,
        shaper::PulseShaper,
        rate_per_ns,
        n_pulses,
        values,
        cdf,
        thresholds_keV;
        noise = h -> 0.0,
        dt_ns = 0.2,
    )
    n = shaper.order
    τ = shaper.fwhm_ns / shaper_fwhm(n)
    support = (n + 12) * τ
    n_kernel = ceil(Int, support / dt_ns)

    times = cumsum(Random.randexp(rng, n_pulses) ./ rate_per_ns)
    heights = [values[min(searchsortedfirst(cdf, rand(rng)), length(values))] for _ in 1:n_pulses]
    heights .+= [randn(rng) * noise(h) for h in heights]

    counts = zeros(Int, length(thresholds_keV))
    last = fill(-Inf, length(thresholds_keV))
    signal = Float64[]
    i = 1
    while i <= n_pulses
        j = i
        while j < n_pulses && times[j + 1] - times[j] < support
            j += 1
        end

        t0 = times[i]
        n_points = ceil(Int, (times[j] - t0) / dt_ns) + n_kernel + 1
        resize!(signal, n_points)
        fill!(signal, 0.0)
        for p in i:j
            offset = (times[p] - t0) / dt_ns
            start = floor(Int, offset)
            for k in 1:n_kernel
                index = start + k + 1
                index > n_points && break
                signal[index] += heights[p] * shaper_pulse((start + k - offset) * dt_ns, τ, n)
            end
        end

        for (m, T) in enumerate(thresholds_keV), k in 2:(n_points - 1)
            event = shaper.retrigger ?
                (signal[k] >= T && signal[k - 1] < signal[k] >= signal[k + 1]) :
                (signal[k - 1] < T <= signal[k])
            if event
                time = t0 + (k - 1) * dt_ns
                if time - last[m] >= shaper.dead_ns
                    counts[m] += 1
                    last[m] = time
                end
            end
        end
        i = j + 1
    end
    return counts
end

"""
    pulse_heights(r::DexelResponse; floor_keV = 1.0, edges_keV = 0.0:0.5:150.0)

The pulses one dexel makes per photon landing on it, histogrammed on `edges_keV`: every dexel of
the neighbourhood that records more than `floor_keV` from a photon makes a pulse. By symmetry, a
neighbour's pulse from a photon on the target dexel is the target's pulse from a photon on that
neighbour, so in a flat field these are the target dexel's own pulses.
"""
function pulse_heights(r::DexelResponse; floor_keV = 1.0, edges_keV = 0.0:0.5:150.0)
    histogram = zeros(length(edges_keV) - 1)
    for value in r.dexel_energy_keV
        value > floor_keV || continue
        histogram[min(searchsortedlast(edges_keV, value), length(histogram))] += 1 / r.n
    end
    return histogram
end

"""
    DexelVariation(; threshold_offset_keV = 0.0, gain = 1.0)

How one dexel departs from the nominal: its thresholds sit `threshold_offset_keV` off, and it
records `gain` times the nominal charge.
"""
Base.@kwdef struct DexelVariation
    threshold_offset_keV::Float64 = 0.0
    gain::Float64 = 1.0
end

"""
    counts_above(values, p, thresholds_keV; variation = DexelVariation(),
                 electronic_noise_keV = 1.5, fano = 0.1, pair_energy_eV = 4.43)

Counts per pulse above each of `thresholds_keV` for pulses of heights `values` (keV) with
probabilities `p`, each seen with Gaussian noise of width √(σₑ² + F w h), by a dexel that departs
from the nominal as `variation` says. Pulses arriving alone: no pile-up.
"""
function counts_above(
        values,
        p,
        thresholds_keV;
        variation::DexelVariation = DexelVariation(),
        electronic_noise_keV::Real = 1.5,
        fano::Real = 0.1,
        pair_energy_eV::Real = 4.43,
    )
    w = pair_energy_eV / 1000
    g, δ = variation.gain, variation.threshold_offset_keV
    noise(h) = sqrt(electronic_noise_keV^2 + fano * w * max(h, 0.0))
    above(T, h) = (1 + erf((g * h - (T + δ)) / (sqrt(2) * noise(g * h)))) / 2
    return [sum(p[i] * above(T, h) for (i, h) in enumerate(values)) for T in thresholds_keV]
end
