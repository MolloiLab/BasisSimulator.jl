# =============================================================================
# Monte Carlo Pulse Pileup Model for PCCT Detectors
# =============================================================================
#
# Author: Hamidreza Khodajou-Chokami, PhD. (original PR #10)
# Per-photon trigger-bin tracking added in 2026 cleanup so the resulting
# spectral migration matrix S is non-degenerate and column-sums encode
# count loss directly.
#
# This file provides the building blocks for Monte Carlo dead-time pile-up
# simulation:
#   1. `simulate_pulse_train`   — one MC trial: Poisson arrivals → spectrum
#                                 sampling → dead-time gating with energy
#                                 summation; returns per-event recorded
#                                 energies + the bin of the photon that
#                                 triggered each event.
#   2. `compute_mc_pileup_matrix` — averages many trials into the (true_bin →
#                                  recorded_bin) migration matrix S.
#                                  S[i, j] = P(true bin-j photon recorded in
#                                  bin i). Column sums ≤ 1, the deficit being
#                                  the count-loss for bin-j photons that piled
#                                  up into other events.
#
# `simulate!(::PCCTWorkspace)` (see `src/api/driver.jl`) caches S in the
# workspace at construction time and applies it as `recorded = S × counts`,
# combined with a renormalized I0 baseline `I0_recorded = S × I0_truth`.  No
# analytical count-factor / Taguchi fallback is used on the live path —
# `seminonparalyzable_count_factor` and `mc_pileup_count_factor` remain only
# as dose-diagnostic helpers (the former emits a deprecation warning).
#
# References:
# - Taguchi 2010, Med Phys 37:3957-3969 (analytical pileup model — superseded)
# - Yang 2025, Med Phys 52:3658-3674 (seminonparalyzable model)
# - Zambon & Amato 2023, Front Phys 11:1205638 (retrigger model)
# =============================================================================

using Random

"""
    PileupResult

Result from a single MC pulse train simulation.

# Fields
- `n_true::Int`: Number of true photons arriving at detector
- `n_recorded::Int`: Number of recorded events (after pileup/dead-time)
- `true_energies::Vector{Float64}`: Energies of arriving photons [keV]
- `recorded_energies::Vector{Float64}`: Energies of recorded events [keV]
- `true_bin_counts::Vector{Int}`: Photon counts per energy bin (true)
- `recorded_bin_counts::Vector{Int}`: Photon counts per energy bin (recorded)
- `trigger_bins::Vector{Int}`: For each recorded event, the bin of the
  *triggering* photon (the first photon to break dead time).  Photons that
  pile up afterwards contribute energy to the event but do not get their own
  trigger entry — that is the count-loss mechanism.  Used by
  `compute_mc_pileup_matrix` to build a non-degenerate (true → recorded)
  migration matrix.  `0` indicates a sub-threshold trigger.
"""
struct PileupResult
    n_true::Int
    n_recorded::Int
    true_energies::Vector{Float64}
    recorded_energies::Vector{Float64}
    true_bin_counts::Vector{Int}
    recorded_bin_counts::Vector{Int}
    trigger_bins::Vector{Int}
end

"""
    simulate_pulse_train(spectrum_weights, energies, count_rate, dead_time_ns;
                          observation_time_s=1e-3,
                          model=:seminonparalyzable,
                          f_retrigger=0.3,
                          thresholds_keV=[20.0, 35.0, 55.0, 70.0],
                          rng=Random.default_rng()) -> PileupResult

Simulate a single pulse train with dead-time effects and pileup.

Generates Poisson-distributed photon arrival times for one pixel, assigns
energies from the incident spectrum, and applies dead-time gating with
energy summation for piled-up events.

# Arguments
- `spectrum_weights`: Normalized spectral weights (probability per energy bin)
- `energies`: Energy values in keV corresponding to weights
- `count_rate`: True photon count rate [photons/s per pixel]
- `dead_time_ns`: Detector dead time [nanoseconds]

# Keyword Arguments
- `observation_time_s`: Duration to simulate [seconds] (default: 1 ms)
- `model`: Dead-time model — `:nonparalyzable` or `:seminonparalyzable`
- `f_retrigger`: Retrigger probability for seminonparalyzable (default: 0.3)
- `thresholds_keV`: Energy thresholds for binning
- `rng`: Random number generator

# Returns
- `PileupResult` with true and recorded counts per bin
"""
function simulate_pulse_train(
    spectrum_weights::AbstractVector{<:Real},
    energies::AbstractVector{<:Real},
    count_rate::Real,
    dead_time_ns::Real;
    observation_time_s::Real=1e-3,
    model::Symbol=:seminonparalyzable,
    f_retrigger::Float64=0.3,
    thresholds_keV::AbstractVector{<:Real}=[20.0, 35.0, 55.0, 70.0],
    rng::AbstractRNG=Random.default_rng()
)
    τ_s = Float64(dead_time_ns) * 1e-9  # Dead time in seconds
    rate = Float64(count_rate)
    T_obs = Float64(observation_time_s)
    n_thresh = length(thresholds_keV)

    # Normalize spectrum weights to probability distribution
    w = Float64.(spectrum_weights)
    w_sum = sum(w)
    if w_sum > 0.0
        w ./= w_sum
    else
        w .= 1.0 / length(w)
    end

    # Build cumulative distribution for energy sampling
    E_vals = Float64.(energies)
    cdf = cumsum(w)

    # Generate Poisson number of photons in observation window
    expected_count = rate * T_obs
    n_photons = Poisson_approx(expected_count)

    if n_photons == 0
        return PileupResult(0, 0,
            Float64[], Float64[],
            zeros(Int, n_thresh), zeros(Int, n_thresh),
            Int[])
    end

    # Generate arrival times (sorted uniform in [0, T_obs])
    # Explicitly create Vector to avoid scalar dispatch when n_photons=1
    arrival_times = Vector{Float64}(undef, n_photons)
    for i in 1:n_photons
        arrival_times[i] = rand(rng) * T_obs
    end
    sort!(arrival_times)

    # Sample photon energies from spectrum
    photon_energies = Vector{Float64}(undef, n_photons)
    for i in 1:n_photons
        u = rand(rng)
        idx = searchsortedfirst(cdf, u)
        idx = clamp(idx, 1, length(E_vals))
        photon_energies[i] = E_vals[idx]
    end

    # Apply dead-time gating and pileup.
    # `trigger_bins[k]` records the bin of the photon that triggered event k
    # (the photon that broke dead time). Photons that pile up onto an
    # already-open event contribute energy but do NOT seed a new trigger —
    # that is the count-loss mechanism.
    recorded_energies = Float64[]
    trigger_bins      = Int[]
    it = 1
    while it <= n_photons
        # This photon triggers an event
        event_energy = photon_energies[it]
        event_time   = arrival_times[it]
        trigger_bin  = _find_threshold_bin(event_energy, thresholds_keV)

        if model == :seminonparalyzable
            # Seminonparalyzable: dead time can be retriggered
            dead_end = event_time + τ_s
            jt = it + 1
            while jt <= n_photons && arrival_times[jt] < dead_end
                # This photon arrives during dead time → pileup (energy adds)
                event_energy += photon_energies[jt]
                # Check retrigger: with probability f_retrigger, dead time resets
                if rand(rng) < f_retrigger
                    dead_end = arrival_times[jt] + τ_s
                end
                jt += 1
            end
            push!(recorded_energies, event_energy)
            push!(trigger_bins, trigger_bin)
            it = jt
        else
            # Nonparalyzable: fixed dead time, no retrigger
            dead_end = event_time + τ_s
            jt = it + 1
            while jt <= n_photons && arrival_times[jt] < dead_end
                event_energy += photon_energies[jt]
                jt += 1
            end
            push!(recorded_energies, event_energy)
            push!(trigger_bins, trigger_bin)
            it = jt
        end
    end

    # Bin true and recorded photons
    true_bins = zeros(Int, n_thresh)
    rec_bins  = zeros(Int, n_thresh)

    for E in photon_energies
        b = _find_threshold_bin(E, thresholds_keV)
        if b > 0
            true_bins[b] += 1
        end
    end

    for E in recorded_energies
        b = _find_threshold_bin(E, thresholds_keV)
        if b > 0
            rec_bins[b] += 1
        end
    end

    return PileupResult(
        n_photons, length(recorded_energies),
        photon_energies, recorded_energies,
        true_bins, rec_bins,
        trigger_bins,
    )
end

"""
    _find_threshold_bin(E, thresholds) -> Int

Find which energy bin a photon with energy E belongs to.
Returns 0 if below lowest threshold.
"""
function _find_threshold_bin(E::Float64, thresholds::AbstractVector{<:Real})
    n = length(thresholds)
    if E < thresholds[1]
        return 0
    end
    for b in n:-1:1
        if E >= thresholds[b]
            return b
        end
    end
    return 0
end

"""
    Poisson_approx(λ) -> Int

Draw from Poisson distribution using inverse CDF method for small λ
or Gaussian approximation for large λ.
"""
function Poisson_approx(λ::Float64)
    if λ <= 0.0
        return 0
    elseif λ < 30.0
        # Knuth's algorithm for small λ
        L = exp(-λ)
        k = 0
        p = 1.0
        while true
            k += 1
            p *= rand()
            if p < L
                return k - 1
            end
        end
    else
        # Gaussian approximation for large λ
        return max(0, round(Int, λ + sqrt(λ) * randn()))
    end
end

"""
    compute_mc_pileup_matrix(thresholds_keV, spectrum_weights, energies,
                              count_rate, dead_time_ns;
                              n_trials=10000,
                              observation_time_s=1e-3,
                              model=:seminonparalyzable,
                              seed=42) -> Matrix{Float64}

Compute the spectral-migration matrix `S` from Monte Carlo pulse-train
simulations.  `S[i, j]` = fraction of true bin-`j` photons that end up
recorded in bin `i`.  Column sums are ≤ 1 — the column-`j` deficit is the
count-loss for bin-`j` photons that fell during another event's dead time
and contributed energy without seeding a recorded event of their own.

# Algorithm
For each MC trial we run `simulate_pulse_train` (Poisson arrivals →
spectrum-CDF energies → dead-time gating with energy summation), and the
trial result links each recorded event to the bin of its *triggering*
photon (the one that broke dead time). We accumulate
`transition_counts[recorded_bin, trigger_bin] += 1` per event, count every
incident photon in `total_true[bin]`, and divide column-wise:
`S[i, j] = transition_counts[i, j] / total_true[j]`.

This is non-degenerate by construction (each column reflects what happens
to the photons of a specific true bin) and obeys the physical asymmetry
that pile-up can only push counts UP in energy (S[i, j] ≈ 0 for i < j).

# Arguments
- `thresholds_keV`: Energy threshold values [keV]
- `spectrum_weights`: Normalized spectral weights
- `energies`: Energy values [keV]
- `count_rate`: True photon count rate [photons/s per pixel]
- `dead_time_ns`: Detector dead time [ns]

# Keyword Arguments
- `n_trials`: Number of MC pulse trains to simulate (default: 10000)
- `observation_time_s`: Duration per trial (default: 1 ms)
- `model`: `:nonparalyzable` or `:seminonparalyzable`
- `seed`: Random seed for reproducibility

# Returns
- `Matrix{Float64}`: [n_bins × n_bins] spectral migration matrix

# Example
```julia
energies, weights = load_spectrum(120)
S = compute_mc_pileup_matrix(
    [20.0, 35.0, 55.0, 70.0],
    weights, energies,
    1e8, 5.0;   # 100 Mcps, 5 ns dead time
    n_trials=50000
)
```
"""
function compute_mc_pileup_matrix(
    thresholds_keV::AbstractVector{<:Real},
    spectrum_weights::AbstractVector{<:Real},
    energies::AbstractVector{<:Real},
    count_rate::Real,
    dead_time_ns::Real;
    n_trials::Int=10000,
    observation_time_s::Real=1e-3,
    model::Symbol=:seminonparalyzable,
    seed::Int=42
)
    rng = MersenneTwister(seed)
    n_bins = length(thresholds_keV)
    thresh = Float64.(thresholds_keV)

    # ─── Safety guard: cap expected photons per trial ───────────────────────
    # `simulate_pulse_train` allocates `Vector{Float64}` of length `n_photons`
    # for arrival times and energies.  If the user passes an unrealistic
    # `count_rate` (e.g. a synthetic test scanner with overwhelming mAs/view),
    # `expected_count = count_rate × observation_time` can blow up to 10¹¹+
    # photons per trial, requiring hundreds of GB to allocate the arrival-time
    # vector.  Shrink `observation_time` so each trial samples no more than
    # ~10⁶ photons; MC statistics still converge over `n_trials`.
    MAX_PHOTONS_PER_TRIAL = 1.0e6
    obs_time_s = Float64(observation_time_s)
    expected = Float64(count_rate) * obs_time_s
    if expected > MAX_PHOTONS_PER_TRIAL
        scale       = MAX_PHOTONS_PER_TRIAL / expected
        obs_time_s *= scale
        @warn """
        compute_mc_pileup_matrix: expected $(round(expected; sigdigits=3)) photons
        per trial at count_rate=$(count_rate) photons/s — would cause arrival-
        time vector to blow up. Shrunk observation_time_s from $(observation_time_s)
        to $(obs_time_s) so each trial samples ≤ $(MAX_PHOTONS_PER_TRIAL) photons.
        MC statistics still converge across $(n_trials) trials.
        """ maxlog=1
    end

    # transition_counts[i, j] tallies (true bin j → recorded bin i) events.
    # total_true[j] tallies every incident bin-j photon, including those that
    # fell in a dead-time window — the deficit gives count-loss naturally.
    transition_counts = zeros(Float64, n_bins, n_bins)
    total_true        = zeros(Float64, n_bins)

    for trial in 1:n_trials
        result = simulate_pulse_train(
            spectrum_weights, energies, count_rate, dead_time_ns;
            observation_time_s = obs_time_s,
            model = model,
            thresholds_keV = thresh,
            rng = rng,
        )

        # Every incident photon counts toward the true-bin denominator.
        for b in 1:n_bins
            total_true[b] += result.true_bin_counts[b]
        end

        # Each recorded event has a known triggering-photon bin AND a known
        # recorded bin (computed from the energy-summed event energy).  Bin
        # the migration directly: trigger_bin → recorded_bin += 1.
        for k in 1:result.n_recorded
            trig_b = result.trigger_bins[k]
            rec_b  = _find_threshold_bin(result.recorded_energies[k], thresh)
            if trig_b > 0 && rec_b > 0
                transition_counts[rec_b, trig_b] += 1.0
            end
        end
    end

    # Normalize column-wise: S[i, j] = N_recorded(j → i) / N_true(j).
    S = zeros(Float64, n_bins, n_bins)
    for j in 1:n_bins
        if total_true[j] > 0.0
            for i in 1:n_bins
                S[i, j] = transition_counts[i, j] / total_true[j]
            end
        else
            S[j, j] = 1.0  # No data for true bin j → fall back to identity column
        end
    end
    return S
end

# =============================================================================
# Exports
# =============================================================================

export PileupResult, simulate_pulse_train, compute_mc_pileup_matrix
