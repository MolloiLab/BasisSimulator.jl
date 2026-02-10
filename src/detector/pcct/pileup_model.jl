# =============================================================================
# Pileup Model — Physics-Based Analytical Model
# =============================================================================
#
# Implements physics-based pulse pileup models for PCCT detectors.
#
# At high flux, two (or more) photons arrive within the dead time. Their
# pulse heights ADD, creating a recorded energy that is the SUM of the
# individual photon energies. This shifts counts from low thresholds to
# high thresholds.
#
# Key observables (from Konrad 2025):
#   - VMR < 1 at high flux (sub-Poisson from merged independent events)
#   - Counts shift from T1,T2 → T3,T4 with increasing flux
#   - VMR ≈ 1.0 at low flux for all thresholds
#
# Models implemented:
#   1. Nonparalyzable: N_rec = N_true / (1 + N_true × τ)
#   2. Seminonparalyzable (Yang 2025): accounts for dead-time retriggering
#      by pulses during dead time, which is more realistic for modern PCCT
#   3. Spectral migration via Taguchi 2010 pileup-order convolution
#
# References:
# - Taguchi et al 2010, "An analytical model of the effects of pulse pileup
#   on the energy spectrum recorded by energy resolved photon counting x-ray
#   detectors", Med Phys 37:3957-3969
# - Grönberg et al 2018, "Count statistics of nonparalyzable photon-counting
#   detectors with applications to spectral CT", Med Phys 45:3800-3811
# - Yang et al 2025, "Analytical model for pulse pileup spectra and count
#   statistics in photon counting detectors with seminonparalyzable behavior",
#   Med Phys 52:3658-3674
# - Zambon & Amato 2023, "Pulse pileup model for spectral resolved X-ray
#   photon-counting detectors with dead time and retrigger capability",
#   Frontiers in Physics 11:1205638
# =============================================================================

"""
    pileup_order_probability(m::Int, aτ::Real) -> Float64

Probability that exactly m photons contribute to a single recorded event
for a nonparalyzable detector (Taguchi 2010, Eq. 3).

During dead time τ after the triggering photon, additional photons arrive
as a Poisson process with rate a. The number of additional photons m_extra
follows Poisson(aτ). The total pileup order is m = 1 + m_extra.

    P(m) = (aτ)^(m-1) × exp(-aτ) / (m-1)!    for m ≥ 1

This is a shifted Poisson: the triggering photon is always present (m ≥ 1),
and (m-1) additional photons pile up during dead time.

# Arguments
- `m`: Total number of photons in the recorded event (m ≥ 1)
- `aτ`: Product of count rate × dead time (dimensionless)

# Returns
- Probability P(m) ∈ [0, 1]
"""
function pileup_order_probability(m::Int, aτ::Real)
    x = Float64(aτ)
    if x ≤ 0.0 || m < 1
        return m == 1 ? 1.0 : 0.0
    end

    # Shifted Poisson: P(m) = Poisson(m-1; λ=aτ)
    # P(m) = (aτ)^(m-1) × exp(-aτ) / (m-1)!  (Taguchi 2010, Eq. 3)
    k = m - 1  # number of additional pileup photons
    if k == 0
        return exp(-x)
    end

    # Use log to avoid overflow for large m or aτ
    # log(k!) computed iteratively (k is small, typically ≤ 5)
    log_kfact = 0.0
    for i in 2:k
        log_kfact += log(Float64(i))
    end
    log_p = k * log(x) - x - log_kfact
    return exp(log_p)
end

"""
    compute_spectral_migration_matrix(thresholds_keV::AbstractVector,
                                       aτ::Real;
                                       max_pileup_order::Int=6,
                                       kVp::Real=140.0,
                                       bin_weights::Union{Nothing, Vector{Float64}}=nothing
                                       ) -> Matrix{Float64}

Compute the spectral migration matrix S that maps true bin counts to
recorded bin counts under pileup effects (Taguchi 2010).

S[j, i] = probability that a count originally in true bin i is recorded
in bin j, accounting for pileup energy summation.

For pileup of order m, the recorded energy is the sum of m photon energies.
The m-1 additional photons are drawn from the incident spectrum (specified
by `bin_weights`).

# Algorithm (Taguchi 2010, Section II.B)
1. For each pileup order m = 1, 2, ..., max_order:
   a. P(m) = Poisson probability of m piled-up photons
   b. For m=1: no pileup → diagonal entries S[i,i] += P(1)
   c. For m=2: convolve each true bin with incident spectrum
   d. For m≥3: m-fold convolution via recursive application

# Arguments
- `thresholds_keV`: Energy threshold values [keV] defining bins
- `aτ`: Count rate × dead time product (dimensionless)
- `max_pileup_order`: Maximum pileup order to consider (default: 6)
- `kVp`: Maximum tube voltage / recordable energy [keV]
- `bin_weights`: Relative probability of a photon being in each bin.
  If `nothing`, assumes uniform weight across bins (1/n_bins each).
  For best accuracy, pass the fractional counts per bin from the
  incident spectrum.

# Returns
- Matrix{Float64}: [n_bins × n_bins] spectral migration matrix.
  Column i describes what happens to photons truly in bin i.
"""
function compute_spectral_migration_matrix(
    thresholds_keV::AbstractVector,
    aτ::Real;
    max_pileup_order::Int=6,
    kVp::Real=140.0,
    bin_weights::Union{Nothing, Vector{Float64}}=nothing,
    # Workspace scratch buffers (optional — allocate internally if not provided)
    ws_S::Union{Nothing, Matrix{Float64}}=nothing,
    ws_thresh::Union{Nothing, Vector{Float64}}=nothing,
    ws_E_low::Union{Nothing, Vector{Float64}}=nothing,
    ws_E_high::Union{Nothing, Vector{Float64}}=nothing,
    ws_E_centers::Union{Nothing, Vector{Float64}}=nothing,
    ws_w::Union{Nothing, Vector{Float64}}=nothing
)
    n_bins = length(thresholds_keV)
    thresh = if ws_thresh !== nothing
        for i in 1:n_bins; ws_thresh[i] = Float64(thresholds_keV[i]); end
        ws_thresh
    else
        Float64.(thresholds_keV)
    end
    x = Float64(aτ)

    # S[j, i]: probability that count in true bin i → recorded bin j
    S = if ws_S !== nothing
        fill!(ws_S, 0.0)
        ws_S
    else
        zeros(Float64, n_bins, n_bins)
    end

    if x ≤ 0.0 || n_bins < 1
        # No pileup: identity
        for i in 1:n_bins
            S[i, i] = 1.0
        end
        return S
    end

    # Compute bin edge energies (lower bounds + upper bound)
    E_low = if ws_E_low !== nothing; ws_E_low else zeros(Float64, n_bins) end
    E_high = if ws_E_high !== nothing; ws_E_high else zeros(Float64, n_bins) end
    for b in 1:n_bins
        E_low[b] = thresh[b]
        E_high[b] = b < n_bins ? thresh[b+1] : Float64(kVp)
    end

    # Bin center energies (used for convolution)
    E_centers = if ws_E_centers !== nothing; ws_E_centers else zeros(Float64, n_bins) end
    for b in 1:n_bins
        E_centers[b] = (E_low[b] + E_high[b]) / 2.0
    end

    # Incident spectrum weights for the pileup companion photon
    # Default: uniform across bins; better: actual spectrum distribution
    w = if ws_w !== nothing
        if bin_weights === nothing
            fill!(ws_w, 1.0 / n_bins)
        else
            s = sum(bin_weights)
            if s > 0.0
                for i in 1:n_bins; ws_w[i] = bin_weights[i] / s; end
            else
                fill!(ws_w, 1.0 / n_bins)
            end
        end
        ws_w
    elseif bin_weights === nothing
        fill(1.0 / n_bins, n_bins)
    else
        # Normalize to probability
        s = sum(bin_weights)
        s > 0.0 ? bin_weights ./ s : fill(1.0 / n_bins, n_bins)
    end

    # Bin-finding: which bin does energy E fall into?
    function find_bin(E::Float64)
        if E < thresh[1]
            return 0  # Below lowest threshold
        end
        for b in n_bins:-1:1
            if E ≥ thresh[b]
                return b
            end
        end
        return 0
    end

    # =========================================================================
    # Pileup order m=1: no pileup → diagonal
    # =========================================================================
    P1 = pileup_order_probability(1, x)
    for i in 1:n_bins
        S[i, i] += P1
    end

    # =========================================================================
    # Pileup order m=2: two photons combine
    # Primary photon from bin i, companion photon from incident spectrum
    # Recorded energy = E_primary + E_companion
    # =========================================================================
    P2 = pileup_order_probability(2, x)
    if P2 > 1e-12
        for i in 1:n_bins
            for k in 1:n_bins
                E_sum = E_centers[i] + E_centers[k]
                j = find_bin(E_sum)
                if j > 0 && j ≤ n_bins
                    # Weight by probability of companion in bin k
                    S[j, i] += P2 * w[k]
                end
                # If E_sum > kVp → count lost (not registered)
            end
        end
    end

    # =========================================================================
    # Pileup order m=3: three photons combine
    # Primary + 2 companions from incident spectrum
    # Recorded energy = E_primary + E_companion1 + E_companion2
    # =========================================================================
    P3 = pileup_order_probability(3, x)
    if P3 > 1e-12
        for i in 1:n_bins
            for k1 in 1:n_bins
                for k2 in 1:n_bins
                    E_sum = E_centers[i] + E_centers[k1] + E_centers[k2]
                    j = find_bin(E_sum)
                    if j > 0 && j ≤ n_bins
                        S[j, i] += P3 * w[k1] * w[k2]
                    end
                end
            end
        end
    end

    # =========================================================================
    # Pileup order m≥4: approximate with mean companion energy
    # For m companions, sum ≈ E_primary + (m-1) × E_mean
    # =========================================================================
    E_mean = sum(w[k] * E_centers[k] for k in 1:n_bins)
    for m in 4:max_pileup_order
        Pm = pileup_order_probability(m, x)
        if Pm < 1e-12
            break
        end
        for i in 1:n_bins
            E_sum = E_centers[i] + (m - 1) * E_mean
            j = find_bin(E_sum)
            if j > 0 && j ≤ n_bins
                S[j, i] += Pm
            end
            # Most m≥4 sums exceed kVp → lost (accounted for by col_sum < 1)
        end
    end

    # Column sums should not exceed 1.0 (probability conservation)
    # Values < 1 represent counts lost above kVp
    for i in 1:n_bins
        col_sum = sum(S[:, i])
        if col_sum > 1.0
            S[:, i] ./= col_sum
        end
    end

    return S
end

# =============================================================================
# Count Rate Models
# =============================================================================

"""
    recorded_count_rate(true_rate::Real, τ_s::Real;
                        model::Symbol=:nonparalyzable) -> Float64

Compute recorded count rate for dead-time models.

## Nonparalyzable (default)
    N_rec = N_true / (1 + N_true × τ)
The detector has a fixed dead time after each recorded event.
New events during dead time are lost but do NOT reset the clock.
(Taguchi 2010, Eq. 2; Konrad 2025)

## Paralyzable
    N_rec = N_true × exp(-N_true × τ)
Each event resets the dead time, causing the detector to "lock up"
at very high rates (maximum at N_true = 1/τ).

## Seminonparalyzable (Yang 2025)
    N_rec ≈ N_true / (1 + N_true × τ_eff)
where τ_eff = τ × (1 + f_retrigger × aτ)
Dead time is retriggered by pulses above a retrigger threshold.
This produces stronger count loss than pure nonparalyzable at high flux.
The retrigger factor f_retrigger ≈ 0.3-0.5 for typical PCCT.

# Arguments
- `true_rate`: True count rate [counts/s per pixel]
- `τ_s`: Dead time [seconds]
- `model`: `:nonparalyzable` (default), `:paralyzable`, or `:seminonparalyzable`

# Returns
- Recorded count rate [counts/s per pixel]
"""
function recorded_count_rate(true_rate::Real, τ_s::Real;
                             model::Symbol=:nonparalyzable)
    a = Float64(true_rate)
    τ = Float64(τ_s)

    if a ≤ 0.0 || τ ≤ 0.0
        return a
    end

    if model == :nonparalyzable
        # Taguchi 2010, Eq. 2
        return a / (1.0 + a * τ)
    elseif model == :paralyzable
        return a * exp(-a * τ)
    elseif model == :seminonparalyzable
        # Yang 2025: dead time is retriggered by pulses during dead time
        # Effective dead time increases with flux due to retrigger events
        # τ_eff = τ × (1 + f_retrigger × aτ)
        # f_retrigger depends on pulse shape and retrigger threshold
        # For CdTe PCCT: f_retrigger ≈ 0.3 (moderate retrigger probability)
        f_retrigger = 0.3
        aτ = a * τ
        τ_eff = τ * (1.0 + f_retrigger * aτ)
        return a / (1.0 + a * τ_eff)
    else
        # Fallback to nonparalyzable
        return a / (1.0 + a * τ)
    end
end

"""
    seminonparalyzable_count_factor(aτ::Real; f_retrigger::Float64=0.3) -> Float64

Compute the count-loss factor for the seminonparalyzable model (Yang 2025).

    factor = 1 / (1 + aτ × (1 + f_retrigger × aτ))

This gives stronger count loss than pure nonparalyzable at high flux due
to dead-time retriggering.

# Arguments
- `aτ`: Count rate × dead time product
- `f_retrigger`: Retrigger probability factor (0.0 = nonparalyzable, ~0.3 for CdTe PCCT)

# Returns
- Count factor ∈ (0, 1]: multiply true counts by this to get recorded counts
"""
function seminonparalyzable_count_factor(aτ::Real; f_retrigger::Float64=0.3)
    x = Float64(aτ)
    if x ≤ 0.0
        return 1.0
    end
    # N_rec/N_true = 1 / (1 + aτ_eff) where aτ_eff = aτ × (1 + f_retrigger × aτ)
    aτ_eff = x * (1.0 + f_retrigger * x)
    return 1.0 / (1.0 + aτ_eff)
end

# =============================================================================
# VMR (Variance-to-Mean Ratio) Prediction
# =============================================================================

"""
    pileup_vmr_prediction(aτ::Real; model::Symbol=:nonparalyzable) -> Float64

Predict the variance-to-mean ratio (VMR) for pileup-affected counts.

## Nonparalyzable (Grönberg 2018, Eq. approx):
    VMR ≈ 1 / (1 + aτ)²

## Seminonparalyzable (Yang 2025):
    VMR ≈ 1 / (1 + aτ_eff)²
where aτ_eff accounts for retrigger.

At low flux (aτ → 0): VMR → 1 (Poisson)
At high flux (aτ → ∞): VMR → 0 (sub-Poisson)

This sub-Poisson behavior occurs because pileup MERGES independent Poisson
events. Merging two independent counts into one reduces variance (each
merge removes one degree of freedom from the Poisson process).

# Arguments
- `aτ`: Count rate × dead time product
- `model`: `:nonparalyzable` (default) or `:seminonparalyzable`

# Returns
- VMR prediction (< 1 at high flux)

# References
- Grönberg et al 2018, Med Phys 45:3800-3811
- Yang et al 2025, Med Phys 52:3658-3674
"""
function pileup_vmr_prediction(aτ::Real; model::Symbol=:nonparalyzable)
    x = Float64(aτ)
    if x ≤ 0.0
        return 1.0
    end

    if model == :seminonparalyzable
        # Yang 2025: stronger VMR reduction due to retrigger
        f_retrigger = 0.3
        aτ_eff = x * (1.0 + f_retrigger * x)
        return 1.0 / (1.0 + aτ_eff)^2
    else
        # Nonparalyzable: Grönberg 2018
        return 1.0 / (1.0 + x)^2
    end
end

"""
    binned_pixel_vmr(aτ::Real, n_pixels::Int, p_share::Float64) -> Float64

Predict VMR for n×n binned (summed) pixels, accounting for charge sharing
correlation between adjacent pixels (Konrad 2025, Eq. 4).

When pixels are summed, charge-sharing creates positive correlations:
a photon shared between two pixels is double-counted in the sum, adding
extra variance (super-Poisson, VMR > 1 at low flux).

At high flux, pileup dominates and drives VMR below 1 even for binned pixels.

    VMR_binned ≈ VMR_single × (1 + (n_pixels - 1) × 2 × p_share × ρ_share)

where ρ_share is the correlation coefficient from charge sharing.

# Arguments
- `aτ`: Count rate × dead time product
- `n_pixels`: Number of pixels summed (e.g., 4 for 2×2 binning)
- `p_share`: Charge sharing probability (0.0-1.0)

# Returns
- VMR prediction for binned pixel group

# Reference
- Konrad 2025 (PMB 70:065004), Section 3.1, Eq. 4, Figs. 5-6
"""
function binned_pixel_vmr(aτ::Real, n_pixels::Int, p_share::Float64)
    vmr_single = pileup_vmr_prediction(aτ)

    # Charge sharing correlation contribution (Konrad 2025, Eq. 4)
    # Each shared event creates correlated counts in adjacent pixels.
    # For T1 (lowest threshold): most sensitive to charge sharing correlations.
    # Correlation ρ ≈ p_share × fraction of perimeter shared
    # For a pixel in a 2×2 group: 2 cardinal neighbors are in the group
    # out of 4 total cardinal neighbors → fraction ≈ 0.5
    n_shared_pairs = n_pixels - 1  # simplified: pairs in binned group
    ρ_contribution = n_shared_pairs * 2.0 * p_share * 0.5 / n_pixels

    vmr_binned = vmr_single * (1.0 + ρ_contribution)
    return max(vmr_binned, 0.0)
end

# =============================================================================
# Exports
# =============================================================================

export pileup_order_probability, compute_spectral_migration_matrix
export recorded_count_rate, seminonparalyzable_count_factor
export pileup_vmr_prediction, binned_pixel_vmr
