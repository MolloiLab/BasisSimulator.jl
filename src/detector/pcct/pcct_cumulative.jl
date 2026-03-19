# =============================================================================
# Cumulative Threshold Sinograms for Clinical PCCT
# =============================================================================
# Author: Hamidreza Khodajou-Chokami, PhD.
#
# Converts differential energy-bin sinograms from pcct_forward_project()
# to cumulative threshold sinograms as used clinically.
#
# Clinical PCCT (NAEOTOM Alpha):
#   4 thresholds: [20, 35, 55, 70] keV
#   4 differential bins: 20-35, 35-55, 55-70, 70+ keV
#
#   Cumulative readouts:
#     T1 (≥20 keV) = bin1 + bin2 + bin3 + bin4  (all detected photons)
#     T2 (≥35 keV) = bin2 + bin3 + bin4
#     T3 (≥55 keV) = bin3 + bin4
#     T4 (≥70 keV) = bin4                        (high-energy only)
#
#   For dual-energy CT imaging, T1 and T4 are used as low/high:
#     Low-energy sinogram  = T1 (≥20 keV)
#     High-energy sinogram = T4 (≥70 keV)
#
# IMPORTANT: Sinograms are in line-integral (log) domain. Summation must be
# performed in COUNT domain, not log domain:
#   N_b = I0_b × exp(-sino_b)          [convert to counts]
#   N_cumul = Σ N_b                     [sum counts]
#   sino_cumul = -log(N_cumul / I0_cumul)  [convert back]
#
# Reference:
# - MC detector response code
# =============================================================================

import AcceleratedKernels as AK

"""
    cumulative_threshold_sinograms(bins, I0_per_bin;
        low_bins=1:4, high_bins=4:4) -> (low_sino, high_sino)

Convert differential bin sinograms (in line-integral domain) to cumulative
threshold sinograms by summing counts in the appropriate bins.

# Physics

PCCT detectors physically produce cumulative threshold counts:
- T1 (≥20 keV): counts ALL photons above 20 keV → low-energy image
- T4 (≥70 keV): counts only photons above 70 keV → high-energy image

The differential bins from BasisSimulator are:
  bin1 = 20-35, bin2 = 35-55, bin3 = 55-70, bin4 = 70+ keV

So  T1 = bin1 + bin2 + bin3 + bin4,  T4 = bin4  (in count domain).

# Arguments
- `bins::Vector{<:AbstractArray}`: Differential bin sinograms (line-integral domain)
- `I0_per_bin::Vector{<:Real}`: Per-bin I₀ values (photon counts without attenuation)

# Keyword Arguments
- `low_bins::UnitRange=1:4`: Bin indices to sum for low-energy (T1 ≥ 20 keV)
- `high_bins::UnitRange=4:4`: Bin indices to sum for high-energy (T4 ≥ 70 keV)

# Returns
- `(low_sino, high_sino)`: Cumulative sinograms in line-integral domain

# Example
```julia
# After pcct_forward_project with 4 thresholds [20, 35, 55, 70]:
I0_bins = compute_per_bin_I0(...)  # per-bin reference counts
low, high = cumulative_threshold_sinograms(pcct_sino.bins, I0_bins)
# low = T1 (≥20 keV), high = T4 (≥70 keV)
```
"""
function cumulative_threshold_sinograms(
    bins::Vector{A},
    I0_per_bin::Vector{<:Real};
    low_bins::UnitRange{Int} = 1:length(bins),
    high_bins::UnitRange{Int} = length(bins):length(bins)
) where {T, A<:AbstractArray{T}}

    # Allocate output arrays (same size/device as input bins)
    low_sino = similar(bins[1])
    high_sino = similar(bins[1])

    cumulative_threshold_sinograms!(low_sino, high_sino, bins, I0_per_bin;
                                     low_bins=low_bins, high_bins=high_bins)

    return low_sino, high_sino
end

"""
    cumulative_threshold_sinograms!(low_out, high_out, bins, I0_per_bin; ...) -> nothing

In-place version: write cumulative threshold sinograms into pre-allocated buffers.
GPU-compatible via AcceleratedKernels.jl.
"""
function cumulative_threshold_sinograms!(
    low_out::AbstractArray{T},
    high_out::AbstractArray{T},
    bins::Vector{A},
    I0_per_bin::Vector{<:Real};
    low_bins::UnitRange{Int} = 1:length(bins),
    high_bins::UnitRange{Int} = length(bins):length(bins)
) where {T, A<:AbstractArray{T}}

    eps_val = T(1e-10)

    # --- Low-energy cumulative (T1): sum counts across all specified bins ---
    I0_low = T(sum(I0_per_bin[low_bins]))

    # Step 1: Accumulate counts N = Σ I0_b × exp(-sino_b) in count domain
    fill!(low_out, zero(T))
    for b in low_bins
        I0_b = T(I0_per_bin[b])
        let lo = low_out, bin_b = bins[b], I0b = I0_b
            AK.foreachindex(lo) do idx
                lo[idx] += I0b * exp(-bin_b[idx])
            end
        end
    end

    # Step 2: Convert back to line-integral domain: sino = -log(N / I0_cumul)
    let lo = low_out, I0l = I0_low, eps = eps_val
        AK.foreachindex(lo) do idx
            lo[idx] = -log(max(lo[idx], eps) / I0l)
        end
    end

    # --- High-energy cumulative (T4): sum counts across high bins ---
    I0_high = T(sum(I0_per_bin[high_bins]))

    fill!(high_out, zero(T))
    for b in high_bins
        I0_b = T(I0_per_bin[b])
        let hi = high_out, bin_b = bins[b], I0b = I0_b
            AK.foreachindex(hi) do idx
                hi[idx] += I0b * exp(-bin_b[idx])
            end
        end
    end

    let hi = high_out, I0h = I0_high, eps = eps_val
        AK.foreachindex(hi) do idx
            hi[idx] = -log(max(hi[idx], eps) / I0h)
        end
    end

    return nothing
end

"""
    compute_per_bin_I0(detector, energies, weights, I0_total; thresholds=nothing)

Compute per-bin I₀ values from spectrum, quantum efficiency, and total I₀.

Each bin's I₀ is proportional to the fraction of spectrum photons that fall
in that energy range, weighted by quantum efficiency.

# Arguments
- `detector::PhotonCountingDetector`: Detector spec (material, thickness, thresholds)
- `energies::AbstractVector`: Spectrum energies (keV)
- `weights::AbstractVector`: Spectrum weights (photon fluence per energy bin)
- `I0_total::Real`: Total reference photon count per pixel per view

# Returns
- `Vector{Float64}`: Per-bin I₀ values that sum to ≤ I0_total
"""
function compute_per_bin_I0(
    detector::PhotonCountingDetector,
    energies::AbstractVector,
    weights::AbstractVector,
    I0_total::Real;
    thresholds::Union{Nothing, AbstractVector} = nothing
)
    thresh = thresholds === nothing ? detector.energy_thresholds_keV : thresholds
    n_bins = length(thresh)
    kVp = maximum(energies)

    # Quantum efficiency at each energy
    η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)

    # Total weighted fluence (denominator for fractional allocation)
    total_fluence = sum(Float64(weights[e]) * Float64(η[e]) for e in eachindex(energies))

    # For each bin, sum the spectrum fraction in its energy range
    I0_bins = zeros(Float64, n_bins)
    for b in 1:n_bins
        E_low = Float64(thresh[b])
        E_high = b < n_bins ? Float64(thresh[b+1]) : Float64(kVp)

        bin_fluence = 0.0
        for e in eachindex(energies)
            E = Float64(energies[e])
            if E >= E_low && E < E_high
                bin_fluence += Float64(weights[e]) * Float64(η[e])
            end
        end

        I0_bins[b] = Float64(I0_total) * bin_fluence / max(total_fluence, 1e-30)
    end

    return I0_bins
end

# =============================================================================
# Exports
# =============================================================================

export cumulative_threshold_sinograms, cumulative_threshold_sinograms!
export compute_per_bin_I0
