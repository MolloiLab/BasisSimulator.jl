# =============================================================================
# MC-Simulated Detector Response Matrix (DRM) for PCCT Detectors
# =============================================================================
# Author: Hamidreza Khodajou-Chokami, PhD.
#
# Loads pre-computed CdTe detector response R(E,t) from Monte Carlo (MC)
# transport simulations and converts it to BasisSimulator's DRM format.
#
# The MC response matrix R(E,t) captures the full detector physics:
#   1. Full photon transport in CdTe (photoelectric, Compton, Rayleigh)
#   2. Fano noise on charge generation
#   3. Dreier 2018 charge cloud transport (thermal diffusion + Coulomb repulsion)
#   4. 3×3 pixel erf-based charge splitting (Gaussian charge sharing)
#   5. Electronic noise (per-pixel Gaussian)
#   6. Energy threshold comparison (cumulative counts ≥ threshold)
#
# Because the MC R matrix already includes charge sharing and spatial effects,
# the analytical charge sharing model (apply_charge_sharing!) should be
# DISABLED when using this DRM.
#
# Reference:
# - Stierstorfer K, 2018 — Charge cloud model
# - Dreier 2018 — Charge transport
# =============================================================================

using Serialization

"""
    default_mc_drm_path() -> String

Return the path to the bundled MC detector response file
(`cdte_response_v4.jls`, co-located with this source file in
`src/detector/pcct/`).
"""
function default_mc_drm_path()
    return joinpath(@__DIR__, "cdte_response_v4.jls")
end

"""
    MCResponseData

Container for MC-simulated CdTe detector response data.

# Fields
- `energies_keV::Vector{Int}`: Incident photon energies [keV] (typically 1:140)
- `thresholds_keV::Vector{Int}`: Energy threshold values [keV]
- `R_total::Matrix{Float64}`: [n_energies × n_thresholds] — cumulative counts per
  incident photon above each threshold (summed over 3×3 pixel neighborhood)
- `R_perpixel::Array{Float64,4}`: [n_energies × 3 × 3 × n_thresholds] — per-pixel response
- `Var_total::Matrix{Float64}`: [n_energies × n_thresholds] — variance of total counts
- `Cov_total::Array{Float64,3}`: [n_energies × n_thresholds × n_thresholds] — covariance
- `Cov_pixpair::Array{Float64,5}`: [n_energies × 9 × 9 × n_thresholds × n_thresholds]
"""
struct MCResponseData
    energies_keV::Vector{Int}
    thresholds_keV::Vector{Int}
    R_total::Matrix{Float64}
    R_perpixel::Array{Float64,4}
    Var_total::Matrix{Float64}
    Cov_total::Array{Float64,3}
    Cov_pixpair::Array{Float64,5}
end

"""
    load_mc_response(npz_path::String) -> MCResponseData

Load MC-simulated CdTe detector response from a .npz file.

The NPZ file must contain arrays produced by the MC detector simulation code:
- `energies_keV`: (n_E,) incident energies
- `thresholds_keV`: (n_t,) threshold values
- `R_total`: (n_E, n_t) cumulative counts ≥ threshold per incident photon
- `R_perpixel`: (n_E, 3, 3, n_t) per-pixel response
- `Var_total`: (n_E, n_t) variance
- `Cov_total`: (n_E, n_t, n_t) covariance
- `Cov_pixpair`: (n_E, 9, 9, n_t, n_t) pixel-pair covariance

# Example
```julia
data = load_mc_response(default_mc_drm_path())
println("Energies: \$(data.energies_keV[1])–\$(data.energies_keV[end]) keV")
println("Thresholds: \$(data.thresholds_keV) keV")
println("R_total size: \$(size(data.R_total))")
```
"""
function load_mc_response(npz_path::String)
    @assert isfile(npz_path) "MC response file not found: $npz_path"

    d = open(deserialize, npz_path)

    return MCResponseData(
        Int.(d["energies_keV"]),
        Int.(d["thresholds_keV"]),
        Float64.(d["R_total"]),
        Float64.(d["R_perpixel"]),
        Float64.(d["Var_total"]),
        Float64.(d["Cov_total"]),
        Float64.(d["Cov_pixpair"])
    )
end

"""
    mc_cumulative_to_bins(R_cumulative, thresholds_keV) -> Matrix{Float64}

Convert cumulative threshold counts R(E, ≥t) to differential bin counts.

MC `R_total[E, k]` = expected counts with deposited energy ≥ threshold[k].
This is cumulative (like survival function). For energy BINS we need:

    R_bin[E, k] = R_cumulative[E, k] - R_cumulative[E, k+1]   (for k < n_t)
    R_bin[E, n_t] = R_cumulative[E, n_t]                       (highest bin: ≥ last threshold)

# Arguments
- `R_cumulative`: [n_E × n_t] cumulative response (counts ≥ threshold)
- `thresholds_keV`: Threshold values (must be sorted ascending)

# Returns
- `Matrix{Float64}`: [n_E × n_t] differential bin counts
"""
function mc_cumulative_to_bins(R_cumulative::Matrix{Float64},
    thresholds_keV::Vector{Int})
    n_E, n_t = size(R_cumulative)
    R_bins = zeros(Float64, n_E, n_t)

    for k in 1:(n_t-1)
        # Bin k: counts between threshold[k] and threshold[k+1]
        R_bins[:, k] = max.(R_cumulative[:, k] .- R_cumulative[:, k+1], 0.0)
    end
    # Last bin: counts ≥ highest threshold (everything above)
    R_bins[:, n_t] = max.(R_cumulative[:, n_t], 0.0)

    return R_bins
end

"""
    compute_mc_drm(detector::PhotonCountingDetector, kVp::Real;
                    npz_path::String=default_mc_drm_path(),
                    n_energy_points::Int=200) -> Matrix{Float64}

Compute a Detector Response Matrix from MC simulation data, compatible with
`compute_unified_drm()` output.

Returns D[i, b] = probability that a photon of energy E_i registers in bin b.

# Algorithm
1. Load MC response (cumulative counts ≥ threshold per incident photon)
2. Convert to differential bins (counts in each energy window)
3. Map MC thresholds to detector's target thresholds (interpolation if needed)
4. Interpolate from MC's 1-keV spacing to BasisSimulator's uniform energy grid

# Arguments
- `detector`: PhotonCountingDetector with target energy thresholds
- `kVp`: Maximum tube voltage [keV]
- `npz_path`: Path to MC `detector_response.npz` file (default: bundled file)
- `n_energy_points`: Output energy grid resolution (default: 200)

# Returns
- `Matrix{Float64}`: [n_energy_points × n_bins] DRM matrix

# Example
```julia
det = naeotom_detector_standard()
D = compute_mc_drm(det, 120.0)  # uses bundled response file
# D has same format as compute_unified_drm() output
```
"""
function compute_mc_drm(detector::PhotonCountingDetector, kVp::Real;
    npz_path::String=default_mc_drm_path(),
    n_energy_points::Int=200)
    # Load MC data
    mc = load_mc_response(npz_path)

    target_thresholds = detector.energy_thresholds_keV
    n_target_bins = length(target_thresholds)

    # Step 1: Convert MC cumulative → differential bins (MC thresholds)
    R_mc_bins = mc_cumulative_to_bins(mc.R_total, mc.thresholds_keV)
    # R_mc_bins: [n_mc_E × n_mc_thresh] — counts per incident photon per MC bin

    # Step 2: Map MC threshold bins → target threshold bins
    # If MC and target thresholds match exactly, this is just a selection.
    # Otherwise, map each MC energy bin to the correct target bin.
    mc_thresholds = Float64.(mc.thresholds_keV)
    mc_E = Float64.(mc.energies_keV)
    n_mc_E = length(mc_E)

    # Build target DRM on MC's native energy grid first
    D_native = zeros(Float64, n_mc_E, n_target_bins)

    for e_idx in 1:n_mc_E
        for target_b in 1:n_target_bins
            T_low = target_thresholds[target_b]
            T_high = target_b < n_target_bins ? target_thresholds[target_b+1] : Inf

            # Sum all MC bins whose energy range overlaps with target bin [T_low, T_high)
            for mc_b in 1:length(mc_thresholds)
                mc_T_low = mc_thresholds[mc_b]
                mc_T_high = mc_b < length(mc_thresholds) ? mc_thresholds[mc_b+1] : Inf

                # Check if MC bin overlaps with target bin
                overlap_low = max(T_low, mc_T_low)
                overlap_high = min(T_high, mc_T_high)

                if overlap_low < overlap_high
                    # Fraction of MC bin that overlaps with target bin
                    mc_width = mc_T_high - mc_T_low
                    if isinf(mc_width) && isinf(T_high)
                        # Both are the last bin (≥ highest threshold)
                        frac = 1.0
                    elseif isinf(mc_width)
                        # MC last bin but target has more bins
                        frac = min((overlap_high - overlap_low) / (Float64(kVp) - mc_T_low + 1.0), 1.0)
                    else
                        frac = (overlap_high - overlap_low) / mc_width
                    end

                    D_native[e_idx, target_b] += R_mc_bins[e_idx, mc_b] * frac
                end
            end
        end
    end

    # Step 3: Interpolate to BasisSimulator's uniform energy grid
    E_grid = range(1.0, Float64(kVp), length=n_energy_points)
    D = zeros(Float64, n_energy_points, n_target_bins)

    for b in 1:n_target_bins
        for (i, E) in enumerate(E_grid)
            # Linear interpolation from MC's integer-keV grid
            if E <= mc_E[1]
                D[i, b] = D_native[1, b]
            elseif E >= mc_E[end]
                D[i, b] = D_native[end, b]
            else
                # Find bracketing indices
                idx_low = clamp(Int(floor(E - mc_E[1])) + 1, 1, n_mc_E - 1)
                idx_high = idx_low + 1
                E_low = mc_E[idx_low]
                E_high = mc_E[idx_high]
                t = (E - E_low) / (E_high - E_low)
                D[i, b] = (1.0 - t) * D_native[idx_low, b] + t * D_native[idx_high, b]
            end
        end
    end

    # Ensure physical constraints: row sums ≤ 1.0, values ≥ 0
    for i in 1:n_energy_points
        row_sum = sum(D[i, :])
        if row_sum > 1.0
            D[i, :] ./= row_sum
        end
        for b in 1:n_target_bins
            D[i, b] = max(D[i, b], 0.0)
        end
    end

    return D
end

"""
    compute_mc_count_moments(detector, energies, weights; η=nothing, R=nothing)

Compute spectrum-averaged first and second moments of differential PCCT bin
counts from the bundled Monte Carlo detector response. The returned covariance
is the compound-Poisson covariance per incident spectral photon,
`E[Cov(Y|E) + E[Y|E]E[Y|E]′]`; an ideal one-count categorical detector therefore
reduces exactly to independent Poisson bins (`fano == 1`, zero correlation).

The MC cumulative-threshold covariance is transformed to the detector's
differential bins, interpolated in energy, and scaled to the same DRM row used
by the forward projector. `mean`, `fano`, `correlation`, and `covariance` are
returned as `Float64` arrays.
"""
function compute_mc_count_moments(
        detector::PhotonCountingDetector,
        energies::AbstractVector,
        weights::AbstractVector;
        η = nothing,
        R = nothing,
        response_path::String = default_mc_drm_path(),
    )
    length(energies) == length(weights) || throw(DimensionMismatch(
        "energies and weights must have equal length"
    ))
    isempty(energies) && throw(ArgumentError("energies must be nonempty"))
    all(isfinite, weights) && all(>=(0), weights) || throw(ArgumentError(
        "weights must be finite and nonnegative"
    ))
    sum(weights) > 0 || throw(ArgumentError("weights must have positive sum"))

    mc = load_mc_response(response_path)
    target_t = Float64.(detector.energy_thresholds_keV)
    source_t = Float64.(mc.thresholds_keV)
    n_bins = length(target_t)

    # H maps cumulative counts at MC thresholds to cumulative counts at target
    # thresholds. Standard NAEOTOM thresholds are exact rows; linear threshold
    # interpolation keeps custom threshold sets well-defined.
    H = zeros(Float64, n_bins, length(source_t))
    for (i, t) in enumerate(target_t)
        if t <= first(source_t)
            H[i, 1] = 1
        elseif t >= last(source_t)
            H[i, end] = 1
        else
            hi = searchsortedfirst(source_t, t)
            lo = hi - 1
            α = (t - source_t[lo]) / (source_t[hi] - source_t[lo])
            H[i, lo] = 1 - α
            H[i, hi] = α
        end
    end
    B = zeros(Float64, n_bins, n_bins)
    for b in 1:(n_bins - 1)
        B[b, b] = 1
        B[b, b + 1] = -1
    end
    B[end, end] = 1
    S = B * H

    kVp = Float64(maximum(energies))
    R_use = R === nothing ? compute_mc_drm(detector, kVp) : Float64.(R)
    η_use = η === nothing ?
        quantum_efficiency_vector(detector.material, detector.thickness_mm, energies) : η
    size(R_use, 2) == n_bins || throw(DimensionMismatch(
        "R has $(size(R_use, 2)) bins; detector has $n_bins"
    ))

    mean_eff = zeros(Float64, n_bins)
    second_eff = zeros(Float64, n_bins, n_bins)
    weight_sum = Float64(sum(weights))
    mc_E = Float64.(mc.energies_keV)
    n_R = size(R_use, 1)

    for i in eachindex(energies, weights)
        w = Float64(weights[i]) / weight_sum
        w == 0 && continue
        E = Float64(energies[i])
        if E <= first(mc_E)
            lo = hi = 1; α = 0.0
        elseif E >= last(mc_E)
            lo = hi = length(mc_E); α = 0.0
        else
            hi = searchsortedfirst(mc_E, E)
            lo = hi - 1
            α = (E - mc_E[lo]) / (mc_E[hi] - mc_E[lo])
        end
        r_cum = (1 - α) .* @view(mc.R_total[lo, :]) .+
            α .* @view(mc.R_total[hi, :])
        c_cum = (1 - α) .* @view(mc.Cov_total[lo, :, :]) .+
            α .* @view(mc.Cov_total[hi, :, :])
        m_raw = S * r_cum
        C_raw = S * c_cum * transpose(S)

        r_idx = clamp(round(Int, (E - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        d = Float64.(@view R_use[r_idx, :])
        scale = [m_raw[b] > eps(Float64) ? d[b] / m_raw[b] : 0.0 for b in 1:n_bins]
        Dscale = Diagonal(scale)
        C = Dscale * C_raw * Dscale
        ηi = Float64(η_use[i])
        event_mean = ηi .* d
        # Bernoulli detection gate followed by a compound-Poisson incident
        # process: Cov_total = η(C + dd′).
        event_second = ηi .* (C .+ d * transpose(d))
        mean_eff .+= w .* event_mean
        second_eff .+= w .* event_second
    end

    second_eff .= (second_eff .+ transpose(second_eff)) ./ 2
    fano = [mean_eff[b] > eps(Float64) ? second_eff[b, b] / mean_eff[b] : 1.0
            for b in 1:n_bins]
    correlation = Matrix{Float64}(I, n_bins, n_bins)
    for j in 1:n_bins, i in 1:n_bins
        denom = sqrt(max(second_eff[i, i] * second_eff[j, j], 0.0))
        correlation[i, j] = denom > eps(Float64) ? second_eff[i, j] / denom : (i == j ? 1.0 : 0.0)
    end
    correlation .= clamp.(correlation, -1.0, 1.0)
    (mean = mean_eff, fano = fano, correlation = correlation, covariance = second_eff)
end

export compute_mc_count_moments

"""
    mc_drm_summary(D::Matrix{Float64}, thresholds::AbstractVector, kVp::Real;
                    n_energy_points::Int=200)

Print a summary of the MC-based DRM for diagnostic purposes.
"""
function mc_drm_summary(D::Matrix{Float64}, thresholds::AbstractVector, kVp::Real;
    n_energy_points::Int=200)
    energies = drm_energy_grid(kVp; n_energy_points=size(D, 1))
    n_E, n_bins = size(D)

    println("MC DRM Summary: $(n_E) energies × $(n_bins) bins")
    println("Energy range: $(energies[1])–$(energies[end]) keV")
    println("Target thresholds: $(Float64.(thresholds)) keV")
    println()

    for E_test in [25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0]
        if E_test > kVp
            continue
        end
        idx = clamp(round(Int, (E_test - 1.0) / (Float64(kVp) - 1.0) * (n_E - 1)) + 1, 1, n_E)
        row = D[idx, :]
        println("E=$(Int(E_test)) keV: bins=$(round.(row, digits=4)), sum=$(round(sum(row), digits=4))")
    end

    row_sums = [sum(D[i, :]) for i in 1:n_E]
    println("\nRow sum stats: min=$(round(minimum(row_sums), digits=4)), " *
            "max=$(round(maximum(row_sums), digits=4)), " *
            "mean=$(round(sum(row_sums)/n_E, digits=4))")
end

# =============================================================================
# Exports
# =============================================================================

export MCResponseData, load_mc_response
export mc_cumulative_to_bins, compute_mc_drm
export mc_drm_summary, default_mc_drm_path
