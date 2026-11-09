# =============================================================================
# MC-Simulated Detector Response Matrix (DRM) for PCCT Detectors
# =============================================================================
# The CdTe detector response R(E, t), counts above each threshold per incident photon, from the
# native-dexel Geant4 table (native_table.jl): photon transport in CdTe, the transported charge
# (drift, diffusion, repulsion, Shockley–Ramo induction, trapping), electronic noise, the
# anti-scatter grid. Converted here to the per-bin response matrix the projector applies.
# =============================================================================

using Serialization

"""
    default_mc_drm_path()

The photon-counting response BasisSimulator uses: [`PCCT_NATIVE_TABLE_PATH`](@ref), the NAEOTOM
Alpha's CdTe from Geant4 (the cell's mean native dexel, the anti-scatter grid included).
"""
default_mc_drm_path() = PCCT_NATIVE_TABLE_PATH

"""
    MCResponseData

A photon-counting response: `energies_keV` and `thresholds_keV` (`Int`), `R_total[E, t]` (the
counts above each threshold per incident photon, absorption included) and `Cov_total[E, ti, tj]`
(their covariance per incident photon).
"""
struct MCResponseData
    energies_keV::Vector{Int}
    thresholds_keV::Vector{Int}
    R_total::Matrix{Float64}
    Cov_total::Array{Float64, 3}
end

"""
    load_mc_response(path::String) -> MCResponseData

Read a photon-counting response: the native-dexel table ([`PCCT_NATIVE_TABLE_PATH`](@ref), read
as the grid cell's mean dexel) or a single-dexel table with `energies_keV`, `thresholds_keV`,
`R_total` and `Cov_total` (BasisSimulator v0.15.1's `cdte_response_v4`).
"""
function load_mc_response(path::String)
    isfile(path) || throw(ArgumentError("photon-counting response not found: $(path)"))
    path == PCCT_NATIVE_TABLE_PATH && return _native_table_response(pcct_native_table())
    d = open(deserialize, path)
    haskey(d, "schema_version") &&
        throw(ArgumentError(
            "$(path) is a native-dexel table; BasisSimulator reads its own copy, PCCT_NATIVE_TABLE_PATH",
        ))
    return MCResponseData(
        Int.(d["energies_keV"]),
        Int.(d["thresholds_keV"]),
        Float64.(d["R_total"]),
        Float64.(d["Cov_total"]),
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
function mc_cumulative_to_bins(R_cumulative::Matrix{Float64}, thresholds_keV::Vector{Int})
    n_E, n_t = size(R_cumulative)
    R_bins = zeros(Float64, n_E, n_t)

    for k in 1:(n_t - 1)
        # Bin k: counts between threshold[k] and threshold[k+1]
        R_bins[:, k] = max.(R_cumulative[:, k] .- R_cumulative[:, k + 1], 0.0)
    end
    # Last bin: counts ≥ highest threshold (everything above)
    R_bins[:, n_t] = max.(R_cumulative[:, n_t], 0.0)

    return R_bins
end

"""
    compute_mc_drm(detector::PhotonCountingDetector, kVp::Real;
                    npz_path::String=default_mc_drm_path(),
                    n_energy_points::Int=200) -> Matrix{Float64}

Compute a Detector Response Matrix from MC simulation data.

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
det = BasisSimulator._build_pcct_detector(PCCTScanner(energy_thresholds = [20.0, 35.0, 55.0, 70.0]))
D = compute_mc_drm(det, 120.0)  # uses bundled response file
```
"""
function compute_mc_drm(
    detector::PhotonCountingDetector,
    kVp::Real;
    npz_path::String = default_mc_drm_path(),
    n_energy_points::Int = 200,
)
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
            T_high = target_b < n_target_bins ? target_thresholds[target_b + 1] : Inf

            # Sum all MC bins whose energy range overlaps with target bin [T_low, T_high)
            for mc_b in eachindex(mc_thresholds)
                mc_T_low = mc_thresholds[mc_b]
                mc_T_high = mc_b < length(mc_thresholds) ? mc_thresholds[mc_b + 1] : Inf

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
                        frac = min(
                            (overlap_high - overlap_low) / (Float64(kVp) - mc_T_low + 1.0),
                            1.0,
                        )
                    else
                        frac = (overlap_high - overlap_low) / mc_width
                    end

                    D_native[e_idx, target_b] += R_mc_bins[e_idx, mc_b] * frac
                end
            end
        end
    end

    # Step 3: Interpolate to BasisSimulator's uniform energy grid
    E_grid = range(1.0, Float64(kVp), length = n_energy_points)
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
                idx_low = clamp(searchsortedlast(mc_E, E), 1, n_mc_E - 1)
                idx_high = idx_low + 1
                E_low = mc_E[idx_low]
                E_high = mc_E[idx_high]
                t = (E - E_low) / (E_high - E_low)
                D[i, b] = (1.0 - t) * D_native[idx_low, b] + t * D_native[idx_high, b]
            end
        end
    end

    # Counts are never negative. A row may sum above 1: a photon whose charge two dexels share
    # counts in both, so counts per incident photon exceed one at the low thresholds.
    D .= max.(D, 0.0)

    return D
end

"""
    compute_mc_count_moments(detector, energies, weights; R=nothing)

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
    R = nothing,
    response_path::String = default_mc_drm_path(),
)
    length(energies) == length(weights) ||
        throw(DimensionMismatch("energies and weights must have equal length"))
    isempty(energies) && throw(ArgumentError("energies must be nonempty"))
    all(isfinite, weights) && all(>=(0), weights) ||
        throw(ArgumentError("weights must be finite and nonnegative"))
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
    size(R_use, 2) == n_bins ||
        throw(DimensionMismatch("R has $(size(R_use, 2)) bins; detector has $n_bins"))

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
            lo = hi = 1
            α = 0.0
        elseif E >= last(mc_E)
            lo = hi = length(mc_E)
            α = 0.0
        else
            hi = searchsortedfirst(mc_E, E)
            lo = hi - 1
            α = (E - mc_E[lo]) / (mc_E[hi] - mc_E[lo])
        end
        r_cum = (1 - α) .* @view(mc.R_total[lo, :]) .+ α .* @view(mc.R_total[hi, :])
        c_cum = (1 - α) .* @view(mc.Cov_total[lo, :, :]) .+
            α .* @view(mc.Cov_total[hi, :, :])
        m_raw = S * r_cum
        C_raw = S * c_cum * transpose(S)

        r_idx = clamp(round(Int, (E - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        d = Float64.(@view R_use[r_idx, :])
        scale = [m_raw[b] > eps(Float64) ? d[b] / m_raw[b] : 0.0 for b in 1:n_bins]
        Dscale = Diagonal(scale)
        C = Dscale * C_raw * Dscale
        event_mean = d
        # A compound-Poisson incident process (R per incident photon, absorption included):
        # Cov_total = C + dd′.
        event_second = C .+ d * transpose(d)
        mean_eff .+= w .* event_mean
        second_eff .+= w .* event_second
    end

    second_eff .= (second_eff .+ transpose(second_eff)) ./ 2
    fano = [
        mean_eff[b] > eps(Float64) ? second_eff[b, b] / mean_eff[b] : 1.0
        for b in 1:n_bins
    ]
    correlation = Matrix{Float64}(I, n_bins, n_bins)
    for j in 1:n_bins, i in 1:n_bins
        denom = sqrt(max(second_eff[i, i] * second_eff[j, j], 0.0))
        correlation[i, j] = denom > eps(Float64) ?
            second_eff[i, j] / denom :
            (i == j ? 1.0 : 0.0)
    end
    correlation .= clamp.(correlation, -1.0, 1.0)
    return (
        mean = mean_eff,
        fano = fano,
        correlation = correlation,
        covariance = second_eff,
    )
end

export compute_mc_count_moments

# =============================================================================
# Exports
# =============================================================================

export MCResponseData, load_mc_response
export mc_cumulative_to_bins, compute_mc_drm
export default_mc_drm_path
