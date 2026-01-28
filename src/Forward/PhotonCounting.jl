"""
    Forward/PhotonCounting.jl

Photon-counting CT (PCCT) detector simulation for BasisSimulator.jl.

# Physics Background

Photon-counting detectors (PCDs) directly convert individual X-ray photons into
electrical signals using semiconductor materials (CdTe, CZT, or Si). Unlike
energy-integrating detectors that sum all absorbed energy, PCDs:

1. **Count individual photons** - each photon registered as a discrete event
2. **Measure photon energy** - pulse height proportional to deposited energy
3. **Apply energy thresholds** - classify photons into energy bins

## Key Physical Effects

### Energy Threshold Binning
PCDs use comparators to classify photons into bins based on energy thresholds.
For N thresholds {T₁, T₂, ..., Tₙ}, counts in bin i are:
    C_i = counts with T_i ≤ E < T_{i+1}

### Charge Sharing
When X-ray absorption occurs near pixel boundaries, the charge cloud may split
between adjacent pixels, causing:
- Single high-energy photon counted as multiple lower-energy photons
- Degraded energy resolution (spectral blurring)
- Spatial PSF broadening

### K-Fluorescence Escape
In high-Z detectors (CdTe/CZT), photoelectric absorption can eject K-shell
electrons. The resulting fluorescent X-rays (Cd K-α: 23.2 keV, Te K-α: 27.4 keV)
can travel ~100 μm, depositing energy in neighboring pixels.

### Pulse Pile-up
At high photon flux rates, multiple photons may arrive within the detector
dead time, causing:
- Count rate saturation
- Incorrect (summed) energy registration
- Spectral distortion

### Anti-Coincidence Logic
Simultaneous signals in adjacent pixels within a coincidence window are summed
and assigned to the primary pixel, correcting for charge sharing.

# GPU Compatibility

All functions are GPU-native via AcceleratedKernels.jl:
- ✅ Metal (Apple Silicon)
- ✅ CUDA (NVIDIA)
- ✅ ROCm (AMD)
- ✅ CPU fallback

# References

1. Willemink MJ, Persson M. "Photon-counting CT: Technical Principles and
   Clinical Prospects." Radiology. 2018;289(2):293-312. doi:10.1148/radiol.2018172656

2. Taguchi K, Iwanczyk JS. "Vision 20/20: Single photon counting x-ray detectors
   in medical imaging." Med Phys. 2013;40(10):100901. doi:10.1118/1.4820371

3. Flohr T, et al. "First performance evaluation of a dual-source CT (DSCT)
   system." Eur Radiol. 2006;16:256-268.

4. FDA 510(k) K201501 - Siemens NAEOTOM Alpha

5. DukeSim v1.2: https://cvit.duke.edu/resource/dukesim-v1-2/

See also: [`DetectorModel`](@ref), [`CrosstalkModel`](@ref)
"""

import AcceleratedKernels as AK
using Random

# =============================================================================
# Photon Counting Detector Types
# =============================================================================

"""
    DetectorMaterialPCCT

Semiconductor materials used in photon-counting detectors.
"""
@enum DetectorMaterialPCCT begin
    CDTE_MATERIAL    # Cadmium Telluride (Siemens NAEOTOM)
    CZT_MATERIAL     # Cadmium Zinc Telluride
    SI_MATERIAL      # Silicon (lower Z, limited stopping power)
end

"""
    PhotonCountingDetector{T<:AbstractFloat}

Photon-counting detector specification for PCCT simulation.

# Fields

## Detector Material Properties
- `material::DetectorMaterialPCCT`: Semiconductor material (CdTe, CZT, Si)
- `thickness_mm::T`: Sensor thickness in mm (typical: 1.5-3.0 mm)
- `pixel_size_mm::Tuple{T,T}`: Pixel size at isocenter (row, col) in mm

## Energy Binning
- `energy_thresholds_keV::Vector{T}`: Energy threshold values in keV
- `energy_resolution_keV::T`: Energy resolution FWHM at 60 keV (typical: 8-12 keV)

## Charge Sharing
- `charge_sharing_fwhm_mm::T`: Charge cloud FWHM in mm (typical: 0.05-0.15 mm)
- `enable_charge_sharing::Bool`: Whether to simulate charge sharing

## Pulse Pile-up
- `dead_time_ns::T`: Detector dead time in nanoseconds (typical: 20-100 ns)
- `enable_pile_up::Bool`: Whether to simulate pulse pile-up

## Anti-Coincidence
- `enable_anti_coincidence::Bool`: Whether to apply anti-coincidence logic
- `coincidence_window_ns::T`: Time window for coincidence detection (typical: 20-50 ns)

## Noise
- `electronic_noise_keV::T`: Electronic noise RMS in keV (typical: 1-2 keV)

## Control
- `seed::Union{Nothing, Int}`: Random seed for reproducibility

# Example

```julia
# NAEOTOM Alpha-like detector
detector = PhotonCountingDetector(
    material = CDTE_MATERIAL,
    thickness_mm = 1.6,
    pixel_size_mm = (0.302, 0.302),
    energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
    charge_sharing_fwhm_mm = 0.08,
    dead_time_ns = 25.0
)
```

See also: [`naeotom_detector_standard`](@ref), [`naeotom_detector_uhr`](@ref)
"""
struct PhotonCountingDetector{T<:AbstractFloat}
    # Material properties
    material::DetectorMaterialPCCT
    thickness_mm::T
    pixel_size_mm::Tuple{T,T}

    # Energy binning
    energy_thresholds_keV::Vector{T}
    energy_resolution_keV::T

    # Charge sharing
    charge_sharing_fwhm_mm::T
    enable_charge_sharing::Bool

    # Pulse pile-up
    dead_time_ns::T
    enable_pile_up::Bool

    # Anti-coincidence
    enable_anti_coincidence::Bool
    coincidence_window_ns::T

    # Electronic noise
    electronic_noise_keV::T

    # Control
    seed::Union{Nothing, Int}
end

"""
    PhotonCountingDetector(; kwargs...)

Construct a PhotonCountingDetector with keyword arguments.

# Keyword Arguments
- `material::DetectorMaterialPCCT=CDTE_MATERIAL`: Detector material
- `thickness_mm::Float64=1.6`: Sensor thickness
- `pixel_size_mm::Tuple{Float64,Float64}=(0.302, 0.302)`: Pixel size (row, col)
- `energy_thresholds_keV::Vector{Float64}=[20.0, 35.0, 55.0, 70.0]`: Energy thresholds
- `energy_resolution_keV::Float64=10.0`: Energy resolution FWHM at 60 keV
- `charge_sharing_fwhm_mm::Float64=0.08`: Charge cloud FWHM
- `enable_charge_sharing::Bool=true`: Enable charge sharing simulation
- `dead_time_ns::Float64=25.0`: Detector dead time
- `enable_pile_up::Bool=true`: Enable pile-up simulation
- `enable_anti_coincidence::Bool=true`: Enable anti-coincidence correction
- `coincidence_window_ns::Float64=30.0`: Coincidence time window
- `electronic_noise_keV::Float64=1.5`: Electronic noise RMS
- `seed::Union{Nothing,Int}=nothing`: Random seed
"""
function PhotonCountingDetector(;
    material::DetectorMaterialPCCT=CDTE_MATERIAL,
    thickness_mm::Float64=1.6,
    pixel_size_mm::Tuple{Float64,Float64}=(0.302, 0.302),
    energy_thresholds_keV::Vector{Float64}=[20.0, 35.0, 55.0, 70.0],
    energy_resolution_keV::Float64=10.0,
    charge_sharing_fwhm_mm::Float64=0.08,
    enable_charge_sharing::Bool=true,
    dead_time_ns::Float64=25.0,
    enable_pile_up::Bool=true,
    enable_anti_coincidence::Bool=true,
    coincidence_window_ns::Float64=30.0,
    electronic_noise_keV::Float64=1.5,
    seed::Union{Nothing, Int}=nothing
)
    return PhotonCountingDetector{Float64}(
        material, thickness_mm, pixel_size_mm,
        energy_thresholds_keV, energy_resolution_keV,
        charge_sharing_fwhm_mm, enable_charge_sharing,
        dead_time_ns, enable_pile_up,
        enable_anti_coincidence, coincidence_window_ns,
        electronic_noise_keV, seed
    )
end

# =============================================================================
# Pre-defined Detector Configurations
# =============================================================================

"""
    naeotom_detector_standard()

Create NAEOTOM Alpha-like detector in standard (binned) mode.

Siemens NAEOTOM Alpha specifications:
- CdTe detector, 1.6 mm thick
- 144 rows × 0.4 mm (57.6 mm z-coverage)
- 0.302 mm pixel size at isocenter (2×2 binned)
- 4 energy thresholds: 20, 35, 55, 70 keV

Reference: FDA 510(k) K201501
"""
function naeotom_detector_standard()
    return PhotonCountingDetector(
        material = CDTE_MATERIAL,
        thickness_mm = 1.6,
        pixel_size_mm = (0.302, 0.302),
        energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
        energy_resolution_keV = 10.0,
        charge_sharing_fwhm_mm = 0.08,
        enable_charge_sharing = true,
        dead_time_ns = 25.0,
        enable_pile_up = true,
        enable_anti_coincidence = true,
        coincidence_window_ns = 30.0,
        electronic_noise_keV = 1.5
    )
end

"""
    naeotom_detector_uhr()

Create NAEOTOM Alpha-like detector in UHR (ultra-high resolution) mode.

UHR mode specifications:
- Unbinned pixels: 0.151 mm at isocenter
- 120 rows × 0.2 mm (24 mm z-coverage)
- Higher spatial resolution, higher dose
- Same 4 energy thresholds

Reference: FDA 510(k) K201501
"""
function naeotom_detector_uhr()
    return PhotonCountingDetector(
        material = CDTE_MATERIAL,
        thickness_mm = 1.6,
        pixel_size_mm = (0.151, 0.151),
        energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
        energy_resolution_keV = 10.0,
        charge_sharing_fwhm_mm = 0.08,
        enable_charge_sharing = true,
        dead_time_ns = 25.0,
        enable_pile_up = true,
        enable_anti_coincidence = true,
        coincidence_window_ns = 30.0,
        electronic_noise_keV = 1.5
    )
end

"""
    pcct_detector_ideal()

Create ideal photon-counting detector with no degradation effects.

Useful for validating spectral imaging algorithms without detector artifacts.
"""
function pcct_detector_ideal()
    return PhotonCountingDetector(
        material = CDTE_MATERIAL,
        thickness_mm = 1.6,
        pixel_size_mm = (0.302, 0.302),
        energy_thresholds_keV = [20.0, 35.0, 55.0, 70.0],
        energy_resolution_keV = 0.0,  # Perfect energy resolution
        charge_sharing_fwhm_mm = 0.0, # No charge sharing
        enable_charge_sharing = false,
        dead_time_ns = 0.0,           # No pile-up
        enable_pile_up = false,
        enable_anti_coincidence = false,
        coincidence_window_ns = 0.0,
        electronic_noise_keV = 0.0    # No electronic noise
    )
end

# =============================================================================
# Energy-Resolved Sinogram Container
# =============================================================================

"""
    EnergyResolvedSinogram{T,A}

Container for energy-resolved sinogram data from photon-counting CT.

Each energy bin contains a full sinogram, allowing spectral analysis and
reconstruction at specific energies or with spectral weighting.

# Fields
- `bins::Vector{A}`: Vector of sinograms, one per energy bin
- `thresholds_keV::Vector{T}`: Energy thresholds defining bins (N+1 values for N bins)
- `n_cols::Int`: Number of detector columns
- `n_rows::Int`: Number of detector rows
- `n_angles::Int`: Number of projection angles

# Energy Bins
For thresholds [T₁, T₂, ..., Tₙ]:
- Bin 1: T₁ ≤ E < T₂
- Bin 2: T₂ ≤ E < T₃
- ...
- Bin N: Tₙ ≤ E < ∞ (or max tube kVp)

# Example

```julia
# After PCCT forward projection
pcct_sino = pcct_forward_project(volume, geom, detector, spectrum)

# Access individual energy bins
bin1 = pcct_sino.bins[1]  # Lowest energy bin
bin4 = pcct_sino.bins[4]  # Highest energy bin

# Total counts (sum of all bins)
total = sum(pcct_sino.bins)
```

See also: [`pcct_forward_project`](@ref), [`PhotonCountingDetector`](@ref)
"""
struct EnergyResolvedSinogram{T<:AbstractFloat, A<:AbstractArray{T,3}}
    bins::Vector{A}
    thresholds_keV::Vector{T}
    n_cols::Int
    n_rows::Int
    n_angles::Int
end

function EnergyResolvedSinogram(bins::Vector{A}, thresholds_keV::Vector{T}) where {T<:AbstractFloat, A<:AbstractArray{T,3}}
    @assert length(bins) == length(thresholds_keV) "Number of bins must match number of thresholds"
    @assert length(bins) > 0 "Must have at least one bin"
    n_cols, n_rows, n_angles = size(bins[1])
    return EnergyResolvedSinogram{T,A}(bins, thresholds_keV, n_cols, n_rows, n_angles)
end

Base.size(es::EnergyResolvedSinogram) = (es.n_cols, es.n_rows, es.n_angles)
Base.eltype(::EnergyResolvedSinogram{T}) where T = T
n_energy_bins(es::EnergyResolvedSinogram) = length(es.bins)

# =============================================================================
# Energy Threshold Model (GPU-native)
# =============================================================================

"""
    apply_energy_thresholds(intensity_spectrum, energies, weights, detector) -> Vector{Array}

Apply energy thresholds to spectral data to produce energy-binned counts.

This is the core PCCT physics: photons are counted in bins based on their
deposited energy relative to the threshold settings.

# Arguments
- `intensity_spectrum::AbstractArray{T,4}`: Spectral intensity [n_cols, n_rows, n_angles, n_energies]
- `energies::Vector{T}`: Energy values in keV
- `weights::Vector{T}`: Spectral weights (normalized)
- `detector::PhotonCountingDetector`: Detector specification

# Returns
Vector of 3D arrays, one per energy bin, containing photon counts.

# Physics

For each energy threshold Tᵢ, photons with E ≥ Tᵢ contribute to that bin's count.
The energy-resolved counts are:

    C_bin(i) = Σ{E ≥ Tᵢ} N(E) × w(E)

where N(E) is the photon count at energy E and w(E) is the spectral weight.

Energy resolution blurring (if enabled) spreads counts between adjacent bins.
"""
function apply_energy_thresholds(
    intensity_spectrum::AbstractArray{T,4},
    energies::AbstractVector,
    weights::AbstractVector,
    detector::PhotonCountingDetector
) where T

    n_cols, n_rows, n_angles, n_energies = size(intensity_spectrum)
    n_bins = length(detector.energy_thresholds_keV)
    thresholds = detector.energy_thresholds_keV

    # Allocate output bins on same device as input
    bins = [similar(intensity_spectrum, n_cols, n_rows, n_angles) for _ in 1:n_bins]
    for bin in bins
        fill!(bin, zero(T))
    end

    # Energy resolution: sigma from FWHM
    σ_E = detector.energy_resolution_keV / (2 * sqrt(2 * log(T(2))))

    # GPU-native threshold application
    for bin_idx in 1:n_bins
        threshold = T(thresholds[bin_idx])
        upper_threshold = bin_idx < n_bins ? T(thresholds[bin_idx + 1]) : T(Inf)
        output_bin = bins[bin_idx]

        AK.foreachindex(output_bin) do idx
            ci = CartesianIndices(output_bin)[idx]
            col, row, angle = Tuple(ci)

            acc = zero(T)
            for e_idx in 1:n_energies
                E = energies[e_idx]
                w = weights[e_idx]
                I = intensity_spectrum[col, row, angle, e_idx]

                # Apply energy resolution blurring if enabled
                if σ_E > zero(T)
                    # Gaussian probability that photon of energy E is registered in this bin
                    # P(T_low ≤ E_registered < T_high)
                    # Using cumulative distribution function approximation
                    z_low = (threshold - E) / σ_E
                    z_high = (upper_threshold - E) / σ_E

                    # Approximate erf using sigmoid for GPU compatibility
                    # erf(x) ≈ 2 * sigmoid(sqrt(π) * x) - 1
                    sqrt_pi = T(1.7724538509055159)
                    sigmoid_low = one(T) / (one(T) + exp(-sqrt_pi * z_low))
                    sigmoid_high = one(T) / (one(T) + exp(-sqrt_pi * z_high))
                    prob = sigmoid_high - sigmoid_low
                else
                    # Perfect energy resolution: binary assignment
                    prob = (E >= threshold && E < upper_threshold) ? one(T) : zero(T)
                end

                acc += I * w * prob
            end

            output_bin[idx] = acc
        end
    end

    return bins
end

# =============================================================================
# Charge Sharing Model (GPU-native)
# =============================================================================

"""
    apply_charge_sharing!(bins, detector) -> bins

Apply charge sharing effects to energy-binned sinograms (in-place, GPU-native).

Charge sharing occurs when X-ray absorption near pixel boundaries causes the
resulting charge cloud to split between adjacent pixels. This causes:
- High-energy photons to be registered as multiple lower-energy counts
- Spectral degradation (low-energy tail)
- Spatial PSF broadening

# Algorithm

1. For each pixel, compute probability that charge cloud extends to neighbors
2. Redistribute counts: fraction of high-energy counts shifted to low-energy bins
3. Apply spatial convolution for PSF effects

The charge sharing kernel depends on pixel size and charge cloud FWHM.
"""
function apply_charge_sharing!(
    bins::Vector{A},
    detector::PhotonCountingDetector
) where {T, A<:AbstractArray{T,3}}

    if !detector.enable_charge_sharing || detector.charge_sharing_fwhm_mm ≤ 0.0
        return bins
    end

    n_bins = length(bins)
    n_cols, n_rows, n_angles = size(bins[1])

    # Cast all detector parameters to T (Float32 for GPU compatibility)
    σ_cloud = T(detector.charge_sharing_fwhm_mm) / (T(2) * sqrt(T(2) * log(T(2))))

    # Pixel size at isocenter (cast to T)
    pixel_row = T(detector.pixel_size_mm[1])
    pixel_col = T(detector.pixel_size_mm[2])

    # Compute charge sharing probability
    boundary_dist_row = pixel_row / T(2)
    boundary_dist_col = pixel_col / T(2)

    z_row = boundary_dist_row / σ_cloud
    z_col = boundary_dist_col / σ_cloud

    # Sigmoid approximation for Gaussian tail
    p_share_row = T(2) / (one(T) + exp(T(1.5) * z_row))
    p_share_col = T(2) / (one(T) + exp(T(1.5) * z_col))
    p_share = min(p_share_row + p_share_col, T(0.5))

    # Primary signal retention
    p_primary = one(T) - p_share

    # Neighbor weight (divide among 8 neighbors)
    nw = p_share / T(8)

    # Energy loss fraction for split events
    energy_loss_fraction = T(0.5)

    # Process each bin: redistribute counts to lower bins due to charge sharing
    for bin_idx in n_bins:-1:1
        # Use let-blocks for GPU kernel capture safety
        let cb = bins[bin_idx], pp = p_primary, nweight = nw,
            nc = Int32(n_cols), nr = Int32(n_rows)

            output = similar(cb)

            AK.foreachindex(cb) do idx
                ci = CartesianIndices(cb)[idx]
                col, row, angle = Tuple(ci)

                val = cb[idx] * pp

                for di in Int32(-1):Int32(1)
                    for dj in Int32(-1):Int32(1)
                        if di == Int32(0) && dj == Int32(0)
                            continue
                        end
                        src_col = clamp(col + di, Int32(1), nc)
                        src_row = clamp(row + dj, Int32(1), nr)
                        val += cb[src_col, src_row, angle] * nweight
                    end
                end

                output[idx] = val
            end

            copyto!(cb, output)
        end

        # Energy redistribution: move fraction of counts to lower bin
        if bin_idx > 1
            let cb = bins[bin_idx], tb = bins[bin_idx - 1],
                lf = p_share * energy_loss_fraction

                AK.foreachindex(cb) do idx
                    transfer = cb[idx] * lf
                    tb[idx] += transfer
                    cb[idx] -= transfer
                end
            end
        end
    end

    return bins
end

# =============================================================================
# Pulse Pile-up Model (GPU-native)
# =============================================================================

"""
    apply_pulse_pileup!(bins, detector, flux_rate) -> bins

Apply pulse pile-up effects to energy-binned counts (in-place, GPU-native).

Pulse pile-up occurs when photons arrive faster than the detector can process
them (within the dead time). Effects include:
- Count rate saturation (recorded counts < true counts)
- Energy distortion (piled-up pulses registered at sum energy)

# Dead Time Models

**Nonparalyzable model** (used here):
    N_recorded = N_true / (1 + N_true × τ)

**Paralyzable model**:
    N_recorded = N_true × exp(-N_true × τ)

# Arguments
- `bins::Vector{Array}`: Energy-binned counts
- `detector::PhotonCountingDetector`: Detector specification
- `flux_rate::Float64`: Photon flux rate in photons/s/mm² (at detector)

# Physics

Dead time τ (typically 20-100 ns for CdTe) limits maximum count rate to ~1/τ.
At typical CT flux rates (~10⁹ photons/s/mm²), pile-up can be significant.

Reference: Taguchi & Iwanczyk, Med Phys 2013
"""
function apply_pulse_pileup!(
    bins::Vector{A},
    detector::PhotonCountingDetector,
    flux_rate::Real
) where {T, A<:AbstractArray{T,3}}

    if !detector.enable_pile_up || detector.dead_time_ns ≤ 0.0
        return bins
    end

    # Cast all to T (Float32 for GPU compatibility)
    τ = T(detector.dead_time_ns) * T(1e-9)
    pixel_area = T(detector.pixel_size_mm[1]) * T(detector.pixel_size_mm[2])
    count_rate = T(flux_rate) * pixel_area

    # Pile-up correction factor (nonparalyzable model)
    pile_up_factor = one(T) / (one(T) + count_rate * τ)

    # Apply count rate reduction to all bins
    for bin in bins
        let pf = pile_up_factor, b = bin
            AK.foreachindex(b) do idx
                b[idx] *= pf
            end
        end
    end

    # Energy pile-up: some counts from lower bins shift to higher bins
    n_bins = length(bins)
    if n_bins >= 2
        p_pileup = one(T) - pile_up_factor

        for bin_idx in 1:(n_bins-1)
            let cb = bins[bin_idx], nb = bins[bin_idx + 1],
                sf = p_pileup * T(0.1)

                AK.foreachindex(cb) do idx
                    transfer = cb[idx] * sf
                    nb[idx] += transfer
                    cb[idx] -= transfer
                end
            end
        end
    end

    return bins
end

# =============================================================================
# Anti-Coincidence Logic (GPU-native)
# =============================================================================

"""
    apply_anti_coincidence!(bins, detector) -> bins

Apply anti-coincidence correction to energy-binned counts (in-place, GPU-native).

Anti-coincidence (charge sharing correction) detects simultaneous signals in
adjacent pixels and sums them to the primary pixel, partially correcting for
charge sharing effects.

# Algorithm

1. Detect coincident events (signals in neighboring pixels within time window)
2. Sum energies: E_total = E_pixel1 + E_pixel2 + ...
3. Assign total to primary pixel (highest signal)

# Performance Impact

Anti-coincidence improves:
- Energy resolution: 20-40% reduction in spectral tails
- Spatial resolution: reduced cross-pixel PSF spread

Trade-offs:
- Reduced count rate capability
- May cause count loss at very high flux

Reference: PMC5975362
"""
function apply_anti_coincidence!(
    bins::Vector{A},
    detector::PhotonCountingDetector
) where {T, A<:AbstractArray{T,3}}

    if !detector.enable_anti_coincidence
        return bins
    end

    n_bins = length(bins)
    n_cols, n_rows, n_angles = size(bins[1])

    # Anti-coincidence recovers some charge-shared events
    # by summing coincident signals in adjacent pixels

    # Compute total counts per pixel (across all energy bins)
    total_counts = similar(bins[1])
    fill!(total_counts, zero(T))
    for bin in bins
        let tc = total_counts, b = bin
            AK.foreachindex(tc) do idx
                tc[idx] += b[idx]
            end
        end
    end

    # Correction factor: fraction of charge-shared events recovered
    rf = T(0.3)  # 30% recovery
    rf8 = rf / T(8)  # Pre-divide by 8 neighbors
    zero_T = T(0)  # Pre-compute to avoid Type capture in kernel

    # For each pixel, look for coincident neighbors and redistribute
    for bin_idx in 1:n_bins
        let cb = bins[bin_idx], tc = total_counts, recovery_per_neighbor = rf8,
            nc = Int32(n_cols), nr = Int32(n_rows), z = zero_T

            output = similar(cb)

            AK.foreachindex(cb) do idx
                ci = CartesianIndices(cb)[idx]
                col, row, angle = Tuple(ci)

                val = cb[idx]
                my_total = tc[idx]

                if my_total > z
                    for di in Int32(-1):Int32(1)
                        for dj in Int32(-1):Int32(1)
                            if di == Int32(0) && dj == Int32(0)
                                continue
                            end

                            src_col = clamp(col + di, Int32(1), nc)
                            src_row = clamp(row + dj, Int32(1), nr)

                            neighbor_total = tc[src_col, src_row, angle]

                            if my_total > neighbor_total && neighbor_total > z
                                neighbor_val = cb[src_col, src_row, angle]
                                val += neighbor_val * recovery_per_neighbor
                            end
                        end
                    end
                end

                output[idx] = val
            end

            copyto!(cb, output)
        end
    end

    return bins
end

# =============================================================================
# Electronic Noise Model for PCCT (GPU-native)
# =============================================================================

"""
    apply_pcct_electronic_noise!(bins, detector) -> bins

Apply electronic noise to energy-binned counts (in-place, GPU-native).

Unlike energy-integrating detectors, PCCT electronic noise affects:
1. Energy discrimination: noise causes threshold jitter
2. Count registration: sub-threshold signals may be missed or false-triggered

For PCCT, electronic noise is typically 1-2 keV RMS, much lower than the
~20 keV lowest threshold, so false triggering is rare.

# Physics

Electronic noise in PCCT manifests as:
- Gaussian jitter on energy measurement (handled in threshold application)
- Additive noise on registered counts (Gaussian, small magnitude)

The count-level noise is much lower than in energy-integrating detectors
because individual photons are counted rather than integrated.
"""
function apply_pcct_electronic_noise!(
    bins::Vector{A},
    detector::PhotonCountingDetector
) where {T, A<:AbstractArray{T,3}}

    if detector.electronic_noise_keV ≤ zero(T)
        return bins
    end

    rng = isnothing(detector.seed) ? Random.default_rng() : MersenneTwister(detector.seed)

    # Electronic noise effect on counts is small for PCCT
    # Model as additive Gaussian noise scaled by count level
    noise_scale = detector.electronic_noise_keV / T(60)  # Relative to 60 keV

    for (bin_idx, bin) in enumerate(bins)
        n_elements = length(bin)

        # Pre-generate noise on CPU
        rand_cpu = randn(rng, T, n_elements)
        rand_gpu = similar(bin, n_elements)
        copyto!(rand_gpu, rand_cpu)

        AK.foreachindex(bin) do idx
            # Noise proportional to sqrt(counts) - Poisson-like
            noise_sigma = sqrt(max(bin[idx], one(T))) * noise_scale
            bin[idx] += noise_sigma * rand_gpu[idx]
            bin[idx] = max(bin[idx], zero(T))  # Ensure non-negative
        end
    end

    return bins
end

# =============================================================================
# Main PCCT Forward Projection API
# =============================================================================

"""
    pcct_forward_project(mask, geom, detector; energies, weights, materials, kwargs...) -> EnergyResolvedSinogram

Perform polychromatic photon-counting CT forward projection with full detector physics.

This is the main PCCT forward projection API. It:
1. Computes energy-dependent μ-volumes from mask + materials (per energy)
2. Forward projects at each energy using Siddon ray-tracing
3. Applies quantum efficiency η(E) weighting (material-dependent)
4. Applies spectral response matrix R(E,b) for realistic energy binning
5. Applies detector physics chain (charge sharing, pileup, anti-coincidence)
6. Returns energy-resolved sinograms in line-integral domain

# Arguments
- `mask::AbstractArray{UInt8,3}`: Material mask (region indices)
- `geom::CTGeometry`: CT geometry
- `detector::PhotonCountingDetector`: PCCT detector specification

# Keyword Arguments
- `energies::AbstractVector`: Spectral energies in keV
- `weights::AbstractVector`: Spectral weights (normalized photon fluence)
- `materials::Vector`: Material vector (from get_region_materials())
- `flux_rate::Real=1e9`: Photon flux rate in photons/s/mm² (for pile-up)
- `I0::Real=1e6`: Reference photon count per detector element
- `apply_spectral_response::Bool=true`: Whether to use spectral response matrix R(E,b)

# Returns
`EnergyResolvedSinogram` with one sinogram per energy bin.

# Physics

The per-bin photon count (before detector effects) is:
    N_b = Σ_E R(E,b) × S(E) × η(E) × exp(-∫μ(x,E)dl)

where:
- R(E,b) = spectral response matrix entry (probability E registers in bin b)
- S(E) = normalized tube spectrum weight
- η(E) = quantum detection efficiency of detector crystal
- ∫μ(x,E)dl = line integral of energy-dependent attenuation

# Example

```julia
detector = naeotom_detector_standard()
energies, weights = load_spectrum(120)
materials = get_region_materials()

pcct_sino = pcct_forward_project(
    phantom.mask, geom, detector;
    energies=energies, weights=weights,
    materials=materials
)

# Access energy bins
low_E_sino = pcct_sino.bins[1]   # 20-35 keV
high_E_sino = pcct_sino.bins[4]  # 70+ keV
```

See also: [`PhotonCountingDetector`](@ref), [`EnergyResolvedSinogram`](@ref)
"""
function pcct_forward_project(
    mask::AbstractArray{UInt8,3},
    geom,
    detector::PhotonCountingDetector;
    energies::AbstractVector,
    weights::AbstractVector,
    materials::Vector = get_region_materials(),
    flux_rate::Real = 1e9,
    I0::Real = 1e6,
    apply_spectral_response::Bool = true
)
    T = Float32  # Use Float32 for GPU efficiency

    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles
    n_energies = length(energies)
    n_bins = length(detector.energy_thresholds_keV)
    thresholds = detector.energy_thresholds_keV
    kVp = maximum(energies)

    # Pre-compute quantum efficiency for all energies (CPU, scalar values)
    η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)

    # Pre-compute spectral response matrix if requested (CPU, precomputed once)
    R = if apply_spectral_response
        compute_spectral_response_matrix(
            detector.material, detector.thickness_mm, thresholds, kVp;
            energy_resolution_keV=detector.energy_resolution_keV,
            pixel_size_mm=detector.pixel_size_mm,
            include_fluorescence=true,
            include_tailing=true,
            n_energy_points=n_energies
        )
    else
        nothing
    end

    # Energy grid for R matrix (maps energy index to row in R)
    R_energies = if apply_spectral_response
        collect(range(1.0, Float64(kVp), length=n_energies))
    else
        nothing
    end

    # Allocate per-bin photon count sinograms (GPU-compatible)
    sino_shape = (n_cols, n_rows, n_angles)
    bins = [similar(mask, T, sino_shape) for _ in 1:n_bins]
    for bin in bins
        fill!(bin, zero(T))
    end

    # Temporary μ-volume (reused across energies, same device as mask)
    μ_volume = similar(mask, T, size(mask))

    # Temporary sinogram buffer (reused across energies)
    sino_buf = similar(mask, T, sino_shape)

    # Per-energy ray-tracing with spectral weighting
    for (e_idx, E) in enumerate(energies)
        E_float = Float64(E)

        # Skip energies below lowest threshold (not detected)
        if E_float < thresholds[1]
            continue
        end

        # Skip negligible spectrum weights
        w = Float64(weights[e_idx])
        if w < 1e-12
            continue
        end

        # Create energy-dependent μ-volume from mask + materials
        create_μ_volume!(μ_volume, mask, materials, E_float)

        # Forward project at this energy (reuses existing Siddon infrastructure)
        fill!(sino_buf, zero(T))
        siddon_forward_project!(sino_buf, μ_volume, geom)

        # Weight by spectrum and quantum efficiency
        # Photon count: N = I₀ × S(E) × η(E) × exp(-∫μ dl)
        η_E = T(η[e_idx])
        w_T = T(w)
        I0_T = T(I0)

        if apply_spectral_response && R !== nothing
            # Use spectral response matrix: distribute photons across bins
            # Find the R matrix row for this energy
            # R is computed on a uniform grid from 1 to kVp
            r_idx = clamp(round(Int, (E_float - 1.0) / (Float64(kVp) - 1.0) * (n_energies - 1)) + 1, 1, n_energies)

            for b in 1:n_bins
                R_val = T(R[r_idx, b])
                if R_val < T(1e-10)
                    continue
                end
                # Use let-block to avoid Core.Box capture in GPU kernel
                let wt = I0_T * w_T * η_E * R_val, ba = bins[b]
                    AK.foreachindex(sino_buf) do idx
                        ba[idx] += wt * exp(-sino_buf[idx])
                    end
                end
            end
        else
            # Ideal binning: photon goes to exactly one bin based on energy
            bin_idx = _find_energy_bin(E_float, thresholds, Float64(kVp))
            if bin_idx > 0
                # Use let-block to avoid Core.Box capture in GPU kernel
                let wt = I0_T * w_T * η_E, ba = bins[bin_idx]
                    AK.foreachindex(sino_buf) do idx
                        ba[idx] += wt * exp(-sino_buf[idx])
                    end
                end
            end
        end
    end

    # Apply detector physics chain (charge sharing, pileup, anti-coincidence)
    apply_charge_sharing!(bins, detector)
    apply_pulse_pileup!(bins, detector, Float64(flux_rate))
    apply_anti_coincidence!(bins, detector)

    # Convert from photon counts to line-integral domain: sino = -log(N / I₀_bin)
    # Pass R to _compute_bin_I0 so it uses the same spectral response as forward projection
    eps_val = T(1e-10)  # Pre-compute to avoid capturing Type{T} in kernel
    for b in 1:n_bins
        I0_bin = _compute_bin_I0(detector, energies, weights, η, thresholds, b, Float64(kVp), Float64(I0); R=R)
        let I0_bin_T = T(I0_bin), ba = bins[b], eps = eps_val
            AK.foreachindex(ba) do idx
                ba[idx] = -log(max(ba[idx], eps) / I0_bin_T)
            end
        end
    end

    return EnergyResolvedSinogram(bins, T.(detector.energy_thresholds_keV))
end

"""
    _find_energy_bin(energy_keV, thresholds, kVp) -> Int

Find which energy bin a photon of given energy belongs to (ideal binning).
Returns 0 if below all thresholds or above kVp.
"""
function _find_energy_bin(energy_keV::Float64, thresholds::Vector{<:Real}, kVp::Float64)
    n_bins = length(thresholds)
    for b in n_bins:-1:1
        if energy_keV >= thresholds[b]
            return b
        end
    end
    return 0
end

"""
    _compute_bin_I0(detector, energies, weights, η, thresholds, bin_idx, kVp, I0; R=nothing) -> Float64

Compute the reference photon count I₀ for a specific energy bin.
This is the unattenuated count expected in this bin (for -log normalization).

When `R` (spectral response matrix) is provided, uses consistent calculation with forward projection.
When `R` is nothing, falls back to ideal binning (photon goes to exactly one bin based on energy).

The spectral response matrix R accounts for:
- Detector energy resolution (Gaussian blurring)
- Charge sharing between pixels
- K-fluorescence escape/reabsorption
- Electronic noise

Using R ensures that -log(N/I0_bin) produces the correct line integral values.
"""
function _compute_bin_I0(detector, energies, weights, η, thresholds, bin_idx, kVp, I0; R=nothing)
    if R !== nothing
        # Use spectral response matrix (consistent with forward projection)
        # This matches how photons are distributed in pcct_forward_project()
        n_energies = length(energies)
        I0_bin = 0.0
        for e_idx in 1:n_energies
            E_float = Float64(energies[e_idx])
            w = Float64(weights[e_idx])
            if w < 1e-12
                continue
            end
            # Same R index calculation as forward projection
            r_idx = clamp(round(Int, (E_float - 1.0) / (kVp - 1.0) * (n_energies - 1)) + 1, 1, n_energies)
            R_val = R[r_idx, bin_idx]
            I0_bin += I0 * w * η[e_idx] * R_val
        end
        return max(I0_bin, 1.0)  # Avoid division by zero
    else
        # Ideal binning fallback: photon goes to exactly one bin based on energy
        n_bins = length(thresholds)
        T_low = Float64(thresholds[bin_idx])
        T_high = bin_idx < n_bins ? Float64(thresholds[bin_idx + 1]) : kVp

        I0_bin = 0.0
        for (i, E) in enumerate(energies)
            E_f = Float64(E)
            # Last bin includes upper bound (E <= kVp)
            in_bin = if bin_idx == n_bins
                E_f >= T_low && E_f <= T_high
            else
                E_f >= T_low && E_f < T_high
            end
            if in_bin
                I0_bin += I0 * Float64(weights[i]) * η[i]
            end
        end
        return max(I0_bin, 1.0)  # Avoid division by zero
    end
end

# Legacy method: raw volume input (deprecated, kept for backward compatibility)
"""
    pcct_forward_project(volume::AbstractArray{T,3}, geom, detector, energies, weights; kwargs...)

DEPRECATED: Use the mask+materials method instead for polychromatic PCCT.

This legacy method projects the same volume at every energy (monochromatic behavior).
For correct polychromatic PCCT, use `pcct_forward_project(mask, geom, detector; ...)`.
"""
function pcct_forward_project(
    volume::AbstractArray{T,3},
    geom,
    detector::PhotonCountingDetector,
    energies::AbstractVector,
    weights::AbstractVector;
    flux_rate::Real = T(1e9),
    I0::Real = T(1e6)
) where T

    @warn "pcct_forward_project(volume, ...) is deprecated. Use pcct_forward_project(mask, geom, detector; energies=..., weights=..., materials=...) for polychromatic PCCT." maxlog=1

    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles
    n_energies = length(energies)
    n_bins = length(detector.energy_thresholds_keV)

    # Allocate spectral intensity array
    intensity_spectrum = similar(volume, n_cols, n_rows, n_angles, n_energies)

    # Project at each energy (SAME volume — monochromatic behavior!)
    for (e_idx, E) in enumerate(energies)
        sino_E = siddon_forward_project(volume, geom)
        I0_E = I0 * weights[e_idx]

        AK.foreachindex(sino_E) do idx
            intensity_spectrum[CartesianIndex(Tuple(CartesianIndices(sino_E)[idx])..., e_idx)] =
                I0_E * exp(-sino_E[idx])
        end
    end

    # Apply energy thresholds to get binned counts
    bins = apply_energy_thresholds(intensity_spectrum, energies, weights, detector)

    # Apply detector physics effects
    apply_charge_sharing!(bins, detector)
    apply_pulse_pileup!(bins, detector, flux_rate)
    apply_anti_coincidence!(bins, detector)
    apply_pcct_electronic_noise!(bins, detector)

    # Convert to line-integral domain
    for (bin_idx, bin) in enumerate(bins)
        threshold = detector.energy_thresholds_keV[bin_idx]
        I0_bin = I0 * sum(w for (E, w) in zip(energies, weights) if E >= threshold)
        AK.foreachindex(bin) do idx
            bin[idx] = -log(max(bin[idx], one(T)) / I0_bin)
        end
    end

    return EnergyResolvedSinogram(bins, T.(detector.energy_thresholds_keV))
end

# =============================================================================
# PCCT Noise Model (PCCT-NOISE-DECOMP)
# =============================================================================

"""
    apply_pcct_noise!(sino::EnergyResolvedSinogram, detector, protocol;
                       seed=nothing, I0=1e6, energies=nothing, weights=nothing) -> EnergyResolvedSinogram

Apply per-bin Poisson noise to PCCT energy-resolved sinograms (in-place).

Key PCCT noise characteristics:
- Each bin has INDEPENDENT Poisson statistics
- NO electronic noise (eliminated by energy thresholds — fundamental PCCT advantage)
- Lower counts per bin → higher relative noise per bin vs conventional
- Total counts (sum of all bins) ≈ conventional EID counts × η_avg

# Arguments
- `sino::EnergyResolvedSinogram`: Energy-resolved sinograms (in line-integral domain)
- `detector::PhotonCountingDetector`: Detector specification
- `protocol`: CTProtocol (for exposure/flux information)

# Keyword Arguments
- `seed::Union{Nothing,Int}`: Random seed for reproducibility
- `I0::Real`: Reference photon count per detector element (total, before binning)
- `energies::Union{Nothing,AbstractVector}`: Spectrum energies (keV) for proper per-bin I₀
- `weights::Union{Nothing,AbstractVector}`: Spectrum weights for proper per-bin I₀

# Physics

For each bin b and pixel:
1. Compute I₀_bin from spectrum: I₀_bin = I₀ × Σ_{E∈bin} S(E) × η(E)
2. Convert from line-integral to photon counts: N = I₀_bin × exp(-sino_value)
3. Sample: N_measured ~ Poisson(N)
4. Convert back: sino_noisy = -log(N_measured / I₀_bin)

When `energies` and `weights` are provided, per-bin I₀ is computed from the
spectrum weighted by quantum efficiency — this gives physically correct noise
levels per bin (lower bins get fewer photons from the spectrum).

# Returns
Modified EnergyResolvedSinogram with noise applied.
"""
function apply_pcct_noise!(
    sino::EnergyResolvedSinogram{T,A},
    detector::PhotonCountingDetector,
    protocol;
    seed::Union{Nothing,Int} = nothing,
    I0::Real = 1e6,
    energies::Union{Nothing,AbstractVector} = nothing,
    weights::Union{Nothing,AbstractVector} = nothing
) where {T, A}

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    n_bins = length(sino.bins)
    thresholds = sino.thresholds_keV

    # Compute per-bin I₀ values
    I0_per_bin = _compute_pcct_noise_I0(detector, n_bins, thresholds, I0, energies, weights)

    for (b, bin) in enumerate(sino.bins)
        I0_bin = T(I0_per_bin[b])

        # Generate Poisson noise on CPU, transfer to device
        bin_cpu = Array(bin)

        for idx in eachindex(bin_cpu)
            # Expected counts: N = I₀_bin × exp(-projection_value)
            N_expected = I0_bin * exp(-bin_cpu[idx])
            N_expected = max(N_expected, T(0.1))  # Floor to avoid zero

            # Poisson sampling (using Gaussian approximation for large N)
            if N_expected > T(20)
                # Gaussian approximation: N ~ Normal(μ=N, σ²=N)
                N_measured = N_expected + sqrt(N_expected) * T(randn(rng))
                N_measured = max(N_measured, T(1))
            else
                # Exact Poisson for small counts
                N_measured = T(_poisson_sample(rng, Float64(N_expected)))
                N_measured = max(N_measured, T(1))
            end

            # Convert back to line-integral domain
            bin_cpu[idx] = -log(N_measured / I0_bin)
        end

        # Transfer back to device
        copyto!(bin, bin_cpu)
    end

    return sino
end

"""
    _compute_pcct_noise_I0(detector, n_bins, thresholds, I0, energies, weights) -> Vector{Float64}

Compute per-bin reference photon counts for noise model.

When spectrum (energies, weights) is provided, computes the proper per-bin I₀
by integrating spectrum × quantum efficiency over each bin's energy range.
Otherwise falls back to uniform distribution (I0/n_bins).
"""
function _compute_pcct_noise_I0(detector, n_bins, thresholds, I0, energies, weights)
    if isnothing(energies) || isnothing(weights)
        # Fallback: uniform distribution across bins
        return fill(Float64(I0) / n_bins, n_bins)
    end

    # Compute quantum efficiency for all energies
    η = quantum_efficiency_vector(detector.material, detector.thickness_mm, energies)
    kVp = maximum(energies)

    I0_per_bin = zeros(Float64, n_bins)
    for b in 1:n_bins
        I0_per_bin[b] = _compute_bin_I0(detector, energies, weights, η, thresholds, b, Float64(kVp), Float64(I0))
    end
    return I0_per_bin
end

"""
    _poisson_sample(rng, λ) -> Int

Sample from Poisson distribution with parameter λ.
Uses Knuth's algorithm for small λ, normal approximation for large λ.
"""
function _poisson_sample(rng, λ::Float64)
    if λ > 30.0
        # Gaussian approximation
        return max(round(Int, λ + sqrt(λ) * randn(rng)), 0)
    elseif λ < 1e-10
        return 0
    else
        # Knuth's algorithm
        L = exp(-λ)
        k = 0
        p = 1.0
        while true
            k += 1
            p *= rand(rng)
            if p < L
                return k - 1
            end
        end
    end
end

# Note: synthesize_vmi and _get_basis_material_attenuation are defined in PCCTSpectral.jl
# (requires PCCTMaterialMap which is defined there)

# =============================================================================
# Utility Functions
# =============================================================================

"""
    get_pcct_detector_info(detector::PhotonCountingDetector) -> NamedTuple

Get diagnostic information about PCCT detector configuration.
"""
function get_pcct_detector_info(detector::PhotonCountingDetector{T}) where T
    n_bins = length(detector.energy_thresholds_keV)

    return (
        material = detector.material,
        thickness_mm = detector.thickness_mm,
        pixel_size_mm = detector.pixel_size_mm,
        n_energy_bins = n_bins,
        thresholds_keV = detector.energy_thresholds_keV,
        energy_resolution_keV = detector.energy_resolution_keV,
        charge_sharing_enabled = detector.enable_charge_sharing,
        charge_sharing_fwhm_mm = detector.charge_sharing_fwhm_mm,
        pile_up_enabled = detector.enable_pile_up,
        dead_time_ns = detector.dead_time_ns,
        anti_coincidence_enabled = detector.enable_anti_coincidence,
        electronic_noise_keV = detector.electronic_noise_keV
    )
end

"""
    print_pcct_detector_info(detector::PhotonCountingDetector)

Print formatted PCCT detector specification.
"""
function print_pcct_detector_info(detector::PhotonCountingDetector)
    info = get_pcct_detector_info(detector)

    println("=" ^ 60)
    println("PHOTON-COUNTING DETECTOR SPECIFICATION")
    println("=" ^ 60)
    println("Material:           $(info.material)")
    println("Thickness:          $(info.thickness_mm) mm")
    println("Pixel Size:         $(info.pixel_size_mm[1]) × $(info.pixel_size_mm[2]) mm")
    println()
    println("ENERGY BINNING")
    println("-" ^ 40)
    println("Number of bins:     $(info.n_energy_bins)")
    println("Thresholds (keV):   $(info.thresholds_keV)")
    println("Energy resolution:  $(info.energy_resolution_keV) keV FWHM")
    println()
    println("DETECTOR EFFECTS")
    println("-" ^ 40)
    println("Charge sharing:     $(info.charge_sharing_enabled ? "ON" : "OFF") (FWHM=$(info.charge_sharing_fwhm_mm) mm)")
    println("Pulse pile-up:      $(info.pile_up_enabled ? "ON" : "OFF") (τ=$(info.dead_time_ns) ns)")
    println("Anti-coincidence:   $(info.anti_coincidence_enabled ? "ON" : "OFF")")
    println("Electronic noise:   $(info.electronic_noise_keV) keV RMS")
    println("=" ^ 60)
end

# =============================================================================
# Material-Dependent Detector Physics (PCCT-MATERIAL-MODEL)
# =============================================================================

# Material-agnostic architecture: ALL physics functions dispatch on
# DetectorMaterialPCCT enum. CZT and Si work by adding entries to
# get_detector_material_properties() — no other code changes needed.

# Abramowitz & Stegun erf approximation (accuracy ~1.5×10⁻⁷)
# Used for spectral response matrix (CPU precomputation, not in GPU hot path)
function _erf_approx(x::Float64)
    # Handle sign
    sign_x = x < 0.0 ? -1.0 : 1.0
    x = abs(x)

    # Constants (Abramowitz & Stegun 7.1.26)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    return sign_x * y
end

"""
    get_detector_material_properties(material::DetectorMaterialPCCT) -> NamedTuple

Master lookup for ALL material-dependent PCCT detector parameters.

This is the SINGLE source of truth for material constants. All physics functions
dispatch through this function to ensure material-agnostic operation.

# Returns NamedTuple with fields:
- `elements`: Element symbols
- `atomic_numbers`: Z values for each element
- `mass_fractions`: Mass fraction of each element in compound
- `density_g_cm3`: Crystal density
- `k_edges_keV`: K-edge energies for each element
- `k_alpha_keV`: K-α fluorescence line energies
- `k_yields`: Fluorescence yields ω_K for each element
- `k_fluorescence_range_mm`: Mean free path of fluorescence X-rays in crystal
- `mu_e_tau_e`: Electron mobility-lifetime product (cm²/V)
- `mu_h_tau_h`: Hole mobility-lifetime product (cm²/V)
- `bias_voltage_V`: Typical operating bias voltage
- `pair_creation_energy_eV`: Energy per electron-hole pair
- `fano_factor`: Fano factor for intrinsic energy resolution
- `charge_cloud_sigma_mm`: Charge cloud size at typical bias
- `energy_resolution_fwhm_keV`: Measured system FWHM at 60 keV

# References
- CdTe: Taguchi & Iwanczyk, Med Phys 2013; del Risco Norrlid et al, NIMPA 2017
- CZT: Pennicard et al, JINST 2014; Veale et al, NIMPA 2014
- Si: Persson et al, Phys Med Biol 2014; Fredenberg et al, NIMPA 2010

# Example
```julia
props = get_detector_material_properties(CDTE_MATERIAL)
props.density_g_cm3  # 5.85
props.k_edges_keV    # [26.7, 31.8]
```
"""
function get_detector_material_properties(material::DetectorMaterialPCCT)
    if material == CDTE_MATERIAL
        # Cadmium Telluride — primary PCCT detector material (Siemens NAEOTOM Alpha)
        # Atomic weights: Cd = 112.411, Te = 127.60 → total = 240.011
        return (
            elements = [:Cd, :Te],
            atomic_numbers = [48, 52],
            mass_fractions = [112.411 / 240.011, 127.60 / 240.011],  # [0.4683, 0.5317]
            density_g_cm3 = 5.85,

            # K-edges and fluorescence (NIST XCOM)
            k_edges_keV = [26.7, 31.8],           # Cd K-edge, Te K-edge
            k_alpha_keV = [23.2, 27.4],           # Cd K-α, Te K-α fluorescence
            k_yields = [0.84, 0.87],              # Fluorescence yields ω_K (Bambynek 1972)
            k_fluorescence_range_mm = [0.12, 0.10], # Mean free path in CdTe crystal

            # Charge transport (Taguchi & Iwanczyk 2013, Table 2)
            mu_e_tau_e = 3.0e-3,                  # cm²/V (electrons — good mobility)
            mu_h_tau_h = 5.0e-5,                  # cm²/V (holes — poor mobility!)
            bias_voltage_V = 800.0,               # Typical operating voltage

            # Intrinsic energy resolution
            pair_creation_energy_eV = 4.43,       # eV per e-h pair
            fano_factor = 0.11,                   # Intrinsic variance reduction
            charge_cloud_sigma_mm = 0.04,         # At 800V bias, 1.6mm thick

            # System-level measured resolution (includes electronics)
            energy_resolution_fwhm_keV = 10.0     # Typical for NAEOTOM at 60 keV
        )
    elseif material == CZT_MATERIAL
        # Cadmium Zinc Telluride (Cd₀.₉Zn₀.₁Te)
        # Zn substitutes ~10% of Cd → slightly different properties
        # Atomic weights: Cd=112.411, Zn=65.38, Te=127.60
        # CZT formula: Cd₀.₉Zn₀.₁Te → effective mass per formula unit
        # = 0.9×112.411 + 0.1×65.38 + 127.60 = 234.307
        eff_mass = 0.9 * 112.411 + 0.1 * 65.38 + 127.60  # 234.307
        return (
            elements = [:Cd, :Zn, :Te],
            atomic_numbers = [48, 30, 52],
            mass_fractions = [0.9 * 112.411 / eff_mass, 0.1 * 65.38 / eff_mass, 127.60 / eff_mass],
            density_g_cm3 = 5.78,                 # Slightly lower than CdTe

            # K-edges and fluorescence
            k_edges_keV = [26.7, 9.66, 31.8],    # Cd, Zn, Te K-edges
            k_alpha_keV = [23.2, 8.63, 27.4],    # Cd, Zn, Te K-α lines
            k_yields = [0.84, 0.47, 0.87],       # ω_K values (Zn lower yield)
            k_fluorescence_range_mm = [0.12, 0.50, 0.10], # Zn K-α has long range (low-E)

            # Charge transport (Pennicard et al 2014)
            mu_e_tau_e = 1.0e-2,                  # cm²/V (better than CdTe)
            mu_h_tau_h = 5.0e-5,                  # cm²/V (similar hole trapping)
            bias_voltage_V = 600.0,               # Lower voltage needed

            # Intrinsic energy resolution
            pair_creation_energy_eV = 4.64,       # eV per e-h pair (Owens 2004)
            fano_factor = 0.089,                  # Better than CdTe (Pennicard 2014)
            charge_cloud_sigma_mm = 0.045,        # Slightly larger cloud

            energy_resolution_fwhm_keV = 8.0      # CZT typically better resolution
        )
    elseif material == SI_MATERIAL
        # Silicon — excellent charge transport but very low stopping power at CT energies
        # Atomic weight: Si = 28.085
        return (
            elements = [:Si],
            atomic_numbers = [14],
            mass_fractions = [1.0],
            density_g_cm3 = 2.33,

            # K-edge (irrelevant for CT — 1.84 keV is far below useful range)
            k_edges_keV = [1.84],
            k_alpha_keV = [1.74],                 # Si K-α (irrelevant at CT energies)
            k_yields = [0.05],                    # Very low fluorescence yield for low-Z
            k_fluorescence_range_mm = [0.001],    # Absorbed immediately

            # Charge transport (excellent! — Persson 2014)
            mu_e_tau_e = 1.0,                     # cm²/V (near-perfect collection)
            mu_h_tau_h = 0.5,                     # cm²/V (also excellent)
            bias_voltage_V = 200.0,               # Low voltage sufficient

            # Intrinsic energy resolution (best of all three)
            pair_creation_energy_eV = 3.62,       # eV per e-h pair (lowest)
            fano_factor = 0.115,                  # Similar to CdTe
            charge_cloud_sigma_mm = 0.015,        # Very small cloud (light material)

            energy_resolution_fwhm_keV = 2.5      # Excellent energy resolution
        )
    else
        error("Unknown detector material: $material")
    end
end

"""
    get_detector_material_attenuation(material::DetectorMaterialPCCT, energy_keV::Real) -> Float64

Get linear attenuation coefficient μ (cm⁻¹) for PCCT detector material at given energy.

Uses the tabulated data in DetectorEfficiency.jl (SCINTILLATOR_MU_DATA) with log-linear
interpolation. This provides NIST XCOM-calibrated values with proper K-edge handling.

# Arguments
- `material::DetectorMaterialPCCT`: Detector material enum
- `energy_keV::Real`: Photon energy in keV

# Returns
- `Float64`: Linear attenuation coefficient in cm⁻¹

# Example
```julia
μ = get_detector_material_attenuation(CDTE_MATERIAL, 60.0)  # ~33 cm⁻¹
μ = get_detector_material_attenuation(SI_MATERIAL, 60.0)    # ~0.46 cm⁻¹
```
"""
function get_detector_material_attenuation(material::DetectorMaterialPCCT, energy_keV::Real)
    mat_name = if material == CDTE_MATERIAL
        "CdTe"
    elseif material == CZT_MATERIAL
        "CZT"
    elseif material == SI_MATERIAL
        "Si"
    else
        error("Unknown detector material: $material")
    end
    return get_scintillator_mu(mat_name, Float64(energy_keV))
end

"""
    quantum_efficiency(material::DetectorMaterialPCCT, thickness_mm::Real, energy_keV::Real) -> Float64

Compute quantum detection efficiency η(E) for a PCCT detector crystal.

The probability that a photon of energy E is absorbed in the detector:
    η(E) = 1 - exp(-μ_det(E) × d)

where μ_det(E) is the linear attenuation of the detector material and d is thickness.

# Arguments
- `material::DetectorMaterialPCCT`: Detector crystal material
- `thickness_mm::Real`: Crystal thickness in mm
- `energy_keV::Real`: Photon energy in keV

# Returns
- `Float64`: Detection efficiency in [0, 1]

# Physics
- CdTe 1.6mm: η ≈ 0.99 at 60 keV (nearly opaque)
- CZT 2.0mm: η ≈ 0.99 at 60 keV (similar to CdTe)
- Si 1.6mm: η ≈ 0.05 at 60 keV (mostly transparent!)
- Si needs 30+ mm thickness for >90% efficiency at CT energies

# Example
```julia
η_cdte = quantum_efficiency(CDTE_MATERIAL, 1.6, 60.0)  # ≈ 0.995
η_si = quantum_efficiency(SI_MATERIAL, 1.6, 60.0)      # ≈ 0.071
```
"""
function quantum_efficiency(material::DetectorMaterialPCCT, thickness_mm::Real, energy_keV::Real)
    μ = get_detector_material_attenuation(material, energy_keV)  # cm⁻¹
    thickness_cm = Float64(thickness_mm) / 10.0
    return 1.0 - exp(-μ * thickness_cm)
end

"""
    quantum_efficiency_vector(material::DetectorMaterialPCCT, thickness_mm::Real,
                              energies::AbstractVector) -> Vector{Float64}

Compute quantum efficiency η(E) for a vector of energies (batch operation).

More efficient than calling `quantum_efficiency` in a loop for spectrum processing.
"""
function quantum_efficiency_vector(material::DetectorMaterialPCCT, thickness_mm::Real,
                                    energies::AbstractVector)
    return [quantum_efficiency(material, thickness_mm, E) for E in energies]
end

# =============================================================================
# K-Fluorescence Escape Model
# =============================================================================

"""
    KFluorescenceParams{T<:AbstractFloat}

Parameters for K-fluorescence escape modeling in PCCT detectors.

When a photon is absorbed above a material's K-edge, a K-shell vacancy is created.
This can lead to fluorescence X-ray emission that may escape the pixel, causing:
- Primary pixel: registers E_photon - E_fluorescence
- Neighbor pixel: registers E_fluorescence (if absorbed there)
- Net effect: spectral distortion near K-edges

# Fields
- `n_lines`: Number of K-fluorescence lines for this material
- `k_edge_energies`: K-edge energies [keV] (absorption threshold)
- `fluorescence_energies`: K-α emission line energies [keV]
- `fluorescence_yields`: ω_K (probability of fluorescence vs Auger)
- `escape_probabilities`: P_escape (fraction of fluorescence X-rays leaving pixel)
"""
struct KFluorescenceParams{T<:AbstractFloat}
    n_lines::Int
    k_edge_energies::Vector{T}
    fluorescence_energies::Vector{T}
    fluorescence_yields::Vector{T}
    escape_probabilities::Vector{T}
end

"""
    compute_fluorescence_escape_probability(material::DetectorMaterialPCCT,
                                             thickness_mm::Real,
                                             pixel_size_mm::Tuple{<:Real,<:Real}) -> KFluorescenceParams

Compute K-fluorescence escape parameters for a given detector geometry.

Escape probability depends on:
1. Mean free path of fluorescence X-ray in crystal (material-dependent)
2. Pixel dimensions (smaller pixels → more escapes)
3. Absorption depth distribution (shallower → more escape from top/bottom)

The model computes P_escape as the fraction of fluorescence photons whose
mean free path exceeds half the pixel dimension (simplified geometric model).

# Arguments
- `material`: Detector material
- `thickness_mm`: Crystal thickness in mm
- `pixel_size_mm`: Pixel dimensions (row, col) in mm

# References
- Cammin et al, "A cascaded model for spectral response of PCCT detectors"
  Med Phys 2014;41:041905
"""
function compute_fluorescence_escape_probability(
    material::DetectorMaterialPCCT,
    thickness_mm::Real,
    pixel_size_mm::Tuple{<:Real,<:Real}
)
    props = get_detector_material_properties(material)
    T = Float64

    n_lines = length(props.k_edges_keV)
    k_edges = T.(props.k_edges_keV)
    k_alphas = T.(props.k_alpha_keV)
    yields = T.(props.k_yields)
    ranges = T.(props.k_fluorescence_range_mm)

    # Compute escape probability for each fluorescence line
    # Geometric model: fraction of isotropically emitted photons that
    # travel far enough to leave the pixel
    escape_probs = zeros(T, n_lines)
    pixel_half_size = min(T(pixel_size_mm[1]), T(pixel_size_mm[2])) / 2.0
    half_thickness = T(thickness_mm) / 2.0

    for i in 1:n_lines
        # Skip lines below useful CT energy range (e.g., Si K-α at 1.74 keV)
        if k_alphas[i] < 5.0
            escape_probs[i] = 0.0
            continue
        end

        # Mean free path of fluorescence X-ray in crystal
        λ = ranges[i]  # mm

        # Probability of escape: P ≈ 1 - (1 - exp(-d/2λ))
        # where d is the smaller of pixel_size and thickness
        # This is a simplified solid-angle model
        # Lateral escape (out sides of pixel):
        p_lateral = exp(-pixel_half_size / λ)
        # Axial escape (out top/bottom of crystal):
        p_axial = exp(-half_thickness / λ)

        # Combined: fraction of fluorescence that escapes the pixel
        # Weight lateral more (4 sides vs 2 faces), but solid angle matters
        # Simplified: P_escape ≈ (4 × p_lateral + 2 × p_axial) / 6
        # but capped to reasonable values
        escape_probs[i] = min((4.0 * p_lateral + 2.0 * p_axial) / 6.0, 0.95)
    end

    return KFluorescenceParams{T}(n_lines, k_edges, k_alphas, yields, escape_probs)
end

"""
    apply_fluorescence_escape(E_incident::Real, fluorescence::KFluorescenceParams) -> Tuple{Float64, Float64}

Compute the probability and energy shift from K-fluorescence escape.

For incident photon energy E, returns:
- `p_escape`: Total probability of fluorescence escape event
- `E_registered`: Energy registered in primary pixel (E - E_fluorescence)

If E is below all K-edges, p_escape = 0 (no fluorescence possible).
If E is above multiple K-edges, the dominant (highest probability) line is used.
"""
function apply_fluorescence_escape(E_incident::Real, fluorescence::KFluorescenceParams{T}) where T
    E = T(E_incident)
    p_escape_total = zero(T)
    E_lost = zero(T)

    for i in 1:fluorescence.n_lines
        if E > fluorescence.k_edge_energies[i]
            # Photon energy is above this K-edge → fluorescence possible
            p_this = fluorescence.fluorescence_yields[i] * fluorescence.escape_probabilities[i]
            p_escape_total += p_this
            E_lost += p_this * fluorescence.fluorescence_energies[i]
        end
    end

    # Cap total escape probability
    p_escape_total = min(p_escape_total, 0.5)

    if p_escape_total > zero(T)
        E_registered = E - E_lost / p_escape_total  # Average energy if escape occurs
        return (Float64(p_escape_total), Float64(E_registered))
    else
        return (0.0, Float64(E))
    end
end

# =============================================================================
# Charge Collection Efficiency — Hecht Equation
# =============================================================================

"""
    ChargeTransportParams{T<:AbstractFloat}

Charge transport parameters for Hecht equation modeling.

# Fields
- `mu_e_tau_e`: Electron mobility-lifetime product (cm²/V)
- `mu_h_tau_h`: Hole mobility-lifetime product (cm²/V)
- `bias_voltage`: Applied voltage (V)
- `thickness_cm`: Crystal thickness (cm)
"""
struct ChargeTransportParams{T<:AbstractFloat}
    mu_e_tau_e::T
    mu_h_tau_h::T
    bias_voltage::T
    thickness_cm::T
end

"""
    get_charge_transport_params(material::DetectorMaterialPCCT, thickness_mm::Real) -> ChargeTransportParams

Get charge transport parameters for Hecht equation calculation.
"""
function get_charge_transport_params(material::DetectorMaterialPCCT, thickness_mm::Real)
    props = get_detector_material_properties(material)
    return ChargeTransportParams{Float64}(
        props.mu_e_tau_e,
        props.mu_h_tau_h,
        props.bias_voltage_V,
        Float64(thickness_mm) / 10.0
    )
end

"""
    charge_collection_efficiency(x_cm::Real, params::ChargeTransportParams) -> Float64

Compute charge collection efficiency (CCE) at absorption depth x using Hecht equation.

    CCE(x) = (μₑτₑ×E_field/d) × [1 - exp(-(d-x)×d/(μₑτₑ×V))]
           + (μₕτₕ×E_field/d) × [1 - exp(-x×d/(μₕτₕ×V))]

where:
- x = absorption depth from cathode (0 = cathode, d = anode)
- d = crystal thickness
- V = bias voltage
- E_field = V/d (uniform field approximation)

# Physics

In CdTe/CZT:
- Electrons (good mobility μₑτₑ ≈ 3×10⁻³): collected efficiently from anywhere
- Holes (poor mobility μₕτₕ ≈ 5×10⁻⁵): only collected if generated near anode

This creates an asymmetric low-energy tail (incomplete charge collection = lower
registered energy), which is the dominant spectral artifact in CdTe detectors.

In Si:
- Both carriers have excellent mobility → CCE ≈ 1.0 everywhere

# Arguments
- `x_cm::Real`: Absorption depth from cathode in cm (0 ≤ x ≤ thickness)
- `params::ChargeTransportParams`: Transport parameters

# Returns
- `Float64`: Charge collection efficiency in [0, 1]
"""
function charge_collection_efficiency(x_cm::Real, params::ChargeTransportParams{T}) where T
    d = params.thickness_cm
    V = params.bias_voltage
    x = clamp(T(x_cm), zero(T), d)

    # Electric field (V/cm) — uniform approximation
    E_field = V / d

    # Electron contribution (generated at x, drifts toward anode at d)
    # Drift length for electrons: (d - x)
    λ_e = params.mu_e_tau_e * E_field  # Mean drift length (cm)
    if λ_e > zero(T)
        cce_e = (λ_e / d) * (one(T) - exp(-(d - x) / λ_e))
    else
        cce_e = zero(T)
    end

    # Hole contribution (generated at x, drifts toward cathode at 0)
    # Drift length for holes: x
    λ_h = params.mu_h_tau_h * E_field  # Mean drift length (cm)
    if λ_h > zero(T)
        cce_h = (λ_h / d) * (one(T) - exp(-x / λ_h))
    else
        cce_h = zero(T)
    end

    return Float64(min(cce_e + cce_h, one(T)))
end

"""
    mean_charge_collection_efficiency(material::DetectorMaterialPCCT,
                                       thickness_mm::Real;
                                       n_points::Int=50) -> Float64

Compute depth-averaged charge collection efficiency for a detector crystal.

Averages CCE over the absorption depth distribution. For simplicity, uses
uniform absorption distribution (valid when μ×d is moderate).

# Returns
Mean CCE in [0, 1]. For CdTe: ~0.90-0.95. For Si: ~0.99+.
"""
function mean_charge_collection_efficiency(material::DetectorMaterialPCCT,
                                            thickness_mm::Real;
                                            n_points::Int=50)
    params = get_charge_transport_params(material, thickness_mm)
    d = params.thickness_cm

    # Average over depth (simple numerical integration)
    cce_sum = 0.0
    for i in 1:n_points
        x = d * (i - 0.5) / n_points
        cce_sum += charge_collection_efficiency(x, params)
    end

    return cce_sum / n_points
end

"""
    hole_tailing_distribution(E_incident::Real, material::DetectorMaterialPCCT,
                               thickness_mm::Real; n_depth::Int=20) -> Tuple{Vector{Float64}, Vector{Float64}}

Compute the energy distribution due to incomplete charge collection (hole tailing).

Returns (energies, probabilities) representing the distribution of registered
energies for a photon of true energy E_incident.

The distribution has:
- A peak near E_incident × mean_CCE (most photons)
- A low-energy tail from deep absorption events (poor hole collection)

# Returns
- `energies`: Registered energy values
- `weights`: Probability weights (sum to 1.0)
"""
function hole_tailing_distribution(E_incident::Real, material::DetectorMaterialPCCT,
                                    thickness_mm::Real; n_depth::Int=20)
    params = get_charge_transport_params(material, thickness_mm)
    d = params.thickness_cm
    E = Float64(E_incident)

    energies = zeros(Float64, n_depth)
    weights = zeros(Float64, n_depth)

    for i in 1:n_depth
        x = d * (i - 0.5) / n_depth
        cce = charge_collection_efficiency(x, params)
        energies[i] = E * cce
        weights[i] = 1.0 / n_depth  # Uniform absorption approximation
    end

    return (energies, weights)
end

# =============================================================================
# Spectral Response Matrix R(E, b)
# =============================================================================

"""
    compute_spectral_response_matrix(material::DetectorMaterialPCCT,
                                      thickness_mm::Real,
                                      thresholds_keV::AbstractVector,
                                      kVp::Real;
                                      energy_resolution_keV::Real=10.0,
                                      pixel_size_mm::Tuple{<:Real,<:Real}=(0.302, 0.302),
                                      include_fluorescence::Bool=true,
                                      include_tailing::Bool=true,
                                      n_energy_points::Int=200) -> Matrix{Float64}

Compute the spectral response matrix R(E, b) for a PCCT detector.

R[i, b] = probability that a photon of true energy E_i is registered in bin b.

The matrix combines three physical effects:
1. **Gaussian energy blur**: Finite energy resolution broadens the step function
2. **K-fluorescence escape**: Photons above K-edge may lose fluorescence energy
3. **Hole tailing**: Incomplete charge collection creates low-energy tail

# Matrix Properties
- Size: [n_energy_points × n_bins]
- Each row sums to ≤ 1.0 (photons below lowest threshold are lost)
- Ideal detector: R is a block-diagonal step function
- Realistic: R has off-diagonal entries (spectral cross-talk between bins)

# Arguments
- `material`: Detector crystal material
- `thickness_mm`: Crystal thickness in mm
- `thresholds_keV`: Energy thresholds defining bins
- `kVp`: Maximum tube voltage (defines upper energy bound)
- `energy_resolution_keV`: System FWHM in keV (default: from material properties)
- `pixel_size_mm`: Pixel dimensions for fluorescence escape calculation
- `include_fluorescence`: Whether to model K-fluorescence escape
- `include_tailing`: Whether to model hole tailing
- `n_energy_points`: Number of energy sample points

# Returns
- `Matrix{Float64}`: [n_energy_points × n_bins] response matrix

# Performance Note
This matrix is computed ONCE per detector configuration and reused for all
projections. It does NOT need GPU acceleration (precomputed on CPU).

# Example
```julia
R = compute_spectral_response_matrix(
    CDTE_MATERIAL, 1.6, [20.0, 35.0, 55.0, 70.0], 120.0
)
# R[100, 3]  → probability that a 60 keV photon registers in bin 3
```
"""
function compute_spectral_response_matrix(
    material::DetectorMaterialPCCT,
    thickness_mm::Real,
    thresholds_keV::AbstractVector,
    kVp::Real;
    energy_resolution_keV::Real=0.0,
    pixel_size_mm::Tuple{<:Real,<:Real}=(0.302, 0.302),
    include_fluorescence::Bool=true,
    include_tailing::Bool=true,
    n_energy_points::Int=200
)
    props = get_detector_material_properties(material)
    n_bins = length(thresholds_keV)

    # Use material-specific resolution if not overridden
    σ_E = if energy_resolution_keV > 0.0
        energy_resolution_keV / (2.0 * sqrt(2.0 * log(2.0)))
    else
        props.energy_resolution_fwhm_keV / (2.0 * sqrt(2.0 * log(2.0)))
    end

    # Energy grid: from 1 keV to kVp
    E_min = 1.0
    E_max = Float64(kVp)
    energies = range(E_min, E_max, length=n_energy_points)

    # Fluorescence parameters
    fluorescence = if include_fluorescence
        compute_fluorescence_escape_probability(material, thickness_mm, pixel_size_mm)
    else
        nothing
    end

    # Charge transport for tailing
    transport_params = if include_tailing
        get_charge_transport_params(material, thickness_mm)
    else
        nothing
    end
    mean_cce = include_tailing ? mean_charge_collection_efficiency(material, thickness_mm) : 1.0

    # Build response matrix
    R = zeros(Float64, n_energy_points, n_bins)

    for (i, E) in enumerate(energies)
        # Threshold values
        T_values = Float64.(thresholds_keV)

        # Start with the primary photopeak at energy E
        # Apply hole tailing: distribute E across CCE values
        if include_tailing && transport_params !== nothing
            tail_energies, tail_weights = hole_tailing_distribution(E, material, thickness_mm; n_depth=20)
        else
            tail_energies = [E]
            tail_weights = [1.0]
        end

        for (t_idx, E_tail) in enumerate(tail_energies)
            w_tail = tail_weights[t_idx]

            # Apply fluorescence escape: split into primary and escape peaks
            if include_fluorescence && fluorescence !== nothing
                p_escape, E_escaped = apply_fluorescence_escape(E_tail, fluorescence)
                # Two components: (1-p_escape) at E_tail, p_escape at E_escaped
                energy_components = [(E_tail, 1.0 - p_escape), (E_escaped, p_escape)]
            else
                energy_components = [(E_tail, 1.0)]
            end

            for (E_comp, w_comp) in energy_components
                # Apply Gaussian energy blur and bin
                for b in 1:n_bins
                    T_low = T_values[b]

                    if σ_E > 0.0
                        z_low = (T_low - E_comp) / (σ_E * sqrt(2.0))
                        if b == n_bins
                            # Last bin: one-sided CDF (everything above T_low)
                            prob = 0.5 * (1.0 - _erf_approx(z_low))
                        else
                            # Interior bins: two-sided CDF
                            T_high = T_values[b + 1]
                            z_high = (T_high - E_comp) / (σ_E * sqrt(2.0))
                            prob = 0.5 * (_erf_approx(z_high) - _erf_approx(z_low))
                        end
                    else
                        # Perfect resolution: delta function binning
                        if b == n_bins
                            prob = E_comp >= T_low ? 1.0 : 0.0
                        else
                            T_high = T_values[b + 1]
                            prob = (E_comp >= T_low && E_comp < T_high) ? 1.0 : 0.0
                        end
                    end

                    R[i, b] += w_tail * w_comp * max(prob, 0.0)
                end
            end
        end
    end

    # Ensure no row exceeds 1.0 (physical constraint: photon counted at most once)
    for i in 1:n_energy_points
        row_sum = sum(R[i, :])
        if row_sum > 1.0
            R[i, :] ./= row_sum
        end
    end

    return R
end

"""
    get_spectral_response_energies(kVp::Real; n_energy_points::Int=200) -> Vector{Float64}

Get the energy grid corresponding to the spectral response matrix.

Returns the energy values (keV) for each row of the matrix returned by
`compute_spectral_response_matrix`.
"""
function get_spectral_response_energies(kVp::Real; n_energy_points::Int=200)
    return collect(range(1.0, Float64(kVp), length=n_energy_points))
end

# =============================================================================
# Exports
# =============================================================================

export DetectorMaterialPCCT, CDTE_MATERIAL, CZT_MATERIAL, SI_MATERIAL
export PhotonCountingDetector
export naeotom_detector_standard, naeotom_detector_uhr, pcct_detector_ideal
export EnergyResolvedSinogram, n_energy_bins
export apply_energy_thresholds
export apply_charge_sharing!, apply_pulse_pileup!
export apply_anti_coincidence!, apply_pcct_electronic_noise!
export pcct_forward_project
export get_pcct_detector_info, print_pcct_detector_info
export get_detector_material_properties, get_detector_material_attenuation
export quantum_efficiency, quantum_efficiency_vector
export KFluorescenceParams, compute_fluorescence_escape_probability, apply_fluorescence_escape
export ChargeTransportParams, get_charge_transport_params
export charge_collection_efficiency, mean_charge_collection_efficiency
export hole_tailing_distribution
export compute_spectral_response_matrix, get_spectral_response_energies
export apply_pcct_noise!
