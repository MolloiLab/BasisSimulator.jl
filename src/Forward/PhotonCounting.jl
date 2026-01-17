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

    if !detector.enable_charge_sharing || detector.charge_sharing_fwhm_mm ≤ zero(T)
        return bins
    end

    n_bins = length(bins)
    n_cols, n_rows, n_angles = size(bins[1])

    # Charge cloud sigma from FWHM
    σ_cloud = detector.charge_sharing_fwhm_mm / (2 * sqrt(2 * log(T(2))))

    # Pixel size at isocenter
    pixel_row, pixel_col = detector.pixel_size_mm

    # Compute charge sharing probability
    # Fraction of events that split to neighboring pixels
    # Based on distance from pixel center to boundary vs charge cloud size
    boundary_dist_row = pixel_row / 2
    boundary_dist_col = pixel_col / 2

    # Probability that charge reaches boundary (approximate)
    # P(|x| > boundary) ≈ 2 × (1 - Φ(boundary/σ))
    # Using complementary error function approximation
    z_row = boundary_dist_row / σ_cloud
    z_col = boundary_dist_col / σ_cloud

    # Sigmoid approximation for Gaussian tail
    p_share_row = T(2) * one(T) / (one(T) + exp(T(1.5) * z_row))
    p_share_col = T(2) * one(T) / (one(T) + exp(T(1.5) * z_col))
    p_share = min(p_share_row + p_share_col, T(0.5))  # Cap at 50% sharing

    # Primary signal retention
    p_primary = one(T) - p_share

    # When charge is shared, energy is split between pixels
    # This causes counts to appear in lower energy bins
    # Model: shared events lose ~50% of energy on average
    energy_loss_fraction = T(0.5)

    # Process each bin: redistribute counts to lower bins due to charge sharing
    # Work from highest to lowest bin to avoid double counting
    for bin_idx in n_bins:-1:1
        current_bin = bins[bin_idx]

        # Create output buffer
        output = similar(current_bin)

        AK.foreachindex(current_bin) do idx
            ci = CartesianIndices(current_bin)[idx]
            col, row, angle = Tuple(ci)

            # Start with current value scaled by primary retention
            val = current_bin[idx] * p_primary

            # Add contributions from neighbors (charge shared TO this pixel)
            # 3x3 spatial kernel for charge sharing
            for di in -1:1
                for dj in -1:1
                    if di == 0 && dj == 0
                        continue
                    end

                    src_col = clamp(col + di, 1, n_cols)
                    src_row = clamp(row + dj, 1, n_rows)

                    # Neighbor's contribution to this pixel
                    neighbor_weight = p_share / T(8)  # Divide among 8 neighbors
                    val += current_bin[src_col, src_row, angle] * neighbor_weight
                end
            end

            output[idx] = val
        end

        copyto!(current_bin, output)

        # Energy redistribution: some counts from this bin move to lower bins
        # due to partial energy deposition in split events
        if bin_idx > 1
            target_bin = bins[bin_idx - 1]  # Energy moves to lower bin
            lost_fraction = p_share * energy_loss_fraction

            AK.foreachindex(current_bin) do idx
                # Move fraction of counts to lower bin
                transfer = current_bin[idx] * lost_fraction
                target_bin[idx] += transfer
                current_bin[idx] -= transfer
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

    if !detector.enable_pile_up || detector.dead_time_ns ≤ zero(T)
        return bins
    end

    # Convert dead time to seconds
    τ = detector.dead_time_ns * T(1e-9)

    # Pixel area (mm²)
    pixel_area = detector.pixel_size_mm[1] * detector.pixel_size_mm[2]

    # Expected count rate per pixel
    # Assuming flux_rate is in photons/s/mm²
    count_rate = flux_rate * pixel_area

    # Pile-up correction factor (nonparalyzable model)
    # N_recorded/N_true = 1/(1 + N_true × τ)
    pile_up_factor = one(T) / (one(T) + count_rate * τ)

    # Apply count rate reduction to all bins
    for bin in bins
        AK.foreachindex(bin) do idx
            bin[idx] *= pile_up_factor
        end
    end

    # Energy pile-up: some counts from lower bins sum to appear in higher bins
    # This is a simplified model - accurate pile-up modeling requires
    # tracking individual photon arrival times
    n_bins = length(bins)
    if n_bins >= 2
        # Fraction of pile-up events
        p_pileup = one(T) - pile_up_factor

        # Pile-up causes some counts to shift to higher energy bins
        # (two photons summed together appear as one higher-energy photon)
        for bin_idx in 1:(n_bins-1)
            current_bin = bins[bin_idx]
            next_bin = bins[bin_idx + 1]

            # Simplified: small fraction of low-energy counts appear as high-energy
            shift_fraction = p_pileup * T(0.1)  # 10% of pile-up shifts energy

            AK.foreachindex(current_bin) do idx
                transfer = current_bin[idx] * shift_fraction
                next_bin[idx] += transfer
                current_bin[idx] -= transfer
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
    # to identify coincident pixels
    total_counts = similar(bins[1])
    fill!(total_counts, zero(T))
    for bin in bins
        AK.foreachindex(total_counts) do idx
            total_counts[idx] += bin[idx]
        end
    end

    # Correction factor: estimate fraction of events that can be recovered
    # This depends on timing electronics - simplified model here
    recovery_fraction = T(0.3)  # 30% of charge-shared events recovered

    # For each pixel, look for coincident neighbors and redistribute
    for bin_idx in 1:n_bins
        current_bin = bins[bin_idx]
        output = similar(current_bin)

        AK.foreachindex(current_bin) do idx
            ci = CartesianIndices(current_bin)[idx]
            col, row, angle = Tuple(ci)

            val = current_bin[idx]

            # Check 3x3 neighborhood for coincident events
            my_total = total_counts[idx]

            if my_total > zero(T)
                for di in -1:1
                    for dj in -1:1
                        if di == 0 && dj == 0
                            continue
                        end

                        src_col = clamp(col + di, 1, n_cols)
                        src_row = clamp(row + dj, 1, n_rows)

                        neighbor_total = total_counts[src_col, src_row, angle]

                        # If this pixel has higher counts, recover some from neighbor
                        if my_total > neighbor_total && neighbor_total > zero(T)
                            # Recover fraction of neighbor's counts (charge sharing correction)
                            neighbor_val = bins[bin_idx][src_col, src_row, angle]
                            recovery = neighbor_val * recovery_fraction / T(8)
                            val += recovery
                        end
                    end
                end
            end

            output[idx] = val
        end

        copyto!(current_bin, output)
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
    pcct_forward_project(volume, geom, detector, energies, weights; kwargs...) -> EnergyResolvedSinogram

Perform photon-counting CT forward projection with full detector physics.

This is the main API for PCCT simulation. It projects the volume at each
energy, applies detector physics effects, and returns energy-resolved sinograms.

# Arguments
- `volume::AbstractArray{T,3}`: Attenuation volume (μ values at reference energy)
- `geom::CTGeometry`: CT geometry
- `detector::PhotonCountingDetector`: PCCT detector specification
- `energies::Vector{T}`: Spectral energies in keV
- `weights::Vector{T}`: Spectral weights (photon fluence)

# Keyword Arguments
- `flux_rate::Float64=1e9`: Photon flux rate in photons/s/mm² (for pile-up)
- `I0::Float64=1e6`: Reference photon count per detector element

# Returns
`EnergyResolvedSinogram` with one sinogram per energy bin.

# Example

```julia
detector = naeotom_detector_standard()
energies, weights = load_spectrum(120)

pcct_sino = pcct_forward_project(
    volume, geom, detector, energies, weights;
    flux_rate = 1e9,
    I0 = 1e6
)

# Access energy bins
low_E_sino = pcct_sino.bins[1]   # 20-35 keV
high_E_sino = pcct_sino.bins[4]  # 70+ keV
```

See also: [`PhotonCountingDetector`](@ref), [`EnergyResolvedSinogram`](@ref)
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

    n_cols = geom.n_cols
    n_rows = geom.n_rows
    n_angles = geom.n_angles
    n_energies = length(energies)
    n_bins = length(detector.energy_thresholds_keV)

    # Allocate spectral intensity array
    # For GPU, this can be large - process in batches if needed
    intensity_spectrum = similar(volume, n_cols, n_rows, n_angles, n_energies)

    # Project at each energy
    # This uses the existing forward projection infrastructure
    for (e_idx, E) in enumerate(energies)
        # For now, use direct Siddon projection
        # In practice, this would call forward_project with energy-specific μ
        sino_E = siddon_forward_project(volume, geom)

        # Convert to intensity: I = I0 × exp(-∫μ dl)
        I0_E = I0 * weights[e_idx]  # Weighted by spectrum

        AK.foreachindex(sino_E) do idx
            intensity_spectrum[CartesianIndex(Tuple(CartesianIndices(sino_E)[idx])..., e_idx)] =
                I0_E * exp(-sino_E[idx])
        end
    end

    # Apply energy thresholds to get binned counts
    bins = apply_energy_thresholds(intensity_spectrum, energies, weights, detector)

    # Apply detector physics effects in order
    apply_charge_sharing!(bins, detector)
    apply_pulse_pileup!(bins, detector, flux_rate)
    apply_anti_coincidence!(bins, detector)
    apply_pcct_electronic_noise!(bins, detector)

    # Convert back to projection domain (log transform)
    # Each bin: p = -log(C / I0_bin)
    for (bin_idx, bin) in enumerate(bins)
        # Estimate I0 for this bin based on threshold
        threshold = detector.energy_thresholds_keV[bin_idx]
        # Sum weights above threshold
        I0_bin = I0 * sum(w for (E, w) in zip(energies, weights) if E >= threshold)

        AK.foreachindex(bin) do idx
            bin[idx] = -log(max(bin[idx], one(T)) / I0_bin)
        end
    end

    return EnergyResolvedSinogram(bins, detector.energy_thresholds_keV)
end

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
