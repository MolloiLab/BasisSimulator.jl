# =============================================================================
# Data Acquisition System (DAS) Model
# =============================================================================
#
# Models the detector signal conversion chain:
# 1. X-ray photons → scintillator light
# 2. Light → photodiode current
# 3. Current → digital counts (ADC)
#
# Reference: CatSim DAS parameters (das_gain, das_enoise, das_lsb)
#
# IMPORTANT: BasisSimulator operates in normalized intensity domain (0-1),
# not in energy (keV) or photon count domain like CatSim. The DAS model
# parameters are scaled internally to work correctly with intensity signals.
#
# The key relationship:
#   CatSim signal (keV) ≈ I0 × mean_energy × intensity
#   where I0 ~ 1e5-1e6 photons, mean_energy ~ 60 keV
#
# Electronic noise in CatSim (e.g., 5000 electrons) relative to signal:
#   noise_ratio = electronic_noise_sigma / (I0 × mean_energy × gain)
#                ≈ 5000 / (1e6 × 60 × 15) ≈ 5.5e-9
#
# =============================================================================

import AcceleratedKernels as AK

export DASModel
export default_das_model, das_clinical
export apply_das_model!, apply_das_model
export get_das_info

# =============================================================================
# DAS Model Type
# =============================================================================

"""
    DASModel

Data Acquisition System model for CT detector signal chain.

# Fields
- `gain`: Multiplicative gain factor applied to signal (dimensionless for normalized intensity)
- `electronic_noise_sigma`: Electronic noise standard deviation (RELATIVE to signal, not absolute)
- `reference_signal`: Expected signal level for noise scaling (default: 1e6, typical photon fluence)
- `lsb`: Least significant bit for quantization (0 = no quantization)
- `min_value`: Minimum output value (for truncation)
- `max_value`: Maximum output value (for saturation)
- `offset`: DC offset added to signal

# Signal chain (for normalized intensity input):
1. signal_out = signal_in × gain
2. noise_sigma_scaled = electronic_noise_sigma / reference_signal
3. signal_out += Normal(0, noise_sigma_scaled)
4. signal_out = round(signal_out / lsb) × lsb  (if lsb > 0)
5. signal_out = clamp(signal_out, min_value, max_value)
6. signal_out += offset

# Note on units:
When using CatSim-like parameters (e.g., electronic_noise_sigma=5000 electrons,
gain=15 electrons/keV), set reference_signal to the expected integrated signal
level (e.g., I0 × mean_energy × gain ≈ 1e6 × 60 × 15 ≈ 9e8).

For BasisSimulator's intensity domain, you can equivalently use:
- gain = 1.0 (no gain change since we normalize)
- electronic_noise_sigma = 0.01 (1% relative noise)
- reference_signal = 1.0 (signal is already normalized)
"""
struct DASModel
    gain::Float64               # multiplicative gain (dimensionless for intensity)
    electronic_noise_sigma::Float64  # noise sigma (in same units as reference_signal)
    reference_signal::Float64   # expected signal level for noise scaling
    lsb::Float64                # quantization step (0 = none)
    min_value::Float64          # minimum output
    max_value::Float64          # maximum output (saturation)
    offset::Float64             # DC offset
end

# =============================================================================
# Default DAS Models
# =============================================================================

"""
    default_das_model(; gain=1.0, electronic_noise_sigma=0.0, reference_signal=1.0, ...)

Create DAS model with specified parameters.

# For BasisSimulator (intensity domain):
- gain: 1.0 (multiplicative, cancels in calibration)
- electronic_noise_sigma: 0.0 (no additional noise beyond quantum noise)
- reference_signal: 1.0 (signal is normalized intensity)

# For CatSim-like parameters:
Use `das_catsim_compatible()` which converts CatSim's electron-based
parameters to work with intensity domain.

# Example
```julia
# No DAS noise (default)
das = default_das_model()

# Small relative electronic noise (1% of signal)
das = default_das_model(electronic_noise_sigma=0.01, reference_signal=1.0)
```
"""
function default_das_model(;
    gain::Real = 1.0,
    electronic_noise_sigma::Real = 0.0,
    reference_signal::Real = 1.0,
    lsb::Real = 0.0,
    min_value::Real = 0.0,
    max_value::Real = Inf,
    offset::Real = 0.0
)
    return DASModel(
        Float64(gain),
        Float64(electronic_noise_sigma),
        Float64(reference_signal),
        Float64(lsb),
        Float64(min_value),
        Float64(max_value),
        Float64(offset)
    )
end


"""
    das_clinical(; noise_level=1.0, I0=1e6, mean_energy_keV=60.0)

Clinical-grade DAS model with realistic electronic noise.

The electronic noise level is calibrated to match CatSim's typical clinical
parameters (5000 electrons σ with gain of 15 e-/keV).

# Parameters
- `noise_level`: Multiplier for electronic noise (1.0 = typical clinical)
- `I0`: Photon fluence (used for noise scaling, default: 1e6)
- `mean_energy_keV`: Mean photon energy (default: 60 keV for 120 kVp)

# Default produces approximately:
- 5-15 HU additional noise std from DAS electronics
- Matches CatSim clinical simulation parameters

# Note
The quantum noise (from `DetectorModel.noise`) typically dominates over
electronic noise in well-designed clinical CT. Electronic noise becomes
significant only at very low dose.
"""
function das_clinical(;
    noise_level::Real = 1.0,
    I0::Real = 1e6,
    mean_energy_keV::Real = 60.0
)
    # Match CatSim clinical parameters:
    # - electronic_noise_electrons = 5000 (typical)
    # - gain_electrons_per_keV = 15 (typical GOS)
    #
    # Convert to relative noise in intensity domain:
    gain = 15.0
    electronic_noise_electrons = 5000.0 * noise_level
    expected_signal = I0 * mean_energy_keV * gain
    noise_relative = electronic_noise_electrons / expected_signal

    return DASModel(
        1.0,              # gain (no change, cancels in calibration)
        noise_relative,   # electronic noise (relative to full signal)
        1.0,              # reference signal (normalized intensity)
        0.0,              # no quantization in simulation
        0.0,              # min value
        Inf,              # max value (no saturation)
        0.0               # no offset
    )
end


# =============================================================================
# DAS Model Application
# =============================================================================

"""
    apply_das_model!(signal, das; seed=nothing)

Apply DAS model to signal (in-place).

This applies the full DAS signal chain:
1. Scale by gain
2. Add electronic noise (scaled by electronic_noise_sigma / reference_signal)
3. Apply quantization (if lsb > 0)
4. Clamp to [min_value, max_value]
5. Add offset

# Arguments
- `signal`: Detector signal array (modified in place)
- `das`: DASModel with parameters

# Keyword Arguments
- `seed`: Random seed for reproducibility (nothing = random)

# Signal Domain
This function works with normalized intensity signals (0-1 range) as used
by BasisSimulator. The electronic noise is scaled by the ratio
(electronic_noise_sigma / reference_signal) to properly match the signal level.

For CatSim-compatible parameters, use `das_catsim_compatible()` which handles
the unit conversion automatically.
"""
function apply_das_model!(
    signal::AbstractArray{T, 3},
    das::DASModel;
    seed::Union{Nothing, Int} = nothing
) where T <: AbstractFloat

    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Extract all scalar parameters at top level to avoid boxing in closures
    gain = T(das.gain)
    lsb = T(das.lsb)
    min_val = T(das.min_value)
    max_val = T(das.max_value)
    offset = T(das.offset)
    has_noise = das.electronic_noise_sigma > 0
    has_quant = lsb > 0

    # Compute scaled noise sigma for intensity domain
    # noise_sigma_scaled = electronic_noise_sigma / reference_signal
    noise_sigma_scaled = T(das.electronic_noise_sigma / das.reference_signal)

    # Generate noise on CPU, transfer to GPU
    if has_noise
        noise_cpu = randn(Float32, size(signal)) .* Float32(noise_sigma_scaled)
        noise = similar(signal)
        copyto!(noise, noise_cpu)

        if has_quant
            AK.foreachindex(signal) do idx
                s = signal[idx] * gain + noise[idx]
                s = round(s / lsb) * lsb  # Quantization
                s = clamp(s, min_val, max_val)
                signal[idx] = s + offset
            end
        else
            AK.foreachindex(signal) do idx
                s = signal[idx] * gain + noise[idx]
                s = clamp(s, min_val, max_val)
                signal[idx] = s + offset
            end
        end
    else
        # No noise - just gain, quantization, clamping
        if has_quant
            AK.foreachindex(signal) do idx
                s = signal[idx] * gain
                s = round(s / lsb) * lsb
                s = clamp(s, min_val, max_val)
                signal[idx] = s + offset
            end
        else
            AK.foreachindex(signal) do idx
                s = signal[idx] * gain
                s = clamp(s, min_val, max_val)
                signal[idx] = s + offset
            end
        end
    end

    return signal
end

"""
    apply_das_model(signal, das; seed=nothing)

Non-mutating version of apply_das_model!.
"""
function apply_das_model(
    signal::AbstractArray{T, 3},
    das::DASModel;
    seed::Union{Nothing, Int} = nothing
) where T <: AbstractFloat
    result = similar(signal)
    copyto!(result, signal)
    return apply_das_model!(result, das; seed=seed)
end


# =============================================================================
# Utilities
# =============================================================================

"""
    get_das_info(das)

Get information about DAS model parameters.
"""
function get_das_info(das::DASModel)
    return (
        gain = das.gain,
        electronic_noise_sigma = das.electronic_noise_sigma,
        reference_signal = das.reference_signal,
        effective_noise = das.electronic_noise_sigma / das.reference_signal,
        lsb = das.lsb,
        min_value = das.min_value,
        max_value = das.max_value,
        offset = das.offset,
        has_quantization = das.lsb > 0,
        has_noise = das.electronic_noise_sigma > 0
    )
end
