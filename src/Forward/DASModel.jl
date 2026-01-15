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
# =============================================================================

import AcceleratedKernels as AK

export DASModel
export default_das_model, das_ideal, das_clinical
export apply_das_model!, apply_das_model
export apply_tube_current!, apply_tube_current
export get_das_info

# =============================================================================
# DAS Model Type
# =============================================================================

"""
    DASModel

Data Acquisition System model for CT detector signal chain.

# Fields
- `gain`: Gain factor (electrons per keV of deposited X-ray energy)
- `electronic_noise_sigma`: Electronic noise standard deviation (in electrons)
- `lsb`: Least significant bit for quantization (0 = no quantization)
- `min_value`: Minimum output value (for truncation)
- `max_value`: Maximum output value (for saturation)
- `offset`: DC offset added to signal

# Signal chain:
1. signal_electrons = deposited_energy_keV × gain
2. signal_electrons += Normal(0, electronic_noise_sigma)
3. signal_digital = round(signal_electrons / lsb) × lsb  (if lsb > 0)
4. signal_digital = clamp(signal_digital, min_value, max_value)
"""
struct DASModel
    gain::Float64               # electrons per keV
    electronic_noise_sigma::Float64  # noise sigma in electrons
    lsb::Float64                # quantization step (0 = none)
    min_value::Float64          # minimum output
    max_value::Float64          # maximum output (saturation)
    offset::Float64             # DC offset
end

# =============================================================================
# Default DAS Models
# =============================================================================

"""
    default_das_model(; gain=20.0, electronic_noise_sigma=100.0, lsb=0.0, min_value=0.0, max_value=Inf, offset=0.0)

Create DAS model with specified parameters.

# Default values approximate typical clinical CT:
- gain: 20 electrons/keV (typical GOS scintillator + photodiode)
- electronic_noise: 100 electrons RMS
- lsb: 0 (no quantization for floating point simulation)
"""
function default_das_model(;
    gain::Real = 20.0,
    electronic_noise_sigma::Real = 100.0,
    lsb::Real = 0.0,
    min_value::Real = 0.0,
    max_value::Real = Inf,
    offset::Real = 0.0
)
    return DASModel(
        Float64(gain),
        Float64(electronic_noise_sigma),
        Float64(lsb),
        Float64(min_value),
        Float64(max_value),
        Float64(offset)
    )
end

"""
    das_ideal()

Ideal DAS with no noise or quantization (for testing).
"""
function das_ideal()
    return DASModel(1.0, 0.0, 0.0, -Inf, Inf, 0.0)
end

"""
    das_clinical(; noise_level=1.0)

Clinical-grade DAS model with realistic parameters.

# Parameters (typical for modern clinical CT):
- gain: 25 electrons/keV
- electronic_noise: 80-150 electrons (scaled by noise_level)
- 16-bit quantization (LSB based on dynamic range)
"""
function das_clinical(; noise_level::Real = 1.0)
    return DASModel(
        25.0,                    # gain
        100.0 * noise_level,     # electronic noise
        0.0,                     # no quantization in simulation
        0.0,                     # min value
        1e7,                     # max value (saturation)
        0.0                      # no offset
    )
end

# =============================================================================
# DAS Model Application
# =============================================================================

"""
    apply_das_model!(signal, das; seed=nothing)

Apply DAS model to signal (in-place).

This applies the full DAS signal chain:
1. Scale by gain (if signal is in energy units)
2. Add electronic noise
3. Apply quantization (if lsb > 0)
4. Clamp to [min_value, max_value]
5. Add offset

# Arguments
- `signal`: Detector signal array (modified in place)
- `das`: DASModel with parameters

# Keyword Arguments
- `seed`: Random seed for reproducibility (nothing = random)

# Note: This function assumes signal is in arbitrary units.
For proper simulation, signal should be in keV or photon counts.
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

    # Generate noise on CPU, transfer to GPU
    if has_noise
        noise_cpu = randn(Float32, size(signal)) .* Float32(das.electronic_noise_sigma)
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
# Tube Current (mA) Modulation
# =============================================================================

"""
    apply_tube_current!(intensity, mA_per_view)

Apply tube current modulation to intensity data.

In clinical CT, tube current (mA) can vary per view for dose modulation.
This scales the photon fluence proportionally.

# Arguments
- `intensity`: Intensity array [n_cols, n_rows, n_angles] (modified in place)
- `mA_per_view`: Either scalar (constant mA) or vector of length n_angles

# Returns
- Modified intensity array

# Example
```julia
# Constant tube current
apply_tube_current!(intensity, 200.0)  # 200 mA

# Angular tube current modulation (lower dose in lateral views)
n_angles = 360
mA = [200.0 * (1 + 0.3 * cos(2π * i / n_angles)) for i in 1:n_angles]
apply_tube_current!(intensity, mA)
```
"""
function apply_tube_current!(
    intensity::AbstractArray{T, 3},
    mA_per_view::Union{Real, Vector}
) where T <: AbstractFloat

    n_cols, n_rows, n_angles = size(intensity)

    if mA_per_view isa Real
        # Constant tube current - just scale everything
        scale = T(mA_per_view / 100.0)  # Normalize to 100 mA reference
        AK.foreachindex(intensity) do idx
            intensity[idx] *= scale
        end
    else
        # Per-view tube current
        @assert length(mA_per_view) == n_angles "mA_per_view must have length n_angles"

        # Transfer to GPU
        mA_gpu = similar(intensity, T, n_angles)
        copyto!(mA_gpu, T.(mA_per_view ./ 100.0))

        AK.foreachindex(intensity) do idx
            ci = CartesianIndices(intensity)[idx]
            col, row, angle = Tuple(ci)
            intensity[idx] *= mA_gpu[angle]
        end
    end

    return intensity
end

"""
    apply_tube_current(intensity, mA_per_view)

Non-mutating version of apply_tube_current!.
"""
function apply_tube_current(
    intensity::AbstractArray{T, 3},
    mA_per_view::Union{Real, Vector}
) where T <: AbstractFloat
    result = similar(intensity)
    copyto!(result, intensity)
    return apply_tube_current!(result, mA_per_view)
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
        lsb = das.lsb,
        min_value = das.min_value,
        max_value = das.max_value,
        offset = das.offset,
        has_quantization = das.lsb > 0,
        has_noise = das.electronic_noise_sigma > 0
    )
end
