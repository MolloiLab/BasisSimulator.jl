"""
    Forward/DetectorLag.jl

Detector lag (afterglow) modeling for CT simulation.

Detector lag occurs when the scintillator's luminescence persists after
X-ray exposure, causing signal from previous views to contaminate the
current view. This causes:
- View-to-view correlations
- Ring artifacts in reconstructed images
- Reduced temporal resolution

The lag is modeled using a multi-exponential decay:
    h(t) = Σ aᵢ × exp(-t/τᵢ)

where aᵢ are amplitudes and τᵢ are time constants.

GPU-native implementation using AcceleratedKernels.jl.
Each output pixel is computed as a weighted sum of previous frames,
which can be parallelized across all (col, row, angle) positions.
"""

import AcceleratedKernels as AK

# =============================================================================
# Lag Model Types
# =============================================================================

"""
    LagModel

Detector lag (afterglow) model with multi-exponential decay.

# Fields
- `amplitudes`: Decay amplitudes for each exponential component
- `time_constants`: Time constants τ in milliseconds
- `frame_time`: Time between frames (views) in milliseconds
"""
struct LagModel
    amplitudes::Vector{Float64}
    time_constants::Vector{Float64}
    frame_time::Float64
end

# =============================================================================
# Pre-defined Lag Models
# =============================================================================

"""
    lag_none()

No detector lag (ideal detector).
"""
function lag_none()
    return LagModel(Float64[], Float64[], 1.0)
end

"""
    lag_gadox()

Typical Gd₂O₂S (GOS/Gadox) scintillator lag.

GOS detectors have moderate afterglow with ~1-2% lag.
Two-component decay: fast (~1ms) and slow (~10ms).
"""
function lag_gadox(; frame_time::Float64=0.5)
    # Two-component model typical for GOS
    amplitudes = [0.01, 0.005]       # 1% fast + 0.5% slow
    time_constants = [1.0, 10.0]     # 1ms and 10ms decay
    return LagModel(amplitudes, time_constants, frame_time)
end

"""
    lag_csi()

Typical CsI scintillator lag.

CsI detectors have less afterglow than GOS (~0.5% total).
"""
function lag_csi(; frame_time::Float64=0.5)
    amplitudes = [0.003, 0.002]      # 0.3% fast + 0.2% slow
    time_constants = [0.5, 5.0]      # Faster decay than GOS
    return LagModel(amplitudes, time_constants, frame_time)
end

"""
    lag_high()

High lag model for older or degraded detectors.

~5% total lag with long time constants.
"""
function lag_high(; frame_time::Float64=0.5)
    amplitudes = [0.03, 0.015, 0.005]
    time_constants = [1.0, 10.0, 50.0]
    return LagModel(amplitudes, time_constants, frame_time)
end

"""
    lag_custom(amplitudes, time_constants; frame_time=0.5)

Create custom lag model with specified parameters.

# Arguments
- `amplitudes`: Vector of decay amplitudes (as fractions)
- `time_constants`: Vector of time constants in milliseconds
- `frame_time`: Time between frames in milliseconds

# Example
```julia
model = lag_custom([0.02, 0.01], [2.0, 15.0]; frame_time=0.5)
```
"""
function lag_custom(
    amplitudes::Vector{Float64},
    time_constants::Vector{Float64};
    frame_time::Float64=0.5
)
    @assert length(amplitudes) == length(time_constants) "Must have same number of amplitudes and time constants"
    @assert all(amplitudes .>= 0) "Amplitudes must be non-negative"
    @assert all(time_constants .> 0) "Time constants must be positive"
    @assert frame_time > 0 "Frame time must be positive"

    return LagModel(amplitudes, time_constants, frame_time)
end

# =============================================================================
# Lag Computation
# =============================================================================

"""
    compute_lag_coefficients(model::LagModel, n_frames::Int) -> Vector{Float64}

Compute lag coefficients for each frame offset.

Returns a vector where coef[k] is the fraction of frame n-k that
contributes to frame n due to lag.

# Arguments
- `model::LagModel`: Lag model
- `n_frames`: Number of frames to compute coefficients for
"""
function compute_lag_coefficients(model::LagModel, n_frames::Int)
    if isempty(model.amplitudes)
        return Float64[1.0]  # No lag: only current frame contributes
    end

    coeffs = zeros(Float64, n_frames)
    total_lag = sum(model.amplitudes)

    for k in 0:(n_frames-1)
        t = k * model.frame_time  # Time delay for frame k frames ago

        if k == 0
            # Current frame: primary signal (1 - total lag)
            coeffs[k+1] = 1.0 - total_lag
        else
            # Previous frames: sum of exponential decays
            for (a, τ) in zip(model.amplitudes, model.time_constants)
                coeffs[k+1] += a * exp(-t / τ)
            end
        end
    end

    # Normalize to sum to 1.0 to preserve total signal
    # This accounts for truncated exponential tail
    coeffs ./= sum(coeffs)

    return coeffs
end

"""
    apply_lag!(sinogram, model::LagModel; n_history::Int=20) -> sinogram

Apply detector lag to sinogram (in-place, GPU-native).

This simulates the temporal persistence of signal between views.
The output at view n is a weighted sum of current and previous views.

Each output pixel (col, row, angle) is computed independently as a
weighted sum over previous angles, enabling full GPU parallelization.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles]
- `model::LagModel`: Lag model
- `n_history`: Number of previous frames to consider (default: 20)

# Returns
Modified sinogram with lag effects.

# Note
Lag is applied in the intensity domain for physical correctness.
"""
function apply_lag!(
    sinogram::AbstractArray{T,3},
    model::LagModel;
    n_history::Int=20
) where T
    # Skip if no lag
    if isempty(model.amplitudes)
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    n_angles = size(sinogram, 3)

    # Compute lag coefficients on CPU
    n_frames = min(n_history, n_angles)
    coeffs_cpu = T.(compute_lag_coefficients(model, n_frames))

    # Transfer coefficients to GPU
    coeffs = similar(sinogram, n_frames)
    copyto!(coeffs, coeffs_cpu)

    # Pre-compute intensity domain (GPU-native)
    intensity = similar(sinogram)
    AK.foreachindex(sinogram) do idx
        intensity[idx] = exp(-sinogram[idx])
    end

    # Output buffer
    output = similar(sinogram)

    # GPU-native: compute each output pixel as weighted sum of previous frames
    # Each (col, row, angle) can be computed independently
    AK.foreachindex(sinogram) do idx
        ci = CartesianIndices(sinogram)[idx]
        col, row, angle = Tuple(ci)

        # Weighted sum over previous frames
        weighted_sum = zero(T)
        for k in 0:(n_frames-1)
            prev_angle = angle - k
            if prev_angle >= 1
                weighted_sum += coeffs[k+1] * intensity[col, row, prev_angle]
            else
                # For early frames, use first available frame
                weighted_sum += coeffs[k+1] * intensity[col, row, 1]
            end
        end

        # Ensure positive and convert back to projection domain
        output[idx] = -log(max(weighted_sum, T(1e-10)))
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates (for backward compatibility)
function apply_lag(
    sinogram::AbstractArray{T,3},
    model::LagModel;
    n_history::Int=20
) where T
    result = copy(sinogram)
    return apply_lag!(result, model; n_history=n_history)
end

"""
    apply_lag_recursive!(sinogram, model::LagModel) -> sinogram

Apply detector lag using recursive IIR filter formulation (in-place, GPU-native).

This uses a pixel-parallel approach where each (col, row) position is processed
independently through all angles. Within each pixel, angles are processed
sequentially to maintain the IIR state, but different pixels run in parallel.

For each exponential component:
    state[n] = decay × state[n-1] + amplitude × input[n]
    output[n] = (1 - total_amp) × input[n] + Σ state[n]

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles]
- `model::LagModel`: Lag model

# Returns
Modified sinogram with lag effects.

# Note
For most cases, apply_lag! (weighted sum approach) is preferred as it
fully parallelizes over all elements. This recursive version maintains
compatibility with the IIR filter formulation.
"""
function apply_lag_recursive!(
    sinogram::AbstractArray{T,3},
    model::LagModel
) where T
    if isempty(model.amplitudes)
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    n_angles = size(sinogram, 3)
    n_components = length(model.amplitudes)

    # Pre-compute decay factors and amplitudes on CPU
    decay_factors_cpu = T.([exp(-model.frame_time / τ) for τ in model.time_constants])
    amplitudes_cpu = T.(model.amplitudes)
    total_amp = sum(amplitudes_cpu)

    # Transfer to GPU
    decay_factors = similar(sinogram, n_components)
    amplitudes_arr = similar(sinogram, n_components)
    copyto!(decay_factors, decay_factors_cpu)
    copyto!(amplitudes_arr, amplitudes_cpu)

    # Pre-compute intensity domain
    intensity = similar(sinogram)
    AK.foreachindex(sinogram) do idx
        intensity[idx] = exp(-sinogram[idx])
    end

    # Output buffer
    output = similar(sinogram)

    # For recursive version, we need to process angles sequentially per pixel
    # But we can parallelize across pixels
    # Create a 2D slice for indexing (col, row) pairs
    pixel_indices = similar(sinogram, n_cols, n_rows)

    # GPU-native: parallelize over pixels, sequential over angles within each pixel
    AK.foreachindex(pixel_indices) do idx
        ci = CartesianIndices(pixel_indices)[idx]
        col, row = Tuple(ci)

        # State for each exponential component (local to this pixel)
        # Using a simple approach with fixed max components
        state1 = zero(T)
        state2 = zero(T)
        state3 = zero(T)

        for angle in 1:n_angles
            current = intensity[col, row, angle]

            # Update states and compute lag contribution
            lag_contribution = zero(T)

            if n_components >= 1
                state1 = decay_factors[1] * state1 + amplitudes_arr[1] * current
                lag_contribution += state1
            end
            if n_components >= 2
                state2 = decay_factors[2] * state2 + amplitudes_arr[2] * current
                lag_contribution += state2
            end
            if n_components >= 3
                state3 = decay_factors[3] * state3 + amplitudes_arr[3] * current
                lag_contribution += state3
            end

            # Output: primary + lag
            result = (T(1) - T(total_amp)) * current + lag_contribution
            output[col, row, angle] = -log(max(result, T(1e-10)))
        end
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrapper that allocates
function apply_lag_recursive(
    sinogram::AbstractArray{T,3},
    model::LagModel
) where T
    result = copy(sinogram)
    return apply_lag_recursive!(result, model)
end

"""
    get_lag_info(model::LagModel) -> NamedTuple

Get diagnostic information about lag model.
"""
function get_lag_info(model::LagModel)
    if isempty(model.amplitudes)
        return (
            n_components = 0,
            total_lag_fraction = 0.0,
            amplitudes = Float64[],
            time_constants_ms = Float64[],
            frame_time_ms = model.frame_time
        )
    end

    return (
        n_components = length(model.amplitudes),
        total_lag_fraction = sum(model.amplitudes),
        amplitudes = model.amplitudes,
        time_constants_ms = model.time_constants,
        frame_time_ms = model.frame_time
    )
end

"""
    compute_lag_impulse_response(model::LagModel, n_frames::Int) -> Vector{Float64}

Compute the impulse response of the lag model.

Returns the response to a single-frame impulse over n_frames.
Useful for visualization and analysis.
"""
function compute_lag_impulse_response(model::LagModel, n_frames::Int)
    if isempty(model.amplitudes)
        response = zeros(Float64, n_frames)
        response[1] = 1.0
        return response
    end

    response = zeros(Float64, n_frames)

    for k in 0:(n_frames-1)
        t = k * model.frame_time

        if k == 0
            # Primary response
            response[k+1] = 1.0 - sum(model.amplitudes)
        end

        # Add exponential decay contributions
        for (a, τ) in zip(model.amplitudes, model.time_constants)
            response[k+1] += a * exp(-t / τ)
        end
    end

    return response
end

# =============================================================================
# Exports
# =============================================================================

export LagModel
export lag_none, lag_gadox, lag_csi, lag_high, lag_custom
export compute_lag_coefficients
export apply_lag!, apply_lag, apply_lag_recursive!, apply_lag_recursive
export get_lag_info, compute_lag_impulse_response
