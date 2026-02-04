"""
    Forward/DetectorLag.jl

Detector lag (afterglow) modeling for CT simulation.

Detector lag occurs when the scintillator's luminescence persists after
X-ray exposure, causing signal from previous views to contaminate the
current view. This causes:
- View-to-view correlations (ghosting artifacts)
- Ring artifacts in reconstructed images
- Reduced temporal resolution

# Mathematical Model

The lag is modeled using a multi-exponential decay (CatSim-exact formulation):

    h(t) = Σ αᵢ × exp(-t/τᵢ)

where αᵢ are amplitudes (fractions) and τᵢ are time constants (ms).

## CatSim Formula (Detection_Lag.py)

CatSim uses midpoint sampling for improved integration accuracy:

    invintegral = α₁(1 - e^(-dt/2τ₁)) + α₂(1 - e^(-dt/2τ₂)) + (1 - α₁ - α₂)
    out = invintegral × current + α₁(1 - e^(-dt/τ₁)) × mem₁ + α₂(1 - e^(-dt/τ₂)) × mem₂
    mem₁' = mem₁ × e^(-dt/τ₁) + current × e^(-dt/2τ₁)
    mem₂' = mem₂ × e^(-dt/τ₂) + current × e^(-dt/2τ₂)

where:
- dt = frame time (ms) = 1000 × rotation_time / views_per_rotation
- invintegral accounts for the integral of decay during the current frame
- mem₁, mem₂ are state variables carrying accumulated afterglow

# Typical Values

| Detector  | τ_fast (ms) | τ_slow (ms) | α_fast | α_slow | Total Lag |
|-----------|-------------|-------------|--------|--------|-----------|
| GOS (Gd₂O₂S) | 0.5-2.0  | 5-15       | 1-2%   | 0.5-1% | 1.5-3%    |
| CsI       | 0.3-1.0     | 3-8        | 0.3%   | 0.2%   | 0.5%      |
| CdTe/CZT  | ~0.1        | ~1         | ~0.1%  | ~0.1%  | ~0.2%     |

# CatSim Compatibility

This implementation provides `apply_lag_catsim` which matches CatSim Detection_Lag.py
exactly, including midpoint sampling and air scan initialization options.

# GPU Implementation

GPU-native using AcceleratedKernels.jl with two strategies:
1. `apply_lag!` - Fully parallel weighted-sum (approximation)
2. `apply_lag_catsim` - CatSim-exact recursive formulation (pixel-parallel)

# References

1. CatSim Detection_Lag.py - Reference implementation
2. Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and
   Recent Advances" Ch. 4 - Detector physics
3. Siewerdsen JH, Jaffray DA. "Optimization of x-ray imaging geometry"
   Med Phys. 1999;26(8):1624-1633. doi:10.1118/1.598657
4. Zhao W, et al. "Investigation of the charge trapping and afterglow"
   Med Phys. 2001;28(2):211-218. doi:10.1118/1.1344222
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
# CatSim-Exact Implementation
# =============================================================================

"""
    apply_lag_catsim(sinogram, model::LagModel; is_air_scan=false) -> Array

Apply detector lag using the exact CatSim Detection_Lag.py formula.

This function implements CatSim's lag model with midpoint sampling for
improved integration accuracy. It processes views sequentially while
parallelizing across detector pixels.

# CatSim Formula

For a two-component model with (τ₁, α₁) and (τ₂, α₂):

    invintegral = α₁(1 - e^(-dt/2τ₁)) + α₂(1 - e^(-dt/2τ₂)) + (1 - α₁ - α₂)
    out = invintegral × I + α₁(1 - e^(-dt/τ₁)) × mem₁ + α₂(1 - e^(-dt/τ₂)) × mem₂

State update:
    mem₁' = mem₁ × e^(-dt/τ₁) + I × e^(-dt/2τ₁)
    mem₂' = mem₂ × e^(-dt/τ₂) + I × e^(-dt/2τ₂)

# Arguments
- `sinogram`: Input sinogram in intensity domain [n_cols, n_rows, n_angles]
- `model::LagModel`: Lag model with amplitudes and time constants
- `is_air_scan::Bool`: If true, use steady-state initialization (CatSim air scan mode)

# Returns
Output sinogram with lag effects applied.

# Air Scan Initialization

When `is_air_scan=true`, state variables are initialized to steady-state values:
    mem₁ = I × e^(-dt/2τ₁) / (1 - e^(-dt/τ₁))
    mem₂ = I × e^(-dt/2τ₂) / (1 - e^(-dt/τ₂))

This produces constant output for constant input (no transient).

# Example
```julia
model = lag_gadox(frame_time=0.5)  # GOS detector, 0.5ms frame time
output = apply_lag_catsim(intensity_sinogram, model)
```

# Note
Input must be in intensity domain (not log-transformed projections).
If you have projection data p, convert via: I = exp(-p)
"""
function apply_lag_catsim(
    sinogram::AbstractArray{T,3},
    model::LagModel;
    is_air_scan::Bool = false
) where T
    # Skip if no lag
    if isempty(model.amplitudes)
        return copy(sinogram)
    end

    n_cols, n_rows, n_angles = size(sinogram)
    n_components = length(model.amplitudes)
    dt = model.frame_time

    # Extract parameters (support up to 3 components for GPU kernel simplicity)
    # CatSim uses 2 components, but we support up to 3
    taus = zeros(3)
    alphas = zeros(3)
    for i in 1:min(n_components, 3)
        taus[i] = model.time_constants[i]
        alphas[i] = model.amplitudes[i]
    end

    tau1, tau2, tau3 = taus
    alpha1, alpha2, alpha3 = alphas
    total_alpha = sum(alphas)

    # CatSim invintegral formula (midpoint sampling)
    unaccounted = 1.0 - total_alpha
    invintegral = unaccounted
    if alpha1 > 0 && tau1 > 0
        invintegral += alpha1 * (1.0 - exp(-dt / 2 / tau1))
    end
    if alpha2 > 0 && tau2 > 0
        invintegral += alpha2 * (1.0 - exp(-dt / 2 / tau2))
    end
    if alpha3 > 0 && tau3 > 0
        invintegral += alpha3 * (1.0 - exp(-dt / 2 / tau3))
    end

    # Pre-compute decay factors
    decay1 = tau1 > 0 ? exp(-dt / tau1) : 0.0
    decay2 = tau2 > 0 ? exp(-dt / tau2) : 0.0
    decay3 = tau3 > 0 ? exp(-dt / tau3) : 0.0

    # Midpoint decay factors (for state update)
    mid_decay1 = tau1 > 0 ? exp(-dt / 2 / tau1) : 0.0
    mid_decay2 = tau2 > 0 ? exp(-dt / 2 / tau2) : 0.0
    mid_decay3 = tau3 > 0 ? exp(-dt / 2 / tau3) : 0.0

    # Contribution factors from memory (CatSim: alpha * (1 - exp(-dt/tau)))
    contrib1 = alpha1 * (1.0 - decay1)
    contrib2 = alpha2 * (1.0 - decay2)
    contrib3 = alpha3 * (1.0 - decay3)

    # Output buffer
    output = similar(sinogram)

    # Process each pixel independently (parallelizable)
    # Within each pixel, process angles sequentially (maintains IIR state)
    for col in 1:n_cols
        for row in 1:n_rows
            # Initialize state variables
            mem1 = zero(T)
            mem2 = zero(T)
            mem3 = zero(T)

            for angle in 1:n_angles
                current = sinogram[col, row, angle]

                # First view initialization (CatSim lines 14-21)
                if angle == 1
                    if is_air_scan
                        # Steady-state initialization
                        if tau1 > 0 && decay1 < 1.0
                            mem1 = current * mid_decay1 / (1.0 - decay1)
                        end
                        if tau2 > 0 && decay2 < 1.0
                            mem2 = current * mid_decay2 / (1.0 - decay2)
                        end
                        if tau3 > 0 && decay3 < 1.0
                            mem3 = current * mid_decay3 / (1.0 - decay3)
                        end
                    else
                        # Phantom scan: zero initialization
                        mem1 = zero(T)
                        mem2 = zero(T)
                        mem3 = zero(T)
                    end
                end

                # Compute output (CatSim line 24)
                out = T(invintegral) * current +
                      T(contrib1) * mem1 +
                      T(contrib2) * mem2 +
                      T(contrib3) * mem3

                output[col, row, angle] = out

                # Update state (CatSim lines 25-26)
                mem1 = mem1 * T(decay1) + current * T(mid_decay1)
                mem2 = mem2 * T(decay2) + current * T(mid_decay2)
                mem3 = mem3 * T(decay3) + current * T(mid_decay3)
            end
        end
    end

    return output
end

"""
    apply_lag_catsim!(sinogram, model::LagModel; is_air_scan=false) -> sinogram

In-place version of apply_lag_catsim.
"""
function apply_lag_catsim!(
    sinogram::AbstractArray{T,3},
    model::LagModel;
    is_air_scan::Bool = false
) where T
    result = apply_lag_catsim(sinogram, model; is_air_scan=is_air_scan)
    copyto!(sinogram, result)
    return sinogram
end

"""
    compute_catsim_lag_parameters(model::LagModel) -> NamedTuple

Compute the CatSim-style lag parameters for verification.

Returns a named tuple with:
- `invintegral`: The scaling factor for current frame
- `decay_factors`: Vector of exp(-dt/τᵢ) for each component
- `contribution_factors`: Vector of αᵢ(1-exp(-dt/τᵢ)) for each component
- `midpoint_factors`: Vector of exp(-dt/2τᵢ) for each component
"""
function compute_catsim_lag_parameters(model::LagModel)
    dt = model.frame_time
    n = length(model.amplitudes)

    if n == 0
        return (
            invintegral = 1.0,
            decay_factors = Float64[],
            contribution_factors = Float64[],
            midpoint_factors = Float64[],
            total_lag = 0.0
        )
    end

    decay_factors = [exp(-dt / τ) for τ in model.time_constants]
    midpoint_factors = [exp(-dt / 2 / τ) for τ in model.time_constants]
    contribution_factors = [α * (1.0 - d) for (α, d) in zip(model.amplitudes, decay_factors)]

    total_lag = sum(model.amplitudes)
    unaccounted = 1.0 - total_lag

    invintegral = unaccounted
    for (α, m) in zip(model.amplitudes, midpoint_factors)
        invintegral += α * (1.0 - m)
    end

    return (
        invintegral = invintegral,
        decay_factors = decay_factors,
        contribution_factors = contribution_factors,
        midpoint_factors = midpoint_factors,
        total_lag = total_lag
    )
end

# =============================================================================
# Exports
# =============================================================================

export LagModel
export lag_none, lag_gadox, lag_csi, lag_high, lag_custom
export compute_lag_coefficients
export apply_lag!, apply_lag, apply_lag_recursive!, apply_lag_recursive
export apply_lag_catsim, apply_lag_catsim!
export get_lag_info, compute_lag_impulse_response
export compute_catsim_lag_parameters
