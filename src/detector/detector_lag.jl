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

Lag is a multi-exponential decay:

    h(t) = Σ αᵢ × exp(-t/τᵢ)

`apply_lag!` applies it as a fully-parallel weighted sum over the previous
N views, where the N weights are pre-computed by `compute_lag_coefficients`
from the αᵢ and τᵢ.

# Reference
- Hsieh J. "Computed Tomography: Principles, Design, Artifacts, and
  Recent Advances" Ch. 4 — Detector physics
"""

import AcceleratedKernels as AK

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

"""
    lag_gadox()

Typical Gd₂O₂S (GOS/Gadox) scintillator lag.

GOS detectors have moderate afterglow with ~1-2% lag.
Two-component decay: fast (~1ms) and slow (~10ms).
"""
function lag_gadox(; frame_time::Float64=0.5)
    amplitudes = [0.01, 0.005]       # 1% fast + 0.5% slow
    time_constants = [1.0, 10.0]     # 1ms and 10ms decay
    return LagModel(amplitudes, time_constants, frame_time)
end

"""
    compute_lag_coefficients(model::LagModel, n_frames::Int) -> Vector{Float64}

Compute lag coefficients for each frame offset.

Returns a vector where `coef[k]` is the fraction of frame `n-k` that
contributes to frame `n` due to lag.
"""
function compute_lag_coefficients(model::LagModel, n_frames::Int)
    if isempty(model.amplitudes)
        return Float64[1.0]
    end

    coeffs = zeros(Float64, n_frames)
    total_lag = sum(model.amplitudes)

    for k in 0:(n_frames-1)
        t = k * model.frame_time

        if k == 0
            coeffs[k+1] = 1.0 - total_lag
        else
            for (a, τ) in zip(model.amplitudes, model.time_constants)
                coeffs[k+1] += a * exp(-t / τ)
            end
        end
    end

    coeffs ./= sum(coeffs)
    return coeffs
end

"""
    apply_lag!(sinogram, model::LagModel; n_history::Int=20) -> sinogram

Apply detector lag to sinogram (in-place, GPU-native).

Each output pixel `(col, row, angle)` is a weighted sum over the previous
`n_history` angles, with the weights from `compute_lag_coefficients`.
Operates in intensity domain (exp / log conversions handled internally).
"""
function apply_lag!(
    sinogram::AbstractArray{T,3},
    model::LagModel;
    n_history::Int=20,
    ws_output=nothing, ws_intensity=nothing, ws_coeffs=nothing
) where T
    if isempty(model.amplitudes)
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)
    n_angles = size(sinogram, 3)

    n_frames = min(n_history, n_angles)
    if ws_coeffs !== nothing
        coeffs = ws_coeffs
    else
        coeffs_cpu = T.(compute_lag_coefficients(model, n_frames))
        coeffs = similar(sinogram, n_frames)
        copyto!(coeffs, coeffs_cpu)
    end

    intensity = ws_intensity !== nothing ? ws_intensity : similar(sinogram)
    let intensity = intensity
        AK.foreachindex(sinogram) do idx
            intensity[idx] = exp(-sinogram[idx])
        end
    end

    output = ws_output !== nothing ? ws_output : similar(sinogram)

    let coeffs = coeffs, intensity = intensity, output = output, n_frames = n_frames, n_cols = n_cols, n_rows = n_rows
        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % Int32(n_cols)) + Int32(1)
            idx_0 = idx_0 ÷ Int32(n_cols)
            row = (idx_0 % Int32(n_rows)) + Int32(1)
            angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

            weighted_sum = zero(T)
            for k in 0:(n_frames-1)
                prev_angle = angle - k
                if prev_angle >= 1
                    weighted_sum += coeffs[k+1] * intensity[col, row, prev_angle]
                else
                    weighted_sum += coeffs[k+1] * intensity[col, row, 1]
                end
            end

            output[idx] = -log(max(weighted_sum, T(1e-10)))
        end
    end

    copyto!(sinogram, output)
    return sinogram
end

# =============================================================================
# Exports
# =============================================================================

export LagModel
export lag_gadox
export compute_lag_coefficients
export apply_lag!
