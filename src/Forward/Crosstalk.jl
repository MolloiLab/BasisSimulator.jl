"""
    Forward/Crosstalk.jl

Detector crosstalk modeling for CT simulation.

Crosstalk occurs when signal from one detector cell "bleeds" into
neighboring cells due to optical or electrical coupling. This:
- Reduces spatial resolution (affects MTF)
- Correlates noise between pixels
- Changes noise texture in reconstructed images

The crosstalk is modeled as a convolution with a small kernel where
a fraction of the signal is shared with nearest neighbors.

Note: Implementation uses FFT convolution for Reactant/XLA compatibility.
"""

using FFTW

# =============================================================================
# Crosstalk Types
# =============================================================================

"""
    CrosstalkModel

Detector crosstalk specification.

# Fields
- `primary_fraction`: Fraction of signal retained in primary pixel (0-1)
- `neighbor_fraction`: Fraction shared with each direct neighbor (4 neighbors)
- `diagonal_fraction`: Fraction shared with each diagonal neighbor (4 neighbors)
- `type`: Crosstalk type (:optical, :electrical, :combined)
"""
struct CrosstalkModel
    primary_fraction::Float64
    neighbor_fraction::Float64
    diagonal_fraction::Float64
    type::Symbol
end

# =============================================================================
# Pre-defined Crosstalk Models
# =============================================================================

"""
    crosstalk_none()

No crosstalk (ideal detector).
"""
function crosstalk_none()
    return CrosstalkModel(1.0, 0.0, 0.0, :none)
end

"""
    crosstalk_low()

Low crosstalk typical of high-quality scintillator detectors.

~5% of signal shared with neighbors.
"""
function crosstalk_low()
    # Total crosstalk ~5%: 1% to each of 4 direct neighbors, 0.25% to diagonals
    return CrosstalkModel(0.95, 0.01, 0.0025, :optical)
end

"""
    crosstalk_medium()

Medium crosstalk typical of standard CT detectors.

~10% of signal shared with neighbors.
"""
function crosstalk_medium()
    # Total crosstalk ~10%: 2% to each direct neighbor, 0.5% to diagonals
    return CrosstalkModel(0.90, 0.02, 0.005, :optical)
end

"""
    crosstalk_high()

High crosstalk for older or lower-quality detectors.

~15% of signal shared with neighbors.
"""
function crosstalk_high()
    # Total crosstalk ~15%: 3% to each direct neighbor, 0.75% to diagonals
    return CrosstalkModel(0.85, 0.03, 0.0075, :combined)
end

"""
    crosstalk_custom(total_fraction; neighbor_ratio=0.8)

Create custom crosstalk model with specified total fraction.

# Arguments
- `total_fraction`: Total fraction of signal shared with neighbors (0-1)
- `neighbor_ratio`: Ratio of direct neighbor sharing vs diagonal (0-1)

# Example
```julia
model = crosstalk_custom(0.12)  # 12% total crosstalk
```
"""
function crosstalk_custom(total_fraction::Float64; neighbor_ratio::Float64=0.8)
    @assert 0 <= total_fraction <= 1 "total_fraction must be in [0, 1]"
    @assert 0 <= neighbor_ratio <= 1 "neighbor_ratio must be in [0, 1]"

    primary = 1.0 - total_fraction

    # Distribute among 4 direct + 4 diagonal neighbors
    direct_total = total_fraction * neighbor_ratio
    diagonal_total = total_fraction * (1 - neighbor_ratio)

    neighbor = direct_total / 4
    diagonal = diagonal_total / 4

    return CrosstalkModel(primary, neighbor, diagonal, :custom)
end

# =============================================================================
# Crosstalk Application
# =============================================================================

"""
    create_crosstalk_kernel(model::CrosstalkModel, n_cols::Int, n_rows::Int)

Create 2D crosstalk convolution kernel.

The kernel is a 3×3 pattern embedded in a full-size array for FFT convolution.
"""
function create_crosstalk_kernel(model::CrosstalkModel, n_cols::Int, n_rows::Int)
    kernel = zeros(Float64, n_cols, n_rows)

    # Center pixel (primary signal)
    cx = 1  # FFT kernel centered at (1,1)
    cy = 1

    kernel[cx, cy] = model.primary_fraction

    # Direct neighbors (up, down, left, right)
    if model.neighbor_fraction > 0
        # Handle wrap-around for FFT
        kernel[cx, mod1(cy+1, n_rows)] = model.neighbor_fraction  # up
        kernel[cx, mod1(cy-1, n_rows)] = model.neighbor_fraction  # down
        kernel[mod1(cx+1, n_cols), cy] = model.neighbor_fraction  # right
        kernel[mod1(cx-1, n_cols), cy] = model.neighbor_fraction  # left
    end

    # Diagonal neighbors
    if model.diagonal_fraction > 0
        kernel[mod1(cx+1, n_cols), mod1(cy+1, n_rows)] = model.diagonal_fraction
        kernel[mod1(cx+1, n_cols), mod1(cy-1, n_rows)] = model.diagonal_fraction
        kernel[mod1(cx-1, n_cols), mod1(cy+1, n_rows)] = model.diagonal_fraction
        kernel[mod1(cx-1, n_cols), mod1(cy-1, n_rows)] = model.diagonal_fraction
    end

    # Normalize to preserve total signal
    total = sum(kernel)
    if total > 0
        kernel ./= total
    end

    return kernel
end

"""
    apply_crosstalk(sinogram, model::CrosstalkModel) -> Array

Apply detector crosstalk to sinogram using convolution.

This simulates signal bleeding between detector pixels in the intensity
domain, then converts back to projection domain.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles] (projection domain)
- `model::CrosstalkModel`: Crosstalk model specification

# Returns
Sinogram with crosstalk effects.

# Note
For physically accurate simulation, crosstalk should be applied in the
intensity domain (before log transform). This function handles the conversion.
"""
function apply_crosstalk(
    sinogram::AbstractArray{T,3},
    model::CrosstalkModel
) where T
    # Skip if no crosstalk
    if model.type == :none || model.primary_fraction >= 1.0
        return sinogram
    end

    n_cols, n_rows, n_angles = size(sinogram)

    # Create crosstalk kernel
    kernel = create_crosstalk_kernel(model, n_cols, n_rows)
    kernel_fft = fft(kernel)

    result = similar(sinogram)

    for angle in 1:n_angles
        proj = sinogram[:, :, angle]

        # Convert to intensity domain
        intensity = exp.(-proj)

        # Apply crosstalk convolution
        intensity_fft = fft(intensity)
        intensity_ct = real(ifft(intensity_fft .* kernel_fft))

        # Ensure positive values
        intensity_ct = max.(intensity_ct, T(1e-10))

        # Convert back to projection domain
        result[:, :, angle] = T.(-log.(intensity_ct))
    end

    return result
end

"""
    apply_crosstalk_intensity(intensity, model::CrosstalkModel) -> Array

Apply detector crosstalk directly to intensity-domain data.

Use this when working in intensity domain (before log transform).

# Arguments
- `intensity`: Input intensity [n_cols, n_rows, n_angles]
- `model::CrosstalkModel`: Crosstalk model specification

# Returns
Intensity with crosstalk effects.
"""
function apply_crosstalk_intensity(
    intensity::AbstractArray{T,3},
    model::CrosstalkModel
) where T
    if model.type == :none || model.primary_fraction >= 1.0
        return intensity
    end

    n_cols, n_rows, n_angles = size(intensity)

    kernel = create_crosstalk_kernel(model, n_cols, n_rows)
    kernel_fft = fft(kernel)

    result = similar(intensity)

    for angle in 1:n_angles
        frame = intensity[:, :, angle]
        frame_fft = fft(frame)
        frame_ct = real(ifft(frame_fft .* kernel_fft))
        result[:, :, angle] = T.(max.(frame_ct, T(0)))
    end

    return result
end

"""
    get_crosstalk_mtf_degradation(model::CrosstalkModel) -> Float64

Estimate MTF degradation factor from crosstalk.

Returns approximate factor by which Nyquist MTF is reduced.
"""
function get_crosstalk_mtf_degradation(model::CrosstalkModel)
    # Simple estimate: MTF at Nyquist ≈ primary_fraction - 2*neighbor_fraction
    # (since Nyquist is where adjacent pixels are out of phase)
    mtf_nyquist = model.primary_fraction - 2 * model.neighbor_fraction
    return max(mtf_nyquist, 0.0)
end

"""
    get_crosstalk_info(model::CrosstalkModel) -> NamedTuple

Get diagnostic information about crosstalk model.
"""
function get_crosstalk_info(model::CrosstalkModel)
    total_shared = 1.0 - model.primary_fraction
    direct_total = 4 * model.neighbor_fraction
    diagonal_total = 4 * model.diagonal_fraction

    return (
        type = model.type,
        primary_fraction = model.primary_fraction,
        total_crosstalk_fraction = total_shared,
        direct_neighbor_total = direct_total,
        diagonal_neighbor_total = diagonal_total,
        estimated_mtf_degradation = get_crosstalk_mtf_degradation(model)
    )
end

# =============================================================================
# Optical Crosstalk (CatSim-style separable kernel)
# =============================================================================

"""
    OpticalCrosstalkModel

Optical crosstalk model using separable convolution kernels.

This models light spreading in the scintillator layer, which is different from
electronic crosstalk (signal coupling in readout electronics).

CatSim uses separable kernels:
- Row kernel: [α, 1-2α, α]
- Column kernel: [β, 1-2β, β]
- 2D kernel: outer product of row and column kernels

# Fields
- `row_coeff`: Row crosstalk coefficient α (typical: 0.04-0.05)
- `col_coeff`: Column crosstalk coefficient β (typical: 0.04-0.05)
"""
struct OpticalCrosstalkModel
    row_coeff::Float64
    col_coeff::Float64
end

"""
    optical_crosstalk_none()

No optical crosstalk.
"""
optical_crosstalk_none() = OpticalCrosstalkModel(0.0, 0.0)

"""
    optical_crosstalk_typical()

Typical optical crosstalk for GOS scintillator detector.

Values from CatSim: row=0.045, col=0.040
"""
optical_crosstalk_typical() = OpticalCrosstalkModel(0.045, 0.040)

"""
    optical_crosstalk_low()

Low optical crosstalk (high-quality CsI detector).
"""
optical_crosstalk_low() = OpticalCrosstalkModel(0.02, 0.02)

"""
    optical_crosstalk_high()

High optical crosstalk (thick scintillator, large pixels).
"""
optical_crosstalk_high() = OpticalCrosstalkModel(0.08, 0.08)

"""
    create_optical_crosstalk_kernel(model::OpticalCrosstalkModel)

Create 3x3 separable optical crosstalk kernel.

Returns the 2D kernel as outer product of row and column 1D kernels.
"""
function create_optical_crosstalk_kernel(model::OpticalCrosstalkModel)
    α = model.row_coeff
    β = model.col_coeff

    # 1D kernels
    row_kernel = [α, 1 - 2α, α]
    col_kernel = [β, 1 - 2β, β]

    # 2D kernel as outer product
    kernel_2d = col_kernel * row_kernel'

    return kernel_2d
end

"""
    apply_optical_crosstalk(sinogram, model::OpticalCrosstalkModel) -> Array

Apply optical crosstalk using separable convolution.

# Arguments
- `sinogram`: Input sinogram [n_cols, n_rows, n_angles] (projection domain)
- `model::OpticalCrosstalkModel`: Optical crosstalk coefficients

# Returns
Sinogram with optical crosstalk effects.
"""
function apply_optical_crosstalk(
    sinogram::AbstractArray{T,3},
    model::OpticalCrosstalkModel
) where T
    # Skip if no crosstalk
    if model.row_coeff ≈ 0 && model.col_coeff ≈ 0
        return sinogram
    end

    n_cols, n_rows, n_angles = size(sinogram)

    # Create 3x3 kernel
    kernel_3x3 = create_optical_crosstalk_kernel(model)

    # Embed in full-size array for FFT (centered at 1,1)
    kernel = zeros(Float64, n_cols, n_rows)
    for di in -1:1
        for dj in -1:1
            ci = mod1(1 + di, n_cols)
            cj = mod1(1 + dj, n_rows)
            kernel[ci, cj] = kernel_3x3[di+2, dj+2]
        end
    end
    kernel_fft = fft(kernel)

    result = similar(sinogram)

    for angle in 1:n_angles
        proj = sinogram[:, :, angle]

        # Convert to intensity domain
        intensity = exp.(-proj)

        # Apply optical crosstalk convolution
        intensity_fft = fft(intensity)
        intensity_xt = real(ifft(intensity_fft .* kernel_fft))

        # Ensure positive values
        intensity_xt = max.(intensity_xt, T(1e-10))

        # Convert back to projection domain
        result[:, :, angle] = T.(-log.(intensity_xt))
    end

    return result
end

"""
    apply_optical_crosstalk_intensity(intensity, model::OpticalCrosstalkModel) -> Array

Apply optical crosstalk directly to intensity-domain data.
"""
function apply_optical_crosstalk_intensity(
    intensity::AbstractArray{T,3},
    model::OpticalCrosstalkModel
) where T
    if model.row_coeff ≈ 0 && model.col_coeff ≈ 0
        return intensity
    end

    n_cols, n_rows, n_angles = size(intensity)

    kernel_3x3 = create_optical_crosstalk_kernel(model)
    kernel = zeros(Float64, n_cols, n_rows)
    for di in -1:1
        for dj in -1:1
            ci = mod1(1 + di, n_cols)
            cj = mod1(1 + dj, n_rows)
            kernel[ci, cj] = kernel_3x3[di+2, dj+2]
        end
    end
    kernel_fft = fft(kernel)

    result = similar(intensity)

    for angle in 1:n_angles
        frame = intensity[:, :, angle]
        frame_fft = fft(frame)
        frame_xt = real(ifft(frame_fft .* kernel_fft))
        result[:, :, angle] = T.(max.(frame_xt, T(0)))
    end

    return result
end

# =============================================================================
# Exports
# =============================================================================

export CrosstalkModel
export crosstalk_none, crosstalk_low, crosstalk_medium, crosstalk_high, crosstalk_custom
export apply_crosstalk, apply_crosstalk_intensity
export get_crosstalk_mtf_degradation, get_crosstalk_info

# Optical crosstalk
export OpticalCrosstalkModel
export optical_crosstalk_none, optical_crosstalk_typical, optical_crosstalk_low, optical_crosstalk_high
export create_optical_crosstalk_kernel
export apply_optical_crosstalk, apply_optical_crosstalk_intensity
