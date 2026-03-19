# =============================================================================
# Optical Crosstalk (CatSim-style separable kernel)
# =============================================================================
#
# Optical crosstalk modeling for scintillator-based CT detectors.
# Light spreading in the scintillator causes signal leakage to neighbors.

import AcceleratedKernels as AK

"""
    OpticalCrosstalkModel

Optical crosstalk model using CatSim-exact separable convolution kernels.

# Physics

Optical crosstalk occurs in scintillator-based CT detectors when light generated
by X-ray absorption in one detector element spreads to neighboring elements before
being collected by the photodiode. The spreading is caused by:

- Light scattering within the scintillator crystal
- Total internal reflection at crystal boundaries
- Imperfect optical isolation between pixels

This is distinct from electronic crosstalk, which occurs in the readout electronics.

# Mathematical Formulation

CatSim models optical crosstalk as a 2D convolution with a separable kernel:

    row_kernel = [α, 1-2α, α]
    col_kernel = [β, 1-2β, β]
    K = col_kernel ⊗ row_kernel  (outer product)

The resulting 3×3 kernel is:

    [αβ,       α(1-2β),     αβ      ]
    [(1-2α)β,  (1-2α)(1-2β), (1-2α)β]
    [αβ,       α(1-2β),     αβ      ]

where the center element (1-2α)(1-2β) represents the primary signal fraction,
and neighbors receive signal proportional to α and/or β.

# Fields
- `row_coeff::Float64`: Row crosstalk coefficient α (typical: 0.04-0.05)
- `col_coeff::Float64`: Column crosstalk coefficient β (typical: 0.04-0.05)

# CatSim Compatibility

This implementation matches CatSim's CalcOptCrossTalk.py exactly.
Default values `optical_crosstalk_typical()` use CatSim's row=0.045, col=0.040.

# Example

```julia
# CatSim typical optical crosstalk
model = optical_crosstalk_typical()
kernel = create_optical_crosstalk_kernel(model)
# Apply to intensity data
result = apply_optical_crosstalk_intensity(intensity, model)
```

# See Also
- [`create_optical_crosstalk_kernel`](@ref)
- [`apply_optical_crosstalk!`](@ref)
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
    apply_optical_crosstalk_intensity!(intensity, model::OpticalCrosstalkModel) -> intensity

Apply optical crosstalk to intensity-domain data (in-place, GPU-native).

Uses spatial domain 3x3 convolution for GPU compatibility.
"""
function apply_optical_crosstalk_intensity!(
    intensity::AbstractArray{T,3},
    model::OpticalCrosstalkModel
) where T
    if model.row_coeff ≈ 0 && model.col_coeff ≈ 0
        return intensity
    end

    n_cols = size(intensity, 1)
    n_rows = size(intensity, 2)

    # Create 3x3 kernel on CPU
    kernel_cpu = T.(create_optical_crosstalk_kernel(model))

    # Transfer kernel to GPU
    kernel = similar(intensity, 3, 3)
    copyto!(kernel, kernel_cpu)

    # Output buffer
    output = similar(intensity)

    # GPU-native spatial convolution
    let kernel = kernel, output = output, n_cols = n_cols, n_rows = n_rows
        AK.foreachindex(intensity) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % Int32(n_cols)) + Int32(1)
            idx_0 = idx_0 ÷ Int32(n_cols)
            row = (idx_0 % Int32(n_rows)) + Int32(1)
            angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

            # Apply 3x3 convolution
            acc = zero(T)
            for di in -1:1
                for dj in -1:1
                    src_col = clamp(col + di, Int32(1), Int32(n_cols))
                    src_row = clamp(row + dj, Int32(1), Int32(n_rows))

                    ki = di + 2
                    kj = dj + 2

                    acc += intensity[src_col, src_row, angle] * kernel[ki, kj]
                end
            end

            output[idx] = max(acc, T(0))
        end
    end

    copyto!(intensity, output)

    return intensity
end

"""
    apply_optical_crosstalk!(sinogram, model::OpticalCrosstalkModel) -> sinogram

Apply optical crosstalk to sinogram (in-place, GPU-native).

Converts to intensity domain, applies crosstalk, converts back.
"""
function apply_optical_crosstalk!(
    sinogram::AbstractArray{T,3},
    model::OpticalCrosstalkModel;
    ws_output=nothing, ws_kernel=nothing
) where T
    if model.row_coeff ≈ 0 && model.col_coeff ≈ 0
        return sinogram
    end

    n_cols = size(sinogram, 1)
    n_rows = size(sinogram, 2)

    # Create 3x3 kernel (or use pre-computed workspace kernel)
    if ws_kernel !== nothing
        kernel = ws_kernel
    else
        kernel_cpu = T.(create_optical_crosstalk_kernel(model))
        kernel = similar(sinogram, 3, 3)
        copyto!(kernel, kernel_cpu)
    end

    # Output buffer (use workspace or allocate)
    output = ws_output !== nothing ? ws_output : similar(sinogram)

    # GPU-native: convert to intensity, apply convolution, convert back
    # let-bind to capture with concrete type (avoids Core.Box on GPU)
    let kernel = kernel, output = output, n_cols = n_cols, n_rows = n_rows
        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % Int32(n_cols)) + Int32(1)
            idx_0 = idx_0 ÷ Int32(n_cols)
            row = (idx_0 % Int32(n_rows)) + Int32(1)
            angle = (idx_0 ÷ Int32(n_rows)) + Int32(1)

            # Apply 3x3 convolution in intensity domain
            acc = zero(T)
            for di in -1:1
                for dj in -1:1
                    src_col = clamp(col + di, Int32(1), Int32(n_cols))
                    src_row = clamp(row + dj, Int32(1), Int32(n_rows))

                    ki = di + 2
                    kj = dj + 2

                    intensity_src = exp(-sinogram[src_col, src_row, angle])
                    acc += intensity_src * kernel[ki, kj]
                end
            end

            output[idx] = -log(max(acc, T(1e-10)))
        end
    end

    copyto!(sinogram, output)

    return sinogram
end

# Convenience wrappers that allocate
function apply_optical_crosstalk(sinogram::AbstractArray{T,3}, model::OpticalCrosstalkModel) where T
    result = copy(sinogram)
    return apply_optical_crosstalk!(result, model)
end

function apply_optical_crosstalk_intensity(intensity::AbstractArray{T,3}, model::OpticalCrosstalkModel) where T
    result = copy(intensity)
    return apply_optical_crosstalk_intensity!(result, model)
end

# =============================================================================
# Exports
# =============================================================================

# Optical crosstalk
export OpticalCrosstalkModel
export optical_crosstalk_none, optical_crosstalk_typical, optical_crosstalk_low, optical_crosstalk_high
export create_optical_crosstalk_kernel
export apply_optical_crosstalk!, apply_optical_crosstalk_intensity!
export apply_optical_crosstalk, apply_optical_crosstalk_intensity
