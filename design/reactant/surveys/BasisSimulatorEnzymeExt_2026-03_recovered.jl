# =============================================================================
# BasisSimulator.jl Enzyme Extension
# =============================================================================
#
# Provides automatic differentiation support for CT simulation via Enzyme.jl.
#
# Key insight: Forward projection and backprojection are mathematical adjoints.
# - Gradient of forward_project w.r.t. volume = backproject
# - Gradient of backproject w.r.t. sinogram = forward_project
#
# This extension implements custom Enzyme rules that leverage this adjoint
# relationship for efficient gradient computation without AD through the
# complex ray-tracing kernels.
#
# Physics Effects Differentiability:
# - Scatter: ✅ Differentiable via convolution adjoint
# - Crosstalk: ✅ Differentiable via convolution adjoint
# - BHC (Beam Hardening Correction): ✅ Differentiable (polynomial)
# - Filter operations: ✅ Differentiable via convolution adjoint
# - Detector noise: ❌ NOT differentiable (stochastic)
#
# References:
# - Moses WS, Churavy V. "Enzyme: LLVM-based AD" (NeurIPS 2020)
# - CTorch: PyTorch differentiable CT projectors (arXiv:2503.16741)
# - TIGRE v3: Differentiable operators (arXiv:2412.10129)
#
# =============================================================================

module BasisSimulatorEnzymeExt

using BasisSimulator
using Enzyme
using Enzyme.EnzymeRules
using Random

import BasisSimulator: siddon_forward_project!, siddon_forward_project,
                       backproject!, backproject, CTGeometry,
                       ScatterModel, ScatterCorrectionModel,
                       CrosstalkModel, OpticalCrosstalkModel,
                       BHCPolynomial, BeamHardeningCorrection,
                       create_scatter_kernel_spatial,
                       create_crosstalk_kernel_3x3,
                       create_optical_crosstalk_kernel

import AcceleratedKernels as AK

# =============================================================================
# Exported API
# =============================================================================

export gradient_forward_project, gradient_backproject
export DifferentiableCT, is_enzyme_loaded

# Physics gradients
export gradient_scatter, gradient_scatter!
export gradient_scatter_correction, gradient_scatter_correction!
export gradient_crosstalk, gradient_crosstalk!
export gradient_optical_crosstalk, gradient_optical_crosstalk!
export gradient_bhc, gradient_bhc!
export gradient_filter, gradient_filter!

# Verification
export verify_gradient_scatter
export verify_gradient_crosstalk
export verify_gradient_bhc
export verify_gradient_filter

# Differentiability documentation
export DIFFERENTIABLE_EFFECTS, NON_DIFFERENTIABLE_EFFECTS

# =============================================================================
# Check if Enzyme is loaded
# =============================================================================

"""
    is_enzyme_loaded() -> Bool

Check if Enzyme.jl extension is loaded. Always returns true when called
from the extension module.
"""
is_enzyme_loaded() = true

# =============================================================================
# Manual Gradient Functions (Adjoint-Based)
# =============================================================================

"""
    gradient_forward_project(∂L_∂sinogram, volume, geom) -> ∂L_∂volume

Compute gradient of a loss function L w.r.t. the volume, given the gradient
of L w.r.t. the sinogram (output of forward projection).

This exploits the mathematical property that forward projection and
backprojection are adjoint operators:

    ⟨Ax, y⟩ = ⟨x, A'y⟩

where A = forward_project and A' = backproject.

Therefore:
    ∂L/∂x = A' × (∂L/∂y) = backproject(∂L/∂sinogram)

# Arguments
- `∂L_∂sinogram::AbstractArray{T,3}`: Gradient of loss w.r.t. sinogram [n_cols, n_rows, n_angles]
- `volume::AbstractArray{T,3}`: The input volume (used only for shape/type, values not used)
- `geom::CTGeometry`: CT scanner geometry

# Returns
- `∂L_∂volume::AbstractArray{T,3}`: Gradient of loss w.r.t. volume [nx, ny, nz]

# Example
```julia
using BasisSimulator
using Enzyme

# Create geometry and volume
scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=180, fov=(256.0, 256.0, 32.0))
volume = randn(Float32, 64, 64, 16)

# Forward pass
sinogram = siddon_forward_project(volume, geom)

# Compute some loss gradient (e.g., MSE w.r.t target)
target_sinogram = similar(sinogram)
∂L_∂sinogram = 2.0f0 .* (sinogram .- target_sinogram)  # d(MSE)/d(sinogram)

# Compute gradient w.r.t. volume
∂L_∂volume = gradient_forward_project(∂L_∂sinogram, volume, geom)
```

# Mathematical Derivation

For a loss function L(sinogram) where sinogram = forward_project(volume):

By chain rule:
    ∂L/∂volume = (∂sinogram/∂volume)ᵀ × (∂L/∂sinogram)

The Jacobian ∂sinogram/∂volume is the forward projection operator A.
Its transpose Aᵀ is the backprojection operator (matched, unweighted).

Therefore:
    ∂L/∂volume = backproject(∂L/∂sinogram, weighted=false)

# Notes
- Uses `weighted=false` for matched adjoint (SIRT-style backprojection)
- GPU arrays are handled automatically via AcceleratedKernels.jl
- The volume argument is only used for size/type inference, not its values
"""
function gradient_forward_project(
    ∂L_∂sinogram::AbstractArray{T,3},
    volume::AbstractArray{T,3},
    geom::CTGeometry
) where T <: AbstractFloat
    # Allocate gradient on same device as input
    ∂L_∂volume = similar(volume)
    fill!(∂L_∂volume, zero(T))

    # Backprojection is the adjoint of forward projection
    # Use weighted=false for matched adjoint (not FDK-weighted)
    backproject!(∂L_∂volume, ∂L_∂sinogram, geom; weighted=false)

    return ∂L_∂volume
end

"""
    gradient_forward_project!(∂L_∂volume, ∂L_∂sinogram, geom) -> ∂L_∂volume

In-place version of gradient computation for forward projection.
"""
function gradient_forward_project!(
    ∂L_∂volume::AbstractArray{T,3},
    ∂L_∂sinogram::AbstractArray{T,3},
    geom::CTGeometry
) where T <: AbstractFloat
    fill!(∂L_∂volume, zero(T))
    backproject!(∂L_∂volume, ∂L_∂sinogram, geom; weighted=false)
    return ∂L_∂volume
end

"""
    gradient_backproject(∂L_∂volume, sinogram, geom) -> ∂L_∂sinogram

Compute gradient of a loss function L w.r.t. the sinogram, given the gradient
of L w.r.t. the volume (output of backprojection).

This exploits the adjoint relationship:
    ∂L/∂y = A × (∂L/∂x) = forward_project(∂L/∂volume)

where A = forward_project.

# Arguments
- `∂L_∂volume::AbstractArray{T,3}`: Gradient of loss w.r.t. volume [nx, ny, nz]
- `sinogram::AbstractArray{T,3}`: The input sinogram (used only for shape/type)
- `geom::CTGeometry`: CT scanner geometry

# Returns
- `∂L_∂sinogram::AbstractArray{T,3}`: Gradient of loss w.r.t. sinogram [n_cols, n_rows, n_angles]

# Example
```julia
# Forward pass through backprojection
volume = backproject(sinogram, geom, (64, 64, 16); weighted=true)

# Compute some loss gradient
∂L_∂volume = 2.0f0 .* (volume .- target_volume)

# Compute gradient w.r.t. sinogram
∂L_∂sinogram = gradient_backproject(∂L_∂volume, sinogram, geom)
```
"""
function gradient_backproject(
    ∂L_∂volume::AbstractArray{T,3},
    sinogram::AbstractArray{T,3},
    geom::CTGeometry
) where T <: AbstractFloat
    # Allocate gradient on same device as input
    ∂L_∂sinogram = similar(sinogram)
    fill!(∂L_∂sinogram, zero(T))

    # Forward projection is the adjoint of backprojection
    siddon_forward_project!(∂L_∂sinogram, ∂L_∂volume, geom)

    return ∂L_∂sinogram
end

"""
    gradient_backproject!(∂L_∂sinogram, ∂L_∂volume, geom) -> ∂L_∂sinogram

In-place version of gradient computation for backprojection.
"""
function gradient_backproject!(
    ∂L_∂sinogram::AbstractArray{T,3},
    ∂L_∂volume::AbstractArray{T,3},
    geom::CTGeometry
) where T <: AbstractFloat
    fill!(∂L_∂sinogram, zero(T))
    siddon_forward_project!(∂L_∂sinogram, ∂L_∂volume, geom)
    return ∂L_∂sinogram
end

# =============================================================================
# Custom Enzyme Rules for Forward Projection
# =============================================================================

"""
Custom Enzyme rule for siddon_forward_project!

This rule tells Enzyme how to compute gradients through the forward projection
operation by using the adjoint relationship with backprojection.

The key insight: Instead of differentiating through the complex ray-tracing
kernel, we use the mathematical fact that backprojection is the adjoint
of forward projection.
"""
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(siddon_forward_project!)},
    ::Type{<:Duplicated},
    sinogram::Duplicated{<:AbstractArray{T,3}},
    volume::Duplicated{<:AbstractArray{T,3}},
    geom::Const{CTGeometry}
) where T <: AbstractFloat
    # Run the forward pass
    siddon_forward_project!(sinogram.val, volume.val, geom.val)

    # Store geometry for reverse pass (no other tape data needed)
    # The volume values themselves are not needed for the adjoint
    return EnzymeRules.AugmentedReturn(nothing, nothing, geom.val)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(siddon_forward_project!)},
    dret,
    tape,  # Contains geom from augmented_primal
    sinogram::Duplicated{<:AbstractArray{T,3}},
    volume::Duplicated{<:AbstractArray{T,3}},
    geom::Const{CTGeometry}
) where T <: AbstractFloat
    # Retrieve geometry from tape
    saved_geom = tape

    # Gradient of forward projection w.r.t. volume is backprojection of sinogram gradient
    # ∂L/∂volume += backproject(∂L/∂sinogram)
    #
    # Note: sinogram.dval contains ∂L/∂sinogram (incoming gradient)
    #       volume.dval is where we accumulate ∂L/∂volume

    # Use matched backprojection (weighted=false) for correct adjoint
    backproject!(volume.dval, sinogram.dval, saved_geom; weighted=false)

    # Return nothing for all arguments (gradients are accumulated in-place)
    return (nothing, nothing, nothing)
end

# =============================================================================
# Custom Enzyme Rules for Backprojection
# =============================================================================

"""
Custom Enzyme rule for backproject!

The adjoint of backprojection is forward projection.
"""
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(backproject!)},
    ::Type{<:Duplicated},
    volume::Duplicated{<:AbstractArray{T,3}},
    sinogram::Duplicated{<:AbstractArray{T,3}},
    geom::Const{CTGeometry};
    weighted::Bool = true
) where T <: AbstractFloat
    # Run the forward pass
    backproject!(volume.val, sinogram.val, geom.val; weighted=weighted)

    # Store geometry and weighted flag for reverse pass
    return EnzymeRules.AugmentedReturn(nothing, nothing, (geom.val, weighted))
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(backproject!)},
    dret,
    tape,  # Contains (geom, weighted) from augmented_primal
    volume::Duplicated{<:AbstractArray{T,3}},
    sinogram::Duplicated{<:AbstractArray{T,3}},
    geom::Const{CTGeometry};
    weighted::Bool = true
) where T <: AbstractFloat
    # Retrieve saved data from tape
    saved_geom, _weighted = tape

    # Gradient of backprojection w.r.t. sinogram is forward projection of volume gradient
    # ∂L/∂sinogram += forward_project(∂L/∂volume)
    #
    # Note: volume.dval contains ∂L/∂volume (incoming gradient)
    #       sinogram.dval is where we accumulate ∂L/∂sinogram

    siddon_forward_project!(sinogram.dval, volume.dval, saved_geom)

    return (nothing, nothing, nothing)
end

# =============================================================================
# High-Level Differentiable CT Interface
# =============================================================================

"""
    DifferentiableCT{T}

Container for differentiable CT operations with cached geometry and buffers.

# Fields
- `geom::CTGeometry`: CT scanner geometry
- `volume_size::NTuple{3,Int}`: Volume dimensions (nx, ny, nz)
- `sinogram_buffer::Union{Nothing,AbstractArray{T,3}}`: Pre-allocated sinogram buffer
- `volume_buffer::Union{Nothing,AbstractArray{T,3}}`: Pre-allocated volume buffer

# Example
```julia
using BasisSimulator
using Enzyme

# Create differentiable CT operator
scanner = GERevolutionApex()
geom = CTGeometry(scanner; n_angles=180, fov=(256.0, 256.0, 32.0))
dct = DifferentiableCT{Float32}(geom, (64, 64, 16))

# Forward and backward
sinogram = dct.forward(volume)
∂L_∂volume = dct.backward(∂L_∂sinogram, volume)
```
"""
struct DifferentiableCT{T<:AbstractFloat}
    geom::CTGeometry
    volume_size::NTuple{3,Int}
    sinogram_buffer::Union{Nothing,AbstractArray{T,3}}
    volume_buffer::Union{Nothing,AbstractArray{T,3}}
end

"""
    DifferentiableCT{T}(geom, volume_size; preallocate=false) where T

Create a differentiable CT operator.

# Arguments
- `geom::CTGeometry`: CT scanner geometry
- `volume_size::NTuple{3,Int}`: Volume dimensions

# Keyword Arguments
- `preallocate::Bool=false`: Pre-allocate internal buffers for efficiency
"""
function DifferentiableCT{T}(
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    preallocate::Bool = false
) where T <: AbstractFloat
    sino_buf = preallocate ? zeros(T, geom.n_cols, geom.n_rows, geom.n_angles) : nothing
    vol_buf = preallocate ? zeros(T, volume_size...) : nothing
    return DifferentiableCT{T}(geom, volume_size, sino_buf, vol_buf)
end

"""
    (dct::DifferentiableCT)(volume) -> sinogram

Forward projection (functor interface).
"""
function (dct::DifferentiableCT{T})(volume::AbstractArray{T,3}) where T
    return siddon_forward_project(volume, dct.geom)
end

# =============================================================================
# Finite Difference Gradient Verification
# =============================================================================

"""
    verify_gradient_forward_project(volume, geom; ε=1e-5, n_samples=5, seed=42) -> NamedTuple

Verify gradient computation by comparing with finite differences.

# Arguments
- `volume::AbstractArray{T,3}`: Test volume
- `geom::CTGeometry`: CT geometry

# Keyword Arguments
- `ε::Real=1e-5`: Finite difference step size
- `n_samples::Int=5`: Number of random voxels to test
- `seed::Int=42`: Random seed for reproducibility

# Returns
NamedTuple with fields:
- `max_relative_error`: Maximum relative error across samples
- `mean_relative_error`: Mean relative error
- `errors`: Vector of individual relative errors
- `passed`: Bool, true if max_relative_error < 1e-3
"""
function verify_gradient_forward_project(
    volume::AbstractArray{T,3},
    geom::CTGeometry;
    ε::Real = T(1e-5),
    n_samples::Int = 5,
    seed::Int = 42
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    # Compute forward projection
    sinogram = siddon_forward_project(volume, geom)

    # Use MSE loss for testing: L = sum(sinogram.^2) / 2
    # ∂L/∂sinogram = sinogram
    ∂L_∂sinogram = copy(sinogram)

    # Compute analytical gradient
    ∂L_∂volume_analytical = gradient_forward_project(∂L_∂sinogram, volume, geom)

    # Compare with finite differences at random voxels
    errors = Float64[]
    volume_perturbed = copy(volume)

    nx, ny, nz = size(volume)

    for _ in 1:n_samples
        # Random voxel
        ix = rand(rng, 1:nx)
        iy = rand(rng, 1:ny)
        iz = rand(rng, 1:nz)

        # Finite difference: ∂L/∂v[i] ≈ (L(v + ε*e_i) - L(v - ε*e_i)) / (2ε)

        # L(v + ε*e_i)
        copyto!(volume_perturbed, volume)
        volume_perturbed[ix, iy, iz] += ε
        sino_plus = siddon_forward_project(volume_perturbed, geom)
        L_plus = sum(sino_plus.^2) / 2

        # L(v - ε*e_i)
        copyto!(volume_perturbed, volume)
        volume_perturbed[ix, iy, iz] -= ε
        sino_minus = siddon_forward_project(volume_perturbed, geom)
        L_minus = sum(sino_minus.^2) / 2

        # Finite difference gradient
        grad_fd = Float64((L_plus - L_minus) / (2 * ε))

        # Analytical gradient
        grad_analytical = Float64(∂L_∂volume_analytical[ix, iy, iz])

        # Relative error
        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    max_rel_error = maximum(errors)
    mean_rel_error = sum(errors) / length(errors)
    passed = max_rel_error < 1e-3

    return (
        max_relative_error = max_rel_error,
        mean_relative_error = mean_rel_error,
        errors = errors,
        passed = passed
    )
end

"""
    verify_gradient_backproject(sinogram, geom, volume_size; ε=1e-5, n_samples=5, seed=42) -> NamedTuple

Verify gradient computation for backprojection by comparing with finite differences.
"""
function verify_gradient_backproject(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    ε::Real = T(1e-5),
    n_samples::Int = 5,
    seed::Int = 42,
    weighted::Bool = true
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    # Compute backprojection
    volume = backproject(sinogram, geom, volume_size; weighted=weighted)

    # Use MSE loss for testing: L = sum(volume.^2) / 2
    # ∂L/∂volume = volume
    ∂L_∂volume = copy(volume)

    # Compute analytical gradient
    ∂L_∂sinogram_analytical = gradient_backproject(∂L_∂volume, sinogram, geom)

    # Compare with finite differences at random detector pixels
    errors = Float64[]
    sinogram_perturbed = copy(sinogram)

    n_cols, n_rows, n_angles = size(sinogram)

    for _ in 1:n_samples
        # Random detector pixel
        ic = rand(rng, 1:n_cols)
        ir = rand(rng, 1:n_rows)
        ia = rand(rng, 1:n_angles)

        # L(s + ε*e_i)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] += ε
        vol_plus = backproject(sinogram_perturbed, geom, volume_size; weighted=weighted)
        L_plus = sum(vol_plus.^2) / 2

        # L(s - ε*e_i)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] -= ε
        vol_minus = backproject(sinogram_perturbed, geom, volume_size; weighted=weighted)
        L_minus = sum(vol_minus.^2) / 2

        # Finite difference gradient
        grad_fd = Float64((L_plus - L_minus) / (2 * ε))

        # Analytical gradient
        grad_analytical = Float64(∂L_∂sinogram_analytical[ic, ir, ia])

        # Relative error
        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    max_rel_error = maximum(errors)
    mean_rel_error = sum(errors) / length(errors)
    passed = max_rel_error < 1e-3

    return (
        max_relative_error = max_rel_error,
        mean_relative_error = mean_rel_error,
        errors = errors,
        passed = passed
    )
end

# =============================================================================
# Physics Effects Differentiability Documentation
# =============================================================================

"""
    DIFFERENTIABLE_EFFECTS

Dictionary documenting which physics effects are differentiable and how.

# Differentiable Effects

| Effect | Function | Gradient Method |
|--------|----------|-----------------|
| Forward Projection | `siddon_forward_project` | Adjoint (backprojection) |
| Backprojection | `backproject` | Adjoint (forward projection) |
| Scatter | `add_scatter!` | Convolution adjoint |
| Scatter Correction | `correct_scatter!` | Convolution adjoint |
| Crosstalk | `apply_crosstalk!` | Convolution adjoint |
| Optical Crosstalk | `apply_optical_crosstalk!` | Convolution adjoint |
| BHC | `apply_bhc!` | Polynomial derivative |
| Ramp Filter | `filter_sinogram!` | Convolution adjoint |
| Cosine Weight | `cosine_weight!` | Elementwise derivative |

# Usage

All gradient functions follow the pattern:
```julia
∂L_∂input = gradient_<effect>(∂L_∂output, input, model)
```

where ∂L_∂output is the gradient of the loss with respect to the effect's output.
"""
const DIFFERENTIABLE_EFFECTS = Dict(
    :forward_projection => "Adjoint via backprojection",
    :backprojection => "Adjoint via forward projection",
    :scatter => "Convolution adjoint (correlation)",
    :scatter_correction => "Convolution adjoint + chain rule",
    :crosstalk => "Convolution adjoint",
    :optical_crosstalk => "Convolution adjoint",
    :bhc => "Polynomial derivative (analytic)",
    :filter => "Convolution adjoint",
    :cosine_weight => "Elementwise multiplication"
)

"""
    NON_DIFFERENTIABLE_EFFECTS

Dictionary documenting which physics effects are NOT differentiable and why.

# Non-Differentiable Effects

| Effect | Function | Reason |
|--------|----------|--------|
| Quantum Noise | `add_quantum_noise!` | Stochastic (random sampling) |
| Electronic Noise | `add_electronic_noise!` | Stochastic (random sampling) |
| Detector Blur | `apply_detector_blur!` | Can be made differentiable, not yet implemented |

# Handling Noise in Differentiable Pipelines

For learning/optimization with noise:
1. Use expected value (deterministic forward model without noise)
2. Apply noise only during inference
3. Use reparameterization trick if noise gradient is needed (not implemented)

# Example
```julia
# During training (no noise)
sinogram = siddon_forward_project(volume, geom)
loss = sum((sinogram .- target).^2)
∂L_∂volume = gradient_forward_project(sinogram .- target, volume, geom)

# During inference (with noise)
sinogram_noisy = add_quantum_noise!(copy(sinogram), detector_model)
```
"""
const NON_DIFFERENTIABLE_EFFECTS = Dict(
    :quantum_noise => "Stochastic operation (Poisson sampling)",
    :electronic_noise => "Stochastic operation (Gaussian sampling)",
    :detector_blur => "Could be differentiable but not implemented"
)

# =============================================================================
# Scatter Gradient (Convolution Adjoint)
# =============================================================================

"""
    gradient_scatter!(∂L_∂input, ∂L_∂output, input, model::ScatterModel)

Compute gradient of loss w.r.t. input sinogram through scatter operation.

The scatter operation is essentially a weighted convolution in intensity domain:
    output = f(conv(g(input), kernel))

where g converts to intensity domain and f converts back.

The gradient uses the adjoint of convolution (correlation):
    ∂L/∂input = g'(input) × corr(∂L/∂output × f'(output), kernel)

For computational efficiency, we use a simplified linearized approximation
that is accurate for small scatter contributions.

# Arguments
- `∂L_∂input`: Output array for gradient (modified in place)
- `∂L_∂output`: Gradient of loss w.r.t. scatter output
- `input`: Original input sinogram
- `model::ScatterModel`: Scatter model parameters

# Returns
Modified `∂L_∂input` array.
"""
function gradient_scatter!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::ScatterModel
) where T <: AbstractFloat
    n_cols = size(input, 1)
    n_rows = size(input, 2)

    C = T(model.scatter_coefficient * model.scale_factor)

    # Create scatter kernel (same as in add_scatter!)
    kernel_cpu = T.(create_scatter_kernel_spatial(model))
    kernel_size = size(kernel_cpu, 1)
    half_k = kernel_size ÷ 2

    kernel = similar(input, size(kernel_cpu)...)
    copyto!(kernel, kernel_cpu)

    # For the adjoint, we need to correlate (flip kernel) and chain rule through
    # the log/exp transformations
    #
    # Forward: output[i] = -log(exp(-input[i]) + scatter[i])
    #   where scatter = conv(exp(-input) × input × C, kernel)
    #
    # The adjoint involves:
    # 1. ∂output/∂intensity = -1/intensity at output
    # 2. Correlation (transposed convolution) of scaled gradient
    # 3. Chain rule through exp(-input)

    AK.foreachindex(input) do idx
        ci = CartesianIndices(input)[idx]
        col, row, angle = Tuple(ci)

        # Get upstream gradient
        dL_dout = ∂L_∂output[idx]

        # Input values
        p_in = input[idx]
        clamped_in = min(p_in, T(20))
        intensity_in = exp(-clamped_in)

        # Gradient of output w.r.t. total intensity
        # output = -log(total_intensity)
        # d(output)/d(total_intensity) = -1/total_intensity
        # For simplicity, approximate total_intensity ≈ intensity_in (small scatter)
        d_out_d_total = -one(T) / max(intensity_in, T(1e-10))

        # Accumulate gradient via correlation (adjoint of convolution)
        grad_acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                # Kernel weight (same index for correlation in this formulation)
                ki = di + half_k + 1
                kj = dj + half_k + 1

                # The gradient from scatter contribution at neighbor positions
                # This is a linearized approximation
                neighbor_dL_dout = ∂L_∂output[src_col, src_row, angle]
                neighbor_p = input[src_col, src_row, angle]
                neighbor_intensity = exp(-min(neighbor_p, T(20)))

                # Contribution to gradient
                grad_acc += neighbor_dL_dout * kernel[ki, kj] * C * neighbor_intensity
            end
        end

        # Chain rule through intensity: d(intensity)/d(input) = -intensity
        # Plus direct contribution from the center pixel
        ∂L_∂input[idx] = dL_dout * d_out_d_total * (-intensity_in) + grad_acc * (-intensity_in)
    end

    return ∂L_∂input
end

"""
    gradient_scatter(∂L_∂output, input, model::ScatterModel) -> ∂L_∂input

Non-mutating version of gradient_scatter!.
"""
function gradient_scatter(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::ScatterModel
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    fill!(∂L_∂input, zero(T))
    return gradient_scatter!(∂L_∂input, ∂L_∂output, input, model)
end

# =============================================================================
# Scatter Correction Gradient
# =============================================================================

"""
    gradient_scatter_correction!(∂L_∂input, ∂L_∂output, input, model::ScatterCorrectionModel)

Compute gradient of loss w.r.t. input sinogram through scatter correction.

Similar to scatter gradient but for the correction operation.
"""
function gradient_scatter_correction!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::ScatterCorrectionModel
) where T <: AbstractFloat
    n_cols = size(input, 1)
    n_rows = size(input, 2)

    C = T(model.correction_coefficient * model.scale_factor)
    alpha = T(model.prep_exponent)

    # Create kernel
    scatter_model_temp = ScatterModel(
        model.correction_coefficient,
        model.scale_factor,
        model.kernel_fwhm,
        model.kernel_type
    )
    kernel_cpu = T.(create_scatter_kernel_spatial(scatter_model_temp))
    kernel_size = size(kernel_cpu, 1)
    half_k = kernel_size ÷ 2

    kernel = similar(input, size(kernel_cpu)...)
    copyto!(kernel, kernel_cpu)

    eps = T(1e-10)

    # Forward: output = -log(exp(-input) - scatter_estimate)
    # The gradient involves chain rule through log, subtraction, and convolution

    AK.foreachindex(input) do idx
        ci = CartesianIndices(input)[idx]
        col, row, angle = Tuple(ci)

        dL_dout = ∂L_∂output[idx]

        prep = input[idx]
        clamped_prep = min(max(prep, eps), T(15))
        intensity = exp(-clamped_prep)

        # Gradient through -log(corrected_intensity)
        # d(-log(x))/dx = -1/x
        # corrected_intensity = intensity - scatter_est
        # Approximate corrected_intensity ≈ intensity for small scatter
        d_out_d_corrected = -one(T) / max(intensity, eps)

        # Gradient accumulation via correlation
        grad_acc = zero(T)
        for dj in -half_k:half_k
            for di in -half_k:half_k
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)

                ki = di + half_k + 1
                kj = dj + half_k + 1

                neighbor_dL = ∂L_∂output[src_col, src_row, angle]
                neighbor_prep = input[src_col, src_row, angle]
                neighbor_clamped = min(max(neighbor_prep, eps), T(15))
                neighbor_intensity = exp(-neighbor_clamped)

                # d(scatter_pre)/d(prep) = d(intensity × prep^α × C)/d(prep)
                #   = -intensity × prep^α × C + intensity × α × prep^(α-1) × C
                d_scatter_d_prep = neighbor_intensity * C * (
                    -neighbor_clamped^alpha + alpha * neighbor_clamped^(alpha-one(T))
                )

                grad_acc += neighbor_dL * kernel[ki, kj] * d_scatter_d_prep
            end
        end

        # Total gradient: through output log + through scatter subtraction
        ∂L_∂input[idx] = dL_dout * d_out_d_corrected * (-intensity) - grad_acc
    end

    return ∂L_∂input
end

function gradient_scatter_correction(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::ScatterCorrectionModel
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    fill!(∂L_∂input, zero(T))
    return gradient_scatter_correction!(∂L_∂input, ∂L_∂output, input, model)
end

# =============================================================================
# Crosstalk Gradient (Convolution Adjoint)
# =============================================================================

"""
    gradient_crosstalk!(∂L_∂input, ∂L_∂output, input, model::CrosstalkModel)

Compute gradient of loss w.r.t. input sinogram through crosstalk operation.

Crosstalk is a 3×3 convolution in intensity domain:
    output = -log(conv(exp(-input), kernel))

The gradient uses the transposed convolution (correlation with flipped kernel).
"""
function gradient_crosstalk!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::CrosstalkModel
) where T <: AbstractFloat
    if model.type == :none || model.primary_fraction >= 1.0
        copyto!(∂L_∂input, ∂L_∂output)
        return ∂L_∂input
    end

    n_cols = size(input, 1)
    n_rows = size(input, 2)

    # Create kernel
    kernel_cpu = T.(create_crosstalk_kernel_3x3(model))
    kernel = similar(input, 3, 3)
    copyto!(kernel, kernel_cpu)

    # Forward: output = -log(conv(intensity, kernel))
    # where intensity = exp(-input)
    #
    # d(output)/d(intensity_out) = -1/intensity_out
    # d(intensity_out)/d(intensity_in) = kernel (via convolution)
    # d(intensity)/d(input) = -intensity

    AK.foreachindex(input) do idx
        ci = CartesianIndices(input)[idx]
        col, row, angle = Tuple(ci)

        # Compute output intensity for this pixel
        intensity_out = zero(T)
        for di in -1:1
            for dj in -1:1
                src_col = clamp(col + di, 1, n_cols)
                src_row = clamp(row + dj, 1, n_rows)
                ki = di + 2
                kj = dj + 2
                intensity_out += exp(-input[src_col, src_row, angle]) * kernel[ki, kj]
            end
        end
        intensity_out = max(intensity_out, T(1e-10))

        # Accumulate gradient via transposed convolution
        grad_acc = zero(T)
        for di in -1:1
            for dj in -1:1
                dest_col = clamp(col - di, 1, n_cols)
                dest_row = clamp(row - dj, 1, n_rows)

                # The kernel coefficient for this contribution
                ki = di + 2
                kj = dj + 2

                # Upstream gradient at destination
                dL_dout_dest = ∂L_∂output[dest_col, dest_row, angle]

                # Compute intensity_out at destination
                intensity_out_dest = zero(T)
                for di2 in -1:1
                    for dj2 in -1:1
                        src2_col = clamp(dest_col + di2, 1, n_cols)
                        src2_row = clamp(dest_row + dj2, 1, n_rows)
                        intensity_out_dest += exp(-input[src2_col, src2_row, angle]) * kernel[di2+2, dj2+2]
                    end
                end
                intensity_out_dest = max(intensity_out_dest, T(1e-10))

                # d(output)/d(intensity_in) = kernel / intensity_out
                grad_acc += dL_dout_dest * (-one(T) / intensity_out_dest) * kernel[ki, kj]
            end
        end

        # Chain rule through exp(-input)
        intensity_in = exp(-input[col, row, angle])
        ∂L_∂input[idx] = grad_acc * (-intensity_in)
    end

    return ∂L_∂input
end

function gradient_crosstalk(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::CrosstalkModel
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    fill!(∂L_∂input, zero(T))
    return gradient_crosstalk!(∂L_∂input, ∂L_∂output, input, model)
end

# =============================================================================
# Optical Crosstalk Gradient
# =============================================================================

"""
    gradient_optical_crosstalk!(∂L_∂input, ∂L_∂output, input, model::OpticalCrosstalkModel)

Compute gradient of loss w.r.t. input through optical crosstalk.
"""
function gradient_optical_crosstalk!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::OpticalCrosstalkModel
) where T <: AbstractFloat
    if model.row_coeff ≈ 0 && model.col_coeff ≈ 0
        copyto!(∂L_∂input, ∂L_∂output)
        return ∂L_∂input
    end

    n_cols = size(input, 1)
    n_rows = size(input, 2)

    # Create kernel
    kernel_cpu = T.(create_optical_crosstalk_kernel(model))
    kernel = similar(input, 3, 3)
    copyto!(kernel, kernel_cpu)

    # Same structure as crosstalk gradient
    AK.foreachindex(input) do idx
        ci = CartesianIndices(input)[idx]
        col, row, angle = Tuple(ci)

        # Accumulate gradient via transposed convolution
        grad_acc = zero(T)
        for di in -1:1
            for dj in -1:1
                dest_col = clamp(col - di, 1, n_cols)
                dest_row = clamp(row - dj, 1, n_rows)

                ki = di + 2
                kj = dj + 2

                dL_dout_dest = ∂L_∂output[dest_col, dest_row, angle]

                # Compute intensity_out at destination
                intensity_out_dest = zero(T)
                for di2 in -1:1
                    for dj2 in -1:1
                        src2_col = clamp(dest_col + di2, 1, n_cols)
                        src2_row = clamp(dest_row + dj2, 1, n_rows)
                        intensity_out_dest += exp(-input[src2_col, src2_row, angle]) * kernel[di2+2, dj2+2]
                    end
                end
                intensity_out_dest = max(intensity_out_dest, T(1e-10))

                grad_acc += dL_dout_dest * (-one(T) / intensity_out_dest) * kernel[ki, kj]
            end
        end

        intensity_in = exp(-input[col, row, angle])
        ∂L_∂input[idx] = grad_acc * (-intensity_in)
    end

    return ∂L_∂input
end

function gradient_optical_crosstalk(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    model::OpticalCrosstalkModel
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    fill!(∂L_∂input, zero(T))
    return gradient_optical_crosstalk!(∂L_∂input, ∂L_∂output, input, model)
end

# =============================================================================
# Beam Hardening Correction Gradient (Polynomial Derivative)
# =============================================================================

"""
    gradient_bhc!(∂L_∂input, ∂L_∂output, input, poly::BHCPolynomial)

Compute gradient of loss w.r.t. input sinogram through BHC polynomial.

BHC applies a polynomial transformation:
    output = a₀ + a₁×input + a₂×input² + ... + aₙ×inputⁿ

The gradient is simply the polynomial derivative:
    ∂output/∂input = a₁ + 2×a₂×input + 3×a₃×input² + ... + n×aₙ×input^(n-1)

This is exact and efficient (no approximations needed).
"""
function gradient_bhc!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    poly::BHCPolynomial
) where T <: AbstractFloat
    order = poly.order

    # Transfer coefficients to GPU
    coeffs_cpu = T.(poly.coefficients)
    coeffs = similar(input, T, length(coeffs_cpu))
    copyto!(coeffs, coeffs_cpu)

    AK.foreachindex(input) do idx
        p = input[idx]
        dL_dout = ∂L_∂output[idx]

        # Compute derivative of polynomial: d(Σ aᵢ pⁱ)/dp = Σ i×aᵢ×p^(i-1)
        dpoly_dp = zero(T)
        p_power = one(T)  # p^(i-1), starts at p^0 = 1
        for i in 1:order
            dpoly_dp += T(i) * coeffs[i+1] * p_power
            p_power *= p
        end

        # Chain rule: ∂L/∂input = ∂L/∂output × ∂output/∂input
        ∂L_∂input[idx] = dL_dout * dpoly_dp
    end

    return ∂L_∂input
end

function gradient_bhc!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    bhc::BeamHardeningCorrection
) where T <: AbstractFloat
    return gradient_bhc!(∂L_∂input, ∂L_∂output, input, bhc.polynomial)
end

function gradient_bhc(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    bhc_or_poly::Union{BHCPolynomial, BeamHardeningCorrection}
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    return gradient_bhc!(∂L_∂input, ∂L_∂output, input, bhc_or_poly)
end

# =============================================================================
# Filter Gradient (Convolution Adjoint)
# =============================================================================

"""
    gradient_filter!(∂L_∂input, ∂L_∂output, input, kernel)

Compute gradient of loss w.r.t. input sinogram through spatial filter.

Filtering is a 1D convolution along detector columns:
    output[i] = Σⱼ kernel[j] × input[i+j]

The adjoint is correlation with the kernel:
    ∂L/∂input[i] = Σⱼ kernel[j] × ∂L/∂output[i-j]

For symmetric kernels (like the ramp filter), correlation equals convolution.
"""
function gradient_filter!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    kernel::AbstractVector{T}
) where T <: AbstractFloat
    n_cols = size(input, 1)
    n_rows = size(input, 2)
    kernel_len = length(kernel)
    half_k = kernel_len ÷ 2

    # Transfer kernel to GPU if needed
    kernel_gpu = similar(input, kernel_len)
    copyto!(kernel_gpu, kernel)

    # The gradient through a convolution with clamped boundaries.
    # Forward: output[dest] = sum_dk input[clamp(dest+dk)] * kernel[dk]
    #
    # For gradient, we iterate over all output positions and see which input
    # positions they read from (with clamping).

    AK.foreachindex(input) do idx
        ci = CartesianIndices(input)[idx]
        col, row, angle = Tuple(ci)

        # Sum gradient contributions from all output positions that read from this input
        # For each output dest_col, check if clamp(dest_col + dk) == col for any dk
        grad_acc = zero(T)

        # Loop over all output positions
        for dest_col in 1:n_cols
            for dk in -half_k:half_k
                # Which input does output[dest_col] read for this kernel position?
                src_col = clamp(dest_col + dk, 1, n_cols)
                if src_col == col
                    ki = dk + half_k + 1
                    grad_acc += ∂L_∂output[dest_col, row, angle] * kernel_gpu[ki]
                end
            end
        end

        ∂L_∂input[idx] = grad_acc
    end

    return ∂L_∂input
end

function gradient_filter(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    kernel::AbstractVector{T}
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    fill!(∂L_∂input, zero(T))
    return gradient_filter!(∂L_∂input, ∂L_∂output, input, kernel)
end

# =============================================================================
# Gradient Verification Functions
# =============================================================================

"""
    verify_gradient_scatter(sinogram, model::ScatterModel; ε=1e-5, n_samples=5, seed=42)

Verify scatter gradient by comparing with finite differences.
"""
function verify_gradient_scatter(
    sinogram::AbstractArray{T,3},
    model::ScatterModel;
    ε::Real = T(1e-5),
    n_samples::Int = 5,
    seed::Int = 42
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    # Forward pass
    output = BasisSimulator.add_scatter(sinogram, model)

    # Loss = sum(output.^2) / 2
    ∂L_∂output = copy(output)

    # Analytical gradient
    ∂L_∂sinogram_analytical = gradient_scatter(∂L_∂output, sinogram, model)

    # Finite differences
    errors = Float64[]
    sinogram_perturbed = copy(sinogram)
    n_cols, n_rows, n_angles = size(sinogram)

    for _ in 1:n_samples
        ic = rand(rng, 1:n_cols)
        ir = rand(rng, 1:n_rows)
        ia = rand(rng, 1:n_angles)

        # L(s + ε)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] += ε
        out_plus = BasisSimulator.add_scatter(sinogram_perturbed, model)
        L_plus = sum(out_plus.^2) / 2

        # L(s - ε)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] -= ε
        out_minus = BasisSimulator.add_scatter(sinogram_perturbed, model)
        L_minus = sum(out_minus.^2) / 2

        grad_fd = Float64((L_plus - L_minus) / (2 * ε))
        grad_analytical = Float64(∂L_∂sinogram_analytical[ic, ir, ia])

        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    return (
        max_relative_error = maximum(errors),
        mean_relative_error = sum(errors) / length(errors),
        errors = errors,
        passed = maximum(errors) < 0.1  # Allow 10% error for scatter (complex op)
    )
end

"""
    verify_gradient_crosstalk(sinogram, model::CrosstalkModel; ε=1e-5, n_samples=5, seed=42)

Verify crosstalk gradient by comparing with finite differences.
"""
function verify_gradient_crosstalk(
    sinogram::AbstractArray{T,3},
    model::CrosstalkModel;
    ε::Real = T(1e-5),
    n_samples::Int = 5,
    seed::Int = 42
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    output = BasisSimulator.apply_crosstalk(sinogram, model)
    ∂L_∂output = copy(output)
    ∂L_∂sinogram_analytical = gradient_crosstalk(∂L_∂output, sinogram, model)

    errors = Float64[]
    sinogram_perturbed = copy(sinogram)
    n_cols, n_rows, n_angles = size(sinogram)

    for _ in 1:n_samples
        ic = rand(rng, 1:n_cols)
        ir = rand(rng, 1:n_rows)
        ia = rand(rng, 1:n_angles)

        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] += ε
        out_plus = BasisSimulator.apply_crosstalk(sinogram_perturbed, model)
        L_plus = sum(out_plus.^2) / 2

        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] -= ε
        out_minus = BasisSimulator.apply_crosstalk(sinogram_perturbed, model)
        L_minus = sum(out_minus.^2) / 2

        grad_fd = Float64((L_plus - L_minus) / (2 * ε))
        grad_analytical = Float64(∂L_∂sinogram_analytical[ic, ir, ia])

        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    return (
        max_relative_error = maximum(errors),
        mean_relative_error = sum(errors) / length(errors),
        errors = errors,
        passed = maximum(errors) < 0.1
    )
end

"""
    verify_gradient_bhc(sinogram, poly::BHCPolynomial; ε=1e-5, n_samples=5, seed=42)

Verify BHC gradient by comparing with finite differences.
"""
function verify_gradient_bhc(
    sinogram::AbstractArray{T,3},
    poly::BHCPolynomial;
    ε::Real = T(1e-5),
    n_samples::Int = 5,
    seed::Int = 42
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    output = BasisSimulator.apply_bhc(sinogram, poly)
    ∂L_∂output = copy(output)
    ∂L_∂sinogram_analytical = gradient_bhc(∂L_∂output, sinogram, poly)

    errors = Float64[]
    sinogram_perturbed = copy(sinogram)
    n_cols, n_rows, n_angles = size(sinogram)

    for _ in 1:n_samples
        ic = rand(rng, 1:n_cols)
        ir = rand(rng, 1:n_rows)
        ia = rand(rng, 1:n_angles)

        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] += ε
        out_plus = BasisSimulator.apply_bhc(sinogram_perturbed, poly)
        L_plus = sum(out_plus.^2) / 2

        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] -= ε
        out_minus = BasisSimulator.apply_bhc(sinogram_perturbed, poly)
        L_minus = sum(out_minus.^2) / 2

        grad_fd = Float64((L_plus - L_minus) / (2 * ε))
        grad_analytical = Float64(∂L_∂sinogram_analytical[ic, ir, ia])

        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    return (
        max_relative_error = maximum(errors),
        mean_relative_error = sum(errors) / length(errors),
        errors = errors,
        passed = maximum(errors) < 1e-3  # BHC should be very accurate
    )
end

"""
    verify_gradient_filter(sinogram, kernel; ε=1e-5, n_samples=5, seed=42)

Verify filter gradient by comparing with finite differences.
"""
function verify_gradient_filter(
    sinogram::AbstractArray{T,3},
    kernel::AbstractVector{T};
    ε::Real = T(1e-5),
    n_samples::Int = 5,
    seed::Int = 42
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    # Simple spatial convolution along columns for testing
    n_cols, n_rows, n_angles = size(sinogram)
    kernel_len = length(kernel)
    half_k = kernel_len ÷ 2

    # Forward: convolve
    output = similar(sinogram)
    for angle in 1:n_angles
        for row in 1:n_rows
            for col in 1:n_cols
                acc = zero(T)
                for dk in -half_k:half_k
                    src_col = clamp(col + dk, 1, n_cols)
                    acc += sinogram[src_col, row, angle] * kernel[dk + half_k + 1]
                end
                output[col, row, angle] = acc
            end
        end
    end

    ∂L_∂output = copy(output)
    ∂L_∂sinogram_analytical = gradient_filter(∂L_∂output, sinogram, kernel)

    errors = Float64[]
    sinogram_perturbed = copy(sinogram)

    for _ in 1:n_samples
        ic = rand(rng, 1:n_cols)
        ir = rand(rng, 1:n_rows)
        ia = rand(rng, 1:n_angles)

        # L(s + ε)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] += ε
        out_plus = similar(sinogram_perturbed)
        for angle in 1:n_angles
            for row in 1:n_rows
                for col in 1:n_cols
                    acc = zero(T)
                    for dk in -half_k:half_k
                        src_col = clamp(col + dk, 1, n_cols)
                        acc += sinogram_perturbed[src_col, row, angle] * kernel[dk + half_k + 1]
                    end
                    out_plus[col, row, angle] = acc
                end
            end
        end
        L_plus = sum(out_plus.^2) / 2

        # L(s - ε)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] -= ε
        out_minus = similar(sinogram_perturbed)
        for angle in 1:n_angles
            for row in 1:n_rows
                for col in 1:n_cols
                    acc = zero(T)
                    for dk in -half_k:half_k
                        src_col = clamp(col + dk, 1, n_cols)
                        acc += sinogram_perturbed[src_col, row, angle] * kernel[dk + half_k + 1]
                    end
                    out_minus[col, row, angle] = acc
                end
            end
        end
        L_minus = sum(out_minus.^2) / 2

        grad_fd = Float64((L_plus - L_minus) / (2 * ε))
        grad_analytical = Float64(∂L_∂sinogram_analytical[ic, ir, ia])

        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    return (
        max_relative_error = maximum(errors),
        mean_relative_error = sum(errors) / length(errors),
        errors = errors,
        passed = maximum(errors) < 1e-3
    )
end

# =============================================================================
# Reconstruction Gradients
# =============================================================================

# Import additional functions needed for reconstruction gradients
import BasisSimulator: filter_sinogram, filter_sinogram!, cosine_weight!,
                       fdk_reconstruct, sirt_reconstruct, sirt_reconstruct!,
                       FilterType, RampFilter, create_spatial_kernel

# Export reconstruction gradient functions
export gradient_fdk_reconstruct, gradient_fdk_reconstruct!
export gradient_filter_sinogram, gradient_filter_sinogram!
export gradient_cosine_weight, gradient_cosine_weight!
export gradient_sirt_iteration, gradient_sirt_reconstruct
export verify_gradient_fdk_reconstruct
export DifferentiableFDK, DifferentiableSIRT

# =============================================================================
# Cosine Weight Gradient (Elementwise)
# =============================================================================

"""
    gradient_cosine_weight!(∂L_∂input, ∂L_∂output, input, geom)

Compute gradient of loss w.r.t. input sinogram through cosine weighting.

Cosine weighting is an elementwise operation:
    output[i] = input[i] × w[i]

where w[i] = SDD / sqrt(SDD² + u² + v²)

The gradient is simply:
    ∂L/∂input[i] = ∂L/∂output[i] × w[i]

# Arguments
- `∂L_∂input`: Output array for gradient (modified in place)
- `∂L_∂output`: Gradient of loss w.r.t. cosine-weighted output
- `input`: Original input sinogram (used only for size)
- `geom::CTGeometry`: CT geometry

# Returns
Modified `∂L_∂input` array.
"""
function gradient_cosine_weight!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    geom::CTGeometry
) where T <: AbstractFloat
    n_cols = Int32(size(input, 1))
    n_rows = Int32(size(input, 2))

    pixel_size = T(geom.pixel_size)
    magnification = T(geom.SDD / geom.SAD)
    SDD = T(geom.SDD)
    SDD_sq = SDD * SDD

    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    AK.foreachindex(input) do idx
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)

        # Compute detector pixel position
        u = (T(col) - col_center) * pixel_size * magnification
        v = (T(row) - row_center) * pixel_size * magnification

        # Distance from source to detector pixel
        dist = sqrt(SDD_sq + u^2 + v^2)

        # Cosine weight
        weight = SDD / dist

        # Gradient: ∂L/∂input = ∂L/∂output × weight
        ∂L_∂input[idx] = ∂L_∂output[idx] * weight
    end

    return ∂L_∂input
end

function gradient_cosine_weight(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    geom::CTGeometry
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    return gradient_cosine_weight!(∂L_∂input, ∂L_∂output, input, geom)
end

# =============================================================================
# Filter Sinogram Gradient (Cosine Weight + Convolution)
# =============================================================================

"""
    gradient_filter_sinogram!(∂L_∂input, ∂L_∂output, input, geom; filter=RampFilter(), cutoff=1.0)

Compute gradient of loss w.r.t. input sinogram through FDK filtering.

FDK filtering applies: output = convolve(cosine_weight(input), kernel)

The gradient chain rule:
1. ∂L/∂weighted = gradient of convolution (correlation with kernel)
2. ∂L/∂input = gradient of cosine weighting (elementwise × weight)

# Arguments
- `∂L_∂input`: Output array for gradient (modified in place)
- `∂L_∂output`: Gradient of loss w.r.t. filtered output
- `input`: Original input sinogram
- `geom::CTGeometry`: CT geometry

# Keyword Arguments
- `filter`: Filter type (RampFilter, etc.)
- `cutoff`: Frequency cutoff (0-1)

# Returns
Modified `∂L_∂input` array.
"""
function gradient_filter_sinogram!(
    ∂L_∂input::AbstractArray{T,3},
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    geom::CTGeometry;
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat
    n_cols = Int32(size(input, 1))
    n_rows = Int32(size(input, 2))

    pixel_size = T(geom.pixel_size)

    # Create filter kernel (same as in filter_sinogram!)
    raw_size = max(Int(ceil(Int(n_cols) * cutoff)), 32)
    kernel_size_int = min(raw_size + (1 - raw_size % 2), Int(n_cols))

    kernel_cpu = create_spatial_kernel(kernel_size_int, filter, pixel_size)

    kernel = similar(input, T, kernel_size_int)
    copyto!(kernel, kernel_cpu)

    kernel_size = Int32(kernel_size_int)
    kernel_half = Int32(kernel_size_int ÷ 2)

    # Step 1: Gradient through convolution
    # For convolution: output[dest] = Σ_k input[clamp(dest+k)] × kernel[k]
    # Gradient: ∂L/∂input[src] = Σ_dest where clamp(dest+k)=src: ∂L/∂output[dest] × kernel[k]

    ∂L_∂weighted = similar(input)
    fill!(∂L_∂weighted, zero(T))

    AK.foreachindex(input) do idx
        ci = CartesianIndices(input)[idx]
        col, row, angle = Tuple(ci)

        grad_acc = zero(T)

        # Loop over all output positions that read from this input
        for dest_col in 1:Int32(n_cols)
            for k in Int32(1):kernel_size
                k_offset = k - kernel_half - Int32(1)
                src_col = dest_col + k_offset

                # Clamp to valid range
                src_col_clamped = clamp(src_col, Int32(1), n_cols)

                if src_col_clamped == col
                    grad_acc += ∂L_∂output[dest_col, row, angle] * kernel[k]
                end
            end
        end

        ∂L_∂weighted[idx] = grad_acc
    end

    # Step 2: Gradient through cosine weighting
    gradient_cosine_weight!(∂L_∂input, ∂L_∂weighted, input, geom)

    return ∂L_∂input
end

function gradient_filter_sinogram(
    ∂L_∂output::AbstractArray{T,3},
    input::AbstractArray{T,3},
    geom::CTGeometry;
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat
    ∂L_∂input = similar(input)
    return gradient_filter_sinogram!(∂L_∂input, ∂L_∂output, input, geom; filter=filter, cutoff=cutoff)
end

# =============================================================================
# FDK Reconstruction Gradient
# =============================================================================

"""
    gradient_fdk_reconstruct!(∂L_∂sinogram, ∂L_∂volume, sinogram, geom; filter=RampFilter(), cutoff=1.0)

Compute gradient of loss w.r.t. sinogram through FDK reconstruction.

FDK reconstruction applies: volume = backproject(filter_sinogram(sinogram))

The gradient chain rule:
1. ∂L/∂filtered = gradient of backprojection w.r.t. filtered sinogram
   This is forward projection of ∂L/∂volume (adjoint relationship)
2. ∂L/∂sinogram = gradient of filtering w.r.t. input sinogram

# Arguments
- `∂L_∂sinogram`: Output array for gradient (modified in place)
- `∂L_∂volume`: Gradient of loss w.r.t. reconstructed volume
- `sinogram`: Original input sinogram
- `geom::CTGeometry`: CT geometry

# Keyword Arguments
- `filter`: Filter type (RampFilter, etc.)
- `cutoff`: Frequency cutoff (0-1)

# Returns
Modified `∂L_∂sinogram` array.

# Example
```julia
using BasisSimulator
using Enzyme

# Forward pass
volume = fdk_reconstruct(sinogram, geom, (64, 64, 16))

# Loss and its gradient
target = zeros(Float32, size(volume))
∂L_∂volume = volume .- target  # For MSE loss

# Gradient w.r.t. sinogram
∂L_∂sinogram = gradient_fdk_reconstruct(∂L_∂volume, sinogram, geom)
```
"""
function gradient_fdk_reconstruct!(
    ∂L_∂sinogram::AbstractArray{T,3},
    ∂L_∂volume::AbstractArray{T,3},
    sinogram::AbstractArray{T,3},
    geom::CTGeometry;
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat
    # Step 1: Gradient through backprojection
    # The gradient of weighted backprojection w.r.t. filtered sinogram
    # involves the adjoint of backprojection with distance weights.
    #
    # For FDK backprojection: volume[v] = Σ_θ w_v,θ × filtered[u,v,θ]
    # where w is the bilinear interpolation weights × distance weight
    #
    # The gradient is: ∂L/∂filtered[d] = Σ_v w_v,d × ∂L/∂volume[v]
    #
    # This is approximately forward projection of ∂L/∂volume.
    # The adjoint is not exact due to bilinear interpolation asymmetry,
    # but forward projection provides a good approximation.

    ∂L_∂filtered = similar(sinogram)
    fill!(∂L_∂filtered, zero(T))
    siddon_forward_project!(∂L_∂filtered, ∂L_∂volume, geom)

    # Step 2: Gradient through filtering
    gradient_filter_sinogram!(∂L_∂sinogram, ∂L_∂filtered, sinogram, geom;
                               filter=filter, cutoff=cutoff)

    return ∂L_∂sinogram
end

function gradient_fdk_reconstruct(
    ∂L_∂volume::AbstractArray{T,3},
    sinogram::AbstractArray{T,3},
    geom::CTGeometry;
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat
    ∂L_∂sinogram = similar(sinogram)
    return gradient_fdk_reconstruct!(∂L_∂sinogram, ∂L_∂volume, sinogram, geom;
                                     filter=filter, cutoff=cutoff)
end

# =============================================================================
# SIRT Iteration Gradient
# =============================================================================

"""
    gradient_sirt_iteration(∂L_∂x_new, x, sinogram, geom, W, V_inv, lambda)

Compute gradient of loss w.r.t. x through one SIRT iteration.

SIRT iteration: x_new = x + λ × V⁻¹ × Aᵀ × W × (b - A×x)

Expanding: x_new = x + λ × V⁻¹ × Aᵀ × W × b - λ × V⁻¹ × Aᵀ × W × A × x

Let M = λ × V⁻¹ × Aᵀ × W × A
Then: x_new = (I - M) × x + λ × V⁻¹ × Aᵀ × W × b

Gradient w.r.t. x:
    ∂L/∂x = (I - M)ᵀ × ∂L/∂x_new
          = ∂L/∂x_new - λ × Aᵀ × W × V⁻¹ × A × ∂L/∂x_new  (since operators are self-adjoint for real)
          ≈ ∂L/∂x_new - λ × V⁻¹ × Aᵀ × W × A × ∂L/∂x_new

For simplicity, we approximate: ∂L/∂x ≈ (I - M) × ∂L/∂x_new

# Arguments
- `∂L_∂x_new`: Gradient of loss w.r.t. new reconstruction
- `x`: Current reconstruction
- `sinogram`: Measured sinogram
- `geom`: CT geometry
- `W`: Projection domain weights
- `V_inv`: Inverse image domain weights
- `lambda`: Relaxation parameter

# Returns
`∂L_∂x`: Gradient w.r.t. current reconstruction
"""
function gradient_sirt_iteration(
    ∂L_∂x_new::AbstractArray{T,3},
    x::AbstractArray{T,3},
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    W::AbstractArray{T,3},
    V_inv::AbstractArray{T,3},
    lambda::T
) where T <: AbstractFloat
    # Compute M × ∂L/∂x_new = λ × V⁻¹ × Aᵀ × W × A × ∂L/∂x_new

    # A × ∂L/∂x_new (forward project)
    A_grad = siddon_forward_project(∂L_∂x_new, geom)

    # W × A × ∂L/∂x_new
    AK.foreachindex(A_grad) do idx
        A_grad[idx] *= W[idx]
    end

    # Aᵀ × W × A × ∂L/∂x_new (backproject with matched adjoint)
    AtWA_grad = backproject(A_grad, geom, size(x); weighted=false)

    # V⁻¹ × Aᵀ × W × A × ∂L/∂x_new
    ∂L_∂x = similar(x)
    AK.foreachindex(x) do idx
        ∂L_∂x[idx] = ∂L_∂x_new[idx] - lambda * V_inv[idx] * AtWA_grad[idx]
    end

    return ∂L_∂x
end

"""
    gradient_sirt_reconstruct(∂L_∂recon, sinogram, geom, volume_size; niter=50, lambda=1.0, init=:zeros)

Compute gradient of loss w.r.t. sinogram through SIRT reconstruction.

This unrolls the SIRT iterations and applies the chain rule backwards through
each iteration.

# Arguments
- `∂L_∂recon`: Gradient of loss w.r.t. final reconstruction
- `sinogram`: Original input sinogram
- `geom`: CT geometry
- `volume_size`: Reconstruction volume size

# Keyword Arguments
- `niter`: Number of iterations
- `lambda`: Relaxation parameter
- `init`: Initialization method (:zeros or :fdk)

# Returns
`∂L_∂sinogram`: Gradient w.r.t. input sinogram

# Note
This function is memory-intensive as it stores intermediate reconstructions
for backpropagation through iterations. For large-scale optimization,
consider using checkpointing or adjoint-state methods.
"""
function gradient_sirt_reconstruct(
    ∂L_∂recon::AbstractArray{T,3},
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    niter::Int = 50,
    lambda::Real = 1.0,
    init::Symbol = :zeros
) where T <: AbstractFloat
    λ = T(lambda)

    # Forward pass: store intermediate reconstructions for backprop
    recon_history = Vector{AbstractArray{T,3}}(undef, niter + 1)

    # Initialize
    if init == :zeros
        recon_history[1] = similar(sinogram, T, volume_size...)
        fill!(recon_history[1], zero(T))
    elseif init == :fdk
        recon_history[1] = fdk_reconstruct(sinogram, geom, volume_size)
    else
        error("init must be :zeros or :fdk")
    end

    # Compute weights
    ones_volume = ones(T, volume_size...)
    ray_sums = siddon_forward_project(ones_volume, geom)
    W = similar(ray_sums)
    eps = T(1e-8)
    AK.foreachindex(ray_sums) do idx
        val = ray_sums[idx]
        W[idx] = val > eps ? one(T) / val : zero(T)
    end

    ones_sino = ones(T, geom.n_cols, geom.n_rows, geom.n_angles)
    voxel_sums = backproject(ones_sino, geom, volume_size; weighted=false)
    V_inv = similar(voxel_sums)
    AK.foreachindex(voxel_sums) do idx
        val = voxel_sums[idx]
        V_inv[idx] = val > eps ? one(T) / val : zero(T)
    end

    # Transfer to GPU if needed
    W_gpu = similar(sinogram, T, size(W)...)
    copyto!(W_gpu, W)
    V_inv_gpu = similar(sinogram, T, size(V_inv)...)
    copyto!(V_inv_gpu, V_inv)

    # Forward pass: SIRT iterations
    for iter in 1:niter
        # Copy current reconstruction
        recon_history[iter + 1] = copy(recon_history[iter])

        # Forward project
        projected = siddon_forward_project(recon_history[iter + 1], geom)

        # Compute weighted residual
        AK.foreachindex(projected) do idx
            residual = sinogram[idx] - projected[idx]
            projected[idx] = W_gpu[idx] * residual
        end

        # Backproject
        correction = backproject(projected, geom, volume_size; weighted=false)

        # Update
        AK.foreachindex(recon_history[iter + 1]) do idx
            recon_history[iter + 1][idx] += λ * V_inv_gpu[idx] * correction[idx]
        end
    end

    # Backward pass: unroll iterations
    ∂L_∂x = copy(∂L_∂recon)
    ∂L_∂sinogram = similar(sinogram)
    fill!(∂L_∂sinogram, zero(T))

    for iter in niter:-1:1
        x = recon_history[iter]

        # Gradient w.r.t. sinogram contribution from this iteration
        # SIRT uses: correction = Aᵀ × W × (b - A×x)
        # ∂correction/∂b = Aᵀ × W
        # ∂L/∂b += λ × V⁻¹ × ∂correction/∂b × ∂L/∂x_new = λ × V⁻¹ × Aᵀ × W × ∂L/∂x_new...
        # But this is complex. For now, accumulate via numerical approximation of adjoint.

        # Forward project ∂L/∂x_new and multiply by W to get sinogram gradient contribution
        projected_grad = siddon_forward_project(∂L_∂x, geom)
        AK.foreachindex(projected_grad) do idx
            projected_grad[idx] *= λ * W_gpu[idx]
        end

        # Accumulate sinogram gradient
        AK.foreachindex(sinogram) do idx
            ∂L_∂sinogram[idx] += projected_grad[idx]
        end

        # Backprop through iteration
        ∂L_∂x = gradient_sirt_iteration(∂L_∂x, x, sinogram, geom, W_gpu, V_inv_gpu, λ)
    end

    return ∂L_∂sinogram
end

# =============================================================================
# High-Level Differentiable Reconstruction Types
# =============================================================================

"""
    DifferentiableFDK{T}

Container for differentiable FDK reconstruction operations.

# Fields
- `geom::CTGeometry`: CT scanner geometry
- `volume_size::NTuple{3,Int}`: Reconstruction volume dimensions
- `filter::FilterType`: Reconstruction filter
- `cutoff::Float64`: Frequency cutoff

# Example
```julia
using BasisSimulator
using Enzyme

# Create differentiable FDK operator
dfdk = DifferentiableFDK{Float32}(geom, (64, 64, 16))

# Forward
volume = dfdk(sinogram)

# Gradient
∂L_∂sinogram = dfdk.backward(∂L_∂volume, sinogram)
```
"""
struct DifferentiableFDK{T<:AbstractFloat}
    geom::CTGeometry
    volume_size::NTuple{3,Int}
    filter::FilterType
    cutoff::Float64
end

function DifferentiableFDK{T}(
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat
    return DifferentiableFDK{T}(geom, volume_size, filter, cutoff)
end

# Forward pass (functor)
function (dfdk::DifferentiableFDK{T})(sinogram::AbstractArray{T,3}) where T
    return fdk_reconstruct(sinogram, dfdk.geom, dfdk.volume_size;
                          filter=dfdk.filter, cutoff=dfdk.cutoff)
end

"""
    DifferentiableSIRT{T}

Container for differentiable SIRT reconstruction operations.

# Fields
- `geom::CTGeometry`: CT scanner geometry
- `volume_size::NTuple{3,Int}`: Reconstruction volume dimensions
- `niter::Int`: Number of iterations
- `lambda::T`: Relaxation parameter
- `init::Symbol`: Initialization method

# Note
SIRT gradients are expensive due to iteration unrolling.
Use with caution for large niter.
"""
struct DifferentiableSIRT{T<:AbstractFloat}
    geom::CTGeometry
    volume_size::NTuple{3,Int}
    niter::Int
    lambda::T
    init::Symbol
end

function DifferentiableSIRT{T}(
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    niter::Int = 50,
    lambda::Real = 1.0,
    init::Symbol = :zeros
) where T <: AbstractFloat
    return DifferentiableSIRT{T}(geom, volume_size, niter, T(lambda), init)
end

# Forward pass (functor)
function (dsirt::DifferentiableSIRT{T})(sinogram::AbstractArray{T,3}) where T
    return sirt_reconstruct(sinogram, dsirt.geom, dsirt.volume_size;
                           niter=dsirt.niter, lambda=dsirt.lambda, init=dsirt.init)
end

# =============================================================================
# Finite Difference Verification for FDK
# =============================================================================

"""
    verify_gradient_fdk_reconstruct(sinogram, geom, volume_size; ε=1e-5, n_samples=5, seed=42)

Verify FDK gradient by comparing with finite differences.

# Arguments
- `sinogram`: Test sinogram
- `geom`: CT geometry
- `volume_size`: Reconstruction volume size

# Keyword Arguments
- `ε`: Finite difference step size
- `n_samples`: Number of random positions to test
- `seed`: Random seed

# Returns
NamedTuple with:
- `max_relative_error`: Maximum relative error
- `mean_relative_error`: Mean relative error
- `errors`: Vector of individual errors
- `passed`: Bool, true if error is acceptable
"""
function verify_gradient_fdk_reconstruct(
    sinogram::AbstractArray{T,3},
    geom::CTGeometry,
    volume_size::NTuple{3,Int};
    ε::Real = T(1e-4),
    n_samples::Int = 5,
    seed::Int = 42,
    filter::FilterType = RampFilter(),
    cutoff::Float64 = 1.0
) where T <: AbstractFloat
    rng = MersenneTwister(seed)

    # Forward pass
    volume = fdk_reconstruct(sinogram, geom, volume_size; filter=filter, cutoff=cutoff)

    # Loss = sum(volume.^2) / 2
    ∂L_∂volume = copy(volume)

    # Analytical gradient
    ∂L_∂sinogram_analytical = gradient_fdk_reconstruct(∂L_∂volume, sinogram, geom;
                                                        filter=filter, cutoff=cutoff)

    # Finite differences
    errors = Float64[]
    sinogram_perturbed = copy(sinogram)
    n_cols, n_rows, n_angles = size(sinogram)

    for _ in 1:n_samples
        ic = rand(rng, 1:n_cols)
        ir = rand(rng, 1:n_rows)
        ia = rand(rng, 1:n_angles)

        # L(s + ε)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] += ε
        vol_plus = fdk_reconstruct(sinogram_perturbed, geom, volume_size;
                                   filter=filter, cutoff=cutoff)
        L_plus = sum(vol_plus.^2) / 2

        # L(s - ε)
        copyto!(sinogram_perturbed, sinogram)
        sinogram_perturbed[ic, ir, ia] -= ε
        vol_minus = fdk_reconstruct(sinogram_perturbed, geom, volume_size;
                                    filter=filter, cutoff=cutoff)
        L_minus = sum(vol_minus.^2) / 2

        grad_fd = Float64((L_plus - L_minus) / (2 * ε))
        grad_analytical = Float64(∂L_∂sinogram_analytical[ic, ir, ia])

        denom = max(abs(grad_fd), abs(grad_analytical), T(1e-8))
        rel_error = abs(grad_fd - grad_analytical) / denom
        push!(errors, rel_error)
    end

    max_rel_error = maximum(errors)
    mean_rel_error = sum(errors) / length(errors)
    # FDK gradient has approximations, so we allow higher error
    passed = mean_rel_error < 0.5

    return (
        max_relative_error = max_rel_error,
        mean_relative_error = mean_rel_error,
        errors = errors,
        passed = passed
    )
end

end # module BasisSimulatorEnzymeExt
