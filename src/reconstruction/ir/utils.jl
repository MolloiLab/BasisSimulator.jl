# =============================================================================
# Iterative Reconstruction Utilities
# =============================================================================
#
# Shared components for iterative reconstruction algorithms (HIR, etc.):
#   - Huber penalty (edge-preserving regularization)
#   - Projection/image domain weight computation
#
# GPU-native via AcceleratedKernels.jl.
# =============================================================================

import AcceleratedKernels as AK

export PenaltyType, HuberPenalty
export compute_huber_penalty, compute_huber_gradient!
export compute_projection_weights, compute_image_weights
export create_ordered_subsets, create_subset_geometry, extract_subset_sinogram

# =============================================================================
# Huber Penalty
# =============================================================================

"""
    PenaltyType

Abstract type for regularization penalties.
"""
abstract type PenaltyType end

"""
    HuberPenalty <: PenaltyType

Huber penalty: edge-preserving quadratic/linear hybrid.

ψ(t) = t²/2           if |t| ≤ δ
ψ(t) = δ|t| - δ²/2    if |t| > δ
"""
struct HuberPenalty <: PenaltyType
    delta::Float32
end

HuberPenalty() = HuberPenalty(0.01f0)

@inline function _huber(t::T, δ::T) where T
    abs_t = abs(t)
    if abs_t ≤ δ
        return t * t / T(2)
    else
        return δ * abs_t - δ * δ / T(2)
    end
end

@inline function _huber_deriv(t::T, δ::T) where T
    abs_t = abs(t)
    if abs_t ≤ δ
        return t
    else
        return δ * sign(t)
    end
end

"""
    compute_huber_penalty(x, delta)

Compute Huber penalty value over 3D volume with 6-connected neighborhood.
"""
function compute_huber_penalty(
    x::AbstractArray{T, 3},
    delta::Real
) where T <: AbstractFloat

    δ = T(delta)
    nx, ny, nz = size(x)
    penalty_vals = similar(x)
    backend = AK.get_backend(x)

    AK.foreachindex(penalty_vals, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        val = x[i, j, k]
        penalty = zero(T)

        if i < nx
            diff = x[i+1, j, k] - val
            penalty += _huber(diff, δ)
        end
        if j < ny
            diff = x[i, j+1, k] - val
            penalty += _huber(diff, δ)
        end
        if k < nz
            diff = x[i, j, k+1] - val
            penalty += _huber(diff, δ)
        end

        penalty_vals[linear_idx] = penalty
    end

    return AK.mapreduce(identity, +, penalty_vals; init=zero(T))
end

"""
    compute_huber_gradient!(grad, x, delta)

Compute gradient of the Huber edge-preserving roughness penalty in-place,
on a 6-connected (face-neighbor) finite-difference stencil.

# Reference
Direct GPU port of Fessler MIRT's potential-derivative + finite-difference
combination:
- `penalty/huber_dpot.m` — `g(t,δ) = t` for `|t| ≤ δ`, else `δ·sign(t)`
  (Huber 1964).
- `penalty/Cdiffs.m` — stacked first-difference operator with face-neighbor
  offsets — what we apply by hand here in 6-connected form.

For the surrounding OS-PWLS hot path that consumes this gradient see
`reconstruct!(::HIRReconWorkspace, …)` in `src/api/driver.jl`.
"""
function compute_huber_gradient!(
    grad::AbstractArray{T, 3},
    x::AbstractArray{T, 3},
    delta::Real
) where T <: AbstractFloat

    δ = T(delta)
    nx, ny, nz = size(x)
    backend = AK.get_backend(x)

    AK.foreachindex(grad, backend) do linear_idx
        i = mod1(linear_idx, nx)
        j = mod1(div(linear_idx - 1, nx) + 1, ny)
        k = div(linear_idx - 1, nx * ny) + 1

        val = x[i, j, k]
        g = zero(T)

        if i < nx
            diff = x[i+1, j, k] - val
            g -= _huber_deriv(diff, δ)
        end
        if j < ny
            diff = x[i, j+1, k] - val
            g -= _huber_deriv(diff, δ)
        end
        if k < nz
            diff = x[i, j, k+1] - val
            g -= _huber_deriv(diff, δ)
        end

        if i > 1
            diff = val - x[i-1, j, k]
            g += _huber_deriv(diff, δ)
        end
        if j > 1
            diff = val - x[i, j-1, k]
            g += _huber_deriv(diff, δ)
        end
        if k > 1
            diff = val - x[i, j, k-1]
            g += _huber_deriv(diff, δ)
        end

        grad[linear_idx] = g
    end

    return grad
end

# =============================================================================
# Projection / Image Domain Weights
# =============================================================================

# `like` is a prototype array that fixes the compute backend.  Both weights are
# a full forward / back projection, so running them on the host when the caller
# holds device arrays costs minutes (measured 62.9 s + 20.3 s at 834×8×500 into
# 512×512×8) to produce arrays that are immediately copied to the device anyway.
# `similar(like, …)` keeps the projection on whatever backend owns the data.
# The operators are unchanged — same kernels, same weights, just not on the CPU.
@inline _ones_like(::Nothing, ::Type{T}, dims...) where {T} = ones(T, dims...)
@inline _ones_like(like::AbstractArray, ::Type{T}, dims...) where {T} =
    fill!(similar(like, T, dims...), one(T))

"""
    compute_projection_weights(geom, volume_size, T; projector, like) -> sinogram-shaped weights

Compute W = 1 / (A · 1). Normalizes for ray length differences.

`like` is an optional prototype array selecting the backend the projection runs
on (defaults to the CPU).  Pass the caller's sinogram to keep this on the GPU.
"""
function compute_projection_weights(
    geom::CTGeometry,
    volume_size::NTuple{3, Int},
    ::Type{T};
    projector::Symbol = :dd_fast,
    like::Union{Nothing, AbstractArray} = nothing,
    circular_support::Bool = false,
) where T <: AbstractFloat
    ones_volume = _ones_like(like, T, volume_size...)
    circular_support && apply_fov_mask!(ones_volume, geom; sentinel_μ = zero(T))
    ray_sums = _project_mono(projector, ones_volume, geom)
    eps = T(1e-8)
    AK.foreachindex(ray_sums, AK.get_backend(ray_sums)) do idx
        val = ray_sums[idx]
        ray_sums[idx] = val > eps ? one(T) / val : zero(T)
    end
    return ray_sums
end

"""
    compute_image_weights(geom, volume_size, T; like) -> volume-shaped weights

Compute V = 1 / (Aᵀ · 1). Accounts for non-uniform voxel sensitivity.

`like` selects the backend, as in [`compute_projection_weights`](@ref).
"""
function compute_image_weights(
    geom::CTGeometry,
    volume_size::NTuple{3, Int},
    ::Type{T};
    like::Union{Nothing, AbstractArray} = nothing,
    projector::Symbol = :dd_fast,
    active_z::Union{Nothing, UnitRange{Int}} = nothing,
    circular_support::Bool = false,
) where T <: AbstractFloat
    ones_sino = _ones_like(like, T, geom.n_cols, geom.n_rows, geom.n_angles)
    voxel_sums = similar(ones_sino, T, volume_size...)
    fill!(voxel_sums, zero(T))
    if projector === :siddon
        _backproject_mono!(projector, voxel_sums, ones_sino, geom)
    else
        _backproject_mono!(projector, voxel_sums, ones_sino, geom;
                           active_z, circular_support)
    end
    eps = T(1e-8)
    AK.foreachindex(voxel_sums, AK.get_backend(voxel_sums)) do idx
        val = voxel_sums[idx]
        voxel_sums[idx] = val > eps ? one(T) / val : zero(T)
    end
    return voxel_sums
end

# =============================================================================
# Ordered Subsets
# =============================================================================

"""
    create_ordered_subsets(n_angles, n_subsets) -> Vector{Vector{Int}}

Distribute angles across subsets for maximum angular separation.
"""
function create_ordered_subsets(n_angles::Int, n_subsets::Int)
    subsets = [Int[] for _ in 1:n_subsets]
    for i in 1:n_angles
        subset_idx = mod1(i, n_subsets)
        push!(subsets[subset_idx], i)
    end
    return subsets
end

"""
    create_subset_geometry(geom, angle_indices) -> CTGeometry

Create geometry for a subset of projection angles.
"""
function create_subset_geometry(geom::CTGeometry, angle_indices::Vector{Int})
    n_subset = length(angle_indices)
    return CTGeometry(
        geom.SAD, geom.SDD, n_subset, geom.n_rows, geom.n_cols,
        geom.pixel_size, geom.pixel_row_size,
        geom.angles[angle_indices],
        geom.source_positions[:, angle_indices],
        geom.detector_centers[:, angle_indices],
        geom.detector_u[:, angle_indices],
        geom.detector_v[:, angle_indices],
        geom.fov, geom.pitch, geom.table_feed, geom.detector_shape
    )
end

"""
    extract_subset_sinogram(sinogram, angle_indices) -> subset sinogram

Extract sinogram views for given angle indices.
"""
function extract_subset_sinogram(
    sinogram::AbstractArray{T, 3},
    angle_indices::Vector{Int}
) where T <: AbstractFloat
    n_cols, n_rows, _ = size(sinogram)
    subset = similar(sinogram, T, n_cols, n_rows, length(angle_indices))
    for (i, idx) in enumerate(angle_indices)
        subset[:, :, i] = sinogram[:, :, idx]
    end
    return subset
end
