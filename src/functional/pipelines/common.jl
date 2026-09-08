"""
    AbstractPipeline{T}

Common supertype of the five-struct pipelines ([`EICTPipeline`](@ref),
[`PCCTPipeline`](@ref)): geometry, volume shape/extent, material count, recon
shape, and the compiled-loop batching (`batching`, `unrolled`).
"""
abstract type AbstractPipeline{T <: AbstractFloat} end

# =============================================================================
# Shared pipeline machinery: material fractions → per-material path lengths,
# the memory-budget batching that sizes every compiled view loop, and the plan
# lifting hooks that move host plan tensors into the traced program.
# =============================================================================

"""
    onehot_fractions(mask, n_mat; T = Float32) -> Array{T,4}

HOST helper: 0-based material label mask `(nx, ny, nz)` → material-fraction
tensor `(nx, ny, nz, n_mat)` with `fractions[:, :, :, m] = (mask .== m-1)`.
This is the differentiable parameterization of the phantom (labeled phantoms are
its one-hot special case; XCIST/XCAT volume-fraction phantoms map onto it directly).
"""
function onehot_fractions(mask::AbstractArray{<:Integer, 3}, n_mat::Integer; T::Type{<:AbstractFloat} = Float32)
    mask_h = Array(mask)
    out = Array{T}(undef, size(mask_h)..., n_mat)
    for m in 1:n_mat
        out[:, :, :, m] = T.(mask_h .== (m - 1))
    end
    return out
end

# stack N-D arrays along a new trailing axis with `cat` (traceable; `stack` is not).
_cat_new_axis(xs) = cat(map(x -> reshape(x, size(x)..., 1), xs)...; dims = ndims(xs[1]) + 1)

"""
    material_paths(fractions, pipe) -> (n_col, n_row, n_view, n_mat)

Per-material path lengths `P[:, :, :, m] = A(fractions[:, :, :, m])` with the
functional DD projector (one static-tap projection per material per view; the
material axis is the batch axis of the linear operator). Differentiable in
`fractions`.
"""
function material_paths(fractions::AbstractArray{<:Any, 4}, pipe::AbstractPipeline{T}) where {T}
    geom = pipe.geom
    if !pipe.unrolled
        runs = dd_run_plans(geom, pipe.vol_shape; view_batch = pipe.batching.dd,
            volume_extent = pipe.volume_extent, eltype = T)
        return reduce((a, b) -> cat(a, b; dims = 3), [dd_project_run(fractions, r) for r in runs])
    end
    plans = [dd_view_plan(geom, v, pipe.vol_shape; volume_extent = pipe.volume_extent, eltype = T)
             for v in 1:geom.n_angles]
    per_mat = [_cat_new_axis([dd_project_view(fractions[:, :, :, m], p) for p in plans])
               for m in 1:size(fractions, 4)]
    return _cat_new_axis(per_mat)
end


"""
    _auto_batching(sino_shape, vol_shape, n_E, n_mat, recon_shape, budget_mb) -> (dd, spectral, fdk)

Views per compiled loop iteration for each stage so that the stage's per-batch
transient stays within `budget_mb` (host estimate of the live Float32 tensors:
DD gather chain `n_col·n_row·n_long·(6 + n_mat)`, spectral sum
`n_col·n_row·n_E·3`, FDK `nx·ny·nz·10` per view).  The program size is then
set by the budget, not by the scan (any number of views, any phantom grid).
"""
function _auto_batching(sino_shape::NTuple{3, Int}, vol_shape::NTuple{3, Int}, n_E::Int, n_mat::Int,
        recon_shape::NTuple{3, Int}, budget_mb::Real)
    n_col, n_row, n_view = sino_shape
    n_long = max(vol_shape[1], vol_shape[2])
    nx, ny, nz = recon_shape
    bytes = budget_mb * 2^20
    per_view(x) = Int(clamp(bytes ÷ (4 * x), 1, n_view))
    return (dd = per_view(n_col * n_row * n_long * (6 + n_mat)),
            spectral = per_view(n_col * n_row * n_E * 3),
            fdk = per_view(nx * ny * nz * 10))
end

function _eict_on_device(p::EICTPlan{T}, ref) where {T}
    μ = _on_device(p.μ_tbl, ref); w = _on_device(p.wη, ref); b = _on_device(p.bt, ref)
    a = _on_device(p.air_ref, ref); hc = _on_device(p.scatter_Hc, ref); hr = _on_device(p.scatter_Hr, ref)
    bc = _on_device(p.bhc_coeffs, ref)
    return EICTPlan{T, typeof(μ), typeof(w), typeof(b), typeof(a), typeof(p.ff_log), typeof(hc), typeof(hr), typeof(bc)}(
        μ, w, b, a, p.I0, p.σ_e, p.use_noise, p.use_enoise, p.ff_log, hc, hr, p.scatter_C, p.scatter_sw,
        bc, p.eps, p.sino_shape, p.energies)
end

_fbp_on_device(p::FBPPlan, ref) = FBPPlan(p; tensors = map(x -> _on_device(x, ref), p.tensors))

