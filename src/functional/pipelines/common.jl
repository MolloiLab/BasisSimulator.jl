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
# A pipeline input: dense per-material fractions (nx, ny, nz, n_mat) — the differentiable form, e.g. the
# K basis materials of a decomposition — or the label volume itself (nx, ny, nz) Integer, whose one-hot
# fractions are formed on device one material batch at a time (never all materials at once).
const PipelineInput = Union{AbstractArray{<:Any, 4}, AbstractArray{<:Integer, 3}}

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
function material_paths(labels::AbstractArray{<:Integer, 3}, pipe::AbstractPipeline{T}) where {T}
    size(labels) == pipe.vol_shape || throw(DimensionMismatch("labels $(size(labels)) do not match the pipeline's phantom $(pipe.vol_shape)"))
    n_mat, mb = pipe.n_mat, pipe.batching.mat
    parts = map(1:mb:n_mat) do lo                                     # static material batches, one one-hot chunk on device at a time
        ms = lo:min(lo + mb - 1, n_mat)
        fr = _cat_new_axis([ifelse.(labels .== (m - 1), one(T), zero(T)) for m in ms])
        material_paths(fr, pipe)
    end
    return reduce((a, b) -> cat(a, b; dims = 4), parts)
end

function material_paths(fractions::AbstractArray{<:Any, 4}, pipe::AbstractPipeline{T}) where {T}
    geom = pipe.geom                                   # linear per material: any number of materials projects (the chain needs the plan's n_mat)
    if !pipe.unrolled && pipe.batching.dense
        runs = dd_run_plans(geom, pipe.vol_shape; view_batch = pipe.batching.dd,
            volume_extent = pipe.volume_extent, eltype = T)
        cb, sb = pipe.batching.col_block, pipe.batching.slab_block
        windowed = cb < geom.n_cols || sb < max(pipe.vol_shape[1], pipe.vol_shape[2])
        proj = windowed ?
            (r -> dd_project_dense_windowed_run(fractions, r, geom; col_block = cb, slab_block = sb, volume_extent = pipe.volume_extent, unroll = !pipe.loop)) :
            (r -> dd_project_dense_run(fractions, r; unroll = !pipe.loop))
        return reduce((a, b) -> cat(a, b; dims = 3), [proj(r) for r in runs])
    elseif !pipe.unrolled && pipe.loop
        runs = dd_run_plans(geom, pipe.vol_shape; view_batch = pipe.batching.dd,
            volume_extent = pipe.volume_extent, eltype = T)
        return reduce((a, b) -> cat(a, b; dims = 3), [dd_project_run(fractions, r) for r in runs])
    elseif !pipe.unrolled
        # gather formulation, unrolled batches: one static program per batch
        batches = dd_batch_plans(geom, pipe.vol_shape; view_batch = pipe.batching.dd,
            volume_extent = pipe.volume_extent, eltype = T)
        per_mat = [reduce((a, b) -> cat(a, b; dims = 3), [dd_project_batch(fractions[:, :, :, m], bp) for bp in batches])
                   for m in 1:size(fractions, 4)]
        return _cat_new_axis(per_mat)
    end
    plans = [dd_view_plan(geom, v, pipe.vol_shape; volume_extent = pipe.volume_extent, eltype = T)
             for v in 1:geom.n_angles]
    per_mat = [_cat_new_axis([dd_project_view(fractions[:, :, :, m], p) for p in plans])
               for m in 1:size(fractions, 4)]
    return _cat_new_axis(per_mat)
end


"""
    _auto_batching(sino_shape, vol_shape, n_E, n_mat, recon_shape, budget_mb) -> (dd, spectral, fdk, dense, col_block, slab_block, mat)

Views per compiled loop iteration for each stage so that the stage's per-batch
transient stays within `budget_mb` (host estimate of the live Float32 tensors:
DD gather chain `n_col·n_row·n_long·(6 + n_mat)`, spectral sum
`n_col·n_row·n_E·3`, FDK `nx·ny·nz·10` per view).  The program size is then
set by the budget, not by the scan (any number of views, any phantom grid).
"""
function _auto_batching(sino_shape::NTuple{3, Int}, vol_shape::NTuple{3, Int}, n_E::Int, n_mat::Int,
        recon_shape::NTuple{3, Int}, budget_mb::Real; projector::Symbol = :dense, geom = nothing, volume_extent = nothing)
    n_col, n_row, n_view = sino_shape
    n_long = max(vol_shape[1], vol_shape[2]); n_t = n_long; nzv = vol_shape[3]
    nx, ny, nz = recon_shape
    bytes = budget_mb * 2^20
    per_view(x) = Int(clamp(bytes ÷ (4 * x), 1, n_view))
    dense = projector !== :gather                      # the dense contractions ARE the projector; :gather is an explicit choice
    col_block, slab_block = n_col, n_long              # full dense = one block
    view_bytes = dd_dense_view_bytes(n_col, n_row, n_t, nzv, n_long, n_mat)
    if dense && view_bytes > bytes
        # windowed dense: shrink the blocks (columns first, then slabs) until one view fits the budget
        geom === nothing && throw(ArgumentError("_auto_batching: the geometry is needed to size the windowed projector"))
        found = false
        for sb in (n_long, 256, 128, 64, 32, 16, 8), cb in (n_col, 256, 128, 64, 32, 16)
            (sb <= n_long && cb <= n_col) || continue
            _, widths, _, _ = dd_windows(geom, vol_shape, 1:n_view, cb, sb; volume_extent, eltype = Float32)
            vb = dd_dense_windowed_view_bytes(n_col, n_row, nzv, n_long, n_mat, widths, cb, sb)
            if vb <= bytes
                col_block, slab_block, view_bytes, found = cb, sb, vb, true
                break
            end
        end
        found || throw(ArgumentError("the dense projector needs $(round(Int, view_bytes / 2^20)) MB per view " *
            "(n_cols $n_col × n_t $n_t × n_long $n_long weights, $n_mat channels) and even the narrowest windows " *
            "(16 columns × 8 slabs) exceed batch_budget_mb = $budget_mb; raise batch_budget_mb — there is no silent fallback"))
    end
    dd = dense ? Int(clamp(bytes ÷ view_bytes, 1, n_view)) : per_view(n_col * n_row * n_long * (6 + n_mat))
    # label input: one-hot fractions of `mat` materials at a time (the volume itself, 4 B per voxel per material)
    mat = Int(clamp(bytes ÷ (4 * prod(vol_shape)), 1, n_mat))
    return (dd = dd, spectral = per_view(n_col * n_row * n_E * 3), fdk = per_view(nx * ny * nz * 10), dense = dense,
            col_block = col_block, slab_block = slab_block, mat = mat)
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

