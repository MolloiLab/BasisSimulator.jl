# =============================================================================
# Looped view runs (M5b): program size independent of the number of views.
#
# A RUN is a maximal range of consecutive views with the same `vertical` axis
# choice (≤ 5 per rotation).  Its per-view scalars — exactly the `T`-typed
# fields of `_consts(DDViewPlan, T)` — are stored as an 18 × n_run table that is
# ONE in-graph constant; the run is processed in batches of `B` views by a
# fixed-trip-count loop (`_batched_loop`, a StableHLO `while` under Reactant)
# whose body slices the table (`_dslice`), rebuilds the (1,1,1,B) constants,
# and runs the same tap bodies as the unrolled batch (`_dd_forward_taps` /
# `_dd_transpose_taps`).  A remainder batch (static shape) runs once after the
# loop.  Multiple channels (materials) share one index computation per batch.
# =============================================================================

const _RUN_TABLE_KEYS = (:sx, :sy, :sz, :dcx, :dcy, :dcz, :ux, :uy, :vvz,
    :s_long, :s_tran, :d_tran, :d_long, :u_tran, :u_long, :Lsd, :cin_tran, :cin_long)

"""
    DDRunPlan{T}

Host description of one run of consecutive same-`vertical` views for
[`dd_project_run`](@ref) / [`dd_transpose_run`](@ref): the per-view constant
table, the batch size `B`, the static loop structure (`nb` full batches +
`tail`), and the batch-maximal tap counts.  Build with [`dd_run_plans`](@ref).
"""
struct DDRunPlan{T <: AbstractFloat}
    views::UnitRange{Int}
    arc::Bool
    vertical::Bool
    table::Matrix{T}                 # 18 × n_run, rows in `_RUN_TABLE_KEYS` order
    B::Int; nb::Int; tail::Int
    n_cols::Int; n_rows::Int
    SAD::Float64; SDD::Float64
    pixel_size::Float64; pixel_row_size::Float64
    nx::Int; ny::Int; nz::Int
    bounds::NTuple{3, Float64}
    KX::Int; KZ::Int; KXT::Int; KZT::Int
end

"""
    dd_run_plans(geom, vol_shape; view_batch, views = 1:geom.n_angles, volume_extent = nothing, eltype = Float64)

Runs of the view list `views` (consecutive entries with equal `vertical`) in
list order, each processed in batches of at most `view_batch` views.  `views`
may be any ordered subset (e.g. an ordered subset of an iterative
reconstruction): the per-view constants are a table, so contiguity is not
required; `run.views` then holds POSITIONS in the list (`1:n`), and the
concatenated run outputs follow the list order.
"""
function dd_run_plans(geom::CTGeometry, vol_shape::NTuple{3, Int};
        view_batch::Int,
        views::AbstractVector{<:Integer} = 1:geom.n_angles,
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        eltype::Type{T} = Float64) where {T <: AbstractFloat}
    view_batch >= 1 || throw(ArgumentError("view_batch must be ≥ 1, got $view_batch"))
    all(1 .<= views .<= geom.n_angles) || throw(ArgumentError("views must lie in 1:$(geom.n_angles)"))
    n = length(views)
    plans = [dd_view_plan(geom, Int(views[i]), vol_shape; volume_extent, eltype = T) for i in 1:n]
    runs = DDRunPlan{T}[]
    v = 1
    while v <= n
        w = v
        while w < n && plans[w + 1].vertical == plans[v].vertical
            w += 1
        end
        ps = plans[v:w]; nr = w - v + 1; p1 = ps[1]
        table = Matrix{T}(undef, length(_RUN_TABLE_KEYS), nr)
        for (j, p) in enumerate(ps)
            c = _consts(p, T)
            for (i, k) in enumerate(_RUN_TABLE_KEYS)
                table[i, j] = getfield(c, k)
            end
        end
        B = min(view_batch, nr)
        push!(runs, DDRunPlan{T}(v:w, p1.arc, p1.vertical, table, B, nr ÷ B, nr % B,
            p1.n_cols, p1.n_rows, p1.SAD, p1.SDD, p1.pixel_size, p1.pixel_row_size,
            p1.nx, p1.ny, p1.nz, p1.bounds,
            maximum(p.KX for p in ps), maximum(p.KZ for p in ps), maximum(p.KXT for p in ps), maximum(p.KZT for p in ps)))
        v = w + 1
    end
    return runs
end

# `_consts` for a table slice `cs :: (18, B)` (host Matrix or traced): view-independent
# scalars as in `_consts`, per-view rows reshaped to (1,1,1,B).
function _consts_run(p::DDRunPlan{T}, cs) where {T}
    S = T; B = size(cs, 2)
    b = p.bounds
    nx, ny, nz = p.nx, p.ny, p.nz
    vmin_x = S(-b[1] / 2); vmin_y = S(-b[2] / 2); vmin_z = S(-b[3] / 2)
    vx = S(b[1]) / S(nx); vy = S(b[2]) / S(ny); vz = S(b[3]) / S(nz)
    mag = S(p.SDD / p.SAD)
    dγ = S(p.pixel_size / p.SAD)
    ps = S(p.pixel_size); prs = S(p.pixel_row_size)
    cc = (S(p.n_cols) + one(S)) / S(2)
    rc = (S(p.n_rows) + one(S)) / S(2)
    if p.vertical
        n_t = nx; v_t = vx; vmin_t = vmin_x
        n_long = ny; v_long = vy; vmin_long = vmin_y
    else
        n_t = ny; v_t = vy; vmin_t = vmin_y
        n_long = nx; v_long = vx; vmin_long = vmin_x
    end
    row(i) = reshape(cs[i:i, :], 1, 1, 1, B)
    return (; arc = p.arc, half = S(0.5), tiny = S(1.0e-12), two = S(2),
        n_cols_hi = S(p.n_cols + 2), n_rows_hi = S(p.n_rows + 2),
        sx = row(1), sy = row(2), sz = row(3), dcx = row(4), dcy = row(5), dcz = row(6),
        ux = row(7), uy = row(8), vvz = row(9), mag, dγ, ps, prs, cc, rc,
        vmin_x, vmin_y, vmin_z, vx, vy, vz,
        s_long = row(10), s_tran = row(11), d_tran = row(12), d_long = row(13),
        u_tran = row(14), u_long = row(15), Lsd = row(16), cin_tran = row(17), cin_long = row(18),
        n_t, v_t, vmin_t, n_long, v_long, vmin_long, nz,
        n_cols = p.n_cols, n_rows = p.n_rows)
end

"""
    dd_project_run(vol4, run::DDRunPlan) -> (n_cols, n_rows, n_run, n_channels)

Forward projection of every channel of `vol4 :: (nx, ny, nz, n_channels)` over
the run's views, batches of `run.B` views inside one compiled loop.  Numerically
the per-view [`dd_project_view`](@ref) (same per-element arithmetic and tap order).
"""
function dd_project_run(vol::AbstractArray{<:Any, 4}, p::DDRunPlan{T}) where {T <: AbstractFloat}
    size(vol)[1:3] == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("volume $(size(vol)[1:3]) does not match plan $((p.nx, p.ny, p.nz))"))
    M = size(vol, 4)
    Vs = ntuple(m -> p.vertical ? permutedims(vol[:, :, :, m], (1, 3, 2)) : permutedims(vol[:, :, :, m], (2, 3, 1)), M)
    tab = _on_device(p.table, vol)
    nr = length(p.views)
    out = _zeros(vol, T, (p.n_cols, p.n_rows, nr, M))
    KX, KZ, n_cols, n_rows = p.KX, p.KZ, p.n_cols, p.n_rows
    chunk_fn = (start, len, tab, Vs) -> begin                                # closes over host data only
        c = _consts_run(p, _dslice(tab, start, len, 2))
        g = _forward_geometry_from(c, n_cols, n_rows, tab, T)
        parts = _dd_forward_taps(Vs, KX, KZ, g, T)                         # M × (n_cols, n_rows, len)
        # pairwise `cat` (a varargs splat of traced arrays can recurse in the tracer)
        return reduce((a, b) -> cat(a, b; dims = 4), map(x -> reshape(x, size(x)..., 1), parts))   # (n_cols, n_rows, len, M)
    end
    return _loop_over_batches(chunk_fn, nr, p.B, out, (tab, Vs), vol)
end

"""
    dd_transpose_run(sino_run, run::DDRunPlan, vol_shape) -> (nx, ny, nz)

Sum over the run's views of [`dd_transpose_view`](@ref) for
`sino_run :: (n_cols, n_rows, n_run)`, batches of `run.B` views inside one
compiled loop.
"""
function dd_transpose_run(sino::AbstractArray{<:Any, 3}, p::DDRunPlan{T}, vol_shape::NTuple{3, Int}) where {T <: AbstractFloat}
    nr = length(p.views)
    size(sino) == (p.n_cols, p.n_rows, nr) ||
        throw(DimensionMismatch("sinogram run $(size(sino)) does not match plan $((p.n_cols, p.n_rows, nr))"))
    vol_shape == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("vol_shape $(vol_shape) does not match plan $((p.nx, p.ny, p.nz))"))
    n_t, n_long = p.vertical ? (p.nx, p.ny) : (p.ny, p.nx)
    tab = _on_device(p.table, sino)
    block = p.n_cols * p.n_rows
    acc0 = _zeros(sino, T, (n_t, p.nz, n_long))
    KXT, KZT = p.KXT, p.KZT
    chunk_fn = (start, len, tab, sino) -> begin                              # closes over host data only
        c = _consts_run(p, _dslice(tab, start, len, 2))
        off = _on_device(reshape(Int32[(b - 1) * block for b in 1:len], 1, 1, 1, len), sino)
        return _dd_transpose_taps(_dslice(sino, start, len, 3), c, KXT, KZT, off, sino, T)
    end
    acc = _sum_over_batches(chunk_fn, nr, p.B, acc0, (tab, sino), sino)
    return p.vertical ? permutedims(acc, (1, 3, 2)) : permutedims(acc, (3, 1, 2))
end

# All-view convenience wrappers on the looped path (used by `dd_project` / `dd_transpose`
# when `view_batch > 1`): concatenate / accumulate over the ≤ 5 runs.
function _dd_project_looped(vol::AbstractArray{<:Any, 3}, geom::CTGeometry, view_batch::Int,
        volume_extent, ::Type{T}) where {T}
    runs = dd_run_plans(geom, size(vol); view_batch, volume_extent, eltype = T)
    vol4 = reshape(vol, size(vol)..., 1)
    return reduce((a, b) -> cat(a, b; dims = 3), [dd_project_run(vol4, r)[:, :, :, 1] for r in runs])
end
function _dd_transpose_looped(sino::AbstractArray{<:Any, 3}, geom::CTGeometry, vol_shape::NTuple{3, Int},
        view_batch::Int, volume_extent, ::Type{T}) where {T}
    acc = nothing
    for r in dd_run_plans(geom, vol_shape; view_batch, volume_extent, eltype = T)
        term = dd_transpose_run(sino[:, :, r.views], r, vol_shape)
        acc = acc === nothing ? term : acc .+ term
    end
    return acc
end

"""
    dd_project_views(vol4, geom, views; view_batch, volume_extent = nothing, eltype) -> (n_cols, n_rows, length(views), n_channels)
    dd_transpose_views(sino_sub, geom, views, vol_shape; view_batch, volume_extent = nothing, eltype) -> (nx, ny, nz)

Looped projection / transpose over an ordered subset of views (the operators an
ordered-subsets reconstruction passes in): `sino_sub` holds the subset's views
in `views` order.
"""
function dd_project_views(vol::AbstractArray{<:Any, 4}, geom::CTGeometry, views::AbstractVector{<:Integer};
        view_batch::Int, volume_extent = nothing, eltype::Type{T}) where {T}
    runs = dd_run_plans(geom, size(vol)[1:3]; view_batch, views, volume_extent, eltype = T)
    return reduce((a, b) -> cat(a, b; dims = 3), [dd_project_run(vol, r) for r in runs])
end
function dd_transpose_views(sino::AbstractArray{<:Any, 3}, geom::CTGeometry, views::AbstractVector{<:Integer},
        vol_shape::NTuple{3, Int}; view_batch::Int, volume_extent = nothing, eltype::Type{T}) where {T}
    acc = nothing
    for r in dd_run_plans(geom, vol_shape; view_batch, views, volume_extent, eltype = T)
        term = dd_transpose_run(sino[:, :, r.views], r, vol_shape)
        acc = acc === nothing ? term : acc .+ term
    end
    return acc
end

"""
    hir_operators(geom, vol_shape; view_batch, volume_extent = nothing, eltype = Float32, support = nothing) -> (A, At)

The projector pair an ordered-subsets reconstruction ([`hir_reconstruct`](@ref))
takes, on the looped DD operators: `A(vol, idx) -> (n_cols, n_rows, length(idx))`
and `At(sino_sub, idx) -> (nx, ny, nz)` for any ordered view list `idx`
(padded subsets included), each batch of `view_batch` views inside one compiled
loop.  `support` (a `(nx, ny, 1)` or `(nx, ny, nz)` Bool mask) restricts the
transpose to the reconstruction support, like the legacy `circular_support`.
"""
function hir_operators(geom::CTGeometry, vol_shape::NTuple{3, Int};
        view_batch::Int, volume_extent = nothing, eltype::Type{T} = Float32,
        support::Union{Nothing, AbstractArray{Bool}} = nothing) where {T}
    A = function (vol, idx)
        out = dd_project_views(reshape(vol, size(vol)..., 1), geom, idx; view_batch, volume_extent, eltype = T)
        return out[:, :, :, 1]
    end
    At = function (sub, idx)
        acc = dd_transpose_views(sub, geom, idx, vol_shape; view_batch, volume_extent, eltype = T)
        return support === nothing ? acc : ifelse.(support, acc, zero(T))
    end
    return A, At
end
