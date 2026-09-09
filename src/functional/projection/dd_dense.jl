# =============================================================================
# Dense-separable distance-driven projection — the SAME DD3 physics as
# dd_projector.jl (identical ox · oz · norm overlap weights) written as two
# batched contractions instead of static-tap gathers:
#
#   A[col, (z, m), l]   = Σ_t  Wx[col, t, l] · V[t, (z, m), l]            (per slab l: matmul)
#   P[col, row, m]      = Σ_l Σ_z  Wz[col, row, z, l] · A[col, (z, m), l]  (batched over col)
#   P                  .*= norm[col, row]
#
# with Wx[col, t, l] = overlap(cell col, voxel t at slab l) for EVERY t (zero
# outside the K taps) and Wz likewise.  Gathers are what XLA's CPU backend
# executes worst (measured: 12× slower than the legacy kernels); dense
# contractions run near peak on CPU and GPU.  Cost per view: n_cols·n_t·n_long
# weights and n_cols·n_t·n_long·nz·M MACs — the memory budget picks the view
# batch, and `material_paths` falls back to the gather path when even one
# view's weights exceed the budget (UHR grids).
# =============================================================================

"""
    dd_project_dense_run(vol4, run::DDRunPlan) -> (n_cols, n_rows, n_run, n_channels)

Dense-separable forward projection of every channel of `vol4` over the run's
views (batches of `run.B` inside one compiled loop, or unrolled with `unroll = true`); numerically the per-view
[`dd_project_view`](@ref) up to summation order.
"""
function dd_project_dense_run(vol::AbstractArray{<:Any, 4}, p::DDRunPlan{T}; unroll::Bool = false, table = nothing) where {T <: AbstractFloat}
    size(vol)[1:3] == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("volume $(size(vol)[1:3]) does not match plan $((p.nx, p.ny, p.nz))"))
    M = size(vol, 4)
    # view-local layout (n_t, nz, n_long, M) → V[t, (z, m), l]
    Vl = p.vertical ? permutedims(vol, (1, 3, 2, 4)) : permutedims(vol, (2, 3, 1, 4))
    n_t, nz, n_long = size(Vl, 1), size(Vl, 2), size(Vl, 3)
    V = reshape(permutedims(Vl, (1, 2, 4, 3)), n_t, nz * M, n_long)
    tab = table === nothing ? _on_device(p.table, vol) : table              # `table`: the (18, n_run) view table as DATA (batch programs)
    nr = length(p.views)
    out = _zeros(vol, T, (p.n_cols, p.n_rows, nr, M))
    n_cols, n_rows = p.n_cols, p.n_rows
    chunk_fn = (start, len, tab, V) -> begin                                  # closes over host data only
        c = _consts_run(p, _dslice(tab, start, len, 2))
        B = len
        colf = reshape(_iota(tab, T, n_cols), :, 1, 1, 1)
        rowf = reshape(_iota(tab, T, n_rows), 1, :, 1, 1)
        ilf = reshape(_iota(tab, T, n_long), 1, 1, :, 1)
        tf = reshape(_iota(tab, T, n_t), 1, :, 1, 1)
        zf = reshape(_iota(tab, T, nz), 1, 1, :, 1, 1)
        (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x) = _x_cells(colf, c)           # (n_cols,1,1,B)
        (dZlo, dZhi, _, norm) = _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)   # (n_cols,n_rows,1,B)
        lp = c.vmin_long .+ (ilf .- c.half) .* c.v_long
        mf = c.s_long ./ (c.s_long .- lp)                                                 # (1,1,n_long,B)
        # transverse overlap for every voxel t at every slab l
        t0 = c.s_tran .+ (c.vmin_t .+ (tf .- one(T)) .* c.v_t .- c.s_tran) .* mf         # (1,n_t,n_long,B)
        t1 = c.s_tran .+ (c.vmin_t .+ tf .* c.v_t .- c.s_tran) .* mf
        Wx = _overlap.(dXlo, dXhi, min.(t0, t1), max.(t0, t1))                            # (n_cols,n_t,n_long,B)
        # axial overlap for every voxel z at every slab l
        mf5 = reshape(mf, 1, 1, 1, n_long, B); sz5 = reshape(c.sz, 1, 1, 1, 1, B)         # per-view constants on the 5-D view axis
        z0 = sz5 .+ (c.vmin_z .+ (zf .- one(T)) .* c.vz .- sz5) .* mf5                   # (1,1,nz,n_long,B)
        z1 = sz5 .+ (c.vmin_z .+ zf .* c.vz .- sz5) .* mf5
        dZlo5 = reshape(dZlo, n_cols, n_rows, 1, 1, B); dZhi5 = reshape(dZhi, n_cols, n_rows, 1, 1, B)
        Wz = _overlap.(dZlo5, dZhi5, min.(z0, z1), max.(z0, z1))                          # (n_cols,n_rows,nz,n_long,B)
        A = _bmm_t(Wx, V)                                                                 # (n_cols, nz·M, n_long, B)
        P = _bmm_zl(Wz, reshape(A, n_cols, nz, M, n_long, B))                             # (n_cols, n_rows, M, B)
        return permutedims(P .* reshape(dropdims(norm; dims = 3), n_cols, n_rows, 1, B), (1, 2, 4, 3))   # (n_cols,n_rows,B,M)
    end
    return _loop_over_batches(chunk_fn, nr, p.B, out, (tab, V), vol; unroll)
end

"""
    dd_dense_view_bytes(n_cols, n_rows, n_t, nz, n_long, n_channels) -> bytes per view

Host estimate of the dense weights and intermediates one view needs
(`Wx`, `Wz`, `A`, in Float32), used by the memory-budget batching.
"""
dd_dense_view_bytes(n_cols, n_rows, n_t, nz, n_long, M) =
    4 * (n_cols * n_t * n_long + n_cols * n_rows * nz * n_long + n_cols * nz * M * n_long + n_t * nz * M * n_long)

"""
    dd_transpose_dense_run(sino_run, run::DDRunPlan, vol_shape) -> (nx, ny, nz)

Exact transpose of [`dd_project_dense_run`](@ref) for one channel over the run's
views (`sino_run :: (n_cols, n_rows, n_run)`), as two batched contractions:
`G[col, z, l, b] = Σ_row Wz[col,row,z,l,b] · (norm · S)[col,row,b]` then
`V[t, z, l] = Σ_{col,b} Wx[col,t,l,b] · G[col,z,l,b]`.  Numerically the per-view
[`dd_transpose_view`](@ref) sum up to summation order.
"""
function dd_transpose_dense_run(sino::AbstractArray{<:Any, 3}, p::DDRunPlan{T}, vol_shape::NTuple{3, Int}; unroll::Bool = false) where {T <: AbstractFloat}
    nr = length(p.views)
    size(sino) == (p.n_cols, p.n_rows, nr) ||
        throw(DimensionMismatch("sinogram run $(size(sino)) does not match plan $((p.n_cols, p.n_rows, nr))"))
    vol_shape == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("vol_shape $(vol_shape) does not match plan $((p.nx, p.ny, p.nz))"))
    n_t, n_long = p.vertical ? (p.nx, p.ny) : (p.ny, p.nx); nz = p.nz
    tab = _on_device(p.table, sino)
    n_cols, n_rows = p.n_cols, p.n_rows
    acc0 = _zeros(sino, T, (n_t, nz, n_long))
    chunk_fn = (start, len, tab, sino) -> begin                                # closes over host data only
        c = _consts_run(p, _dslice(tab, start, len, 2))
        B = len
        colf = reshape(_iota(tab, T, n_cols), :, 1, 1, 1)
        rowf = reshape(_iota(tab, T, n_rows), 1, :, 1, 1)
        ilf = reshape(_iota(tab, T, n_long), 1, 1, :, 1)
        tf = reshape(_iota(tab, T, n_t), 1, :, 1, 1)
        zf = reshape(_iota(tab, T, nz), 1, 1, :, 1, 1)
        (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x) = _x_cells(colf, c)
        (dZlo, dZhi, _, norm) = _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)
        lp = c.vmin_long .+ (ilf .- c.half) .* c.v_long
        mf = c.s_long ./ (c.s_long .- lp)
        t0 = c.s_tran .+ (c.vmin_t .+ (tf .- one(T)) .* c.v_t .- c.s_tran) .* mf
        t1 = c.s_tran .+ (c.vmin_t .+ tf .* c.v_t .- c.s_tran) .* mf
        Wx = _overlap.(dXlo, dXhi, min.(t0, t1), max.(t0, t1))                            # (n_cols,n_t,n_long,B)
        mf5 = reshape(mf, 1, 1, 1, n_long, B); sz5 = reshape(c.sz, 1, 1, 1, 1, B)
        z0 = sz5 .+ (c.vmin_z .+ (zf .- one(T)) .* c.vz .- sz5) .* mf5
        z1 = sz5 .+ (c.vmin_z .+ zf .* c.vz .- sz5) .* mf5
        dZlo5 = reshape(dZlo, n_cols, n_rows, 1, 1, B); dZhi5 = reshape(dZhi, n_cols, n_rows, 1, 1, B)
        Wz = _overlap.(dZlo5, dZhi5, min.(z0, z1), max.(z0, z1))                          # (n_cols,n_rows,nz,n_long,B)
        S = _dslice(sino, start, len, 3) .* dropdims(norm; dims = 3)                      # (n_cols,n_rows,B)
        G = _bmm_rows(Wz, S)                                                              # (n_cols, nz, n_long, B)
        return _bmm_cols(Wx, G)                                                           # (n_t, nz, n_long)
    end
    acc = _sum_over_batches(chunk_fn, nr, p.B, acc0, (tab, sino), sino; unroll)
    return p.vertical ? permutedims(acc, (1, 3, 2)) : permutedims(acc, (3, 1, 2))
end

# -----------------------------------------------------------------------------
# Windowed dense forward: memory ∝ n_cols · w · n_long instead of n_cols · n_t · n_long.
# Detector columns are cut into STATIC blocks and the slabs (the longitudinal
# axis) into STATIC blocks; for each (column block, slab block, view) the
# transverse voxels the block's rays can touch form a window whose start is a
# per-view integer (host-computed from the geometry) and whose width `w` is
# static (the maximum over views and slab blocks).  A fan-beam ray drifts
# across transverse voxels with depth, so windows are only narrow PER SLAB
# BLOCK — that is why both axes are blocked.  Inside the program the window
# of `V` is a dynamic slice (transverse start dynamic, slab start dynamic in
# the inner loop), the block's weights are dense over the window, and the
# contraction result is written into its (static column, dynamic slab) slice
# of `A`.  Same physics, same contractions; this is what makes UHR grids fit.
# -----------------------------------------------------------------------------

"""
    dd_windows(geom, vol_shape, views, col_block, slab_block; volume_extent = nothing, eltype = Float64)
        -> (starts :: Array{Int32, 3} (n_col_blocks, n_slab_blocks, n_views), widths :: Vector{Int} (per column block),
            col_blocks, slab_blocks)

For every (column block, slab block, view): the first transverse voxel any of
the block's cells can overlap within those slabs (1-based, clamped so that the
static-width window fits); per column block the static window width (the
maximum over views and slab blocks, one voxel of guard on each side).
"""
function dd_windows(geom::CTGeometry, vol_shape::NTuple{3, Int}, views::AbstractVector{<:Integer}, col_block::Int, slab_block::Int;
        volume_extent = nothing, eltype::Type{T} = Float64) where {T}
    n_cols = geom.n_cols
    col_blocks = [lo:min(lo + col_block - 1, n_cols) for lo in 1:col_block:n_cols]
    p1 = dd_view_plan(geom, Int(views[1]), vol_shape; volume_extent, eltype = T)
    n_long = _consts(p1, T).n_long; n_t = _consts(p1, T).n_t
    slab_blocks = [lo:min(lo + slab_block - 1, n_long) for lo in 1:slab_block:n_long]
    ncb, nsb, nv = length(col_blocks), length(slab_blocks), length(views)
    lo = fill(typemax(Int), ncb, nsb, nv); hi = fill(typemin(Int), ncb, nsb, nv)
    for (j, v) in enumerate(views)
        p = dd_view_plan(geom, Int(v), vol_shape; volume_extent, eltype = T)
        c = _consts(p, T)
        colf = reshape(collect(T, 1:n_cols), :, 1)
        ilf = reshape(collect(T, 1:c.n_long), 1, :)
        (dXlo, dXhi, _, _, _, _) = _x_cells(colf, c)
        lp = c.vmin_long .+ (ilf .- c.half) .* c.v_long
        inv_mf = (c.s_long .- lp) ./ c.s_long                                         # 1×n_long
        cx_a = ((dXlo .- c.s_tran) .* inv_mf .- (c.vmin_t .- c.s_tran)) ./ c.v_t     # n_cols×n_long (0-based voxel units)
        cx_b = ((dXhi .- c.s_tran) .* inv_mf .- (c.vmin_t .- c.s_tran)) ./ c.v_t
        i_lo = floor.(Int, min.(cx_a, cx_b))                                          # first voxel (1-based) that can overlap, minus one of guard
        i_hi = ceil.(Int, max.(cx_a, cx_b)) .+ 2                                      # one past the last, plus one of guard
        for (bi, sr) in enumerate(slab_blocks), (bj, cr) in enumerate(col_blocks)
            lo[bj, bi, j] = clamp(minimum(@view i_lo[cr, sr]), 1, n_t)
            hi[bj, bi, j] = clamp(maximum(@view i_hi[cr, sr]), 1, n_t + 1)
        end
    end
    widths = [min(maximum(hi[bj, :, :] .- lo[bj, :, :]), n_t) for bj in 1:ncb]
    starts = Array{Int32, 3}(undef, ncb, nsb, nv)
    for bj in 1:ncb, bi in 1:nsb, j in 1:nv
        starts[bj, bi, j] = Int32(clamp(lo[bj, bi, j], 1, n_t - widths[bj] + 1))
    end
    return starts, widths, col_blocks, slab_blocks
end

"""
    dd_dense_windowed_view_bytes(n_cols, n_rows, nz, n_long, n_channels, widths, col_block, slab_block) -> bytes per view

Host estimate of the windowed dense weights and intermediates one view needs.
"""
function dd_dense_windowed_view_bytes(n_cols, n_rows, nz, n_long, M, widths, col_block, slab_block)
    K = nz * M
    wx = sum(min(col_block, n_cols) * w * min(slab_block, n_long) for w in widths)          # one (column block × slab block) at a time
    return 4 * (wx + n_cols * n_rows * nz * n_long + n_cols * K * n_long + maximum(widths) * K * min(slab_block, n_long))
end

"""
    dd_project_dense_windowed_run(vol4, run::DDRunPlan, geom; col_block, slab_block, volume_extent = nothing, unroll = false)

[`dd_project_dense_run`](@ref) with (column block × slab block) windows: the
same contractions over only the transverse voxels each block of rays can touch.
`col_block = n_cols, slab_block = n_long` is the full dense projector.
"""
function dd_project_dense_windowed_run(vol::AbstractArray{<:Any, 4}, p::DDRunPlan{T}, geom::CTGeometry;
        col_block::Int, slab_block::Int, volume_extent = nothing, unroll::Bool = false, table = nothing, windows = nothing) where {T <: AbstractFloat}
    size(vol)[1:3] == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("volume $(size(vol)[1:3]) does not match plan $((p.nx, p.ny, p.nz))"))
    M = size(vol, 4)
    Vl = p.vertical ? permutedims(vol, (1, 3, 2, 4)) : permutedims(vol, (2, 3, 1, 4))
    n_t, nz, n_long = size(Vl, 1), size(Vl, 2), size(Vl, 3)
    V = reshape(permutedims(Vl, (1, 2, 4, 3)), n_t, nz * M, n_long)
    # `windows = (starts, widths, col_blocks, slab_blocks)`: precomputed (`dd_windows`), the starts as DATA — widths static
    starts_h, widths, col_blocks, slab_blocks = windows === nothing ?
        dd_windows(geom, (p.nx, p.ny, p.nz), collect(p.views), col_block, slab_block; volume_extent, eltype = T) : windows
    ncb, nsb = length(col_blocks), length(slab_blocks)
    bl = length(slab_blocks[1]); tail_l = length(slab_blocks[end]); nsb_full = tail_l == bl ? nsb : nsb - 1
    tab = table === nothing ? _on_device(p.table, vol) : table
    tabI = windows === nothing ? _on_device(reshape(starts_h, ncb * nsb, :), vol) : reshape(starts_h, ncb * nsb, :)   # (ncb·nsb, n_run) Int32
    nr = length(p.views)
    out = _zeros(vol, T, (p.n_cols, p.n_rows, nr, M))
    n_cols, n_rows = p.n_cols, p.n_rows
    K = nz * M
    chunk_fn = (start, len, tab, tabI, V) -> begin                                     # closes over host data only
        c = _consts_run(p, _dslice(tab, start, len, 2))
        st = _dslice(tabI, start, len, 2)                                              # (ncb·nsb, B)
        B = len
        colf = reshape(_iota(tab, T, n_cols), :, 1, 1, 1)
        rowf = reshape(_iota(tab, T, n_rows), 1, :, 1, 1)
        ilf = reshape(_iota(tab, T, n_long), 1, 1, :, 1)
        zf = reshape(_iota(tab, T, nz), 1, 1, :, 1, 1)
        (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x) = _x_cells(colf, c)
        (dZlo, dZhi, _, norm) = _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)
        lp = c.vmin_long .+ (ilf .- c.half) .* c.v_long
        mf = c.s_long ./ (c.s_long .- lp)                                              # (1,1,n_long,B)
        mf5 = reshape(mf, 1, 1, 1, n_long, B); sz5 = reshape(c.sz, 1, 1, 1, 1, B)
        z0 = sz5 .+ (c.vmin_z .+ (zf .- one(T)) .* c.vz .- sz5) .* mf5
        z1 = sz5 .+ (c.vmin_z .+ zf .* c.vz .- sz5) .* mf5
        dZlo5 = reshape(dZlo, n_cols, n_rows, 1, 1, B); dZhi5 = reshape(dZhi, n_cols, n_rows, 1, 1, B)
        Wz = _overlap.(dZlo5, dZhi5, min.(z0, z1), max.(z0, z1))                       # (n_cols,n_rows,nz,n_long,B)
        A = _zeros(tab, T, (n_cols, K, n_long, B))
        for (bj, cr) in enumerate(col_blocks)
            w = widths[bj]; c0 = first(cr); bc = length(cr)
            dXlo_j = dXlo[cr, :, :, :]; dXhi_j = dXhi[cr, :, :, :]                     # (bc,1,1,B)
            # one slab block: window start per view from the table row (bj, bi).  Every traced value the
            # block reads is an argument (loop operand), never a closure capture.
            block_fn = (A_in, bi, l0, blen, st, mf, V, dXlo_j, dXhi_j, c, tab) -> begin
                row = (bi - 1) * ncb + bj                                              # row of (bj, bi) in tabI (column-major over (bj, bi))
                s_j = reshape(_plain(_dslice(st, row, 1, 1)), 1, 1, 1, B)              # (1,1,1,B) Int32 window starts
                mf_i = _dslice(mf, l0, blen, 3)                                        # (1,1,blen,B)
                tf = _to_float(T, s_j) .+ reshape(_iota(tab, T, w), 1, :, 1, 1) .- one(T)   # (1,w,1,B) voxel index (1-based)
                t0 = c.s_tran .+ (c.vmin_t .+ (tf .- one(T)) .* c.v_t .- c.s_tran) .* mf_i   # (1,w,blen,B)
                t1 = c.s_tran .+ (c.vmin_t .+ tf .* c.v_t .- c.s_tran) .* mf_i
                Wx = _overlap.(dXlo_j, dXhi_j, min.(t0, t1), max.(t0, t1))             # (bc,w,blen,B)
                Vw = _window_stack2(V, st, row, w, l0, blen, B)                        # (w,K,blen,B)
                Aj = _bmm_tb(Wx, Vw)                                                    # (bc,K,blen,B)
                return _dupdate_at(A_in, Aj, (c0, 1, l0, 1))
            end
            ops = (st, mf, V, dXlo_j, dXhi_j, c, tab)
            if nsb_full >= 1
                sb_body = (state, bi, cs) -> block_fn(state, bi, (bi - 1) * bl + 1, bl, cs...)
                A = (unroll || nsb_full == 1) ? _unrolled_loop(sb_body, nsb_full, A, ops) : _batched_loop(sb_body, nsb_full, A, ops, tab)
            end
            if tail_l != bl
                A = block_fn(A, nsb, nsb_full * bl + 1, tail_l, ops...)
            end
        end
        P = _bmm_zl(Wz, reshape(A, n_cols, nz, M, n_long, B))                          # (n_cols, n_rows, M, B)
        return permutedims(P .* reshape(dropdims(norm; dims = 3), n_cols, n_rows, 1, B), (1, 2, 4, 3))
    end
    return _loop_over_batches(chunk_fn, nr, p.B, out, (tab, tabI, V), vol; unroll)
end

# stack the per-view (transverse window × slab block) of V: Vw[:, :, :, b] = V[s_b : s_b+w-1, :, l0 : l0+blen-1]
function _window_stack2(V, st, row, w::Int, l0, blen::Int, B::Int)
    parts = [reshape(_dslice_at(V, (_scalar_start(st, row, b), 1, l0), (w, size(V, 2), blen)), w, size(V, 2), blen, 1) for b in 1:B]
    return reduce((a, b) -> cat(a, b; dims = 4), parts)
end
_scalar_start(st, j, b::Int) = Int(st[j, b])                 # host (j :: Int); the extension returns a traced integer
