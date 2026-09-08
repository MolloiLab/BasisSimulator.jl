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
views (batches of `run.B` inside one compiled loop); numerically the per-view
[`dd_project_view`](@ref) up to summation order.
"""
function dd_project_dense_run(vol::AbstractArray{<:Any, 4}, p::DDRunPlan{T}) where {T <: AbstractFloat}
    size(vol)[1:3] == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("volume $(size(vol)[1:3]) does not match plan $((p.nx, p.ny, p.nz))"))
    M = size(vol, 4)
    # view-local layout (n_t, nz, n_long, M) → V[t, (z, m), l]
    Vl = p.vertical ? permutedims(vol, (1, 3, 2, 4)) : permutedims(vol, (2, 3, 1, 4))
    n_t, nz, n_long = size(Vl, 1), size(Vl, 2), size(Vl, 3)
    V = reshape(permutedims(Vl, (1, 2, 4, 3)), n_t, nz * M, n_long)
    tab = _on_device(p.table, vol)
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
    return _loop_over_batches(chunk_fn, nr, p.B, out, (tab, V), vol)
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
function dd_transpose_dense_run(sino::AbstractArray{<:Any, 3}, p::DDRunPlan{T}, vol_shape::NTuple{3, Int}) where {T <: AbstractFloat}
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
    acc = _sum_over_batches(chunk_fn, nr, p.B, acc0, (tab, sino), sino)
    return p.vertical ? permutedims(acc, (1, 3, 2)) : permutedims(acc, (3, 1, 2))
end
