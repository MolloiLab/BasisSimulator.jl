# =============================================================================
# Dense, tiled FDK backprojection — the bilinear, FDK-weighted sampling of
# `backproject` written as contractions instead of gathers.  Why: Enzyme's
# reverse of a gather is a scatter, the slowest op XLA's CPU backend has (the
# pipeline gradient was 7× its forward with the projector's own gradient at
# 1.5×); the reverse of a contraction is a contraction.
#
# Per view batch (B views) and per image TILE (tile × tile voxels, all z; one
# compiled loop over the tiles, program size independent of the grid):
#   * the tile's rays hit a window of `w` detector columns whose start is a
#     per-(tile, view) integer from a host table (`fdk_tile_windows`), `w` static;
#   * the bilinear weights are written densely — over the rows (all of them) and
#     over the column window — as exact indicator sums of the two edge-clamped
#     taps (`_lin_idx`), so
#       G[vox, k]   = Σ_row wr[vox, row] · filt[start + k − 1, row]      (batched matmul)
#       vol[vox]   += Σ_k  wc[vox, k] · G[vox, k] · inb · w_fdk          (reduction over the window)
#   Rows are dense because the arithmetic is dominated by the column window
#   (w ≫ n_row for a multi-slice scan) and this keeps the code to two steps.
# =============================================================================

_tiles(n::Int, tile::Int) = [lo:min(lo + tile - 1, n) for lo in 1:tile:n]

"""
    fdk_tile_windows(plan::FBPPlan, tile; views = 1:plan.n_view) -> (starts :: Matrix{Int32} (n_tiles, n_views), width, tiles_x, tiles_y)

Per (image tile, view): the first detector column any voxel of the tile can
sample (its taps and one column of guard), clamped so that the static window
`width` (the maximum over tiles and views, ≤ n_col) fits.  Tiles are numbered
column-major over (x-tile, y-tile).  A voxel's detector column depends on
`(x, y)` only, and the fan footprint of a convex tile is bounded by its corner
rays, so this costs four ray evaluations per (tile, view).
"""
function fdk_tile_windows(plan::FBPPlan{T, ARC}, tile::Int; views = 1:plan.n_view) where {T, ARC}
    tile >= 1 || throw(ArgumentError("tile must be ≥ 1, got $tile"))
    tb = plan.tensors
    nx, ny, n_col = plan.nx, plan.ny, plan.n_col
    tiles_x, tiles_y = _tiles(nx, tile), _tiles(ny, tile)
    X, Y, Z = vec(Array(tb.X)), vec(Array(tb.Y)), vec(Array(tb.Z))
    G = Array(_geom_table(tb))
    guard_eps = T(1e-10); oob = -one(T)
    ntx, nty, nv = length(tiles_x), length(tiles_y), length(views)
    lo = Array{Int}(undef, ntx, nty, nv); hi = similar(lo)
    for (j, v) in enumerate(views)
        g = view(G, :, v)
        colf(x, y) = ARC ?
            _col_arc(x, y, Z[1], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11], g[12],
                     plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, plan.dγ, guard_eps, oob) :
            _col_flat(x, y, Z[1], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11], g[12],
                      plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, guard_eps, oob)
        for (tj, yr) in enumerate(tiles_y), (ti, xr) in enumerate(tiles_x)
            xa, xb, ya, yb = X[first(xr)], X[last(xr)], Y[first(yr)], Y[last(yr)]
            cs = (colf(xa, ya), colf(xb, ya), colf(xa, yb), colf(xb, yb))
            cmin = clamp(minimum(cs), one(T), T(n_col)); cmax = clamp(maximum(cs), one(T), T(n_col))
            lo[ti, tj, j] = clamp(floor(Int, cmin) - 1, 1, n_col)
            hi[ti, tj, j] = clamp(ceil(Int, cmax) + 2, 1, n_col)
        end
    end
    width = min(maximum(hi .- lo) + 1, n_col)
    starts = Matrix{Int32}(undef, ntx * nty, nv)
    for j in 1:nv, tj in 1:nty, ti in 1:ntx
        starts[(tj - 1) * ntx + ti, j] = Int32(clamp(lo[ti, tj, j], 1, n_col - width + 1))
    end
    return starts, width, tiles_x, tiles_y
end

"""
    backproject_dense(filt, plan; weighted = true, view_batch, tile = 16, loop = true) -> (nx, ny, nz)

[`backproject`](@ref) as dense windowed contractions (see the header of this
file); numerically the gather version up to summation order.  `tile` is the
static x/y tile edge (full tiles run in one compiled loop; a ragged edge is
finished by up to three static tail tiles).
"""
function backproject_dense(filt::AbstractArray{<:Any, 3}, plan::FBPPlan{T, ARC};
        weighted::Bool = true, view_batch::Int, tile::Int = 16, loop::Bool = true) where {T, ARC}
    _check_sino(filt, plan, "backproject_dense")
    nx, ny, nz, n_col, n_row, n_view = plan.nx, plan.ny, plan.nz, plan.n_col, plan.n_row, plan.n_view
    starts_h, w, tiles_x, tiles_y = fdk_tile_windows(plan, tile)
    ntx, nty = length(tiles_x), length(tiles_y)
    nfx, nfy = nx ÷ tile, ny ÷ tile                        # full tiles
    tab = _on_device(starts_h, filt)
    G = _geom_table(plan.tensors)
    X, Y, Z = plan.tensors.X, plan.tensors.Y, plan.tensors.Z
    (size(X, 1) == nx && size(Y, 2) == ny && size(Z, 3) == nz && length(X) == nx && length(Y) == ny && length(Z) == nz) ||
        throw(ArgumentError("backproject_dense: plan grids must be X (nx,1,1,…), Y (1,ny,1,…), Z (1,1,nz,…); got $(size(X)), $(size(Y)), $(size(Z))"))
    acc0 = _zeros(filt, T, (nx, ny, nz))
    chunk_fn = (start, len, G, filt, tab, X, Y, Z) -> begin
        B = len
        gt = _dslice(G, start, len, 2); st = _dslice(tab, start, len, 2); fc = _dslice(filt, start, len, 3)
        out = _zeros(filt, T, (nx, ny, nz))
        # one tile: x0/y0/trow may be traced (loop index), bx/by static
        tile_fn = (out, x0, y0, bx, by, trow, gt, fc, st, X, Y, Z) ->
            _dupdate_at(out, _fdk_tile(plan, x0, y0, bx, by, trow, gt, fc, st, X, Y, Z, w, B, weighted), (x0, y0, 1))
        if nfx * nfy >= 1
            body = (o, q, c) -> begin                       # column-major over the full tiles
                ti = rem(q - 1, nfx) + 1; tj = div(q - 1, nfx) + 1
                tile_fn(o, (ti - 1) * tile + 1, (tj - 1) * tile + 1, tile, tile, (tj - 1) * ntx + ti, c...)
            end
            out = loop ? _batched_loop(body, nfx * nfy, out, (gt, fc, st, X, Y, Z), filt) :
                         _unrolled_loop(body, nfx * nfy, out, (gt, fc, st, X, Y, Z))
        end
        # ragged edges (static sizes): the x-tail strip, the y-tail strip, the corner
        tx, ty = nx - nfx * tile, ny - nfy * tile
        for tj in 1:nfy
            tx > 0 && (out = tile_fn(out, nfx * tile + 1, (tj - 1) * tile + 1, tx, tile, (tj - 1) * ntx + ntx, gt, fc, st, X, Y, Z))
        end
        for ti in 1:nfx
            ty > 0 && (out = tile_fn(out, (ti - 1) * tile + 1, nfy * tile + 1, tile, ty, (nty - 1) * ntx + ti, gt, fc, st, X, Y, Z))
        end
        (tx > 0 && ty > 0) && (out = tile_fn(out, nfx * tile + 1, nfy * tile + 1, tx, ty, ntx * nty, gt, fc, st, X, Y, Z))
        return out
    end
    acc = _sum_over_batches(chunk_fn, n_view, view_batch, acc0, (G, filt, tab, X, Y, Z), filt; unroll = !loop)
    return weighted ? acc .* plan.pi_over_angles : acc
end

# The contribution of B views to one (bx, by, nz) tile: (bx, by, nz), summed over the views.
function _fdk_tile(plan::FBPPlan{T, ARC}, x0, y0, bx::Int, by::Int, trow, gt, fc, st, X, Y, Z, w::Int, B::Int, weighted::Bool) where {T, ARC}
    n_col, n_row, nz = plan.n_col, plan.n_row, plan.nz
    r4(i) = reshape(gt[i:i, :], 1, 1, 1, B)
    sx = r4(1); sy = r4(2); sz = r4(3); dcx = r4(4); dcy = r4(5); dcz = r4(6)
    dux = r4(7); duy = r4(8); duz = r4(9); dvx = r4(10); dvy = r4(11); dvz = r4(12)
    Xt = _dslice(X, x0, bx, 1); Yt = _dslice(Y, y0, by, 2)
    guard_eps = T(1e-10); oob = -one(T); half = T(0.5); nc = T(n_col); nr = T(n_row)
    col_f, row_f = if ARC
        (_bc(_col_arc, Xt, Yt, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, plan.dγ, guard_eps, oob),
         _bc(_row_arc, Xt, Yt, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, plan.dγ, guard_eps, oob))
    else
        (_bc(_col_flat, Xt, Yt, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, guard_eps, oob),
         _bc(_row_flat, Xt, Yt, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, guard_eps, oob))
    end                                                                       # (bx,by,nz,B)
    inb = _bc(_inb, col_f, row_f, half, nc + half, nr + half)
    mask = ifelse.(inb, one(T), zero(T))
    if weighted
        mask = mask .* (ARC ? _bc(_fdk_weight_arc, Xt, Yt, Z, sx, sy, sz, plan.SAD_sq) :
                              _bc(_fdk_weight_flat, Xt, Yt, Z, sx, sy, sz, plan.SAD_sq))
    end
    col_lo = _bc(_floor_idx, col_f, nc + T(2)); row_lo = _bc(_floor_idx, row_f, nr + T(2))
    wct = reshape(_bc(_frac, col_f, nc + T(2)), bx, by, nz, 1, B)
    wrt = reshape(_bc(_frac, row_f, nr + T(2)), bx, by, nz, 1, B)
    c0 = reshape(clamp.(col_lo, Int32(1), Int32(n_col)), bx, by, nz, 1, B)              # edge-clamped taps (as `_lin_idx`)
    c1 = reshape(clamp.(col_lo .+ Int32(1), Int32(1), Int32(n_col)), bx, by, nz, 1, B)
    r0 = reshape(clamp.(row_lo, Int32(1), Int32(n_row)), bx, by, nz, 1, B)
    r1 = reshape(clamp.(row_lo .+ Int32(1), Int32(1), Int32(n_row)), bx, by, nz, 1, B)
    s5 = reshape(_dslice(st, trow, 1, 1), 1, 1, 1, 1, B)                                # window starts (Int32)
    kabs = reshape(_iota(fc, Int32, w), 1, 1, 1, w, 1) .+ (s5 .- Int32(1))              # absolute column of window position k
    rows = reshape(_iota(fc, Int32, n_row), 1, 1, 1, n_row, 1)
    o = one(T); z = zero(T)
    wc = ifelse.(kabs .== c0, o .- wct, z) .+ ifelse.(kabs .== c1, wct, z)               # (bx,by,nz,w,B)
    wr = ifelse.(rows .== r0, o .- wrt, z) .+ ifelse.(rows .== r1, wrt, z)              # (bx,by,nz,n_row,B)
    N = bx * by * nz
    fw = _fdk_window_stack(fc, st, trow, w, B)                                          # (w, n_row, B)
    Gt = _bmm_fdk(reshape(wr, N, n_row, B), fw)                                         # (N, w, B)
    val = reshape(dropdims(sum(reshape(wc, N, w, B) .* Gt; dims = 2); dims = 2), bx, by, nz, B)
    return dropdims(sum(val .* mask; dims = 4); dims = 4)
end

# per-view column window of the filtered chunk: fw[:, :, b] = fc[s_b : s_b + w − 1, :, b]
function _fdk_window_stack(fc, st, trow, w::Int, B::Int)
    parts = [reshape(_dslice_at(fc, (_scalar_start(st, trow, b), 1, b), (w, size(fc, 2), 1)), w, size(fc, 2), 1) for b in 1:B]
    return reduce((a, b) -> cat(a, b; dims = 3), parts)
end
