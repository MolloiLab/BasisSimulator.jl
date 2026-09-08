# =============================================================================
# Distance-driven (DD3) projector + exact transpose as a static-tap tensor program
# =============================================================================
#
# Same DD3 model as `src/projection/dd.jl` (`_dd_col_setup`, `_dd_row_setup`,
# `_dd_bounds`, `_dd_overlap`) and `src/projection/dd_transpose.jl`
# (`dd_backproject!`): identical footprint, identical overlap weights
# `ox · oz · norm`, identical `vertical` axis choice per view.  Only the
# *control structure* changes — the legacy per-cell `while` walks become a fixed
# number of broadcast gathers.
#
# Derivation (per view, per longitudinal slab `il`):
#   * voxel boundary `b` (0-based) maps to the iso-plane affinely,
#       pos(b) = s + (vmin + b·v − s) · mf(il),   mf = s_long / (s_long − lp(il)),
#     so the cell interval `[lo, hi]` corresponds to voxel-boundary units
#       c(lo) = ((lo − s)/mf − (vmin − s)) / v          (this is `_dd_bounds`),
#     and the voxels with non-zero overlap are `floor(c(lo))+1 … ceil(c(hi))`,
#     at most `floor((hi−lo)/(v·mf)) + 2` of them → static tap count.
#   * the transverse interval `[dXlo, dXhi]` is per (view, col); the axial
#     interval `[dZlo, dZhi]` is per (view, col, row) because `_dd_row_setup`
#     scales by the column-averaged magnification `scale_col`; `norm` is per
#     (view, col, row).  Hence the weight of voxel (it, ip) in slab il is
#       ox(col, il, it) · oz(col, row, il, ip) · norm(col, row)
#     and the per-view forward is `KX·KZ` gathers of shape (n_cols, n_rows, n_long)
#     summed over slabs.  The transpose swaps roles: per voxel (it, ip, il) the
#     first overlapping cell is the floor of the *inverse* boundary map
#     (closed form for both flat and arc detectors), and `KXT·KZT` gathers of
#     the sinogram with the same `ox · oz · norm` produce the exact adjoint —
#     gather form, deterministic, no scatter.
#
# Typing rule (Reactant): arrays are typed `AbstractArray{<:Any,N}` and the
# scalar type `T` comes ONLY from the plan (`DDViewPlan{T}`) or an explicit
# `::Type{T}`; `TracedRArray{T,N} <: AbstractArray{TracedRNumber{T},N}`, so a
# `where T` bound from an array eltype would be a `TracedRNumber` inside a trace.
#
# View-local volume layout: `V[it, ip, il]` = (transverse, z, longitudinal),
# obtained by `permutedims(vol, (1,3,2))` when `vertical` (slab axis = y) and
# `permutedims(vol, (2,3,1))` otherwise (slab axis = x).
# =============================================================================

"""
    DDViewPlan{T}

Immutable per-view host description of the distance-driven operator: world
geometry of the view, detector and volume grids, the legacy `vertical` axis
choice (evaluated in `T` exactly as the kernels do), and the static tap counts
`KX, KZ` (forward: voxels per cell) and `KXT, KZT` (transpose: cells per voxel).
Every field is a plain scalar, so a plan is a compile-time constant of the
traced program; nothing in it scales with the number of cells or voxels.
Build with [`dd_view_plan`](@ref).
"""
struct DDViewPlan{T <: AbstractFloat}
    view::Int
    arc::Bool
    vertical::Bool
    sx::Float64; sy::Float64; sz::Float64
    dcx::Float64; dcy::Float64; dcz::Float64
    ux::Float64; uy::Float64; vvz::Float64
    n_cols::Int; n_rows::Int
    SAD::Float64; SDD::Float64
    pixel_size::Float64; pixel_row_size::Float64
    nx::Int; ny::Int; nz::Int
    bounds::NTuple{3, Float64}
    KX::Int; KZ::Int; KXT::Int; KZT::Int
end

# -----------------------------------------------------------------------------
# Index vectors.  Default: host constants.  A backend extension may override
# `_iota(ref::TracedRArray, ...)` to emit an in-graph iota so that every
# geometry array downstream is computed inside the compiled program rather than
# embedded as a literal.
# -----------------------------------------------------------------------------
_iota(::AbstractArray, ::Type{T}, n::Integer) where {T} = collect(T, 1:n)

# -----------------------------------------------------------------------------
# Per-view scalar constants in precision `S`, converted exactly as the legacy
# kernels convert them (so Float32 parity holds to summation order).
# -----------------------------------------------------------------------------
function _consts(p::DDViewPlan, ::Type{S}) where {S <: AbstractFloat}
    b = p.bounds
    nx, ny, nz = p.nx, p.ny, p.nz
    vmin_x = S(-b[1] / 2); vmin_y = S(-b[2] / 2); vmin_z = S(-b[3] / 2)
    vx = S(b[1]) / S(nx); vy = S(b[2]) / S(ny); vz = S(b[3]) / S(nz)
    mag = S(p.SDD / p.SAD)
    dγ = S(p.pixel_size / p.SAD)
    ps = S(p.pixel_size); prs = S(p.pixel_row_size)
    cc = (S(p.n_cols) + one(S)) / S(2)
    rc = (S(p.n_rows) + one(S)) / S(2)
    sx = S(p.sx); sy = S(p.sy); sz = S(p.sz)
    dcx = S(p.dcx); dcy = S(p.dcy); dcz = S(p.dcz)
    ux = S(p.ux); uy = S(p.uy); vvz = S(p.vvz)
    if p.vertical
        s_long = sy; s_tran = sx
        d_tran = dcx; d_long = dcy; u_tran = ux; u_long = uy
        n_t = nx; v_t = vx; vmin_t = vmin_x
        n_long = ny; v_long = vy; vmin_long = vmin_y
    else
        s_long = sx; s_tran = sy
        d_tran = dcy; d_long = dcx; u_tran = uy; u_long = ux
        n_t = ny; v_t = vy; vmin_t = vmin_y
        n_long = nx; v_long = vx; vmin_long = vmin_x
    end
    cin_x = dcx - sx; cin_y = dcy - sy
    Lsd = sqrt(cin_x * cin_x + cin_y * cin_y)
    cin_tran = p.vertical ? cin_x : cin_y
    cin_long = p.vertical ? cin_y : cin_x
    return (; arc = p.arc, half = S(0.5), tiny = S(1.0e-12), two = S(2),
        n_cols_hi = S(p.n_cols + 2), n_rows_hi = S(p.n_rows + 2),
        sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz, mag, dγ, ps, prs, cc, rc,
        vmin_x, vmin_y, vmin_z, vx, vy, vz,
        s_long, s_tran, d_tran, d_long, u_tran, u_long, Lsd, cin_tran, cin_long,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long, nz,
        n_cols = p.n_cols, n_rows = p.n_rows)
end

# -----------------------------------------------------------------------------
# Cell setup — broadcast twins of `_dd_col_setup` / `_dd_row_setup`.
# `colf`/`rowf` are floating column/row indices of any (broadcastable) shape.
# -----------------------------------------------------------------------------
function _x_cells(colf, c)
    half = c.half
    if c.arc
        γL = (colf .- half .- c.cc) .* c.dγ
        γU = (colf .+ half .- c.cc) .* c.dγ
        cL = cos.(γL); sL = sin.(γL); cU = cos.(γU); sU = sin.(γU)
        dtL = c.s_tran .+ cL .* c.cin_tran .+ sL .* c.Lsd .* c.u_tran
        dlL = c.s_long .+ cL .* c.cin_long .+ sL .* c.Lsd .* c.u_long
        dtU = c.s_tran .+ cU .* c.cin_tran .+ sU .* c.Lsd .* c.u_tran
        dlU = c.s_long .+ cU .* c.cin_long .+ sU .* c.Lsd .* c.u_long
    else
        uL = (colf .- half .- c.cc) .* c.ps .* c.mag
        uU = (colf .+ half .- c.cc) .* c.ps .* c.mag
        dtL = c.d_tran .+ uL .* c.u_tran
        dlL = c.d_long .+ uL .* c.u_long
        dtU = c.d_tran .+ uU .* c.u_tran
        dlU = c.d_long .+ uU .* c.u_long
    end
    scaleL = c.s_long ./ (c.s_long .- dlL)
    scaleU = c.s_long ./ (c.s_long .- dlU)
    dXa = c.s_tran .+ (dtL .- c.s_tran) .* scaleL
    dXb = c.s_tran .+ (dtU .- c.s_tran) .* scaleU
    dXlo = min.(dXa, dXb)
    dXhi = max.(dXa, dXb)
    detXstep = dXhi .- dXlo
    deltaT = half .* (dXa .+ dXb) .- c.s_tran
    scale_col = half .* (scaleL .+ scaleU)
    valid_x = detXstep .> c.tiny
    return (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x)
end

function _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)
    half = c.half
    wL = (rowf .- half .- c.rc) .* c.prs .* c.mag
    wU = (rowf .+ half .- c.rc) .* c.prs .* c.mag
    zdL = c.dcz .+ wL .* c.vvz
    zdU = c.dcz .+ wU .* c.vvz
    dZa = c.sz .+ scale_col .* (zdL .- c.sz)
    dZb = c.sz .+ scale_col .* (zdU .- c.sz)
    dZlo = min.(dZa, dZb)
    dZhi = max.(dZa, dZb)
    detZstep = dZhi .- dZlo
    deltaZ = half .* (dZa .+ dZb) .- c.sz
    valid = valid_x .& (detZstep .> c.tiny)
    invCos = sqrt.(c.s_long * c.s_long .+ deltaT .* deltaT .+ deltaZ .* deltaZ) ./ abs(c.s_long) .* c.v_long
    den = ifelse.(valid, detXstep .* detZstep, one(c.half))
    norm = ifelse.(valid, invCos ./ den, zero(c.half))
    return (dZlo, dZhi, detZstep, norm)
end

# Inverse of the cell-boundary → iso-plane map: continuous column coordinate β
# (cell `col` has boundaries β = col ∓ 1/2) whose boundary projects to iso-plane
# coordinate `X`.  Flat: linear-fractional in u.  Arc: X = s_tran − s_long·(A cosγ +
# B sinγ)/(C cosγ + D sinγ) ⇒ tanγ = −(Y·C + s_long·A)/(Y·D + s_long·B), Y = X − s_tran.
function _beta_of_x(X, c)
    Y = X .- c.s_tran
    if c.arc
        num = Y .* c.cin_long .+ c.s_long .* c.cin_tran
        den = Y .* (c.Lsd * c.u_long) .+ c.s_long .* (c.Lsd * c.u_tran)
        γ = atan.(-num .* sign.(den), abs.(den))          # principal branch, |γ| < π/2
        return γ ./ c.dγ .+ c.cc
    else
        num = Y .* (c.s_long - c.d_long) .- (c.d_tran - c.s_tran) * c.s_long
        den = c.u_tran * c.s_long .+ Y .* c.u_long
        den = ifelse.(abs.(den) .< c.tiny, c.tiny, den)
        u = num ./ den
        return u ./ (c.ps * c.mag) .+ c.cc
    end
end

# First candidate cell of the transpose windows (floating and Int32 twins from
# the same clamped value so they agree bit-for-bit).  Column: floor of the
# inverse boundary map at the voxel's iso-plane interval; row: inverse of the
# affine `dZ(ρ) = sz + scale_col·(dcz + (ρ − rc)·prs·mag·vvz − sz)`.
function _col_start(tvlo, tvhi, c)
    βa = _beta_of_x(tvlo, c); βb = _beta_of_x(tvhi, c)
    c0c = _clamp_start.(min.(βa, βb) .- c.half, -c.two, c.n_cols_hi)   # one cell before the first overlap
    return (floor.(c0c), floor.(Int32, c0c))
end

function _row_start(zvlo, zvhi, scale_col, c)
    zden = c.prs * c.mag * c.vvz
    zden = abs(zden) < c.tiny ? c.tiny : zden
    sc = ifelse.(abs.(scale_col) .< c.tiny, c.tiny, scale_col)
    ρa = ((zvlo .- c.sz) ./ sc .+ c.sz .- c.dcz) ./ zden .+ c.rc
    ρb = ((zvhi .- c.sz) ./ sc .+ c.sz .- c.dcz) ./ zden .+ c.rc
    r0c = _clamp_start.(min.(ρa, ρb) .- c.half, -c.two, c.n_rows_hi)   # one row before the first overlap
    return (floor.(r0c), floor.(Int32, r0c))
end

# -----------------------------------------------------------------------------
# Per-view geometry arrays (forward).  Host-callable (plain arrays) and
# traceable (when `_iota` yields traced index vectors).
# -----------------------------------------------------------------------------
function _forward_geometry(p::DDViewPlan{T}, ref::AbstractArray) where {T}
    c = _consts(p, T)
    colf = reshape(_iota(ref, T, p.n_cols), :, 1, 1)
    rowf = reshape(_iota(ref, T, p.n_rows), 1, :, 1)
    ilf = reshape(_iota(ref, T, c.n_long), 1, 1, :)
    (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x) = _x_cells(colf, c)
    (dZlo, dZhi, _, norm) = _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)
    lp = c.vmin_long .+ (ilf .- c.half) .* c.v_long
    mf = c.s_long ./ (c.s_long .- lp)                                   # 1×1×n_long
    inv_mf = one(T) ./ mf
    cx_lo = ((dXlo .- c.s_tran) .* inv_mf .- (c.vmin_t - c.s_tran)) ./ c.v_t   # n_cols×1×n_long
    cz_lo = ((dZlo .- c.sz) .* inv_mf .- (c.vmin_z - c.sz)) ./ c.vz            # n_cols×n_rows×n_long
    i0c = _clamp_start.(cx_lo, -T(2), T(c.n_t + 2))
    k0c = _clamp_start.(cz_lo, -T(2), T(c.nz + 2))
    i0f = floor.(i0c);  i0 = floor.(Int32, i0c)        # one voxel before the first overlap (see _tap_counts)
    k0f = floor.(k0c);  k0 = floor.(Int32, k0c)
    sofs = (floor.(Int32, ilf) .- Int32(1)) .* Int32(c.n_t * c.nz)      # 1×1×n_long
    return (; c, dXlo, dXhi, dZlo, dZhi, norm, mf, i0f, i0, k0f, k0, sofs)
end

"""
    dd_project_view(vol, plan::DDViewPlan) -> (n_cols, n_rows)

Distance-driven forward projection of one view as a pure tensor program: the
exact `ox · oz · norm` weights of `dd_forward_project!`, evaluated with `KX·KZ`
static-tap gathers.  No mutation, no scalar indexing, no data-dependent loops.
"""
function dd_project_view(vol::AbstractArray{<:Any, 3}, p::DDViewPlan{T}) where {T <: AbstractFloat}
    size(vol) == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("volume $(size(vol)) does not match plan $((p.nx, p.ny, p.nz))"))
    V = p.vertical ? permutedims(vol, (1, 3, 2)) : permutedims(vol, (2, 3, 1))   # (n_t, nz, n_long)
    g = _forward_geometry(p, vol)
    c = g.c
    n_t = Int32(c.n_t); nz = Int32(c.nz)
    acc = nothing
    for dx in 0:(p.KX - 1)
        fi = g.i0f .+ T(dx)
        i = g.i0 .+ Int32(dx)
        t0 = c.s_tran .+ (c.vmin_t .+ (fi .- one(T)) .* c.v_t .- c.s_tran) .* g.mf
        t1 = c.s_tran .+ (c.vmin_t .+ fi .* c.v_t .- c.s_tran) .* g.mf
        ox = ifelse.(_inrange.(i, n_t), _overlap.(g.dXlo, g.dXhi, min.(t0, t1), max.(t0, t1)), zero(T))
        ic = clamp.(i, Int32(1), n_t)
        for dz in 0:(p.KZ - 1)
            fk = g.k0f .+ T(dz)
            k = g.k0 .+ Int32(dz)
            z0 = c.sz .+ (c.vmin_z .+ (fk .- one(T)) .* c.vz .- c.sz) .* g.mf
            z1 = c.sz .+ (c.vmin_z .+ fk .* c.vz .- c.sz) .* g.mf
            oz = ifelse.(_inrange.(k, nz), _overlap.(g.dZlo, g.dZhi, min.(z0, z1), max.(z0, z1)), zero(T))
            kc = clamp.(k, Int32(1), nz)
            L = ic .+ (kc .- Int32(1)) .* n_t .+ g.sofs                    # n_cols×n_rows×n_long
            term = ox .* oz .* _gather(V, L)
            acc = acc === nothing ? term : acc .+ term
        end
    end
    P = dropdims(sum(acc; dims = 3); dims = 3)
    return P .* dropdims(g.norm; dims = 3)
end

"""
    dd_transpose_view(sino_view, plan::DDViewPlan, vol_shape) -> (nx, ny, nz)

Exact transpose of [`dd_project_view`](@ref) for one view, in gather form:
each voxel gathers the `KXT·KZT` candidate cells around the floor of the inverse
boundary map and applies the identical `ox · oz · norm` weights.  Matches the
legacy `dd_backproject!` (which is the exact transpose of `dd_forward_project!`).
"""
function dd_transpose_view(sino::AbstractArray{<:Any, 2}, p::DDViewPlan{T},
        vol_shape::NTuple{3, Int}) where {T <: AbstractFloat}
    size(sino) == (p.n_cols, p.n_rows) ||
        throw(DimensionMismatch("sinogram view $(size(sino)) does not match plan $((p.n_cols, p.n_rows))"))
    vol_shape == (p.nx, p.ny, p.nz) ||
        throw(DimensionMismatch("vol_shape $(vol_shape) does not match plan $((p.nx, p.ny, p.nz))"))
    c = _consts(p, T)
    n_t, nz, n_long = c.n_t, c.nz, c.n_long
    ncol = Int32(p.n_cols); nrow = Int32(p.n_rows)
    itf = reshape(_iota(sino, T, n_t), :, 1, 1)
    ipf = reshape(_iota(sino, T, nz), 1, :, 1)
    ilf = reshape(_iota(sino, T, n_long), 1, 1, :)
    lp = c.vmin_long .+ (ilf .- c.half) .* c.v_long
    mf = c.s_long ./ (c.s_long .- lp)
    t0 = c.s_tran .+ (c.vmin_t .+ (itf .- one(T)) .* c.v_t .- c.s_tran) .* mf
    t1 = c.s_tran .+ (c.vmin_t .+ itf .* c.v_t .- c.s_tran) .* mf
    tvlo = min.(t0, t1); tvhi = max.(t0, t1)                                # n_t×1×n_long
    z0 = c.sz .+ (c.vmin_z .+ (ipf .- one(T)) .* c.vz .- c.sz) .* mf
    z1 = c.sz .+ (c.vmin_z .+ ipf .* c.vz .- c.sz) .* mf
    zvlo = min.(z0, z1); zvhi = max.(z0, z1)                                # 1×nz×n_long
    col0f, col0 = _col_start(tvlo, tvhi, c)
    acc = nothing
    for dc in 0:(p.KXT - 1)
        colf = col0f .+ T(dc)
        col = col0 .+ Int32(dc)
        (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x) = _x_cells(colf, c)
        ox = ifelse.(_inrange.(col, ncol), _overlap.(dXlo, dXhi, tvlo, tvhi), zero(T))
        colc = clamp.(col, Int32(1), ncol)
        row0f, row0 = _row_start(zvlo, zvhi, scale_col, c)                   # n_t×nz×n_long
        for dr in 0:(p.KZT - 1)
            rowf = row0f .+ T(dr)
            row = row0 .+ Int32(dr)
            (dZlo, dZhi, _, norm) = _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)
            oz = ifelse.(_inrange.(row, nrow), _overlap.(dZlo, dZhi, zvlo, zvhi), zero(T))
            rowc = clamp.(row, Int32(1), nrow)
            L = colc .+ (rowc .- Int32(1)) .* ncol                           # n_t×nz×n_long
            term = ox .* oz .* norm .* _gather(sino, L)
            acc = acc === nothing ? term : acc .+ term
        end
    end
    return p.vertical ? permutedims(acc, (1, 3, 2)) : permutedims(acc, (3, 1, 2))
end

# -----------------------------------------------------------------------------
# Plan construction (host).  Static tap counts are proven bounds:
#   forward:   an interval of length `span` (voxel-boundary units) meets at most
#              floor(span) + 2 voxels, the first being floor(c_lo) + 1.  We take
#              K = floor(span_max + δ) + 3 taps starting at floor(c_lo): the
#              spare tap sits *before* the first overlap, so a ±1 ulp rounding
#              of `c_lo` at an exact boundary (where the legacy t-domain test may
#              keep an ulp-weight voxel on either side) is always covered, and
#              the maximal count floor(span)+2 cannot coincide with an integer
#              `c_lo`.  δ = 1e-3 guards spans within rounding of an integer.
#   transpose: cells per voxel ≤ the exact sliding-window maximum over the cell
#              boundaries for an interval of the max magnified voxel width
#              (`_max_cells_spanned`, itself conservative by one), +1, starting
#              one cell before the floor of the inverse boundary map.
# -----------------------------------------------------------------------------
const _SPAN_GUARD = 1.0e-3
"""
    dd_view_plan(geom, view, vol_shape; volume_extent=nothing, eltype=Float64)

Build the [`DDViewPlan`](@ref) of view `view` for a volume of shape `vol_shape`
(physical extent `volume_extent`, default `geom.fov`).  `eltype` is the array
element type the plan will be used with; the per-view `vertical` axis choice is
evaluated in that type exactly like the legacy kernels.
"""
function dd_view_plan(geom::CTGeometry, view::Integer, vol_shape::NTuple{3, Int};
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        eltype::Type{T} = Float64) where {T <: AbstractFloat}
    1 <= view <= geom.n_angles || throw(ArgumentError("view $view out of 1:$(geom.n_angles)"))
    bounds = volume_extent === nothing ? geom.fov : volume_extent
    sx, sy, sz = geom.source_positions[1, view], geom.source_positions[2, view], geom.source_positions[3, view]
    dcx, dcy, dcz = geom.detector_centers[1, view], geom.detector_centers[2, view], geom.detector_centers[3, view]
    ux, uy = geom.detector_u[1, view], geom.detector_u[2, view]
    vvz = geom.detector_v[3, view]
    vertical = abs(T(sy)) >= abs(T(sx))                 # legacy: computed in T
    p0 = DDViewPlan{T}(Int(view), is_arc(geom), vertical, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
        geom.n_cols, geom.n_rows, geom.SAD, geom.SDD, geom.pixel_size, geom.pixel_row_size,
        vol_shape[1], vol_shape[2], vol_shape[3], bounds, 0, 0, 0, 0)
    KX, KZ, KXT, KZT = _tap_counts(p0)
    return DDViewPlan{T}(Int(view), is_arc(geom), vertical, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
        geom.n_cols, geom.n_rows, geom.SAD, geom.SDD, geom.pixel_size, geom.pixel_row_size,
        vol_shape[1], vol_shape[2], vol_shape[3], bounds, KX, KZ, KXT, KZT)
end

function _tap_counts(p::DDViewPlan)
    c = _consts(p, Float64)
    colf = reshape(collect(Float64, 1:p.n_cols), :, 1)
    rowf = reshape(collect(Float64, 1:p.n_rows), 1, :)
    (dXlo, dXhi, detXstep, deltaT, scale_col, valid_x) = _x_cells(colf, c)
    (_, _, detZstep, _) = _z_cells(rowf, scale_col, deltaT, detXstep, valid_x, c)
    lp_a = c.vmin_long + 0.5 * c.v_long
    lp_b = c.vmin_long + (c.n_long - 0.5) * c.v_long
    mf_a = c.s_long / (c.s_long - lp_a); mf_b = c.s_long / (c.s_long - lp_b)
    (mf_a > 0 && mf_b > 0) || throw(ArgumentError("dd_view_plan: source inside the volume along the longitudinal axis (view $(p.view))"))
    mf_min = min(mf_a, mf_b); mf_max = max(mf_a, mf_b)
    vx = vec(valid_x)
    any(vx) || return (1, 1, 1, 1)
    dX_max = maximum(vec(detXstep)[vx])
    valid2 = valid_x .& (detZstep .> c.tiny)
    dZ_max = any(valid2) ? maximum(detZstep[valid2]) : 0.0
    KX = floor(Int, dX_max / (c.v_t * mf_min) + _SPAN_GUARD) + 3
    KZ = floor(Int, dZ_max / (c.vz * mf_min) + _SPAN_GUARD) + 3
    boundaries = unique(sort(vcat(vec(dXlo)[vx], vec(dXhi)[vx])))
    KXT = _max_cells_spanned(boundaries, c.v_t * mf_max) + 1
    z_spacing = minimum(abs.(vec(scale_col)[vx])) * abs(c.prs * c.mag * c.vvz)
    KZT = floor(Int, c.vz * mf_max / max(z_spacing, c.tiny) + _SPAN_GUARD) + 3
    return (KX, KZ, KXT, KZT)
end

"""
    dd_project(vol, geom; volume_extent=nothing, eltype=Base.eltype(vol)) -> (n_cols, n_rows, n_views)

All-view convenience wrapper over [`dd_project_view`](@ref) (host loop over
views, results stacked).  Numerically the legacy `dd_forward_project`.  `eltype`
is the primitive scalar type used to build the plans; pass it explicitly (e.g.
`Float32`) when `vol` is a traced array, whose `Base.eltype` is not primitive.
"""
function dd_project(vol::AbstractArray{<:Any, 3}, geom::CTGeometry;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        eltype::Type{T} = Base.eltype(vol)) where {T <: AbstractFloat}
    views = [dd_project_view(vol, dd_view_plan(geom, v, size(vol); volume_extent, eltype = T))
             for v in 1:geom.n_angles]
    return stack(views)                                   # (n_cols, n_rows, n_views)
end

"""
    dd_transpose(sino, geom, vol_shape; volume_extent=nothing, eltype=Base.eltype(sino)) -> (nx, ny, nz)

All-view exact transpose: sum over views of [`dd_transpose_view`](@ref).
Numerically the legacy `dd_backproject!` (unweighted, full z, no support mask).
`eltype` as in [`dd_project`](@ref).
"""
function dd_transpose(sino::AbstractArray{<:Any, 3}, geom::CTGeometry, vol_shape::NTuple{3, Int};
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        eltype::Type{T} = Base.eltype(sino)) where {T <: AbstractFloat}
    acc = nothing
    for v in 1:geom.n_angles
        plan = dd_view_plan(geom, v, vol_shape; volume_extent, eltype = T)
        term = dd_transpose_view(sino[:, :, v], plan, vol_shape)
        acc = acc === nothing ? term : acc .+ term
    end
    return acc
end
