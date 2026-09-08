# =============================================================================
# BasisSimulator.Functional — FBP / FDK reconstruction as a pure tensor program
# =============================================================================
#
# Pure, mutation-free, array-generic re-implementation of the legacy axial FDK
# chain
#
#     filter_sinogram!  →  backproject!(weighted = true)  →  apply_fov_mask!
#
# (src/reconstruction/core/filtering.jl, core/backprojection.jl, fbp/fdk.jl,
# and the `reconstruct!(::FDKReconWorkspace)` driver).  The legacy
# AcceleratedKernels code is the numerical ORACLE: with plain `Array`s and
# `view_batch = 1` every per-element operation below is the same sequence of
# IEEE operations the legacy kernels execute, so the outputs agree to rounding
# (see test/functional/test_fbp.jl for the measured parity numbers).
#
# Rules followed by every stage function in this file:
#   * pure function of arrays + host scalars + an immutable `FBPPlan`;
#     inputs are never mutated, a new array is returned;
#   * every array shape is a function of host config (plan), never of data;
#   * no scalar indexing of stage arrays — only broadcasting, `reshape`,
#     `sum(; dims)`, one matmul, and gathers `vec(x)[idx]` with Int32 index
#     tensors computed by broadcasting;
#   * masks via `ifelse`/`min`/`max` inside branch-free scalar helpers, no
#     data-dependent branches or loops;
#   * arc-vs-flat detector is a TYPE parameter of the plan (`FBPPlan{T,ARC}`),
#     resolved on the host before any array code runs;
#   * the only loop is over views (static trip count `n_view ÷ view_batch`);
#   * array arguments are typed `AbstractArray{<:Any, N}` and the scalar type
#     `T` is taken ONLY from the plan: `Reactant.TracedRArray{T,N} <:
#     AbstractArray{TracedRNumber{T},N}`, so a `where {T}` bound from an
#     array's eltype would not match inside `Reactant.@compile`.
#
# Host-side precomputation of geometry-only constants (cosine weights, ramp
# taps, the Toeplitz filter matrix, voxel grids, FOV mask) is done ONCE in
# `fbp_plan`, re-using the legacy host helpers `BS.create_spatial_kernel`
# (with its window functions) and `BS.equiangular_kernel_scale!`.
#
# The enclosing module must bind `BS` to `BasisSimulator`
# (`import ..BasisSimulator as BS` inside `BasisSimulator.Functional`,
# `const BS = BasisSimulator` in a scratch test module).
#
# Sinogram layout is `(n_col, n_row, n_view)`; volumes are `(nx, ny, nz)`.
# =============================================================================

# Public API (nothing is exported — call `BasisSimulator.Functional.<name>`):
#   FBPPlan, fbp_plan, filter_views, backproject_view, backproject, fov_mask, fdk

# -----------------------------------------------------------------------------
# Plan
# -----------------------------------------------------------------------------

"""
    FBPPlan{T, ARC, D}

Immutable, host-built description of one axial FDK reconstruction problem.

* `T`   — element type of every tensor (`Float32` or `Float64`).
* `ARC` — `true` for an equiangular (`:arc`) detector, `false` for `:flat`.
  Being a type parameter, the arc/flat branch is resolved by dispatch before
  any array code runs.
* `D`   — the `NamedTuple` type of `tensors` (see below).

Fields
* `n_col, n_row, n_view, nx, ny, nz` — static sizes.
* `kernel::Vector{T}` — the spatial ramp taps actually used (odd length `K`,
  centred at `K ÷ 2 + 1`, window applied, `(γ/sin γ)²`-scaled when `ARC`).
  Kept for inspection only; the filter stage uses `tensors.H`.
* `tensors::D` — every array that participates in the tensor program:
  - `cos_weights :: (n_col, n_row)` FDK cosine pre-weight (`cosine_weight!`);
  - `H           :: (n_col, n_col)` zero-padded Toeplitz matrix of `kernel`,
    `H[i, j] = kernel[j - i + K÷2 + 1]` for `|j - i| ≤ K÷2`, else `0`;
  - `source_positions, detector_centers, detector_u, detector_v :: (3, n_view)`;
  - `X :: (nx,1,1,1)`, `Y :: (1,ny,1,1)`, `Z :: (1,1,nz,1)` voxel centres;
  - `fov_outside :: (nx, ny, 1)` `Bool`, `true` outside the inscribed circle.
  These are plain host `Array`s after `fbp_plan`.  To run the stages inside a
  compiled Reactant program, move them to the device with
  `Reactant.to_rarray(plan.tensors)` and rebuild the plan with
  `FBPPlan(plan; tensors = ...)`, so the per-view index tensors are built
  in-graph rather than baked in as trace-time constants.
* scalars `SAD, SAD_sq, SDD, pixel_mag, pixel_row_mag, col_center, row_center,
  dγ, pi_over_angles, sentinel :: T`.
* records `filter::BS.FilterType`, `cutoff::Float64`, `fov::NTuple{3,Float64}`.
"""
struct FBPPlan{T <: AbstractFloat, ARC, D <: NamedTuple}
    n_col::Int
    n_row::Int
    n_view::Int
    nx::Int
    ny::Int
    nz::Int
    kernel::Vector{T}
    tensors::D
    SAD::T
    SAD_sq::T
    SDD::T
    pixel_mag::T
    pixel_row_mag::T
    col_center::T
    row_center::T
    dγ::T
    pi_over_angles::T
    sentinel::T
    filter::BS.FilterType
    cutoff::Float64
    fov::NTuple{3, Float64}
end

"""
    FBPPlan(plan::FBPPlan; tensors = plan.tensors)

Rebuild a plan with a replaced `tensors` NamedTuple (same field names, any
array types) — the hook for moving the plan's constants to a device / into a
Reactant program.
"""
function FBPPlan(plan::FBPPlan{T, ARC}; tensors = plan.tensors) where {T, ARC}
    keys(tensors) == keys(plan.tensors) ||
        throw(ArgumentError("FBPPlan: replacement tensors must have keys $(keys(plan.tensors))"))
    return FBPPlan{T, ARC, typeof(tensors)}(
        plan.n_col, plan.n_row, plan.n_view, plan.nx, plan.ny, plan.nz,
        plan.kernel, tensors,
        plan.SAD, plan.SAD_sq, plan.SDD, plan.pixel_mag, plan.pixel_row_mag,
        plan.col_center, plan.row_center, plan.dγ, plan.pi_over_angles,
        plan.sentinel, plan.filter, plan.cutoff, plan.fov)
end

is_arc_plan(::FBPPlan{T, ARC}) where {T, ARC} = ARC

function Base.show(io::IO, p::FBPPlan{T, ARC}) where {T, ARC}
    print(io, "FBPPlan{$T, ", ARC ? ":arc" : ":flat", "}(sino=($(p.n_col), $(p.n_row), $(p.n_view)), ",
        "vol=($(p.nx), $(p.ny), $(p.nz)), K=$(length(p.kernel)), filter=$(typeof(p.filter)), cutoff=$(p.cutoff))")
end

# -- host helpers -------------------------------------------------------------

"""
    _kernel_length(n_col, cutoff) -> K

Legacy kernel-length rule (`filter_sinogram!` / `create_fdk_recon_workspace`):
`raw = max(ceil(2·n_col·cutoff), 64)`, made odd, capped at full support
`2·n_col − 1`.
"""
function _kernel_length(n_col::Int, cutoff::Float64)
    raw_size = max(Int(ceil(2 * n_col * cutoff)), 64)
    return min(raw_size + (1 - raw_size % 2), 2 * n_col - 1)
end

"""
    _toeplitz(kernel, n_col) -> H :: (n_col, n_col)

Zero-padded correlation matrix of the odd, centred tap vector `kernel`.
The legacy convolution loop is

    filtered[c] = Σ_{k=1}^{K} sino[c + k − K÷2 − 1] · kernel[k]   (terms with
                  1 ≤ c + k − K÷2 − 1 ≤ n_col only)

and substituting `j = c + k − K÷2 − 1` gives exactly `Σ_j H[c, j]·sino[j]`
with `H[c, j] = kernel[j − c + K÷2 + 1]` for `|j − c| ≤ K÷2`, zero otherwise.
Hence `H * sino_rows` IS the legacy zero-padded spatial convolution, term for
term; only the floating-point summation order differs (BLAS vs sequential).
"""
function _toeplitz(kernel::Vector{T}, n_col::Int) where {T}
    K = length(kernel)
    isodd(K) || throw(ArgumentError("_toeplitz: kernel length must be odd (got $K)"))
    kh = K ÷ 2
    H = zeros(T, n_col, n_col)
    for j in 1:n_col, i in 1:n_col
        d = j - i
        if -kh <= d <= kh
            H[i, j] = kernel[d + kh + 1]
        end
    end
    return H
end

"""
    _cosine_weights(T, geom, n_col, n_row, arc) -> (n_col, n_row)

Host mirror of `cosine_weight!` with the same `T` arithmetic order:
`:flat` → `SDD / √(SDD² + u² + v²)`; `:arc` → `cos(γ)·SDD / √(SDD² + v²)`.
"""
function _cosine_weights(::Type{T}, geom::BS.CTGeometry, n_col::Int, n_row::Int, arc::Bool) where {T}
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    magnification = T(geom.SDD / geom.SAD)
    SDD = T(geom.SDD)
    SDD_sq = SDD * SDD
    col_center = (T(n_col) + one(T)) / T(2)
    row_center = (T(n_row) + one(T)) / T(2)
    dγ = T(geom.pixel_size / geom.SAD)
    w = Matrix{T}(undef, n_col, n_row)
    for row in 1:n_row, col in 1:n_col
        w[col, row] = if arc
            γ = (T(col) - col_center) * dγ
            v = (T(row) - row_center) * pixel_row_size * magnification
            cos(γ) * SDD / sqrt(SDD_sq + v^2)
        else
            u = (T(col) - col_center) * pixel_size * magnification
            v = (T(row) - row_center) * pixel_row_size * magnification
            SDD / sqrt(SDD_sq + u^2 + v^2)
        end
    end
    return w
end

"""
    _voxel_axis(T, fov_extent, n) -> Vector{T}

Host mirror of the `backproject!` voxel-centre rule
`vol_min + (i − ½)·voxel_size` with `vol_min = T(−fov/2)`, `voxel_size = T(fov)/T(n)`.
"""
function _voxel_axis(::Type{T}, fov_extent::Float64, n::Int) where {T}
    vol_min = T(-fov_extent / 2)
    voxel_size = T(fov_extent) / T(n)
    half = T(0.5)
    return T[vol_min + (T(i) - half) * voxel_size for i in 1:n]
end

"""
    _fov_outside(T, fov, nx, ny) -> Array{Bool,3} (nx, ny, 1)

Host mirror of the `apply_fov_mask!` predicate: `true` where the voxel centre
`((i − ½ − n/2)·Δ)` lies outside the circle of radius `min(fov_x, fov_y)/2`.
"""
function _fov_outside(::Type{T}, fov::NTuple{3, Float64}, nx::Int, ny::Int) where {T}
    fov_x, fov_y = fov[1], fov[2]
    radius = T(min(fov_x, fov_y) / 2)
    radius_sq = radius * radius
    voxel_x = T(fov_x / nx)
    voxel_y = T(fov_y / ny)
    m = Array{Bool, 3}(undef, nx, ny, 1)
    for iy in 1:ny, ix in 1:nx
        x = (T(ix) - T(0.5) - T(nx) / T(2)) * voxel_x
        y = (T(iy) - T(0.5) - T(ny) / T(2)) * voxel_y
        m[ix, iy, 1] = x * x + y * y > radius_sq
    end
    return m
end

"""
    fbp_plan(geom, vol_shape; filter = :ram_lak, cutoff = 1.0, T = Float32,
             fov = geom.fov, sentinel = -0.04) -> FBPPlan

Build the immutable plan for the functional FDK stage from a `CTGeometry`
and the output volume shape `(nx, ny, nz)`.

* `filter` — a `Symbol` (`:ram_lak, :shepp_logan, :cosine, :hamming, :hann,
  :standard, :soft, :bone`) or a legacy `FilterType` instance.  NOTE the legacy
  workspace default is `StandardFilter()`; this stage defaults to `:ram_lak`.
* `cutoff` — legacy kernel-truncation fraction (see `_kernel_length`).
* `T` — tensor element type.
* `fov` — reconstruction FOV `(fx, fy, fz)` (same units as `geom.fov`);
  defaults to `geom.fov` (the `fdk_reconstruct(sino, geom, size, fov)` variant).
* `sentinel` — μ written outside the inscribed FOV circle (legacy `-0.04`).

Throws for helical geometries: the legacy path routes those to the rebinned
WFBP chain, which is out of scope for this stage.
"""
function fbp_plan(
        geom::BS.CTGeometry, vol_shape::NTuple{3, Int};
        filter::Union{Symbol, BS.FilterType} = :ram_lak,
        cutoff::Real = 1.0,
        T::Type{<:AbstractFloat} = Float32,
        fov::NTuple{3, <:Real} = geom.fov,
        sentinel::Real = -0.04,
    )
    BS.is_helical(geom) && throw(ArgumentError(
        "fbp_plan: helical geometry (table_feed = $(geom.table_feed) ≠ 0) is not supported by " *
        "the functional FDK stage; the legacy path uses the rebinned WFBP chain " *
        "(`wfbp_helical_reconstruct`) for helical scans, which is out of scope here."))
    all(>(0), vol_shape) || throw(ArgumentError("fbp_plan: vol_shape must be positive, got $vol_shape"))
    0 < cutoff <= 1 || throw(ArgumentError("fbp_plan: cutoff must be in (0, 1], got $cutoff"))

    ftype = filter isa Symbol ? BS.filter_from_symbol(filter) : filter
    cutoff64 = Float64(cutoff)
    fov64 = (Float64(fov[1]), Float64(fov[2]), Float64(fov[3]))
    arc = BS.is_arc(geom)

    n_col, n_row = geom.n_cols, geom.n_rows
    n_view = size(geom.source_positions, 2)
    n_view == geom.n_angles || throw(ArgumentError(
        "fbp_plan: geom.n_angles = $(geom.n_angles) but source_positions has $n_view columns"))
    nx, ny, nz = vol_shape

    # ── filtering constants (legacy host helpers) ────────────────────────────
    K = _kernel_length(n_col, cutoff64)
    kernel = BS.create_spatial_kernel(K, ftype, T(geom.pixel_size))
    if arc
        # equiangular fan-beam kernel correction (axial arc only — matches
        # create_fdk_recon_workspace's `is_arc(geom) && !is_helical(geom)`)
        BS.equiangular_kernel_scale!(kernel, geom.pixel_size / geom.SAD)
    end
    H = _toeplitz(kernel, n_col)
    cos_weights = _cosine_weights(T, geom, n_col, n_row, arc)

    # ── backprojection constants (mirror backproject!'s typed scalars) ───────
    magnification = T(geom.SDD / geom.SAD)
    pixel_size = T(geom.pixel_size)
    pixel_row_size = T(geom.pixel_row_size)
    SAD = T(geom.SAD)
    SAD_sq = SAD * SAD
    SDD = T(geom.SDD)
    pixel_mag = pixel_size * magnification
    pixel_row_mag = pixel_row_size * magnification
    dγ = T(geom.pixel_size / geom.SAD)
    col_center = (T(n_col) + one(T)) / T(2)
    row_center = (T(n_row) + one(T)) / T(2)
    pi_over_angles = T(π) / T(n_view)

    X = reshape(_voxel_axis(T, fov64[1], nx), nx, 1, 1, 1)
    Y = reshape(_voxel_axis(T, fov64[2], ny), 1, ny, 1, 1)
    Z = reshape(_voxel_axis(T, fov64[3], nz), 1, 1, nz, 1)

    tensors = (
        cos_weights = cos_weights,
        H = H,
        source_positions = Matrix{T}(T.(geom.source_positions)),
        detector_centers = Matrix{T}(T.(geom.detector_centers)),
        detector_u = Matrix{T}(T.(geom.detector_u)),
        detector_v = Matrix{T}(T.(geom.detector_v)),
        X = X, Y = Y, Z = Z,
        fov_outside = _fov_outside(T, fov64, nx, ny),
    )

    return FBPPlan{T, arc, typeof(tensors)}(
        n_col, n_row, n_view, nx, ny, nz, kernel, tensors,
        SAD, SAD_sq, SDD, pixel_mag, pixel_row_mag, col_center, row_center, dγ,
        pi_over_angles, T(sentinel), ftype, cutoff64, fov64)
end

# -----------------------------------------------------------------------------
# Shape guards (sizes are static → host checks)
# -----------------------------------------------------------------------------

function _check_sino(a::AbstractArray{<:Any, 3}, plan::FBPPlan, what::String)
    size(a) == (plan.n_col, plan.n_row, plan.n_view) || throw(DimensionMismatch(
        "$what: expected (n_col, n_row, n_view) = $((plan.n_col, plan.n_row, plan.n_view)), got $(size(a))"))
    return nothing
end

function _check_view(a::AbstractArray{<:Any, 2}, plan::FBPPlan, what::String)
    size(a) == (plan.n_col, plan.n_row) || throw(DimensionMismatch(
        "$what: expected (n_col, n_row) = $((plan.n_col, plan.n_row)), got $(size(a))"))
    return nothing
end

function _check_vol(a::AbstractArray{<:Any, 3}, plan::FBPPlan, what::String)
    size(a) == (plan.nx, plan.ny, plan.nz) || throw(DimensionMismatch(
        "$what: expected (nx, ny, nz) = $((plan.nx, plan.ny, plan.nz)), got $(size(a))"))
    return nothing
end

# -----------------------------------------------------------------------------
# Stage 1 — cosine weighting + ramp filtering
# -----------------------------------------------------------------------------

"""
    filter_views(sino, plan) -> filtered :: (n_col, n_row, n_view)

Pure equivalent of `filter_sinogram!(copy(sino), geom; filter, cutoff)`:

1. cosine pre-weight every `(col, row)` by `plan.tensors.cos_weights`;
2. zero-padded spatial-domain ramp convolution along columns, expressed as
   ONE matmul with the Toeplitz matrix `H` (see `_toeplitz` for the term-by-term
   identity with the legacy tap loop).

`sino` is not modified.
"""
function filter_views(sino::AbstractArray{<:Any, 3}, plan::FBPPlan{T}) where {T}
    _check_sino(sino, plan, "filter_views")
    w = reshape(plan.tensors.cos_weights, plan.n_col, plan.n_row, 1)
    weighted = sino .* w
    rows = reshape(weighted, plan.n_col, plan.n_row * plan.n_view)
    filt = plan.tensors.H * rows
    return reshape(filt, plan.n_col, plan.n_row, plan.n_view)
end

# -----------------------------------------------------------------------------
# Stage 2 — voxel-driven FDK backprojection
# -----------------------------------------------------------------------------
#
# The per-voxel, per-view arithmetic of `backproject_voxel` is written as a
# few BRANCH-FREE scalar helpers (only `ifelse`, `min`, `max`, `&`) and applied
# with one flat broadcast each.  Two reasons:
#   * Julia compiles a flat broadcast of a scalar function in well under a
#     second, whereas the equivalent ~35 nested fused `.op` chains took
#     ~10 s of inference per (T, arc) specialisation;
#   * Reactant traces the scalar body on `TracedRNumber`s into exactly the
#     same StableHLO ops, so nothing leaves the graph.
# Every helper keeps the legacy operation order so that in-bounds voxels are
# arithmetically identical to the AcceleratedKernels kernel.

# NaN-safe clamp: comparisons with NaN are false, so NaN → lo (a valid index
# after floor) instead of throwing in `floor(Int32, NaN)`; in-range values are
# returned unchanged.
@inline _clamp_nan(x, lo, hi) = ifelse(x >= lo, ifelse(x <= hi, x, hi), lo)

# Ray → (col_f, row_f) for a planar detector (legacy :flat branch).  Returns
# (-1, -1) when the legacy `abs(sv_dot_sd) < 1e-10` guard would skip the view
# (that value fails the in-bounds test below).
@inline function _ray_flat(x, y, z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
        pixel_mag, pixel_row_mag, col_center, row_center, guard_eps, oob)
    sv_x = x - sx; sv_y = y - sy; sv_z = z - sz
    sd_x = dcx - sx; sd_y = dcy - sy; sd_z = dcz - sz
    sd_len_sq = sd_x * sd_x + sd_y * sd_y + sd_z * sd_z
    sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z
    ok = abs(sv_dot_sd) >= guard_eps
    t = sd_len_sq / ifelse(ok, sv_dot_sd, one(sv_dot_sd))
    proj_x = sx + t * sv_x; proj_y = sy + t * sv_y; proj_z = sz + t * sv_z
    dp_x = proj_x - dcx; dp_y = proj_y - dcy; dp_z = proj_z - dcz
    u = (dp_x * dux + dp_y * duy + dp_z * duz) / pixel_mag
    v = (dp_x * dvx + dp_y * dvy + dp_z * dvz) / pixel_row_mag
    return (ifelse(ok, u + col_center, oob), ifelse(ok, v + row_center, oob))
end

# Ray → (col_f, row_f) for an equiangular detector (legacy :arc branch).
@inline function _ray_arc(x, y, z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
        pixel_mag, pixel_row_mag, col_center, row_center, dγ, guard_eps, oob)
    sv_x = x - sx; sv_y = y - sy; sv_z = z - sz
    sd_x = dcx - sx; sd_y = dcy - sy; sd_z = dcz - sz
    sv_dot_sd = sv_x * sd_x + sv_y * sd_y + sv_z * sd_z
    ok = abs(sv_dot_sd) >= guard_eps
    sd_len = sqrt(sd_x * sd_x + sd_y * sd_y)          # = SDD (in-plane)
    a_c = (sv_x * sd_x + sv_y * sd_y) / sd_len        # along central ray
    a_u = sv_x * dux + sv_y * duy                     # along detector û
    γv = atan(a_u, a_c)
    L_in = sqrt(sv_x * sv_x + sv_y * sv_y)
    v_arc = sv_z * (sd_len / L_in) / pixel_row_mag
    return (ifelse(ok, γv / dγ + col_center, oob), ifelse(ok, v_arc + row_center, oob))
end

@inline _col_flat(args...) = _ray_flat(args...)[1]
@inline _row_flat(args...) = _ray_flat(args...)[2]
@inline _col_arc(args...) = _ray_arc(args...)[1]
@inline _row_arc(args...) = _ray_arc(args...)[2]

# FDK distance weight SAD²/dist² (arc: in-plane distance; flat: full 3-D).
@inline function _fdk_weight_arc(x, y, z, sx, sy, sz, SAD_sq)
    sv_x = x - sx; sv_y = y - sy
    return SAD_sq / (sv_x * sv_x + sv_y * sv_y)
end
@inline function _fdk_weight_flat(x, y, z, sx, sy, sz, SAD_sq)
    sv_x = x - sx; sv_y = y - sy; sv_z = z - sz
    return SAD_sq / (sv_x * sv_x + sv_y * sv_y + sv_z * sv_z)
end

# in-bounds on the half-pixel-extended detector (legacy guard)
@inline _inb(col_f, row_f, lo, c_hi, r_hi) =
    (col_f >= lo) & (col_f <= c_hi) & (row_f >= lo) & (row_f <= r_hi)

# Legacy: `unsafe_trunc(Int32, col_f)` evaluated only in-bounds (col_f ≥ 0.5 > 0)
# where trunc == floor.  We floor a NaN-safe clamped copy so out-of-bounds
# voxels (masked later) never yield an invalid index; in-bounds untouched.
@inline _floor_idx(f, hi) = floor(Int32, _clamp_nan(f, zero(f), hi))
@inline function _frac(f, hi)
    c = _clamp_nan(f, zero(f), hi)
    return c - floor(c)
end

# edge-clamped linear index of the (col_lo + dc, row_lo + dr) sample
@inline function _lin_idx(col_lo, row_lo, dc, dr, n_col, n_row, off)
    ic = min(max(col_lo + dc, Int32(1)), n_col)
    ir = min(max(row_lo + dr, Int32(1)), n_row)
    return ic + (ir - Int32(1)) * n_col + off
end

# bilinear sample × weight, masked (legacy: accumulate only when in-bounds)
@inline function _bilinear(inb, wc, wr, s00, s10, s01, s11, w)
    o = one(wc)
    val = (o - wc) * (o - wr) * s00 + wc * (o - wr) * s10 + (o - wc) * wr * s01 + wc * wr * s11
    return ifelse(inb, val * w, zero(val))
end

# `f.(args...)` behind a @noinline barrier: each broadcast is compiled once as
# its own small method instead of having its loop inlined (and re-optimised by
# LLVM) inside the large `_bp_views` body — ~2× less compile time per
# specialisation, identical arithmetic.  Reactant's broadcast overloads
# dispatch through `Base.broadcast` unchanged.
@noinline _bc(f, args...) = broadcast(f, args...)

"""
    _bp_views(filt_vec, plan, views, offsets, weighted) -> (nx, ny, nz, 1)

Contribution of the views `views` (a `UnitRange`, static length `B`) to the
backprojection, summed over the batch dimension.  `filt_vec` is the flattened
filtered data; `offsets[b]` is the Int32 linear offset of view `views[b]`'s
`(n_col, n_row)` block inside `filt_vec` (0 for a single 2-D view).

Mirrors `backproject_voxel` (weighted) / `backproject_voxel_matched`
(unweighted) per view: ray → detector coordinates, FDK weight `SAD²/dist²`,
in-bounds mask on the half-pixel-extended detector, bilinear gather with
edge-clamped indices.  The π/n_view scaling is applied by the caller.
"""
# Per-view geometry as one (12, n_view) table: rows = source (3), detector centre (3), u (3), v (3).
_geom_table(tb) = vcat(tb.source_positions, tb.detector_centers, tb.detector_u, tb.detector_v)

@noinline function _bp_views(filt_vec::AbstractVector, plan::FBPPlan{T, ARC},
        views::UnitRange{Int}, offsets::AbstractVector{Int32}, weighted::Bool) where {T, ARC}
    gt = _geom_table(plan.tensors)[:, views]
    return _bp_chunk(filt_vec, plan, gt, reshape(offsets, 1, 1, 1, length(views)), weighted)
end

# Backprojection of one chunk of views: `gt :: (12, B)` geometry table slice (host or traced),
# `off :: (1,1,1,B)` linear offsets of the chunk's views inside `filt_vec`.
@noinline function _bp_chunk(filt_vec::AbstractVector, plan::FBPPlan{T, ARC},
        gt::AbstractMatrix, off::AbstractArray{<:Integer, 4}, weighted::Bool) where {T, ARC}
    B = size(gt, 2)
    tb = plan.tensors
    r4(i) = reshape(gt[i:i, :], 1, 1, 1, B)

    # per-view geometry as (1,1,1,B) tensors
    sx  = r4(1);  sy  = r4(2);  sz  = r4(3)
    dcx = r4(4);  dcy = r4(5);  dcz = r4(6)
    dux = r4(7);  duy = r4(8);  duz = r4(9)
    dvx = r4(10); dvy = r4(11); dvz = r4(12)
    X, Y, Z = tb.X, tb.Y, tb.Z

    # host-resolved arc/flat dispatch (type parameter), host scalars
    guard_eps = T(1e-10)
    oob = -one(T)
    col_f, row_f = if ARC
        (_bc(_col_arc, X, Y, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, plan.dγ, guard_eps, oob),
         _bc(_row_arc, X, Y, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, plan.dγ, guard_eps, oob))
    else
        (_bc(_col_flat, X, Y, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, guard_eps, oob),
         _bc(_row_flat, X, Y, Z, sx, sy, sz, dcx, dcy, dcz, dux, duy, duz, dvx, dvy, dvz,
             plan.pixel_mag, plan.pixel_row_mag, plan.col_center, plan.row_center, guard_eps, oob))
    end

    half = T(0.5)
    nc = T(plan.n_col)
    nr = T(plan.n_row)
    inb = _bc(_inb, col_f, row_f, half, nc + half, nr + half)

    col_lo = _bc(_floor_idx, col_f, nc + T(2))
    row_lo = _bc(_floor_idx, row_f, nr + T(2))
    w_col = _bc(_frac, col_f, nc + T(2))
    w_row = _bc(_frac, row_f, nr + T(2))

    n_col32 = Int32(plan.n_col)
    n_row32 = Int32(plan.n_row)
    i00 = _bc(_lin_idx, col_lo, row_lo, Int32(0), Int32(0), n_col32, n_row32, off)
    i10 = _bc(_lin_idx, col_lo, row_lo, Int32(1), Int32(0), n_col32, n_row32, off)
    i01 = _bc(_lin_idx, col_lo, row_lo, Int32(0), Int32(1), n_col32, n_row32, off)
    i11 = _bc(_lin_idx, col_lo, row_lo, Int32(1), Int32(1), n_col32, n_row32, off)

    s00 = filt_vec[i00]
    s10 = filt_vec[i10]
    s01 = filt_vec[i01]
    s11 = filt_vec[i11]

    # FDK distance weight; the matched adjoint uses a unit weight of the SAME
    # array type so `weighted` is a plain runtime Bool (one compiled body per
    # (T, ARC) instead of a Val specialisation).
    w = if weighted
        ARC ? _bc(_fdk_weight_arc, X, Y, Z, sx, sy, sz, plan.SAD_sq) :
              _bc(_fdk_weight_flat, X, Y, Z, sx, sy, sz, plan.SAD_sq)
    else
        fill(one(T), 1, 1, 1, 1)
    end
    contrib = _bc(_bilinear, inb, w_col, w_row, s00, s10, s01, s11, w)
    return sum(contrib; dims = 4)
end

"""
    _view_batches(n_view, view_batch) -> Vector{UnitRange{Int}}

Static partition of `1:n_view` into consecutive chunks of length ≤ `view_batch`.
"""
function _view_batches(n_view::Int, view_batch::Int)
    view_batch >= 1 || throw(ArgumentError("view_batch must be ≥ 1, got $view_batch"))
    return [s:min(s + view_batch - 1, n_view) for s in 1:view_batch:n_view]
end

"""
    backproject_view(filt_view, plan, view_index; weighted = true) -> (nx, ny, nz)

Backprojection of ONE filtered view `filt_view :: (n_col, n_row)` at geometry
index `view_index`.  With `weighted = true` this includes the FDK distance
weight and the `π/n_view` scaling, so the sum of `backproject_view` over all
views equals `backproject(filt, plan)` up to summation order.
"""
function backproject_view(filt_view::AbstractArray{<:Any, 2}, plan::FBPPlan{T}, view_index::Int;
        weighted::Bool = true) where {T}
    _check_view(filt_view, plan, "backproject_view")
    1 <= view_index <= plan.n_view || throw(ArgumentError("view_index $view_index ∉ 1:$(plan.n_view)"))
    c = _bp_views(vec(filt_view), plan, view_index:view_index, Int32[0], weighted)
    out = reshape(c, plan.nx, plan.ny, plan.nz)
    return weighted ? out .* plan.pi_over_angles : out
end

"""
    backproject(filt, plan; weighted = true, view_batch = 1) -> (nx, ny, nz)

Pure equivalent of `backproject!(zeros(nx,ny,nz), filt, geom; weighted)` for an
axial geometry.

* `weighted = true`  — FDK: `π/n_view · Σ_views SAD²/dist² · bilinear(filt)`
  (`backproject_voxel`).
* `weighted = false` — matched/unweighted `Σ_views bilinear(filt)`
  (`backproject_voxel_matched`).
* `view_batch` — views processed per broadcast step (host config; every
  intermediate is `(nx, ny, nz, view_batch)`).  `1` reproduces the legacy
  sequential view summation order exactly; larger values shrink the graph.
* `loop` — with `view_batch > 1`, run the batches inside one compiled loop
  (`_sum_over_batches`; program size independent of `n_view`) instead of
  unrolling them.

Legacy `backproject_voxel` tracks `w_acc = Σ weight` but only uses it as a
`w_acc > 0` guard (never as a divisor); since `acc == 0` whenever `w_acc == 0`
the guard is the identity and is not reproduced here.
"""
function backproject(filt::AbstractArray{<:Any, 3}, plan::FBPPlan{T};
        weighted::Bool = true, view_batch::Int = 1, loop::Bool = true) where {T}
    _check_sino(filt, plan, "backproject")
    fv = vec(filt)
    block = plan.n_col * plan.n_row
    if view_batch > 1 && loop
        # one compiled loop over view batches (program size independent of n_view)
        G = _geom_table(plan.tensors)
        acc0 = _zeros(filt, T, (plan.nx, plan.ny, plan.nz, 1))
        chunk_fn = (start, len, G, filt, plan) -> begin                       # closes over host data only
            gt = _dslice(G, start, len, 2)
            fc = _dslice(filt, start, len, 3)
            off = _on_device(reshape(Int32[(k - 1) * block for k in 1:len], 1, 1, 1, len), filt)
            return _bp_chunk(vec(fc), plan, gt, off, weighted)
        end
        acc = _sum_over_batches(chunk_fn, plan.n_view, view_batch, acc0, (G, filt, plan), filt)
        out = reshape(acc, plan.nx, plan.ny, plan.nz)
        return weighted ? out .* plan.pi_over_angles : out
    end
    batches = _view_batches(plan.n_view, view_batch)
    offs(r) = Int32[(a - 1) * block for a in r]
    acc = _bp_views(fv, plan, batches[1], offs(batches[1]), weighted)
    for b in 2:length(batches)
        r = batches[b]
        acc = acc .+ _bp_views(fv, plan, r, offs(r), weighted)
    end
    out = reshape(acc, plan.nx, plan.ny, plan.nz)
    return weighted ? out .* plan.pi_over_angles : out
end

# -----------------------------------------------------------------------------
# Stage 3 — FOV mask
# -----------------------------------------------------------------------------

"""
    fov_mask(vol, plan; sentinel = plan.sentinel) -> (nx, ny, nz)

Pure equivalent of `apply_fov_mask!(copy(vol), geom; sentinel_μ = sentinel)`:
voxels whose centre lies outside the inscribed circle of the xy FOV are set to
`sentinel`, everything else is passed through untouched.
"""
function fov_mask(vol::AbstractArray{<:Any, 3}, plan::FBPPlan{T}; sentinel::Real = plan.sentinel) where {T}
    _check_vol(vol, plan, "fov_mask")
    return _bc(_select_sentinel, plan.tensors.fov_outside, T(sentinel), vol)
end

# `ifelse(outside, sentinel, v)` with the sentinel promoted to the element type
# of `v` FIRST: with a host `Bool` mask and a traced `v`, a bare
# `ifelse(::Bool, ::Float32, ::TracedRNumber)` infers a `Union` eltype and
# Reactant's broadcast cannot allocate it; `oftype` keeps both branches the
# same type in plain Julia and under tracing alike.
@inline _select_sentinel(outside, s, v) = ifelse(outside, oftype(v, s), v)

# -----------------------------------------------------------------------------
# Full FDK
# -----------------------------------------------------------------------------

"""
    fdk(sino, plan; view_batch = 1, sentinel = plan.sentinel) -> (nx, ny, nz)

`fov_mask(backproject(filter_views(sino, plan), plan), plan)` — the pure
counterpart of `reconstruct!(::FDKReconWorkspace, sino, geom)` /
`fdk_reconstruct(sino, geom, size; filter, cutoff)` for AXIAL geometries
(`:flat` and `:arc`).
"""
function fdk(sino::AbstractArray{<:Any, 3}, plan::FBPPlan{T};
        view_batch::Int = 1, loop::Bool = true, sentinel::Real = plan.sentinel) where {T}
    filt = filter_views(sino, plan)
    vol = backproject(filt, plan; weighted = true, view_batch = view_batch, loop = loop)
    return fov_mask(vol, plan; sentinel = sentinel)
end
