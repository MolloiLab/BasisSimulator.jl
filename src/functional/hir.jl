# =============================================================================
# Functional Hybrid IR — ordered-subsets PWLS with a Huber prior
# =============================================================================
#
# Pure, mutation-free, array-generic re-implementation of
# `reconstruct!(::HIRReconWorkspace, …)` (src/api/driver.jl) written as a
# tensor program that Reactant can compile and Enzyme can differentiate.
#
# Contract (see the `Functional` module preamble):
#   • every function is a pure function of arrays + host scalars / an immutable
#     `HIRPlan`; inputs are never mutated and the iterate is REBOUND, never
#     updated in place;
#   • every array shape is a function of host configuration only — ordered
#     subsets are padded to EQUAL size and the padding is neutralised by a
#     multiplicative validity mask (see `HIRPlan.subset_valid`);
#   • no scalar indexing inside stage functions: broadcasting, static slices,
#     static gathers along the view / z axis, `ifelse.`/`clamp.` masks;
#   • loop trip counts (epochs × subsets) are host constants; no data-dependent
#     control flow, no early exit on data;
#   • element type generic `T <: AbstractFloat`; no Float64 literals in the
#     array code;
#   • the projector pair is PASSED IN as pure operator functions so the stage
#     is operator-agnostic:
#         A(vol, view_indices)      -> (n_col, n_row, length(view_indices))
#         At(sino_sub, view_indices) -> work volume (nx, ny, nz_work)
#     with `At` the matched (unweighted) transpose of `A` restricted to the
#     circular support, i.e. exactly what the legacy loop calls through
#     `_project_mono_hir!` / `_backproject_mono!(…; circular_support = true)`.
#
# Legacy semantics reproduced (operation order preserved for bit-level parity):
#   1. seed the halo'd work iterate from the FDK init by z-clamped continuation
#      (`_hir_seed_work!`);
#   2. `nepochs == 0` ⇒ extract + FOV mask and return (pure FBP path);
#   3. zero the iterate outside the circular FOV (`apply_fov_mask!` with a zero
#      sentinel);
#   4. statistical weights `exp(-clamp(y, 0, 10)) + ε` (× air reference when
#      given), folded once with `W_proj = 1 / (A·support)`;
#   5. epoch × subset loop: Huber gradient (per subset for DD, per epoch for
#      Siddon), subset forward projection, weighted residual, matched
#      backprojection, SIRT-style update with `V_inv = 1 / (Aᵀ·1)`,
#      `reg_views_scale = n_views / 1000`, `:siddon` λ scale, circular-support
#      zeroing inside the update;
#   6. extract the output slab and apply the clinical FOV sentinel (−0.04).
#
# Dispatch note: array arguments are typed `AbstractArray{<:Any, 3}` and the
# scalar type `T` is taken from the plan ONLY.  Reactant's `TracedRArray{T,N}`
# is an `AbstractArray{TracedRNumber{T}, N}`, so a signature that binds `T`
# from the array element type (`x::AbstractArray{T,3}, plan::HIRPlan{T}`) can
# never match a traced array.
# =============================================================================

"""
    HIRPlan{T}

Host-side constants for one functional HIR reconstruction.  Immutable; built
once by [`hir_plan`](@ref) and shared by every call of
[`hir_reconstruct`](@ref) on the same geometry / strength.

# Fields
- `params::HIRParams` — the strength table row (host only).
- `projector::Symbol` — `:dd_fast` (default), `:dd`, or `:siddon`; selects the
  λ scale and the Huber-gradient cadence, exactly as the legacy loop does.
- `geom`, `work_geom::CTGeometry` — caller geometry and the halo'd iterate
  geometry (the operators `A`/`At` must be built on `work_geom`).
- `sino_shape`, `vol_shape`, `work_shape` — static array sizes.
- `output_z::UnitRange{Int}` — output slab inside the work volume.
- `seed_z::Vector{Int}` — static z-gather that seeds the work volume from the
  init (clamped continuation into the halo).
- `all_views::Vector{Int}` — `1:n_views`, for the normalisation projections.
- `subset_views::Vector{Vector{Int}}` — per-subset global view indices, PADDED
  to `subset_size` by repeating the subset's last view.
- `subset_valid::Vector{Array{T,3}}` — per-subset `(1, 1, subset_size)`
  multiplicative masks: `1` on real views, `0` on padding.
- `fov_mask::Array{Bool,3}` — `(nx, ny, 1)`, `true` inside the circular FOV
  using the `apply_fov_mask!` voxel-centre formula.
- `update_mask::Array{Bool,3}` — `(nx, ny, 1)`, same circle but with the
  update-kernel's arithmetic (`fov / nx` computed in `T`); the two can differ
  by one ulp at the rim, so both are kept for bit-level fidelity.
- `air_reference` — `nothing` or an `(n_col, n_row, 1)` array of `T`.
- scalars: `λ`, `δ`, `relaxation`, `subset_scale`, `reg_views_scale`,
  `stat_eps`, `weight_eps`, `sentinel`, `y_clip`; `nepochs`, `n_subsets`,
  `subset_size`, `reg_per_subset`.
"""
struct HIRPlan{T <: AbstractFloat, AR}
    params::BS.HIRParams
    projector::Symbol
    geom::BS.CTGeometry
    work_geom::BS.CTGeometry
    sino_shape::NTuple{3, Int}
    vol_shape::NTuple{3, Int}
    work_shape::NTuple{3, Int}
    output_z::UnitRange{Int}
    seed_z::Vector{Int}
    all_views::Vector{Int}
    n_subsets::Int
    subset_size::Int
    subset_views::Vector{Vector{Int}}
    subset_valid::Vector{Array{T, 3}}
    fov_mask::Array{Bool, 3}
    update_mask::Array{Bool, 3}
    air_reference::AR
    nepochs::Int
    reg_per_subset::Bool
    λ::T
    δ::T
    relaxation::T
    subset_scale::T
    reg_views_scale::T
    stat_eps::T
    weight_eps::T
    sentinel::T
    y_clip::T
end

# -----------------------------------------------------------------------------
# Host-side plan construction
# -----------------------------------------------------------------------------

# Circular FOV membership with the exact arithmetic of `apply_fov_mask!`
# (src/reconstruction/fbp/fdk.jl): voxel pitch computed in Float64 then
# converted.
function _fov_mask_apply_formula(::Type{T}, nx::Int, ny::Int, fov) where {T}
    fov_x, fov_y = fov[1], fov[2]
    radius = T(min(fov_x, fov_y) / 2)
    radius_sq = radius * radius
    voxel_x = T(fov_x / nx)
    voxel_y = T(fov_y / ny)
    mask = Array{Bool, 3}(undef, nx, ny, 1)
    for iy in 1:ny, ix in 1:nx
        x = (T(ix) - T(0.5) - T(nx) / T(2)) * voxel_x
        y = (T(iy) - T(0.5) - T(ny) / T(2)) * voxel_y
        mask[ix, iy, 1] = !(x * x + y * y > radius_sq)
    end
    return mask
end

# Circular FOV membership with the exact arithmetic of the legacy update kernel
# in `reconstruct!(::HIRReconWorkspace)`: voxel pitch computed in `T`.
function _fov_mask_update_formula(::Type{T}, nx::Int, ny::Int, fov) where {T}
    dx = T(fov[1]) / T(nx)
    dy = T(fov[2]) / T(ny)
    radius_sq = T(min(fov[1], fov[2]) / 2)^2
    half = T(0.5)
    mask = Array{Bool, 3}(undef, nx, ny, 1)
    for iy in 1:ny, ix in 1:nx
        x = (T(ix) - half - T(nx) / T(2)) * dx
        y = (T(iy) - half - T(ny) / T(2)) * dy
        mask[ix, iy, 1] = !(x * x + y * y > radius_sq)
    end
    return mask
end

# Ordered subsets padded to equal size.  The legacy `create_ordered_subsets`
# deals views round-robin (`mod1(i, n_subsets)`), so the first
# `n_views % n_subsets` subsets carry one extra view.  Static shapes require
# every subset to have the same length: shorter subsets are padded by
# REPEATING THEIR LAST VIEW and the padded slot is neutralised by a zero in
# `subset_valid`.  Because the residual on a padded slot is multiplied by
# exactly zero and the backprojector is linear, the padded view contributes
# exact zeros to the correction, so the update is unchanged; the only cost is
# one wasted forward/back projection view per padded subset.
function _padded_subsets(::Type{T}, n_views::Int, n_subsets::Int) where {T}
    raw = BS.create_ordered_subsets(n_views, n_subsets)
    subset_size = maximum(length, raw)
    views = Vector{Vector{Int}}(undef, n_subsets)
    valid = Vector{Array{T, 3}}(undef, n_subsets)
    for s in 1:n_subsets
        r = raw[s]
        isempty(r) && throw(ArgumentError(
            "n_views = $n_views is too small for $n_subsets ordered subsets"))
        padded = vcat(r, fill(r[end], subset_size - length(r)))
        v = zeros(T, 1, 1, subset_size)
        v[1, 1, 1:length(r)] .= one(T)
        views[s] = padded
        valid[s] = v
    end
    return subset_size, views, valid
end

"""
    hir_plan(geom, sino_shape, vol_shape, strength; projector = :dd_fast,
             air_reference = nothing, T = Float32) -> HIRPlan{T}
    hir_plan(geom, sino_shape, vol_shape, params::HIRParams; …)

Build the host constants for a functional HIR reconstruction.  Mirrors
`create_hir_recon_workspace` (subset construction, axial halo via
`_hir_axial_support`, strength table via `get_hir_params`) but allocates no
iteration buffers.  The second form accepts an explicit `HIRParams` (e.g. to
run a single epoch in a smoothness test).

`air_reference` may be an `(n_col, n_row)` or `(n_col, n_row, 1)` array; it is
converted to `T` and stored as `(n_col, n_row, 1)` so it broadcasts across
views exactly like the legacy per-ray lookup.
"""
function hir_plan(
        geom::BS.CTGeometry, sino_shape::NTuple{3, Int}, vol_shape::NTuple{3, Int},
        strength::Integer;
        projector::Symbol = :dd_fast,
        air_reference = nothing,
        T::Type{<:AbstractFloat} = Float32,
    )
    return hir_plan(geom, sino_shape, vol_shape, BS.get_hir_params(strength);
                    projector, air_reference, T)
end

function hir_plan(
        geom::BS.CTGeometry, sino_shape::NTuple{3, Int}, vol_shape::NTuple{3, Int},
        params::BS.HIRParams;
        projector::Symbol = :dd_fast,
        air_reference = nothing,
        T::Type{<:AbstractFloat} = Float32,
    )
    (projector === :dd || projector === :dd_fast || projector === :siddon) ||
        throw(ArgumentError("projector must be :dd, :dd_fast, or :siddon, got :$projector"))
    n_views = geom.n_angles
    sino_shape[3] == n_views || throw(DimensionMismatch(
        "sino_shape[3] = $(sino_shape[3]) must equal geom.n_angles = $n_views"))
    (sino_shape[1] == geom.n_cols && sino_shape[2] == geom.n_rows) || throw(DimensionMismatch(
        "sino_shape[1:2] = $(sino_shape[1:2]) must equal (n_cols, n_rows) = $((geom.n_cols, geom.n_rows))"))

    nepochs = params.nepochs
    work_geom, work_shape, output_z = BS._hir_axial_support(geom, vol_shape, nepochs > 0)
    nx, ny, nz = vol_shape
    work_nz = work_shape[3]
    z0 = first(output_z)
    seed_z = [clamp(kw - z0 + 1, 1, nz) for kw in 1:work_nz]

    n_subsets = params.n_subsets
    n_subsets > 0 || throw(ArgumentError("HIRParams.n_subsets must be > 0; got $n_subsets"))
    subset_size, subset_views, subset_valid = _padded_subsets(T, n_views, n_subsets)

    fov_mask = _fov_mask_apply_formula(T, nx, ny, geom.fov)
    update_mask = _fov_mask_update_formula(T, nx, ny, work_geom.fov)

    aref = if air_reference === nothing
        nothing
    else
        size(air_reference, 1) == sino_shape[1] && size(air_reference, 2) == sino_shape[2] ||
            throw(DimensionMismatch(
                "air_reference has size $(size(air_reference)); expected $(sino_shape[1:2])"))
        reshape(Array{T}(T.(air_reference)), sino_shape[1], sino_shape[2], 1)
    end

    # Same λ conversion as the legacy loop: the public table is calibrated to the
    # Siddon/BP normalisation; exact DDᵀ needs a 0.1× scale.
    λ = T(params.lambda) * (projector === :siddon ? one(T) : T(0.1))

    return HIRPlan{T, typeof(aref)}(
        params, projector, geom, work_geom,
        sino_shape, vol_shape, work_shape, output_z, seed_z, collect(1:n_views),
        n_subsets, subset_size, subset_views, subset_valid,
        fov_mask, update_mask, aref,
        nepochs, projector !== :siddon,
        λ, T(params.huber_delta), T(params.relaxation), T(n_subsets),
        T(n_views) / T(1000), T(1.0e-6), T(1.0e-8), T(-0.04), T(10),
    )
end

# -----------------------------------------------------------------------------
# Array stages (pure)
# -----------------------------------------------------------------------------

# `ifelse.(mask, a, b)` with a scalar `b`.  The scalar branch is lifted to the
# element type of `a` (`zero(a_i) + b == b` exactly) so the broadcast has one
# element type: under Reactant, `ifelse(::Bool, ::TracedRNumber, ::Float64)`
# infers `Number` and the broadcast cannot be materialised.
@inline _where(mask, a, b::Real) = ifelse.(mask, a, zero.(a) .+ b)

"""
    huber_gradient(x, δ) -> gradient of the 6-connected Huber roughness penalty

Pure port of `compute_huber_gradient!` (src/reconstruction/ir/utils.jl).  Face
differences are formed with static shifted slices; the six boundary cases of
the legacy kernel become explicit zero padding, and the Huber derivative
`ψ'(t) = t` for `|t| ≤ δ`, `δ·sign(t)` otherwise is exactly `clamp(t, -δ, δ)`.
The accumulation order `−x⁺ −y⁺ −z⁺ +x⁻ +y⁻ +z⁻` of the legacy kernel is kept.
"""
function huber_gradient(x::AbstractArray{<:Any, 3}, δ::Real)
    # Forward face differences, Huber-clipped: size (nx-1, ny, nz) etc.
    fx = clamp.(x[2:end, :, :] .- x[1:(end - 1), :, :], -δ, δ)
    fy = clamp.(x[:, 2:end, :] .- x[:, 1:(end - 1), :], -δ, δ)
    fz = clamp.(x[:, :, 2:end] .- x[:, :, 1:(end - 1)], -δ, δ)
    zx = zero.(x[1:1, :, :])
    zy = zero.(x[:, 1:1, :])
    zz = zero.(x[:, :, 1:1])
    # Voxel i sees +ψ'(x[i+1]-x[i]) with a negative sign (forward neighbour,
    # zero at i == nx) and ψ'(x[i]-x[i-1]) with a positive sign (backward
    # neighbour, zero at i == 1).
    xf = cat(fx, zx; dims = 1); xb = cat(zx, fx; dims = 1)
    yf = cat(fy, zy; dims = 2); yb = cat(zy, fy; dims = 2)
    zf = cat(fz, zz; dims = 3); zb = cat(zz, fz; dims = 3)
    return zero(δ) .- xf .- yf .- zf .+ xb .+ yb .+ zb
end

"""
    projection_weights(sino, plan) -> statistical weights, sinogram shaped

`w = air_ref(col,row) · exp(−clamp(y, 0, 10)) + ε` (air reference optional),
exactly the legacy `data_weights` kernel before the `W_proj` fold.
"""
function projection_weights(sino::AbstractArray{<:Any, 3}, plan::HIRPlan{T}) where {T}
    yc = clamp.(sino, zero(T), plan.y_clip)
    return _stat_weights(yc, plan.air_reference, plan.stat_eps)
end

_stat_weights(yc, ::Nothing, ε) = exp.(.-yc) .+ ε
_stat_weights(yc, aref::AbstractArray, ε) = aref .* exp.(.-yc) .+ ε

"""
    ray_normalization(plan, A, like_vol) -> W_proj = 1 / (A · support)

Pure port of `compute_projection_weights(…; circular_support = true)`: forward
project the circular support indicator (built from `like_vol`, a work-shaped
prototype array) over all views and invert with the `1e-8` guard.  The guard
is applied through `max` so the unselected branch of `ifelse` stays finite
under AD.
"""
function ray_normalization(plan::HIRPlan{T}, A, like_vol::AbstractArray{<:Any, 3}) where {T}
    support = _where(plan.fov_mask, one.(like_vol), zero(T))
    ray_sums = A(support, plan.all_views)
    ε = plan.weight_eps
    return _where(ray_sums .> ε, one(T) ./ max.(ray_sums, ε), zero(T))
end

"""
    image_weights(plan, At, like_sino) -> V_inv = 1 / (Aᵀ · 1)

Pure port of `compute_image_weights(…; active_z = 1:nz_work, circular_support =
true)`: backproject a sinogram of ones (built from `like_sino`) over all views
and invert with the `1e-8` guard.  The circular-support zeroing lives inside
the operator `At`, as in the legacy.
"""
function image_weights(plan::HIRPlan{T}, At, like_sino::AbstractArray{<:Any, 3}) where {T}
    voxel_sums = At(one.(like_sino), plan.all_views)
    ε = plan.weight_eps
    return _where(voxel_sums .> ε, one(T) ./ max.(voxel_sums, ε), zero(T))
end

"""
    hir_seed(init_vol, plan) -> work volume

Seed the halo'd iterate by clamped z-continuation of the init volume
(`_hir_seed_work!`).  Static gather along z.
"""
hir_seed(init_vol::AbstractArray{<:Any, 3}, plan::HIRPlan) = init_vol[:, :, plan.seed_z]

"""
    hir_finish(work, plan) -> output volume

Extract the output slab (`_hir_extract_output!`) and apply the clinical FOV
sentinel (`apply_fov_mask!`, −0.04 cm⁻¹ outside the circle).
"""
function hir_finish(work::AbstractArray{<:Any, 3}, plan::HIRPlan)
    out = work[:, :, plan.output_z]
    return _where(plan.fov_mask, out, plan.sentinel)
end

# One ordered-subset update.  `dw` is the folded data weight (W_proj ⊙ stat_w),
# `rg` the Huber gradient at the current iterate.  Operation order mirrors the
# legacy kernels term by term:
#   residual  = dw ⊙ (y_s − A x)
#   corr      = Aᵀ residual
#   x        += relaxation · V_inv · n_subsets · corr − λ · rvs · V_inv · rg
# with the circular support re-imposed by `update_mask`.
function _hir_subset_step(vol, sino, dw, vinv, rg, plan::HIRPlan{T}, A, At, s::Int) where {T}
    idx = plan.subset_views[s]
    ax = A(vol, idx)
    y_s = sino[:, :, idx]
    dw_s = dw[:, :, idx] .* plan.subset_valid[s]
    residual = dw_s .* (y_s .- ax)
    corr = At(residual, idx)
    updated = vol .+ (plan.relaxation .* vinv .* plan.subset_scale .* corr .-
                      plan.λ .* plan.reg_views_scale .* vinv .* rg)
    return _where(plan.update_mask, updated, zero(T))
end

"""
    hir_reconstruct(sino, init_vol, plan, A, At) -> volume
    hir_reconstruct(sino, init_vol, plan, A, At, W_proj, V_inv) -> volume

Functional OS-PWLS HIR.  Returns exactly what `reconstruct!(::HIRReconWorkspace,
…; init_volume = init_vol)` returns: the output-slab volume with the −0.04
FOV sentinel outside the reconstruction circle.

The 5-argument form computes `W_proj`/`V_inv` from the operators on every
call (one extra forward + back projection over all views); the 7-argument
form lets the caller reuse them across calls (`ray_normalization`,
`image_weights`).

`A(vol, view_indices)` and `At(sino_sub, view_indices)` must be pure functions
of their arguments, built on `plan.work_geom`.
"""
function hir_reconstruct(sino::AbstractArray{<:Any, 3}, init_vol::AbstractArray{<:Any, 3},
                         plan::HIRPlan{T}, A, At) where {T}
    plan.nepochs == 0 && return hir_finish(hir_seed(init_vol, plan), plan)
    W_proj = ray_normalization(plan, A, hir_seed(init_vol, plan))
    V_inv = image_weights(plan, At, sino)
    return hir_reconstruct(sino, init_vol, plan, A, At, W_proj, V_inv)
end

function hir_reconstruct(sino::AbstractArray{<:Any, 3}, init_vol::AbstractArray{<:Any, 3},
                         plan::HIRPlan{T}, A, At,
                         W_proj::AbstractArray{<:Any, 3}, V_inv::AbstractArray{<:Any, 3}) where {T}
    size(sino) == plan.sino_shape || throw(DimensionMismatch(
        "sinogram has size $(size(sino)); plan expects $(plan.sino_shape)"))
    size(init_vol) == plan.vol_shape || throw(DimensionMismatch(
        "init_vol has size $(size(init_vol)); plan expects $(plan.vol_shape)"))

    vol = hir_seed(init_vol, plan)
    # strength = 0 ⇒ the (FDK) init IS the result; no weights, no loop.
    plan.nepochs == 0 && return hir_finish(vol, plan)

    # Circular support of the iterative system matrix, kept active throughout.
    vol = _where(plan.fov_mask, vol, zero(T))

    # Fold the projection normalisation into the statistical weights once.
    dw = W_proj .* projection_weights(sino, plan)

    δ = plan.δ
    if plan.reg_per_subset
        # Exact DD: gradient of the CURRENT ordered-subset iterate.
        for _ in 1:plan.nepochs, s in 1:plan.n_subsets
            rg = huber_gradient(vol, δ)
            vol = _hir_subset_step(vol, sino, dw, V_inv, rg, plan, A, At, s)
        end
    else
        # Historical Siddon cadence: one gradient per epoch.
        for _ in 1:plan.nepochs
            rg = huber_gradient(vol, δ)
            for s in 1:plan.n_subsets
                vol = _hir_subset_step(vol, sino, dw, V_inv, rg, plan, A, At, s)
            end
        end
    end

    return hir_finish(vol, plan)
end
