# =============================================================================
# :dd_fast — single-pass per-material path-length Distance-Driven projection
# =============================================================================
#
# SAME distance-driven (DD3) model as `dd.jl` — identical `_dd_cell_setup`
# footprint, `_dd_bounds` voxel ranges, and `_dd_overlap` weights — with the
# fused-kernel ACCUMULATION reassociated by linearity:
#
#     L_e = Σ_v w_v · μ[m_v, e]          (legacy: NTuple{N_E} per-energy sums,
#                                         N_E FMAs + table reads per voxel,
#                                         N_E live registers → spills for
#                                         clinical spectra, hence the hosts'
#                                         K=16 energy tiling that re-walks the
#                                         volume n_tiles times)
#         = Σ_m μ[m, e] · P_m,   P_m = Σ_{v : m_v = m} w_v
#
# so ONE volume walk accumulates per-MATERIAL path lengths P (energy-independent
# registers; above 16 materials, the line integrals at a few pivot energies instead) and every energy converts once per detector
# cell.  Results agree with the per-energy `dd.jl` kernels to floating-point ordering.
#
# Measured (M4 Metal, 234-bin polychromatic forward, 512²×64 vol,
# 736×16×720 sino): 113.8 s (dd.jl tiled hosts) → 2.4 s (:dd_fast single-pass),
# agreement mean_rel ≈ 5e-7 — and 4.4x faster than Siddon's tiled 234-bin
# path.  UHR slab (1024²×32): 109.3 s → 5.3 s.
#
# Selected via `projector = :dd_fast` (SimOptions / create_hir_recon_workspace
# / the `_project_fused_*` shims).  Mono projection has no energy loop, so
# `:dd_fast` mono IS `dd_forward_project!` — see `select_projector.jl`.
# Any number of materials: up to 16 one lane each, above that the few lanes of an attenuation
# basis (a physical table's rank is at most its number of elements). Only a table no basis of at
# most `_PLEN_MAX_MATERIALS` energies reproduces warns and falls back to the legacy tiled kernels.
# =============================================================================

export dd_fast_fused_poly_project!, dd_fast_fused_spectral_project!

# =============================================================================
# Per-MATERIAL path-length accumulation (fused kernels)
# =============================================================================
#
# The fused kernels need, per detector cell, L_e = Σ_v w_v·μ[m_v, e] for every
# energy e.  The legacy formulation accumulates an NTuple{N_E} of per-energy
# sums: N_E FMAs + N_E μ-table reads per overlapped voxel and N_E live
# registers per thread — which spills for clinical spectra (N_E ≈ 234) and
# forced the host-side K=16 energy tiling that re-walks the volume n_tiles
# times.  By linearity of the line integral,
#
#     L_e = Σ_m μ[m, e] · P_m,    P_m = Σ_{v : m_v = m} w_v,
#
# so we accumulate per-material path lengths P (n_mat registers,
# energy-independent) during ONE volume walk and convert to the energies once
# per cell.  Footprint, overlap weights, and anti-aliasing are IDENTICAL —
# this is a reassociation of a linear sum (floating-point ordering only).
# Up to 16 materials one lane each; above that the lanes of an attenuation basis
# (`_plen_lanes`), whatever the number of materials.

# Branchless scatter of `a · b` into `plens[mat]`: the product once, then a select per lane (no
# divergence). Keep the multiply out of the lanes: v0.16.0–0.16.3 read the lane by a select chain,
# `fma`-ed and wrote it back (and an `fma` or `muladd` in every lane is the same), which made the
# spectral GPU kernel 30× slower (24 s against 0.8 s per 120 Alpha views). A GPU fuses `p + w` into
# the same `fma` either way, so its sums are unchanged; the CPU rounds the product separately, so
# CPU and GPU agree to rounding, not to the bit.
@generated function _plen_accum(plens::NTuple{M, T}, mat::Int32, a::T, b::T) where {M, T}
    exprs = [:(ifelse(mat == Int32($m), plens[$m] + w, plens[$m])) for m in 1:M]
    return quote
        w = a * b
        tuple($(exprs...))
    end
end

# L_e = Σ_m plens[m] · μ_tbl[m, e]  (chained binary + — no varargs on GPU).
@generated function _plen_line_integral(plens::NTuple{M, T}, μ_tbl, e::Int32) where {M, T}
    ex = :(plens[1] * @inbounds(μ_tbl[1, e]))
    for m in 2:M
        ex = :($ex + plens[$m] * @inbounds(μ_tbl[$m, e]))
    end
    return ex
end

# The walk's lanes: `nothing` keeps one lane per material (`_plen_accum`); a matrix `Ct` (R × M) keeps
# R lanes of an attenuation basis, voxel material `m` adding `w · Ct[:, m]` (see
# `_attenuation_basis`): R products per voxel however many materials there are.
@inline _lane_add(plens::NTuple, ::Nothing, mat::Int32, a, b) = _plen_accum(plens, mat, a, b)
@inline function _lane_add(plens::NTuple{R, T}, Ct::AbstractMatrix, mat::Int32, a::T, b::T) where {R, T}
    w = a * b
    return ntuple(k -> plens[k] + w * @inbounds(Ct[k, mat]), Val(R))
end

# Max n_materials for the path-length kernels (NTuple stays in registers).
const _PLEN_MAX_MATERIALS = 64

# -----------------------------------------------------------------------------
# An attenuation basis: any number of materials in a few lanes
# -----------------------------------------------------------------------------
# A material's attenuation is a mixture of its elements', so the table μ[m, e] (materials ×
# energies) has a low rank: at most the number of distinct elements, whatever the number of
# materials (64 mixtures of 14 elements: rank 9).
# A column-pivoted QR picks the R energies J that span it, and μ = μ[:, J] · G with G = μ[:, J] \ μ.
# The walk then accumulates P_k = Σ_v w_v μ[m_v, J_k], the line integral at pivot energy J_k (a sum
# of positive terms, as the per-material path lengths are), and every energy's line integral is
# L_e = Σ_k P_k G[k, e]: the same `_plen_line_integral` with G as its table. R is the smallest rank
# that reproduces every material at every weighted energy to 1e-6 of itself or a few Float32 ulps
# of its own largest value, so the projection agrees with the per-material walk to Float32
# rounding. Cached per table (the workspace's μ table is fixed).

# Up to this many materials the walk keeps one lane per material; above it, the basis.
const _DIRECT_MAX_MATERIALS = Ref(16)

const _BASIS_CACHE = Dict{Tuple{UInt64, DataType, DataType}, Any}()   # by the table's content and active energies
const _BASIS_LOCK = ReentrantLock()

# The Float64 μ table each workspace computed before rounding it for the device, by the hash of the
# device table's content: fitted to the exact table, a basis has the table's true rank (64 mixtures of
# 14 elements: 9); fitted to its Float32 rounding, the rounding noise raises it (29).
const _EXACT_ATTENUATION = Dict{UInt64, Matrix{Float64}}()

"Register the exact Float64 table (`n_regions × n_energies`) behind a device μ table (padded with zeros)."
function _register_exact_attenuation!(μ_table, μ_exact::AbstractMatrix{Float64})
    exact = zeros(size(μ_table))
    exact[:, 1:size(μ_exact, 2)] .= μ_exact
    lock(() -> (_EXACT_ATTENUATION[hash(Array(μ_table))] = exact), _BASIS_LOCK)
    return μ_table
end

"""
    _attenuation_basis(μ_table, weight) -> (; Ct, G, J) or nothing

The pivot energies `J`, the lanes' coefficients `Ct = μ[:, J]'` (R × M) and the map to every energy
`G` (R × E), on the device of `μ_table`; `nothing` when no rank ≤ `_PLEN_MAX_MATERIALS` reproduces
the table to tolerance. Only the energies the projection weights (`weight[e] > 0`) are fitted, and
pivots are chosen among those with at least 1 % of the peak weight, by the shape of their
attenuation (columns normalised), not its size: a pivot at
a few keV, where nothing is transmitted, would carry line integrals of thousands and cancel away the
Float32 precision of the energies that matter. `G` is zero at inactive energies.
"""
function _attenuation_basis(μ_table::AbstractMatrix{T}, weight::AbstractVector{<:Real}) where {T}
    host = Array(μ_table)
    key = (hash((host, weight)), typeof(μ_table), T)
    cached = lock(() -> get(_BASIS_CACHE, key, missing), _BASIS_LOCK)
    cached === missing || return cached
    exact = lock(() -> get(_EXACT_ATTENUATION, hash(host), nothing), _BASIS_LOCK)
    μ = exact === nothing ? Float64.(host) : exact
    M, E = size(μ)
    on = findall(>(0), weight)
    μa = μ[:, on]
    scale = isempty(μa) ? 0.0 : maximum(abs, μa)
    basis = nothing
    if scale > 0
        # pivots only where the projection's weight is at least 1 % of its peak: an energy that
        # carries little weight carries large line integrals, and the map back from it amplifies
        # their rounding; the others are still reproduced to tolerance
        wa = weight[on]
        cand = findall(>=(1.0e-2 * maximum(wa)), wa)
        # each material judged on its own scale: the table is Float32 (rounded to ~1e-7 of each
        # entry), and fitted unscaled the large materials' rounding would land on air's tiny row
        rows = [max(maximum(abs, r), floatmin(Float64)) for r in eachrow(μa)]
        μn = μa ./ rows
        norms = [max(sqrt(sum(abs2, c)), eps()) for c in eachcol(μn)]
        p = cand[qr(μn[:, cand] ./ norms[cand]', ColumnNorm()).p]
        # 1e-6 of the entry, or the precision of the row: the exact table's (1e-12) when registered,
        # else ~4 ulps of its device rounding
        tol = 1.0e-6 .* abs.(μn) .+ (exact === nothing ? 4 * eps(T) : 1.0e-12)
        for R in 1:min(M, length(p), _PLEN_MAX_MATERIALS)
            J = p[1:R]
            Ga = μn[:, J] \ μn
            if all(abs.(μn[:, J] * Ga .- μn) .<= tol)
                G = zeros(R, E)
                G[:, on] = Ga
                device(x) = copyto!(similar(μ_table, T, size(x)), T.(x))
                basis = (Ct = device(permutedims(μa[:, J])), G = device(G), J = on[J])
                break
            end
        end
    end
    lock(() -> (_BASIS_CACHE[key] = basis), _BASIS_LOCK)
    return basis
end

# The walk's lanes for a μ table and the projection's weight of each energy: `(lanes, table,
# n)` with `lanes` `nothing` (one per material) or
# the basis coefficients, `table` what `_plen_line_integral` reads, `n` the lane count; `nothing`
# when neither fits in registers (the legacy tiled kernels).
function _plen_lanes(μ_table::AbstractMatrix, weight::AbstractVector{<:Real})
    M = size(μ_table, 1)
    M <= _DIRECT_MAX_MATERIALS[] && return (lanes = nothing, table = μ_table, n = M)
    b = _attenuation_basis(μ_table, weight)
    b === nothing || return (lanes = b.Ct, table = b.G, n = length(b.J))
    # no basis: one lane per material, each added with a product (identity coefficients), not
    # selected (64 materials: 3.7 s against 210 s per 120 views)
    M <= _PLEN_MAX_MATERIALS || return nothing
    eye = copyto!(similar(μ_table, M, M), Matrix{eltype(μ_table)}(I, M, M))
    return (lanes = eye, table = μ_table, n = M)
end

# The projection's weight of each energy: the fused polychromatic sums' (|wη| + |w2|), the
# spectral tile's (each energy's |W| summed over the bins, zero outside the tile).
function _poly_weight(wη_gpu, w2_gpu)
    weight = abs.(Float64.(Array(wη_gpu)))
    w2_gpu === nothing || (weight .+= abs.(Float64.(Array(w2_gpu))))
    return weight
end
function _spectral_weight(μ_table_gpu, W_gpu, tile_start::Integer, K::Integer)
    weight = zeros(size(μ_table_gpu, 2))
    tile = Int(tile_start):min(Int(tile_start) + K - 1, size(W_gpu, 1))
    weight[tile] .= vec(sum(abs.(Float64.(Array(W_gpu)[tile, :])); dims = 2))
    return weight
end

"""
    _dd_fast_fits(μ_table, weight) -> Bool

Whether the `:dd_fast` single-pass kernels project this table: always up to `_PLEN_MAX_MATERIALS`
materials, and for any number of materials whose attenuation a basis of at most
`_PLEN_MAX_MATERIALS` energies reproduces (every physical table: its rank is at most the number of
elements). Callers ask before choosing the single pass over their own tiled fallback.
"""
_dd_fast_fits(μ_table::AbstractMatrix, weight::AbstractVector{<:Real}) = _plen_lanes(μ_table, weight) !== nothing

function _warn_dd_fast_fallback(n_materials::Integer)
    @warn """
    DD_FAST SINGLE-PASS DISABLED: the material table has $n_materials entries, and no attenuation basis of at most $(_PLEN_MAX_MATERIALS) energies reproduces it.
    Falling back to the legacy tiled distance-driven projector, which can be tens of times slower.
    Call compact_materials(phantom) before GPU transfer to remove inactive materials, or merge materials when physically appropriate.
    The requested :dd_fast path is NOT effective for this projection.
    """ maxlog = 1
    return nothing
end

# =============================================================================
# Kernel launches (internal) — the walk, once, then what each caller does with it
# =============================================================================

# The distance-driven walk itself: one detector element's footprint stepped through the volume,
# returning the path length (cm) it accumulates in each of the `M` materials.  It is ~95 % of the
# cost of a polychromatic projection and the one thing the three `:dd_fast` kernels below must do
# IDENTICALLY — the fused polychromatic kernel, the fused spectral kernel and the cached-paths
# kernel differ only in what they do with the tuple that comes back.  Writing it once is not
# tidiness: the helical backprojector's arc-mapping bug was two copies of a mapping drifting
# apart, and three copies of forty lines of index arithmetic is the same bet.
#
# `@inline` because every caller is a GPU kernel body: this has to disappear into it, `plens`
# staying in registers, or the single-pass design has no point.  Its scalars come from
# `_dd_walk_setup` below, which is the other half of the same rule.
@inline function _dd_plen_walk(
        ::Val{M}, lanes, mask, col::Int32, row::Int32, angle::Int32,
        sp, dc, du, dv,
        vmx::T, vmy::T, vmz::T, vsx::T, vsy::T, vsz::T,
        nx::Int32, ny::Int32, nz::Int32,
        mag::T, ps::T, prs::T, cc::T, rc::T, arc_det::Bool, dγ::T,
    ) where {M, T <: AbstractFloat}

    sx = sp[1, angle]; sy = sp[2, angle]; sz = sp[3, angle]
    dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
    ux = du[1, angle]; uy = du[2, angle]; vvz = dv[3, angle]

    (
        valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long,
    ) = _dd_cell_setup(
        col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
        mag, ps, prs, cc, rc, nx, ny, vmx, vmy, vsx, vsy, arc_det, dγ
    )

    plens = ntuple(_ -> zero(T), Val(M))
    valid || return plens

    il = Int32(1)
    while il <= n_long
        lp = fma(T(il) - T(0.5), v_long, vmin_long)
        mag_fac = s_long / (s_long - lp)
        inv_mf = one(T) / mag_fac
        (it_s, it_e) = _dd_bounds(dXlo, dXhi, s_tran, vmin_t, v_t, inv_mf, n_t)
        (ip_s, ip_e) = _dd_bounds(dZlo, dZhi, sz, vmz, vsz, inv_mf, nz)
        it = it_s
        while it <= it_e
            t0 = _dd_edge(s_tran, vmin_t, T(it - Int32(1)), v_t, mag_fac)
            t1 = _dd_edge(s_tran, vmin_t, T(it), v_t, mag_fac)
            ox = _dd_overlap(dXlo, dXhi, min(t0, t1), max(t0, t1))
            if ox > zero(T)
                ixv = vertical ? it : il
                iyv = vertical ? il : it
                ip = ip_s
                while ip <= ip_e
                    z0 = _dd_edge(sz, vmz, T(ip - Int32(1)), vsz, mag_fac)
                    z1 = _dd_edge(sz, vmz, T(ip), vsz, mag_fac)
                    oz = _dd_overlap(dZlo, dZhi, min(z0, z1), max(z0, z1))
                    if oz > zero(T)
                        mat = Int32(mask[ixv, iyv, ip]) + Int32(1)
                        plens = _lane_add(plens, lanes, mat, ox * oz, norm)
                    end
                    ip += Int32(1)
                end
            end
            it += Int32(1)
        end
        il += Int32(1)
    end
    return plens
end


# =============================================================================
# Fused polychromatic — mirror siddon_fused_poly_project!
# =============================================================================

# Path-length kernel launch: one walk accumulating per-material P (Val(M)
# registers), then a runtime loop over ALL n_E energies at the cell level.
function _dd_fused_poly_plen!(
        ::Val{M}, lanes,
        sinogram::AbstractArray{T, 3}, mask, μ_table_gpu, wη_gpu, _bt, has_bowtie::Bool,
        sp, dc, du, dv,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T,
        nx::Int32, ny::Int32, nz::Int32, n_cols::Int32, n_rows::Int32,
        mag::T, ps::T, prs::T, col_center::T, row_center::T, nc_nr::Int32,
        arc_det::Bool, dγ::T,
        w2_gpu, out2, has2::Bool,
    ) where {M, T <: AbstractFloat}

    n_E = Int32(length(wη_gpu))

    let mask = mask, μ_tbl = μ_table_gpu, wη = wη_gpu, arc_det = arc_det, dγ = dγ, lanes = lanes,
            sp = sp, dc = dc, du = du, dv = dv, bt = _bt,
            vmx = vmin_x, vmy = vmin_y, vmz = vmin_z, vsx = vx, vsy = vy, vsz = vz,
            nx = nx, ny = ny, nz = nz, nc = n_cols, nr = n_rows,
            mag = mag, ps = ps, prs = prs, cc = col_center, rc = row_center,
            ncnr = nc_nr, hbt = has_bowtie, nE = n_E, w2 = w2_gpu, o2 = out2, h2 = has2

        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            plens = _dd_plen_walk(
                Val(M), lanes, mask, col, row, angle, sp, dc, du, dv,
                vmx, vmy, vmz, vsx, vsy, vsz, nx, ny, nz,
                mag, ps, prs, cc, rc, arc_det, dγ
            )

            _plen_spectral_sums!(
                sinogram, o2, idx, plens, μ_tbl, wη, w2, h2, bt, hbt,
                Int32(col) + (Int32(row) - Int32(1)) * nc, ncnr, nE,
            )
        end
    end
    return sinogram
end

# The energy loop both path-length kernels end in: S₁ = Σ w B e^{−L} into `sinogram` and,
# when `h2`, S₂ = Σ w₂ B e^{−L} into `o2`, each stored as −log S.
@inline function _plen_spectral_sums!(
        sinogram::AbstractArray{T}, o2, idx, plens, μ_tbl, wη, w2, h2::Bool, bt, hbt::Bool,
        bt_base::Int32, ncnr::Int32, nE::Int32,
    ) where {T}

    I_1 = zero(T)
    I_2 = zero(T)
    e = Int32(1)
    while e <= nE
        ex = exp(-_plen_line_integral(plens, μ_tbl, e))
        if hbt
            ex *= @inbounds bt[bt_base + (e - Int32(1)) * ncnr]
        end
        I_1 += (@inbounds wη[e]) * ex
        if h2
            I_2 += (@inbounds w2[e]) * ex
        end
        e += Int32(1)
    end
    @inbounds sinogram[idx] = -log(max(I_1, T(1.0e-10)))
    if h2
        @inbounds o2[idx] = -log(max(I_2, T(1.0e-10)))
    end
    return nothing
end


# =============================================================================
# Fused spectral (PCCT, tiled) — mirror siddon_fused_spectral_project!
# =============================================================================

# Path-length kernel launch (spectral): one walk accumulating per-material P,
# then per output bin a runtime loop over the K tile energies at cell level.
# Every bin's sum gains `W[e, b] · trans` (a function, so the GPU closure captures no reassigned tuple).
@inline _bin_sums(acc::NTuple{NB, T}, W, e::Int32, trans::T) where {NB, T} =
    ntuple(b -> @inbounds(acc[b] + W[e, b] * trans), Val(NB))

function _dd_fused_spectral_plen!(
        ::Val{M}, ::Val{NB}, lanes, K::Int32, ts::Int32,
        pilot::AbstractArray{T, 3}, outputs_flat, n_bins::Int32,
        mask, μ_table_gpu, W_gpu, _bt, has_src_spectral::Bool,
        sp, dc, du, dv,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T,
        nx::Int32, ny::Int32, nz::Int32, n_cols::Int32, n_rows::Int32, n_elem::Int,
        mag::T, ps::T, prs::T, col_center::T, row_center::T, nc_nr::Int32,
        arc_det::Bool, dγ::T,
    ) where {M, NB, T <: AbstractFloat}

    let mask = mask, μ_tbl = μ_table_gpu, W = W_gpu, ts = ts, kk = K, arc_det = arc_det, dγ = dγ, lanes = lanes,
            sp = sp, dc = dc, du = du, dv = dv, bt = _bt,
            vmx = vmin_x, vmy = vmin_y, vmz = vmin_z, vsx = vx, vsy = vy, vsz = vz,
            nx = nx, ny = ny, nz = nz, nc = n_cols, nr = n_rows,
            ne = n_elem, nb = n_bins, oflat = outputs_flat,
            mag = mag, ps = ps, prs = prs, cc = col_center, rc = row_center,
            ncnr = nc_nr, hbt = has_src_spectral

        AK.foreachindex(pilot) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            plens = _dd_plen_walk(
                Val(M), lanes, mask, col, row, angle, sp, dc, du, dv,
                vmx, vmy, vmz, vsx, vsy, vsz, nx, ny, nz,
                mag, ps, prs, cc, rc, arc_det, dγ
            )

            # each energy's line integral and transmission once, into every bin's sum (each bin
            # still sums its energies in the same order)
            bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
            acc = ntuple(_ -> zero(T), Val(NB))
            i = Int32(0)
            while i < kk
                e = ts + i
                L = _plen_line_integral(plens, μ_tbl, e)
                trans = exp(-L)
                if hbt
                    # match legacy clamp: W · bt · min(exp(-L), 1e30)
                    trans = min(trans, T(1.0e30)) * @inbounds bt[bt_base + (e - Int32(1)) * ncnr]
                end
                acc = _bin_sums(acc, W, e, trans)
                i += Int32(1)
            end
            for b in 1:NB
                @inbounds oflat[idx + (b - 1) * ne] += acc[b]
            end
        end
    end
    return outputs_flat
end


# =============================================================================
# :dd_fast — single-pass per-material path-length DD (same DD model)
# =============================================================================
#
# Public wrappers for the `_dd_fused_*_plen!` kernels above.  Same signatures
# and semantics as `dd_fused_poly_project!` / `dd_fused_spectral_project!`
# (drop-in via the `:dd_fast` projector symbol), same DD3 footprint, bounds,
# and overlap weights — only the accumulation is reassociated (per-material
# path lengths instead of per-energy sums), so results agree with the dd.jl kernels
# to floating-point ordering.  The payoff: no per-energy registers, so the
# FULL spectrum runs in ONE volume walk (the hosts skip the K=16 energy tiling
# that re-walks the volume n_tiles times).  Measured on M4 Metal, 234-bin
# polychromatic forward (512²×64 vol, 736×16×720 sino): 113.8 s (dd.jl tiled)
# → 2.4 s (:dd_fast single-pass), agreement mean_rel ≈ 5e-7.
#
# Any number of materials (`_plen_lanes`); a table no basis reproduces warns and falls back to
# the legacy `dd_*` kernels (the hosts ask `_dd_fast_fits` first and keep their tiled loop).

"""
    dd_fast_fused_poly_project!(sinogram, mask, geom, μ_table_gpu, wη_gpu, Val(N_E); kwargs...)

Single-pass per-material path-length variant of [`dd_fused_poly_project!`](@ref)
(the `:dd_fast` projector).  Identical distance-driven footprint and weights;
accumulation reassociated by linearity so ALL `N_E` energies are produced in
one volume walk with `n_materials` registers.  Falls back to the legacy kernel
when `size(μ_table_gpu, 1) > 64`, with a prominent performance warning.
`ws_w2_gpu` with `ws_out2` adds a second weighted sum in the same walk, as in
[`dd_fast_poly_from_paths!`](@ref).
"""
function dd_fast_fused_poly_project!(
        sinogram::AbstractArray{T, 3},
        mask::AbstractArray{<:Unsigned, 3},
        geom::CTGeometry,
        μ_table_gpu::AbstractArray{T, 2},
        wη_gpu::AbstractArray{T, 1},
        ::Val{N_E};
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        ws_source_positions = nothing,
        ws_detector_centers = nothing,
        ws_detector_u = nothing,
        ws_detector_v = nothing,
        ws_bowtie_spectral = nothing,
        ws_w2_gpu = nothing,
        ws_out2 = nothing,
    ) where {T <: AbstractFloat, N_E}

    (ws_w2_gpu === nothing) == (ws_out2 === nothing) ||
        throw(ArgumentError("ws_w2_gpu and ws_out2 come together"))
    lanes = _plen_lanes(μ_table_gpu, _poly_weight(wη_gpu, ws_w2_gpu))
    if lanes === nothing
        _warn_dd_fast_fallback(size(μ_table_gpu, 1))
        if ws_w2_gpu !== nothing
            # the legacy kernel has one accumulator: the second sum is a second projection
            dd_fused_poly_project!(
                ws_out2, mask, geom, μ_table_gpu, ws_w2_gpu, Val(N_E);
                volume_extent,
                ws_source_positions,
                ws_detector_centers,
                ws_detector_u,
                ws_detector_v,
                ws_bowtie_spectral,
            )
        end
        return dd_fused_poly_project!(
            sinogram, mask, geom, μ_table_gpu, wη_gpu, Val(N_E);
            volume_extent = volume_extent,
            ws_source_positions = ws_source_positions,
            ws_detector_centers = ws_detector_centers,
            ws_detector_u = ws_detector_u,
            ws_detector_v = ws_detector_v,
            ws_bowtie_spectral = ws_bowtie_spectral
        )
    end

    g = _dd_walk_setup(
        T, size(mask), (size(sinogram, 1), size(sinogram, 2)), geom, volume_extent, sinogram,
        ws_source_positions, ws_detector_centers, ws_detector_u, ws_detector_v,
    )
    has_bowtie = ws_bowtie_spectral !== nothing
    _bt = has_bowtie ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    return _dd_fused_poly_plen!(
        Val(lanes.n), lanes.lanes, sinogram, mask, lanes.table, wη_gpu, _bt, has_bowtie,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows,
        g.mag, g.ps, g.prs, g.col_center, g.row_center, g.nc_nr,
        g.arc_det, g.dγ,
        something(ws_w2_gpu, wη_gpu), something(ws_out2, sinogram), ws_w2_gpu !== nothing
    )
end

# Every distance-driven walk over this geometry needs the same scalars and the same four
# geometry arrays. Derived once, here, and stepped once, in `_dd_plen_walk`, so the single-pass
# and two-pass paths cannot drift apart — the helical backprojector's arc-mapping bug was
# exactly that kind of divergence.
function _dd_walk_setup(
        ::Type{T}, mask_size, det_size, geom, volume_extent, like,
        ws_source_positions, ws_detector_centers, ws_detector_u, ws_detector_v,
    ) where {T}
    nx = Int32(mask_size[1]); ny = Int32(mask_size[2]); nz = Int32(mask_size[3])
    n_cols = Int32(det_size[1]); n_rows = Int32(det_size[2])

    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vmin_x = T(-vol_bounds[1] / 2); vmin_y = T(-vol_bounds[2] / 2); vmin_z = T(-vol_bounds[3] / 2)
    vx = T(vol_bounds[1]) / T(nx); vy = T(vol_bounds[2]) / T(ny); vz = T(vol_bounds[3]) / T(nz)
    _dd_check_isotropy(vx, vy)

    return (;
        nx, ny, nz, n_cols, n_rows,
        vmin_x, vmin_y, vmin_z, vx, vy, vz,
        mag = T(geom.SDD / geom.SAD),
        arc_det = is_arc(geom),
        dγ = T(geom.pixel_size / geom.SAD),
        ps = T(geom.pixel_size), prs = T(geom.pixel_row_size),
        col_center = column_center(T, geom),
        row_center = (T(n_rows) + one(T)) / T(2),
        nc_nr = n_cols * n_rows,
        sp = _dd_geom_array(ws_source_positions, like, geom.source_positions, T),
        dc = _dd_geom_array(ws_detector_centers, like, geom.detector_centers, T),
        du = _dd_geom_array(ws_detector_u, like, geom.detector_u, T),
        dv = _dd_geom_array(ws_detector_v, like, geom.detector_v, T),
    )
end

"""
    dd_fast_fused_spectral_project!(pilot, outputs_flat, n_bins, mask, geom,
        μ_table_gpu, W_gpu, Val(K), tile_start; kwargs...)

Single-pass per-material path-length variant of
[`dd_fused_spectral_project!`](@ref) (the `:dd_fast` projector).  Identical
distance-driven footprint and weights; `K` may be the FULL padded energy count
in a single call (no per-energy registers, no per-tile volume re-walk).  Falls
back to the legacy kernel when `size(μ_table_gpu, 1) > 64`, with a prominent
performance warning.
"""
function dd_fast_fused_spectral_project!(
        pilot::AbstractArray{T, 3},
        outputs_flat::AbstractArray{T, 1},
        n_bins::Int32,
        mask::AbstractArray{<:Unsigned, 3},
        geom::CTGeometry,
        μ_table_gpu::AbstractArray{T, 2},
        W_gpu::AbstractArray{T, 2},
        ::Val{K},
        tile_start::Int32;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        ws_source_positions = nothing,
        ws_detector_centers = nothing,
        ws_detector_u = nothing,
        ws_detector_v = nothing,
        ws_bowtie_spectral = nothing,
    ) where {T <: AbstractFloat, K}

    # the tile's energies that any bin weights
    lanes = _plen_lanes(μ_table_gpu, _spectral_weight(μ_table_gpu, W_gpu, tile_start, K))
    if lanes === nothing
        _warn_dd_fast_fallback(size(μ_table_gpu, 1))
        return dd_fused_spectral_project!(
            pilot, outputs_flat, n_bins, mask, geom, μ_table_gpu, W_gpu,
            Val(K), tile_start;
            volume_extent = volume_extent,
            ws_source_positions = ws_source_positions,
            ws_detector_centers = ws_detector_centers,
            ws_detector_u = ws_detector_u,
            ws_detector_v = ws_detector_v,
            ws_bowtie_spectral = ws_bowtie_spectral
        )
    end

    g = _dd_walk_setup(
        T, size(mask), (size(pilot, 1), size(pilot, 2)), geom, volume_extent, mask,
        ws_source_positions, ws_detector_centers, ws_detector_u, ws_detector_v,
    )
    n_elem = length(pilot)   # an Int: bin b's offset (b − 1)·n_elem passes typemax(Int32) at native Alpha sizes
    has_src_spectral = ws_bowtie_spectral !== nothing
    _bt = has_src_spectral ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    return _dd_fused_spectral_plen!(
        Val(lanes.n), Val(Int(n_bins)), lanes.lanes, Int32(K), tile_start,
        pilot, outputs_flat, n_bins,
        mask, lanes.table, W_gpu, _bt, has_src_spectral,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows, n_elem,
        g.mag, g.ps, g.prs, g.col_center, g.row_center, g.nc_nr,
        g.arc_det, g.dγ
    )
end

# =============================================================================
# Two-pass polychromatic projection — the volume walk, cached
# =============================================================================
#
# `_dd_fused_poly_plen!` above walks the volume once and converts its per-material path lengths
# to every energy in the same kernel. The walk is ~95 % of that kernel's time and depends only
# on the geometry and the material map: NOT on the tube voltage, the filtration, the bowtie or
# the detector. A study that scans one phantom at several kVp therefore repeats the expensive
# part for nothing.
#
# Splitting it in two — walk once into `paths`, then convert per spectrum — costs
# n_materials × n_cols × n_rows × n_views floats of memory and gives every later spectrum the
# conversion alone. Measured on a 1024² × 60 Gammex phantom, 834 × 34 × 1000 sinogram, 16
# materials, four tube voltages: 2.64 s → 0.78 s of forward projection, bit-identical at every
# kVp (the accumulation order inside a cell is unchanged; only its destination is).

# Pass 1: the walk. Writes per-material path lengths (cm) for every detector element.
function _dd_material_paths!(
        ::Val{M}, lanes, offset::Int32,
        paths::AbstractArray{T, 4}, mask,
        sp, dc, du, dv,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T,
        nx::Int32, ny::Int32, nz::Int32, n_cols::Int32, n_rows::Int32,
        mag::T, ps::T, prs::T, col_center::T, row_center::T,
        arc_det::Bool, dγ::T,
    ) where {M, T <: AbstractFloat}

    # The iteration domain is one detector element per thread, so it is the material-1 slice of
    # `paths` — a device array, unlike a plain range, and its linear index is exactly the
    # (col, row, view) ordering the kernels below assume. Only `idx` is taken from it; every
    # write addresses `paths` by its four indices.
    domain = @view paths[1, :, :, :]

    let mask = mask, paths = paths, domain = domain, arc_det = arc_det, dγ = dγ, lanes = lanes, offset = offset,
            sp = sp, dc = dc, du = du, dv = dv,
            vmx = vmin_x, vmy = vmin_y, vmz = vmin_z, vsx = vx, vsy = vy, vsz = vz,
            nx = nx, ny = ny, nz = nz, nc = n_cols, nr = n_rows,
            mag = mag, ps = ps, prs = prs, cc = col_center, rc = row_center

        AK.foreachindex(domain) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            plens = _dd_plen_walk(
                Val(M), lanes, mask, col, row, angle, sp, dc, du, dv,
                vmx, vmy, vmz, vsx, vsy, vsz, nx, ny, nz,
                mag, ps, prs, cc, rc, arc_det, dγ
            )

            m = Int32(1)
            while m <= Int32(M)
                @inbounds paths[offset + m, col, row, angle] = plens[m]
                m += Int32(1)
            end
        end
    end
    return paths
end

# Pass 2: the spectral conversion. Reads the cached path lengths; never touches the volume.
# A cached ray's lanes: its per-material paths (`nothing`), or their sums on the basis `Ct` (R × M),
# a runtime loop over the materials.
@inline _path_lanes(::Val{M}, ::Nothing, paths, col, row, angle) where {M} =
    ntuple(m -> @inbounds(paths[m, col, row, angle]), Val(M))
@inline function _path_lanes(::Val{R}, Ct::AbstractMatrix, paths, col, row, angle) where {R}
    T = eltype(paths)
    acc = ntuple(_ -> zero(T), Val(R))
    m = Int32(1)
    while m <= Int32(size(paths, 1))
        acc = _lane_add(acc, Ct, m, @inbounds(paths[m, col, row, angle]), one(T))
        m += Int32(1)
    end
    return acc
end

function _dd_poly_from_paths!(
        ::Val{M}, lanes,
        sinogram::AbstractArray{T, 3}, paths::AbstractArray{T, 4},
        μ_table_gpu, wη_gpu, _bt, has_bowtie::Bool,
        n_cols::Int32, n_rows::Int32, nc_nr::Int32,
        w2_gpu, out2, has2::Bool,
    ) where {M, T <: AbstractFloat}

    n_E = Int32(length(wη_gpu))

    let paths = paths, μ_tbl = μ_table_gpu, wη = wη_gpu, bt = _bt, lanes = lanes,
            nc = n_cols, nr = n_rows, ncnr = nc_nr, hbt = has_bowtie, nE = n_E,
            w2 = w2_gpu, o2 = out2, h2 = has2

        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            plens = _path_lanes(Val(M), lanes, paths, col, row, angle)

            _plen_spectral_sums!(
                sinogram, o2, idx, plens, μ_tbl, wη, w2, h2, bt, hbt,
                Int32(col) + (Int32(row) - Int32(1)) * nc, ncnr, nE,
            )
        end
    end
    return sinogram
end

"""
    dd_fast_material_paths!(paths, mask, geom; kwargs...) -> paths

Pass 1 of the two-pass polychromatic projection: walk the volume once and store the per-material
path length of every detector element, in cm.

`paths` is `(n_materials, n_cols, n_rows, n_views)`. The walk depends only on the geometry and
the material map, so the same `paths` serves every spectrum measured through that geometry —
every tube voltage, filtration, bowtie and detector. Feed it to
[`dd_fast_poly_from_paths!`](@ref).

!!! warning
    `paths` belongs to the phantom and geometry it was computed for. Nothing downstream can tell
    that it was not, so reusing it across a different phantom silently reconstructs the old one.
"""
function dd_fast_material_paths!(
        paths::AbstractArray{T, 4},
        mask::AbstractArray{<:Unsigned, 3},
        geom::CTGeometry;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        ws_source_positions = nothing,
        ws_detector_centers = nothing,
        ws_detector_u = nothing,
        ws_detector_v = nothing,
    ) where {T <: AbstractFloat}

    n_materials = size(paths, 1)
    # The detector shape has to be the geometry's: `col_center`/`row_center` come from `paths`,
    # so a smaller cache would silently trace rays through the middle of the detector.
    (size(paths, 2), size(paths, 3), size(paths, 4)) ==
        (geom.n_cols, geom.n_rows, geom.n_angles) || throw(
        DimensionMismatch(
            "paths is $(size(paths)[2:4]) (col, row, view); the geometry is " *
                "$((geom.n_cols, geom.n_rows, geom.n_angles))"
        )
    )

    g = _dd_walk_setup(
        T, size(mask), (size(paths, 2), size(paths, 3)), geom, volume_extent, paths,
        ws_source_positions, ws_detector_centers, ws_detector_u, ws_detector_v,
    )
    walk(R, lanes, offset) = _dd_material_paths!(
        Val(R), lanes, Int32(offset), paths, mask,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows,
        g.mag, g.ps, g.prs, g.col_center, g.row_center,
        g.arc_det, g.dγ
    )
    if n_materials <= _DIRECT_MAX_MATERIALS[]
        walk(n_materials, nothing, 0)
    else
        # more materials than select lanes stay fast for: identity lanes (each voxel adds its length
        # to its own material's lane with a product, no select), at most `_PLEN_MAX_MATERIALS` per
        # walk; any number of materials in as many walks
        for m0 in 0:_PLEN_MAX_MATERIALS:(n_materials - 1)
            R = min(_PLEN_MAX_MATERIALS, n_materials - m0)
            eye = zeros(T, R, n_materials)
            for k in 1:R
                eye[k, m0 + k] = one(T)
            end
            walk(R, copyto!(similar(paths, T, R, n_materials), eye), m0)
        end
    end
    return paths
end

"""
    dd_fast_lane_sums!(sums, mask, geom, Ct; volume_extent = nothing) -> sums

Walk every detector element's footprint through `mask` as [`dd_fast_material_paths!`](@ref) does,
but accumulate `R = size(Ct, 1)` weighted sums `sums[k, col, row, view] = Σ_m Ct[k, m] · L_m` of
the per-material path lengths `L_m` (cm) instead of the lengths themselves: R products per voxel
whatever the number of materials. `R ≤ $(_PLEN_MAX_MATERIALS)`.
"""
function dd_fast_lane_sums!(
        sums::AbstractArray{T, 4},
        mask::AbstractArray{<:Unsigned, 3},
        geom::CTGeometry,
        Ct::AbstractMatrix;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ) where {T <: AbstractFloat}
    R = size(Ct, 1)
    R == size(sums, 1) <= _PLEN_MAX_MATERIALS ||
        throw(DimensionMismatch("one sum per lane, at most $(_PLEN_MAX_MATERIALS)"))
    (size(sums, 2), size(sums, 3), size(sums, 4)) == (geom.n_cols, geom.n_rows, geom.n_angles) ||
        throw(DimensionMismatch("sums is $(size(sums)[2:4]); the geometry is $((geom.n_cols, geom.n_rows, geom.n_angles))"))
    g = _dd_walk_setup(
        T, size(mask), (size(sums, 2), size(sums, 3)), geom, volume_extent, sums,
        nothing, nothing, nothing, nothing,
    )
    return _dd_material_paths!(
        Val(R), Ct, Int32(0), sums, mask,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows,
        g.mag, g.ps, g.prs, g.col_center, g.row_center,
        g.arc_det, g.dγ
    )
end

"""
    dd_fast_poly_from_paths!(sinogram, paths, μ_table_gpu, wη_gpu; ws_bowtie_spectral=nothing,
                             ws_w2_gpu=nothing, ws_out2=nothing)

Pass 2 of the two-pass polychromatic projection: convert cached per-material path lengths
([`dd_fast_material_paths!`](@ref)) into one polychromatic log line-integral sinogram.
`ws_w2_gpu` with `ws_out2` adds a second weighted sum over the same rays, `−log Σ w₂ B e^{−L}`,
written to `ws_out2` — the energy-integrating detector's second moment.

Bit-identical to the single-pass [`dd_fast_fused_poly_project!`](@ref) on the same inputs: the
accumulation inside a cell and the energy loop are the same expressions in the same order.
"""
function dd_fast_poly_from_paths!(
        sinogram::AbstractArray{T, 3},
        paths::AbstractArray{T, 4},
        μ_table_gpu::AbstractArray{T, 2},
        wη_gpu::AbstractArray{T, 1};
        ws_bowtie_spectral = nothing,
        ws_w2_gpu = nothing,
        ws_out2 = nothing,
    ) where {T <: AbstractFloat}

    (ws_w2_gpu === nothing) == (ws_out2 === nothing) ||
        throw(ArgumentError("ws_w2_gpu and ws_out2 come together"))
    size(paths, 1) == size(μ_table_gpu, 1) || throw(
        DimensionMismatch(
            "paths has $(size(paths, 1)) materials, the attenuation table has $(size(μ_table_gpu, 1))"
        )
    )
    (size(paths, 2), size(paths, 3), size(paths, 4)) == size(sinogram) || throw(
        DimensionMismatch("paths $(size(paths)[2:4]) does not match the sinogram $(size(sinogram))")
    )

    n_cols = Int32(size(sinogram, 1))
    nc_nr = n_cols * Int32(size(sinogram, 2))
    has_bowtie = ws_bowtie_spectral !== nothing
    _bt = has_bowtie ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    lanes = _plen_lanes(μ_table_gpu, _poly_weight(wη_gpu, ws_w2_gpu))
    lanes === nothing && throw(
        ArgumentError(
            "dd_fast_poly_from_paths!: no attenuation basis of at most $(_PLEN_MAX_MATERIALS) energies " *
                "reproduces these $(size(μ_table_gpu, 1)) materials"
        )
    )
    return _dd_poly_from_paths!(
        Val(lanes.n), lanes.lanes, sinogram, paths, lanes.table, wη_gpu, _bt, has_bowtie,
        n_cols, Int32(size(sinogram, 2)), nc_nr,
        something(ws_w2_gpu, wη_gpu), something(ws_out2, sinogram), ws_w2_gpu !== nothing
    )
end

export dd_fast_material_paths!, dd_fast_poly_from_paths!
