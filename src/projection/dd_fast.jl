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
# so ONE volume walk accumulates per-MATERIAL path lengths P (n_mat ≤ 64
# registers, energy-independent) and every energy converts once per detector
# cell.  Results agree with legacy `:dd` to floating-point ordering.
#
# Measured (M4 Metal, 234-bin polychromatic forward, 512²×64 vol,
# 736×16×720 sino): 113.8 s (:dd tiled hosts) → 2.4 s (:dd_fast single-pass),
# agreement mean_rel ≈ 5e-7 — and 4.4x faster than Siddon's tiled 234-bin
# path.  UHR slab (1024²×32): 109.3 s → 5.3 s.
#
# Selected via `projector = :dd_fast` (SimOptions / create_hir_recon_workspace
# / the `_project_fused_*` shims).  Mono projection has no energy loop, so
# `:dd_fast` mono IS `dd_forward_project!` — see `select_projector.jl`.
# Requires n_materials ≤ `_PLEN_MAX_MATERIALS` (= 64); larger tables emit a
# prominent warning and fall back to the legacy tiled `dd_*` kernels.
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
# Used when n_materials ≤ 64 (validated register budget); larger tables fall back to
# the legacy per-energy kernels.

# Branchless scatter of `w` into `plens[mat]` (select chain, no divergence).
@generated function _plen_accum(plens::NTuple{M, T}, mat::Int32, w::T) where {M, T}
    exprs = [:(ifelse(mat == Int32($m), plens[$m] + w, plens[$m])) for m in 1:M]
    return :(tuple($(exprs...)))
end

# L_e = Σ_m plens[m] · μ_tbl[m, e]  (chained binary + — no varargs on GPU).
@generated function _plen_line_integral(plens::NTuple{M, T}, μ_tbl, e::Int32) where {M, T}
    ex = :(plens[1] * @inbounds(μ_tbl[1, e]))
    for m in 2:M
        ex = :($ex + plens[$m] * @inbounds(μ_tbl[$m, e]))
    end
    return ex
end

# Max n_materials for the path-length kernels (NTuple stays in registers).
const _PLEN_MAX_MATERIALS = 64

function _warn_dd_fast_fallback(n_materials::Integer)
    @warn """
    DD_FAST SINGLE-PASS DISABLED: material table has $n_materials entries; the optimized limit is $(_PLEN_MAX_MATERIALS).
    Falling back to the legacy tiled distance-driven projector, which can be tens of times slower.
    Call compact_materials(phantom) before GPU transfer to remove inactive materials, merge materials when physically appropriate,
    or deliberately select projector=:dd. The requested :dd_fast path is NOT effective for this projection.
    """ maxlog = 1
    nothing
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
        ::Val{M}, mask, col::Int32, row::Int32, angle::Int32,
        sp, dc, du, dv,
        vmx::T, vmy::T, vmz::T, vsx::T, vsy::T, vsz::T,
        nx::Int32, ny::Int32, nz::Int32,
        mag::T, ps::T, prs::T, cc::T, rc::T, arc_det::Bool, dγ::T,
    ) where {M, T <: AbstractFloat}

    sx = sp[1, angle]; sy = sp[2, angle]; sz = sp[3, angle]
    dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
    ux = du[1, angle]; uy = du[2, angle]; vvz = dv[3, angle]

    (valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long) = _dd_cell_setup(
        col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
        mag, ps, prs, cc, rc, nx, ny, vmx, vmy, vsx, vsy, arc_det, dγ)

    plens = ntuple(_ -> zero(T), Val(M))
    valid || return plens

    il = Int32(1)
    while il <= n_long
        lp = vmin_long + (T(il) - T(0.5)) * v_long
        mag_fac = s_long / (s_long - lp)
        inv_mf = one(T) / mag_fac
        (it_s, it_e) = _dd_bounds(dXlo, dXhi, s_tran, vmin_t, v_t, inv_mf, n_t)
        (ip_s, ip_e) = _dd_bounds(dZlo, dZhi, sz, vmz, vsz, inv_mf, nz)
        it = it_s
        while it <= it_e
            t0 = s_tran + (vmin_t + T(it - Int32(1)) * v_t - s_tran) * mag_fac
            t1 = s_tran + (vmin_t + T(it) * v_t - s_tran) * mag_fac
            ox = _dd_overlap(dXlo, dXhi, min(t0, t1), max(t0, t1))
            if ox > zero(T)
                ixv = vertical ? it : il
                iyv = vertical ? il : it
                ip = ip_s
                while ip <= ip_e
                    z0 = sz + (vmz + T(ip - Int32(1)) * vsz - sz) * mag_fac
                    z1 = sz + (vmz + T(ip) * vsz - sz) * mag_fac
                    oz = _dd_overlap(dZlo, dZhi, min(z0, z1), max(z0, z1))
                    if oz > zero(T)
                        mat = Int32(mask[ixv, iyv, ip]) + Int32(1)
                        plens = _plen_accum(plens, mat, ox * oz * norm)
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
        ::Val{M},
        sinogram::AbstractArray{T, 3}, mask, μ_table_gpu, wη_gpu, _bt, has_bowtie::Bool,
        sp, dc, du, dv,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T,
        nx::Int32, ny::Int32, nz::Int32, n_cols::Int32, n_rows::Int32,
        mag::T, ps::T, prs::T, col_center::T, row_center::T, nc_nr::Int32,
        arc_det::Bool, dγ::T,
    ) where {M, T <: AbstractFloat}

    n_E = Int32(length(wη_gpu))

    let mask = mask, μ_tbl = μ_table_gpu, wη = wη_gpu, arc_det = arc_det, dγ = dγ,
            sp = sp, dc = dc, du = du, dv = dv, bt = _bt,
            vmx = vmin_x, vmy = vmin_y, vmz = vmin_z, vsx = vx, vsy = vy, vsz = vz,
            nx = nx, ny = ny, nz = nz, nc = n_cols, nr = n_rows,
            mag = mag, ps = ps, prs = prs, cc = col_center, rc = row_center,
            ncnr = nc_nr, hbt = has_bowtie, nE = n_E

        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            plens = _dd_plen_walk(
                Val(M), mask, col, row, angle, sp, dc, du, dv,
                vmx, vmy, vmz, vsx, vsy, vsz, nx, ny, nz,
                mag, ps, prs, cc, rc, arc_det, dγ)

            bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
            I_total = zero(T)
            e = Int32(1)
            while e <= nE
                L = _plen_line_integral(plens, μ_tbl, e)
                wt = @inbounds wη[e]
                if hbt
                    wt *= @inbounds bt[bt_base + (e - Int32(1)) * ncnr]
                end
                I_total += wt * exp(-L)
                e += Int32(1)
            end
            sinogram[idx] = -log(max(I_total, T(1.0e-10)))
        end
    end
    return sinogram
end


# =============================================================================
# Fused spectral (PCCT, tiled) — mirror siddon_fused_spectral_project!
# =============================================================================

# Path-length kernel launch (spectral): one walk accumulating per-material P,
# then per output bin a runtime loop over the K tile energies at cell level.
function _dd_fused_spectral_plen!(
        ::Val{M}, K::Int32, ts::Int32,
        pilot::AbstractArray{T, 3}, outputs_flat, n_bins::Int32,
        mask, μ_table_gpu, W_gpu, _bt, has_src_spectral::Bool,
        sp, dc, du, dv,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T,
        nx::Int32, ny::Int32, nz::Int32, n_cols::Int32, n_rows::Int32, n_elem::Int32,
        mag::T, ps::T, prs::T, col_center::T, row_center::T, nc_nr::Int32,
        arc_det::Bool, dγ::T,
    ) where {M, T <: AbstractFloat}

    let mask = mask, μ_tbl = μ_table_gpu, W = W_gpu, ts = ts, kk = K, arc_det = arc_det, dγ = dγ,
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
                Val(M), mask, col, row, angle, sp, dc, du, dv,
                vmx, vmy, vmz, vsx, vsy, vsz, nx, ny, nz,
                mag, ps, prs, cc, rc, arc_det, dγ)

            bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
            b = Int32(1)
            while b <= nb
                acc = zero(T)
                i = Int32(0)
                while i < kk
                    e = ts + i
                    L = _plen_line_integral(plens, μ_tbl, e)
                    trans = exp(-L)
                    if hbt
                        # match legacy clamp: W · bt · min(exp(-L), 1e30)
                        trans = min(trans, T(1.0e30)) * @inbounds bt[bt_base + (e - Int32(1)) * ncnr]
                    end
                    acc += (@inbounds W[e, b]) * trans
                    i += Int32(1)
                end
                oflat[idx + (b - Int32(1)) * ne] += acc
                b += Int32(1)
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
# path lengths instead of per-energy sums), so results agree with legacy `:dd`
# to floating-point ordering.  The payoff: no per-energy registers, so the
# FULL spectrum runs in ONE volume walk (the hosts skip the K=16 energy tiling
# that re-walks the volume n_tiles times).  Measured on M4 Metal, 234-bin
# polychromatic forward (512²×64 vol, 736×16×720 sino): 113.8 s (:dd tiled)
# → 2.4 s (:dd_fast single-pass), agreement mean_rel ≈ 5e-7.
#
# Requires n_materials ≤ `_PLEN_MAX_MATERIALS` (= 64); larger tables warn and
# fall back to the legacy `dd_*` kernels (the hosts keep their tiled loop).

"""
    dd_fast_fused_poly_project!(sinogram, mask, geom, μ_table_gpu, wη_gpu, Val(N_E); kwargs...)

Single-pass per-material path-length variant of [`dd_fused_poly_project!`](@ref)
(the `:dd_fast` projector).  Identical distance-driven footprint and weights;
accumulation reassociated by linearity so ALL `N_E` energies are produced in
one volume walk with `n_materials` registers.  Falls back to the legacy kernel
when `size(μ_table_gpu, 1) > 64`, with a prominent performance warning.
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
    ) where {T <: AbstractFloat, N_E}

    if size(μ_table_gpu, 1) > _PLEN_MAX_MATERIALS
        _warn_dd_fast_fallback(size(μ_table_gpu, 1))
        return dd_fused_poly_project!(
            sinogram, mask, geom, μ_table_gpu, wη_gpu, Val(N_E);
            volume_extent = volume_extent,
            ws_source_positions = ws_source_positions,
            ws_detector_centers = ws_detector_centers,
            ws_detector_u = ws_detector_u,
            ws_detector_v = ws_detector_v,
            ws_bowtie_spectral = ws_bowtie_spectral)
    end

    g = _dd_walk_setup(
        T, size(mask), (size(sinogram, 1), size(sinogram, 2)), geom, volume_extent, sinogram,
        ws_source_positions, ws_detector_centers, ws_detector_u, ws_detector_v,
    )
    has_bowtie = ws_bowtie_spectral !== nothing
    _bt = has_bowtie ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    return _dd_fused_poly_plen!(
        Val(size(μ_table_gpu, 1)), sinogram, mask, μ_table_gpu, wη_gpu, _bt, has_bowtie,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows,
        g.mag, g.ps, g.prs, g.col_center, g.row_center, g.nc_nr,
        g.arc_det, g.dγ)
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

    if size(μ_table_gpu, 1) > _PLEN_MAX_MATERIALS
        _warn_dd_fast_fallback(size(μ_table_gpu, 1))
        return dd_fused_spectral_project!(
            pilot, outputs_flat, n_bins, mask, geom, μ_table_gpu, W_gpu,
            Val(K), tile_start;
            volume_extent = volume_extent,
            ws_source_positions = ws_source_positions,
            ws_detector_centers = ws_detector_centers,
            ws_detector_u = ws_detector_u,
            ws_detector_v = ws_detector_v,
            ws_bowtie_spectral = ws_bowtie_spectral)
    end

    g = _dd_walk_setup(
        T, size(mask), (size(pilot, 1), size(pilot, 2)), geom, volume_extent, mask,
        ws_source_positions, ws_detector_centers, ws_detector_u, ws_detector_v,
    )
    n_elem = g.nc_nr * Int32(size(pilot, 3))
    has_src_spectral = ws_bowtie_spectral !== nothing
    _bt = has_src_spectral ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    return _dd_fused_spectral_plen!(
        Val(size(μ_table_gpu, 1)), Int32(K), tile_start,
        pilot, outputs_flat, n_bins,
        mask, μ_table_gpu, W_gpu, _bt, has_src_spectral,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows, n_elem,
        g.mag, g.ps, g.prs, g.col_center, g.row_center, g.nc_nr,
        g.arc_det, g.dγ)
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
        ::Val{M},
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

    let mask = mask, paths = paths, domain = domain, arc_det = arc_det, dγ = dγ,
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
                Val(M), mask, col, row, angle, sp, dc, du, dv,
                vmx, vmy, vmz, vsx, vsy, vsz, nx, ny, nz,
                mag, ps, prs, cc, rc, arc_det, dγ)

            m = Int32(1)
            while m <= Int32(M)
                @inbounds paths[m, col, row, angle] = plens[m]
                m += Int32(1)
            end
        end
    end
    return paths
end

# Pass 2: the spectral conversion. Reads the cached path lengths; never touches the volume.
function _dd_poly_from_paths!(
        ::Val{M},
        sinogram::AbstractArray{T, 3}, paths::AbstractArray{T, 4},
        μ_table_gpu, wη_gpu, _bt, has_bowtie::Bool,
        n_cols::Int32, n_rows::Int32, nc_nr::Int32,
    ) where {M, T <: AbstractFloat}

    n_E = Int32(length(wη_gpu))

    let paths = paths, μ_tbl = μ_table_gpu, wη = wη_gpu, bt = _bt,
            nc = n_cols, nr = n_rows, ncnr = nc_nr, hbt = has_bowtie, nE = n_E

        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            plens = ntuple(m -> @inbounds(paths[m, col, row, angle]), Val(M))

            bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
            I_total = zero(T)
            e = Int32(1)
            while e <= nE
                L = _plen_line_integral(plens, μ_tbl, e)
                wt = @inbounds wη[e]
                if hbt
                    wt *= @inbounds bt[bt_base + (e - Int32(1)) * ncnr]
                end
                I_total += wt * exp(-L)
                e += Int32(1)
            end
            sinogram[idx] = -log(max(I_total, T(1.0e-10)))
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
    n_materials <= _PLEN_MAX_MATERIALS || throw(ArgumentError(
        "dd_fast_material_paths!: $(n_materials) materials exceeds the path-length limit of " *
            "$(_PLEN_MAX_MATERIALS); call compact_materials(phantom) first"
    ))
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
    return _dd_material_paths!(
        Val(n_materials), paths, mask,
        g.sp, g.dc, g.du, g.dv,
        g.vmin_x, g.vmin_y, g.vmin_z, g.vx, g.vy, g.vz,
        g.nx, g.ny, g.nz, g.n_cols, g.n_rows,
        g.mag, g.ps, g.prs, g.col_center, g.row_center,
        g.arc_det, g.dγ)
end

"""
    dd_fast_poly_from_paths!(sinogram, paths, μ_table_gpu, wη_gpu; ws_bowtie_spectral=nothing)

Pass 2 of the two-pass polychromatic projection: convert cached per-material path lengths
([`dd_fast_material_paths!`](@ref)) into one polychromatic log line-integral sinogram.

Bit-identical to the single-pass [`dd_fast_fused_poly_project!`](@ref) on the same inputs: the
accumulation inside a cell and the energy loop are the same expressions in the same order.
"""
function dd_fast_poly_from_paths!(
        sinogram::AbstractArray{T, 3},
        paths::AbstractArray{T, 4},
        μ_table_gpu::AbstractArray{T, 2},
        wη_gpu::AbstractArray{T, 1};
        ws_bowtie_spectral = nothing,
    ) where {T <: AbstractFloat}

    size(paths, 1) == size(μ_table_gpu, 1) || throw(DimensionMismatch(
        "paths has $(size(paths, 1)) materials, the attenuation table has $(size(μ_table_gpu, 1))"
    ))
    (size(paths, 2), size(paths, 3), size(paths, 4)) == size(sinogram) || throw(
        DimensionMismatch("paths $(size(paths)[2:4]) does not match the sinogram $(size(sinogram))")
    )

    n_cols = Int32(size(sinogram, 1))
    nc_nr = n_cols * Int32(size(sinogram, 2))
    has_bowtie = ws_bowtie_spectral !== nothing
    _bt = has_bowtie ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    return _dd_poly_from_paths!(
        Val(size(μ_table_gpu, 1)), sinogram, paths, μ_table_gpu, wη_gpu, _bt, has_bowtie,
        n_cols, Int32(size(sinogram, 2)), nc_nr)
end

export dd_fast_material_paths!, dd_fast_poly_from_paths!
