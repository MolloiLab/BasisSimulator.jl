# =============================================================================
# Distance-Driven (DD) Forward Projection — AcceleratedKernels.jl
# =============================================================================
#
# Julia / AcceleratedKernels.jl port of the **distance-driven** 3D forward
# projector from CatSim / XCIST (gecatsim), specifically
# `clib_build/src/DD3Proj_roi_notrans_mm.cpp` (the physical-mm, no-transpose
# variant).  Upstream is GE Precision HealthCare, Apache-2.0 / XCIST license
# (https://github.com/xcist/main).  Modifications relative to the original
# C++ are Copyright (c) 2025 MolloiLab and contributors (MIT, see ../../LICENSE).
#
# Reference: De Man B, Basu S. "Distance-driven projection and backprojection
# in three dimensions." Phys Med Biol. 2004;49(11):2463-2475.
# doi:10.1088/0031-9155/49/11/024
#
# Why a second projector
# ----------------------
# `siddon.jl` traces each ray independently (ray-driven).  Distance-driven
# instead maps image-voxel boundaries and detector-cell boundaries onto a
# common iso-plane (perpendicular to the dominant ray axis) and integrates
# their *overlap*.  This is the algorithm CatSim/XCIST uses and gives cleaner
# adjoint / lower high-frequency artifact behaviour than Siddon.
#
# Drop-in contract
# ----------------
# The four public entry points mirror the Siddon signatures and semantics
# EXACTLY, so every downstream consumer (BHC, IR, polychromatic EI,
# photon-counting PCCT, FDK) works unchanged:
#
#   dd_forward_project! / dd_forward_project   ←→ siddon_forward_project! / …
#   dd_fused_poly_project!                     ←→ siddon_fused_poly_project!
#   dd_fused_spectral_project!                 ←→ siddon_fused_spectral_project!
#
# Output-stationary (gather) reformulation
# ----------------------------------------
# The upstream C is image-stationary *scatter* (loops image rows, accumulates
# into a shared per-view detector buffer — a reduction).  We reformulate it
# output-stationary: ONE thread per output sinogram cell gathers the
# distance-driven overlap of its own back-projected footprint.  This is
# faithful DD (only the floating-point summation order differs) and maps onto
# `AK.foreachindex(sinogram)` exactly like Siddon — no atomics, no races,
# portable to Metal / CUDA / ROCm / CPU.
#
# Per-view axis choice (DD3 `vertical`): integrate along whichever world axis
# the source most aligns with (|s_y| ≥ |s_x| → integrate along y, else along x),
# keeping the boundary→iso-plane mapping well conditioned.  Each image row's
# contributing voxel range is bounded by inverse projection so per-cell work is
# O(n_rows × footprint), comparable to a Siddon ray walk.
#
# The fused poly/spectral kernels reuse Siddon's `@generated` NTuple helpers
# (`_fused_accum_energies`, `_fused_beer_lambert[_bt]`, `_tiled_*`) verbatim —
# only the per-voxel weight changes from Siddon's chord `path_length` to DD's
# overlap `ox·oz·norm`.
#
# Assumptions (match the DD3_mm variant)
# --------------------------------------
# - Flat detector, detector_v = ẑ (BasisSimulator's CTGeometry construction).
# - Isotropic in-plane voxels (vox_x ≈ vox_y); the DD3_mm obliquity scaling
#   uses a single in-plane voxel size.  Enforced with a warning.
# - Volume centred at isocentre with physical extent `volume_extent` (else
#   `geom.fov`), identical to Siddon.
#
# =============================================================================

import AcceleratedKernels as AK

export dd_forward_project!, dd_forward_project,
    dd_fused_poly_project!, dd_fused_spectral_project!

# NOTE (dd-raytracing branch): distance-driven is THE projector here.  The
# internal pipeline (`_forward_project_poly!`, `pcct_forward_project`, BHC, IR,
# IR-recon) calls the `dd_*` functions below directly, exactly as it used to
# call `siddon_*` — same signatures, so no downstream/notebook code changed.
# `siddon.jl` is retained only for reference/benchmarking; on merge it is
# expected to be deleted and these become the sole projector.

# -----------------------------------------------------------------------------
# Overlap length of two 1-D intervals [a_lo, a_hi] ∩ [b_lo, b_hi].
# -----------------------------------------------------------------------------
@inline function _dd_overlap(a_lo::T, a_hi::T, b_lo::T, b_hi::T) where {T}
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    return hi > lo ? hi - lo : zero(T)
end

# -----------------------------------------------------------------------------
# Contributing voxel-index range [i_start, i_end] along one axis: the voxels
# whose projected boundaries straddle [lo, hi] on the iso-plane.  Inverse of
#   pos(b) = s0 + (vmin + b*vstep - s0) * mag_fac   (boundary index b, 0-based)
# Widened by ±1 for float safety; non-overlapping extras are filtered by the
# `_dd_overlap > 0` test in the loop.
# -----------------------------------------------------------------------------
@inline function _dd_bounds(
        lo::T, hi::T, s0::T, vmin::T, vstep::T, inv_mf::T, n::Int32,
    ) where {T}
    c_lo = ((lo - s0) * inv_mf - (vmin - s0)) / vstep
    c_hi = ((hi - s0) * inv_mf - (vmin - s0)) / vstep
    c_lo = clamp(c_lo, zero(T), T(n) + one(T))
    c_hi = clamp(c_hi, zero(T), T(n) + one(T))
    i_start = max(Int32(1), unsafe_trunc(Int32, floor(c_lo)))
    i_end = min(n, unsafe_trunc(Int32, ceil(c_hi)) + Int32(1))
    return (i_start, i_end)
end

# -----------------------------------------------------------------------------
# Per-cell distance-driven geometry: maps the detector cell (col,row) onto the
# iso-plane and returns the footprint + normalisation shared by every DD kernel.
# Returns a flat tuple (compiler-inlined): the value-tuple plus a `valid` flag.
# -----------------------------------------------------------------------------
@inline function _dd_cell_setup(
        col::Int32, row::Int32,
        sx::T, sy::T, sz::T,
        dcx::T, dcy::T, dcz::T, ux::T, uy::T, vvz::T,
        mag::T, ps::T, prs::T, col_center::T, row_center::T,
        nx::Int32, ny::Int32,
        vmin_x::T, vmin_y::T, vx::T, vy::T,
    ) where {T}

    eps = T(1.0e-12)
    half = T(0.5)

    # detector COLUMN boundaries (transaxial), world (x,y) at detector mid-row
    uL = (T(col) - half - col_center) * ps * mag
    uU = (T(col) + half - col_center) * ps * mag
    bxL = dcx + uL * ux;  byL = dcy + uL * uy
    bxU = dcx + uU * ux;  byU = dcy + uU * uy

    # integration axis = dominant source component
    vertical = abs(sy) >= abs(sx)
    if vertical
        s_long = sy;  s_tran = sx
        dtL = bxL;  dlL = byL;  dtU = bxU;  dlU = byU
        n_t = nx;  v_t = vx;  vmin_t = vmin_x
        n_long = ny;  v_long = vy;  vmin_long = vmin_y
    else
        s_long = sx;  s_tran = sy
        dtL = byL;  dlL = bxL;  dtU = byU;  dlU = bxU
        n_t = ny;  v_t = vy;  vmin_t = vmin_y
        n_long = nx;  v_long = vx;  vmin_long = vmin_x
    end

    # project the two column boundaries onto the iso-plane (long = 0)
    scaleL = s_long / (s_long - dlL)
    scaleU = s_long / (s_long - dlU)
    dXa = s_tran + (dtL - s_tran) * scaleL
    dXb = s_tran + (dtU - s_tran) * scaleU
    dXlo = min(dXa, dXb);  dXhi = max(dXa, dXb)
    detXstep = dXhi - dXlo
    deltaT = half * (dXa + dXb) - s_tran
    scale_col = half * (scaleL + scaleU)

    # detector ROW boundaries (axial), world z, projected to iso-plane
    wL = (T(row) - half - row_center) * prs * mag
    wU = (T(row) + half - row_center) * prs * mag
    zdL = dcz + wL * vvz
    zdU = dcz + wU * vvz
    dZa = sz + scale_col * (zdL - sz)
    dZb = sz + scale_col * (zdU - sz)
    dZlo = min(dZa, dZb);  dZhi = max(dZa, dZb)
    detZstep = dZhi - dZlo
    deltaZ = half * (dZa + dZb) - sz

    valid = detXstep > eps && detZstep > eps
    invCos = sqrt(s_long * s_long + deltaT * deltaT + deltaZ * deltaZ) /
        abs(s_long) * v_long
    norm = valid ? invCos / (detXstep * detZstep) : zero(T)

    return (valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long)
end

# -----------------------------------------------------------------------------
# Single detector-cell mono line integral (Σμ·l), gather form.
# -----------------------------------------------------------------------------
@inline function _dd_trace_cell(
        volume::AbstractArray{T, 3},
        col::Int32, row::Int32,
        sx::T, sy::T, sz::T,
        dcx::T, dcy::T, dcz::T, ux::T, uy::T, vvz::T,
        mag::T, ps::T, prs::T, col_center::T, row_center::T,
        nx::Int32, ny::Int32, nz::Int32,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T,
    ) where {T}

    (valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long) = _dd_cell_setup(
        col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
        mag, ps, prs, col_center, row_center, nx, ny, vmin_x, vmin_y, vx, vy)

    valid || return zero(T)

    accum = zero(T)
    il = Int32(1)
    while il <= n_long
        lp = vmin_long + (T(il) - T(0.5)) * v_long
        mag_fac = s_long / (s_long - lp)
        inv_mf = one(T) / mag_fac

        (it_s, it_e) = _dd_bounds(dXlo, dXhi, s_tran, vmin_t, v_t, inv_mf, n_t)
        (ip_s, ip_e) = _dd_bounds(dZlo, dZhi, sz, vmin_z, vz, inv_mf, nz)

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
                    z0 = sz + (vmin_z + T(ip - Int32(1)) * vz - sz) * mag_fac
                    z1 = sz + (vmin_z + T(ip) * vz - sz) * mag_fac
                    oz = _dd_overlap(dZlo, dZhi, min(z0, z1), max(z0, z1))
                    if oz > zero(T)
                        accum += ox * oz * volume[ixv, iyv, ip]
                    end
                    ip += Int32(1)
                end
            end
            it += Int32(1)
        end
        il += Int32(1)
    end

    return norm * accum
end

# -----------------------------------------------------------------------------
# Helper: build a typed geometry array on the same backend as `ref`.
# -----------------------------------------------------------------------------
@inline function _dd_geom_array(ws, ref, src, ::Type{T}) where {T}
    ws !== nothing && return ws
    a = similar(ref, T, size(src)...)
    copyto!(a, T.(src))
    return a
end

# =============================================================================
# Mono: in-place + allocating — mirror siddon_forward_project! / …project
# =============================================================================
"""
    dd_forward_project!(sinogram, volume, geom; volume_extent=nothing, ...) -> sinogram

Distance-driven forward projection (CatSim/XCIST DD3, gather form).  Drop-in
replacement for [`siddon_forward_project!`](@ref): same arguments, same
`[n_cols, n_rows, n_angles]` sinogram, same cm units and Σμ·l output.
"""
function dd_forward_project!(
        sinogram::AbstractArray{T, 3},
        volume::AbstractArray{T, 3},
        geom::CTGeometry;
        ws_source_positions = nothing,
        ws_detector_centers = nothing,
        ws_detector_u = nothing,
        ws_detector_v = nothing,
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ) where {T <: AbstractFloat}

    nx = Int32(size(volume, 1)); ny = Int32(size(volume, 2)); nz = Int32(size(volume, 3))
    n_cols = Int32(size(sinogram, 1)); n_rows = Int32(size(sinogram, 2))

    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vmin_x = T(-vol_bounds[1] / 2); vmin_y = T(-vol_bounds[2] / 2); vmin_z = T(-vol_bounds[3] / 2)
    vx = T(vol_bounds[1]) / T(nx); vy = T(vol_bounds[2]) / T(ny); vz = T(vol_bounds[3]) / T(nz)
    _dd_check_isotropy(vx, vy)

    mag = T(geom.SDD / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    sp = _dd_geom_array(ws_source_positions, volume, geom.source_positions, T)
    dc = _dd_geom_array(ws_detector_centers, volume, geom.detector_centers, T)
    du = _dd_geom_array(ws_detector_u, volume, geom.detector_u, T)
    dv = _dd_geom_array(ws_detector_v, volume, geom.detector_v, T)

    AK.foreachindex(sinogram) do idx
        idx_0 = Int32(idx - 1)
        col = (idx_0 % n_cols) + Int32(1)
        idx_0 = idx_0 ÷ n_cols
        row = (idx_0 % n_rows) + Int32(1)
        angle = (idx_0 ÷ n_rows) + Int32(1)

        sinogram[idx] = _dd_trace_cell(
            volume, col, row,
            sp[1, angle], sp[2, angle], sp[3, angle],
            dc[1, angle], dc[2, angle], dc[3, angle], du[1, angle], du[2, angle], dv[3, angle],
            mag, ps, prs, col_center, row_center,
            nx, ny, nz, vmin_x, vmin_y, vmin_z, vx, vy, vz,
        )
    end
    return sinogram
end

"""
    dd_forward_project(volume, geom; volume_extent=nothing) -> sinogram

Allocating distance-driven forward projection.  Drop-in for
[`siddon_forward_project`](@ref).
"""
function dd_forward_project(
        volume::AbstractArray{T, 3},
        geom::CTGeometry;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ) where {T <: AbstractFloat}
    sinogram = similar(volume, T, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, zero(T))
    return dd_forward_project!(sinogram, volume, geom; volume_extent = volume_extent)
end

@inline function _dd_check_isotropy(vx::T, vy::T) where {T}
    if abs(vx - vy) > T(1.0e-4) * max(vx, vy)
        @warn "dd projector: in-plane voxels not isotropic (vx=$vx, vy=$vy); \
               DD3 obliquity scaling assumes vox_x ≈ vox_y — results may be biased." maxlog = 1
    end
    return nothing
end

"""
    dd_fused_poly_project!(sinogram, mask, geom, μ_table_gpu, wη_gpu, Val(N_E); kwargs...)

Distance-driven analog of [`siddon_fused_poly_project!`](@ref): one gather pass
per detector cell through the material `mask`, accumulating line integrals for
all `N_E` energy bins, then Beer-Lambert `-log(Σ_e wη[e]·exp(-L_e))`.  Same
arguments, output, and bowtie-spectral handling as the Siddon version.

See [`dd_fast_fused_poly_project!`](@ref) for the single-pass per-material
path-length variant (identical DD model, no per-energy register pressure).
"""
function dd_fused_poly_project!(
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

    nx = Int32(size(mask, 1)); ny = Int32(size(mask, 2)); nz = Int32(size(mask, 3))
    n_cols = Int32(size(sinogram, 1)); n_rows = Int32(size(sinogram, 2))

    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vmin_x = T(-vol_bounds[1] / 2); vmin_y = T(-vol_bounds[2] / 2); vmin_z = T(-vol_bounds[3] / 2)
    vx = T(vol_bounds[1]) / T(nx); vy = T(vol_bounds[2]) / T(ny); vz = T(vol_bounds[3]) / T(nz)
    _dd_check_isotropy(vx, vy)

    mag = T(geom.SDD / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    sp = _dd_geom_array(ws_source_positions, sinogram, geom.source_positions, T)
    dc = _dd_geom_array(ws_detector_centers, sinogram, geom.detector_centers, T)
    du = _dd_geom_array(ws_detector_u, sinogram, geom.detector_u, T)
    dv = _dd_geom_array(ws_detector_v, sinogram, geom.detector_v, T)

    nc_nr = n_cols * n_rows
    has_bowtie = ws_bowtie_spectral !== nothing
    _bt = has_bowtie ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    let mask = mask, μ_tbl = μ_table_gpu, wη = wη_gpu,
            sp = sp, dc = dc, du = du, dv = dv, bt = _bt,
            vmx = vmin_x, vmy = vmin_y, vmz = vmin_z, vsx = vx, vsy = vy, vsz = vz,
            nx = nx, ny = ny, nz = nz, nc = n_cols, nr = n_rows,
            mag = mag, ps = ps, prs = prs, cc = col_center, rc = row_center,
            ncnr = nc_nr, hbt = has_bowtie

        AK.foreachindex(sinogram) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            idx_0 = idx_0 ÷ nc
            row = (idx_0 % nr) + Int32(1)
            angle = (idx_0 ÷ nr) + Int32(1)

            sx = sp[1, angle]; sy = sp[2, angle]; sz = sp[3, angle]
            dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
            ux = du[1, angle]; uy = du[2, angle]; vvz = dv[3, angle]

            (valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
                n_t, v_t, vmin_t, n_long, v_long, vmin_long) = _dd_cell_setup(
                col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
                mag, ps, prs, cc, rc, nx, ny, vmx, vmy, vsx, vsy)

            accums = ntuple(_ -> zero(T), Val(N_E))
            if valid
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
                                    accums = _fused_accum_energies(accums, μ_tbl, mat, ox * oz * norm)
                                end
                                ip += Int32(1)
                            end
                        end
                        it += Int32(1)
                    end
                    il += Int32(1)
                end
            end

            I_total = if hbt
                bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
                _fused_beer_lambert_bt(accums, wη, bt, bt_base, ncnr)
            else
                _fused_beer_lambert(accums, wη)
            end
            sinogram[idx] = -log(max(I_total, T(1.0e-10)))
        end
    end
    return sinogram
end

"""
    dd_fused_spectral_project!(pilot, outputs_flat, n_bins, mask, geom,
        μ_table_gpu, W_gpu, Val(K), tile_start; kwargs...)

Distance-driven analog of [`siddon_fused_spectral_project!`](@ref): one gather
pass per detector cell accumulating `K` energy bins (tile starting at
`tile_start`), then ADDS per-bin partial Beer-Lambert sums to `outputs_flat`.
Same tiled-accumulation contract as the Siddon version (zero-init before the
first tile; tiles accumulate).

See [`dd_fast_fused_spectral_project!`](@ref) for the single-pass per-material
path-length variant (identical DD model; `K` can be the full energy count).
"""
function dd_fused_spectral_project!(
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

    nx = Int32(size(mask, 1)); ny = Int32(size(mask, 2)); nz = Int32(size(mask, 3))
    n_cols = Int32(size(pilot, 1)); n_rows = Int32(size(pilot, 2)); n_angles = Int32(size(pilot, 3))
    n_elem = n_cols * n_rows * n_angles

    vol_bounds = volume_extent !== nothing ? volume_extent : geom.fov
    vmin_x = T(-vol_bounds[1] / 2); vmin_y = T(-vol_bounds[2] / 2); vmin_z = T(-vol_bounds[3] / 2)
    vx = T(vol_bounds[1]) / T(nx); vy = T(vol_bounds[2]) / T(ny); vz = T(vol_bounds[3]) / T(nz)
    _dd_check_isotropy(vx, vy)

    mag = T(geom.SDD / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    col_center = (T(n_cols) + one(T)) / T(2)
    row_center = (T(n_rows) + one(T)) / T(2)

    sp = _dd_geom_array(ws_source_positions, mask, geom.source_positions, T)
    dc = _dd_geom_array(ws_detector_centers, mask, geom.detector_centers, T)
    du = _dd_geom_array(ws_detector_u, mask, geom.detector_u, T)
    dv = _dd_geom_array(ws_detector_v, mask, geom.detector_v, T)

    nc_nr = n_cols * n_rows
    has_src_spectral = ws_bowtie_spectral !== nothing
    _bt = has_src_spectral ? ws_bowtie_spectral : similar(μ_table_gpu, T, 1, 1, 1)

    let mask = mask, μ_tbl = μ_table_gpu, W = W_gpu, ts = tile_start,
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

            sx = sp[1, angle]; sy = sp[2, angle]; sz = sp[3, angle]
            dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
            ux = du[1, angle]; uy = du[2, angle]; vvz = dv[3, angle]

            (valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
                n_t, v_t, vmin_t, n_long, v_long, vmin_long) = _dd_cell_setup(
                col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
                mag, ps, prs, cc, rc, nx, ny, vmx, vmy, vsx, vsy)

            accums = ntuple(_ -> zero(T), Val(K))
            if valid
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
                                    accums = _tiled_accum_energies(accums, μ_tbl, mat, ox * oz * norm, ts)
                                end
                                ip += Int32(1)
                            end
                        end
                        it += Int32(1)
                    end
                    il += Int32(1)
                end
            end

            if hbt
                bt_base = Int32(col) + (Int32(row) - Int32(1)) * nc
                for b in Int32(1):nb
                    oflat[idx + (b - Int32(1)) * ne] += _tiled_beer_lambert_col_bt(accums, W, b, ts, bt, bt_base, ncnr)
                end
            else
                for b in Int32(1):nb
                    oflat[idx + (b - Int32(1)) * ne] += _tiled_beer_lambert_col(accums, W, b, ts)
                end
            end
        end
    end
    return outputs_flat
end
