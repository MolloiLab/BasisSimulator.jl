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
# The mono entry points mirror the Siddon signatures and semantics EXACTLY, so
# every downstream consumer (BHC, IR, FDK) works unchanged:
#
#   dd_forward_project! / dd_forward_project   ←→ siddon_forward_project! / …
#
# The polychromatic and spectral projections are dd_fast.jl's single-pass
# kernels (the same footprint and overlap weights, any number of materials).
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

export dd_forward_project!, dd_forward_project

# NOTE (dd-raytracing branch): distance-driven is THE projector here.  The
# internal pipeline (`_forward_project_poly!`, `pcct_forward_project`, BHC, IR,
# IR-recon) calls the `dd_*` functions below directly, exactly as it used to
# call `siddon_*` — same signatures, so no downstream/notebook code changed.
# `siddon.jl` is retained only for reference/benchmarking; on merge it is
# expected to be deleted and these become the sole projector.

# -----------------------------------------------------------------------------
# Every multiply-add of the walk's geometry is an explicit `fma`: a GPU compiler fuses `a * b + c`
# on its own and the CPU's does not, so one rounding everywhere keeps the CPU's and the device's
# footprints alike. (The `:dd_fast` path-length sums are agreed to rounding, not to the bit: see
# `_plen_accum`.)
#
# The iso-plane position of voxel boundary `b` (0-based) along an axis starting at `vmin` in steps
# of `vstep`, seen from the source at `s0` with magnification `mag_fac` (`_dd_bounds` inverts it).
# -----------------------------------------------------------------------------
@inline _dd_edge(s0::T, vmin::T, b::T, vstep::T, mag_fac::T) where {T} =
    fma(fma(b, vstep, vmin) - s0, mag_fac, s0)

# `(sin(γ), cos(γ))` of a fan angle by one Taylor polynomial in `fma`s, for the same reason: the
# CPU's `sin` and the device's differ by ulps. Its terms reach Float64's rounding for |γ| ≤ π/4,
# wider than any CT fan.
const _SIN_TERMS = ntuple(k -> (-1)^(k - 1) / factorial(2k - 1), 9)
const _COS_TERMS = ntuple(k -> (-1)^(k - 1) / factorial(2k - 2), 9)

@inline function _dd_sincos(γ::T) where {T}
    γ2 = γ * γ
    s = foldr((a, acc) -> fma(acc, γ2, T(a)), _SIN_TERMS[1:(end - 1)]; init = T(_SIN_TERMS[end]))
    c = foldr((a, acc) -> fma(acc, γ2, T(a)), _COS_TERMS[1:(end - 1)]; init = T(_COS_TERMS[end]))
    return (γ * s, c)
end

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
    c_lo = fma(lo - s0, inv_mf, s0 - vmin) / vstep
    c_hi = fma(hi - s0, inv_mf, s0 - vmin) / vstep
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
@inline function _dd_col_setup(
        col::Int32,
        sx::T, sy::T, sz::T,
        dcx::T, dcy::T, ux::T, uy::T,
        mag::T, ps::T, col_center::T,
        nx::Int32, ny::Int32,
        vmin_x::T, vmin_y::T, vx::T, vy::T,
        arc_det::Bool, dγ::T,
    ) where {T}
    half = T(0.5)

    # detector COLUMN boundaries (transaxial), world (x,y) at detector mid-row.
    # :arc → boundaries at fan angles γ ± Δγ/2 on the source-centred cylinder;
    # :flat → linear offsets along detector_u.
    bxL, byL, bxU, byU = if arc_det
        sL, cL = _dd_sincos((T(col) - half - col_center) * dγ)
        sU, cU = _dd_sincos((T(col) + half - col_center) * dγ)
        cin_x = dcx - sx
        cin_y = dcy - sy
        Lsd = sqrt(fma(cin_x, cin_x, cin_y * cin_y))
        (
            fma(sL * Lsd, ux, fma(cL, cin_x, sx)),
            fma(sL * Lsd, uy, fma(cL, cin_y, sy)),
            fma(sU * Lsd, ux, fma(cU, cin_x, sx)),
            fma(sU * Lsd, uy, fma(cU, cin_y, sy)),
        )
    else
        uL = (T(col) - half - col_center) * ps * mag
        uU = (T(col) + half - col_center) * ps * mag
        (
            fma(uL, ux, dcx), fma(uL, uy, dcy),
            fma(uU, ux, dcx), fma(uU, uy, dcy),
        )
    end

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
    dXa = fma(dtL - s_tran, scaleL, s_tran)
    dXb = fma(dtU - s_tran, scaleU, s_tran)
    dXlo = min(dXa, dXb);  dXhi = max(dXa, dXb)
    detXstep = dXhi - dXlo
    deltaT = fma(half, dXa + dXb, -s_tran)
    scale_col = half * (scaleL + scaleU)

    return (
        detXstep > T(1.0e-12), vertical, s_long, s_tran,
        dXlo, dXhi, detXstep, deltaT, scale_col,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long,
    )
end

@inline function _dd_row_setup(
        row::Int32, sz::T, dcz::T, vvz::T,
        mag::T, prs::T, row_center::T, scale_col::T,
        s_long::T, deltaT::T, v_long::T, detXstep::T,
    ) where {T}
    half = T(0.5)
    wL = (T(row) - half - row_center) * prs * mag
    wU = (T(row) + half - row_center) * prs * mag
    zdL = fma(wL, vvz, dcz)
    zdU = fma(wU, vvz, dcz)
    dZa = fma(scale_col, zdL - sz, sz)
    dZb = fma(scale_col, zdU - sz, sz)
    dZlo = min(dZa, dZb);  dZhi = max(dZa, dZb)
    detZstep = dZhi - dZlo
    deltaZ = fma(half, dZa + dZb, -sz)

    valid = detXstep > T(1.0e-12) && detZstep > T(1.0e-12)
    invCos = sqrt(fma(deltaZ, deltaZ, fma(deltaT, deltaT, s_long * s_long))) /
        abs(s_long) * v_long
    norm = valid ? invCos / (detXstep * detZstep) : zero(T)

    return (valid, dZlo, dZhi, norm)
end

@inline function _dd_cell_setup(
        col::Int32, row::Int32,
        sx::T, sy::T, sz::T,
        dcx::T, dcy::T, dcz::T, ux::T, uy::T, vvz::T,
        mag::T, ps::T, prs::T, col_center::T, row_center::T,
        nx::Int32, ny::Int32,
        vmin_x::T, vmin_y::T, vx::T, vy::T,
        arc_det::Bool, dγ::T,
    ) where {T}
    (
        valid_x, vertical, s_long, s_tran, dXlo, dXhi, detXstep,
        deltaT, scale_col, n_t, v_t, vmin_t, n_long, v_long,
        vmin_long,
    ) = _dd_col_setup(
        col, sx, sy, sz, dcx, dcy, ux, uy, mag, ps, col_center,
        nx, ny, vmin_x, vmin_y, vx, vy, arc_det, dγ
    )
    (valid_z, dZlo, dZhi, norm) = _dd_row_setup(
        row, sz, dcz, vvz, mag, prs, row_center, scale_col,
        s_long, deltaT, v_long, detXstep
    )
    valid = valid_x && valid_z
    return (
        valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi,
        valid ? norm : zero(T),
        n_t, v_t, vmin_t, n_long, v_long, vmin_long,
    )
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
        arc_det::Bool, dγ::T,
    ) where {T}

    (
        valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
        n_t, v_t, vmin_t, n_long, v_long, vmin_long,
    ) = _dd_cell_setup(
        col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz,
        mag, ps, prs, col_center, row_center, nx, ny, vmin_x, vmin_y, vx, vy,
        arc_det, dγ
    )

    valid || return zero(T)

    accum = zero(T)
    il = Int32(1)
    while il <= n_long
        lp = fma(T(il) - T(0.5), v_long, vmin_long)
        mag_fac = s_long / (s_long - lp)
        inv_mf = one(T) / mag_fac

        (it_s, it_e) = _dd_bounds(dXlo, dXhi, s_tran, vmin_t, v_t, inv_mf, n_t)
        (ip_s, ip_e) = _dd_bounds(dZlo, dZhi, sz, vmin_z, vz, inv_mf, nz)

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
                    z0 = _dd_edge(sz, vmin_z, T(ip - Int32(1)), vz, mag_fac)
                    z1 = _dd_edge(sz, vmin_z, T(ip), vz, mag_fac)
                    oz = _dd_overlap(dZlo, dZhi, min(z0, z1), max(z0, z1))
                    if oz > zero(T)
                        accum = fma(ox * oz, volume[ixv, iyv, ip], accum)
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

# Four adjacent arc-detector rows share the complete column/view setup and the
# transaxial voxel traversal.  Each accumulator still observes voxel
# contributions in the scalar kernel's exact `il -> it -> ip` order.
@inline function _dd_trace_rows4(
        volume::AbstractArray{T, 3}, col::Int32, row1::Int32, n_rows::Int32,
        sx::T, sy::T, sz::T, dcx::T, dcy::T, dcz::T, ux::T, uy::T, vvz::T,
        mag::T, ps::T, prs::T, col_center::T, row_center::T,
        nx::Int32, ny::Int32, nz::Int32,
        vmin_x::T, vmin_y::T, vmin_z::T, vx::T, vy::T, vz::T, dγ::T,
    ) where {T}
    (
        valid_x, vertical, s_long, s_tran, dXlo, dXhi, detXstep,
        deltaT, scale_col, n_t, v_t, vmin_t, n_long, v_long,
        vmin_long,
    ) = _dd_col_setup(
        col, sx, sy, sz, dcx, dcy, ux, uy, mag, ps, col_center,
        nx, ny, vmin_x, vmin_y, vx, vy, true, dγ
    )

    row2 = row1 + Int32(1); row3 = row1 + Int32(2); row4 = row1 + Int32(3)
    use1 = row1 <= n_rows; use2 = row2 <= n_rows
    use3 = row3 <= n_rows; use4 = row4 <= n_rows
    valid1, dZlo1, dZhi1, norm1 = _dd_row_setup(
        row1, sz, dcz, vvz, mag, prs, row_center, scale_col,
        s_long, deltaT, v_long, detXstep
    )
    valid2, dZlo2, dZhi2, norm2 = _dd_row_setup(
        row2, sz, dcz, vvz, mag, prs, row_center, scale_col,
        s_long, deltaT, v_long, detXstep
    )
    valid3, dZlo3, dZhi3, norm3 = _dd_row_setup(
        row3, sz, dcz, vvz, mag, prs, row_center, scale_col,
        s_long, deltaT, v_long, detXstep
    )
    valid4, dZlo4, dZhi4, norm4 = _dd_row_setup(
        row4, sz, dcz, vvz, mag, prs, row_center, scale_col,
        s_long, deltaT, v_long, detXstep
    )
    valid1 = valid_x && valid1 && use1; valid2 = valid_x && valid2 && use2
    valid3 = valid_x && valid3 && use3; valid4 = valid_x && valid4 && use4

    acc1 = zero(T); acc2 = zero(T); acc3 = zero(T); acc4 = zero(T)
    il = Int32(1)
    while il <= n_long
        lp = fma(T(il) - T(0.5), v_long, vmin_long)
        mag_fac = s_long / (s_long - lp)
        inv_mf = one(T) / mag_fac
        it_s, it_e = _dd_bounds(dXlo, dXhi, s_tran, vmin_t, v_t, inv_mf, n_t)
        ip_s1, ip_e1 = _dd_bounds(dZlo1, dZhi1, sz, vmin_z, vz, inv_mf, nz)
        ip_s2, ip_e2 = _dd_bounds(dZlo2, dZhi2, sz, vmin_z, vz, inv_mf, nz)
        ip_s3, ip_e3 = _dd_bounds(dZlo3, dZhi3, sz, vmin_z, vz, inv_mf, nz)
        ip_s4, ip_e4 = _dd_bounds(dZlo4, dZhi4, sz, vmin_z, vz, inv_mf, nz)

        it = it_s
        while it <= it_e
            t0 = _dd_edge(s_tran, vmin_t, T(it - Int32(1)), v_t, mag_fac)
            t1 = _dd_edge(s_tran, vmin_t, T(it), v_t, mag_fac)
            ox = _dd_overlap(dXlo, dXhi, min(t0, t1), max(t0, t1))
            if ox > zero(T)
                ixv = vertical ? it : il
                iyv = vertical ? il : it
                if valid1
                    ip = ip_s1
                    while ip <= ip_e1
                        z0 = _dd_edge(sz, vmin_z, T(ip - Int32(1)), vz, mag_fac)
                        z1 = _dd_edge(sz, vmin_z, T(ip), vz, mag_fac)
                        oz = _dd_overlap(dZlo1, dZhi1, min(z0, z1), max(z0, z1))
                        oz > zero(T) && (acc1 = fma(ox * oz, volume[ixv, iyv, ip], acc1))
                        ip += Int32(1)
                    end
                end
                if valid2
                    ip = ip_s2
                    while ip <= ip_e2
                        z0 = _dd_edge(sz, vmin_z, T(ip - Int32(1)), vz, mag_fac)
                        z1 = _dd_edge(sz, vmin_z, T(ip), vz, mag_fac)
                        oz = _dd_overlap(dZlo2, dZhi2, min(z0, z1), max(z0, z1))
                        oz > zero(T) && (acc2 = fma(ox * oz, volume[ixv, iyv, ip], acc2))
                        ip += Int32(1)
                    end
                end
                if valid3
                    ip = ip_s3
                    while ip <= ip_e3
                        z0 = _dd_edge(sz, vmin_z, T(ip - Int32(1)), vz, mag_fac)
                        z1 = _dd_edge(sz, vmin_z, T(ip), vz, mag_fac)
                        oz = _dd_overlap(dZlo3, dZhi3, min(z0, z1), max(z0, z1))
                        oz > zero(T) && (acc3 = fma(ox * oz, volume[ixv, iyv, ip], acc3))
                        ip += Int32(1)
                    end
                end
                if valid4
                    ip = ip_s4
                    while ip <= ip_e4
                        z0 = _dd_edge(sz, vmin_z, T(ip - Int32(1)), vz, mag_fac)
                        z1 = _dd_edge(sz, vmin_z, T(ip), vz, mag_fac)
                        oz = _dd_overlap(dZlo4, dZhi4, min(z0, z1), max(z0, z1))
                        oz > zero(T) && (acc4 = fma(ox * oz, volume[ixv, iyv, ip], acc4))
                        ip += Int32(1)
                    end
                end
            end
            it += Int32(1)
        end
        il += Int32(1)
    end
    return (norm1 * acc1, norm2 * acc2, norm3 * acc3, norm4 * acc4)
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
    arc_det = is_arc(geom)
    dγ = T(geom.pixel_size / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    col_center = column_center(T, geom)
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
            arc_det, dγ,
        )
    end
    return sinogram
end

# HIR-only arc fast path: four detector rows share one column/view DD setup.
# The public `dd_forward_project!` remains the generic reference entry point.
function _dd_forward_project_arc_rowtile4!(
        sinogram::AbstractArray{T, 3}, volume::AbstractArray{T, 3}, geom::CTGeometry;
        ws_source_positions = nothing, ws_detector_centers = nothing,
        ws_detector_u = nothing, ws_detector_v = nothing,
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
    ) where {T <: AbstractFloat}
    is_arc(geom) || return dd_forward_project!(
        sinogram, volume, geom; ws_source_positions, ws_detector_centers,
        ws_detector_u, ws_detector_v, volume_extent
    )

    nx = Int32(size(volume, 1)); ny = Int32(size(volume, 2)); nz = Int32(size(volume, 3))
    nc = Int32(size(sinogram, 1)); nr = Int32(size(sinogram, 2))
    na = Int32(size(sinogram, 3)); ntiles = (nr + Int32(3)) ÷ Int32(4)
    bounds = volume_extent === nothing ? geom.fov : volume_extent
    vmx = T(-bounds[1] / 2); vmy = T(-bounds[2] / 2); vmz = T(-bounds[3] / 2)
    vsx = T(bounds[1]) / T(nx); vsy = T(bounds[2]) / T(ny); vsz = T(bounds[3]) / T(nz)
    _dd_check_isotropy(vsx, vsy)
    mag = T(geom.SDD / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    cc = column_center(T, geom); rc = (T(nr) + one(T)) / T(2)
    dγ = T(geom.pixel_size / geom.SAD)
    sp = _dd_geom_array(ws_source_positions, volume, geom.source_positions, T)
    dc = _dd_geom_array(ws_detector_centers, volume, geom.detector_centers, T)
    du = _dd_geom_array(ws_detector_u, volume, geom.detector_u, T)
    dv = _dd_geom_array(ws_detector_v, volume, geom.detector_v, T)
    let sino = sinogram, vol = volume, sp = sp, dc = dc, du = du, dv = dv,
            nx = nx, ny = ny, nz = nz, nc = nc, nr = nr, na = na,
            vmx = vmx, vmy = vmy, vmz = vmz, vsx = vsx, vsy = vsy, vsz = vsz,
            mag = mag, ps = ps, prs = prs, cc = cc, rc = rc, dγ = dγ
        AK.foreachindex(sino) do idx
            idx0 = Int32(idx - 1)
            col = (idx0 % nc) + Int32(1)
            idx0 = idx0 ÷ nc
            slot = idx0 % nr
            angle = (idx0 ÷ nr) + Int32(1)
            slot >= ntiles && return
            tile = slot
            row1 = tile * Int32(4) + Int32(1)
            row2 = row1 + Int32(1); row3 = row1 + Int32(2); row4 = row1 + Int32(3)
            a1, a2, a3, a4 = _dd_trace_rows4(
                vol, col, row1, nr,
                sp[1, angle], sp[2, angle], sp[3, angle],
                dc[1, angle], dc[2, angle], dc[3, angle],
                du[1, angle], du[2, angle], dv[3, angle],
                mag, ps, prs, cc, rc, nx, ny, nz,
                vmx, vmy, vmz, vsx, vsy, vsz, dγ
            )
            sino[col, row1, angle] = a1
            row2 <= nr && (sino[col, row2, angle] = a2)
            row3 <= nr && (sino[col, row3, angle] = a3)
            row4 <= nr && (sino[col, row4, angle] = a4)
        end
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
