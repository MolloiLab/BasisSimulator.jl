# =============================================================================
# Exact transpose of the monochromatic distance-driven projector
# =============================================================================

export dd_backproject!

const _DD_BP_Z_TILE = Int32(4)

@inline function _dd_arc_row_bounds(
        iz::Int32, vmz::T, vsz::T, sz::T, dcz::T, sd_xy::T,
        rho_min::T, rho_max::T, prm::T, rc::T, nr::Int32,
    ) where T
    half = T(0.5)
    zlo = vmz + T(iz - Int32(1)) * vsz - sz
    zhi = vmz + T(iz) * vsz - sz
    a = (sz + sd_xy * zlo / rho_min - dcz) / prm + rc
    b = (sz + sd_xy * zlo / rho_max - dcz) / prm + rc
    c = (sz + sd_xy * zhi / rho_min - dcz) / prm + rc
    d = (sz + sd_xy * zhi / rho_max - dcz) / prm + rc
    rmin = min(a, b, c, d); rmax = max(a, b, c, d)
    return (
        max(Int32(1), unsafe_trunc(Int32, ceil(rmin - half)) - Int32(1)),
        min(nr, unsafe_trunc(Int32, floor(rmax + half)) + Int32(1)),
    )
end

@inline function _dd_project_point(
        x::T, y::T, z::T,
        sx::T, sy::T, sz::T,
        dcx::T, dcy::T, dcz::T,
        ux::T, uy::T, uz::T,
        vx::T, vy::T, vz::T,
        pixel_mag::T, pixel_row_mag::T,
        col_center::T, row_center::T,
        arc_det::Bool, dγ::T,
    ) where T
    svx = x - sx; svy = y - sy; svz = z - sz
    sdx = dcx - sx; sdy = dcy - sy; sdz = dcz - sz
    if arc_det
        sd_xy = sqrt(sdx * sdx + sdy * sdy)
        sv_xy = sqrt(svx * svx + svy * svy)
        if sd_xy <= T(1.0e-12) || sv_xy <= T(1.0e-12)
            return (T(Inf), T(Inf))
        end
        along = (svx * sdx + svy * sdy) / sd_xy
        across = svx * ux + svy * uy
        γ = atan(across, along)
        row = svz * (sd_xy / sv_xy) / pixel_row_mag + row_center
        return (γ / dγ + col_center, row)
    end

    denom = svx * sdx + svy * sdy + svz * sdz
    abs(denom) <= T(1.0e-12) && return (T(Inf), T(Inf))
    t = (sdx * sdx + sdy * sdy + sdz * sdz) / denom
    dpx = sx + t * svx - dcx
    dpy = sy + t * svy - dcy
    dpz = sz + t * svz - dcz
    return (
        (dpx * ux + dpy * uy + dpz * uz) / pixel_mag + col_center,
        (dpx * vx + dpy * vy + dpz * vz) / pixel_row_mag + row_center,
    )
end

# Arc-detector HIR hot path.  Four active z voxels share all transverse
# voxel/view geometry while retaining an independent accumulator whose update
# order is angle -> column -> row, exactly as in the scalar reference kernel.
# The fixed tile avoids dynamic local arrays (and their GPU address-space
# lowering) on both Metal and CUDA; the final tile is simply predicated.
function _dd_backproject_arc_tile4!(
        volume::AbstractArray{T, 3}, sinogram::AbstractArray{T, 3},
        geom::CTGeometry, bounds::NTuple{3, Float64}, active_z::UnitRange{Int},
        circular_support::Bool, sp, dc, du, dv,
    ) where T <: AbstractFloat
    nx = Int32(size(volume, 1)); ny = Int32(size(volume, 2)); nz = Int32(size(volume, 3))
    nc = Int32(size(sinogram, 1)); nr = Int32(size(sinogram, 2))
    na = Int32(size(sinogram, 3))
    z_first = Int32(first(active_z)); z_last = Int32(last(active_z))
    active_nz = z_last - z_first + Int32(1)
    ntiles = (active_nz + _DD_BP_Z_TILE - Int32(1)) ÷ _DD_BP_Z_TILE

    vmx = T(-bounds[1] / 2); vmy = T(-bounds[2] / 2); vmz = T(-bounds[3] / 2)
    vsx = T(bounds[1]) / T(nx); vsy = T(bounds[2]) / T(ny); vsz = T(bounds[3]) / T(nz)
    mag = T(geom.SDD / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    prm = prs * mag
    cc = (T(nc) + one(T)) / T(2); rc = (T(nr) + one(T)) / T(2)
    dγ = T(geom.pixel_size / geom.SAD)
    radius_sq = T(min(bounds[1], bounds[2]) / 2)^2
    # Only the shape is used for iteration.  `ntiles <= active_nz <= nz`.
    tile_domain = view(volume, :, :, Int(z_first):(Int(z_first) + Int(ntiles) - 1))

    let vol = volume, sino = sinogram, sp = sp, dc = dc, du = du, dv = dv,
            nx = nx, ny = ny, nc = nc, nr = nr, na = na,
            z_first = z_first, z_last = z_last,
            vmx = vmx, vmy = vmy, vmz = vmz, vsx = vsx, vsy = vsy, vsz = vsz,
            mag = mag, ps = ps, prs = prs, prm = prm, cc = cc, rc = rc, dγ = dγ,
            circular_support = circular_support, radius_sq = radius_sq
        AK.foreachindex(tile_domain) do idx
            idx0 = Int32(idx - 1)
            ix = (idx0 % nx) + Int32(1)
            iy = ((idx0 ÷ nx) % ny) + Int32(1)
            tile = idx0 ÷ (nx * ny)
            iz1 = z_first + tile * Int32(4)
            iz2 = iz1 + Int32(1); iz3 = iz1 + Int32(2); iz4 = iz1 + Int32(3)
            use2 = iz2 <= z_last; use3 = iz3 <= z_last; use4 = iz4 <= z_last

            half = T(0.5)
            xc = vmx + (T(ix) - half) * vsx
            yc = vmy + (T(iy) - half) * vsy
            if circular_support && xc * xc + yc * yc > radius_sq
                vol[ix, iy, iz1] = zero(T)
                use2 && (vol[ix, iy, iz2] = zero(T))
                use3 && (vol[ix, iy, iz3] = zero(T))
                use4 && (vol[ix, iy, iz4] = zero(T))
                return
            end

            acc1 = zero(T); acc2 = zero(T); acc3 = zero(T); acc4 = zero(T)
            angle = Int32(1)
            while angle <= na
                sx = sp[1, angle]; sy = sp[2, angle]; sz = sp[3, angle]
                dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
                ux = du[1, angle]; uy = du[2, angle]
                vz = dv[3, angle]

                vertical = abs(sy) >= abs(sx)
                s_long = vertical ? sy : sx
                s_tran = vertical ? sx : sy
                il = vertical ? iy : ix
                it = vertical ? ix : iy
                v_long = vertical ? vsy : vsx
                vmin_long = vertical ? vmy : vmx
                v_t = vertical ? vsx : vsy
                vmin_t = vertical ? vmx : vmy
                lp = vmin_long + (T(il) - half) * v_long
                mf = s_long / (s_long - lp)
                t0 = s_tran + (vmin_t + T(it - Int32(1)) * v_t - s_tran) * mf
                t1 = s_tran + (vmin_t + T(it) * v_t - s_tran) * mf
                tvlo = min(t0, t1); tvhi = max(t0, t1)

                # Shared fan-angle and radial extrema for all four z lanes.
                sdx = dcx - sx; sdy = dcy - sy
                sd_xy = sqrt(sdx * sdx + sdy * sdy)
                cmin = T(Inf); cmax = T(-Inf)
                bx = Int32(-1)
                while bx <= Int32(1)
                    by = Int32(-1)
                    while by <= Int32(1)
                        svx = xc + T(bx) * half * vsx - sx
                        svy = yc + T(by) * half * vsy - sy
                        along = (svx * sdx + svy * sdy) / sd_xy
                        across = svx * ux + svy * uy
                        c = atan(across, along) / dγ + cc
                        cmin = min(cmin, c); cmax = max(cmax, c)
                        by += Int32(2)
                    end
                    bx += Int32(2)
                end
                c0 = max(Int32(1), unsafe_trunc(Int32, ceil(cmin - half)) - Int32(1))
                c1 = min(nc, unsafe_trunc(Int32, floor(cmax + half)) + Int32(1))

                x0 = xc - half * vsx; x1 = xc + half * vsx
                y0 = yc - half * vsy; y1 = yc + half * vsy
                qx = sx - clamp(sx, x0, x1); qy = sy - clamp(sy, y0, y1)
                rho_min = max(sqrt(qx * qx + qy * qy), T(1.0e-12))
                rho_max_sq = zero(T)
                bx = Int32(-1)
                while bx <= Int32(1)
                    by = Int32(-1)
                    while by <= Int32(1)
                        qxc = xc + T(bx) * half * vsx - sx
                        qyc = yc + T(by) * half * vsy - sy
                        rho_max_sq = max(rho_max_sq, qxc * qxc + qyc * qyc)
                        by += Int32(2)
                    end
                    bx += Int32(2)
                end
                rho_max = sqrt(rho_max_sq)

                # Lane-local axial candidate ranges and projected voxel bounds.
                z01 = sz + (vmz + T(iz1 - Int32(1)) * vsz - sz) * mf
                z11 = sz + (vmz + T(iz1) * vsz - sz) * mf
                z02 = sz + (vmz + T(iz2 - Int32(1)) * vsz - sz) * mf
                z12 = sz + (vmz + T(iz2) * vsz - sz) * mf
                z03 = sz + (vmz + T(iz3 - Int32(1)) * vsz - sz) * mf
                z13 = sz + (vmz + T(iz3) * vsz - sz) * mf
                z04 = sz + (vmz + T(iz4 - Int32(1)) * vsz - sz) * mf
                z14 = sz + (vmz + T(iz4) * vsz - sz) * mf

                r01, r11 = _dd_arc_row_bounds(
                    iz1, vmz, vsz, sz, dcz, sd_xy, rho_min, rho_max, prm, rc, nr)
                r02, r12 = _dd_arc_row_bounds(
                    iz2, vmz, vsz, sz, dcz, sd_xy, rho_min, rho_max, prm, rc, nr)
                r03, r13 = _dd_arc_row_bounds(
                    iz3, vmz, vsz, sz, dcz, sd_xy, rho_min, rho_max, prm, rc, nr)
                r04, r14 = _dd_arc_row_bounds(
                    iz4, vmz, vsz, sz, dcz, sd_xy, rho_min, rho_max, prm, rc, nr)

                col = c0
                while col <= c1
                    valid_x, _, s_long_c, _, dXlo, dXhi, detXstep,
                        deltaT, scale_col, _, _, _, _, v_long_c, _ = _dd_col_setup(
                        col, sx, sy, sz, dcx, dcy, ux, uy,
                        mag, ps, cc, nx, ny, vmx, vmy, vsx, vsy, true, dγ)
                    if valid_x
                        ox = _dd_overlap(dXlo, dXhi, tvlo, tvhi)
                        if ox > zero(T)
                            row = r01
                            while row <= r11
                                valid_z, dZlo, dZhi, norm = _dd_row_setup(
                                    row, sz, dcz, vz, mag, prs, rc, scale_col,
                                    s_long_c, deltaT, v_long_c, detXstep)
                                if valid_z
                                    oz = _dd_overlap(dZlo, dZhi, min(z01, z11), max(z01, z11))
                                    oz > zero(T) && (acc1 += sino[col, row, angle] * ox * oz * norm)
                                end
                                row += Int32(1)
                            end
                            if use2
                                row = r02
                                while row <= r12
                                    valid_z, dZlo, dZhi, norm = _dd_row_setup(
                                        row, sz, dcz, vz, mag, prs, rc, scale_col,
                                        s_long_c, deltaT, v_long_c, detXstep)
                                    if valid_z
                                        oz = _dd_overlap(dZlo, dZhi, min(z02, z12), max(z02, z12))
                                        oz > zero(T) && (acc2 += sino[col, row, angle] * ox * oz * norm)
                                    end
                                    row += Int32(1)
                                end
                            end
                            if use3
                                row = r03
                                while row <= r13
                                    valid_z, dZlo, dZhi, norm = _dd_row_setup(
                                        row, sz, dcz, vz, mag, prs, rc, scale_col,
                                        s_long_c, deltaT, v_long_c, detXstep)
                                    if valid_z
                                        oz = _dd_overlap(dZlo, dZhi, min(z03, z13), max(z03, z13))
                                        oz > zero(T) && (acc3 += sino[col, row, angle] * ox * oz * norm)
                                    end
                                    row += Int32(1)
                                end
                            end
                            if use4
                                row = r04
                                while row <= r14
                                    valid_z, dZlo, dZhi, norm = _dd_row_setup(
                                        row, sz, dcz, vz, mag, prs, rc, scale_col,
                                        s_long_c, deltaT, v_long_c, detXstep)
                                    if valid_z
                                        oz = _dd_overlap(dZlo, dZhi, min(z04, z14), max(z04, z14))
                                        oz > zero(T) && (acc4 += sino[col, row, angle] * ox * oz * norm)
                                    end
                                    row += Int32(1)
                                end
                            end
                        end
                    end
                    col += Int32(1)
                end
                angle += Int32(1)
            end
            vol[ix, iy, iz1] = acc1
            use2 && (vol[ix, iy, iz2] = acc2)
            use3 && (vol[ix, iy, iz3] = acc3)
            use4 && (vol[ix, iy, iz4] = acc4)
        end
    end
    return volume
end

"""
    dd_backproject!(volume, sinogram, geom; volume_extent=nothing, ...)

Apply the exact algebraic transpose of [`dd_forward_project!`](@ref). Each
voxel gathers the detector cells overlapped by its projected footprint and
reuses `_dd_cell_setup` plus the identical `ox * oz * norm` coefficient from
the forward operator. The gather formulation is deterministic and needs no
floating-point atomics on GPU backends.
"""
function dd_backproject!(
        volume::AbstractArray{T, 3},
        sinogram::AbstractArray{T, 3},
        geom::CTGeometry;
        volume_extent::Union{Nothing, NTuple{3, Float64}} = nothing,
        active_z::Union{Nothing, UnitRange{Int}} = nothing,
        circular_support::Bool = false,
        ws_source_positions = nothing,
        ws_detector_centers = nothing,
        ws_detector_u = nothing,
        ws_detector_v = nothing,
    ) where T <: AbstractFloat
    nx = Int32(size(volume, 1)); ny = Int32(size(volume, 2)); nz = Int32(size(volume, 3))
    nc = Int32(size(sinogram, 1)); nr = Int32(size(sinogram, 2))
    na = Int32(size(sinogram, 3))

    if active_z !== nothing && (isempty(active_z) || first(active_z) < 1 || last(active_z) > nz)
        throw(ArgumentError("active_z must be a nonempty unit range within 1:$(nz), got $(active_z)"))
    end

    bounds = volume_extent === nothing ? geom.fov : volume_extent
    vmx = T(-bounds[1] / 2); vmy = T(-bounds[2] / 2); vmz = T(-bounds[3] / 2)
    vsx = T(bounds[1]) / T(nx); vsy = T(bounds[2]) / T(ny); vsz = T(bounds[3]) / T(nz)
    _dd_check_isotropy(vsx, vsy)

    mag = T(geom.SDD / geom.SAD)
    ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    pm = ps * mag; prm = prs * mag
    cc = (T(nc) + one(T)) / T(2); rc = (T(nr) + one(T)) / T(2)
    arc_det = is_arc(geom); dγ = T(geom.pixel_size / geom.SAD)
    z_first = active_z === nothing ? Int32(1) : Int32(first(active_z))
    z_last = active_z === nothing ? nz : Int32(last(active_z))
    active_volume = view(volume, :, :, Int(z_first):Int(z_last))
    active_nz = z_last - z_first + Int32(1)
    radius_sq = T(min(bounds[1], bounds[2]) / 2)^2

    sp = _dd_geom_array(ws_source_positions, volume, geom.source_positions, T)
    dc = _dd_geom_array(ws_detector_centers, volume, geom.detector_centers, T)
    du = _dd_geom_array(ws_detector_u, volume, geom.detector_u, T)
    dv = _dd_geom_array(ws_detector_v, volume, geom.detector_v, T)

    if arc_det && active_z !== nothing
        return _dd_backproject_arc_tile4!(
            volume, sinogram, geom, bounds, active_z, circular_support,
            sp, dc, du, dv)
    end

    let sino = sinogram, sp = sp, dc = dc, du = du, dv = dv,
            nx = nx, ny = ny, nz = nz, nc = nc, nr = nr, na = na,
            vmx = vmx, vmy = vmy, vmz = vmz, vsx = vsx, vsy = vsy, vsz = vsz,
            mag = mag, ps = ps, prs = prs, pm = pm, prm = prm,
            cc = cc, rc = rc, arc_det = arc_det, dγ = dγ,
            z_first = z_first, z_last = z_last, circular_support = circular_support,
            radius_sq = radius_sq
        AK.foreachindex(active_volume) do idx
            idx0 = Int32(idx - 1)
            ix = (idx0 % nx) + Int32(1)
            iy = ((idx0 ÷ nx) % ny) + Int32(1)
            iz = (idx0 ÷ (nx * ny)) + z_first
            xc = vmx + (T(ix) - T(0.5)) * vsx
            yc = vmy + (T(iy) - T(0.5)) * vsy
            if circular_support && xc * xc + yc * yc > radius_sq
                active_volume[idx] = zero(T)
                return
            end
            zc = vmz + (T(iz) - T(0.5)) * vsz
            half = T(0.5)
            acc = zero(T)

            angle = Int32(1)
            while angle <= na
                sx = sp[1, angle]; sy = sp[2, angle]; sz = sp[3, angle]
                dcx = dc[1, angle]; dcy = dc[2, angle]; dcz = dc[3, angle]
                ux = du[1, angle]; uy = du[2, angle]; uz = du[3, angle]
                vx = dv[1, angle]; vy = dv[2, angle]; vz = dv[3, angle]

                # Voxel/view geometry is independent of the candidate detector
                # cell.  Hoist it out of the hot col×row loop; `_dd_cell_setup`
                # returns the same values but historically caused these divides
                # and projected voxel-boundary calculations to be repeated for
                # every candidate cell.
                vertical_v = abs(sy) >= abs(sx)
                s_long_v = vertical_v ? sy : sx
                s_tran_v = vertical_v ? sx : sy
                il_v = vertical_v ? iy : ix
                it_v = vertical_v ? ix : iy
                v_long_v = vertical_v ? vsy : vsx
                vmin_long_v = vertical_v ? vmy : vmx
                v_t_v = vertical_v ? vsx : vsy
                vmin_t_v = vertical_v ? vmx : vmy
                lp_v = vmin_long_v + (T(il_v) - half) * v_long_v
                mf_v = s_long_v / (s_long_v - lp_v)
                t0_v = s_tran_v +
                    (vmin_t_v + T(it_v - Int32(1)) * v_t_v - s_tran_v) * mf_v
                t1_v = s_tran_v +
                    (vmin_t_v + T(it_v) * v_t_v - s_tran_v) * mf_v
                z0_v = sz + (vmz + T(iz - Int32(1)) * vsz - sz) * mf_v
                z1_v = sz + (vmz + T(iz) * vsz - sz) * mf_v

                cmin = T(Inf); cmax = T(-Inf)
                rmin = T(Inf); rmax = T(-Inf)
                if arc_det
                    sdx = dcx - sx; sdy = dcy - sy
                    sd_xy = sqrt(sdx * sdx + sdy * sdy)
                    # Fan-angle extrema occur at the four transverse corners.
                    bx = Int32(-1)
                    while bx <= Int32(1)
                        by = Int32(-1)
                        while by <= Int32(1)
                            svx = xc + T(bx) * half * vsx - sx
                            svy = yc + T(by) * half * vsy - sy
                            along = (svx * sdx + svy * sdy) / sd_xy
                            across = svx * ux + svy * uy
                            c = atan(across, along) / dγ + cc
                            cmin = min(cmin, c); cmax = max(cmax, c)
                            by += Int32(2)
                        end
                        bx += Int32(2)
                    end

                    # Axial cylindrical projection depends on 1/rho. The
                    # closest xy point can lie on a box face, so corner-only
                    # bounds are not conservative here.
                    x0 = xc - half * vsx; x1 = xc + half * vsx
                    y0 = yc - half * vsy; y1 = yc + half * vsy
                    qx = sx - clamp(sx, x0, x1)
                    qy = sy - clamp(sy, y0, y1)
                    rho_min = sqrt(qx * qx + qy * qy)
                    rho_max_sq = zero(T)
                    bx = Int32(-1)
                    while bx <= Int32(1)
                        by = Int32(-1)
                        while by <= Int32(1)
                            qxc = xc + T(bx) * half * vsx - sx
                            qyc = yc + T(by) * half * vsy - sy
                            rho_max_sq = max(rho_max_sq, qxc * qxc + qyc * qyc)
                            by += Int32(2)
                        end
                        bx += Int32(2)
                    end
                    rho_min = max(rho_min, T(1.0e-12))
                    rho_max = sqrt(rho_max_sq)
                    bz = Int32(-1)
                    while bz <= Int32(1)
                        zeta = zc + T(bz) * half * vsz - sz
                        ra = (sz + sd_xy * zeta / rho_min - dcz) / prm + rc
                        rb = (sz + sd_xy * zeta / rho_max - dcz) / prm + rc
                        rmin = min(rmin, ra, rb); rmax = max(rmax, ra, rb)
                        bz += Int32(2)
                    end
                else
                    # Planar detector coordinates are linear-fractional over
                    # the voxel box; with the source outside the box their
                    # extrema occur at the eight corners.
                    bx = Int32(-1)
                    while bx <= Int32(1)
                        by = Int32(-1)
                        while by <= Int32(1)
                            bz = Int32(-1)
                            while bz <= Int32(1)
                                c, r = _dd_project_point(
                                    xc + T(bx) * half * vsx,
                                    yc + T(by) * half * vsy,
                                    zc + T(bz) * half * vsz,
                                    sx, sy, sz, dcx, dcy, dcz,
                                    ux, uy, uz, vx, vy, vz,
                                    pm, prm, cc, rc, false, dγ,
                                )
                                cmin = min(cmin, c); cmax = max(cmax, c)
                                rmin = min(rmin, r); rmax = max(rmax, r)
                                bz += Int32(2)
                            end
                            by += Int32(2)
                        end
                        bx += Int32(2)
                    end
                end

                # A detector cell centred at `i` overlaps the projected voxel
                # interval iff [i-1/2, i+1/2] intersects [min,max].  The
                # conservative point bounds above give the mathematical
                # integer limits. Retain one outward safety cell because the
                # arc row envelope and DD's column-averaged scale are evaluated
                # through different floating-point expressions; exact overlap
                # tests below discard every nonintersecting safety candidate.
                c0 = max(Int32(1), unsafe_trunc(Int32, ceil(cmin - half)) - Int32(1))
                c1 = min(nc, unsafe_trunc(Int32, floor(cmax + half)) + Int32(1))
                r0 = max(Int32(1), unsafe_trunc(Int32, ceil(rmin - half)) - Int32(1))
                r1 = min(nr, unsafe_trunc(Int32, floor(rmax + half)) + Int32(1))

                col = c0
                while col <= c1
                    valid_x, _, s_long_c, _, dXlo, dXhi, detXstep,
                        deltaT, scale_col, _, _, _, _, v_long_c,
                        _ = _dd_col_setup(
                        col, sx, sy, sz, dcx, dcy, ux, uy,
                        mag, ps, cc, nx, ny, vmx, vmy, vsx, vsy,
                        arc_det, dγ,
                    )
                    if valid_x
                        ox = _dd_overlap(
                            dXlo, dXhi, min(t0_v, t1_v), max(t0_v, t1_v))
                        if ox > zero(T)
                            row = r0
                            while row <= r1
                                valid_z, dZlo, dZhi, norm = _dd_row_setup(
                                    row, sz, dcz, vz, mag, prs, rc, scale_col,
                                    s_long_c, deltaT, v_long_c, detXstep,
                                )
                                if valid_z
                                    oz = _dd_overlap(
                                        dZlo, dZhi, min(z0_v, z1_v), max(z0_v, z1_v))
                                    if oz > zero(T)
                                        acc += sino[col, row, angle] * ox * oz * norm
                                    end
                                end
                                row += Int32(1)
                            end
                        end
                    end
                    col += Int32(1)
                end
                angle += Int32(1)
            end
            active_volume[idx] = acc
        end
    end
    return volume
end
