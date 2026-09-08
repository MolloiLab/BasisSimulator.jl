# =============================================================================
# BasisSimulator.Functional distance-driven projector — oracle tests.
#
# The legacy `dd_forward_project` / `dd_backproject!` kernels are the oracle.
# Contracts (same fixtures and tolerances as test/projection.jl:31-96 plus the
# Float32 / helical fixtures of :157-203 and :339-362):
#   (1) Float64: forward ≡ legacy to ≤1e-12 rel; ⟨Ax,y⟩=⟨x,Aᵀy⟩ at 2e-11;
#       brute-force Aᵀy at 2e-12; both :flat and :arc.
#   (2) Float32 water-cylinder parity (max abs ≤ 1e-4, mean rel ≤ 1e-5).
#   (3) helical arc geometry with volume_extent, parity ≤ 1e-4 abs (Float32).
#   (4) exhaustive proof that the static tap windows cover every non-zero
#       legacy overlap, forward (per cell/slab) and transpose (per voxel).
# =============================================================================

const BSF = BasisSimulator.Functional

_oracle_geom(shape) = begin
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 7, detector_cols = 24,
        detector_row_size = 0.8, detector_col_size = 0.8,
        detector_shape = shape)
    BS.CTGeometry(scanner; n_angles = 9, n_rows = 7, n_cols = 24, fov_cm = 8.0, z_cm = 1.4)
end

function _water_cylinder(nx, ny, nz; μ = 0.2f0, T = Float32)
    vol = zeros(T, nx, ny, nz)
    cx = (nx + 1) / 2; cy = (ny + 1) / 2; R = 0.6 * (nx / 2)
    for k in 1:nz, j in 1:ny, i in 1:nx
        (i - cx)^2 + (j - cy)^2 <= R^2 && (vol[i, j, k] = T(μ))
    end
    return vol
end

# --- exhaustive window-coverage helpers (legacy scalar kernels as ground truth)
function _legacy_forward_ranges(geom, vol_shape, view, vext, ::Type{T}) where {T}
    nx, ny, nz = Int32.(vol_shape)
    b = vext === nothing ? geom.fov : vext
    vmx = T(-b[1] / 2); vmy = T(-b[2] / 2); vmz = T(-b[3] / 2)
    vx = T(b[1]) / T(nx); vy = T(b[2]) / T(ny); vz = T(b[3]) / T(nz)
    mag = T(geom.SDD / geom.SAD); ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    cc = (T(geom.n_cols) + 1) / 2; rc = (T(geom.n_rows) + 1) / 2
    arc = BS.is_arc(geom); dγ = T(geom.pixel_size / geom.SAD)
    sp = geom.source_positions; dc = geom.detector_centers; du = geom.detector_u; dv = geom.detector_v
    sx, sy, sz = T(sp[1, view]), T(sp[2, view]), T(sp[3, view])
    dcx, dcy, dcz = T(dc[1, view]), T(dc[2, view]), T(dc[3, view])
    ux, uy, vvz = T(du[1, view]), T(du[2, view]), T(dv[3, view])
    n_long_max = max(nx, ny)
    it_lo = fill(typemax(Int32), geom.n_cols, geom.n_rows, n_long_max); it_hi = fill(typemin(Int32), size(it_lo))
    ip_lo = fill(typemax(Int32), size(it_lo)); ip_hi = fill(typemin(Int32), size(it_lo))
    n_long_v = 0; vertical_v = false
    for row in Int32(1):Int32(geom.n_rows), col in Int32(1):Int32(geom.n_cols)
        (valid, vertical, s_long, s_tran, dXlo, dXhi, dZlo, dZhi, norm,
            n_t, v_t, vmin_t, n_long, v_long, vmin_long) = BS._dd_cell_setup(
            col, row, sx, sy, sz, dcx, dcy, dcz, ux, uy, vvz, mag, ps, prs, cc, rc,
            nx, ny, vmx, vmy, vx, vy, arc, dγ)
        n_long_v = n_long; vertical_v = vertical
        valid || continue
        for il in Int32(1):n_long
            lp = vmin_long + (T(il) - T(0.5)) * v_long
            mf = s_long / (s_long - lp); inv_mf = one(T) / mf
            it_s, it_e = BS._dd_bounds(dXlo, dXhi, s_tran, vmin_t, v_t, inv_mf, n_t)
            ip_s, ip_e = BS._dd_bounds(dZlo, dZhi, sz, vmz, vz, inv_mf, nz)
            for it in it_s:it_e
                t0 = s_tran + (vmin_t + T(it - 1) * v_t - s_tran) * mf
                t1 = s_tran + (vmin_t + T(it) * v_t - s_tran) * mf
                BS._dd_overlap(dXlo, dXhi, min(t0, t1), max(t0, t1)) > 0 || continue
                it_lo[col, row, il] = min(it_lo[col, row, il], it); it_hi[col, row, il] = max(it_hi[col, row, il], it)
            end
            for ip in ip_s:ip_e
                z0 = sz + (vmz + T(ip - 1) * vz - sz) * mf
                z1 = sz + (vmz + T(ip) * vz - sz) * mf
                BS._dd_overlap(dZlo, dZhi, min(z0, z1), max(z0, z1)) > 0 || continue
                ip_lo[col, row, il] = min(ip_lo[col, row, il], ip); ip_hi[col, row, il] = max(ip_hi[col, row, il], ip)
            end
        end
    end
    return (it_lo, it_hi, ip_lo, ip_hi, n_long_v, vertical_v)
end

function _check_forward_windows(geom, vol_shape; vext = nothing, T = Float64)
    ok = true
    for view in 1:geom.n_angles
        p = BSF.dd_view_plan(geom, view, vol_shape; volume_extent = vext, eltype = T)
        g = BSF._forward_geometry(p, zeros(T, vol_shape))
        it_lo, it_hi, ip_lo, ip_hi, n_long, vertical = _legacy_forward_ranges(geom, vol_shape, view, vext, T)
        vertical == p.vertical || (ok = false)
        for il in 1:n_long, row in 1:geom.n_rows, col in 1:geom.n_cols
            if it_lo[col, row, il] <= it_hi[col, row, il]
                i0 = g.i0[col, 1, il]
                (i0 <= it_lo[col, row, il] && i0 + p.KX - 1 >= it_hi[col, row, il]) || (ok = false)
            end
            if ip_lo[col, row, il] <= ip_hi[col, row, il]
                k0 = g.k0[col, row, il]
                (k0 <= ip_lo[col, row, il] && k0 + p.KZ - 1 >= ip_hi[col, row, il]) || (ok = false)
            end
        end
    end
    return ok
end

function _check_transpose_windows(geom, vol_shape; vext = nothing, T = Float64)
    ok = true
    nx, ny, nz = Int32.(vol_shape)
    b = vext === nothing ? geom.fov : vext
    vmx = T(-b[1] / 2); vmy = T(-b[2] / 2); vmz = T(-b[3] / 2)
    vx = T(b[1]) / T(nx); vy = T(b[2]) / T(ny); vz = T(b[3]) / T(nz)
    mag = T(geom.SDD / geom.SAD); ps = T(geom.pixel_size); prs = T(geom.pixel_row_size)
    cc = (T(geom.n_cols) + 1) / 2; rc = (T(geom.n_rows) + 1) / 2
    arc = BS.is_arc(geom); dγ = T(geom.pixel_size / geom.SAD)
    for view in 1:geom.n_angles
        p = BSF.dd_view_plan(geom, view, vol_shape; volume_extent = vext, eltype = T)
        c = BSF._consts(p, T)
        sx, sy, sz = c.sx, c.sy, c.sz; dcx, dcy, dcz = c.dcx, c.dcy, c.dcz; ux, uy, vvz = c.ux, c.uy, c.vvz
        for il in 1:c.n_long, ip in 1:nz, it in 1:c.n_t
            lp = c.vmin_long + (T(il) - T(0.5)) * c.v_long; mf = c.s_long / (c.s_long - lp)
            t0 = c.s_tran + (c.vmin_t + T(it - 1) * c.v_t - c.s_tran) * mf
            t1 = c.s_tran + (c.vmin_t + T(it) * c.v_t - c.s_tran) * mf
            tvlo, tvhi = minmax(t0, t1)
            z0 = sz + (vmz + T(ip - 1) * vz - sz) * mf; z1 = sz + (vmz + T(ip) * vz - sz) * mf
            zvlo, zvhi = minmax(z0, z1)
            col0f, col0 = BSF._col_start(tvlo, tvhi, c)
            cmin = typemax(Int); cmax = typemin(Int)
            for col in Int32(1):Int32(geom.n_cols)
                valid_x, _, s_long_c, _, dXlo, dXhi, detXstep, deltaT, scale_col, _, _, _, _, v_long_c, _ =
                    BS._dd_col_setup(col, sx, sy, sz, dcx, dcy, ux, uy, mag, ps, cc, nx, ny, vmx, vmy, vx, vy, arc, dγ)
                valid_x || continue
                BS._dd_overlap(dXlo, dXhi, tvlo, tvhi) > 0 || continue
                rmin = typemax(Int); rmax = typemin(Int)
                for row in Int32(1):Int32(geom.n_rows)
                    valid_z, dZlo, dZhi, norm = BS._dd_row_setup(row, sz, dcz, vvz, mag, prs, rc, scale_col,
                        s_long_c, deltaT, v_long_c, detXstep)
                    (valid_z && BS._dd_overlap(dZlo, dZhi, zvlo, zvhi) > 0) || continue
                    rmin = min(rmin, Int(row)); rmax = max(rmax, Int(row))
                end
                rmin <= rmax || continue
                cmin = min(cmin, Int(col)); cmax = max(cmax, Int(col))
                # row window for THIS candidate column (what the transpose evaluates)
                (_, _, _, _, sc_col, _) = BSF._x_cells(T(col), c)
                row0f, row0 = BSF._row_start(zvlo, zvhi, sc_col, c)
                (row0 <= rmin && row0 + p.KZT - 1 >= rmax) || (ok = false)
            end
            cmin <= cmax || continue
            (col0 <= cmin && col0 + p.KXT - 1 >= cmax) || (ok = false)
        end
    end
    return ok
end

@testset "Functional.dd — Float64 oracle contracts (flat + arc)" begin
    for shape in (:flat, :arc)
        geom = _oracle_geom(shape)
        rng = MersenneTwister(0xDD3 + (shape === :arc))
        x = randn(rng, Float64, 9, 9, 5)
        y = randn(rng, Float64, geom.n_cols, geom.n_rows, geom.n_angles)

        Ax = BSF.dd_project(x, geom)
        Ax_legacy = BS.dd_forward_project(x, geom)
        @test size(Ax) == size(Ax_legacy)
        @test parity_report(Ax, Ax_legacy).max_rel <= 1.0e-12

        Aty = BSF.dd_transpose(y, geom, size(x))
        Aty_legacy = zeros(Float64, size(x)); BS.dd_backproject!(Aty_legacy, y, geom)
        @test parity_report(Aty, Aty_legacy).max_rel <= 1.0e-12

        dot = adjoint_dot_test(v -> BSF.dd_project(v, geom), s -> BSF.dd_transpose(s, geom, size(x)), x, y)
        @test dot.lhs ≈ dot.rhs rtol = 2.0e-11 atol = 2.0e-11

        # Brute-force operator oracle (identical to test/projection.jl:81-94).
        oracle_shape = (3, 3, 3)
        A = brute_force_matrix(v -> BSF.dd_project(v, geom), oracle_shape, size(y))
        A_legacy = brute_force_matrix(v -> BS.dd_forward_project(v, geom), oracle_shape, size(y))
        @test A == A_legacy                                    # unit-basis columns bit-identical
        oracle = reshape(transpose(A) * vec(y), oracle_shape)
        @test BSF.dd_transpose(y, geom, oracle_shape) ≈ oracle rtol = 2.0e-12 atol = 2.0e-12

        # Determinism: pure functions, repeated calls bit-identical.
        @test BSF.dd_project(x, geom) == Ax
        @test BSF.dd_transpose(y, geom, size(x)) == Aty

        @testset "static tap windows cover every legacy non-zero overlap ($shape)" begin
            @test _check_forward_windows(geom, size(x))
            @test _check_transpose_windows(geom, size(x))
        end
    end
end

@testset "Functional.dd — Float32 parity on the water cylinder" begin
    geom = _toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
    vol = _water_cylinder(64, 64, 8)
    sino_legacy = BS.dd_forward_project(vol, geom)
    sino = BSF.dd_project(vol, geom)
    @test eltype(sino) == Float32
    r = parity_report(sino, sino_legacy; floor = 1.0e-4)
    @test r.max_abs <= 1.0e-4
    @test r.mean_rel <= 1.0e-5
    @test r.n > 1000

    y = randn(MersenneTwister(7), Float32, size(sino)...)
    bp_legacy = zeros(Float32, size(vol)); BS.dd_backproject!(bp_legacy, y, geom)
    bp = BSF.dd_transpose(y, geom, size(vol))
    rb = parity_report(bp, bp_legacy; floor = 1.0e-4)
    @test rb.max_rel <= 1.0e-5
    @test rb.mean_rel <= 1.0e-5

    @test _check_forward_windows(geom, size(vol); T = Float32)
end

@testset "Functional.dd — helical arc geometry with volume_extent" begin
    scanner_h = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = :arc)
    geom_h = BS.CTGeometry(scanner_h; n_angles = 16, fov_cm = 10.0, z_cm = 1.0, pitch = 1.0, n_rotations = 2.0)
    @test BS.is_helical(geom_h) && BS.is_arc(geom_h)
    vext = (10.0, 10.0, 1.0)
    vol = _water_cylinder(64, 64, 8)[17:48, 17:48, 3:6]
    vol[10:12, 20:22, :] .= 1.0f0                     # off-centre rod

    sino_legacy = BS.dd_forward_project(vol, geom_h; volume_extent = vext)
    sino = BSF.dd_project(vol, geom_h; volume_extent = vext)
    r = parity_report(sino, sino_legacy; floor = 1.0e-4)
    @test r.max_abs <= 1.0e-4
    @test r.n > 500

    y = randn(MersenneTwister(11), Float32, size(sino)...)
    bp_legacy = zeros(Float32, size(vol)); BS.dd_backproject!(bp_legacy, y, geom_h; volume_extent = vext)
    bp = BSF.dd_transpose(y, geom_h, size(vol); volume_extent = vext)
    @test parity_report(bp, bp_legacy; floor = 1.0e-4).max_rel <= 1.0e-5

    x64 = Float64.(vol); y64 = Float64.(y)
    dot = adjoint_dot_test(v -> BSF.dd_project(v, geom_h; volume_extent = vext),
        s -> BSF.dd_transpose(s, geom_h, size(vol); volume_extent = vext), x64, y64)
    @test dot.rel <= 2.0e-11

    @testset "static tap windows (helical arc)" begin
        @test _check_forward_windows(geom_h, size(vol); vext = vext, T = Float32)
        @test _check_transpose_windows(geom_h, size(vol); vext = vext, T = Float32)
    end
end

@testset "Functional.dd — view batching (M5): batched ≡ per-view, adjoint, legacy" begin
    function _cyl(nx, ny, nz, ::Type{T}) where {T}
        vol = zeros(T, nx, ny, nz); cx = (nx + 1) / 2; cy = (ny + 1) / 2; R = 0.6 * (nx / 2)
        for k in 1:nz, j in 1:ny, i in 1:nx
            (i - cx)^2 + (j - cy)^2 <= R^2 && (vol[i, j, k] = T(0.2) + T(0.01) * k)
        end
        return vol
    end
    for shape in (:flat, :arc), T in (Float64, Float32), vb in (2, 4, 9)
        geom = _oracle_geom(shape); vol = _cyl(16, 16, 4, T)
        P1 = BSF.dd_project(vol, geom; eltype = T)
        Pb = BSF.dd_project(vol, geom; eltype = T, view_batch = vb)               # compiled-loop path (runs, batches, remainders)
        Pu = BSF.dd_project(vol, geom; eltype = T, view_batch = vb, loop = false) # unrolled batches
        @test size(Pb) == size(P1)
        @test Pb == P1                                        # forward: identical per-element arithmetic and tap order
        @test Pu == P1
        vol4 = cat(vol, 2 .* vol; dims = 4)                   # multi-channel run shares one index computation
        runs = BSF.dd_run_plans(geom, size(vol); view_batch = vb, eltype = T)
        @test vcat((collect(r.views) for r in runs)...) == collect(1:geom.n_angles)
        P4 = cat((BSF.dd_project_run(vol4, r) for r in runs)...; dims = 3)
        @test P4[:, :, :, 1] == P1 && P4[:, :, :, 2] == 2 .* P1
        y = rand(MersenneTwister(1), T, size(P1)...)
        B1 = BSF.dd_transpose(y, geom, size(vol); eltype = T)
        Bb = BSF.dd_transpose(y, geom, size(vol); eltype = T, view_batch = vb)
        @test maximum(abs.(Bb .- B1)) <= (T == Float64 ? 1e-12 : 2e-6) * maximum(abs.(B1))   # view reduction order
        @test abs(dot(Pb, y) - dot(vol, Bb)) <= (T == Float64 ? 1e-11 : 1e-5) * abs(dot(Pb, y))
        batches = BSF.dd_batch_plans(geom, size(vol); view_batch = vb, eltype = T)
        @test vcat((bp.views for bp in batches)...) == collect(1:geom.n_angles)
        @test all(length(bp.views) <= vb for bp in batches)
        @test all(allunique([bp.vertical for bp in batches][i:i+1]) for i in 1:length(batches)-1 if length(batches[i].views) < vb)
    end
    geom = _oracle_geom(:arc); vol = _cyl(16, 16, 4, Float64)
    Pl = BS.dd_forward_project(vol, geom)
    @test maximum(abs.(BSF.dd_project(vol, geom; view_batch = 4) .- Pl)) <= 1e-12 * maximum(abs.(Pl))
    @test_throws ArgumentError BSF.dd_batch_plans(geom, size(vol); view_batch = 0)
end
