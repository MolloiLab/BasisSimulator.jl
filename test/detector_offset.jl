# Quarter-detector offset (`detector_col_offset`, in columns).
#
# Every clinical scanner shifts its detector by a quarter column so that a ray at view θ and
# the ray at θ + π interleave instead of coinciding; that doubles the in-plane sampling and is
# the scanner's own anti-alias measure.  The offset is a scanner field that the geometry,
# every forward projector, the fan weighting and every backprojector must honour identically —
# a simulation with the offset applied on one side but not the other would put every ray in
# the wrong place.  These tests pin all of that.

using Test
using Statistics
using Random
import BasisSimulator as BS

function _offset_geom(offset; shape = :arc, n_cols = 96, n_angles = 48, fov_cm = 8.0)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 3, detector_cols = n_cols,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_col_offset = offset, detector_shape = shape,
    )
    return BS.CTGeometry(scanner; n_angles, fov_cm, z_cm = 0.3)
end

# an off-centre disk of μ = 1 on a fine grid, so the sinogram has sub-column structure
function _disk_volume(n; cx, cy, r)
    v = zeros(Float64, n, n, 3)
    for i in 1:n, j in 1:n
        x = (i - (n + 1) / 2); y = (j - (n + 1) / 2)
        hypot(x - cx, y - cy) <= r && (v[i, j, :] .= 1.0)
    end
    return v
end

@testset "quarter-detector offset" begin
    for shape in (:arc, :flat)
        g0 = _offset_geom(0.0; shape)
        gq = _offset_geom(0.25; shape)
        vol = _disk_volume(128; cx = 20.0, cy = -12.0, r = 9.0)

        @testset "$shape: the offset moves the rays by a quarter column" begin
            s0 = BS.dd_forward_project(vol, g0)
            sq = BS.dd_forward_project(vol, gq)
            @test s0 != sq
            # the central ray now crosses column (n + 1)/2 + ¼, so a fixed object's projection
            # moves by +¼ column: the centroid of each view's profile shifts by that amount
            centroid(p) = sum(p .* (1:length(p))) / sum(p)
            shifts = [centroid(sq[:, 2, k]) - centroid(s0[:, 2, k]) for k in 1:g0.n_angles]
            @test all(isapprox.(shifts, 0.25; atol = 0.03))
        end

        @testset "$shape: projector and backprojector agree on where the rays are" begin
            # reconstruct the offset sinogram with the offset geometry: the disk must land
            # where it is, to a fraction of a pixel; reconstructing it with the un-offset
            # geometry must put it in the wrong place by the quarter column's worth
            sq = BS.dd_forward_project(vol, gq)
            n = 128
            # the object's position: the centroid of what is above half its peak (the FBP ring
            # lobes around a small disk are positive too, and would pull a max(·, 0) centroid)
            function centroid_xy(img)
                w = img[:, :, 2] .> 0.5 * maximum(img[:, :, 2])
                xs = ((1:n) .- (n + 1) / 2) * (8.0 / n)
                (sum(w .* xs) / sum(w), sum(w .* xs') / sum(w))
            end
            ok = BS.fdk_reconstruct(sq, gq, (n, n, 3))
            bad = BS.fdk_reconstruct(sq, g0, (n, n, 3))
            px = 8.0 / n
            truth = (20.0 * px, -12.0 * px)
            cx_ok, cy_ok = centroid_xy(ok)
            cx_bad, cy_bad = centroid_xy(bad)
            @test isapprox(cx_ok, truth[1]; atol = 0.25px)
            @test isapprox(cy_ok, truth[2]; atol = 0.25px)
            # the mismatched pair is wrong in the way a one-sided offset is wrong: on a full
            # rotation opposite views push opposite ways, so the disk does not move, its edge
            # blurs by the quarter column (measured: 3.14 → 3.26 px, both shapes)
            function edge_width(img)
                rs = Float64[]; vs = Float64[]
                for i in 1:n, j in 1:n
                    r = hypot(i - (n + 1) / 2 - 20.0, j - (n + 1) / 2 + 12.0)
                    5 <= r <= 13 && (push!(rs, r); push!(vs, img[i, j, 2]))
                end
                bins = 5:0.5:13
                prof = [mean(vs[(rs .>= b) .& (rs .< b + 0.5)]) for b in bins[1:(end - 1)]]
                hi = mean(prof[1:4]); lo = mean(prof[(end - 3):end]); q = (prof .- lo) ./ (hi - lo)
                cross(v) = (k = findfirst(<(v), q); bins[k - 1] + (q[k - 1] - v) / (q[k - 1] - q[k]) * 0.5)
                return cross(0.1) - cross(0.9)
            end
            @test isapprox(cx_bad, truth[1]; atol = 0.25px)      # not shifted
            @test edge_width(bad) > 1.02 * edge_width(ok)        # blurred
        end

        @testset "$shape: opposing rays interleave" begin
            # at view θ + π the offset ray positions fall half-way between the θ ones: the
            # union of the two views' column positions, seen from the isocentre, has a spacing
            # of half a column.  Without the offset the two views coincide.
            for (g, expect_half) in ((gq, true), (g0, false))
                k1 = 1; k2 = g.n_angles ÷ 2 + 1   # θ and θ + π
                # column positions along u for a ray through isocentre, in column units
                u(k, c) = (c - (g.n_cols + 1) / 2 - BS.column_offset(g)) * (k == k1 ? 1 : -1)
                pos = sort(vcat([u(k1, c) for c in 1:g.n_cols], [u(k2, c) for c in 1:g.n_cols]))
                gaps = diff(pos)
                interleaved = all(isapprox.(gaps, 0.5; atol = 1e-9))
                @test interleaved == expect_half
            end
        end
    end
end

# The labels of a simulated scan are placed by `resample_field_to_recon` through the geometry's
# own voxel-to-world affine, never by the projector.  A geometry change that moves where the
# reconstruction's pixels sit — a detector offset, a halo, a matrix — must move the labels
# identically, or a label and its image drift apart by a fraction of a pixel and nothing
# downstream notices.  This pins that the reconstruction of a fine object and the object's own
# label on the same grid agree in position, with and without the offset.
@testset "labels and reconstruction agree in position" begin
    for offset in (0.0, 0.25), shape in (:arc, :flat)
        g = _offset_geom(offset; shape, n_cols = 128, n_angles = 96, fov_cm = 8.0)
        nfine = 256
        fine = _disk_volume(nfine; cx = 44.0, cy = -30.0, r = 12.0)   # a 0.75 mm-pixel object
        dx = 8.0 / nfine
        origin = (-4.0 + dx / 2, -4.0 + dx / 2, -0.15 + 0.1 / 2)
        n = 128
        # the object's label on the reconstruction grid: exact box average through the affine
        label = BS.resample_field_to_recon(fine, (dx, dx, 0.1), origin, g, (n, n, 3))
        # the object's image: project and reconstruct through the same geometry
        image = BS.fdk_reconstruct(BS.dd_forward_project(label, g), g, (n, n, 3))
        function centroid_xy(img)
            w = img[:, :, 2] .> 0.5 * maximum(img[:, :, 2])
            xs = ((1:n) .- (n + 1) / 2) * (8.0 / n)
            (sum(w .* xs) / sum(w), sum(w .* xs') / sum(w))
        end
        cl = centroid_xy(label); ci = centroid_xy(image)
        px = 8.0 / n
        @test isapprox(cl[1], ci[1]; atol = 0.1px)
        @test isapprox(cl[2], ci[2]; atol = 0.1px)
        # and the label sits where the object is, independent of the detector offset
        @test isapprox(cl[1], 44.0 * dx; atol = 0.05px)
        @test isapprox(cl[2], -30.0 * dx; atol = 0.05px)
    end
end

# The offset through every projector path: the exact adjoint pair (distance-driven forward and
# transpose) must stay adjoint at any offset on both detector shapes, the row-tiled fast path
# must equal the plain one, and the Siddon projector must put a point object in the column the
# geometry predicts.  (From the round-1 adversarial review's experiment.)
using LinearAlgebra
@testset "offset through every projector path" begin
    function analytic_col(g, p, k)
        s = g.source_positions[:, k]; d = g.detector_centers[:, k]; u = g.detector_u[:, k]
        ray = p .- s; cen = d .- s
        if BS.is_arc(g)
            γ = atan(dot(ray, u), dot(ray, cen) / norm(cen))
            return BS.column_center(Float64, g) + γ / (g.pixel_size / g.SAD)
        else
            nrm = cen / norm(cen); t = dot(d .- s, nrm) / dot(ray, nrm)
            hit = s .+ t .* ray
            return BS.column_center(Float64, g) + dot(hit .- d, u) / (g.pixel_size * g.SDD / g.SAD)
        end
    end
    for shape in (:arc, :flat), offset in (0.0, 0.25, -0.25, 3.7)
        g = _offset_geom(offset; shape, n_cols = 64, n_angles = 36, fov_cm = 8.0)
        rng = MersenneTwister(11)
        x = rand(rng, 48, 48, 3); y = rand(rng, g.n_cols, g.n_rows, g.n_angles)
        Ax = zeros(g.n_cols, g.n_rows, g.n_angles); BS.dd_forward_project!(Ax, x, g)
        Aty = zeros(48, 48, 3); BS.dd_backproject!(Aty, y, g)
        @test abs(dot(Ax, y) - dot(x, Aty)) / abs(dot(Ax, y)) < 1e-12
        if shape === :arc
            Ax2 = zeros(size(Ax)); BS._dd_forward_project_arc_rowtile4!(Ax2, x, g)
            @test maximum(abs, Ax2 .- Ax) < 1e-12
        end
        # a point object lands in the predicted column for both projectors
        gp = _offset_geom(offset; shape, n_cols = 96, n_angles = 24, fov_cm = 8.0)
        nx = 64; px = 8.0 / nx; xp = zeros(nx, nx, 3); i0, j0 = 50, 21; xp[i0, j0, :] .= 1.0
        p = [(i0 - (nx + 1) / 2) * px, (j0 - (nx + 1) / 2) * px, 0.0]
        for f! in (BS.dd_forward_project!, BS.siddon_forward_project!)
            s = zeros(gp.n_cols, gp.n_rows, gp.n_angles); f!(s, xp, gp)
            errs = Float64[]
            for k in 1:gp.n_angles
                prof = s[:, 2, k]; sum(prof) > 0 || continue
                push!(errs, sum(prof .* (1:gp.n_cols)) / sum(prof) - analytic_col(gp, p, k))
            end
            @test abs(mean(errs)) < 0.03
            @test maximum(abs, errs) < 0.4          # one voxel's footprint spans a column or so
        end
    end
end
