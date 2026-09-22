# HIR's internal support domain.
#
# The iterative model must represent everything the detector sees.  Along z that is the cone
# beyond the requested slab (the axial halo).  In plane it is the part of the patient outside the
# requested reconstruction circle: a chest is wider than a 35 cm field of view, the rays through
# the shoulders are measured, and a model with nowhere to put that attenuation piles it onto the
# circle's edge — a bright smear in the outermost centimetre that FBP does not have.  This pins
# that HIR on a body wider than the field of view matches FBP at the edge and in the interior,
# and that a body inside the circle is untouched by the extension.

using Test
using Statistics
import BasisSimulator as BS

_hir(sino, g, size3) = BS.reconstruct!(BS.create_hir_recon_workspace(sino, g, size3; strength = 60), sino, g)

function _support_scanner(; n_cols = 400)
    return BS.EICTScanner(
        source_to_isocenter = 600.0, source_to_detector = 1100.0,
        detector_rows = 4, detector_cols = n_cols,
        detector_row_size = 1.0, detector_col_size = 1.25,   # 50 cm fan at isocentre
        detector_col_offset = 0.0, detector_shape = :arc,
    )
end

# an ellipse of μ, axes a × b (cm), sampled on a grid of `n` pixels over `fov` cm
function _ellipse(n, fov; a, b, μ = 0.2)
    v = zeros(Float64, n, n, 4)
    px = fov / n
    for i in 1:n, j in 1:n
        x = (i - (n + 1) / 2) * px; y = (j - (n + 1) / 2) * px
        (x / a)^2 + (y / b)^2 <= 1 && (v[i, j, :] .= μ)
    end
    return v
end

@testset "HIR support beyond the reconstruction circle" begin
    scanner = _support_scanner(; n_cols = 320)                # 40 cm fan at isocentre
    n = 128; fov = 16.0; px = fov / n                         # 1.25 mm pixels, a 16 cm request
    g_full = BS.CTGeometry(scanner; n_angles = 120, fov_cm = 40.0, z_cm = 0.4)
    g = BS.CTGeometry(scanner; n_angles = 120, fov_cm = fov, z_cm = 0.4)
    xs = ((1:n) .- (n + 1) / 2) * px
    r = [hypot(xs[i], xs[j]) for i in 1:n, j in 1:n]
    hu(v, m) = mean(v[:, :, 2][m]) / 0.2 * 1000 - 1000
    # the complete-model reference: HIR asked for a grid that contains the whole body, read
    # back on the 16 cm window (what a scanner's own IR does — reconstruct the scan field,
    # display a part of it)
    function reference(sino, nwide)
        gw = BS.CTGeometry(scanner; n_angles = 120, fov_cm = nwide * px, z_cm = 0.4)
        w = _hir(sino, gw, (nwide, nwide, 4)); o = (nwide - n) ÷ 2
        return w[(o + 1):(o + n), (o + 1):(o + n), :]
    end

    @testset "a body wider than the circle" begin
        wide = _ellipse(2n, 32.0; a = 14.0, b = 6.0)         # 28 × 12 cm: past the 16 cm circle
        sino = BS.dd_forward_project(wide, g_full)
        fbp = BS.fdk_reconstruct(sino, g, (n, n, 4))
        hir = _hir(sino, g, (n, n, 4))
        ref = reference(sino, 256)                          # 32 cm contains the body
        inside = [(xs[i] / 14.0)^2 + (xs[j] / 6.0)^2 <= 0.9 for i in 1:n, j in 1:n]
        edge = inside .& (r .> 6.8) .& (r .< 7.9)
        interior = inside .& (r .< 5.0)
        # HIR must not invent attenuation at the circle's edge …
        @test abs(hu(hir, edge) - hu(fbp, edge)) < 10
        # … and must give the same image as the complete model everywhere
        @test abs(hu(hir, edge) - hu(ref, edge)) < 2
        @test abs(hu(hir, interior) - hu(ref, interior)) < 2
    end

    @testset "a body inside the circle gets the same answer" begin
        small = _ellipse(2n, 32.0; a = 6.0, b = 5.0)          # fits in the 16 cm circle
        sino = BS.dd_forward_project(small, g_full)
        hir = _hir(sino, g, (n, n, 4))
        ref = reference(sino, 256)
        inside = [(xs[i] / 6.0)^2 + (xs[j] / 5.0)^2 <= 0.9 for i in 1:n, j in 1:n]
        @test abs(hu(hir, inside) - hu(ref, inside)) < 2
        # pointwise too, inside the requested circle (outside it the request is masked)
        within = r .< 0.98 * fov / 2
        @test maximum(abs, (hir[:, :, 2] .- ref[:, :, 2])[within]) / 0.2 * 1000 < 25
    end

    @testset "the support is the scan circle, no larger" begin
        ws = BS.create_hir_recon_workspace(zeros(320, 4, 120), g, (n, n, 4); strength = 60)
        d = BS.scan_circle_diameter(g)
        # the circle tangent to the extreme rays of a 40 cm (arc-length) fan: 2·SAD·sin(γ_max)
        @test isapprox(d, 2 * 60.0 * sin(20.0 / 60.0); atol = 1e-9)
        @test size(ws.work_volume, 1) * px >= d
        @test size(ws.work_volume, 1) * px < d + 2px
        @test ws.output_x == ws.output_y
        @test length(ws.output_x) == n
        # strength 0 is plain FBP and needs no support domain
        ws0 = BS.create_hir_recon_workspace(zeros(320, 4, 120), g, (n, n, 4); strength = 0)
        @test size(ws0.work_volume) == (n, n, 4)
    end
end
