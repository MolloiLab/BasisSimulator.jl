# Helical WFBP: the arc row mapping, the coverage output, and the guards around them. CPU only,
# and no forward simulation — the backprojector is driven with synthetic rebinned data whose row
# dependence is known, so what it samples can be read straight off the result.

function _helical_scanner(; shape = :arc, rows = 16, cols = 128)
    return BS.EICTScanner(
        source_to_isocenter = 541.0, source_to_detector = 949.0,
        detector_rows = rows, detector_cols = cols,
        detector_row_size = 0.625, detector_col_size = 1.0, detector_shape = shape,
    )
end

_helical_geom(; shape = :arc, pitch = 1.0, n_angles = 360, fov_cm = 20.0, z_cm = 4.0, rows = 16) =
    BS.CTGeometry(
    _helical_scanner(; shape, rows); n_angles = n_angles, fov_cm = fov_cm, z_cm = z_cm,
    pitch = pitch,
)

# Rebinned data that depends only on the detector row, so a backprojected voxel reports the row
# coordinate the mapping sent it to.
_row_ramp(geom) = Float32[Float32(r) for _ in 1:(geom.n_cols), r in 1:(geom.n_rows), _ in 1:(geom.n_angles)]

@testset "arc detectors get the cylindrical row mapping" begin
    arc = _helical_geom(shape = :arc)
    flat = _helical_geom(shape = :flat)
    @test BS.is_helical(arc) && BS.is_arc(arc) && !BS.is_arc(flat)
    @test arc.table_feed == flat.table_feed          # same trajectory, different panel

    n = 24
    size3 = (n, n, 6)
    v_arc = zeros(Float32, size3)
    v_flat = zeros(Float32, size3)
    BS._wfbp_backproject!(v_arc, _row_ramp(arc), arc, Float32(arc.pixel_size))
    BS._wfbp_backproject!(v_flat, _row_ramp(flat), flat, Float32(flat.pixel_size))

    # Near the axis of rotation the fan angle is ~0, where the cylindrical mapping (dz·SDD/d)
    # and the planar one (dz·SDD/cos γ/d) are the same expression; far from it they diverge as
    # 1/cos γ. So the mapping is what the difference measures, and it has to grow with radius.
    relative(i, j, k) = abs(v_arc[i, j, k] - v_flat[i, j, k]) / abs(v_arc[i, j, k])
    near = relative(n ÷ 2, n ÷ 2, 3)                 # 0.6 cm off axis
    far = relative(n ÷ 2 + 8, n ÷ 2 + 8, 3)          # 8.8 cm off axis
    @test near < 1.0e-4                              # float32 rounding, not geometry
    @test far > 100 * near
    @test all(isfinite, v_arc) && all(isfinite, v_flat)
end

@testset "the row mapping is the one the projector uses" begin
    # The near/far test above only shows the two branches agree on-axis and diverge off-axis; a
    # wrong formula (cos γ for 1/cos γ, say) passes it just the same. This pins the mapping: one
    # voxel is forward-projected through the package's own projector, and per view the row
    # centroid of its footprint is compared with BOTH candidate formulas. The error of the wrong
    # one is antisymmetric over the views — it cancels in the mean — so the discriminator is the
    # RMS: the right formula is within a tenth of a row, the wrong one off by most of a row at 20 cm.
    function row_errors(shape, r_cm)
        scanner = BS.EICTScanner(
            source_to_isocenter = 541.0, source_to_detector = 949.0,
            detector_rows = 32, detector_cols = 200, detector_row_size = 1.25,
            detector_col_size = 2.4, detector_shape = shape,
        )
        geom = BS.CTGeometry(
            scanner; n_angles = 90, fov_cm = 48.0, z_cm = 4.0, pitch = 1.0, n_rotations = 1
        )
        nx, nz = 64, 24
        vsx, vsz = geom.fov[1] / nx, geom.fov[3] / nz
        vol = zeros(Float32, nx, nx, nz)
        ix = round(Int, r_cm / vsx + nx / 2 + 0.5)
        iy, iz = nx ÷ 2 + 1, nz ÷ 2 + 4
        vol[ix, iy, iz] = 1.0f0
        x, y = (ix - 0.5 - nx / 2) * vsx, (iy - 0.5 - nx / 2) * vsx
        z = (iz - 0.5 - nz / 2) * vsz
        sino = BS.dd_forward_project(vol, geom; volume_extent = geom.fov)
        SDD, prm = geom.SDD, geom.pixel_row_size * geom.SDD / geom.SAD
        row_centre = (geom.n_rows + 1) / 2
        err_arc, err_flat = Float64[], Float64[]
        for j in 1:geom.n_angles
            s = geom.source_positions[:, j]
            c = geom.detector_centers[:, j]
            sv = (x - s[1], y - s[2])
            ctr = (c[1] - s[1], c[2] - s[2])
            D = hypot(sv...)
            cosγ = (sv[1] * ctr[1] + sv[2] * ctr[2]) / (D * hypot(ctr...))
            dz = z - s[3]
            slab = vec(sum(sino[:, :, j]; dims = 1))
            total = sum(slab)
            total > 1.0e-6 || continue
            measured = sum((1:geom.n_rows) .* slab) / total
            push!(err_arc, measured - (dz * SDD / D / prm + row_centre))
            push!(err_flat, measured - (dz * (SDD / cosγ) / D / prm + row_centre))
        end
        rms(v) = sqrt(mean(abs2, v))
        return (arc = rms(err_arc), flat = rms(err_flat), n = length(err_arc))
    end

    near_arc, far_arc = row_errors(:arc, 12.0), row_errors(:arc, 20.0)
    near_flat, far_flat = row_errors(:flat, 12.0), row_errors(:flat, 20.0)
    @test far_arc.n > 40 && far_flat.n > 40
    # the matching formula is right to a fraction of a row on either detector, at any radius
    @test near_arc.arc < 0.15 && far_arc.arc < 0.15
    @test near_flat.flat < 0.15 && far_flat.flat < 0.15
    # the other formula's error grows with radius (it is a 1/cos γ effect) …
    @test far_arc.flat > near_arc.flat
    @test far_flat.arc > near_flat.arc
    # … and at 20 cm it is several times the matching formula's, on both detectors
    @test far_arc.flat > 3 * far_arc.arc
    @test far_flat.arc > 3 * far_flat.flat
    # on the axis the two formulas coincide (cos γ = 1), which is why the earlier test cannot
    # tell them apart there
    axis = row_errors(:arc, 0.0)
    @test axis.arc < 0.15 && axis.flat < 0.15
end

@testset "coverage says which voxels the helix sampled" begin
    # A short helix: the requested volume is longer in z than the trajectory can fully cover, so
    # the middle is complete and the ends are not.
    geom = _helical_geom(pitch = 1.0, n_angles = 2 * 360, z_cm = 6.0)
    size3 = (12, 12, 24)
    volume = zeros(Float32, size3)
    coverage = zeros(Float32, size3)
    BS._wfbp_backproject!(volume, _row_ramp(geom), geom, Float32(geom.pixel_size); coverage)

    @test all(0 .<= coverage .<= 1)
    centre = coverage[6, 6, :]
    @test maximum(centre) ≈ 1.0f0                    # the middle of the helix is fully sampled
    @test centre[1] < 1 && centre[end] < 1           # both ends are not
    @test centre[1] < centre[argmax(centre)] && centre[end] < centre[argmax(centre)]
    # coverage is the fraction of half-turn families that found data, so it is quantised by them
    n_half = round(Int, π / (geom.angles[2] - geom.angles[1]))
    @test all(c -> isapprox(c * n_half, round(c * n_half); atol = 1.0e-4), coverage)
    # passing no array is the same reconstruction
    plain = zeros(Float32, size3)
    BS._wfbp_backproject!(plain, _row_ramp(geom), geom, Float32(geom.pixel_size))
    @test plain == volume

    # through the public entry point, on a rebinnable sinogram
    sinogram = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    fill!(sinogram, 0.01f0)
    cover2 = zeros(Float32, size3)
    vol2 = BS.wfbp_helical_reconstruct(sinogram, geom, size3; coverage = cover2)
    @test size(vol2) == size3 && any(<(1), cover2) && maximum(cover2) ≈ 1.0f0
    @test_throws DimensionMismatch BS.wfbp_helical_reconstruct(
        sinogram, geom, size3; coverage = zeros(Float32, 4, 4, 4)
    )
    # fdk_reconstruct routes a helical geometry here and forwards both knobs
    @test BS.fdk_reconstruct(sinogram, geom, size3; coverage = cover2) ≈ vol2
    @test BS.fdk_reconstruct(sinogram, geom, size3; helical_q = 1.0) != vol2
end

@testset "helical guards" begin
    # An odd number of views per rotation misaligns the conjugate families; it is warned about
    # rather than silently biased.
    odd = _helical_geom(n_angles = 361)
    volume = zeros(Float32, 8, 8, 4)
    @test_logs (:warn, r"views per rotation") BS._wfbp_backproject!(
        volume, _row_ramp(odd), odd, Float32(odd.pixel_size)
    )
    even = _helical_geom(n_angles = 360)
    @test_logs BS._wfbp_backproject!(
        zeros(Float32, 8, 8, 4), _row_ramp(even), even, Float32(even.pixel_size)
    )

    # the helical path masks outside the reconstruction circle like the axial path does
    sinogram = fill(0.01f0, even.n_cols, even.n_rows, even.n_angles)
    masked = BS.wfbp_helical_reconstruct(sinogram, even, (16, 16, 4))
    unmasked = BS.wfbp_helical_reconstruct(sinogram, even, (16, 16, 4); mask_fov = false)
    @test masked[1, 1, 1] != unmasked[1, 1, 1] && masked[1, 1, 1] ≈ -0.04f0
    @test masked[8, 8, :] ≈ unmasked[8, 8, :]

    # coverage is meaningless for a circular orbit and is refused rather than filled with ones
    axial = BS.CTGeometry(_helical_scanner(); n_angles = 180, fov_cm = 20.0, z_cm = 1.0)
    @test !BS.is_helical(axial)
    axial_sino = fill(0.01f0, axial.n_cols, axial.n_rows, axial.n_angles)
    @test_throws ArgumentError BS.fdk_reconstruct(
        axial_sino, axial, (8, 8, 4); coverage = zeros(Float32, 8, 8, 4)
    )
end
