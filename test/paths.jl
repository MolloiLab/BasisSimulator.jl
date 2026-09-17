# Cached per-material path lengths: the two-pass polychromatic projection has to agree with the
# single-pass kernel exactly, because the whole point is that a cached walk is not an
# approximation of walking again. CPU only.

function _paths_fixture(; n_energies = 12, n_materials = 4, n_angles = 24)
    scanner = BS.EICTScanner(
        source_to_isocenter = 541.0, source_to_detector = 949.0,
        detector_rows = 6, detector_cols = 48,
        detector_row_size = 1.0, detector_col_size = 1.0, detector_shape = :arc,
    )
    geom = BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = 20.0, z_cm = 2.0)
    # a blocky material map: every material present, none of them symmetric
    nx = ny = 32
    nz = 8
    mask = zeros(UInt8, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        r = hypot(i - 16.5, j - 16.5)
        mask[i, j, k] = r < 12 ? UInt8(1 + (i + 2j + 3k) % n_materials) : UInt8(0)
    end
    energies = collect(range(30.0, 120.0; length = n_energies))
    μ_table = Float32[0.02 * m * (100.0 / e) for m in 1:(n_materials + 1), e in energies]
    μ_table[1, :] .= 0.0f0                                   # label 0 is air
    wη = Float32.(exp.(-((energies .- 70.0) ./ 30.0) .^ 2))
    wη ./= sum(wη)
    return (; geom, mask, μ_table, wη, n_materials = n_materials + 1, extent = (20.0, 20.0, 2.0))
end

function _single_pass(f)
    sinogram = zeros(Float32, f.geom.n_cols, f.geom.n_rows, f.geom.n_angles)
    BS.dd_fast_fused_poly_project!(
        sinogram, f.mask, f.geom, f.μ_table, f.wη, Val(length(f.wη));
        volume_extent = f.extent,
    )
    return sinogram
end

function _two_pass(f; bowtie = nothing)
    paths = zeros(Float32, f.n_materials, f.geom.n_cols, f.geom.n_rows, f.geom.n_angles)
    BS.dd_fast_material_paths!(paths, f.mask, f.geom; volume_extent = f.extent)
    sinogram = zeros(Float32, f.geom.n_cols, f.geom.n_rows, f.geom.n_angles)
    BS.dd_fast_poly_from_paths!(
        sinogram, paths, f.μ_table, f.wη; ws_bowtie_spectral = bowtie
    )
    return sinogram, paths
end

@testset "two-pass projection is bit-identical to one pass" begin
    f = _paths_fixture()
    reference = _single_pass(f)
    cached, paths = _two_pass(f)
    @test cached == reference                              # not approximately: exactly
    @test any(>(0), reference)                             # the fixture actually attenuates

    # the same cache serves every spectrum, which is the reason it exists
    for scale in (0.5f0, 2.0f0)                            # a harder and a softer beam
        g = (; f..., μ_table = f.μ_table .* scale)
        ref = _single_pass(g)
        out = zeros(Float32, size(ref))
        BS.dd_fast_poly_from_paths!(out, paths, g.μ_table, g.wη)
        @test out == ref
    end

    # with a bowtie, per detector element and energy
    bowtie = Float32[0.4 + 0.6 * exp(-((c - 24.0) / 12.0)^2) for c in 1:(f.geom.n_cols),
        _ in 1:(f.geom.n_rows), _ in 1:length(f.wη)]
    with_bt = zeros(Float32, size(reference))
    BS.dd_fast_poly_from_paths!(with_bt, paths, f.μ_table, f.wη; ws_bowtie_spectral = bowtie)
    ref_bt = zeros(Float32, size(reference))
    BS.dd_fast_fused_poly_project!(
        ref_bt, f.mask, f.geom, f.μ_table, f.wη, Val(length(f.wη));
        volume_extent = f.extent, ws_bowtie_spectral = bowtie,
    )
    @test with_bt == ref_bt
    @test with_bt != reference
end

@testset "path lengths are path lengths" begin
    f = _paths_fixture()
    _, paths = _two_pass(f)
    @test all(>=(0), paths)
    # A ray's total path is the sum over materials, and it is the length of its chord through
    # the whole reconstructed box — air inside the box included, which is material 1 here. So
    # it is bounded below by the box thickness and above by its diagonal.
    total = dropdims(sum(paths; dims = 1); dims = 1)
    @test maximum(total) <= sqrt(20.0^2 + 20.0^2 + 2.0^2)
    @test minimum(total) > 0
    # The object is a cylinder of radius 12 voxels of 32 across a 20 cm box, so a chord through
    # it is at most its diameter measured to the outer edge of the last voxel the footprint
    # touches, 2·12.5/32·20 cm, and a central ray crosses more than the radius.
    object = dropdims(sum(paths[2:end, :, :, :]; dims = 1); dims = 1)
    @test maximum(object) <= 2 * (12.5 / 32) * 20.0
    @test maximum(object) > (12 / 32) * 20.0
    # this detector subtends only ~27 mm at isocentre, so every ray crosses the object
    @test minimum(object) > 0
    # and the line integral the conversion computes is the one the path lengths imply
    e = 5
    μ = f.μ_table[:, e]
    single_energy = zeros(Float32, size(total))
    BS.dd_fast_poly_from_paths!(
        single_energy, paths, reshape(f.μ_table[:, e], :, 1), ones(Float32, 1)
    )
    by_hand = [
        sum(μ[m] * paths[m, i, j, k] for m in axes(paths, 1))
            for i in axes(total, 1), j in axes(total, 2), k in axes(total, 3)
    ]
    @test single_energy ≈ by_hand rtol = 1.0e-5
end

@testset "the cache refuses to be misused" begin
    f = _paths_fixture()
    sinogram = zeros(Float32, f.geom.n_cols, f.geom.n_rows, f.geom.n_angles)
    good = zeros(Float32, f.n_materials, f.geom.n_cols, f.geom.n_rows, f.geom.n_angles)

    # the view count has to be the geometry's
    @test_throws DimensionMismatch BS.dd_fast_material_paths!(
        zeros(Float32, f.n_materials, f.geom.n_cols, f.geom.n_rows, 3), f.mask, f.geom;
        volume_extent = f.extent,
    )
    # more materials than the path-length kernel can hold in registers
    @test_throws ArgumentError BS.dd_fast_material_paths!(
        zeros(Float32, 65, f.geom.n_cols, f.geom.n_rows, f.geom.n_angles), f.mask, f.geom;
        volume_extent = f.extent,
    )
    # a cache and an attenuation table that disagree about how many materials there are
    @test_throws DimensionMismatch BS.dd_fast_poly_from_paths!(
        sinogram, good, f.μ_table[1:2, :], f.wη
    )
    # a cache of the wrong detector shape
    @test_throws DimensionMismatch BS.dd_fast_poly_from_paths!(
        sinogram, zeros(Float32, f.n_materials, 4, 4, 4), f.μ_table, f.wη
    )
end
