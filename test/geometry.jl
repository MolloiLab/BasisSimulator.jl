# Tests for src/geometry/ — Scanner + CTGeometry + affine transforms.
#
# Coverage policy: every exported symbol from src/geometry/ is exercised:
#   scanner.jl  : Scanner, CTGeometry, _build_pcct_detector, _infer_pcct_material.
#   affine.jl   : phantom_to_world_affine, recon_to_world_affine, resample_to_recon.

# -----------------------------------------------------------------------------
# Scanner struct + ctor
# -----------------------------------------------------------------------------
@testset "Scanner — defaults + kwarg propagation" begin
    s = BS.Scanner()  # all defaults
    # CatSim-style defaults.
    @test s.source_to_isocenter == 540.0
    @test s.source_to_detector == 950.0
    @test s.detector_rows == 64
    @test s.detector_cols == 900
    @test s.detector_type === :energy_integrating
    @test s.n_energy_bins == 1
    @test isempty(s.energy_thresholds)
    @test s.binning_factor == 1
    # native dexel auto-derived = pixel * (SDD/SAD) / binning.
    mag = s.source_to_detector / s.source_to_isocenter
    @test s.native_dexel_col_mm ≈ s.detector_col_size * mag / s.binning_factor
    @test s.native_dexel_row_mm ≈ s.detector_row_size * mag / s.binning_factor
end

@testset "Scanner — flat-panel-style geometry kwargs" begin
    s = BS.Scanner(
        detector_rows = 512, detector_cols = 512,
        detector_row_size = 0.15, detector_col_size = 0.15,
    )
    @test s.detector_rows == 512
    @test s.detector_cols == 512
end

@testset "Scanner — explicit native_dexel override wins over inference" begin
    s = BS.Scanner(
        detector_col_size = 1.0, detector_row_size = 1.0,
        native_dexel_col_mm = 0.4, native_dexel_row_mm = 0.4,
    )
    @test s.native_dexel_col_mm == 0.4
    @test s.native_dexel_row_mm == 0.4
end

# -----------------------------------------------------------------------------
# Scanner — PCCT validation paths
# -----------------------------------------------------------------------------
@testset "Scanner — PCCT validation" begin
    pcct_kwargs = (;
        detector_type = :photon_counting,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 50.0, 80.0, 110.0],
        energy_resolution = 5.0,
        charge_sharing_fwhm = 0.05,
        dead_time_ns = 20.0,
        pixel_mode = :standard,
    )

    @testset "valid PCCT scanner constructs" begin
        s = BS.Scanner(; pcct_kwargs...)
        @test s.detector_type === :photon_counting
        @test s.energy_thresholds == [20.0, 50.0, 80.0, 110.0]
        @test s.energy_resolution == 5.0
    end

    @testset "empty energy_thresholds errors" begin
        @test_throws ErrorException BS.Scanner(;
            pcct_kwargs..., energy_thresholds = Float64[], n_energy_bins = 0,
        )
    end

    @testset "n_energy_bins ≠ length(energy_thresholds) errors" begin
        @test_throws ErrorException BS.Scanner(;
            pcct_kwargs..., n_energy_bins = 3,  # but 4 thresholds supplied
        )
    end

    @testset "unsorted energy_thresholds errors" begin
        @test_throws ErrorException BS.Scanner(;
            pcct_kwargs..., energy_thresholds = [50.0, 20.0, 80.0, 110.0],
        )
    end

    @testset "unknown pixel_mode errors" begin
        @test_throws ErrorException BS.Scanner(; pcct_kwargs..., pixel_mode = :nonsense)
    end

    @testset "unknown detector_type errors" begin
        @test_throws ErrorException BS.Scanner(; detector_type = :nonsense)
    end

    @testset "binning_factor < 1 errors" begin
        @test_throws ErrorException BS.Scanner(; binning_factor = 0)
        @test_throws ErrorException BS.Scanner(; binning_factor = -1)
    end
end

# -----------------------------------------------------------------------------
# CTGeometry ctor — derivations + shape contracts.
# -----------------------------------------------------------------------------
@testset "CTGeometry — defaults + overrides" begin
    s = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = 32, detector_cols = 256,
        detector_row_size = 1.0, detector_col_size = 1.0,
    )

    @testset "defaults: full coverage, n_rows/n_cols from scanner" begin
        g = BS.CTGeometry(s; n_angles = 16)
        @test g.SAD == 54.0
        @test g.SDD == 108.0
        @test g.n_angles == 16
        @test g.n_rows == 32
        @test g.n_cols == 256
        @test g.pixel_size ≈ 0.1
        @test g.pixel_row_size ≈ 0.1
        # FOV = n_cols × pixel_size at iso.
        @test g.fov[1] ≈ 0.1 * 256
        @test g.fov[2] ≈ 0.1 * 256
        @test g.fov[3] ≈ 0.1 * 32   # z = n_rows × pixel_row
    end

    @testset "explicit fov_cm / z_cm override" begin
        g = BS.CTGeometry(s; n_angles = 8, fov_cm = 35.0, z_cm = 5.0)
        @test g.fov[1] == 35.0
        @test g.fov[2] == 35.0
        @test g.fov[3] == 5.0
    end

    @testset "n_rows / n_cols override scanner values" begin
        g = BS.CTGeometry(s; n_angles = 8, n_rows = 8, n_cols = 128)
        @test g.n_rows == 8
        @test g.n_cols == 128
    end

    @testset "collimation_mm → n_rows derivation" begin
        # 32 rows × 1.0 mm = 32 mm max; ask for 16 mm = 16 rows.
        g = BS.CTGeometry(s; n_angles = 8, collimation_mm = 16.0)
        @test g.n_rows == 16
    end

    @testset "axial cone guards cover the full reconstruction cylinder" begin
        pcct = BS.Scanner(
            source_to_isocenter = 610.0, source_to_detector = 1113.0,
            detector_rows = 144, detector_cols = 1195,
            detector_row_size = 0.3529559748427673,
            detector_col_size = 0.3015274034141959,
        )
        @test BS.required_axial_detector_rows(pcct; fov_cm = 35.0, z_cm = 0.5) == 20
        g = BS.CTGeometry(pcct; n_angles = 8, fov_cm = 35.0, z_cm = 0.5,
                          collimation_mm = 5.0)
        @test g.n_rows == 20
        @test g.fov[3] == 0.5 # guard acquisition never changes the saved z grid
    end

    @testset "collimation_mm vs n_rows mutual exclusion" begin
        @test_throws ErrorException BS.CTGeometry(
            s;
            n_angles = 8, collimation_mm = 16.0, n_rows = 16,
        )
    end

    @testset "collimation_mm exceeds scanner physical max errors" begin
        @test_throws ErrorException BS.CTGeometry(
            s;
            n_angles = 8, collimation_mm = 999.0,
        )
    end

    @testset "extended_collimation bypasses cap with @warn" begin
        # 32 rows × 1.0 mm = 32 mm physical max; ask for 80 mm extended.
        local g
        @test_logs (:warn, r"EXTENDED-COLLIMATION MODE"i) match_mode=:any begin
            g = BS.CTGeometry(s; n_angles = 8, collimation_mm = 80.0,
                              extended_collimation = true)
        end
        @test g.n_rows == 80   # round(80 / 1.0)
        # Mutual-exclusion still holds in extended mode.
        @test_throws ErrorException BS.CTGeometry(
            s;
            n_angles = 8, collimation_mm = 80.0, n_rows = 80,
            extended_collimation = true,
        )
        # Default (extended=false) still errors.
        @test_throws ErrorException BS.CTGeometry(
            s;
            n_angles = 8, collimation_mm = 80.0,
        )
    end

    @testset "angles cover [0, 2π) uniformly" begin
        n = 16
        g = BS.CTGeometry(s; n_angles = n)
        @test length(g.angles) == n
        @test g.angles[1] == 0.0
        @test g.angles[end] ≈ 2π - 2π / n
        # Uniform spacing.
        Δ = diff(g.angles)
        @test maximum(abs.(Δ .- 2π / n)) < 1.0e-12
    end

    @testset "source positions on a circle of radius SAD" begin
        g = BS.CTGeometry(s; n_angles = 16)
        for i in 1:g.n_angles
            r = sqrt(g.source_positions[1, i]^2 + g.source_positions[2, i]^2)
            @test r ≈ g.SAD  atol = 1.0e-10
            @test g.source_positions[3, i] == 0.0
        end
    end

    @testset "detector center on a circle of radius SDD - SAD, opposite source" begin
        g = BS.CTGeometry(s; n_angles = 16)
        det_dist = g.SDD - g.SAD
        for i in 1:g.n_angles
            r = sqrt(g.detector_centers[1, i]^2 + g.detector_centers[2, i]^2)
            @test r ≈ det_dist  atol = 1.0e-10
            # Opposite source: detector x = -det_dist/SAD · source x.
            @test g.detector_centers[1, i] ≈ -det_dist / g.SAD * g.source_positions[1, i]
            @test g.detector_centers[2, i] ≈ -det_dist / g.SAD * g.source_positions[2, i]
        end
    end

    @testset "u-axis ⊥ v-axis, v = +ẑ" begin
        g = BS.CTGeometry(s; n_angles = 8)
        for i in 1:g.n_angles
            u = g.detector_u[:, i]
            v = g.detector_v[:, i]
            @test v == [0.0, 0.0, 1.0]
            @test abs(u[1]^2 + u[2]^2 + u[3]^2 - 1) < 1.0e-12   # |u| = 1
            @test abs(sum(u .* v)) < 1.0e-12                     # u ⊥ v
        end
    end
end

# -----------------------------------------------------------------------------
# _build_pcct_detector + _infer_pcct_material
# -----------------------------------------------------------------------------
@testset "_build_pcct_detector" begin
    pcct_kwargs = (;
        detector_type = :photon_counting,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 50.0, 80.0, 110.0],
        energy_resolution = 5.0,
        charge_sharing_fwhm = 0.05,
        dead_time_ns = 20.0,
        pixel_mode = :standard,
        binning_factor = 2,
    )

    @testset "EID scanner triggers @assert" begin
        eid = BS.Scanner()  # default :energy_integrating
        @test_throws AssertionError BS._build_pcct_detector(eid)
    end

    @testset "fields propagate to PhotonCountingDetector" begin
        s = BS.Scanner(; detector_material = :cdte, pcct_kwargs...)
        d = BS._build_pcct_detector(s)
        @test d.thickness_mm == s.detector_depth
        @test d.energy_thresholds_keV == s.energy_thresholds
        @test d.energy_resolution_keV == s.energy_resolution
        @test d.charge_sharing_fwhm_mm == s.charge_sharing_fwhm
        @test d.dead_time_ns == s.dead_time_ns
        @test d.binning_factor == s.binning_factor
        # PCCT eliminates electronic noise via thresholding.
        @test d.electronic_noise_keV == 0.0
        # Pile-up + charge-sharing toggles derived from non-zero values.
        @test d.enable_pile_up == true
        @test d.enable_charge_sharing == true
    end

    @testset "zero charge_sharing_fwhm / dead_time_ns ⇒ effects disabled" begin
        s = BS.Scanner(;
            detector_material = :cdte,
            pcct_kwargs...,
            charge_sharing_fwhm = 0.0,
            dead_time_ns = 0.0,
        )
        d = BS._build_pcct_detector(s)
        @test d.enable_pile_up == false
        @test d.enable_charge_sharing == false
    end
end

@testset "_infer_pcct_material — every advertised alias" begin
    @test BS._infer_pcct_material(:cdte) === BS.CDTE_MATERIAL
    @test BS._infer_pcct_material(:CdTe) === BS.CDTE_MATERIAL
    @test BS._infer_pcct_material(:CDTE) === BS.CDTE_MATERIAL
    @test BS._infer_pcct_material(:czt) === BS.CZT_MATERIAL
    @test BS._infer_pcct_material(:CZT) === BS.CZT_MATERIAL
    @test BS._infer_pcct_material(:CdZnTe) === BS.CZT_MATERIAL
    @test BS._infer_pcct_material(:si) === BS.SI_MATERIAL
    @test BS._infer_pcct_material(:Si) === BS.SI_MATERIAL
    @test BS._infer_pcct_material(:silicon) === BS.SI_MATERIAL
    @test BS._infer_pcct_material(:Silicon) === BS.SI_MATERIAL
    # Unknown → CdTe fallback with @warn.
    @test BS._infer_pcct_material(:germanium) === BS.CDTE_MATERIAL
end

# -----------------------------------------------------------------------------
# affine.jl — phantom_to_world_affine + recon_to_world_affine + resample_to_recon
# -----------------------------------------------------------------------------

# Tiny synthetic phantom: 4×4×2 grid of unique labels, 0.5 cm voxels, centered.
function _toy_phantom()
    nx, ny, nz = 4, 4, 2
    labels = reshape(collect(UInt8(1):UInt8(nx * ny * nz)), nx, ny, nz)
    materials = Dict{Int, Symbol}(i => :water for i in 1:Int(nx * ny * nz))
    return BS.Phantom(labels, materials, (0.5, 0.5, 0.5))
end

@testset "phantom_to_world_affine" begin
    ph = _toy_phantom()
    A = BS.phantom_to_world_affine(ph)
    @test size(A) == (4, 4)
    # Diagonal scale = voxel_size.
    @test A[1, 1] == ph.voxel_size[1]
    @test A[2, 2] == ph.voxel_size[2]
    @test A[3, 3] == ph.voxel_size[3]
    # Translate column = origin (last column rows 1..3).
    @test A[1, 4] == ph.origin[1]
    @test A[2, 4] == ph.origin[2]
    @test A[3, 4] == ph.origin[3]
    # Homogeneous row.
    @test A[4, :] == [0.0, 0.0, 0.0, 1.0]
    # Off-diagonals = 0 (no rotation, no shear).
    for i in 1:3, j in 1:3
        i == j && continue
        @test A[i, j] == 0.0
    end
end

@testset "recon_to_world_affine" begin
    s = BS.Scanner(detector_rows = 32, detector_cols = 256)
    g = BS.CTGeometry(s; n_angles = 4, fov_cm = 16.0, z_cm = 4.0)
    matrix_size = (32, 32, 8)
    A = BS.recon_to_world_affine(g, matrix_size)
    @test size(A) == (4, 4)
    # Diagonal scale = fov / matrix_size = voxel size.
    @test A[1, 1] ≈ 16.0 / 32
    @test A[2, 2] ≈ 16.0 / 32
    @test A[3, 3] ≈ 4.0 / 8
    # Translate column places voxel (0,0,0) at -fov/2 + vox/2 (centered grid).
    @test A[1, 4] ≈ -16.0 / 2 + A[1, 1] / 2
    @test A[2, 4] ≈ -16.0 / 2 + A[2, 2] / 2
    @test A[3, 4] ≈ -4.0 / 2 + A[3, 3] / 2
    @test A[4, :] == [0.0, 0.0, 0.0, 1.0]
end

@testset "resample_to_recon — identity grid" begin
    ph = _toy_phantom()
    # Build a geom whose recon grid matches the phantom dimensions + voxel size.
    s = BS.Scanner()
    nx, ny, nz = size(ph.mask)
    fov_xy = ph.voxel_size[1] * nx
    fov_z = ph.voxel_size[3] * nz
    g = BS.CTGeometry(
        s;
        n_angles = 4, fov_cm = fov_xy, z_cm = fov_z,
        n_rows = nz, n_cols = nx,
    )

    @testset "nearest method preserves labels" begin
        out = BS.resample_to_recon(ph, g, (nx, ny, nz); method = :nearest)
        @test size(out) == (nx, ny, nz)
        @test eltype(out) == UInt8
        # Same grid, same labels — should be identical to phantom mask.
        @test out == ph.mask
    end

    @testset "linear method returns Float32 with same values on aligned grid" begin
        out = BS.resample_to_recon(ph, g, (nx, ny, nz); method = :linear)
        @test eltype(out) == Float32
        @test size(out) == (nx, ny, nz)
        # On-grid sampling matches mask values exactly.
        @test maximum(abs.(out .- Float32.(ph.mask))) < 1.0e-5
    end

    @testset "unknown method errors" begin
        @test_throws ErrorException BS.resample_to_recon(ph, g, (nx, ny, nz); method = :bogus)
    end
end

@testset "resample_to_recon — out-of-bounds tolerated" begin
    # Recon grid larger than phantom → out-of-bounds voxels stay zero.
    ph = _toy_phantom()
    s = BS.Scanner()
    g = BS.CTGeometry(
        s;
        n_angles = 4, fov_cm = 10.0, z_cm = 5.0,
        n_rows = 8, n_cols = 16,
    )
    out = BS.resample_to_recon(ph, g, (16, 16, 8); method = :nearest)
    @test size(out) == (16, 16, 8)
    @test eltype(out) == UInt8
    # Some voxels lie outside phantom extent (2×2×1 cm); they should be 0.
    @test minimum(out) == 0
end

# -----------------------------------------------------------------------------
# Helical trajectory (CTGeometry pitch/n_rotations) — geometry contract.
# -----------------------------------------------------------------------------
@testset "helical CTGeometry" begin
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = 32,
        detector_cols = 128,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
    )

    @testset "axial default unchanged" begin
        g = BS.CTGeometry(scanner; n_angles = 90, fov_cm = 20.0)
        @test !BS.is_helical(g)
        @test g.pitch == 0.0
        @test g.table_feed == 0.0
        @test g.n_angles == 90
        @test all(g.source_positions[3, :] .== 0.0)
        @test all(g.detector_centers[3, :] .== 0.0)
    end

    @testset "helical trajectory: feed, centering, total views" begin
        # 32 rows × 1.0 mm at iso = 3.2 cm collimation; pitch 1 → feed 3.2 cm/rot
        g = BS.CTGeometry(scanner; n_angles = 90, fov_cm = 20.0,
            pitch = 1.0, n_rotations = 4.0)
        @test BS.is_helical(g)
        @test g.pitch == 1.0
        @test g.table_feed ≈ 3.2
        @test g.n_angles == 360                          # 90 × 4 total views
        @test length(g.angles) == 360
        @test g.angles[end] ≈ 2π * 4 * (1 - 1 / 360) atol = 1e-9
        # helix centred on isocentre: z spans ±travel/2
        travel = g.table_feed * 4
        @test g.source_positions[3, 1] ≈ -travel / 2
        @test g.source_positions[3, end] ≈ travel / 2 - g.table_feed / 90 atol = 1e-9
        # source and detector translate TOGETHER
        @test g.source_positions[3, :] == g.detector_centers[3, :]
        # z ramps linearly in view index (uniform sampling)
        dz = diff(g.source_positions[3, :])
        @test maximum(abs.(dz .- dz[1])) < 1e-12
        # default recon z-extent = travel − collimation
        @test g.fov[3] ≈ travel - 3.2
    end

    @testset "half pitch halves the feed" begin
        g = BS.CTGeometry(scanner; n_angles = 90, fov_cm = 20.0,
            pitch = 0.5, n_rotations = 2.0)
        @test g.table_feed ≈ 1.6
    end

    @testset "invalid trajectory inputs fail consistently" begin
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 1)
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 90, pitch = 0.0)
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 90, pitch = -0.5)
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 90,
            pitch = 1.0, n_rotations = 0.5)
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 90,
            collimation_mm = 0.0)
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 90, n_rows = 0)
        @test_throws ArgumentError BS.CTGeometry(scanner; n_angles = 90, n_cols = 0)
    end

    @testset "subset geometry carries helical metadata" begin
        g = BS.CTGeometry(scanner; n_angles = 90, fov_cm = 20.0,
            pitch = 1.0, n_rotations = 2.0)
        sub = BS.create_subset_geometry(g, collect(1:5:180))
        @test BS.is_helical(sub)
        @test sub.pitch == g.pitch
        @test sub.table_feed == g.table_feed
    end
end

# -----------------------------------------------------------------------------
# Helical FDK round-trip (CPU, tiny): aperture-weighted WFBP-family recon of a
# water cylinder — μ accuracy, z-uniformity (no banding), axial parity.
# -----------------------------------------------------------------------------
@testset "helical FDK round-trip (CPU)" begin
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = 16,
        detector_cols = 128,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
    )
    geom = BS.CTGeometry(scanner; n_angles = 96, fov_cm = 12.8,
        pitch = 1.0, n_rotations = 3.0)      # feed 1.6 cm/rot, travel 4.8 cm

    nx, nz = 64, 48
    vol_extent = (12.8, 12.8, 6.4)
    vol = zeros(Float32, nx, nx, nz)
    c = (nx + 1) / 2
    for k in 1:nz, j in 1:nx, i in 1:nx
        if (i - c)^2 + (j - c)^2 <= (0.35 * nx)^2
            vol[i, j, k] = 0.2f0
        end
    end

    sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    BS.dd_forward_project!(sino, vol, geom; volume_extent = vol_extent)
    rec = BS.fdk_reconstruct(sino, geom, (64, 64, 24))

    # μ accuracy inside the cylinder, central slice
    roi = rec[25:40, 25:40, 12]
    μ̄ = sum(roi) / length(roi)
    @test abs(μ̄ - 0.2) < 0.01                         # within 5%

    # z-uniformity across the usable fov_z: no helical banding
    zmeans = [sum(rec[25:40, 25:40, k]) / 256 for k in 2:23]
    @test (maximum(zmeans) - minimum(zmeans)) < 0.02   # < 10% of water μ

    # axial parity: same phantom, circular scan
    geom_ax = BS.CTGeometry(scanner; n_angles = 96, fov_cm = 12.8)
    sino_ax = zeros(Float32, geom_ax.n_cols, geom_ax.n_rows, geom_ax.n_angles)
    BS.dd_forward_project!(sino_ax, vol, geom_ax; volume_extent = vol_extent)
    rec_ax = BS.fdk_reconstruct(sino_ax, geom_ax, (64, 64, 8))
    roi_ax = rec_ax[25:40, 25:40, 4]
    @test abs(μ̄ - sum(roi_ax) / length(roi_ax)) < 0.005
end

# -----------------------------------------------------------------------------
# Arc detector — FDK round-trip (equiangular weighting) + helical WFBP on arc.
# -----------------------------------------------------------------------------
@testset "arc detector — reconstruction round-trips" begin
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 16, detector_cols = 128,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = :arc)

    nx, nz = 64, 48
    vol = zeros(Float32, nx, nx, nz)
    c = (nx + 1) / 2
    for k in 1:nz, j in 1:nx, i in 1:nx
        if (i - c)^2 + (j - c)^2 <= (0.35 * nx)^2
            vol[i, j, k] = 0.2f0
        end
    end

    @testset "axial equiangular FDK: μ accuracy + radial flatness" begin
        geom = BS.CTGeometry(scanner; n_angles = 96, fov_cm = 12.8)
        sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.dd_forward_project!(sino, vol, geom; volume_extent = (12.8, 12.8, 9.6))
        rec = BS.fdk_reconstruct(sino, geom, (64, 64, 8))
        roi_c = rec[29:36, 29:36, 4]                       # centre
        roi_e = rec[43:50, 29:36, 4]                       # off-centre (in water)
        @test abs(sum(roi_c) / length(roi_c) - 0.2) < 0.01
        # equiangular weighting keeps the profile flat centre→edge
        @test abs(sum(roi_e) / length(roi_e) - sum(roi_c) / length(roi_c)) < 0.008
    end

    @testset "helical WFBP on arc: z-uniformity" begin
        geom = BS.CTGeometry(scanner; n_angles = 96, fov_cm = 12.8,
            pitch = 1.0, n_rotations = 3.0)
        sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.dd_forward_project!(sino, vol, geom; volume_extent = (12.8, 12.8, 9.6))
        rec = BS.fdk_reconstruct(sino, geom, (64, 64, 24))
        zmeans = [sum(rec[25:40, 25:40, k]) / 256 for k in 2:23]
        @test abs(sum(zmeans) / length(zmeans) - 0.2) < 0.012
        @test (maximum(zmeans) - minimum(zmeans)) < 0.015   # no banding on arc
    end
end
