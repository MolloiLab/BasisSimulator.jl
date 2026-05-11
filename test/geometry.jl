# Tests for src/geometry/ — Scanner + CTGeometry + affine transforms.
#
# Coverage policy: every exported symbol from src/geometry/ is exercised:
#   scanner.jl  : Scanner, DetectorShape, CURVED_DETECTOR, FLAT_DETECTOR,
#                 CTGeometry, _build_pcct_detector, _infer_pcct_material.
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
    @test s.detector_shape === BS.CURVED_DETECTOR
    @test s.detector_type === :energy_integrating
    @test s.n_energy_bins == 1
    @test isempty(s.energy_thresholds)
    @test s.binning_factor == 1
    # native dexel auto-derived = pixel * (SDD/SAD) / binning.
    mag = s.source_to_detector / s.source_to_isocenter
    @test s.native_dexel_col_mm ≈ s.detector_col_size * mag / s.binning_factor
    @test s.native_dexel_row_mm ≈ s.detector_row_size * mag / s.binning_factor
end

@testset "Scanner — flat panel detector dispatch" begin
    s = BS.Scanner(
        detector_shape = BS.FLAT_DETECTOR,
        detector_rows = 512, detector_cols = 512,
        detector_row_size = 0.15, detector_col_size = 0.15,
    )
    @test s.detector_shape === BS.FLAT_DETECTOR
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
