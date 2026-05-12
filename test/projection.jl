# Tests for src/projection/ — Siddon forward projection + polychromatic helpers.
#
# Coverage policy:
#   siddon.jl       : siddon_forward_project, siddon_forward_project!
#                     (siddon_fused_poly_project! and siddon_fused_spectral_project!
#                     are exercised end-to-end through test/api.jl's
#                     simulate!(PCCTWorkspace) and simulate!(EICTWorkspace);
#                     duplicating that here would just retest the workspace path).
#   polychromatic.jl: create_μ_volume! (the live public helper).
#                     `_apply_physics_no_noise!` and `_forward_project_poly!` are
#                     reached end-to-end through test/api.jl's simulate! tests —
#                     they're private + need the full PhysicsConfig + workspace
#                     dance to set up, so we skip duplicate behavioral coverage
#                     here.

# -----------------------------------------------------------------------------
# Helper: tiny scanner + geometry for projection tests.
# -----------------------------------------------------------------------------
function _toy_proj_geom(; n_cols = 16, n_rows = 4, n_angles = 4, fov_cm = 5.0)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = n_rows,
        detector_cols = n_cols,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
    )
    return BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = fov_cm, z_cm = 1.0)
end

# -----------------------------------------------------------------------------
# siddon_forward_project — allocating + identity case.
# -----------------------------------------------------------------------------
@testset "siddon_forward_project (allocating)" begin
    geom = _toy_proj_geom()
    matrix_size = (16, 16, 4)

    @testset "zero volume → zero sinogram" begin
        vol = zeros(Float32, matrix_size...)
        sino = BS.siddon_forward_project(vol, geom)
        @test size(sino) == (geom.n_cols, geom.n_rows, geom.n_angles)
        @test eltype(sino) == Float32
        @test all(sino .== 0)
    end

    @testset "constant μ block → constant-ish sinogram on rays through it" begin
        vol = fill(Float32(0.2), matrix_size...)
        sino = BS.siddon_forward_project(vol, geom)
        @test size(sino) == (geom.n_cols, geom.n_rows, geom.n_angles)
        # All rays through a uniform block have positive line integral.
        # Pick the middle column (where the FOV is fully covered).
        mid_c = geom.n_cols ÷ 2 + 1
        mid_r = geom.n_rows ÷ 2 + 1
        @test all(>(0), sino[mid_c, mid_r, :])
        # Line integral = μ × chord length.  Chord ≤ fov × √2; with μ = 0.2 cm⁻¹
        # and fov = 5 cm, max line integral ≈ 0.2 × 5 × √2 ≈ 1.41.
        @test maximum(sino) ≤ 0.2 * 5.0 * sqrt(2) + 1.0e-3
    end

    @testset "linearity in μ — 2× scale doubles the sinogram" begin
        vol = fill(Float32(0.1), matrix_size...)
        sino_1x = BS.siddon_forward_project(vol, geom)
        sino_2x = BS.siddon_forward_project(2 .* vol, geom)
        # Linearity holds exactly (Siddon is a linear operator on μ).
        @test maximum(abs.(sino_2x .- 2 .* sino_1x)) < 1.0e-4
    end
end

# -----------------------------------------------------------------------------
# siddon_forward_project! — in-place equivalence.
# -----------------------------------------------------------------------------
@testset "siddon_forward_project! (in-place)" begin
    geom = _toy_proj_geom()
    matrix_size = (16, 16, 4)
    vol = fill(Float32(0.15), matrix_size...)

    sino_alloc = BS.siddon_forward_project(vol, geom)
    sino_inplace = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    BS.siddon_forward_project!(sino_inplace, vol, geom)

    @test maximum(abs.(sino_alloc .- sino_inplace)) < 1.0e-4
end

# -----------------------------------------------------------------------------
# create_μ_volume! — label → material → μ at energy mapping.
# -----------------------------------------------------------------------------
@testset "create_μ_volume!" begin
    nx, ny, nz = 8, 8, 2
    labels = zeros(UInt8, nx, ny, nz)
    labels[1:4, :, :] .= 0x01    # half = water
    labels[5:8, :, :] .= 0x02    # other half = Ca_100

    # Index 1 = label 0 (air), 2 = label 1 (water), 3 = label 2 (Ca_100).
    materials = [
        BS.XA.Materials.air,
        BS.XA.Materials.water,
        BS.get_material(:Ca_100),
    ]
    E = 80.0
    μ_vol = zeros(Float32, nx, ny, nz)
    BS.create_μ_volume!(μ_vol, labels, materials, E)

    μ_water = Float32(BS.compute_μ_at_energy(BS.XA.Materials.water, E))
    μ_ca100 = Float32(BS.compute_μ_at_energy(BS.get_material(:Ca_100), E))

    # Spot-check both halves carry the right μ.
    @test μ_vol[1, 1, 1] ≈ μ_water
    @test μ_vol[5, 1, 1] ≈ μ_ca100
    # Calcium should attenuate more than water.
    @test μ_ca100 > μ_water

    # Re-running at a different energy gives a different μ.
    μ_vol_2 = zeros(Float32, nx, ny, nz)
    BS.create_μ_volume!(μ_vol_2, labels, materials, 120.0)
    @test μ_vol_2[1, 1, 1] < μ_vol[1, 1, 1]   # water μ drops with E
end

@testset "create_μ_volume! — Float64 / Float32 element type respected" begin
    labels = reshape(UInt8[1, 1, 1, 1], 2, 1, 2)
    materials = [BS.XA.Materials.air, BS.XA.Materials.water]
    μ32 = zeros(Float32, size(labels)...)
    μ64 = zeros(Float64, size(labels)...)
    BS.create_μ_volume!(μ32, labels, materials, 70.0)
    BS.create_μ_volume!(μ64, labels, materials, 70.0)
    @test eltype(μ32) == Float32
    @test eltype(μ64) == Float64
    @test μ32[1, 1, 1] ≈ Float32(μ64[1, 1, 1])
end
