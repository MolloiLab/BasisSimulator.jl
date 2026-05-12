# Tests for src/object/ — materials registry, attenuation getters, Phantom struct.
#
# Coverage policy: every exported symbol is exercised:
#   materials.jl    : get_material, get_region_materials.
#   attenuation.jl  : compute_μ_at_energy, compute_mass_μ_at_energy, μ_to_HU,
#                     HU_to_μ, get_reference_μ_water, to_hounsfield, get_basis_mu.
#   phantom.jl      : RegionLabel + REGION_* enum members, Phantom, compute_μ,
#                     create_gammex_472, create_phantom_from_mask.

# -----------------------------------------------------------------------------
# materials.jl
# -----------------------------------------------------------------------------
@testset "get_material — registry + XA.Materials fallback" begin
    # Gammex inserts via registry.
    for sym in (
            :Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600,
            :I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0,
            :solid_water,
        )
        m = BS.get_material(sym)
        @test m isa BS.XA.Material
    end
    # Common materials.
    @test BS.get_material(:water) isa BS.XA.Material
    @test BS.get_material(:air) isa BS.XA.Material
    @test BS.get_material(:bone) === BS.get_material(:cortical_bone)
    @test BS.get_material(:bone) === BS.get_material(:corticalbone)
    @test BS.get_material(:soft_tissue) === BS.get_material(:softtissue)
    # Fallback to XA.Materials property if not in registry.
    @test BS.get_material(:water) === BS.XA.Materials.water
    # Truly unknown symbol errors with a discoverable message.
    @test_throws ErrorException BS.get_material(:not_a_material)
end

@testset "get_region_materials — region-indexed vector" begin
    mats = BS.get_region_materials()
    @test length(mats) == 27
    # Index 1 = REGION_BACKGROUND = air, index 3 = REGION_WATER = water,
    # index 4 = REGION_SOLID_WATER = Gammex solid water.
    @test mats[1] === BS.XA.Materials.air
    @test mats[3] === BS.XA.Materials.water
    # Index 11 = REGION_CA_50, index 17 = REGION_CA_600.
    @test mats[11] === BS.get_material(:Ca_50)
    @test mats[17] === BS.get_material(:Ca_600)
    # Index 21 = REGION_I_2_0, index 27 = REGION_I_20_0.
    @test mats[21] === BS.get_material(:I_2_0)
    @test mats[27] === BS.get_material(:I_20_0)
    # Unused slots (e.g., 5-10) are filled with air.
    for i in 5:10
        @test mats[i] === BS.XA.Materials.air
    end
end

# -----------------------------------------------------------------------------
# attenuation.jl — physical bounds + round-trip
# -----------------------------------------------------------------------------
@testset "compute_μ_at_energy + compute_mass_μ_at_energy" begin
    water = BS.XA.Materials.water
    # Water linear μ in the diagnostic range: ~0.2 at 60 keV, dropping
    # to ~0.15 at 100 keV.  Bounds chosen wide enough to survive XA updates.
    @test 0.18 < BS.compute_μ_at_energy(water, 60.0) < 0.22
    @test 0.13 < BS.compute_μ_at_energy(water, 100.0) < 0.18
    # μ decreases monotonically with E in this range.
    @test BS.compute_μ_at_energy(water, 60.0) > BS.compute_μ_at_energy(water, 100.0)
    # Mass μ = μ / ρ; for water ρ ≈ 1 g/cm³, so the two should be close.
    μ_lin = BS.compute_μ_at_energy(water, 60.0)
    μ_mass = BS.compute_mass_μ_at_energy(water, 60.0)
    @test abs(μ_lin - μ_mass) < 0.02   # water density = 1.000 g/cm³
end

@testset "μ_to_HU + HU_to_μ — round-trip" begin
    μ_water = 0.193
    # Water → 0 HU.
    @test BS.μ_to_HU(μ_water, μ_water) ≈ 0.0  atol = 1.0e-10
    # 1.5x water → +500 HU.
    @test BS.μ_to_HU(1.5 * μ_water, μ_water) ≈ 500.0
    # Round-trip.
    for hu in (-500.0, 0.0, 100.0, 1000.0)
        μ = BS.HU_to_μ(hu, μ_water)
        @test BS.μ_to_HU(μ, μ_water) ≈ hu  atol = 1.0e-10
    end
    # Array overload.
    μs = [0.5μ_water, μ_water, 1.5μ_water]
    @test BS.μ_to_HU(μs, μ_water) ≈ [-500.0, 0.0, 500.0]
end

@testset "get_reference_μ_water + to_hounsfield" begin
    # Default reference energy (60 keV) is positive + finite + in water range.
    μ_60 = BS.get_reference_μ_water()
    μ_70 = BS.get_reference_μ_water(70.0)
    @test 0.15 < μ_60 < 0.25
    @test 0.15 < μ_70 < 0.25
    @test μ_60 > μ_70   # hardens with E

    @testset "to_hounsfield with manual μ_water" begin
        recon = Float32[1.0 1.5; 2.0 0.5] .* Float32(μ_60)
        hu = BS.to_hounsfield(recon; μ_water = μ_60)
        @test hu[1, 1] ≈ 0.0   atol = 1.0e-4
        @test hu[1, 2] ≈ 500.0 atol = 1.0e-4
        @test hu[2, 1] ≈ 1000.0 atol = 1.0e-4
        @test hu[2, 2] ≈ -500.0 atol = 1.0e-4
    end

    @testset "to_hounsfield with water_mask (empirical calibration)" begin
        recon = Float32[1.0 1.5; 2.0 0.5] .* Float32(μ_60)
        mask = Bool[true false; false true]    # avg(1.0, 0.5) * μ_60 = 0.75 μ_60
        hu = BS.to_hounsfield(recon; water_mask = mask)
        # mean(recon[mask]) → calibrated μ_water = 0.75 μ_60.
        μ_cal = 0.75 * μ_60
        @test hu[1, 1] ≈ Float32(BS.μ_to_HU(Float64(recon[1, 1]), μ_cal))  atol = 1.0e-3
    end

    @testset "to_hounsfield default → NIST water at 70 keV" begin
        recon = Float32[1.0;;] .* Float32(BS.get_reference_μ_water(70.0))
        hu = BS.to_hounsfield(recon)
        @test hu[1, 1] ≈ 0.0  atol = 1.0e-4
    end
end

@testset "get_basis_mu — :water / :iodine / :calcium dispatch" begin
    E = 70.0
    μ_water = BS.get_basis_mu(:water, E)
    μ_iod = BS.get_basis_mu(:iodine, E)
    μ_cal = BS.get_basis_mu(:calcium, E)
    @test μ_water > 0
    # Iodine + calcium variants add a contrast on top of water → must
    # exceed pure water.
    @test μ_iod > μ_water
    @test μ_cal > μ_water
    @test μ_water == BS.compute_μ_at_energy(BS.XA.Materials.water, E)
    # Unknown material errors.
    @test_throws ErrorException BS.get_basis_mu(:gadolinium, E)
end

# -----------------------------------------------------------------------------
# phantom.jl — region enum + ctor + factories
# -----------------------------------------------------------------------------
@testset "RegionLabel enum + REGION_* constants" begin
    @test BS.REGION_BACKGROUND isa BS.RegionLabel
    @test UInt8(BS.REGION_BACKGROUND) == 0
    @test UInt8(BS.REGION_AIR) == 1
    @test UInt8(BS.REGION_WATER) == 2
    @test UInt8(BS.REGION_SOLID_WATER) == 3
    @test UInt8(BS.REGION_CA_50) == 10
    @test UInt8(BS.REGION_CA_600) == 16
    @test UInt8(BS.REGION_I_2_0) == 20
    @test UInt8(BS.REGION_I_20_0) == 26
end

@testset "Phantom ctor + field invariants" begin
    nx, ny, nz = 4, 4, 2
    labels = reshape(collect(UInt8(0):UInt8(nx * ny * nz - 1)), nx, ny, nz)
    mats = Dict{Int, BS.XA.Material}(
        i => BS.XA.Materials.water for i in 0:(nx * ny * nz - 1)
    )
    ph = BS.Phantom(labels, mats, (0.5, 0.5, 0.5))
    @test size(ph.mask) == (nx, ny, nz)
    @test ph.voxel_size == (0.5, 0.5, 0.5)
    # Phantom is centered at isocenter by default → origin is the
    # center of the first voxel = -extent/2 + voxel/2.
    @test ph.origin[1] ≈ -nx * 0.5 / 2 + 0.5 / 2
    @test ph.origin[2] ≈ -ny * 0.5 / 2 + 0.5 / 2
    @test ph.origin[3] ≈ -nz * 0.5 / 2 + 0.5 / 2
    @test ph.extent == (nx * 0.5, ny * 0.5, nz * 0.5)
    # Materials vector length = max_label + 1 (air-padded by build_materials_vector).
    @test length(ph.materials) == maximum(keys(mats)) + 1
end

@testset "compute_μ — consistent with compute_μ_at_energy" begin
    # 2×2×2 labeled volume. Literal `labels[a b; c d;;; e f; g h]` gives
    # the slice-1 layout [a b; c d] with labels[1,1,1]=a, labels[1,2,1]=b,
    # labels[2,1,1]=c, labels[2,2,1]=d.
    labels = UInt8[1 2; 3 1;;; 2 3; 1 2]
    mats = Dict{Int, BS.XA.Material}(
        1 => BS.XA.Materials.water,
        2 => BS.get_material(:solid_water),
        3 => BS.get_material(:Ca_100),
    )
    ph = BS.Phantom(labels, mats, (0.1, 0.1, 0.1))
    μ_vol = BS.compute_μ(ph, 80.0)
    @test size(μ_vol) == size(ph.mask)
    @test eltype(μ_vol) == Float32
    # Spot-check label → material mapping per indices above.
    @test μ_vol[1, 1, 1] ≈ Float32(BS.compute_μ_at_energy(BS.XA.Materials.water, 80.0))
    @test μ_vol[1, 2, 1] ≈ Float32(BS.compute_μ_at_energy(BS.get_material(:solid_water), 80.0))
    @test μ_vol[2, 1, 1] ≈ Float32(BS.compute_μ_at_energy(BS.get_material(:Ca_100), 80.0))
end

@testset "create_gammex_472 — canonical 16-rod phantom" begin
    ph = BS.create_gammex_472(n_voxels = 64, n_slices = 2, fov_cm = 35.0)
    # Mask carries the documented region labels.
    @test eltype(ph.mask) == UInt8
    unique_labels = Set(Array(ph.mask))
    # Must contain solid water + at least some Ca / I inserts.
    @test UInt8(BS.REGION_SOLID_WATER) in unique_labels
    @test UInt8(BS.REGION_CA_50) in unique_labels
    @test UInt8(BS.REGION_I_20_0) in unique_labels
    @test size(ph.mask) == (64, 64, 2)
    # μ at 70 keV runs without error.
    μ = BS.compute_μ(ph, 70.0)
    @test all(isfinite, μ)
    @test all(>=(0), μ)
end

@testset "create_phantom_from_mask — thin wrapper" begin
    labels = reshape(UInt8[1, 2, 2, 1], 2, 1, 2)
    mats = Dict{Int, BS.XA.Material}(
        1 => BS.XA.Materials.water,
        2 => BS.XA.Materials.air,
    )
    ph = BS.create_phantom_from_mask(labels, mats, (0.5, 0.5, 0.5))
    @test ph isa BS.Phantom
    @test size(ph.mask) == size(labels)
    @test ph.voxel_size == (0.5, 0.5, 0.5)
end
