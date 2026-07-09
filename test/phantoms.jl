# Tests for src/phantoms/xcat_artifacts.jl — offline (no network): the registry,
# the material map, and the XCIST voxelized parser on a synthetic phantom whose
# assembly is predicted by hand.

using Test
using BasisSimulator
const BS = BasisSimulator
import BasisSimulator: _parse_xcist_voxelized
import XrayAttenuation as XA
import JSON

@testset "XCAT registry + material map" begin
    @test BS.xcat_phantoms() == [:female_chest, :female_slab, :male_chest, :male_slab]
    m = BS.xcat_default_materials()
    for n in ("ncat_water", "ncat_muscle", "ncat_lung", "ncat_dry_spine", "ncat_dry_rib",
              "ncat_adipose", "ncat_blood", "ncat_heart", "ncat_cartilage",
              "ncat_liver", "ncat_intestine", "ncat_spleen")
        @test haskey(m, n)
        @test m[n] isa XA.Material
    end
    # every entry has valid-length hashes (sha256 hex = 64, git-tree-sha1 hex = 40)
    for (_, e) in BS.XCAT_REGISTRY
        @test length(e.sha256) == 64
        @test length(e.tree_hash) == 40
    end
end

@testset "XCIST voxelized parse (synthetic, offline)" begin
    dir = mktempdir()
    # Two fully-occupied 2×2×1 material boxes, offset so material 2 is shifted
    # by one voxel in +col/+row.  With the isocenter-alignment placement, the
    # assembled grid is 3×3×1 (see the hand computation in the loader docstring):
    #   label 1 (water):  [1,1] [2,1] [1,2]            → 3 voxels
    #   label 2 (muscle): [2,2] [3,2] [2,3] [3,3]      → 4 voxels (overwrites (2,2))
    #   label 0 (air):    [3,1] [1,3]                  → 2 voxels
    write(joinpath(dir, "syn.density_0"), Int8[1, 1, 1, 1])   # water,  2×2
    write(joinpath(dir, "syn.density_1"), Int8[1, 1, 1, 1])   # muscle, 2×2
    hdr = Dict(
        "n_materials" => 2,
        "mat_name" => ["ncat_water", "ncat_muscle"],
        "volumefractionmap_filename" => ["syn.density_0", "syn.density_1"],
        "volumefractionmap_datatype" => ["int8", "int8"],
        "cols" => [2, 2], "rows" => [2, 2], "slices" => [1, 1],
        "x_size" => [1.0, 1.0], "y_size" => [1.0, 1.0], "z_size" => [1.0, 1.0],
        "x_offset" => [1.5, 0.5], "y_offset" => [1.5, 0.5], "z_offset" => [1.0, 1.0],
    )
    open(io -> JSON.print(io, hdr), joinpath(dir, "syn.json"), "w")

    ph = _parse_xcist_voxelized(dir)
    @test size(ph.mask) == (3, 3, 1)
    @test sort(unique(ph.mask)) == UInt8[0, 1, 2]
    @test count(==(0x01), ph.mask) == 3
    @test count(==(0x02), ph.mask) == 4
    @test count(==(0x00), ph.mask) == 2
    @test length(ph.materials) == 3              # air + water + muscle (indexed by label+1)
    @test ph.voxel_size == (0.1, 0.1, 0.1)       # 1 mm header → 0.1 cm

    # material override by XCIST name
    mats = BS.xcat_default_materials()
    mats["ncat_water"] = XA.Materials.air
    ph2 = _parse_xcist_voxelized(dir; materials = mats, voxel_size_cm = (0.2, 0.2, 0.2))
    @test ph2.voxel_size == (0.2, 0.2, 0.2)

    # an unmapped material name is a clear error
    @test_throws ErrorException _parse_xcist_voxelized(dir; materials = Dict("ncat_muscle" => XA.Materials.muscle))
end
