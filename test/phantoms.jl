# Tests for src/phantoms/xcat_artifacts.jl — offline (no network): the registry,
# the material map, and the XCIST voxelized parser on a synthetic phantom whose
# assembly is predicted by hand.

using Test
using BasisSimulator
const BS = BasisSimulator
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

    # path= bypasses download; `name` is irrelevant with a local path.
    p = load_xcat_phantom(:female_slab; path = dir)
    @test size(p.mask) == (3, 3, 1)
    @test sort(unique(p.mask)) == UInt8[0, 1, 2]
    @test count(==(0x01), p.mask) == 3
    @test count(==(0x02), p.mask) == 4
    @test count(==(0x00), p.mask) == 2
    @test p.label_names == Dict(1 => "ncat_water", 2 => "ncat_muscle")
    @test length(p.materials) == 3               # air + water + muscle (label-keyed, 0 = air)
    @test p.voxel_size_cm == (0.1, 0.1, 0.1)      # 1 mm header → 0.1 cm

    # the pieces build a Phantom in one line
    phantom = BS.Phantom(p.mask, p.materials, p.voxel_size_cm)
    @test size(phantom.mask) == (3, 3, 1)

    # material override by XCIST name + explicit voxel size
    mats = BS.xcat_default_materials()
    mats["ncat_water"] = XA.Materials.air
    p2 = load_xcat_phantom(:female_slab; path = dir, materials = mats, voxel_size_cm = (0.2, 0.2, 0.2))
    @test p2.voxel_size_cm == (0.2, 0.2, 0.2)

    # an unmapped material name is a clear error
    @test_throws ErrorException load_xcat_phantom(:female_slab; path = dir, materials = Dict("ncat_muscle" => XA.Materials.muscle))
end

# Regression: each piece's rows are stored BOTTOM-UP, so a piece must be read in
# reverse along y before being placed at `-y_offset` (the same rule as x and z).
# Reading rows forward mis-registers every piece by its own height.  The `XCIST
# voxelized parse` phantom above cannot catch that — its two pieces are the same
# height and diagonally symmetric, so the wrong convention merely mirrors the
# result.  Here the pieces have DIFFERENT heights and tile the grid exactly, so
# the buggy convention stacks them in the opposite order and the mask differs.
@testset "XCIST y rows are stored bottom-up (regression)" begin
    dir = mktempdir()
    # water: 2×1 (one row) and muscle: 2×2 (two rows) tile a 2×3 grid, no air.
    write(joinpath(dir, "syn.density_0"), Int8[1, 1])           # water,  cols=2 rows=1
    write(joinpath(dir, "syn.density_1"), Int8[1, 1, 1, 1])     # muscle, cols=2 rows=2
    hdr = Dict(
        "n_materials" => 2,
        "mat_name" => ["ncat_water", "ncat_muscle"],
        "volumefractionmap_filename" => ["syn.density_0", "syn.density_1"],
        "volumefractionmap_datatype" => ["int8", "int8"],
        "cols" => [2, 2], "rows" => [1, 2], "slices" => [1, 1],
        "x_size" => [1.0, 1.0], "y_size" => [1.0, 1.0], "z_size" => [1.0, 1.0],
        "x_offset" => [0.5, 0.5], "y_offset" => [1.5, 3.5], "z_offset" => [1.0, 1.0],
    )
    open(io -> JSON.print(io, hdr), joinpath(dir, "syn.json"), "w")

    p = load_xcat_phantom(:female_slab; path = dir)
    @test size(p.mask) == (2, 3, 1)

    # muscle occupies the first two rows, water the last.  Reading rows forward
    # swaps the two pieces (water row 1, muscle rows 2–3) — this is the assertion
    # that fails on the bug.
    @test p.mask[:, :, 1] == UInt8[2 2 1; 2 2 1]

    # the pieces tile exactly: every voxel claimed once, none twice, no air
    @test count(==(0x00), p.mask) == 0
    @test count(==(0x01), p.mask) == 2          # water occupancy preserved in full
    @test count(==(0x02), p.mask) == 4          # muscle occupancy preserved in full
end
