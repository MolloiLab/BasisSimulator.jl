"""
Test Gammex 472 Materials

Validation that Gammex 472 materials are defined correctly with proper
elemental compositions from manufacturer specifications.

Run with:
    julia --project=. test/test_materials.jl
"""

using Test
using BasisSimulator
import XrayAttenuation as XA

@testset "Gammex 472 Materials" begin

    @testset "Material Instantiation" begin
        # Check that all materials are defined as XA.Material
        @test Ca_50 isa XA.Material
        @test Ca_100 isa XA.Material
        @test Ca_200 isa XA.Material
        @test Ca_300 isa XA.Material
        @test Ca_400 isa XA.Material
        @test Ca_500 isa XA.Material
        @test Ca_600 isa XA.Material

        @test I_2_0 isa XA.Material
        @test I_2_5 isa XA.Material
        @test I_5_0 isa XA.Material
        @test I_7_5 isa XA.Material
        @test I_10_0 isa XA.Material
        @test I_15_0 isa XA.Material
        @test I_20_0 isa XA.Material
    end

    @testset "Material Registry" begin
        # Check CUSTOM_MATERIALS dictionary
        @test haskey(CUSTOM_MATERIALS, :Ca_50)
        @test haskey(CUSTOM_MATERIALS, :I_2_0)
        @test length(CUSTOM_MATERIALS) == 14  # 7 Ca + 7 I

        # Check get_material function
        @test get_material(:Ca_50) === Ca_50
        @test get_material(:I_2_0) === I_2_0
        @test get_material(:water) === XA.Materials.water
    end

    @testset "Attenuation Coefficients" begin
        E = 60.0  # keV (typical effective energy for CT)

        # Water reference
        μ_water = get_linear_attenuation(XA.Materials.water, E)
        @test 0.1 < μ_water < 0.5  # cm⁻¹

        # Calcium inserts should have higher attenuation than water
        μ_ca50 = get_linear_attenuation(Ca_50, E)
        μ_ca100 = get_linear_attenuation(Ca_100, E)
        μ_ca600 = get_linear_attenuation(Ca_600, E)

        @test μ_ca50 > μ_water
        @test μ_ca100 > μ_ca50
        @test μ_ca600 > μ_ca100

        # Iodine inserts should have higher attenuation (high Z)
        μ_i2 = get_linear_attenuation(I_2_0, E)
        μ_i20 = get_linear_attenuation(I_20_0, E)

        @test μ_i2 > μ_water
        @test μ_i20 > μ_i2
    end

    @testset "Hounsfield Units" begin
        E = 60.0  # keV

        # Calculate HU for each material
        hu_ca50 = validate_material_hu(:Ca_50, E)
        hu_ca100 = validate_material_hu(:Ca_100, E)
        hu_ca600 = validate_material_hu(:Ca_600, E)

        hu_i2 = validate_material_hu(:I_2_0, E)
        hu_i20 = validate_material_hu(:I_20_0, E)

        # Calcium HU should be positive and increase with concentration
        @test hu_ca50 > 0
        @test hu_ca100 > hu_ca50
        @test hu_ca600 > hu_ca100

        # Iodine HU should be positive and increase with concentration
        @test hu_i2 > 0
        @test hu_i20 > hu_i2

        # Sanity checks based on manufacturer-specified compositions
        @test 100 < hu_ca50 < 600
        @test 1000 < hu_ca600 < 3000
        @test 0 < hu_i20 < 500
    end

    @testset "Energy Dependence" begin
        # Check beam hardening: attenuation should decrease with energy
        energies = [40.0, 60.0, 80.0, 100.0]

        μ_values_ca = [get_linear_attenuation(Ca_200, E) for E in energies]
        μ_values_i = [get_linear_attenuation(I_5_0, E) for E in energies]

        # Attenuation should generally decrease with energy
        @test μ_values_ca[1] > μ_values_ca[end]
        @test μ_values_i[1] > μ_values_i[end]
    end

    @testset "Density Validation" begin
        # Check manufacturer-specified densities
        using Unitful: ustrip, @u_str

        @test ustrip(u"g/cm^3", Ca_50.density) ≈ 1.17
        @test ustrip(u"g/cm^3", Ca_100.density) ≈ 1.24
        @test ustrip(u"g/cm^3", Ca_600.density) ≈ 2.01

        @test ustrip(u"g/cm^3", I_2_0.density) ≈ 1.03
        @test ustrip(u"g/cm^3", I_20_0.density) ≈ 1.04
    end

end

# Print material properties table
println("\n" * "="^70)
println("Gammex 472 Material Properties (Manufacturer Specifications)")
println("="^70)
print_material_properties()

println("\n✅ All Gammex 472 material tests passed!")
