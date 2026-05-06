using Test
using BasisSimulator
const BS = BasisSimulator

@testset "BasisSimulator.jl" begin
    @testset "api/" begin
        include("api.jl")
    end
end
