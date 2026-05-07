using Test
using BasisSimulator
using Random
using Statistics: mean, std
const BS = BasisSimulator

@testset "BasisSimulator.jl" begin
    @testset "api/" begin
        include("api.jl")
    end
end
