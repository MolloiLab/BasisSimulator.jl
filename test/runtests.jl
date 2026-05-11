using Test
using BasisSimulator
using Random
using Statistics: mean, std
using LinearAlgebra
const BS = BasisSimulator

@testset "BasisSimulator.jl" begin
    @testset "api/" begin
        include("api.jl")
    end
    @testset "bowtie/" begin
        include("bowtie.jl")
    end
    @testset "correction/" begin
        include("correction.jl")
    end
    @testset "denoising/" begin
        include("denoising.jl")
    end
end
