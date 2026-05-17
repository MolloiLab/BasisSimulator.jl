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
    @testset "detector/" begin
        include("detector.jl")
    end
    @testset "geometry/" begin
        include("geometry.jl")
    end
    @testset "object/" begin
        include("object.jl")
    end
    @testset "projection/" begin
        include("projection.jl")
    end
    @testset "source/" begin
        include("source.jl")
    end
end
