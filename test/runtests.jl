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
        include("detector_offset.jl")
    end
    @testset "dose/" begin
        include("dose.jl")
    end
    @testset "geometry/" begin
        include("geometry.jl")
    end
    @testset "filtering/" begin
        include("filtering.jl")
    end
    @testset "helical/" begin
        include("heel_axis.jl")
        include("helical.jl")
        include("hir_support.jl")
    end
    @testset "memory lifecycle/" begin
        include("memory_lifecycle.jl")
    end
    @testset "nchannel/" begin
        include("nchannel.jl")
        include("hypr.jl")
    end
    @testset "object/" begin
        include("object.jl")
    end
    @testset "paths/" begin
        include("paths.jl")
        include("pcct_bowtie.jl")
    end
    @testset "phantoms/" begin
        include("phantoms.jl")
    end
    @testset "projection/" begin
        include("projection.jl")
    end
    @testset "scanner families/" begin
        include("scanner_split.jl")
    end
    @testset "source/" begin
        include("source.jl")
    end
end
