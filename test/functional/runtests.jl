# BasisSimulator.Functional — oracle tests for the pure tensor-program core.
# Included from test/runtests.jl (needs `BS`, `_toy_proj_geom`, Random, LinearAlgebra).
include("harness.jl")
@testset "functional/dd_projector" begin
    include("test_dd_projector.jl")
end
@testset "functional/fbp" begin
    include("test_fbp.jl")
end
@testset "functional/eict" begin
    include("test_eict.jl")
end
@testset "functional/hir" begin
    include("test_hir.jl")
end
@testset "functional/denoise" begin
    include("test_denoise.jl")
end
@testset "functional/vmi" begin
    include("test_vmi.jl")
end
@testset "functional/pcct" begin
    include("test_pcct.jl")
end
@testset "functional/pipeline" begin
    include("test_pipeline.jl")
end
