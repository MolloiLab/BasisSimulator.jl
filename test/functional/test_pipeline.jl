# Oracle test for the end-to-end functional EICT pipeline (phantom → HU) against
# the legacy notebook-01 chain: simulate! → apply_bhc_water → reconstruct!(FDK)
# → to_hounsfield.  Standalone: `julia --project=. -t 2 test/functional/test_pipeline.jl`.
module FunctionalPipelineTests

using Test, Statistics, LinearAlgebra, Random
using BasisSimulator
const BS = BasisSimulator
const BSF = BasisSimulator.Functional

function _pipeline_fixture(; use_noise, n_voxels = 32, n_slices = 4, views = 16)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 64,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0,
        electronic_noise = 5.0, detection_gain = 10.0,
    )
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = views, rotation_time = 0.5)
    sim_opts = BS.SimOptions(;
        fidelity = :eict, seed = 42,
        use_noise = use_noise, use_scatter = false,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false,
    )
    recon_opts = BS.ReconOptions(matrix_size = (n_voxels, n_voxels, n_slices), fov_cm = 20.0)
    phantom = BS.create_gammex_472(n_voxels = n_voxels, n_slices = n_slices, fov_cm = 20.0, z_cm = 2.0)
    return (; scanner, protocol, sim_opts, recon_opts, phantom)
end

# Legacy oracle: exactly the nb01 corrected pipeline.
function _legacy_chain(fx)
    ws = BS.create_eict_workspace(fx.scanner, fx.protocol, fx.sim_opts, fx.recon_opts, fx.phantom)
    BS.simulate!(ws, fx.phantom, fx.protocol, fx.sim_opts)
    ε = copy(ws.noise_rand_cpu); ε_e = copy(ws.enoise_rand_cpu)
    model = BS.calibrate_bhc_water(fx.sim_opts, fx.protocol; scanner = fx.scanner, geom = ws.geom)
    sino_bhc = BS.apply_bhc_water(ws.sinogram, model)
    ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, ws.geom, fx.recon_opts.matrix_size)
    recon = copy(BS.reconstruct!(ws_fdk, sino_bhc, ws.geom))
    hu = BS.to_hounsfield(recon; μ_water = model.μ_water_ref)
    return (; sino_raw = copy(ws.sinogram), sino_bhc, recon, hu, ε, ε_e, model)
end

relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(Float32))

@testset "functional/pipeline: phantom → HU vs legacy nb01 chain" begin
    for use_noise in (false, true)
        fx = _pipeline_fixture(; use_noise)
        ref = _legacy_chain(fx)
        pipe = BSF.eict_pipeline(fx.phantom, fx.scanner, fx.protocol, fx.sim_opts, fx.recon_opts)
        @test BSF.eltype(pipe) === Float32
        @test pipe.n_mat == length(fx.phantom.materials)
        @test pipe.μ_water ≈ ref.model.μ_water_ref
        fr = BSF.onehot_fractions(fx.phantom.mask, pipe.n_mat)
        @test size(fr) == (size(fx.phantom.mask)..., pipe.n_mat)
        @test all(sum(fr; dims = 4) .== 1)

        ε, ε_e = BSF.draw_eict_noise(pipe; seed = 42)
        if use_noise
            @test ε == ref.ε                      # same MersenneTwister stream as simulate!
            @test ε_e !== nothing && ε_e == ref.ε_e
        end
        sino = BSF.simulate_sino(fr, pipe, ε, ε_e)
        @test size(sino) == size(ref.sino_bhc)
        gate = use_noise ? 1e-4 : 1e-5
        @test relmax(sino, ref.sino_bhc) < gate
        μ = BSF.reconstruct_μ(sino, pipe)
        @test relmax(μ, ref.recon) < gate
        hu = BSF.eict_forward(fr, pipe, ε, ε_e)
        @test size(hu) == fx.recon_opts.matrix_size
        @test relmax(hu, ref.hu) < gate
        @test hu == BSF.eict_forward(fr, pipe, ε, ε_e)   # deterministic / pure
        @info "functional/pipeline use_noise=$use_noise: sino relmax=$(relmax(sino, ref.sino_bhc)) μ relmax=$(relmax(μ, ref.recon)) HU relmax=$(relmax(hu, ref.hu))"
    end
end

@testset "functional/pipeline: no-BHC variant and Float64 plans" begin
    fx = _pipeline_fixture(; use_noise = false)
    pipe = BSF.eict_pipeline(fx.phantom, fx.scanner, fx.protocol, fx.sim_opts, fx.recon_opts; bhc = nothing, T = Float64)
    @test BSF.eltype(pipe) === Float64
    fr = BSF.onehot_fractions(fx.phantom.mask, pipe.n_mat; T = Float64)
    hu = BSF.eict_forward(fr, pipe)
    @test eltype(hu) === Float64 && all(isfinite, hu)
    @test pipe.μ_water ≈ BS.get_reference_μ_water(70.0)
end

end # module

@testset "Functional.pipeline — view_batch reproduces the per-view forward" begin
    scanner = BS.Scanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 32, detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 12, rotation_time = 0.5)
    sim_opts = BS.SimOptions(; fidelity = :eict, seed = 7, use_noise = false, use_scatter = false,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false)
    recon_opts = BS.ReconOptions(matrix_size = (16, 16, 2), fov_cm = 20.0)
    phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 16, n_slices = 2, fov_cm = 20.0, z_cm = 1.0))
    p1 = BSF.eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts)
    p8 = BSF.eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; view_batch = 8)
    @test p1.view_batch == 1 && p8.view_batch == 8
    fr = BSF.onehot_fractions(phantom.mask, p1.n_mat)
    @test BSF.material_paths(fr, p8) == BSF.material_paths(fr, p1)
    h1 = BSF.eict_forward(fr, p1); h8 = BSF.eict_forward(fr, p8)
    @test maximum(abs.(h8 .- h1)) <= 5e-3                        # FDK view-batch reduction order only
    @test_throws ArgumentError BSF.eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; view_batch = 0)
end
