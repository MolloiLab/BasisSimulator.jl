# Reactant + Enzyme smoke for VIEW BATCHING (M5): the end-to-end pipeline compiled with a
# looped `view_batch = 4` (StableHLO while loop over view batches in DD, spectral sum and FDK)
# must reproduce the per-view unrolled program (HU and gradient), and the IR must carry the loop.
#
#   until mkdir /tmp/bs_reactant.lock 2>/dev/null; do sleep 30; done; trap 'rmdir /tmp/bs_reactant.lock' EXIT
#   julia --project=envs/reactant -t 2 --heap-size-hint=3G test/functional/reactant/smoke_view_batch.jl
using Test, Statistics, LinearAlgebra, Random, Printf
using Reactant, Enzyme
using BasisSimulator
const BS = BasisSimulator
const BSF = BasisSimulator.Functional

function fixture(; view_batch)
    scanner = BS.Scanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 32, detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 8, rotation_time = 0.5)
    sim_opts = BS.SimOptions(; fidelity = :eict, seed = 7, use_noise = false, use_scatter = false,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false)
    recon_opts = BS.ReconOptions(matrix_size = (16, 16, 2), fov_cm = 20.0)
    phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 16, n_slices = 2, fov_cm = 20.0, z_cm = 1.0))
    pipe = BSF.eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; view_batch)
    return pipe, BSF.onehot_fractions(phantom.mask, pipe.n_mat)
end
loss(fr, pipe, target) = sum((BSF.eict_forward(fr, pipe) .- target) .^ 2)
gradf(fr, pipe, target) = Enzyme.gradient(Reverse, loss, fr, Const(pipe), Const(target))

@testset "view batching under Reactant/Enzyme" begin
    p1, fr = fixture(; view_batch = 1); p4, _ = fixture(; view_batch = 4)
    hu_host = BSF.eict_forward(fr, p1)
    fr_r = Reactant.to_rarray(fr); tgt = Reactant.to_rarray(hu_host .+ 5f0)
    t1 = @elapsed f1 = @compile sync = true BSF.eict_forward(fr_r, p1)
    t4 = @elapsed f4 = @compile sync = true BSF.eict_forward(fr_r, p4)
    h1 = Array(f1(fr_r, p1)); h4 = Array(f4(fr_r, p4))
    ir = sprint(show, @code_hlo optimize = false BSF.eict_forward(fr_r, p4))
    println(@sprintf("VIEWBATCH IR: %d while loops, %d chars (unrolled: %d chars)", count("stablehlo.while", ir), length(ir),
        length(sprint(show, @code_hlo optimize = false BSF.eict_forward(fr_r, p1)))))
    @test count("stablehlo.while", ir) >= 1
    println(@sprintf("VIEWBATCH compile: per-view %.1f s, batched(4) %.1f s;  HU max|Δ| batched-vs-perview %.2e, perview-vs-host %.2e",
        t1, t4, maximum(abs.(h4 .- h1)), maximum(abs.(h1 .- hu_host))))
    @test maximum(abs.(h4 .- h1)) <= 5e-3
    g1c = @compile sync = true gradf(fr_r, p1, tgt); g4c = @compile sync = true gradf(fr_r, p4, tgt)
    g1 = Array(g1c(fr_r, p1, tgt)[1]); g4 = Array(g4c(fr_r, p4, tgt)[1])
    println(@sprintf("VIEWBATCH gradient: max|Δ| %.2e (|g| max %.2e)", maximum(abs.(g4 .- g1)), maximum(abs.(g1))))
    @test maximum(abs.(g4 .- g1)) <= 1e-4 * maximum(abs.(g1))
end
println("SMOKE_VIEW_BATCH_DONE")
