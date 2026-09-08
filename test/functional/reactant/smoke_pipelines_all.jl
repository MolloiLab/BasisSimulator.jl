# Reactant + Enzyme smoke for the THREE five-struct pipelines with compiled view loops:
#   eict (HU), pcct (per-channel μ), vmi (dual-kVp n-channel VMI stack) — compile the forward,
#   compare with the host run, and compile the Enzyme gradient of a scalar loss w.r.t. the fractions.
#   until mkdir /tmp/bs_reactant.lock 2>/dev/null; do sleep 30; done; trap 'rmdir /tmp/bs_reactant.lock' EXIT
#   julia --project=envs/reactant -t 2 --heap-size-hint=4G test/functional/reactant/smoke_pipelines_all.jl
using Test, Statistics, LinearAlgebra, Random, Printf
using Reactant, Enzyme
using BasisSimulator
const BS = BasisSimulator
const BSF = BasisSimulator.Functional
T0 = time(); say(m) = (println(@sprintf("[%6.1f s] ", time() - T0), m); flush(stdout))

recon_opts = BS.ReconOptions(matrix_size = (16, 16, 2), fov_cm = 20.0)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 16, n_slices = 2, fov_cm = 20.0, z_cm = 1.0))
opts = BS.SimOptions(; use_noise = false, use_scatter = false, use_lag = false, use_focal_spot = false, use_optical_crosstalk = false)
eict = BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4, detector_cols = 240,
    detector_row_size = 1.0, detector_col_size = 1.0, detector_material = :lumex, detector_depth = 3.0, electronic_noise = 0.0, detection_gain = 10.0)
pcct = BS.PCCTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4, detector_cols = 240,
    detector_row_size = 1.0, detector_col_size = 1.0, detector_material = :CdTe, detector_depth = 1.6,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0], dead_time_ns = 25.0, pileup = false)
proto(kvp, mA) = BS.CTProtocol(mA = mA, kVp = kvp, views = 16, rotation_time = 0.5)

function check(name, pipe, fwd, fr; grad = true)
    fr_r = Reactant.to_rarray(fr)
    host = fwd(fr, pipe)
    t = @elapsed c = @compile sync = true fwd(fr_r, pipe)
    dev = Array(c(fr_r, pipe))
    say(@sprintf("%-5s forward compile %.0f s; compiled vs host max abs %.3e (range %.3g..%.3g)", name, t, maximum(abs.(dev .- host)), minimum(host), maximum(host)))
    @test maximum(abs.(dev .- host)) <= 1e-2 * max(1.0, maximum(abs.(host)))
    if grad
        tgt = Reactant.to_rarray(host .+ eltype(host)(0.01) .* maximum(abs.(host)))
        loss(f, p, t) = sum((fwd(f, p) .- t) .^ 2)
        g(f, p, t) = Enzyme.gradient(Reverse, loss, f, Const(p), Const(t))
        tg = @elapsed cg = @compile sync = true g(fr_r, pipe, tgt)
        gr = Array(cg(fr_r, pipe, tgt)[1])
        say(@sprintf("%-5s gradient compile %.0f s; |g| max %.3e, finite %s", name, tg, maximum(abs.(gr)), all(isfinite, gr)))
        @test all(isfinite, gr) && maximum(abs.(gr)) > 0
    end
end

@testset "eict" begin
    pipe = BSF.pipeline(phantom, eict, proto(120, 200.0), opts, recon_opts; view_batch = 2)
    fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat)
    check("eict", pipe, BSF.forward, fr)
end
@testset "pcct" begin
    pipe = BSF.pipeline(phantom, pcct, proto(120, 2.5), opts, recon_opts; view_batch = 2, groups = [[1, 2], [3, 4]])
    fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat)
    check("pcct", pipe, BSF.forward, fr)
end
@testset "vmi" begin
    vp = BSF.vmi_pipeline(phantom, eict, [proto(80, 300.0), proto(140, 150.0)], opts, recon_opts; energies = [50.0, 70.0], view_batch = 2)
    fr = BSF.onehot_fractions(phantom.mask, vp.pipes[1].n_mat)
    vmi_fwd(f, p) = BSF.vmi_forward(f, p).vmis
    check("vmi", vp, vmi_fwd, fr)
end
say("SMOKE_PIPELINES_ALL_DONE")
