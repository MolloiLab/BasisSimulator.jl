# Reactant + Enzyme smoke for the END-TO-END functional EICT pipeline:
#   phantom material fractions → DD path lengths → EICT chain (+BHC) → FDK → HU
# compiled as ONE XLA program and differentiated with Enzyme w.r.t. the fractions.
#
# Run (serialized system-wide, one Julia process, toy size):
#   until mkdir /tmp/bs_reactant.lock 2>/dev/null; do sleep 30; done
#   trap 'rmdir /tmp/bs_reactant.lock' EXIT
#   julia --project=envs/reactant -t 2 --heap-size-hint=3G test/functional/reactant/smoke_pipeline.jl
using Test, Statistics, LinearAlgebra, Random
using Reactant, Enzyme
using BasisSimulator
const BS = BasisSimulator
const BSF = BasisSimulator.Functional

function fixture(; use_noise, T)
    scanner = BS.Scanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 32, detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 8, rotation_time = 0.5)
    sim_opts = BS.SimOptions(; fidelity = :eict, seed = 7, use_noise, use_scatter = false,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false)
    recon_opts = BS.ReconOptions(matrix_size = (16, 16, 2), fov_cm = 20.0)
    phantom = BS.create_gammex_472(n_voxels = 16, n_slices = 2, fov_cm = 20.0, z_cm = 1.0)
    pipe = BSF.eict_pipeline(phantom, scanner, protocol, sim_opts, recon_opts; T)
    fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat; T)
    ε, ε_e = BSF.draw_eict_noise(pipe; seed = 7)
    return pipe, fr, ε, ε_e
end

relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(Float32))
timed(f) = (t = @elapsed r = f(); (r, t))

@testset "pipeline: Float32 compile parity (noise on)" begin
    pipe, fr, ε, ε_e = fixture(; use_noise = true, T = Float32)
    hu_plain = BSF.eict_forward(fr, pipe, ε, ε_e)
    fr_r = Reactant.to_rarray(fr); ε_r = Reactant.to_rarray(ε); εe_r = Reactant.to_rarray(ε_e)
    fwd = (fr, ε, ε_e) -> BSF.eict_forward(fr, pipe, ε, ε_e)
    (thunk, tc) = timed(() -> @compile sync = true fwd(fr_r, ε_r, εe_r))
    hu_c = Array(thunk(fr_r, ε_r, εe_r))
    (_, tr) = timed(() -> thunk(fr_r, ε_r, εe_r))
    r = relmax(hu_c, hu_plain)
    println("PIPE  Float32 e2e: compile $(round(tc; digits = 1)) s, run $(round(tr * 1000; digits = 1)) ms, parity relmax $r  (plain Julia $(round(1000 * (@elapsed BSF.eict_forward(fr, pipe, ε, ε_e)); digits = 1)) ms)")
    @test r < 1e-4
    @test all(isfinite, hu_c)
end

@testset "pipeline: Enzyme reverse gradient w.r.t. material fractions (Float64)" begin
    pipe, fr, ε, ε_e = fixture(; use_noise = false, T = Float64)
    rng = MersenneTwister(1)
    w = randn(rng, Float64, pipe.recon_shape)
    loss(fr) = sum(w .* BSF.eict_forward(fr, pipe))
    g_of(fr) = Enzyme.gradient(Reverse, loss, fr)[1]
    fr_r = Reactant.to_rarray(fr)
    (gthunk, tc) = timed(() -> @compile sync = true g_of(fr_r))
    g = Array(gthunk(fr_r))
    (_, tr) = timed(() -> gthunk(fr_r))
    @test size(g) == size(fr) && all(isfinite, g) && any(!=(0), g)
    # directional derivative vs plain-Array central differences
    d = randn(rng, Float64, size(fr)); d ./= norm(d)
    h = 1e-4
    fd = (loss(fr .+ h .* d) - loss(fr .- h .* d)) / (2h)
    ad = dot(g, d)
    rel = abs(ad - fd) / max(abs(fd), 1e-12)
    println("PIPE  Float64 gradient: compile $(round(tc; digits = 1)) s, run $(round(tr * 1000; digits = 1)) ms, ⟨∇,d⟩ AD $ad vs FD $fd  rel $rel")
    @test rel < 1e-3
    # a couple of per-entry probes on voxels inside the FOV
    for idx in (CartesianIndex(8, 8, 1, 2), CartesianIndex(6, 10, 2, 1))
        e = zeros(size(fr)); e[idx] = 1
        fd_i = (loss(fr .+ h .* e) - loss(fr .- h .* e)) / (2h)
        @test abs(g[idx] - fd_i) <= 1e-3 * max(abs(fd_i), 1e-6) + 1e-9
    end
end
