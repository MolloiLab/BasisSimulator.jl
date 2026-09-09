# The compiled EICT driver (compile_pipeline / forward / pullback): forward parity with the host
# oracle, pullback parity with Enzyme through the single while-loop program (same Float32 math,
# different summation order), and a Float64 finite-difference check of one directional derivative.
#   julia --project=envs/reactant -t 4 --heap-size-hint=5G test/functional/reactant/smoke_compiled.jl
using Reactant, Enzyme, BasisSimulator, Printf, Random
const BS = BasisSimulator; const BSF = BS.Functional
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = 40, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon_opts = BS.ReconOptions(matrix_size = (32, 32, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 32, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
fr = BSF.onehot_fractions(phantom.mask, length(phantom.materials)); fr_r = Reactant.to_rarray(fr)
pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = 8)
ok = true
tc = @elapsed cp = BSF.compile_pipeline(pipe, fr_r)
say(@sprintf("compile_pipeline: %d batches, %d forward + %d pullback programs, %.0f s", length(cp.batches), length(cp.fwd), length(cp.vjp), tc))
ref = BSF.eict_forward(fr, pipe)
hu = Array(BSF.forward(cp, fr_r)); e1 = maximum(abs.(hu .- ref)); global ok &= e1 < 0.1
say(@sprintf("forward vs host oracle: max |Δ| %.2e HU %s", e1, e1 < 0.1 ? "ok" : "FAIL"))
tgt = ref .+ 5f0; tgt_r = Reactant.to_rarray(tgt)
hu2, back = BSF.pullback(cp, fr_r)
hbar = Reactant.to_rarray(2f0 .* (Array(hu2) .- tgt))
g_host = Array(back(hbar))
loss(f, p, t) = sum((BSF.eict_forward(f, p) .- t) .^ 2)
gl = @jit Enzyme.gradient(Reverse, loss, fr_r, Const(pipe), Const(tgt_r))
g_loop = Array(gl isa Tuple ? gl[1] : gl)
e2 = maximum(abs.(g_host .- g_loop)) / maximum(abs.(g_loop)); global ok &= e2 < 1e-3
say(@sprintf("pullback vs loop-program Enzyme gradient: max rel %.2e %s", e2, e2 < 1e-3 ? "ok" : "FAIL"))
# Float64 directional finite difference through the HOST pipeline vs the compiled gradient
let dir = randn(MersenneTwister(1), Float32, size(fr)); dir ./= maximum(abs.(dir))
    p64 = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = 8, T = Float64)
    l64(x) = sum((BSF.eict_forward(Float64.(x), p64) .- Float64.(tgt)) .^ 2)
    h = 1e-3; fd = (l64(fr .+ h .* dir) - l64(fr .- h .* dir)) / (2h)
    an = sum(Float64.(g_host) .* Float64.(dir))
    e3 = abs(an - fd) / abs(fd); global ok &= e3 < 2e-2
    say(@sprintf("directional derivative: compiled %.6e  Float64 finite-diff %.6e  rel %.2e %s", an, fd, e3, e3 < 2e-2 ? "ok" : "FAIL"))
end
say(ok ? "SMOKE_COMPILED_OK" : "SMOKE_COMPILED_FAIL")
