# Where does the time go on XLA CPU?  Same scan three ways at one size:
#   (a) legacy kernels on CPU arrays (AcceleratedKernels CPU backend), (b) compiled program with
#   unrolled batches (loop = false), (c) compiled program with while loops (loop = true);
#   forward run times + a gradient step for (b)/(c).  Outputs must agree.
#   NV=64 VIEWS=100 RECON=64 VB=25 julia --project=envs/reactant -t 4 --heap-size-hint=5G design/reactant/probes/bench_cpu.jl
using Reactant, Enzyme, BasisSimulator, Statistics, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); RECON = envi("RECON", 64); VB = envi("VB", 25); GRAD = envi("GRAD", 1); UNROLLED = envi("UNROLLED", 1)
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
say("config NV=$NV VIEWS=$VIEWS RECON=$RECON VB=$VB threads=$(Threads.nthreads())")
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon_opts = BS.ReconOptions(matrix_size = (RECON, RECON, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
say("phantom $(size(phantom.mask)), $(length(phantom.materials)) materials")
# (a) legacy on CPU arrays
ws = BS.create_eict_workspace(scanner, protocol, opts, recon_opts, phantom)
BS.simulate!(ws, phantom, protocol, opts)                                   # warm-up (JIT)
t_leg = @elapsed BS.simulate!(ws, phantom, protocol, opts)
model = BS.calibrate_bhc_water(opts, protocol; scanner, geom = ws.geom)
sino_bhc = BS.apply_bhc_water(ws.sinogram, model)
ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, ws.geom, recon_opts.matrix_size)
t_leg_rec = @elapsed BS.reconstruct!(ws_fdk, sino_bhc, ws.geom)
hu_leg = Float32.(BS.to_hounsfield(copy(BS.reconstruct!(ws_fdk, sino_bhc, ws.geom)); μ_water = model.μ_water_ref))
say(@sprintf("(a) legacy CPU: simulate! %.2f s + recon %.2f s  [mask backend %s]", t_leg, t_leg_rec, typeof(phantom.mask).name.name))
fr = BSF.onehot_fractions(phantom.mask, length(phantom.materials)); fr_r = Reactant.to_rarray(fr)
m = [hypot(i - (RECON + 1) / 2, j - (RECON + 1) / 2) < 0.47RECON for i in 1:RECON, j in 1:RECON, k in 1:2]
results = Dict{String, Any}()
for (name, loop) in (UNROLLED == 1 ? (("(b) unrolled batches", false), ("(c) while loops", true)) : (("(c) while loops", true),))
    pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = VB, loop)
    t_c = @elapsed fwd = @compile sync = true BSF.eict_forward(fr_r, pipe)
    hu = Array(fwd(fr_r, pipe)); t_f = @elapsed fwd(fr_r, pipe)
    say(@sprintf("%s: compile %.0f s, forward run %.2f s, vs legacy rms %.3f HU (max %.2f)", name, t_c, t_f,
        sqrt(mean((hu .- hu_leg)[m] .^ 2)), maximum(abs.((hu .- hu_leg)[m]))))
    results[name] = hu
    if GRAD == 1
        tgt = Reactant.to_rarray(hu_leg)
        loss(f, p, t) = sum((BSF.eict_forward(f, p) .- t) .^ 2)
        g(f, p, t) = Enzyme.gradient(Reverse, loss, f, Const(p), Const(t))
        t_gc = @elapsed cg = @compile sync = true g(fr_r, pipe, tgt)
        cg(fr_r, pipe, tgt); t_g = @elapsed cg(fr_r, pipe, tgt)
        say(@sprintf("%s: gradient compile %.0f s, gradient step %.2f s", name, t_gc, t_g))
    end
end
UNROLLED == 1 && say(@sprintf("(b) vs (c) max abs %.3e HU", maximum(abs.(results["(b) unrolled batches"] .- results["(c) while loops"]))))
say("BENCH_DONE")
