# Where does the GRADIENT time go?  64/100, dense projector: loop (checkpointed) vs unrolled batches,
# forward-only vs gradient, plus the gradient of the DD stage alone.
#   NV=64 VIEWS=100 VB=25 julia --project=envs/reactant -t 4 --heap-size-hint=5G design/reactant/probes/bench_grad.jl
using Reactant, Enzyme, BasisSimulator, Statistics, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); VB = envi("VB", 25)
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon_opts = BS.ReconOptions(matrix_size = (NV, NV, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
fr = BSF.onehot_fractions(phantom.mask, length(phantom.materials)); fr_r = Reactant.to_rarray(fr)
for (name, loop) in (("loop", true), ("unrolled", false))
    pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = VB, loop)
    # DD stage alone
    dd(f, p) = sum(BSF.material_paths(f, p) .^ 2)
    gdd(f, p) = Enzyme.gradient(Reverse, dd, f, Const(p))
    t = @elapsed cf = @compile sync = true BSF.material_paths(fr_r, pipe); cf(fr_r, pipe); tf = @elapsed cf(fr_r, pipe)
    t2 = @elapsed cg = @compile sync = true gdd(fr_r, pipe); cg(fr_r, pipe); tg = @elapsed cg(fr_r, pipe)
    say(@sprintf("%-8s DD stage: forward %.2f s (compile %.0f s) | gradient %.2f s (compile %.0f s) → ratio %.1f", name, tf, t, tg, t2, tg / tf))
    # full pipeline
    tgt = Reactant.to_rarray(BSF.eict_forward(fr, pipe))
    loss(f, p, t) = sum((BSF.eict_forward(f, p) .- t) .^ 2)
    g(f, p, t) = Enzyme.gradient(Reverse, loss, f, Const(p), Const(t))
    t = @elapsed cf2 = @compile sync = true BSF.eict_forward(fr_r, pipe); cf2(fr_r, pipe); tf2 = @elapsed cf2(fr_r, pipe)
    t2 = @elapsed cg2 = @compile sync = true g(fr_r, pipe, tgt); cg2(fr_r, pipe, tgt); tg2 = @elapsed cg2(fr_r, pipe, tgt)
    say(@sprintf("%-8s pipeline: forward %.2f s (compile %.0f s) | gradient %.2f s (compile %.0f s) → ratio %.1f", name, tf2, t, tg2, t2, tg2 / tf2))
end
say("GRAD_BENCH_DONE")
