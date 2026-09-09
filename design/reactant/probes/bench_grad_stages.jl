# Cumulative per-stage gradient cost under Reactant/Enzyme (64/100 by default):
# paths → log sinogram → filtered → backprojected (gather | dense) → HU.  Each line compiles the
# forward up to that stage and the gradient of sum(x²) w.r.t. the fractions.
#   NV=64 VIEWS=100 VB=25 julia --project=envs/reactant -t 4 --heap-size-hint=5G design/reactant/probes/bench_grad_stages.jl
using Reactant, Enzyme, BasisSimulator, Printf
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
pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = VB)
b = pipe.batching
say("batching: $(b)")
paths(f) = BSF.material_paths(f, pipe)
sino(f) = (P = paths(f); BSF.eict_chain(P, BSF._eict_on_device(pipe.eict, P); view_batch = b.spectral, loop = true))
filt(f) = (s = sino(f); BSF.filter_views(s, BSF._fbp_on_device(pipe.fbp, s)))
# lift the plan with the sinogram (a traced array; the filtered output may be a wrapper the device hook ignores)
bp_gather(f) = (s = sino(f); pl = BSF._fbp_on_device(pipe.fbp, s); BSF.backproject(BSF.filter_views(s, pl), pl; view_batch = b.fdk, loop = true))
hu(f) = BSF.eict_forward(f, pipe)
for (name, fn) in (("paths", paths), ("+eict_chain", sino), ("+filter", filt), ("+backproject", bp_gather), ("HU (pipeline)", hu))
    loss = f -> sum(fn(f) .^ 2)
    g = f -> Enzyme.gradient(Reverse, loss, f)
    tc = @elapsed cf = @compile sync = true fn(fr_r); cf(fr_r); tf = @elapsed cf(fr_r)
    tc2 = @elapsed cg = @compile sync = true g(fr_r); cg(fr_r); tg = @elapsed cg(fr_r)
    say(@sprintf("%-22s forward %.3f s (compile %.0f s) | gradient %.3f s (compile %.0f s) → ratio %.1f", name, tf, tc, tg, tc2, tg / tf))
end
say("GRAD_STAGES_DONE")
