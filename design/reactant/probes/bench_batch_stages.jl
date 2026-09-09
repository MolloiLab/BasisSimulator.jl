# One view batch as a loop-free program: cumulative stage forward vs pullback cost
# (DD → +chain → +filter → +backprojection) — where the per-batch pullback's time goes.
#   NV=256 VIEWS=984 BUDGET_MB=1024 julia --project=envs/reactant -t 4 --heap-size-hint=6G design/reactant/probes/bench_batch_stages.jl
using Reactant, Enzyme, BasisSimulator, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); BUDGET_MB = envi("BUDGET_MB", 512)
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
pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = :auto, batch_budget_mb = BUDGET_MB)
bs = BSF.eict_batches(pipe); b = bs[argmax(length(x.views) for x in bs)]
d = Reactant.to_rarray(BSF.batch_data(b, pipe))
say("batching $(pipe.batching); probing the $(length(b.views))-view batch $(b.views) (windowed $(b.windowed))")
dd(x, d) = b.windowed ?
    BSF.dd_project_dense_windowed_run(x, b.run, pipe.geom; col_block = pipe.batching.col_block, slab_block = pipe.batching.slab_block,
        volume_extent = pipe.volume_extent, unroll = true, table = d.table, windows = (d.starts, b.widths, b.col_blocks, b.slab_blocks)) :
    BSF.dd_project_dense_run(x, b.run; unroll = true, table = d.table)
chain(x, d) = (P = dd(x, d); BSF.eict_chain(P, BSF._eict_on_device(pipe.eict, P); view_batch = 0))
filt(x, d) = (s = chain(x, d); BSF.filter_views(s, BSF._fbp_on_device(pipe.fbp, s)))
bp(x, d) = BSF.eict_batch_vol(x, pipe, b, d)
for (name, fn) in (("DD", dd), ("+chain", chain), ("+filter", filt), ("+backprojection", bp))
    loss = (x, d) -> sum(fn(x, d) .^ 2)
    g = (x, d) -> Enzyme.gradient(Reverse, loss, x, Const(d))
    tc = @elapsed cf = @compile sync = true fn(fr_r, d); cf(fr_r, d); tf = minimum(@elapsed(cf(fr_r, d)) for _ in 1:3)
    tc2 = @elapsed cg = @compile sync = true g(fr_r, d); cg(fr_r, d); tg = minimum(@elapsed(cg(fr_r, d)) for _ in 1:3)
    say(@sprintf("%-16s forward %.3f s (compile %.0f s) | gradient %.3f s (compile %.0f s) → ratio %.1f", name, tf, tc, tg, tc2, tg / tf))
end
say("BATCH_STAGES_DONE")
