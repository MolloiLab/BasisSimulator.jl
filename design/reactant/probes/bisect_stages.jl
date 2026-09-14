# Stage-by-stage bisect INSIDE eict_batch_vol for ONE view batch: each compiled stage is fed the
# HOST output of the previous stage, so a discrepancy cannot compound and localizes to one stage.
#   BACKEND=gpu NV=32 NS=2 R=32 NZ=2 VIEWS=40 julia --project=envs/reactant design/reactant/probes/bisect_stages.jl
using Reactant, BasisSimulator, Printf, Statistics
BACK = get(ENV, "BACKEND", "gpu"); Reactant.set_default_backend(BACK)
const BS = BasisSimulator; const BSF = BS.Functional
NV = parse(Int, get(ENV, "NV", "32"));  NS = parse(Int, get(ENV, "NS", "2"))
R  = parse(Int, get(ENV, "R", "32"));   NZ = parse(Int, get(ENV, "NZ", "2"))
VIEWS = parse(Int, get(ENV, "VIEWS", "40")); BUDGET = parse(Int, get(ENV, "BUDGET", "16384"))
T0 = time(); say(m) = (@printf("[%s][%7.1f s] %s\n", BACK, time() - T0, m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon = BS.ReconOptions(matrix_size = (R, R, NZ), fov_cm = 35.0, z_cm = 0.5)
ph = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = NS, fov_cm = 45.0, z_cm = 1.0))
fr = BSF.onehot_fractions(ph.mask, length(ph.materials)); fr_r = Reactant.to_rarray(fr)
pipe = BSF.eict_pipeline(ph, scanner, protocol, opts, recon; batch_budget_mb = BUDGET)
bs = BSF.eict_batches(pipe); b = bs[1]; dh = BSF.batch_data(b, pipe); dr = Reactant.to_rarray(dh)
say("batch 1 of $(length(bs)): views $(b.views), windowed=$(b.windowed)")
cmp(name, h, c) = say(@sprintf("%-34s host|c| %.4e  maxabs diff %.3e  mean diff %+.3e  rel %.2e", name, maximum(abs.(h)), maximum(abs.(c .- h)), mean(c .- h), maximum(abs.(c .- h)) / max(maximum(abs.(h)), 1e-30)))
proj(x, d) = b.windowed ?
    BSF.dd_project_dense_windowed_run(x, b.run, pipe.geom; col_block = pipe.batching.col_block, slab_block = pipe.batching.slab_block,
        volume_extent = pipe.volume_extent, unroll = true, table = d.table, windows = (d.starts, b.widths, b.col_blocks, b.slab_blocks)) :
    BSF.dd_project_dense_run(x, b.run; unroll = true, table = d.table)
# 1. projector
Ph = proj(fr, dh)
Pc = Array((Reactant.@compile sync = true ((x, d) -> proj(x, d))(fr_r, dr))(fr_r, dr))
cmp("1 DD projector P", Ph, Pc)
# 2. EICT chain fed the HOST P
chain(P) = BSF.eict_chain(P, BSF._eict_on_device(pipe.eict, P), nothing, nothing; view_batch = 0)
sh = chain(Ph); Ph_r = Reactant.to_rarray(Ph)
sc = Array((Reactant.@compile sync = true chain(Ph_r))(Ph_r))
cmp("2 eict_chain s (given host P)", sh, sc)
# 3. ramp filter fed the HOST s
filt(s) = BSF.filter_views(s, BSF._fbp_on_device(pipe.fbp, s))
fh = filt(sh); sh_r = Reactant.to_rarray(sh)
fc = Array((Reactant.@compile sync = true filt(sh_r))(sh_r))
cmp("3 filter_views (given host s)", fh, fc)
# 4. backprojection fed the HOST filtered views
bp(f, d) = (fp = BSF._fbp_on_device(pipe.fbp, f); dropdims(BSF._bp_chunk(vec(f), fp, d.gt, d.off, true); dims = 4) .* fp.pi_over_angles)
vh = bp(fh, dh); fh_r = Reactant.to_rarray(fh)
vc = Array((Reactant.@compile sync = true ((f, d) -> bp(f, d))(fh_r, dr))(fh_r, dr))
cmp("4 _bp_chunk vol (given host filt)", vh, vc)
# 5. HU map fed the HOST volume
hh = BSF.eict_vol_to_hu(vh, pipe); vh_r = Reactant.to_rarray(vh)
hc = Array((Reactant.@compile sync = true BSF.eict_vol_to_hu(vh_r, pipe))(vh_r, pipe))
cmp("5 eict_vol_to_hu (given host vol)", hh, hc)
say("DONE")
