# Windowed vs full dense projector under Reactant at one size: compile, run, parity, and the
# per-view weight memory each needs.
#   NV=128 VIEWS=200 CB=64 SB=32 julia --project=envs/reactant -t 4 --heap-size-hint=5G design/reactant/probes/bench_windowed.jl
using Reactant, BasisSimulator, Statistics, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 128); VIEWS = envi("VIEWS", 200); CB = envi("CB", 64); SB = envi("SB", 32); VB = envi("VB", 4)
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon_opts = BS.ReconOptions(matrix_size = (NV, NV, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = VB)
geom = pipe.geom; T = Float32
fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat); fr_r = Reactant.to_rarray(fr)
runs = BSF.dd_run_plans(geom, pipe.vol_shape; view_batch = VB, volume_extent = pipe.volume_extent, eltype = T)
_, widths, cbs, sbs = BSF.dd_windows(geom, pipe.vol_shape, 1:VIEWS, CB, SB; volume_extent = pipe.volume_extent, eltype = T)
say(@sprintf("NV=%d VIEWS=%d: full dense %.0f MB/view; windowed (cb %d × sb %d → %d×%d blocks, widths %d..%d of %d) %.0f MB/view", NV, VIEWS,
    BSF.dd_dense_view_bytes(geom.n_cols, geom.n_rows, NV, 2, NV, pipe.n_mat) / 2^20, CB, SB, length(cbs), length(sbs), minimum(widths), maximum(widths), NV,
    BSF.dd_dense_windowed_view_bytes(geom.n_cols, geom.n_rows, 2, NV, pipe.n_mat, widths, CB, SB) / 2^20))
full(fr) = reduce((a, b) -> cat(a, b; dims = 3), [BSF.dd_project_dense_run(fr, r) for r in runs])
win(fr) = reduce((a, b) -> cat(a, b; dims = 3), [BSF.dd_project_dense_windowed_run(fr, r, geom; col_block = CB, slab_block = SB, volume_extent = pipe.volume_extent) for r in runs])
t = @elapsed cf = @compile sync = true full(fr_r); Pf = Array(cf(fr_r)); tf = @elapsed cf(fr_r)
say(@sprintf("full dense:     compile %.0f s, run %.2f s", t, tf))
t = @elapsed cw = @compile sync = true win(fr_r); Pw = Array(cw(fr_r)); tw = @elapsed cw(fr_r)
say(@sprintf("windowed dense: compile %.0f s, run %.2f s; vs full max rel %.2e", t, tw, maximum(abs.(Pw .- Pf)) / maximum(abs.(Pf))))
say("WINDOWED_DONE")
