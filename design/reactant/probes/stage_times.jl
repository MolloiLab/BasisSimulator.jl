# Per-stage run time of the compiled pipeline (loops), to locate the CPU bottleneck.
#   NV=64 VIEWS=100 RECON=64 VB=25 julia --project=envs/reactant -t 4 --heap-size-hint=4G design/reactant/probes/stage_times.jl
using Reactant, BasisSimulator, Statistics, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); RECON = envi("RECON", 64); VB = envi("VB", 25)
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon_opts = BS.ReconOptions(matrix_size = (RECON, RECON, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon_opts; view_batch = VB)
fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat); fr_r = Reactant.to_rarray(fr)
say("materials $(pipe.n_mat), sino $(pipe.eict.sino_shape), taps " * string(maximum(r.KX * r.KZ for r in BSF.dd_run_plans(pipe.geom, pipe.vol_shape; view_batch = VB, volume_extent = pipe.volume_extent, eltype = Float32))))
f1 = @compile sync = true BSF.material_paths(fr_r, pipe); P = f1(fr_r, pipe); t = @elapsed f1(fr_r, pipe); say(@sprintf("DD path lengths (%d materials): %.2f s", pipe.n_mat, t))
chain(P, pipe) = BSF.eict_chain(P, BSF._eict_on_device(pipe.eict, P); view_batch = pipe.batching.spectral)
f2 = @compile sync = true chain(P, pipe); sino = f2(P, pipe); t = @elapsed f2(P, pipe); say(@sprintf("EICT chain (spectral sum + BHC): %.2f s", t))
f3 = @compile sync = true BSF.reconstruct_μ(sino, pipe); f3(sino, pipe); t = @elapsed f3(sino, pipe); say(@sprintf("FDK: %.2f s", t))
one_mat(fr, pipe) = BSF.material_paths(fr[:, :, :, 1:1], pipe)
f4 = @compile sync = true one_mat(fr_r, pipe); f4(fr_r, pipe); t = @elapsed f4(fr_r, pipe); say(@sprintf("DD path lengths, 1 material: %.2f s", t))
say("STAGES_DONE")
