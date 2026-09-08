# Scaling probe for view batching: the GE-Revolution-class arc scanner, a Gammex 472 phantom on an
# NV×NV×2 grid (45 cm FOV), VIEWS views, RECON×RECON×2 recon, pipeline compiled with view_batch = VB.
#   NV=128 VIEWS=200 RECON=128 VB=25 GRAD=1 julia --project=envs/reactant -t 2 --heap-size-hint=4G design/reactant/probes/scale_view_batch.jl
using Reactant, Enzyme, Statistics, Printf
using BasisSimulator
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 128); VIEWS = envi("VIEWS", 200); RECON = envi("RECON", 128); VB = envi("VB", 25); GRAD = envi("GRAD", 1); LEGACY = envi("LEGACY", 1)
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
say("config NV=$NV VIEWS=$VIEWS RECON=$RECON VB=$VB GRAD=$GRAD")
scanner = BS.Scanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9,
    electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
simo = BS.SimOptions(fidelity = :eict, seed = 1234, projector = :dd_fast, use_noise = false, use_scatter = false,
    use_focal_spot = false, use_optical_crosstalk = false, use_lag = false)
recon_opts = BS.ReconOptions(matrix_size = (RECON, RECON, 2), fov_cm = 35.0, z_cm = 0.5)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
say("phantom $(size(phantom.mask)), $(length(phantom.materials)) materials")
t = @elapsed pipe = BSF.eict_pipeline(phantom, scanner, protocol, simo, recon_opts; view_batch = VB)
nb = length(BSF.dd_batch_plans(pipe.geom, pipe.vol_shape; view_batch = VB, volume_extent = pipe.volume_extent, eltype = Float32))
say(@sprintf("pipeline built in %.1f s: sino %s, %d view batches, taps KX·KZ max %d", t, string(pipe.eict.sino_shape), nb,
    maximum(bp.KX * bp.KZ for bp in BSF.dd_batch_plans(pipe.geom, pipe.vol_shape; view_batch = VB, volume_extent = pipe.volume_extent, eltype = Float32))))
fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat); fr_r = Reactant.to_rarray(fr)
t_c = @elapsed fwd = @compile sync = true BSF.eict_forward(fr_r, pipe)
say(@sprintf("forward compiled in %.1f s", t_c))
hu = Array(fwd(fr_r, pipe)); t_f = @elapsed fwd(fr_r, pipe)
say(@sprintf("forward run %.2f s; HU range %.0f..%.0f", t_f, minimum(hu), maximum(hu)))
if LEGACY == 1
    ws = BS.create_eict_workspace(scanner, protocol, simo, recon_opts, phantom)
    t_l = @elapsed BS.simulate!(ws, phantom, protocol, simo)
    model = BS.calibrate_bhc_water(simo, protocol; scanner, geom = ws.geom)
    sino_bhc = BS.apply_bhc_water(ws.sinogram, model)
    ws_fdk = BS.create_fdk_recon_workspace(sino_bhc, ws.geom, recon_opts.matrix_size)
    hu_l = Float32.(BS.to_hounsfield(copy(BS.reconstruct!(ws_fdk, sino_bhc, ws.geom)); μ_water = model.μ_water_ref))
    say(@sprintf("legacy simulate! %.1f s; twin vs legacy: max|Δ| %.3f HU, rms %.4f HU (range %.0f..%.0f)", t_l,
        maximum(abs.(hu .- hu_l)), sqrt(mean((hu .- hu_l) .^ 2)), minimum(hu_l), maximum(hu_l)))
end
if GRAD == 1
    tgt = Reactant.to_rarray(hu .+ 10f0)
    loss(f, p, t) = sum((BSF.eict_forward(f, p) .- t) .^ 2)
    gradf(f, p, t) = Enzyme.gradient(Reverse, loss, f, Const(p), Const(t))
    t_gc = @elapsed g = @compile sync = true gradf(fr_r, pipe, tgt)
    say(@sprintf("gradient compiled in %.1f s", t_gc))
    r = g(fr_r, pipe, tgt); t_g = @elapsed g(fr_r, pipe, tgt)
    say(@sprintf("gradient run %.2f s; |g| max %.3e", t_g, maximum(abs.(Array(r[1])))))
end
say("SCALE_DONE")
