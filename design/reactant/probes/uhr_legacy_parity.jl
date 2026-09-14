# HEAD-ON check at the real grid: legacy AcceleratedKernels scan (Array backend) vs the compiled
# twin (compile_pipeline, full-f32), inside the reconstructed circle.
#   VIEWS=24 BUDGET=32768 julia --project=envs/reactant design/reactant/probes/uhr_legacy_parity.jl
using Reactant, BasisSimulator, Printf, Statistics
const BS = BasisSimulator; const BSF = BS.Functional
T0 = time(); say(m) = (@printf("[%7.1f s] %s\n", time() - T0, m); flush(stdout))
VIEWS = parse(Int, get(ENV, "VIEWS", "24")); BUDGET = parse(Int, get(ENV, "BUDGET", "32768"))
scanner = BS.EICTScanner(source_to_isocenter=625.6, source_to_detector=1100.0, detector_rows=256, detector_cols=834,
    detector_row_size=0.625, detector_col_size=0.6, detector_shape=:arc, focal_spot_width=1.0, focal_spot_length=1.0,
    target_angle=10.0, flat_filter_material=:aluminum, flat_filter_thickness=2.5, bowtie_filter=:ge_revolution_large,
    detector_material=:lumex, detector_depth=3.0, fill_factor_row=0.9, fill_factor_col=0.9, electronic_noise=0, detection_gain=10.0)
protocol = BS.CTProtocol(kVp=120, mA=200.0, views=VIEWS, rotation_time=1.0, collimation_mm=5.0, additional_filters=[("Al",4.5)])
opts = BS.SimOptions(use_noise=false, use_scatter=false, use_focal_spot=false, use_optical_crosstalk=false, use_lag=false, seed=1234)
R = 512; recon = BS.ReconOptions(matrix_size=(R,R,8), fov_cm=35.0, z_cm=0.5)
ph = BS.compact_materials(BS.create_gammex_472(n_voxels=1125, n_slices=16, fov_cm=45.0, z_cm=1.0))
tl = @elapsed begin
    ws = BS.create_eict_workspace(scanner, protocol, opts, recon, ph); BS.simulate!(ws, ph, protocol, opts)
    sino = Array(ws.sinogram); geom = ws.geom; ws = nothing; GC.gc()
    model = BS.calibrate_bhc_water(opts, protocol; scanner, geom)
    sb = BS.apply_bhc_water(sino, model)
    hu_leg = Float32.(BS.to_hounsfield(copy(BS.reconstruct!(BS.create_fdk_recon_workspace(sb, geom, recon.matrix_size), sb, geom)); μ_water=model.μ_water_ref))
end
say(@sprintf("legacy scan (Array): %.1f s for %d views", tl, VIEWS))
pipe = BSF.eict_pipeline(ph, scanner, protocol, opts, recon; batch_budget_mb=BUDGET)
fr_r = Reactant.to_rarray(BSF.onehot_fractions(ph.mask, pipe.n_mat))
tc = @elapsed cp = BSF.compile_pipeline(pipe, fr_r)
hu = Array(BSF.forward(cp, fr_r)); tf = @elapsed Array(BSF.forward(cp, fr_r))
circ = [hypot(i-(R+1)/2, j-(R+1)/2) < 0.47R for i in 1:R, j in 1:R, k in 1:8]
d = (hu .- hu_leg)[circ]
say(@sprintf("COMPILED vs LEGACY at 1125^2x16 inside FOV: mean %+.4f HU  rms %.4f HU  max %.4f HU | compile %.0f s, forward %.3f s (%.1f ms/view)",
    mean(d), sqrt(mean(abs2, d)), maximum(abs.(d)), tc, tf, 1000tf/VIEWS))
say("DONE")
