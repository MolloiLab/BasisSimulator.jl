# Bisect the constant HU offset of the COMPILED chain against the host oracle.  The host functional
# model (`eict_forward` on plain arrays) matches the legacy kernels to ~0.01 HU on the GE Revolution
# setup, but the compiled batch driver came out ~-31.7 HU everywhere inside the FOV.  This isolates
# WHERE: host batched composition | @compile of the whole-scan program | compile_pipeline driver |
# one batch stage-by-stage, on a chosen backend.
#   BACKEND=gpu NV=32 NS=2 R=32 NZ=2 VIEWS=40 julia --project=envs/reactant design/reactant/probes/bisect_compiled_offset.jl
using Reactant, BasisSimulator, Printf, Statistics
BACK = get(ENV, "BACKEND", "gpu"); Reactant.set_default_backend(BACK)
const BS = BasisSimulator; const BSF = BS.Functional
NV = parse(Int, get(ENV, "NV", "32"));  NS = parse(Int, get(ENV, "NS", "2"))
R  = parse(Int, get(ENV, "R", "32"));   NZ = parse(Int, get(ENV, "NZ", "2"))
VIEWS = parse(Int, get(ENV, "VIEWS", "40")); BUDGET = parse(Int, get(ENV, "BUDGET", "16384"))
FULLPROG = get(ENV, "FULLPROG", "1") == "1"
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
say("phantom $(size(ph.mask)) recon $(recon.matrix_size) views $VIEWS | batching dd=$(pipe.batching.dd) windowed=$(pipe.batching.col_block < 834)")
circ = [hypot(i - (R + 1) / 2, j - (R + 1) / 2) < 0.47R for i in 1:R, j in 1:R, k in 1:NZ]
st(a, b) = (d = (a .- b)[circ]; @sprintf("mean %+9.4f  rms %8.4f  max %8.4f", mean(d), sqrt(mean(abs2, d)), maximum(abs.(d))))
ref = BSF.eict_forward(fr, pipe)
say("HOST ORACLE (eict_forward, plain arrays)        : baseline, mean HU inside circle $(round(mean(ref[circ]), digits=2))")
say("host batched (eict_forward_batched) vs oracle : " * st(BSF.eict_forward_batched(fr, pipe), ref))
if FULLPROG
    f1 = Reactant.@compile sync = true BSF.eict_forward(fr_r, pipe)
    say("@compile eict_forward (one program)  vs oracle : " * st(Array(f1(fr_r, pipe)), ref))
end
cp = BSF.compile_pipeline(pipe, fr_r)
say("compile_pipeline forward (driver)    vs oracle : " * st(Array(BSF.forward(cp, fr_r)), ref))
# one batch, stage by stage: host vs compiled
b = cp.batches[1]; dh = BSF.batch_data(b, pipe); dr = Reactant.to_rarray(dh)
vh = BSF.eict_batch_vol(fr, pipe, b, dh)
gv = Reactant.@compile sync = true ((x, d) -> BSF.eict_batch_vol(x, pipe, b, d))(fr_r, dr)
vr = Array(gv(fr_r, dr))
say(@sprintf("batch-1 eict_batch_vol: host sum %.6e  compiled sum %.6e  maxabs diff %.3e  mean diff %+.3e", sum(vh), sum(vr), maximum(abs.(vr .- vh)), mean(vr .- vh)))
# and the HU map alone
th = BSF.eict_vol_to_hu(vh, pipe); tr = Array((Reactant.@compile sync = true BSF.eict_vol_to_hu(Reactant.to_rarray(vh), pipe))(Reactant.to_rarray(vh), pipe))
say(@sprintf("eict_vol_to_hu on the SAME host volume: host vs compiled maxabs diff %.3e mean %+.3e", maximum(abs.(tr .- th)), mean(tr .- th)))
say("DONE")
