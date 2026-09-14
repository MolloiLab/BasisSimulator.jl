# Compiled driver vs host oracle ON THE DEFAULT BACKEND at the GE Revolution geometry (arc detector,
# bowtie), asserting parity inside the reconstructed circle. On a CUDA device this is the test that
# catches TF32: with `precision = :default` XLA:GPU runs the chain's matmuls at 10-bit mantissa and
# the ramp filter's Toeplitz product biases the whole reconstruction by ~-31 HU; `compile_pipeline`'s
# default `:highest` brings it to ~0.001 HU. On CPU both pass (no TF32 path) — the test is only
# informative on GPU, so it also REPORTS the default-precision offset rather than hiding it.
#   julia --project=envs/reactant -t 4 --heap-size-hint=8G test/functional/reactant/smoke_gpu_parity.jl
using Reactant, BasisSimulator, Printf, Statistics
const BS = BasisSimulator; const BSF = BS.Functional
T0 = time(); say(m) = (println(@sprintf("[%7.1f s] ", time() - T0), m); flush(stdout))
scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256, detector_cols = 834,
    detector_row_size = 0.625, detector_col_size = 0.6, detector_shape = :arc, focal_spot_width = 1.0, focal_spot_length = 1.0,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = 40, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
R = 32; recon = BS.ReconOptions(matrix_size = (R, R, 2), fov_cm = 35.0, z_cm = 0.5)
ph = BS.compact_materials(BS.create_gammex_472(n_voxels = 32, n_slices = 2, fov_cm = 45.0, z_cm = 1.0))
fr = BSF.onehot_fractions(ph.mask, length(ph.materials)); fr_r = Reactant.to_rarray(fr)
pipe = BSF.eict_pipeline(ph, scanner, protocol, opts, recon; view_batch = 8)
circ = [hypot(i - (R + 1) / 2, j - (R + 1) / 2) < 0.47R for i in 1:R, j in 1:R, k in 1:2]
ref = BSF.eict_forward(fr, pipe)
dev = try string(Reactant.XLA.device_kind(first(Reactant.devices()))) catch; "?" end
say("backend device: $dev")
ok = true
for prec in (:highest, :default)
    cp = withenv("BASISSIM_ALLOW_TF32" => (prec === :default ? "1" : "0")) do   # :default is refused without this
        BSF.compile_pipeline(pipe, fr_r; precision = prec)
    end
    d = (Array(BSF.forward(cp, fr_r)) .- ref)[circ]
    m, mx = mean(d), maximum(abs.(d))
    pass = abs(m) < 0.1 && mx < 0.5          # bias is the TF32 signature (~-31 HU); max absorbs summation-order drift
    prec === :highest && (global ok &= pass)
    say(@sprintf("precision = %-8s vs host oracle inside FOV: mean %+8.4f HU, max |Δ| %.4f HU %s", prec, m, mx,
        prec === :highest ? (pass ? "ok" : "FAIL") : (pass ? "(no TF32 on this backend)" : "(TF32 bias — expected on GPU, and why :highest is the default)")))
end
say(ok ? "PASS" : "FAIL"); exit(ok ? 0 : 1)
