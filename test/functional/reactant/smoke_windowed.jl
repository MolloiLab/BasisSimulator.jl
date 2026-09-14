# The WINDOWED dense projector under Reactant — forward parity and gradient parity.
#
# This path had ZERO traced coverage: `smoke_compiled.jl` builds its pipeline with an integer
# `view_batch`, which takes the non-`:auto` branch (col_block = n_cols, slab_block = n_long) and so
# never sets `windowed`, while `test_batches.jl` reaches the windowed projector but only on plain
# host arrays and only checks forward VALUES. A windowed projector that is forward-exact and
# adjoint-broken passed everything in the repo.
#
#   julia --project=envs/reactant -t 4 --heap-size-hint=5G test/functional/reactant/smoke_windowed.jl
using Reactant, Enzyme, BasisSimulator, Printf, Random
const BS = BasisSimulator; const BSF = BS.Functional
T0 = time(); say(m) = (println(@sprintf("[%6.1f s] ", time() - T0), m); flush(stdout))
ok = true

scanner = BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
    detector_rows = 4, detector_cols = 64, detector_row_size = 1.0, detector_col_size = 1.0,
    detector_material = :lumex, detector_depth = 3.0, electronic_noise = 0, detection_gain = 10.0)
protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 12, rotation_time = 0.5)
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false,
    use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon = BS.ReconOptions(matrix_size = (24, 24, 2), fov_cm = 20.0, z_cm = 0.4)
phantom = BS.compact_materials(BS.create_gammex_472(n_voxels = 24, n_slices = 2, fov_cm = 20.0, z_cm = 0.4))

# `:auto` + a deliberately tiny budget is what forces the windowed branch.
pipe = BSF.eict_pipeline(phantom, scanner, protocol, opts, recon; batch_budget_mb = 0.36)
windowed = pipe.batching.col_block < pipe.geom.n_cols ||
           pipe.batching.slab_block < max(pipe.vol_shape[1], pipe.vol_shape[2])
say("col_block=$(pipe.batching.col_block)/$(pipe.geom.n_cols), slab_block=$(pipe.batching.slab_block), dd=$(pipe.batching.dd), windowed=$windowed")
windowed || (say("FAIL: budget did not force the windowed projector"); exit(1))
bs = BSF.eict_batches(pipe)
say("$(length(bs)) batches, $(length(unique(BSF._batch_key.(bs)))) distinct program keys")

fr = BSF.onehot_fractions(phantom.mask, pipe.n_mat); fr_r = Reactant.to_rarray(fr)

# 1. forward parity against the plain-host oracle
ref = BSF.eict_forward(fr, pipe)
cp = BSF.compile_pipeline(pipe, fr_r)
hu = Array(BSF.forward(cp, fr_r))
e1 = maximum(abs.(hu .- ref)); global ok &= e1 < 0.1
say(@sprintf("forward (windowed) vs host oracle: max |Δ| %.3e HU %s", e1, e1 < 0.1 ? "ok" : "FAIL"))

# 2. gradient parity: compiled pullback vs Enzyme through the single-program path
tgt = ref .+ 5.0f0; tgt_r = Reactant.to_rarray(tgt)
hu2, back = BSF.pullback(cp, fr_r)
hbar = Reactant.to_rarray(2.0f0 .* (Array(hu2) .- tgt))
g_host = Array(back(hbar))
loss(f, p, t) = sum((BSF.eict_forward(f, p) .- t) .^ 2)
gl = @jit Enzyme.gradient(Reverse, loss, fr_r, Const(pipe), Const(tgt_r))
g_loop = Array(gl isa Tuple ? gl[1] : gl)
den = max(1.0f-20, maximum(abs.(g_loop)))
e2 = maximum(abs.(g_host .- g_loop)) / den; global ok &= e2 < 5.0e-3
say(@sprintf("pullback (windowed) vs Enzyme: max rel %.3e %s", e2, e2 < 5.0e-3 ? "ok" : "FAIL"))
global ok &= all(isfinite, g_host)
say("gradient finite: $(all(isfinite, g_host))")

say(ok ? "ALL OK" : "FAILURES")
exit(ok ? 0 : 1)
