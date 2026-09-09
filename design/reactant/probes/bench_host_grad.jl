# Host-composed gradient: one compiled batch program per (orientation, batch length), called from the
# host for every view batch — forward = Σ_b vol_b, gradient = Σ_b ∂⟨v̄, vol_b⟩/∂fractions. Compared with
# the single while-loop program (`eict_forward` + Enzyme) at the same batching.
#   NV=64 VIEWS=100 BUDGET_MB=512 julia --project=envs/reactant -t 4 --heap-size-hint=6G design/reactant/probes/bench_host_grad.jl
using Reactant, Enzyme, BasisSimulator, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); BUDGET_MB = envi("BUDGET_MB", 512); LOOP = envi("LOOP", 1)
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
bs = BSF.eict_batches(pipe)
say("batching $(pipe.batching); $(length(bs)) batches, lengths $(unique(length(b.views) for b in bs)), windowed $(bs[1].windowed)")
ref_hu = BSF.eict_forward(fr, pipe)                      # host oracle
# --- compile one forward and one pullback program per (orientation, length); the ACCUMULATOR is an
#     argument of each program (acc .+ contribution inside the compiled program — a `.+` on concrete
#     Reactant arrays outside a program falls to element-wise host execution: 0.9 s per call at 256²)
key(b) = (b.run.vertical, length(b.views))
fwd_c = Dict{Any, Any}(); vjp_c = Dict{Any, Any}(); data_r = Dict{Int, Any}()
vol0 = Reactant.to_rarray(zeros(Float32, NV, NV, 2)); g0 = Reactant.to_rarray(zeros(Float32, size(fr)))
tcomp = @elapsed for (i, b) in enumerate(bs)
    d = Reactant.to_rarray(BSF.batch_data(b, pipe)); data_r[i] = d
    k = key(b)
    haskey(fwd_c, k) && continue
    f = (acc, x, d) -> acc .+ BSF.eict_batch_vol(x, pipe, b, d)
    g = (acc, x, vbar, d) -> acc .+ Enzyme.gradient(Reverse, (x, vbar, d) -> sum(vbar .* BSF.eict_batch_vol(x, pipe, b, d)), x, Const(vbar), Const(d))[1]
    fwd_c[k] = @compile sync = true f(vol0, fr_r, d)
    vjp_c[k] = @compile sync = true g(g0, fr_r, vol0, d)
end
say(@sprintf("compiled %d forward + %d pullback batch programs in %.0f s", length(fwd_c), length(vjp_c), tcomp))
tohu = @compile sync = true BSF.eict_vol_to_hu(vol0, pipe)
zero_vol = @compile sync = true (v -> v .* 0f0)(vol0)
zero_g = @compile sync = true (g -> g .* 0f0)(g0)
# --- forward
function forward!(x)
    vol = zero_vol(vol0)
    for (i, b) in enumerate(bs)
        vol = fwd_c[key(b)](vol, x, data_r[i])
    end
    return vol
end
vol = forward!(fr_r); hu = tohu(vol, pipe)
tf = @elapsed (vol = forward!(fr_r); hu = tohu(vol, pipe))
err = maximum(abs.(Array(hu) .- ref_hu))
say(@sprintf("host-composed forward: %.2f s (%d batch calls, %.1f ms each), max |Δ| vs host oracle %.2e HU", tf, length(bs), 1e3 * tf / length(bs), err))
# --- gradient of Σ (HU − target)²: v̄ = ∂loss/∂vol through eict_vol_to_hu (compiled), then Σ_b pullbacks
tgt = Reactant.to_rarray(ref_hu .+ 5f0)
vbar_fn = (vol, t) -> Enzyme.gradient(Reverse, (vol, t) -> sum((BSF.eict_vol_to_hu(vol, pipe) .- t) .^ 2), vol, Const(t))[1]
vbar_c = @compile sync = true vbar_fn(vol, tgt)
function gradient!(x)
    vol = forward!(x); vb = vbar_c(vol, tgt)
    gsum = zero_g(g0)
    for (i, b) in enumerate(bs)
        gsum = vjp_c[key(b)](gsum, x, vb, data_r[i])
    end
    return gsum
end
gh = gradient!(fr_r); tg = @elapsed gh = gradient!(fr_r)
say(@sprintf("host-composed gradient step: %.2f s (forward + v̄ + %d pullback calls, %.1f ms each) → ratio %.1f", tg, length(bs), 1e3 * (tg - tf) / length(bs), tg / tf))
# --- the single-program while-loop gradient for comparison
if LOOP == 1
    loss(f, p, t) = sum((BSF.eict_forward(f, p) .- t) .^ 2)
    g = (f, p, t) -> Enzyme.gradient(Reverse, loss, f, Const(p), Const(t))
    tc = @elapsed cg = @compile sync = true g(fr_r, pipe, tgt); gl = cg(fr_r, pipe, tgt); tl = @elapsed cg(fr_r, pipe, tgt)
    gla = Array(gl isa Tuple ? gl[1] : gl); gha = Array(gh)
    say(@sprintf("while-loop program gradient step: %.2f s (compile %.0f s); host-composed vs loop gradient: max |Δ| %.2e (max |g| %.2e)",
        tl, tc, maximum(abs.(gha .- gla)), maximum(abs.(gla))))
end
say("HOST_GRAD_DONE")
