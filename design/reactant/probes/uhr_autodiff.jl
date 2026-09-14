# Full-UHR autodiff validation: does the compiled differentiable twin fit, run fast, and produce a
# CORRECT gradient through the ENTIRE imaging chain (fractions -> DD projection -> polychromatic
# detector + BHC -> FDK backprojection -> HU)?
#
# Correctness is a directional finite difference on the real scalar loss, not a finiteness check:
#     L(x) = sum(w .* HU(x))          w a fixed random weight tensor
#     dL/dv = <grad L, v>             v a fixed random direction
#     FD    = (L(x + h v) - L(x - h v)) / 2h
# Every stage of the chain is inside L, so agreement validates the whole chain at once.
#
#   BUDGET=29366 VIEWS=984 FD_VIEWS=36 julia --project=envs/reactant design/reactant/probes/uhr_autodiff.jl
using Reactant, BasisSimulator, Printf, Random, Statistics
const BS = BasisSimulator; const BSF = BS.Functional
T0 = time(); say(m) = (@printf("[%7.1f s] %s\n", time() - T0, m); flush(stdout))
vram() = try strip(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader`, String)) catch; "n/a" end

GRID   = parse(Int, get(ENV, "GRID", "1125"))
SLICES = parse(Int, get(ENV, "SLICES", "16"))
VIEWS  = parse(Int, get(ENV, "VIEWS", "984"))
FDV    = parse(Int, get(ENV, "FD_VIEWS", "36"))
BUDGET = parse(Int, get(ENV, "BUDGET", "29366"))

scanner = BS.EICTScanner(source_to_isocenter=625.6, source_to_detector=1100.0, detector_rows=256, detector_cols=834,
    detector_row_size=0.625, detector_col_size=0.6, detector_shape=:arc, focal_spot_width=1.0, focal_spot_length=1.0,
    target_angle=10.0, flat_filter_material=:aluminum, flat_filter_thickness=2.5, bowtie_filter=:ge_revolution_large,
    detector_material=:lumex, detector_depth=3.0, fill_factor_row=0.9, fill_factor_col=0.9, electronic_noise=0, detection_gain=10.0)
opts  = BS.SimOptions(use_noise=false, use_scatter=false, use_focal_spot=false, use_optical_crosstalk=false, use_lag=false, seed=1234)
recon = BS.ReconOptions(matrix_size=(512,512,8), fov_cm=35.0, z_cm=0.5)

say("phantom: Gammex 472 at $(GRID)^2 x $(SLICES) (UHR object grid)")
ph = BS.compact_materials(BS.create_gammex_472(n_voxels=GRID, n_slices=SLICES, fov_cm=45.0, z_cm=1.0))
fr = BSF.onehot_fractions(ph.mask, length(ph.materials))
say(@sprintf("fractions %s = %.2f GiB", string(size(fr)), sizeof(fr)/2^30))

results = Dict{String,Any}(); allok = true

function build(nviews)
    protocol = BS.CTProtocol(kVp=120, mA=200.0, views=nviews, rotation_time=1.0, collimation_mm=5.0, additional_filters=[("Al",4.5)])
    pipe = BSF.eict_pipeline(ph, scanner, protocol, opts, recon; batch_budget_mb=BUDGET)
    windowed = pipe.batching.col_block < pipe.geom.n_cols || pipe.batching.slab_block < max(pipe.vol_shape[1], pipe.vol_shape[2])
    say("  pipe($nviews views): dd=$(pipe.batching.dd), n_rows=$(pipe.geom.n_rows), windowed=$windowed, n_mat=$(pipe.n_mat)")
    pipe
end

# ---------------------------------------------------------------- A. full-size UHR: fit + speed
say("=== A. full-size UHR, $VIEWS views ===")
pipeF = build(VIEWS); frF = Reactant.to_rarray(fr)
tc = @elapsed cpF = BSF.compile_pipeline(pipeF, frF)
say(@sprintf("  compile: %d batches, %d fwd + %d vjp programs, %.0f s", length(cpF.batches), length(cpF.fwd), length(cpF.vjp), tc))
tf = @elapsed huF = Array(BSF.forward(cpF, frF))
say(@sprintf("  FORWARD  %.1f s  (%.1f ms/view)  HU range [%.1f, %.1f]  VRAM %s", tf, 1000tf/VIEWS, minimum(huF), maximum(huF), vram()))
Random.seed!(7); wF = Reactant.to_rarray(randn(Float32, size(huF)))
tg = @elapsed begin
    _, backF = BSF.pullback(cpF, frF); gF = Array(backF(wF))
end
finite = all(isfinite, gF); allok &= finite
say(@sprintf("  GRADIENT %.1f s  (%.1f ms/view)  %s  |g|_1=%.4e  finite=%s  VRAM %s",
    tg, 1000tg/VIEWS, string(size(gF)), sum(abs, gF), finite, vram()))
results["forward_s"]=tf; results["gradient_s"]=tg; results["compile_s"]=tc

# ---------------------------------------------------------------- B. gradient correctness (FD)
say("=== B. gradient correctness at UHR, $FDV views (directional finite difference) ===")
pipeD = build(FDV)
cpD = BSF.compile_pipeline(pipeD, frF)
hu0 = Array(BSF.forward(cpD, frF))
Random.seed!(11)
w = randn(Float32, size(hu0)); w_r = Reactant.to_rarray(w)
v = randn(Float32, size(fr)); v ./= sqrt(sum(abs2, v))          # unit direction, full chain exercised
_, backD = BSF.pullback(cpD, frF)
gD = Array(backD(w_r))
analytic = sum(Float64.(gD) .* Float64.(v))
for h in (1.0f-2, 1.0f-3)
    Lp = sum(Float64.(w) .* Float64.(Array(BSF.forward(cpD, Reactant.to_rarray(fr .+ h .* v)))))
    Lm = sum(Float64.(w) .* Float64.(Array(BSF.forward(cpD, Reactant.to_rarray(fr .- h .* v)))))
    fd = (Lp - Lm) / (2h)
    rel = abs(fd - analytic) / max(abs(analytic), 1e-8)
    good = rel < 5e-2; global allok &= good
    say(@sprintf("  h=%.0e  FD=%.6e  analytic=%.6e  rel=%.2e  %s", h, fd, analytic, rel, good ? "OK" : "FAIL"))
    results["fd_rel_h$(h)"] = rel
end

say(allok ? "=== ALL CHECKS PASSED ===" : "=== FAILURES PRESENT ===")
for (k,v) in sort(collect(results), by=first); @printf("  %-16s %s\n", k, v); end
exit(allok ? 0 : 1)
