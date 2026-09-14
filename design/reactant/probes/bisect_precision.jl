# Does forcing full-f32 dot_general precision remove the GPU-only HU offset?  Same measurements as
# bisect_compiled_offset.jl, with every compile traced under `Reactant.with_config(...)`.
#   PREC=HIGHEST [ALG=F32_F32_F32|TF32_TF32_F32_X3] BACKEND=gpu NV=32 NS=2 R=32 NZ=2 VIEWS=40 julia --project=envs/reactant design/reactant/probes/bisect_precision.jl
using Reactant, BasisSimulator, Printf, Statistics
BACK = get(ENV, "BACKEND", "gpu"); Reactant.set_default_backend(BACK)
const BS = BasisSimulator; const BSF = BS.Functional
NV = parse(Int, get(ENV, "NV", "32"));  NS = parse(Int, get(ENV, "NS", "2"))
R  = parse(Int, get(ENV, "R", "32"));   NZ = parse(Int, get(ENV, "NZ", "2"))
VIEWS = parse(Int, get(ENV, "VIEWS", "40")); BUDGET = parse(Int, get(ENV, "BUDGET", "16384"))
PREC = getproperty(Reactant.PrecisionConfig, Symbol(get(ENV, "PREC", "DEFAULT")))
ALG  = haskey(ENV, "ALG") ? getproperty(Reactant.DotGeneralAlgorithmPreset, Symbol(ENV["ALG"])) : missing
TAG = "$(BACK)/$(get(ENV,"PREC","DEFAULT"))$(haskey(ENV,"ALG") ? "+"*ENV["ALG"] : "")"
T0 = time(); say(m) = (@printf("[%s][%7.1f s] %s\n", TAG, time() - T0, m); flush(stdout))
withprec(f) = ALG === missing ? Reactant.with_config(f; dot_general_precision = PREC, convolution_precision = PREC) :
                                Reactant.with_config(f; dot_general_precision = PREC, convolution_precision = PREC, dot_general_algorithm = ALG)
include(joinpath(@__DIR__, "ge_revolution.jl"))          # the GE Revolution scanner (shared fixture)
protocol = BS.CTProtocol(kVp = 120, mA = 200.0, views = VIEWS, rotation_time = 1.0, collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
opts = BS.SimOptions(use_noise = false, use_scatter = false, use_focal_spot = false, use_optical_crosstalk = false, use_lag = false, seed = 1234)
recon = BS.ReconOptions(matrix_size = (R, R, NZ), fov_cm = 35.0, z_cm = 0.5)
ph = BS.compact_materials(BS.create_gammex_472(n_voxels = NV, n_slices = NS, fov_cm = 45.0, z_cm = 1.0))
fr = BSF.onehot_fractions(ph.mask, length(ph.materials)); fr_r = Reactant.to_rarray(fr)
pipe = BSF.eict_pipeline(ph, scanner, protocol, opts, recon; batch_budget_mb = BUDGET)
say("phantom $(size(ph.mask)) recon $(recon.matrix_size) views $VIEWS | dd=$(pipe.batching.dd) windowed=$(pipe.batching.col_block < 834)")
circ = [hypot(i - (R + 1) / 2, j - (R + 1) / 2) < 0.47R for i in 1:R, j in 1:R, k in 1:NZ]
st(a, b) = (d = (a .- b)[circ]; @sprintf("mean %+9.4f  rms %8.4f  max %8.4f", mean(d), sqrt(mean(abs2, d)), maximum(abs.(d))))
ref = BSF.eict_forward(fr, pipe)
tc = @elapsed cp = withprec(() -> BSF.compile_pipeline(pipe, fr_r))
hu = Array(BSF.forward(cp, fr_r)); tf = @elapsed Array(BSF.forward(cp, fr_r))
say(@sprintf("compile_pipeline forward vs host oracle: %s | compile %.0f s, forward %.3f s (%.4f s/view)", st(hu, ref), tc, tf, tf / VIEWS))
if get(ENV, "GRAD", "0") == "1"
    tgt = hu .+ 5f0
    tg = @elapsed begin
        hu2, back = BSF.pullback(cp, fr_r)
        g = Array(back(Reactant.to_rarray(2f0 .* (Array(hu2) .- tgt))))
    end
    say(@sprintf("PULLBACK: grad %s ||g|| %.4e nonzero %.1f%% finite=%s | first-call %.2f s", string(size(g)), sqrt(sum(abs2, g)), 100 * count(!iszero, g) / length(g), all(isfinite, g), tg))
    tg2 = @elapsed Array(back(Reactant.to_rarray(2f0 .* (Array(hu2) .- tgt))))
    say(@sprintf("PULLBACK steady-state: %.3f s for %d views (%.4f s/view) -> 984 views = %.1f s", tg2, VIEWS, tg2 / VIEWS, tg2 / VIEWS * 984))
end
say("DONE")
