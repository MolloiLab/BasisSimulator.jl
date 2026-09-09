# The spectral chain alone under Reactant/Enzyme: forward and gradient w.r.t. the path lengths,
# for (a) one shot (view_batch = n_view), (b) compiled while loop, (c) unrolled batches; each with the
# per-pixel bowtie and with bt = nothing (pure matvec spectral sum); plus the spectral sum alone.
#   NV=64 VIEWS=100 VB=25 julia --project=envs/reactant -t 4 --heap-size-hint=5G design/reactant/probes/bench_chain.jl
using Reactant, Enzyme, BasisSimulator, Printf
const BS = BasisSimulator; const BSF = BS.Functional
envi(k, d) = parse(Int, get(ENV, k, string(d)))
NV = envi("NV", 64); VIEWS = envi("VIEWS", 100); VB = envi("VB", 25)
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
fr = BSF.onehot_fractions(phantom.mask, length(phantom.materials))
P = BSF.material_paths(fr, pipe); P_r = Reactant.to_rarray(P)
plan = pipe.eict
say(@sprintf("P %s, %d energies, bowtie %s", size(P), length(plan.wη), plan.bt === nothing ? "none" : string(size(plan.bt))))
without_bowtie(p::BSF.EICTPlan{T}) where {T} =
    BSF.EICTPlan{T, typeof(p.μ_tbl), typeof(p.wη), Nothing, typeof(p.air_ref), typeof(p.ff_log), typeof(p.scatter_Hc), typeof(p.scatter_Hr), typeof(p.bhc_coeffs)}(
        p.μ_tbl, p.wη, nothing, p.air_ref, p.I0, p.σ_e, p.use_noise, p.use_enoise, p.ff_log, p.scatter_Hc, p.scatter_Hr, p.scatter_C, p.scatter_sw,
        p.bhc_coeffs, p.eps, p.sino_shape, p.energies)
nobt = without_bowtie(plan)
cases = (
    ("chain one-shot",        p -> BSF.eict_chain(p, BSF._eict_on_device(plan, p); view_batch = 0)),
    ("chain loop",            p -> BSF.eict_chain(p, BSF._eict_on_device(plan, p); view_batch = VB, loop = true)),
    ("chain unrolled",        p -> BSF.eict_chain(p, BSF._eict_on_device(plan, p); view_batch = VB, loop = false)),
    ("chain loop, no bowtie", p -> BSF.eict_chain(p, BSF._eict_on_device(nobt, p); view_batch = VB, loop = true)),
    ("spectral sum one-shot", p -> (d = BSF._eict_on_device(plan, p); BSF.poly_log_sinogram(p, d.μ_tbl, d.wη, d.bt))),
    ("spectral sum loop",     p -> (d = BSF._eict_on_device(plan, p); BSF.poly_log_sinogram_looped(p, d.μ_tbl, d.wη, d.bt, VB))),
)
for (name, fn) in cases
    loss = p -> sum(fn(p) .^ 2)
    g = p -> Enzyme.gradient(Reverse, loss, p)
    tc = @elapsed cf = @compile sync = true fn(P_r); cf(P_r); tf = @elapsed cf(P_r)
    tc2 = @elapsed cg = @compile sync = true g(P_r); cg(P_r); tg = @elapsed cg(P_r)
    say(@sprintf("%-24s forward %.3f s (compile %.0f s) | gradient %.3f s (compile %.0f s) → ratio %.1f", name, tf, tc, tg, tc2, tg / tf))
end
say("CHAIN_BENCH_DONE")
