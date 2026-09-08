# =============================================================================
# Reactant/Enzyme smoke test for BasisSimulator.Functional's VMI stage (Cong).
#
#   julia -t 2 --heap-size-hint=3G --project=envs/reactant test/functional/reactant/smoke_vmi.jl
#
# What it proves (CPU XLA on this machine; same thunks run on CUDA):
#   1. `cong_decompose` (fixed-iteration vectorized Cong 2022 solver) traces
#      under `Reactant.@compile` with the host `CongPlan` closed over, and
#      matches the plain-Array run (Float64: ≤ 1e-5 rel; Float32 reported).
#   2. `Enzyme.gradient(Reverse, …)` INSIDE the compiled program of a scalar
#      loss `Σ ā·a + c̄·c` w.r.t. BOTH `(p_L, p_H)` — reverse mode through the
#      unrolled bisection / expansion / Newton-polish solver — agrees with
#      (i) the implicit-function-theorem VJP `cong_decompose_ift_vjp` and
#      (ii) central finite differences of the plain-Array solver (≤ 1e-3 rel).
#   3. Compile / run timings at the toy size.
#
# `vmi.jl` is included into a scratch module here until Functional.jl wires it.
# =============================================================================
using Reactant, Enzyme, BasisSimulator, LinearAlgebra, Statistics, Test
const BS = BasisSimulator
module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "vmi.jl"))
end
const F = FStage

println("Reactant ", pkgversion(Reactant), "  Enzyme ", pkgversion(Enzyme), "  Julia ", VERSION,
    "  threads=", Threads.nthreads(), "  device=", Reactant.devices()[1])

# ------------------------------------------------------------ fixture
μρ_I(E) = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(E))
μρ_W(E) = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(E))
function fixture_basis()
    scanner = BS.EICTScanner(source_to_isocenter = 625.6, source_to_detector = 1100.0,
        detector_rows = 4, detector_cols = 16, detector_row_size = 0.625, detector_col_size = 0.6,
        focal_spot_width = 1.0, focal_spot_length = 1.0, target_angle = 10.0,
        flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :ge_revolution_large,
        detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9,
        electronic_noise = 0, detection_gain = 10.0)
    prot(kVp) = BS.CTProtocol(kVp = kVp, mA = 400.0, views = 8, rotation_time = 0.5,
                              collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
    so = BS.SimOptions(seed = 1234, projector = :dd_fast)
    e_L, w_L = BS.resolve_source_spectrum_without_bowtie(so, prot(80);  scanner = scanner)
    e_H, w_H = BS.resolve_source_spectrum_without_bowtie(so, prot(140); scanner = scanner)
    (ŵ_L = Float32.(w_L ./ sum(w_L)), p_L = Float32[μρ_I(E) for E in e_L], q_L = Float32[μρ_W(E) for E in e_L],
     ŵ_H = Float32.(w_H ./ sum(w_H)), p_H = Float32[μρ_I(E) for E in e_H], q_H = Float32[μρ_W(E) for E in e_H])
end
function forward_logs(basis, a, c)
    S = size(a); p_L = Array{Float64}(undef, S); p_H = Array{Float64}(undef, S)
    for idx in CartesianIndices(S)
        p_L[idx] = -log(sum(Float64.(basis.ŵ_L) .* exp.(-(Float64.(basis.p_L) .* a[idx] .+ Float64.(basis.q_L) .* c[idx]))))
        p_H[idx] = -log(sum(Float64.(basis.ŵ_H) .* exp.(-(Float64.(basis.p_H) .* a[idx] .+ Float64.(basis.q_H) .* c[idx]))))
    end
    (p_L, p_H)
end
relmax_floor(x, y, floor) = maximum(abs.(Float64.(x) .- Float64.(y)) ./ max.(abs.(Float64.(y)), floor))

basis = fixture_basis()
a_true = reshape([0.0, 0.001, 0.02, 0.1, 0.3, 0.05, 0.2, 0.01, 0.4, 0.0, 0.15, 0.03], 4, 3, 1)
c_true = reshape([5.0, 10.0, 20.0, 30.0, 15.0, 2.0, 40.0, 25.0, 8.0, 0.0, 12.0, 35.0], 4, 3, 1)
p_L64, p_H64 = forward_logs(basis, a_true, c_true)
p_L64[end] = 0.0; p_H64[end] = 0.0                  # exact air ray → gate
water_basis = (a = 0f0, c = 1f0)
plan64 = F.cong_plan(basis; water_basis = water_basis, T = Float64, n_expand = 10, n_bisect_y = 36, n_bisect_water = 36, n_newton_y = 2)
plan32 = F.cong_plan(basis; water_basis = water_basis, T = Float32, n_expand = 10)

# ------------------------------------------------------------ 1. parity
ref64 = F.cong_decompose(p_L64, p_H64, plan64)
st64 = F.cong_solve(p_L64, p_H64, plan64)
println("plain-Array Float64: main-path rays = ", count(st64.main), " / ", length(p_L64))
pL_r = Reactant.to_rarray(p_L64); pH_r = Reactant.to_rarray(p_H64)
dec64(pl, ph) = F.cong_decompose(pl, ph, plan64)
tc1 = @elapsed dec64_c = @compile sync = true dec64(pL_r, pH_r)
a_x, c_x = dec64_c(pL_r, pH_r)
par_a = relmax_floor(Array(a_x), ref64[1], 1e-3); par_c = relmax_floor(Array(c_x), ref64[2], 1e-2)
println("Float64 parity XLA vs Array: a $(par_a)  c $(par_c)  (compile $(round(tc1; digits = 1)) s)")
@test par_a <= 1e-5
@test par_c <= 1e-5

p_L32 = Float32.(p_L64); p_H32 = Float32.(p_H64)
ref32 = F.cong_decompose(p_L32, p_H32, plan32)
pL32_r = Reactant.to_rarray(p_L32); pH32_r = Reactant.to_rarray(p_H32)
dec32(pl, ph) = F.cong_decompose(pl, ph, plan32)
tc1b = @elapsed dec32_c = @compile sync = true dec32(pL32_r, pH32_r)
a32_x, c32_x = dec32_c(pL32_r, pH32_r)
par_a32 = relmax_floor(Array(a32_x), ref32[1], 1e-3); par_c32 = relmax_floor(Array(c32_x), ref32[2], 1e-2)
println("Float32 parity XLA vs Array (floored rel): a $(par_a32)  c $(par_c32)  (compile $(round(tc1b; digits = 1)) s)")
@test par_a32 <= 1e-4
@test par_c32 <= 1e-4

# ------------------------------------------------------------ 2. Enzyme gradient
ā = reshape([1.0, -0.5, 2.0, 0.3, 0.0, 1.0, 0.7, -1.2, 0.4, 1.0, 2.0, -0.3], 4, 3, 1)
c̄ = reshape([0.2, 1.0, -1.0, 0.0, 0.5, 2.0, -0.7, 0.1, 1.5, 1.0, -2.0, 0.6], 4, 3, 1)
loss(pl, ph) = begin
    a, c = F.cong_decompose(pl, ph, plan64)
    sum(ā .* a .+ c̄ .* c)
end
grad(pl, ph) = Enzyme.gradient(Reverse, Const(loss), pl, ph)
tc2 = @elapsed grad_c = @compile sync = true grad(pL_r, pH_r)
gL_x, gH_x = grad_c(pL_r, pH_r)
gL = Array(gL_x); gH = Array(gH_x)

# (i) IFT VJP (plain Array, Float64)
vL, vH = F.cong_decompose_ift_vjp(p_L64, p_H64, ā, c̄, plan64; state = st64)
# (ii) central finite differences of the plain-Array solver
h = 1e-6
fdL = similar(p_L64); fdH = similar(p_H64)
for i in eachindex(p_L64)
    e = zeros(size(p_L64)); e[i] = h
    fdL[i] = (loss(p_L64 .+ e, p_H64) - loss(p_L64 .- e, p_H64)) / (2h)
    fdH[i] = (loss(p_L64, p_H64 .+ e) - loss(p_L64, p_H64 .- e)) / (2h)
end
scale = max(maximum(abs.(vL)), maximum(abs.(vH)))
err(x, y) = maximum(abs.(x .- y) ./ (abs.(y) .+ 1e-3 * scale))
e_ift_L = err(gL, vL); e_ift_H = err(gH, vH); e_fd_L = err(gL, fdL); e_fd_H = err(gH, fdH)
println("Enzyme reverse vs IFT VJP: p_L $(e_ift_L)  p_H $(e_ift_H);  vs central FD: p_L $(e_fd_L)  p_H $(e_fd_H)  (compile $(round(tc2; digits = 1)) s)")
@test e_ift_L <= 1e-3 && e_ift_H <= 1e-3
@test e_fd_L <= 1e-3 && e_fd_H <= 1e-3

# ------------------------------------------------------------ 3. timing
# `@compile sync = true` makes the compiled calls synchronous, so a plain @elapsed is a wall time.
timeit(f, args...) = (f(args...); @elapsed f(args...))
t_dec = timeit(dec64_c, pL_r, pH_r); t_grad = timeit(grad_c, pL_r, pH_r)
t_cpu = (F.cong_decompose(p_L64, p_H64, plan64); @elapsed F.cong_decompose(p_L64, p_H64, plan64))
println()
println("| quantity | value |")
println("|---|---|")
println("| rays / n_E (low, high) | $(length(p_L64)) / ($(length(basis.p_L)), $(length(basis.p_H))) |")
println("| compile s: decompose Float64 / Float32 / Enzyme-gradient | $(round(tc1; digits=1)) / $(round(tc1b; digits=1)) / $(round(tc2; digits=1)) |")
println("| parity XLA vs Array (floored rel): Float64 a,c / Float32 a,c | $(par_a), $(par_c) / $(par_a32), $(par_c32) |")
println("| Enzyme reverse vs IFT VJP (rel): p_L / p_H | $(e_ift_L) / $(e_ift_H) |")
println("| Enzyme reverse vs central FD (rel): p_L / p_H | $(e_fd_L) / $(e_fd_H) |")
println("| XLA-CPU run: decompose / gradient ; plain Array decompose | $(round(t_dec*1e3; digits=1)) ms / $(round(t_grad*1e3; digits=1)) ms ; $(round(t_cpu*1e3; digits=1)) ms |")
