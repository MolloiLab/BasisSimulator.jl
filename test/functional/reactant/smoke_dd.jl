# =============================================================================
# Reactant/Enzyme smoke test for BasisSimulator.Functional's DD projector.
#
#   julia -t 2 --heap-size-hint=3G --project=envs/reactant test/functional/reactant/smoke_dd.jl
#
# What it proves (CPU XLA on this machine; same thunks run on CUDA):
#   1. `dd_project_view` / `dd_transpose_view` trace under `Reactant.@compile`
#      and match the plain-Array run (≤ 1e-4 rel, Float32 — the legacy
#      `:dd_fast ≡ :dd` summation-order contract).
#   2. `Enzyme.gradient(Reverse, …)` INSIDE the compiled program of
#      `sum(dd_project_view(vol) .* s)` w.r.t. `vol` equals the gather-form
#      transpose `dd_transpose_view(s)` (≤ 1e-4 rel) — the adjoint falls out of
#      generic AD with no hand-written rule.
#   3. Timings (second call, `sync=true`) for forward / transpose / gradient per
#      view at a toy size (kept small on purpose: the 16 GB host is shared).
#
# The `_iota` override below makes the index vectors in-graph iotas, so every
# geometry array (weights, tap indices) is computed inside the XLA program
# instead of being embedded as a literal — this is the hook a future
# `BasisSimulatorReactantExt` should own.
# =============================================================================
using Reactant, Enzyme, BasisSimulator, LinearAlgebra, Statistics, Test
const BS = BasisSimulator
const BSF = BS.Functional

BSF._iota(::Reactant.TracedRArray, ::Type{T}, n::Integer) where {T} =
    Reactant.Ops.iota(T, [Int(n)]; iota_dimension = 1) .+ one(T)

relmax(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / max(maximum(abs.(Float64.(b))), eps())
function water_cylinder(nx, ny, nz; μ = 0.2f0)
    vol = zeros(Float32, nx, ny, nz); cx = (nx + 1) / 2; cy = (ny + 1) / 2; R = 0.6 * (nx / 2)
    for k in 1:nz, j in 1:ny, i in 1:nx
        (i - cx)^2 + (j - cy)^2 <= R^2 && (vol[i, j, k] = μ)
    end
    vol
end
function toy_geom(; n_cols, n_rows, n_angles, fov_cm)
    scanner = BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = n_rows, detector_cols = n_cols, detector_row_size = 1.0, detector_col_size = 1.0)
    BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = fov_cm, z_cm = 1.0)
end
timeit(f, args...) = (f(args...); @elapsed f(args...))      # thunks are compiled with sync=true

println("Reactant ", pkgversion(Reactant), "  Enzyme ", pkgversion(Enzyme), "  Julia ", VERSION,
    "  threads=", Threads.nthreads(), "  device=", Reactant.devices()[1])

# ---------------------------------------------------------------- 1. parity
geom = toy_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
vol = water_cylinder(64, 64, 8)
view = 2                                                   # 45° — exercises the vertical tie-break
plan = BSF.dd_view_plan(geom, view, size(vol); eltype = Float32)
s = randn(Float32, geom.n_cols, geom.n_rows)
fwd_ref = BSF.dd_project_view(vol, plan)
bp_ref = BSF.dd_transpose_view(s, plan, size(vol))

vol_r = Reactant.to_rarray(vol); s_r = Reactant.to_rarray(s)
fwd(v) = BSF.dd_project_view(v, plan)
bpj(x) = BSF.dd_transpose_view(x, plan, size(vol))
tc1 = @elapsed fwd_c = @compile sync = true fwd(vol_r)
tc2 = @elapsed bp_c = @compile sync = true bpj(s_r)
fwd_x = Array(fwd_c(vol_r)); bp_x = Array(bp_c(s_r))
par_f = relmax(fwd_x, fwd_ref); par_b = relmax(bp_x, bp_ref)
println("parity forward=", par_f, "  transpose=", par_b, "  (nan? ", any(isnan, fwd_x), "/", any(isnan, bp_x), ")")

# ------------------------------------------------------- 2. Enzyme gradient
loss(v, x) = sum(BSF.dd_project_view(v, plan) .* x)
grad(v, x) = Enzyme.gradient(Reverse, Const(loss), v, Const(x))[1]
tc3 = @elapsed grad_c = @compile sync = true grad(vol_r, s_r)
g_x = Array(grad_c(vol_r, s_r))
par_g = relmax(g_x, bp_ref)
dot_lhs = sum(Float64.(fwd_x) .* Float64.(s)); dot_rhs = sum(Float64.(vol) .* Float64.(g_x))
println("gradient vs transpose=", par_g, "  dot rel=", abs(dot_lhs - dot_rhs) / abs(dot_lhs))

# ----------------------------------------------------------------- 3. timing
# Toy size only (shared 16 GB host): 128²×8 volume, 256×8 detector, one view.
geom2 = toy_geom(n_cols = 256, n_rows = 8, n_angles = 8, fov_cm = 20.0)
vol2 = water_cylinder(128, 128, 8)
plan2 = BSF.dd_view_plan(geom2, 3, size(vol2); eltype = Float32)
s2 = randn(Float32, geom2.n_cols, geom2.n_rows)
v2 = Reactant.to_rarray(vol2); x2 = Reactant.to_rarray(s2)
fwd2(v) = BSF.dd_project_view(v, plan2)
bp2(x) = BSF.dd_transpose_view(x, plan2, size(vol2))
loss2(v, x) = sum(BSF.dd_project_view(v, plan2) .* x)
grad2(v, x) = Enzyme.gradient(Reverse, Const(loss2), v, Const(x))[1]
fwd2_c = @compile sync = true fwd2(v2)
bp2_c = @compile sync = true bp2(x2)
grad2_c = @compile sync = true grad2(v2, x2)
t_f = timeit(fwd2_c, v2); t_b = timeit(bp2_c, x2); t_g = timeit(grad2_c, v2, x2)
t_f_cpu = (BSF.dd_project_view(vol2, plan2); @elapsed BSF.dd_project_view(vol2, plan2))
t_b_cpu = (BSF.dd_transpose_view(s2, plan2, size(vol2)); @elapsed BSF.dd_transpose_view(s2, plan2, size(vol2)))
par_g2 = relmax(Array(grad2_c(v2, x2)), BSF.dd_transpose_view(s2, plan2, size(vol2)))

println()
println("| quantity | value |")
println("|---|---|")
println("| taps (KX,KZ,KXT,KZT) 64-col fixture / timing fixture | $((plan.KX, plan.KZ, plan.KXT, plan.KZT)) / $((plan2.KX, plan2.KZ, plan2.KXT, plan2.KZT)) |")
println("| compile s: forward / transpose / Enzyme-gradient | $(round(tc1; digits=1)) / $(round(tc2; digits=1)) / $(round(tc3; digits=1)) |")
println("| parity vs plain Array: forward / transpose (max rel) | $(par_f) / $(par_b) |")
println("| Enzyme reverse gradient vs gather transpose (max rel) | $(par_g) |")
println("| ⟨A v, s⟩ vs ⟨v, ∇⟩ rel diff | $(abs(dot_lhs - dot_rhs) / abs(dot_lhs)) |")
println("| XLA-CPU per view @ 128²×8 vol, 256×8 det: forward / transpose / gradient | $(round(t_f*1e3; digits=2)) ms / $(round(t_b*1e3; digits=2)) ms / $(round(t_g*1e3; digits=2)) ms |")
println("| plain Julia Array per view (same): forward / transpose | $(round(t_f_cpu*1e3; digits=2)) ms / $(round(t_b_cpu*1e3; digits=2)) ms |")
println("| Enzyme gradient vs gather transpose, timing fixture (max rel) | $(par_g2) |")
println()
# Float32 tolerances follow the legacy `:dd_fast ≡ :dd` contract (1e-4): XLA re-associates
# the slab sum / fuses multiply-adds, so agreement is at summation-order level, not bitwise.
@testset "reactant smoke" begin
    @test par_f <= 1.0e-4
    @test par_b <= 1.0e-4
    @test par_g <= 1.0e-4
    @test abs(dot_lhs - dot_rhs) / abs(dot_lhs) <= 1.0e-4
    @test par_g2 <= 1.0e-4
end
