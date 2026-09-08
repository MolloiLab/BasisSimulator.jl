# Reactant / Enzyme smoke test for the functional FBP/FDK stage.
#
#   julia --project=envs/reactant -t 2 test/functional/reactant/smoke_fbp.jl
#
# Compiles `fdk` (cosine weight → Toeplitz ramp matmul → voxel-driven FDK
# backprojection with in-graph bilinear gathers → FOV mask) on the toy
# geometry of test/api.jl (32 cols × 8 rows × 16 views, 16×16×4 volume,
# z_cm = 0.4 so every voxel is in-bounds), for the :arc and :flat detectors.
#
# The plan's constant tensors (`plan.tensors`: cosine weights, Toeplitz H,
# per-view geometry, voxel grids, FOV mask) are moved to the device with
# `Reactant.to_rarray` and the plan is rebuilt INSIDE the traced function from
# them, so the per-view index tensors are built in-graph.  The host-constant
# plan (tensors left as plain Arrays → trace-time constants) is also compiled.
#
# Gates:
#   • compiled == plain-Array `fdk`: Float64 to ≤ 1e-5 rel of max|ref|,
#     Float32 to ≤ 1e-4 (device-tensor plan, view_batch 1 and 4; host-constant
#     plan).  On a random-noise sinogram Float32 sits at ~1e-5: XLA's atan2 /
#     fused arithmetic moves the sub-pixel detector coordinates by ~1e-6
#     columns and white-noise data turns that straight into value error.  The
#     compiled Float32 result is also compared against the plain-Array
#     Float64 reference and must be as accurate as plain Float32 (reported).
#   • `Enzyme.gradient(Reverse, …)` of `sum(w .* fdk(sino))` w.r.t. `sino`
#     (Float64) agrees with plain-Array central differences to ≤ 1e-3 rel
#     (directional derivative + per-entry probes).

using Test
using Reactant, Enzyme
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics   # fbp.jl needs no FFTW itself
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "fbp.jl"))
end
const F = FStage

const _T0 = time()
_ts(label) = (println(stderr, "[smoke_fbp t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))
relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))

# ----------------------------------------------------------------------------- fixture
function toy_geom(; shape = :arc)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = shape)
    return BS.CTGeometry(scanner; n_angles = 16, fov_cm = 20.0, z_cm = 0.4)
end
const VOL = (16, 16, 4)

# plan rebuilt inside the trace from the (traced) tensor bundle
fdk_dev(y, tensors, plan_host; view_batch = 1) = F.fdk(y, F.FBPPlan(plan_host; tensors = tensors); view_batch = view_batch)

_ts("start")
@testset "Reactant smoke: functional FBP/FDK" begin
    for shape in (:arc, :flat), T in (Float32, Float64)
        tol = T === Float64 ? 1e-5 : 1e-4
        geom = toy_geom(shape = shape)
        plan = F.fbp_plan(geom, VOL; T = T)
        sino = randn(MersenneTwister(4), T, geom.n_cols, geom.n_rows, geom.n_angles)
        ref = F.fdk(sino, plan)
        @test any(v -> v != 0 && v != plan.sentinel, ref)     # non-degenerate fixture
        # plain-Array Float64 reference of the same problem (accuracy yardstick)
        ref64 = F.fdk(Float64.(sino), F.fbp_plan(geom, VOL; T = Float64))

        sino_r = Reactant.to_rarray(sino)
        tensors_r = Reactant.to_rarray(plan.tensors)

        # ── device-tensor plan, view_batch = 1 ──────────────────────────────
        _ts("$shape / $T: compile fdk (device tensors, view_batch = 1)")
        t0 = time()
        run1 = @compile fdk_dev(sino_r, tensors_r, plan)
        println("  [$shape/$T] compile time (vb=1): $(round(time() - t0; digits = 1)) s")
        out1 = Array(run1(sino_r, tensors_r, plan))
        e1 = relmax(out1, ref)
        println("  [$shape/$T] compiled (device tensors, vb=1) vs plain-Array rel = $e1")
        @test e1 ≤ tol
        @test all(isfinite, out1)
        a_c = relmax(Float64.(out1), ref64); a_p = relmax(Float64.(ref), ref64)
        println("  [$shape/$T] accuracy vs Float64 reference: compiled = $a_c, plain-Array = $a_p")
        @test a_c ≤ 2 * max(a_p, 1e-6)          # compiled is as accurate as plain

        # ── device-tensor plan, view_batch = 4 (batched per-view broadcasts) ─
        fdk_dev4(y, tb, p) = fdk_dev(y, tb, p; view_batch = 4)
        _ts("$shape / $T: compile fdk (device tensors, view_batch = 4)")
        t0 = time()
        run4 = @compile fdk_dev4(sino_r, tensors_r, plan)
        println("  [$shape/$T] compile time (vb=4): $(round(time() - t0; digits = 1)) s")
        out4 = Array(run4(sino_r, tensors_r, plan))
        e4 = relmax(out4, ref)
        println("  [$shape/$T] compiled (device tensors, vb=4) vs plain-Array rel = $e4")
        @test e4 ≤ tol

        # ── host-constant plan (tensors baked in as trace-time constants) ────
        if T === Float32
            _ts("$shape / $T: compile fdk (host-constant plan)")
            t0 = time()
            runh = @compile F.fdk(sino_r, plan)
            println("  [$shape/$T] compile time (host plan): $(round(time() - t0; digits = 1)) s")
            outh = Array(runh(sino_r, plan))
            eh = relmax(outh, ref)
            println("  [$shape/$T] compiled (host-constant plan) vs plain-Array rel = $eh")
            @test eh ≤ tol
        end

        # ── Enzyme reverse-mode gradient w.r.t. the sinogram ─────────────────
        if T === Float64
            w = randn(MersenneTwister(11), T, VOL)
            loss(y, tb, p) = sum(w .* fdk_dev(y, tb, p))
            _ts("$shape / $T: compile Enzyme.gradient")
            t0 = time()
            grad_c = @compile Enzyme.gradient(Reverse, loss, sino_r, Const(tensors_r), Const(plan))
            println("  [$shape/$T] gradient compile time: $(round(time() - t0; digits = 1)) s")
            g = Array(grad_c(Reverse, loss, sino_r, Const(tensors_r), Const(plan))[1])
            @test size(g) == size(sino)
            @test all(isfinite, g)
            @test any(!=(0), g)

            # plain-Array Float64 loss for finite differences (fdk is linear in
            # the sinogram, so central differences are exact up to rounding)
            loss_h(y) = sum(w .* F.fdk(y, plan))
            d = randn(MersenneTwister(21), T, size(sino))
            h = 1e-3
            dd_fd = (loss_h(sino .+ h .* d) - loss_h(sino .- h .* d)) / (2h)
            dd_ad = sum(g .* d)
            rel_dir = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
            println("  [$shape/$T] directional derivative: AD = $dd_ad, FD = $dd_fd, rel = $rel_dir")
            @test rel_dir ≤ 1e-3
            rng = MersenneTwister(5)
            worst = 0.0
            for I in rand(rng, CartesianIndices(sino), 4)
                e1v = zeros(T, size(sino)); e1v[I] = h
                fd = (loss_h(sino .+ e1v) - loss_h(sino .- e1v)) / (2h)
                rel = abs(g[I] - fd) / max(abs(fd), 1e-9)
                worst = max(worst, rel)
                println("  [$shape/$T] ∂loss/∂sino$(Tuple(I)): AD = $(g[I]), FD = $fd, rel = $rel")
            end
            @test worst ≤ 1e-3
        end
    end
end
_ts("done")
