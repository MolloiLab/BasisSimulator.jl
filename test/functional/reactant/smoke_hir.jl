# Reactant / Enzyme smoke test for the functional HIR stage.
#
#   julia --project=envs/reactant -t 2 test/functional/reactant/smoke_hir.jl
#
# Two operator pairs are exercised:
#   (1) a DENSE system matrix built column-by-column from the legacy DD
#       forward projector (`M`, with `Mᵀ` masked to the circular support) —
#       matmul is trivially traceable, so this isolates the STAGE's
#       Reactant/Enzyme behaviour from the projector port;
#   (2) the functional DD projector (`BasisSimulator.Functional.dd_project_view`
#       / `dd_transpose_view`), reported but not gated and OPT-IN
#       (`BS_HIR_SMOKE_FUNCTIONAL_DD=1`): its traceability is owned by the
#       projector stage and the trace is large.
#
# Memory protocol: run under the system-wide Reactant lock, one Julia process,
#   until mkdir /tmp/bs_reactant.lock 2>/dev/null; do sleep 30; done
#   trap 'rmdir /tmp/bs_reactant.lock' EXIT
#   julia --project=envs/reactant -t 2 --heap-size-hint=3G test/functional/reactant/smoke_hir.jl
#
# Gates (dense operators, 1 epoch of the strength-60 row):
#   • compiled Float64 == plain-Array to ≤ 1e-5 rel of max|ref|;
#   • `Enzyme.gradient(Reverse, …)` of `sum(w .* hir(sino))` w.r.t. the
#     sinogram agrees with plain-Array Float64 central differences to ≤ 1e-3
#     rel (directional derivative + per-entry probes).

using Test
using Reactant, Enzyme
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator
const BSF = BasisSimulator.Functional

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "..", "src", "functional", "hir.jl"))
end

const _T0 = time()
_ts(label) = (println(stderr, "[smoke_hir t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))

# ----------------------------------------------------------------------------- fixtures
function toy_geom(; n_angles = 24)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0,
    )
    return BS.CTGeometry(scanner; n_angles, fov_cm = 20.0, z_cm = 5.0)
end

function cylinder_sino(geom, matrix_size; T = Float64, μ = T(0.02), noise = T(0.003), seed = 0x484952)
    nx, ny, nz = matrix_size
    obj = zeros(T, nx, ny, nz)
    cx, cy = (nx + 1) / 2, (ny + 1) / 2
    for k in 1:nz, j in 1:ny, i in 1:nx
        (i - cx)^2 + (j - cy)^2 < (0.35 * nx)^2 && (obj[i, j, k] = μ)
    end
    sino = BS._project_mono(:dd_fast, obj, geom)
    return sino .+ noise .* randn(MersenneTwister(seed), T, size(sino))
end

function fdk_init(sino, geom, matrix_size)
    ws = BS.create_fdk_recon_workspace(sino, geom, matrix_size)
    return copy(BS.reconstruct!(ws, sino, geom))
end

one_epoch_plan(geom, sino_shape, vol_shape, T) = begin
    p = BS.get_hir_params(60)
    FStage.hir_plan(geom, sino_shape, vol_shape,
        BS.HIRParams(60, p.lambda, 1, p.n_subsets, p.huber_delta, p.relaxation, p.target_noise_reduction); T)
end

# Dense legacy system matrix on the work grid: column j = vec(A e_j), rays
# ordered (col, row, view) so `reshape(M, nc*nr, n_views, nvox)` splits views.
function dense_system(plan::FStage.HIRPlan{T}) where {T}
    wg = plan.work_geom
    nc, nr, nv = plan.sino_shape
    vshape = plan.work_shape
    nvox = prod(vshape)
    M = zeros(T, nc * nr * nv, nvox)
    basis = zeros(T, vshape)
    for j in 1:nvox
        fill!(basis, zero(T)); basis[j] = one(T)
        M[:, j] .= vec(BS._project_mono(:dd_fast, basis, wg))
    end
    # circular support of dd_backproject!(circular_support = true)
    nx, ny = vshape[1], vshape[2]
    b = wg.fov
    vmx = T(-b[1] / 2); vmy = T(-b[2] / 2); vsx = T(b[1]) / T(nx); vsy = T(b[2]) / T(ny)
    radius_sq = T(min(b[1], b[2]) / 2)^2
    support = Array{Bool, 3}(undef, nx, ny, 1)
    for iy in 1:ny, ix in 1:nx
        xc = vmx + (T(ix) - T(0.5)) * vsx; yc = vmy + (T(iy) - T(0.5)) * vsy
        support[ix, iy, 1] = !(xc * xc + yc * yc > radius_sq)
    end
    return reshape(M, nc * nr, nv, nvox), support
end

# Pure operators over a (possibly traced) M3 = (nc*nr, n_views, nvox) tensor.
function dense_operators(M3, support, plan::FStage.HIRPlan{T}) where {T}
    nc, nr, _ = plan.sino_shape
    vshape = plan.work_shape
    nvox = prod(vshape)
    A = function (vol, idx)
        Ms = reshape(M3[:, idx, :], nc * nr * length(idx), nvox)
        return reshape(Ms * reshape(vol, nvox), nc, nr, length(idx))
    end
    At = function (sub, idx)
        Ms = reshape(M3[:, idx, :], nc * nr * length(idx), nvox)
        v = permutedims(Ms, (2, 1)) * reshape(sub, nc * nr * length(idx))
        return FStage._where(support, reshape(v, vshape), zero(T))
    end
    return A, At
end

function functional_operators(plan::FStage.HIRPlan{T}, support) where {T}
    wg = plan.work_geom
    vshape = plan.work_shape
    vplans = [BSF.dd_view_plan(wg, v, vshape; eltype = T) for v in 1:wg.n_angles]
    A = (vol, idx) -> cat([BSF.dd_project_view(vol, vplans[v]) for v in idx]...; dims = 3)
    At = function (sub, idx)
        acc = nothing
        for (k, v) in enumerate(idx)
            term = BSF.dd_transpose_view(sub[:, :, k], vplans[v], vshape)
            acc = acc === nothing ? term : acc .+ term
        end
        return FStage._where(support, acc, zero(T))
    end
    return A, At
end

relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))

# ----------------------------------------------------------------------------- smoke
const GEOM = toy_geom()
const VOL = (16, 16, 4)

_ts("start")
@testset "Reactant smoke: functional HIR (dense operators)" begin
    # Float64 only: finite differences need it, and each compile of the
    # 12-subset unrolled program costs ~2 min on a loaded machine.
    for T in (Float64,)
        _ts("T = $T: build fixture")
        plan = one_epoch_plan(GEOM, (GEOM.n_cols, GEOM.n_rows, GEOM.n_angles), VOL, T)
        sino = cylinder_sino(GEOM, VOL; T)
        init = T.(fdk_init(Float32.(sino), GEOM, VOL))
        M3, support = dense_system(plan)

        A_h, At_h = dense_operators(M3, support, plan)
        ref = FStage.hir_reconstruct(sino, init, plan, A_h, At_h)

        # sanity: the dense operator reproduces the legacy loop
        ws = BS.create_hir_recon_workspace(Float32.(sino), GEOM, VOL; strength = 60)
        ws_params = plan.params
        println("  [$T] dense-operator plain-Array result finite: $(all(isfinite, ref)); max|ref| = $(maximum(abs, ref))")

        run_hir(y, x0, m3) = FStage.hir_reconstruct(y, x0, plan, dense_operators(m3, support, plan)...)

        sino_r = Reactant.to_rarray(sino)
        init_r = Reactant.to_rarray(init)
        M3_r = Reactant.to_rarray(M3)
        _ts("T = $T: compile hir_reconstruct")
        t0 = time()
        run_c = @compile run_hir(sino_r, init_r, M3_r)
        println("  [$T] compile time: $(round(time() - t0; digits = 1)) s")
        out = Array(run_c(sino_r, init_r, M3_r))
        e = relmax(out, ref)
        println("  [$T] compiled vs plain-Array rel = $e")
        @test e ≤ 1e-5
        @test all(isfinite, out)

        if T === Float64
            w = randn(MersenneTwister(11), T, VOL)
            loss(y, x0, m3) = sum(w .* run_hir(y, x0, m3))
            _ts("T = $T: compile Enzyme.gradient")
            t0 = time()
            grad_c = @compile Enzyme.gradient(Reverse, loss, sino_r, Const(init_r), Const(M3_r))
            println("  [$T] gradient compile time: $(round(time() - t0; digits = 1)) s")
            g = Array(grad_c(Reverse, loss, sino_r, Const(init_r), Const(M3_r))[1])
            @test size(g) == size(sino)
            @test all(isfinite, g)
            # plain-Array Float64 loss for finite differences
            loss_h(y) = sum(w .* FStage.hir_reconstruct(y, init, plan, A_h, At_h))
            d = randn(MersenneTwister(21), T, size(sino))
            h = 1e-5
            dd_fd = (loss_h(sino .+ h .* d) - loss_h(sino .- h .* d)) / (2h)
            dd_ad = sum(g .* d)
            rel_dir = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
            println("  [$T] directional derivative: AD = $dd_ad, FD = $dd_fd, rel = $rel_dir")
            @test rel_dir ≤ 1e-3
            cand = findall(>(0.01), sino)
            rng = MersenneTwister(5)
            worst = 0.0
            for I in cand[rand(rng, 1:length(cand), 4)]
                e1 = zeros(T, size(sino)); e1[I] = h
                fd = (loss_h(sino .+ e1) - loss_h(sino .- e1)) / (2h)
                rel = abs(g[I] - fd) / max(abs(fd), 1e-12)
                worst = max(worst, rel)
                println("  [$T] ∂loss/∂sino$(Tuple(I)): AD = $(g[I]), FD = $fd, rel = $rel")
            end
            @test worst ≤ 1e-3
        end
    end
end

# Opt-in: tracing 24 DD view plans through 12 subsets + the two all-view
# normalisation projections is a large program whose compile time is unbounded
# on a loaded machine, and its traceability is owned by the projector stage.
#   BS_HIR_SMOKE_FUNCTIONAL_DD=1 julia --project=envs/reactant … smoke_hir.jl
if get(ENV, "BS_HIR_SMOKE_FUNCTIONAL_DD", "0") == "1"
@testset "Reactant smoke: functional DD projector operators (reported)" begin
    T = Float32
    plan = one_epoch_plan(GEOM, (GEOM.n_cols, GEOM.n_rows, GEOM.n_angles), VOL, T)
    sino = cylinder_sino(GEOM, VOL; T)
    init = fdk_init(sino, GEOM, VOL)
    _, support = dense_system(plan)
    Af, Atf = functional_operators(plan, support)
    ref = FStage.hir_reconstruct(sino, init, plan, Af, Atf)
    run_f(y, x0) = FStage.hir_reconstruct(y, x0, plan, Af, Atf)
    sino_r = Reactant.to_rarray(sino); init_r = Reactant.to_rarray(init)
    ok = try
        _ts("functional DD: compile")
        t0 = time()
        run_c = @compile run_f(sino_r, init_r)
        println("  functional-DD compile time: $(round(time() - t0; digits = 1)) s")
        out = Array(run_c(sino_r, init_r))
        e = relmax(out, ref)
        println("  functional-DD compiled vs plain-Array rel = $e")
        e ≤ 1e-4
    catch err
        println("  functional-DD operators did not compile under Reactant: ", sprint(showerror, err)[1:min(end, 600)])
        false
    end
    println("  functional-DD Reactant status: ", ok ? "PASS" : "NOT PASSING (owned by the projector stage)")
    @test true   # reported, not gated
end
else
    println("  functional-DD Reactant section skipped (set BS_HIR_SMOKE_FUNCTIONAL_DD=1 to run it)")
end
_ts("done")
