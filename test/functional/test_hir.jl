# Standalone contract tests for the functional HIR stage (src/functional/hir.jl)
# against the legacy `reconstruct!(::HIRReconWorkspace)` oracle.
#
#   julia --project=. -t 2 test/functional/test_hir.jl
#
# The stage is exercised with the LEGACY projector pair wrapped as pure
# allocating operators (see `legacy_operators`), so every difference measured
# here is attributable to the stage itself, not to the projector port.

using Test
using BasisSimulator
using LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "src", "functional", "hir.jl"))
end

const _T0 = time()
_ts(label) = (println(stderr, "[test_hir t=$(round(time() - _T0; digits = 1))s] ", label); flush(stderr))

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

# Mirrors `_toy_hir_setup` in test/api.jl:1221-1232.
function toy_setup(; matrix_size = (16, 16, 4), strength = 60, n_angles = 24,
                   T = Float32, projector = :dd_fast)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0,
    )
    geom = BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = 20.0, z_cm = 5.0)
    sino = zeros(T, geom.n_cols, geom.n_rows, geom.n_angles)
    ws = BS.create_hir_recon_workspace(sino, geom, matrix_size; strength, projector)
    plan = FStage.hir_plan(geom, size(sino), matrix_size, strength; projector, T)
    return (; scanner, geom, sino, ws, plan, matrix_size, strength)
end

# A physical-looking sinogram: a water-like cylinder projected through the
# object geometry, plus a small fixed-seed Gaussian perturbation.
function cylinder_sino(geom, matrix_size; T = Float32, μ = T(0.02), noise = T(0.003), seed = 0x484952)
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

# Wrap the legacy projector pair exactly as `reconstruct!` calls it: subset
# geometry from `work_geom`, HIR row-tiled forward kernel, exact-DD transpose
# with `active_z = 1:nz_work, circular_support = true` (Siddon: legacy matched
# voxel-driven backprojection without circular support).
function legacy_operators(plan::FStage.HIRPlan{T}) where {T}
    wg = plan.work_geom
    nc, nr = plan.sino_shape[1], plan.sino_shape[2]
    nz_work = plan.work_shape[3]
    proj = plan.projector
    A = function (vol, idx)
        g = BS.create_subset_geometry(wg, collect(Int, idx))
        out = zeros(T, nc, nr, length(idx))
        BS._project_mono_hir!(proj, out, vol, g)
        return out
    end
    At = function (sub, idx)
        g = BS.create_subset_geometry(wg, collect(Int, idx))
        out = zeros(T, plan.work_shape)
        if proj === :siddon
            BS._backproject_mono!(proj, out, sub, g)
        else
            BS._backproject_mono!(proj, out, sub, g;
                                  active_z = 1:nz_work, circular_support = true)
        end
        return out
    end
    return A, At
end

relmax(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))

# The FUNCTIONAL projector pair (src/functional/dd_projector.jl) wrapped with
# the same contract: per-view plans on `work_geom`, views concatenated along
# dim 3, transpose summed over the subset's views and restricted to the
# circular support with the arithmetic of `dd_backproject!(circular_support =
# true)` (`vmx + (ix − ½)·vsx`, pitch computed in `T`).
const BSF = BasisSimulator.Functional
function functional_operators(plan::FStage.HIRPlan{T}) where {T}
    wg = plan.work_geom
    vshape = plan.work_shape
    vplans = [BSF.dd_view_plan(wg, v, vshape; eltype = T) for v in 1:wg.n_angles]
    nx, ny = vshape[1], vshape[2]
    b = wg.fov
    vmx = T(-b[1] / 2); vmy = T(-b[2] / 2)
    vsx = T(b[1]) / T(nx); vsy = T(b[2]) / T(ny)
    radius_sq = T(min(b[1], b[2]) / 2)^2
    support = Array{Bool, 3}(undef, nx, ny, 1)
    for iy in 1:ny, ix in 1:nx
        xc = vmx + (T(ix) - T(0.5)) * vsx
        yc = vmy + (T(iy) - T(0.5)) * vsy
        support[ix, iy, 1] = !(xc * xc + yc * yc > radius_sq)
    end
    A = function (vol, idx)
        views = [BSF.dd_project_view(vol, vplans[v]) for v in idx]
        return cat(views...; dims = 3)
    end
    At = function (sub, idx)
        acc = nothing
        for (k, v) in enumerate(idx)
            term = BSF.dd_transpose_view(sub[:, :, k], vplans[v], vshape)
            acc = acc === nothing ? term : acc .+ term
        end
        return ifelse.(support, acc, zero(T))
    end
    return A, At
end

# -----------------------------------------------------------------------------
_ts("start")
@testset "functional HIR vs legacy reconstruct!(HIRReconWorkspace)" begin

    @testset "plan mirrors the workspace bookkeeping" begin
        s = toy_setup(strength = 60)
        @test s.plan.work_shape == size(s.ws.work_volume)
        @test s.plan.output_z == s.ws.output_z
        @test s.plan.work_shape[3] > s.matrix_size[3]          # toy has a z-halo
        @test s.plan.n_subsets == 12
        @test s.plan.subset_size == 2
        @test [v[1:length(sv)] for (v, sv) in zip(s.plan.subset_views, s.ws.subsets)] == s.ws.subsets
        @test all(all(==(1f0), v) for v in s.plan.subset_valid)
        @test s.plan.λ == Float32(s.ws.params.lambda) * 0.1f0
        @test s.plan.reg_per_subset
        @test s.plan.reg_views_scale == 24f0 / 1000f0
        s0 = toy_setup(strength = 0)
        @test s0.plan.nepochs == 0
        @test s0.plan.work_shape == s0.matrix_size
        @test s0.plan.output_z == 1:4
    end

    @testset "strength 0: bit-identical to the FOV-masked init" begin
        s = toy_setup(strength = 0)
        sino = cylinder_sino(s.geom, s.matrix_size)
        init = fdk_init(sino, s.geom, s.matrix_size)
        A, At = legacy_operators(s.plan)
        out = FStage.hir_reconstruct(sino, init, s.plan, A, At)
        ref = copy(BS.reconstruct!(s.ws, sino, s.geom; init_volume = init))
        @test out == ref
        # …and to the internal-FDK path (FDK init is what the workspace builds).
        s2 = toy_setup(strength = 0)
        ref2 = copy(BS.reconstruct!(s2.ws, sino, s2.geom))
        @test out == ref2
    end

    _ts("parity 60/100")
    parity = Dict{Int, NamedTuple}()
    @testset "strength $strength parity (legacy-wrapped operators)" for strength in (60, 100)
        s = toy_setup(; strength)
        sino = cylinder_sino(s.geom, s.matrix_size)
        init = fdk_init(sino, s.geom, s.matrix_size)
        A, At = legacy_operators(s.plan)
        ref = copy(BS.reconstruct!(s.ws, sino, s.geom; init_volume = init))
        # (a) fully functional: weights from the operators
        out = FStage.hir_reconstruct(sino, init, s.plan, A, At)
        # (b) legacy W_proj / V_inv injected: isolates the loop from the weights
        out_b = FStage.hir_reconstruct(sino, init, s.plan, A, At, s.ws.W_proj, s.ws.V_inv)
        e_a = relmax(out, ref); e_b = relmax(out_b, ref)
        parity[strength] = (; full = e_a, injected = e_b, ref_max = maximum(abs.(ref)),
                              changed = relmax(ref, ifelse.(s.plan.fov_mask, init, -0.04f0)))
        @test all(isfinite, out)
        @test size(out) == s.matrix_size
        @test e_a ≤ 1e-4
        @test e_b ≤ 1e-4
        # HIR actually moved away from the init (the gate is not vacuous)
        @test parity[strength].changed > 1e-3
        # the FOV sentinel is reproduced exactly
        @test all(out[.!s.plan.fov_mask[:, :, 1], :] .== -0.04f0)
        @test all(ref[.!s.plan.fov_mask[:, :, 1], :] .== -0.04f0)
    end
    for k in sort(collect(keys(parity)))
        println("  parity strength $k: full-functional rel = $(parity[k].full), " *
                "injected-weights rel = $(parity[k].injected), " *
                "HIR-vs-init rel change = $(parity[k].changed)")
    end

    @testset "W_proj closed form (rtol 2e-6)" begin
        s = toy_setup(strength = 60)
        A, At = legacy_operators(s.plan)
        like = zeros(Float32, s.plan.work_shape)
        W = FStage.ray_normalization(s.plan, A, like)
        support = ones(Float32, s.plan.work_shape)
        BS.apply_fov_mask!(support, s.plan.work_geom; sentinel_μ = 0.0f0)
        ray_sum = BS._project_mono(:dd_fast, support, s.plan.work_geom)
        expected = @. ifelse(ray_sum > 1.0f-8, inv(ray_sum), 0.0f0)
        @test W ≈ expected rtol = 2.0f-6 atol = 2.0f-6
        @test W ≈ s.ws.W_proj rtol = 2.0f-6 atol = 2.0f-6
        V = FStage.image_weights(s.plan, At, s.sino)
        @test V == s.ws.V_inv          # same kernel, same call → bit-identical
        # statistical weights closed form
        sino = cylinder_sino(s.geom, s.matrix_size)
        sw = FStage.projection_weights(sino, s.plan)
        @test sw == exp.(.-clamp.(sino, 0f0, 10f0)) .+ 1f-6
    end

    @testset "deterministic and pure" begin
        s = toy_setup(strength = 60)
        sino = cylinder_sino(s.geom, s.matrix_size)
        init = fdk_init(sino, s.geom, s.matrix_size)
        A, At = legacy_operators(s.plan)
        sino_c, init_c = copy(sino), copy(init)
        o1 = FStage.hir_reconstruct(sino, init, s.plan, A, At)
        o2 = FStage.hir_reconstruct(sino, init, s.plan, A, At)
        @test o1 == o2
        @test sino == sino_c && init == init_c   # inputs untouched
    end

    @testset "ragged subsets: padding is exact vs legacy (26 views)" begin
        s = toy_setup(strength = 60, n_angles = 26)
        @test s.plan.subset_size == 3
        @test sum(length, s.ws.subsets) == 26
        @test count(v -> v[1, 1, end] == 0f0, s.plan.subset_valid) == 10   # 2 subsets of 3, 10 of 2
        sino = cylinder_sino(s.geom, s.matrix_size)
        init = fdk_init(sino, s.geom, s.matrix_size)
        A, At = legacy_operators(s.plan)
        ref = copy(BS.reconstruct!(s.ws, sino, s.geom; init_volume = init))
        out = FStage.hir_reconstruct(sino, init, s.plan, A, At, s.ws.W_proj, s.ws.V_inv)
        e = relmax(out, ref)
        println("  ragged 26-view parity (injected weights) rel = $e")
        @test e ≤ 1e-5
    end

    @testset "air_reference path parity" begin
        s0 = toy_setup(strength = 60)
        n_col, n_row = size(s0.sino, 1), size(s0.sino, 2)
        aref = 0.5f0 .+ rand(MersenneTwister(7), Float32, n_col, n_row)
        plan = FStage.hir_plan(s0.geom, size(s0.sino), s0.matrix_size, 60; air_reference = aref)
        sino = cylinder_sino(s0.geom, s0.matrix_size)
        init = fdk_init(sino, s0.geom, s0.matrix_size)
        A, At = legacy_operators(plan)
        sw = FStage.projection_weights(sino, plan)
        @test sw == reshape(aref, n_col, n_row, 1) .* exp.(.-clamp.(sino, 0f0, 10f0)) .+ 1f-6
        ref = copy(BS.reconstruct!(s0.ws, sino, s0.geom; init_volume = init, air_reference = aref))
        out = FStage.hir_reconstruct(sino, init, plan, A, At, s0.ws.W_proj, s0.ws.V_inv)
        e = relmax(out, ref)
        println("  air_reference parity (injected weights) rel = $e")
        @test e ≤ 1e-5
    end

    @testset "siddon path: per-epoch Huber cadence and λ scale" begin
        s = toy_setup(strength = 60, projector = :siddon)
        @test !s.plan.reg_per_subset
        @test s.plan.λ == 4f0
        sino = cylinder_sino(s.geom, s.matrix_size)
        init = fdk_init(sino, s.geom, s.matrix_size)
        A, At = legacy_operators(s.plan)
        ref = copy(BS.reconstruct!(s.ws, sino, s.geom; init_volume = init))
        out = FStage.hir_reconstruct(sino, init, s.plan, A, At, s.ws.W_proj, s.ws.V_inv)
        e = relmax(out, ref)
        println("  siddon parity (injected weights) rel = $e")
        @test e ≤ 1e-5
    end

    @testset "Huber gradient vs compute_huber_gradient!" begin
        x = 0.05f0 .* randn(MersenneTwister(3), Float32, 9, 7, 5)
        g_ref = similar(x); BS.compute_huber_gradient!(g_ref, x, 0.06f0)
        g = FStage.huber_gradient(x, 0.06f0)
        @test g == g_ref
        @test iszero(FStage.huber_gradient(fill(0.25f0, 9, 7, 5), 0.06f0))
        # degenerate axes (nz == 1) still work
        x1 = 0.05f0 .* randn(MersenneTwister(4), Float32, 6, 5, 1)
        g1_ref = similar(x1); BS.compute_huber_gradient!(g1_ref, x1, 0.06f0)
        @test FStage.huber_gradient(x1, 0.06f0) == g1_ref
    end

    _ts("texture gate")
    @testset "HIR60 damps high-pass texture vs FDK (api.jl gate, reduced views)" begin
        scanner = BS.EICTScanner(
            source_to_isocenter = 610.0, source_to_detector = 1113.0,
            detector_rows = 24, detector_cols = 64,
            detector_row_size = 0.353, detector_col_size = 1.0,
        )
        n_angles = 48
        geom_out = BS.CTGeometry(scanner; n_angles, n_rows = 18, n_cols = 64, fov_cm = 20.0, z_cm = 0.44)
        geom_object = BS.CTGeometry(scanner; n_angles, n_rows = 18, n_cols = 64, fov_cm = 20.0, z_cm = 1.0)
        object = zeros(Float32, 32, 32, 25)
        for k in axes(object, 3), j in axes(object, 2), i in axes(object, 1)
            (i - 16.5)^2 + (j - 16.5)^2 < 12^2 && (object[i, j, k] = 0.02f0)
        end
        sino = BS._project_mono(:dd_fast, object, geom_object)
        noisy = sino .+ 0.003f0 .* randn(MersenneTwister(0x484952), Float32, size(sino))
        vol_shape = (32, 32, 11)
        plan = FStage.hir_plan(geom_out, size(noisy), vol_shape, 60)
        A, At = legacy_operators(plan)
        init = fdk_init(noisy, geom_out, vol_shape)
        hir = FStage.hir_reconstruct(noisy, init, plan, A, At)
        ws = BS.create_hir_recon_workspace(noisy, geom_out, vol_shape; strength = 60)
        ref = copy(BS.reconstruct!(ws, noisy, geom_out; init_volume = init))
        e = relmax(hir, ref)
        println("  texture fixture parity rel = $e")
        @test e ≤ 1e-4
        inner = falses(32, 32); inner[10:23, 10:23] .= true
        function hp_std(vol, k)
            img = @view vol[:, :, k]
            hp = img .- (circshift(img, (1, 0)) .+ circshift(img, (-1, 0)) .+
                         circshift(img, (0, 1)) .+ circshift(img, (0, -1))) ./ 4
            std(hp[inner])
        end
        fbp_hp = [hp_std(init, k) for k in axes(init, 3)]
        hir_hp = [hp_std(hir, k) for k in axes(hir, 3)]
        println("  hp texture: FDK mean = $(mean(fbp_hp)), HIR mean = $(mean(hir_hp)), ratio = $(mean(hir_hp) / mean(fbp_hp))")
        @test mean(hir_hp) < 0.9 * mean(fbp_hp)
        @test max(hir_hp[1], hir_hp[end]) / hir_hp[6] < 1.25
        @test all(isfinite, hir)
    end

    _ts("functional projector operators")
    @testset "functional DD operators: strength $strength parity vs legacy" for strength in (60, 100)
        s = toy_setup(; strength)
        sino = cylinder_sino(s.geom, s.matrix_size)
        init = fdk_init(sino, s.geom, s.matrix_size)
        Af, Atf = functional_operators(s.plan)
        Al, Atl = legacy_operators(s.plan)
        # operator-level parity on the halo'd work grid
        work = FStage.hir_seed(init, s.plan)
        idx = s.plan.subset_views[1]
        e_A = relmax(Af(work, idx), Al(work, idx))
        sub = sino[:, :, idx]
        e_At = relmax(Atf(sub, idx), Atl(sub, idx))
        ref = copy(BS.reconstruct!(s.ws, sino, s.geom; init_volume = init))
        out = FStage.hir_reconstruct(sino, init, s.plan, Af, Atf)
        e = relmax(out, ref)
        println("  functional-operator parity, strength $strength: A rel = $e_A, Aᵀ rel = $e_At, HIR rel = $e")
        @test e_A ≤ 1e-4
        @test e_At ≤ 1e-4
        @test e ≤ 1e-4
        @test all(isfinite, out)
    end

    _ts("finite differences")
    @testset "Float64 finite-difference smoothness, 1 epoch" begin
        T = Float64
        s = toy_setup(strength = 60, T = T)
        p60 = BS.get_hir_params(60)
        params1 = BS.HIRParams(60, p60.lambda, 1, p60.n_subsets, p60.huber_delta,
                               p60.relaxation, p60.target_noise_reduction)
        plan = FStage.hir_plan(s.geom, size(s.sino), s.matrix_size, params1; T)
        @test plan.nepochs == 1
        sino = cylinder_sino(s.geom, s.matrix_size; T)
        init = T.(fdk_init(Float32.(sino), s.geom, s.matrix_size))
        A, At = legacy_operators(plan)
        W = FStage.ray_normalization(plan, A, zeros(T, plan.work_shape))
        V = FStage.image_weights(plan, At, sino)
        w = randn(MersenneTwister(11), T, s.matrix_size)
        loss(y) = sum(w .* FStage.hir_reconstruct(y, init, plan, A, At, W, V))
        rng = MersenneTwister(5)
        # probe rays that actually see the object (non-trivial weights)
        cand = findall(>(0.01), sino)
        probes = cand[rand(rng, 1:length(cand), 6)]
        h = 1e-4
        worst = 0.0
        for I in probes
            function cd(hh)
                e = zeros(T, size(sino)); e[I] = hh
                (loss(sino .+ e) - loss(sino .- e)) / (2hh)
            end
            d1 = cd(h); d2 = cd(h / 2)
            rich = (4d2 - d1) / 3
            rel = abs(d2 - rich) / max(abs(rich), 1e-12)
            worst = max(worst, rel)
            println("  FD probe $(Tuple(I)): cd(h) = $d1, cd(h/2) = $d2, Richardson = $rich, rel = $rel")
        end
        println("  worst central-difference vs Richardson rel = $worst")
        @test worst ≤ 1e-3
    end
end
_ts("done")
