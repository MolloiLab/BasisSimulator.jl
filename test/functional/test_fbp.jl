# =============================================================================
# Functional FBP/FDK stage — parity against the legacy AcceleratedKernels oracle
#
# Run standalone:  julia --project=. -t 2 test/functional/test_fbp.jl
# =============================================================================

# Also includable from test/functional/runtests.jl (inside a @testset): `include`
# evaluates this file at Main top level, so the `using`/`module`/`const` lines
# below are fine there; every helper is prefixed `fbp_` to avoid clashing with
# other test files' bindings.
using Test, Statistics, LinearAlgebra, Random
using BasisSimulator
const BS = BasisSimulator

# Prefer the stage as wired into BasisSimulator.Functional; fall back to a
# scratch module that includes the stage file directly.
if !(isdefined(BasisSimulator, :Functional) && isdefined(BasisSimulator.Functional, :fbp_plan))
    @eval module FStageFBP
        using BasisSimulator, LinearAlgebra, Statistics
        const BS = BasisSimulator
        include(joinpath(@__DIR__, "..", "..", "src", "functional", "reconstruction", "fbp.jl"))
    end
end
const F = (isdefined(BasisSimulator, :Functional) && isdefined(BasisSimulator.Functional, :fbp_plan)) ?
    BasisSimulator.Functional : FStageFBP
println("test_fbp: testing ", F)

# ── metrics ───────────────────────────────────────────────────────────────────
# rel_max  = max|a − b| / max|b|        (scale-free worst case)
# rel_mean = mean|a − b| / mean|b|      (scale-free average)
fbp_rel_max(a, b)  = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))
fbp_rel_mean(a, b) = mean(abs.(a .- b)) / max(mean(abs.(b)), eps(eltype(b)))

const FBP_PARITY = Dict{String, Float64}()   # collected numbers for the report
fbp_record!(k, v) = (FBP_PARITY[k] = Float64(v); v)

# ── fixtures ──────────────────────────────────────────────────────────────────
# test/api.jl:1098-1108 `_toy_fdk_setup` (Scanner default detector_shape = :arc)
# NOTE: with the legacy z_cm = 5.0 the 8 detector rows (0.8 cm of z at
# isocentre) cover NONE of the ±2.5 cm voxel slab, so every voxel is out of
# bounds and the legacy backprojection is identically zero — fine for the
# contract tests, vacuous for parity.  `z_cm = 0.4` keeps every voxel in-bounds
# and is used wherever a non-trivial reference is required.
function fbp_toy_geom(; shape = :arc, z_cm = 5.0)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 32,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = shape)
    return BS.CTGeometry(scanner; n_angles = 16, fov_cm = 20.0, z_cm = z_cm)
end
const FBP_TOY_VOL = (16, 16, 4)
const FBP_TOY_ZCM = 0.4

# test/geometry.jl:525-550 arc fixture (128 cols × 16 rows, 96 views)
function fbp_arc128_scanner()
    BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 16, detector_cols = 128,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = :arc)
end
fbp_arc128_geom() = BS.CTGeometry(fbp_arc128_scanner(); n_angles = 96, fov_cm = 12.8)

const FBP_FILTERS5 = (:ram_lak, :shepp_logan, :cosine, :hamming, :hann)
const FBP_FILTERS_ALL = (FBP_FILTERS5..., :standard, :soft, :bone)

fbp_t_start = time()

@testset "Functional FBP/FDK stage" begin

    # -------------------------------------------------------------------------
    @testset "plan construction" begin
        geom = fbp_toy_geom()
        plan = F.fbp_plan(geom, FBP_TOY_VOL)
        @test plan isa F.FBPPlan{Float32, true}
        @test F.is_arc_plan(plan)
        @test (plan.n_col, plan.n_row, plan.n_view) == (32, 8, 16)
        @test (plan.nx, plan.ny, plan.nz) == FBP_TOY_VOL
        @test length(plan.kernel) == 63 == 2 * 32 - 1          # full support rule
        @test size(plan.tensors.H) == (32, 32)
        @test size(plan.tensors.cos_weights) == (32, 8)
        @test size(plan.tensors.fov_outside) == (16, 16, 1)
        @test plan.sentinel == -0.04f0

        # kernel length rule vs legacy for a few (n_col, cutoff) pairs
        for (n, c) in ((32, 1.0), (32, 0.7), (128, 1.0), (128, 0.5), (888, 0.9), (16, 1.0))
            raw = max(Int(ceil(2 * n * c)), 64)
            @test F._kernel_length(n, c) == min(raw + (1 - raw % 2), 2n - 1)
        end
        # kernel taps are the legacy taps (window + arc scaling included)
        k_ref = BS.create_spatial_kernel(63, BS.RampFilter(), Float32(geom.pixel_size))
        BS.equiangular_kernel_scale!(k_ref, geom.pixel_size / geom.SAD)
        @test plan.kernel == k_ref

        # flat plan, Float64, custom fov, FilterType instance
        pf = F.fbp_plan(fbp_toy_geom(shape = :flat), FBP_TOY_VOL; T = Float64,
            filter = BS.HannFilter(), cutoff = 0.7, fov = (10.0, 10.0, 5.0))
        @test pf isa F.FBPPlan{Float64, false}
        @test !F.is_arc_plan(pf)
        @test length(pf.kernel) == F._kernel_length(32, 0.7)
        @test pf.fov == (10.0, 10.0, 5.0)

        # Toeplitz identity: H * e_j reproduces the legacy tap loop column-wise
        K = length(plan.kernel); kh = K ÷ 2
        for c in (1, 7, 32), j in (1, 5, 32)
            d = j - c
            expect = -kh <= d <= kh ? plan.kernel[d + kh + 1] : 0f0
            @test plan.tensors.H[c, j] == expect
        end

        # replacement-tensor constructor keeps everything else
        p2 = F.FBPPlan(plan; tensors = plan.tensors)
        @test p2.tensors === plan.tensors && p2.pi_over_angles == plan.pi_over_angles
        @test_throws ArgumentError F.FBPPlan(plan; tensors = (H = plan.tensors.H,))

        # helical → informative error
        geom_h = BS.CTGeometry(fbp_arc128_scanner(); n_angles = 96, fov_cm = 12.8, pitch = 1.0, n_rotations = 3.0)
        @test BS.is_helical(geom_h)
        @test_throws ArgumentError F.fbp_plan(geom_h, (64, 64, 24))
        @test_throws ArgumentError F.fbp_plan(geom, (0, 16, 4))
        @test_throws ArgumentError F.fbp_plan(geom, FBP_TOY_VOL; cutoff = 0.0)
        # shape guards
        @test_throws DimensionMismatch F.filter_views(zeros(Float32, 31, 8, 16), plan)   # per view: any view count, the pixel grid must match
        @test size(F.filter_views(zeros(Float32, 32, 8, 15), plan)) == (32, 8, 15)
        @test_throws DimensionMismatch F.backproject(zeros(Float32, 31, 8, 16), plan)
        @test_throws DimensionMismatch F.fov_mask(zeros(Float32, 16, 16, 5), plan)
    end

    # -------------------------------------------------------------------------
    @testset "filter_views parity vs filter_sinogram!" begin
        rng = MersenneTwister(1)
        cases = (("toy-arc", fbp_toy_geom(z_cm = FBP_TOY_ZCM)), ("toy-flat", fbp_toy_geom(shape = :flat, z_cm = FBP_TOY_ZCM)),
                 ("arc128", fbp_arc128_geom()))
        for (name, geom) in cases, T in (Float32, Float64)
            tol = T === Float32 ? 1e-5 : 1e-12
            sino = randn(rng, T, geom.n_cols, geom.n_rows, geom.n_angles)
            worst = 0.0
            for f in FBP_FILTERS_ALL, cutoff in (1.0, 0.7)
                plan = F.fbp_plan(geom, FBP_TOY_VOL; filter = f, cutoff = cutoff, T = T)
                ref = BS.filter_sinogram(sino, geom; filter = BS.filter_from_symbol(f), cutoff = cutoff)
                out = F.filter_views(sino, plan)
                @test size(out) == size(sino) && eltype(out) === T
                r = fbp_rel_max(out, ref)
                worst = max(worst, r)
                @test r <= tol
            end
            fbp_record!("filter/$name/$T", worst)
            # purity
            s0 = copy(sino)
            F.filter_views(sino, F.fbp_plan(geom, FBP_TOY_VOL; T = T))
            @test sino == s0
        end
    end

    # -------------------------------------------------------------------------
    @testset "backproject parity vs backproject! (weighted / matched)" begin
        rng = MersenneTwister(2)
        # the legacy toy fixture (z_cm = 5.0) backprojects to all zeros — document it
        let geom = fbp_toy_geom(), filt = randn(rng, Float32, geom.n_cols, geom.n_rows, geom.n_angles)
            @test all(==(0), BS.backproject(filt, geom, FBP_TOY_VOL))
            @test all(==(0), F.backproject(filt, F.fbp_plan(geom, FBP_TOY_VOL)))
        end
        # (T, arc) pairs: every (T, arc) combination is a separate compiled
        # specialisation (~4-8 s CPU each); Float64-flat is not required by the
        # acceptance list and is skipped to keep the file's runtime down.
        for (name, geom, vs, Ts) in (("toy-arc", fbp_toy_geom(z_cm = FBP_TOY_ZCM), FBP_TOY_VOL, (Float64, Float32)),
                                     ("toy-flat", fbp_toy_geom(shape = :flat, z_cm = FBP_TOY_ZCM), FBP_TOY_VOL, (Float32,)),
                                     ("arc128", fbp_arc128_geom(), (24, 24, 6), (Float64,)))
            for T in Ts
                filt = randn(rng, T, geom.n_cols, geom.n_rows, geom.n_angles)
                plan = F.fbp_plan(geom, vs; T = T)
                for weighted in (true, false)
                    ref = BS.backproject(filt, geom, vs; weighted = weighted)
                    @test any(!=(0), ref)                       # fixture is non-trivial
                    for vb in (1, 5)
                        out = F.backproject(filt, plan; weighted = weighted, view_batch = vb)
                        @test size(out) == vs && eltype(out) === T
                        r = fbp_rel_max(out, ref)
                        @test r <= 1e-5
                        vb == 1 && fbp_record!("backproject/$name/$T/weighted=$weighted", r)
                    end
                end
                # single-view API sums to the full backprojection
                acc = zeros(T, vs)
                for a in 1:geom.n_angles
                    acc = acc .+ F.backproject_view(filt[:, :, a], plan, a)
                end
                @test fbp_rel_max(acc, BS.backproject(filt, geom, vs)) <= 1e-5
                # purity
                f0 = copy(filt)
                F.backproject(filt, plan)
                @test filt == f0
            end
        end
    end

    # -------------------------------------------------------------------------
    @testset "fov_mask parity vs apply_fov_mask!" begin
        geom = fbp_toy_geom()
        plan = F.fbp_plan(geom, FBP_TOY_VOL)
        vol = randn(MersenneTwister(3), Float32, FBP_TOY_VOL...)
        ref = BS.apply_fov_mask!(copy(vol), geom)
        out = F.fov_mask(vol, plan)
        @test out == ref
        @test F.fov_mask(vol, plan; sentinel = -1.0) == BS.apply_fov_mask!(copy(vol), geom; sentinel_μ = -1.0)
        @test count(plan.tensors.fov_outside) > 0 && count(!, plan.tensors.fov_outside) > 0
    end

    # -------------------------------------------------------------------------
    @testset "fdk vs reconstruct!(FDKReconWorkspace) — toy fixture contracts" begin
        for shape in (:arc, :flat)
            # exact legacy fixture (test/api.jl:1098-1108) → contract tests
            geom = fbp_toy_geom(shape = shape)
            plan = F.fbp_plan(geom, FBP_TOY_VOL)
            sino = randn(MersenneTwister(4), Float32, geom.n_cols, geom.n_rows, geom.n_angles)
            ws = BS.create_fdk_recon_workspace(sino, geom, FBP_TOY_VOL; filter = :ram_lak)
            ref = copy(BS.reconstruct!(ws, sino, geom))
            out = F.fdk(sino, plan)
            @test size(out) == FBP_TOY_VOL && eltype(out) === Float32
            @test all(isfinite, out)
            @test out == ref                                   # both are {0, sentinel}
            @test F.fdk(sino, plan) == out                     # deterministic
            s0 = copy(sino); F.fdk(sino, plan); @test sino == s0   # pure

            # zero sinogram → {0 inside FOV, sentinel outside}
            z = F.fdk(zeros(Float32, size(sino)), plan)
            @test all(v -> v == 0f0 || v ≈ -0.04f0, z)
            nx, ny, nz = FBP_TOY_VOL
            @test z[nx ÷ 2 + 1, ny ÷ 2 + 1, nz ÷ 2 + 1] == 0f0
            for k in 1:nz
                @test z[1, 1, k] ≈ -0.04f0 && z[nx, 1, k] ≈ -0.04f0
                @test z[1, ny, k] ≈ -0.04f0 && z[nx, ny, k] ≈ -0.04f0
            end

            # non-degenerate fixture (z_cm = 0.4) → real parity numbers
            geom_nd = fbp_toy_geom(shape = shape, z_cm = FBP_TOY_ZCM)
            plan_nd = F.fbp_plan(geom_nd, FBP_TOY_VOL)
            ws_nd = BS.create_fdk_recon_workspace(sino, geom_nd, FBP_TOY_VOL; filter = :ram_lak)
            ref_nd = copy(BS.reconstruct!(ws_nd, sino, geom_nd))
            @test any(v -> v != 0 && v != -0.04f0, ref_nd)        # non-trivial reference
            out_nd = F.fdk(sino, plan_nd)
            mx = maximum(abs.(out_nd .- ref_nd)); scale = maximum(abs.(ref_nd))
            fbp_record!("fdk/toy-$shape/Float32/max_abs_over_max_ref", mx / scale)
            fbp_record!("fdk/toy-$shape/Float32/mean_rel", fbp_rel_mean(out_nd, ref_nd))
            @test mx <= 1e-4 * scale
            @test fbp_rel_mean(out_nd, ref_nd) <= 1e-5
            @test fbp_rel_max(F.fdk(sino, plan_nd; view_batch = 5), ref_nd) <= 1e-5
            @test F.fdk(sino, plan_nd) == out_nd                     # deterministic
            # corner voxels carry the sentinel in every slice, centre does not
            for k in 1:nz
                @test out_nd[1, 1, k] ≈ -0.04f0 && out_nd[nx, 1, k] ≈ -0.04f0
                @test out_nd[1, ny, k] ≈ -0.04f0 && out_nd[nx, ny, k] ≈ -0.04f0
            end
            @test out_nd[nx ÷ 2 + 1, ny ÷ 2 + 1, nz ÷ 2 + 1] != -0.04f0
            @test out_nd[nx ÷ 2 + 1, ny ÷ 2 + 1, nz ÷ 2 + 1] != 0f0

            # non-trivial central bump → finite, non-uniform, matches legacy
            bump = zeros(Float32, size(sino))
            mid = geom.n_cols ÷ 2 + 1
            for v in 1:geom.n_angles, r in 1:geom.n_rows, c in 1:geom.n_cols
                bump[c, r, v] = 0.5f0 * exp(-((c - mid)^2) / (2 * 4f0^2))
            end
            ob = F.fdk(bump, plan_nd)
            @test all(isfinite, ob) && std(ob[plan_nd.tensors.fov_outside[:, :, 1] .== false, :]) > 0
            wsb = BS.create_fdk_recon_workspace(bump, geom_nd, FBP_TOY_VOL; filter = :ram_lak)
            @test fbp_rel_max(ob, BS.reconstruct!(wsb, bump, geom_nd)) <= 1e-4

            # Float64 end-to-end is tight (arc only — see the specialisation note above)
            if shape === :arc
                s64 = Float64.(sino)
                ws64 = BS.create_fdk_recon_workspace(s64, geom_nd, FBP_TOY_VOL; filter = :ram_lak)
                ref64 = copy(BS.reconstruct!(ws64, s64, geom_nd))
                out64 = F.fdk(s64, F.fbp_plan(geom_nd, FBP_TOY_VOL; T = Float64))
                r64 = fbp_rel_max(out64, ref64)
                fbp_record!("fdk/toy-$shape/Float64/rel_max", r64)
                @test r64 <= 1e-12
            end
        end
    end

    # -------------------------------------------------------------------------
    @testset "all filter kernels run and match legacy end-to-end" begin
        geom = fbp_toy_geom(z_cm = FBP_TOY_ZCM)
        sino = randn(MersenneTwister(5), Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        worst = 0.0
        for f in FBP_FILTERS_ALL
            plan = F.fbp_plan(geom, FBP_TOY_VOL; filter = f)
            ws = BS.create_fdk_recon_workspace(sino, geom, FBP_TOY_VOL; filter = f)
            ref = copy(BS.reconstruct!(ws, sino, geom))
            out = F.fdk(sino, plan)
            @test all(isfinite, out)
            r = fbp_rel_max(out, ref)
            worst = max(worst, r)
            @test r <= 1e-4
        end
        fbp_record!("fdk/toy-arc/all-filters/rel_max", worst)
    end

    # -------------------------------------------------------------------------
    @testset "arc axial fixture (test/geometry.jl:525-550): legacy parity + physical gates" begin
        scanner = fbp_arc128_scanner()
        nx, nz = 64, 48
        vol = zeros(Float32, nx, nx, nz)
        c = (nx + 1) / 2
        for k in 1:nz, j in 1:nx, i in 1:nx
            if (i - c)^2 + (j - c)^2 <= (0.35 * nx)^2
                vol[i, j, k] = 0.2f0
            end
        end
        geom = BS.CTGeometry(scanner; n_angles = 96, fov_cm = 12.8)
        sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.dd_forward_project!(sino, vol, geom; volume_extent = (12.8, 12.8, 9.6))

        # legacy default filter is StandardFilter(); match it explicitly
        vs = (64, 64, 8)
        ref = BS.fdk_reconstruct(sino, geom, vs)                       # default StandardFilter
        plan = F.fbp_plan(geom, vs; filter = BS.StandardFilter())
        rec = F.fdk(sino, plan)
        r = fbp_rel_max(rec, ref)
        fbp_record!("fdk/arc128-cylinder/Float32/rel_max", r)
        @test r <= 1e-4
        # batched-view graph variant
        @test fbp_rel_max(F.fdk(sino, plan; view_batch = 8), ref) <= 1e-4

        # same physical gates as the legacy test
        roi_c = rec[29:36, 29:36, 4]
        roi_e = rec[43:50, 29:36, 4]
        μc = sum(roi_c) / length(roi_c)
        μe = sum(roi_e) / length(roi_e)
        fbp_record!("fdk/arc128-cylinder/centre_mu", μc)
        fbp_record!("fdk/arc128-cylinder/radial_flatness", abs(μe - μc))
        @test abs(μc - 0.2) < 0.01
        @test abs(μe - μc) < 0.008

        # Float64 tight parity on the same physics
        s64 = Float64.(sino)
        ref64 = BS.fdk_reconstruct(s64, geom, vs)
        rec64 = F.fdk(s64, F.fbp_plan(geom, vs; filter = BS.StandardFilter(), T = Float64))
        r64 = fbp_rel_max(rec64, ref64)
        fbp_record!("fdk/arc128-cylinder/Float64/rel_max", r64)
        @test r64 <= 1e-12
    end
end

println("\n── parity numbers ─────────────────────────────────────────────")
for k in sort(collect(keys(FBP_PARITY)))
    println(rpad(k, 58), "  ", FBP_PARITY[k])
end
println("elapsed: ", round(time() - fbp_t_start; digits = 1), " s")
