#=
Parity tests: BasisSimulator.Functional VMI stage vs the legacy oracles.

Run standalone:  julia --project=. -t 2 test/functional/test_vmi.jl
=#
using Test
using BasisSimulator
using LinearAlgebra, Statistics
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "src", "functional", "reconstruction", "vmi", "cong_cmv.jl"))
end
const F = FStage

# ────────────────────────────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────────────────────────────

function vmi_fixture_spectra()
    scanner = BS.EICTScanner(
        source_to_isocenter = 625.6, source_to_detector = 1100.0,
        detector_rows = 4, detector_cols = 16,
        detector_row_size = 0.625, detector_col_size = 0.6,
        focal_spot_width = 1.0, focal_spot_length = 1.0, target_angle = 10.0,
        flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
        bowtie_filter = :ge_revolution_large, detector_material = :lumex,
        detector_depth = 3.0, fill_factor_row = 0.9, fill_factor_col = 0.9,
        electronic_noise = 0, detection_gain = 10.0,
    )
    prot(kVp) = BS.CTProtocol(kVp = kVp, mA = 400.0, views = 8, rotation_time = 0.5,
                              collimation_mm = 5.0, additional_filters = [("Al", 4.5)])
    sim_opts = BS.SimOptions(seed = 1234, projector = :dd_fast)
    e_L, w_L = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot(80);  scanner = scanner)
    e_H, w_H = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot(140); scanner = scanner)
    (e_L = Float64.(e_L), w_L = Float64.(w_L), e_H = Float64.(e_H), w_H = Float64.(w_H))
end

μρ_I(E) = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(E))
μρ_W(E) = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(E))

# Material-direct Cong basis (iodine, water) exactly as the notebooks built it.
function material_basis(sp; bowtie::Union{Nothing, Tuple{Int, Int}} = nothing)
    ŵ_L = Float32.(sp.w_L ./ sum(sp.w_L)); ŵ_H = Float32.(sp.w_H ./ sum(sp.w_H))
    p_L = Float32[μρ_I(E) for E in sp.e_L]; q_L = Float32[μρ_W(E) for E in sp.e_L]
    p_H = Float32[μρ_I(E) for E in sp.e_H]; q_H = Float32[μρ_W(E) for E in sp.e_H]
    if bowtie !== nothing
        # synthetic per-ray bowtie: extra water-equivalent thickness t(col,row) hardens the beam
        nc, nr = bowtie
        t = Float64[0.05 * ((c - (nc + 1) / 2)^2 / max(nc, 1) + 0.2 * abs(r - (nr + 1) / 2)) for c in 1:nc, r in 1:nr]
        harden(w, q) = begin
            out = Array{Float32}(undef, nc, nr, length(w))
            for r in 1:nr, c in 1:nc
                v = Float64.(w) .* exp.(-t[c, r] .* Float64.(q))
                out[c, r, :] .= Float32.(v ./ sum(v))
            end
            out
        end
        ŵ_L = harden(ŵ_L, q_L); ŵ_H = harden(ŵ_H, q_H)
    end
    (ŵ_L = ŵ_L, p_L = p_L, q_L = q_L, ŵ_H = ŵ_H, p_H = p_H, q_H = q_H)
end

# Polychromatic forward model in Float64:  p = -log Σ ŵ e^{-p_E a - q_E c}
function forward_logs(basis, a::AbstractArray, c::AbstractArray)
    S = size(a)
    per_ray = ndims(basis.ŵ_L) == 3
    p_L = Array{Float32}(undef, S); p_H = Array{Float32}(undef, S)
    for idx in CartesianIndices(S)
        col, row = idx[1], idx[2]
        wL = per_ray ? Float64.(basis.ŵ_L[col, row, :]) : Float64.(basis.ŵ_L)
        wH = per_ray ? Float64.(basis.ŵ_H[col, row, :]) : Float64.(basis.ŵ_H)
        aa = Float64(a[idx]); cc = Float64(c[idx])
        p_L[idx] = Float32(-log(sum(wL .* exp.(-(Float64.(basis.p_L) .* aa .+ Float64.(basis.q_L) .* cc)))))
        p_H[idx] = Float32(-log(sum(wH .* exp.(-(Float64.(basis.p_H) .* aa .+ Float64.(basis.q_H) .* cc)))))
    end
    (p_L, p_H)
end

# Gammex-like ray set, shape (4, 5, 5) = 100 rays.
function cong_ray_set()
    waters  = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    iodines = [0.0, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4]
    a = Float64[]; c = Float64[]
    for w in waters, i in iodines
        push!(a, i); push!(c, w)
    end
    # extremes
    append!(a, [0.0, 0.5, 0.0, 0.6, 1.0, 0.02]); append!(c, [60.0, 50.0, 0.1, 0.0, 30.0, 0.05])
    # pad to 100 with mid-range rays
    while length(a) < 100
        push!(a, 0.03 + 0.001 * length(a)); push!(c, 12.0 + 0.1 * length(a))
    end
    (reshape(a[1:100], 4, 5, 5), reshape(c[1:100], 4, 5, 5))
end

# Special measured rays that exercise the legacy gates / branches (overwrite slots).
function inject_special_rays!(p_L, p_H)
    specials = [
        (0f0, 0f0),            # exact air
        (1f-3, 3f-3),          # symmetric gate → air
        (6f-3, 0f0),           # NOT gated: |p_L| ≥ gate, tiny high → G(0) > 0 path
        (-0.05f0, 0.02f0),     # y_max ≤ 0 → (0, c̄)
        (-0.5f0, 0.1f0),       # invalid water bracket → (0, 0)
        (0.4f0, 0.4f0),        # p_H ≈ p_L: G(0) > 0 with negative-window root
        (2.0f0, 1.4f0),        # water-only-ish ray, sub-zero noise
        (3.0f0, 2.9f0),        # strong high channel → root below 0 or no root (fallback)
        (0f0, 0.01f0),         # p_L = 0 exactly, not gated: L = 0, y_max = 0 → (0, c̄ = 0)
    ]
    for (k, (l, h)) in enumerate(specials)
        p_L[end - k + 1] = l; p_H[end - k + 1] = h
    end
    (p_L, p_H)
end

function legacy_cong(basis, p_L, p_H; water_basis = (a = 0f0, c = 1f0))
    ws = BS.create_cong_workspace(p_L, basis)
    sy = zeros(Float32, size(p_L)); sc = zeros(Float32, size(p_L))
    BS.apply_cong!(ws, sy, sc, Float32.(p_L), Float32.(p_H); water_basis = water_basis)
    (sy, sc)
end

# Relative error with an absolute floor: Float32 root noise is ~1e-8 in a and
# ~1e-6 in c for BOTH solvers (their spectral sums round differently), so
# rays whose true value sits below the floor are compared absolutely.
const A_FLOOR = 1e-3   # g/cm² iodine  → absolute tolerance 1e-7 at 1e-4 rel
const C_FLOOR = 1e-2   # g/cm² water   → absolute tolerance 1e-6 at 1e-4 rel
relerr(x, y; floor = 1e-6) = abs(x - y) / max(abs(y), floor)
relerr_a(x, y) = relerr(x, y; floor = A_FLOOR)
relerr_c(x, y) = relerr(x, y; floor = C_FLOOR)

# ────────────────────────────────────────────────────────────────────────
#  Tests
# ────────────────────────────────────────────────────────────────────────

t_start = time()
sp = vmi_fixture_spectra()
water_basis = (a = 0f0, c = 1f0)

@testset "Functional VMI stage" begin

    @testset "cong_decompose parity (shared spectrum, Float32)" begin
        basis = material_basis(sp)
        a_true, c_true = cong_ray_set()
        p_L, p_H = forward_logs(basis, a_true, c_true)
        # noisy rays: Gaussian noise on a block of water/iodine rays
        rng_seed = 7; noise = Float32[sin(1000.0 * k + rng_seed) * 0.01 for k in 1:100]
        p_L[1:40] .+= noise[1:40]; p_H[1:40] .+= reverse(noise[1:40])
        inject_special_rays!(p_L, p_H)

        t0 = time(); a_leg, c_leg = legacy_cong(basis, p_L, p_H); t_leg = time() - t0
        plan = F.cong_plan(basis; water_basis = water_basis, T = Float32)
        @test plan.L_lo == 0f0   # W(-1) is NaN for this spectrum → legacy-effective bracket [0, L_hi]
        t0 = time(); st = F.cong_solve(p_L, p_H, plan); t_first = time() - t0
        t0 = time(); a_f, c_f = F.cong_decompose(p_L, p_H, plan); t_second = time() - t0
        println("  [cong shared] legacy apply_cong! $(round(t_leg; digits = 2)) s (incl. compile); cong_solve first call $(round(t_first; digits = 2)) s, second call $(round(t_second; digits = 3)) s")
        @test a_f == st.a && c_f == st.c
        @test size(a_f) == size(p_L)

        # branch bookkeeping
        n_main = count(st.main); n_air = count(st.air); n_badL = count(.!st.ok_L .& .!st.air)
        n_ymax = count(st.ymax_le0 .& st.ok_L .& .!st.air); n_fb = count(.!st.ok_y .& st.ok_L .& .!st.air .& .!st.ymax_le0)
        n_fixed = count(st.fixed_y .& st.main)
        println("  [cong shared] main=$(n_main) air=$(n_air) bad_water_bracket=$(n_badL) ymax≤0=$(n_ymax) fallback=$(n_fb) fixed_y=$(n_fixed)")
        @test n_air >= 2 && n_badL >= 1 && n_fb >= 1 && n_main >= 80

        # parity on every ray, by branch
        worst_main = 0.0; worst_other = 0.0
        for idx in eachindex(p_L)
            ea = relerr_a(Float64(a_f[idx]), Float64(a_leg[idx])); ec = relerr_c(Float64(c_f[idx]), Float64(c_leg[idx]))
            if st.main[idx]
                worst_main = max(worst_main, ea, ec)
            else
                worst_other = max(worst_other, ea, ec)
            end
        end
        println("  [cong shared] max rel err vs legacy: main-path $(worst_main), other branches $(worst_other)")
        @test worst_main <= 1e-4
        @test worst_other <= 1e-4
        # exact fallback values
        for idx in eachindex(p_L)
            if st.air[idx] || !st.ok_L[idx]
                @test a_leg[idx] == 0f0 && c_leg[idx] == 0f0 && a_f[idx] == 0f0 && c_f[idx] == 0f0
            elseif st.ymax_le0[idx]
                @test a_leg[idx] == 0f0 && a_f[idx] == 0f0
            end
        end
        # evidence on iteration sufficiency
        println("  [cong shared] max expansions used = $(Int(maximum(st.n_exp_used))) (n_expand=$(plan.n_expand)); " *
                "max |G| residual on main rays = $(maximum(st.G_res[st.main])); max water residual = $(maximum(abs.(st.W_res[st.ok_L])))")
        @test maximum(st.n_exp_used) < plan.n_expand
        # ground-truth recovery on the clean, in-range rays (legacy vs functional)
        clean = falses(size(p_L)); clean[41:56] .= true
        worst_truth_leg = 0.0; worst_truth_f = 0.0
        for idx in findall(clean)
            st.main[idx] || continue
            worst_truth_leg = max(worst_truth_leg, abs(a_leg[idx] - a_true[idx]) + abs(c_leg[idx] - c_true[idx]) / 100)
            worst_truth_f   = max(worst_truth_f,   abs(a_f[idx]   - a_true[idx]) + abs(c_f[idx]   - c_true[idx]) / 100)
        end
        println("  [cong shared] ground-truth recovery (|Δa| + |Δc|/100): legacy $(worst_truth_leg), functional $(worst_truth_f)")
        @test worst_truth_f <= max(2 * worst_truth_leg, 1e-5)
    end

    @testset "cong_decompose parity (per-ray bowtie ŵ, Float32)" begin
        nc, nr, nv = 4, 5, 5
        basis = material_basis(sp; bowtie = (nc, nr))
        a_true, c_true = cong_ray_set()
        p_L, p_H = forward_logs(basis, a_true, c_true)
        inject_special_rays!(p_L, p_H)
        a_leg, c_leg = legacy_cong(basis, p_L, p_H)
        plan = F.cong_plan(basis; water_basis = water_basis, T = Float32)
        @test ndims(plan.ŵ_L) == 3
        st = F.cong_solve(p_L, p_H, plan)
        worst = maximum(max(relerr_a(Float64(st.a[i]), Float64(a_leg[i])), relerr_c(Float64(st.c[i]), Float64(c_leg[i]))) for i in eachindex(p_L))
        println("  [cong per-ray] max rel err vs legacy (all rays) = $(worst); main rays = $(count(st.main))")
        @test worst <= 1e-4
    end

    @testset "cong_decompose Float64 + ground truth" begin
        basis = material_basis(sp)
        a_true, c_true = cong_ray_set()
        p_L, p_H = forward_logs(basis, a_true, c_true)
        plan64 = F.cong_plan(basis; water_basis = water_basis, T = Float64, n_bisect_y = 40, n_bisect_water = 40, n_newton_y = 3)
        @test plan64.L_lo == -1.0    # W(-1) is finite in Float64 → original legacy bracket
        st = F.cong_solve(Float64.(p_L), Float64.(p_H), plan64)
        st32 = F.cong_solve(p_L, p_H, F.cong_plan(basis; water_basis = water_basis, T = Float32))
        a_leg, c_leg = legacy_cong(basis, p_L, p_H)
        # Float32 legacy falls back (quintic Newton → NaN) on the most extreme
        # rays where Float64 still roots; compare where both precisions are main-path.
        both = st.main .& st32.main
        worst = 0.0
        for i in findall(both)
            worst = max(worst, relerr_a(st.a[i], Float64(a_leg[i])), relerr_c(st.c[i], Float64(c_leg[i])))
        end
        rooted = st.main .& .!st.fixed_y
        println("  [cong Float64] main rays: Float64 $(count(st.main)), Float32 $(count(st32.main)); max rel err vs Float32 legacy on the $(count(both)) shared main rays = $(worst); max |G| residual (rooted rays) = $(maximum(st.G_res[rooted]))")
        @test worst <= 2e-4
        @test maximum(st.G_res[rooted]) <= 1e-12
        # Ground-truth recovery: the quintic is a local Taylor model, so the
        # legacy algorithm itself is only accurate to ~1e-2 at 0.4 g/cm² iodine;
        # Float64 must recover the truth at least as well as Float32 legacy and
        # be tight on the low-iodine grid (a ≤ 0.005 g/cm²).
        grid_err(a, c) = maximum(abs(a[i] - a_true[i]) + abs(c[i] - c_true[i]) / 100 for i in 1:56 if st.main[i])
        low = [i for i in 1:56 if st.main[i] && a_true[i] <= 0.005]
        low_err = maximum(abs(st.a[i] - a_true[i]) + abs(st.c[i] - c_true[i]) / 100 for i in low)
        println("  [cong Float64] ground-truth recovery (|Δa| + |Δc|/100): grid Float64 $(grid_err(st.a, st.c)) vs legacy $(grid_err(a_leg, c_leg)); low-iodine rays $(low_err)")
        @test grid_err(st.a, st.c) <= 1.05 * grid_err(a_leg, c_leg) + 1e-6
        @test low_err <= 2e-4
    end

    @testset "IFT Jacobian / VJP vs finite differences (Float64)" begin
        basis = material_basis(sp)
        a_true = reshape([0.05, 0.2, 0.01, 0.1, 0.3, 0.02], 6, 1, 1)
        c_true = reshape([10.0, 20.0, 5.0, 30.0, 15.0, 2.0], 6, 1, 1)
        p_L32, p_H32 = forward_logs(basis, a_true, c_true)
        p_L = Float64.(p_L32); p_H = Float64.(p_H32)
        plan64 = F.cong_plan(basis; water_basis = water_basis, T = Float64, n_bisect_y = 40, n_bisect_water = 40, n_newton_y = 3, n_newton_water = 3)
        st = F.cong_solve(p_L, p_H, plan64)
        @test all(st.main)
        J_aL, J_aH, J_cL, J_cH = F.cong_decompose_jacobian(p_L, p_H, plan64; state = st)
        h = 1e-6
        fd(pl, ph) = F.cong_decompose(pl, ph, plan64)
        aLp, cLp = fd(p_L .+ h, p_H); aLm, cLm = fd(p_L .- h, p_H)
        aHp, cHp = fd(p_L, p_H .+ h); aHm, cHm = fd(p_L, p_H .- h)
        J_aL_fd = (aLp .- aLm) ./ (2h); J_cL_fd = (cLp .- cLm) ./ (2h)
        J_aH_fd = (aHp .- aHm) ./ (2h); J_cH_fd = (cHp .- cHm) ./ (2h)
        scale = maximum(abs.(vcat(vec(J_aL), vec(J_aH), vec(J_cL), vec(J_cH))))
        err(J, Jfd) = maximum(abs.(J .- Jfd) ./ (abs.(J) .+ 1e-3 * scale))
        e = (err(J_aL, J_aL_fd), err(J_aH, J_aH_fd), err(J_cL, J_cL_fd), err(J_cH, J_cH_fd))
        println("  [IFT] Jacobian vs central FD rel errs (aL, aH, cL, cH) = $(e)")
        @test all(x -> x <= 1e-4, e)
        # VJP consistency with the Jacobian
        ā = reshape([1.0, -0.5, 2.0, 0.3, 0.0, 1.0], 6, 1, 1); c̄ = reshape([0.2, 1.0, -1.0, 0.0, 0.5, 2.0], 6, 1, 1)
        pbL, pbH = F.cong_decompose_ift_vjp(p_L, p_H, ā, c̄, plan64; state = st)
        @test pbL ≈ ā .* J_aL .+ c̄ .* J_cL
        @test pbH ≈ ā .* J_aH .+ c̄ .* J_cH
        # scalar loss FD check of the VJP
        loss(pl, ph) = begin
            aa, cc = fd(pl, ph)
            sum(ā .* aa .+ c̄ .* cc)
        end
        for i in 1:6
            eL = zeros(6, 1, 1); eL[i] = h
            fdL = (loss(p_L .+ eL, p_H) - loss(p_L .- eL, p_H)) / (2h)
            fdH = (loss(p_L, p_H .+ eL) - loss(p_L, p_H .- eL)) / (2h)
            @test abs(fdL - pbL[i]) <= 1e-4 * (abs(pbL[i]) + 1e-3 * scale)
            @test abs(fdH - pbH[i]) <= 1e-4 * (abs(pbH[i]) + 1e-3 * scale)
        end
    end

    @testset "cmv_decompose parity" begin
        for bowtie in (nothing, (4, 5))
            basis = material_basis(sp; bowtie = bowtie)
            a_true, c_true = cong_ray_set()
            p_L, p_H = forward_logs(basis, a_true, c_true)
            si = zeros(Float32, size(p_L)); sw = zeros(Float32, size(p_L))
            BS.apply_cmv!(si, sw, p_L, p_H; basis = basis)
            plan = F.cmv_plan(basis; T = Float32)
            ai, aw = F.cmv_decompose(p_L, p_H, plan)
            worst = maximum(max(relerr(Float64(ai[i]), Float64(si[i]); floor = 1e-4), relerr(Float64(aw[i]), Float64(sw[i]); floor = 1e-3)) for i in eachindex(p_L))
            println("  [cmv bowtie=$(bowtie !== nothing)] max rel err vs apply_cmv! = $(worst)")
            @test worst <= 1e-6
            @test all(ai .>= 0) && all(aw .>= 0)
        end
    end

    @testset "synth_vmi_hu parity" begin
        nx, ny, nz = 24, 20, 2
        a_vol = Float32.(0.02 .* rand(nx, ny, nz) .+ 0.01 .* (1:nx) ./ nx)
        c_vol = Float32.(1.0 .+ 0.1 .* rand(nx, ny, nz))
        # legacy synth uses the photo/Compton basis; scale a into its range
        a_vol .*= Float32(1e-3)
        energies = [50.0, 70.0, 100.0]
        leg = BS.synth_vmi_hu(a_vol, c_vol, energies; verbose = false)
        plan = F.vmi_synth_plan(energies, (nx, ny); T = Float32)
        vols = F.synth_vmi_hu(a_vol, c_vol, plan)
        worst = 0.0
        for k in eachindex(energies)
            worst = max(worst, maximum(abs.(vols[k] .- leg.volumes[k]) ./ max.(abs.(leg.volumes[k]), 1f0)))
            @test count(vols[k] .== -1000f0) == count(leg.volumes[k] .== -1000f0)
        end
        println("  [synth] max rel err vs legacy synth_vmi_hu = $(worst)")
        @test worst <= 1e-5
        plan_nomask = F.vmi_synth_plan(energies, (nx, ny); T = Float32, fov_mask_radius_frac = nothing)
        leg_nomask = BS.synth_vmi_hu(a_vol, c_vol, energies; verbose = false, fov_mask_radius_frac = nothing)
        v2 = F.synth_vmi_hu(a_vol, c_vol, 2, plan_nomask)
        @test maximum(abs.(v2 .- leg_nomask.volumes[2]) ./ max.(abs.(leg_nomask.volumes[2]), 1f0)) <= 1e-5
        plan_mat = F.vmi_synth_plan(energies, (nx, ny); T = Float32, basis = :material)
        @test plan_mat.p_E[2] ≈ Float32(μρ_I(70.0)) && plan_mat.q_E[2] ≈ Float32(μρ_W(70.0))
    end

    @testset "synth_vmi_2basis parity (live notebook synth)" begin
        nx, ny, nz = 24, 20, 2
        c_water  = Float32.(0.9 .+ 0.3 .* rand(nx, ny, nz))          # g/mL
        c_iodine = Float32.(20.0 .* rand(nx, ny, nz) .- 2.0)         # mg/mL (iodine g/cm³ · 1000)
        energies = [40.0, 50.0, 70.0, 100.0, 140.0]
        plan = F.vmi_2basis_plan(energies; T = Float32)
        vols = F.synth_vmi_2basis(c_water, c_iodine, plan)
        worst = 0.0
        for (k, E) in enumerate(energies)
            leg = BS.synth_vmi_2basis(c_water, c_iodine; energy_keV = E)
            worst = max(worst, maximum(abs.(vols[k] .- leg) ./ max.(abs.(leg), 1f0)))
            @test vols[k] == leg      # same op order → bit parity
        end
        println("  [synth_2basis] max rel err vs legacy synth_vmi_2basis = $(worst)")
        @test worst <= 1e-6
        @test F.synth_vmi_2basis(c_water, c_iodine, 3, plan) == vols[3]
    end

end

println("test_vmi.jl wall time: $(round(time() - t_start; digits = 1)) s")
