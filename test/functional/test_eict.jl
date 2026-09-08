# Standalone acceptance tests for src/functional/eict.jl (the pure EICT chain).
# Run:  julia --project=. -t 2 test/functional/test_eict.jl
#
# Oracles: BS.dd_fast_fused_poly_project! (spectral conversion),
#          BS.simulate!(::EICTWorkspace) with captured ε buffers (full chain),
#          BS.apply_bhc! (BHC), central finite differences (gradients).
using Test
using BasisSimulator, LinearAlgebra, Statistics, Random
const BS = BasisSimulator

module FStage
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "src", "functional", "eict.jl"))
end
const F = FStage

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function _toy_proj_geom(; n_cols = 16, n_rows = 4, n_angles = 4, fov_cm = 5.0)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = n_rows, detector_cols = n_cols,
        detector_row_size = 1.0, detector_col_size = 1.0,
    )
    return BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = fov_cm, z_cm = 1.0)
end

# Per-material path lengths from the LEGACY mono DD projector on one-hot volumes
# (material ids are 0-based in the mask: dd_fast.jl:151 `mat = mask + 1`).
function onehot_pathlengths(::Type{T}, mask, geom, n_mat; volume_extent = nothing) where {T}
    P = zeros(T, geom.n_cols, geom.n_rows, geom.n_angles, n_mat)
    for m in 1:n_mat
        vol = T.(mask .== (m - 1))
        P[:, :, :, m] .= BS.dd_forward_project(vol, geom; volume_extent = volume_extent)
    end
    return P
end

# Float64 direct (loop) reference of the spectral conversion.
function direct_poly_reference(P::Array{Float64, 4}, μ::Matrix{Float64}, wη::Vector{Float64}, bt)
    n_col, n_row, n_view, n_mat = size(P)
    n_E = length(wη)
    out = zeros(Float64, n_col, n_row, n_view)
    for v in 1:n_view, r in 1:n_row, c in 1:n_col
        I = 0.0
        for e in 1:n_E
            L = 0.0
            for m in 1:n_mat
                L += P[c, r, v, m] * μ[m, e]
            end
            wt = wη[e] * (bt === nothing ? 1.0 : bt[c, r, e])
            I += wt * exp(-L)
        end
        out[c, r, v] = -log(max(I, 1.0e-10))
    end
    return out
end

maxabs(a, b) = maximum(abs.(a .- b))
maxrel(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps(eltype(b))))

const T_START = time()
_ts(msg) = println("[", lpad(round(time() - T_START; digits = 1), 6), " s] ", msg)

# ---------------------------------------------------------------------------
@testset "Functional EICT chain" begin

# ===========================================================================
_ts("spectral conversion vs dd_fast_fused_poly_project!")
@testset "poly_log_sinogram ≡ dd_fast fused poly (64-col, 3-material fixture)" begin
    geom = _toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
    nx = ny = 64; nz = 8; N_E = 8
    mask = zeros(UInt16, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - 32.5)^2 + (j - 32.5)^2 <= (0.6 * 32)^2
            mask[i, j, k] = UInt16(1)
        end
        if (i - 45.5)^2 + (j - 32.5)^2 <= 3.0^2
            mask[i, j, k] = UInt16(2)
        end
    end
    μ_table = zeros(Float32, 3, N_E)
    for e in 1:N_E
        μ_table[2, e] = 0.30f0 - 0.12f0 * (e - 1) / (N_E - 1)
        μ_table[3, e] = 1.50f0 - 0.90f0 * (e - 1) / (N_E - 1)
    end
    wη = fill(Float32(1 / N_E), N_E)

    sino_legacy = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    BS.dd_fast_fused_poly_project!(sino_legacy, mask, geom, μ_table, wη, Val(N_E))
    P32 = onehot_pathlengths(Float32, mask, geom, 3)
    sino_f = F.poly_log_sinogram(P32, μ_table, wη, nothing)
    @test size(sino_f) == size(sino_legacy)
    d = maxabs(sino_f, sino_legacy)
    println("    no-bowtie Float32 max|Δ| vs legacy fused = ", d)
    @test d <= 1.0f-4

    # Float64 direct (loop) reference of the spectral conversion on the same P
    P64 = Float64.(P32)
    μ64 = Float64.(μ_table); wη64 = Float64.(wη)
    ref64 = direct_poly_reference(P64, μ64, wη64, nothing)
    s64 = F.poly_log_sinogram(P64, μ64, wη64, nothing)
    d64 = maxabs(s64, ref64)
    println("    no-bowtie Float64 max|Δ| vs direct loop = ", d64)
    @test d64 <= 1.0e-10

    # chunked variant (static view chunks) — identical up to float ordering
    @test maxabs(F.poly_log_sinogram_chunked(P32, μ_table, wη, nothing, Val(2)), sino_f) <= 1.0f-6
    @test maxabs(F.poly_log_sinogram_chunked(P32, μ_table, wη, nothing, Val(4)), sino_f) <= 1.0f-6
    @test_throws ArgumentError F.poly_log_sinogram_chunked(P32, μ_table, wη, nothing, Val(3))

    @testset "helical arc geometry + spectral bowtie" begin
        scanner_h = BS.EICTScanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 4, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
            detector_shape = :arc,
        )
        geom_h = BS.CTGeometry(scanner_h; n_angles = 16, fov_cm = 10.0,
            z_cm = 1.0, pitch = 1.0, n_rotations = 2.0)
        mask_h = mask[17:48, 17:48, 3:6]
        bowtie = Array{Float32}(undef, geom_h.n_cols, geom_h.n_rows, N_E)
        for e in 1:N_E, r in 1:geom_h.n_rows, c in 1:geom_h.n_cols
            bowtie[c, r, e] = 0.7f0 + 0.25f0 * (c - 1) / (geom_h.n_cols - 1) +
                0.03f0 * (e - 1) / (N_E - 1)
        end
        ext = (10.0, 10.0, 1.0)
        sino_l = zeros(Float32, geom_h.n_cols, geom_h.n_rows, geom_h.n_angles)
        BS.dd_fast_fused_poly_project!(sino_l, mask_h, geom_h, μ_table, wη, Val(N_E);
            volume_extent = ext, ws_bowtie_spectral = bowtie)
        Ph = onehot_pathlengths(Float32, mask_h, geom_h, 3; volume_extent = ext)
        sino_fb = F.poly_log_sinogram(Ph, μ_table, wη, bowtie)
        db = maxabs(sino_fb, sino_l)
        println("    bowtie Float32 max|Δ| vs legacy fused = ", db)
        @test db <= 1.0f-4
        # chunked + bowtie
        @test maxabs(F.poly_log_sinogram_chunked(Ph, μ_table, wη, bowtie, Val(4)), sino_fb) <= 1.0f-6
        # Float64 direct reference with bowtie
        Ph64 = Float64.(Ph)
        bt64 = Float64.(bowtie)
        ref_b = direct_poly_reference(Ph64, μ64, wη64, bt64)
        db64 = maxabs(F.poly_log_sinogram(Ph64, μ64, wη64, bt64), ref_b)
        println("    bowtie Float64 max|Δ| vs direct loop = ", db64)
        @test db64 <= 1.0e-10
    end
end

# ===========================================================================
_ts("BHC parity vs apply_bhc!")
@testset "bhc_apply ≡ apply_bhc! (per-column static-order polynomial)" begin
    n_col, n_row, n_view = 16, 4, 8
    energies = collect(30.0:10.0:100.0)
    n_E = length(energies)
    w_col = zeros(Float64, n_E, n_col)
    for c in 1:n_col, e in 1:n_E
        w_col[e, c] = exp(-((energies[e] - 60.0 - 0.3 * c) / 25.0)^2)
    end
    for order in (3, 5)
        wb = BS.calibrate_bhc_water(energies, w_col; order = order, reference_energy_keV = 70.0)
        polys = [b.polynomial for b in wb.water_bhc_per_col]
        for T in (Float32, Float64)
            rng = MersenneTwister(7)
            sino = T.(6 .* rand(rng, n_col, n_row, n_view))
            ref = copy(sino)
            BS.apply_bhc!(ref, polys)
            coeffs = F.bhc_coeff_matrix(T, wb)
            @test size(coeffs) == (order + 1, n_col)
            @test coeffs == F.bhc_coeff_matrix(T, polys) == F.bhc_coeff_matrix(T, wb.water_bhc_per_col)
            out = F.bhc_apply(sino, coeffs)
            r = maxrel(out, ref)
            println("    order $order $T max rel Δ vs apply_bhc! = ", r)
            @test r <= 1.0e-6
            @test sino == T.(6 .* rand(MersenneTwister(7), n_col, n_row, n_view))   # input untouched
        end
    end
    x = rand(Float32, 4, 2, 3)
    @test F.bhc_apply(x, nothing) === x
end

# ===========================================================================
# Full chain vs legacy simulate!(::EICTWorkspace) on the toy Gammex setup
# (test/api.jl:582-613 `_toy_eict_setup`, CPU arrays).
# ===========================================================================
_ts("building toy EICT workspaces (scatter off / on)")
function toy_setup(; use_scatter, use_noise, seed = 42)
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 64,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_material = :lumex, detector_depth = 3.0,
        electronic_noise = 5.0, detection_gain = 10.0,
    )
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    sim_opts = BS.SimOptions(; use_noise = use_noise, use_scatter = use_scatter,
        use_lag = false, use_focal_spot = false, use_optical_crosstalk = false, seed = seed)
    recon_opts = BS.ReconOptions(matrix_size = (32, 32, 4), fov_cm = 20.0)
    phantom = BS.create_gammex_472(n_voxels = 32, fov_cm = 20.0, z_cm = 2.0)
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
    return (; scanner, protocol, sim_opts, recon_opts, phantom, ws)
end

s_off = toy_setup(use_scatter = false, use_noise = false)
s_on = toy_setup(use_scatter = true, use_noise = true)
_ts("workspaces built; computing one-hot path lengths")
P_toy = onehot_pathlengths(Float32, s_off.phantom.mask, s_off.ws.geom, length(s_off.phantom.materials);
    volume_extent = s_off.phantom.extent)
_ts("path lengths done")

@testset "eict_chain ≡ simulate!(EICTWorkspace)" begin
    @test s_off.ws.config.fill_factor !== nothing            # fill-factor path exercised
    @test s_off.ws.bowtie_spectral !== nothing               # bowtie × heel exercised
    @test s_off.ws.σ_e_photon > 0                            # electronic-noise path exercised

    @testset "noise off, scatter off (≤1e-5)" begin
        opts = s_off.sim_opts
        BS.simulate!(s_off.ws, s_off.phantom, s_off.protocol, opts)
        ref = copy(s_off.ws.sinogram)
        plan = F.eict_plan(s_off.ws, s_off.protocol, opts)
        @test plan.use_noise == false
        @test plan.scatter_Hc === nothing
        @test plan.bhc_coeffs === nothing
        out = F.eict_chain(P_toy, plan)
        d = maxabs(out, ref)
        println("    noise off/scatter off: max|Δ| = ", d, "  (sino range ", extrema(ref), ")")
        @test d <= 1.0e-5
        @test all(isfinite, out)
    end

    @testset "noise on (quantum + electronic), scatter off (≤1e-4)" begin
        opts = BS.SimOptions(; use_noise = true, use_scatter = false,
            use_lag = false, use_focal_spot = false, use_optical_crosstalk = false, seed = 42)
        BS.simulate!(s_off.ws, s_off.phantom, s_off.protocol, opts)
        ref = copy(s_off.ws.sinogram)
        ε = copy(s_off.ws.noise_rand_cpu)        # exact draws used by the legacy kernel
        ε_e = copy(s_off.ws.enoise_rand_cpu)
        plan = F.eict_plan(s_off.ws, s_off.protocol, opts)
        @test plan.use_noise && plan.use_enoise
        out = F.eict_chain(P_toy, plan, ε, ε_e)
        d = maxabs(out, ref)
        println("    noise on/scatter off: max|Δ| = ", d)
        @test d <= 1.0e-4
        # the noise actually did something, and a different ε changes the output
        @test maxabs(out, F.eict_chain(P_toy, F.eict_plan(s_off.ws, s_off.protocol, s_off.sim_opts))) > 1.0e-3
        @test_throws ArgumentError F.eict_chain(P_toy, plan)            # ε required
        @test_throws ArgumentError F.eict_chain(P_toy, plan, ε)         # ε_e required (σ_e > 0)
        # ε given as a 3-D tensor is accepted too
        @test out == F.eict_chain(P_toy, plan, reshape(ε, size(ref)), reshape(ε_e, size(ref)))
    end

    @testset "noise on, scatter on (≤1e-4)" begin
        opts = s_on.sim_opts
        BS.simulate!(s_on.ws, s_on.phantom, s_on.protocol, opts)
        ref = copy(s_on.ws.sinogram)
        ε = copy(s_on.ws.noise_rand_cpu)
        ε_e = copy(s_on.ws.enoise_rand_cpu)
        plan = F.eict_plan(s_on.ws, s_on.protocol, opts)
        @test plan.scatter_Hc !== nothing && plan.scatter_sw > 0
        out = F.eict_chain(P_toy, plan, ε, ε_e)
        d = maxabs(out, ref)
        println("    noise on/scatter on: max|Δ| = ", d)
        @test d <= 1.0e-4
    end

    @testset "noise off, scatter on (noise-free scatter-subtraction branch)" begin
        opts = BS.SimOptions(; use_noise = false, use_scatter = true,
            use_lag = false, use_focal_spot = false, use_optical_crosstalk = false, seed = 42)
        BS.simulate!(s_on.ws, s_on.phantom, s_on.protocol, opts)
        ref = copy(s_on.ws.sinogram)
        plan = F.eict_plan(s_on.ws, s_on.protocol, opts)
        @test !plan.use_noise && plan.scatter_Hc !== nothing
        out = F.eict_chain(P_toy, plan)
        d = maxabs(out, ref)
        println("    noise off/scatter on: max|Δ| = ", d)
        @test d <= 1.0e-4
    end

    @testset "BHC on top of the chain ≡ apply_bhc_water(simulate! output)" begin
        opts = s_off.sim_opts
        BS.simulate!(s_off.ws, s_off.phantom, s_off.protocol, opts)
        # per-column detected spectrum of the central row (wη × bowtie×heel), low-level knobless BHC
        plan0 = F.eict_plan(s_off.ws, s_off.protocol, opts)
        w_col = Float64.(permutedims(plan0.bt[:, 4, :], (2, 1)) .* plan0.wη)   # (n_E, n_col)
        wb = BS.calibrate_bhc_water(s_off.ws.energies, w_col; order = 5, reference_energy_keV = 70.0)
        ref = BS.apply_bhc_water(s_off.ws.sinogram, wb)
        plan = F.eict_plan(s_off.ws, s_off.protocol, opts; bhc = wb)
        @test size(plan.bhc_coeffs) == (6, 64)
        out = F.eict_chain(P_toy, plan)
        d = maxabs(out, ref)
        println("    chain+BHC vs simulate!+apply_bhc_water: max|Δ| = ", d)
        @test d <= 2.0e-4
    end

    @testset "plan refuses effects the chain does not reproduce (lag)" begin
        cfg = s_off.ws.config
        cfg_lag = BS.PhysicsConfig(cfg.fill_factor, cfg.scatter, cfg.optical_crosstalk, cfg.focal_spot,
            cfg.detector_efficiency, BS.lag_gadox(), cfg.noise_seed, cfg.energy_keV, cfg.heel_effect)
        s_off.ws.config = cfg_lag
        try
            @test_throws ArgumentError F.eict_plan(s_off.ws, s_off.protocol, s_off.sim_opts)
        finally
            s_off.ws.config = cfg
        end
    end
end

# ===========================================================================
# Gradient sanity (Float64, plain Arrays, tiny problem, central differences)
# ===========================================================================
_ts("gradient checks")
@testset "hand-derived VJPs vs finite differences (Float64)" begin
    rng = MersenneTwister(11)
    n_col, n_row, n_view, n_mat, n_E = 6, 2, 3, 3, 5
    P = 4.0 .* rand(rng, n_col, n_row, n_view, n_mat)
    μ_tbl = [0.0 0.0 0.0 0.0 0.0; 0.35 0.30 0.25 0.22 0.20; 1.2 0.9 0.7 0.55 0.45]
    wη = [0.1, 0.25, 0.3, 0.25, 0.1]
    bt = 0.6 .+ 0.4 .* rand(rng, n_col, n_row, n_E)
    air_ref = vec(sum(reshape(wη, 1, 1, :) .* bt; dims = 3))
    air_ref = reshape(air_ref, n_col, n_row)
    w = randn(rng, n_col, n_row, n_view)

    function fd_grad(f, x; h = 1.0e-6)
        g = zero(x)
        for i in eachindex(x)
            xp = copy(x); xm = copy(x)
            xp[i] += h; xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2h)
        end
        return g
    end

    @testset "poly_log_sinogram_vjp (no bowtie / bowtie)" begin
        for b in (nothing, bt)
            f = X -> sum(w .* F.poly_log_sinogram(X, μ_tbl, wη, b))
            g_fd = fd_grad(f, P)
            g_an = F.poly_log_sinogram_vjp(P, μ_tbl, wη, b, w)
            r = maxabs(g_an, g_fd) / maximum(abs.(g_fd))
            println("    poly_log_sinogram_vjp (bt=", b === nothing ? "nothing" : "array", ") rel err = ", r)
            @test r <= 1.0e-6
        end
    end

    # Full chain configurations
    k1d = [0.05, 0.2, 0.5, 0.2, 0.05]
    Hc = F.clamped_conv_matrix(k1d, n_col); Hr = F.clamped_conv_matrix(k1d, n_row)
    coeffs = [0.01 .+ 0.001 .* (1:n_col)'; 1.02 .+ 0.001 .* (1:n_col)'; 0.003 .* ones(1, n_col); -0.0004 .* ones(1, n_col)]
    ε = randn(rng, n_col, n_row, n_view)
    ε_e = randn(rng, n_col, n_row, n_view)
    shape = (n_col, n_row, n_view)
    # Three configurations cover every VJP branch: the superset (bowtie, air ref,
    # fill factor, quantum+electronic noise, scatter, BHC), the noise-free scatter
    # subtraction branch, and the minimal no-bowtie / quantum-only chain.
    base = (; μ_tbl, wη, sino_shape = shape, I0 = 2.0e6, bt, air_ref, σ_e = 3.0, ff_log = log(0.81))
    configs = [
        ("noise on (q+e), scatter on, BHC, bowtie/air/ff", F.EICTPlan(; base..., use_noise = true, scatter_Hc = Hc, scatter_Hr = Hr, scatter_C = 0.02, scatter_sw = 0.9, bhc_coeffs = coeffs)),
        ("noise off, scatter on, BHC",                      F.EICTPlan(; base..., use_noise = false, scatter_Hc = Hc, scatter_Hr = Hr, scatter_C = 0.02, scatter_sw = 0.9, bhc_coeffs = coeffs)),
        ("no bowtie/air/ff, quantum noise only, no BHC",    F.EICTPlan(; μ_tbl, wη, sino_shape = shape, I0 = 2.0e6, σ_e = 0.0, use_noise = true)),
    ]
    for (name, plan) in configs
        f = X -> sum(w .* F.eict_chain(X, plan, ε, ε_e))
        g_fd = fd_grad(f, P)
        g_an = F.eict_chain_vjp(P, plan, ε, ε_e, w)
        r = maxabs(g_an, g_fd) / maximum(abs.(g_fd))
        println("    eict_chain_vjp [", name, "] rel err = ", r)
        @test r <= 1.0e-3
        @test all(isfinite, g_an)
    end

    @testset "purity: inputs never mutated" begin
        plan = configs[1][2]
        P0 = copy(P); ε0 = copy(ε); ε_e0 = copy(ε_e)
        F.eict_chain(P, plan, ε, ε_e)
        F.eict_chain_vjp(P, plan, ε, ε_e, w)
        @test P == P0 && ε == ε0 && ε_e == ε_e0
    end
end

end # top-level testset
_ts("done")
