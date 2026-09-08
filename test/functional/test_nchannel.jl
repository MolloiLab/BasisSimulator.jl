# =============================================================================
# Functional n-channel VMI estimator — parity against the VERBATIM notebook code
#
# The oracle is the notebook itself: the `nchannel_*` cells of
# docs/notebooks/04_pcct_vmi.jl (global tables, K = 4) and
# docs/notebooks/03_dual_kvp_switching_vmi.jl (per-ray tables, K = 2) are
# extracted by cell UUID and evaluated with `include_string` in scratch
# modules; nb12's kernel is checked for textual drift against nb03's.
#
# Run standalone:  julia --project=. -t 2 test/functional/test_nchannel.jl
# =============================================================================

using Test, Statistics, LinearAlgebra, Random
using BasisSimulator
const BS = BasisSimulator

module FStageNChannel
    using BasisSimulator, LinearAlgebra, Statistics
    const BS = BasisSimulator
    include(joinpath(@__DIR__, "..", "..", "src", "functional", "nchannel.jl"))
end
const NCH = FStageNChannel

const _NCH_T0 = time()

# ── metrics ───────────────────────────────────────────────────────────────────
nch_rel_max(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))
const NCH_PARITY = Dict{String, Float64}()
nch_record!(k, v) = (NCH_PARITY[k] = Float64(v); v)

# ── notebook cell extraction ──────────────────────────────────────────────────
const NB_DIR = joinpath(@__DIR__, "..", "..", "docs", "notebooks")
const NB03 = joinpath(NB_DIR, "03_dual_kvp_switching_vmi.jl")
const NB04 = joinpath(NB_DIR, "04_pcct_vmi.jl")
const NB12 = joinpath(NB_DIR, "12_siemens_flash_ufc.jl")

const UUID_CONTROLS = "4985581f-616d-4bb7-ab9b-967d7250b28b"   # nb03 + nb04
const UUID_KERNEL = "73371177-0498-4eda-897b-651c94f43e83"     # nb03 + nb04 (helpers + tile kernel)
const UUID_BASIS = "4ca28c64-ee96-47c8-b7c3-f0e0c4c99423"      # nb03 + nb04 (different bodies)
const UUID_TLBF = "f3de45a6-4818-4ee1-ad56-c65797119dee"       # nb04 T-LBF cell
const UUID12_CONTROLS = "1200000a-0000-4000-8000-000000000005"
const UUID12_KERNEL = "1200000a-0000-4000-8000-000000000008"
const UUID12_BASIS = "1200000a-0000-4000-8000-000000000015"

function notebook_cell(path, uuid)
    text = read(path, String)
    marker = "# ╔═╡ $uuid"
    i = findfirst(marker, text)
    i === nothing && error("cell $uuid not found in $path")
    start = last(i) + 1
    nxt = findnext("# ╔═╡ ", text, start)
    return nxt === nothing ? text[start:end] : text[start:(first(nxt) - 1)]
end

# Oracle modules: the notebook cells run here unchanged (`to_gpu` = identity;
# `BS.AK.foreachindex` runs on plain Arrays on the CPU).
module NB04Oracle
    using BasisSimulator, Statistics, LinearAlgebra
    const BS = BasisSimulator
    to_gpu(x) = x
end
module NB03Oracle
    using BasisSimulator, Statistics, LinearAlgebra
    const BS = BasisSimulator
    to_gpu(x) = x
end
Base.include_string(NB04Oracle, notebook_cell(NB04, UUID_CONTROLS), "nb04_controls.jl")
Base.include_string(NB04Oracle, notebook_cell(NB04, UUID_KERNEL), "nb04_kernel.jl")
Base.include_string(NB04Oracle, notebook_cell(NB04, UUID_TLBF), "nb04_tlbf.jl")
Base.include_string(NB03Oracle, notebook_cell(NB03, UUID_CONTROLS), "nb03_controls.jl")
Base.include_string(NB03Oracle, notebook_cell(NB03, UUID_KERNEL), "nb03_kernel.jl")

# ── fixtures ──────────────────────────────────────────────────────────────────
# Real xspect tungsten spectra (0.5 keV) re-binned onto a 2-keV grid, 20…140 keV.
const E_GRID = collect(20.0:2.0:140.0)
function rebin_spectrum(kVp)
    e, w = BS.load_spectrum(kVp)
    out = zeros(Float64, length(E_GRID))
    for (ei, wi) in zip(e, w)
        j = round(Int, (ei - E_GRID[1]) / 2) + 1
        1 <= j <= length(out) && (out[j] += wi)
    end
    return out ./ sum(out)
end

# nb04-style global tables, K = 4 PCCT bins: 120 kVp × soft energy windows.
function pcct4_tables(; I0_total = 4.0e4)
    w = rebin_spectrum(120)
    edges = (45.0, 60.0, 75.0)
    soft(x) = 1 / (1 + exp(-x / 2.0))
    R = hcat(
        [1 - soft(e - edges[1]) for e in E_GRID],
        [soft(e - edges[1]) * (1 - soft(e - edges[2])) for e in E_GRID],
        [soft(e - edges[2]) * (1 - soft(e - edges[3])) for e in E_GRID],
        [soft(e - edges[3]) for e in E_GRID],
    )
    Φ = I0_total .* w .* R                      # (nE, 4) absolute counts per energy
    return (energies = E_GRID, W_applied = Φ, I0_bins = vec(sum(Φ; dims = 1)))
end

# nb03-style per-ray tables, K = 2 (80 / 140 kVp) with a column-dependent
# water-equivalent bowtie so every (col, row) has its own response.
function dual2_tables(n_col, n_row; I0_total = 1.0e5)
    μw = [BS.compute_mass_μ_at_energy(BS.XA.Materials.water, e) for e in E_GRID]
    t = [3.0 * ((c - (n_col + 1) / 2) / (n_col / 2))^2 + 0.1 * (r - 1) for c in 1:n_col, r in 1:n_row]
    channels = map((80, 140)) do kVp
        w = rebin_spectrum(kVp)
        Φ = zeros(Float32, n_col, n_row, length(E_GRID))
        for c in 1:n_col, r in 1:n_row
            Φ[c, r, :] .= Float32.(I0_total .* w .* exp.(-μw .* t[c, r]))
        end
        (Φ = Φ, I0 = dropdims(sum(Φ; dims = 3); dims = 3), energies = Float32.(E_GRID))
    end
    return (Φ = [c.Φ for c in channels], I0 = [c.I0 for c in channels],
        energies = [c.energies for c in channels])
end

# Ground-truth (A, C) grid over a (n_col, n_row, n_view) tile.
function truth_grid(n_col, n_row, n_view)
    n = n_col * n_row * n_view
    As = range(-0.05, 0.35; length = 8)
    Cs = range(0.0, 40.0; length = 8)
    A = zeros(Float64, n); C = zeros(Float64, n)
    for i in 1:n
        A[i] = As[(i - 1) % 8 + 1]
        C[i] = Cs[((i - 1) ÷ 8) % 8 + 1]
    end
    return reshape(A, n_col, n_row, n_view), reshape(C, n_col, n_row, n_view)
end

# Channel log-transmissions from the notebook's own Float64 forward model.
function synth_h(forward, A, C, Φ_of_ray, μI, μW, I0_of_ray; rng = nothing, noise_scale = 1.0)
    rs = size(A)
    K = length(I0_of_ray(1, 1))
    h = zeros(Float32, rs..., K)
    for v in 1:rs[3], r in 1:rs[2], c in 1:rs[1]
        λ = forward(A[c, r, v], C[c, r, v], Φ_of_ray(c, r), μI, μW).λ
        y = copy(λ)
        if rng !== nothing
            y .= max.(λ .+ noise_scale .* sqrt.(λ) .* randn(rng, length(λ)), 1.0)
        end
        h[c, r, v, :] .= Float32.(-log.(y ./ I0_of_ray(c, r)))
    end
    return h
end

# Verbatim kernel driver (the notebook's per-tile `nchannel_profile_tile!` call on Arrays).
function oracle_tile(mod, h::Array{Float32, 4}, tables, controls)
    shape = size(h)[1:3]
    hs = Tuple(h[:, :, :, k] for k in 1:size(h, 4))
    sino_I = Array{Float32}(undef, shape); sino_W = similar(sino_I)
    fAA = similar(sino_I); fAC = similar(sino_I); fCC = similar(sino_I)
    flags = Array{UInt8}(undef, shape); score = similar(sino_I)
    outer = Array{UInt8}(undef, shape); inner = Array{UInt8}(undef, shape)
    mod.nchannel_profile_tile!(sino_I, sino_W, fAA, fAC, fCC, flags, score, outer, inner, hs,
        tables...,
        controls)
    return (sino_iodine = sino_I, sino_water = sino_W, quality_flag = flags,
        fisher = (AA = fAA, AC = fAC, CC = fCC), score_norm = score,
        outer_iterations = outer, inner_iterations = inner)
end

function compare_estimates(label, mine, ref)
    rI = nch_record!("$label/sino_iodine", nch_rel_max(mine.sino_iodine, ref.sino_iodine))
    rW = nch_record!("$label/sino_water", nch_rel_max(mine.sino_water, ref.sino_water))
    rAA = nch_record!("$label/fisher_AA", nch_rel_max(mine.fisher.AA, ref.fisher.AA))
    rAC = nch_record!("$label/fisher_AC", nch_rel_max(mine.fisher.AC, ref.fisher.AC))
    rCC = nch_record!("$label/fisher_CC", nch_rel_max(mine.fisher.CC, ref.fisher.CC))
    rS = nch_record!("$label/score_norm", maximum(abs.(mine.score_norm .- ref.score_norm)))
    flag_mismatch = count(NCH.nchannel_flags_u8(mine.quality_flag) .!= ref.quality_flag)
    outer_mismatch = count(NCH.nchannel_counts_u8(mine.outer_iterations) .!= ref.outer_iterations)
    inner_mismatch = count(NCH.nchannel_counts_u8(mine.inner_iterations) .!= ref.inner_iterations)
    nch_record!("$label/flag_mismatch", flag_mismatch)
    nch_record!("$label/outer_count_mismatch", outer_mismatch)
    nch_record!("$label/inner_count_mismatch", inner_mismatch)
    nch_record!("$label/max_outer_used", maximum(ref.outer_iterations))
    nch_record!("$label/max_inner_used", maximum(ref.inner_iterations))
    @test rI <= 1.0e-5
    @test rW <= 1.0e-5
    @test rAA <= 1.0e-4
    @test rAC <= 1.0e-4
    @test rCC <= 1.0e-4
    @test rS <= 1.0e-3            # score_norm is a near-zero residual: absolute
    @test flag_mismatch == 0
    @test outer_mismatch <= 1
    @test inner_mismatch <= 2
    return (rI, rW)
end

# =============================================================================
@testset "n-channel estimator — verbatim notebook parity" begin

    # ── 1. Mirror drift ──────────────────────────────────────────────────────
    @testset "mirror drift nb03 / nb04 / nb12" begin
        norm_ws(s) = join(filter(!isempty, strip.(split(s, '\n'))), '\n')
        kernel_of(s) = s[first(findfirst("function nchannel_profile_tile!", s)):end]
        k03 = norm_ws(kernel_of(notebook_cell(NB03, UUID_KERNEL)))
        k04 = norm_ws(kernel_of(notebook_cell(NB04, UUID_KERNEL)))
        k12 = norm_ws(kernel_of(notebook_cell(NB12, UUID12_KERNEL)))
        # nb12 kernel ≡ nb03 kernel (module-level function vs begin-block: nb03's
        # cell carries one extra trailing `end`; otherwise whitespace only)
        strip_end(s) = replace(s, r"\nend\s*$" => "")
        @test k12 == strip_end(k03)
        # nb04 kernel ≡ nb03 kernel after the per-ray → global table substitutions
        g = k03
        g = replace(g, "# Ray-dependent dual-kVp responses use detector-column initializer terms.\n" => "")
        g = replace(g, "ncol=size(sino_I,1)\nnrow=size(sino_I,2)\ncol=mod1(idx,ncol)\nrow=mod1(cld(idx,ncol),nrow)\n" => "")
        g = replace(g, "nII=normal_II[col,row]\nnIW=normal_IW[col,row]\nnWW=normal_WW[col,row]\n" => "")
        g = replace(g, "nII" => "normal_II", "nIW" => "normal_IW", "nWW" => "normal_WW")
        g = replace(g, "[col,row,e,k]" => "[e,k]", "[col,row,k]" => "[k]")
        g = replace(g, "normal_II, normal_IW, normal_WW, controls," =>
            "normal_II::Float32, normal_IW::Float32, normal_WW::Float32, controls,")
        @test g == k04
        # controls identical in the three notebooks (comments stripped)
        strip_comments(s) = replace(norm_ws(replace(s, r"#[^\n]*" => "")), " " => "")
        @test strip_comments(notebook_cell(NB03, UUID_CONTROLS)) == strip_comments(notebook_cell(NB04, UUID_CONTROLS))
        @test strip_comments(notebook_cell(NB12, UUID12_CONTROLS)) == strip_comments(notebook_cell(NB03, UUID_CONTROLS))
        # basis builders nb03 ≡ nb12 modulo whitespace
        b03 = norm_ws(notebook_cell(NB03, UUID_BASIS)); b12 = norm_ws(notebook_cell(NB12, UUID12_BASIS))
        @test replace(b03, " " => "") == replace(b12, " " => "")
        # reference solvers nb03 ≡ nb04 byte-for-byte
        head_of(s) = s[1:(first(findfirst("function nchannel_profile_tile!", s)) - 1)]
        @test head_of(notebook_cell(NB03, UUID_KERNEL)) == head_of(notebook_cell(NB04, UUID_KERNEL))
    end

    # ── 2. nb04 global tables, K = 4 ─────────────────────────────────────────
    sim4 = pcct4_tables()
    Core.eval(NB04Oracle, :(sim_bins = $sim4))
    Base.include_string(NB04Oracle, notebook_cell(NB04, UUID_BASIS), "nb04_basis.jl")
    basis4 = NB04Oracle.nchannel_basis
    ctrl = NB04Oracle.nchannel_controls
    plan4 = NCH.nchannel_plan(sim4.W_applied, sim4.energies, sim4.I0_bins; T = Float32)

    @testset "plan tables reproduce nb04 nchannel_basis" begin
        @test plan4.K == 4 && plan4.nE == length(E_GRID) && !plan4.per_ray
        @test vec(plan4.μI_eff) == basis4.μI_eff
        @test vec(plan4.μW_eff) == basis4.μW_eff
        @test plan4.normal_II[1] == basis4.normal_II
        @test plan4.normal_IW[1] == basis4.normal_IW
        @test plan4.normal_WW[1] == basis4.normal_WW
        @test plan4.μρ_I == basis4.μρ_I && plan4.μρ_W == basis4.μρ_W
        @test plan4.moment_table[:, 1:4] == basis4.Φ
        @test plan4.I0_relerr == basis4.I0_relerr
    end

    @testset "forward model vs nchannel_forward" begin
        A = reshape(Float32[0.1, 0.3, -0.05, 0.0], 2, 1, 2)
        C = reshape(Float32[10.0, 30.0, 0.0, 45.0], 2, 1, 2)
        λ, dA, dC = NCH.nchannel_moments(A, C, plan4)
        for (c, v) in ((1, 1), (2, 1), (1, 2), (2, 2))
            ref = NB04Oracle.nchannel_forward(Float64(A[c, 1, v]), Float64(C[c, 1, v]), basis4.Φ, basis4.μρ_I, basis4.μρ_W)
            @test isapprox(λ[c, 1, v, :], ref.λ; rtol = 2.0e-6)
            @test isapprox(dA[c, 1, v, :], ref.dA; rtol = 2.0e-6)
            @test isapprox(dC[c, 1, v, :], ref.dC; rtol = 2.0e-6)
            @test isapprox(NCH.nchannel_total_counts(A, C, plan4)[c, 1, v], sum(ref.λ); rtol = 2.0e-6)
        end
    end

    tables4 = (basis4.Φ, basis4.μρ_I, basis4.μρ_W, basis4.I0, basis4.μI_eff, basis4.μW_eff,
        basis4.normal_II, basis4.normal_IW, basis4.normal_WW)
    A4, C4 = truth_grid(8, 1, 8)
    Φ4_of(c, r) = basis4.Φ
    I04_of(c, r) = basis4.I0
    h4_clean = synth_h(NB04Oracle.nchannel_forward, A4, C4, Φ4_of, basis4.μρ_I, basis4.μρ_W, I04_of)
    h4_noisy = synth_h(NB04Oracle.nchannel_forward, A4, C4, Φ4_of, basis4.μρ_I, basis4.μρ_W, I04_of;
        rng = MersenneTwister(0x4c4), noise_scale = 1.0)

    @testset "K = 4 global tables: noise-free tile" begin
        ref = oracle_tile(NB04Oracle, h4_clean, tables4, ctrl)
        mine = NCH.nchannel_estimate_tile(h4_clean, plan4)
        compare_estimates("pcct4_clean", mine, ref)
        # ground truth recovery (both): the estimator is exact on noise-free rays
        @test maximum(abs.(ref.sino_iodine .- Float32.(A4))) < 2.0e-3
        @test maximum(abs.(ref.sino_water .- Float32.(C4))) < 2.0e-2
        @test count(==(0x00), ref.quality_flag) >= 56
    end

    @testset "K = 4 global tables: noisy tile" begin
        ref = oracle_tile(NB04Oracle, h4_noisy, tables4, ctrl)
        mine = NCH.nchannel_estimate_tile(h4_noisy, plan4)
        compare_estimates("pcct4_noisy", mine, ref)
        # tiled driver == single tile
        tiled = NCH.nchannel_estimate(h4_noisy, plan4; tile_views = 3)
        @test tiled.sino_iodine == mine.sino_iodine && tiled.sino_water == mine.sino_water
        @test tiled.quality_flag == mine.quality_flag
    end

    @testset "row combination (nb04 nchannel_slab_counts) + scale" begin
        h_rows = cat(h4_noisy, h4_noisy .* 1.02f0, h4_noisy .* 0.98f0; dims = 2)  # 3 rows
        hΣ = NCH.nchannel_combine_rows(h_rows, 1:3)
        ref = -log.(max.(dropdims(sum(exp.(-h_rows); dims = 2); dims = 2), 1.0f-12) ./ 3.0f0)
        @test size(hΣ) == (8, 1, 8, 4)
        @test isapprox(hΣ[:, 1, :, :], ref; rtol = 1.0e-6)
        plan4s = NCH.nchannel_plan(sim4.W_applied, sim4.energies, sim4.I0_bins; T = Float32, scale = 3)
        tables4s = (3.0f0 .* basis4.Φ, basis4.μρ_I, basis4.μρ_W, 3.0f0 .* basis4.I0, basis4.μI_eff,
            basis4.μW_eff, basis4.normal_II, basis4.normal_IW, basis4.normal_WW)
        refs = oracle_tile(NB04Oracle, hΣ, tables4s, ctrl)
        mines = NCH.nchannel_estimate_tile(hΣ, plan4s)
        compare_estimates("pcct4_slab3", mines, refs)
    end

    # ── 3. nb03 per-ray tables, K = 2 ────────────────────────────────────────
    n_col, n_row, n_view = 8, 2, 4
    slab2 = dual2_tables(n_col, n_row)
    Core.eval(NB03Oracle, :(nchannel_slab_counts = $slab2))
    Base.include_string(NB03Oracle, notebook_cell(NB03, UUID_BASIS), "nb03_basis.jl")
    basis2 = NB03Oracle.nchannel_basis
    ctrl3 = NB03Oracle.nchannel_controls
    E2, Φ2 = NCH.nchannel_merge_channels(slab2.energies, slab2.Φ)
    I02 = cat(slab2.I0...; dims = 3)
    plan2 = NCH.nchannel_plan(Φ2, E2, I02; T = Float32)

    @testset "plan tables reproduce nb03 build_nchannel_basis" begin
        @test plan2.per_ray && plan2.K == 2
        @test E2 == basis2.E
        @test Φ2 == basis2.Φ
        @test plan2.moment_table[:, :, :, 1:2] == basis2.Φ
        @test dropdims(plan2.μI_eff; dims = 3) == basis2.μI_eff
        @test dropdims(plan2.μW_eff; dims = 3) == basis2.μW_eff
        @test dropdims(plan2.normal_II; dims = 3) == basis2.normal_II
        @test dropdims(plan2.normal_IW; dims = 3) == basis2.normal_IW
        @test dropdims(plan2.normal_WW; dims = 3) == basis2.normal_WW
        @test plan2.μρ_I == basis2.μρ_I && plan2.μρ_W == basis2.μρ_W
    end

    tables2 = (basis2.Φ, basis2.μρ_I, basis2.μρ_W, basis2.I0, basis2.μI_eff, basis2.μW_eff,
        basis2.normal_II, basis2.normal_IW, basis2.normal_WW)
    A2, C2 = truth_grid(n_col, n_row, n_view)
    Φ2_of(c, r) = basis2.Φ[c, r, :, :]
    I02_of(c, r) = basis2.I0[c, r, :]
    h2_clean = synth_h(NB03Oracle.nchannel_forward, A2, C2, Φ2_of, basis2.μρ_I, basis2.μρ_W, I02_of)
    h2_noisy = synth_h(NB03Oracle.nchannel_forward, A2, C2, Φ2_of, basis2.μρ_I, basis2.μρ_W, I02_of;
        rng = MersenneTwister(0x303), noise_scale = 1.0)

    @testset "K = 2 per-ray tables: noise-free tile" begin
        ref = oracle_tile(NB03Oracle, h2_clean, tables2, ctrl3)
        mine = NCH.nchannel_estimate_tile(h2_clean, plan2)
        compare_estimates("dual2_clean", mine, ref)
        @test maximum(abs.(ref.sino_iodine .- Float32.(A2))) < 2.0e-3
        @test maximum(abs.(ref.sino_water .- Float32.(C2))) < 2.0e-2
    end

    @testset "K = 2 per-ray tables: noisy tile" begin
        ref = oracle_tile(NB03Oracle, h2_noisy, tables2, ctrl3)
        mine = NCH.nchannel_estimate_tile(h2_noisy, plan2)
        compare_estimates("dual2_noisy", mine, ref)
    end

    # ── 4. Float64 smoothness in the channel sinograms ───────────────────────
    @testset "Float64 finite-difference smoothness (well-conditioned rays)" begin
        plan64 = NCH.nchannel_plan(sim4.W_applied, sim4.energies, sim4.I0_bins;
            T = Float64, parameter_tolerance = 1.0e-11)
        worst_richardson = 0.0
        worst_ift = 0.0
        n_checked = 0
        for (A_true, C_true) in ((0.10, 10.0), (0.20, 25.0), (0.05, 30.0), (0.30, 5.0), (0.0, 20.0), (0.15, 15.0))
            λ = NB04Oracle.nchannel_forward(A_true, C_true, basis4.Φ, basis4.μρ_I, basis4.μρ_W).λ
            h0 = reshape(-log.(λ ./ Float64.(basis4.I0)), 1, 1, 1, 4)
            est0 = NCH.nchannel_estimate_tile(h0, plan64)
            est0.quality_flag[1] == 0 || continue
            n_checked += 1
            θ(h) = (r = NCH.nchannel_estimate_tile(h, plan64); [r.sino_iodine[1], r.sino_water[1]])
            # analytic IFT sensitivity at the MLE with y = λ: dθ/dh_k = -F⁻¹ ∇λ_k
            Fm = [est0.fisher.AA[1] est0.fisher.AC[1]; est0.fisher.AC[1] est0.fisher.CC[1]]
            A3 = est0.sino_iodine; C3 = est0.sino_water
            _, dA, dC = NCH.nchannel_moments(A3, C3, plan64)
            for k in 1:4
                e = zeros(4); e[k] = 1.0
                fd(ε) = (θ(h0 .+ reshape(ε .* e, 1, 1, 1, 4)) .- θ(h0 .- reshape(ε .* e, 1, 1, 1, 4))) ./ (2ε)
                d1 = fd(1.0e-4); d2 = fd(2.0e-4)
                rich = maximum(abs.(d1 .- d2)) / max(maximum(abs.(d1)), 1.0e-12)
                ift = -(Fm \ [dA[1, 1, 1, k], dC[1, 1, 1, k]])
                ift_err = maximum(abs.(d1 .- ift)) / max(maximum(abs.(ift)), 1.0e-12)
                worst_richardson = max(worst_richardson, rich)
                worst_ift = max(worst_ift, ift_err)
            end
        end
        nch_record!("fd/worst_richardson_rel", worst_richardson)
        nch_record!("fd/worst_ift_rel", worst_ift)
        nch_record!("fd/rays_checked", n_checked)
        @test n_checked >= 4
        @test worst_richardson <= 1.0e-4
        @test worst_ift <= 1.0e-3
    end

    # ── 5. nb04 T-LBF parity ─────────────────────────────────────────────────
    @testset "T-LBF (Lee 2025) vs tlbf_filter_pair" begin
        n_v = 12
        A_t, C_t = truth_grid(8, 1, n_v)
        h_t = synth_h(NB04Oracle.nchannel_forward, A_t, C_t, Φ4_of, basis4.μρ_I, basis4.μρ_W, I04_of;
            rng = MersenneTwister(0x71bf), noise_scale = 1.0)
        est = NCH.nchannel_estimate_tile(h_t, plan4)
        # notebook inputs
        bins = [h_t[:, :, :, k] for k in 1:4]
        measured_ref = NB04Oracle.total_measured_counts(bins, basis4.I0, 1)
        Φ_total = 1 .* vec(sum(Float64.(basis4.Φ); dims = 2))
        expected_ref = NB04Oracle.total_expected_counts(est.sino_iodine, est.sino_water, Φ_total, basis4.μρ_I, basis4.μρ_W)
        measured = NCH.nchannel_total_measured(h_t, plan4)
        expected = NCH.nchannel_total_expected(est.sino_iodine, est.sino_water, plan4)
        @test nch_rel_max(measured, measured_ref) <= 1.0e-5
        @test nch_rel_max(expected, expected_ref) <= 1.0e-5
        for a2 in (24.635648571666497, Inf, 0.0)
            ref = NB04Oracle.tlbf_filter_pair(est.sino_iodine, est.sino_water, expected_ref, measured_ref, a2; alpha1 = 0.9)
            tp = NCH.nchannel_tlbf_plan(8, 1, n_v; alpha1 = 0.9, alpha2 = a2, T = Float32)
            mine = NCH.nchannel_tlbf(est.sino_iodine, est.sino_water, expected, measured, tp)
            rI = nch_record!("tlbf_a2=$(a2)/sino_iodine", nch_rel_max(mine.sino_iodine, ref.sino_iodine))
            rW = nch_record!("tlbf_a2=$(a2)/sino_water", nch_rel_max(mine.sino_water, ref.sino_water))
            @test rI <= 1.0e-5
            @test rW <= 1.0e-5
        end
    end

    # ── 5b. nb04 angular anti-alias apodization ──────────────────────────────
    @testset "angular apodization vs nb04 FFT window" begin
        for (nview, matrix_n) in ((48, 32), (60, 512), (37, 20))
            # verbatim nb04 cell ddfde8bb (nchannel_fbp_pass_mode / _angular_response / fft filter)
            nchannel_fbp_pass_mode = min(nview ÷ 2, ceil(Int, π * matrix_n / 4))
            nchannel_fbp_angular_response = [
                let mode = min(j - 1, nview - (j - 1))
                    mode ≤ nchannel_fbp_pass_mode ? 1.0 :
                    0.5 * (1 + cos(
                        π * (mode - nchannel_fbp_pass_mode) /
                        (nview ÷ 2 - nchannel_fbp_pass_mode),
                    ))
                end
                for j in 1:nview
            ]
            sino = randn(MersenneTwister(nview), Float32, 16, 2, nview)
            spectrum = BS.FFTW.fft(Float64.(sino), 3)
            ref = Float32.(real.(BS.FFTW.ifft(
                spectrum .* reshape(nchannel_fbp_angular_response, 1, 1, nview), 3)))
            ap = NCH.nchannel_angular_plan(nview, matrix_n; T = Float32)
            @test ap.pass_mode == nchannel_fbp_pass_mode
            @test ap.response == nchannel_fbp_angular_response
            mine = NCH.nchannel_angular_apodize(sino, ap)
            r = nch_record!("angular_n$(nview)/rel", nch_rel_max(mine, ref))
            @test r <= 1.0e-5
            # Float64 plan is exact to rounding
            ap64 = NCH.nchannel_angular_plan(nview, matrix_n; T = Float64)
            @test nch_rel_max(NCH.nchannel_angular_apodize(Float64.(sino), ap64), Float64.(ref)) <= 1.0e-6
        end
    end

    # ── 6. End-to-end VMI with the legacy FBP / ACNR / synthesis ─────────────
    @testset "end-to-end VMI HU parity (legacy FBP + ACNR + synth)" begin
        nc, nr, nv = 64, 2, 48
        scanner = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = nr, detector_cols = nc,
            detector_row_size = 1.0, detector_col_size = 1.0)
        # 64 × 1 mm cells at SDD 1080 / SAD 540 see ±16 mm at isocentre: keep the
        # whole phantom inside that (no truncation), 0.94 mm pixels.
        geom = BS.CTGeometry(scanner; n_angles = nv, fov_cm = 3.0, z_cm = 0.1)
        nx = 32
        # water disk with an iodine insert; per-basis density volumes (g/cm³)
        vol_W = zeros(Float32, nx, nx, 2); vol_I = zeros(Float32, nx, nx, 2)
        for j in 1:nx, i in 1:nx
            r2 = (i - 16.5)^2 + (j - 16.5)^2
            r2 <= 13.0^2 && (vol_W[i, j, :] .= 1.0f0)
            (i - 21.5)^2 + (j - 16.5)^2 <= 3.5^2 && (vol_I[i, j, :] .= 0.01f0)
            (i - 11.5)^2 + (j - 16.5)^2 <= 3.5^2 && (vol_I[i, j, :] .= 0.005f0)
        end
        sino_W_true = BS.dd_forward_project(vol_W, geom)
        sino_I_true = BS.dd_forward_project(vol_I, geom)
        # units: bring the projections into the estimator's g/cm² ranges
        sW = 25.0 / maximum(sino_W_true); sI = 0.15 / maximum(sino_I_true)
        C_true = Float64.(sino_W_true) .* sW
        A_true = Float64.(sino_I_true) .* sI
        slab = dual2_tables(nc, nr)
        Core.eval(NB03Oracle, :(nchannel_slab_counts = $slab))
        Base.include_string(NB03Oracle, notebook_cell(NB03, UUID_BASIS), "nb03_basis_e2e.jl")
        b2 = NB03Oracle.nchannel_basis
        Ee, Φe = NCH.nchannel_merge_channels(slab.energies, slab.Φ)
        plan = NCH.nchannel_plan(Φe, Ee, cat(slab.I0...; dims = 3); T = Float32)
        h = synth_h(NB03Oracle.nchannel_forward, A_true, C_true, (c, r) -> b2.Φ[c, r, :, :],
            b2.μρ_I, b2.μρ_W, (c, r) -> b2.I0[c, r, :]; rng = MersenneTwister(0xe2e), noise_scale = 1.0)
        tables = (b2.Φ, b2.μρ_I, b2.μρ_W, b2.I0, b2.μI_eff, b2.μW_eff, b2.normal_II, b2.normal_IW, b2.normal_WW)
        # notebook: tiles of 8 views through the verbatim kernel
        ref_parts = [oracle_tile(NB03Oracle, h[:, :, r, :], tables, ctrl3) for r in NCH._nch_tile_ranges(nv, 8)]
        ref = (sino_iodine = cat((p.sino_iodine for p in ref_parts)...; dims = 3),
            sino_water = cat((p.sino_water for p in ref_parts)...; dims = 3))
        # nb03 §04 per-basis kernels, §05 ACNR, §06 synthesis — the legacy functions
        iodine_filter = BS.CustomFilter((0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 0.40, 0.12, 0.03, 0.001))
        water_filter = BS.CustomFilter((0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 0.8744, 0.6003, 0.3031, 0.0266))
        fbp(sino, filter) = Float32.(BS.fdk_reconstruct(Float32.(sino), geom, (nx, nx, 2); filter = filter))
        function acnr(W, I)
            W = copy(W); I = copy(I)
            BS.apply_acnr_kalender!(W, I; hp_sigma_px = 1.5, window = 4, passes = 5, beta_max = 14.0)
            (; water = W, iodine = I)
        end
        energies = [50.0, 70.0, 100.0, 140.0]
        synth(W, I, es) = cat((BS.synth_vmi_2basis(W, I .* 1000.0f0; energy_keV = e) for e in es)...; dims = 4)

        chain = NCH.nchannel_vmi_chain(h, plan; fbp_iodine = s -> fbp(s, iodine_filter),
            fbp_water = s -> fbp(s, water_filter), acnr = acnr, synth = synth, energies = energies)
        rI = nch_record!("e2e/sino_iodine", nch_rel_max(chain.sino_iodine, ref.sino_iodine))
        rW = nch_record!("e2e/sino_water", nch_rel_max(chain.sino_water, ref.sino_water))
        @test rI <= 1.0e-5
        @test rW <= 1.0e-5
        # notebook path on the notebook sinograms
        vI_ref = fbp(ref.sino_iodine, iodine_filter); vW_ref = fbp(ref.sino_water, water_filter)
        a_ref = acnr(vW_ref, vI_ref)
        vmi_ref = synth(a_ref.water, a_ref.iodine, energies)
        rV = nch_record!("e2e/vmi_HU", nch_rel_max(chain.vmis, vmi_ref))
        @test rV <= 1.0e-4
        @test all(isfinite, chain.vmis)
        # pure synthesis == legacy synth_vmi_2basis
        α = NCH.nchannel_vmi_alphas(energies; T = Float32)
        @test nch_rel_max(NCH.nchannel_synth_vmi(a_ref.water, a_ref.iodine, α), vmi_ref) <= 1.0e-6
        # sanity: the water basis image correlates with the phantom
        @test cor(vec(chain.vol_water[:, :, 1]), vec(vol_W[:, :, 1])) > 0.9
        nch_record!("e2e/max_outer_used", maximum(maximum(p.outer_iterations) for p in ref_parts))
        nch_record!("e2e/max_inner_used", maximum(maximum(p.inner_iterations) for p in ref_parts))
    end

    println("\n── n-channel parity numbers ──")
    for k in sort(collect(keys(NCH_PARITY)))
        println(rpad(k, 44), NCH_PARITY[k])
    end
    println("elapsed: ", round(time() - _NCH_T0; digits = 1), " s")
end
