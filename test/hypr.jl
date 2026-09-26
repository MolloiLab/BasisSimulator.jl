# Generalized HYPR-LR: window profiles and kernels, the projection-domain instance (unbiased in
# counts for weights computed from the totals, rows never mixed, the dispersion read off air rays),
# the image-domain instance, and `vmi_pipeline(; denoiser = SpectralHYPR())`. Synthetic and
# CPU-only; uses `_toy_response` / `_toy_channels` / `_noisy_channels` from nchannel.jl.

using Statistics: var

@testset "HYPR profiles and kernels" begin
    @test BS.profile_weights(BS.BoxProfile(), 5) == ones(5)
    @test BS.profile_weights(BS.TriangleProfile(), 3) ≈ [0.5, 1.0, 0.5]
    p = BS.CustomProfile((0.0, 0.5, 1.0), (1.0, 0.8, 0.2))
    @test BS.profile_weight(p, 0.25) ≈ 0.9
    @test BS.profile_weight(p, 2.0) ≈ 0.2
    @test_throws ArgumentError BS.CustomProfile((0.5, 0.0), (1.0, 1.0))
    @test_throws ArgumentError BS.CustomProfile((0.0, 1.0), (1.0, -0.1))
    k = BS.HYPRKernel((3, 5))
    @test k.guided && k.linear && k.profile isa BS.BoxProfile
    @test !BS.HYPRKernel((3, 3, 7)).linear
    @test_throws ArgumentError BS.HYPRKernel((4, 3))
    @test_throws ArgumentError BS.HYPRKernel((3, 3, 7); linear = true)
    @test_throws ArgumentError BS.HYPRKernel((3,))
    d = BS.SpectralHYPR()
    @test d.projection.kernel.window == (3, 3) && d.projection.dispersion === :measured
    @test d.projection.view_stride == 2
    @test d.image.noise_window == 15 && d.image.candidates == (3, 5, 7, 11, 15, 21, 31, 41)
    @test BS.SpectralHYPR(image = nothing).image === nothing
    @test_throws ArgumentError BS.ProjectionHYPR(dispersion = :bogus)
    @test_throws ArgumentError BS.ProjectionHYPR(view_stride = 3)
    @test_throws ArgumentError BS.ImageHYPR(candidates = (3, 4))
    @test_throws ArgumentError BS.ImageHYPR(noise_window = 4)
    @test occursin("3 × 3", sprint(show, k)) || occursin("3 × 5", sprint(show, k))
end

@testset "hypr_lr keeps totals, is unbiased and never mixes rows" begin
    # rays of one spectral shape per row; row 2 has a different split than row 1
    nc, nr, nv, K = 48, 2, 48, 2
    λ = [600.0 400.0; 300.0 700.0]            # row × channel expected counts
    I0 = ones(Float64, nc, nr, K) .* 2000.0
    h_true = [fill(Float32(-log(λ[r, k] / 2000)), nc, 1, nv) for k in 1:K, r in 1:nr]
    clean = [cat(h_true[k, 1], h_true[k, 2]; dims = 2) for k in 1:K]
    # noise-free input comes back exactly, row by row
    out = BS.hypr_lr(clean, I0; kernel = BS.HYPRKernel((5, 5)))
    for k in 1:K
        @test maximum(abs.(out[k] .- clean[k])) < 1.0e-4
    end
    # Poisson draws: every ray keeps its own total, and the mean of ŷ is λ
    rng = Random.MersenneTwister(3)
    sums = zeros(nc, nr, nv, K)
    R = 12
    for _ in 1:R
        noisy = [similar(clean[k]) for k in 1:K]
        for k in 1:K, idx in CartesianIndices(clean[k])
            n = max(BS._poisson_sample(rng, λ[idx[2], k]), 1)
            noisy[k][idx] = Float32(-log(n / 2000))
        end
        den = BS.hypr_lr(noisy, I0; kernel = BS.HYPRKernel((5, 5)))
        y(h) = 2000 .* exp.(-Float64.(h))
        @test maximum(abs.(sum(y.(den)) .- sum(y.(noisy)))) < 1.0e-2 * maximum(sum(y.(noisy)))
        for k in 1:K
            sums[:, :, :, k] .+= y(den[k])
        end
    end
    for r in 1:nr, k in 1:K
        @test mean(sums[:, r, :, k]) / R ≈ λ[r, k] rtol = 2.0e-3
    end
end

@testset "local linear split follows a gradient" begin
    # the split varies linearly along the columns; a locally constant fit is biased at the
    # detector edge (an asymmetric window), the local linear one is not
    nc, nv = 40, 16
    p1 = range(0.3, 0.7, length = nc)
    T = 1000.0
    I0 = fill(2000.0, nc, 1, 2)
    h1 = Float32.(repeat(reshape(-log.(T .* p1 ./ 2000), nc, 1, 1), 1, 1, nv))
    h2 = Float32.(repeat(reshape(-log.(T .* (1 .- p1) ./ 2000), nc, 1, 1), 1, 1, nv))
    lin = BS.hypr_lr([h1, h2], I0; kernel = BS.HYPRKernel((7, 3); guided = false))
    con = BS.hypr_lr([h1, h2], I0; kernel = BS.HYPRKernel((7, 3); guided = false, linear = false))
    @test maximum(abs.(lin[1] .- h1)) < 1.0e-4
    @test maximum(abs.(con[1][1, 1, :] .- h1[1, 1, :])) > 1.0e-2
end

@testset "estimate_dispersion reads the variance-to-mean ratio off air rays" begin
    rng = Random.MersenneTwister(5)
    nc, nr, nv = 16, 2, 400
    I0 = fill(5.0e4, nc, nr, 2)
    D = (1.0, 3.0)
    channels = map(1:2) do k
        h = Array{Float32}(undef, nc, nr, nv)
        for idx in CartesianIndices(h)
            # the object covers the central columns; the outer four on each side are air
            att = 4 < idx[1] <= nc - 4 ? 1.5 : 0.0
            s = D[k] * BS._poisson_sample(rng, 5.0e4 * exp(-att) / D[k])
            h[idx] = Float32(-log(max(s, 1) / 5.0e4))
        end
        h
    end
    est = BS.estimate_dispersion(channels, I0)
    @test est[1] ≈ 1.0 rtol = 0.08
    @test est[2] ≈ 3.0 rtol = 0.08
end

@testset "estimate_dispersion: air rays with residual attenuation, and rays grazing the object" begin
    # the air of a phantom's volume: every ray that misses the object still crosses a few mm of
    # attenuating air, which changes slowly with the view (a square volume), and two columns graze the
    # object in part of the rotation
    rng = Random.MersenneTwister(11)
    nc, nr, nv = 40, 2, 600
    I0 = fill(8.0e4, nc, nr, 2)
    D = (1.3, 1.25)
    channels = map(1:2) do k
        h = Array{Float32}(undef, nc, nr, nv)
        for idx in CartesianIndices(h)
            c, v = idx[1], idx[3]
            air = 0.006 + 0.002 * abs(sin(2π * v / nv))
            att = 10 < c <= nc - 10 ? 1.5 : (c in (10, nc - 9) && v <= nv ÷ 3 ? 0.02 : 0.0)
            s = D[k] * BS._poisson_sample(rng, 8.0e4 * exp(-(air + att)) / D[k])
            h[idx] = Float32(-log(max(s, 1) / 8.0e4))
        end
        h
    end
    est = BS.estimate_dispersion(channels, I0)
    @test est[1] ≈ 1.3 rtol = 0.03
    @test est[2] ≈ 1.25 rtol = 0.03
end

@testset "guided_pool stops at structure" begin
    rng = Random.MersenneTwister(7)
    nx, nz = 32, 5
    step = [i <= 16 ? 0.0 : 100.0 for i in 1:nx, j in 1:nx, z in 1:nz]
    noisy = step .+ randn(rng, nx, nx, nz)
    k = BS.HYPRKernel((5, 5, 1); linear = false)
    pooled = BS.guided_pool(noisy, noisy, ones(size(noisy)), k)
    @test std(pooled[1:14, :, :]) < 0.7 * std(noisy[1:14, :, :])
    @test abs(mean(pooled[16, :, :])) < 1.0 && abs(mean(pooled[17, :, :]) - 100) < 1.0
    unguided = BS.guided_pool(noisy, noisy, ones(size(noisy)), BS.HYPRKernel((5, 5, 1); guided = false, linear = false))
    @test mean(unguided[16, :, :]) > 20                     # plain box blurs across the edge
    @test BS.guided_pool(fill(3.0, 8, 8, 2), zeros(8, 8, 2), ones(8, 8, 2), k) ≈ fill(3.0, 8, 8, 2)
    # within the slice only: a window across slices is refused, and slices never exchange values
    @test_throws ArgumentError BS.guided_pool(noisy, noisy, ones(size(noisy)), BS.HYPRKernel((5, 5, 3); linear = false))
    layered = cat(zeros(8, 8, 1), fill(100.0, 8, 8, 1); dims = 3)
    @test BS.guided_pool(layered, layered, ones(8, 8, 2), BS.HYPRKernel((5, 5, 1); guided = false, linear = false)) ≈ layered
end

@testset "view_stride pools one parity of views" begin
    # with a stride of 2, the pooled split of an even view is a function of the even views alone
    rng = Random.MersenneTwister(11)
    nc, nr, nv = 12, 1, 24
    I0 = fill(2000.0, nc, nr, 2)
    base = [Float32.(-log.((400 .+ 200 .* rand(rng, nc, nr, nv)) ./ 2000)) for _ in 1:2]
    moved = [copy(b) for b in base]
    for b in moved
        b[:, :, 1:2:nv] .+= 0.3f0                          # perturb the odd views only
    end
    k = BS.HYPRKernel((3, 3))
    a2, b2 = BS.hypr_lr(base, I0; kernel = k, view_stride = 2), BS.hypr_lr(moved, I0; kernel = k, view_stride = 2)
    @test all(maximum(abs.(a2[c][:, :, 2:2:nv] .- b2[c][:, :, 2:2:nv])) < 1.0e-6 for c in 1:2)
    a1, b1 = BS.hypr_lr(base, I0; kernel = k), BS.hypr_lr(moved, I0; kernel = k)
    @test maximum(abs.(a1[1][:, :, 2:2:nv] .- b1[1][:, :, 2:2:nv])) > 1.0e-3   # adjacent views mix
end

@testset "local_noise measures a nonstationary noise level" begin
    rng = Random.MersenneTwister(13)
    nx, nz = 64, 2
    σtrue = [i <= 32 ? 1.0 : 3.0 for i in 1:nx, j in 1:nx, z in 1:nz]
    d = σtrue .* randn(rng, nx, nx, nz)
    σ = BS.local_noise(d, 15)
    @test mean(σ[4:24, :, :]) ≈ 1.0 rtol = 0.05
    @test mean(σ[42:60, :, :]) ≈ 3.0 rtol = 0.05
    @test_throws ArgumentError BS.local_noise(d, 4)
end

@testset "vmi_pipeline with SpectralHYPR" begin
    scanner = BS.PCCTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4,
        detector_cols = 128, detector_row_size = 1.0, detector_col_size = 1.5,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
    )
    geom = BS.CTGeometry(scanner; n_angles = 180, fov_cm = 16.0, z_cm = 0.4)
    γ = ((1:geom.n_cols) .- (geom.n_cols + 1) / 2) .* geom.pixel_size ./ geom.SAD
    chord = @. 2 * sqrt(max(5.0^2 - (geom.SAD * sin(γ))^2, 0.0))
    spread(v) = repeat(reshape(v, :, 1, 1), 1, geom.n_rows, geom.n_angles)
    toy = _toy_response()
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    clean = _toy_channels(basis, spread(0.005 .* chord), spread(chord))
    noisy = _noisy_channels(clean, basis)
    common = (; basis, geom, matrix_size = (64, 64, geom.n_rows), vmi_energies = (40, 70))
    centre = 28:37

    ref = BS.vmi_pipeline(; channels = clean, common..., use_acnr = false)
    plain = BS.vmi_pipeline(; channels = noisy, common..., use_acnr = false)
    out = BS.vmi_pipeline(; channels = noisy, common..., denoiser = BS.SpectralHYPR())
    @test out.settings.acnr.on === :complement               # ACNR runs, on the complement, first
    @test out.settings.denoiser.projection.kernel.window == (3, 3)
    @test out.settings.denoiser.projection.view_stride == 2
    @test all(d -> 0.8 < d < 1.25, out.settings.denoiser.dispersion)   # photon counting ≈ 1
    est = out.settings.denoiser.image_estimates
    @test 40 <= est.Estar <= 140
    @test est.window in BS.ImageHYPR().candidates && first(est.risks) == (3, 1.0)
    for k in 1:geom.n_rows
        @test mean(out.images.water[centre, centre, k]) ≈ mean(ref.images.water[centre, centre, k]) rtol = 0.02
    end
    # the image-domain instance takes out most of the 40 keV spectral noise
    σ(x) = std(x.vmis[centre, centre, :, 1])
    @test σ(out) < 0.7 * σ(plain)
    # either instance alone
    proj = BS.vmi_pipeline(; channels = noisy, common..., use_acnr = false, denoiser = BS.SpectralHYPR(image = nothing))
    @test proj.settings.denoiser.image === nothing && σ(proj) <= 1.05 * σ(plain)
    img = BS.vmi_pipeline(; channels = noisy, common..., denoiser = BS.SpectralHYPR(projection = nothing))
    @test img.settings.denoiser.dispersion === nothing && σ(img) < 0.7 * σ(plain)
    # the sinograms, for the pair tests
    sinos = BS.vmi_pipeline(; channels = noisy, common..., use_acnr = false, keep_sinograms = true).sinograms
    sp(filter; kw...) = BS.spectral_pair(sinos.water, sinos.iodine, geom, common.matrix_size; filter, kw...)
    # one window: FDK of each basis image; the same window on composite and complement is that window
    one = sp(BS.SoftFilter())
    @test one.water ≈ plain.images.water && one.iodine ≈ plain.images.iodine
    # (inside the reconstruction circle: FDK fills the outside with a constant, which the pair
    # algebra does not carry)
    twice = sp(BS.PairFilter(BS.SoftFilter(), BS.SoftFilter()))
    circle = repeat(BS._reconstruction_circle(64, 64), 1, 1, geom.n_rows)
    @test twice.water[circle] ≈ one.water[circle] rtol = 1.0e-4
    @test twice.iodine[circle] ≈ one.iodine[circle] rtol = 1.0e-4
    @test 40 <= one.Estar <= 140 && size(one.Σ) == (2, 2) && length(one.halves) == 2
    # the composite is reconstructed with its window alone, whatever the complement's
    pf = sp(BS.PairFilter(BS.SoftFilter(), BS.BoneFilter()))
    M(p) = p.f[1] .* p.iodine .+ p.f[2] .* p.water
    @test M(pf)[circle] ≈ M(twice)[circle] rtol = 1.0e-4
    @test !(pf.iodine[circle] ≈ twice.iodine[circle])
    # a fixed basis is used as given
    fixed = sp(BS.PairFilter(BS.SoftFilter(), BS.BoneFilter()); basis = (Estar = 70.0, β = 0.0))
    @test fixed.basis == (Estar = 70.0, β = 0.0)
    # the composite ACNR and the image-domain instance act on is the reconstructed pair's own
    # minimum-noise VMI, so its noise is uncorrelated with the complement's
    μ(E) = [BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, E), BS.compute_mass_μ_at_energy(BS.XA.Materials.water, E)]
    v(E) = let g = μ(E) ./ μ(E)[2]; g' * fixed.Σ * g end
    @test 40 <= fixed.Estar <= 140 && v(fixed.Estar) <= min(v(40.0), v(140.0))
    # ACNR on the complement leaves the composite exactly where it was
    before = M(pf)
    BS.acnr_complement!(pf)
    @test M(pf) ≈ before rtol = 1.0e-5
    @test !(pf.iodine ≈ sp(BS.PairFilter(BS.SoftFilter(), BS.BoneFilter())).iodine)
    # the image-domain instance keeps an unpooled composite and returns its selected window
    x = BS.image_hypr(sp(BS.SoftFilter()))
    @test M((f = one.f, water = x.water, iodine = x.iodine))[circle] ≈ M(one)[circle] rtol = 1.0e-4
    # the complement uses the pair's β as measured before ACNR, not re-measured after it
    q = sp(BS.PairFilter(BS.SoftFilter(), BS.BoneFilter())); β0 = q.β
    BS.acnr_complement!(q)
    @test BS.image_hypr(q).β == β0
    one1 = BS.image_hypr(sp(BS.SoftFilter()); image = BS.ImageHYPR(candidates = (7,)))
    @test one1.window == 7
    # a fixed composite energy is used as given, its β measured at that energy
    fe = sp(BS.SoftFilter(); composite_energy = 65.0)
    @test fe.Estar == 65.0 && fe.f ≈ μ(65.0)
    # vmi_pipeline with a PairFilter and no image instance: the pair, then ACNR on its complement
    pp = BS.vmi_pipeline(; channels = noisy, common..., fbp_filter = BS.PairFilter(BS.SoftFilter(), BS.BoneFilter()))
    @test pp.settings.acnr.on === :complement && pp.settings.pair.Estar == pf.Estar
    # basis-vmi's plain chain is unchanged: one window, ACNR on the pair
    @test BS.vmi_pipeline(; channels = noisy, common...).settings.acnr.on === :pair
    # the spectral pair measures its noise from FDK halves, so HIR is refused
    @test_throws ArgumentError BS.vmi_pipeline(; channels = noisy, common...,
        denoiser = BS.SpectralHYPR(), recon_method = :hir)
    @test_throws ArgumentError BS.vmi_pipeline(; channels = noisy, common...,
        fbp_filter = BS.PairFilter(BS.SoftFilter(), BS.SoftFilter()), recon_method = :hir)
end
