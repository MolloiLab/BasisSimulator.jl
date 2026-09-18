# K-channel profile-likelihood decomposition, count-domain channel preparation, T-LBF and the
# VMI chain. Everything here is synthetic and CPU-only: the forward model is evaluated exactly,
# so the estimator has a known answer to recover.

# A toy absolute response: a smooth spectrum on 20:2:140 keV split into K windows.
function _toy_response(; K = 4, total = 4.0e4, n_col = 1, n_row = 1, tilt = 0.0)
    E = collect(20.0:2.0:140.0)
    spectrum = @. max(E - 18, 0) * max(142 - E, 0) * exp(-25 / E)
    edges = K == 4 ? [20.0, 35.0, 55.0, 70.0, Inf] : [20.0, 62.0, Inf]
    Φ = zeros(Float64, n_col, n_row, length(E), K)
    for k in 1:K, (e, energy) in pairs(E)
        edges[k] <= energy < edges[k + 1] || continue
        for r in 1:n_row, c in 1:n_col
            Φ[c, r, e, k] = spectrum[e] * (1 + tilt * (c - 1) + 0.5tilt * (r - 1))
        end
    end
    Φ .*= total / sum(Φ[1, 1, :, :])
    I0 = dropdims(sum(Φ; dims = 3); dims = 3)
    return (; E, Φ, I0)
end

# Exact noise-free corrected log transmission of channel k for a basis pair (A, C) in g/cm².
function _toy_channels(basis, A::AbstractArray, C::AbstractArray)
    K = basis.n_channels
    return map(1:K) do k
        h = similar(A, Float32)
        for idx in CartesianIndices(A)
            c = size(basis.Φ, 1) == 1 ? 1 : idx[1]
            r = size(basis.Φ, 2) == 1 ? 1 : idx[2]
            λ = sum(
                Float64(basis.Φ[c, r, e, k]) *
                    exp(-Float64(basis.μρ_I[e]) * A[idx] - Float64(basis.μρ_W[e]) * C[idx])
                    for e in eachindex(basis.E)
            )
            h[idx] = Float32(-log(λ / Float64(basis.I0[c, r, k])))
        end
        h
    end
end

@testset "spectral_basis" begin
    toy = _toy_response()
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    @test basis.n_channels == 4 && !basis.ray_resolved
    @test size(basis.Φ) == (1, 1, length(toy.E), 4) && size(basis.I0) == (1, 1, 4)
    @test basis.I0_relerr < 5.0e-5
    @test all(basis.μI_eff .> basis.μW_eff)                # iodine attenuates more per g/cm²
    @test issorted(vec(basis.μW_eff); rev = true)          # harder windows attenuate less
    # the likelihood needs an absolute response: a rescaled I0 is an error, not a renormalisation
    @test_throws ErrorException BS.spectral_basis(
        energies = toy.E, response = toy.Φ, I0 = 1.01 .* toy.I0,
    )
    @test_throws DimensionMismatch BS.spectral_basis(
        energies = toy.E[1:(end - 1)], response = toy.Φ, I0 = toy.I0,
    )

    W = reshape(toy.Φ, length(toy.E), 4)
    from_bins = BS.spectral_basis_from_bins(energies = toy.E, W_applied = W, I0_bins = vec(toy.I0))
    @test from_bins.Φ == basis.Φ && from_bins.I0 == basis.I0

    # two acquisitions on different grids merge onto their union, each scaled by its own air counts
    low = (energies = 20.0:2.0:80.0, response = ones(2, 1, 31), I0_ray = fill(1.0e4, 2, 1))
    high = (energies = 20.0:2.0:140.0, response = ones(2, 1, 61), I0_ray = fill(3.0e4, 2, 1))
    pair = BS.spectral_basis_from_acquisitions(acquisitions = [low, high])
    @test pair.n_channels == 2 && pair.ray_resolved && length(pair.E) == 61
    @test pair.I0[:, 1, 1] ≈ fill(1.0f4, 2) && pair.I0[:, 1, 2] ≈ fill(3.0f4, 2)
    @test all(pair.Φ[:, :, 32:end, 1] .== 0)               # nothing above 80 keV in the low scan
end

@testset "decompose_nchannel recovers the basis pair" begin
    for K in (4, 2)
        toy = _toy_response(; K)
        basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
        A = Float64[a for a in (0.0, 0.01, 0.05, 0.15), _ in 1:1, _ in 1:3]
        C = Float64[c for _ in 1:4, _ in 1:1, c in (2.0, 12.0, 30.0)]
        channels = _toy_channels(basis, A, C)
        out = BS.decompose_nchannel(; channels, basis, tile_views = 2, keep_diagnostics = true)
        @test out.method === :nchannel && out.n_channels == K
        @test size(out.sino_iodine) == size(A)
        @test maximum(abs.(out.sino_iodine .- A)) < 2.0e-4
        @test maximum(abs.(out.sino_water .- C)) < 5.0e-3
        @test out.quality.frac_not_converged == 0 && out.quality.frac_invalid == 0
        @test out.quality.frac_infeasible == 0 && out.quality.frac_bound_iodine == 0
        @test all(out.diagnostics.fisher.AA .> 0)
        # tiling is an implementation detail: any tile size gives the same answer
        whole = BS.decompose_nchannel(; channels, basis, tile_views = 3)
        @test whole.sino_iodine == out.sino_iodine && whole.sino_water == out.sino_water
    end

    # a ray-resolved response (what a bowtie produces) is looked up per ray
    toy = _toy_response(; n_col = 3, tilt = 0.4)
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    @test basis.ray_resolved
    A = fill(0.08, 3, 1, 2)
    C = fill(15.0, 3, 1, 2)
    out = BS.decompose_nchannel(; channels = _toy_channels(basis, A, C), basis)
    @test maximum(abs.(out.sino_iodine .- A)) < 2.0e-4
    @test maximum(abs.(out.sino_water .- C)) < 5.0e-3

    # rays outside the bounds are flagged, not silently clamped
    toy = _toy_response()
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    channels = _toy_channels(basis, fill(0.0, 1, 1, 1), fill(80.0, 1, 1, 1))
    out = BS.decompose_nchannel(; channels, basis)
    @test out.quality.frac_bound_water == 1 && out.quality.frac_infeasible == 1

    @test_throws DimensionMismatch BS.decompose_nchannel(; channels = channels[1:3], basis)

    # A real spectrum grid reaches down to a few keV, where the response is exactly zero and
    # exp(+μ_water·2) overflows Float32. Zero-response energies must not turn into 0·Inf = NaN:
    # that would flag every ray infeasible and silently skip the bisection start.
    E_low = vcat([3.0, 4.0], toy.E)
    Φ_low = cat(zeros(1, 1, 2, 4), toy.Φ; dims = 3)
    @test exp(Float32(BS.compute_mass_μ_at_energy(BS.XA.Materials.water, 3.0)) * 2.0f0) == Inf32
    low = BS.spectral_basis(energies = E_low, response = Φ_low, I0 = toy.I0)
    truth_A, truth_C = fill(0.05, 2, 1, 2), fill(12.0, 2, 1, 2)
    out = BS.decompose_nchannel(; channels = _toy_channels(low, truth_A, truth_C), basis = low)
    @test out.quality.frac_infeasible == 0 && out.quality.frac_invalid == 0
    @test maximum(abs.(out.sino_iodine .- truth_A)) < 2.0e-4
    @test all(isfinite, BS.total_expected_counts(truth_A, fill(-2.0, 2, 1, 2), low.Φ, low.μρ_I, low.μρ_W))
    controls = BS.NChannelControls(outer_iterations = 1, inner_iterations = 1)
    @test controls.tile_views == 8 && controls.iodine_bounds == (-0.1f0, 0.4f0)
end

@testset "channel preparation sums counts" begin
    toy = _toy_response(; n_col = 2, n_row = 3, tilt = 0.5)
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    A = fill(0.06, 2, 3, 2)
    C = fill(18.0, 2, 3, 2)
    channels = _toy_channels(basis, A, C)

    reduced = BS.reduce_detector_rows(; channels, basis)
    @test reduced.n_rows == 3 && size(reduced.channels[1]) == (2, 1, 2)
    @test size(reduced.basis.Φ) == (2, 1, length(toy.E), 4)
    @test reduced.basis.I0[:, 1, :] ≈ dropdims(sum(basis.I0; dims = 2); dims = 2)
    for k in 1:4
        counts = sum(Float64.(basis.I0[:, :, k]) .* exp.(-Float64.(channels[k])); dims = 2)
        expected = -log.(counts ./ sum(Float64.(basis.I0[:, :, k]); dims = 2))
        @test reduced.channels[k] ≈ Float32.(expected) rtol = 1.0e-5
    end
    # the summed measurement is still explained by the summed response: same (A, C) comes back
    out = BS.decompose_nchannel(channels = reduced.channels, basis = reduced.basis)
    @test maximum(abs.(out.sino_iodine .- 0.06)) < 2.0e-4
    @test maximum(abs.(out.sino_water .- 18.0)) < 5.0e-3
    # a subset of rows is a smaller sum
    @test BS.reduce_detector_rows(; channels, basis, rows = 2:3).n_rows == 2

    merged = BS.merge_channels(; channels, basis, groups = [1:2, 3:4])
    @test merged.basis.n_channels == 2
    @test merged.basis.I0[:, :, 1] ≈ basis.I0[:, :, 1] .+ basis.I0[:, :, 2]
    @test merged.basis.Φ[:, :, :, 2] ≈ basis.Φ[:, :, :, 3] .+ basis.Φ[:, :, :, 4]
    two = BS.decompose_nchannel(channels = merged.channels, basis = merged.basis)
    @test maximum(abs.(two.sino_iodine .- 0.06)) < 2.0e-4

    # The groups have to partition the channels. Overlapping ones would feed one physical
    # measurement into the likelihood twice as though it were independent; missing ones would be
    # dropped without a word.
    @test_throws ArgumentError BS.merge_channels(; channels, basis, groups = [1:2, 2:3])
    @test_throws ArgumentError BS.merge_channels(; channels, basis, groups = [1:2])
    @test_throws ArgumentError BS.merge_channels(; channels, basis, groups = [1:3, 3:4])
    @test_throws DimensionMismatch BS.merge_channels(;
        channels = channels[1:3], basis, groups = [1:2, 3:4]
    )
    # and one channel cannot determine two materials
    single = BS.merge_channels(; channels, basis, groups = [1:4])
    @test_throws ArgumentError BS.decompose_nchannel(
        channels = single.channels, basis = single.basis
    )

    both = BS.prepare_channels(; channels, basis, merge_groups = [1:2, 3:4], reduce_rows = true)
    @test both.basis.n_channels == 2 && both.n_rows == 3 && size(both.channels[1]) == (2, 1, 2)
    untouched = BS.prepare_channels(; channels, basis)
    @test untouched.channels === channels && untouched.n_rows === nothing
end

@testset "decompose_cong shares the interface" begin
    toy = _toy_response(; K = 2)
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    A = fill(0.05, 4, 1, 2)
    C = fill(12.0, 4, 1, 2)
    channels = _toy_channels(basis, A, C)
    out = BS.decompose_cong(; channels, basis)
    @test out.method === :cong && out.n_channels == 2
    @test maximum(abs.(out.sino_iodine .- A)) < 1.0e-3
    @test maximum(abs.(out.sino_water .- C)) < 2.0e-2
    material = BS.cong_material_basis(basis)
    @test sum(material.ŵ_L) ≈ 1 && sum(material.ŵ_H) ≈ 1

    toy4 = _toy_response()
    four = BS.spectral_basis(energies = toy4.E, response = toy4.Φ, I0 = toy4.I0)
    @test_throws ArgumentError BS.cong_material_basis(four)
    @test_throws ArgumentError BS.decompose_cong(channels = [channels..., channels[1]], basis = basis)
end

@testset "tlbf_denoise" begin
    rng = Random.Xoshiro(7)
    n_col, n_view = 24, 16
    iodine = Float32.(0.05 .+ 0.01 .* randn(rng, n_col, 1, n_view))
    water = Float32.(20.0 .+ 0.5 .* randn(rng, n_col, 1, n_view))
    expected = fill(1.0f4, n_col, 1, n_view)
    measured = Float32.(1.0e4 .+ 100 .* randn(rng, n_col, 1, n_view))

    identity_run = BS.tlbf_denoise(iodine, water, expected, measured; alpha2 = 0)
    @test identity_run.sino_iodine == iodine && identity_run.sino_water == water

    # equal expected counts everywhere: every likelihood weight is 1, so it is the spatial filter
    spatial = BS.tlbf_denoise(iodine, water, expected, measured; alpha2 = Inf)
    default = BS.tlbf_denoise(iodine, water, expected, measured)
    @test default.sino_iodine ≈ spatial.sino_iodine && default.sino_water ≈ spatial.sino_water
    @test std(spatial.sino_water) < 0.7 * std(water)          # it smooths
    @test mean(spatial.sino_water) ≈ mean(water) rtol = 1.0e-3 # and preserves the mean

    # by hand at an interior ray, radius 1: 3×3 Gaussian with circular views
    one = BS.tlbf_denoise(iodine, water, expected, measured; alpha2 = Inf, radius = 1, alpha1 = 0.9)
    weights = [exp(-(dc^2 + dv^2) / (2 * 0.9^2)) for dc in -1:1, dv in -1:1]
    c, v = 5, 1                                                # view 1 wraps to the last view
    by_hand = sum(
        weights[dc + 2, dv + 2] * water[c + dc, 1, mod1(v + dv, n_view)] for dc in -1:1, dv in -1:1
    ) / sum(weights)
    @test one.sino_water[c, 1, v] ≈ by_hand rtol = 1.0e-5

    # a neighbour that explains the centre's counts badly is down-weighted: an edge survives
    step_water = Float32.(vcat(fill(10.0, 12), fill(30.0, 12)) .* ones(1, 1, n_view))
    step_expected = Float32.(vcat(fill(2.0e4, 12), fill(2.0e3, 12)) .* ones(1, 1, n_view))
    edge = BS.tlbf_denoise(fill(0.0f0, n_col, 1, n_view), step_water, step_expected, step_expected)
    blur = BS.tlbf_denoise(fill(0.0f0, n_col, 1, n_view), step_water, step_expected, step_expected; alpha2 = Inf)
    @test abs(edge.sino_water[12, 1, 4] - 10) < 1.0e-3 < abs(blur.sino_water[12, 1, 4] - 10)

    # Rows: the neighbourhood is (column, view) and never crosses rows, so a multi-row sinogram is
    # the published single-row filter applied to each row — bit for bit, whatever the other rows hold.
    rows3 = (
        cat(iodine, 2 .* iodine, iodine .+ 0.01f0; dims = 2),
        cat(water, water .+ 1, 0.5f0 .* water; dims = 2),
        cat(expected, 1.1f0 .* expected, expected; dims = 2),
        cat(measured, measured, 0.9f0 .* measured; dims = 2),
    )
    together = BS.tlbf_denoise(rows3...)
    for r in 1:3
        alone = BS.tlbf_denoise((view(a, :, r:r, :) for a in rows3)...)
        @test together.sino_iodine[:, r:r, :] == alone.sino_iodine
        @test together.sino_water[:, r:r, :] == alone.sino_water
    end
    @test_throws DimensionMismatch BS.tlbf_denoise(rows3[1], rows3[2], rows3[3], measured)

    # the two count maps the filter is built from
    toy = _toy_response()
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    A = fill(0.04, 3, 1, 2)
    C = fill(9.0, 3, 1, 2)
    channels = _toy_channels(basis, A, C)
    seen = BS.total_measured_counts(channels, basis.I0)
    model = BS.total_expected_counts(A, C, basis.Φ, basis.μρ_I, basis.μρ_W)
    @test seen ≈ model rtol = 1.0e-4                           # noise-free: measured = expected
    @test_throws DimensionMismatch BS.total_measured_counts(channels[1:2], basis.I0)
end

@testset "angular_antialias_response" begin
    @test BS.angular_antialias_response(n_views = 360, matrix_nx = 512) == ones(360)
    response = BS.angular_antialias_response(n_views = 1200, matrix_nx = 512)
    pass = ceil(Int, π * 512 / 4)                              # 403
    @test all(response[1:(pass + 1)] .== 1)
    @test response[601] ≈ 0 atol = 1.0e-12
    @test response[2:end] ≈ reverse(response[2:end])           # symmetric in ±mode
    @test issorted(response[(pass + 1):601]; rev = true)
end

# The oracle is the published notebook itself: the estimator and T-LBF cells of
# docs/notebooks/04_pcct_vmi.jl are extracted by cell UUID and evaluated unchanged in a scratch
# module, then run side by side with the package code on the same inputs.
module NB04Oracle
    using BasisSimulator
    const BS = BasisSimulator
    to_gpu(x) = x
end

@testset "parity with the published notebook cells" begin
    notebook = joinpath(@__DIR__, "..", "docs", "notebooks", "04_pcct_vmi.jl")
    function notebook_cell(uuid)
        text = read(notebook, String)
        marker = findfirst("# ╔═╡ $uuid", text)
        marker === nothing && error("cell $uuid not found in $notebook")
        start = last(marker) + 1
        stop = findnext("# ╔═╡ ", text, start)
        return stop === nothing ? text[start:end] : text[start:(first(stop) - 1)]
    end
    Base.include_string(NB04Oracle, notebook_cell("4985581f-616d-4bb7-ab9b-967d7250b28b"), "nb04_controls")
    Base.include_string(NB04Oracle, notebook_cell("73371177-0498-4eda-897b-651c94f43e83"), "nb04_kernel")
    Base.include_string(NB04Oracle, notebook_cell("f3de45a6-4818-4ee1-ad56-c65797119dee"), "nb04_tlbf")

    toy = _toy_response()
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    rng = Random.Xoshiro(11)
    shape = (16, 1, 6)
    A = 0.05 .* rand(rng, shape...)
    C = 5.0 .+ 25.0 .* rand(rng, shape...)
    # Poisson-like perturbation of the exact line integrals, so the solver has work to do
    channels = [h .+ Float32.(0.02 .* randn(rng, shape...)) for h in _toy_channels(basis, A, C)]

    # the package controls are the published ones
    published = NB04Oracle.nchannel_controls
    controls = BS.NChannelControls()
    for name in propertynames(published)
        @test getproperty(controls, name) == getproperty(published, name)
    end

    buffers() = (
        zeros(Float32, shape), zeros(Float32, shape), zeros(Float32, shape), zeros(Float32, shape),
        zeros(Float32, shape), zeros(UInt8, shape), zeros(Float32, shape), zeros(UInt8, shape),
        zeros(UInt8, shape),
    )
    theirs, ours = buffers(), buffers()
    NB04Oracle.nchannel_profile_tile!(
        theirs..., Tuple(channels),
        basis.Φ[1, 1, :, :], basis.μρ_I, basis.μρ_W, vec(basis.I0), vec(basis.μI_eff), vec(basis.μW_eff),
        basis.normal_II[1], basis.normal_IW[1], basis.normal_WW[1], published,
    )
    BS.nchannel_profile_tile!(
        ours..., Tuple(channels),
        basis.Φ, basis.μρ_I, basis.μρ_W, basis.I0, basis.μI_eff, basis.μW_eff,
        basis.normal_II, basis.normal_IW, basis.normal_WW, controls,
    )
    @test ours[1] == theirs[1]                                 # iodine, bit for bit
    @test ours[2] == theirs[2]                                 # water
    @test ours[3:5] == theirs[3:5]                             # Fisher information
    @test ours[6] == theirs[6] && ours[7] == theirs[7]         # quality flags, score
    @test ours[8] == theirs[8] && ours[9] == theirs[9]         # iteration counts

    expected = BS.total_expected_counts(ours[1], ours[2], basis.Φ, basis.μρ_I, basis.μρ_W)
    measured = BS.total_measured_counts(channels, basis.I0)
    @test expected == NB04Oracle.total_expected_counts(
        ours[1], ours[2], vec(sum(basis.Φ[1, 1, :, :]; dims = 2)), basis.μρ_I, basis.μρ_W,
    )
    @test measured ≈ NB04Oracle.total_measured_counts(channels, vec(basis.I0), 1) rtol = 1.0e-6
    for alpha2 in (24.635648571666497, 5.0)
        filtered = BS.tlbf_denoise(ours[1], ours[2], expected, measured; alpha2)
        reference = NB04Oracle.tlbf_filter_pair(ours[1], ours[2], expected, measured, alpha2)
        @test filtered.sino_iodine == reference[1] && filtered.sino_water == reference[2]
    end
end

@testset "vmi_pipeline end to end" begin
    # A centred 5 cm water disk carrying 5 mg/mL iodine. Its chords are analytic and the same in
    # every view, so the whole chain has a known answer: 1 g/mL water, 0.005 g/cm³ iodine, and
    # HU_E = 5·α_E with α_E the iodine-to-water mass-attenuation ratio.
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
    channels = _toy_channels(basis, spread(0.005 .* chord), spread(chord))

    out = BS.vmi_pipeline(;
        channels, basis, geom, reduce_rows = true, use_tlbf = true,
        matrix_size = (64, 64, 1), vmi_energies = (40, 70, 140), keep_sinograms = true,
    )
    centre = 28:37
    @test mean(out.images.water[centre, centre, 1]) ≈ 1.0 rtol = 0.02
    @test mean(out.images.iodine[centre, centre, 1]) ≈ 0.005 rtol = 0.03
    for (i, E) in pairs(out.energies)
        α = BS.compute_mass_μ_at_energy(BS.XA.Elements.Iodine, Float64(E)) /
            BS.compute_mass_μ_at_energy(BS.XA.Materials.water, Float64(E))
        @test mean(out.vmis[centre, centre, 1, i]) ≈ 5α atol = 15
    end
    @test size(out.vmis) == (64, 64, 1, 3) && size(out.sinograms.iodine) == (geom.n_cols, 1, geom.n_angles)
    @test out.settings.n_rows == geom.n_rows && out.settings.tlbf.radius == 2
    @test out.quality.frac_not_converged == 0

    # the two-measurement comparator runs through the same chain on the same acquisition
    cong = BS.vmi_pipeline(;
        channels, basis, geom, method = :cong, merge_groups = [1:2, 3:4], reduce_rows = true,
        matrix_size = (64, 64, 1), vmi_energies = (70,), use_acnr = false,
    )
    @test cong.settings.method === :cong && cong.settings.n_channels == 2
    @test mean(cong.images.water[centre, centre, 1]) ≈ 1.0 rtol = 0.02

    # Without reducing the rows, the whole chain runs on every row: T-LBF row by row, a
    # reconstruction onto a multi-slice grid, ACNR slice by slice, a VMI per slice. The disk is
    # z-invariant, so every slice must carry it.
    stack = BS.vmi_pipeline(;
        channels, basis, geom, use_tlbf = true, matrix_size = (64, 64, geom.n_rows),
        vmi_energies = (40, 70), use_acnr = true,
    )
    @test size(stack.vmis) == (64, 64, geom.n_rows, 2)
    @test size(stack.images.water) == (64, 64, geom.n_rows)
    for k in 1:geom.n_rows
        @test mean(stack.images.water[centre, centre, k]) ≈ 1.0 rtol = 0.03
    end
    @test stack.settings.n_rows === nothing && stack.settings.tlbf !== nothing
    # `rows` picks what to sum; without summing it would be silently ignored, so it is refused
    @test_throws ArgumentError BS.vmi_pipeline(;
        channels, basis, geom, rows = 1:2, matrix_size = (64, 64, 1),
    )
    # a row count that is neither one nor the geometry's cannot be reconstructed
    @test_throws DimensionMismatch BS.reconstruct_basis_slice(
        stack.images.water[:, 1:2, 1:1], geom, (64, 64, 1)
    )
    # …and an acquisition that already has one row needs no reduction to be filtered
    single_row = [h[:, 1:1, :] for h in channels]
    one_row = BS.vmi_pipeline(;
        channels = single_row, basis, geom, use_tlbf = true, matrix_size = (32, 32, 1),
        vmi_energies = (70,), use_acnr = false,
    )
    @test size(one_row.vmis) == (32, 32, 1, 1) && one_row.settings.tlbf !== nothing
    @test_throws ArgumentError BS.vmi_pipeline(; channels, basis, geom, method = :bogus, matrix_size = (32, 32, 1))
    # the grid is the caller's to state
    @test_throws UndefKeywordError BS.vmi_pipeline(; channels, basis, geom)
end

# Poisson counts on the toy channels: I0·exp(−h) drawn, then back to a log channel.
function _noisy_channels(channels, basis; seed = 11)
    rng = Random.MersenneTwister(seed)
    return map(eachindex(channels)) do k
        h = channels[k]
        I0 = Float64(basis.I0[1, 1, k])
        out = similar(h)
        for idx in eachindex(h)
            n = max(BS._poisson_sample(rng, I0 * exp(-Float64(h[idx]))), 1)
            out[idx] = Float32(-log(n / I0))
        end
        out
    end
end

@testset "HIR weights follow what the sinogram is" begin
    # A log-transmission sinogram of a water disk, as any scanner family produces one. The
    # default weighting is exp(−y); `:uniform` and an all-ones array must agree with each other
    # and differ from it; the guards refuse what cannot be meant.
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4,
        detector_cols = 128, detector_row_size = 1.0, detector_col_size = 1.5,
    )
    geom = BS.CTGeometry(scanner; n_angles = 120, fov_cm = 16.0, z_cm = 0.4)
    μ = zeros(Float32, 48, 48, 1)
    for j in 1:48, i in 1:48
        hypot(i - 24.5, j - 24.5) < 12 && (μ[i, j, 1] = 0.19f0)
    end
    sino = BS.dd_forward_project(μ, geom; volume_extent = geom.fov)
    recon(; kw...) = begin
        ws = BS.create_hir_recon_workspace(sino, geom, (48, 48, 1); strength = 60)
        copy(BS.reconstruct!(ws, sino, geom; kw...))
    end
    transmission = recon()
    uniform = recon(weights = :uniform)
    ones_map = recon(weights = ones(Float32, size(sino)))
    @test uniform ≈ ones_map rtol = 1.0e-4          # the same weighting, spelled two ways
    @test uniform != transmission                   # and not the Poisson heuristic
    @test mean(uniform[19:30, 19:30, 1]) ≈ 0.19 rtol = 0.05
    @test_throws ArgumentError recon(weights = :bogus)
    @test_throws DimensionMismatch recon(weights = ones(Float32, 4, 4, 4))
    @test_throws ArgumentError recon(
        weights = ones(Float32, size(sino)), air_reference = ones(Float32, geom.n_cols, geom.n_rows)
    )
end

@testset "the basis pair reconstructs iteratively, and soundly" begin
    # The same 5 cm water disk with 5 mg/mL iodine as above, now with Poisson counts. HIR must
    # return the same concentrations as FDK and be quieter in the flat interior for BOTH
    # materials — which the exp(−y) weighting could not do, because on the water sinogram (~10 g/cm²
    # here, 20-30 through a body) it switches the data term off and on the iodine one it is
    # uniform, so the two materials would have met two different reconstructors.
    scanner = BS.PCCTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4,
        detector_cols = 128, detector_row_size = 1.0, detector_col_size = 1.5,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
    )
    geom = BS.CTGeometry(scanner; n_angles = 180, fov_cm = 16.0, z_cm = 0.4)
    γ = ((1:geom.n_cols) .- (geom.n_cols + 1) / 2) .* geom.pixel_size ./ geom.SAD
    chord = @. 2 * sqrt(max(5.0^2 - (geom.SAD * sin(γ))^2, 0.0))
    spread(v) = repeat(reshape(v, :, 1, 1), 1, geom.n_rows, geom.n_angles)
    toy = _toy_response(total = 2.0e5)
    basis = BS.spectral_basis(energies = toy.E, response = toy.Φ, I0 = toy.I0)
    channels = _noisy_channels(_toy_channels(basis, spread(0.005 .* chord), spread(chord)), basis)
    centre = 28:37

    common = (;
        channels, basis, geom, reduce_rows = true, matrix_size = (64, 64, 1),
        vmi_energies = (40, 70, 140), use_acnr = false, keep_diagnostics = true,
        keep_sinograms = true,
    )
    fbp = BS.vmi_pipeline(; common...)
    hir = BS.vmi_pipeline(; common..., recon_method = :hir, hir_strength = 60)

    # what was done is recorded, and only the iterative run carries the dial
    @test fbp.settings.recon.method === :fbp && fbp.settings.recon.hir === nothing
    @test hir.settings.recon.method === :hir && hir.settings.recon.hir.strength == 60
    @test hir.settings.recon.hir.weights === :fisher && hir.settings.recon.hir.reference_kev == 70.0
    @test haskey(hir, :diagnostics) && all(>(0), hir.diagnostics.fisher.AA)

    # same object in the same units on both arms: water ≈ 1 g/cm³, iodine ≈ 0.005 g/cm³
    for out in (fbp, hir)
        @test mean(out.images.water[centre, centre, 1]) ≈ 1.0 rtol = 0.05
        @test mean(out.images.iodine[centre, centre, 1]) ≈ 0.005 rtol = 0.15
    end
    # and the iterative arm is quieter in the interior for BOTH materials — that is the whole
    # point, and the one thing a mis-weighted reconstructor could not deliver for water
    @test std(hir.images.water[centre, centre, 1]) < std(fbp.images.water[centre, centre, 1])
    @test std(hir.images.iodine[centre, centre, 1]) < std(fbp.images.iodine[centre, centre, 1])
    @test size(hir.vmis) == size(fbp.vmis) && all(isfinite, hir.vmis)

    # strength 0 is FDK through the one-row geometry: the same slice the repeated-row FDK gives,
    # to within the cone-angle difference between one row and four
    plain = BS.vmi_pipeline(; common..., recon_method = :hir, hir_strength = 0)
    @test mean(plain.images.water[centre, centre, 1]) ≈
        mean(fbp.images.water[centre, centre, 1]) rtol = 0.03

    # the two-measurement comparator has no Fisher information, so it is weighted uniformly
    cong = BS.vmi_pipeline(;
        common..., method = :cong, merge_groups = [1:2, 3:4], recon_method = :hir,
        keep_diagnostics = false,
    )
    @test cong.settings.recon.hir.weights === :uniform
    @test mean(cong.images.water[centre, centre, 1]) ≈ 1.0 rtol = 0.05

    @test_throws ArgumentError BS.vmi_pipeline(; common..., recon_method = :bogus)
    @test_throws ArgumentError BS.reconstruct_basis_slice(
        fbp.images.water, geom, (64, 64, 1); method = :bogus
    )
    # a reduced-row sinogram reconstructs one slice
    @test_throws ArgumentError BS.reconstruct_basis_slice(
        hir.sinograms.water, geom, (64, 64, 2); method = :hir
    )
end

@testset "the same iterative arm serves a dual-energy pair" begin
    # Two acquisitions instead of four bins, no row reduction: the multi-row path through the
    # original geometry, with the Fisher weights of a K = 2 decomposition. This is the DECT case;
    # nothing in the reconstruction knows or cares which detector produced the channels.
    scanner = BS.EICTScanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0, detector_rows = 4,
        detector_cols = 128, detector_row_size = 1.0, detector_col_size = 1.5,
    )
    geom = BS.CTGeometry(scanner; n_angles = 180, fov_cm = 16.0, z_cm = 0.4)
    γ = ((1:geom.n_cols) .- (geom.n_cols + 1) / 2) .* geom.pixel_size ./ geom.SAD
    chord = @. 2 * sqrt(max(5.0^2 - (geom.SAD * sin(γ))^2, 0.0))
    spread(v) = repeat(reshape(v, :, 1, 1), 1, geom.n_rows, geom.n_angles)
    toy = _toy_response(K = 2, total = 2.0e5)
    acquisitions = [
        (energies = toy.E, response = toy.Φ[:, :, :, k], I0_ray = toy.I0[:, :, k]) for k in 1:2
    ]
    basis = BS.spectral_basis_from_acquisitions(; acquisitions)
    @test basis.n_channels == 2
    channels = _noisy_channels(_toy_channels(basis, spread(0.005 .* chord), spread(chord)), basis)
    centre = 28:37

    common = (; channels, basis, geom, matrix_size = (64, 64, 1), vmi_energies = (70,), use_acnr = false)
    fbp = BS.vmi_pipeline(; common...)
    hir = BS.vmi_pipeline(; common..., recon_method = :hir)
    # rows were not reduced, which `settings.n_rows` records as `nothing`
    @test hir.settings.n_rows === nothing && hir.settings.recon.hir.weights === :fisher
    for out in (fbp, hir)
        @test mean(out.images.water[centre, centre, 1]) ≈ 1.0 rtol = 0.05
        @test mean(out.images.iodine[centre, centre, 1]) ≈ 0.005 rtol = 0.2
    end
    @test std(hir.images.water[centre, centre, 1]) < std(fbp.images.water[centre, centre, 1])
end
