# The bowtie across the fan, photon-counting path.
#
# A bowtie's purpose is the flux profile: it passes the full beam at the fan centre and a small
# fraction at the edge, so that peripheral rays — which cross less patient — are not over-exposed.
# The energy-integrating path applies it per column.  The photon-counting path folded only the
# centre-column spectral hardening into its one spectrum, so every ray got the central flux
# (measured on the semmd PCCT model: air-scan edge/centre counts 1.00 where the bowtie's
# transmission is 0.10).  This pins that the photon-counting simulation delivers the bowtie's own
# per-column transmission, that its per-bin air normalisation is per column too (air reads
# p ≡ 0 in every column), and that a scanner with no bowtie is flat.

using Test
using Statistics
import BasisSimulator as BS

function _pcct(; bowtie)
    return BS.PCCTScanner(
        source_to_isocenter = 610.0, source_to_detector = 1113.0,
        detector_rows = 12, detector_cols = 240,
        detector_row_size = 0.4, detector_col_size = 2.0,      # 48 cm fan at isocentre
        detector_material = :cdte, detector_depth = 1.6,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],
        pileup = false, pileup_correction = false, scatter_correction = false,
        bowtie_filter = bowtie, detector_col_offset = 0.0, detector_shape = :arc,
    )
end

_air() = BS.Phantom(zeros(UInt8, 8, 8, 8), [BS.XA.Materials.air], (1.0, 1.0, 1.0), (-4.0, -4.0, -4.0), (8.0, 8.0, 8.0))

@testset "photon-counting bowtie across the fan" begin
    protocol = BS.CTProtocol(kVp = 140, mA = 100.0, views = 60, rotation_time = 0.5, collimation_mm = 4.8, additional_filters = [("Ti", 0.9)])
    # the heel effect is a separate per-ray factor (and today on the wrong axis — see heel_effect.jl);
    # this test isolates the bowtie
    opts = BS.SimOptions(use_noise = false, use_focal_spot = false, use_scatter = false, use_heel_effect = false, seed = 1)
    recon = BS.ReconOptions(matrix_size = (64, 64, 4), fov_cm = 30.0, z_cm = 0.16)

    for bowtie in (:large_body, :none)
        scanner = _pcct(; bowtie)
        ws = BS.create_workspace(scanner, protocol, opts, recon, _air())
        res = BS.simulate!(ws, _air(), protocol, opts)
        geom = ws.geom
        n_cols = geom.n_cols; r = geom.n_rows ÷ 2 + 1
        # measured counts per ray: the bins are log-transmissions against the per-bin air response
        counts = sum(Array(res.raw_counts[b])[:, r, 1] for b in 1:4)
        centre = counts[n_cols ÷ 2]
        profile = counts ./ centre

        @testset "$bowtie: the fan profile is the bowtie's transmission" begin
            # the "air" phantom is 8 cm of real air (μ ≈ 2.5e-4 /cm): the central rays cross
            # it, the edge rays miss it, so the flat profile is 1 to within e^0.002
            if bowtie === :none
                @test all(isapprox.(profile, 1.0; atol = 3e-3))
            else
                # the detected profile: the applied response W[e, b] (spectrum × efficiency × bin
                # response) through the bowtie's own transmission table — the beam hardens
                # towards the edge, and the detector weights a harder spectrum differently, so
                # the detected ratio is not the incident one
                bt = BS.resolve_bowtie_filter(bowtie)
                B = BS.compute_bowtie_attenuation_spectral(bt, geom, Float64.(ws.energies))
                W = Float64.(Array(ws.W_matrix_gpu))[1:length(ws.energies), :]
                detected(c) = sum(W[e, b] * Float64(B[c, r, e]) for e in axes(W, 1), b in axes(W, 2))
                expected(c) = detected(c) / detected(n_cols ÷ 2)
                for c in (1, round(Int, 0.15n_cols), round(Int, 0.3n_cols), n_cols ÷ 2)
                    @test isapprox(profile[c], expected(c); rtol = 0.03)
                end
                @test profile[1] < 0.2                       # the edge sees a small fraction of the flux
            end
        end

        @testset "$bowtie: air reads zero in every column and bin" begin
            # to within the 8 cm of air the central rays cross (0.002)
            for b in 1:4
                p = Array(res.pcct_sino.bins[b])[:, r, 1]
                @test maximum(abs, p) < 3e-3
            end
        end
    end
end

@testset "per-ray counts: noise, pile-up, scatter with the bowtie" begin
    protocol = BS.CTProtocol(kVp = 140, mA = 100.0, views = 60, rotation_time = 0.5, collimation_mm = 4.8, additional_filters = [("Ti", 0.9)])
    recon = BS.ReconOptions(matrix_size = (64, 64, 4), fov_cm = 30.0, z_cm = 0.16)
    r = 6

    @testset "noise per ray follows the ray's own flux" begin
        scanner = _pcct(; bowtie = :large_body)
        opts = BS.SimOptions(use_noise = true, use_focal_spot = false, use_scatter = false, use_heel_effect = false, seed = 3)
        ws = BS.create_workspace(scanner, protocol, opts, recon, _air())
        res = BS.simulate!(ws, _air(), protocol, opts)
        n_cols = ws.geom.n_cols
        # relative noise of the total counts over the 60 views, centre vs edge: Poisson gives
        # σ/μ = 1/√N, and the edge sees a tenth of the flux
        tot(c) = [sum(Array(res.raw_counts[b])[c, r, v] for b in 1:4) for v in 1:60]
        rel(c) = std(tot(c)) / mean(tot(c))
        centre, edge = n_cols ÷ 2, 1
        @test isapprox(rel(centre), 1 / sqrt(mean(tot(centre))); rtol = 0.35)
        @test isapprox(rel(edge), 1 / sqrt(mean(tot(edge))); rtol = 0.35)
        @test rel(edge) > 2 * rel(centre)
    end

    @testset "pile-up at each ray's own count rate, and its correction" begin
        pileup_scanner(τ_ns) = BS.PCCTScanner(
            source_to_isocenter = 610.0, source_to_detector = 1113.0,
            detector_rows = 12, detector_cols = 240, detector_row_size = 0.4, detector_col_size = 2.0,
            detector_material = :cdte, detector_depth = 1.6, energy_thresholds = [20.0, 35.0, 55.0, 70.0],
            pileup = true, dead_time_ns = τ_ns, pileup_correction = false, scatter_correction = false,
            bowtie_filter = :large_body, detector_col_offset = 0.0, detector_shape = :arc,
        )
        opts = BS.SimOptions(use_noise = false, use_focal_spot = false, use_scatter = false, use_heel_effect = false, seed = 1)
        # a clinical regime: rate · τ ≈ 0.1 at the fan centre in air (the toy protocol's 60 views
        # make its per-view counts, and so its rate, far higher than a scanner's)
        rate_air = BS.create_workspace(pileup_scanner(1.0), protocol, opts, recon, _air()).pileup_rate_air
        scanner = pileup_scanner(0.1 / rate_air * 1.0e9)
        ws = BS.create_workspace(scanner, protocol, opts, recon, _air())
        @test size(ws.pileup_S, 3) == length(ws.pileup_rates) == 5
        @test ws.pileup_S[:, :, 1] == [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]      # no counts, no pile-up
        # loss grows with rate: column sums of S fall monotonically up the rate grid
        loss(k) = 1 - sum(ws.pileup_S[:, 1, k])
        @test loss(length(ws.pileup_rates)) > loss(1)
        res = BS.simulate!(ws, _air(), protocol, opts)
        n_cols = ws.geom.n_cols
        recorded(c) = sum(Array(res.raw_counts[b])[c, r, 1] for b in 1:4)
        truth(c) = sum(ws.I0_cpu[c, r, :])
        # the fan centre piles up at the air rate; the fan edge, at a tenth of the flux, loses
        # far less of its counts
        loss_centre = 1 - recorded(n_cols ÷ 2) / truth(n_cols ÷ 2)
        loss_edge = 1 - recorded(1) / truth(1)
        @test 0.03 < loss_centre < 0.3
        @test loss_edge < 0.3 * loss_centre       # a tenth of the flux, well under a third of the loss
        # the correction inverts it per ray
        bins = [copy(b) for b in res.pcct_sino.bins]
        BS.apply_pcct_pileup_correction!(bins, ws.I0, ws.pileup_S, ws.pileup_rates, ws.pileup_rate_air)
        for b in 1:4
            @test maximum(abs, Array(bins[b])[:, r, 1]) < 5e-3    # air again, within the air path
        end
    end

    @testset "scatter is injected against each ray's own flux" begin
        scanner = _pcct(; bowtie = :large_body)
        opts = BS.SimOptions(use_noise = false, use_focal_spot = false, use_scatter = true, use_heel_effect = false, seed = 1)
        ws = BS.create_workspace(scanner, protocol, opts, recon, _air())
        res = BS.simulate!(ws, _air(), protocol, opts)
        n_cols = ws.geom.n_cols
        # in air the scatter field is small and the bins must stay near zero everywhere — a
        # scatter term scaled by the central flux would show as a large negative p at the edge
        for b in 1:4
            p = Array(res.pcct_sino.bins[b])[:, r, 1]
            @test p[1] > -0.05 && p[n_cols ÷ 2] > -0.05
        end
    end
end

@testset "binned detector (bf = 2): I0, basis and air agree per ray" begin
    # the NAEOTOM-style native path: 2 × 2 dexels summed into each binned pixel
    protocol = BS.CTProtocol(kVp = 140, mA = 100.0, views = 60, rotation_time = 0.5, collimation_mm = 4.8, additional_filters = [("Ti", 0.9)])
    recon = BS.ReconOptions(matrix_size = (64, 64, 4), fov_cm = 30.0, z_cm = 0.16)
    opts = BS.SimOptions(use_noise = false, use_focal_spot = false, use_scatter = false, use_heel_effect = false, seed = 1)
    for bowtie in (:large_body, :none)
        scanner = BS.PCCTScanner(
            source_to_isocenter = 610.0, source_to_detector = 1113.0,
            detector_rows = 12, detector_cols = 240, detector_row_size = 0.4, detector_col_size = 2.0,
            detector_material = :cdte, detector_depth = 1.6, energy_thresholds = [20.0, 35.0, 55.0, 70.0],
            pileup = false, pileup_correction = false, scatter_correction = false,
            bowtie_filter = bowtie, detector_col_offset = 0.25, detector_shape = :arc,
            native_dexel_col_mm = 2.0 * 1113.0 / 610.0 / 2, native_dexel_row_mm = 0.4 * 1113.0 / 610.0 / 2, binning_factor = 2,
        )
        ws = BS.create_workspace(scanner, protocol, opts, recon, _air())
        @test ws.native_geom !== nothing && ws.I0_native !== nothing
        # the binned pixel's air count is PHYSICAL: no more than the photons it receives (the
        # detected fraction is below 1), and the sum of its four dexels
        incident = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights))
        c = ws.geom.n_cols ÷ 2; rr = ws.geom.n_rows ÷ 2 + 1
        @test sum(ws.I0_cpu[c, rr, :]) < incident
        @test sum(ws.I0_cpu[c, rr, :]) > 0.3 * incident
        @test isapprox(sum(ws.I0_cpu[c, rr, :]), sum(Array(ws.I0_native)[(2c - 1):(2c), (2rr - 1):(2rr), :]); rtol = 1e-5)
        # the basis the decomposition inverts is consistent with the air response it was built from
        basis = BS.spectral_basis(ws)
        @test basis.I0_relerr < 5e-5
        @test basis.ray_resolved == (bowtie !== :none)   # one response for every ray unless a bowtie varies it
        res = BS.simulate!(ws, _air(), protocol, opts)
        r = ws.geom.n_rows ÷ 2 + 1
        for b in 1:4
            @test maximum(abs, Array(res.pcct_sino.bins[b])[:, r, 1]) < 3e-3    # air, within the 8 cm of air
        end
    end
end
