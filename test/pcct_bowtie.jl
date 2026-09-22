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
    opts = BS.SimOptions(use_noise = false, use_focal_spot = false, use_scatter = false, seed = 1)
    recon = BS.ReconOptions(matrix_size = (64, 64, 4), fov_cm = 30.0, z_cm = 0.16)

    for bowtie in (:large_body, :none)
        scanner = _pcct(; bowtie)
        ws = BS.create_workspace(scanner, protocol, opts, recon, _air())
        res = BS.simulate!(ws, _air(), protocol, opts)
        geom = ws.geom
        n_cols = geom.n_cols; r = geom.n_rows ÷ 2 + 1
        # measured counts per ray: the bins are log-transmissions against the per-bin air response
        counts = res.raw_counts === nothing ?
            sum(res.I0_bins[:, b] .* exp.(-Array(res.pcct_sino.bins[b])[:, r, 1]) for b in 1:4) :
            sum(Array(res.raw_counts[b])[:, r, 1] for b in 1:4)
        centre = counts[n_cols ÷ 2]
        profile = counts ./ centre

        @testset "$bowtie: the fan profile is the bowtie's transmission" begin
            # the "air" phantom is 8 cm of real air (μ ≈ 2.5e-4 /cm): the central rays cross
            # it, the edge rays miss it, so the flat profile is 1 to within e^0.002
            if bowtie === :none
                @test all(isapprox.(profile, 1.0; atol = 3e-3))
            else
                bt = BS.resolve_bowtie_filter(bowtie)
                e, w = BS.resolve_source_spectrum_without_bowtie(opts, protocol; scanner)
                B = BS.compute_bowtie_attenuation_spectral(bt, geom, Float64.(e))
                expected(c) = sum(Float64.(w) .* Float64.(B[c, r, :])) / sum(Float64.(w) .* Float64.(B[n_cols ÷ 2, r, :]))
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
