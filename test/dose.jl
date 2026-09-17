# CTDI100 / CTDIw / CTDIvol / DLP from the simulated beam. CPU only.

_dose_scanner(; kwargs...) = BS.EICTScanner(;
    source_to_isocenter = 625.6, source_to_detector = 1100.0, detector_rows = 256,
    detector_cols = 834, detector_row_size = 0.625, detector_col_size = 0.6,
    target_angle = 10.0, flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large, kwargs...,
)
_dose_protocol(; kVp = 120, mA = 350.0, kwargs...) = BS.CTProtocol(;
    kVp, mA, views = 1000, rotation_time = 1.0, collimation_mm = 15.0,
    additional_filters = [("Al", 4.5)], kwargs...,
)

@testset "the beam in absolute units" begin
    @test BS.muen_rho_air(60.0) ≈ 0.03041
    @test BS.muen_rho_air(sqrt(60.0 * 80.0)) ≈ sqrt(0.03041 * 0.02407) rtol = 1.0e-12   # log-log midpoint
    bare = BS.EICTScanner(
        source_to_isocenter = 1000.0, source_to_detector = 1500.0, detector_rows = 16,
        detector_cols = 64, flat_filter_material = :aluminum, flat_filter_thickness = 7.0,
        bowtie_filter = :none,
    )
    kerma(kVp; filters = Tuple{String, Float64}[]) = BS.air_kerma_free_in_air(
        BS.dose_source(bare, BS.CTProtocol(; kVp, mA = 100.0, additional_filters = filters))
    )
    # IPEM-78, 10° tungsten anode behind 7 mm Al, at 1 m: 83 µGy/mAs at 120 kVp
    @test 1000 * kerma(120) ≈ 83.0 rtol = 0.01
    @test issorted([kerma(k) for k in (80, 100, 120, 140)])
    @test kerma(120; filters = [("Al", 2.0)]) < kerma(120)

    src = BS.dose_source(_dose_scanner(), _dose_protocol())
    @test src.nominal_collimation_mm == 15.0 && src.SAD_mm == 625.6
    @test src.filters == [("aluminum", 2.5), ("Al", 4.5)]
    @test BS.air_kerma_free_in_air(src; fan_angle = 0.2) < BS.air_kerma_free_in_air(src)   # bowtie
    # no collimation given: the whole detector, at isocentre
    open_beam = BS.dose_source(_dose_scanner(), BS.CTProtocol(kVp = 120, mA = 100.0))
    @test open_beam.nominal_collimation_mm == 256 * 0.625
    # a spectrum override is in detector units and is carried to isocentre by inverse square
    flat = BS.dose_source(_dose_scanner(), _dose_protocol(); spectrum = ([60.0], [1.0e5]))
    @test flat.phi_iso ≈ [1.0e5 * (1100.0 / 625.6)^2]
end

@testset "CTDI100 in the PMMA phantoms" begin
    src = BS.dose_source(_dose_scanner(), _dose_protocol())
    spectrum_before = copy(src.phi_iso)
    body = BS.ctdi100(src; phantom = :body32, n_histories = 400_000)
    head = BS.ctdi100(src; phantom = :head16, n_histories = 400_000)
    @test body === BS.ctdi100(src; phantom = :body32, n_histories = 400_000)      # cached
    # …and the cache key has to separate the questions that have different answers
    @test BS.ctdi100(src; phantom = :body32, beam_width_mm = 18.0, n_histories = 400_000).ctdi_w !=
        body.ctdi_w
    @test BS.ctdi100(src; phantom = :body32, n_histories = 800_000).ctdi_w != body.ctdi_w
    @test head.ctdi_w != body.ctdi_w
    # the transport samples the spectrum, it does not consume it
    @test spectrum_before == src.phi_iso
    @test 1.7 < body.periphery / body.center < 2.6
    @test 1.05 < head.periphery / head.center < 1.3
    @test 1.8 < head.ctdi_w / body.ctdi_w < 2.3
    @test body.rel_stat_uncertainty < 0.02
    # scatter dominates the centre of the body phantom and matters little at its edge
    @test 5 < body.center / body.primary_center - 1 < 9
    @test 0.9 < body.periphery / body.primary_periphery - 1 < 1.6
    # pinned: GE-Revolution-like beam, 9.5 mm Al on the central ray, beam = N·T
    @test 100 * body.ctdi_w ≈ 6.82 rtol = 0.03
    @test 100 * head.ctdi_w ≈ 13.47 rtol = 0.03
    # a different seed is an independent estimate of the same number
    other = BS.ctdi100(src; phantom = :body32, n_histories = 400_000, seed = 2)
    @test other.ctdi_w != body.ctdi_w
    @test other.ctdi_w ≈ body.ctdi_w rtol = 0.03
    @test_throws ArgumentError BS.ctdi100(src; phantom = :torso)
end

@testset "CTDIvol and DLP" begin
    scanner = _dose_scanner()
    n = 400_000
    axial = BS.compute_dose(scanner, _dose_protocol(); n_histories = n)
    @test axial isa BS.DoseReport && axial.pitch === nothing
    @test axial.table_increment_mm == 15.0 && axial.scan_length_cm ≈ 1.5     # one rotation
    @test axial.ctdi_vol_mGy ≈ axial.ctdi_w_mGy_per_100mAs * 3.5
    @test axial.dlp_mGy_cm ≈ axial.ctdi_vol_mGy * 1.5
    @test BS.compute_ctdi_vol(scanner, _dose_protocol(); n_histories = n) == axial.ctdi_vol_mGy
    @test BS.compute_dlp(axial) == axial.dlp_mGy_cm

    # linear in tube current and in rotation time
    @test BS.compute_dose(scanner, _dose_protocol(mA = 700.0); n_histories = n).ctdi_vol_mGy ≈
        2 * axial.ctdi_vol_mGy
    @test BS.compute_dose(scanner, _dose_protocol(rotation_time = 0.5); n_histories = n).ctdi_vol_mGy ≈
        axial.ctdi_vol_mGy / 2
    # kVp trend of the GE console table: CTDIvol per mAs rises about 4.6× from 80 to 140 kVp
    per_mAs(kVp) = BS.compute_dose(scanner, _dose_protocol(; kVp); n_histories = n).ctdi_w_mGy_per_100mAs
    @test 4.1 < per_mAs(140) / per_mAs(80) < 5.2

    # helical: CTDIvol ∝ 1/pitch; the dose-length product over a fixed travel does not depend on it
    helical(pitch, n_rot) = BS.compute_dose(
        scanner, _dose_protocol(; pitch, n_rotations = n_rot); n_histories = n,
    )
    one, half = helical(1.0, 4.0), helical(0.5, 8.0)
    @test one.ctdi_vol_mGy ≈ axial.ctdi_vol_mGy && half.ctdi_vol_mGy ≈ 2 * one.ctdi_vol_mGy
    @test one.scan_length_cm ≈ 6.0 && half.scan_length_cm ≈ 6.0
    @test half.dlp_mGy_cm ≈ 2 * one.dlp_mGy_cm          # twice the rotations over the same 6 cm
    # axial with gaps: half the dose index, the same energy spread over twice the length
    gapped = BS.compute_dose(scanner, _dose_protocol(); table_increment_mm = 30.0, n_histories = n)
    @test gapped.ctdi_vol_mGy ≈ axial.ctdi_vol_mGy / 2 && gapped.dlp_mGy_cm ≈ axial.dlp_mGy_cm

    # over-beaming, a second tube and a calibration factor are plain multipliers
    @test BS.compute_dose(scanner, _dose_protocol(); overbeam_mm = 3.0, n_histories = n).ctdi_vol_mGy ≈
        axial.ctdi_vol_mGy * 18 / 15 rtol = 0.03
    @test BS.compute_dose(scanner, _dose_protocol(); n_tubes = 2, n_histories = n).ctdi_vol_mGy ≈
        2 * axial.ctdi_vol_mGy
    @test BS.compute_dose(scanner, _dose_protocol(); dose_calibration = 0.9, n_histories = n).ctdi_vol_mGy ≈
        0.9 * axial.ctdi_vol_mGy

    # N·T > 40 mm: the IEC reference-beam rule, so CTDIw per mAs equals the 20 mm value
    wide = BS.compute_dose(scanner, _dose_protocol(collimation_mm = 160.0); n_histories = n)
    narrow = BS.compute_dose(scanner, _dose_protocol(collimation_mm = 20.0); n_histories = n)
    @test wide.wide_beam_reference_mm == 20.0 && narrow.wide_beam_reference_mm === nothing
    @test wide.ctdi_w_mGy_per_100mAs ≈ narrow.ctdi_w_mGy_per_100mAs

    @test_throws ArgumentError BS.compute_dose(scanner, _dose_protocol(); table_increment_mm = 0.0)
    @test occursin("CTDIvol", sprint(show, MIME("text/plain"), axial))
end
