# The clinical scanners are documented constants of the two scanner structs, and a variant of any
# scanner or protocol is the copy constructor: every parameter copied, the keywords given replaced.

# The fields that differ between two scanners of the same family
_differing(a, b) = [f for f in (fieldnames(ScannerGeometry)..., fieldnames(typeof(a))[2:end]...) if getproperty(a, f) != getproperty(b, f)]

@testset "clinical scanners are constants of the scanner structs" begin
    for (s, family) in (
            (REVOLUTION_APEX, EICTScanner),
            (NAEOTOM_ALPHA, PCCTScanner),
            (SOMATOM_FORCE, EICTScanner),
            (SOMATOM_FORCE_B, EICTScanner),
            (SOMATOM_DEFINITION_FLASH, EICTScanner),
            (SOMATOM_DEFINITION_FLASH_B, EICTScanner),
        )
        @test s isa family{Float64}
        @test family(s) == s                       # a copy with nothing replaced is the same scanner
        @test hash(family(s)) == hash(s)
    end

    # the values the documentation's notebooks use
    @test REVOLUTION_APEX.detector_material == :lumex && REVOLUTION_APEX.detector_rows == 256 && REVOLUTION_APEX.electronic_noise == 3500.0
    @test NAEOTOM_ALPHA.detector_col_size ≈ 0.275 * 2 / (1113 / 610) && NAEOTOM_ALPHA.detector_cols == 1659
    @test NAEOTOM_ALPHA.energy_thresholds == [20.0, 35.0, 55.0, 70.0] && NAEOTOM_ALPHA.binning_factor == 2
    @test NAEOTOM_ALPHA.native_dexel_col_mm == 0.275 && NAEOTOM_ALPHA.native_dexel_row_mm == 0.322
    @test SOMATOM_FORCE.detector_material == :ufc && SOMATOM_FORCE.detector_cols == 920
    @test SOMATOM_DEFINITION_FLASH.detector_material == :ufc_flash && SOMATOM_DEFINITION_FLASH.flat_filter_thickness == 8.4

    # each dual-source scanner's tube/detector B is its A with detector B's smaller field
    @test _differing(SOMATOM_FORCE, SOMATOM_FORCE_B) == [:detector_cols, :scan_diameter]
    @test SOMATOM_FORCE_B.detector_cols == 640 && SOMATOM_FORCE_B.scan_diameter == 355.0
    @test _differing(SOMATOM_DEFINITION_FLASH, SOMATOM_DEFINITION_FLASH_B) == [:detector_cols, :scan_diameter]
    @test SOMATOM_DEFINITION_FLASH_B.detector_cols == 480 && SOMATOM_DEFINITION_FLASH_B.scan_diameter == 334.0
end

@testset "scanner variants: the copy constructor" begin
    # one keyword replaces one parameter and nothing else
    v = EICTScanner(SOMATOM_FORCE; detector_rows = 48)
    @test v.detector_rows == 48 && _differing(SOMATOM_FORCE, v) == [:detector_rows]
    v = EICTScanner(REVOLUTION_APEX; electronic_noise = 0.0, bowtie_filter = :none)
    @test Set(_differing(REVOLUTION_APEX, v)) == Set([:electronic_noise, :bowtie_filter])
    # the constructor's checks run on the variant
    @test_throws ArgumentError EICTScanner(SOMATOM_FORCE; energy_thresholds = [20.0])
    @test_throws ErrorException EICTScanner(SOMATOM_FORCE; detector_shape = :sphere)

    # photon counting: new thresholds set the bin count afresh; the native dexel is the sensor's own
    p = PCCTScanner(NAEOTOM_ALPHA; energy_thresholds = [20.0, 70.0])
    @test p.n_energy_bins == 2 && Set(_differing(NAEOTOM_ALPHA, p)) == Set([:energy_thresholds, :n_energy_bins])
    @test_throws ErrorException PCCTScanner(NAEOTOM_ALPHA; energy_thresholds = [20.0, 70.0], n_energy_bins = 4)
    p = PCCTScanner(NAEOTOM_ALPHA; binning_factor = 1)
    @test p.native_dexel_col_mm == NAEOTOM_ALPHA.native_dexel_col_mm && _differing(NAEOTOM_ALPHA, p) == [:binning_factor]
    @test_throws ErrorException PCCTScanner(NAEOTOM_ALPHA; noise_reduction = 2.0)

    # a variant never shares the thresholds with its original
    p = PCCTScanner(NAEOTOM_ALPHA)
    p.energy_thresholds[1] = 25.0
    @test NAEOTOM_ALPHA.energy_thresholds[1] == 20.0
end

@testset "protocol variants: the copy constructor" begin
    low = CTProtocol(
        kVp = 100, mA = 380.0, views = 1160, rotation_time = 0.5, collimation_mm = 4.8, anode_angle = 8,
        additional_filters = [("Ti", 0.9)]
    )
    @test CTProtocol(low) == low && hash(CTProtocol(low)) == hash(low)
    high = CTProtocol(low; kVp = 140, mA = 190.0, additional_filters = [("Ti", 0.9), ("Sn", 0.6)])
    @test high.kVp == 140 && high.mA == 190.0 && high.views == 1160 && high.collimation_mm == 4.8 && high.anode_angle == 8
    @test high.additional_filters == [("Ti", 0.9), ("Sn", 0.6)] && low.additional_filters == [("Ti", 0.9)]
    # a new mAs sets the current afresh; a variant keeps an axial scan axial
    @test CTProtocol(low; mAs = 100.0).mA == 200.0
    @test CTProtocol(low; rotation_time = 1.0).mA == 380.0 && CTProtocol(low).pitch === nothing
    @test_throws ArgumentError CTProtocol(low; pitch = -1.0)
    # a variant never shares the added filters with its original
    copied = CTProtocol(low)
    push!(copied.additional_filters, ("Sn", 0.6))
    @test low.additional_filters == [("Ti", 0.9)]
end
