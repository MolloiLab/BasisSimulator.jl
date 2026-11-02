# EICTScanner / PCCTScanner over a shared ScannerGeometry: the abstract `Scanner{T}` is the
# interface every consumer dispatches on; each family carries only its own detector model.
@testset "Scanner families: EICTScanner / PCCTScanner" begin
    e = EICTScanner(
        source_to_isocenter = 626.0, detector_rows = 8, detector_cols = 256, detector_material = :lumex,
        detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0
    )
    @test e isa Scanner{Float64} && e isa EICTScanner
    @test e.geometry isa ScannerGeometry{Float64}
    @test e.source_to_isocenter == 626.0 && e.detector_rows == 8 && e.detector_shape == :arc     # geometry reachable directly
    @test e.electronic_noise == 5.0 && e.detection_gain == 10.0
    @test :source_to_isocenter in propertynames(e) && :detection_gain in propertynames(e) && !(:energy_thresholds in propertynames(e))
    @test_throws FieldError e.energy_thresholds                                          # not part of an EICT detector
    @test_throws ArgumentError EICTScanner(energy_thresholds = [20.0, 40.0])             # PCCT kwarg on an EICT scanner
    p = PCCTScanner(
        source_to_isocenter = 540.0, detector_rows = 8, detector_cols = 64, detector_material = :CdTe,
        detector_depth = 1.6, energy_thresholds = [20.0, 35.0, 55.0, 70.0], dead_time_ns = 25.0
    )
    @test p isa Scanner{Float64} && p isa PCCTScanner
    @test length(p.energy_thresholds) == 4 && p.energy_thresholds == [20.0, 35.0, 55.0, 70.0] && p.dead_time_ns == 25.0
    @test p.pileup && p.scatter_correction && p.noise_reduction == 0.0    # detector-model defaults
    @test_throws FieldError p.detection_gain                                             # not part of a PCCT detector
    @test_throws ErrorException PCCTScanner()                                            # thresholds required
    @test_throws ErrorException PCCTScanner(energy_thresholds = [70.0, 20.0])
    @test_throws ErrorException PCCTScanner(energy_thresholds = [20.0], noise_reduction = 2.0)
    @test_throws ArgumentError PCCTScanner(energy_thresholds = [20.0], detection_gain = 3.0)
    @test PCCTScanner(energy_thresholds = [20.0, 40.0], pileup = false, noise_reduction = 0.7).noise_reduction == 0.7
    # geometry consumers dispatch on the abstract family
    g = CTGeometry(e; n_angles = 4, fov_cm = 20.0, z_cm = 1.0)
    @test g.n_cols == 256 && g.n_angles == 4
    @test CTGeometry(p; n_angles = 4, fov_cm = 20.0, z_cm = 1.0).n_cols == 64
    # SimOptions carries only the common physics toggles
    o = SimOptions()
    @test o.use_noise && o.use_scatter && o.use_focal_spot && o.use_lag && !o.use_optical_crosstalk && o.seed == 42
    @test !hasproperty(o, :use_pcct_pileup) && !hasproperty(o, :pcct_noise_reduction)
    @test_throws MethodError SimOptions(fidelity = :pcct)
end

# What both families share has one name, one meaning and one default: the scatter kernels and
# their correction, the dexels' gain spread and the seed of their fixed pattern.
@testset "Scanner families: shared detector fields are aligned" begin
    e = EICTScanner()
    p = PCCTScanner(energy_thresholds = [20.0])
    shared = (:scatter_kernels, :scatter_correction, :gain_spread, :defect_seed)
    @test all(f -> hasfield(EICTScanner, f) && hasfield(PCCTScanner, f), shared)
    @test all(f -> fieldtype(EICTScanner{Float64}, f) == fieldtype(PCCTScanner{Float64}, f), shared)
    @test all(f -> getfield(e, f) == getfield(p, f), shared)
    @test e.scatter_correction && p.scatter_correction                    # real scanners correct scatter
    @test e.scatter_kernels == p.scatter_kernels == ""
    @test SimOptions().use_scatter                                         # and scatter is on
    # every clinical constant corrects its scatter and names its kernels
    for s in (
            REVOLUTION_APEX, SOMATOM_FORCE, SOMATOM_FORCE_B, SOMATOM_DEFINITION_FLASH,
            SOMATOM_DEFINITION_FLASH_B, NAEOTOM_ALPHA, NAEOTOM_ALPHA_UHR,
        )
        @test s.scatter_correction && !isempty(s.scatter_kernels)
    end
    # the toggle is a keyword of both constructors and survives the copy constructor
    @test !EICTScanner(SOMATOM_FORCE; scatter_correction = false).scatter_correction
    @test !PCCTScanner(NAEOTOM_ALPHA; scatter_correction = false).scatter_correction
    @test EICTScanner(EICTScanner(SOMATOM_FORCE; scatter_correction = false); detector_rows = 8).scatter_correction == false
    # the defect spreads are non-negative on both
    @test_throws ArgumentError EICTScanner(gain_spread = -0.01)
    @test_throws ArgumentError PCCTScanner(energy_thresholds = [20.0], gain_spread = -0.01)
    @test_throws ArgumentError PCCTScanner(energy_thresholds = [20.0], threshold_spread_keV = -0.1)
end
