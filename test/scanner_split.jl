# EICTScanner / PCCTScanner over a shared ScannerGeometry: the abstract `Scanner{T}` is the
# interface every consumer dispatches on; each family carries only its own detector model.
@testset "Scanner families: EICTScanner / PCCTScanner" begin
    e = EICTScanner(source_to_isocenter = 626.0, detector_rows = 8, detector_cols = 256, detector_material = :lumex,
        detector_depth = 3.0, electronic_noise = 5.0, detection_gain = 10.0)
    @test e isa Scanner{Float64} && e isa EICTScanner
    @test e.geometry isa ScannerGeometry{Float64}
    @test e.source_to_isocenter == 626.0 && e.detector_rows == 8 && e.detector_shape == :arc     # geometry reachable directly
    @test e.electronic_noise == 5.0 && e.detection_gain == 10.0
    @test :source_to_isocenter in propertynames(e) && :detection_gain in propertynames(e) && !(:energy_thresholds in propertynames(e))
    @test_throws FieldError e.energy_thresholds                                          # not part of an EICT detector
    @test_throws ArgumentError EICTScanner(energy_thresholds = [20.0, 40.0])             # PCCT kwarg on an EICT scanner
    p = PCCTScanner(source_to_isocenter = 540.0, detector_rows = 8, detector_cols = 64, detector_material = :CdTe,
        detector_depth = 1.6, energy_thresholds = [20.0, 35.0, 55.0, 70.0], dead_time_ns = 25.0)
    @test p isa Scanner{Float64} && p isa PCCTScanner
    @test p.n_energy_bins == 4 && p.energy_thresholds == [20.0, 35.0, 55.0, 70.0] && p.dead_time_ns == 25.0
    @test p.pileup && !p.pileup_correction && !p.scatter_correction && p.noise_reduction == 0.0   # detector-model defaults
    @test p.native_dexel_col_mm ≈ 1.0 * (950.0 / 540.0)                                # inferred: pixel × magnification / binning
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
    # the functional entry point dispatches on the family
    BSF = BasisSimulator.Functional
    @test hasmethod(BSF.pipeline, Tuple{Phantom, EICTScanner, Any, Any, Any})
    @test hasmethod(BSF.pipeline, Tuple{Phantom, PCCTScanner, Any, Any, Any})
    @test !hasmethod(BSF.pipeline, Tuple{Phantom, Scanner, Any, Any, Any})
end
