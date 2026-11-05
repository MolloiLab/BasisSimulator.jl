# Dexel-to-dexel defects and the corrections calibrated for them from simulated calibration
# scans: the energy integrator's gain and offset, the photon counter's gain and threshold offsets
# (calibrate_flat_field → apply_flat_field!), and pile-up (calibrate_pileup → apply_pileup!). The
# residuals are measured against the ideal detector, a copy with the spreads at zero. The scans and
# the calibrations run where `to_gpu` puts the phantom (api.jl: a GPU when the test environment has
# one); uses `_eict_cylinder`, `_EICT_TEST_SCANNER` and `_EICT_RECON` from eict_signal.jl.

_rms(x) = sqrt(sum(abs2, x) / length(x))
_noise_free(; kw...) = BS.SimOptions(;
    use_noise = false, use_scatter = false, use_heel_effect = false, use_focal_spot = false,
    seed = 1, kw...,
)

_on_backend(p) = BS.Phantom(to_gpu(p.mask), p.materials, p.voxel_size, p.origin, p.extent)

function _scan(scanner, phantom, protocol, opts)
    ws = BS.create_workspace(scanner, protocol, opts, _EICT_RECON, phantom)
    r = BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
    return scanner isa BS.EICTScanner ? Array(ws.sinogram) : (bins = [Array(b) for b in r.pcct_sino.bins], ws = ws)
end

@testset "energy integrator: gain and offset, corrected by two air scans" begin
    opts = _noise_free()
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 60)
    phantom = _on_backend(_eict_cylinder(; diameter_cm = 20.0))
    defective = BS.EICTScanner(_EICT_TEST_SCANNER; gain_spread = 0.02, offset_spread_keV = 2.0e4, defect_seed = 7)
    ideal = BS.EICTScanner(defective; gain_spread = 0.0, offset_spread_keV = 0.0)
    @test BS.EICTScanner(ideal; defect_seed = 1) == _EICT_TEST_SCANNER
    truth = _scan(ideal, phantom, protocol, opts)
    raw = _scan(defective, phantom, protocol, opts)
    cal = BS.calibrate_flat_field(defective, protocol, _EICT_RECON, phantom; sim_opts = opts)
    corrected = BS.apply_flat_field!(copy(raw), cal)
    @test _rms(raw .- truth) > 0.01                           # a 2 % gain spread rings
    @test _rms(corrected .- truth) < 1.0e-4                   # gain and offset are energy-blind: exact
    @test cal.gain ≈ 1 .+ 0.02 .* BS._dexel_pattern(7, 2, size(cal.gain)...)[:, :, 1] rtol = 1.0e-4
end

@testset "photon counter: gain and threshold offsets, a flat field through air and water slabs" begin
    opts = _noise_free()
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 50.0, views = 60)
    phantom = _on_backend(_eict_cylinder(; diameter_cm = 20.0))
    base = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA; detector_rows = 2, detector_cols = 96, detector_col_size = 3.0, detector_row_size = 1.0,
        bowtie_filter = :none, pileup = false, scatter_correction = false,
        gain_spread = 0.0, threshold_spread_keV = 0.0,              # the ideal detector, for the truth
    )
    defective = BS.PCCTScanner(base; gain_spread = 0.02, threshold_spread_keV = 0.35, defect_seed = 3)
    truth = _scan(base, phantom, protocol, opts).bins
    raw = _scan(defective, phantom, protocol, opts).bins
    cal = BS.calibrate_flat_field(defective, protocol, _EICT_RECON, phantom; sim_opts = opts)
    corrected = BS.apply_flat_field!([copy(b) for b in raw], cal)
    before = _rms(reduce(vcat, vec.(raw .- truth)))
    after = _rms(reduce(vcat, vec.(corrected .- truth)))
    # the air gains alone leave the hardened part: the offsets weigh a hardened beam differently
    air_only = BS.apply_flat_field!([copy(b) for b in raw], BS.FlatFieldCalibration(cal.gain, nothing))
    residual_air = _rms(reduce(vcat, vec.(air_only .- truth)))
    @test before > 5.0e-3
    # the slabs take the gains through the hardening; what is left is at the cylinder's edges, where
    # this toy detector's 3 mm pixels hold dexels of very different line integrals (the NAEOTOM
    # Alpha's 0.37 mm pixels leave under 0.6 HU in a Gammex image)
    @test after < residual_air / 5
    @test after < 2.0e-3
    @info "photon-counting flat field, rms log error against the ideal detector" before residual_air after
    @test all(isfinite, cal.gain) && size(cal.gain, 3) == 4
    @test size(cal.hardening.r) == (size(cal.gain)..., length(BS._FLAT_FIELD_SLABS_CM))
    # the ideal detector has no hardening to calibrate
    @test BS.calibrate_flat_field(base, protocol, _EICT_RECON, phantom; sim_opts = opts).hardening === nothing
end

@testset "pile-up: calibrated from air and water scans at rising currents" begin
    opts = _noise_free()
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 2.0, views = 30)
    phantom = _on_backend(_eict_cylinder(; diameter_cm = 20.0))
    base = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA; detector_rows = 2, detector_cols = 96, detector_col_size = 3.0, detector_row_size = 1.0,
        bowtie_filter = :large_body, scatter_correction = false,
        gain_spread = 0.0, threshold_spread_keV = 0.0,              # pile-up alone
    )
    free = _scan(BS.PCCTScanner(base; pileup = false), phantom, protocol, opts).bins
    piled = _scan(base, phantom, protocol, opts)
    cal = BS.calibrate_pileup(base, protocol, _EICT_RECON, phantom; sim_opts = opts)
    corrected = BS.apply_pileup!([copy(b) for b in piled.bins], piled.ws.I0_cpu, cal, protocol)
    before = _rms(reduce(vcat, vec.(piled.bins .- free)))
    after = _rms(reduce(vcat, vec.(corrected .- free)))
    @test 0.01 < before < 0.3                                 # the loss is visible, the regime clinical
    @test after < before / 5
    @test issorted(cal.rates) && issorted(cal.hardness)
    @test all(cal.C[:, :, 1, m] ≈ Matrix(1.0I, 4, 4) for m in eachindex(cal.hardness))   # no counts, no pile-up
    @info "pile-up, rms log error against the pile-up-free scan" before after
end

@testset "the calibrations' two views match the protocol's full views" begin
    # An air scan does not change with the rotation, so the calibrations scan two views, each at
    # one angle; the reference scans every view of the protocol at five sub-views (`_flat_field`
    # and `_pileup` scan exactly the protocol and options they are given).
    full = _noise_free(; use_focal_spot = true, view_samples = 5)
    phantom = _on_backend(_eict_cylinder(; diameter_cm = 20.0))

    eict = BS.EICTScanner(
        _EICT_TEST_SCANNER;
        gain_spread = 0.02,
        offset_spread_keV = 2.0e4,
        defect_seed = 7,
        use_lag = true,
    )
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 120)
    few = BS.calibrate_flat_field(eict, protocol, _EICT_RECON, phantom; sim_opts = full)
    ref = BS._flat_field(eict, protocol, _EICT_RECON, full, phantom)
    @test maximum(abs.(few.gain .- ref.gain)) < 2.0e-5
    @test maximum(abs.(few.offset .- ref.offset)) < 2.0e-5

    pcct = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA;
        detector_rows = 2,
        detector_cols = 96,
        detector_col_size = 3.0,
        detector_row_size = 1.0,
        scatter_correction = false,
    )
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 2.0, views = 120)
    few = BS.calibrate_flat_field(pcct, protocol, _EICT_RECON, phantom; sim_opts = full)
    ref = BS._flat_field(pcct, protocol, _EICT_RECON, full, phantom)
    @test maximum(abs.(few.gain .- ref.gain)) < 2.0e-5

    # pile-up: the corrections the two calibrations make to one scan
    piled = _scan(pcct, phantom, protocol, full)
    currents = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1 / sqrt(2), 1, sqrt(2)]   # calibrate_pileup's own
    corrected(cal) = BS.apply_pileup!([copy(b) for b in piled.bins], piled.ws.I0_cpu, cal, protocol)
    a = corrected(BS.calibrate_pileup(pcct, protocol, _EICT_RECON, phantom; sim_opts = full))
    b = corrected(BS._pileup(pcct, protocol, _EICT_RECON, full, phantom, currents))
    # the calibration's water cylinders are voxelised, so two angles see them a little differently
    # from 120 × 5 sub-views: 2e-5, a twentieth of what the correction leaves against the truth
    @test maximum(maximum(abs.(x .- y)) for (x, y) in zip(a, b)) < 3.0e-5
end

@testset "the calibrations run on the scan phantom's device" begin
    # A phantom on the CPU calibrates on the CPU, one on the GPU on the GPU; the calibration
    # scans air on that device, never the phantom, and both devices agree.
    opts = _noise_free()
    cpu = _eict_cylinder(; diameter_cm = 20.0)
    @test BS._air_phantom(cpu).mask isa Array
    eict = BS.EICTScanner(_EICT_TEST_SCANNER; gain_spread = 0.02, offset_spread_keV = 2.0e4, defect_seed = 7)
    pcct = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA;
        detector_rows = 2,
        detector_cols = 96,
        detector_col_size = 3.0,
        detector_row_size = 1.0,
        scatter_correction = false,
    )
    p_eict = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 60)
    p_pcct = BS.CTProtocol(; kVp = 120.0, mAs = 2.0, views = 30)
    calibrations(phantom) = (
        eict = BS.calibrate_flat_field(eict, p_eict, _EICT_RECON, phantom; sim_opts = opts),
        pcct = BS.calibrate_flat_field(pcct, p_pcct, _EICT_RECON, phantom; sim_opts = opts),
        pileup = BS.calibrate_pileup(pcct, p_pcct, _EICT_RECON, phantom; sim_opts = opts),
    )
    on_cpu = calibrations(cpu)
    @test on_cpu.eict.gain isa Matrix{Float64} && on_cpu.pcct.gain isa Array{Float64, 3}
    if !HAS_GPU
        @info "Skipping the GPU calibration: no GPU backend"
    else
        gpu = _on_backend(cpu)
        air = BS._air_phantom(gpu).mask
        @test !(air isa Array) && typeof(air) == typeof(similar(gpu.mask, UInt8, 4, 4, 2))
        on_gpu = calibrations(gpu)
        @test maximum(abs.(on_gpu.eict.gain .- on_cpu.eict.gain)) < 2.0e-5
        @test maximum(abs.(on_gpu.eict.offset .- on_cpu.eict.offset)) < 2.0e-5
        @test maximum(abs.(on_gpu.pcct.gain .- on_cpu.pcct.gain)) < 2.0e-5
        # the pile-up curves, as the correction they make to one scan
        piled = _scan(pcct, cpu, p_pcct, opts)
        corrected(cal) = BS.apply_pileup!([copy(b) for b in piled.bins], piled.ws.I0_cpu, cal, p_pcct)
        a, b = corrected(on_gpu.pileup), corrected(on_cpu.pileup)
        # the fit takes the scans' Float32 rounding on the GPU through a least-squares solve
        @test maximum(maximum(abs.(x .- y)) for (x, y) in zip(a, b)) < 3.0e-5
    end
end
