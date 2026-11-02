# Dexel-to-dexel defects and the corrections calibrated for them from simulated calibration
# scans: the energy integrator's gain and offset, the photon counter's gain and threshold offsets
# (calibrate_flat_field → apply_flat_field!), and pile-up (calibrate_pileup → apply_pileup!). The
# residuals are measured against the ideal detector, a copy with the spreads at zero. CPU only;
# uses `_eict_cylinder`, `_EICT_TEST_SCANNER` and `_EICT_RECON` from eict_signal.jl.

_rms(x) = sqrt(sum(abs2, x) / length(x))
_noise_free(; kw...) = BS.SimOptions(;
    use_noise = false, use_scatter = false, use_heel_effect = false, use_focal_spot = false,
    use_optical_crosstalk = false, use_lag = false, seed = 1, kw...,
)

function _scan(scanner, phantom, protocol, opts)
    ws = BS.create_workspace(scanner, protocol, opts, _EICT_RECON, phantom)
    r = BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
    return scanner isa BS.EICTScanner ? Array(ws.sinogram) : (bins = [Array(b) for b in r.pcct_sino.bins], ws = ws)
end

@testset "energy integrator: gain and offset, corrected by two air scans" begin
    opts = _noise_free()
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 60)
    phantom = _eict_cylinder(; diameter_cm = 20.0)
    defective = BS.EICTScanner(_EICT_TEST_SCANNER; gain_spread = 0.02, offset_spread_keV = 2.0e4, defect_seed = 7)
    ideal = BS.EICTScanner(defective; gain_spread = 0.0, offset_spread_keV = 0.0)
    @test BS.EICTScanner(ideal; defect_seed = 1) == _EICT_TEST_SCANNER
    truth = _scan(ideal, phantom, protocol, opts)
    raw = _scan(defective, phantom, protocol, opts)
    cal = BS.calibrate_flat_field(defective, protocol, _EICT_RECON; sim_opts = opts)
    corrected = BS.apply_flat_field!(copy(raw), cal)
    @test _rms(raw .- truth) > 0.01                           # a 2 % gain spread rings
    @test _rms(corrected .- truth) < 1.0e-4                   # gain and offset are energy-blind: exact
    @test cal.gain ≈ 1 .+ 0.02 .* BS._dexel_pattern(7, 2, size(cal.gain)...)[:, :, 1] rtol = 1.0e-4
end

@testset "photon counter: gain and threshold offsets, an air flat field, a spectral residual" begin
    opts = _noise_free()
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 50.0, views = 60)
    phantom = _eict_cylinder(; diameter_cm = 20.0)
    base = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA; detector_rows = 2, detector_cols = 96, detector_col_size = 3.0, detector_row_size = 1.0,
        bowtie_filter = :none, pileup = false, scatter_correction = false,
    )
    defective = BS.PCCTScanner(base; gain_spread = 0.02, threshold_spread_keV = 0.35, defect_seed = 3)
    truth = _scan(base, phantom, protocol, opts).bins
    raw = _scan(defective, phantom, protocol, opts).bins
    cal = BS.calibrate_flat_field(defective, protocol, _EICT_RECON; sim_opts = opts)
    corrected = BS.apply_flat_field!([copy(b) for b in raw], cal)
    before = _rms(reduce(vcat, vec.(raw .- truth)))
    after = _rms(reduce(vcat, vec.(corrected .- truth)))
    @test before > 5.0e-3
    @test after < before / 3                                  # the air flat field removes most of it
    @test after > 0                                           # not all: the offsets weigh a hardened beam differently
    @info "photon-counting flat field, rms log error against the ideal detector" before after
    @test all(isfinite, cal.gain) && size(cal.gain, 3) == 4
end

@testset "pile-up: calibrated from air scans at rising currents" begin
    opts = _noise_free()
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 2.0, views = 30)
    phantom = _eict_cylinder(; diameter_cm = 20.0)
    base = BS.PCCTScanner(
        BS.NAEOTOM_ALPHA; detector_rows = 2, detector_cols = 96, detector_col_size = 3.0, detector_row_size = 1.0,
        bowtie_filter = :large_body, scatter_correction = false,
    )
    free = _scan(BS.PCCTScanner(base; pileup = false), phantom, protocol, opts).bins
    piled = _scan(base, phantom, protocol, opts)
    cal = BS.calibrate_pileup(base, protocol, _EICT_RECON; sim_opts = opts)
    corrected = BS.apply_pileup!([copy(b) for b in piled.bins], piled.ws.I0_cpu, cal, protocol)
    before = _rms(reduce(vcat, vec.(piled.bins .- free)))
    after = _rms(reduce(vcat, vec.(corrected .- free)))
    @test 0.01 < before < 0.3                                 # the loss is visible, the regime clinical
    @test after < before / 5
    @test issorted(cal.recorded) && all(cal.ratio[1, :] .>= 0.999)   # the lowest bin only loses counts
    @info "pile-up, rms log error against the pile-up-free scan" before after
end
