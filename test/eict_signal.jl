# The energy-integrating detector's signal model, checked against its analytic values: the signal
# moments m₁ and m₂, the air scan's mean and variance, NEQ = ff·N·DQE(0), the compound-Poisson
# dispersion behind water, the electronic noise, beam-hardening correction of water, and a
# noise-free spectral decomposition (scatter: scatter.jl). CPU only.

using Statistics: var

# A water cylinder (diameter in cm; 0 is an empty field) on a 0.25 cm grid.
function _eict_cylinder(; diameter_cm, n = 160, nz = 2, insert_cm = 0.0)
    vox = (0.25, 0.25, 0.25)
    mask = zeros(UInt8, n, n, nz)
    c = (n + 1) / 2
    for k in 1:nz, j in 1:n, i in 1:n
        r = hypot(i - c, j - c) * vox[1]
        r <= diameter_cm / 2 && (mask[i, j, k] = 0x01)
    end
    mats = Dict{Int, Any}(0 => BS.XA.Materials.air, 1 => BS.XA.Materials.water)
    return BS.Phantom(mask, mats, vox)
end

_eict_opts(seed; noise = true, kwargs...) = BS.SimOptions(;
    use_noise = noise,
    use_scatter = false,
    use_heel_effect = false,
    use_focal_spot = false,
    seed,
    kwargs...,
)

# A small arc scanner: 256 columns of 1.5 mm and two rows, the Force's detector.
const _EICT_TEST_SCANNER = BS.EICTScanner(
    BS.SOMATOM_FORCE;
    detector_rows = 2,
    detector_cols = 256,
    detector_col_size = 1.5,
    electronic_noise = 0.0,
    use_lag = false,
)
const _EICT_RECON = BS.ReconOptions(; matrix_size = (128, 128, 1), fov_cm = 36.0, z_cm = 0.5)

# Noise-free and two noisy draws of one acquisition; the variance of a dexel is half that of the
# difference of the draws, which cancels every deterministic pattern.
function _eict_draws(scanner, phantom; protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 90))
    ws = BS.create_eict_workspace(scanner, protocol, _eict_opts(1), _EICT_RECON, phantom)
    BS.simulate!(ws, phantom, protocol, _eict_opts(1; noise = false); report_dose = false)
    clean = Array(ws.sinogram)
    dispersion = Array(ws.dispersion)
    BS.simulate!(ws, phantom, protocol, _eict_opts(1); report_dose = false)
    a = Array(ws.sinogram)
    BS.simulate!(ws, phantom, protocol, _eict_opts(2); report_dose = false)
    b = Array(ws.sinogram)
    I0 = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights))
    return (; ws, protocol, clean, dispersion, a, b, I0, ff = 0.9 * 0.9)
end

@testset "eid_signal_moments: m₁ = E·efficiency, m₂ = m₁²/dqe0" begin
    for (material, table) in (
            (:lumex, BS.GEMSTONE_MC_EFFICIENCY_LUT),
            (:ufc, BS.UFC_MC_EFFICIENCY_LUT),
            (:ufc_flash, BS.UFC_FLASH_MC_EFFICIENCY_LUT),
        )
        model = BS.DetectorEfficiency(material, table.thickness_mm, BS.MC_LUT)
        m1, m2 = BS.eid_signal_moments(model, table.energies)
        @test m1 ≈ table.energies .* table.efficiency
        @test m2 ≈ m1 .^ 2 ./ table.dqe0
        # absorption and the Swank spread: the mean square exceeds the square of the mean
        @test all(0 .< table.dqe0 .<= 1) && all(m2 .>= m1 .^ 2)
        # linear between the table's energies
        m1h, _ = BS.eid_signal_moments(model, [60.5])
        @test m1h[1] ≈ 60.5 * (table.efficiency[60] + table.efficiency[61]) / 2
        # a table is made at one depth
        @test_throws ArgumentError BS.DetectorEfficiency(material, table.thickness_mm + 1, BS.MC_LUT)
        bl1, bl2 = BS.eid_signal_moments(
            BS.DetectorEfficiency(material, table.thickness_mm + 1, BS.BEER_LAMBERT), [40.0, 80.0]
        )
        @test bl2 ≈ [40.0, 80.0] .* bl1
    end
    @test BS.eid_signal_moments(nothing, [50.0]) == ([50.0], [2500.0])
    @test_throws ArgumentError BS.DetectorEfficiency(:unobtainium, 1.0, BS.MC_LUT)
    # the Flash's crystal is the depth of its table
    @test BS.SOMATOM_DEFINITION_FLASH.detector_depth == BS.UFC_FLASH_MC_EFFICIENCY_LUT.thickness_mm
end

@testset "air scan: zero line integral, variance S₂/(I₀·ff·S₁²), η once" begin
    # no bowtie: the calibration that used to read −log η_eff
    for bowtie in (:none, :large_body)
        scanner = BS.EICTScanner(_EICT_TEST_SCANNER; bowtie_filter = bowtie)
        d = _eict_draws(scanner, _eict_cylinder(; diameter_cm = 0.0, n = 8))
        @test abs(sum(d.clean) / length(d.clean)) < 2.0e-4     # 2 cm of air at most
        air1 = Array(d.ws.air_reference)
        w̄ = d.ws.weights ./ sum(d.ws.weights)
        E_ref = sum(w̄ .* d.ws.m1)
        B = bowtie === :none ? ones(size(air1)..., length(w̄)) :
            Float64.(Array(d.ws.bowtie_spectral)[:, :, 1:length(w̄)])
        air2 = dropdims(sum(B .* reshape(w̄ .* d.ws.m2 ./ E_ref^2, 1, 1, :); dims = 3); dims = 3)
        predicted = air2 ./ (d.I0 * d.ff .* air1 .^ 2)               # per dexel
        measured = dropdims(var(d.a .- d.b; dims = 3); dims = 3) ./ 2
        ratio = sum(measured) / sum(predicted)
        @test ratio ≈ 1 atol = 0.02                              # η twice would read ≈ 1/η̄ ≈ 1.1
        @test d.dispersion[:, :, 1] ≈ air2 ./ air1 rtol = 1.0e-4
    end
end

@testset "NEQ in air = ff·N·DQE(0): the SOMATOM Force at 120 kV" begin
    scanner = BS.EICTScanner(_EICT_TEST_SCANNER; bowtie_filter = :none)
    d = _eict_draws(scanner, _eict_cylinder(; diameter_cm = 0.0, n = 8))
    w̄ = d.ws.weights ./ sum(d.ws.weights)
    dqe = sum(w̄ .* d.ws.m1)^2 / sum(w̄ .* d.ws.m2)                # the spectrum's DQE(0)
    @test 0.8 < dqe < 0.87
    neq = 1 / (sum(var(d.a .- d.b; dims = 3)) / 2 / (size(d.a, 1) * size(d.a, 2)))
    @test neq / (d.ff * d.I0) ≈ dqe rtol = 0.02
end

@testset "behind 20 and 30 cm of water: the dispersion rises, the variance follows it" begin
    scanner = BS.EICTScanner(_EICT_TEST_SCANNER; bowtie_filter = :none)
    D_centre = Float64[]
    for diameter in (0.0, 20.0, 30.0)
        d = _eict_draws(scanner, _eict_cylinder(; diameter_cm = diameter))
        c = (size(d.clean, 1) ÷ 2 - 3):(size(d.clean, 1) ÷ 2 + 4)
        λ = d.I0 * d.ff .* exp.(-Float64.(d.clean[c, :, :]))
        predicted = d.dispersion[c, :, :] ./ λ
        measured = (d.a[c, :, :] .- d.b[c, :, :]) .^ 2 ./ 2
        @test sum(measured) / sum(predicted) ≈ 1 atol = 0.06
        push!(D_centre, sum(d.dispersion[c, :, :]) / length(predicted))
    end
    @test D_centre[1] < D_centre[2] < D_centre[3]
    @test D_centre[3] / D_centre[1] > 1.15                       # hardening raises E²/E
end

@testset "electronic noise adds σ², in keV through the gain" begin
    gain = 10.0
    electrons = 1.5e5                                             # big enough to matter at 30 cm
    scanner = BS.EICTScanner(
        _EICT_TEST_SCANNER; bowtie_filter = :none, electronic_noise = electrons,
        detection_gain = gain
    )
    d = _eict_draws(scanner, _eict_cylinder(; diameter_cm = 30.0))
    w̄ = d.ws.weights ./ sum(d.ws.weights)
    σ = electrons / gain / sum(w̄ .* d.ws.m1)                     # in E_ref units
    @test d.ws.σ_electronic ≈ σ rtol = 1.0e-5
    c = (size(d.clean, 1) ÷ 2 - 3):(size(d.clean, 1) ÷ 2 + 4)
    λ = d.I0 * d.ff .* exp.(-Float64.(d.clean[c, :, :]))
    quantum = d.dispersion[c, :, :] ./ λ
    total = quantum .+ σ^2 ./ λ .^ 2                               # (D·λ + σ²) / λ²
    @test sum(σ^2 ./ λ .^ 2) / sum(quantum) > 0.5                # the electronic share matters
    measured = (d.a[c, :, :] .- d.b[c, :, :]) .^ 2 ./ 2
    @test sum(measured) / sum(total) ≈ 1 atol = 0.06
end

@testset "water BHC: 0 ± 1 HU, cupping < 2 HU" begin
    # Against the same scan at one energy (70 keV, no hardening), so what is measured is the
    # signal model and its correction, not the reconstruction's own grid (≈ −2 to −3 HU here).
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 360)
    opts = _eict_opts(1; noise = false)
    for diameter in (20.0, 30.0)
        phantom = _eict_cylinder(; diameter_cm = diameter)
        n = _EICT_RECON.matrix_size[1]
        r = [hypot(i - (n + 1) / 2, j - (n + 1) / 2) * _EICT_RECON.fov_cm / n for i in 1:n, j in 1:n]
        centre = r .< 3.0
        rim = (r .> 0.6 * diameter / 2) .& (r .< 0.8 * diameter / 2)
        inside = r .< 0.8 * diameter / 2
        roi(hu, m) = sum(hu[:, :, 1][m]) / count(m)
        hu = map((:mono, :poly)) do beam
            spectrum_override = beam === :mono ? ([70.0], [1.0e6]) : nothing
            ws = BS.create_eict_workspace(
                _EICT_TEST_SCANNER, protocol, opts, _EICT_RECON, phantom;
                spectrum_override
            )
            BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
            sino, μ_water = if beam === :mono
                ws.sinogram, BS.compute_mass_μ_at_energy(BS.XA.Materials.water, 70.0)
            else
                bhc = BS.calibrate_bhc_water(opts, protocol; scanner = _EICT_TEST_SCANNER, geom = ws.geom)
                BS.apply_bhc_water(ws.sinogram, bhc), bhc.μ_water_ref
            end
            recon = BS.create_fdk_recon_workspace(sino, ws.geom, _EICT_RECON.matrix_size)
            BS.to_hounsfield(Array(BS.reconstruct!(recon, sino, ws.geom)); μ_water)
        end
        mono, poly = hu
        @test abs(roi(poly, inside) - roi(mono, inside)) < 1.0
        @test abs((roi(poly, rim) - roi(poly, centre)) - (roi(mono, rim) - roi(mono, centre))) < 2.0
    end
end

@testset "spectral basis: noise-free dual-kVp decomposition returns the water" begin
    opts = _eict_opts(1; noise = false)
    phantom = _eict_cylinder(; diameter_cm = 20.0)
    acquisitions = map((80.0, 140.0)) do kVp
        protocol = BS.CTProtocol(; kVp, mAs = 200.0, views = 12)
        ws = BS.create_eict_workspace(_EICT_TEST_SCANNER, protocol, opts, _EICT_RECON, phantom)
        BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
        (sinogram = Array(ws.sinogram), basis = BS.spectral_basis(ws, protocol))
    end
    basis = BS.spectral_basis_from_acquisitions(; acquisitions = [a.basis for a in acquisitions])
    @test basis.n_channels == 2
    out = BS.decompose_nchannel(; channels = [a.sinogram for a in acquisitions], basis)
    # the truth: the water's own path, from a monoenergetic projection at 70 keV
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 12)
    ws70 = BS.create_eict_workspace(
        _EICT_TEST_SCANNER, protocol, opts, _EICT_RECON, phantom;
        spectrum_override = ([70.0], [1.0e6])
    )
    BS.simulate!(ws70, phantom, protocol, opts; report_dose = false)
    μρ70 = BS.compute_mass_μ_at_energy(BS.XA.Materials.water, 70.0)
    truth = Array(ws70.sinogram) ./ μρ70                            # g/cm²
    c = (size(truth, 1) ÷ 2 - 20):(size(truth, 1) ÷ 2 + 20)
    @test maximum(abs.(out.sino_water[c, :, :] .- truth[c, :, :])) < 0.02 * maximum(truth)
    @test maximum(abs.(out.sino_iodine[c, :, :])) < 1.0e-3
end

@testset "the scintillator's switches on EICTScanner have their v0.16.0 effect" begin
    # v0.16.0 set them on SimOptions; v0.16.1 moved them onto the scanner. A noisy CPU scan per
    # switch, pinned to what v0.16.0 returned with the same switch on SimOptions (v0.16.1 measured
    # bit-identical).
    base = BS.EICTScanner(
        BS.SOMATOM_FORCE;
        detector_rows = 2,
        detector_cols = 256,
        detector_col_size = 1.5,
    )
    protocol = BS.CTProtocol(; kVp = 120.0, mAs = 200.0, views = 24)
    recon = BS.ReconOptions(; matrix_size = (64, 64, 1), fov_cm = 36.0, z_cm = 0.5)
    opts = BS.SimOptions(; use_scatter = false, seed = 3)
    phantom = _eict_cylinder(; diameter_cm = 20.0, n = 96)
    v0160 = [
        (NamedTuple(), 20156.499474268134, 3.9915876f0, 0.0015148066f0),
        ((use_lag = false,), 20156.652922001464, 3.9915876f0, 0.0015065091f0),
        ((use_optical_crosstalk = true,), 20154.322786129305, 3.991612f0, 0.0014678875f0),
        ((use_fill_factor = false,), 20156.494888265355, 3.9917183f0, 0.0014770802f0),
        ((use_detector_efficiency = false,), 20075.22057167095, 3.9753575f0, 0.0014953463f0),
        ((detector_efficiency_mode = :beer_lambert,), 20126.74811192186, 3.9856682f0, 0.001500659f0),
    ]
    for (switch, total, centre, edge) in v0160
        scanner = BS.EICTScanner(base; switch...)
        ws = BS.create_eict_workspace(scanner, protocol, opts, recon, phantom)
        BS.simulate!(ws, phantom, protocol, opts; report_dose = false)
        s = Array(ws.sinogram)
        @test sum(Float64, s) ≈ total rtol = 1.0e-7
        @test s[128, 1, 1] ≈ centre rtol = 1.0e-5
        @test s[40, 2, 13] ≈ edge rtol = 1.0e-4
    end
end
