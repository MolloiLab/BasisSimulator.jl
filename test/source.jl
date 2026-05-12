# Tests for src/source/ — spectrum loaders + filtration, protocol struct +
# dose helpers + transformers, focal-spot model + blur kernel, heel effect
# (sinogram-domain apply + spectral-domain compute).
#
# Coverage policy: every exported symbol from src/source/{spectrum,protocol,
# focal_spot,heel_effect}.jl is exercised below (bowtie_filter.jl is covered
# by test/bowtie.jl).  Both LIVE pipeline paths (full behavioral tests) and
# scanner-config-vocabulary symbols (real behavioral tests, narrower scope)
# get the same standard of coverage — no smoke tests.

# -----------------------------------------------------------------------------
# spectrum.jl
# -----------------------------------------------------------------------------
@testset "load_spectrum — xspect + xcist sources" begin
    @testset "xspect default (target=7°)" begin
        for kVp in (80, 100, 120, 140)
            e, w = BS.load_spectrum(kVp; source = :xspect)
            @test length(e) == length(w)
            @test length(e) > 0
            @test all(>=(0), w)        # photon counts non-negative
            @test issorted(e)          # energy grid monotonic
            @test maximum(e) < kVp + 1.0  # max E bounded by kVp
        end
    end

    @testset "xspect target=10°" begin
        e, w = BS.load_spectrum(120; source = :xspect, target_angle = 10.0)
        @test length(e) > 0
        @test maximum(e) < 121.0
    end

    @testset "xcist source" begin
        for kVp in (80, 100, 120, 140)
            e, w = BS.load_spectrum(kVp; source = :xcist)
            @test length(e) > 0
            @test all(>=(0), w)
            @test issorted(e)
        end
    end

    @testset "validation errors" begin
        @test_throws ErrorException BS.load_spectrum(60)                       # bad kVp
        @test_throws ErrorException BS.load_spectrum(120; target_angle = 5.0)  # bad angle
        @test_throws ErrorException BS.load_spectrum(120; source = :bogus)     # bad source
        @test_throws ErrorException BS.load_spectrum(90; source = :xcist)      # 90 kVp not in xcist
    end
end

@testset "load_spectrum_unfiltered — IPEM 78" begin
    for angle in (8, 10), kVp in (80, 100, 120, 140)
        e, w = BS.load_spectrum_unfiltered(kVp; anode_angle = angle)
        @test length(e) == length(w)
        @test all(>=(0), w)
        @test all(>(0), w)            # zero-flux tails dropped
        @test issorted(e)
        @test maximum(e) <= kVp + 0.5  # max E bounded by kVp
        # First non-zero bin should be > 0 — characteristic line / bremsstrahlung onset.
        @test minimum(e) > 0
    end

    @test_throws ErrorException BS.load_spectrum_unfiltered(50)
    @test_throws ErrorException BS.load_spectrum_unfiltered(120; anode_angle = 7)
end

@testset "spectrum_mean_energy" begin
    @testset "delta spectrum returns its energy" begin
        @test BS.spectrum_mean_energy([60.0], [1.0]) ≈ 60.0
        @test BS.spectrum_mean_energy([60.0], [100.0]) ≈ 60.0
    end

    @testset "weighted mean of two energies" begin
        # Equal weights → arithmetic mean.
        @test BS.spectrum_mean_energy([50.0, 90.0], [1.0, 1.0]) ≈ 70.0
        # 3:1 weighting biases toward heavier.
        @test BS.spectrum_mean_energy([50.0, 90.0], [3.0, 1.0]) ≈ 60.0
    end

    @testset "real spectra: filtered > raw mean E (filtration hardens)" begin
        e, w = BS.load_spectrum_unfiltered(120)
        ē_raw = BS.spectrum_mean_energy(e, w)
        @test 30.0 < ē_raw < 60.0   # unfiltered tungsten 120 kVp ~43 keV
        # After Al-2mm filtration the mean E should rise (filtration hardens).
        _, w_f = BS.filter_spectrum(e, w; filters = [("Al", 2.0)])
        ē_filt = BS.spectrum_mean_energy(e, w_f)
        @test ē_filt > ē_raw
    end

    @test_throws ErrorException BS.spectrum_mean_energy([60.0], [0.0])  # zero fluence
end

@testset "downsample_spectrum" begin
    e, w = BS.load_spectrum_unfiltered(120)
    n_orig = length(e)

    @testset "n_bins ≥ original → copy passthrough" begin
        e_d, w_d = BS.downsample_spectrum(e, w, n_orig + 100)
        @test e_d == e
        @test w_d == w
        @test e_d !== e   # fresh allocation
    end

    @testset "halving bin count preserves total fluence" begin
        e_d, w_d = BS.downsample_spectrum(e, w, n_orig ÷ 2)
        @test length(e_d) == n_orig ÷ 2
        @test length(w_d) == n_orig ÷ 2
        # Sum of weights is conserved (downsample sums weights within each bin).
        @test sum(w_d) ≈ sum(w) atol = 1.0e-6 * sum(w)
        # Mean energy preserved to within a bin width.
        bin_width = (maximum(e) - minimum(e)) / n_orig
        @test abs(BS.spectrum_mean_energy(e_d, w_d) - BS.spectrum_mean_energy(e, w)) < 2 * bin_width
    end

    @testset "n_bins = 1 collapses to total fluence" begin
        e_d, w_d = BS.downsample_spectrum(e, w, 1)
        @test length(e_d) == 1
        @test length(w_d) == 1
        @test w_d[1] ≈ sum(w)  atol = 1.0e-6 * sum(w)
    end
end

@testset "filter_spectrum — Beer-Lambert filtration" begin
    e, w = BS.load_spectrum_unfiltered(120)

    @testset "empty filters = identity (with default SDD)" begin
        e_f, w_f = BS.filter_spectrum(e, w; filters = Tuple{String, Float64}[])
        @test e_f == e
        @test w_f ≈ w
    end

    @testset "single Al-2mm filter is exact Beer-Lambert" begin
        e_f, w_f = BS.filter_spectrum(e, w; filters = [("Al", 2.0)])
        @test e_f == e
        # Expected: w[i] × exp(-μ_Al(E_i) × 0.2 cm).  Use the internal helper.
        for i in 1:length(e)
            μ = BasisSimulator.get_filter_mu("Al", e[i])
            @test w_f[i] ≈ w[i] * exp(-μ * 0.2)  atol = 1.0e-12 * w[i]
        end
    end

    @testset "multi-filter composition multiplies attenuations" begin
        e_a, w_a = BS.filter_spectrum(e, w; filters = [("Al", 1.0)])
        e_ab, w_ab = BS.filter_spectrum(e_a, w_a; filters = [("Cu", 0.2)])
        e_both, w_both = BS.filter_spectrum(e, w; filters = [("Al", 1.0), ("Cu", 0.2)])
        @test maximum(abs.(w_ab .- w_both)) < 1.0e-10 * maximum(w)
    end

    @testset "SDD ≠ 750 mm applies inverse-square-law" begin
        _, w_at_1500 = BS.filter_spectrum(e, w; filters = Tuple{String, Float64}[], sdd_mm = 1500.0)
        @test w_at_1500 ≈ 0.25 .* w   # (750/1500)² = 1/4
    end

    @testset "filtered spectrum is harder than raw (mean E increases)" begin
        e_raw = e; w_raw = w
        _, w_filt = BS.filter_spectrum(e, w; filters = [("Al", 5.0)])  # 5 mm Al
        @test BS.spectrum_mean_energy(e, w_filt) > BS.spectrum_mean_energy(e_raw, w_raw)
    end
end

# -----------------------------------------------------------------------------
# protocol.jl
# -----------------------------------------------------------------------------
@testset "CTProtocol — defaults + kwarg propagation" begin
    p = BS.CTProtocol(; mA = 200.0)
    @test p.mA == 200.0
    @test p.kVp == 120.0
    @test p.views == 984
    @test p.rotation_time == 1.0
    @test p.n_rotations == 1.0
    @test p.collimation_mm === nothing
    @test p.anode_angle == 10
    @test isempty(p.additional_filters)
end

@testset "CTProtocol — mA vs mAs exclusivity" begin
    @testset "mAs path: mA = mAs / rotation_time" begin
        p = BS.CTProtocol(; mAs = 100.0, rotation_time = 0.5)
        @test p.mA == 200.0   # 100 / 0.5
    end

    @testset "mA explicit wins when both supplied (mA takes priority)" begin
        p = BS.CTProtocol(; mA = 300.0, mAs = 100.0)
        @test p.mA == 300.0
    end

    @testset "neither → default 200 mA" begin
        p = BS.CTProtocol()
        @test p.mA == 200.0
    end
end

@testset "CTProtocol — additional_filters carries through" begin
    p = BS.CTProtocol(; mA = 200.0, additional_filters = [("Al", 4.5), ("Cu", 0.2)])
    @test length(p.additional_filters) == 2
    @test p.additional_filters[1] == ("Al", 4.5)
end

@testset "validate_protocol" begin
    scanner = BS.Scanner(detector_rows = 64, detector_row_size = 1.0)

    @testset "valid protocol passes" begin
        p = BS.CTProtocol(; mA = 200.0, kVp = 120.0, views = 984, rotation_time = 0.5)
        valid, _ = BS.validate_protocol(p, scanner)
        @test valid
    end

    @testset "bad kVp errors" begin
        p_low = BS.CTProtocol(; mA = 200.0, kVp = 50.0)
        p_high = BS.CTProtocol(; mA = 200.0, kVp = 200.0)
        @test !BS.validate_protocol(p_low, scanner)[1]
        @test !BS.validate_protocol(p_high, scanner)[1]
    end

    @testset "bad mA errors" begin
        p_low = BS.CTProtocol(; mA = 5.0)
        p_high = BS.CTProtocol(; mA = 2000.0)
        @test !BS.validate_protocol(p_low, scanner)[1]
        @test !BS.validate_protocol(p_high, scanner)[1]
    end

    @testset "collimation exceeds scanner max" begin
        # 64 × 1.0 = 64 mm max; ask for 100.
        p = BS.CTProtocol(; mA = 200.0, collimation_mm = 100.0)
        valid, msgs = BS.validate_protocol(p, scanner)
        @test !valid
        @test any(occursin("collimation_mm", m) for m in msgs)
    end

    @testset "negative collimation" begin
        p = BS.CTProtocol(; mA = 200.0, collimation_mm = -1.0)
        @test !BS.validate_protocol(p, scanner)[1]
    end
end

@testset "compute_ctdi_vol + compute_dlp" begin
    p = BS.CTProtocol(; mA = 200.0, kVp = 120.0, rotation_time = 1.0)
    base_ctdi = BS.compute_ctdi_vol(p)
    @test base_ctdi > 0

    @testset "linear in mAs" begin
        p2 = BS.CTProtocol(; mA = 400.0, kVp = 120.0, rotation_time = 1.0)
        @test BS.compute_ctdi_vol(p2) ≈ 2 * base_ctdi
    end

    @testset "kVp^2.5 scaling" begin
        p3 = BS.CTProtocol(; mA = 200.0, kVp = 80.0, rotation_time = 1.0)
        @test BS.compute_ctdi_vol(p3) ≈ base_ctdi * (80.0 / 120.0)^2.5
    end

    @testset "diameter^-2 scaling" begin
        ctdi_160 = BS.compute_ctdi_vol(p; phantom_diameter = 160.0)
        @test ctdi_160 ≈ base_ctdi * (320.0 / 160.0)^2
    end

    @testset "DLP = CTDI × length × n_rotations" begin
        dlp = BS.compute_dlp(p, 30.0)
        @test dlp ≈ base_ctdi * 30.0 * p.n_rotations
        # n_rotations propagates.
        p_helical = BS.CTProtocol(; mA = 200.0, n_rotations = 5.0)
        @test BS.compute_dlp(p_helical, 30.0) ≈
            BS.compute_ctdi_vol(p_helical) * 30.0 * 5.0
    end
end

@testset "dose_report — NamedTuple shape + consistency" begin
    s = BS.Scanner()
    p = BS.CTProtocol(; mA = 200.0, kVp = 120.0, views = 100, rotation_time = 0.5)
    g = BS.CTGeometry(s; n_angles = 100, fov_cm = 35.0, z_cm = 5.0)
    report = BS.dose_report(p, g, 1.0e8; scan_length_cm = 30.0)
    @test report.ctdi_vol == BS.compute_ctdi_vol(p)
    @test report.dlp == BS.compute_dlp(p, 30.0)
    @test report.mAs == p.mA * p.rotation_time
    @test report.kVp == p.kVp
    @test report.views == p.views
    @test report.I0_per_view > 0
    @test report.total_photons > 0
end

@testset "constant_dose_protocol" begin
    base = BS.CTProtocol(; mA = 200.0, views = 1000)
    new = BS.constant_dose_protocol(base, 500)
    @test new.mA == 200.0           # mA preserved
    @test new.views == 500          # views changed
    @test new.kVp == base.kVp        # other fields preserved
    @test new.rotation_time == base.rotation_time
    @test new.collimation_mm == base.collimation_mm
end

@testset "constant_noise_protocol" begin
    base = BS.CTProtocol(; mA = 200.0, views = 1000)
    # Half the views → double the mA per view to keep per-view noise constant.
    halved = BS.constant_noise_protocol(base, 500)
    @test halved.mA == 100.0   # 200 × (500/1000)
    @test halved.views == 500
    # Doubled views → halve mA.
    doubled = BS.constant_noise_protocol(base, 2000)
    @test doubled.mA == 400.0
end

# -----------------------------------------------------------------------------
# focal_spot.jl
# -----------------------------------------------------------------------------
@testset "FocalSpot struct + factory size ordering" begin
    @test BS.focal_spot_small().width == 0.5
    @test BS.focal_spot_medium().width == 0.8
    @test BS.focal_spot_large().width == 1.2
    @test BS.focal_spot_point().width == 0.0
    @test BS.focal_spot_small().width < BS.focal_spot_medium().width < BS.focal_spot_large().width
    @test BS.focal_spot_medium().shape == :gaussian
end

@testset "compute_focal_spot_blur_fwhm — geometric magnification" begin
    s = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_col_size = 1.0, detector_row_size = 1.0
    )
    g = BS.CTGeometry(s; n_angles = 8, fov_cm = 35.0, z_cm = 5.0)
    fs = BS.focal_spot_medium()   # 0.8 × 0.8 mm

    @testset "at isocenter (SOD = SAD)" begin
        bw, bl = BS.compute_focal_spot_blur_fwhm(fs, g, g.SAD)
        # Expected: fs_size_cm × (SDD/SOD - 1) = 0.08 × (108/54 - 1) = 0.08 × 1 = 0.08 cm
        # Then / pixel_size_det (= 0.1 × SDD/SAD = 0.2 cm) → 0.4 pixels.
        @test bw ≈ 0.4  atol = 1.0e-10
        @test bl ≈ 0.4  atol = 1.0e-10
    end

    @testset "closer SOD → more blur" begin
        bw_iso, _ = BS.compute_focal_spot_blur_fwhm(fs, g, g.SAD)
        bw_near, _ = BS.compute_focal_spot_blur_fwhm(fs, g, g.SAD * 0.7)
        @test bw_near > bw_iso
    end

    @testset "farther SOD → less blur" begin
        bw_iso, _ = BS.compute_focal_spot_blur_fwhm(fs, g, g.SAD)
        bw_far, _ = BS.compute_focal_spot_blur_fwhm(fs, g, g.SAD * 1.3)
        @test bw_far < bw_iso
    end

    @testset "point source → zero blur" begin
        bw, bl = BS.compute_focal_spot_blur_fwhm(BS.focal_spot_point(), g, g.SAD)
        @test bw == 0.0
        @test bl == 0.0
    end
end

@testset "create_focal_spot_kernel_spatial" begin
    fs = BS.focal_spot_medium()
    @testset "Gaussian kernel: sum=1, peaked at center, symmetric" begin
        kernel = BS.create_focal_spot_kernel_spatial(fs, (3.0, 3.0))
        @test sum(kernel) ≈ 1.0
        n = size(kernel, 1)
        center = (n + 1) ÷ 2 + (n % 2 == 0 ? 0 : 0)
        @test n == size(kernel, 2)   # square
        # Peak at center.
        @test kernel[center, center] == maximum(kernel)
        # Symmetric across both axes.
        for i in 1:size(kernel, 1), j in 1:size(kernel, 2)
            @test kernel[i, j] ≈ kernel[end + 1 - i, j]    atol = 1.0e-12
            @test kernel[i, j] ≈ kernel[i, end + 1 - j]    atol = 1.0e-12
        end
    end

    @testset "size capped at MAX_FOCAL_SPOT_KERNEL_SIZE" begin
        # Ask for a huge blur — kernel should not exceed 15.
        kernel = BS.create_focal_spot_kernel_spatial(fs, (50.0, 50.0))
        @test size(kernel, 1) <= 15
        @test size(kernel, 1) == size(kernel, 2)
    end

    @testset "zero FWHM → delta-like kernel" begin
        kernel = BS.create_focal_spot_kernel_spatial(BS.focal_spot_point(), (0.0, 0.0))
        @test sum(kernel) ≈ 1.0
        center = (size(kernel, 1) + 1) ÷ 2
        @test kernel[center, center] == 1.0   # all mass at center
    end

    @testset "uniform shape produces flat-top kernel" begin
        fs_u = BS.FocalSpot(1.0, 1.0, :uniform, 3)
        kernel = BS.create_focal_spot_kernel_spatial(fs_u, (3.0, 3.0))
        @test sum(kernel) ≈ 1.0
        # Uniform kernel has constant values in its support — many entries equal.
        center = (size(kernel, 1) + 1) ÷ 2
        @test kernel[center - 1, center - 1] ≈ kernel[center + 1, center + 1]
    end
end

@testset "apply_focal_spot_blur! — physical behavior" begin
    s = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_cols = 64, detector_rows = 4
    )
    g = BS.CTGeometry(s; n_angles = 4, fov_cm = 35.0, z_cm = 1.0)

    @testset "point source = identity (sub-pixel blur skipped)" begin
        sino = Float32.(rand(MersenneTwister(0), 64, 4, 4))
        sino_in = copy(sino)
        BS.apply_focal_spot_blur!(sino, BS.focal_spot_point(), g)
        @test sino == sino_in
    end

    @testset "delta sinogram → kernel-shaped output (mass conserved)" begin
        sino = zeros(Float32, 64, 4, 4)
        sino[32, 2, 2] = 1.0f0
        BS.apply_focal_spot_blur!(sino, BS.focal_spot_large(), g)
        @test sum(sino) ≈ 1.0  atol = 1.0e-4   # convolution preserves mass
        # Peak still near the original location (no shift).
        max_idx = argmax(sino[:, 2, 2])
        @test abs(max_idx - 32) ≤ 1
    end

    @testset "blur on uniform sinogram leaves uniform sinogram (DC preserved)" begin
        sino = fill(0.5f0, 64, 4, 4)
        BS.apply_focal_spot_blur!(sino, BS.focal_spot_large(), g)
        # Interior pixels (away from boundary clamps) stay at 0.5.
        @test all(abs.(sino[10:54, 2, 2] .- 0.5f0) .< 1.0f-5)
    end
end

@testset "apply_focal_spot_blur (allocating wrapper)" begin
    s = BS.Scanner(detector_cols = 32, detector_rows = 4)
    g = BS.CTGeometry(s; n_angles = 4, fov_cm = 20.0, z_cm = 1.0)
    sino = Float32.(rand(MersenneTwister(0), 32, 4, 4))
    sino_orig = copy(sino)

    sino_out = BS.apply_focal_spot_blur(sino, BS.focal_spot_large(), g)

    @test sino == sino_orig          # input unchanged
    @test size(sino_out) == size(sino)
    @test sino_out !== sino           # fresh allocation

    # Bang vs alloc produce the same output.
    sino_bang = copy(sino)
    BS.apply_focal_spot_blur!(sino_bang, BS.focal_spot_large(), g)
    @test maximum(abs.(sino_out .- sino_bang)) < 1.0f-5
end

@testset "generate_focal_spot_samples" begin
    @testset "gaussian shape: weights sum to 1, samples inside [-w/2, +w/2]" begin
        fs = BS.FocalSpot(1.0, 1.0, :gaussian, 5)
        positions, weights = BS.generate_focal_spot_samples(fs)
        @test length(positions) == 5 * 5
        @test length(weights) == length(positions)
        @test sum(weights) ≈ 1.0  atol = 1.0e-12
        for (x, y) in positions
            @test -0.5 - 1.0e-9 ≤ x ≤ 0.5 + 1.0e-9
            @test -0.5 - 1.0e-9 ≤ y ≤ 0.5 + 1.0e-9
        end
        # Gaussian weighting: center sample has highest weight.
        @test maximum(weights) > 1 / length(weights)  # not uniform
    end

    @testset "uniform shape: equal weights" begin
        fs = BS.FocalSpot(1.0, 1.0, :uniform, 3)
        positions, weights = BS.generate_focal_spot_samples(fs)
        @test length(positions) == 9
        @test all(w ≈ 1 / 9 for w in weights)
    end

    @testset "point source (n_samples = 1)" begin
        fs = BS.FocalSpot(0.0, 0.0, :gaussian, 1)
        positions, weights = BS.generate_focal_spot_samples(fs)
        @test positions == [(0.0, 0.0)]
        @test weights == [1.0]
    end
end

# -----------------------------------------------------------------------------
# heel_effect.jl
# -----------------------------------------------------------------------------
@testset "HeelEffect struct + factories" begin
    @testset "default_heel_effect — sensible defaults" begin
        h = BS.default_heel_effect()
        @test h.anode_angle_deg == 7.0
        @test h.target_material === :tungsten
        @test h.effective_thickness_mm == 0.01
        @test h.enabled == true
    end

    @testset "default_heel_effect — kwarg override" begin
        h = BS.default_heel_effect(; anode_angle_deg = 12.0, effective_thickness_mm = 0.05)
        @test h.anode_angle_deg == 12.0
        @test h.effective_thickness_mm == 0.05
    end

    @testset "heel_effect_none — disabled flag, zero thickness" begin
        h = BS.heel_effect_none()
        @test h.enabled == false
        @test h.effective_thickness_mm == 0.0
    end
end

@testset "compute_heel_spectral — LIVE pipeline path" begin
    s = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_cols = 64, detector_rows = 4
    )
    g = BS.CTGeometry(s; n_angles = 4, fov_cm = 35.0, z_cm = 1.0)

    @testset "disabled heel → array of 1s" begin
        T_none = BS.compute_heel_spectral(BS.heel_effect_none(), g, [60.0, 80.0])
        @test size(T_none) == (g.n_cols, g.n_rows, 2)
        @test all(T_none .== 1.0)
    end

    @testset "enabled heel: shape contract + normalization" begin
        h = BS.default_heel_effect(effective_thickness_mm = 0.05)
        energies = [40.0, 60.0, 80.0, 120.0]
        T = BS.compute_heel_spectral(h, g, energies)
        @test size(T) == (g.n_cols, g.n_rows, length(energies))
        @test all(>(0), T)
        # Heel is fan-only — values constant within a row.
        for e in 1:length(energies), c in 1:g.n_cols
            row1 = T[c, 1, e]
            for r in 2:g.n_rows
                @test T[c, r, e] == row1
            end
        end
    end

    @testset "enabled heel: anode side dimmer than cathode side" begin
        h = BS.default_heel_effect(effective_thickness_mm = 0.05)
        T = BS.compute_heel_spectral(h, g, [60.0])
        mid_r = g.n_rows ÷ 2 + 1
        # Convention: col=1 is anode side, col=n_cols is cathode side.
        @test T[1, mid_r, 1] < T[g.n_cols, mid_r, 1]
    end

    @testset "enabled heel: lower-E photons more attenuated (energy-dependent μ_W)" begin
        h = BS.default_heel_effect(effective_thickness_mm = 0.05)
        T = BS.compute_heel_spectral(h, g, [40.0, 100.0])
        # At the anode-side column, the low-E transmission ratio is smaller.
        @test T[1, 1, 1] < T[1, 1, 2]
    end
end

@testset "apply_heel_effect! + apply_heel_effect" begin
    s = BS.Scanner(detector_cols = 64, detector_rows = 4)
    g = BS.CTGeometry(s; n_angles = 4, fov_cm = 35.0, z_cm = 1.0)

    @testset "disabled heel = identity" begin
        intensity = fill(1.0f0, 64, 4, 4)
        intensity_in = copy(intensity)
        BS.apply_heel_effect!(intensity, BS.heel_effect_none(), g)
        @test intensity == intensity_in
    end

    @testset "enabled heel: anode side dimmer than cathode side" begin
        h = BS.default_heel_effect(effective_thickness_mm = 0.05)
        intensity = fill(1.0f0, 64, 4, 4)
        BS.apply_heel_effect!(intensity, h, g)
        # Heel normalization is at γ = 0 (the fan-angle origin) but for even
        # n_cols no single column sits exactly there — adjacent ones straddle
        # it.  What matters is the anode/cathode asymmetry across the fan.
        @test intensity[1, 2, 2] < intensity[end, 2, 2]   # anode dimmer than cathode
        @test intensity[10, 2, 2] < intensity[end - 10, 2, 2]  # holds away from edges too
        # Sanity bounds: every entry stays finite, non-negative.
        @test all(>(0), intensity)
        @test all(isfinite, intensity)
    end

    @testset "alloc wrapper does not mutate input" begin
        h = BS.default_heel_effect(effective_thickness_mm = 0.05)
        intensity = fill(1.0f0, 64, 4, 4)
        intensity_in = copy(intensity)
        out = BS.apply_heel_effect(intensity, h, g)
        @test intensity == intensity_in
        @test out !== intensity
        @test size(out) == size(intensity)
    end

    @testset "bang vs alloc equivalence" begin
        h = BS.default_heel_effect(effective_thickness_mm = 0.05)
        intensity1 = fill(1.0f0, 64, 4, 4)
        intensity2 = copy(intensity1)
        out_alloc = BS.apply_heel_effect(intensity1, h, g)
        BS.apply_heel_effect!(intensity2, h, g)
        @test maximum(abs.(out_alloc .- intensity2)) < 1.0f-6
    end
end
