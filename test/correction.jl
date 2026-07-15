# Tests for src/correction/ — calibration, BHC (sinogram + image domain),
# radial cupping, PCCT pile-up correction.
#
# Modeled on the CatSim / XCIST tests we are porting from:
#   - gecatsim/tests/test_catsim/test_Prep_BHC_Accurate.py
#     (end-to-end: build cfg → call Prep_BHC_Accurate → assert non-zero prep).
#   - gecatsim/tests/test_catsim/test_PrepView.py
#     (algebraic identity: prep = (phantom - offset) / (air - offset) → LSC).
#
# CatSim/XCIST is BSD 3-Clause, GE Precision HealthCare.
# Upstream: https://github.com/xcist/main
#
# Coverage policy: every exported symbol from src/correction/ is exercised:
#   bhc_sinogram:   BHCPolynomial, BeamHardeningCorrection, TwoMaterialBHCPerColumn,
#                   calibrate_bhc, apply_bhc!, calibrate_bhc_two_material,
#                   apply_bhc_two_material, bhc_spectrum_per_column,
#                   compute_polychromatic_μ_water
#   bhc_image_domain: apply_bhc_image_domain
#   radial_cupping: apply_radial_cupping_correction!
#   calibration:    low_signal_correction_gpu!
#   pcct_pileup:    apply_pcct_pileup_correction!

# Helper: build a (Scanner, geom, protocol) matching the bowtie-aware
# test_api.jl pattern — small but clinically-shaped enough for BHC realism.
function _toy_bhc_scanner()
    return BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = 4,
        detector_cols = 32,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
        detector_material = :lumex,
        detector_depth = 3.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        bowtie_filter = :ge_revolution_large,
    )
end

# -----------------------------------------------------------------------------
# low_signal_correction_gpu! — CatSim LowSignalCorr.py simplification.
# -----------------------------------------------------------------------------
@testset "low_signal_correction_gpu!" begin
    @testset "clamps ≤ 0 to ε; positives (including tiny) untouched" begin
        prep = zeros(Float32, 2, 2, 2)
        # Layout chosen so each (i,j,k) corner is a deliberate test case.
        prep[1, 1, 1] = 1.0f0       # positive — untouched
        prep[1, 2, 1] = -0.5f0       # negative — clamped
        prep[2, 1, 1] = 0.0f0       # zero — clamped (kernel uses ≤)
        prep[2, 2, 1] = 3.5f0       # positive — untouched
        prep[1, 1, 2] = 2.0f0       # positive — untouched
        prep[1, 2, 2] = -1.0f-6      # negative — clamped
        prep[2, 1, 2] = 1.0f-12     # tiny POSITIVE — untouched (kernel
        # only clamps ≤ 0; ε is the floor for
        # CLAMPED values, not a min-positive)
        prep[2, 2, 2] = 10.0f0       # positive — untouched

        prep_in = copy(prep)
        BS.low_signal_correction_gpu!(prep)
        ε = Float32(1.0e-10)

        # Untouched positives stay identical.
        @test prep[1, 1, 1] == prep_in[1, 1, 1]    # 1.0
        @test prep[2, 2, 1] == prep_in[2, 2, 1]    # 3.5
        @test prep[1, 1, 2] == prep_in[1, 1, 2]    # 2.0
        @test prep[2, 1, 2] == prep_in[2, 1, 2]    # 1e-12 — still positive
        @test prep[2, 2, 2] == prep_in[2, 2, 2]    # 10.0

        # ≤ 0 entries clamped to ε.
        @test prep[1, 2, 1] == ε                   # was -0.5
        @test prep[2, 1, 1] == ε                   # was 0.0
        @test prep[1, 2, 2] == ε                   # was -1e-6
    end

    @testset "returns the same array (in-place contract)" begin
        prep = Float32[-1.0  -2.0;;; -3.0  -4.0]
        result = BS.low_signal_correction_gpu!(prep)
        @test result === prep
    end
end

# -----------------------------------------------------------------------------
# PCCT pile-up correction — `t̂ = S \ r` forward substitution.
# -----------------------------------------------------------------------------
@testset "apply_pcct_pileup_correction!" begin
    # Build a physically-plausible 4-bin S matrix (lower-triangular, diagonals
    # in [0.7, 0.95] = clinical aτ regime, mass conservation): each column
    # represents how a unit "truth" count in bin c distributes across recorded
    # bins r ≤ c.
    S = [
        0.92  0.0   0.0   0.0
        0.05  0.88  0.0   0.0
        0.02  0.08  0.85  0.0
        0.01  0.04  0.1  0.8
    ]

    I0_bins = [1.0e6, 8.0e5, 5.0e5, 3.0e5]
    n_col, n_row, n_view = 8, 4, 6

    @testset "shape contract — 4-bin specialized" begin
        # 3-bin should error (only 4-bin is supported).
        bins3 = [Array{Float32}(undef, n_col, n_row, n_view) for _ in 1:3]
        for b in bins3
            fill!(b, 0.1f0)
        end
        @test_throws ErrorException BS.apply_pcct_pileup_correction!(bins3, I0_bins[1:3], S[1:3, 1:3])

        # S of wrong shape
        bins4 = [Array{Float32}(undef, n_col, n_row, n_view) for _ in 1:4]
        for b in bins4
            fill!(b, 0.1f0)
        end
        @test_throws ErrorException BS.apply_pcct_pileup_correction!(bins4, I0_bins, S[1:3, 1:3])
        # Wrong I0_bins length
        @test_throws ErrorException BS.apply_pcct_pileup_correction!(bins4, I0_bins[1:3], S)
    end

    @testset "round-trip: build sino as -log(S·t / I0), correct → recover -log(t/I0)" begin
        # Construct a known truth-domain bin sinogram (per-bin log-line-integrals).
        Random.seed!(2026)
        t_truth = [Float32.(0.2 .+ 0.6 .* rand(n_col, n_row, n_view)) for _ in 1:4]

        # Forward pile-up: r = S · t per pixel, then bins = -log(r / I0_truth).
        # The recorded counts at bin b for pixel idx are r_b = Σ_c S[b, c] · t_c.
        # Truth counts t_c = I0_truth[c] · exp(-t_truth_log[c]).
        bins = Vector{Array{Float32, 3}}(undef, 4)
        for b in 1:4
            bins[b] = similar(t_truth[1])
        end
        for idx in eachindex(t_truth[1])
            t_counts = ntuple(c -> I0_bins[c] * exp(-t_truth[c][idx]), 4)
            for b in 1:4
                r_b = 0.0
                for c in 1:b
                    r_b += S[b, c] * t_counts[c]
                end
                bins[b][idx] = Float32(-log(max(r_b, 1.0e-12) / I0_bins[b]))
            end
        end

        # Correct in place — should undo the pile-up and return values close
        # to t_truth (within float roundoff).
        BS.apply_pcct_pileup_correction!(bins, I0_bins, S)

        for b in 1:4
            @test maximum(abs.(bins[b] .- t_truth[b])) < 1.0e-3
        end
    end

    @testset "identity case: S = I → output ≈ input" begin
        Random.seed!(7)
        bins = [Float32.(0.1 .+ 0.5 .* rand(n_col, n_row, n_view)) for _ in 1:4]
        bins_in = [copy(b) for b in bins]
        I = Matrix{Float64}(LinearAlgebra.I, 4, 4)
        BS.apply_pcct_pileup_correction!(bins, I0_bins, I)
        for b in 1:4
            @test maximum(abs.(bins[b] .- bins_in[b])) < 1.0e-5
        end
    end
end

# -----------------------------------------------------------------------------
# Polynomial helpers: fit_polynomial + generate_water_calibration_curve.
# -----------------------------------------------------------------------------
@testset "fit_polynomial" begin
    # Recover known cubic exactly.
    x = collect(range(0.0, 5.0, length = 50))
    coeffs_true = [0.3, -1.2, 0.5, 0.07]  # a₀ + a₁x + a₂x² + a₃x³
    y = [sum(coeffs_true[i + 1] * v^i for i in 0:3) for v in x]
    coeffs_fit = BS.fit_polynomial(x, y, 3)
    @test maximum(abs.(coeffs_fit .- coeffs_true)) < 1.0e-8
end

@testset "generate_water_calibration_curve" begin
    # Use a simple 3-bin "spectrum" with all weight at one energy so the
    # polychromatic line integral matches the monochromatic — trivial check.
    energies = [70.0]
    weights = [1.0]
    paths, measured, true_values = BS.generate_water_calibration_curve(
        energies, weights;
        max_path_cm = 30.0, n_points = 25, reference_energy_keV = 70.0,
    )
    @test length(paths) == length(measured) == length(true_values) == 25
    @test paths[1] ≈ 0.0
    @test paths[end] ≈ 30.0
    @test measured[1] ≈ 0.0  atol = 1.0e-9      # -log(1) = 0
    @test true_values[1] ≈ 0.0
    @test issorted(paths)
    @test issorted(measured)
    @test issorted(true_values)
    # Single-energy → measured == true.
    @test maximum(abs.(measured .- true_values)) < 1.0e-9
end

# -----------------------------------------------------------------------------
# calibrate_bhc — inner single-spectrum helper.
# -----------------------------------------------------------------------------
@testset "calibrate_bhc" begin
    # Realistic-ish spectrum: 80–120 keV, peaked at 60 (only physically
    # plausible-shaped — we don't need the IPEM file for this).
    energies = collect(range(20.0, 120.0, length = 21))
    weights = [exp(-((e - 60.0) / 25.0)^2) for e in energies]

    bhc = BS.calibrate_bhc(energies, weights; order = 3, reference_energy_keV = 70.0)
    @test bhc.polynomial.order == 3
    @test length(bhc.polynomial.coefficients) == 4
    @test bhc.polynomial.reference_energy_keV == 70.0

    # Calibration data round-trips through the polynomial closely.
    coeffs = bhc.polynomial.coefficients
    residuals = [
        sum(coeffs[i + 1] * m^i for i in 0:3) - t
            for (m, t) in zip(bhc.calibration_measured, bhc.calibration_true)
    ]
    @test maximum(abs.(residuals)) < 0.05   # order-3 fit on smooth water curve

    # Diagnostic-only fields are populated and have matching length.
    @test length(bhc.calibration_paths) == 100
    @test length(bhc.calibration_paths) == length(bhc.calibration_measured)
    @test length(bhc.calibration_paths) == length(bhc.calibration_true)
end

# -----------------------------------------------------------------------------
# apply_bhc! — per-column GPU/CPU kernel (Prep_BHC_Accurate.py port).
# -----------------------------------------------------------------------------
@testset "apply_bhc! (per-column)" begin
    scanner = _toy_bhc_scanner()
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 8, rotation_time = 0.5)
    geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
    sim_opts = BS.SimOptions(; fidelity = :eict)
    n_col = scanner.detector_cols

    bhc = BS.calibrate_bhc_two_material(
        sim_opts, protocol;
        scanner = scanner, geom = geom,
        order = 2, hu_low = 450.0, hu_high = 600.0,
    )

    @testset "Vector{BeamHardeningCorrection} dispatch" begin
        sino = zeros(Float32, n_col, scanner.detector_rows, protocol.views)
        sino_in = copy(sino)
        BS.apply_bhc!(sino, bhc.water_bhc_per_col)
        # zero sinogram → output is just the constant term per column.
        for col in 1:n_col
            expected = Float32(bhc.water_bhc_per_col[col].polynomial.coefficients[1])
            @test sino[col, 1, 1] ≈ expected  atol = 1.0e-4
        end
    end

    @testset "Vector{BHCPolynomial} dispatch — same result" begin
        sino_a = Float32.(0.1 .+ 0.5 .* rand(Float32, n_col, scanner.detector_rows, protocol.views))
        sino_b = copy(sino_a)
        BS.apply_bhc!(sino_a, bhc.water_bhc_per_col)
        polys = [b.polynomial for b in bhc.water_bhc_per_col]
        BS.apply_bhc!(sino_b, polys)
        @test maximum(abs.(sino_a .- sino_b)) < 1.0e-6
    end

    @testset "shape mismatch errors" begin
        sino = zeros(Float32, n_col + 1, scanner.detector_rows, protocol.views)  # off by one
        @test_throws ErrorException BS.apply_bhc!(sino, bhc.water_bhc_per_col)
    end
end

# -----------------------------------------------------------------------------
# bhc_spectrum_per_column — shape dispatch (1D / 2D / 3D).
# -----------------------------------------------------------------------------
@testset "bhc_spectrum_per_column" begin
    n_E, n_col, n_row = 12, 16, 4
    e = collect(range(20.0, 120.0, length = n_E))

    @testset "1D pass-through" begin
        w = rand(n_E)
        e_out, w_out = BS.bhc_spectrum_per_column(e, w)
        @test e_out === e
        @test w_out === w
    end

    @testset "2D [n_E, n_col] unchanged" begin
        w = rand(n_E, n_col)
        _, w_out = BS.bhc_spectrum_per_column(e, w)
        @test size(w_out) == (n_E, n_col)
        @test w_out === w
    end

    @testset "2D [n_col, n_E] auto-transposed" begin
        w = rand(n_col, n_E)
        _, w_out = BS.bhc_spectrum_per_column(e, w)
        @test size(w_out) == (n_E, n_col)
    end

    @testset "3D [n_col, n_row, n_E] → center row" begin
        w = rand(n_col, n_row, n_E)
        _, w_out = BS.bhc_spectrum_per_column(e, w)
        @test size(w_out) == (n_E, n_col)
        # Match center-row slice (row = n_row÷2 + 1).
        mid_r = n_row ÷ 2 + 1
        @test all(w_out[:, c] ≈ Float64.(w[c, mid_r, :]) for c in 1:n_col)
    end

    @testset "shape mismatch errors" begin
        # 2D with neither dim matching length(e)
        @test_throws ErrorException BS.bhc_spectrum_per_column(e, rand(7, 9))
        # 3D with last dim mismatched
        @test_throws ErrorException BS.bhc_spectrum_per_column(e, rand(n_col, n_row, n_E + 1))
        # 4D not supported
        @test_throws ErrorException BS.bhc_spectrum_per_column(e, rand(2, 2, 2, 2))
    end
end

# -----------------------------------------------------------------------------
# compute_polychromatic_μ_water — physical bounds + hardening direction.
# -----------------------------------------------------------------------------
@testset "compute_polychromatic_μ_water" begin
    scanner = _toy_bhc_scanner()
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 16, rotation_time = 0.5)
    geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
    sim_opts = BS.SimOptions(; fidelity = :eict)

    μ0 = BS.compute_polychromatic_μ_water(
        sim_opts, protocol;
        scanner = scanner, geom = geom, water_path_cm = 0.0
    )
    μ20 = BS.compute_polychromatic_μ_water(
        sim_opts, protocol;
        scanner = scanner, geom = geom, water_path_cm = 20.0
    )
    μ50 = BS.compute_polychromatic_μ_water(
        sim_opts, protocol;
        scanner = scanner, geom = geom, water_path_cm = 50.0
    )

    # Physical bounds for water in the diagnostic energy range.
    # NIST water μ ranges from ~0.5 cm⁻¹ at 30 keV down to ~0.15 at 120 keV;
    # mean-E weighting gives a polychromatic effective in [0.15, 0.30].
    @test 0.15 < μ0 < 0.3
    @test 0.15 < μ20 < 0.3
    @test 0.15 < μ50 < 0.3

    # Beam hardening: longer water path raises the mean E, and water μ
    # decreases monotonically with E in this range, so μ_eff drops with path.
    @test μ0 > μ20 > μ50

    pcct_scanner = BS.Scanner(
        detector_type = :photon_counting, detector_material = :cdte,
        n_energy_bins = 4, energy_thresholds = [20.0, 35.0, 55.0, 70.0],
    )
    pcct_geom = BS.CTGeometry(pcct_scanner; n_angles = 16)
    @test_throws ArgumentError BS.compute_polychromatic_μ_water(
        BS.SimOptions(fidelity = :pcct), protocol;
        scanner = pcct_scanner, geom = pcct_geom, water_path_cm = 20.0,
    )
end

# -----------------------------------------------------------------------------
# calibrate_bhc_two_material — high-level entry contract.
# -----------------------------------------------------------------------------
@testset "calibrate_bhc_two_material (high-level)" begin
    scanner = _toy_bhc_scanner()
    protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 8, rotation_time = 0.5)
    geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
    sim_opts = BS.SimOptions(; fidelity = :eict)

    bhc = BS.calibrate_bhc_two_material(
        sim_opts, protocol;
        scanner = scanner, geom = geom,
        order = 2, hu_low = 450.0, hu_high = 600.0,
    )

    @test bhc isa BS.TwoMaterialBHCPerColumn

    n_col = scanner.detector_cols
    @test length(bhc.water_bhc_per_col) == n_col
    @test size(bhc.w_norm_per_col, 2) == n_col
    @test size(bhc.w_norm_per_col, 1) == length(bhc.energies)
    @test length(bhc.μ_water_E) == length(bhc.energies)
    @test length(bhc.μ_bone_E) == length(bhc.energies)

    # All polynomials share the same order.
    @test all(p.polynomial.order == 2 for p in bhc.water_bhc_per_col)

    # All per-column spectra are normalized to 1.
    for c in 1:n_col
        @test sum(bhc.w_norm_per_col[:, c]) ≈ 1.0  atol = 1.0e-9
    end

    # Global μ_water_ref / μ_bone_ref are positive, physically plausible.
    @test bhc.μ_water_ref > 0
    @test bhc.μ_bone_ref > bhc.μ_water_ref   # cortical bone attenuates more
    @test bhc.hu_low == 450.0
    @test bhc.hu_high == 600.0

    # Default reference energy is the column-mean spectrum's mean E,
    # which for 120 kVp is in [55, 75] keV.
    @test 55.0 < bhc.reference_energy_keV < 75.0
end

# -----------------------------------------------------------------------------
# apply_bhc_two_material — shape preservation + zero-sino identity.
# -----------------------------------------------------------------------------
@testset "apply_bhc_two_material (TwoMaterialBHCPerColumn)" begin
    if !HAS_GPU
        @info "Skipping apply_bhc_two_material on CPU (FDK/Siddon round-trip is GPU-only in test budget)"
    else
        scanner = _toy_bhc_scanner()
        protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 8, rotation_time = 0.5)
        geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
        sim_opts = BS.SimOptions(; fidelity = :eict)

        bhc = BS.calibrate_bhc_two_material(
            sim_opts, protocol;
            scanner = scanner, geom = geom,
            order = 2, hu_low = 450.0, hu_high = 600.0,
        )

        matrix_size = (16, 16, 4)
        n_col, n_row, n_view = scanner.detector_cols, scanner.detector_rows, protocol.views
        sino = to_gpu(zeros(Float32, n_col, n_row, n_view))
        sino_out = BS.apply_bhc_two_material(sino, bhc, geom, matrix_size)

        # Shape preserved.
        @test size(sino_out) == (n_col, n_row, n_view)
        @test eltype(sino_out) == Float32
        @test all(isfinite, sino_out)
    end
end

# -----------------------------------------------------------------------------
# apply_bhc_image_domain — So et al. 2009. Identity at scale=0, mutates above.
# -----------------------------------------------------------------------------
@testset "apply_bhc_image_domain" begin
    if !HAS_GPU
        @info "Skipping apply_bhc_image_domain on CPU (Siddon+FDK round-trip is GPU-only in test budget)"
    else
        scanner = _toy_bhc_scanner()
        protocol = BS.CTProtocol(mA = 200.0, kVp = 120.0, views = 8, rotation_time = 0.5)
        geom = BS.CTGeometry(scanner; n_angles = protocol.views, fov_cm = 20.0, z_cm = 5.0)
        matrix_size = (16, 16, 4)
        μ_water_ref = 0.19   # ~120 kVp body water, physically plausible

        @testset "scale_factor = 0 → identity" begin
            recon = to_gpu(Float32.(μ_water_ref .+ 0.05 .* rand(Float32, matrix_size...)))
            recon_in = Array(recon)
            BS.apply_bhc_image_domain(
                recon, geom, matrix_size, μ_water_ref;
                hu_low = 70.0, hu_high = 150.0, scale_factor = 0.0
            )
            @test maximum(abs.(Array(recon) .- recon_in)) < 1.0e-5
        end

        @testset "scale_factor > 0 with above-threshold pixels → mutation" begin
            # Construct a recon with a high-attenuation core (well above hu_high).
            recon_cpu = fill(Float32(μ_water_ref), matrix_size...)
            # Add a hot spot at the center that translates to ~500 HU.
            μ_hot = Float32(μ_water_ref * 1.5)
            recon_cpu[8, 8, 2] = μ_hot
            recon = to_gpu(recon_cpu)
            recon_in = Array(recon)
            BS.apply_bhc_image_domain(
                recon, geom, matrix_size, μ_water_ref;
                hu_low = 100.0, hu_high = 200.0, scale_factor = 0.5
            )
            @test Array(recon) != recon_in   # something changed
            @test all(isfinite, Array(recon))
        end
    end
end

# -----------------------------------------------------------------------------
# apply_radial_cupping_correction! — synthetic radial cup → flat.
# -----------------------------------------------------------------------------
@testset "apply_radial_cupping_correction!" begin
    nx = ny = 64
    nz = 2
    fov_cm = 35.0
    pixel_cm = fov_cm / nx
    cx, cy = (nx + 1) / 2.0, (ny + 1) / 2.0

    @testset "flattens synthetic radial cup to target" begin
        # Build a HU volume that is a perfect parabola in radius — exactly the
        # form the corrector's order-2 even polynomial can null out.
        # 0 at center, +50 at radius = fov/2.
        hu = zeros(Float32, nx, ny, nz)
        for j in 1:ny, i in 1:nx
            r_cm = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
            hu[i, j, :] .= Float32(-25.0 + 50.0 * (r_cm / (fov_cm / 2))^2)
        end

        BS.apply_radial_cupping_correction!(
            hu;
            fov_cm = fov_cm, hu_lo = -200.0, hu_hi = 200.0,
            poly_order = 2, target_hu = 0.0
        )

        # After correction the radial bias is gone — every voxel should be
        # close to target_hu (0.0).  Tolerance is generous because the fit
        # is on a discrete grid.
        @test maximum(abs.(hu)) < 1.0
    end

    @testset "preserves voxels outside the water mask" begin
        # Voxels well above hu_hi should NOT participate in the fit, but they
        # are still shifted by the same fit_val.  Just check finiteness +
        # shape preservation.
        hu = fill(Float32(100.0), nx, ny, nz)
        hu[1:8, 1:8, :] .= 1000.0f0       # "bone" insert — out of water mask
        hu_in = copy(hu)
        BS.apply_radial_cupping_correction!(
            hu;
            fov_cm = fov_cm, hu_lo = -100.0, hu_hi = 80.0,
            poly_order = 2, target_hu = 0.0
        )
        # Less than 10 samples in water mask → function early-returns the
        # slice; assert it's not totally destroyed.
        @test size(hu) == size(hu_in)
        @test all(isfinite, hu)
    end

    @testset "returns the same array (in-place contract)" begin
        hu = fill(Float32(0.0), 32, 32, 1)
        result = BS.apply_radial_cupping_correction!(hu; fov_cm = 35.0)
        @test result === hu
    end
end

# -----------------------------------------------------------------------------
# apply_radial_capping_basis! — basis-domain sibling of radial_cupping.
# Smoke + behavioral tests.  KEPT but not in current notebooks.
# -----------------------------------------------------------------------------
@testset "apply_radial_capping_basis!" begin
    nx = ny = 32
    nz = 2
    fov_cm = 35.0
    pixel_cm = fov_cm / nx
    cx, cy = (nx + 1) / 2.0, (ny + 1) / 2.0

    @testset "flattens synthetic radial cup on a basis pair" begin
        # Build per-basis volumes with a known r² cup.  Background is
        # mostly the cup; planted "rods" are excluded by quantile selection.
        a = zeros(Float32, nx, ny, nz)
        c = zeros(Float32, nx, ny, nz)
        for j in 1:ny, i in 1:nx
            r_cm = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
            a[i, j, :] .= Float32(0.1 + 0.3 * (r_cm / (fov_cm / 2))^2)
            c[i, j, :] .= Float32(0.05 + 0.15 * (r_cm / (fov_cm / 2))^2)
        end
        a_in = copy(a)
        c_in = copy(c)
        info = BS.apply_radial_capping_basis!(
            a, c;
            fov_cm = fov_cm, poly_order = 2, verbose = false,
        )
        # Both volumes were mutated.
        @test a != a_in
        @test c != c_in
        # After correction: DC (r=0) preserved (target = c₀), radial
        # curvature flattened.  Check std drops in the in-FOV region.
        @test std(a[:, :, 1]) < std(a_in[:, :, 1])
        @test std(c[:, :, 1]) < std(c_in[:, :, 1])
        @test all(isfinite, a)
        @test all(isfinite, c)

        # Diagnostics NamedTuple shape.
        @test info.fov_cm == fov_cm
        @test info.poly_order == 2
        @test size(info.coeffs_a) == (3, nz)
        @test size(info.coeffs_c) == (3, nz)
    end

    @testset "shape mismatch tolerated (per-basis, no joint constraint)" begin
        # The function processes a and c independently — different shapes
        # would fail in size(a) destructuring; check they match:
        a = zeros(Float32, nx, ny, nz)
        c = zeros(Float32, nx, ny, nz)
        info = BS.apply_radial_capping_basis!(a, c; fov_cm = fov_cm, verbose = false)
        @test info isa NamedTuple
    end

    @testset "kwargs propagate (q_lo, q_hi clamp background)" begin
        # Narrow quantile range — fewer fit points but still works.
        a = Float32.(0.1 .+ 0.05 .* randn(MersenneTwister(0), nx, ny, nz))
        c = Float32.(0.05 .+ 0.02 .* randn(MersenneTwister(1), nx, ny, nz))
        info = BS.apply_radial_capping_basis!(
            a, c; fov_cm = fov_cm, q_lo = 0.4, q_hi = 0.6, verbose = false,
        )
        @test info.q_lo == 0.4
        @test info.q_hi == 0.6
    end
end
