# Tests for src/source/bowtie_filter.jl and src/bowtie/{large,medium,small}.txt.
#
# Modeled on the CatSim / XCIST tests we are porting from:
#   - gecatsim/tests/test_catsim/test_Xray_Filter.py
#     (end-to-end: build cfg → call Xray_Filter → reshape → assert
#      trans.max() ≈ 0.9266918, places=4 + shape (900, 16, 20))
#   - gecatsim/tests/test_catsim/test_Resample_Spectrum_Bowtie_FlatFilter.py
#     (algebraic identity: spec.netIvec == spec.Ivec * bowtie.transVec * FiltrationTransVec)
#
# CatSim/XCIST is BSD 3-Clause, GE Precision HealthCare.
# Upstream: https://github.com/xcist/main
#
# Coverage policy: every exported symbol in src/source/bowtie_filter.jl is
# exercised below. The post-audit public surface is:
#   BowtieFilter, bowtie_filter_head, bowtie_filter_none,
#   load_catsim_bowtie, load_builtin_bowtie, resolve_bowtie_filter,
#   interpolate_thickness, compute_bowtie_attenuation_spectral, get_bowtie_mu.

# Path to the upstream-format bundled bowtie data (BSD 3-Clause GE/CatSim).
const _BOWTIE_DATA_DIR = joinpath(@__DIR__, "..", "src", "bowtie")

# Reference first-row values from gecatsim/bowtie/{large,medium,small}.txt
# (verified byte-identical via diff -q against /tmp/gecatsim_audit at audit
# time, 2026-05-11).  These pin the bundled data files: any drift here means
# someone edited the upstream-licensed asset.
const _CATSIM_FIRST_ROW = Dict(
    # size => (angle_rad, t_Al_cm, t_graphite_cm, t_Cu_cm, t_Ti_cm)
    "large" => (-0.479582, 3.711864, 0.0, 0.0, 0.0),
    "medium" => (-0.479582, 3.537235, 0.0, 0.0, 0.0),
    "small" => (-0.479852, 3.547008, 0.0, 0.0, 0.0),
)

# Clinical-scale geometry matching xcist's test_Xray_Filter.py.  Shape parity:
# n_cols = 900, n_rows = 16, ~25° fan width — same regime where the bowtie
# thickness variation across γ is physically meaningful.
function _clinical_scanner_with_bowtie(bowtie::Symbol)
    return BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = 16,
        detector_cols = 900,
        detector_row_size = 1.0,
        detector_col_size = 0.6,
        detector_material = :lumex,
        detector_depth = 3.0,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        bowtie_filter = bowtie,
    )
end

# Verify the upstream-licensed data files in src/bowtie/ have not drifted
# from the CatSim/XCIST release we ported.
@testset "bundled bowtie data (BSD 3-Clause CatSim/XCIST)" begin
    for body in ("large", "medium", "small")
        filepath = joinpath(_BOWTIE_DATA_DIR, "$(body).txt")
        @test isfile(filepath)

        # Copyright header must be preserved — that's our in-band attribution.
        first_line = readlines(filepath)[1]
        @test occursin("GE Precision HealthCare", first_line)
        @test occursin("xcist", first_line)

        # 3 header lines + 888 data rows = 891 total.
        lines = readlines(filepath)
        @test length(lines) == 891

        # First non-comment row matches upstream verbatim.
        bf = BS.load_catsim_bowtie(filepath; name = "test_$(body)")
        expected = _CATSIM_FIRST_ROW[body]
        @test bf.angles[1] ≈ expected[1]  atol = 1.0e-6
        @test bf.thickness[1, 1] ≈ expected[2]  atol = 1.0e-6
        @test bf.thickness[1, 2] ≈ expected[3]  atol = 1.0e-6
        @test bf.thickness[1, 3] ≈ expected[4]  atol = 1.0e-6
        @test bf.thickness[1, 4] ≈ expected[5]  atol = 1.0e-6
    end
end

@testset "BowtieFilter struct" begin
    bf = BS.BowtieFilter(
        [0.0, 0.1, 0.2],
        [1.0 0.5; 0.8 0.4; 0.6 0.3],
        ["Al", "graphite"],
        "manual",
    )
    @test bf.angles == [0.0, 0.1, 0.2]
    @test size(bf.thickness) == (3, 2)
    @test bf.materials == ["Al", "graphite"]
    @test bf.name == "manual"
end

@testset "load_catsim_bowtie" begin
    bf = BS.load_catsim_bowtie(
        joinpath(_BOWTIE_DATA_DIR, "medium.txt");
        name = "catsim_medium_under_test"
    )
    @test bf.name == "catsim_medium_under_test"
    @test bf.materials == ["Al", "graphite", "Cu", "Ti"]
    @test length(bf.angles) == 888
    @test size(bf.thickness) == (888, 4)

    # Angles are in radians, span roughly [-0.48, +0.48] (≈ 55° total fan).
    @test minimum(bf.angles) ≈ -0.479582  atol = 1.0e-6
    @test maximum(bf.angles) ≈ 0.477424  atol = 1.0e-6

    # Thickness is in cm.  Medium body Al column tapers center-thin / edge-thick:
    # ~0.09 cm at center, ~3.5 cm at edges.
    @test minimum(bf.thickness[:, 1]) < 0.2
    @test maximum(bf.thickness[:, 1]) > 3.0
end

@testset "load_builtin_bowtie" begin
    for body in ("large", "medium", "small")
        bf = BS.load_builtin_bowtie(body)
        @test bf.name == "catsim_$(body)"
        @test length(bf.angles) == 888
        @test size(bf.thickness) == (888, 4)
        @test bf.materials == ["Al", "graphite", "Cu", "Ti"]
    end

    @test_throws ErrorException BS.load_builtin_bowtie("XXX_nonexistent")
end

@testset "resolve_bowtie_filter" begin
    # CatSim-data branches: each Symbol maps to the bundled 888-point profile.
    for (sym, expected_name) in (
            (:large_body, "catsim_large"),
            (:medium_body, "catsim_medium"),
            (:small_body, "catsim_small"),
            (:ge_revolution_large, "catsim_large"),
            (:ge_revolution_medium, "catsim_medium"),
            (:ge_revolution_small, "catsim_small"),
        )
        bf = BS.resolve_bowtie_filter(sym)
        @test bf.name == expected_name
        @test length(bf.angles) == 888
    end

    # Generic-factory branches still wired in for back-compat.
    @test BS.resolve_bowtie_filter(:none).name == "none"
    @test BS.resolve_bowtie_filter(:head).name == "head"

    # Unknown symbols error with a discoverable message.
    @test_throws ErrorException BS.resolve_bowtie_filter(:not_a_filter)
end

@testset "interpolate_thickness" begin
    # Hand-built 3-node filter, single material, so interpolation is checkable.
    bf = BS.BowtieFilter(
        [0.0, 0.1, 0.2],
        reshape([1.0, 0.5, 0.2], 3, 1),
        ["Al"], "test",
    )

    # At node: exact value.
    @test BS.interpolate_thickness(bf, 0.0)[1] ≈ 1.0
    @test BS.interpolate_thickness(bf, 0.1)[1] ≈ 0.5
    @test BS.interpolate_thickness(bf, 0.2)[1] ≈ 0.2

    # Midway: linear blend.
    @test BS.interpolate_thickness(bf, 0.05)[1] ≈ 0.75
    @test BS.interpolate_thickness(bf, 0.15)[1] ≈ 0.35

    # Out of range: clamps to the endpoint (one-sided extrapolation).
    @test BS.interpolate_thickness(bf, 0.5)[1] ≈ 0.2
    @test BS.interpolate_thickness(bf, -0.5)[1] ≈ 0.2  # uses |fan_angle|
    @test BS.interpolate_thickness(bf, -0.1)[1] ≈ 0.5  # symmetric on |γ|
end

# End-to-end smoke test, modeled directly on CatSim's test_Xray_Filter.py.
# We build a clinical-scale (900, 16) detector + medium bowtie + 20-energy
# grid (matching the shape (900, 16, 20) used upstream), then assert:
#   1. Output shape matches (900, 16, 20).
#   2. All transmission factors lie in (0, 1] — bowtie attenuates, never
#      amplifies, and the center has the minimum thickness so trans.max()
#      should be close to (but strictly less than) 1.
#   3. Energy monotonicity — at any pixel, higher E sees less attenuation.
#   4. Cone correction — at fixed col, attenuation increases with |row offset|.
#   5. Fan modulation — center has thinnest bowtie, so center.max > edge.max.
@testset "compute_bowtie_attenuation_spectral (CatSim Xray_Filter port)" begin
    scanner = _clinical_scanner_with_bowtie(:medium_body)
    geom = BS.CTGeometry(scanner; n_angles = 16, fov_cm = 50.0, z_cm = 5.0)
    energies = collect(range(20.0, 120.0, length = 20))  # matches xcist's 20-bin grid
    bowtie = BS.resolve_bowtie_filter(scanner.bowtie_filter)

    trans = BS.compute_bowtie_attenuation_spectral(bowtie, geom, energies)

    # (1) Shape parity with xcist test_Xray_Filter.py.
    @test size(trans) == (900, 16, 20)

    # (2) Bounds — transmittance is a probability.
    @test all(>(0.0), trans)
    @test all(<=(1.0), trans)
    @test all(isfinite, trans)
    # Center is thinnest in CatSim body bowties (~0.09 cm Al for medium),
    # so the maximum across the array should be near 1 but below it.
    @test 0.85 < maximum(trans) < 1.0
    # Edge is ~3.5 cm Al → exp(-μ·t) drops far below 1 at the low end.
    @test minimum(trans) < 0.1

    # (3) Energy monotonicity at the geometric center pixel.
    mid_c = size(trans, 1) ÷ 2 + 1
    mid_r = size(trans, 2) ÷ 2 + 1
    center_E_profile = trans[mid_c, mid_r, :]
    @test issorted(center_E_profile)  # exp(-μ(E)·t) increases as μ(E) decreases with E

    # (4) Cone-angle correction: at fixed col, paths through bowtie get longer
    # as cone angle |α| grows (t / cos(α)), so transmission drops.
    #   row=1 (top edge) and row=end (bottom edge) should both see less
    #   transmission than the middle row.  Pick the lowest-energy bin to
    #   maximize the effect.
    @test trans[mid_c, 1, 1] < trans[mid_c, mid_r, 1]
    @test trans[mid_c, end, 1] < trans[mid_c, mid_r, 1]

    # (5) Fan modulation: at fixed (row, E), center column has the thinnest
    # bowtie → highest transmission.  Edge column has the thickest → lowest.
    @test trans[mid_c, mid_r, end] > trans[1, mid_r, end]
    @test trans[mid_c, mid_r, end] > trans[end, mid_r, end]
end

# CatSim's "no bowtie" branch: bowtie_filter() short-circuits when
# cfg.protocol.bowtie is falsy.  Our equivalent is :none → bowtie_filter_none()
# with zero thickness.  Smoke-test that transmission collapses to 1 everywhere.
@testset "compute_bowtie_attenuation_spectral: :none → transmission == 1" begin
    scanner = _clinical_scanner_with_bowtie(:none)
    geom = BS.CTGeometry(scanner; n_angles = 8, fov_cm = 50.0, z_cm = 5.0)
    energies = [60.0, 80.0, 100.0]
    bowtie = BS.resolve_bowtie_filter(scanner.bowtie_filter)
    trans = BS.compute_bowtie_attenuation_spectral(bowtie, geom, energies)

    @test all(trans .≈ 1.0)
end

# Reference parity for all three bundled body bowties — the three CatSim
# .txt files load and produce three *distinct, physically valid* profiles.
#
# Note: CatSim's small/medium/large naming refers to the *patient* size the
# filter is optimized for, NOT bowtie thickness ordering. At γ ≈ 0:
#   medium ≈ 0.094 cm Al (thinnest)
#   small  ≈ 0.110 cm Al
#   large  ≈ 0.255 cm Al
# So we don't enforce a thickness ordering — we just check each profile is
# well-formed and the three differ from one another.
@testset "compute_bowtie_attenuation_spectral: three CatSim profiles distinct" begin
    geom_params = (; n_angles = 8, fov_cm = 50.0, z_cm = 5.0)
    energies = [60.0]

    function _center_trans(bowtie_sym)
        scanner = _clinical_scanner_with_bowtie(bowtie_sym)
        geom = BS.CTGeometry(scanner; geom_params...)
        bf = BS.resolve_bowtie_filter(scanner.bowtie_filter)
        t = BS.compute_bowtie_attenuation_spectral(bf, geom, energies)
        # Transmission at the geometric-center pixel — purely the bowtie min.
        mid_c = size(t, 1) ÷ 2 + 1
        mid_r = size(t, 2) ÷ 2 + 1
        return t[mid_c, mid_r, 1]
    end

    t_small = _center_trans(:small_body)
    t_medium = _center_trans(:medium_body)
    t_large = _center_trans(:large_body)

    # Every center-of-fan transmission is in a physically reasonable range
    # (Al, 60 keV, 0.05-0.5 cm of bowtie → exp(-μ·t) ∈ ~[0.7, 0.97]).
    @test 0.7 < t_small < 1.0
    @test 0.7 < t_medium < 1.0
    @test 0.7 < t_large < 1.0

    # The three CatSim files are distinct — guards against accidental copy.
    @test t_small != t_medium
    @test t_small != t_large
    @test t_medium != t_large
end
