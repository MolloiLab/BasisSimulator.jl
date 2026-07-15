# Tests for src/projection/ — Siddon forward projection + polychromatic helpers.
#
# Coverage policy:
#   siddon.jl       : siddon_forward_project, siddon_forward_project!
#                     (siddon_fused_poly_project! and siddon_fused_spectral_project!
#                     are exercised end-to-end through test/api.jl's
#                     simulate!(PCCTWorkspace) and simulate!(EICTWorkspace);
#                     duplicating that here would just retest the workspace path).
#   polychromatic.jl: create_μ_volume! (the live public helper).
#                     `_apply_physics_no_noise!` and `_forward_project_poly!` are
#                     reached end-to-end through test/api.jl's simulate! tests —
#                     they're private + need the full PhysicsConfig + workspace
#                     dance to set up, so we skip duplicate behavioral coverage
#                     here.

# -----------------------------------------------------------------------------
# Helper: tiny scanner + geometry for projection tests.
# -----------------------------------------------------------------------------
function _toy_proj_geom(; n_cols = 16, n_rows = 4, n_angles = 4, fov_cm = 5.0)
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 1080.0,
        detector_rows = n_rows,
        detector_cols = n_cols,
        detector_row_size = 1.0,
        detector_col_size = 1.0,
    )
    return BS.CTGeometry(scanner; n_angles = n_angles, fov_cm = fov_cm, z_cm = 1.0)
end

# -----------------------------------------------------------------------------
# siddon_forward_project — allocating + identity case.
# -----------------------------------------------------------------------------
@testset "siddon_forward_project (allocating)" begin
    geom = _toy_proj_geom()
    matrix_size = (16, 16, 4)

    @testset "zero volume → zero sinogram" begin
        vol = zeros(Float32, matrix_size...)
        sino = BS.siddon_forward_project(vol, geom)
        @test size(sino) == (geom.n_cols, geom.n_rows, geom.n_angles)
        @test eltype(sino) == Float32
        @test all(sino .== 0)
    end

    @testset "constant μ block → constant-ish sinogram on rays through it" begin
        vol = fill(Float32(0.2), matrix_size...)
        sino = BS.siddon_forward_project(vol, geom)
        @test size(sino) == (geom.n_cols, geom.n_rows, geom.n_angles)
        # All rays through a uniform block have positive line integral.
        # Pick the middle column (where the FOV is fully covered).
        mid_c = geom.n_cols ÷ 2 + 1
        mid_r = geom.n_rows ÷ 2 + 1
        @test all(>(0), sino[mid_c, mid_r, :])
        # Line integral = μ × chord length.  Chord ≤ fov × √2; with μ = 0.2 cm⁻¹
        # and fov = 5 cm, max line integral ≈ 0.2 × 5 × √2 ≈ 1.41.
        @test maximum(sino) ≤ 0.2 * 5.0 * sqrt(2) + 1.0e-3
    end

    @testset "linearity in μ — 2× scale doubles the sinogram" begin
        vol = fill(Float32(0.1), matrix_size...)
        sino_1x = BS.siddon_forward_project(vol, geom)
        sino_2x = BS.siddon_forward_project(2 .* vol, geom)
        # Linearity holds exactly (Siddon is a linear operator on μ).
        @test maximum(abs.(sino_2x .- 2 .* sino_1x)) < 1.0e-4
    end
end

# -----------------------------------------------------------------------------
# siddon_forward_project! — in-place equivalence.
# -----------------------------------------------------------------------------
@testset "siddon_forward_project! (in-place)" begin
    geom = _toy_proj_geom()
    matrix_size = (16, 16, 4)
    vol = fill(Float32(0.15), matrix_size...)

    sino_alloc = BS.siddon_forward_project(vol, geom)
    sino_inplace = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    BS.siddon_forward_project!(sino_inplace, vol, geom)

    @test maximum(abs.(sino_alloc .- sino_inplace)) < 1.0e-4
end

# -----------------------------------------------------------------------------
# dd_forward_project — distance-driven projector (drop-in for Siddon).
# Must agree with Siddon on line integrals (both discretise ∫μ·dl) and be a
# linear operator on μ.  Small edge differences (DD footprint vs Siddon chord)
# are expected; we bound the mean relative deviation on rays that hit the object.
# -----------------------------------------------------------------------------
@testset "dd_forward_project (distance-driven)" begin
    # Larger toy grid so the object spans many detector rays (fair DD-vs-Siddon stats).
    geom = _toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
    nx = ny = 64
    nz = 8

    @testset "shape + zero volume → zero sinogram" begin
        vol = zeros(Float32, nx, ny, nz)
        sino = BS.dd_forward_project(vol, geom)
        @test size(sino) == (geom.n_cols, geom.n_rows, geom.n_angles)
        @test eltype(sino) == Float32
        @test all(sino .== 0)
    end

    # Centred water cylinder, μ = 0.2 cm⁻¹.
    vol = zeros(Float32, nx, ny, nz)
    cx = (nx + 1) / 2
    cy = (ny + 1) / 2
    R = 0.6 * (nx / 2)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - cx)^2 + (j - cy)^2 <= R^2
            vol[i, j, k] = 0.2f0
        end
    end

    sino_siddon = BS.siddon_forward_project(vol, geom)
    sino_dd = BS.dd_forward_project(vol, geom)

    @testset "agrees with Siddon on object rays" begin
        mask = sino_siddon .> 1.0f-4
        relΔ = abs.(sino_dd[mask] .- sino_siddon[mask]) ./ sino_siddon[mask]
        @test count(mask) > 1000                 # object actually illuminated
        @test sum(relΔ) / length(relΔ) < 0.01    # mean relative deviation < 1%
        @test maximum(relΔ) < 0.05               # worst-ray deviation < 5%
    end

    @testset "in-place == allocating" begin
        sino_ip = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.dd_forward_project!(sino_ip, vol, geom)
        @test maximum(abs.(sino_ip .- sino_dd)) < 1.0e-5
    end

    @testset "linearity in μ — 2× scale doubles the sinogram" begin
        sino_2x = BS.dd_forward_project(2 .* vol, geom)
        @test maximum(abs.(sino_2x .- 2 .* sino_dd)) < 1.0e-4
    end
end

# -----------------------------------------------------------------------------
# dd_fused_poly_project! / dd_fused_spectral_project! — must agree with the
# Siddon fused kernels (same mask / μ-table / weights) within the DD-vs-Siddon
# discretisation tolerance.  These are the kernels the EI (polychromatic) and
# PCCT (photon-counting) pipelines call.
# -----------------------------------------------------------------------------
@testset "dd fused projectors vs Siddon fused" begin
    geom = _toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
    nx = ny = 64
    nz = 8
    N_E = 8

    # material mask: 0 = air, 1 = water cylinder
    mask = zeros(UInt16, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - 32.5)^2 + (j - 32.5)^2 <= (0.6 * 32)^2
            mask[i, j, k] = UInt16(1)
        end
    end
    μ_table = zeros(Float32, 2, N_E)             # row 1 = air (0), row 2 = water
    for e in 1:N_E
        μ_table[2, e] = 0.30f0 - 0.12f0 * (e - 1) / (N_E - 1)
    end
    wη = fill(Float32(1 / N_E), N_E)

    @testset "fused poly" begin
        sino_s = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        sino_d = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.siddon_fused_poly_project!(sino_s, mask, geom, μ_table, wη, Val(N_E))
        BS.dd_fused_poly_project!(sino_d, mask, geom, μ_table, wη, Val(N_E))
        m = sino_s .> 1.0f-4
        relΔ = abs.(sino_d[m] .- sino_s[m]) ./ sino_s[m]
        @test count(m) > 1000
        @test sum(relΔ) / length(relΔ) < 0.01      # mean within 1%
        @test maximum(relΔ) < 0.05
    end

    @testset "fused spectral (PCCT, single tile)" begin
        n_bins = Int32(2)
        K = N_E
        W = zeros(Float32, N_E, 2)
        for e in 1:N_E
            W[e, 1] = wη[e] * (1.0f0 - (e - 1) / (N_E - 1))
            W[e, 2] = wη[e] * ((e - 1) / (N_E - 1))
        end
        pilot = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        ne = length(pilot)
        out_s = zeros(Float32, ne * 2)
        out_d = zeros(Float32, ne * 2)
        BS.siddon_fused_spectral_project!(pilot, out_s, n_bins, mask, geom, μ_table, W, Val(K), Int32(1))
        BS.dd_fused_spectral_project!(pilot, out_d, n_bins, mask, geom, μ_table, W, Val(K), Int32(1))
        for b in 1:2
            seg = ((b - 1) * ne + 1):(b * ne)
            bs = out_s[seg]
            bd = out_d[seg]
            mm = bs .> 1.0f-6
            relb = abs.(bd[mm] .- bs[mm]) ./ bs[mm]
            @test sum(relb) / length(relb) < 0.02   # bin mean within 2%
        end
    end
end

# -----------------------------------------------------------------------------
# dd_fast fused projectors — per-material path-length variant of the DD fused
# kernels.  The contract is EQUIVALENCE to legacy :dd (same footprint/overlap
# weights, accumulation reassociated by linearity → floating-point-ordering
# differences only), plus agreement with Siddon within the usual DD tolerance.
# -----------------------------------------------------------------------------
@testset "dd_fast fused projectors ≡ dd fused (path-length reassociation)" begin
    geom = _toy_proj_geom(n_cols = 64, n_rows = 8, n_angles = 8, fov_cm = 20.0)
    nx = ny = 64
    nz = 8
    N_E = 8

    # 3 materials so the per-material accumulator is genuinely exercised.
    mask = zeros(UInt16, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - 32.5)^2 + (j - 32.5)^2 <= (0.6 * 32)^2
            mask[i, j, k] = UInt16(1)
        end
        if (i - 45.5)^2 + (j - 32.5)^2 <= 3.0^2
            mask[i, j, k] = UInt16(2)
        end
    end
    μ_table = zeros(Float32, 3, N_E)
    for e in 1:N_E
        μ_table[2, e] = 0.30f0 - 0.12f0 * (e - 1) / (N_E - 1)
        μ_table[3, e] = 1.50f0 - 0.90f0 * (e - 1) / (N_E - 1)
    end
    wη = fill(Float32(1 / N_E), N_E)

    @testset "fused poly: dd_fast ≡ dd (float-ordering only)" begin
        sino_d = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        sino_f = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.dd_fused_poly_project!(sino_d, mask, geom, μ_table, wη, Val(N_E))
        BS.dd_fast_fused_poly_project!(sino_f, mask, geom, μ_table, wη, Val(N_E))
        @test maximum(abs.(sino_f .- sino_d)) < 1.0f-4        # reassociation noise only
    end

    @testset "fused poly: dd_fast tracks Siddon exactly as legacy dd does" begin
        # On this hard-edged rod phantom Siddon-vs-DD legitimately disagree on
        # grazing rays (the aliasing DD exists to fix), so the invariant is:
        # dd_fast deviates from Siddon NO MORE than legacy dd does.
        sino_s = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        sino_d = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        sino_f = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.siddon_fused_poly_project!(sino_s, mask, geom, μ_table, wη, Val(N_E))
        BS.dd_fused_poly_project!(sino_d, mask, geom, μ_table, wη, Val(N_E))
        BS.dd_fast_fused_poly_project!(sino_f, mask, geom, μ_table, wη, Val(N_E))
        m = sino_s .> 1.0f-4
        relΔ_f = abs.(sino_f[m] .- sino_s[m]) ./ sino_s[m]
        relΔ_d = abs.(sino_d[m] .- sino_s[m]) ./ sino_s[m]
        @test count(m) > 1000
        @test sum(relΔ_f) / length(relΔ_f) < 0.01
        @test maximum(relΔ_f) <= maximum(relΔ_d) + 1.0f-4
    end

    @testset "fused spectral: dd_fast ≡ dd (single tile)" begin
        n_bins = Int32(2)
        K = N_E
        W = zeros(Float32, N_E, 2)
        for e in 1:N_E
            W[e, 1] = wη[e] * (1.0f0 - (e - 1) / (N_E - 1))
            W[e, 2] = wη[e] * ((e - 1) / (N_E - 1))
        end
        pilot = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        ne = length(pilot)
        out_d = zeros(Float32, ne * 2)
        out_f = zeros(Float32, ne * 2)
        BS.dd_fused_spectral_project!(pilot, out_d, n_bins, mask, geom, μ_table, W, Val(K), Int32(1))
        BS.dd_fast_fused_spectral_project!(pilot, out_f, n_bins, mask, geom, μ_table, W, Val(K), Int32(1))
        @test maximum(abs.(out_f .- out_d)) < 1.0f-4          # reassociation noise only
    end

    @testset "helical arc geometry + spectral bowtie remain equivalent" begin
        scanner_h = BS.Scanner(
            source_to_isocenter = 540.0, source_to_detector = 1080.0,
            detector_rows = 4, detector_cols = 32,
            detector_row_size = 1.0, detector_col_size = 1.0,
            detector_shape = :arc,
        )
        geom_h = BS.CTGeometry(scanner_h; n_angles = 16, fov_cm = 10.0,
            z_cm = 1.0, pitch = 1.0, n_rotations = 2.0)
        mask_h = mask[17:48, 17:48, 3:6]
        bowtie = Array{Float32}(undef, geom_h.n_cols, geom_h.n_rows, N_E)
        for e in 1:N_E, r in 1:geom_h.n_rows, c in 1:geom_h.n_cols
            bowtie[c, r, e] = 0.7f0 + 0.25f0 * (c - 1) / (geom_h.n_cols - 1) +
                0.03f0 * (e - 1) / (N_E - 1)
        end

        sino_d = zeros(Float32, geom_h.n_cols, geom_h.n_rows, geom_h.n_angles)
        sino_f = similar(sino_d)
        BS.dd_fused_poly_project!(sino_d, mask_h, geom_h, μ_table, wη, Val(N_E);
            volume_extent = (10.0, 10.0, 1.0), ws_bowtie_spectral = bowtie)
        BS.dd_fast_fused_poly_project!(sino_f, mask_h, geom_h, μ_table, wη, Val(N_E);
            volume_extent = (10.0, 10.0, 1.0), ws_bowtie_spectral = bowtie)
        @test maximum(abs.(sino_f .- sino_d)) < 1.0f-4
    end

    @testset "M > 64 materials warns and falls back to the legacy dd kernel" begin
        μ_big = zeros(Float32, 65, N_E)
        μ_big[2, :] .= μ_table[2, :]
        μ_big[3, :] .= μ_table[3, :]
        sino_d = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        sino_f = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        BS.dd_fused_poly_project!(sino_d, mask, geom, μ_big, wη, Val(N_E))
        @test_logs (:warn, r"DD_FAST SINGLE-PASS DISABLED.*65"s) BS.dd_fast_fused_poly_project!(
            sino_f, mask, geom, μ_big, wη, Val(N_E))
        @test sino_f == sino_d                                # same code path, bit-identical
    end
end

# -----------------------------------------------------------------------------
# create_μ_volume! — label → material → μ at energy mapping.
# -----------------------------------------------------------------------------
@testset "create_μ_volume!" begin
    nx, ny, nz = 8, 8, 2
    labels = zeros(UInt8, nx, ny, nz)
    labels[1:4, :, :] .= 0x01    # half = water
    labels[5:8, :, :] .= 0x02    # other half = Ca_100

    # Index 1 = label 0 (air), 2 = label 1 (water), 3 = label 2 (Ca_100).
    materials = [
        BS.XA.Materials.air,
        BS.XA.Materials.water,
        BS.get_material(:Ca_100),
    ]
    E = 80.0
    μ_vol = zeros(Float32, nx, ny, nz)
    BS.create_μ_volume!(μ_vol, labels, materials, E)

    μ_water = Float32(BS.compute_μ_at_energy(BS.XA.Materials.water, E))
    μ_ca100 = Float32(BS.compute_μ_at_energy(BS.get_material(:Ca_100), E))

    # Spot-check both halves carry the right μ.
    @test μ_vol[1, 1, 1] ≈ μ_water
    @test μ_vol[5, 1, 1] ≈ μ_ca100
    # Calcium should attenuate more than water.
    @test μ_ca100 > μ_water

    # Re-running at a different energy gives a different μ.
    μ_vol_2 = zeros(Float32, nx, ny, nz)
    BS.create_μ_volume!(μ_vol_2, labels, materials, 120.0)
    @test μ_vol_2[1, 1, 1] < μ_vol[1, 1, 1]   # water μ drops with E
end

@testset "create_μ_volume! — Float64 / Float32 element type respected" begin
    labels = reshape(UInt8[1, 1, 1, 1], 2, 1, 2)
    materials = [BS.XA.Materials.air, BS.XA.Materials.water]
    μ32 = zeros(Float32, size(labels)...)
    μ64 = zeros(Float64, size(labels)...)
    BS.create_μ_volume!(μ32, labels, materials, 70.0)
    BS.create_μ_volume!(μ64, labels, materials, 70.0)
    @test eltype(μ32) == Float32
    @test eltype(μ64) == Float64
    @test μ32[1, 1, 1] ≈ Float32(μ64[1, 1, 1])
end

# -----------------------------------------------------------------------------
# Projector selection — :dd / :siddon dispatch shims must route to the exact
# underlying projector (bit-identical), and validate the symbol.
# -----------------------------------------------------------------------------
@testset "projector selection (_project_mono / _validate_projector)" begin
    geom = _toy_proj_geom(n_cols = 32, n_rows = 4, n_angles = 4, fov_cm = 10.0)
    vol = fill(Float32(0.15), 32, 32, 4)

    @testset "_validate_projector" begin
        @test BS._validate_projector(:dd) === :dd
        @test BS._validate_projector(:siddon) === :siddon
        @test_throws ArgumentError BS._validate_projector(:bogus)
    end

    @testset "allocating shim routes to the exact projector" begin
        @test BS._project_mono(:dd, vol, geom) == BS.dd_forward_project(vol, geom)
        @test BS._project_mono(:siddon, vol, geom) == BS.siddon_forward_project(vol, geom)
    end

    @testset "in-place shim routes to the exact projector" begin
        sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
        ref_dd = BS.dd_forward_project(vol, geom)
        BS._project_mono!(:dd, sino, vol, geom)
        @test sino == ref_dd

        ref_si = BS.siddon_forward_project(vol, geom)
        fill!(sino, 0.0f0)
        BS._project_mono!(:siddon, sino, vol, geom)
        @test sino == ref_si
    end

    @testset ":dd and :siddon are genuinely different code paths" begin
        # Same line integral (both discretise ∫μ·dl) but not byte-identical.
        s_dd = BS._project_mono(:dd, vol, geom)
        s_si = BS._project_mono(:siddon, vol, geom)
        @test s_dd != s_si                       # different projectors
        mid = geom.n_cols ÷ 2 + 1
        @test isapprox(s_dd[mid, 2, 1], s_si[mid, 2, 1]; rtol = 0.05)  # but close
    end
end

# -----------------------------------------------------------------------------
# Arc (equiangular) detector — projector correctness.
#
# 1. MATLAB-fanbeam-style cross-validation: a FLAT sinogram resampled onto the
#    arc's column grid (col ↦ γ ↦ u = SDD·tanγ) must match the NATIVE arc
#    sinogram (this is the "simple interpolation" conversion — here used as a
#    test oracle rather than as the implementation).
# 2. The three projectors must agree with each other on the arc exactly as
#    they do on the flat panel; dd_fast ≡ dd to float ordering.
# -----------------------------------------------------------------------------
@testset "arc detector — projectors" begin
    scanner_flat = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 128,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = :flat)
    scanner_arc = BS.Scanner(
        source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 8, detector_cols = 128,
        detector_row_size = 1.0, detector_col_size = 1.0,
        detector_shape = :arc)
    gf = BS.CTGeometry(scanner_flat; n_angles = 8, fov_cm = 20.0)
    ga = BS.CTGeometry(scanner_arc; n_angles = 8, fov_cm = 20.0)
    @test BS.is_arc(ga) && !BS.is_arc(gf)

    nx = ny = 64
    nz = 8
    vol = zeros(Float32, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if (i - 32.5)^2 + (j - 32.5)^2 <= (0.6 * 32)^2
            vol[i, j, k] = 0.2f0
        end
        if (i - 45.5)^2 + (j - 32.5)^2 <= 3.0^2
            vol[i, j, k] = 1.0f0
        end
    end

    sino_flat = BS.dd_forward_project(vol, gf)
    sino_arc = BS.dd_forward_project(vol, ga)

    @testset "flat→arc resampling oracle (MATLAB fanbeam-style)" begin
        SAD = gf.SAD; SDD = gf.SDD
        dγ = gf.pixel_size / SAD
        pm = gf.pixel_size * (SDD / SAD)
        nc = gf.n_cols
        cc = (nc + 1) / 2
        resampled = similar(sino_flat)
        for a in 1:size(sino_flat, 3), r in 1:size(sino_flat, 2), i in 1:nc
            γ = (i - cc) * dγ
            colf = SDD * tan(γ) / pm + cc
            c0 = clamp(floor(Int, colf), 1, nc - 1)
            w = clamp(colf - c0, 0.0, 1.0)
            resampled[i, r, a] = (1 - w) * sino_flat[c0, r, a] + w * sino_flat[c0 + 1, r, a]
        end
        # Interior chords only: tangent (grazing) rays have near-singular
        # gradients where ANY resampling shows large relative error — that is
        # interpolation physics, not a geometry defect.  Measured interior:
        # mean 0.25%, p99 2.3%, max 4%.
        m = (sino_arc .> 0.5f0) .& (resampled .> 0.5f0)
        relΔ = sort(abs.(sino_arc[m] .- resampled[m]) ./ resampled[m])
        @test count(m) > 3000
        @test sum(relΔ) / length(relΔ) < 0.005
        @test relΔ[ceil(Int, 0.99 * length(relΔ))] < 0.05
        @test relΔ[end] < 0.10
    end

    @testset "central columns: arc ≈ flat (small-angle limit)" begin
        mid = 57:72                                 # |γ| < 0.03 rad
        Δ = abs.(sino_arc[mid, :, :] .- sino_flat[mid, :, :])
        rel = Δ ./ max.(sino_flat[mid, :, :], 1.0f-3)
        @test maximum(rel[sino_flat[mid, :, :] .> 0.5f0]) < 0.01
    end

    @testset "siddon / dd / dd_fast agree on the arc" begin
        # On this hard-edged rod phantom siddon-vs-dd differ by design (the
        # aliasing DD exists to fix) — the invariant is that the ARC panel
        # behaves like the FLAT panel, not an absolute bound.
        sino_sid = BS.siddon_forward_project(vol, ga)
        m = sino_sid .> 1.0f-3
        relΔ = abs.(sino_arc[m] .- sino_sid[m]) ./ sino_sid[m]
        sino_sid_f = BS.siddon_forward_project(vol, gf)
        mf = sino_sid_f .> 1.0f-3
        relΔ_f = abs.(sino_flat[mf] .- sino_sid_f[mf]) ./ sino_sid_f[mf]
        @test sum(relΔ) / length(relΔ) <= 1.3 * sum(relΔ_f) / length(relΔ_f) + 0.002

        N_E = 4
        mask8 = UInt8.(vol .> 0) .+ UInt8.(vol .> 0.5f0)   # 0 air, 1 water, 2 rod
        μt = zeros(Float32, 3, N_E); μt[2, :] .= 0.2f0; μt[3, :] .= 1.0f0
        wη = fill(Float32(1 / N_E), N_E)
        sd = zeros(Float32, ga.n_cols, ga.n_rows, ga.n_angles)
        sf = zeros(Float32, ga.n_cols, ga.n_rows, ga.n_angles)
        BS.dd_fused_poly_project!(sd, mask8, ga, μt, wη, Val(N_E))
        BS.dd_fast_fused_poly_project!(sf, mask8, ga, μt, wη, Val(N_E))
        @test maximum(abs.(sf .- sd)) < 1.0f-4     # path-length ≡ legacy, on arc
    end
end
