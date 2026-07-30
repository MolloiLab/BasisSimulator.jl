# Tests for src/denoising/ — five algorithms:
#
#   LIVE (used by current notebooks nb03/04/07, full behavioral coverage):
#     - sino_sfjsd.jl : 2-channel projection-domain SF-JSD (Black, in prep.)
#     - median_z.jl   : z-direction median filter (BasisSim-original)
#
#   KEPT (advertised but not currently in notebooks, smoke-tested so they
#   don't bit-rot):
#     - sino_svd.jl : N-channel projection-domain SVD joint denoiser
#     - rskr.jl     : Rank-Sparse Kernel Regression (Clark/Badea 2023)
#     - acnr.jl     : Anti-Correlated Noise Reduction (Kalender 1988)

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

# Build a synthetic 2-channel log-line-integral sinogram pair with a single
# vertical edge.  Channel 1 is a "low energy" with stronger attenuation;
# channel 2 a "high energy" with weaker.  Used for SF-JSD + sino_svd.
function _two_channel_sino(;
        n_col = 64, n_row = 4, n_view = 32,
        p_lo_max = 1.5, p_hi_max = 0.8, seed = 2026
    )
    Random.seed!(seed)
    sino_lo = zeros(Float32, n_col, n_row, n_view)
    sino_hi = zeros(Float32, n_col, n_row, n_view)
    # Half-of-fan attenuator: cols 1..n_col/2 see full path, rest see zero.
    half = n_col ÷ 2
    sino_lo[1:half, :, :] .= Float32(p_lo_max)
    sino_hi[1:half, :, :] .= Float32(p_hi_max)
    # Add Poisson-flavored Gaussian noise via the log-Poisson approximation
    # σ_p ≈ 1 / sqrt(N).  At I0 = 1e5 and p ≈ 1, N ≈ 3.7e4 → σ ≈ 0.005.
    σ_p_lo = 0.02f0
    σ_p_hi = 0.015f0
    sino_lo .+= Float32.(σ_p_lo .* randn(Float32, size(sino_lo)))
    sino_hi .+= Float32.(σ_p_hi .* randn(Float32, size(sino_hi)))
    return sino_lo, sino_hi
end

# Single-channel 3D volume with a centered cube of value +1 in a zero
# background; useful for median_z and ACNR/RSKR shape tests.
function _cube_volume(; n = 16, nz = 8, val = 1.0f0)
    vol = zeros(Float32, n, n, nz)
    c = n ÷ 2
    vol[(c - 1):(c + 1), (c - 1):(c + 1), :] .= val
    return vol
end

# -----------------------------------------------------------------------------
# SF-JSD — full behavioral tests (LIVE).
# -----------------------------------------------------------------------------
@testset "apply_sino_sfjsd_denoise (SF-JSD)" begin
    sino_lo, sino_hi = _two_channel_sino()
    I0 = [1.0e5, 8.0e4]

    @testset "shape contract" begin
        @test_throws ErrorException BS.apply_sino_sfjsd_denoise(
            [sino_lo], I0; σ₀ = 1.0, verbose = false,
        )  # 1 channel
        @test_throws ErrorException BS.apply_sino_sfjsd_denoise(
            [sino_lo, sino_hi, sino_lo], I0[1:2]; σ₀ = 1.0, verbose = false,
        )  # 3 channels
        @test_throws ErrorException BS.apply_sino_sfjsd_denoise(
            [sino_lo, sino_hi], [1.0e5]; σ₀ = 1.0, verbose = false,
        )  # mismatched I0
        @test_throws ErrorException BS.apply_sino_sfjsd_denoise(
            [sino_lo, zeros(Float32, 8, 4, 32)], I0; σ₀ = 1.0, verbose = false,
        )  # mismatched shapes
    end

    @testset "explicit σ₀ > 0 (SURE skipped)" begin
        out = BS.apply_sino_sfjsd_denoise(
            [sino_lo, sino_hi], I0; σ₀ = 1.5, verbose = false,
        )
        @test length(out) == 2
        @test size(out[1]) == size(sino_lo)
        @test size(out[2]) == size(sino_hi)
        @test eltype(out[1]) == Float32
        @test all(isfinite, out[1])
        @test all(isfinite, out[2])
    end

    @testset "noise reduction on a flat ROI" begin
        # Pick a flat region (cols 1..n_col/2 = constant attenuator), compute
        # std before/after. After SF-JSD the local σ must drop noticeably.
        out = BS.apply_sino_sfjsd_denoise(
            [sino_lo, sino_hi], I0; σ₀ = 2.0, verbose = false,
        )
        # Interior flat region — stay away from the edge transition.
        roi_in_lo = sino_lo[5:25, :, :]
        roi_out_lo = out[1][5:25, :, :]
        roi_in_hi = sino_hi[5:25, :, :]
        roi_out_hi = out[2][5:25, :, :]
        @test std(roi_out_lo) < std(roi_in_lo)
        @test std(roi_out_hi) < std(roi_in_hi)
    end

    @testset "edge preservation (mean-of-flat-ROI is preserved)" begin
        # The mean across the flat attenuator region should be approximately
        # the same before and after — SF-JSD is unbiased in expectation.
        out = BS.apply_sino_sfjsd_denoise(
            [sino_lo, sino_hi], I0; σ₀ = 2.0, verbose = false,
        )
        @test abs(mean(out[1][5:25, :, :]) - mean(sino_lo[5:25, :, :])) < 0.05
        @test abs(mean(out[2][5:25, :, :]) - mean(sino_hi[5:25, :, :])) < 0.05
    end

    @testset "σ₀ = 0 invokes SURE auto-select" begin
        # Small grid keeps SURE fast (golden-section ~10 evaluations).
        sino_lo_s, sino_hi_s = _two_channel_sino(; n_col = 32, n_row = 2, n_view = 16)
        out = BS.apply_sino_sfjsd_denoise(
            [sino_lo_s, sino_hi_s], I0; σ₀ = 0.0, verbose = false,
        )
        @test length(out) == 2
        @test all(isfinite, out[1])
        @test all(isfinite, out[2])
    end
end

# -----------------------------------------------------------------------------
# median_z — full behavioral tests (LIVE).
# -----------------------------------------------------------------------------
@testset "apply_median_z / apply_median_z!" begin
    @testset "adjacent_slices = 0 → identity (copy)" begin
        vol = _cube_volume()
        out = BS.apply_median_z(vol; adjacent_slices = 0)
        @test out == vol
        @test out !== vol  # identity but fresh allocation
    end

    @testset "removes single-slice xy impulses" begin
        # Z-invariant signal + a single planted impulse in one slice.
        vol = _cube_volume()
        # Plant a 100x spike at (3, 3, 4) — not on the cube. Should be
        # wiped because z-neighbors (slices 3, 5) are zero.
        vol[3, 3, 4] = 100.0f0
        out = BS.apply_median_z(vol; adjacent_slices = 1)
        @test out[3, 3, 4] == 0.0f0   # impulse removed
        # In-plane structure preserved at non-impulse locations.
        c = size(vol, 1) ÷ 2
        @test out[c, c, 4] ≈ 1.0f0    # cube voxel still 1.0
    end

    @testset "preserves z-correlated signal" begin
        # Z-invariant volume — median over any z-window returns the same value.
        vol = _cube_volume()
        for r in (1, 2, 3)
            out = BS.apply_median_z(vol; adjacent_slices = r)
            @test out ≈ vol
        end
    end

    @testset "in-place vs allocating return same result" begin
        vol = _cube_volume()
        vol[3, 3, 4] = 100.0f0
        a = BS.apply_median_z(vol; adjacent_slices = 1)
        b = similar(vol)
        BS.apply_median_z!(b, vol; adjacent_slices = 1)
        @test a == b
    end

    @testset "shape mismatch + negative adjacent_slices error" begin
        vol = _cube_volume()
        bad_out = similar(vol, size(vol, 1) + 1, size(vol, 2), size(vol, 3))
        @test_throws ErrorException BS.apply_median_z!(bad_out, vol)
        @test_throws ErrorException BS.apply_median_z(vol; adjacent_slices = -1)
        @test_throws ErrorException BS.apply_median_z!(similar(vol), vol; adjacent_slices = -1)
    end

    @testset "in-place mutates out, leaves src unchanged" begin
        vol = _cube_volume()
        vol[3, 3, 4] = 100.0f0
        src_copy = copy(vol)
        out = similar(vol)
        BS.apply_median_z!(out, vol; adjacent_slices = 1)
        @test vol == src_copy
        @test out != src_copy   # something changed
        @test out[3, 3, 4] == 0.0f0
    end
end

# -----------------------------------------------------------------------------
# sino_svd — smoke tests (KEPT).  Predecessor of SF-JSD.
# -----------------------------------------------------------------------------
@testset "apply_sino_svd_denoise / apply_sino_svd_denoise!" begin
    sino_lo, sino_hi = _two_channel_sino(; n_col = 32, n_row = 2, n_view = 16)

    @testset "σ_px = 0 → passthrough copy" begin
        out = BS.apply_sino_svd_denoise([sino_lo, sino_hi]; σ_px = 0)
        @test length(out) == 2
        @test out[1] == sino_lo
        @test out[2] == sino_hi
    end

    @testset "σ_px > 0 reduces variance in a flat ROI" begin
        out = BS.apply_sino_svd_denoise([sino_lo, sino_hi]; σ_px = 2.0)
        @test length(out) == 2
        @test size(out[1]) == size(sino_lo)
        # Variance should drop on the flat region (post-SVD residual smoothed).
        @test std(out[1][5:25, :, :]) < std(sino_lo[5:25, :, :])
    end

    @testset "N = 4 channels works (no special-case path)" begin
        sino_3 = copy(sino_lo) .+ 0.01f0
        sino_4 = copy(sino_hi) .+ 0.01f0
        out = BS.apply_sino_svd_denoise([sino_lo, sino_hi, sino_3, sino_4]; σ_px = 1.0)
        @test length(out) == 4
        for o in out
            @test size(o) == size(sino_lo)
            @test all(isfinite, o)
        end
    end

    @testset "shape contract" begin
        # < 2 channels errors
        @test_throws ErrorException BS.apply_sino_svd_denoise([sino_lo]; σ_px = 1.0)
        # mismatched channel shapes errors
        bad = zeros(Float32, 8, 2, 16)
        @test_throws ErrorException BS.apply_sino_svd_denoise([sino_lo, bad]; σ_px = 1.0)
        # length(out) ≠ length(channels) errors
        @test_throws ErrorException BS.apply_sino_svd_denoise!(
            [similar(sino_lo)], [sino_lo, sino_hi]; σ_px = 1.0,
        )
    end

    @testset "in-place vs allocating return same result" begin
        out_a = BS.apply_sino_svd_denoise([sino_lo, sino_hi]; σ_px = 1.0)
        out_b = [similar(sino_lo) for _ in 1:2]
        BS.apply_sino_svd_denoise!(out_b, [sino_lo, sino_hi]; σ_px = 1.0)
        @test maximum(abs.(out_a[1] .- out_b[1])) < 1.0e-5
        @test maximum(abs.(out_a[2] .- out_b[2])) < 1.0e-5
    end

    @testset "bilateral remains finite on constant channels" begin
        # Constant/linearly dependent channels yield zero-MAD residual SVD
        # components. This specifically guards against 0*Inf in the bilateral
        # range exponent.
        flat = fill(0.5f0, 16, 2, 8)
        out = BS.apply_sino_svd_denoise_bilateral(
            [flat,2f0 .* flat,3f0 .* flat,4f0 .* flat];
            bilat_radius=2,bilat_sigma_s=1.5,bilat_range_k=2.0,
        )
        @test all(o -> all(isfinite,o),out)
        @test all(o -> maximum(abs,o) > 0,out)
    end
end

# -----------------------------------------------------------------------------
# RSKR — smoke tests (KEPT).  Clark/Badea 2023.
# -----------------------------------------------------------------------------
@testset "apply_rskr (Clark/Badea 2023)" begin
    n = 16; nz = 4
    Random.seed!(2026)
    vol_lo = Float32.(_cube_volume(; n = n, nz = nz, val = 1.0f0) .+ 0.02 .* randn(n, n, nz))
    vol_hi = Float32.(_cube_volume(; n = n, nz = nz, val = 0.6f0) .+ 0.02 .* randn(n, n, nz))

    @testset "2-channel CPU smoke" begin
        out = BS.apply_rskr(
            [vol_lo, vol_hi];
            n_iter = 1, h_param = 1.0, radius = 1,
            gpu_arr_type = identity, verbose = false,
        )
        @test length(out) == 2
        @test size(out[1]) == size(vol_lo)
        @test size(out[2]) == size(vol_hi)
        @test eltype(out[1]) == Float32
        @test all(isfinite, out[1])
        @test all(isfinite, out[2])
    end

    @testset "4-channel CPU smoke" begin
        vol3 = copy(vol_lo) .+ 0.01f0
        vol4 = copy(vol_hi) .+ 0.01f0
        out = BS.apply_rskr(
            [vol_lo, vol_hi, vol3, vol4];
            n_iter = 1, h_param = 1.0, radius = 1,
            gpu_arr_type = identity, verbose = false,
        )
        @test length(out) == 4
        for o in out
            @test size(o) == size(vol_lo)
            @test all(isfinite, o)
        end
    end

    @testset "n_iter ≥ 2 still finite" begin
        out = BS.apply_rskr(
            [vol_lo, vol_hi];
            n_iter = 2, h_param = 1.0, radius = 1,
            gpu_arr_type = identity, verbose = false,
        )
        @test all(isfinite, out[1])
        @test all(isfinite, out[2])
    end

    @testset "shape contract — only 2 or 4 channels allowed" begin
        @test_throws ErrorException BS.apply_rskr(
            [vol_lo, vol_hi, vol_lo];
            gpu_arr_type = identity, verbose = false,
        )
        @test_throws ErrorException BS.apply_rskr(
            [vol_lo];
            gpu_arr_type = identity, verbose = false,
        )
    end

    @testset "mad_haar_σ returns finite positive scalar" begin
        σ = BS.mad_haar_σ(vol_lo)
        @test σ isa Float32
        @test isfinite(σ)
        @test σ ≥ 0
        # Zero-volume → MAD = 0
        @test BS.mad_haar_σ(zeros(Float32, 8, 8, 4)) == 0.0f0
    end
end

# -----------------------------------------------------------------------------
# ACNR — smoke tests (KEPT).  Kalender/Klotz/Kostaridou 1988.
# -----------------------------------------------------------------------------
@testset "apply_acnr / apply_acnr! (Kalender 1988)" begin
    sino_a = Float32.(0.5 .+ 0.1 .* randn(MersenneTwister(0), 32, 4, 16))
    sino_b = Float32.(0.3 .+ 0.1 .* randn(MersenneTwister(1), 32, 4, 16))

    @testset "Gaussian smoother σ keyword path" begin
        a_in = copy(sino_a); b_in = copy(sino_b)
        a_out, b_out, info = BS.apply_acnr(
            sino_a, sino_b; c_a = 0.7, c_b = 0.3, σ = 2.0, verbose = false,
        )
        # Returns a tuple of (sino_a_corr, sino_b_corr, diagnostics).
        @test size(a_out) == size(sino_a)
        @test size(b_out) == size(sino_b)
        @test all(isfinite, a_out)
        @test all(isfinite, b_out)
        # Originals untouched (allocating wrapper).
        @test sino_a == a_in
        @test sino_b == b_in
        @test info.smoother isa String
        @test info.σ_n_orth ≥ 0
        @test info.σ_s_smooth ≥ 0
    end

    @testset "Tikhonov smoother λ keyword path" begin
        _, _, info = BS.apply_acnr(
            sino_a, sino_b; c_a = 0.7, c_b = 0.3, λ = 4.0, verbose = false,
        )
        @test occursin("Tikhonov", info.smoother)
    end

    @testset "γ = 0 ⇒ identity (no correction applied)" begin
        a_in = copy(sino_a); b_in = copy(sino_b)
        a_out, b_out, _ = BS.apply_acnr(
            sino_a, sino_b; c_a = 0.7, c_b = 0.3, σ = 2.0, γ = 0.0,
            verbose = false,
        )
        @test maximum(abs.(a_out .- a_in)) < 1.0e-5
        @test maximum(abs.(b_out .- b_in)) < 1.0e-5
    end

    @testset "in-place version mutates" begin
        a = copy(sino_a); b = copy(sino_b)
        BS.apply_acnr!(
            a, b; c_a = 0.7, c_b = 0.3, σ = 2.0, verbose = false,
        )
        @test a != sino_a   # something changed
        @test b != sino_b
    end

    @testset "must specify exactly one of σ or λ" begin
        @test_throws ErrorException BS.apply_acnr(
            sino_a, sino_b; c_a = 0.7, c_b = 0.3, verbose = false,
        )  # neither
        @test_throws ErrorException BS.apply_acnr(
            sino_a, sino_b; c_a = 0.7, c_b = 0.3, σ = 2.0, λ = 4.0, verbose = false,
        )  # both
    end

    @testset "(c_a, c_b) both zero errors" begin
        @test_throws ErrorException BS.apply_acnr(
            sino_a, sino_b; c_a = 0.0, c_b = 0.0, σ = 2.0, verbose = false,
        )
    end
end
