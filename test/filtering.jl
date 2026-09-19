# The FBP kernel: exact frequency-domain windows, and the bandlimit that ties them to the image
# grid rather than to the detector.
using FFTW: fft

@testset "frequency windows" begin
    for F in (BS.RampFilter(), BS.SheppLoganFilter(), BS.CosineFilter(), BS.HammingFilter(),
            BS.HannFilter(), BS.StandardFilter(), BS.SoftFilter(), BS.BoneFilter())
        @test BS.frequency_window(F, 0.0) ≈ 1.0
    end
    @test BS.frequency_window(BS.SheppLoganFilter(), 1.0) ≈ 2 / π
    @test BS.frequency_window(BS.CosineFilter(), 1.0) ≈ 0.0 atol = 1e-12
    @test BS.frequency_window(BS.HammingFilter(), 1.0) ≈ 0.08
    @test BS.frequency_window(BS.HannFilter(), 1.0) ≈ 0.0 atol = 1e-12
    @test BS.frequency_window(BS.StandardFilter(), 0.5) ≈ 0.7441
    @test BS.frequency_window(BS.SoftFilter(), 1.0) ≈ 0.0
    @test BS.frequency_window(BS.CustomFilter((0.0, 1.0), (1.0, 0.5)), 0.5) ≈ 0.75
end

@testset "kernel spectrum: window stretched to the bandlimit, zero above it" begin
    n, Δ = 511, 0.03f0
    nyquist = n / 2
    spectrum(k) = abs.(fft(Complex{Float64}.(circshift(k, -(n ÷ 2)))))
    at(spec, f_norm) = spec[round(Int, f_norm * nyquist) + 1]
    for F in (BS.RampFilter(), BS.StandardFilter(), BS.SoftFilter(), BS.HannFilter(), BS.SheppLoganFilter())
        full = BS.create_spatial_kernel(n, F, Δ)
        @test full == BS.create_spatial_kernel(n, F, Δ; bandlimit = 1.0)
        # a ramp has no DC: what remains is the truncated tail of the odd taps, 2/(π²Δ)·Σ_{k odd > n/2} 1/k²
        tail = 2 / (π^2 * Δ) * sum(1 / k^2 for k in (n ÷ 2 + 1):2:2_000_001)
        @test abs(sum(full)) ≈ tail rtol = 0.05
        b = 0.4
        limited = BS.create_spatial_kernel(n, F, Δ; bandlimit = b)
        s_full, s_lim = spectrum(full), spectrum(limited)
        # nothing above the bandlimit
        @test maximum(s_lim[round(Int, 0.43 * nyquist):(n ÷ 2 + 1)]) < 2e-3 * maximum(s_full)
        # below it, the same ramp times the window read at f / b instead of f
        for f in (0.1, 0.2, 0.3)
            expected = BS.frequency_window(F, f / b) / BS.frequency_window(F, f)
            @test at(s_lim, f) / at(s_full, f) ≈ expected rtol = 0.03
        end
    end
    @test_throws ArgumentError BS.create_spatial_kernel(n, BS.StandardFilter(), Δ; bandlimit = 0.0)
    @test_throws ArgumentError BS.create_spatial_kernel(n, BS.StandardFilter(), Δ; bandlimit = 1.2)
end

@testset "grid_bandlimit from the geometry" begin
    fine = BS.EICTScanner(source_to_isocenter = 610.0, source_to_detector = 1113.0,
        detector_rows = 4, detector_cols = 1200, detector_row_size = 0.35, detector_col_size = 0.30)
    geom = BS.CTGeometry(fine; n_angles = 8, fov_cm = 35.0, z_cm = 0.14)
    @test BS.grid_bandlimit(geom, (512, 512, 1)) ≈ 0.030 / (35.0 / 512) rtol = 1e-9
    @test BS.grid_bandlimit(geom, (2048, 2048, 1)) == 1.0                  # grid finer than the detector
    @test BS.grid_bandlimit(geom, (512, 256, 1)) ≈ 0.030 / (35.0 / 256)   # the coarser axis decides
    @test BS.grid_bandlimit(geom, (512, 512, 1); ray_spacing = 0.1) == 1.0
    coarse = BS.EICTScanner(source_to_isocenter = 540.0, source_to_detector = 1080.0,
        detector_rows = 4, detector_cols = 300, detector_row_size = 1.0, detector_col_size = 1.0)
    geom_c = BS.CTGeometry(coarse; n_angles = 8, fov_cm = 20.0, z_cm = 0.4)
    @test BS.grid_bandlimit(geom_c, (128, 128, 1)) ≈ 0.64        # 1.0 mm detector, 1.56 mm grid
    @test BS.grid_bandlimit(geom_c, (256, 256, 1)) == 1.0         # 0.78 mm grid, finer than the 1.0 mm detector
end

@testset "FDK on a fine detector: same water value, less noise, and the coarse case unchanged" begin
    # a centred water disk, its fan-beam line integrals written down exactly, plus white noise
    scanner = BS.EICTScanner(source_to_isocenter = 600.0, source_to_detector = 1100.0,
        detector_rows = 4, detector_cols = 256, detector_row_size = 1.0, detector_col_size = 0.5)
    geom = BS.CTGeometry(scanner; n_angles = 180, fov_cm = 12.0, z_cm = 0.4)
    μ, R = 0.2, 4.0
    dγ = geom.pixel_size / geom.SAD
    sino = zeros(Float32, geom.n_cols, geom.n_rows, geom.n_angles)
    for c in 1:geom.n_cols
        d = geom.SAD * sin((c - (geom.n_cols + 1) / 2) * dγ)
        chord = abs(d) < R ? 2 * sqrt(R^2 - d^2) : 0.0
        sino[c, :, :] .= Float32(μ * chord)
    end
    rng = Random.MersenneTwister(3)
    noisy = sino .+ 0.02f0 .* randn(rng, Float32, size(sino))
    volume_size = (64, 64, 2)                                 # 1.875 mm pixels on a 0.5 mm detector
    @test BS.grid_bandlimit(geom, volume_size) ≈ 0.05 / (12.0 / 64) rtol = 1e-9
    limited = BS.fdk_reconstruct(noisy, geom, volume_size)
    unlimited = BS.backproject(
        BS.filter_sinogram(noisy, geom; bandlimit = 1.0), geom, volume_size)
    x = ((1:64) .- 32.5) .* (12.0 / 64)
    inside = [sqrt(x[i]^2 + x[j]^2) < 0.6R for i in 1:64, j in 1:64]
    m_lim, m_unl = mean(limited[:, :, 1][inside]), mean(unlimited[:, :, 1][inside])
    @test m_lim ≈ μ rtol = 0.03
    @test m_lim ≈ m_unl rtol = 0.01                            # the bandlimit changes no CT number
    @test std(limited[:, :, 1][inside]) < 0.7 * std(unlimited[:, :, 1][inside])
    # a grid finer than the detector (0.47 mm pixels on a 0.5 mm detector) reconstructs exactly as before
    fine_size = (256, 256, 2)
    @test BS.grid_bandlimit(geom, fine_size) == 1.0
    ref = BS.backproject(BS.filter_sinogram(noisy, geom; bandlimit = 1.0), geom, fine_size)
    @test BS.fdk_reconstruct(noisy, geom, fine_size; mask_fov = false) ≈ ref
end
