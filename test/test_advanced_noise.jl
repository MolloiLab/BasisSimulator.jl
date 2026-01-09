"""
Test advanced noise models (1/f noise and NPS computation).

This script verifies that:
1. 1/f noise has correct power spectral density
2. NPS computation produces valid results
3. Both integrate with full CT pipeline
"""

using BasisSimulator
using Statistics
using FFTW

println("\n" * "="^70)
println("ADVANCED NOISE MODELS TEST")
println("="^70)

# ==============================================================================
# 1. Test 1/f Noise Generation
# ==============================================================================
println("\n1. Testing 1/f noise generation...")

# Create test signal (detector dimensions)
n_rows, n_cols, n_angles = 64, 64, 100
signal = ones(Float64, n_rows, n_cols, n_angles) .* 1e8  # Constant signal

# Test pink noise (α=1)
pink_noise_signal = add_1_over_f_noise(
    signal,
    alpha=1.0,
    amplitude=1000.0,
    seed=42
)

# Check that noise was added
noise_added = pink_noise_signal .- signal
mean_noise = mean(noise_added)
std_noise = std(noise_added)

println("   ✅ Pink noise (α=1.0) applied")
println("      Mean noise: $(round(mean_noise, digits=2)) (should be ~0)")
println("      Std noise: $(round(std_noise, digits=2)) (should be ~1000)")
println("      Signal range: $(round(minimum(pink_noise_signal), digits=0)) to $(round(maximum(pink_noise_signal), digits=0))")

# Test brown noise (α=2)
brown_noise_signal = add_1_over_f_noise(
    signal,
    alpha=2.0,
    amplitude=500.0,
    seed=43
)

brown_noise = brown_noise_signal .- signal
println("   ✅ Brown noise (α=2.0) applied")
println("      Std noise: $(round(std(brown_noise), digits=2)) (should be ~500)")

# Verify power spectral density for 1D sequence
println("\n2. Verifying 1/f power spectral density...")

# Generate 1/f sequence
n = 1024
alpha_test = 1.0
sequence_1f = BasisSimulator.generate_1_over_f_sequence(n, alpha_test)

# Compute power spectrum
fft_seq = fft(sequence_1f)
power_spectrum = abs2.(fft_seq[1:div(n,2)])
freqs = (0:(div(n,2)-1)) ./ n

# Check that power follows 1/f law (skip DC component)
# Log-log slope should be ≈ -α
log_freqs = log10.(freqs[2:end])
log_power = log10.(power_spectrum[2:end])

# Linear fit in log-log space
A = hcat(ones(length(log_freqs)), log_freqs)
coeffs = A \ log_power
slope = coeffs[2]

println("   ✅ Power spectral density analyzed")
println("      Expected slope: -$(alpha_test)")
println("      Measured slope: $(round(slope, digits=2))")
println("      Deviation: $(round(abs(slope + alpha_test), digits=2))")

# ==============================================================================
# 3. Test NPS Computation
# ==============================================================================
println("\n3. Testing NPS computation...")

# Create noisy image
image_size = 256
base_image = zeros(Float64, image_size, image_size)

# Add Gaussian noise
noise_std = 50.0
noisy_image = base_image .+ noise_std .* randn(image_size, image_size)

# Compute NPS
nps = compute_nps(noisy_image, roi_size=64, n_rois=50, detrend=true)

# Check NPS properties
nps_mean = mean(nps)
nps_max = maximum(nps)
nps_std = std(nps)

println("   ✅ NPS computed")
println("      NPS size: $(size(nps))")
println("      NPS mean: $(round(nps_mean, digits=2))")
println("      NPS max: $(round(nps_max, digits=2))")
println("      NPS std: $(round(nps_std, digits=2))")

# Verify NPS integrates to noise variance (approximately)
# σ² ≈ sum(NPS) * df² where df = 1/roi_size
roi_size = 64
df = 1.0 / roi_size
integrated_nps = sum(nps) * df^2

println("      Expected variance: $(round(noise_std^2, digits=2))")
println("      Integrated NPS: $(round(integrated_nps, digits=2))")
println("      Ratio: $(round(integrated_nps / noise_std^2, digits=2))")

# ==============================================================================
# 4. Test NPS Detrending
# ==============================================================================
println("\n4. Testing NPS detrending...")

# Create image with linear trend
x_coords = repeat(1:image_size, 1, image_size)'
y_coords = repeat(1:image_size, 1, image_size)
trend_image = 100.0 .+ 0.5 .* x_coords .+ 0.3 .* y_coords

# Detrend
detrended = BasisSimulator.detrend_2d(trend_image)

# Check that trend is removed
trend_mean = mean(detrended)
trend_std = std(detrended)

println("   ✅ Detrending tested")
println("      Original mean: $(round(mean(trend_image), digits=2))")
println("      Detrended mean: $(round(trend_mean, digits=4)) (should be ~0)")
println("      Detrended std: $(round(trend_std, digits=4)) (should be ~0)")

# ==============================================================================
# 5. Full Pipeline Integration Test
# ==============================================================================
println("\n5. Testing full pipeline integration...")

# Create small phantom
phantom = create_water_cylinder(
    diameter_mm=100.0,
    height_mm=20.0,
    resolution_mm=2.0
)

# Generate spectrum
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

# Create geometry (minimal projections for speed)
protocol = ScanProtocol(
    kVp=120.0,
    mAs=200.0,
    scan_fov_mm=100.0,
    num_projections=10
)
geometry = create_aquilion_one(protocol=protocol)

# Run simulation
detector_signal = simulate_ct_scan(
    phantom=phantom,
    geometry=geometry,
    spectrum=spectrum
)

println("   ✅ Baseline simulation complete")
println("      Signal shape: $(size(detector_signal))")
println("      Signal mean: $(round(mean(detector_signal), digits=0))")

# Apply combined noise:  Poisson + Electronic + 1/f
signal_poisson = apply_poisson_noise(detector_signal, dose_factor=0.5, seed=42)
signal_elec = add_electronic_noise(signal_poisson, sigma=1000.0, seed=43)
signal_full_noise = add_1_over_f_noise(signal_elec, alpha=1.0, amplitude=500.0, seed=44)

println("   ✅ All noise models applied")
println("      Poisson relative change: $(round(std(signal_poisson .- detector_signal) / mean(detector_signal), digits=4))")
println("      Electronic added std: $(round(std(signal_elec .- signal_poisson), digits=1))")
println("      1/f added std: $(round(std(signal_full_noise .- signal_elec), digits=1))")

# Reconstruct with noisy data
I0 = estimate_air_scan(spectrum)
sinogram_noisy = convert_to_attenuation(signal_full_noise, I0)

# Simple FDK reconstruction
fov_cm = 10.0
image_size_recon = 64
recon_x = collect(range(-fov_cm/2, fov_cm/2, length=image_size_recon))
recon_y = collect(range(-fov_cm/2, fov_cm/2, length=image_size_recon))
recon_z = collect(range(-1.0, 1.0, length=10))

volume_noisy = reconstruct_fdk(
    sinogram_noisy,
    geometry.SAD_cm,
    geometry.SDD_cm,
    geometry.pixel_width_cm,
    geometry.pixel_height_cm,
    geometry.angles,
    recon_x,
    recon_y,
    recon_z,
    filter_type=ramlak
)

println("   ✅ Reconstruction with all noise models complete")
println("      Volume shape: $(size(volume_noisy))")
println("      Volume range: $(round(minimum(volume_noisy), digits=2)) to $(round(maximum(volume_noisy), digits=2)) cm^-1")

# Compute NPS from reconstructed slice
center_slice = volume_noisy[:, :, div(end, 2)]
nps_recon = compute_nps(center_slice, roi_size=32, n_rois=20, detrend=true)

println("   ✅ NPS computed from reconstruction")
println("      NPS mean: $(round(mean(nps_recon), digits=4))")
println("      NPS max: $(round(maximum(nps_recon), digits=4))")

# ==============================================================================
# 6. Validation Summary
# ==============================================================================
println("\n" * "="^70)
println("VALIDATION SUMMARY")
println("="^70)

checks = []

# 1/f noise
push!(checks, ("Pink noise std", std_noise, 800.0, 1200.0, ""))
push!(checks, ("1/f power law slope", abs(slope + alpha_test), 0.0, 0.3, ""))

# NPS
push!(checks, ("NPS mean > 0", nps_mean > 0, true, true, ""))
push!(checks, ("Detrended mean ~0", abs(trend_mean), 0.0, 1e-10, ""))

# Pipeline
push!(checks, ("Reconstruction completed", !isnan(mean(volume_noisy)), true, true, ""))
push!(checks, ("NPS from reconstruction", nps_recon !== nothing && !any(isnan.(nps_recon)), true, true, ""))

println()
global all_passed = true
for (name, value, min_val, max_val, unit) in checks
    if value isa Bool
        passed = value == min_val
        status = passed ? "✅" : "❌"
        println("$status $name")
    else
        passed = min_val <= value <= max_val
        status = passed ? "✅" : "❌"
        unit_str = isempty(unit) ? "" : " $unit"
        println("$status $name: $(round(value, digits=3))$unit_str (expected: $(min_val)-$(max_val))")
    end
    global all_passed = all_passed && passed
end

println("\n" * "="^70)
if all_passed
    println("✅ ALL ADVANCED NOISE TESTS PASSED")
else
    println("❌ SOME TESTS FAILED - REQUIRES INVESTIGATION")
end
println("="^70 * "\n")
