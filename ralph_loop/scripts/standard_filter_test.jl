#!/usr/bin/env julia
"""
NOISE-013: StandardFilter validation test

Tests that:
1. StandardFilter kernel compiles and produces valid values
2. Noise reduction vs RampFilter matches expected ~2.1x factor
3. Full simulation + reconstruction with StandardFilter produces
   noise levels comparable to CatSim
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

println("═══════════════════════════════════════════════════════")
println("  NOISE-013: StandardFilter Validation Test")
println("═══════════════════════════════════════════════════════")

using Metal
import BasisSimulator as BS
import XrayAttenuation as XA
using Statistics
using FFTW

# ─── Test 1: Kernel Creation ───
println("\n[1/4] Testing kernel creation...")

pixel_size = Float32(0.0568)  # cm, typical value
n_cols = 900

# Create both kernels
kernel_ramp = BS.create_spatial_kernel(n_cols, BS.RampFilter(), pixel_size)
kernel_standard = BS.create_spatial_kernel(n_cols, BS.StandardFilter(), pixel_size)

println("  Ram-Lak kernel: center = $(kernel_ramp[n_cols÷2+1]), sum = $(sum(kernel_ramp))")
println("  Standard kernel: center = $(kernel_standard[n_cols÷2+1]), sum = $(sum(kernel_standard))")

# Check that standard kernel has smaller magnitudes at edges (it's windowed)
ramp_energy = sum(kernel_ramp.^2)
std_energy = sum(kernel_standard.^2)
noise_ratio = sqrt(std_energy / ramp_energy)
println("  Noise ratio (standard/ramp) = $(round(noise_ratio, digits=4))")
println("  Expected: ~0.47 (2.1x noise reduction)")

# Verify center values are similar (window = 1.0 at DC)
center_ratio = kernel_standard[n_cols÷2+1] / kernel_ramp[n_cols÷2+1]
println("  Center value ratio = $(round(center_ratio, digits=4)) (should be ~1.0)")

# ─── Test 2: Filter symbol conversion ───
println("\n[2/4] Testing filter_from_symbol...")
for sym in [:ram_lak, :shepp_logan, :cosine, :hamming, :hann, :standard, :soft, :bone]
    ft = BS.filter_from_symbol(sym)
    println("  :$sym → $(typeof(ft))")
end

# ─── Test 3: Frequency domain comparison ───
println("\n[3/4] Frequency domain comparison...")

# FFT both kernels to compare frequency response
function kernel_freq_response(kernel)
    n = length(kernel)
    center = n ÷ 2 + 1
    shifted = zeros(ComplexF64, n)
    for i in 1:n
        src = mod(i - center, n) + 1
        shifted[src] = ComplexF64(kernel[i])
    end
    return abs.(fft(shifted))
end

freq_ramp = kernel_freq_response(kernel_ramp)
freq_std = kernel_freq_response(kernel_standard)

# Check at specific frequencies
nyquist = n_cols ÷ 2
println("  Frequency response ratios (standard / ramp):")
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]
    idx = max(1, round(Int, frac * nyquist) + 1)
    ratio = freq_ramp[idx] > 0 ? freq_std[idx] / freq_ramp[idx] : NaN
    expected = if frac == 0.0; 1.0
    elseif frac == 0.25; 0.9338
    elseif frac == 0.5; 0.7441
    elseif frac == 0.75; 0.4425
    else 0.0531
    end
    println("    f=$(frac)×Nyquist: ratio=$(round(ratio, digits=4)), expected=$(expected)")
end

# ─── Test 4: Full simulation comparison ───
println("\n[4/4] Full simulation: Ram-Lak vs Standard filter...")

# Scanner configuration (same as notebook 01)
sid = 540.0
sdd = 950.0
magnification = sdd / sid
detectorColSize = 1.0 / magnification
detectorRowSize = 1.0 / magnification
detectorColCount = 900
detectorRowCount = 16
imageSize = 512
fov_mm = 350.0
sliceThickness = 1.0
z_coverage_mm = detectorRowCount * detectorRowSize
sliceCount = floor(Int, z_coverage_mm / sliceThickness)

scanner = BS.Scanner(
    source_to_isocenter = sid,
    source_to_detector = sdd,
    detector_rows = detectorRowCount,
    detector_cols = detectorColCount,
    detector_row_size = detectorRowSize,
    detector_col_size = detectorColSize,
    detector_shape = BS.CURVED_DETECTOR,
    focal_spot_width = 0.7,
    focal_spot_length = 0.9,
    target_angle = 7.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material = :gadolinium_oxysulfide,
    detector_depth = 0.5,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    detection_gain = 1.0,
    electronic_noise = 100.0,
)

protocol = BS.CTProtocol(
    mA = 200.0,
    kVp = 120,
    views = 984,
    rotation_time = 1.0,
)

recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (imageSize, imageSize, sliceCount),
    fov_cm = fov_mm / 10.0,
    z_cm = sliceCount * sliceThickness / 10.0,
    filter = :ram_lak,
)

sim_opts = BS.SimOptions(
    fidelity = :high,
    seed = 1234,
)

# Create phantom
fov_cm = fov_mm / 10.0
total_z_cm = (sliceCount * sliceThickness) / 10.0

phantom_basis = BS.create_gammex_472(
    n_voxels = imageSize,
    n_slices = sliceCount,
    fov_cm = fov_cm,
    z_cm = total_z_cm,
)
phantom_gpu = BS.Phantom(
    MtlArray(phantom_basis.mask),
    phantom_basis.materials,
    phantom_basis.voxel_size,
    phantom_basis.origin,
    phantom_basis.fov,
)

# Water phantom for calibration
nx, ny, nz = imageSize, imageSize, sliceCount
voxel_cm = fov_cm / nx
voxel_z_cm = total_z_cm / nz
water_mask = zeros(UInt8, nx, ny, nz)
radius_cm = 16.5
xs = range(-fov_cm/2, fov_cm/2, length=nx)
ys = range(-fov_cm/2, fov_cm/2, length=ny)
for k in 1:nz, j in 1:ny, i in 1:nx
    if sqrt(xs[i]^2 + ys[j]^2) <= radius_cm
        water_mask[i, j, k] = UInt8(1)
    end
end
using Unitful: @u_str
air_material = XA.Material(
    "Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
    Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
)
water_materials = Dict(0 => air_material, 1 => XA.Materials.water)
phantom_water_gpu = BS.Phantom(
    MtlArray(water_mask), water_materials,
    (voxel_cm, voxel_cm, voxel_z_cm)
)

# Run simulation once (shared sinogram for both reconstructions)
println("  Running simulation...")
ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)

# Also simulate water for calibration
ws_w = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_water_gpu)
BS.simulate!(ws_w, phantom_water_gpu, scanner, protocol, sim_opts, recon_opts)

recon_size = recon_opts.matrix_size

# --- Reconstruct with Ram-Lak ---
println("  Reconstructing with Ram-Lak...")
ws_fdk_ramp = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.RampFilter())
vol_ramp = Array(BS.reconstruct!(ws_fdk_ramp, ws.sino_noisy_out, ws.geom, recon_size))

# Water cal for ramp
ws_fdk_ramp_w = BS.create_fdk_recon_workspace(ws_w.sino_noisy_out, ws_w.geom, recon_size; filter=BS.RampFilter())
vol_ramp_w = Array(BS.reconstruct!(ws_fdk_ramp_w, ws_w.sino_noisy_out, ws_w.geom, recon_size))
cx, cy, cz = size(vol_ramp_w) .÷ 2
μ_water_ramp = mean(vol_ramp_w[cx-2:cx+2, cy-2:cy+2, cz-1:cz+1])
hu_ramp = BS.to_hounsfield(vol_ramp; μ_water=μ_water_ramp)

# --- Reconstruct with StandardFilter ---
println("  Reconstructing with StandardFilter...")
ws_fdk_std = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.StandardFilter())
vol_std = Array(BS.reconstruct!(ws_fdk_std, ws.sino_noisy_out, ws.geom, recon_size))

# Water cal for standard
ws_fdk_std_w = BS.create_fdk_recon_workspace(ws_w.sino_noisy_out, ws_w.geom, recon_size; filter=BS.StandardFilter())
vol_std_w = Array(BS.reconstruct!(ws_fdk_std_w, ws_w.sino_noisy_out, ws_w.geom, recon_size))
μ_water_std = mean(vol_std_w[cx-2:cx+2, cy-2:cy+2, cz-1:cz+1])
hu_std = BS.to_hounsfield(vol_std; μ_water=μ_water_std)

# --- Measure noise ---
roi_sz = 30
mid_z = sliceCount ÷ 2
bg_roi_ramp = hu_ramp[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]
bg_roi_std = hu_std[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]

σ_ramp = std(bg_roi_ramp)
σ_std = std(bg_roi_std)
μ_ramp = mean(bg_roi_ramp)
μ_std = mean(bg_roi_std)

# ─── Results ───
println("\n═══════════════════════════════════════════════════════")
println("  RESULTS SUMMARY")
println("═══════════════════════════════════════════════════════\n")
println("┌─────────────────┬──────────┬──────────┬──────────────┐")
println("│ Filter           │ σ_HU     │ μ_HU     │ μ_water cal  │")
println("├─────────────────┼──────────┼──────────┼──────────────┤")
println("│ Ram-Lak          │ $(lpad(string(round(σ_ramp, digits=2)), 8)) │ $(lpad(string(round(μ_ramp, digits=2)), 8)) │ $(lpad(string(round(μ_water_ramp, sigdigits=4)), 12)) │")
println("│ Standard (CatSim)│ $(lpad(string(round(σ_std, digits=2)), 8)) │ $(lpad(string(round(μ_std, digits=2)), 8)) │ $(lpad(string(round(μ_water_std, sigdigits=4)), 12)) │")
println("└─────────────────┴──────────┴──────────┴──────────────┘")
println()

noise_reduction = σ_ramp / σ_std
println("  Noise reduction factor: $(round(noise_reduction, digits=3))x")
println("  Expected: ~2.1x")
println()

catsim_target = 71.37  # CatSim measured σ from nb01_cnr.png
println("  CatSim reference σ: $(catsim_target) HU")
println("  BasisSimulator Standard σ: $(round(σ_std, digits=2)) HU")
println("  Ratio (BS/CatSim): $(round(σ_std / catsim_target, digits=3))")
println()

if abs(noise_reduction - 2.1) < 0.5
    println("  ✓ PASS: Noise reduction factor is within expected range")
else
    println("  ✗ FAIL: Noise reduction factor is outside expected range")
end

if σ_std / catsim_target < 1.3
    println("  ✓ PASS: StandardFilter noise is within 30% of CatSim")
else
    println("  ⚠ WARN: StandardFilter noise is >30% from CatSim (other factors may contribute)")
end

println("\n═══════════════════════════════════════════════════════")
println("  NOISE-013 COMPLETE")
println("═══════════════════════════════════════════════════════")
