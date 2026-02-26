#!/usr/bin/env julia
"""
Test: FFT-based vs Spatial-domain ramp filtering

Does our spatial domain convolution produce different noise than FFT-based filtering?
This directly tests whether the filtering approach is the noise discrepancy source.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Metal
import BasisSimulator as BS
using Statistics
using Random
using FFTW

println("═══════════════════════════════════════════════════════")
println("  FFT vs Spatial Domain Ramp Filter Test")
println("═══════════════════════════════════════════════════════")

# Setup geometry
sid = 540.0
sdd = 950.0
magnification = sdd / sid
detectorColSize = 1.0 / magnification
detectorRowSize = 1.0 / magnification

scanner = BS.Scanner(
    source_to_isocenter = sid, source_to_detector = sdd,
    detector_rows = 16, detector_cols = 900,
    detector_row_size = detectorColSize, detector_col_size = detectorColSize,
    detector_shape = BS.CURVED_DETECTOR,
    flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
    detector_material = :gadolinium_oxysulfide, detector_depth = 0.5,
)
protocol = BS.CTProtocol(mA=200.0, kVp=120, views=984, rotation_time=1.0)
recon_opts = BS.ReconOptions(
    algorithm = :fdk, matrix_size = (512, 512, 9),
    fov_cm = 35.0, z_cm = 9 * 1.0 / 10.0, filter = :ram_lak,
)
geom = BS.CTGeometry(scanner; n_angles=984, fov_cm=35.0, z_cm=0.9)

n_cols = 900
n_rows = 16
n_angles = 984
pixel_size_cm = Float64(geom.pixel_size)

println("  pixel_size = $(round(pixel_size_cm, sigdigits=5)) cm")
println("  n_cols = $n_cols, n_rows = $n_rows, n_angles = $n_angles")

# ═══════════════════════════════════════════════════════════
# Create test sinogram: uniform + noise
# ═══════════════════════════════════════════════════════════
Random.seed!(42)
sino_mean = 6.8  # typical water path
σ_noise = 0.045  # typical sinogram noise

sino_cpu = fill(Float32(sino_mean), n_cols, n_rows, n_angles) .+
           Float32(σ_noise) .* randn(Float32, n_cols, n_rows, n_angles)

println("\n[1] Sinogram: mean=$(round(mean(sino_cpu), digits=4)), std=$(round(std(sino_cpu), sigdigits=4))")

# ═══════════════════════════════════════════════════════════
# Method 1: Our spatial domain filtering (GPU)
# ═══════════════════════════════════════════════════════════
println("\n[2] Spatial domain filtering (our code)...")
sino_gpu = MtlArray(copy(sino_cpu))
recon_size = recon_opts.matrix_size
ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size)
vol_spatial = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

cx, cy, cz = size(vol_spatial) .÷ 2
roi_spatial = vol_spatial[cx-30:cx+30, cy-30:cy+30, cz]
σ_spatial = std(roi_spatial)
μ_spatial = mean(roi_spatial)
println("  μ_recon = $(round(μ_spatial, sigdigits=5)) cm⁻¹")
println("  σ_recon = $(round(σ_spatial, sigdigits=4)) cm⁻¹")
ws_fdk = nothing; GC.gc(true)

# ═══════════════════════════════════════════════════════════
# Method 2: FFT-based filtering + same backprojection
# ═══════════════════════════════════════════════════════════
println("\n[3] FFT-based filtering + same backprojection...")

# Create the spatial kernel (same as our code)
kernel_cpu = BS.create_spatial_kernel(n_cols, BS.RampFilter(), Float64(pixel_size_cm))
println("  kernel h[0] = $(round(kernel_cpu[n_cols÷2+1], digits=4))")

# FFT-based convolution
# Pad to next power of 2 (like CatSim)
fft_size = nextpow(2, 2 * n_cols)
println("  FFT size = $fft_size (vs n_cols = $n_cols)")

# Create kernel in FFT domain
kernel_padded = zeros(Float64, fft_size)
kernel_center = n_cols ÷ 2 + 1
# Center the kernel at the beginning (for FFT convention)
for i in 1:n_cols
    k = i - kernel_center  # offset from center
    # Wrap around for FFT convention
    fft_idx = mod(k, fft_size) + 1
    kernel_padded[fft_idx] = kernel_cpu[i]
end
FFT_kernel = fft(kernel_padded)

# Apply FFT filtering + cosine weighting to the sinogram
sino_fft = copy(sino_cpu)

# First apply cosine weighting (same as our code)
SDD = Float64(geom.SDD)
SDD_sq = SDD^2
mag = SDD / Float64(geom.SAD)
col_center = (Float64(n_cols) + 1.0) / 2.0
row_center = (Float64(n_rows) + 1.0) / 2.0

for angle in 1:n_angles, row in 1:n_rows, col in 1:n_cols
    u = (Float64(col) - col_center) * pixel_size_cm * mag
    v = (Float64(row) - row_center) * pixel_size_cm * mag
    dist = sqrt(SDD_sq + u^2 + v^2)
    weight = SDD / dist
    sino_fft[col, row, angle] *= Float32(weight)
end

# Then apply FFT-based ramp filter (row by row)
for angle in 1:n_angles, row in 1:n_rows
    # Extract row
    data = Float64.(sino_fft[:, row, angle])

    # Zero-pad and FFT
    data_padded = zeros(Float64, fft_size)
    data_padded[1:n_cols] = data
    FFT_data = fft(data_padded)

    # Multiply in frequency domain
    filtered = real(ifft(FFT_data .* FFT_kernel))

    # Copy back
    sino_fft[:, row, angle] .= Float32.(filtered[1:n_cols])
end

# Backproject using our same backprojector
sino_fft_gpu = MtlArray(sino_fft)
recon_fft_size = recon_opts.matrix_size
vol_fft = Array(BS.backproject(sino_fft_gpu, geom, recon_fft_size))

roi_fft = vol_fft[cx-30:cx+30, cy-30:cy+30, cz]
σ_fft = std(roi_fft)
μ_fft = mean(roi_fft)
println("  μ_recon = $(round(μ_fft, sigdigits=5)) cm⁻¹")
println("  σ_recon = $(round(σ_fft, sigdigits=4)) cm⁻¹")

# ═══════════════════════════════════════════════════════════
# Method 3: CatSim-style FFT with dimensionless kernel + /DeltaUW
# ═══════════════════════════════════════════════════════════
println("\n[4] CatSim-style FFT filtering + our backprojection...")

DecFanAng = 2 * atan(n_cols / 2 * 1.0 / sdd)
DeltaUW = DecFanAng / n_cols

# CatSim R-L kernel (dimensionless)
catsim_kernel = zeros(Float64, n_cols)
kernel_center_cs = n_cols ÷ 2
for i in 0:n_cols-1
    k = i - kernel_center_cs
    if k == 0
        catsim_kernel[i+1] = 0.25
    elseif k % 2 == 0
        catsim_kernel[i+1] = 0.0
    else
        catsim_kernel[i+1] = -sin(π * k / 2)^2 / (π^2 * k^2)
    end
end

# Pad and FFT (CatSim style)
nn2_cs = nextpow(2, 2 * n_cols)
catsim_kernel_padded = zeros(Float64, nn2_cs)
# CatSim circular shift: right half goes to front, left half goes to back
k_half = n_cols ÷ 2
catsim_kernel_padded[1:k_half] = catsim_kernel[k_half+1:n_cols]
catsim_kernel_padded[k_half+n_cols+1:nn2_cs] .= 0  # already zero
catsim_kernel_padded[nn2_cs-k_half+1:nn2_cs] = catsim_kernel[1:k_half]
# Multiply by i (imaginary unit) like CatSim
catsim_kernel_complex = catsim_kernel_padded .* im
FFT_F_cs = fft(catsim_kernel_complex)

# CatSim-style pre-weighting (equi-angle)
sino_cs_style = copy(sino_cpu)
for angle in 1:n_angles, row in 1:n_rows, col in 1:n_cols
    z_offset = (Float64(row) - row_center) * Float64(geom.pixel_row_size) * mag
    dist_weight = sdd / 10.0 / sqrt((sdd/10.0)^2 + z_offset^2)  # convert to cm
    fan_angle = (Float64(col) - col_center) * DeltaUW
    cosine_weight = cos(fan_angle)
    sino_cs_style[col, row, angle] *= Float32(dist_weight * cosine_weight)
end

# FFT filter (CatSim style)
for angle in 1:n_angles, row in 1:n_rows
    data = Float64.(sino_cs_style[:, row, angle])
    data_padded = zeros(Float64, nn2_cs)
    data_padded[1:n_cols] = data
    FFT_S = fft(data_padded)
    TempData = ifft(FFT_S .* FFT_F_cs)
    # Extract imaginary part, negate (CatSim convention)
    for k in 1:n_cols
        sino_cs_style[k, row, angle] = Float32(-imag(TempData[k]))
    end
end

# Divide by DeltaUW (CatSim final scaling)
sino_cs_style ./= Float32(DeltaUW)

# Backproject using CatSim-style weights:
# CatSim: accumulates filtered / Dlocal², then *= -ScanR × π / ProjNum
# We need to modify the filtered sinogram so our backprojector gives the right answer.
# Our backprojector does: acc × π / N × SAD² / dist² per angle
# CatSim does: -ScanR × π / N × 1/Dlocal² per angle
# At center: our = π/N × filtered_ours, CatSim = -ScanR × π / N × filtered_cs / ScanR²
#           = our: π/N × filtered_ours
#           = cs:  π / (N × ScanR) × (-filtered_cs)  [ScanR in mm]
# To use our backprojector: filtered_ours = (-filtered_cs) / ScanR_cm / SAD_sq_cm × SAD_sq_cm
# = (-filtered_cs) / ScanR_cm
# Actually: our backprojector computes Σ (val × SAD²/dist²) × π/N
# CatSim result = Σ (val / Dlocal²) × (-ScanR × π / N)
# For them to match: val_ours × SAD² / dist² = -val_cs / Dlocal² × ScanR
# At center: val_ours = -val_cs × ScanR / SAD² = -val_cs / SAD  (since ScanR = SAD in mm, and SAD in cm = ScanR/10)
# Hmm, units are confusing. Let me just try direct comparison.

# Just backproject the CatSim-filtered sino directly (wrong normalization but let's see the noise)
sino_cs_gpu = MtlArray(sino_cs_style)
vol_cs = Array(BS.backproject(sino_cs_gpu, geom, recon_fft_size))

roi_cs = vol_cs[cx-30:cx+30, cy-30:cy+30, cz]
σ_cs = std(roi_cs)
μ_cs = mean(roi_cs)
println("  μ_recon = $(round(μ_cs, sigdigits=5)) cm⁻¹ (normalization may differ)")
println("  σ_recon = $(round(σ_cs, sigdigits=4)) cm⁻¹")

# ═══════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════
println("\n═══════════════════════════════════════════════════════")
println("  COMPARISON")
println("═══════════════════════════════════════════════════════")
println("                     μ_recon       σ_recon")
println("  Spatial (ours):    $(lpad(round(μ_spatial, sigdigits=5), 12))  $(round(σ_spatial, sigdigits=4))")
println("  FFT (same kernel): $(lpad(round(μ_fft, sigdigits=5), 12))  $(round(σ_fft, sigdigits=4))")
println("  CatSim-style FFT:  $(lpad(round(μ_cs, sigdigits=5), 12))  $(round(σ_cs, sigdigits=4))")
println()
println("  FFT/Spatial noise ratio:     $(round(σ_fft / σ_spatial, digits=4))")
println("  CatSim-style/Spatial ratio:  $(round(σ_cs / σ_spatial, digits=4))")
println()
if abs(σ_fft / σ_spatial - 1.0) < 0.05
    println("  → FFT and Spatial give SAME noise (within 5%)")
    println("  → Filtering approach is NOT the problem")
else
    println("  → FFT and Spatial give DIFFERENT noise")
    println("  → Filtering approach IS a contributor!")
end

println("\n═══════════════════════════════════════════════════════")
println("  TEST COMPLETE")
println("═══════════════════════════════════════════════════════")
