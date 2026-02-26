#!/usr/bin/env julia
"""
FDK Noise Validation Test

Create a uniform sinogram with known white noise, reconstruct with our FDK,
and compare the output noise with the analytical prediction.

Analytical prediction for FDK noise (parallel beam, center voxel):
    σ²_recon = σ²_sino × π / (2 × N_angles) × (1/Δ²) × Σ_k h²_k × (1/N_cols)

Actually, the exact formula depends on the geometry. Let's just measure it empirically
and compare with CatSim by running both reconstructions on the same synthetic data.

Simpler approach: just verify our FDK produces correct noise by checking that
σ_recon scales correctly with σ_sino (should be linear).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Metal
import BasisSimulator as BS
using Statistics
using Random

println("═══════════════════════════════════════════════════════")
println("  FDK Noise Validation: Synthetic Sinogram Test")
println("═══════════════════════════════════════════════════════")

# Parameters matching notebook 01
sid = 540.0
sdd = 950.0
magnification = sdd / sid
detectorColSize_face = 1.0
detectorRowSize_face = 1.0
detectorColSize = detectorColSize_face / magnification
detectorRowSize = detectorRowSize_face / magnification
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
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material = :gadolinium_oxysulfide,
    detector_depth = 0.5,
)

protocol = BS.CTProtocol(mA=200.0, kVp=120, views=984, rotation_time=1.0)

recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (imageSize, imageSize, sliceCount),
    fov_cm = fov_mm / 10.0,
    z_cm = sliceCount * sliceThickness / 10.0,
    filter = :ram_lak,
)

# Create CTGeometry
geom = BS.CTGeometry(scanner; n_angles=protocol.views, fov_cm=recon_opts.fov_cm, z_cm=recon_opts.z_cm)
println("  geom.pixel_size = $(geom.pixel_size) cm")
println("  geom.SAD = $(geom.SAD) cm")
println("  geom.SDD = $(geom.SDD) cm")
println("  geom.n_angles = $(geom.n_angles)")

# ═══════════════════════════════════════════════════════════
# Test 1: Uniform sinogram with noise
# ═══════════════════════════════════════════════════════════
println("\n[Test 1] Uniform sinogram + white noise")
n_cols, n_rows, n_angles = detectorColCount, detectorRowCount, 984

# Create uniform sinogram (equivalent to homogeneous water phantom)
# Mean path length through 33cm water at 60 keV: μ_w = 0.2059 cm⁻¹, D = 33 cm
μ_water = 0.2059  # cm⁻¹ at 60 keV
D_water = 33.0    # cm (Gammex phantom diameter)
sino_mean = μ_water * D_water  # ≈ 6.8

# Expected noise in sinogram (Poisson, Gaussian approx):
# I0 ≈ 451,000 photons/pixel/view (from NOISE-001)
I0 = 451000.0
λ = I0 * exp(-sino_mean)  # detected photons through center
σ_sino = 1 / sqrt(λ)      # Gaussian approx for log-domain noise

println("  sino_mean = $sino_mean (μ_w × D)")
println("  I0 = $I0 photons/pixel/view")
println("  λ (detected through center) = $(round(λ, digits=1)) photons")
println("  σ_sino (analytical) = $(round(σ_sino, sigdigits=4))")

# Create synthetic sinograms with different noise levels
Random.seed!(1234)
for (label, σ_test) in [("zero noise", 0.0),
                         ("analytical σ", σ_sino),
                         ("2× analytical σ", 2*σ_sino)]
    println("\n  --- $label (σ = $(round(σ_test, sigdigits=4))) ---")

    # Create uniform sinogram + noise
    sino_cpu = fill(Float32(sino_mean), n_cols, n_rows, n_angles)
    if σ_test > 0
        sino_cpu .+= Float32(σ_test) .* randn(Float32, n_cols, n_rows, n_angles)
    end
    sino_gpu = MtlArray(sino_cpu)

    # Reconstruct with FDK
    recon_size = recon_opts.matrix_size
    ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size)
    vol = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    # Measure center ROI
    cx, cy, cz = size(vol) .÷ 2
    roi = vol[cx-30:cx+30, cy-30:cy+30, cz]
    σ_recon = std(roi)
    μ_recon = mean(roi)

    # Convert to HU
    hu = BS.to_hounsfield(vol; μ_water=μ_water)
    roi_hu = hu[cx-30:cx+30, cy-30:cy+30, cz]
    σ_hu = std(roi_hu)
    μ_hu = mean(roi_hu)

    println("    μ_recon = $(round(μ_recon, sigdigits=5)) cm⁻¹ (expected $(round(μ_water, sigdigits=5)))")
    println("    σ_recon = $(round(σ_recon, sigdigits=4)) cm⁻¹")
    println("    σ_HU    = $(round(σ_hu, digits=2)) HU")

    if σ_test > 0
        # Theoretical noise gain through FDK
        println("    noise gain (σ_recon / σ_sino) = $(round(σ_recon / σ_test, digits=4))")
    end

    ws_fdk = nothing; vol = nothing; hu = nothing
    GC.gc(true)
end

# ═══════════════════════════════════════════════════════════
# Test 2: Compare our sinogram noise with theoretical
# ═══════════════════════════════════════════════════════════
println("\n[Test 2] Verify sinogram noise from actual simulation matches theory")

# Run noise-only simulation on water phantom
using Unitful: @u_str
import XrayAttenuation as XA

fov_cm = fov_mm / 10.0
total_z_cm = (sliceCount * sliceThickness) / 10.0
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
air_material = XA.Material(
    "Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
    Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
)
water_materials = Dict(0 => air_material, 1 => XA.Materials.water)
phantom_water_gpu = BS.Phantom(
    MtlArray(water_mask), water_materials,
    (voxel_cm, voxel_cm, voxel_z_cm)
)

# Run simulation with and without noise
for (label, use_noise) in [("noiseless", false), ("noisy", true)]
    println("\n  --- Water phantom: $label ---")
    sim_opts = BS.SimOptions(fidelity=:ideal, use_noise=use_noise, seed=42)
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_water_gpu)
    BS.simulate!(ws, phantom_water_gpu, scanner, protocol, sim_opts, recon_opts)

    sino = ws.sino_noisy_out
    sino_ideal = ws.sino_ideal_out

    # Center of sinogram (through center of phantom)
    s_cx = size(sino, 1) ÷ 2
    s_row = size(sino, 2) ÷ 2

    # Sample a few central pixels across all angles
    sino_center_vals = Float64.(sino[s_cx, s_row, :])
    sino_ideal_center = Float64.(sino_ideal[s_cx, s_row, :])

    μ_sino = mean(sino_center_vals)
    σ_sino_measured = std(sino_center_vals .- sino_ideal_center)

    # Theoretical sinogram noise
    I0_computed = BS.compute_detector_I0(ws.geom, protocol)
    λ_through_center = I0_computed * exp(-μ_sino)
    σ_sino_theory = 1.0 / sqrt(λ_through_center)

    println("    Mean sinogram (center pixel): $(round(μ_sino, digits=4))")
    println("    I0 = $(round(I0_computed, digits=0))")
    println("    λ_through_center = $(round(λ_through_center, digits=1))")
    println("    σ_sino (measured) = $(round(σ_sino_measured, sigdigits=4))")
    println("    σ_sino (theory)   = $(round(σ_sino_theory, sigdigits=4))")
    if use_noise
        println("    Ratio (measured/theory) = $(round(σ_sino_measured / σ_sino_theory, digits=3))")
    end

    # Also reconstruct
    recon_size = recon_opts.matrix_size
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

    cx, cy, cz = size(vol) .÷ 2
    roi = vol[cx-30:cx+30, cy-30:cy+30, cz]
    σ_μ = std(roi)
    μ_mean = mean(roi)

    println("    μ_recon (center)  = $(round(μ_mean, sigdigits=5)) cm⁻¹")
    println("    σ_recon (center)  = $(round(σ_μ, sigdigits=4)) cm⁻¹")

    if use_noise
        σ_hu_calc = 1000 * σ_μ / μ_mean
        println("    σ_HU (using measured μ_water) = $(round(σ_hu_calc, digits=1)) HU")
        σ_hu_calc2 = 1000 * σ_μ / 0.2
        println("    σ_HU (using μ_water=0.2)      = $(round(σ_hu_calc2, digits=1)) HU")
    end

    ws = nothing; ws_fdk = nothing; vol = nothing
    GC.gc(true)
end

println("\n═══════════════════════════════════════════════════════")
println("  FDK Noise Validation COMPLETE")
println("═══════════════════════════════════════════════════════")
