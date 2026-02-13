#!/usr/bin/env julia
"""
NOISE-006: Noise Isolation Test

Runs the same phantom/scanner/protocol as notebook 01, but with:
  - SimOptions(fidelity=:ideal, use_noise=true)  → noise-only (no physics effects)
  - SimOptions(fidelity=:high)                    → full physics (for comparison)

Then measures σ (std of water ROI) in both cases and compares to CatSim's expected σ.

If noise-only σ ≈ CatSim σ → problem is in physics effects
If noise-only σ ≈ 2x CatSim σ → problem is in I0 computation or FDK normalization
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

println("═══════════════════════════════════════════════════════")
println("  NOISE-006: Noise Isolation Test")
println("═══════════════════════════════════════════════════════")

using Metal
import BasisSimulator as BS
import XrayAttenuation as XA
using Statistics
using Unitful: @u_str

# ─── 1. Scanner configuration (same as notebook 01) ───
println("\n[1/6] Creating scanner configuration...")
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

# ─── 2. Protocol (same as notebook 01) ───
protocol = BS.CTProtocol(
    mA = 200.0,
    kVp = 120,
    views = 984,
    rotation_time = 1.0,
)

# ─── 3. ReconOptions (same as notebook 01) ───
recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (imageSize, imageSize, sliceCount),
    fov_cm = fov_mm / 10.0,
    z_cm = sliceCount * sliceThickness / 10.0,
    filter = :ram_lak,
)

# ─── 4. Create phantoms ───
println("[2/6] Creating phantom...")
fov_cm = fov_mm / 10.0
total_z_cm = (sliceCount * sliceThickness) / 10.0

# Gammex phantom
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
air_material = XA.Material(
    "Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
    Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
)
water_materials = Dict(0 => air_material, 1 => XA.Materials.water)
phantom_water_gpu = BS.Phantom(
    MtlArray(water_mask), water_materials,
    (voxel_cm, voxel_cm, voxel_z_cm)
)

# ─── 5. Helper function to run simulation + reconstruction + measure noise ───
function run_and_measure(phantom_gpu, phantom_water_gpu, scanner, protocol, sim_opts, recon_opts, label)
    println("  [$label] Creating workspace...")

    # Water calibration
    ws_w = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_water_gpu)
    BS.simulate!(ws_w, phantom_water_gpu, scanner, protocol, sim_opts, recon_opts)
    recon_size = recon_opts.matrix_size
    ws_fdk_w = BS.create_fdk_recon_workspace(ws_w.sino_noisy_out, ws_w.geom, recon_size)
    vol_w = Array(BS.reconstruct!(ws_fdk_w, ws_w.sino_noisy_out, ws_w.geom, recon_size))
    cx, cy, cz = size(vol_w) .÷ 2
    μ_water = mean(vol_w[cx-2:cx+2, cy-2:cy+2, cz-1:cz+1])
    ws_w = nothing; ws_fdk_w = nothing; vol_w = nothing; GC.gc(true)

    println("  [$label] μ_water = $μ_water cm⁻¹")

    # Gammex simulation
    println("  [$label] Running simulation...")
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
    @time BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)

    # Reconstruct
    println("  [$label] Reconstructing...")
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

    # Convert to HU
    hu = BS.to_hounsfield(vol; μ_water=μ_water)

    # Measure noise in central water ROI (same as notebook 01)
    nx, ny, nz_vol = size(hu)
    cx, cy, mid_z = nx ÷ 2, ny ÷ 2, nz_vol ÷ 2
    roi_sz = 30
    bg_roi = hu[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]
    σ_hu = std(bg_roi)
    μ_hu = mean(bg_roi)

    # Also measure sinogram statistics
    sino_ideal = ws.sino_ideal_out
    sino_noisy = ws.sino_noisy_out
    sino_diff = sino_noisy .- sino_ideal

    # Center region of sinogram
    s_cx, s_cy = size(sino_ideal, 1) ÷ 2, size(sino_ideal, 2) ÷ 2
    sino_center = sino_ideal[s_cx-5:s_cx+5, s_cy-1:s_cy+1, :]
    sino_noise_center = sino_diff[s_cx-5:s_cx+5, s_cy-1:s_cy+1, :]

    # Compute I0 for reference
    geom = ws.geom
    I0 = BS.compute_detector_I0(geom, protocol)

    ws = nothing; ws_fdk = nothing; vol = nothing; GC.gc(true)

    results = (
        label = label,
        σ_hu = σ_hu,
        μ_hu_water = μ_hu,
        μ_water_cal = μ_water,
        I0 = I0,
        sino_mean = mean(sino_center),
        sino_noise_std = std(sino_noise_center),
    )

    println("  [$label] Results:")
    println("    σ_HU (water ROI, 61×61 center) = $(round(σ_hu, digits=2)) HU")
    println("    μ_HU (water ROI mean)           = $(round(μ_hu, digits=2)) HU")
    println("    μ_water (calibration)            = $(round(μ_water, sigdigits=4)) cm⁻¹")
    println("    I0 (photons/pixel/view)          = $(round(Int, I0))")
    println("    Mean sinogram (center)           = $(round(sino_mean, digits=4))")
    println("    Sinogram noise σ (center)        = $(round(sino_noise_std, sigdigits=4))")
    println()

    return results
end

# ─── 6. Run experiments ───

# Experiment A: Noise-only (fidelity=:ideal, use_noise=true)
println("\n[3/6] Running EXPERIMENT A: Noise-only (ideal + noise)...")
sim_opts_noise_only = BS.SimOptions(
    fidelity = :ideal,
    use_noise = true,
    seed = 1234,
)
results_noise_only = run_and_measure(
    phantom_gpu, phantom_water_gpu, scanner, protocol, sim_opts_noise_only, recon_opts,
    "NOISE-ONLY"
)

# Experiment B: Full physics (fidelity=:high)
println("[4/6] Running EXPERIMENT B: Full physics (high fidelity)...")
sim_opts_full = BS.SimOptions(
    fidelity = :high,
    seed = 1234,
)
results_full = run_and_measure(
    phantom_gpu, phantom_water_gpu, scanner, protocol, sim_opts_full, recon_opts,
    "FULL-PHYSICS"
)

# Experiment C: No noise at all (ideal, no noise) — measures FDK artifacts only
println("[5/6] Running EXPERIMENT C: No noise (ideal baseline)...")
sim_opts_ideal = BS.SimOptions(
    fidelity = :ideal,
    use_noise = false,
    seed = 1234,
)
results_noiseless = run_and_measure(
    phantom_gpu, phantom_water_gpu, scanner, protocol, sim_opts_ideal, recon_opts,
    "NOISELESS"
)

# ─── 7. Summary ───
println("\n[6/6] ═══════════════════════════════════════════════════════")
println("  NOISE-006: RESULTS SUMMARY")
println("═══════════════════════════════════════════════════════")
println()
println("CatSim reference: σ ≈ 15-20 HU (expected for 120kVp, 200mA, 984 views)")
println()
println("┌────────────────┬──────────┬──────────────┬──────────────┐")
println("│ Configuration   │ σ_HU     │ μ_water_cal  │ Sino noise σ │")
println("├────────────────┼──────────┼──────────────┼──────────────┤")
Printf = @__MODULE__
for r in [results_noiseless, results_noise_only, results_full]
    println("│ $(rpad(r.label, 14)) │ $(lpad(string(round(r.σ_hu, digits=2)), 8)) │ $(lpad(string(round(r.μ_water_cal, sigdigits=4)), 12)) │ $(lpad(string(round(r.sino_noise_std, sigdigits=4)), 12)) │")
end
println("└────────────────┴──────────┴──────────────┴──────────────┘")
println()

# Partition analysis
ratio_noise_to_full = results_noise_only.σ_hu / results_full.σ_hu
ratio_full_to_catsim_est = results_full.σ_hu / 17.0  # CatSim expected ~17 HU

println("ANALYSIS:")
println("  Noise-only / Full-physics σ ratio: $(round(ratio_noise_to_full, digits=3))")
println("  Full-physics / CatSim-est σ ratio: $(round(ratio_full_to_catsim_est, digits=3))")
println()

if ratio_noise_to_full > 0.8
    println("  → NOISE-ONLY dominates → Problem is in I0 computation or FDK normalization")
    println("    (Physics effects contribute < $(round((1 - ratio_noise_to_full) * 100, digits=1))% of total noise)")
else
    println("  → PHYSICS EFFECTS contribute significantly → Problem is in physics chain")
    println("    (Physics adds $(round((1/ratio_noise_to_full - 1) * 100, digits=1))% more noise)")
end
println()

# Water calibration comparison
println("  μ_water comparison:")
println("    Noise-only μ_water: $(round(results_noise_only.μ_water_cal, sigdigits=4)) cm⁻¹")
println("    Full-physics μ_water: $(round(results_full.μ_water_cal, sigdigits=4)) cm⁻¹")
println("    CatSim μ_water: 0.02 cm⁻¹ (= 0.2 mm⁻¹)")
println("    NIST water @ 60 keV: 0.02059 cm⁻¹")
println()

println("═══════════════════════════════════════════════════════")
println("  NOISE-006 COMPLETE")
println("═══════════════════════════════════════════════════════")
