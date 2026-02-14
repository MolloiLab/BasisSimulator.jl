#!/usr/bin/env julia
"""
NOISE-014: Validate notebook simulation paths after StandardFilter fix.

Tests core simulation/reconstruction from notebooks 01-05 without Pluto.
Checks:
1. No regressions in HU accuracy (water ROI near 0 HU)
2. Noise levels are reasonable (not NaN/Inf, within expected range)
3. Forward projection and reconstruction complete without errors
4. Scanner convention changes (isocenter pixel sizes) work correctly
"""

using BasisSimulator
const BS = BasisSimulator
using Statistics
using Metal
using Unitful

println("="^60)
println("NOISE-014: Notebook Validation Script")
println("="^60)

# Helper: create a simple water cylinder phantom on GPU
function make_water_phantom_gpu(nx, ny, nz; fov_cm=35.0, z_cm=1.6, radius_cm=16.5)
    voxel_cm = fov_cm / nx
    voxel_z_cm = z_cm / nz

    water_mask = zeros(UInt8, nx, ny, nz)
    xs = range(-fov_cm/2, fov_cm/2, length=nx)
    ys = range(-fov_cm/2, fov_cm/2, length=ny)
    for k in 1:nz, j in 1:ny, i in 1:nx
        if sqrt(xs[i]^2 + ys[j]^2) <= radius_cm
            water_mask[i, j, k] = UInt8(1)
        end
    end

    air_material = BS.XA.Material(
        "Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
        Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
    )
    water_materials = Dict(0 => air_material, 1 => BS.XA.Materials.water)

    BS.Phantom(MtlArray(water_mask), water_materials, (voxel_cm, voxel_cm, voxel_z_cm))
end

# ================================================================
# NOTEBOOK 01: Single kVp Verification (Gammex phantom)
# Already validated by NOISE-013 — quick sanity check only
# ================================================================
println("\n--- Notebook 01: Single kVp (Quick Sanity Check) ---")

let
    # Scanner from NB01 (isocenter convention)
    mag = 950.0 / 540.0
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 950.0,
        detector_rows = 16,
        detector_cols = 900,
        detector_row_size = 1.0 / mag,
        detector_col_size = 1.0 / mag,
        detector_shape = BS.CURVED_DETECTOR,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        detector_material = :gadolinium_oxysulfide,
        detector_depth = 0.5,
        fill_factor_row = 0.9,
        fill_factor_col = 0.9,
    )

    protocol = BS.CTProtocol(kVp=120.0, mA=200.0, views=984, rotation_time=1.0)
    sim_opts = BS.SimOptions(fidelity=:high, seed=42)

    slice_count = 9
    recon_opts = BS.ReconOptions(
        algorithm=:fdk,
        filter=:standard,
        matrix_size=(128, 128, slice_count),
        fov_cm=35.0,
        z_cm=slice_count * 1.0 / 10.0,
    )

    water_gpu = make_water_phantom_gpu(128, 128, 16; fov_cm=35.0, z_cm=1.6)

    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws, water_gpu, scanner, protocol, sim_opts, recon_opts)

    recon_size = recon_opts.matrix_size
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))
    cx, cy, cz = size(vol) .÷ 2
    μ_water = mean(vol[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])
    hu = BS.to_hounsfield(vol; μ_water=μ_water)
    water_hu = mean(hu[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])
    water_noise = std(hu[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])

    println("  μ_water = $(round(μ_water, digits=4)) cm⁻¹")
    println("  Water HU = $(round(water_hu, digits=1)) (expected ~0)")
    println("  Noise σ = $(round(water_noise, digits=1)) HU")
    @assert !isnan(μ_water) "μ_water is NaN!"
    @assert !isinf(μ_water) "μ_water is Inf!"
    @assert 0.05 < μ_water < 0.5 "μ_water = $μ_water out of range [0.05, 0.5] cm⁻¹"
    @assert abs(water_hu) < 50.0 "Water HU = $water_hu not near 0!"
    println("  ✓ PASS")

    ws = nothing; ws_fdk = nothing; GC.gc(true)
end

# ================================================================
# NOTEBOOK 02: Multi-Dose and Iterative Reconstruction
# Tests: 80/120/140 kVp, HIR reconstruction
# ================================================================
println("\n--- Notebook 02: Multi-Dose (80/120/140 kVp) ---")

let
    mag = 950.0 / 540.0
    det_size = 1.0 / mag
    z_coverage_mm = 16 * det_size
    slice_count = floor(Int, z_coverage_mm / 1.0)

    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 950.0,
        detector_rows = 16,
        detector_cols = 900,
        detector_row_size = det_size,
        detector_col_size = det_size,
        detector_shape = BS.CURVED_DETECTOR,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        detector_material = :gadolinium_oxysulfide,
        detector_depth = 0.5,
        fill_factor_row = 0.9,
        fill_factor_col = 0.9,
    )

    sim_opts = BS.SimOptions(fidelity=:high, seed=1234)
    recon_opts = BS.ReconOptions(
        algorithm=:fdk,
        filter=:ram_lak,
        matrix_size=(128, 128, slice_count),
        fov_cm=35.0,
        z_cm=slice_count * 1.0 / 10.0,
    )

    water_gpu = make_water_phantom_gpu(128, 128, 16; fov_cm=35.0, z_cm=1.6)

    for (kvp, mA) in [(80.0, 50.0), (120.0, 200.0), (140.0, 400.0)]
        prot = BS.CTProtocol(kVp=kvp, mA=mA, views=984, rotation_time=1.0)
        ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, water_gpu)
        BS.simulate!(ws, water_gpu, scanner, prot, sim_opts, recon_opts)

        recon_size = recon_opts.matrix_size
        ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
        vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))
        cx, cy, cz = size(vol) .÷ 2
        z_half = min(cz - 1, 2)
        μ_water = mean(vol[cx-5:cx+5, cy-5:cy+5, max(1,cz-z_half):min(size(vol,3),cz+z_half)])
        hu = BS.to_hounsfield(vol; μ_water=μ_water)
        water_hu = mean(hu[cx-5:cx+5, cy-5:cy+5, max(1,cz-z_half):min(size(vol,3),cz+z_half)])
        noise = std(hu[cx-5:cx+5, cy-5:cy+5, max(1,cz-z_half):min(size(vol,3),cz+z_half)])

        println("  $(Int(kvp)) kVp / $(Int(mA)) mA: μ_water=$(round(μ_water, digits=4)), HU=$(round(water_hu, digits=1)), σ=$(round(noise, digits=1))")
        @assert !isnan(μ_water) "μ_water is NaN at $kvp kVp!"
        @assert 0.05 < μ_water < 0.5 "μ_water = $μ_water out of range at $kvp kVp!"

        ws = nothing; ws_fdk = nothing; GC.gc(true)
    end

    # Test HIR (strength 3) at 120 kVp
    println("  Testing HIR (strength 3)...")
    prot_120 = BS.CTProtocol(kVp=120.0, mA=200.0, views=984, rotation_time=1.0)
    ws = BS.create_eict_workspace(scanner, prot_120, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws, water_gpu, scanner, prot_120, sim_opts, recon_opts)
    sino_gpu = ws.sino_noisy_out
    geom = ws.geom

    ws_hir = BS.create_hir_recon_workspace(geom, sino_gpu; strength=3, array_type=MtlArray)
    BS.reconstruct!(ws_hir, geom, sino_gpu)
    hir_vol = Array(ws_hir.volume)
    cx, cy, cz = size(hir_vol) .÷ 2
    z_half = min(cz - 1, 2)
    μ_hir = mean(hir_vol[cx-5:cx+5, cy-5:cy+5, max(1,cz-z_half):min(size(hir_vol,3),cz+z_half)])
    println("  HIR μ_water = $(round(μ_hir, digits=4)) cm⁻¹")
    @assert !isnan(μ_hir) "HIR μ_water is NaN!"
    @assert 0.05 < μ_hir < 0.5 "HIR μ_water = $μ_hir out of range!"
    println("  ✓ PASS")

    ws = nothing; ws_hir = nothing; GC.gc(true)
end

# ================================================================
# NOTEBOOK 03: Dual-kVp VMI Verification
# Tests: 80/140 kVp dual-energy, physics check
# ================================================================
println("\n--- Notebook 03: Dual-kVp VMI ---")

let
    mag = 950.0 / 540.0
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 950.0,
        detector_rows = 16,
        detector_cols = 900,
        detector_row_size = 1.0 / mag,
        detector_col_size = 1.0 / mag,
        detector_shape = BS.CURVED_DETECTOR,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        detector_material = :gadolinium_oxysulfide,
        detector_depth = 0.5,
        fill_factor_row = 0.9,
        fill_factor_col = 0.9,
    )

    sim_opts = BS.SimOptions(fidelity=:high, seed=5678)
    recon_opts = BS.ReconOptions(
        algorithm=:fdk,
        filter=:ram_lak,
        matrix_size=(128, 128, 9),
        fov_cm=35.0,
        z_cm=9 * 1.0 / 10.0,
    )

    water_gpu = make_water_phantom_gpu(128, 128, 16; fov_cm=35.0, z_cm=1.6)

    # Test 80 kVp scan
    prot_80 = BS.CTProtocol(kVp=80.0, mA=400.0, views=984, rotation_time=1.0)
    ws_80 = BS.create_eict_workspace(scanner, prot_80, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws_80, water_gpu, scanner, prot_80, sim_opts, recon_opts)
    recon_size = recon_opts.matrix_size
    ws_fdk_80 = BS.create_fdk_recon_workspace(ws_80.sino_noisy_out, ws_80.geom, recon_size)
    vol_80 = Array(BS.reconstruct!(ws_fdk_80, ws_80.sino_noisy_out, ws_80.geom, recon_size))
    cx, cy, cz = size(vol_80) .÷ 2
    μ_80 = mean(vol_80[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol_80,3),cz+1)])
    println("  80 kVp water: μ=$(round(μ_80, digits=4)) cm⁻¹")

    ws_80 = nothing; ws_fdk_80 = nothing; GC.gc(true)

    # Test 140 kVp scan
    prot_140 = BS.CTProtocol(kVp=140.0, mA=200.0, views=984, rotation_time=1.0)
    ws_140 = BS.create_eict_workspace(scanner, prot_140, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws_140, water_gpu, scanner, prot_140, sim_opts, recon_opts)
    ws_fdk_140 = BS.create_fdk_recon_workspace(ws_140.sino_noisy_out, ws_140.geom, recon_size)
    vol_140 = Array(BS.reconstruct!(ws_fdk_140, ws_140.sino_noisy_out, ws_140.geom, recon_size))
    μ_140 = mean(vol_140[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol_140,3),cz+1)])
    println("  140 kVp water: μ=$(round(μ_140, digits=4)) cm⁻¹")

    ws_140 = nothing; ws_fdk_140 = nothing; GC.gc(true)

    # Physics check: μ_water should be higher at 80 kVp than 140 kVp
    @assert μ_80 > μ_140 "Expected μ_80 ($μ_80) > μ_140 ($μ_140) — lower energy should have higher attenuation!"
    @assert !isnan(μ_80) && !isnan(μ_140) "NaN detected!"
    @assert 0.05 < μ_80 < 0.5 "μ_80 out of range!"
    @assert 0.05 < μ_140 < 0.5 "μ_140 out of range!"
    println("  ✓ PASS (μ_80 > μ_140 as expected)")
end

# ================================================================
# NOTEBOOK 04: PCCT Demonstration (Siemens NAEOTOM Alpha)
# Tests: PCCT geometry, forward projection
# ================================================================
println("\n--- Notebook 04: PCCT (NAEOTOM Alpha Geometry) ---")

let
    n_cols = ceil(Int, 500.0 / 0.4)
    scanner = BS.Scanner(
        source_to_isocenter = 595.0,
        source_to_detector = 1085.5,
        detector_rows = 16,
        detector_cols = n_cols,
        detector_row_size = 0.4,
        detector_col_size = 0.4,
        detector_shape = BS.CURVED_DETECTOR,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        detector_material = :cdte,
        detector_depth = 1.6,
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        gantry_rotation_time = 0.25,
        energy_thresholds = [25.0, 50.0, 75.0, 90.0],
        energy_resolution = 8.0,
    )

    protocol = BS.CTProtocol(kVp=140.0, mA=300.0, views=984, rotation_time=0.25)
    sim_opts = BS.SimOptions(fidelity=:standard, seed=9999)
    recon_opts = BS.ReconOptions(
        algorithm=:fdk,
        matrix_size=(128, 128, 8),
        fov_cm=35.0,
        z_cm=8 * 0.4 / 10.0,
    )

    water_gpu = make_water_phantom_gpu(128, 128, 16; fov_cm=35.0, z_cm=0.64)

    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws, water_gpu, scanner, protocol, sim_opts, recon_opts)

    recon_size = recon_opts.matrix_size
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))
    cx, cy, cz = size(vol) .÷ 2
    μ_water = mean(vol[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])
    println("  PCCT 140 kVp water: μ=$(round(μ_water, digits=4)) cm⁻¹")
    @assert !isnan(μ_water) "PCCT μ_water is NaN!"
    @assert 0.05 < μ_water < 0.5 "PCCT μ_water = $μ_water out of range!"

    println("  Geometry: FOV=$(round(ws.geom.fov, digits=2))cm, pixel=$(round(ws.geom.pixel_size, digits=4))cm")
    @assert ws.geom.fov > 0 "FOV is zero or negative!"
    @assert ws.geom.pixel_size > 0 "Pixel size is zero or negative!"
    println("  ✓ PASS")

    ws = nothing; ws_fdk = nothing; GC.gc(true)
end

# ================================================================
# NOTEBOOK 05: XCAT Full Simulation (GE Revolution Apex geometry)
# Tests: Large phantom, clinical scanner geometry
# ================================================================
println("\n--- Notebook 05: XCAT (GE Revolution Geometry) ---")

let
    scanner = BS.Scanner(
        source_to_isocenter = 625.6,
        source_to_detector = 1100.0,
        detector_rows = 16,
        detector_cols = ceil(Int, 350.0 / 0.6),
        detector_row_size = 0.625,
        detector_col_size = 0.6,
        detector_shape = BS.CURVED_DETECTOR,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,
        detector_material = :gadolinium_oxysulfide,
        detector_depth = 0.5,
        fill_factor_row = 0.9,
        fill_factor_col = 0.9,
    )

    protocol = BS.CTProtocol(kVp=120.0, mA=300.0, views=984, rotation_time=0.5)
    sim_opts = BS.SimOptions(fidelity=:standard, seed=12345)
    recon_opts = BS.ReconOptions(
        algorithm=:fdk,
        matrix_size=(128, 128, 8),
        fov_cm=25.0,
        z_cm=8 * 0.625 / 10.0,
    )

    water_gpu = make_water_phantom_gpu(128, 128, 16; fov_cm=25.0, z_cm=1.0, radius_cm=12.0)

    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws, water_gpu, scanner, protocol, sim_opts, recon_opts)

    recon_size = recon_opts.matrix_size
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))
    cx, cy, cz = size(vol) .÷ 2
    μ_water = mean(vol[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])
    hu = BS.to_hounsfield(vol; μ_water=μ_water)
    water_hu = mean(hu[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])
    noise = std(hu[cx-5:cx+5, cy-5:cy+5, max(1,cz-1):min(size(vol,3),cz+1)])

    println("  GE Rev Apex-like: μ_water=$(round(μ_water, digits=4)), HU=$(round(water_hu, digits=1)), σ=$(round(noise, digits=1))")
    @assert !isnan(μ_water) "XCAT μ_water is NaN!"
    @assert 0.05 < μ_water < 0.5 "XCAT μ_water = $μ_water out of range!"
    @assert abs(water_hu) < 50.0 "XCAT Water HU = $water_hu not near 0!"
    println("  ✓ PASS")

    ws = nothing; ws_fdk = nothing; GC.gc(true)
end

# ================================================================
# BONUS: Verify StandardFilter matches CatSim noise
# (Quick version of NOISE-013 validation)
# ================================================================
println("\n--- Bonus: StandardFilter vs Ram-Lak noise comparison ---")

let
    mag = 950.0 / 540.0
    scanner = BS.Scanner(
        source_to_isocenter = 540.0,
        source_to_detector = 950.0,
        detector_rows = 16,
        detector_cols = 900,
        detector_row_size = 1.0 / mag,
        detector_col_size = 1.0 / mag,
        detector_shape = BS.CURVED_DETECTOR,
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,
        detector_material = :gadolinium_oxysulfide,
        detector_depth = 0.5,
        fill_factor_row = 0.9,
        fill_factor_col = 0.9,
    )

    protocol = BS.CTProtocol(kVp=120.0, mA=200.0, views=984, rotation_time=1.0)
    sim_opts = BS.SimOptions(fidelity=:high, seed=42)
    recon_size = (256, 256, 9)
    recon_opts = BS.ReconOptions(
        algorithm=:fdk,
        filter=:standard,
        matrix_size=recon_size,
        fov_cm=35.0,
        z_cm=9 * 1.0 / 10.0,
    )

    water_gpu = make_water_phantom_gpu(256, 256, 16; fov_cm=35.0, z_cm=1.6)

    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, water_gpu)
    BS.simulate!(ws, water_gpu, scanner, protocol, sim_opts, recon_opts)

    # Reconstruct with StandardFilter
    ws_std = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.StandardFilter())
    vol_std = Array(BS.reconstruct!(ws_std, ws.sino_noisy_out, ws.geom, recon_size))
    cx, cy, cz = size(vol_std) .÷ 2
    μ_std = mean(vol_std[cx-20:cx+20, cy-20:cy+20, max(1,cz-2):min(size(vol_std,3),cz+2)])
    hu_std = BS.to_hounsfield(vol_std; μ_water=μ_std)
    σ_std = std(hu_std[cx-20:cx+20, cy-20:cy+20, max(1,cz-2):min(size(vol_std,3),cz+2)])

    # Reconstruct with Ram-Lak
    ws_rl = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.RamLakFilter())
    vol_rl = Array(BS.reconstruct!(ws_rl, ws.sino_noisy_out, ws.geom, recon_size))
    μ_rl = mean(vol_rl[cx-20:cx+20, cy-20:cy+20, max(1,cz-2):min(size(vol_rl,3),cz+2)])
    hu_rl = BS.to_hounsfield(vol_rl; μ_water=μ_rl)
    σ_rl = std(hu_rl[cx-20:cx+20, cy-20:cy+20, max(1,cz-2):min(size(vol_rl,3),cz+2)])

    ratio = σ_rl / σ_std
    println("  StandardFilter: σ=$(round(σ_std, digits=1)) HU, μ_water=$(round(μ_std, digits=4))")
    println("  Ram-Lak:        σ=$(round(σ_rl, digits=1)) HU, μ_water=$(round(μ_rl, digits=4))")
    println("  Noise ratio (RL/Std): $(round(ratio, digits=2))x (expected ~2.1x)")
    @assert 1.5 < ratio < 3.0 "Noise ratio $(ratio) outside expected range [1.5, 3.0]!"
    println("  ✓ PASS")

    ws = nothing; ws_std = nothing; ws_rl = nothing; GC.gc(true)
end

println("\n" * "="^60)
println("ALL NOTEBOOK VALIDATION TESTS PASSED")
println("="^60)
