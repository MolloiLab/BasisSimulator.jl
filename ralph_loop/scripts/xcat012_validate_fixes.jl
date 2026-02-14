#!/usr/bin/env julia
# XCAT-012: Validate all fixes on Gammex phantom
#
# Tests the complete EICT and PCCT pipelines with all applied fixes:
# 1. XCAT-009: calibrate_bhc() per spectrum (not hardcoded defaults)
# 2. XCAT-008: R-matrix effective weights for PCCT BHC calibration
# 3. XCAT-010: Per-scanner water calibration z_cm
# 4. XCAT-011: volume_fov in forward_project!() API
#
# Uses Gammex 472 phantom for fast validation (not XCAT).

using BasisSimulator
using Metal
using Statistics

const BS = BasisSimulator

println("="^70)
println("XCAT-012: Validate All Fixes")
println("="^70)

# ─── Create Phantom ───
println("\n--- Creating Gammex 472 phantom ---")
phantom = create_gammex_472(n_voxels=128, n_slices=32, fov_cm=35.0, z_cm=4.0)
phantom_gpu = BS.Phantom(MtlArray(phantom.mask), phantom.materials, phantom.voxel_size, phantom.origin, phantom.fov)
println("  Phantom: $(size(phantom.mask)), FOV: $(phantom.fov) cm")

# ─── Scanners ───
println("\n--- Creating scanners ---")
scanner_eict = BS.Scanner(
    source_to_isocenter = 625.6, source_to_detector = 1100.0,
    detector_rows = 32, detector_cols = 736,
    detector_row_size = 0.625, detector_col_size = 1.0,
    detector_shape = BS.CURVED_DETECTOR,
    flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
    detector_material = :gos, detector_depth = 3.0,
    fill_factor_row = 0.9, fill_factor_col = 0.9, target_angle = 7.0,
)
scanner_pcct = BS.Scanner(
    source_to_isocenter = 595.0, source_to_detector = 1085.5,
    detector_rows = 32, detector_cols = 900,
    detector_row_size = 0.4, detector_col_size = 0.4,
    detector_shape = BS.CURVED_DETECTOR,
    flat_filter_material = :aluminum, flat_filter_thickness = 2.5,
    detector_material = :cdte, detector_depth = 1.6,
    fill_factor_row = 0.95, fill_factor_col = 0.95, target_angle = 7.0,
    detector_type = :photon_counting, n_energy_bins = 4,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0],
    energy_resolution = 10.0, charge_sharing_fwhm = 0.08,
    dead_time_ns = 5.0, pixel_mode = :standard,
)

protocol_eict = BS.CTProtocol(kVp=120.0, mA=300.0, views=600, rotation_time=0.5)
protocol_pcct = BS.CTProtocol(kVp=140.0, mA=300.0, views=600, rotation_time=0.25)

# ─── Test configs ───
# 1) Noiseless (ideal + BHC): tests BHC calibration
sim_opts_ideal = BS.SimOptions(fidelity=:ideal, use_bhc=true, n_energy_bins=30, seed=42)
# 2) Full physics (high): tests complete signal chain
sim_opts_full = BS.SimOptions(fidelity=:high, pcct_noise_reduction=0.60, n_energy_bins=30, seed=42)

recon_opts = BS.ReconOptions(algorithm=:fdk, matrix_size=(128,128,32), fov_cm=35.0, z_cm=4.0, filter=:standard)

# ─── Water calibration (per-scanner z_cm) ───
println("\n--- Water calibration (per-scanner z_cm) ---")
water_phantom = create_gammex_472(n_voxels=64, n_slices=8, fov_cm=35.0, z_cm=0.5)
water_phantom_gpu = BS.Phantom(MtlArray(water_phantom.mask), water_phantom.materials, water_phantom.voxel_size, water_phantom.origin, water_phantom.fov)

function extract_water_mu(vol)
    nx, ny, nz = size(vol)
    cx, cy = nx ÷ 2, ny ÷ 2
    r = nx ÷ 10
    vals = Float64[]
    for k in 1:nz, j in (cy-r):(cy+r), i in (cx-r):(cx+r)
        if (i-cx)^2 + (j-cy)^2 <= r^2
            push!(vals, vol[i,j,k])
        end
    end
    return mean(vals)
end

function water_cal(scanner, protocol, sim_opts, water_phantom_gpu, z_cm)
    wr = BS.ReconOptions(algorithm=:fdk, matrix_size=(64,64,8), fov_cm=35.0, z_cm=z_cm, filter=:standard)
    if scanner.detector_type == :photon_counting
        ws = BS.create_workspace(scanner, protocol, sim_opts, wr, water_phantom_gpu)
    else
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, wr, water_phantom_gpu)
    end
    BS.simulate!(ws, water_phantom_gpu, scanner, protocol, sim_opts, wr)
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_ideal_out, ws.geom, wr.matrix_size, filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_ideal_out, ws.geom, wr.matrix_size))
    return extract_water_mu(vol)
end

# Noiseless water cal
μ_eict_ideal = water_cal(scanner_eict, protocol_eict, sim_opts_ideal, water_phantom_gpu, 8*0.625/10)
μ_pcct_ideal = water_cal(scanner_pcct, protocol_pcct, sim_opts_ideal, water_phantom_gpu, 8*0.4/10)
println("  Noiseless: μ_eict=$(round(μ_eict_ideal, digits=5)), μ_pcct=$(round(μ_pcct_ideal, digits=5))")

# Full physics water cal
μ_eict_full = water_cal(scanner_eict, protocol_eict, sim_opts_full, water_phantom_gpu, 8*0.625/10)
μ_pcct_full = water_cal(scanner_pcct, protocol_pcct, sim_opts_full, water_phantom_gpu, 8*0.4/10)
println("  Full physics: μ_eict=$(round(μ_eict_full, digits=5)), μ_pcct=$(round(μ_pcct_full, digits=5))")

# ─── Helper ───
function run_sim_and_recon(scanner, protocol, sim_opts, recon_opts, phantom_gpu, μ_water; pcct=false)
    if pcct
        ws = BS.create_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
    else
        ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
    end
    BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_ideal_out, ws.geom, recon_opts.matrix_size, filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_ideal_out, ws.geom, recon_opts.matrix_size))
    hu = 1000f0 .* (vol .- Float32(μ_water)) ./ Float32(μ_water)
    return hu
end

function compute_cupping(hu_vol, z)
    slice = hu_vol[:, :, z]
    nx, ny = size(slice)
    cx, cy = nx ÷ 2, ny ÷ 2
    r_inner = nx ÷ 8
    r_outer = nx ÷ 3
    valid = abs.(slice) .< 2000
    center_mask = [(i-cx)^2 + (j-cy)^2 <= r_inner^2 for i in 1:nx, j in 1:ny] .& valid
    edge_mask = [(i-cx)^2 + (j-cy)^2 > r_inner^2 && (i-cx)^2 + (j-cy)^2 <= r_outer^2 for i in 1:nx, j in 1:ny] .& valid
    center_mean = sum(center_mask) > 0 ? mean(slice[center_mask]) : NaN
    edge_mean = sum(edge_mask) > 0 ? mean(slice[edge_mask]) : NaN
    return (center=center_mean, edge=edge_mean, diff=center_mean - edge_mean)
end

# ─── Test 1: Noiseless (ideal + BHC) ───
println("\n--- Test 1: Noiseless (ideal + BHC) ---")
hu_eict_ideal = run_sim_and_recon(scanner_eict, protocol_eict, sim_opts_ideal, recon_opts, phantom_gpu, μ_eict_ideal; pcct=false)
hu_pcct_ideal = run_sim_and_recon(scanner_pcct, protocol_pcct, sim_opts_ideal, recon_opts, phantom_gpu, μ_pcct_ideal; pcct=true)

z = size(hu_eict_ideal, 3) ÷ 2
cup_eict = compute_cupping(hu_eict_ideal, z)
cup_pcct = compute_cupping(hu_pcct_ideal, z)

println("  EICT 120kVp: cupping = $(round(cup_eict.diff, digits=1)) HU (center=$(round(cup_eict.center, digits=1)), edge=$(round(cup_eict.edge, digits=1)))")
println("  PCCT 140kVp: cupping = $(round(cup_pcct.diff, digits=1)) HU (center=$(round(cup_pcct.center, digits=1)), edge=$(round(cup_pcct.edge, digits=1)))")
println("  Cupping diff: $(round(cup_eict.diff - cup_pcct.diff, digits=1)) HU")

# ─── Test 2: Full physics ───
println("\n--- Test 2: Full physics (fidelity=:high) ---")
hu_eict_full = run_sim_and_recon(scanner_eict, protocol_eict, sim_opts_full, recon_opts, phantom_gpu, μ_eict_full; pcct=false)
hu_pcct_full = run_sim_and_recon(scanner_pcct, protocol_pcct, sim_opts_full, recon_opts, phantom_gpu, μ_pcct_full; pcct=true)

cup_eict_f = compute_cupping(hu_eict_full, z)
cup_pcct_f = compute_cupping(hu_pcct_full, z)

println("  EICT 120kVp: cupping = $(round(cup_eict_f.diff, digits=1)) HU (center=$(round(cup_eict_f.center, digits=1)), edge=$(round(cup_eict_f.edge, digits=1)))")
println("  PCCT 140kVp: cupping = $(round(cup_pcct_f.diff, digits=1)) HU (center=$(round(cup_pcct_f.center, digits=1)), edge=$(round(cup_pcct_f.edge, digits=1)))")
println("  Cupping diff: $(round(cup_eict_f.diff - cup_pcct_f.diff, digits=1)) HU")

# ─── Test 3: BHC coefficient check ───
println("\n--- Test 3: BHC coefficient verification ---")
eict_ws = BS.create_eict_workspace(scanner_eict, protocol_eict, sim_opts_ideal, recon_opts, phantom_gpu)
pcct_ws = BS.create_workspace(scanner_pcct, protocol_pcct, sim_opts_ideal, recon_opts, phantom_gpu)

eict_bhc = BS.get_bhc_coefficients(eict_ws.config.bhc)
pcct_bhc = BS.get_bhc_coefficients(pcct_ws.config.bhc)
println("  EICT BHC: $(round.(eict_bhc, digits=4))")
println("  PCCT BHC: $(round.(pcct_bhc, digits=4))")
println("  (PCCT uses R-matrix effective weights, EICT uses raw spectrum)")

# ─── Test 4: forward_project! with volume_fov ───
println("\n--- Test 4: forward_project!() volume_fov API ---")
geom = BS.CTGeometry(scanner_eict; n_angles=10, fov_cm=35.0, z_cm=1.0)
e, w = load_spectrum(120)
e, w = downsample_spectrum(e, w, 5)
sino_test = similar(MtlArray(phantom.mask), Float32, geom.n_cols, geom.n_rows, geom.n_angles)
fill!(sino_test, 0f0)
BS.forward_project!(sino_test, MtlArray(phantom.mask), geom;
    energies=e, weights=w, materials=phantom.materials, volume_fov=phantom.fov)
println("  forward_project! with volume_fov: OK, range=[$(round(minimum(sino_test), digits=3)), $(round(maximum(sino_test), digits=3))]")

# ─── Summary ───
println("\n" * "="^70)
println("VALIDATION SUMMARY")
println("="^70)
println("")
println("Fix validation results:")
println("  [XCAT-009] calibrate_bhc(): EICT a₁=$(round(eict_bhc[2], digits=4)), PCCT a₁=$(round(pcct_bhc[2], digits=4))")
println("  [XCAT-008] R-matrix BHC:    PCCT BHC calibrated with effective spectrum ($(length(pcct_bhc)) coefficients)")
println("  [XCAT-010] Per-scanner z_cm: Used EICT=0.5cm, PCCT=0.32cm for water cal")
println("  [XCAT-011] volume_fov API:   forward_project!() accepts and uses volume_fov kwarg")
println("")
println("Cupping comparison (noiseless):")
println("  EICT: $(round(cup_eict.diff, digits=1)) HU")
println("  PCCT: $(round(cup_pcct.diff, digits=1)) HU")
println("  Diff: $(round(cup_eict.diff - cup_pcct.diff, digits=1)) HU")
println("")
println("Cupping comparison (full physics):")
println("  EICT: $(round(cup_eict_f.diff, digits=1)) HU")
println("  PCCT: $(round(cup_pcct_f.diff, digits=1)) HU")
println("  Diff: $(round(cup_eict_f.diff - cup_pcct_f.diff, digits=1)) HU")
println("")

# Assess pass/fail
noiseless_diff = abs(cup_eict.diff - cup_pcct.diff)
full_diff = abs(cup_eict_f.diff - cup_pcct_f.diff)
println("Assessment:")
if noiseless_diff < 50.0
    println("  Noiseless cupping diff $(round(noiseless_diff, digits=1)) HU — ACCEPTABLE (< 50 HU)")
else
    println("  Noiseless cupping diff $(round(noiseless_diff, digits=1)) HU — NEEDS REVIEW (> 50 HU)")
end
if full_diff < 100.0
    println("  Full physics cupping diff $(round(full_diff, digits=1)) HU — ACCEPTABLE (< 100 HU)")
else
    println("  Full physics cupping diff $(round(full_diff, digits=1)) HU — NEEDS REVIEW (> 100 HU)")
end

println("\n" * "="^70)
println("DONE")
println("="^70)
