#!/usr/bin/env julia
# XCAT-006: Noiseless PCCT vs EICT comparison
#
# Tests whether cupping artifacts appear WITHOUT noise, isolating
# structural issues in the signal chain from noise interactions.
#
# Uses Gammex 472 phantom (built-in) for fast execution.

using BasisSimulator
using Metal
using Statistics

const BS = BasisSimulator

println("="^70)
println("XCAT-006: Noiseless PCCT vs EICT Comparison")
println("="^70)

# ─── Create Phantom ───
println("\n--- Creating Gammex 472 phantom ---")
phantom = create_gammex_472(n_voxels=128, n_slices=32, fov_cm=35.0, z_cm=4.0)
phantom_gpu = BS.Phantom(MtlArray(phantom.mask), phantom.materials, phantom.voxel_size)
println("  Phantom: $(size(phantom.mask)), FOV: $(phantom.fov) cm")

# ─── Scanners ───
println("\n--- Creating scanners ---")

# EICT scanner (simplified GE Revolution)
scanner_eict = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,
    detector_rows = 32,
    detector_cols = 736,
    detector_row_size = 0.625,
    detector_col_size = 1.0,
    detector_shape = BS.CURVED_DETECTOR,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material = :gos,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    target_angle = 7.0,
)

# PCCT scanner (simplified NAEOTOM Alpha)
scanner_pcct = BS.Scanner(
    source_to_isocenter = 595.0,
    source_to_detector = 1085.5,
    detector_rows = 32,
    detector_cols = 900,
    detector_row_size = 0.4,
    detector_col_size = 0.4,
    detector_shape = BS.CURVED_DETECTOR,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material = :cdte,
    detector_depth = 1.6,
    fill_factor_row = 0.95,
    fill_factor_col = 0.95,
    target_angle = 7.0,
    detector_type = :photon_counting,
    n_energy_bins = 4,
    energy_thresholds = [20.0, 35.0, 55.0, 70.0],
    energy_resolution = 10.0,
    charge_sharing_fwhm = 0.08,
    dead_time_ns = 5.0,
    pixel_mode = :standard,
)

println("  EICT: $(scanner_eict.detector_cols) cols × $(scanner_eict.detector_rows) rows")
println("  PCCT: $(scanner_pcct.detector_cols) cols × $(scanner_pcct.detector_rows) rows")

# ─── Protocols ───
protocol_eict = BS.CTProtocol(kVp=120.0, mA=300.0, views=600, rotation_time=0.5)
protocol_pcct = BS.CTProtocol(kVp=140.0, mA=300.0, views=600, rotation_time=0.25)

# ─── Sim/Recon Options ───
# NOISELESS with polychromatic projection + BHC only
sim_opts_noiseless = BS.SimOptions(
    fidelity = :ideal,        # No physics effects
    use_bhc = true,           # Enable BHC to test its effect
    n_energy_bins = 30,
    seed = 42,
)

# Same but with noise (for comparison)
sim_opts_noisy = BS.SimOptions(
    fidelity = :high,
    pcct_noise_reduction = 0.60,
    n_energy_bins = 30,
    seed = 42,
)

recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (128, 128, 32),
    fov_cm = 35.0,
    z_cm = 4.0,
    filter = :standard,
)

# ─── Water calibration ───
println("\n--- Water calibration ---")
water_phantom = create_gammex_472(n_voxels=64, n_slices=8, fov_cm=35.0, z_cm=0.5)
water_phantom_gpu = BS.Phantom(MtlArray(water_phantom.mask), water_phantom.materials, water_phantom.voxel_size)
water_recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = (64, 64, 8),
    fov_cm = 35.0,
    z_cm = 0.5,
    filter = :standard,
)

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

# EICT water cal (noiseless)
println("  EICT water calibration...")
μ_water_eict = let
    ws = BS.create_eict_workspace(scanner_eict, protocol_eict, sim_opts_noiseless, water_recon_opts, water_phantom_gpu)
    BS.simulate!(ws, water_phantom_gpu, scanner_eict, protocol_eict, sim_opts_noiseless, water_recon_opts)
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_ideal_out, ws.geom, water_recon_opts.matrix_size, filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_ideal_out, ws.geom, water_recon_opts.matrix_size))
    extract_water_mu(vol)
end
println("  μ_water_eict = $(round(μ_water_eict, digits=5)) cm⁻¹")

# PCCT water cal (noiseless)
println("  PCCT water calibration...")
μ_water_pcct = let
    ws = BS.create_workspace(scanner_pcct, protocol_pcct, sim_opts_noiseless, water_recon_opts, water_phantom_gpu)
    BS.simulate!(ws, water_phantom_gpu, scanner_pcct, protocol_pcct, sim_opts_noiseless, water_recon_opts)
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_ideal_out, ws.geom, water_recon_opts.matrix_size, filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_ideal_out, ws.geom, water_recon_opts.matrix_size))
    extract_water_mu(vol)
end
println("  μ_water_pcct = $(round(μ_water_pcct, digits=5)) cm⁻¹")

# ─── Run EICT noiseless ───
println("\n--- EICT noiseless simulation ---")
(recon_eict_hu, sino_eict) = let
    ws = BS.create_eict_workspace(scanner_eict, protocol_eict, sim_opts_noiseless, recon_opts, phantom_gpu)
    @time BS.simulate!(ws, phantom_gpu, scanner_eict, protocol_eict, sim_opts_noiseless, recon_opts)
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_ideal_out, ws.geom, recon_opts.matrix_size, filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_ideal_out, ws.geom, recon_opts.matrix_size))
    hu = 1000f0 .* (vol .- Float32(μ_water_eict)) ./ Float32(μ_water_eict)
    (hu, copy(ws.sino_ideal_out))
end

# ─── Run PCCT noiseless ───
println("\n--- PCCT noiseless simulation ---")
(recon_pcct_hu, sino_pcct) = let
    ws = BS.create_workspace(scanner_pcct, protocol_pcct, sim_opts_noiseless, recon_opts, phantom_gpu)
    @time BS.simulate!(ws, phantom_gpu, scanner_pcct, protocol_pcct, sim_opts_noiseless, recon_opts)
    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_ideal_out, ws.geom, recon_opts.matrix_size, filter=BS.StandardFilter())
    vol = Array(BS.reconstruct!(ws_fdk, ws.sino_ideal_out, ws.geom, recon_opts.matrix_size))
    hu = 1000f0 .* (vol .- Float32(μ_water_pcct)) ./ Float32(μ_water_pcct)
    (hu, copy(ws.sino_ideal_out))
end

# ─── Analysis ───
println("\n" * "="^70)
println("RESULTS — NOISELESS COMPARISON")
println("="^70)

z = size(recon_eict_hu, 3) ÷ 2

println("\n--- Central slice (z=$z) statistics ---")
for (label, vol, μw) in [("EICT 120kVp", recon_eict_hu, μ_water_eict),
                           ("PCCT 140kVp", recon_pcct_hu, μ_water_pcct)]
    slice = vol[:, :, z]
    # Center ROI
    nx, ny = size(slice)
    cx, cy = nx ÷ 2, ny ÷ 2
    r_center = 5
    center_vals = [slice[i,j] for i in (cx-r_center):(cx+r_center)
                                   for j in (cy-r_center):(cy+r_center)
                                   if (i-cx)^2 + (j-cy)^2 <= r_center^2]

    # Radial profile (horizontal through center)
    profile = slice[:, cy]

    println("\n$label:")
    println("  μ_water = $(round(μw, digits=5)) cm⁻¹")
    println("  Center ROI (r=5): mean=$(round(mean(center_vals), digits=1)) HU, std=$(round(std(center_vals), digits=1)) HU")
    println("  Profile range: min=$(round(minimum(profile), digits=1)), max=$(round(maximum(profile), digits=1)) HU")

    # Cupping metric: average center vs average edge (within phantom body)
    # Find valid region (non-background)
    valid = abs.(slice) .< 2000  # exclude background/air
    if sum(valid) > 0
        r_inner = nx ÷ 8
        r_outer = nx ÷ 3
        center_mask = [(i-cx)^2 + (j-cy)^2 <= r_inner^2 for i in 1:nx, j in 1:ny] .& valid
        edge_mask = [(i-cx)^2 + (j-cy)^2 > r_inner^2 && (i-cx)^2 + (j-cy)^2 <= r_outer^2 for i in 1:nx, j in 1:ny] .& valid

        if sum(center_mask) > 0 && sum(edge_mask) > 0
            center_mean = mean(slice[center_mask])
            edge_mean = mean(slice[edge_mask])
            println("  CUPPING: center=$(round(center_mean, digits=1)) HU, edge=$(round(edge_mean, digits=1)) HU, diff=$(round(center_mean - edge_mean, digits=1)) HU")
        end
    end
end

# ─── Sinogram comparison ───
println("\n--- Sinogram statistics ---")
for (label, sino) in [("EICT", sino_eict), ("PCCT", sino_pcct)]
    println("$label sinogram: size=$(size(sino)), range=[$(round(minimum(sino), digits=3)), $(round(maximum(sino), digits=3))]")
end

println("\n" * "="^70)
println("DONE")
println("="^70)
