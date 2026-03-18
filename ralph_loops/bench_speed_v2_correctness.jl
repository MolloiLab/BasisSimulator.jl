#!/usr/bin/env julia
# Correctness test: tiled vs unfused output comparison
# Run: julia --project=verification ralph_loops/bench_speed_v2_correctness.jl

using Metal
import BasisSimulator as BS

println("=" ^ 60)
println("SPEED BUILD V2 — Correctness Verification")
println("GPU: ", Metal.current_device())
println("=" ^ 60)

# Setup
phantom_cpu = BS.create_gammex_472(n_voxels=128, fov_cm=35.0, z_cm=4.0)
phantom_gpu = BS.Phantom(
    MtlArray(phantom_cpu.mask), phantom_cpu.materials,
    phantom_cpu.voxel_size, phantom_cpu.origin, phantom_cpu.extent,
)
scanner = BS.Scanner(
    source_to_isocenter=625.6, source_to_detector=1100.0,
    detector_rows=64, detector_cols=256,
    detector_row_size=0.625, detector_col_size=0.6,
    detector_shape=BS.CURVED_DETECTOR,
    focal_spot_width=1.0, focal_spot_length=1.0, target_angle=10.0,
    flat_filter_material=:aluminum, flat_filter_thickness=2.5,
    bowtie_filter=:ge_revolution_large,
    detector_material=:lumex, detector_depth=3.0,
    fill_factor_row=0.9, fill_factor_col=0.9,
    electronic_noise=0, detection_gain=10.0,
)
sim_opts = BS.SimOptions(fidelity=:high, seed=42)
protocol = BS.CTProtocol(kVp=120, mA=150.0, views=360, rotation_time=1.0,
    collimation_mm=20.0, additional_filters=[("Al", 4.5)])
recon_opts = BS.ReconOptions(algorithm=:fdk, matrix_size=(128, 128, 32),
    fov_cm=35.0, z_cm=2.0)

ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)

# Run 1: TILED path (default, fused=false, μ_table_gpu available)
println("\nRunning tiled path...")
BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
Metal.synchronize()
sino_tiled = Array(ws.sinogram)

# Run 2: UNFUSED path (force no μ_table_gpu by passing fused=false, no workspace)
println("Running unfused path (reference)...")
fill!(ws.sinogram, 0f0)
# Call _forward_project_poly! directly with fused=false and WITHOUT ws_μ_table_gpu
# to force the sequential energy loop path
BS._forward_project_poly!(ws.sinogram, phantom_gpu.mask, ws.geom,
    ws.energies, ws.weights, ws.mats;
    ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
    ws_I_transmitted=ws.I_transmitted,
    ws_weights_norm=ws.weights_norm,
    ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
    ws_μ_table=ws.μ_table,
    ws_μ_table_gpu=nothing,  # Force unfused path
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    volume_extent=phantom_gpu.extent,
    ws_η=ws.η_vec,
    ws_bowtie_spectral=ws.bowtie_spectral,
    ws_wη_gpu=nothing,  # Force unfused path
    fused=false)
Metal.synchronize()
sino_unfused = Array(ws.sinogram)

# Compare
println("\n" * "=" ^ 60)
println("CORRECTNESS COMPARISON: Tiled vs Unfused")
println("=" ^ 60)

# Note: sino_tiled includes full signal chain (noise, BHC, etc.)
# sino_unfused is just the forward projection
# For a fair comparison, we need to compare the forward projection output only
# Let's re-run tiled as forward proj only

fill!(ws.sinogram, 0f0)
fill!(ws.I_transmitted, 0f0)
BS._forward_project_poly!(ws.sinogram, phantom_gpu.mask, ws.geom,
    ws.energies, ws.weights, ws.mats;
    ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
    ws_I_transmitted=ws.I_transmitted,
    ws_weights_norm=ws.weights_norm,
    ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
    ws_μ_table=ws.μ_table,
    ws_μ_table_gpu=ws.μ_table_gpu,
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    volume_extent=phantom_gpu.extent,
    ws_η=ws.η_vec,
    ws_bowtie_spectral=ws.bowtie_spectral,
    ws_wη_gpu=ws.wη_gpu,
    fused=false)
Metal.synchronize()
sino_tiled_fwdonly = Array(ws.sinogram)

# Compare forward projection outputs
diff = abs.(sino_tiled_fwdonly .- sino_unfused)
max_diff = maximum(diff)
mean_diff = sum(diff) / length(diff)
rms_diff = sqrt(sum(diff.^2) / length(diff))

# Compare non-zero elements
nonzero = sino_unfused .!= 0
if sum(nonzero) > 0
    rel_diff = diff[nonzero] ./ abs.(sino_unfused[nonzero])
    max_rel = maximum(rel_diff)
    mean_rel = sum(rel_diff) / sum(nonzero)
else
    max_rel = NaN
    mean_rel = NaN
end

println("Max abs diff:  $(max_diff)")
println("Mean abs diff: $(mean_diff)")
println("RMS diff:      $(rms_diff)")
println("Max rel diff:  $(max_rel)")
println("Mean rel diff: $(mean_rel)")
println()
println("Sinogram range (unfused): [$(minimum(sino_unfused)), $(maximum(sino_unfused))]")
println("Sinogram range (tiled):   [$(minimum(sino_tiled_fwdonly)), $(maximum(sino_tiled_fwdonly))]")
println()

if max_diff < 1e-4
    println("✓ PASS — Tiled output matches unfused within Float32 tolerance (max diff $(max_diff) < 1e-4)")
elseif max_diff < 1e-3
    println("~ CLOSE — Max diff $(max_diff) is small but >1e-4. Check if acceptable.")
else
    println("✗ FAIL — Max diff $(max_diff) too large! Investigate.")
end
