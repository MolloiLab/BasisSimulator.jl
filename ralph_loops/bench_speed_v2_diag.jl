#!/usr/bin/env julia
# Diagnostic benchmark: compare per-tile approaches
# Run: julia --project=verification ralph_loops/bench_speed_v2_diag.jl

using Metal
import BasisSimulator as BS

function gpu_time(f, n_runs=5)
    times = Float64[]
    for _ in 1:n_runs
        Metal.synchronize()
        t0 = time_ns()
        f()
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return sort(times)[div(length(times), 2) + 1]
end

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
n_energies = length(ws.energies)
K = 16

println("Sinogram: $(size(ws.sinogram)), Energies: $n_energies")
println("μ_table_gpu: $(size(ws.μ_table_gpu))")

# Test 1: Original siddon_fused_poly_project! with Val(16) + subset table
println("\n--- Test 1: Original fused kernel with Val(16), subset table ---")
μ_sub = MtlArray(Array(ws.μ_table_gpu)[:, 1:K])
wη_sub = MtlArray(Array(ws.wη_gpu)[1:K])

# Warmup
BS.siddon_fused_poly_project!(ws.sinogram, phantom_gpu.mask, ws.geom,
    μ_sub, wη_sub, Val(K);
    volume_extent=phantom_gpu.extent,
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    ws_bowtie_spectral=ws.bowtie_spectral)
Metal.synchronize()

t_fused_k16 = gpu_time(5) do
    BS.siddon_fused_poly_project!(ws.sinogram, phantom_gpu.mask, ws.geom,
        μ_sub, wη_sub, Val(K);
        volume_extent=phantom_gpu.extent,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_bowtie_spectral=ws.bowtie_spectral)
end
println("Original fused Val(16) per tile: $(round(t_fused_k16 * 1000, digits=1)) ms")
println("× 15 tiles = $(round(t_fused_k16 * 15 * 1000, digits=1)) ms")

# Test 2: New tiled kernel per tile
println("\n--- Test 2: New tiled kernel per tile ---")
fill!(ws.I_transmitted, 0f0)
BS.siddon_tiled_poly_project!(ws.I_transmitted, phantom_gpu.mask, ws.geom,
    ws.μ_table_gpu, ws.wη_gpu, Val(K), Int32(1);
    volume_extent=phantom_gpu.extent,
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    ws_bowtie_spectral=ws.bowtie_spectral)
Metal.synchronize()

t_tiled_k16 = gpu_time(5) do
    BS.siddon_tiled_poly_project!(ws.I_transmitted, phantom_gpu.mask, ws.geom,
        ws.μ_table_gpu, ws.wη_gpu, Val(K), Int32(1);
        volume_extent=phantom_gpu.extent,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_bowtie_spectral=ws.bowtie_spectral)
end
println("Tiled kernel per tile: $(round(t_tiled_k16 * 1000, digits=1)) ms")
println("× 15 tiles = $(round(t_tiled_k16 * 15 * 1000, digits=1)) ms")

# Test 3: Unfused single energy siddon_forward_project!
println("\n--- Test 3: Unfused single energy ---")
# Create μ_volume for energy 1
BS.create_μ_volume!(ws.μ_volume, phantom_gpu.mask, ws.mats, ws.energies[1];
    ws_μ_table=ws.μ_table, energy_idx=1, ws_μ_table_gpu=ws.μ_table_gpu)
Metal.synchronize()

t_unfused_1 = gpu_time(5) do
    BS.siddon_forward_project!(ws.sino_mono, ws.μ_volume, ws.geom;
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom_gpu.extent)
end
println("Single energy siddon: $(round(t_unfused_1 * 1000, digits=1)) ms")
println("× 234 = $(round(t_unfused_1 * 234 * 1000, digits=1)) ms")

# Test 4: Multitile single kernel
println("\n--- Test 4: Multitile single kernel ---")
n_tiles = Int32(cld(n_energies, K))
BS.siddon_multitile_poly_project!(ws.I_transmitted, phantom_gpu.mask, ws.geom,
    ws.μ_table_gpu, ws.wη_gpu, Val(K), n_tiles;
    volume_extent=phantom_gpu.extent,
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    ws_bowtie_spectral=ws.bowtie_spectral)
Metal.synchronize()

t_multitile = gpu_time(5) do
    BS.siddon_multitile_poly_project!(ws.I_transmitted, phantom_gpu.mask, ws.geom,
        ws.μ_table_gpu, ws.wη_gpu, Val(K), n_tiles;
        volume_extent=phantom_gpu.extent,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_bowtie_spectral=ws.bowtie_spectral)
end
println("Multitile single kernel: $(round(t_multitile * 1000, digits=1)) ms")

println("\n" * "=" ^ 60)
println("COMPARISON")
println("=" ^ 60)
println("Fused Val(16) per tile:    $(round(t_fused_k16 * 1000, digits=1)) ms  (discovery: 95.3 ms)")
println("Tiled kernel per tile:     $(round(t_tiled_k16 * 1000, digits=1)) ms")
println("Unfused single energy:     $(round(t_unfused_1 * 1000, digits=1)) ms  (discovery: 22.85 ms)")
println("Multitile single kernel:   $(round(t_multitile * 1000, digits=1)) ms")
println("\nProjected totals:")
println("  15 × fused K=16:  $(round(t_fused_k16 * 15 * 1000, digits=1)) ms")
println("  15 × tiled K=16:  $(round(t_tiled_k16 * 15 * 1000, digits=1)) ms")
println("  234 × unfused:    $(round(t_unfused_1 * 234 * 1000, digits=1)) ms")
println("  Multitile:        $(round(t_multitile * 1000, digits=1)) ms")
