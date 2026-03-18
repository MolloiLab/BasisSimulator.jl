#!/usr/bin/env julia
# GPU benchmark for speed build v2 stories
# Run: julia --project=verification ralph_loops/bench_speed_v2.jl

using Metal
import BasisSimulator as BS

println("=" ^ 60)
println("SPEED BUILD V2 — GPU Benchmark")
println("GPU: ", Metal.current_device())
println("=" ^ 60)

# Helper: time a GPU operation properly
function gpu_time(f, n_runs=5)
    times = Float64[]
    for _ in 1:n_runs
        Metal.synchronize()
        t0 = time_ns()
        f()
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return sort(times)[div(length(times), 2) + 1]  # median
end

# =============================================================================
# Setup — match bench_speed.jl (Gammex 472, 128^3, :high fidelity)
# =============================================================================

phantom_cpu = BS.create_gammex_472(n_voxels=128, fov_cm=35.0, z_cm=4.0)
phantom_gpu = BS.Phantom(
    MtlArray(phantom_cpu.mask),
    phantom_cpu.materials,
    phantom_cpu.voxel_size,
    phantom_cpu.origin,
    phantom_cpu.extent,
)

scanner = BS.Scanner(
    source_to_isocenter=625.6,
    source_to_detector=1100.0,
    detector_rows=64,
    detector_cols=256,
    detector_row_size=0.625,
    detector_col_size=0.6,
    detector_shape=BS.CURVED_DETECTOR,
    focal_spot_width=1.0,
    focal_spot_length=1.0,
    target_angle=10.0,
    flat_filter_material=:aluminum,
    flat_filter_thickness=2.5,
    bowtie_filter=:ge_revolution_large,
    detector_material=:lumex,
    detector_depth=3.0,
    fill_factor_row=0.9,
    fill_factor_col=0.9,
    electronic_noise=0,
    detection_gain=10.0,
)

sim_opts = BS.SimOptions(fidelity=:high, seed=42)
protocol = BS.CTProtocol(
    kVp=120, mA=150.0, views=360,
    rotation_time=1.0, collimation_mm=20.0,
    additional_filters=[("Al", 4.5)],
)
recon_opts = BS.ReconOptions(
    algorithm=:fdk,
    matrix_size=(128, 128, 32),
    fov_cm=35.0,
    z_cm=2.0,
)

ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)

println("\nPhantom: $(size(phantom_cpu.mask)), extent=$(phantom_cpu.extent) cm")
println("Sinogram: $(size(ws.sinogram))")
println("Energies: $(length(ws.energies)) bins")
println("μ_table_gpu: $(size(ws.μ_table_gpu)) (padded for K=16 tiling)")
println("wη_gpu: $(length(ws.wη_gpu)) (padded)")
println("Has bowtie: $(ws.bowtie_spectral !== nothing)")
println("Has signal chain: $(ws.has_signal_chain)")

# =============================================================================
# Warmup (includes JIT for tiled kernel Val(16))
# =============================================================================
println("\nWarming up (JIT compilation for tiled kernel)...")
Metal.@sync BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
println("Warmup done.")

# =============================================================================
# Benchmark: Full simulate!() with tiled projection
# =============================================================================
println("\n--- Tiled path (K=16, V2-002) ---")
t_tiled = gpu_time(5) do
    BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
end
println("Tiled simulate!() median: $(round(t_tiled * 1000, digits=1)) ms")

# =============================================================================
# Results
# =============================================================================
println("\n" * "=" ^ 60)
println("RESULTS")
println("=" ^ 60)
println("Tiled (V2-002) median:    $(round(t_tiled * 1000, digits=1)) ms")
println("Unfused (V2-001):         5,259 ms")
println("Fused baseline:           19,182 ms")
println("Speedup vs unfused:       $(round(5.259 / t_tiled, digits=2))×")
println("Speedup vs fused default: $(round(19.182 / t_tiled, digits=2))×")
println("\nTargets:")
println("  Forward proj < 1,800 ms (3.0× min): $(t_tiled < 1.8 ? "likely ✓" : "check forward proj separately")")
println("  simulate!()  < 2,000 ms:            $(t_tiled < 2.0 ? "✓ PASS" : "✗ FAIL")")
println("  10× target   < 1,918 ms:            $(t_tiled < 1.918 ? "✓ PASS" : "✗ FAIL")")
