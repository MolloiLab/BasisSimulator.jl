# =============================================================================
# SPEED-001: Tiled Energy Fusion — GPU Benchmark on Metal
# =============================================================================
# Run from project root:
#   julia --project=verification ralph_loops/bench_speed_001.jl
#
# Tests whether processing multiple energy bins per DDA pass (tiled fusion)
# is faster than 234 sequential Siddon calls (unfused path).
#
# Uses the EXISTING siddon_fused_poly_project! kernel at small Val(K)
# to measure per-tile DDA + K-accumulation cost on Metal GPU.

using Metal
import BasisSimulator as BS
import XrayAttenuation as XA

println("=" ^ 70)
println("SPEED-001: Tiled Energy Fusion — GPU Benchmark")
println("GPU: ", Metal.current_device())
println("=" ^ 70)
println()

# Helper: time a GPU operation properly (median of n_runs)
function gpu_time(f, n_runs=5)
    times = Float64[]
    for _ in 1:n_runs
        Metal.synchronize()
        t0 = time_ns()
        f()
        Metal.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return sort(times)[div(length(times)+1, 2)]  # median
end

# =============================================================================
# 1. Setup — same phantom/scanner as SPEED-000 baseline
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
    source_to_isocenter=625.6, source_to_detector=1100.0,
    detector_rows=64, detector_cols=256,
    detector_row_size=0.625, detector_col_size=0.6,
    detector_shape=BS.CURVED_DETECTOR,
    focal_spot_width=1.0, focal_spot_length=1.0,
    target_angle=10.0,
    flat_filter_material=:aluminum, flat_filter_thickness=2.5,
    bowtie_filter=:ge_revolution_large,
    detector_material=:lumex, detector_depth=3.0,
    fill_factor_row=0.9, fill_factor_col=0.9,
    electronic_noise=0, detection_gain=10.0,
)

sim_opts = BS.SimOptions(fidelity=:high, seed=42)
protocol = BS.CTProtocol(
    kVp=120, mA=150.0, views=360,
    rotation_time=1.0, collimation_mm=20.0,
    additional_filters=[("Al", 4.5)],
)
recon_opts = BS.ReconOptions(
    algorithm=:fdk, matrix_size=(128, 128, 32),
    fov_cm=35.0, z_cm=2.0,
)

ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)

println("Phantom: $(size(phantom_cpu.mask)), extent=$(phantom_cpu.extent) cm")
println("Sinogram: $(size(ws.sinogram))")
n_energies = length(ws.energies)
println("Energies: $n_energies bins")
n_materials = size(ws.μ_table_gpu, 1)
println("μ_table: $(size(ws.μ_table_gpu)) ($n_materials materials × $n_energies energies)")
println()

# =============================================================================
# 2. Warmup full simulate!() (JIT + Metal kernel compilation)
# =============================================================================

println("Warming up full simulate!()...")
Metal.@sync BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
println("Warmup done.")
println()

# =============================================================================
# 3. REFERENCE: Unfused forward projection (234 sequential Siddon calls)
# =============================================================================

println("=" ^ 70)
println("REFERENCE: Unfused forward projection ($n_energies sequential Siddon calls)")
println("=" ^ 70)

t_unfused = gpu_time(3) do
    fill!(ws.sinogram, zero(Float32))
    BS._forward_project_poly!(
        ws.sinogram, phantom_gpu.mask, ws.geom,
        ws.energies, ws.weights, ws.mats;
        ws_μ_volume=ws.μ_volume, ws_sino_mono=ws.sino_mono,
        ws_I_transmitted=ws.I_transmitted, ws_weights_norm=ws.weights_norm,
        ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
        ws_μ_table=ws.μ_table, ws_μ_table_gpu=ws.μ_table_gpu,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom_gpu.extent,
        ws_η=ws.η_vec, ws_bowtie_spectral=ws.bowtie_spectral,
        ws_wη_gpu=ws.wη_gpu, fused=false
    )
end
# Save reference sinogram for correctness check
sino_ref = Array(ws.sinogram)
println("  Unfused time: $(round(t_unfused*1000, digits=1)) ms")
println("  Per-bin avg:  $(round(t_unfused/n_energies*1000, digits=2)) ms")
println("  Output range: [$(round(minimum(sino_ref), digits=4)), $(round(maximum(sino_ref), digits=4))]")
println()

# =============================================================================
# 4. REFERENCE: Single Siddon call (Float32 volume)
# =============================================================================

println("=" ^ 70)
println("REFERENCE: Single siddon_forward_project! (Float32 volume)")
println("=" ^ 70)

# Create μ volume for first energy using the full API
Metal.@sync BS.create_μ_volume!(ws.μ_volume, phantom_gpu.mask, ws.mats, ws.energies[1];
    ws_μ_lut_cpu=ws.μ_lut_cpu, ws_μ_lut_gpu=ws.μ_lut_gpu,
    ws_μ_table=ws.μ_table, energy_idx=1, ws_μ_table_gpu=ws.μ_table_gpu)

fill!(ws.sino_mono, zero(Float32))
t_single_siddon = gpu_time(10) do
    BS.siddon_forward_project!(
        ws.sino_mono, ws.μ_volume, ws.geom;
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom_gpu.extent
    )
end
println("  Single Siddon: $(round(t_single_siddon*1000, digits=2)) ms")
println("  × $n_energies = $(round(n_energies * t_single_siddon, digits=3))s (Siddon-only, no μ_volume creation)")
println()

# =============================================================================
# 5. TILED FUSION: Test fused kernel at tile sizes K = 8, 16, 32
# =============================================================================
# Uses the existing siddon_fused_poly_project! with a sub-table of K columns.
# This measures: DDA traversal + K μ_table lookups + K accumulators + K exp + -log
# For the real tiled approach, we'd accumulate partial sums instead of -log,
# but -log cost is negligible. This gives a good estimate of per-tile cost.

println("=" ^ 70)
println("TILED FUSION: Testing fused kernel at small tile sizes")
println("=" ^ 70)
println()

# Full μ_table and wη on CPU for slicing
μ_table_cpu = Array(ws.μ_table_gpu)  # [n_materials, n_energies]
wη_cpu = Array(ws.wη_gpu)            # [n_energies]

# Bowtie sub-arrays: GPU kernel requires a valid bowtie array (not nothing)
# because Metal compiler can't optimize away the dead branch.
# We create sub-arrays of the real bowtie for each K.
nc = size(ws.sinogram, 1)
nr = size(ws.sinogram, 2)
nc_nr = nc * nr
bt_full_cpu = ws.bowtie_spectral !== nothing ? Array(ws.bowtie_spectral) : nothing

results = Dict{Int, Float64}()

for K in [8, 16, 32]
    println("-" ^ 50)
    println("  Tile size K = $K")
    println("-" ^ 50)

    # Extract first K columns for a representative tile
    μ_sub_cpu = μ_table_cpu[:, 1:K]
    wη_sub_cpu = wη_cpu[1:K]

    μ_sub = MtlArray(Float32.(μ_sub_cpu))
    wη_sub = MtlArray(Float32.(wη_sub_cpu))

    # Create bowtie sub-array for K energies (or ones if no bowtie)
    bt_sub_gpu = if bt_full_cpu !== nothing
        MtlArray(Float32.(bt_full_cpu[1:(nc_nr * K)]))
    else
        MtlArray(ones(Float32, nc_nr * K))
    end

    # Warmup (JIT + Metal kernel compilation for this Val(K))
    println("    Compiling kernel for Val($K)...")
    flush(stdout)
    fill!(ws.sinogram, zero(Float32))
    Metal.@sync BS.siddon_fused_poly_project!(
        ws.sinogram, phantom_gpu.mask, ws.geom,
        μ_sub, wη_sub, Val(K);
        volume_extent=phantom_gpu.extent,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_bowtie_spectral=bt_sub_gpu
    )
    println("    Compiled.")

    # Benchmark: time one tile (fused kernel with K energies)
    t_tile = gpu_time(5) do
        fill!(ws.sinogram, zero(Float32))
        BS.siddon_fused_poly_project!(
            ws.sinogram, phantom_gpu.mask, ws.geom,
            μ_sub, wη_sub, Val(K);
            volume_extent=phantom_gpu.extent,
            ws_source_positions=ws.geom_source_positions,
            ws_detector_centers=ws.geom_detector_centers,
            ws_detector_u=ws.geom_detector_u,
            ws_detector_v=ws.geom_detector_v,
            ws_bowtie_spectral=bt_sub_gpu
        )
    end

    n_tiles = ceil(Int, n_energies / K)
    est_total = n_tiles * t_tile
    speedup_vs_unfused = t_unfused / est_total

    results[K] = t_tile

    println("    Per-tile time:    $(round(t_tile*1000, digits=2)) ms")
    println("    N tiles needed:   $n_tiles (ceil($n_energies / $K))")
    println("    Estimated total:  $(round(est_total*1000, digits=1)) ms")
    println("    Speedup vs unfused: $(round(speedup_vs_unfused, digits=2))×")
    println()
end

# (Section 6 removed — bowtie is already included in section 5 benchmarks above)
println()

# =============================================================================
# 7. OVERHEAD ESTIMATION: fill! + accumulation kernel costs
# =============================================================================
# The tiled approach needs:
# - One fill!(I_transmitted, 0) before all tiles
# - After each tile: I_transmitted[idx] += partial_sum[idx] (one AK.foreachindex)
# - After all tiles: sinogram[idx] = -log(I_transmitted[idx])
# The fused kernel already includes -log. The real tiled kernel would skip -log
# and add an accumulation step. Let's measure these overheads.

println("=" ^ 70)
println("OVERHEAD: fill! + accumulate + final -log costs")
println("=" ^ 70)

buf = similar(ws.sinogram)

# fill! cost
t_fill = gpu_time(10) do
    fill!(buf, zero(Float32))
end
println("  fill!(sinogram):    $(round(t_fill*1000, digits=2)) ms")

# Element-wise add: buf[idx] += sinogram[idx]
fill!(buf, zero(Float32))
t_add = gpu_time(10) do
    let b = buf, s = ws.sinogram
        BS.AK.foreachindex(b) do idx
            b[idx] += s[idx]
        end
    end
end
println("  accumulate (+=):    $(round(t_add*1000, digits=2)) ms")

# Final -log
t_log = gpu_time(10) do
    let b = buf, eps_v = Float32(1e-10)
        BS.AK.foreachindex(b) do idx
            b[idx] = -log(max(b[idx], eps_v))
        end
    end
end
println("  -log(I_total):      $(round(t_log*1000, digits=2)) ms")
println()

# =============================================================================
# 8. FULL TILED ESTIMATE: per-tile DDA + accumulation overhead
# =============================================================================
# For each tile: fused_kernel_time (sans -log) + accumulate
# Plus one fill! at start + one -log at end

println("=" ^ 70)
println("FULL TILED ESTIMATE: DDA + accumulation + overhead")
println("=" ^ 70)
println()

for K in [8, 16, 32]
    t_tile = results[K]
    n_tiles = ceil(Int, n_energies / K)

    # The fused kernel includes -log, but tiled kernel wouldn't.
    # Subtract ~t_log from per-tile cost, add one t_log at end.
    # Also add t_add per tile for accumulation into I_transmitted.
    t_per_tile_real = t_tile - t_log + t_add  # remove -log, add accumulate
    t_total = t_fill + n_tiles * t_per_tile_real + t_log
    speedup = t_unfused / t_total

    println("  K=$K: $(n_tiles) tiles × $(round(t_per_tile_real*1000, digits=2))ms + overhead = $(round(t_total*1000, digits=1))ms ($(round(speedup, digits=2))× vs unfused)")
end
println()

# =============================================================================
# 9. SUMMARY
# =============================================================================

println("=" ^ 70)
println("SPEED-001 SUMMARY")
println("=" ^ 70)
println()
println("Baseline:")
println("  Unfused forward proj:   $(round(t_unfused*1000, digits=1)) ms ($n_energies bins)")
println("  Single Siddon (Float32): $(round(t_single_siddon*1000, digits=2)) ms")
println("  Full simulate!() baseline: ~5925 ms (from SPEED-000)")
println()
println("Tiled fusion per-tile costs (DDA + K accums + K exp):")
for K in [8, 16, 32]
    println("  K=$(lpad(K, 2)): $(round(results[K]*1000, digits=2)) ms/tile")
end
println()
println("Signal chain + physics (from SPEED-000): ~670 ms")
println()

# Total simulate!() estimate
best_K = 0
best_total = Inf
for K in [8, 16, 32]
    t_tile = results[K]
    n_tiles = ceil(Int, n_energies / K)
    t_per_tile_real = t_tile - t_log + t_add
    t_fwd = t_fill + n_tiles * t_per_tile_real + t_log
    t_total_sim = t_fwd + 0.670  # signal chain from SPEED-000
    speedup = 5.925 / t_total_sim
    println("Projected simulate!() with K=$K: $(round(t_total_sim, digits=3))s ($(round(speedup, digits=2))× vs baseline 5.925s)")
    if t_total_sim < best_total
        best_total = t_total_sim
        best_K = K
    end
end
println()
println("Best tile size: K=$best_K → projected $(round(best_total, digits=3))s ($(round(5.925/best_total, digits=2))× speedup)")
println()
println("Done!")
