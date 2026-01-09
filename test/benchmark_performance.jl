"""
Performance Benchmarks for BasisSimulator

Measures computational performance across key operations:
1. Spectrum generation
2. Ray tracing throughput
3. Forward simulation (full pipeline)
4. FDK reconstruction
5. Memory usage

Results will be used for:
- Publication performance claims
- Optimization targets
- Comparison with GECATSIM (when available)
"""

using BasisSimulator
using Printf
import XrayAttenuation as XA

println("\n" * "="^70)
println("BASISSIMULATOR PERFORMANCE BENCHMARKS")
println("="^70)

# System info
println("\nSystem Information:")
println("   Julia Version: $(VERSION)")
println("   Threads: $(Threads.nthreads())")
println("   CPU: $(Sys.cpu_info()[1].model)")

# ==============================================================================
# Benchmark 1: Spectrum Generation
# ==============================================================================
println("\n" * "-"^70)
println("BENCHMARK 1: X-RAY SPECTRUM GENERATION")
println("-"^70)

n_runs = 100
times = Float64[]

for i in 1:n_runs
    t_start = time()
    spectrum = generate_spectrum(kVp=120.0, mAs=200.0)
    t_end = time()
    push!(times, (t_end - t_start) * 1000)  # Convert to ms
end

avg_time = sum(times) / length(times)
std_time = sqrt(sum((times .- avg_time).^2) / (length(times) - 1))

println("\nSpectrum Generation ($n_runs runs):")
@printf("   Mean: %.3f ms\n", avg_time)
@printf("   Std:  %.3f ms\n", std_time)
@printf("   Min:  %.3f ms\n", minimum(times))
@printf("   Max:  %.3f ms\n", maximum(times))

# ==============================================================================
# Benchmark 2: Ray Tracing Throughput
# ==============================================================================
println("\n" * "-"^70)
println("BENCHMARK 2: RAY TRACING THROUGHPUT")
println("-"^70)

# Create small test phantom
phantom = create_water_cylinder(
    diameter_mm=100.0,
    height_mm=20.0,
    resolution_mm=2.0
)

protocol = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=150.0, num_projections=10)
geometry = create_aquilion_one(protocol=protocol)

# Setup for ray tracing
grid_meta = GridMeta(
    nx = phantom.grid.nx,
    ny = phantom.grid.ny,
    nz = phantom.grid.nz,
    fov_xy = phantom.grid.fov_xy_cm,
    fov_z = phantom.grid.fov_z_cm
)

unique_mat_ids = sort(unique(phantom.material_ids))
n_materials = length(unique_mat_ids)
id_to_idx = Dict{UInt8, Int}()
for (idx, mat_id) in enumerate(unique_mat_ids)
    id_to_idx[mat_id] = idx
end
id_lut = zeros(Int, 256)
for (mat_id, idx) in id_to_idx
    id_lut[Int(mat_id) + 1] = idx
end

# Benchmark ray tracing
n_rays = 1000
ray_times = Float64[]

angle_idx = 1
source_pos = geometry.source_positions[:, angle_idx]
det_center = geometry.det_centers[:, angle_idx]
u_vec = geometry.det_u_vecs[:, angle_idx]
v_vec = geometry.det_v_vecs[:, angle_idx]

for i in 1:n_rays
    # Random detector pixel
    row = rand(1:geometry.n_rows)
    col = rand(1:geometry.n_cols)

    u_offset = (col - geometry.n_cols/2 - 0.5) * geometry.pixel_width_cm
    v_offset = (row - geometry.n_rows/2 - 0.5) * geometry.pixel_height_cm
    detector_pos = det_center .+ (u_offset .* u_vec) .+ (v_offset .* v_vec)

    t_start = time()
    path_lengths = trace_ray_material_paths(
        grid_meta,
        phantom.material_ids,
        phantom.densities,
        id_lut,
        n_materials,
        source_pos[1], source_pos[2], source_pos[3],
        detector_pos[1], detector_pos[2], detector_pos[3]
    )
    t_end = time()

    push!(ray_times, (t_end - t_start) * 1e6)  # Convert to μs
end

avg_ray_time = sum(ray_times) / length(ray_times)
throughput = 1.0 / (avg_ray_time * 1e-6)  # rays/second

println("\nRay Tracing ($n_rays rays):")
@printf("   Mean time: %.2f μs/ray\n", avg_ray_time)
@printf("   Throughput: %.2e rays/second\n", throughput)
@printf("   Phantom size: %d × %d × %d voxels\n", phantom.grid.nx, phantom.grid.ny, phantom.grid.nz)

# ==============================================================================
# Benchmark 3: Forward Simulation (Single Projection)
# ==============================================================================
println("\n" * "-"^70)
println("BENCHMARK 3: FORWARD SIMULATION (SINGLE PROJECTION)")
println("-"^70)

# Test with few projections to benchmark single projection time
n_test_projs = 10
protocol_single = ScanProtocol(kVp=120.0, mAs=200.0, scan_fov_mm=150.0, num_projections=n_test_projs)
geometry_single = create_aquilion_one(protocol=protocol_single)
spectrum = generate_spectrum(kVp=120.0, mAs=200.0)

println("\nRunning forward simulation ($n_test_projs projections, 320×300 detector)...")
t_start = time()
sinogram_single = simulate_ct_scan(
    phantom = phantom,
    geometry = geometry_single,
    spectrum = spectrum,
    verbose = false
)
t_end = time()

# Divide by number of projections to get average single projection time
single_proj_time = (t_end - t_start) / n_test_projs
rays_per_proj = geometry_single.n_rows * geometry_single.n_cols
rays_throughput = rays_per_proj / single_proj_time

println("\nSingle Projection:")
@printf("   Time: %.3f seconds\n", single_proj_time)
@printf("   Detector: %d × %d = %d rays\n", geometry_single.n_rows, geometry_single.n_cols, rays_per_proj)
@printf("   Throughput: %.2e rays/second\n", rays_throughput)

# Estimate full scan time
n_projs_full = 360
estimated_full_time = single_proj_time * n_projs_full

println("\nEstimated Full Scan (360 projections):")
@printf("   Estimated time: %.1f seconds (%.1f minutes)\n", estimated_full_time, estimated_full_time / 60)

# ==============================================================================
# Benchmark 4: FDK Reconstruction
# ==============================================================================
println("\n" * "-"^70)
println("BENCHMARK 4: FDK RECONSTRUCTION")
println("-"^70)

# Create dummy attenuation sinogram for reconstruction timing
println("\nRunning FDK reconstruction...")
dummy_sino = randn(geometry_single.n_rows, geometry_single.n_cols, n_test_projs) .* 0.1 .+ 0.5

# Create reconstruction grid matching phantom
recon_x_cm = collect(range(-phantom.grid.fov_xy_cm/2, phantom.grid.fov_xy_cm/2, length=phantom.grid.nx))
recon_y_cm = collect(range(-phantom.grid.fov_xy_cm/2, phantom.grid.fov_xy_cm/2, length=phantom.grid.ny))
recon_z_cm = collect(range(-phantom.grid.fov_z_cm/2, phantom.grid.fov_z_cm/2, length=phantom.grid.nz))
angles_deg = rad2deg.(geometry_single.angles)

t_start = time()
recon = reconstruct_fdk(
    dummy_sino,
    geometry_single.SAD_cm,
    geometry_single.SDD_cm,
    geometry_single.pixel_width_cm,
    geometry_single.pixel_height_cm,
    angles_deg,
    recon_x_cm,
    recon_y_cm,
    recon_z_cm,
    filter_type = ramlak
)
t_end = time()

# Divide by number of projections for average time per projection
recon_time = (t_end - t_start) / n_test_projs

println("\nFDK Reconstruction (average per projection):")
@printf("   Time: %.3f seconds\n", recon_time)
@printf("   Output size: %d × %d × %d\n", size(recon, 1), size(recon, 2), size(recon, 3))

# Estimate for full scan
estimated_recon_full = recon_time * n_projs_full

println("\nEstimated Full Reconstruction (360 projections):")
@printf("   Estimated time: %.1f seconds (%.1f minutes)\n", estimated_recon_full, estimated_recon_full / 60)

# ==============================================================================
# Benchmark 5: Memory Usage
# ==============================================================================
println("\n" * "-"^70)
println("BENCHMARK 5: MEMORY USAGE")
println("-"^70)

# Phantom memory
phantom_mem_ids = sizeof(phantom.material_ids) / 1024^2  # MB
phantom_mem_dens = sizeof(phantom.densities) / 1024^2
phantom_mem_total = phantom_mem_ids + phantom_mem_dens

# Sinogram memory (full scan)
sino_full_mem = (geometry_single.n_rows * geometry_single.n_cols * n_projs_full * sizeof(Float64)) / 1024^2

# Reconstruction memory
recon_mem = sizeof(recon) / 1024^2

println("\nMemory Footprint:")
@printf("   Phantom (%d³ voxels): %.1f MB\n", phantom.grid.nx, phantom_mem_total)
@printf("      Material IDs (UInt8): %.1f MB\n", phantom_mem_ids)
@printf("      Densities (Float32): %.1f MB\n", phantom_mem_dens)
@printf("   Sinogram (320×300×360): %.1f MB\n", sino_full_mem)
@printf("   Reconstruction: %.1f MB\n", recon_mem)
@printf("   Total (estimated): %.1f MB\n", phantom_mem_total + sino_full_mem + recon_mem)

# ==============================================================================
# Summary
# ==============================================================================
println("\n" * "="^70)
println("PERFORMANCE SUMMARY")
println("="^70)

println("\nKey Metrics:")
@printf("   Spectrum generation:    %.2f ms\n", avg_time)
@printf("   Ray tracing:            %.2f μs/ray\n", avg_ray_time)
@printf("   Forward projection:     %.2f s (320×300 detector)\n", single_proj_time)
@printf("   FDK reconstruction:     %.2f s (1 projection)\n", recon_time)
@printf("   Memory (full scan):     %.1f MB\n", phantom_mem_total + sino_full_mem + recon_mem)

println("\nEstimated Full CT Scan (360 projections):")
@printf("   Forward simulation:     %.1f min\n", estimated_full_time / 60)
@printf("   Reconstruction:         %.1f min\n", estimated_recon_full / 60)
@printf("   Total pipeline:         %.1f min\n", (estimated_full_time + estimated_recon_full) / 60)

println("\nOptimization Opportunities:")
println("   • Multithreading: Using $(Threads.nthreads()) threads")
println("   • GPU compilation: Reactant.jl (not yet tested)")
println("   • Material LUT: Pre-compute attenuation matrix")

println("\n" * "="^70 * "\n")
