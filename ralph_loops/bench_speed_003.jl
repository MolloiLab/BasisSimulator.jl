# =============================================================================
# SPEED-003/004: Per-bin Cost Decomposition + Signal Chain Optimization
# =============================================================================
# Run from project root:
#   julia --project=verification ralph_loops/bench_speed_003.jl
#
# Goals:
# 1. Decompose the 6ms per-bin floor cost (exp vs DDA-loop FMA vs overhead)
# 2. Quantify redundant -log/exp round-trip between forward proj and signal chain
# 3. Measure signal chain fusion potential
# 4. Measure air voxel fraction and estimate empty-space skipping benefit
# 5. Test μ_table access patterns

using Metal
import BasisSimulator as BS
const AK = BS.AK

println("=" ^ 70)
println("SPEED-003/004: Per-bin Cost Decomposition + Signal Chain")
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
# 1. Setup — same phantom/scanner as SPEED-000/001 baseline
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
n_elements = length(ws.sinogram)
println("Energies: $n_energies bins")
println("Sinogram elements: $n_elements ($(round(n_elements * 4 / 1e6, digits=1)) MB)")
println()

# Warmup
println("Warming up full simulate!()...")
Metal.@sync BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
println("Warmup done.")
println()

# =============================================================================
# 2. AIR VOXEL ANALYSIS — what fraction of the phantom is air?
# =============================================================================

println("=" ^ 70)
println("2. AIR VOXEL ANALYSIS")
println("=" ^ 70)

mask_cpu = Array(phantom_cpu.mask)
total_voxels = length(mask_cpu)
air_voxels = count(x -> x == 0x0000, mask_cpu)
tissue_voxels = total_voxels - air_voxels
air_fraction = air_voxels / total_voxels

# Count unique materials
unique_mats = sort(unique(mask_cpu))
println("  Total voxels:   $total_voxels")
println("  Air voxels:     $air_voxels ($(round(100*air_fraction, digits=1))%)")
println("  Tissue voxels:  $tissue_voxels ($(round(100*(1-air_fraction), digits=1))%)")
println("  Unique materials: $(length(unique_mats)) — indices: $unique_mats")
println()

# Check if air material (index 0) has zero μ for all energies
μ_table_cpu = Array(ws.μ_table_gpu)
air_μ_max = maximum(abs.(μ_table_cpu[1, :]))  # mat index 0 → row 1 (0-indexed + 1)
println("  Air (mat=0) max |μ|: $air_μ_max (should be ~0)")
println()

# =============================================================================
# 3. ELEMENTWISE MICROBENCHMARKS — exp, log, FMA costs on Metal
# =============================================================================

println("=" ^ 70)
println("3. ELEMENTWISE MICROBENCHMARKS ($(n_elements) elements)")
println("=" ^ 70)

buf_a = MtlArray(randn(Float32, size(ws.sinogram)) .* 0.1f0 .+ 2.0f0)
buf_b = similar(buf_a)
fill!(buf_b, 1.0f0)

# Warmup all kernels
Metal.@sync begin
    let a = buf_a
        AK.foreachindex(a) do idx; a[idx] = exp(-a[idx]); end
    end
end
Metal.@sync begin
    let a = buf_a, b = buf_b
        AK.foreachindex(a) do idx; b[idx] = -log(max(a[idx], Float32(1e-10))); end
    end
end
Metal.@sync begin
    let a = buf_a, b = buf_b
        AK.foreachindex(a) do idx; b[idx] += a[idx] * Float32(0.02); end
    end
end

# exp(-x)
fill!(buf_a, 2.0f0)
t_exp = gpu_time(10) do
    let a = buf_a
        AK.foreachindex(a) do idx
            a[idx] = exp(-a[idx])
        end
    end
end
println("  exp(-x):         $(round(t_exp*1000, digits=2)) ms")

# -log(x)
fill!(buf_a, 0.5f0)
t_log = gpu_time(10) do
    let a = buf_a
        AK.foreachindex(a) do idx
            a[idx] = -log(max(a[idx], Float32(1e-10)))
        end
    end
end
println("  -log(x):         $(round(t_log*1000, digits=2)) ms")

# FMA: b += a * scalar
fill!(buf_a, 0.02f0)
fill!(buf_b, 1.0f0)
t_fma = gpu_time(10) do
    let a = buf_a, b = buf_b
        AK.foreachindex(a) do idx
            b[idx] += a[idx] * Float32(0.02)
        end
    end
end
println("  FMA (b+=a*c):    $(round(t_fma*1000, digits=2)) ms")

# Simple copy
t_copy = gpu_time(10) do
    let a = buf_a, b = buf_b
        AK.foreachindex(a) do idx
            b[idx] = a[idx]
        end
    end
end
println("  copy (b=a):      $(round(t_copy*1000, digits=2)) ms")

# 16× exp (simulating K=16 Beer-Lambert per element)
fill!(buf_a, 2.0f0)
t_exp16 = gpu_time(10) do
    let a = buf_a
        AK.foreachindex(a) do idx
            v = a[idx]
            s = exp(-v) + exp(-v*Float32(1.01)) + exp(-v*Float32(1.02)) + exp(-v*Float32(1.03))
            s += exp(-v*Float32(1.04)) + exp(-v*Float32(1.05)) + exp(-v*Float32(1.06)) + exp(-v*Float32(1.07))
            s += exp(-v*Float32(1.08)) + exp(-v*Float32(1.09)) + exp(-v*Float32(1.10)) + exp(-v*Float32(1.11))
            s += exp(-v*Float32(1.12)) + exp(-v*Float32(1.13)) + exp(-v*Float32(1.14)) + exp(-v*Float32(1.15))
            a[idx] = s
        end
    end
end
println("  16× exp per elem: $(round(t_exp16*1000, digits=2)) ms")
println("  → per exp call:  $(round(t_exp16/16*1000, digits=2)) ms (for $n_elements elements)")
println()

# Bandwidth analysis
mem_bytes = n_elements * 4  # Float32
bandwidth_GBs = 400.0  # M3 Max approx
t_bandwidth = (2 * mem_bytes) / (bandwidth_GBs * 1e9) * 1000  # ms, read+write
println("  Theoretical bandwidth floor (read+write): $(round(t_bandwidth, digits=2)) ms")
println("  Measured copy:  $(round(t_copy*1000, digits=2)) ms")
println("  Measured exp:   $(round(t_exp*1000, digits=2)) ms ($(round(t_exp/t_copy, digits=1))× copy)")
println()

# =============================================================================
# 4. REDUNDANT -LOG/EXP ROUND-TRIP QUANTIFICATION
# =============================================================================

println("=" ^ 70)
println("4. REDUNDANT -LOG/EXP ROUND-TRIP")
println("=" ^ 70)
println()
println("Forward proj outputs: sinogram = -log(I_total)")
println("Signal chain starts:  I = exp(-sinogram) = I_total  ← REDUNDANT!")
println()
println("Cost of redundant round-trip:")
println("  -log at end of forward proj:  $(round(t_log*1000, digits=2)) ms")
println("  exp at start of signal chain: $(round(t_exp*1000, digits=2)) ms")
println("  TOTAL WASTED:                 $(round((t_log+t_exp)*1000, digits=2)) ms")
println()

# =============================================================================
# 5. SIGNAL CHAIN INDIVIDUAL STEP TIMING
# =============================================================================

println("=" ^ 70)
println("5. SIGNAL CHAIN STEP TIMING")
println("=" ^ 70)

# First: time the full simulate!() to re-confirm baseline
t_simulate = gpu_time(3) do
    BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
end
println("  Full simulate!(): $(round(t_simulate*1000, digits=1)) ms")
println()

# Time forward projection (unfused)
t_fwd_proj = gpu_time(3) do
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
println("  Forward proj (unfused): $(round(t_fwd_proj*1000, digits=1)) ms")

# Time physics pipeline only
t_physics = gpu_time(5) do
    BS._apply_physics_no_noise!(ws.sinogram, ws.geom, ws.config;
        ws_output=ws.physics_output,
        ws_scatter_kernel=ws.scatter_kernel,
        ws_scatter_correct_kernel=ws.scatter_correct_kernel,
        ws_scatter_temp=ws.scatter_temp,
        ws_scatter_kernel_1d=ws.scatter_kernel_1d,
        ws_scatter_correct_kernel_1d=ws.scatter_correct_kernel_1d,
        ws_crosstalk_kernel=ws.crosstalk_kernel,
        ws_optical_crosstalk_kernel=ws.optical_crosstalk_kernel,
        ws_focal_spot_kernel=ws.focal_spot_kernel,
        ws_flat_filter_projection=ws.flat_filter_projection,
        ws_bowtie_projection=nothing,
        ws_lag_output=ws.physics_output,
        ws_lag_intensity=ws.lag_intensity,
        ws_lag_coeffs=ws.lag_coeffs)
end
println("  Physics pipeline: $(round(t_physics*1000, digits=1)) ms")

# Time individual signal chain steps
# Step 3: exp(-sinogram) → intensity domain
fill!(ws.sinogram, 2.0f0)  # realistic line integral value
t_step_exp = gpu_time(10) do
    let sino = ws.sinogram
        AK.foreachindex(sino) do idx
            sino[idx] = exp(-clamp(sino[idx], Float32(-1), Float32(15)))
        end
    end
end
println("  Step 3 - exp(-sino):    $(round(t_step_exp*1000, digits=2)) ms")

# Step 6: Air scan build (bowtie ref application)
fill!(ws.air_scan, one(Float32))
if ws.bowtie_air_reference !== nothing
    t_step_air = gpu_time(10) do
        let air = ws.air_scan, ref = ws.bowtie_air_reference, nc = size(ws.air_scan, 1), nr = size(ws.air_scan, 2)
            AK.foreachindex(air) do idx
                idx_0 = Int32(idx - 1)
                col = (idx_0 % Int32(nc)) + Int32(1)
                row = ((idx_0 ÷ Int32(nc)) % Int32(nr)) + Int32(1)
                ref_idx = col + (row - 1) * nc
                air[idx] *= ref[ref_idx]
            end
        end
    end
    println("  Step 6 - Air scan build: $(round(t_step_air*1000, digits=2)) ms")
else
    t_step_air = 0.0
    println("  Step 6 - Air scan build: SKIPPED (no bowtie ref)")
end

# Step 7: Calibration (sino / air)
fill!(ws.sinogram, 0.5f0)
fill!(ws.air_scan, 1.0f0)
eps_val = Float32(1e-10)
t_step_cal = gpu_time(10) do
    let sino = ws.sinogram, air = ws.air_scan, eps = eps_val
        AK.foreachindex(sino) do idx
            air_val = max(air[idx], eps)
            sino[idx] = sino[idx] / air_val
        end
    end
end
println("  Step 7 - Calibration:   $(round(t_step_cal*1000, digits=2)) ms")

# Step 9: -log transform
fill!(ws.sinogram, 0.5f0)
t_step_log = gpu_time(10) do
    let sino = ws.sinogram, eps = eps_val
        AK.foreachindex(sino) do idx
            sino[idx] = -log(max(sino[idx], eps))
        end
    end
end
println("  Step 9 - -log(sino):    $(round(t_step_log*1000, digits=2)) ms")

# Sum of individual steps
t_chain_sum = t_step_exp + t_step_air + t_step_cal + t_step_log
println()
println("  Individual steps sum: $(round(t_chain_sum*1000, digits=1)) ms")
println()

# =============================================================================
# 6. FUSED SIGNAL CHAIN PROTOTYPE
# =============================================================================

println("=" ^ 70)
println("6. FUSED SIGNAL CHAIN — combine exp→cal→-log into 1 kernel")
println("=" ^ 70)

# The key insight: for the IDEAL path (no noise), the signal chain simplifies.
# After forward projection: sinogram = -log(I_total)
# After signal chain: output = sinogram + log(bowtie_air_ref)
# This is just ONE elementwise add if we precompute log(bowtie_air_ref)!
#
# For the NOISY path: we need intensity domain for noise application.
# But we can still skip the -log/exp round-trip by outputting I_total directly.
#
# Let's measure the fused approach:

# Approach A: Full bypass — just add log(bowtie_ref)
# (Only valid for ideal sinogram without noise)
if ws.bowtie_air_reference !== nothing
    # Precompute log(bowtie_ref) on GPU
    ref_cpu = Array(ws.bowtie_air_reference)
    log_ref_cpu = Float32.(-log.(max.(ref_cpu, 1f-10)))
    log_ref_gpu = MtlArray(log_ref_cpu)
    nc = size(ws.sinogram, 1)
    nr = size(ws.sinogram, 2)

    # Fused: sinogram += log_bowtie_ref (broadcasting over angles)
    fill!(ws.sinogram, 2.0f0)
    # Warmup
    Metal.@sync begin
        let sino = ws.sinogram, lref = log_ref_gpu, nc = Int32(nc), nr = Int32(nr)
            AK.foreachindex(sino) do idx
                idx_0 = Int32(idx - 1)
                col = (idx_0 % nc) + Int32(1)
                row = ((idx_0 ÷ nc) % nr) + Int32(1)
                ref_idx = col + (row - 1) * nc
                sino[idx] += lref[ref_idx]
            end
        end
    end

    t_fused_ideal = gpu_time(10) do
        let sino = ws.sinogram, lref = log_ref_gpu, nc = Int32(nc), nr = Int32(nr)
            AK.foreachindex(sino) do idx
                idx_0 = Int32(idx - 1)
                col = (idx_0 % nc) + Int32(1)
                row = ((idx_0 ÷ nc) % nr) + Int32(1)
                ref_idx = col + (row - 1) * nc
                sino[idx] += lref[ref_idx]
            end
        end
    end
    println("  Fused ideal (sino += log_ref):  $(round(t_fused_ideal*1000, digits=2)) ms")
    println("  vs separate chain steps:        $(round(t_chain_sum*1000, digits=1)) ms")
    println("  Savings:                        $(round((t_chain_sum - t_fused_ideal)*1000, digits=1)) ms")
else
    t_fused_ideal = t_step_log  # fallback: at least save the exp round-trip
    println("  No bowtie ref — ideal path saves only log/exp round-trip")
end
println()

# Approach B: Fused noisy path — exp→heel→gain→/air→noise→-log in 2 kernels
# Kernel 1: prep = exp(-sino) / air_ref  (skip separate heel/gain since they cancel)
# Kernel 2: output = -log(max(prep + noise, eps))  (with low signal correction)
#
# But low_signal_correction is a spatial filter (neighbors), not fuseable with pointwise.
# So minimum kernels for noisy path: 3 (prep, low_signal, -log+noise)

fill!(ws.sinogram, 2.0f0)
fill!(ws.air_scan, 1.0f0)
# Warmup
Metal.@sync begin
    let sino = ws.sinogram, air = ws.air_scan, eps = eps_val
        AK.foreachindex(sino) do idx
            air_val = max(air[idx], eps)
            sino[idx] = -log(max(exp(-clamp(sino[idx], Float32(-1), Float32(15))) / air_val, eps))
        end
    end
end

fill!(ws.sinogram, 2.0f0)
t_fused_noisy = gpu_time(10) do
    let sino = ws.sinogram, air = ws.air_scan, eps = eps_val
        AK.foreachindex(sino) do idx
            air_val = max(air[idx], eps)
            sino[idx] = -log(max(exp(-clamp(sino[idx], Float32(-1), Float32(15))) / air_val, eps))
        end
    end
end
println("  Fused noisy (exp→/air→-log):   $(round(t_fused_noisy*1000, digits=2)) ms")
println("  vs separate (exp+cal+log):      $(round((t_step_exp+t_step_cal+t_step_log)*1000, digits=1)) ms")
println("  Savings:                         $(round(((t_step_exp+t_step_cal+t_step_log) - t_fused_noisy)*1000, digits=1)) ms")
println()

# =============================================================================
# 7. PER-BIN COST DECOMPOSITION — exp() fraction of 6ms
# =============================================================================

println("=" ^ 70)
println("7. PER-BIN COST DECOMPOSITION")
println("=" ^ 70)

# From SPEED-001: per-tile K=16 = 97.45ms, per-bin marginal ≈ 6ms
# The per-bin cost inside the fused kernel has TWO parts:
# A) DDA loop work: μ_table lookup + FMA per voxel (×~128 voxels avg)
# B) Beer-Lambert: exp() + FMA per ray (×2.95M rays)
#
# We can estimate (B) from our exp microbenchmark:
# 1 exp per ray per bin = t_exp for 2.95M elements

exp_per_bin_ms = t_exp * 1000  # ms for 2.95M exp calls
fma_per_bin_ms = t_fma * 1000  # ms for 2.95M FMAs

# Average DDA path length (voxels per ray)
# For 128×128×15 volume with 35cm extent:
# Diagonal of volume ≈ sqrt(128² + 128² + 15²) ≈ 182 voxels
# Average ray through center ≈ 128 voxels
# But many rays are shorter (hit edges, thin z)
# Rough estimate: ~60-80 effective voxels per ray on average
avg_path_voxels = 80  # conservative estimate

# DDA loop work per bin per ray: 1 μ_table read + 1 FMA
# Per voxel: ~same as 1 FMA (memory read from L1 cache + multiply-add)
dda_work_per_bin_est = avg_path_voxels * t_fma * 1000 / n_elements  # ms per ray (scaled)
# Total DDA work for 2.95M rays:
dda_work_total_ms = avg_path_voxels * fma_per_bin_ms  # rough: path_len × FMA_time

println("  exp() for 2.95M elements:  $(round(exp_per_bin_ms, digits=2)) ms")
println("  FMA for 2.95M elements:    $(round(fma_per_bin_ms, digits=2)) ms")
println()
println("  Per-bin cost estimate (assuming ~$avg_path_voxels voxels/ray avg):")
println("    Beer-Lambert (1 exp + 1 FMA per ray):  $(round(exp_per_bin_ms + fma_per_bin_ms, digits=2)) ms")
println("    DDA work ($avg_path_voxels × read+FMA per ray): $(round(dda_work_total_ms, digits=2)) ms")
println("    Sum:                                    $(round(exp_per_bin_ms + fma_per_bin_ms + dda_work_total_ms, digits=2)) ms")
println("    Measured per-bin from SPEED-001:         ~6.0 ms")
println()

# =============================================================================
# 8. TILED FUSION CONFIRMATION — K=16 timing
# =============================================================================

println("=" ^ 70)
println("8. TILED FUSION K=16 — Confirm SPEED-001 Numbers")
println("=" ^ 70)

K = 16
μ_table_cpu_full = Array(ws.μ_table_gpu)
wη_cpu_full = Array(ws.wη_gpu)
μ_sub_cpu = μ_table_cpu_full[:, 1:K]
wη_sub_cpu = wη_cpu_full[1:K]
μ_sub = MtlArray(Float32.(μ_sub_cpu))
wη_sub = MtlArray(Float32.(wη_sub_cpu))

nc = size(ws.sinogram, 1)
nr = size(ws.sinogram, 2)
nc_nr = nc * nr
bt_full_cpu = ws.bowtie_spectral !== nothing ? Array(ws.bowtie_spectral) : nothing
bt_sub_gpu = if bt_full_cpu !== nothing
    MtlArray(Float32.(bt_full_cpu[1:(nc_nr * K)]))
else
    MtlArray(ones(Float32, nc_nr * K))
end

# Warmup
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

# Benchmark
t_tile_k16 = gpu_time(5) do
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
t_tiled_total = n_tiles * t_tile_k16
println("  K=16 per-tile: $(round(t_tile_k16*1000, digits=2)) ms")
println("  $n_tiles tiles × $(round(t_tile_k16*1000, digits=2))ms = $(round(t_tiled_total*1000, digits=1)) ms")
println("  Per-bin marginal: $(round(t_tile_k16/K*1000, digits=2)) ms")
println()

# =============================================================================
# 9. ALL-ZERO μ_TABLE TEST — isolate DDA overhead from accumulation work
# =============================================================================

println("=" ^ 70)
println("9. ALL-ZERO μ_TABLE — DDA + overhead without useful accumulation")
println("=" ^ 70)

# With all-zero μ_table, the kernel still does full DDA traversal + μ_table reads + FMAs,
# but all accumulators stay zero and exp(0)=1. Comparing with real μ_table shows if
# non-zero values cause different performance (cache effects, divergence, etc.)

μ_zero = MtlArray(zeros(Float32, size(μ_sub)))
wη_ones = MtlArray(ones(Float32, K) ./ K)  # normalize so result is meaningful

# Warmup
fill!(ws.sinogram, zero(Float32))
Metal.@sync BS.siddon_fused_poly_project!(
    ws.sinogram, phantom_gpu.mask, ws.geom,
    μ_zero, wη_ones, Val(K);
    volume_extent=phantom_gpu.extent,
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    ws_bowtie_spectral=bt_sub_gpu
)

t_tile_zero = gpu_time(5) do
    fill!(ws.sinogram, zero(Float32))
    BS.siddon_fused_poly_project!(
        ws.sinogram, phantom_gpu.mask, ws.geom,
        μ_zero, wη_ones, Val(K);
        volume_extent=phantom_gpu.extent,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_bowtie_spectral=bt_sub_gpu
    )
end
println("  K=16, μ_table=zeros: $(round(t_tile_zero*1000, digits=2)) ms")
println("  K=16, μ_table=real:  $(round(t_tile_k16*1000, digits=2)) ms")
println("  Difference (non-zero μ work): $(round((t_tile_k16 - t_tile_zero)*1000, digits=2)) ms")
println("  → $(round((1 - t_tile_zero/t_tile_k16)*100, digits=1))% of tile time is from non-zero μ accumulation")
println()

# =============================================================================
# 10. ALL-TISSUE vs ORIGINAL PHANTOM — measure air fraction impact
# =============================================================================

println("=" ^ 70)
println("10. ALL-TISSUE PHANTOM — no air voxels")
println("=" ^ 70)

# Create an all-tissue phantom (fill mask with material 1 = first non-air material)
first_tissue_mat = unique_mats[findfirst(x -> x != 0x0000, unique_mats)]
mask_tissue = MtlArray(fill(first_tissue_mat, size(phantom_cpu.mask)))

# Warmup with tissue mask
fill!(ws.sinogram, zero(Float32))
Metal.@sync BS.siddon_fused_poly_project!(
    ws.sinogram, mask_tissue, ws.geom,
    μ_sub, wη_sub, Val(K);
    volume_extent=phantom_gpu.extent,
    ws_source_positions=ws.geom_source_positions,
    ws_detector_centers=ws.geom_detector_centers,
    ws_detector_u=ws.geom_detector_u,
    ws_detector_v=ws.geom_detector_v,
    ws_bowtie_spectral=bt_sub_gpu
)

t_tile_tissue = gpu_time(5) do
    fill!(ws.sinogram, zero(Float32))
    BS.siddon_fused_poly_project!(
        ws.sinogram, mask_tissue, ws.geom,
        μ_sub, wη_sub, Val(K);
        volume_extent=phantom_gpu.extent,
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        ws_bowtie_spectral=bt_sub_gpu
    )
end
println("  K=16, all tissue:   $(round(t_tile_tissue*1000, digits=2)) ms")
println("  K=16, real phantom: $(round(t_tile_k16*1000, digits=2)) ms")
println("  Difference: $(round((t_tile_tissue - t_tile_k16)*1000, digits=2)) ms")
println("  → All-tissue is $(round(t_tile_tissue/t_tile_k16, digits=2))× of real phantom")
println()
println("  Implication for air skipping:")
println("    If adding mat==0 check saves ~$(round(air_fraction*100, digits=0))% of FMA work,")
println("    per-tile savings ≈ $(round(air_fraction * (t_tile_k16 - t_tile_zero)*1000, digits=1)) ms")
println("    ($(round(air_fraction * (t_tile_k16 - t_tile_zero) / t_tile_k16 * 100, digits=1))% of tile time)")
println()

# =============================================================================
# 11. COMPREHENSIVE SPEEDUP PROJECTION
# =============================================================================

println("=" ^ 70)
println("11. COMPREHENSIVE SPEEDUP PROJECTION")
println("=" ^ 70)
println()

# Baseline
baseline_total = t_simulate * 1000  # ms
println("BASELINE: $(round(baseline_total, digits=0)) ms total simulate!()")
println()

# Optimization A: Tiled fusion K=16 (from SPEED-001)
tiled_fwd = n_tiles * t_tile_k16 * 1000  # ms
println("A. Tiled fusion K=16:")
println("   Forward proj: $(round(t_fwd_proj*1000, digits=0)) ms → $(round(tiled_fwd, digits=0)) ms ($(round(t_fwd_proj*1000/tiled_fwd, digits=2))×)")

# Optimization B: Skip redundant -log/exp round-trip
# In tiled approach, we output I_transmitted (no -log needed)
# Signal chain receives I_total directly (no exp needed)
savings_logexp = (t_log + t_exp) * 1000  # ms
println("B. Skip -log/exp round-trip: saves $(round(savings_logexp, digits=0)) ms")

# Optimization C: Fused signal chain (ideal path)
# Replace 4 separate kernels with 1 addition (sino += log_ref)
chain_baseline = t_chain_sum * 1000  # ms
if ws.bowtie_air_reference !== nothing
    chain_fused = t_fused_ideal * 1000
else
    chain_fused = t_step_log * 1000  # at minimum save the exp
end
println("C. Fused signal chain (ideal): $(round(chain_baseline, digits=0)) ms → $(round(chain_fused, digits=0)) ms")

# Optimization D: Air voxel skipping (estimated)
# Based on: real vs all-zero μ_table shows how much work is from accumulation
# Air fraction × that work = savings
air_skip_per_tile = air_fraction * (t_tile_k16 - t_tile_zero) * 1000
tiled_fwd_with_air_skip = (n_tiles * (t_tile_k16 * 1000 - air_skip_per_tile))
println("D. Air-skip (est. $(round(air_fraction*100, digits=0))% air): forward proj $(round(tiled_fwd, digits=0)) ms → $(round(tiled_fwd_with_air_skip, digits=0)) ms")

println()
println("PROJECTED TOTAL WITH ALL OPTIMIZATIONS:")

# Progressive stacking
# Note: savings_logexp is already incorporated in tiled + fused chain
# Tiled gives us I_total (no -log), fused chain takes I_total (no exp needed)
t_fwd_optimized = tiled_fwd  # Tiled K=16
t_chain_optimized = chain_fused  # Fused signal chain
t_physics_ms = t_physics * 1000
t_noise_est = 150.0  # from SPEED-000
t_copies_est = 5.0   # from SPEED-000

total_A = t_fwd_optimized + chain_baseline + t_physics_ms + t_noise_est + t_copies_est
total_AB = t_fwd_optimized + (chain_baseline - savings_logexp) + t_physics_ms + t_noise_est + t_copies_est
total_ABC = t_fwd_optimized + chain_fused + t_physics_ms + t_noise_est + t_copies_est
total_ABCD = tiled_fwd_with_air_skip + chain_fused + t_physics_ms + t_noise_est + t_copies_est

println()
println("  Baseline:                 $(round(baseline_total, digits=0)) ms  (1.0×)")
println("  +A (tiled K=16):          $(round(total_A, digits=0)) ms  ($(round(baseline_total/total_A, digits=2))×)")
println("  +AB (+ skip log/exp):     $(round(total_AB, digits=0)) ms  ($(round(baseline_total/total_AB, digits=2))×)")
println("  +ABC (+ fused chain):     $(round(total_ABC, digits=0)) ms  ($(round(baseline_total/total_ABC, digits=02))×)")
println("  +ABCD (+ air skip est):   $(round(total_ABCD, digits=0)) ms  ($(round(baseline_total/total_ABCD, digits=2))×)")
println()

target_10x = baseline_total / 10
remaining_gap = total_ABCD - target_10x
println("  10× target:               $(round(target_10x, digits=0)) ms")
println("  Best projected:           $(round(total_ABCD, digits=0)) ms")
if total_ABCD > target_10x
    println("  Gap to 10×:              $(round(remaining_gap, digits=0)) ms still needed")
    println("  → Need $(round(total_ABCD/target_10x, digits=1))× more speedup beyond A+B+C+D")
else
    println("  ✓ 10× ACHIEVED!")
end

println()
println("Done!")
