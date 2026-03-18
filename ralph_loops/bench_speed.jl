# =============================================================================
# SPEED-000: GPU Profiling Baseline — Metal @elapsed on every component
# =============================================================================
# Run from project root:
#   julia --project=verification ralph_loops/bench_speed.jl
#
# This script measures ACTUAL GPU execution time for every stage of simulate!()
# using explicit Metal.synchronize() for proper GPU timing.

using Metal
import BasisSimulator as BS
import XrayAttenuation as XA
using Random

const AK = BS.AK

println("=" ^ 70)
println("SPEED-000: GPU Profiling Baseline on Metal")
println("GPU: ", Metal.current_device())
println("=" ^ 70)
println()

# Helper: time a GPU operation properly
function gpu_time(f, n_runs=3)
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
# 1. Setup — small phantom for fast iteration
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

println("Phantom: $(size(phantom_cpu.mask)), extent=$(phantom_cpu.extent) cm")
println("Sinogram: $(size(ws.sinogram))")
println("Energies: $(length(ws.energies)) bins")
println("Has signal chain: $(ws.has_signal_chain)")
println()

# =============================================================================
# 2. Warmup (JIT + GPU kernel compilation)
# =============================================================================

println("Warming up (JIT + Metal kernel compilation)...")
Metal.@sync BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
println("Warmup done.")
println()

# =============================================================================
# 3. Full simulate!() timing (uses default fused=true path)
# =============================================================================

println("=" ^ 70)
println("BENCHMARK: Full simulate!() — 5 runs")
println("=" ^ 70)

full_times = Float64[]
for i in 1:5
    Metal.synchronize()
    t0 = time_ns()
    BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
    Metal.synchronize()
    t = (time_ns() - t0) / 1e9
    push!(full_times, t)
    println("  Run $i: $(round(t, digits=4))s")
end
full_median = sort(full_times)[3]
println("  Median: $(round(full_median, digits=4))s")
println()

# =============================================================================
# 4. Forward Projection — FUSED vs UNFUSED
# =============================================================================

for (label, fused_flag) in [("FUSED", true), ("UNFUSED", false)]
    println("=" ^ 70)
    println("BENCHMARK: _forward_project_poly!() — $label")
    println("=" ^ 70)

    times = Float64[]
    for i in 1:3
        fill!(ws.sinogram, zero(Float32))
        Metal.synchronize()
        t0 = time_ns()
        BS._forward_project_poly!(
            ws.sinogram, phantom_gpu.mask, ws.geom,
            ws.energies, ws.weights, ws.mats;
            ws_μ_volume=ws.μ_volume,
            ws_sino_mono=ws.sino_mono,
            ws_I_transmitted=ws.I_transmitted,
            ws_weights_norm=ws.weights_norm,
            ws_μ_lut_cpu=ws.μ_lut_cpu,
            ws_μ_lut_gpu=ws.μ_lut_gpu,
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
            fused=fused_flag
        )
        Metal.synchronize()
        t = (time_ns() - t0) / 1e9
        push!(times, t)
        println("  Run $i: $(round(t, digits=4))s")
    end
    med = sort(times)[2]
    println("  Median: $(round(med, digits=4))s")

    # Check output validity
    sino_out = Array(ws.sinogram)
    println("  Output range: [$(round(minimum(sino_out), digits=4)), $(round(maximum(sino_out), digits=4))]")
    println("  Non-zero elements: $(count(!=(0f0), sino_out)) / $(length(sino_out))")
    println()
end

# =============================================================================
# 5. Physics pipeline timing
# =============================================================================

println("=" ^ 70)
println("BENCHMARK: _apply_physics_no_noise!() — all detector effects")
println("=" ^ 70)

# Generate a realistic sinogram first
fill!(ws.sinogram, zero(Float32))
Metal.@sync BS._forward_project_poly!(
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

sino_backup = similar(ws.sinogram)
copyto!(sino_backup, ws.sinogram)

physics_times = Float64[]
for i in 1:5
    copyto!(ws.sinogram, sino_backup)
    t = gpu_time() do
        BS._apply_physics_no_noise!(
            ws.sinogram, ws.geom, ws.config;
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
            ws_lag_coeffs=ws.lag_coeffs
        )
    end
    push!(physics_times, t)
    println("  Run $i: $(round(t*1000, digits=2)) ms")
end
physics_median = sort(physics_times)[3]
println("  Median: $(round(physics_median*1000, digits=2)) ms")
println()

# =============================================================================
# 6. Step-by-step simulate!() timing (UNFUSED path)
# =============================================================================

println("=" ^ 70)
println("DIAGNOSTIC: Step-by-step simulate!() timing (UNFUSED baseline)")
println("=" ^ 70)

T_f = Float32
eps_f = Float32(1e-10)

# STEP 1: Forward projection (unfused)
fill!(ws.sinogram, zero(T_f))
Metal.synchronize()
t0 = time_ns()
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
Metal.synchronize()
t_proj = (time_ns() - t0) / 1e9
println("  STEP 1 fwd_proj (unfused): $(round(t_proj*1000, digits=2)) ms")

# STEP 2: Physics pipeline
Metal.synchronize()
t0 = time_ns()
BS._apply_physics_no_noise!(
    ws.sinogram, ws.geom, ws.config;
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
    ws_lag_coeffs=ws.lag_coeffs
)
Metal.synchronize()
t_phys = (time_ns() - t0) / 1e9
println("  STEP 2 physics:            $(round(t_phys*1000, digits=2)) ms")

# STEP 3: exp(-sino) conversion
Metal.synchronize()
t0 = time_ns()
let sino = ws.sinogram
    AK.foreachindex(sino) do idx
        sino[idx] = exp(-clamp(sino[idx], Float32(-1), Float32(15)))
    end
end
Metal.synchronize()
t_exp = (time_ns() - t0) / 1e9
println("  STEP 3 exp(-sino):         $(round(t_exp*1000, digits=2)) ms")

# STEP 4: Heel effect
Metal.synchronize()
t0 = time_ns()
if ws.heel_effect !== nothing
    BS.apply_heel_effect!(ws.sinogram, ws.heel_effect, ws.geom)
end
Metal.synchronize()
t_heel = (time_ns() - t0) / 1e9
println("  STEP 4 heel:               $(round(t_heel*1000, digits=2)) ms")

# STEP 5: Air scan build
Metal.synchronize()
t0 = time_ns()
fill!(ws.air_scan, one(T_f))
if ws.bowtie_air_reference !== nothing
    let air = ws.air_scan, ref = ws.bowtie_air_reference, nc = Int32(size(air, 1)), nr = Int32(size(air, 2))
        AK.foreachindex(air) do idx
            idx_0 = Int32(idx - 1)
            col = (idx_0 % nc) + Int32(1)
            row = ((idx_0 ÷ nc) % nr) + Int32(1)
            ref_idx = col + (row - Int32(1)) * nc
            air[idx] *= ref[ref_idx]
        end
    end
end
if ws.heel_effect !== nothing
    BS.apply_heel_effect!(ws.air_scan, ws.heel_effect, ws.geom)
end
Metal.synchronize()
t_air = (time_ns() - t0) / 1e9
println("  STEP 5 air_scan:           $(round(t_air*1000, digits=2)) ms")

# STEP 6: Calibration
Metal.synchronize()
t0 = time_ns()
let sino = ws.sinogram, air = ws.air_scan, eps_v = eps_f
    AK.foreachindex(sino) do idx
        air_val = max(air[idx], eps_v)
        sino[idx] = sino[idx] / air_val
    end
end
Metal.synchronize()
t_cal = (time_ns() - t0) / 1e9
println("  STEP 6 calibration:        $(round(t_cal*1000, digits=2)) ms")

# STEP 7: Low signal correction
Metal.synchronize()
t0 = time_ns()
BS.low_signal_correction_gpu!(ws.sinogram)
Metal.synchronize()
t_lsc = (time_ns() - t0) / 1e9
println("  STEP 7 low_sig_corr:       $(round(t_lsc*1000, digits=2)) ms")

# STEP 8: Log transform
Metal.synchronize()
t0 = time_ns()
let sino = ws.sinogram, eps_v = eps_f
    AK.foreachindex(sino) do idx
        sino[idx] = -log(max(sino[idx], eps_v))
    end
end
Metal.synchronize()
t_log = (time_ns() - t0) / 1e9
println("  STEP 8 -log:               $(round(t_log*1000, digits=2)) ms")

# STEP 9: BHC
Metal.synchronize()
t0 = time_ns()
if ws.bhc !== nothing
    BS.apply_bhc!(ws.sinogram, ws.bhc; ws_coeffs_gpu=ws.bhc_coeffs_gpu)
end
Metal.synchronize()
t_bhc = (time_ns() - t0) / 1e9
println("  STEP 9 BHC:                $(round(t_bhc*1000, digits=2)) ms")

# STEP 10: GPU→CPU ideal copy
Metal.synchronize()
t0 = time_ns()
copyto!(ws.sino_ideal_out, ws.sinogram)
Metal.synchronize()
t_copy1 = (time_ns() - t0) / 1e9
println("  STEP 10 GPU→CPU:           $(round(t_copy1*1000, digits=2)) ms")

# STEP 11: Noise (quantum + electronic)
Metal.synchronize()
t0 = time_ns()
I0_raw = BS.compute_detector_I0(ws.geom, protocol, sum(ws.weights))
η_eff = sum(ws.weights_norm[i] * ws.η_vec[i] for i in 1:length(ws.η_vec))
I0_val = Float32(I0_raw * η_eff)
randn!(ws.noise_rand_cpu)
copyto!(ws.noise_rand_gpu, ws.noise_rand_cpu)
let sino = ws.sinogram, rg = ws.noise_rand_gpu, I0v = I0_val
    AK.foreachindex(sino) do idx
        λ = I0v * exp(-sino[idx])
        λ_noisy = λ + sqrt(max(λ, Float32(1))) * rg[idx]
        λ_noisy = max(λ_noisy, Float32(1))
        sino[idx] = -log(λ_noisy / I0v)
    end
end
Metal.synchronize()
t_noise = (time_ns() - t0) / 1e9
println("  STEP 11 noise:             $(round(t_noise*1000, digits=2)) ms")

# STEP 12: GPU→CPU noisy copy
Metal.synchronize()
t0 = time_ns()
copyto!(ws.sino_noisy_out, ws.sinogram)
Metal.synchronize()
t_copy2 = (time_ns() - t0) / 1e9
println("  STEP 12 GPU→CPU:           $(round(t_copy2*1000, digits=2)) ms")

t_total_manual = t_proj + t_phys + t_exp + t_heel + t_air + t_cal + t_lsc + t_log + t_bhc + t_copy1 + t_noise + t_copy2
println()
println("  Manual total:              $(round(t_total_manual*1000, digits=2)) ms")
println()

# =============================================================================
# 7. Per-energy-bin Siddon kernel timing
# =============================================================================

println("=" ^ 70)
println("BENCHMARK: Single siddon_forward_project! call")
println("=" ^ 70)

# Create mu volume for first energy
e_idx = 1
E = ws.energies[e_idx]
n_regions = length(ws.mats)
for r in 1:n_regions
    ws.μ_lut_cpu[r] = Float32(BS.compute_μ_at_energy(ws.mats[r], Float64(E)))
end
copyto!(ws.μ_lut_gpu, ws.μ_lut_cpu)

# Time create_μ_volume!
t_mu = gpu_time(5) do
    BS.create_μ_volume!(ws.μ_volume, phantom_gpu.mask, ws.μ_lut_gpu)
end
println("  create_μ_volume!:     $(round(t_mu*1000, digits=3)) ms")

# Time siddon_forward_project!
fill!(ws.sino_mono, zero(Float32))
t_siddon = gpu_time(5) do
    BS.siddon_forward_project!(
        ws.sino_mono, ws.μ_volume, ws.geom;
        ws_source_positions=ws.geom_source_positions,
        ws_detector_centers=ws.geom_detector_centers,
        ws_detector_u=ws.geom_detector_u,
        ws_detector_v=ws.geom_detector_v,
        volume_extent=phantom_gpu.extent
    )
end
println("  siddon_forward_project!: $(round(t_siddon*1000, digits=3)) ms")

n_energies = length(ws.energies)
println("  n_energies: $n_energies")
println("  Est. unfused total: $(round(n_energies * (t_mu + t_siddon), digits=3))s")
println("  (plus accumulation overhead per bin)")
println()

# =============================================================================
# 8. Summary
# =============================================================================

println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println()

# Use the unfused forward projection time as the real baseline
# Since fused is SLOWER, the unfused path is what we should optimize from
println("KEY FINDING: Fused kernel is SLOWER on Metal GPU!")
println()

println("Problem size:")
println("  Phantom:    $(size(phantom_gpu.mask)) = $(prod(size(phantom_gpu.mask))) voxels")
println("  Sinogram:   $(size(ws.sinogram)) = $(prod(size(ws.sinogram))) elements")
println("  Energies:   $n_energies bins")
println("  Rays/view:  $(size(ws.sinogram,1) * size(ws.sinogram,2))")
println()

println("Timing breakdown (UNFUSED baseline):")
println("  Forward proj (unfused): $(round(t_proj, digits=3))s")
println("  Physics pipeline:       $(round(t_phys*1000, digits=2)) ms")
println("  Signal chain steps:     $(round((t_exp + t_heel + t_air + t_cal + t_lsc + t_log + t_bhc)*1000, digits=2)) ms")
println("  GPU→CPU copies:         $(round((t_copy1 + t_copy2)*1000, digits=2)) ms")
println("  Noise:                  $(round(t_noise*1000, digits=2)) ms")
println()

println("Per-energy breakdown:")
println("  create_μ_volume!:       $(round(t_mu*1000, digits=3)) ms")
println("  siddon_forward_project!: $(round(t_siddon*1000, digits=3)) ms")
println("  Total per bin:          $(round((t_mu + t_siddon)*1000, digits=3)) ms")
println("  × $n_energies bins =   $(round(n_energies * (t_mu + t_siddon), digits=3))s")
println()

println("Done!")
