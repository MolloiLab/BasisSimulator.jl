#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# Difference Image Diagnostic Script
#
# Runs TWO simulations (bare vs polychromatic) and creates difference images
# to visualize exactly what the polychromatic pathway changes.
#
# Also tests flat_filter and det_eff individually to confirm all 4 spectral
# triggers produce the same artifact pattern.
#
# Usage: julia --project=verification ralph_loop_diagnose/diag_diff.jl
# ═══════════════════════════════════════════════════════════════════════════════

using Printf

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DETECTOR_COL_SIZE  = 1.0     # mm
DETECTOR_ROW_SIZE  = 0.625   # mm
DETECTOR_COLS      = nothing  # auto-compute
DETECTOR_ROWS      = 64
VIEWS              = 1600
KVP                = 120.0
MA                 = 700.0
RECON_XY           = 512
RECON_SLICES       = 64
RECON_FOV_CM       = 35.0
DOWNSAMPLE_FACTOR  = 2
OUTPUT_DIR         = joinpath(@__DIR__, "outputs")
HU_WINDOW          = (-300, 400)

# Which tests to run (set false to skip)
RUN_FLAT_FILTER    = true
RUN_DET_EFF        = true
RUN_LAG            = true
RUN_FILL_FACTOR    = true
RUN_HEEL           = true
# ─── END CONFIG ──────────────────────────────────────────────────────────────

mkpath(OUTPUT_DIR)

println("=" ^ 70)
println("DIFFERENCE IMAGE DIAGNOSTIC")
println("=" ^ 70)

# ─── SETUP ────────────────────────────────────────────────────────────────────
println("Loading packages...")
t0 = time()

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "verification"))

using Revise
import BasisSimulator as BS
import CairoMakie as CM
import Metal
import XLSX
import XrayAttenuation as XA
using Unitful: @u_str, ustrip
using Statistics: mean, std

println("  Packages loaded in $(@sprintf("%.1f", time() - t0))s")

# ─── LOAD XCAT PHANTOM ───────────────────────────────────────────────────────
println("Loading XCAT phantom...")
t1 = time()

ROOT_DIR = joinpath(@__DIR__, "..", "verification")
PHANTOM_PATH = joinpath(ROOT_DIR, "data/xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin")
MATERIAL_XLSX_PATH = joinpath(ROOT_DIR, "data/xcat/Material_Spreadsheets/vmale_50_materials_heart_high_contrast.xlsx")

function load_phantom_bin(filepath; cols=1600, rows=1400, slices=500, dtype=UInt8)
    expected_size = cols * rows * slices * sizeof(dtype)
    actual_size = filesize(filepath)
    @assert actual_size == expected_size "File size mismatch"
    data = Vector{dtype}(undef, cols * rows * slices)
    open(filepath, "r") do io
        read!(io, data)
    end
    phantom = reshape(data, (cols, rows, slices))
    return reverse(phantom, dims=(2, 3))
end

function downsample_phantom(phantom::AbstractArray{T, 3}, factor::Int) where T
    factor == 1 && return phantom
    new_size = size(phantom) .÷ factor
    result = similar(phantom, new_size)
    for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
        oi = (i - 1) * factor + factor ÷ 2 + 1
        oj = (j - 1) * factor + factor ÷ 2 + 1
        ok = (k - 1) * factor + factor ÷ 2 + 1
        result[i, j, k] = phantom[oi, oj, ok]
    end
    return result
end

phantom_raw = load_phantom_bin(PHANTOM_PATH)
phantom_labeled = downsample_phantom(phantom_raw, DOWNSAMPLE_FACTOR)
phantom_raw = nothing
GC.gc(true)

println("  Phantom size: $(size(phantom_labeled))")

# ─── LOAD MATERIALS ──────────────────────────────────────────────────────────
function compute_ZA_ratio(composition::Dict{Int, Float64})
    atomic_masses = Dict(
        1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
        15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904
    )
    Z_sum = 0.0; A_sum = 0.0
    for (Z, mass_frac) in composition
        A = get(atomic_masses, Z, Float64(Z)*2)
        Z_sum += mass_frac * Z / A
        A_sum += mass_frac
    end
    return Z_sum / A_sum
end

function compute_mean_excitation_energy(composition::Dict{Int, Float64})
    I_values = Dict(
        1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
        15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0
    )
    atomic_masses = Dict(
        1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
        15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904
    )
    log_I_sum = 0.0; Z_A_sum = 0.0
    for (Z, mass_frac) in composition
        A = get(atomic_masses, Z, Float64(Z)*2)
        I = get(I_values, Z, 10.0 * Z)
        Z_A = mass_frac * Z / A
        log_I_sum += Z_A * log(I)
        Z_A_sum += Z_A
    end
    return exp(log_I_sum / Z_A_sum) * u"eV"
end

function load_materials_from_xlsx(xlsx_path)
    xf = XLSX.readxlsx(xlsx_path)
    sheet = xf["Sheet1"]
    data = sheet["A2:P34"]
    materials = Dict{Int, XA.Material}()
    for i in 1:size(data, 1)
        try
            name = String(data[i, 1])
            organ_id = Int(data[i, 16])
            density = Float64(data[i, 15]) * u"g/cm^3"
            comp = Dict{Int, Float64}()
            comp[1] = Float64(data[i, 2])
            comp[6] = Float64(data[i, 3])
            comp[7] = Float64(data[i, 4])
            comp[8] = Float64(data[i, 5])
            comp[11] = Float64(data[i, 6])
            comp[12] = Float64(data[i, 7])
            comp[15] = Float64(data[i, 8])
            comp[16] = Float64(data[i, 9])
            comp[17] = Float64(data[i, 10])
            comp[19] = Float64(data[i, 11])
            comp[20] = Float64(data[i, 12])
            comp[26] = Float64(data[i, 13])
            comp[53] = Float64(data[i, 14])
            filter!(p -> p.second > 0, comp)
            ZA = compute_ZA_ratio(comp)
            I = compute_mean_excitation_energy(comp)
            mat = XA.Material(name, ZA, I, density, comp)
            materials[organ_id] = mat
        catch e
            @warn "Failed to parse row $i" exception=(e, catch_backtrace())
        end
    end
    return materials
end

materials_dict = load_materials_from_xlsx(MATERIAL_XLSX_PATH)
println("  Materials loaded: $(length(materials_dict))")
println("  Loaded in $(@sprintf("%.1f", time() - t1))s")

# ─── VOXEL SIZE ──────────────────────────────────────────────────────────────
base_voxel_cm = (0.03, 0.03, 0.1)
voxel_size_cm = base_voxel_cm .* DOWNSAMPLE_FACTOR

phantom_extent_mm = max(
    size(phantom_labeled, 1) * voxel_size_cm[1],
    size(phantom_labeled, 2) * voxel_size_cm[2]
) * 10.0

# ─── GPU PHANTOM ─────────────────────────────────────────────────────────────
println("Creating GPU phantom...")
phantom_mask_gpu = Metal.MtlArray(phantom_labeled)
phantom_gpu = BS.Phantom(phantom_mask_gpu, materials_dict, voxel_size_cm)
phantom_labeled = nothing
GC.gc(true)

# ─── SCANNER ─────────────────────────────────────────────────────────────────
SID = 625.6
SDD = 1100.0
mag = SDD / SID

det_cols = if DETECTOR_COLS === nothing
    ceil(Int, phantom_extent_mm * mag / DETECTOR_COL_SIZE)
else
    DETECTOR_COLS
end

println("Detector: $(det_cols) cols × $(DETECTOR_ROWS) rows, $(DETECTOR_COL_SIZE)mm × $(DETECTOR_ROW_SIZE)mm")

scanner = BS.Scanner(
    source_to_isocenter = SID,
    source_to_detector = SDD,
    detector_rows = DETECTOR_ROWS,
    detector_cols = det_cols,
    detector_row_size = DETECTOR_ROW_SIZE,
    detector_col_size = DETECTOR_COL_SIZE,
    detector_shape = BS.CURVED_DETECTOR,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 7.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    detector_material = :gos,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    detection_gain = 1.0,
)

protocol = BS.CTProtocol(kVp = KVP, mA = MA, views = VIEWS, rotation_time = 0.5)

recon_size = (RECON_XY, RECON_XY, RECON_SLICES)
recon_opts = BS.ReconOptions(algorithm = :fdk, matrix_size = recon_size, fov_cm = RECON_FOV_CM)

# ─── HELPER: Run simulation + reconstruction ─────────────────────────────────
function run_sim_recon(name, sim_opts)
    println("\n--- $name ---")
    t = time()
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
    BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
    sino = ws.sino_noisy_out
    geom = ws.geom
    println("  Sim time: $(@sprintf("%.1f", time() - t))s")

    ws_fdk = BS.create_fdk_recon_workspace(sino, geom, recon_size; filter=BS.HannFilter())
    recon = Array(BS.reconstruct!(ws_fdk, sino, geom, recon_size))
    sino_cpu = Array(sino)

    # Extract μ_water-like center value for reference
    nx, ny, nz = size(recon)
    cx, cy = nx ÷ 2, ny ÷ 2
    r = nx ÷ 10
    vals = Float64[]
    for j in (cy - r):(cy + r), i in (cx - r):(cx + r)
        if (i - cx)^2 + (j - cy)^2 <= r^2
            push!(vals, recon[i, j, nz ÷ 2])
        end
    end
    center_μ = mean(vals)

    ws = nothing; ws_fdk = nothing
    GC.gc(true)

    println("  Center μ: $(round(center_μ, sigdigits=4))")
    println("  Range: [$(round(minimum(recon), sigdigits=4)), $(round(maximum(recon), sigdigits=4))]")

    return recon, sino_cpu, center_μ
end

function make_sim_opts(;
    fill_factor=false, flat_filter=false, bowtie=false,
    det_eff=false, crosstalk=false, optical=false,
    focal_spot=false, lag=false, heel=false, bhc=false
)
    return BS.SimOptions(
        fidelity = :high,
        use_fill_factor = fill_factor,
        use_flat_filter = flat_filter,
        use_bowtie_filter = bowtie,
        use_detector_efficiency = det_eff,
        use_scatter = false, use_scatter_correction = false,
        use_crosstalk = crosstalk,
        use_optical_crosstalk = optical,
        use_focal_spot = focal_spot,
        use_noise = false,
        use_lag = lag,
        use_heel_effect = heel,
        use_das = false, use_bhc = bhc,
        use_pcct_corrections = false,
        pcct_noise_reduction = 0.0,
        n_energy_bins = 100,
        seed = 42
    )
end

slice_idx = RECON_SLICES ÷ 2

# ─── BASELINE: Bare Siddon ────────────────────────────────────────────────────
recon_bare, sino_bare, μ_bare = run_sim_recon("BARE SIDDON", make_sim_opts())

# ─── TEST 1: flat_filter only (triggers polychromatic) ────────────────────────
if RUN_FLAT_FILTER
    recon_flat, sino_flat, μ_flat = run_sim_recon("FLAT FILTER ONLY", make_sim_opts(flat_filter=true))

    # Difference image: flat_filter - bare (in raw μ space)
    diff_flat = recon_flat .- recon_bare

    f = CM.Figure(size=(1800, 500))
    ax1 = CM.Axis(f[1,1], title="Bare Siddon", aspect=CM.DataAspect())
    CM.heatmap!(ax1, recon_bare[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.5))
    ax2 = CM.Axis(f[1,2], title="Flat Filter Only", aspect=CM.DataAspect())
    CM.heatmap!(ax2, recon_flat[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.5))
    ax3 = CM.Axis(f[1,3], title="Difference (flat - bare)", aspect=CM.DataAspect())
    diff_range = max(abs(minimum(diff_flat[:,:,slice_idx])), abs(maximum(diff_flat[:,:,slice_idx]))) * 0.5
    CM.heatmap!(ax3, diff_flat[:,:,slice_idx], colormap=:RdBu, colorrange=(-diff_range, diff_range))
    CM.Colorbar(f[1,4], colormap=:RdBu, colorrange=(-diff_range, diff_range), label="Δμ")
    CM.save(joinpath(OUTPUT_DIR, "diff_flat_filter.png"), f)
    println("  Saved diff_flat_filter.png")

    # Also save standalone recon
    f2 = CM.Figure(size=(600, 600))
    ax = CM.Axis(f2[1,1], title="plus_flat_filter (slice $slice_idx)", aspect=CM.DataAspect())
    recon_flat_hu = 1000f0 .* (recon_flat .- μ_bare) ./ μ_bare
    CM.heatmap!(ax, recon_flat_hu[:,:,slice_idx], colormap=:grays, colorrange=HU_WINDOW)
    CM.save(joinpath(OUTPUT_DIR, "diag_plus_flat_filter.png"), f2)
    println("  Saved diag_plus_flat_filter.png")

    recon_flat = nothing; diff_flat = nothing; sino_flat = nothing
    GC.gc(true)
end

# ─── TEST 2: det_eff only (triggers polychromatic) ────────────────────────────
if RUN_DET_EFF
    recon_eff, sino_eff, μ_eff = run_sim_recon("DET EFFICIENCY ONLY", make_sim_opts(det_eff=true))

    diff_eff = recon_eff .- recon_bare

    f = CM.Figure(size=(1800, 500))
    ax1 = CM.Axis(f[1,1], title="Bare Siddon", aspect=CM.DataAspect())
    CM.heatmap!(ax1, recon_bare[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.5))
    ax2 = CM.Axis(f[1,2], title="Det Efficiency Only", aspect=CM.DataAspect())
    CM.heatmap!(ax2, recon_eff[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.5))
    ax3 = CM.Axis(f[1,3], title="Difference (det_eff - bare)", aspect=CM.DataAspect())
    diff_range = max(abs(minimum(diff_eff[:,:,slice_idx])), abs(maximum(diff_eff[:,:,slice_idx]))) * 0.5
    CM.heatmap!(ax3, diff_eff[:,:,slice_idx], colormap=:RdBu, colorrange=(-diff_range, diff_range))
    CM.Colorbar(f[1,4], colormap=:RdBu, colorrange=(-diff_range, diff_range), label="Δμ")
    CM.save(joinpath(OUTPUT_DIR, "diff_det_eff.png"), f)
    println("  Saved diff_det_eff.png")

    recon_eff = nothing; diff_eff = nothing; sino_eff = nothing
    GC.gc(true)
end

# ─── TEST 3: lag only (does NOT trigger polychromatic) ────────────────────────
if RUN_LAG
    recon_lag, sino_lag, μ_lag = run_sim_recon("LAG ONLY", make_sim_opts(lag=true))

    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1,1], title="plus_lag (slice $slice_idx)", aspect=CM.DataAspect())
    recon_lag_hu = 1000f0 .* (recon_lag .- μ_bare) ./ μ_bare
    CM.heatmap!(ax, recon_lag_hu[:,:,slice_idx], colormap=:grays, colorrange=HU_WINDOW)
    CM.save(joinpath(OUTPUT_DIR, "diag_plus_lag.png"), f)
    println("  Saved diag_plus_lag.png")

    recon_lag = nothing; sino_lag = nothing
    GC.gc(true)
end

# ─── TEST 4: fill_factor only (does NOT trigger polychromatic) ────────────────
if RUN_FILL_FACTOR
    recon_ff, sino_ff, μ_ff = run_sim_recon("FILL FACTOR ONLY", make_sim_opts(fill_factor=true))

    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1,1], title="plus_fill_factor (slice $slice_idx)", aspect=CM.DataAspect())
    recon_ff_hu = 1000f0 .* (recon_ff .- μ_bare) ./ μ_bare
    CM.heatmap!(ax, recon_ff_hu[:,:,slice_idx], colormap=:grays, colorrange=HU_WINDOW)
    CM.save(joinpath(OUTPUT_DIR, "diag_plus_fill_factor.png"), f)
    println("  Saved diag_plus_fill_factor.png")

    recon_ff = nothing; sino_ff = nothing
    GC.gc(true)
end

# ─── TEST 5: heel_effect only (does NOT trigger polychromatic) ────────────────
if RUN_HEEL
    recon_heel, sino_heel, μ_heel = run_sim_recon("HEEL EFFECT ONLY", make_sim_opts(heel=true))

    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1,1], title="plus_heel (slice $slice_idx)", aspect=CM.DataAspect())
    recon_heel_hu = 1000f0 .* (recon_heel .- μ_bare) ./ μ_bare
    CM.heatmap!(ax, recon_heel_hu[:,:,slice_idx], colormap=:grays, colorrange=HU_WINDOW)
    CM.save(joinpath(OUTPUT_DIR, "diag_plus_heel.png"), f)
    println("  Saved diag_plus_heel.png")

    recon_heel = nothing; sino_heel = nothing
    GC.gc(true)
end

# ─── FINAL: Difference image for bowtie (reuse existing data if possible) ─────
# Re-run bowtie to get raw μ values for difference imaging
println("\n--- BOWTIE ONLY (for difference image) ---")
recon_bowtie, sino_bowtie, μ_bowtie = run_sim_recon("BOWTIE ONLY", make_sim_opts(bowtie=true))

diff_bowtie = recon_bowtie .- recon_bare

f = CM.Figure(size=(1800, 500))
ax1 = CM.Axis(f[1,1], title="Bare Siddon", aspect=CM.DataAspect())
CM.heatmap!(ax1, recon_bare[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.5))
ax2 = CM.Axis(f[1,2], title="Bowtie Only", aspect=CM.DataAspect())
CM.heatmap!(ax2, recon_bowtie[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.5))
ax3 = CM.Axis(f[1,3], title="Difference (bowtie - bare)", aspect=CM.DataAspect())
diff_range = max(abs(minimum(diff_bowtie[:,:,slice_idx])), abs(maximum(diff_bowtie[:,:,slice_idx]))) * 0.5
CM.heatmap!(ax3, diff_bowtie[:,:,slice_idx], colormap=:RdBu, colorrange=(-diff_range, diff_range))
CM.Colorbar(f[1,4], colormap=:RdBu, colorrange=(-diff_range, diff_range), label="Δμ")
CM.save(joinpath(OUTPUT_DIR, "diff_bowtie.png"), f)
println("  Saved diff_bowtie.png")

# ─── SINOGRAM DIFFERENCES ────────────────────────────────────────────────────
# Compare sinograms: bare vs bowtie (both should show the spectral effect)
sino_diff_bowtie = sino_bowtie .- sino_bare
mid_row = size(sino_bare, 2) ÷ 2

f3 = CM.Figure(size=(1800, 500))
ax1 = CM.Axis(f3[1,1], title="Sinogram: Bare", xlabel="Col", ylabel="View")
CM.heatmap!(ax1, sino_bare[:, mid_row, :]')
ax2 = CM.Axis(f3[1,2], title="Sinogram: Bowtie", xlabel="Col", ylabel="View")
CM.heatmap!(ax2, sino_bowtie[:, mid_row, :]')
ax3 = CM.Axis(f3[1,3], title="Sinogram Diff (bowtie - bare)", xlabel="Col", ylabel="View")
sdiff_range = max(abs(minimum(sino_diff_bowtie[:, mid_row, :])), abs(maximum(sino_diff_bowtie[:, mid_row, :]))) * 0.3
CM.heatmap!(ax3, sino_diff_bowtie[:, mid_row, :]', colormap=:RdBu, colorrange=(-sdiff_range, sdiff_range))
CM.save(joinpath(OUTPUT_DIR, "sino_diff_bowtie.png"), f3)
println("  Saved sino_diff_bowtie.png")

# ─── PROFILE COMPARISON ──────────────────────────────────────────────────────
# Line profiles through the center of a rib
profile_row = RECON_XY ÷ 2

f4 = CM.Figure(size=(800, 400))
ax = CM.Axis(f4[1,1], title="Profile comparison (y=$profile_row, slice $slice_idx)",
    xlabel="x pixel", ylabel="μ (cm⁻¹)")
CM.lines!(ax, recon_bare[:, profile_row, slice_idx], label="Bare Siddon", color=:blue)
CM.lines!(ax, recon_bowtie[:, profile_row, slice_idx], label="Bowtie only", color=:red)
CM.axislegend(ax)
CM.save(joinpath(OUTPUT_DIR, "profile_bare_vs_bowtie.png"), f4)
println("  Saved profile_bare_vs_bowtie.png")

# ─── NARROW WINDOW COMPARISON ────────────────────────────────────────────────
# Display with very narrow window to see subtle artifacts
narrow_window = (-50, 100)  # Tight around soft tissue

f5 = CM.Figure(size=(1200, 500))
recon_bare_hu = 1000f0 .* (recon_bare .- μ_bare) ./ μ_bare
recon_bowtie_hu = 1000f0 .* (recon_bowtie .- μ_bare) ./ μ_bare  # use bare μ for fair comparison

ax1 = CM.Axis(f5[1,1], title="Bare (narrow window)", aspect=CM.DataAspect())
CM.heatmap!(ax1, recon_bare_hu[:,:,slice_idx], colormap=:grays, colorrange=narrow_window)
ax2 = CM.Axis(f5[1,2], title="Bowtie (narrow window)", aspect=CM.DataAspect())
CM.heatmap!(ax2, recon_bowtie_hu[:,:,slice_idx], colormap=:grays, colorrange=narrow_window)
CM.save(joinpath(OUTPUT_DIR, "narrow_window_comparison.png"), f5)
println("  Saved narrow_window_comparison.png")

# ─── CLEANUP ─────────────────────────────────────────────────────────────────
recon_bare = nothing; recon_bowtie = nothing
sino_bare = nothing; sino_bowtie = nothing
GC.gc(true)

println("\n" * "=" ^ 70)
println("DONE — All difference images generated")
println("Total time: $(@sprintf("%.1f", time() - t0))s")
println("=" ^ 70)
