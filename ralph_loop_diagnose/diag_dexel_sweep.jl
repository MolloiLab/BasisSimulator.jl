#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# Dexel Sweep Diagnostic Script
#
# Tests the user's observation: "more dexels = worse artifacts"
# Runs polychromatic simulation (flat_filter=true) at 4 detector resolutions.
#
# Usage: julia --project=verification ralph_loop_diagnose/diag_dexel_sweep.jl
# ═══════════════════════════════════════════════════════════════════════════════

using Printf

# ─── DEXEL SWEEP CONFIG ──────────────────────────────────────────────────────
# Each entry: (col_size_mm, label)
DEXEL_CONFIGS = [
    (2.0,  "coarse_2mm"),
    (1.0,  "default_1mm"),
    (0.5,  "fine_05mm"),
    (0.25, "ultrafine_025mm"),
]

# Also test: square pixels (col_size = row_size = 1.0mm)
RUN_SQUARE_PIXEL_TEST = true

# Common settings
DETECTOR_ROW_SIZE  = 0.625   # mm (kept constant)
DETECTOR_ROWS      = 64
VIEWS              = 1600
KVP                = 120.0
MA                 = 700.0
RECON_XY           = 512
RECON_SLICES       = 64
RECON_FOV_CM       = 35.0
DOWNSAMPLE_FACTOR  = 2
OUTPUT_DIR         = joinpath(@__DIR__, "outputs")

# ─── END CONFIG ──────────────────────────────────────────────────────────────

mkpath(OUTPUT_DIR)

println("=" ^ 70)
println("DEXEL SWEEP DIAGNOSTIC — Polychromatic (flat_filter)")
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

SID = 625.6
SDD = 1100.0
mag = SDD / SID

recon_size = (RECON_XY, RECON_XY, RECON_SLICES)
recon_opts = BS.ReconOptions(algorithm = :fdk, matrix_size = recon_size, fov_cm = RECON_FOV_CM)
protocol = BS.CTProtocol(kVp = KVP, mA = MA, views = VIEWS, rotation_time = 0.5)

slice_idx = RECON_SLICES ÷ 2

# Collect results for comparison
results = Dict{String, Any}()

for (col_size, label) in DEXEL_CONFIGS
    println("\n" * "=" ^ 70)
    println("DEXEL TEST: $label (col_size=$(col_size)mm)")
    println("=" ^ 70)

    det_cols = ceil(Int, phantom_extent_mm * mag / col_size)
    println("  Detector: $(det_cols) cols × $(DETECTOR_ROWS) rows")

    scanner = BS.Scanner(
        source_to_isocenter = SID,
        source_to_detector = SDD,
        detector_rows = DETECTOR_ROWS,
        detector_cols = det_cols,
        detector_row_size = DETECTOR_ROW_SIZE,
        detector_col_size = col_size,
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

    # Use flat_filter=true to trigger polychromatic pathway
    sim_opts = BS.SimOptions(
        fidelity = :high,
        use_fill_factor = false,
        use_flat_filter = true,   # triggers polychromatic
        use_bowtie_filter = false,
        use_detector_efficiency = false,
        use_scatter = false, use_scatter_correction = false,
        use_crosstalk = false, use_optical_crosstalk = false,
        use_focal_spot = false, use_noise = false,
        use_lag = false, use_heel_effect = false,
        use_das = false, use_bhc = false,
        use_pcct_corrections = false,
        pcct_noise_reduction = 0.0,
        n_energy_bins = 100,
        seed = 42
    )

    t = time()
    ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
    BS.simulate!(ws, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
    sim_time = time() - t

    sino = ws.sino_noisy_out
    geom = ws.geom
    println("  Sim time: $(@sprintf("%.1f", sim_time))s")
    println("  Sinogram: $(size(sino))")

    ws_fdk = BS.create_fdk_recon_workspace(sino, geom, recon_size; filter=BS.HannFilter())
    recon = Array(BS.reconstruct!(ws_fdk, sino, geom, recon_size))

    println("  Recon range: [$(round(minimum(recon), sigdigits=4)), $(round(maximum(recon), sigdigits=4))]")

    # Save reconstruction (raw μ, consistent window for all)
    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1,1], title="$label ($(det_cols) cols)", aspect=CM.DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    CM.heatmap!(ax, recon[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.45))
    CM.Colorbar(f[1,2], colormap=:grays, colorrange=(0.0, 0.45), label="μ (cm⁻¹)")
    CM.save(joinpath(OUTPUT_DIR, "dexel_$(label).png"), f)
    println("  Saved dexel_$(label).png")

    # Also save a zoomed-in view around ribs (where artifacts should be visible)
    cx, cy = RECON_XY ÷ 2, RECON_XY ÷ 2
    zoom_r = 80  # ~80 pixel radius zoom

    f2 = CM.Figure(size=(600, 600))
    ax2 = CM.Axis(f2[1,1], title="$label ZOOMED", aspect=CM.DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    x_range = max(1, cx-zoom_r):min(RECON_XY, cx+zoom_r)
    y_range = max(1, cy-zoom_r):min(RECON_XY, cy+zoom_r)
    CM.heatmap!(ax2, recon[x_range, y_range, slice_idx], colormap=:grays, colorrange=(0.15, 0.35))
    CM.Colorbar(f2[1,2], colormap=:grays, colorrange=(0.15, 0.35), label="μ (cm⁻¹)")
    CM.save(joinpath(OUTPUT_DIR, "dexel_$(label)_zoom.png"), f2)
    println("  Saved dexel_$(label)_zoom.png")

    results[label] = recon[:,:,slice_idx]

    ws = nothing; ws_fdk = nothing
    GC.gc(true)
end

# ─── SQUARE PIXEL TEST ────────────────────────────────────────────────────────
if RUN_SQUARE_PIXEL_TEST
    println("\n" * "=" ^ 70)
    println("SQUARE PIXEL TEST: col_size = row_size = 1.0mm")
    println("=" ^ 70)

    det_cols = ceil(Int, phantom_extent_mm * mag / 1.0)
    println("  Detector: $(det_cols) cols × $(DETECTOR_ROWS) rows, 1.0mm × 1.0mm")

    scanner_sq = BS.Scanner(
        source_to_isocenter = SID,
        source_to_detector = SDD,
        detector_rows = DETECTOR_ROWS,
        detector_cols = det_cols,
        detector_row_size = 1.0,   # Same as col_size!
        detector_col_size = 1.0,
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

    sim_opts_sq = BS.SimOptions(
        fidelity = :high,
        use_fill_factor = false,
        use_flat_filter = true,
        use_bowtie_filter = false,
        use_detector_efficiency = false,
        use_scatter = false, use_scatter_correction = false,
        use_crosstalk = false, use_optical_crosstalk = false,
        use_focal_spot = false, use_noise = false,
        use_lag = false, use_heel_effect = false,
        use_das = false, use_bhc = false,
        use_pcct_corrections = false,
        pcct_noise_reduction = 0.0,
        n_energy_bins = 100,
        seed = 42
    )

    t = time()
    ws = BS.create_eict_workspace(scanner_sq, protocol, sim_opts_sq, recon_opts, phantom_gpu)
    BS.simulate!(ws, phantom_gpu, scanner_sq, protocol, sim_opts_sq, recon_opts)
    println("  Sim time: $(@sprintf("%.1f", time() - t))s")

    ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=BS.HannFilter())
    recon_sq = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1,1], title="Square pixels 1.0mm", aspect=CM.DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    CM.heatmap!(ax, recon_sq[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.45))
    CM.Colorbar(f[1,2], colormap=:grays, colorrange=(0.0, 0.45), label="μ (cm⁻¹)")
    CM.save(joinpath(OUTPUT_DIR, "dexel_square_1mm.png"), f)
    println("  Saved dexel_square_1mm.png")

    results["square_1mm"] = recon_sq[:,:,slice_idx]
    ws = nothing; ws_fdk = nothing
    GC.gc(true)
end

# ─── COMPARISON FIGURE ────────────────────────────────────────────────────────
println("\nCreating comparison figure...")
labels_in_order = [c[2] for c in DEXEL_CONFIGS]
n = length(labels_in_order)

f_cmp = CM.Figure(size=(300 * n, 300))
for (i, label) in enumerate(labels_in_order)
    ax = CM.Axis(f_cmp[1, i], title=label, aspect=CM.DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    CM.heatmap!(ax, results[label], colormap=:grays, colorrange=(0.0, 0.45))
end
CM.save(joinpath(OUTPUT_DIR, "dexel_sweep_comparison.png"), f_cmp)
println("  Saved dexel_sweep_comparison.png")

# ─── ZOOMED COMPARISON ───────────────────────────────────────────────────────
cx, cy = RECON_XY ÷ 2, RECON_XY ÷ 2
zoom_r = 80

f_zoom = CM.Figure(size=(300 * n, 300))
for (i, label) in enumerate(labels_in_order)
    ax = CM.Axis(f_zoom[1, i], title=label, aspect=CM.DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    x_range = max(1, cx-zoom_r):min(RECON_XY, cx+zoom_r)
    y_range = max(1, cy-zoom_r):min(RECON_XY, cy+zoom_r)
    CM.heatmap!(ax, results[label][x_range, y_range], colormap=:grays, colorrange=(0.15, 0.35))
end
CM.save(joinpath(OUTPUT_DIR, "dexel_sweep_zoom.png"), f_zoom)
println("  Saved dexel_sweep_zoom.png")

# ─── LINE PROFILE COMPARISON ─────────────────────────────────────────────────
profile_row = RECON_XY ÷ 2

f_prof = CM.Figure(size=(1000, 400))
ax_prof = CM.Axis(f_prof[1,1], title="Line profiles (y=$profile_row) across dexel configurations",
    xlabel="x pixel", ylabel="μ (cm⁻¹)")
colors = [:blue, :green, :orange, :red]
for (i, label) in enumerate(labels_in_order)
    CM.lines!(ax_prof, results[label][:, profile_row], label=label, color=colors[i])
end
CM.axislegend(ax_prof, position=:rt)
CM.save(joinpath(OUTPUT_DIR, "dexel_sweep_profiles.png"), f_prof)
println("  Saved dexel_sweep_profiles.png")

println("\n" * "=" ^ 70)
println("DONE — Dexel sweep complete")
println("Total time: $(@sprintf("%.1f", time() - t0))s")
println("=" ^ 70)
