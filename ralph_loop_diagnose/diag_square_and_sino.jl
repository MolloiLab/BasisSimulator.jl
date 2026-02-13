#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# Combined: Square Pixel Test + Sinogram Inspection
#
# 1) Square pixel (col_size = row_size = 1.0mm) — TEST-DEXEL-SWEEP
# 2) Sinogram inspection at default resolution — INSPECT-SINOGRAM
#    (raw sinogram views, difference between adjacent views, line profiles,
#     filtered sinogram)
#
# Usage: julia --project=verification ralph_loop_diagnose/diag_square_and_sino.jl
# ═══════════════════════════════════════════════════════════════════════════════

using Printf

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DETECTOR_ROW_SIZE  = 0.625   # mm
DETECTOR_ROWS      = 64
VIEWS              = 1600
KVP                = 120.0
MA                 = 700.0
RECON_XY           = 512
RECON_SLICES       = 64
RECON_FOV_CM       = 35.0
DOWNSAMPLE_FACTOR  = 2
OUTPUT_DIR         = joinpath(@__DIR__, "outputs")
SID                = 625.6
SDD                = 1100.0

mkpath(OUTPUT_DIR)

println("=" ^ 70)
println("SQUARE PIXEL + SINOGRAM INSPECTION")
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

mag = SDD / SID
recon_size = (RECON_XY, RECON_XY, RECON_SLICES)
recon_opts = BS.ReconOptions(algorithm = :fdk, matrix_size = recon_size, fov_cm = RECON_FOV_CM)
protocol = BS.CTProtocol(kVp = KVP, mA = MA, views = VIEWS, rotation_time = 0.5)
slice_idx = RECON_SLICES ÷ 2

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: SQUARE PIXEL TEST (1.0mm x 1.0mm)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PART 1: SQUARE PIXEL TEST (col=1.0mm, row=1.0mm)")
println("=" ^ 70)

det_cols_sq = ceil(Int, phantom_extent_mm * mag / 1.0)
println("  Detector: $(det_cols_sq) cols × $(DETECTOR_ROWS) rows, 1.0mm × 1.0mm")

scanner_sq = BS.Scanner(
    source_to_isocenter = SID,
    source_to_detector = SDD,
    detector_rows = DETECTOR_ROWS,
    detector_cols = det_cols_sq,
    detector_row_size = 1.0,   # SQUARE: same as col
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
ws_sq = BS.create_eict_workspace(scanner_sq, protocol, sim_opts_sq, recon_opts, phantom_gpu)
BS.simulate!(ws_sq, phantom_gpu, scanner_sq, protocol, sim_opts_sq, recon_opts)
sim_time_sq = time() - t

println("  Sim time: $(@sprintf("%.1f", sim_time_sq))s")
println("  Sinogram: $(size(ws_sq.sino_noisy_out))")

ws_fdk_sq = BS.create_fdk_recon_workspace(ws_sq.sino_noisy_out, ws_sq.geom, recon_size; filter=BS.HannFilter())
recon_sq = Array(BS.reconstruct!(ws_fdk_sq, ws_sq.sino_noisy_out, ws_sq.geom, recon_size))

center_val_sq = recon_sq[RECON_XY÷2, RECON_XY÷2, slice_idx]
println("  Recon range: [$(round(minimum(recon_sq), sigdigits=4)), $(round(maximum(recon_sq), sigdigits=4))]")
println("  Center μ: $(round(center_val_sq, sigdigits=4))")

# Save square pixel reconstruction
f = CM.Figure(size=(600, 600))
ax = CM.Axis(f[1,1], title="Square pixels 1.0mm ($(det_cols_sq) cols)", aspect=CM.DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false)
CM.heatmap!(ax, recon_sq[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.45))
CM.Colorbar(f[1,2], colormap=:grays, colorrange=(0.0, 0.45), label="μ (cm⁻¹)")
CM.save(joinpath(OUTPUT_DIR, "dexel_square_1mm.png"), f)
println("  Saved dexel_square_1mm.png")

# Zoomed version
cx, cy = RECON_XY ÷ 2, RECON_XY ÷ 2
zoom_r = 80
f2 = CM.Figure(size=(600, 600))
ax2 = CM.Axis(f2[1,1], title="Square 1mm ZOOMED", aspect=CM.DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false)
x_range = max(1, cx-zoom_r):min(RECON_XY, cx+zoom_r)
y_range = max(1, cy-zoom_r):min(RECON_XY, cy+zoom_r)
CM.heatmap!(ax2, recon_sq[x_range, y_range, slice_idx], colormap=:grays, colorrange=(0.15, 0.35))
CM.Colorbar(f2[1,2], colormap=:grays, colorrange=(0.15, 0.35), label="μ (cm⁻¹)")
CM.save(joinpath(OUTPUT_DIR, "dexel_square_1mm_zoom.png"), f2)
println("  Saved dexel_square_1mm_zoom.png")

ws_sq = nothing; ws_fdk_sq = nothing
GC.gc(true)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: SINOGRAM INSPECTION (default 1.0mm col, 0.625mm row)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PART 2: SINOGRAM INSPECTION (polychromatic, 844 cols)")
println("=" ^ 70)

det_cols = ceil(Int, phantom_extent_mm * mag / 1.0)
println("  Detector: $(det_cols) cols × $(DETECTOR_ROWS) rows")

scanner = BS.Scanner(
    source_to_isocenter = SID,
    source_to_detector = SDD,
    detector_rows = DETECTOR_ROWS,
    detector_cols = det_cols,
    detector_row_size = DETECTOR_ROW_SIZE,
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
println("  Sim time: $(@sprintf("%.1f", sim_time))s")

sino_cpu = Array(ws.sino_noisy_out)
geom = ws.geom
n_cols_s, n_rows_s, n_angles = size(sino_cpu)
println("  Sinogram size: $(size(sino_cpu))")
println("  Sino range: [$(round(minimum(sino_cpu), sigdigits=4)), $(round(maximum(sino_cpu), sigdigits=4))]")

# ─── Sinogram Visualization 1: Full sinogram (all angles, middle row) ────────
println("  Generating sinogram visualizations...")

mid_row = n_rows_s ÷ 2
f1 = CM.Figure(size=(1000, 400))
ax1 = CM.Axis(f1[1,1], title="Full sinogram (row=$mid_row, all angles)",
    xlabel="Detector column", ylabel="View angle")
CM.heatmap!(ax1, sino_cpu[:, mid_row, :]', colormap=:viridis)
CM.Colorbar(f1[1,2], colormap=:viridis, label="μ·L")
CM.save(joinpath(OUTPUT_DIR, "sino_full_polychromatic.png"), f1)
println("    Saved sino_full_polychromatic.png")

# ─── Sinogram Visualization 2: Single view at angle n_angles/4 ──────────────
angle_idx = n_angles ÷ 4
f2s = CM.Figure(size=(800, 500))
ax2s = CM.Axis(f2s[1,1], title="Single view (angle=$(angle_idx)/$(n_angles))",
    xlabel="Detector column", ylabel="Detector row")
CM.heatmap!(ax2s, sino_cpu[:, :, angle_idx]', colormap=:viridis)
CM.Colorbar(f2s[1,2], colormap=:viridis, label="μ·L")
CM.save(joinpath(OUTPUT_DIR, "sino_single_view.png"), f2s)
println("    Saved sino_single_view.png")

# ─── Sinogram Visualization 3: Difference between adjacent views ────────────
# Average over 10 adjacent-view differences to reduce noise
diff_sum = zeros(Float32, n_cols_s, n_rows_s)
n_diffs = min(20, n_angles - 1)
for i in 1:n_diffs
    diff_sum .+= abs.(sino_cpu[:, :, i+1] .- sino_cpu[:, :, i])
end
diff_avg = diff_sum ./ n_diffs

f3 = CM.Figure(size=(800, 500))
ax3 = CM.Axis(f3[1,1], title="Mean |view[i+1] - view[i]| (first $n_diffs views)",
    xlabel="Detector column", ylabel="Detector row")
CM.heatmap!(ax3, diff_avg', colormap=:inferno)
CM.Colorbar(f3[1,2], colormap=:inferno, label="|Δμ·L|")
CM.save(joinpath(OUTPUT_DIR, "sino_view_diff.png"), f3)
println("    Saved sino_view_diff.png")

# ─── Sinogram Visualization 4: Line profiles at rib location ────────────────
# Look for high-contrast regions in a single row
profile_angles = [1, n_angles÷8, n_angles÷4, n_angles÷2]
f4 = CM.Figure(size=(1000, 400))
ax4 = CM.Axis(f4[1,1], title="Sinogram profiles (row=$mid_row)",
    xlabel="Detector column", ylabel="μ·L")
colors = [:blue, :green, :orange, :red]
for (i, ai) in enumerate(profile_angles)
    CM.lines!(ax4, sino_cpu[:, mid_row, ai], label="angle=$ai", color=colors[i])
end
CM.axislegend(ax4, position=:rt)
CM.save(joinpath(OUTPUT_DIR, "sino_profiles.png"), f4)
println("    Saved sino_profiles.png")

# ─── Sinogram Visualization 5: Check for oscillations ────────────────────────
# Plot sinogram value at a fixed detector column across ALL angles
# Choose a column where bone is present (high attenuation peak)
# Find the column with max value at angle 1
peak_col = argmax(sino_cpu[:, mid_row, 1])
f5 = CM.Figure(size=(1000, 400))
ax5 = CM.Axis(f5[1,1], title="Sinogram value at peak column=$peak_col across all angles",
    xlabel="View angle", ylabel="μ·L")
CM.lines!(ax5, sino_cpu[peak_col, mid_row, :], color=:blue)
CM.save(joinpath(OUTPUT_DIR, "sino_angle_trace.png"), f5)
println("    Saved sino_angle_trace.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: BARE vs POLYCHROMATIC SINOGRAM COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PART 3: BARE SINOGRAM (monochromatic) for comparison")
println("=" ^ 70)

sim_opts_bare = BS.SimOptions(
    fidelity = :high,
    use_fill_factor = false,
    use_flat_filter = false,
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

ws = nothing; GC.gc(true)

t = time()
ws_bare = BS.create_eict_workspace(scanner, protocol, sim_opts_bare, recon_opts, phantom_gpu)
BS.simulate!(ws_bare, phantom_gpu, scanner, protocol, sim_opts_bare, recon_opts)
println("  Bare sim time: $(@sprintf("%.1f", time() - t))s")

sino_bare = Array(ws_bare.sino_noisy_out)
println("  Bare sino range: [$(round(minimum(sino_bare), sigdigits=4)), $(round(maximum(sino_bare), sigdigits=4))]")

# Sinogram difference: polychromatic - bare
sino_diff = sino_cpu .- sino_bare

f6 = CM.Figure(size=(1000, 400))
ax6 = CM.Axis(f6[1,1], title="Sinogram difference: polychromatic - bare (row=$mid_row)",
    xlabel="Detector column", ylabel="View angle")
dmax = maximum(abs.(sino_diff[:, mid_row, :]))
CM.heatmap!(ax6, sino_diff[:, mid_row, :]', colormap=:RdBu, colorrange=(-dmax, dmax))
CM.Colorbar(f6[1,2], colormap=:RdBu, colorrange=(-dmax, dmax), label="Δ(μ·L)")
CM.save(joinpath(OUTPUT_DIR, "sino_diff_poly_vs_bare.png"), f6)
println("  Saved sino_diff_poly_vs_bare.png")

# Line profile comparison: bare vs polychromatic at same angle
f7 = CM.Figure(size=(1000, 400))
ax7 = CM.Axis(f7[1,1], title="Sinogram profiles: bare vs polychromatic (angle=1, row=$mid_row)",
    xlabel="Detector column", ylabel="μ·L")
CM.lines!(ax7, sino_bare[:, mid_row, 1], label="bare (mono)", color=:blue)
CM.lines!(ax7, sino_cpu[:, mid_row, 1], label="polychromatic", color=:red)
CM.axislegend(ax7, position=:rt)
CM.save(joinpath(OUTPUT_DIR, "sino_profile_bare_vs_poly.png"), f7)
println("  Saved sino_profile_bare_vs_poly.png")

# Difference profile
f8 = CM.Figure(size=(1000, 400))
ax8 = CM.Axis(f8[1,1], title="Sinogram difference profile (angle=1, row=$mid_row)",
    xlabel="Detector column", ylabel="Δ(μ·L)")
CM.lines!(ax8, sino_diff[:, mid_row, 1], color=:purple)
CM.hlines!(ax8, [0.0], color=:gray, linestyle=:dash)
CM.save(joinpath(OUTPUT_DIR, "sino_diff_profile.png"), f8)
println("  Saved sino_diff_profile.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: RECONSTRUCTION COMPARISON (bare recon for side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PART 4: RECONSTRUCTION — bare vs polychromatic")
println("=" ^ 70)

# Reconstruct polychromatic
ws2 = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gpu)
BS.simulate!(ws2, phantom_gpu, scanner, protocol, sim_opts, recon_opts)
ws_fdk_poly = BS.create_fdk_recon_workspace(ws2.sino_noisy_out, ws2.geom, recon_size; filter=BS.HannFilter())
recon_poly = Array(BS.reconstruct!(ws_fdk_poly, ws2.sino_noisy_out, ws2.geom, recon_size))

# Reconstruct bare
ws_fdk_bare = BS.create_fdk_recon_workspace(ws_bare.sino_noisy_out, ws_bare.geom, recon_size; filter=BS.HannFilter())
recon_bare = Array(BS.reconstruct!(ws_fdk_bare, ws_bare.sino_noisy_out, ws_bare.geom, recon_size))

# Recon difference image
recon_diff = recon_poly[:,:,slice_idx] .- recon_bare[:,:,slice_idx]

f9 = CM.Figure(size=(1200, 400))
ax_b = CM.Axis(f9[1,1], title="Bare (mono)", aspect=CM.DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false)
CM.heatmap!(ax_b, recon_bare[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.45))

ax_p = CM.Axis(f9[1,2], title="Polychromatic", aspect=CM.DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false)
CM.heatmap!(ax_p, recon_poly[:,:,slice_idx], colormap=:grays, colorrange=(0.0, 0.45))

ax_d = CM.Axis(f9[1,3], title="Difference (poly - bare)", aspect=CM.DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false)
rdmax = maximum(abs.(recon_diff)) * 0.5  # scale to 50% for better visibility
CM.heatmap!(ax_d, recon_diff, colormap=:RdBu, colorrange=(-rdmax, rdmax))

CM.save(joinpath(OUTPUT_DIR, "recon_bare_vs_poly_comparison.png"), f9)
println("  Saved recon_bare_vs_poly_comparison.png")

# Print stats
println("\n  Bare recon:  center=$(round(recon_bare[cx,cy,slice_idx], sigdigits=4)), range=[$(round(minimum(recon_bare), sigdigits=4)), $(round(maximum(recon_bare), sigdigits=4))]")
println("  Poly recon:  center=$(round(recon_poly[cx,cy,slice_idx], sigdigits=4)), range=[$(round(minimum(recon_poly), sigdigits=4)), $(round(maximum(recon_poly), sigdigits=4))]")
println("  Diff range:  [$(round(minimum(recon_diff), sigdigits=4)), $(round(maximum(recon_diff), sigdigits=4))]")

println("\n" * "=" ^ 70)
println("DONE — Total time: $(@sprintf("%.1f", time() - t0))s")
println("=" ^ 70)
