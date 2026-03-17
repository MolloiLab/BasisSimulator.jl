#!/usr/bin/env julia
#
# debug_brain_compare.jl — Diagnostic comparison of three brain perfusion volumes
#
# Usage:  julia --project=verification verification/scripts/debug_brain_compare.jl
#
# Compares:
#   1. Current 160-slice output  (broken)
#   2. Previous 176-slice output (working)
#   3. CatSim reference 160-slice
#

using Statistics, Printf
import CairoMakie as CM

# ─── paths ──────────────────────────────────────────────────────────────────────

const BASE = dirname(dirname(@__DIR__))  # BasisSimulator.jl/
const RAW_DIR = joinpath(BASE, "verification", "results", "brain_perfusion", "raw")
const REF_DIR = joinpath(BASE, "verification", "data", "brain_perfusion", "reference", "t23001")
const OUT_DIR = joinpath(BASE, "verification", "results", "brain_perfusion", "debug")

const PATH_160 = joinpath(RAW_DIR, "brain_fdk_t0s_512x512x160.raw")
const PATH_176 = joinpath(RAW_DIR, "brain_fdk_t0s_512x512x176.raw")
const PATH_REF = joinpath(REF_DIR, "contrast_brain_phantom_t23001_512x512x160.raw")

mkpath(OUT_DIR)

# ─── helpers ────────────────────────────────────────────────────────────────────

function load_raw(path::String, nx::Int, ny::Int, nz::Int)
    @assert isfile(path) "File not found: $path"
    nbytes = nx * ny * nz * sizeof(Float32)
    fsize = filesize(path)
    @assert fsize == nbytes "Size mismatch: expected $nbytes bytes, got $fsize for $(basename(path))"
    vol = Array{Float32}(undef, nx, ny, nz)
    open(path, "r") do io
        read!(io, vol)
    end
    return vol
end

function print_stats(name::String, vol::Array{Float32,3})
    println("\n", "="^70)
    println("  $name  ($(size(vol,1))×$(size(vol,2))×$(size(vol,3)))")
    println("="^70)
    v = vec(vol)

    n_nan = count(isnan, v)
    n_inf = count(isinf, v)
    v_clean = filter(x -> isfinite(x), v)

    @printf("  Min        : %12.4f\n", minimum(v_clean))
    @printf("  Max        : %12.4f\n", maximum(v_clean))
    @printf("  Mean       : %12.4f\n", mean(v_clean))
    @printf("  Median     : %12.4f\n", median(v_clean))
    @printf("  Std        : %12.4f\n", std(v_clean))
    println()

    pcts = [1, 5, 25, 50, 75, 95, 99]
    for p in pcts
        @printf("  P%02d        : %12.4f\n", p, quantile(v_clean, p / 100))
    end
    println()

    @printf("  NaN count  : %d\n", n_nan)
    @printf("  Inf count  : %d\n", n_inf)
    @printf("  Brain HU [0,80]    : %d  (%.1f%%)\n",
        count(x -> 0 ≤ x ≤ 80, v_clean), 100 * count(x -> 0 ≤ x ≤ 80, v_clean) / length(v_clean))
    @printf("  Bone HU [200,2000] : %d  (%.1f%%)\n",
        count(x -> 200 ≤ x ≤ 2000, v_clean), 100 * count(x -> 200 ≤ x ≤ 2000, v_clean) / length(v_clean))
    @printf("  Negative HU        : %d  (%.1f%%)\n",
        count(x -> x < 0, v_clean), 100 * count(x -> x < 0, v_clean) / length(v_clean))
end

function mid_slice(vol::Array{Float32,3})
    return vol[:, :, size(vol, 3) ÷ 2 + 1]
end

function center_roi(slice::Matrix{Float32}; half=10)
    cx, cy = size(slice, 1) ÷ 2, size(slice, 2) ÷ 2
    return slice[cx-half+1:cx+half, cy-half+1:cy+half]
end

function window_slice(slice::Matrix{Float32}; W=80, L=40)
    lo = L - W / 2
    hi = L + W / 2
    return clamp.((slice .- lo) ./ (hi - lo), 0f0, 1f0)
end

function save_slice_png(path::String, slice::Matrix{Float32}; W=80, L=40, title="")
    img = window_slice(slice; W, L)
    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1, 1]; title, aspect=CM.DataAspect(), yreversed=true)
    CM.heatmap!(ax, img'; colormap=:grays, colorrange=(0, 1))
    CM.hidedecorations!(ax)
    CM.save(path, f, px_per_unit=2)
    println("  Saved: $path")
end

function save_diff_png(path::String, diff::Matrix{Float32}; clim=50, title="")
    f = CM.Figure(size=(600, 600))
    ax = CM.Axis(f[1, 1]; title, aspect=CM.DataAspect(), yreversed=true)
    CM.heatmap!(ax, diff'; colormap=:RdBu, colorrange=(-clim, clim))
    CM.Colorbar(f[1, 2], colormap=:RdBu, limits=(-clim, clim), label="HU")
    CM.hidedecorations!(ax)
    CM.save(path, f, px_per_unit=2)
    println("  Saved: $path")
end

# ─── Step 1: Load volumes ──────────────────────────────────────────────────────

println("\n", "▶"^5, " Loading volumes...")

vol_160 = load_raw(PATH_160, 512, 512, 160)
vol_176 = load_raw(PATH_176, 512, 512, 176)
vol_ref = load_raw(PATH_REF, 512, 512, 160)

println("  ✓ Current  160-slice: $(size(vol_160))")
println("  ✓ Previous 176-slice: $(size(vol_176))")
println("  ✓ CatSim reference  : $(size(vol_ref))")

# ─── Step 2: Global statistics ──────────────────────────────────────────────────

print_stats("Current 160-slice (broken)", vol_160)
print_stats("Previous 176-slice (working)", vol_176)
print_stats("CatSim reference 160-slice", vol_ref)

# ─── Step 3: Mid-slice comparison ──────────────────────────────────────────────

println("\n", "="^70)
println("  Mid-slice center ROI (20×20 px) comparison")
println("="^70)

for (name, vol) in [("Current 160", vol_160), ("Previous 176", vol_176), ("CatSim ref", vol_ref)]
    sl = mid_slice(vol)
    roi = center_roi(sl)
    @printf("  %-14s : mean = %8.2f HU,  std = %8.2f HU\n", name, mean(roi), std(roi))
end

println("\n  Horizontal line profile through center (every 50th pixel):")
for (name, vol) in [("Current 160", vol_160), ("Previous 176", vol_176), ("CatSim ref", vol_ref)]
    sl = mid_slice(vol)
    cy = size(sl, 2) ÷ 2 + 1
    idxs = 1:50:size(sl, 1)
    vals = sl[idxs, cy]
    print("  $name: ")
    for (i, v) in zip(idxs, vals)
        @printf("[%3d]=%7.1f  ", i, v)
    end
    println()
end

# ─── Step 4: Difference maps (160 vs reference) ────────────────────────────────

println("\n", "="^70)
println("  Difference: Current 160 − CatSim reference")
println("="^70)

diff_160_ref = vol_160 .- vol_ref
d = vec(diff_160_ref)
d_finite = filter(isfinite, d)
@printf("  RMSE               : %12.4f HU\n", sqrt(mean(d_finite .^ 2)))
@printf("  Max |diff|          : %12.4f HU\n", maximum(abs.(d_finite)))
@printf("  Mean diff (bias)    : %12.4f HU\n", mean(d_finite))
@printf("  Median diff         : %12.4f HU\n", median(d_finite))
@printf("  Std of diff         : %12.4f HU\n", std(d_finite))

# ─── Step 5: Z-alignment check (176 vs 160) ────────────────────────────────────

println("\n", "="^70)
println("  Z-alignment: Previous 176 vs Current 160")
println("="^70)

# The 176-slice volume has 8 extra slices on each side (center-aligned)
offset = (176 - 160) ÷ 2  # = 8
vol_176_cropped = vol_176[:, :, offset+1:offset+160]

diff_176_160 = vol_176_cropped .- vol_160
d2 = vec(diff_176_160)
d2_finite = filter(isfinite, d2)
@printf("  RMSE (176-cropped vs 160)  : %12.4f HU\n", sqrt(mean(d2_finite .^ 2)))
@printf("  Max |diff|                  : %12.4f HU\n", maximum(abs.(d2_finite)))
@printf("  Mean diff (bias)            : %12.4f HU\n", mean(d2_finite))

println("\n  Per-slice mean (every 20th slice):")
@printf("  %5s  %12s  %12s  %12s  %12s\n", "Slice", "Current160", "Prev176crop", "CatSimRef", "Δ(160-ref)")
for z in 1:20:160
    m160 = mean(vol_160[:, :, z])
    m176 = mean(vol_176_cropped[:, :, z])
    mref = mean(vol_ref[:, :, z])
    @printf("  %5d  %12.2f  %12.2f  %12.2f  %12.2f\n", z, m160, m176, mref, m160 - mref)
end

# ─── Step 6: Tissue ROI analysis ───────────────────────────────────────────────

println("\n", "="^70)
println("  Tissue ROI analysis (center brain, 200:300 × 200:300)")
println("="^70)

roi_range = 200:300
mz = 80  # mid-z for 160 slices

for (name, vol, z) in [
    ("Current 160", vol_160, mz),
    ("Previous 176 (cropped)", vol_176_cropped, mz),
    ("CatSim reference", vol_ref, mz),
]
    roi = vol[roi_range, roi_range, z]
    @printf("  %-25s : mean = %8.2f HU,  std = %8.2f HU,  [min,max] = [%.1f, %.1f]\n",
        name, mean(roi), std(roi), minimum(roi), maximum(roi))
end

println("\n  Expected tissue values:")
println("    Gray matter  : 30–45 HU")
println("    White matter : 20–30 HU")
println("    CSF          :  0–15 HU")
println("    Bone         : 200–2000 HU")

# ─── Step 7: Save comparison PNGs ──────────────────────────────────────────────

println("\n", "="^70)
println("  Saving comparison PNGs")
println("="^70)

save_slice_png(
    joinpath(OUT_DIR, "mid_current_160.png"),
    mid_slice(vol_160);
    title="Current 160-slice (broken)",
)
save_slice_png(
    joinpath(OUT_DIR, "mid_previous_176.png"),
    mid_slice(vol_176);
    title="Previous 176-slice (working)",
)
save_slice_png(
    joinpath(OUT_DIR, "mid_catsim_ref.png"),
    mid_slice(vol_ref);
    title="CatSim reference 160-slice",
)

# Difference: current 160 - reference
save_diff_png(
    joinpath(OUT_DIR, "diff_current_vs_ref.png"),
    mid_slice(diff_160_ref);
    clim=50,
    title="Current 160 − CatSim ref",
)

# Difference: previous 176 (cropped) - current 160
save_diff_png(
    joinpath(OUT_DIR, "diff_prev176_vs_current160.png"),
    mid_slice(diff_176_160);
    clim=50,
    title="Previous 176 (cropped) − Current 160",
)

# Difference: previous 176 (cropped) - CatSim reference
diff_176_ref = vol_176_cropped .- vol_ref
save_diff_png(
    joinpath(OUT_DIR, "diff_prev176_vs_ref.png"),
    mid_slice(diff_176_ref);
    clim=50,
    title="Previous 176 (cropped) − CatSim ref",
)

# ─── Summary ────────────────────────────────────────────────────────────────────

println("\n", "="^70)
println("  DIAGNOSTIC SUMMARY")
println("="^70)

bias = mean(filter(isfinite, vec(diff_160_ref)))
rmse_160_ref = sqrt(mean(filter(isfinite, vec(diff_160_ref)) .^ 2))
rmse_176_160 = sqrt(mean(filter(isfinite, vec(diff_176_160)) .^ 2))
rmse_176_ref = sqrt(mean(filter(isfinite, vec(diff_176_ref)) .^ 2))

@printf("  RMSE current160 vs ref     : %8.2f HU\n", rmse_160_ref)
@printf("  RMSE prev176crop vs 160    : %8.2f HU\n", rmse_176_160)
@printf("  RMSE prev176crop vs ref    : %8.2f HU\n", rmse_176_ref)
@printf("  Global HU bias (160−ref)   : %8.2f HU\n", bias)

if abs(bias) > 20
    println("\n  ⚠  Large global HU offset detected → likely μ_water calibration issue")
elseif rmse_160_ref > 50
    println("\n  ⚠  High RMSE without large bias → likely spatial artifacts (BHC, cupping, filter)")
else
    println("\n  ✓  Volumes are reasonably close")
end

# ─── Step 8: Spatial artifact analysis ──────────────────────────────────────

println("\n", "="^70)
println("  Step 8: Spatial artifact localization")
println("="^70)

# 8a. Per-slice statistics to find z-dependent artifacts (cone-beam)
println("\n  Per-slice stats for Current 160 (every 10th slice):")
@printf("  %5s  %10s  %10s  %10s  %12s  %12s\n", "Slice", "Mean", "Std", "Median", "Min", "Max")
for z in 1:10:160
    sl = vol_160[:, :, z]
    @printf("  %5d  %10.1f  %10.1f  %10.1f  %12.1f  %12.1f\n",
        z, mean(sl), std(sl), median(sl), minimum(sl), maximum(sl))
end

# 8b. Radial profile — are extreme values at periphery or center?
println("\n  Radial HU profile (mid-slice, Current 160):")
sl_mid = mid_slice(vol_160)
nx, ny = size(sl_mid)
cx, cy = nx ÷ 2, ny ÷ 2
pixel_cm = 40.0 / 512.0
fov_radius = 20.0  # cm

# Bin voxels by radius
n_bins = 20
bin_edges = range(0, fov_radius * sqrt(2), length=n_bins+1)
bin_means = zeros(n_bins)
bin_stds = zeros(n_bins)
bin_maxs = fill(-Inf, n_bins)
bin_counts = zeros(Int, n_bins)

for j in 1:ny, i in 1:nx
    r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
    b = searchsortedlast(collect(bin_edges), r)
    if 1 <= b <= n_bins
        bin_counts[b] += 1
        bin_means[b] += sl_mid[i, j]
        bin_maxs[b] = max(bin_maxs[b], sl_mid[i, j])
    end
end
for b in 1:n_bins
    if bin_counts[b] > 0
        bin_means[b] /= bin_counts[b]
    end
end
# Second pass for std
for j in 1:ny, i in 1:nx
    r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
    b = searchsortedlast(collect(bin_edges), r)
    if 1 <= b <= n_bins && bin_counts[b] > 0
        bin_stds[b] += (sl_mid[i, j] - bin_means[b])^2
    end
end
for b in 1:n_bins
    if bin_counts[b] > 1
        bin_stds[b] = sqrt(bin_stds[b] / (bin_counts[b] - 1))
    end
end

@printf("  %8s  %10s  %10s  %12s  %8s\n", "R (cm)", "Mean HU", "Std HU", "Max HU", "Count")
for b in 1:n_bins
    r_mid = (bin_edges[b] + bin_edges[b+1]) / 2
    @printf("  %8.1f  %10.1f  %10.1f  %12.1f  %8d\n",
        r_mid, bin_means[b], bin_stds[b], bin_maxs[b], bin_counts[b])
end

# 8c. Count extreme voxels (>1000 HU or < -1100 HU) per slice
println("\n  Extreme voxel count per slice (>1000 HU):")
@printf("  %5s  %10s  %10s  %12s\n", "Slice", ">1000 HU", ">10000 HU", ">100000 HU")
for z in 1:10:160
    sl = vol_160[:, :, z]
    @printf("  %5d  %10d  %10d  %12d\n",
        z, count(x -> x > 1000, sl), count(x -> x > 10000, sl), count(x -> x > 100000, sl))
end

# 8d. FOV boundary check — voxels at FOV edge
println("\n  FOV boundary analysis (ring at r=19-20 cm, mid-slice):")
fov_ring = Float32[]
for j in 1:ny, i in 1:nx
    r = sqrt(((i - cx) * pixel_cm)^2 + ((j - cy) * pixel_cm)^2)
    if 19.0 <= r <= 20.0
        push!(fov_ring, sl_mid[i, j])
    end
end
if !isempty(fov_ring)
    @printf("  FOV edge ring: mean=%8.1f, std=%8.1f, min=%8.1f, max=%8.1f, N=%d\n",
        mean(fov_ring), std(fov_ring), minimum(fov_ring), maximum(fov_ring), length(fov_ring))
end

# ─── Step 9: Multi-window visualization ─────────────────────────────────────

println("\n", "="^70)
println("  Step 9: Multi-window PNGs (matching what you see in notebook)")
println("="^70)

# Wide window to see the rings/white background you described
save_slice_png(joinpath(OUT_DIR, "mid_160_wide_window.png"),
    mid_slice(vol_160); W=3000, L=0, title="Current 160 — Wide window (W=3000 L=0)")
save_slice_png(joinpath(OUT_DIR, "mid_160_soft_tissue.png"),
    mid_slice(vol_160); W=400, L=40, title="Current 160 — Soft tissue (W=400 L=40)")
save_slice_png(joinpath(OUT_DIR, "mid_160_bone.png"),
    mid_slice(vol_160); W=2000, L=500, title="Current 160 — Bone window (W=2000 L=500)")

# Same windows for CatSim reference
save_slice_png(joinpath(OUT_DIR, "mid_ref_wide_window.png"),
    mid_slice(vol_ref); W=3000, L=0, title="CatSim ref — Wide window (W=3000 L=0)")
save_slice_png(joinpath(OUT_DIR, "mid_ref_soft_tissue.png"),
    mid_slice(vol_ref); W=400, L=40, title="CatSim ref — Soft tissue (W=400 L=40)")

# Top and bottom slices (where cone-beam artifacts are worst)
save_slice_png(joinpath(OUT_DIR, "slice5_160_wide.png"),
    vol_160[:, :, 5]; W=3000, L=0, title="Current 160 — Slice 5 (near top, W=3000)")
save_slice_png(joinpath(OUT_DIR, "slice155_160_wide.png"),
    vol_160[:, :, 155]; W=3000, L=0, title="Current 160 — Slice 155 (near bottom, W=3000)")
save_slice_png(joinpath(OUT_DIR, "slice80_160_wide.png"),
    vol_160[:, :, 80]; W=3000, L=0, title="Current 160 — Slice 80 (center, W=3000)")

# Radial profile plot
let
    f = CM.Figure(size=(800, 400))
    ax = CM.Axis(f[1, 1]; title="Radial HU profile (mid-slice, Current 160)",
        xlabel="Radius (cm)", ylabel="Mean HU")
    r_mids = [(bin_edges[b] + bin_edges[b+1]) / 2 for b in 1:n_bins]
    CM.band!(ax, r_mids, bin_means .- bin_stds, bin_means .+ bin_stds;
        color=(:blue, 0.2))
    CM.lines!(ax, r_mids, bin_means; color=:blue, linewidth=2, label="Mean ± Std")
    CM.lines!(ax, r_mids, Float64.(bin_maxs); color=:red, linewidth=1, linestyle=:dash, label="Max")
    CM.vlines!(ax, [20.0]; color=:gray, linestyle=:dot, label="FOV radius")
    CM.axislegend(ax; position=:lt)
    CM.save(joinpath(OUT_DIR, "radial_profile_160.png"), f, px_per_unit=2)
    println("  Saved: radial_profile_160.png")
end

# ─── Step 10: Compare a contrast timepoint if available ─────────────────────

println("\n", "="^70)
println("  Step 10: Contrast timepoint check")
println("="^70)

# Check for a contrast-enhanced timepoint (e.g., t23s — peak enhancement)
contrast_path = joinpath(RAW_DIR, "brain_fdk_t23s_512x512x160.raw")
if isfile(contrast_path)
    vol_contrast = load_raw(contrast_path, 512, 512, 160)
    diff_contrast = vol_contrast .- vol_160
    println("  Contrast (t=23s) minus baseline (t=0s):")
    roi_c = diff_contrast[roi_range, roi_range, mz]
    @printf("    Center brain ROI Δ: mean=%6.1f HU, std=%6.1f HU, [%.1f, %.1f]\n",
        mean(roi_c), std(roi_c), minimum(roi_c), maximum(roi_c))
    println("    Expected: arteries +100–200 HU, tissue +5–20 HU")

    save_slice_png(joinpath(OUT_DIR, "mid_contrast_t23s.png"),
        mid_slice(vol_contrast); W=80, L=40,
        title="Contrast t=23s — Brain window")
    save_diff_png(joinpath(OUT_DIR, "diff_contrast_minus_baseline.png"),
        mid_slice(diff_contrast); clim=100,
        title="t=23s minus t=0s (iodine enhancement)")
else
    # Find any contrast timepoint
    contrast_files = filter(f -> occursin(r"brain_fdk_t\d+s_512x512x160\.raw", f) && !occursin("t0s", f),
        readdir(RAW_DIR))
    if !isempty(contrast_files)
        cf = first(sort(contrast_files))
        println("  Found contrast file: $cf")
        vol_c = load_raw(joinpath(RAW_DIR, cf), 512, 512, 160)
        diff_c = vol_c .- vol_160
        roi_c = diff_c[roi_range, roi_range, mz]
        @printf("    Center brain ROI Δ: mean=%6.1f HU, std=%6.1f HU\n", mean(roi_c), std(roi_c))

        save_slice_png(joinpath(OUT_DIR, "mid_contrast_found.png"),
            mid_slice(vol_c); W=80, L=40, title="$cf — Brain window")
        save_diff_png(joinpath(OUT_DIR, "diff_contrast_found.png"),
            mid_slice(diff_c); clim=100, title="$(cf) minus baseline")
    else
        println("  No contrast timepoint files found")
    end
end

println("\n  PNGs saved to: $OUT_DIR")
println("  Done.\n")
