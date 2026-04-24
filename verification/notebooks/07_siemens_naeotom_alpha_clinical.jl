### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 08010001-0000-4000-8000-000000000000
begin
    using Pkg: Pkg
    Pkg.activate(dirname(pwd()))
    Pkg.resolve()
    Pkg.instantiate()
    using Revise
end

# ╔═╡ a129a323-d627-4272-934c-5b4734286f37
using Statistics: quantile

# ╔═╡ 08010002-0000-4000-8000-000000000000
using Markdown

# ╔═╡ 08010003-0000-4000-8000-000000000000
using FileIO

# ╔═╡ 08010004-0000-4000-8000-000000000000
using ImageMagick

# ╔═╡ 08010005-0000-4000-8000-000000000000
using Unitful: @u_str

# ╔═╡ 08010006-0000-4000-8000-000000000000
using LinearAlgebra

# ╔═╡ 08010007-0000-4000-8000-000000000000
using FFTW

# ╔═╡ 08010008-0000-4000-8000-000000000000
using Random

# ╔═╡ 08010009-0000-4000-8000-000000000000
using Metal

# ╔═╡ 08010010-0000-4000-8000-000000000000
using JLD2: JLD2

# ╔═╡ 08010011-0000-4000-8000-000000000000
using DelimitedFiles: DelimitedFiles

# ╔═╡ 08010030-0000-4000-8000-000000000000
md"""
# 1. Siemens NAEOTOM Alpha.Peak — Clinical Gammex 472 Scans (PCCT)

* **Scanner:** Siemens NAEOTOM Alpha.Peak (Photon-Counting CT, syngo CT VB10A)
* **Date:** 2026-03-13 | **Protocol:** UCI ABD PEL ROUTINE | **Site:** UCI Chao Family Cancer Center
* **Phantom:** Gammex 472 multi-energy CT calibration phantom
* **Recon FOV:** 350 mm | **Matrix:** 512 × 512 | **Pixel:** 0.684 mm | **Slice:** 0.4 mm | **Kernel:** Br44f

**Alpha.Peak** is the original (flagship) NAEOTOM Alpha — dual-source, 2×144 slices, 57.6 mm z-coverage, 66 ms temporal resolution. Retroactively rebranded at RSNA 2024 when Siemens introduced the Alpha.Pro and Alpha.Prime. UCI installed it July 2024 (first PCCT in Southern California). All three variants share the same CdTe detector, 0.11 mm resolution, and 4 energy thresholds at 20/35/55/70 keV.

Four axial acquisitions. Each has Poly FBP, Poly QIR3, and VMI at 40/70/100/140 keV (QIR3 only — no VMI FBP available).

| # | kVp | mA | mAs | CTDIvol (mGy) | Recon |
|---|-----|-----|-----|---------------|-------|
| 1 | 140 |  52 |  26 |  3.03 | Poly FBP + QIR3; VMI 40/70/100/140 keV (QIR3) |
| 2 | 140 | 174 |  88 | 10.12 | Poly FBP + QIR3; VMI 40/70/100/140 keV (QIR3) |
| 3 | 140 | 347 | 176 | 20.25 | Poly FBP + QIR3; VMI 40/70/100/140 keV (QIR3) |
| 4 | 120 | 253 | 128 | 10.15 | Poly FBP + QIR3; VMI 40/70/100/140 keV (QIR3) |

Common: Axial, 0.5 s rotation, 144 × 0.4 mm collimation, W1 (Tungsten) filter, 512 × 512, 350 mm FOV
"""

# ╔═╡ b629dcd4-fe9a-4ddd-8f45-0a74918f0093
pwd()

# ╔═╡ 08010012-0000-4000-8000-000000000000
import PlutoUI as UI

# ╔═╡ 08010013-0000-4000-8000-000000000000
import BasisSimulator as BS

# ╔═╡ 08010014-0000-4000-8000-000000000000
import CairoMakie as CM

# ╔═╡ 08010015-0000-4000-8000-000000000000
import Statistics: mean, std, cor

# ╔═╡ 08010016-0000-4000-8000-000000000000
import Statistics: median

# ╔═╡ 08010017-0000-4000-8000-000000000000
import XrayAttenuation as XA

# ╔═╡ 08010018-0000-4000-8000-000000000000
import DICOM as DCM

# ╔═╡ 08010019-0000-4000-8000-000000000000
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures"); mkpath(FIGURES_DIR)

# ╔═╡ 08010020-0000-4000-8000-000000000000
const RESULTS_DIR = joinpath(dirname(@__DIR__), "results", "naeotom_alpha"); mkpath(RESULTS_DIR)

# ╔═╡ 08010021-0000-4000-8000-000000000000
UI.TableOfContents()

# ╔═╡ 08020001-0000-4000-8000-000000000000
md"""
## 2. Helper Functions

Measurement and phantom-creation functions (shared with nb06).
"""

# ╔═╡ 08020002-0000-4000-8000-000000000000
"""
    load_hu_volume(dcms) -> Array{Float32, 3}

Convert pre-loaded DICOM datasets into a Float32 HU volume.
Siemens NAEOTOM Alpha: uncompressed UInt16 pixel data, slope=1.0, intercept=-8192.
"""
function load_hu_volume(dcms::Vector)
    sort!(dcms; by = d -> d.meta[(0x0020, 0x0013)])
    slope = Float32(dcms[1].meta[(0x0028, 0x1053)])
    intercept = Float32(dcms[1].meta[(0x0028, 0x1052)])
    slices = map(dcms) do dcm
        raw = dcm.meta[(0x7fe0, 0x0010)]   # Matrix{UInt16} for Siemens (uncompressed)
        Float32.(raw) .* slope .+ intercept
    end
    return cat(slices...; dims = 3)
end

# ╔═╡ 08020002-b000-4000-8000-000000000001
# Inner ring angular offset correction (degrees) — tune in Pluto to align I 2.0 / I 5.0
inner_ring_offset_deg = 0.0

# ╔═╡ 08020003-0000-4000-8000-000000000000
"""
    segment_gammex_rods(hu_slice; fov_cm, ...) -> (mask, rod_info, center_info)

Segment all 16 Gammex 472 insert rods from a 2D CT slice.
"""
function segment_gammex_rods(
        hu_slice;
        fov_cm = 35.0,
        body_threshold_hu = -400.0,
        body_radius_cm = 16.5,
        outer_ring_cm = 10.5,
        inner_ring_cm = 5.5,
        rod_radius_cm = 1.4,
        roi_fraction = 0.6,
        clockwise = true,
    )
    nx, ny = size(hu_slice)
    pixel_cm = fov_cm / nx

    # Step 1: Find phantom center
    body = hu_slice .> body_threshold_hu
    total = max(Float64(sum(body)), 1.0)
    cx = sum(Float64(i) * body[i, j] for j in 1:ny, i in 1:nx) / total
    cy = sum(Float64(j) * body[i, j] for j in 1:ny, i in 1:nx) / total

    body_r_sq = (body_radius_cm / pixel_cm)^2
    for _ in 1:3
        sx, sy, cnt = 0.0, 0.0, 0.0
        for j in 1:ny, i in 1:nx
            if (i - cx)^2 + (j - cy)^2 <= body_r_sq && body[i, j]
                sx += i; sy += j; cnt += 1.0
            end
        end
        if cnt > 0
            cx = sx / cnt; cy = sy / cnt
        end
    end

    # Step 2: Detect rotation via outer ring angular HU profile
    r_outer_pix = outer_ring_cm / pixel_cm
    n_sample = 720
    sample_angles = range(0, 2 * pi - 2 * pi / n_sample, length = n_sample)

    profile = zeros(Float64, n_sample)
    for (k, th) in enumerate(sample_angles)
        s, c = 0.0, 0
        for dr in -3:3
            xi = round(Int, cx + (r_outer_pix + dr) * cos(th))
            yi = round(Int, cy + (r_outer_pix + dr) * sin(th))
            if 1 <= xi <= nx && 1 <= yi <= ny
                s += hu_slice[xi, yi]; c += 1
            end
        end
        profile[k] = c > 0 ? s / c : 0.0
    end

    smooth_w = max(1, round(Int, 5.0 / (360.0 / n_sample)))
    smoothed = similar(profile)
    for k in 1:n_sample
        s, c = 0.0, 0
        for d in -smooth_w:smooth_w
            s += profile[mod1(k + d, n_sample)]; c += 1
        end
        smoothed[k] = s / c
    end

    k_max = argmax(smoothed)
    k_prev = mod1(k_max - 1, n_sample)
    k_next = mod1(k_max + 1, n_sample)
    y_m, y_0, y_p = smoothed[k_prev], smoothed[k_max], smoothed[k_next]
    denom = y_m - 2y_0 + y_p
    d_sub = abs(denom) > 1.0e-12 ? 0.5 * (y_m - y_p) / denom : 0.0
    d_sub = clamp(d_sub, -0.5, 0.5)
    th_ca400 = sample_angles[k_max] + d_sub * (2 * pi / n_sample)
    rotation = th_ca400

    # Step 3: Place ROIs at all 16 rod positions
    # clockwise=true: clinical scan convention (CW from Ca400)
    # clockwise=false: simulated phantom convention (CCW, mirrored by permutedims)
    dir = clockwise ? 1 : -1
    outer_start = th_ca400 - dir * 3 * pi / 4
    outer_angles = [outer_start + dir * (i - 1) * pi / 4 for i in 1:8]
    outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]
    outer_names = [
        "Ca 100", "Ca 200", "Ca 300", "Ca 400",
        "Water (O)", "SW ref 1", "SW ref 2", "Ca 50",
    ]

    inner_start = outer_start - dir * pi / 8 + deg2rad(inner_ring_offset_deg)
    inner_angles = [inner_start + dir * (i - 1) * pi / 4 for i in 1:8]
    inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]
    inner_names = [
        "I 2.5", "I 5.0", "I 7.5", "I 10.0",
        "I 15.0", "I 20.0", "Water (I)", "I 2.0",
    ]

    roi_r_pix = rod_radius_cm * roi_fraction / pixel_cm
    roi_r_sq = roi_r_pix^2
    search_r_pix = rod_radius_cm / pixel_cm * 1.5

    mask = zeros(UInt8, nx, ny)
    rod_info = []

    for (angles, labels, names, ring_cm, ring_sym) in [
            (outer_angles, outer_labels, outer_names, outer_ring_cm, :outer),
            (inner_angles, inner_labels, inner_names, inner_ring_cm, :inner),
        ]
        ring_pix = ring_cm / pixel_cm
        for (th, lbl, name) in zip(angles, labels, names)
            ecx = cx + ring_pix * cos(th)
            ecy = cy + ring_pix * sin(th)

            si_lo = max(1, floor(Int, ecx - search_r_pix))
            si_hi = min(nx, ceil(Int, ecx + search_r_pix))
            sj_lo = max(1, floor(Int, ecy - search_r_pix))
            sj_hi = min(ny, ceil(Int, ecy + search_r_pix))

            local_vals = Float64[]
            for sj in sj_lo:sj_hi, si in si_lo:si_hi
                if (si - ecx)^2 + (sj - ecy)^2 <= search_r_pix^2
                    push!(local_vals, Float64(hu_slice[si, sj]))
                end
            end

            # Use geometric position directly — no refinement
            rcx, rcy = ecx, ecy

            i_lo = max(1, floor(Int, rcx - roi_r_pix - 1))
            i_hi = min(nx, ceil(Int, rcx + roi_r_pix + 1))
            j_lo = max(1, floor(Int, rcy - roi_r_pix - 1))
            j_hi = min(ny, ceil(Int, rcy + roi_r_pix + 1))

            vals = Float64[]
            for j in j_lo:j_hi, i in i_lo:i_hi
                if (i - rcx)^2 + (j - rcy)^2 <= roi_r_sq
                    mask[i, j] = lbl
                    push!(vals, Float64(hu_slice[i, j]))
                end
            end

            push!(
                rod_info, (
                    label = lbl, name = name, ring = ring_sym,
                    cx = rcx, cy = rcy,
                    angle_deg = round(rad2deg(th), digits = 1),
                    mean_hu = isempty(vals) ? NaN : mean(vals),
                    std_hu = length(vals) > 1 ? std(vals) : NaN,
                    n_pixels = length(vals),
                )
            )
        end
    end

    return mask, rod_info, (cx = cx, cy = cy, rotation_deg = round(rad2deg(rotation), digits = 2))
end

# ╔═╡ 08020004-0000-4000-8000-000000000000
"""
    measure_nps_local(hu_slice, pixel_mm; roi_center, roi_radius_mm, ...)

Local NPS with quadratic detrending + Hann window.
"""
function measure_nps_local(
        hu_slice::AbstractMatrix, pixel_mm::Real;
        roi_center::Tuple{Int, Int},
        roi_radius_mm::Real,
        roi_size::Int = 64,
        n_rois::Int = 64,
        overlap::Float64 = 0.5,
        unit::Symbol = :lp_cm,
        smooth_hw::Int = 1,
    )
    img = Array(Float64.(hu_slice))
    ny, nx = size(img)

    cy, cx = roi_center
    roi_radius_px = roi_radius_mm / pixel_mm
    rows_in = [i for i in 1:ny if any(j -> sqrt((i - cy)^2 + (j - cx)^2) <= roi_radius_px, 1:nx)]
    cols_in = [j for j in 1:nx if any(i -> sqrt((i - cy)^2 + (j - cx)^2) <= roi_radius_px, 1:ny)]
    img_roi = img[minimum(rows_in):maximum(rows_in), minimum(cols_in):maximum(cols_in)]
    ny_roi, nx_roi = size(img_roi)

    step = max(round(Int, roi_size * (1 - overlap)), 1)
    n_y = (ny_roi - roi_size) / step + 1 |> x -> floor(Int, x)
    n_x = (nx_roi - roi_size) / step + 1 |> x -> floor(Int, x)
    rois = Matrix{Float64}[]
    for iy in 0:(n_y - 1), ix in 0:(n_x - 1)
        r0 = 1 + iy * step; c0 = 1 + ix * step
        if r0 + roi_size - 1 <= ny_roi && c0 + roi_size - 1 <= nx_roi
            push!(rois, copy(img_roi[r0:(r0 + roi_size - 1), c0:(c0 + roi_size - 1)]))
        end
    end
    if length(rois) > n_rois
        rois = rois[randperm(length(rois))[1:n_rois]]
    end
    actual_n = length(rois)

    w1d = [0.5 * (1 - cos(2 * pi * i / (roi_size - 1))) for i in 0:(roi_size - 1)]
    win = w1d * w1d'

    power_sum = zeros(Float64, roi_size, roi_size)
    for roi in rois
        x = Float64.(repeat(1:roi_size, 1, roi_size)')
        y = Float64.(repeat(1:roi_size, 1, roi_size))
        xf = vec(x); yf = vec(y); zf = vec(roi)
        A = [xf .^ 2 yf .^ 2 xf .* yf xf yf ones(length(xf))]
        c = A \ zf
        trend = c[1] .* x .^ 2 .+ c[2] .* y .^ 2 .+ c[3] .* x .* y .+ c[4] .* x .+ c[5] .* y .+ c[6]
        detrended = roi .- trend
        windowed = detrended .* win
        power_sum .+= abs2.(fft(windowed))
    end

    nps_2d = fftshift(power_sum ./ actual_n .* (pixel_mm^2 / roi_size^2))

    freq_axis = fftshift(BS.fftfreq(roi_size, 1.0 / pixel_mm))
    df = length(freq_axis) > 1 ? abs(freq_axis[2] - freq_axis[1]) : 1.0
    n_bins = length(freq_axis) / 2 + 1 |> x -> round(Int, x)
    radial_freqs = collect(range(0, stop = (n_bins - 1) * df, length = n_bins))
    nps_1d = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)
    for j in 1:roi_size, i in 1:roi_size
        fx = freq_axis[min(j, length(freq_axis))]
        fy = freq_axis[min(i, length(freq_axis))]
        r = sqrt(fx^2 + fy^2)
        bin = round(Int, r / df) + 1
        if 1 <= bin <= n_bins
            nps_1d[bin] += nps_2d[i, j]
            counts[bin] += 1
        end
    end
    for i in 1:n_bins
        counts[i] > 0 && (nps_1d[i] /= counts[i])
    end

    radial_freqs = radial_freqs[2:end]
    nps_1d = nps_1d[2:end]

    if unit == :lp_cm
        radial_freqs .*= 10.0
        nps_1d ./= 100.0
    end

    peak_idx = length(nps_1d) > 1 ? argmax(nps_1d[2:end]) + 1 : 1
    peak_freq = radial_freqs[peak_idx]
    peak_val = nps_1d[peak_idx]
    df_out = length(radial_freqs) > 1 ? radial_freqs[2] - radial_freqs[1] : 1.0
    integrated = sum(nps_1d) * df_out

    nnps_1d = integrated > 0 ? nps_1d ./ integrated : copy(nps_1d)

    if smooth_hw > 0
        n = length(nps_1d)
        smoothed_nps = similar(nps_1d)
        smoothed_nnps = similar(nnps_1d)
        for k in 1:n
            lo = max(1, k - smooth_hw)
            hi = min(n, k + smooth_hw)
            smoothed_nps[k] = mean(nps_1d[lo:hi])
            smoothed_nnps[k] = mean(nnps_1d[lo:hi])
        end
        nps_1d = smoothed_nps
        nnps_1d = smoothed_nnps
    end

    return (
        frequencies = radial_freqs, nps_1d = nps_1d, nnps_1d = nnps_1d,
        peak_frequency = peak_freq, peak_value = peak_val,
        integrated_nps = integrated, n_rois = actual_n,
        roi_size = (roi_size, roi_size), unit = unit,
    )
end

# ╔═╡ 08020005-0000-4000-8000-000000000000
"""
    measure_mtf_circular_edge(hu_slice, cx, cy, radius_cm; fov_cm, ...)

Circular-edge MTF from the phantom body boundary.
"""
function measure_mtf_circular_edge(
        hu_slice::AbstractMatrix,
        cx::Float64, cy::Float64,
        radius_cm::Float64;
        fov_cm = 35.0,
        n_angles = 720,
        oversample = 4,
        margin_inner = 15.0,
        margin_outer = 5.0,
        fov_guard_pix = 3.0,
    )
    nx, ny = size(hu_slice)
    pixel_cm = fov_cm / nx
    pixel_mm = pixel_cm * 10.0
    edge_r_pix = radius_cm / pixel_cm
    fov_r_pix = nx / 2.0

    img_cx = (nx + 1) / 2.0
    img_cy = (ny + 1) / 2.0

    r_min = edge_r_pix - margin_inner
    r_max = edge_r_pix + margin_outer
    n_r = round(Int, (r_max - r_min) * oversample)
    radii = range(r_min, r_max, length = n_r)
    positions_mm = collect((radii .- edge_r_pix) .* pixel_mm)

    esf = zeros(Float64, n_r)
    counts = zeros(Int, n_r)
    angles = range(0, 2 * pi - 2 * pi / n_angles, length = n_angles)
    n_used = 0

    for th in angles
        costh, sinth = cos(th), sin(th)
        x_outer = cx + r_max * costh
        y_outer = cy + r_max * sinth
        dist_from_img_center = sqrt((x_outer - img_cx)^2 + (y_outer - img_cy)^2)
        if dist_from_img_center > fov_r_pix - fov_guard_pix
            continue
        end
        n_used += 1
        for (k, r) in enumerate(radii)
            xi = cx + r * costh
            yi = cy + r * sinth
            x0 = floor(Int, xi); y0 = floor(Int, yi)
            x1 = x0 + 1; y1 = y0 + 1
            if 1 <= x0 && x1 <= nx && 1 <= y0 && y1 <= ny
                fx = xi - x0; fy = yi - y0
                val = (1 - fx) * (1 - fy) * hu_slice[x0, y0] + fx * (1 - fy) * hu_slice[x1, y0] +
                    (1 - fx) * fy * hu_slice[x0, y1] + fx * fy * hu_slice[x1, y1]
                esf[k] += Float64(val)
                counts[k] += 1
            end
        end
    end

    for k in 1:n_r
        counts[k] > 0 && (esf[k] /= counts[k])
    end

    if esf[1] < esf[end]
        reverse!(esf); reverse!(positions_mm)
    end

    dp = positions_mm[2] - positions_mm[1]
    lsf = diff(esf) ./ dp
    lsf_pos = (positions_mm[1:(end - 1)] .+ positions_mm[2:end]) ./ 2

    lsf_max = maximum(abs.(lsf))
    lsf_max > 0 && (lsf ./= lsf_max)

    n_pad = nextpow(2, length(lsf) * 4)
    lsf_padded = zeros(Float64, n_pad)
    offset = round(Int, (n_pad - length(lsf)) / 2)
    lsf_padded[(offset + 1):(offset + length(lsf))] .= lsf

    mtf_complex = fft(lsf_padded)
    mtf_vals = abs.(mtf_complex)
    mtf_vals ./= mtf_vals[1]

    n_pos = Int(n_pad / 2)
    freq_lp_mm = collect(0:(n_pos - 1)) ./ n_pad .* (1.0 / abs(dp))
    freq_lp_cm = freq_lp_mm .* 10.0
    mtf_curve = mtf_vals[1:n_pos]

    nyquist_lp_cm = 1.0 / (2.0 * pixel_mm) * 10.0
    keep = freq_lp_cm .<= nyquist_lp_cm
    freq_lp_cm = freq_lp_cm[keep]
    mtf_curve = mtf_curve[keep]

    function find_freq_at(level)
        for i in 1:(length(mtf_curve) - 1)
            if mtf_curve[i] >= level && mtf_curve[i + 1] < level
                t = (level - mtf_curve[i]) / (mtf_curve[i + 1] - mtf_curve[i])
                return freq_lp_cm[i] + t * (freq_lp_cm[i + 1] - freq_lp_cm[i])
            end
        end
        return mtf_curve[end] >= level ? freq_lp_cm[end] : 0.0
    end

    f50 = find_freq_at(0.5)
    f10 = find_freq_at(0.1)

    return (
        frequencies = freq_lp_cm, mtf = mtf_curve, mtf50 = f50, mtf10 = f10,
        n_angles_used = n_used,
    )
end

# ╔═╡ 08020006-0000-4000-8000-000000000000
"""
    measure_scan(hu_vol, seg_mask, seg_rods, seg_center, scan_name; fov_cm)

Full measurement suite for one reconstruction.
"""
function measure_scan(
        hu_vol::Array{Float32, 3},
        seg_mask,
        seg_rods,
        seg_center,
        scan_name::String;
        fov_cm = 35.0,
    )
    nx, ny, nz = size(hu_vol)
    mid_z = round(Int, nz / 2)
    hu_slice = hu_vol[:, :, mid_z]
    pixel_mm = fov_cm / nx * 10.0

    rod_order = [
        "Water (O)", "SW ref 1", "SW ref 2",
        "Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400",
        "Water (I)",
        "I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0",
    ]

    rod_means = Float64[]
    rod_stds = Float64[]
    rod_names = String[]

    for name in rod_order
        r = first(filter(x -> x.name == name, seg_rods))
        roi_r_pix = 1.4 * 0.6 / (fov_cm / nx)
        roi_r_sq = roi_r_pix^2
        vals = Float64[]
        i_lo = max(1, floor(Int, r.cx - roi_r_pix - 1))
        i_hi = min(nx, ceil(Int, r.cx + roi_r_pix + 1))
        j_lo = max(1, floor(Int, r.cy - roi_r_pix - 1))
        j_hi = min(ny, ceil(Int, r.cy + roi_r_pix + 1))
        for j in j_lo:j_hi, i in i_lo:i_hi
            if (i - r.cx)^2 + (j - r.cy)^2 <= roi_r_sq
                push!(vals, Float64(hu_slice[i, j]))
            end
        end
        push!(rod_means, mean(vals))
        push!(rod_stds, std(vals))
        push!(rod_names, name)
    end

    water_idx = 1
    mu_water = rod_means[water_idx]
    sigma_water = rod_stds[water_idx]
    rod_cnr = [(rod_means[i] - mu_water) / sigma_water for i in 1:length(rod_means)]

    center_row = round(Int, seg_center.cx)
    center_col = round(Int, seg_center.cy)

    nps_result = measure_nps_local(
        hu_slice, pixel_mm;
        roi_center = (center_row, center_col),
        roi_radius_mm = 30.0,
        roi_size = 32,
        n_rois = 256,
        unit = :lp_cm,
        overlap = 0.75,
    )

    mtf_result = measure_mtf_circular_edge(
        hu_slice, Float64(seg_center.cx), Float64(seg_center.cy), 16.5;
        fov_cm = fov_cm,
        margin_inner = 15.0,
        margin_outer = 5.0,
        fov_guard_pix = 3.0,
    )

    return (
        name = scan_name,
        rod_names = rod_names,
        rod_means = rod_means,
        rod_stds = rod_stds,
        rod_cnr = rod_cnr,
        nps = nps_result,
        mtf = mtf_result,
        nps_peak_freq = nps_result.peak_frequency,
        nps_area = nps_result.integrated_nps,
        mtf_f50 = mtf_result.mtf50,
        mtf_f10 = mtf_result.mtf10,
    )
end

# ╔═╡ 08030001-0000-4000-8000-000000000000
md"""
## 3. Clinical Data Loading

DICOM data from Siemens NAEOTOM Alpha (2026-03-13). Uncompressed UInt16 pixel data.
"""

# ╔═╡ 08030002-0000-4000-8000-000000000000
rootdir = "/Users/daleblack/Desktop/SCANS/13apr2026 (siemens alpha peak)"

# ╔═╡ 08040001-0000-4000-8000-000000000000
md"""
### Scan 1: 140 kVp / 52 mA / 3.03 mGy
"""

# ╔═╡ 08040002-0000-4000-8000-000000000000
begin  # 140 kVp / 52 mA / 3.03 mGy
    dcms_140_low_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_52mA_3.03mGyCTDI/Poly/0"))
    dcms_140_low_qir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_52mA_3.03mGyCTDI/Poly/3"))
    hu_140_low_fbp = load_hu_volume(dcms_140_low_fbp)
    hu_140_low_qir = load_hu_volume(dcms_140_low_qir)
    dcms_140_low_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_52mA_3.03mGyCTDI/VMI/3/40"))
    dcms_140_low_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_52mA_3.03mGyCTDI/VMI/3/70"))
    dcms_140_low_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_52mA_3.03mGyCTDI/VMI/3/100"))
    dcms_140_low_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_52mA_3.03mGyCTDI/VMI/3/140"))
    hu_140_low_vmi40 = load_hu_volume(dcms_140_low_vmi40)
    hu_140_low_vmi70 = load_hu_volume(dcms_140_low_vmi70)
    hu_140_low_vmi100 = load_hu_volume(dcms_140_low_vmi100)
    hu_140_low_vmi140 = load_hu_volume(dcms_140_low_vmi140)
end;

# ╔═╡ 08050001-0000-4000-8000-000000000000
md"""
### Scan 2: 140 kVp / 174 mA / 10.12 mGy
"""

# ╔═╡ 08050002-0000-4000-8000-000000000000
begin  # 140 kVp / 174 mA / 10.12 mGy
    dcms_140_mid_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/Poly/0"))
    dcms_140_mid_qir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/Poly/3"))
    hu_140_mid_fbp = load_hu_volume(dcms_140_mid_fbp)
    hu_140_mid_qir = load_hu_volume(dcms_140_mid_qir)
    dcms_140_mid_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/40"))
    dcms_140_mid_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/70"))
    dcms_140_mid_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/100"))
    dcms_140_mid_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/140"))
    hu_140_mid_vmi40 = load_hu_volume(dcms_140_mid_vmi40)
    hu_140_mid_vmi70 = load_hu_volume(dcms_140_mid_vmi70)
    hu_140_mid_vmi100 = load_hu_volume(dcms_140_mid_vmi100)
    hu_140_mid_vmi140 = load_hu_volume(dcms_140_mid_vmi140)
end;

# ╔═╡ 08060001-0000-4000-8000-000000000000
md"""
### Scan 3: 140 kVp / 347 mA / 20.25 mGy
"""

# ╔═╡ 08060002-0000-4000-8000-000000000000
begin  # 140 kVp / 347 mA / 20.25 mGy
    dcms_140_high_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/Poly/0"))
    dcms_140_high_qir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/Poly/3"))
    hu_140_high_fbp = load_hu_volume(dcms_140_high_fbp)
    hu_140_high_qir = load_hu_volume(dcms_140_high_qir)
    dcms_140_high_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI/3/40"))
    dcms_140_high_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI/3/70"))
    dcms_140_high_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI/3/100"))
    dcms_140_high_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI/3/140"))
    hu_140_high_vmi40 = load_hu_volume(dcms_140_high_vmi40)
    hu_140_high_vmi70 = load_hu_volume(dcms_140_high_vmi70)
    hu_140_high_vmi100 = load_hu_volume(dcms_140_high_vmi100)
    hu_140_high_vmi140 = load_hu_volume(dcms_140_high_vmi140)
end;

# ╔═╡ 08070001-0000-4000-8000-000000000000
md"""
### Scan 4: 120 kVp / 253 mA / 10.15 mGy
"""

# ╔═╡ 08070002-0000-4000-8000-000000000000
begin  # 120 kVp / 253 mA / 10.15 mGy
    dcms_120_mid_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/Poly/0"))
    dcms_120_mid_qir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/Poly/3"))
    hu_120_mid_fbp = load_hu_volume(dcms_120_mid_fbp)
    hu_120_mid_qir = load_hu_volume(dcms_120_mid_qir)
    dcms_120_mid_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI/3/40"))
    dcms_120_mid_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI/3/70"))
    dcms_120_mid_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI/3/100"))
    dcms_120_mid_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI/3/140"))
    hu_120_mid_vmi40 = load_hu_volume(dcms_120_mid_vmi40)
    hu_120_mid_vmi70 = load_hu_volume(dcms_120_mid_vmi70)
    hu_120_mid_vmi100 = load_hu_volume(dcms_120_mid_vmi100)
    hu_120_mid_vmi140 = load_hu_volume(dcms_120_mid_vmi140)
end;

# ╔═╡ 08070003-0000-4000-8000-000000000000
md"""
## 4. Clinical Segmentation
"""

# ╔═╡ 08070004-0000-4000-8000-000000000000
# Segment on the best-quality Poly QIR3 at 10 mGy
seg_result = let
    mid_z = size(hu_140_mid_qir, 3) ÷ 2
    hu_slice = hu_140_mid_qir[:, :, mid_z]
    mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm = 35.0)
    (mask = mask, rods = rod_info, center = center, slice_idx = mid_z)
end;

# ╔═╡ 08070005-0000-4000-8000-000000000000
# Segmentation overlay visualization
let
    mid_z = seg_result.slice_idx
    hu_slice = hu_140_mid_qir[:, :, mid_z]
    fig = CM.Figure(size = (500, 500), fontsize = 10)
    ax = CM.Axis(fig[1, 1]; title = "Segmentation — 140 kVp / 174 mA / QIR3", yreversed = true)
    CM.heatmap!(ax, hu_slice; colormap = :grays, colorrange = (-200, 500))
    for r in seg_result.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (35.0 / size(hu_slice, 1))
        CM.lines!(ax, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th); color = :red, linewidth = 1)
        CM.text!(ax, r.cx, r.cy - rpx - 2; text = r.name, fontsize = 6, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax); CM.hidespines!(ax)
    fig
end

# ╔═╡ 08070006-0000-4000-8000-000000000000
md"""
## 5. Clinical Measurements
"""

# ╔═╡ 08070007-0000-4000-8000-000000000000
clinical_measurements = let
    scans = [
        # Poly FBP
        (hu_140_low_fbp, "140kVp_52mA_FBP"),
        (hu_140_mid_fbp, "140kVp_174mA_FBP"),
        (hu_140_high_fbp, "140kVp_347mA_FBP"),
        (hu_120_mid_fbp, "120kVp_253mA_FBP"),
        # Poly QIR3
        (hu_140_low_qir, "140kVp_52mA_QIR3"),
        (hu_140_mid_qir, "140kVp_174mA_QIR3"),
        (hu_140_high_qir, "140kVp_347mA_QIR3"),
        (hu_120_mid_qir, "120kVp_253mA_QIR3"),
        # VMI at 10 mGy (140 kVp)
        (hu_140_mid_vmi40, "140kVp_174mA_VMI40"),
        (hu_140_mid_vmi70, "140kVp_174mA_VMI70"),
        (hu_140_mid_vmi100, "140kVp_174mA_VMI100"),
        (hu_140_mid_vmi140, "140kVp_174mA_VMI140"),
    ]
    [measure_scan(vol, seg_result.mask, seg_result.rods, seg_result.center, name)
     for (vol, name) in scans]
end;

# ╔═╡ 08070008-0000-4000-8000-000000000000
md"""
## 6. Clinical Qualitative — Poly FBP
"""

# ╔═╡ 08070009-0000-4000-8000-000000000000
# Poly FBP qualitative montage: 4 acquisitions
let
    vols = [hu_140_low_fbp, hu_140_mid_fbp, hu_140_high_fbp, hu_120_mid_fbp]
    labels = [
        "140 kVp / 52 mA / 3.03 mGy",
        "140 kVp / 174 mA / 10.12 mGy",
        "140 kVp / 347 mA / 20.25 mGy",
        "120 kVp / 253 mA / 10.15 mGy",
    ]
    mid_z = seg_result.slice_idx
    fig = CM.Figure(size = (900, 250), fontsize = 10)
    for (i, (vol, lbl)) in enumerate(zip(vols, labels))
        ax = CM.Axis(fig[1, i]; title = "FBP — $lbl", yreversed = true)
        CM.heatmap!(ax, vol[:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax); CM.hidespines!(ax)
    end
    CM.save(joinpath(RESULTS_DIR, "alpha_poly_fbp_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08070010-0000-4000-8000-000000000000
md"""
## 6b. Clinical Qualitative — Poly QIR3
"""

# ╔═╡ 08070011-0000-4000-8000-000000000000
# Poly QIR3 qualitative montage
let
    vols = [hu_140_low_qir, hu_140_mid_qir, hu_140_high_qir, hu_120_mid_qir]
    labels = [
        "140 kVp / 52 mA / 3.03 mGy",
        "140 kVp / 174 mA / 10.12 mGy",
        "140 kVp / 347 mA / 20.25 mGy",
        "120 kVp / 253 mA / 10.15 mGy",
    ]
    mid_z = seg_result.slice_idx
    fig = CM.Figure(size = (900, 250), fontsize = 10)
    for (i, (vol, lbl)) in enumerate(zip(vols, labels))
        ax = CM.Axis(fig[1, i]; title = "QIR3 — $lbl", yreversed = true)
        CM.heatmap!(ax, vol[:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax); CM.hidespines!(ax)
    end
    CM.save(joinpath(RESULTS_DIR, "alpha_poly_qir3_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08070012-0000-4000-8000-000000000000
md"""
## 6c. Clinical Qualitative — VMI (140 kVp / 10 mGy)
"""

# ╔═╡ 08070013-0000-4000-8000-000000000000
# VMI qualitative montage at 10 mGy: 40/70/100/140 keV
let
    vols = [hu_140_mid_vmi40, hu_140_mid_vmi70, hu_140_mid_vmi100, hu_140_mid_vmi140]
    energies = [40, 70, 100, 140]
    mid_z = seg_result.slice_idx
    fig = CM.Figure(size = (900, 250), fontsize = 10)
    for (i, (vol, E)) in enumerate(zip(vols, energies))
        ax = CM.Axis(fig[1, i]; title = "VMI $(E) keV (QIR3)", yreversed = true)
        CM.heatmap!(ax, vol[:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax); CM.hidespines!(ax)
    end
    CM.save(joinpath(RESULTS_DIR, "alpha_vmi_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08070014-0000-4000-8000-000000000000
md"""
## 6d. Clinical Noise — FBP vs QIR3 (Dose Ladder + kVp)
"""

# ╔═╡ 08070015-0000-4000-8000-000000000000
# Noise bar chart: FBP vs QIR3 across dose levels and kVp
let
    water_idx = 1  # "Water (O)" is first in rod_order

    # Dose ladder: 140 kVp at 3 / 10 / 20 mGy
    dose_labels = [
        "140 kVp / 52 mA\n(3.03 mGy)",
        "140 kVp / 174 mA\n(10.12 mGy)",
        "140 kVp / 347 mA\n(20.25 mGy)",
    ]
    dose_fbp_idx = [1, 2, 3]   # indices into clinical_measurements
    dose_qir_idx = [5, 6, 7]
    n_dose = 3

    dose_fbp_σ = [clinical_measurements[dose_fbp_idx[i]].rod_stds[water_idx] for i in 1:n_dose]
    dose_qir_σ = [clinical_measurements[dose_qir_idx[i]].rod_stds[water_idx] for i in 1:n_dose]

    # kVp comparison at ~10 mGy
    kvp_labels = [
        "140 kVp / 174 mA\n(10.12 mGy)",
        "120 kVp / 253 mA\n(10.15 mGy)",
    ]
    kvp_fbp_idx = [2, 4]
    kvp_qir_idx = [6, 8]
    n_kvp = 2

    kvp_fbp_σ = [clinical_measurements[kvp_fbp_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]
    kvp_qir_σ = [clinical_measurements[kvp_qir_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]

    fig = CM.Figure(size = (1000, 700), fontsize = 13)
    bw = 0.20

    # Top: Dose Ladder
    ax1 = CM.Axis(fig[1, 1]; title = "140 kVp — Dose Ladder", ylabel = "Water σ (HU)",
        xticks = (1:n_dose, dose_labels))
    x = collect(1:n_dose)
    CM.barplot!(ax1, x .- 0.12, dose_fbp_σ; width = bw, color = :steelblue, label = "Poly FBP")
    CM.barplot!(ax1, x .+ 0.12, dose_qir_σ; width = bw, color = :darkorange, label = "Poly QIR3")
    for (xi, v) in zip(x .- 0.12, dose_fbp_σ)
        CM.text!(ax1, xi, v + 0.5; text = "$(round(v, digits=1))", fontsize = 9, align = (:center, :bottom))
    end
    for (xi, v) in zip(x .+ 0.12, dose_qir_σ)
        CM.text!(ax1, xi, v + 0.5; text = "$(round(v, digits=1))", fontsize = 9, align = (:center, :bottom))
    end
    CM.ylims!(ax1, 0, nothing)
    CM.axislegend(ax1; position = :rt)

    # Bottom: kVp Series
    ax2 = CM.Axis(fig[2, 1]; title = "~10 mGy CTDIvol — kVp Series", ylabel = "Water σ (HU)",
        xticks = (1:n_kvp, kvp_labels))
    x2 = collect(1:n_kvp)
    CM.barplot!(ax2, x2 .- 0.12, kvp_fbp_σ; width = bw, color = :steelblue, label = "Poly FBP")
    CM.barplot!(ax2, x2 .+ 0.12, kvp_qir_σ; width = bw, color = :darkorange, label = "Poly QIR3")
    for (xi, v) in zip(x2 .- 0.12, kvp_fbp_σ)
        CM.text!(ax2, xi, v + 0.5; text = "$(round(v, digits=1))", fontsize = 9, align = (:center, :bottom))
    end
    for (xi, v) in zip(x2 .+ 0.12, kvp_qir_σ)
        CM.text!(ax2, xi, v + 0.5; text = "$(round(v, digits=1))", fontsize = 9, align = (:center, :bottom))
    end
    CM.ylims!(ax2, 0, nothing)
    CM.axislegend(ax2; position = :rt)

    CM.save(joinpath(RESULTS_DIR, "alpha_poly_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08070016-0000-4000-8000-000000000000
md"""
## 6e. Clinical Noise — VMI Energy Dependence (140 kVp / 10 mGy)
"""

# ╔═╡ 08070018-0000-4000-8000-000000000000
md"""
## 6f. Clinical HU Scatter — Poly FBP (140 kVp Dose Ladder)
"""

# ╔═╡ 08070019-0000-4000-8000-000000000000
# Scatter plot: HU accuracy across dose levels (FBP only, Ca + I rods)
let
    fbp_idx = [1, 2, 3]  # 140 kVp FBP at 3/10/20 mGy
    fbp_labels = ["52 mA (3.03 mGy)", "174 mA (10.12 mGy)", "347 mA (20.25 mGy)"]
    colors = [:dodgerblue, :orangered, :seagreen]

    # Reference: highest-dose FBP as "ground truth"
    ref = clinical_measurements[3]  # 20 mGy FBP

    fig = CM.Figure(size = (750, 900), fontsize = 11)

    # Top: Calcium rods
    ax_ca = CM.Axis(fig[1, 1]; title = "Calcium Rods — 140 kVp Dose Ladder (FBP)",
        xlabel = "Reference HU (20 mGy FBP)", ylabel = "HU")
    ca_names = [n for n in ref.rod_names if startswith(n, "Ca")]
    ca_idx = [findfirst(==(n), ref.rod_names) for n in ca_names]

    for (k, ci) in enumerate(fbp_idx)
        m = clinical_measurements[ci]
        CM.scatter!(ax_ca, ref.rod_means[ca_idx], m.rod_means[ca_idx];
            color = colors[k], markersize = 10, label = fbp_labels[k])
    end
    rng_ca = [minimum(ref.rod_means[ca_idx]) - 50, maximum(ref.rod_means[ca_idx]) + 50]
    CM.lines!(ax_ca, rng_ca, rng_ca; color = :gray60, linestyle = :dash, label = "Unity")
    CM.axislegend(ax_ca; position = :lt, labelsize = 9)

    # Bottom: Iodine rods
    ax_i = CM.Axis(fig[2, 1]; title = "Iodine Rods — 140 kVp Dose Ladder (FBP)",
        xlabel = "Reference HU (20 mGy FBP)", ylabel = "HU")
    i_names = [n for n in ref.rod_names if startswith(n, "I ")]
    i_idx = [findfirst(==(n), ref.rod_names) for n in i_names]

    for (k, ci) in enumerate(fbp_idx)
        m = clinical_measurements[ci]
        CM.scatter!(ax_i, ref.rod_means[i_idx], m.rod_means[i_idx];
            color = colors[k], markersize = 10, label = fbp_labels[k])
    end
    rng_i = [minimum(ref.rod_means[i_idx]) - 20, maximum(ref.rod_means[i_idx]) + 20]
    CM.lines!(ax_i, rng_i, rng_i; color = :gray60, linestyle = :dash, label = "Unity")
    CM.axislegend(ax_i; position = :lt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "alpha_poly_scatter_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08080001-0000-4000-8000-000000000000
md"""
## 8. Digital Phantom — Gammex 472

Custom phantom with labeled material inserts for quantitative validation.
"""

# ╔═╡ 08080002-0000-4000-8000-000000000000
begin
    GAMMEX_SOLID_WATER = XA.Materials.gammex_472_solidwater

    ALL_INSERT_MATERIALS = Dict{UInt8, XA.Material}(
        UInt8(10) => XA.Materials.gammex_472_ca50_0,
        UInt8(11) => XA.Materials.gammex_472_ca100_0,
        UInt8(12) => XA.Materials.gammex_472_ca200_0,
        UInt8(13) => XA.Materials.gammex_472_ca300_0,
        UInt8(14) => XA.Materials.gammex_472_ca400_0,
        UInt8(20) => XA.Materials.gammex_472_i2_0,
        UInt8(21) => XA.Materials.gammex_472_i2_5,
        UInt8(22) => XA.Materials.gammex_472_i5_0,
        UInt8(23) => XA.Materials.gammex_472_i7_5,
        UInt8(24) => XA.Materials.gammex_472_i10_0,
        UInt8(25) => XA.Materials.gammex_472_i15_0,
        UInt8(26) => XA.Materials.gammex_472_i20_0,
    )

    MATERIAL_INFO = Dict(
        UInt8(0) => (name = "Air", color = :gray15),
        UInt8(1) => (name = "Solid Water", color = :lightskyblue),
        UInt8(2) => (name = "Pure Water", color = :royalblue),
        UInt8(3) => (name = "SW Reference", color = :paleturquoise),
        UInt8(10) => (name = "Ca 50 mg/mL", color = :wheat),
        UInt8(11) => (name = "Ca 100 mg/mL", color = :sandybrown),
        UInt8(12) => (name = "Ca 200 mg/mL", color = :orange),
        UInt8(13) => (name = "Ca 300 mg/mL", color = :darkorange),
        UInt8(14) => (name = "Ca 400 mg/mL", color = :orangered),
        UInt8(20) => (name = "I 2.0 mg/mL", color = :honeydew),
        UInt8(21) => (name = "I 2.5 mg/mL", color = :palegreen),
        UInt8(22) => (name = "I 5.0 mg/mL", color = :lightgreen),
        UInt8(23) => (name = "I 7.5 mg/mL", color = :mediumseagreen),
        UInt8(24) => (name = "I 10.0 mg/mL", color = :seagreen),
        UInt8(25) => (name = "I 15.0 mg/mL", color = :forestgreen),
        UInt8(26) => (name = "I 20.0 mg/mL", color = :darkgreen),
    )
end;

# ╔═╡ 08080003-0000-4000-8000-000000000000
function create_custom_gammex_472(;
        n_voxels::Int = 1750,
        n_slices::Int = 250,
        fov_cm::Float64 = 35.0,
        z_cm::Float64 = 5.0,
    )
    dx = fov_cm / n_voxels
    dy = fov_cm / n_voxels
    dz = z_cm / n_slices

    x = range(-fov_cm / 2 + dx / 2, fov_cm / 2 - dx / 2, length = n_voxels)
    y = range(-fov_cm / 2 + dy / 2, fov_cm / 2 - dy / 2, length = n_voxels)

    body_radius = 16.5
    rod_radius = 1.4
    air_gap_water = 0.03
    air_gap_other = 0.01
    rod_radius_sq = rod_radius^2
    outer_ring_R = 10.5
    inner_ring_R = 5.5

    outer_start = pi / 2 - pi / 8
    outer_angles = [outer_start - (i - 1) * pi / 4 for i in 1:8]
    outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]

    # WARNING: I 15.0 (label 25) and I 20.0 (label 26) are SWAPPED relative to the
    # standard Gammex 472 layout. Clinical HU measurements on the UCI NAEOTOM Alpha
    # confirm the physical rods were inserted in swapped positions. The digital phantom
    # matches the actual clinical configuration, NOT the nominal datasheet order.
    inner_start = pi / 2
    inner_angles = [inner_start - (i - 1) * pi / 4 for i in 1:8]
    inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]

    outer_hole_r_sq = [(rod_radius + (l == UInt8(2) ? air_gap_water : air_gap_other))^2 for l in outer_labels]
    inner_hole_r_sq = [(rod_radius + (l == UInt8(2) ? air_gap_water : air_gap_other))^2 for l in inner_labels]

    outer_cx = [outer_ring_R * cos(a) for a in outer_angles]
    outer_cy = [outer_ring_R * sin(a) for a in outer_angles]
    inner_cx = [inner_ring_R * cos(a) for a in inner_angles]
    inner_cy = [inner_ring_R * sin(a) for a in inner_angles]

    slice = zeros(UInt8, n_voxels, n_voxels)

    for j in 1:n_voxels, i in 1:n_voxels
        xi, yj = x[i], y[j]

        if xi^2 + yj^2 <= body_radius^2
            slice[i, j] = UInt8(1)

            for idx in 1:8
                d_sq = (xi - outer_cx[idx])^2 + (yj - outer_cy[idx])^2
                if d_sq <= outer_hole_r_sq[idx]
                    slice[i, j] = d_sq <= rod_radius_sq ? outer_labels[idx] : UInt8(0)
                    @goto next_voxel
                end
            end

            for idx in 1:8
                d_sq = (xi - inner_cx[idx])^2 + (yj - inner_cy[idx])^2
                if d_sq <= inner_hole_r_sq[idx]
                    slice[i, j] = d_sq <= rod_radius_sq ? inner_labels[idx] : UInt8(0)
                    break
                end
            end
        end
        @label next_voxel
    end

    slice = reverse(permutedims(rot180(slice)), dims = 2)

    mask = Array{UInt8, 3}(undef, n_voxels, n_voxels, n_slices)
    @views for k in 1:n_slices
        mask[:, :, k] .= slice
    end

    max_label = 26
    materials_vec = Vector{XA.Material}(undef, max_label + 1)
    fill!(materials_vec, XA.Materials.air)
    materials_vec[1] = XA.Materials.air
    materials_vec[2] = GAMMEX_SOLID_WATER
    materials_vec[3] = XA.Materials.water
    materials_vec[4] = GAMMEX_SOLID_WATER

    for (lbl, mat) in ALL_INSERT_MATERIALS
        materials_vec[Int(lbl) + 1] = mat
    end

    origin = (-fov_cm / 2 + dx / 2, -fov_cm / 2 + dy / 2, -z_cm / 2 + dz / 2)
    extent = (Float64(fov_cm), Float64(fov_cm), Float64(z_cm))

    return BS.Phantom(mask, materials_vec, (dx, dy, dz), origin, extent)
end

# ╔═╡ 08080004-0000-4000-8000-000000000000
begin
    # Phantom extent (45 cm) > recon FOV (35 cm) so the forward projector sees air
    # voxels at all angles beyond the phantom body (33 cm diameter).
    # Clinical data collection diameter is 50.4 cm (table/air visible in recon).
    # ~0.2 mm voxels (matching nb06 resolution) for proper partial-volume edges.
    sim_phantom_cpu = create_custom_gammex_472(
        n_voxels = 2250,
        n_slices = 250,
        fov_cm = 45.0,
        z_cm = 5.0,
    )
  
    sim_phantom_gpu = BS.Phantom(
        MtlArray(sim_phantom_cpu.mask),
        sim_phantom_cpu.materials,
        sim_phantom_cpu.voxel_size,
        sim_phantom_cpu.origin,
        sim_phantom_cpu.extent,
    )
end

# ╔═╡ 08080005-0000-4000-8000-000000000000
let
    fig = CM.Figure(size = (800, 800), fontsize = 14)
    ax = CM.Axis(fig[1, 1], title = "Gammex 472 Digital Phantom (Central Slice)", aspect = CM.DataAspect())
    CM.heatmap!(ax, Array(sim_phantom_gpu.mask[:, :, size(sim_phantom_gpu.mask, 3) ÷ 2]), colormap = :viridis)
    # CM.hidedecorations!(ax)
    fig
end

# ╔═╡ 08090001-0000-4000-8000-000000000000
md"""
## 9. Simulation Parameters

**Scanner:** NAEOTOM Alpha (standard mode, 2×2 binned from native 0.275×0.322 mm dexels)

**Energy thresholds:** 4-threshold clinical configuration (T1=20, T2=35, T3=55, T4=70 keV):
- Bin 1: 20–35 keV | Bin 2: 35–55 keV | Bin 3: 55–70 keV | Bin 4: >70 keV
"""

# ╔═╡ 08090002-0000-4000-8000-000000000000
# NAEOTOM Alpha — standard mode with 2-threshold clinical configuration
#
# Binning pipeline: native 0.275×0.322 mm dexels at detector face → 2×2 binned in DAS
# → pcct_forward_project ray-traces at NATIVE resolution, applies detector physics
#   (charge sharing, pileup, anti-coincidence) at native res, then spatial_bin!()
#   sums bf×bf dexels → binned pixels.
# detector_col_size / detector_row_size = binned pixel at ISOCENTER (not the 0.4mm clinical spec)
sim_scanner = let
    # Physical constants
    native_col_mm = 0.275    # at detector face (Konrad 2025)
    native_row_mm = 0.322    # at detector face (Konrad 2025)
    sid = 610.0              # source-to-isocenter (mm)
    sdd = 1113.0             # source-to-detector (mm)
    magnification = sdd / sid  # ~1.824
    bf = 2                   # standard mode: 2×2 binning

    # Binned pixel size at isocenter (geometric projection)
    pixel_col_iso = (native_col_mm * bf) / magnification  # ~0.302 mm
    pixel_row_iso = (native_row_mm * bf) / magnification  # ~0.353 mm
    n_cols = ceil(Int, 500.0 / pixel_col_iso)             # for 50cm scan diameter

    BS.Scanner(
        # GEOMETRY (Konrad 2025, FDA K201501)
        source_to_isocenter = sid,
        source_to_detector = sdd,

        # DETECTOR ARRAY (standard mode: 2×2 binned from native dexels)
        detector_rows = 144,
        detector_cols = n_cols,
        detector_row_size = pixel_row_iso,           # ~0.353 mm at isocenter
        detector_col_size = pixel_col_iso,           # ~0.302 mm at isocenter
        detector_shape = BS.CURVED_DETECTOR,
        detector_row_offset = 0.0,
        detector_col_offset = pixel_col_iso / 2,     # quarter-detector offset

        # X-RAY SOURCE
        focal_spot_width = 0.4,     # mm
        focal_spot_length = 0.5,    # mm
        target_angle = 7.0,         # degrees

        # GANTRY
        gantry_rotation_time = 0.5,   # matches clinical protocol
        scan_diameter = 500.0,
        gantry_aperture = 820.0,

        # FILTRATION — 3 mm Al + 0.9 mm Ti (Siemens Vectron tube, all Straton/Vectron systems)
        # Al is scanner flat filter; Ti added via protocol additional_filters
        flat_filter_material = :aluminum,
        flat_filter_thickness = 3.0,  # mm

        # DETECTOR PHYSICS — CdTe direct-conversion
        detector_material = :cdte,
        detector_depth = 1.6,         # mm (Konrad 2025)
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        detection_gain = 1.0,         # direct conversion (no scintillator gain)
        electronic_noise = 0.0,       # PCCT thresholds eliminate electronic noise

        # PCCT-SPECIFIC — 4 thresholds matching clinical protocol
        detector_type = :photon_counting,
        n_energy_bins = 4,
        energy_thresholds = [20.0, 35.0, 55.0, 70.0],  # 4-threshold clinical config
        energy_resolution = 10.0,              # keV FWHM (superseded by DRM at :eict)
        charge_sharing_fwhm = 0.08,            # mm (superseded by Koch-Mehrin at :eict)
        dead_time_ns = 5.0,                    # ns (Yang 2025 pileup)
        pixel_mode = :standard,

        # Native dexel parameters — drives native-res ray tracing + spatial binning
        native_dexel_col_mm = native_col_mm,
        native_dexel_row_mm = native_row_mm,
        binning_factor = bf,
    )
end

# ╔═╡ 08090003-0000-4000-8000-000000000000
md"""
### Scanner Configuration
- **Detector**: CdTe $(sim_scanner.detector_depth) mm | $(sim_scanner.detector_rows) × $(sim_scanner.detector_cols) at $(sim_scanner.detector_col_size) mm pitch
- **Energy bins**: $(sim_scanner.n_energy_bins) (thresholds: $(sim_scanner.energy_thresholds) keV)
- **SID**: $(sim_scanner.source_to_isocenter) mm | **SDD**: $(sim_scanner.source_to_detector) mm
- **Binning**: $(sim_scanner.binning_factor)×$(sim_scanner.binning_factor) (native $(sim_scanner.native_dexel_col_mm)×$(sim_scanner.native_dexel_row_mm) mm)
"""

# ╔═╡ 08090004-0000-4000-8000-000000000000
begin
    # Common protocol parameters
    sim_rotation_time = 0.5       # seconds (matches clinical)
    sim_collimation_mm = 5.0      # ~14 rows at iso — fast dev mode (real clinical: 144×0.4mm ≈ 57.6mm)
    sim_n_views = 1200            # single-tube equivalent; real Alpha.Peak has ~2400 effective (dual-source)
    sim_fov_cm = 35.0             # reconstruction FOV

    # Reconstruction — z-extent matched to collimation
    sim_slice_thickness_mm = 0.4    # native PCCT detector element (standard mode)
    sim_n_recon_slices = round(Int, sim_collimation_mm / sim_slice_thickness_mm)
    sim_matrix_size = (512, 512, sim_n_recon_slices)
    sim_recon_z_cm = sim_collimation_mm / 10.0

    # Dose levels — mA values are estimates, tune to match clinical CTDIvol
    sim_mA_scan1 = 50.0    # ~3 mGy target
    sim_mA_scan2 = 174.0   # exact clinical (10.12 mGy, 140 kVp)
    sim_mA_scan3 = 400.0   # ~20 mGy target
    sim_mA_scan4 = 300.0   # ~10 mGy target (120 kVp, higher mA to compensate)
end

# ╔═╡ 08090005-0000-4000-8000-000000000000
sim_opts = BS.SimOptions(
  fidelity = :pcct,
  seed = 1234,
  
  pcct_noise_reduction = 0.3,  # DAS corrections: anti-coincidence, gain cal, pixel interpolation
)

# ╔═╡ 08090006-0000-4000-8000-000000000000
sim_recon_opts = BS.ReconOptions(
    algorithm = :fdk,
    matrix_size = sim_matrix_size,
    fov_cm = sim_fov_cm,
    filter = :standard,
    vmi_energies = [40.0, 70.0, 100.0, 140.0],
    vmi_basis = [:water, :iodine, :calcium],
)

# ╔═╡ 08090007-0000-4000-8000-000000000000
# Br36 kernel approximation — medium-sharp body kernel (Siemens)
# Control points: (normalized_freq, amplitude) — tune to match clinical Br36
sim_custom_filter = BS.CustomFilter(
  (0.0, 0.25, 0.5, 0.75, 1.0),
  (1.0, 0.75, 0.6, 0.2, 0.001),
  # (1.0, 0.70, 0.08, 0.002, 0.0001),
)

# ╔═╡ 08090007-b000-4000-8000-000000000001
# VMI uses the same filter as poly FBP — noise is handled upstream via
# pcct_noise_reduction (DAS correction model), not via softer reconstruction kernel.
sim_vmi_filter = sim_custom_filter

# ╔═╡ 08090008-0000-4000-8000-000000000000
# System noise floor (dose-independent, applied post-reconstruction)
sim_noise_floor_hu = 0.0  # HU — disabled for debugging (was 15.0)

# ╔═╡ 08100001-0000-4000-8000-000000000000
md"""
## 10. Water Attenuation Calibration

Analytical μ\_water from spectrum — no simulation needed.

1. `resolve_source_spectrum_without_bowtie` → filtered spectrum (energies `e`, weights `w`)
2. Mean energy → `compute_μ_at_energy(water, mean_E)` for each bin's energy range
3. Per-bin: window spectrum to bin boundaries, compute bin-specific mean energy
"""

# ╔═╡ 08100002-0000-4000-8000-000000000000
# Analytical μ_water at 140 kVp (combined spectrum)
μ_water_140 = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e, w = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
    mean_E = sum(e .* w) / sum(w)
    mu = BS.compute_μ_at_energy(XA.Materials.water, mean_E)
    @info "140 kVp μ_water: $(round(mu, digits = 5)) (E̅=$(round(mean_E, digits = 1)) keV)"
    mu
end

# ╔═╡ 08100003-0000-4000-8000-000000000000
md"""
**Water attenuation (140 kVp, analytical):** μ\_water = $(round(μ_water_140, sigdigits=4)) cm⁻¹
"""

# ╔═╡ 08100004-0000-4000-8000-000000000000
# Analytical μ_water at 120 kVp (for Scan 4)
# μ_water_120 = let
#     prot = BS.CTProtocol(kVp = 120.0)
#     e, w = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
#     mean_E = sum(e .* w) / sum(w)
#     mu = BS.compute_μ_at_energy(XA.Materials.water, mean_E)
#     @info "120 kVp μ_water: $(round(mu, digits = 5)) (E̅=$(round(mean_E, digits = 1)) keV)"
#     mu
# end

# ╔═╡ 08100005-0000-4000-8000-000000000000
# md"**Water attenuation (120 kVp, analytical):** μ\_water = $(round(μ_water_120, sigdigits=4)) cm⁻¹"

# ╔═╡ 08130001-0000-4000-8000-000000000000
md"""
## Scan 1: 140 kVp / ~3 mGy

Placeholder — same pipeline as Scan 2 with lower mA.
"""

# ╔═╡ 08130002-0000-4000-8000-000000000000
sim_scan1 = nothing  # TODO: implement (same as Scan 2 with mA = sim_mA_scan1)

# ╔═╡ 08130003-a000-4000-8000-000000000001
md"### Poly"

# ╔═╡ 08130003-0000-4000-8000-000000000000
begin
    sim_scan1_poly_fbp = nothing  # TODO: poly FBP recon
    sim_scan1_poly_hir = nothing  # TODO: poly HIR recon
end

# ╔═╡ 08130003-b000-4000-8000-000000000001
md"### VMI"

# ╔═╡ 08130003-b000-4000-8000-000000000002
# TODO: VMI pipeline for Scan 1 (same structure as Scan 2).
nothing

# ╔═╡ 08110001-0000-4000-8000-000000000000
md"""
## Scan 2: 140 kVp / ~10 mGy

Full PCCT simulation with energy-resolved sinograms and material decomposition.
"""

# ╔═╡ 08110002-0000-4000-8000-000000000000
sim_scan2 = let
    prot = BS.CTProtocol(
        kVp = 140.0,
        mA = sim_mA_scan2,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = [("Ti", 0.9)],  # 0.9 mm Ti (Vectron tube inherent)
    )

    ws = BS.create_workspace(sim_scanner, prot, sim_opts, sim_recon_opts, sim_phantom_gpu)
    result = BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_opts)

    geom = ws.geom
    pcct_sino = result.pcct_sino
    I0_bins_cpu = copy(result.I0_bins)
    combined_cpu = copy(result.combined)
    bins_cpu = [Array(pcct_sino.bins[i]) for i in 1:length(pcct_sino.bins)]
    result_scatter_field = result.scatter_field
    result_scatter_bin_weights = result.scatter_bin_weights

    # Cleanup GPU
    ws = nothing; result = nothing; GC.gc(true)

    (
        bins = bins_cpu,
        I0_bins = I0_bins_cpu,
        combined = combined_cpu,
        geom = geom,
        scatter_field = result_scatter_field,
        scatter_bin_weights = result_scatter_bin_weights,
    )
end;

# ╔═╡ 08120002-b000-4000-8000-000000000001
# Diagnostic: I0 values and per-bin sinogram stats
let
    I0 = sim_scan2.I0_bins
    bins = sim_scan2.bins

    println("I0 per bin: ", round.(I0, sigdigits=4))
    println("I0 total: ", round(sum(I0), sigdigits=4))
    println()

    # Per-bin sinogram stats
    for (b, s) in enumerate(bins)
        println("Bin $b: min=$(round(minimum(s), digits=4)) max=$(round(maximum(s), digits=4)) mean=$(round(mean(s), digits=4)) I0=$(round(I0[b], sigdigits=4))")
    end
end

# ╔═╡ 08120001-0000-4000-8000-000000000000
md"""
### Poly
"""

# ╔═╡ 08120002-0000-4000-8000-000000000000
# Poly FBP — scatter correction (decoupled) + FDK reconstruction
# Combined sinogram from simulate! has scatter but no correction applied.
# Exact model subtraction using returned scatter_field + bin_weights.
sim_scan2_poly_fbp = let
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    # Combined sinogram from simulate! has scatter but is NOT corrected
    # (scatter correction is decoupled). Apply exact model subtraction using
    # the returned scatter artifacts before poly FBP reconstruction.
    combined = copy(sim_scan2.combined)
    if sim_scan2.scatter_field !== nothing
        sf = sim_scan2.scatter_field
        bw = sim_scan2.scatter_bin_weights
        I0 = sim_scan2.I0_bins
        I0_total = sum(I0)
        # Total scatter weight for combined sinogram = Σ(I0_b × frac_b) / I0_total
        total_weight = sum(I0[b] * bw[b] for b in eachindex(bw)) / I0_total
        eps = Float32(1e-10)
        for idx in eachindex(combined)
            intensity = exp(-combined[idx])
            scatter = sf[idx] * Float32(total_weight)
            combined[idx] = -log(max(intensity - scatter, eps))
        end
    end

    sino_gpu = MtlArray(combined)
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = sim_custom_filter
    )
    recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_fdk = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end;

# ╔═╡ 08120003-0000-4000-8000-000000000000
md"**Polyenergetic HIR (strength=3)**"

# ╔═╡ 08120004-0000-4000-8000-000000000000
# TODO: Poly HIR — uncomment when ready for HIR comparison
# sim_scan2_poly_hir = let
#     sino_gpu = MtlArray(sim_scan2.combined)
#     geom = sim_scan2.geom
#     recon_size = sim_matrix_size
#     ws_hir = BS.create_hir_recon_workspace(
#         sino_gpu, geom, recon_size;
#         strength = 3, filter = sim_custom_filter
#     )
#     BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
#     recon_μ = Array(ws_hir.volume)
#     recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
#     BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
#     ws_hir = nothing; sino_gpu = nothing; GC.gc(true)
#     recon_hu
# end

# ╔═╡ 08120004-a000-4000-8000-000000000001
md"**Clinical vs Simulated Poly FBP**"

# ╔═╡ 08120004-a000-4000-8000-000000000002
# Side-by-side: clinical FBP vs simulated poly FBP — soft tissue / bone / lung windows
let
    mid_z = sim_n_recon_slices ÷ 2
    clin_slice = hu_140_mid_fbp[:, :, seg_result.slice_idx]
    sim_slice = sim_scan2_poly_fbp[:, :, mid_z]

    windows = [
        ("Soft Tissue", (-200, 500)),
        ("Bone",        (-500, 1500)),
        ("Lung",        (-1200, 200)),
    ]

    fig = CM.Figure(size = (1200, 1500), fontsize = 14)
    # CM.Label(fig[0, :], text = "Scan 2 (140 kVp / 10 mGy): Clinical vs Simulated", fontsize = 16, font = :bold)

    for (row, (wname, wrange)) in enumerate(windows)
        # Clinical
        ax1 = CM.Axis(fig[row, 1];
            title = row == 1 ? "Clinical FBP" : "",
            subtitle = row == 1 ? "140 kVp, 174 mA, Br44f" : "",
            yreversed = true
         )
        CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = wrange)
        CM.hidedecorations!(ax1; label = false)
        CM.hidespines!(ax1)

        # Simulated
        ax2 = CM.Axis(fig[row, 2];
            title = row == 1 ? "Simulated Poly FBP" : "",
            subtitle = row == 1 ? "140 kVp, 174 mA" : "",)
        hm = CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = wrange)
        CM.hidedecorations!(ax2)
        CM.hidespines!(ax2)

        CM.Colorbar(fig[row, 3], hm; label = "HU", width = 12)
    end

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_clinical_vs_sim.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08120004-b000-4000-8000-000000000001
md"**MTF Measurement Diagnostic**"

# ╔═╡ 08120004-b000-4000-8000-000000000003
md"**NPS Measurement Diagnostic**"

# ╔═╡ 08120005-a000-4000-8000-000000000001
md"""
### VMI

**Per-bin scatter correction (decoupled).** `simulate!` injects scatter into
per-bin sinograms (for correct Poisson noise statistics) and returns the
scatter field + per-bin weights for exact model-based subtraction.  Pipeline
mirrors the injection in reverse: reconstruct combined → `estimate_scatter_field!`
(Ohnesorge et al., Eur Radiol 1999) → `compute_scatter_bin_weights` via DRM
→ subtract per-bin scatter counts.
"""

# ╔═╡ 08120005-a000-4000-8000-000000000002
# Per-bin scatter correction: estimate from combined, subtract from each bin
sim_scan2_bins_corrected = let
    bins_raw = sim_scan2.bins
    I0_bins = sim_scan2.I0_bins
    I0_total = Float32(sum(I0_bins))
    eps = Float32(1e-10)

    # Step 1: Reconstruct combined primary sinogram from bins
    # (same math as driver.jl lines 143-157)
    combined = zeros(Float32, size(bins_raw[1]))
    for (b, bin_sino) in enumerate(bins_raw)
        I0b = Float32(I0_bins[b])
        @. combined += I0b * exp(-bin_sino)
    end
    @. combined = -log(max(combined, eps) / I0_total)

    # Step 2: Estimate scatter field from combined signal
    # (same model as simulate! uses internally)
    voxel_size_mm = sim_phantom_cpu.voxel_size .* 10.0
    phantom_diam_cm = BS.estimate_phantom_diameter_cm(sim_phantom_cpu.mask, voxel_size_mm)
    scatter_model = BS.geometry_aware_scatter_model(sim_scanner; phantom_diameter_cm=phantom_diam_cm)

    scatter_field = similar(combined)
    BS.estimate_scatter_field!(scatter_field, combined, scatter_model)

    @info "Scatter field: mean=$(round(mean(scatter_field), sigdigits=3)), max=$(round(maximum(scatter_field), sigdigits=3))"

    # Step 3: Per-energy scatter weights → per-bin via DRM (unified API)
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
    pcct_det = BS._build_pcct_detector(sim_scanner)
    kVp_val = Float64(maximum(e_full))
    R_mat = BS.compute_mc_drm(pcct_det, kVp_val)
    η_vec = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
    ew = BS.compute_scatter_energy_weights(Float64.(e_full))
    scatter_fracs = BS.compute_scatter_bin_weights(
        Float64.(e_full), Float64.(w_full), ew, Float64.(η_vec), R_mat, kVp_val)

    @info "Scatter bin fractions: $(round.(scatter_fracs, digits=3))"

    # Step 4: Subtract per-bin scatter (exact inverse of injection in driver.jl lines 167-177)
    bins_corrected = [copy(Float32.(b)) for b in bins_raw]
    for (b, bin_sino) in enumerate(bins_corrected)
        I0b = Float32(I0_bins[b])
        frac = Float32(scatter_fracs[b])
        for idx in eachindex(bin_sino)
            N_measured = I0b * exp(-bin_sino[idx])
            N_scatter = scatter_field[idx] * I0_total * frac
            N_corrected = N_measured - max(N_scatter, Float32(0))
            bin_sino[idx] = -log(max(N_corrected, eps) / I0b)
        end
    end

    # Diagnostic: compare per-bin scatter magnitude
    for b in 1:length(bins_raw)
        Δ = mean(Float64.(bins_corrected[b]) .- Float64.(bins_raw[b]))
        @info "Bin $b scatter correction: Δmean_p=$(round(Δ, sigdigits=3)), frac=$(round(scatter_fracs[b], digits=3))"
    end

    bins_corrected
end;

# ╔═╡ 08120005-0000-4000-8000-000000000000
md"""
**Low/high bin combination.** Combine 4 PCCT threshold bins into 2 effective
sinograms for VMI: low = bins 1+2 (20–55 keV, photoelectric / iodine); high =
bins 3+4 (>55 keV, Compton).  Count-domain combine:
`sino = -log( (I0_a·exp(-sino_a) + I0_b·exp(-sino_b)) / (I0_a + I0_b) )`.
"""

# ╔═╡ 08120006-0000-4000-8000-000000000000
# Combine bins → low (20–55 keV) and high (>55 keV) sinograms
sim_scan2_lohi = let
    bins = sim_scan2_bins_corrected   # ← use scatter-corrected bins
    I0 = sim_scan2.I0_bins

    function combine_bins(bin_indices, bins, I0)
        I0_sum = sum(I0[b] for b in bin_indices)
        counts = zeros(Float32, size(bins[1]))
        for b in bin_indices
            @. counts += Float32(I0[b]) * exp(-bins[b])
        end
        @. -log(max(counts, Float32(1e-10)) / Float32(I0_sum))
    end

    sino_low  = combine_bins([1, 2], bins, I0)   # 20–55 keV
    sino_high = combine_bins([3, 4], bins, I0)   # >55 keV
    I0_low  = I0[1] + I0[2]
    I0_high = I0[3] + I0[4]

    @info "Low bin (20–55 keV): I0=$(round(I0_low, sigdigits=4)), mean sino=$(round(mean(sino_low), digits=3))"
    @info "High bin (>55 keV):  I0=$(round(I0_high, sigdigits=4)), mean sino=$(round(mean(sino_high), digits=3))"

    (sino_low = sino_low, sino_high = sino_high, I0_low = I0_low, I0_high = I0_high)
end;

# ╔═╡ 08120006-b000-4000-8000-000000000001
# Diagnostic: check for negative line integrals (polynomial blow-up source)
let
    sl = sim_scan2_lohi.sino_low
    sh = sim_scan2_lohi.sino_high
    n_neg_low = count(x -> x < 0, sl)
    n_neg_high = count(x -> x < 0, sh)
    n_total = length(sl)
    min_low = minimum(sl); min_high = minimum(sh)
    @info "sino_low:  $(n_neg_low)/$(n_total) negative ($(round(100*n_neg_low/n_total, digits=2))%), min=$(round(min_low, digits=4))"
    @info "sino_high: $(n_neg_high)/$(n_total) negative ($(round(100*n_neg_high/n_total, digits=2))%), min=$(round(min_high, digits=4))"
end

# ╔═╡ 08120009-0000-4000-8000-000000000000
md"""
**Material decomposition.** Chain: **CMV (poly) → RWLS-GN 3-bin → PWLS-L₂
2-bin** → single `(sino_w, sino_I)` pair.  Each stage has a `use_*` toggle +
hyperparam cell; off → pass-through.  Intermediate viz after each stage shows
iodine/water sinograms + quick FBP.

- **CMV (poly)** — `pcct_vmi_calibration` applied per ray.  Fast, noisy,
  correct HU.  Canonical RWLS warm start.
- **RWLS-GN** (Ducros 2017) — 3-bin (A=1+2, B=3, C=4) Gauss-Newton with
  Poisson weights + FFT quadratic prior.  Single α, β_w, β_I.
- **PWLS-L₂** (Long/Fessler 2014 §IV-B) — 2-bin 2×2 matrix-curvature polish
  on the RWLS output, captures iodine↔water anti-correlated noise.

ACNR (Kalender 1988) is per-energy so it lives in the VMI synthesis block
below, not here.
"""

# ╔═╡ 08120e00-0000-4000-8000-000000000001
# Shared viz helpers used by the per-stage intermediate viz cells below.
# _decomp_viz   — 2×2 heatmap: (∫ρ_I·dr, ∫ρ_W·dr) sinograms + (ρ_I, ρ_W) FBP.
# _vmi_row_viz  — 1×4 row: VMI HU at each target energy for one Dict of volumes.
function _decomp_viz(sino_I, sino_W, stage_name)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size
    function _fbp(s)
        g = MtlArray(Float32.(s))
        ws = BS.create_fdk_recon_workspace(g, geom, recon_size; filter = sim_vmi_filter)
        v = Array(BS.reconstruct!(ws, g, geom, recon_size))
        ws = nothing; g = nothing; GC.gc(true)
        Float32.(v)
    end
    fbp_I = _fbp(sino_I);  fbp_W = _fbp(sino_W)
    mid_v = size(sino_I, 2) ÷ 2
    mid_z = size(fbp_I, 3) ÷ 2

    I_lo, I_hi = quantile(vec(sino_I[:, mid_v, :]), 0.01), quantile(vec(sino_I[:, mid_v, :]), 0.995)
    W_lo, W_hi = quantile(vec(sino_W[:, mid_v, :]), 0.01), quantile(vec(sino_W[:, mid_v, :]), 0.995)
    fI_lo, fI_hi = quantile(vec(fbp_I[:, :, mid_z]), 0.01), quantile(vec(fbp_I[:, :, mid_z]), 0.995)
    fW_lo, fW_hi = quantile(vec(fbp_W[:, :, mid_z]), 0.01), quantile(vec(fbp_W[:, :, mid_z]), 0.995)

    fig = CM.Figure(size = (1100, 700), fontsize = 11)
    ax1 = CM.Axis(fig[1, 1]; title = "$stage_name — ∫ρ_I·dr (sino mid-view)", aspect = CM.DataAspect())
    CM.heatmap!(ax1, sino_I[:, mid_v, :]; colormap = :viridis, colorrange = (I_lo, I_hi))
    ax2 = CM.Axis(fig[1, 2]; title = "$stage_name — ∫ρ_W·dr (sino mid-view)", aspect = CM.DataAspect())
    CM.heatmap!(ax2, sino_W[:, mid_v, :]; colormap = :viridis, colorrange = (W_lo, W_hi))
    ax3 = CM.Axis(fig[2, 1]; title = "$stage_name — ρ_I FBP (slice $mid_z) [g/cm³]", aspect = CM.DataAspect())
    CM.heatmap!(ax3, fbp_I[:, :, mid_z]; colormap = :viridis, colorrange = (fI_lo, fI_hi))
    ax4 = CM.Axis(fig[2, 2]; title = "$stage_name — ρ_W FBP (slice $mid_z) [g/cm³]", aspect = CM.DataAspect())
    CM.heatmap!(ax4, fbp_W[:, :, mid_z]; colormap = :viridis, colorrange = (fW_lo, fW_hi))
    fig
end

function _vmi_row_viz(vmi_dict, stage_name; clim = (-200, 500))
    energies = sort(collect(keys(vmi_dict)))
    vol0 = first(values(vmi_dict))
    mid_z = size(vol0, 3) ÷ 2
    fig = CM.Figure(size = (1200, 350), fontsize = 11)
    for (i, E) in enumerate(energies)
        ax = CM.Axis(fig[1, i]; title = "$stage_name  $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax, vmi_dict[E][:, :, mid_z]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax); CM.hidespines!(ax)
    end
    fig
end

# ╔═╡ 08120e01-0000-4000-8000-000000000001
md"""
**CMV (polynomial method).** Alvarez & Macovski 1976 sinogram-domain
decomposition — two steps:

1. **Calibration** — build per-bin effective spectra from the source spectrum,
   forward-project a synthetic water × iodine step-wedge, fit an inverse
   polynomial `(p_low, p_high) → (t_water, t_iodine)` → `pcct_vmi_calibration`.
2. **Decomposition** — evaluate that polynomial per ray on the low/high bin
   sinogram pair → `cmv_decomp` (`sino_iodine`, `sino_water`).

Fast, noisy, correct HU; canonical RWLS warm start.
"""

# ╔═╡ 08120008-c000-4000-8000-000000000001
# ── CMV polynomial calibration config ─────────────────────────────────────
# All hyperparams for the polynomial fit (calibration cell below) and the
# per-ray decomposition (cmv_decomp cell).
begin
    cmv_low_bins         = 1:2         # PCCT bins in the "low" channel
    cmv_high_bins        = 3:4         # PCCT bins in the "high" channel
    cmv_poly_order       = 4           # bivariate total-order polynomial
    cmv_n_water          = 40          # Chebyshev grid size (water axis)
    cmv_n_iodine         = 25          # Chebyshev grid size (iodine axis)
    cmv_max_water_cm     = 50.0        # step-wedge max water path (cm)
    cmv_max_iodine_g_cm2 = 0.15        # step-wedge max iodine density (g/cm²)
end

# ╔═╡ 08120008-0000-4000-8000-000000000000
# Polynomial calibration → src BS.calibrate_pcct_vmi_poly.
# Maps (p_low, p_high) sinogram pairs → (t_water, t_iodine) material paths.
pcct_vmi_calibration = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    BS.calibrate_pcct_vmi_poly(
        sim_scanner, prot;
        sim_opts         = sim_opts,
        low_bins         = cmv_low_bins,
        high_bins        = cmv_high_bins,
        order            = cmv_poly_order,
        n_water          = cmv_n_water,
        n_iodine         = cmv_n_iodine,
        max_water_cm     = cmv_max_water_cm,
        max_iodine_g_cm2 = cmv_max_iodine_g_cm2,
    )
end;

# ╔═╡ 08120e01-0000-4000-8000-000000000003
# CMV material decomposition → src BS.apply_pcct_vmi_poly.
# Evaluates the inverse polynomial per pixel on the 2-bin (low/high) sinogram
# pair from §12c to produce the canonical (sino_iodine, sino_water) warm start
# for RWLS / PWLS downstream, and for intermediate viz.
cmv_decomp = let
    sl = Float32.(sim_scan2_lohi.sino_low)
    sh = Float32.(sim_scan2_lohi.sino_high)
    sino_water, sino_iodine = BS.apply_pcct_vmi_poly(sl, sh, pcct_vmi_calibration)
    @info "[CMV poly]  ⟨∫ρ_W·dr⟩ = $(round(mean(sino_water), sigdigits = 4)) cm   ⟨∫ρ_I·dr⟩ = $(round(mean(sino_iodine), sigdigits = 4)) g/cm²"
    (sino_iodine = sino_iodine, sino_water = sino_water, geom = sim_scan2.geom)
end;

# ╔═╡ 08120e01-0000-4000-8000-000000000002
# CMV viz — (∫ρ_I·dr, ∫ρ_W·dr) sinograms + quick FBPs from the standalone
# cmv_decomp cell.
let
    _decomp_viz(cmv_decomp.sino_iodine, cmv_decomp.sino_water, "CMV (poly)")
end

# ╔═╡ 08120e03-0000-4000-8000-000000000001
md"""
**RWLS-GN (Ducros 2017).** 3-bin (A=1+2, B=3, C=4) Gauss-Newton in count
domain, Poisson weights + FFT quadratic spatial prior.  Single α, β_w, β_I.
When `use_rwls = false`, passes the warm start through unchanged.
"""

# ╔═╡ 08120009-a000-4000-8000-000000000003
# §12e.3 RWLS-GN config — 3-bin (A=1+2, B=3, C=4) Poisson-weighted Gauss-Newton
# on all bins simultaneously.  Consumed by `rwls_decomp`, which calls the
# library `BS.apply_rwls!`.  Output: single (sino_w, sino_I) pair that's
# energy-independent — VMI synthesis per-energy happens in §12f.
begin
    use_rwls             = true
    rwls_bin_groups      = [[1, 2], [3], [4]]  # 3-bin: A = 1+2, B = 3, C = 4
    rwls_n_iter          = 3
    rwls_α               = 0.3    # spatial reg weight in FFT proximal
    rwls_β_w             = 1.0    # water-channel reg scale
    rwls_β_I             = 1.0    # iodine-channel reg scale
    rwls_step_lim_iodine = 0.75   # per-iter |Δ| clamp (g/cm²) on iodine
    rwls_step_lim_water  = 5.0    # per-iter |Δ| clamp (g/cm²) on water
    rwls_relax           = 0.5    # GN relaxation factor (0.5 typical)
end

# ╔═╡ 08120010-0000-4000-8000-000000000040
# 12e.3 RWLS-GN material decomposition → src BS.apply_rwls!.
# N-bin (default: A=1+2, B=3, C=4) count-domain Poisson-weighted Gauss-Newton
# with FFT quadratic spatial prior.  Warm-started from `cmv_decomp`.  Output:
# single (sino_w, sino_I) pair; VMI synthesis per-energy happens in §12f.
rwls_decomp = let
    if !use_rwls
        @info "[RWLS] DISABLED — passing through polynomial CMV warm start"
        (sino_w = Float32.(cmv_decomp.sino_water),
         sino_I = Float32.(cmv_decomp.sino_iodine))
    else
        prot  = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
        basis = BS.pcct_rwls_basis(sim_scanner, prot;
                                    sim_opts   = sim_opts,
                                    bin_groups = rwls_bin_groups)
        combo = BS.combine_pcct_bin_counts(sim_scan2_bins_corrected,
                                            sim_scan2.I0_bins, rwls_bin_groups)

        sino_I_gpu = MtlArray(Float32.(cmv_decomp.sino_iodine))
        sino_W_gpu = MtlArray(Float32.(cmv_decomp.sino_water))
        bins_gpu   = [MtlArray(c) for c in combo.counts]

        BS.apply_rwls!(
            sino_I_gpu, sino_W_gpu, bins_gpu, combo.I0;
            basis, n_iter = rwls_n_iter,
            α = rwls_α, β_iodine = rwls_β_I, β_water = rwls_β_w,
            step_lim_iodine = rwls_step_lim_iodine,
            step_lim_water  = rwls_step_lim_water,
            relax = rwls_relax, verbose = true,
        )

        sino_w_cpu = Array(sino_W_gpu)
        sino_I_cpu = Array(sino_I_gpu)
        sino_I_gpu = nothing; sino_W_gpu = nothing; bins_gpu = nothing
        GC.gc(true)
        (sino_w = sino_w_cpu, sino_I = sino_I_cpu)
    end
end;

# ╔═╡ 08120e03-0000-4000-8000-000000000002
# RWLS viz — single material decomp (sino + FBP of ρ_I, ρ_W).
let
    _decomp_viz(rwls_decomp.sino_I, rwls_decomp.sino_w, "RWLS-GN")
end

# ╔═╡ 08120e05-0000-4000-8000-000000000001
md"""
**PWLS-L₂ (Long/Fessler 2014 §IV-B).** Single-pass 2-bin 2×2 matrix-curvature
polish, warm-started from RWLS.  Full 2×2 Gauss-Newton data curvature
captures iodine↔water anti-correlated noise; biharmonic (`CᵀC`) spatial
penalty in the 2-bin basis.  When `use_pwls = false`, `pwls_decomp = rwls_decomp`.
"""

# ╔═╡ 08120009-a000-4000-8000-000000000005
# §12e Stage 3 — PWLS-L₂ polish config.  Consumed by `pwls_decomp`, which
# calls the library `BS.apply_pwls!` with a 2-bin (low/high) PCCT basis.
begin
    use_pwls      = true
    pwls_low_bins  = [1, 2]     # PCCT bins for the "low" channel
    pwls_high_bins = [3, 4]     # PCCT bins for the "high" channel
    pwls_n_iter   = 20
    pwls_κ_iodine = 32.0        # De Pierro row-sum bound (iodine)
    pwls_κ_water  = 32.0        # De Pierro row-sum bound (water)
    pwls_relax    = 1.0         # SQS relaxation (1.0 unrelaxed)
end

# ╔═╡ 08120010-a000-4000-8000-000000000003
# 12e.4 PWLS-L₂ polish → src BS.apply_pwls!.
# 2-bin (low = 1+2, high = 3+4) Noh 2009 cost with Long/Fessler 2014 §IV-B
# 2×2 matrix-curvature per ray.  Warm-started from `rwls_decomp`.  Output:
# single (sino_w, sino_I) pair.  Pass-through when use_pwls = false.
pwls_decomp = let
    if !use_pwls
        @info "[PWLS] DISABLED — passing through RWLS output"
        (sino_w = rwls_decomp.sino_w, sino_I = rwls_decomp.sino_I)
    else
        prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
        basis_raw = BS.pcct_pwls_basis(sim_scanner, prot;
                                        sim_opts  = sim_opts,
                                        low_bins  = pwls_low_bins,
                                        high_bins = pwls_high_bins)
        # Renormalize ŵ per bin to Σ_k ŵ = 1 so the PWLS forward model
        # f_m = −log(Σ ŵ_m · exp(−μL)) → 0 on air rays.  Idempotent against
        # the current src (which already normalizes) and also repairs any
        # older loaded src that skipped this — so PWLS is correct regardless
        # of which src is precompiled.
        ŵ_L_n = Float32.(basis_raw.ŵ_bins[1] ./ sum(basis_raw.ŵ_bins[1]))
        ŵ_H_n = Float32.(basis_raw.ŵ_bins[2] ./ sum(basis_raw.ŵ_bins[2]))
        basis = (ŵ_bins = [ŵ_L_n, ŵ_H_n], p = basis_raw.p, q = basis_raw.q)

        sino_I_gpu = MtlArray(Float32.(rwls_decomp.sino_I))
        sino_W_gpu = MtlArray(Float32.(rwls_decomp.sino_w))
        h_low_gpu  = MtlArray(Float32.(sim_scan2_lohi.sino_low))
        h_high_gpu = MtlArray(Float32.(sim_scan2_lohi.sino_high))

        BS.apply_pwls!(
            sino_I_gpu, sino_W_gpu, h_low_gpu, h_high_gpu;
            basis,
            κ_iodine = pwls_κ_iodine,
            κ_water  = pwls_κ_water,
            n_iter   = pwls_n_iter,
            relax    = pwls_relax,
            verbose  = true,
        )

        sino_w_cpu = Array(sino_W_gpu)
        sino_I_cpu = Array(sino_I_gpu)
        sino_I_gpu = nothing; sino_W_gpu = nothing
        h_low_gpu  = nothing; h_high_gpu = nothing
        GC.gc(true)
        (sino_w = sino_w_cpu, sino_I = sino_I_cpu)
    end
end;

# ╔═╡ 08120e05-0000-4000-8000-000000000002
# PWLS viz — single material decomp (sino + FBP of ρ_I, ρ_W).
let
    _decomp_viz(pwls_decomp.sino_I, pwls_decomp.sino_w, "PWLS-L₂")
end

# ╔═╡ 08120f00-0000-4000-8000-000000000001
md"""
**VMI synthesis.** Consumes the single post-PWLS material decomp and
synthesises per-energy VMI HU volumes at target keV.

- **ACNR** — Kalender 1988 anti-correlated noise reduction, per-energy
  (`γ` tuned to preserve VMI at `acnr_E_ref` exactly).
- **VMI FBP** — `vmi_sino(E) = μρ_w(E)·sino_w + μρ_I(E)·sino_I` → FDK → HU.
- **VMI HIR** — optional Huber-regularised replacement for plain FBP.
- **Mono+** — optional frequency-split polish (Grant 2014), per-energy σ.

Output → `sim_scan2_vmi`.
"""

# ╔═╡ 08120009-a000-4000-8000-000000000001
# §12e pipeline — common setting: VMI target energies.
# Changing this re-runs the VMI synthesis but not the material-decomp stages.
vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 08120e04-0000-4000-8000-000000000001
md"""
**ACNR (Kalender 1988).** Anti-correlated noise reduction — projects out the
noise component orthogonal to `acnr_E_ref` in (water, iodine) material space.
Zero blur at the reference energy by construction; `γ_per_energy` tunable at
other energies.  Per-energy by design: takes the single post-PWLS decomp in,
produces a per-energy Dict out.  `use_acnr = false` ⇒ identity broadcast.
Viz below shows the 40 keV entry (largest γ → most visible effect).
"""

# ╔═╡ 08120009-a000-4000-8000-000000000004
# §12e Stage 2 — ACNR config.  Only the ACNR cell reads these; RWLS stays cached.
begin
    use_acnr   = true
    acnr_E_ref = 70.0    # VMI at this E unchanged (γ = 0)
    acnr_σ     = 1.0     # Gaussian σ for noise estimation
    acnr_γ_per_energy = Dict(
        40.0  => 0.95,
        70.0  => 0.0,
        100.0 => 0.50,
        140.0 => 0.60,
    )
end

# ╔═╡ 08120010-0000-4000-8000-000000000200
# 12f.1 ACNR — Anti-Correlated Noise Reduction → src BS.apply_acnr.
# Per-energy: projects out the noise component orthogonal to
# (c_water, c_iodine) at `acnr_E_ref` in material space, produces a
# Dict{E => (sino_w, sino_I)}.  γ = 0 at E_ref ⇒ identity by construction.
vmi_acnr, acnr_on = let
    base = (sino_w = pwls_decomp.sino_w, sino_I = pwls_decomp.sino_I)
    energies = [40.0, 70.0, 100.0, 140.0]
    if !use_acnr
        @info "[ACNR] DISABLED — broadcasting pwls_decomp to per-energy dict"
        (Dict{Float64, NamedTuple}(E => base for E in energies), false)
    else
        c_w = Float64(BS.compute_mass_μ_at_energy(XA.Materials.water,  acnr_E_ref))
        c_I = Float64(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, acnr_E_ref))
        corrected = Dict{Float64, NamedTuple}()
        for E in energies
            γ_E = get(acnr_γ_per_energy, E, 1.0)
            if γ_E ≈ 0.0
                corrected[E] = base   # identity at reference energy
            else
                sw_corr, sI_corr, _info = BS.apply_acnr(
                    base.sino_w, base.sino_I;
                    c_a = c_w, c_b = c_I, σ = acnr_σ, γ = γ_E, verbose = false)
                corrected[E] = (sino_w = sw_corr, sino_I = sI_corr)
            end
        end
        corrected, true
    end
end;

# ╔═╡ 08120e04-0000-4000-8000-000000000002
# ACNR viz — canonical at 40 keV (where γ is largest so the effect is visible).
let
    E_viz = 40.0
    s = vmi_acnr[E_viz]
    _decomp_viz(s.sino_I, s.sino_w, "ACNR ($(Int(E_viz)) keV)")
end

# ╔═╡ 08120f01-0000-4000-8000-000000000001
md"**VMI FBP.**"

# ╔═╡ 08120010-a000-4000-8000-000000000001
# 12e-2: VMI sinogram synthesis → src BS.synth_vmi_sino_domain.
# Per-energy: vmi_sino(E) = μρ_w(E)·sino_w_E + μρ_I(E)·sino_I_E → FDK → HU.
# Consumes the per-energy ACNR dict (pass-through if use_acnr = false).
vmi_fbp = let
    geom       = sim_scan2.geom
    recon_size = sim_matrix_size
    per_E      = vmi_acnr

    sino_w_by_E = Dict{Float64, Array{Float32, 3}}(E => per_E[E].sino_w for E in vmi_energies)
    sino_I_by_E = Dict{Float64, Array{Float32, 3}}(E => per_E[E].sino_I for E in vmi_energies)
    μρ_w_by_E   = [BS.compute_mass_μ_at_energy(XA.Materials.water,  E) for E in vmi_energies]
    μρ_I_by_E   = [BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E) for E in vmi_energies]

    # Closures stage each linear-combined sinogram to GPU for FDK and return
    # the recon volume (library stages it back via `Array(…)` for HU conv).
    fbp_ws_builder = (template, g, sz) -> begin
        tmpl_gpu = MtlArray(Float32.(template))
        BS.create_fdk_recon_workspace(tmpl_gpu, g, sz; filter = sim_vmi_filter)
    end
    fbp_recon = (ws, sino, g, sz) -> begin
        sino_gpu = MtlArray(Float32.(sino))
        BS.reconstruct!(ws, sino_gpu, g, sz)
    end

    result = BS.synth_vmi_sino_domain(
        sino_w_by_E, sino_I_by_E, vmi_energies;
        μρ_a_by_E = μρ_w_by_E,
        μρ_b_by_E = μρ_I_by_E,
        fbp_workspace_builder = fbp_ws_builder,
        fbp_recon!            = fbp_recon,
        geom                  = geom,
        matrix_size           = recon_size,
        fov_mask_radius_frac  = nothing,
        verbose               = true,
    )

    raw_vmi = Dict{Float64, Array{Float32, 3}}(E => result.volumes[i] for (i, E) in enumerate(result.energies))
    mid_z   = recon_size[3] ÷ 2
    for E in vmi_energies
        roi = raw_vmi[E][200:300, 200:300, mid_z]
        @info "FBP VMI $(Int(E)) keV — water σ=$(round(std(roi), digits = 1)) HU, mean=$(round(mean(roi), digits = 1)) HU"
    end

    (vmi = raw_vmi, energies = collect(result.energies))
end;

# ╔═╡ 08120f01-0000-4000-8000-000000000002
# VMI FBP viz — HU at all target energies (mid slice).
let
    _vmi_row_viz(vmi_fbp.vmi, "FBP")
end

# ╔═╡ 08120f02-0000-4000-8000-000000000001
md"**VMI HIR.** Optional Huber-regularised replacement for plain FBP (`use_hir`)."

# ╔═╡ 08120009-a000-4000-8000-000000000006
# §12e Stage 5 — HIR recon config (optional replacement for plain FBP).
# Only the vmi_hir cell reads these; FBP / Mono+ stay cached.
begin
    use_hir         = false
    hir_strength    = 3
    hir_lambda      = 10.0f0
    hir_nepochs     = 2
    hir_n_subsets   = 12
    hir_huber_delta = 0.06f0
    hir_relaxation  = 0.35f0
end

# ╔═╡ 08120010-a000-4000-8000-000000000002
# 12e-3: HIR on VMI sinograms → src BS.synth_vmi_sino_domain with HIR closures.
# Same per-energy linear-combine → recon → HU pattern as vmi_fbp, but the
# workspace / recon closures swap FDK for HIR (Huber-regularised iterative).
# Pass-through = vmi_fbp when use_hir = false.
vmi_hir, hir_on = let
    result = if !use_hir
        vmi_fbp
    else
        geom       = sim_scan2.geom
        recon_size = sim_matrix_size
        per_E      = vmi_acnr

        sino_w_by_E = Dict{Float64, Array{Float32, 3}}(E => per_E[E].sino_w for E in vmi_energies)
        sino_I_by_E = Dict{Float64, Array{Float32, 3}}(E => per_E[E].sino_I for E in vmi_energies)
        μρ_w_by_E   = [BS.compute_mass_μ_at_energy(XA.Materials.water,  E) for E in vmi_energies]
        μρ_I_by_E   = [BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E) for E in vmi_energies]

        # HIR workspace builder — build once, patch `.params` with the tune
        # from the HIR config cell so all 4 energies share identical settings.
        hir_ws_builder = (template, g, sz) -> begin
            tmpl_gpu = MtlArray(Float32.(template))
            ws = BS.create_hir_recon_workspace(tmpl_gpu, g, sz;
                                                strength = hir_strength,
                                                filter   = sim_vmi_filter)
            ws.params = BS.HIRParams(hir_strength, hir_lambda, 30, hir_nepochs,
                                      hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35))
            ws
        end
        hir_recon = (ws, sino, g, sz) -> begin
            sino_gpu = MtlArray(Float32.(sino))
            BS.reconstruct!(ws, sino_gpu, g, sz)
        end

        r = BS.synth_vmi_sino_domain(
            sino_w_by_E, sino_I_by_E, vmi_energies;
            μρ_a_by_E = μρ_w_by_E,
            μρ_b_by_E = μρ_I_by_E,
            fbp_workspace_builder = hir_ws_builder,
            fbp_recon!            = hir_recon,
            geom                  = geom,
            matrix_size           = recon_size,
            fov_mask_radius_frac  = nothing,
            verbose               = true,
        )

        raw_vmi = Dict{Float64, Array{Float32, 3}}(E => r.volumes[i] for (i, E) in enumerate(r.energies))
        mid_z   = recon_size[3] ÷ 2
        for E in vmi_energies
            roi     = raw_vmi[E][200:300, 200:300, mid_z]
            roi_fbp = vmi_fbp.vmi[E][200:300, 200:300, mid_z]
            @info "VMI $(Int(E)) keV — FBP σ=$(round(std(roi_fbp), digits = 1)) → HIR σ=$(round(std(roi), digits = 1)) HU, mean=$(round(mean(roi), digits = 1)) HU"
        end

        (vmi = raw_vmi, energies = collect(r.energies))
    end
    result, use_hir
end;

# ╔═╡ 08120f02-0000-4000-8000-000000000002
# HIR viz — only distinct from FBP viz when use_hir = true.
let
    tag = hir_on ? "HIR" : "FBP (HIR disabled)"
    _vmi_row_viz(vmi_hir.vmi, tag)
end

# ╔═╡ 08120f03-0000-4000-8000-000000000001
md"**Mono+ (final).** Grant 2014 frequency-split per energy; σ = 0 ⇒ identity."

# ╔═╡ 08120009-a000-4000-8000-000000000007
# §12e Stage 6 — Mono+ config.  Identical pattern to notebook 06 (sim_vmi_plus).
# Per-energy σ in pixels, aligned with `vmi_energies = [40, 70, 100, 140]`.
# σ at `vmip_E_noise_opt` is an identity — Mono+(E_opt) = VMI_opt regardless of
# σ — so a value there is ignored.  Larger σ → more noise reduction, less
# fine-detail contrast at that energy.  Tune independently per keV.
#
# Typical values: σ ≈ 1.0 mild, σ ≈ 2.0 balanced, σ ≈ 3.0 aggressive.
# Effective HP cutoff frequency ≈ 1/(2πσ) cycles/pixel.
begin
    use_mono_plus     = true
    vmip_E_noise_opt  = 70.0
    vmip_σ_lp_px      = Float64[1.5, 0.0, 0.0, 1.0]
end

# ╔═╡ 08120010-b000-4000-8000-000000000001
# 12f.4: Mono+ (Grant 2014) — identical pattern to notebook 06 (sim_vmi_plus).
# Per-energy σ: call `BS.apply_mono_plus` once per target keV with that energy's
# σ, take only that energy's slot from each call.  Extra cost is
# length(E)−1 redundant LP(VMI_opt) computations, negligible vs the upstream
# recon.  Output is a Dict keyed by energy to stay compatible with this
# notebook's downstream viz/measurement cells.
sim_scan2_vmi, mono_on = let
    energies = Float64.(vmi_energies)
    vols_in  = [vmi_hir.vmi[E] for E in energies]

    if !use_mono_plus
        @info "[Mono+] DISABLED — passing vmi_hir through unchanged"
        out = Dict{Float64, Array{Float32, 3}}(E => vmi_hir.vmi[E] for E in energies)
        out, use_mono_plus
    else
        length(vmip_σ_lp_px) == length(energies) ||
            error("vmip_σ_lp_px length $(length(vmip_σ_lp_px)) ≠ energies length $(length(energies))")
        haskey(vmi_hir.vmi, vmip_E_noise_opt) ||
            error("Mono+ reference vmip_E_noise_opt=$(vmip_E_noise_opt) keV not in vmi_energies=$energies")

        σ_effective = Float64.(vmip_σ_lp_px)
        out = Dict{Float64, Array{Float32, 3}}()
        mid_z = sim_matrix_size[3] ÷ 2
        for (i, σ_i) in enumerate(σ_effective)
            result_i = BS.apply_mono_plus(vols_in, energies;
                E_noise_opt = vmip_E_noise_opt,
                σ_lp_px     = σ_i,
                verbose     = false)
            vol_E = vols_in[i]
            mp    = result_i.volumes[i]
            roi_before = vol_E[200:300, 200:300, mid_z]
            roi_after  = mp[200:300, 200:300, mid_z]
            out[energies[i]] = mp
            @info "Mono+ $(Int(energies[i])) keV (E_opt=$(Int(vmip_E_noise_opt)), σ_lp=$(σ_i) px): σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"
        end
        out, use_mono_plus
    end
end;

# ╔═╡ 08120f03-0000-4000-8000-000000000002
# Mono+ viz — final sim_scan2_vmi after optional 40 keV frequency-split polish.
let
    tag = mono_on ? "Mono+" : "pass-through"
    _vmi_row_viz(sim_scan2_vmi, "sim_scan2_vmi ($tag)")
end

# ╔═╡ 08070017-0000-4000-8000-000000000000
# VMI noise vs energy: Clinical QIR3, Sim FBP, Sim HIR
let
    water_idx = 1
    vmi_idx = [9, 10, 11, 12]  # VMI 40/70/100/140 in clinical_measurements
    energies = [40, 70, 100, 140]
    energy_labels = ["40 keV", "70 keV", "100 keV", "140 keV"]
    mid_z = sim_matrix_size[3] ÷ 2

    # Clinical VMI noise (ALL clinical VMI is QIR3 — no VMI FBP available)
    clin_σ = [clinical_measurements[vmi_idx[i]].rod_stds[water_idx] for i in 1:4]

    # Simulated VMI noise: FBP and HIR
    sim_fbp_σ = [std(vmi_fbp.vmi[Float64(energies[i])][200:300, 200:300, mid_z]) for i in 1:4]
    sim_hir_σ = [std(sim_scan2_vmi[Float64(energies[i])][200:300, 200:300, mid_z]) for i in 1:4]

    fig = CM.Figure(size = (700, 420), fontsize = 13)
    ax = CM.Axis(fig[1, 1];
        title = "VMI Water Noise — 140 kVp / 10 mGy\n(No clinical VMI FBP available)",
        xlabel = "VMI Energy (keV)", ylabel = "Water σ (HU)",
        xticks = (1:4, energy_labels))

    bw = 0.22
    sw = 2.0
    x = collect(1:4)

    # Solid bar: Simulated FBP
    CM.barplot!(ax, x .- 0.24, sim_fbp_σ; width = bw,
        color = :darkorange, label = "Simulated (FBP)")

    # Outline bars: iterative (QIR3 / HIR)
    CM.barplot!(ax, x, clin_σ; width = bw,
        color = (:steelblue, 0.15), strokecolor = :steelblue, strokewidth = sw,
        label = "Clinical (QIR3)")
    CM.barplot!(ax, x .+ 0.24, sim_hir_σ; width = bw,
        color = (:darkorange, 0.15), strokecolor = :darkorange, strokewidth = sw,
        label = "Simulated (HIR)")

    CM.ylims!(ax, 0, nothing)
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "alpha_vmi_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08120011-0000-4000-8000-000000000000
# VMI qualitative montage — Sim FBP / Clinical QIR3 / Sim HIR+Mono+
let
    energies = [40.0, 70.0, 100.0, 140.0]
    clin_vols = [hu_140_mid_vmi40, hu_140_mid_vmi70, hu_140_mid_vmi100, hu_140_mid_vmi140]
    clin_mid_z = seg_result.slice_idx
    sim_mid_z = sim_matrix_size[3] ÷ 2
    cr = (-200, 500)

    acnr_tag = acnr_on ? " +ACNR" : ""
    row1_label = "Sim FBP$(acnr_tag)"
    row3_label = hir_on ? "Sim HIR$(acnr_tag)" : "Sim FBP$(acnr_tag)"
    mono_tag(E) = (mono_on && E == 40.0) ? " +Mono+" : ""

    fig = CM.Figure(size = (1000, 800), fontsize = 10)

    for (i, E) in enumerate(energies)
        # Row 1: Simulated FBP
        ax1 = CM.Axis(fig[1, i]; title = "$row1_label $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax1, vmi_fbp.vmi[E][:, :, sim_mid_z]; colormap = :grays, colorrange = cr)
        CM.hidedecorations!(ax1); CM.hidespines!(ax1)

        # Row 2: Clinical QIR3
        ax2 = CM.Axis(fig[2, i]; title = "Clinical QIR3 $(Int(E)) keV", aspect = CM.DataAspect(), yreversed = true)
        CM.heatmap!(ax2, clin_vols[i][:, :, clin_mid_z]; colormap = :grays, colorrange = cr)
        CM.hidedecorations!(ax2); CM.hidespines!(ax2)

        # Row 3: Final pipeline output (HIR/FBP + Mono+)
        ax3 = CM.Axis(fig[3, i]; title = "$row3_label$(mono_tag(E)) $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax3, sim_scan2_vmi[E][:, :, sim_mid_z]; colormap = :grays, colorrange = cr)
        CM.hidedecorations!(ax3); CM.hidespines!(ax3)
    end

    CM.save(joinpath(RESULTS_DIR, "alpha_pcct_vmi_montage.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08120012-0000-4000-8000-000000000000
md"**VMI vs Clinical VMI.**"

# ╔═╡ 08120013-0000-4000-8000-000000000000
# Side-by-side: clinical VMI vs simulated VMI at each energy
# nothing

# ╔═╡ 08140001-0000-4000-8000-000000000000
md"""
## Scan 3: 140 kVp / ~20 mGy

Placeholder — same pipeline as Scan 2 with higher mA.
"""

# ╔═╡ 08140002-0000-4000-8000-000000000000
sim_scan3 = nothing  # TODO: implement (same as Scan 2 with mA = sim_mA_scan3)

# ╔═╡ 08140003-a000-4000-8000-000000000001
md"### Poly"

# ╔═╡ 08140003-0000-4000-8000-000000000000
begin
    sim_scan3_poly_fbp = nothing  # TODO: poly FBP recon
    sim_scan3_poly_hir = nothing  # TODO: poly HIR recon
end

# ╔═╡ 08140003-b000-4000-8000-000000000001
md"### VMI"

# ╔═╡ 08140003-b000-4000-8000-000000000002
# TODO: VMI pipeline for Scan 3 (same structure as Scan 2).
nothing

# ╔═╡ 08150001-0000-4000-8000-000000000000
md"""
## Scan 4: 120 kVp / ~10 mGy

Placeholder — different kVp (120 vs 140); uses `μ_water_120` for HU conversion.
"""

# ╔═╡ 08150002-0000-4000-8000-000000000000
sim_scan4 = nothing  # TODO: implement (kVp=120, mA = sim_mA_scan4)

# ╔═╡ 08150003-a000-4000-8000-000000000001
md"### Poly"

# ╔═╡ 08150003-0000-4000-8000-000000000000
begin
    sim_scan4_poly_fbp = nothing  # TODO: poly FBP recon (use μ_water_120)
    sim_scan4_poly_hir = nothing  # TODO: poly HIR recon (use μ_water_120)
end

# ╔═╡ 08150003-b000-4000-8000-000000000001
md"### VMI"

# ╔═╡ 08150003-b000-4000-8000-000000000002
# TODO: VMI pipeline for Scan 4 (same structure as Scan 2, kVp=120).
nothing

# ╔═╡ 08160001-0000-4000-8000-000000000000
md"""
## 16. Simulated Segmentation & Measurements

Segment Scan 2 FBP reconstruction, then measure.
"""

# ╔═╡ 08160002-0000-4000-8000-000000000000
# Segment simulated Scan 2 (use poly FBP as reference)
sim_seg_result = let
    ref = sim_scan2_poly_fbp
    mid_z = size(ref, 3) ÷ 2
    mask, rods, center = segment_gammex_rods(ref[:, :, mid_z]; fov_cm = sim_fov_cm, clockwise = false)
    (mask = mask, rods = rods, center = center)
end

# ╔═╡ 08120004-a000-4000-8000-000000000000
# Segmentation overlay + debug print
let
    # Print inner ring rod positions and HU
    inner_rods = [r for r in seg_result.rods if r.ring == :inner]
    rod_strs = ["$(r.name): cx=$(round(r.cx,digits=1)) cy=$(round(r.cy,digits=1)) ang=$(r.angle_deg)° hu=$(round(r.mean_hu,digits=1))" for r in inner_rods]
    println(join(rod_strs, "\n"))

    fig = CM.Figure(size = (1200, 550), fontsize = 11)

    # Clinical segmentation
    clin_slice = hu_140_mid_fbp[:, :, seg_result.slice_idx]
    ax1 = CM.Axis(fig[1, 1], title = "Clinical Segmentation", subtitle = "140 kVp FBP", aspect = CM.DataAspect(), yreversed = true)
    CM.heatmap!(ax1, clin_slice, colormap = :grays, colorrange = (-200, 500))
    for r in seg_result.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (35.0 / size(clin_slice, 1))
        CM.lines!(ax1, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th), color = :red, linewidth = 1.5)
        CM.text!(ax1, r.cx, r.cy - rpx - 2, text = r.name, fontsize = 6, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax1); CM.hidespines!(ax1)

    # Simulated segmentation
    mid_z = sim_n_recon_slices ÷ 2
    sim_slice = sim_scan2_poly_fbp[:, :, mid_z]
    ax2 = CM.Axis(fig[1, 2], title = "Simulated Segmentation", subtitle = "140 kVp Poly FBP", aspect = CM.DataAspect())
    CM.heatmap!(ax2, sim_slice, colormap = :grays, colorrange = (-200, 500))
    for r in sim_seg_result.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (sim_fov_cm / size(sim_slice, 1))
        CM.lines!(ax2, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th), color = :red, linewidth = 1.5)
        CM.text!(ax2, r.cx, r.cy - rpx - 2, text = r.name, fontsize = 6, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax2); CM.hidespines!(ax2)

    CM.save(joinpath(RESULTS_DIR, "alpha_segmentation_overlay.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08120004-b000-4000-8000-000000000002
# Diagnostic: show exactly WHERE the circular-edge MTF is sampled on both images
let
    fov_cm = 35.0
    body_radius_cm = 16.5
    margin_inner_pix = 15.0
    margin_outer_pix = 5.0
    fov_guard_pix = 3.0

    fig = CM.Figure(size = (1200, 550), fontsize = 12)

    for (col, (hu_slice, seg_ctr, ttl, do_yrev)) in enumerate([
        (hu_140_mid_fbp[:, :, seg_result.slice_idx], seg_result.center, "Clinical FBP (Br44f)", true),
        (sim_scan2_poly_fbp[:, :, sim_n_recon_slices ÷ 2], sim_seg_result.center, "Simulated Poly FBP", false),
    ])
        nx, ny = size(hu_slice)
        pixel_cm = fov_cm / nx
        cx, cy = Float64(seg_ctr.cx), Float64(seg_ctr.cy)
        img_cx, img_cy = (nx + 1) / 2.0, (ny + 1) / 2.0
        edge_r_pix = body_radius_cm / pixel_cm
        r_min = edge_r_pix - margin_inner_pix
        r_max = edge_r_pix + margin_outer_pix
        fov_r_pix = nx / 2.0

        ax = CM.Axis(fig[1, col]; title = ttl, aspect = CM.DataAspect(), yreversed = do_yrev)
        CM.heatmap!(ax, hu_slice; colormap = :grays, colorrange = (-200, 500))

        # Phantom body edge (where ESF is measured)
        th = range(0, 2π, length = 360)
        CM.lines!(ax, cx .+ edge_r_pix .* cos.(th), cy .+ edge_r_pix .* sin.(th);
            color = :cyan, linewidth = 2, label = "Body edge (r=16.5cm)")

        # Inner sampling boundary
        CM.lines!(ax, cx .+ r_min .* cos.(th), cy .+ r_min .* sin.(th);
            color = :lime, linewidth = 1, linestyle = :dash, label = "Inner margin (−15 px)")

        # Outer sampling boundary
        CM.lines!(ax, cx .+ r_max .* cos.(th), cy .+ r_max .* sin.(th);
            color = :red, linewidth = 1, linestyle = :dash, label = "Outer margin (+5 px)")

        # FOV guard circle (angles whose outer point falls outside this are skipped)
        guard_r = fov_r_pix - fov_guard_pix
        CM.lines!(ax, img_cx .+ guard_r .* cos.(th), img_cy .+ guard_r .* sin.(th);
            color = :yellow, linewidth = 1, linestyle = :dot, label = "FOV guard (skip if beyond)")

        # Mark which angles are used vs skipped
        n_angles = 720
        sample_angles = range(0, 2π - 2π / n_angles, length = n_angles)
        for sa in sample_angles
            x_out = cx + r_max * cos(sa)
            y_out = cy + r_max * sin(sa)
            dist = sqrt((x_out - img_cx)^2 + (y_out - img_cy)^2)
            if dist > guard_r
                # This angle is SKIPPED — mark in red
                CM.scatter!(ax, [x_out], [y_out]; color = (:red, 0.4), markersize = 2)
            end
        end

        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end
    CM.Legend(fig[2, :],
        [CM.LineElement(color = :cyan, linewidth = 2),
         CM.LineElement(color = :lime, linestyle = :dash),
         CM.LineElement(color = :red, linestyle = :dash),
         CM.LineElement(color = :yellow, linestyle = :dot),
         CM.MarkerElement(color = :red, marker = :circle, markersize = 6)],
        ["Body edge (ESF center, r=16.5cm)",
         "Inner sampling limit (−15 px ≈ −10.3mm)",
         "Outer sampling limit (+5 px ≈ +3.4mm)",
         "FOV guard circle (skip angles beyond this)",
         "Skipped angle sample points"],
        orientation = :horizontal, framevisible = false, labelsize = 10, nbanks = 2)
    fig
end

# ╔═╡ 08120004-b000-4000-8000-000000000004
# Diagnostic: show NPS ROI placement (center of phantom, 30mm radius, sub-ROI patches)
let
    fov_cm = 35.0
    nps_roi_radius_mm = 30.0
    nps_roi_size = 32  # pixels per sub-ROI patch

    fig = CM.Figure(size = (1200, 550), fontsize = 12)

    for (col, (hu_slice, seg_ctr, ttl, do_yrev)) in enumerate([
        (hu_140_mid_fbp[:, :, seg_result.slice_idx], seg_result.center, "Clinical FBP (Br44f)", true),
        (sim_scan2_poly_fbp[:, :, sim_n_recon_slices ÷ 2], sim_seg_result.center, "Simulated Poly FBP", false),
    ])
        nx, ny = size(hu_slice)
        pixel_mm = fov_cm / nx * 10.0
        cx, cy = round(Int, seg_ctr.cx), round(Int, seg_ctr.cy)
        roi_r_px = nps_roi_radius_mm / pixel_mm

        ax = CM.Axis(fig[1, col]; title = ttl, aspect = CM.DataAspect(), yreversed = do_yrev)
        CM.heatmap!(ax, hu_slice; colormap = :grays, colorrange = (-200, 500))

        # NPS sampling region (30mm radius circle)
        th = range(0, 2π, length = 360)
        CM.lines!(ax, Float64(cx) .+ roi_r_px .* cos.(th), Float64(cy) .+ roi_r_px .* sin.(th);
            color = :cyan, linewidth = 2, label = "NPS region (r=30mm)")

        # Show some example sub-ROI patches within the region
        rows_in = [i for i in 1:nx if abs(i - cx) <= roi_r_px]
        cols_in = [j for j in 1:ny if abs(j - cy) <= roi_r_px]
        r0_base = isempty(rows_in) ? cx : minimum(rows_in)
        c0_base = isempty(cols_in) ? cy : minimum(cols_in)
        nr = isempty(rows_in) ? 0 : length(rows_in)
        nc = isempty(cols_in) ? 0 : length(cols_in)

        step = max(round(Int, nps_roi_size * 0.25), 1)  # 75% overlap
        n_shown = 0
        for iy in 0:((nr - nps_roi_size) ÷ step), ix in 0:((nc - nps_roi_size) ÷ step)
            pr = r0_base + iy * step
            pc = c0_base + ix * step
            # Check all corners are within the NPS circle
            corners_in = all([(pr - cx)^2 + (pc - cy)^2,
                              (pr + nps_roi_size - 1 - cx)^2 + (pc - cy)^2,
                              (pr - cx)^2 + (pc + nps_roi_size - 1 - cy)^2,
                              (pr + nps_roi_size - 1 - cx)^2 + (pc + nps_roi_size - 1 - cy)^2
                             ] .<= roi_r_px^2)
            if corners_in
                xs = [pr, pr + nps_roi_size - 1, pr + nps_roi_size - 1, pr, pr]
                ys = [pc, pc, pc + nps_roi_size - 1, pc + nps_roi_size - 1, pc]
                CM.lines!(ax, Float64.(xs), Float64.(ys); color = (:lime, 0.3), linewidth = 0.5)
                n_shown += 1
            end
        end

        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end
    CM.Legend(fig[2, :],
        [CM.LineElement(color = :cyan, linewidth = 2),
         CM.LineElement(color = (:lime, 0.6), linewidth = 1)],
        ["NPS sampling region (r=30mm from center)",
         "32×32 px sub-ROI patches (75% overlap, quadratic detrend + Hann window)"],
        orientation = :horizontal, framevisible = false, labelsize = 10)
    fig
end

# ╔═╡ 08160003-0000-4000-8000-000000000000
# Measurements: Poly FBP + VMI at each energy
sim_measurements_scan2 = let
    results = []
    push!(
        results, measure_scan(
            sim_scan2_poly_fbp, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_poly_fbp"; fov_cm = sim_fov_cm
        )
    )
    # Add VMI measurements at each energy
    for E in [40.0, 70.0, 100.0, 140.0]
        if haskey(sim_scan2_vmi, E)
            push!(
                results, measure_scan(
                    sim_scan2_vmi[E], sim_seg_result.mask,
                    sim_seg_result.rods, sim_seg_result.center,
                    "scan2_VMI_$(Int(E))keV"; fov_cm = sim_fov_cm
                )
            )
        end
    end
    results
end;

# ╔═╡ 08160004-0000-4000-8000-000000000000
# Display summary: Water ROI noise for Scan 2 poly FBP
let
    m = sim_measurements_scan2[1]
    water_idx = findfirst(n -> startswith(n, "Water"), m.rod_names)
    water_σ = water_idx !== nothing ? m.rod_stds[water_idx] : NaN

    @info "Scan 2 Poly FBP — Water noise: σ = $(round(water_σ, digits = 1)) HU"
end

# ╔═╡ 08170001-0000-4000-8000-000000000000
md"""
## 17. Comparison Figures

Active for simulated data. Clinical comparisons are placeholder until DICOM loaded.
"""

# ╔═╡ 08170002-0000-4000-8000-000000000000
# HU scatter: Clinical FBP vs Simulated Poly FBP + VMI (Ca and I rods)
let
    # Clinical references
    cm_fbp = clinical_measurements[2]    # 140kVp_174mA_FBP
    cm_vmi = Dict(
        40  => findfirst(m -> m.name == "140kVp_174mA_VMI40", clinical_measurements),
        70  => findfirst(m -> m.name == "140kVp_174mA_VMI70", clinical_measurements),
        100 => findfirst(m -> m.name == "140kVp_174mA_VMI100", clinical_measurements),
        140 => findfirst(m -> m.name == "140kVp_174mA_VMI140", clinical_measurements),
    )

    # Simulated: index 1 = poly FBP, 2-5 = VMI 40/70/100/140
    sm_fbp = sim_measurements_scan2[1]
    sm_vmi = Dict(
        40  => length(sim_measurements_scan2) >= 2 ? sim_measurements_scan2[2] : nothing,
        70  => length(sim_measurements_scan2) >= 3 ? sim_measurements_scan2[3] : nothing,
        100 => length(sim_measurements_scan2) >= 4 ? sim_measurements_scan2[4] : nothing,
        140 => length(sim_measurements_scan2) >= 5 ? sim_measurements_scan2[5] : nothing,
    )

    vmi_colors = Dict(40 => :purple, 70 => :seagreen, 100 => :darkorange, 140 => :crimson)

    fig = CM.Figure(size = (750, 900), fontsize = 11)

    # --- Top: Calcium rods ---
    ax_ca = CM.Axis(
        fig[1, 1], title = "Calcium Rods", subtitle = "Clinical vs Simulated (Poly FBP + VMI)",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    ca_idx = [i for i in 1:length(cm_fbp.rod_names) if startswith(cm_fbp.rod_names[i], "Ca")]
    if !isempty(ca_idx)
        # Poly FBP
        CM.scatter!(ax_ca, cm_fbp.rod_means[ca_idx], sm_fbp.rod_means[ca_idx];
            color = :steelblue, markersize = 10, label = "Poly FBP")
        CM.lines!(ax_ca, [-100, 3000], [-100, 3000]; color = :gray60, linestyle = :dash, label = "Unity")

        # VMI at each energy (clinical VMI vs simulated VMI)
        for E in [40, 70, 100, 140]
            ci = cm_vmi[E]
            sv = sm_vmi[E]
            if ci !== nothing && sv !== nothing
                cm_v = clinical_measurements[ci]
                CM.scatter!(ax_ca, cm_v.rod_means[ca_idx], sv.rod_means[ca_idx];
                    color = vmi_colors[E], markersize = 8, marker = :diamond,
                    label = "VMI $(E) keV")
            end
        end
        CM.axislegend(ax_ca; position = :lt, labelsize = 9)
    end

    # --- Bottom: Iodine rods ---
    ax_i = CM.Axis(
        fig[2, 1], title = "Iodine Rods", subtitle = "Clinical vs Simulated (Poly FBP + VMI)",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    i_idx = [i for i in 1:length(cm_fbp.rod_names) if startswith(cm_fbp.rod_names[i], "I ")]
    if !isempty(i_idx)
        # Poly FBP
        CM.scatter!(ax_i, cm_fbp.rod_means[i_idx], sm_fbp.rod_means[i_idx];
            color = :steelblue, markersize = 10, label = "Poly FBP")
        CM.lines!(ax_i, [-50, 1800], [-50, 1800]; color = :gray60, linestyle = :dash, label = "Unity")

        # VMI at each energy
        for E in [40, 70, 100, 140]
            ci = cm_vmi[E]
            sv = sm_vmi[E]
            if ci !== nothing && sv !== nothing
                cm_v = clinical_measurements[ci]
                CM.scatter!(ax_i, cm_v.rod_means[i_idx], sv.rod_means[i_idx];
                    color = vmi_colors[E], markersize = 8, marker = :diamond,
                    label = "VMI $(E) keV")
            end
        end
        CM.axislegend(ax_i; position = :lt, labelsize = 9)
    end

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_scatter_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08170003-0000-4000-8000-000000000000
# Noise bar chart: Clinical vs Simulated — Poly FBP + VMI (water σ)
let
    water_idx = findfirst(n -> startswith(n, "Water"), clinical_measurements[2].rod_names)

    # Clinical: FBP + VMI 40/70/100/140
    cm_fbp = clinical_measurements[2]   # 140kVp_174mA_FBP
    cm_vmi_idx = Dict(
        40  => findfirst(m -> m.name == "140kVp_174mA_VMI40", clinical_measurements),
        70  => findfirst(m -> m.name == "140kVp_174mA_VMI70", clinical_measurements),
        100 => findfirst(m -> m.name == "140kVp_174mA_VMI100", clinical_measurements),
        140 => findfirst(m -> m.name == "140kVp_174mA_VMI140", clinical_measurements),
    )

    # Simulated: index 1 = poly FBP, 2-5 = VMI 40/70/100/140
    sm_fbp = sim_measurements_scan2[1]

    # Build labels and values
    labels = String["Poly FBP"]
    clin_vals = Float64[cm_fbp.rod_stds[water_idx]]
    sim_vals = Float64[sm_fbp.rod_stds[water_idx]]

    for E in [40, 70, 100, 140]
        push!(labels, "VMI $(E)")
        ci = cm_vmi_idx[E]
        clin_vals = push!(clin_vals, ci !== nothing ? clinical_measurements[ci].rod_stds[water_idx] : NaN)
        # Find matching simulated VMI
        si = findfirst(m -> m.name == "scan2_VMI_$(E)keV", sim_measurements_scan2)
        push!(sim_vals, si !== nothing ? sim_measurements_scan2[si].rod_stds[water_idx] : NaN)
    end

    n = length(labels)
    x = collect(1:n)
    bw = 0.3

    fig = CM.Figure(size = (800, 400), fontsize = 11)
    ax = CM.Axis(fig[1, 1]; title = "Water ROI Noise — 140 kVp / 10 mGy",
        xlabel = "Reconstruction", ylabel = "Water σ (HU)",
        xticks = (x, labels))
    CM.barplot!(ax, x .- 0.16, clin_vals; width = bw, color = :steelblue, label = "Clinical (Br44f, Poly=FBP, VMI=QIR3)")
    CM.barplot!(ax, x .+ 0.16, sim_vals; width = bw, color = :coral, label = "Simulated (FBP)")
    for (xi, v) in zip(x .- 0.16, clin_vals)
        isnan(v) && continue
        CM.text!(ax, xi, v + 0.5; text = "$(round(v, digits=1))", align = (:center, :bottom), fontsize = 9)
    end
    for (xi, v) in zip(x .+ 0.16, sim_vals)
        isnan(v) && continue
        CM.text!(ax, xi, v + 0.5; text = "$(round(v, digits=1))", align = (:center, :bottom), fontsize = 9)
    end
    CM.ylims!(ax, 0, nothing)
    CM.axislegend(ax; position = :lt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08170004-0000-4000-8000-000000000000
# NPS comparison: Clinical FBP vs Simulated Poly FBP
let
    cm = clinical_measurements[2]
    sm = sim_measurements_scan2[1]

    fig = CM.Figure(size = (600, 400), fontsize = 11)
    ax = CM.Axis(fig[1, 1]; title = "NPS — Clinical vs Simulated FBP (140 kVp / 10 mGy)",
        xlabel = "Spatial Frequency (lp/mm)", ylabel = "NPS (HU² mm²)")

    if hasproperty(cm, :nps) && cm.nps !== nothing
        CM.lines!(ax, cm.nps.frequencies, cm.nps.nps_1d; color = :steelblue, linewidth = 2, label = "Clinical FBP (Br36)")
    end
    if hasproperty(sm, :nps) && sm.nps !== nothing
        CM.lines!(ax, sm.nps.frequencies, sm.nps.nps_1d; color = :coral, linewidth = 2, label = "Simulated Poly FBP")
    end
    CM.axislegend(ax; position = :rt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_nps.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08170005-0000-4000-8000-000000000000
# MTF comparison: Clinical FBP vs Simulated Poly FBP
let
    cm = clinical_measurements[2]
    sm = sim_measurements_scan2[1]

    fig = CM.Figure(size = (600, 400), fontsize = 11)
    ax = CM.Axis(fig[1, 1]; title = "MTF — Clinical vs Simulated FBP (140 kVp / 10 mGy)",
        xlabel = "Spatial Frequency (lp/mm)", ylabel = "MTF")

    if hasproperty(cm, :mtf) && cm.mtf !== nothing
        CM.lines!(ax, cm.mtf.frequencies, cm.mtf.mtf; color = :steelblue, linewidth = 2, label = "Clinical FBP (Br36)")
    end
    if hasproperty(sm, :mtf) && sm.mtf !== nothing
        CM.lines!(ax, sm.mtf.frequencies, sm.mtf.mtf; color = :coral, linewidth = 2, label = "Simulated Poly FBP")
    end
    CM.hlines!(ax, [0.5, 0.1]; color = :gray60, linestyle = :dash, linewidth = 0.5)
    CM.axislegend(ax; position = :rt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08180001-0000-4000-8000-000000000000
md"""
## 18. Export Results
"""

# ╔═╡ 08180002-0000-4000-8000-000000000000
# # Export all measurements to CSV + JLD2
# let
#     # CSV: rod HU means, stds, CNR for each scan/recon combo
#     header = [
#         "scan_name",
#         [["$(n)_mean", "$(n)_std", "$(n)_cnr"] for n in sim_measurements_scan2[1].rod_names]...,
#         "nps_peak_freq", "nps_area", "mtf_f50", "mtf_f10",
#     ]
#     header_flat = vcat(header[1], reduce(vcat, header[2:(end - 4)]), header[(end - 3):end])

#     rows = []
#     for m in sim_measurements_scan2
#         row = Any[m.name]
#         for i in 1:length(m.rod_names)
#             push!(row, round(m.rod_means[i], digits = 2))
#             push!(row, round(m.rod_stds[i], digits = 2))
#             push!(row, round(m.rod_cnr[i], digits = 2))
#         end
#         push!(row, round(m.nps_peak_freq, digits = 2))
#         push!(row, round(m.nps_area, digits = 2))
#         push!(row, round(m.mtf_f50, digits = 2))
#         push!(row, round(m.mtf_f10, digits = 2))
#         push!(rows, row)
#     end

#     csv_path = joinpath(RESULTS_DIR, "naeotom_alpha_scan2_measurements.csv")
#     open(csv_path, "w") do io
#         println(io, join(header_flat, ","))
#         for row in rows
#             println(io, join(row, ","))
#         end
#     end

#     # JLD2: NPS + MTF curves
#     JLD2.jldsave(
#         joinpath(RESULTS_DIR, "naeotom_alpha_scan2_nps.jld2");
#         Dict(
#             m.name => (freq = m.nps.frequencies, nps = m.nps.nps_1d, nnps = m.nps.nnps_1d)
#                 for m in sim_measurements_scan2
#         )...
#     )

#     JLD2.jldsave(
#         joinpath(RESULTS_DIR, "naeotom_alpha_scan2_mtf.jld2");
#         Dict(
#             m.name => (freq = m.mtf.frequencies, mtf = m.mtf.mtf)
#                 for m in sim_measurements_scan2
#         )...
#     )

#     md"**Exported to:** `$(RESULTS_DIR)`"
# end

# ╔═╡ 08190001-0000-4000-8000-000000000000
md"""
## 19. Appendix

### Phantom Geometry
- **Body:** 33 cm diameter × 5 cm thick, Gammex 472 Solid Water
- **Outer ring (R=10.5 cm):** Ca 50/100/200/300/400, Water, SW refs — 8 rods at 45° spacing
- **Inner ring (R=5.5 cm):** I 2.0/2.5/5.0/7.5/10.0/15.0/20.0, Water — 8 rods at 45° spacing
- **Rod radius:** 1.4 cm with thin air gaps

### NIST Expected HU (140 kVp, approximate)

| Insert | Expected HU |
|--------|------------|
| Water | 0 |
| Ca 50 | ~55 |
| Ca 100 | ~120 |
| Ca 200 | ~260 |
| Ca 300 | ~400 |
| Ca 400 | ~540 |
| I 2.0 | ~45 |
| I 5.0 | ~115 |
| I 10.0 | ~235 |
| I 20.0 | ~465 |

### Scan Protocol Reference

| Parameter | Value |
|-----------|-------|
| Scanner | Siemens NAEOTOM Alpha |
| Software | syngo CT VB10A |
| Detector | CdTe, 144 × 0.4 mm (standard mode) |
| Filter | W1 (DICOM label; actual: 3 mm Al + 0.9 mm Ti) |
| Rotation time | 0.5 s |
| Collimation | 144 × 0.4 mm = 57.6 mm |
| FOV | 350 mm |
| Matrix | 512 × 512 |
| Pixel spacing | 0.684 × 0.684 mm |
| Slice thickness | 0.4 mm |
| Kernel | Br44f (medium body) |
| IR | QIR strength 3 (all VMI series; Poly has both FBP and QIR3) |
"""

# ╔═╡ 08190002-0000-4000-8000-000000000000
md"""
### Current Tuning Parameters
- **Electronic noise:** $(sim_scanner.electronic_noise)
- **Detection gain:** $(sim_scanner.detection_gain)
- **Noise floor:** $(sim_noise_floor_hu) HU
- **Custom filter (Br36 approx):** $(sim_custom_filter.control_x) → $(sim_custom_filter.control_y)
"""

# ╔═╡ Cell order:
# ╠═08010001-0000-4000-8000-000000000000
# ╠═a129a323-d627-4272-934c-5b4734286f37
# ╠═08010002-0000-4000-8000-000000000000
# ╠═08010003-0000-4000-8000-000000000000
# ╠═08010004-0000-4000-8000-000000000000
# ╠═08010005-0000-4000-8000-000000000000
# ╠═08010006-0000-4000-8000-000000000000
# ╠═08010007-0000-4000-8000-000000000000
# ╠═08010008-0000-4000-8000-000000000000
# ╠═08010009-0000-4000-8000-000000000000
# ╠═08010010-0000-4000-8000-000000000000
# ╠═08010011-0000-4000-8000-000000000000
# ╟─08010030-0000-4000-8000-000000000000
# ╠═b629dcd4-fe9a-4ddd-8f45-0a74918f0093
# ╠═08010012-0000-4000-8000-000000000000
# ╠═08010013-0000-4000-8000-000000000000
# ╠═08010014-0000-4000-8000-000000000000
# ╠═08010015-0000-4000-8000-000000000000
# ╠═08010016-0000-4000-8000-000000000000
# ╠═08010017-0000-4000-8000-000000000000
# ╠═08010018-0000-4000-8000-000000000000
# ╠═08010019-0000-4000-8000-000000000000
# ╠═08010020-0000-4000-8000-000000000000
# ╠═08010021-0000-4000-8000-000000000000
# ╟─08020001-0000-4000-8000-000000000000
# ╠═08020002-0000-4000-8000-000000000000
# ╠═08020002-b000-4000-8000-000000000001
# ╠═08020003-0000-4000-8000-000000000000
# ╠═08020004-0000-4000-8000-000000000000
# ╠═08020005-0000-4000-8000-000000000000
# ╠═08020006-0000-4000-8000-000000000000
# ╟─08030001-0000-4000-8000-000000000000
# ╠═08030002-0000-4000-8000-000000000000
# ╟─08040001-0000-4000-8000-000000000000
# ╠═08040002-0000-4000-8000-000000000000
# ╟─08050001-0000-4000-8000-000000000000
# ╠═08050002-0000-4000-8000-000000000000
# ╟─08060001-0000-4000-8000-000000000000
# ╠═08060002-0000-4000-8000-000000000000
# ╟─08070001-0000-4000-8000-000000000000
# ╠═08070002-0000-4000-8000-000000000000
# ╟─08070003-0000-4000-8000-000000000000
# ╠═08070004-0000-4000-8000-000000000000
# ╟─08070005-0000-4000-8000-000000000000
# ╟─08070006-0000-4000-8000-000000000000
# ╠═08070007-0000-4000-8000-000000000000
# ╟─08070008-0000-4000-8000-000000000000
# ╟─08070009-0000-4000-8000-000000000000
# ╟─08070010-0000-4000-8000-000000000000
# ╟─08070011-0000-4000-8000-000000000000
# ╟─08070012-0000-4000-8000-000000000000
# ╟─08070013-0000-4000-8000-000000000000
# ╟─08070014-0000-4000-8000-000000000000
# ╟─08070015-0000-4000-8000-000000000000
# ╟─08070016-0000-4000-8000-000000000000
# ╟─08070018-0000-4000-8000-000000000000
# ╟─08070019-0000-4000-8000-000000000000
# ╟─08080001-0000-4000-8000-000000000000
# ╠═08080002-0000-4000-8000-000000000000
# ╠═08080003-0000-4000-8000-000000000000
# ╠═08080004-0000-4000-8000-000000000000
# ╟─08080005-0000-4000-8000-000000000000
# ╟─08090001-0000-4000-8000-000000000000
# ╠═08090002-0000-4000-8000-000000000000
# ╟─08090003-0000-4000-8000-000000000000
# ╠═08090004-0000-4000-8000-000000000000
# ╠═08090005-0000-4000-8000-000000000000
# ╠═08090006-0000-4000-8000-000000000000
# ╠═08090007-0000-4000-8000-000000000000
# ╠═08090007-b000-4000-8000-000000000001
# ╠═08090008-0000-4000-8000-000000000000
# ╟─08100001-0000-4000-8000-000000000000
# ╠═08100002-0000-4000-8000-000000000000
# ╟─08100003-0000-4000-8000-000000000000
# ╠═08100004-0000-4000-8000-000000000000
# ╟─08100005-0000-4000-8000-000000000000
# ╟─08130001-0000-4000-8000-000000000000
# ╠═08130002-0000-4000-8000-000000000000
# ╟─08130003-a000-4000-8000-000000000001
# ╠═08130003-0000-4000-8000-000000000000
# ╟─08130003-b000-4000-8000-000000000001
# ╠═08130003-b000-4000-8000-000000000002
# ╟─08110001-0000-4000-8000-000000000000
# ╠═08110002-0000-4000-8000-000000000000
# ╟─08120002-b000-4000-8000-000000000001
# ╟─08120001-0000-4000-8000-000000000000
# ╠═08120002-0000-4000-8000-000000000000
# ╟─08120003-0000-4000-8000-000000000000
# ╠═08120004-0000-4000-8000-000000000000
# ╟─08120004-a000-4000-8000-000000000001
# ╟─08120004-a000-4000-8000-000000000002
# ╟─08120004-b000-4000-8000-000000000001
# ╟─08120004-b000-4000-8000-000000000003
# ╟─08120005-a000-4000-8000-000000000001
# ╠═08120005-a000-4000-8000-000000000002
# ╟─08120005-0000-4000-8000-000000000000
# ╠═08120006-0000-4000-8000-000000000000
# ╠═08120006-b000-4000-8000-000000000001
# ╟─08120009-0000-4000-8000-000000000000
# ╠═08120e00-0000-4000-8000-000000000001
# ╟─08120e01-0000-4000-8000-000000000001
# ╠═08120008-c000-4000-8000-000000000001
# ╠═08120008-0000-4000-8000-000000000000
# ╠═08120e01-0000-4000-8000-000000000003
# ╟─08120e01-0000-4000-8000-000000000002
# ╟─08120e03-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000003
# ╠═08120010-0000-4000-8000-000000000040
# ╟─08120e03-0000-4000-8000-000000000002
# ╟─08120e05-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000005
# ╠═08120010-a000-4000-8000-000000000003
# ╟─08120e05-0000-4000-8000-000000000002
# ╟─08120f00-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000001
# ╟─08120e04-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000004
# ╠═08120010-0000-4000-8000-000000000200
# ╟─08120e04-0000-4000-8000-000000000002
# ╟─08120f01-0000-4000-8000-000000000001
# ╠═08120010-a000-4000-8000-000000000001
# ╟─08120f01-0000-4000-8000-000000000002
# ╟─08120f02-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000006
# ╠═08120010-a000-4000-8000-000000000002
# ╟─08120f02-0000-4000-8000-000000000002
# ╟─08120f03-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000007
# ╠═08120010-b000-4000-8000-000000000001
# ╟─08120f03-0000-4000-8000-000000000002
# ╟─08070017-0000-4000-8000-000000000000
# ╟─08120011-0000-4000-8000-000000000000
# ╟─08120012-0000-4000-8000-000000000000
# ╠═08120013-0000-4000-8000-000000000000
# ╟─08140001-0000-4000-8000-000000000000
# ╠═08140002-0000-4000-8000-000000000000
# ╟─08140003-a000-4000-8000-000000000001
# ╠═08140003-0000-4000-8000-000000000000
# ╟─08140003-b000-4000-8000-000000000001
# ╠═08140003-b000-4000-8000-000000000002
# ╟─08150001-0000-4000-8000-000000000000
# ╠═08150002-0000-4000-8000-000000000000
# ╟─08150003-a000-4000-8000-000000000001
# ╠═08150003-0000-4000-8000-000000000000
# ╟─08150003-b000-4000-8000-000000000001
# ╠═08150003-b000-4000-8000-000000000002
# ╟─08160001-0000-4000-8000-000000000000
# ╠═08160002-0000-4000-8000-000000000000
# ╟─08120004-a000-4000-8000-000000000000
# ╟─08120004-b000-4000-8000-000000000002
# ╟─08120004-b000-4000-8000-000000000004
# ╠═08160003-0000-4000-8000-000000000000
# ╟─08160004-0000-4000-8000-000000000000
# ╟─08170001-0000-4000-8000-000000000000
# ╟─08170002-0000-4000-8000-000000000000
# ╟─08170003-0000-4000-8000-000000000000
# ╟─08170004-0000-4000-8000-000000000000
# ╟─08170005-0000-4000-8000-000000000000
# ╟─08180001-0000-4000-8000-000000000000
# ╠═08180002-0000-4000-8000-000000000000
# ╟─08190001-0000-4000-8000-000000000000
# ╟─08190002-0000-4000-8000-000000000000
