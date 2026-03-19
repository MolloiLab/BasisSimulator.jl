### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 08010001-0000-4000-8000-000000000000
begin
    using Pkg: Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.resolve()
    Pkg.instantiate()
    using Revise
end

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

Four axial acquisitions. Each has Poly FBP, Poly QIR3, and VMI at 40/70/100/140 keV (QIR3).

| # | kVp | mA | mAs | CTDIvol (mGy) | Recon |
|---|-----|-----|-----|---------------|-------|
| 1 | 140 |  46 |  23 |  2.68 | Poly FBP + QIR3; VMI 40/70/100/140 keV QIR3 |
| 2 | 140 | 174 |  88 | 10.12 | Poly FBP + QIR3; VMI 40/70/100/140 keV QIR3 |
| 3 | 140 | 347 | 176 | 20.25 | Poly FBP + QIR3; VMI 40/70/100/140 keV QIR3 |
| 4 | 120 | 253 | 128 | 10.15 | Poly FBP + QIR3; VMI 40/70/100/140 keV QIR3 |

Common: Axial, 0.5 s rotation, 144 × 0.4 mm collimation, W1 (Tungsten) filter, 512 × 512, 350 mm FOV
"""

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
rootdir = "/Users/daleblack/Desktop/SCANS/03132026 (alpha)"

# ╔═╡ 08040001-0000-4000-8000-000000000000
md"""
### Scan 1: 140 kVp / 46 mA / 2.68 mGy
"""

# ╔═╡ 08040002-0000-4000-8000-000000000000
begin  # 140 kVp / 46 mA / 2.68 mGy
    dcms_140_low_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_46mA_2.68mGyCTDI/Poly_FBP"))
    dcms_140_low_qir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_46mA_2.68mGyCTDI/Poly_QIR3"))
    hu_140_low_fbp = load_hu_volume(dcms_140_low_fbp)
    hu_140_low_qir = load_hu_volume(dcms_140_low_qir)
    dcms_140_low_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_46mA_2.68mGyCTDI/VMI_40keV"))
    dcms_140_low_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_46mA_2.68mGyCTDI/VMI_70keV"))
    dcms_140_low_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_46mA_2.68mGyCTDI/VMI_100keV"))
    dcms_140_low_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_46mA_2.68mGyCTDI/VMI_140keV"))
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
    dcms_140_mid_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/Poly_FBP"))
    dcms_140_mid_qir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/Poly_QIR3"))
    hu_140_mid_fbp = load_hu_volume(dcms_140_mid_fbp)
    hu_140_mid_qir = load_hu_volume(dcms_140_mid_qir)
    dcms_140_mid_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI_40keV"))
    dcms_140_mid_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI_70keV"))
    dcms_140_mid_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI_100keV"))
    dcms_140_mid_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI_140keV"))
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
    dcms_140_high_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/Poly_FBP"))
    dcms_140_high_qir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/Poly_QIR3"))
    hu_140_high_fbp = load_hu_volume(dcms_140_high_fbp)
    hu_140_high_qir = load_hu_volume(dcms_140_high_qir)
    dcms_140_high_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI_40keV"))
    dcms_140_high_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI_70keV"))
    dcms_140_high_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI_100keV"))
    dcms_140_high_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_347mA_20.25mGyCTDI/VMI_140keV"))
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
    dcms_120_mid_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/Poly_FBP"))
    dcms_120_mid_qir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/Poly_QIR3"))
    hu_120_mid_fbp = load_hu_volume(dcms_120_mid_fbp)
    hu_120_mid_qir = load_hu_volume(dcms_120_mid_qir)
    dcms_120_mid_vmi40 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI_40keV"))
    dcms_120_mid_vmi70 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI_70keV"))
    dcms_120_mid_vmi100 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI_100keV"))
    dcms_120_mid_vmi140 = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_253mA_10.15mGyCTDI/VMI_140keV"))
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
        (hu_140_low_fbp, "140kVp_46mA_FBP"),
        (hu_140_mid_fbp, "140kVp_174mA_FBP"),
        (hu_140_high_fbp, "140kVp_347mA_FBP"),
        (hu_120_mid_fbp, "120kVp_253mA_FBP"),
        # Poly QIR3
        (hu_140_low_qir, "140kVp_46mA_QIR3"),
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
        "140 kVp / 46 mA / 2.68 mGy",
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
        "140 kVp / 46 mA / 2.68 mGy",
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
        "140 kVp / 46 mA\n(2.68 mGy)",
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

# ╔═╡ 08070017-0000-4000-8000-000000000000
# VMI noise vs energy bar chart
let
    water_idx = 1
    vmi_idx = [9, 10, 11, 12]  # VMI 40/70/100/140 in clinical_measurements
    energies = [40, 70, 100, 140]
    vmi_σ = [clinical_measurements[vmi_idx[i]].rod_stds[water_idx] for i in 1:4]

    fig = CM.Figure(size = (600, 350), fontsize = 13)
    ax = CM.Axis(fig[1, 1]; title = "VMI Water Noise — 140 kVp / 10.12 mGy (QIR3)",
        xlabel = "VMI Energy (keV)", ylabel = "Water σ (HU)",
        xticks = (1:4, string.(energies)))
    CM.barplot!(ax, 1:4, vmi_σ; width = 0.6, color = :steelblue)
    for (xi, v) in zip(1:4, vmi_σ)
        CM.text!(ax, xi, v + 0.5; text = "$(round(v, digits=1))", fontsize = 10, align = (:center, :bottom))
    end
    CM.ylims!(ax, 0, nothing)
    CM.save(joinpath(RESULTS_DIR, "alpha_vmi_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08070018-0000-4000-8000-000000000000
md"""
## 6f. Clinical HU Scatter — Poly FBP (140 kVp Dose Ladder)
"""

# ╔═╡ 08070019-0000-4000-8000-000000000000
# Scatter plot: HU accuracy across dose levels (FBP only, Ca + I rods)
let
    fbp_idx = [1, 2, 3]  # 140 kVp FBP at 3/10/20 mGy
    fbp_labels = ["46 mA (2.68 mGy)", "174 mA (10.12 mGy)", "347 mA (20.25 mGy)"]
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
    sim_phantom_cpu = create_custom_gammex_472(
        n_voxels = 512,
        n_slices = 32,
        fov_cm = 35.0,
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
    CM.hidedecorations!(ax)
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
    sim_n_views = 984             # standard NAEOTOM view count
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
    (1.0, 0.9, 0.65, 0.25, 0.001),
)

# ╔═╡ 08090008-0000-4000-8000-000000000000
# System noise floor (dose-independent, applied post-reconstruction)
sim_noise_floor_hu = 0.0  # HU — disabled for debugging (was 15.0)

# ╔═╡ 08100001-0000-4000-8000-000000000000
md"""
## 10. Water Attenuation Calibration

Analytical μ\_water from spectrum — no simulation needed.

1. `resolve_spectrum` → filtered spectrum (energies `e`, weights `w`)
2. Mean energy → `compute_μ_at_energy(water, mean_E)` for each bin's energy range
3. Per-bin: window spectrum to bin boundaries, compute bin-specific mean energy
"""

# ╔═╡ 08100002-0000-4000-8000-000000000000
# Analytical μ_water at 140 kVp (combined spectrum)
μ_water_140 = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
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
#     e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
#     mean_E = sum(e .* w) / sum(w)
#     mu = BS.compute_μ_at_energy(XA.Materials.water, mean_E)
#     @info "120 kVp μ_water: $(round(mu, digits = 5)) (E̅=$(round(mean_E, digits = 1)) keV)"
#     mu
# end

# ╔═╡ 08100005-0000-4000-8000-000000000000
# md"**Water attenuation (120 kVp, analytical):** μ\_water = $(round(μ_water_120, sigdigits=4)) cm⁻¹"

# ╔═╡ 08110001-0000-4000-8000-000000000000
md"""
## 11. Scan 2 Simulation: 140 kVp / ~10 mGy

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
    combined_sino = ws.combined
    pcct_sino = result.pcct_sino

    # Save I0 values for notebook-level combine
    I0_bins_combine = copy(ws.I0_bins)
    I0_bins_norm = copy(ws.I0_bins_norm)

    # Copy to CPU — 4 bins
    combined_cpu = Array(combined_sino)
    bins_cpu = [Array(pcct_sino.bins[i]) for i in 1:length(pcct_sino.bins)]

    # Cleanup
    ws = nothing; result = nothing; GC.gc(true)

    (
        combined = combined_cpu,
        bins = bins_cpu,
        I0_bins = I0_bins_combine,
        I0_bins_norm = I0_bins_norm,
        geom = geom,
    )
end

# ╔═╡ 08120001-0000-4000-8000-000000000000
md"""
## 12. Scan 2 Reconstructions

### 12a. Polyenergetic FBP
"""

# ╔═╡ 08120002-0000-4000-8000-000000000000
# Poly FBP — notebook-level I0-weighted combine + diagnostic
#
# Proper T3D combine: recover raw counts per bin, sum, normalize.
# Compare against ws.combined to find the bug.
sim_scan2_poly_fbp = let
    geom = sim_scan2.geom
    recon_size = sim_matrix_size
    bins = sim_scan2.bins
    I0 = sim_scan2.I0_bins  # per-bin I0 from workspace

    # ── Notebook-level proper combine ──
    # N_b = I0_b × exp(-sino_b)   (recover raw photon counts)
    # N_total = Σ N_b              (sum counts — noise reduces as √N)
    # sino = -log(N_total / I0_total)
    I0_total = Float32(sum(I0))
    combined = zeros(Float32, size(bins[1]))
    for (b, sino_b) in enumerate(bins)
        I0b = Float32(I0[b])
        combined .+= I0b .* exp.(.-sino_b)
    end
    combined .= .-log.(max.(combined, Float32(1e-10)) ./ I0_total)

    # Use the notebook combine (known correct math)
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
end

# ╔═╡ 08120002-b000-4000-8000-000000000001
# Diagnostic: I0 values and ws.combined vs notebook combine
let
    I0 = sim_scan2.I0_bins
    I0_norm = sim_scan2.I0_bins_norm
    bins = sim_scan2.bins
    ws_comb = sim_scan2.combined

    println("I0 per bin combine: ", round.(I0, sigdigits=4))
    println("I0 per bin norm:    ", round.(I0_norm, sigdigits=4))
    println("I0 total: ", round(sum(I0), sigdigits=4))
    println("Ratio (element-wise): ", round.(I0 ./ I0_norm, sigdigits=4))
    println()

    # Redo notebook combine
    I0_total = Float32(sum(I0))
    nb_comb = zeros(Float32, size(bins[1]))
    for (b, sino_b) in enumerate(bins)
        nb_comb .+= Float32(I0[b]) .* exp.(.-sino_b)
    end
    nb_comb .= .-log.(max.(nb_comb, Float32(1e-10)) ./ I0_total)

    diff = ws_comb .- nb_comb

    println("Notebook combine: min=", round(minimum(nb_comb), digits=4),
            " max=", round(maximum(nb_comb), digits=4),
            " mean=", round(mean(nb_comb), digits=4))
    println("ws.combined:      min=", round(minimum(ws_comb), digits=4),
            " max=", round(maximum(ws_comb), digits=4),
            " mean=", round(mean(ws_comb), digits=4))
    println("Diff (ws - nb):   max_abs=", round(maximum(abs.(diff)), digits=6),
            " mean_abs=", round(mean(abs.(diff)), digits=6))
    println()

    # Per-bin sinogram stats
    bin_stats = ["Bin $b: min=$(round(minimum(s), digits=4)) max=$(round(maximum(s), digits=4)) mean=$(round(mean(s), digits=4)) I0=$(round(I0[b], sigdigits=4))" for (b, s) in enumerate(bins)]
    println(join(bin_stats, "\n"))
end

# ╔═╡ 08120003-0000-4000-8000-000000000000
md"### 12b. Polyenergetic HIR (strength=3)"

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

# ╔═╡ 08120004-a000-4000-8000-000000000001
md"### 12b½. Clinical vs Simulated Poly FBP (Scan 2, 140 kVp)"

# ╔═╡ 08120004-a000-4000-8000-000000000002
# Side-by-side: clinical FBP vs simulated poly FBP (mid-slice)
let
    mid_z = sim_n_recon_slices ÷ 2
    window = (-200, 500)

    fig = CM.Figure(size = (1200, 550), fontsize = 14)

    # Clinical — yreversed to match DICOM convention (as in clinical-only sections)
    ax1 = CM.Axis(fig[1, 1], title = "Clinical FBP", subtitle = "140 kVp, 174 mA, Br36", aspect = CM.DataAspect(), yreversed = true)
    CM.heatmap!(ax1, hu_140_mid_fbp[:, :, seg_result.slice_idx], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax1)

    # Simulated
    ax2 = CM.Axis(fig[1, 2], title = "Simulated Poly FBP", subtitle = "140 kVp, 174 mA", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax2, sim_scan2_poly_fbp[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax2)

    CM.Colorbar(fig[1, 3], hm, label = "HU")
    CM.Label(fig[0, :], text = "Scan 2 (140 kVp / 10 mGy): Clinical vs Simulated", fontsize = 16, font = :bold)

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_clinical_vs_sim.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08120005-0000-4000-8000-000000000000
md"""
### 12c. Low-Threshold Bin — FBP + HIR (4-bin config — TODO: update for new thresholds)
"""

# ╔═╡ 08120006-0000-4000-8000-000000000000
# TODO: Low-bin FBP — needs update for 4-bin config
# sim_scan2_low_fbp = nothing

# ╔═╡ 08120007-0000-4000-8000-000000000000
# TODO: Low-bin HIR — needs update for 4-bin config
# sim_scan2_low_hir = nothing

# ╔═╡ 08120008-0000-4000-8000-000000000000
md"""
### 12d. High-Threshold Bin (>65 keV) — FBP + HIR
"""

# ╔═╡ 08120009-0000-4000-8000-000000000000
# TODO: High-bin FBP — needs update for 4-bin config
# sim_scan2_high_fbp = nothing

# ╔═╡ 08120010-0000-4000-8000-000000000000
# TODO: High-bin HIR — needs update for 4-bin config
# sim_scan2_high_hir = nothing

# ╔═╡ 08120011-0000-4000-8000-000000000000
md"""
### 12e. VMI 40/70/100/140 keV (HIR)

Virtual monoenergetic images synthesized from 2-material decomposition.
"""

# ╔═╡ 08120012-0000-4000-8000-000000000000
# TODO: VMI reconstruction — uncomment when ready
# sim_scan2_vmi = Dict{Float64, Array{Float32, 3}}()

# ╔═╡ 08120013-0000-4000-8000-000000000000
# TODO: FBP vs HIR comparison — uncomment when HIR recon is enabled
# nothing

# ╔═╡ 08120014-0000-4000-8000-000000000000
# TODO: Threshold-binned recon comparison — needs update for 4-bin config
# nothing

# ╔═╡ 08120015-0000-4000-8000-000000000000
# TODO: VMI montage — uncomment when VMI recon is enabled
# nothing

# ╔═╡ 08120016-0000-4000-8000-000000000000
# TODO: VMI energy-dependent contrast curves — uncomment when VMI recon is enabled
# nothing

# ╔═╡ 08126001-0000-4000-8000-000000000000
md"""
### 12f. Mono+ VMI — Noise-Optimized Monoenergetic (Grant et al. 2014)

Standard VMI at low keV (e.g. 40 keV) enhances iodine contrast but also amplifies
noise, often **worsening** CNR. The Mono+ algorithm overcomes this via
frequency-domain recombination (Grant et al., *Invest Radiol* 2014;49:586–592):

1. Compute standard VMI at target energy `E` (high contrast, high noise)
2. Compute standard VMI at optimal energy `E_opt` ≈ 70 keV (minimum noise)
3. Gaussian low-pass (LP) filter both images
4. Combine: **Mono+(E) = LP(VMI(E)) + HP(VMI(E\_opt))**

Equivalently: `Mono+(E) = VMI(E_opt) + LP(VMI(E) − VMI(E_opt))`

This preserves low-frequency contrast structure from the low-keV image while
retaining high-frequency noise texture from the optimal-keV image.

**Foundation:** Yu L, Leng S, McCollough CH. *AJR* 2012;199:S9–S15 — image-domain
VMI via 2-material decomposition (Eq 2): `μᵏ = (μ/ρ)₁ᵏ·ρ₁ + (μ/ρ)₂ᵏ·ρ₂`,
`k = L, H`. Solving for ρ₁, ρ₂ then synthesizing at energy E via Eq 1.

**Tunable:** `σ_lp_mm` controls the LP cutoff. Larger = more noise reduction,
less fine-detail contrast at target E.
"""

# ╔═╡ 08126002-0000-4000-8000-000000000000
"""
    mono_plus_vmi(vmi_target, vmi_optimal; σ_lp_mm, pixel_mm) -> Array

Mono+ algorithm (Grant et al., Invest Radiol 2014;49:586–592).

Frequency-split recombination: low-frequency contrast from target-energy VMI
combined with high-frequency noise texture from optimal-energy VMI.

    Mono+(E) = LP(VMI(E)) + [VMI(E_opt) − LP(VMI(E_opt))]

where LP = 2D Gaussian low-pass with spatial-domain width σ_lp_mm.

Applied per-slice for 3D volumes.

# Arguments
- `vmi_target`: VMI at desired energy (e.g., 40 keV) — high contrast, high noise
- `vmi_optimal`: VMI at optimal energy (~70 keV) — minimum noise
- `σ_lp_mm`: Gaussian LP kernel width in mm (controls frequency split)
- `pixel_mm`: pixel size in mm
"""
function mono_plus_vmi(
        vmi_target::Array{T},
        vmi_optimal::Array{T};
        σ_lp_mm::Float64 = 2.0,
        pixel_mm::Float64 = 0.684,
    ) where {T <: AbstractFloat}
    σ_pix = σ_lp_mm / pixel_mm

    if ndims(vmi_target) == 2
        return _mono_plus_slice(vmi_target, vmi_optimal, σ_pix)
    end

    # 3D: apply per-slice
    nx, ny, nz = size(vmi_target)
    result = similar(vmi_target)
    for k in 1:nz
        result[:, :, k] = _mono_plus_slice(
            vmi_target[:, :, k], vmi_optimal[:, :, k], σ_pix
        )
    end
    return result
end

# ╔═╡ 285deb3d-9c3b-4fe2-8640-ac286f37ae47
"""
    _mono_plus_slice(target, optimal, σ_pix)

Single-slice Mono+ via 2D Gaussian LP in Fourier domain.
H(fx,fy) = exp(−2π²σ²(fx² + fy²)), σ in pixels, f in cycles/pixel.
"""
function _mono_plus_slice(
        target::AbstractMatrix{T},
        optimal::AbstractMatrix{T},
        σ_pix::Float64,
    ) where {T}
    nx, ny = size(target)

    # Gaussian LP filter in frequency domain (FFTW ordering)
    coeff = -2.0 * π^2 * σ_pix^2
    H = Matrix{Float64}(undef, nx, ny)
    for j in 1:ny
        fy = j - 1 <= ny ÷ 2 ? (j - 1) / ny : (j - 1 - ny) / ny
        fy2 = fy^2
        for i in 1:nx
            fx = i - 1 <= nx ÷ 2 ? (i - 1) / nx : (i - 1 - nx) / nx
            H[i, j] = exp(coeff * (fx^2 + fy2))
        end
    end

    # LP filter both images
    F_target = fft(Float64.(target))
    F_optimal = fft(Float64.(optimal))
    LP_target = real(ifft(F_target .* H))
    LP_optimal = real(ifft(F_optimal .* H))

    # Mono+ = LP(target) + HP(optimal) = LP(target) + optimal − LP(optimal)
    return T.(LP_target .+ Float64.(optimal) .- LP_optimal)
end

# ╔═╡ 08126003-0000-4000-8000-000000000000
# TODO: Mono+ tuning parameters — uncomment when VMI is enabled
# begin
#     mono_plus_σ_lp_mm = 2.0
#     mono_plus_E_optimal = 70.0
#     mono_plus_pixel_mm = sim_fov_cm / 512 * 10.0
# end

# ╔═╡ 08126004-0000-4000-8000-000000000000
# TODO: Compute Mono+ — uncomment when VMI is enabled
# sim_scan2_mono_plus = Dict{Float64, Array{Float32, 3}}()

# ╔═╡ 08126005-0000-4000-8000-000000000000
# TODO: Mono vs Mono+ at 40 keV — uncomment when VMI is enabled
# nothing

# ╔═╡ 08126006-0000-4000-8000-000000000000
# TODO: Full Mono montage — uncomment when VMI is enabled
# nothing

# ╔═╡ 08126007-0000-4000-8000-000000000000
# TODO: Mono CNR/noise/contrast curves — uncomment when VMI is enabled
# nothing

# ╔═╡ 08126008-0000-4000-8000-000000000000
# TODO: Mono+ σ sensitivity analysis — uncomment when VMI is enabled
# nothing

# ╔═╡ 08130001-0000-4000-8000-000000000000
md"""
## 13. Scan 1 Simulation: 140 kVp / ~3 mGy

**Placeholder** — Same pipeline as Scan 2 with lower mA.
"""

# ╔═╡ 08130002-0000-4000-8000-000000000000
sim_scan1 = nothing  # TODO: implement (same as §11 with mA = sim_mA_scan1)

# ╔═╡ 08130003-0000-4000-8000-000000000000
begin
    sim_scan1_poly_fbp = nothing  # TODO: poly FBP recon
    sim_scan1_poly_hir = nothing  # TODO: poly HIR recon
end

# ╔═╡ 08140001-0000-4000-8000-000000000000
md"""
## 14. Scan 3 Simulation: 140 kVp / ~20 mGy

**Placeholder** — Same pipeline as Scan 2 with higher mA.
"""

# ╔═╡ 08140002-0000-4000-8000-000000000000
sim_scan3 = nothing  # TODO: implement (same as §11 with mA = sim_mA_scan3)

# ╔═╡ 08140003-0000-4000-8000-000000000000
begin
    sim_scan3_poly_fbp = nothing  # TODO: poly FBP recon
    sim_scan3_poly_hir = nothing  # TODO: poly HIR recon
end

# ╔═╡ 08150001-0000-4000-8000-000000000000
md"""
## 15. Scan 4 Simulation: 120 kVp / ~10 mGy

**Placeholder** — Different kVp (120 instead of 140). Uses μ\_water\_120 for HU conversion.
"""

# ╔═╡ 08150002-0000-4000-8000-000000000000
sim_scan4 = nothing  # TODO: implement (kVp=120, mA = sim_mA_scan4)

# ╔═╡ 08150003-0000-4000-8000-000000000000
begin
    sim_scan4_poly_fbp = nothing  # TODO: poly FBP recon (use μ_water_120)
    sim_scan4_poly_hir = nothing  # TODO: poly HIR recon (use μ_water_120)
end

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

# ╔═╡ 08160003-0000-4000-8000-000000000000
# Measurements: Poly FBP only (expand later with HIR, bins, VMI)
sim_measurements_scan2 = let
    results = []
    push!(
        results, measure_scan(
            sim_scan2_poly_fbp, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_poly_fbp"; fov_cm = sim_fov_cm
        )
    )
    results
end

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
# HU scatter: Clinical FBP vs Simulated Poly FBP (Ca and I rods)
let
    # Clinical measurement (140 kVp / 174 mA / FBP — index 2 in clinical_measurements)
    cm = clinical_measurements[2]   # 140kVp_174mA_FBP
    sm = sim_measurements_scan2[1]  # scan2_poly_fbp

    fig = CM.Figure(size = (750, 900), fontsize = 11)

    # --- Top: Calcium rods ---
    ax_ca = CM.Axis(
        fig[1, 1], title = "Calcium Rods", subtitle = "Clinical vs Simulated FBP",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    ca_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "Ca")]
    if !isempty(ca_idx)
        ca_clin = cm.rod_means[ca_idx]
        ca_sim = sm.rod_means[ca_idx]
        CM.scatter!(ax_ca, ca_clin, ca_sim, color = :steelblue, markersize = 10, label = "140 kVp 174 mA")
        CM.lines!(ax_ca, [-100, 1400], [-100, 1400], color = :gray60, linestyle = :dash, label = "Unity (y=x)")
        if length(ca_clin) > 1
            b, m_ = hcat(ones(length(ca_clin)), ca_clin) \ ca_sim
            r = cor(ca_clin, ca_sim)
            rmse = sqrt(sum((ca_sim .- ca_clin) .^ 2) / length(ca_clin))
            nrmse = rmse / (maximum(ca_clin) - minimum(ca_clin)) * 100
            eq = "y = $(round(m_, digits = 3))x $(b >= 0 ? "+" : "-") $(round(abs(b), digits = 1))"
            CM.lines!(ax_ca, [extrema(ca_clin)...], m_ .* [extrema(ca_clin)...] .+ b, color = :black, linewidth = 0.8, label = "Linear fit")
            CM.poly!(ax_ca, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)], color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
            CM.text!(ax_ca, 0.62, 0.18, text = "$(eq)\nr = $(round(r, digits = 4))\nnRMSE = $(round(nrmse, digits = 1))%", space = :relative, align = (:left, :top), fontsize = 10)
        end
        CM.axislegend(ax_ca, position = :lt, labelsize = 9)
    end

    # --- Bottom: Iodine rods ---
    ax_i = CM.Axis(
        fig[2, 1], title = "Iodine Rods", subtitle = "Clinical vs Simulated FBP",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    i_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "I ")]
    if !isempty(i_idx)
        i_clin = cm.rod_means[i_idx]
        i_sim = sm.rod_means[i_idx]
        CM.scatter!(ax_i, i_clin, i_sim, color = :orangered, markersize = 10, label = "140 kVp 174 mA")
        CM.lines!(ax_i, [-50, 500], [-50, 500], color = :gray60, linestyle = :dash, label = "Unity (y=x)")
        if length(i_clin) > 1
            b, m_ = hcat(ones(length(i_clin)), i_clin) \ i_sim
            r = cor(i_clin, i_sim)
            rmse = sqrt(sum((i_sim .- i_clin) .^ 2) / length(i_clin))
            nrmse = rmse / (maximum(i_clin) - minimum(i_clin)) * 100
            eq = "y = $(round(m_, digits = 3))x $(b >= 0 ? "+" : "-") $(round(abs(b), digits = 1))"
            CM.lines!(ax_i, [extrema(i_clin)...], m_ .* [extrema(i_clin)...] .+ b, color = :black, linewidth = 0.8, label = "Linear fit")
            CM.poly!(ax_i, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)], color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
            CM.text!(ax_i, 0.62, 0.18, text = "$(eq)\nr = $(round(r, digits = 4))\nnRMSE = $(round(nrmse, digits = 1))%", space = :relative, align = (:left, :top), fontsize = 10)
        end
        CM.axislegend(ax_i, position = :lt, labelsize = 9)
    end

    CM.save(joinpath(RESULTS_DIR, "alpha_fbp_scatter_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08170003-0000-4000-8000-000000000000
# TODO: NPS comparison FBP vs HIR — uncomment when HIR is enabled
# nothing

# ╔═╡ 08170004-0000-4000-8000-000000000000
# TODO: MTF comparison FBP vs HIR — uncomment when HIR is enabled
# nothing

# ╔═╡ 08170005-0000-4000-8000-000000000000
# TODO: Line profiles FBP vs HIR — uncomment when HIR is enabled
# nothing

# ╔═╡ 08180001-0000-4000-8000-000000000000
md"""
## 18. Export Results
"""

# ╔═╡ 08180002-0000-4000-8000-000000000000
# Export all measurements to CSV + JLD2
let
    # CSV: rod HU means, stds, CNR for each scan/recon combo
    header = [
        "scan_name",
        [["$(n)_mean", "$(n)_std", "$(n)_cnr"] for n in sim_measurements_scan2[1].rod_names]...,
        "nps_peak_freq", "nps_area", "mtf_f50", "mtf_f10",
    ]
    header_flat = vcat(header[1], reduce(vcat, header[2:(end - 4)]), header[(end - 3):end])

    rows = []
    for m in sim_measurements_scan2
        row = Any[m.name]
        for i in 1:length(m.rod_names)
            push!(row, round(m.rod_means[i], digits = 2))
            push!(row, round(m.rod_stds[i], digits = 2))
            push!(row, round(m.rod_cnr[i], digits = 2))
        end
        push!(row, round(m.nps_peak_freq, digits = 2))
        push!(row, round(m.nps_area, digits = 2))
        push!(row, round(m.mtf_f50, digits = 2))
        push!(row, round(m.mtf_f10, digits = 2))
        push!(rows, row)
    end

    csv_path = joinpath(RESULTS_DIR, "naeotom_alpha_scan2_measurements.csv")
    open(csv_path, "w") do io
        println(io, join(header_flat, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end

    # JLD2: NPS + MTF curves
    JLD2.jldsave(
        joinpath(RESULTS_DIR, "naeotom_alpha_scan2_nps.jld2");
        Dict(
            m.name => (freq = m.nps.frequencies, nps = m.nps.nps_1d, nnps = m.nps.nnps_1d)
                for m in sim_measurements_scan2
        )...
    )

    JLD2.jldsave(
        joinpath(RESULTS_DIR, "naeotom_alpha_scan2_mtf.jld2");
        Dict(
            m.name => (freq = m.mtf.frequencies, mtf = m.mtf.mtf)
                for m in sim_measurements_scan2
        )...
    )

    md"**Exported to:** `$(RESULTS_DIR)`"
end

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
| IR | QIR strength 3 |
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
# ╟─08010030-0000-4000-8000-000000000000
# ╠═08010001-0000-4000-8000-000000000000
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
# ╠═08070005-0000-4000-8000-000000000000
# ╟─08070006-0000-4000-8000-000000000000
# ╠═08070007-0000-4000-8000-000000000000
# ╟─08070008-0000-4000-8000-000000000000
# ╠═08070009-0000-4000-8000-000000000000
# ╟─08070010-0000-4000-8000-000000000000
# ╠═08070011-0000-4000-8000-000000000000
# ╟─08070012-0000-4000-8000-000000000000
# ╠═08070013-0000-4000-8000-000000000000
# ╟─08070014-0000-4000-8000-000000000000
# ╠═08070015-0000-4000-8000-000000000000
# ╟─08070016-0000-4000-8000-000000000000
# ╠═08070017-0000-4000-8000-000000000000
# ╟─08070018-0000-4000-8000-000000000000
# ╠═08070019-0000-4000-8000-000000000000
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
# ╠═08090008-0000-4000-8000-000000000000
# ╟─08100001-0000-4000-8000-000000000000
# ╠═08100002-0000-4000-8000-000000000000
# ╟─08100003-0000-4000-8000-000000000000
# ╠═08100004-0000-4000-8000-000000000000
# ╟─08100005-0000-4000-8000-000000000000
# ╟─08110001-0000-4000-8000-000000000000
# ╠═08110002-0000-4000-8000-000000000000
# ╟─08120002-b000-4000-8000-000000000001
# ╟─08120001-0000-4000-8000-000000000000
# ╠═08120002-0000-4000-8000-000000000000
# ╟─08120003-0000-4000-8000-000000000000
# ╠═08120004-0000-4000-8000-000000000000
# ╠═08120004-a000-4000-8000-000000000000
# ╟─08120004-a000-4000-8000-000000000001
# ╠═08120004-a000-4000-8000-000000000002
# ╟─08120005-0000-4000-8000-000000000000
# ╠═08120006-0000-4000-8000-000000000000
# ╠═08120007-0000-4000-8000-000000000000
# ╟─08120008-0000-4000-8000-000000000000
# ╠═08120009-0000-4000-8000-000000000000
# ╠═08120010-0000-4000-8000-000000000000
# ╟─08120011-0000-4000-8000-000000000000
# ╠═08120012-0000-4000-8000-000000000000
# ╟─08120013-0000-4000-8000-000000000000
# ╟─08120014-0000-4000-8000-000000000000
# ╟─08120015-0000-4000-8000-000000000000
# ╟─08120016-0000-4000-8000-000000000000
# ╟─08126001-0000-4000-8000-000000000000
# ╠═08126002-0000-4000-8000-000000000000
# ╠═285deb3d-9c3b-4fe2-8640-ac286f37ae47
# ╠═08126003-0000-4000-8000-000000000000
# ╠═08126004-0000-4000-8000-000000000000
# ╟─08126005-0000-4000-8000-000000000000
# ╟─08126006-0000-4000-8000-000000000000
# ╟─08126007-0000-4000-8000-000000000000
# ╟─08126008-0000-4000-8000-000000000000
# ╟─08130001-0000-4000-8000-000000000000
# ╠═08130002-0000-4000-8000-000000000000
# ╠═08130003-0000-4000-8000-000000000000
# ╟─08140001-0000-4000-8000-000000000000
# ╠═08140002-0000-4000-8000-000000000000
# ╠═08140003-0000-4000-8000-000000000000
# ╟─08150001-0000-4000-8000-000000000000
# ╠═08150002-0000-4000-8000-000000000000
# ╠═08150003-0000-4000-8000-000000000000
# ╟─08160001-0000-4000-8000-000000000000
# ╠═08160002-0000-4000-8000-000000000000
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
