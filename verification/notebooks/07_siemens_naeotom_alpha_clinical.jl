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

# ╔═╡ d335e1a7-cb56-44a9-ad4a-c54f3e1fc280
using Unitful

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
    # VMI QIR3-reconstructed (`VMI/3/...`) — existing naming: hu_140_mid_vmi{E}
    dcms_140_mid_vmi40_qir3 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/40"))
    dcms_140_mid_vmi70_qir3 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/70"))
    dcms_140_mid_vmi100_qir3 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/100"))
    dcms_140_mid_vmi140_qir3 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/3/140"))
    hu_140_mid_vmi40_qir3 = load_hu_volume(dcms_140_mid_vmi40_qir3)
    hu_140_mid_vmi70_qir3 = load_hu_volume(dcms_140_mid_vmi70_qir3)
    hu_140_mid_vmi100_qir3 = load_hu_volume(dcms_140_mid_vmi100_qir3)
    hu_140_mid_vmi140_qir3 = load_hu_volume(dcms_140_mid_vmi140_qir3)
    # VMI FBP-reconstructed (`VMI/0/...`) — `_qir0` suffix (QIR0 = FBP)
    dcms_140_mid_vmi40_qir0  = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/0/40"))
    dcms_140_mid_vmi70_qir0  = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/0/70"))
    dcms_140_mid_vmi100_qir0 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/0/100"))
    dcms_140_mid_vmi140_qir0 = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_174mA_10.12mGyCTDI/VMI/0/140"))
    hu_140_mid_vmi40_qir0  = load_hu_volume(dcms_140_mid_vmi40_qir0)
    hu_140_mid_vmi70_qir0  = load_hu_volume(dcms_140_mid_vmi70_qir0)
    hu_140_mid_vmi100_qir0 = load_hu_volume(dcms_140_mid_vmi100_qir0)
    hu_140_mid_vmi140_qir0 = load_hu_volume(dcms_140_mid_vmi140_qir0)
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
        # VMI at 3 mGy (140 kVp / 52 mA) — QIR3-reconstructed
        (hu_140_low_vmi40,  "140kVp_52mA_VMI40_QIR3"),
        (hu_140_low_vmi70,  "140kVp_52mA_VMI70_QIR3"),
        (hu_140_low_vmi100, "140kVp_52mA_VMI100_QIR3"),
        (hu_140_low_vmi140, "140kVp_52mA_VMI140_QIR3"),
        # VMI at 10 mGy (140 kVp) — QIR3-reconstructed
        (hu_140_mid_vmi40_qir3, "140kVp_174mA_VMI40_QIR3"),
        (hu_140_mid_vmi70_qir3, "140kVp_174mA_VMI70_QIR3"),
        (hu_140_mid_vmi100_qir3, "140kVp_174mA_VMI100_QIR3"),
        (hu_140_mid_vmi140_qir3, "140kVp_174mA_VMI140_QIR3"),
        # VMI at 10 mGy (140 kVp) — FBP-reconstructed
        (hu_140_mid_vmi40_qir0,  "140kVp_174mA_VMI40_QIR0"),
        (hu_140_mid_vmi70_qir0,  "140kVp_174mA_VMI70_QIR0"),
        (hu_140_mid_vmi100_qir0, "140kVp_174mA_VMI100_QIR0"),
        (hu_140_mid_vmi140_qir0, "140kVp_174mA_VMI140_QIR0"),
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
    vols = [hu_140_mid_vmi40_qir3, hu_140_mid_vmi70_qir3, hu_140_mid_vmi100_qir3, hu_140_mid_vmi140_qir3]
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

    # Dose levels — matched exactly to clinical acquisitions.
    sim_mA_scan1 = 52.0    # clinical 140 kVp / 52 mA / 3.03 mGy
    sim_mA_scan2 = 174.0   # clinical 140 kVp / 174 mA / 10.12 mGy
    sim_mA_scan3 = 347.0   # clinical 140 kVp / 347 mA / 20.25 mGy
    sim_mA_scan4 = 253.0   # clinical 120 kVp / 253 mA / 10.15 mGy
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

# ╔═╡ 8d4be3af-9475-4378-a318-0fbe03f07663
sim_custom_poly_filter = BS.CustomFilter(
  (0.0, 0.25, 0.5, 0.75, 1.0),
  # (1.0, 0.75, 0.6, 0.2, 0.001),
  (1.0, 0.70, 0.08, 0.002, 0.0001),
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

Full PCCT simulation with energy-resolved sinograms and material decomposition.
Same pipeline as Scan 2 with lower mA (52 mA → 3.03 mGy CTDI).  Pipeline
**Cong → RWLS → Mono+ → HIR**.  Scan-1-specific hyperparameters live in
the `*_s1` config cells below.
"""

# ╔═╡ 08130002-0000-4000-8000-000000000000
# Scan 1: 140 kVp / 52 mA / ~3 mGy — full PCCT simulation, same pattern as sim_scan2.
sim_scan1 = let
    prot = BS.CTProtocol(
        kVp = 140.0,
        mA = sim_mA_scan1,
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

# ╔═╡ 08131002-b000-4000-8000-000000000001
# Diagnostic: I0 values and per-bin sinogram stats
let
    I0 = sim_scan1.I0_bins
    bins = sim_scan1.bins

    println("I0 per bin: ", round.(I0, sigdigits=4))
    println("I0 total: ", round(sum(I0), sigdigits=4))
    println()

    for (b, s) in enumerate(bins)
        println("Bin $b: min=$(round(minimum(s), digits=4)) max=$(round(maximum(s), digits=4)) mean=$(round(mean(s), digits=4)) I0=$(round(I0[b], sigdigits=4))")
    end
end

# ╔═╡ 08131001-0000-4000-8000-000000000000
md"""
### Poly
"""

# ╔═╡ 08131003-0000-4000-8000-000000000000
md"**Polyenergetic HIR (strength=3)** — iterative counterpart to the poly FBP; matches the row-4 Poly cell of the Scan 1 comparison grids below."

# ╔═╡ 08131005-a000-4000-8000-000000000002
# Per-bin scatter correction: estimate from combined, subtract from each bin.
#
# DEBUG TOGGLE: set `disable_scatter_correction_s1 = true` to skip scatter
# correction entirely (use raw bins).  Tests whether body-envelope ring
# artifacts at 40 keV come from scatter mis-correction at the body boundary.
disable_scatter_correction_s1 = false

sim_scan1_bins_corrected = let
    bins_raw = sim_scan1.bins
    I0_bins = sim_scan1.I0_bins
    I0_total = Float32(sum(I0_bins))
    eps = Float32(1e-10)

    if disable_scatter_correction_s1
        @info "[scatter correction DISABLED — passing raw bins through]"
        [Float32.(b) for b in bins_raw]
    else
        combined = zeros(Float32, size(bins_raw[1]))
        for (b, bin_sino) in enumerate(bins_raw)
            I0b = Float32(I0_bins[b])
            @. combined += I0b * exp(-bin_sino)
        end
        @. combined = -log(max(combined, eps) / I0_total)

        voxel_size_mm = sim_phantom_cpu.voxel_size .* 10.0
        phantom_diam_cm = BS.estimate_phantom_diameter_cm(sim_phantom_cpu.mask, voxel_size_mm)
        scatter_model = BS.geometry_aware_scatter_model(sim_scanner; phantom_diameter_cm=phantom_diam_cm)

        scatter_field = similar(combined)
        BS.estimate_scatter_field!(scatter_field, combined, scatter_model)

        @info "Scatter field: mean=$(round(mean(scatter_field), sigdigits=3)), max=$(round(maximum(scatter_field), sigdigits=3))"

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

        for b in 1:length(bins_raw)
            Δ = mean(Float64.(bins_corrected[b]) .- Float64.(bins_raw[b]))
            @info "Bin $b scatter correction: Δmean_p=$(round(Δ, sigdigits=3)), frac=$(round(scatter_fracs[b], digits=3))"
        end

        bins_corrected
    end
end;

# ╔═╡ 08131002-0000-4000-8000-000000000000
# Poly FBP — recombined from the per-bin scatter-corrected sinograms.
sim_scan1_poly_combined = let
    I0       = sim_scan1.I0_bins
    I0_total = Float32(sum(I0))
    combined = zeros(Float32, size(sim_scan1_bins_corrected[1]))
    for (b, h) in enumerate(sim_scan1_bins_corrected)
        @. combined += Float32(I0[b]) * exp(-h)
    end
    @. combined = -log(max(combined, Float32(1e-10)) / I0_total)
    combined
end;

# ╔═╡ 08131002-a000-4000-8000-000000000001
sim_scan1_poly_fbp = let
    geom = sim_scan1.geom
    recon_size = sim_matrix_size

    sino_gpu = MtlArray(sim_scan1_poly_combined)
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = sim_custom_poly_filter)
    recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_fdk = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end;

# ╔═╡ 08131004-0000-4000-8000-000000000000
# Poly HIR — same recombined scatter-corrected combined sinogram as poly FBP.
sim_scan1_poly_hir = let
    geom = sim_scan1.geom
    recon_size = sim_matrix_size

    sino_gpu = MtlArray(sim_scan1_poly_combined)
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = 3, filter = sim_custom_poly_filter)
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
    recon_μ = Array(ws_hir.volume)

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    ws_hir = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end;

# ╔═╡ 08131005-0000-4000-8000-000000000000
md"""
**Low/high bin combination.** Same as Scan 2 — combine 4 PCCT threshold bins
into 2 effective sinograms: low = bins 1+2 (20–55 keV), high = bins 3+4 (>55 keV).
"""

# ╔═╡ 08131006-a000-4000-8000-000000000001
# ── PCCT 4-bin → 2-bin grouping for Cong + PWLS (scan 1) ──
# Default: low = bins 1+2 (20–55 keV), high = bins 3+4 (>55 keV).  Adjust to
# explore other splits — e.g. [[1], [2, 3, 4]] for a more aggressive separation.
pcct_lohi_grouping_s1 = [[1, 2], [3, 4]];

# ╔═╡ 08131006-0000-4000-8000-000000000000
sim_scan1_lohi = let
    bins = sim_scan1_bins_corrected
    I0   = sim_scan1.I0_bins

    function combine_bins(bin_indices, bins, I0)
        I0_sum = sum(I0[b] for b in bin_indices)
        counts = zeros(Float32, size(bins[1]))
        for b in bin_indices
            @. counts += Float32(I0[b]) * exp(-bins[b])
        end
        @. -log(max(counts, Float32(1e-10)) / Float32(I0_sum))
    end

    grp_low  = pcct_lohi_grouping_s1[1]
    grp_high = pcct_lohi_grouping_s1[2]

    sino_low  = combine_bins(grp_low,  bins, I0)
    sino_high = combine_bins(grp_high, bins, I0)
    I0_low    = sum(I0[b] for b in grp_low)
    I0_high   = sum(I0[b] for b in grp_high)

    @info "[scan 1] low  bin grp=$(grp_low):  I0=$(round(I0_low, sigdigits=4)),  mean sino=$(round(mean(sino_low),  digits=3))"
    @info "[scan 1] high bin grp=$(grp_high): I0=$(round(I0_high, sigdigits=4)), mean sino=$(round(mean(sino_high), digits=3))"

    (sino_low = sino_low, sino_high = sino_high, I0_low = I0_low, I0_high = I0_high)
end;

# ╔═╡ 08131006-b000-4000-8000-000000000001
# Diagnostic: check for negative line integrals
let
    sl = sim_scan1_lohi.sino_low
    sh = sim_scan1_lohi.sino_high
    n_neg_low = count(x -> x < 0, sl)
    n_neg_high = count(x -> x < 0, sh)
    n_total = length(sl)
    min_low = minimum(sl); min_high = minimum(sh)
    @info "sino_low:  $(n_neg_low)/$(n_total) negative ($(round(100*n_neg_low/n_total, digits=2))%), min=$(round(min_low, digits=4))"
    @info "sino_high: $(n_neg_high)/$(n_total) negative ($(round(100*n_neg_high/n_total, digits=2))%), min=$(round(min_high, digits=4))"
end

# ╔═╡ 08131005-a000-4000-8000-000000000001
md"""
### VMI

**Per-bin scatter correction (decoupled).** Same procedure as Scan 2: `simulate!`
injects scatter into per-bin sinograms (for correct Poisson noise statistics)
and returns the scatter field + per-bin weights for exact model-based subtraction.
"""

# ╔═╡ 08131009-0000-4000-8000-000000000000
md"""
**Material decomposition + VMI synthesis (Scan 1).** Active chain: **bin grouping → Cong → PWLS-L₂ → VMI(per-E FBP apod) → Mono+ → HIR (optional)**.

Cong (per-ray analytic decomp on the 2-bin lo/hi grouping) is the canonical `(sino_iodine, sino_water)` pair — fed directly into PWLS as warm start.  PWLS-L₂ (Long/Fessler 2014, 2×2 matrix curvature) polishes the basis sinograms.  Per-energy VMI synthesis runs FBP with a per-energy apodization curve, then Mono+ frequency-split polish, then optional HIR warm-started from Mono+ output.
"""

# ╔═╡ 08131e02-0000-4000-8000-000000000001
md"""
**Cong 2022 (inline).** Per-ray analytic DE decomposition in a **photoelectric + Compton** basis.  `μ(E) = p(E)·a(r) + q(E)·c(r)` where `(a, c)` are the two basis paths and `p(E), q(E)` are shared tables.

⚠ **Why empirical p/q, not the Cong Eq 3c/3d closed forms?**  The analytical `BS.p_photoelectric(E)` / `BS.q_compton(E)` formulas are schematic — they underpredict real water μ(E) by ~60 % at 40 keV.  Cong's outer Brent on water path `L` then never hits `T_L_meas` and every ray falls through to the `(0, 0)` failure branch → decomp returns identically zero.  Fix: fit `(p_L, q_L)` and `(p_H, q_H)` per spectral bin by least-squares so `p·a_w + q·c_w = μ_water(E)` and `p·a_I + q·c_I = μ_iodine(E)` **exactly**.  Same trick as `00_example_ge_gammex_phantom_dual.jl`.

Inline so we can tweak rigorously:
1. **PCCT Cong basis (empirical p/q)** — low = bins 1+2 (20–55 keV), high = bins 3+4 (>55 keV).  Per-bin spectrum from `BS.pcct_effective_spectrum`; `p(E), q(E)` from the 2×2 fit against (water, iodine).
2. **`BS.apply_cong!`** — outer Brent on water-L → inner Newton on Eq 8 quintic → outer Brent on `y`.  Output: (sino_y, sino_c) = `(∫a·dr, ∫c·dr)`.
3. **Exact basis change (y, c) → (t_I, t_W)** — because p/q were fit to reproduce (water, iodine) exactly, the mapping is the physical 2×2 `[a_I a_W; c_I c_W]^-1` (no reference-energy tradeoff).

`cong_decomp_s1` is the canonical (sino_iodine, sino_water) for Scan 1 — fed directly into RWLS as warm start.
"""

# ╔═╡ 08131e02-0000-4000-8000-000000000002
# Cong config — all hyperparams exposed for tuning.
begin
    cong_run_on_cpu        = false      # true → run apply_cong! on CPU (isolates Metal issues)
    cong_use_bowtie        = true       # true → per-ray 3D ŵ[col,row,E] (bowtie-aware); false → 1D centered
    cong_low_bins          = [1, 2]     # low PCCT channel
    cong_high_bins         = [3, 4]     # high PCCT channel
    cong_newton_max_iter   = 5          # inner Newton iters (match notebook 00's working setup)
    cong_newton_tol        = eps(Float32)
    cong_y_max_factor      = 0.25       # safety factor on outer Brent y_max
    cong_y_max_cap         = Float32(1e7)
end

# ╔═╡ 08131e02-0000-4000-8000-000000000003
# Cong PCCT basis — per-bin-group spectrum × DIRECT material mass attenuation
# (p = μρ_iodine, q = μρ_water).  Same setup as notebook 06's working DE-kVp
# Cong: the basis IS material, so apply_cong! outputs land directly in
# (sino_iodine, sino_water) line integrals (g/cm²) — no A_coef / M_inv
# transform.  The earlier photo/Compton LSQ fit (`fit_pq` + `A_coef`) cannot
# represent iodine's K-edge discontinuity at 33.2 keV with a smooth (p, q)
# pair; the residual was exponentially amplified by ∫ŵ·exp(-p·a-q·c)·dE,
# Brent/Newton converged to wrong roots, and M_inv smeared the error across
# iodine — exactly the body-envelope-shaped iodine noise we kept seeing.
cong_basis_s1 = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e_full, ŵ_bins = BS.pcct_effective_spectrum(
        sim_scanner, prot;
        sim_opts   = sim_opts,
        bin_groups = [cong_low_bins, cong_high_bins],
    )

    μρ_iodine(E) = BS.compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(E))
    μρ_water(E)  = BS.compute_mass_μ_at_energy(XA.Materials.water,  Float64(E))
    p_L = Float32[Float32(μρ_iodine(e)) for e in e_full]   # iodine mass atten
    q_L = Float32[Float32(μρ_water(e))  for e in e_full]   # water  mass atten
    p_H = p_L                                               # same energy grid
    q_H = q_L

    # ── Assemble spectral weights (1D centered or 3D bowtie-aware) ──
    # `cong_use_bowtie = true` replaces the centered 1D ŵ with a per-ray 3D
    # ŵ[col, row, E] = w_src(E) · B(col, row, E) · η(E) · Σ_{b∈grp} DRM(E → b),
    # normalized Σ_E ŵ = 1 per ray.  This matches what every ray actually saw
    # through the bowtie — removing a systematic bias at off-iso voxels where
    # the centered spectrum is too soft.  apply_cong! dispatches on ndims(ŵ).
    if !cong_use_bowtie
        # Centered 1D — identical to before.  Defensive renormalization stays.
        ŵ_L = Float32.(Float64.(ŵ_bins[1]) ./ sum(Float64.(ŵ_bins[1])))
        ŵ_H = Float32.(Float64.(ŵ_bins[2]) ./ sum(Float64.(ŵ_bins[2])))
        @info "[Cong basis]  1D centered spectrum (no bowtie)   nE=$(length(e_full))"
    else
        # Bowtie-aware 3D.  Build ŵ_src×B per ray (Σ_E = 1), then fold in
        # detector QE × DRM column sum per bin group, then renormalize per ray.
        _, w_src = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
        ŵ_srcB   = BS.apply_bowtie_to_spectrum(w_src, e_full, sim_scanner, sim_scan1.geom, prot;
                                                include_bowtie = true, label = "Cong")
        pcct_det = BS._build_pcct_detector(sim_scanner)
        kVp      = Float64(maximum(e_full))
        R_mat    = BS.compute_mc_drm(pcct_det, kVp)
        η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        n_R      = size(R_mat, 1)
        drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        n_E      = length(e_full)
        drm_col_sum = [sum(R_mat[drm_row(e_full[i]), b] for b in cong_low_bins)  for i in 1:n_E]
        drm_col_sum_H = [sum(R_mat[drm_row(e_full[i]), b] for b in cong_high_bins) for i in 1:n_E]

        function _bake_3d(drm_col)
            if ndims(ŵ_srcB) == 3
                n_col, n_row = size(ŵ_srcB, 1), size(ŵ_srcB, 2)
                out = Array{Float32, 3}(undef, n_col, n_row, n_E)
                @inbounds for ci in 1:n_col, ri in 1:n_row
                    s = 0.0
                    for Ei in 1:n_E
                        w = Float64(ŵ_srcB[ci, ri, Ei]) * Float64(η_vec[Ei]) * Float64(drm_col[Ei])
                        out[ci, ri, Ei] = Float32(w)
                        s += w
                    end
                    inv_s = Float32(1.0 / max(s, 1e-20))
                    for Ei in 1:n_E
                        out[ci, ri, Ei] *= inv_s
                    end
                end
                out
            else
                # No bowtie on scanner; emit 1D vector so apply_cong! stays 1D.
                w1d = [Float64(ŵ_srcB[i]) * Float64(η_vec[i]) * Float64(drm_col[i]) for i in 1:n_E]
                Float32.(w1d ./ sum(w1d))
            end
        end

        ŵ_L = _bake_3d(drm_col_sum)
        ŵ_H = _bake_3d(drm_col_sum_H)

        if ndims(ŵ_L) == 3
            # Diagnostic: mean E at center vs edge ray to see the bowtie shift.
            mid_c = size(ŵ_L, 1) ÷ 2 + 1
            mid_r = size(ŵ_L, 2) ÷ 2 + 1
            e_ctr_L = sum(Float64(e_full[k]) * ŵ_L[mid_c, mid_r, k] for k in 1:n_E)
            e_ctr_H = sum(Float64(e_full[k]) * ŵ_H[mid_c, mid_r, k] for k in 1:n_E)
            e_edg_L = sum(Float64(e_full[k]) * ŵ_L[1,     mid_r, k] for k in 1:n_E)
            e_edg_H = sum(Float64(e_full[k]) * ŵ_H[1,     mid_r, k] for k in 1:n_E)
            @info "[Cong basis]  3D bowtie-aware   size=$(size(ŵ_L))   ⟨E⟩_L: ctr=$(round(e_ctr_L,digits=2)) edge=$(round(e_edg_L,digits=2)) keV (Δ=$(round(e_edg_L-e_ctr_L,digits=2)))"
            @info "[Cong basis]                                      ⟨E⟩_H: ctr=$(round(e_ctr_H,digits=2)) edge=$(round(e_edg_H,digits=2)) keV (Δ=$(round(e_edg_H-e_ctr_H,digits=2)))"
        else
            e_m_L = sum(Float64.(e_full) .* Float64.(ŵ_L))
            e_m_H = sum(Float64.(e_full) .* Float64.(ŵ_H))
            @info "[Cong basis]  scanner has no bowtie — fell back to 1D   ⟨E⟩_L=$(round(e_m_L,digits=2)) ⟨E⟩_H=$(round(e_m_H,digits=2)) keV"
        end
    end

    (ŵ_L = ŵ_L, ŵ_H = ŵ_H,
     p_L = p_L, p_H = p_H,
     q_L = q_L, q_H = q_H)
end;

# ╔═╡ 08131e02-0000-4000-8000-000000000033
# DIAGNOSTIC: Sample Cong's per-ray spectrum at 3 radial positions and show
# effective μρ for water and iodine.  This tests whether the bowtie / DRM
# modeling produces physically sensible per-ray spectra — if effective μρ
let
    ŵ_L = cong_basis_s1.ŵ_L;  ŵ_H = cong_basis_s1.ŵ_H
    if ndims(ŵ_L) != 3
        @info "[Cong bowtie diagnostic] scanner has no bowtie (1D ŵ); skipping per-ray sampling"
    else
        n_col, n_row, n_E = size(ŵ_L)
        # Need e_full to compute mean energy.  Recompute (matches cong_basis_s1).
        prot   = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
        e_full, _ = BS.pcct_effective_spectrum(sim_scanner, prot;
                                                sim_opts = sim_opts,
                                                bin_groups = [cong_low_bins, cong_high_bins])

        # Sample at 3 radial positions: edge (col=1), mid (col=n_col/4), center (col=n_col/2+1)
        r_mid = n_row ÷ 2 + 1
        positions = [
            ("edge",   1),
            ("mid",    n_col ÷ 4),
            ("center", n_col ÷ 2 + 1),
        ]

        @info "[Cong bowtie diagnostic] per-ray spectrum @ row=$(r_mid), n_E=$(n_E)"
        @info "                              kVp=140, low_bins=$(cong_low_bins), high_bins=$(cong_high_bins)"

        # Material μρ tables (cm²/g), used for effective μρ at each ray.
        μρ_W_vec = Float64[BS.compute_mass_μ_at_energy(XA.Materials.water,   Float64(E)) for E in e_full]
        μρ_I_vec = Float64[BS.compute_mass_μ_at_energy(XA.Elements.Iodine,   Float64(E)) for E in e_full]

        for (label, c) in positions
            ŵ_L_ray = Float64.(Array(ŵ_L)[c, r_mid, :])
            ŵ_H_ray = Float64.(Array(ŵ_H)[c, r_mid, :])
            ŵ_L_sum = sum(ŵ_L_ray);  ŵ_H_sum = sum(ŵ_H_ray)
            ŵ_L_norm = ŵ_L_ray ./ max(ŵ_L_sum, 1e-30)
            ŵ_H_norm = ŵ_H_ray ./ max(ŵ_H_sum, 1e-30)
            mean_E_L = sum(Float64.(e_full) .* ŵ_L_norm)
            mean_E_H = sum(Float64.(e_full) .* ŵ_H_norm)
            μρ_W_L_eff = sum(μρ_W_vec .* ŵ_L_norm)   # effective water μρ in low bin at this ray
            μρ_W_H_eff = sum(μρ_W_vec .* ŵ_H_norm)
            μρ_I_L_eff = sum(μρ_I_vec .* ŵ_L_norm)
            μρ_I_H_eff = sum(μρ_I_vec .* ŵ_H_norm)

            @info "[$(rpad(label, 6))]  col=$(c)  <E>=( L:$(round(mean_E_L, digits=1)),  H:$(round(mean_E_H, digits=1))) keV"
            @info "         μρ_water  eff = ( L:$(round(μρ_W_L_eff, sigdigits=3)),  H:$(round(μρ_W_H_eff, sigdigits=3))) cm²/g"
            @info "         μρ_iodine eff = ( L:$(round(μρ_I_L_eff, sigdigits=3)),  H:$(round(μρ_I_H_eff, sigdigits=3))) cm²/g"
            @info "         L/H μρ ratio  = ( water: $(round(μρ_W_L_eff/μρ_W_H_eff, digits=2))×,  iodine: $(round(μρ_I_L_eff/μρ_I_H_eff, digits=2))×)"
        end

        @info "─── interpretation ────────────────────────────────────────────────"
        @info "  Bowtie hardens spectrum at edge → <E> at edge > center."
        @info "  Iodine L/H ratio drops at edge (less K-edge contrast).  This is what"
        @info "  the per-ray Cong solver sees.  If center vs edge values look wrong"
        @info "  (e.g., μρ_W differs by >50%, or iodine L/H ratio < 1.5 anywhere),"
        @info "  there's a bowtie/DRM modeling bug."
    end
end;

# ╔═╡ 08131e02-0000-4000-8000-000000000004
# Cong per-ray decomp (GPU).  Basis is material-direct (p=μρ_iodine,
# q=μρ_water), so apply_cong! writes directly into sino_iodine / sino_water
# (g/cm² line integrals).  water_basis = (a=0, c=1): "water in this basis"
# is (0, 1), i.e. Cong's outer water-path Brent solves
# ∫Ŝ_L(ε)·exp(-μρ_water(ε)·L) dε = T_L_meas → L is the water line integral.
cong_decomp_s1 = let
    # ── Diagnostic: inputs Cong will see ──
    sl_cpu = Float32.(sim_scan1_lohi.sino_low)
    sh_cpu = Float32.(sim_scan1_lohi.sino_high)
    n_air_low  = count(<(1f-6), sl_cpu)
    n_air_high = count(<(1f-6), sh_cpu)
    @info "[Cong inputs]  sino_low  min=$(round(minimum(sl_cpu),digits=4)) max=$(round(maximum(sl_cpu),digits=4)) mean=$(round(mean(sl_cpu),digits=4))  n<1e-6=$(n_air_low)/$(length(sl_cpu))"
    @info "[Cong inputs]  sino_high min=$(round(minimum(sh_cpu),digits=4)) max=$(round(maximum(sh_cpu),digits=4)) mean=$(round(mean(sh_cpu),digits=4))  n<1e-6=$(n_air_high)/$(length(sh_cpu))"

    # ── Diagnostic: basis tables ──
    p_L = Float64.(cong_basis_s1.p_L);  q_L = Float64.(cong_basis_s1.q_L)
    @info "[Cong basis]  nE=$(length(p_L))   p_L (μρ_iodine) range=[$(round(minimum(p_L),sigdigits=3)), $(round(maximum(p_L),sigdigits=3))] cm²/g   q_L (μρ_water) range=[$(round(minimum(q_L),sigdigits=3)), $(round(maximum(q_L),sigdigits=3))] cm²/g"
    let ŵ = cong_basis_s1.ŵ_L
        if ndims(ŵ) == 3
            rs = dropdims(sum(Float64.(ŵ); dims = 3); dims = 3)
            @info "[Cong basis]  ŵ_L per-ray Σ range = [$(round(minimum(rs), digits=6)), $(round(maximum(rs), digits=6))]  (3D, should all be ~1.0)"
        else
            @info "[Cong basis]  ŵ_L Σ = $(round(sum(ŵ), digits=6))  (1D, should be ~1.0)"
        end
    end

    # water_basis = (0, 1) because the Cong basis IS (iodine, water): the
    # 'a' coordinate selects iodine, 'c' coordinate selects water.
    water_basis = (a = 0.0f0, c = 1.0f0)

    # ── Run apply_cong! (CPU or GPU based on toggle) ──
    if cong_run_on_cpu
        @info "[Cong] running on CPU (cong_run_on_cpu = true)"
        sino_iodine = similar(sl_cpu);  fill!(sino_iodine, 0f0)
        sino_water  = similar(sh_cpu);  fill!(sino_water,  0f0)
        ws_cong_s1 = BS.create_cong_workspace(sl_cpu, cong_basis_s1)
        t0 = time()
        BS.apply_cong!(
            ws_cong_s1, sino_iodine, sino_water, sl_cpu, sh_cpu;
            water_basis      = water_basis,
            newton_max_iter  = cong_newton_max_iter,
            newton_tol       = cong_newton_tol,
            y_max_factor     = cong_y_max_factor,
            y_max_cap        = cong_y_max_cap,
        )
        @info "[Cong] CPU apply_cong! done in $(round(time()-t0, digits=1)) s"
    else
        @info "[Cong] running on Metal GPU"
        sl_gpu = MtlArray(sl_cpu)
        sh_gpu = MtlArray(sh_cpu)
        sI_gpu = similar(sl_gpu);  fill!(sI_gpu, 0f0)
        sW_gpu = similar(sl_gpu);  fill!(sW_gpu, 0f0)
        ws_cong_s1 = BS.create_cong_workspace(sl_gpu, cong_basis_s1)
        t0 = time()
        BS.apply_cong!(
            ws_cong_s1, sI_gpu, sW_gpu, sl_gpu, sh_gpu;
            water_basis      = water_basis,
            newton_max_iter  = cong_newton_max_iter,
            newton_tol       = cong_newton_tol,
            y_max_factor     = cong_y_max_factor,
            y_max_cap        = cong_y_max_cap,
        )
        @info "[Cong] GPU apply_cong! done in $(round(time()-t0, digits=1)) s"
        sino_iodine = Array(sI_gpu)
        sino_water  = Array(sW_gpu)
        sl_gpu = nothing; sh_gpu = nothing; sI_gpu = nothing; sW_gpu = nothing
        ws_cong_s1 = nothing
        GC.gc(true)
    end

    n_I_zero = count(iszero, sino_iodine)
    n_W_zero = count(iszero, sino_water)
    @info "[Cong out]   sino_iodine min=$(round(minimum(sino_iodine),sigdigits=4)) max=$(round(maximum(sino_iodine),sigdigits=4)) mean=$(round(mean(sino_iodine),sigdigits=4))  nzero=$(n_I_zero)/$(length(sino_iodine))"
    @info "[Cong out]   sino_water  min=$(round(minimum(sino_water), sigdigits=4)) max=$(round(maximum(sino_water), sigdigits=4)) mean=$(round(mean(sino_water), sigdigits=4))  nzero=$(n_W_zero)/$(length(sino_water))"
    @info "[Cong decomp]  ⟨∫ρ_I·dr⟩ = $(round(mean(sino_iodine), sigdigits=4)) g/cm²   ⟨∫ρ_W·dr⟩ = $(round(mean(sino_water), sigdigits=4)) g/cm²"

    (sino_iodine = sino_iodine,
     sino_water  = sino_water,
     geom        = sim_scan1.geom)
end;

# ╔═╡ 08131e02-1000-4000-8000-000000000001
md"""
### Cong-PCCT (this paper) — replaces Cong for downstream

Cong et al. 2022 generalized to PCCT with N≥3 effective spectral channels merged from native PCCT bins.  Two-material (water, iodine) decomposition.  Per-ray algorithm:

1. **2×2 linearization** at (0, 0) gives initial `(y₀, z₀)` and sets `z̄ = z₀` (paper §2.3).
2. **Anchor channel** `b*` selected per-ray by spectral leverage `m_{b,1}² / Σ_{bb}` (Eq 22).
3. **Polynomial residual** `T_{b*,K}(x; y) = I_{b*}` solved via Newton; `K = 7` (handles iodine K-edge per §2.4 Table — relative error ≤ 3×10⁻³ at clinical iodine concentrations).
4. **Mahalanobis fit** on the remaining `N − 1` channels: `Φ(y) = Σ_{b≠b*} (I_b − F_b(y))² / σ²_b` with diagonal `Σ` (no pile-up coupling in our simulator).
5. **Brent root-find** on `dΦ/dy = 0` over `[0, y_max]` with closed-form analytic gradient (Eq 27–30, implicit `dh/dy`).

Replaces `cong_decomp_s1` for downstream consumption.  Existing 2-bin Cong cells (`cong_basis_s1`, `cong_decomp_s1`) kept around but unconsumed (idle).
"""

# ╔═╡ 08131e02-1000-4000-8000-000000000002
# Cong-PCCT config — N=3 channel merging (canonical clinical low/mid/high) + tuning knobs.
begin
    cong_pcct_bin_groups_s1     = [[1, 2], [3], [4]]   # N=3 channels: low/mid/high
    cong_pcct_newton_iter_s1    = 6                    # inner Newton on T(x; y) = I_{b*}
    cong_pcct_brent_max_iter_s1 = 100
    cong_pcct_y_max_factor_s1   = 1.5                  # bracket safety factor (paper §2.7)
    cong_pcct_run_on_cpu_s1     = false                # GPU by default
    cong_pcct_use_bowtie_s1     = true                 # 3D ŵ_bins[k][col,row,E] when bowtie present
end

# ╔═╡ 08131e02-1000-4000-8000-000000000003
# Cong-PCCT basis: per-channel bowtie-aware ŵ_bins[k] + material μρ tables.
# Hardcodes K = 7 universally (paper §2.4: K=7 handles iodine K-edge with
# relative error ≤ 3×10⁻³ at clinical concentrations; cost over K=5 is
# negligible since spectral moments are computed once per Brent eval).
cong_pcct_basis_s1 = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e_full, ŵ_bins_centered = BS.pcct_effective_spectrum(
        sim_scanner, prot;
        sim_opts   = sim_opts,
        bin_groups = cong_pcct_bin_groups_s1,
    )

    μρ_water  = Float32[Float32(BS.compute_mass_μ_at_energy(XA.Materials.water,  Float64(E))) for E in e_full]
    μρ_iodine = Float32[Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(E))) for E in e_full]

    n_E = length(e_full)
    if !cong_pcct_use_bowtie_s1
        ŵ_bins = [Float32.(Float64.(ŵ) ./ sum(Float64.(ŵ))) for ŵ in ŵ_bins_centered]
        @info "[Cong-PCCT basis]  1D centered spectrum   N=$(length(ŵ_bins)) channels   nE=$n_E"
    else
        # 3D bowtie-aware: per-ray ŵ_bins[k][col, row, E] = source × bowtie × η × DRM_col_sum, normalized.
        _, w_src = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
        ŵ_srcB   = BS.apply_bowtie_to_spectrum(w_src, e_full, sim_scanner, sim_scan1.geom, prot;
                                                include_bowtie = true, label = "Cong-PCCT")
        pcct_det = BS._build_pcct_detector(sim_scanner)
        kVp_max  = Float64(maximum(e_full))
        R_mat    = BS.compute_mc_drm(pcct_det, kVp_max)
        η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        n_R      = size(R_mat, 1)
        drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp_max - 1.0) * (n_R - 1)) + 1, 1, n_R)

        function _bake_3d(grp)
            drm_col = [sum(R_mat[drm_row(e_full[i]), b] for b in grp) for i in 1:n_E]
            if ndims(ŵ_srcB) == 3
                n_col, n_row = size(ŵ_srcB, 1), size(ŵ_srcB, 2)
                out = Array{Float32, 3}(undef, n_col, n_row, n_E)
                @inbounds for ci in 1:n_col, ri in 1:n_row
                    s = 0.0
                    for Ei in 1:n_E
                        w = Float64(ŵ_srcB[ci, ri, Ei]) * Float64(η_vec[Ei]) * Float64(drm_col[Ei])
                        out[ci, ri, Ei] = Float32(w)
                        s += w
                    end
                    inv_s = Float32(1.0 / max(s, 1e-20))
                    for Ei in 1:n_E
                        out[ci, ri, Ei] *= inv_s
                    end
                end
                out
            else
                # Scanner has no bowtie — emit 1D normalized vector.
                w1d = [Float64(ŵ_srcB[i]) * Float64(η_vec[i]) * Float64(drm_col[i]) for i in 1:n_E]
                Float32.(w1d ./ sum(w1d))
            end
        end

        ŵ_bins = [_bake_3d(grp) for grp in cong_pcct_bin_groups_s1]
        @info "[Cong-PCCT basis]  bowtie-aware ŵ_bins   N=$(length(ŵ_bins)) channels   ŵ shape=$(size(ŵ_bins[1]))"
    end

    @info "  bin groups: $(cong_pcct_bin_groups_s1)   K = 7 (paper §2.4 universal-safe)"

    (ŵ_bins    = ŵ_bins,
     e         = e_full,
     μρ_water  = μρ_water,
     μρ_iodine = μρ_iodine,
     bin_groups = cong_pcct_bin_groups_s1)
end;

# ╔═╡ 08131e02-1000-4000-8000-000000000004
# N-channel sino builder: combine scatter-corrected per-bin sinos into N
# effective channel line integrals (one per bin group), with per-channel I0.
sim_scan1_n_channel_s1 = let
    bins = sim_scan1_bins_corrected
    I0   = sim_scan1.I0_bins

    function combine_bins_n(grp, bins, I0)
        I0_sum = sum(I0[b] for b in grp)
        counts = zeros(Float32, size(bins[1]))
        for b in grp
            @. counts += Float32(I0[b]) * exp(-bins[b])
        end
        @. -log(max(counts, Float32(1e-10)) / Float32(I0_sum))
    end

    h_channels  = [combine_bins_n(grp, bins, I0) for grp in cong_pcct_bin_groups_s1]
    I0_channels = [Float64(sum(I0[b] for b in grp)) for grp in cong_pcct_bin_groups_s1]

    @info "[Cong-PCCT inputs]  N=$(length(h_channels)) channels   sino shape=$(size(h_channels[1]))"
    for k in 1:length(h_channels)
        @info "  channel $k (bins $(cong_pcct_bin_groups_s1[k])):  I0=$(round(I0_channels[k], sigdigits=4))   mean(h)=$(round(mean(h_channels[k]), digits=3))"
    end

    (h_channels = h_channels, I0_channels = I0_channels)
end;

# ╔═╡ 08131e02-1000-4000-8000-000000000005
# Cong-PCCT 3-channel per-ray decomposition kernel (paper §2.3-§2.7).
#
# Per-pixel algorithm:
#   Step 1 (linearize):  ⟨μ_w⟩_b = Σ_E ŵ_b·μ_w,  ⟨μ_I⟩_b = Σ_E ŵ_b·μ_I.  Solve
#       [⟨μ_w⟩_1 ⟨μ_I⟩_1 ; ⟨μ_w⟩_3 ⟨μ_I⟩_3] · (y, z)ᵀ = (h_1, h_3)ᵀ.
#       Set z̄ = max(z_lin, 0).
#
#   Step 2 (anchor):  m_{b,1}(y_lin) = Σ_E ŵ_b·μ_I·exp(-μ_w·y_lin - μ_I·z_lin).
#       σ²_b = I_b/I0_b (Poisson on counts; diagonal Σ — no pile-up coupling).
#       b* = argmax_b m_{b,1}² / σ²_b.
#
#   Step 3 (Brent root):  G(y) = dΦ/dy = -2·Σ_{b≠b*}(I_b - F_b(y))·dF_b/dy / σ²_b.
#       Per y eval (single E-pass at anchor + single E-pass at residuals):
#         (a) m_k = Σ_E ŵ_{b*}·μ_I^k·e_yz̄,  n_k = Σ_E ŵ_{b*}·μ_w·μ_I^k·e_yz̄  for k=0..7
#             (e_yz̄ = exp(-μ_w·y - μ_I·z̄))
#         (b) c_k = (-1)^k/k! · m_k,  dc_k/dy = -(-1)^k/k! · n_k  (Eq 30)
#         (c) Newton on T(x) = Σ_k c_k·x^k = I_{b*}  → x = h_{b*}(y; K=7)
#         (d) F_b = Σ_E ŵ_b·exp(-μ_w·y - μ_I·(z̄+x)),  A_b = +μ_w-weighted,  B_b = +μ_I-weighted
#         (e) dh/dy = -dT/dy / dT/dx;  dF_b/dy = -(A_b + B_b·dh/dy)
#         (f) G(y) = -2·Σ(I_b-F_b)·dF_b/dy/σ²_b
#       Brent finds root y_opt; recover x_opt via Newton at y_opt; output (z̄+x_opt, y_opt).
#
# Hardcoded N=3 (paper canonical worked instance).  K=7 universal (handles K-edge).
function _apply_cong_pcct_s1!(
        sino_iodine::AbstractArray{Float32, 3},
        sino_water::AbstractArray{Float32, 3},
        h_channels::AbstractVector,
        I0_channels::AbstractVector{Float64};
        basis,
        newton_iter::Integer    = 6,
        brent_max_iter::Integer = 100,
        y_max_factor::Real      = 1.5,
        verbose::Bool           = true,
    )
    length(h_channels) == length(I0_channels) ||
        error("_apply_cong_pcct_s1!: h_channels/I0_channels length mismatch.")
    length(h_channels) == 3 ||
        error("_apply_cong_pcct_s1!: this implementation hardcodes N=3 (got $(length(h_channels))).")
    sino_shape = size(sino_iodine)
    sino_shape == size(sino_water) == size(h_channels[1]) ||
        error("_apply_cong_pcct_s1!: sino / h_channels[1] must share shape.")
    n_col = Int(sino_shape[1])
    n_row = Int(sino_shape[2])

    # Stage to backend.
    h1 = BS._match_backend(h_channels[1], sino_iodine)
    h2 = BS._match_backend(h_channels[2], sino_iodine)
    h3 = BS._match_backend(h_channels[3], sino_iodine)
    ŵ1 = BS._match_backend(Float32.(basis.ŵ_bins[1]), sino_iodine)
    ŵ2 = BS._match_backend(Float32.(basis.ŵ_bins[2]), sino_iodine)
    ŵ3 = BS._match_backend(Float32.(basis.ŵ_bins[3]), sino_iodine)
    μρ_w = BS._match_backend(Float32.(basis.μρ_water),  sino_iodine)
    μρ_I = BS._match_backend(Float32.(basis.μρ_iodine), sino_iodine)

    per_ray = ndims(ŵ1) == 3
    nE = per_ray ? size(ŵ1, 3) : length(ŵ1)

    # Capture invariants as locals (avoid Core.Box on AK closure).
    I0_1 = Float32(I0_channels[1]); I0_2 = Float32(I0_channels[2]); I0_3 = Float32(I0_channels[3])
    n_iter_inner = Int(newton_iter)
    brent_iter   = Int(brent_max_iter)
    y_fac        = Float32(y_max_factor)

    t0 = time()
    AK.foreachindex(sino_iodine) do idx
        i0  = idx - 1
        col = (i0 % n_col) + 1
        row = ((i0 ÷ n_col) % n_row) + 1

        h1v = h1[idx];  h2v = h2[idx];  h3v = h3[idx]

        # Air gate.
        if h1v < 1f-6 && h2v < 1f-6 && h3v < 1f-6
            sino_water[idx]  = 0f0
            sino_iodine[idx] = 0f0
            return
        end

        I_1 = exp(-h1v);  I_2 = exp(-h2v);  I_3 = exp(-h3v)
        # σ²_b = Var(I_b) = I_b / I0_b (Poisson on counts → relative variance, diagonal Σ).
        # Floor I_b at 1/I0_b (= 1-photon transmission), which caps inv_σ²_b at I0_b²
        # so a single-photon-starved pixel doesn't get infinite weight in the Mahalanobis fit.
        inv_σ²_1 = I0_1 / max(I_1, 1f0 / I0_1)
        inv_σ²_2 = I0_2 / max(I_2, 1f0 / I0_2)
        inv_σ²_3 = I0_3 / max(I_3, 1f0 / I0_3)

        # ── Step 1: spectral averages + 2×2 linearization ──
        # ⟨μ_w⟩_b, ⟨μ_I⟩_b at zero attenuation — used both for linearization and y_max bracket.
        a11 = 0f0; a12 = 0f0; a21 = 0f0; a31 = 0f0; a32 = 0f0
        if per_ray
            @inbounds for k in 1:nE
                w1k = ŵ1[col, row, k]; w2k = ŵ2[col, row, k]; w3k = ŵ3[col, row, k]
                μw = μρ_w[k]; μI = μρ_I[k]
                a11 += w1k * μw;  a12 += w1k * μI
                a21 += w2k * μw                                     # only μ_w needed for y_max
                a31 += w3k * μw;  a32 += w3k * μI
            end
        else
            @inbounds for k in 1:nE
                w1k = ŵ1[k]; w2k = ŵ2[k]; w3k = ŵ3[k]
                μw = μρ_w[k]; μI = μρ_I[k]
                a11 += w1k * μw;  a12 += w1k * μI
                a21 += w2k * μw
                a31 += w3k * μw;  a32 += w3k * μI
            end
        end
        det_2x2 = a11 * a32 - a12 * a31
        if abs(det_2x2) < 1f-12
            # Singular linearization → fallback: water-only.
            sino_water[idx]  = max(h1v / max(a11, 1f-12), 0f0)
            sino_iodine[idx] = 0f0
            return
        end
        y_lin = (a32 * h1v - a12 * h3v) / det_2x2
        z_lin = (a11 * h3v - a31 * h1v) / det_2x2
        y_lin = max(y_lin, 0f0)
        z_lin = max(z_lin, 0f0)
        z_bar = z_lin

        # ── Step 2: anchor selection (Eq 22) — score_b = m_{b,1}(y_lin)² / σ²_b ──
        m1_1 = 0f0; m1_2 = 0f0; m1_3 = 0f0
        if per_ray
            @inbounds for k in 1:nE
                eYZ = exp(-μρ_w[k]*y_lin - μρ_I[k]*z_lin)
                μI_eYZ = μρ_I[k] * eYZ
                m1_1 += ŵ1[col, row, k] * μI_eYZ
                m1_2 += ŵ2[col, row, k] * μI_eYZ
                m1_3 += ŵ3[col, row, k] * μI_eYZ
            end
        else
            @inbounds for k in 1:nE
                eYZ = exp(-μρ_w[k]*y_lin - μρ_I[k]*z_lin)
                μI_eYZ = μρ_I[k] * eYZ
                m1_1 += ŵ1[k] * μI_eYZ
                m1_2 += ŵ2[k] * μI_eYZ
                m1_3 += ŵ3[k] * μI_eYZ
            end
        end
        score_1 = m1_1 * m1_1 * inv_σ²_1
        score_2 = m1_2 * m1_2 * inv_σ²_2
        score_3 = m1_3 * m1_3 * inv_σ²_3
        b_star = (score_1 ≥ score_2 && score_1 ≥ score_3) ? 1 :
                 (score_2 ≥ score_3 ? 2 : 3)

        # y_max bracket per §2.7: y_max = y_fac · max_b(h_b) / min_b(⟨μ_w⟩_b)
        min_μw = min(a11, min(a21, a31))
        max_h  = max(h1v, max(h2v, h3v))
        y_max  = y_fac * max_h / max(min_μw, 1f-12)

        # ── Pre-bind anchor and residual channel arrays/scalars via ternary ──
        # Each variable is assigned exactly ONCE syntactically (no if/elseif/else
        # multi-branch assignment, which can trip Julia's Core.Box heuristic on
        # closure-captured variables → break Metal compile).
        I_star    = b_star == 1 ? I_1       : (b_star == 2 ? I_2       : I_3)
        ŵ_r1      = b_star == 1 ? ŵ2        : ŵ1
        ŵ_r2      = b_star == 1 ? ŵ3        : (b_star == 2 ? ŵ3        : ŵ2)
        I_r1      = b_star == 1 ? I_2       : I_1
        I_r2      = b_star == 1 ? I_3       : (b_star == 2 ? I_3       : I_2)
        inv_σ²_r1 = b_star == 1 ? inv_σ²_2  : inv_σ²_1
        inv_σ²_r2 = b_star == 1 ? inv_σ²_3  : (b_star == 2 ? inv_σ²_3  : inv_σ²_2)

        # G(y) = dΦ/dy.  Single flat closure (cong.jl solve_quintic pattern):
        # one E-loop for anchor moments, inline Newton on T(x; y) = I_star,
        # one E-loop for residual channels, return G_val::Float32.
        # Capture: ŵ1/ŵ2/ŵ3, μρ_w, μρ_I, z_bar, b_star, I_star, ŵ_r1/ŵ_r2,
        # I_r1/I_r2, inv_σ²_r1/inv_σ²_r2, n_iter_inner, n_col, n_row, nE, per_ray, col, row.
        G = function (y::Float32)
            # ── Step A: anchor channel — accumulate m_k = Σ ŵ_b*·μ_I^k·e_yz̄  and
            #            n_k = Σ ŵ_b*·μ_w·μ_I^k·e_yz̄  for k=0..7  in ONE E-pass.
            m0 = 0f0; m1 = 0f0; m2 = 0f0; m3 = 0f0
            m4 = 0f0; m5 = 0f0; m6 = 0f0; m7 = 0f0
            n0 = 0f0; n1 = 0f0; n2 = 0f0; n3 = 0f0
            n4 = 0f0; n5 = 0f0; n6 = 0f0; n7 = 0f0
            if per_ray
                @inbounds for kk in 1:nE
                    μw = μρ_w[kk]; μI = μρ_I[kk]
                    ŵv = b_star == 1 ? ŵ1[col, row, kk] :
                         b_star == 2 ? ŵ2[col, row, kk] :
                                       ŵ3[col, row, kk]
                    we = ŵv * exp(-μw*y - μI*z_bar)
                    μw_we = μw * we
                    pp = 1f0
                    m0 += we;             n0 += μw_we
                    pp *= μI; m1 += pp*we; n1 += pp*μw_we
                    pp *= μI; m2 += pp*we; n2 += pp*μw_we
                    pp *= μI; m3 += pp*we; n3 += pp*μw_we
                    pp *= μI; m4 += pp*we; n4 += pp*μw_we
                    pp *= μI; m5 += pp*we; n5 += pp*μw_we
                    pp *= μI; m6 += pp*we; n6 += pp*μw_we
                    pp *= μI; m7 += pp*we; n7 += pp*μw_we
                end
            else
                @inbounds for kk in 1:nE
                    μw = μρ_w[kk]; μI = μρ_I[kk]
                    ŵv = b_star == 1 ? ŵ1[kk] :
                         b_star == 2 ? ŵ2[kk] :
                                       ŵ3[kk]
                    we = ŵv * exp(-μw*y - μI*z_bar)
                    μw_we = μw * we
                    pp = 1f0
                    m0 += we;             n0 += μw_we
                    pp *= μI; m1 += pp*we; n1 += pp*μw_we
                    pp *= μI; m2 += pp*we; n2 += pp*μw_we
                    pp *= μI; m3 += pp*we; n3 += pp*μw_we
                    pp *= μI; m4 += pp*we; n4 += pp*μw_we
                    pp *= μI; m5 += pp*we; n5 += pp*μw_we
                    pp *= μI; m6 += pp*we; n6 += pp*μw_we
                    pp *= μI; m7 += pp*we; n7 += pp*μw_we
                end
            end

            # ── Step B: c_k = (-1)^k/k! · m_k ; dc_k/dy = -(-1)^k/k! · n_k (Eq 30) ──
            c0 =  m0;                  dc0 = -n0
            c1 = -m1;                  dc1 =  n1
            c2 =  m2 * 0.5f0;          dc2 = -n2 * 0.5f0
            c3 = -m3 * (1f0/6f0);      dc3 =  n3 * (1f0/6f0)
            c4 =  m4 * (1f0/24f0);     dc4 = -n4 * (1f0/24f0)
            c5 = -m5 * (1f0/120f0);    dc5 =  n5 * (1f0/120f0)
            c6 =  m6 * (1f0/720f0);    dc6 = -n6 * (1f0/720f0)
            c7 = -m7 * (1f0/5040f0);   dc7 =  n7 * (1f0/5040f0)

            # ── Step C: Newton on T(x) = c0 + c1·x + ... + c7·x^7 = I_star  for x ──
            x = 0f0
            for _ in 1:n_iter_inner
                T_  = c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*(c6 + x*c7))))))
                dT_ = c1 + x*(2f0*c2 + x*(3f0*c3 + x*(4f0*c4 + x*(5f0*c5 + x*(6f0*c6 + x*7f0*c7)))))
                if abs(dT_) < 1f-30
                    break
                end
                Δ = (T_ - I_star) / dT_
                x -= Δ
                if abs(Δ) < 1f-7
                    break
                end
            end

            # ── Step D: dT/dx, dT/dy at x; dh/dy via implicit derivative (Eq 29) ──
            dT_dx = c1 + x*(2f0*c2 + x*(3f0*c3 + x*(4f0*c4 + x*(5f0*c5 + x*(6f0*c6 + x*7f0*c7)))))
            dT_dy = dc0 + x*(dc1 + x*(dc2 + x*(dc3 + x*(dc4 + x*(dc5 + x*(dc6 + x*dc7))))))
            dh_dy = abs(dT_dx) < 1f-30 ? 0f0 : -dT_dy / dT_dx

            # ── Step E: residual channels' F_b, A_b, B_b at (y, z̄+x) — single E-pass ──
            zarg = z_bar + x
            F_r1 = 0f0; A_r1 = 0f0; B_r1 = 0f0
            F_r2 = 0f0; A_r2 = 0f0; B_r2 = 0f0
            if per_ray
                @inbounds for kk in 1:nE
                    μw = μρ_w[kk]; μI = μρ_I[kk]
                    eYZ = exp(-μw*y - μI*zarg)
                    we1 = ŵ_r1[col, row, kk] * eYZ
                    we2 = ŵ_r2[col, row, kk] * eYZ
                    F_r1 += we1;  A_r1 += we1*μw;  B_r1 += we1*μI
                    F_r2 += we2;  A_r2 += we2*μw;  B_r2 += we2*μI
                end
            else
                @inbounds for kk in 1:nE
                    μw = μρ_w[kk]; μI = μρ_I[kk]
                    eYZ = exp(-μw*y - μI*zarg)
                    we1 = ŵ_r1[kk] * eYZ
                    we2 = ŵ_r2[kk] * eYZ
                    F_r1 += we1;  A_r1 += we1*μw;  B_r1 += we1*μI
                    F_r2 += we2;  A_r2 += we2*μw;  B_r2 += we2*μI
                end
            end
            dF_r1 = -(A_r1 + B_r1 * dh_dy)
            dF_r2 = -(A_r2 + B_r2 * dh_dy)

            # ── Step F: G(y) = dΦ/dy — single Float32 return ──
            -2f0 * (I_r1 - F_r1) * dF_r1 * inv_σ²_r1 -
                2f0 * (I_r2 - F_r2) * dF_r2 * inv_σ²_r2
        end

        # Brent root-find on G ∈ [0, y_max] with explicit Float32 tolerances.
        y_opt, ok_y = BS.brent_solve(G, 0f0, y_max;
                                     xabstol = 1f-7, xreltol = 1f-5,
                                     maxiters = brent_iter)
        if !ok_y
            sino_water[idx]  = y_lin
            sino_iodine[idx] = z_lin
            return
        end

        # Recover x_opt = h_{b*}(y_opt) via inline Newton (parallel structure to G).
        # Same anchor-moment + Newton block; we don't need n_k or derivatives here.
        let y = y_opt
            m0 = 0f0; m1 = 0f0; m2 = 0f0; m3 = 0f0
            m4 = 0f0; m5 = 0f0; m6 = 0f0; m7 = 0f0
            if per_ray
                @inbounds for kk in 1:nE
                    μw = μρ_w[kk]; μI = μρ_I[kk]
                    ŵv = b_star == 1 ? ŵ1[col, row, kk] :
                         b_star == 2 ? ŵ2[col, row, kk] :
                                       ŵ3[col, row, kk]
                    we = ŵv * exp(-μw*y - μI*z_bar)
                    pp = 1f0
                    m0 += we
                    pp *= μI; m1 += pp*we
                    pp *= μI; m2 += pp*we
                    pp *= μI; m3 += pp*we
                    pp *= μI; m4 += pp*we
                    pp *= μI; m5 += pp*we
                    pp *= μI; m6 += pp*we
                    pp *= μI; m7 += pp*we
                end
            else
                @inbounds for kk in 1:nE
                    μw = μρ_w[kk]; μI = μρ_I[kk]
                    ŵv = b_star == 1 ? ŵ1[kk] :
                         b_star == 2 ? ŵ2[kk] :
                                       ŵ3[kk]
                    we = ŵv * exp(-μw*y - μI*z_bar)
                    pp = 1f0
                    m0 += we
                    pp *= μI; m1 += pp*we
                    pp *= μI; m2 += pp*we
                    pp *= μI; m3 += pp*we
                    pp *= μI; m4 += pp*we
                    pp *= μI; m5 += pp*we
                    pp *= μI; m6 += pp*we
                    pp *= μI; m7 += pp*we
                end
            end
            c0 =  m0
            c1 = -m1
            c2 =  m2 * 0.5f0
            c3 = -m3 * (1f0/6f0)
            c4 =  m4 * (1f0/24f0)
            c5 = -m5 * (1f0/120f0)
            c6 =  m6 * (1f0/720f0)
            c7 = -m7 * (1f0/5040f0)
            x = 0f0
            for _ in 1:n_iter_inner
                T_  = c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*(c6 + x*c7))))))
                dT_ = c1 + x*(2f0*c2 + x*(3f0*c3 + x*(4f0*c4 + x*(5f0*c5 + x*(6f0*c6 + x*7f0*c7)))))
                if abs(dT_) < 1f-30
                    break
                end
                Δ = (T_ - I_star) / dT_
                x -= Δ
                if abs(Δ) < 1f-7
                    break
                end
            end
            sino_water[idx]  = max(y_opt, 0f0)
            sino_iodine[idx] = max(z_bar + x, 0f0)
        end
    end
    dt = time() - t0

    if verbose
        basis_mode = per_ray ? "bowtie ŵ" : "centered ŵ"
        @info "[Cong-PCCT (3-channel, $basis_mode)] $(round(dt, digits=2)) s"
    end
    nothing
end

# ╔═╡ 08131e02-1000-4000-8000-000000000006
# Cong-PCCT decomposition runner — calls the kernel + post-staging.
cong_pcct_decomp_s1 = let
    sino_shape = size(sim_scan1_n_channel_s1.h_channels[1])

    if cong_pcct_run_on_cpu_s1
        @info "[Cong-PCCT] running on CPU (cong_pcct_run_on_cpu_s1 = true)"
        sino_iodine = zeros(Float32, sino_shape)
        sino_water  = zeros(Float32, sino_shape)
        h_chs       = [Float32.(h) for h in sim_scan1_n_channel_s1.h_channels]
    else
        @info "[Cong-PCCT] running on Metal GPU"
        sino_iodine = MtlArray(zeros(Float32, sino_shape))
        sino_water  = MtlArray(zeros(Float32, sino_shape))
        h_chs       = [MtlArray(Float32.(h)) for h in sim_scan1_n_channel_s1.h_channels]
    end

    t0 = time()
    _apply_cong_pcct_s1!(
        sino_iodine, sino_water,
        h_chs,
        sim_scan1_n_channel_s1.I0_channels;
        basis           = cong_pcct_basis_s1,
        newton_iter     = cong_pcct_newton_iter_s1,
        brent_max_iter  = cong_pcct_brent_max_iter_s1,
        y_max_factor    = cong_pcct_y_max_factor_s1,
        verbose         = true,
    )
    @info "[Cong-PCCT decomp] total $(round(time()-t0, digits=1)) s"

    sino_I_cpu = Array(sino_iodine)
    sino_W_cpu = Array(sino_water)
    sino_iodine = nothing; sino_water = nothing; h_chs = nothing
    GC.gc(true)

    n_I_zero = count(iszero, sino_I_cpu)
    n_W_zero = count(iszero, sino_W_cpu)
    @info "[Cong-PCCT out]  sino_iodine min=$(round(minimum(sino_I_cpu),sigdigits=4)) max=$(round(maximum(sino_I_cpu),sigdigits=4)) mean=$(round(mean(sino_I_cpu),sigdigits=4))  nzero=$(n_I_zero)/$(length(sino_I_cpu))"
    @info "[Cong-PCCT out]  sino_water  min=$(round(minimum(sino_W_cpu), sigdigits=4)) max=$(round(maximum(sino_W_cpu), sigdigits=4)) mean=$(round(mean(sino_W_cpu), sigdigits=4))  nzero=$(n_W_zero)/$(length(sino_W_cpu))"

    (sino_iodine = sino_I_cpu, sino_water = sino_W_cpu, geom = sim_scan1.geom)
end;

# ╔═╡ 08131e03-0000-4000-8000-000000000001
md"""
**PWLS-L₂ (Long/Fessler 2014).** 2-bin (low, high) sinogram-domain restoration via
2×2 matrix-curvature SQS.  Warm-started from Cong; uses the same 2-bin basis
Cong already built (`cong_basis_s1.{ŵ_L, ŵ_H, p_L, p_H, q_L, q_H}`).  Tune via
Scan 1's `use_pwls_s1`, `pwls_*_s1` config cell.
"""

# ╔═╡ 08131010-0000-4000-8000-000000000041
# PWLS basis = Cong basis (material-direct now).  cong_basis_s1.{p,q} are
# already (μρ_iodine, μρ_water), so PWLS's `exp(-p·s_I − q·s_W)` forward
# matches without any A_coef transform.
pwls_basis_s1 = let
    @info "[PWLS basis]  μρ_iodine(low) range=[$(round(minimum(cong_basis_s1.p_L), sigdigits=3)), $(round(maximum(cong_basis_s1.p_L), sigdigits=3))] cm²/g"
    @info "[PWLS basis]  μρ_water(low)  range=[$(round(minimum(cong_basis_s1.q_L), sigdigits=3)), $(round(maximum(cong_basis_s1.q_L), sigdigits=3))] cm²/g"
    cong_basis_s1
end;

# ╔═╡ 08131010-0000-4000-8000-000000000042
# Separable 2D Gaussian smoothing on (col, row) plane, per-view independent.
# CPU-side, threaded over views.  Applied to Cong warm start before PWLS to
# replace white per-pixel noise with bandlimited noise that hyper3's wpot
# can actually smooth (MIRT pwls_example.m:36 init-smoothing pattern).
function _gauss_smooth_2d!(
        arr::AbstractArray{Float32, 3},
        σ::Real,
        scratch::AbstractArray{Float32, 3},
    )
    σ_f = Float32(σ)
    σ_f <= 0f0 && return arr
    radius = max(1, ceil(Int, 3 * σ_f))
    n_col, n_row, n_view = size(arr)
    inv_2σ²  = 1f0 / (2f0 * σ_f * σ_f)
    weights  = Float32[exp(-(Float32(i)^2) * inv_2σ²) for i in -radius:radius]
    weights ./= sum(weights)

    # x-pass:  arr → scratch
    Threads.@threads for v in 1:n_view
        @inbounds for r in 1:n_row, c in 1:n_col
            s = 0f0
            for k in -radius:radius
                cc = clamp(c + k, 1, n_col)
                s += arr[cc, r, v] * weights[k + radius + 1]
            end
            scratch[c, r, v] = s
        end
    end
    # y-pass:  scratch → arr
    Threads.@threads for v in 1:n_view
        @inbounds for r in 1:n_row, c in 1:n_col
            s = 0f0
            for k in -radius:radius
                rr = clamp(r + k, 1, n_row)
                s += scratch[c, rr, v] * weights[k + radius + 1]
            end
            arr[c, r, v] = s
        end
    end
    return arr
end

# ╔═╡ 08131010-0000-4000-8000-000000000045
# ── Scan 1 QPWLS via Preconditioned CG — 1:1 port of MIRT qpwls_pcg1.m ──
# https://github.com/JeffFessler/mirt/blob/main/wls/qpwls_pcg1.m
# (Jeff Fessler 1998).  Quadratic penalized weighted least squares with PCG.
#
# MIRT cost:
#     Φ(x) = ½ (y − A·x)' W (y − A·x)  +  ½ x' C'C x
#
# Mapped to 2-bin material PCCT (per pixel, 2-channel):
#     x      = (s_I, s_W)             stacked iodine/water sino
#     y      = (h_low, h_high)         measured bin line integrals
#     A·x    = J·x  per pixel          linearized polychromatic Jacobian
#                                      J = [P_L Q_L; P_H Q_H] frozen at s_Cong
#     W      = diag(w_L, w_H)          Poisson weights  w_b = I0_b·exp(−h_b_meas)
#     C      = β · ∇                   1st-order forward difference, periodic
#                                      (β factored INTO C so qpwls_pcg1 uses raw ‖Cx‖²)
#
# Linearization at Cong (one-shot Gauss-Newton):
#     F_b(s)        ≈  F_b(s_Cong) + J_b·(s − s_Cong)
#     ⇒  effective y_b  =  h_b − F_b(s_Cong) + J_b·s_Cong
#     residual = y_b − A·s = h_b − F_b(s)  (linearized)
#     where F_b(s) = -log( ∫ ŵ_b(E)·exp(−p(E)·s_I − q(E)·s_W) dE )
#
# Algorithm (qpwls_pcg1.m lines 79-153 — Fletcher-Reeves CG with exact step):
#     init:   Ax = A·x;  Cx = C·x
#     loop:
#         g          = A'·W·(y − Ax) − C'·Cx                   (negative gradient)
#         M·g        = preconditioner (we use M = I)
#         γ          = (g'·M·g) / (oldg'·M·oldg)               (FR; γ=0 first iter)
#         d          = M·g + γ·d                               (search direction)
#         Ad         = A·d;  Cd = C·d
#         denom      = Ad'·W·Ad + Cd'·Cd
#         α          = (d'·g) / denom                          (exact line search)
#         x         += α·d;  Ax += α·Ad;  Cx += α·Cd
#
# Why PCG > SQS for QUADRATIC cost: exact line search (no step-limit knob),
# Fletcher-Reeves momentum (faster convergence than SQS row-sum bound), no
# `relax`, no `step_lim_*`, no body mask, no air threshold.  Fewer knobs.
function _apply_mirt_qpwls_pcg1_s1!(
        sino_iodine::AbstractArray{Float32, 3},     # IN: Cong warm start; OUT: PWLS-polished
        sino_water::AbstractArray{Float32, 3},
        h_low::AbstractArray{Float32, 3},           # measured bin line integrals
        h_high::AbstractArray{Float32, 3};
        cong_basis,                                  # NamedTuple: .ŵ_L, .ŵ_H, .p_L, .q_L, .p_H, .q_H
        I0_low::Real, I0_high::Real,
        β_iodine::Real = 1.0f0,                      # √ folded into C — auto-scaled by mean(D_II)/8
        β_water::Real  = 1.0f0,                      # √ folded into C — auto-scaled by mean(D_WW)/8
        n_iter::Integer = 30,
        verbose::Bool   = true,
    )
    sino_shape = size(sino_iodine)
    sino_shape == size(sino_water) == size(h_low) == size(h_high) ||
        error("_apply_mirt_qpwls_pcg1_s1!: sino / h_low / h_high must share shape.")
    n_col, n_row, n_view = Int(sino_shape[1]), Int(sino_shape[2]), Int(sino_shape[3])

    # ── Stage basis to backend (per-bin Σ_E ŵ = 1 normalization) ──
    function _norm_ŵ(ŵ_raw)
        ŵ = Float32.(Array(ŵ_raw))
        if ndims(ŵ) == 1
            ŵ ./= sum(ŵ)
        else
            nc, nr, nE = size(ŵ)
            @inbounds for r in 1:nr, c in 1:nc
                s = 0f0
                for k in 1:nE; s += ŵ[c, r, k]; end
                if s > 0f0
                    inv_s = 1f0 / s
                    for k in 1:nE; ŵ[c, r, k] *= inv_s; end
                end
            end
        end
        ŵ
    end
    ŵ_L = BS._match_backend(_norm_ŵ(cong_basis.ŵ_L), sino_iodine)
    ŵ_H = BS._match_backend(_norm_ŵ(cong_basis.ŵ_H), sino_iodine)
    p_L = BS._match_backend(Float32.(cong_basis.p_L), sino_iodine)
    q_L = BS._match_backend(Float32.(cong_basis.q_L), sino_iodine)
    p_H = BS._match_backend(Float32.(cong_basis.p_H), sino_iodine)
    q_H = BS._match_backend(Float32.(cong_basis.q_H), sino_iodine)
    per_ray = ndims(ŵ_L) == 3
    nE_L = per_ray ? size(ŵ_L, 3) : length(ŵ_L)
    nE_H = per_ray ? size(ŵ_H, 3) : length(ŵ_H)

    I0_L = Float32(I0_low);  I0_H = Float32(I0_high)

    # ── Step 1: linearize forward at Cong's (s_I, s_W) ──
    # Compute per pixel: J = [P_L Q_L; P_H Q_H], h_b_pred = -log(Z_b),
    # and W = diag(w_b) where w_b = I0_b·exp(-h_b_meas) (Poisson).
    P_L = similar(sino_iodine);  Q_L = similar(sino_iodine)
    P_H = similar(sino_iodine);  Q_H = similar(sino_iodine)
    h_L_pred = similar(sino_iodine);  h_H_pred = similar(sino_iodine)
    w_L = similar(sino_iodine);  w_H = similar(sino_iodine)

    AK.foreachindex(sino_iodine) do idx
        Iv = sino_iodine[idx];  Wv = sino_water[idx]
        i0 = idx - 1
        c = (i0 % n_col) + 1
        r = ((i0 ÷ n_col) % n_row) + 1

        # Bin L: Z_L = Σ ŵ·exp(-p·sI - q·sW),  ZpL = Σ p·ŵ·exp(...),  ZqL = Σ q·ŵ·exp(...)
        Z_L = 0f0;  ZpL = 0f0;  ZqL = 0f0
        if per_ray
            for k in 1:nE_L
                wk = ŵ_L[c, r, k] * exp(-p_L[k]*Iv - q_L[k]*Wv)
                Z_L += wk;  ZpL += p_L[k]*wk;  ZqL += q_L[k]*wk
            end
        else
            for k in 1:nE_L
                wk = ŵ_L[k] * exp(-p_L[k]*Iv - q_L[k]*Wv)
                Z_L += wk;  ZpL += p_L[k]*wk;  ZqL += q_L[k]*wk
            end
        end
        invZ_L = 1f0 / max(Z_L, 1f-20)
        P_L[idx] = ZpL * invZ_L              # ∂h_L/∂s_I
        Q_L[idx] = ZqL * invZ_L              # ∂h_L/∂s_W
        h_L_pred[idx] = -log(max(Z_L, 1f-20))

        # Bin H
        Z_H = 0f0;  ZpH = 0f0;  ZqH = 0f0
        if per_ray
            for k in 1:nE_H
                wk = ŵ_H[c, r, k] * exp(-p_H[k]*Iv - q_H[k]*Wv)
                Z_H += wk;  ZpH += p_H[k]*wk;  ZqH += q_H[k]*wk
            end
        else
            for k in 1:nE_H
                wk = ŵ_H[k] * exp(-p_H[k]*Iv - q_H[k]*Wv)
                Z_H += wk;  ZpH += p_H[k]*wk;  ZqH += q_H[k]*wk
            end
        end
        invZ_H = 1f0 / max(Z_H, 1f-20)
        P_H[idx] = ZpH * invZ_H
        Q_H[idx] = ZqH * invZ_H
        h_H_pred[idx] = -log(max(Z_H, 1f-20))

        # Poisson weights: w_b = N_b = I0_b·exp(-h_b_meas)
        w_L[idx] = I0_L * exp(-h_low[idx])
        w_H[idx] = I0_H * exp(-h_high[idx])
    end

    # ── β scaling: auto-scale so β_user=1 ≈ "reg row-sum ≈ data row-sum" ──
    # Diagonal data Hessian D_diag = w_L·P_b² + w_H·P_b²  per channel.
    # Laplacian row-sum bound = 8 (4 in 2D × 2 sides).  β_eff = β_user·D_typ/8.
    D_II_mean = Float32(mean(@. w_L*P_L*P_L + w_H*P_H*P_H))
    D_WW_mean = Float32(mean(@. w_L*Q_L*Q_L + w_H*Q_H*Q_H))
    β_I_eff   = Float32(β_iodine) * max(D_II_mean, 1f-20) / 8f0
    β_W_eff   = Float32(β_water)  * max(D_WW_mean, 1f-20) / 8f0
    s_β_I     = sqrt(β_I_eff)               # √β folded into C: ‖Cx‖² = β·‖∇x‖²
    s_β_W     = sqrt(β_W_eff)

    # ── Build effective y: ỹ_b = h_b_meas − h_b_pred + (P_b·sI_C + Q_b·sW_C) ──
    # qpwls_pcg1 minimizes ‖y − Ax‖²_W; with A·x = J·x and linearization at s_Cong:
    #     residual = ỹ_b − (P_b·s_I + Q_b·s_W)  =  h_b_meas − F_b(s)  (linearized)
    s_Cong_I = copy(sino_iodine)
    s_Cong_W = copy(sino_water)
    ỹ_L = similar(sino_iodine);  ỹ_H = similar(sino_iodine)
    AK.foreachindex(ỹ_L) do idx
        ỹ_L[idx] = h_low[idx]  - h_L_pred[idx] + P_L[idx]*s_Cong_I[idx] + Q_L[idx]*s_Cong_W[idx]
        ỹ_H[idx] = h_high[idx] - h_H_pred[idx] + P_H[idx]*s_Cong_I[idx] + Q_H[idx]*s_Cong_W[idx]
    end

    # ── PCG state (qpwls_pcg1.m lines 71-77, 80-95) ──
    # x is implicit in (sino_iodine, sino_water).  Track Ax (= predicted h_b
    # under linearized A) and Cx (= ∇x scaled by √β) in pre-allocated buffers.
    Ax_L = similar(sino_iodine);  Ax_H = similar(sino_iodine)
    Cx_I_h = similar(sino_iodine);  Cx_I_v = similar(sino_iodine)
    Cx_W_h = similar(sino_iodine);  Cx_W_v = similar(sino_iodine)

    # Compute initial Ax = J·x_init, Cx = √β·∇x_init
    AK.foreachindex(Ax_L) do idx
        Ax_L[idx] = P_L[idx]*sino_iodine[idx] + Q_L[idx]*sino_water[idx]
        Ax_H[idx] = P_H[idx]*sino_iodine[idx] + Q_H[idx]*sino_water[idx]
    end
    function _apply_C!(out_h, out_v, x, β_sqrt)
        AK.foreachindex(out_h) do idx
            i0 = idx - 1
            c  = (i0 % n_col) + 1
            r  = ((i0 ÷ n_col) % n_row) + 1
            v  = ((i0 ÷ (n_col * n_row))) + 1
            cp1 = c == n_col ? 1 : c + 1
            rp1 = r == n_row ? 1 : r + 1
            out_h[idx] = β_sqrt * (x[cp1, r, v] - x[c, r, v])
            out_v[idx] = β_sqrt * (x[c, rp1, v] - x[c, r, v])
        end
    end
    function _apply_Ct!(out, in_h, in_v, β_sqrt)
        AK.foreachindex(out) do idx
            i0 = idx - 1
            c  = (i0 % n_col) + 1
            r  = ((i0 ÷ n_col) % n_row) + 1
            v  = ((i0 ÷ (n_col * n_row))) + 1
            cm1 = c == 1 ? n_col : c - 1
            rm1 = r == 1 ? n_row : r - 1
            out[idx] = β_sqrt * ((in_h[c, r, v] - in_h[cm1, r, v]) +
                                 (in_v[c, r, v] - in_v[c, rm1, v]))
        end
    end
    _apply_C!(Cx_I_h, Cx_I_v, sino_iodine, s_β_I)
    _apply_C!(Cx_W_h, Cx_W_v, sino_water,  s_β_W)

    # PCG buffers — gradient g, search direction d, Ad, Cd
    g_I = similar(sino_iodine);  g_W = similar(sino_iodine)
    d_I = similar(sino_iodine);  d_W = similar(sino_iodine)
    Ad_L = similar(sino_iodine); Ad_H = similar(sino_iodine)
    Cd_I_h = similar(sino_iodine); Cd_I_v = similar(sino_iodine)
    Cd_W_h = similar(sino_iodine); Cd_W_v = similar(sino_iodine)
    Ctc_I = similar(sino_iodine);  Ctc_W = similar(sino_iodine)

    oldinprod = 0f0
    t0 = time()
    for iter in 1:Int(n_iter)
        # negative gradient g = A'·W·(y − Ax) − C'·Cx
        # A'·W·r per pixel (2-vec output): (P_L·w_L·r_L + P_H·w_H·r_H, Q_L·w_L·r_L + Q_H·w_H·r_H)
        AK.foreachindex(g_I) do idx
            r_L = ỹ_L[idx] - Ax_L[idx]
            r_H = ỹ_H[idx] - Ax_H[idx]
            g_I[idx] = P_L[idx]*w_L[idx]*r_L + P_H[idx]*w_H[idx]*r_H
            g_W[idx] = Q_L[idx]*w_L[idx]*r_L + Q_H[idx]*w_H[idx]*r_H
        end
        _apply_Ct!(Ctc_I, Cx_I_h, Cx_I_v, s_β_I)
        _apply_Ct!(Ctc_W, Cx_W_h, Cx_W_v, s_β_W)
        AK.foreachindex(g_I) do idx
            g_I[idx] -= Ctc_I[idx]
            g_W[idx] -= Ctc_W[idx]
        end

        # Fletcher-Reeves CG (precon = I): inprod = g'·g
        newinprod = Float32(sum(@. g_I*g_I + g_W*g_W))
        if iter == 1
            d_I .= g_I;  d_W .= g_W
        else
            γ = oldinprod > 0f0 ? newinprod / oldinprod : 0f0
            AK.foreachindex(d_I) do idx
                d_I[idx] = g_I[idx] + γ * d_I[idx]
                d_W[idx] = g_W[idx] + γ * d_W[idx]
            end
        end
        oldinprod = newinprod

        # Ad = A·d, Cd = √β·∇d
        AK.foreachindex(Ad_L) do idx
            Ad_L[idx] = P_L[idx]*d_I[idx] + Q_L[idx]*d_W[idx]
            Ad_H[idx] = P_H[idx]*d_I[idx] + Q_H[idx]*d_W[idx]
        end
        _apply_C!(Cd_I_h, Cd_I_v, d_I, s_β_I)
        _apply_C!(Cd_W_h, Cd_W_v, d_W, s_β_W)

        # denom = Ad'·W·Ad + Cd'·Cd
        denom = Float32(sum(@. w_L*Ad_L*Ad_L + w_H*Ad_H*Ad_H)) +
                Float32(sum(@. Cd_I_h*Cd_I_h + Cd_I_v*Cd_I_v +
                              Cd_W_h*Cd_W_h + Cd_W_v*Cd_W_v))
        # exact step α = (d'·g) / denom
        dg = Float32(sum(@. d_I*g_I + d_W*g_W))
        α  = denom > 0f0 ? dg / denom : 0f0

        # x += α·d, Ax += α·Ad, Cx += α·Cd
        AK.foreachindex(sino_iodine) do idx
            sino_iodine[idx] += α * d_I[idx]
            sino_water[idx]  += α * d_W[idx]
            Ax_L[idx]        += α * Ad_L[idx]
            Ax_H[idx]        += α * Ad_H[idx]
            Cx_I_h[idx]      += α * Cd_I_h[idx]
            Cx_I_v[idx]      += α * Cd_I_v[idx]
            Cx_W_h[idx]      += α * Cd_W_h[idx]
            Cx_W_v[idx]      += α * Cd_W_v[idx]
        end

        if verbose && (iter == 1 || iter % 10 == 0 || iter == n_iter)
            @info "[QPWLS-PCG1] iter $(iter)/$(n_iter)  α=$(round(α, sigdigits=3))  ‖g‖²=$(round(newinprod, sigdigits=3))"
        end
    end
    dt = time() - t0

    if verbose
        basis_mode = per_ray ? "per-ray bowtie ŵ" : "centered (1D) ŵ"
        @info "[QPWLS-PCG1 (MIRT-style, $basis_mode)] $(n_iter) iter, $(round(dt, digits=1)) s"
        @info "  β_user (I, W) = ($(β_iodine), $(β_water))   β_eff = ($(round(β_I_eff, sigdigits=3)), $(round(β_W_eff, sigdigits=3)))"
        @info "  D_typical: I=$(round(D_II_mean, sigdigits=3))  W=$(round(D_WW_mean, sigdigits=3))"
    end

    (n_iter = Int(n_iter), β_iodine = Float64(β_iodine), β_water = Float64(β_water),
     β_iodine_eff = Float64(β_I_eff), β_water_eff = Float64(β_W_eff))
end


# ╔═╡ 08131010-0000-4000-8000-000000000080
# ── Scan 1 post-Cong denoising via ACNR (BS.apply_acnr! — Kalender 1988) ──
# https://github.com/MolloiLab/BasisSimulator.jl  src/reconstruction/vmi/acnr.jl
#
# THIS IS THE RIGHT TOOL.  apply_acnr! already exists and does exactly what
# the user described: post-processing, multi-material aware, anti-correlation
# preserving by construction.  Kalender et al. 1988 IEEE TMI 7(3):218-224.
#
# Algorithm (fully detailed in acnr.jl docstring):
#   Pick reference energy E_ref where VMI synthesis is most important (E_opt).
#   Define signal direction (c_a, c_b) = (μρ_iodine, μρ_water) at E_ref.
#   Decompose:
#     s_∥ = (c_a·s_I + c_b·s_W) / √(c_a² + c_b²)      ∝ VMI at E_ref (signal)
#     s_⊥ = -c_b·s_I + c_a·s_W                          orthogonal (anti-corr noise)
#   Smooth ONLY s_⊥ via FFT Gaussian:
#     s_⊥_smooth = LP(s_⊥)
#     n_⊥        = s_⊥ - s_⊥_smooth                    extracted noise
#   Project back with opposite signs (preserves signal direction):
#     s_I' = s_I + γ·c_b·n_⊥ / (c_a² + c_b²)
#     s_W' = s_W - γ·c_a·n_⊥ / (c_a² + c_b²)
#   ⇒ c_a·s_I' + c_b·s_W' = c_a·s_I + c_b·s_W          ← PIXEL-PERFECT preservation
#
# Why this beats every PWLS variant we tried:
#   • Pure post-processor (no data-fit iteration → no envelope-ray amplification)
#   • Anti-correlation preserved BY CONSTRUCTION (signal direction guaranteed intact)
#   • VMI at E_ref has ZERO resolution loss (structural guarantee)
#   • No per-pixel eigendecomp / rotation needed (single global signal direction)
#
# Optional post-Gaussian per material for residual cleanup of signal-direction
# noise that ACNR by construction doesn't touch.  Mild σ_post (~1-2 px).
function _apply_pwls_acnr_s1!(
        sino_iodine::AbstractArray{Float32, 3},      # mutated in place
        sino_water::AbstractArray{Float32, 3};       # mutated in place
        E_ref::Real     = 70.0,                      # keV; typically VMI E_opt
        σ_acnr::Real    = 5.0,                       # FFT Gaussian px on s_⊥ (anti-corr noise)
        σ_post::Real    = 0.0,                       # mild per-material Gaussian; 0 = disable
        γ_acnr::Real    = 1.0,                       # ACNR correction strength ∈ [0, 1]
        verbose::Bool   = true,
    )
    # Stage to CPU (BS.apply_acnr! uses FFTW; cheap on Apple unified memory).
    s_I_cpu = sino_iodine isa Array ? sino_iodine : Array(sino_iodine)
    s_W_cpu = sino_water  isa Array ? sino_water  : Array(sino_water)

    # Lookup material μρ at reference energy (cm²/g).
    μρ_iodine = BS.compute_mass_μ_at_energy(XA.Elements.Iodine,  Float64(E_ref))
    μρ_water  = BS.compute_mass_μ_at_energy(XA.Materials.water,  Float64(E_ref))

    # ── Step 1: ACNR (anti-correlated noise reduction) ──
    # For (sino_iodine, sino_water) → c_a = μρ_iodine, c_b = μρ_water.
    # Signal direction (c_a, c_b) gets pixel-perfect preserved.
    info_acnr = BS.apply_acnr!(
        s_I_cpu, s_W_cpu;
        c_a     = μρ_iodine,
        c_b     = μρ_water,
        σ       = σ_acnr,
        γ       = γ_acnr,
        verbose = verbose,
    )

    # ── Step 2: Optional per-material Gaussian post-smoothing ──
    # ACNR preserves the signal direction perfectly.  Any noise along that
    # direction (the "co-correlated" component) is untouched.  Mild Gaussian
    # cleans up the residual without significantly distorting the signal
    # direction (since both materials are smoothed by the same kernel).
    info_post = (σ_post = 0.0,)
    if σ_post > 0
        scratch = similar(s_I_cpu)
        _gauss_smooth_2d!(s_I_cpu, σ_post, scratch)
        _gauss_smooth_2d!(s_W_cpu, σ_post, scratch)
        info_post = (σ_post = Float64(σ_post),)
        verbose && @info "[ACNR + post-Gauss] applied σ_post=$(σ_post) px to each material"
    end

    # Copy back to GPU input arrays if needed.
    if !(sino_iodine isa Array)
        copyto!(sino_iodine, s_I_cpu)
        copyto!(sino_water,  s_W_cpu)
    end

    if verbose
        @info "[_apply_pwls_acnr_s1!] E_ref = $(E_ref) keV"
        @info "  μρ_iodine(E_ref) = $(round(μρ_iodine, sigdigits=3)) cm²/g  (c_a)"
        @info "  μρ_water(E_ref)  = $(round(μρ_water,  sigdigits=3)) cm²/g  (c_b)"
    end

    (acnr   = info_acnr,
     post   = info_post,
     E_ref  = Float64(E_ref),
     σ_acnr = Float64(σ_acnr),
     σ_post = Float64(σ_post),
     γ_acnr = Float64(γ_acnr))
end

# ╔═╡ 08131009-a000-4000-8000-000000000011
# ── Scan 1 post-Cong polish config — two paths only ──
#   :mirt_pwls — MIRT qpwls_pcg1.m 1:1 port (QPWLS via PCG, exact line search)
#   :acnr      — BS.apply_acnr! anti-correlated noise reduction (Kalender 1988)
begin
    use_pwls_s1                   = true
    pwls_path_s1                  = :acnr

    # ── MIRT QPWLS-PCG1 knobs (path = :mirt_pwls) ──
    # Pure quadratic PWLS via Fletcher-Reeves PCG with exact line search.
    # β auto-scales to mean(D)/8 internally so β_user = 1 ≈ "reg row-sum ≈
    # data row-sum at typical pixel" — scale-free convention.
    #   β ≈ 0.1  → barely smooth (almost Cong)
    #   β ≈ 1    → moderate smoothing
    #   β ≈ 10   → aggressive
    pwls_β_iodine_mirt_s1         = 1.0
    pwls_β_water_mirt_s1          = 1.0
    pwls_n_iter_mirt_s1           = 30

    # ── ACNR knobs (path = :acnr) — BS.apply_acnr! Kalender 1988 ──
    # Pixel-perfect preserves VMI(E_ref).  Smooths ONLY the anti-correlated
    # noise direction.  Optional mild post-Gaussian for signal-direction cleanup.
    pwls_E_ref_acnr_s1            = 70.0    # keV — typically VMI E_opt
    pwls_σ_acnr_s1                = 5.0     # FFT Gaussian px on s_⊥
    pwls_σ_post_acnr_s1           = 1.5     # per-material post-Gaussian (0 = disable)
    pwls_γ_acnr_s1                = 1.0     # correction strength ∈ [0, 1]
end

# ╔═╡ 08131010-0000-4000-8000-000000000040
# Scan 1 post-Cong polish — dispatcher.  Two paths:
#   :mirt_pwls — MIRT qpwls_pcg1.m 1:1 port (cell 0045)
#   :acnr      — Kalender 1988 anti-correlated noise reduction (cell 0080)
pwls_decomp_s1 = let
    path = pwls_path_s1
    @info "[PWLS s1] entering cell.  use_pwls_s1=$(use_pwls_s1), path=$(path)"
    if !use_pwls_s1
        @info "[PWLS s1] DISABLED — passing through Cong-PCCT warm start"
        (sino_w = Float32.(cong_pcct_decomp_s1.sino_water),
         sino_I = Float32.(cong_pcct_decomp_s1.sino_iodine))
    else
        sino_I_gpu = MtlArray(copy(cong_pcct_decomp_s1.sino_iodine))
        sino_W_gpu = MtlArray(copy(cong_pcct_decomp_s1.sino_water))

        if path == :acnr
            _apply_pwls_acnr_s1!(
                sino_I_gpu, sino_W_gpu;
                E_ref   = pwls_E_ref_acnr_s1,
                σ_acnr  = pwls_σ_acnr_s1,
                σ_post  = pwls_σ_post_acnr_s1,
                γ_acnr  = pwls_γ_acnr_s1,
                verbose = true,
            )
        elseif path == :mirt_pwls
            h_lo_gpu = MtlArray(Float32.(sim_scan1_lohi.sino_low))
            h_hi_gpu = MtlArray(Float32.(sim_scan1_lohi.sino_high))
            _apply_mirt_qpwls_pcg1_s1!(
                sino_I_gpu, sino_W_gpu, h_lo_gpu, h_hi_gpu;
                cong_basis = pwls_basis_s1,
                I0_low     = sim_scan1_lohi.I0_low,
                I0_high    = sim_scan1_lohi.I0_high,
                β_iodine   = pwls_β_iodine_mirt_s1,
                β_water    = pwls_β_water_mirt_s1,
                n_iter     = pwls_n_iter_mirt_s1,
                verbose    = true,
            )
            h_lo_gpu = nothing;  h_hi_gpu = nothing
        else
            error("pwls_decomp_s1: unknown pwls_path_s1 = $(path).  Use :acnr or :mirt_pwls.")
        end

        sino_I_cpu = Array(sino_I_gpu)
        sino_W_cpu = Array(sino_W_gpu)
        sino_I_gpu = nothing;  sino_W_gpu = nothing
        GC.gc(true)

        (sino_w = sino_W_cpu, sino_I = sino_I_cpu, path = path)
    end
end;

# ╔═╡ 08131f00-0000-4000-8000-000000000001
md"""
**VMI synthesis.** Per-energy HIR recon + Mono+ polish, fused into one cell.
Config: `vmi_energies_s1` selects synthesis energies; `hir_*_s1` controls HIR; `vmip_*_s1` controls Mono+.
"""

# ╔═╡ 08131009-a000-4000-8000-000000000001
# ── Scan 1 VMI target energies ────────────────────────────────────────────
vmi_energies_s1 = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 08131009-a000-4000-8000-000000000007
# ── Scan 1 Mono+ config ───────────────────────────────────────────────────
begin
    use_mono_plus_s1     = true
    vmip_E_noise_opt_s1  = 70.0
    vmip_σ_lp_px_s1      = Float64[1.5, 0.0, 1.5, 3.0]
end

# ╔═╡ 08131f02-0000-4000-8000-000000000001
md"""
**VMI pipeline (Scan 1).** Mirrors notebook 06's tuned chain:
1. Reads `(sino_w, sino_I)` from `pwls_decomp_s1`.
2. For each `E ∈ vmi_energies_s1`: synth `vmi_sino(E) = μρ_W·sino_w + μρ_I·sino_I` → **per-E FBP apod** → HU.
3. **Mono+** frequency-split polish across all energies (LP from `vmip_E_noise_opt_s1`, HP from each target, σ in `vmip_σ_lp_px_s1`).
4. **HIR** (optional) per-E, warm-started from the Mono+ output for that energy.

`use_hir_s1 = false` → return Mono+ output as final.  `use_mono_plus_s1 = false` → skip Mono+, optionally still HIR on raw FBP.
Tune via Scan 1's own `use_hir_s1` / `hir_*_s1` and `use_mono_plus_s1` / `vmip_*_s1` config cells, plus `vmi_apod_y_per_E_s1` inside the cell.
"""

# ╔═╡ 08131009-a000-4000-8000-000000000006
# ── Scan 1 HIR config ─────────────────────────────────────────────────────
begin
    use_hir_s1         = true
    hir_strength_s1    = 2
    hir_lambda_s1      = 5.0f0
    hir_nepochs_s1     = 2
    hir_n_subsets_s1   = 12
    hir_huber_delta_s1 = 0.06f0
    hir_relaxation_s1  = 0.35f0
end

# ╔═╡ 08131010-a000-4000-8000-000000000002
# ── Scan 1 VMI pipeline: PWLS → VMI(per-E FBP apod) → Mono+ → HIR(optional) ──
# Mirrors nb 06's tuned chain.  Variable name `sim_scan1_vmi_hir` preserved
# for downstream consumers (Dict{Float64, Array{Float32, 3}} of HU volumes).
sim_scan1_vmi_hir = let
    geom       = sim_scan1.geom
    recon_size = sim_matrix_size
    sino_w     = pwls_decomp_s1.sino_w
    sino_I     = pwls_decomp_s1.sino_I
    energies   = Float64.(vmi_energies_s1)
    mid_z      = recon_size[3] ÷ 2

    # ── Per-E FBP apodization (mirrors nb 06's tuned per-energy filter) ──
    vmi_apod_x = (0.0, 0.25, 0.5, 0.75, 1.0)
    vmi_apod_y_per_E_s1 = [
        (1.0, 0.72, 0.45, 0.20, 0.05),    # 40 keV  — aggressive, HF replaced by VMI_70 via Mono+
        (1.0, 0.85, 0.65, 0.40, 0.15),    # 70 keV  — calibration anchor
        (1.0, 0.70, 0.42, 0.18, 0.04),    # 100 keV — Mono+ identity here, NPS softer than 70
        (1.0, 0.50, 0.30, 0.12, 0.03),    # 140 keV — LF tightened, HF replaced by VMI_70
    ]
    length(vmi_apod_y_per_E_s1) == length(energies) ||
        error("vmi_apod_y_per_E_s1 must have one entry per VMI energy ($(length(energies)))")

    # ── Step 1: VMI synthesis with per-E FBP apod (no HIR here) ──
    raw_fbp = Dict{Float64, Array{Float32, 3}}()
    for (k, E) in enumerate(energies)
        μ_w_E  = BS.compute_μ_at_energy(XA.Materials.water, E)
        μρ_w_E = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water,  E))
        μρ_I_E = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E))

        vmi_sino = @. μρ_w_E * sino_w + μρ_I_E * sino_I
        sino_gpu = MtlArray(vmi_sino)
        ws_fdk   = BS.create_fdk_recon_workspace(
            sino_gpu, geom, recon_size;
            filter = BS.CustomFilter(vmi_apod_x, vmi_apod_y_per_E_s1[k]),
        )
        recon_μ  = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))
        sino_gpu = nothing; ws_fdk = nothing

        recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_w_E))
        raw_fbp[E] = recon_hu

        roi = recon_hu[200:300, 200:300, mid_z]
        @info "[scan 1 VMI FBP] $(Int(E)) keV (Nyquist apod=$(vmi_apod_y_per_E_s1[k][end])): σ=$(round(std(roi), digits=1)) HU, mean=$(round(mean(roi), digits=1)) HU"
    end
    GC.gc(true)

    # ── Step 2: Mono+ frequency-split polish (per-E σ from vmip_σ_lp_px_s1) ──
    mono_out = if !use_mono_plus_s1
        @info "[scan 1 Mono+] DISABLED — passing raw FBP VMIs through"
        raw_fbp
    else
        haskey(raw_fbp, vmip_E_noise_opt_s1) ||
            error("Mono+ reference vmip_E_noise_opt_s1=$(vmip_E_noise_opt_s1) keV not in vmi_energies_s1=$energies")

        vols_in   = [raw_fbp[E] for E in energies]
        ws_mono_s1 = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(energies))
        res = BS.apply_mono_plus!(ws_mono_s1, vols_in, energies;
            E_noise_opt = vmip_E_noise_opt_s1,
            σ_lp_px     = Float64.(vmip_σ_lp_px_s1),
            verbose     = true)

        out = Dict{Float64, Array{Float32, 3}}()
        for (i, E) in enumerate(energies)
            mp = copy(res.volumes[i])
            roi_before = vols_in[i][200:300, 200:300, mid_z]
            roi_after  = mp[200:300, 200:300, mid_z]
            @info "[scan 1 Mono+] $(Int(E)) keV (σ_lp=$(res.σ_lp_px[i]) px): σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"
            out[E] = mp
        end
        ws_mono_s1 = nothing; GC.gc(true)
        out
    end

    # ── Step 3: HIR per E (optional), warm-started from Mono+ output ──
    if !use_hir_s1
        @info "[scan 1 HIR] DISABLED — returning Mono+ output as final VMI"
        mono_out
    else
        out = Dict{Float64, Array{Float32, 3}}()
        ws_hir = nothing
        for E in energies
            μ_w_E  = BS.compute_μ_at_energy(XA.Materials.water, E)
            μρ_w_E = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water,  E))
            μρ_I_E = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E))

            vmi_sino = @. μρ_w_E * sino_w + μρ_I_E * sino_I
            sino_gpu = MtlArray(vmi_sino)

            if ws_hir === nothing
                ws_hir = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size;
                    strength = hir_strength_s1, filter = sim_vmi_filter)
                ws_hir.params = BS.HIRParams(hir_strength_s1, hir_lambda_s1, 30, hir_nepochs_s1,
                                              hir_n_subsets_s1, hir_huber_delta_s1, hir_relaxation_s1, (25, 35))
            end

            init_hu  = mono_out[E]
            init_μ   = Float32.(Float64(μ_w_E) .* (Float64.(init_hu) ./ 1000.0 .+ 1.0))
            init_gpu = MtlArray(init_μ)

            BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; init_volume = init_gpu)
            recon_μ  = Array(ws_hir.volume)
            recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_w_E))
            out[E]   = recon_hu

            roi_in  = init_hu[200:300, 200:300, mid_z]
            roi_out = recon_hu[200:300, 200:300, mid_z]
            @info "[scan 1 HIR  init=Mono+] $(Int(E)) keV: σ $(round(std(roi_in), digits=1)) → $(round(std(roi_out), digits=1)) HU"

            sino_gpu = nothing; init_gpu = nothing
        end
        ws_hir = nothing; GC.gc(true)
        out
    end
end;

# ╔═╡ 08135000-0000-4000-8000-000000000001
md"""
### Results

Cross-recon comparison grids at **Scan 1** (140 kVp / 52 mA / 3.03 mGy).
Rows: **Clin FBP · Sim FBP · Clin QIR3 · Sim HIR**.  Cols: **Poly · VMI 40 ·
VMI 70 · VMI 100 · VMI 140 keV**.

- **Poly column**: full 4-row comparison (FBP and iterative on both clinical and sim sides).
- **VMI columns**: only rows 3 (Clin QIR3) and 4 (Sim HIR) carry data — Scan 1 didn't acquire clinical FBP VMIs at 3 mGy and our sim pipeline reconstructs VMI with HIR-only (Cong → RWLS → HIR → Mono+).  FBP VMI cells render as "not acquired" placeholders.
"""

# ╔═╡ 08135010-0000-4000-8000-000000000001
md"**Segmentation.** Detected Gammex rods on clinical QIR3 slice vs simulated poly-FBP slice."

# ╔═╡ 08135010-0000-4000-8000-000000000002
sim_seg_result_s1 = let
    ref = sim_scan1_poly_fbp
    mid_z = size(ref, 3) ÷ 2
    mask, rods, center = segment_gammex_rods(ref[:, :, mid_z]; fov_cm = sim_fov_cm, clockwise = false)
    (mask = mask, rods = rods, center = center)
end;

# ╔═╡ 08135010-0000-4000-8000-000000000003
let
    fig = CM.Figure(size = (1200, 560), fontsize = 11)

    clin_slice = hu_140_low_qir[:, :, seg_result.slice_idx]
    ax1 = CM.Axis(fig[1, 1]; title = "Clinical QIR3", subtitle = "140 kVp / 52 mA",
        aspect = CM.DataAspect(), yreversed = true)
    CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = (-200, 500))
    for r in seg_result.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (35.0 / size(clin_slice, 1))
        CM.lines!(ax1, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th), color = :red, linewidth = 1.2)
        CM.text!(ax1, r.cx, r.cy - rpx - 2, text = r.name, fontsize = 7, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax1); CM.hidespines!(ax1)

    sim_mid_z = sim_matrix_size[3] ÷ 2
    sim_slice = sim_scan1_poly_fbp[:, :, sim_mid_z]
    ax2 = CM.Axis(fig[1, 2]; title = "Simulated Poly FBP", subtitle = "140 kVp / 52 mA",
        aspect = CM.DataAspect())
    CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = (-200, 500))
    for r in sim_seg_result_s1.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (sim_fov_cm / size(sim_slice, 1))
        CM.lines!(ax2, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th), color = :red, linewidth = 1.2)
        CM.text!(ax2, r.cx, r.cy - rpx - 2, text = r.name, fontsize = 7, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax2); CM.hidespines!(ax2)

    CM.save(joinpath(RESULTS_DIR, "alpha_scan1_segmentation_overlay.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08135020-0000-4000-8000-000000000001
# Measurements driven by sim_seg_result_s1.  Poly has both FBP and HIR; VMI
# is HIR-only (no FBP recon — Mono+ is the final step on the HIR output).
sim_measurements_scan1 = let
    seg = sim_seg_result_s1
    m(vol, name) = measure_scan(vol, seg.mask, seg.rods, seg.center, name; fov_cm = sim_fov_cm)
    (
        poly_fbp = m(sim_scan1_poly_fbp, "scan1_poly_fbp"),
        poly_hir = m(sim_scan1_poly_hir, "scan1_poly_hir"),
        vmi_hir  = Dict(E => m(sim_scan1_vmi_hir[E], "scan1_vmi_hir_$(Int(E))keV") for E in vmi_energies_s1),
    )
end;

# ╔═╡ 08135020-0000-4000-8000-000000000002
# 4×5 comparison grid.
#   Rows: Clin FBP · Sim FBP · Clin QIR3 · Sim HIR
#   Cols: Poly · VMI 40 · 70 · 100 · 140 keV
# Poly column has data in all 4 rows.  VMI columns: only rows 3 (Clin QIR3)
# and 4 (Sim HIR) — Scan 1 didn't acquire clinical FBP VMIs and we don't
# reconstruct simulated FBP VMIs in this pipeline (HIR is the only VMI recon).
scan1_grid = let
    row_labels = ["Clin FBP", "Sim FBP", "Clin QIR3", "Sim HIR"]
    col_labels = ["Poly", "VMI 40 keV", "VMI 70 keV", "VMI 100 keV", "VMI 140 keV"]
    vmi_E      = vmi_energies_s1

    clin_m_vmi_qir3(E) = first(filter(m -> m.name == "140kVp_52mA_VMI$(Int(E))_QIR3", clinical_measurements))

    vols = Matrix{Any}(undef, 4, 5)
    meas = Matrix{Any}(undef, 4, 5)

    # Row 1: Clinical FBP — poly only; VMI cells "not acquired".
    clin_m_poly_fbp = first(filter(m -> m.name == "140kVp_52mA_FBP",  clinical_measurements))
    vols[1, 1] = hu_140_low_fbp;        meas[1, 1] = clin_m_poly_fbp
    for c in 1:length(vmi_E)
        vols[1, c + 1] = nothing
        meas[1, c + 1] = nothing
    end

    # Row 2: Simulated FBP — poly only; VMI cells skipped (no FBP VMI recon).
    vols[2, 1] = sim_scan1_poly_fbp;    meas[2, 1] = sim_measurements_scan1.poly_fbp
    for c in 1:length(vmi_E)
        vols[2, c + 1] = nothing
        meas[2, c + 1] = nothing
    end

    # Row 3: Clinical QIR3 — poly + QIR3 VMIs.
    clin_m_poly_qir3 = first(filter(m -> m.name == "140kVp_52mA_QIR3", clinical_measurements))
    vols[3, 1] = hu_140_low_qir;        meas[3, 1] = clin_m_poly_qir3
    for (c, E) in enumerate(vmi_E)
        vols[3, c + 1] = E == 40.0  ? hu_140_low_vmi40  :
                         E == 70.0  ? hu_140_low_vmi70  :
                         E == 100.0 ? hu_140_low_vmi100 : hu_140_low_vmi140
        meas[3, c + 1] = clin_m_vmi_qir3(E)
    end

    # Row 4: Simulated HIR — poly + HIR-Mono+ VMIs (the headline output).
    vols[4, 1] = sim_scan1_poly_hir;    meas[4, 1] = sim_measurements_scan1.poly_hir
    for (c, E) in enumerate(vmi_E)
        vols[4, c + 1] = sim_scan1_vmi_hir[E]
        meas[4, c + 1] = sim_measurements_scan1.vmi_hir[E]
    end

    is_clinical = r -> r == 1 || r == 3
    slice_of = (row, vol) -> vol === nothing ? nothing :
        (is_clinical(row) ? vol[:, :, seg_result.slice_idx] : vol[:, :, size(vol, 3) ÷ 2])

    (rows = row_labels, cols = col_labels, vols = vols, meas = meas,
     is_clinical = is_clinical, slice_of = slice_of)
end;

# ╔═╡ 08135030-0000-4000-8000-000000000001
md"**Qualitative.** Soft-tissue window (−200, 500) HU across all cells."

# ╔═╡ 08135030-0000-4000-8000-000000000002
let
    g = scan1_grid
    fig = CM.Figure(size = (1400, 1180), fontsize = 11)
    for r in 1:4, c in 1:5
        ax = CM.Axis(fig[r, c];
            title    = r == 1 ? g.cols[c] : "",
            ylabel   = c == 1 ? g.rows[r] : "",
            aspect   = CM.DataAspect(),
            yreversed = g.is_clinical(r))
        slice = g.slice_of(r, g.vols[r, c])
        if slice === nothing
            ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
            CM.text!(ax, 0.5, 0.5; text = "not acquired", align = (:center, :center),
                     fontsize = 11, color = :gray35)
            CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
        else
            CM.heatmap!(ax, slice; colormap = :grays, colorrange = (-200, 500))
        end
        CM.hidedecorations!(ax; label = false)
        CM.hidespines!(ax)
    end
    CM.rowgap!(fig.layout, 4);  CM.colgap!(fig.layout, 4)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan1_qualitative_4x5.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08135040-0000-4000-8000-000000000001
md"**HU accuracy — simulated vs clinical.** 2×2 scatter (rows = rod category, cols = recon pairing).  Scan 1 FBP pairing uses clinical FBP Poly only (no clinical FBP VMIs at 3 mGy); Iterative pairing covers all 5 columns."

# ╔═╡ 08135040-0000-4000-8000-000000000002
let
    g = scan1_grid
    ca_idx = 4:8
    i_idx  = 10:16
    energy_colors = [:steelblue, :purple, :seagreen, :darkorange, :crimson]

    pairings = [
        ("FBP ↔ FBP",           1, 2),
        ("Iterative (QIR3 ↔ HIR)", 3, 4),
    ]
    cats = [("Calcium Rods", ca_idx), ("Iodine Rods", i_idx)]

    fig = CM.Figure(size = (1300, 1100), fontsize = 12)

    for (pr, (cat_name, cat_idx)) in enumerate(cats)
        for (pc, (pair_name, clin_row, sim_row)) in enumerate(pairings)
            ax = CM.Axis(fig[pr, pc];
                title    = "$cat_name — $pair_name",
                xlabel   = "Clinical HU",
                ylabel   = "Simulated HU",
                aspect   = 1.0)

            xs_all = Float64[]; ys_all = Float64[]
            for c in 1:length(g.cols)
                clin_m = g.meas[clin_row, c]
                sim_m  = g.meas[sim_row,  c]
                (clin_m === nothing || sim_m === nothing) && continue
                xs = Float64.(clin_m.rod_means[cat_idx])
                ys = Float64.(sim_m.rod_means[cat_idx])
                append!(xs_all, xs); append!(ys_all, ys)
                CM.scatter!(ax, xs, ys;
                    color = energy_colors[c],
                    markersize = 10,
                    strokecolor = :black, strokewidth = 0.3,
                    label = g.cols[c])
            end

            if isempty(xs_all)
                ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
                CM.text!(ax, 0.5, 0.5; text = "no data", align = (:center, :center),
                         fontsize = 11, color = :gray35)
                CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
                continue
            end

            lo = min(minimum(xs_all), minimum(ys_all))
            hi = max(maximum(xs_all), maximum(ys_all))
            pad = 0.05 * (hi - lo + eps())
            xlo, xhi = lo - pad, hi + pad

            CM.lines!(ax, [xlo, xhi], [xlo, xhi];
                color = (:gray60, 0.7), linestyle = :dash, linewidth = 1.0,
                label = "Unity (y = x)")

            n  = length(xs_all)
            Σx = sum(xs_all);  Σy = sum(ys_all)
            Σxx = sum(xs_all .* xs_all);  Σxy = sum(xs_all .* ys_all)
            slope     = (n * Σxy - Σx * Σy) / (n * Σxx - Σx^2)
            intercept = (Σy - slope * Σx) / n
            r_val     = cor(xs_all, ys_all)
            y_hat     = slope .* xs_all .+ intercept
            y_range   = maximum(ys_all) - minimum(ys_all)
            nrmse     = sqrt(mean((ys_all .- y_hat) .^ 2)) / max(y_range, eps()) * 100
            CM.lines!(ax, [xlo, xhi], [slope * xlo + intercept, slope * xhi + intercept];
                color = :black, linewidth = 1.6, label = "Linear fit")

            CM.xlims!(ax, xlo, xhi);  CM.ylims!(ax, xlo, xhi)

            sgn = intercept ≥ 0 ? "+" : "−"
            stat_txt =
                "y = $(round(slope, digits = 3))·x $sgn $(round(abs(intercept), digits = 1))\n" *
                "r = $(round(r_val, digits = 4))\n" *
                "nRMSE = $(round(nrmse, digits = 1))%"
            CM.text!(ax, xhi - 0.02 * (xhi - xlo), xlo + 0.04 * (xhi - xlo);
                text = stat_txt, align = (:right, :bottom),
                fontsize = 10, color = :black)

            if pr == 1 && pc == 1
                CM.axislegend(ax; position = :lt, labelsize = 9, framevisible = true)
            end
        end
    end

    CM.rowgap!(fig.layout, 18);  CM.colgap!(fig.layout, 18)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan1_hu_accuracy.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08135070-0000-4000-8000-000000000001
md"**Water noise (σ) summary.** 5 energy-groups × 4 bars (Clin FBP · Sim FBP · Clin QIR3 · Sim HIR).  Clinical FBP VMIs absent → those bars are blank."

# ╔═╡ 08135070-0000-4000-8000-000000000002
let
    g = scan1_grid
    water_idx = 1
    n_cols = length(g.cols)
    xs = collect(1:n_cols)
    bw = 0.18
    offsets = (-1.5, -0.5, 0.5, 1.5) .* bw

    σ = Matrix{Float64}(undef, 4, n_cols)
    for r in 1:4, c in 1:n_cols
        m = g.meas[r, c]
        σ[r, c] = m === nothing ? NaN : Float64(m.rod_stds[water_idx])
    end

    row_style = [
        ("Clinical FBP",  :steelblue,  :solid,   :steelblue),
        ("Simulated FBP", :darkorange, :solid,   :darkorange),
        ("Clinical QIR3", :steelblue,  :outline, :steelblue),
        ("Simulated HIR", :darkorange, :outline, :darkorange),
    ]

    fig = CM.Figure(size = (1150, 500), fontsize = 12)
    ax  = CM.Axis(fig[1, 1];
        title  = "Scan 1 (140 kVp / 52 mA / 3 mGy) — Water ROI noise",
        xlabel = "Reconstruction / Energy",
        ylabel = "Water σ (HU)",
        xticks = (xs, g.cols))

    for (r, (lab, base_color, style, stroke_color)) in enumerate(row_style)
        y = σ[r, :]
        x_r = xs .+ offsets[r]
        if style == :solid
            CM.barplot!(ax, x_r, y;
                width = bw, color = base_color, strokecolor = stroke_color,
                strokewidth = 0.5, label = lab)
        else
            CM.barplot!(ax, x_r, y;
                width = bw, color = (base_color, 0.12),
                strokecolor = stroke_color, strokewidth = 1.8, label = lab)
        end
        for (xi, yi) in zip(x_r, y)
            isnan(yi) && continue
            CM.text!(ax, xi, yi; text = string(round(yi, digits = 1)),
                align = (:center, :bottom), fontsize = 8, color = :gray20, offset = (0, 2))
        end
    end

    CM.ylims!(ax, 0, nothing)
    CM.axislegend(ax; position = :rt, labelsize = 10, framevisible = true)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan1_noise_bar.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08135050-0000-4000-8000-000000000001
md"**MTF — Clinical vs Simulated.** 2×5 grid: rows = recon pairing (FBP · Iterative); cols = energy.  FBP VMI columns lack clinical data at 3 mGy → show simulated alone."

# ╔═╡ 08135050-0000-4000-8000-000000000002
let
    g = scan1_grid
    pairings = [("FBP",       1, 2),
                ("Iterative", 3, 4)]
    n_cols = length(g.cols)

    fig = CM.Figure(size = (1500, 620), fontsize = 11)

    for (pr, (pair_label, clin_row, sim_row)) in enumerate(pairings)
        for c in 1:n_cols
            ax = CM.Axis(fig[pr, c];
                title    = pr == 1 ? g.cols[c] : "",
                subtitle = pr == 1 ? "Clinical vs Simulated" : "",
                xlabel   = pr == 2 ? "Spatial frequency (lp/mm)" : "",
                ylabel   = c == 1 ? "$(pair_label)\nMTF" : "")
            clin = g.meas[clin_row, c]
            sim  = g.meas[sim_row,  c]

            has_clin = clin !== nothing && hasproperty(clin, :mtf) && clin.mtf !== nothing
            has_sim  = sim  !== nothing && hasproperty(sim,  :mtf) && sim.mtf  !== nothing

            if !has_clin && !has_sim
                ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
                CM.text!(ax, 0.5, 0.5; text = "no data", align = (:center, :center),
                         fontsize = 10, color = :gray35)
                CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
                CM.hidedecorations!(ax; label = false)
                continue
            end

            has_clin && CM.lines!(ax, clin.mtf.frequencies, clin.mtf.mtf;
                                  color = :steelblue, linewidth = 1.8, label = "Clinical")
            has_sim  && CM.lines!(ax, sim.mtf.frequencies,  sim.mtf.mtf;
                                  color = :darkorange, linewidth = 1.8,
                                  linestyle = :dash,   label = "Simulated")

            CM.hlines!(ax, [0.5, 0.1]; color = :gray70, linestyle = :dash, linewidth = 0.5)
            CM.ylims!(ax, 0, 1.1)

            lines = String[]
            has_clin && push!(lines, "clin f50=$(round(clin.mtf_f50, digits = 2))")
            has_sim  && push!(lines,  "sim  f50=$(round(sim.mtf_f50,  digits = 2))")
            CM.text!(ax, 0.96 * maximum(has_clin ? clin.mtf.frequencies : sim.mtf.frequencies), 1.07;
                text = join(lines, "\n"), align = (:right, :top), fontsize = 8, color = :gray30)

            if pr == 1 && c == 1
                CM.axislegend(ax; position = :rt, labelsize = 9, framevisible = true)
            end
        end
    end
    CM.rowgap!(fig.layout, 12);  CM.colgap!(fig.layout, 10)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan1_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08135060-0000-4000-8000-000000000001
md"**NPS — Clinical vs Simulated.** 2×5 grid; same layout as MTF.  y-axis shared per column."

# ╔═╡ 08135060-0000-4000-8000-000000000002
let
    g = scan1_grid
    pairings = [("FBP",       1, 2),
                ("Iterative", 3, 4)]
    n_cols = length(g.cols)

    col_ymax = map(1:n_cols) do c
        ymax = 0.0
        for (_, cr, sr) in pairings, row in (cr, sr)
            m = g.meas[row, c]
            if m !== nothing && hasproperty(m, :nps) && m.nps !== nothing
                ymax = max(ymax, maximum(m.nps.nps_1d))
            end
        end
        ymax == 0.0 ? 1.0 : ymax * 1.05
    end

    fig = CM.Figure(size = (1500, 620), fontsize = 11)

    for (pr, (pair_label, clin_row, sim_row)) in enumerate(pairings)
        for c in 1:n_cols
            ax = CM.Axis(fig[pr, c];
                title    = pr == 1 ? g.cols[c] : "",
                subtitle = pr == 1 ? "Clinical vs Simulated" : "",
                xlabel   = pr == 2 ? "Spatial frequency (lp/mm)" : "",
                ylabel   = c == 1 ? "$(pair_label)\nNPS" : "")
            clin = g.meas[clin_row, c]
            sim  = g.meas[sim_row,  c]

            has_clin = clin !== nothing && hasproperty(clin, :nps) && clin.nps !== nothing
            has_sim  = sim  !== nothing && hasproperty(sim,  :nps) && sim.nps  !== nothing

            if !has_clin && !has_sim
                ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
                CM.text!(ax, 0.5, 0.5; text = "no data", align = (:center, :center),
                         fontsize = 10, color = :gray35)
                CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
                CM.hidedecorations!(ax; label = false)
                continue
            end

            has_clin && CM.lines!(ax, clin.nps.frequencies, clin.nps.nps_1d;
                                  color = :steelblue,  linewidth = 1.8, label = "Clinical")
            has_sim  && CM.lines!(ax, sim.nps.frequencies,  sim.nps.nps_1d;
                                  color = :darkorange, linewidth = 1.8,
                                  linestyle = :dash,   label = "Simulated")
            CM.ylims!(ax, 0, col_ymax[c])

            lines = String[]
            has_clin && push!(lines, "clin peak=$(round(clin.nps_peak_freq, digits = 2))  ∫=$(round(clin.nps_area, sigdigits = 3))")
            has_sim  && push!(lines,  "sim  peak=$(round(sim.nps_peak_freq,  digits = 2))  ∫=$(round(sim.nps_area,  sigdigits = 3))")
            CM.text!(ax, 0.96 * maximum(has_clin ? clin.nps.frequencies : sim.nps.frequencies), col_ymax[c];
                text = join(lines, "\n"), align = (:right, :top), fontsize = 8, color = :gray30)

            if pr == 1 && c == 1
                CM.axislegend(ax; position = :rt, labelsize = 9, framevisible = true)
            end
        end
    end
    CM.rowgap!(fig.layout, 12);  CM.colgap!(fig.layout, 10)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan1_nps.png"), fig, px_per_unit = 2)
    fig
end

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

# ╔═╡ 08120003-0000-4000-8000-000000000000
md"**Polyenergetic HIR (strength=3)** — iterative counterpart to the poly FBP; matches the row-4 Poly cell of the Scan 2 comparison grids below."

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

# ╔═╡ 08120002-0000-4000-8000-000000000000
# Poly FBP — recombined from the per-bin scatter-corrected sinograms
# (`sim_scan2_bins_corrected`), then FDK.  Same preprocessing rigor as the
# VMI pipeline: each bin got its own scatter fraction (Σ frac_b = 1 via
# compute_scatter_bin_weights) instead of a single I0-averaged correction
# on the combined transmission.
sim_scan2_poly_combined = let
    I0       = sim_scan2.I0_bins
    I0_total = Float32(sum(I0))
    combined = zeros(Float32, size(sim_scan2_bins_corrected[1]))
    for (b, h) in enumerate(sim_scan2_bins_corrected)
        @. combined += Float32(I0[b]) * exp(-h)
    end
    @. combined = -log(max(combined, Float32(1e-10)) / I0_total)
    combined
end;

# ╔═╡ 08120002-a000-4000-8000-000000000001
sim_scan2_poly_fbp = let
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    sino_gpu = MtlArray(sim_scan2_poly_combined)
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = sim_custom_poly_filter)
    recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_fdk = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end;

# ╔═╡ 08120004-0000-4000-8000-000000000000
# Poly HIR — same recombined scatter-corrected combined sinogram as poly FBP,
# then HIR (strength 3).
sim_scan2_poly_hir = let
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    sino_gpu = MtlArray(sim_scan2_poly_combined)
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = 3, filter = sim_custom_poly_filter)
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
    recon_μ = Array(ws_hir.volume)

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    ws_hir = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end;

# ╔═╡ 08120005-0000-4000-8000-000000000000
md"""
**Low/high bin combination.** Combine 4 PCCT threshold bins into 2 effective
sinograms for VMI: low = bins 1+2 (20–55 keV, photoelectric / iodine); high =
bins 3+4 (>55 keV, Compton).  Count-domain combine:
`sino = -log( (I0_a·exp(-sino_a) + I0_b·exp(-sino_b)) / (I0_a + I0_b) )`.
"""

# ╔═╡ 08120006-a000-4000-8000-000000000001
# ── PCCT 4-bin → 2-bin grouping for Cong + PWLS (scan 2) ──
# Default: low = bins 1+2 (20–55 keV), high = bins 3+4 (>55 keV).  Adjust to
# explore other splits — e.g. [[1], [2, 3, 4]] for a more aggressive separation.
pcct_lohi_grouping_s2 = [[1, 2], [3, 4]];

# ╔═╡ 08120006-0000-4000-8000-000000000000
# Combine bins → low and high sinograms per `pcct_lohi_grouping_s2` knob.
sim_scan2_lohi = let
    bins = sim_scan2_bins_corrected   # scatter-corrected bins
    I0   = sim_scan2.I0_bins

    function combine_bins(bin_indices, bins, I0)
        I0_sum = sum(I0[b] for b in bin_indices)
        counts = zeros(Float32, size(bins[1]))
        for b in bin_indices
            @. counts += Float32(I0[b]) * exp(-bins[b])
        end
        @. -log(max(counts, Float32(1e-10)) / Float32(I0_sum))
    end

    grp_low  = pcct_lohi_grouping_s2[1]
    grp_high = pcct_lohi_grouping_s2[2]

    sino_low  = combine_bins(grp_low,  bins, I0)
    sino_high = combine_bins(grp_high, bins, I0)
    I0_low    = sum(I0[b] for b in grp_low)
    I0_high   = sum(I0[b] for b in grp_high)

    @info "[scan 2] low  bin grp=$(grp_low):  I0=$(round(I0_low, sigdigits=4)),  mean sino=$(round(mean(sino_low),  digits=3))"
    @info "[scan 2] high bin grp=$(grp_high): I0=$(round(I0_high, sigdigits=4)), mean sino=$(round(mean(sino_high), digits=3))"

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
**Material decomposition (Scan 2).** Chain: **bin grouping → Cong (warm start) → PWLS-L₂** →
single `(sino_w, sino_I)` pair.  Each stage has a `use_*` toggle + hyperparam
cell; off → pass-through.  Intermediate viz after each stage shows
iodine/water sinograms + quick FBP.

- **Cong 2022** — per-ray analytic DE decomp in the photoelectric + Compton
  basis with empirical (p, q) and an optional 3D bowtie-aware ŵ.  Canonical
  PWLS warm start.
- **PWLS-L₂** (Long/Fessler 2014) — 2-bin sinogram-domain restoration with
  2×2 matrix-curvature SQS.  Reuses Cong's basis; output `(sino_w, sino_I)`
  feeds VMI synthesis.
"""

# ╔═╡ 08120e00-0000-4000-8000-000000000001
# Shared viz helpers used by the per-stage intermediate viz cells below.
# _decomp_viz   — 2×2 heatmap: (∫ρ_I·dr, ∫ρ_W·dr) sinograms + (ρ_I, ρ_W) FBP.
# _vmi_row_viz  — 1×4 row: VMI HU at each target energy for one Dict of volumes.
function _decomp_viz(sino_I, sino_W, stage_name; geom = sim_scan2.geom)
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

    # 2×4 grid: row-1 sinos span 2 cols each; row-2 pairs FBP with its colorbar.
    fig = CM.Figure(size = (1300, 720), fontsize = 11)

    ax1 = CM.Axis(fig[1, 1:2]; title = "$stage_name — ∫ρ_I·dr (sino mid-view)", aspect = CM.DataAspect())
    CM.heatmap!(ax1, sino_I[:, mid_v, :]; colormap = :viridis, colorrange = (I_lo, I_hi))

    ax2 = CM.Axis(fig[1, 3:4]; title = "$stage_name — ∫ρ_W·dr (sino mid-view)", aspect = CM.DataAspect())
    CM.heatmap!(ax2, sino_W[:, mid_v, :]; colormap = :viridis, colorrange = (W_lo, W_hi))

    ax3 = CM.Axis(fig[2, 1]; title = "$stage_name — ρ_I FBP (slice $mid_z) [g/cm³]", aspect = CM.DataAspect())
    hm3 = CM.heatmap!(ax3, fbp_I[:, :, mid_z]; colormap = :viridis, colorrange = (fI_lo, fI_hi))
    CM.Colorbar(fig[2, 2], hm3; label = "ρ_I [g/cm³]", labelrotation = 0, width = 12, tickalign = 1)

    ax4 = CM.Axis(fig[2, 3]; title = "$stage_name — ρ_W FBP (slice $mid_z) [g/cm³]", aspect = CM.DataAspect())
    hm4 = CM.heatmap!(ax4, fbp_W[:, :, mid_z]; colormap = :viridis, colorrange = (fW_lo, fW_hi))
    CM.Colorbar(fig[2, 4], hm4; label = "ρ_W [g/cm³]", labelrotation = 0, width = 12, tickalign = 1)

    # Keep the colorbars tight against the FBP panels.
    CM.colgap!(fig.layout, 1, 20)   # between sino_I span and sino_W span / FBP_I and cbar_I
    CM.colgap!(fig.layout, 2, 6)    # cbar_I ↔ FBP_W
    CM.colgap!(fig.layout, 3, 6)    # FBP_W ↔ cbar_W
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

# ╔═╡ 08131e02-0000-4000-8000-000000000005
let
    _decomp_viz(cong_decomp_s1.sino_iodine, cong_decomp_s1.sino_water, "Cong (inline)"; geom = sim_scan1.geom)
end

# ╔═╡ 08131e03-0000-4000-8000-000000000002
let
    _decomp_viz(pwls_decomp_s1.sino_I, pwls_decomp_s1.sino_w, "PWLS-L₂"; geom = sim_scan1.geom)
end

# ╔═╡ 08131f02-0000-4000-8000-000000000002
let
    tag = use_hir_s1 ? "HIR (init = Mono+ FBP)" : "Mono+ FBP pass-through"
    _vmi_row_viz(sim_scan1_vmi_hir, tag)
end

# ╔═╡ 08120e02-0000-4000-8000-000000000001
md"""
**Cong 2022 (inline).** Per-ray analytic DE decomposition in the photoelectric + Compton basis.  Identical setup to Scan 1 — empirical (p, q) fit to (water, iodine) + optional 3D bowtie-aware ŵ + per-g/cm³ basis change producing `sino_iodine` in g/cm² (mass-density line integrals).  Output is the canonical RWLS warm start.
"""

# ╔═╡ 08120e02-0000-4000-8000-000000000002
# Scan 2 Cong config — mirror of Scan 1's `cong_*_s1` defaults.
begin
    cong_run_on_cpu_s2        = false      # true → run apply_cong! on CPU (isolates Metal issues)
    cong_use_bowtie_s2        = true       # true → per-ray 3D ŵ[col,row,E] (bowtie-aware); false → 1D centered
    cong_low_bins_s2          = [1, 2]     # low PCCT channel
    cong_high_bins_s2         = [3, 4]     # high PCCT channel
    cong_newton_max_iter_s2   = 5
    cong_newton_tol_s2        = eps(Float32)
    cong_y_max_factor_s2      = 2.0
    cong_y_max_cap_s2         = Float32(1e7)
end

# ╔═╡ 08120e02-0000-4000-8000-000000000003
# Scan 2 Cong PCCT basis — empirical (p, q) fit to (water, iodine) μ(E),
# per-g/cm³ A_coef, optional bowtie-aware 3D ŵ.  See Scan 1 basis cell for
# the full explanation; this is a straight 1:1 port.
cong_basis_s2 = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e_full, ŵ_bins = BS.pcct_effective_spectrum(
        sim_scanner, prot;
        sim_opts   = sim_opts,
        bin_groups = [cong_low_bins_s2, cong_high_bins_s2],
    )

    atomic_mass = Dict{Int, Float64}(
        let el = getproperty(XA.Elements, n)
            el.Z => el.Z / el.ZA_ratio
        end
        for n in propertynames(XA.Elements)
    )
    function physical_ac(mat)
        ρ = Unitful.ustrip(Unitful.u"g/cm^3", mat.density)
        if hasproperty(mat, :composition)
            a = 0.0; c = 0.0
            for (Z, m_frac) in mat.composition
                A = atomic_mass[Z]
                a += m_frac * Z^4 / A
                c += m_frac * Z / A
            end
            (ρ * a, ρ * c)
        else
            Z = mat.Z;  A = Z / mat.ZA_ratio
            (ρ * Z^4 / A, ρ * Z / A)
        end
    end

    # Per-g/cm³ A_coef (iodine row stripped of ρ_iodine_pure so output is in
    # mass density units — drop-in with CMV).
    ref_mats = [XA.Materials.water, XA.Elements.Iodine]
    A_coef = Matrix{Float64}(undef, length(ref_mats), 2)
    for (i, m) in enumerate(ref_mats)
        ρ_m = Unitful.ustrip(Unitful.u"g/cm^3", m.density)
        a_ρ, c_ρ = physical_ac(m)
        A_coef[i, 1] = a_ρ / ρ_m
        A_coef[i, 2] = c_ρ / ρ_m
    end
    function fit_pq(energies)
        p = zeros(Float32, length(energies))
        q = zeros(Float32, length(energies))
        for (i, E) in enumerate(energies)
            μρ = Float64[BS.compute_mass_μ_at_energy(m, Float64(E)) for m in ref_mats]
            sol = A_coef \ μρ
            p[i] = Float32(sol[1]);  q[i] = Float32(sol[2])
        end
        p, q
    end

    p_L, q_L = fit_pq(e_full)
    p_H, q_H = fit_pq(e_full)

    if !cong_use_bowtie_s2
        ŵ_L = Float32.(Float64.(ŵ_bins[1]) ./ sum(Float64.(ŵ_bins[1])))
        ŵ_H = Float32.(Float64.(ŵ_bins[2]) ./ sum(Float64.(ŵ_bins[2])))
        @info "[Cong basis S2]  1D centered spectrum (no bowtie)   nE=$(length(e_full))"
    else
        _, w_src = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
        ŵ_srcB   = BS.apply_bowtie_to_spectrum(w_src, e_full, sim_scanner, sim_scan2.geom, prot;
                                                include_bowtie = true, label = "Cong S2")
        pcct_det = BS._build_pcct_detector(sim_scanner)
        kVp      = Float64(maximum(e_full))
        R_mat    = BS.compute_mc_drm(pcct_det, kVp)
        η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
        n_R      = size(R_mat, 1)
        drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)
        n_E      = length(e_full)
        drm_col_sum_L = [sum(R_mat[drm_row(e_full[i]), b] for b in cong_low_bins_s2)  for i in 1:n_E]
        drm_col_sum_H = [sum(R_mat[drm_row(e_full[i]), b] for b in cong_high_bins_s2) for i in 1:n_E]

        function _bake_3d(drm_col)
            if ndims(ŵ_srcB) == 3
                n_col, n_row = size(ŵ_srcB, 1), size(ŵ_srcB, 2)
                out = Array{Float32, 3}(undef, n_col, n_row, n_E)
                @inbounds for ci in 1:n_col, ri in 1:n_row
                    s = 0.0
                    for Ei in 1:n_E
                        w = Float64(ŵ_srcB[ci, ri, Ei]) * Float64(η_vec[Ei]) * Float64(drm_col[Ei])
                        out[ci, ri, Ei] = Float32(w)
                        s += w
                    end
                    inv_s = Float32(1.0 / max(s, 1e-20))
                    for Ei in 1:n_E
                        out[ci, ri, Ei] *= inv_s
                    end
                end
                out
            else
                w1d = [Float64(ŵ_srcB[i]) * Float64(η_vec[i]) * Float64(drm_col[i]) for i in 1:n_E]
                Float32.(w1d ./ sum(w1d))
            end
        end

        ŵ_L = _bake_3d(drm_col_sum_L)
        ŵ_H = _bake_3d(drm_col_sum_H)

        if ndims(ŵ_L) == 3
            mid_c = size(ŵ_L, 1) ÷ 2 + 1
            mid_r = size(ŵ_L, 2) ÷ 2 + 1
            e_ctr_L = sum(Float64(e_full[k]) * ŵ_L[mid_c, mid_r, k] for k in 1:n_E)
            e_ctr_H = sum(Float64(e_full[k]) * ŵ_H[mid_c, mid_r, k] for k in 1:n_E)
            e_edg_L = sum(Float64(e_full[k]) * ŵ_L[1,     mid_r, k] for k in 1:n_E)
            e_edg_H = sum(Float64(e_full[k]) * ŵ_H[1,     mid_r, k] for k in 1:n_E)
            @info "[Cong basis S2]  3D bowtie-aware   size=$(size(ŵ_L))   ⟨E⟩_L: ctr=$(round(e_ctr_L,digits=2)) edge=$(round(e_edg_L,digits=2)) keV (Δ=$(round(e_edg_L-e_ctr_L,digits=2)))"
            @info "[Cong basis S2]                                       ⟨E⟩_H: ctr=$(round(e_ctr_H,digits=2)) edge=$(round(e_edg_H,digits=2)) keV (Δ=$(round(e_edg_H-e_ctr_H,digits=2)))"
        else
            e_m_L = sum(Float64.(e_full) .* Float64.(ŵ_L))
            e_m_H = sum(Float64.(e_full) .* Float64.(ŵ_H))
            @info "[Cong basis S2]  scanner has no bowtie — fell back to 1D   ⟨E⟩_L=$(round(e_m_L,digits=2)) ⟨E⟩_H=$(round(e_m_H,digits=2)) keV"
        end
    end

    (ŵ_L = ŵ_L, ŵ_H = ŵ_H,
     p_L = p_L, p_H = p_H,
     q_L = q_L, q_H = q_H,
     A_coef = A_coef)
end;

# ╔═╡ 08120e02-0000-4000-8000-000000000004
# Scan 2 Cong per-ray decomp + basis change → (sino_water, sino_iodine) in
# g/cm² mass line integrals (drop-in with CMV / RWLS / PWLS downstream).
cong_decomp_s2 = let
    sl_cpu = Float32.(sim_scan2_lohi.sino_low)
    sh_cpu = Float32.(sim_scan2_lohi.sino_high)
    n_air_low  = count(<(1f-6), sl_cpu)
    n_air_high = count(<(1f-6), sh_cpu)
    @info "[Cong S2 inputs]  sino_low  min=$(round(minimum(sl_cpu),digits=4)) max=$(round(maximum(sl_cpu),digits=4)) mean=$(round(mean(sl_cpu),digits=4))  n<1e-6=$(n_air_low)/$(length(sl_cpu))"
    @info "[Cong S2 inputs]  sino_high min=$(round(minimum(sh_cpu),digits=4)) max=$(round(maximum(sh_cpu),digits=4)) mean=$(round(mean(sh_cpu),digits=4))  n<1e-6=$(n_air_high)/$(length(sh_cpu))"

    p_L = Float64.(cong_basis_s2.p_L);  q_L = Float64.(cong_basis_s2.q_L)
    @info "[Cong S2 basis]  nE=$(length(p_L))   p_L range=[$(round(minimum(p_L),sigdigits=3)), $(round(maximum(p_L),sigdigits=3))]   q_L range=[$(round(minimum(q_L),sigdigits=3)), $(round(maximum(q_L),sigdigits=3))]"
    let ŵ = cong_basis_s2.ŵ_L
        if ndims(ŵ) == 3
            rs = dropdims(sum(Float64.(ŵ); dims = 3); dims = 3)
            @info "[Cong S2 basis]  ŵ_L per-ray Σ range = [$(round(minimum(rs), digits=6)), $(round(maximum(rs), digits=6))]  (3D, should all be ~1.0)"
        else
            @info "[Cong S2 basis]  ŵ_L Σ = $(round(sum(ŵ), digits=6))  (1D, should be ~1.0)"
        end
    end

    if cong_run_on_cpu_s2
        @info "[Cong S2] running on CPU"
        sy_cpu = similar(sl_cpu);  fill!(sy_cpu, 0f0)
        sc_cpu = similar(sh_cpu);  fill!(sc_cpu, 0f0)
        ws_cong_s2 = BS.create_cong_workspace(sl_cpu, cong_basis_s2)
        t0 = time()
        BS.apply_cong!(
            ws_cong_s2, sy_cpu, sc_cpu, sl_cpu, sh_cpu;
            water_basis      = BS.water_basis_constants(),
            newton_max_iter  = cong_newton_max_iter_s2,
            newton_tol       = cong_newton_tol_s2,
            y_max_factor     = cong_y_max_factor_s2,
            y_max_cap        = cong_y_max_cap_s2,
        )
        @info "[Cong S2] CPU apply_cong! done in $(round(time()-t0, digits=1)) s"
        sino_y = sy_cpu
        sino_c = sc_cpu
    else
        @info "[Cong S2] running on Metal GPU"
        sl_gpu = MtlArray(sl_cpu)
        sh_gpu = MtlArray(sh_cpu)
        sy_gpu = similar(sl_gpu);  fill!(sy_gpu, 0f0)
        sc_gpu = similar(sl_gpu);  fill!(sc_gpu, 0f0)
        ws_cong_s2 = BS.create_cong_workspace(sl_gpu, cong_basis_s2)
        t0 = time()
        BS.apply_cong!(
            ws_cong_s2, sy_gpu, sc_gpu, sl_gpu, sh_gpu;
            water_basis      = BS.water_basis_constants(),
            newton_max_iter  = cong_newton_max_iter_s2,
            newton_tol       = cong_newton_tol_s2,
            y_max_factor     = cong_y_max_factor_s2,
            y_max_cap        = cong_y_max_cap_s2,
        )
        @info "[Cong S2] GPU apply_cong! done in $(round(time()-t0, digits=1)) s"
        sino_y = Array(sy_gpu)
        sino_c = Array(sc_gpu)
        sl_gpu = nothing; sh_gpu = nothing; sy_gpu = nothing; sc_gpu = nothing
        ws_cong_s2 = nothing
        GC.gc(true)
    end

    n_y_zero = count(iszero, sino_y)
    n_c_zero = count(iszero, sino_c)
    @info "[Cong S2 raw]   sino_y  min=$(round(minimum(sino_y),sigdigits=4)) max=$(round(maximum(sino_y),sigdigits=4)) mean=$(round(mean(sino_y),sigdigits=4))  nzero=$(n_y_zero)/$(length(sino_y))"
    @info "[Cong S2 raw]   sino_c  min=$(round(minimum(sino_c),sigdigits=4)) max=$(round(maximum(sino_c),sigdigits=4)) mean=$(round(mean(sino_c),sigdigits=4))  nzero=$(n_c_zero)/$(length(sino_c))"

    # Basis change — A_coef is per-g/cm³ so recovered (t_W, t_I) are mass
    # density line integrals in g/cm², matching CMV.
    a_w, c_w = cong_basis_s2.A_coef[1, :]
    a_I, c_I = cong_basis_s2.A_coef[2, :]
    M_ac  = [a_w a_I; c_w c_I]
    M_inv = inv(M_ac)

    m_W_y, m_W_c = Float32(M_inv[1, 1]), Float32(M_inv[1, 2])
    m_I_y, m_I_c = Float32(M_inv[2, 1]), Float32(M_inv[2, 2])

    @info "[Cong S2 basis change]  a_w=$(round(a_w,digits=3)) c_w=$(round(c_w,digits=4))   a_I=$(round(a_I,sigdigits=4)) c_I=$(round(c_I,digits=3))"
    @info "                        M_inv rows: W ← [$(round(m_W_y,sigdigits=4)) $(round(m_W_c,sigdigits=4))]   I ← [$(round(m_I_y,sigdigits=4)) $(round(m_I_c,sigdigits=4))]"

    sino_water  = @. m_W_y * sino_y + m_W_c * sino_c
    sino_iodine = @. m_I_y * sino_y + m_I_c * sino_c

    @info "[Cong S2 decomp]  ⟨∫ρ_W·dr⟩ = $(round(mean(sino_water), sigdigits=4)) cm   ⟨∫ρ_I·dr⟩ = $(round(mean(sino_iodine), sigdigits=4)) g/cm²"

    (sino_iodine = sino_iodine,
     sino_water  = sino_water,
     sino_y      = sino_y,
     sino_c      = sino_c,
     M_inv       = M_inv,
     geom        = sim_scan2.geom)
end;

# ╔═╡ 08120e02-0000-4000-8000-000000000005
let
    _decomp_viz(cong_decomp_s2.sino_iodine, cong_decomp_s2.sino_water, "Cong (inline)"; geom = sim_scan2.geom)
end

# ╔═╡ 08120e03-0000-4000-8000-000000000001
md"""
**PWLS-L₂ (Long/Fessler 2014).** 2-bin (low, high) sinogram-domain restoration
via 2×2 matrix-curvature SQS.  Warm-started from Cong; reuses the same 2-bin
basis Cong already built (`cong_basis_s2.{ŵ_L, ŵ_H, p_L, p_H, q_L, q_H}`).
When `use_pwls_s2 = false`, passes the warm start through unchanged.
"""

# ╔═╡ 08120009-a000-4000-8000-000000000011
# ── Scan 2 PWLS-L₂ config (Long/Fessler 2014, 2×2 matrix curvature) ──
# Independent from scan 1's PWLS knobs — tune separately since dose differs.
# PWLS uses the same 2-bin (low, high) basis that Cong already built
# (cong_basis_s2.{ŵ_L, ŵ_H, p_L, p_H, q_L, q_H}).  Smaller κ = bigger smoothing
# step per iter (per src docstring's De Pierro row-sum bound).  Tuned defaults
# from notebook 06 — adjust if needed.
begin
    use_pwls_s2   = true
    pwls_n_iter_s2   =20
    pwls_κ_iodine_s2 = 32.0
    pwls_κ_water_s2  = 32.0
    pwls_relax_s2    = 1.0
end

# ╔═╡ 08120010-0000-4000-8000-000000000040
# Scan 2 PWLS-L₂ sinogram restoration → src `BS.apply_pwls!`.
# Warm start = Cong (`cong_decomp_s2`).  Output = `pwls_decomp_s2.{sino_w, sino_I}`
# (matching notebook 07's existing field naming for downstream consumers).
# When `use_pwls_s2 = false`, passes Cong through unchanged.
pwls_decomp_s2 = let
    if !use_pwls_s2
        @info "[PWLS s2] DISABLED — passing through Cong warm start"
        (sino_w = Float32.(cong_decomp_s2.sino_water),
         sino_I = Float32.(cong_decomp_s2.sino_iodine))
    else
        sino_I_gpu = MtlArray(copy(cong_decomp_s2.sino_iodine))
        sino_W_gpu = MtlArray(copy(cong_decomp_s2.sino_water))
        h_low_gpu  = MtlArray(Float32.(sim_scan2_lohi.sino_low))
        h_high_gpu = MtlArray(Float32.(sim_scan2_lohi.sino_high))

        basis = (
            ŵ_bins = [cong_basis_s2.ŵ_L, cong_basis_s2.ŵ_H],
            p_bins = [cong_basis_s2.p_L, cong_basis_s2.p_H],
            q_bins = [cong_basis_s2.q_L, cong_basis_s2.q_H],
        )

        ws_pwls_s2 = BS.create_pwls_workspace(sino_I_gpu)

        info = BS.apply_pwls!(
            ws_pwls_s2, sino_I_gpu, sino_W_gpu, h_low_gpu, h_high_gpu;
            basis    = basis,
            I0_L     = sim_scan2_lohi.I0_low,    # PCCT-correct Poisson weighting:
            I0_H     = sim_scan2_lohi.I0_high,   # bin-grouped I0 differs significantly between low/high.
            n_iter   = pwls_n_iter_s2,
            κ_iodine = pwls_κ_iodine_s2,
            κ_water  = pwls_κ_water_s2,
            relax    = pwls_relax_s2,
            verbose  = true,
        )

        sino_I_cpu = Array(sino_I_gpu)
        sino_W_cpu = Array(sino_W_gpu)
        sino_I_gpu = nothing; sino_W_gpu = nothing
        h_low_gpu  = nothing; h_high_gpu = nothing
        ws_pwls_s2 = nothing
        GC.gc(true)

        (sino_w = sino_W_cpu, sino_I = sino_I_cpu,
         n_iter = info.n_iter, κ_iodine = Float64(info.κ_iodine), κ_water = Float64(info.κ_water))
    end
end;

# ╔═╡ 08120e03-0000-4000-8000-000000000002
# PWLS viz — single material decomp (sino + FBP of ρ_I, ρ_W).
let
    _decomp_viz(pwls_decomp_s2.sino_I, pwls_decomp_s2.sino_w, "PWLS-L₂")
end

# ╔═╡ 08120e05-0000-4000-8000-000000000001

# ╔═╡ 08120f00-0000-4000-8000-000000000001
md"""
**VMI synthesis.** Consumes the post-RWLS material decomp and synthesises
per-energy VMI HU volumes at target keV.

- **VMI FBP** — `vmi_sino(E) = μρ_w(E)·sino_w + μρ_I(E)·sino_I` → FDK → HU.
- **VMI HIR** — optional Huber-regularised replacement for plain FBP.
- **Mono+** — frequency-split polish (Grant 2014), per-energy σ.

Output → `sim_scan2_vmi`.
"""

# ╔═╡ 08120009-a000-4000-8000-000000000001
# §12e pipeline — common setting: VMI target energies.
# Changing this re-runs the VMI synthesis but not the material-decomp stages.
vmi_energies = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 08120f01-0000-4000-8000-000000000001
md"**VMI FBP.**"

# ╔═╡ 08120010-a000-4000-8000-000000000001
# Scan 2 VMI FBP — per-energy line-integral synthesis with **per-E FBP apod**.
# Reads from `pwls_decomp_s2`.  Output: `vmi_fbp.vmi[E]` per target keV.
vmi_fbp = let
    geom       = sim_scan2.geom
    recon_size = sim_matrix_size
    sino_w     = pwls_decomp_s2.sino_w
    sino_I     = pwls_decomp_s2.sino_I
    energies   = Float64.(vmi_energies)
    mid_z      = recon_size[3] ÷ 2

    # ── Per-E FBP apodization (mirrors nb 06's tuned per-energy filter) ──
    vmi_apod_x = (0.0, 0.25, 0.5, 0.75, 1.0)
    vmi_apod_y_per_E_s2 = [
        (1.0, 0.72, 0.45, 0.20, 0.05),    # 40 keV  — aggressive, HF replaced by VMI_70 via Mono+
        (1.0, 0.85, 0.65, 0.40, 0.15),    # 70 keV  — calibration anchor
        (1.0, 0.70, 0.42, 0.18, 0.04),    # 100 keV — Mono+ identity here, NPS softer than 70
        (1.0, 0.50, 0.30, 0.12, 0.03),    # 140 keV — LF tightened, HF replaced by VMI_70
    ]
    length(vmi_apod_y_per_E_s2) == length(energies) ||
        error("vmi_apod_y_per_E_s2 must have one entry per VMI energy ($(length(energies)))")

    raw_vmi = Dict{Float64, Array{Float32, 3}}()
    for (k, E) in enumerate(energies)
        μ_w_E  = BS.compute_μ_at_energy(XA.Materials.water, E)
        μρ_w_E = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water,  E))
        μρ_I_E = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E))

        vmi_sino = @. μρ_w_E * sino_w + μρ_I_E * sino_I
        sino_gpu = MtlArray(vmi_sino)
        ws_fdk   = BS.create_fdk_recon_workspace(
            sino_gpu, geom, recon_size;
            filter = BS.CustomFilter(vmi_apod_x, vmi_apod_y_per_E_s2[k]),
        )
        recon_μ  = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))
        sino_gpu = nothing; ws_fdk = nothing

        recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_w_E))
        raw_vmi[E] = recon_hu

        roi = recon_hu[200:300, 200:300, mid_z]
        @info "[scan 2 VMI FBP] $(Int(E)) keV (Nyquist apod=$(vmi_apod_y_per_E_s2[k][end])): σ=$(round(std(roi), digits=1)) HU, mean=$(round(mean(roi), digits=1)) HU"
    end
    GC.gc(true)

    (vmi = raw_vmi, energies = collect(energies))
end;

# ╔═╡ 08120f01-0000-4000-8000-000000000002
# VMI FBP viz — HU at all target energies (mid slice).
let
    _vmi_row_viz(vmi_fbp.vmi, "FBP")
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

    vmip_σ_lp_px      = Float64[1.5, 0.0, 1.5, 3.0]
end

# ╔═╡ 08120010-b000-4000-8000-000000000001
# 12f.4: Mono+ FBP (Grant 2014) — frequency-split polish on the raw FBP
# VMI images, per-energy σ from `vmip_σ_lp_px`.  Output `sim_scan2_vmi_fbp`
# is the FINAL "Sim FBP" VMI (= Row 2 VMI cells in Scan 2 Results) AND the
# warm start for the HIR branch below.
sim_scan2_vmi_fbp = let
    energies = Float64.(vmi_energies)
    mid_z    = sim_matrix_size[3] ÷ 2

    # Mono+ on the raw FBP VMI images — this is the FINAL "Sim FBP" VMI
    # output (= Row 2 in the Results grids).  The HIR branch downstream
    # uses THIS as its warm start.
    if !use_mono_plus
        @info "[Mono+ FBP] DISABLED — passing raw FBP VMI through unchanged"
        Dict{Float64, Array{Float32, 3}}(E => vmi_fbp.vmi[E] for E in energies)
    else
        haskey(vmi_fbp.vmi, vmip_E_noise_opt) ||
            error("Mono+ reference vmip_E_noise_opt=$(vmip_E_noise_opt) keV not in vmi_energies=$energies")

        vols_in = [vmi_fbp.vmi[E] for E in energies]
        ws_mono_s2 = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(energies))
        res = BS.apply_mono_plus!(ws_mono_s2, vols_in, energies;
            E_noise_opt = vmip_E_noise_opt,
            σ_lp_px     = Float64.(vmip_σ_lp_px),
            verbose     = true)
        out = Dict{Float64, Array{Float32, 3}}()
        for (i, E) in enumerate(energies)
            mp         = copy(res.volumes[i])   # copy: workspace owns scratch
            roi_before = vols_in[i][200:300, 200:300, mid_z]
            roi_after  = mp[200:300, 200:300, mid_z]
            out[E] = mp
            @info "[Mono+ FBP] $(Int(E)) keV (σ_lp=$(res.σ_lp_px[i]) px): σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"
        end
        out
    end
end;

# ╔═╡ 08120f03-0000-4000-8000-000000000002
# Mono+ viz — final VMI FBP output (post Mono+ polish).  This is also
# the warm start fed into the HIR branch below.
let
    tag = use_mono_plus ? "Mono+" : "pass-through"
    _vmi_row_viz(sim_scan2_vmi_fbp, "VMI FBP ($tag)")
end

# ╔═╡ 08120f02-0000-4000-8000-000000000001
md"**VMI HIR.** Huber-regularised iterative recon per energy, **warm-started from the Mono+ FBP image** for that energy (`sim_scan2_vmi_fbp[E]`).  Pipeline: `RWLS → FBP → Mono+ → HIR`.  When `use_hir = false`, this stage is an identity pass-through of Mono+ FBP."

# ╔═╡ 08120009-a000-4000-8000-000000000006
# §12e Stage 5 — HIR recon config (optional replacement for plain FBP).
# Only the vmi_hir cell reads these; FBP / Mono+ stay cached.
begin
    use_hir         = true      # Row 4 (Sim HIR) is only meaningful with HIR on.
    hir_strength    = 3
    hir_lambda      = 10.0f0
    hir_nepochs     = 2
    hir_n_subsets   = 12
    hir_huber_delta = 0.06f0
    hir_relaxation  = 0.35f0
end

# ╔═╡ 08120010-a000-4000-8000-000000000002
# 12e-3: HIR on VMI sinograms, initialized from the Mono+ FBP image (the
# `sim_scan2_vmi_fbp` volume for that energy).  Per-energy line-integrals
# built from RWLS sino_w/sino_I, same as VMI FBP — only the recon + warm
# start differ.  Pass-through = sim_scan2_vmi_fbp when use_hir = false.
sim_scan2_vmi_hir = let
    if !use_hir
        @info "[VMI HIR] DISABLED — falling back to Mono+ FBP output"
        Dict{Float64, Array{Float32, 3}}(E => sim_scan2_vmi_fbp[E] for E in vmi_energies)
    else
        geom       = sim_scan2.geom
        recon_size = sim_matrix_size
        base       = (sino_w = pwls_decomp_s2.sino_w, sino_I = pwls_decomp_s2.sino_I)
        mid_z      = recon_size[3] ÷ 2

        out = Dict{Float64, Array{Float32, 3}}()
        ws_hir = nothing

        for E in vmi_energies
            μ_w_E  = BS.compute_μ_at_energy(XA.Materials.water, E)
            μρ_w_E = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water,  E))
            μρ_I_E = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E))

            # Per-energy VMI sinogram (line integrals) from RWLS output.
            vmi_sino = @. μρ_w_E * base.sino_w + μρ_I_E * base.sino_I
            sino_gpu = MtlArray(vmi_sino)

            # Reusable HIR workspace (sinogram shape constant across energies).
            if ws_hir === nothing
                ws_hir = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size;
                    strength = hir_strength, filter = sim_vmi_filter)
                ws_hir.params = BS.HIRParams(hir_strength, hir_lambda, 30, hir_nepochs,
                                              hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35))
            end

            # Warm start = Mono+ FBP image for this energy (HU → μ).
            init_hu  = sim_scan2_vmi_fbp[E]
            init_μ   = Float32.(Float64(μ_w_E) .* (Float64.(init_hu) ./ 1000.0 .+ 1.0))
            init_gpu = MtlArray(init_μ)

            BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; init_volume = init_gpu)
            recon_μ  = Array(ws_hir.volume)
            recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_w_E))
            out[E]   = recon_hu

            roi_in  = init_hu[200:300, 200:300, mid_z]
            roi_out = recon_hu[200:300, 200:300, mid_z]
            @info "[VMI HIR  init=Mono+ FBP] $(Int(E)) keV: σ $(round(std(roi_in), digits=1)) → $(round(std(roi_out), digits=1)) HU"

            sino_gpu = nothing; init_gpu = nothing
        end
        ws_hir = nothing; GC.gc(true)
        out
    end
end;

# ╔═╡ 08120f02-0000-4000-8000-000000000002
# HIR viz — final VMI HIR (init = Mono+ FBP) output.
let
    tag = use_hir ? "HIR (init = Mono+ FBP)" : "Mono+ FBP pass-through"
    _vmi_row_viz(sim_scan2_vmi_hir, tag)
end

# ╔═╡ 08120010-c000-4000-8000-000000000001
# Alias NamedTuple so downstream scan2_grid / measurements / viz can read
# .fbp, .hir, .mono_on — matches the previous sim_scan2_vmi API.
sim_scan2_vmi = (fbp     = sim_scan2_vmi_fbp,
                 hir     = sim_scan2_vmi_hir,
                 mono_on = use_mono_plus);

# ╔═╡ 08125000-0000-4000-8000-000000000001
md"""
### Results

Four cross-recon comparison grids at **Scan 2** (140 kVp / 10 mGy).
Rows: **Clin FBP · Sim FBP · Clin QIR3 · Sim HIR**.  Cols: **Poly · VMI 40 ·
VMI 70 · VMI 100 · VMI 140 keV**.  Mono+ is part of the VMI pipeline, so
**both** simulated VMI rows include Mono+ polish when `use_mono_plus = true`;
the only thing differentiating Row 2 from Row 4 in the VMI columns is
whether HIR was used instead of FBP.  Clinical VMIs loaded from both
`VMI/0/...` (FBP → row 1) and `VMI/3/...` (QIR3 → row 3).
"""

# ╔═╡ 08125010-0000-4000-8000-000000000001
md"**Segmentation.** Detected Gammex rods on clinical QIR3 slice vs simulated poly-FBP slice."

# ╔═╡ 08125010-0000-4000-8000-000000000002
# Segment the simulated Scan 2 reconstruction (poly FBP reference) and cache
# the mask + rod centers for the measurements below.
sim_seg_result = let
    ref = sim_scan2_poly_fbp
    mid_z = size(ref, 3) ÷ 2
    mask, rods, center = segment_gammex_rods(ref[:, :, mid_z]; fov_cm = sim_fov_cm, clockwise = false)
    (mask = mask, rods = rods, center = center)
end;

# ╔═╡ 08125010-0000-4000-8000-000000000003
# Segmentation overlay: clinical QIR3 slice | simulated poly-FBP slice,
# each with rod-ROI circles + rod names.
let
    fig = CM.Figure(size = (1200, 560), fontsize = 11)

    clin_slice = hu_140_mid_qir[:, :, seg_result.slice_idx]
    ax1 = CM.Axis(fig[1, 1]; title = "Clinical QIR3", subtitle = "140 kVp / 174 mA",
        aspect = CM.DataAspect(), yreversed = true)
    CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = (-200, 500))
    for r in seg_result.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (35.0 / size(clin_slice, 1))
        CM.lines!(ax1, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th), color = :red, linewidth = 1.2)
        CM.text!(ax1, r.cx, r.cy - rpx - 2, text = r.name, fontsize = 7, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax1); CM.hidespines!(ax1)

    sim_mid_z = sim_matrix_size[3] ÷ 2
    sim_slice = sim_scan2_poly_fbp[:, :, sim_mid_z]
    ax2 = CM.Axis(fig[1, 2]; title = "Simulated Poly FBP", subtitle = "140 kVp / 174 mA",
        aspect = CM.DataAspect())
    CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = (-200, 500))
    for r in sim_seg_result.rods
        th = range(0, 2π, length = 60)
        rpx = 1.4 * 0.6 / (sim_fov_cm / size(sim_slice, 1))
        CM.lines!(ax2, r.cx .+ rpx .* cos.(th), r.cy .+ rpx .* sin.(th), color = :red, linewidth = 1.2)
        CM.text!(ax2, r.cx, r.cy - rpx - 2, text = r.name, fontsize = 7, align = (:center, :bottom), color = :yellow)
    end
    CM.hidedecorations!(ax2); CM.hidespines!(ax2)

    CM.save(joinpath(RESULTS_DIR, "alpha_segmentation_overlay.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08125020-0000-4000-8000-000000000001
# All Scan 2 measurements in one place — poly FBP + poly HIR + per-energy
# VMI (FBP and HIR).  Measurements use sim_seg_result and the notebook's
# `measure_scan` helper (HU rod means/stds + NPS + MTF).
sim_measurements_scan2 = let
    seg = sim_seg_result
    m(vol, name) = measure_scan(vol, seg.mask, seg.rods, seg.center, name; fov_cm = sim_fov_cm)
    # VMI measurements operate on the FINAL VMI volumes (FBP- or HIR-reconstructed,
    # both with optional Mono+).  Mono+ is part of the VMI pipeline itself.
    (
        poly_fbp = m(sim_scan2_poly_fbp, "scan2_poly_fbp"),
        poly_hir = m(sim_scan2_poly_hir, "scan2_poly_hir"),
        vmi_fbp  = Dict(E => m(sim_scan2_vmi.fbp[E], "scan2_vmi_fbp_$(Int(E))keV") for E in vmi_energies),
        vmi_hir  = Dict(E => m(sim_scan2_vmi.hir[E], "scan2_vmi_hir_$(Int(E))keV") for E in vmi_energies),
    )
end;

# ╔═╡ 08125020-0000-4000-8000-000000000002
# Shared lookup — 4×5 grids of volumes + measurements indexed by
# (row, col) where rows = [Clin FBP, Sim FBP, Clin QIR3, Sim HIR] and
# cols = [Poly, VMI 40, VMI 70, VMI 100, VMI 140].  `nothing` = "not acquired".
scan2_grid = let
    row_labels = ["Clin FBP", "Sim FBP", "Clin QIR3", "Sim HIR"]
    col_labels = ["Poly", "VMI 40 keV", "VMI 70 keV", "VMI 100 keV", "VMI 140 keV"]
    vmi_E      = vmi_energies  # [40.0, 70.0, 100.0, 140.0]

    # Clinical VMI lookups by name — QIR3 (`VMI/3/…`) and FBP (`VMI/0/…`).
    clin_m_vmi_qir3(E) = first(filter(m -> m.name == "140kVp_174mA_VMI$(Int(E))_QIR3", clinical_measurements))
    clin_m_vmi_qir0(E) = first(filter(m -> m.name == "140kVp_174mA_VMI$(Int(E))_QIR0", clinical_measurements))

    vols = Matrix{Any}(undef, 4, 5)
    meas = Matrix{Any}(undef, 4, 5)

    # Row 1: Clinical FBP (poly + VMI FBP at every energy).
    vols[1, 1] = hu_140_mid_fbp;        meas[1, 1] = clinical_measurements[2]
    for (c, E) in enumerate(vmi_E)
        vols[1, c + 1] = E == 40.0  ? hu_140_mid_vmi40_qir0  :
                         E == 70.0  ? hu_140_mid_vmi70_qir0  :
                         E == 100.0 ? hu_140_mid_vmi100_qir0 : hu_140_mid_vmi140_qir0
        meas[1, c + 1] = clin_m_vmi_qir0(E)
    end

    # Row 2: Simulated FBP (Poly FBP + VMI-FBP with Mono+).
    vols[2, 1] = sim_scan2_poly_fbp;    meas[2, 1] = sim_measurements_scan2.poly_fbp
    for (c, E) in enumerate(vmi_E)
        vols[2, c + 1] = sim_scan2_vmi.fbp[E]
        meas[2, c + 1] = sim_measurements_scan2.vmi_fbp[E]
    end

    # Row 3: Clinical QIR3 (poly + VMIs, all QIR3-backed).
    vols[3, 1] = hu_140_mid_qir;        meas[3, 1] = clinical_measurements[6]
    for (c, E) in enumerate(vmi_E)
        vols[3, c + 1] = E == 40.0  ? hu_140_mid_vmi40_qir3  :
                         E == 70.0  ? hu_140_mid_vmi70_qir3  :
                         E == 100.0 ? hu_140_mid_vmi100_qir3 : hu_140_mid_vmi140_qir3
        meas[3, c + 1] = clin_m_vmi_qir3(E)
    end

    # Row 4: Simulated HIR (Poly HIR + VMI-HIR with Mono+).
    vols[4, 1] = sim_scan2_poly_hir;    meas[4, 1] = sim_measurements_scan2.poly_hir
    for (c, E) in enumerate(vmi_E)
        vols[4, c + 1] = sim_scan2_vmi.hir[E]
        meas[4, c + 1] = sim_measurements_scan2.vmi_hir[E]
    end

    # Helpers (closures over seg_result for clinical slice indexing).
    is_clinical = r -> r == 1 || r == 3
    slice_of = (row, vol) -> vol === nothing ? nothing :
        (is_clinical(row) ? vol[:, :, seg_result.slice_idx] : vol[:, :, size(vol, 3) ÷ 2])

    (rows = row_labels, cols = col_labels, vols = vols, meas = meas,
     is_clinical = is_clinical, slice_of = slice_of)
end;

# ╔═╡ 08125030-0000-4000-8000-000000000001
md"**Qualitative.** Soft-tissue window (−200, 500) HU across all cells."

# ╔═╡ 08125030-0000-4000-8000-000000000002
# Qualitative 4×5 montage.
let
    g = scan2_grid
    fig = CM.Figure(size = (1400, 1180), fontsize = 11)
    for r in 1:4, c in 1:5
        ax = CM.Axis(fig[r, c];
            title    = r == 1 ? g.cols[c] : "",
            ylabel   = c == 1 ? g.rows[r] : "",
            aspect   = CM.DataAspect(),
            yreversed = g.is_clinical(r))
        slice = g.slice_of(r, g.vols[r, c])
        if slice === nothing
            # "not acquired" placeholder: flat grey + label
            ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
            CM.text!(ax, 0.5, 0.5; text = "not acquired", align = (:center, :center),
                     fontsize = 11, color = :gray35)
            CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
        else
            CM.heatmap!(ax, slice; colormap = :grays, colorrange = (-200, 500))
        end
        CM.hidedecorations!(ax; label = false)
        CM.hidespines!(ax)
    end
    CM.rowgap!(fig.layout, 4);  CM.colgap!(fig.layout, 4)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan2_qualitative_4x5.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08125040-0000-4000-8000-000000000001
md"**HU accuracy — simulated vs clinical.** 2×2 scatter: rows = rod category (Calcium, Iodine); cols = recon pairing (FBP ↔ FBP, Iterative ↔ Iterative).  One dot per rod, colored by energy.  Dashed = unity (y = x); solid = pooled linear fit with slope / intercept / r / nRMSE in the corner."

# ╔═╡ 08125040-0000-4000-8000-000000000002
# HU accuracy — Sim vs Clin scatter.  2×2:
#   panel row = rod category (Calcium rods, Iodine rods)
#   panel col = recon pairing (FBP–FBP, Iterative–Iterative)
# Points colored by energy (Poly + 4 VMIs).  Unity y=x dashed, pooled linear
# fit solid with slope/intercept/r/nRMSE in the corner.
let
    g = scan2_grid

    # Rod indices from the `rod_order` in measure_scan:
    # Water(O)=1, SW ref 1=2, SW ref 2=3, Ca 50=4..Ca 400=8, Water(I)=9,
    # I 2.0=10..I 20.0=16.
    ca_idx = 4:8          # 5 calcium rods
    i_idx  = 10:16        # 7 iodine rods

    # Energy → color (match the notebook's usual palette).
    energy_colors = [:steelblue, :purple, :seagreen, :darkorange, :crimson]

    # Recon pairings: (panel col label, clinical row, simulated row)
    pairings = [
        ("FBP ↔ FBP",           1, 2),
        ("Iterative (QIR3 ↔ HIR)", 3, 4),
    ]
    cats = [("Calcium Rods", ca_idx), ("Iodine Rods", i_idx)]

    fig = CM.Figure(size = (1300, 1100), fontsize = 12)

    for (pr, (cat_name, cat_idx)) in enumerate(cats)
        for (pc, (pair_name, clin_row, sim_row)) in enumerate(pairings)
            ax = CM.Axis(fig[pr, pc];
                title    = "$cat_name — $pair_name",
                xlabel   = "Clinical HU",
                ylabel   = "Simulated HU",
                aspect   = 1.0)

            xs_all = Float64[]; ys_all = Float64[]
            for c in 1:length(g.cols)
                clin_m = g.meas[clin_row, c]
                sim_m  = g.meas[sim_row,  c]
                (clin_m === nothing || sim_m === nothing) && continue
                xs = Float64.(clin_m.rod_means[cat_idx])
                ys = Float64.(sim_m.rod_means[cat_idx])
                append!(xs_all, xs); append!(ys_all, ys)
                CM.scatter!(ax, xs, ys;
                    color = energy_colors[c],
                    markersize = 10,
                    strokecolor = :black, strokewidth = 0.3,
                    label = g.cols[c])
            end

            # Axis limits with a modest pad.
            lo = min(minimum(xs_all), minimum(ys_all))
            hi = max(maximum(xs_all), maximum(ys_all))
            pad = 0.05 * (hi - lo + eps())
            xlo, xhi = lo - pad, hi + pad

            # Unity y = x.
            CM.lines!(ax, [xlo, xhi], [xlo, xhi];
                color = (:gray60, 0.7), linestyle = :dash, linewidth = 1.0,
                label = "Unity (y = x)")

            # Pooled linear fit.
            n  = length(xs_all)
            Σx = sum(xs_all);  Σy = sum(ys_all)
            Σxx = sum(xs_all .* xs_all);  Σxy = sum(xs_all .* ys_all)
            slope     = (n * Σxy - Σx * Σy) / (n * Σxx - Σx^2)
            intercept = (Σy - slope * Σx) / n
            r_val     = cor(xs_all, ys_all)
            y_hat     = slope .* xs_all .+ intercept
            y_range   = maximum(ys_all) - minimum(ys_all)
            nrmse     = sqrt(mean((ys_all .- y_hat) .^ 2)) / max(y_range, eps()) * 100
            CM.lines!(ax, [xlo, xhi], [slope * xlo + intercept, slope * xhi + intercept];
                color = :black, linewidth = 1.6, label = "Linear fit")

            CM.xlims!(ax, xlo, xhi);  CM.ylims!(ax, xlo, xhi)

            # Stats annotation in the bottom-right corner.
            sgn = intercept ≥ 0 ? "+" : "−"
            stat_txt =
                "y = $(round(slope, digits = 3))·x $sgn $(round(abs(intercept), digits = 1))\n" *
                "r = $(round(r_val, digits = 4))\n" *
                "nRMSE = $(round(nrmse, digits = 1))%"
            CM.text!(ax, xhi - 0.02 * (xhi - xlo), xlo + 0.04 * (xhi - xlo);
                text = stat_txt, align = (:right, :bottom),
                fontsize = 10, color = :black)

            # Legend on the top-left panel only (shared across panels).
            if pr == 1 && pc == 1
                CM.axislegend(ax; position = :lt, labelsize = 9, framevisible = true)
            end
        end
    end

    CM.rowgap!(fig.layout, 18);  CM.colgap!(fig.layout, 18)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan2_hu_accuracy.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08125070-0000-4000-8000-000000000001
md"**Water noise (σ) summary.** 5 energy-groups × 4 bars (Clin FBP · Sim FBP · Clin QIR3 · Sim HIR).  Solid fill = FBP, outline = iterative.  Each bar is the std-dev of HU values inside the outer water ROI."

# ╔═╡ 08125070-0000-4000-8000-000000000002
# Water-noise bar chart — Scan 2.  Groups = {Poly, VMI 40/70/100/140};
# 4 bars per group matching the Qualitative / HU / MTF / NPS grid rows.
let
    g = scan2_grid
    water_idx = 1   # Water (O) is rod index 1 in measure_scan's rod_order.
    n_cols = length(g.cols)
    xs = collect(1:n_cols)
    bw = 0.18
    offsets = (-1.5, -0.5, 0.5, 1.5) .* bw

    # Extract σ_water per (row, col).
    σ = Matrix{Float64}(undef, 4, n_cols)
    for r in 1:4, c in 1:n_cols
        m = g.meas[r, c]
        σ[r, c] = m === nothing ? NaN : Float64(m.rod_stds[water_idx])
    end

    row_style = [                                 # (label, color, fill, stroke)
        ("Clinical FBP",  :steelblue,  :solid,   :steelblue),
        ("Simulated FBP", :darkorange, :solid,   :darkorange),
        ("Clinical QIR3", :steelblue,  :outline, :steelblue),
        ("Simulated HIR", :darkorange, :outline, :darkorange),
    ]

    fig = CM.Figure(size = (1150, 500), fontsize = 12)
    ax  = CM.Axis(fig[1, 1];
        title  = "Scan 2 (140 kVp / 174 mA / 10 mGy) — Water ROI noise",
        xlabel = "Reconstruction / Energy",
        ylabel = "Water σ (HU)",
        xticks = (xs, g.cols))

    for (r, (lab, base_color, style, stroke_color)) in enumerate(row_style)
        y = σ[r, :]
        x_r = xs .+ offsets[r]
        if style == :solid
            CM.barplot!(ax, x_r, y;
                width = bw, color = base_color, strokecolor = stroke_color,
                strokewidth = 0.5, label = lab)
        else  # outline (iterative)
            CM.barplot!(ax, x_r, y;
                width = bw, color = (base_color, 0.12),
                strokecolor = stroke_color, strokewidth = 1.8, label = lab)
        end
        # Value label above each bar.
        for (xi, yi) in zip(x_r, y)
            isnan(yi) && continue
            CM.text!(ax, xi, yi; text = string(round(yi, digits = 1)),
                align = (:center, :bottom), fontsize = 8, color = :gray20, offset = (0, 2))
        end
    end

    CM.ylims!(ax, 0, nothing)
    CM.axislegend(ax; position = :rt, labelsize = 10, framevisible = true)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan2_noise_bar.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08125050-0000-4000-8000-000000000001
md"**MTF — Clinical vs Simulated.** 2×5 grid: rows = recon pairing (FBP · Iterative); cols = energy (Poly + 4 VMIs).  Solid blue = Clinical; dashed orange = Simulated.  Dashed refs at MTF = 0.5 and 0.1; f50/f10 annotated per panel."

# ╔═╡ 08125050-0000-4000-8000-000000000002
# MTF — Clinical vs Simulated overlay, 2 rows × 5 cols.
#   row 1: FBP pairing (Clin FBP vs Sim FBP)
#   row 2: Iterative pairing (Clin QIR3 vs Sim HIR)
let
    g = scan2_grid
    pairings = [("FBP",       1, 2),   # (label, clin_row, sim_row)
                ("Iterative", 3, 4)]
    n_cols = length(g.cols)

    fig = CM.Figure(size = (1500, 620), fontsize = 11)

    for (pr, (pair_label, clin_row, sim_row)) in enumerate(pairings)
        for c in 1:n_cols
            ax = CM.Axis(fig[pr, c];
                title    = pr == 1 ? g.cols[c] : "",
                subtitle = pr == 1 ? "Clinical vs Simulated" : "",
                xlabel   = pr == 2 ? "Spatial frequency (lp/mm)" : "",
                ylabel   = c == 1 ? "$(pair_label)\nMTF" : "")
            clin = g.meas[clin_row, c]
            sim  = g.meas[sim_row,  c]

            has_clin = clin !== nothing && hasproperty(clin, :mtf) && clin.mtf !== nothing
            has_sim  = sim  !== nothing && hasproperty(sim,  :mtf) && sim.mtf  !== nothing

            if !has_clin && !has_sim
                ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
                CM.text!(ax, 0.5, 0.5; text = "no data", align = (:center, :center),
                         fontsize = 10, color = :gray35)
                CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
                CM.hidedecorations!(ax; label = false)
                continue
            end

            has_clin && CM.lines!(ax, clin.mtf.frequencies, clin.mtf.mtf;
                                  color = :steelblue, linewidth = 1.8, label = "Clinical")
            has_sim  && CM.lines!(ax, sim.mtf.frequencies,  sim.mtf.mtf;
                                  color = :darkorange, linewidth = 1.8,
                                  linestyle = :dash,   label = "Simulated")

            CM.hlines!(ax, [0.5, 0.1]; color = :gray70, linestyle = :dash, linewidth = 0.5)
            CM.ylims!(ax, 0, 1.1)

            # f50 / f10 annotation in the top-right.
            lines = String[]
            has_clin && push!(lines, "clin f50=$(round(clin.mtf_f50, digits = 2))")
            has_sim  && push!(lines,  "sim  f50=$(round(sim.mtf_f50,  digits = 2))")
            CM.text!(ax, 0.96 * maximum(has_clin ? clin.mtf.frequencies : sim.mtf.frequencies), 1.07;
                text = join(lines, "\n"), align = (:right, :top), fontsize = 8, color = :gray30)

            if pr == 1 && c == 1
                CM.axislegend(ax; position = :rt, labelsize = 9, framevisible = true)
            end
        end
    end
    CM.rowgap!(fig.layout, 12);  CM.colgap!(fig.layout, 10)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan2_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08125060-0000-4000-8000-000000000001
md"**NPS — Clinical vs Simulated.** 2×5 grid: same row/col layout as MTF.  Solid blue = Clinical; dashed orange = Simulated.  y-axis shared per column so magnitudes within a column are directly comparable; peak frequency + integrated NPS annotated per panel."

# ╔═╡ 08125060-0000-4000-8000-000000000002
# NPS — Clinical vs Simulated overlay, 2 rows × 5 cols.
let
    g = scan2_grid
    pairings = [("FBP",       1, 2),
                ("Iterative", 3, 4)]
    n_cols = length(g.cols)

    # Shared y-max per column (across both pairings and both curves).
    col_ymax = map(1:n_cols) do c
        ymax = 0.0
        for (_, cr, sr) in pairings, row in (cr, sr)
            m = g.meas[row, c]
            if m !== nothing && hasproperty(m, :nps) && m.nps !== nothing
                ymax = max(ymax, maximum(m.nps.nps_1d))
            end
        end
        ymax == 0.0 ? 1.0 : ymax * 1.05
    end

    fig = CM.Figure(size = (1500, 620), fontsize = 11)

    for (pr, (pair_label, clin_row, sim_row)) in enumerate(pairings)
        for c in 1:n_cols
            ax = CM.Axis(fig[pr, c];
                title    = pr == 1 ? g.cols[c] : "",
                subtitle = pr == 1 ? "Clinical vs Simulated" : "",
                xlabel   = pr == 2 ? "Spatial frequency (lp/mm)" : "",
                ylabel   = c == 1 ? "$(pair_label)\nNPS" : "")
            clin = g.meas[clin_row, c]
            sim  = g.meas[sim_row,  c]

            has_clin = clin !== nothing && hasproperty(clin, :nps) && clin.nps !== nothing
            has_sim  = sim  !== nothing && hasproperty(sim,  :nps) && sim.nps  !== nothing

            if !has_clin && !has_sim
                ax.backgroundcolor[] = CM.RGBAf(0.85, 0.85, 0.85, 0.6)
                CM.text!(ax, 0.5, 0.5; text = "no data", align = (:center, :center),
                         fontsize = 10, color = :gray35)
                CM.xlims!(ax, 0, 1);  CM.ylims!(ax, 0, 1)
                CM.hidedecorations!(ax; label = false)
                continue
            end

            has_clin && CM.lines!(ax, clin.nps.frequencies, clin.nps.nps_1d;
                                  color = :steelblue,  linewidth = 1.8, label = "Clinical")
            has_sim  && CM.lines!(ax, sim.nps.frequencies,  sim.nps.nps_1d;
                                  color = :darkorange, linewidth = 1.8,
                                  linestyle = :dash,   label = "Simulated")
            CM.ylims!(ax, 0, col_ymax[c])

            # Peak / ∫NPS annotation.
            lines = String[]
            has_clin && push!(lines, "clin peak=$(round(clin.nps_peak_freq, digits = 2))  ∫=$(round(clin.nps_area, sigdigits = 3))")
            has_sim  && push!(lines,  "sim  peak=$(round(sim.nps_peak_freq,  digits = 2))  ∫=$(round(sim.nps_area,  sigdigits = 3))")
            CM.text!(ax, 0.96 * maximum(has_clin ? clin.nps.frequencies : sim.nps.frequencies), col_ymax[c];
                text = join(lines, "\n"), align = (:right, :top), fontsize = 8, color = :gray30)

            if pr == 1 && c == 1
                CM.axislegend(ax; position = :rt, labelsize = 9, framevisible = true)
            end
        end
    end
    CM.rowgap!(fig.layout, 12);  CM.colgap!(fig.layout, 10)
    CM.save(joinpath(RESULTS_DIR, "alpha_scan2_nps.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 08140001-0000-4000-8000-000000000000
md"""
## Scan 3: 140 kVp / ~20 mGy

Placeholder — same pipeline as Scan 2 with higher mA.
"""

# ╔═╡ 08140002-0000-4000-8000-000000000000
sim_scan3 = let
    prot = BS.CTProtocol(
        kVp = 140.0,
        mA = sim_mA_scan3,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = [("Ti", 0.9)],
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
sim_scan4 = let
    prot = BS.CTProtocol(
        kVp = 120.0,
        mA = sim_mA_scan4,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = [("Ti", 0.9)],
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
# ╠═8d4be3af-9475-4378-a318-0fbe03f07663
# ╠═08090007-b000-4000-8000-000000000001
# ╠═08090008-0000-4000-8000-000000000000
# ╟─08100001-0000-4000-8000-000000000000
# ╠═08100002-0000-4000-8000-000000000000
# ╟─08100003-0000-4000-8000-000000000000
# ╠═08100004-0000-4000-8000-000000000000
# ╟─08100005-0000-4000-8000-000000000000
# ╟─08130001-0000-4000-8000-000000000000
# ╠═08130002-0000-4000-8000-000000000000
# ╟─08131002-b000-4000-8000-000000000001
# ╟─08131001-0000-4000-8000-000000000000
# ╟─08131003-0000-4000-8000-000000000000
# ╠═08131005-a000-4000-8000-000000000002
# ╠═08131002-0000-4000-8000-000000000000
# ╠═08131002-a000-4000-8000-000000000001
# ╠═08131004-0000-4000-8000-000000000000
# ╟─08131005-0000-4000-8000-000000000000
# ╠═08131006-a000-4000-8000-000000000001
# ╠═08131006-0000-4000-8000-000000000000
# ╠═08131006-b000-4000-8000-000000000001
# ╟─08131005-a000-4000-8000-000000000001
# ╟─08131009-0000-4000-8000-000000000000
# ╟─08131e02-0000-4000-8000-000000000001
# ╠═08131e02-0000-4000-8000-000000000002
# ╠═d335e1a7-cb56-44a9-ad4a-c54f3e1fc280
# ╠═08131e02-0000-4000-8000-000000000003
# ╠═08131e02-0000-4000-8000-000000000033
# ╠═08131e02-0000-4000-8000-000000000004
# ╟─08131e02-1000-4000-8000-000000000001
# ╠═08131e02-1000-4000-8000-000000000002
# ╠═08131e02-1000-4000-8000-000000000003
# ╠═08131e02-1000-4000-8000-000000000004
# ╠═08131e02-1000-4000-8000-000000000005
# ╠═08131e02-1000-4000-8000-000000000006
# ╟─08131e02-0000-4000-8000-000000000005
# ╟─08131e03-0000-4000-8000-000000000001
# ╠═08131010-0000-4000-8000-000000000041
# ╠═08131010-0000-4000-8000-000000000042
# ╠═08131010-0000-4000-8000-000000000045
# ╠═08131010-0000-4000-8000-000000000080
# ╠═08131009-a000-4000-8000-000000000011
# ╠═08131010-0000-4000-8000-000000000040
# ╟─08131e03-0000-4000-8000-000000000002
# ╟─08131f00-0000-4000-8000-000000000001
# ╠═08131009-a000-4000-8000-000000000001
# ╠═08131009-a000-4000-8000-000000000007
# ╟─08131f02-0000-4000-8000-000000000001
# ╠═08131009-a000-4000-8000-000000000006
# ╠═08131010-a000-4000-8000-000000000002
# ╟─08131f02-0000-4000-8000-000000000002
# ╟─08135000-0000-4000-8000-000000000001
# ╟─08135010-0000-4000-8000-000000000001
# ╠═08135010-0000-4000-8000-000000000002
# ╟─08135010-0000-4000-8000-000000000003
# ╠═08135020-0000-4000-8000-000000000001
# ╠═08135020-0000-4000-8000-000000000002
# ╟─08135030-0000-4000-8000-000000000001
# ╟─08135030-0000-4000-8000-000000000002
# ╟─08135040-0000-4000-8000-000000000001
# ╟─08135040-0000-4000-8000-000000000002
# ╟─08135070-0000-4000-8000-000000000001
# ╟─08135070-0000-4000-8000-000000000002
# ╟─08135050-0000-4000-8000-000000000001
# ╟─08135050-0000-4000-8000-000000000002
# ╟─08135060-0000-4000-8000-000000000001
# ╟─08135060-0000-4000-8000-000000000002
# ╟─08110001-0000-4000-8000-000000000000
# ╠═08110002-0000-4000-8000-000000000000
# ╟─08120002-b000-4000-8000-000000000001
# ╟─08120001-0000-4000-8000-000000000000
# ╟─08120003-0000-4000-8000-000000000000
# ╟─08120005-a000-4000-8000-000000000001
# ╠═08120005-a000-4000-8000-000000000002
# ╠═08120002-0000-4000-8000-000000000000
# ╠═08120002-a000-4000-8000-000000000001
# ╠═08120004-0000-4000-8000-000000000000
# ╟─08120005-0000-4000-8000-000000000000
# ╠═08120006-a000-4000-8000-000000000001
# ╠═08120006-0000-4000-8000-000000000000
# ╠═08120006-b000-4000-8000-000000000001
# ╟─08120009-0000-4000-8000-000000000000
# ╠═08120e00-0000-4000-8000-000000000001
# ╟─08120e02-0000-4000-8000-000000000001
# ╠═08120e02-0000-4000-8000-000000000002
# ╠═08120e02-0000-4000-8000-000000000003
# ╠═08120e02-0000-4000-8000-000000000004
# ╟─08120e02-0000-4000-8000-000000000005
# ╟─08120e03-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000011
# ╠═08120010-0000-4000-8000-000000000040
# ╟─08120e03-0000-4000-8000-000000000002
# ╟─08120e05-0000-4000-8000-000000000001
# ╟─08120f00-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000001
# ╟─08120f01-0000-4000-8000-000000000001
# ╠═08120010-a000-4000-8000-000000000001
# ╟─08120f01-0000-4000-8000-000000000002
# ╟─08120f03-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000007
# ╠═08120010-b000-4000-8000-000000000001
# ╟─08120f03-0000-4000-8000-000000000002
# ╟─08120f02-0000-4000-8000-000000000001
# ╠═08120009-a000-4000-8000-000000000006
# ╠═08120010-a000-4000-8000-000000000002
# ╟─08120f02-0000-4000-8000-000000000002
# ╠═08120010-c000-4000-8000-000000000001
# ╟─08125000-0000-4000-8000-000000000001
# ╟─08125010-0000-4000-8000-000000000001
# ╠═08125010-0000-4000-8000-000000000002
# ╟─08125010-0000-4000-8000-000000000003
# ╠═08125020-0000-4000-8000-000000000001
# ╠═08125020-0000-4000-8000-000000000002
# ╟─08125030-0000-4000-8000-000000000001
# ╟─08125030-0000-4000-8000-000000000002
# ╟─08125040-0000-4000-8000-000000000001
# ╟─08125040-0000-4000-8000-000000000002
# ╟─08125070-0000-4000-8000-000000000001
# ╟─08125070-0000-4000-8000-000000000002
# ╟─08125050-0000-4000-8000-000000000001
# ╟─08125050-0000-4000-8000-000000000002
# ╟─08125060-0000-4000-8000-000000000001
# ╟─08125060-0000-4000-8000-000000000002
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
# ╟─08180001-0000-4000-8000-000000000000
# ╠═08180002-0000-4000-8000-000000000000
# ╟─08190001-0000-4000-8000-000000000000
# ╟─08190002-0000-4000-8000-000000000000
