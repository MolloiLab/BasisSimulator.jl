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

# ╔═╡ 08010018-a000-4000-8000-000000000000
import AcceleratedKernels as AK

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
# Br36-ish (sharper) kernel approximation — used for VMI / bin FBP.
# Control points: (normalized_freq, amplitude) — higher amplitudes at
# mid/high frequencies = more spatial-resolution preservation, more noise.
# Previous (softer) profile preserved in comment for easy reverting.
sim_custom_filter = BS.CustomFilter(
  (0.0, 0.25, 0.5, 0.75, 1.0),
  (1.0, 0.75, 0.6,  0.2,  0.001),   # original Br36-ish (softer)
  # (1.0,  0.92, 0.82, 0.55, 0.10),     # sharper variant for testing
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

# ╔═╡ 08131006-d000-4000-8000-000000000001
# ── STAGE 1 — 4-bin → 2-bin combined sinograms (mid-view heatmaps) ──
let
    sl = sim_scan1_lohi.sino_low
    sh = sim_scan1_lohi.sino_high
    mid_v = size(sl, 2) ÷ 2

    cr_l = (quantile(vec(sl[:, mid_v, :]), 0.01), quantile(vec(sl[:, mid_v, :]), 0.995))
    cr_h = (quantile(vec(sh[:, mid_v, :]), 0.01), quantile(vec(sh[:, mid_v, :]), 0.995))

    fig = CM.Figure(size = (1300, 480), fontsize = 11)
    CM.Label(fig[0, 1:2], "STAGE 1 — 4-bin → 2-bin combined sinograms (mid-view) — bins 1+2 (low) / 3+4 (high)";
        fontsize = 14, halign = :left)
    ax_l = CM.Axis(fig[1, 1]; title = "sino_low (bins 1+2)",  aspect = CM.DataAspect())
    CM.heatmap!(ax_l, sl[:, mid_v, :]; colormap = :viridis, colorrange = cr_l)
    CM.hidedecorations!(ax_l); CM.hidespines!(ax_l)
    ax_h = CM.Axis(fig[1, 2]; title = "sino_high (bins 3+4)", aspect = CM.DataAspect())
    CM.heatmap!(ax_h, sh[:, mid_v, :]; colormap = :viridis, colorrange = cr_h)
    CM.hidedecorations!(ax_h); CM.hidespines!(ax_h)
    fig
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
**Material decomposition + VMI synthesis (Scan 1).** Pipeline: **4-bin → 2-bin combine → FBP → RSKR-2ch → image-domain decomp via M_cal⁻¹ → VMI**.

The 4 PCCT bins are combined via Beer's law into low/high effective sinos (bins 1+2 and 3+4).  Each is FBP-reconstructed independently, then jointly denoised by RSKR-2ch (SVD + bilateral filter on U-vectors).  A 2×2 path-effective calibrated `M_cal` matrix maps the denoised `(vol_low, vol_high)` pair to material-density volumes `(ρ_water, ρ_iodine)`.  Per-energy VMI synthesis evaluates `μ_E = ρ_w·μρ_w(E) + ρ_I·μρ_I(E)` in image domain.
"""

# ╔═╡ 08131d00-0000-4000-8000-000000000001
md"""
### RSKR pipeline (Clark & Badea 2023, MCR Toolkit) — Scan 1

Replaces sino-domain CMV/ACNR with the Duke MCR Toolkit's image-domain
post-reconstruction architecture:

1. **Per-bin FBP** — reconstruct each of the 4 PCCT bin sinograms separately
   into 4 bin volumes (`vol_b[v] ≈ ⟨μ⟩_b` at voxel v).
2. **RSKR (Rank-Sparse Kernel Regression)** — joint denoise the 4 bin volumes
   via SVD-decomposed channel space + joint bilateral filter on left singular
   vectors with **adaptive σ from MAD-Haar** (no manual tuning).
3. **Image-domain material decomposition** — per-voxel LSQ:
   `vol_b[v] = Σ_m M[b, m] · ρ_m[v]`  →  `ρ[v] = M_pinv · vol[v]`
   where `M[b, m] = Σ_E ŵ_b(E) · μρ_m(E)` is the bin-averaged mass attenuation.
4. **VMI resynthesis** (image-domain): `μ(E)[v] = Σ_m ρ_m[v] · μρ_m(E)`,
   then convert to HU.

This is the architecturally correct Duke pipeline (Clark/Badea 2023).
Inlined here to verify-and-iterate freely (no `src/` edits).
Reference: https://gitlab.oit.duke.edu/dpc18/mcr-toolkit-public
"""

# ╔═╡ 0cbcb5fe-7e29-48fc-9d68-ab7cb05ac3c4
# HU window for all polychromatic-bin viz cells (raw FBP / RSKR / combined).
# All FBP-stage volumes are HU-converted per-bin before display so a single
# window applies uniformly across plots.
rskr_viz_hu_clim_s1 = (-200.0, 500.0)

# ╔═╡ 08131d00-0000-4000-8000-000000000003
# Helper functions for RSKR pipeline (all inlined in this section).
#   _mad_haar_σ        — adaptive σ estimator from Haar wavelet high-pass MAD
#   _joint_bf_4ch_gpu! — joint bilateral filter on 4 channels (Metal GPU)
#   _rskr_4ch          — RSKR driver: SVD → per-SV jBF → reconstitute, iterated
begin
    # ── MAD-Haar adaptive σ (paper Eq 7) ──
    # σ_e = 1.4826 · median(|∇χ_e(n)|) where ∇ = Haar high-pass (cross-diff approx).
    # Computed on mid-z axial slice for efficiency (paper §2.2).
    function _mad_haar_σ(vol::AbstractArray{Float32, 3})
        n_col, n_row, n_z = size(vol)
        mid_z = max(1, n_z ÷ 2 + 1)
        slice = @view vol[:, :, mid_z]
        # 2D Haar HH-component: ∇²(i,j) = x(i+1,j+1) - x(i+1,j) - x(i,j+1) + x(i,j)
        n_pairs = (n_col - 1) * (n_row - 1)
        absvec  = Vector{Float32}(undef, n_pairs)
        idx = 0
        @inbounds for j in 1:n_row-1, i in 1:n_col-1
            v = slice[i+1, j+1] - slice[i+1, j] - slice[i, j+1] + slice[i, j]
            idx += 1
            absvec[idx] = abs(v)
        end
        Float32(1.4826) * Float32(median(absvec))
    end

    # ── Joint bilateral filter, hardcoded for N=4 channels (PCCT bins) ──
    # Operates on Metal GPU.  Spherical spatial mask of radius `r`, joint range
    # kernel = product over channels (paper Eq 6).  No spatial Gaussian — paper
    # uses uniform spherical D (kernel diameter 13 = radius 6).
    function _joint_bf_4ch_gpu!(
            out1::AbstractArray{Float32, 3},
            out2::AbstractArray{Float32, 3},
            out3::AbstractArray{Float32, 3},
            out4::AbstractArray{Float32, 3},
            in1::AbstractArray{Float32, 3},
            in2::AbstractArray{Float32, 3},
            in3::AbstractArray{Float32, 3},
            in4::AbstractArray{Float32, 3},
            σ1::Float32, σ2::Float32, σ3::Float32, σ4::Float32,
            h::Float32, radius::Int,
        )
        n_col, n_row, n_z = size(in1)
        # Precompute inverse 2(h·σ)² per channel (Float32, captured by closure).
        inv_2hσ²_1 = 1f0 / (2f0 * (h * σ1) * (h * σ1) + 1f-30)
        inv_2hσ²_2 = 1f0 / (2f0 * (h * σ2) * (h * σ2) + 1f-30)
        inv_2hσ²_3 = 1f0 / (2f0 * (h * σ3) * (h * σ3) + 1f-30)
        inv_2hσ²_4 = 1f0 / (2f0 * (h * σ4) * (h * σ4) + 1f-30)
        radius² = Int32(radius * radius)
        r_i32   = Int32(radius)

        AK.foreachindex(in1) do idx
            i0 = idx - 1
            ci = (i0 % n_col) + 1
            ri = ((i0 ÷ n_col) % n_row) + 1
            zi = ((i0 ÷ (n_col * n_row)) % n_z) + 1

            v1 = in1[ci, ri, zi]
            v2 = in2[ci, ri, zi]
            v3 = in3[ci, ri, zi]
            v4 = in4[ci, ri, zi]

            sum_1 = 0f0; sum_2 = 0f0; sum_3 = 0f0; sum_4 = 0f0
            wtot  = 0f0

            for dz in -r_i32:r_i32
                zi2 = zi + dz
                (zi2 < 1 || zi2 > n_z) && continue
                dz² = Int32(dz) * Int32(dz)
                for dr in -r_i32:r_i32
                    ri2 = ri + dr
                    (ri2 < 1 || ri2 > n_row) && continue
                    dr² = Int32(dr) * Int32(dr)
                    drz² = dz² + dr²
                    drz² > radius² && continue
                    for dc in -r_i32:r_i32
                        ci2 = ci + dc
                        (ci2 < 1 || ci2 > n_col) && continue
                        dist² = drz² + Int32(dc) * Int32(dc)
                        dist² > radius² && continue

                        nv1 = in1[ci2, ri2, zi2]
                        nv2 = in2[ci2, ri2, zi2]
                        nv3 = in3[ci2, ri2, zi2]
                        nv4 = in4[ci2, ri2, zi2]

                        d1 = nv1 - v1
                        d2 = nv2 - v2
                        d3 = nv3 - v3
                        d4 = nv4 - v4

                        log_w = -d1 * d1 * inv_2hσ²_1 -
                                 d2 * d2 * inv_2hσ²_2 -
                                 d3 * d3 * inv_2hσ²_3 -
                                 d4 * d4 * inv_2hσ²_4
                        w = exp(log_w)

                        sum_1 += w * nv1
                        sum_2 += w * nv2
                        sum_3 += w * nv3
                        sum_4 += w * nv4
                        wtot  += w
                    end
                end
            end

            inv_w = 1f0 / max(wtot, 1f-30)
            out1[ci, ri, zi] = sum_1 * inv_w
            out2[ci, ri, zi] = sum_2 * inv_w
            out3[ci, ri, zi] = sum_3 * inv_w
            out4[ci, ri, zi] = sum_4 * inv_w
        end
        nothing
    end

    # ── RSKR driver for 4 channels (paper §2.2 + Fig 1b, simplified for static data) ──
    # 1. SVD of (n_voxels × 4) matrix → U (n_voxels × 4), Σ (4×4), V₀ (4×4)
    # 2. Compute per-SV adaptive σ from MAD-Haar of each U column reshaped to volume
    # 3. Apply joint bilateral filter to U columns (rank-sparse h scaling: smaller
    #    SVs get larger h → more smoothing, paper Fig 1d step 4.3)
    # 4. Reconstitute V_denoised = U' · Σ · V₀'
    # 5. Iterate `n_iter` times (re-SVD each round to re-balance noise across channels)
    function _rskr_4ch(
            bin_vols::Vector{Array{Float32, 3}};
            n_iter::Int    = 4,
            h_param::Real  = 1.0,
            radius::Int    = 6,
            γ::Real        = 0.5,
            verbose::Bool  = true,
        )
        length(bin_vols) == 4 || error("_rskr_4ch: requires exactly 4 bin volumes (PCCT)")
        sz = size(bin_vols[1])
        n_vox = prod(sz)
        # Stack (n_voxels × 4) for SVD
        V_mat = Matrix{Float32}(undef, n_vox, 4)
        for b in 1:4
            V_mat[:, b] .= vec(bin_vols[b])
        end
        h_f = Float32(h_param)
        γ_f = Float32(γ)

        for iter in 1:n_iter
            t0 = time()
            # ── SVD ──
            F = svd(V_mat; full = false)
            U = F.U                  # (n_vox × 4)
            Σ = F.S                  # 4-vec
            V₀ = F.V                 # (4 × 4)

            # ── Adaptive σ + rank-sparse h per SV ──
            # h_e = h_0 · (Σ_1/Σ_e)^γ → smaller SVs get more smoothing.
            σ_per_sv = Vector{Float32}(undef, 4)
            h_per_sv = Vector{Float32}(undef, 4)
            U_vols   = [reshape(U[:, e], sz) for e in 1:4]
            for e in 1:4
                σ_per_sv[e] = _mad_haar_σ(U_vols[e])
                h_per_sv[e] = h_f * (Float32(Σ[1]) / max(Float32(Σ[e]), 1f-12)) ^ γ_f
            end

            # ── Joint bilateral filter on U columns ──
            in1 = MtlArray(U_vols[1])
            in2 = MtlArray(U_vols[2])
            in3 = MtlArray(U_vols[3])
            in4 = MtlArray(U_vols[4])
            out1 = similar(in1); out2 = similar(in2); out3 = similar(in3); out4 = similar(in4)
            # h common across SVs is captured globally; per-SV σ varies.  We pass the
            # max h across SVs and let σ-scaling adjust per channel — equivalent to per-channel h.
            # Effective range scale = h_per_sv[e] · σ_per_sv[e].
            σ_eff_1 = h_per_sv[1] * σ_per_sv[1]
            σ_eff_2 = h_per_sv[2] * σ_per_sv[2]
            σ_eff_3 = h_per_sv[3] * σ_per_sv[3]
            σ_eff_4 = h_per_sv[4] * σ_per_sv[4]
            _joint_bf_4ch_gpu!(out1, out2, out3, out4,
                               in1,  in2,  in3,  in4,
                               σ_eff_1, σ_eff_2, σ_eff_3, σ_eff_4,
                               1f0, radius)
            U_denoised = hcat(vec(Array(out1)), vec(Array(out2)),
                              vec(Array(out3)), vec(Array(out4)))
            in1 = nothing; in2 = nothing; in3 = nothing; in4 = nothing
            out1 = nothing; out2 = nothing; out3 = nothing; out4 = nothing
            GC.gc(true)

            # ── Reconstitute V = U_denoised · diag(Σ) · V₀' ──
            V_mat = U_denoised * Diagonal(Σ) * V₀'

            dt = time() - t0
            verbose && @info "[RSKR iter $(iter)/$(n_iter)] $(round(dt, digits=2))s   σ_per_SV = $(round.(σ_per_sv, sigdigits=3))   h_per_SV = $(round.(h_per_sv, sigdigits=3))   Σ = $(round.(Σ, sigdigits=3))"
        end

        # Unstack
        out_vols = [reshape(V_mat[:, b], sz) for b in 1:4]
        out_vols
    end

    # ── Joint bilateral filter, 2-channel variant for IBHC + RSKR pipeline ──
    function _joint_bf_2ch_gpu!(
            out1::AbstractArray{Float32, 3},
            out2::AbstractArray{Float32, 3},
            in1::AbstractArray{Float32, 3},
            in2::AbstractArray{Float32, 3},
            σ1::Float32, σ2::Float32,
            h::Float32, radius::Int,
        )
        n_col, n_row, n_z = size(in1)
        inv_2hσ²_1 = 1f0 / (2f0 * (h * σ1) * (h * σ1) + 1f-30)
        inv_2hσ²_2 = 1f0 / (2f0 * (h * σ2) * (h * σ2) + 1f-30)
        radius² = Int32(radius * radius)
        r_i32   = Int32(radius)

        AK.foreachindex(in1) do idx
            i0 = idx - 1
            ci = (i0 % n_col) + 1
            ri = ((i0 ÷ n_col) % n_row) + 1
            zi = ((i0 ÷ (n_col * n_row)) % n_z) + 1

            v1 = in1[ci, ri, zi]
            v2 = in2[ci, ri, zi]

            sum_1 = 0f0; sum_2 = 0f0
            wtot  = 0f0

            for dz in -r_i32:r_i32
                zi2 = zi + dz
                (zi2 < 1 || zi2 > n_z) && continue
                dz² = Int32(dz) * Int32(dz)
                for dr in -r_i32:r_i32
                    ri2 = ri + dr
                    (ri2 < 1 || ri2 > n_row) && continue
                    dr² = Int32(dr) * Int32(dr)
                    drz² = dz² + dr²
                    drz² > radius² && continue
                    for dc in -r_i32:r_i32
                        ci2 = ci + dc
                        (ci2 < 1 || ci2 > n_col) && continue
                        dist² = drz² + Int32(dc) * Int32(dc)
                        dist² > radius² && continue

                        nv1 = in1[ci2, ri2, zi2]
                        nv2 = in2[ci2, ri2, zi2]

                        d1 = nv1 - v1
                        d2 = nv2 - v2

                        log_w = -d1 * d1 * inv_2hσ²_1 -
                                 d2 * d2 * inv_2hσ²_2
                        w = exp(log_w)

                        sum_1 += w * nv1
                        sum_2 += w * nv2
                        wtot  += w
                    end
                end
            end

            inv_w = 1f0 / max(wtot, 1f-30)
            out1[ci, ri, zi] = sum_1 * inv_w
            out2[ci, ri, zi] = sum_2 * inv_w
        end
        nothing
    end

    # ── RSKR driver for 2 channels (low/high combined PCCT bins) ──
    function _rskr_2ch(
            bin_vols::Vector{Array{Float32, 3}};
            n_iter::Int    = 4,
            h_param::Real  = 1.0,
            radius::Int    = 6,
            γ::Real        = 0.5,
            verbose::Bool  = true,
        )
        length(bin_vols) == 2 || error("_rskr_2ch: requires exactly 2 bin volumes (low, high)")
        sz = size(bin_vols[1])
        n_vox = prod(sz)
        V_mat = Matrix{Float32}(undef, n_vox, 2)
        for b in 1:2
            V_mat[:, b] .= vec(bin_vols[b])
        end
        h_f = Float32(h_param)
        γ_f = Float32(γ)

        for iter in 1:n_iter
            t0 = time()
            F = svd(V_mat; full = false)
            U = F.U; Σ = F.S; V₀ = F.V

            σ_per_sv = Vector{Float32}(undef, 2)
            h_per_sv = Vector{Float32}(undef, 2)
            U_vols   = [reshape(U[:, e], sz) for e in 1:2]
            for e in 1:2
                σ_per_sv[e] = _mad_haar_σ(U_vols[e])
                h_per_sv[e] = h_f * (Float32(Σ[1]) / max(Float32(Σ[e]), 1f-12)) ^ γ_f
            end

            in1 = MtlArray(U_vols[1])
            in2 = MtlArray(U_vols[2])
            out1 = similar(in1); out2 = similar(in2)
            σ_eff_1 = h_per_sv[1] * σ_per_sv[1]
            σ_eff_2 = h_per_sv[2] * σ_per_sv[2]
            _joint_bf_2ch_gpu!(out1, out2, in1, in2,
                               σ_eff_1, σ_eff_2,
                               1f0, radius)
            U_denoised = hcat(vec(Array(out1)), vec(Array(out2)))
            in1 = nothing; in2 = nothing
            out1 = nothing; out2 = nothing
            GC.gc(true)

            V_mat = U_denoised * Diagonal(Σ) * V₀'

            dt = time() - t0
            verbose && @info "[RSKR-2ch iter $(iter)/$(n_iter)] $(round(dt, digits=2))s   σ_per_SV = $(round.(σ_per_sv, sigdigits=3))   h_per_SV = $(round.(h_per_sv, sigdigits=3))   Σ = $(round.(Σ, sigdigits=3))"
        end

        out_vols = [reshape(V_mat[:, b], sz) for b in 1:2]
        out_vols
    end

    # ── Image-domain ring correction (port of MCR Toolkit `suppress_rings.m`) ──
    # The MCR algorithm operates on sinograms:
    #   Ymv = mean along view-axis at each (det_row, det_col)
    #   Ymr = 2D median filter on each (det_row, det_col) slice
    #   diff = Ymv - Ymr  (persistent-in-views, high-freq-in-detector → ring offset)
    #   Y -= clamp(diff, |Y|)
    #
    # In image domain after FBP, ring artifacts are concentric circles around
    # the scan center.  In polar coords (r, θ) around that center:
    #   - Rings are CONSTANT IN θ at each fixed r          (≡ persistent in views)
    #   - Rings are HIGH-FREQUENCY IN r                    (≡ high-freq in detector)
    # So the algorithm becomes:
    #   1. Cartesian → polar (bilinear interp, around `center`)
    #   2. mean_along_θ[r] = mean(polar[r, :])             ← Ymv equivalent
    #   3. smoothed_r[r] = box-smooth mean_along_θ in r    ← Ymr equivalent
    #   4. ring_per_r[r] = mean_along_θ[r] - smoothed_r[r] ← ring estimate per r
    #   5. ring_polar[r, θ] = ring_per_r[r]                (constant in θ)
    #   6. Clamp: |ring_polar| ≤ |polar|
    #   7. Inverse polar → ring_cartesian
    #   8. Subtract from slice
    #
    # `rad_smooth` plays the role of MCR's `rad` (default 7).  Larger ⇒ more
    # aggressive ring removal (smooths over a wider r window).
    function _suppress_rings_image!(
            vol::AbstractArray{Float32, 3},
            center::NTuple{2, <:Real};
            rad_smooth::Int = 7,
            n_θ::Int = 720,
            verbose::Bool = false,
        )
        n_col, n_row, n_z = size(vol)
        cx = Float32(center[1])
        cy = Float32(center[2])
        n_r = ceil(Int, hypot(max(cx - 1, n_col - cx), max(cy - 1, n_row - cy))) + 2

        polar    = Array{Float32}(undef, n_r, n_θ)
        ring_pol = Array{Float32}(undef, n_r, n_θ)
        cosθ = [cos(2π * (j - 1) / n_θ) for j in 1:n_θ]
        sinθ = [sin(2π * (j - 1) / n_θ) for j in 1:n_θ]

        function _box_smooth_1d(x::Vector{Float32}, rad::Int)
            n = length(x)
            out = similar(x)
            w = 2 * rad + 1
            @inbounds for i in 1:n
                lo = max(1, i - rad); hi = min(n, i + rad)
                s = 0.0; c = 0
                for k in lo:hi
                    s += x[k]; c += 1
                end
                out[i] = Float32(s / c)
            end
            out
        end

        t0 = time()
        for k in 1:n_z
            slice = @view vol[:, :, k]

            # 1. Cartesian → polar (bilinear interp).
            @inbounds Threads.@threads for jθ in 1:n_θ
                cθ = cosθ[jθ]; sθ = sinθ[jθ]
                for ir in 1:n_r
                    r = Float32(ir - 1)
                    x = cx + r * cθ
                    y = cy + r * sθ
                    if 1f0 ≤ x ≤ Float32(n_col) && 1f0 ≤ y ≤ Float32(n_row)
                        i0 = floor(Int, x); j0 = floor(Int, y)
                        fx = x - i0; fy = y - j0
                        i1 = clamp(i0 + 1, 1, n_col); j1 = clamp(j0 + 1, 1, n_row)
                        i0 = clamp(i0, 1, n_col); j0 = clamp(j0, 1, n_row)
                        polar[ir, jθ] = (1f0-fx)*(1f0-fy)*slice[i0, j0] +
                                        fx     *(1f0-fy)*slice[i1, j0] +
                                        (1f0-fx)*fy     *slice[i0, j1] +
                                        fx     *fy     *slice[i1, j1]
                    else
                        polar[ir, jθ] = 0f0
                    end
                end
            end

            # 2. Mean along θ at each radius (Ymv equivalent).
            mean_per_r = Vector{Float32}(undef, n_r)
            @inbounds for ir in 1:n_r
                s = 0.0
                for jθ in 1:n_θ
                    s += polar[ir, jθ]
                end
                mean_per_r[ir] = Float32(s / n_θ)
            end

            # 3. Box-smooth mean_per_r in r (Ymr equivalent).
            smoothed_per_r = _box_smooth_1d(mean_per_r, rad_smooth)

            # 4. ring_per_r = mean_per_r - smoothed_per_r (HF-in-r component).
            ring_per_r = mean_per_r .- smoothed_per_r

            # 5. Build ring map (constant in θ at each r) and clamp by |polar|.
            @inbounds for jθ in 1:n_θ, ir in 1:n_r
                d = ring_per_r[ir]
                v = polar[ir, jθ]
                if abs(d) > abs(v)
                    d = sign(d) * abs(v)
                end
                ring_pol[ir, jθ] = d
            end

            # 6. Polar → Cartesian (inverse bilinear interp), subtract from slice.
            @inbounds Threads.@threads for jy in 1:n_row
                for ix in 1:n_col
                    dx = Float32(ix) - cx
                    dy = Float32(jy) - cy
                    r  = hypot(dx, dy)
                    θ  = atan(dy, dx)
                    if θ < 0f0
                        θ += Float32(2π)
                    end
                    rf = r + 1f0
                    θf = θ * Float32(n_θ) / Float32(2π) + 1f0
                    r0 = floor(Int, rf); θ0 = floor(Int, θf)
                    fr = rf - r0; fθ = θf - θ0
                    r0 = clamp(r0, 1, n_r - 1)
                    θ0 = ((θ0 - 1) % n_θ) + 1
                    θ1 = (θ0 % n_θ) + 1
                    ring_val = (1f0-fr)*(1f0-fθ)*ring_pol[r0,   θ0] +
                               fr     *(1f0-fθ)*ring_pol[r0+1, θ0] +
                               (1f0-fr)*fθ     *ring_pol[r0,   θ1] +
                               fr     *fθ     *ring_pol[r0+1, θ1]
                    vol[ix, jy, k] -= ring_val
                end
            end
        end

        if verbose
            @info "[ring-suppress] $(round(time()-t0, digits=2))s   $(n_z) slices   center=($(round(cx, digits=1)), $(round(cy, digits=1)))   rad=$(rad_smooth)   nr×nθ=$(n_r)×$(n_θ)"
        end
        nothing
    end
end;

# ╔═╡ 08131d00-0000-4000-8000-000000000010
# Sino-domain bin combination + per-channel FBP — RSKR bypassed.
# Combines bin sinograms via I0-weighted Beer's law (proper polychromatic
# wide-bin measurement; equivalent to scanning at the I0-weighted combined
# spectrum):
#   N_grp(ray)  = Σ_{b∈grp} I0_b · exp(-p_b(ray))
#   p_grp(ray)  = -log( N_grp / Σ_{b∈grp} I0_b )
# Then FBP each combined sinogram → (vol_low, vol_high) in cm⁻¹.
# This is the canonical input for image-domain DECT material decomp
# (McCollough 2015, Yu 2012).
sim_scan1_combined_bin_volumes_s1 = let
    bins = sim_scan1_bins_corrected               # 4 corrected bin sinos (post-log)
    I0   = sim_scan1.I0_bins
    grp_low  = [1, 2]                              # 20-35 + 35-55 keV
    grp_high = [3, 4]                              # 55-70 + 70-140 keV
    eps_f    = Float32(1e-10)

    function _combine_sino(grp)
        I0_sum = Float32(sum(Float64.(I0[grp])))
        N = zeros(Float32, size(bins[1]))
        for b in grp
            I0b = Float32(I0[b])
            @. N += I0b * exp(-bins[b])
        end
        @. -log(max(N, eps_f) / I0_sum)
    end
    sino_low  = _combine_sino(grp_low)
    sino_high = _combine_sino(grp_high)

    geom       = sim_scan1.geom
    recon_size = sim_matrix_size
    function _fbp_one(s_cpu)
        g  = MtlArray(Float32.(s_cpu))
        ws = BS.create_fdk_recon_workspace(g, geom, recon_size; filter = sim_vmi_filter)
        v  = Array(BS.reconstruct!(ws, g, geom, recon_size))
        ws = nothing; g = nothing
        Float32.(v)
    end

    t0 = time()
    vol_low  = _fbp_one(sino_low)
    vol_high = _fbp_one(sino_high)
    GC.gc(true)

    @info "[Sino-combine + FBP]  $(round(time()-t0, digits=2))s   (RSKR bypassed)"
    @info "  vol_low  (bins $(grp_low)):  μ=[$(round(minimum(vol_low),  sigdigits=3)), $(round(maximum(vol_low),  sigdigits=3))]  mean=$(round(mean(vol_low),  sigdigits=3)) cm⁻¹"
    @info "  vol_high (bins $(grp_high)): μ=[$(round(minimum(vol_high), sigdigits=3)), $(round(maximum(vol_high), sigdigits=3))]  mean=$(round(mean(vol_high), sigdigits=3)) cm⁻¹"

    (vol_low  = vol_low,  vol_high = vol_high,
     sino_low = sino_low, sino_high = sino_high,
     grp_low  = grp_low,  grp_high = grp_high)
end;

# ╔═╡ 08131d00-0000-4000-8000-000000000006
# 2×2 image-domain sensitivity matrices M for (water, iodine) decomp.
# After sino-domain combination, the effective spectrum for each combined
# channel is the I0-weighted average of per-bin spectra:
#   ŵ_X(E) = Σ_{b∈grp_X} (I0_b / Σ I0_b) · ŵ_b(E)
#
# Three flavors exposed:
#   M           — spectrum-averaged ⟨μρ⟩_X (Yu 2012 Eq 2, naive linearization)
#   M_mono      — μρ at equivalent monoenergy E_mono = ⟨E⟩_X (IBHC limit)
#   M_cal       — path-effective μ̄_X(t) at representative material thicknesses,
#                 i.e. μ̄_X_b(t) = -log(∫ŵ_b·exp(-μρ_X·ρ_X·t)dE) / (ρ_X·t).
#                 Linearizes the polychromatic forward model around clinical
#                 paths (chest water + typical iodine contrast), giving a
#                 single-shot M with much smaller BH residual than naive M.
#
# Per-voxel decomp: vol_X(v) = M[X,w]·ρ_w(v) + M[X,I]·ρ_I(v).
# K-edge integral is exact because we sum the full μρ_I(E) curve (project
# memory `project_cong_basis_photo_compton.md`).
sim_scan1_M_matrix_s1 = let
    prot = BS.CTProtocol(kVp = 140.0, additional_filters = [("Ti", 0.9)])
    e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
    pcct_det = BS._build_pcct_detector(sim_scanner)
    kVp      = Float64(maximum(e_full))
    R_mat    = BS.compute_mc_drm(pcct_det, kVp)
    η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
    n_R      = size(R_mat, 1)
    drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)

    e = Float64.(e_full)
    μρ_w = [BS.compute_mass_μ_at_energy(XA.Materials.water, E) for E in e]
    μρ_I = [BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E) for E in e]

    grp_low  = sim_scan1_combined_bin_volumes_s1.grp_low
    grp_high = sim_scan1_combined_bin_volumes_s1.grp_high
    I0       = sim_scan1.I0_bins

    function _eff_spectrum(grp)
        I0_sum = sum(Float64.(I0[grp]))
        wc = zeros(Float64, length(e))
        for b in grp
            wb = [Float64(w_full[i]) * Float64(η_vec[i]) * Float64(R_mat[drm_row(e[i]), b]) for i in eachindex(e)]
            sb = sum(wb)
            sb > 0 || error("_eff_spectrum: bin $b has zero spectral weight")
            wbn = wb ./ sb
            weight = Float64(I0[b]) / I0_sum
            wc .+= weight .* wbn
        end
        wc ./= sum(wc)
        wc
    end

    w_low  = _eff_spectrum(grp_low)
    w_high = _eff_spectrum(grp_high)

    μρ_w_low  = Float32(sum(w_low  .* μρ_w))    # cm²/g (spectrum-averaged)
    μρ_I_low  = Float32(sum(w_low  .* μρ_I))
    μρ_w_high = Float32(sum(w_high .* μρ_w))
    μρ_I_high = Float32(sum(w_high .* μρ_I))

    # Equivalent monoenergies (Fan et al. 2025 IBHC: spectrum-weighted mean).
    E_mono_low  = sum(e .* w_low)
    E_mono_high = sum(e .* w_high)

    # Mass attenuations evaluated AT the equivalent monoenergies.  After IBHC
    # convergence the corrected sinos are linearized to mono behaviour, so
    # decomp uses these (not the spectrum-averaged ⟨μρ⟩ above).
    μρ_w_mono_low  = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water, E_mono_low))
    μρ_I_mono_low  = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E_mono_low))
    μρ_w_mono_high = Float32(BS.compute_mass_μ_at_energy(XA.Materials.water, E_mono_high))
    μρ_I_mono_high = Float32(BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E_mono_high))

    # 2×2 spectrum-averaged M (legacy / reference).
    M = Float32[
        μρ_w_low   μρ_I_low  ;
        μρ_w_high  μρ_I_high ;
    ]
    M_inv = inv(M)

    # 2×2 monoenergetic M used by IBHC (Fan 2025 Eqs 13–14).
    M_mono = Float32[
        μρ_w_mono_low   μρ_I_mono_low  ;
        μρ_w_mono_high  μρ_I_mono_high ;
    ]
    M_mono_inv = inv(M_mono)

    # ── Path-effective calibrated M ──
    # μ̄_X_b(t) = -log(∫ŵ_b(E)·exp(-μρ_X(E)·ρ_X·t)dE) / (ρ_X·t)
    # Reference paths chosen for clinical chest/abdomen + typical iodine
    # contrast.  These are the BH-aware effective mass attenuations a beam
    # would experience after passing through ρ·t mass-density of the material.
    t_water_ref_g_cm2_s1  = 25.0    # 25 cm of water (clinical chest path)
    t_iodine_ref_g_cm2_s1 = 0.05    # typical clinical iodine path
    function _μ̄_path(w_eff::Vector{Float64}, μρ_curve::Vector{Float64}, ρt::Float64)
        ρt > 0 || error("_μ̄_path: ρt must be positive")
        trans = sum(w_eff[i] * exp(-μρ_curve[i] * ρt) for i in eachindex(w_eff))
        -log(max(trans, 1e-30)) / ρt
    end
    μ̄_w_low_cal  = Float32(_μ̄_path(w_low,  μρ_w, t_water_ref_g_cm2_s1))
    μ̄_I_low_cal  = Float32(_μ̄_path(w_low,  μρ_I, t_iodine_ref_g_cm2_s1))
    μ̄_w_high_cal = Float32(_μ̄_path(w_high, μρ_w, t_water_ref_g_cm2_s1))
    μ̄_I_high_cal = Float32(_μ̄_path(w_high, μρ_I, t_iodine_ref_g_cm2_s1))
    M_cal = Float32[
        μ̄_w_low_cal   μ̄_I_low_cal  ;
        μ̄_w_high_cal  μ̄_I_high_cal ;
    ]
    M_cal_inv = inv(M_cal)

    # Per-bin (4-vector) μ_water for HU-conversion of per-bin viz.
    function _bin_μρ_w(b::Int)
        wb = [Float64(w_full[i]) * Float64(η_vec[i]) * Float64(R_mat[drm_row(e[i]), b]) for i in eachindex(e)]
        sb = sum(wb); sb > 0 || error("_bin_μρ_w: bin $b has zero spectral weight")
        wbn = wb ./ sb
        Float32(sum(wbn .* μρ_w))
    end
    μ_water_per_bin  = Float32[_bin_μρ_w(b) for b in 1:4]   # cm⁻¹ at ρ_water = 1
    μ_water_combined = Float32[μρ_w_low, μρ_w_high]         # cm⁻¹ at ρ_water = 1

    # Per-energy spectra (normalized — sum = 1) and material mass attenuation
    # curves for IBHC's polychromatic forward sino prediction (Fan 2025 Eqs 22-23).
    e_grid       = Float32.(e)              # n_E
    S_low_arr    = Float32.(w_low)           # n_E, sums to 1
    S_high_arr   = Float32.(w_high)          # n_E, sums to 1
    μρ_w_curve   = Float32.(μρ_w)            # n_E
    μρ_I_curve   = Float32.(μρ_I)            # n_E

    @info "[M (2×2, spectrum-averaged)]:"
    @info "  low  (bins $(grp_low)):   μρ_water=$(round(μρ_w_low,  sigdigits=4))   μρ_iodine=$(round(μρ_I_low,  sigdigits=4))   E_eff≈$(round(E_mono_low,  digits=1)) keV"
    @info "  high (bins $(grp_high)):   μρ_water=$(round(μρ_w_high, sigdigits=4))   μρ_iodine=$(round(μρ_I_high, sigdigits=4))   E_eff≈$(round(E_mono_high, digits=1)) keV"
    @info "  cond(M) = $(round(cond(M), sigdigits=4))"
    @info "[M_mono (2×2, IBHC monoenergetic at E_mono)]:"
    @info "  low  E_mono=$(round(E_mono_low, digits=1)) keV: μρ_water=$(round(μρ_w_mono_low, sigdigits=4))  μρ_iodine=$(round(μρ_I_mono_low, sigdigits=4))"
    @info "  high E_mono=$(round(E_mono_high, digits=1)) keV: μρ_water=$(round(μρ_w_mono_high, sigdigits=4))  μρ_iodine=$(round(μρ_I_mono_high, sigdigits=4))"
    @info "  cond(M_mono) = $(round(cond(M_mono), sigdigits=4))"
    @info "[M_cal (2×2, path-effective at t_w=$(t_water_ref_g_cm2_s1) cm, t_I=$(t_iodine_ref_g_cm2_s1) g/cm²)]:"
    @info "  low :  μ̄_water=$(round(μ̄_w_low_cal,  sigdigits=4))   μ̄_iodine=$(round(μ̄_I_low_cal,  sigdigits=4))"
    @info "  high:  μ̄_water=$(round(μ̄_w_high_cal, sigdigits=4))   μ̄_iodine=$(round(μ̄_I_high_cal, sigdigits=4))"
    @info "  cond(M_cal) = $(round(cond(M_cal), sigdigits=4))"

    (M = M, M_inv = M_inv,
     M_mono = M_mono, M_mono_inv = M_mono_inv,
     M_cal  = M_cal,  M_cal_inv  = M_cal_inv,
     t_water_ref_g_cm2 = Float32(t_water_ref_g_cm2_s1),
     t_iodine_ref_g_cm2 = Float32(t_iodine_ref_g_cm2_s1),
     grp_low = grp_low, grp_high = grp_high,
     μ_water_per_bin   = μ_water_per_bin,
     μ_water_combined  = μ_water_combined,
     E_mono_low  = Float32(E_mono_low),
     E_mono_high = Float32(E_mono_high),
     μρ_w_mono_low  = μρ_w_mono_low,
     μρ_I_mono_low  = μρ_I_mono_low,
     μρ_w_mono_high = μρ_w_mono_high,
     μρ_I_mono_high = μρ_I_mono_high,
     e_grid     = e_grid,
     S_low      = S_low_arr,
     S_high     = S_high_arr,
     μρ_w_curve = μρ_w_curve,
     μρ_I_curve = μρ_I_curve)
end;

# ╔═╡ 08131d00-0000-4000-8000-000000000002
# ── STAGE 3 — RSKR config (hyperparams for the cell directly below) ──
# Smaller h_param / fewer n_iter ⇒ less denoising (more residual noise).
# Larger h_param / more n_iter   ⇒ more denoising (smoother volumes).
begin
    rskr_n_iter_s1   = 10         # Bregman iters (paper §2.5: 3-4 typical)
    rskr_h_param_s1  = 2.0       # range-kernel scale (paper Fig 1d: 1.2-1.5 typical)
    rskr_radius_s1   = 2         # spatial kernel half-radius
    rskr_γ_s1        = 0.5       # rank-sparse h scaling exponent
end

# ╔═╡ 08131d00-0000-4000-8000-000000000009
# RSKR-2ch denoising on the combined (vol_low, vol_high) pair.
# Joint SVD + bilateral filter on the U-vectors preserves the anti-correlated
# (water ↔ iodine) noise structure by construction (Clark/Badea 2023).
sim_scan1_combined_bin_volumes_rskr_s1 = let
    vol_low  = sim_scan1_combined_bin_volumes_s1.vol_low
    vol_high = sim_scan1_combined_bin_volumes_s1.vol_high

    t0 = time()
    out = _rskr_2ch([vol_low, vol_high];
        n_iter  = rskr_n_iter_s1,
        h_param = rskr_h_param_s1,
        radius  = rskr_radius_s1,
        γ       = rskr_γ_s1,
        verbose = true,
    )
    @info "[RSKR-2ch on combined low/high] $(round(time()-t0, digits=1))s"
    @info "  vol_low  σ_in=$(round(_mad_haar_σ(vol_low),  sigdigits=3))  →  σ_out=$(round(_mad_haar_σ(out[1]), sigdigits=3))   (mean: $(round(mean(vol_low),  sigdigits=4)) → $(round(mean(out[1]), sigdigits=4)))"
    @info "  vol_high σ_in=$(round(_mad_haar_σ(vol_high), sigdigits=3))  →  σ_out=$(round(_mad_haar_σ(out[2]), sigdigits=3))   (mean: $(round(mean(vol_high), sigdigits=4)) → $(round(mean(out[2]), sigdigits=4)))"

    (vol_low = out[1], vol_high = out[2])
end;

# ╔═╡ 08131d11-0000-4000-8000-000000000001
# ── STAGE 4 — Ring artifact correction (port of MCR `suppress_rings.m`) ──
# Image-domain implementation via polar transform: at each radial bin r,
# subtract the high-frequency-in-r component of the per-r mean.  Done
# AFTER RSKR per the user's pipeline so denoising can't smear ring offsets
# into surrounding voxels.
#
# `ring_rad_s1` plays the role of MCR's `rad` (default 7).  Larger ⇒ more
# aggressive (smooths over wider r window).  `use_ring_correction_s1 = false`
# passes (vol_low, vol_high) through unchanged.
begin
    use_ring_correction_s1 = false
    ring_rad_s1            = 7
end;

# ╔═╡ a4a1ff64-2559-4b13-93f0-a953d33fc1b2
# ── STAGE 6 — Apply image-domain decomp per voxel → c_iodine map (mg/mL) ──
# The iodine map is despeckled via a 1D z-direction median filter (zero
# in-plane resolution loss — see `_median_filter_z` in shared helpers).
# Single-voxel xy impulses don't repeat in z-neighbors so the median
# wipes them while leaving every (x,y) feature untouched.
begin
    use_c_iodine_median_s1    = true
    c_iodine_median_radius_s1 = 1   # 1 ⇒ 3-slice z-window, 2 ⇒ 5-slice
end;

# ╔═╡ 08131f00-0000-4000-8000-000000000001
md"""
**VMI synthesis.** Per-energy HIR recon + Mono+ polish, fused into one cell.
Config: `vmi_energies_s1` selects synthesis energies; `hir_*_s1` controls HIR; `vmip_*_s1` controls Mono+.
"""

# ╔═╡ 08131009-a000-4000-8000-000000000001
# ── Scan 1 VMI target energies ────────────────────────────────────────────
vmi_energies_s1 = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 08131f02-0000-4000-8000-000000000001
md"""
**VMI synthesis (Scan 1, image-domain).** Per-energy synthesis from `c_iodine` map.
1. Reads `c_iodine` (mg/mL) and `vol_low_HU` from `sim_scan1_imdomain_decomp_s1`.
2. For each `E ∈ vmi_energies_s1`: swap iodine HU contribution at low bin for iodine HU contribution at E.
   `HU_E[v] = HU_low[v] + c_iodine[v] · (α_iod_E_phys − α_iod_low_cal)`
   where `α_iod_E_phys = μρ_iodine(E) / μρ_water(E)` (HU per mg/mL).
3. Optional Mono+ (frequency-split polish across energies) on top.
"""

# ╔═╡ 08131009-a000-4000-8000-000000000008
# ── Scan 1 radial cupping correction config (per-VMI) ────────────────────
# Even-polynomial radial fit on water-like background voxels → flatten cupping
# residual left in each VMI by the image-domain decomp.  Applied BEFORE Mono+.
begin
    vmi_cap_enable_s1     = true
    vmi_cap_fov_cm_s1     = sim_fov_cm
    vmi_cap_hu_lo_s1      = -70.0
    vmi_cap_hu_hi_s1      = 100.0
    vmi_cap_poly_order_s1 = 3
    vmi_cap_target_hu_s1  = 0.0
end

# ╔═╡ 08131009-a000-4000-8000-000000000007
# ── STAGE 8 — Mono+ config (per-keV LP sigma) ──
# `vmip_σ_per_E_s1`: Dict{E (keV) => σ_lp (px)}.  σ=0 → no smoothing at that
# energy (preserves spatial detail).  σ>0 → Gaussian LP at that energy
# extracts low-frequency content from `vmip_E_noise_opt_s1` (reference E)
# and replaces the high-frequency band of the target energy with its own
# noisy detail.  Higher σ at noisier energies (e.g. 40 keV) ⇒ more
# noise-suppression there.
#
# Typical Siemens Mono+ behavior: moderate σ at low E (40 keV is noisiest),
# zero at the reference E (70 keV — preserved as-is).
begin
    use_mono_plus_s1     = true
    vmip_E_noise_opt_s1  = 70.0    # reference energy (preserved noise-wise)
    vmip_σ_per_E_s1      = Dict{Float64, Float64}(
        40.0  => 2.0,    # px σ at 40 keV  — high (low-E VMI is noisiest)
        70.0  => 0.0,    # px σ at 70 keV  — reference, no smoothing
        100.0 => 2.0,    # px σ at 100 keV — moderate
        140.0 => 2.0,    # px σ at 140 keV — low (high-E VMI is cleaner)
    )
end

# ╔═╡ 08131009-a000-4000-8000-00000000000a
# ── Scan 1 Mono+ edge-mask config ──────────────────────────────────────
# Mono+ frequency-split adds a bright ring at the phantom-air boundary
# (HP_opt = VMI_opt − LP_opt has a positive spike at the sharp edge).
# When `use_mono_plus_edge_mask_s1`, the apply_mono_plus! call gets the
# eroded phantom ground-truth mask so outside-mask voxels are reverted to
# the raw VMI_E.  `mono_plus_edge_erode_px_s1` should be ≥ ~3·max(σ).
begin
    use_mono_plus_edge_mask_s1 = true
    mono_plus_edge_erode_px_s1 = 8.0   # ≥ 3·max(σ_lp_px) recommended
end

# ╔═╡ 08131009-a000-4000-8000-000000000006
# ── Scan 1 HIR config ─────────────────────────────────────────────────────
begin
    use_hir_s1         = true
    hir_strength_s1    = 3
    hir_lambda_s1      = 4.0f0
    hir_nepochs_s1     = 2
    hir_n_subsets_s1   = 12
    hir_huber_delta_s1 = 0.06f0
    hir_relaxation_s1  = 0.35f0
end

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

# ╔═╡ 08131d11-0000-4000-8000-000000000002
sim_scan1_combined_bin_volumes_ring_s1 = let
    if !use_ring_correction_s1
        @info "[ring-correct] DISABLED — passing RSKR (vol_low, vol_high) through unchanged"
        (vol_low  = sim_scan1_combined_bin_volumes_rskr_s1.vol_low,
         vol_high = sim_scan1_combined_bin_volumes_rskr_s1.vol_high)
    else
        seg = sim_seg_result_s1
        cx, cy = Float64(seg.center.cx), Float64(seg.center.cy)

        vol_low  = copy(sim_scan1_combined_bin_volumes_rskr_s1.vol_low)
        vol_high = copy(sim_scan1_combined_bin_volumes_rskr_s1.vol_high)

        _suppress_rings_image!(vol_low,  (cx, cy); rad_smooth = Int(ring_rad_s1), verbose = true)
        _suppress_rings_image!(vol_high, (cx, cy); rad_smooth = Int(ring_rad_s1), verbose = true)

        Δ_low  = mean(abs.(vol_low  .- sim_scan1_combined_bin_volumes_rskr_s1.vol_low))
        Δ_high = mean(abs.(vol_high .- sim_scan1_combined_bin_volumes_rskr_s1.vol_high))
        @info "[ring-correct] mean |Δμ| removed:  low=$(round(Δ_low, sigdigits=3)) cm⁻¹   high=$(round(Δ_high, sigdigits=3)) cm⁻¹"

        (vol_low = vol_low, vol_high = vol_high)
    end
end;

# ╔═╡ 08131d15-0000-4000-8000-000000000001
# ── STAGE 5 — Image-domain decomp calibration (Ding 2012, Eq 3) ──
# Linear 3-parameter fit per material:
#   c_iodine  = a₀ + a₁·HU_low + a₂·HU_high      (mg/mL)
# Calibration ROIs: water rod (c=0) + 7 iodine rods (c = 2.0..20.0 mg/mL).
# 8 calibration points → over-determined LSQ for 3 unknowns → robust fit.
#
# Also fits the iodine HU sensitivity at each bin:
#   α_iod_low_cal  = mean rod-HU per (mg/mL) at low bin
#   α_iod_high_cal = mean rod-HU per (mg/mL) at high bin
# These are needed for VMI synthesis to subtract iodine contribution from
# HU_low and add iodine contribution at the target energy.
sim_scan1_imdomain_cal_s1 = let
    seg = sim_seg_result_s1
    μ_w_low, μ_w_high = sim_scan1_M_matrix_s1.μ_water_combined
    vol_low_HU  = @. (sim_scan1_combined_bin_volumes_ring_s1.vol_low  - μ_w_low ) / μ_w_low  * 1000f0
    vol_high_HU = @. (sim_scan1_combined_bin_volumes_ring_s1.vol_high - μ_w_high) / μ_w_high * 1000f0

    nx = size(vol_low_HU, 1)
    nz = size(vol_low_HU, 3)
    mid_z = round(Int, nz / 2)
    slice_low  = vol_low_HU[:, :, mid_z]
    slice_high = vol_high_HU[:, :, mid_z]

    roi_r_pix = 1.4 * 0.6 / (sim_fov_cm / nx)
    roi_r_sq  = roi_r_pix ^ 2
    function _rod_mean(slice, rod)
        i_lo = max(1, floor(Int, rod.cx - roi_r_pix - 1))
        i_hi = min(nx, ceil(Int, rod.cx + roi_r_pix + 1))
        j_lo = max(1, floor(Int, rod.cy - roi_r_pix - 1))
        j_hi = min(nx, ceil(Int, rod.cy + roi_r_pix + 1))
        s = 0.0; n = 0
        @inbounds for j in j_lo:j_hi, i in i_lo:i_hi
            if (i - rod.cx)^2 + (j - rod.cy)^2 <= roi_r_sq
                s += slice[i, j]; n += 1
            end
        end
        s / n
    end

    # Calibration rods: water rod + 7 iodine rods.  Iodine concentrations
    # are encoded in the rod names (e.g. "I 5.0" → 5.0 mg/mL).
    cal_rods_s1 = [
        ("Water (O)",  0.0),
        ("I 2.0",      2.0),
        ("I 2.5",      2.5),
        ("I 5.0",      5.0),
        ("I 7.5",      7.5),
        ("I 10.0",    10.0),
        ("I 15.0",    15.0),
        ("I 20.0",    20.0),
    ]
    rods_by_name = Dict(r.name => r for r in seg.rods)

    HU_low_cal  = Float64[]
    HU_high_cal = Float64[]
    c_iod_cal   = Float64[]
    for (nm, c) in cal_rods_s1
        r = rods_by_name[nm]
        push!(HU_low_cal,  _rod_mean(slice_low,  r))
        push!(HU_high_cal, _rod_mean(slice_high, r))
        push!(c_iod_cal,   c)
    end
    n_cal = length(c_iod_cal)

    # Ding Eq 3 LSQ: solve for (a_0, a_1, a_2) via normal equations.
    # design matrix A = [1, HU_low, HU_high], target = c_iodine.
    A_design = hcat(ones(n_cal), HU_low_cal, HU_high_cal)
    coeffs_iod = A_design \ c_iod_cal     # 3-vec [a_0, a_1, a_2]
    pred_c     = A_design * coeffs_iod
    rms_c      = sqrt(mean((pred_c .- c_iod_cal) .^ 2))

    # Iodine HU-per-(mg/mL) sensitivity at each bin (slope through origin
    # since pure water rod has HU≈0 and c=0).
    # Use simple LSQ slope: slope = Σ(c·HU) / Σ(c²)
    # Skipping water rod (c=0) for slope fit; iodine rods only.
    iod_idx = findall(c -> c > 0, c_iod_cal)
    α_iod_low_cal  = sum(c_iod_cal[iod_idx] .* HU_low_cal[iod_idx])  / sum(c_iod_cal[iod_idx] .^ 2)
    α_iod_high_cal = sum(c_iod_cal[iod_idx] .* HU_high_cal[iod_idx]) / sum(c_iod_cal[iod_idx] .^ 2)

    @info "[image-domain decomp cal] $(n_cal) ROIs (1 water + 7 iodine):"
    for i in 1:n_cal
        @info "  $(rpad(cal_rods_s1[i][1], 10))  c=$(c_iod_cal[i])  HU_low=$(round(HU_low_cal[i], digits=1))  HU_high=$(round(HU_high_cal[i], digits=1))  → c_pred=$(round(pred_c[i], digits=2))"
    end
    @info "  Ding Eq 3 fit:  a₀=$(round(coeffs_iod[1], digits=3))  a₁=$(round(coeffs_iod[2], sigdigits=4))  a₂=$(round(coeffs_iod[3], sigdigits=4))   RMS=$(round(rms_c, digits=3)) mg/mL"
    @info "  Iodine HU-per-(mg/mL) at bins:  α_low=$(round(α_iod_low_cal, digits=2))   α_high=$(round(α_iod_high_cal, digits=2))"

    (coeffs_iod      = Float32.(coeffs_iod),
     α_iod_low_cal   = Float32(α_iod_low_cal),
     α_iod_high_cal  = Float32(α_iod_high_cal),
     cal_rods        = cal_rods_s1,
     HU_low_cal      = HU_low_cal,
     HU_high_cal     = HU_high_cal,
     c_iod_cal       = c_iod_cal,
     pred_c          = pred_c,
     rms_c           = rms_c)
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

# ╔═╡ 08135030-0000-4000-8000-000000000001
md"**Qualitative.** Soft-tissue window (−200, 500) HU across all cells."

# ╔═╡ 08135040-0000-4000-8000-000000000001
md"**HU accuracy — simulated vs clinical.** 2×2 scatter (rows = rod category, cols = recon pairing).  Scan 1 FBP pairing uses clinical FBP Poly only (no clinical FBP VMIs at 3 mGy); Iterative pairing covers all 5 columns."

# ╔═╡ 08135042-0000-4000-8000-000000000001
md"**Water mean-HU sanity check.** Per-energy water rod (mean HU) for clinical and sim across all VMIs + Poly. Water should sit at 0 HU at every energy.  Any systematic offset reveals a calibration issue (wrong μ_water reference, BH residual, etc.)."

# ╔═╡ 08135070-0000-4000-8000-000000000001
md"**Water noise (σ) summary.** 5 energy-groups × 4 bars (Clin FBP · Sim FBP · Clin QIR3 · Sim HIR).  Clinical FBP VMIs absent → those bars are blank."

# ╔═╡ 08135050-0000-4000-8000-000000000001
md"**MTF — Clinical vs Simulated.** 2×5 grid: rows = recon pairing (FBP · Iterative); cols = energy.  FBP VMI columns lack clinical data at 3 mGy → show simulated alone."

# ╔═╡ 08135060-0000-4000-8000-000000000001
md"**NPS — Clinical vs Simulated.** 2×5 grid; same layout as MTF.  y-axis shared per column."

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

# ╔═╡ 08120e00-0000-4000-8000-000000000002
# Shared helpers used by Mono+ edge-mask + iodine-map z-despeckle wiring.
#   _build_recon_phantom_mask : nearest-neighbor resample of the ground-truth
#       phantom mask into recon space.
#   _erode_mask_2d/3d         : FFT-Gaussian + tight threshold "soft" erosion.
#   _median_filter_z          : 1D median along z per (x,y), edge-preserving
#       speckle removal with zero in-plane blur — ideal for z-invariant
#       phantoms (Gammex 472 rods).
begin
    function _build_recon_phantom_mask(
            phantom_mask_cpu::AbstractArray{<:Integer, 3},
            phantom_voxel_size::NTuple{3, <:Real},
            recon_size::NTuple{3, Int},
            recon_fov_cm::Real,
        )
        nx, ny, nz = recon_size
        pnx, pny, pnz = size(phantom_mask_cpu)
        rx = Float64(recon_fov_cm) / nx                         # cm/pixel (recon)
        px = Float64(phantom_voxel_size[1])                     # cm/pixel (phantom)
        cx_p = pnx / 2 + 0.5; cy_p = pny / 2 + 0.5
        cx_r = nx  / 2 + 0.5; cy_r = ny  / 2 + 0.5
        z_mid_p = pnz ÷ 2 + 1
        slice2d  = view(phantom_mask_cpu, :, :, z_mid_p)

        mask2d = falses(nx, ny)
        for j in 1:ny, i in 1:nx
            x_cm = (i - cx_r) * rx
            y_cm = (j - cy_r) * rx
            ip = round(Int, cx_p + x_cm / px)
            jp = round(Int, cy_p + y_cm / px)
            if 1 ≤ ip ≤ pnx && 1 ≤ jp ≤ pny
                mask2d[i, j] = slice2d[ip, jp] > 0
            end
        end

        mask3d = falses(nx, ny, nz)
        for k in 1:nz
            mask3d[:, :, k] .= mask2d
        end
        mask3d
    end

    # Soft erosion via FFT Gaussian smoothing of the binary mask, then a
    # tight threshold.  `erode_px` ≈ pixels removed from each edge.
    function _erode_mask_2d(mask2d::AbstractMatrix{Bool}, erode_px::Real)
        erode_px > 0 || return copy(mask2d)
        nx, ny = size(mask2d)
        σ = Float64(erode_px)
        σ² = σ^2
        fx = [min(i - 1, nx - (i - 1)) / nx for i in 1:nx]
        fy = [min(j - 1, ny - (j - 1)) / ny for j in 1:ny]
        kernel = [exp(-2π^2 * σ² * (fx[i]^2 + fy[j]^2)) for i in 1:nx, j in 1:ny]
        blurred = real.(FFTW.ifft(FFTW.fft(Float64.(mask2d)) .* kernel))
        blurred .≥ 0.999
    end

    function _erode_mask_3d(mask3d::AbstractArray{Bool, 3}, erode_px::Real)
        nx, ny, nz = size(mask3d)
        slice = _erode_mask_2d(view(mask3d, :, :, 1), erode_px)
        out = falses(nx, ny, nz)
        for k in 1:nz
            out[:, :, k] .= slice
        end
        out
    end

    # 1D median filter along z, per (x,y).  Exploits z-axis correlation in
    # the Gammex 472 phantom (rods are z-invariant) to wipe single-voxel
    # impulse noise in the c_iodine map with **zero in-plane resolution
    # loss**.  A speckle that exists at one slice and not its z-neighbors
    # gets replaced by the median of the [k-r .. k+r] z-window.
    # `radius=1` ⇒ 3-slice window, `radius=2` ⇒ 5-slice, etc.
    # At slice boundaries the window shrinks (no padding bias).
    function _median_filter_z(src::AbstractArray{Float32, 3}, radius::Int)
        radius ≥ 1 || return copy(src)
        nx, ny, nz = size(src)
        n_max = 2 * radius + 1
        out = similar(src)
        Threads.@threads for k in 1:nz
            klo = max(1, k - radius); khi = min(nz, k + radius)
            n = khi - klo + 1
            buf = Vector{Float32}(undef, n_max)
            @inbounds for j in 1:ny, i in 1:nx
                @inbounds for (m, kk) in enumerate(klo:khi)
                    buf[m] = src[i, j, kk]
                end
                sort!(view(buf, 1:n))
                out[i, j, k] = buf[(n + 1) ÷ 2]
            end
        end
        out
    end
end

# ╔═╡ 08131d15-0000-4000-8000-000000000004
sim_scan1_imdomain_decomp_s1 = let
    cal = sim_scan1_imdomain_cal_s1
    μ_w_low, μ_w_high = sim_scan1_M_matrix_s1.μ_water_combined
    vol_low_HU  = @. (sim_scan1_combined_bin_volumes_ring_s1.vol_low  - μ_w_low ) / μ_w_low  * 1000f0
    vol_high_HU = @. (sim_scan1_combined_bin_volumes_ring_s1.vol_high - μ_w_high) / μ_w_high * 1000f0

    a0, a1, a2 = cal.coeffs_iod
    c_iodine_map = @. a0 + a1 * vol_low_HU + a2 * vol_high_HU

    @info "[image-domain decomp]  c_iodine map (raw)  range=[$(round(minimum(c_iodine_map), digits=2)), $(round(maximum(c_iodine_map), digits=2))] mg/mL   mean=$(round(mean(c_iodine_map), digits=3))"

    # ── Optional z-direction median filter (zero in-plane blur) ──
    # Speckles are single-voxel xy impulses that don't repeat in adjacent
    # z slices (rods are z-invariant), so a 1D z-median replaces them
    # with the median of [k-r .. k+r] without touching xy resolution.
    if use_c_iodine_median_s1 && c_iodine_median_radius_s1 ≥ 1
        c_med = _median_filter_z(c_iodine_map, Int(c_iodine_median_radius_s1))
        Δrms_m = sqrt(mean((c_med .- c_iodine_map) .^ 2))
        c_iodine_map = c_med
        @info "[image-domain decomp]  z-median filter applied (radius=$(c_iodine_median_radius_s1), $(2*Int(c_iodine_median_radius_s1)+1)-slice window)  →  c_iodine despeckled.  RMS Δ = $(round(Δrms_m, digits=3)) mg/mL"
    else
        @info "[image-domain decomp]  z-median filter DISABLED — c_iodine impulse noise untouched"
    end

    (c_iodine = c_iodine_map,
     vol_low_HU  = vol_low_HU,
     vol_high_HU = vol_high_HU,
     geom = sim_scan1.geom)
end;

# ╔═╡ 08131d15-0000-4000-8000-000000000003
# ── Image-domain decomp viz: c_iodine map + rod HUs at calibration ROIs ──
let
    cal    = sim_scan1_imdomain_cal_s1
    decomp = sim_scan1_imdomain_decomp_s1
    seg    = sim_seg_result_s1

    mid_z = size(decomp.c_iodine, 3) ÷ 2
    c_slice = decomp.c_iodine[:, :, mid_z]
    lo_c, hi_c = quantile(vec(c_slice), 0.01), quantile(vec(c_slice), 0.99)

    fig = CM.Figure(size = (1400, 760), fontsize = 11)

    # Calibration rod fit chart: known c vs predicted c, with linear fit.
    ax_fit = CM.Axis(fig[1, 1]; title = "Calibration fit",
        xlabel = "Known c_iodine [mg/mL]", ylabel = "Predicted c_iodine [mg/mL]",
        aspect = 1.0)
    CM.scatter!(ax_fit, cal.c_iod_cal, cal.pred_c; markersize = 10, color = :steelblue)
    cmax = maximum(cal.c_iod_cal) + 1
    CM.lines!(ax_fit, [0.0, cmax], [0.0, cmax]; color = (:gray60, 0.7), linestyle = :dash, label = "Unity")
    CM.text!(ax_fit, 0.05 * cmax, 0.85 * cmax;
        text = "RMS = $(round(cal.rms_c, digits=3)) mg/mL\nα_iod_low = $(round(cal.α_iod_low_cal, digits=2)) HU/(mg/mL)\nα_iod_high = $(round(cal.α_iod_high_cal, digits=2)) HU/(mg/mL)",
        fontsize = 11, align = (:left, :top))

    ax_c = CM.Axis(fig[1, 2]; title = "c_iodine (mid-z)", aspect = CM.DataAspect())
    hm_c = CM.heatmap!(ax_c, c_slice; colormap = :viridis, colorrange = (lo_c, hi_c))
    CM.hidedecorations!(ax_c); CM.hidespines!(ax_c)
    for r in seg.rods
        CM.scatter!(ax_c, [r.cx], [r.cy]; markersize = 4, color = :cyan)
    end
    CM.Colorbar(fig[1, 3], hm_c; label = "c_iodine [mg/mL]", width = 18)

    CM.Label(fig[0, 1:3], "STAGE 5/6 — Image-domain decomp:  c_iodine map (mg/mL)";
        fontsize = 14, halign = :left)

    fig
end

# ╔═╡ 08131d00-0000-4000-8000-000000000008
# ── STAGE 7 — Image-domain VMI synthesis ──
# At each voxel:
#   HU_E[v] = HU_low[v] + c_iodine[v] · (α_iod_E_phys − α_iod_low_cal)
# where:
#   α_iod_E_phys      = μρ_iodine(E) / μρ_water(E)   (HU per mg/mL at E, physics)
#   α_iod_low_cal     = empirical iodine HU sensitivity at the low bin
#                      (fitted in Stage 5 from rod measurements)
#
# Intuition:  HU_low already contains the iodine contribution at the low-bin
# response (= c_iodine · α_iod_low_cal).  To get the VMI at energy E, swap
# that contribution for c_iodine · α_iod_E_phys.
# Output: `sim_scan1_vmi_rskr_s1` = Dict{Float64, Array{Float32, 3}} of HU volumes.
sim_scan1_vmi_rskr_s1 = let
    cal      = sim_scan1_imdomain_cal_s1
    decomp   = sim_scan1_imdomain_decomp_s1
    HU_low   = decomp.vol_low_HU
    c_iod    = decomp.c_iodine
    α_low_cal = Float32(cal.α_iod_low_cal)
    energies = Float64.(vmi_energies_s1)

    out = Dict{Float64, Array{Float32, 3}}()
    for E in energies
        μρ_w_E = BS.compute_mass_μ_at_energy(XA.Materials.water,  E)
        μρ_I_E = BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E)
        α_E_phys = Float32(μρ_I_E / μρ_w_E)   # HU per (mg/mL) at energy E

        Δα = α_E_phys - α_low_cal
        hu_vol = @. HU_low + c_iod * Δα
        out[E] = Float32.(hu_vol)

        mid_z = size(hu_vol, 3) ÷ 2
        roi   = hu_vol[200:300, 200:300, mid_z]
        @info "[VMI image-domain $(Int(E)) keV]  α_phys=$(round(α_E_phys, digits=2))  Δα=$(round(Δα, digits=2))  →  σ=$(round(std(roi), digits=1)) HU   mean=$(round(mean(roi), digits=1)) HU"
    end
    out
end;

# ╔═╡ 08131d05-0000-4000-8000-000000000001
# ── STAGE 7b — Radial cupping correction on raw VMI (Scan 1) ──
# Even-polynomial radial fit on background voxels flattens residual cupping
# left in each VMI by the image-domain decomp.  Applied per-energy BEFORE
# Mono+, mirroring the notebook 06 placement.
sim_scan1_vmi_capped_s1 = let
    energies = Float64.(vmi_energies_s1)
    raw_vmi  = sim_scan1_vmi_rskr_s1

    if !vmi_cap_enable_s1
        @info "[scan 1 VMI capping] DISABLED — pass-through"
        raw_vmi
    else
        out = Dict{Float64, Array{Float32, 3}}()
        for E in energies
            v = copy(raw_vmi[E])
            BS.apply_radial_cupping_correction!(v;
                fov_cm     = Float64(vmi_cap_fov_cm_s1),
                hu_lo      = Float64(vmi_cap_hu_lo_s1),
                hu_hi      = Float64(vmi_cap_hu_hi_s1),
                poly_order = Int(vmi_cap_poly_order_s1),
                target_hu  = Float64(vmi_cap_target_hu_s1),
            )
            out[E] = v
        end
        @info "[scan 1 VMI capping] $(length(energies)) energies  fov=$(vmi_cap_fov_cm_s1) cm  HU∈[$(vmi_cap_hu_lo_s1), $(vmi_cap_hu_hi_s1)]  poly=$(vmi_cap_poly_order_s1)  target=$(vmi_cap_target_hu_s1)"
        out
    end
end;

# ╔═╡ 08131f02-0000-4000-8000-000000000006
# ── Scan 1 raw vs capped VMI side-by-side per energy (diagnostic) ──
let
    raw    = sim_scan1_vmi_rskr_s1
    capped = sim_scan1_vmi_capped_s1
    energies = sort(collect(keys(raw)))
    mid_z = size(first(values(raw)), 3) ÷ 2
    fig = CM.Figure(size = (1200, 350 * length(energies)), fontsize = 11)
    for (i, E) in enumerate(energies)
        ax_r = CM.Axis(fig[i, 1]; title = "Scan 1 raw  $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax_r, raw[E][:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax_r); CM.hidespines!(ax_r)

        ax_c = CM.Axis(fig[i, 2]; title = "Scan 1 capped  $(Int(E)) keV  (poly=$(vmi_cap_poly_order_s1))",
            aspect = CM.DataAspect())
        CM.heatmap!(ax_c, capped[E][:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax_c); CM.hidespines!(ax_c)
    end
    CM.Label(fig[0, :]; text = "Scan 1 raw vs capping-corrected VMI  —  Soft-tissue window (−200 … 500 HU)",
        fontsize = 13, halign = :left)
    fig
end

# ╔═╡ 08131d05-0000-4000-8000-000000000002
# ── Scan 1 phantom mask in recon space (eroded) for Mono+ edge fix ──
# Built from the ground-truth `sim_phantom_cpu.mask`, resampled to the
# recon grid and eroded by `mono_plus_edge_erode_px_s1` pixels so the
# bright Mono+ ring at the phantom-air boundary falls outside the mask.
sim_scan1_phantom_mask_s1 = let
    base = _build_recon_phantom_mask(
        Array(sim_phantom_cpu.mask),
        sim_phantom_cpu.voxel_size,
        sim_matrix_size,
        sim_fov_cm)
    eroded = _erode_mask_3d(base, mono_plus_edge_erode_px_s1)
    n_in = count(eroded); n_tot = length(eroded)
    @info "[scan 1 phantom mask] recon $(sim_matrix_size)  erode=$(mono_plus_edge_erode_px_s1) px  inside=$(n_in)/$(n_tot) ($(round(100*n_in/n_tot, digits=1))%)"
    eroded
end;

# ╔═╡ 08131010-a000-4000-8000-000000000002
# ── STAGE 8 — Mono+ frequency-split polish (Scan 1) — "FBP equivalent" ──
# Mono+ runs on the capped VMI volumes (post-radial-cupping correction).  Its
# output is the headline "FBP-equivalent" VMI for measurements/grids — no
# iterative recon is performed on top of this in the FBP track.
sim_scan1_vmi_mono_s1 = let
    energies = Float64.(vmi_energies_s1)
    capped   = sim_scan1_vmi_capped_s1
    mid_z    = size(capped[energies[1]], 3) ÷ 2

    if !use_mono_plus_s1
        @info "[scan 1 Mono+] DISABLED — returning capped VMI volumes as final"
        capped
    else
        haskey(capped, vmip_E_noise_opt_s1) ||
            error("Mono+ reference vmip_E_noise_opt_s1=$(vmip_E_noise_opt_s1) keV not in vmi_energies_s1=$energies")
        for E in energies
            haskey(vmip_σ_per_E_s1, E) ||
                error("vmip_σ_per_E_s1 missing key for $(E) keV — define σ for every energy in vmi_energies_s1")
        end

        vols_in = [capped[E] for E in energies]
        σ_vec   = Float64[vmip_σ_per_E_s1[E] for E in energies]
        ws_mono = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(energies))
        res = BS.apply_mono_plus!(ws_mono, vols_in, energies;
            E_noise_opt = vmip_E_noise_opt_s1,
            σ_lp_px     = σ_vec,
            verbose     = true)

        # Post-Mono+ phantom edge-mask: where !mask, revert to raw capped VMI_E.
        # Eliminates the bright ring at phantom-air boundary that Mono+ injects
        # via HP_opt = VMI_opt − LP_opt.  Pure notebook-side post-processing —
        # does NOT require BS.apply_mono_plus! to support phantom_mask.
        edge_mask = use_mono_plus_edge_mask_s1 ? sim_scan1_phantom_mask_s1 : nothing

        out = Dict{Float64, Array{Float32, 3}}()
        for (i, E) in enumerate(energies)
            mp = copy(res.volumes[i])
            if edge_mask !== nothing && i != findfirst(==(Float64(vmip_E_noise_opt_s1)), Float64.(energies)) && σ_vec[i] != 0.0
                src = vols_in[i]
                @inbounds for j in eachindex(mp)
                    if !edge_mask[j]
                        mp[j] = src[j]
                    end
                end
            end
            roi_before = vols_in[i][200:300, 200:300, mid_z]
            roi_after  = mp[200:300, 200:300, mid_z]
            @info "[scan 1 Mono+] $(Int(E)) keV (σ_lp=$(σ_vec[i]) px): σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"
            out[E] = mp
        end
        ws_mono = nothing; GC.gc(true)
        out
    end
end;

# ╔═╡ 08131011-0000-4000-8000-000000000001
# ── STAGE 9 — HIR on top of Mono+ (Scan 1) — "HIR equivalent" ──
# Per-energy iterative refinement.  Pipeline mirrors single-kVp HIR in
# notebook 06: forward-project the (capped) VMI image to a synthetic
# single-energy sinogram, then run HIR.  Mono+ image at energy E warm-starts
# the iterate so HIR refines around a smooth init rather than the FDK init.
# No basis sinograms are involved — everything stays per-energy and
# downstream of the image-domain decomp.
sim_scan1_vmi_hir_s1 = let
    energies = Float64.(vmi_energies_s1)
    capped   = sim_scan1_vmi_capped_s1
    mono     = sim_scan1_vmi_mono_s1
    mid_z    = size(mono[energies[1]], 3) ÷ 2

    if !use_hir_s1
        @info "[scan 1 HIR] DISABLED — returning Mono+ VMI volumes as HIR output"
        mono
    else
        geom       = sim_scan1.geom
        recon_size = sim_matrix_size

        out = Dict{Float64, Array{Float32, 3}}()
        for E in energies
            # μ_water at the per-energy keV for HU↔μ conversion.
            μw_E = Float32(BS.compute_μ_at_energy(XA.Materials.water, Float64(E)))

            # Sanitize: clamp HU into a physically plausible range so
            # tail values (NaN/Inf/large negative) don't blow up the μ
            # conversion or HIR's forward-projection / Huber gradient.
            hu_capped = clamp.(replace(capped[E], NaN => -1000f0, Inf => -1000f0, -Inf => -1000f0),
                               -1000f0, 3000f0)
            hu_mono   = clamp.(replace(mono[E],   NaN => -1000f0, Inf => -1000f0, -Inf => -1000f0),
                               -1000f0, 3000f0)

            # GPU-side conversion + forward project — keeps the data path
            # consistent with HIR's GPU-side internal forward projection.
            μ_data_gpu = MtlArray(@. μw_E * (1f0 + hu_capped / 1000f0))
            sino_gpu   = BS.forward_project(μ_data_gpu, geom)
            init_gpu   = MtlArray(@. μw_E * (1f0 + hu_mono / 1000f0))

            ws_hir = BS.create_hir_recon_workspace(
                sino_gpu, geom, recon_size;
                strength = hir_strength_s1,
                filter   = sim_custom_poly_filter)
            ws_hir.params = BS.HIRParams(
                hir_strength_s1, hir_lambda_s1, 30, hir_nepochs_s1,
                hir_n_subsets_s1, hir_huber_delta_s1, hir_relaxation_s1, (25, 35))
            BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; init_volume = init_gpu)
            recon_μ = Array(ws_hir.volume)

            recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μw_E))
            BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

            roi_before = mono[E][200:300, 200:300, mid_z]
            roi_after  = recon_hu[200:300, 200:300, mid_z]
            @info "[scan 1 HIR-on-Mono+] $(Int(E)) keV  μw_E=$(round(μw_E, sigdigits=3)) cm⁻¹  σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"

            out[E] = recon_hu
            ws_hir = nothing; sino_gpu = nothing; init_gpu = nothing
            μ_data_gpu = nothing; hu_capped = nothing; hu_mono = nothing
            GC.gc(true)
        end
        out
    end
end;

# ╔═╡ 08135020-0000-4000-8000-000000000001
# Measurements driven by sim_seg_result_s1.  Poly has both FBP and HIR.
# VMI carries TWO sets:
#   • vmi_fbp = post-Mono+ image (Mono+ is the FBP-equivalent — capping → Mono+, no iter recon).
#   • vmi_hir = HIR-on-Mono+ image (per-energy HIR with Mono+ warm start).
sim_measurements_scan1 = let
    seg = sim_seg_result_s1
    m(vol, name) = measure_scan(vol, seg.mask, seg.rods, seg.center, name; fov_cm = sim_fov_cm)
    (
        poly_fbp = m(sim_scan1_poly_fbp, "scan1_poly_fbp"),
        poly_hir = m(sim_scan1_poly_hir, "scan1_poly_hir"),
        vmi_fbp  = Dict(E => m(sim_scan1_vmi_mono_s1[E], "scan1_vmi_fbp_$(Int(E))keV") for E in vmi_energies_s1),
        vmi_hir  = Dict(E => m(sim_scan1_vmi_hir_s1[E],  "scan1_vmi_hir_$(Int(E))keV") for E in vmi_energies_s1),
    )
end;

# ╔═╡ 08135020-0000-4000-8000-000000000002
# 4×5 comparison grid.
#   Rows: Clin FBP · Sim FBP · Clin QIR3 · Sim HIR
#   Cols: Poly · VMI 40 · 70 · 100 · 140 keV
# Poly column has data in all 4 rows.  VMI columns: rows 2 (Sim FBP =
# Mono+ output), 3 (Clin QIR3) and 4 (Sim HIR = HIR-on-Mono+).
# Row 1 (Clin FBP) VMI cells stay "not acquired" — Scan 1 didn't acquire
# clinical FBP VMIs at 3 mGy.
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

    # Row 2: Simulated FBP — Poly FBP + image-domain VMI + capping + Mono+.
    vols[2, 1] = sim_scan1_poly_fbp;    meas[2, 1] = sim_measurements_scan1.poly_fbp
    for (c, E) in enumerate(vmi_E)
        vols[2, c + 1] = sim_scan1_vmi_mono_s1[E]
        meas[2, c + 1] = sim_measurements_scan1.vmi_fbp[E]
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

    # Row 4: Simulated HIR — Poly HIR + HIR-on-Mono+ VMIs (headline output).
    vols[4, 1] = sim_scan1_poly_hir;    meas[4, 1] = sim_measurements_scan1.poly_hir
    for (c, E) in enumerate(vmi_E)
        vols[4, c + 1] = sim_scan1_vmi_hir_s1[E]
        meas[4, c + 1] = sim_measurements_scan1.vmi_hir[E]
    end

    is_clinical = r -> r == 1 || r == 3
    slice_of = (row, vol) -> vol === nothing ? nothing :
        (is_clinical(row) ? vol[:, :, seg_result.slice_idx] : vol[:, :, size(vol, 3) ÷ 2])

    (rows = row_labels, cols = col_labels, vols = vols, meas = meas,
     is_clinical = is_clinical, slice_of = slice_of)
end;

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

# ╔═╡ 08135042-0000-4000-8000-000000000002
let
    g = scan1_grid
    water_idx = 1   # "Water (O)" is rod index 1
    n_cols = length(g.cols)
    xs = collect(1:n_cols)
    bw = 0.18
    offsets = (-1.5, -0.5, 0.5, 1.5) .* bw

    μ = Matrix{Float64}(undef, 4, n_cols)
    for r in 1:4, c in 1:n_cols
        m = g.meas[r, c]
        μ[r, c] = m === nothing ? NaN : Float64(m.rod_means[water_idx])
    end

    row_style = [
        ("Clinical FBP",  :steelblue,  :solid),
        ("Simulated FBP", :darkorange, :solid),
        ("Clinical QIR3", :steelblue,  :outline),
        ("Simulated HIR", :darkorange, :outline),
    ]

    fig = CM.Figure(size = (1150, 500), fontsize = 12)
    ax  = CM.Axis(fig[1, 1];
        title  = "Scan 1 (140 kVp / 52 mA / 3 mGy) — Water mean HU (target = 0)",
        xlabel = "Reconstruction / Energy",
        ylabel = "Water mean (HU)",
        xticks = (xs, g.cols))

    for (r, (lab, color, style)) in enumerate(row_style)
        y = μ[r, :]
        x_r = xs .+ offsets[r]
        valid = .!isnan.(y)
        if !any(valid); continue; end
        if style == :solid
            CM.barplot!(ax, x_r[valid], y[valid]; width = bw, color = color,
                strokecolor = :black, strokewidth = 0.4, label = lab)
        else
            CM.barplot!(ax, x_r[valid], y[valid]; width = bw,
                color = (color, 0.0), strokecolor = color, strokewidth = 1.5, label = lab)
        end
    end
    CM.hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 1.0)
    CM.axislegend(ax; position = :lt, framevisible = true)

    @info "[Scan 1 water mean HU] target = 0:"
    for c in 1:n_cols
        @info "  $(rpad(g.cols[c], 14))  ClinFBP=$(round(μ[1,c], digits=1)) ClinQIR3=$(round(μ[3,c], digits=1)) HU   SimFBP=$(round(μ[2,c], digits=1)) SimHIR=$(round(μ[4,c], digits=1)) HU"
    end
    fig
end

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

# _lohi_stage_viz — used by per-stage scan-1 intermediate viz cells
# (FBP, RSKR, ring-corrected).  Takes (vol_low, vol_high) in cm⁻¹, converts
# to HU using each bin's water reference μ, draws low/high heatmaps and a
# bar chart of rod HUs at the calcium + iodine inserts.  Logs rod HUs too.
function _lohi_stage_viz(
        vol_low::AbstractArray{Float32, 3},
        vol_high::AbstractArray{Float32, 3},
        μ_w_low::Real,
        μ_w_high::Real,
        stage_name::String;
        seg = sim_seg_result_s1,
        fov_cm::Real = sim_fov_cm,
        clim_HU = (-200.0, 1500.0),
    )
    μ_w_low_f  = Float32(μ_w_low)
    μ_w_high_f = Float32(μ_w_high)
    mid_z = size(vol_low, 3) ÷ 2
    slice_low  = @. (vol_low[:, :, mid_z]  - μ_w_low_f ) / μ_w_low_f  * 1000f0
    slice_high = @. (vol_high[:, :, mid_z] - μ_w_high_f) / μ_w_high_f * 1000f0

    nx, ny = size(slice_low)
    roi_r_pix = 1.4 * 0.6 / (fov_cm / nx)
    roi_r_sq  = roi_r_pix ^ 2
    function _rod_mean(slice, rod)
        i_lo = max(1, floor(Int, rod.cx - roi_r_pix - 1))
        i_hi = min(nx, ceil(Int, rod.cx + roi_r_pix + 1))
        j_lo = max(1, floor(Int, rod.cy - roi_r_pix - 1))
        j_hi = min(ny, ceil(Int, rod.cy + roi_r_pix + 1))
        s = 0.0; n = 0
        @inbounds for j in j_lo:j_hi, i in i_lo:i_hi
            if (i - rod.cx)^2 + (j - rod.cy)^2 <= roi_r_sq
                s += slice[i, j]; n += 1
            end
        end
        s / n
    end

    rod_names = ["Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400",
                 "I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0"]
    rods_by_name = Dict(r.name => r for r in seg.rods)
    HU_low_rod  = Float64[_rod_mean(slice_low,  rods_by_name[nm]) for nm in rod_names]
    HU_high_rod = Float64[_rod_mean(slice_high, rods_by_name[nm]) for nm in rod_names]

    fig = CM.Figure(size = (1400, 760), fontsize = 11)
    CM.Label(fig[0, 1:4], "$(stage_name) — (vol_low, vol_high) HU + rod means";
        fontsize = 14, halign = :left)

    ax_l = CM.Axis(fig[1, 1:2]; title = "Low bin (HU, slice $mid_z)", aspect = CM.DataAspect())
    CM.heatmap!(ax_l, slice_low; colormap = :grays, colorrange = clim_HU)
    CM.hidedecorations!(ax_l); CM.hidespines!(ax_l)
    for r in seg.rods
        CM.scatter!(ax_l, [r.cx], [r.cy]; markersize = 4, color = :red)
    end

    ax_h = CM.Axis(fig[1, 3:4]; title = "High bin (HU, slice $mid_z)", aspect = CM.DataAspect())
    CM.heatmap!(ax_h, slice_high; colormap = :grays, colorrange = clim_HU)
    CM.hidedecorations!(ax_h); CM.hidespines!(ax_h)
    for r in seg.rods
        CM.scatter!(ax_h, [r.cx], [r.cy]; markersize = 4, color = :red)
    end

    n_rods = length(rod_names)
    xs = collect(1:n_rods)
    ax_b = CM.Axis(fig[2, 1:4];
        title = "Rod HUs",
        xlabel = "Rod", ylabel = "HU",
        xticks = (xs, rod_names),
        xticklabelrotation = π/4)
    CM.barplot!(ax_b, xs .- 0.2, HU_low_rod; width = 0.4, color = :steelblue, label = "Low bin")
    CM.barplot!(ax_b, xs .+ 0.2, HU_high_rod; width = 0.4, color = :darkorange, label = "High bin")
    CM.hlines!(ax_b, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.8)
    CM.axislegend(ax_b; position = :lt)

    @info "[$(stage_name)] rod HUs:"
    for (i, nm) in enumerate(rod_names)
        @info "  $(rpad(nm, 7))  low=$(round(HU_low_rod[i], digits=1)) HU   high=$(round(HU_high_rod[i], digits=1)) HU"
    end

    fig
end

# ╔═╡ 08131d10-0000-4000-8000-000000000001
# ── STAGE 2 — FBP rod measurements (intermediate viz) ──
# `sim_scan1_combined_bin_volumes_s1` has the raw FBP (vol_low, vol_high) in
# cm⁻¹.  Convert to HU per bin, measure rod means, plot.
let
    μ_w_low, μ_w_high = sim_scan1_M_matrix_s1.μ_water_combined
    _lohi_stage_viz(
        sim_scan1_combined_bin_volumes_s1.vol_low,
        sim_scan1_combined_bin_volumes_s1.vol_high,
        μ_w_low, μ_w_high,
        "STAGE 2 — FBP (raw)";
        clim_HU = Float64.(rskr_viz_hu_clim_s1),
    )
end

# ╔═╡ 08131d10-0000-4000-8000-000000000002
# ── RSKR-stage rod measurements (intermediate viz) ──
let
    μ_w_low, μ_w_high = sim_scan1_M_matrix_s1.μ_water_combined
    _lohi_stage_viz(
        sim_scan1_combined_bin_volumes_rskr_s1.vol_low,
        sim_scan1_combined_bin_volumes_rskr_s1.vol_high,
        μ_w_low, μ_w_high,
        "STAGE 3 — RSKR (denoised)";
        clim_HU = Float64.(rskr_viz_hu_clim_s1),
    )
end

# ╔═╡ 08131d11-0000-4000-8000-000000000003
# ── Ring-corrected stage rod measurements (intermediate viz) ──
let
    μ_w_low, μ_w_high = sim_scan1_M_matrix_s1.μ_water_combined
    _lohi_stage_viz(
        sim_scan1_combined_bin_volumes_ring_s1.vol_low,
        sim_scan1_combined_bin_volumes_ring_s1.vol_high,
        μ_w_low, μ_w_high,
        "STAGE 4 — Ring-corrected";
        clim_HU = Float64.(rskr_viz_hu_clim_s1),
    )
end

# ╔═╡ 08131f02-0000-4000-8000-000000000004
let
    _vmi_row_viz(sim_scan1_vmi_rskr_s1, "Scan 1 raw VMI (pre-capping)")
end

# ╔═╡ 08131f02-0000-4000-8000-000000000005
let
    _vmi_row_viz(sim_scan1_vmi_capped_s1, "Scan 1 capped VMI (post-cupping correction)")
end

# ╔═╡ 08131f02-0000-4000-8000-000000000002
let
    _vmi_row_viz(sim_scan1_vmi_mono_s1, "Scan 1 Mono+ (FBP equivalent)")
end

# ╔═╡ 08131f02-0000-4000-8000-000000000003
let
    tag = use_hir_s1 ? "Scan 1 HIR-on-Mono+ (HIR equivalent)" : "Scan 1 (HIR disabled — Mono+ pass-through)"
    _vmi_row_viz(sim_scan1_vmi_hir_s1, tag)
end

# ╔═╡ 08120009-0000-4000-8000-000000000001
md"""
**Scan 2 VMI pipeline.** 1:1 port of the Scan 1 image-domain pipeline:
**4-bin → 2-bin combine → FBP → RSKR → Ring → image-domain decomp (Ding Eq 3) → VMI synth → Mono+**.
All cells use the `_s2` suffix and have their own toggles independent of Scan 1.
"""

# ╔═╡ 08120d00-0000-4000-8000-000000000010
# ── STAGE 2 — Combined low/high FBP for Scan 2 ──
sim_scan2_combined_bin_volumes_s2 = let
    bins = sim_scan2_bins_corrected
    I0   = sim_scan2.I0_bins
    grp_low  = pcct_lohi_grouping_s2[1]
    grp_high = pcct_lohi_grouping_s2[2]
    eps_f    = Float32(1e-10)

    function _combine_sino(grp)
        I0_sum = Float32(sum(Float64.(I0[grp])))
        N = zeros(Float32, size(bins[1]))
        for b in grp
            I0b = Float32(I0[b])
            @. N += I0b * exp(-bins[b])
        end
        @. -log(max(N, eps_f) / I0_sum)
    end
    sino_low  = _combine_sino(grp_low)
    sino_high = _combine_sino(grp_high)

    geom       = sim_scan2.geom
    recon_size = sim_matrix_size
    function _fbp_one(s_cpu)
        g  = MtlArray(Float32.(s_cpu))
        ws = BS.create_fdk_recon_workspace(g, geom, recon_size; filter = sim_vmi_filter)
        v  = Array(BS.reconstruct!(ws, g, geom, recon_size))
        ws = nothing; g = nothing
        Float32.(v)
    end

    t0 = time()
    vol_low  = _fbp_one(sino_low)
    vol_high = _fbp_one(sino_high)
    GC.gc(true)
    @info "[Scan 2 sino-combine + FBP]  $(round(time()-t0, digits=2))s"
    @info "  vol_low  μ=$(round(mean(vol_low),  sigdigits=3)) cm⁻¹  range=[$(round(minimum(vol_low),  sigdigits=3)), $(round(maximum(vol_low),  sigdigits=3))]"
    @info "  vol_high μ=$(round(mean(vol_high), sigdigits=3)) cm⁻¹  range=[$(round(minimum(vol_high), sigdigits=3)), $(round(maximum(vol_high), sigdigits=3))]"

    (vol_low  = vol_low,  vol_high = vol_high,
     sino_low = sino_low, sino_high = sino_high,
     grp_low  = grp_low,  grp_high = grp_high)
end;

# ╔═╡ 08120d00-0000-4000-8000-000000000006
# ── 2×2 image-domain sensitivity matrices M for Scan 2 (water, iodine) ──
# Mirror of scan 1's M_matrix cell, with scan-2 protocol (140 kVp / 174 mA).
sim_scan2_M_matrix_s2 = let
    prot = BS.CTProtocol(kVp = 140.0, mA = sim_mA_scan2,
                         views = sim_n_views,
                         rotation_time = sim_rotation_time,
                         collimation_mm = sim_collimation_mm,
                         additional_filters = [("Ti", 0.9)])
    e_full, w_full = BS.resolve_source_spectrum_without_bowtie(sim_opts, prot; scanner = sim_scanner)
    pcct_det = BS._build_pcct_detector(sim_scanner)
    kVp      = Float64(maximum(e_full))
    R_mat    = BS.compute_mc_drm(pcct_det, kVp)
    η_vec    = BS.quantum_efficiency_vector(pcct_det.material, pcct_det.thickness_mm, e_full)
    n_R      = size(R_mat, 1)
    drm_row(E) = clamp(round(Int, (Float64(E) - 1.0) / (kVp - 1.0) * (n_R - 1)) + 1, 1, n_R)

    e = Float64.(e_full)
    μρ_w = [BS.compute_mass_μ_at_energy(XA.Materials.water, E) for E in e]
    μρ_I = [BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E) for E in e]

    grp_low  = sim_scan2_combined_bin_volumes_s2.grp_low
    grp_high = sim_scan2_combined_bin_volumes_s2.grp_high
    I0       = sim_scan2.I0_bins

    function _eff_spectrum(grp)
        I0_sum = sum(Float64.(I0[grp]))
        wc = zeros(Float64, length(e))
        for b in grp
            wb = [Float64(w_full[i]) * Float64(η_vec[i]) * Float64(R_mat[drm_row(e[i]), b]) for i in eachindex(e)]
            sb = sum(wb)
            sb > 0 || error("_eff_spectrum: bin $b has zero spectral weight")
            wbn = wb ./ sb
            weight = Float64(I0[b]) / I0_sum
            wc .+= weight .* wbn
        end
        wc ./= sum(wc)
        wc
    end

    w_low  = _eff_spectrum(grp_low)
    w_high = _eff_spectrum(grp_high)

    μρ_w_low  = Float32(sum(w_low  .* μρ_w))
    μρ_I_low  = Float32(sum(w_low  .* μρ_I))
    μρ_w_high = Float32(sum(w_high .* μρ_w))
    μρ_I_high = Float32(sum(w_high .* μρ_I))

    μ_water_combined = Float32[μρ_w_low, μρ_w_high]   # cm⁻¹ at ρ_water = 1

    @info "[Scan 2 M matrix]  spectrum-averaged ⟨μρ⟩:"
    @info "  low  (bins $(grp_low)):  μρ_water=$(round(μρ_w_low,  sigdigits=4))   μρ_iodine=$(round(μρ_I_low,  sigdigits=4))"
    @info "  high (bins $(grp_high)): μρ_water=$(round(μρ_w_high, sigdigits=4))   μρ_iodine=$(round(μρ_I_high, sigdigits=4))"

    (μ_water_combined = μ_water_combined,
     μρ_w_low  = μρ_w_low,  μρ_I_low  = μρ_I_low,
     μρ_w_high = μρ_w_high, μρ_I_high = μρ_I_high,
     grp_low = grp_low, grp_high = grp_high)
end;

# ╔═╡ 08120d00-0000-4000-8000-000000000002
# ── STAGE 3 — RSKR config (Scan 2) ──
begin
    rskr_n_iter_s2   = 2
    rskr_h_param_s2  = 0.5
    rskr_radius_s2   = 2
    rskr_γ_s2        = 0.5
end

# ╔═╡ 08120d00-0000-4000-8000-000000000009
# ── STAGE 3 — RSKR-2ch on the combined (vol_low, vol_high) pair ──
sim_scan2_combined_bin_volumes_rskr_s2 = let
    vol_low  = sim_scan2_combined_bin_volumes_s2.vol_low
    vol_high = sim_scan2_combined_bin_volumes_s2.vol_high

    t0 = time()
    out = _rskr_2ch([vol_low, vol_high];
        n_iter  = rskr_n_iter_s2,
        h_param = rskr_h_param_s2,
        radius  = rskr_radius_s2,
        γ       = rskr_γ_s2,
        verbose = true,
    )
    @info "[Scan 2 RSKR-2ch on combined low/high] $(round(time()-t0, digits=1))s"
    @info "  vol_low  σ_in=$(round(_mad_haar_σ(vol_low),  sigdigits=3))  →  σ_out=$(round(_mad_haar_σ(out[1]), sigdigits=3))"
    @info "  vol_high σ_in=$(round(_mad_haar_σ(vol_high), sigdigits=3))  →  σ_out=$(round(_mad_haar_σ(out[2]), sigdigits=3))"

    (vol_low = out[1], vol_high = out[2])
end;

# ╔═╡ 08120d11-0000-4000-8000-000000000001
# ── STAGE 4 — Ring artifact correction config (Scan 2) ──
begin
    use_ring_correction_s2 = false
    ring_rad_s2            = 7
end;

# ╔═╡ 08120d15-0000-4000-8000-000000000002
# ── STAGE 6 — Iodine map post-process config (Scan 2) ──
# Z-direction median filter — zero in-plane blur.  See scan 1 cell.
begin
    use_c_iodine_median_s2    = true
    c_iodine_median_radius_s2 = 1   # 1 ⇒ 3-slice z-window, 2 ⇒ 5-slice
end;

# ╔═╡ 08120009-a000-4000-8000-000000000001
# ── Scan 2 VMI target energies ──
vmi_energies_s2 = [40.0, 70.0, 100.0, 140.0];

# ╔═╡ 08120f02-0000-4000-8000-000000000001
md"""
**VMI synthesis (Scan 2, image-domain).** Per-energy synthesis from `c_iodine` map.
1. Reads `c_iodine` and `vol_low_HU` from `sim_scan2_imdomain_decomp_s2`.
2. For each `E ∈ vmi_energies_s2`: `HU_E[v] = HU_low[v] + c_iodine[v] · (α_iod_E_phys − α_iod_low_cal)`
3. Optional Mono+ on top.
"""

# ╔═╡ 08120009-a000-4000-8000-000000000008
# ── Scan 2 radial cupping correction config (per-VMI) ────────────────────
begin
    vmi_cap_enable_s2     = true
    vmi_cap_fov_cm_s2     = sim_fov_cm
    vmi_cap_hu_lo_s2      = -70.0
    vmi_cap_hu_hi_s2      = 100.0
    vmi_cap_poly_order_s2 = 3
    vmi_cap_target_hu_s2  = 0.0
end

# ╔═╡ 08120009-a000-4000-8000-000000000007
# ── STAGE 8 — Mono+ config (Scan 2, per-keV LP sigma) ──
begin
    use_mono_plus_s2     = true
    vmip_E_noise_opt_s2  = 70.0
    vmip_σ_per_E_s2      = Dict{Float64, Float64}(
        40.0  => 2.0,
        70.0  => 0.0,
        100.0 => 2.0,
        140.0 => 2.0,
    )
end

# ╔═╡ 08120009-a000-4000-8000-00000000000a
# ── Scan 2 Mono+ edge-mask config ──────────────────────────────────────
begin
    use_mono_plus_edge_mask_s2 = true
    mono_plus_edge_erode_px_s2 = 8.0   # ≥ 3·max(σ_lp_px) recommended
end

# ╔═╡ 08120d05-0000-4000-8000-000000000002
# ── Scan 2 phantom mask in recon space (eroded) for Mono+ edge fix ──
sim_scan2_phantom_mask_s2 = let
    base = _build_recon_phantom_mask(
        Array(sim_phantom_cpu.mask),
        sim_phantom_cpu.voxel_size,
        sim_matrix_size,
        sim_fov_cm)
    eroded = _erode_mask_3d(base, mono_plus_edge_erode_px_s2)
    n_in = count(eroded); n_tot = length(eroded)
    @info "[Scan 2 phantom mask] recon $(sim_matrix_size)  erode=$(mono_plus_edge_erode_px_s2) px  inside=$(n_in)/$(n_tot) ($(round(100*n_in/n_tot, digits=1))%)"
    eroded
end;

# ╔═╡ 08120009-a000-4000-8000-000000000009
# ── Scan 2 HIR config (per-VMI HIR-on-Mono+) ─────────────────────────────
begin
    use_hir_s2         = true
    hir_strength_s2    = 3
    hir_lambda_s2      = 4.0f0
    hir_nepochs_s2     = 2
    hir_n_subsets_s2   = 12
    hir_huber_delta_s2 = 0.06f0
    hir_relaxation_s2  = 0.35f0
end

# ╔═╡ 08125000-0000-4000-8000-000000000001
md"""
### Results

Cross-recon comparison grids at **Scan 2** (140 kVp / 174 mA / 10 mGy).
Rows: **Clin FBP · Sim FBP · Clin QIR3 · Sim HIR**.  Cols: **Poly · VMI 40 · VMI 70 · VMI 100 · VMI 140 keV**.

- **Poly column**: FBP and iterative recons on both clinical and sim sides.
- **VMI columns**: both Sim FBP (Row 2) and Sim HIR (Row 4) show the same image-domain VMI output (with Mono+ applied as the final stage of the pipeline).  Mono+ is part of the standard VMI output, so it's included on both rows.  The two rows differ only in the Poly column.  Clinical VMIs from `VMI/0/...` (FBP → Row 1) and `VMI/3/...` (QIR3 → Row 3).
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

# ╔═╡ 08120d10-0000-4000-8000-000000000001
# ── STAGE 2 — FBP rod measurements (intermediate viz) ──
let
    μ_w_low, μ_w_high = sim_scan2_M_matrix_s2.μ_water_combined
    _lohi_stage_viz(
        sim_scan2_combined_bin_volumes_s2.vol_low,
        sim_scan2_combined_bin_volumes_s2.vol_high,
        μ_w_low, μ_w_high,
        "SCAN 2 STAGE 2 — FBP (raw)";
        seg = sim_seg_result,
        clim_HU = Float64.(rskr_viz_hu_clim_s1),
    )
end

# ╔═╡ 08120d10-0000-4000-8000-000000000002
# ── STAGE 3 — RSKR rod measurements (intermediate viz) ──
let
    μ_w_low, μ_w_high = sim_scan2_M_matrix_s2.μ_water_combined
    _lohi_stage_viz(
        sim_scan2_combined_bin_volumes_rskr_s2.vol_low,
        sim_scan2_combined_bin_volumes_rskr_s2.vol_high,
        μ_w_low, μ_w_high,
        "SCAN 2 STAGE 3 — RSKR (denoised)";
        seg = sim_seg_result,
        clim_HU = Float64.(rskr_viz_hu_clim_s1),
    )
end

# ╔═╡ 08120d11-0000-4000-8000-000000000002
sim_scan2_combined_bin_volumes_ring_s2 = let
    if !use_ring_correction_s2
        @info "[Scan 2 ring-correct] DISABLED — passing RSKR (vol_low, vol_high) through unchanged"
        (vol_low  = sim_scan2_combined_bin_volumes_rskr_s2.vol_low,
         vol_high = sim_scan2_combined_bin_volumes_rskr_s2.vol_high)
    else
        seg = sim_seg_result
        cx, cy = Float64(seg.center.cx), Float64(seg.center.cy)

        vol_low  = copy(sim_scan2_combined_bin_volumes_rskr_s2.vol_low)
        vol_high = copy(sim_scan2_combined_bin_volumes_rskr_s2.vol_high)

        _suppress_rings_image!(vol_low,  (cx, cy); rad_smooth = Int(ring_rad_s2), verbose = true)
        _suppress_rings_image!(vol_high, (cx, cy); rad_smooth = Int(ring_rad_s2), verbose = true)

        Δ_low  = mean(abs.(vol_low  .- sim_scan2_combined_bin_volumes_rskr_s2.vol_low))
        Δ_high = mean(abs.(vol_high .- sim_scan2_combined_bin_volumes_rskr_s2.vol_high))
        @info "[Scan 2 ring-correct] mean |Δμ| removed:  low=$(round(Δ_low, sigdigits=3))  high=$(round(Δ_high, sigdigits=3)) cm⁻¹"

        (vol_low = vol_low, vol_high = vol_high)
    end
end;

# ╔═╡ 08120d11-0000-4000-8000-000000000003
# ── STAGE 4 — Ring-corrected stage rod measurements (intermediate viz) ──
let
    μ_w_low, μ_w_high = sim_scan2_M_matrix_s2.μ_water_combined
    _lohi_stage_viz(
        sim_scan2_combined_bin_volumes_ring_s2.vol_low,
        sim_scan2_combined_bin_volumes_ring_s2.vol_high,
        μ_w_low, μ_w_high,
        "SCAN 2 STAGE 4 — Ring-corrected";
        seg = sim_seg_result,
        clim_HU = Float64.(rskr_viz_hu_clim_s1),
    )
end

# ╔═╡ 08120d15-0000-4000-8000-000000000001
# ── STAGE 5 — Image-domain decomp calibration (Scan 2, Ding 2012 Eq 3) ──
sim_scan2_imdomain_cal_s2 = let
    seg = sim_seg_result
    μ_w_low, μ_w_high = sim_scan2_M_matrix_s2.μ_water_combined
    vol_low_HU  = @. (sim_scan2_combined_bin_volumes_ring_s2.vol_low  - μ_w_low ) / μ_w_low  * 1000f0
    vol_high_HU = @. (sim_scan2_combined_bin_volumes_ring_s2.vol_high - μ_w_high) / μ_w_high * 1000f0

    nx = size(vol_low_HU, 1)
    nz = size(vol_low_HU, 3)
    mid_z = round(Int, nz / 2)
    slice_low  = vol_low_HU[:, :, mid_z]
    slice_high = vol_high_HU[:, :, mid_z]

    roi_r_pix = 1.4 * 0.6 / (sim_fov_cm / nx)
    roi_r_sq  = roi_r_pix ^ 2
    function _rod_mean(slice, rod)
        i_lo = max(1, floor(Int, rod.cx - roi_r_pix - 1))
        i_hi = min(nx, ceil(Int, rod.cx + roi_r_pix + 1))
        j_lo = max(1, floor(Int, rod.cy - roi_r_pix - 1))
        j_hi = min(nx, ceil(Int, rod.cy + roi_r_pix + 1))
        s = 0.0; n = 0
        @inbounds for j in j_lo:j_hi, i in i_lo:i_hi
            if (i - rod.cx)^2 + (j - rod.cy)^2 <= roi_r_sq
                s += slice[i, j]; n += 1
            end
        end
        s / n
    end

    cal_rods_s2 = [
        ("Water (O)",  0.0),
        ("I 2.0",      2.0),
        ("I 2.5",      2.5),
        ("I 5.0",      5.0),
        ("I 7.5",      7.5),
        ("I 10.0",    10.0),
        ("I 15.0",    15.0),
        ("I 20.0",    20.0),
    ]
    rods_by_name = Dict(r.name => r for r in seg.rods)

    HU_low_cal  = Float64[]; HU_high_cal = Float64[]; c_iod_cal = Float64[]
    for (nm, c) in cal_rods_s2
        r = rods_by_name[nm]
        push!(HU_low_cal,  _rod_mean(slice_low,  r))
        push!(HU_high_cal, _rod_mean(slice_high, r))
        push!(c_iod_cal,   c)
    end
    n_cal = length(c_iod_cal)

    A_design = hcat(ones(n_cal), HU_low_cal, HU_high_cal)
    coeffs_iod = A_design \ c_iod_cal
    pred_c     = A_design * coeffs_iod
    rms_c      = sqrt(mean((pred_c .- c_iod_cal) .^ 2))

    iod_idx = findall(c -> c > 0, c_iod_cal)
    α_iod_low_cal  = sum(c_iod_cal[iod_idx] .* HU_low_cal[iod_idx])  / sum(c_iod_cal[iod_idx] .^ 2)
    α_iod_high_cal = sum(c_iod_cal[iod_idx] .* HU_high_cal[iod_idx]) / sum(c_iod_cal[iod_idx] .^ 2)

    @info "[Scan 2 image-domain decomp cal] $(n_cal) ROIs (1 water + 7 iodine):"
    for i in 1:n_cal
        @info "  $(rpad(cal_rods_s2[i][1], 10))  c=$(c_iod_cal[i])  HU_low=$(round(HU_low_cal[i], digits=1))  HU_high=$(round(HU_high_cal[i], digits=1))  → c_pred=$(round(pred_c[i], digits=2))"
    end
    @info "  Ding Eq 3 fit:  a₀=$(round(coeffs_iod[1], digits=3))  a₁=$(round(coeffs_iod[2], sigdigits=4))  a₂=$(round(coeffs_iod[3], sigdigits=4))   RMS=$(round(rms_c, digits=3)) mg/mL"
    @info "  Iodine HU-per-(mg/mL):  α_low=$(round(α_iod_low_cal, digits=2))   α_high=$(round(α_iod_high_cal, digits=2))"

    (coeffs_iod      = Float32.(coeffs_iod),
     α_iod_low_cal   = Float32(α_iod_low_cal),
     α_iod_high_cal  = Float32(α_iod_high_cal),
     cal_rods        = cal_rods_s2,
     HU_low_cal      = HU_low_cal,
     HU_high_cal     = HU_high_cal,
     c_iod_cal       = c_iod_cal,
     pred_c          = pred_c,
     rms_c           = rms_c)
end;

# ╔═╡ 08120d15-0000-4000-8000-000000000004
sim_scan2_imdomain_decomp_s2 = let
    cal = sim_scan2_imdomain_cal_s2
    μ_w_low, μ_w_high = sim_scan2_M_matrix_s2.μ_water_combined
    vol_low_HU  = @. (sim_scan2_combined_bin_volumes_ring_s2.vol_low  - μ_w_low ) / μ_w_low  * 1000f0
    vol_high_HU = @. (sim_scan2_combined_bin_volumes_ring_s2.vol_high - μ_w_high) / μ_w_high * 1000f0

    a0, a1, a2 = cal.coeffs_iod
    c_iodine_map = @. a0 + a1 * vol_low_HU + a2 * vol_high_HU

    @info "[Scan 2 image-domain decomp]  c_iodine map (raw)  range=[$(round(minimum(c_iodine_map), digits=2)), $(round(maximum(c_iodine_map), digits=2))] mg/mL   mean=$(round(mean(c_iodine_map), digits=3))"

    # ── Optional z-direction median filter (zero in-plane blur) ──
    if use_c_iodine_median_s2 && c_iodine_median_radius_s2 ≥ 1
        c_med = _median_filter_z(c_iodine_map, Int(c_iodine_median_radius_s2))
        Δrms_m = sqrt(mean((c_med .- c_iodine_map) .^ 2))
        c_iodine_map = c_med
        @info "[Scan 2 image-domain decomp]  z-median filter applied (radius=$(c_iodine_median_radius_s2), $(2*Int(c_iodine_median_radius_s2)+1)-slice window)  →  c_iodine despeckled.  RMS Δ = $(round(Δrms_m, digits=3)) mg/mL"
    else
        @info "[Scan 2 image-domain decomp]  z-median filter DISABLED — c_iodine impulse noise untouched"
    end

    (c_iodine = c_iodine_map,
     vol_low_HU  = vol_low_HU,
     vol_high_HU = vol_high_HU,
     geom = sim_scan2.geom)
end;

# ╔═╡ 08120d00-0000-4000-8000-000000000008
# ── STAGE 7 — Image-domain VMI synthesis (Scan 2) ──
sim_scan2_vmi_rskr_s2 = let
    cal      = sim_scan2_imdomain_cal_s2
    decomp   = sim_scan2_imdomain_decomp_s2
    HU_low   = decomp.vol_low_HU
    c_iod    = decomp.c_iodine
    α_low_cal = Float32(cal.α_iod_low_cal)
    energies = Float64.(vmi_energies_s2)

    out = Dict{Float64, Array{Float32, 3}}()
    for E in energies
        μρ_w_E = BS.compute_mass_μ_at_energy(XA.Materials.water,  E)
        μρ_I_E = BS.compute_mass_μ_at_energy(XA.Elements.Iodine, E)
        α_E_phys = Float32(μρ_I_E / μρ_w_E)

        Δα = α_E_phys - α_low_cal
        hu_vol = @. HU_low + c_iod * Δα
        out[E] = Float32.(hu_vol)

        mid_z = size(hu_vol, 3) ÷ 2
        roi   = hu_vol[200:300, 200:300, mid_z]
        @info "[Scan 2 VMI image-domain $(Int(E)) keV]  α_phys=$(round(α_E_phys, digits=2))  Δα=$(round(Δα, digits=2))  →  σ=$(round(std(roi), digits=1)) HU   mean=$(round(mean(roi), digits=1)) HU"
    end
    out
end;

# ╔═╡ 08120f02-0000-4000-8000-000000000004
let
    _vmi_row_viz(sim_scan2_vmi_rskr_s2, "Scan 2 raw VMI (pre-capping)")
end

# ╔═╡ 08120d05-0000-4000-8000-000000000001
# ── STAGE 7b — Radial cupping correction on raw VMI (Scan 2) ──
sim_scan2_vmi_capped_s2 = let
    energies = Float64.(vmi_energies_s2)
    raw_vmi  = sim_scan2_vmi_rskr_s2

    if !vmi_cap_enable_s2
        @info "[Scan 2 VMI capping] DISABLED — pass-through"
        raw_vmi
    else
        out = Dict{Float64, Array{Float32, 3}}()
        for E in energies
            v = copy(raw_vmi[E])
            BS.apply_radial_cupping_correction!(v;
                fov_cm     = Float64(vmi_cap_fov_cm_s2),
                hu_lo      = Float64(vmi_cap_hu_lo_s2),
                hu_hi      = Float64(vmi_cap_hu_hi_s2),
                poly_order = Int(vmi_cap_poly_order_s2),
                target_hu  = Float64(vmi_cap_target_hu_s2),
            )
            out[E] = v
        end
        @info "[Scan 2 VMI capping] $(length(energies)) energies  fov=$(vmi_cap_fov_cm_s2) cm  HU∈[$(vmi_cap_hu_lo_s2), $(vmi_cap_hu_hi_s2)]  poly=$(vmi_cap_poly_order_s2)  target=$(vmi_cap_target_hu_s2)"
        out
    end
end;

# ╔═╡ 08120f02-0000-4000-8000-000000000005
let
    _vmi_row_viz(sim_scan2_vmi_capped_s2, "Scan 2 capped VMI (post-cupping correction)")
end

# ╔═╡ 08120010-a000-4000-8000-000000000002
# ── STAGE 8 — Mono+ on capped VMI (Scan 2) — "FBP equivalent" ──
sim_scan2_vmi_mono_s2 = let
    energies = Float64.(vmi_energies_s2)
    capped   = sim_scan2_vmi_capped_s2

    if !use_mono_plus_s2
        @info "[Scan 2 Mono+] DISABLED — returning capped VMI volumes as final"
        capped
    else
        haskey(capped, vmip_E_noise_opt_s2) ||
            error("Mono+ reference vmip_E_noise_opt_s2=$(vmip_E_noise_opt_s2) keV not in vmi_energies_s2=$energies")
        for E in energies
            haskey(vmip_σ_per_E_s2, E) ||
                error("vmip_σ_per_E_s2 missing key for $(E) keV")
        end

        vols_in = [capped[E] for E in energies]
        σ_vec   = Float64[vmip_σ_per_E_s2[E] for E in energies]
        ws_mono = BS.create_mono_plus_workspace(vols_in[1]; n_energies = length(energies))
        res = BS.apply_mono_plus!(ws_mono, vols_in, energies;
            E_noise_opt = vmip_E_noise_opt_s2,
            σ_lp_px     = σ_vec,
            verbose     = true)

        # Post-Mono+ phantom edge-mask: notebook-side replacement of outside-
        # mask voxels with raw capped VMI_E.  See scan 1 cell for rationale.
        edge_mask = use_mono_plus_edge_mask_s2 ? sim_scan2_phantom_mask_s2 : nothing

        out = Dict{Float64, Array{Float32, 3}}()
        mid_z = size(vols_in[1], 3) ÷ 2
        for (i, E) in enumerate(energies)
            mp = copy(res.volumes[i])
            if edge_mask !== nothing && i != findfirst(==(Float64(vmip_E_noise_opt_s2)), Float64.(energies)) && σ_vec[i] != 0.0
                src = vols_in[i]
                @inbounds for j in eachindex(mp)
                    if !edge_mask[j]
                        mp[j] = src[j]
                    end
                end
            end
            roi_before = vols_in[i][200:300, 200:300, mid_z]
            roi_after  = mp[200:300, 200:300, mid_z]
            @info "[Scan 2 Mono+] $(Int(E)) keV (σ_lp=$(σ_vec[i]) px): σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"
            out[E] = mp
        end
        ws_mono = nothing; GC.gc(true)
        out
    end
end;

# ╔═╡ 08120f02-0000-4000-8000-000000000002
let
    _vmi_row_viz(sim_scan2_vmi_mono_s2, "Scan 2 Mono+ (FBP equivalent)")
end

# ╔═╡ 08120011-0000-4000-8000-000000000001
# ── STAGE 9 — HIR on top of Mono+ (Scan 2) — "HIR equivalent" ──
# Per-energy HIR.  Forward-projects the capped VMI image to a synthetic
# single-energy sinogram and runs HIR with the Mono+ image at energy E as
# the warm-start iterate (init_volume).  No basis sinograms involved.
sim_scan2_vmi_hir_s2 = let
    energies = Float64.(vmi_energies_s2)
    capped   = sim_scan2_vmi_capped_s2
    mono     = sim_scan2_vmi_mono_s2
    mid_z    = size(mono[energies[1]], 3) ÷ 2

    if !use_hir_s2
        @info "[Scan 2 HIR] DISABLED — returning Mono+ VMI volumes as HIR output"
        mono
    else
        geom       = sim_scan2.geom
        recon_size = sim_matrix_size

        out = Dict{Float64, Array{Float32, 3}}()
        for E in energies
            μw_E = Float32(BS.compute_μ_at_energy(XA.Materials.water, Float64(E)))

            hu_capped = clamp.(replace(capped[E], NaN => -1000f0, Inf => -1000f0, -Inf => -1000f0),
                               -1000f0, 3000f0)
            hu_mono   = clamp.(replace(mono[E],   NaN => -1000f0, Inf => -1000f0, -Inf => -1000f0),
                               -1000f0, 3000f0)

            μ_data_gpu = MtlArray(@. μw_E * (1f0 + hu_capped / 1000f0))
            sino_gpu   = BS.forward_project(μ_data_gpu, geom)
            init_gpu   = MtlArray(@. μw_E * (1f0 + hu_mono / 1000f0))

            ws_hir = BS.create_hir_recon_workspace(
                sino_gpu, geom, recon_size;
                strength = hir_strength_s2,
                filter   = sim_custom_poly_filter)
            ws_hir.params = BS.HIRParams(
                hir_strength_s2, hir_lambda_s2, 30, hir_nepochs_s2,
                hir_n_subsets_s2, hir_huber_delta_s2, hir_relaxation_s2, (25, 35))
            BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; init_volume = init_gpu)
            recon_μ = Array(ws_hir.volume)

            recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μw_E))
            BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

            roi_before = mono[E][200:300, 200:300, mid_z]
            roi_after  = recon_hu[200:300, 200:300, mid_z]
            @info "[Scan 2 HIR-on-Mono+] $(Int(E)) keV  μw_E=$(round(μw_E, sigdigits=3)) cm⁻¹  σ=$(round(std(roi_before), digits=1)) → $(round(std(roi_after), digits=1)) HU"

            out[E] = recon_hu
            ws_hir = nothing; sino_gpu = nothing; init_gpu = nothing
            μ_data_gpu = nothing; hu_capped = nothing; hu_mono = nothing
            GC.gc(true)
        end
        out
    end
end;

# ╔═╡ 08120f02-0000-4000-8000-000000000003
let
    tag = use_hir_s2 ? "Scan 2 HIR-on-Mono+ (HIR equivalent)" : "Scan 2 (HIR disabled — Mono+ pass-through)"
    _vmi_row_viz(sim_scan2_vmi_hir_s2, tag)
end

# ╔═╡ 08120f02-0000-4000-8000-000000000006
# ── Scan 2 raw vs capped VMI side-by-side per energy (diagnostic) ──
let
    raw    = sim_scan2_vmi_rskr_s2
    capped = sim_scan2_vmi_capped_s2
    energies = sort(collect(keys(raw)))
    mid_z = size(first(values(raw)), 3) ÷ 2
    fig = CM.Figure(size = (1200, 350 * length(energies)), fontsize = 11)
    for (i, E) in enumerate(energies)
        ax_r = CM.Axis(fig[i, 1]; title = "Scan 2 raw  $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax_r, raw[E][:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax_r); CM.hidespines!(ax_r)

        ax_c = CM.Axis(fig[i, 2]; title = "Scan 2 capped  $(Int(E)) keV  (poly=$(vmi_cap_poly_order_s2))",
            aspect = CM.DataAspect())
        CM.heatmap!(ax_c, capped[E][:, :, mid_z]; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax_c); CM.hidespines!(ax_c)
    end
    CM.Label(fig[0, :]; text = "Scan 2 raw vs capping-corrected VMI  —  Soft-tissue window (−200 … 500 HU)",
        fontsize = 13, halign = :left)
    fig
end

# ╔═╡ 08120d15-0000-4000-8000-000000000003
# ── STAGE 5/6 viz: c_iodine map + calibration fit ──
let
    cal    = sim_scan2_imdomain_cal_s2
    decomp = sim_scan2_imdomain_decomp_s2
    seg    = sim_seg_result

    mid_z = size(decomp.c_iodine, 3) ÷ 2
    c_slice = decomp.c_iodine[:, :, mid_z]
    lo_c, hi_c = quantile(vec(c_slice), 0.50), quantile(vec(c_slice), 0.99)

    fig = CM.Figure(size = (1400, 760), fontsize = 11)

    ax_fit = CM.Axis(fig[1, 1]; title = "Calibration fit",
        xlabel = "Known c_iodine [mg/mL]", ylabel = "Predicted c_iodine [mg/mL]",
        aspect = 1.0)
    CM.scatter!(ax_fit, cal.c_iod_cal, cal.pred_c; markersize = 10, color = :steelblue)
    cmax = maximum(cal.c_iod_cal) + 1
    CM.lines!(ax_fit, [0.0, cmax], [0.0, cmax]; color = (:gray60, 0.7), linestyle = :dash)
    CM.text!(ax_fit, 0.05 * cmax, 0.85 * cmax;
        text = "RMS = $(round(cal.rms_c, digits=3)) mg/mL\nα_iod_low = $(round(cal.α_iod_low_cal, digits=2)) HU/(mg/mL)\nα_iod_high = $(round(cal.α_iod_high_cal, digits=2)) HU/(mg/mL)",
        fontsize = 11, align = (:left, :top))

    ax_c = CM.Axis(fig[1, 2]; title = "c_iodine (mid-z)", aspect = CM.DataAspect())
    hm_c = CM.heatmap!(ax_c, c_slice; colormap = :viridis, colorrange = (lo_c, hi_c))
    CM.hidedecorations!(ax_c); CM.hidespines!(ax_c)
    for r in seg.rods
        CM.scatter!(ax_c, [r.cx], [r.cy]; markersize = 4, color = :cyan)
    end
    CM.Colorbar(fig[1, 3], hm_c; label = "c_iodine [mg/mL]", width = 18)

    CM.Label(fig[0, 1:3], "SCAN 2 STAGE 5/6 — Image-domain decomp:  c_iodine map (mg/mL)";
    fontsize = 14, halign = :left)

    fig
end

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
#   • vmi_fbp = post-Mono+ image (capping → Mono+, no iter recon).
#   • vmi_hir = HIR-on-Mono+ image (per-energy HIR with Mono+ warm start).
sim_measurements_scan2 = let
    seg = sim_seg_result
    m(vol, name) = measure_scan(vol, seg.mask, seg.rods, seg.center, name; fov_cm = sim_fov_cm)
    (
        poly_fbp = m(sim_scan2_poly_fbp, "scan2_poly_fbp"),
        poly_hir = m(sim_scan2_poly_hir, "scan2_poly_hir"),
        vmi_fbp  = Dict(E => m(sim_scan2_vmi_mono_s2[E], "scan2_vmi_fbp_$(Int(E))keV") for E in vmi_energies_s2),
        vmi_hir  = Dict(E => m(sim_scan2_vmi_hir_s2[E],  "scan2_vmi_hir_$(Int(E))keV") for E in vmi_energies_s2),
    )
end;

# ╔═╡ 08125020-0000-4000-8000-000000000002
# Shared lookup — 4×5 grids of volumes + measurements indexed by
# (row, col) where rows = [Clin FBP, Sim FBP, Clin QIR3, Sim HIR] and
# cols = [Poly, VMI 40, VMI 70, VMI 100, VMI 140].  `nothing` = "not acquired".
scan2_grid = let
    row_labels = ["Clin FBP", "Sim FBP", "Clin QIR3", "Sim HIR"]
    col_labels = ["Poly", "VMI 40 keV", "VMI 70 keV", "VMI 100 keV", "VMI 140 keV"]
    vmi_E      = vmi_energies_s2

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

    # Row 2: Simulated FBP — Poly FBP + image-domain VMI + capping + Mono+.
    vols[2, 1] = sim_scan2_poly_fbp;    meas[2, 1] = sim_measurements_scan2.poly_fbp
    for (c, E) in enumerate(vmi_E)
        vols[2, c + 1] = sim_scan2_vmi_mono_s2[E]
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

    # Row 4: Simulated HIR — Poly HIR + HIR-on-Mono+ VMIs.
    vols[4, 1] = sim_scan2_poly_hir;    meas[4, 1] = sim_measurements_scan2.poly_hir
    for (c, E) in enumerate(vmi_E)
        vols[4, c + 1] = sim_scan2_vmi_hir_s2[E]
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

# ╔═╡ 08131d15-0000-4000-8000-000000000002

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
# ╠═08010018-a000-4000-8000-000000000000
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
# ╠═08131006-d000-4000-8000-000000000001
# ╟─08131005-a000-4000-8000-000000000001
# ╟─08131009-0000-4000-8000-000000000000
# ╠═d335e1a7-cb56-44a9-ad4a-c54f3e1fc280
# ╟─08131d00-0000-4000-8000-000000000001
# ╠═0cbcb5fe-7e29-48fc-9d68-ab7cb05ac3c4
# ╠═08131d00-0000-4000-8000-000000000003
# ╠═08131d00-0000-4000-8000-000000000010
# ╟─08131d10-0000-4000-8000-000000000001
# ╠═08131d00-0000-4000-8000-000000000006
# ╠═08131d00-0000-4000-8000-000000000002
# ╠═08131d00-0000-4000-8000-000000000009
# ╟─08131d10-0000-4000-8000-000000000002
# ╠═08131d11-0000-4000-8000-000000000001
# ╠═08131d11-0000-4000-8000-000000000002
# ╟─08131d11-0000-4000-8000-000000000003
# ╠═08131d15-0000-4000-8000-000000000001
# ╠═a4a1ff64-2559-4b13-93f0-a953d33fc1b2
# ╠═08131d15-0000-4000-8000-000000000004
# ╟─08131d15-0000-4000-8000-000000000003
# ╟─08131f00-0000-4000-8000-000000000001
# ╠═08131009-a000-4000-8000-000000000001
# ╟─08131f02-0000-4000-8000-000000000001
# ╠═08131d00-0000-4000-8000-000000000008
# ╟─08131f02-0000-4000-8000-000000000004
# ╠═08131009-a000-4000-8000-000000000008
# ╠═08131d05-0000-4000-8000-000000000001
# ╟─08131f02-0000-4000-8000-000000000005
# ╟─08131f02-0000-4000-8000-000000000006
# ╠═08131009-a000-4000-8000-000000000007
# ╠═08131009-a000-4000-8000-00000000000a
# ╠═08131d05-0000-4000-8000-000000000002
# ╠═08131010-a000-4000-8000-000000000002
# ╟─08131f02-0000-4000-8000-000000000002
# ╠═08131009-a000-4000-8000-000000000006
# ╠═08131011-0000-4000-8000-000000000001
# ╟─08131f02-0000-4000-8000-000000000003
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
# ╟─08135042-0000-4000-8000-000000000001
# ╟─08135042-0000-4000-8000-000000000002
# ╟─08135070-0000-4000-8000-000000000001
# ╟─08135070-0000-4000-8000-000000000002
# ╟─08135050-0000-4000-8000-000000000001
# ╟─08135050-0000-4000-8000-000000000002
# ╟─08135060-0000-4000-8000-000000000001
# ╠═08135060-0000-4000-8000-000000000002
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
# ╠═08120e00-0000-4000-8000-000000000002
# ╠═08120e00-0000-4000-8000-000000000001
# ╟─08120009-0000-4000-8000-000000000001
# ╠═08120d00-0000-4000-8000-000000000010
# ╠═08120d10-0000-4000-8000-000000000001
# ╠═08120d00-0000-4000-8000-000000000006
# ╠═08120d00-0000-4000-8000-000000000002
# ╠═08120d00-0000-4000-8000-000000000009
# ╠═08120d10-0000-4000-8000-000000000002
# ╠═08120d11-0000-4000-8000-000000000001
# ╠═08120d11-0000-4000-8000-000000000002
# ╠═08120d11-0000-4000-8000-000000000003
# ╠═08120d15-0000-4000-8000-000000000001
# ╠═08120d15-0000-4000-8000-000000000002
# ╠═08120d15-0000-4000-8000-000000000004
# ╟─08120d15-0000-4000-8000-000000000003
# ╠═08120009-a000-4000-8000-000000000001
# ╟─08120f02-0000-4000-8000-000000000001
# ╠═08120d00-0000-4000-8000-000000000008
# ╟─08120f02-0000-4000-8000-000000000004
# ╠═08120009-a000-4000-8000-000000000008
# ╠═08120d05-0000-4000-8000-000000000001
# ╟─08120f02-0000-4000-8000-000000000005
# ╟─08120f02-0000-4000-8000-000000000006
# ╠═08120009-a000-4000-8000-000000000007
# ╠═08120009-a000-4000-8000-00000000000a
# ╠═08120d05-0000-4000-8000-000000000002
# ╠═08120010-a000-4000-8000-000000000002
# ╟─08120f02-0000-4000-8000-000000000002
# ╠═08120009-a000-4000-8000-000000000009
# ╠═08120011-0000-4000-8000-000000000001
# ╟─08120f02-0000-4000-8000-000000000003
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
# ╠═08131d15-0000-4000-8000-000000000002
