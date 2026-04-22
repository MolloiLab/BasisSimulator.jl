### A Pluto.jl notebook ###
# v0.19.0

# ╔═╡ 07010002-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
begin
    using Pkg: Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.resolve()
    Pkg.instantiate()
    using Revise
end

# ╔═╡ 783d265d-506c-4dd4-aa76-bd352f532c6d
using Markdown

# ╔═╡ 07010003-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using FileIO

# ╔═╡ 07010004-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using ImageMagick

# ╔═╡ 07010011-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using Unitful: @u_str

# ╔═╡ 07010012-0000-4000-8000-000000000000
using LinearAlgebra

# ╔═╡ 07010013-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using FFTW

# ╔═╡ 0701001a-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using Roots

# ╔═╡ 0701001b-0000-4000-8000-000000000000
using Statistics: quantile

# ╔═╡ 07010014-0000-4000-8000-000000000000
using Random

# ╔═╡ 07010015-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using Metal

# ╔═╡ 07010017-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using JLD2: JLD2

# ╔═╡ 07010018-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
using DelimitedFiles: DelimitedFiles

# ╔═╡ f3a91573-63b4-40df-8a9a-3a44d390bc79
using Unitful

# ╔═╡ 07010001-0000-4000-8000-000000000000
md"""
# 1. GE Revolution Apex Elite — Clinical Gammex 472 Scans
* **Scanner:** GE Revolution Apex (S/N REV2X2300094CN, cadence\_ct\_25.68)
* **Date:** 2026-02-23 | **Protocol:** 5.1 CHEST W/O (axial, no pitch)
* **Phantom:** Gammex 472 multi-energy CT calibration phantom
* **Recon FOV:** 350 mm | **Matrix:** 512 × 512 | **Pixel:** 0.684 mm | **Slice:** 0.625 mm | **Kernel:** STANDARD

SE folders each contain `0%/` (FBP) and `50%/` (ASiR-V 50%). CTDIvol from RDSR (32 cm body phantom).

| # | Type | kVp | mA | mAs | Rotation | Rows × Collim | CTDIvol (mGy) | DLP (mGy·cm) | Recon |
|---|------|-----|----|-----|----------|---------------|---------------|--------------|-------|
| 1 | SE | 120 |  50 |  50 | 1.0 s | 128 × 80 mm |  3.38 |  27.03 | FBP + ASiR-V 50% |
| 2 | SE | 120 | 150 | 150 | 1.0 s | 128 × 80 mm | 10.16 |  81.29 | FBP + ASiR-V 50% |
| 3 | SE | 120 | 300 | 300 | 1.0 s | 128 × 80 mm | 20.38 | 163.06 | FBP + ASiR-V 50% |
| 4 | SE |  80 | 480 | 480 | 1.0 s | 128 × 80 mm | 10.32 |  82.58 | FBP + ASiR-V 50% |
| 5 | SE | 100 | 250 | 250 | 1.0 s | 128 × 80 mm | 10.53 |  84.26 | FBP + ASiR-V 50% |
| 6 | SE | 140 | 110 | 110 | 1.0 s | 128 × 80 mm | 10.85 |  86.83 | FBP + ASiR-V 50% |
| 7 | DE (GSI) | 80/140 | 203.5/202.5 | 204 | 0.5 s | 64 × 40 mm | 10.07 | 40.27 | VMI 40/70/100/140 keV all @ 0% (FBP)|
"""

# ╔═╡ 07010005-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ 07010006-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 07010007-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ 07010008-0000-4000-8000-000000000000
import Statistics: mean, std, cor

# ╔═╡ 07010009-0000-4000-8000-000000000000
import Statistics: median

# ╔═╡ 07010010-0000-4000-8000-000000000000
import XrayAttenuation as XA

# ╔═╡ 07010016-0000-4000-8000-000000000000
# ╠═╡ show_logs = false
import DICOM as DCM

# ╔═╡ a13bf90b-b0a1-4786-9554-132b1a346334
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")

# ╔═╡ 07010019-0000-4000-8000-000000000000
const RESULTS_DIR = joinpath(dirname(@__DIR__), "results", "ge_apex_elite"); mkpath(RESULTS_DIR)

# ╔═╡ 07010020-0000-4000-8000-000000000000
UI.TableOfContents()

# ╔═╡ 07020001-0000-4000-8000-000000000000
md"""
## 2. Helper Functions

All measurement and phantom-creation functions defined up front.
"""

# ╔═╡ 07020002-0000-4000-8000-000000000000
"""
    load_hu_volume(dcms) -> Array{Float32, 3}   (rows x cols x slices)

Convert a pre-loaded vector of DICOM datasets into a Float32 HU volume.
Uses ImageMagick.jl (via FileIO) for J2K decompression.
"""
function load_hu_volume(dcms::Vector)
    sort!(dcms; by = d -> d.meta[(0x0020, 0x0013)])
    slope = Float32(dcms[1].meta[(0x0028, 0x1053)])
    intercept = Float32(dcms[1].meta[(0x0028, 0x1052)])
    slices = map(dcms) do dcm
        j2k = dcm.meta[(0x7fe0, 0x0010)][2]
        tmp = tempname() * ".jp2"
        write(tmp, j2k)
        img = FileIO.load(tmp)
        rm(tmp)
        raw_u16 = round.(UInt16, Float32.(img) .* 65535.0f0)
        i16 = Int16.(Int32.(raw_u16) .- Int32(32768))
        Float32.(i16) .* slope .+ intercept
    end
    return cat(slices...; dims = 3)
end

# ╔═╡ 07020003-0000-4000-8000-000000000000
"""
    segment_gammex_rods(hu_slice; fov_cm, ...) -> (mask, rod_info, center_info)

Segment all 16 Gammex 472 insert rods from a 2D CT slice.
Uses known phantom geometry + intensity-based rotation detection.
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
    )
    nx, ny = size(hu_slice)
    pixel_cm = fov_cm / nx

    # Step 1: Find phantom center (table-robust)
    body = hu_slice .> body_threshold_hu
    total = max(Float64(sum(body)), 1.0)
    cx = sum(Float64(i) * body[i, j] for j in 1:ny, i in 1:nx) / total
    cy = sum(Float64(j) * body[i, j] for j in 1:ny, i in 1:nx) / total

    # Iterative refinement: restrict centroid to expected body circle
    body_r_sq = (body_radius_cm / pixel_cm)^2
    for _ in 1:3
        sx, sy, cnt = 0.0, 0.0, 0.0
        for j in 1:ny, i in 1:nx
            if (i - cx)^2 + (j - cy)^2 <= body_r_sq && body[i, j]
                sx += i
                sy += j
                cnt += 1.0
            end
        end
        if cnt > 0
            cx = sx / cnt
            cy = sy / cnt
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
                s += hu_slice[xi, yi]
                c += 1
            end
        end
        profile[k] = c > 0 ? s / c : 0.0
    end

    # Smooth with +/-5 deg circular window
    smooth_w = max(1, round(Int, 5.0 / (360.0 / n_sample)))
    smoothed = similar(profile)
    for k in 1:n_sample
        s, c = 0.0, 0
        for d in -smooth_w:smooth_w
            s += profile[mod1(k + d, n_sample)]
            c += 1
        end
        smoothed[k] = s / c
    end

    # Ca400 is the highest HU rod on the outer ring
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
    outer_start = th_ca400 - 3 * pi / 4
    outer_angles = [outer_start + (i - 1) * pi / 4 for i in 1:8]
    outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]
    outer_names = [
        "Ca 100", "Ca 200", "Ca 300", "Ca 400",
        "Water (O)", "SW ref 1", "SW ref 2", "Ca 50",
    ]

    inner_start = outer_start - pi / 8
    inner_angles = [inner_start + (i - 1) * pi / 4 for i in 1:8]
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

            if length(local_vals) > 10
                med = median(local_vals)
                dev = maximum(abs.(local_vals .- med))
                thresh = dev * 0.3
                wx, wy, wt = 0.0, 0.0, 0.0
                for sj in sj_lo:sj_hi, si in si_lo:si_hi
                    if (si - ecx)^2 + (sj - ecy)^2 <= search_r_pix^2
                        v = Float64(hu_slice[si, sj])
                        if abs(v - med) > thresh
                            wx += si
                            wy += sj
                            wt += 1.0
                        end
                    end
                end
                rcx = wt > 5 ? wx / wt : ecx
                rcy = wt > 5 ? wy / wt : ecy
            else
                rcx, rcy = ecx, ecy
            end

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

# ╔═╡ 07020004-0000-4000-8000-000000000000
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

    # Area-normalized nNPS
    nnps_1d = integrated > 0 ? nps_1d ./ integrated : copy(nps_1d)

    # Moving-average smoothing
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

# ╔═╡ 07020005-0000-4000-8000-000000000000
"""
    measure_mtf_circular_edge(hu_slice, cx, cy, radius_cm; fov_cm, ...)

Circular-edge MTF from a circular boundary.
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
            x0 = floor(Int, xi)
            y0 = floor(Int, yi)
            x1 = x0 + 1
            y1 = y0 + 1
            if 1 <= x0 && x1 <= nx && 1 <= y0 && y1 <= ny
                fx = xi - x0
                fy = yi - y0
                val = (1 - fx) * (1 - fy) * hu_slice[x0, y0] + fx * (1 - fy) * hu_slice[x1, y0] +
                    (1 - fx) * fy * hu_slice[x0, y1] + fx * fy * hu_slice[x1, y1]
                esf[k] += Float64(val)
                counts[k] += 1
            end
        end
    end

    for k in 1:n_r
        if counts[k] > 0
            esf[k] /= counts[k]
        end
    end

    if esf[1] < esf[end]
        reverse!(esf)
        reverse!(positions_mm)
    end

    dp = positions_mm[2] - positions_mm[1]
    lsf = diff(esf) ./ dp
    lsf_pos = (positions_mm[1:(end - 1)] .+ positions_mm[2:end]) ./ 2

    lsf_max = maximum(abs.(lsf))
    if lsf_max > 0
        lsf ./= lsf_max
    end

    n_pad = nextpow(2, length(lsf) * 4)
    lsf_padded = zeros(Float64, n_pad)
    offset = (n_pad - length(lsf)) / 2 |> x -> round(Int, x)
    lsf_padded[(offset + 1):(offset + length(lsf))] .= lsf

    mtf_complex = fft(lsf_padded)
    mtf_vals = abs.(mtf_complex)
    mtf_vals ./= mtf_vals[1]

    n_pos = n_pad / 2 |> Int
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

# ╔═╡ 07020006-0000-4000-8000-000000000000
"""
    measure_scan(hu_vol, seg_mask, seg_rods, seg_center, scan_name; fov_cm)

Compute full measurement suite for one reconstruction.
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
    mid_z = nz / 2 |> x -> round(Int, x)
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

# ╔═╡ 07020007-0000-4000-8000-000000000000
# Build a custom spatial-domain filter kernel from frequency-domain control points.
# Custom filter kernel construction now handled by BS.CustomFilter + BS.create_spatial_kernel
nothing

# ╔═╡ 07020008-0000-4000-8000-000000000000
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

# ╔═╡ 07020009-0000-4000-8000-000000000000
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

# ╔═╡ 07030001-0000-4000-8000-000000000000
md"""
## 3. Single-Energy Scans — ASiR-V 0% (FBP) vs ASiR-V 50%

Six axial acquisitions of the Gammex 472 phantom on the GE Revolution Apex Elite.
Each protocol is loaded once: `_fbp` = ASiR-V 0% (pure FBP), `_ir` = ASiR-V 50%.
DICOM pixel data is JPEG 2000 compressed; decoded via ImageMagick.jl + DICOM.jl.
"""

# ╔═╡ 07030002-0000-4000-8000-000000000000
rootdir = "/Users/daleblack/Desktop/SCANS/02232026 (revolution)"

# ╔═╡ 07030003-0000-4000-8000-000000000000
begin  # 120 kVp / 50 mA / 3.38 mGy
    dcms_120_low_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_50mA_3.38mGyCTDI/0%"))
    dcms_120_low_ir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_50mA_3.38mGyCTDI/50%"))
    hu_120_low_fbp = load_hu_volume(dcms_120_low_fbp)
    hu_120_low_ir = load_hu_volume(dcms_120_low_ir)
end;

# ╔═╡ 07030004-0000-4000-8000-000000000000
begin  # 120 kVp / 150 mA / 10.16 mGy
    dcms_120_mid_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_150mA_10.16mGyCTDI/0%"))
    dcms_120_mid_ir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_150mA_10.16mGyCTDI/50%"))
    hu_120_mid_fbp = load_hu_volume(dcms_120_mid_fbp)
    hu_120_mid_ir = load_hu_volume(dcms_120_mid_ir)
end;

# ╔═╡ 07030005-0000-4000-8000-000000000000
begin  # 120 kVp / 300 mA / 20.38 mGy
    dcms_120_high_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_300mA_20.38mGyCTDI/0%"))
    dcms_120_high_ir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_300mA_20.38mGyCTDI/50%"))
    hu_120_high_fbp = load_hu_volume(dcms_120_high_fbp)
    hu_120_high_ir = load_hu_volume(dcms_120_high_ir)
end;

# ╔═╡ 07030006-0000-4000-8000-000000000000
begin  # 80 kVp / 480 mA / 10.32 mGy
    dcms_80_fbp = DCM.dcmdir_parse(joinpath(rootdir, "80kVp_480mA_10.32mGyCTDI/0%"))
    dcms_80_ir = DCM.dcmdir_parse(joinpath(rootdir, "80kVp_480mA_10.32mGyCTDI/50%"))
    hu_80_fbp = load_hu_volume(dcms_80_fbp)
    hu_80_ir = load_hu_volume(dcms_80_ir)
end;

# ╔═╡ 07030007-0000-4000-8000-000000000000
begin  # 100 kVp / 250 mA / 10.53 mGy
    dcms_100_fbp = DCM.dcmdir_parse(joinpath(rootdir, "100kVp_250mA_10.53mGyCTDI/0%"))
    dcms_100_ir = DCM.dcmdir_parse(joinpath(rootdir, "100kVp_250mA_10.53mGyCTDI/50%"))
    hu_100_fbp = load_hu_volume(dcms_100_fbp)
    hu_100_ir = load_hu_volume(dcms_100_ir)
end;

# ╔═╡ 07030008-0000-4000-8000-000000000000
begin  # 140 kVp / 110 mA / 10.85 mGy
    dcms_140_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_110mA_10.85mGyCTDI/0%"))
    dcms_140_ir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_110mA_10.85mGyCTDI/50%"))
    hu_140_fbp = load_hu_volume(dcms_140_fbp)
    hu_140_ir = load_hu_volume(dcms_140_ir)
end;

# ╔═╡ 07030009-0000-4000-8000-000000000000
let
    mid = size(hu_120_low_fbp, 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 1000))

    for (col, (clim, wname)) in enumerate(windows)
        ax = CM.Axis(
            fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_120_low_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        ax = CM.Axis(
            fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_120_low_ir[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end

    CM.Label(fig[0, :]; text = "120 kVp · 50 mA · CTDIvol 3.38 mGy", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_se_clinical_120kVp_50mA.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07030010-0000-4000-8000-000000000000
let
    mid = size(hu_120_mid_fbp, 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 1000))

    for (col, (clim, wname)) in enumerate(windows)
        ax = CM.Axis(
            fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_120_mid_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        ax = CM.Axis(
            fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_120_mid_ir[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end

    CM.Label(fig[0, :]; text = "120 kVp · 150 mA · CTDIvol 10.16 mGy", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_se_clinical_120kVp_150mA.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07030011-0000-4000-8000-000000000000
let
    mid = size(hu_120_high_fbp, 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 1000))

    for (col, (clim, wname)) in enumerate(windows)
        ax = CM.Axis(
            fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_120_high_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        ax = CM.Axis(
            fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_120_high_ir[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end

    CM.Label(fig[0, :]; text = "120 kVp · 300 mA · CTDIvol 20.38 mGy", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_se_clinical_120kVp_300mA.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07030012-0000-4000-8000-000000000000
let
    mid = size(hu_80_fbp, 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 1000))

    for (col, (clim, wname)) in enumerate(windows)
        ax = CM.Axis(
            fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_80_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        ax = CM.Axis(
            fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_80_ir[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end

    CM.Label(fig[0, :]; text = "80 kVp · 480 mA · CTDIvol 10.32 mGy", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_se_clinical_80kVp_480mA.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07030013-0000-4000-8000-000000000000
let
    mid = size(hu_100_fbp, 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 1000))

    for (col, (clim, wname)) in enumerate(windows)
        ax = CM.Axis(
            fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_100_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        ax = CM.Axis(
            fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_100_ir[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end

    CM.Label(fig[0, :]; text = "100 kVp · 250 mA · CTDIvol 10.53 mGy", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_se_clinical_100kVp_250mA.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07030014-0000-4000-8000-000000000000
let
    mid = size(hu_140_fbp, 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 1000))

    for (col, (clim, wname)) in enumerate(windows)
        ax = CM.Axis(
            fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_140_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)

        ax = CM.Axis(
            fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
            aspect = CM.DataAspect(), yreversed = true
        )
        CM.heatmap!(ax, hu_140_ir[:, :, mid]; colormap = :grays, colorrange = clim)
        CM.hidedecorations!(ax)
        CM.hidespines!(ax)
    end

    CM.Label(fig[0, :]; text = "140 kVp · 110 mA · CTDIvol 10.85 mGy", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_se_clinical_140kVp_110mA.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07040001-0000-4000-8000-000000000000
md"""
## 4. Automatic Rod Segmentation — 120 kVp / ASiR-V 50% (10.16 mGy)

Intensity-based segmentation of all 16 Gammex 472 insert rods.
"""

# ╔═╡ 07040002-0000-4000-8000-000000000000
seg_result = let
    mid_z = size(hu_120_mid_ir, 3) ÷ 2
    hu_slice = hu_120_mid_ir[:, :, mid_z]
    mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm = 35.0)
    (mask = mask, rods = rod_info, center = center, slice_idx = mid_z)
end;

# ╔═╡ 07040003-0000-4000-8000-000000000000
let
    hu = hu_120_mid_ir[:, :, seg_result.slice_idx]
    rods = seg_result.rods
    pixel_cm = 35.0 / size(hu, 1)
    roi_r_pix = 1.4 * 0.7 / pixel_cm

    fig = CM.Figure(size = (1100, 500), fontsize = 11)

    ax1 = CM.Axis(
        fig[1, 1]; title = "120 kVp ASiR-V 50% — Segmented ROIs (slice $(seg_result.slice_idx))",
        aspect = CM.DataAspect(), yreversed = true
    )
    CM.heatmap!(ax1, hu; colormap = :grays, colorrange = (-200, 500))

    th_circle = range(0, 2 * pi, length = 61)
    for r in rods
        xs = r.cx .+ roi_r_pix .* cos.(th_circle)
        ys = r.cy .+ roi_r_pix .* sin.(th_circle)
        c = r.ring == :outer ? :orange : :lime
        CM.lines!(ax1, xs, ys; color = c, linewidth = 1.5)
        CM.text!(
            ax1, r.cx, r.cy + roi_r_pix + 4;
            text = r.name, fontsize = 7, align = (:center, :bottom), color = c
        )
    end
    CM.scatter!(
        ax1, [seg_result.center.cx], [seg_result.center.cy];
        color = :red, marker = :cross, markersize = 12
    )
    CM.hidedecorations!(ax1)
    CM.hidespines!(ax1)

    ax2 = CM.Axis(fig[1, 2]; title = "Rod Label Mask", aspect = CM.DataAspect(), yreversed = true)
    mask_vis = Float32.(seg_result.mask)
    mask_vis[mask_vis .== 0] .= NaN
    CM.heatmap!(ax2, hu; colormap = :grays, colorrange = (-200, 500))
    CM.heatmap!(ax2, mask_vis; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
    CM.hidedecorations!(ax2)
    CM.hidespines!(ax2)

    CM.save(joinpath(RESULTS_DIR, "ge_se_segmentation_rois.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07040004-0000-4000-8000-000000000000
let
    rods = seg_result.rods
    order = [
        "Water (O)", "SW ref 1", "SW ref 2",
        "Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400",
        "Water (I)",
        "I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0",
    ]
    idx = [findfirst(r -> r.name == nm, rods) for nm in order]
    rods_sorted = rods[idx]
    n = length(rods_sorted)

    names = [r.name for r in rods_sorted]
    means = [r.mean_hu for r in rods_sorted]
    stds = [r.std_hu for r in rods_sorted]

    colors = map(rods_sorted) do r
        r.ring == :outer ?
            (startswith(r.name, "Ca") ? :darkorange : :steelblue) :
            (startswith(r.name, "I") ? :forestgreen : :steelblue)
    end

    fig = CM.Figure(size = (1000, 500), fontsize = 11)
    ax = CM.Axis(
        fig[1, 1];
        title = "ROI Measurements — 120 kVp · 150 mA · ASiR-V 50% [rotation=$(seg_result.center.rotation_deg)°]",
        ylabel = "Mean HU ± σ",
        xticks = (1:n, names),
        xticklabelrotation = pi / 4
    )
    CM.barplot!(ax, 1:n, means; color = colors)
    CM.errorbars!(ax, 1:n, means, stds; color = :black, whiskerwidth = 4)

    for (i, (m, s)) in enumerate(zip(means, stds))
        CM.text!(
            ax, i, m + s + 5;
            text = "$(round(Int, m))", fontsize = 8, align = (:center, :bottom)
        )
    end

    CM.save(joinpath(RESULTS_DIR, "ge_se_rod_hu_bars.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07040005-0000-4000-8000-000000000000
se_measurements = let
    se_scans = [
        (hu_120_low_fbp, "120kVp_50mA_FBP"),
        (hu_120_low_ir, "120kVp_50mA_ASiRV50"),
        (hu_120_mid_fbp, "120kVp_150mA_FBP"),
        (hu_120_mid_ir, "120kVp_150mA_ASiRV50"),
        (hu_120_high_fbp, "120kVp_300mA_FBP"),
        (hu_120_high_ir, "120kVp_300mA_ASiRV50"),
        (hu_80_fbp, "80kVp_480mA_FBP"),
        (hu_80_ir, "80kVp_480mA_ASiRV50"),
        (hu_100_fbp, "100kVp_250mA_FBP"),
        (hu_100_ir, "100kVp_250mA_ASiRV50"),
        (hu_140_fbp, "140kVp_110mA_FBP"),
        (hu_140_ir, "140kVp_110mA_ASiRV50"),
    ]

    [
        measure_scan(vol, seg_result.mask, seg_result.rods, seg_result.center, name)
            for (vol, name) in se_scans
    ]
end;

# ╔═╡ 07050001-0000-4000-8000-000000000000
md"""
## 5. Dual-Energy 80/140 kVp — Virtual Monoenergetic Images (VMI)

GE GSI acquisition at 80/140 kVp rapid switching. CTDIvol = 10.07 mGy.
VMI reconstructions at 40, 70, 100, and 140 keV.
"""

# ╔═╡ 07050002-0000-4000-8000-000000000000
hu_de_40keV = let
    dcms = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/40"))
    load_hu_volume(dcms)
end;

# ╔═╡ 07050003-0000-4000-8000-000000000000
hu_de_70keV = let
    dcms = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/70"))
    load_hu_volume(dcms)
end;

# ╔═╡ 07050004-0000-4000-8000-000000000000
hu_de_100keV = let
    dcms = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/100"))
    load_hu_volume(dcms)
end;

# ╔═╡ 07050005-0000-4000-8000-000000000000
hu_de_140keV = let
    dcms = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/140"))
    load_hu_volume(dcms)
end;

# ╔═╡ 07050006-0000-4000-8000-000000000000
let
    vols = [hu_de_40keV, hu_de_70keV, hu_de_100keV, hu_de_140keV]
    labels = ["40 keV", "70 keV", "100 keV", "140 keV"]
    mid = size(vols[1], 3) ÷ 2
    soft = (-160, 240)
    bone = (-350, 1150)
    lung = (-1350, 150)
    windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

    fig = CM.Figure(size = (1500, 2000))

    for (row, (vol, keV_label)) in enumerate(zip(vols, labels))
        for (col, (clim, wname)) in enumerate(windows)
            ax = CM.Axis(
                fig[row, col]; title = "$keV_label — $wname", titlesize = 12,
                aspect = CM.DataAspect(), yreversed = true
            )
            CM.heatmap!(ax, vol[:, :, mid]; colormap = :grays, colorrange = clim)
            CM.hidedecorations!(ax)
            CM.hidespines!(ax)
        end
    end

    CM.Label(fig[0, :]; text = "DE 80/140 kVp — VMI (CTDIvol 10.07 mGy)", fontsize = 16, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "ge_de_clinical_vmi_grid.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07050007-0000-4000-8000-000000000000
md"""
### 5b. Automatic Rod Segmentation — DE 100 keV VMI
"""

# ╔═╡ 07050008-0000-4000-8000-000000000000
seg_result_de = let
    mid_z = size(hu_de_100keV, 3) ÷ 2
    hu_slice = hu_de_100keV[:, :, mid_z]
    mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm = 35.0)
    (mask = mask, rods = rod_info, center = center, slice_idx = mid_z)
end;

# ╔═╡ 07050009-0000-4000-8000-000000000000
let
    hu = hu_de_100keV[:, :, seg_result_de.slice_idx]
    rods = seg_result_de.rods
    pixel_cm = 35.0 / size(hu, 1)
    roi_r_pix = 1.4 * 0.7 / pixel_cm

    fig = CM.Figure(size = (1100, 500), fontsize = 11)

    ax1 = CM.Axis(
        fig[1, 1]; title = "DE 100 keV VMI — Segmented ROIs (slice $(seg_result_de.slice_idx))",
        aspect = CM.DataAspect(), yreversed = true
    )
    CM.heatmap!(ax1, hu; colormap = :grays, colorrange = (-200, 500))

    th_circle = range(0, 2 * pi, length = 61)
    for r in rods
        xs = r.cx .+ roi_r_pix .* cos.(th_circle)
        ys = r.cy .+ roi_r_pix .* sin.(th_circle)
        c = r.ring == :outer ? :orange : :lime
        CM.lines!(ax1, xs, ys; color = c, linewidth = 1.5)
        CM.text!(
            ax1, r.cx, r.cy + roi_r_pix + 4;
            text = r.name, fontsize = 7, align = (:center, :bottom), color = c
        )
    end
    CM.scatter!(
        ax1, [seg_result_de.center.cx], [seg_result_de.center.cy];
        color = :red, marker = :cross, markersize = 12
    )
    CM.hidedecorations!(ax1)
    CM.hidespines!(ax1)

    ax2 = CM.Axis(fig[1, 2]; title = "Rod Label Mask", aspect = CM.DataAspect(), yreversed = true)
    mask_vis = Float32.(seg_result_de.mask)
    mask_vis[mask_vis .== 0] .= NaN
    CM.heatmap!(ax2, hu; colormap = :grays, colorrange = (-200, 500))
    CM.heatmap!(ax2, mask_vis; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
    CM.hidedecorations!(ax2)
    CM.hidespines!(ax2)

    CM.save(joinpath(RESULTS_DIR, "ge_de_segmentation_rois.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07050010-0000-4000-8000-000000000000
let
    rods = seg_result_de.rods
    order = [
        "Water (O)", "SW ref 1", "SW ref 2",
        "Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400",
        "Water (I)",
        "I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0",
    ]
    idx = [findfirst(r -> r.name == nm, rods) for nm in order]
    rods_sorted = rods[idx]
    n = length(rods_sorted)

    names = [r.name for r in rods_sorted]
    means = [r.mean_hu for r in rods_sorted]
    stds = [r.std_hu for r in rods_sorted]

    colors = map(rods_sorted) do r
        r.ring == :outer ?
            (startswith(r.name, "Ca") ? :darkorange : :steelblue) :
            (startswith(r.name, "I") ? :forestgreen : :steelblue)
    end

    fig = CM.Figure(size = (1000, 500), fontsize = 11)
    ax = CM.Axis(
        fig[1, 1];
        title = "ROI Measurements — DE 100 keV VMI [rotation=$(seg_result_de.center.rotation_deg)°]",
        ylabel = "Mean HU ± σ",
        xticks = (1:n, names),
        xticklabelrotation = pi / 4
    )
    CM.barplot!(ax, 1:n, means; color = colors)
    CM.errorbars!(ax, 1:n, means, stds; color = :black, whiskerwidth = 4)

    for (i, (m, s)) in enumerate(zip(means, stds))
        CM.text!(
            ax, i, m + s + 5;
            text = "$(round(Int, m))", fontsize = 8, align = (:center, :bottom)
        )
    end

    CM.save(joinpath(RESULTS_DIR, "ge_de_rod_hu_bars.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07050011-0000-4000-8000-000000000000
de_measurements = let
    de_scans = [
        (hu_de_40keV, "DE_40keV"),
        (hu_de_70keV, "DE_70keV"),
        (hu_de_100keV, "DE_100keV"),
        (hu_de_140keV, "DE_140keV"),
    ]

    [
        measure_scan(vol, seg_result_de.mask, seg_result_de.rods, seg_result_de.center, name)
            for (vol, name) in de_scans
    ]
end;

# ╔═╡ 07060001-0000-4000-8000-000000000000
md"""
## 6. Quantitative Measurements — NPS, MTF, Export
"""

# ╔═╡ 07060002-0000-4000-8000-000000000000
let
    fig = CM.Figure(size = (1200, 500), fontsize = 11)
    ax1 = CM.Axis(
        fig[1, 1]; title = "nNPS — Single-Energy Scans", xlabel = "Spatial frequency (lp/cm)", ylabel = "nNPS (A.U.)"
    )
    ax2 = CM.Axis(
        fig[1, 2]; title = "nNPS — Dual-Energy VMI",
        xlabel = "Spatial frequency (lp/cm)", ylabel = "nNPS (A.U.)"
    )
    se_colors = CM.cgrad(:tab10, 12, categorical = true)
    for (i, m) in enumerate(se_measurements)
        freqs = m.nps.frequencies
        vals = m.nps.nps_1d
        good = vals .> 0
        CM.lines!(ax1, m.nps.frequencies, m.nps.nnps_1d; label = m.name, color = se_colors[i])
    end
    CM.axislegend(ax1; position = :rt, labelsize = 8, nbanks = 2)

    de_colors = [:purple, :teal, :darkorange, :crimson]
    for (i, m) in enumerate(de_measurements)
        freqs = m.nps.frequencies
        vals = m.nps.nps_1d
        good = vals .> 0
        CM.lines!(ax2, m.nps.frequencies, m.nps.nnps_1d; label = m.name, color = de_colors[i], linewidth = 2)
    end
    CM.axislegend(ax2; position = :rt, labelsize = 10)

    CM.save(joinpath(RESULTS_DIR, "ge_clinical_nps.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07060003-0000-4000-8000-000000000000
let
    fig = CM.Figure(size = (1200, 500), fontsize = 11)
    ax1 = CM.Axis(
        fig[1, 1]; title = "MTF — Single-Energy Scans",
        xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
        limits = (nothing, nothing, 0, 1.05)
    )
    ax2 = CM.Axis(
        fig[1, 2]; title = "MTF — Dual-Energy VMI",
        xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
        limits = (nothing, nothing, 0, 1.05)
    )

    CM.hlines!(ax1, [0.5, 0.1]; color = :gray80, linestyle = :dash, linewidth = 0.8)
    CM.hlines!(ax2, [0.5, 0.1]; color = :gray80, linestyle = :dash, linewidth = 0.8)

    se_colors = CM.cgrad(:tab10, 12, categorical = true)
    for (i, m) in enumerate(se_measurements)
        CM.lines!(ax1, m.mtf.frequencies, m.mtf.mtf; label = m.name, color = se_colors[i])
    end
    CM.axislegend(ax1; position = :rt, labelsize = 8, nbanks = 2)

    de_colors = [:purple, :teal, :darkorange, :crimson]
    for (i, m) in enumerate(de_measurements)
        CM.lines!(ax2, m.mtf.frequencies, m.mtf.mtf; label = m.name, color = de_colors[i], linewidth = 2)
    end
    CM.axislegend(ax2; position = :rt, labelsize = 10)

    CM.save(joinpath(RESULTS_DIR, "ge_clinical_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07060004-0000-4000-8000-000000000000
let
    all_m = vcat(se_measurements, de_measurements)
    outdir = joinpath(dirname(@__DIR__), "results")
    mkpath(outdir)

    rod_order = all_m[1].rod_names
    header = ["scan_name"]
    for nm in rod_order
        tag = replace(replace(nm, " " => "_"), "(" => "", ")" => "")
        push!(header, "hu_mean_$tag")
    end
    for nm in rod_order
        tag = replace(replace(nm, " " => "_"), "(" => "", ")" => "")
        push!(header, "hu_std_$tag")
    end
    for nm in rod_order
        tag = replace(replace(nm, " " => "_"), "(" => "", ")" => "")
        push!(header, "cnr_$tag")
    end
    append!(
        header, [
            "nps_peak_freq_lp_cm", "nps_area_HU2cm2",
            "mtf_f50_lp_cm", "mtf_f10_lp_cm",
        ]
    )

    rows = Vector{Any}[]
    for m in all_m
        row = Any[m.name]
        append!(row, round.(m.rod_means, digits = 2))
        append!(row, round.(m.rod_stds, digits = 2))
        append!(row, round.(m.rod_cnr, digits = 2))
        push!(row, round(m.nps_peak_freq, digits = 3))
        push!(row, round(m.nps_area, digits = 3))
        push!(row, round(m.mtf_f50, digits = 3))
        push!(row, round(m.mtf_f10, digits = 3))
        push!(rows, row)
    end

    csv_path = joinpath(outdir, "clinical_measurements.csv")
    open(csv_path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end

    nps_path = joinpath(outdir, "nps_curves.jld2")
    JLD2.jldopen(nps_path, "w") do f
        for m in all_m
            f[m.name] = (freq = m.nps.frequencies, nps = m.nps.nps_1d)
        end
    end

    mtf_path = joinpath(outdir, "mtf_curves.jld2")
    JLD2.jldopen(mtf_path, "w") do f
        for m in all_m
            f[m.name] = (freq = m.mtf.frequencies, mtf = m.mtf.mtf)
        end
    end

    md"""
    **Exported:**
    - `results/clinical_measurements.csv` — $(length(all_m)) rows × $(length(header)) columns
    - `results/nps_curves.jld2` — radial NPS curves
    - `results/mtf_curves.jld2` — radial MTF curves
    """
end

# ╔═╡ 07060005-0000-4000-8000-000000000000
md"""
### 6b. Scalar Summary
"""

# ╔═╡ 07060006-0000-4000-8000-000000000000
let
    all_m = vcat(se_measurements, de_measurements)
    n = length(all_m)

    fig = CM.Figure(size = (1400, 400), fontsize = 10)

    ax1 = CM.Axis(
        fig[1, 1]; title = "NPS Area (Noise Variance)",
        ylabel = "NPS integral (HU²·cm²)",
        xticks = (1:n, [m.name for m in all_m]),
        xticklabelrotation = pi / 3
    )
    CM.barplot!(
        ax1, 1:n, [m.nps_area for m in all_m];
        color = vcat(
            fill(:steelblue, length(se_measurements)),
            fill(:darkorange, length(de_measurements))
        )
    )

    ax2 = CM.Axis(
        fig[1, 2]; title = "MTF f50 and f10",
        ylabel = "Frequency (lp/cm)",
        xticks = (1:n, [m.name for m in all_m]),
        xticklabelrotation = pi / 3
    )
    CM.barplot!(
        ax2, collect(1:n) .- 0.15, [m.mtf_f50 for m in all_m];
        width = 0.3, color = :teal, label = "f50"
    )
    CM.barplot!(
        ax2, collect(1:n) .+ 0.15, [m.mtf_f10 for m in all_m];
        width = 0.3, color = :salmon, label = "f10"
    )
    CM.axislegend(ax2; position = :rt)

    CM.save(joinpath(RESULTS_DIR, "ge_clinical_scalar_summary.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07070001-0000-4000-8000-000000000000
md"""
## 7. Digital Phantom — Gammex 472 Replica

High-resolution (0.2mm isotropic) digital phantom for forward projection.
Body: 330 mm diameter Gammex Model 451 Solid Water.
Outer ring (R=10.5 cm): Ca50-Ca400, Water, SW ref.
Inner ring (R=5.5 cm): I2.0-I20.0, Water.
"""

# ╔═╡ 07070002-0000-4000-8000-000000000000
begin
    sim_phantom = create_custom_gammex_472(
        n_voxels = 1750,
        n_slices = 250,
        fov_cm = 35.0,
        z_cm = 5.0,
    )

    sim_phantom_gpu = BS.Phantom(
        MtlArray(sim_phantom.mask),
        sim_phantom.materials,
        sim_phantom.voxel_size,
        sim_phantom.origin,
        sim_phantom.extent,
    )
end;

# ╔═╡ 07070003-0000-4000-8000-000000000000
let
    mid = size(sim_phantom.mask, 3) ÷ 2
    nz = size(sim_phantom.mask, 3)
    slice_data = sim_phantom.mask[:, :, mid]

    unique_labels = sort(unique(slice_data))
    n_labels = length(unique_labels)

    lut = zeros(Float32, 27)
    for (i, l) in enumerate(unique_labels)
        lut[Int(l) + 1] = Float32(i)
    end
    mapped = lut[Int.(slice_data) .+ 1]

    colors = [MATERIAL_INFO[l].color for l in unique_labels]
    cmap = CM.cgrad(colors, n_labels, categorical = true)
    names = [MATERIAL_INFO[l].name for l in unique_labels]

    fig = CM.Figure(size = (1000, 850), fontsize = 12)
    ax = CM.Axis(
        fig[1, 1];
        title = "Custom Gammex 472 — Slice $mid / $nz (0.2mm voxels)",
        aspect = CM.DataAspect()
    )
    hm = CM.heatmap!(ax, mapped; colormap = cmap, colorrange = (0.5, n_labels + 0.5))
    CM.Colorbar(fig[1, 2], hm; ticks = (1:n_labels, names), ticklabelsize = 11, width = 15)
    CM.save(joinpath(RESULTS_DIR, "ge_phantom_gammex_472.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07080001-0000-4000-8000-000000000000
md"""
## 8. Simulation Parameters [TUNE: PHYSICS]

### ⚙️ Re-run `simulate!()` after changing any parameter in this section.

These parameters control the forward projection physics model. Changing them
requires re-running all 6 sinogram simulations (§9).
"""

# ╔═╡ 07080002-0000-4000-8000-000000000000
# DAS electronic noise floor (e⁻ std dev). Set to 0 for quantum-only noise.
# CatSim default: 3500, BasisSimulator default: 5000
sim_electronic_noise = 0

# ╔═╡ 07080003-0000-4000-8000-000000000000
# Scintillator + photodiode gain (e⁻/keV). Higher → smaller photon-equivalent noise.
# CatSim default: 17, BasisSimulator default: 15
sim_detection_gain = 10.0

# ╔═╡ 07080004-0000-4000-8000-000000000000
# Extra Al filtration beyond the scanner's built-in 2.5mm.
# GE Revolution Apex has ~7-8mm Al equivalent inherent filtration total.
# Scanner applies 2.5mm in spectrum domain; we add 4.5mm more → ~7mm total.
additional_filters = [("Al", 4.5)]

# ╔═╡ 07080005-0000-4000-8000-000000000000
sim_scanner = BS.Scanner(
    source_to_isocenter = 625.6,
    source_to_detector = 1100.0,
    detector_rows = 256,
    detector_cols = 834,
    detector_row_size = 0.625,
    detector_col_size = 0.6,
    detector_shape = BS.CURVED_DETECTOR,
    focal_spot_width = 1.0,
    focal_spot_length = 1.0,
    target_angle = 10.0,
    flat_filter_material = :aluminum,
    flat_filter_thickness = 2.5,
    bowtie_filter = :ge_revolution_large,
    detector_material = :lumex,
    detector_depth = 3.0,
    fill_factor_row = 0.9,
    fill_factor_col = 0.9,
    electronic_noise = sim_electronic_noise,
    detection_gain = sim_detection_gain,
)

# ╔═╡ 07080006-0000-4000-8000-000000000000
md"""
### Noise Model
- **Quantum noise:** Poisson statistics in intensity domain, scaled by I₀ × η\_eff
- **Electronic noise:** Gaussian additive in intensity domain, σ\_e = electronic\_noise / (mean\_E × detection\_gain)
- **MC LUT (Lumex):** η(E) folded into I₀\_eff — orthogonal to DAS noise

**Tunable parameters** (in cells above):
- `sim_electronic_noise` — DAS readout noise floor (e⁻). Set to 0 for quantum-only.
- `sim_detection_gain` — scintillator gain (e⁻/keV). Higher gain → smaller photon-equivalent noise.
"""

# ╔═╡ 07080007-0000-4000-8000-000000000000
sim_opts = BS.SimOptions(
    fidelity = :eict,
    seed = 1234,
)

# ╔═╡ 07080008-0000-4000-8000-000000000000
begin
    sim_rotation_time = 1.0     # seconds (clinical 1.0s rotation)
    sim_collimation_mm = 5.0    # 8 × 0.625mm — fast single-slice dev mode
    sim_n_views = 984           # standard GE Revolution
end

# ╔═╡ 07080009-0000-4000-8000-000000000000
SE_SIM_SCANS = [
    (name = "120kVp_50mA_FBP", kvp = 120, mA = 50.0),
    (name = "120kVp_150mA_FBP", kvp = 120, mA = 150.0),
    (name = "120kVp_300mA_FBP", kvp = 120, mA = 300.0),
    (name = "80kVp_480mA_FBP", kvp = 80, mA = 480.0),
    (name = "100kVp_250mA_FBP", kvp = 100, mA = 250.0),
    (name = "140kVp_110mA_FBP", kvp = 140, mA = 110.0),
]

# ╔═╡ 07080010-0000-4000-8000-000000000000
begin
    sim_recon_xy = 512
    sim_recon_fov_cm = 35.0
    sim_slice_thickness_mm = 0.625
    sim_recon_z_cm = sim_collimation_mm / 10.0
    sim_n_recon_slices = round(Int, sim_collimation_mm / sim_slice_thickness_mm)
    sim_matrix_size = (sim_recon_xy, sim_recon_xy, sim_n_recon_slices)

    sim_recon_geom = BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = sim_matrix_size,
        fov_cm = sim_recon_fov_cm,
        z_cm = sim_recon_z_cm,
    )
end

# ╔═╡ 07090001-0000-4000-8000-000000000000
md"""
## 9. SE Sinogram Simulation

Forward-project the digital Gammex 472 phantom through the GE Revolution model.
Each scan matches clinical kVp, mA, rotation time, and collimation.
"""

# ╔═╡ 07090002-0000-4000-8000-000000000000
# Scan 1: 120 kVp / 50 mA — SIMULATE (GPU → sinogram → free GPU)
sim_sino_1 = let
    sc = SE_SIM_SCANS[1]
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = additional_filters,
    )
    @info "Simulating: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner, prot, sim_opts, sim_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 07090003-0000-4000-8000-000000000000
# Scan 2: 120 kVp / 150 mA — SIMULATE (GPU → sinogram → free GPU)
sim_sino_2 = let
    sc = SE_SIM_SCANS[2]
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = additional_filters,
    )
    opts = sim_opts
    @info "Simulating: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner, prot, opts, sim_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, opts, sim_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 07090004-0000-4000-8000-000000000000
# Scan 3: 120 kVp / 300 mA — SIMULATE (GPU → sinogram → free GPU)
sim_sino_3 = let
    sc = SE_SIM_SCANS[3]
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = additional_filters,
    )
    @info "Simulating: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner, prot, sim_opts, sim_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 07090005-0000-4000-8000-000000000000
# Scan 4: 80 kVp / 480 mA — SIMULATE (GPU → sinogram → free GPU)
sim_sino_4 = let
    sc = SE_SIM_SCANS[4]
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = additional_filters,
    )
    @info "Simulating: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner, prot, sim_opts, sim_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 07090006-0000-4000-8000-000000000000
# Scan 5: 100 kVp / 250 mA — SIMULATE (GPU → sinogram → free GPU)
sim_sino_5 = let
    sc = SE_SIM_SCANS[5]
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = additional_filters,
    )
    @info "Simulating: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner, prot, sim_opts, sim_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 07090007-0000-4000-8000-000000000000
# Scan 6: 140 kVp / 110 mA — SIMULATE (GPU → sinogram → free GPU)
sim_sino_6 = let
    sc = SE_SIM_SCANS[6]
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = sim_n_views,
        rotation_time = sim_rotation_time,
        collimation_mm = sim_collimation_mm,
        additional_filters = additional_filters,
    )
    @info "Simulating: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner, prot, sim_opts, sim_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 07100001-0000-4000-8000-000000000000
md"""
## 10. Reconstruction Parameters [TUNE: RECON]

### 🎛️ No re-simulation needed — changes only re-run FDK/HIR + HU conversion.

These parameters control reconstruction filtering, noise floor, and beam hardening
correction. Changing them does NOT re-trigger `simulate!()`.
"""

# ╔═╡ 07100002-0000-4000-8000-000000000000
# Filter type now set directly via BS.CustomFilter in recon cells
nothing

# ╔═╡ 07100003-0000-4000-8000-000000000000
# Custom filter apodization — tune these control points to match GE STANDARD kernel MTF.
# Format: (f_norm, weight) where f_norm ∈ [0,1] (fraction of Nyquist).
custom_filter_control = (
    x = (0.0, 0.25, 0.5, 0.75, 1.0),
    y = (1.0, 0.85, 0.6, 0.15, 0.001),
)

# ╔═╡ 07100004-0000-4000-8000-000000000000
# Dose-independent noise floor (σ HU) — tune to match clinical high-mA noise.
# σ_total = √(σ_quantum² + σ_floor²). Start at 0, increase until 300mA matches.
sim_noise_floor_hu = 28.0

# ╔═╡ 07100005-0000-4000-8000-000000000000
# Two-material BHC toggle — tweak freely (does NOT re-run simulate!)
# When enabled, applies projection-domain water+bone BHC to noisy sinogram before FDK.
# When disabled, raw polychromatic sinogram goes straight to FDK (no BHC at all).
bhc_enabled = true

# ╔═╡ 07100006-0000-4000-8000-000000000000
begin
    # Sinogram-domain BHC parameters (Martinez/Fessler 2022)
    sino_bhc_hu_low = 450.0
    sino_bhc_hu_high = 600.0
    sino_order = 2
end

# ╔═╡ 07100007-0000-4000-8000-000000000000
begin
    # Image-domain BHC parameters (So et al. 2009) — tune freely, no re-sim needed.
    # hu_low/hu_high: HU threshold range for high-attenuation segmentation
    # scale_factor: strength of error image subtraction (0.0–1.0, typical 0.2–0.7)
    img_bhc_hu_low = 50.0
    img_bhc_hu_high = 150.0
    img_bhc_scale_factor = 0.2
end

# ╔═╡ 07100008-0000-4000-8000-000000000000
md"""
### Beam Hardening Correction (BHC)

#### Stage 1 — Sinogram Domain (Two-Material Decomposition)
Based on *Elbakri & Fessler (2003)*:
1. Apply water-only polynomial BHC to sinogram, reconstruct preliminary image.
2. Segment bone via smoothstep (`sino_bhc_hu_low`–`sino_bhc_hu_high`), forward-project bone-only image.
3. Decompose each ray into water path Lw and bone path Lb.
4. Compute polychromatic two-material line integral and correct raw sinogram to monochromatic equivalent.

#### Stage 2 — Image Domain (Residual Correction)
Based on *So et al. (2009, PMB 54:3031)*:
1. Reconstruct from Stage 1 corrected sinogram.
2. Segment residual high-attenuation voxels via smoothstep (`img_bhc_hu_low`–`img_bhc_hu_high`).
3. Forward-project weighted high-attenuation image, reconstruct error image.
4. Subtract scaled error image using `img_bhc_scale_factor` (typically 0.2–0.7).

Toggle `bhc_enabled` to enable or disable.
"""

# ╔═╡ 07100009-0000-4000-8000-000000000000
# Calibrate two-material (water + bone) BHC models (one per kVp).
bhc_models, bhc_μ_water = let
    models = Dict{Int, BS.TwoMaterialBHC}()
    μw = Dict{Int, Float64}()
    for kvp in [120, 80, 100, 140]
        prot = BS.CTProtocol(kVp = kvp, additional_filters = additional_filters)
        e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
        ref_E = sum(e .* w) / sum(w)
        models[kvp] = BS.calibrate_bhc_two_material(
            e, w;
            order = sino_order,
            reference_energy_keV = ref_E,
            hu_low = sino_bhc_hu_low,
            hu_high = sino_bhc_hu_high
        )
        μw[kvp] = models[kvp].μ_water_ref
        @info "BHC $(kvp) kVp: ref_E=$(round(ref_E, digits = 1)) keV, μ_water=$(round(μw[kvp], digits = 5))"
    end
    models, μw
end

# ╔═╡ d5a0ffa5-1d40-4bfc-89de-e462c5358b24
# Radial cupping/capping correction now in BS.apply_radial_cupping_correction!
nothing

# ╔═╡ 07110001-0000-4000-8000-000000000000
md"""
## 11. SE FBP Reconstruction & Comparison

FDK reconstruction with custom filter kernel and two-stage BHC.
Each scan uses the correct kVp-specific BHC model.
"""

# ╔═╡ 07110002-0000-4000-8000-000000000000
# Scan 1: 120 kVp / 50 mA — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_1 = let
    sino_gpu = MtlArray(sim_sino_1.sino)
    geom = sim_sino_1.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[120], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[120];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_sino_1.name, kvp = sim_sino_1.kvp)
end

# ╔═╡ 07110003-0000-4000-8000-000000000000
# Scan 1: 120 kVp / 50 mA — HU CONVERSION (CPU only)
sim_hu_1 = let
    μ_w = bhc_μ_water[120]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_1.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_1.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 07110004-0000-4000-8000-000000000000
# Scan 2: 120 kVp / 150 mA — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_2 = let
    sino_gpu = MtlArray(sim_sino_2.sino)
    geom = sim_sino_2.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[120], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[120];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_sino_2.name, kvp = sim_sino_2.kvp)
end

# ╔═╡ 07110005-0000-4000-8000-000000000000
# Scan 2: 120 kVp / 150 mA — HU CONVERSION (CPU only)
sim_hu_2 = let
    μ_w = bhc_μ_water[120]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_2.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_2.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 07110006-0000-4000-8000-000000000000
# Scan 3: 120 kVp / 300 mA — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_3 = let
    sino_gpu = MtlArray(sim_sino_3.sino)
    geom = sim_sino_3.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[120], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[120];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_sino_3.name, kvp = sim_sino_3.kvp)
end

# ╔═╡ 07110007-0000-4000-8000-000000000000
# Scan 3: 120 kVp / 300 mA — HU CONVERSION (CPU only)
sim_hu_3 = let
    μ_w = bhc_μ_water[120]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_3.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_3.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 07110008-0000-4000-8000-000000000000
# Scan 4: 80 kVp / 480 mA — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_4 = let
    sino_gpu = MtlArray(sim_sino_4.sino)
    geom = sim_sino_4.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[80], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[80];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_sino_4.name, kvp = sim_sino_4.kvp)
end

# ╔═╡ 07110009-0000-4000-8000-000000000000
# Scan 4: 80 kVp / 480 mA — HU CONVERSION (CPU only)
sim_hu_4 = let
    μ_w = bhc_μ_water[80]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_4.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_4.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 07110010-0000-4000-8000-000000000000
# Scan 5: 100 kVp / 250 mA — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_5 = let
    sino_gpu = MtlArray(sim_sino_5.sino)
    geom = sim_sino_5.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[100], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[100];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_sino_5.name, kvp = sim_sino_5.kvp)
end

# ╔═╡ 07110011-0000-4000-8000-000000000000
# Scan 5: 100 kVp / 250 mA — HU CONVERSION (CPU only)
sim_hu_5 = let
    μ_w = bhc_μ_water[100]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_5.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_5.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 07110012-0000-4000-8000-000000000000
# Scan 6: 140 kVp / 110 mA — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_6 = let
    sino_gpu = MtlArray(sim_sino_6.sino)
    geom = sim_sino_6.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[140], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[140];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_sino_6.name, kvp = sim_sino_6.kvp)
end

# ╔═╡ 07110013-0000-4000-8000-000000000000
# Scan 6: 140 kVp / 110 mA — HU CONVERSION (CPU only)
sim_hu_6 = let
    μ_w = bhc_μ_water[140]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_6.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_6.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 07110020-0000-4000-8000-000000000000
sim_results_fbp = [sim_hu_1, sim_hu_2, sim_hu_3, sim_hu_4, sim_hu_5, sim_hu_6]

# ╔═╡ 07110021-0000-4000-8000-000000000000
sim_oriented = let
    orient = identity
    orient = (f -> s -> reverse(f(s), dims = 2))(orient)  # Flip Left-Right
    [
        (
                name = r.name,
                recon = Float32.(mapslices(orient, r.recon, dims = (1, 2))),
                mu_water = r.mu_water,
            )
            for r in sim_results_fbp
    ]
end

# ╔═╡ 07110022-0000-4000-8000-000000000000
# Segment the simulated recon independently (finds center + Ca400 anchor in sim image)
sim_seg_result = let
    ref = sim_oriented[min(2, end)].recon
    mid_z = 3
    mask, rods, center = segment_gammex_rods(ref[:, :, mid_z]; fov_cm = 35.0)
    (mask = mask, rods = rods, center = center, slice_idx = mid_z)
end;

# ╔═╡ 07110023-0000-4000-8000-000000000000
sim_measurements = [
    measure_scan(r.recon, sim_seg_result.mask, sim_seg_result.rods, sim_seg_result.center, "sim_$(r.name)")
        for r in sim_oriented
];

# ╔═╡ 07110030-0000-4000-8000-000000000000
md"""
### FBP: Qualitative Side-by-Side
"""

# ╔═╡ 07110031-0000-4000-8000-000000000000
# FBP: 2×3 qualitative side-by-side (Clinical FBP vs Simulated FBP)
let
    clin_vols = [
        hu_120_low_fbp, hu_120_mid_fbp, hu_120_high_fbp,
        hu_80_fbp, hu_100_fbp, hu_140_fbp,
    ]
    scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n = min(length(clin_vols), length(sim_oriented))
    fig = CM.Figure(size = (800, n * 400), fontsize = 10)

    for i in 1:n
        clin_slice = clin_vols[i][:, :, seg_result.slice_idx]
        sim_slice = sim_oriented[i].recon[:, :, sim_seg_result.slice_idx]

        ax1 = CM.Axis(fig[i, 1]; title = "Clinical — $(scan_labels[i])", yreversed = true)
        CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax1); CM.hidespines!(ax1)

        ax2 = CM.Axis(fig[i, 2]; title = "Simulated FBP — $(scan_labels[i])", yreversed = true)
        CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax2); CM.hidespines!(ax2)
    end

    CM.save(joinpath(RESULTS_DIR, "ge_fbp_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 8e294031-bcdb-4424-8936-a62a802d5aa1
md"""
### FBP: Line Profiles (Clinical vs Simulated)
"""

# ╔═╡ 0403ce0e-1854-40e2-af6d-c53125aca803
# Line profiles through center — 120 kVp / 150 mA and 80 kVp / 480 mA
let
    fig = CM.Figure(size = (1000, 700), fontsize = 11)

    # --- Top: 120 kVp / 150 mA ---
    clin_slice_1 = hu_120_mid_fbp[:, :, seg_result.slice_idx]
    sim_slice_1 = sim_oriented[2].recon[:, :, sim_seg_result.slice_idx]
    mid_row_c1 = size(clin_slice_1, 1) ÷ 2
    mid_row_s1 = size(sim_slice_1, 1) ÷ 2

    sim_shift_px = 4.0

    pixel_mm_c1 = 350.0 / size(clin_slice_1, 2)
    pixel_mm_s1 = 350.0 / size(sim_slice_1, 2)
    x_c1 = range(0, step = pixel_mm_c1, length = size(clin_slice_1, 2))
    x_s1 = range(0, step = pixel_mm_s1, length = size(sim_slice_1, 2))
    x_s1 = range(sim_shift_px * pixel_mm_s1, step = pixel_mm_s1, length = size(sim_slice_1, 2))

    ax1 = CM.Axis(fig[1, 1]; title = "120 kVp / 150 mA — Horizontal Line Profile (mid-row)", xlabel = "Position (mm)", ylabel = "HU")
    CM.lines!(
        ax1, collect(x_c1), Float64.(clin_slice_1[mid_row_c1, :]);
        color = :steelblue, linewidth = 1.2, label = "Clinical FBP"
    )
    CM.lines!(
        ax1, collect(x_s1), Float64.(sim_slice_1[mid_row_s1, :]);
        color = :orangered, linewidth = 1.2, label = "Simulated FBP"
    )
    CM.hlines!(ax1, [0.0]; color = :gray70, linestyle = :dash, linewidth = 0.6)
    CM.axislegend(ax1; position = :rt, labelsize = 9)

    CM.ylims!(ax1, low = -600)

    # --- Bottom: 80 kVp / 480 mA ---
    clin_slice_2 = hu_80_fbp[:, :, seg_result.slice_idx]
    sim_slice_2 = sim_oriented[4].recon[:, :, sim_seg_result.slice_idx]
    mid_row_c2 = size(clin_slice_2, 1) ÷ 2
    mid_row_s2 = size(sim_slice_2, 1) ÷ 2

    pixel_mm_c2 = 350.0 / size(clin_slice_2, 2)
    pixel_mm_s2 = 350.0 / size(sim_slice_2, 2)
    x_c2 = range(0, step = pixel_mm_c2, length = size(clin_slice_2, 2))
    x_s2 = range(0, step = pixel_mm_s2, length = size(sim_slice_2, 2))
    x_s2 = range(sim_shift_px * pixel_mm_s2, step = pixel_mm_s2, length = size(sim_slice_2, 2))

    ax2 = CM.Axis(
        fig[2, 1]; title = "80 kVp / 480 mA — Horizontal Line Profile (mid-row)",
        xlabel = "Position (mm)", ylabel = "HU"
    )
    CM.lines!(
        ax2, collect(x_c2), Float64.(clin_slice_2[mid_row_c2, :]);
        color = :steelblue, linewidth = 1.2, label = "Clinical FBP"
    )
    CM.lines!(
        ax2, collect(x_s2), Float64.(sim_slice_2[mid_row_s2, :]);
        color = :orangered, linewidth = 1.2, label = "Simulated FBP"
    )
    CM.hlines!(ax2, [0.0]; color = :gray70, linestyle = :dash, linewidth = 0.6)
    CM.axislegend(ax2; position = :rt, labelsize = 9)
    CM.ylims!(ax2, low = -600)

    CM.save(joinpath(RESULTS_DIR, "ge_fbp_line_profiles.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07110032-0000-4000-8000-000000000000
md"""
### FBP: Scatter Plot (HU)
"""

# ╔═╡ 07110033-0000-4000-8000-000000000000
# FBP: Ca/I scatter — Clinical FBP vs Simulated FBP
let
    base_clinical_idx = [1, 3, 5, 7, 9, 11]
    base_scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n_sims = min(length(sim_measurements), length(base_clinical_idx))
    active_clinical_idx = base_clinical_idx[1:n_sims]
    active_labels = base_scan_labels[1:n_sims]
    colors = CM.cgrad(:tab10, max(n_sims, 2), categorical = true)

    fig = CM.Figure(size = (750, 900), fontsize = 11)

    # --- Top: Calcium rods ---
    ax_ca = CM.Axis(
        fig[1, 1], title = "Calcium Rods", subtitle = "Clinical vs Simulated",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    ca_clin_all, ca_sim_all = Float64[], Float64[]

    for (k, (ci, sm_k)) in enumerate(zip(active_clinical_idx, sim_measurements[1:n_sims]))
        cm = se_measurements[ci]
        ca_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "Ca")]
        if !isempty(ca_idx)
            CM.scatter!(
                ax_ca, cm.rod_means[ca_idx], sm_k.rod_means[ca_idx];
                color = colors[k], markersize = 10, label = active_labels[k]
            )
            append!(ca_clin_all, cm.rod_means[ca_idx])
            append!(ca_sim_all, sm_k.rod_means[ca_idx])
        end
    end
    CM.lines!(ax_ca, [-100, 1400], [-100, 1400], color = :gray60, linestyle = :dash, label = "Unity (y=x)")
    if length(ca_clin_all) > 1
        b_ca, m_ca = hcat(ones(length(ca_clin_all)), ca_clin_all) \ ca_sim_all
        r_ca = cor(ca_clin_all, ca_sim_all)
        rmse_ca = sqrt(sum((ca_sim_all .- ca_clin_all) .^ 2) / length(ca_clin_all))
        nrmse_ca = rmse_ca / (maximum(ca_clin_all) - minimum(ca_clin_all)) * 100
        eq_ca = "y = $(round(m_ca, digits = 3))x $(b_ca >= 0 ? "+" : "-") $(round(abs(b_ca), digits = 1))"
        CM.lines!(ax_ca, [extrema(ca_clin_all)...], m_ca .* [extrema(ca_clin_all)...] .+ b_ca, color = :black, linewidth = 0.8, label = "Linear fit")
        CM.poly!(ax_ca, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)], color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
        CM.text!(ax_ca, 0.62, 0.18, text = "$(eq_ca)\nr = $(round(r_ca, digits = 4))\nnRMSE = $(round(nrmse_ca, digits = 1))%", space = :relative, align = (:left, :top), fontsize = 10)
    end
    CM.axislegend(ax_ca, position = :lt, labelsize = 9)

    # --- Bottom: Iodine rods ---
    ax_i = CM.Axis(
        fig[2, 1], title = "Iodine Rods", subtitle = "Clinical vs Simulated",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    i_clin_all, i_sim_all = Float64[], Float64[]

    for (k, (ci, sm_k)) in enumerate(zip(active_clinical_idx, sim_measurements[1:n_sims]))
        cm = se_measurements[ci]
        i_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "I ")]
        if !isempty(i_idx)
            CM.scatter!(
                ax_i, cm.rod_means[i_idx], sm_k.rod_means[i_idx];
                color = colors[k], markersize = 10, label = active_labels[k]
            )
            append!(i_clin_all, cm.rod_means[i_idx])
            append!(i_sim_all, sm_k.rod_means[i_idx])
        end
    end
    CM.lines!(ax_i, [-50, 500], [-50, 500], color = :gray60, linestyle = :dash, label = "Unity (y=x)")
    if length(i_clin_all) > 1
        b_i, m_i = hcat(ones(length(i_clin_all)), i_clin_all) \ i_sim_all
        r_i = cor(i_clin_all, i_sim_all)
        rmse_i = sqrt(sum((i_sim_all .- i_clin_all) .^ 2) / length(i_clin_all))
        nrmse_i = rmse_i / (maximum(i_clin_all) - minimum(i_clin_all)) * 100
        eq_i = "y = $(round(m_i, digits = 3))x $(b_i >= 0 ? "+" : "-") $(round(abs(b_i), digits = 1))"
        CM.lines!(ax_i, [extrema(i_clin_all)...], m_i .* [extrema(i_clin_all)...] .+ b_i, color = :black, linewidth = 0.8, label = "Linear fit")
        CM.poly!(ax_i, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)], color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
        CM.text!(ax_i, 0.62, 0.18, text = "$(eq_i)\nr = $(round(r_i, digits = 4))\nnRMSE = $(round(nrmse_i, digits = 1))%", space = :relative, align = (:left, :top), fontsize = 10)
    end
    CM.axislegend(ax_i, position = :lt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "ge_fbp_scatter_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07110038-0000-4000-8000-000000000000
md"""
### FBP: Noise
"""

# ╔═╡ 07110039-0000-4000-8000-000000000000
let
    water_idx = 1
    dose_sim_idx = [1, 2, 3]
    dose_clin_idx = [1, 3, 5]
    dose_labels = ["120 kVp / 50 mA\n(3.38 mGy)", "120 kVp / 150 mA\n(10.16 mGy)", "120 kVp / 300 mA\n(20.38 mGy)"]
    n_dose = min(length(dose_sim_idx), length(sim_measurements))

    dose_clin_σ = [se_measurements[dose_clin_idx[i]].rod_stds[water_idx] for i in 1:n_dose]
    dose_sim_σ = [sim_measurements[dose_sim_idx[i]].rod_stds[water_idx] for i in 1:n_dose]

    kvp_sim_idx = [4, 5, 2, 6]
    kvp_clin_idx = [7, 9, 3, 11]
    kvp_labels = ["80 kVp / 480 mA\n(10.32 mGy)", "100 kVp / 250 mA\n(10.53 mGy)", "120 kVp / 150 mA\n(10.16 mGy)", "140 kVp / 110 mA\n(10.85 mGy)"]
    n_kvp = min(length(kvp_sim_idx), length(sim_measurements))

    kvp_clin_σ = [se_measurements[kvp_clin_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]
    kvp_sim_σ = [sim_measurements[kvp_sim_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]

    fig = CM.Figure(size = (1000, 800), fontsize = 13)

    ax1 = CM.Axis(
        fig[1, 1]; title = "120 kVp — Dose Ladder",
        ylabel = "Water σ (HU)", xticks = (1:n_dose, dose_labels)
    )
    CM.barplot!(
        ax1, collect(1:n_dose) .- 0.2, dose_clin_σ; width = 0.35,
        color = :steelblue, label = "Clinical FBP"
    )
    CM.barplot!(
        ax1, collect(1:n_dose) .+ 0.2, dose_sim_σ; width = 0.35,
        color = :darkorange, label = "Simulated FBP"
    )
    CM.axislegend(ax1; position = :rb)

    ax2 = CM.Axis(
        fig[2, 1]; title = "~10 mGy CTDIvol — kVp Series",
        ylabel = "Water σ (HU)", xticks = (1:n_kvp, kvp_labels)
    )
    CM.barplot!(
        ax2, collect(1:n_kvp) .- 0.2, kvp_clin_σ; width = 0.35,
        color = :steelblue, label = "Clinical FBP"
    )
    CM.barplot!(
        ax2, collect(1:n_kvp) .+ 0.2, kvp_sim_σ; width = 0.35,
        color = :darkorange, label = "Simulated FBP"
    )
    CM.axislegend(ax2; position = :rb)

    CM.save(joinpath(RESULTS_DIR, "ge_fbp_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07110040-0000-4000-8000-000000000000
md"""
### FBP: NPS
"""

# ╔═╡ e85ee5fb-67f6-43b8-91fd-6018bfc1535a
let
    base_clinical_idx = [1, 3, 5, 7, 9, 11]
    base_scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n_sims = min(length(sim_measurements), length(base_clinical_idx))
    fig = CM.Figure(size = (900, 900), fontsize = 11)

    for i in 1:n_sims
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        ax = CM.Axis(
            fig[row, col]; title = base_scan_labels[i],
            subtitle = "Clinical vs Simulated",
            xlabel = "Spatial frequency (lp/cm)", ylabel = "nNPS (A.U.)"
        )
        cm = se_measurements[base_clinical_idx[i]]
        sm = sim_measurements[i]
        CM.lines!(
            ax, cm.nps.frequencies, cm.nps.nnps_1d;
            color = :steelblue, linewidth = 1.5, label = "Clinical"
        )
        CM.lines!(
            ax, sm.nps.frequencies, sm.nps.nnps_1d;
            color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Simulated"
        )
        CM.axislegend(ax; position = :rt, labelsize = 8)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_fbp_nps.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07110042-0000-4000-8000-000000000000
md"""
### FBP: MTF
"""

# ╔═╡ 07110043-0000-4000-8000-000000000000
let
    base_clinical_idx = [1, 3, 5, 7, 9, 11]
    base_scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n_sims = min(length(sim_measurements), length(base_clinical_idx))
    fig = CM.Figure(size = (900, 900), fontsize = 11)

    for i in 1:n_sims
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        ax = CM.Axis(
            fig[row, col]; title = base_scan_labels[i],
            subtitle = "Clinical vs Simulated",
            xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
            limits = (nothing, nothing, 0, 1.05)
        )
        CM.hlines!(ax, [0.5, 0.1]; color = :gray80, linestyle = :dash, linewidth = 0.8)
        cm = se_measurements[base_clinical_idx[i]]
        sm = sim_measurements[i]
        CM.lines!(
            ax, cm.mtf.frequencies, cm.mtf.mtf;
            color = :steelblue, linewidth = 1.5, label = "Clinical"
        )
        CM.lines!(
            ax, sm.mtf.frequencies, sm.mtf.mtf;
            color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Simulated"
        )
        CM.axislegend(ax; position = :rt, labelsize = 8)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_fbp_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 6f63608c-dd46-4d0a-8f1d-133dc56bc8dd
md"""
## 12. SE HIR Reconstruction & Comparison

Hybrid Iterative Reconstruction (HIR) — our equivalent of GE's ASiR-V.
Comparison plots use clinical ASiR-V 50% as the reference.
"""

# ╔═╡ f61c87a4-a27e-4bc6-bc4d-502c3e3a47fe
begin
    hir_strength = 3
    hir_lambda = 10.0f0
    hir_nepochs = 2
    hir_n_subsets = 12
    hir_huber_delta = 0.06f0
    hir_relaxation = 0.35f0
end

# ╔═╡ f925dafa-e0da-4673-8d84-82fe636bd6c2
# Scan 1: 120 kVp / 50 mA — HIR RECONSTRUCT (BHC → GPU HIR → CPU volume)
sim_recon_hir_1 = let
    sino_gpu = MtlArray(sim_sino_1.sino)
    air_ref_gpu = sim_sino_1.air_ref !== nothing ? MtlArray(Float32.(sim_sino_1.air_ref)) : nothing
    geom = sim_sino_1.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[120], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; air_reference = air_ref_gpu)
    recon_μ = ws_hir.volume
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[120];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = replace(sim_sino_1.name, "FBP" => "HIR"), kvp = sim_sino_1.kvp)
end

# ╔═╡ b84d07e5-64fd-4365-aca6-d03b4b5a1032
# Scan 1: 120 kVp / 50 mA — HIR HU CONVERSION (CPU only)
sim_hu_hir_1 = let
    μ_w = bhc_μ_water[120]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_hir_1.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_hir_1.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 12c84e0f-9cb4-49a0-a1fc-5cbe52b92f8a
# Scan 2: 120 kVp / 150 mA — HIR RECONSTRUCT (BHC → GPU HIR → CPU volume)
sim_recon_hir_2 = let
    sino_gpu = MtlArray(sim_sino_2.sino)
    air_ref_gpu = sim_sino_2.air_ref !== nothing ? MtlArray(Float32.(sim_sino_2.air_ref)) : nothing
    geom = sim_sino_2.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[120], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; air_reference = air_ref_gpu)
    recon_μ = ws_hir.volume
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[120];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = replace(sim_sino_2.name, "FBP" => "HIR"), kvp = sim_sino_2.kvp)
end

# ╔═╡ 4ca8d674-a0c1-407a-ab1d-84f06d966159
# Scan 2: 120 kVp / 150 mA — HIR HU CONVERSION (CPU only)
sim_hu_hir_2 = let
    μ_w = bhc_μ_water[120]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_hir_2.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_hir_2.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 6331e7a1-6dce-43bf-97b9-4ecbf766d40c
# Scan 3: 120 kVp / 300 mA — HIR RECONSTRUCT (BHC → GPU HIR → CPU volume)
sim_recon_hir_3 = let
    sino_gpu = MtlArray(sim_sino_3.sino)
    air_ref_gpu = sim_sino_3.air_ref !== nothing ? MtlArray(Float32.(sim_sino_3.air_ref)) : nothing
    geom = sim_sino_3.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[120], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; air_reference = air_ref_gpu)
    recon_μ = ws_hir.volume
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[120];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = replace(sim_sino_3.name, "FBP" => "HIR"), kvp = sim_sino_3.kvp)
end

# ╔═╡ 4917411f-f885-46bb-9f7e-a3de406d236c
# Scan 3: 120 kVp / 300 mA — HIR HU CONVERSION (CPU only)
sim_hu_hir_3 = let
    μ_w = bhc_μ_water[120]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_hir_3.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_hir_3.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ cad1f6b8-66fc-47f1-8386-e10c97ce0e4a
# Scan 4: 80 kVp / 480 mA — HIR RECONSTRUCT (BHC → GPU HIR → CPU volume)
sim_recon_hir_4 = let
    sino_gpu = MtlArray(sim_sino_4.sino)
    air_ref_gpu = sim_sino_4.air_ref !== nothing ? MtlArray(Float32.(sim_sino_4.air_ref)) : nothing
    geom = sim_sino_4.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[80], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; air_reference = air_ref_gpu)
    recon_μ = ws_hir.volume
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[80];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = replace(sim_sino_4.name, "FBP" => "HIR"), kvp = sim_sino_4.kvp)
end

# ╔═╡ 00aeca04-810f-4543-bef1-70db235be25c
# Scan 4: 80 kVp / 480 mA — HIR HU CONVERSION (CPU only)
sim_hu_hir_4 = let
    μ_w = bhc_μ_water[80]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_hir_4.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_hir_4.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ cf576e1e-5ef4-4478-9209-f584c373ef49
# Scan 5: 100 kVp / 250 mA — HIR RECONSTRUCT (BHC → GPU HIR → CPU volume)
sim_recon_hir_5 = let
    sino_gpu = MtlArray(sim_sino_5.sino)
    air_ref_gpu = sim_sino_5.air_ref !== nothing ? MtlArray(Float32.(sim_sino_5.air_ref)) : nothing
    geom = sim_sino_5.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[100], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; air_reference = air_ref_gpu)
    recon_μ = ws_hir.volume
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[100];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = replace(sim_sino_5.name, "FBP" => "HIR"), kvp = sim_sino_5.kvp)
end

# ╔═╡ 487bb689-9149-45e4-b153-2bad6138ebf5
# Scan 5: 100 kVp / 250 mA — HIR HU CONVERSION (CPU only)
sim_hu_hir_5 = let
    μ_w = bhc_μ_water[100]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_hir_5.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_hir_5.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 3793479d-79e9-4948-85c1-ff3c049f1cd0
# Scan 6: 140 kVp / 110 mA — HIR RECONSTRUCT (BHC → GPU HIR → CPU volume)
sim_recon_hir_6 = let
    sino_gpu = MtlArray(sim_sino_6.sino)
    air_ref_gpu = sim_sino_6.air_ref !== nothing ? MtlArray(Float32.(sim_sino_6.air_ref)) : nothing
    geom = sim_sino_6.geom
    recon_size = sim_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[140], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = hir_strength,
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    ws_hir.params = BS.HIRParams(
        hir_strength, hir_lambda, 30, hir_nepochs,
        hir_n_subsets, hir_huber_delta, hir_relaxation, (25, 35)
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size; air_reference = air_ref_gpu)
    recon_μ = ws_hir.volume
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[140];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_hir = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = replace(sim_sino_6.name, "FBP" => "HIR"), kvp = sim_sino_6.kvp)
end

# ╔═╡ 903f4e17-1824-418a-b8f9-21546163d740
# Scan 6: 140 kVp / 110 mA — HIR HU CONVERSION (CPU only)
sim_hu_hir_6 = let
    μ_w = bhc_μ_water[140]
    recon_hu = Float32.(BS.to_hounsfield(sim_recon_hir_6.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_recon_hir_6.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ cbe20e99-ca03-4045-a898-2de307d19fce
sim_results_hir = [sim_hu_hir_1, sim_hu_hir_2, sim_hu_hir_3, sim_hu_hir_4, sim_hu_hir_5, sim_hu_hir_6];

# ╔═╡ ed7cbb89-2f22-4587-ad7f-1f3224b73f80
sim_oriented_hir = let
    orient = identity
    orient = (f -> s -> reverse(f(s), dims = 2))(orient)  # Flip Left-Right
    [
        (
                name = r.name,
                recon = Float32.(mapslices(orient, r.recon, dims = (1, 2))),
                mu_water = r.mu_water,
            )
            for r in sim_results_hir
    ]
end;

# ╔═╡ 10a11afd-004a-467a-b808-7e761cb21678
# Reuse FBP segmentation (same phantom geometry) for HIR measurements
sim_measurements_hir = [
    measure_scan(r.recon, sim_seg_result.mask, sim_seg_result.rods, sim_seg_result.center, "sim_hir_$(r.name)")
        for r in sim_oriented_hir
];

# ╔═╡ 3924da10-33fd-415b-9666-f6022d0adaf6
md"""
### HIR: Qualitative Side-by-Side
"""

# ╔═╡ ebc9865c-feed-4e79-9cac-39e583a3796d
# HIR: 2×3 qualitative side-by-side (Clinical ASiR-V 50% vs Simulated HIR)
let
    clin_vols = [
        hu_120_low_ir, hu_120_mid_ir, hu_120_high_ir,
        hu_80_ir, hu_100_ir, hu_140_ir,
    ]
    scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n = min(length(clin_vols), length(sim_oriented_hir))
    fig = CM.Figure(size = (800, n * 400), fontsize = 10)

    for i in 1:n
        clin_slice = clin_vols[i][:, :, seg_result.slice_idx]
        sim_slice = sim_oriented_hir[i].recon[:, :, sim_seg_result.slice_idx]

        ax1 = CM.Axis(fig[i, 1]; title = "Clinical ASiR-V — $(scan_labels[i])", yreversed = true)
        CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax1); CM.hidespines!(ax1)

        ax2 = CM.Axis(fig[i, 2]; title = "Simulated HIR — $(scan_labels[i])", yreversed = true)
        CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax2); CM.hidespines!(ax2)
    end

    CM.save(joinpath(RESULTS_DIR, "ge_hir_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 1cae28b1-775a-4f4f-90d2-5ae693b7d2ac
md"""
### HIR: Line Profiles (Clinical vs Simulated)
"""

# ╔═╡ 681131e3-887d-4426-aefe-f904abb4667a
# Line profiles through center — 120 kVp / 150 mA and 80 kVp / 480 mA
let
    fig = CM.Figure(size = (1000, 700), fontsize = 11)

    # --- Top: 120 kVp / 150 mA ---
    clin_slice_1 = hu_120_mid_ir[:, :, seg_result.slice_idx]
    sim_slice_1 = sim_oriented_hir[2].recon[:, :, sim_seg_result.slice_idx]
    mid_row_c1 = size(clin_slice_1, 1) ÷ 2
    mid_row_s1 = size(sim_slice_1, 1) ÷ 2

    sim_shift_px = 4.0

    pixel_mm_c1 = 350.0 / size(clin_slice_1, 2)
    pixel_mm_s1 = 350.0 / size(sim_slice_1, 2)
    x_c1 = range(0, step = pixel_mm_c1, length = size(clin_slice_1, 2))
    x_s1 = range(0, step = pixel_mm_s1, length = size(sim_slice_1, 2))
    x_s1 = range(sim_shift_px * pixel_mm_s1, step = pixel_mm_s1, length = size(sim_slice_1, 2))


    ax1 = CM.Axis(fig[1, 1]; title = "120 kVp / 150 mA — Horizontal Line Profile (mid-row)", xlabel = "Position (mm)", ylabel = "HU")
    CM.lines!(
        ax1, collect(x_c1), Float64.(clin_slice_1[mid_row_c1, :]);
        color = :steelblue, linewidth = 1.2, label = "Clinical ASiR-V 50%"
    )
    CM.lines!(
        ax1, collect(x_s1), Float64.(sim_slice_1[mid_row_s1, :]);
        color = :orangered, linewidth = 1.2, label = "Simulated HIR"
    )
    CM.hlines!(ax1, [0.0]; color = :gray70, linestyle = :dash, linewidth = 0.6)
    CM.axislegend(ax1; position = :rt, labelsize = 9)

    CM.ylims!(ax1, low = -600)

    # --- Bottom: 80 kVp / 480 mA ---
    clin_slice_2 = hu_80_ir[:, :, seg_result.slice_idx]
    sim_slice_2 = sim_oriented_hir[4].recon[:, :, sim_seg_result.slice_idx]
    mid_row_c2 = size(clin_slice_2, 1) ÷ 2
    mid_row_s2 = size(sim_slice_2, 1) ÷ 2

    pixel_mm_c2 = 350.0 / size(clin_slice_2, 2)
    pixel_mm_s2 = 350.0 / size(sim_slice_2, 2)
    x_c2 = range(0, step = pixel_mm_c2, length = size(clin_slice_2, 2))
    x_s2 = range(0, step = pixel_mm_s2, length = size(sim_slice_2, 2))
    x_s2 = range(sim_shift_px * pixel_mm_s2, step = pixel_mm_s2, length = size(sim_slice_2, 2))

    ax2 = CM.Axis(
        fig[2, 1]; title = "80 kVp / 480 mA — Horizontal Line Profile (mid-row)",
        xlabel = "Position (mm)", ylabel = "HU"
    )
    CM.lines!(
        ax2, collect(x_c2), Float64.(clin_slice_2[mid_row_c2, :]);
        color = :steelblue, linewidth = 1.2, label = "Clinical ASiR-V 50%"
    )
    CM.lines!(
        ax2, collect(x_s2), Float64.(sim_slice_2[mid_row_s2, :]);
        color = :orangered, linewidth = 1.2, label = "Simulated HIR"
    )
    CM.hlines!(ax2, [0.0]; color = :gray70, linestyle = :dash, linewidth = 0.6)
    CM.axislegend(ax2; position = :rt, labelsize = 9)
    CM.ylims!(ax2, low = -600)

    CM.save(joinpath(RESULTS_DIR, "ge_hir_line_profiles.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ c42f625b-9db4-42a5-90c6-fa2c37fe8b18
md"""
### HIR: Scatter Plot (HU)
"""

# ╔═╡ bfd647b4-44d7-4582-8b5b-dd5910efd729
# HIR: Ca/I scatter — Clinical ASiR-V 50% vs Simulated HIR
let
    asirv_clinical_idx = [2, 4, 6, 8, 10, 12]
    scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n_sims = min(length(sim_measurements_hir), length(asirv_clinical_idx))
    active_clinical_idx = asirv_clinical_idx[1:n_sims]
    active_labels = scan_labels[1:n_sims]
    colors = CM.cgrad(:tab10, max(n_sims, 2), categorical = true)

    fig = CM.Figure(size = (750, 900), fontsize = 11)

    ax_ca = CM.Axis(
        fig[1, 1]; title = "Calcium Rods", subtitle = "Clinical ASiR-V vs Simulated HIR",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    ca_clin_all, ca_sim_all = Float64[], Float64[]

    for (k, (ci, sm_k)) in enumerate(zip(active_clinical_idx, sim_measurements_hir[1:n_sims]))
        cm = se_measurements[ci]
        ca_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "Ca")]
        if !isempty(ca_idx)
            CM.scatter!(
                ax_ca, cm.rod_means[ca_idx], sm_k.rod_means[ca_idx];
                color = colors[k], markersize = 10, label = active_labels[k]
            )
            append!(ca_clin_all, cm.rod_means[ca_idx])
            append!(ca_sim_all, sm_k.rod_means[ca_idx])
        end
    end
    CM.lines!(ax_ca, [-300, 2000], [-300, 2000]; color = :gray60, linestyle = :dash, linewidth = 1, label = "Unity (y = x)")
    if length(ca_clin_all) > 1
        X_ca = hcat(ones(length(ca_clin_all)), ca_clin_all)
        b_ca, m_ca = X_ca \ ca_sim_all
        r_ca = cor(ca_clin_all, ca_sim_all)
        rmse_ca = sqrt(sum((ca_sim_all .- ca_clin_all) .^ 2) / length(ca_clin_all))
        nrmse_ca = rmse_ca / (maximum(ca_clin_all) - minimum(ca_clin_all)) * 100
        x_fit_ca = range(extrema(ca_clin_all)..., length = 100)
        sign_ca = b_ca >= 0 ? " + " : " - "
        eq_ca = "y = $(round(m_ca, digits = 3))x$(sign_ca)$(round(abs(b_ca), digits = 1))"
        CM.lines!(
            ax_ca, collect(x_fit_ca), m_ca .* collect(x_fit_ca) .+ b_ca;
            color = :black, linewidth = 0.8, label = "Linear fit"
        )
        CM.poly!(
            ax_ca, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)];
            color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative
        )
        CM.text!(
            ax_ca, 0.62, 0.18; space = :relative, align = (:left, :top), fontsize = 10,
            text = "$(eq_ca)\nr = $(round(r_ca, digits = 4))\nnRMSE = $(round(nrmse_ca, digits = 1))%"
        )
    end
    CM.axislegend(ax_ca; position = :lt, labelsize = 9)

    ax_i = CM.Axis(
        fig[2, 1]; title = "Iodine Rods", subtitle = "Clinical ASiR-V vs Simulated HIR",
        xlabel = "Clinical HU", ylabel = "Simulated HU"
    )
    i_clin_all, i_sim_all = Float64[], Float64[]

    for (k, (ci, sm_k)) in enumerate(zip(active_clinical_idx, sim_measurements_hir[1:n_sims]))
        cm = se_measurements[ci]
        i_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "I ")]
        if !isempty(i_idx)
            CM.scatter!(
                ax_i, cm.rod_means[i_idx], sm_k.rod_means[i_idx];
                color = colors[k], markersize = 10, label = active_labels[k]
            )
            append!(i_clin_all, cm.rod_means[i_idx])
            append!(i_sim_all, sm_k.rod_means[i_idx])
        end
    end
    CM.lines!(ax_i, [-150, 1000], [-150, 1000]; color = :gray60, linestyle = :dash, linewidth = 1, label = "Unity (y = x)")
    if length(i_clin_all) > 1
        X_i = hcat(ones(length(i_clin_all)), i_clin_all)
        b_i, m_i = X_i \ i_sim_all
        r_i = cor(i_clin_all, i_sim_all)
        rmse_i = sqrt(sum((i_sim_all .- i_clin_all) .^ 2) / length(i_clin_all))
        nrmse_i = rmse_i / (maximum(i_clin_all) - minimum(i_clin_all)) * 100
        x_fit_i = range(extrema(i_clin_all)..., length = 100)
        sign_i = b_i >= 0 ? " + " : " - "
        eq_i = "y = $(round(m_i, digits = 3))x$(sign_i)$(round(abs(b_i), digits = 1))"
        CM.lines!(
            ax_i, collect(x_fit_i), m_i .* collect(x_fit_i) .+ b_i;
            color = :black, linewidth = 0.8, label = "Linear fit"
        )
        CM.poly!(
            ax_i, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)];
            color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative
        )
        CM.text!(
            ax_i, 0.62, 0.18; space = :relative, align = (:left, :top), fontsize = 10,
            text = "$(eq_i)\nr = $(round(r_i, digits = 4))\nnRMSE = $(round(nrmse_i, digits = 1))%"
        )
    end
    CM.axislegend(ax_i; position = :lt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "ge_hir_scatter_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 28e45f60-2542-4305-8c9c-3aecf1b3aa23
md"""
### HIR: Noise
"""

# ╔═╡ d35a2ac4-467a-4ade-afaf-dc668880559a
let
    water_idx = 1
    dose_sim_idx = [1, 2, 3]
    dose_clin_idx = [2, 4, 6]  # ASiR-V indices
    dose_labels = ["120 kVp / 50 mA\n(3.38 mGy)", "120 kVp / 150 mA\n(10.16 mGy)", "120 kVp / 300 mA\n(20.38 mGy)"]
    n_dose = min(length(dose_sim_idx), length(sim_measurements_hir))

    dose_clin_σ = [se_measurements[dose_clin_idx[i]].rod_stds[water_idx] for i in 1:n_dose]
    dose_sim_σ = [sim_measurements_hir[dose_sim_idx[i]].rod_stds[water_idx] for i in 1:n_dose]

    kvp_sim_idx = [4, 5, 2, 6]
    kvp_clin_idx = [8, 10, 4, 12]  # ASiR-V indices
    kvp_labels = ["80 kVp / 480 mA\n(10.32 mGy)", "100 kVp / 250 mA\n(10.53 mGy)", "120 kVp / 150 mA\n(10.16 mGy)", "140 kVp / 110 mA\n(10.85 mGy)"]
    n_kvp = min(length(kvp_sim_idx), length(sim_measurements_hir))

    kvp_clin_σ = [se_measurements[kvp_clin_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]
    kvp_sim_σ = [sim_measurements_hir[kvp_sim_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]

    fig = CM.Figure(size = (1000, 800), fontsize = 13)

    ax1 = CM.Axis(
        fig[1, 1]; title = "120 kVp — Dose Ladder",
        ylabel = "Water σ (HU)", xticks = (1:n_dose, dose_labels)
    )
    CM.barplot!(
        ax1, collect(1:n_dose) .- 0.2, dose_clin_σ; width = 0.35,
        color = :steelblue, label = "Clinical ASiR-V 50%"
    )
    CM.barplot!(
        ax1, collect(1:n_dose) .+ 0.2, dose_sim_σ; width = 0.35,
        color = :darkorange, label = "Simulated HIR"
    )
    CM.axislegend(ax1; position = :rb)

    ax2 = CM.Axis(
        fig[2, 1]; title = "~10 mGy CTDIvol — kVp Series",
        ylabel = "Water σ (HU)", xticks = (1:n_kvp, kvp_labels)
    )
    CM.barplot!(
        ax2, collect(1:n_kvp) .- 0.2, kvp_clin_σ; width = 0.35,
        color = :steelblue, label = "Clinical ASiR-V 50%"
    )
    CM.barplot!(
        ax2, collect(1:n_kvp) .+ 0.2, kvp_sim_σ; width = 0.35,
        color = :darkorange, label = "Simulated HIR"
    )
    CM.axislegend(ax2; position = :rb)

    CM.save(joinpath(RESULTS_DIR, "ge_hir_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 4776ad7f-edee-4c44-91bb-5ad76a63832e
md"""
### HIR: NPS
"""

# ╔═╡ 405ad9a1-3cac-4b19-87b3-1d041d8c0813
let
    asirv_clinical_idx = [2, 4, 6, 8, 10, 12]
    scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n_sims = min(length(sim_measurements_hir), length(asirv_clinical_idx))
    fig = CM.Figure(size = (900, 900), fontsize = 11)

    for i in 1:n_sims
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        ax = CM.Axis(
            fig[row, col]; title = scan_labels[i],
            subtitle = "Clinical ASiR-V 50% vs Simulated HIR",
            xlabel = "Spatial frequency (lp/cm)", ylabel = "nNPS (A.U.)"
        )
        cm = se_measurements[asirv_clinical_idx[i]]
        sm = sim_measurements_hir[i]
        f_c, v_c = cm.nps.frequencies, cm.nps.nps_1d
        good_c = v_c .> 0
        CM.lines!(
            ax, cm.nps.frequencies, cm.nps.nnps_1d;
            color = :steelblue, linewidth = 1.5, label = "Clinical ASiR-V"
        )
        f_s, v_s = sm.nps.frequencies, sm.nps.nps_1d
        good_s = v_s .> 0
        CM.lines!(
            ax, sm.nps.frequencies, sm.nps.nnps_1d;
            color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Simulated HIR"
        )
        CM.axislegend(ax; position = :rt, labelsize = 8)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_hir_nps.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ d3bd445c-ad9f-4133-93d9-8af91ed7f85d
md"""
### HIR: MTF
"""

# ╔═╡ 5298fd86-136b-4af3-aa0f-be0e810f5d24
let
    asirv_clinical_idx = [2, 4, 6, 8, 10, 12]
    scan_labels = [
        "120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
        "80kVp 480mA", "100kVp 250mA", "140kVp 110mA",
    ]

    n_sims = min(length(sim_measurements_hir), length(asirv_clinical_idx))
    fig = CM.Figure(size = (900, 900), fontsize = 11)

    for i in 1:n_sims
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        ax = CM.Axis(
            fig[row, col]; title = scan_labels[i],
            subtitle = "Clinical ASiR-V 50% vs Simulated HIR",
            xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
            limits = (nothing, nothing, 0, 1.05)
        )
        CM.hlines!(ax, [0.5, 0.1]; color = :gray80, linestyle = :dash, linewidth = 0.8)
        cm = se_measurements[asirv_clinical_idx[i]]
        sm = sim_measurements_hir[i]
        CM.lines!(
            ax, cm.mtf.frequencies, cm.mtf.mtf;
            color = :steelblue, linewidth = 1.5, label = "Clinical ASiR-V"
        )
        CM.lines!(
            ax, sm.mtf.frequencies, sm.mtf.mtf;
            color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Simulated HIR"
        )
        CM.axislegend(ax; position = :rt, labelsize = 8)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_hir_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 4f8ee88d-a948-44e7-a610-a2711e6da234
md"""
## 13. DE Simulation — GE GSI (Rapid kVp Switching)

GE Revolution Apex GSI dual-energy uses **rapid kVp switching** (single tube, 80/140 kVp
on alternating views) with an **additional spectral filter on the 140 kVp tube** to improve
spectral separation.

**Clinical DE protocol** (from DICOM + RDSR):
- **kVp:** 80 / 140 alternating per view
- **mA:** 203.5 (80 kVp) / 202.5 (140 kVp) — GE private tags `(0053,1085)` / `(0053,1083)`
- **Gantry rotation:** 0.5 s — per-kVp effective exposure: **0.25 s** (rapid switching)
- **Collimation:** 64 × 0.625 mm = 40 mm (half of SE 80 mm)
- **Views/kVp/rotation:** ~984
- **Filtration:** Large body bowtie + inherent Al + **spectral filter on 140 kVp** (empirically tuned)
- **CTDIvol:** 10.07 mGy (32 cm body)
- **Decomposition:** Projection-based, interleaved
"""

# ╔═╡ f9da78c0-9c6a-4c83-9f4e-592eecca5aaf
de_mA_80 = 407 * 0.65 # 407 instantaneous × 0.5 duty cycle

# ╔═╡ 3aefce3f-90e3-48fb-bae7-b7c965b7d6f8
de_mA_140 = 405 * 0.35 # 405 instantaneous × 0.5 duty cycle

# ╔═╡ 6f5ce242-a5e2-4cd5-8cac-47f528dc21bb
# DE scan parameters — matching clinical GE GSI protocol
begin
    de_rotation_time = 0.5        # seconds — actual gantry rotation
    de_collimation_mm = 5.0       # dev mode — same as SE (real clinical: 40mm)
    de_n_views = 984              # views per kVp per rotation

    # mA: DICOM reports instantaneous 407/405 mA, but each kVp only fires
    # for half the rotation (rapid switching). Effective mA = instantaneous × 0.5

    DE_SIM_SCANS = [
        (name = "DE_80kVp_204mA", kvp = 80, mA = de_mA_80),
        (name = "DE_140kVp_203mA", kvp = 140, mA = de_mA_140),
    ]
    DE_VMI_ENERGIES = [40, 70, 100, 140]  # keV for VMI synthesis

    # Two separate scanner references for DE — same physical hardware (rapid kVp switching)
    # but having named references makes the simulation flow clearer.
    sim_scanner_de_80 = sim_scanner
    sim_scanner_de_140 = sim_scanner
end

# ╔═╡ ff650b6f-76e6-491e-8076-1dd002c80d7e
# SPECTRAL FILTRATION — Gd₂O₂S K-edge filter (rapid kVp switching):
# In rapid kVp switching both beams traverse the same physical filter.
# Clinical GE GSI has no K-edge filter; this is a research optimization:
#   - Gd K-edge at 50.2 keV sits between 80/140 kVp spectra
#   - Preferentially attenuates 80 kVp photons above 50 keV → better separation
#   - Yao & Pelc (Med Phys 2014, doi:10.1118/1.4866381): 0.142–0.150 mm
#   - Stanford patent US9435900B2; PMC11272100 (2024): 0.09 mm
# Thickness is tunable; 0.1 mm is a conservative default.
# de_kedge_filter = [("Gd2O2S", 0.15)]  # mm — applied to BOTH kVp tubes
de_kedge_filter = Tuple{String,Float64}[]

# ╔═╡ 2a6f0b64-05a7-4a01-9009-66888902edcb
# # Visualize 80 & 140 kVp spectra before/after Gd₂O₂S K-edge filtration
# let
    # # Unfiltered spectra (scanner flat filter + Al only, no K-edge)
    # prot_80_no = BS.CTProtocol(kVp = 80, additional_filters = additional_filters)
    # prot_140_no = BS.CTProtocol(kVp = 140, additional_filters = additional_filters)
    # e80_no, w80_no = BS.resolve_spectrum(sim_opts, prot_80_no; scanner = sim_scanner_de_80)
    # e140_no, w140_no = BS.resolve_spectrum(sim_opts, prot_140_no; scanner = sim_scanner_de_140)

    # # Filtered spectra (scanner flat filter + Al + Gd₂O₂S)
    # de_filters = vcat(additional_filters, de_kedge_filter)
    # prot_80_gd = BS.CTProtocol(kVp = 80, additional_filters = de_filters)
    # prot_140_gd = BS.CTProtocol(kVp = 140, additional_filters = de_filters)
    # e80_gd, w80_gd = BS.resolve_spectrum(sim_opts, prot_80_gd; scanner = sim_scanner_de_80)
    # e140_gd, w140_gd = BS.resolve_spectrum(sim_opts, prot_140_gd; scanner = sim_scanner_de_140)

    # # Normalize to peak for visual comparison
    # norm80 = maximum(w80_no)
    # norm140 = maximum(w140_no)

    # # Mean energies (fluence-weighted)
    # mean_E(e, w) = sum(e .* w) / sum(w)
    # mE_80_no = mean_E(e80_no, w80_no)
    # mE_140_no = mean_E(e140_no, w140_no)
    # mE_80_gd = mean_E(e80_gd, w80_gd)
    # mE_140_gd = mean_E(e140_gd, w140_gd)

    # fig = CM.Figure(size = (900, 400), fontsize = 12)

    # # Left: before K-edge filter
    # ax1 = CM.Axis(
        # fig[1, 1]; title = "Before Gd₂O₂S filter",
        # xlabel = "Energy (keV)", ylabel = "Relative fluence"
    # )
    # CM.lines!(ax1, e80_no, w80_no ./ norm80; color = :dodgerblue, linewidth = 1.5, label = "80 kVp")
    # CM.lines!(ax1, e140_no, w140_no ./ norm140; color = :orangered, linewidth = 1.5, label = "140 kVp")
    # CM.vlines!(ax1, [50.2]; color = :gray50, linestyle = :dash, linewidth = 0.8, label = "Gd K-edge (50.2 keV)")
    # CM.vlines!(ax1, [mE_80_no]; color = :dodgerblue, linestyle = :dash, linewidth = 1.2, label = "⟨E⟩ 80 = $(round(mE_80_no; digits = 1)) keV")
    # CM.vlines!(ax1, [mE_140_no]; color = :orangered, linestyle = :dash, linewidth = 1.2, label = "⟨E⟩ 140 = $(round(mE_140_no; digits = 1)) keV")
    # CM.axislegend(ax1; position = :rt)

    # # Right: after K-edge filter
    # ax2 = CM.Axis(
        # fig[1, 2]; title = "After Gd₂O₂S filter ($(de_kedge_filter[1][2]) mm)",
        # xlabel = "Energy (keV)", ylabel = "Relative fluence"
    # )
    # CM.lines!(ax2, e80_gd, w80_gd ./ norm80; color = :dodgerblue, linewidth = 1.5, label = "80 kVp")
    # CM.lines!(ax2, e140_gd, w140_gd ./ norm140; color = :orangered, linewidth = 1.5, label = "140 kVp")
    # CM.vlines!(ax2, [50.2]; color = :gray50, linestyle = :dash, linewidth = 0.8, label = "Gd K-edge (50.2 keV)")
    # CM.vlines!(ax2, [mE_80_gd]; color = :dodgerblue, linestyle = :dash, linewidth = 1.2, label = "⟨E⟩ 80 = $(round(mE_80_gd; digits = 1)) keV")
    # CM.vlines!(ax2, [mE_140_gd]; color = :orangered, linestyle = :dash, linewidth = 1.2, label = "⟨E⟩ 140 = $(round(mE_140_gd; digits = 1)) keV")
    # CM.axislegend(ax2; position = :rt)

    # CM.linkaxes!(ax1, ax2)

    # CM.save(joinpath(RESULTS_DIR, "ge_gsi_spectra_filter.png"), fig, px_per_unit = 2)
    # fig
# end

# ╔═╡ bb68196f-26e4-41fc-85bc-203373d91b4a
# DE recon geometry — same FOV/matrix as SE, different z for collimation
begin
    de_recon_z_cm = de_collimation_mm / 10.0
    de_n_recon_slices = round(Int, de_collimation_mm / sim_slice_thickness_mm)
    de_matrix_size = (sim_recon_xy, sim_recon_xy, de_n_recon_slices)

    de_recon_geom = BS.ReconOptions(
        algorithm = :fdk,
        matrix_size = de_matrix_size,
        fov_cm = sim_recon_fov_cm,
        z_cm = de_recon_z_cm,
    )
end

# ╔═╡ 37298bb1-4b57-4f5c-9fea-ec81dbfb3394
# DE: 80 kVp / ~204 mA effective — SIMULATE (GPU → sinogram → free GPU)
sim_de_sino_low = let
    sc = DE_SIM_SCANS[1]
    low_kvp_filters = vcat(additional_filters, de_kedge_filter)
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = de_n_views,
        rotation_time = de_rotation_time,
        collimation_mm = de_collimation_mm,
        additional_filters = low_kvp_filters,
    )
    @info "Simulating DE: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner_de_80, prot, sim_opts, de_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner_de_80, prot, sim_opts, de_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ fd123cde-bb22-489d-a496-e00363f46157
# DE: 140 kVp / ~203 mA effective — SIMULATE (GPU → sinogram → free GPU)
sim_de_sino_high = let
    sc = DE_SIM_SCANS[2]
    high_kvp_filters = vcat(additional_filters, de_kedge_filter)
    prot = BS.CTProtocol(
        kVp = sc.kvp,
        mA = sc.mA,
        views = de_n_views,
        rotation_time = de_rotation_time,
        collimation_mm = de_collimation_mm,
        additional_filters = high_kvp_filters,
    )
    @info "Simulating DE: $(sc.name)..."
    ws = BS.create_eict_workspace(
        sim_scanner_de_140, prot, sim_opts, de_recon_geom, sim_phantom_gpu,
    )
    BS.simulate!(ws, sim_phantom_gpu, sim_scanner_de_140, prot, sim_opts, de_recon_geom)
    air_ref = ws.bowtie_air_reference !== nothing ? Array(ws.bowtie_air_reference) : nothing
    result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, air_ref = air_ref, name = sc.name, kvp = sc.kvp)
    ws = nothing
    GC.gc(true)
    result
end

# ╔═╡ 89c1ff84-d4bc-47ca-b08d-8ada2bd469f4
# DE: 80 kVp — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_de_recon_low = let
    sino_gpu = MtlArray(sim_de_sino_low.sino)
    geom = sim_de_sino_low.geom
    recon_size = de_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[80], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[80];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_de_sino_low.name, kvp = sim_de_sino_low.kvp)
end

# ╔═╡ 7e06d734-4023-4078-9b8f-3b55c2e389e6
# DE: 80 kVp — HU CONVERSION (CPU only)
sim_de_hu_low = let
    μ_w = bhc_μ_water[80]
    recon_hu = Float32.(BS.to_hounsfield(sim_de_recon_low.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_de_recon_low.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 36986f73-d54b-4190-a52c-f3f23e53c533
# DE: 140 kVp — FBP RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_de_recon_high = let
    sino_gpu = MtlArray(sim_de_sino_high.sino)
    geom = sim_de_sino_high.geom
    recon_size = de_matrix_size
    if bhc_enabled
        sino_corrected = BS.apply_bhc_two_material(
            sino_gpu, bhc_models[140], geom, recon_size;
            volume_extent = sim_phantom_gpu.extent
        )
        sino_gpu = MtlArray(sino_corrected)
    end
    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y)
    )
    recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)
    if bhc_enabled
        BS.apply_bhc_image_domain(
            recon_μ, geom, recon_size, bhc_μ_water[140];
            hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
            scale_factor = img_bhc_scale_factor,
            volume_extent = sim_phantom_gpu.extent
        )
    end
    vol = Array(recon_μ)
    ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; GC.gc(true)
    (volume = vol, name = sim_de_sino_high.name, kvp = sim_de_sino_high.kvp)
end

# ╔═╡ 4efd48c9-753a-4418-8095-fa12b7cc5a95
# DE: 140 kVp — HU CONVERSION (CPU only)
sim_de_hu_high = let
    μ_w = bhc_μ_water[140]
    recon_hu = Float32.(BS.to_hounsfield(sim_de_recon_high.volume; μ_water = μ_w))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
    BS.apply_radial_cupping_correction!(recon_hu; fov_cm = 35.0)
    (name = sim_de_recon_high.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ e8af3f62-e606-4f10-9a09-9e0620910f58
# DE: Orient simulated images (match SE orientation)
sim_de_oriented = let
    orient = identity
    orient = (f -> s -> reverse(f(s), dims = 2))(orient)  # Flip Left-Right
    results = [sim_de_hu_low, sim_de_hu_high]
    [
        (
                name = r.name,
                recon = Float32.(mapslices(orient, r.recon, dims = (1, 2))),
                mu_water = r.mu_water,
            )
            for r in results
    ]
end;

# ╔═╡ 5cc1fa5c-3722-4fa1-b0f0-d816e204c8bf
# DE: Qualitative — Low vs High kVp
let
    scan_labels = ["80 kVp / $(de_mA_80) mA eff.", "140 kVp / $(de_mA_140) mA eff."]
    n = length(sim_de_oriented)
    mid_z = size(sim_de_oriented[1].recon, 3) ÷ 2 + 1
    fig = CM.Figure(size = (800, 400), fontsize = 10)

    for i in 1:n
        sim_slice = sim_de_oriented[i].recon[:, :, mid_z]

        ax = CM.Axis(fig[1, i]; title = "Simulated DE — $(scan_labels[i])", yreversed = true)
        CM.heatmap!(ax, sim_slice; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax); CM.hidespines!(ax)
    end

    CM.save(joinpath(RESULTS_DIR, "ge_de_sim_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080001-0000-4000-8000-000000000004
md"""
## 5. Photoelectric + Compton Basis Decomposition

**Pipeline:** Cong analytic per-ray decomp → PWLS-L₂ sinogram restoration
(Noh 2009 cost + Long/Fessler 2014 2×2 matrix curvature) → FBP → post-FBP
radial capping correction. Each stage below has its own sinogram +
intermediate FBP so the streak / noise suppression is visible stage-by-stage.

| Stage | Role | Handles |
|-------|------|---------|
| §5.1 Cong 2022 | Per-ray polychromatic inversion (no spatial coupling) | Beam-hardening, basis inversion |
| §5.2 PWLS-L₂   | Statistical penalized-WLS with 2×2 per-ray matrix curvature | Streaks from per-ray decomp errors; iodine↔water anti-correlated noise |
| §5.3 Capping   | Per-slice radial even-polynomial fit on the basis images | Residual radial cupping/capping from incomplete BHC |
| §5.4 Final FBP | FDK on cleaned basis sinograms (feeds §5.3 capping); final viz shows all three corrections | Reconstruction output consumed by §6 VMI |

### 5.1 Cong et al. 2022 — Per-ray Analytic Decomposition

**[CDW22] Cong, De Man, Wang (2022)** *J X-Ray Sci Technol* 30:725–736,
*"Projection decomposition via univariate optimization for dual-energy CT."*
DOI 10.3233/XST-221153.

Invert the polychromatic dual-energy forward model per ray. **No calibration
scan required** — only physical constants and the already-resolved x-ray
spectrum (`BS.resolve_spectrum`, which contains source × bowtie × detector).

#### Physical basis ([CDW22] Eqs 3a–3e, 4)

- `μ(r, ε) = p(ε)·a(r) + q(ε)·c(r)`
- `a(r) = ρ·⟨Z⁴/A⟩` — photoelectric spatial component
- `c(r) = ρ·⟨Z/A⟩` — Compton spatial component
- `p(ε) = N_A·α⁴·(8/3)π·r_e²·√(32/ε⁷)`,  `ε = E / m_e c²`
- `q(ε) = N_A·f_kn(ε)` — Klein-Nishina (Eq 3e)

#### Per-ray sinogram decomposition (Eqs 6–10, three Roots.jl calls)

1. **Water-based 1D Brent inversion** for water-equivalent path `L`:
   `∫Ŝ_L(ε)·exp(−(p(ε)·a_w + q(ε)·c_w)·L) dε = T_L_meas`, then `c̄ = c_w·L`.
   Absorbs the water beam-hardening exactly, so the Taylor-5 residual
   `x = C − c̄` carries only Ca/I/air deviations from water — bounded, small.
2. **Inner Newton on Eq 8 quintic** for `x = h(y)` — strictly convex ⇒ unique
   real root. Newton from `x = 0` converges in ≤ 8 iterations.
3. **Outer Roots.Brent on Eq 10.** `G(y) = T_H_pred(y, c̄+h(y)) − T_H_meas` is
   monotonic on `[0, τ_L/min(p_L))`. `Roots.find_zero(G, (0, y_max), Brent())`.

**Sinogram outputs (line integrals, per ray):**
- `sim_de_decomp.sino_iodine = y = ∫ρ_I(r)·dr`       (g/cm²)
- `sim_de_decomp.sino_water  = W = ∫ρ_W(r)·dr`       (g/cm²)

Cong runs in the **direct (iodine, water) basis** defined by §0 —
NIST mass-atten tables at each spectral bin, `water_basis = (0, 1)`.
Forward-model Jacobian has κ ≈ 160 (vs. κ ≈ 110 k for the (Z⁴/A, Z/A)
photo/Compton basis), so post-decomposition noise in `(sino_I, sino_W)`
is well-conditioned and balanced — prerequisite for the monotonic σ(E)
trend in §6 VMI.

The intermediate FBP below shows **raw Cong output** — no spatial coupling
yet, so per-ray decomposition errors show up as coherent streaks through
high-attenuation rods. §5.2 cleans those up.
"""

# ╔═╡ 00080002-0000-4000-8000-000000000004
# Direct (water, iodine) basis tables for Cong's per-ray decomposition.
#
# No (Z⁴/A, Z/A) intermediate, no LS fit.  We feed apply_cong! the NIST
# mass-attenuation coefficients (cm²/g) of iodine and water sampled at
# every spectral bin of the resolved 80/140 kVp beams:
#
#   p_L,H[k] = (μ/ρ)_iodine(ε_k)       ← iodine mass atten
#   q_L,H[k] = (μ/ρ)_water(ε_k)        ← water  mass atten
#
# Cong's kernel is basis-agnostic: it solves a 2-unknown nonlinear system
# per ray for whatever (a_line, c_line) its `p, q` tables describe.  With
# the tables above + `water_basis = (0, 1)` (§0b), apply_cong! directly
# outputs `(sino_iodine, sino_water)` = (∫ρ_I·dr, ∫ρ_W·dr) — material
# line integrals in g/cm².  No change-of-basis needed downstream.
#
# Why this conditioning is better than (photoelectric, Compton):
#   · Forward-model Jacobian at working point is the spectrum-effective
#     mass-atten matrix M = [μρ_W_L μρ_I_L; μρ_W_H μρ_I_H],  κ(M) ≈ 160.
#   · The (a, c) = (Z⁴/A, Z/A) basis has C = [a_W a_I; c_W c_I] with
#     κ(C) ≈ 112 000 because water and iodine are almost parallel in the
#     Z/A direction.  That's what we're avoiding.
#
# K-edge fidelity: iodine's 33 keV jump in (μ/ρ)(ε) is captured natively
# by every spectral bin above the edge (NIST tables).
de_basis = let
    prot_L = BS.CTProtocol(kVp = 80,  additional_filters = additional_filters)
    prot_H = BS.CTProtocol(kVp = 140, additional_filters = additional_filters)
    e_L, w_L = BS.resolve_spectrum(sim_opts, prot_L; scanner = sim_scanner)
    e_H, w_H = BS.resolve_spectrum(sim_opts, prot_H; scanner = sim_scanner)

    ŵ_L = Float32.(Float64.(w_L) ./ sum(Float64.(w_L)))
    ŵ_H = Float32.(Float64.(w_H) ./ sum(Float64.(w_H)))

    μρ_iodine(E) = BS.compute_mass_μ_at_energy(XA.Elements.Iodine, Float64(E))
    μρ_water(E)  = BS.compute_mass_μ_at_energy(XA.Materials.water,  Float64(E))

    p_L = Float32[Float32(μρ_iodine(e)) for e in e_L]   # iodine  mass atten over low-kVp spectral bins
    q_L = Float32[Float32(μρ_water(e))  for e in e_L]   # water   mass atten
    p_H = Float32[Float32(μρ_iodine(e)) for e in e_H]
    q_H = Float32[Float32(μρ_water(e))  for e in e_H]

    # Cache VMI target-E mass atten (cm²/g) for §6 VMI synthesis.
    vmi_energies = [40.0, 70.0, 100.0, 140.0]
    μρ_I_vmi     = Float32[Float32(μρ_iodine(E)) for E in vmi_energies]
    μρ_W_vmi     = Float32[Float32(μρ_water(E))  for E in vmi_energies]

    @info "[direct (W, I) basis] NIST mass atten at $(length(e_L)) L-bins + $(length(e_H)) H-bins"
    @info "  p_L = μρ_iodine range: [$(round(minimum(p_L), sigdigits=3)), $(round(maximum(p_L), sigdigits=3))]  cm²/g"
    @info "  q_L = μρ_water  range: [$(round(minimum(q_L), sigdigits=3)), $(round(maximum(q_L), sigdigits=3))]  cm²/g"
    for (k, E) in enumerate(vmi_energies)
        @info "  E=$(Int(E)) keV (VMI target):  μρ_I=$(round(μρ_I_vmi[k], sigdigits=4))   μρ_W=$(round(μρ_W_vmi[k], sigdigits=4)) cm²/g"
    end

    (ŵ_L = ŵ_L, p_L = p_L, q_L = q_L,
     ŵ_H = ŵ_H, p_H = p_H, q_H = q_H,
     vmi_energies = vmi_energies,
     μρ_I_vmi     = μρ_I_vmi,
     μρ_W_vmi     = μρ_W_vmi)
end

# ╔═╡ 00080003-0000-4000-8000-000000000004
# Cong's step-1 Brent solve is on a "pure-reference column" whose
# (a_ref, c_ref) in the BASIS we've chosen describe the material.  We
# picked (iodine, water) and want the reference column to be pure water
# at 1 g/cm³ (the medical DECT-appropriate "dominant material" seed).
# In (iodine, water) coords that is (0, 1) ⇒ Cong's step-1 Brent
# solves  ∫ Ŝ_L(ε) exp(−μρ_water(ε)·L) dε = T_L_meas  → `L` IS the
# water line integral directly (no `c_w` scale factor needed).
water_basis = (a = 0.0f0, c = 1.0f0)

# ╔═╡ 00080004-0000-4000-8000-000000000004
# Per-ray Cong 2022 decomposition: Brent(water-L) → Newton(quintic h(y)) → Brent(y).
# Runs via BS.apply_cong! which dispatches through AK.foreachindex, so Metal
# or CUDA arrays execute the same kernel on the GPU without a separate code
# path.  Brent is a 1:1 port of Roots.Brent (test/vmi_brent_parity.jl).
sim_de_decomp = let
    # Stage sinograms onto GPU (MtlArray) — Metal is the GE Apex target.
    sino_L_gpu = MtlArray(Float32.(sim_de_sino_low.sino))
    sino_H_gpu = MtlArray(Float32.(sim_de_sino_high.sino))

    t1 = time()
    sino_I_gpu, sino_W_gpu = BS.apply_cong(
        sino_L_gpu, sino_H_gpu;
        basis  = de_basis,
        water_basis = water_basis,

        # ── Per-ray solver tuning (defaults below) ──
        newton_max_iter = 5,              # default = 12; try 20 for dense iodine
        newton_tol      = eps(Float32),   # try 5f-7 if Newton spins
        y_max_factor    = 0.2,            # default = 0.99; try 2.0 to widen Brent bracket
        y_max_cap       = 1f7,            # overflow guard; rarely relevant
    )
    dt = time() - t1

    sino_iodine = Array(sino_I_gpu)       # ∫ρ_I·dr   (g/cm²)
    sino_water  = Array(sino_W_gpu)       # ∫ρ_W·dr   (g/cm²)
    sino_L_gpu = nothing; sino_H_gpu = nothing
    sino_I_gpu = nothing; sino_W_gpu = nothing
    GC.gc(true)

    @info "[CDW22] apply_cong in (W, I) basis done in $(round(dt, digits=1)) s"
    @info "  ⟨∫ρ_I·dr⟩ = $(round(mean(sino_iodine), sigdigits=4))   iodine line integral  (g/cm²)"
    @info "  ⟨∫ρ_W·dr⟩ = $(round(mean(sino_water),  sigdigits=4))   water  line integral  (g/cm²)"

    (sino_iodine = sino_iodine, sino_water = sino_water, geom = sim_de_sino_low.geom)
end;

# ╔═╡ 00080005-0000-4000-8000-000000000004
# Iodine + water sinograms — mid-view and mid-row.
let
    sino_I = sim_de_decomp.sino_iodine
    sino_W = sim_de_decomp.sino_water
    mid_view = size(sino_I, 2) ÷ 2
    mid_row  = size(sino_I, 3) ÷ 2

    fig = CM.Figure(size = (1400, 800), fontsize = 13)

    for (row, name, sino) in [(1, "Iodine  y = ∫ρ_I(r)dr  (g/cm²)",  sino_I),
                              (2, "Water  W = ∫ρ_W(r)dr  (g/cm²)",   sino_W)]
        ax1 = CM.Axis(
            fig[row, 1]; title = "$name — mid-view (view $mid_view)",
            xlabel = "Detector column", ylabel = "Detector row",
            aspect = CM.DataAspect()
        )
        hm1 = CM.heatmap!(ax1, sino[:, mid_view, :]'; colormap = :viridis)
        CM.Colorbar(fig[row, 2], hm1; width = 12)

        ax2 = CM.Axis(
            fig[row, 3]; title = "$name — mid-row (row $mid_row)",
            xlabel = "Detector column", ylabel = "View angle"
        )
        hm2 = CM.heatmap!(ax2, sino[:, :, mid_row]'; colormap = :viridis)
        CM.Colorbar(fig[row, 4], hm2; width = 12)
    end

    CM.save(joinpath(RESULTS_DIR, "sinogram_iodine_water.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080012-0000-4000-8000-000000000004
# Intermediate FBP of RAW Cong output — no sinogram restoration yet.
# Shows the per-ray decomposition streaks §5.2 PWLS-SQS will clean up.
sim_recon_cong = let
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = MtlArray(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, sim_de_decomp.geom, de_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp.geom, de_matrix_size))
        ws = nothing; sino_gpu = nothing; GC.gc(true)
        Float32.(vol)
    end

    t1 = time()
    ρ_I_img = _fbp(sim_de_decomp.sino_iodine)
    ρ_W_img = _fbp(sim_de_decomp.sino_water)
    @info "Intermediate FBP (Cong only, no smoothing) done in $(round(time() - t1, digits=1)) s"
    (iodine = ρ_I_img, water = ρ_W_img)
end;

# ╔═╡ 00080013-0000-4000-8000-000000000004
# Cong-only FBP mid-slice — streaks from per-ray analytic decomp should be visible.
let
    ρ_I = sim_recon_cong.iodine
    ρ_W = sim_recon_cong.water
    mid = size(ρ_I, 3) ÷ 2
    ρ_I_slice = ρ_I[:, :, mid]
    ρ_W_slice = ρ_W[:, :, mid]

    I_lo, I_hi = quantile(vec(ρ_I_slice), 0.01), quantile(vec(ρ_I_slice), 0.995)
    W_lo, W_hi = quantile(vec(ρ_W_slice), 0.01), quantile(vec(ρ_W_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Iodine  ρ_I(r)  (g/cm³) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρ_I_slice; colormap = :viridis, colorrange = (I_lo, I_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Water  ρ_W(r)  (g/cm³) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρ_W_slice; colormap = :viridis, colorrange = (W_lo, W_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_iodine_water_cong.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080014-0000-4000-8000-000000000004
md"""
### 5.2 PWLS-SQS Sinogram Restoration — Noh, Fessler, Kinahan 2009

**[NFK09] Noh, Fessler, Kinahan (2009)** *IEEE Trans Med Imaging* 28(11):
1688–1702. *"Statistical Sinogram Restoration in Dual-Energy CT for PET
Attenuation Correction."* DOI 10.1109/TMI.2009.2023988.

Cong 2022 (§5.1) gives a per-ray analytic decomposition with **zero spatial
coupling** — each detector column/view is solved in isolation. At low dose
or through high-attenuation rods, per-ray decomposition errors + Poisson
measurement noise stack up as coherent streaks in the intermediate FBP
above. NFK09 instead jointly denoises both basis sinograms in the sinogram
domain by minimizing a penalized weighted least-squares cost:

    Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))²  +  Σ_l ½·γ_l·‖C·s_l‖²   [NFK09 Eq 12]

where
- `s = (s_I, s_W)` — basis-sinogram state (iodine + water line integrals, g/cm²)
- `h_{mi}` — measured polychromatic transmissions (`−log y_{mi}/I_m`)
- `f_m(s_i) = −log Σ_k ŵ_m[k]·exp(−p_m[k]·s_y − q_m[k]·s_C)` — **same**
  polychromatic forward model used in §5.1
- `w_{mi} ≈ y_{mi} ∝ exp(−h_{mi})` — Poisson inverse-variance weights
  (NFK09 Eq 13); high-attenuation rays get low weight so their residuals
  don't dominate the fit
- `C` — 2D 2nd-order difference operator (detector-col + view axes), Neumann
  BC; γ_l controls smoothness per basis

**Why warm-start from Cong?** The data-fit landscape is nonconvex (because
`f_m` is nonlinear), but Cong already sits in the right basin — starting
there skips the global-descent phase, letting a handful of SQS iterations
denoise without introducing its own bias.

Solved via **Separable Paraboloidal Surrogates (SPS)** with Fessler-Erdogan
precomputed curvature (MIRT `de_wls_dercurv.m`):

    M[m, l]        = Σ_k ŵ_m[k]·c_l[k]                  — spectral MAC matrix, 2×2
    curv_geom[l]   = Σ_m |M[m, l]| · Σ_{l'} |M[m, l']|  — data-term majorant
    curv_R_l       = γ_l · 32                           — De Pierro row-sum bound
                                                          for |C'C| in 2D
    curv_l[i]      = max_m(w_{mi}) · curv_geom[l] + curv_R_l   — per-pixel, constant

SQS update (MIRT `pwls_sqs_os.m`, adapted: G = I for sinogram domain):

    grad_l[i]    = Σ_m w_{mi}·(f_m(s_i) − h_{mi})·∂f_m/∂s_l
    grad_R_l[i]  = γ_l·(C'C·s_l)[i]
    s_l[i] ← max( s_l[i] − relax·(grad_l[i] + grad_R_l[i]) / curv_l[i], 0 )

SPS guarantees **monotonic descent of Φ(s)** at every iteration (Fessler
2000). We log Φ per iter and warn if it ever increases — catches any sign
or curvature-bound bug immediately.

**Implementation.** 1:1 port of MIRT's `ct/de_wls_dercurv.m` (data-term
gradient + precomputed curvature) and `wls/pwls_sqs_os.m` (SQS loop).
Per-slice threading over detector rows. See cell §5.2 body below for the
code and the hyperparameter cell above it for `κ_iodine`, `κ_water`,
`n_iter`, `relax`.
"""

# ╔═╡ 00080021-0000-4000-8000-000000000004
# ── PWLS-L₂ sinogram restoration hyperparameters (Noh 2009 cost + Long/Fessler
# 2014 §IV-B 2×2 matrix curvature; implementation in cell below). ─────────────
#
# Reparameterization: γ in the Noh/Fessler cost is FIXED to 1 here.  κ below
# is the De Pierro row-sum bound on the reg Hessian, exposed directly.  Smaller
# κ = bigger smoothing step per iter = more total smoothing at fixed n_iter.
# Stability: κ ≥ ‖CᵀC‖_op/2 ≈ 8 (biharmonic in 2D) guarantees monotonic
# descent of the pure-reg SQS step; in the full M_i = C̈_data + diag(κ) matrix
# the data curvature adds to the diagonal and only raises the M_i spectrum,
# so the stability bound on κ stays the same regardless of basis.
#
# ── Scale asymmetry in the (W, I) basis ──────────────────────────────────────
# Data curvature per-ray is cd_II = Σ_m w_m · ⟨μρ_I⟩²_m, cd_WW = Σ_m w_m · ⟨μρ_W⟩²_m.
# With ⟨μρ_I⟩ ≈ 5–10 cm²/g (iodine, spectrum-weighted) and ⟨μρ_W⟩ ≈ 0.2 cm²/g
# (water), cd_II / cd_WW ≈ (5/0.2)² ≈ 625×.  So iodine is strongly data-
# constrained, water is reg-constrained unless κ_water is scaled down.
# Starting point: κ_iodine = 32, κ_water = 1.0.  With κ_water = 32 (same as
# in the old (photo, Compton) basis), the water updates are reg-dominated
# and smooth aggressively — OK for clean recon but can over-blur water.
# Tune κ_water down to 0.1 for more data-driven water updates, or up to 32
# to match the old behavior.
#
#   pwls_enable   — master switch; false = pass-through Cong
#   pwls_n_iter   — SQS iterations
#   pwls_κ_iodine — iodine-direction reg damping  (scale-matched to cd_II ≈ 25–100)
#   pwls_κ_water  — water-direction reg damping   (scale-matched to cd_WW ≈ 0.04)
#   pwls_relax    — SQS relaxation (default 1.0 = unrelaxed).
#
# Paper: Noh, Fessler, Kinahan. IEEE TMI 28(11):1688–1702, 2009 (cost).
#        Long, Fessler.     IEEE TMI 33(8):1614–1626, 2014 (L₂ surrogate).
begin
    pwls_enable   = true
    pwls_n_iter   = 20
    pwls_κ_iodine = 32.0     # default 32  — matched to cd_II ~ O(25–100)
    pwls_κ_water  = 32.0      # default 1.0 — matched to cd_WW ~ O(0.04); start here, retune if needed
    pwls_relax    = 1.0
end

# ╔═╡ 00080022-0000-4000-8000-000000000004
# PWLS-SQS sinogram restoration — Noh 2009 COST with Long & Fessler 2014 §IV-B
# L₂ SURROGATE (per-ray 2×2 matrix curvature, eq 26-28) instead of MIRT's
# diagonal scalar bound (which is L₃ collapsed to L₀=1).  Stays in sinogram
# domain; same SQS outer loop; same γ/n_iter/relax hyperparameters.  The only
# change is the curvature: off-diagonal data-term GN term now captures the
# iodine↔water anti-correlation that the diagonal majorant drops.  Published
# per Niu 2014 / Zhang 2014 / Persson-Adler 2017: ~2× noise-variance reduction
# at matched resolution + suppression of the anti-correlated artifacts, which
# is why the post-hoc ACNR smoother (Kalender 1988) is no longer needed.
#
# Cost (unchanged, Noh 2009 Eq 12):
#     Φ(s) = ½·Σ_{i,m} w_{mi}·(h_{mi} − f_m(s_i))²  +  Σ_l ½·γ_l·‖C·s_l‖²
#
# Per-ray update (Long & Fessler 2014 eq 26-28 in sino domain, L_0 = 2 basis):
#     J_m,i = (⟨μρ_I⟩_β, ⟨μρ_W⟩_β)ᵀ     — 2-vector, β = current Beer-weighted spectrum
#                                          p_L[k] = μρ_iodine(ε_k), q_L[k] = μρ_water(ε_k)
#     C̈_i  = Σ_m w_{mi} · J_m,i · J_m,iᵀ — 2×2 PSD data curvature (GN matrix)
#     M_i  = C̈_i + diag(κ_iodine, κ_water)   — reg curvature (γ=1 folded into κ)
#     g_i  = Σ_m w_{mi}·(f_m(s_i) − h_{mi})·J_m,i  +  ((CᵀC·s_I)_i, (CᵀC·s_W)_i)
#     s_i ← max( s_i − relax · M_i⁻¹ · g_i , 0 )    — closed-form 2×2 inverse
#
# Forward model is physically correct Beer-Lambert under the (W, I) basis:
#     f_m(s_I, s_W) = −log Σ_k ŵ_m[k] · exp(−μρ_iodine(ε_k)·s_I − μρ_water(ε_k)·s_W)
# i.e. exp sum of negative atten-length = polychromatic transmission for a
# ray containing s_I g/cm² iodine and s_W g/cm² water.
sim_de_decomp_pwls = let
    if !pwls_enable
        @info "PWLS restoration: DISABLED (pass-through Cong)"
        (sino_iodine = sim_de_decomp.sino_iodine,
         sino_water  = sim_de_decomp.sino_water,
         geom        = sim_de_decomp.geom,
         n_iter      = 0, κ_iodine = 0.0, κ_water = 0.0,
         relax       = 0.0, cost_history = Float64[])
    else
        sino_I = copy(sim_de_decomp.sino_iodine)    # Cong warm start (iodine line integrals, g/cm²)
        sino_W = copy(sim_de_decomp.sino_water)     # Cong warm start (water  line integrals, g/cm²)
        h_low  = sim_de_sino_low.sino
        h_high = sim_de_sino_high.sino

        ŵ_L = de_basis.ŵ_L;  p_L = de_basis.p_L;  q_L = de_basis.q_L    # p_L = μρ_iodine, q_L = μρ_water
        ŵ_H = de_basis.ŵ_H;  p_H = de_basis.p_H;  q_H = de_basis.q_H
        nE_L = length(ŵ_L);  nE_H = length(ŵ_H)

        n_col, n_view, n_row = size(sino_I)
        κ_I        = Float32(pwls_κ_iodine)    # regularizer step damping, iodine
        κ_W        = Float32(pwls_κ_water)     #   … water
        relax_f    = Float32(pwls_relax)
        reg_curv_I = κ_I                        # SQS step = reg_grad / κ
        reg_curv_W = κ_W

        # Per-slice 2D Laplacian (CᵀC·s) — two passes of the 3-stencil C = [1,-2,1]
        # with Neumann BC, applied along col AND view axes then summed.  C is
        # symmetric ⇒ CᵀC = C·C.  Same regularizer as the original Noh port.
        apply_CtC_slice! = function (out::AbstractArray{Float32, 2},
                                     s  ::AbstractArray{Float32, 2},
                                     tmp::AbstractArray{Float32, 2})
            nc, nv = size(s)
            # x-axis: tmp = Cx·s, then out = Cx·tmp
            @inbounds for v in 1:nv, c in 1:nc
                sl = c == 1  ? s[c, v] : s[c-1, v]
                sr = c == nc ? s[c, v] : s[c+1, v]
                tmp[c, v] = sl - 2f0*s[c, v] + sr
            end
            @inbounds for v in 1:nv, c in 1:nc
                tl = c == 1  ? tmp[c, v] : tmp[c-1, v]
                tr = c == nc ? tmp[c, v] : tmp[c+1, v]
                out[c, v] = tl - 2f0*tmp[c, v] + tr
            end
            # y-axis: tmp = Cy·s, then out += Cy·tmp
            @inbounds for v in 1:nv, c in 1:nc
                su = v == 1  ? s[c, v] : s[c, v-1]
                sd = v == nv ? s[c, v] : s[c, v+1]
                tmp[c, v] = su - 2f0*s[c, v] + sd
            end
            @inbounds for v in 1:nv, c in 1:nc
                tu = v == 1  ? tmp[c, v] : tmp[c, v-1]
                td = v == nv ? tmp[c, v] : tmp[c, v+1]
                out[c, v] += tu - 2f0*tmp[c, v] + td
            end
        end

        cost_history = Float64[]

        t0 = time()
        for iter in 1:pwls_n_iter
            Φ_total = Threads.Atomic{Float64}(0.0)

            Threads.@threads for r in 1:n_row
                # Per-thread slice-sized scratch; re-used within the slice.
                reg_I = zeros(Float32, n_col, n_view)
                reg_W = zeros(Float32, n_col, n_view)
                tmp_b = zeros(Float32, n_col, n_view)

                # Snapshot regularizer gradients (Jacobi SQS): fixed within iter.
                apply_CtC_slice!(reg_I, @view(sino_I[:, :, r]), tmp_b)
                apply_CtC_slice!(reg_W, @view(sino_W[:, :, r]), tmp_b)

                Φ_slice = 0.0

                @inbounds for v in 1:n_view, c in 1:n_col
                    Iv = sino_I[c, v, r]
                    Wv = sino_W[c, v, r]

                    # Low-kVp spectral Beer moments.  p_L[k] = μρ_iodine(ε_k),
                    # q_L[k] = μρ_water(ε_k), so P_L = ⟨μρ_I⟩_β, Q_L = ⟨μρ_W⟩_β
                    # under the current Beer-weighted spectrum β.
                    Z_L = 0f0;  Z_Lp = 0f0;  Z_Lq = 0f0
                    for k in 1:nE_L
                        wk = ŵ_L[k] * exp(-p_L[k]*Iv - q_L[k]*Wv)
                        Z_L  += wk;  Z_Lp += p_L[k]*wk;  Z_Lq += q_L[k]*wk
                    end
                    invZ_L = 1f0 / max(Z_L, 1f-20)
                    P_L = Z_Lp * invZ_L;  Q_L = Z_Lq * invZ_L
                    f_L = -log(max(Z_L, 1f-20))

                    # High-kVp spectral Beer moments.
                    Z_H = 0f0;  Z_Hp = 0f0;  Z_Hq = 0f0
                    for k in 1:nE_H
                        wk = ŵ_H[k] * exp(-p_H[k]*Iv - q_H[k]*Wv)
                        Z_H  += wk;  Z_Hp += p_H[k]*wk;  Z_Hq += q_H[k]*wk
                    end
                    invZ_H = 1f0 / max(Z_H, 1f-20)
                    P_H = Z_Hp * invZ_H;  Q_H = Z_Hq * invZ_H
                    f_H = -log(max(Z_H, 1f-20))

                    h_L = h_low[c, v, r];  h_H = h_high[c, v, r]
                    res_L = f_L - h_L
                    res_H = f_H - h_H
                    w_L = exp(-h_L)     # Poisson inv-var (Noh 2009 Eq 13)
                    w_H = exp(-h_H)

                    # Cost accumulation (data term only; reg term below).
                    Φ_slice += 0.5 * (w_L * res_L * res_L + w_H * res_H * res_H)

                    # Data gradient (2-vector): first slot = iodine, second = water.
                    g_d_I = w_L*res_L*P_L + w_H*res_H*P_H
                    g_d_W = w_L*res_L*Q_L + w_H*res_H*Q_H

                    # Data curvature (Long/Fessler eq 27 GN form, symmetric 2×2 PSD).
                    cd_II = w_L*P_L*P_L + w_H*P_H*P_H
                    cd_IW = w_L*P_L*Q_L + w_H*P_H*Q_H
                    cd_WW = w_L*Q_L*Q_L + w_H*Q_H*Q_H

                    # Regularizer contribution: grad via CᵀC, curv = κ.  γ=1 is
                    # folded into κ (see hyperparam cell), so no γ multiplier here.
                    rg_I = reg_I[c, v]
                    rg_W = reg_W[c, v]

                    # Total gradient + total 2×2 curvature.
                    gI = g_d_I + rg_I
                    gW = g_d_W + rg_W
                    m_II = cd_II + reg_curv_I
                    m_IW = cd_IW
                    m_WW = cd_WW + reg_curv_W

                    # Closed-form 2×2 solve: Δ = M⁻¹·g.
                    det_m   = m_II*m_WW - m_IW*m_IW
                    inv_det = 1f0 / max(det_m, 1f-20)
                    ΔI = inv_det * (m_WW*gI - m_IW*gW)
                    ΔW = inv_det * (m_II*gW - m_IW*gI)

                    # SQS update with non-neg clip (matches Noh behaviour).
                    sino_I[c, v, r] = max(Iv - relax_f * ΔI, 0f0)
                    sino_W[c, v, r] = max(Wv - relax_f * ΔW, 0f0)
                end

                # Regularizer cost: ½ γ_l · s_lᵀ · (CᵀC·s_l) — use snapshot reg_l.
                @inbounds for v in 1:n_view, c in 1:n_col
                    Φ_slice += 0.5 * sino_I[c, v, r] * reg_I[c, v]   # γ=1
                    Φ_slice += 0.5 * sino_W[c, v, r] * reg_W[c, v]
                end

                Threads.atomic_add!(Φ_total, Φ_slice)
            end

            push!(cost_history, Φ_total[])

            if iter > 1 && cost_history[iter] > cost_history[iter-1]
                @warn "PWLS-L₂: cost increased at iter $iter  (Φ_prev=$(cost_history[iter-1]), Φ_curr=$(cost_history[iter])).  Monotonicity violated — consider smaller relax."
            end
        end
        dt = time() - t0

        @info "[PWLS-L₂] 2×2 matrix curvature, $(pwls_n_iter) iters × $(n_row) slices in $(round(dt, digits=1)) s  ($(round(1000*dt/pwls_n_iter, digits=0)) ms/iter)  |  κ_I=$(κ_I), κ_W=$(κ_W), Φ: $(round(cost_history[1], sigdigits=4)) → $(round(cost_history[end], sigdigits=4))"

        (sino_iodine = sino_I, sino_water = sino_W,
         geom = sim_de_decomp.geom,
         n_iter = pwls_n_iter,
         κ_iodine = pwls_κ_iodine, κ_water = pwls_κ_water,
         relax = pwls_relax,
         cost_history = cost_history)
    end
end;

# ╔═╡ 00080015-0000-4000-8000-000000000004
# Iodine + water sinograms AFTER PWLS-SQS restoration — should look smoother
# (with streaks suppressed) compared to the raw Cong output in §5.1.
let
    sino_I = sim_de_decomp_pwls.sino_iodine
    sino_W = sim_de_decomp_pwls.sino_water
    mid_view = size(sino_I, 2) ÷ 2
    mid_row  = size(sino_I, 3) ÷ 2

    fig = CM.Figure(size = (1400, 800), fontsize = 13)
    tag = "κ_I=$(sim_de_decomp_pwls.κ_iodine), κ_W=$(sim_de_decomp_pwls.κ_water), k=$(sim_de_decomp_pwls.n_iter)"
    for (row, name, sino) in [(1, "Iodine y — post-PWLS ($tag)", sino_I),
                              (2, "Water  W — post-PWLS ($tag)", sino_W)]
        ax1 = CM.Axis(fig[row, 1]; title = "$name — mid-view (view $mid_view)",
                      xlabel = "Detector column", ylabel = "Detector row", aspect = CM.DataAspect())
        hm1 = CM.heatmap!(ax1, sino[:, mid_view, :]'; colormap = :viridis)
        CM.Colorbar(fig[row, 2], hm1; width = 12)
        ax2 = CM.Axis(fig[row, 3]; title = "$name — mid-row (row $mid_row)",
                      xlabel = "Detector column", ylabel = "View angle")
        hm2 = CM.heatmap!(ax2, sino[:, :, mid_row]'; colormap = :viridis)
        CM.Colorbar(fig[row, 4], hm2; width = 12)
    end
    CM.save(joinpath(RESULTS_DIR, "sinogram_iodine_water_pwls.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080016-0000-4000-8000-000000000004
# Intermediate FBP with PWLS-L₂ restoration applied.  Compare to sim_recon_cong
# above: basis-sinogram noise should be suppressed by the 2×2 matrix curvature
# (captures the iodine↔water anti-correlation in the update step itself) +
# 2D quadratic roughness penalty (radial and view directions).
sim_recon_pwls = let
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = MtlArray(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, sim_de_decomp_pwls.geom, de_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp_pwls.geom, de_matrix_size))
        ws = nothing; sino_gpu = nothing; GC.gc(true)
        Float32.(vol)
    end

    t1 = time()
    ρ_I = _fbp(sim_de_decomp_pwls.sino_iodine)
    ρ_W = _fbp(sim_de_decomp_pwls.sino_water)
    @info "Intermediate FBP (Cong + PWLS) done in $(round(time() - t1, digits=1)) s"
    (iodine = ρ_I, water = ρ_W)
end;

# ╔═╡ c1139ae3-5186-445e-81b8-5d932ca5ef98
# Cong-only FBP mid-slice — streaks from per-ray analytic decomp should be visible.
let
    ρ_I = sim_recon_cong.iodine
    ρ_W = sim_recon_cong.water
    mid = size(ρ_I, 3) ÷ 2
    ρ_I_slice = ρ_I[:, :, mid]
    ρ_W_slice = ρ_W[:, :, mid]

    I_lo, I_hi = quantile(vec(ρ_I_slice), 0.01), quantile(vec(ρ_I_slice), 0.995)
    W_lo, W_hi = quantile(vec(ρ_W_slice), 0.01), quantile(vec(ρ_W_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Iodine  ρ_I(r)  (g/cm³) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρ_I_slice; colormap = :viridis, colorrange = (I_lo, I_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Water  ρ_W(r)  (g/cm³) — Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρ_W_slice; colormap = :viridis, colorrange = (W_lo, W_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_iodine_water_cong.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080017-0000-4000-8000-000000000004
# Cong + PWLS FBP mid-slice — noise floor should drop vs. Cong-only, rods
# preserved (quadratic penalty is only radial 2nd-order diff, minimal blur).
let
    ρ_I = sim_recon_pwls.iodine
    ρ_W = sim_recon_pwls.water
    mid = size(ρ_I, 3) ÷ 2
    ρ_I_slice = ρ_I[:, :, mid]
    ρ_W_slice = ρ_W[:, :, mid]

    I_lo, I_hi = quantile(vec(ρ_I_slice), 0.01), quantile(vec(ρ_I_slice), 0.995)
    W_lo, W_hi = quantile(vec(ρ_W_slice), 0.01), quantile(vec(ρ_W_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Iodine  ρ_I(r)  (g/cm³) — Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρ_I_slice; colormap = :viridis, colorrange = (I_lo, I_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Water  ρ_W(r)  (g/cm³) — Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρ_W_slice; colormap = :viridis, colorrange = (W_lo, W_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_iodine_water_pwls.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00080020-0000-4000-8000-000000000004
md"""
### 5.3 Final FBP on Cleaned Basis Sinograms

Plain FDK on each fully-cleaned basis sinogram (shared Shepp-Logan-style
mild apodization). Filter preserves high-frequency edges (vessels, rod
walls) so the downstream VMI+ step has real structural content to work
with — heavy Hann would throw that away at the FBP stage and no amount of
downstream processing can recover it.

Output:
- `sim_recon_iodine_water.iodine/.water` — (ρ_I, ρ_W) density maps in
  g/cm³ straight out of FBP.  No capping on the density images — residual
  radial cupping is corrected downstream on the VMI HU output (see §6.2).

Cong outputs (W, I) directly from §5.1 (no photo/Compton intermediate), so
FBPing each sinogram produces a density map in g/cm³ immediately — no
basis inversion anywhere in §5.

HU conversion happens downstream in §6 VMI via the NIST mass-atten tables:
`μ(E) = μρ_I(E)·ρ_I + μρ_W(E)·ρ_W`.
"""

# ╔═╡ 00090002-0000-4000-8000-000000000004
# FBP each CLEANED basis sinogram on GPU. Shared smooth apodization filter.
sim_recon_iodine_water = let
    # Shepp-Logan-style mild apodization.  Preserves high-frequency edges
    # (vessels, rod walls) so the downstream guided-filter Mono+ step has real
    # structural content to work with — heavy Hann would throw that away at the
    # FBP stage and no amount of downstream processing can recover it.
    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    function _fbp(sino::Array{Float32, 3})
        sino_gpu = MtlArray(sino)
        ws = BS.create_fdk_recon_workspace(
            sino_gpu, sim_de_decomp_pwls.geom, de_matrix_size;
            filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
        )
        vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp_pwls.geom, de_matrix_size))
        ws = nothing; sino_gpu = nothing; GC.gc(true)
        Float32.(vol)
    end

    t1 = time()
    ρ_I = _fbp(sim_de_decomp_pwls.sino_iodine)
    ρ_W = _fbp(sim_de_decomp_pwls.sino_water)
    @info "FBP iodine + water done in $(round(time() - t1, digits=1)) s"
    @info "  ρ_I(r) range: [$(round(minimum(ρ_I), sigdigits=3)), $(round(maximum(ρ_I), sigdigits=3))]  g/cm³"
    @info "  ρ_W(r) range: [$(round(minimum(ρ_W), sigdigits=3)), $(round(maximum(ρ_W), sigdigits=3))]  g/cm³"

    (iodine = ρ_I, water = ρ_W)
end;

# ╔═╡ fd9969e0-415f-409d-8672-fe2d963b6486
# Cong-only FBP mid-slice — streaks from per-ray analytic decomp should be visible.
let
    ρ_I = sim_recon_cong.iodine
    ρ_W = sim_recon_cong.water
    mid = size(ρ_I, 3) ÷ 2
    ρ_I_slice = ρ_I[:, :, mid]
    ρ_W_slice = ρ_W[:, :, mid]

    I_lo, I_hi = quantile(vec(ρ_I_slice), 0.01), quantile(vec(ρ_I_slice), 0.995)
    W_lo, W_hi = quantile(vec(ρ_W_slice), 0.01), quantile(vec(ρ_W_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Iodine  ρ_I(r)  (g/cm³)", subtitle = "Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρ_I_slice; colormap = :viridis, colorrange = (I_lo, I_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Water  ρ_W(r)  (g/cm³)", subtitle = "Cong only  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρ_W_slice; colormap = :viridis, colorrange = (W_lo, W_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_iodine_water_cong.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 2b6fd506-c624-4be3-9a71-1d366ae58ada
# Cong + PWLS FBP mid-slice — noise floor should drop vs. Cong-only, rods
# preserved (quadratic penalty is only radial 2nd-order diff, minimal blur).
let
    ρ_I = sim_recon_pwls.iodine
    ρ_W = sim_recon_pwls.water
    mid = size(ρ_I, 3) ÷ 2
    ρ_I_slice = ρ_I[:, :, mid]
    ρ_W_slice = ρ_W[:, :, mid]

    I_lo, I_hi = quantile(vec(ρ_I_slice), 0.01), quantile(vec(ρ_I_slice), 0.995)
    W_lo, W_hi = quantile(vec(ρ_W_slice), 0.01), quantile(vec(ρ_W_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)
    ax1 = CM.Axis(fig[1, 1]; title = "Iodine  ρ_I(r)  (g/cm³)", subtitle = "Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm1 = CM.heatmap!(ax1, ρ_I_slice; colormap = :viridis, colorrange = (I_lo, I_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(fig[1, 3]; title = "Water  ρ_W(r)  (g/cm³)", subtitle = "Cong+PWLS  (slice $mid)", aspect = CM.DataAspect())
    hm2 = CM.heatmap!(ax2, ρ_W_slice; colormap = :viridis, colorrange = (W_lo, W_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_iodine_water_pwls.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 00090003-0000-4000-8000-000000000004
# Mid-slice of both density images — this is what §6 VMI synthesis consumes
# (sinogram-domain, so these images are just for inspection / ROI work).
let
    ρ_I = sim_recon_iodine_water.iodine
    ρ_W = sim_recon_iodine_water.water
    mid = size(ρ_I, 3) ÷ 2
    ρ_I_slice = ρ_I[:, :, mid]
    ρ_W_slice = ρ_W[:, :, mid]

    I_lo, I_hi = quantile(vec(ρ_I_slice), 0.01), quantile(vec(ρ_I_slice), 0.995)
    W_lo, W_hi = quantile(vec(ρ_W_slice), 0.01), quantile(vec(ρ_W_slice), 0.995)

    fig = CM.Figure(size = (1250, 570), fontsize = 13)

    ax1 = CM.Axis(
        fig[1, 1]; title = "Iodine ρ_I(r) (g/cm³)", subtitle = "Cong + PWLS-L₂ (slice $mid)",
        aspect = CM.DataAspect()
    )
    hm1 = CM.heatmap!(ax1, ρ_I_slice; colormap = :viridis, colorrange = (I_lo, I_hi))
    CM.Colorbar(fig[1, 2], hm1; width = 12)

    ax2 = CM.Axis(
        fig[1, 3]; title = "Water  ρ_W(r) (g/cm³)", subtitle = "Cong + PWLS-L₂ (slice $mid)",
        aspect = CM.DataAspect()
    )
    hm2 = CM.heatmap!(ax2, ρ_W_slice; colormap = :viridis, colorrange = (W_lo, W_hi))
    CM.Colorbar(fig[1, 4], hm2; width = 12)

    CM.save(joinpath(RESULTS_DIR, "recon_iodine_water.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000a0001-0000-4000-8000-000000000004
md"""
## 6. Virtual Monoenergetic Imaging — Mono+ / VMI+

### References

- **[G14] Grant, Flohr, Krauss, Sedlmair, Thomas, Schmidt (2014)** *Invest Radiol*
  49(9):586–592. DOI: 10.1097/RLI.0000000000000060 · PMID: 24710203.
  Siemens Healthcare DE CT team — reference paper for the syngo.via "Mono+"
  prototype (later VMI+).
- **[S14] Schabel et al. (2014)** *Fortschr Röntgenstr* 186:591–597.
  DOI: 10.1055/s-0034-1366423 — clinical evaluation.

### 6.1 Plain VMI (sinogram-domain synthesis + single FBP)

For each target energy E, form the VMI **sinogram** directly from the
post-PWLS basis-material sinograms (§5.2 `sim_de_decomp_pwls`):

    VMI_sino(E)[i, v, r] = (μ/ρ)_I(E)·sino_I[i, v, r] + (μ/ρ)_W(E)·sino_W[i, v, r]

then run a single FBP per target E to get the linear-μ VMI image,
converting to HU via NIST water linear atten at E:

    VMI_μ(E, r)  = FBP( VMI_sino(E) )
    VMI_HU(E, r) = 1000 · (VMI_μ(E, r) / μ_water(E) − 1)

(μ/ρ)_I and (μ/ρ)_W are the NIST MASS attenuation coefficients (cm²/g)
of elemental iodine and water at the target energy, cached in
`de_basis.μρ_I_vmi / μρ_W_vmi`. Multiplying by our (ρ_I, ρ_W) line
integrals in g/cm² gives linear-μ line integrals in cm⁻¹ — exactly
what FBP wants.

HU referenced to `BS.compute_μ_at_energy(XA.Materials.water, E)`.

Output: `sim_vmi.volumes` — VMI HU stacks at 40 / 70 / 100 / 140 keV.
"""

# ╔═╡ 000a0002-0000-4000-8000-000000000004
# Plain VMI synthesis — sinogram domain, single FBP per target energy.
#
#     VMI_sino(E) = μρ_I(E)·sino_I + μρ_W(E)·sino_W           [cm²/g × g/cm² = cm⁻¹·cm]
#     VMI_μ(E)    = FBP( VMI_sino(E) )                         [cm⁻¹]
#     VMI_HU(E)   = 1000·(VMI_μ(E)/μ_water_lin(E) − 1)
#
# Inputs: `sim_de_decomp_pwls.{sino_iodine, sino_water}` (post-PWLS
# (W, I) basis sinograms straight out of §5.2 — Cong decomposes directly
# into (W, I), so NO image-domain basis inversion anywhere).  FBP uses
# the same Shepp-Logan-style mild apodization as the intermediate recons.
sim_vmi = let
    energies = de_basis.vmi_energies
    sino_I   = sim_de_decomp_pwls.sino_iodine
    sino_W   = sim_de_decomp_pwls.sino_water

    filter_ctrl = (
        x = (0.0, 0.25, 0.5, 0.75, 1.0),
        y = (1.0, 0.95, 0.85, 0.65, 0.4),
    )

    # Shared FBP workspace: allocate once, reuse across energies (same
    # geometry, same matrix size, same filter).
    sino_gpu = MtlArray(similar(sino_I))
    ws = BS.create_fdk_recon_workspace(
        sino_gpu, sim_de_decomp_pwls.geom, de_matrix_size;
        filter = BS.CustomFilter(filter_ctrl.x, filter_ctrl.y)
    )

    # FOV mask geometry
    nx_img = de_matrix_size[1]
    ny_img = de_matrix_size[2]
    nz_img = de_matrix_size[3]
    cx, cy = (nx_img + 1) / 2, (ny_img + 1) / 2
    r_fov_sq = (0.5 * nx_img)^2

    volumes = Vector{Array{Float32, 3}}(undef, length(energies))
    t0 = time()
    for (k, E) in enumerate(energies)
        μρ_I_E = de_basis.μρ_I_vmi[k]       # cm²/g
        μρ_W_E = de_basis.μρ_W_vmi[k]

        # Linear combine in sinogram domain.
        sino_host = @. μρ_I_E * sino_I + μρ_W_E * sino_W    # cm⁻¹·cm  (Float32)
        copyto!(sino_gpu, sino_host)

        μ_vol = Array(BS.reconstruct!(ws, sino_gpu, sim_de_decomp_pwls.geom, de_matrix_size))   # cm⁻¹

        μ_w_E = Float64(BS.compute_μ_at_energy(XA.Materials.water, Float64(E)))
        hu = Float32.(BS.to_hounsfield(μ_vol; μ_water = μ_w_E))

        @inbounds for k2 in 1:nz_img, j in 1:ny_img, i in 1:nx_img
            if (i - cx)^2 + (j - cy)^2 > r_fov_sq
                hu[i, j, k2] = -1000f0
            end
        end
        volumes[k] = hu
    end
    ws = nothing; sino_gpu = nothing; GC.gc(true)
    dt = time() - t0

    @info "VMI (sino-domain synthesis + 1× FBP per E): $(Int.(Float64.(energies))) keV — $(length(energies)) energies in $(round(dt, digits=1)) s"
    (energies = energies, volumes = volumes)
end

# ╔═╡ 000a0004-0000-4000-8000-000000000004
# Plain VMI (raw, no capping) at 40 / 70 / 100 / 140 keV — soft-tissue window.
let
    fig = CM.Figure(size = (1400, 400), fontsize = 13)
    mid = size(sim_vmi.volumes[1], 3) ÷ 2
    for (i, E) in enumerate(sim_vmi.energies)
        ax = CM.Axis(fig[1, i]; title = "VMI  $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax, sim_vmi.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))
    end
    CM.Label(fig[0, :]; text = "Plain VMI (raw) — Soft Tissue Window (−200 … 500 HU)",
             fontsize = 15, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "vmi_all_energies.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000a0010-0000-4000-8000-000000000004
md"""
### 6.2 Post-VMI Radial Capping Correction

Residual radial **cupping/capping** (background HU slowly drifting away from
zero as a function of image radius) persists in the raw VMI because Cong +
PWLS-L₂ don't model every beam-hardening path exactly.  We apply a per-slice
even-polynomial radial correction directly on each HU volume.

For each target energy E and each slice:
1. Select water-like voxels (`hu_lo ≤ v ≤ hu_hi`) — excludes air and the
   high-contrast inserts without needing material-specific thresholds.
2. Fit an even polynomial in radius:  `offset(r) = c₀ + c₁·r² + c₂·r⁴ + …`
3. Subtract `offset(r) − target_hu` from every voxel in the slice, so the
   water background sits at `target_hu = 0` HU.

Runs via `BS.apply_radial_cupping_correction!` (src/correction/
beam_hardening_correction.jl:790).  Applied per-energy because each VMI
has its own HU scaling — a single basis-image correction wouldn't
generalise cleanly to every target E.

Output: `sim_vmi_flat.volumes` — capping-corrected HU stacks, same shape /
indexing as `sim_vmi.volumes`.  Downstream Mono+ / ROI / comparison cells
consume `sim_vmi_flat` so the flat background propagates.
"""

# ╔═╡ 000a0011-0000-4000-8000-000000000004
# ── Post-VMI radial capping hyperparameters ──────────────────────────────────
#
#   vmi_cap_enable    — master switch; false = pass raw VMI through
#   vmi_cap_fov_cm    — transverse FOV in cm (pixel→cm scale for the fit)
#   vmi_cap_hu_lo/hi  — HU range for water-like background voxels
#                        [-100, 80] excludes air (~-1000) + inserts (≫100)
#   vmi_cap_poly_order — even-polynomial terms beyond c₀
#                        1 ⇒ c₀ + c₁r²       (pure parabolic cupping)
#                        2 ⇒ c₀ + c₁r² + c₂r⁴  (default, matches clinical 06)
#                        3+ for stronger radial structure
#   vmi_cap_target_hu — target HU for water background after correction (0.0)
begin
    vmi_cap_enable     = true
    vmi_cap_fov_cm     = 35.0
    vmi_cap_hu_lo      = -100.0
    vmi_cap_hu_hi      = 80.0
    vmi_cap_poly_order = 2
    vmi_cap_target_hu  = 0.0
end

# ╔═╡ 000a0012-0000-4000-8000-000000000004
# Apply radial cupping correction to each VMI HU volume.
#   sim_vmi_flat.volumes[k] = capping(sim_vmi.volumes[k])   per energy
sim_vmi_flat = let
    if !vmi_cap_enable
        @info "Post-VMI radial capping: DISABLED (pass-through)"
        (energies = sim_vmi.energies,
         volumes  = [copy(v) for v in sim_vmi.volumes])
    else
        t0 = time()
        vols = Vector{Array{Float32, 3}}(undef, length(sim_vmi.volumes))
        for (k, v) in enumerate(sim_vmi.volumes)
            vols[k] = copy(v)
            BS.apply_radial_cupping_correction!(vols[k];
                fov_cm     = Float64(vmi_cap_fov_cm),
                hu_lo      = Float64(vmi_cap_hu_lo),
                hu_hi      = Float64(vmi_cap_hu_hi),
                poly_order = Int(vmi_cap_poly_order),
                target_hu  = Float64(vmi_cap_target_hu),
            )
        end
        dt = time() - t0
        @info "Post-VMI radial capping: $(length(vols)) energies in $(round(dt, digits=1)) s  [fov=$(vmi_cap_fov_cm) cm, HU ∈ [$(vmi_cap_hu_lo), $(vmi_cap_hu_hi)], poly=$(vmi_cap_poly_order), target=$(vmi_cap_target_hu)]"
        (energies = sim_vmi.energies, volumes = vols)
    end
end

# ╔═╡ 000a0013-0000-4000-8000-000000000004
# VMI (raw) vs VMI (capping-corrected) side-by-side at each energy.
let
    energies = sim_vmi.energies
    fig = CM.Figure(size = (1100, 1900), fontsize = 13)
    mid = size(sim_vmi.volumes[1], 3) ÷ 2
    for (i, E) in enumerate(energies)
        ax1 = CM.Axis(fig[i, 1]; title = "VMI raw  $(Int(E)) keV", aspect = CM.DataAspect())
        CM.heatmap!(ax1, sim_vmi.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))

        ax2 = CM.Axis(fig[i, 2]; title = "VMI + capping  $(Int(E)) keV  (poly=$(vmi_cap_poly_order))",
                      aspect = CM.DataAspect())
        CM.heatmap!(ax2, sim_vmi_flat.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))
    end
    CM.Label(fig[0, :]; text = "VMI raw vs VMI + radial capping  —  Soft Tissue Window (−200 … 500 HU)",
             fontsize = 15, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "vmi_raw_vs_capped.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000a0005-0000-4000-8000-000000000004
md"""
### 6.3 Mono+ / VMI+ — Grant 2014 1:1 parity

This is a straight port of **[G14] §Technique for Calculating Mono+ Images**.
Where the paper is explicit we copy it exactly; where it's silent we make a
single, documented best guess.

#### What the paper says verbatim

> "…low-keV images (in which iodine pixels have a high contrast to the
> surrounding tissue) and images of optimal keV from a noise perspective
> (typically, minimum image noise is obtained at approximately 70 keV)
> are computed. By means of a frequency-split technique, both the low-keV
> images and the images with minimum image noise are decomposed into 2
> sets of subimages. The first set contains only lower spatial frequencies
> and, hence, most of the object information, the second set contains the
> remaining high spatial frequencies and, hence, mostly image noise.
> Finally, the lower spatial frequency stack at low keV is combined with
> the high spatial frequency stack at optimal keV from a noise perspective…"

Extracted hard constraints:

| # | Specification (paper) | Implementation |
|---|----------------------|----------------|
| 1 | Two inputs: `VMI_target` + `VMI_opt`                                       | ✓ Read from `sim_vmi.volumes` at `E_target` and `E_noise_opt` |
| 2 | Noise-optimal energy ≈ 70 keV                                              | ✓ `vmip_E_noise_opt = 70.0` |
| 3 | Frequency-split decomposition of **both** images into LP + HP subimages    | ✓ Same linear LP applied to both branches |
| 4 | Final = LP(VMI_target) + HP(VMI_opt)                                       | ✓ `Mono+(E) = LP(VMI_E) + (VMI_opt − LP(VMI_opt))` |
| 5 | LP + HP are complementary (partition the frequency axis)                   | ✓ HP ≡ I − LP by construction (linear LP ⇒ complementary) |

#### What the paper does NOT specify — documented best guesses

| Decision | Paper | **Best guess** | Rationale |
|----------|-------|----------------|-----------|
| LP filter shape           | Unstated (proprietary) — abstract mentions *"regional spatial frequency-based recombination"* but body text only says "frequency-split" | **2D Gaussian LP** (FFT-diagonal) | Simplest linear LP that cleanly partitions the frequency axis; "regional" in the abstract reads as *spatial-domain* rather than spatially-adaptive; matches the plain "frequency-split" language |
| LP kernel size (σ)        | Unstated                                         | **σ = 2.0 px**                        | Rough match to the scale where image noise lives (~1–2 px) and object features (≥5 px) survive; tunable via `vmip_σ_lp_px` |
| Same LP for both branches | Strongly implied ("both…decomposed into 2 sets") | **Yes, identical σ and kernel**       | Only way LP + HP = I across both branches → Mono+(E_opt) = VMI_opt identically (sanity check) |
| Any pre-denoising of VMI_opt | Not mentioned                                  | **None**                              | Paper treats VMI_opt as-is; strict parity ⇒ no hidden preconditioning |
| 2D vs 3D LP               | Paper figures are axial 2D                       | **Per-slice 2D**                      | Matches clinical axial viewing; 3D LP would couple across slice thickness, not described by paper |
| Boundary conditions       | Unstated                                         | **FFT periodic**                      | No artifacts in body interior; tiny ring at scanner-bore edge is masked by the FOV circle anyway |

#### Final formula

    Mono+(E_target)  =  LP_σ(VMI_E_target)  +  VMI_opt  −  LP_σ(VMI_opt)

where `LP_σ` is a 2D Gaussian low-pass with std-dev `σ` pixels. At
`E_target = E_noise_opt` the formula collapses to `Mono+(E_opt) = VMI_opt`
exactly — a good consistency check.

#### Expected result ([G14] Figs 4–5)

Mono+ CNR rises monotonically as keV drops, peaking at 40 keV. Plain VMI
CNR peaks near 80 keV and drops at low keV because noise grows faster
than iodine contrast.
"""

# ╔═╡ 6454952b-4cdb-4beb-979e-c41595dbe204
# VMI+ / Mono+ — 1:1 port of Grant et al. 2014 [G14] §Technique for
# Calculating Mono+ Images.
#
# Paper specs (HARD constraints — faithfully implemented):
#   · Inputs: VMI at target keV + VMI at noise-optimal keV (~70 keV)
#   · Frequency-split decomposition of BOTH images into LP + HP subimages
#   · Combination: Mono+(E) = LP(VMI_E) + HP(VMI_opt)
#   · LP + HP complementary (so Mono+(E_opt) = VMI_opt identically)
#
# Paper is SILENT on (best guesses documented below, all tunable):
#   · LP filter shape      →  [BEST GUESS] 2D Gaussian LP via FFT diagonal
#   · LP kernel size       →  [BEST GUESS] σ = `vmip_σ_lp_px` pixels
#   · Same LP for both?    →  YES (strongly implied by paper wording)
#   · Pre-denoise VMI_opt? →  NO (strict parity → use as-is)
#   · 2D per-slice vs 3D?  →  [BEST GUESS] per-slice 2D (matches paper figs)
#   · Boundary conditions  →  [BEST GUESS] FFT periodic
#
# Input: sim_vmi_flat (capping-corrected VMI) — Mono+ inherits the flat
# radial background from §6.2 so downstream bias stays zero.
begin
    # Noise-optimal energy. [G14] body: "approximately 70 keV". MUST be in
    # sim_vmi_flat.energies.
    vmip_E_noise_opt = 70.0

    # LP Gaussian σ in pixels. [BEST GUESS — paper unspecified.] Typical
    # values: σ ≈ 1.0 preserves most detail (mild mix), σ ≈ 3.0 heavily
    # hands HF to VMI_opt (aggressive denoising at low-keV), σ ≈ 2.0 is
    # balanced. Effective HP cutoff frequency ≈ 1/(2πσ) cycles/pixel.
    vmip_σ_lp_px = 1.5
end

# ╔═╡ 000b0002-0000-4000-8000-000000000004
# Mono+ frequency-split via FFT Gaussian LP — Grant 2014 1:1 parity.
# Mono+(E) = LP_σ(VMI_E) + VMI_opt − LP_σ(VMI_opt). Identity at E_opt.
sim_vmi_plus = BS.apply_mono_plus(
    sim_vmi_flat.volumes, sim_vmi_flat.energies;
    E_noise_opt = vmip_E_noise_opt,
    σ_lp_px     = vmip_σ_lp_px,
)

# ╔═╡ 000b0003-0000-4000-8000-000000000004
# VMI (capping-corrected) vs Mono+ side-by-side at each energy — soft tissue window.
let
    fig = CM.Figure(size = (1100, 1900), fontsize = 13)
    mid = size(sim_vmi_flat.volumes[1], 3) ÷ 2
    for (i, E) in enumerate(sim_vmi_flat.energies)
        ax1 = CM.Axis(
            fig[i, 1]; title = "VMI + capping  $(Int(E)) keV",
            aspect = CM.DataAspect()
        )
        CM.heatmap!(ax1, sim_vmi_flat.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))

        ax2 = CM.Axis(
            fig[i, 2]; title = "Mono+ $(Int(E)) keV  (σ_LP=$(sim_vmi_plus.σ_lp_px) px, HP@$(Int(sim_vmi_plus.E_noise_opt)) keV)",
            aspect = CM.DataAspect()
        )
        CM.heatmap!(ax2, sim_vmi_plus.volumes[i][:, :, mid];
                    colormap = :grays, colorrange = (-200, 500))
    end
    CM.Label(fig[0, :]; text = "VMI + capping vs Mono+ (Grant 2014)  —  Soft Tissue Window (−200 … 500 HU)",
             fontsize = 15, font = :bold)
    CM.save(joinpath(RESULTS_DIR, "vmi_vs_vmi_plus.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 000b0004-0000-4000-8000-000000000004
# Water-ROI noise — VMI raw vs VMI + capping vs Mono+.  All three curves
# should decrease monotonically with keV; Mono+ should sit ~flat after
# frequency-split against the 70 keV noise-optimal energy.
let
    mid = size(sim_vmi.volumes[1], 3) ÷ 2
    nx, ny = size(sim_vmi.volumes[1], 1), size(sim_vmi.volumes[1], 2)
    cx, cy = nx ÷ 2, ny ÷ 2
    r = 15
    mask_idx = [(dx, dy) for dy in -r:r for dx in -r:r if dx^2 + dy^2 <= r^2]

    function water_σ(vol)
        vals = [vol[cx + dx, cy + dy, mid] for (dx, dy) in mask_idx]
        std(vals)
    end

    σ_raw    = [water_σ(v) for v in sim_vmi.volumes]
    σ_capped = [water_σ(v) for v in sim_vmi_flat.volumes]
    σ_plus   = [water_σ(v) for v in sim_vmi_plus.volumes]

    fig = CM.Figure(size = (800, 500), fontsize = 13)
    ax = CM.Axis(
        fig[1, 1]; title = "Central-water-ROI σ — VMI raw vs VMI + capping vs Mono+",
        xlabel = "Energy (keV)", ylabel = "σ (HU)",
        xticks = (1:length(sim_vmi.energies), string.(Int.(sim_vmi.energies)))
    )
    CM.scatterlines!(ax, 1:length(sim_vmi.energies), σ_raw;    color = :firebrick,  linewidth = 2, markersize = 10, label = "VMI raw")
    CM.scatterlines!(ax, 1:length(sim_vmi.energies), σ_capped; color = :darkorange, linewidth = 2, markersize = 10, label = "VMI + capping")
    CM.scatterlines!(ax, 1:length(sim_vmi.energies), σ_plus;   color = :steelblue,  linewidth = 2, markersize = 10, label = "Mono+ (σ_LP=$(sim_vmi_plus.σ_lp_px) px)")
    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "vmi_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 06126004-0000-4000-8000-000000000000
# Backward-compat wrapper: build sim_de_mono_plus in the shape §13's downstream
# clinical comparison cells expect (array of tuples with .recon/.name/.energy_keV).
# Uses the Mono+ output (sim_vmi_plus) so ROI / HU scatter / line profiles
# inherit the frequency-split noise shaping on top of the flat-background VMI.
# Applies the same orientation transform used on SE/DE recons (sim_de_oriented).
sim_de_mono_plus = let
    orient_fn = s -> reverse(s, dims = 2)
    [
        (name = "VMI_$(Int(E))keV",
         recon = Float32.(mapslices(orient_fn, sim_vmi_plus.volumes[i], dims = (1, 2))),
         energy_keV = Int(E))
            for (i, E) in enumerate(sim_vmi_plus.energies)
    ]
end;

# ╔═╡ c3bafd40-fda9-4ec2-8ec3-8dc109fc4ecb
sim_de_vmi_measurements = let
    vmi_scans = [(r.recon, r.name) for r in sim_de_mono_plus]

    # Segment on mid-slice of 100 keV Sim VMI (index 3 = 100 keV)
    vmi_100 = sim_de_mono_plus[3].recon
    mid_z = size(vmi_100, 3) ÷ 2 + 1
    mask, rods, center = segment_gammex_rods(vmi_100[:, :, mid_z]; fov_cm = 35.0)

    [
        measure_scan(vol, mask, rods, center, name)
            for (vol, name) in vmi_scans
    ]
end;

# ╔═╡ 3a1f9c02-de47-4a8b-b1e3-f8c7d2e10a01
# VMI Segmentation Check — Clinical vs Simulated ROI overlay on 100 keV VMI
let
    # Clinical: segment on DE 100 keV VMI
    clin_vol = hu_de_100keV
    clin_mid_z = size(clin_vol, 3) ÷ 2
    clin_slice = clin_vol[:, :, clin_mid_z]
    clin_mask, clin_rods, clin_center = segment_gammex_rods(clin_slice; fov_cm = 35.0)

    # Simulated: segment on DE 100 keV VMI
    sim_vol = sim_de_mono_plus[3].recon  # index 3 = 100 keV
    sim_mid_z = size(sim_vol, 3) ÷ 2 + 1
    sim_slice = sim_vol[:, :, sim_mid_z]
    sim_mask, sim_rods, sim_center = segment_gammex_rods(sim_slice; fov_cm = 35.0)

    pixel_cm_c = 35.0 / size(clin_slice, 1)
    pixel_cm_s = 35.0 / size(sim_slice, 1)
    roi_r_c = 1.4 * 0.7 / pixel_cm_c
    roi_r_s = 1.4 * 0.7 / pixel_cm_s
    th = range(0, 2 * pi, length = 61)

    fig = CM.Figure(size = (1100, 500), fontsize = 11)

    # Clinical with ROIs
    ax1 = CM.Axis(
        fig[1, 1]; title = "Clinical VMI 100 keV — Segmentation",
        aspect = CM.DataAspect(), yreversed = true
    )
    CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = (-200, 500))
    for r in clin_rods
        xs = r.cx .+ roi_r_c .* cos.(th)
        ys = r.cy .+ roi_r_c .* sin.(th)
        c = r.ring == :outer ? :orange : :lime
        CM.lines!(ax1, xs, ys; color = c, linewidth = 1.5)
        CM.text!(
            ax1, r.cx, r.cy + roi_r_c + 4;
            text = r.name, fontsize = 7, align = (:center, :bottom), color = c
        )
    end
    CM.scatter!(
        ax1, [clin_center.cx], [clin_center.cy];
        color = :red, marker = :cross, markersize = 12
    )
    CM.hidedecorations!(ax1); CM.hidespines!(ax1)

    # Simulated with ROIs
    ax2 = CM.Axis(
        fig[1, 2]; title = "Sim VMI 100 keV — Segmentation",
        aspect = CM.DataAspect(), yreversed = true
    )
    CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = (-200, 500))
    for r in sim_rods
        xs = r.cx .+ roi_r_s .* cos.(th)
        ys = r.cy .+ roi_r_s .* sin.(th)
        c = r.ring == :outer ? :orange : :lime
        CM.lines!(ax2, xs, ys; color = c, linewidth = 1.5)
        CM.text!(
            ax2, r.cx, r.cy + roi_r_s + 4;
            text = r.name, fontsize = 7, align = (:center, :bottom), color = c
        )
    end
    CM.scatter!(
        ax2, [sim_center.cx], [sim_center.cy];
        color = :red, marker = :cross, markersize = 12
    )
    CM.hidedecorations!(ax2); CM.hidespines!(ax2)

    CM.save(joinpath(RESULTS_DIR, "ge_vmi_segmentation_check.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 03c1d3a1-5604-4b50-b4e9-117260a23cf4
md"""
### VMI: Qualitative (Clinical vs Simulated)
"""

# ╔═╡ 08d5d8aa-bc95-427b-8a6a-d429881f6034
# VMI Qualitative Comparison — Clinical vs Simulated VMI at each energy
let
    clinical = [hu_de_40keV, hu_de_70keV, hu_de_100keV, hu_de_140keV]
    simulated = sim_de_mono_plus
    n = length(DE_VMI_ENERGIES)

    fig = CM.Figure(size = (500, 1000), fontsize = 10)
    for (i, E) in enumerate(DE_VMI_ENERGIES)
        clin_mid = size(clinical[i], 3) ÷ 2 + 0
        sim_mid = size(simulated[i].recon, 3) ÷ 2 + 0
        clin_slice = clinical[i][:, :, clin_mid]
        sim_slice = simulated[i].recon[:, :, sim_mid]

        ax1 = CM.Axis(fig[i, 1]; title = "Clinical VMI $(E) keV", yreversed = true)
        CM.heatmap!(ax1, clin_slice; colormap = :grays, colorrange = (-200, 500))
        CM.hidedecorations!(ax1); CM.hidespines!(ax1)

        ax2 = CM.Axis(fig[i, 2]; title = "Sim VMI $(E) keV", yreversed = true)
        CM.heatmap!(ax2, sim_slice; colormap = :grays, colorrange = (-200, 500))
        # CM.heatmap!(ax2, sim_slice; colormap = :grays)
        CM.hidedecorations!(ax2); CM.hidespines!(ax2)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_vmi_qualitative.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ b922a52a-b4f8-4385-b5a0-7b8eb69e8cfe
md"""
### VMI: Line Profiles (Clinical vs Simulated VMI)
"""

# ╔═╡ 157920c7-9a64-4407-8bd5-398f15e4842d
# VMI Line Profiles — Clinical vs Simulated VMI horizontal profile at each energy
let
    clinical = [hu_de_40keV, hu_de_70keV, hu_de_100keV, hu_de_140keV]
    simulated = sim_de_mono_plus
    n_vmi = length(DE_VMI_ENERGIES)

    sim_shift_px = 4.0  # pixel offset to align simulated with clinical

    fig = CM.Figure(size = (900, 1200), fontsize = 11)

    for (i, E) in enumerate(DE_VMI_ENERGIES)
        clin_vol = clinical[i]
        sim_vol = simulated[i].recon
        clin_mid_z = size(clin_vol, 3) ÷ 2
        sim_mid_z = size(sim_vol, 3) ÷ 2 + 1
        clin_slice = clin_vol[:, :, clin_mid_z]
        sim_slice = sim_vol[:, :, sim_mid_z]

        mid_row_c = size(clin_slice, 1) ÷ 2
        mid_row_s = size(sim_slice, 1) ÷ 2

        pixel_mm_c = 350.0 / size(clin_slice, 2)
        pixel_mm_s = 350.0 / size(sim_slice, 2)
        x_c = range(0, step = pixel_mm_c, length = size(clin_slice, 2))
        x_s = range(sim_shift_px * pixel_mm_s, step = pixel_mm_s, length = size(sim_slice, 2))

        ax = CM.Axis(
            fig[i, 1];
            title = "VMI $(E) keV — Horizontal Line Profile",
            xlabel = i == n_vmi ? "Position (mm)" : "",
            ylabel = "HU"
        )
        CM.lines!(
            ax, collect(x_c), Float64.(clin_slice[mid_row_c, :]);
            color = :steelblue, linewidth = 1.2, label = "Clinical"
        )
        CM.lines!(
            ax, collect(x_s), Float64.(sim_slice[mid_row_s, :]);
            color = :orangered, linewidth = 1.2, label = "Sim VMI"
        )
        CM.hlines!(ax, [0.0]; color = :gray70, linestyle = :dash, linewidth = 0.6)
        CM.axislegend(ax; position = :rt, labelsize = 8)
        CM.ylims!(ax, low = -600)
    end

    CM.save(joinpath(RESULTS_DIR, "ge_vmi_line_profiles.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ bf457bc7-9349-4b2f-b278-ef355f98cede
md"""
### VMI: Scatter Plot (Clinical vs Simulated VMI)
"""

# ╔═╡ 8e37657c-d02f-4b74-aba8-73299fd705c9
# VMI: Ca/I scatter — Clinical VMI vs Simulated VMI across energies
let
    vmi_labels = ["$(E) keV" for E in DE_VMI_ENERGIES]
    n_vmi = length(DE_VMI_ENERGIES)
    colors = CM.cgrad(:tab10, max(n_vmi, 2), categorical = true)

    fig = CM.Figure(size = (750, 900), fontsize = 11)

    # --- Top: Calcium rods ---
    ax_ca = CM.Axis(
        fig[1, 1], title = "Calcium Rods", subtitle = "Clinical VMI vs Simulated",
        xlabel = "Clinical HU", ylabel = "Sim VMI HU"
    )
    ca_clin_all, ca_sim_all = Float64[], Float64[]

    for (k, E) in enumerate(DE_VMI_ENERGIES)
        cm = de_measurements[k]
        sm = sim_de_vmi_measurements[k]
        ca_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "Ca")]
        if !isempty(ca_idx)
            CM.scatter!(
                ax_ca, cm.rod_means[ca_idx], sm.rod_means[ca_idx];
                color = colors[k], markersize = 10, label = vmi_labels[k]
            )
            append!(ca_clin_all, cm.rod_means[ca_idx])
            append!(ca_sim_all, sm.rod_means[ca_idx])
        end
    end
    CM.lines!(ax_ca, [-300, 3000], [-300, 3000], color = :gray60, linestyle = :dash, label = "Unity (y=x)")
    if length(ca_clin_all) > 1
        b_ca, m_ca = hcat(ones(length(ca_clin_all)), ca_clin_all) \ ca_sim_all
        r_ca = cor(ca_clin_all, ca_sim_all)
        rmse_ca = sqrt(sum((ca_sim_all .- ca_clin_all) .^ 2) / length(ca_clin_all))
        nrmse_ca = rmse_ca / (maximum(ca_clin_all) - minimum(ca_clin_all)) * 100
        eq_ca = "y = $(round(m_ca, digits = 3))x $(b_ca >= 0 ? "+" : "-") $(round(abs(b_ca), digits = 1))"
        CM.lines!(ax_ca, [extrema(ca_clin_all)...], m_ca .* [extrema(ca_clin_all)...] .+ b_ca, color = :black, linewidth = 0.8, label = "Linear fit")
        CM.poly!(ax_ca, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)], color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
        CM.text!(ax_ca, 0.62, 0.18, text = "$(eq_ca)\nr = $(round(r_ca, digits = 4))\nnRMSE = $(round(nrmse_ca, digits = 1))%", space = :relative, align = (:left, :top), fontsize = 10)
    end
    CM.axislegend(ax_ca, position = :lt, labelsize = 9)

    # --- Bottom: Iodine rods ---
    ax_i = CM.Axis(
        fig[2, 1], title = "Iodine Rods", subtitle = "Clinical VMI vs Simulated",
        xlabel = "Clinical HU", ylabel = "Sim VMI HU"
    )
    i_clin_all, i_sim_all = Float64[], Float64[]

    for (k, E) in enumerate(DE_VMI_ENERGIES)
        cm = de_measurements[k]
        sm = sim_de_vmi_measurements[k]
        i_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "I ")]
        if !isempty(i_idx)
            CM.scatter!(
                ax_i, cm.rod_means[i_idx], sm.rod_means[i_idx];
                color = colors[k], markersize = 10, label = vmi_labels[k]
            )
            append!(i_clin_all, cm.rod_means[i_idx])
            append!(i_sim_all, sm.rod_means[i_idx])
        end
    end
    CM.lines!(ax_i, [-150, 1500], [-150, 1500], color = :gray60, linestyle = :dash, label = "Unity (y=x)")
    if length(i_clin_all) > 1
        b_i, m_i = hcat(ones(length(i_clin_all)), i_clin_all) \ i_sim_all
        r_i = cor(i_clin_all, i_sim_all)
        rmse_i = sqrt(sum((i_sim_all .- i_clin_all) .^ 2) / length(i_clin_all))
        nrmse_i = rmse_i / (maximum(i_clin_all) - minimum(i_clin_all)) * 100
        eq_i = "y = $(round(m_i, digits = 3))x $(b_i >= 0 ? "+" : "-") $(round(abs(b_i), digits = 1))"
        CM.lines!(ax_i, [extrema(i_clin_all)...], m_i .* [extrema(i_clin_all)...] .+ b_i, color = :black, linewidth = 0.8, label = "Linear fit")
        CM.poly!(ax_i, CM.Point2f[(0.6, 0.02), (0.98, 0.02), (0.98, 0.22), (0.6, 0.22)], color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
        CM.text!(ax_i, 0.62, 0.18, text = "$(eq_i)\nr = $(round(r_i, digits = 4))\nnRMSE = $(round(nrmse_i, digits = 1))%", space = :relative, align = (:left, :top), fontsize = 10)
    end
    CM.axislegend(ax_i, position = :lt, labelsize = 9)

    CM.save(joinpath(RESULTS_DIR, "ge_vmi_scatter_hu.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 20ccfc65-0e2a-4d50-9460-bd64b29d2cc8
md"""
### VMI: Noise (Clinical vs Simulated VMI)
"""

# ╔═╡ 55ba2d06-51fe-4ba3-b428-78bcf6107b9b
# VMI Noise — Clinical vs Simulated VMI water σ at each VMI energy
let
    n_vmi = length(DE_VMI_ENERGIES)
    vmi_labels = ["$(E) keV" for E in DE_VMI_ENERGIES]

    clin_σ = [de_measurements[i].rod_stds[1] for i in 1:n_vmi]  # water rod = index 1
    sim_σ = [sim_de_vmi_measurements[i].rod_stds[1] for i in 1:n_vmi]

    fig = CM.Figure(size = (800, 400), fontsize = 13)

    ax = CM.Axis(
        fig[1, 1]; title = "VMI Water Noise — Clinical vs Simulated",
        ylabel = "Water σ (HU)", xlabel = "VMI Energy",
        xticks = (1:n_vmi, vmi_labels)
    )
    CM.barplot!(
        ax, collect(1:n_vmi) .- 0.2, clin_σ; width = 0.35,
        color = :steelblue, label = "Clinical VMI"
    )
    CM.barplot!(
        ax, collect(1:n_vmi) .+ 0.2, sim_σ; width = 0.35,
        color = :darkorange, label = "Sim VMI"
    )

    # Annotate bars with σ values
    for i in 1:n_vmi
        CM.text!(
            ax, i - 0.2, clin_σ[i] + 0.5;
            text = "$(round(clin_σ[i], digits = 1))", align = (:center, :bottom), fontsize = 9
        )
        CM.text!(
            ax, i + 0.2, sim_σ[i] + 0.5;
            text = "$(round(sim_σ[i], digits = 1))", align = (:center, :bottom), fontsize = 9
        )
    end

    CM.axislegend(ax; position = :rt)
    CM.save(joinpath(RESULTS_DIR, "ge_vmi_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 314c530e-caf1-4235-9985-05e1ef81ccd1
md"""
### VMI: NPS (Clinical vs Simulated VMI)
"""

# ╔═╡ b5c7a40b-2e23-4a85-a734-b8dc86949b7f
# VMI NPS comparison — Clinical vs Simulated VMI at each energy
let
    fig = CM.Figure(size = (900, 900), fontsize = 11)

    for (i, E) in enumerate(DE_VMI_ENERGIES)
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        ax = CM.Axis(
            fig[row, col]; title = "VMI $(E) keV",
            subtitle = "Clinical vs Simulated VMI",
            xlabel = "Spatial frequency (lp/cm)", ylabel = "nNPS (A.U.)"
        )
        cm = de_measurements[i]
        sm = sim_de_vmi_measurements[i]
        f_c, v_c = cm.nps.frequencies, cm.nps.nps_1d
        good_c = v_c .> 0
        CM.lines!(
            ax, cm.nps.frequencies, cm.nps.nnps_1d;
            color = :steelblue, linewidth = 1.5, label = "Clinical"
        )
        f_s, v_s = sm.nps.frequencies, sm.nps.nps_1d
        good_s = v_s .> 0
        CM.lines!(
            ax, sm.nps.frequencies, sm.nps.nnps_1d;
            color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Sim VMI"
        )
        CM.axislegend(ax; position = :rt, labelsize = 8)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_vmi_nps.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 6ee9f9ae-977a-4782-b435-9a28bf45346c
md"""
### VMI: MTF (Clinical vs Simulated VMI)
"""

# ╔═╡ e6439f40-ade4-4de9-817c-96663a5ae453
# VMI MTF comparison — Clinical vs Simulated VMI at each energy
let
    fig = CM.Figure(size = (900, 900), fontsize = 11)

    for (i, E) in enumerate(DE_VMI_ENERGIES)
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        ax = CM.Axis(
            fig[row, col]; title = "VMI $(E) keV",
            subtitle = "Clinical vs Simulated VMI",
            xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
            limits = (nothing, nothing, 0, 1.05)
        )
        CM.hlines!(ax, [0.5, 0.1]; color = :gray80, linestyle = :dash, linewidth = 0.8)
        cm = de_measurements[i]
        sm = sim_de_vmi_measurements[i]
        CM.lines!(
            ax, cm.mtf.frequencies, cm.mtf.mtf;
            color = :steelblue, linewidth = 1.5, label = "Clinical"
        )
        CM.lines!(
            ax, sm.mtf.frequencies, sm.mtf.mtf;
            color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Sim VMI"
        )
        CM.axislegend(ax; position = :rt, labelsize = 8)
    end
    CM.save(joinpath(RESULTS_DIR, "ge_vmi_mtf.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ a89fe53a-c546-4036-b7e9-1be92206cd62
md"""
### Final Export — All Measurements (Clinical + Simulated VMI)
"""

# ╔═╡ 8cf1b41b-105d-4dc5-952a-cedfc4c8f4ae
# Comprehensive export: all clinical + simulated measurements → CSV + JLD2
let
    tagged = vcat(
        [(m, "clinical_SE") for m in se_measurements],
        [(m, "clinical_DE_VMI") for m in de_measurements],
        [(m, "sim_FBP") for m in sim_measurements],
        [(m, "sim_HIR") for m in sim_measurements_hir],
        [(m, "sim_DE_VMI") for m in sim_de_vmi_measurements],
    )

    rod_order = tagged[1][1].rod_names
    header = ["scan_name", "category"]
    for nm in rod_order
        tag = replace(replace(nm, " " => "_"), "(" => "", ")" => "")
        push!(header, "hu_mean_$tag")
    end
    for nm in rod_order
        tag = replace(replace(nm, " " => "_"), "(" => "", ")" => "")
        push!(header, "hu_std_$tag")
    end
    for nm in rod_order
        tag = replace(replace(nm, " " => "_"), "(" => "", ")" => "")
        push!(header, "cnr_$tag")
    end
    append!(
        header, [
            "nps_peak_freq_lp_cm", "nps_area_HU2cm2",
            "mtf_f50_lp_cm", "mtf_f10_lp_cm",
        ]
    )

    rows = Vector{Any}[]
    for (m, cat) in tagged
        row = Any[m.name, cat]
        append!(row, round.(m.rod_means, digits = 2))
        append!(row, round.(m.rod_stds, digits = 2))
        append!(row, round.(m.rod_cnr, digits = 2))
        push!(row, round(m.nps_peak_freq, digits = 3))
        push!(row, round(m.nps_area, digits = 3))
        push!(row, round(m.mtf_f50, digits = 3))
        push!(row, round(m.mtf_f10, digits = 3))
        push!(rows, row)
    end

    csv_path = joinpath(RESULTS_DIR, "ge_apex_elite_all_measurements.csv")
    open(csv_path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end

    nps_path = joinpath(RESULTS_DIR, "ge_apex_elite_all_nps.jld2")
    JLD2.jldopen(nps_path, "w") do f
        for (m, cat) in tagged
            f["$(cat)/$(m.name)"] = (freq = m.nps.frequencies, nps = m.nps.nps_1d)
        end
    end

    mtf_path = joinpath(RESULTS_DIR, "ge_apex_elite_all_mtf.jld2")
    JLD2.jldopen(mtf_path, "w") do f
        for (m, cat) in tagged
            f["$(cat)/$(m.name)"] = (freq = m.mtf.frequencies, mtf = m.mtf.mtf)
        end
    end

    md"""
    **Exported (all measurements):**
    - `ge_apex_elite_all_measurements.csv` — $(length(tagged)) rows × $(length(header)) columns
    - `ge_apex_elite_all_nps.jld2` — NPS curves ($(length(tagged)) scans)
    - `ge_apex_elite_all_mtf.jld2` — MTF curves ($(length(tagged)) scans)
    """
end

# ╔═╡ e4d416cf-9085-4733-a04e-45b3a84f577e
md"""
## 14. Combined Analysis
"""

# ╔═╡ 79a6fd16-4c56-4bd4-98d4-889ddc2e1e01
# Combined FBP + HIR noise bar chart
let
    water_idx = 1

    # ── Dose Ladder data (120 kVp: 50, 150, 300 mA) ─────────────────
    dose_labels = [
        "120 kVp / 50 mA\n(3.38 mGy)",
        "120 kVp / 150 mA\n(10.16 mGy)",
        "120 kVp / 300 mA\n(20.38 mGy)",
    ]
    dose_sim_idx = [1, 2, 3]
    dose_clin_fbp_idx = [1, 3, 5]
    dose_clin_hir_idx = [2, 4, 6]
    n_dose = length(dose_sim_idx)

    dose_clin_fbp_σ = [se_measurements[dose_clin_fbp_idx[i]].rod_stds[water_idx] for i in 1:n_dose]
    dose_sim_fbp_σ = [sim_measurements[dose_sim_idx[i]].rod_stds[water_idx] for i in 1:n_dose]
    dose_clin_hir_σ = [se_measurements[dose_clin_hir_idx[i]].rod_stds[water_idx] for i in 1:n_dose]
    dose_sim_hir_σ = [sim_measurements_hir[dose_sim_idx[i]].rod_stds[water_idx] for i in 1:n_dose]

    # ── kVp Series data (~10 mGy) ───────────────────────────────────
    kvp_labels = [
        "80 kVp / 480 mA\n(10.32 mGy)",
        "100 kVp / 250 mA\n(10.53 mGy)",
        "120 kVp / 150 mA\n(10.16 mGy)",
        "140 kVp / 110 mA\n(10.85 mGy)",
    ]
    kvp_sim_idx = [4, 5, 2, 6]
    kvp_clin_fbp_idx = [7, 9, 3, 11]
    kvp_clin_hir_idx = [8, 10, 4, 12]
    n_kvp = length(kvp_sim_idx)

    kvp_clin_fbp_σ = [se_measurements[kvp_clin_fbp_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]
    kvp_sim_fbp_σ = [sim_measurements[kvp_sim_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]
    kvp_clin_hir_σ = [se_measurements[kvp_clin_hir_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]
    kvp_sim_hir_σ = [sim_measurements_hir[kvp_sim_idx[i]].rod_stds[water_idx] for i in 1:n_kvp]

    # ── Plot ─────────────────────────────────────────────────────────
    fig = CM.Figure(size = (1100, 800), fontsize = 13)
    bw = 0.18  # bar width
    sw = 2.0   # stroke width for HIR outline bars

    # Top: Dose Ladder
    ax1 = CM.Axis(
        fig[1, 1]; title = "120 kVp — Dose Ladder",
        ylabel = "Water σ (HU)", xticks = (1:n_dose, dose_labels)
    )
    x = collect(1:n_dose)
    CM.barplot!(
        ax1, x .- 0.27, dose_clin_fbp_σ; width = bw,
        color = :steelblue, label = "Clinical FBP"
    )
    CM.barplot!(
        ax1, x .- 0.09, dose_sim_fbp_σ; width = bw,
        color = :darkorange, label = "Simulated FBP"
    )
    CM.barplot!(
        ax1, x .+ 0.09, dose_clin_hir_σ; width = bw,
        color = (:steelblue, 0.15), strokecolor = :steelblue, strokewidth = sw,
        label = "Clinical ASiR-V 50%"
    )
    CM.barplot!(
        ax1, x .+ 0.27, dose_sim_hir_σ; width = bw,
        color = (:darkorange, 0.15), strokecolor = :darkorange, strokewidth = sw,
        label = "Simulated HIR"
    )
    CM.ylims!(ax1, 0, nothing)
    CM.axislegend(ax1; position = :rt)

    # Bottom: kVp Series
    ax2 = CM.Axis(
        fig[2, 1]; title = "~10 mGy CTDIvol — kVp Series",
        ylabel = "Water σ (HU)", xticks = (1:n_kvp, kvp_labels)
    )
    x2 = collect(1:n_kvp)
    CM.barplot!(
        ax2, x2 .- 0.27, kvp_clin_fbp_σ; width = bw,
        color = :steelblue, label = "Clinical FBP"
    )
    CM.barplot!(
        ax2, x2 .- 0.09, kvp_sim_fbp_σ; width = bw,
        color = :darkorange, label = "Simulated FBP"
    )
    CM.barplot!(
        ax2, x2 .+ 0.09, kvp_clin_hir_σ; width = bw,
        color = (:steelblue, 0.15), strokecolor = :steelblue, strokewidth = sw,
        label = "Clinical ASiR-V 50%"
    )
    CM.barplot!(
        ax2, x2 .+ 0.27, kvp_sim_hir_σ; width = bw,
        color = (:darkorange, 0.15), strokecolor = :darkorange, strokewidth = sw,
        label = "Simulated HIR"
    )
    CM.ylims!(ax2, 0, nothing)
    CM.axislegend(ax2; position = :rt)

    CM.save(joinpath(RESULTS_DIR, "ge_combined_noise.png"), fig, px_per_unit = 2)
    fig
end

# ╔═╡ 07140001-0000-4000-8000-000000000000
md"""
## 15. Appendix: Parameter Readout

Current tuning parameter values for quick inspection.
"""

# ╔═╡ 07140002-0000-4000-8000-000000000000
sim_electronic_noise

# ╔═╡ 07140003-0000-4000-8000-000000000000
sim_detection_gain

# ╔═╡ 07140004-0000-4000-8000-000000000000
sim_noise_floor_hu

# ╔═╡ Cell order:
# ╟─07010001-0000-4000-8000-000000000000
# ╠═07010002-0000-4000-8000-000000000000
# ╠═783d265d-506c-4dd4-aa76-bd352f532c6d
# ╠═07010003-0000-4000-8000-000000000000
# ╠═07010004-0000-4000-8000-000000000000
# ╠═07010005-0000-4000-8000-000000000000
# ╠═07010006-0000-4000-8000-000000000000
# ╠═07010007-0000-4000-8000-000000000000
# ╠═07010008-0000-4000-8000-000000000000
# ╠═07010009-0000-4000-8000-000000000000
# ╠═07010010-0000-4000-8000-000000000000
# ╠═07010011-0000-4000-8000-000000000000
# ╠═07010012-0000-4000-8000-000000000000
# ╠═07010013-0000-4000-8000-000000000000
# ╠═0701001a-0000-4000-8000-000000000000
# ╠═0701001b-0000-4000-8000-000000000000
# ╠═07010014-0000-4000-8000-000000000000
# ╠═07010015-0000-4000-8000-000000000000
# ╠═07010016-0000-4000-8000-000000000000
# ╠═07010017-0000-4000-8000-000000000000
# ╠═07010018-0000-4000-8000-000000000000
# ╠═a13bf90b-b0a1-4786-9554-132b1a346334
# ╟─07010019-0000-4000-8000-000000000000
# ╠═07010020-0000-4000-8000-000000000000
# ╟─07020001-0000-4000-8000-000000000000
# ╠═07020002-0000-4000-8000-000000000000
# ╠═07020003-0000-4000-8000-000000000000
# ╠═07020004-0000-4000-8000-000000000000
# ╠═07020005-0000-4000-8000-000000000000
# ╠═07020006-0000-4000-8000-000000000000
# ╠═07020007-0000-4000-8000-000000000000
# ╠═07020008-0000-4000-8000-000000000000
# ╠═07020009-0000-4000-8000-000000000000
# ╟─07030001-0000-4000-8000-000000000000
# ╠═07030002-0000-4000-8000-000000000000
# ╠═07030003-0000-4000-8000-000000000000
# ╠═07030004-0000-4000-8000-000000000000
# ╠═07030005-0000-4000-8000-000000000000
# ╠═07030006-0000-4000-8000-000000000000
# ╠═07030007-0000-4000-8000-000000000000
# ╠═07030008-0000-4000-8000-000000000000
# ╟─07030009-0000-4000-8000-000000000000
# ╟─07030010-0000-4000-8000-000000000000
# ╟─07030011-0000-4000-8000-000000000000
# ╟─07030012-0000-4000-8000-000000000000
# ╟─07030013-0000-4000-8000-000000000000
# ╟─07030014-0000-4000-8000-000000000000
# ╟─07040001-0000-4000-8000-000000000000
# ╠═07040002-0000-4000-8000-000000000000
# ╟─07040003-0000-4000-8000-000000000000
# ╟─07040004-0000-4000-8000-000000000000
# ╠═07040005-0000-4000-8000-000000000000
# ╟─07050001-0000-4000-8000-000000000000
# ╠═07050002-0000-4000-8000-000000000000
# ╠═07050003-0000-4000-8000-000000000000
# ╠═07050004-0000-4000-8000-000000000000
# ╠═07050005-0000-4000-8000-000000000000
# ╟─07050006-0000-4000-8000-000000000000
# ╟─07050007-0000-4000-8000-000000000000
# ╠═07050008-0000-4000-8000-000000000000
# ╟─07050009-0000-4000-8000-000000000000
# ╟─07050010-0000-4000-8000-000000000000
# ╠═07050011-0000-4000-8000-000000000000
# ╟─07060001-0000-4000-8000-000000000000
# ╟─07060002-0000-4000-8000-000000000000
# ╟─07060003-0000-4000-8000-000000000000
# ╠═07060004-0000-4000-8000-000000000000
# ╟─07060005-0000-4000-8000-000000000000
# ╟─07060006-0000-4000-8000-000000000000
# ╟─07070001-0000-4000-8000-000000000000
# ╠═07070002-0000-4000-8000-000000000000
# ╟─07070003-0000-4000-8000-000000000000
# ╟─07080001-0000-4000-8000-000000000000
# ╠═07080002-0000-4000-8000-000000000000
# ╠═07080003-0000-4000-8000-000000000000
# ╠═07080004-0000-4000-8000-000000000000
# ╠═07080005-0000-4000-8000-000000000000
# ╟─07080006-0000-4000-8000-000000000000
# ╠═07080007-0000-4000-8000-000000000000
# ╠═07080008-0000-4000-8000-000000000000
# ╠═07080009-0000-4000-8000-000000000000
# ╠═07080010-0000-4000-8000-000000000000
# ╟─07090001-0000-4000-8000-000000000000
# ╠═07090002-0000-4000-8000-000000000000
# ╠═07090003-0000-4000-8000-000000000000
# ╠═07090004-0000-4000-8000-000000000000
# ╠═07090005-0000-4000-8000-000000000000
# ╠═07090006-0000-4000-8000-000000000000
# ╠═07090007-0000-4000-8000-000000000000
# ╟─07100001-0000-4000-8000-000000000000
# ╠═07100002-0000-4000-8000-000000000000
# ╠═07100003-0000-4000-8000-000000000000
# ╠═07100004-0000-4000-8000-000000000000
# ╠═07100005-0000-4000-8000-000000000000
# ╠═07100006-0000-4000-8000-000000000000
# ╠═07100007-0000-4000-8000-000000000000
# ╟─07100008-0000-4000-8000-000000000000
# ╠═07100009-0000-4000-8000-000000000000
# ╠═d5a0ffa5-1d40-4bfc-89de-e462c5358b24
# ╟─07110001-0000-4000-8000-000000000000
# ╠═07110002-0000-4000-8000-000000000000
# ╠═07110003-0000-4000-8000-000000000000
# ╠═07110004-0000-4000-8000-000000000000
# ╠═07110005-0000-4000-8000-000000000000
# ╠═07110006-0000-4000-8000-000000000000
# ╠═07110007-0000-4000-8000-000000000000
# ╠═07110008-0000-4000-8000-000000000000
# ╠═07110009-0000-4000-8000-000000000000
# ╠═07110010-0000-4000-8000-000000000000
# ╠═07110011-0000-4000-8000-000000000000
# ╠═07110012-0000-4000-8000-000000000000
# ╠═07110013-0000-4000-8000-000000000000
# ╠═07110020-0000-4000-8000-000000000000
# ╠═07110021-0000-4000-8000-000000000000
# ╠═07110022-0000-4000-8000-000000000000
# ╠═07110023-0000-4000-8000-000000000000
# ╟─07110030-0000-4000-8000-000000000000
# ╟─07110031-0000-4000-8000-000000000000
# ╟─8e294031-bcdb-4424-8936-a62a802d5aa1
# ╟─0403ce0e-1854-40e2-af6d-c53125aca803
# ╟─07110032-0000-4000-8000-000000000000
# ╟─07110033-0000-4000-8000-000000000000
# ╟─07110038-0000-4000-8000-000000000000
# ╟─07110039-0000-4000-8000-000000000000
# ╟─07110040-0000-4000-8000-000000000000
# ╟─e85ee5fb-67f6-43b8-91fd-6018bfc1535a
# ╟─07110042-0000-4000-8000-000000000000
# ╟─07110043-0000-4000-8000-000000000000
# ╟─6f63608c-dd46-4d0a-8f1d-133dc56bc8dd
# ╠═f61c87a4-a27e-4bc6-bc4d-502c3e3a47fe
# ╠═f925dafa-e0da-4673-8d84-82fe636bd6c2
# ╠═b84d07e5-64fd-4365-aca6-d03b4b5a1032
# ╠═12c84e0f-9cb4-49a0-a1fc-5cbe52b92f8a
# ╠═4ca8d674-a0c1-407a-ab1d-84f06d966159
# ╠═6331e7a1-6dce-43bf-97b9-4ecbf766d40c
# ╠═4917411f-f885-46bb-9f7e-a3de406d236c
# ╠═cad1f6b8-66fc-47f1-8386-e10c97ce0e4a
# ╠═00aeca04-810f-4543-bef1-70db235be25c
# ╠═cf576e1e-5ef4-4478-9209-f584c373ef49
# ╠═487bb689-9149-45e4-b153-2bad6138ebf5
# ╠═3793479d-79e9-4948-85c1-ff3c049f1cd0
# ╠═903f4e17-1824-418a-b8f9-21546163d740
# ╠═cbe20e99-ca03-4045-a898-2de307d19fce
# ╠═ed7cbb89-2f22-4587-ad7f-1f3224b73f80
# ╠═10a11afd-004a-467a-b808-7e761cb21678
# ╟─3924da10-33fd-415b-9666-f6022d0adaf6
# ╟─ebc9865c-feed-4e79-9cac-39e583a3796d
# ╟─1cae28b1-775a-4f4f-90d2-5ae693b7d2ac
# ╟─681131e3-887d-4426-aefe-f904abb4667a
# ╟─c42f625b-9db4-42a5-90c6-fa2c37fe8b18
# ╟─bfd647b4-44d7-4582-8b5b-dd5910efd729
# ╟─28e45f60-2542-4305-8c9c-3aecf1b3aa23
# ╟─d35a2ac4-467a-4ade-afaf-dc668880559a
# ╟─4776ad7f-edee-4c44-91bb-5ad76a63832e
# ╟─405ad9a1-3cac-4b19-87b3-1d041d8c0813
# ╟─d3bd445c-ad9f-4133-93d9-8af91ed7f85d
# ╟─5298fd86-136b-4af3-aa0f-be0e810f5d24
# ╟─4f8ee88d-a948-44e7-a610-a2711e6da234
# ╠═f9da78c0-9c6a-4c83-9f4e-592eecca5aaf
# ╠═3aefce3f-90e3-48fb-bae7-b7c965b7d6f8
# ╠═6f5ce242-a5e2-4cd5-8cac-47f528dc21bb
# ╠═ff650b6f-76e6-491e-8076-1dd002c80d7e
# ╠═2a6f0b64-05a7-4a01-9009-66888902edcb
# ╠═bb68196f-26e4-41fc-85bc-203373d91b4a
# ╠═37298bb1-4b57-4f5c-9fea-ec81dbfb3394
# ╠═fd123cde-bb22-489d-a496-e00363f46157
# ╠═89c1ff84-d4bc-47ca-b08d-8ada2bd469f4
# ╠═7e06d734-4023-4078-9b8f-3b55c2e389e6
# ╠═36986f73-d54b-4190-a52c-f3f23e53c533
# ╠═4efd48c9-753a-4418-8095-fa12b7cc5a95
# ╠═e8af3f62-e606-4f10-9a09-9e0620910f58
# ╟─5cc1fa5c-3722-4fa1-b0f0-d816e204c8bf
# ╟─00080001-0000-4000-8000-000000000004
# ╠═f3a91573-63b4-40df-8a9a-3a44d390bc79
# ╠═00080002-0000-4000-8000-000000000004
# ╠═00080003-0000-4000-8000-000000000004
# ╠═00080004-0000-4000-8000-000000000004
# ╟─00080005-0000-4000-8000-000000000004
# ╠═00080012-0000-4000-8000-000000000004
# ╟─00080013-0000-4000-8000-000000000004
# ╟─00080014-0000-4000-8000-000000000004
# ╠═00080021-0000-4000-8000-000000000004
# ╠═00080022-0000-4000-8000-000000000004
# ╟─00080015-0000-4000-8000-000000000004
# ╠═00080016-0000-4000-8000-000000000004
# ╟─c1139ae3-5186-445e-81b8-5d932ca5ef98
# ╟─00080017-0000-4000-8000-000000000004
# ╟─00080020-0000-4000-8000-000000000004
# ╠═00090002-0000-4000-8000-000000000004
# ╟─fd9969e0-415f-409d-8672-fe2d963b6486
# ╟─2b6fd506-c624-4be3-9a71-1d366ae58ada
# ╟─00090003-0000-4000-8000-000000000004
# ╟─000a0001-0000-4000-8000-000000000004
# ╠═000a0002-0000-4000-8000-000000000004
# ╟─000a0004-0000-4000-8000-000000000004
# ╟─000a0010-0000-4000-8000-000000000004
# ╠═000a0011-0000-4000-8000-000000000004
# ╠═000a0012-0000-4000-8000-000000000004
# ╟─000a0013-0000-4000-8000-000000000004
# ╟─000a0005-0000-4000-8000-000000000004
# ╠═6454952b-4cdb-4beb-979e-c41595dbe204
# ╠═000b0002-0000-4000-8000-000000000004
# ╟─000b0003-0000-4000-8000-000000000004
# ╟─000b0004-0000-4000-8000-000000000004
# ╠═06126004-0000-4000-8000-000000000000
# ╠═c3bafd40-fda9-4ec2-8ec3-8dc109fc4ecb
# ╟─3a1f9c02-de47-4a8b-b1e3-f8c7d2e10a01
# ╟─03c1d3a1-5604-4b50-b4e9-117260a23cf4
# ╟─08d5d8aa-bc95-427b-8a6a-d429881f6034
# ╟─b922a52a-b4f8-4385-b5a0-7b8eb69e8cfe
# ╟─157920c7-9a64-4407-8bd5-398f15e4842d
# ╟─bf457bc7-9349-4b2f-b278-ef355f98cede
# ╟─8e37657c-d02f-4b74-aba8-73299fd705c9
# ╟─20ccfc65-0e2a-4d50-9460-bd64b29d2cc8
# ╟─55ba2d06-51fe-4ba3-b428-78bcf6107b9b
# ╟─314c530e-caf1-4235-9985-05e1ef81ccd1
# ╟─b5c7a40b-2e23-4a85-a734-b8dc86949b7f
# ╟─6ee9f9ae-977a-4782-b435-9a28bf45346c
# ╟─e6439f40-ade4-4de9-817c-96663a5ae453
# ╟─a89fe53a-c546-4036-b7e9-1be92206cd62
# ╠═8cf1b41b-105d-4dc5-952a-cedfc4c8f4ae
# ╟─e4d416cf-9085-4733-a04e-45b3a84f577e
# ╟─79a6fd16-4c56-4bd4-98d4-889ddc2e1e01
# ╟─07140001-0000-4000-8000-000000000000
# ╠═07140002-0000-4000-8000-000000000000
# ╠═07140003-0000-4000-8000-000000000000
# ╠═07140004-0000-4000-8000-000000000000
