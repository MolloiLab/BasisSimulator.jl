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
# Siemens NAEOTOM Alpha — Clinical Gammex 472 Scans (PCCT)

* **Scanner:** Siemens NAEOTOM Alpha (Photon-Counting CT)
* **Phantom:** Gammex 472 multi-energy CT calibration phantom
* **Recon FOV:** 350 mm | **Matrix:** 512 × 512 | **Kernel:** Br36 | **QIR:** 3

Four axial scans, each reconstructed with various settings. Clinical data to be loaded after scan day.

| Scan | kVp | CTDIvol | Recon Series |
|------|-----|---------|--------------|
| 1 | 140 | ~3 mGy | Poly FBP + QIR 3 |
| 2 | 140 | ~10 mGy | Poly FBP+QIR; Low-thresh FBP+QIR; High-thresh FBP+QIR; VMI 40/70/100/140 keV (QIR 3) |
| 3 | 140 | ~20 mGy | Poly FBP + QIR 3 |
| 4 | 120 | ~10 mGy | Poly FBP + QIR 3 |

Common: Axial, 1.0 s rotation, 144 × 0.4 mm collimation, 512 × 512, 350 mm FOV, Br36, QIR 3
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
Uses ImageMagick.jl (via FileIO) for J2K decompression.
Note: May need adaptation for Siemens DICOM pixel encoding.
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
                            wx += si; wy += sj; wt += 1.0
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
## 3. Clinical Scan 1: 140 kVp / ~3 mGy

**Placeholder** — Load DICOM paths after scan day.
- Poly FBP + QIR 3
"""

# ╔═╡ 08030002-0000-4000-8000-000000000000
begin
    clinical_scan1_poly_fbp = nothing  # TODO: DCM.dcmdir_parse("path/to/scan1/poly_fbp")
    clinical_scan1_poly_qir = nothing  # TODO: DCM.dcmdir_parse("path/to/scan1/poly_qir3")
end

# ╔═╡ 08040001-0000-4000-8000-000000000000
md"""
## 4. Clinical Scan 2: 140 kVp / ~10 mGy

**Placeholder** — The most complex scan with poly, threshold-binned, and VMI reconstructions.
"""

# ╔═╡ 08040002-0000-4000-8000-000000000000
begin
    # Polyenergetic (energy-summed)
    clinical_scan2_poly_fbp = nothing  # TODO: load DICOM path
    clinical_scan2_poly_qir = nothing  # TODO: load DICOM path

    # Low-threshold bin (25–65 keV)
    clinical_scan2_low_fbp = nothing   # TODO: load DICOM path
    clinical_scan2_low_qir = nothing   # TODO: load DICOM path

    # High-threshold bin (>65 keV)
    clinical_scan2_high_fbp = nothing  # TODO: load DICOM path
    clinical_scan2_high_qir = nothing  # TODO: load DICOM path

    # VMI at 40, 70, 100, 140 keV (all QIR 3)
    clinical_scan2_vmi_40 = nothing    # TODO: load DICOM path
    clinical_scan2_vmi_70 = nothing    # TODO: load DICOM path
    clinical_scan2_vmi_100 = nothing   # TODO: load DICOM path
    clinical_scan2_vmi_140 = nothing   # TODO: load DICOM path
end

# ╔═╡ 08050001-0000-4000-8000-000000000000
md"""
## 5. Clinical Scan 3: 140 kVp / ~20 mGy

**Placeholder** — Load DICOM paths after scan day.
- Poly FBP + QIR 3
"""

# ╔═╡ 08050002-0000-4000-8000-000000000000
begin
    clinical_scan3_poly_fbp = nothing  # TODO: DCM.dcmdir_parse("path/to/scan3/poly_fbp")
    clinical_scan3_poly_qir = nothing  # TODO: DCM.dcmdir_parse("path/to/scan3/poly_qir3")
end

# ╔═╡ 08060001-0000-4000-8000-000000000000
md"""
## 6. Clinical Scan 4: 120 kVp / ~10 mGy

**Placeholder** — Load DICOM paths after scan day.
- Poly FBP + QIR 3
"""

# ╔═╡ 08060002-0000-4000-8000-000000000000
begin
    clinical_scan4_poly_fbp = nothing  # TODO: DCM.dcmdir_parse("path/to/scan4/poly_fbp")
    clinical_scan4_poly_qir = nothing  # TODO: DCM.dcmdir_parse("path/to/scan4/poly_qir3")
end

# ╔═╡ 08070001-0000-4000-8000-000000000000
md"""
## 7. Clinical Segmentation & Measurements

**Placeholder** — Run after DICOM data is loaded.
"""

# ╔═╡ 08070002-0000-4000-8000-000000000000
clinical_measurements = nothing  # TODO: segment + measure after DICOM loaded

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

**Key modification from defaults:** 2 energy thresholds (T1=25, T2=65 keV) matching clinical protocol, instead of the default 4-threshold configuration. This gives:
- Bin 1: 25–65 keV (low energy, sensitive to iodine K-edge at 33 keV)
- Bin 2: >65 keV (high energy)
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
    sid = 595.0              # source-to-isocenter (mm)
    sdd = 1085.5             # source-to-detector (mm)
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
        gantry_rotation_time = 1.0,   # matches clinical protocol
        scan_diameter = 500.0,
        gantry_aperture = 820.0,

        # FILTRATION
        flat_filter_material = :aluminum,
        flat_filter_thickness = 2.5,  # mm

        # DETECTOR PHYSICS — CdTe direct-conversion
        detector_material = :cdte,
        detector_depth = 1.6,         # mm (Konrad 2025)
        fill_factor_row = 0.95,
        fill_factor_col = 0.95,
        detection_gain = 1.0,         # direct conversion (no scintillator gain)
        electronic_noise = 0.0,       # PCCT thresholds eliminate electronic noise

        # PCCT-SPECIFIC — 2 thresholds matching clinical protocol
        detector_type = :photon_counting,
        n_energy_bins = 2,
        energy_thresholds = [25.0, 65.0],     # T1=25 keV, T2=65 keV
        energy_resolution = 10.0,              # keV FWHM (superseded by DRM at :high)
        charge_sharing_fwhm = 0.08,            # mm (superseded by Koch-Mehrin at :high)
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
    sim_rotation_time = 1.0       # seconds (matches clinical)
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
    sim_mA_scan2 = 200.0   # ~10 mGy target (140 kVp)
    sim_mA_scan3 = 400.0   # ~20 mGy target
    sim_mA_scan4 = 300.0   # ~10 mGy target (120 kVp, higher mA to compensate)
end

# ╔═╡ 08090005-0000-4000-8000-000000000000
sim_opts = BS.SimOptions(
    fidelity = :high,
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
sim_noise_floor_hu = 15.0  # HU — tune to match clinical (PCCT typically lower than EiCT)

# ╔═╡ 08100001-0000-4000-8000-000000000000
md"""
## 10. Water Attenuation Calibration

Analytical μ\_water from spectrum — no simulation needed.

1. `resolve_spectrum` → filtered spectrum (energies `e`, weights `w`)
2. Mean energy → `compute_μ_at_energy(water, mean_E)` for each bin's energy range
3. Per-bin: window spectrum to bin boundaries, compute bin-specific mean energy
"""

# ╔═╡ 08100002-0000-4000-8000-000000000000
# Analytical μ_water at 140 kVp: combined + per-bin
(μ_water_140, μ_water_140_low, μ_water_140_high) = let
    prot = BS.CTProtocol(kVp = 140.0)
    e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)

    # Combined (full spectrum) — mean energy
    mean_E = sum(e .* w) / sum(w)
    mu_combined = BS.compute_μ_at_energy(XA.Materials.water, mean_E)

    # Per-bin: window spectrum to threshold boundaries
    thresholds = sim_scanner.energy_thresholds  # [25.0, 65.0]

    # Low bin: 25–65 keV
    low_mask = (e .>= thresholds[1]) .& (e .< thresholds[2])
    e_low, w_low = e[low_mask], w[low_mask]
    mean_E_low = sum(e_low .* w_low) / sum(w_low)
    mu_low = BS.compute_μ_at_energy(XA.Materials.water, mean_E_low)

    # High bin: >65 keV (up to kVp)
    high_mask = e .>= thresholds[2]
    e_high, w_high = e[high_mask], w[high_mask]
    mean_E_high = sum(e_high .* w_high) / sum(w_high)
    mu_high = BS.compute_μ_at_energy(XA.Materials.water, mean_E_high)

    @info "140 kVp μ_water: combined=$(round(mu_combined, digits = 5)) " *
        "(E̅=$(round(mean_E, digits = 1)) keV), " *
        "low=$(round(mu_low, digits = 5)) (E̅=$(round(mean_E_low, digits = 1)) keV), " *
        "high=$(round(mu_high, digits = 5)) (E̅=$(round(mean_E_high, digits = 1)) keV)"

    (mu_combined, mu_low, mu_high)
end

# ╔═╡ 08100003-0000-4000-8000-000000000000
md"""
**Water attenuation (140 kVp, analytical):**
- Combined: μ\_water = $(round(μ_water_140, sigdigits=4)) cm⁻¹
- Low bin (25–65 keV): μ\_water = $(round(μ_water_140_low, sigdigits=4)) cm⁻¹
- High bin (>65 keV): μ\_water = $(round(μ_water_140_high, sigdigits=4)) cm⁻¹
"""

# ╔═╡ 08100004-0000-4000-8000-000000000000
# Analytical μ_water at 120 kVp (for Scan 4)
μ_water_120 = let
    prot = BS.CTProtocol(kVp = 120.0)
    e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
    mean_E = sum(e .* w) / sum(w)
    mu = BS.compute_μ_at_energy(XA.Materials.water, mean_E)
    @info "120 kVp μ_water: $(round(mu, digits = 5)) (E̅=$(round(mean_E, digits = 1)) keV)"
    mu
end

# ╔═╡ 08100005-0000-4000-8000-000000000000
md"**Water attenuation (120 kVp, analytical):** μ\_water = $(round(μ_water_120, sigdigits=4)) cm⁻¹"

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
    )

    ws = BS.create_workspace(sim_scanner, prot, sim_opts, sim_recon_opts, sim_phantom_gpu)
    result = BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_opts)

    geom = ws.geom
    combined_sino = ws.combined
    vmi_sino_buf = ws.vmi_sino
    mat_map = result.mat_map
    pcct_sino = result.pcct_sino

    # Copy to CPU
    combined_cpu = Array(combined_sino)
    bin_low_cpu = Array(pcct_sino.bins[1])
    bin_high_cpu = Array(pcct_sino.bins[2])

    # Save material map components for VMI (need GPU re-upload later)
    mat1_cpu = mat_map !== nothing ? Array(mat_map.material1) : nothing
    mat2_cpu = mat_map !== nothing ? Array(mat_map.material2) : nothing
    mat1_name = mat_map !== nothing ? mat_map.material1_name : :water
    mat2_name = mat_map !== nothing ? mat_map.material2_name : :iodine

    # Cleanup
    ws = nothing; result = nothing; GC.gc(true)

    (
        combined = combined_cpu,
        bin_low = bin_low_cpu,
        bin_high = bin_high_cpu,
        mat1 = mat1_cpu,
        mat2 = mat2_cpu,
        mat1_name = mat1_name,
        mat2_name = mat2_name,
        geom = geom,
    )
end

# ╔═╡ 08120001-0000-4000-8000-000000000000
md"""
## 12. Scan 2 Reconstructions

### 12a. Polyenergetic FBP
"""

# ╔═╡ 08120002-0000-4000-8000-000000000000
# Poly FBP — combined sinogram, no BHC needed for PCCT
sim_scan2_poly_fbp = let
    sino_gpu = MtlArray(sim_scan2.combined)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = sim_custom_filter
    )
    recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    # To HU + noise floor
    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_fdk = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end

# ╔═╡ 08120003-0000-4000-8000-000000000000
md"### 12b. Polyenergetic QIR (HIR strength=3)"

# ╔═╡ 08120004-0000-4000-8000-000000000000
# Poly QIR — Hybrid IR with strength 3 (approximates Siemens QIR 3)
sim_scan2_poly_qir = let
    sino_gpu = MtlArray(sim_scan2.combined)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = 3, filter = sim_custom_filter
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
    recon_μ = Array(ws_hir.volume)

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_hir = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end

# ╔═╡ 08120005-0000-4000-8000-000000000000
md"""
### 12c. Low-Threshold Bin (25–65 keV) — FBP + QIR
"""

# ╔═╡ 08120006-0000-4000-8000-000000000000
# Low-bin FBP
sim_scan2_low_fbp = let
    sino_gpu = MtlArray(sim_scan2.bin_low)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = sim_custom_filter
    )
    recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140_low))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_fdk = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end

# ╔═╡ 08120007-0000-4000-8000-000000000000
# Low-bin QIR
sim_scan2_low_qir = let
    sino_gpu = MtlArray(sim_scan2.bin_low)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = 3, filter = sim_custom_filter
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
    recon_μ = Array(ws_hir.volume)

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140_low))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_hir = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end

# ╔═╡ 08120008-0000-4000-8000-000000000000
md"""
### 12d. High-Threshold Bin (>65 keV) — FBP + QIR
"""

# ╔═╡ 08120009-0000-4000-8000-000000000000
# High-bin FBP
sim_scan2_high_fbp = let
    sino_gpu = MtlArray(sim_scan2.bin_high)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    ws_fdk = BS.create_fdk_recon_workspace(
        sino_gpu, geom, recon_size;
        filter = sim_custom_filter
    )
    recon_μ = Array(BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size))

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140_high))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_fdk = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end

# ╔═╡ 08120010-0000-4000-8000-000000000000
# High-bin QIR
sim_scan2_high_qir = let
    sino_gpu = MtlArray(sim_scan2.bin_high)
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    ws_hir = BS.create_hir_recon_workspace(
        sino_gpu, geom, recon_size;
        strength = 3, filter = sim_custom_filter
    )
    BS.reconstruct!(ws_hir, sino_gpu, geom, recon_size)
    recon_μ = Array(ws_hir.volume)

    recon_hu = Float32.(BS.to_hounsfield(recon_μ; μ_water = μ_water_140_high))
    BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)

    ws_hir = nothing; sino_gpu = nothing; GC.gc(true)
    recon_hu
end

# ╔═╡ 08120011-0000-4000-8000-000000000000
md"""
### 12e. VMI 40/70/100/140 keV (QIR)

Virtual monoenergetic images synthesized from 2-material decomposition.
"""

# ╔═╡ 08120012-0000-4000-8000-000000000000
# VMI reconstruction at 4 energies using HIR (QIR strength=3)
sim_scan2_vmi = let
    geom = sim_scan2.geom
    recon_size = sim_matrix_size

    if sim_scan2.mat1 === nothing
        Dict{Float64, Array{Float32, 3}}()
    else
        # Re-upload material maps to GPU
        mat1_gpu = MtlArray(sim_scan2.mat1)
        mat2_gpu = MtlArray(sim_scan2.mat2)
        mat_map = BS.MaterialMap(
            mat1_gpu, mat2_gpu;
            material1_name = sim_scan2.mat1_name,
            material2_name = sim_scan2.mat2_name,
            domain = :projection
        )
        vmi_sino_buf = similar(mat1_gpu)

        vmi_results = Dict{Float64, Array{Float32, 3}}()
        for E in [40.0, 70.0, 100.0, 140.0]
            vmi_sino = BS.virtual_monoenergetic(mat_map, E; ws_output = vmi_sino_buf)

            # HIR reconstruction (QIR 3)
            ws_hir = BS.create_hir_recon_workspace(
                vmi_sino, geom, recon_size;
                strength = 3, filter = sim_custom_filter
            )
            BS.reconstruct!(ws_hir, vmi_sino, geom, recon_size)

            # HU conversion using energy-specific water attenuation
            μ_water_E = BS.get_water_attenuation_vmi(E)
            vol_hu = Float32.(BS.to_hounsfield(Array(ws_hir.volume); μ_water = μ_water_E))
            BS.add_system_noise_floor!(vol_hu, sim_noise_floor_hu)

            vmi_results[E] = vol_hu
            ws_hir = nothing; GC.gc(true)
        end

        mat1_gpu = nothing; mat2_gpu = nothing; mat_map = nothing
        vmi_sino_buf = nothing; GC.gc(true)
        vmi_results
    end
end

# ╔═╡ 08120013-0000-4000-8000-000000000000
# Scan 2 reconstruction comparison: FBP vs QIR
let
    mid_z = sim_n_recon_slices ÷ 2
    window = (-200, 500)

    fig = CM.Figure(size = (1400, 600), fontsize = 14)
    ax1 = CM.Axis(fig[1, 1], title = "Poly FBP", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax1, sim_scan2_poly_fbp[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax1)

    ax2 = CM.Axis(fig[1, 2], title = "Poly QIR (Strength 3)", aspect = CM.DataAspect())
    CM.heatmap!(ax2, sim_scan2_poly_qir[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax2)

    CM.Colorbar(fig[1, 3], hm, label = "HU")
    CM.Label(fig[0, :], text = "PCCT Scan 2 (140 kVp, ~10 mGy): FBP vs QIR-3", fontsize = 16, font = :bold)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_fbp_vs_qir.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08120014-0000-4000-8000-000000000000
# Threshold-binned reconstruction comparison
let
    mid_z = sim_n_recon_slices ÷ 2
    window = (-200, 500)

    fig = CM.Figure(size = (1800, 500), fontsize = 14)

    ax1 = CM.Axis(fig[1, 1], title = "Low Bin FBP\n(25–65 keV)", aspect = CM.DataAspect())
    CM.heatmap!(ax1, sim_scan2_low_fbp[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax1)

    ax2 = CM.Axis(fig[1, 2], title = "Low Bin QIR", aspect = CM.DataAspect())
    CM.heatmap!(ax2, sim_scan2_low_qir[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax2)

    ax3 = CM.Axis(fig[1, 3], title = "High Bin FBP\n(>65 keV)", aspect = CM.DataAspect())
    CM.heatmap!(ax3, sim_scan2_high_fbp[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax3)

    ax4 = CM.Axis(fig[1, 4], title = "High Bin QIR", aspect = CM.DataAspect())
    hm = CM.heatmap!(ax4, sim_scan2_high_qir[:, :, mid_z], colormap = :grays, colorrange = window)
    CM.hidedecorations!(ax4)

    CM.Colorbar(fig[1, 5], hm, label = "HU")
    CM.Label(fig[0, :], text = "Threshold-Binned Reconstructions (Scan 2)", fontsize = 16, font = :bold)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_binned_recons.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08120015-0000-4000-8000-000000000000
# VMI montage
let
    vmi_E = [40.0, 70.0, 100.0, 140.0]
    mid_z = sim_n_recon_slices ÷ 2
    window = (-200, 800)

    fig = CM.Figure(size = (1600, 450), fontsize = 14)
    for (idx, E) in enumerate(vmi_E)
        ax = CM.Axis(fig[1, idx], title = "VMI $(Int(E)) keV", titlesize = 13, aspect = CM.DataAspect())
        if haskey(sim_scan2_vmi, E)
            CM.heatmap!(ax, sim_scan2_vmi[E][:, :, mid_z], colormap = :grays, colorrange = window)
        end
        CM.hidedecorations!(ax)
    end
    CM.Label(fig[0, :], text = "PCCT VMI Energy Sweep (Scan 2, QIR 3)", fontsize = 16, font = :bold)
    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_vmi_montage.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08120016-0000-4000-8000-000000000000
# VMI energy-dependent contrast curves
let
    mid_z = sim_n_recon_slices ÷ 2
    mask_cpu = Array(sim_phantom_gpu.mask)
    mask_slice = mask_cpu[:, :, size(mask_cpu, 3) ÷ 2]

    roi_defs = [
        ("Solid Water", UInt8(3)),
        ("Ca 200 mg/cc", UInt8(12)),
        ("Iodine 10 mg/mL", UInt8(24)),
    ]

    fig = CM.Figure(size = (1100, 600), fontsize = 14)
    ax = CM.Axis(
        fig[1, 1], xlabel = "VMI Energy (keV)", ylabel = "HU",
        title = "VMI Energy-Dependent Contrast (Mask-Based ROIs)"
    )

    vmi_E = sort(collect(keys(sim_scan2_vmi)))
    colors = [:blue, :red, :green]
    for (ci, (name, region_val)) in enumerate(roi_defs)
        roi_mask = mask_slice .== region_val
        !any(roi_mask) && continue

        hu_values = Float64[]
        for E in vmi_E
            vol_slice = sim_scan2_vmi[E][:, :, mid_z]
            push!(hu_values, Float64(mean(vol_slice[roi_mask])))
        end
        CM.lines!(ax, vmi_E, hu_values, linewidth = 3, color = colors[ci], label = name)
        CM.scatter!(ax, vmi_E, hu_values, color = colors[ci], markersize = 8)
    end
    CM.hlines!(ax, [0.0], color = :gray, linestyle = :dash, linewidth = 1.5)
    CM.axislegend(ax, position = :rt)
    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_vmi_curves.png"), fig, px_per_unit = 3)
    fig
end

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
begin
    # ── Mono+ tuning parameters [TUNE: MONO+] ──
    # σ_lp_mm: Gaussian LP width (mm). Larger = more aggressive noise reduction.
    #   ~1.0 mm: mild   (preserves detail, modest noise reduction)
    #   ~2.0 mm: moderate (good CNR improvement, some fine-detail loss)
    #   ~3.0 mm: aggressive (maximum CNR improvement, coarsens fine detail)
    mono_plus_σ_lp_mm = 2.0

    # E_optimal: energy with minimum noise (~65–77 keV for 140 kVp PCCT)
    mono_plus_E_optimal = 70.0  # keV

    # Derived
    mono_plus_pixel_mm = sim_fov_cm / 512 * 10.0
end

# ╔═╡ 08126004-0000-4000-8000-000000000000
# Compute Mono+ for all VMI energies
sim_scan2_mono_plus = let
    vmi = sim_scan2_vmi
    E_opt = mono_plus_E_optimal

    if isempty(vmi) || !haskey(vmi, E_opt)
        Dict{Float64, Array{Float32, 3}}()
    else
        vmi_opt = vmi[E_opt]
        result = Dict{Float64, Array{Float32, 3}}()
        for E in sort(collect(keys(vmi)))
            if E >= E_opt
                # At or above optimal energy: Mono+ = standard VMI (no benefit)
                result[E] = copy(vmi[E])
            else
                # Below optimal: frequency-split recombination
                result[E] = mono_plus_vmi(
                    vmi[E], vmi_opt;
                    σ_lp_mm = mono_plus_σ_lp_mm,
                    pixel_mm = mono_plus_pixel_mm
                )
            end
        end
        result
    end
end

# ╔═╡ 08126005-0000-4000-8000-000000000000
# Mono vs Mono+ at 40 keV — maximum visual difference
let
    E = 40.0
    mid_z = sim_n_recon_slices ÷ 2

    if !isempty(sim_scan2_vmi) && haskey(sim_scan2_vmi, E)
        mono = sim_scan2_vmi[E][:, :, mid_z]
        mplus = sim_scan2_mono_plus[E][:, :, mid_z]
        diff = mplus .- mono
        window = (-200, 800)

        fig = CM.Figure(size = (1800, 550), fontsize = 14)

        ax1 = CM.Axis(
            fig[1, 1], title = "Standard VMI 40 keV",
            aspect = CM.DataAspect()
        )
        CM.heatmap!(ax1, mono, colormap = :grays, colorrange = window)
        CM.hidedecorations!(ax1)

        ax2 = CM.Axis(
            fig[1, 2], title = "Mono+ 40 keV (σ=$(mono_plus_σ_lp_mm) mm)",
            aspect = CM.DataAspect()
        )
        CM.heatmap!(ax2, mplus, colormap = :grays, colorrange = window)
        CM.hidedecorations!(ax2)

        ax3 = CM.Axis(
            fig[1, 3], title = "Difference (Mono+ − Mono)",
            aspect = CM.DataAspect()
        )
        hm = CM.heatmap!(ax3, diff, colormap = :RdBu, colorrange = (-50, 50))
        CM.hidedecorations!(ax3)
        CM.Colorbar(fig[1, 4], hm, label = "ΔHU")

        CM.Label(
            fig[0, :], text = "Mono+ vs Standard VMI at 40 keV",
            fontsize = 16, font = :bold
        )
        CM.save(joinpath(FIGURES_DIR, "nb07_mono_plus_40keV.png"), fig, px_per_unit = 3)
        fig
    end
end

# ╔═╡ 08126006-0000-4000-8000-000000000000
# Full montage: standard Mono (top) vs Mono+ (bottom)
let
    vmi_E = [40.0, 70.0, 100.0, 140.0]
    mid_z = sim_n_recon_slices ÷ 2
    window = (-200, 800)

    if !isempty(sim_scan2_vmi)
        fig = CM.Figure(size = (1600, 850), fontsize = 13)

        for (col, E) in enumerate(vmi_E)
            ax_t = CM.Axis(
                fig[1, col], title = "Mono $(Int(E)) keV",
                titlesize = 12, aspect = CM.DataAspect()
            )
            if haskey(sim_scan2_vmi, E)
                CM.heatmap!(
                    ax_t, sim_scan2_vmi[E][:, :, mid_z],
                    colormap = :grays, colorrange = window
                )
            end
            CM.hidedecorations!(ax_t)

            ax_b = CM.Axis(
                fig[2, col], title = "Mono+ $(Int(E)) keV",
                titlesize = 12, aspect = CM.DataAspect()
            )
            if haskey(sim_scan2_mono_plus, E)
                CM.heatmap!(
                    ax_b, sim_scan2_mono_plus[E][:, :, mid_z],
                    colormap = :grays, colorrange = window
                )
            end
            CM.hidedecorations!(ax_b)
        end

        # CM.Label(fig[1, 0], text = "Mono", fontsize = 14, font = :bold, rotation = pi / 2)
        # CM.Label(fig[2, 0], text = "Mono+", fontsize = 14, font = :bold, rotation = pi / 2)
        # CM.Label(
        # fig[0, 1:4],
        # text = "Standard VMI vs Mono+ (σ_LP=$(mono_plus_σ_lp_mm) mm, E_opt=$(Int(mono_plus_E_optimal)) keV)",
        # fontsize = 16, font = :bold
        # )

        CM.save(joinpath(FIGURES_DIR, "nb07_mono_vs_monoplus_montage.png"), fig, px_per_unit = 3)
        fig
    end
end

# ╔═╡ 08126007-0000-4000-8000-000000000000
# Iodine CNR, noise, and contrast: Mono vs Mono+ across energies
let
    mid_z = sim_n_recon_slices ÷ 2
    mask_cpu = Array(sim_phantom_gpu.mask)
    mask_slice = mask_cpu[:, :, size(mask_cpu, 3) ÷ 2]

    # ROI masks from phantom labels
    iodine_mask = mask_slice .== UInt8(24)   # I 10.0 mg/mL
    water_mask = mask_slice .== UInt8(2)    # Water (outer + inner)

    if !isempty(sim_scan2_vmi) && any(iodine_mask) && any(water_mask)
        vmi_E = sort(collect(keys(sim_scan2_vmi)))

        cnr_mono = Float64[];  cnr_plus = Float64[]
        noise_mono = Float64[];  noise_plus = Float64[]
        contrast_mono = Float64[];  contrast_plus = Float64[]

        for E in vmi_E
            # Standard Mono
            sl_m = sim_scan2_vmi[E][:, :, mid_z]
            μ_i_m = Float64(mean(sl_m[iodine_mask]))
            μ_w_m = Float64(mean(sl_m[water_mask]))
            σ_w_m = Float64(std(sl_m[water_mask]))
            c_m = abs(μ_i_m - μ_w_m)
            push!(contrast_mono, c_m)
            push!(noise_mono, σ_w_m)
            push!(cnr_mono, σ_w_m > 0 ? c_m / σ_w_m : 0.0)

            # Mono+
            sl_p = sim_scan2_mono_plus[E][:, :, mid_z]
            μ_i_p = Float64(mean(sl_p[iodine_mask]))
            μ_w_p = Float64(mean(sl_p[water_mask]))
            σ_w_p = Float64(std(sl_p[water_mask]))
            c_p = abs(μ_i_p - μ_w_p)
            push!(contrast_plus, c_p)
            push!(noise_plus, σ_w_p)
            push!(cnr_plus, σ_w_p > 0 ? c_p / σ_w_p : 0.0)
        end

        fig = CM.Figure(size = (1400, 500), fontsize = 14)

        # Panel A: CNR (cf. Grant Fig 4–5)
        ax1 = CM.Axis(
            fig[1, 1], xlabel = "VMI Energy (keV)", ylabel = "Iodine CNR",
            title = "A. Contrast-to-Noise Ratio"
        )
        CM.lines!(ax1, vmi_E, cnr_mono, linewidth = 2.5, color = :steelblue, label = "Mono")
        CM.scatter!(ax1, vmi_E, cnr_mono, color = :steelblue, markersize = 8)
        CM.lines!(ax1, vmi_E, cnr_plus, linewidth = 2.5, color = :coral, label = "Mono+")
        CM.scatter!(ax1, vmi_E, cnr_plus, color = :coral, markersize = 8)
        CM.axislegend(ax1, position = :rt)

        # Panel B: Noise (cf. Yu Fig 5)
        ax2 = CM.Axis(
            fig[1, 2], xlabel = "VMI Energy (keV)", ylabel = "σ Water (HU)",
            title = "B. Image Noise"
        )
        CM.lines!(ax2, vmi_E, noise_mono, linewidth = 2.5, color = :steelblue, label = "Mono")
        CM.scatter!(ax2, vmi_E, noise_mono, color = :steelblue, markersize = 8)
        CM.lines!(ax2, vmi_E, noise_plus, linewidth = 2.5, color = :coral, label = "Mono+")
        CM.scatter!(ax2, vmi_E, noise_plus, color = :coral, markersize = 8)
        CM.axislegend(ax2, position = :rt)

        # Panel C: Contrast (cf. Yu Fig 4)
        ax3 = CM.Axis(
            fig[1, 3], xlabel = "VMI Energy (keV)", ylabel = "|ΔHU| Iodine−Water",
            title = "C. Iodine Contrast"
        )
        CM.lines!(ax3, vmi_E, contrast_mono, linewidth = 2.5, color = :steelblue, label = "Mono")
        CM.scatter!(ax3, vmi_E, contrast_mono, color = :steelblue, markersize = 8)
        CM.lines!(ax3, vmi_E, contrast_plus, linewidth = 2.5, color = :coral, label = "Mono+")
        CM.scatter!(ax3, vmi_E, contrast_plus, color = :coral, markersize = 8)
        CM.axislegend(ax3, position = :rt)

        CM.Label(
            fig[0, :],
            text = "Mono vs Mono+ — Iodine 10 mg/mL (σ_LP=$(mono_plus_σ_lp_mm) mm)",
            fontsize = 16, font = :bold
        )

        CM.save(joinpath(FIGURES_DIR, "nb07_mono_vs_monoplus_cnr.png"), fig, px_per_unit = 3)
        fig
    end
end

# ╔═╡ 08126008-0000-4000-8000-000000000000
# Sensitivity analysis: CNR at 40 keV as function of σ_lp_mm
let
    E_target = 40.0
    E_opt = mono_plus_E_optimal
    mid_z = sim_n_recon_slices ÷ 2
    mask_cpu = Array(sim_phantom_gpu.mask)
    mask_slice = mask_cpu[:, :, size(mask_cpu, 3) ÷ 2]
    iodine_mask = mask_slice .== UInt8(24)
    water_mask = mask_slice .== UInt8(2)

    if !isempty(sim_scan2_vmi) && haskey(sim_scan2_vmi, E_target) &&
            haskey(sim_scan2_vmi, E_opt) && any(iodine_mask) && any(water_mask)

        σ_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        cnr_values = Float64[]
        noise_values = Float64[]

        vmi_t = sim_scan2_vmi[E_target]
        vmi_o = sim_scan2_vmi[E_opt]

        for σ in σ_values
            mp = mono_plus_vmi(
                vmi_t, vmi_o;
                σ_lp_mm = σ, pixel_mm = mono_plus_pixel_mm
            )
            sl = mp[:, :, mid_z]
            μ_i = Float64(mean(sl[iodine_mask]))
            μ_w = Float64(mean(sl[water_mask]))
            σ_w = Float64(std(sl[water_mask]))
            push!(cnr_values, σ_w > 0 ? abs(μ_i - μ_w) / σ_w : 0.0)
            push!(noise_values, σ_w)
        end

        # Also measure standard VMI baseline
        sl_std = sim_scan2_vmi[E_target][:, :, mid_z]
        cnr_std = let
            μ_i = Float64(mean(sl_std[iodine_mask]))
            μ_w = Float64(mean(sl_std[water_mask]))
            σ_w = Float64(std(sl_std[water_mask]))
            σ_w > 0 ? abs(μ_i - μ_w) / σ_w : 0.0
        end

        fig = CM.Figure(size = (1000, 500), fontsize = 14)
        ax = CM.Axis(
            fig[1, 1],
            xlabel = "σ_LP (mm)", ylabel = "Iodine CNR at 40 keV",
            title = "Mono+ Sensitivity: LP Cutoff vs CNR (I 10 mg/mL)"
        )
        CM.lines!(
            ax, σ_values, cnr_values, linewidth = 2.5, color = :coral,
            label = "Mono+"
        )
        CM.scatter!(ax, σ_values, cnr_values, color = :coral, markersize = 8)
        CM.hlines!(
            ax, [cnr_std], color = :steelblue, linestyle = :dash,
            linewidth = 2, label = "Standard Mono (40 keV)"
        )
        CM.vlines!(
            ax, [mono_plus_σ_lp_mm], color = :gray60, linestyle = :dot,
            linewidth = 1.5, label = "Current σ=$(mono_plus_σ_lp_mm)"
        )
        CM.axislegend(ax, position = :rb)

        CM.save(joinpath(FIGURES_DIR, "nb07_mono_plus_sigma_sweep.png"), fig, px_per_unit = 3)
        fig
    end
end

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
    sim_scan1_poly_qir = nothing  # TODO: poly QIR recon
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
    sim_scan3_poly_qir = nothing  # TODO: poly QIR recon
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
    sim_scan4_poly_qir = nothing  # TODO: poly QIR recon (use μ_water_120)
end

# ╔═╡ 08160001-0000-4000-8000-000000000000
md"""
## 16. Simulated Segmentation & Measurements

Segment Scan 2 QIR reconstruction, then measure all implemented scans.
"""

# ╔═╡ 08160002-0000-4000-8000-000000000000
# Segment simulated Scan 2 (use poly QIR as reference)
sim_seg_result = let
    ref = sim_scan2_poly_qir
    mid_z = size(ref, 3) ÷ 2
    mask, rods, center = segment_gammex_rods(ref[:, :, mid_z]; fov_cm = sim_fov_cm)
    (mask = mask, rods = rods, center = center)
end

# ╔═╡ 08160003-0000-4000-8000-000000000000
# Measurements for all implemented Scan 2 reconstructions
sim_measurements_scan2 = let
    results = []

    # Poly
    push!(
        results, measure_scan(
            sim_scan2_poly_fbp, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_poly_fbp"; fov_cm = sim_fov_cm
        )
    )
    push!(
        results, measure_scan(
            sim_scan2_poly_qir, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_poly_qir"; fov_cm = sim_fov_cm
        )
    )

    # Low-threshold bin
    push!(
        results, measure_scan(
            sim_scan2_low_fbp, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_low_fbp"; fov_cm = sim_fov_cm
        )
    )
    push!(
        results, measure_scan(
            sim_scan2_low_qir, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_low_qir"; fov_cm = sim_fov_cm
        )
    )

    # High-threshold bin
    push!(
        results, measure_scan(
            sim_scan2_high_fbp, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_high_fbp"; fov_cm = sim_fov_cm
        )
    )
    push!(
        results, measure_scan(
            sim_scan2_high_qir, sim_seg_result.mask,
            sim_seg_result.rods, sim_seg_result.center, "scan2_high_qir"; fov_cm = sim_fov_cm
        )
    )

    results
end

# ╔═╡ 08160004-0000-4000-8000-000000000000
# Display summary: Water ROI noise across Scan 2 reconstructions
let
    names = [m.name for m in sim_measurements_scan2]
    water_stds = [m.rod_stds[1] for m in sim_measurements_scan2]  # Water (O) is first rod

    fig = CM.Figure(size = (900, 500), fontsize = 14)
    ax = CM.Axis(
        fig[1, 1], title = "Water ROI Noise — Scan 2 Reconstructions",
        ylabel = "σ (HU)", xticks = (1:length(names), names), xticklabelrotation = pi / 6
    )
    CM.barplot!(
        ax, 1:length(names), water_stds,
        color = [:coral, :steelblue, :lightsalmon, :lightsteelblue, :sandybrown, :lightcyan]
    )
    CM.ylims!(ax, 0, maximum(water_stds) * 1.3)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_noise_bars.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08170001-0000-4000-8000-000000000000
md"""
## 17. Comparison Figures

Active for simulated data. Clinical comparisons are placeholder until DICOM loaded.
"""

# ╔═╡ 08170002-0000-4000-8000-000000000000
# HU scatter: simulated rod means vs NIST expected
let
    # Approximate NIST expected HU at 140 kVp (polyenergetic)
    expected_hu = Dict(
        "Water (O)" => 0.0, "Water (I)" => 0.0,
        "SW ref 1" => 0.0, "SW ref 2" => 0.0,
        "Ca 50" => 55.0, "Ca 100" => 120.0, "Ca 200" => 260.0,
        "Ca 300" => 400.0, "Ca 400" => 540.0,
        "I 2.0" => 45.0, "I 2.5" => 58.0, "I 5.0" => 115.0,
        "I 7.5" => 175.0, "I 10.0" => 235.0, "I 15.0" => 350.0, "I 20.0" => 465.0,
    )

    m = sim_measurements_scan2[2]  # poly QIR
    fig = CM.Figure(size = (800, 700), fontsize = 14)
    ax = CM.Axis(
        fig[1, 1], xlabel = "Expected HU (NIST)", ylabel = "Simulated HU",
        title = "Scan 2 Poly QIR: Simulated vs Expected"
    )

    xs, ys, labels = Float64[], Float64[], String[]
    for (i, name) in enumerate(m.rod_names)
        if haskey(expected_hu, name)
            push!(xs, expected_hu[name])
            push!(ys, m.rod_means[i])
            push!(labels, name)
        end
    end

    CM.scatter!(ax, xs, ys, color = :steelblue, markersize = 10)
    lims = (minimum(vcat(xs, ys)) - 50, maximum(vcat(xs, ys)) + 50)
    CM.lines!(
        ax, [lims[1], lims[2]], [lims[1], lims[2]], color = :gray, linestyle = :dash,
        linewidth = 1.5, label = "Unity"
    )
    CM.xlims!(ax, lims...); CM.ylims!(ax, lims...)
    CM.axislegend(ax, position = :lt)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_hu_scatter.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08170003-0000-4000-8000-000000000000
# NPS comparison: Poly FBP vs Poly QIR
let
    m_fbp = sim_measurements_scan2[1]
    m_qir = sim_measurements_scan2[2]

    fig = CM.Figure(size = (900, 550), fontsize = 14)
    ax = CM.Axis(
        fig[1, 1], xlabel = "Spatial Frequency (lp/cm)", ylabel = "nNPS",
        title = "Scan 2: Poly FBP vs QIR — Noise Power Spectrum"
    )
    CM.lines!(
        ax, m_fbp.nps.frequencies, m_fbp.nps.nnps_1d, linewidth = 2.5,
        color = :coral, label = "FBP"
    )
    CM.lines!(
        ax, m_qir.nps.frequencies, m_qir.nps.nnps_1d, linewidth = 2.5,
        color = :steelblue, label = "QIR 3"
    )
    CM.axislegend(ax, position = :rt)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_nps_comparison.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08170004-0000-4000-8000-000000000000
# MTF comparison: Poly FBP vs Poly QIR
let
    m_fbp = sim_measurements_scan2[1]
    m_qir = sim_measurements_scan2[2]

    fig = CM.Figure(size = (900, 550), fontsize = 14)
    ax = CM.Axis(
        fig[1, 1], xlabel = "Spatial Frequency (lp/cm)", ylabel = "MTF",
        title = "Scan 2: Poly FBP vs QIR — Modulation Transfer Function"
    )
    CM.lines!(
        ax, m_fbp.mtf.frequencies, m_fbp.mtf.mtf, linewidth = 2.5,
        color = :coral, label = "FBP (f50=$(round(m_fbp.mtf_f50, digits = 1)))"
    )
    CM.lines!(
        ax, m_qir.mtf.frequencies, m_qir.mtf.mtf, linewidth = 2.5,
        color = :steelblue, label = "QIR (f50=$(round(m_qir.mtf_f50, digits = 1)))"
    )
    CM.hlines!(ax, [0.5, 0.1], color = :gray, linestyle = :dot, linewidth = 1)
    CM.axislegend(ax, position = :rt)
    CM.ylims!(ax, 0, 1.05)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_mtf_comparison.png"), fig, px_per_unit = 3)
    fig
end

# ╔═╡ 08170005-0000-4000-8000-000000000000
# Line profiles through phantom center: FBP vs QIR
let
    mid_z = sim_n_recon_slices ÷ 2
    mid_row = 256
    fbp_profile = sim_scan2_poly_fbp[mid_row, :, mid_z]
    qir_profile = sim_scan2_poly_qir[mid_row, :, mid_z]

    pixel_mm = sim_fov_cm / 512 * 10.0
    x_mm = (1:512) .* pixel_mm .- 512 * pixel_mm / 2

    fig = CM.Figure(size = (1100, 500), fontsize = 14)
    ax = CM.Axis(
        fig[1, 1], xlabel = "Position (mm)", ylabel = "HU",
        title = "Horizontal Line Profile (Scan 2, 140 kVp)"
    )
    CM.lines!(ax, x_mm, fbp_profile, linewidth = 1.5, color = :coral, label = "FBP")
    CM.lines!(ax, x_mm, qir_profile, linewidth = 1.5, color = :steelblue, label = "QIR 3")
    CM.hlines!(ax, [0.0], color = :gray, linestyle = :dash, linewidth = 1)
    CM.axislegend(ax, position = :rt)

    CM.save(joinpath(FIGURES_DIR, "nb07_scan2_line_profiles.png"), fig, px_per_unit = 3)
    fig
end

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
| Detector | CdTe, 144 × 0.4 mm (standard mode) |
| Energy thresholds | T1=25 keV, T2=65 keV |
| Rotation time | 1.0 s |
| Views per rotation | 984 |
| Collimation | 144 × 0.4 mm = 57.6 mm |
| FOV | 350 mm |
| Matrix | 512 × 512 |
| Kernel | Br36 (medium-sharp body) |
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
# ╟─08120001-0000-4000-8000-000000000000
# ╠═08120002-0000-4000-8000-000000000000
# ╟─08120003-0000-4000-8000-000000000000
# ╠═08120004-0000-4000-8000-000000000000
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
