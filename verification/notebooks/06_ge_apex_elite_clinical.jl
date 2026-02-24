### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ db7ada07-c245-4427-a9d8-1fa43e74e884
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
	# Pkg.update()

	using Revise
end

# ╔═╡ 9d514655-cb7c-4852-8e8e-d2a98527fe0a
using Unitful: @u_str

# ╔═╡ 1ce13813-fa59-43a1-9e0e-99741eeace58
using LinearAlgebra

# ╔═╡ 02bf2bbb-e785-4e5f-8476-58346735b7fc
using FFTW

# ╔═╡ 9636e23b-9d46-41be-988c-843c59de3cf6
using Random

# ╔═╡ 993c7d4b-ae9d-4234-ad8d-994a904bd83c
using Metal

# ╔═╡ 05487717-8249-4d89-bf94-f7e7dc379bfb
md"""
# GE Revolution Apex Elite — Clinical Gammex 472 Scans
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
| 7 | DE (GSI) | 80/140 | 405 | 204 | 0.5 s | 64 × 40 mm | 10.07 | 40.27 | VMI 40/70/100/140 keV all @ 0% (FBP)|
"""

# ╔═╡ 2872d85c-3d09-414f-a194-4b52764fca93
import PlutoUI as UI

# ╔═╡ eff433a8-e36a-4b6f-bfb5-73664ceac2f3
import BasisSimulator as BS

# ╔═╡ 1caeb654-e68e-47d8-a10d-c4020c676c7b
import FileIO, ImageMagick

# ╔═╡ 9cd57146-a7ec-4fe8-a313-4d46659d8795
import CairoMakie as CM

# ╔═╡ 6cf5c19a-88f2-4ba7-91d6-f68b4fb816e1
import Statistics: mean, std, cor

# ╔═╡ fb762acd-fe7f-4e9a-adb6-ad77ba81f2e1
import XrayAttenuation as XA

# ╔═╡ 0e1cad14-1b0a-421e-ab85-19e5601d63d6
import DICOM as DCM

# ╔═╡ a1b2c3d4-0001-4000-8000-000000000001
import JLD2

# ╔═╡ a1b2c3d4-0002-4000-8000-000000000002
import DelimitedFiles

# ╔═╡ 9dd493db-806d-474c-8352-58011ab725e7
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")

# ╔═╡ 64b81aff-4ff0-4cc7-bcaa-a16162524620
UI.TableOfContents()

# ╔═╡ 6e93590f-e900-4c9e-91cb-a03db349b185
md"""
## 1. Single-Energy Scans — ASiR-V 0% (FBP) vs ASiR-V 50%

Six axial acquisitions of the Gammex 472 phantom on the GE Revolution Apex Elite.
Each protocol is loaded once: `_fbp` = ASiR-V 0% (pure FBP), `_ir` = ASiR-V 50%.
DICOM pixel data is JPEG 2000 compressed; decoded via ImageMagick.jl + DICOM.jl.

Three display windows per protocol:
- **Soft tissue** W400 / L40 → (−160, 240) HU
- **Bone** W1500 / L400 → (−350, 1150) HU
- **Lung** W1500 / L−600 → (−1350, 150) HU
"""

# ╔═╡ f466466a-c364-4841-8a38-987cbc5777c9
rootdir = "/Volumes/Molloilab/dale/02232026 scans"

# ╔═╡ c19991fb-c50a-4e56-a3f6-bba1e41fd9c9
"""
	load_hu_volume(dcms) → Array{Float32, 3}   (rows × cols × slices)

Convert a pre-loaded vector of DICOM datasets into a Float32 HU volume.
Uses ImageMagick.jl (via FileIO) for J2K decompression.

GE scanners encode signed pixel values in J2K with a +32768 unsigned offset;
the inverse transform `Int16(UInt16_value − 32768)` recovers the original stored
pixel, which is then mapped to HU via the DICOM linear modality LUT:
    HU = StoredPixelValue × RescaleSlope + RescaleIntercept

Slices are sorted by InstanceNumber (0020,0013).
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
		raw_u16 = round.(UInt16, Float32.(img) .* 65535f0)
		i16 = Int16.(Int32.(raw_u16) .- Int32(32768))
		Float32.(i16) .* slope .+ intercept
	end
	return cat(slices...; dims=3)
end

# ╔═╡ 119755f9-4c74-4d15-89a7-793295d21441
begin  # 120 kVp — low dose  (50 mA, 3.38 mGy)
	dcms_120_low_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_50mA_3.38mGyCTDI/0%"))
	dcms_120_low_ir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_50mA_3.38mGyCTDI/50%"))
	hu_120_low_fbp = load_hu_volume(dcms_120_low_fbp)
	hu_120_low_ir = load_hu_volume(dcms_120_low_ir)
end;

# ╔═╡ bdc45741-e374-4d78-ba6b-43b5ef411f2a
let
	mid = size(hu_120_low_fbp, 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title="FBP (ASiR-V 0%) — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_120_low_fbp[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title="ASiR-V 50% — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_120_low_ir[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text="120 kVp · 50 mA · CTDIvol 3.38 mGy", fontsize=16, font=:bold)
	fig
end

# ╔═╡ 41f11735-2a1c-4d36-8a0d-3c57de9cbae0
begin  # 120 kVp — mid dose  (150 mA, 10.16 mGy)
	dcms_120_mid_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_150mA_10.16mGyCTDI/0%"))
	dcms_120_mid_ir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_150mA_10.16mGyCTDI/50%"))
	hu_120_mid_fbp = load_hu_volume(dcms_120_mid_fbp)
	hu_120_mid_ir = load_hu_volume(dcms_120_mid_ir)
end;

# ╔═╡ c0dfa3a5-88f2-47b3-94ab-0393132b0dc1
let
	mid = size(hu_120_mid_fbp, 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title="FBP (ASiR-V 0%) — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_120_mid_fbp[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title="ASiR-V 50% — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_120_mid_ir[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text="120 kVp · 150 mA · CTDIvol 10.16 mGy", fontsize=16, font=:bold)
	fig
end

# ╔═╡ fec5488f-09e4-41af-b3e3-0fef62cd2bfa
begin  # 120 kVp — high dose  (300 mA, 20.38 mGy)
	dcms_120_high_fbp = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_300mA_20.38mGyCTDI/0%"))
	dcms_120_high_ir = DCM.dcmdir_parse(joinpath(rootdir, "120kVp_300mA_20.38mGyCTDI/50%"))
	hu_120_high_fbp = load_hu_volume(dcms_120_high_fbp)
	hu_120_high_ir = load_hu_volume(dcms_120_high_ir)
end;

# ╔═╡ 042d2f32-0a93-48c9-acd1-2701263f8be9
let
	mid = size(hu_120_high_fbp, 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title="FBP (ASiR-V 0%) — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_120_high_fbp[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title="ASiR-V 50% — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_120_high_ir[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text="120 kVp · 300 mA · CTDIvol 20.38 mGy", fontsize=16, font=:bold)
	
	fig
end

# ╔═╡ 90fe8208-f128-4b59-bb14-cb3f6829d9da
begin  # 80 kVp  (480 mA, 10.32 mGy)
	dcms_80_fbp = DCM.dcmdir_parse(joinpath(rootdir, "80kVp_480mA_10.32mGyCTDI/0%"))
	dcms_80_ir = DCM.dcmdir_parse(joinpath(rootdir, "80kVp_480mA_10.32mGyCTDI/50%"))
	hu_80_fbp = load_hu_volume(dcms_80_fbp)
	hu_80_ir = load_hu_volume(dcms_80_ir)
end;

# ╔═╡ e4d91faa-e427-48d3-9f86-fe07a33d8bff
let
	mid = size(hu_80_fbp, 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title="FBP (ASiR-V 0%) — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_80_fbp[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title="ASiR-V 50% — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_80_ir[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text="80 kVp · 480 mA · CTDIvol 10.32 mGy", fontsize=16, font=:bold)
	
	fig
end

# ╔═╡ c95c638e-60a7-4242-8c95-ee352ac96a02
begin  # 100 kVp  (250 mA, 10.53 mGy)
	dcms_100_fbp = DCM.dcmdir_parse(joinpath(rootdir, "100kVp_250mA_10.53mGyCTDI/0%"))
	dcms_100_ir = DCM.dcmdir_parse(joinpath(rootdir, "100kVp_250mA_10.53mGyCTDI/50%"))
	hu_100_fbp = load_hu_volume(dcms_100_fbp)
	hu_100_ir = load_hu_volume(dcms_100_ir)
end;

# ╔═╡ 5d85356a-9524-4041-9c7e-5df155180f8f
let
	mid = size(hu_100_fbp, 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 1000))


	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title="FBP (ASiR-V 0%) — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_100_fbp[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title="ASiR-V 50% — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_100_ir[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text="100 kVp · 250 mA · CTDIvol 10.53 mGy", fontsize=16, font=:bold)
	
	fig
end

# ╔═╡ c2b384e5-0ebd-4e32-bbcc-d7699c5bb9c8
begin  # 140 kVp  (110 mA, 10.85 mGy)
	dcms_140_fbp = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_110mA_10.85mGyCTDI/0%"))
	dcms_140_ir = DCM.dcmdir_parse(joinpath(rootdir, "140kVp_110mA_10.85mGyCTDI/50%"))
	hu_140_fbp = load_hu_volume(dcms_140_fbp)
	hu_140_ir = load_hu_volume(dcms_140_ir)
end;

# ╔═╡ 79d134f6-1a23-4c7d-af2a-4c70a57841db
let
	mid = size(hu_140_fbp, 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title="FBP (ASiR-V 0%) — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_140_fbp[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title="ASiR-V 50% — $wname", titlesize=12,
			aspect=CM.DataAspect(), yreversed=true)
		CM.heatmap!(ax, hu_140_ir[:, :, mid]; colormap=:grays, colorrange=clim)
		CM.hidedecorations!(ax); CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text="140 kVp · 110 mA · CTDIvol 10.85 mGy", fontsize=16, font=:bold)
	
	fig
end

# ╔═╡ 262d5985-d5fa-49f1-968d-f760ac4c6fc7
md"""
### 1b. Automatic Rod Segmentation — 120 kVp / ASiR-V 50% (10.16 mGy)

Intensity-based segmentation of all 16 Gammex 472 insert rods from the clinical
120 kVp mid-dose ASiR-V 50% reconstruction. The algorithm:
1. Finds the phantom center via thresholded centroid + iterative circular refinement
2. Detects phantom rotation by locating Ca 400 (highest HU on the outer ring)
3. Places circular ROIs (70% of rod diameter) at all 16 known angular positions

This provides ground-truth ROI measurements (mean HU, noise σ, pixel count) for
quantitative comparison against BasisSimulator forward projections.
"""

# ╔═╡ dca3c26e-3d13-407f-aa83-f8a9ce5f9d23
"""
	segment_gammex_rods(hu_slice; fov_cm, ...) → (mask, rod_info, center_info)

Segment all 16 Gammex 472 insert rods from a 2D CT slice.
Uses known phantom geometry + intensity-based rotation detection.

Returns:
- `mask`: UInt8 2D array with rod labels (10-14=Ca, 20-26=I, 2=water, 3=SW)
- `rod_info`: Vector of NamedTuples with per-rod measurements
- `center_info`: NamedTuple `(cx, cy, rotation_deg)`
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
	body_r² = (body_radius_cm / pixel_cm)^2
	for _ in 1:3
		sx, sy, cnt = 0.0, 0.0, 0.0
		for j in 1:ny, i in 1:nx
			if (i - cx)^2 + (j - cy)^2 <= body_r² && body[i, j]
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
	sample_angles = range(0, 2π - 2π / n_sample, length=n_sample)

	profile = zeros(Float64, n_sample)
	for (k, θ) in enumerate(sample_angles)
		s, c = 0.0, 0
		for dr in -3:3
			xi = round(Int, cx + (r_outer_pix + dr) * cos(θ))
			yi = round(Int, cy + (r_outer_pix + dr) * sin(θ))
			if 1 <= xi <= nx && 1 <= yi <= ny
				s += hu_slice[xi, yi]; c += 1
			end
		end
		profile[k] = c > 0 ? s / c : 0.0
	end

	# Smooth with ±5° circular window
	smooth_w = max(1, round(Int, 5.0 / (360.0 / n_sample)))
	smoothed = similar(profile)
	for k in 1:n_sample
		s, c = 0.0, 0
		for d in -smooth_w:smooth_w
			s += profile[mod1(k + d, n_sample)]; c += 1
		end
		smoothed[k] = s / c
	end

	# Ca400 is the highest HU rod on the outer ring
	θ_ca400 = sample_angles[argmax(smoothed)]
	rotation = θ_ca400

	# Step 3: Place ROIs at all 16 rod positions
	# Clinical DICOM has CCW angular ordering (+ direction) due to image orientation
	outer_start = θ_ca400 - 3 * π/4  # Ca100 is 3 steps before Ca400
	outer_angles = [outer_start + (i - 1) * π/4 for i in 1:8]
	outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]
	outer_names = ["Ca 100", "Ca 200", "Ca 300", "Ca 400",
		"Water (O)", "SW ref 1", "SW ref 2", "Ca 50"]

	inner_start = outer_start - π/8  # I2.5 at gap center (π/8 before Ca100)
	inner_angles = [inner_start + (i - 1) * π/4 for i in 1:8]
	inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]
	inner_names = ["I 2.5", "I 5.0", "I 7.5", "I 10.0",
		"I 15.0", "I 20.0", "Water (I)", "I 2.0"]

	roi_r_pix = rod_radius_cm * roi_fraction / pixel_cm
	roi_r² = roi_r_pix^2

	mask = zeros(UInt8, nx, ny)
	rod_info = []

	for (angles, labels, names, ring_cm, ring_sym) in [
		(outer_angles, outer_labels, outer_names, outer_ring_cm, :outer),
		(inner_angles, inner_labels, inner_names, inner_ring_cm, :inner),
	]
		ring_pix = ring_cm / pixel_cm
		for (θ, lbl, name) in zip(angles, labels, names)
			rcx = cx + ring_pix * cos(θ)
			rcy = cy + ring_pix * sin(θ)

			i_lo = max(1, floor(Int, rcx - roi_r_pix - 1))
			i_hi = min(nx, ceil(Int, rcx + roi_r_pix + 1))
			j_lo = max(1, floor(Int, rcy - roi_r_pix - 1))
			j_hi = min(ny, ceil(Int, rcy + roi_r_pix + 1))

			vals = Float64[]
			for j in j_lo:j_hi, i in i_lo:i_hi
				if (i - rcx)^2 + (j - rcy)^2 <= roi_r²
					mask[i, j] = lbl
					push!(vals, Float64(hu_slice[i, j]))
				end
			end

			push!(rod_info, (
				label = lbl, name = name, ring = ring_sym,
				cx = rcx, cy = rcy,
				angle_deg = round(rad2deg(θ), digits=1),
				mean_hu = isempty(vals) ? NaN : mean(vals),
				std_hu = length(vals) > 1 ? std(vals) : NaN,
				n_pixels = length(vals),
			))
		end
	end

	return mask, rod_info, (cx=cx, cy=cy, rotation_deg=round(rad2deg(rotation), digits=2))
end

# ╔═╡ 63b0c15a-8e01-4f8b-9ff7-a9109e2b5933
seg_result = let
	mid_z = size(hu_120_mid_ir, 3) ÷ 2
	hu_slice = hu_120_mid_ir[:, :, mid_z]
	mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm=35.0)
	(mask=mask, rods=rod_info, center=center, slice_idx=mid_z)
end;

# ╔═╡ 451bfa43-42d9-4023-afb0-52bbf4b5a487
let
	hu = hu_120_mid_ir[:, :, seg_result.slice_idx]
	rods = seg_result.rods
	pixel_cm = 35.0 / size(hu, 1)
	roi_r_pix = 1.4 * 0.7 / pixel_cm

	fig = CM.Figure(size=(1100, 500), fontsize=11)

	# Left: CT with ROI circles
	ax1 = CM.Axis(fig[1, 1]; title="120 kVp ASiR-V 50% — Segmented ROIs (slice $(seg_result.slice_idx))",
		aspect=CM.DataAspect(), yreversed=true)
	CM.heatmap!(ax1, hu; colormap=:grays, colorrange=(-200, 500))

	θ_circle = range(0, 2π, length=61)
	for r in rods
		xs = r.cx .+ roi_r_pix .* cos.(θ_circle)
		ys = r.cy .+ roi_r_pix .* sin.(θ_circle)
		c = r.ring == :outer ? :orange : :lime
		CM.lines!(ax1, xs, ys; color=c, linewidth=1.5)
		CM.text!(ax1, r.cx, r.cy + roi_r_pix + 4;
			text=r.name, fontsize=7, align=(:center, :bottom), color=c)
	end
	CM.scatter!(ax1, [seg_result.center.cx], [seg_result.center.cy];
		color=:red, marker=:cross, markersize=12)
	CM.hidedecorations!(ax1); CM.hidespines!(ax1)

	# Right: labeled mask overlay
	ax2 = CM.Axis(fig[1, 2]; title="Rod Label Mask", aspect=CM.DataAspect(), yreversed=true)
	mask_vis = Float32.(seg_result.mask)
	mask_vis[mask_vis .== 0] .= NaN
	CM.heatmap!(ax2, hu; colormap=:grays, colorrange=(-200, 500))
	CM.heatmap!(ax2, mask_vis; colormap=:turbo, colorrange=(1, 27), nan_color=:transparent)
	CM.hidedecorations!(ax2); CM.hidespines!(ax2)

	fig
end

# ╔═╡ 04a99179-ae4f-40ba-b4bf-aa5f5276bbc1
let
	rods = seg_result.rods

	# Reorder: outer water/SW → Ca ascending → inner water → I ascending
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

	fig = CM.Figure(size=(1000, 500), fontsize=11)
	ax = CM.Axis(fig[1, 1];
		title="ROI Measurements — 120 kVp · 150 mA · ASiR-V 50% [rotation=$(seg_result.center.rotation_deg)°]",
		ylabel="Mean HU ± σ",
		xticks=(1:n, names),
		xticklabelrotation=π / 4)
	CM.barplot!(ax, 1:n, means; color=colors)
	CM.errorbars!(ax, 1:n, means, stds; color=:black, whiskerwidth=4)

	for (i, (m, s)) in enumerate(zip(means, stds))
		CM.text!(ax, i, m + s + 5;
			text="$(round(Int, m))", fontsize=8, align=(:center, :bottom))
	end

	fig
end

# ╔═╡ 3c0d7acf-6de3-4662-90b9-05cebcfffd1c
md"""
## 2. Dual-Energy 80/140 kVp — Virtual Monoenergetic Images (VMI)

GE GSI (Gemstone Spectral Imaging) acquisition at 80/140 kVp rapid switching.
CTDIvol = 10.07 mGy, DLP = 40.27 mGy·cm, 0.25 s rotation, 40 mm collimation
(64 rows × 0.625 mm). VMI reconstructions at 40, 70, 100, and 140 keV.

Same three display windows as single-energy:
- **Soft tissue** W400 / L40
- **Bone** W1500 / L400
- **Lung** W1500 / L−600
"""

# ╔═╡ c7df06b6-014e-4115-a811-cb5a190cbb2a
begin  # DE 80/140 kVp — VMI series
	dcms_de_40keV = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/40"))
	dcms_de_70keV = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/70"))
	dcms_de_100keV = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/100"))
	dcms_de_140keV = DCM.dcmdir_parse(joinpath(rootdir, "DE_80_140kVp_10.07mGyCTDI/140"))
	hu_de_40keV = load_hu_volume(dcms_de_40keV)
	hu_de_70keV = load_hu_volume(dcms_de_70keV)
	hu_de_100keV = load_hu_volume(dcms_de_100keV)
	hu_de_140keV = load_hu_volume(dcms_de_140keV)
end;

# ╔═╡ b5a1b6ce-a719-4444-8f70-76aff9040515
let
	vols = [hu_de_40keV, hu_de_70keV, hu_de_100keV, hu_de_140keV]
	labels = ["40 keV", "70 keV", "100 keV", "140 keV"]
	mid = size(vols[1], 3) ÷ 2
	soft = (-160, 240)
	bone = (-350, 1150)
	lung = (-1350, 150)
	windows = [(soft, "Soft Tissue"), (bone, "Bone"), (lung, "Lung")]

	fig = CM.Figure(size=(1500, 2000))

	for (row, (vol, keV_label)) in enumerate(zip(vols, labels))
		for (col, (clim, wname)) in enumerate(windows)
			ax = CM.Axis(fig[row, col]; title="$keV_label — $wname", titlesize=12,
				aspect=CM.DataAspect(), yreversed=true)
			CM.heatmap!(ax, vol[:, :, mid]; colormap=:grays, colorrange=clim)
			CM.hidedecorations!(ax); CM.hidespines!(ax)
		end
	end

	CM.Label(fig[0, :]; text="DE 80/140 kVp — VMI (CTDIvol 10.07 mGy)", fontsize=16, font=:bold)
	fig
end

# ╔═╡ ead991fd-d38c-4ebe-81f8-c35c84d0df52
md"""
### 2b. Automatic Rod Segmentation — DE 100 keV VMI (10.07 mGy)

Same `segment_gammex_rods` algorithm applied to the 100 keV virtual monoenergetic
reconstruction from the dual-energy 80/140 kVp acquisition. At 100 keV, calcium and
iodine rods have moderate contrast — close to conventional 120 kVp appearance — making
it a good reference energy for ROI validation.
"""

# ╔═╡ 88990577-029e-4f3b-be94-3ac413e526fe
seg_result_de = let
	mid_z = size(hu_de_100keV, 3) ÷ 2
	hu_slice = hu_de_100keV[:, :, mid_z]
	mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm=35.0)
	(mask=mask, rods=rod_info, center=center, slice_idx=mid_z)
end;

# ╔═╡ ce7b3519-78ee-43e0-a8a9-5b0270a68901
let
	hu = hu_de_100keV[:, :, seg_result_de.slice_idx]
	rods = seg_result_de.rods
	pixel_cm = 35.0 / size(hu, 1)
	roi_r_pix = 1.4 * 0.7 / pixel_cm

	fig = CM.Figure(size=(1100, 500), fontsize=11)

	ax1 = CM.Axis(fig[1, 1]; title="DE 100 keV VMI — Segmented ROIs (slice $(seg_result_de.slice_idx))",
		aspect=CM.DataAspect(), yreversed=true)
	CM.heatmap!(ax1, hu; colormap=:grays, colorrange=(-200, 500))

	θ_circle = range(0, 2π, length=61)
	for r in rods
		xs = r.cx .+ roi_r_pix .* cos.(θ_circle)
		ys = r.cy .+ roi_r_pix .* sin.(θ_circle)
		c = r.ring == :outer ? :orange : :lime
		CM.lines!(ax1, xs, ys; color=c, linewidth=1.5)
		CM.text!(ax1, r.cx, r.cy + roi_r_pix + 4;
			text=r.name, fontsize=7, align=(:center, :bottom), color=c)
	end
	CM.scatter!(ax1, [seg_result_de.center.cx], [seg_result_de.center.cy];
		color=:red, marker=:cross, markersize=12)
	CM.hidedecorations!(ax1); CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title="Rod Label Mask", aspect=CM.DataAspect(), yreversed=true)
	mask_vis = Float32.(seg_result_de.mask)
	mask_vis[mask_vis .== 0] .= NaN
	CM.heatmap!(ax2, hu; colormap=:grays, colorrange=(-200, 500))
	CM.heatmap!(ax2, mask_vis; colormap=:turbo, colorrange=(1, 27), nan_color=:transparent)
	CM.hidedecorations!(ax2); CM.hidespines!(ax2)

	fig
end

# ╔═╡ e26846b8-719f-4d24-a531-7aac1fe6672a
let
	rods = seg_result_de.rods

	# Reorder: outer water/SW → Ca ascending → inner water → I ascending
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

	fig = CM.Figure(size=(1000, 500), fontsize=11)
	ax = CM.Axis(fig[1, 1];
		title="ROI Measurements — DE 100 keV VMI [rotation=$(seg_result_de.center.rotation_deg)°]",
		ylabel="Mean HU ± σ",
		xticks=(1:n, names),
		xticklabelrotation=π / 4)
	CM.barplot!(ax, 1:n, means; color=colors)
	CM.errorbars!(ax, 1:n, means, stds; color=:black, whiskerwidth=4)

	for (i, (m, s)) in enumerate(zip(means, stds))
		CM.text!(ax, i, m + s + 5;
			text="$(round(Int, m))", fontsize=8, align=(:center, :bottom))
	end

	fig
end

# ╔═╡ a1b2c3d4-0010-4000-8000-000000000010
md"""
## 3. Quantitative Measurements — Full Suite

Comprehensive image quality measurements across all SE and DE reconstructions.
For each scan we compute:

| Metric | Method | Stored as |
|--------|--------|-----------|
| **Mean HU** per rod | ROI from `segment_gammex_rods` | 16 scalar columns |
| **Noise σ** per rod | Std dev within each ROI | 16 scalar columns |
| **CNR** per rod | `(HU_rod − HU_water) / σ_water` | 14 scalar columns |
| **NPS** (radial) | AAPM TG-233, 64×64 ROIs from body center | full curve → JLD2 |
| **MTF** (radial) | Circular-edge (body boundary) | full curve → JLD2 |

Scalar summary → `clinical_measurements.csv`
Full NPS/MTF curves → `nps_curves.jld2`, `mtf_curves.jld2`

The SE segmentation mask (from 120 kVp / 150 mA / ASiR-V 50%) is reused for all
6 SE scans × 2 recon types = 12 measurements. The DE mask (from 100 keV VMI) is
reused for all 4 DE VMI energies.
"""

# ╔═╡ a1b2c3d4-0011-4000-8000-000000000011
begin
	"""
		measure_mtf_circular_edge(hu_slice, cx, cy, radius_cm; fov_cm, n_angles, oversample, margin_pix)

	Circular-edge MTF from a circular boundary (rod insert or body edge).
	Samples radial ESF profiles at `n_angles` around the edge,
	averages them (angular super-resolution), differentiates → LSF → FFT → MTF.

	Returns: (frequencies_lp_cm, mtf_curve, f50, f10)
	"""
	function measure_mtf_circular_edge(
		hu_slice::AbstractMatrix,
		cx::Float64, cy::Float64,
		radius_cm::Float64;
		fov_cm = 35.0,
		n_angles = 360,
		oversample = 4,
		margin_pix = 12.0,
	)
		nx, ny = size(hu_slice)
		pixel_cm = fov_cm / nx
		pixel_mm = pixel_cm * 10.0
		body_r_pix = radius_cm / pixel_cm

		# Sample radial ESF: from edge - margin to edge + margin
		r_min = body_r_pix - margin_pix
		r_max = body_r_pix + margin_pix
		n_r = round(Int, (r_max - r_min) * oversample)
		radii = range(r_min, r_max, length=n_r)
		positions_mm = collect((radii .- body_r_pix) .* pixel_mm)  # mm from edge

		# Average ESF across all angles
		esf = zeros(Float64, n_r)
		counts = zeros(Int, n_r)
		angles = range(0, 2π - 2π/n_angles, length=n_angles)

		for θ in angles
			cosθ, sinθ = cos(θ), sin(θ)
			for (k, r) in enumerate(radii)
				xi = cx + r * cosθ
				yi = cy + r * sinθ
				# Bilinear interpolation
				x0 = floor(Int, xi); y0 = floor(Int, yi)
				x1 = x0 + 1; y1 = y0 + 1
				if 1 <= x0 && x1 <= nx && 1 <= y0 && y1 <= ny
					fx = xi - x0; fy = yi - y0
					val = (1-fx)*(1-fy)*hu_slice[x0,y0] + fx*(1-fy)*hu_slice[x1,y0] +
						  (1-fx)*fy*hu_slice[x0,y1] + fx*fy*hu_slice[x1,y1]
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

		# Ensure ESF goes from high (inside body) to low (outside / air)
		if esf[1] < esf[end]
			reverse!(esf)
			reverse!(positions_mm)
		end

		# Differentiate ESF → LSF
		dp = positions_mm[2] - positions_mm[1]  # mm spacing
		lsf = diff(esf) ./ dp
		lsf_pos = (positions_mm[1:end-1] .+ positions_mm[2:end]) ./ 2

		# Normalize LSF peak to 1
		lsf_max = maximum(abs.(lsf))
		if lsf_max > 0
			lsf ./= lsf_max
		end

		# Zero-pad and FFT
		n_pad = nextpow(2, length(lsf) * 4)
		lsf_padded = zeros(Float64, n_pad)
		offset = (n_pad - length(lsf)) ÷ 2
		lsf_padded[offset+1:offset+length(lsf)] .= lsf

		mtf_complex = fft(lsf_padded)
		mtf_vals = abs.(mtf_complex)
		mtf_vals ./= mtf_vals[1]  # normalize DC = 1

		# Frequency axis (lp/cm)
		n_pos = n_pad ÷ 2
		freq_lp_mm = collect(0:n_pos-1) ./ n_pad .* (1.0 / abs(dp))
		freq_lp_cm = freq_lp_mm .* 10.0
		mtf_curve = mtf_vals[1:n_pos]

		# Trim to Nyquist of the original pixel grid
		nyquist_lp_cm = 1.0 / (2.0 * pixel_mm) * 10.0
		keep = freq_lp_cm .<= nyquist_lp_cm
		freq_lp_cm = freq_lp_cm[keep]
		mtf_curve = mtf_curve[keep]

		# Find f50 and f10 by interpolation
		function find_freq_at(level)
			for i in 1:(length(mtf_curve)-1)
				if mtf_curve[i] >= level && mtf_curve[i+1] < level
					t = (level - mtf_curve[i]) / (mtf_curve[i+1] - mtf_curve[i])
					return freq_lp_cm[i] + t * (freq_lp_cm[i+1] - freq_lp_cm[i])
				end
			end
			return mtf_curve[end] >= level ? freq_lp_cm[end] : 0.0
		end

		f50 = find_freq_at(0.5)
		f10 = find_freq_at(0.1)

		return (frequencies=freq_lp_cm, mtf=mtf_curve, mtf50=f50, mtf10=f10)
	end

	"""
		measure_scan(hu_vol, seg_mask, seg_rods, seg_center, scan_name; fov_cm) → NamedTuple

	Compute full measurement suite for one reconstruction.
	Uses the pre-computed segmentation mask from `segment_gammex_rods`.
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
		mid_z = nz ÷ 2
		hu_slice = hu_vol[:, :, mid_z]
		pixel_mm = fov_cm / nx * 10.0  # mm

		# --- Rod HU measurements (reuse mask position, sample from this scan) ---
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
			roi_r² = roi_r_pix^2
			vals = Float64[]
			i_lo = max(1, floor(Int, r.cx - roi_r_pix - 1))
			i_hi = min(nx, ceil(Int, r.cx + roi_r_pix + 1))
			j_lo = max(1, floor(Int, r.cy - roi_r_pix - 1))
			j_hi = min(ny, ceil(Int, r.cy + roi_r_pix + 1))
			for j in j_lo:j_hi, i in i_lo:i_hi
				if (i - r.cx)^2 + (j - r.cy)^2 <= roi_r²
					push!(vals, Float64(hu_slice[i, j]))
				end
			end
			push!(rod_means, mean(vals))
			push!(rod_stds, std(vals))
			push!(rod_names, name)
		end

		# Water reference for CNR (outer ring water rod)
		water_idx = 1  # "Water (O)"
		μ_water = rod_means[water_idx]
		σ_water = rod_stds[water_idx]

		rod_cnr = [(rod_means[i] - μ_water) / σ_water for i in 1:length(rod_means)]

		# --- NPS (from uniform body center, avoiding rods) ---
		center_row = round(Int, seg_center.cx)
		center_col = round(Int, seg_center.cy)
		nps_result = BS.measure_nps(
			hu_slice, pixel_mm;
			config = BS.NPSConfig(roi_size=64, n_rois=32, overlap=0.5),
			roi_center = (center_row, center_col),
			roi_radius_mm = 35.0,
			unit = :lp_cm,
		)

		# --- MTF (circular edge of Ca 400 rod — highest contrast insert) ---
		ca400_rod = first(filter(x -> x.name == "Ca 400", seg_rods))
		mtf_result = measure_mtf_circular_edge(
			hu_slice, ca400_rod.cx, ca400_rod.cy, 1.4;
			fov_cm = fov_cm, margin_pix = 12.0,
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
end;

# ╔═╡ a1b2c3d4-0012-4000-8000-000000000012
md"""
### 3a. Single-Energy Measurements (12 scans)

All SE scans use the segmentation mask from `seg_result` (120 kVp / 150 mA / ASiR-V 50%).
"""

# ╔═╡ a1b2c3d4-0013-4000-8000-000000000013
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

	[measure_scan(vol, seg_result.mask, seg_result.rods, seg_result.center, name)
	 for (vol, name) in se_scans]
end;

# ╔═╡ a1b2c3d4-0014-4000-8000-000000000014
md"""
### 3b. Dual-Energy Measurements (4 VMI energies)

All DE scans use the segmentation mask from `seg_result_de` (100 keV VMI).
"""

# ╔═╡ a1b2c3d4-0015-4000-8000-000000000015
de_measurements = let
	de_scans = [
		(hu_de_40keV, "DE_40keV"),
		(hu_de_70keV, "DE_70keV"),
		(hu_de_100keV, "DE_100keV"),
		(hu_de_140keV, "DE_140keV"),
	]

	[measure_scan(vol, seg_result_de.mask, seg_result_de.rods, seg_result_de.center, name)
	 for (vol, name) in de_scans]
end;

# ╔═╡ a1b2c3d4-0016-4000-8000-000000000016
md"""
### 3c. NPS Curves — All Scans
"""

# ╔═╡ a1b2c3d4-0017-4000-8000-000000000017
let
	all_m = vcat(se_measurements, de_measurements)

	fig = CM.Figure(size=(1200, 500), fontsize=11)
	ax1 = CM.Axis(fig[1, 1]; title="NPS — Single-Energy Scans",
		xlabel="Spatial frequency (lp/cm)", ylabel="NPS (HU²·cm²)",
		yscale=log10)
	ax2 = CM.Axis(fig[1, 2]; title="NPS — Dual-Energy VMI",
		xlabel="Spatial frequency (lp/cm)", ylabel="NPS (HU²·cm²)",
		yscale=log10)

	se_colors = CM.cgrad(:tab10, 12, categorical=true)
	for (i, m) in enumerate(se_measurements)
		freqs = m.nps.frequencies
		vals = m.nps.nps_1d
		good = vals .> 0
		CM.lines!(ax1, freqs[good], vals[good]; label=m.name, color=se_colors[i])
	end
	CM.axislegend(ax1; position=:rt, labelsize=8, nbanks=2)

	de_colors = [:purple, :teal, :darkorange, :crimson]
	for (i, m) in enumerate(de_measurements)
		freqs = m.nps.frequencies
		vals = m.nps.nps_1d
		good = vals .> 0
		CM.lines!(ax2, freqs[good], vals[good]; label=m.name, color=de_colors[i], linewidth=2)
	end
	CM.axislegend(ax2; position=:rt, labelsize=10)

	fig
end

# ╔═╡ a1b2c3d4-0018-4000-8000-000000000018
md"""
### 3d. MTF Curves — All Scans
"""

# ╔═╡ a1b2c3d4-0019-4000-8000-000000000019
let
	fig = CM.Figure(size=(1200, 500), fontsize=11)
	ax1 = CM.Axis(fig[1, 1]; title="MTF — Single-Energy Scans",
		xlabel="Spatial frequency (lp/cm)", ylabel="MTF",
		limits=(nothing, nothing, 0, 1.05))
	ax2 = CM.Axis(fig[1, 2]; title="MTF — Dual-Energy VMI",
		xlabel="Spatial frequency (lp/cm)", ylabel="MTF",
		limits=(nothing, nothing, 0, 1.05))

	CM.hlines!(ax1, [0.5, 0.1]; color=:gray80, linestyle=:dash, linewidth=0.8)
	CM.hlines!(ax2, [0.5, 0.1]; color=:gray80, linestyle=:dash, linewidth=0.8)

	se_colors = CM.cgrad(:tab10, 12, categorical=true)
	for (i, m) in enumerate(se_measurements)
		CM.lines!(ax1, m.mtf.frequencies, m.mtf.mtf; label=m.name, color=se_colors[i])
	end
	CM.axislegend(ax1; position=:rt, labelsize=8, nbanks=2)

	de_colors = [:purple, :teal, :darkorange, :crimson]
	for (i, m) in enumerate(de_measurements)
		CM.lines!(ax2, m.mtf.frequencies, m.mtf.mtf; label=m.name, color=de_colors[i], linewidth=2)
	end
	CM.axislegend(ax2; position=:rt, labelsize=10)

	fig
end

# ╔═╡ a1b2c3d4-0020-4000-8000-000000000020
md"""
### 3e. Summary Table & Export

Scalar summary exported to CSV. Full NPS/MTF curves exported to JLD2.
"""

# ╔═╡ a1b2c3d4-0021-4000-8000-000000000021
let
	all_m = vcat(se_measurements, de_measurements)
	outdir = joinpath(dirname(@__DIR__), "results")
	mkpath(outdir)

	# --- Build CSV rows ---
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
	append!(header, ["nps_peak_freq_lp_cm", "nps_area_HU2cm2",
		"mtf_f50_lp_cm", "mtf_f10_lp_cm"])

	rows = Vector{Any}[]
	for m in all_m
		row = Any[m.name]
		append!(row, round.(m.rod_means, digits=2))
		append!(row, round.(m.rod_stds, digits=2))
		append!(row, round.(m.rod_cnr, digits=2))
		push!(row, round(m.nps_peak_freq, digits=3))
		push!(row, round(m.nps_area, digits=3))
		push!(row, round(m.mtf_f50, digits=3))
		push!(row, round(m.mtf_f10, digits=3))
		push!(rows, row)
	end

	# Write CSV
	csv_path = joinpath(outdir, "clinical_measurements.csv")
	open(csv_path, "w") do io
		println(io, join(header, ","))
		for row in rows
			println(io, join(row, ","))
		end
	end

	# --- Save NPS curves to JLD2 ---
	nps_path = joinpath(outdir, "nps_curves.jld2")
	JLD2.jldopen(nps_path, "w") do f
		for m in all_m
			f[m.name] = (freq=m.nps.frequencies, nps=m.nps.nps_1d)
		end
	end

	# --- Save MTF curves to JLD2 ---
	mtf_path = joinpath(outdir, "mtf_curves.jld2")
	JLD2.jldopen(mtf_path, "w") do f
		for m in all_m
			f[m.name] = (freq=m.mtf.frequencies, mtf=m.mtf.mtf)
		end
	end

	md"""
	**Exported:**
	- `results/clinical_measurements.csv` — $(length(all_m)) rows × $(length(header)) columns
	- `results/nps_curves.jld2` — radial NPS curves (freq + values per scan)
	- `results/mtf_curves.jld2` — radial MTF curves (freq + values per scan)
	"""
end

# ╔═╡ a1b2c3d4-0022-4000-8000-000000000022
md"""
### 3f. Scalar Summary — Quick View
"""

# ╔═╡ a1b2c3d4-0023-4000-8000-000000000023
let
	all_m = vcat(se_measurements, de_measurements)
	n = length(all_m)

	fig = CM.Figure(size=(1400, 400), fontsize=10)

	# NPS area (noise variance)
	ax1 = CM.Axis(fig[1, 1]; title="NPS Area (Noise Variance)",
		ylabel="NPS integral (HU²·cm²)",
		xticks=(1:n, [m.name for m in all_m]),
		xticklabelrotation=π/3)
	CM.barplot!(ax1, 1:n, [m.nps_area for m in all_m];
		color=vcat(fill(:steelblue, length(se_measurements)),
			fill(:darkorange, length(de_measurements))))

	# MTF f50 and f10
	ax2 = CM.Axis(fig[1, 2]; title="MTF f₅₀ and f₁₀",
		ylabel="Frequency (lp/cm)",
		xticks=(1:n, [m.name for m in all_m]),
		xticklabelrotation=π/3)
	CM.barplot!(ax2, collect(1:n) .- 0.15, [m.mtf_f50 for m in all_m];
		width=0.3, color=:teal, label="f₅₀")
	CM.barplot!(ax2, collect(1:n) .+ 0.15, [m.mtf_f10 for m in all_m];
		width=0.3, color=:salmon, label="f₁₀")
	CM.axislegend(ax2; position=:rt)

	fig
end

# ╔═╡ d9ad5eb6-df82-466b-a291-310417adc0bf
md"""
## 4. Matching Gammex 472 Phantom

Digital replica of the physical Gammex 472 multi-energy CT phantom at **0.2 mm isotropic**
resolution (1750 × 1750 × 250 voxels, 35 cm FOV × 5 cm thick). The layout uses **8 rods
per ring** matching the actual phantom, with a **0.3 mm air gap** around each insert
(the clearance needed to slide rods in and out). The mask is transposed + flipped to
match the clinical DICOM display orientation.

**Body:** 330 mm diameter Gammex Model 451 Solid Water (ρ = 1.02 g/cm³)

**Outer ring** (R = 10.5 cm, 8 positions at 45°, gap at 12 o'clock CW):
Ca100 → Ca200 → Ca300 → Ca400 → Water → SW ref → SW ref → Ca50

**Inner ring** (R = 5.0 cm, 8 positions at 45°, starts at 12 o'clock CW):
I2.5 → I5.0 → I7.5 → I10 → I15 → I20 → Water → I2.0

**Rod diameter:** 28 mm | **Air gap:** 0.3 mm | **Labels:** 0 = air, 1 = solid water,
2 = pure water, 3 = SW ref, 10–14 = Ca (50–400 mg/mL), 20–26 = I (2.0–20.0 mg/mL)
"""

# ╔═╡ 830ee323-3fd3-464d-a4bf-20348c206909
begin
	const GAMMEX_SOLID_WATER = XA.Materials.gammex_472_solidwater

	const ALL_INSERT_MATERIALS = Dict{UInt8, XA.Material}(
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

	const MATERIAL_INFO = Dict(
		UInt8(0) => (name="Air", color=:gray15),
		UInt8(1) => (name="Solid Water", color=:lightskyblue),
		UInt8(2) => (name="Pure Water", color=:royalblue),
		UInt8(3) => (name="SW Reference", color=:paleturquoise),
		UInt8(10) => (name="Ca 50 mg/mL", color=:wheat),
		UInt8(11) => (name="Ca 100 mg/mL", color=:sandybrown),
		UInt8(12) => (name="Ca 200 mg/mL", color=:orange),
		UInt8(13) => (name="Ca 300 mg/mL", color=:darkorange),
		UInt8(14) => (name="Ca 400 mg/mL", color=:orangered),
		UInt8(20) => (name="I 2.0 mg/mL", color=:honeydew),
		UInt8(21) => (name="I 2.5 mg/mL", color=:palegreen),
		UInt8(22) => (name="I 5.0 mg/mL", color=:lightgreen),
		UInt8(23) => (name="I 7.5 mg/mL", color=:mediumseagreen),
		UInt8(24) => (name="I 10.0 mg/mL", color=:seagreen),
		UInt8(25) => (name="I 15.0 mg/mL", color=:forestgreen),
		UInt8(26) => (name="I 20.0 mg/mL", color=:darkgreen),
	)

function create_custom_gammex_472(;
		n_voxels::Int = 1750,
		n_slices::Int = 250,
		fov_cm::Float64 = 35.0,
		z_cm::Float64 = 5.0,
	)
		dx = fov_cm / n_voxels
		dy = fov_cm / n_voxels
		dz = z_cm / n_slices

		x = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)
		y = range(-fov_cm/2 + dy/2, fov_cm/2 - dy/2, length=n_voxels)

		body_radius = 16.5
		rod_radius = 1.4
		air_gap = 0.03         # 0.3mm clearance around each rod
		rod_radius² = rod_radius^2
		hole_radius² = (rod_radius + air_gap)^2
		outer_ring_R = 10.5
		inner_ring_R = 5.5

		outer_start = π/2 - π/8
		outer_angles = [outer_start - (i-1) * π/4 for i in 1:8]
		outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]

		inner_start = π/2
		inner_angles = [inner_start - (i-1) * π/4 for i in 1:8]
		inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]

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
					d² = (xi - outer_cx[idx])^2 + (yj - outer_cy[idx])^2
					if d² <= hole_radius²
						slice[i, j] = d² <= rod_radius² ? outer_labels[idx] : UInt8(0)
						@goto next_voxel
					end
				end

				for idx in 1:8
					d² = (xi - inner_cx[idx])^2 + (yj - inner_cy[idx])^2
					if d² <= hole_radius²
						slice[i, j] = d² <= rod_radius² ? inner_labels[idx] : UInt8(0)
						break
					end
				end
			end
			@label next_voxel
		end

		# Flip in y to match clinical DICOM orientation
		slice = reverse(permutedims(rot180(slice)), dims=2)

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

		origin = (-fov_cm/2 + dx/2, -fov_cm/2 + dy/2, -z_cm/2 + dz/2)
		extent = (Float64(fov_cm), Float64(fov_cm), Float64(z_cm))

		return BS.Phantom(mask, materials_vec, (dx, dy, dz), origin, extent)
	end

	phantom = create_custom_gammex_472()
	phantom_mask = phantom.mask
end;

# ╔═╡ 0a8833cf-06c6-4777-9fe8-c17f7138e6b8
let
	mid = size(phantom_mask, 3) ÷ 2
	nz = size(phantom_mask, 3)
	slice_data = phantom_mask[:, :, mid]

	unique_labels = sort(unique(slice_data))
	n_labels = length(unique_labels)

	lut = zeros(Float32, 27)
	for (i, l) in enumerate(unique_labels)
		lut[Int(l) + 1] = Float32(i)
	end
	mapped = lut[Int.(slice_data) .+ 1]

	colors = [MATERIAL_INFO[l].color for l in unique_labels]
	cmap = CM.cgrad(colors, n_labels, categorical=true)
	names = [MATERIAL_INFO[l].name for l in unique_labels]

	fig = CM.Figure(size=(1000, 850), fontsize=12)
	ax = CM.Axis(fig[1, 1];
		title="Custom Gammex 472 — Slice $mid / $nz (0.2mm voxels)",
		aspect=CM.DataAspect())
	hm = CM.heatmap!(ax, mapped; colormap=cmap, colorrange=(0.5, n_labels + 0.5))
	CM.Colorbar(fig[1, 2], hm; ticks=(1:n_labels, names), ticklabelsize=11, width=15)
	fig
end

# ╔═╡ a1b2c3d4-5001-4000-8000-000000000001
md"""
## 5. Single-kVp Simulation — GE Revolution Apex Elite

BasisSimulator forward projection + FDK reconstruction matching the 6 SE clinical
protocols. Each simulated scan uses the same scanner geometry, kVp, mA, rotation time,
and collimation as the clinical acquisition, enabling direct clinical-vs-simulated
comparison of HU accuracy, noise, and spatial resolution.

**Phantom:** High-resolution (1750×1750×250, 0.2mm isotropic) Gammex 472 digital replica
with z = 5 cm matching the real phantom thickness. The 80mm collimation extends beyond
the phantom, with air above and below — exactly as in the clinical acquisition.
The high-resolution input avoids artificial pixelation in the forward projection.

**Reconstruction:** 512×512×64 at 35cm FOV / 8cm z (matching clinical 0.684mm pixels,
1.25mm slices, STANDARD kernel).
"""

# ╔═╡ a1b2c3d4-5002-4000-8000-000000000002
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
	target_angle = 7.0,
	flat_filter_material = :aluminum,
	flat_filter_thickness = 2.5,
	detector_material = :gos,
	detector_depth = 3.0,
	fill_factor_row = 0.9,
	fill_factor_col = 0.9,
	detection_gain = 1.0,
)

# ╔═╡ a1b2c3d4-5003-4000-8000-000000000003
begin
	sim_rotation_time = 1.0     # seconds (clinical 1.0s rotation)
	sim_collimation_mm = 80.0   # 128 × 0.625mm active rows
	sim_n_views = 984           # standard GE Revolution

	SE_SIM_SCANS = [
		(name="120kVp_50mA_FBP",    kvp=120, mA=50.0),
		(name="120kVp_150mA_FBP",   kvp=120, mA=150.0),
		(name="120kVp_300mA_FBP",   kvp=120, mA=300.0),
		(name="80kVp_480mA_FBP",    kvp=80,  mA=480.0),
		(name="100kVp_250mA_FBP",   kvp=100, mA=250.0),
		(name="140kVp_110mA_FBP",   kvp=140, mA=110.0),
	]

	sim_protocols = Dict(
		sc.name => BS.CTProtocol(
			kVp = sc.kvp,
			mA = sc.mA,
			views = sim_n_views,
			rotation_time = sim_rotation_time,
			collimation_mm = sim_collimation_mm,
		)
		for sc in SE_SIM_SCANS
	)
end

# ╔═╡ a1b2c3d4-5004-4000-8000-000000000004
begin
	sim_opts = BS.SimOptions(fidelity = :high, seed = 1234)

	sim_recon_xy = 512
	sim_recon_fov_cm = 35.0
	sim_slice_thickness_mm = 1.25
	sim_recon_z_cm = sim_collimation_mm / 10.0                            # 8.0 cm
	sim_n_recon_slices = round(Int, sim_collimation_mm / sim_slice_thickness_mm)  # 64

	recon_opts_fbp = BS.ReconOptions(
		algorithm = :fdk,
		matrix_size = (sim_recon_xy, sim_recon_xy, sim_n_recon_slices),
		fov_cm = sim_recon_fov_cm,
		z_cm = sim_recon_z_cm,
		filter = :standard,
	)
end

# ╔═╡ a1b2c3d4-5005-4000-8000-000000000005
begin
	# Real Gammex 472 is 5cm thick — sits inside the 8cm collimation window
	# with air above/below, exactly matching the clinical acquisition geometry
	sim_phantom_z_cm = 5.0

	# High-res phantom: 0.2mm isotropic for accurate forward projection
	sim_phantom = create_custom_gammex_472(
		n_voxels = 1750,   # 35cm / 1750 = 0.02cm = 0.2mm xy
		n_slices = 250,    # 5cm / 250 = 0.02cm = 0.2mm z (isotropic)
		fov_cm = sim_recon_fov_cm,
		z_cm = sim_phantom_z_cm,
	)

	sim_phantom_gpu = BS.Phantom(
		MtlArray(sim_phantom.mask),
		sim_phantom.materials,
		sim_phantom.voxel_size,
		sim_phantom.origin,
		sim_phantom.extent,
	)
end;

# ╔═╡ a1b2c3d4-5006-4000-8000-000000000006
# ╠═╡ show_logs = false
μ_water_cal = let
	# Water phantom — uniform, so coarse resolution (512×512) is fine
	# Same 5cm z-extent as Gammex phantom (air above/below in 8cm collimation)
	nx, ny = sim_recon_xy, sim_recon_xy
	nz_water = 40
	voxel_cm = sim_recon_fov_cm / nx
	voxel_z_cm = sim_phantom_z_cm / nz_water

	water_mask = zeros(UInt8, nx, ny, nz_water)
	radius_cm = 16.5
	xs = range(-sim_recon_fov_cm/2 + voxel_cm/2, sim_recon_fov_cm/2 - voxel_cm/2, length=nx)
	ys = range(-sim_recon_fov_cm/2 + voxel_cm/2, sim_recon_fov_cm/2 - voxel_cm/2, length=ny)
	for k in 1:nz_water, j in 1:ny, i in 1:nx
		if sqrt(xs[i]^2 + ys[j]^2) <= radius_cm
			water_mask[i, j, k] = UInt8(1)
		end
	end

	water_materials = [XA.Materials.air, XA.Materials.water]
	origin = (-sim_recon_fov_cm/2 + voxel_cm/2, -sim_recon_fov_cm/2 + voxel_cm/2, -sim_phantom_z_cm/2 + voxel_z_cm/2)
	extent = (Float64(sim_recon_fov_cm), Float64(sim_recon_fov_cm), Float64(sim_phantom_z_cm))

	phantom_water = BS.Phantom(water_mask, water_materials, (voxel_cm, voxel_cm, voxel_z_cm), origin, extent)
	phantom_water_gpu = BS.Phantom(
		MtlArray(phantom_water.mask), phantom_water.materials,
		phantom_water.voxel_size, phantom_water.origin, phantom_water.extent,
	)

	recon_size = recon_opts_fbp.matrix_size
	result = Dict{Int, Float64}()

	for kvp in sort(unique([sc.kvp for sc in SE_SIM_SCANS]))
		ref_scan = first(filter(sc -> sc.kvp == kvp, SE_SIM_SCANS))
		prot = sim_protocols[ref_scan.name]

		@info "Water calibration: $(kvp) kVp..."
		ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, phantom_water_gpu)
		BS.simulate!(ws, phantom_water_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

		ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
		vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

		cx, cy, cz = size(vol) .÷ 2
		z_half = min(cz - 1, 4)
		μ_empirical = mean(vol[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(vol,3),cz+z_half)])

		result[kvp] = μ_empirical
		@info "  μ_water($kvp kVp) = $(round(μ_empirical, digits=6)) cm⁻¹"

		ws_fdk = nothing; ws = nothing; GC.gc(true)
	end

	phantom_water_gpu = nothing; GC.gc(true)
	result
end

# ╔═╡ a1b2c3d4-5007-4000-8000-000000000007
# ╠═╡ show_logs = false
# Scan 1: 120 kVp / 50 mA (low dose)
sim_scan_1 = let
	sc = SE_SIM_SCANS[1]
	prot = sim_protocols[sc.name]
	μ_w = μ_water_cal[sc.kvp]
	recon_size = recon_opts_fbp.matrix_size

	@info "BasisSim: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, sim_phantom_gpu)
	@time BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
	recon_hu = Float32.(Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	)))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(name=sc.name, recon=recon_hu, mu_water=μ_w)
end

# ╔═╡ a1b2c3d4-5008-4000-8000-000000000008
# ╠═╡ show_logs = false
# Scan 2: 120 kVp / 150 mA (mid dose)
sim_scan_2 = let
	sc = SE_SIM_SCANS[2]
	prot = sim_protocols[sc.name]
	μ_w = μ_water_cal[sc.kvp]
	recon_size = recon_opts_fbp.matrix_size

	@info "BasisSim: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, sim_phantom_gpu)
	@time BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
	recon_hu = Float32.(Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	)))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(name=sc.name, recon=recon_hu, mu_water=μ_w)
end

# ╔═╡ a1b2c3d4-5009-4000-8000-000000000009
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
# Scan 3: 120 kVp / 300 mA (high dose)
sim_scan_3 = let
	sc = SE_SIM_SCANS[3]
	prot = sim_protocols[sc.name]
	μ_w = μ_water_cal[sc.kvp]
	recon_size = recon_opts_fbp.matrix_size

	@info "BasisSim: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, sim_phantom_gpu)
	@time BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
	recon_hu = Float32.(Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	)))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(name=sc.name, recon=recon_hu, mu_water=μ_w)
end
  ╠═╡ =#

# ╔═╡ a1b2c3d4-5010-4000-8000-000000000010
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
# Scan 4: 80 kVp / 480 mA
sim_scan_4 = let
	sc = SE_SIM_SCANS[4]
	prot = sim_protocols[sc.name]
	μ_w = μ_water_cal[sc.kvp]
	recon_size = recon_opts_fbp.matrix_size

	@info "BasisSim: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, sim_phantom_gpu)
	@time BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
	recon_hu = Float32.(Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	)))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(name=sc.name, recon=recon_hu, mu_water=μ_w)
end
  ╠═╡ =#

# ╔═╡ a1b2c3d4-5011-4000-8000-000000000011
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
# Scan 5: 100 kVp / 250 mA
sim_scan_5 = let
	sc = SE_SIM_SCANS[5]
	prot = sim_protocols[sc.name]
	μ_w = μ_water_cal[sc.kvp]
	recon_size = recon_opts_fbp.matrix_size

	@info "BasisSim: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, sim_phantom_gpu)
	@time BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
	recon_hu = Float32.(Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	)))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(name=sc.name, recon=recon_hu, mu_water=μ_w)
end
  ╠═╡ =#

# ╔═╡ a1b2c3d4-5012-4000-8000-000000000012
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
# Scan 6: 140 kVp / 110 mA
sim_scan_6 = let
	sc = SE_SIM_SCANS[6]
	prot = sim_protocols[sc.name]
	μ_w = μ_water_cal[sc.kvp]
	recon_size = recon_opts_fbp.matrix_size

	@info "BasisSim: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, recon_opts_fbp, sim_phantom_gpu)
	@time BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, recon_opts_fbp)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts_fbp.filter)
	recon_hu = Float32.(Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	)))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(name=sc.name, recon=recon_hu, mu_water=μ_w)
end
  ╠═╡ =#

# ╔═╡ a1b2c3d4-5013-4000-8000-000000000013
sim_results_fbp = [
	sim_scan_1,
	sim_scan_2,
	# sim_scan_3,
	# sim_scan_4,
	# sim_scan_5,
	# sim_scan_6
]

# ╔═╡ a1b2c3d4-5014-4000-8000-000000000014
md"""
### 5a. HIR Reconstruction (Planned)

Hybrid Iterative Reconstruction can be applied to the same sinogram data for
noise reduction comparison. To enable, save sinograms in the individual scan cells
above and run:

```julia
ws_hir = BS.create_hir_recon_workspace(sinogram, geom, volume_size; strength=3)
BS.reconstruct!(ws_hir, sinogram, geom, volume_size)
hir_result = Array(ws_hir.volume)
hir_hu = BS.to_hounsfield(hir_result; μ_water=μ_w)
```
"""

# ╔═╡ e54d80d3-6f67-4cd4-a85f-0e150755bf6e
md"""
## 6. Measurements
"""

# ╔═╡ fc69166c-0bec-441c-a2e5-e40eeaac1831
# Uncomment ONE orient line that makes the sim recon match clinical DICOM orientation.
# Check the alignment plot below — green circles should land on rods.
sim_oriented = let
	orient(s) = s                                  # identity (no transform)
	orient(s) = rotr90(s)                        # 90° clockwise
	# orient(s) = rot180(s)                        # 180°
	# orient(s) = rotl90(s)                        # 90° counter-clockwise
	orient(s) = reverse(s, dims=2)               # flip left-right
	# orient(s) = reverse(s, dims=1)               # flip up-down
	# orient(s) = permutedims(s)                   # transpose
	# orient(s) = reverse(permutedims(s), dims=1)  # anti-transpose

	[(name=r.name, recon=Float32.(mapslices(orient, r.recon, dims=(1,2))), mu_water=r.mu_water)
	 for r in sim_results_fbp]
end;

# ╔═╡ c1349a54-553e-45ec-b37d-631629baa4e3
# Measurements using CLINICAL segmentation (seg_result) as ground truth
sim_measurements = [
	measure_scan(r.recon, seg_result.mask, seg_result.rods, seg_result.center, "sim_$(r.name)")
	for r in sim_oriented
];

# ╔═╡ 07d10464-2c39-4a4a-aeb0-5549c34b705d
let
	# Clinical reference slice (from seg_result)
	clin_hu = hu_120_mid_ir[:, :, seg_result.slice_idx]

	# Simulated — use its own mid-slice (different z-count than clinical)
	sim_vol = sim_oriented[min(2, end)].recon
	sim_mid = size(sim_vol, 3) ÷ 2
	sim_hu = sim_vol[:, :, sim_mid]

	# Mask overlay
	mask_vis = Float32.(seg_result.mask)
	mask_vis[mask_vis .== 0] .= NaN

	fig = CM.Figure(size=(1100, 500), fontsize=11)

	# Left: clinical DICOM + colored mask
	ax1 = CM.Axis(fig[1, 1]; title="Clinical DICOM (120 kVp 150 mA)",
				  aspect=CM.DataAspect(), yreversed=true)
	CM.heatmap!(ax1, clin_hu; colormap=:grays, colorrange=(-200, 500))
	CM.heatmap!(ax1, mask_vis; colormap=:turbo, colorrange=(1, 27), nan_color=:transparent)
	CM.hidedecorations!(ax1); CM.hidespines!(ax1)

	# Right: simulated + same colored mask
	ax2 = CM.Axis(fig[1, 2]; title="Simulated (oriented) — slice $sim_mid",
				  aspect=CM.DataAspect(), yreversed=true)
	CM.heatmap!(ax2, sim_hu; colormap=:grays, colorrange=(-200, 500))
	CM.heatmap!(ax2, mask_vis; colormap=:turbo, colorrange=(1, 27), nan_color=:transparent)
	CM.hidedecorations!(ax2); CM.hidespines!(ax2)

	fig
end

# ╔═╡ 7165348e-58e8-4bd9-b8fd-828021890d50
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
						"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]
	
	n_sims = min(length(sim_measurements), length(base_clinical_idx))
	active_clinical_idx = base_clinical_idx[1:n_sims]
	active_labels = base_scan_labels[1:n_sims]
	colors = CM.cgrad(:tab10, max(n_sims, 2), categorical=true)

	fig = CM.Figure(size=(700, 650), fontsize=11)
	ax = CM.Axis(fig[1, 1]; title="HU Accuracy: Clinical vs Simulated (all rods)",
				 xlabel="Clinical HU", ylabel="Simulated HU", aspect=CM.DataAspect())

	CM.lines!(ax, [-200, 1400], [-200, 1400]; color=:gray60, linestyle=:dash, linewidth=1)

	for (k, (ci, sm)) in enumerate(zip(active_clinical_idx, sim_measurements[1:n_sims]))
		cm = se_measurements[ci]
		CM.scatter!(ax, cm.rod_means, sm.rod_means;
					color=colors[k], markersize=8, label=active_labels[k])
	end
	CM.axislegend(ax; position=:lt)

	all_clin = vcat([se_measurements[ci].rod_means for ci in active_clinical_idx]...)
	all_sim = vcat([sm.rod_means for sm in sim_measurements[1:n_sims]]...)
	
	if length(all_clin) > 1 && length(all_clin) == length(all_sim)
		r = cor(all_clin, all_sim)
		CM.text!(ax, 800, -100; text="r = $(round(r, digits=4))", fontsize=12)
	end

	fig
end

# ╔═╡ 502a0f41-e6cd-4e2e-b2f9-6196d87f91f8
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
						"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]
	
	# Target index 2 (120/150) if available, otherwise use 1
	target_idx = min(2, length(sim_measurements))
	
	cm = se_measurements[base_clinical_idx[target_idx]]
	sm = sim_measurements[target_idx]
	n = length(cm.rod_names)
	label = base_scan_labels[target_idx]

	fig = CM.Figure(size=(1100, 500), fontsize=10)
	ax = CM.Axis(fig[1, 1]; title="Rod HU — $label: Clinical vs Simulated",
				 ylabel="Mean HU", xticks=(1:n, cm.rod_names), xticklabelrotation=π/4)

	CM.barplot!(ax, collect(1:n) .- 0.2, cm.rod_means; width=0.35, color=:steelblue, label="Clinical")
	CM.errorbars!(ax, collect(1:n) .- 0.2, cm.rod_means, cm.rod_stds; color=:black, whiskerwidth=3)
	CM.barplot!(ax, collect(1:n) .+ 0.2, sm.rod_means; width=0.35, color=:darkorange, label="Simulated")
	CM.errorbars!(ax, collect(1:n) .+ 0.2, sm.rod_means, sm.rod_stds; color=:black, whiskerwidth=3)
	CM.axislegend(ax; position=:lt)

	fig
end

# ╔═╡ 8e7279b8-c4e4-4f1a-bf9a-c4c5140ab2c3
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120/50", "120/150", "120/300", "80/480", "100/250", "140/110"]
	
	n_sims = min(length(sim_measurements), length(base_clinical_idx))
	active_labels = base_scan_labels[1:n_sims]
	
	water_idx = 1  # "Water (O)" is first rod
	clin_σ = [se_measurements[base_clinical_idx[i]].rod_stds[water_idx] for i in 1:n_sims]
	sim_σ = [sim_measurements[i].rod_stds[water_idx] for i in 1:n_sims]

	fig = CM.Figure(size=(800, 400), fontsize=11)
	ax = CM.Axis(fig[1, 1]; title="Water ROI Noise (σ): Clinical vs Simulated",
				 ylabel="σ (HU)", xticks=(1:n_sims, active_labels), xlabel="kVp / mA")
	CM.barplot!(ax, collect(1:n_sims) .- 0.2, clin_σ; width=0.35, color=:steelblue, label="Clinical")
	CM.barplot!(ax, collect(1:n_sims) .+ 0.2, sim_σ; width=0.35, color=:darkorange, label="Simulated")
	CM.axislegend(ax; position=:rt)

	fig
end

# ╔═╡ fbe54d81-8722-4660-ac83-24d760b11502
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
						"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]
	
	n_sims = min(length(sim_measurements), length(base_clinical_idx))
	colors = CM.cgrad(:tab10, max(n_sims, 2), categorical=true)

	fig = CM.Figure(size=(1200, 500), fontsize=11)
	ax1 = CM.Axis(fig[1, 1]; title="NPS — Clinical FBP",
				  xlabel="Spatial frequency (lp/cm)", ylabel="NPS (HU²·cm²)", yscale=log10)
	ax2 = CM.Axis(fig[1, 2]; title="NPS — Simulated FBP",
				  xlabel="Spatial frequency (lp/cm)", ylabel="NPS (HU²·cm²)", yscale=log10)

	for i in 1:n_sims
		cm = se_measurements[base_clinical_idx[i]]
		sm = sim_measurements[i]
		label = base_scan_labels[i]

		f_c, v_c = cm.nps.frequencies, cm.nps.nps_1d
		good_c = v_c .> 0
		CM.lines!(ax1, f_c[good_c], v_c[good_c]; color=colors[i], label=label)

		f_s, v_s = sm.nps.frequencies, sm.nps.nps_1d
		good_s = v_s .> 0
		CM.lines!(ax2, f_s[good_s], v_s[good_s]; color=colors[i], label=label)
	end
	
	if n_sims > 0
		CM.axislegend(ax1; position=:rt, labelsize=8)
		CM.axislegend(ax2; position=:rt, labelsize=8)
	end
	CM.linkaxes!(ax1, ax2)

	fig
end

# ╔═╡ 451ac7f3-7f12-4218-9990-527393c6db9b
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
						"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]
	
	n_sims = min(length(sim_measurements), length(base_clinical_idx))
	colors = CM.cgrad(:tab10, max(n_sims, 2), categorical=true)

	fig = CM.Figure(size=(800, 500), fontsize=11)
	ax = CM.Axis(fig[1, 1]; title="MTF — Clinical (solid) vs Simulated (dashed)",
				 xlabel="Spatial frequency (lp/cm)", ylabel="MTF",
				 limits=(nothing, nothing, 0, 1.05))
	CM.hlines!(ax, [0.5, 0.1]; color=:gray80, linestyle=:dash, linewidth=0.8)

	for i in 1:n_sims
		cm = se_measurements[base_clinical_idx[i]]
		sm = sim_measurements[i]
		label = base_scan_labels[i]

		CM.lines!(ax, cm.mtf.frequencies, cm.mtf.mtf;
				  color=colors[i], linewidth=1.5, label=label)
		CM.lines!(ax, sm.mtf.frequencies, sm.mtf.mtf;
				  color=colors[i], linewidth=1.5, linestyle=:dash)
	end
	
	if n_sims > 0
		CM.axislegend(ax; position=:rt, labelsize=8)
	end

	fig
end

# ╔═╡ 24487a3b-30f9-4033-8f82-ce7d80511d1f
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
						"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]
	
	target_idx = min(2, length(sim_measurements))
	cm = se_measurements[base_clinical_idx[target_idx]]
	sm = sim_measurements[target_idx]
	label = base_scan_labels[target_idx]
	
	Δ = sm.rod_means .- cm.rod_means

	fig = CM.Figure(size=(1100, 400), fontsize=10)
	ax = CM.Axis(fig[1, 1]; title="HU Error (Simulated − Clinical) — $label",
				 ylabel="ΔHU", xticks=(1:length(cm.rod_names), cm.rod_names),
				 xticklabelrotation=π/4)
	CM.barplot!(ax, 1:length(Δ), Δ; color=ifelse.(Δ .> 0, :salmon, :steelblue))
	CM.hlines!(ax, [0]; color=:black, linewidth=0.8)

	for (i, d) in enumerate(Δ)
		CM.text!(ax, i, d + sign(d) * 3;
				 text="$(round(d, digits=1))", fontsize=8, align=(:center, d > 0 ? :bottom : :top))
	end

	fig
end

# ╔═╡ Cell order:
# ╟─05487717-8249-4d89-bf94-f7e7dc379bfb
# ╠═db7ada07-c245-4427-a9d8-1fa43e74e884
# ╠═2872d85c-3d09-414f-a194-4b52764fca93
# ╠═eff433a8-e36a-4b6f-bfb5-73664ceac2f3
# ╠═1caeb654-e68e-47d8-a10d-c4020c676c7b
# ╠═9cd57146-a7ec-4fe8-a313-4d46659d8795
# ╠═6cf5c19a-88f2-4ba7-91d6-f68b4fb816e1
# ╠═fb762acd-fe7f-4e9a-adb6-ad77ba81f2e1
# ╠═9d514655-cb7c-4852-8e8e-d2a98527fe0a
# ╠═1ce13813-fa59-43a1-9e0e-99741eeace58
# ╠═02bf2bbb-e785-4e5f-8476-58346735b7fc
# ╠═9636e23b-9d46-41be-988c-843c59de3cf6
# ╠═993c7d4b-ae9d-4234-ad8d-994a904bd83c
# ╠═0e1cad14-1b0a-421e-ab85-19e5601d63d6
# ╠═a1b2c3d4-0001-4000-8000-000000000001
# ╠═a1b2c3d4-0002-4000-8000-000000000002
# ╠═9dd493db-806d-474c-8352-58011ab725e7
# ╠═64b81aff-4ff0-4cc7-bcaa-a16162524620
# ╟─6e93590f-e900-4c9e-91cb-a03db349b185
# ╠═f466466a-c364-4841-8a38-987cbc5777c9
# ╠═c19991fb-c50a-4e56-a3f6-bba1e41fd9c9
# ╠═119755f9-4c74-4d15-89a7-793295d21441
# ╟─bdc45741-e374-4d78-ba6b-43b5ef411f2a
# ╠═41f11735-2a1c-4d36-8a0d-3c57de9cbae0
# ╟─c0dfa3a5-88f2-47b3-94ab-0393132b0dc1
# ╠═fec5488f-09e4-41af-b3e3-0fef62cd2bfa
# ╟─042d2f32-0a93-48c9-acd1-2701263f8be9
# ╠═90fe8208-f128-4b59-bb14-cb3f6829d9da
# ╟─e4d91faa-e427-48d3-9f86-fe07a33d8bff
# ╠═c95c638e-60a7-4242-8c95-ee352ac96a02
# ╟─5d85356a-9524-4041-9c7e-5df155180f8f
# ╠═c2b384e5-0ebd-4e32-bbcc-d7699c5bb9c8
# ╟─79d134f6-1a23-4c7d-af2a-4c70a57841db
# ╟─262d5985-d5fa-49f1-968d-f760ac4c6fc7
# ╠═dca3c26e-3d13-407f-aa83-f8a9ce5f9d23
# ╠═63b0c15a-8e01-4f8b-9ff7-a9109e2b5933
# ╟─451bfa43-42d9-4023-afb0-52bbf4b5a487
# ╟─04a99179-ae4f-40ba-b4bf-aa5f5276bbc1
# ╟─3c0d7acf-6de3-4662-90b9-05cebcfffd1c
# ╠═c7df06b6-014e-4115-a811-cb5a190cbb2a
# ╟─b5a1b6ce-a719-4444-8f70-76aff9040515
# ╟─ead991fd-d38c-4ebe-81f8-c35c84d0df52
# ╠═88990577-029e-4f3b-be94-3ac413e526fe
# ╟─ce7b3519-78ee-43e0-a8a9-5b0270a68901
# ╟─e26846b8-719f-4d24-a531-7aac1fe6672a
# ╟─a1b2c3d4-0010-4000-8000-000000000010
# ╠═a1b2c3d4-0011-4000-8000-000000000011
# ╟─a1b2c3d4-0012-4000-8000-000000000012
# ╠═a1b2c3d4-0013-4000-8000-000000000013
# ╟─a1b2c3d4-0014-4000-8000-000000000014
# ╠═a1b2c3d4-0015-4000-8000-000000000015
# ╟─a1b2c3d4-0016-4000-8000-000000000016
# ╟─a1b2c3d4-0017-4000-8000-000000000017
# ╟─a1b2c3d4-0018-4000-8000-000000000018
# ╟─a1b2c3d4-0019-4000-8000-000000000019
# ╟─a1b2c3d4-0020-4000-8000-000000000020
# ╠═a1b2c3d4-0021-4000-8000-000000000021
# ╟─a1b2c3d4-0022-4000-8000-000000000022
# ╟─a1b2c3d4-0023-4000-8000-000000000023
# ╟─d9ad5eb6-df82-466b-a291-310417adc0bf
# ╠═830ee323-3fd3-464d-a4bf-20348c206909
# ╟─0a8833cf-06c6-4777-9fe8-c17f7138e6b8
# ╟─a1b2c3d4-5001-4000-8000-000000000001
# ╠═a1b2c3d4-5002-4000-8000-000000000002
# ╠═a1b2c3d4-5003-4000-8000-000000000003
# ╠═a1b2c3d4-5004-4000-8000-000000000004
# ╠═a1b2c3d4-5005-4000-8000-000000000005
# ╠═a1b2c3d4-5006-4000-8000-000000000006
# ╠═a1b2c3d4-5007-4000-8000-000000000007
# ╠═a1b2c3d4-5008-4000-8000-000000000008
# ╠═a1b2c3d4-5009-4000-8000-000000000009
# ╠═a1b2c3d4-5010-4000-8000-000000000010
# ╠═a1b2c3d4-5011-4000-8000-000000000011
# ╠═a1b2c3d4-5012-4000-8000-000000000012
# ╠═a1b2c3d4-5013-4000-8000-000000000013
# ╟─a1b2c3d4-5014-4000-8000-000000000014
# ╟─e54d80d3-6f67-4cd4-a85f-0e150755bf6e
# ╠═fc69166c-0bec-441c-a2e5-e40eeaac1831
# ╠═c1349a54-553e-45ec-b37d-631629baa4e3
# ╟─07d10464-2c39-4a4a-aeb0-5549c34b705d
# ╟─7165348e-58e8-4bd9-b8fd-828021890d50
# ╟─502a0f41-e6cd-4e2e-b2f9-6196d87f91f8
# ╟─8e7279b8-c4e4-4f1a-bf9a-c4c5140ab2c3
# ╟─fbe54d81-8722-4660-ac83-24d760b11502
# ╟─451ac7f3-7f12-4218-9990-527393c6db9b
# ╟─24487a3b-30f9-4033-8f82-ce7d80511d1f
