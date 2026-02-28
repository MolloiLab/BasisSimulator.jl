### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ db7ada07-c245-4427-a9d8-1fa43e74e884
begin
	using Pkg: Pkg
	Pkg.activate(dirname(@__DIR__))
	Pkg.instantiate()
	# Pkg.update()

	using Revise
end

# ╔═╡ a0839e3b-bd86-44e6-a282-046dcc6a6d39
using FileIO

# ╔═╡ 1caeb654-e68e-47d8-a10d-c4020c676c7b
using ImageMagick

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

# ╔═╡ a1b2c3d4-0001-4000-8000-000000000001
using JLD2: JLD2

# ╔═╡ a1b2c3d4-0002-4000-8000-000000000002
using DelimitedFiles: DelimitedFiles

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

# ╔═╡ 9cd57146-a7ec-4fe8-a313-4d46659d8795
import CairoMakie as CM

# ╔═╡ 6cf5c19a-88f2-4ba7-91d6-f68b4fb816e1
import Statistics: mean, std, cor

# ╔═╡ ff611999-d9e4-45e1-80e5-a46176c4845a
import Statistics: median

# ╔═╡ fb762acd-fe7f-4e9a-adb6-ad77ba81f2e1
import XrayAttenuation as XA

# ╔═╡ 0e1cad14-1b0a-421e-ab85-19e5601d63d6
import DICOM as DCM

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
rootdir = "/Users/daleblack/Desktop/SCANS"

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
		raw_u16 = round.(UInt16, Float32.(img) .* 65535.0f0)
		i16 = Int16.(Int32.(raw_u16) .- Int32(32768))
		Float32.(i16) .* slope .+ intercept
	end
	return cat(slices...; dims = 3)
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

	fig = CM.Figure(size = (1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_120_low_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_120_low_ir[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text = "120 kVp · 50 mA · CTDIvol 3.38 mGy", fontsize = 16, font = :bold)
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

	fig = CM.Figure(size = (1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_120_mid_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_120_mid_ir[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text = "120 kVp · 150 mA · CTDIvol 10.16 mGy", fontsize = 16, font = :bold)
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

	fig = CM.Figure(size = (1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_120_high_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_120_high_ir[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text = "120 kVp · 300 mA · CTDIvol 20.38 mGy", fontsize = 16, font = :bold)

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

	fig = CM.Figure(size = (1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_80_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_80_ir[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text = "80 kVp · 480 mA · CTDIvol 10.32 mGy", fontsize = 16, font = :bold)

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

	fig = CM.Figure(size = (1500, 1000))


	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_100_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_100_ir[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text = "100 kVp · 250 mA · CTDIvol 10.53 mGy", fontsize = 16, font = :bold)

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

	fig = CM.Figure(size = (1500, 1000))

	for (col, (clim, wname)) in enumerate(windows)
		ax = CM.Axis(fig[1, col]; title = "FBP (ASiR-V 0%) — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_140_fbp[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)

		ax = CM.Axis(fig[2, col]; title = "ASiR-V 50% — $wname", titlesize = 12,
			aspect = CM.DataAspect(), yreversed = true)
		CM.heatmap!(ax, hu_140_ir[:, :, mid]; colormap = :grays, colorrange = clim)
		CM.hidedecorations!(ax)
		CM.hidespines!(ax)
	end

	CM.Label(fig[0, :]; text = "140 kVp · 110 mA · CTDIvol 10.85 mGy", fontsize = 16, font = :bold)

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
	sample_angles = range(0, 2π - 2π / n_sample, length = n_sample)

	profile = zeros(Float64, n_sample)
	for (k, θ) in enumerate(sample_angles)
		s, c = 0.0, 0
		for dr in -3:3
			xi = round(Int, cx + (r_outer_pix + dr) * cos(θ))
			yi = round(Int, cy + (r_outer_pix + dr) * sin(θ))
			if 1 <= xi <= nx && 1 <= yi <= ny
				s += hu_slice[xi, yi]
				c += 1
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
			s += profile[mod1(k + d, n_sample)]
			c += 1
		end
		smoothed[k] = s / c
	end

	# Ca400 is the highest HU rod on the outer ring
	# Sub-sample parabolic interpolation for sub-0.1° precision
	k_max = argmax(smoothed)
	k_prev = mod1(k_max - 1, n_sample)
	k_next = mod1(k_max + 1, n_sample)
	y_m, y_0, y_p = smoothed[k_prev], smoothed[k_max], smoothed[k_next]
	denom = y_m - 2y_0 + y_p
	δ_sub = abs(denom) > 1e-12 ? 0.5 * (y_m - y_p) / denom : 0.0
	δ_sub = clamp(δ_sub, -0.5, 0.5)
	θ_ca400 = sample_angles[k_max] + δ_sub * (2π / n_sample)
	rotation = θ_ca400

	# Step 3: Place ROIs at all 16 rod positions
	# Clinical DICOM has CCW angular ordering (+ direction) due to image orientation
	outer_start = θ_ca400 - 3 * π / 4  # Ca100 is 3 steps before Ca400
	outer_angles = [outer_start + (i - 1) * π / 4 for i in 1:8]
	outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]
	outer_names = ["Ca 100", "Ca 200", "Ca 300", "Ca 400",
		"Water (O)", "SW ref 1", "SW ref 2", "Ca 50"]

	inner_start = outer_start - π / 8  # I2.5 at gap center (π/8 before Ca100)
	inner_angles = [inner_start + (i - 1) * π / 4 for i in 1:8]
	inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]
	inner_names = ["I 2.5", "I 5.0", "I 7.5", "I 10.0",
		"I 15.0", "I 20.0", "Water (I)", "I 2.0"]

	roi_r_pix = rod_radius_cm * roi_fraction / pixel_cm
	roi_r² = roi_r_pix^2
	search_r_pix = rod_radius_cm / pixel_cm * 1.5  # search window slightly larger than rod

	mask = zeros(UInt8, nx, ny)
	rod_info = []

	for (angles, labels, names, ring_cm, ring_sym) in [
		(outer_angles, outer_labels, outer_names, outer_ring_cm, :outer),
		(inner_angles, inner_labels, inner_names, inner_ring_cm, :inner),
	]
		ring_pix = ring_cm / pixel_cm
		for (θ, lbl, name) in zip(angles, labels, names)
			# Expected center from hardcoded angle
			ecx = cx + ring_pix * cos(θ)
			ecy = cy + ring_pix * sin(θ)

			# Refine: find actual rod center via local centroid of distinctive pixels
			si_lo = max(1, floor(Int, ecx - search_r_pix))
			si_hi = min(nx, ceil(Int, ecx + search_r_pix))
			sj_lo = max(1, floor(Int, ecy - search_r_pix))
			sj_hi = min(ny, ceil(Int, ecy + search_r_pix))

			# Collect HU in search window
			local_vals = Float64[]
			for sj in sj_lo:sj_hi, si in si_lo:si_hi
				if (si - ecx)^2 + (sj - ecy)^2 <= search_r_pix^2
					push!(local_vals, Float64(hu_slice[si, sj]))
				end
			end

			# Background is roughly the body mean (~0 HU for water); rod is distinct
			# Use pixels that deviate from the median by > half the range
			if length(local_vals) > 10
				med = median(local_vals)
				dev = maximum(abs.(local_vals .- med))
				thresh = dev * 0.3  # pixels that are clearly part of the rod
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

			# Place ROI at refined center
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
				angle_deg = round(rad2deg(θ), digits = 1),
				mean_hu = isempty(vals) ? NaN : mean(vals),
				std_hu = length(vals) > 1 ? std(vals) : NaN,
				n_pixels = length(vals),
			))
		end
	end

	return mask, rod_info, (cx = cx, cy = cy, rotation_deg = round(rad2deg(rotation), digits = 2))
end

# ╔═╡ 63b0c15a-8e01-4f8b-9ff7-a9109e2b5933
seg_result = let
	mid_z = size(hu_120_mid_ir, 3) ÷ 2
	hu_slice = hu_120_mid_ir[:, :, mid_z]
	mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm = 35.0)
	(mask = mask, rods = rod_info, center = center, slice_idx = mid_z)
end;

# ╔═╡ 451bfa43-42d9-4023-afb0-52bbf4b5a487
let
	hu = hu_120_mid_ir[:, :, seg_result.slice_idx]
	rods = seg_result.rods
	pixel_cm = 35.0 / size(hu, 1)
	roi_r_pix = 1.4 * 0.7 / pixel_cm

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	# Left: CT with ROI circles
	ax1 = CM.Axis(fig[1, 1]; title = "120 kVp ASiR-V 50% — Segmented ROIs (slice $(seg_result.slice_idx))",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, hu; colormap = :grays, colorrange = (-200, 500))

	θ_circle = range(0, 2π, length = 61)
	for r in rods
		xs = r.cx .+ roi_r_pix .* cos.(θ_circle)
		ys = r.cy .+ roi_r_pix .* sin.(θ_circle)
		c = r.ring == :outer ? :orange : :lime
		CM.lines!(ax1, xs, ys; color = c, linewidth = 1.5)
		CM.text!(ax1, r.cx, r.cy + roi_r_pix + 4;
			text = r.name, fontsize = 7, align = (:center, :bottom), color = c)
	end
	CM.scatter!(ax1, [seg_result.center.cx], [seg_result.center.cy];
		color = :red, marker = :cross, markersize = 12)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	# Right: labeled mask overlay
	ax2 = CM.Axis(fig[1, 2]; title = "Rod Label Mask", aspect = CM.DataAspect(), yreversed = true)
	mask_vis = Float32.(seg_result.mask)
	mask_vis[mask_vis.==0] .= NaN
	CM.heatmap!(ax2, hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax2, mask_vis; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

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

	fig = CM.Figure(size = (1000, 500), fontsize = 11)
	ax = CM.Axis(fig[1, 1];
		title = "ROI Measurements — 120 kVp · 150 mA · ASiR-V 50% [rotation=$(seg_result.center.rotation_deg)°]",
		ylabel = "Mean HU ± σ",
		xticks = (1:n, names),
		xticklabelrotation = π / 4)
	CM.barplot!(ax, 1:n, means; color = colors)
	CM.errorbars!(ax, 1:n, means, stds; color = :black, whiskerwidth = 4)

	for (i, (m, s)) in enumerate(zip(means, stds))
		CM.text!(ax, i, m + s + 5;
			text = "$(round(Int, m))", fontsize = 8, align = (:center, :bottom))
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

	fig = CM.Figure(size = (1500, 2000))

	for (row, (vol, keV_label)) in enumerate(zip(vols, labels))
		for (col, (clim, wname)) in enumerate(windows)
			ax = CM.Axis(fig[row, col]; title = "$keV_label — $wname", titlesize = 12,
				aspect = CM.DataAspect(), yreversed = true)
			CM.heatmap!(ax, vol[:, :, mid]; colormap = :grays, colorrange = clim)
			CM.hidedecorations!(ax)
			CM.hidespines!(ax)
		end
	end

	CM.Label(fig[0, :]; text = "DE 80/140 kVp — VMI (CTDIvol 10.07 mGy)", fontsize = 16, font = :bold)
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
	mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm = 35.0)
	(mask = mask, rods = rod_info, center = center, slice_idx = mid_z)
end;

# ╔═╡ ce7b3519-78ee-43e0-a8a9-5b0270a68901
let
	hu = hu_de_100keV[:, :, seg_result_de.slice_idx]
	rods = seg_result_de.rods
	pixel_cm = 35.0 / size(hu, 1)
	roi_r_pix = 1.4 * 0.7 / pixel_cm

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "DE 100 keV VMI — Segmented ROIs (slice $(seg_result_de.slice_idx))",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, hu; colormap = :grays, colorrange = (-200, 500))

	θ_circle = range(0, 2π, length = 61)
	for r in rods
		xs = r.cx .+ roi_r_pix .* cos.(θ_circle)
		ys = r.cy .+ roi_r_pix .* sin.(θ_circle)
		c = r.ring == :outer ? :orange : :lime
		CM.lines!(ax1, xs, ys; color = c, linewidth = 1.5)
		CM.text!(ax1, r.cx, r.cy + roi_r_pix + 4;
			text = r.name, fontsize = 7, align = (:center, :bottom), color = c)
	end
	CM.scatter!(ax1, [seg_result_de.center.cx], [seg_result_de.center.cy];
		color = :red, marker = :cross, markersize = 12)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Rod Label Mask", aspect = CM.DataAspect(), yreversed = true)
	mask_vis = Float32.(seg_result_de.mask)
	mask_vis[mask_vis.==0] .= NaN
	CM.heatmap!(ax2, hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax2, mask_vis; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

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

	fig = CM.Figure(size = (1000, 500), fontsize = 11)
	ax = CM.Axis(fig[1, 1];
		title = "ROI Measurements — DE 100 keV VMI [rotation=$(seg_result_de.center.rotation_deg)°]",
		ylabel = "Mean HU ± σ",
		xticks = (1:n, names),
		xticklabelrotation = π / 4)
	CM.barplot!(ax, 1:n, means; color = colors)
	CM.errorbars!(ax, 1:n, means, stds; color = :black, whiskerwidth = 4)

	for (i, (m, s)) in enumerate(zip(means, stds))
		CM.text!(ax, i, m + s + 5;
			text = "$(round(Int, m))", fontsize = 8, align = (:center, :bottom))
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

# ╔═╡ 0c47ac3f-6d83-47ac-8c94-bc9172db716d
"""
	measure_nps_local(hu_slice, pixel_mm; roi_center, roi_radius_mm, roi_size=64,
	                  n_rois=64, overlap=0.5)

Local NPS with quadratic detrending + Hann window (no package restart needed).
Returns a NamedTuple matching BS.NPSResult fields used downstream.
"""
function measure_nps_local(
	hu_slice::AbstractMatrix, pixel_mm::Real;
	roi_center::Tuple{Int,Int},
	roi_radius_mm::Real,
	roi_size::Int = 64,
	n_rois::Int = 64,
	overlap::Float64 = 0.5,
	unit::Symbol = :lp_cm,
)
	img = Array(Float64.(hu_slice))
	ny, nx = size(img)

	# Crop to circular ROI bounding box
	cy, cx = roi_center
	roi_radius_px = roi_radius_mm / pixel_mm
	rows_in = [i for i in 1:ny if any(j -> sqrt((i-cy)^2 + (j-cx)^2) <= roi_radius_px, 1:nx)]
	cols_in = [j for j in 1:nx if any(i -> sqrt((i-cy)^2 + (j-cx)^2) <= roi_radius_px, 1:ny)]
	img_roi = img[minimum(rows_in):maximum(rows_in), minimum(cols_in):maximum(cols_in)]
	ny_roi, nx_roi = size(img_roi)

	# Extract ROI patches
	step = max(round(Int, roi_size * (1 - overlap)), 1)
	n_y = (ny_roi - roi_size) ÷ step + 1
	n_x = (nx_roi - roi_size) ÷ step + 1
	rois = Matrix{Float64}[]
	for iy in 0:(n_y-1), ix in 0:(n_x-1)
		r0 = 1 + iy * step; c0 = 1 + ix * step
		if r0 + roi_size - 1 <= ny_roi && c0 + roi_size - 1 <= nx_roi
			push!(rois, copy(img_roi[r0:r0+roi_size-1, c0:c0+roi_size-1]))
		end
	end
	if length(rois) > n_rois
		rois = rois[randperm(length(rois))[1:n_rois]]
	end
	actual_n = length(rois)

	# Hann window
	w1d = [0.5 * (1 - cos(2π * i / (roi_size - 1))) for i in 0:(roi_size-1)]
	win = w1d * w1d'

	# Compute NPS
	power_sum = zeros(Float64, roi_size, roi_size)
	for roi in rois
		# Quadratic detrend: fit ax²+by²+cxy+dx+ey+f, subtract
		x = Float64.(repeat(1:roi_size, 1, roi_size)')
		y = Float64.(repeat(1:roi_size, 1, roi_size))
		xf = vec(x); yf = vec(y); zf = vec(roi)
		A = [xf.^2 yf.^2 xf.*yf xf yf ones(length(xf))]
		c = A \ zf
		trend = c[1].*x.^2 .+ c[2].*y.^2 .+ c[3].*x.*y .+ c[4].*x .+ c[5].*y .+ c[6]
		detrended = roi .- trend

		windowed = detrended .* win
		power_sum .+= abs2.(fft(windowed))
	end

	nps_2d = fftshift(power_sum ./ actual_n .* (pixel_mm^2 / roi_size^2))

	# Radial average
	freq_axis = fftshift(BS.fftfreq(roi_size, 1.0 / pixel_mm))
	df = length(freq_axis) > 1 ? abs(freq_axis[2] - freq_axis[1]) : 1.0
	n_bins = length(freq_axis) ÷ 2 + 1
	radial_freqs = collect(range(0, stop=(n_bins-1)*df, length=n_bins))
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

	# Skip DC
	radial_freqs = radial_freqs[2:end]
	nps_1d = nps_1d[2:end]

	# Unit conversion
	if unit == :lp_cm
		radial_freqs .*= 10.0
		nps_1d ./= 100.0
	end

	# Peak
	peak_idx = length(nps_1d) > 1 ? argmax(nps_1d[2:end]) + 1 : 1
	peak_freq = radial_freqs[peak_idx]
	peak_val = nps_1d[peak_idx]
	df_out = length(radial_freqs) > 1 ? radial_freqs[2] - radial_freqs[1] : 1.0
	integrated = sum(nps_1d) * df_out

	return (
		frequencies = radial_freqs, nps_1d = nps_1d,
		peak_frequency = peak_freq, peak_value = peak_val,
		integrated_nps = integrated, n_rois = actual_n,
		roi_size = (roi_size, roi_size), unit = unit,
	)
end

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
		n_angles = 720,
		oversample = 4,
		margin_inner = 15.0,   # pixels inside edge (toward body center) — generous
		margin_outer = 5.0,    # pixels outside edge (toward air/FOV) — limited by clearance
		fov_guard_pix = 3.0,   # skip angles where outer samples get this close to FOV edge
	)
		nx, ny = size(hu_slice)
		pixel_cm = fov_cm / nx
		pixel_mm = pixel_cm * 10.0
		edge_r_pix = radius_cm / pixel_cm
		fov_r_pix = nx / 2.0  # FOV radius in pixels (image center to edge)

		# Image center (FOV center)
		img_cx = (nx + 1) / 2.0
		img_cy = (ny + 1) / 2.0

		# Sample radial ESF: asymmetric margins around the edge
		r_min = edge_r_pix - margin_inner
		r_max = edge_r_pix + margin_outer
		n_r = round(Int, (r_max - r_min) * oversample)
		radii = range(r_min, r_max, length = n_r)
		positions_mm = collect((radii .- edge_r_pix) .* pixel_mm)  # mm from edge

		# Average ESF across angles, skipping those too close to FOV boundary
		esf = zeros(Float64, n_r)
		counts = zeros(Int, n_r)
		angles = range(0, 2π - 2π / n_angles, length = n_angles)
		n_used = 0

		for θ in angles
			cosθ, sinθ = cos(θ), sin(θ)

			# Check: does the outermost sample point stay within FOV?
			x_outer = cx + r_max * cosθ
			y_outer = cy + r_max * sinθ
			dist_from_img_center = sqrt((x_outer - img_cx)^2 + (y_outer - img_cy)^2)
			if dist_from_img_center > fov_r_pix - fov_guard_pix
				continue  # skip this angle — too close to FOV edge
			end

			n_used += 1
			for (k, r) in enumerate(radii)
				xi = cx + r * cosθ
				yi = cy + r * sinθ
				# Bilinear interpolation
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

		return (frequencies = freq_lp_cm, mtf = mtf_curve, mtf50 = f50, mtf10 = f10,
				n_angles_used = n_used)
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

		# --- NPS (uniform body center, inside inner rod ring at 5.5 cm) ---
		center_row = round(Int, seg_center.cx)
		center_col = round(Int, seg_center.cy)
		# nps_result = BS.measure_nps(
		# 	hu_slice, pixel_mm;
		# 	config = BS.NPSConfig(roi_size = 64, n_rois = 64, overlap = 0.5, detrend = :linear, window = :hann), 
		# 	roi_center = (center_row, center_col),
		# 	roi_radius_mm = 45.0,
		# 	unit = :lp_cm,
		# )

		nps_result = measure_nps_local(
			hu_slice, pixel_mm;
			roi_center = (center_row, center_col),
			roi_radius_mm = 64.0,
			unit = :lp_cm,
			overlap = 0.25,
		)

		# --- MTF (phantom outer body / air boundary) ---
		mtf_result = measure_mtf_circular_edge(
			hu_slice, Float64(seg_center.cx), Float64(seg_center.cy), 16.5;
			fov_cm = fov_cm,
			margin_inner = 15.0,   # plenty of room inside body
			margin_outer = 5.0,    # ~5 pixels into air (limited by FOV clearance)
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

# ╔═╡ e6322fc0-9a18-43a6-a98d-5af80c2a5f01
let
	all_m = vcat(se_measurements, de_measurements)

	fig = CM.Figure(size = (1200, 500), fontsize = 11)
	ax1 = CM.Axis(fig[1, 1]; title = "NPS — Single-Energy Scans",
		xlabel = "Spatial frequency (lp/cm)", ylabel = "NPS (HU²·cm²)",
		yscale = log10)
	ax2 = CM.Axis(fig[1, 2]; title = "NPS — Dual-Energy VMI",
		xlabel = "Spatial frequency (lp/cm)", ylabel = "NPS (HU²·cm²)",
		yscale = log10)

	se_colors = CM.cgrad(:tab10, 12, categorical = true)
	for (i, m) in enumerate(se_measurements)
		freqs = m.nps.frequencies
		vals = m.nps.nps_1d
		good = vals .> 0
		CM.lines!(ax1, freqs[good], vals[good]; label = m.name, color = se_colors[i])
	end
	CM.axislegend(ax1; position = :rt, labelsize = 8, nbanks = 2)

	de_colors = [:purple, :teal, :darkorange, :crimson]
	for (i, m) in enumerate(de_measurements)
		freqs = m.nps.frequencies
		vals = m.nps.nps_1d
		good = vals .> 0
		CM.lines!(ax2, freqs[good], vals[good]; label = m.name, color = de_colors[i], linewidth = 2)
	end
	CM.axislegend(ax2; position = :rt, labelsize = 10)

	fig
end

# ╔═╡ a1b2c3d4-0016-4000-8000-000000000016
md"""
### 3c. NPS Curves — All Scans
"""

# ╔═╡ a1b2c3d4-0017-4000-8000-000000000017
let
	all_m = vcat(se_measurements, de_measurements)

	fig = CM.Figure(size = (1200, 500), fontsize = 11)
	ax1 = CM.Axis(fig[1, 1]; title = "NPS — Single-Energy Scans",
		xlabel = "Spatial frequency (lp/cm)", ylabel = "NPS (HU²·cm²)",
		yscale = log10)
	ax2 = CM.Axis(fig[1, 2]; title = "NPS — Dual-Energy VMI",
		xlabel = "Spatial frequency (lp/cm)", ylabel = "NPS (HU²·cm²)",
		yscale = log10)

	se_colors = CM.cgrad(:tab10, 12, categorical = true)
	for (i, m) in enumerate(se_measurements)
		freqs = m.nps.frequencies
		vals = m.nps.nps_1d
		good = vals .> 0
		CM.lines!(ax1, freqs[good], vals[good]; label = m.name, color = se_colors[i])
	end
	CM.axislegend(ax1; position = :rt, labelsize = 8, nbanks = 2)

	de_colors = [:purple, :teal, :darkorange, :crimson]
	for (i, m) in enumerate(de_measurements)
		freqs = m.nps.frequencies
		vals = m.nps.nps_1d
		good = vals .> 0
		CM.lines!(ax2, freqs[good], vals[good]; label = m.name, color = de_colors[i], linewidth = 2)
	end
	CM.axislegend(ax2; position = :rt, labelsize = 10)

	fig
end

# ╔═╡ a1b2c3d4-0018-4000-8000-000000000018
md"""
### 3d. MTF Curves — All Scans
"""

# ╔═╡ a1b2c3d4-0019-4000-8000-000000000019
let
	fig = CM.Figure(size = (1200, 500), fontsize = 11)
	ax1 = CM.Axis(fig[1, 1]; title = "MTF — Single-Energy Scans",
		xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
		limits = (nothing, nothing, 0, 1.05))
	ax2 = CM.Axis(fig[1, 2]; title = "MTF — Dual-Energy VMI",
		xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
		limits = (nothing, nothing, 0, 1.05))

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
		append!(row, round.(m.rod_means, digits = 2))
		append!(row, round.(m.rod_stds, digits = 2))
		append!(row, round.(m.rod_cnr, digits = 2))
		push!(row, round(m.nps_peak_freq, digits = 3))
		push!(row, round(m.nps_area, digits = 3))
		push!(row, round(m.mtf_f50, digits = 3))
		push!(row, round(m.mtf_f10, digits = 3))
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
			f[m.name] = (freq = m.nps.frequencies, nps = m.nps.nps_1d)
		end
	end

	# --- Save MTF curves to JLD2 ---
	mtf_path = joinpath(outdir, "mtf_curves.jld2")
	JLD2.jldopen(mtf_path, "w") do f
		for m in all_m
			f[m.name] = (freq = m.mtf.frequencies, mtf = m.mtf.mtf)
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

	fig = CM.Figure(size = (1400, 400), fontsize = 10)

	# NPS area (noise variance)
	ax1 = CM.Axis(fig[1, 1]; title = "NPS Area (Noise Variance)",
		ylabel = "NPS integral (HU²·cm²)",
		xticks = (1:n, [m.name for m in all_m]),
		xticklabelrotation = π / 3)
	CM.barplot!(ax1, 1:n, [m.nps_area for m in all_m];
		color = vcat(fill(:steelblue, length(se_measurements)),
			fill(:darkorange, length(de_measurements))))

	# MTF f50 and f10
	ax2 = CM.Axis(fig[1, 2]; title = "MTF f₅₀ and f₁₀",
		ylabel = "Frequency (lp/cm)",
		xticks = (1:n, [m.name for m in all_m]),
		xticklabelrotation = π / 3)
	CM.barplot!(ax2, collect(1:n) .- 0.15, [m.mtf_f50 for m in all_m];
		width = 0.3, color = :teal, label = "f₅₀")
	CM.barplot!(ax2, collect(1:n) .+ 0.15, [m.mtf_f10 for m in all_m];
		width = 0.3, color = :salmon, label = "f₁₀")
	CM.axislegend(ax2; position = :rt)

	fig
end

# ╔═╡ d9ad5eb6-df82-466b-a291-310417adc0bf
md"""
## 4. Matching Gammex 472 Phantom

Digital replica of the physical Gammex 472 multi-energy CT phantom at **0.2 mm isotropic**
resolution (1750 × 1750 × 250 voxels, 35 cm FOV × 5 cm thick). The layout uses **8 rods
per ring** matching the actual phantom. Pure water rods have a **0.3 mm air gap**;
all other inserts (Ca, I, SW ref) have a **0.15 mm air gap** (tighter fit).
The mask is transposed + flipped to match the clinical DICOM display orientation.

**Body:** 330 mm diameter Gammex Model 451 Solid Water (ρ = 1.02 g/cm³)

**Outer ring** (R = 10.5 cm, 8 positions at 45°, gap at 12 o'clock CW):
Ca100 → Ca200 → Ca300 → Ca400 → Water → SW ref → SW ref → Ca50

**Inner ring** (R = 5.0 cm, 8 positions at 45°, starts at 12 o'clock CW):
I2.5 → I5.0 → I7.5 → I10 → I15 → I20 → Water → I2.0

**Rod diameter:** 28 mm | **Air gap:** 0.3 mm (water) / 0.15 mm (others) | **Labels:** 0 = air, 1 = solid water,
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
		air_gap_water = 0.03   # 0.3mm clearance for pure water rods
		air_gap_other = 0.010  # 0.10mm clearance for all other insert rods
		rod_radius² = rod_radius^2
		outer_ring_R = 10.5
		inner_ring_R = 5.5

		outer_start = π / 2 - π / 8
		outer_angles = [outer_start - (i - 1) * π / 4 for i in 1:8]
		outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]

		inner_start = π / 2
		inner_angles = [inner_start - (i - 1) * π / 4 for i in 1:8]
		inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]

		# Per-rod hole radius²: water rods (label 2) get full gap, others get half
		outer_hole_r² = [(rod_radius + (l == UInt8(2) ? air_gap_water : air_gap_other))^2 for l in outer_labels]
		inner_hole_r² = [(rod_radius + (l == UInt8(2) ? air_gap_water : air_gap_other))^2 for l in inner_labels]

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
					if d² <= outer_hole_r²[idx]
						slice[i, j] = d² <= rod_radius² ? outer_labels[idx] : UInt8(0)
						@goto next_voxel
					end
				end

				for idx in 1:8
					d² = (xi - inner_cx[idx])^2 + (yj - inner_cy[idx])^2
					if d² <= inner_hole_r²[idx]
						slice[i, j] = d² <= rod_radius² ? inner_labels[idx] : UInt8(0)
						break
					end
				end
			end
			@label next_voxel
		end

		# Flip in y to match clinical DICOM orientation
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
			materials_vec[Int(lbl)+1] = mat
		end

		origin = (-fov_cm / 2 + dx / 2, -fov_cm / 2 + dy / 2, -z_cm / 2 + dz / 2)
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
		lut[Int(l)+1] = Float32(i)
	end
	mapped = lut[Int.(slice_data).+1]

	colors = [MATERIAL_INFO[l].color for l in unique_labels]
	cmap = CM.cgrad(colors, n_labels, categorical = true)
	names = [MATERIAL_INFO[l].name for l in unique_labels]

	fig = CM.Figure(size = (1000, 850), fontsize = 12)
	ax = CM.Axis(fig[1, 1];
		title = "Custom Gammex 472 — Slice $mid / $nz (0.2mm voxels)",
		aspect = CM.DataAspect())
	hm = CM.heatmap!(ax, mapped; colormap = cmap, colorrange = (0.5, n_labels + 0.5))
	CM.Colorbar(fig[1, 2], hm; ticks = (1:n_labels, names), ticklabelsize = 11, width = 15)
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
begin
	# ─── DAS noise parameters (tune these to match GE clinical scans) ───
	# CatSim defaults: electronic_noise=3500 e⁻, detection_gain=17 e⁻/keV
	# BasisSimulator defaults: electronic_noise=5000 e⁻, detection_gain=15 e⁻/keV
	# At 120kVp, mean_E ≈ 60 keV → σ_e_photon = 5000/(60×15) ≈ 5.6 photons
	# Set electronic_noise=0 to disable (quantum-only, previous behavior)
	# sim_electronic_noise = 1500.0  # e⁻ (std dev of DAS readout noise) — high value for noise floor matching clinical
	sim_electronic_noise = 500.0
	sim_detection_gain = 10.0      # e⁻/keV (scintillator + photodiode gain)

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
		target_angle = 10.0, # GE Quantix 160 tube: 10° anode angle
		flat_filter_material = :aluminum, # Built-in Al inherent filtration (2.5mm)
		flat_filter_thickness = 2.5, # Applied in spectrum domain by resolve_spectrum
		bowtie_filter = :ge_revolution_large, # CatSim 889-point GE Revolution profile
		detector_material = :lumex, # GE Gemstone Ce:(Tb,Lu)₃Al₅O₁₂ → MC LUT efficiency
		detector_depth = 3.0,
		fill_factor_row = 0.9,
		fill_factor_col = 0.9,
		electronic_noise = sim_electronic_noise,
		detection_gain = sim_detection_gain,
	)
end

# ╔═╡ a1b2c3d4-7000-4000-8000-e00000000001
md"""
## Noise Model
- **Quantum noise:** Poisson statistics in intensity domain, scaled by I₀ × η\_eff
- **Electronic noise:** Gaussian additive in intensity domain, σ\_e = electronic\_noise / (mean\_E × detection\_gain)
- **MC LUT (Lumex):** η(E) folded into I₀\_eff — orthogonal to DAS noise

**Tunable parameters** (in scanner cell above):
- `sim_electronic_noise` — DAS readout noise floor (e⁻). CatSim default: 3500, ours: 5000. Set to 0 for quantum-only.
- `sim_detection_gain` — scintillator gain (e⁻/keV). CatSim default: 17, ours: 15. Higher gain → smaller photon-equivalent noise.
"""

# ╔═╡ a1b2c3d4-7001-4000-8000-000000000001
# Extra Al filtration beyond the scanner's built-in 2.5mm.
# GE Revolution Apex has ~7-8mm Al equivalent inherent filtration total.
# The scanner already applies 2.5mm in the spectrum domain (via resolve_spectrum),
# so we add 4.5mm more here to reach ~7mm total.
additional_filters = [("Al", 4.5)]

# ╔═╡ a1b2c3d4-5003-4000-8000-000000000003
begin
	sim_rotation_time = 1.0     # seconds (clinical 1.0s rotation)
	sim_collimation_mm = 5.0    # 8 × 0.625mm — fast single-slice dev mode
	sim_n_views = 984           # standard GE Revolution

	SE_SIM_SCANS = [
		(name = "120kVp_50mA_FBP", kvp = 120, mA = 50.0),
		(name = "120kVp_150mA_FBP", kvp = 120, mA = 150.0),
		(name = "120kVp_300mA_FBP", kvp = 120, mA = 300.0),
		(name = "80kVp_480mA_FBP", kvp = 80, mA = 480.0),
		(name = "100kVp_250mA_FBP", kvp = 100, mA = 250.0),
		(name = "140kVp_110mA_FBP", kvp = 140, mA = 110.0),
	]
end

# ╔═╡ a1b2c3d4-5004-4000-8000-000000000004
# Simulation options + geometry (stable — changing these re-runs simulate!)
begin
	sim_opts = BS.SimOptions(
		fidelity = :high,
		seed = 1234,
		# n_energy_bins = 50
	)

	sim_recon_xy = 512
	sim_recon_fov_cm = 35.0
	sim_slice_thickness_mm = 1.25
	sim_recon_z_cm = sim_collimation_mm / 10.0                            # 8.0 cm
	sim_n_recon_slices = round(Int, sim_collimation_mm / sim_slice_thickness_mm)  # 64
	sim_matrix_size = (sim_recon_xy, sim_recon_xy, sim_n_recon_slices)

	# Geometry-only ReconOptions for simulate! workspace creation
	# (does NOT affect reconstruction filter — change sim_recon_filter for that)
	sim_recon_geom = BS.ReconOptions(
		algorithm = :fdk,
		matrix_size = sim_matrix_size,
		fov_cm = sim_recon_fov_cm,
		z_cm = sim_recon_z_cm,
	)
end

# ╔═╡ 53fdf44c-c940-4f9e-9ec3-4f77748ba370
sim_opts

# ╔═╡ 02f63ae1-4850-4cb6-8daa-8536c3458898
# Build a custom spatial-domain filter kernel from frequency-domain control points.
# Uses same approach as CatSim: ramp kernel → FFT → apply window → IFFT.
function build_custom_filter_kernel(n_kernel::Int, pixel_size::Float32,
	control_x::NTuple, control_y::NTuple)
	# 1. Build ramp (Ram-Lak) kernel in spatial domain
	Δ = pixel_size
	center = (n_kernel + 1) ÷ 2
	kernel = zeros(Float32, n_kernel)
	for i in 1:n_kernel
		k = i - center
		if k == 0
			kernel[i] = 1.0f0 / (4.0f0 * Δ)
		elseif k % 2 != 0
			kernel[i] = -1.0f0 / (Float32(π)^2 * Float32(k)^2 * Δ)
		end
	end

	# 2. FFT → apply piecewise-linear window → IFFT
	shifted = fftshift(kernel)
	freq = fft(shifted)
	nyquist = n_kernel ÷ 2
	for k in 0:(n_kernel-1)
		f_idx = k <= nyquist ? k : n_kernel - k
		f_norm = Float64(f_idx) / Float64(nyquist)
		# Piecewise linear interpolation of control points
		w = 0.0
		for j in 1:(length(control_x)-1)
			if control_x[j] <= f_norm <= control_x[j+1]
				t = (f_norm - control_x[j]) / (control_x[j+1] - control_x[j])
				w = control_y[j] * (1.0 - t) + control_y[j+1] * t
				break
			end
		end
		freq[k+1] *= w
	end
	result = ifft(freq)
	Float32.(real(ifftshift(result)))
end

# ╔═╡ 24e12422-f5a0-4bb7-a46e-2949df45d75c
# Custom filter apodization — tune these control points to match GE STANDARD kernel MTF.
# Format: (f_norm, weight) where f_norm ∈ [0,1] (fraction of Nyquist).
# Our :standard uses CatSim's [1.0, 0.934, 0.744, 0.443, 0.053].
# GE STANDARD kernel rolls off faster — try more aggressive apodization.
custom_filter_control = (
	x = (0.0, 0.25, 0.5, 0.6, 0.75, 1.0),
	y = (1.0, 0.95, 0.60, 0.55, 0.25, 0.05),   # ← tweak these to match clinical MTF
	# y = (1.0, 0.78, 0.38, 0.10, 0.005), #Lower y-values = softer kernel (less high-frequency content)
	# y = (1.0, 0.70, 0.30, 0.5, 0.005)
)

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

# ╔═╡ a1b2c3d4-6062-4000-8000-000000000062
md"""
### Two-Material Beam Hardening Correction

**Algorithm:** Martinez/Fessler 2022 "2DCalBH" — projection-domain, no hard segmentation.

1. Water-only polynomial BHC → FDK → preliminary HU image
2. Smooth tissue fraction decomposition (C1 smoothstep, 100–500 HU)
3. Forward-project bone image → bone/soft line integrals per ray
4. Exact 2-material polychromatic correction from known spectrum
5. Apply correction to raw sinogram → final FDK

Toggle `bhc_enabled` above to enable/disable (does NOT re-run simulate!).
"""

# ╔═╡ a1b2c3d4-6060-4000-8000-000000000060
# Two-material BHC toggle — tweak freely (does NOT re-run simulate!)
# When enabled, applies projection-domain water+bone BHC to noisy sinogram before FDK.
# When disabled, raw polychromatic sinogram goes straight to FDK (no BHC at all).
bhc_enabled = true

# ╔═╡ a1b2c3d4-6010-4000-8000-000000000010
# Scan 1: 120 kVp / 50 mA — SIMULATE (GPU → ideal sinogram → free GPU)
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
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
		sim_phantom_gpu,
	)
	BS.simulate!(
		ws,
		sim_phantom_gpu,
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
	)

	result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, name = sc.name, kvp = sc.kvp)
	ws = nothing
	GC.gc(true)
	result
end

# ╔═╡ a1b2c3d4-6013-4000-8000-000000000013
# Scan 2: 120 kVp / 150 mA — SIMULATE
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
	@info "Simulating: $(sc.name)..."
	ws = BS.create_eict_workspace(
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
		sim_phantom_gpu,
	)
	BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
	result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, name = sc.name, kvp = sc.kvp)
	ws = nothing
	GC.gc(true)
	result
end

# ╔═╡ a1b2c3d4-6016-4000-8000-000000000016
# Scan 3: 120 kVp / 300 mA — SIMULATE
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
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
		sim_phantom_gpu,
	)
	BS.simulate!(
		ws,
		sim_phantom_gpu,
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
	)
	result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, name = sc.name, kvp = sc.kvp)
	ws = nothing
	GC.gc(true)
	result
end

# ╔═╡ a1b2c3d4-6019-4000-8000-000000000019
# Scan 4: 80 kVp / 480 mA — SIMULATE
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
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
		sim_phantom_gpu,
	)
	BS.simulate!(
		ws,
		sim_phantom_gpu,
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
	)
	result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, name = sc.name, kvp = sc.kvp)
	ws = nothing
	GC.gc(true)
	result
end

# ╔═╡ a1b2c3d4-6022-4000-8000-000000000022
# Scan 5: 100 kVp / 250 mA — SIMULATE
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
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom,
		sim_phantom_gpu,
	)
	BS.simulate!(ws, sim_phantom_gpu, sim_scanner, prot, sim_opts, sim_recon_geom)
	result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, name = sc.name, kvp = sc.kvp)
	ws = nothing
	GC.gc(true)
	result
end

# ╔═╡ a1b2c3d4-6025-4000-8000-000000000025
# Scan 6: 140 kVp / 110 mA — SIMULATE
sim_sino_6 = let
	sc = SE_SIM_SCANS[6]
	prot = BS.CTProtocol(
		kVp = sc.kvp,
		mA = sc.mA,
		views = sim_n_views,
		rotation_time = sim_rotation_time,
		collimation_mm = sim_collimation_mm,
		additional_filters = additional_filters
	)
	@info "Simulating: $(sc.name)..."
	ws = BS.create_eict_workspace(sim_scanner, prot, sim_opts, sim_recon_geom, sim_phantom_gpu)
	
	BS.simulate!(
		ws,
		sim_phantom_gpu,
		sim_scanner,
		prot,
		sim_opts,
		sim_recon_geom
	)
	result = (sino = Array(ws.sino_noisy_out), geom = ws.geom, name = sc.name, kvp = sc.kvp)
	ws = nothing
	GC.gc(true)
	result
end

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

# ╔═╡ c8334797-0992-4e3b-ac6a-f207f313013e
md"""
## Lung Window
"""

# ╔═╡ a1b2c3d4-6061-4000-8000-000000000061
# Calibrate two-material BHC models (one per kVp) from resolve_spectrum pipeline.
# Reference energy = spectrum effective energy (kVp-dependent), so μ_water_ref and
# μ_bone_ref correctly vary with tube voltage — matching clinical scanner behavior.
bhc_models = let
	models = Dict{Int, BS.TwoMaterialBHC}()
	for kvp in [120, 80, 100, 140]
		prot = BS.CTProtocol(kVp = kvp, additional_filters = additional_filters)
		e, w = BS.resolve_spectrum(sim_opts, prot; scanner = sim_scanner)
		ref_E = sum(e .* w) / sum(w)
		models[kvp] = BS.calibrate_bhc_two_material(e, w; order = 3, reference_energy_keV = ref_E, hu_low = 800, hu_high = 5000)
		@info "BHC $(kvp) kVp: ref_E=$(round(ref_E, digits=1)) keV, μ_water_ref=$(round(models[kvp].μ_water_ref, digits=5)), μ_bone_ref=$(round(models[kvp].μ_bone_ref, digits=5))"
	end
	models
end

# ╔═╡ fb6cdb7c-19d0-4f86-950d-464307f0d786
bhc_models[120].μ_water_ref

# ╔═╡ 4370ad68-eebd-4f09-85f0-acb556be9fe2
begin
	# Reconstruction filter (tweak freely — only re-runs recon + HU, NOT simulate!)
	sim_recon_filter = :standard
	sim_recon_cutoff = 1.0 # MUST STAY at 1.0
	# Dose-independent noise floor (σ HU) — tune to match clinical high-mA noise
	# σ_total = √(σ_quantum² + σ_floor²). Start at 0, increase until 300mA matches.
	sim_noise_floor_hu = 28.0
end

# ╔═╡ a1b2c3d4-6011-4000-8000-000000000011
# Scan 1: 120 kVp / 50 mA — RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_1 = let
	sino_gpu = MtlArray(sim_sino_1.sino)
	sino_bhc = nothing
	if bhc_enabled
		sino_bhc = BS.apply_bhc_two_material(sino_gpu, bhc_models[120],
			sim_sino_1.geom, sim_matrix_size; volume_extent = sim_phantom_gpu.extent)
	end
	sino_fdk = MtlArray(something(sino_bhc, sim_sino_1.sino))
	geom = sim_sino_1.geom
	recon_size = sim_matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(
		sino_fdk,
		geom,
		recon_size;
		filter = sim_recon_filter, cutoff = sim_recon_cutoff,
	)
	vol = Array(BS.reconstruct!(ws_fdk, sino_fdk, geom, recon_size))
	ws_fdk = nothing
	sino_gpu = nothing
	sino_fdk = nothing
	GC.gc(true)
	(volume = vol, name = sim_sino_1.name, kvp = sim_sino_1.kvp)
end

# ╔═╡ a1b2c3d4-6012-4000-8000-000000000012
# Scan 1: 120 kVp / 50 mA — HU CONVERSION (CPU only)
sim_hu_1 = let
	μ_w = bhc_models[120].μ_water_ref
	recon_hu = Float32.(BS.to_hounsfield(sim_recon_1.volume; μ_water = μ_w))
	BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
	(name = sim_recon_1.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ 9f983faf-9ba9-49ff-84ec-dc549e697d4c
let
	f = CM.Figure()
	ax = CM.Axis(f[1, 1], aspect = CM.DataAspect(), title = "120 kVp / 50 mA")
	hm = CM.heatmap!(sim_hu_1.recon[:, :, 3]; colormap = :grays, colorrange = (-200, 500))
	CM.Colorbar(f[1, 2], hm)
	f
end

# ╔═╡ a1b2c3d4-6014-4000-8000-000000000014
sim_recon_2 = let
	sino_gpu = MtlArray(sim_sino_2.sino)
	sino_bhc = nothing
	if bhc_enabled
		sino_bhc = BS.apply_bhc_two_material(sino_gpu, bhc_models[120],
			sim_sino_2.geom, sim_matrix_size; volume_extent = sim_phantom_gpu.extent)
	end
	sino_fdk = MtlArray(something(sino_bhc, sim_sino_2.sino))
	geom = sim_sino_2.geom
	recon_size = sim_matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(
		sino_fdk,
		geom,
		recon_size;
		filter = sim_recon_filter, cutoff = sim_recon_cutoff,
	)
	# # Inject custom filter kernel — overwrite workspace's pre-computed kernel
	# n_kernel = length(ws_fdk.filter_kernel)
	# pixel_size = Float32(geom.pixel_size)  # cm (same units as workspace)
	# custom_kernel = build_custom_filter_kernel(
	# 	n_kernel,
	# 	pixel_size,
	# 	custom_filter_control.x,
	# 	custom_filter_control.y
	# )
	# copyto!(ws_fdk.filter_kernel, custom_kernel)
	vol = Array(BS.reconstruct!(ws_fdk, sino_fdk, geom, recon_size))
	ws_fdk = nothing
	sino_gpu = nothing
	sino_fdk = nothing
	GC.gc(true)
	(volume = vol, name = sim_sino_2.name, kvp = sim_sino_2.kvp)
end

# ╔═╡ a1b2c3d4-6015-4000-8000-000000000015
# Scan 2: 120 kVp / 150 mA — HU CONVERSION (CPU only)
sim_hu_2 = let
	μ_w = bhc_models[120].μ_water_ref
	recon_hu = Float32.(BS.to_hounsfield(sim_recon_2.volume; μ_water = μ_w))
	BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
	(name = sim_recon_2.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ cbf028e3-738f-4adc-8caa-3cb14f61d16d
let
	f = CM.Figure()
	ax = CM.Axis(f[1, 1], aspect = CM.DataAspect(), title = "120 kVp / 150 mA")
	hm = CM.heatmap!(sim_hu_2.recon[:, :, 3]; colormap = :grays, colorrange = (-200, 500))
	CM.Colorbar(f[1, 2], hm)
	f
end

# ╔═╡ a1b2c3d4-6017-4000-8000-000000000017
# Scan 3: 120 kVp / 300 mA — RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_3 = let
	sino_gpu = MtlArray(sim_sino_3.sino)
	sino_bhc = nothing
	if bhc_enabled
		sino_bhc = BS.apply_bhc_two_material(sino_gpu, bhc_models[120],
			sim_sino_3.geom, sim_matrix_size; volume_extent = sim_phantom_gpu.extent)
	end
	sino_fdk = MtlArray(something(sino_bhc, sim_sino_3.sino))
	geom = sim_sino_3.geom
	recon_size = sim_matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(
		sino_fdk,
		geom,
		recon_size;
		filter = sim_recon_filter, cutoff = sim_recon_cutoff,
	)
	vol = Array(BS.reconstruct!(ws_fdk, sino_fdk, geom, recon_size))
	ws_fdk = nothing
	sino_gpu = nothing
	sino_fdk = nothing
	GC.gc(true)
	(volume = vol, name = sim_sino_3.name, kvp = sim_sino_3.kvp)
end

# ╔═╡ a1b2c3d4-6018-4000-8000-000000000018
# Scan 3: 120 kVp / 300 mA — HU CONVERSION (CPU only)
sim_hu_3 = let
	μ_w = bhc_models[120].μ_water_ref
	recon_hu = Float32.(BS.to_hounsfield(sim_recon_3.volume; μ_water = μ_w))
	BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
	(name = sim_recon_3.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ a1b2c3d4-6020-4000-8000-000000000020
# Scan 4: 80 kVp / 480 mA — RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_4 = let
	sino_gpu = MtlArray(sim_sino_4.sino)

	sino_bhc = nothing
	if bhc_enabled
		sino_bhc = BS.apply_bhc_two_material(sino_gpu, bhc_models[80],
			sim_sino_4.geom, sim_matrix_size; volume_extent = sim_phantom_gpu.extent)
	end

	sino_fdk = MtlArray(something(sino_bhc, sim_sino_4.sino))
	geom = sim_sino_4.geom
	recon_size = sim_matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(
		sino_fdk,
		geom,
		recon_size;
		filter = sim_recon_filter, cutoff = sim_recon_cutoff,
	)
	vol = Array(BS.reconstruct!(ws_fdk, sino_fdk, geom, recon_size))
	ws_fdk = nothing
	sino_gpu = nothing
	sino_fdk = nothing
	GC.gc(true)
	(volume = vol, name = sim_sino_4.name, kvp = sim_sino_4.kvp)
end

# ╔═╡ a1b2c3d4-6021-4000-8000-000000000021
# Scan 4: 80 kVp / 480 mA — HU CONVERSION (CPU only)
sim_hu_4 = let
	μ_w = bhc_models[80].μ_water_ref
	recon_hu = Float32.(BS.to_hounsfield(sim_recon_4.volume; μ_water = μ_w))
	BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
	(name = sim_recon_4.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ a1b2c3d4-6023-4000-8000-000000000023
# Scan 5: 100 kVp / 250 mA — RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_5 = let
	sino_gpu = MtlArray(sim_sino_5.sino)
	sino_bhc = bhc_enabled ? BS.apply_bhc_two_material(sino_gpu, bhc_models[100],
		sim_sino_5.geom, sim_matrix_size; volume_extent = sim_phantom_gpu.extent) : nothing
	sino_fdk = MtlArray(something(sino_bhc, sim_sino_5.sino))
	geom = sim_sino_5.geom
	recon_size = sim_matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(sino_fdk, geom, recon_size; filter = sim_recon_filter, cutoff = sim_recon_cutoff)
	vol = Array(BS.reconstruct!(ws_fdk, sino_fdk, geom, recon_size))
	ws_fdk = nothing
	sino_gpu = nothing
	sino_fdk = nothing
	GC.gc(true)
	(volume = vol, name = sim_sino_5.name, kvp = sim_sino_5.kvp)
end

# ╔═╡ a1b2c3d4-6024-4000-8000-000000000024
# Scan 5: 100 kVp / 250 mA — HU CONVERSION (CPU only)
sim_hu_5 = let
	μ_w = bhc_models[100].μ_water_ref
	recon_hu = Float32.(BS.to_hounsfield(sim_recon_5.volume; μ_water = μ_w))
	BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
	(name = sim_recon_5.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ a1b2c3d4-6026-4000-8000-000000000026
# Scan 6: 140 kVp / 110 mA — RECONSTRUCT (BHC → GPU FDK → CPU volume)
sim_recon_6 = let
	sino_gpu = MtlArray(sim_sino_6.sino)
	sino_bhc = bhc_enabled ? BS.apply_bhc_two_material(sino_gpu, bhc_models[140],
		sim_sino_6.geom, sim_matrix_size; volume_extent = sim_phantom_gpu.extent) : nothing
	sino_fdk = MtlArray(something(sino_bhc, sim_sino_6.sino))
	geom = sim_sino_6.geom
	recon_size = sim_matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(sino_fdk, geom, recon_size; filter = sim_recon_filter, cutoff = sim_recon_cutoff)
	vol = Array(BS.reconstruct!(ws_fdk, sino_fdk, geom, recon_size))
	ws_fdk = nothing
	sino_gpu = nothing
	sino_fdk = nothing
	GC.gc(true)
	(volume = vol, name = sim_sino_6.name, kvp = sim_sino_6.kvp)
end

# ╔═╡ a1b2c3d4-6027-4000-8000-000000000027
# Scan 6: 140 kVp / 110 mA — HU CONVERSION (CPU only)
sim_hu_6 = let
	μ_w = bhc_models[140].μ_water_ref
	recon_hu = Float32.(BS.to_hounsfield(sim_recon_6.volume; μ_water = μ_w))
	BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
	(name = sim_recon_6.name, recon = recon_hu, mu_water = μ_w)
end

# ╔═╡ a1b2c3d4-5013-4000-8000-000000000013
sim_results_fbp = [
	sim_hu_1,
	sim_hu_2,
	sim_hu_3,
	sim_hu_4,
	sim_hu_5,
	sim_hu_6,
]

# ╔═╡ d90ed7f3-76bf-4c44-8a0f-2e4f7515c2b6
sim_oriented = let
	# --- 1. BASE ORIENTATION (Pick one) ---
	orient = identity
	# orient = rotr90
	# orient = rot180
	# orient = rotl90

	# --- 2. LAYERED MODIFIERS (Uncomment to add a flip/transpose) ---
	# orient = (f -> s -> reverse(f(s), dims=1))(orient)  # Flip Up-Down
	orient = (f -> s -> reverse(f(s), dims = 2))(orient)  # Flip Left-Right
	# orient = (f -> s -> permutedims(f(s)))(orient)     # Transpose

	# --- 3. EXECUTION ---
	[(name = r.name,
		recon = Float32.(mapslices(orient, r.recon, dims = (1, 2))),
		mu_water = r.mu_water)
	 for r in sim_results_fbp]
end

# ╔═╡ 76883b62-cb08-435d-8214-9b0e86c404d8
# Segment the simulated recon independently (finds center + Ca400 anchor in sim image)
sim_seg_result = let
	ref = sim_oriented[min(2, end)].recon
	mid_z = size(ref, 3) ÷ 2
	mask, rods, center = segment_gammex_rods(ref[:, :, mid_z]; fov_cm = 35.0)
	(mask = mask, rods = rods, center = center, slice_idx = mid_z)
end;

# ╔═╡ b3197a6f-3a23-43f3-b63f-f2f49a1ecb45
sim_measurements = [
	measure_scan(r.recon, sim_seg_result.mask, sim_seg_result.rods, sim_seg_result.center, "sim_$(r.name)")
	for r in sim_oriented
];

# ╔═╡ 5a61d098-5d28-4c00-b360-3dd76da478d4
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
		"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]

	n_sims = min(length(sim_measurements), length(base_clinical_idx))
	active_clinical_idx = base_clinical_idx[1:n_sims]
	active_labels = base_scan_labels[1:n_sims]
	colors = CM.cgrad(:tab10, max(n_sims, 2), categorical = true)

	fig = CM.Figure(size = (750, 900), fontsize = 11)

	# --- Top: Calcium rods ---
	ax_ca = CM.Axis(fig[1, 1]; title = "Calcium Rods", subtitle = "Clinical vs Simulated",
		xlabel = "Clinical HU", ylabel = "Simulated HU")

	ca_clin_all = Float64[]
	ca_sim_all = Float64[]

	for (k, (ci, sm_k)) in enumerate(zip(active_clinical_idx, sim_measurements[1:n_sims]))
		cm = se_measurements[ci]
		ca_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "Ca")]
		if !isempty(ca_idx)
			CM.scatter!(ax_ca, cm.rod_means[ca_idx], sm_k.rod_means[ca_idx];
				color = colors[k], markersize = 10, label = active_labels[k])
			append!(ca_clin_all, cm.rod_means[ca_idx])
			append!(ca_sim_all, sm_k.rod_means[ca_idx])
		end
	end
	CM.lines!(ax_ca, [-100, 1400], [-100, 1400];
		color = :gray60, linestyle = :dash, linewidth = 1, label = "Unity (y = x)")
	if length(ca_clin_all) > 1
		X_ca = hcat(ones(length(ca_clin_all)), ca_clin_all)
		b_ca, m_ca = X_ca \ ca_sim_all
		r_ca = cor(ca_clin_all, ca_sim_all)
		rmse_ca = sqrt(sum((ca_sim_all .- ca_clin_all) .^ 2) / length(ca_clin_all))
		x_fit_ca = range(extrema(ca_clin_all)..., length = 100)
		sign_ca = b_ca >= 0 ? " + " : " - "
		eq_ca = "y = $(round(m_ca, digits=3))x$(sign_ca)$(round(abs(b_ca), digits=1))"
		CM.lines!(ax_ca, collect(x_fit_ca), m_ca .* collect(x_fit_ca) .+ b_ca;
			color = :black, linewidth = 0.8, label = "Linear fit")
		# Stats box bottom-right via poly! background + text
		CM.poly!(ax_ca,
			CM.Point2f[(0.60, 0.02), (0.98, 0.02), (0.98, 0.22), (0.60, 0.22)];
			color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
		CM.text!(ax_ca, 0.62, 0.18; space = :relative, align = (:left, :top), fontsize = 10,
			text = "$(eq_ca)\nr = $(round(r_ca, digits=4))\nRMSE = $(round(rmse_ca, digits=1)) HU")
	end
	CM.axislegend(ax_ca; position = :lt, labelsize = 9)

	# --- Bottom: Iodine rods ---
	ax_i = CM.Axis(fig[2, 1]; title = "Iodine Rods", subtitle = "Clinical vs Simulated",
		xlabel = "Clinical HU", ylabel = "Simulated HU")

	i_clin_all = Float64[]
	i_sim_all = Float64[]

	for (k, (ci, sm_k)) in enumerate(zip(active_clinical_idx, sim_measurements[1:n_sims]))
		cm = se_measurements[ci]
		i_idx = [i for i in 1:length(cm.rod_names) if startswith(cm.rod_names[i], "I ")]
		if !isempty(i_idx)
			CM.scatter!(ax_i, cm.rod_means[i_idx], sm_k.rod_means[i_idx];
				color = colors[k], markersize = 10, label = active_labels[k])
			append!(i_clin_all, cm.rod_means[i_idx])
			append!(i_sim_all, sm_k.rod_means[i_idx])
		end
	end
	CM.lines!(ax_i, [-50, 500], [-50, 500];
		color = :gray60, linestyle = :dash, linewidth = 1, label = "Unity (y = x)")
	if length(i_clin_all) > 1
		X_i = hcat(ones(length(i_clin_all)), i_clin_all)
		b_i, m_i = X_i \ i_sim_all
		r_i = cor(i_clin_all, i_sim_all)
		rmse_i = sqrt(sum((i_sim_all .- i_clin_all) .^ 2) / length(i_clin_all))
		x_fit_i = range(extrema(i_clin_all)..., length = 100)
		sign_i = b_i >= 0 ? " + " : " - "
		eq_i = "y = $(round(m_i, digits=3))x$(sign_i)$(round(abs(b_i), digits=1))"
		CM.lines!(ax_i, collect(x_fit_i), m_i .* collect(x_fit_i) .+ b_i;
			color = :black, linewidth = 0.8, label = "Linear fit")
		# Stats box bottom-right via poly! background + text
		CM.poly!(ax_i,
			CM.Point2f[(0.60, 0.02), (0.98, 0.02), (0.98, 0.22), (0.60, 0.22)];
			color = (:white, 0.9), strokecolor = :gray50, strokewidth = 1, space = :relative)
		CM.text!(ax_i, 0.62, 0.18; space = :relative, align = (:left, :top), fontsize = 10,
			text = "$(eq_i)\nr = $(round(r_i, digits=4))\nRMSE = $(round(rmse_i, digits=1)) HU")
	end
	CM.axislegend(ax_i; position = :lt, labelsize = 9)

	fig
end

# ╔═╡ ae4465c7-65f0-4ecf-8d97-f537eadce60c
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
		"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]

	target_idx = min(2, length(sim_measurements))

	cm = se_measurements[base_clinical_idx[target_idx]]
	sm = sim_measurements[target_idx]
	label = base_scan_labels[target_idx]

	# Filter: only Ca and I rods (skip Water, SW ref)
	keep = [i for i in 1:length(cm.rod_names)
			if startswith(cm.rod_names[i], "Ca") || startswith(cm.rod_names[i], "I ")]
	names = cm.rod_names[keep]
	cm_means = cm.rod_means[keep]
	cm_stds = cm.rod_stds[keep]
	sm_means = sm.rod_means[keep]
	sm_stds = sm.rod_stds[keep]
	n = length(names)

	# Color by material: Ca = blue/orange tones, I = green/red tones
	is_ca = [startswith(nm, "Ca") for nm in names]
	clin_colors = [c ? :steelblue : :seagreen for c in is_ca]
	sim_colors = [c ? :darkorange : :indianred for c in is_ca]

	fig = CM.Figure(size = (1100, 500), fontsize = 10)
	ax = CM.Axis(fig[1, 1]; title = "Rod HU ($label)", subtitle = "Clinical vs Simulated", ylabel = "Mean HU", xticks = (1:n, names), xticklabelrotation = π / 4)

	# Bars — Ca group
	ca_idx = findall(is_ca)
	if !isempty(ca_idx)
		CM.barplot!(ax, ca_idx .- 0.2, cm_means[ca_idx]; width = 0.35,
			color = :steelblue, label = "Clinical Ca")
		CM.errorbars!(ax, ca_idx .- 0.2, cm_means[ca_idx], cm_stds[ca_idx];
			color = :black, whiskerwidth = 3)
		CM.barplot!(ax, ca_idx .+ 0.2, sm_means[ca_idx]; width = 0.35,
			color = :darkorange, label = "Simulated Ca")
		CM.errorbars!(ax, ca_idx .+ 0.2, sm_means[ca_idx], sm_stds[ca_idx];
			color = :black, whiskerwidth = 3)
	end

	# Bars — I group
	i_idx = findall(.!is_ca)
	if !isempty(i_idx)
		CM.barplot!(ax, i_idx .- 0.2, cm_means[i_idx]; width = 0.35,
			color = :seagreen, label = "Clinical I")
		CM.errorbars!(ax, i_idx .- 0.2, cm_means[i_idx], cm_stds[i_idx];
			color = :black, whiskerwidth = 3)
		CM.barplot!(ax, i_idx .+ 0.2, sm_means[i_idx]; width = 0.35,
			color = :indianred, label = "Simulated I")
		CM.errorbars!(ax, i_idx .+ 0.2, sm_means[i_idx], sm_stds[i_idx];
			color = :black, whiskerwidth = 3)
	end

	CM.axislegend(ax; position = :lt)
	fig
end

# ╔═╡ dc0ce201-0efa-4bc0-bd3c-2075650fc67a
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120/50", "120/150", "120/300", "80/480", "100/250", "140/110"]

	n_sims = min(length(sim_measurements), length(base_clinical_idx))
	active_labels = base_scan_labels[1:n_sims]

	water_idx = 1  # "Water (O)" is first rod
	clin_σ = [se_measurements[base_clinical_idx[i]].rod_stds[water_idx] for i in 1:n_sims]
	@info clin_σ
	sim_σ = [sim_measurements[i].rod_stds[water_idx] for i in 1:n_sims]
	@info sim_σ

	fig = CM.Figure(size = (800, 400), fontsize = 11)
	ax = CM.Axis(fig[1, 1]; title = "Water ROI Noise (σ)", subtitle = "Clinical vs Simulated",
		ylabel = "σ (HU)", xticks = (1:n_sims, active_labels), xlabel = "kVp / mA")
	CM.barplot!(ax, collect(1:n_sims) .- 0.2, clin_σ; width = 0.35, color = :steelblue, label = "Clinical")
	CM.barplot!(ax, collect(1:n_sims) .+ 0.2, sim_σ; width = 0.35, color = :darkorange, label = "Simulated")
	CM.axislegend(ax; position = :rt)

	fig
end

# ╔═╡ 4e15fc9c-217c-4e95-9a0c-bab491f99e4d
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_120_low_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[1].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-200, 500))
	# CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-200, 500))
	# CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ f42424d5-73ec-43a2-afb0-ae741629d8d6
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_120_low_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[1].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ cb2b2b03-9aa7-446a-ad61-9e7152f9b1eb
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_120_mid_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[min(2, end)].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-200, 500))
	# CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-200, 500))
	# CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ 0d6bbebf-e15d-4ae2-bb99-974b5e3f5793
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_120_mid_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[min(2, end)].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ 21eda799-2429-4fb3-a89c-783bfcdb3beb
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_120_high_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[3].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ 8f805112-add3-4505-81c3-b465cbd5f512
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_80_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[4].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-200, 500))
	CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ 83f368fa-178a-4538-b2bd-279bb199b881
# Each image segmented independently — clinical uses seg_result, simulated uses sim_seg_result
let
	# Clinical
	clin_hu = hu_120_mid_fbp[:, :, seg_result.slice_idx]
	clin_mask = Float32.(seg_result.mask)
	clin_mask[clin_mask.==0] .= NaN

	# Simulated
	sim_hu = sim_oriented[min(2, end)].recon[:, :, sim_seg_result.slice_idx]
	sim_mask = Float32.(sim_seg_result.mask)
	sim_mask[sim_mask.==0] .= NaN

	fig = CM.Figure(size = (1100, 500), fontsize = 11)

	ax1 = CM.Axis(fig[1, 1]; title = "Clinical",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax1, clin_hu; colormap = :grays, colorrange = (-2000, 200))
	CM.heatmap!(ax1, clin_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax1)
	CM.hidespines!(ax1)

	ax2 = CM.Axis(fig[1, 2]; title = "Simulated",
		aspect = CM.DataAspect(), yreversed = true)
	CM.heatmap!(ax2, sim_hu; colormap = :grays, colorrange = (-1200, 200))
	CM.heatmap!(ax2, sim_mask; colormap = :turbo, colorrange = (1, 27), nan_color = :transparent)
	CM.hidedecorations!(ax2)
	CM.hidespines!(ax2)

	fig
end

# ╔═╡ 8d4d7038-429f-4402-9dcc-4b4263e9db41
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
		"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]

	n_sims = min(length(sim_measurements), length(base_clinical_idx))

	fig = CM.Figure(size = (900, 900), fontsize = 11)

	for i in 1:n_sims
		row = (i - 1) ÷ 2 + 1
		col = (i - 1) % 2 + 1

		ax = CM.Axis(fig[row, col]; title = base_scan_labels[i],
			subtitle = "Clinical vs Simulated",
			xlabel = "Spatial frequency (lp/cm)", ylabel = "NPS (HU²·cm²)", yscale = log10)

		cm = se_measurements[base_clinical_idx[i]]
		sm = sim_measurements[i]

		f_c, v_c = cm.nps.frequencies, cm.nps.nps_1d
		good_c = v_c .> 0
		CM.lines!(ax, f_c[good_c], v_c[good_c];
			color = :steelblue, linewidth = 1.5, label = "Clinical")

		f_s, v_s = sm.nps.frequencies, sm.nps.nps_1d
		good_s = v_s .> 0
		CM.lines!(ax, f_s[good_s], v_s[good_s];
			color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Simulated")

		CM.axislegend(ax; position = :rt, labelsize = 8)
	end

	fig
end

# ╔═╡ 4de369da-bad2-43bd-bf5a-ba654d26c76f
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
		"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]

	n_sims = min(length(sim_measurements), length(base_clinical_idx))

	fig = CM.Figure(size = (900, 900), fontsize = 11)

	for i in 1:n_sims
		row = (i - 1) ÷ 2 + 1
		col = (i - 1) % 2 + 1

		ax = CM.Axis(fig[row, col]; title = base_scan_labels[i],
			subtitle = "Clinical vs Simulated",
			xlabel = "Spatial frequency (lp/cm)", ylabel = "MTF",
			limits = (nothing, nothing, 0, 1.05))
		CM.hlines!(ax, [0.5, 0.1]; color = :gray80, linestyle = :dash, linewidth = 0.8)

		cm = se_measurements[base_clinical_idx[i]]
		sm = sim_measurements[i]

		CM.lines!(ax, cm.mtf.frequencies, cm.mtf.mtf;
			color = :steelblue, linewidth = 1.5, label = "Clinical")
		CM.lines!(ax, sm.mtf.frequencies, sm.mtf.mtf;
			color = :orangered, linewidth = 1.5, linestyle = :dash, label = "Simulated")

		CM.axislegend(ax; position = :rt, labelsize = 8)
	end

	fig
end

# ╔═╡ 11f8b1f8-4927-41a3-a781-9a169739610a
let
	base_clinical_idx = [1, 3, 5, 7, 9, 11]
	base_scan_labels = ["120kVp 50mA", "120kVp 150mA", "120kVp 300mA",
		"80kVp 480mA", "100kVp 250mA", "140kVp 110mA"]

	target_idx = min(2, length(sim_measurements))
	cm = se_measurements[base_clinical_idx[target_idx]]
	sm = sim_measurements[target_idx]
	label = base_scan_labels[target_idx]

	Δ = sm.rod_means .- cm.rod_means

	fig = CM.Figure(size = (1100, 400), fontsize = 10)
	ax = CM.Axis(fig[1, 1]; title = "HU Error", subtitle = "(Simulated − Clinical) — $label", ylabel = "ΔHU", xticks = (1:length(cm.rod_names), cm.rod_names), xticklabelrotation = π / 4)
	CM.barplot!(ax, 1:length(Δ), Δ; color = ifelse.(Δ .> 0, :salmon, :steelblue))
	CM.hlines!(ax, [0]; color = :black, linewidth = 0.8)

	for (i, d) in enumerate(Δ)
		CM.text!(ax, i, d + sign(d) * 3;
			text = "$(round(d, digits=1))", fontsize = 8, align = (:center, d > 0 ? :bottom : :top))
	end

	fig
end

# ╔═╡ Cell order:
# ╟─05487717-8249-4d89-bf94-f7e7dc379bfb
# ╠═db7ada07-c245-4427-a9d8-1fa43e74e884
# ╠═2872d85c-3d09-414f-a194-4b52764fca93
# ╠═eff433a8-e36a-4b6f-bfb5-73664ceac2f3
# ╠═a0839e3b-bd86-44e6-a282-046dcc6a6d39
# ╠═1caeb654-e68e-47d8-a10d-c4020c676c7b
# ╠═9cd57146-a7ec-4fe8-a313-4d46659d8795
# ╠═6cf5c19a-88f2-4ba7-91d6-f68b4fb816e1
# ╠═ff611999-d9e4-45e1-80e5-a46176c4845a
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
# ╠═0c47ac3f-6d83-47ac-8c94-bc9172db716d
# ╠═e6322fc0-9a18-43a6-a98d-5af80c2a5f01
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
# ╟─a1b2c3d4-7000-4000-8000-e00000000001
# ╠═a1b2c3d4-7001-4000-8000-000000000001
# ╠═a1b2c3d4-5003-4000-8000-000000000003
# ╠═53fdf44c-c940-4f9e-9ec3-4f77748ba370
# ╠═a1b2c3d4-5004-4000-8000-000000000004
# ╠═02f63ae1-4850-4cb6-8daa-8536c3458898
# ╠═24e12422-f5a0-4bb7-a46e-2949df45d75c
# ╠═a1b2c3d4-5005-4000-8000-000000000005
# ╟─a1b2c3d4-6062-4000-8000-000000000062
# ╠═a1b2c3d4-6060-4000-8000-000000000060
# ╠═a1b2c3d4-6010-4000-8000-000000000010
# ╠═a1b2c3d4-6011-4000-8000-000000000011
# ╠═fb6cdb7c-19d0-4f86-950d-464307f0d786
# ╠═a1b2c3d4-6012-4000-8000-000000000012
# ╟─9f983faf-9ba9-49ff-84ec-dc549e697d4c
# ╠═a1b2c3d4-6013-4000-8000-000000000013
# ╠═a1b2c3d4-6014-4000-8000-000000000014
# ╠═a1b2c3d4-6015-4000-8000-000000000015
# ╟─cbf028e3-738f-4adc-8caa-3cb14f61d16d
# ╠═a1b2c3d4-6016-4000-8000-000000000016
# ╠═a1b2c3d4-6017-4000-8000-000000000017
# ╠═a1b2c3d4-6018-4000-8000-000000000018
# ╠═a1b2c3d4-6019-4000-8000-000000000019
# ╠═a1b2c3d4-6020-4000-8000-000000000020
# ╠═a1b2c3d4-6021-4000-8000-000000000021
# ╠═a1b2c3d4-6022-4000-8000-000000000022
# ╠═a1b2c3d4-6023-4000-8000-000000000023
# ╠═a1b2c3d4-6024-4000-8000-000000000024
# ╠═a1b2c3d4-6025-4000-8000-000000000025
# ╠═a1b2c3d4-6026-4000-8000-000000000026
# ╠═a1b2c3d4-6027-4000-8000-000000000027
# ╠═a1b2c3d4-5013-4000-8000-000000000013
# ╟─a1b2c3d4-5014-4000-8000-000000000014
# ╟─e54d80d3-6f67-4cd4-a85f-0e150755bf6e
# ╠═d90ed7f3-76bf-4c44-8a0f-2e4f7515c2b6
# ╠═76883b62-cb08-435d-8214-9b0e86c404d8
# ╠═b3197a6f-3a23-43f3-b63f-f2f49a1ecb45
# ╟─4e15fc9c-217c-4e95-9a0c-bab491f99e4d
# ╟─f42424d5-73ec-43a2-afb0-ae741629d8d6
# ╟─cb2b2b03-9aa7-446a-ad61-9e7152f9b1eb
# ╟─0d6bbebf-e15d-4ae2-bb99-974b5e3f5793
# ╟─21eda799-2429-4fb3-a89c-783bfcdb3beb
# ╟─8f805112-add3-4505-81c3-b465cbd5f512
# ╟─c8334797-0992-4e3b-ac6a-f207f313013e
# ╟─83f368fa-178a-4538-b2bd-279bb199b881
# ╟─5a61d098-5d28-4c00-b360-3dd76da478d4
# ╠═a1b2c3d4-6061-4000-8000-000000000061
# ╟─ae4465c7-65f0-4ecf-8d97-f537eadce60c
# ╟─dc0ce201-0efa-4bc0-bd3c-2075650fc67a
# ╠═4370ad68-eebd-4f09-85f0-acb556be9fe2
# ╟─8d4d7038-429f-4402-9dcc-4b4263e9db41
# ╟─4de369da-bad2-43bd-bf5a-ba654d26c76f
# ╟─11f8b1f8-4927-41a3-a781-9a169739610a
