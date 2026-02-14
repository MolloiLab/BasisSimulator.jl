### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b07a644b-594b-42a2-bab8-37a5eab2a8c4
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()

	using Revise
end

# ╔═╡ f0000002-0001-0001-0001-000000000011
using Unitful: @u_str

# ╔═╡ c2c038a2-675a-40ca-8809-56d19d78600c
using Metal

# ╔═╡ 688cbbde-bdd5-462f-87a3-05c9d26242b5
md"""
# BasisSimulator.jl: Multi-Protocol Physics & Hybrid IR

**A Publication-Quality Comparison of CT Simulation Frameworks**

*Part 2: Spectral Sensitivity, Dose Scaling, and Hybrid Iterative Reconstruction*
"""

# ╔═╡ e69a816d-43f8-48de-8645-7fd16168a928
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ 0735094c-9b93-4fb7-ba47-72a499ffadff
import BasisSimulator as BS

# ╔═╡ dbdff1fe-ce08-427b-b271-c0dad574c1be
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ 26aaea60-657b-4d2c-a26e-2312fa56a76e
import Statistics: mean, std

# ╔═╡ f0000002-0001-0001-0001-000000000010
import XrayAttenuation as XA

# ╔═╡ f0000002-0001-0001-0001-000000000001
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")

# ╔═╡ 0edf6fa8-a8d8-4aa1-b2b3-8f8abb1773b4
UI.TableOfContents()

# ╔═╡ 6d84c6c6-863f-42e6-8ae7-ef00154a9507
md"""
## 1. Simulation Configuration

We use the exact same geometric configuration as Part 1 to ensure 1:1 parity, only varying the protocol parameters.
"""

# ╔═╡ ea44a25a-5358-4aef-b301-d462798141c8
# Simulation configuration - Single Source of Truth
SIM_CONFIG = let
    # --- Scanner Geometry ---
    sid = 540.0                 # Source-to-Iso (mm)
    sdd = 950.0                 # Source-to-Detector (mm)
    magnification = sdd / sid   # 1.759
    detectorColCount = 900      # Total columns
    detectorRowCount = 16       # Total rows

    # CatSim uses 1.0mm at detector face; convert to isocenter for Scanner
    detectorColSize = 1.0 / magnification   # ≈ 0.569 mm at isocenter
    detectorRowSize = 1.0 / magnification   # ≈ 0.569 mm at isocenter

    # --- Clinical Reconstruction Parameters ---
    z_coverage_mm = detectorRowCount * detectorRowSize  # at isocenter
    sliceThickness = 1.0        # mm (clinical slice thickness)
    sliceCount = floor(Int, z_coverage_mm / sliceThickness)

    (
        imageSize = 512,
        fov_mm = 350.0,

        sid = sid,
        sdd = sdd,
        detectorColCount = detectorColCount,
        detectorRowCount = detectorRowCount,
        detectorColSize = detectorColSize,
        detectorRowSize = detectorRowSize,

        sliceCount = sliceCount,
        sliceThickness = sliceThickness,
        z_coverage_mm = z_coverage_mm,

        viewsPerRotation = 984,
        rotationTime = 1.0,

        n_energy_bins = 15,
    )
end

# ╔═╡ 62222977-9635-434c-b3dd-21720c19402c
# Define the Scenarios
SCENARIOS = [
	(name="Low Dose",  kvp=80,  mA=50.0,  label="80kVp/50mA"),
	(name="Standard",  kvp=120, mA=200.0, label="120kVp/200mA"),
	(name="High Dose", kvp=140, mA=400.0, label="140kVp/400mA")
]

# ╔═╡ 3a07da3e-98d4-41f6-b295-cfdb9fbdd19d
protocols = Dict(
	"Low Dose" => BS.CTProtocol(
		kVp=80, mA=50,
		views=SIM_CONFIG.viewsPerRotation,
		rotation_time=SIM_CONFIG.rotationTime
	),

	"Standard" => BS.CTProtocol(
		kVp=120, mA=200,
		views=SIM_CONFIG.viewsPerRotation,
		rotation_time=SIM_CONFIG.rotationTime
	),

	"High Dose" => BS.CTProtocol(
		kVp=140, mA=400,
		views=SIM_CONFIG.viewsPerRotation,
		rotation_time=SIM_CONFIG.rotationTime
	)
)

# ╔═╡ dda5d6eb-0194-4956-99b7-1798cabc497f
let
	fig_verify = CM.Figure(size=(800, 350))
	ax_v = CM.Axis(fig_verify[1,1],
		title="Physics Engine Verification (Protocol Output)",
		xlabel="Energy (keV)", ylabel="Fluence"
	)

	colors = Dict("Low Dose"=>:blue, "Standard"=>:black, "High Dose"=>:red)

	for name in ["Low Dose", "Standard", "High Dose"]
		prot = protocols[name]
		e_raw, w_raw = BS.get_spectrum(prot)
		e_plot, w_plot = BS.downsample_spectrum(e_raw, w_raw, SIM_CONFIG.n_energy_bins)

		CM.lines!(ax_v, e_plot, w_plot, label=name, color=colors[name], linewidth=2)
		CM.scatter!(ax_v, e_plot, w_plot, color=colors[name], markersize=5)
	end

	CM.axislegend(ax_v)
	CM.save(joinpath(FIGURES_DIR, "nb02_dose_comparison.png"), fig_verify, px_per_unit=2)
	fig_verify
end

# ╔═╡ 7e16e89f-f215-41f9-a90c-2541e37c8a7a
md"""
---
## 2. Phantom Generation (Gammex 472)

We create two phantoms using the workspace-based API:
1.  **Gammex 472**: The full phantom with Calcium and Iodine inserts.
2.  **Calibration Water**: A uniform water cylinder for spectral calibration.
"""

# ╔═╡ f0000002-0002-0001-0001-000000000020
# Water calibration phantom (matching notebook 01 pattern)
phantom_water_gpu = let
	nx, ny, nz = SIM_CONFIG.imageSize, SIM_CONFIG.imageSize, SIM_CONFIG.sliceCount
	water_fov_cm = SIM_CONFIG.fov_mm / 10.0
	voxel_cm = water_fov_cm / nx
	z_cm = (SIM_CONFIG.sliceCount * SIM_CONFIG.sliceThickness) / 10.0
	voxel_z_cm = z_cm / nz

	water_mask = zeros(UInt8, nx, ny, nz)
	radius_cm = 16.5
	xs = range(-water_fov_cm/2, water_fov_cm/2, length=nx)
	ys = range(-water_fov_cm/2, water_fov_cm/2, length=ny)
	for k in 1:nz, j in 1:ny, i in 1:nx
		if sqrt(xs[i]^2 + ys[j]^2) <= radius_cm
			water_mask[i, j, k] = UInt8(1)
		end
	end

	air_material = XA.Material(
		"Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
		Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
	)
	water_materials = Dict(0 => air_material, 1 => XA.Materials.water)

	BS.Phantom(Metal.MtlArray(water_mask), water_materials, (voxel_cm, voxel_cm, voxel_z_cm))
end;

# ╔═╡ f0000002-0002-0001-0001-000000000021
# Gammex 472 phantom (using built-in generator)
phantom_basis = BS.create_gammex_472(
	n_voxels = SIM_CONFIG.imageSize,
	n_slices = SIM_CONFIG.sliceCount,
	fov_cm = SIM_CONFIG.fov_mm / 10.0,
	z_cm = (SIM_CONFIG.sliceCount * SIM_CONFIG.sliceThickness) / 10.0
);

# ╔═╡ f0000002-0002-0001-0001-000000000022
# GPU-backed Phantom (preserves materials, voxel_size, origin, fov)
phantom_gammex_gpu = BS.Phantom(
	Metal.MtlArray(phantom_basis.mask),
	phantom_basis.materials,
	phantom_basis.voxel_size,
	phantom_basis.origin,
	phantom_basis.fov
);

# ╔═╡ d7c40e60-b546-41cc-b26c-5363abb24e51
scanner = BS.Scanner(
	source_to_isocenter = SIM_CONFIG.sid,      # mm
	source_to_detector = SIM_CONFIG.sdd,       # mm
	detector_rows = SIM_CONFIG.detectorRowCount,
	detector_cols = SIM_CONFIG.detectorColCount,
	detector_row_size = SIM_CONFIG.detectorRowSize,

	# detector_col_size is the element pitch at isocenter (mm).
	detector_col_size = SIM_CONFIG.detectorColSize,

	detector_shape = BS.CURVED_DETECTOR,

	# --- Hardware fields for build_physics_config() ---
	focal_spot_width = 0.7,                    # mm (small focal spot)
	focal_spot_length = 0.9,                   # mm
	target_angle = 7.0,                        # degrees (anode angle)
	flat_filter_material = :aluminum,
	flat_filter_thickness = 2.5,               # mm inherent filtration
	detector_material = :gadolinium_oxysulfide,
	detector_depth = 0.5,                      # mm GOS scintillator
	fill_factor_row = 0.9,                     # 90% geometric efficiency
	fill_factor_col = 0.9,
	detection_gain = 1.0,
	electronic_noise = 100.0,                  # ADC noise (DAS broken/unused)
);

# ╔═╡ 7d202fe4-6a60-4ee8-aa49-52843e524c53
md"""
## 3. Multi-Protocol Simulation & Calibration

We iterate through the protocols. For each scenario, we perform two distinct scans:
1.  **Water Calibration Scan**: We scan the pure water phantom to measure the mean attenuation (μ\_calib). This accounts for beam hardening and spectral shifts inherent to that specific kVp.
2.  **Object Scan**: We scan the Gammex phantom and reconstruct it.
3.  **HU Conversion**: We use the specific μ\_calib from step 1 to normalize the Gammex reconstruction.
"""

# ╔═╡ d190489f-d0ec-4428-81a6-6bff43be401c
sim_opts = BS.SimOptions(fidelity=:high, seed=1234)

# ╔═╡ 286bb6a0-ce50-4370-a4f0-1f5c8f62f60c
recon_opts = BS.ReconOptions(
    algorithm=:fdk,
    filter=:standard,
    matrix_size=(SIM_CONFIG.imageSize, SIM_CONFIG.imageSize, SIM_CONFIG.sliceCount),
    fov_cm=SIM_CONFIG.fov_mm/10.0,
    z_cm=SIM_CONFIG.sliceCount * SIM_CONFIG.sliceThickness / 10.0,
)

# ╔═╡ ac0e51fd-a1d8-4521-8bfd-7f105f637d16
# Multi-protocol simulation: water calibration + Gammex scan for each kVp
sim_results = let
	results = Dict{String, NamedTuple}()
	recon_size = recon_opts.matrix_size

	for sc in SCENARIOS
		@info "Processing $(sc.name): $(sc.kvp) kVp..."
		prot = protocols[sc.name]

		# --- Water Calibration Scan ---
		@info "  > Calibration..."
		ws_cal = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_water_gpu)
		BS.simulate!(ws_cal, phantom_water_gpu, scanner, prot, sim_opts, recon_opts)

		ws_fdk_cal = BS.create_fdk_recon_workspace(ws_cal.sino_noisy_out, ws_cal.geom, recon_size)
		vol_cal = Array(BS.reconstruct!(ws_fdk_cal, ws_cal.sino_noisy_out, ws_cal.geom, recon_size))
		cx, cy, cz = size(vol_cal) .÷ 2
		z_half = min(cz - 1, 4)
		μ_calib = mean(vol_cal[cx-10:cx+10, cy-10:cy+10, cz-z_half:cz+z_half])

		ws_fdk_cal = nothing; vol_cal = nothing
		ws_cal = nothing; GC.gc(true)

		# --- Object Scan ---
		@info "  > Object Scan..."
		ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
		BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

		# Keep sinogram on CPU for later HIR use
		sino_cpu = Array(ws.sino_noisy_out)

		# FDK reconstruction → HU
		ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
		fdk_hu = BS.to_hounsfield(
			Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
			μ_water=μ_calib
		)
		# Store geometry before freeing
		geom_copy = ws.geom

		ws_fdk = nothing; GC.gc(true)
		ws = nothing; GC.gc(true)

		results[sc.name] = (;
			sinogram = sino_cpu,
			recon = fdk_hu,
			meta = sc,
			mu_calib = μ_calib,
			geom = geom_copy
		)
	end
	results
end

# ╔═╡ 022bcd91-ab8a-404b-a4fe-7a0f213fd9a9
md"""
### Analysis: Spectral Sensitivity & Noise

We measure the HU values of an Iodine insert (20 mg/mL) and the standard deviation of background water.
* **Hypothesis 1 (Physics):** Iodine HU should be **highest at 80 kVp** and lowest at 140 kVp due to the Photoelectric effect dominating at lower energies.
* **Hypothesis 2 (Noise):** Noise should decrease as mAs increases (50 → 400 mAs).
"""

# ╔═╡ e8c85147-2cc2-4530-922a-0898ad9fdf6a
physics_metrics = let
	data = []

	cx, cy, cz = SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.sliceCount ÷ 2

	pix = SIM_CONFIG.fov_mm / SIM_CONFIG.imageSize
	r_pix = 105.0 / pix

	# Iodine 20mg/mL insert (outer ring, last insert)
	ang = 2π * 6 / 7 + π/7
	ix_i = cx + round(Int, r_pix * cos(ang))
	iy_i = cy + round(Int, r_pix * sin(ang))

	r = 5

	for sc in SCENARIOS
		res = sim_results[sc.name]
		vol = res.recon

		roi_iodine = vol[ix_i-r:ix_i+r, iy_i-r:iy_i+r, cz]
		roi_water = vol[cx-20:cx+20, cy-20:cy+20, cz-2:cz+2]

		push!(data, (
			label = sc.label,
			kvp = sc.kvp,
			mAs = sc.mA * SIM_CONFIG.rotationTime,
			iodine_hu = mean(roi_iodine),
			water_noise = std(roi_water)
		))
	end
	data
end

# ╔═╡ c717df6c-f356-43c9-98b1-f0c926eb9384
@bind scen_slice UI.Slider(1:SIM_CONFIG.sliceCount, default=SIM_CONFIG.sliceCount ÷ 2, show_value=true)

# ╔═╡ f889723d-a117-47a9-a120-43f16e827cd8
let
	scen_details = Dict(
		"Low Dose"   => "80 kVp, 50 mA",
		"Standard"   => "120 kVp, 200 mA",
		"High Dose"  => "140 kVp, 400 mA",
	)

	ordered_keys = ["Low Dose", "Standard", "High Dose"]

	fig = CM.Figure(size=(1000, 350))
	vmin, vmax = -200, 500

	for (i, name) in enumerate(ordered_keys)
		if haskey(sim_results, name)
			kvp_text = scen_details[name]
			full_title = "$name\n$kvp_text"

			ax = CM.Axis(fig[1,i], title=full_title, aspect=CM.DataAspect())
			vol = sim_results[name].recon
			hm = CM.heatmap!(ax, vol[:,:,scen_slice], colormap=:grays, colorrange=(vmin, vmax))
			CM.hidedecorations!(ax)

			if i == 3
				CM.Colorbar(fig[1, 4], hm, label="HU")
			end
		end
	end
	CM.save(joinpath(FIGURES_DIR, "nb02_dose_comparison_recon.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ 67d7edad-5929-43dd-8681-0b079c08753d
let
	fig = CM.Figure(size=(1000, 400))

	# Plot 1: Spectral Sensitivity (Iodine Contrast)
	ax1 = CM.Axis(fig[1,1], title="Spectral Sensitivity (Iodine 20mg/mL)", xlabel="Protocol", ylabel="Mean HU")

	x_vals = 1:3
	y_iodine = [d.iodine_hu for d in physics_metrics]

	CM.barplot!(ax1, x_vals, y_iodine, color=:purple)
	ax1.xticks = (x_vals, [d.label for d in physics_metrics])

	# Plot 2: Noise Levels
	ax2 = CM.Axis(fig[1,2], title="Image Noise (Water StdDev)", xlabel="Protocol", ylabel="StdDev (HU)")

	y_noise = [d.water_noise for d in physics_metrics]
	CM.barplot!(ax2, x_vals, y_noise, color=:orange)
	ax2.xticks = (x_vals, [d.label for d in physics_metrics])

	CM.save(joinpath(FIGURES_DIR, "nb02_spectral_noise.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ 59c9285a-bb87-4ecc-babd-9f5943a8cc53
md"""
## 4. Hybrid Iterative Reconstruction (Low Dose Rescue)

The **80 kVp / 50 mA** protocol yielded the highest contrast but also the highest noise. This is a classic candidate for Hybrid Iterative Reconstruction (HIR).

We take the raw sinogram from that specific run and reconstruct it using HIR at three strength levels (1, 3, 5), comparing against FDK. We use the same μ\_calib derived from the 80 kVp water scan to ensure consistent HU calibration.

HIR uses PWLS refinement with Huber regularization on top of an FDK initialization. Higher strength → more aggressive smoothing.
"""

# ╔═╡ f0000002-0003-0001-0001-000000000001
# HIR reconstruction of Low Dose (80 kVp) data at three strength levels
hir_results = let
	ld = sim_results["Low Dose"]
	sino_gpu = Metal.MtlArray(ld.sinogram)
	geom = ld.geom
	recon_size = recon_opts.matrix_size
	μ_ref = ld.mu_calib

	results = Dict{String, Array{Float32, 3}}()

	# FDK baseline (already computed, just copy)
	results["FDK"] = ld.recon

	# HIR Strength 1 (mild)
	@info "HIR Strength 1..."
	ws_hir1 = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=1)
	vol_hir1 = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir1, sino_gpu, geom, recon_size));
		μ_water=μ_ref
	)
	results["HIR-1"] = vol_hir1
	ws_hir1 = nothing; vol_hir1 = nothing; GC.gc(true)

	# HIR Strength 3 (moderate)
	@info "HIR Strength 3..."
	ws_hir3 = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=3)
	vol_hir3 = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir3, sino_gpu, geom, recon_size));
		μ_water=μ_ref
	)
	results["HIR-3"] = vol_hir3
	ws_hir3 = nothing; vol_hir3 = nothing; GC.gc(true)

	# HIR Strength 5 (aggressive)
	@info "HIR Strength 5..."
	ws_hir5 = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=5)
	vol_hir5 = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir5, sino_gpu, geom, recon_size));
		μ_water=μ_ref
	)
	results["HIR-5"] = vol_hir5
	ws_hir5 = nothing; vol_hir5 = nothing; GC.gc(true)

	sino_gpu = nothing; GC.gc(true)
	results
end

# ╔═╡ 928b2d90-d554-4eac-a045-82336de4b323
md"""
### Visual Comparison: Low Dose (80 kVp / 50 mA)

Comparing FDK vs HIR at strength 1, 3, and 5.
"""

# ╔═╡ afd76526-df37-4fa2-8a62-8032a8641225
@bind slice_idx UI.Slider(1:SIM_CONFIG.sliceCount, default=SIM_CONFIG.sliceCount ÷ 2, show_value=true)

# ╔═╡ 94441708-b5c3-4555-a8f1-7b40fcdeaf9f
let
	methods = ["FDK", "HIR-1", "HIR-3", "HIR-5"]
	n_methods = length(methods)

	fig = CM.Figure(size=(300 * n_methods + 100, 350))
	vmin, vmax = -200, 400

	for (i, m) in enumerate(methods)
		ax = CM.Axis(fig[1,i], title=m, aspect=CM.DataAspect())
		vol = hir_results[m]
		hm = CM.heatmap!(ax, vol[:,:,slice_idx], colormap=:grays, colorrange=(vmin, vmax))
		CM.hidedecorations!(ax)
		if i == n_methods
			 CM.Colorbar(fig[1, i+1], hm, label="HU")
		end
	end
	CM.Label(fig[0, :], text="Low Dose (80 kVp / 50 mA)", fontsize=20, font=:bold)
	CM.save(joinpath(FIGURES_DIR, "nb02_hir_comparison.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ f0000002-0003-0001-0001-000000000002
md"""
### Noise & CNR Analysis

We measure water noise (σ) and insert CNR for each reconstruction method.
"""

# ╔═╡ 348a1367-16fe-41ee-9374-47f80a1c0513
let
	cx, cy, cz = SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.sliceCount ÷ 2
	pix = SIM_CONFIG.fov_mm / SIM_CONFIG.imageSize
	r_pix = 105.0 / pix
	ang = 2π * 6 / 7 + π/7
	ix, iy = cx + round(Int, r_pix * cos(ang)), cy + round(Int, r_pix * sin(ang))
	r = 5

	methods = ["FDK", "HIR-1", "HIR-3", "HIR-5"]
	n_m = length(methods)
	cnr_vals = Float64[]
	noise_vals = Float64[]

	for m in methods
		vol = hir_results[m]
		S = mean(vol[ix-r:ix+r, iy-r:iy+r, cz])
		B = mean(vol[cx-20:cx+20, cy-20:cy+20, cz])
		sigma = std(vol[cx-20:cx+20, cy-20:cy+20, cz])
		push!(cnr_vals, sigma == 0 ? 0.0 : abs(S - B) / sigma)
		push!(noise_vals, sigma)
	end

	fig = CM.Figure(size=(1000, 400))

	# CNR comparison
	ax1 = CM.Axis(fig[1,1], title="CNR (Iodine 20mg/mL)", ylabel="CNR")
	CM.barplot!(ax1, 1:n_m, cnr_vals, color=[:gray, :steelblue, :purple, :green])
	ax1.xticks = (1:n_m, methods)
	for (i, v) in enumerate(cnr_vals)
		CM.text!(ax1, i, v + 0.5, text=string(round(v, digits=1)), align=(:center, :bottom))
	end
	CM.ylims!(ax1, 0, maximum(cnr_vals) * 1.15)

	# Noise comparison
	ax2 = CM.Axis(fig[1,2], title="Water Background Noise", ylabel="σ (HU)")
	CM.barplot!(ax2, 1:n_m, noise_vals, color=[:gray, :steelblue, :purple, :green])
	ax2.xticks = (1:n_m, methods)
	for (i, v) in enumerate(noise_vals)
		CM.text!(ax2, i, v + 0.5, text=string(round(v, digits=1)), align=(:center, :bottom))
	end
	CM.ylims!(ax2, 0, maximum(noise_vals) * 1.15)

	CM.save(joinpath(FIGURES_DIR, "nb02_hir_cnr_noise.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ f0000002-0003-0001-0001-000000000003
md"""
### HIR on Standard Dose (120 kVp / 200 mA)

We also demonstrate HIR on the standard-dose data for comparison.
"""

# ╔═╡ f0000002-0003-0001-0001-000000000004
# HIR reconstruction of Standard (120 kVp) data
hir_results_std = let
	std_data = sim_results["Standard"]
	sino_gpu = Metal.MtlArray(std_data.sinogram)
	geom = std_data.geom
	recon_size = recon_opts.matrix_size
	μ_ref = std_data.mu_calib

	results = Dict{String, Array{Float32, 3}}()
	results["FDK"] = std_data.recon

	# HIR Strength 1
	@info "Standard: HIR Strength 1..."
	ws1 = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=1)
	v1 = BS.to_hounsfield(Array(BS.reconstruct!(ws1, sino_gpu, geom, recon_size)); μ_water=μ_ref)
	results["HIR-1"] = v1
	ws1 = nothing; v1 = nothing; GC.gc(true)

	# HIR Strength 3
	@info "Standard: HIR Strength 3..."
	ws3 = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=3)
	v3 = BS.to_hounsfield(Array(BS.reconstruct!(ws3, sino_gpu, geom, recon_size)); μ_water=μ_ref)
	results["HIR-3"] = v3
	ws3 = nothing; v3 = nothing; GC.gc(true)

	# HIR Strength 5
	@info "Standard: HIR Strength 5..."
	ws5 = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength=5)
	v5 = BS.to_hounsfield(Array(BS.reconstruct!(ws5, sino_gpu, geom, recon_size)); μ_water=μ_ref)
	results["HIR-5"] = v5
	ws5 = nothing; v5 = nothing; GC.gc(true)

	sino_gpu = nothing; GC.gc(true)
	results
end

# ╔═╡ f0000002-0003-0001-0001-000000000005
@bind slice_idx_std UI.Slider(1:SIM_CONFIG.sliceCount, default=SIM_CONFIG.sliceCount ÷ 2, show_value=true)

# ╔═╡ f0000002-0003-0001-0001-000000000006
let
	methods = ["FDK", "HIR-1", "HIR-3", "HIR-5"]
	n_methods = length(methods)

	fig = CM.Figure(size=(300 * n_methods + 100, 350))
	vmin, vmax = -200, 400

	for (i, m) in enumerate(methods)
		ax = CM.Axis(fig[1,i], title=m, aspect=CM.DataAspect())
		vol = hir_results_std[m]
		hm = CM.heatmap!(ax, vol[:,:,slice_idx_std], colormap=:grays, colorrange=(vmin, vmax))
		CM.hidedecorations!(ax)
		if i == n_methods
			 CM.Colorbar(fig[1, i+1], hm, label="HU")
		end
	end
	CM.Label(fig[0, :], text="Standard Dose (120 kVp / 200 mA)", fontsize=20, font=:bold)
	CM.save(joinpath(FIGURES_DIR, "nb02_hir_comparison_std.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ f0000002-0003-0001-0001-000000000007
let
	cx, cy, cz = SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.sliceCount ÷ 2
	pix = SIM_CONFIG.fov_mm / SIM_CONFIG.imageSize
	r_pix = 105.0 / pix
	ang = 2π * 6 / 7 + π/7
	ix, iy = cx + round(Int, r_pix * cos(ang)), cy + round(Int, r_pix * sin(ang))
	r = 5

	methods = ["FDK", "HIR-1", "HIR-3", "HIR-5"]
	n_m = length(methods)
	cnr_vals = Float64[]
	noise_vals = Float64[]

	for m in methods
		vol = hir_results_std[m]
		S = mean(vol[ix-r:ix+r, iy-r:iy+r, cz])
		B = mean(vol[cx-20:cx+20, cy-20:cy+20, cz])
		sigma = std(vol[cx-20:cx+20, cy-20:cy+20, cz])
		push!(cnr_vals, sigma == 0 ? 0.0 : abs(S - B) / sigma)
		push!(noise_vals, sigma)
	end

	fig = CM.Figure(size=(1000, 400))

	ax1 = CM.Axis(fig[1,1], title="CNR (Iodine 20mg/mL) — Standard Dose", ylabel="CNR")
	CM.barplot!(ax1, 1:n_m, cnr_vals, color=[:gray, :steelblue, :purple, :green])
	ax1.xticks = (1:n_m, methods)
	for (i, v) in enumerate(cnr_vals)
		CM.text!(ax1, i, v + 0.5, text=string(round(v, digits=1)), align=(:center, :bottom))
	end
	CM.ylims!(ax1, 0, maximum(cnr_vals) * 1.15)

	ax2 = CM.Axis(fig[1,2], title="Water Background Noise — Standard Dose", ylabel="σ (HU)")
	CM.barplot!(ax2, 1:n_m, noise_vals, color=[:gray, :steelblue, :purple, :green])
	ax2.xticks = (1:n_m, methods)
	for (i, v) in enumerate(noise_vals)
		CM.text!(ax2, i, v + 0.5, text=string(round(v, digits=1)), align=(:center, :bottom))
	end
	CM.ylims!(ax2, 0, maximum(noise_vals) * 1.15)

	CM.save(joinpath(FIGURES_DIR, "nb02_hir_cnr_noise_std.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ Cell order:
# ╟─688cbbde-bdd5-462f-87a3-05c9d26242b5
# ╠═b07a644b-594b-42a2-bab8-37a5eab2a8c4
# ╠═e69a816d-43f8-48de-8645-7fd16168a928
# ╠═0735094c-9b93-4fb7-ba47-72a499ffadff
# ╠═dbdff1fe-ce08-427b-b271-c0dad574c1be
# ╠═26aaea60-657b-4d2c-a26e-2312fa56a76e
# ╠═f0000002-0001-0001-0001-000000000010
# ╠═f0000002-0001-0001-0001-000000000011
# ╠═c2c038a2-675a-40ca-8809-56d19d78600c
# ╠═f0000002-0001-0001-0001-000000000001
# ╠═0edf6fa8-a8d8-4aa1-b2b3-8f8abb1773b4
# ╟─6d84c6c6-863f-42e6-8ae7-ef00154a9507
# ╠═ea44a25a-5358-4aef-b301-d462798141c8
# ╠═62222977-9635-434c-b3dd-21720c19402c
# ╠═3a07da3e-98d4-41f6-b295-cfdb9fbdd19d
# ╟─dda5d6eb-0194-4956-99b7-1798cabc497f
# ╟─7e16e89f-f215-41f9-a90c-2541e37c8a7a
# ╠═f0000002-0002-0001-0001-000000000020
# ╠═f0000002-0002-0001-0001-000000000021
# ╠═f0000002-0002-0001-0001-000000000022
# ╠═d7c40e60-b546-41cc-b26c-5363abb24e51
# ╟─7d202fe4-6a60-4ee8-aa49-52843e524c53
# ╠═d190489f-d0ec-4428-81a6-6bff43be401c
# ╠═286bb6a0-ce50-4370-a4f0-1f5c8f62f60c
# ╠═ac0e51fd-a1d8-4521-8bfd-7f105f637d16
# ╟─022bcd91-ab8a-404b-a4fe-7a0f213fd9a9
# ╠═e8c85147-2cc2-4530-922a-0898ad9fdf6a
# ╟─c717df6c-f356-43c9-98b1-f0c926eb9384
# ╟─f889723d-a117-47a9-a120-43f16e827cd8
# ╟─67d7edad-5929-43dd-8681-0b079c08753d
# ╟─59c9285a-bb87-4ecc-babd-9f5943a8cc53
# ╠═f0000002-0003-0001-0001-000000000001
# ╟─928b2d90-d554-4eac-a045-82336de4b323
# ╟─afd76526-df37-4fa2-8a62-8032a8641225
# ╟─94441708-b5c3-4555-a8f1-7b40fcdeaf9f
# ╟─f0000002-0003-0001-0001-000000000002
# ╟─348a1367-16fe-41ee-9374-47f80a1c0513
# ╟─f0000002-0003-0001-0001-000000000003
# ╠═f0000002-0003-0001-0001-000000000004
# ╟─f0000002-0003-0001-0001-000000000005
# ╟─f0000002-0003-0001-0001-000000000006
# ╟─f0000002-0003-0001-0001-000000000007
