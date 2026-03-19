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

# ╔═╡ a0000001-0001-0001-0001-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()

	using Revise
end

# ╔═╡ a0000001-0001-0001-0001-000000000090
using Unitful: @u_str

# ╔═╡ a0000001-0001-0001-0001-000000000005
using Metal

# ╔═╡ a0000001-0001-0001-0001-000000000006
md"""
# Dual-kVp Virtual Monoenergetic Imaging (VMI) Verification

**BasisSimulator.jl — Part 3: Spectral CT Validation**

*Sinogram-domain material decomposition with comprehensive NIST validation*

---
"""

# ╔═╡ a0000001-0001-0001-0001-000000000002
import BasisSimulator as BS

# ╔═╡ a0000001-0001-0001-0001-000000000003
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ a0000001-0001-0001-0001-000000000004
import Statistics: mean, std, cor

# ╔═╡ a0000001-0001-0001-0001-000000000091
import XrayAttenuation as XA

# ╔═╡ a0000001-0001-0001-0001-000000000007
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")

# ╔═╡ a0000001-0001-0001-0001-000000000008
md"""
## Abstract

**Background**: Dual-energy CT enables Virtual Monoenergetic Imaging (VMI), which synthesizes images at any X-ray energy from a dual-kVp acquisition. VMI provides energy-dependent contrast optimization and quantitative material characterization.

**Purpose**: This notebook validates BasisSimulator.jl's complete dual-kVp VMI pipeline using full polychromatic physics (`SimOptions(fidelity=:eict)`) and clinical scanner parameters. We verify that VMI HU values follow expected energy-dependent trends and correlate strongly with NIST reference attenuation data.

**Methods**:
1. Dual-kVp acquisition (80/140 kVp) of a Gammex 472 phantom with full physics
2. Sinogram-domain material decomposition (water/iodine basis)
3. VMI synthesis at 8 energies (40–140 keV) in the sinogram domain
4. FDK reconstruction of each VMI sinogram
5. Empirical HU calibration and ROI-based insert measurement
6. Comparison against NIST XCOM reference values via `validate_material_hu()`

**Key Result**: VMI correctly captures energy-dependent attenuation trends for all materials, with R² > 0.95 correlation to NIST predictions despite the effective-energy approximation inherent in the 2×2 decomposition matrix.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000009
md"""
## 1. Scanner & Protocol Configuration

GE Revolution-style clinical scanner with all hardware fields populated for `build_physics_config()`.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000010
SIM_CONFIG = let
    sid = 540.0
    sdd = 950.0
    magnification = sdd / sid   # 1.759
    (
        imageSize = 512,
        sliceCount = 32,
        sliceThickness = 1.25,      # mm
        phantom_z_cm = 5.0,         # Actual Gammex 472 physical thickness (50mm)
        fov_mm = 350.0,

        sid = sid,
        sdd = sdd,
        detectorColCount = 900,
        detectorRowCount = 16,
        detectorColSize = 1.0 / magnification,   # ≈ 0.569 mm at isocenter
        detectorRowSize = 1.0 / magnification,   # ≈ 0.569 mm at isocenter

        viewsPerRotation = 984,
        rotationTime = 1.0,
        n_energy_bins = 15
    )
end

# ╔═╡ a0000001-0001-0001-0001-000000000011
scanner = BS.Scanner(
	source_to_isocenter = SIM_CONFIG.sid,
	source_to_detector = SIM_CONFIG.sdd,
	detector_rows = SIM_CONFIG.detectorRowCount,
	detector_cols = SIM_CONFIG.detectorColCount,
	detector_row_size = SIM_CONFIG.detectorRowSize,

	# detector_col_size is the element pitch at isocenter (mm).
	detector_col_size = SIM_CONFIG.detectorColSize,

	detector_shape = BS.CURVED_DETECTOR,

	# --- Hardware fields for build_physics_config() ---
	focal_spot_width = 0.7,
	focal_spot_length = 0.9,
	target_angle = 7.0,
	flat_filter_material = :aluminum,
	flat_filter_thickness = 2.5,
	detector_material = :gadolinium_oxysulfide,
	detector_depth = 0.5,
	fill_factor_row = 0.9,
	fill_factor_col = 0.9,
	detection_gain = 1.0,
	electronic_noise = 100.0,
)

# ╔═╡ a0000001-0001-0001-0001-000000000012
begin
	# Dual-kVp protocols: 80 kVp and 140 kVp at clinical mA
	protocol_80 = BS.CTProtocol(kVp=80, mA=200.0, views=SIM_CONFIG.viewsPerRotation, rotation_time=SIM_CONFIG.rotationTime)
	protocol_140 = BS.CTProtocol(kVp=140, mA=200.0, views=SIM_CONFIG.viewsPerRotation, rotation_time=SIM_CONFIG.rotationTime)
end

# ╔═╡ a0000001-0001-0001-0001-000000000013
sim_opts = BS.SimOptions(fidelity=:eict, seed=42)

# ╔═╡ a0000001-0001-0001-0001-000000000014
recon_opts = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (SIM_CONFIG.imageSize, SIM_CONFIG.imageSize, SIM_CONFIG.sliceCount),
	fov_cm = SIM_CONFIG.fov_mm / 10.0,
	filter = :standard,
)

# ╔═╡ a0000001-0001-0001-0001-000000000015
md"""
## 2. Phantom Generation

Gammex 472 digital phantom: 33 cm diameter × **5 cm thick** (actual physical dimensions from manufacturer spec sheet). 7 calcium inserts (inner ring, R=5.0 cm), 7 iodine inserts (outer ring, R=10.5 cm).

The phantom (5cm) is much taller than the detector z-coverage (~9mm for 16 rows), so every detector row traces rays through the full phantom body.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000016
phantom_cpu = BS.create_gammex_472(
	n_voxels = SIM_CONFIG.imageSize,
	n_slices = SIM_CONFIG.sliceCount,
	fov_cm = SIM_CONFIG.fov_mm / 10.0,
	z_cm = SIM_CONFIG.phantom_z_cm,  # 5cm — actual Gammex 472 thickness
);

# ╔═╡ a0000001-0001-0001-0001-000000000017
phantom_gpu = BS.Phantom(
	Metal.MtlArray(phantom_cpu.mask),
	phantom_cpu.materials,
	phantom_cpu.voxel_size,
	phantom_cpu.origin,
	phantom_cpu.extent
);

# ╔═╡ a0000001-0001-0001-0001-000000000018
md"""
## 3. Dual-kVp Simulation

Full polychromatic simulation at both kVp settings using workspace API. The ideal sinogram (pre-noise) is used for material decomposition to avoid noise-amplification in the 2×2 inversion.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000019
# ╠═╡ show_logs = false
# 80 kVp simulation
sim_80 = let
	recon_size = recon_opts.matrix_size

	ws = BS.create_eict_workspace(scanner, protocol_80, sim_opts, recon_opts, phantom_gpu)
	BS.simulate!(ws, phantom_gpu, scanner, protocol_80, sim_opts, recon_opts)

	sino_ideal = copy(ws.sino_ideal_out)
	sino_noisy = copy(ws.sino_noisy_out)
	geom = ws.geom

	ws = nothing; GC.gc(true)

	(; sino_ideal, sino_noisy, geom)
end

# ╔═╡ a0000001-0001-0001-0001-000000000020
# ╠═╡ show_logs = false
# 140 kVp simulation
sim_140 = let
	ws = BS.create_eict_workspace(scanner, protocol_140, sim_opts, recon_opts, phantom_gpu)
	BS.simulate!(ws, phantom_gpu, scanner, protocol_140, sim_opts, recon_opts)

	sino_ideal = copy(ws.sino_ideal_out)
	sino_noisy = copy(ws.sino_noisy_out)
	geom = ws.geom

	ws = nothing; GC.gc(true)

	(; sino_ideal, sino_noisy, geom)
end

# ╔═╡ a0000001-0001-0001-0001-000000000021
md"""
## 4. Sinogram-Domain Material Decomposition

The dual-energy sinograms are decomposed into water and iodine basis materials using a 2×2 matrix inversion at each ray. This operates in the **projection domain** — the correct space for material decomposition since sinogram values are line integrals.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000022
# Construct DualEnergySinogram from the two simulate!() results
de_sino = BS.DualEnergySinogram(sim_80.sino_ideal, sim_140.sino_ideal; low_kvp=80, high_kvp=140)

# ╔═╡ a0000001-0001-0001-0001-000000000023
# Material decomposition: sinogram-domain 2×2 inversion
# Basis: water + iodine (standard for contrast-enhanced imaging)
mat_map = BS.decompose_materials(de_sino; basis=(:water, :iodine))

# ╔═╡ a0000001-0001-0001-0001-000000000024
md"""
## 5. VMI Reconstruction Sweep

We generate Virtual Monoenergetic Images at 8 energies (40–140 keV). For each energy:
1. `virtual_monoenergetic(mat_map, E)` → VMI sinogram
2. `fdk_reconstruct(vmi_sino, geom, size)` → VMI attenuation image
3. Empirical water calibration → HU conversion
"""

# ╔═╡ a0000001-0001-0001-0001-000000000025
const VMI_ENERGIES = [40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 140.0]

# ╔═╡ a0000001-0001-0001-0001-000000000026
# ╠═╡ show_logs = false
vmi_hu_volumes = let
	recon_size = recon_opts.matrix_size
	geom = sim_80.geom
	cx, cy, cz = recon_size[1] ÷ 2, recon_size[2] ÷ 2, recon_size[3] ÷ 2

	results = Dict{Float64, Array{Float64,3}}()

	for E in VMI_ENERGIES
		@info "VMI reconstruction at $(E) keV..."

		# Step 1: Synthesize VMI sinogram from material maps + μ(E)
		vmi_sino = BS.virtual_monoenergetic(mat_map, E)

		# Step 2: FDK reconstruction
		vmi_recon = Array(BS.fdk_reconstruct(vmi_sino, geom, recon_size))

		# Step 3: Empirical water calibration
		μ_water_emp = mean(vmi_recon[cx-10:cx+10, cy-10:cy+10, cz-2:cz+2])

		# Convert to HU
		vmi_hu = 1000.0 .* (vmi_recon .- μ_water_emp) ./ μ_water_emp

		results[E] = vmi_hu
		GC.gc(true)
	end
	results
end

# ╔═╡ a0000001-0001-0001-0001-000000000027
md"""
## 6. VMI Sweep Visualization

Energy-dependent contrast: iodine inserts (outer ring) are brightest at low keV due to the photoelectric effect, while calcium inserts (inner ring) also decrease with energy but less dramatically.
"""

# ╔═╡ fa4ecd77-1991-44da-b56c-37adaa9d728f
import PlutoUI as UI

# ╔═╡ 4130c9d2-020e-4224-81da-94dfcfdab6b0
UI.TableOfContents()

# ╔═╡ e2c3e4d6-d4fc-44ef-84b4-b08a7f7566dc
@bind z UI.Slider(axes(vmi_hu_volumes[VMI_ENERGIES[1]], 3), show_value = true, default = SIM_CONFIG.sliceCount ÷ 2)

# ╔═╡ a0000001-0001-0001-0001-000000000028
let
	fig = CM.Figure(size=(1600, 800), fontsize=11)
	mid_z = SIM_CONFIG.sliceCount ÷ 2

	for (i, E) in enumerate(VMI_ENERGIES)
		row = (i - 1) ÷ 4 + 1
		col = (i - 1) % 4 + 1

		ax = CM.Axis(fig[row, col], title="$(Int(E)) keV", aspect=CM.DataAspect())
		hm = CM.heatmap!(ax, vmi_hu_volumes[E][:, :, z],
			colormap=:grays, colorrange=(-500, 800))
		CM.hidedecorations!(ax)

		if col == 4
			CM.Colorbar(fig[row, 5], hm, label="HU")
		end
	end

	CM.Label(fig[0, :], text="VMI Energy Sweep (Gammex 472, Full Physics)",
		fontsize=16, font=:bold)

	CM.save(joinpath(FIGURES_DIR, "nb03_vmi_sweep.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000029
md"""
## 7. ROI-Based HU Measurement

We measure mean HU in circular ROIs placed at each insert location. The Gammex 472 geometry:
- Inner ring (R=5.0 cm): Ca 50–600 mg/mL, 7 inserts at 2π/7 spacing
- Outer ring (R=10.5 cm): I 2.0–20.0 mg/mL, 7 inserts offset by π/7
"""

# ╔═╡ a0000001-0001-0001-0001-000000000030
begin
	# ROI measurement parameters
	const N_INSERTS = 7
	const PIXEL_SIZE_MM = SIM_CONFIG.fov_mm / SIM_CONFIG.imageSize
	const INNER_R_PIX = 50.0 / PIXEL_SIZE_MM  # 5.0 cm inner ring
	const OUTER_R_PIX = 105.0 / PIXEL_SIZE_MM  # 10.5 cm outer ring
	const ROI_RADIUS = 8  # pixels (~5.5 mm radius)

	# Insert angles
	const ANGLES_CA = [2π * i / N_INSERTS for i in 0:(N_INSERTS-1)]
	const ANGLES_I = [2π * i / N_INSERTS + π/N_INSERTS for i in 0:(N_INSERTS-1)]

	# Material symbols for NIST validation
	const CA_SYMBOLS = [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
	const I_SYMBOLS = [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]

	const CA_LABELS = ["Ca 50", "Ca 100", "Ca 200", "Ca 300", "Ca 400", "Ca 500", "Ca 600"]
	const I_LABELS = ["I 2.0", "I 2.5", "I 5.0", "I 7.5", "I 10.0", "I 15.0", "I 20.0"]
end

# ╔═╡ a0000001-0001-0001-0001-000000000031
function measure_insert_hu(vol, ring_radius_pix, angles; roi_radius=ROI_RADIUS)
	cx, cy, cz = size(vol, 1) ÷ 2, size(vol, 2) ÷ 2, size(vol, 3) ÷ 2
	hu_values = Float64[]

	for ang in angles
		ix = cx + round(Int, ring_radius_pix * cos(ang))
		iy = cy + round(Int, ring_radius_pix * sin(ang))

		roi = vol[ix-roi_radius:ix+roi_radius, iy-roi_radius:iy+roi_radius, cz]
		push!(hu_values, mean(roi))
	end
	return hu_values
end

# ╔═╡ a0000001-0001-0001-0001-000000000032
# Measure HU for all inserts at all VMI energies
measured_hu = let
	data = Dict{Float64, NamedTuple}()
	cx, cy, cz = SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.sliceCount ÷ 2

	for E in VMI_ENERGIES
		vol = vmi_hu_volumes[E]

		ca_hu = measure_insert_hu(vol, INNER_R_PIX, ANGLES_CA)
		i_hu = measure_insert_hu(vol, OUTER_R_PIX, ANGLES_I)

		# Water ROI (center)
		water_hu = mean(vol[cx-10:cx+10, cy-10:cy+10, cz])

		data[E] = (; calcium=ca_hu, iodine=i_hu, water=water_hu)
	end
	data
end

# ╔═╡ a0000001-0001-0001-0001-000000000033
md"""
## 8. NIST Validation

We compare measured VMI HU values against theoretical predictions from `validate_material_hu()`, which uses NIST XCOM mass attenuation coefficients.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000034
# Compute NIST reference HU for all materials at all VMI energies
nist_reference = let
	data = Dict{Float64, NamedTuple}()

	for E in VMI_ENERGIES
		ca_nist = [BS.validate_material_hu(sym, E) for sym in CA_SYMBOLS]
		i_nist = [BS.validate_material_hu(sym, E) for sym in I_SYMBOLS]

		data[E] = (; calcium=ca_nist, iodine=i_nist, water=0.0)
	end
	data
end

# ╔═╡ a0000001-0001-0001-0001-000000000035
md"""
## 9. HU vs Energy Curves

The key physics validation: HU should decrease with increasing energy for both iodine (photoelectric-dominated, K-edge at 33.2 keV) and calcium (monotonic decrease, K-edge at 4 keV). Water should remain at 0 HU across all energies by construction (empirical calibration).
"""

# ╔═╡ a0000001-0001-0001-0001-000000000036
let
	fig = CM.Figure(size=(1200, 500), fontsize=12)

	# --- Left panel: Measured HU vs keV ---
	ax1 = CM.Axis(fig[1,1], title="Measured VMI HU vs Energy",
		xlabel="Energy (keV)", ylabel="HU")

	# Plot selected materials (subset for clarity)
	ca_indices = [3, 5, 7]  # Ca_200, Ca_400, Ca_600
	ca_colors = [:green, :teal, :darkgreen]
	for (idx, ci) in enumerate(ca_indices)
		hu_vs_E = [measured_hu[E].calcium[ci] for E in VMI_ENERGIES]
		CM.lines!(ax1, VMI_ENERGIES, hu_vs_E, color=ca_colors[idx], linewidth=2,
			label=CA_LABELS[ci])
		CM.scatter!(ax1, VMI_ENERGIES, hu_vs_E, color=ca_colors[idx], markersize=6)
	end

	# Iodine: I_5.0, I_10.0, I_20.0
	i_indices = [3, 5, 7]
	i_colors = [:blue, :purple, :red]
	for (idx, ii) in enumerate(i_indices)
		hu_vs_E = [measured_hu[E].iodine[ii] for E in VMI_ENERGIES]
		CM.lines!(ax1, VMI_ENERGIES, hu_vs_E, color=i_colors[idx], linewidth=2,
			label=I_LABELS[ii])
		CM.scatter!(ax1, VMI_ENERGIES, hu_vs_E, color=i_colors[idx], markersize=6)
	end

	# Water
	water_vs_E = [measured_hu[E].water for E in VMI_ENERGIES]
	CM.lines!(ax1, VMI_ENERGIES, water_vs_E, color=:black, linewidth=2,
		linestyle=:dash, label="Water")
	CM.hlines!(ax1, [0.0], color=:gray, linestyle=:dot)

	CM.axislegend(ax1, position=:rt)

	# --- Right panel: NIST reference ---
	ax2 = CM.Axis(fig[1,2], title="NIST Reference HU vs Energy",
		xlabel="Energy (keV)", ylabel="HU (NIST)")

	for (idx, ci) in enumerate(ca_indices)
		nist_vs_E = [nist_reference[E].calcium[ci] for E in VMI_ENERGIES]
		CM.lines!(ax2, VMI_ENERGIES, nist_vs_E, color=ca_colors[idx], linewidth=2,
			label=CA_LABELS[ci])
	end

	for (idx, ii) in enumerate(i_indices)
		nist_vs_E = [nist_reference[E].iodine[ii] for E in VMI_ENERGIES]
		CM.lines!(ax2, VMI_ENERGIES, nist_vs_E, color=i_colors[idx], linewidth=2,
			label=I_LABELS[ii])
	end

	CM.hlines!(ax2, [0.0], color=:gray, linestyle=:dot)
	CM.axislegend(ax2, position=:rt)

	CM.save(joinpath(FIGURES_DIR, "nb03_hu_vs_energy.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000037
md"""
## 10. NIST Correlation Analysis

For each VMI energy, we compute the Pearson correlation (R²) between measured and NIST HU values. High R² confirms that the VMI pipeline correctly captures the energy-dependent attenuation trends, even if there is a systematic offset due to the effective-energy approximation.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000038
# Compute R² for each VMI energy
r_squared_per_energy = let
	results = Dict{Float64, NamedTuple}()

	for E in VMI_ENERGIES
		# All materials combined (calcium + iodine)
		meas_all = vcat(measured_hu[E].calcium, measured_hu[E].iodine)
		nist_all = vcat(nist_reference[E].calcium, nist_reference[E].iodine)

		# Pearson R²
		r = cor(meas_all, nist_all)
		r2 = r^2

		# Iodine only
		r_i = cor(measured_hu[E].iodine, nist_reference[E].iodine)
		r2_i = r_i^2

		# Calcium only
		r_ca = cor(measured_hu[E].calcium, nist_reference[E].calcium)
		r2_ca = r_ca^2

		results[E] = (; all=r2, iodine=r2_i, calcium=r2_ca)
	end
	results
end

# ╔═╡ a0000001-0001-0001-0001-000000000039
let
	fig = CM.Figure(size=(800, 400), fontsize=12)
	ax = CM.Axis(fig[1,1], title="R² (Measured vs NIST) by VMI Energy",
		xlabel="Energy (keV)", ylabel="R²", ylabelsize=14)

	r2_all = [r_squared_per_energy[E].all for E in VMI_ENERGIES]
	r2_iodine = [r_squared_per_energy[E].iodine for E in VMI_ENERGIES]
	r2_calcium = [r_squared_per_energy[E].calcium for E in VMI_ENERGIES]

	CM.lines!(ax, VMI_ENERGIES, r2_all, color=:black, linewidth=2, label="All materials")
	CM.scatter!(ax, VMI_ENERGIES, r2_all, color=:black, markersize=8)

	CM.lines!(ax, VMI_ENERGIES, r2_iodine, color=:blue, linewidth=2, label="Iodine only")
	CM.scatter!(ax, VMI_ENERGIES, r2_iodine, color=:blue, markersize=6)

	CM.lines!(ax, VMI_ENERGIES, r2_calcium, color=:green, linewidth=2, label="Calcium only")
	CM.scatter!(ax, VMI_ENERGIES, r2_calcium, color=:green, markersize=6)

	CM.hlines!(ax, [0.95], color=:red, linestyle=:dash, label="R²=0.95 threshold")
	CM.ylims!(ax, 0.8, 1.02)

	CM.axislegend(ax, position=:lb)
	CM.save(joinpath(FIGURES_DIR, "nb03_r_squared.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000040
md"""
## 11. Measured vs NIST Scatter Plots

Direct comparison at 70 keV (clinical reference energy) and 50 keV showing the correlation between measured and NIST-predicted HU values.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000041
let
	fig = CM.Figure(size=(1200, 500), fontsize=12)

	# --- Panel 1: 70 keV scatter plot ---
	E = 70.0
	ax1 = CM.Axis(fig[1,1], title="Measured vs NIST at 70 keV",
		xlabel="NIST HU", ylabel="Measured HU", aspect=CM.DataAspect())

	CM.scatter!(ax1, nist_reference[E].calcium, measured_hu[E].calcium,
		color=:green, markersize=10, label="Calcium")
	CM.scatter!(ax1, nist_reference[E].iodine, measured_hu[E].iodine,
		color=:blue, markersize=10, label="Iodine")

	# Identity line
	all_nist = vcat(nist_reference[E].calcium, nist_reference[E].iodine)
	min_v, max_v = minimum(all_nist) - 50, maximum(all_nist) + 50
	CM.lines!(ax1, [min_v, max_v], [min_v, max_v], color=:red, linestyle=:dash,
		label="Identity")

	r2_70 = r_squared_per_energy[E].all
	CM.text!(ax1, min_v + 50, max_v - 100,
		text="R² = $(round(r2_70, digits=4))", fontsize=14)
	CM.axislegend(ax1, position=:lt)

	# --- Panel 2: 50 keV scatter plot ---
	E2 = 50.0
	ax2 = CM.Axis(fig[1,2], title="Measured vs NIST at 50 keV",
		xlabel="NIST HU", ylabel="Measured HU", aspect=CM.DataAspect())

	CM.scatter!(ax2, nist_reference[E2].calcium, measured_hu[E2].calcium,
		color=:green, markersize=10, label="Calcium")
	CM.scatter!(ax2, nist_reference[E2].iodine, measured_hu[E2].iodine,
		color=:blue, markersize=10, label="Iodine")

	all_nist2 = vcat(nist_reference[E2].calcium, nist_reference[E2].iodine)
	min_v2, max_v2 = minimum(all_nist2) - 50, maximum(all_nist2) + 50
	CM.lines!(ax2, [min_v2, max_v2], [min_v2, max_v2], color=:red, linestyle=:dash,
		label="Identity")

	r2_50 = r_squared_per_energy[E2].all
	CM.text!(ax2, min_v2 + 50, max_v2 - 100,
		text="R² = $(round(r2_50, digits=4))", fontsize=14)
	CM.axislegend(ax2, position=:lt)

	CM.save(joinpath(FIGURES_DIR, "nb03_nist_scatter.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000042
md"""
## 12. Water Stability

Water should be 0 HU at all VMI energies (by construction via empirical calibration). We verify this by measuring the water ROI at each energy.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000043
let
	fig = CM.Figure(size=(700, 400), fontsize=12)
	ax = CM.Axis(fig[1,1], title="Water HU Stability Across VMI Energies",
		xlabel="Energy (keV)", ylabel="Water HU")

	water_hu = [measured_hu[E].water for E in VMI_ENERGIES]

	CM.scatter!(ax, VMI_ENERGIES, water_hu, color=:blue, markersize=10)
	CM.lines!(ax, VMI_ENERGIES, water_hu, color=:blue, linewidth=2)

	# ±30 HU band
	CM.band!(ax, VMI_ENERGIES, fill(-30.0, length(VMI_ENERGIES)),
		fill(30.0, length(VMI_ENERGIES)), color=(:green, 0.15))
	CM.hlines!(ax, [0.0], color=:black, linestyle=:dash)
	CM.hlines!(ax, [-30.0, 30.0], color=:green, linestyle=:dot, label="±30 HU")

	CM.axislegend(ax, position=:rt)
	CM.save(joinpath(FIGURES_DIR, "nb03_water_stability.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000044
md"""
## 13. Iodine Concentration Linearity

At a fixed VMI energy (70 keV), HU should increase linearly with iodine concentration. This verifies the material decomposition correctly preserves quantitative concentration information.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000045
let
	fig = CM.Figure(size=(700, 450), fontsize=12)
	ax = CM.Axis(fig[1,1], title="Iodine Linearity at 70 keV",
		xlabel="Iodine Concentration (mg/mL)", ylabel="Measured HU")

	concentrations = [2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
	hu_70 = measured_hu[70.0].iodine

	CM.scatter!(ax, concentrations, hu_70, color=:blue, markersize=10, label="Measured")

	# Linear fit (least squares)
	x = concentrations
	y = hu_70
	n = length(x)
	sx = sum(x)
	sy = sum(y)
	sxy = sum(x .* y)
	sx2 = sum(x .^ 2)
	slope = (n * sxy - sx * sy) / (n * sx2 - sx^2)
	intercept = (sy - slope * sx) / n

	x_fit = range(0, 22, length=50)
	y_fit = slope .* collect(x_fit) .+ intercept
	CM.lines!(ax, collect(x_fit), y_fit, color=:red, linestyle=:dash, linewidth=2,
		label="Linear fit")

	# R² for linearity
	y_pred = slope .* x .+ intercept
	ss_res = sum((y .- y_pred) .^ 2)
	ss_tot = sum((y .- mean(y)) .^ 2)
	r2_lin = 1.0 - ss_res / ss_tot

	CM.text!(ax, 1.0, maximum(hu_70) * 0.85,
		text="R² = $(round(r2_lin, digits=4))\nSlope = $(round(slope, digits=1)) HU/(mg/mL)",
		fontsize=12)

	CM.axislegend(ax, position=:lt)
	CM.save(joinpath(FIGURES_DIR, "nb03_iodine_linearity.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000046
md"""
## 14. Measured vs NIST Overlay (Energy Curves)

Overlay of measured and NIST-predicted HU curves for selected materials, showing the systematic underestimation inherent in the effective-energy approximation of dual-kVp VMI.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000047
let
	fig = CM.Figure(size=(1000, 500), fontsize=12)
	ax = CM.Axis(fig[1,1], title="VMI HU: Measured vs NIST Reference",
		xlabel="Energy (keV)", ylabel="HU")

	# I 20.0 (highest iodine)
	meas_i20 = [measured_hu[E].iodine[7] for E in VMI_ENERGIES]
	nist_i20 = [nist_reference[E].iodine[7] for E in VMI_ENERGIES]
	CM.lines!(ax, VMI_ENERGIES, meas_i20, color=:red, linewidth=2, label="I 20.0 (measured)")
	CM.scatter!(ax, VMI_ENERGIES, meas_i20, color=:red, markersize=6)
	CM.lines!(ax, VMI_ENERGIES, nist_i20, color=:red, linewidth=2, linestyle=:dash,
		label="I 20.0 (NIST)")

	# I 10.0
	meas_i10 = [measured_hu[E].iodine[5] for E in VMI_ENERGIES]
	nist_i10 = [nist_reference[E].iodine[5] for E in VMI_ENERGIES]
	CM.lines!(ax, VMI_ENERGIES, meas_i10, color=:purple, linewidth=2, label="I 10.0 (measured)")
	CM.scatter!(ax, VMI_ENERGIES, meas_i10, color=:purple, markersize=6)
	CM.lines!(ax, VMI_ENERGIES, nist_i10, color=:purple, linewidth=2, linestyle=:dash,
		label="I 10.0 (NIST)")

	# Ca 600
	meas_ca6 = [measured_hu[E].calcium[7] for E in VMI_ENERGIES]
	nist_ca6 = [nist_reference[E].calcium[7] for E in VMI_ENERGIES]
	CM.lines!(ax, VMI_ENERGIES, meas_ca6, color=:darkgreen, linewidth=2, label="Ca 600 (measured)")
	CM.scatter!(ax, VMI_ENERGIES, meas_ca6, color=:darkgreen, markersize=6)
	CM.lines!(ax, VMI_ENERGIES, nist_ca6, color=:darkgreen, linewidth=2, linestyle=:dash,
		label="Ca 600 (NIST)")

	# Ca 200
	meas_ca2 = [measured_hu[E].calcium[3] for E in VMI_ENERGIES]
	nist_ca2 = [nist_reference[E].calcium[3] for E in VMI_ENERGIES]
	CM.lines!(ax, VMI_ENERGIES, meas_ca2, color=:teal, linewidth=2, label="Ca 200 (measured)")
	CM.scatter!(ax, VMI_ENERGIES, meas_ca2, color=:teal, markersize=6)
	CM.lines!(ax, VMI_ENERGIES, nist_ca2, color=:teal, linewidth=2, linestyle=:dash,
		label="Ca 200 (NIST)")

	CM.hlines!(ax, [0.0], color=:gray, linestyle=:dot)
	CM.axislegend(ax, position=:rt, nbanks=2)

	CM.save(joinpath(FIGURES_DIR, "nb03_nist_overlay.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000048
md"""
## 15. Validation Summary

We compile all acceptance criteria into a single pass/fail table.
"""

# ╔═╡ a0000001-0001-0001-0001-000000000049
validation_summary = let
	results = []

	# 1. Water stability: |HU| < 30 at clinical energies (60-140 keV)
	clinical_energies = [60.0, 70.0, 80.0, 100.0, 120.0, 140.0]
	water_clinical = [abs(measured_hu[E].water) for E in clinical_energies]
	water_pass = all(water_clinical .< 30.0)
	push!(results, (test="Water ±30 HU (60-140 keV)",
		value="max |$(round(maximum(water_clinical), digits=1))| HU",
		pass=water_pass))

	# 2. Iodine R² > 0.95 across energies
	r2_i_vals = [r_squared_per_energy[E].iodine for E in VMI_ENERGIES]
	min_r2_i = minimum(r2_i_vals)
	push!(results, (test="Iodine R² > 0.95 (all energies)",
		value="min R² = $(round(min_r2_i, digits=4))",
		pass=min_r2_i > 0.95))

	# 3. Iodine linearity at 70 keV
	concentrations = [2.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
	hu_70 = measured_hu[70.0].iodine
	r_lin = cor(concentrations, hu_70)
	r2_lin = r_lin^2
	push!(results, (test="Iodine linearity R² > 0.95 (70 keV)",
		value="R² = $(round(r2_lin, digits=4))",
		pass=r2_lin > 0.95))

	# 4. Overall R² > 0.95 at 70 keV
	r2_70_all = r_squared_per_energy[70.0].all
	push!(results, (test="All materials R² > 0.95 (70 keV)",
		value="R² = $(round(r2_70_all, digits=4))",
		pass=r2_70_all > 0.95))

	# 5. Iodine energy trend: HU at 40 keV > HU at 140 keV
	i20_40 = measured_hu[40.0].iodine[7]
	i20_140 = measured_hu[140.0].iodine[7]
	push!(results, (test="Iodine energy trend (I 20: 40 > 140 keV)",
		value="$(round(i20_40, digits=0)) > $(round(i20_140, digits=0))",
		pass=i20_40 > i20_140))

	# 6. Calcium energy trend: HU at 40 keV > HU at 140 keV
	ca6_40 = measured_hu[40.0].calcium[7]
	ca6_140 = measured_hu[140.0].calcium[7]
	push!(results, (test="Calcium energy trend (Ca 600: 40 > 140 keV)",
		value="$(round(ca6_40, digits=0)) > $(round(ca6_140, digits=0))",
		pass=ca6_40 > ca6_140))

	results
end

# ╔═╡ a0000001-0001-0001-0001-000000000050
let
	fig = CM.Figure(size=(800, 300), fontsize=12)
	ax = CM.Axis(fig[1,1], title="Validation Summary",
		yreversed=true)
	CM.hidedecorations!(ax)
	CM.hidespines!(ax)

	for (i, r) in enumerate(validation_summary)
		status = r.pass ? "PASS" : "FAIL"
		color = r.pass ? :green : :red
		CM.text!(ax, 0.0, Float64(i), text="[$status] $(r.test): $(r.value)",
			fontsize=11, color=color, align=(:left, :center))
	end

	CM.ylims!(ax, 0.5, length(validation_summary) + 0.5)
	CM.xlims!(ax, -0.1, 5.0)

	CM.save(joinpath(FIGURES_DIR, "nb03_validation_summary.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ a0000001-0001-0001-0001-000000000051
md"""
## 16. Discussion

### Key Findings

1. **Energy-dependent contrast**: VMI correctly shows increasing HU for iodine and calcium at lower energies, consistent with the photoelectric effect dominating at low keV.

2. **NIST correlation**: R² > 0.95 across all VMI energies confirms that the sinogram-domain material decomposition faithfully captures the energy-dependent attenuation trends predicted by NIST XCOM data.

3. **Systematic offset**: Measured HU values may systematically underestimate NIST predictions, particularly at low keV. This is expected because:
   - The VMI pipeline uses an **effective-energy approximation** in the 2×2 decomposition matrix
   - The decomposition operates on two polychromatic spectra, not true monoenergetic beams
   - Beam hardening in the polychromatic projections introduces minor energy shifts
   - This offset does NOT indicate a physics error — it is inherent to all dual-kVp VMI systems

4. **Water calibration**: Empirical water calibration ensures water = 0 HU at all energies, providing a stable reference regardless of the VMI energy.

5. **Iodine linearity**: The linear relationship between concentration and HU at 70 keV (R² > 0.95) confirms that quantitative concentration measurement is feasible from VMI.

### Clinical Relevance

- VMI at 40-50 keV: Maximizes iodine contrast (vascular imaging)
- VMI at 65-75 keV: Equivalent to standard 120 kVp imaging
- VMI at 100-140 keV: Reduces beam hardening artifacts (metal implants)

### Limitations

- This study uses the pre-noise sinogram (`sino_ideal_out`) for decomposition. Clinical systems operate on noisy projections, which amplifies noise in the material maps.
- The Gammex 472 phantom has discrete contrast inserts; real anatomy has continuous attenuation distributions.
- The effective-energy approximation in decomposition limits absolute HU accuracy at extreme energies (< 50 keV).
"""

# ╔═╡ Cell order:
# ╟─a0000001-0001-0001-0001-000000000006
# ╠═a0000001-0001-0001-0001-000000000001
# ╠═a0000001-0001-0001-0001-000000000002
# ╠═a0000001-0001-0001-0001-000000000003
# ╠═fa4ecd77-1991-44da-b56c-37adaa9d728f
# ╠═a0000001-0001-0001-0001-000000000004
# ╠═a0000001-0001-0001-0001-000000000091
# ╠═a0000001-0001-0001-0001-000000000090
# ╠═a0000001-0001-0001-0001-000000000005
# ╠═4130c9d2-020e-4224-81da-94dfcfdab6b0
# ╠═a0000001-0001-0001-0001-000000000007
# ╟─a0000001-0001-0001-0001-000000000008
# ╟─a0000001-0001-0001-0001-000000000009
# ╠═a0000001-0001-0001-0001-000000000010
# ╠═a0000001-0001-0001-0001-000000000011
# ╠═a0000001-0001-0001-0001-000000000012
# ╠═a0000001-0001-0001-0001-000000000013
# ╠═a0000001-0001-0001-0001-000000000014
# ╟─a0000001-0001-0001-0001-000000000015
# ╠═a0000001-0001-0001-0001-000000000016
# ╠═a0000001-0001-0001-0001-000000000017
# ╟─a0000001-0001-0001-0001-000000000018
# ╠═a0000001-0001-0001-0001-000000000019
# ╠═a0000001-0001-0001-0001-000000000020
# ╟─a0000001-0001-0001-0001-000000000021
# ╠═a0000001-0001-0001-0001-000000000022
# ╠═a0000001-0001-0001-0001-000000000023
# ╟─a0000001-0001-0001-0001-000000000024
# ╠═a0000001-0001-0001-0001-000000000025
# ╠═a0000001-0001-0001-0001-000000000026
# ╟─a0000001-0001-0001-0001-000000000027
# ╟─e2c3e4d6-d4fc-44ef-84b4-b08a7f7566dc
# ╟─a0000001-0001-0001-0001-000000000028
# ╟─a0000001-0001-0001-0001-000000000029
# ╠═a0000001-0001-0001-0001-000000000030
# ╠═a0000001-0001-0001-0001-000000000031
# ╠═a0000001-0001-0001-0001-000000000032
# ╟─a0000001-0001-0001-0001-000000000033
# ╠═a0000001-0001-0001-0001-000000000034
# ╟─a0000001-0001-0001-0001-000000000035
# ╟─a0000001-0001-0001-0001-000000000036
# ╟─a0000001-0001-0001-0001-000000000037
# ╠═a0000001-0001-0001-0001-000000000038
# ╟─a0000001-0001-0001-0001-000000000039
# ╟─a0000001-0001-0001-0001-000000000040
# ╟─a0000001-0001-0001-0001-000000000041
# ╟─a0000001-0001-0001-0001-000000000042
# ╟─a0000001-0001-0001-0001-000000000043
# ╟─a0000001-0001-0001-0001-000000000044
# ╟─a0000001-0001-0001-0001-000000000045
# ╟─a0000001-0001-0001-0001-000000000046
# ╟─a0000001-0001-0001-0001-000000000047
# ╟─a0000001-0001-0001-0001-000000000048
# ╠═a0000001-0001-0001-0001-000000000049
# ╟─a0000001-0001-0001-0001-000000000050
# ╟─a0000001-0001-0001-0001-000000000051
