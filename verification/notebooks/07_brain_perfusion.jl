### A Pluto.jl notebook ###
# v0.20.21

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

# ╔═╡ 00010001-0000-4000-8000-000000000001
begin
	using Pkg: Pkg
	Pkg.activate(dirname(@__DIR__))
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 00010002-0000-4000-8000-000000000001
using Metal # Choose one or the other

# ╔═╡ 00010008-0000-4000-8000-000000000001
using MAT

# ╔═╡ 00010009-0000-4000-8000-000000000001
using Unitful: ustrip, @u_str

# ╔═╡ 00010003-0000-4000-8000-000000000001
# using CUDA # Choose one or the other

# ╔═╡ 00010004-0000-4000-8000-000000000001
import PlutoUI as UI

# ╔═╡ 00010005-0000-4000-8000-000000000001
import BasisSimulator as BS

# ╔═╡ 00010006-0000-4000-8000-000000000001
import CairoMakie as CM

# ╔═╡ 00010007-0000-4000-8000-000000000001
import XrayAttenuation as XA

# ╔═╡ 00010010-0000-4000-8000-000000000001
import Statistics: mean, std

# ╔═╡ 00010011-0000-4000-8000-000000000001
UI.TableOfContents()

# ╔═╡ 00020001-0000-4000-8000-000000000001
md"""
# Notebook 07: Brain Perfusion CT Simulation & Analysis

Dynamic contrast-enhanced brain CT using XCAT P1 (material) and P2 (segment)
phantoms, with Julia-native perfusion analysis (CBF/CBV/MTT maps).

**Workflow:**
1. Load P1/P2 XCAT brain phantom (400×400×400, UInt16)
2. Apply time-varying iodine contrast via `update_structures!`
3. BHC calibration → spectrum-analytical μ\_water (no water phantom needed)
4. CT simulation (fidelity=:high) → water BHC → FDK (custom filter) → image-domain BHC → HU → noise floor
5. Perfusion analysis: AIF extraction, CBF/CBV/MTT maps
6. Ground truth comparison (Divel et al. 2021)

**Data:** Sarah E. Divel et al., *Med. Phys.* 2021. https://doi.org/10.1002/mp.14887
"""

# ╔═╡ 00020002-0000-4000-8000-000000000001
begin
	"""Move array to GPU. Tries Metal then CUDA, falls back to CPU."""
	function to_gpu(x::AbstractArray)
		if @isdefined(Metal) && Metal.functional()
			return Metal.MtlArray(x)
		elseif @isdefined(CUDA) && CUDA.functional()
			return CUDA.CuArray(x)
		else
			@warn "No GPU backend available, using CPU"
			return x
		end
	end

	"""Force GPU memory cleanup."""
	function clear_gpu!()
		GC.gc(true)
		if @isdefined(CUDA) && CUDA.functional()
			CUDA.reclaim()
		end
		nothing
	end
end

# ╔═╡ 00030001-0000-4000-8000-000000000001
md"""
## 1. Paths & Configuration

> **Large data files are not tracked in git.**
> Phantom data from: Sarah E. Divel et al. (2021), *Med. Phys.*, https://doi.org/10.1002/mp.14887
>
> Copy the following files from the group share drive into `verification/data/brain_perfusion/`:
>
> **Share drive path:** `/Molloilab/Wenbo/brain phantom/Caedin Files/dynamic_brain_phantom`
>
> Required files:
> - `P1_brain_all_2020_RAW_400_400_400.raw` (122 MB)
> - `P2_brain_all_2020_RAW_400_400_400.raw` (122 MB)
> - `iodine_mass_data.mat` (501 MB)
> - `structure_info.mat`
"""

# ╔═╡ 00030002-0000-4000-8000-000000000001
begin
	PHANTOM_DIR    = joinpath(dirname(@__DIR__), "data", "brain_perfusion")
	P1_RAW_PATH    = joinpath(PHANTOM_DIR, "P1_brain_all_2020_RAW_400_400_400.raw")
	P1_TABLE_PATH  = joinpath(PHANTOM_DIR, "P1_voxelize_table.txt")
	P2_RAW_PATH    = joinpath(PHANTOM_DIR, "P2_brain_all_2020_RAW_400_400_400.raw")
	P2_TABLE_PATH  = joinpath(PHANTOM_DIR, "P2_vozelize_table.txt")
	STRUCT_INFO_PATH  = joinpath(PHANTOM_DIR, "structure_info.mat")
	IODINE_DATA_PATH  = joinpath(PHANTOM_DIR, "iodine_mass_data.mat")
	RESULTS_DIR       = joinpath(dirname(@__DIR__), "results", "brain_perfusion")
	mkpath(RESULTS_DIR)

	@assert isfile(P1_RAW_PATH)       "P1 raw file not found: $P1_RAW_PATH"
	@assert isfile(P1_TABLE_PATH)     "P1 table not found: $P1_TABLE_PATH"
	@assert isfile(P2_RAW_PATH)       "P2 raw file not found: $P2_RAW_PATH"
	@assert isfile(P2_TABLE_PATH)     "P2 table not found: $P2_TABLE_PATH"
	@assert isfile(STRUCT_INFO_PATH)  "structure_info.mat not found: $STRUCT_INFO_PATH"
	@assert isfile(IODINE_DATA_PATH)  "iodine_mass_data.mat not found: $IODINE_DATA_PATH"

	"All input paths verified"
end

# ╔═╡ 00040001-0000-4000-8000-000000000001
md"""
## 2. Load P1 / P2 Raw Files
"""

# ╔═╡ 00040002-0000-4000-8000-000000000001
P1_raw_file = let
	buf = Array{UInt16}(undef, 400, 400, 400)
	open(P1_RAW_PATH, "r") do io
		read!(io, buf)
	end
	arr = Int.(buf)
	arr = reverse(arr, dims=2)   # flip y: Julia col-major vs ImageJ row-major
	BS.relabel_zero_islands_2d!(arr; newlabel=10)
	arr
end;

# ╔═╡ 00040003-0000-4000-8000-000000000001
P2_raw_file = let
	buf = Array{UInt16}(undef, 400, 400, 400)
	open(P2_RAW_PATH, "r") do io
		read!(io, buf)
	end
	arr = Int.(buf)
	reverse(arr, dims=2)
end;

# ╔═╡ 00040004-0000-4000-8000-000000000001
(P1_structure_map, P2_structure_map) = (BS.load_structure_map(P1_TABLE_PATH), BS.load_structure_map(P2_TABLE_PATH))

# ╔═╡ 00040005-0000-4000-8000-000000000001
md"""
**P1:** $(length(unique(vec(P1_raw_file)))) unique material label IDs

**P2:** $(length(unique(vec(P2_raw_file)))) unique segment IDs
"""

# ╔═╡ 00050001-0000-4000-8000-000000000001
md"""
## 3. Load Materials

| ID | Material | Example structures |
|----|----------|--------------------|
| 0  | `:air` | background, oral cavity, tendons, spinal cord |
| 1  | `:muscle` | scalp/neck muscles, tongue, parotid glands, eyes |
| 2  | `:air` | throat (airway passages) |
| 3  | `:bone` | cervical spine (atlas, axis, C3-C5) |
| 5  | `:soft_tissue` | head surface, ears |
| 10 | `:soft_tissue` | interior zero islands (relabeled by `relabel_zero_islands_2d!`) |
| 13 | `:bone` | skull (frontal, temporal, parietal, occipital, mandible) |
| 14 | `:soft_tissue` | intervertebral disks |
| 17 | `:csf` | ventricles (lateral, third, fourth), cerebral aqueduct |
| 18 | `:gray_matter` | brain parenchyma + 78 named GM segments |
| 19 | `:white_matter` | cerebral lobes + 117 named WM segments |
| 21 | `:blood` | arteries: 399 segments (ICA, MCA, ACA, basilar, vertebral, ...) |
| 22 | `:blood` | veins: 235 segments (jugular, sagittal/transverse sinus, ...) |
"""

# ╔═╡ 00050002-0000-4000-8000-000000000001
MATERIAL_MAP_BASE = Dict{Int, Symbol}(
	0  => :air,
	1  => :muscle,
	2  => :air,
	3  => :bone,
	5  => :soft_tissue,
	10 => :soft_tissue,
	13 => :bone,
	14 => :soft_tissue,
	17 => :csf,
	18 => :gray_matter,
	19 => :white_matter,
	21 => :blood,
	22 => :blood,
)

# ╔═╡ 00050003-0000-4000-8000-000000000001
(material_map_init, material_list_init) = let
	unique_ids    = sort(unique(vec(P1_raw_file)))
	material_map  = Dict{Int, Symbol}(k => v for (k, v) in MATERIAL_MAP_BASE if k in unique_ids)
	material_list = Dict{Symbol, XA.Material}(
		sym => BS.get_material(sym) for sym in unique(collect(values(material_map)))
	)
	(material_map, material_list)
end

# ╔═╡ 00050004-0000-4000-8000-000000000001
md"""
**Base materials loaded:** $(length(material_list_init))

Symbols: $(sort(collect(keys(material_list_init)), by=string))
"""

# ╔═╡ 00060001-0000-4000-8000-000000000001
md"""
## 4. Load Iodine Contrast Data
"""

# ╔═╡ 00060002-0000-4000-8000-000000000001
(artery_info, vein_info, gm_info, wm_info) = let
	si = matread(STRUCT_INFO_PATH)
	function parse_info(d)
		names = vec(String.(d["name"]))
		vols  = vec(Float64.(d["volume"]))
		Dict{String, Any}("name" => names, "volume" => vols)
	end
	(parse_info(si["artery_info"]),
	 parse_info(si["vein_info"]),
	 parse_info(si["gm_info"]),
	 parse_info(si["wm_info"]))
end

# ╔═╡ 00060003-0000-4000-8000-000000000001
(iodine_artery, iodine_vein, iodine_gm, iodine_wm) = let
	d = matread(IODINE_DATA_PATH)
	(Float64.(d["mass_arteries"]),
	 Float64.(d["mass_vein"]),
	 Float64.(d["mass_gm"]),
	 Float64.(d["mass_wm"]))
end

# ╔═╡ 00060004-0000-4000-8000-000000000001
md"""
**Artery segments:** $(size(iodine_artery, 1)) | **Vein segments:** $(size(iodine_vein, 1))

**GM segments:** $(size(iodine_gm, 1)) | **WM segments:** $(size(iodine_wm, 1))

**Time points:** $(size(iodine_artery, 2)) (0-85 s at 1 ms resolution)
"""

# ╔═╡ 00060005-0000-4000-8000-000000000001
md"""
### 4a. Data Verification: Time-Concentration Curves
"""

# ╔═╡ 00060006-0000-4000-8000-000000000001
let
	tac_indices = round.(Int, range(1, 85001; length=200))
	tac_sec     = (tac_indices .- 1) ./ 1000.0

	artery44_curve = [iodine_artery[44, t] for t in tac_indices]
	other_arteries = [mean(iodine_artery[setdiff(1:size(iodine_artery,1), [44]), t]) for t in tac_indices]

	f = CM.Figure(size=(900, 500))
	ax = CM.Axis(f[1, 1],
		title  = "Artery 44 (L-M1 MCA) vs Other Arteries",
		xlabel = "Time (s)",
		ylabel = "Iodine Mass (mg)",
	)
	CM.lines!(ax, tac_sec, artery44_curve; color=:red, linewidth=2, label="Artery 44 (L-M1 MCA)")
	CM.lines!(ax, tac_sec, other_arteries; color=:blue, linewidth=2, label="Other arteries (mean)")
	CM.axislegend(ax; position=:rt)
	CM.save(joinpath(RESULTS_DIR, "artery44_vs_others.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00060007-0000-4000-8000-000000000001
let
	tac_indices = round.(Int, range(1, 85001; length=100))
	tac_sec     = (tac_indices .- 1) ./ 1000.0
	artery_mean = [mean(iodine_artery[:, t]) for t in tac_indices]
	vein_mean   = [mean(iodine_vein[:,   t]) for t in tac_indices]
	gm_mean     = [mean(iodine_gm[:,     t]) for t in tac_indices]
	wm_mean     = [mean(iodine_wm[:,     t]) for t in tac_indices]

	f  = CM.Figure(size=(900, 500))
	ax = CM.Axis(f[1, 1],
		title  = "Time-Iodine Curves: Mean per Tissue Type",
		xlabel = "Time (s)",
		ylabel = "Iodine (mg/mL or mg/g)",
	)
	CM.lines!(ax, tac_sec, artery_mean; color=:red,    linewidth=2, label="Artery")
	CM.lines!(ax, tac_sec, vein_mean;   color=:blue,   linewidth=2, label="Vein")
	CM.lines!(ax, tac_sec, gm_mean;     color=:green,  linewidth=2, label="Gray Matter")
	CM.lines!(ax, tac_sec, wm_mean;     color=:orange, linewidth=2, label="White Matter")
	CM.axislegend(ax; position=:rt)
	CM.save(joinpath(RESULTS_DIR, "tac_all_tissues.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00060008-0000-4000-8000-000000000001
md"""
### Divel et al. 2021 Ground Truth (Table I)

| Tissue | CBF (mL/min/100g) | CBV (mL/100g) | MTT (s) |
|--------|-------------------|---------------|---------|
| Healthy GM | 54.50 | 5.20 | 5.72 |
| Ischemic GM | 10.90 | 1.77 | 9.72 |
| Healthy WM | 22.20 | 2.70 | 7.30 |
| Ischemic WM | 4.44 | 0.84 | 11.30 |
"""

# ╔═╡ 00070001-0000-4000-8000-000000000001
md"""
## 5. Scanner & Protocol: GE Revolution Apex (120 kVp)
"""

# ╔═╡ 00070002-0000-4000-8000-000000000001
additional_filters = [("Al", 4.5)]

# ╔═╡ 00070003-0000-4000-8000-000000000001
begin
	brain_extent_mm = 400 * 0.1 * 10.0        # phantom FOV in mm
	brain_eict_mag  = 1100.0 / 625.6           # magnification SDD/SID
	brain_det_cols  = ceil(Int, brain_extent_mm * brain_eict_mag / 1.0)
	brain_det_rows  = 256
	brain_recon_fov = 40.0  # cm
	brain_vox_cm    = 0.1   # phantom voxel size (cm)

	# Scanner physical z-coverage limit
	brain_max_collimation_mm = brain_det_rows * 0.625  # 256 × 0.625 = 160 mm
	brain_max_z_slices = floor(Int, brain_max_collimation_mm / (brain_vox_cm * 10.0))  # max phantom slices that fit

	# Auto-detect tissue extent along z from P1 phantom, clamped to scanner coverage
	_z_any       = vec(any(P1_raw_file .!= 0, dims=(1,2)))
	_z_first     = findfirst(_z_any)
	_z_last      = findlast(_z_any)
	_z_tissue    = max(1, _z_first - 4) : min(size(P1_raw_file, 3), _z_last + 4)
	# Center-crop to scanner max if phantom z-extent exceeds detector coverage
	if length(_z_tissue) > brain_max_z_slices
		_z_mid   = (_z_tissue[1] + _z_tissue[end]) ÷ 2
		_z_half  = brain_max_z_slices ÷ 2
		BRAIN_Z_CROP = max(1, _z_mid - _z_half + 1) : min(size(P1_raw_file, 3), _z_mid + _z_half)
	else
		BRAIN_Z_CROP = _z_tissue
	end
	brain_z_cm   = length(BRAIN_Z_CROP) * brain_vox_cm
	brain_recon_xy = 512
	brain_n_slices = round(Int, brain_z_cm / brain_vox_cm)
end

# ╔═╡ 00070004-0000-4000-8000-000000000001
scanner_brain = BS.Scanner(
	source_to_isocenter   = 625.6,
	source_to_detector    = 1100.0,
	detector_rows         = brain_det_rows,
	detector_cols         = brain_det_cols,
	detector_row_size     = 0.625,
	detector_col_size     = 1.0,
	detector_shape        = BS.CURVED_DETECTOR,
	focal_spot_width      = 1.0,
	focal_spot_length     = 1.0,
	target_angle          = 7.0,
	flat_filter_material  = :aluminum,
	flat_filter_thickness = 2.5,
	bowtie_filter         = :ge_revolution_large,
	detector_material     = :lumex,
	detector_depth        = 3.0,
	fill_factor_row       = 0.9,
	fill_factor_col       = 0.9,
	electronic_noise      = 0,
	detection_gain        = 10.0,
)

# ╔═╡ 00070005-0000-4000-8000-000000000001
protocol_brain = BS.CTProtocol(
	kVp           = 120,
	mA            = 300.0,
	views         = 984,
	rotation_time = 0.5,
	collimation_mm = brain_z_cm * 10.0,
	additional_filters = additional_filters,
)

# ╔═╡ 00070006-0000-4000-8000-000000000001
sim_opts_brain = BS.SimOptions(fidelity = :high, seed = 1234)

# ╔═╡ 00070007-0000-4000-8000-000000000001
recon_opts_brain = BS.ReconOptions(
	algorithm   = :fdk,
	matrix_size = (brain_recon_xy, brain_recon_xy, brain_n_slices),
	fov_cm      = brain_recon_fov,
	z_cm        = brain_z_cm,
)

# ╔═╡ 00080001-0000-4000-8000-000000000001
md"""
## 6. BHC Calibration, FBP Filter & μ\_water

Spectrum-analytical calibration following notebook 00. μ\_water is derived from the
BHC model — no water phantom simulation needed. This avoids scatter contamination
that inflates μ\_water with `fidelity=:high` (simulator has no scatter correction).

Custom FBP filter, noise floor, and BHC parameters all match notebook 00.
"""

# ╔═╡ 00080002-0000-4000-8000-000000000001
# Custom FBP filter (verified against clinical kernel — matching notebook 00)
custom_filter_control = (
	x = (0.0, 0.25, 0.5, 0.75, 1.0),
	y = (1.0, 0.85, 0.60, 0.15, 0.001),
)

# ╔═╡ 00080002-a000-4000-8000-000000000001
# Noise floor: dose-independent system noise (scatter residuals, electronics, calibration)
sim_noise_floor_hu = 28.0

# ╔═╡ 00080002-b000-4000-8000-000000000001
begin
	# BHC parameters (matching notebook 00 pattern)
	bhc_enabled = true
	sino_bhc_hu_low = 450.0
	sino_bhc_hu_high = 600.0
	sino_bhc_order = 2
	# Image-domain BHC (residual high-Z correction)
	img_bhc_hu_low = 50.0
	img_bhc_hu_high = 150.0
	img_bhc_scale_factor = 0.2
end

# ╔═╡ 00080003-0000-4000-8000-000000000001
# Calibrate two-material BHC model for 120 kVp
bhc_model, μ_water_brain = let
	e, w = BS.resolve_spectrum(sim_opts_brain, protocol_brain; scanner = scanner_brain)
	ref_E = sum(e .* w) / sum(w)
	model = BS.calibrate_bhc_two_material(e, w;
		order = sino_bhc_order,
		reference_energy_keV = ref_E,
		hu_low = sino_bhc_hu_low,
		hu_high = sino_bhc_hu_high)
	@info "BHC 120 kVp: E_ref=$(round(ref_E, digits=1)) keV, μ_water=$(round(model.μ_water_ref, digits=5)) cm⁻¹"
	model, model.μ_water_ref
end

# ╔═╡ 00080004-0000-4000-8000-000000000001
md"""
**μ\_water @ 120 kVp:** $(round(μ_water_brain, sigdigits=4)) cm⁻¹ (spectrum-analytical, from BHC model)

Expected: ~0.19–0.21 cm⁻¹ (NIST monochromatic at 60–70 keV)
"""

# ╔═╡ 00090001-0000-4000-8000-000000000001
md"""
## 7. Configurable Parameters
"""

# ╔═╡ 00090002-0000-4000-8000-000000000001
begin
	# Time points (seconds) — 5s spacing covers baseline, upslope, peak, and washout.
	# Maximum slope CTP is valid at 5s temporal resolution.
	CONTRAST_TIME_S = [0, 5, 10, 15, 20]
	CONTRAST_INDICES = CONTRAST_TIME_S .* 1000 .+ 1
end

# ╔═╡ 00100001-0000-4000-8000-000000000001
md"""
## 8. Pre-compute: All Time Points

Runs all contrast time points upfront. Reconstruction pipeline is **identical to
notebook 00**: `simulate!` → water polynomial BHC → FDK (custom filter) →
`apply_bhc_image_domain` → `to_hounsfield` → noise floor.
Note: two-material sinogram BHC and cupping correction are disabled for brain
(cone-beam artifacts at 160mm collimation; polynomial extrapolation in 40cm FOV).
"""

# ╔═╡ 00100002-0000-4000-8000-000000000001
begin
	all_fdk_hu = Dict{Int, Array{Float32, 3}}()
	all_contrast_phantom = Dict{Int, Array{Int, 3}}()

	let
		recon_size = (brain_recon_xy, brain_recon_xy, brain_n_slices)
		P1_stamped   = copy(P1_raw_file)[:, :, BRAIN_Z_CROP]
		P2_raw_crop  = P2_raw_file[:, :, BRAIN_Z_CROP]

		# Pre-build segment -> voxel index map once
		segment_index_map = Dict{Int, Vector{CartesianIndex{3}}}()
		for (id, name) in P2_structure_map
			first(name) in ('2', '3', '4', '5') || continue
			idxs = findall(==(id), P2_raw_crop)
			isempty(idxs) || (segment_index_map[id] = idxs)
		end

		# Stamp all segment IDs into P1 once
		for (id, idxs) in segment_index_map
			P1_stamped[idxs] .= id
		end

		mask_gpu = to_gpu(UInt16.(P1_stamped))

		# Helper: rebuild materials_dict for one time point
		function build_phantom(t_contrast)
			materials_dict = Dict{Int, XA.Material}(
				id => material_list_init[sym] for (id, sym) in material_map_init
			)

			for (prefix, info, iodine, base_sym) in [
				("5", artery_info, iodine_artery, :blood),
				("4", vein_info,   iodine_vein,   :blood),
				("3", wm_info,     iodine_wm,     :white_matter),
				("2", gm_info,     iodine_gm,     :gray_matter),
			]
				seg_mats = BS.update_structures!(
					P1_stamped, P2_structure_map, prefix, P2_raw_crop,
					base_sym, material_list_init[base_sym], info, iodine, t_contrast;
					index_map = segment_index_map
				)
				merge!(materials_dict, seg_mats)
			end

			# Fill any unstamped segments with their base material
			_prefix_to_base = Dict("5" => :blood, "4" => :blood, "3" => :white_matter, "2" => :gray_matter)
			for (id, name) in P2_structure_map
				id ∈ keys(materials_dict) && continue
				id ∈ keys(segment_index_map) || continue
				c = string(first(name))
				c ∈ keys(_prefix_to_base) || continue
				materials_dict[id] = material_list_init[_prefix_to_base[c]]
			end

			let (nx, ny, nz) = size(P1_stamped)
				dx, dy, dz = 0.1, 0.1, 0.1
				BS.Phantom(mask_gpu, BS.build_materials_vector(materials_dict),
					(dx, dy, dz),
					(-dx*nx/2 + dx/2, -dy*ny/2 + dy/2, -dz*nz/2 + dz/2),
					(dx*nx, dy*ny, dz*nz))
			end
		end

		# Build first phantom to size the workspace
		phantom_t_0 = build_phantom(CONTRAST_INDICES[1])
		for i in eachindex(CONTRAST_INDICES)
			all_contrast_phantom[i] = copy(P1_stamped)
		end

		ws = BS.create_eict_workspace(
			scanner_brain, protocol_brain,
			sim_opts_brain, recon_opts_brain, phantom_t_0
		)
		geom = ws.geom

		# Warmup run
		BS.simulate!(ws, phantom_t_0, scanner_brain, protocol_brain, sim_opts_brain, recon_opts_brain)

		for (i, t_contrast) in enumerate(CONTRAST_INDICES)
			println("▶ Time point $i/$(length(CONTRAST_INDICES)): t = $(CONTRAST_TIME_S[i]) s  (index $t_contrast)")

			phantom_t = i == 1 ? phantom_t_0 : build_phantom(t_contrast)

			@time begin
				BS.simulate!(
					ws, phantom_t, scanner_brain, protocol_brain,
					sim_opts_brain, recon_opts_brain
				)

				sino_gpu = to_gpu(ws.sino_noisy_out)

				# Water-only polynomial BHC (sinogram domain)
				# Note: two-material sinogram BHC (apply_bhc_two_material) is disabled
				# for brain — its intermediate FDK fails at wide cone angles (7.3° with
				# 160 mm collimation). Image-domain BHC handles residual bone correction.
				if bhc_enabled
					BS.apply_bhc!(sino_gpu, bhc_model.water_bhc)
				end

				# FDK reconstruction (with custom filter — matching notebook 00)
				ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size;
					filter = BS.CustomFilter(custom_filter_control.x, custom_filter_control.y))
				recon_μ = BS.reconstruct!(ws_fdk, sino_gpu, geom, recon_size)

				# Image-domain BHC (residual high-Z correction)
				if bhc_enabled
					BS.apply_bhc_image_domain(recon_μ, geom, recon_size, μ_water_brain;
						hu_low = img_bhc_hu_low, hu_high = img_bhc_hu_high,
						scale_factor = img_bhc_scale_factor,
						volume_extent = phantom_t.extent)
				end

				vol = Array(recon_μ)

				recon_hu = Float32.(BS.to_hounsfield(vol; μ_water = μ_water_brain))
				BS.add_system_noise_floor!(recon_hu, sim_noise_floor_hu)
				# Note: radial cupping correction is disabled for brain — the polynomial
				# extrapolates catastrophically beyond the skull (brain tissue at r=2-8 cm
				# in a 40 cm FOV). BHC already corrects beam-hardening cupping.
				all_fdk_hu[i] = reverse(recon_hu, dims=3)

				ws_fdk = nothing; sino_gpu = nothing; recon_μ = nothing; vol = nothing
				phantom_t = nothing
			end
			println("   done t=$(CONTRAST_TIME_S[i]) s")
		end

		ws = nothing
		clear_gpu!()
		println("All time points complete.")
	end;
end

# ╔═╡ 00110001-0000-4000-8000-000000000001
md"""
## 9. Visualizations
"""

# ╔═╡ 00110002-0000-4000-8000-000000000001
md"""
### 9.1 Phantoms
"""

# ╔═╡ 00110003-0000-4000-8000-000000000001
md"""Select phantom z slice:"""

# ╔═╡ 00110004-0000-4000-8000-000000000001
@bind z_preview UI.Slider(1:brain_n_slices; default=90, show_value=true)

# ╔═╡ 00110005-0000-4000-8000-000000000001
let
	p1_slice  = P1_raw_file[:, :, BRAIN_Z_CROP[z_preview]]
	seg_slice = all_contrast_phantom[1][:, :, z_preview]
	uvals = sort(unique(vec(seg_slice)))
	n     = length(uvals)
	v2i   = Dict(v => Float32(i-1) / max(n-1, 1) for (i,v) in enumerate(uvals))
	disp  = [v2i[v] for v in seg_slice]

	f = CM.Figure(size=(1200, 520))
	ax1 = CM.Axis(f[1, 1],
		title  = "P1 Material Labels (z = $z_preview)",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	hm1 = CM.heatmap!(ax1, p1_slice; colormap = :tab20)
	CM.Colorbar(f[1, 2], hm1; label = "Material ID")

	ax2 = CM.Axis(f[1, 3],
		title  = "Contrast Phantom: Segment IDs at t = $(CONTRAST_TIME_S[1]) s",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	CM.heatmap!(ax2, disp; colormap = :turbo, colorrange = (0, 1))

	CM.save(joinpath(RESULTS_DIR, "phantom_slices.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00120001-0000-4000-8000-000000000001
md"""
### 9.2 CT Scans
"""

# ╔═╡ 00120002-0000-4000-8000-000000000001
md"""Select time point index:"""

# ╔═╡ 00120003-0000-4000-8000-000000000001
@bind t_idx UI.Slider(1:length(CONTRAST_TIME_S); default=min(3, length(CONTRAST_TIME_S)), show_value=true)

# ╔═╡ 00120004-0000-4000-8000-000000000001
md"""
**Selected time:** $(CONTRAST_TIME_S[t_idx]) s  (index $(CONTRAST_INDICES[t_idx]))
"""

# ╔═╡ 00120005-0000-4000-8000-000000000001
md"""Select CT scan z slice:"""

# ╔═╡ 00120006-0000-4000-8000-000000000001
@bind mid_Z UI.Slider(1:brain_n_slices; default=92, show_value=true)

# ╔═╡ 00120007-0000-4000-8000-000000000001
let
	fdk_hu = all_fdk_hu[t_idx]
	t_s    = CONTRAST_TIME_S[t_idx]

	f = CM.Figure(size=(1000, 480))

	# Brain window: W=80, L=40 → range [0, 80]
	ax1 = CM.Axis(f[1, 1],
		title  = "FDK: Brain Window (W=80, L=40)  t=$(t_s)s  z=$mid_Z",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	hm1 = CM.heatmap!(ax1, fdk_hu[:, :, mid_Z];
		colormap = :grays, colorrange = (0, 80))
	CM.Colorbar(f[1, 2], hm1; label = "HU")

	# Soft tissue window: W=400, L=40 → range [-160, 240]
	ax2 = CM.Axis(f[1, 3],
		title  = "FDK: Soft Tissue Window  t=$(t_s)s  z=$mid_Z",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	hm2 = CM.heatmap!(ax2, fdk_hu[:, :, mid_Z];
		colormap = :grays, colorrange = (-160, 240))
	CM.Colorbar(f[1, 4], hm2; label = "HU")

	CM.save(joinpath(RESULTS_DIR, "brain_fdk_windows.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00120008-0000-4000-8000-000000000001
let
	fdk_hu = all_fdk_hu[t_idx]
	t_s    = CONTRAST_TIME_S[t_idx]

	f = CM.Figure(size=(500, 480))

	ax = CM.Axis(f[1, 1],
		title  = "FDK: Bone Window (W=2000, L=500)  t=$(t_s)s  z=$mid_Z",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	hm = CM.heatmap!(ax, fdk_hu[:, :, mid_Z];
		colormap = :grays, colorrange = (-500, 1500))
	CM.Colorbar(f[1, 2], hm; label = "HU")

	CM.save(joinpath(RESULTS_DIR, "brain_bone_window.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00120009-0000-4000-8000-000000000001
md"""
### 9.3 Time-Series Montage
"""

# ╔═╡ 00120010-0000-4000-8000-000000000001
let
	n_t = length(CONTRAST_TIME_S)
	mid_z = brain_n_slices ÷ 2
	f = CM.Figure(size=(250 * n_t, 280))

	for (i, t_s) in enumerate(CONTRAST_TIME_S)
		ax = CM.Axis(f[1, i],
			title = "t=$(t_s)s",
			aspect = CM.DataAspect(),
			xticksvisible = false, yticksvisible = false,
			xticklabelsvisible = false, yticklabelsvisible = false,
		)
		CM.heatmap!(ax, all_fdk_hu[i][:, :, mid_z];
			colormap = :grays, colorrange = (0, 80))
	end

	CM.save(joinpath(RESULTS_DIR, "time_series_montage.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00130001-0000-4000-8000-000000000001
md"""
## 10. Perfusion Analysis (Julia Native)

Extract AIF from known ICA segments in P2, compute per-voxel CBF/CBV/MTT maps
using the maximum slope method.
"""

# ╔═╡ 00130002-0000-4000-8000-000000000001
md"""
### 10.1 Extract AIF from Known Artery Segments
"""

# ╔═╡ 00130003-0000-4000-8000-000000000001
begin
	# Identify ICA segment IDs from P2 structure map
	ica_segment_ids = Int[]
	for (id, name) in P2_structure_map
		if occursin(r"carotid|ICA"i, name) && startswith(name, "5")
			push!(ica_segment_ids, id)
		end
	end

	# If no ICA found, fall back to first few artery segments
	if isempty(ica_segment_ids)
		artery_ids = sort([id for (id, name) in P2_structure_map if startswith(name, "5")])
		ica_segment_ids = artery_ids[1:min(5, length(artery_ids))]
	end

	P2_crop_for_perf = P2_raw_file[:, :, BRAIN_Z_CROP]
	ica_mask_phantom = falses(size(P2_crop_for_perf))
	for id in ica_segment_ids
		ica_mask_phantom .|= (P2_crop_for_perf .== id)
	end

	@info "ICA segments: $(length(ica_segment_ids)) segments, $(sum(ica_mask_phantom)) phantom voxels"
end

# ╔═╡ 00130004-0000-4000-8000-000000000001
begin
	P1_crop_for_perf = P1_raw_file[:, :, BRAIN_Z_CROP]
	brain_tissue_mask_phantom = (P1_crop_for_perf .== 18) .| (P1_crop_for_perf .== 19)

	# Map phantom (400×400) to recon (512×512) coordinates
	scale_factor = brain_recon_xy / 400.0
	recon_nz = size(all_fdk_hu[1], 3)

	function phantom_to_recon_idx(px, py, pz)
		rx = clamp(round(Int, (px - 0.5) * scale_factor + 0.5), 1, brain_recon_xy)
		ry = clamp(round(Int, (py - 0.5) * scale_factor + 0.5), 1, brain_recon_xy)
		rz = clamp(pz, 1, recon_nz)
		return (rx, ry, rz)
	end

	ica_mask_recon = falses(brain_recon_xy, brain_recon_xy, recon_nz)
	brain_mask_recon = falses(brain_recon_xy, brain_recon_xy, recon_nz)

	for idx in CartesianIndices(ica_mask_phantom)
		if ica_mask_phantom[idx]
			rx, ry, rz = phantom_to_recon_idx(idx[1], idx[2], idx[3])
			ica_mask_recon[rx, ry, rz] = true
		end
	end

	for idx in CartesianIndices(brain_tissue_mask_phantom)
		if brain_tissue_mask_phantom[idx]
			rx, ry, rz = phantom_to_recon_idx(idx[1], idx[2], idx[3])
			brain_mask_recon[rx, ry, rz] = true
		end
	end

	@info "Recon-space masks: ICA=$(sum(ica_mask_recon)) voxels, Brain=$(sum(brain_mask_recon)) voxels"
end

# ╔═╡ 00130005-0000-4000-8000-000000000001
begin
	times_s = Float64.(CONTRAST_TIME_S)
	n_timepoints = length(CONTRAST_TIME_S)

	aif_fdk = [mean(all_fdk_hu[t][ica_mask_recon]) for t in 1:n_timepoints]
	aif_enhancement_fdk = aif_fdk .- aif_fdk[1]

	@info "AIF peak enhancement (FDK): $(round(maximum(aif_enhancement_fdk), digits=1)) HU"
	@info "AIF baseline HU (FDK): $(round(aif_fdk[1], digits=1)) HU"
end

# ╔═╡ 00130006-0000-4000-8000-000000000001
let
	f = CM.Figure(size=(800, 420))
	ax = CM.Axis(f[1, 1],
		title  = "Arterial Input Function (AIF) from ICA Segments",
		xlabel = "Time (s)",
		ylabel = "Enhancement (HU)",
	)
	CM.scatterlines!(ax, times_s, aif_enhancement_fdk; color=:steelblue, linewidth=2,
		markersize=8, label="FDK AIF")
	CM.hlines!(ax, [0]; color=:gray, linestyle=:dash, linewidth=1)
	CM.axislegend(ax; position=:rt)
	CM.save(joinpath(RESULTS_DIR, "aif_curves.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00140001-0000-4000-8000-000000000001
md"""
### 10.2 Compute Perfusion Maps: Maximum Slope Method

Per brain voxel:
- **CBF** = max\_slope(tissue\_curve) / max(AIF)  (mL/min/100g)
- **CBV** = tissue\_area / AIF\_area  (mL/100g)
- **MTT** = CBV / CBF  (seconds, central volume principle)
"""

# ╔═╡ 00140002-0000-4000-8000-000000000001
"""Trapezoidal integration."""
function trapz(x::AbstractVector, y::AbstractVector)
	s = 0.0
	for i in 2:length(x)
		s += 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
	end
	return s
end

# ╔═╡ 00140003-0000-4000-8000-000000000001
function compute_perfusion_maps(hu_volumes, times, aif_curve, brain_mask)
	n_t = length(times)
	baseline = hu_volumes[1]

	enhancement = [vol .- baseline for vol in hu_volumes]

	aif_enhancement = aif_curve .- aif_curve[1]
	aif_peak = maximum(aif_enhancement)
	aif_auc = trapz(times, aif_enhancement)

	if aif_peak <= 0 || aif_auc <= 0
		@warn "AIF peak ($aif_peak) or AUC ($aif_auc) is non-positive, returning zero maps"
		z = zeros(Float32, size(baseline))
		return z, copy(z), copy(z), copy(z)
	end

	cbf_map  = zeros(Float32, size(baseline))
	cbv_map  = zeros(Float32, size(baseline))
	mtt_map  = zeros(Float32, size(baseline))
	tmax_map = zeros(Float32, size(baseline))

	dt = diff(times)

	for idx in CartesianIndices(baseline)
		brain_mask[idx] || continue

		tissue_curve = Float64[enhancement[t][idx] for t in 1:n_t]

		slopes = diff(tissue_curve) ./ dt
		max_slope = maximum(slopes)
		cbf_map[idx] = Float32((max_slope / aif_peak) * 100.0 * 60.0)

		tissue_auc = trapz(times, tissue_curve)
		cbv_map[idx] = Float32((tissue_auc / aif_auc) * 100.0)

		if cbf_map[idx] > 0
			mtt_map[idx] = Float32(cbv_map[idx] / cbf_map[idx] * 60.0)
		end

		_, peak_idx = findmax(tissue_curve)
		tmax_map[idx] = Float32(times[peak_idx])
	end

	return cbf_map, cbv_map, mtt_map, tmax_map
end

# ╔═╡ 00140004-0000-4000-8000-000000000001
(cbf_fdk, cbv_fdk, mtt_fdk, tmax_fdk) = compute_perfusion_maps(
	[all_fdk_hu[i] for i in 1:n_timepoints],
	times_s, aif_fdk, brain_mask_recon)

# ╔═╡ 00150001-0000-4000-8000-000000000001
md"""
### 10.3 Perfusion Map Visualization
"""

# ╔═╡ 00150002-0000-4000-8000-000000000001
md"""Select perfusion map z slice:"""

# ╔═╡ 00150003-0000-4000-8000-000000000001
@bind z_perf UI.Slider(1:recon_nz; default=recon_nz÷2, show_value=true)

# ╔═╡ 00150004-0000-4000-8000-000000000001
let
	f = CM.Figure(size=(1400, 500), fontsize=12)

	for (col, map, title, crange, cmap) in [
		(1, cbf_fdk, "CBF (mL/min/100g)", (0, 80), :jet),
		(2, cbv_fdk, "CBV (mL/100g)", (0, 8), :jet),
		(3, mtt_fdk, "MTT (s)", (0, 15), :jet),
		(4, tmax_fdk, "Tmax (s)", (0, maximum(times_s)), :jet),
	]
		ax = CM.Axis(f[1, col], title=title, aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		slice = map[:, :, z_perf]
		display_slice = copy(slice)
		display_slice[.!brain_mask_recon[:, :, z_perf]] .= NaN32
		hm = CM.heatmap!(ax, display_slice; colormap=cmap, colorrange=crange, nan_color=:black)
		CM.Colorbar(f[1, col+4], hm; width=10)
	end

	CM.Label(f[0, 1:4], "FDK Perfusion Maps (z = $z_perf)", fontsize=16)

	CM.save(joinpath(RESULTS_DIR, "perfusion_maps.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00160001-0000-4000-8000-000000000001
md"""
### 10.4 Ground Truth Comparison

Compare computed perfusion values against Divel et al. 2021 ground truth.
Uses P1 labels (18=GM, 19=WM) for tissue type and a CBF threshold to separate
healthy vs ischemic territory (L-M1 MCA occlusion).
"""

# ╔═╡ 00160002-0000-4000-8000-000000000001
begin
	gm_mask_recon = falses(brain_recon_xy, brain_recon_xy, recon_nz)
	wm_mask_recon = falses(brain_recon_xy, brain_recon_xy, recon_nz)

	for idx in CartesianIndices(P1_crop_for_perf)
		rx, ry, rz = phantom_to_recon_idx(idx[1], idx[2], idx[3])
		if P1_crop_for_perf[idx] == 18
			gm_mask_recon[rx, ry, rz] = true
		elseif P1_crop_for_perf[idx] == 19
			wm_mask_recon[rx, ry, rz] = true
		end
	end

	cbf_ischemic_threshold = 32.7  # mL/min/100g

	gm_healthy_mask = gm_mask_recon .& (cbf_fdk .>= cbf_ischemic_threshold)
	gm_ischemic_mask = gm_mask_recon .& (cbf_fdk .> 0) .& (cbf_fdk .< cbf_ischemic_threshold)
	wm_healthy_mask = wm_mask_recon .& (cbf_fdk .>= cbf_ischemic_threshold)
	wm_ischemic_mask = wm_mask_recon .& (cbf_fdk .> 0) .& (cbf_fdk .< cbf_ischemic_threshold)

	healthy_gm_cbf = cbf_fdk[gm_healthy_mask]
	healthy_gm_cbv = cbv_fdk[gm_healthy_mask]
	healthy_gm_mtt = mtt_fdk[gm_healthy_mask .& (mtt_fdk .> 0)]
	ischemic_gm_cbf = cbf_fdk[gm_ischemic_mask]
	ischemic_gm_cbv = cbv_fdk[gm_ischemic_mask]
	ischemic_gm_mtt = mtt_fdk[gm_ischemic_mask .& (mtt_fdk .> 0)]

	healthy_wm_cbf = cbf_fdk[wm_healthy_mask]
	healthy_wm_cbv = cbv_fdk[wm_healthy_mask]
	healthy_wm_mtt = mtt_fdk[wm_healthy_mask .& (mtt_fdk .> 0)]
	ischemic_wm_cbf = cbf_fdk[wm_ischemic_mask]
	ischemic_wm_cbv = cbv_fdk[wm_ischemic_mask]
	ischemic_wm_mtt = mtt_fdk[wm_ischemic_mask .& (mtt_fdk .> 0)]

	_fmt(vals) = isempty(vals) ? "N/A" : "$(round(mean(vals), digits=2)) ± $(round(std(vals), digits=2))"

	md"""
	### Computed Perfusion Values (FDK) — Healthy vs Ischemic

	| Tissue | CBF (mL/min/100g) | CBV (mL/100g) | MTT (s) |
	|--------|-------------------|---------------|---------|
	| Healthy GM (n=$(length(healthy_gm_cbf))) | $(_fmt(healthy_gm_cbf)) | $(_fmt(healthy_gm_cbv)) | $(_fmt(healthy_gm_mtt)) |
	| Ischemic GM (n=$(length(ischemic_gm_cbf))) | $(_fmt(ischemic_gm_cbf)) | $(_fmt(ischemic_gm_cbv)) | $(_fmt(ischemic_gm_mtt)) |
	| Healthy WM (n=$(length(healthy_wm_cbf))) | $(_fmt(healthy_wm_cbf)) | $(_fmt(healthy_wm_cbv)) | $(_fmt(healthy_wm_mtt)) |
	| Ischemic WM (n=$(length(ischemic_wm_cbf))) | $(_fmt(ischemic_wm_cbf)) | $(_fmt(ischemic_wm_cbv)) | $(_fmt(ischemic_wm_mtt)) |

	### Divel Ground Truth (Table I)

	| Tissue | CBF | CBV | MTT |
	|--------|-----|-----|-----|
	| Healthy GM | 54.50 | 5.20 | 5.72 |
	| Ischemic GM | 10.90 | 1.77 | 9.72 |
	| Healthy WM | 22.20 | 2.70 | 7.30 |
	| Ischemic WM | 4.44 | 0.84 | 11.30 |
	"""
end

# ╔═╡ 00160003-0000-4000-8000-000000000001
let
	ground_truth = Dict(
		"Healthy GM" => (cbf=54.50, cbv=5.20, mtt=5.72),
		"Healthy WM" => (cbf=22.20, cbv=2.70, mtt=7.30),
		"Ischemic GM" => (cbf=10.90, cbv=1.77, mtt=9.72),
		"Ischemic WM" => (cbf=4.44, cbv=0.84, mtt=11.30),
	)

	_m(v) = isempty(v) ? 0.0 : mean(v)
	computed = Dict(
		"Healthy GM"  => (cbf=_m(healthy_gm_cbf),  cbv=_m(healthy_gm_cbv),  mtt=_m(healthy_gm_mtt)),
		"Ischemic GM" => (cbf=_m(ischemic_gm_cbf), cbv=_m(ischemic_gm_cbv), mtt=_m(ischemic_gm_mtt)),
		"Healthy WM"  => (cbf=_m(healthy_wm_cbf),  cbv=_m(healthy_wm_cbv),  mtt=_m(healthy_wm_mtt)),
		"Ischemic WM" => (cbf=_m(ischemic_wm_cbf), cbv=_m(ischemic_wm_cbv), mtt=_m(ischemic_wm_mtt)),
	)

	f = CM.Figure(size=(1100, 450), fontsize=12)

	labels = ["Healthy\nGM\n(GT)", "Healthy\nGM\n(Sim)", "Ischemic\nGM\n(GT)", "Ischemic\nGM\n(Sim)",
	          "Healthy\nWM\n(GT)", "Healthy\nWM\n(Sim)", "Ischemic\nWM\n(GT)", "Ischemic\nWM\n(Sim)"]
	colors = [:steelblue, :darkorange, :steelblue, :darkorange,
	          :steelblue, :darkorange, :steelblue, :darkorange]

	ax1 = CM.Axis(f[1, 1], title="CBF (mL/min/100g)", xticks=(1:8, labels), xticklabelrotation=0.0, xticklabelsize=9)
	CM.barplot!(ax1, 1:8, [
		ground_truth["Healthy GM"].cbf, computed["Healthy GM"].cbf,
		ground_truth["Ischemic GM"].cbf, computed["Ischemic GM"].cbf,
		ground_truth["Healthy WM"].cbf, computed["Healthy WM"].cbf,
		ground_truth["Ischemic WM"].cbf, computed["Ischemic WM"].cbf]; color=colors)

	ax2 = CM.Axis(f[1, 2], title="CBV (mL/100g)", xticks=(1:8, labels), xticklabelrotation=0.0, xticklabelsize=9)
	CM.barplot!(ax2, 1:8, [
		ground_truth["Healthy GM"].cbv, computed["Healthy GM"].cbv,
		ground_truth["Ischemic GM"].cbv, computed["Ischemic GM"].cbv,
		ground_truth["Healthy WM"].cbv, computed["Healthy WM"].cbv,
		ground_truth["Ischemic WM"].cbv, computed["Ischemic WM"].cbv]; color=colors)

	ax3 = CM.Axis(f[1, 3], title="MTT (s)", xticks=(1:8, labels), xticklabelrotation=0.0, xticklabelsize=9)
	CM.barplot!(ax3, 1:8, [
		ground_truth["Healthy GM"].mtt, computed["Healthy GM"].mtt,
		ground_truth["Ischemic GM"].mtt, computed["Ischemic GM"].mtt,
		ground_truth["Healthy WM"].mtt, computed["Healthy WM"].mtt,
		ground_truth["Ischemic WM"].mtt, computed["Ischemic WM"].mtt]; color=colors)

	CM.Label(f[0, 1:3], "Ground Truth (blue) vs Simulated (orange)", fontsize=14)
	CM.save(joinpath(RESULTS_DIR, "perfusion_vs_ground_truth.png"), f, px_per_unit=2)
	f
end

# ╔═╡ 00170001-0000-4000-8000-000000000001
md"""
## 11. Export Raw Files

Open in ImageJ with: width=nx, height=ny, nSlices=nz, 32-bit float, little-endian.
"""

# ╔═╡ 00170002-0000-4000-8000-000000000001
begin
	RAW_DIR = joinpath(RESULTS_DIR, "raw")
	mkpath(RAW_DIR)
	for (i, t_s) in enumerate(CONTRAST_TIME_S)
		vol = all_fdk_hu[i]
		nx, ny, nz = size(vol)
		vol_ij = vol[:, end:-1:1, :]  # flip Y for ImageJ
		fname = "brain_fdk_t$(t_s)s_$(nx)x$(ny)x$(nz).raw"
		open(joinpath(RAW_DIR, fname), "w") do io
			write(io, vec(vol_ij))
		end
	end
	println("RAW files saved to: $RAW_DIR")
end

# ╔═╡ 00180001-0000-4000-8000-000000000001
md"""
## Summary

| Parameter | Value |
|-----------|-------|
| Phantom | P1/P2 XCAT Brain, 400×400×400, 1 mm voxels |
| Scanner | GE Revolution Apex, SID=625.6 mm, SDD=1100 mm |
| Protocol | 120 kVp, 300 mA, 984 views, 0.5 s rotation, $(round(brain_z_cm * 10; digits=1)) mm collimation |
| Detector | $(brain_det_rows) rows × $(brain_det_cols) cols, 0.625 mm rows, 1.0 mm cols |
| Recon FOV | $(brain_recon_fov) cm, $(brain_recon_xy)×$(brain_recon_xy)×$(brain_n_slices) matrix |
| μ\_water (120 kVp) | $(round(μ_water_brain, sigdigits=4)) cm⁻¹ (spectrum-analytical BHC) |
| BHC | Two-stage: sinogram-domain + image-domain (notebook 00 pattern) |
| Pre-computed time points | $(join(CONTRAST_TIME_S, ", ")) s |
| Reconstruction | FDK with custom filter (matching notebook 00) |
| Perfusion method | Maximum slope (CBF), area ratio (CBV), central volume (MTT) |

**Physics (fidelity=:high):** fill factor, detector efficiency, scatter + scatter correction, crosstalk, focal spot, Poisson noise, lag, heel effect, bowtie filter
"""

# ╔═╡ 00180002-0000-4000-8000-000000000001
md"""
**Authors / Attribution:**
- Shu Nie (nies1@uci.edu) — BasisSimulator integration & perfusion analysis
- Caedin Miller (caedinm@uci.edu) — Original brain perfusion workflow
- Data: Sarah E. Divel et al. (2021), *Med. Phys.*, https://doi.org/10.1002/mp.14887
"""

# ╔═╡ Cell order:
# ╠═00010001-0000-4000-8000-000000000001
# ╠═00010002-0000-4000-8000-000000000001
# ╠═00010003-0000-4000-8000-000000000001
# ╠═00010004-0000-4000-8000-000000000001
# ╠═00010005-0000-4000-8000-000000000001
# ╠═00010006-0000-4000-8000-000000000001
# ╠═00010007-0000-4000-8000-000000000001
# ╠═00010008-0000-4000-8000-000000000001
# ╠═00010009-0000-4000-8000-000000000001
# ╠═00010010-0000-4000-8000-000000000001
# ╠═00010011-0000-4000-8000-000000000001
# ╟─00020001-0000-4000-8000-000000000001
# ╠═00020002-0000-4000-8000-000000000001
# ╟─00030001-0000-4000-8000-000000000001
# ╠═00030002-0000-4000-8000-000000000001
# ╟─00040001-0000-4000-8000-000000000001
# ╠═00040002-0000-4000-8000-000000000001
# ╠═00040003-0000-4000-8000-000000000001
# ╠═00040004-0000-4000-8000-000000000001
# ╟─00040005-0000-4000-8000-000000000001
# ╟─00050001-0000-4000-8000-000000000001
# ╠═00050002-0000-4000-8000-000000000001
# ╠═00050003-0000-4000-8000-000000000001
# ╟─00050004-0000-4000-8000-000000000001
# ╟─00060001-0000-4000-8000-000000000001
# ╠═00060002-0000-4000-8000-000000000001
# ╠═00060003-0000-4000-8000-000000000001
# ╟─00060004-0000-4000-8000-000000000001
# ╟─00060005-0000-4000-8000-000000000001
# ╟─00060006-0000-4000-8000-000000000001
# ╟─00060007-0000-4000-8000-000000000001
# ╟─00060008-0000-4000-8000-000000000001
# ╟─00070001-0000-4000-8000-000000000001
# ╠═00070002-0000-4000-8000-000000000001
# ╠═00070003-0000-4000-8000-000000000001
# ╠═00070004-0000-4000-8000-000000000001
# ╠═00070005-0000-4000-8000-000000000001
# ╠═00070006-0000-4000-8000-000000000001
# ╠═00070007-0000-4000-8000-000000000001
# ╟─00080001-0000-4000-8000-000000000001
# ╠═00080002-0000-4000-8000-000000000001
# ╠═00080002-a000-4000-8000-000000000001
# ╠═00080002-b000-4000-8000-000000000001
# ╠═00080003-0000-4000-8000-000000000001
# ╟─00080004-0000-4000-8000-000000000001
# ╟─00090001-0000-4000-8000-000000000001
# ╠═00090002-0000-4000-8000-000000000001
# ╟─00100001-0000-4000-8000-000000000001
# ╠═00100002-0000-4000-8000-000000000001
# ╟─00110001-0000-4000-8000-000000000001
# ╟─00110002-0000-4000-8000-000000000001
# ╟─00110003-0000-4000-8000-000000000001
# ╟─00110004-0000-4000-8000-000000000001
# ╟─00110005-0000-4000-8000-000000000001
# ╟─00120001-0000-4000-8000-000000000001
# ╟─00120002-0000-4000-8000-000000000001
# ╟─00120003-0000-4000-8000-000000000001
# ╟─00120004-0000-4000-8000-000000000001
# ╟─00120005-0000-4000-8000-000000000001
# ╟─00120006-0000-4000-8000-000000000001
# ╟─00120007-0000-4000-8000-000000000001
# ╟─00120008-0000-4000-8000-000000000001
# ╟─00120009-0000-4000-8000-000000000001
# ╠═00120010-0000-4000-8000-000000000001
# ╟─00130001-0000-4000-8000-000000000001
# ╟─00130002-0000-4000-8000-000000000001
# ╠═00130003-0000-4000-8000-000000000001
# ╠═00130004-0000-4000-8000-000000000001
# ╠═00130005-0000-4000-8000-000000000001
# ╟─00130006-0000-4000-8000-000000000001
# ╟─00140001-0000-4000-8000-000000000001
# ╟─00140002-0000-4000-8000-000000000001
# ╟─00140003-0000-4000-8000-000000000001
# ╠═00140004-0000-4000-8000-000000000001
# ╟─00150001-0000-4000-8000-000000000001
# ╟─00150002-0000-4000-8000-000000000001
# ╟─00150003-0000-4000-8000-000000000001
# ╟─00150004-0000-4000-8000-000000000001
# ╟─00160001-0000-4000-8000-000000000001
# ╟─00160002-0000-4000-8000-000000000001
# ╟─00160003-0000-4000-8000-000000000001
# ╟─00170001-0000-4000-8000-000000000001
# ╟─00170002-0000-4000-8000-000000000001
# ╟─00180001-0000-4000-8000-000000000001
# ╟─00180002-0000-4000-8000-000000000001
