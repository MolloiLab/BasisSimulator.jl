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

# ╔═╡ d6d62fae-012d-11f1-1efc-67e7f251ff8c
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.resolve()
    Pkg.instantiate()
end

# ╔═╡ 18a569fd-c6a5-49e9-b7ea-6d27fe4a4df4
using Revise

# ╔═╡ 2ab74942-1680-47dd-a8dc-f5242387253e
# ╠═╡ show_logs = false
using Unitful: @u_str, ustrip

# ╔═╡ b4370b8e-b684-11f0-3520-1713b81c9b2f
using MAT

# ╔═╡ 953ef431-2eb3-413d-aadf-f9b5bfff640a
using Statistics

# ╔═╡ 00000001-0000-0000-0000-000000000001
md"""
# Notebook 07: Brain Perfusion CT Simulation — P1/P2 XCAT Phantom

This notebook simulates dynamic contrast-enhanced brain CT using the XCAT P1 (material) and P2 (segment) phantoms, replicating the `Dynamic_Contrast_Addition.jl` workflow with the BasisSimulator Julia API.

**Workflow:**
1. Load P1/P2 XCAT brain phantom raw files (400×400×400, UInt16)
2. Apply time-varying iodine contrast via `update_structures!` (arteries, veins, gray matter, white matter)
3. Build a `BS.Phantom` with native UInt16 mask (900+ segment IDs passed directly)
4. Water phantom calibration (per scanner/kVp)
5. Pre-compute all 6 time points (0, 5, 10, 15, 20, 25 s): CT simulation + FDK + Hybrid IR
6. Interactive visualizations: phantom anatomy, HU images, time-attenuation curves — slider responds instantly
"""

# ╔═╡ 00891cd0-96da-4f83-9f8d-c857259ed5d7
# ╠═╡ disabled = true
#=╠═╡
Pkg.add("MAT")
  ╠═╡ =#

# ╔═╡ cca08041-b05c-4045-a462-18b30fd0559f
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ 3ad61ec7-ba66-449d-8fd1-e79a2345a9d7
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 3cbd1220-7118-42c7-9562-27c8c2e1b608
import CairoMakie as CM

# ╔═╡ 2a00221a-e861-4d53-bba6-bde7b1bc909f
# ╠═╡ show_logs = false
# Metal is loaded automatically by BS.gpu_array_type() — no explicit import needed

# ╔═╡ dc9e51f9-2531-4483-b80a-622f5ecf4d0c
import XrayAttenuation as XA

# ╔═╡ f36f0c25-3c0c-4202-b890-917b031fa9e8
import Statistics: mean, std

# ╔═╡ 4ca1063f-1cc1-4253-a411-4817f1e584a2
UI.TableOfContents()

# ╔═╡ 20920020-fd6b-4a64-9e19-e00bfd616ee6
md"""
## 1. Paths & Configuration
> **Large data files are not tracked in git.**
> Phantom data from: Sarah E. Divel et al., "A dynamic simulation framework for CT perfusion in stroke
> assessment built from first principles," *Med. Phys.* 2021. https://doi.org/10.1002/mp.14887
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

# ╔═╡ c744a9d3-5810-4465-82ee-2b8d9b5f68b1
begin
	PHANTOM_DIR    = joinpath(dirname(@__DIR__), "data", "brain_perfusion")
	P1_RAW_PATH    = joinpath(PHANTOM_DIR, "P1_brain_all_2020_RAW_400_400_400.raw")
	P1_TABLE_PATH  = joinpath(PHANTOM_DIR, "P1_voxelize_table.txt")
	P2_RAW_PATH    = joinpath(PHANTOM_DIR, "P2_brain_all_2020_RAW_400_400_400.raw")
	P2_TABLE_PATH  = joinpath(PHANTOM_DIR, "P2_vozelize_table.txt")
	STRUCT_INFO_PATH  = joinpath(PHANTOM_DIR, "structure_info.mat")
	IODINE_DATA_PATH  = joinpath(PHANTOM_DIR, "iodine_mass_data.mat")
	FIGURES_DIR       = joinpath(dirname(@__DIR__), "figures", "brain_perfusion")
	mkpath(FIGURES_DIR)

	@assert isfile(P1_RAW_PATH)       "P1 raw file not found: $P1_RAW_PATH"
	@assert isfile(P1_TABLE_PATH)     "P1 table not found: $P1_TABLE_PATH"
	@assert isfile(P2_RAW_PATH)       "P2 raw file not found: $P2_RAW_PATH"
	@assert isfile(P2_TABLE_PATH)     "P2 table not found: $P2_TABLE_PATH"
	@assert isfile(STRUCT_INFO_PATH)  "structure_info.mat not found: $STRUCT_INFO_PATH"
	@assert isfile(IODINE_DATA_PATH)  "iodine_mass_data.mat not found: $IODINE_DATA_PATH"

	"All input paths verified ✓"
end

# ╔═╡ 00000004-0000-0000-0000-000000000001
md"""
## 2. Load P1 / P2 Raw Files
"""

# ╔═╡ 1a2f50df-3f2a-49a2-bf9f-f197237fe9ce
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

# ╔═╡ 71d5cf58-45e1-45bb-8f13-33ad5283c89f
P2_raw_file = let
	buf = Array{UInt16}(undef, 400, 400, 400)
	open(P2_RAW_PATH, "r") do io
		read!(io, buf)
	end
	arr = Int.(buf)
	reverse(arr, dims=2)   # flip y: Julia col-major vs ImageJ row-major
end;

# ╔═╡ 7140bca6-1b57-40db-9b4a-f48132879b93
(P1_structure_map, P2_structure_map) = (BS.load_structure_map(P1_TABLE_PATH), BS.load_structure_map(P2_TABLE_PATH))

# ╔═╡ 00000005-0000-0000-0000-000000000001
md"""
**P1:** $(length(unique(vec(P1_raw_file)))) unique material label IDs

**P2:** $(length(unique(vec(P2_raw_file)))) unique segment IDs
"""

# ╔═╡ 00000006-0000-0000-0000-000000000001
md"""
## 3. Load Materials
These IDs come from `P1_voxelize_table.txt` (tab-separated `name → ID`):

| ID | Material | Example structures |
|----|----------|--------------------|
| 0  | `:air` | background, oral cavity, tendons, spinal cord |
| 1  | `:muscle` | scalp/neck muscles, tongue, parotid glands, eyes, optic nerves |
| 2  | `:air` | throat (airway passages) |
| 3  | `:bone` | cervical spine (atlas, axis, C3–C5) |
| 5  | `:soft_tissue` | head surface, ears |
| 10 | `:soft_tissue` | interior zero islands (relabeled by `relabel_zero_islands_2d!`) |
| 13 | `:bone` | skull (frontal, temporal, parietal, occipital, mandible) |
| 14 | `:soft_tissue` | intervertebral disks (disk1–disk3) |
| 17 | `:csf` | ventricles (lateral, third, fourth), cerebral aqueduct |
| 18 | `:gray_matter` | brain parenchyma + 78 named GM segments (2001\_gm\_*) |
| 19 | `:white_matter` | cerebral lobes + 117 named WM segments (3001\_wm\_*) |
| 21 | `:blood` | arteries: 399 segments (internal carotid, MCA, ACA, basilar, vertebral, …) |
| 22 | `:blood` | veins: 235 segments (jugular, sagittal/transverse/straight sinus, cerebral veins, …) |

"""

# ╔═╡ b7161fac-2eda-41be-93d4-1162587050cd
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

# ╔═╡ 242bc1e2-d806-4cee-b9b6-d4726c3696b8
(material_map_init, material_list_init) = let
	unique_ids    = sort(unique(vec(P1_raw_file)))
	material_map  = Dict{Int, Symbol}(k => v for (k, v) in MATERIAL_MAP_BASE if k in unique_ids)
	material_list = Dict{Symbol, XA.Material}(
		sym => BS.get_material(sym) for sym in unique(collect(values(material_map)))
	)
	(material_map, material_list)
end

# ╔═╡ 00000007-0000-0000-0000-000000000001
md"""
**Base materials loaded:** $(length(material_list_init))

Symbols: $(sort(collect(keys(material_list_init)), by=string))
"""

# ╔═╡ 00000008-0000-0000-0000-000000000001
md"""
## 4. Load Iodine Contrast Data
"""

# ╔═╡ 2959ddd3-d949-40a0-a0ab-ad4e0c9a4239
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

# ╔═╡ 207000f3-cd92-40b1-8c5e-bba098d4b79c
(iodine_artery, iodine_vein, iodine_gm, iodine_wm) = let
	d = matread(IODINE_DATA_PATH)
	(Float64.(d["mass_arteries"]),
	 Float64.(d["mass_vein"]),
	 Float64.(d["mass_gm"]),
	 Float64.(d["mass_wm"]))
end

# ╔═╡ 00000009-0000-0000-0000-000000000002
md"""
**Artery segments:** $(size(iodine_artery, 1)) | **Vein segments:** $(size(iodine_vein, 1))

**GM segments:** $(size(iodine_gm, 1)) | **WM segments:** $(size(iodine_wm, 1))

**Time points:** $(size(iodine_artery, 2)) (0–85 s at 1 ms resolution)
"""

# ╔═╡ 00000015-0000-0000-0000-000000000001
md"""
## 5. Scanner & Protocol — GE Revolution Apex (120 kVp)
"""

# ╔═╡ a70bc722-e769-4132-b082-b0e89a68228e
begin
	brain_extent_mm = 400 * 0.1 * 10.0        # phantom FOV in mm (400 vox × 1 mm/vox)
	brain_eict_mag  = 1100.0 / 625.6           # magnification SDD/SID
	brain_det_cols  = ceil(Int, brain_extent_mm * brain_eict_mag / 1.0)
	# GE Revolution Apex: 256 rows × 0.625 mm pitch
	brain_det_rows  = 256
	brain_recon_fov = 40.0  # cm (matches 400-vox × 0.1 cm phantom)
	brain_vox_cm    = 0.1   # phantom voxel size (cm)
	# Auto-detect tissue extent along dim3 (the slice axis) from P1 phantom.
	# Pad by 4 slices on each side, clamped to array bounds.
	_z_any       = vec(any(P1_raw_file .!= 0, dims=(1,2)))
	_z_first     = findfirst(_z_any)
	_z_last      = findlast(_z_any)
	BRAIN_Z_CROP = max(1, _z_first - 4) : min(size(P1_raw_file, 3), _z_last + 4)
	brain_z_cm   = length(BRAIN_Z_CROP) * brain_vox_cm
	brain_recon_xy = 512
	brain_n_slices = round(Int, brain_z_cm / brain_vox_cm)
end

# ╔═╡ 00000006-0000-0000-0000-000000000002
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
	detector_material     = :gos,
	detector_depth        = 3.0,
	fill_factor_row       = 0.9,
	fill_factor_col       = 0.9,
	detection_gain        = 1.0,
)

# ╔═╡ a70bc722-e769-4132-b082-b0e89a68228f
protocol_brain = BS.CTProtocol(
	kVp           = 120.0,
	mA            = 300.0,
	views         = 984,
	rotation_time = 0.5,
)

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822b1
sim_opts_brain = BS.SimOptions(fidelity = :low, seed = 42)

# ╔═╡ 00000011-0000-0000-0000-000000000002
	recon_opts_brain = BS.ReconOptions(
	algorithm   = :fdk,
	matrix_size = (brain_recon_xy, brain_recon_xy, brain_n_slices),
	fov_cm      = brain_recon_fov,
	z_cm        = brain_z_cm,   # explicit z-FOV = cropped phantom extent
	)

# ╔═╡ a0b1c2d3-e4f5-6789-abcd-000000000001
md"""
## 6. Water Phantom Calibration
"""

# ╔═╡ a0b1c2d3-e4f5-6789-abcd-000000000002
md"""
### Analytical μ_water from NIST (post-filter spectrum)

Compute μ_water directly from NIST attenuation data using the post-filter
effective spectrum.  This avoids the systematic overestimation introduced by
the flat-filter / air-scan mismatch in the signal chain.

Method:
1. Load 120 kVp spectrum and downsample to 30 bins
2. Apply Al flat-filter transmission (2.5 mm) energy-by-energy
3. Compute spectrum-weighted effective energy: E_eff = Σ(w·E) / Σw
4. Look up μ_water(E_eff) from NIST via `compute_μ_at_energy`

This gives the physically correct value (~0.19–0.21 cm⁻¹ at 120 kVp).
"""

# ╔═╡ d9dfaa24-2254-4953-993f-f9fdb0c3326d
μ_water_brain = let
	# 1. Load and downsample spectrum
	energies_raw, weights_raw = BS.load_spectrum(120)
	energies, weights = BS.downsample_spectrum(energies_raw, weights_raw, 30)

	# 2. Apply 2.5 mm Al flat-filter transmission (same as scanner_brain)
	filter_mat_str  = "Al"
	filter_t_cm     = scanner_brain.flat_filter_thickness / 10.0  # mm → cm
	filter_trans    = [exp(-BS.get_bowtie_mu(filter_mat_str, Float64(e)) * filter_t_cm)
	                   for e in energies]
	filtered_weights = weights .* filter_trans

	# 3. Spectrum-weighted effective energy
	E_eff = sum(energies .* filtered_weights) / sum(filtered_weights)

	# 4. NIST lookup at effective energy
	BS.compute_μ_at_energy(XA.Materials.water, E_eff)
end

# ╔═╡ a0b1c2d3-e4f5-6789-abcd-000000000004
md"""
**μ_water @ 120 kVp (NIST, post-filter):** $(round(μ_water_brain, sigdigits=4)) cm⁻¹
"""

# ╔═╡ 00000016-0000-0000-0000-000000000001
md"""
## 7. Pre-compute: All Time Points

Runs all 6 time points upfront (0, 5, 10, 15, 20, 25 s). After this cell completes the
slider responds instantly — no re-simulation on every move.

**Performance optimizations:**
- `segment_index_map` is built once from `P2_raw_crop` before the time loop (eliminates 834 `findall` scans × 6 time points = 5004 redundant full-array scans).
- `ws.μ_table` is recomputed in-place at the start of each `simulate!` call from the current materials, restoring the fast table-lookup path in `create_μ_volume!` (avoids 900+ NIST XCOM calls × 30 energy bins per GPU kernel).
- `mask_gpu` (UInt16, GPU) is uploaded once and shared across all time points — only `materials_dict` changes.
"""

# ╔═╡ 00000017-0000-0000-0000-000000000001
begin
	# Pre-compute: simulate and reconstruct all 6 contrast time points.
	# Results stored as plain CPU arrays — slider only indexes into these dicts.
	const CONTRAST_TIME_S = [0, 5, 10, 15, 20, 25]
	const CONTRAST_INDICES = CONTRAST_TIME_S .* 1000 .+ 1   # → [1, 5001, 10001, 15001, 20001, 25001]
	
	all_fdk_hu = Dict{Int, Array{Float32, 3}}()
	all_hir_hu = Dict{Int, Array{Float32, 3}}()
	all_contrast_phantom = Dict{Int, Array{Int, 3}}()
	
	let
		recon_size = (brain_recon_xy, brain_recon_xy, brain_n_slices)
		# BRAIN_Z_CROP is defined in the scanner/recon parameters cell above.
		P1_stamped   = copy(P1_raw_file)[:, :, BRAIN_Z_CROP]
		P2_raw_crop  = P2_raw_file[:, :, BRAIN_Z_CROP]

		# Pre-build a segment→voxel index map once (avoids 834 findall scans × 6 time points).
		# Keys are P2 segment IDs whose names start with "2" (GM), "3" (WM), "4" (vein), "5" (artery).
		segment_index_map = Dict{Int, Vector{CartesianIndex{3}}}()
		for (id, name) in P2_structure_map
			first(name) in ('2', '3', '4', '5') || continue
			idxs = findall(==(id), P2_raw_crop)
			isempty(idxs) || (segment_index_map[id] = idxs)
		end

		# Stamp all segment IDs into P1 once — voxel labels are time-invariant.
		for (id, idxs) in segment_index_map
			P1_stamped[idxs] .= id
		end

		mask_gpu = BS.gpu_array_type()(UInt16.(P1_stamped))   # upload once, reused

		# Helper: rebuild materials_dict for one time point (no array copy needed).
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

			# Some P2 segments are stamped into the mask but skipped by update_structures! when
			# their row index is out of bounds in the iodine table (e.g. segments 5398, 5399).
			# Fill them with their base material so all mask labels have a materials entry.
			_prefix_to_base = Dict("5" => :blood, "4" => :blood, "3" => :white_matter, "2" => :gray_matter)
			for (id, name) in P2_structure_map
				id ∈ keys(materials_dict) && continue
				id ∈ keys(segment_index_map) || continue
				c = string(first(name))
				c ∈ keys(_prefix_to_base) || continue
				materials_dict[id] = material_list_init[_prefix_to_base[c]]
			end
			BS.Phantom(mask_gpu, materials_dict, (0.1, 0.1, 0.1))
			end

		# Build first phantom to size the workspace.
		phantom_t_0 = build_phantom(CONTRAST_INDICES[1])
		for i in eachindex(CONTRAST_INDICES)
			all_contrast_phantom[i] = P1_stamped
		end

		ws = BS.create_eict_workspace(
			scanner_brain, protocol_brain,
			sim_opts_brain, recon_opts_brain, phantom_t_0
		)
		geom = ws.geom

		# Run once to get GPU sinogram, then create workspaces once (weights on GPU)
		BS.simulate!(ws, phantom_t_0, scanner_brain, protocol_brain, sim_opts_brain, recon_opts_brain)

		# ── GPU profiler: one dedicated profile run at t=0 ──────────────────
		# Runs profile_simulate! once (after JIT warmup above) to capture
		# per-stage wall times. Results printed to notebook output.
		begin
			_prof = BS.GPUProfiler()
			BS.profile_simulate!(ws, phantom_t_0, scanner_brain, protocol_brain,
				sim_opts_brain, recon_opts_brain; profiler=_prof)
			BS.print_profile(_prof)
		end
		sino_gpu = ws.sinogram  # GPU sinogram
		ws_fdk = BS.create_fdk_recon_workspace(sino_gpu, geom, recon_size)
		ws_hir = BS.create_hir_recon_workspace(sino_gpu, geom, recon_size; strength = 2)

		for (i, t_contrast) in enumerate(CONTRAST_INDICES)
			println("▶ Time point $i/$(length(CONTRAST_INDICES)): t = $(CONTRAST_TIME_S[i]) s  (index $t_contrast)")

			phantom_t = i == 1 ? phantom_t_0 : build_phantom(t_contrast)

			@time begin
				# Run simulation (writes to ws.sinogram in-place on GPU)
				BS.simulate!(
					ws, phantom_t, scanner_brain, protocol_brain,
					sim_opts_brain, recon_opts_brain
				)

				# Use GPU sinogram directly — no GPU→CPU copy
				sino = ws.sinogram

				# --- FDK reconstruction (reuses ws_fdk) ---
				# reverse(dims=3): recon iz=1 maps to vol_min_z (inferior),
				# but phantom dim3=1 is superior. Flip to match anatomical order.
				all_fdk_hu[i] = reverse(BS.to_hounsfield(
					Array(BS.reconstruct!(ws_fdk, sino, geom, recon_size));
					μ_water = μ_water_brain
				), dims=3)
				# --- Hybrid IR (reuses ws_hir) ---
				all_hir_hu[i] = reverse(BS.to_hounsfield(
					Array(BS.reconstruct!(ws_hir, sino, geom, recon_size));
					μ_water = μ_water_brain
				), dims=3)
				phantom_t = nothing
			end
			println("   ✓ done t=$(CONTRAST_TIME_S[i]) s")
		end
		ws_fdk = nothing; ws_hir = nothing; ws = nothing
		GC.gc(true)
		println("All time points complete.")
	end;
end

# ╔═╡ 00000018-0000-0000-0000-000000000001
# Export ImageJ-compatible .raw files for all time points.
# Julia arrays are column-major (x fastest in memory). ImageJ raw import also
# reads the first dimension as fastest-varying, so write vec(vol) directly and
# open in ImageJ with: width=nx, height=ny, nSlices=nz, 32-bit float, little-endian.
# No permutation needed — permutedims would rotate the image 90°.
# Filename: brain_<recon>_t<seconds>s_<nx>x<ny>x<nz>.raw
begin
	RAW_DIR = joinpath(dirname(@__DIR__), "figures", "brain_perfusion", "raw")
	mkpath(RAW_DIR)
	for (i, t_s) in enumerate(CONTRAST_TIME_S)
		for (name, vols) in [("fdk", all_fdk_hu), ("hir", all_hir_hu)]
			vol = vols[i]
			nx, ny, nz = size(vol)
			fname = "brain_$(name)_t$(t_s)s_$(nx)x$(ny)x$(nz).raw"
			open(joinpath(RAW_DIR, fname), "w") do io
				write(io, vec(vol))  # little-endian Float32 (native on x86/ARM); open in ImageJ as width=nx height=ny nSlices=nz
			end
		end
	end
	println("RAW files saved to: $RAW_DIR")
end

# ╔═╡ 00000024-0000-0000-0000-000000000001
md"""
## 8. Visualizations
"""

# ╔═╡ 01c4dd1a-a5b7-4b89-bab8-80152fa5da9f
md"""
### 8.1 Phantoms
"""

# ╔═╡ 4136f466-fad8-4325-ae51-9632efb00e07
md"""
Select phantom z slice:
"""

# ╔═╡ 237bd8ca-6ea5-4aff-81f4-8a77bb45f4fd
@bind z_preview UI.Slider(1:brain_n_slices; default=90, show_value=true)

# ╔═╡ 91139300-7464-4a84-a22c-180e97b8e692
md"""
### 8.2 CT scans
"""

# ╔═╡ e3ed6608-c9d5-4970-ade1-e622bd712674
md"""
Select time point index (1 = 0 s · 2 = 5 s · 3 = 10 s · 4 = 15 s · 5 = 20 s · 6 = 25 s):
"""

# ╔═╡ e3ed6608-c9d5-4970-ade1-e622bd712675
@bind t_idx UI.Slider(1:6; default=3, show_value=true)

# ╔═╡ 00000011-0000-0000-0000-000000000001
md"""
**Selected time:** $(CONTRAST_TIME_S[t_idx]) s  (index $(CONTRAST_INDICES[t_idx]))
"""

# ╔═╡ 5dee3049-e56a-41dd-8148-fd9f5de3411b
let
	p1_slice  = P1_raw_file[:, :, z_preview]
	seg_slice = all_contrast_phantom[t_idx][:, :, z_preview]
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
		title  = "Contrast Phantom — Segment IDs at t = $(CONTRAST_TIME_S[t_idx]) s",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	CM.heatmap!(ax2, disp; colormap = :turbo, colorrange = (0, 1))

	f
end

# ╔═╡ b1479395-1af2-4684-895c-7227e0396a26
md"""
Select CT scan z slice:
"""

# ╔═╡ b6bc4ee2-20f3-4d84-a210-5608b0eab80b
@bind mid_Z UI.Slider(1:brain_n_slices; default=92, show_value=true)

# ╔═╡ 00000024-0000-0000-0000-000000000002
let
	fdk_hu = all_fdk_hu[t_idx]
	hir_hu = all_hir_hu[t_idx]
	t_s    = CONTRAST_TIME_S[t_idx]

	f     = CM.Figure(size=(1000, 480))

	ax1 = CM.Axis(f[1, 1],
		title  = "FDK — Brain Window (W=80, L=40)  t=$(t_s)s  z=$mid_Z",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	hm1 = CM.heatmap!(ax1, fdk_hu[:, :, mid_Z];
		colormap = :grays, colorrange = (0, 80))
	CM.Colorbar(f[1, 2], hm1; label = "HU")

	ax2 = CM.Axis(f[1, 3],
		title  = "Hybrid IR (str=2) — Brain Window  t=$(t_s)s",
		aspect = CM.DataAspect(),
		xticksvisible = false, yticksvisible = false,
		xticklabelsvisible = false, yticklabelsvisible = false,
	)
	hm2 = CM.heatmap!(ax2, hir_hu[:, :, mid_Z];
		colormap = :grays, colorrange = (0, 80))
	CM.Colorbar(f[1, 4], hm2; label = "HU")

	CM.save(joinpath(FIGURES_DIR, "nb07_brain_fdk_vs_hir.png"), f)
	f
end

# ╔═╡ 00000024-0000-0000-0000-000000000003
let
	fdk_hu = all_fdk_hu[t_idx]
	hir_hu = all_hir_hu[t_idx]
	t_s    = CONTRAST_TIME_S[t_idx]

	f     = CM.Figure(size=(1000, 480))

	for (col, vol, title) in [
		(1, fdk_hu, "FDK — Bone Window  t=$(t_s)s"),
		(3, hir_hu, "Hybrid IR — Bone Window  t=$(t_s)s"),
	]
		ax = CM.Axis(f[1, col],
			title  = title,
			aspect = CM.DataAspect(),
			xticksvisible = false, yticksvisible = false,
			xticklabelsvisible = false, yticklabelsvisible = false,
		)
		hm = CM.heatmap!(ax, vol[:, :, mid_Z];
			colormap = :grays, colorrange = (-500, 700))
		CM.Colorbar(f[1, col+1], hm; label = "HU")
	end

	CM.save(joinpath(FIGURES_DIR, "nb07_brain_bone_window.png"), f)
	f
end

# ╔═╡ 00000024-0000-0000-0000-000000000006
md"""
### 8.3 Time-Attenuation Curves (TAC)
"""

# ╔═╡ 00000024-0000-0000-0000-000000000007
tac_data = let
	tac_indices = round.(Int, range(1, 85001; length=15))
	tac_sec     = (tac_indices .- 1) ./ 1000.0
	artery_mean = [mean(iodine_artery[:, t]) for t in tac_indices]
	vein_mean   = [mean(iodine_vein[:,   t]) for t in tac_indices]
	gm_mean     = [mean(iodine_gm[:,     t]) for t in tac_indices]
	wm_mean     = [mean(iodine_wm[:,     t]) for t in tac_indices]
	(tac_sec, artery_mean, vein_mean, gm_mean, wm_mean)
end

# ╔═╡ 00000024-0000-0000-0000-000000000008
let
	tac_sec, artery, vein, gm, wm = tac_data
	t_sel_sec = Float64(CONTRAST_TIME_S[t_idx])

	f  = CM.Figure(size=(800, 420))
	ax = CM.Axis(f[1, 1],
		title  = "Time-Iodine Curves — mean concentration per tissue type",
		xlabel = "Time (s)",
		ylabel = "Iodine Concentration (mg/mL or mg/g)",
	)
	CM.lines!(ax, tac_sec, artery; color=:red,    linewidth=2, label="Artery (mg/mL)")
	CM.lines!(ax, tac_sec, vein;   color=:blue,   linewidth=2, label="Vein (mg/mL)")
	CM.lines!(ax, tac_sec, gm;     color=:green,  linewidth=2, label="Gray Matter (mg/g)")
	CM.lines!(ax, tac_sec, wm;     color=:orange, linewidth=2, label="White Matter (mg/g)")
	CM.vlines!(ax, [t_sel_sec]; color=:black, linestyle=:dash, linewidth=1.5, label="Selected t")
	CM.axislegend(ax; position=:rt)

	CM.save(joinpath(FIGURES_DIR, "nb07_brain_tac.png"), f)
	f
end

# ╔═╡ 00000025-0000-0000-0000-000000000001
md"""
## Summary

| Parameter | Value |
|-----------|-------|
| Phantom | P1/P2 XCAT Brain, 400×400×400, 1 mm voxels |
| Scanner | GE Revolution Apex, SID=625.6 mm, SDD=1100 mm |
| Protocol | 120 kVp, 300 mA, 984 views, 0.5 s rotation |
| Detector | $(brain_det_rows) rows × $(brain_det_cols) cols, 0.625 mm rows, 1.0 mm cols |
| Recon FOV | $(brain_recon_fov) cm, $(brain_recon_xy)×$(brain_recon_xy)×$(brain_n_slices) matrix |
| μ_water (120 kVp) | ~$(round(μ_water_brain, sigdigits=4)) cm⁻¹ |
| Pre-computed time points | $(join(CONTRAST_TIME_S, ", ")) s |
| Selected time | $(CONTRAST_TIME_S[t_idx]) s |

**Physics (fidelity=:high):** fill factor · flat filter · scatter · crosstalk · focal spot · Poisson noise · lag · heel effect · BHC
"""

# ╔═╡ 00000002-0000-0000-0000-000000000002
md"""
**Authors / Attribution:**
- Shu Nie (nies1@uci.edu) — BasisSimulator integration
- Caedin Miller (caedinm@uci.edu) — Original brain perfusion workflow (Dynamic\_Contrast\_Addition.jl)
- Data: Sarah E. Divel et al. (2021), *Med. Phys.*, https://doi.org/10.1002/mp.14887
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000001
# ╠═d6d62fae-012d-11f1-1efc-67e7f251ff8c
# ╠═00891cd0-96da-4f83-9f8d-c857259ed5d7
# ╠═18a569fd-c6a5-49e9-b7ea-6d27fe4a4df4
# ╠═cca08041-b05c-4045-a462-18b30fd0559f
# ╠═3ad61ec7-ba66-449d-8fd1-e79a2345a9d7
# ╠═3cbd1220-7118-42c7-9562-27c8c2e1b608
# ╠═2a00221a-e861-4d53-bba6-bde7b1bc909f
# ╠═dc9e51f9-2531-4483-b80a-622f5ecf4d0c
# ╠═f36f0c25-3c0c-4202-b890-917b031fa9e8
# ╠═2ab74942-1680-47dd-a8dc-f5242387253e
# ╠═b4370b8e-b684-11f0-3520-1713b81c9b2f
# ╠═953ef431-2eb3-413d-aadf-f9b5bfff640a
# ╠═4ca1063f-1cc1-4253-a411-4817f1e584a2
# ╠═20920020-fd6b-4a64-9e19-e00bfd616ee6
# ╠═c744a9d3-5810-4465-82ee-2b8d9b5f68b1
# ╟─00000004-0000-0000-0000-000000000001
# ╠═1a2f50df-3f2a-49a2-bf9f-f197237fe9ce
# ╠═71d5cf58-45e1-45bb-8f13-33ad5283c89f
# ╠═7140bca6-1b57-40db-9b4a-f48132879b93
# ╟─00000005-0000-0000-0000-000000000001
# ╟─00000006-0000-0000-0000-000000000001
# ╠═b7161fac-2eda-41be-93d4-1162587050cd
# ╠═242bc1e2-d806-4cee-b9b6-d4726c3696b8
# ╟─00000007-0000-0000-0000-000000000001
# ╟─00000008-0000-0000-0000-000000000001
# ╠═2959ddd3-d949-40a0-a0ab-ad4e0c9a4239
# ╠═207000f3-cd92-40b1-8c5e-bba098d4b79c
# ╟─00000009-0000-0000-0000-000000000002
# ╠═00000015-0000-0000-0000-000000000001
# ╠═a70bc722-e769-4132-b082-b0e89a68228e
# ╠═00000006-0000-0000-0000-000000000002
# ╠═a70bc722-e769-4132-b082-b0e89a68228f
# ╠═a70bc722-e769-4132-b082-b0e89a6822b1
# ╠═00000011-0000-0000-0000-000000000002
# ╟─a0b1c2d3-e4f5-6789-abcd-000000000001
# ╟─a0b1c2d3-e4f5-6789-abcd-000000000002
# ╠═d9dfaa24-2254-4953-993f-f9fdb0c3326d
# ╟─a0b1c2d3-e4f5-6789-abcd-000000000004
# ╠═00000016-0000-0000-0000-000000000001
# ╠═00000017-0000-0000-0000-000000000001
# ╠═00000018-0000-0000-0000-000000000001
# ╠═00000024-0000-0000-0000-000000000001
# ╠═01c4dd1a-a5b7-4b89-bab8-80152fa5da9f
# ╟─4136f466-fad8-4325-ae51-9632efb00e07
# ╟─237bd8ca-6ea5-4aff-81f4-8a77bb45f4fd
# ╟─00000011-0000-0000-0000-000000000001
# ╟─5dee3049-e56a-41dd-8148-fd9f5de3411b
# ╟─91139300-7464-4a84-a22c-180e97b8e692
# ╟─e3ed6608-c9d5-4970-ade1-e622bd712674
# ╟─e3ed6608-c9d5-4970-ade1-e622bd712675
# ╟─b1479395-1af2-4684-895c-7227e0396a26
# ╟─b6bc4ee2-20f3-4d84-a210-5608b0eab80b
# ╟─00000024-0000-0000-0000-000000000002
# ╟─00000024-0000-0000-0000-000000000003
# ╟─00000024-0000-0000-0000-000000000006
# ╟─00000024-0000-0000-0000-000000000008
# ╟─00000024-0000-0000-0000-000000000007
# ╟─00000025-0000-0000-0000-000000000001
# ╠═00000002-0000-0000-0000-000000000002
