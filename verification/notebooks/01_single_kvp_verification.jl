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

# ╔═╡ 90ba5323-144a-4e09-8c8d-469115f01a95
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()

	using Revise
end

# ╔═╡ 0f1119d9-c5b9-47c0-9b96-48bff346ecef
# ╠═╡ show_logs = false
using PythonCall

# ╔═╡ f0000003-0001-0001-0001-000000000001
using Unitful: @u_str

# ╔═╡ e1b82d96-8029-451a-bf97-5534c9569330
using LinearAlgebra

# ╔═╡ ce79b686-5fde-47a1-820b-48779e79a384
using FFTW

# ╔═╡ 29122cf4-0dec-42f7-93a7-6b4b316093f2
using Random

# ╔═╡ 04ddc034-3cf7-4bda-9242-82bb6de4b598
# ╠═╡ show_logs = false
using Metal

# ╔═╡ 58bbe3ac-ef92-4a99-877e-716ee5f91750
md"""
# Verification of BasisSimulator.jl Against XCIST/CatSim

**A Publication-Quality Comparison of CT Simulation Frameworks**

*Single-kVp Verification Study*

---
"""

# ╔═╡ 8a6908cd-434a-4f18-be3e-a038786f6ef7
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ 3cfa2553-ab67-48cb-a179-7ad583b7f176
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 53b5042f-b653-4665-beae-6392b45de84e
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ 365fac15-be61-4fd3-89c1-acd3a63b3127
import Statistics: mean, std, cor

# ╔═╡ f0000002-0001-0001-0001-000000000001
import XrayAttenuation as XA

# ╔═╡ f0000001-0001-0001-0001-000000000001
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")

# ╔═╡ 1e5e1a0b-ab52-4b44-b17e-bff4c8e9802b
UI.TableOfContents()

# ╔═╡ 59dc49d1-875e-4288-a61b-cf33ada02fdd
md"""
## 1. Simulation Configuration

We define the configuration **once** here. Both `BasisSimulator` and `CatSim` will read from this single source of truth to ensure 1:1 parity.
"""

# ╔═╡ 45f217d7-1c9c-4623-a8ff-fe22787831e8
begin
	# --- Scanner Geometry ---
	sid = 540.0                 # Source-to-Iso (mm)
	sdd = 950.0                 # Source-to-Detector (mm)
	magnification = sdd / sid   # 1.759
	detectorColCount = 900      # Total columns
	detectorRowCount = 16       # Total rows

	# CatSim uses detector-face pitch (1.0mm); BasisSimulator uses isocenter pitch
	detectorColSize_face = 1.0  # Column pitch at detector face (mm) — for CatSim
	detectorRowSize_face = 1.0  # Row pitch at detector face (mm) — for CatSim
	detectorColSize = detectorColSize_face / magnification   # ≈ 0.569 mm at isocenter
	detectorRowSize = detectorRowSize_face / magnification   # ≈ 0.569 mm at isocenter

	# --- Clinical Reconstruction Parameters ---
	z_coverage_mm = detectorRowCount * detectorRowSize       # at isocenter (≈ 9.09 mm)
	sliceThickness = 1.0        # mm (clinical standard — 0.5, 0.625, 1.0, 1.25, 2.5, 5.0)
	sliceCount = floor(Int, z_coverage_mm / sliceThickness)  # = 9
end

# ╔═╡ ad03b067-dfd5-4a2d-8b91-a81b5d29c1bf
# Simulation configuration - Single Source of Truth
# RENAMED to match CatSim conventions exactly to prevent mismatch errors
SIM_CONFIG = (
	imageSize = 512,            # Reconstruction Matrix X/Y
	fov_mm = 350.0,             # Field of View (mm)
	
	sid = sid,
	sdd = sdd,
	detectorColCount = detectorColCount,
	detectorRowCount = detectorRowCount,
	detectorColSize = detectorColSize,
	detectorRowSize = detectorRowSize,
	detectorColSize_face = detectorColSize_face,
	detectorRowSize_face = detectorRowSize_face,
	
	sliceCount = sliceCount,
	sliceThickness = sliceThickness,
	z_coverage_mm = z_coverage_mm,
	
	kvp = 120,
	mA = 200,
	viewsPerRotation = 984,
	rotationTime = 1.0,         # seconds
	
	n_energy_bins = 15,
)

# ╔═╡ 888e1831-dc35-4f21-b32d-d20a7b046152
md"""
---
## 2. CatSim (XCIST) Integration

These wrapper functions provide a robust Julia interface to XCIST/CatSim via `PythonCall.jl`. They include critical fixes for detector geometry (preventing "braided" sinograms) and reconstruction units (ensuring correct HU output).
"""

# ╔═╡ 279f4105-4ba9-48f3-99da-17327ede3a6a
const _catsim = Ref{Py}()

# ╔═╡ 36f11d21-ac5f-46f2-af6c-5bc45f2bdaca
const _recon_mod = Ref{Py}()

# ╔═╡ 63cacc29-7298-49eb-a58d-ac6e8d728d49
const _np = Ref{Py}()

# ╔═╡ 75db6abe-45e7-4bc5-9fb4-91cf4702a753
const _catsim_initialized = Ref(false)

# ╔═╡ 6ee238df-747c-4896-84fe-9e1ce21da6d8
const _cfg_path = Ref("")

# ╔═╡ c37961d0-a102-491e-9056-6a9529574be0
function catsim_init()
	# Check if the reference is assigned. 
	# This is robust against notebook re-runs/state resets.
	if !isassigned(_catsim)
		_catsim[] = pyimport("gecatsim")
		_recon_mod[] = pyimport("gecatsim.reconstruction.pyfiles.recon")
		_np[] = pyimport("numpy")
		
		# Find config files path safely
		spec = pyimport("importlib.util")
		gecatsim_spec = spec.find_spec("gecatsim")
		gecatsim_path = pyconvert(String, gecatsim_spec.origin)
		base_path = dirname(dirname(gecatsim_path))
		_cfg_path[] = joinpath(base_path, "gecatsim", "examples", "cfg")
	end
	return _catsim[], _recon_mod[], _np[], _cfg_path[]
end

# ╔═╡ 10e8be47-0d1c-4f1f-a1bc-ad9cee0dfc51
function catsim_create_simulation(;
	phantom_cfg="Phantom_Sample.cfg",
	scanner_cfg="Scanner_Sample_generic.cfg",
	protocol_cfg="Protocol_Sample_axial.cfg"
)
	xc, _, _, cfg_path = catsim_init()
	ct = xc.CatSim(
		joinpath(cfg_path, phantom_cfg),
		joinpath(cfg_path, scanner_cfg),
		joinpath(cfg_path, protocol_cfg)
	)
	return ct
end

# ╔═╡ 39cd5260-00f2-4a99-b76c-491190be1060
"""
	catsim_configure_scanner!

Configures the scanner geometry. 
**CRITICAL FIX:** Sets `detectorColsPerMod = 1` to prevent CatSim from inserting inter-module gaps which cause "braided" sinograms.
"""
function catsim_configure_scanner!(ct; sid, sdd, cols, rows, pitch)
	ct.scanner.sid = sid
	ct.scanner.sdd = sdd
	ct.scanner.detectorColCount = cols
	ct.scanner.detectorRowCount = rows
	ct.scanner.detectorColSize = pitch
	ct.scanner.detectorRowSize = pitch
	
	# --- FIX 1: Prevent "Braided" Sinogram ---
	# We treat every pixel as its own module to stop CatSim from padding bytes
	ct.scanner.detectorColsPerMod = 1
	ct.scanner.detectorRowsPerMod = rows 

	# --- FIX 2: Prevent "Squashed" Sinogram ---
	# Since every pixel is a module, we must force the gap between modules to 0.
	# Otherwise, CatSim adds a physical gap after every pixel, expanding the arc.
	ct.scanner.detectorColSkip = 0.0
	ct.scanner.detectorRowSkip = 0.0
	
	return ct
end

# ╔═╡ 3b913784-0725-4f00-8382-d45a10225cc9
function catsim_configure_protocol!(ct; mA, views, rot_time, kvp)
	ct.protocol.mA = mA
	ct.protocol.viewsPerRotation = views
	ct.protocol.viewCount = views
	ct.protocol.stopViewId = views - 1
	ct.protocol.rotationTime = rot_time
	ct.protocol.spectrumFilename = "tungsten_tar7.0_$(kvp)_filt.dat"
	return ct
end

# ╔═╡ e7010f7f-70cc-4260-b1e1-2eec2bf4f7f5
"""
	catsim_configure_recon!

Configures FDK reconstruction.
**CRITICAL FIX:** Sets `unit = "HU"` and `mu = 0.02` (water) to ensure output is in Hounsfield Units, preventing "black" images.
"""
function catsim_configure_recon!(ct; fov, size, slices, thickness)
	xc, _, _, cfg_path = catsim_init()
	# Load the base C-style recon config
	xc.source_cfg(joinpath(cfg_path, "Recon_Sample_2d.cfg"), ct)
	
	ct.recon.fov = fov
	ct.recon.imageSize = size
	ct.recon.sliceCount = slices
	ct.recon.sliceThickness = thickness
	
	# --- THE FIX FOR THE "BLACK" IMAGE ---
	# Force output to Hounsfield Units so values are -1000 to +1000
	# instead of tiny raw attenuation coefficients (0.02).
	ct.recon.unit = "HU"
	ct.recon.mu = 0.02  # Water reference (mm^-1)
	ct.recon.huOffset = -1000
	return ct
end

# ╔═╡ 2b347002-10e0-4a2b-9748-7aa17f04471f
function catsim_configure_phantom!(ct, json_path; scale=1.0, offset=[0.0,0.0,0.0])
	ct.phantom.callback = "Phantom_Voxelized"
	ct.phantom.projectorCallback = "C_Projector_Voxelized"
	ct.phantom.filename = json_path
	ct.phantom.scale = scale
	ct.phantom.centerOffset = pylist(offset)
	return ct
end

# ╔═╡ 1309c03d-6766-4ac4-9d41-134f7a102d40
"""
    catsim_forward_project

Runs forward projection and reads the binary output directly using Julia I/O for speed and robustness.
"""
function catsim_forward_project(ct; results_name="catsim_out")
	ct.resultsName = results_name
	ct.run_all()

	# Read binary output (Native Julia)
	# CatSim C-Order on disk: [Views, Rows, Cols]
	# Julia reads flat.
	# Reshaping to (Cols, Rows, Views) in Julia correctly maps:
	#   Fastest dim -> Cols
	#   Medium dim  -> Rows
	#   Slowest dim -> Views
	
	rows = Int(pyconvert(Float64, ct.scanner.detectorRowCount))
	cols = Int(pyconvert(Float64, ct.scanner.detectorColCount))
	views = Int(pyconvert(Float64, ct.protocol.viewCount))
	
	scan_file = "$(results_name).prep"
	raw_bytes = read(scan_file)
	sino_flat = reinterpret(Float32, raw_bytes)
	
	return reshape(sino_flat, (cols, rows, views))
end

# ╔═╡ 259ed104-73f4-4383-96ae-a50465aa3a48
"""
    catsim_reconstruct_fdk

Runs FDK reconstruction and reads binary output directly.
"""
function catsim_reconstruct_fdk(ct; results_name="catsim_out")
	_, recon_mod, _, _ = catsim_init()
	ct.resultsName = results_name
	
	# Ensure correct recon filename matching
	ct.recon.filename = ct.resultsName
	ct.do_Recon = 1
	
	recon_mod.recon(ct)

	nx = Int(pyconvert(Float64, ct.recon.imageSize))
	ny = Int(pyconvert(Float64, ct.recon.imageSize))
	nz = Int(pyconvert(Float64, ct.recon.sliceCount))

	# CatSim appends size to filename
	recon_file = "$(results_name)_$(nx)x$(ny)x$(nz).raw"
	
	if !isfile(recon_file)
		error("Recon file not found: $recon_file")
	end

	raw_bytes = read(recon_file)
	vol_flat = reinterpret(Float32, raw_bytes)
	
	# Reshape (X, Y, Z)
	return reshape(vol_flat, (nx, ny, nz))
end

# ╔═╡ d8f1521e-8cb8-467b-be20-2020f87974b6
function catsim_cleanup(results_name)
	for ext in [".air", ".offset", ".scan", ".prep"]
		f = results_name * ext
		isfile(f) && rm(f)
	end
	for f in readdir(".")
		if startswith(f, results_name) && endswith(f, ".raw")
			rm(f)
		end
	end
end

# ╔═╡ e9bb86d2-5da3-4851-8114-d8a810d4d693
md"""
---
## 3. Phantom Creation & Export

We create the **Gammex 472** phantom using `BasisSimulator`. To ensure parity, we then export this exact phantom's mask to CatSim using a **Multi-Material Export** strategy. This ensures both simulators interact with the exact same voxel geometry and material definitions.
"""

# ╔═╡ 8c1f521d-5db0-4e42-ae7f-6042632cad7c
REGION_TO_CATSIM = Dict(
	2 => "water",
	10 => "Gammex472_Ca_50",
	11 => "Gammex472_Ca_100",
	12 => "Gammex472_Ca_200",
	13 => "Gammex472_Ca_300",
	14 => "Gammex472_Ca_400",
	15 => "Gammex472_Ca_500",
	16 => "Gammex472_Ca_600",
	
	20 => "Gammex472_I_2_0",
	21 => "Gammex472_I_2_5",
	22 => "Gammex472_I_5_0",
	23 => "Gammex472_I_7_5",
	24 => "Gammex472_I_10_0",
	25 => "Gammex472_I_15_0",
	26 => "Gammex472_I_20_0"
)

# ╔═╡ 0d039ffd-1b0f-432c-aac7-7c802562d2de
# --- 2. Phantom Geometry Generator ---
function generate_gammex_labels(; n_voxels=512, fov_cm=35.0, z_cm=2.0)
	n_slices = 16 
	dx = fov_cm / n_voxels
	dy = fov_cm / n_voxels
	dz = z_cm / n_slices # Slice thickness for the label map

	x = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)
	y = range(-fov_cm/2 + dy/2, fov_cm/2 - dy/2, length=n_voxels)
	
	labels = zeros(Int, n_voxels, n_voxels, n_slices)

	# Geometry Specs
	body_radius = 16.5
	rod_radius = 1.4
	inner_ring_R = 5.0
	outer_ring_R = 10.5

	# Insert Angles
	n_inserts = 7
	angles_ca = [2π * i / n_inserts for i in 0:(n_inserts-1)]
	angles_i = [2π * i / n_inserts + π/n_inserts for i in 0:(n_inserts-1)] 

	ca_ids = [10, 11, 12, 13, 14, 15, 16]
	i_ids = [20, 21, 22, 23, 24, 25, 26]

	# --- VOXEL LOOP ---
	for j in 1:n_voxels, i in 1:n_voxels
		xi = x[i]
		yj = y[j]
		r = sqrt(xi^2 + yj^2)
		lbl = 0 

		if r <= body_radius
			lbl = 2 # Water
			
			# Check Inner Ring
			for (idx, ang) in enumerate(angles_ca)
				cx, cy = inner_ring_R * cos(ang), inner_ring_R * sin(ang)
				if sqrt((xi-cx)^2 + (yj-cy)^2) <= rod_radius
					lbl = ca_ids[idx]; break
				end
			end
			
			# Check Outer Ring
			for (idx, ang) in enumerate(angles_i)
				cx, cy = outer_ring_R * cos(ang), outer_ring_R * sin(ang)
				if sqrt((xi-cx)^2 + (yj-cy)^2) <= rod_radius
					lbl = i_ids[idx]; break
				end
			end
		end
		labels[i, j, :] .= lbl
	end
	
	return (labels=labels, voxel_size=(dx, dy, dz), dims=(n_voxels, n_voxels, n_slices))
end

# ╔═╡ cdfdab68-0813-433e-9628-d3168cf48041
function create_and_export_phantom(output_dir, basename, config)
	# CALCULATE Z-EXTENT CORRECTLY
	# Fix: Use sliceCount * sliceThickness (Total Recon Volume)
	# instead of detectorRows (which is just one rotation height)
	total_z_cm = (config.sliceCount * config.sliceThickness) / 10.0
	
	phantom = generate_gammex_labels(
		n_voxels=config.imageSize, 
		fov_cm=config.fov_mm/10.0, 
		z_cm=total_z_cm 
	)
	
	mkpath(output_dir)
	unique_labels = unique(phantom.labels)
	filter!(l -> l != 0, unique_labels)
	
	json_materials, json_filenames, json_datatypes = String[], String[], String[]
	json_cols, json_rows, json_slices = Int[], Int[], Int[]
	json_xsize, json_ysize, json_zsize = Float64[], Float64[], Float64[]
	json_xoffset, json_yoffset, json_zoffset, json_densscale = Float64[], Float64[], Float64[], Float64[]

	nx, ny, nz = phantom.dims
	vx, vy, vz = phantom.voxel_size .* 10.0 # cm -> mm
	
	for lbl in unique_labels
		if !haskey(REGION_TO_CATSIM, lbl) continue end
		mat_name = REGION_TO_CATSIM[lbl]
		
		# Mask generation (Julian column-major is fine here, CatSim handles the JSON map)
		mask = Float32.(phantom.labels .== lbl)
		fname = "$(basename)_mat$(lbl).density"
		write(joinpath(output_dir, fname), mask) 
		
		push!(json_materials, mat_name)
		push!(json_filenames, fname)
		push!(json_datatypes, "float")
		push!(json_cols, nx); push!(json_rows, ny); push!(json_slices, nz)
		push!(json_xsize, vx); push!(json_ysize, vy); push!(json_zsize, vz)
		push!(json_xoffset, (nx+1)/2.0); push!(json_yoffset, (ny+1)/2.0); push!(json_zoffset, (nz+1)/2.0)
		push!(json_densscale, 1.0)
	end
	
	json_data = Dict(
		"n_materials" => length(json_materials),
		"mat_name" => json_materials,
		"volumefractionmap_filename" => json_filenames,
		"volumefractionmap_datatype" => json_datatypes,
		"cols" => json_cols, "rows" => json_rows, "slices" => json_slices,
		"x_size" => json_xsize, "y_size" => json_ysize, "z_size" => json_zsize,
		"x_offset" => json_xoffset, "y_offset" => json_yoffset, "z_offset" => json_zoffset,
		"density_scale" => json_densscale
	)
	
	json_path = joinpath(output_dir, "$(basename).json")
	open(json_path, "w") do f
		write(f, "{\n")
		for (i, (k, v)) in enumerate(json_data)
			val_str = (v isa Vector{String}) ? "[\""*join(v, "\", \"")*"\"]" : (v isa Vector ? "["*join(v, ", ")*"]" : string(v))
			write(f, "  \"$k\": $val_str" * (i < length(json_data) ? "," : "") * "\n")
		end
		write(f, "}\n")
	end
	return json_path
end

# ╔═╡ cf4eac74-b132-47df-88cb-4f6e355f0fde
md"""
---
## 4. BasisSimulator Execution

We generate the sinogram and reconstruction using the workspace-based `BasisSimulator` API:

```
ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom)
BS.simulate!(ws, phantom, scanner, protocol, sim_opts, recon_opts)
ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size)
vol = BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size)
hu = BS.to_hounsfield(Array(vol); μ_water=μ_water)
```

The pipeline:
1. `create_eict_workspace` pre-allocates all GPU buffers (sinograms, physics kernels, etc.)
2. `simulate!` runs forward projection with the selected physics fidelity (polychromatic, noise, etc.)
3. `create_fdk_recon_workspace` + `reconstruct!` runs FDK backprojection into pre-allocated volume
4. `to_hounsfield` converts from attenuation (cm⁻¹) to HU using empirical water calibration
5. GPU workspaces are scoped in `let` blocks so memory is freed after each step

We first simulate a uniform water phantom to obtain the empirical μ\_water calibration value, then simulate the full Gammex 472 phantom.
"""

# ╔═╡ eaa684ba-e382-4c0a-b0b8-86d23b32078d
# 2. Protocol (Technique)
protocol = BS.CTProtocol(
	mA = Float64(SIM_CONFIG.mA),
	kVp = SIM_CONFIG.kvp,
	views = SIM_CONFIG.viewsPerRotation,
	rotation_time = SIM_CONFIG.rotationTime,
);

# ╔═╡ de1ebb42-6491-4de9-b302-b39d7784e9f4
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
	# Without these, build_physics_config() falls back to factory defaults.
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

# ╔═╡ 7ee90622-cc6d-47af-bb05-f7af27c5a4c6
# SimOptions fidelity presets:
#   :ideal  → monochromatic, no noise, no physics effects
#   :low    → monochromatic + noise
#   :medium → noise + focal_spot + crosstalk + flat_filter + bhc (polychromatic)
#   :high   → all 14 effects ON except DAS (polychromatic, full physics)
#
# :high enables all physics for realistic comparison against CatSim:
#   fill_factor, flat_filter, bowtie_filter, detector_efficiency, scatter,
#   scatter_correction, crosstalk, optical_crosstalk, focal_spot, noise,
#   lag, heel_effect, bhc. Spectrum auto-loaded from protocol.kVp.
sim_opts = BS.SimOptions(
	fidelity = :high,
	# use_scatter_correction = true,
	# use_scatter_correction = false,
	seed = 1234,
);

# ╔═╡ 7ea4849b-fdb2-4fdd-93b2-cdda9c673fe9
sim_opts

# ╔═╡ d9a00e8c-4be6-41d8-b04f-de07cb01502f
recon_opts = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (SIM_CONFIG.imageSize, SIM_CONFIG.imageSize, SIM_CONFIG.sliceCount),
	fov_cm = SIM_CONFIG.fov_mm / 10.0,
	z_cm = SIM_CONFIG.sliceCount * SIM_CONFIG.sliceThickness / 10.0,
	filter = :standard,  # CatSim uses 'standard' kernel (apodized ramp); use :ram_lak for pure ramp
);

# ╔═╡ e9425189-6734-4b69-8251-4e093c09367c
# Note: simulate!() internally loads the spectrum from protocol.kVp when fidelity=:high.
# We load it here explicitly for reference and verification against CatSim's spectrum.
begin
	energies_raw, weights_raw = BS.load_spectrum(SIM_CONFIG.kvp)
	energies, weights = BS.downsample_spectrum(energies_raw, weights_raw, SIM_CONFIG.n_energy_bins)
	materials = BS.get_region_materials()
end;

# ╔═╡ 67cf1256-34a0-46e8-bb15-a7dc5e1f1ffb
phantom_water_gpu = let
	# 20cm diameter water cylinder (matching notebook 06 pattern)
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
			water_mask[i, j, k] = UInt8(1)  # Water = label 1
		end
	end

	# Air = label 0, Water = label 1
	air_material = XA.Material(
		"Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
		Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
	)
	water_materials = Dict(0 => air_material, 1 => XA.Materials.water)

	# GPU-backed Phantom
	BS.Phantom(Metal.MtlArray(water_mask), water_materials, (voxel_cm, voxel_cm, voxel_z_cm))
end;

# ╔═╡ 6c63ee86-ed7a-435d-b6d9-12adc99d7b3e
# # Water calibration — workspace scoped locally to free GPU memory
# μ_water_calibrated = let
# 	ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_water_gpu)
# 	BS.simulate!(ws, phantom_water_gpu, scanner, protocol, sim_opts, recon_opts)

# 	recon_size = recon_opts.matrix_size
# 	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
# 	vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

# 	cx, cy, cz = size(vol) .÷ 2
# 	result = mean(vol[cx-2:cx+2, cy-2:cy+2, cz-1:cz+1])

# 	ws = nothing; ws_fdk = nothing; vol = nothing
# 	GC.gc(true)

# 	result
# end

μ_water_calibrated = 0.23

# ╔═╡ 539cb27b-cae4-4461-ad5b-357a1840035b
phantom_basis = BS.create_gammex_472(
    n_voxels = SIM_CONFIG.imageSize,
    n_slices = SIM_CONFIG.sliceCount,
    fov_cm = SIM_CONFIG.fov_mm / 10.0,
    z_cm = (SIM_CONFIG.sliceCount * SIM_CONFIG.sliceThickness) / 10.0
);

# ╔═╡ a1ca5b27-83c2-4dc8-8822-69b723a3bf12
# GPU-backed Phantom (preserves materials, voxel_size, etc.)
phantom_gammex_gpu = BS.Phantom(
	Metal.MtlArray(phantom_basis.mask),
	phantom_basis.materials,
	phantom_basis.voxel_size,
	phantom_basis.origin,
	phantom_basis.fov
);

# ╔═╡ 743c4088-97e9-4584-b324-c54b765417c4
# Gammex simulation — workspace-based with GC after each step
(sino_basis, recon_basis_hu) = let
	ws = BS.create_eict_workspace(scanner, protocol, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, protocol, sim_opts, recon_opts)

	# Copy sinogram to CPU
	sino_cpu = Array(ws.sino_noisy_out)

	# FDK reconstruction → CPU HU
	recon_size = recon_opts.matrix_size
	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	fdk_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_water_calibrated
	)
	ws_fdk = nothing; GC.gc(true)

	# Free simulation workspace
	ws = nothing
	GC.gc(true)

	(sino_cpu, fdk_hu)
end;

# ╔═╡ 5ab81a4e-bc92-46f9-8716-31ddb77faf36
md"""
## 5. CatSim Execution

We run CatSim using the exact same configuration variables (`SIM_CONFIG`) and the phantom we just exported.
"""

# ╔═╡ 422f6213-a960-4637-b43e-a7a651f0711a
catsim_results = let
    # Clean up stale output from any previous failed runs
    catsim_cleanup("sim_final")

    work_dir = joinpath(@__DIR__, "catsim_work")
    phantom_json = create_and_export_phantom(work_dir, "phantom_120kvp", SIM_CONFIG)

    ct = catsim_create_simulation()
    catsim_configure_phantom!(ct, phantom_json)
    
    # Updated to use new config keys directly
    catsim_configure_scanner!(ct, 
        sid=SIM_CONFIG.sid,
        sdd=SIM_CONFIG.sdd, 
        cols=SIM_CONFIG.detectorColCount,
        rows=SIM_CONFIG.detectorRowCount,
        pitch=SIM_CONFIG.detectorColSize_face # CatSim expects detector-face pitch
    )
    
    catsim_configure_protocol!(ct, 
        mA=SIM_CONFIG.mA,
        views=SIM_CONFIG.viewsPerRotation, 
        rot_time=SIM_CONFIG.rotationTime,
        kvp=SIM_CONFIG.kvp
    )
    
    catsim_configure_recon!(ct, 
        fov=SIM_CONFIG.fov_mm,
        size=SIM_CONFIG.imageSize, 
        slices=SIM_CONFIG.sliceCount,     # Use sliceCount, NOT detectorRows
        thickness=SIM_CONFIG.sliceThickness
    )
    
    # 3. Run
    @info "Running Forward Projection..."
    sino = catsim_forward_project(ct, results_name="sim_final")
    
    @info "Running FDK Reconstruction..."
    recon = catsim_reconstruct_fdk(ct, results_name="sim_final")
    
    catsim_cleanup("sim_final")
    
    # 4. Visualization Orientation
    # Julia's `heatmap` expects (X, Y).
    # CatSim data read as (Cols, Rows, Views). 
    # For Sinogram: (Cols, Views) is standard. So we use sino[:, center_row, :].
    # For Recon: (X, Y, Z). This is already correct for heatmap!
    
    (sinogram=sino, reconstruction=recon)
end;

# ╔═╡ 0052bf89-0ca5-4a91-9602-b24b7b8af497
# Extract results for comparison
sino_catsim = catsim_results.sinogram;

# ╔═╡ 4fb5c9a9-4b90-43c2-804f-082c99ec7be8
recon_catsim = catsim_results.reconstruction; # Already in HU thanks to configuration fix

# ╔═╡ 05257e76-ccac-4d9f-aa38-1f42a67b8bed
md"""
---
## 6. Forward Projection Comparison

We compare the raw sinograms generated by both engines.
"""

# ╔═╡ 26ae3a31-0d58-4c14-8247-a6d546d0406b
sino_metrics = let
    # CatSim output (Cols, Rows, Views) matches BasisSimulator (Cols, Rows, Views)
    # thanks to our careful reshaping in the wrapper functions.
    
    sino_bs = sino_basis
    sino_cs = sino_catsim

    diff = sino_bs .- sino_cs
    rmse = sqrt(mean(diff.^2))
    max_diff = maximum(abs.(diff))
    correlation = cor(vec(sino_bs), vec(sino_cs))

    mean_val = mean([mean(sino_bs), mean(sino_cs)])
    nrmse_pct = 100.0 * rmse / mean_val

    (
        rmse = rmse,
        nrmse_pct = nrmse_pct,
        correlation = correlation,
        sino_bs = sino_bs,
        sino_cs = sino_cs,
        diff = diff
    )
end

# ╔═╡ 4bf718c1-5d64-43ba-846e-94c92603893a
let
	fig = CM.Figure(size=(1200, 400), fontsize=12)
	central_row = SIM_CONFIG.detectorRowCount ÷ 2

	ax1 = CM.Axis(fig[1,1], title="BasisSimulator Sinogram", xlabel="Col", ylabel="View")
	hm1 = CM.heatmap!(ax1, sino_metrics.sino_bs[:, central_row, :], colormap=:grays)

	ax2 = CM.Axis(fig[1,2], title="CatSim Sinogram", xlabel="Col", ylabel="View")
	hm2 = CM.heatmap!(ax2, sino_metrics.sino_cs[:, central_row, :], colormap=:grays)

	ax3 = CM.Axis(fig[1,3], title="Difference (Basis - CatSim)", xlabel="Col", ylabel="View")
	diff_slice = sino_metrics.diff[:, central_row, :]
	max_diff = maximum(abs.(diff_slice))
	if max_diff == 0
		max_diff = 1.0
	end
	hm3 = CM.heatmap!(ax3, diff_slice, colormap=:RdBu, colorrange=(-max_diff, max_diff))
	CM.Colorbar(fig[1,4], hm3)

	CM.save(joinpath(FIGURES_DIR, "nb01_forward_projection.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ 7ac94cf5-d669-4a70-b813-bd2657d700b6
md"""
---
## 7. Reconstruction Comparison (HU Accuracy)

We compare the FDK reconstructions. Note that `BasisSimulator` was converted to HU manually, while `CatSim` was configured to output HU directly.
"""

# ╔═╡ 74d9a742-6428-499b-8161-1bf3954c4da3
@bind recon_slice UI.Slider(1:size(recon_basis_hu, 3), default=size(recon_basis_hu, 3) ÷ 2, show_value=true)

# ╔═╡ abbda23d-c67e-4ce8-bc4e-fccbdb6bead7
let
	fig = CM.Figure(size=(1200, 400), fontsize=12)
	vmin, vmax = -200, 400

	ax1 = CM.Axis(fig[1,1], title="BasisSimulator (Slice $recon_slice)", aspect=CM.DataAspect())
	hm1 = CM.heatmap!(ax1, recon_basis_hu[:,:,recon_slice], colormap=:grays, colorrange=(vmin, vmax))

	ax2 = CM.Axis(fig[1,2], title="CatSim (Slice $recon_slice)", aspect=CM.DataAspect())
	hm2 = CM.heatmap!(ax2, recon_catsim[:,:,recon_slice], colormap=:grays, colorrange=(vmin, vmax))
	CM.Colorbar(fig[1,3], hm2, label="HU")

	ax3 = CM.Axis(fig[1,4], title="Difference", aspect=CM.DataAspect())
	diff = recon_basis_hu[:,:,recon_slice] .- recon_catsim[:,:,recon_slice]
	max_d = max(abs(minimum(diff)), abs(maximum(diff)), 10.0)
	hm3 = CM.heatmap!(ax3, diff, colormap=:RdBu, colorrange=(-max_d, max_d))
	CM.Colorbar(fig[1,5], hm3, label="ΔHU")

	CM.save(joinpath(FIGURES_DIR, "nb01_fdk_comparison.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ 5caa1e46-8e9e-4b5d-b66c-bc0318a4fae8
md"""
---
## 8. Quantitative Accuracy

We extract regions of interest (ROIs) for the Calcium and Iodine inserts to verify HU accuracy.
"""

# ╔═╡ 3651a048-d7e0-4f84-a448-b0d8da3e1e28
hu_accuracy_results = let
	central_slice = size(recon_basis_hu, 3) ÷ 2
	# Update to use imageSize
	pixel_size_mm = SIM_CONFIG.fov_mm / SIM_CONFIG.imageSize
	
	cx, cy = SIM_CONFIG.imageSize ÷ 2, SIM_CONFIG.imageSize ÷ 2
	
	# Geometry of Gammex 472
	inner_ring_r = 50.0 / pixel_size_mm
	outer_ring_r = 105.0 / pixel_size_mm
	n_inserts = 7
	roi_radius = 5 # pixels

	results = []

	# Helper
	function measure(name, expected, r, idx, offset_angle=0)
		angle = 2π * (idx-1) / n_inserts + offset_angle
		ix = cx + round(Int, r * cos(angle))
		iy = cy + round(Int, r * sin(angle))
		
		roi_bs = recon_basis_hu[ix-roi_radius:ix+roi_radius, iy-roi_radius:iy+roi_radius, central_slice]
		roi_cs = recon_catsim[ix-roi_radius:ix+roi_radius, iy-roi_radius:iy+roi_radius, central_slice]
		
		push!(results, (
			name=name, expected=expected, 
			basis_hu=mean(roi_bs), catsim_hu=mean(roi_cs)
		))
	end

	# Calcium (Inner Ring)
	measure("Ca 50", 75, inner_ring_r, 1)
	measure("Ca 100", 150, inner_ring_r, 2)
	measure("Ca 200", 300, inner_ring_r, 3)
	measure("Ca 300", 450, inner_ring_r, 4)
	measure("Ca 400", 600, inner_ring_r, 5)
	measure("Ca 500", 750, inner_ring_r, 6)
	measure("Ca 600", 900, inner_ring_r, 7)

	# Iodine (Outer Ring) - Offset by pi/7
	offset = π / n_inserts
	measure("I 2.0", 55, outer_ring_r, 1, offset)
	measure("I 2.5", 70, outer_ring_r, 2, offset)
	measure("I 5.0", 135, outer_ring_r, 3, offset)
	measure("I 7.5", 200, outer_ring_r, 4, offset)
	measure("I 10.0", 270, outer_ring_r, 5, offset)
	measure("I 15.0", 400, outer_ring_r, 6, offset)
	measure("I 20.0", 535, outer_ring_r, 7, offset)

	results
end

# ╔═╡ 751c7086-603b-480f-9be6-de51f6cc446d
let
	fig = CM.Figure(size=(900, 900), fontsize=16)

	ca_idx = [i for i in eachindex(hu_accuracy_results) if startswith(hu_accuracy_results[i].name, "Ca")]
	i_idx = [i for i in eachindex(hu_accuracy_results) if startswith(hu_accuracy_results[i].name, "I")]

	# --- Calcium ---
	ax1 = CM.Axis(fig[1, 1], xlabel="CatSim HU", ylabel="BasisSimulator HU", title="Calcium")

	ca_x = [hu_accuracy_results[i].catsim_hu for i in ca_idx]
	ca_y = [hu_accuracy_results[i].basis_hu for i in ca_idx]

	CM.scatter!(ax1, ca_x, ca_y, color=:blue, marker=:circle, markersize=12, label="Calcium")

	ca_lo, ca_hi = min(minimum(ca_x), minimum(ca_y)), max(maximum(ca_x), maximum(ca_y))
	ca_pad = 0.05 * (ca_hi - ca_lo)
	CM.lines!(ax1, [ca_lo - ca_pad, ca_hi + ca_pad], [ca_lo - ca_pad, ca_hi + ca_pad], color=:gray, linestyle=:dash, label="Identity")

	ca_n = length(ca_x)
	ca_slope = (ca_n * sum(ca_x .* ca_y) - sum(ca_x) * sum(ca_y)) / (ca_n * sum(ca_x .^ 2) - sum(ca_x)^2)
	ca_int = (sum(ca_y) - ca_slope * sum(ca_x)) / ca_n
	ca_xfit = collect(range(ca_lo - ca_pad, ca_hi + ca_pad, length=100))
	ca_sign = ca_int >= 0 ? "+" : "-"
	ca_eq = "y = $(round(ca_slope, digits=3))x $(ca_sign) $(round(abs(ca_int), digits=1))"
	CM.lines!(ax1, ca_xfit, ca_int .+ ca_slope .* ca_xfit, color=:blue, linewidth=2, label=ca_eq)

	CM.axislegend(ax1, position=:rb)

	# --- Iodine ---
	ax2 = CM.Axis(fig[2, 1], xlabel="CatSim HU", ylabel="BasisSimulator HU", title="Iodine")

	i_x = [hu_accuracy_results[i].catsim_hu for i in i_idx]
	i_y = [hu_accuracy_results[i].basis_hu for i in i_idx]

	CM.scatter!(ax2, i_x, i_y, color=:red, marker=:utriangle, markersize=12, label="Iodine")

	i_lo, i_hi = min(minimum(i_x), minimum(i_y)), max(maximum(i_x), maximum(i_y))
	i_pad = 0.05 * (i_hi - i_lo)
	CM.lines!(ax2, [i_lo - i_pad, i_hi + i_pad], [i_lo - i_pad, i_hi + i_pad], color=:gray, linestyle=:dash, label="Identity")

	i_n = length(i_x)
	i_slope = (i_n * sum(i_x .* i_y) - sum(i_x) * sum(i_y)) / (i_n * sum(i_x .^ 2) - sum(i_x)^2)
	i_int = (sum(i_y) - i_slope * sum(i_x)) / i_n
	i_xfit = collect(range(i_lo - i_pad, i_hi + i_pad, length=100))
	i_sign = i_int >= 0 ? "+" : "-"
	i_eq = "y = $(round(i_slope, digits=3))x $(i_sign) $(round(abs(i_int), digits=1))"
	CM.lines!(ax2, i_xfit, i_int .+ i_slope .* i_xfit, color=:red, linewidth=2, label=i_eq)

	CM.axislegend(ax2, position=:rb)

	CM.save(joinpath(FIGURES_DIR, "nb01_hu_accuracy.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ 0081a253-6a6a-4a1b-bad0-6b4ca76e1b29
md"""
## 9. Noise Power Spectrum (NPS) Analysis

We calculate the Noise Power Spectrum (NPS) to evaluate the noise texture characteristics of the existing reconstructions.
"""

# ╔═╡ 4a65cf5c-26c9-47b3-9f1b-779757148567
function compute_nps(image_vol; roi_size=64, pix_mm=SIM_CONFIG.fov_mm/SIM_CONFIG.imageSize)
	# Get dimensions
	nx, ny, nz = size(image_vol)
	cx, cy, mid_z = nx ÷ 2, ny ÷ 2, nz ÷ 2
	
	# Define offsets for 4 ROIs clustered near the center (in water)
	# We avoid the exact center pixel to avoid alignment artifacts
	half_r = roi_size ÷ 2
	offsets = [
		(-half_r-10, -half_r-10), (half_r+10, -half_r-10), 
		(-half_r-10, half_r+10), (half_r+10, half_r+10)
	]
	
	ps_accum = zeros(Float64, roi_size, roi_size)
	count = 0
	
	# Hanning Window to reduce spectral leakage
	win_1d = 0.5 .* (1.0 .- cos.(2π .* (0:roi_size-1) ./ (roi_size-1)))
	win_2d = win_1d * win_1d'
	
	for (dx, dy) in offsets
		# Aggregate over 3 central slices
		for z_off in -1:1
			x0 = cx + dx - half_r
			y0 = cy + dy - half_r
			
			# Extract ROI
			roi = Float64.(image_vol[x0:x0+roi_size-1, y0:y0+roi_size-1, mid_z+z_off])
			
			# Detrend (subtract mean)
			roi .-= mean(roi)
			
			# Apply Window and FFT
			roi_fft = fft(roi .* win_2d)
			ps_accum .+= abs2.(roi_fft)
			count += 1
		end
	end
	
	# Normalize (Units: HU² * mm²)
	# Factor includes pixel area and number of averages
	nps_2d = fftshift(ps_accum ./ count) .* (pix_mm^2 / roi_size^2)
	
	# Radial Average
	freqs = fftshift(fftfreq(roi_size, 1/pix_mm))
	n_bins = roi_size ÷ 2
	radial_prof = zeros(n_bins)
	radial_counts = zeros(Int, n_bins)
	
	max_f = maximum(freqs)
	center_idx = roi_size ÷ 2 + 1
	
	for i in 1:roi_size, j in 1:roi_size
		f_val = sqrt(freqs[i]^2 + freqs[j]^2)
		bin_idx = floor(Int, (f_val / max_f) * (n_bins-1)) + 1
		
		if bin_idx <= n_bins
			radial_prof[bin_idx] += nps_2d[i,j]
			radial_counts[bin_idx] += 1
		end
	end
	
	# Average bins
	mask = radial_counts .> 0
	radial_prof[mask] ./= radial_counts[mask]
	radial_freqs = range(0, stop=max_f, length=n_bins)
	
	return nps_2d, radial_prof, radial_freqs
end

# ╔═╡ 4eac87de-e0a2-4e9f-9fc4-045f75ba5cdd
begin
    # Compute NPS for both simulators
    nps_2d_basis, nps_rad_basis, f_basis = compute_nps(recon_basis_hu)
    nps_2d_catsim, nps_rad_catsim, f_catsim = compute_nps(recon_catsim)
end

# ╔═╡ 13b97d04-167b-48f6-b553-ffd1c524463c
let
	fig = CM.Figure(size=(1000, 800))

	ax1 = CM.Axis(fig[1,1], title="NPS 2D (BasisSimulator)", aspect=CM.DataAspect())
	hm1 = CM.heatmap!(ax1, log10.(max.(nps_2d_basis, 1e-3)), colormap=:viridis)

	ax2 = CM.Axis(fig[1,2], title="NPS 2D (CatSim)", aspect=CM.DataAspect())
	hm2 = CM.heatmap!(ax2, log10.(max.(nps_2d_catsim, 1e-3)), colormap=:viridis)
	CM.Colorbar(fig[1,3], hm2, label="log10(NPS)")

	ax3 = CM.Axis(fig[2, 1:2], title="Radial NPS Profile", xlabel="Frequency (mm⁻¹)", ylabel="Power (HU² mm²)")
	CM.lines!(ax3, f_basis, nps_rad_basis, label="BasisSimulator", color=:blue, linewidth=2)
	CM.lines!(ax3, f_catsim, nps_rad_catsim, label="CatSim", color=:red, linestyle=:dash, linewidth=2)
	CM.axislegend(ax3)

	CM.save(joinpath(FIGURES_DIR, "nb01_nps.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ d7de5375-e94f-4e95-a38f-8c3e14a95dac
md"""
## 10. Modulation Transfer Function (MTF)

We estimate the MTF using the Edge Spread Function (ESF) method at the high-contrast boundary of the phantom body.
"""

# ╔═╡ 20f09224-754f-4a6d-b531-3d4d52bbf26e
function compute_mtf(image_vol; pix_mm=SIM_CONFIG.fov_mm/SIM_CONFIG.imageSize)
	# The phantom body radius is 16.5 cm = 165 mm
	nx, ny, nz = size(image_vol)
	cx, cy, mid_z = nx ÷ 2, ny ÷ 2, nz ÷ 2

	# Find the edge on the X-axis (Right side)
	# Radius in pixels
	r_pix = 165.0 / pix_mm
	edge_est = cx + round(Int, r_pix)
	
	# Extract a profile across the edge (Water -> Air)
	# We reduce the window to +/- 12 pixels (~8mm) to fit within the FOV margin
	win = 12 
	start_x = edge_est - win
	end_x = edge_est + win
	
	# Boundary safety check
	if end_x > nx 
		end_x = nx 
	end
	if start_x < 1 
		start_x = 1 
	end
	
	# Average 5 central rows to reduce noise
	prof_accum = zeros(Float64, end_x - start_x + 1)
	rows = -2:2
	for dy in rows
		prof_accum .+= image_vol[start_x:end_x, cy+dy, mid_z]
	end
	esf = prof_accum ./ length(rows)
	
	# Calculate LSF (Derivative of ESF)
	lsf = diff(esf)
	
	# Invert if edge is falling (Water -> Air) so LSF is positive
	if mean(lsf) < 0
		lsf = -lsf
	end
	
	# Apply Hanning window to LSF to smooth tails
	# This assumes the edge is centered in the window
	win_func = 0.5 .* (1.0 .- cos.(2π .* (0:length(lsf)-1) ./ (length(lsf)-1)))
	lsf = lsf .* win_func
	
	# MTF is magnitude of FFT of LSF
	mtf_raw = abs.(fft(lsf))
	
	# Normalize DC component to 1.0
	# Guard against zero division if LSF is flat (empty image)
	if mtf_raw[1] > 0
		mtf = mtf_raw ./ mtf_raw[1]
	else
		mtf = zeros(length(mtf_raw))
	end
	
	# Frequency axis (Nyquist is 0.5 * sampling rate)
	freqs = fftfreq(length(lsf), 1/pix_mm)
	
	# Return only the positive half
	n_out = length(freqs) ÷ 2
	return freqs[1:n_out], mtf[1:n_out], esf
end

# ╔═╡ c2f56289-0789-4596-9cd8-c45fa43a1b85
begin
    f_mtf, mtf_basis, esf_basis = compute_mtf(recon_basis_hu)
    _, mtf_catsim, esf_catsim = compute_mtf(recon_catsim)
end

# ╔═╡ bfc1b2ef-04c6-4fcc-a77f-a3a050990831
let
	fig = CM.Figure(size=(900, 400))

	ax1 = CM.Axis(fig[1,1], title="Edge Spread Function (Water-Air)", xlabel="Pixel Index", ylabel="HU")
	CM.lines!(ax1, esf_basis, color=:blue, label="BasisSimulator")
	CM.lines!(ax1, esf_catsim, color=:red, linestyle=:dash, label="CatSim")
	CM.axislegend(ax1, position=:lb)

	ax2 = CM.Axis(fig[1,2], title="MTF Comparison", xlabel="Frequency (lp/mm)", ylabel="Modulation Factor")
	CM.lines!(ax2, f_mtf, mtf_basis, color=:blue, linewidth=2, label="BasisSimulator")
	CM.lines!(ax2, f_mtf, mtf_catsim, color=:red, linewidth=2, linestyle=:dash, label="CatSim")
	CM.hlines!(ax2, [0.5, 0.1], color=:gray, linestyle=:dot)
	CM.text!(ax2, 0.1, 0.52, text="50%")
	CM.text!(ax2, 0.1, 0.12, text="10%")
	CM.axislegend(ax2)

	CM.save(joinpath(FIGURES_DIR, "nb01_mtf.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ ccb9db0f-68c9-401f-879e-8552dd2dc59d
md"""
## 11. Contrast-to-Noise Ratio (CNR)

We calculate the CNR for all inserts using the formula: 
$$CNR = \frac{|HU_{insert} - HU_{background}|}{\sigma_{background}}$$
"""

# ╔═╡ b943dd03-e572-44e8-bc58-1109848cd0c7
cnr_data = let
	# 1. Measure Background Statistics (Center Water Region)
	nx, ny, nz = size(recon_basis_hu)
	cx, cy, mid_z = nx ÷ 2, ny ÷ 2, nz ÷ 2
	roi_sz = 30
	
	# Extract center ROI (avoiding inserts which are at radius > 50mm)
	bg_roi_basis = recon_basis_hu[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]
	bg_roi_catsim = recon_catsim[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]
	
	μ_bg_b = mean(bg_roi_basis)
	σ_bg_b = std(bg_roi_basis)
	
	μ_bg_c = mean(bg_roi_catsim)
	σ_bg_c = std(bg_roi_catsim)
	
	# 2. Calculate CNR using insert means from Section 8
	labels = [r.name for r in hu_accuracy_results]
	
	# Basis CNR
	cnr_b = [abs(r.basis_hu - μ_bg_b) / σ_bg_b for r in hu_accuracy_results]
	
	# CatSim CNR
	cnr_c = [abs(r.catsim_hu - μ_bg_c) / σ_bg_c for r in hu_accuracy_results]
	
	(labels=labels, cnr_b=cnr_b, cnr_c=cnr_c, sigma_b=σ_bg_b, sigma_c=σ_bg_c)
end

# ╔═╡ 665431fd-6298-441c-8788-20270692b266
let
	fig = CM.Figure(size=(1000, 500))

	ax = CM.Axis(fig[1,1], title="Contrast-to-Noise Ratio (CNR)", xticklabelrotation=pi/4, ylabel="CNR")

	x = 1:length(cnr_data.labels)
	CM.barplot!(ax, x .- 0.2, cnr_data.cnr_b, width=0.4, color=:blue, label="BasisSimulator")
	CM.barplot!(ax, x .+ 0.2, cnr_data.cnr_c, width=0.4, color=:red, label="CatSim")

	ax.xticks = (x, cnr_data.labels)
	CM.axislegend(ax, position=:lt)

	text_str = "Background Noise (σ):\nBasis: $(round(cnr_data.sigma_b, digits=2)) HU\nCatSim: $(round(cnr_data.sigma_c, digits=2)) HU"
	CM.text!(ax, length(x)-4, maximum(cnr_data.cnr_b)*0.8, text=text_str, fontsize=12)

	CM.save(joinpath(FIGURES_DIR, "nb01_cnr.png"), fig, px_per_unit=2)
	fig
end

# ╔═╡ Cell order:
# ╟─58bbe3ac-ef92-4a99-877e-716ee5f91750
# ╠═90ba5323-144a-4e09-8c8d-469115f01a95
# ╠═8a6908cd-434a-4f18-be3e-a038786f6ef7
# ╠═3cfa2553-ab67-48cb-a179-7ad583b7f176
# ╠═0f1119d9-c5b9-47c0-9b96-48bff346ecef
# ╠═53b5042f-b653-4665-beae-6392b45de84e
# ╠═365fac15-be61-4fd3-89c1-acd3a63b3127
# ╠═f0000002-0001-0001-0001-000000000001
# ╠═f0000003-0001-0001-0001-000000000001
# ╠═e1b82d96-8029-451a-bf97-5534c9569330
# ╠═ce79b686-5fde-47a1-820b-48779e79a384
# ╠═29122cf4-0dec-42f7-93a7-6b4b316093f2
# ╠═04ddc034-3cf7-4bda-9242-82bb6de4b598
# ╠═f0000001-0001-0001-0001-000000000001
# ╠═1e5e1a0b-ab52-4b44-b17e-bff4c8e9802b
# ╟─59dc49d1-875e-4288-a61b-cf33ada02fdd
# ╠═45f217d7-1c9c-4623-a8ff-fe22787831e8
# ╠═ad03b067-dfd5-4a2d-8b91-a81b5d29c1bf
# ╟─888e1831-dc35-4f21-b32d-d20a7b046152
# ╠═279f4105-4ba9-48f3-99da-17327ede3a6a
# ╠═36f11d21-ac5f-46f2-af6c-5bc45f2bdaca
# ╠═63cacc29-7298-49eb-a58d-ac6e8d728d49
# ╠═75db6abe-45e7-4bc5-9fb4-91cf4702a753
# ╠═6ee238df-747c-4896-84fe-9e1ce21da6d8
# ╠═c37961d0-a102-491e-9056-6a9529574be0
# ╠═10e8be47-0d1c-4f1f-a1bc-ad9cee0dfc51
# ╠═39cd5260-00f2-4a99-b76c-491190be1060
# ╠═3b913784-0725-4f00-8382-d45a10225cc9
# ╠═e7010f7f-70cc-4260-b1e1-2eec2bf4f7f5
# ╠═2b347002-10e0-4a2b-9748-7aa17f04471f
# ╠═1309c03d-6766-4ac4-9d41-134f7a102d40
# ╠═259ed104-73f4-4383-96ae-a50465aa3a48
# ╠═d8f1521e-8cb8-467b-be20-2020f87974b6
# ╟─e9bb86d2-5da3-4851-8114-d8a810d4d693
# ╠═8c1f521d-5db0-4e42-ae7f-6042632cad7c
# ╠═0d039ffd-1b0f-432c-aac7-7c802562d2de
# ╠═cdfdab68-0813-433e-9628-d3168cf48041
# ╟─cf4eac74-b132-47df-88cb-4f6e355f0fde
# ╠═eaa684ba-e382-4c0a-b0b8-86d23b32078d
# ╠═de1ebb42-6491-4de9-b302-b39d7784e9f4
# ╠═7ee90622-cc6d-47af-bb05-f7af27c5a4c6
# ╠═7ea4849b-fdb2-4fdd-93b2-cdda9c673fe9
# ╠═d9a00e8c-4be6-41d8-b04f-de07cb01502f
# ╠═e9425189-6734-4b69-8251-4e093c09367c
# ╠═67cf1256-34a0-46e8-bb15-a7dc5e1f1ffb
# ╠═6c63ee86-ed7a-435d-b6d9-12adc99d7b3e
# ╠═539cb27b-cae4-4461-ad5b-357a1840035b
# ╠═a1ca5b27-83c2-4dc8-8822-69b723a3bf12
# ╠═743c4088-97e9-4584-b324-c54b765417c4
# ╟─5ab81a4e-bc92-46f9-8716-31ddb77faf36
# ╠═422f6213-a960-4637-b43e-a7a651f0711a
# ╠═0052bf89-0ca5-4a91-9602-b24b7b8af497
# ╠═4fb5c9a9-4b90-43c2-804f-082c99ec7be8
# ╟─05257e76-ccac-4d9f-aa38-1f42a67b8bed
# ╠═26ae3a31-0d58-4c14-8247-a6d546d0406b
# ╟─4bf718c1-5d64-43ba-846e-94c92603893a
# ╟─7ac94cf5-d669-4a70-b813-bd2657d700b6
# ╟─74d9a742-6428-499b-8161-1bf3954c4da3
# ╟─abbda23d-c67e-4ce8-bc4e-fccbdb6bead7
# ╟─5caa1e46-8e9e-4b5d-b66c-bc0318a4fae8
# ╠═3651a048-d7e0-4f84-a448-b0d8da3e1e28
# ╟─751c7086-603b-480f-9be6-de51f6cc446d
# ╟─0081a253-6a6a-4a1b-bad0-6b4ca76e1b29
# ╠═4a65cf5c-26c9-47b3-9f1b-779757148567
# ╠═4eac87de-e0a2-4e9f-9fc4-045f75ba5cdd
# ╟─13b97d04-167b-48f6-b553-ffd1c524463c
# ╟─d7de5375-e94f-4e95-a38f-8c3e14a95dac
# ╠═20f09224-754f-4a6d-b531-3d4d52bbf26e
# ╠═c2f56289-0789-4596-9cd8-c45fa43a1b85
# ╟─bfc1b2ef-04c6-4fcc-a77f-a3a050990831
# ╟─ccb9db0f-68c9-401f-879e-8552dd2dc59d
# ╠═b943dd03-e572-44e8-bc58-1109848cd0c7
# ╟─665431fd-6298-441c-8788-20270692b266
