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

# ╔═╡ d6d62fae-012d-11f1-1efc-67e7f251ff8c
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()

	using Revise
end

# ╔═╡ 2ab74942-1680-47dd-a8dc-f5242387253e
# ╠═╡ show_logs = false
using Unitful: @u_str, ustrip

# ╔═╡ 00000001-0000-0000-0000-000000000001
md"""
# Notebook 05: Full XCAT Simulation — Three Scanner Comparison

This notebook demonstrates a complete clinical CT simulation workflow using the XCAT phantom:

1. **Load XCAT phantom** (1600×1400×500 voxels) with custom materials
2. **Three scanner configurations**:
   - EICT Single-kVp (GE Revolution Apex, 120 kVp)
   - EICT Dual-kVp (GE Revolution Apex GSI, 80/140 kVp)
   - PCCT Standard (Siemens NAEOTOM Alpha, 140 kVp, 4 energy bins, v24.0 physics)
3. **Water phantom calibration** for accurate HU conversion per scanner
4. **Clinical reconstruction**: 512×512 output with FDK and Hybrid IR (strength 3) for all scanners
5. **VMI generation** at 40, 70, 100, 140 keV
6. **Visualization**: FDK vs Hybrid IR comparison, noise reduction analysis, ROI statistics in HU

All simulations run on **Apple Metal GPU** for maximum performance.
"""

# ╔═╡ cca08041-b05c-4045-a462-18b30fd0559f
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ 3ad61ec7-ba66-449d-8fd1-e79a2345a9d7
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ 3cbd1220-7118-42c7-9562-27c8c2e1b608
import CairoMakie as CM

# ╔═╡ f36f0c25-3c0c-4202-b890-917b031fa9e8
import Statistics: mean, std

# ╔═╡ 2a00221a-e861-4d53-bba6-bde7b1bc909f
# ╠═╡ show_logs = false
import Metal

# ╔═╡ c071c54d-0950-4260-83b7-7dc79300609e
import XLSX

# ╔═╡ dc9e51f9-2531-4483-b80a-622f5ecf4d0c
import XrayAttenuation as XA

# ╔═╡ 4ca1063f-1cc1-4253-a411-4817f1e584a2
UI.TableOfContents()

# ╔═╡ 00000002-0000-0000-0000-000000000001
begin
	FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")
	mkpath(FIGURES_DIR)
end

# ╔═╡ 20920020-fd6b-4a64-9e19-e00bfd616ee6
md"""
## 1. Load XCAT Phantom
"""

# ╔═╡ c744a9d3-5810-4465-82ee-2b8d9b5f68b1
ROOT_DIR = dirname(@__DIR__)

# ╔═╡ 7812a13f-0120-47c0-9bbf-e03918a8113a
PHANTOM_PATH = joinpath(
	ROOT_DIR, "data/xcat/vmale_50_1600x1400x500_8bit_little_endian_act_1.bin"
)

# ╔═╡ b1c2c0ee-e2ca-4ac3-a643-376aeafb0c6c
MATERIAL_XLSX_PATH = joinpath(
	ROOT_DIR,
	"data/xcat/Material_Spreadsheets/vmale_50_materials_heart_high_contrast.xlsx"
)

# ╔═╡ b38ffd25-d549-475d-b92a-5d08ebd6cc53
function load_phantom_bin(
		filepath::String;
		cols=1600,
		rows=1400,
		slices=500,
		dtype::Type=UInt8,
		order::Symbol=:F,
		perm::Union{Nothing,NTuple{3,Int}}=nothing,
		reverse_dims::Union{Nothing,Tuple{Vararg{Int}}}=(2,3)
	)

	expected_size = cols * rows * slices * sizeof(dtype)
	actual_size = filesize(filepath)

	@assert actual_size == expected_size "File size mismatch: expected $expected_size bytes, got $actual_size"

	data = Vector{dtype}(undef, cols * rows * slices)
	open(filepath, "r") do io
		read!(io, data)
	end

	if order == :F
		phantom = reshape(data, (cols, rows, slices))
	else
		phantom = permutedims(reshape(data, (slices, rows, cols)), (3, 2, 1))
	end

	if reverse_dims !== nothing && !isempty(reverse_dims)
		phantom = reverse(phantom, dims=reverse_dims)
	end

	return phantom
end

# ╔═╡ 72791f6b-0178-44dd-8f36-8071aa451b7c
phantom_labeled_raw = load_phantom_bin(PHANTOM_PATH);

# ╔═╡ 72791f6b-0178-44dd-8f36-8071aa451b8c
"""
	downsample_phantom(phantom, factor)

Downsample a labeled phantom by integer factor using nearest-neighbor (mode) sampling.
Preserves label integrity (no interpolation artifacts).

Examples:
- `factor=1` → 1600×1400×500 (UHR, original)
- `factor=2` → 800×700×250 (HR)
- `factor=4` → 400×350×125 (Standard)
- `factor=5` → 320×280×100 (Fast, ~512×512 equivalent)
"""
function downsample_phantom(phantom::AbstractArray{T, 3}, factor::Int) where T
	factor == 1 && return phantom

	old_size = size(phantom)
	new_size = old_size .÷ factor

	result = similar(phantom, new_size)
	for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
		# Use center of each block for nearest-neighbor
		oi = (i - 1) * factor + factor ÷ 2 + 1
		oj = (j - 1) * factor + factor ÷ 2 + 1
		ok = (k - 1) * factor + factor ÷ 2 + 1
		result[i, j, k] = phantom[oi, oj, ok]
	end
	return result
end

# ╔═╡ a1858da5-8231-460e-a471-35115d2b1476
md"""
### Resulting Behavior by Downsample Factor

| Factor | Phantom | recon_xy | Behavior |
| :--- | :--- | :--- | :--- |
| **1 (UHR)** | 1600×1400 | 512 | Clinical standard |
| **2 (HR)** | 800×700 | 512 | Still clinical standard (phantom ≥ 512) |
| **4 (Std)** | 400×350 | 350 | Matched to phantom |
| **5 (Fast)** | 320×280 | 280 | Matched to phantom |
"""

# ╔═╡ 72791f6b-0178-44dd-8f36-8071aa451b9c
begin
	# ═══════════════════════════════════════════════════════════════════════════
	# RESOLUTION CONTROL — Uncomment ONE line below:
	# ═══════════════════════════════════════════════════════════════════════════

	# DOWNSAMPLE_FACTOR = 1   # UHR: 1600×1400×500 (original, slowest)
	# DOWNSAMPLE_FACTOR = 2   # HR:  800×700×250  (4× faster)
	DOWNSAMPLE_FACTOR = 4   # Std: 400×350×125  (16× faster)
	# DOWNSAMPLE_FACTOR = 5   # Fast: 320×280×100 (~25× faster, good for testing)

	phantom_labeled = downsample_phantom(phantom_labeled_raw, DOWNSAMPLE_FACTOR)
end

# ╔═╡ 702aca88-e592-4718-9f7a-1d3aa7c950ce
unique_phantom_regions = sort!(Int.(unique(phantom_labeled)))

# ╔═╡ 00000003-0000-0000-0000-000000000001
md"""
**Phantom dimensions:** $(size(phantom_labeled)) voxels

**Unique regions:** $(length(unique_phantom_regions)) labels
"""

# ╔═╡ 237bd8ca-6ea5-4aff-81f4-8a77bb45f4fd
@bind z_preview UI.Slider(axes(phantom_labeled, 3); show_value = true, default=size(phantom_labeled, 3) ÷ 2)

# ╔═╡ 5dee3049-e56a-41dd-8148-fd9f5de3411b
let
	f = CM.Figure()
	ax = CM.Axis(f[1, 1], title="XCAT Phantom (slice $z_preview)", aspect = CM.DataAspect())
	CM.heatmap!(ax, phantom_labeled[:, :, z_preview], colormap=:glasbey_hv_n256)
	f
end

# ╔═╡ 00000004-0000-0000-0000-000000000001
md"""
### Load Materials from Excel
"""

# ╔═╡ 31006d4c-ad69-43db-a833-71182e2cf069
function compute_ZA_ratio(composition::Dict{Int, Float64})
	atomic_masses = Dict(
		1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
		15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904
	)

	Z_sum = 0.0
	A_sum = 0.0

	for (Z, mass_frac) in composition
		A = get(atomic_masses, Z, Float64(Z)*2)
		Z_sum += mass_frac * Z / A
		A_sum += mass_frac
	end

	return Z_sum / A_sum
end

# ╔═╡ cbd9b4b0-828c-4105-9583-b1b4f2ec74b6
function compute_mean_excitation_energy(composition::Dict{Int, Float64})
	I_values = Dict(
		1=>19.2, 6=>81.0, 7=>82.0, 8=>95.0, 11=>149.0, 12=>156.0,
		15=>173.0, 16=>180.0, 17=>174.0, 19=>190.0, 20=>191.0, 26=>286.0, 53=>491.0
	)

	atomic_masses = Dict(
		1=>1.008, 6=>12.011, 7=>14.007, 8=>15.999, 11=>22.990, 12=>24.305,
		15=>30.974, 16=>32.06, 17=>35.45, 19=>39.098, 20=>40.078, 26=>55.845, 53=>126.904
	)

	log_I_sum = 0.0
	Z_A_sum = 0.0

	for (Z, mass_frac) in composition
		A = get(atomic_masses, Z, Float64(Z)*2)
		I = get(I_values, Z, 10.0 * Z)

		Z_A = mass_frac * Z / A
		log_I_sum += Z_A * log(I)
		Z_A_sum += Z_A
	end

	I_mean = exp(log_I_sum / Z_A_sum)
	return I_mean * u"eV"
end

# ╔═╡ d70bc21c-0b77-4504-af16-7154980a32db
"""
Load materials from Excel spreadsheet with elemental composition.
Returns: Dict{Int, Material} mapping organ_id to Material
"""
function load_materials_from_xlsx(xlsx_path::String)
	xf = XLSX.readxlsx(xlsx_path)
	sheet = xf["Sheet1"]
	data = sheet["A2:P34"]

	materials = Dict{Int, XA.Material}()

	for i in 1:size(data, 1)
		try
			name = String(data[i, 1])
			organ_id = Int(data[i, 16])
			density = Float64(data[i, 15]) * u"g/cm^3"

			comp = Dict{Int, Float64}()
			comp[1] = Float64(data[i, 2])   # H
			comp[6] = Float64(data[i, 3])   # C
			comp[7] = Float64(data[i, 4])   # N
			comp[8] = Float64(data[i, 5])   # O
			comp[11] = Float64(data[i, 6])  # Na
			comp[12] = Float64(data[i, 7])  # Mg
			comp[15] = Float64(data[i, 8])  # P
			comp[16] = Float64(data[i, 9])  # S
			comp[17] = Float64(data[i, 10]) # Cl
			comp[19] = Float64(data[i, 11]) # K
			comp[20] = Float64(data[i, 12]) # Ca
			comp[26] = Float64(data[i, 13]) # Fe
			comp[53] = Float64(data[i, 14]) # I

			filter!(p -> p.second > 0, comp)

			ZA = compute_ZA_ratio(comp)
			I = compute_mean_excitation_energy(comp)

			mat = XA.Material(name, ZA, I, density, comp)
			materials[organ_id] = mat

		catch e
			@warn "Failed to parse row $i" exception=(e, catch_backtrace())
		end
	end

	return materials
end

# ╔═╡ 3deeccc4-d188-495d-82a2-868a947840d5
materials_dict = load_materials_from_xlsx(MATERIAL_XLSX_PATH)

# ╔═╡ 00000005-0000-0000-0000-000000000001
md"""
**Materials loaded:** $(length(materials_dict))

**Check coverage:** Missing labels = $(setdiff(unique_phantom_regions, keys(materials_dict)))
"""

# ╔═╡ 90070343-33a7-4873-ba42-32d1e80bde31
md"""
## 2. Create `Phantom()`

### Voxel Size Calculation

XCAT phantom dimensions: **1600 × 1400 × 500** voxels

Assuming clinical torso FOV:
- X (lateral): ~48 cm → voxel = 0.03 cm = 0.3 mm
- Y (AP): ~42 cm → voxel = 0.03 cm = 0.3 mm
- Z (SI): ~50 cm → voxel = 0.1 cm = 1.0 mm

This gives **high input resolution** (0.3mm in-plane).
"""

# ╔═╡ f3861cff-abd7-4cb9-9eb6-2a4f63677d29
begin
	# Base XCAT voxel dimensions (at original 1600×1400×500 resolution)
	base_voxel_cm = (0.03, 0.03, 0.1)  # 0.3mm × 0.3mm × 1.0mm

	# Scale voxel size by downsample factor to maintain correct FOV
	voxel_size_cm = base_voxel_cm .* DOWNSAMPLE_FACTOR

	# Computed FOV (should be ~48cm × 42cm × 50cm regardless of downsample)
	fov_x_cm = size(phantom_labeled, 1) * voxel_size_cm[1]
	fov_y_cm = size(phantom_labeled, 2) * voxel_size_cm[2]
	fov_z_cm = size(phantom_labeled, 3) * voxel_size_cm[3]
end

# ╔═╡ aebd1afc-64d9-4be3-a7d9-10e2c7a3d0ad
md"""
**Computed FOV:** $(round(fov_x_cm, digits=1)) × $(round(fov_y_cm, digits=1)) × $(round(fov_z_cm, digits=1)) cm
"""

# ╔═╡ bb8ec964-3fe8-4b67-8e32-356e8d10b942
begin
	# Create GPU phantom — MtlArray mask → GPU simulation
	phantom_mask_gpu = Metal.MtlArray(phantom_labeled)
	phantom_gpu = BS.Phantom(phantom_mask_gpu, materials_dict, voxel_size_cm)
end

# ╔═╡ 6562ecaa-0df4-460b-b2aa-7c1f53505070
md"""
**GPU Phantom created:**
- Mask size: $(size(phantom_gpu.mask))
- Mask type: $(typeof(phantom_gpu.mask))
- Materials: $(length(phantom_gpu.materials))
- FOV: $(phantom_gpu.fov) cm
- Downsample: $(DOWNSAMPLE_FACTOR)× (voxel = $(round.(voxel_size_cm .* 10, digits=2)) mm)
"""

# ╔═╡ 349fcb3d-b8fa-4423-8cdd-05a1a842c345
md"""
## 3. Create `Scanner()`

Three scanner configurations matching clinical systems:
"""

# ╔═╡ b1a2c3d4-e5f6-7890-abcd-111111111111
md"""
#### Auto-Sized Detector Arrays

Detector columns and rows are derived from the phantom FOV and scanner magnification.
This ensures full lateral coverage at each scanner's native pixel pitch while keeping
the z-dimension reasonable for GPU memory.

- **`detector_cols`** = `ceil(phantom_extent_mm × magnification / pixel_pitch)`
- **`detector_rows`** = `min(clinical_max, n_recon_slices)` — caps z-coverage
- **`pcct_n_views`** = 600 (standard Siemens clinical)

All PCCT physics (charge cloud, K-fluorescence, CCE, pileup, DRM) are per-pixel —
reducing array size does not change the physics model.
"""

# ╔═╡ 619f09c1-315e-42b6-9e44-2f36a98f8fee
# Phantom lateral extent (cm → mm)
phantom_extent_mm = max(
	size(phantom_labeled, 1) * voxel_size_cm[1],
	size(phantom_labeled, 2) * voxel_size_cm[2]
) * 10.0

# ╔═╡ ea2db1d2-d7ef-4c5c-9d74-e273ffda11b1
md"""
### 3.1 EICT Single-kVp — GE Revolution Apex (Quantix 160, 120 kVp)

**Scanner:** GE Revolution Apex with Quantix 160 — most advanced GE EICT available (2026)

**VERIFIED specs:**
- SID = 625.6mm — from Fluka MC validation ([PMC6448170](https://pmc.ncbi.nlm.nih.gov/articles/PMC6448170/))
- Detector rows = 256 × 0.625mm = 160mm z-coverage ([FDA K213715](https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213715.pdf))
- Gantry bore = 80cm, Gemstone Clarity detector (GOS)

**ESTIMATED specs:**
- SDD ≈ 1100mm (magnification ratio ~1.76)
- Detector cols ≈ 880 (50cm FOV calculation)
- In-plane pitch ≈ 1.0mm

**⚠️ WARNING:** XCIST defaults (SID=540, SDD=950) are for LightSpeed VCT, NOT Revolution!
"""

# ╔═╡ c9b47012-24b1-42da-bd74-d77fb8f55cb4
# ═══════════════════════════════════════════════════════════════════════════════
# GE REVOLUTION APEX (Quantix 160) — Wide-coverage EICT scanner
# Most advanced GE EICT currently available (2026)
# ═══════════════════════════════════════════════════════════════════════════════
#
# SOURCES:
#   - FDA K213715: Revolution Apex 510(k) clearance (detector rows, z-coverage)
#   - PMC6448170: Fluka MC validation of GE Revolution CT (SID = 625.6mm)
#   - GE product docs: Gemstone Clarity detector, 80cm bore, 160mm coverage
#
# NOTE: XCIST defaults (SID=540, SDD=950) are for LightSpeed VCT, NOT Revolution!
# ═══════════════════════════════════════════════════════════════════════════════

# ╔═╡ b733752a-4e10-4909-82c0-35e333fb2758
md"""
### 3.2 EICT Dual-kVp — GE Revolution Apex GSI (80/140 kVp)

Same scanner geometry, but protocol enables rapid kVp switching.
"""

# ╔═╡ e52eb0ca-a322-4cbc-9c92-9d5240c55664
md"""
### 3.3 PCCT Standard — Siemens NAEOTOM Alpha (0.4mm dexels, 2×2 binned)

Specs from FDA 510(k) K201501 and Siemens documentation:
- Source-to-isocenter: 595 mm
- Source-to-detector: 1085.5 mm
- **Scan FOV: 50 cm** at isocenter
- CdTe detector, 1.6 mm thick
- 4 energy bins: 20, 35, 55, 70 keV thresholds
- Standard mode: **144 × 0.4 mm** collimation (2×2 binned from 0.2mm native)

**v24.0 PCCT Physics** (activated via `fidelity = :pcct`):
- Physics-based charge cloud transport (Koch-Mehrin ODE, σ ≈ 13 μm)
- Full K-fluorescence (5 K-lines/element, Te→Cd cascade)
- Hecht CCE with small-pixel weighting potential (w/L ≈ 0.17)
- Yang 2025 seminonparalyzable pileup (VMR sub-Poisson at high flux)
- Unified DRM (FWHM 3.55 keV)
"""

# ╔═╡ f319b38b-7378-43fc-91a7-2db0554ab934
# ═══════════════════════════════════════════════════════════════════════════════
# SIEMENS NAEOTOM ALPHA — Photon-counting CT scanner
# Specs from FDA 510(k) K201501, Konrad 2025 (PMB), manufacturer docs
# ═══════════════════════════════════════════════════════════════════════════════

# ╔═╡ a70bc722-e769-4132-b082-b0e89a68228d
md"""
## 4. Create `Protocol()`

Three protocol configurations for the different scanner modes:
"""

# ╔═╡ a70bc722-e769-4132-b082-b0e89a68228f
md"""
### 4.1 EICT Single-kVp Protocol
"""

# ╔═╡ a70bc722-e769-4132-b082-b0e89a68228e
protocol_eict_single = BS.CTProtocol(
	kVp = 120.0,
	mA = 300.0,
	views = 984,
	# views = 1600,
	rotation_time = 0.5
)

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822a0
md"""
### 4.2 EICT Dual-kVp Protocol (GSI)
"""

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822a1
protocol_eict_dual = BS.CTProtocol(
	dual_energy = true,
	kVp = 140.0,
	mA = 200.0,
	kVp_low = 80.0,
	mA_low = 350.0,
	views = 984,
	# views = 1600,
	rotation_time = 0.5
)

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822a2
md"""
### 4.3 PCCT Standard Protocol
"""

# ╔═╡ 4ae79dad-dd72-4db6-98f5-0f2d28cb9f5e
# pcct_n_views = 984 # standard clinical
pcct_n_views = 1600 # standard clinical

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822a3
protocol_pcct_standard = BS.CTProtocol(
	kVp = 140.0,
	mA = 300.0,
	views = pcct_n_views,
	rotation_time = 0.25  # NAEOTOM Alpha fast gantry
)

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822b0
md"""
## 5. Create `SimOptions()`

Clinical-quality simulation with full physics:
"""

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822b1
sim_opts_eict = BS.SimOptions(
	fidelity = :high,
	# n_energy_bins = 100,
	# n_energy_bins = 1,
	
	# use_fill_factor = true,
	# use_flat_filter = true,
	# use_bowtie_filter = true,
	# use_detector_efficiency = true,
	# use_scatter = true,
	# use_scatter_correction = true,
	# use_crosstalk = true,
	# use_optical_crosstalk = true,
	# use_focal_spot = true,
	# use_noise = true,
	# use_lag = true,
	# use_heel_effect = true,
	# use_das = false,
	# use_bhc = true,
	# use_pcct_corrections = false,
	# pcct_noise_reduction = 0.0,
	seed = 42
)

# ╔═╡ a70bc722-e769-4132-b082-b0e89a6822b2
sim_opts_pcct = BS.SimOptions(
	fidelity = :high,
	pcct_noise_reduction = 0.60,
	# n_energy_bins = 100,
	# n_energy_bins = 2,
	seed = 42
)

# ╔═╡ 937ad7ea-f5c3-42c1-9f5d-6c6600fffaba
md"""
## 6. Create `ReconOptions()`

**Output resolution:** auto-matched to phantom — 512×512 at full resolution, scales down with downsample factor.

For z-coverage, we reconstruct a subset of slices for demonstration:
"""

# ╔═╡ 00000009-0000-0000-0000-000000000001
begin
	# Clinical cardiac reconstruction FOV (25cm for cardiac)
	recon_fov_cm = 25.0

	# Output matrix: match recon to phantom resolution
	# At full resolution (factor=1,2): 512×512 clinical standard
	# At downsampled (factor≥4): cap to phantom in-plane size so we don't
	# reconstruct finer than the input data supports
	recon_xy = min(512, min(size(phantom_labeled, 1), size(phantom_labeled, 2)))

	# Z slices: 128 for ~8cm coverage (practical for GPU)
	n_recon_slices = 128

	# VMI energies for spectral imaging
	vmi_energies = [40.0, 70.0, 100.0, 140.0]
end

# ╔═╡ f8fd4ab6-7c18-4d89-9045-41cf60666d63
begin
	# ─── EICT (GE Revolution Apex) ───
	eict_col_size_iso = 0.6  # mm at isocenter (estimated, not published)
	eict_det_cols = ceil(Int, phantom_extent_mm / eict_col_size_iso) # isocenter convention
	eict_det_rows = min(256, n_recon_slices) # cap at clinical max
end

# ╔═╡ 00000006-0000-0000-0000-000000000001
scanner_eict = BS.Scanner(
	# GEOMETRY — from MC validation (PMC6448170)
	source_to_isocenter = 625.6,  # mm — VERIFIED (PMC6448170: Fluka MC validation)
	source_to_detector = 1100.0,  # mm — ESTIMATED (magnification ~1.76, same as XCIST ratio)

	# DETECTOR ARRAY — auto-sized from phantom FOV (see cell above)
	detector_rows = eict_det_rows, # auto: min(256, n_recon_slices)
	detector_cols = eict_det_cols, # auto: phantom_extent × mag / 1.0mm pitch
	detector_row_size = 0.625, # mm at isocenter — VERIFIED (FDA K213715, GE docs)
	detector_col_size = eict_col_size_iso, # mm at isocenter — ESTIMATED (in-plane pitch not published)
	detector_shape = BS.CURVED_DETECTOR,

	# X-RAY SOURCE
	focal_spot_width = 1.0, # mm — ESTIMATED ⚠️ USED if sim_opts.use_focal_spot=true
	focal_spot_length = 1.0, # mm — ESTIMATED ⚠️ USED if sim_opts.use_focal_spot=true
	target_angle = 7.0, # degrees — typical GE value ⚠️ USED if sim_opts.use_heel_effect=true

	# FILTRATION
	flat_filter_material = :aluminum, # ⚠️ USED if sim_opts.use_flat_filter=true
	flat_filter_thickness = 2.5, # mm — ESTIMATED ⚠️ USED if sim_opts.use_flat_filter=true

	# DETECTOR PHYSICS — GE Gemstone Clarity (GOS scintillator)
	detector_material = :gos, # Gemstone Clarity = GOS — VERIFIED (GE docs)
	detector_depth = 3.0, # mm — ESTIMATED ⚠️ USED if sim_opts.use_detector_efficiency=true
	fill_factor_row = 0.9, # ESTIMATED ⚠️ USED if sim_opts.use_fill_factor=true
	fill_factor_col = 0.9, # ESTIMATED ⚠️ USED if sim_opts.use_fill_factor=true
	detection_gain = 1.0, # ⚠️ NOT USED — display only, not propagated to simulation
	# electronic_noise = 5000.0 # ⚠️ NOT USED — display only; noise comes from sim_opts
)

# ╔═╡ b1a2c3d4-e5f6-7890-abcd-222222222222
begin
	# ─── PCCT (Siemens NAEOTOM Alpha) ───
	pcct_det_cols = ceil(Int, phantom_extent_mm / 0.4) # 0.4mm at isocenter
	pcct_det_rows = min(144, n_recon_slices) # cap at clinical max
end

# ╔═╡ a21a509a-3cb2-452d-ae46-53f1339e0f37
@info "Auto-sized detectors from $(round(phantom_extent_mm/10, digits=1)) cm phantom" eict_det_cols eict_det_rows pcct_det_cols pcct_det_rows pcct_n_views

# ╔═╡ 00000007-0000-0000-0000-000000000001
scanner_pcct_standard = BS.Scanner(
	# GEOMETRY — VERIFIED (Konrad 2025, FDA K201501)
	source_to_isocenter = 595.0, # mm — VERIFIED (Konrad 2025)
	source_to_detector = 1085.5, # mm — VERIFIED (Konrad 2025)

	# DETECTOR ARRAY — auto-sized from phantom FOV (see cell above)
	detector_rows = pcct_det_rows, # auto: min(144, n_recon_slices)
	detector_cols = pcct_det_cols, # auto: phantom_extent × mag / 0.4mm pitch
	detector_row_size = 0.4, # mm — VERIFIED (2×2 binned from 0.2mm native dexels)
	detector_col_size = 0.4, # mm — VERIFIED
	detector_shape = BS.CURVED_DETECTOR,
	detector_row_offset = 0.0,
	detector_col_offset = 0.2, # quarter-detector offset

	# X-RAY SOURCE
	focal_spot_width = 0.4, # mm ⚠️ USED if sim_opts.use_focal_spot=true
	focal_spot_length = 0.5, # mm ⚠️ USED if sim_opts.use_focal_spot=true
	target_angle = 7.0, # degrees ⚠️ USED if sim_opts.use_heel_effect=true

	# FILTRATION
	flat_filter_material = :aluminum, # ⚠️ USED if sim_opts.use_flat_filter=true
	flat_filter_thickness = 2.5, # mm — ESTIMATE ⚠️ USED if sim_opts.use_flat_filter=true

	# DETECTOR PHYSICS — CdTe direct-conversion (v24.0 physics)
	detector_material = :cdte, # VERIFIED ⚠️ USED if sim_opts.use_detector_efficiency=true
	detector_depth = 1.6, # mm — VERIFIED (Konrad 2025) ⚠️ USED if sim_opts.use_detector_efficiency=true
	fill_factor_row = 0.95, # ⚠️ USED if sim_opts.use_fill_factor=true
	fill_factor_col = 0.95, # ⚠️ USED if sim_opts.use_fill_factor=true
	detection_gain = 1.0, # ⚠️ NOT USED — display only
	electronic_noise = 0.0, # PCCT: thresholds eliminate electronic noise ⚠️ NOT USED — display only

	# PCCT-SPECIFIC FIELDS — all actively used in pcct_forward_project()
	detector_type = :photon_counting,
	n_energy_bins = 4,
	energy_thresholds = [20.0, 35.0, 55.0, 70.0], # keV — clinical NAEOTOM thresholds
	energy_resolution = 10.0, # keV FWHM — superseded by unified DRM at fidelity=:pcct
	charge_sharing_fwhm = 0.08, # mm — superseded by Koch-Mehrin ODE at fidelity=:pcct
	dead_time_ns = 5.0, # ns — used in Yang 2025 pileup model
	pixel_mode = :standard # :standard (0.4mm), :uhr (0.2mm), :macro (0.8mm)
)

# ╔═╡ a369a59e-22b4-476f-a195-a16c9f13dc7e
# Use PCCT z-coverage for ALL scanners (smallest common z)
common_z_cm = n_recon_slices * 0.4 / 10.0  # 5.12 cm

# ╔═╡ 00000010-0000-0000-0000-000000000001
md"""
### 6.1 EICT Single-kVp Reconstruction
- FDK (fast, clinical baseline)
- Hybrid IR strength 3 added post-simulation via `hybrid_ir_reconstruct()`
"""

# ╔═╡ 00000011-0000-0000-0000-000000000001
recon_opts_eict_single = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (recon_xy, recon_xy, n_recon_slices),
	fov_cm = recon_fov_cm,
	# z_cm = n_recon_slices * 0.625 / 10.0,  # GE native slice thickness 0.625mm
	z_cm = common_z_cm,
	filter = :standard
)

# ╔═╡ 00000012-0000-0000-0000-000000000001
md"""
### 6.2 EICT Dual-kVp Reconstruction
- FDK for each energy (80 kVp, 140 kVp)
- VMI at 40, 70, 100, 140 keV
- Hybrid IR strength 3 added post-simulation
"""

# ╔═╡ 00000013-0000-0000-0000-000000000001
recon_opts_eict_dual = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (recon_xy, recon_xy, n_recon_slices),
	fov_cm = recon_fov_cm,
	# z_cm = n_recon_slices * 0.625 / 10.0,  # GE native slice thickness 0.625mm
	z_cm = common_z_cm,
	filter = :standard,
	vmi_energies = vmi_energies,
	vmi_basis = (:water, :iodine)
)

# ╔═╡ 00000014-0000-0000-0000-000000000001
md"""
### 6.3 PCCT Standard Reconstruction (512×512)
- FDK from 2×2 binned pixels (0.4mm dexels)
- VMI at 40, 70, 100, 140 keV (3-material basis)
- Hybrid IR strength 3 added post-simulation
"""

# ╔═╡ 00000015-0000-0000-0000-000000000001
recon_opts_pcct_standard = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (recon_xy, recon_xy, n_recon_slices),
	fov_cm = recon_fov_cm,
	# z_cm = n_recon_slices * 0.4 / 10.0,  # NAEOTOM native slice thickness 0.4mm
	z_cm = common_z_cm,
	filter = :standard,
	vmi_energies = vmi_energies,
	vmi_basis = [:water, :iodine, :calcium]
)

# ╔═╡ a0b1c2d3-e4f5-6789-abcd-000000000001
md"""
## 6.5 Water Phantom Calibration

Run a small water cylinder phantom through each scanner/protocol to measure μ\_water for accurate HU conversion. This avoids NIST reference mismatches from spectrum/filtration differences.
"""

# ╔═╡ a0b1c2d3-e4f5-6789-abcd-000000000002
begin
	# 20cm diameter water cylinder — small grid for speed
	water_size = (400, 400, 16)
	water_voxel_cm = (0.05, 0.05, 0.1)  # 0.5mm in-plane, 1mm slice

	water_mask = zeros(UInt8, water_size...)
	cx_w, cy_w = water_size[1] ÷ 2, water_size[2] ÷ 2
	r_w = 200  # 10cm radius = 200 voxels at 0.05cm
	for i in 1:water_size[1], j in 1:water_size[2]
		if (i - cx_w)^2 + (j - cy_w)^2 <= r_w^2
			water_mask[i, j, :] .= UInt8(1)
		end
	end

	# Air = label 0, Water = label 1
	air_material = XA.Material(
		"Air", 0.499, 85.7u"eV", 0.001205u"g/cm^3",
		Dict(7 => 0.7553, 8 => 0.2318, 18 => 0.0129)
	)
	water_materials = Dict(0 => air_material, 1 => XA.Materials.water)
	# GPU phantom for fast calibration
	water_mask_gpu = Metal.MtlArray(water_mask)
	water_phantom_gpu = BS.Phantom(water_mask_gpu, water_materials, water_voxel_cm)
end

# ╔═╡ 0d37c90b-89ec-4a2f-b03e-bf7514971d10
# Small recon for calibration — per-scanner z_cm from native row pitch
water_recon_opts_eict = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (256, 256, 8),
	fov_cm = 25.0,
	z_cm = 8 * scanner_eict.detector_row_size / 10.0,  # GE: 8×0.625mm = 0.5cm
	filter = :standard
)

# ╔═╡ 808b0a9b-a3ff-4fad-88f9-3bc2b1df543c
water_recon_opts_pcct = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (256, 256, 8),
	fov_cm = 25.0,
	z_cm = 8 * scanner_pcct_standard.detector_row_size / 10.0,  # NAEOTOM: 8×0.4mm = 0.32cm
	filter = :standard
)

# ╔═╡ ee23461e-378d-4172-8c01-5783e40ef3b8
"""
	extract_water_mu(vol)

Extract mean μ_water from center of a water phantom reconstruction volume.
Uses a 20% radius circular ROI at image center, middle half of z-slices.
Robust to any reconstruction matrix size.
"""
function extract_water_mu(vol)
	vol = Array(vol)
	nx, ny, nz = size(vol)
	cx, cy = nx ÷ 2, ny ÷ 2

	# 20% of image half-width — well inside the 20cm water cylinder
	r = nx ÷ 10

	# Middle half of z slices
	z_start = max(1, nz ÷ 4)
	z_end = min(nz, 3 * nz ÷ 4)

	vals = Float64[]
	for k in z_start:z_end, j in (cy - r):(cy + r), i in (cx - r):(cx + r)
		if (i - cx)^2 + (j - cy)^2 <= r^2
			push!(vals, vol[i, j, k])
		end
	end
	return mean(vals)
end

# ╔═╡ d9dfaa24-2254-4953-993f-f9fdb0c3326d
# EICT 120kVp calibration — workspace scoped locally to free GPU memory
μ_water_eict = let
	ws = BS.create_eict_workspace(
		scanner_eict, protocol_eict_single,
		sim_opts_eict, water_recon_opts_eict, water_phantom_gpu
	)
	BS.simulate!(
		ws, water_phantom_gpu, scanner_eict,
		protocol_eict_single, sim_opts_eict, water_recon_opts_eict
	)
	ws_fdk = BS.create_fdk_recon_workspace(
		ws.sino_noisy_out, ws.geom, water_recon_opts_eict.matrix_size,
		filter=BS.StandardFilter()
	)
	vol = Array(BS.reconstruct!(
		ws_fdk, ws.sino_noisy_out, ws.geom, water_recon_opts_eict.matrix_size
	))
	
	result = extract_water_mu(vol)

	# Kill GPU refs explicitly, then force full GC
    ws = nothing
    ws_fdk = nothing
    vol = nothing
    GC.gc(true)
    
    result  # only this tiny scalar survives
end

# ╔═╡ 5fe71f6b-e96e-45c7-ae60-dad1f13f110a
# EICT Dual-kVp calibration — workspace scoped locally to free GPU memory
(μ_water_dual_low, μ_water_dual_high) = let
	ws = BS.create_eict_dual_workspace(
		scanner_eict, protocol_eict_dual,
		sim_opts_eict, water_recon_opts_eict, water_phantom_gpu
	)
	BS.simulate!(
		ws, water_phantom_gpu, scanner_eict,
		protocol_eict_dual, sim_opts_eict, water_recon_opts_eict
	)
	# Low-kVp (80 kVp)
	ws_fdk_low = BS.create_fdk_recon_workspace(
		ws.sino_noisy_out_low, ws.geom, water_recon_opts_eict.matrix_size
	)
	vol_low = Array(BS.reconstruct!(
		ws_fdk_low, ws.sino_noisy_out_low, ws.geom, water_recon_opts_eict.matrix_size
	))
	# High-kVp (140 kVp)
	ws_fdk_high = BS.create_fdk_recon_workspace(
		ws.sino_noisy_out_high, ws.geom, water_recon_opts_eict.matrix_size
	)
	vol_high = Array(BS.reconstruct!(
		ws_fdk_high, ws.sino_noisy_out_high, ws.geom, water_recon_opts_eict.matrix_size
	))

	result = (extract_water_mu(vol_low), extract_water_mu(vol_high))

	ws = nothing
	ws_fdk_low = nothing
	ws_fdk_high = nothing
	vol_low = nothing
	vol_high = nothing
	GC.gc(true)

	result
end

# ╔═╡ 8396962e-dc91-436c-90fb-1525d5459a8a
# PCCT 140kVp calibration — workspace scoped locally to free GPU memory
μ_water_pcct = let
	ws = BS.create_workspace(
		scanner_pcct_standard, protocol_pcct_standard,
		sim_opts_pcct, water_recon_opts_pcct, water_phantom_gpu
	)
	BS.simulate!(
		ws, water_phantom_gpu, scanner_pcct_standard,
		protocol_pcct_standard, sim_opts_pcct, water_recon_opts_pcct
	)
	ws_fdk = BS.create_fdk_recon_workspace(
		ws.sino_noisy_out, ws.geom, water_recon_opts_pcct.matrix_size
	)
	vol = Array(BS.reconstruct!(
		ws_fdk, ws.sino_noisy_out, ws.geom, water_recon_opts_pcct.matrix_size
	))
	nx, ny, nz = size(vol)
	cx, cy = nx ÷ 2, ny ÷ 2
	r = nx ÷ 10
	z_start = max(1, nz ÷ 4)
	z_end = min(nz, 3 * nz ÷ 4)
	vals = Float64[]
	for k in z_start:z_end, j in (cy - r):(cy + r), i in (cx - r):(cx + r)
		if (i - cx)^2 + (j - cy)^2 <= r^2
			push!(vals, vol[i, j, k])
		end
	end
	
	result = mean(vals)

	ws = nothing
	ws_fdk = nothing
	vol = nothing
	GC.gc(true)

	result
end

# ╔═╡ a0b1c2d3-e4f5-6789-abcd-000000000004
md"""
**Water calibration values (μ\_water in cm⁻¹):**
- EICT 120 kVp: $(round(μ_water_eict, sigdigits=4))
- EICT Dual 80 kVp: $(round(μ_water_dual_low, sigdigits=4))
- EICT Dual 140 kVp: $(round(μ_water_dual_high, sigdigits=4))
- PCCT 140 kVp: $(round(μ_water_pcct, sigdigits=4))

Expected range: ~0.19–0.21 cm⁻¹
"""

# ╔═╡ 1b8aa963-7a95-4cc6-8670-de2e2caf28ab
md"""
## 7. Run Simulations
"""

# ╔═╡ 00000016-0000-0000-0000-000000000001
md"""
### 7.1 EICT Single-kVp (120 kVp)
"""

# ╔═╡ 00000017-0000-0000-0000-000000000001
# Simulate + reconstruct in a single let block — GPU workspaces are freed when block exits
(recon_eict_fdk_hu, recon_eict_hir_hu) = let
	recon_size = (recon_xy, recon_xy, n_recon_slices)

	# --- Simulate ---
	ws = BS.create_eict_workspace(
		scanner_eict, protocol_eict_single,
		sim_opts_eict, recon_opts_eict_single, phantom_gpu
	)
	@time BS.simulate!(
		ws, phantom_gpu, scanner_eict, protocol_eict_single,
		sim_opts_eict, recon_opts_eict_single
	)

	sino = ws.sino_noisy_out
	geom = ws.geom

	# --- FDK reconstruction → CPU HU ---
	ws_fdk = BS.create_fdk_recon_workspace(sino, geom, recon_size; filter=BS.StandardFilter())
	fdk_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, sino, geom, recon_size));
		μ_water=μ_water_eict
	)
	ws_fdk = nothing
	GC.gc(true)

	# --- Hybrid IR (strength 3) → CPU HU ---
	ws_hir = BS.create_hir_recon_workspace(sino, geom, recon_size; strength = 3)
	hir_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir, sino, geom, recon_size));
		μ_water=μ_water_eict
	)
	ws_hir = nothing
	ws = nothing
	sino = nothing
	GC.gc(true)

	(fdk_hu, hir_hu)
end;

# ╔═╡ f409ebb5-12b9-455b-ac5e-4e96b95e0410
function plot_scanner_comparison_eict_only(volumes, titles; slice_idx=32, window=(-300, 500))
	n = length(volumes)
	f = CM.Figure(size=(350 * n + 50, 400))

	for (i, (vol, title)) in enumerate(zip(volumes, titles))
		ax = CM.Axis(f[1, i], title=title, aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		CM.heatmap!(ax, vol[:, :, slice_idx], colormap=:grays, colorrange=window)
	end

	CM.Colorbar(f[1, n+1], colormap=:grays, colorrange=window, label="HU")
	f
end

# ╔═╡ 3944d6e8-2109-4464-aad2-dee03ad9b0f5
@bind z2 UI.Slider(axes(recon_eict_fdk_hu, 3); show_value = true, default = size(recon_eict_fdk_hu, 3) ÷ 2)

# ╔═╡ fc8b2628-60cc-4110-a0ba-b9c44b08ce6b
let
	fig = plot_scanner_comparison_eict_only(
		[recon_eict_fdk_hu, recon_eict_hir_hu],
		["EICT 120 kVp\n(FDK)", "EICT 120 kVp\n(HIR)"];
		slice_idx=z2
	)
	# CM.save(joinpath(FIGURES_DIR, "nb05_scanner_comparison.png"), fig)
	fig
end

# ╔═╡ 00000018-0000-0000-0000-000000000001
md"""
### 7.2 EICT Dual-kVp (80/140 kVp GSI)
"""

# ╔═╡ 00000020-0000-0000-0000-000000000001
# Simulate + reconstruct all dual-kVp in one let block — GPU workspaces freed when block exits
(recon_dual_80kVp_fdk_hu, recon_dual_140kVp_fdk_hu,
 recon_dual_80kVp_hir_hu, recon_dual_140kVp_hir_hu,
 dual_vmi_volumes) = let
	recon_size = (recon_xy, recon_xy, n_recon_slices)

	# --- Simulate ---
	ws = BS.create_eict_dual_workspace(
		scanner_eict, protocol_eict_dual,
		sim_opts_eict, recon_opts_eict_dual, phantom_gpu
	)
	@time result = BS.simulate!(
		ws, phantom_gpu, scanner_eict, protocol_eict_dual,
		sim_opts_eict, recon_opts_eict_dual
	)

	sino_low = ws.sino_noisy_out_low
	sino_high = ws.sino_noisy_out_high
	geom = ws.geom
	mat_map = result.mat_map

	# --- FDK: 80 kVp (low) → CPU HU ---
	ws_fdk_low = BS.create_fdk_recon_workspace(sino_low, geom, recon_size)
	fdk_80_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk_low, sino_low, geom, recon_size));
		μ_water=μ_water_dual_low
	)
	ws_fdk_low = nothing
	GC.gc(true)

	# --- FDK: 140 kVp (high) → CPU HU ---
	ws_fdk_high = BS.create_fdk_recon_workspace(sino_high, geom, recon_size)
	fdk_140_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk_high, sino_high, geom, recon_size));
		μ_water=μ_water_dual_high
	)
	ws_fdk_high = nothing
	GC.gc(true)

	# --- Hybrid IR: 80 kVp (low) → CPU HU ---
	ws_hir_low = BS.create_hir_recon_workspace(sino_low, geom, recon_size; strength = 3)
	hir_80_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir_low, sino_low, geom, recon_size));
		μ_water=μ_water_dual_low
	)
	ws_hir_low = nothing
	GC.gc(true)

	# --- Hybrid IR: 140 kVp (high) → CPU HU ---
	ws_hir_high = BS.create_hir_recon_workspace(sino_high, geom, recon_size; strength = 3)
	hir_140_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir_high, sino_high, geom, recon_size));
		μ_water=μ_water_dual_high
	)
	ws_hir_high = nothing; GC.gc(true)

	# --- VMI volumes (one at a time, GC between each) ---
	vmi_dict = Dict{Float64, Array{Float32, 3}}()
	for E in vmi_energies
		vmi_sino = BS.virtual_monoenergetic(mat_map, E)
		ws_fdk_vmi = BS.create_fdk_recon_workspace(vmi_sino, geom, recon_size)
		vmi_recon = Array(BS.reconstruct!(ws_fdk_vmi, vmi_sino, geom, recon_size))
		vmi_dict[E] = BS.vmi_to_hu(vmi_recon, E)
		ws_fdk_vmi = nothing
		GC.gc(true)
	end

	# --- Free the big simulation workspace last ---
	ws = nothing
	result = nothing
	sino_low = nothing
	sino_high = nothing
	mat_map = nothing

	GC.gc(true)

	(fdk_80_hu, fdk_140_hu, hir_80_hu, hir_140_hu, vmi_dict)
end;

# ╔═╡ 00000021-0000-0000-0000-000000000001
md"""
### 7.3 PCCT Standard (512×512, 0.4mm dexels)
"""

# ╔═╡ 00000023-0000-0000-0000-000000000001
# Simulate + reconstruct all PCCT in one let block — GPU workspaces freed when block exits
(recon_pcct_fdk_hu, recon_pcct_hir_hu, pcct_vmi_volumes) = let
	recon_size = (recon_xy, recon_xy, n_recon_slices)

	# --- Create workspace and simulate ---
	ws = BS.create_workspace(
		scanner_pcct_standard, protocol_pcct_standard,
		sim_opts_pcct, recon_opts_pcct_standard, phantom_gpu
	)
	@time result = BS.simulate!(
		ws, phantom_gpu, scanner_pcct_standard,
		protocol_pcct_standard, sim_opts_pcct, recon_opts_pcct_standard
	)

	# Grab what we need from ws before we start freeing recon workspaces
	geom = ws.geom
	combined_sino = ws.combined # still a GPU ref inside ws
	vmi_sino_buf = ws.vmi_sino
	mat_map = result.mat_map

	# --- FDK reconstruction → CPU HU ---
	ws_fdk = BS.create_fdk_recon_workspace(combined_sino, geom, recon_size)
	fdk_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, combined_sino, geom, recon_size));
		μ_water=μ_water_pcct
	)
	ws_fdk = nothing
	GC.gc(true)

	# --- Hybrid IR (strength 3) → CPU HU ---
	ws_hir = BS.create_hir_recon_workspace(combined_sino, geom, recon_size; strength = 3)
	hir_hu = BS.to_hounsfield(
		Array(BS.reconstruct!(ws_hir, combined_sino, geom, recon_size));
		μ_water=μ_water_pcct
	)
	ws_hir = nothing
	GC.gc(true)

	# --- VMI volumes (one at a time, GC between each) ---
	vmi_dict = Dict{Float64, Array{Float32, 3}}()
	for E in vmi_energies
		vmi_sino = BS.virtual_monoenergetic(mat_map, E; ws_output=vmi_sino_buf)
		ws_fdk_vmi = BS.create_fdk_recon_workspace(vmi_sino, geom, recon_size)
		vmi_recon = Array(BS.reconstruct!(ws_fdk_vmi, vmi_sino, geom, recon_size))
		vmi_dict[E] = BS.vmi_to_hu(vmi_recon, E)
		ws_fdk_vmi = nothing
		GC.gc(true)
	end

	# --- Free the big simulation workspace last ---
	ws = nothing
	result = nothing
	combined_sino = nothing
	vmi_sino_buf = nothing
	mat_map = nothing
	GC.gc(true)

	(fdk_hu, hir_hu, vmi_dict)
end;

# ╔═╡ 00000024-0000-0000-0000-000000000001
md"""
## 8. Visualize Results

All images displayed in Hounsfield Units (HU) using empirical water phantom calibration.
"""

# ╔═╡ 00000024-0000-0000-0000-000000000002
md"""
### 8.1 Three-Scanner FDK Comparison

Side-by-side FDK comparison of EICT Single-kVp, EICT Dual-kVp, and PCCT Standard at the same anatomical slice. Soft tissue window (W=400, L=40).
"""

# ╔═╡ 00000024-0000-0000-0000-000000000003
function plot_scanner_comparison(volumes, titles; slice_idx=64, window=(-300, 400))
	n = length(volumes)
	f = CM.Figure(size=(350 * n + 50, 400))

	for (i, (vol, title)) in enumerate(zip(volumes, titles))
		ax = CM.Axis(f[1, i], title=title, aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		CM.heatmap!(ax, vol[:, :, slice_idx], colormap=:grays, colorrange=window)
	end

	CM.Colorbar(f[1, n+1], colormap=:grays, colorrange=window, label="HU")
	f
end

# ╔═╡ 620d6d45-8e4d-40f1-80f7-c62bc8a65fcf
@bind z3 UI.Slider(
	axes(recon_eict_fdk_hu, 3);
	default = size(recon_eict_fdk_hu, 3) ÷ 2,
	show_value = true
)

# ╔═╡ 00000024-0000-0000-0000-000000000004
let
	fig = plot_scanner_comparison(
		[recon_eict_fdk_hu, recon_dual_80kVp_fdk_hu, recon_dual_140kVp_fdk_hu, recon_pcct_fdk_hu],
		["EICT 120 kVp\n(FDK)", "Dual 80 kVp\n(FDK)", "Dual 140 kVp\n(FDK)", "PCCT Standard\n(FDK)"];
		slice_idx=z3
	)
	CM.save(joinpath(FIGURES_DIR, "nb05_scanner_comparison.png"), fig)
	fig
end

# ╔═╡ c0d1e2f3-a4b5-6789-abcd-000000000001
md"""
### 8.2 FDK vs Hybrid IR Comparison

Two rows: FDK (top) vs Hybrid IR strength 3 (bottom) for each scanner. Soft tissue window.
"""

# ╔═╡ c0d1e2f3-a4b5-6789-abcd-000000000002
let
	fdk_volumes = [recon_eict_fdk_hu, recon_dual_80kVp_fdk_hu, recon_dual_140kVp_fdk_hu, recon_pcct_fdk_hu]
	hir_volumes = [recon_eict_hir_hu, recon_dual_80kVp_hir_hu, recon_dual_140kVp_hir_hu, recon_pcct_hir_hu]
	titles = ["EICT 120 kVp", "Dual 80 kVp", "Dual 140 kVp", "PCCT Standard"]
	window = (-300, 400)
	slice_idx = 32

	f = CM.Figure(size=(1500, 700))

	# Row 1: FDK
	for (i, (vol, title)) in enumerate(zip(fdk_volumes, titles))
		ax = CM.Axis(f[1, i], title="FDK: $title", aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		CM.heatmap!(ax, vol[:, :, slice_idx], colormap=:grays, colorrange=window)
	end

	# Row 2: Hybrid IR
	for (i, (vol, title)) in enumerate(zip(hir_volumes, titles))
		ax = CM.Axis(f[2, i], title="HIR-3: $title", aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		CM.heatmap!(ax, vol[:, :, slice_idx], colormap=:grays, colorrange=window)
	end

	CM.Colorbar(f[1:2, length(titles)+1], colormap=:grays, colorrange=window, label="HU")
	CM.save(joinpath(FIGURES_DIR, "nb05_fdk_vs_hir.png"), f)
	f
end

# ╔═╡ 82f0501e-ee06-4907-8b9e-45d74b0c8f19
md"""
### 8.3 EICT vs PCCT Detail View

Zoomed-in comparison of EICT Single-kVp (1.0mm dexels) and PCCT Standard (0.4mm dexels) in HU. Soft tissue window.
"""

# ╔═╡ 7bd356b3-7d43-40f8-a3d4-b7a818f57b69
function plot_detail_comparison(vol_eict_hu, vol_pcct_hu; slice_idx=32,
		zoom_center=(256, 256), zoom_size=100, window=(-300, 400))

	f = CM.Figure(size=(900, 750))

	labels = ["EICT 120 kVp", "PCCT Standard (0.4mm dexels)"]
	vols = [vol_eict_hu, vol_pcct_hu]
	cx, cy = zoom_center
	r = zoom_size ÷ 2

	for (row, (vol, label)) in enumerate(zip(vols, labels))
		img = vol[:, :, slice_idx]

		ax1 = CM.Axis(f[row, 1], title=label, aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		CM.heatmap!(ax1, img, colormap=:grays, colorrange=window)
		CM.lines!(ax1, [cx-r, cx+r, cx+r, cx-r, cx-r],
			[cy-r, cy-r, cy+r, cy+r, cy-r],
			color=:red, linewidth=2)

		ax2 = CM.Axis(f[row, 2], title="$label Zoomed", aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		CM.heatmap!(ax2, img[cx-r:cx+r, cy-r:cy+r],
			colormap=:grays, colorrange=window)
	end

	CM.Colorbar(f[1:2, 3], colormap=:grays, colorrange=window, label="HU")
	f
end

# ╔═╡ 01e76cfe-7061-455b-8f7e-0b07fbd17ca3
let
	fig = plot_detail_comparison(
		recon_eict_fdk_hu, recon_pcct_fdk_hu;
		slice_idx=32, zoom_center=(Int.(recon_xy ÷ 2.2), Int.(recon_xy ÷ 2.2)), zoom_size=min(80, recon_xy ÷ 4)
	)
	CM.save(joinpath(FIGURES_DIR, "nb05_detail_comparison.png"), fig)
	fig
end

# ╔═╡ 00000024-0000-0000-0000-000000000008
md"""
### 8.4 VMI Energy Sweep (Dual-kVp vs PCCT)

Comparison of Virtual Monoenergetic Images at 40, 70, 100, 140 keV.
"""

# ╔═╡ 00000024-0000-0000-0000-000000000009
function plot_vmi_comparison(dual_vmi_vols, pcct_vmi_vols; slice_idx=64, window=(-300, 500))
	energies = [40, 70, 100, 140]

	f = CM.Figure(size=(1200, 600))

	# Row 1: Dual-kVp VMI
	for (i, E) in enumerate(energies)
		ax = CM.Axis(f[1, i], title="Dual-kVp $(E) keV",
			aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		img = dual_vmi_vols[Float64(E)][:, :, slice_idx]
		CM.heatmap!(ax, img, colormap=:grays, colorrange=window)
	end

	# Row 2: PCCT VMI (from virtual_monoenergetic + reconstruct!)
	for (i, E) in enumerate(energies)
		ax = CM.Axis(f[2, i], title="PCCT $(E) keV",
			aspect=CM.DataAspect(),
			xticksvisible=false, yticksvisible=false,
			xticklabelsvisible=false, yticklabelsvisible=false)
		img = pcct_vmi_vols[Float64(E)][:, :, slice_idx]
		CM.heatmap!(ax, img, colormap=:grays, colorrange=window)
	end

	CM.Colorbar(f[1:2, 5], colormap=:grays, colorrange=window, label="HU")
	f
end

# ╔═╡ 7b2d3e4f-a5c6-4890-bcde-000000000001
@bind z_vmi UI.Slider(
	axes(recon_eict_fdk_hu, 3);
	default = size(recon_eict_fdk_hu, 3) ÷ 2,
	show_value = true
)

# ╔═╡ 00000024-0000-0000-0000-000000000010
let
	fig = plot_vmi_comparison(dual_vmi_volumes, pcct_vmi_volumes; slice_idx=z_vmi)
	CM.save(joinpath(FIGURES_DIR, "nb05_vmi_comparison.png"), fig)
	fig
end

# ╔═╡ 00000024-0000-0000-0000-000000000011
md"""
### 8.5 Noise & HU Accuracy Analysis

ROI statistics comparing noise (std) and mean HU for FDK and Hybrid IR across all scanners.
"""

# ╔═╡ 00000024-0000-0000-0000-000000000012
function analyze_roi_stats(volumes, names; slice_idx=32, roi_center=nothing, roi_radius=20)
	stats = []

	for (vol, name) in zip(volumes, names)
		img = vol[:, :, slice_idx]
		cx, cy = roi_center === nothing ? (size(img, 1) ÷ 2, size(img, 2) ÷ 2) : roi_center
		r = roi_radius

		# Extract circular ROI
		roi_vals = Float64[]
		for i in (cx-r):(cx+r), j in (cy-r):(cy+r)
			if (i - cx)^2 + (j - cy)^2 <= r^2
				push!(roi_vals, img[i, j])
			end
		end

		push!(stats, (
			name = name,
			mean_hu = mean(roi_vals),
			std_hu = std(roi_vals),
			n_pixels = length(roi_vals)
		))
	end

	return stats
end

# ╔═╡ 00000024-0000-0000-0000-000000000013
let
	volumes = [
		recon_eict_fdk_hu, recon_eict_hir_hu,
		recon_dual_80kVp_fdk_hu, recon_dual_80kVp_hir_hu,
		recon_dual_140kVp_fdk_hu, recon_dual_140kVp_hir_hu,
		recon_pcct_fdk_hu, recon_pcct_hir_hu
	]
	names = [
		"EICT FDK", "EICT HIR",
		"Dual 80 FDK", "Dual 80 HIR",
		"Dual 140 FDK", "Dual 140 HIR",
		"PCCT FDK", "PCCT HIR"
	]
	n_vols = length(volumes)

	stats = analyze_roi_stats(volumes, names)

	f = CM.Figure(size=(1200, 350))

	ax1 = CM.Axis(f[1, 1], title="Mean HU", ylabel="HU",
		xticks=(1:n_vols, names), xticklabelrotation=π/4)
	CM.barplot!(ax1, 1:n_vols, [s.mean_hu for s in stats],
		color=repeat([:steelblue, :royalblue], n_vols ÷ 2))
	CM.hlines!(ax1, [0], color=:red, linestyle=:dash, label="Water (0 HU)")

	ax2 = CM.Axis(f[1, 2], title="Noise (Std Dev)", ylabel="HU",
		xticks=(1:n_vols, names), xticklabelrotation=π/4)
	CM.barplot!(ax2, 1:n_vols, [s.std_hu for s in stats],
		color=repeat([:coral, :salmon], n_vols ÷ 2))

	CM.save(joinpath(FIGURES_DIR, "nb05_noise_stats.png"), f)
	f
end

# ╔═╡ c0d1e2f3-a4b5-6789-abcd-000000000003
md"""
### 8.6 Noise Reduction: FDK → Hybrid IR

Bar chart showing noise (std dev in uniform ROI) for FDK vs Hybrid IR, with noise reduction percentage.
"""

# ╔═╡ c0d1e2f3-a4b5-6789-abcd-000000000004
let
	fdk_volumes = [recon_eict_fdk_hu, recon_dual_80kVp_fdk_hu, recon_dual_140kVp_fdk_hu, recon_pcct_fdk_hu]
	hir_volumes = [recon_eict_hir_hu, recon_dual_80kVp_hir_hu, recon_dual_140kVp_hir_hu, recon_pcct_hir_hu]
	scanner_names = ["EICT 120kVp", "Dual 80kVp", "Dual 140kVp", "PCCT Std"]
	n_scanners = length(scanner_names)

	fdk_stats = analyze_roi_stats(fdk_volumes, scanner_names)
	hir_stats = analyze_roi_stats(hir_volumes, scanner_names)

	fdk_noise = [s.std_hu for s in fdk_stats]
	hir_noise = [s.std_hu for s in hir_stats]
	reduction_pct = @. 100.0 * (1.0 - hir_noise / fdk_noise)

	f = CM.Figure(size=(900, 400))

	ax = CM.Axis(f[1, 1], title="Noise Comparison: FDK vs Hybrid IR (Strength 3)",
		ylabel="Noise (HU std dev)",
		xticks=(1:n_scanners, scanner_names))

	# Grouped bar: FDK and HIR side-by-side
	x_fdk = [i - 0.2 for i in 1:n_scanners]
	x_hir = [i + 0.2 for i in 1:n_scanners]
	CM.barplot!(ax, x_fdk, fdk_noise, width=0.35, color=:coral, label="FDK")
	CM.barplot!(ax, x_hir, hir_noise, width=0.35, color=:steelblue, label="Hybrid IR")

	# Annotate noise reduction %
	for (i, pct) in enumerate(reduction_pct)
		CM.text!(ax, x_hir[i], hir_noise[i] + 0.5,
			text="−$(round(Int, pct))%", align=(:center, :bottom), fontsize=12)
	end

	CM.axislegend(ax, position=:rt)

	CM.save(joinpath(FIGURES_DIR, "nb05_noise_reduction.png"), f)
	f
end

# ╔═╡ aaa00001-0000-0000-0000-a00000000001
md"""
### 8.7 Ground Truth Overlay — EICT vs PCCT

Resample the XCAT phantom labels onto the reconstruction grid using `resample_to_recon`, then show side-by-side with EICT and PCCT FDK reconstructions.
"""

# ╔═╡ aaa00002-0000-0000-0000-a00000000002
# Resample phantom labels onto each scanner's actual reconstruction grid
(ground_truth_eict, ground_truth_pcct) = let
	# EICT: use scanner_eict + recon_opts_eict_single
	geom_eict = BS.CTGeometry(scanner_eict;
		n_angles = 1,
		n_rows = eict_det_rows,
		n_cols = eict_det_cols,
		fov_cm = recon_opts_eict_single.fov_cm,
		z_cm = recon_opts_eict_single.z_cm
	)
	gt_eict = BS.resample_to_recon(
		phantom_gpu, geom_eict, recon_opts_eict_single.matrix_size
	)

	# PCCT: use scanner_pcct_standard + recon_opts_pcct_standard
	geom_pcct = BS.CTGeometry(scanner_pcct_standard;
		n_angles = 1,
		n_rows = pcct_det_rows,
		n_cols = pcct_det_cols,
		fov_cm = recon_opts_pcct_standard.fov_cm,
		z_cm = recon_opts_pcct_standard.z_cm
	)
	gt_pcct = BS.resample_to_recon(
		phantom_gpu, geom_pcct, recon_opts_pcct_standard.matrix_size
	)

	(gt_eict, gt_pcct)
end

# ╔═╡ aaa00003-0000-0000-0000-a00000000003
@bind z_gt UI.Slider(
	axes(recon_eict_fdk_hu, 3);
	default = size(recon_eict_fdk_hu, 3) ÷ 2,
	show_value = true
)

# ╔═╡ aaa00004-0000-0000-0000-a00000000004
let
	slice = z_gt
	window = (-300, 400)

	f = CM.Figure(size=(1200, 650))

	# Row 1: EICT — Ground Truth | FDK
	ax1 = CM.Axis(f[1, 1], title="EICT Ground Truth", aspect=CM.DataAspect(),
		xticksvisible=false, yticksvisible=false,
		xticklabelsvisible=false, yticklabelsvisible=false)
	CM.heatmap!(ax1, Float32.(ground_truth_eict[:, :, slice]),
		colormap=:glasbey_hv_n256)

	ax2 = CM.Axis(f[1, 2], title="EICT 120 kVp (FDK)", aspect=CM.DataAspect(),
		xticksvisible=false, yticksvisible=false,
		xticklabelsvisible=false, yticklabelsvisible=false)
	CM.heatmap!(ax2, recon_eict_fdk_hu[:, :, slice],
		colormap=:grays, colorrange=window)

	# Row 2: PCCT — Ground Truth | FDK
	ax3 = CM.Axis(f[2, 1], title="PCCT Ground Truth", aspect=CM.DataAspect(),
		xticksvisible=false, yticksvisible=false,
		xticklabelsvisible=false, yticklabelsvisible=false)
	CM.heatmap!(ax3, Float32.(ground_truth_pcct[:, :, slice]),
		colormap=:glasbey_hv_n256)

	ax4 = CM.Axis(f[2, 2], title="PCCT Standard (FDK)", aspect=CM.DataAspect(),
		xticksvisible=false, yticksvisible=false,
		xticklabelsvisible=false, yticklabelsvisible=false)
	CM.heatmap!(ax4, recon_pcct_fdk_hu[:, :, slice],
		colormap=:grays, colorrange=window)

	CM.Colorbar(f[1:2, 3], colormap=:grays, colorrange=window, label="HU")
	CM.save(joinpath(FIGURES_DIR, "nb05_ground_truth_overlay.png"), f)
	f
end

# ╔═╡ 00000025-0000-0000-0000-000000000001
# md"""
# ## 9. Summary

# | Scanner | Dexel | Protocol | Recon | Output |
# |---------|-------|----------|-------|--------|
# | GE Revolution Apex | 1.0mm | 120 kVp, 300 mA, 984 views | FDK + Hybrid IR (strength 3) | $(recon_xy)×$(recon_xy)×$(n_recon_slices) HU |
# | GE Revolution Apex GSI | 1.0mm | 80/140 kVp, 984 views | FDK + Hybrid IR + VMI | $(recon_xy)×$(recon_xy)×$(n_recon_slices) + VMI HU |
# | NAEOTOM Alpha Standard | 0.4mm | 140 kVp, 300 mA, $(pcct_n_views) views | FDK + Hybrid IR + VMI | $(recon_xy)×$(recon_xy)×$(n_recon_slices) HU |

# **Key points:**
# - Input phantom: $(size(phantom_labeled)) ($(round.(voxel_size_cm .* 10, digits=2)) mm) — downsample factor $(DOWNSAMPLE_FACTOR)
# - Recon output: $(recon_xy)×$(recon_xy)×$(n_recon_slices) — auto-matched to phantom resolution (capped at 512)
# - **HU calibration:** Empirical water phantom (20cm cylinder) per scanner/protocol
# - **Hybrid IR:** TRUE iterative reconstruction (PWLS + FDK init, not blending) for all 3 scanners
# - **PCCT v24.0 physics:** Koch-Mehrin charge cloud, K-fluorescence, Hecht CCE, Yang pileup, unified DRM
# - All simulations run on Metal GPU (mask auto-uploaded internally); `Array()` for CPU conversion
# - Phantom API: `compute_μ(phantom, energy)` for on-demand attenuation
# """

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000001
# ╠═d6d62fae-012d-11f1-1efc-67e7f251ff8c
# ╠═cca08041-b05c-4045-a462-18b30fd0559f
# ╠═3ad61ec7-ba66-449d-8fd1-e79a2345a9d7
# ╠═3cbd1220-7118-42c7-9562-27c8c2e1b608
# ╠═f36f0c25-3c0c-4202-b890-917b031fa9e8
# ╠═2a00221a-e861-4d53-bba6-bde7b1bc909f
# ╠═c071c54d-0950-4260-83b7-7dc79300609e
# ╠═dc9e51f9-2531-4483-b80a-622f5ecf4d0c
# ╠═2ab74942-1680-47dd-a8dc-f5242387253e
# ╠═4ca1063f-1cc1-4253-a411-4817f1e584a2
# ╠═00000002-0000-0000-0000-000000000001
# ╟─20920020-fd6b-4a64-9e19-e00bfd616ee6
# ╠═c744a9d3-5810-4465-82ee-2b8d9b5f68b1
# ╠═7812a13f-0120-47c0-9bbf-e03918a8113a
# ╠═b1c2c0ee-e2ca-4ac3-a643-376aeafb0c6c
# ╠═b38ffd25-d549-475d-b92a-5d08ebd6cc53
# ╠═72791f6b-0178-44dd-8f36-8071aa451b7c
# ╠═72791f6b-0178-44dd-8f36-8071aa451b8c
# ╟─a1858da5-8231-460e-a471-35115d2b1476
# ╠═72791f6b-0178-44dd-8f36-8071aa451b9c
# ╠═702aca88-e592-4718-9f7a-1d3aa7c950ce
# ╟─00000003-0000-0000-0000-000000000001
# ╟─237bd8ca-6ea5-4aff-81f4-8a77bb45f4fd
# ╟─5dee3049-e56a-41dd-8148-fd9f5de3411b
# ╟─00000004-0000-0000-0000-000000000001
# ╠═31006d4c-ad69-43db-a833-71182e2cf069
# ╠═cbd9b4b0-828c-4105-9583-b1b4f2ec74b6
# ╠═d70bc21c-0b77-4504-af16-7154980a32db
# ╠═3deeccc4-d188-495d-82a2-868a947840d5
# ╟─00000005-0000-0000-0000-000000000001
# ╟─90070343-33a7-4873-ba42-32d1e80bde31
# ╠═f3861cff-abd7-4cb9-9eb6-2a4f63677d29
# ╟─aebd1afc-64d9-4be3-a7d9-10e2c7a3d0ad
# ╠═bb8ec964-3fe8-4b67-8e32-356e8d10b942
# ╟─6562ecaa-0df4-460b-b2aa-7c1f53505070
# ╟─349fcb3d-b8fa-4423-8cdd-05a1a842c345
# ╟─b1a2c3d4-e5f6-7890-abcd-111111111111
# ╠═619f09c1-315e-42b6-9e44-2f36a98f8fee
# ╠═f8fd4ab6-7c18-4d89-9045-41cf60666d63
# ╠═b1a2c3d4-e5f6-7890-abcd-222222222222
# ╠═a21a509a-3cb2-452d-ae46-53f1339e0f37
# ╟─ea2db1d2-d7ef-4c5c-9d74-e273ffda11b1
# ╠═c9b47012-24b1-42da-bd74-d77fb8f55cb4
# ╠═00000006-0000-0000-0000-000000000001
# ╟─b733752a-4e10-4909-82c0-35e333fb2758
# ╟─e52eb0ca-a322-4cbc-9c92-9d5240c55664
# ╠═f319b38b-7378-43fc-91a7-2db0554ab934
# ╠═00000007-0000-0000-0000-000000000001
# ╟─a70bc722-e769-4132-b082-b0e89a68228d
# ╟─a70bc722-e769-4132-b082-b0e89a68228f
# ╠═a70bc722-e769-4132-b082-b0e89a68228e
# ╟─a70bc722-e769-4132-b082-b0e89a6822a0
# ╠═a70bc722-e769-4132-b082-b0e89a6822a1
# ╟─a70bc722-e769-4132-b082-b0e89a6822a2
# ╠═4ae79dad-dd72-4db6-98f5-0f2d28cb9f5e
# ╠═a70bc722-e769-4132-b082-b0e89a6822a3
# ╟─a70bc722-e769-4132-b082-b0e89a6822b0
# ╠═a70bc722-e769-4132-b082-b0e89a6822b1
# ╠═a70bc722-e769-4132-b082-b0e89a6822b2
# ╟─937ad7ea-f5c3-42c1-9f5d-6c6600fffaba
# ╠═00000009-0000-0000-0000-000000000001
# ╠═a369a59e-22b4-476f-a195-a16c9f13dc7e
# ╟─00000010-0000-0000-0000-000000000001
# ╠═00000011-0000-0000-0000-000000000001
# ╟─00000012-0000-0000-0000-000000000001
# ╠═00000013-0000-0000-0000-000000000001
# ╟─00000014-0000-0000-0000-000000000001
# ╠═00000015-0000-0000-0000-000000000001
# ╟─a0b1c2d3-e4f5-6789-abcd-000000000001
# ╠═a0b1c2d3-e4f5-6789-abcd-000000000002
# ╠═0d37c90b-89ec-4a2f-b03e-bf7514971d10
# ╠═808b0a9b-a3ff-4fad-88f9-3bc2b1df543c
# ╠═ee23461e-378d-4172-8c01-5783e40ef3b8
# ╠═d9dfaa24-2254-4953-993f-f9fdb0c3326d
# ╠═5fe71f6b-e96e-45c7-ae60-dad1f13f110a
# ╠═8396962e-dc91-436c-90fb-1525d5459a8a
# ╟─a0b1c2d3-e4f5-6789-abcd-000000000004
# ╟─1b8aa963-7a95-4cc6-8670-de2e2caf28ab
# ╟─00000016-0000-0000-0000-000000000001
# ╠═00000017-0000-0000-0000-000000000001
# ╠═f409ebb5-12b9-455b-ac5e-4e96b95e0410
# ╟─3944d6e8-2109-4464-aad2-dee03ad9b0f5
# ╟─fc8b2628-60cc-4110-a0ba-b9c44b08ce6b
# ╟─00000018-0000-0000-0000-000000000001
# ╠═00000020-0000-0000-0000-000000000001
# ╟─00000021-0000-0000-0000-000000000001
# ╠═00000023-0000-0000-0000-000000000001
# ╟─00000024-0000-0000-0000-000000000001
# ╟─00000024-0000-0000-0000-000000000002
# ╠═00000024-0000-0000-0000-000000000003
# ╟─620d6d45-8e4d-40f1-80f7-c62bc8a65fcf
# ╟─00000024-0000-0000-0000-000000000004
# ╟─c0d1e2f3-a4b5-6789-abcd-000000000001
# ╟─c0d1e2f3-a4b5-6789-abcd-000000000002
# ╟─82f0501e-ee06-4907-8b9e-45d74b0c8f19
# ╟─7bd356b3-7d43-40f8-a3d4-b7a818f57b69
# ╟─01e76cfe-7061-455b-8f7e-0b07fbd17ca3
# ╟─00000024-0000-0000-0000-000000000008
# ╟─00000024-0000-0000-0000-000000000009
# ╟─7b2d3e4f-a5c6-4890-bcde-000000000001
# ╟─00000024-0000-0000-0000-000000000010
# ╟─00000024-0000-0000-0000-000000000011
# ╟─00000024-0000-0000-0000-000000000012
# ╟─00000024-0000-0000-0000-000000000013
# ╟─c0d1e2f3-a4b5-6789-abcd-000000000003
# ╟─c0d1e2f3-a4b5-6789-abcd-000000000004
# ╟─aaa00001-0000-0000-0000-a00000000001
# ╠═aaa00002-0000-0000-0000-a00000000002
# ╟─aaa00003-0000-0000-0000-a00000000003
# ╟─aaa00004-0000-0000-0000-a00000000004
# ╟─00000025-0000-0000-0000-000000000001
