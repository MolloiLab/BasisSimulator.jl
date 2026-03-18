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

# ╔═╡ b0000001-0006-0001-0001-000000000001
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
	# Pkg.update()

	using Revise
end

# ╔═╡ b0000001-0006-0001-0001-000000000002
# ╠═╡ show_logs = false
using PythonCall

# ╔═╡ b0000001-0006-0001-0001-000000000003
using Unitful: @u_str

# ╔═╡ b0000001-0006-0001-0001-000000000004
using LinearAlgebra

# ╔═╡ b0000001-0006-0001-0001-000000000005
using FFTW

# ╔═╡ b0000001-0006-0001-0001-000000000006
using Random

# ╔═╡ b0000001-0006-0001-0001-000000000007
# ╠═╡ show_logs = false
using Metal

# ╔═╡ b0000001-0006-0001-0001-000000000008
md"""
# GE Revolution Apex — CatSim Baseline Reference (6 Protocols)

**BasisSimulator vs. XCIST/CatSim: Multi-Protocol FDK Comparison**
Generates reference PNGs + rod measurement CSVs for regression testing.

Clinically realistic simulation of the **GE Revolution Apex** scanner imaging a
**Gammex 472** phantom across 6 scan protocols (3 dose levels + 3 kVp values).

| Scan | kVp | Target CTDIvol | Est. mA (0.5s) |
|------|-----|----------------|----------------|
| 1 | 120 | ~3 mGy | 75 mA |
| 2 | 120 | ~10 mGy | 250 mA |
| 3 | 120 | ~20 mGy | 500 mA |
| 4 | 80 | ~10 mGy | 700 mA |
| 5 | 100 | ~10 mGy | 360 mA |
| 6 | 140 | ~10 mGy | 185 mA |

Julia structs are the **single source of truth**: `Scanner → CTProtocol → SimOptions → ReconOptions → Phantom`. CatSim parameters are **derived** from them.

---
"""

# ╔═╡ b0000001-0006-0001-0001-000000000009
# ╠═╡ show_logs = false
import PlutoUI as UI

# ╔═╡ b0000001-0006-0001-0001-000000000010
# ╠═╡ show_logs = false
import BasisSimulator as BS

# ╔═╡ b0000001-0006-0001-0001-000000000011
# ╠═╡ show_logs = false
import CairoMakie as CM

# ╔═╡ b0000001-0006-0001-0001-000000000012
import Statistics: mean, std, cor

# ╔═╡ b0000001-0006-0001-0001-000000000013
import XrayAttenuation as XA

# ╔═╡ b0000001-0006-0001-0001-000000000014
const FIGURES_DIR = joinpath(dirname(@__DIR__), "figures")

# ╔═╡ b0000001-0006-0001-0001-000000000015
UI.TableOfContents()

# ╔═╡ b0000001-0006-0001-0002-000000000001
md"""
## 1. Scanner — GE Revolution Apex

**Verified specs (PMC6448170, FDA K213715):**
- SID = 625.6mm, 256 rows × 0.625mm = 160mm z-coverage
- Gemstone Clarity detector (GOS), 80cm bore

**Estimated specs:**
- SDD ≈ 1100mm (magnification ≈ 1.758)
- 834 columns (50cm scan diameter / 0.6mm pitch at isocenter)

BasisSimulator stores detector pitch at **isocenter**. CatSim expects **detector face** pitch.
The wrapper converts: `face_pitch = iso_pitch × magnification`.
"""

# ╔═╡ b0000001-0006-0001-0002-000000000002
scanner = BS.Scanner(
	# GEOMETRY — from MC validation (PMC6448170)
	source_to_isocenter = 625.6,  # mm — VERIFIED
	source_to_detector = 1100.0,  # mm — ESTIMATED (magnification ≈ 1.758)

	# DETECTOR ARRAY — full physical hardware
	detector_rows = 256,          # physical max (256 × 0.625mm = 160mm)
	detector_cols = 834,          # 50cm scan diameter / 0.6mm pitch
	detector_row_size = 0.625,    # mm at isocenter — VERIFIED (FDA K213715)
	detector_col_size = 0.6,      # mm at isocenter — ESTIMATED
	detector_shape = BS.CURVED_DETECTOR,

	# X-RAY SOURCE
	focal_spot_width = 1.0,       # mm — ESTIMATED
	focal_spot_length = 1.0,      # mm — ESTIMATED
	target_angle = 7.0,           # degrees — typical GE value

	# FILTRATION
	flat_filter_material = :aluminum,
	flat_filter_thickness = 2.5,  # mm — ESTIMATED

	# DETECTOR PHYSICS — GE Gemstone Clarity (GOS scintillator)
	detector_material = :gos,
	detector_depth = 3.0,         # mm — ESTIMATED
	fill_factor_row = 0.9,
	fill_factor_col = 0.9,
	detection_gain = 1.0,
)

# ╔═╡ b0000001-0006-0001-0003-000000000001
md"""
## 2. Protocols — 6 Scan Configurations

All protocols use **0.5s rotation** and **40mm collimation** (64 active rows × 0.625mm).
- Scans 1–3: Dose variation at 120 kVp (3, 10, 20 mGy)
- Scans 4–6: kVp variation at ~10 mGy (80, 100, 140 kVp)
"""

# ╔═╡ b0000001-0006-0001-0003-000000000002
begin
	rotation_time = 0.5  # seconds (from protocol table)
	collimation_mm = 40.0  # 64 × 0.625mm active rows = 4cm z-coverage (< 5cm phantom)
	n_views = 984  # standard GE Revolution
end

# ╔═╡ b0000001-0006-0001-0003-000000000003
# Ordered scan definitions — single source of truth for the 6 protocols
SCANS = [
	(id=1, name="120kVp_3mGy",  label="Scan 1: 120 kVp / 3 mGy",  kvp=120, mA=75.0),
	(id=2, name="120kVp_10mGy", label="Scan 2: 120 kVp / 10 mGy", kvp=120, mA=250.0),
	(id=3, name="120kVp_20mGy", label="Scan 3: 120 kVp / 20 mGy", kvp=120, mA=500.0),
	(id=4, name="80kVp_10mGy",  label="Scan 4: 80 kVp / 10 mGy",  kvp=80,  mA=700.0),
	(id=5, name="100kVp_10mGy", label="Scan 5: 100 kVp / 10 mGy", kvp=100, mA=360.0),
	(id=6, name="140kVp_10mGy", label="Scan 6: 140 kVp / 10 mGy", kvp=140, mA=185.0),
]

# ╔═╡ b0000001-0006-0001-0003-000000000004
# Create CTProtocol for each scan
protocols = Dict(
	sc.name => BS.CTProtocol(
		kVp = sc.kvp,
		mA = sc.mA,
		views = n_views,
		rotation_time = rotation_time,
		collimation_mm = collimation_mm,
	)
	for sc in SCANS
)

# ╔═╡ b0000001-0006-0001-0004-000000000001
md"""
## 3. SimOptions & ReconOptions

Reconstruction geometry is derived from scanner + collimation.
- **FOV:** 35cm (Gammex 472 body = 33cm diameter)
- **Collimation:** 40mm → 64 active rows → 4cm recon z-extent
- **Slice thickness:** 1.25mm → 32 slices (clinical GE body standard)
- **Phantom z:** 5cm (actual Gammex 472 thickness, > 4cm collimation)
"""

# ╔═╡ b0000001-0006-0001-0004-000000000002
sim_opts = BS.SimOptions(fidelity = :eict, seed = 1234)

# ╔═╡ b0000001-0006-0001-0004-000000000003
begin
	recon_fov_cm = 35.0
	recon_xy = 512
	slice_thickness_mm = 1.25  # clinical GE body standard
	active_rows = round(Int, collimation_mm / scanner.detector_row_size)  # 64
	recon_z_cm = collimation_mm / 10.0  # 4.0 cm
	n_recon_slices = round(Int, collimation_mm / slice_thickness_mm)  # 32

	# Gammex 472 physical thickness (> collimation z-coverage)
	phantom_z_cm = 5.0
end

# ╔═╡ b0000001-0006-0001-0004-000000000004
recon_opts = BS.ReconOptions(
	algorithm = :fdk,
	matrix_size = (recon_xy, recon_xy, n_recon_slices),
	fov_cm = recon_fov_cm,
	z_cm = recon_z_cm,
	filter = :standard,
)

# ╔═╡ b0000001-0006-0001-0005-000000000001
md"""
---
## 4. CatSim Integration (Struct-Based Wrappers)

All wrapper functions accept **Julia structs** directly. CatSim parameters are derived
from struct fields.

Key conversions:
- Detector pitch: `face_pitch = iso_pitch × (SDD / SID)`
- Active rows: `round(Int, protocol.collimation_mm / scanner.detector_row_size)`
- Slice thickness: `recon_opts.z_cm × 10 / recon_opts.matrix_size[3]`
"""

# ╔═╡ b0000001-0006-0001-0005-000000000002
const _catsim = Ref{Py}()

# ╔═╡ b0000001-0006-0001-0005-000000000003
const _recon_mod = Ref{Py}()

# ╔═╡ b0000001-0006-0001-0005-000000000004
const _np = Ref{Py}()

# ╔═╡ b0000001-0006-0001-0005-000000000005
const _cfg_path = Ref("")

# ╔═╡ b0000001-0006-0001-0005-000000000006
function catsim_init()
	if !isassigned(_catsim)
		_catsim[] = pyimport("gecatsim")
		_recon_mod[] = pyimport("gecatsim.reconstruction.pyfiles.recon")
		_np[] = pyimport("numpy")

		spec = pyimport("importlib.util")
		gecatsim_spec = spec.find_spec("gecatsim")
		gecatsim_path = pyconvert(String, gecatsim_spec.origin)
		base_path = dirname(dirname(gecatsim_path))
		_cfg_path[] = joinpath(base_path, "gecatsim", "examples", "cfg")
	end
	return _catsim[], _recon_mod[], _np[], _cfg_path[]
end

# ╔═╡ b0000001-0006-0001-0005-000000000007
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

# ╔═╡ b0000001-0006-0001-0005-000000000008
"""
	catsim_configure_scanner!(ct, scanner, protocol)

Configures CatSim scanner geometry from `BS.Scanner` and `BS.CTProtocol` structs.

**Key conversions:**
- Detector pitch: isocenter → face via `magnification = SDD / SID`
- Active rows: from `protocol.collimation_mm / scanner.detector_row_size`

**CRITICAL FIX:** Sets `detectorColsPerMod = 1` to prevent braided sinograms.
"""
function catsim_configure_scanner!(ct, scanner, protocol)
	magnification = scanner.source_to_detector / scanner.source_to_isocenter

	# Active rows from collimation (or full detector if no collimation)
	n_active_rows = if protocol.collimation_mm !== nothing
		round(Int, protocol.collimation_mm / scanner.detector_row_size)
	else
		scanner.detector_rows
	end

	ct.scanner.sid = scanner.source_to_isocenter
	ct.scanner.sdd = scanner.source_to_detector
	ct.scanner.detectorColCount = scanner.detector_cols
	ct.scanner.detectorRowCount = n_active_rows
	ct.scanner.detectorColSize = scanner.detector_col_size * magnification  # iso → face
	ct.scanner.detectorRowSize = scanner.detector_row_size * magnification  # iso → face

	# FIX 1: Prevent "Braided" sinograms — every pixel is its own module
	ct.scanner.detectorColsPerMod = 1
	ct.scanner.detectorRowsPerMod = n_active_rows

	# FIX 2: Prevent "Squashed" sinograms — zero inter-module gap
	ct.scanner.detectorColSkip = 0.0
	ct.scanner.detectorRowSkip = 0.0

	return ct
end

# ╔═╡ b0000001-0006-0001-0005-000000000009
"""
	catsim_configure_protocol!(ct, protocol)

Configures CatSim protocol from a `BS.CTProtocol` struct.
"""
function catsim_configure_protocol!(ct, protocol)
	ct.protocol.mA = protocol.mA
	ct.protocol.viewsPerRotation = protocol.views
	ct.protocol.viewCount = protocol.views
	ct.protocol.stopViewId = protocol.views - 1
	ct.protocol.rotationTime = protocol.rotation_time
	ct.protocol.spectrumFilename = "tungsten_tar7.0_$(Int(protocol.kVp))_filt.dat"
	return ct
end

# ╔═╡ b0000001-0006-0001-0005-000000000010
"""
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=nothing)

Configures CatSim FDK reconstruction from a `BS.ReconOptions` struct.
Slice thickness is derived: `z_cm × 10 / n_slices`.

Pass `μ_water_cm` (cm⁻¹) for kVp-specific water calibration.
Falls back to 0.02 mm⁻¹ (≈ 0.2 cm⁻¹) if not provided.
"""
function catsim_configure_recon!(ct, recon_opts; μ_water_cm=nothing)
	xc, _, _, cfg_path = catsim_init()
	xc.source_cfg(joinpath(cfg_path, "Recon_Sample_2d.cfg"), ct)

	# Derive slice thickness from recon_opts
	n_slices = recon_opts.matrix_size[3]
	slice_thick_mm = recon_opts.z_cm * 10.0 / n_slices  # cm → mm

	ct.recon.fov = recon_opts.fov_cm * 10.0  # cm → mm
	ct.recon.imageSize = recon_opts.matrix_size[1]
	ct.recon.sliceCount = n_slices
	ct.recon.sliceThickness = slice_thick_mm

	ct.recon.unit = "HU"
	ct.recon.mu = μ_water_cm !== nothing ? μ_water_cm / 10.0 : 0.02  # cm⁻¹ → mm⁻¹
	ct.recon.huOffset = -1000
	return ct
end

# ╔═╡ b0000001-0006-0001-0005-000000000011
function catsim_configure_phantom!(ct, json_path; scale=1.0, offset=[0.0, 0.0, 0.0])
	ct.phantom.callback = "Phantom_Voxelized"
	ct.phantom.projectorCallback = "C_Projector_Voxelized"
	ct.phantom.filename = json_path
	ct.phantom.scale = scale
	ct.phantom.centerOffset = pylist(offset)
	return ct
end

# ╔═╡ b0000001-0006-0001-0005-000000000012
"""
    catsim_forward_project

Runs forward projection and reads the binary output directly using Julia I/O.
CatSim C-order on disk: [Views, Rows, Cols] → reshape to (Cols, Rows, Views).
"""
function catsim_forward_project(ct; results_name="catsim_out")
	ct.resultsName = results_name
	ct.run_all()

	rows = Int(pyconvert(Float64, ct.scanner.detectorRowCount))
	cols = Int(pyconvert(Float64, ct.scanner.detectorColCount))
	views = Int(pyconvert(Float64, ct.protocol.viewCount))

	scan_file = "$(results_name).prep"
	raw_bytes = read(scan_file)
	sino_flat = reinterpret(Float32, raw_bytes)

	return reshape(sino_flat, (cols, rows, views))
end

# ╔═╡ b0000001-0006-0001-0005-000000000013
"""
    catsim_reconstruct_fdk

Runs FDK reconstruction and reads binary output directly.
"""
function catsim_reconstruct_fdk(ct; results_name="catsim_out")
	_, recon_mod, _, _ = catsim_init()
	ct.resultsName = results_name
	ct.recon.filename = ct.resultsName
	ct.do_Recon = 1

	recon_mod.recon(ct)

	nx = Int(pyconvert(Float64, ct.recon.imageSize))
	ny = Int(pyconvert(Float64, ct.recon.imageSize))
	nz = Int(pyconvert(Float64, ct.recon.sliceCount))

	recon_file = "$(results_name)_$(nx)x$(ny)x$(nz).raw"
	if !isfile(recon_file)
		error("Recon file not found: $recon_file")
	end

	raw_bytes = read(recon_file)
	vol_flat = reinterpret(Float32, raw_bytes)

	return reshape(vol_flat, (nx, ny, nz))
end

# ╔═╡ b0000001-0006-0001-0005-000000000014
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

# ╔═╡ b0000001-0006-0001-0006-000000000001
md"""
---
## 5. Custom Phantom — Modified Gammex 472

**Geometry** (manufacturer spec): 33cm diameter × 5cm thick solid water body at **0.2mm** isotropic resolution.
- **Outer ring** (R = 10.5cm, 8 positions at 45°, gap centered at 12 o'clock):
  Ca100 → Ca200 → Ca300 → Ca400 → Water → SW ref → SW ref → Ca50
- **Inner ring** (R = 5.0cm, 8 positions at 45°, starts at 12 o'clock):
  I2.5 → I5.0 → I7.5 → I10 → I15 → I20 → Water → I2.0
- Insert diameter: 2.8cm

**Body material:** Gammex Model 451 Solid Water (ρ = 1.02 g/cm³).

**Labels:** 0 = air, 1 = solid water body, 2 = pure water, 3 = SW reference rod,
10–14 = Ca (50–400), 20–26 = I (2.0–20.0).

The phantom is **thicker** (5cm) than the collimation (4cm) so every detector row
traces rays through the full phantom body.
"""

# ╔═╡ b0000001-0006-0001-0006-000000000002
# Material display info for categorical phantom visualization
const MATERIAL_INFO = Dict(
	UInt8(0)  => (name="Air",              color=:gray15),
	UInt8(1)  => (name="Solid Water",      color=:lightskyblue),
	UInt8(2)  => (name="Pure Water",       color=:royalblue),
	UInt8(3)  => (name="SW Reference",     color=:paleturquoise),
	UInt8(10) => (name="Ca 50 mg/mL",      color=:wheat),
	UInt8(11) => (name="Ca 100 mg/mL",     color=:sandybrown),
	UInt8(12) => (name="Ca 200 mg/mL",     color=:orange),
	UInt8(13) => (name="Ca 300 mg/mL",     color=:darkorange),
	UInt8(14) => (name="Ca 400 mg/mL",     color=:orangered),
	UInt8(20) => (name="I 2.0 mg/mL",      color=:honeydew),
	UInt8(21) => (name="I 2.5 mg/mL",      color=:palegreen),
	UInt8(22) => (name="I 5.0 mg/mL",      color=:lightgreen),
	UInt8(23) => (name="I 7.5 mg/mL",      color=:mediumseagreen),
	UInt8(24) => (name="I 10.0 mg/mL",     color=:seagreen),
	UInt8(25) => (name="I 15.0 mg/mL",     color=:forestgreen),
	UInt8(26) => (name="I 20.0 mg/mL",     color=:darkgreen),
)

# ╔═╡ b0000001-0006-0001-0006-000000000012
# Full label → CatSim material name mapping (only active labels get exported)
# CatSim doesn't have a dedicated solid water material, so we map it to "water"
REGION_TO_CATSIM = Dict(
	1 => "water",   # solid water body (≈ water for CatSim)
	2 => "water",   # true water vials
	3 => "water",   # solid water reference rods (≈ water for CatSim)
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
	26 => "Gammex472_I_20_0",
)

# ╔═╡ b0000001-0006-0001-0006-000000000013
# Full label → XA.Material mapping (all 14 Gammex 472 inserts)
ALL_INSERT_MATERIALS = Dict{UInt8, XA.Material}(
	UInt8(10) => XA.Materials.gammex_472_ca50_0,
	UInt8(11) => XA.Materials.gammex_472_ca100_0,
	UInt8(12) => XA.Materials.gammex_472_ca200_0,
	UInt8(13) => XA.Materials.gammex_472_ca300_0,
	UInt8(14) => XA.Materials.gammex_472_ca400_0,
	UInt8(15) => XA.Materials.gammex_472_ca500_0,
	UInt8(16) => XA.Materials.gammex_472_ca600_0,
	UInt8(20) => XA.Materials.gammex_472_i2_0,
	UInt8(21) => XA.Materials.gammex_472_i2_5,
	UInt8(22) => XA.Materials.gammex_472_i5_0,
	UInt8(23) => XA.Materials.gammex_472_i7_5,
	UInt8(24) => XA.Materials.gammex_472_i10_0,
	UInt8(25) => XA.Materials.gammex_472_i15_0,
	UInt8(26) => XA.Materials.gammex_472_i20_0,
)

# ╔═╡ b0000001-0006-0001-0006-000000000005
# Gammex Model 451 Solid Water (from XrayAttenuation.jl >= 0.2.3)
const GAMMEX_SOLID_WATER = XA.Materials.gammex_472_solidwater

# ╔═╡ b0000001-0006-0001-0006-000000000003
"""
	create_custom_gammex_472(; n_voxels, n_slices, fov_cm, z_cm)

Build a Gammex 472 phantom at 0.2mm resolution matching the physical layout:
- **Body:** Gammex Model 451 Solid Water (label 1), 33cm diameter × 5cm thick
- **Outer ring** (R=10.5cm, gap at 12 o'clock, clockwise):
  Ca100 → Ca200 → Ca300 → Ca400 → Water → SW ref → SW ref → Ca50
- **Inner ring** (R=5.0cm, starts at 12 o'clock, clockwise):
  I2.5 → I5.0 → I7.5 → I10 → I15 → I20 → Water → I2.0
- Rod diameter: 28mm

The 2D slice is computed once and replicated across all z-slices.
Returns a `BS.Phantom` ready for simulation.
"""
function create_custom_gammex_472(;
	n_voxels::Int = 1750,   # 35cm / 0.02cm → 0.2mm isotropic
	n_slices::Int = 250,    # 5cm / 0.02cm → 0.2mm isotropic
	fov_cm::Float64 = 35.0,
	z_cm::Float64 = 5.0,
)
	dx = fov_cm / n_voxels
	dy = fov_cm / n_voxels
	dz = z_cm / n_slices

	x = range(-fov_cm/2 + dx/2, fov_cm/2 - dx/2, length=n_voxels)
	y = range(-fov_cm/2 + dy/2, fov_cm/2 - dy/2, length=n_voxels)

	# Gammex 472 geometry (cm)
	body_radius  = 16.5   # 330mm diameter
	rod_radius²  = 1.4^2  # 28mm diameter inserts (squared for fast distance check)
	outer_ring_R = 10.5   # Calcium ring (105mm from center)
	inner_ring_R = 5.0    # Iodine ring (50mm from center)

	# ── Outer ring: 8 rods, gap centered at 12 o'clock ──
	# First rod at π/2 - π/8 (≈ 1 o'clock), going clockwise
	outer_start = π/2 - π/8
	outer_angles = [outer_start - (i-1) * π/4 for i in 1:8]
	outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]
	# → Ca100, Ca200, Ca300, Ca400, Water, SW ref, SW ref, Ca50

	# ── Inner ring: 8 rods, first rod at 12 o'clock ──
	inner_start = π/2
	inner_angles = [inner_start - (i-1) * π/4 for i in 1:8]
	inner_labels = UInt8[21, 22, 23, 24, 25, 26, 2, 20]
	# → I2.5, I5.0, I7.5, I10, I15, I20, Water, I2.0

	# Pre-compute rod center positions (cm)
	outer_cx = [outer_ring_R * cos(a) for a in outer_angles]
	outer_cy = [outer_ring_R * sin(a) for a in outer_angles]
	inner_cx = [inner_ring_R * cos(a) for a in inner_angles]
	inner_cy = [inner_ring_R * sin(a) for a in inner_angles]

	# Build 2D slice (all z-slices are identical for this phantom)
	slice = zeros(UInt8, n_voxels, n_voxels)

	for j in 1:n_voxels, i in 1:n_voxels
		xi, yj = x[i], y[j]

		if xi^2 + yj^2 <= body_radius^2
			slice[i, j] = UInt8(1)  # solid water body

			# Check outer ring (Calcium + water + SW rods)
			for idx in 1:8
				if (xi - outer_cx[idx])^2 + (yj - outer_cy[idx])^2 <= rod_radius²
					slice[i, j] = outer_labels[idx]
					@goto next_voxel
				end
			end

			# Check inner ring (Iodine + water rods)
			for idx in 1:8
				if (xi - inner_cx[idx])^2 + (yj - inner_cy[idx])^2 <= rod_radius²
					slice[i, j] = inner_labels[idx]
					break
				end
			end
		end
		@label next_voxel
	end

	# Replicate 2D slice across all z-slices
	mask = Array{UInt8, 3}(undef, n_voxels, n_voxels, n_slices)
	@views for k in 1:n_slices
		mask[:, :, k] .= slice
	end

	# Build materials vector (index = label + 1, Julia 1-based)
	max_label = 26
	materials_vec = Vector{XA.Material}(undef, max_label + 1)
	fill!(materials_vec, XA.Materials.air)
	materials_vec[1] = XA.Materials.air            # label 0 (background)
	materials_vec[2] = GAMMEX_SOLID_WATER          # label 1 (solid water body)
	materials_vec[3] = XA.Materials.water           # label 2 (pure water)
	materials_vec[4] = GAMMEX_SOLID_WATER          # label 3 (SW reference rods)

	for (lbl, mat) in ALL_INSERT_MATERIALS
		materials_vec[Int(lbl) + 1] = mat
	end

	origin = (-fov_cm/2 + dx/2, -fov_cm/2 + dy/2, -z_cm/2 + dz/2)
	extent = (Float64(fov_cm), Float64(fov_cm), Float64(z_cm))

	return BS.Phantom(mask, materials_vec, (dx, dy, dz), origin, extent)
end

# ╔═╡ b0000001-0006-0001-0006-000000000004
"""
	export_phantom_for_catsim(phantom, output_dir, basename)

Export a `BS.Phantom` mask to CatSim voxelized JSON format.
Voxel sizes are converted from cm → mm. Only labels present in
`REGION_TO_CATSIM` are exported.
"""
function export_phantom_for_catsim(phantom, output_dir, basename)
	mask_cpu = phantom.mask isa Array ? phantom.mask : Array(phantom.mask)
	nx, ny, nz = size(mask_cpu)
	vx, vy, vz = phantom.voxel_size .* 10.0  # cm → mm

	mkpath(output_dir)
	unique_labels = sort(unique(mask_cpu))
	filter!(l -> l != 0, unique_labels)

	json_materials, json_filenames, json_datatypes = String[], String[], String[]
	json_cols, json_rows, json_slices = Int[], Int[], Int[]
	json_xsize, json_ysize, json_zsize = Float64[], Float64[], Float64[]
	json_xoffset, json_yoffset, json_zoffset, json_densscale = Float64[], Float64[], Float64[], Float64[]

	for lbl in unique_labels
		lbl_int = Int(lbl)
		if !haskey(REGION_TO_CATSIM, lbl_int) continue end
		mat_name = REGION_TO_CATSIM[lbl_int]

		density_map = Float32.(mask_cpu .== lbl)
		fname = "$(basename)_mat$(lbl_int).density"
		write(joinpath(output_dir, fname), density_map)

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

# ╔═╡ b0000001-0006-0001-0007-000000000001
md"""
---
## 6. Phantoms & Theoretical Water Calibration

**Theoretical μ\_water** (primary): spectrum-weighted μ from NIST XCOM via XrayAttenuation.jl.
Empirical water calibration cells are in Sections 7 and 8, paired with their corresponding scans.
"""

# ╔═╡ b0000001-0006-0001-0007-000000000002
# Spectrum-weighted μ_water per kVp (theoretical, no simulation needed)
μ_water_per_kvp = let
	result = Dict{Int, Float64}()
	for kvp in sort(unique([sc.kvp for sc in SCANS]))
		e, w = BS.load_spectrum(kvp)
		result[kvp] = BS.compute_effective_μ_material(XA.Materials.water, e, w)
	end
	result
end

# ╔═╡ b0000001-0006-0001-0007-000000000003
# Water phantom (same body geometry as Gammex 472, uniform water)
phantom_water = let
	nx, ny, nz = recon_xy, recon_xy, n_recon_slices
	voxel_cm = recon_fov_cm / nx
	voxel_z_cm = phantom_z_cm / nz

	water_mask = zeros(UInt8, nx, ny, nz)
	radius_cm = 16.5  # Gammex body radius
	xs = range(-recon_fov_cm/2 + voxel_cm/2, recon_fov_cm/2 - voxel_cm/2, length=nx)
	ys = range(-recon_fov_cm/2 + voxel_cm/2, recon_fov_cm/2 - voxel_cm/2, length=ny)
	for k in 1:nz, j in 1:ny, i in 1:nx
		if sqrt(xs[i]^2 + ys[j]^2) <= radius_cm
			water_mask[i, j, k] = UInt8(1)
		end
	end

	# Materials vector: index 1 = label 0 (air), index 2 = label 1 (water)
	water_materials = [XA.Materials.air, XA.Materials.water]

	origin = (-recon_fov_cm/2 + voxel_cm/2, -recon_fov_cm/2 + voxel_cm/2, -phantom_z_cm/2 + voxel_z_cm/2)
	extent = (Float64(recon_fov_cm), Float64(recon_fov_cm), Float64(phantom_z_cm))

	BS.Phantom(water_mask, water_materials, (voxel_cm, voxel_cm, voxel_z_cm), origin, extent)
end;

# ╔═╡ b0000001-0006-0001-0007-000000000004
# Custom Gammex 472 at 0.2mm isotropic resolution
phantom_gammex = create_custom_gammex_472(
	n_voxels = 1750,      # 35cm / 0.02cm → 0.2mm xy
	n_slices = 250,       # 5cm / 0.02cm → 0.2mm z
	fov_cm = recon_fov_cm,
	z_cm = phantom_z_cm,  # 5cm — actual Gammex 472 thickness (> 4cm collimation)
);

# ╔═╡ b0000001-0006-0001-0007-000000000010
@bind phantom_slice UI.Slider(1:size(phantom_gammex.mask, 3), default=size(phantom_gammex.mask, 3) ÷ 2, show_value=true)

# ╔═╡ b0000001-0006-0001-0007-000000000009
# Categorical visualization of phantom materials
let
	slice_data = phantom_gammex.mask[:, :, phantom_slice]
	nz = size(phantom_gammex.mask, 3)

	# Map labels → sequential indices for categorical colormap
	unique_labels = sort(unique(slice_data))
	n_labels = length(unique_labels)

	lut = zeros(Float32, 27)   # max label = 26 → index 27
	for (i, l) in enumerate(unique_labels)
		lut[Int(l) + 1] = Float32(i)
	end
	mapped = lut[Int.(slice_data) .+ 1]

	# Categorical colormap from MATERIAL_INFO
	colors = [MATERIAL_INFO[l].color for l in unique_labels]
	cmap = CM.cgrad(colors, n_labels, categorical=true)
	names = [MATERIAL_INFO[l].name for l in unique_labels]

	fig = CM.Figure(size=(1000, 850), fontsize=12)

	ax = CM.Axis(fig[1, 1],
		title="Gammex 472 Phantom — Slice $phantom_slice / $nz  (0.2mm voxels)",
		aspect=CM.DataAspect(),
	)
	hm = CM.heatmap!(ax, mapped,
		colormap=cmap,
		colorrange=(0.5, n_labels + 0.5),
	)

	CM.Colorbar(fig[1, 2], hm,
		ticks=(1:n_labels, names),
		ticklabelsize=11,
		width=15,
	)

	fig
end

# ╔═╡ b0000001-0006-0001-0007-000000000005
# GPU-backed phantoms
phantom_gammex_gpu = BS.Phantom(
	Metal.MtlArray(phantom_gammex.mask),
	phantom_gammex.materials,
	phantom_gammex.voxel_size,
	phantom_gammex.origin,
	phantom_gammex.extent,
);

# ╔═╡ b0000001-0006-0001-0007-000000000008
phantom_water_gpu = BS.Phantom(
	Metal.MtlArray(phantom_water.mask),
	phantom_water.materials,
	phantom_water.voxel_size,
	phantom_water.origin,
	phantom_water.extent,
);

# ╔═╡ b0000001-0006-0001-0008-000000000001
md"""
---
## 7. BasisSimulator — Water Calibration & Scan Execution

Each water calibration cell is paired with its corresponding scan(s).
All cells use `let ... end` blocks to scope GPU memory and `Array(...)` to finalize results on CPU.

- **120 kVp water cal** → Scans 1, 2, 3 (dose variation)
- **80 kVp water cal** → Scan 4
- **100 kVp water cal** → Scan 5
- **140 kVp water cal** → Scan 6
"""

# ╔═╡ b0000001-0006-0001-0007-000000000006
# ╠═╡ show_logs = false
# BasisSim water calibration — 120 kVp
basis_water_120kvp = let
	prot = protocols["120kVp_10mGy"]
	recon_size = recon_opts.matrix_size

	@info "BasisSim water cal: 120 kVp..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_water_gpu)
	BS.simulate!(ws, phantom_water_gpu, scanner, prot, sim_opts, recon_opts)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

	cx, cy, cz = size(vol) .÷ 2
	z_half = min(cz - 1, 4)
	μ_empirical = mean(vol[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(vol,3),cz+z_half)])

	ws_fdk = nothing; ws = nothing; vol = nothing; GC.gc(true)
	μ_empirical
end

# ╔═╡ b0000001-0006-0001-0008-000000000003
# ╠═╡ show_logs = false
# Scan 1: 120 kVp / 3 mGy
basis_scan_1 = let
	sc = SCANS[1]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	recon_size = recon_opts.matrix_size

	@info "BasisSim: $(sc.label)..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

	sino_cpu = Array(ws.sino_noisy_out)
	geom_copy = ws.geom

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	recon_hu = Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(sinogram=sino_cpu, recon=recon_hu, mu_water=μ_w, geom=geom_copy)
end

# ╔═╡ ba7ca05f-a73d-4c7b-98bc-ab6aa06361e5
@bind slice1 UI.Slider(axes(basis_scan_1.recon, 3); show_value = true, default = 1)

# ╔═╡ 2c9082ae-2305-4ecd-bc8e-9f5115a3946b
let
	slice_data = phantom_gammex.mask[:, :, 64]
	
	f = CM.Figure()
	ax = CM.Axis(f[1, 1]; aspect = CM.DataAspect(), title = "CT Simulation")
	CM.heatmap!(basis_scan_1.recon[:, :, slice1]; colormap = :grays)

	ax = CM.Axis(f[1, 2]; aspect = CM.DataAspect(), title = "Ground Truth")
	CM.heatmap!(slice_data; colormap = :grays)
	f
end

# ╔═╡ b0000001-0006-0001-0008-000000000020
md"""
### 7a. Automatic Rod Segmentation

Geometry-driven segmentation of all 16 Gammex 472 insert rods from the CT reconstruction.

**Algorithm:**
1. **Center detection** — thresholded centroid with iterative circular refinement (robust to table/bed)
2. **Rotation detection** — angular HU profile at outer ring radius → find Ca400 peak (highest HU)
3. **ROI placement** — circular ROIs at known (r, θ) positions, rotated to match phantom orientation
4. **Measurement** — mean/std HU within each conservative ROI (70% of rod diameter)

Produces a **2D labeled mask** that can be propagated across z-slices (phantom is z-symmetric).
"""

# ╔═╡ b0000001-0006-0001-0008-000000000021
"""
	segment_gammex_rods(hu_slice; fov_cm=35.0, ...) → (mask, rod_info, center_info)

Segment all 16 Gammex 472 insert rods from a 2D CT reconstruction slice.

Uses known phantom geometry + intensity-based rotation detection.
Works on simulation and clinical scans at any matrix size.

**Returns:**
- `mask`: UInt8 2D array with rod labels (phantom scheme: 10-14=Ca, 20-26=I, 2=water, 3=SW)
- `rod_info`: Vector of 16 NamedTuples with per-rod measurements
- `center_info`: NamedTuple `(cx, cy, rotation_deg)`
"""
function segment_gammex_rods(
		hu_slice;
		fov_cm = 35.0,
		body_threshold_hu = -400.0,
		body_radius_cm = 16.5,
		outer_ring_cm = 10.5,
		inner_ring_cm = 5.0,
		rod_radius_cm = 1.4,
		roi_fraction = 0.7,
	)
	
	nx, ny = size(hu_slice)
	pixel_cm = fov_cm / nx

	# ── Step 1: Find phantom center (table-robust) ──
	body = hu_slice .> body_threshold_hu
	total = max(Float64(sum(body)), 1.0)
	cx = sum(Float64(i) * body[i,j] for j in 1:ny, i in 1:nx) / total
	cy = sum(Float64(j) * body[i,j] for j in 1:ny, i in 1:nx) / total

	# Iterative refinement: restrict centroid to expected body circle
	# (eliminates table/bed bias on clinical scans)
	body_r² = (body_radius_cm / pixel_cm)^2
	for _ in 1:3
		sx, sy, cnt = 0.0, 0.0, 0.0
		for j in 1:ny, i in 1:nx
			if (i - cx)^2 + (j - cy)^2 <= body_r² && body[i,j]
				sx += i; sy += j; cnt += 1.0
			end
		end
		if cnt > 0
			cx = sx / cnt; cy = sy / cnt
		end
	end

	# ── Step 2: Detect rotation via outer ring angular profile ──
	r_outer_pix = outer_ring_cm / pixel_cm
	n_sample = 720
	sample_angles = range(0, 2π - 2π/n_sample, length=n_sample)

	# Sample HU around the outer ring with radial averaging (±3 pixels)
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

	# Smooth with ±5° circular window for noise robustness
	smooth_w = max(1, round(Int, 5.0 / (360.0 / n_sample)))
	smoothed = similar(profile)
	for k in 1:n_sample
		s, c = 0.0, 0
		for d in -smooth_w:smooth_w
			s += profile[mod1(k + d, n_sample)]; c += 1
		end
		smoothed[k] = s / c
	end

	# Ca400 is the highest HU rod on the outer ring → find its angle
	θ_ca400_measured = sample_angles[argmax(smoothed)]
	θ_ca400_expected = π/2 - π/8 - 3 * π/4   # = -3π/8 (unrotated)
	rotation = θ_ca400_measured - θ_ca400_expected

	# ── Step 3: Place ROIs at all 16 rod positions ──
	outer_start = π/2 - π/8 + rotation
	outer_angles = [outer_start - (i-1) * π/4 for i in 1:8]
	outer_labels = UInt8[11, 12, 13, 14, 2, 3, 3, 10]
	outer_names = ["Ca 100", "Ca 200", "Ca 300", "Ca 400",
	               "Water (O)", "SW ref 1", "SW ref 2", "Ca 50"]

	inner_start = π/2 + rotation
	inner_angles = [inner_start - (i-1) * π/4 for i in 1:8]
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

# ╔═╡ b0000001-0006-0001-0008-000000000022
# Segment rods from central slice of Scan 1
seg_result = let
	mid_z = size(basis_scan_1.recon, 3) ÷ 2
	hu_slice = basis_scan_1.recon[:, :, mid_z]
	mask, rod_info, center = segment_gammex_rods(hu_slice; fov_cm=recon_fov_cm)
	@info "Phantom center: ($(round(center.cx,digits=1)), $(round(center.cy,digits=1))), rotation: $(center.rotation_deg)°"
	(mask=mask, rods=rod_info, center=center, slice_idx=mid_z)
end

# ╔═╡ b0000001-0006-0001-0008-000000000023
# Overlay: CT image + segmented ROIs with material labels
let
	hu = basis_scan_1.recon[:, :, seg_result.slice_idx]
	rods = seg_result.rods
	pixel_cm = recon_fov_cm / size(hu, 1)
	roi_r_pix = 1.4 * 0.7 / pixel_cm  # same as segmentation

	fig = CM.Figure(size=(1100, 500), fontsize=11)

	# Left: CT with ROI circles
	ax1 = CM.Axis(fig[1, 1], title="Scan 1 — Segmented ROIs (slice $(seg_result.slice_idx))",
		aspect=CM.DataAspect())
	CM.heatmap!(ax1, hu, colormap=:grays, colorrange=(-200, 500))

	θ_circle = range(0, 2π, length=61)
	for r in rods
		xs = r.cx .+ roi_r_pix .* cos.(θ_circle)
		ys = r.cy .+ roi_r_pix .* sin.(θ_circle)
		c = r.ring == :outer ? :orange : :lime
		CM.lines!(ax1, xs, ys, color=c, linewidth=1.5)
		CM.text!(ax1, r.cx, r.cy + roi_r_pix + 4;
			text=r.name, fontsize=7, align=(:center, :bottom), color=c)
	end

	# Mark detected center
	CM.scatter!(ax1, [seg_result.center.cx], [seg_result.center.cy],
		color=:red, marker=:cross, markersize=12)

	# Right: labeled mask
	ax2 = CM.Axis(fig[1, 2], title="Rod Label Mask", aspect=CM.DataAspect())
	mask_vis = Float32.(seg_result.mask)
	mask_vis[mask_vis .== 0] .= NaN
	CM.heatmap!(ax2, hu, colormap=:grays, colorrange=(-200, 500))
	CM.heatmap!(ax2, mask_vis, colormap=:turbo, colorrange=(1, 27), nan_color=:transparent)

	fig
end

# ╔═╡ b0000001-0006-0001-0008-000000000024
# ROI measurement summary: bar chart by material
let
	rods = seg_result.rods
	n = length(rods)

	fig = CM.Figure(size=(1000, 500), fontsize=11)

	names = [r.name for r in rods]
	means = [r.mean_hu for r in rods]
	stds = [r.std_hu for r in rods]

	colors = map(rods) do r
		r.ring == :outer ?
			(startswith(r.name, "Ca") ? :darkorange : :steelblue) :
			(startswith(r.name, "I") ? :forestgreen : :steelblue)
	end

	ax = CM.Axis(fig[1, 1],
		title="ROI Measurements — $(SCANS[1].label)  [rotation=$(seg_result.center.rotation_deg)°]",
		ylabel="Mean HU ± σ",
		xticks=(1:n, names),
		xticklabelrotation=π/4,
	)
	CM.barplot!(ax, 1:n, means, color=colors)
	CM.errorbars!(ax, 1:n, means, stds, color=:black, whiskerwidth=4)

	# Annotate HU values above bars
	for (i, (m, s)) in enumerate(zip(means, stds))
		CM.text!(ax, i, m + s + 5;
			text="$(round(Int, m))",
			fontsize=8, align=(:center, :bottom))
	end

	fig
end

# ╔═╡ b0000001-0006-0001-0008-000000000025
"""
	save_scan_reference(base_dir, scan_name, hu_volume, rod_info, center_info; kwargs...)

Save reference images and rod measurements for a single scan protocol.

Creates `base_dir/scan_name/` with:
- `recon.png` — middle-slice reconstruction (grayscale, windowed)
- `rods.png` — same slice with segmented ROI overlay + material labels
- `rod_measurements.csv` — per-rod name, label, ring, mean_hu, std_hu, n_pixels

**Usage for building reference library:**
```julia
# CatSim reference (fixed ground truth — single top-level folder)
save_scan_reference("test/references/catsim",
    "120kVp_10mGy", catsim_recon, rods, center)

# Clinical reference (fixed ground truth — one folder per scanner)
save_scan_reference("test/references/ge_revolution_apex",
    "120kVp_10mGy", dicom_recon, rods, center)
```
"""
function save_scan_reference(base_dir, scan_name, hu_volume, rod_info, center_info;
	hu_window = (-200, 500),
	slice_idx = nothing,
	fov_cm = 35.0,
	roi_fraction = 0.7,
	rod_radius_cm = 1.4,
)
	mid_z = something(slice_idx, size(hu_volume, 3) ÷ 2)
	hu_slice = hu_volume[:, :, mid_z]
	nx = size(hu_slice, 1)
	pixel_cm = fov_cm / nx
	roi_r_pix = rod_radius_cm * roi_fraction / pixel_cm

	dir = joinpath(base_dir, scan_name)
	mkpath(dir)

	# ── 1. Reconstruction PNG (clean, no axes) ──
	fig1 = CM.Figure(size=(512, 512), figure_padding=0)
	ax1 = CM.Axis(fig1[1,1], aspect=CM.DataAspect())
	CM.hidedecorations!(ax1); CM.hidespines!(ax1)
	CM.heatmap!(ax1, hu_slice, colormap=:grays, colorrange=hu_window)
	CM.save(joinpath(dir, "recon.png"), fig1, px_per_unit=1)

	# ── 2. Rod overlay PNG ──
	fig2 = CM.Figure(size=(700, 600), fontsize=10)
	ax2 = CM.Axis(fig2[1,1], title="$scan_name — slice $mid_z", aspect=CM.DataAspect())
	CM.heatmap!(ax2, hu_slice, colormap=:grays, colorrange=hu_window)

	θ_circle = range(0, 2π, length=61)
	for r in rod_info
		xs = r.cx .+ roi_r_pix .* cos.(θ_circle)
		ys = r.cy .+ roi_r_pix .* sin.(θ_circle)
		c = r.ring == :outer ? :orange : :lime
		CM.lines!(ax2, xs, ys, color=c, linewidth=1.5)
		CM.text!(ax2, r.cx, r.cy + roi_r_pix + 3;
			text=r.name, fontsize=7, align=(:center, :bottom), color=c)
	end
	CM.scatter!(ax2, [center_info.cx], [center_info.cy],
		color=:red, marker=:cross, markersize=10)
	CM.save(joinpath(dir, "rods.png"), fig2, px_per_unit=2)

	# ── 3. Rod measurements CSV ──
	csv_path = joinpath(dir, "rod_measurements.csv")
	open(csv_path, "w") do f
		println(f, "name,label,ring,mean_hu,std_hu,n_pixels,cx,cy,angle_deg")
		for r in rod_info
			println(f, join([
				r.name, Int(r.label), r.ring,
				round(r.mean_hu, digits=2), round(r.std_hu, digits=2), r.n_pixels,
				round(r.cx, digits=1), round(r.cy, digits=1), r.angle_deg
			], ","))
		end
	end

	# ── 4. Metadata ──
	open(joinpath(dir, "metadata.txt"), "w") do f
		println(f, "center_x=", round(center_info.cx, digits=2))
		println(f, "center_y=", round(center_info.cy, digits=2))
		println(f, "rotation_deg=", center_info.rotation_deg)
		println(f, "slice_idx=", mid_z)
		println(f, "matrix_size=", nx)
		println(f, "fov_cm=", fov_cm)
		println(f, "hu_window=", hu_window)
	end

	@info "Saved reference → $dir  ($(length(rod_info)) rods)"
	return dir
end

# ╔═╡ b0000001-0006-0001-0008-000000000026
"""
	load_rod_reference(csv_path) → Vector{NamedTuple}

Load saved rod measurements CSV back for comparison against BasisSimulator results.
"""
function load_rod_reference(csv_path)
	lines = readlines(csv_path)
	header = split(lines[1], ",")
	rods = map(lines[2:end]) do line
		vals = split(line, ",")
		(
			name = String(vals[1]),
			label = parse(UInt8, vals[2]),
			ring = Symbol(vals[3]),
			mean_hu = parse(Float64, vals[4]),
			std_hu = parse(Float64, vals[5]),
			n_pixels = parse(Int, vals[6]),
		)
	end
	return rods
end

# ╔═╡ b0000001-0006-0001-0008-000000000027
const REFERENCES_DIR = joinpath(dirname(dirname(@__DIR__)), "test", "references")

# ╔═╡ b0000001-0006-0001-0008-000000000004
# ╠═╡ show_logs = false
# Scan 2: 120 kVp / 10 mGy
basis_scan_2 = let
	sc = SCANS[2]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	recon_size = recon_opts.matrix_size

	@info "BasisSim: $(sc.label)..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

	sino_cpu = Array(ws.sino_noisy_out)
	geom_copy = ws.geom

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	recon_hu = Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(sinogram=sino_cpu, recon=recon_hu, mu_water=μ_w, geom=geom_copy)
end

# ╔═╡ b0000001-0006-0001-0008-000000000005
# ╠═╡ show_logs = false
# Scan 3: 120 kVp / 20 mGy
basis_scan_3 = let
	sc = SCANS[3]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	recon_size = recon_opts.matrix_size

	@info "BasisSim: $(sc.label)..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

	sino_cpu = Array(ws.sino_noisy_out)
	geom_copy = ws.geom

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	recon_hu = Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(sinogram=sino_cpu, recon=recon_hu, mu_water=μ_w, geom=geom_copy)
end

# ╔═╡ b0000001-0006-0001-0008-000000000002
# ╠═╡ show_logs = false
# BasisSim water calibration — 80 kVp
basis_water_80kvp = let
	prot = protocols["80kVp_10mGy"]
	recon_size = recon_opts.matrix_size

	@info "BasisSim water cal: 80 kVp..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_water_gpu)
	BS.simulate!(ws, phantom_water_gpu, scanner, prot, sim_opts, recon_opts)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

	cx, cy, cz = size(vol) .÷ 2
	z_half = min(cz - 1, 4)
	μ_empirical = mean(vol[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(vol,3),cz+z_half)])

	ws_fdk = nothing; ws = nothing; vol = nothing; GC.gc(true)
	μ_empirical
end

# ╔═╡ b0000001-0006-0001-0008-000000000006
# ╠═╡ show_logs = false
# Scan 4: 80 kVp / 10 mGy
basis_scan_4 = let
	sc = SCANS[4]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	recon_size = recon_opts.matrix_size

	@info "BasisSim: $(sc.label)..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

	sino_cpu = Array(ws.sino_noisy_out)
	geom_copy = ws.geom

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	recon_hu = Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(sinogram=sino_cpu, recon=recon_hu, mu_water=μ_w, geom=geom_copy)
end

# ╔═╡ b0000001-0006-0001-0008-000000000010
# ╠═╡ show_logs = false
# BasisSim water calibration — 100 kVp
basis_water_100kvp = let
	prot = protocols["100kVp_10mGy"]
	recon_size = recon_opts.matrix_size

	@info "BasisSim water cal: 100 kVp..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_water_gpu)
	BS.simulate!(ws, phantom_water_gpu, scanner, prot, sim_opts, recon_opts)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

	cx, cy, cz = size(vol) .÷ 2
	z_half = min(cz - 1, 4)
	μ_empirical = mean(vol[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(vol,3),cz+z_half)])

	ws_fdk = nothing; ws = nothing; vol = nothing; GC.gc(true)
	μ_empirical
end

# ╔═╡ b0000001-0006-0001-0008-000000000007
# ╠═╡ show_logs = false
# Scan 5: 100 kVp / 10 mGy
basis_scan_5 = let
	sc = SCANS[5]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	recon_size = recon_opts.matrix_size

	@info "BasisSim: $(sc.label)..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

	sino_cpu = Array(ws.sino_noisy_out)
	geom_copy = ws.geom

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	recon_hu = Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(sinogram=sino_cpu, recon=recon_hu, mu_water=μ_w, geom=geom_copy)
end

# ╔═╡ b0000001-0006-0001-0008-000000000011
# ╠═╡ show_logs = false
# BasisSim water calibration — 140 kVp
basis_water_140kvp = let
	prot = protocols["140kVp_10mGy"]
	recon_size = recon_opts.matrix_size

	@info "BasisSim water cal: 140 kVp..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_water_gpu)
	BS.simulate!(ws, phantom_water_gpu, scanner, prot, sim_opts, recon_opts)

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	vol = Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size))

	cx, cy, cz = size(vol) .÷ 2
	z_half = min(cz - 1, 4)
	μ_empirical = mean(vol[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(vol,3),cz+z_half)])

	ws_fdk = nothing; ws = nothing; vol = nothing; GC.gc(true)
	μ_empirical
end

# ╔═╡ b0000001-0006-0001-0008-000000000008
# ╠═╡ show_logs = false
# Scan 6: 140 kVp / 10 mGy
basis_scan_6 = let
	sc = SCANS[6]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	recon_size = recon_opts.matrix_size

	@info "BasisSim: $(sc.label)..."
	ws = BS.create_eict_workspace(scanner, prot, sim_opts, recon_opts, phantom_gammex_gpu)
	@time BS.simulate!(ws, phantom_gammex_gpu, scanner, prot, sim_opts, recon_opts)

	sino_cpu = Array(ws.sino_noisy_out)
	geom_copy = ws.geom

	ws_fdk = BS.create_fdk_recon_workspace(ws.sino_noisy_out, ws.geom, recon_size; filter=recon_opts.filter)
	recon_hu = Array(BS.to_hounsfield(
		Array(BS.reconstruct!(ws_fdk, ws.sino_noisy_out, ws.geom, recon_size));
		μ_water=μ_w
	))

	ws_fdk = nothing; ws = nothing; GC.gc(true)
	(sinogram=sino_cpu, recon=recon_hu, mu_water=μ_w, geom=geom_copy)
end

# ╔═╡ b0000001-0006-0001-0008-000000000009
# Collect all BasisSimulator results (enable after running desired scans)
basis_results = Dict(
	SCANS[1].name => basis_scan_1,
	SCANS[2].name => basis_scan_2,
	SCANS[3].name => basis_scan_3,
	SCANS[4].name => basis_scan_4,
	SCANS[5].name => basis_scan_5,
	SCANS[6].name => basis_scan_6,
)

# ╔═╡ b0000001-0006-0001-0009-000000000001
md"""
---
## 8. CatSim — Water Calibration & Scan Execution

Same 6 protocols, same phantom, configured entirely from Julia structs.
Enable the phantom export cell first, then water cal + scan cells.

- **120 kVp water cal** → Scans 1, 2, 3 (dose variation)
- **80 kVp water cal** → Scan 4
- **100 kVp water cal** → Scan 5
- **140 kVp water cal** → Scan 6
"""

# ╔═╡ b0000001-0006-0001-0009-000000000002
# Export phantom to CatSim voxelized JSON (must run before any CatSim scan)
catsim_phantom_json = let
	work_dir = joinpath(@__DIR__, "catsim_work")
	export_phantom_for_catsim(phantom_gammex, work_dir, "phantom_gammex")
end

# ╔═╡ b0000001-0006-0001-0007-000000000007
# ╠═╡ show_logs = false
# CatSim water calibration — 120 kVp
catsim_water_120kvp = let
	work_dir = joinpath(@__DIR__, "catsim_work")
	water_json = export_phantom_for_catsim(phantom_water, work_dir, "water_cal")
	prot = protocols["120kVp_10mGy"]
	μ_w = μ_water_per_kvp[120]
	tag = "watercal_120kvp"

	@info "CatSim water cal: 120 kVp..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, water_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)

	cx, cy, cz = size(recon) .÷ 2
	z_half = min(cz - 1, 4)
	mean_hu = mean(recon[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(recon,3),cz+z_half)])
	@info "  120 kVp → water mean HU = $(round(mean_hu, digits=2))"
	mean_hu
end

# ╔═╡ b0000001-0006-0001-0009-000000000004
# ╠═╡ show_logs = false
# Scan 1: 120 kVp / 3 mGy
catsim_scan_1 = let
	sc = SCANS[1]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	tag = "sim_$(sc.name)"

	@info "CatSim: $(sc.label)..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, catsim_phantom_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)
	(sinogram=sino, reconstruction=recon)
end

# ╔═╡ b0000001-0006-0001-0009-000000000005
# ╠═╡ show_logs = false
#=╠═╡
# Scan 2: 120 kVp / 10 mGy
catsim_scan_2 = let
	sc = SCANS[2]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	tag = "sim_$(sc.name)"

	@info "CatSim: $(sc.label)..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, catsim_phantom_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)
	(sinogram=sino, reconstruction=recon)
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000006
# ╠═╡ show_logs = false
#=╠═╡
# Scan 3: 120 kVp / 20 mGy
catsim_scan_3 = let
	sc = SCANS[3]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	tag = "sim_$(sc.name)"

	@info "CatSim: $(sc.label)..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, catsim_phantom_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)
	(sinogram=sino, reconstruction=recon)
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000003
# ╠═╡ show_logs = false
#=╠═╡
# CatSim water calibration — 80 kVp
catsim_water_80kvp = let
	work_dir = joinpath(@__DIR__, "catsim_work")
	water_json = export_phantom_for_catsim(phantom_water, work_dir, "water_cal")
	prot = protocols["80kVp_10mGy"]
	μ_w = μ_water_per_kvp[80]
	tag = "watercal_80kvp"

	@info "CatSim water cal: 80 kVp..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, water_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)

	cx, cy, cz = size(recon) .÷ 2
	z_half = min(cz - 1, 4)
	mean_hu = mean(recon[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(recon,3),cz+z_half)])
	@info "  80 kVp → water mean HU = $(round(mean_hu, digits=2))"
	mean_hu
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000007
# ╠═╡ show_logs = false
#=╠═╡
# Scan 4: 80 kVp / 10 mGy
catsim_scan_4 = let
	sc = SCANS[4]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	tag = "sim_$(sc.name)"

	@info "CatSim: $(sc.label)..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, catsim_phantom_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)
	(sinogram=sino, reconstruction=recon)
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000011
# ╠═╡ show_logs = false
#=╠═╡
# CatSim water calibration — 100 kVp
catsim_water_100kvp = let
	work_dir = joinpath(@__DIR__, "catsim_work")
	water_json = export_phantom_for_catsim(phantom_water, work_dir, "water_cal")
	prot = protocols["100kVp_10mGy"]
	μ_w = μ_water_per_kvp[100]
	tag = "watercal_100kvp"

	@info "CatSim water cal: 100 kVp..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, water_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)

	cx, cy, cz = size(recon) .÷ 2
	z_half = min(cz - 1, 4)
	mean_hu = mean(recon[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(recon,3),cz+z_half)])
	@info "  100 kVp → water mean HU = $(round(mean_hu, digits=2))"
	mean_hu
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000008
# ╠═╡ show_logs = false
#=╠═╡
# Scan 5: 100 kVp / 10 mGy
catsim_scan_5 = let
	sc = SCANS[5]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	tag = "sim_$(sc.name)"

	@info "CatSim: $(sc.label)..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, catsim_phantom_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)
	(sinogram=sino, reconstruction=recon)
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000012
# ╠═╡ show_logs = false
#=╠═╡
# CatSim water calibration — 140 kVp
catsim_water_140kvp = let
	work_dir = joinpath(@__DIR__, "catsim_work")
	water_json = export_phantom_for_catsim(phantom_water, work_dir, "water_cal")
	prot = protocols["140kVp_10mGy"]
	μ_w = μ_water_per_kvp[140]
	tag = "watercal_140kvp"

	@info "CatSim water cal: 140 kVp..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, water_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)

	cx, cy, cz = size(recon) .÷ 2
	z_half = min(cz - 1, 4)
	mean_hu = mean(recon[cx-10:cx+10, cy-10:cy+10, max(1,cz-z_half):min(size(recon,3),cz+z_half)])
	@info "  140 kVp → water mean HU = $(round(mean_hu, digits=2))"
	mean_hu
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000009
# ╠═╡ show_logs = false
#=╠═╡
# Scan 6: 140 kVp / 10 mGy
catsim_scan_6 = let
	sc = SCANS[6]
	prot = protocols[sc.name]
	μ_w = μ_water_per_kvp[sc.kvp]
	tag = "sim_$(sc.name)"

	@info "CatSim: $(sc.label)..."
	catsim_cleanup(tag)
	ct = catsim_create_simulation()
	catsim_configure_phantom!(ct, catsim_phantom_json)
	catsim_configure_scanner!(ct, scanner, prot)
	catsim_configure_protocol!(ct, prot)
	catsim_configure_recon!(ct, recon_opts; μ_water_cm=μ_w)

	sino = catsim_forward_project(ct, results_name=tag)
	recon = catsim_reconstruct_fdk(ct, results_name=tag)
	catsim_cleanup(tag)
	(sinogram=sino, reconstruction=recon)
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0009-000000000010
#=╠═╡
# Collect all CatSim results (enable after running desired scans)
catsim_results = Dict(
	SCANS[1].name => catsim_scan_1,
	SCANS[2].name => catsim_scan_2,
	SCANS[3].name => catsim_scan_3,
	SCANS[4].name => catsim_scan_4,
	SCANS[5].name => catsim_scan_5,
	SCANS[6].name => catsim_scan_6,
)
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0016-000000000001
md"""
---
## 9. Reference Library Auto-Save

Saves CatSim ground truth references for all 6 protocols to `test/references/catsim/`.
Enable after running all CatSim scans. These are the **fixed baselines** that
BasisSimulator gets compared against — they never change when tuning BasisSim.

**Output per protocol** (`test/references/catsim/{protocol}/`):
- `recon.png` — CatSim FDK reconstruction (middle slice, grayscale)
- `mask.png` — Rod segmentation overlay
- `rod_measurements.csv` — Per-rod: mean HU, noise (σ), CNR
- `nps.csv` — Radial NPS profile (freq vs power)
- `mtf.csv` — MTF curve (freq vs modulation)
- `metadata.txt` — Protocol, geometry, bg noise, MTF50/MTF10, NPS peak

**Master summary**: `test/references/catsim/summary.csv` — all protocols × 16 rods × full metrics
"""

# ╔═╡ b0000001-0006-0001-0016-000000000002
# ╠═╡ show_logs = false
#=╠═╡
# Auto-save CatSim references: 6 protocols × (recon PNG + mask PNG + rod CSV + NPS/MTF CSVs)
ref_summary = let
	all_catsim = [catsim_scan_1, catsim_scan_2, catsim_scan_3, catsim_scan_4, catsim_scan_5, catsim_scan_6]

	hu_window = (-200, 500)
	pix_mm = recon_fov_cm * 10.0 / recon_xy
	summary_rows = NamedTuple[]

	for (i, sc) in enumerate(SCANS)
		pdir = joinpath(REFERENCES_DIR, "catsim", sc.name)
		mkpath(pdir)

		cs_vol = all_catsim[i].reconstruction
		nx, ny, nz = size(cs_vol)
		mid_z = nz ÷ 2
		cs_slice = cs_vol[:, :, mid_z]

		# ── Segment rods from CatSim reconstruction ──
		seg_mask, cs_rods, center = segment_gammex_rods(cs_slice; fov_cm=recon_fov_cm)

		# ── Background noise (center ROI in solid water region) ──
		cx, cy = nx ÷ 2, ny ÷ 2
		bg_roi_half = 30
		bg_roi = Float64.(cs_slice[cx-bg_roi_half:cx+bg_roi_half, cy-bg_roi_half:cy+bg_roi_half])
		bg_mean_hu = mean(bg_roi)
		bg_std_hu = std(bg_roi)

		# ── NPS (radial profile from background ROIs) ──
		_, nps_radial, nps_freqs = compute_nps(cs_vol; roi_size=64, pix_mm=pix_mm)
		nps_peak_freq = length(nps_freqs) > 0 ? nps_freqs[argmax(nps_radial)] : NaN
		nps_area = length(nps_freqs) > 1 ? sum(nps_radial) * (nps_freqs[2] - nps_freqs[1]) : NaN

		# ── MTF (body edge ESF → LSF → FFT) ──
		mtf_freqs, mtf_vals, _ = compute_mtf(cs_vol; pix_mm=pix_mm)
		mtf50 = NaN; mtf10 = NaN
		for k in 2:length(mtf_vals)
			if isnan(mtf50) && mtf_vals[k] <= 0.5
				# Linear interpolation
				t = (0.5 - mtf_vals[k]) / (mtf_vals[k-1] - mtf_vals[k])
				mtf50 = mtf_freqs[k] + t * (mtf_freqs[k-1] - mtf_freqs[k])
			end
			if isnan(mtf10) && mtf_vals[k] <= 0.1
				t = (0.1 - mtf_vals[k]) / (mtf_vals[k-1] - mtf_vals[k])
				mtf10 = mtf_freqs[k] + t * (mtf_freqs[k-1] - mtf_freqs[k])
			end
		end

		# ── 1. Recon PNG ──
		fig = CM.Figure(size=(512, 512), figure_padding=0)
		ax = CM.Axis(fig[1,1], aspect=CM.DataAspect())
		CM.hidedecorations!(ax); CM.hidespines!(ax)
		CM.heatmap!(ax, cs_slice, colormap=:grays, colorrange=hu_window)
		CM.save(joinpath(pdir, "recon.png"), fig, px_per_unit=1)

		# ── 2. Mask overlay PNG ──
		pixel_cm = recon_fov_cm / nx
		roi_r_pix = 1.4 * 0.7 / pixel_cm
		fig = CM.Figure(size=(700, 600), fontsize=10)
		ax = CM.Axis(fig[1,1], title="$(sc.label) — rod segmentation", aspect=CM.DataAspect())
		CM.heatmap!(ax, cs_slice, colormap=:grays, colorrange=hu_window)
		θ_circ = range(0, 2π, length=61)
		for r in cs_rods
			xs = r.cx .+ roi_r_pix .* cos.(θ_circ)
			ys = r.cy .+ roi_r_pix .* sin.(θ_circ)
			c = r.ring == :outer ? :orange : :lime
			CM.lines!(ax, xs, ys, color=c, linewidth=1.5)
			CM.text!(ax, r.cx, r.cy + roi_r_pix + 3;
				text=r.name, fontsize=7, align=(:center, :bottom), color=c)
		end
		CM.scatter!(ax, [center.cx], [center.cy], color=:red, marker=:cross, markersize=10)
		CM.save(joinpath(pdir, "mask.png"), fig, px_per_unit=2)

		# ── 3. Rod measurements CSV (HU + noise + CNR per rod) ──
		open(joinpath(pdir, "rod_measurements.csv"), "w") do f
			println(f, "name,label,ring,angle_deg,mean_hu,std_hu,cnr,n_pixels,cx,cy")
			for r in cs_rods
				cnr = bg_std_hu > 0 ? abs(r.mean_hu - bg_mean_hu) / bg_std_hu : 0.0
				println(f, join([
					r.name, Int(r.label), r.ring, r.angle_deg,
					round(r.mean_hu, digits=2), round(r.std_hu, digits=2),
					round(cnr, digits=2), r.n_pixels,
					round(r.cx, digits=1), round(r.cy, digits=1)
				], ","))
			end
		end

		# ── 4. NPS radial profile CSV ──
		open(joinpath(pdir, "nps.csv"), "w") do f
			println(f, "freq_lp_per_mm,nps_hu2_mm2")
			for k in eachindex(nps_freqs)
				println(f, round(nps_freqs[k], digits=4), ",", round(nps_radial[k], sigdigits=5))
			end
		end

		# ── 5. MTF curve CSV ──
		open(joinpath(pdir, "mtf.csv"), "w") do f
			println(f, "freq_lp_per_mm,mtf")
			for k in eachindex(mtf_freqs)
				println(f, round(mtf_freqs[k], digits=4), ",", round(mtf_vals[k], digits=4))
			end
		end

		# ── 6. Metadata ──
		open(joinpath(pdir, "metadata.txt"), "w") do f
			println(f, "protocol=", sc.name)
			println(f, "kvp=", sc.kvp)
			println(f, "mA=", sc.mA)
			println(f, "scanner=GE Revolution Apex (CatSim)")
			println(f, "center_x=", round(center.cx, digits=2))
			println(f, "center_y=", round(center.cy, digits=2))
			println(f, "rotation_deg=", center.rotation_deg)
			println(f, "slice_idx=", mid_z)
			println(f, "matrix_size=", nx)
			println(f, "fov_cm=", recon_fov_cm)
			println(f, "pix_mm=", round(pix_mm, digits=4))
			println(f, "bg_mean_hu=", round(bg_mean_hu, digits=2))
			println(f, "bg_std_hu=", round(bg_std_hu, digits=2))
			println(f, "mtf50_lp_per_mm=", round(mtf50, digits=3))
			println(f, "mtf10_lp_per_mm=", round(mtf10, digits=3))
			println(f, "nps_peak_freq_lp_per_mm=", round(nps_peak_freq, digits=3))
			println(f, "nps_area_hu2=", round(nps_area, digits=2))
		end

		# Accumulate summary
		for j in eachindex(cs_rods)
			cnr = bg_std_hu > 0 ? abs(cs_rods[j].mean_hu - bg_mean_hu) / bg_std_hu : 0.0
			push!(summary_rows, (
				protocol = sc.name, kvp = sc.kvp,
				name = cs_rods[j].name, label = cs_rods[j].label, ring = cs_rods[j].ring,
				mean_hu = round(cs_rods[j].mean_hu, digits=2),
				std_hu = round(cs_rods[j].std_hu, digits=2),
				cnr = round(cnr, digits=2),
				bg_noise = round(bg_std_hu, digits=2),
				mtf50 = round(mtf50, digits=3),
				mtf10 = round(mtf10, digits=3),
			))
		end

		@info "Saved: $(sc.label) → $pdir  ($(length(cs_rods)) rods, MTF50=$(round(mtf50, digits=2)) lp/mm)"
	end

	# ── Master summary CSV ──
	master_csv = joinpath(REFERENCES_DIR, "catsim", "summary.csv")
	open(master_csv, "w") do f
		println(f, "protocol,kvp,name,label,ring,mean_hu,std_hu,cnr,bg_noise,mtf50_lp_per_mm,mtf10_lp_per_mm")
		for r in summary_rows
			println(f, join([r.protocol, r.kvp, r.name, Int(r.label), r.ring,
				r.mean_hu, r.std_hu, r.cnr, r.bg_noise, r.mtf50, r.mtf10], ","))
		end
	end
	@info "Master summary → $master_csv  ($(length(summary_rows)) rows across $(length(SCANS)) protocols)"

	summary_rows
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0016-000000000003
# ╠═╡ disabled = true
#=╠═╡
# Summary: CatSim reference quality — noise + MTF50 per protocol
let
	fig = CM.Figure(size=(1000, 400), fontsize=11)

	# Background noise per protocol
	ax1 = CM.Axis(fig[1,1],
		title="CatSim Background Noise (σ)",
		xlabel="Protocol", ylabel="σ (HU)",
		xticks=(1:length(SCANS), [sc.name for sc in SCANS]),
		xticklabelrotation=π/6,
	)
	noise_vals = map(SCANS) do sc
		rows = filter(r -> r.protocol == sc.name, ref_summary)
		isempty(rows) ? 0.0 : first(rows).bg_noise
	end
	colors = map(SCANS) do sc
		sc.kvp == 120 ? :steelblue : sc.kvp == 80 ? :firebrick :
		sc.kvp == 100 ? :forestgreen : :darkorange
	end
	CM.barplot!(ax1, 1:length(SCANS), noise_vals, color=colors)
	for (i, v) in enumerate(noise_vals)
		CM.text!(ax1, i, v + 0.3; text="$(round(v, digits=1))",
			fontsize=9, align=(:center, :bottom))
	end

	# MTF50 per protocol
	ax2 = CM.Axis(fig[1,2],
		title="CatSim Spatial Resolution (MTF50)",
		xlabel="Protocol", ylabel="MTF50 (lp/mm)",
		xticks=(1:length(SCANS), [sc.name for sc in SCANS]),
		xticklabelrotation=π/6,
	)
	mtf_vals = map(SCANS) do sc
		rows = filter(r -> r.protocol == sc.name, ref_summary)
		isempty(rows) ? 0.0 : first(rows).mtf50
	end
	CM.barplot!(ax2, 1:length(SCANS), mtf_vals, color=colors)
	for (i, v) in enumerate(mtf_vals)
		CM.text!(ax2, i, v + 0.01; text="$(round(v, digits=2))",
			fontsize=9, align=(:center, :bottom))
	end

	CM.save(joinpath(FIGURES_DIR, "nb06_ref_quality.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0010-000000000001
md"""
---
## 10. Reconstruction Gallery

Interactive comparison: select protocol and slice.
"""

# ╔═╡ b0000001-0006-0001-0010-000000000002
@bind selected_scan_idx UI.Slider(1:length(SCANS), default=2, show_value=true)

# ╔═╡ b0000001-0006-0001-0010-000000000003
@bind gallery_slice UI.Slider(1:n_recon_slices, default=n_recon_slices ÷ 2, show_value=true)

# ╔═╡ b0000001-0006-0001-0010-000000000004
#=╠═╡
let
	sc = SCANS[selected_scan_idx]
	bs_vol = basis_results[sc.name].recon
	cs_vol = catsim_results[sc.name].reconstruction

	fig = CM.Figure(size=(1200, 400), fontsize=12)
	vmin, vmax = -200, 400

	ax1 = CM.Axis(fig[1,1], title="BasisSimulator", aspect=CM.DataAspect())
	hm1 = CM.heatmap!(ax1, bs_vol[:,:,gallery_slice], colormap=:grays, colorrange=(vmin, vmax))

	ax2 = CM.Axis(fig[1,2], title="CatSim", aspect=CM.DataAspect())
	hm2 = CM.heatmap!(ax2, cs_vol[:,:,gallery_slice], colormap=:grays, colorrange=(vmin, vmax))
	CM.Colorbar(fig[1,3], hm2, label="HU")

	ax3 = CM.Axis(fig[1,4], title="Difference", aspect=CM.DataAspect())
	diff_img = bs_vol[:,:,gallery_slice] .- cs_vol[:,:,gallery_slice]
	max_d = max(abs(minimum(diff_img)), abs(maximum(diff_img)), 10.0)
	hm3 = CM.heatmap!(ax3, diff_img, colormap=:RdBu, colorrange=(-max_d, max_d))
	CM.Colorbar(fig[1,5], hm3, label="ΔHU")

	CM.Label(fig[0, :], text="$(sc.label) — Slice $gallery_slice", fontsize=16, font=:bold)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0010-000000000005
#=╠═╡
# 6-panel overview: one row per protocol, central slice
let
	fig = CM.Figure(size=(700, 1200), fontsize=10)
	vmin, vmax = -200, 400
	mid_z = n_recon_slices ÷ 2

	for (row, sc) in enumerate(SCANS)
		bs_vol = basis_results[sc.name].recon
		cs_vol = catsim_results[sc.name].reconstruction

		ax1 = CM.Axis(fig[row, 1], title=(row==1 ? "BasisSimulator" : ""), ylabel=sc.label, aspect=CM.DataAspect())
		CM.heatmap!(ax1, bs_vol[:,:,mid_z], colormap=:grays, colorrange=(vmin, vmax))
		CM.hidedecorations!(ax1, label=false)

		ax2 = CM.Axis(fig[row, 2], title=(row==1 ? "CatSim" : ""), aspect=CM.DataAspect())
		hm = CM.heatmap!(ax2, cs_vol[:,:,mid_z], colormap=:grays, colorrange=(vmin, vmax))
		CM.hidedecorations!(ax2)

		if row == length(SCANS)
			CM.Colorbar(fig[row, 3], hm, label="HU")
		end
	end

	CM.save(joinpath(FIGURES_DIR, "nb06_gallery.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0011-000000000001
md"""
---
## 11. Forward Projection Comparison

Sinogram metrics (RMSE, NRMSE, correlation) for all 6 protocols.
"""

# ╔═╡ b0000001-0006-0001-0011-000000000002
#=╠═╡
# Compute sinogram metrics for all protocols
sino_metrics_all = let
	metrics = []
	for sc in SCANS
		sino_bs = basis_results[sc.name].sinogram
		sino_cs = catsim_results[sc.name].sinogram

		diff = sino_bs .- sino_cs
		rmse = sqrt(mean(diff.^2))
		mean_val = mean([mean(sino_bs), mean(sino_cs)])
		nrmse_pct = 100.0 * rmse / mean_val
		correlation = cor(vec(sino_bs), vec(sino_cs))

		push!(metrics, (
			label=sc.label, name=sc.name,
			rmse=rmse, nrmse_pct=nrmse_pct, correlation=correlation,
			sino_bs=sino_bs, sino_cs=sino_cs, diff=diff
		))
	end
	metrics
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0011-000000000003
#=╠═╡
# Sinogram comparison for selected protocol
let
	sc = SCANS[selected_scan_idx]
	m = sino_metrics_all[selected_scan_idx]
	central_row = active_rows ÷ 2

	fig = CM.Figure(size=(1200, 400), fontsize=12)

	ax1 = CM.Axis(fig[1,1], title="BasisSimulator", xlabel="Col", ylabel="View")
	CM.heatmap!(ax1, m.sino_bs[:, central_row, :], colormap=:grays)

	ax2 = CM.Axis(fig[1,2], title="CatSim", xlabel="Col", ylabel="View")
	CM.heatmap!(ax2, m.sino_cs[:, central_row, :], colormap=:grays)

	ax3 = CM.Axis(fig[1,3], title="Difference", xlabel="Col", ylabel="View")
	diff_slice = m.diff[:, central_row, :]
	max_diff = max(maximum(abs.(diff_slice)), 1e-6)
	hm3 = CM.heatmap!(ax3, diff_slice, colormap=:RdBu, colorrange=(-max_diff, max_diff))
	CM.Colorbar(fig[1,4], hm3)

	CM.Label(fig[0, :], text="$(sc.label) — RMSE=$(round(m.rmse, digits=4)), r=$(round(m.correlation, digits=5))", fontsize=14, font=:bold)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0011-000000000004
#=╠═╡
# Summary table
let
	fig = CM.Figure(size=(800, 300), fontsize=14)

	labels = [m.label for m in sino_metrics_all]
	rmses = [m.rmse for m in sino_metrics_all]
	nrmses = [m.nrmse_pct for m in sino_metrics_all]
	corrs = [m.correlation for m in sino_metrics_all]

	ax = CM.Axis(fig[1,1], title="Sinogram NRMSE (%)", ylabel="NRMSE %", xticks=(1:6, labels), xticklabelrotation=pi/6)
	CM.barplot!(ax, 1:6, nrmses, color=:steelblue)
	for (i, v) in enumerate(nrmses)
		CM.text!(ax, i, v + 0.1, text="$(round(v, digits=2))%", align=(:center, :bottom), fontsize=10)
	end

	ax2 = CM.Axis(fig[1,2], title="Sinogram Correlation", ylabel="r", xticks=(1:6, labels), xticklabelrotation=pi/6)
	CM.barplot!(ax2, 1:6, corrs, color=:coral)
	CM.ylims!(ax2, min(0.99, minimum(corrs) - 0.002), 1.001)

	CM.save(joinpath(FIGURES_DIR, "nb06_sino_metrics.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0012-000000000001
md"""
---
## 12. Quantitative HU Accuracy

ROI measurements for all 14 inserts (7 Ca + 7 I) across all 6 protocols.
"""

# ╔═╡ b0000001-0006-0001-0012-000000000002
#=╠═╡
# Compute ROI measurements for all protocols
# Outer ring (Ca): gap at 12 o'clock, Ca100→Ca200→Ca300→Ca400→...→Ca50
# Inner ring (I): starts at 12 o'clock, I2.5→I5.0→I7.5→I10→I15→I20→...→I2.0
hu_results_all = let
	fov_mm = recon_fov_cm * 10.0
	pixel_size_mm = fov_mm / recon_xy
	cx, cy = recon_xy ÷ 2, recon_xy ÷ 2

	outer_ring_r = 105.0 / pixel_size_mm   # Ca ring
	inner_ring_r = 50.0 / pixel_size_mm    # I ring
	roi_radius = 5

	# Outer ring Ca inserts (exact angles, gap at top)
	outer_start = π/2 - π/8
	ca_inserts = [
		("Ca 100", outer_start - 0*π/4),
		("Ca 200", outer_start - 1*π/4),
		("Ca 300", outer_start - 2*π/4),
		("Ca 400", outer_start - 3*π/4),
		("Ca 50",  outer_start - 7*π/4),
	]

	# Inner ring I inserts (exact angles, starts at top)
	inner_start = π/2
	i_inserts = [
		("I 2.5",  inner_start - 0*π/4),
		("I 5.0",  inner_start - 1*π/4),
		("I 7.5",  inner_start - 2*π/4),
		("I 10.0", inner_start - 3*π/4),
		("I 15.0", inner_start - 4*π/4),
		("I 20.0", inner_start - 5*π/4),
		("I 2.0",  inner_start - 7*π/4),
	]

	all_inserts = vcat(
		[(name, outer_ring_r, ang) for (name, ang) in ca_inserts],
		[(name, inner_ring_r, ang) for (name, ang) in i_inserts],
	)

	all_results = Dict{String, Vector}()

	for sc in SCANS
		bs_vol = basis_results[sc.name].recon
		cs_vol = catsim_results[sc.name].reconstruction
		central_slice = size(bs_vol, 3) ÷ 2

		results = []
		for (name, ring_r, angle) in all_inserts
			ix = cx + round(Int, ring_r * cos(angle))
			iy = cy + round(Int, ring_r * sin(angle))

			roi_bs = bs_vol[ix-roi_radius:ix+roi_radius, iy-roi_radius:iy+roi_radius, central_slice]
			roi_cs = cs_vol[ix-roi_radius:ix+roi_radius, iy-roi_radius:iy+roi_radius, central_slice]

			push!(results, (name=name, basis_hu=mean(roi_bs), catsim_hu=mean(roi_cs)))
		end

		all_results[sc.name] = results
	end
	all_results
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0012-000000000003
#=╠═╡
# Scatter plot: CatSim HU vs BasisSimulator HU for selected protocol
let
	sc = SCANS[selected_scan_idx]
	results = hu_results_all[sc.name]

	ca_results = filter(r -> startswith(r.name, "Ca"), results)
	i_results = filter(r -> startswith(r.name, "I"), results)

	fig = CM.Figure(size=(900, 900), fontsize=16)

	# Calcium
	ax1 = CM.Axis(fig[1,1], xlabel="CatSim HU", ylabel="BasisSimulator HU", title="Calcium — $(sc.label)")
	ca_x = [r.catsim_hu for r in ca_results]
	ca_y = [r.basis_hu for r in ca_results]
	CM.scatter!(ax1, ca_x, ca_y, color=:blue, marker=:circle, markersize=12, label="Calcium")

	all_vals = vcat(ca_x, ca_y)
	lo, hi = minimum(all_vals), maximum(all_vals)
	pad = 0.05 * (hi - lo)
	CM.lines!(ax1, [lo-pad, hi+pad], [lo-pad, hi+pad], color=:gray, linestyle=:dash, label="Identity")

	n = length(ca_x)
	slope = (n * sum(ca_x .* ca_y) - sum(ca_x) * sum(ca_y)) / (n * sum(ca_x.^2) - sum(ca_x)^2)
	intercept = (sum(ca_y) - slope * sum(ca_x)) / n
	xfit = collect(range(lo-pad, hi+pad, length=100))
	sign_str = intercept >= 0 ? "+" : "-"
	eq = "y = $(round(slope, digits=3))x $(sign_str) $(round(abs(intercept), digits=1))"
	CM.lines!(ax1, xfit, intercept .+ slope .* xfit, color=:blue, linewidth=2, label=eq)
	CM.axislegend(ax1, position=:rb)

	# Iodine
	ax2 = CM.Axis(fig[2,1], xlabel="CatSim HU", ylabel="BasisSimulator HU", title="Iodine — $(sc.label)")
	i_x = [r.catsim_hu for r in i_results]
	i_y = [r.basis_hu for r in i_results]
	CM.scatter!(ax2, i_x, i_y, color=:red, marker=:utriangle, markersize=12, label="Iodine")

	all_i = vcat(i_x, i_y)
	lo_i, hi_i = minimum(all_i), maximum(all_i)
	pad_i = 0.05 * (hi_i - lo_i)
	CM.lines!(ax2, [lo_i-pad_i, hi_i+pad_i], [lo_i-pad_i, hi_i+pad_i], color=:gray, linestyle=:dash, label="Identity")

	n_i = length(i_x)
	slope_i = (n_i * sum(i_x .* i_y) - sum(i_x) * sum(i_y)) / (n_i * sum(i_x.^2) - sum(i_x)^2)
	int_i = (sum(i_y) - slope_i * sum(i_x)) / n_i
	xfit_i = collect(range(lo_i-pad_i, hi_i+pad_i, length=100))
	sign_i = int_i >= 0 ? "+" : "-"
	eq_i = "y = $(round(slope_i, digits=3))x $(sign_i) $(round(abs(int_i), digits=1))"
	CM.lines!(ax2, xfit_i, int_i .+ slope_i .* xfit_i, color=:red, linewidth=2, label=eq_i)
	CM.axislegend(ax2, position=:rb)

	CM.save(joinpath(FIGURES_DIR, "nb06_hu_accuracy.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0013-000000000001
md"""
---
## 13. Noise Power Spectrum (NPS)

Radial NPS comparison across all 6 protocols.
"""

# ╔═╡ b0000001-0006-0001-0013-000000000002
function compute_nps(image_vol; roi_size=64, pix_mm=recon_fov_cm * 10.0 / recon_xy)
	nx, ny, nz = size(image_vol)
	cx, cy, mid_z = nx ÷ 2, ny ÷ 2, nz ÷ 2

	half_r = roi_size ÷ 2
	offsets = [
		(-half_r-10, -half_r-10), (half_r+10, -half_r-10),
		(-half_r-10, half_r+10), (half_r+10, half_r+10)
	]

	ps_accum = zeros(Float64, roi_size, roi_size)
	count = 0

	win_1d = 0.5 .* (1.0 .- cos.(2π .* (0:roi_size-1) ./ (roi_size-1)))
	win_2d = win_1d * win_1d'

	# Average over central slices (guard against edge)
	z_range = max(1, mid_z-1):min(nz, mid_z+1)

	for (dx, dy) in offsets
		for z in z_range
			x0 = cx + dx - half_r
			y0 = cy + dy - half_r

			# Bounds check
			if x0 < 1 || y0 < 1 || x0+roi_size-1 > nx || y0+roi_size-1 > ny
				continue
			end

			roi = Float64.(image_vol[x0:x0+roi_size-1, y0:y0+roi_size-1, z])
			roi .-= mean(roi)

			roi_fft = fft(roi .* win_2d)
			ps_accum .+= abs2.(roi_fft)
			count += 1
		end
	end

	if count == 0
		return zeros(roi_size, roi_size), zeros(roi_size ÷ 2), range(0, 1, length=roi_size ÷ 2)
	end

	nps_2d = fftshift(ps_accum ./ count) .* (pix_mm^2 / roi_size^2)

	freqs = fftshift(fftfreq(roi_size, 1/pix_mm))
	n_bins = roi_size ÷ 2
	radial_prof = zeros(n_bins)
	radial_counts = zeros(Int, n_bins)
	max_f = maximum(freqs)

	for i in 1:roi_size, j in 1:roi_size
		f_val = sqrt(freqs[i]^2 + freqs[j]^2)
		bin_idx = floor(Int, (f_val / max_f) * (n_bins-1)) + 1
		if bin_idx <= n_bins
			radial_prof[bin_idx] += nps_2d[i,j]
			radial_counts[bin_idx] += 1
		end
	end

	mask = radial_counts .> 0
	radial_prof[mask] ./= radial_counts[mask]
	radial_freqs = range(0, stop=max_f, length=n_bins)

	return nps_2d, radial_prof, radial_freqs
end

# ╔═╡ b0000001-0006-0001-0013-000000000003
#=╠═╡
# Compute NPS for all protocols
nps_all = let
	results = Dict{String, NamedTuple}()
	for sc in SCANS
		_, rad_bs, f_bs = compute_nps(basis_results[sc.name].recon)
		_, rad_cs, f_cs = compute_nps(catsim_results[sc.name].reconstruction)
		results[sc.name] = (rad_bs=rad_bs, f_bs=f_bs, rad_cs=rad_cs, f_cs=f_cs)
	end
	results
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0013-000000000004
#=╠═╡
# NPS comparison — dose variation (120 kVp)
let
	fig = CM.Figure(size=(1000, 800))

	# Dose variation (Scans 1-3)
	ax1 = CM.Axis(fig[1,1], title="NPS — Dose Variation (120 kVp)", xlabel="Frequency (mm⁻¹)", ylabel="Power (HU² mm²)")
	colors = [:steelblue, :navy, :darkblue]
	for (i, sc_idx) in enumerate([1, 2, 3])
		sc = SCANS[sc_idx]
		nps = nps_all[sc.name]
		CM.lines!(ax1, nps.f_bs, nps.rad_bs, color=colors[i], linewidth=2, label="BS: $(sc.label)")
		CM.lines!(ax1, nps.f_cs, nps.rad_cs, color=colors[i], linewidth=2, linestyle=:dash, label="CS: $(sc.label)")
	end
	CM.axislegend(ax1, position=:rt, fontsize=9)

	# kVp variation (Scans 4-6 + Scan 2)
	ax2 = CM.Axis(fig[2,1], title="NPS — kVp Variation (~10 mGy)", xlabel="Frequency (mm⁻¹)", ylabel="Power (HU² mm²)")
	kvp_colors = [:orange, :green, :navy, :purple]
	kvp_scans = [4, 5, 2, 6]
	for (i, sc_idx) in enumerate(kvp_scans)
		sc = SCANS[sc_idx]
		nps = nps_all[sc.name]
		CM.lines!(ax2, nps.f_bs, nps.rad_bs, color=kvp_colors[i], linewidth=2, label="BS: $(sc.label)")
		CM.lines!(ax2, nps.f_cs, nps.rad_cs, color=kvp_colors[i], linewidth=2, linestyle=:dash, label="CS: $(sc.label)")
	end
	CM.axislegend(ax2, position=:rt, fontsize=9)

	CM.save(joinpath(FIGURES_DIR, "nb06_nps.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0014-000000000001
md"""
---
## 14. Modulation Transfer Function (MTF)

ESF method at the phantom body edge (water → air).
"""

# ╔═╡ b0000001-0006-0001-0014-000000000002
function compute_mtf(image_vol; pix_mm=recon_fov_cm * 10.0 / recon_xy)
	nx, ny, nz = size(image_vol)
	cx, cy, mid_z = nx ÷ 2, ny ÷ 2, nz ÷ 2

	r_pix = 165.0 / pix_mm  # phantom body radius in pixels
	edge_est = cx + round(Int, r_pix)

	win = 12
	start_x = max(1, edge_est - win)
	end_x = min(nx, edge_est + win)

	prof_accum = zeros(Float64, end_x - start_x + 1)
	rows = -2:2
	for dy in rows
		prof_accum .+= image_vol[start_x:end_x, cy+dy, mid_z]
	end
	esf = prof_accum ./ length(rows)

	lsf = diff(esf)
	if mean(lsf) < 0
		lsf = -lsf
	end

	win_func = 0.5 .* (1.0 .- cos.(2π .* (0:length(lsf)-1) ./ (length(lsf)-1)))
	lsf = lsf .* win_func

	mtf_raw = abs.(fft(lsf))
	mtf = mtf_raw[1] > 0 ? mtf_raw ./ mtf_raw[1] : zeros(length(mtf_raw))

	freqs = fftfreq(length(lsf), 1/pix_mm)
	n_out = length(freqs) ÷ 2
	return freqs[1:n_out], mtf[1:n_out], esf
end

# ╔═╡ b0000001-0006-0001-0014-000000000003
#=╠═╡
# MTF for standard protocol (120 kVp / 10 mGy)
let
	sc = SCANS[2]  # Standard: 120 kVp / 10 mGy
	f_mtf, mtf_bs, esf_bs = compute_mtf(basis_results[sc.name].recon)
	_, mtf_cs, esf_cs = compute_mtf(catsim_results[sc.name].reconstruction)

	fig = CM.Figure(size=(900, 400))

	ax1 = CM.Axis(fig[1,1], title="Edge Spread Function — $(sc.label)", xlabel="Pixel Index", ylabel="HU")
	CM.lines!(ax1, esf_bs, color=:blue, label="BasisSimulator")
	CM.lines!(ax1, esf_cs, color=:red, linestyle=:dash, label="CatSim")
	CM.axislegend(ax1, position=:lb)

	ax2 = CM.Axis(fig[1,2], title="MTF — $(sc.label)", xlabel="Frequency (lp/mm)", ylabel="Modulation Factor")
	CM.lines!(ax2, f_mtf, mtf_bs, color=:blue, linewidth=2, label="BasisSimulator")
	CM.lines!(ax2, f_mtf, mtf_cs, color=:red, linewidth=2, linestyle=:dash, label="CatSim")
	CM.hlines!(ax2, [0.5, 0.1], color=:gray, linestyle=:dot)
	CM.text!(ax2, 0.1, 0.52, text="50%")
	CM.text!(ax2, 0.1, 0.12, text="10%")
	CM.axislegend(ax2)

	CM.save(joinpath(FIGURES_DIR, "nb06_mtf.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0015-000000000001
md"""
---
## 15. Contrast-to-Noise Ratio (CNR)

$$CNR = \frac{|HU_{insert} - HU_{background}|}{\sigma_{background}}$$

Compared across all 6 protocols for both simulators.
"""

# ╔═╡ b0000001-0006-0001-0015-000000000002
#=╠═╡
# CNR for all protocols
cnr_all = let
	results = Dict{String, NamedTuple}()
	cx, cy = recon_xy ÷ 2, recon_xy ÷ 2
	roi_sz = 30

	for sc in SCANS
		bs_vol = basis_results[sc.name].recon
		cs_vol = catsim_results[sc.name].reconstruction
		mid_z = size(bs_vol, 3) ÷ 2

		bg_bs = bs_vol[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]
		bg_cs = cs_vol[cx-roi_sz:cx+roi_sz, cy-roi_sz:cy+roi_sz, mid_z]

		μ_bg_bs = mean(bg_bs)
		σ_bg_bs = std(bg_bs)
		μ_bg_cs = mean(bg_cs)
		σ_bg_cs = std(bg_cs)

		hu = hu_results_all[sc.name]
		labels = [r.name for r in hu]
		cnr_bs = [σ_bg_bs > 0 ? abs(r.basis_hu - μ_bg_bs) / σ_bg_bs : 0.0 for r in hu]
		cnr_cs = [σ_bg_cs > 0 ? abs(r.catsim_hu - μ_bg_cs) / σ_bg_cs : 0.0 for r in hu]

		results[sc.name] = (labels=labels, cnr_bs=cnr_bs, cnr_cs=cnr_cs, sigma_bs=σ_bg_bs, sigma_cs=σ_bg_cs)
	end
	results
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0015-000000000003
#=╠═╡
# CNR bar chart for selected protocol
let
	sc = SCANS[selected_scan_idx]
	cnr = cnr_all[sc.name]

	fig = CM.Figure(size=(1000, 500))

	ax = CM.Axis(fig[1,1], title="CNR — $(sc.label)", xticklabelrotation=pi/4, ylabel="CNR")

	x = 1:length(cnr.labels)
	CM.barplot!(ax, x .- 0.2, cnr.cnr_bs, width=0.4, color=:blue, label="BasisSimulator")
	CM.barplot!(ax, x .+ 0.2, cnr.cnr_cs, width=0.4, color=:red, label="CatSim")

	ax.xticks = (x, cnr.labels)
	CM.axislegend(ax, position=:lt)

	text_str = "Background Noise (σ):\nBasis: $(round(cnr.sigma_bs, digits=2)) HU\nCatSim: $(round(cnr.sigma_cs, digits=2)) HU"
	CM.text!(ax, length(x)-4, maximum(vcat(cnr.cnr_bs, cnr.cnr_cs))*0.8, text=text_str, fontsize=12)

	CM.save(joinpath(FIGURES_DIR, "nb06_cnr.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0015-000000000004
#=╠═╡
# Noise summary: σ across all 6 protocols
let
	fig = CM.Figure(size=(900, 400))

	labels = [sc.label for sc in SCANS]
	sigma_bs = [cnr_all[sc.name].sigma_bs for sc in SCANS]
	sigma_cs = [cnr_all[sc.name].sigma_cs for sc in SCANS]

	ax = CM.Axis(fig[1,1], title="Background Noise (σ) — All Protocols", ylabel="σ (HU)", xticks=(1:6, labels), xticklabelrotation=pi/6)
	CM.barplot!(ax, (1:6) .- 0.2, sigma_bs, width=0.4, color=:blue, label="BasisSimulator")
	CM.barplot!(ax, (1:6) .+ 0.2, sigma_cs, width=0.4, color=:red, label="CatSim")
	for (i, (vb, vc)) in enumerate(zip(sigma_bs, sigma_cs))
		CM.text!(ax, i - 0.2, vb + 0.5, text="$(round(vb, digits=1))", align=(:center, :bottom), fontsize=9)
		CM.text!(ax, i + 0.2, vc + 0.5, text="$(round(vc, digits=1))", align=(:center, :bottom), fontsize=9)
	end
	CM.axislegend(ax, position=:lt)

	CM.save(joinpath(FIGURES_DIR, "nb06_noise_summary.png"), fig, px_per_unit=2)
	fig
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0008-000000000029
#=╠═╡
# Compare BasisSim Scan 1 against saved CatSim reference
let
	ref_csv = joinpath(REFERENCES_DIR, "catsim", SCANS[1].name, "rod_measurements.csv")
	if isfile(ref_csv)
		ref_rods = load_rod_reference(ref_csv)
		basis_rods = seg_result.rods

		fig = CM.Figure(size=(1000, 500), fontsize=11)
		n = length(ref_rods)
		names = [r.name for r in ref_rods]
		Δhu = [basis_rods[i].mean_hu - ref_rods[i].mean_hu for i in 1:n]

		colors = map(ref_rods) do r
			r.ring == :outer ?
				(startswith(r.name, "Ca") ? :darkorange : :steelblue) :
				(startswith(r.name, "I") ? :forestgreen : :steelblue)
		end

		ax = CM.Axis(fig[1,1],
			title="BasisSim vs CatSim Reference — $(SCANS[1].label)",
			ylabel="ΔHU (BasisSim − Reference)",
			xticks=(1:n, names),
			xticklabelrotation=π/4,
		)
		CM.barplot!(ax, 1:n, Δhu, color=colors)
		CM.hlines!(ax, [0], color=:black, linewidth=0.5)

		for (i, d) in enumerate(Δhu)
			CM.text!(ax, i, d + sign(d) * 3;
				text="$(round(d, digits=1))",
				fontsize=8, align=(:center, d >= 0 ? :bottom : :top))
		end

		fig
	else
		md"**No CatSim reference found.** Run and save CatSim scans first."
	end
end
  ╠═╡ =#

# ╔═╡ b0000001-0006-0001-0008-000000000028
#=╠═╡
# Save CatSim Scan 1 as reference (enable after running CatSim scans)
let
	# Segment CatSim reconstruction
	cs_recon = catsim_scan_1.reconstruction
	mid_z = size(cs_recon, 3) ÷ 2
	_, cs_rods, cs_center = segment_gammex_rods(cs_recon[:,:,mid_z]; fov_cm=recon_fov_cm)

	save_scan_reference(
		joinpath(REFERENCES_DIR, "catsim"),
		SCANS[1].name,
		cs_recon, cs_rods, cs_center;
		fov_cm=recon_fov_cm,
	)
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─b0000001-0006-0001-0001-000000000008
# ╠═b0000001-0006-0001-0001-000000000001
# ╠═b0000001-0006-0001-0001-000000000009
# ╠═b0000001-0006-0001-0001-000000000010
# ╠═b0000001-0006-0001-0001-000000000002
# ╠═b0000001-0006-0001-0001-000000000011
# ╠═b0000001-0006-0001-0001-000000000012
# ╠═b0000001-0006-0001-0001-000000000013
# ╠═b0000001-0006-0001-0001-000000000003
# ╠═b0000001-0006-0001-0001-000000000004
# ╠═b0000001-0006-0001-0001-000000000005
# ╠═b0000001-0006-0001-0001-000000000006
# ╠═b0000001-0006-0001-0001-000000000007
# ╠═b0000001-0006-0001-0001-000000000014
# ╠═b0000001-0006-0001-0001-000000000015
# ╟─b0000001-0006-0001-0002-000000000001
# ╠═b0000001-0006-0001-0002-000000000002
# ╟─b0000001-0006-0001-0003-000000000001
# ╠═b0000001-0006-0001-0003-000000000002
# ╠═b0000001-0006-0001-0003-000000000003
# ╠═b0000001-0006-0001-0003-000000000004
# ╟─b0000001-0006-0001-0004-000000000001
# ╠═b0000001-0006-0001-0004-000000000002
# ╠═b0000001-0006-0001-0004-000000000003
# ╠═b0000001-0006-0001-0004-000000000004
# ╟─b0000001-0006-0001-0005-000000000001
# ╠═b0000001-0006-0001-0005-000000000002
# ╠═b0000001-0006-0001-0005-000000000003
# ╠═b0000001-0006-0001-0005-000000000004
# ╠═b0000001-0006-0001-0005-000000000005
# ╠═b0000001-0006-0001-0005-000000000006
# ╠═b0000001-0006-0001-0005-000000000007
# ╠═b0000001-0006-0001-0005-000000000008
# ╠═b0000001-0006-0001-0005-000000000009
# ╠═b0000001-0006-0001-0005-000000000010
# ╠═b0000001-0006-0001-0005-000000000011
# ╠═b0000001-0006-0001-0005-000000000012
# ╠═b0000001-0006-0001-0005-000000000013
# ╠═b0000001-0006-0001-0005-000000000014
# ╟─b0000001-0006-0001-0006-000000000001
# ╠═b0000001-0006-0001-0006-000000000002
# ╠═b0000001-0006-0001-0006-000000000012
# ╠═b0000001-0006-0001-0006-000000000013
# ╠═b0000001-0006-0001-0006-000000000005
# ╠═b0000001-0006-0001-0006-000000000003
# ╠═b0000001-0006-0001-0006-000000000004
# ╟─b0000001-0006-0001-0007-000000000001
# ╠═b0000001-0006-0001-0007-000000000002
# ╠═b0000001-0006-0001-0007-000000000003
# ╠═b0000001-0006-0001-0007-000000000004
# ╟─b0000001-0006-0001-0007-000000000010
# ╟─b0000001-0006-0001-0007-000000000009
# ╠═b0000001-0006-0001-0007-000000000005
# ╠═b0000001-0006-0001-0007-000000000008
# ╟─b0000001-0006-0001-0008-000000000001
# ╠═b0000001-0006-0001-0007-000000000006
# ╠═b0000001-0006-0001-0008-000000000003
# ╟─ba7ca05f-a73d-4c7b-98bc-ab6aa06361e5
# ╟─2c9082ae-2305-4ecd-bc8e-9f5115a3946b
# ╟─b0000001-0006-0001-0008-000000000020
# ╠═b0000001-0006-0001-0008-000000000021
# ╠═b0000001-0006-0001-0008-000000000022
# ╟─b0000001-0006-0001-0008-000000000023
# ╟─b0000001-0006-0001-0008-000000000024
# ╠═b0000001-0006-0001-0008-000000000025
# ╠═b0000001-0006-0001-0008-000000000026
# ╠═b0000001-0006-0001-0008-000000000027
# ╠═b0000001-0006-0001-0008-000000000004
# ╠═b0000001-0006-0001-0008-000000000005
# ╠═b0000001-0006-0001-0008-000000000002
# ╠═b0000001-0006-0001-0008-000000000006
# ╠═b0000001-0006-0001-0008-000000000010
# ╠═b0000001-0006-0001-0008-000000000007
# ╠═b0000001-0006-0001-0008-000000000011
# ╠═b0000001-0006-0001-0008-000000000008
# ╠═b0000001-0006-0001-0008-000000000009
# ╟─b0000001-0006-0001-0009-000000000001
# ╠═b0000001-0006-0001-0009-000000000002
# ╠═b0000001-0006-0001-0007-000000000007
# ╠═b0000001-0006-0001-0009-000000000004
# ╠═b0000001-0006-0001-0009-000000000005
# ╠═b0000001-0006-0001-0009-000000000006
# ╠═b0000001-0006-0001-0009-000000000003
# ╠═b0000001-0006-0001-0009-000000000007
# ╠═b0000001-0006-0001-0009-000000000011
# ╠═b0000001-0006-0001-0009-000000000008
# ╠═b0000001-0006-0001-0009-000000000012
# ╠═b0000001-0006-0001-0009-000000000009
# ╠═b0000001-0006-0001-0009-000000000010
# ╟─b0000001-0006-0001-0016-000000000001
# ╠═b0000001-0006-0001-0016-000000000002
# ╟─b0000001-0006-0001-0016-000000000003
# ╟─b0000001-0006-0001-0010-000000000001
# ╟─b0000001-0006-0001-0010-000000000002
# ╟─b0000001-0006-0001-0010-000000000003
# ╟─b0000001-0006-0001-0010-000000000004
# ╟─b0000001-0006-0001-0010-000000000005
# ╟─b0000001-0006-0001-0011-000000000001
# ╠═b0000001-0006-0001-0011-000000000002
# ╟─b0000001-0006-0001-0011-000000000003
# ╟─b0000001-0006-0001-0011-000000000004
# ╟─b0000001-0006-0001-0012-000000000001
# ╠═b0000001-0006-0001-0012-000000000002
# ╟─b0000001-0006-0001-0012-000000000003
# ╟─b0000001-0006-0001-0013-000000000001
# ╠═b0000001-0006-0001-0013-000000000002
# ╠═b0000001-0006-0001-0013-000000000003
# ╟─b0000001-0006-0001-0013-000000000004
# ╟─b0000001-0006-0001-0014-000000000001
# ╠═b0000001-0006-0001-0014-000000000002
# ╟─b0000001-0006-0001-0014-000000000003
# ╟─b0000001-0006-0001-0015-000000000001
# ╠═b0000001-0006-0001-0015-000000000002
# ╟─b0000001-0006-0001-0015-000000000003
# ╟─b0000001-0006-0001-0015-000000000004
# ╠═b0000001-0006-0001-0008-000000000029
# ╠═b0000001-0006-0001-0008-000000000028
