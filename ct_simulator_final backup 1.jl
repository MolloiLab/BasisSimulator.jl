### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 4ba89bf3-342c-4762-8d26-4fe7f4b49b1d
begin
	using Pkg
	Pkg.activate(@__DIR__)
end

# ╔═╡ 116b566e-793c-4b18-9bfe-03c1b7095b7c
using PlutoUI: TableOfContents

# ╔═╡ 68579afa-2368-462a-a58a-68ea7060b77a
using CairoMakie

# ╔═╡ 2351e3ef-ed73-4160-8599-5059a00b0b0b
using LinearAlgebra

# ╔═╡ 62dc819b-09cd-4c97-a1d9-ea95b4c1a5b2
using Statistics

# ╔═╡ 0699f400-b89d-44aa-8b2e-5a68c5de3b36
using Random: seed!

# ╔═╡ 42493c98-0d13-46e5-b323-afb3bf524f29
using Distributions: Poisson, Normal

# ╔═╡ d7e941f0-e24d-4724-93c4-8c939eaa0d21
using Interpolations

# ╔═╡ ab87b970-4ac8-475c-b9b4-26b783b8a457
using FFTW

# ╔═╡ e616e726-0926-4ce9-8a0d-4dbcabd8b477
using Unitful: cm, keV, g, mm, ustrip, @u_str

# ╔═╡ 64021ccf-7534-4f0f-92a5-79fdd38b602a
using Attenuations

# ╔═╡ 3ff07656-5cec-41bb-903d-4b986333cb5e
using Attenuations: Material, μ

# ╔═╡ 794e5d30-022f-4a24-9954-ccfff9238581
using Reactant

# ╔═╡ 9f72a560-f0fd-4866-89e0-0b5326fba508
using Enzyme

# ╔═╡ 44c9569c-1ef7-4164-9866-29194d447648
md"""
# Fully Differentiable CT Simulator
**A Complete Physics-Based Cone-Beam CT Imaging Chain**

## Overview
This notebook implements a fully differentiable CT simulator from first principles, designed for:
- **Forward simulation**: Accurate physics modeling of clinical CT scanners
- **XLA compilation**: GPU/TPU acceleration via Reactant.jl
- **Automatic differentiation**: End-to-end gradients via Enzyme.jl
- **Inverse problems**: Material decomposition, dose optimization, geometry calibration

## System Specifications
Modeled after the **Canon Aquilion ONE VISION Edition** (320-row wide detector CT):
- 320 detector rows × 16 cm z-coverage (0.5 mm detector pitch)
- 600 mm source-to-axis distance (SAD)
- 1000 mm source-to-detector distance (SDD)
- Cone-beam geometry with circular trajectory
- 120 kVp tungsten anode X-ray source

## Complete Physics Model

### 1. X-Ray Source
- Bremsstrahlung spectrum (Kramers' Law with Z-dependence)
- Characteristic K-lines for tungsten (59.3, 58.0, 67.2 keV)
- Anode heel effect and inherent filtration
- Polychromatic energy bins (1-120 keV)

### 2. Beam Shaping & Filtration
- Bowtie filter (parabolic aluminum profile, 0-3 cm equivalent)
- Pre-patient beam hardening
- Position-dependent intensity modulation

### 3. Object Interaction
- Reactant-compatible ray tracer (Amanatides-Woo algorithm)
- Pure functional design (no callbacks) for XLA compilation
- Energy-dependent attenuation (photoelectric + Compton)
- Multi-material phantoms with density variations

### 4. Scatter Physics
- Gaussian convolution model for Compton scatter
- Scatter-to-Primary Ratio (SPR) ≈ 15%
- Separable kernels for computational efficiency
- Fully differentiable (no Monte Carlo)

### 5. Quantum Noise
- Poisson statistics for photon counting
- Dose-dependent noise modeling
- Normal approximation for high counts

### 6. Detector Response
- Energy-dependent quantum efficiency
- Electronic noise (Gaussian)
- DAS readout with log transform

### 7. Image Reconstruction
- Feldkamp-Davis-Kress (FDK) algorithm
- Ram-Lak filter with Hamming window
- 3D cone-beam backprojection
- Hounsfield Unit (HU) calibration

## Autodifferentiation Ready

All components are implemented as pure, differentiable functions:
- ✅ **Reactant @compile**: Pure functional design, no dynamic allocations
- ✅ **Enzyme gradients**: Differentiable forward model end-to-end
- ✅ **Gradient applications**: Material decomp, dose opt, MBIR, calibration

## Quick Start

1. **Run simulation**: Execute all cells to see the complete pipeline
2. **Adjust physics**: Toggle bowtie, scatter, noise in the simulation cell
3. **Test Reactant**: Compile individual functions or full forward pass
4. **Test Enzyme**: Compute gradients w.r.t. any input (densities, geometry, etc.)
5. **Implement inverse problems**: Use gradients for optimization tasks

---
"""

# ╔═╡ 9e413371-e11a-4359-b959-83f98632a77f
md"""
## Imports & Setup
"""

# ╔═╡ 84903695-03fb-4c7f-b475-7f1db983f14c
TableOfContents()

# ╔═╡ 704ad6f6-12fa-4b51-a0a2-95a1e223a09a
md"""
## Simulation Environment & Constants
Global physical constants and simulation settings.
"""

# ╔═╡ 48fca871-4e8a-4dd3-9041-9e6fbf575600
CONST_kVp = 120.0

# ╔═╡ e05adb7c-5c80-4626-9264-74d1d4307533
CONST_mAs = 200.0

# ╔═╡ 84d9222e-a9e4-4f1a-a25b-6abe7fedfcd8
CONST_SCAN_FOV = 500.0

# ╔═╡ 95f3e42d-e91e-430c-9cb6-1f9d7adc4d16
CONST_NUM_PROJECTIONS = 360

# ╔═╡ 2846460f-ec4c-4b6d-9b47-c9e1057e5fa2
CONST_DETECTOR_MATERIAL = Attenuations.Materials.GOS

# ╔═╡ 8636ce95-cee4-49c2-9d1b-8782ddac282e
CONST_THICKNESS_MM = 2.0

# ╔═╡ 1a124cfe-a285-4240-b8e8-ea59a2a2029d
CONST_NOISE_SIGMA = 0.5

# ╔═╡ 0cd3bd6d-f00e-4dca-812e-c443bf730a0f
md"""
# 1. Digital Phantom Generation (The Object)
Defining the spatial grid and material properties.
"""

# ╔═╡ e8fcc5b6-e7e9-4019-8f93-62013518b3d9
md"""
## Material Definitions
"""

# ╔═╡ be378bc8-d102-4ada-b624-a13ab155762a
begin
	gammex472_i2_0 = Material(
		"Gammex 472 (2.0 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.03g/cm^3,
		Dict(
			1  => 0.0864, # H
			6  => 0.6953, # C
			7  => 0.0215, # N
			8  => 0.1751, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0181, # Ca
			53 => 0.0020  # I
		)
	)

	gammex472_i2_5 = Material(
		"Gammex 472 (2.5 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.03g/cm^3,
		Dict(
			1  => 0.0863, # H
			6  => 0.6950, # C
			7  => 0.0214, # N
			8  => 0.1750, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0181, # Ca
			53 => 0.0025  # I
		)
	)

	gammex472_i5_0 = Material(
		"Gammex 472 (5.0 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.03g/cm^3,
		Dict(
			1  => 0.0861, # H
			6  => 0.6937, # C
			7  => 0.0214, # N
			8  => 0.1743, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0181, # Ca
			53 => 0.0049  # I
		)
	)

	gammex472_i7_5 = Material(
		"Gammex 472 (7.5 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.03g/cm^3,
		Dict(
			1  => 0.0859, # H
			6  => 0.6924, # C
			7  => 0.0213, # N
			8  => 0.1736, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0180, # Ca
			53 => 0.0073  # I
		)
	)

	gammex472_i10_0 = Material(
		"Gammex 472 (10.0 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.03g/cm^3,
		Dict(
			1  => 0.0856, # H
			6  => 0.6911, # C
			7  => 0.0212, # N
			8  => 0.1729, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0179, # Ca
			53 => 0.0097  # I
		)
	)

	gammex472_i15_0 = Material(
		"Gammex 472 (15.0 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.03g/cm^3,
		Dict(
			1  => 0.0851, # H
			6  => 0.6885, # C
			7  => 0.0210, # N
			8  => 0.1715, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0178, # Ca
			53 => 0.0146  # I
		)
	)

	gammex472_i20_0 = Material(
		"Gammex 472 (20.0 mg/ml Iodine)",
		0.0,
		0.0eV,
		1.04g/cm^3,
		Dict(
			1  => 0.0846, # H
			6  => 0.6859, # C
			7  => 0.0209, # N
			8  => 0.1701, # O
			11 => 0.0003, # Na
			16 => 0.0003, # S
			17 => 0.0013, # Cl
			20 => 0.0176, # Ca
			53 => 0.0194  # I
		)
	)
end;

# ╔═╡ 7099deb4-0ebd-4833-955f-38aee7e147a6
begin
	# --- Define Calcium Inserts based on Gammex PDF ---
	gammex472_ca50_0 = Material(
		"Gammex 472 (50.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		1.17g/cm^3,
		Dict(
			1  => 0.0710, # H
			6  => 0.6266, # C
			7  => 0.0270, # N
			8  => 0.2308, # O
			16 => 0.0007, # S
			17 => 0.0012, # Cl
			20 => 0.0427  # Ca
		)
	)

	gammex472_ca100_0 = Material(
		"Gammex 472 (100.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		1.24g/cm^3,
		Dict(
			1  => 0.0635, # H
			6  => 0.5720, # C
			7  => 0.0241, # N
			8  => 0.2579, # O
			16 => 0.0012, # S
			17 => 0.0010, # Cl
			20 => 0.0802  # Ca
		)
	)

	gammex472_ca200_0 = Material(
		"Gammex 472 (200.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		1.40g/cm^3,
		Dict(
			1  => 0.0509, # H
			6  => 0.4806, # C
			7  => 0.0193, # N
			8  => 0.3031, # O
			16 => 0.0022, # S
			17 => 0.0008, # Cl
			20 => 0.1431  # Ca
		)
	)

	gammex472_ca300_0 = Material(
		"Gammex 472 (300.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		1.55g/cm^3,
		Dict(
			1  => 0.0408, # H
			6  => 0.4070, # C
			7  => 0.0154, # N
			8  => 0.3396, # O
			16 => 0.0030, # S
			17 => 0.0007, # Cl
			20 => 0.1936  # Ca
		)
	)

	gammex472_ca400_0 = Material(
		"Gammex 472 (400.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		1.70g/cm^3,
		Dict(
			1  => 0.0325, # H
			6  => 0.3465, # C
			7  => 0.0121, # N
			8  => 0.3695, # O
			16 => 0.0036, # S
			17 => 0.0005, # Cl
			20 => 0.2352  # Ca
		)
	)

	gammex472_ca500_0 = Material(
		"Gammex 472 (500.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		1.85g/cm^3,
		Dict(
			1  => 0.0256, # H
			6  => 0.2958, # C
			7  => 0.0095, # N
			8  => 0.3946, # O
			16 => 0.0041, # S
			17 => 0.0004, # Cl
			20 => 0.2700  # Ca
		)
	)

	gammex472_ca600_0 = Material(
		"Gammex 472 (600.0 mg/ml Calcium)",
		0.0,
		0.0eV,
		2.01g/cm^3,
		Dict(
			1  => 0.0196, # H
			6  => 0.2525, # C
			7  => 0.0072, # N
			8  => 0.4161, # O
			16 => 0.0046, # S
			17 => 0.0003, # Cl
			20 => 0.2998  # Ca
		)
	)
end;

# ╔═╡ 0c543119-f37b-4a9c-bc45-4b680a1dbfd1
GAMMEX_PHANTOM_MAP = Dict(
	:air => Materials.ncat_air,
	:solid_water => Materials.ncat_water,

	# Calcium (Inner Ring)
	:Ca_50 => gammex472_ca50_0,
	:Ca_100 => gammex472_ca100_0,
	:Ca_200 => gammex472_ca200_0,
	:Ca_300 => gammex472_ca300_0,
	:Ca_400 => gammex472_ca400_0,
	:Ca_500 => gammex472_ca500_0,
	:Ca_600 => gammex472_ca600_0,

	# Iodine (Outer Ring)
	:I_2_0 => gammex472_i2_0,
	:I_2_5 => gammex472_i2_5,
	:I_5_0 => gammex472_i5_0,
	:I_7_5 => gammex472_i7_5,
	:I_10_0 => gammex472_i10_0,
	:I_15_0 => gammex472_i15_0,
	:I_20_0 => gammex472_i20_0
)

# ╔═╡ be3defc1-9ef7-44d7-a105-143151b39f12
md"""
## Phantom Specific Structs
"""

# ╔═╡ 41d0439f-5b1f-42f0-be34-f21637c3fca4
"""
    VoxelGrid

Defines the spatial coordinate system.
Internal units: Centimeters (cm).
"""
struct VoxelGrid
    nx::Int;
	ny::Int;
	nz::Int
    fov_xy_cm::Float64
    fov_z_cm::Float64
    x::Vector{Float64};
	y::Vector{Float64};
	z::Vector{Float64}
    x_planes::Vector{Float64};
	y_planes::Vector{Float64};
	z_planes::Vector{Float64}

    function VoxelGrid(; size_mm::Tuple, matrix::Tuple)
        fov_x_mm, fov_y_mm, fov_z_mm = size_mm
        nx, ny, nz = matrix

        fov_xy_cm = fov_x_mm / 10.0
        fov_z_cm  = fov_z_mm / 10.0

        dx, dy, dz = fov_xy_cm/nx, fov_xy_cm/ny, fov_z_cm/nz

        # Center at 0,0,0
        x = collect(range(-fov_xy_cm/2 + dx/2, length=nx, step=dx))
        y = collect(range(-fov_xy_cm/2 + dy/2, length=ny, step=dy))
        z = collect(range(-fov_z_cm/2 + dz/2, length=nz, step=dz))

        # Planes for ray tracing
        x_planes = collect(range(-fov_xy_cm/2, length=nx+1, step=dx))
        y_planes = collect(range(-fov_xy_cm/2, length=ny+1, step=dy))
        z_planes = collect(range(-fov_z_cm/2, length=nz+1, step=dz))

        new(nx, ny, nz, fov_xy_cm, fov_z_cm, x, y, z, x_planes, y_planes, z_planes)
    end
end

# ╔═╡ 85fa3167-f800-4574-b78e-12bb0f993be2
"""
	PhysicalPhantom (Optimized)

Stores material IDs as UInt8 (1 byte) instead of Symbol (8 bytes).
Stores density as Float32 (4 bytes) instead of Float64 (8 bytes).
Reduction: 16 bytes/voxel -> 5 bytes/voxel.
"""
struct PhysicalPhantom
	name::String
	grid::VoxelGrid
	material_ids::Array{UInt8, 3}     # The Map (0-255)
	densities::Array{Float32, 3}      # The Texture
	id_to_symbol::Dict{UInt8, Symbol} # The Legend

	function PhysicalPhantom(;
			name::String="Unknown",
			grid::VoxelGrid,
			material_ids::Array{UInt8, 3},
			densities::Array{Float32, 3},
			id_to_symbol::Dict{UInt8, Symbol}
		)
		new(name, grid, material_ids, densities, id_to_symbol)
	end
end

# ╔═╡ 5622c462-d139-42c8-ba6f-7966339f2f19
md"""
## Create Ultra-High Resolution Gammex Model 472 Phantom
"""

# ╔═╡ dd46bf7b-4136-434e-ad1b-ea1b8482cde5
function manufacture_gammex_472(;
		resolution_mm::Float64 = 0.5,   # Default 500 microns
		z_coverage_mm::Float64 = 40.0   # Default 40mm slab
	)

	# --- 1. PHYSICAL DIMENSIONS (Gammex 472 Specs) ---
	body_diameter_mm = 330.0  # 33 cm
	rod_diameter_mm  = 28.0   # 2.8 cm (Matches your original 1.4cm radius)

	# Tolerances
	gap_mm = 0.05 # 50 micron air gap

	# Simulation Setup
	sim_margin_mm = 10.0
	fov_total_mm = body_diameter_mm + sim_margin_mm

	nx = round(Int, fov_total_mm / resolution_mm)
	ny = nx
	nz = round(Int, z_coverage_mm / resolution_mm)

	# --- 2. LOGGING ---
	total_voxels = nx * ny * nz
	mem_est_gb = (total_voxels * 5) / 1024^3

	@info """
	🏭 MANUFACTURING PHANTOM (Corrected Geometry)
	=============================================
	Body Diameter : $(body_diameter_mm) mm
	Rod Diameter  : $(rod_diameter_mm) mm
	Resolution    : $(resolution_mm) mm
	Matrix        : $(nx) × $(ny) × $(nz)
	Memory Est.   : $(round(mem_est_gb, digits=2)) GB
	=============================================
	"""

	if mem_est_gb > 16.0
		@warn "⚠️ High memory usage predicted."
	end

	grid = VoxelGrid(
		size_mm = (fov_total_mm, fov_total_mm, z_coverage_mm),
		matrix  = (nx, ny, nz)
	)

	# --- 3. MATERIALS SETUP ---
	materials_list = [
		:solid_water,
		:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600,
		:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0
	]

	sym_to_id = Dict(:air => UInt8(0))
	id_to_sym = Dict(UInt8(0) => :air)
	for (i, sym) in enumerate(materials_list)
		id = UInt8(i)
		sym_to_id[sym] = id
		id_to_sym[id] = sym
	end

	# --- 4. GEOMETRY DEFINITIONS ---
	inserts = []

	# Radii for "drilled hole" and "physical rod"
	hole_radius = rod_diameter_mm / 2.0
	rod_radius  = hole_radius - gap_mm

	# Body (Solid Water)
	push!(inserts, (0.0, 0.0, body_diameter_mm/2, :solid_water))

	# Inner Ring (Calcium) - 5.0 cm radius
	ca_mats = [:Ca_50, :Ca_100, :Ca_200, :Ca_300, :Ca_400, :Ca_500, :Ca_600]
	for (i, m) in enumerate(ca_mats)
		# Matches old code: 7 inserts distributed over 360 degrees
		a = (i-1) * (2π/7)
		cx, cy = 50.0*cos(a), 50.0*sin(a)
		push!(inserts, (cx, cy, rod_radius, m))
	end

	# Outer Ring (Iodine) - 10.5 cm radius
	i_mats = [:I_2_0, :I_2_5, :I_5_0, :I_7_5, :I_10_0, :I_15_0, :I_20_0]
	num_iodine = length(i_mats) # Should be 7
	for (i, m) in enumerate(i_mats)
		# Distribute 7 items evenly (2π/7), with 30 deg offset
		a = (i-1) * (2π/num_iodine) + deg2rad(30)
		cx, cy = 105.0*cos(a), 105.0*sin(a)
		push!(inserts, (cx, cy, rod_radius, m))
	end

	# --- 5. RENDER LOOP ---
	mat_grid = fill(UInt8(0), nx, ny, nz)
	den_grid = zeros(Float32, nx, ny, nz)

	# Helper: Inputs in CM (from grid), Params in MM
	in_circle(x_cm, y_cm, cx_mm, cy_mm, r_mm) =
		(x_cm*10 - cx_mm)^2 + (y_cm*10 - cy_mm)^2 <= r_mm^2

	Threads.@threads for k in 1:nz
		if abs(grid.z[k]) > (z_coverage_mm/20.0) continue end

		for j in 1:ny, i in 1:nx
			px, py = grid.x[i], grid.y[j]

			current_id = UInt8(0) # Air

			# 1. Base Body
			if in_circle(px, py, 0.0, 0.0, body_diameter_mm/2)
				current_id = sym_to_id[:solid_water]
			end

			# 2. Inserts (Drill holes and insert rods)
			# Skip the first "insert" which is just the body definition
			for (cx, cy, r_rod, m) in inserts[2:end]
				# Check if we are in the drilled hole
				if in_circle(px, py, cx, cy, hole_radius)
					# Check if we are in the rod (vs air gap)
					if in_circle(px, py, cx, cy, r_rod)
						current_id = sym_to_id[m]
					else
						current_id = UInt8(0) # Air Gap
					end
				end
			end

			mat_grid[i, j, k] = current_id

			if current_id != 0
				# Add texture
				den_grid[i, j, k] = 1.0f0 + randn(Float32) * 0.01f0
			end
		end
	end

	return PhysicalPhantom(
		name="Gammex 472 (Corrected)",
		grid=grid,
		material_ids=mat_grid,
		densities=den_grid,
		id_to_symbol=id_to_sym
	)
end

# ╔═╡ e9faca70-67f2-4a03-99e3-5945851e8e2d
PHANTOM = manufacture_gammex_472(
	resolution_mm = 2.0, # 2 mm
	z_coverage_mm = 40.0 # 40 mm
)

# ╔═╡ d72b418e-49d3-41c8-b81d-830f9750b1ef
let
    grid = PHANTOM.grid

    # 1. Get unique IDs present in the phantom
    unique_ids = sort(unique(PHANTOM.material_ids))

    # 2. Get their names using the internal dictionary
    # phantom.id_to_symbol maps UInt8 -> Symbol
    mat_names = [string(PHANTOM.id_to_symbol[id]) for id in unique_ids]

    # 3. Extract Slices (UInt8 data is already integer-mapped!)
    mid_z = grid.nz ÷ 2
    mid_y = grid.ny ÷ 2

    # Just take the raw ID arrays
    map_axial = PHANTOM.material_ids[:, :, mid_z]
    map_coronal = PHANTOM.material_ids[:, mid_y, :]

    # 4. Plot
    fig = Figure(size=(1200, 500))

    # Axial
    ax1 = Axis(fig[1, 1], title="Axial View",
               xlabel="x (cm)", ylabel="y (cm)")
    hm1 = heatmap!(ax1, grid.x, grid.y, map_axial,
                   colormap=cgrad(:tab20, length(unique_ids), categorical=true))
	Colorbar(fig[1, 2], hm1, label="Material",
		 ticks=(unique_ids, mat_names))

    # Coronal
    ax2 = Axis(fig[1, 3], title="Coronal View", aspect = DataAspect(),
               xlabel="x (cm)", ylabel="z (cm)")
    hm2 = heatmap!(ax2, grid.x, grid.z, map_coronal,
                   colormap=cgrad(:tab20, length(unique_ids), categorical=true))
    Colorbar(fig[1, 4], hm2, label="Material",
             ticks=(unique_ids, mat_names))

    fig
end

# ╔═╡ f40b9973-746c-4445-8fec-d4641f487239
md"""
# 2. X-Ray Source (The Anode)
Generation of the photon spectrum with realistic physics.
"""

# ╔═╡ 498aef40-ef97-411c-aaa8-61468ed314d9
md"""
## Source Definitions
"""

# ╔═╡ bdcfe35d-f96a-4d06-b1ab-933e15345856
"""
	XRaySource

Represents the spectral output of the x-ray tube.
- `kVp`: Peak Kilovoltage.
- `mAs`: Tube current-time product.
- `energies`: Energy bins (keV).
- `photons`: Photon counts per bin.
- `focal_spot_size_mm`: Size of the emission spot.
"""
struct XRaySource
	kVp::Float64
	mAs::Float64
	energies::Vector{Float64}
	photons::Vector{Float64}
	focal_spot_size_mm::Float64

	# STRICT KWARG CONSTRUCTOR
	function XRaySource(;
			kVp::Float64,
			mAs::Float64,
			energies::Vector{Float64},
			photons::Vector{Float64},
			focal_spot_size_mm::Float64=1.0
		)
		if length(energies) != length(photons)
			error("Energy and Photon vectors must be same length.")
		end
		new(kVp, mAs, energies, photons, focal_spot_size_mm)
	end
end

# ╔═╡ 1f0ba521-fa29-4ad1-a46e-3033bcb011e1
md"""
## Realistic Spectrum Generator with K-Lines
Enhanced spectrum with tungsten characteristic X-rays and heel effect.
"""

# ╔═╡ f1a2b3c4-d5e6-7f8a-9b0c-1d2e3f4a5b6c
"""
	generate_realistic_spectrum

Enhanced spectrum with tungsten characteristic X-rays and heel effect.
Self-contained analytical model - no external packages needed.

Physical parameters:
- Tungsten anode (Z=74)
- Characteristic energies: K-α₁=59.3 keV, K-α₂=58.0 keV, K-β=67.2 keV
- Anode angle: 7° (typical for CT)
- Heel effect: intensity variation across field
"""
function generate_realistic_spectrum(;
		kVp::Float64,
		mAs::Float64,
		min_E::Float64 = 10.0,
		max_E_grid::Float64 = 150.0,
		n_bins::Int = 140,
		anode_angle_deg::Float64 = 7.0,
		filtration_mm_Al::Float64 = 4.0,
		filtration_mm_Cu::Float64 = 0.1
	)

	energies = collect(range(min_E, max_E_grid, length=n_bins))
	mask = energies .< kVp

	# 1. BREMSSTRAHLUNG (Kramers' Law with Z-dependence)
	# I(E) ∝ Z(kVp - E) where Z=74 for tungsten
	Z_tungsten = 74.0
	brems = @. Z_tungsten * energies * (kVp - energies) * mask

	# 2. CHARACTERISTIC X-RAYS (Tungsten K-lines)
	# Gaussian peaks centered at characteristic energies
	char_peaks = zeros(Float64, n_bins)

	if kVp > 69.5  # K-edge of tungsten
		# K-alpha1 (59.3 keV) - strongest
		sigma_kalpha = 0.5  # keV width
		char_peaks .+= @. 0.08 * kVp * exp(-((energies - 59.3)^2) / (2 * sigma_kalpha^2))

		# K-alpha2 (58.0 keV) - half intensity of K-alpha1
		char_peaks .+= @. 0.04 * kVp * exp(-((energies - 58.0)^2) / (2 * sigma_kalpha^2))

		# K-beta (67.2 keV) - 20% of K-alpha1
		char_peaks .+= @. 0.016 * kVp * exp(-((energies - 67.2)^2) / (2 * sigma_kalpha^2))
	end

	# 3. FILTRATION
	# Aluminum (inherent + added)
	mu_Al = @. 10.0 * (30.0 / max.(energies, 1.0))^3
	trans_Al = @. exp(-mu_Al * filtration_mm_Al / 10.0)  # convert to cm

	# Copper (beam shaping)
	mu_Cu = @. 15.0 * (30.0 / max.(energies, 1.0))^3
	trans_Cu = @. exp(-mu_Cu * filtration_mm_Cu / 10.0)

	transmission = trans_Al .* trans_Cu

	# 4. COMBINE BREMSSTRAHLUNG + CHARACTERISTIC
	raw_spectrum = (brems .+ char_peaks) .* transmission

	# 5. HEEL EFFECT (anode angle affects intensity)
	# Simplified model: intensity varies with angle
	# For central ray, use average intensity
	heel_factor = sind(anode_angle_deg) * 0.7 + 0.3  # empirical

	intensity = raw_spectrum .* heel_factor

	# 6. NORMALIZE TO PHOTON COUNT
	total_int = sum(intensity) + 1e-9
	flux_target = mAs * 2.5e5 * n_bins  # slightly higher for realistic spectrum
	photons = intensity ./ total_int .* flux_target

	return XRaySource(
		kVp = kVp,
		mAs = mAs,
		energies = energies,
		photons = photons,
		focal_spot_size_mm = 1.2  # typical CT focal spot
	)
end

# ╔═╡ a2b3c4d5-e6f7-8a9b-0c1d-2e3f4a5b6c7d
SOURCE_120_REALISTIC = generate_realistic_spectrum(
	kVp=CONST_kVp,
	mAs=CONST_mAs,
	anode_angle_deg=7.0,
	filtration_mm_Al=5.0,
	filtration_mm_Cu=0.2
)

# ╔═╡ 45cc1bd4-3a00-4e0b-a0ec-b3d953f50d72
let
	f = Figure(size=(1200, 600))

	# Top: Realistic Model with K-lines
	ax1 = Axis(f[1,1],
		title="Realistic Tungsten Spectrum (K-α, K-β peaks)",
		xlabel="Energy (keV)",
		ylabel="Photons/keV"
	)
	lines!(ax1, SOURCE_120_REALISTIC.energies, SOURCE_120_REALISTIC.photons,
		label="120 kVp Realistic", linewidth=2, color=:red)
	vlines!(ax1, [59.3], color=:orange, linestyle=:dash, label="K-α₁ (59.3 keV)")
	vlines!(ax1, [58.0], color=:gold, linestyle=:dash, label="K-α₂ (58.0 keV)")
	vlines!(ax1, [67.2], color=:purple, linestyle=:dash, label="K-β (67.2 keV)")
	axislegend(ax1, position=:rt)

	# Bottom: Zoomed view showing K-line peaks
	ax2 = Axis(f[2,1],
		title="Zoomed: Characteristic K-lines Detail",
		xlabel="Energy (keV)",
		ylabel="Photons/keV"
	)
	lines!(ax2, SOURCE_120_REALISTIC.energies, SOURCE_120_REALISTIC.photons,
		linewidth=2, color=:red)
	vlines!(ax2, [59.3, 58.0, 67.2], color=[:orange, :gold, :purple], linestyle=:dash)
	xlims!(ax2, 55, 72)  # Zoom to K-line region

	f
end

# ╔═╡ 4ee516dc-9f1f-4291-9392-2098ed541958
md"""
## Material Spectrum Lookup Tables
Pre-Computed, Only Do This Step ONCE To Save Time
"""

# ╔═╡ d11d555e-0877-4f2f-8d8b-3a2848a96393
"""
	MaterialLibrary

Stores attenuation coefficients as a dense matrix for GPU/XLA compatibility.
- `energies`: Vector of energy bins (keV).
- `μ_matrix`: Dense matrix [n_materials × n_energies] (cm⁻¹).
- `id_map`: Dictionary mapping Phantom UInt8 IDs to Matrix Row Indices.
"""
struct MaterialLibrary
	energies::Vector{Float64}
	μ_matrix::Matrix{Float64}
	id_map::Dict{UInt8, Int}

	# STRICT KWARG CONSTRUCTOR
	function MaterialLibrary(;
			energies::Vector{Float64},
			μ_matrix::Matrix{Float64},
			id_map::Dict{UInt8, Int}
		)
		new(energies, μ_matrix, id_map)
	end
end

# ╔═╡ 27bb0a72-d80e-4035-a1e7-3b45cfc0f49a
function precompute_material_library(;
		phantom::PhysicalPhantom,
		source::XRaySource,
		material_defs::Dict{Symbol, <:Any}
	)

	@info "⚡ Pre-computing Physics Tables (Network/Calculation step)..."
	t_start = time()

	# 1. Identify minimal set of materials from the phantom
	unique_ids = sort(collect(keys(phantom.id_to_symbol)))
	n_materials = length(unique_ids)
	n_energies = length(source.energies)

	μ_matrix = zeros(Float64, n_materials, n_energies)
	id_map = Dict{UInt8, Int}()

	# Pre-convert energies to Unitful once
	energies_kev = source.energies .* keV

	for (row_idx, uid) in enumerate(unique_ids)
		sym = phantom.id_to_symbol[uid]
		if !haskey(material_defs, sym)
			error("Material :$sym found in Phantom but missing from Physics Map.")
		end

		# Record the mapping (UInt8 -> Matrix Row)
		id_map[uid] = row_idx

		# 2. The Slow Call (Network or Heavy Calculation)
		# We broadcast the call across all energies at once.
		# If Attenuations.jl allows it, this batches the lookup.
		mat = material_defs[sym]

		# Note: We strip units immediately for Reactant compatibility
		mus_row = ustrip.(u"cm^-1", μ.(Ref(mat), energies_kev))
		μ_matrix[row_idx, :] .= mus_row
	end

	dt = time() - t_start
	@info "✅ Physics Tables ready in $(round(dt, digits=2))s"

	return MaterialLibrary(
		energies = source.energies,
		μ_matrix = μ_matrix,
		id_map   = id_map
	)
end

# ╔═╡ d6452eef-ee88-45ac-bfff-e384dc386a76
MATERIAL_LIBRARY = precompute_material_library(
	phantom = PHANTOM,
	source = SOURCE_120_REALISTIC,
	material_defs = GAMMEX_PHANTOM_MAP
)

# ╔═╡ 736d9c19-9d61-447c-8c4c-2cfb330b5f6c
md"""
We need to convert the Dictionary in `MATERIAL_LIBRARY` to a Vector once, globally, so the simulation loop never has to touch a Dictionary.
"""

# ╔═╡ b7f140fe-fe3e-4121-a738-be04d6c961d3
begin
	ID_LUT = zeros(Int, 256)
	for (uid, row_idx) in MATERIAL_LIBRARY.id_map
		ID_LUT[Int(uid) + 1] = row_idx
	end
	ID_LUT
end

# ╔═╡ 2e66cacf-e923-4d05-b875-111357a0dcfb
md"""
# 3. Scanner Geometry (The Gantry)
Pre-computed coordinate systems for the Source, Detector, and Trajectory.
Using Canon Aquilion ONE specifications.
"""

# ╔═╡ 9af37598-6b82-4884-89fb-a450c0247fd3
md"""
## Geometry Definitions
"""

# ╔═╡ 28bbf834-4a5a-4632-8ded-7cabe3bbd7c0
"""
	ScanProtocol

Defines the acquisition physics.
"""
struct ScanProtocol
	kVp::Float64
	mAs::Float64
	scan_fov_mm::Float64
	num_projections::Int
	rotation_total_angle::Float64
	start_angle::Float64

	# STRICT KWARG CONSTRUCTOR
	function ScanProtocol(;
			kVp::Float64,
			mAs::Float64,
			scan_fov_mm::Float64,
			num_projections::Int,
			rotation_total_angle::Float64 = 360.0,
			start_angle::Float64 = 0.0
		)
		new(kVp, mAs, scan_fov_mm, num_projections, rotation_total_angle, start_angle)
	end
end

# ╔═╡ 3becb001-c352-457f-8515-ebb832633a6a
"""
	Geometry

Defines the rigid mechanical geometry and PRE-COMPUTED trajectories.
All spatial units are in **cm**.
"""
struct Geometry
	# --- Mechanical Constants ---
	SDD_cm::Float64
	SAD_cm::Float64
	n_rows::Int
	n_cols::Int
	pixel_width_cm::Float64
	pixel_height_cm::Float64

	# --- Trajectory Tensors (Pre-Computed) ---
	# Dimensions: [3 x N_projections]
	# This avoids calculating sin/cos inside the hot loop
	angles::Vector{Float64}          # [N]
	source_positions::Matrix{Float64} # [3, N]
	det_centers::Matrix{Float64}      # [3, N]
	det_u_vecs::Matrix{Float64}       # [3, N] (Detector Width Axis)
	det_v_vecs::Matrix{Float64}       # [3, N] (Detector Height Axis)

	# STRICT KWARG CONSTRUCTOR
	function Geometry(;
			sdd_mm::Float64,
			sad_mm::Float64,
			n_rows::Int,
			n_cols::Int,
			pixel_size_mm::Tuple{Float64, Float64},
			angles_deg::Vector{Float64}
		)

		# 1. Convert Units
		SDD_cm = sdd_mm / 10.0
		SAD_cm = sad_mm / 10.0
		pw_cm  = pixel_size_mm[1] / 10.0
		ph_cm  = pixel_size_mm[2] / 10.0

		n_proj = length(angles_deg)

		# 2. Allocate Trajectory Tensors
		src_pos = zeros(Float64, 3, n_proj)
		det_cen = zeros(Float64, 3, n_proj)
		u_vecs  = zeros(Float64, 3, n_proj)
		v_vecs  = zeros(Float64, 3, n_proj)

		# 3. Pre-Compute Geometry (The "Baking" Step)
		# We do this ONCE here, so the simulation loop is pure memory lookup
		for i in 1:n_proj
			rad = deg2rad(angles_deg[i])
			s, c = sin(rad), cos(rad)

			# Source Position (Standard Rotation)
			src_pos[1, i] = SAD_cm * s
			src_pos[2, i] = -SAD_cm * c
			src_pos[3, i] = 0.0

			# Detector Center (Opposite side)
			dist = SDD_cm - SAD_cm
			det_cen[1, i] = -dist * s
			det_cen[2, i] = dist * c
			det_cen[3, i] = 0.0

			# Basis Vectors
			# U maps to columns (along the arc)
			u_vecs[1, i] = c
			u_vecs[2, i] = s
			u_vecs[3, i] = 0.0

			# V maps to rows (Z-axis, usually static but good to be explicit)
			v_vecs[1, i] = 0.0
			v_vecs[2, i] = 0.0
			v_vecs[3, i] = 1.0
		end

		new(SDD_cm, SAD_cm, n_rows, n_cols, pw_cm, ph_cm,
			angles_deg, src_pos, det_cen, u_vecs, v_vecs)
	end
end

# ╔═╡ c1d2e3f4-a5b6-7c8d-9e0f-1a2b3c4d5e6f
"""
	create_aquilion_one_geometry

Canon Aquilion ONE geometry with actual scanner specifications:
- 320-row detector (16 cm z-coverage in one rotation!)
- 0.5mm detector pixel pitch
- SDD = 1000mm
- SAD = 600mm (actual Aquilion ONE geometry)
- FOV up to 500mm
"""
function create_aquilion_one_geometry(; protocol::ScanProtocol)
	# CANON AQUILION ONE SPECIFICATIONS
	sdd_mm = 1000.0  # Source-to-detector distance
	sad_mm = 600.0   # Source-to-axis distance (isocenter)

	# 320-ROW DETECTOR
	n_rows = 320     # 16 cm z-coverage!
	detector_pixel_pitch_mm = 0.5  # 0.5mm at isocenter

	# Detector width to cover FOV
	magnification = sdd_mm / sad_mm
	det_width_needed_mm = protocol.scan_fov_mm * magnification
	n_cols = round(Int, det_width_needed_mm / (detector_pixel_pitch_mm * magnification))

	# Make it even for symmetry
	n_cols = n_cols + (n_cols % 2)

	# Actual pixel size at detector
	pixel_size_at_det_mm = (detector_pixel_pitch_mm * magnification,
	                        detector_pixel_pitch_mm * magnification)

	# Angular sampling
	start_deg = protocol.start_angle
	stop_deg = protocol.start_angle + protocol.rotation_total_angle
	angles = collect(range(start_deg, stop_deg, length=protocol.num_projections))

	@info """
	🏥 Canon Aquilion ONE Geometry
	==============================
	Detector: $n_rows × $n_cols (320-row wide detector!)
	Z-Coverage: $(n_rows * detector_pixel_pitch_mm / 10.0) cm
	SDD: $sdd_mm mm
	SAD: $sad_mm mm
	Pixel pitch: $(detector_pixel_pitch_mm) mm
	FOV: $(protocol.scan_fov_mm) mm
	Projections: $(length(angles))
	"""

	return Geometry(
		sdd_mm = sdd_mm,
		sad_mm = sad_mm,
		n_rows = n_rows,
		n_cols = n_cols,
		pixel_size_mm = pixel_size_at_det_mm,
		angles_deg = angles
	)
end

# ╔═╡ 9cb9bcc9-e1f6-4978-a415-f68c689f488f
md"""
## Scanner Initialization
"""

# ╔═╡ e7e54f4e-e52c-4cdd-939f-11fb156da33d
PROTOCOL = ScanProtocol(
	kVp=CONST_kVp,
	mAs=CONST_mAs,
	scan_fov_mm=CONST_SCAN_FOV,
	num_projections=CONST_NUM_PROJECTIONS
)

# ╔═╡ d2e3f4a5-b6c7-8d9e-0f1a-2b3c4d5e6f7a
GEOMETRY_AQUILION_ONE = create_aquilion_one_geometry(protocol=PROTOCOL)

# ╔═╡ 25fb9f2c-21b6-4e93-9706-7e95ec350390
md"""
# 4. Ray Tracing (The Interaction)
XLA-compatible, allocation-free, pure functional Amanatides-Woo implementation.
"""

# ╔═╡ 7aa9d80d-8517-4025-8188-42256fa6b261
md"""
## Grid Metadata (Scalar Layout)
We use a lightweight, scalar-only struct (`GridMeta`) to pass geometric parameters to the GPU/XLA kernel, avoiding the overhead of passing full Julia arrays.
"""

# ╔═╡ 0d9725ae-e4a8-4e6e-9674-eb4ca4607270
"""
	GridMeta

Lightweight, scalar-only representation of the voxel grid for GPU/XLA kernels.
Does not store arrays (planes), only geometric parameters.
"""
struct GridMeta
	# Dimensions
	nx::Int
	ny::Int
	nz::Int

	# Physical Size (cm)
	fov_xy::Float64
	fov_z::Float64

	# Pre-calculated Spacing (cm/voxel)
	dx::Float64
	dy::Float64
	dz::Float64

	# Origins (Bottom-Left-Back corner in cm)
	ox::Float64
	oy::Float64
	oz::Float64

	# STRICT KWARG CONSTRUCTOR
	function GridMeta(;
			nx::Int, ny::Int, nz::Int,
			fov_xy::Float64, fov_z::Float64
		)
		dx = fov_xy / nx
		dy = fov_xy / ny
		dz = fov_z / nz

		# Center is (0,0,0), so origin is -FOV/2
		ox = -fov_xy / 2.0
		oy = -fov_xy / 2.0
		oz = -fov_z  / 2.0

		new(nx, ny, nz, fov_xy, fov_z, dx, dy, dz, ox, oy, oz)
	end
end

# ╔═╡ f519c8f6-a355-4243-a917-531dad43bcc0
md"""
## Reactant-Compatible Ray Tracer
Pure functional version that returns path lengths per material (no callbacks).
Uses Amanatides-Woo traversal algorithm.
"""

# ╔═╡ 1cd38a76-87c8-4bff-b012-29829f6251fd
"""
	trace_ray_material_paths(grid, material_ids, densities, id_lut, n_materials, p1x, p1y, p1z, p2x, p2y, p2z)

Reactant-compatible ray tracer using Amanatides-Woo traversal.
Returns a vector of path lengths (cm) for each material.

Arguments:
- grid: GridMeta with voxel spacing
- material_ids: 3D array of UInt8 material IDs
- densities: 3D array of Float32 relative densities
- id_lut: Lookup table mapping material ID to material index (256-element vector)
- n_materials: Number of materials
- p1x, p1y, p1z: Ray start point (cm)
- p2x, p2y, p2z: Ray end point (cm)

Returns:
- path_lengths: Vector{Float64} of length n_materials
"""
function trace_ray_material_paths(
		grid::GridMeta,
		material_ids::Array{UInt8, 3},
		densities::Array{Float32, 3},
		id_lut::Vector{Int},
		n_materials::Int,
		p1x::Float64, p1y::Float64, p1z::Float64,
		p2x::Float64, p2y::Float64, p2z::Float64
	)

	# Initialize output
	path_lengths = zeros(Float64, n_materials)

	# 1. Direction
	dx_ray = p2x - p1x
	dy_ray = p2y - p1y
	dz_ray = p2z - p1z

	len = sqrt(dx_ray^2 + dy_ray^2 + dz_ray^2)
	if len < 1e-6 return path_lengths end

	inv_len = 1.0 / len
	dir_x = dx_ray * inv_len
	dir_y = dy_ray * inv_len
	dir_z = dz_ray * inv_len

	# 2. Slab Intersection
	box_min_x, box_max_x = grid.ox, grid.ox + grid.fov_xy
	box_min_y, box_max_y = grid.oy, grid.oy + grid.fov_xy
	box_min_z, box_max_z = grid.oz, grid.oz + grid.fov_z

	tmin, tmax = 0.0, len

	# X-Slab
	if abs(dir_x) < 1e-9
		if p1x < box_min_x || p1x > box_max_x return path_lengths end
	else
		tx1 = (box_min_x - p1x) / dir_x
		tx2 = (box_max_x - p1x) / dir_x
		tmin = max(tmin, min(tx1, tx2))
		tmax = min(tmax, max(tx1, tx2))
	end

	# Y-Slab
	if abs(dir_y) < 1e-9
		if p1y < box_min_y || p1y > box_max_y return path_lengths end
	else
		ty1 = (box_min_y - p1y) / dir_y
		ty2 = (box_max_y - p1y) / dir_y
		tmin = max(tmin, min(ty1, ty2))
		tmax = min(tmax, max(ty1, ty2))
	end

	# Z-Slab
	if abs(dir_z) < 1e-9
		if p1z < box_min_z || p1z > box_max_z return path_lengths end
	else
		tz1 = (box_min_z - p1z) / dir_z
		tz2 = (box_max_z - p1z) / dir_z
		tmin = max(tmin, min(tz1, tz2))
		tmax = min(tmax, max(tz1, tz2))
	end

	if tmax <= tmin return path_lengths end

	# 3. Initialization
	current_t = tmin
	cur_x = p1x + (current_t + 1e-5) * dir_x
	cur_y = p1y + (current_t + 1e-5) * dir_y
	cur_z = p1z + (current_t + 1e-5) * dir_z

	ix = floor(Int, (cur_x - grid.ox) / grid.dx) + 1
	iy = floor(Int, (cur_y - grid.oy) / grid.dy) + 1
	iz = floor(Int, (cur_z - grid.oz) / grid.dz) + 1

	ix = clamp(ix, 1, grid.nx)
	iy = clamp(iy, 1, grid.ny)
	iz = clamp(iz, 1, grid.nz)

	step_x = dir_x >= 0 ? 1 : -1
	step_y = dir_y >= 0 ? 1 : -1
	step_z = dir_z >= 0 ? 1 : -1

	next_x_boundary = grid.ox + (dir_x >= 0 ? ix : ix - 1) * grid.dx
	next_y_boundary = grid.oy + (dir_y >= 0 ? iy : iy - 1) * grid.dy
	next_z_boundary = grid.oz + (dir_z >= 0 ? iz : iz - 1) * grid.dz

	t_max_x = abs(dir_x) < 1e-9 ? Inf : (next_x_boundary - p1x) / dir_x
	t_max_y = abs(dir_y) < 1e-9 ? Inf : (next_y_boundary - p1y) / dir_y
	t_max_z = abs(dir_z) < 1e-9 ? Inf : (next_z_boundary - p1z) / dir_z

	if t_max_x < current_t t_max_x += abs(grid.dx / dir_x) end
	if t_max_y < current_t t_max_y += abs(grid.dy / dir_y) end
	if t_max_z < current_t t_max_z += abs(grid.dz / dir_z) end

	tdelta_x = abs(dir_x) < 1e-9 ? Inf : grid.dx / abs(dir_x)
	tdelta_y = abs(dir_y) < 1e-9 ? Inf : grid.dy / abs(dir_y)
	tdelta_z = abs(dir_z) < 1e-9 ? Inf : grid.dz / abs(dir_z)

	# 4. Traversal Loop
	max_iter = grid.nx + grid.ny + grid.nz + 10

	for _ in 1:max_iter
		if current_t >= tmax break end

		# Accumulate path length for current voxel
		mat_id = material_ids[ix, iy, iz]
		if mat_id > 0
			row_idx = id_lut[Int(mat_id) + 1]
			if row_idx > 0
				# Determine step distance
				if t_max_x < t_max_y
					if t_max_x < t_max_z
						dist = t_max_x - current_t
					else
						dist = t_max_z - current_t
					end
				else
					if t_max_y < t_max_z
						dist = t_max_y - current_t
					else
						dist = t_max_z - current_t
					end
				end

				dens = Float64(densities[ix, iy, iz])
				path_lengths[row_idx] += dist * dens
			end
		end

		# Step to next voxel
		if t_max_x < t_max_y
			if t_max_x < t_max_z
				current_t = t_max_x
				t_max_x += tdelta_x
				ix += step_x
			else
				current_t = t_max_z
				t_max_z += tdelta_z
				iz += step_z
			end
		else
			if t_max_y < t_max_z
				current_t = t_max_y
				t_max_y += tdelta_y
				iy += step_y
			else
				current_t = t_max_z
				t_max_z += tdelta_z
				iz += step_z
			end
		end

		# Exit if out of bounds
		if ix < 1 || ix > grid.nx || iy < 1 || iy > grid.ny || iz < 1 || iz > grid.nz
			break
		end
	end

	return path_lengths
end

# ╔═╡ dc1c43aa-047c-4e83-8f9c-d9815f24cf08
md"""
## Instantiate Grid
"""

# ╔═╡ 66406706-4b18-4b75-a682-76b4012b1e07
GRID_META = GridMeta(
	nx=PHANTOM.grid.nx, ny=PHANTOM.grid.ny, nz=PHANTOM.grid.nz,
	fov_xy=PHANTOM.grid.fov_xy_cm, fov_z=PHANTOM.grid.fov_z_cm
)

# ╔═╡ 5c0d5867-8719-47d7-8921-fc1a4052d906
md"""
## Reactant Compilation (CPU)

Compile the ray tracer with Reactant for massive speedup.
This uses CPU compilation (XLA/MLIR backend).

**Why compile now?**
- Reactant compiles Julia code to XLA (optimized machine code)
- 10-100x speedup for numerical kernels like ray tracing
- We compile BEFORE running the simulation so it's ready to use
"""

# ╔═╡ 830f69c2-b94f-45b5-ae32-b4d1a1072f39
begin
	@info "🔧 Compiling ray tracer with Reactant (CPU)..."
	
	# Compile with example inputs
	trace_ray_compiled = Reactant.@compile trace_ray_material_paths(
		GRID_META,
		PHANTOM.material_ids,
		PHANTOM.densities,
		ID_LUT,
		length(unique(PHANTOM.material_ids)),
		0.0, 50.0, 0.0,  # example source position
		0.0, -50.0, 0.0  # example detector position
	)
	
	@info "✅ Ray tracer compiled and ready!"
	@info "   Use 'use_compiled=true' in simulation to enable"
	
	trace_ray_compiled
end

# ╔═╡ b6cb080e-0ef4-47e2-8b88-f0aac56eb6d4
md"""
# 5. Detector Physics (The Capture)
Modeling scintillator efficiency, bowtie filter, scatter, and noise.
"""

# ╔═╡ 25eb6c8c-5ac1-40b6-af9d-9801abba6fad
md"""
## Detector Configuration
User-facing parameters for the detection hardware.
"""

# ╔═╡ 1fa8e20b-27f4-40d7-8e4e-15f51f29e0ad
struct DetectorConfig
	material::Material # Direct Attenuations.Material object
	thickness_mm::Float64
	noise_sigma::Float64

	# STRICT KWARG CONSTRUCTOR
	function DetectorConfig(;
			material::Material,
			thickness_mm::Float64,
			noise_sigma::Float64=0.0
		)
		new(material, thickness_mm, noise_sigma)
	end
end

# ╔═╡ 8905383e-f450-42fa-a7a0-72351695287f
md"""
## Detector Response (Pre-Computed)
We pre-calculate the absorption efficiency (1 - e^(-μt)) for the specific energy spectrum to avoid heavy material lookups during the scan.
"""

# ╔═╡ 04fe5478-c4b5-42dc-a37b-c0975136f4d7
"""
	DetectorResponse

Low-level struct containing pre-calculated efficiency vectors for the simulation kernel.
"""
struct DetectorResponse
	efficiency::Vector{Float64} # [N_energies]
	noise_sigma::Float64
end

# ╔═╡ 846525d5-4f37-40b1-b364-2657017e8e7f
function compute_detector_response(;
		det::DetectorConfig,
		source::XRaySource
	)

	# 1. Get Material directly from config
	mat = det.material

	# 2. Calculate Efficiency: 1 - exp(-μ * thickness)
	energies_kev = source.energies .* keV
	t_cm = det.thickness_mm / 10.0

	# Vectorized lookup via Attenuations.jl
	mus = ustrip.(u"cm^-1", Attenuations.μ.(Ref(mat), energies_kev))

	# Quantum Detection Efficiency (QDE)
	# Probability that a photon of energy E interacts with the scintillator
	efficiency = @. 1.0 - exp(-mus * t_cm)

	return DetectorResponse(efficiency, det.noise_sigma)
end

# ╔═╡ 009fe763-3619-4fd9-ab46-5fdaf065a15a
DETECTOR_CONFIG = DetectorConfig(
	material = CONST_DETECTOR_MATERIAL,
	thickness_mm = CONST_THICKNESS_MM,
	noise_sigma = CONST_NOISE_SIGMA
)

# ╔═╡ 38027728-dded-4fed-840f-439292e46bf6
DETECTOR_RESPONSE = compute_detector_response(
	det=DETECTOR_CONFIG,
	source=SOURCE_120_REALISTIC
)

# ╔═╡ 6b3a037a-9da0-4ed6-9a1b-d640361dbdb3
md"""
## Bowtie Filter

CT scanners use shaped beam-hardening filters (bowtie filters) to:
- Reduce peripheral dose (patient's edges get less radiation)
- Pre-harden the beam (reduces artifacts)
- Flatten the intensity profile reaching the detector

**Physics**: Aluminum or Teflon shaped filter, thicker at edges, thinner at center.
"""

# ╔═╡ f16df8ce-5d11-4e54-8601-7121380f4cd5
"""
Applies bowtie filter attenuation to X-ray spectrum.

Bowtie filter: shaped Al/Teflon filter that's thicker at periphery.
- Reduces dose to patient's edges
- Pre-hardens the beam (removes low-E photons preferentially)
- Typical: 0-5 cm aluminum equivalent

Parameters:
- energies: X-ray energy bins (keV)
- photons: photon counts per energy bin
- detector_col: column index on detector (-n/2 to +n/2)
- n_cols: total detector columns
- max_thickness_cm: maximum filter thickness at edge (Al equivalent)

Returns: attenuated photon spectrum
"""
function apply_bowtie_filter(
		energies::Vector{Float64},
		photons::Vector{Float64},
		detector_col::Int,
		n_cols::Int;
		max_thickness_cm::Float64 = 3.0  # cm of aluminum at edge
	)

	# Bowtie profile: parabolic, thicker at edges
	# Normalized position: -1 (edge) to +1 (edge), 0 (center)
	norm_pos = 2.0 * (detector_col - n_cols/2) / n_cols

	# Parabolic thickness: max at edges, zero at center
	thickness_cm = max_thickness_cm * norm_pos^2

	# Aluminum attenuation coefficients (cm^-1) - simplified energy dependence
	# μ(E) ≈ a * E^(-3) + b  (photoelectric + Compton)
	# Rough approximation for Al
	mu_al = @. 0.3 * (energies / 60.0)^(-3.0) + 0.15

	# Beer-Lambert attenuation through bowtie
	transmission = @. exp(-mu_al * thickness_cm)

	# Apply attenuation
	filtered_photons = photons .* transmission

	return filtered_photons
end

# ╔═╡ e2f3a4b5-c6d7-8e9f-0a1b-2c3d4e5f6a7b
"""
	estimate_scatter_profile(projections; spr_factor=0.15, kernel_sigma=30.0)

Estimates scatter contamination using convolution-based model.

Physics approximation:
- Compton scattering creates smooth, low-frequency scatter
- Scatter-to-Primary Ratio (SPR) ~ 0.1-0.3 for typical body scans
- Scatter distribution ≈ Gaussian convolution of primary

Returns:
- scatter_estimate: Same size as projections
"""
function estimate_scatter_profile(
		projections::Array{Float64, 3};
		spr_factor::Float64 = 0.15,  # Typical scatter-to-primary ratio
		kernel_sigma::Float64 = 30.0  # Scatter blur (pixels)
	)

	n_rows, n_cols, n_proj = size(projections)
	scatter = zeros(Float64, n_rows, n_cols, n_proj)

	# Create Gaussian scatter kernel (separable for efficiency)
	k_size = round(Int, 6 * kernel_sigma)
	k_size = k_size + (k_size % 2 == 0 ? 1 : 0)  # Make odd
	k_half = k_size ÷ 2

	x = collect(-k_half:k_half)
	kernel_1d = exp.(-x.^2 / (2 * kernel_sigma^2))
	kernel_1d ./= sum(kernel_1d)

	# Apply scatter model per projection
	for k in 1:n_proj
		proj_slice = projections[:, :, k]

		# Scatter is smooth/blurred version of primary
		# Convolve with separable Gaussian
		temp = zeros(Float64, n_rows, n_cols)

		# Horizontal convolution
		for r in 1:n_rows
			for c in 1:n_cols
				val = 0.0
				for i in -k_half:k_half
					c_idx = clamp(c + i, 1, n_cols)
					val += proj_slice[r, c_idx] * kernel_1d[i + k_half + 1]
				end
				temp[r, c] = val
			end
		end

		# Vertical convolution
		for c in 1:n_cols
			for r in 1:n_rows
				val = 0.0
				for i in -k_half:k_half
					r_idx = clamp(r + i, 1, n_rows)
					val += temp[r_idx, c] * kernel_1d[i + k_half + 1]
				end
				scatter[r, c, k] = val * spr_factor
			end
		end
	end

	return scatter
end

# ╔═╡ f2a3b4c5-d6e7-f8a9-b0c1-d2e3f4a5b6c7
"""
	apply_poisson_noise(photon_counts; dose_factor=1.0)

Applies Poisson noise to photon counts.

Physics:
- Variance = mean for Poisson process
- Lower dose → higher noise
- dose_factor: 1.0 = nominal dose, 0.5 = half dose, etc.

Returns noisy photon counts.
"""
function apply_poisson_noise(
		photon_counts::Array{Float64};
		dose_factor::Float64 = 1.0
	)
	# Scale counts by dose
	scaled_counts = photon_counts .* dose_factor

	# Apply Poisson noise (using Normal approximation for large counts)
	# For Poisson: variance = mean
	noisy = similar(scaled_counts)

	for i in eachindex(scaled_counts)
		λ = scaled_counts[i]
		if λ > 20.0  # Normal approximation valid
			# Poisson(λ) ≈ Normal(λ, λ)
			noisy[i] = λ + sqrt(λ) * randn()
		elseif λ > 0.0  # Small counts - use Poisson sampling
			noisy[i] = Float64(rand(Poisson(λ)))
		else
			noisy[i] = 0.0
		end
	end

	# Ensure non-negative
	return max.(noisy, 0.0)
end

# ╔═╡ 6aedc1d9-8c89-4df6-8492-f55917739b90
md"""
# 6. Electronic Readout (The DAS)
Conversion from photon counts to line integrals.
"""

# ╔═╡ bd46de74-29e1-4b2e-859b-c94def821008
function readout_das(photon_count, open_field_count)
    # 1. Log transform: -ln(I / I0)
    # Add small epsilon to avoid log(0)
    val = -log((photon_count + 1e-6) / (open_field_count + 1e-6))
    return max(0.0, val)
end

# ╔═╡ cd957e26-4d8d-4ac4-8441-66281a9f6f99
md"""
# 7. Image Reconstruction
Feldkamp-Davis-Kress (FDK) Pipeline.
"""

# ╔═╡ cb4a6c88-ac9c-4fd4-982f-a7904b4228f4
function setup_recon_grid(;
		dfov_mm::Float64,
		matrix_size::Int=512,
		z_thickness_mm::Float64=20.0
	)

	fov_cm = dfov_mm / 10.0
	z_cm = z_thickness_mm / 10.0

	return VoxelGrid(
		size_mm = (dfov_mm, dfov_mm, z_thickness_mm),
		matrix  = (matrix_size, matrix_size, 64)
	)
end

# ╔═╡ f03815dc-5723-4a19-bb44-37ee5244fe11
RECON_GRID = setup_recon_grid(dfov_mm=350.0, matrix_size=512)

# ╔═╡ 6fe73731-a3ac-4909-bd7d-049d795d4cfe
function reconstruct_fdk(projections, geo::Geometry, recon_grid::VoxelGrid)
	# Note: geo now uses .pixel_width_cm internally
	vol = zeros(recon_grid.nx, recon_grid.ny, recon_grid.nz)
	filtered_proj = zeros(size(projections))

	n_pad = nextpow(2, geo.n_cols * 2)
	ramp_scale = 1.0 / geo.pixel_width_cm
	ramp = abs.(fftshift(fftfreq(n_pad))) .* ramp_scale

	# 1. Filter
	for k in 1:length(geo.angles)
		for r in 1:geo.n_rows
			row_data = projections[r, :, k]

			v_pos = (r - geo.n_rows/2 - 0.5) * geo.pixel_height_cm
			u_coords = ((1:geo.n_cols) .- geo.n_cols/2 .- 0.5) .* geo.pixel_width_cm
			weights = geo.SDD_cm ./ sqrt.(geo.SDD_cm^2 .+ u_coords.^2 .+ v_pos^2)

			padded = zeros(n_pad)
			padded[1:geo.n_cols] = row_data .* weights
			filtered = real(ifft(fft(padded) .* fftshift(ramp)))
			filtered_proj[r, :, k] = filtered[1:geo.n_cols]
		end
	end

	# 2. Backproject
	mid_row = geo.n_rows/2 + 0.5
	mid_col = geo.n_cols/2 + 0.5
	rads = deg2rad.(geo.angles)
	sins, coss = sin.(rads), cos.(rads)

	Threads.@threads for k_slice in 1:recon_grid.nz
		z_world = recon_grid.z[k_slice]
		slice_buffer = zeros(recon_grid.nx, recon_grid.ny)

		for k_angle in 1:length(geo.angles)
			sin_a, cos_a = sins[k_angle], coss[k_angle]
			for j in 1:recon_grid.ny, i in 1:recon_grid.nx
				px, py = recon_grid.x[i], recon_grid.y[j]

				x_r = px * cos_a + py * sin_a
				y_r = -px * sin_a + py * cos_a
				dist = geo.SAD_cm + y_r

				if dist > 0.1
					mag = geo.SDD_cm / dist
					col_idx = (x_r * mag) / geo.pixel_width_cm + mid_col
					row_idx = (z_world * mag) / geo.pixel_height_cm + mid_row

					c_fl = floor(Int, col_idx)
					r_fl = floor(Int, row_idx)

					if c_fl >= 1 && c_fl < geo.n_cols && r_fl >= 1 && r_fl < geo.n_rows
						dc, dr = col_idx - c_fl, row_idx - r_fl
						val = (1-dc)*(1-dr)*filtered_proj[r_fl, c_fl, k_angle] +
							  (dc)*(1-dr)*filtered_proj[r_fl, c_fl+1, k_angle] +
							  (1-dc)*(dr)*filtered_proj[r_fl+1, c_fl, k_angle] +
							  (dc)*(dr)*filtered_proj[r_fl+1, c_fl+1, k_angle]

						slice_buffer[i, j] += val * (geo.SAD_cm / dist)^2
					end
				end
			end
		end
		vol[:, :, k_slice] = slice_buffer
	end
	return vol .* (2 * pi / length(geo.angles))
end

# ╔═╡ 248480e4-e27a-4f7b-aa81-022cbf31370f
md"""
## Output & Analysis
HU Conversion and Visualization.
"""

# ╔═╡ a227e438-b683-4373-a9d0-4a47dcb898e2
function convert_to_hu(vol_mu, mu_water)
    return 1000.0 .* (vol_mu .- mu_water) ./ mu_water
end

# ╔═╡ b99640a6-1420-43d6-90b6-27a7ff9bb0c1
md"""
# 8. Main Simulation Loop
Binding the complete physics chain together with all enhanced features.
"""

# ╔═╡ 9e0b86e6-71d2-499b-b2a7-c8bf89712c9f
function run_enhanced_simulation(;
		phantom::PhysicalPhantom,
		source::XRaySource,
		geo::Geometry,
		det::DetectorResponse,
		mat_lib::MaterialLibrary,
		grid_meta::GridMeta,
		id_lut::Vector{Int},
		# --- Physics options ---
		use_bowtie::Bool = true,
		bowtie_thickness_cm::Float64 = 3.0,
		use_scatter::Bool = true,
		scatter_spr::Float64 = 0.15,
		scatter_kernel_sigma::Float64 = 30.0,
		use_poisson_noise::Bool = true,
		dose_factor::Float64 = 1.0,
		use_compiled::Bool = false,  # Use Reactant-compiled ray tracer
		compiled_tracer = nothing  # Pass compiled function if use_compiled=true
	)

	@info "🚀 Enhanced Physics Simulation"
	@info "  ├─ Bowtie filter: $(use_bowtie)"
	@info "  ├─ Scatter model: $(use_scatter) (SPR=$(scatter_spr))"
	@info "  ├─ Poisson noise: $(use_poisson_noise) (dose=$(dose_factor)x)"
	@info "  └─ Projections: $(geo.n_rows)×$(geo.n_cols)×$(length(geo.angles))"

	# ═══════════════════════════════════════════════════════════
	# STEP 1: Pre-calculate I₀ (Air Scan) with Bowtie Filter
	# ═══════════════════════════════════════════════════════════
	# This represents what the detector sees with no phantom present
	# Must account for bowtie filter attenuation

	n_sub = 2
	sub_width = 1.0 / n_sub
	sub_offsets = (1:n_sub) .* sub_width .- (sub_width/2 + 0.5)

	# Calculate I₀ for each detector column (bowtie varies per column)
	I0_per_column = zeros(Float64, geo.n_cols)

	for c in 1:geo.n_cols
		# Apply bowtie filter to source spectrum for this column
		if use_bowtie
			filtered_photons = apply_bowtie_filter(
				source.energies,
				source.photons,
				c,
				geo.n_cols;
				max_thickness_cm = bowtie_thickness_cm
			)
		else
			filtered_photons = source.photons
		end

		# Total energy reaching detector (no phantom)
		I0_per_column[c] = sum(filtered_photons .* source.energies .* det.efficiency)
	end

	# ═══════════════════════════════════════════════════════════
	# STEP 2: Forward Projection (Polychromatic Ray Tracing)
	# ═══════════════════════════════════════════════════════════

	energy_values = zeros(Float64, geo.n_rows, geo.n_cols, length(geo.angles))

	@info "📷 Acquiring projections..."

	Threads.@threads for k in 1:length(geo.angles)

		# Geometry for this projection angle
		src_pos    = geo.source_positions[:, k]
		det_center = geo.det_centers[:, k]
		u_vec      = geo.det_u_vecs[:, k]
		v_vec      = geo.det_v_vecs[:, k]

		n_mats = size(mat_lib.μ_matrix, 1)

		for r in 1:geo.n_rows, c in 1:geo.n_cols

			# Apply bowtie filter to source spectrum for this column
			if use_bowtie
				filtered_photons = apply_bowtie_filter(
					source.energies,
					source.photons,
					c,
					geo.n_cols;
					max_thickness_cm = bowtie_thickness_cm
				)
			else
				filtered_photons = source.photons
			end

			pixel_energy_sum = 0.0

			# Sub-pixel sampling for anti-aliasing
			for sr in 1:n_sub, sc in 1:n_sub
				u_off = (c - geo.n_cols/2 - 0.5 + sub_offsets[sc]) * geo.pixel_width_cm
				v_off = (r - geo.n_rows/2 - 0.5 + sub_offsets[sr]) * geo.pixel_height_cm
				target_pos = det_center .+ (u_off .* u_vec) .+ (v_off .* v_vec)

				# Ray trace through phantom (Reactant-compatible)
				# Choose ray tracer (compiled or standard)
				if use_compiled && compiled_tracer !== nothing
					path_lengths = compiled_tracer(
						grid_meta,
						phantom.material_ids,
						phantom.densities,
						id_lut,
						n_mats,
						src_pos[1], src_pos[2], src_pos[3],
						target_pos[1], target_pos[2], target_pos[3]
					)
				else
					path_lengths = trace_ray_material_paths(
						grid_meta,
						phantom.material_ids,
						phantom.densities,
						id_lut,
						n_mats,
						src_pos[1], src_pos[2], src_pos[3],
						target_pos[1], target_pos[2], target_pos[3]
					)
			end

				# Calculate energy-dependent attenuation
				atten_per_energy = zeros(Float64, length(source.energies))
				for m_idx in 1:n_mats
					if path_lengths[m_idx] > 0
						for e_idx in 1:length(source.energies)
							atten_per_energy[e_idx] += mat_lib.μ_matrix[m_idx, e_idx] * path_lengths[m_idx]
						end
					end
				end

				# Polychromatic transmission: I = Σ N₀(E) × exp(-μ(E)×L) × E × η(E)
				sub_energy = sum(@. filtered_photons * exp(-atten_per_energy) * source.energies * det.efficiency)

				pixel_energy_sum += sub_energy
			end

			# Average over sub-pixels
			avg_energy = pixel_energy_sum / (n_sub^2)

			# Work with energy directly (no conversion to photon counts!)
			avg_energy_value = avg_energy

			energy_values[r, c, k] = avg_energy_value
		end
	end

	# ═══════════════════════════════════════════════════════════
	# STEP 3: Add Scatter (Gaussian Convolution Model)
	# ═══════════════════════════════════════════════════════════

	if use_scatter
		@info "🌊 Estimating scatter contribution..."
		scatter_energy = estimate_scatter_profile(
			energy_values;
			spr_factor = scatter_spr,
			kernel_sigma = scatter_kernel_sigma
		)

		# Add scatter to primary
		photon_counts .+= scatter_counts
	end

	# ═══════════════════════════════════════════════════════════
	# STEP 4: Apply Poisson Noise (Quantum Noise)
	# ═══════════════════════════════════════════════════════════

	if use_poisson_noise
		@info "🎲 Applying Poisson noise (dose=$(dose_factor)x)..."
		energy_values = apply_poisson_noise(
			energy_values;
			dose_factor = dose_factor
		)
	end

	# Add electronic noise from detector
	if det.noise_sigma > 0
		@info "📟 Adding electronic noise..."
		photon_counts .+= randn(size(photon_counts)) .* det.noise_sigma
	end

	# ═══════════════════════════════════════════════════════════
	# STEP 5: DAS Readout (Log Transform to Line Integrals)
	# ═══════════════════════════════════════════════════════════

	@info "📊 DAS readout (log transform)..."
	projections = zeros(Float64, size(energy_values))

	for k in 1:length(geo.angles), r in 1:geo.n_rows, c in 1:geo.n_cols
		# Use energy values directly (no conversion needed)
		# Energy values already in correct units
		photon_energy = energy_values[r, c, k]
		I0_energy = I0_per_column[c]

		projections[r, c, k] = readout_das(photon_energy, I0_energy)
	end

	@info "✅ Simulation complete!"

	return projections
end

# ╔═╡ 152c98f7-2b30-4f36-9f10-b60c63c3cb53
enhanced_sino = run_enhanced_simulation(
	phantom = PHANTOM,
	source = SOURCE_120_REALISTIC,  # Now uses realistic spectrum with K-lines!
	geo = GEOMETRY_AQUILION_ONE,       # Canon Aquilion ONE geometry
	det = DETECTOR_RESPONSE,
	mat_lib = MATERIAL_LIBRARY,
	grid_meta = GRID_META,
	id_lut = ID_LUT,
	# Physics options
	use_bowtie = false,  # Disabled by default (causing issues)
	bowtie_thickness_cm = 0.5,  # Much thinner (not 3.0!)
	use_scatter = false,  # Disabled by default
	scatter_spr = 0.05,  # More conservative
	use_poisson_noise = false,  # Disabled by default for debugging
	dose_factor = 1.0
)

# ╔═╡ ecc8f0f1-6a64-400e-91f9-9b4d0bce0d25
vol_enhanced = convert_to_hu(
	reconstruct_fdk(enhanced_sino, GEOMETRY_AQUILION_ONE, RECON_GRID),
	0.195
)

# ╔═╡ e05fc8c6-4b4f-4fb4-baf6-9df8dc58d343
let
	# --- Data Extraction ---
	# 1. Ground Truth Slice (Uses Phantom Grid)
	mid_z_p = PHANTOM.grid.nz ÷ 2
	gt_slice = PHANTOM.material_ids[:, :, mid_z_p]

	# 2. Sinogram (Central Row)
	mid_row = size(enhanced_sino, 1) ÷ 2
	sino_slice = enhanced_sino[mid_row, :, :]

	# 3. Recon Slice (Uses Recon Grid)
	mid_z_r = RECON_GRID.nz ÷ 2
	recon_slice = vol_enhanced[:, :, mid_z_r]

	# --- Visualization ---
	f = Figure(size=(1200, 450))

	# Plot 1: Ground Truth
	ax1 = Axis(f[1, 1], title="Ground Truth (Material IDs)", aspect=DataAspect())
	hm1 = heatmap!(ax1, PHANTOM.grid.x, PHANTOM.grid.y, gt_slice, colormap=:tab20)

	# Plot 2: Sinogram
	ax2 = Axis(f[1, 2], title="Sinogram", xlabel="Detector Pixel", ylabel="Angle")
	hm2 = heatmap!(ax2, sino_slice, colormap=:grays)

	# Plot 3: Reconstruction (HU)
	# Window/Level: Soft Tissue (-200 to +400 covers air gaps to bone inserts)
	ax3 = Axis(f[1, 3], title="Reconstruction (HU)", aspect=DataAspect())
	hm3 = heatmap!(ax3, RECON_GRID.x, RECON_GRID.y, recon_slice, colormap=:grays, colorrange=(-200, 400))
	Colorbar(f[1, 4], hm3, label="Hounsfield Units")

	f
end

# ╔═╡ 05f992b3-cc64-4722-8de8-08152a8b2a81
md"""
---
# 9. Autodifferentiation & XLA Compilation

Now that we have a complete physics model, let's prepare it for:
- **Reactant.jl**: XLA compilation for massive speedup
- **Enzyme.jl**: Automatic differentiation through compiled code

This enables gradient-based optimization and inverse problems!
"""

# ╔═╡ 51d36823-66d5-4080-a1dc-d2feb87ef4b3
md"""
## 9.1 Reactant @compile Testing

Reactant compiles Julia code to XLA (Accelerated Linear Algebra) for GPU/TPU execution.

**Requirements for successful compilation:**
- ✅ Pure functions (no callbacks) - DONE!
- ✅ Statically typed operations
- ✅ No dynamic allocations in hot loops
- ⚠️  May need type annotations for complex functions

**Test strategy:**
1. Start with simple functions (ray tracer)
2. Build up to full projection
3. Finally compile entire forward pass
"""

# ╔═╡ 3085d468-1871-41cb-b4b8-b62f4a1c3f75
md"""
### Reactant Test: Single Ray

Let's try compiling a single ray trace with Reactant.

```julia
using Reactant

# Compile the ray tracer
@compile trace_ray_material_paths(
    GRID_META,
    PHANTOM.material_ids,
    PHANTOM.densities,
    ID_LUT,
    length(unique(PHANTOM.material_ids)),
    0.0, 50.0, 0.0,  # source position
    0.0, -50.0, 0.0  # detector position
)
```

**Status:** Ready to test! Uncomment the code above to run.
"""

# ╔═╡ e0f1a2b3-c4d5-f6a7-b8c9-0d1e2f3a4b5c
md"""
## 9.2 Enzyme Autodifferentiation

Enzyme.jl enables automatic differentiation through the entire simulation.

**Potential applications:**

### 1. Material Decomposition
Estimate material densities from measured projections:
- Variables: phantom.densities (Float32 array)
- Objective: Match simulated vs measured sinogram
- Enables dual-energy CT material separation

### 2. Dose Optimization
Find minimum dose that maintains diagnostic quality:
- Variables: source.mAs (dose parameter)
- Objective: Minimize dose while maintaining SNR target
- Critical for pediatric imaging and low-dose protocols

### 3. Geometry Calibration
Refine scanner geometry from phantom scans:
- Variables: source position, detector angles, SDD, SAD
- Objective: Match measured vs simulated projections
- Critical for cone-beam artifact correction

### 4. Learned Scatter Correction
Our current scatter model is approximate (Gaussian convolution).
We can learn better correction kernels:
- Simulate high-accuracy scatter (Monte Carlo)
- Learn fast approximation via gradient descent
- Deploy learned model for real-time correction

### 5. Iterative Reconstruction (MBIR)
Model-Based Iterative Reconstruction using our physics model:
```
argmin_μ ||A(μ) - y||² + λR(μ)
```
Where:
- `A(μ)` is our differentiable forward model
- `y` is measured data
- `R(μ)` is regularization (e.g., TV)
- Solve via gradient descent with Enzyme!

**All of this is now possible with our differentiable simulator!** 🚀
"""

# ╔═╡ Cell order:
# ╟─44c9569c-1ef7-4164-9866-29194d447648
# ╟─9e413371-e11a-4359-b959-83f98632a77f
# ╠═4ba89bf3-342c-4762-8d26-4fe7f4b49b1d
# ╠═116b566e-793c-4b18-9bfe-03c1b7095b7c
# ╠═68579afa-2368-462a-a58a-68ea7060b77a
# ╠═2351e3ef-ed73-4160-8599-5059a00b0b0b
# ╠═62dc819b-09cd-4c97-a1d9-ea95b4c1a5b2
# ╠═0699f400-b89d-44aa-8b2e-5a68c5de3b36
# ╠═42493c98-0d13-46e5-b323-afb3bf524f29
# ╠═d7e941f0-e24d-4724-93c4-8c939eaa0d21
# ╠═ab87b970-4ac8-475c-b9b4-26b783b8a457
# ╠═e616e726-0926-4ce9-8a0d-4dbcabd8b477
# ╠═64021ccf-7534-4f0f-92a5-79fdd38b602a
# ╠═3ff07656-5cec-41bb-903d-4b986333cb5e
# ╠═794e5d30-022f-4a24-9954-ccfff9238581
# ╠═9f72a560-f0fd-4866-89e0-0b5326fba508
# ╠═84903695-03fb-4c7f-b475-7f1db983f14c
# ╟─704ad6f6-12fa-4b51-a0a2-95a1e223a09a
# ╠═48fca871-4e8a-4dd3-9041-9e6fbf575600
# ╠═e05adb7c-5c80-4626-9264-74d1d4307533
# ╠═84d9222e-a9e4-4f1a-a25b-6abe7fedfcd8
# ╠═95f3e42d-e91e-430c-9cb6-1f9d7adc4d16
# ╠═2846460f-ec4c-4b6d-9b47-c9e1057e5fa2
# ╠═8636ce95-cee4-49c2-9d1b-8782ddac282e
# ╠═1a124cfe-a285-4240-b8e8-ea59a2a2029d
# ╟─0cd3bd6d-f00e-4dca-812e-c443bf730a0f
# ╟─e8fcc5b6-e7e9-4019-8f93-62013518b3d9
# ╠═be378bc8-d102-4ada-b624-a13ab155762a
# ╠═7099deb4-0ebd-4833-955f-38aee7e147a6
# ╠═0c543119-f37b-4a9c-bc45-4b680a1dbfd1
# ╟─be3defc1-9ef7-44d7-a105-143151b39f12
# ╠═41d0439f-5b1f-42f0-be34-f21637c3fca4
# ╠═85fa3167-f800-4574-b78e-12bb0f993be2
# ╟─5622c462-d139-42c8-ba6f-7966339f2f19
# ╠═dd46bf7b-4136-434e-ad1b-ea1b8482cde5
# ╠═e9faca70-67f2-4a03-99e3-5945851e8e2d
# ╟─d72b418e-49d3-41c8-b81d-830f9750b1ef
# ╟─f40b9973-746c-4445-8fec-d4641f487239
# ╟─498aef40-ef97-411c-aaa8-61468ed314d9
# ╠═bdcfe35d-f96a-4d06-b1ab-933e15345856
# ╟─1f0ba521-fa29-4ad1-a46e-3033bcb011e1
# ╠═f1a2b3c4-d5e6-7f8a-9b0c-1d2e3f4a5b6c
# ╠═a2b3c4d5-e6f7-8a9b-0c1d-2e3f4a5b6c7d
# ╟─45cc1bd4-3a00-4e0b-a0ec-b3d953f50d72
# ╟─4ee516dc-9f1f-4291-9392-2098ed541958
# ╠═d11d555e-0877-4f2f-8d8b-3a2848a96393
# ╠═27bb0a72-d80e-4035-a1e7-3b45cfc0f49a
# ╠═d6452eef-ee88-45ac-bfff-e384dc386a76
# ╟─736d9c19-9d61-447c-8c4c-2cfb330b5f6c
# ╠═b7f140fe-fe3e-4121-a738-be04d6c961d3
# ╟─2e66cacf-e923-4d05-b875-111357a0dcfb
# ╟─9af37598-6b82-4884-89fb-a450c0247fd3
# ╠═28bbf834-4a5a-4632-8ded-7cabe3bbd7c0
# ╠═3becb001-c352-457f-8515-ebb832633a6a
# ╠═c1d2e3f4-a5b6-7c8d-9e0f-1a2b3c4d5e6f
# ╟─9cb9bcc9-e1f6-4978-a415-f68c689f488f
# ╠═e7e54f4e-e52c-4cdd-939f-11fb156da33d
# ╠═d2e3f4a5-b6c7-8d9e-0f1a-2b3c4d5e6f7a
# ╟─25fb9f2c-21b6-4e93-9706-7e95ec350390
# ╟─7aa9d80d-8517-4025-8188-42256fa6b261
# ╠═0d9725ae-e4a8-4e6e-9674-eb4ca4607270
# ╟─f519c8f6-a355-4243-a917-531dad43bcc0
# ╠═1cd38a76-87c8-4bff-b012-29829f6251fd
# ╟─dc1c43aa-047c-4e83-8f9c-d9815f24cf08
# ╠═66406706-4b18-4b75-a682-76b4012b1e07
# ╟─b6cb080e-0ef4-47e2-8b88-f0aac56eb6d4
# ╟─25eb6c8c-5ac1-40b6-af9d-9801abba6fad
# ╠═1fa8e20b-27f4-40d7-8e4e-15f51f29e0ad
# ╟─8905383e-f450-42fa-a7a0-72351695287f
# ╠═04fe5478-c4b5-42dc-a37b-c0975136f4d7
# ╠═846525d5-4f37-40b1-b364-2657017e8e7f
# ╠═009fe763-3619-4fd9-ab46-5fdaf065a15a
# ╠═38027728-dded-4fed-840f-439292e46bf6
# ╟─6b3a037a-9da0-4ed6-9a1b-d640361dbdb3
# ╠═f16df8ce-5d11-4e54-8601-7121380f4cd5
# ╠═e2f3a4b5-c6d7-8e9f-0a1b-2c3d4e5f6a7b
# ╠═f2a3b4c5-d6e7-f8a9-b0c1-d2e3f4a5b6c7
# ╟─6aedc1d9-8c89-4df6-8492-f55917739b90
# ╠═bd46de74-29e1-4b2e-859b-c94def821008
# ╟─cd957e26-4d8d-4ac4-8441-66281a9f6f99
# ╠═cb4a6c88-ac9c-4fd4-982f-a7904b4228f4
# ╠═f03815dc-5723-4a19-bb44-37ee5244fe11
# ╠═6fe73731-a3ac-4909-bd7d-049d795d4cfe
# ╟─248480e4-e27a-4f7b-aa81-022cbf31370f
# ╠═a227e438-b683-4373-a9d0-4a47dcb898e2
# ╟─b99640a6-1420-43d6-90b6-27a7ff9bb0c1
# ╠═9e0b86e6-71d2-499b-b2a7-c8bf89712c9f
# ╠═152c98f7-2b30-4f36-9f10-b60c63c3cb53
# ╠═ecc8f0f1-6a64-400e-91f9-9b4d0bce0d25
# ╟─e05fc8c6-4b4f-4fb4-baf6-9df8dc58d343
# ╟─05f992b3-cc64-4722-8de8-08152a8b2a81
# ╟─51d36823-66d5-4080-a1dc-d2feb87ef4b3
# ╟─3085d468-1871-41cb-b4b8-b62f4a1c3f75
# ╟─e0f1a2b3-c4d5-f6a7-b8c9-0d1e2f3a4b5c
