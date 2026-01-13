### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 8a5d7c3e-b1f2-4a89-9c0d-3e8f5a6b7d12
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 9b6e8d4f-c2a3-5b9a-0d1e-4f9a6b7c8e23
begin
	using BasisSimulator
	using Statistics
	using CairoMakie
end

# ╔═╡ 0c7f9e5a-d3b4-6c0b-1e2f-5a0b7c8d9f34
md"""
# Realistic CT Simulation with BasisSimulator.jl

This notebook demonstrates **ultra-high resolution (UHR) CT simulation** using the
**GE Revolution Apex Elite** scanner configuration with physically accurate parameters.

## Scanner: GE Revolution Apex Elite
- **FDA 510(k)**: K213715
- **Detector**: 256-row Gemstone Clarity (832 columns)
- **SAD/SDD**: 626 mm / 1097 mm
- **Max views per rotation**: 2496 (using 984 for standard protocols)
- **kVp options**: 70, 80, 100, 120, 140 kV

## Simulation Configuration
- **Phantom**: 1024×1024×40 voxels (UHR "ground truth")
- **Reconstruction**: 512×512×20 voxels (clinical output)
- **Projections**: 984 angles per rotation (clinical standard)

## Key API Functions
- `simulate_sinogram()` - Forward projection with configurable physical effects
- `reconstruct()` - FDK, SIRT, or CGLS reconstruction
- `simulate_and_reconstruct()` - Complete pipeline in one call
"""

# ╔═╡ 1d8a0f6b-e4c5-7d1c-2f3a-6b1c8d9e0a45
md"""
## 1. Scanner and Acquisition Configuration

### GE Revolution Apex Elite Parameters
"""

# ╔═╡ 2e9b1a7c-f5d6-8e2d-3a4b-7c2d9e0f1b56
begin
	# Ultra-high resolution phantom (1024x1024x40)
	# This represents the "true" physical object at very high resolution
	PHANTOM_SIZE = 1024
	PHANTOM_SLICES = 40

	# Clinical reconstruction output (512x512x20)
	RECON_SIZE = 512
	RECON_SLICES = 20

	# Clinical projection parameters (GE Revolution Apex Elite)
	N_ANGLES = 984          # Views per rotation (clinical standard)
	N_DETECTOR_COLS = 832   # GE Gemstone Clarity detector columns
	N_DETECTOR_ROWS = 64    # 64 rows = 40mm z-coverage at 0.625mm

	# GE Revolution Apex Elite geometry (FDA K213715)
	SAD = 62.6   # cm (626 mm source-to-isocenter)
	SDD = 109.7  # cm (1097 mm source-to-detector)
end

# ╔═╡ 3f0c2b8d-a6e7-9f3e-4b5c-8d3e0f1a2c67
md"""
### Clinical Protocol Definitions

Each protocol specifies exact **kVp** and **mAs** values matching clinical practice:
"""

# ╔═╡ 4a1d3c9e-b7f8-0a4f-5c6d-9e4f1a2b3d78
begin
	# Protocol 1: Standard Chest CT (120 kVp, 200 mAs)
	# Typical for routine chest imaging
	PROTOCOL_CHEST = (
		name = "Standard Chest CT",
		kVp = 120,
		mAs = 200,
		rotation_time = 0.5,  # seconds
		I0 = 2e5              # Photon fluence (proportional to mAs)
	)

	# Protocol 2: Low-Dose Screening (100 kVp, 50 mAs)
	# Used for lung cancer screening (LDCT)
	PROTOCOL_LOW_DOSE = (
		name = "Low-Dose Screening CT",
		kVp = 100,
		mAs = 50,
		rotation_time = 0.5,
		I0 = 5e4              # Lower fluence for reduced dose
	)

	# Protocol 3: High-Resolution Abdomen (120 kVp, 400 mAs)
	# Contrast-enhanced abdominal imaging
	PROTOCOL_ABDOMEN = (
		name = "High-Resolution Abdomen CT",
		kVp = 120,
		mAs = 400,
		rotation_time = 0.5,
		I0 = 4e5              # Higher fluence for reduced noise
	)

	# Protocol 4: Pediatric (80 kVp, 100 mAs)
	# Dose-optimized for pediatric patients
	PROTOCOL_PEDIATRIC = (
		name = "Pediatric CT",
		kVp = 80,
		mAs = 100,
		rotation_time = 0.5,
		I0 = 1e5
	)
end

# ╔═╡ 5b2e4d0f-c8a9-1b5a-6d7e-0f5a2b3c4e89
md"""
**Selected Protocols for Comparison:**

| Protocol | kVp | mAs | Use Case |
|----------|-----|-----|----------|
| $(PROTOCOL_LOW_DOSE.name) | $(PROTOCOL_LOW_DOSE.kVp) | $(PROTOCOL_LOW_DOSE.mAs) | Lung screening |
| $(PROTOCOL_CHEST.name) | $(PROTOCOL_CHEST.kVp) | $(PROTOCOL_CHEST.mAs) | Routine chest |
| $(PROTOCOL_ABDOMEN.name) | $(PROTOCOL_ABDOMEN.kVp) | $(PROTOCOL_ABDOMEN.mAs) | Contrast studies |
"""

# ╔═╡ 0a7d9c5e-b3f4-6a0f-1c2d-5e0f7a8b9d34
md"""
## 2. Create UHR Phantom and GE Apex Geometry
"""

# ╔═╡ 1b8e0d6f-c4a5-7b1a-2d3e-6f1a8b9c0e45
md"""
### 2a. Ultra-High Resolution Phantom (1024×1024×40)

The phantom is created at very high resolution to represent the "true" physical object.
CT reconstruction naturally produces lower resolution images due to sampling.
"""

# ╔═╡ 6c3f5e1a-d9b0-2c6b-7e8f-1a6b3c4d5f90
phantom = create_gammex_472(n_voxels=PHANTOM_SIZE, n_slices=PHANTOM_SLICES)

# ╔═╡ 2c9f1e7a-d5b6-8c2b-3e4f-7a2b9c0d1f56
md"""
### 2b. GE Revolution Apex Elite Scanner Geometry

Creating cone-beam geometry matching the GE Revolution Apex Elite specifications:
- **$(N_ANGLES) projections** per rotation (clinical standard)
- **$(N_DETECTOR_COLS)×$(N_DETECTOR_ROWS) detector** (Gemstone Clarity)
- **SAD/SDD**: $(SAD) cm / $(SDD) cm
"""

# ╔═╡ 3d0a2f8b-e6c7-9d3c-4f5a-8b3c0d1e2a67
geom = create_aquilion_one(
	n_angles=N_ANGLES,
	n_rows=N_DETECTOR_ROWS,
	n_cols=N_DETECTOR_COLS,
	fov_cm=phantom.fov[1],
	sad=SAD,
	sdd=SDD
)

# ╔═╡ 4e1b3a9c-f7d8-0e4d-5a6b-9c4d1e2f3b78
begin
	# Output reconstruction size: 512×512×20
	output_size = (RECON_SIZE, RECON_SIZE, RECON_SLICES)

	# Reference μ for water at effective energy
	μ_water_ref = get_reference_μ_water(60.0)

	md"""
	**Reconstruction Output**: $(output_size[1])×$(output_size[2])×$(output_size[3]) voxels

	**Phantom FOV**: $(phantom.fov[1]) × $(phantom.fov[2]) × $(phantom.fov[3]) cm
	"""
end

# ╔═╡ 5f2c4b0d-a8e9-1f5e-6b7c-0d5e2f3a4c89
let
	mid = PHANTOM_SLICES ÷ 2
	fig = Figure(size=(700, 280))

	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Phantom Regions (UHR: $(PHANTOM_SIZE)×$(PHANTOM_SIZE))")
	heatmap!(ax1, Float64.(phantom.mask[:, :, mid])'; colormap=:tab20)
	hidedecorations!(ax1)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Attenuation μ (cm⁻¹)")
	hm = heatmap!(ax2, phantom.μ[:, :, mid]'; colormap=:viridis)
	Colorbar(fig[1, 3], hm, label="μ (cm⁻¹)")
	hidedecorations!(ax2)

	fig
end

# ╔═╡ 6a3d5c1e-b9f0-2a6f-7c8d-1e6f3a4b5d90
md"""
## 3. CT Simulation with Clinical Protocols

Each simulation uses specific **kVp** and **mAs** values with full physical effects:
- Polychromatic X-ray spectrum (beam hardening)
- Flat filter (2.5mm Al + 0.1mm Cu)
- Bowtie filter (medium body)
- Scatter radiation
- Detector noise (quantum + electronic, scaled by mAs)
"""

# ╔═╡ 7b4e6d2f-c0a1-3b7a-8d9e-2f7a4b5c6e01
md"""
### 3a. Protocol: $(PROTOCOL_LOW_DOSE.name) ($(PROTOCOL_LOW_DOSE.kVp) kVp, $(PROTOCOL_LOW_DOSE.mAs) mAs)

Low-dose technique for lung cancer screening. Higher noise, lower radiation exposure.
"""

# ╔═╡ 8c5f7e3a-d1b2-4c8b-9e0f-3a8b5c6d7f12
sino_low_dose = simulate_sinogram(phantom, geom;
	polychromatic=true,
	kVp=PROTOCOL_LOW_DOSE.kVp,
	detector=default_detector_model(I0=PROTOCOL_LOW_DOSE.I0),
	seed=42
)

# ╔═╡ 9d6a8f4b-e2c3-5d9c-0f1a-4b9c6d7e8a23
md"""
### 3b. Protocol: $(PROTOCOL_CHEST.name) ($(PROTOCOL_CHEST.kVp) kVp, $(PROTOCOL_CHEST.mAs) mAs)

Standard chest imaging protocol. Balanced noise and dose.
"""

# ╔═╡ 0e7b9a5c-f3d4-6e0d-1a2b-5c0d7e8f9b34
sino_standard = simulate_sinogram(phantom, geom;
	polychromatic=true,
	kVp=PROTOCOL_CHEST.kVp,
	detector=default_detector_model(I0=PROTOCOL_CHEST.I0),
	seed=42
)

# ╔═╡ 1f8c0b6d-a4e5-7f1e-2b3c-6d1e8f9a0c45
md"""
### 3c. Protocol: $(PROTOCOL_ABDOMEN.name) ($(PROTOCOL_ABDOMEN.kVp) kVp, $(PROTOCOL_ABDOMEN.mAs) mAs)

High-quality abdomen imaging. Lower noise for contrast detection.
"""

# ╔═╡ 2a9d1c7e-b5f6-8a2f-3c4d-7e2f9a0b1d56
sino_high_quality = simulate_sinogram(phantom, geom;
	polychromatic=true,
	kVp=PROTOCOL_ABDOMEN.kVp,
	detector=default_detector_model(I0=PROTOCOL_ABDOMEN.I0),
	seed=42
)

# ╔═╡ 3b0e2d8f-c6a7-9b3a-4d5e-8f3a0b1c2e67
md"""
### 3d. Ideal Reference (Monochromatic, No Effects)

For comparison: ideal monochromatic simulation without physical effects.
"""

# ╔═╡ 4c1f3e9a-d7b8-0c4b-5e6f-9a4b1c2d3f78
sino_ideal = simulate_sinogram(phantom, geom;
	polychromatic=false,
	flat_filter=nothing,
	bowtie_filter=nothing,
	scatter=nothing,
	detector=nothing,
	crosstalk=nothing,
	lag=nothing,
	optical_crosstalk=nothing,
	fill_factor=nothing,
	focal_spot=nothing
)

# ╔═╡ 5d2a4f0b-e8c9-1d5c-6f7a-0b5c2d3e4a89
let
	fig = Figure(size=(1000, 250))
	mid_row = N_DETECTOR_ROWS ÷ 2

	sinos = [sino_ideal, sino_low_dose, sino_standard, sino_high_quality]
	titles = [
		"Ideal (mono)",
		"$(PROTOCOL_LOW_DOSE.kVp)kVp/$(PROTOCOL_LOW_DOSE.mAs)mAs",
		"$(PROTOCOL_CHEST.kVp)kVp/$(PROTOCOL_CHEST.mAs)mAs",
		"$(PROTOCOL_ABDOMEN.kVp)kVp/$(PROTOCOL_ABDOMEN.mAs)mAs"
	]

	for (i, (sino, title)) in enumerate(zip(sinos, titles))
		ax = Axis(fig[1, i], xlabel="Column", ylabel="Angle", title=title)
		heatmap!(ax, sino[:, mid_row, :]'; colormap=:hot)
	end

	fig
end

# ╔═╡ 6e3b5a1c-f9d0-2e6d-7a8b-1c6d3e4f5b90
md"""
## 4. FDK Reconstruction (512×512×20)

Reconstructing from $(PHANTOM_SIZE)×$(PHANTOM_SIZE)×$(PHANTOM_SLICES) phantom projections
to $(RECON_SIZE)×$(RECON_SIZE)×$(RECON_SLICES) clinical output.
"""

# ╔═╡ 7f4c6b2d-a0e1-3f7e-8b9c-2d7e4f5a6c01
recon_ideal = reconstruct(sino_ideal, geom, output_size, phantom.fov)

# ╔═╡ 8a5d7c3e-b1f2-4a8f-9c0d-3e8f5a6b7d13
recon_low_dose = reconstruct(sino_low_dose, geom, output_size, phantom.fov)

# ╔═╡ 9b6e8d4f-c2a3-5b9a-0d1e-4f9a6b7c8e24
recon_standard = reconstruct(sino_standard, geom, output_size, phantom.fov)

# ╔═╡ 0c7f9e5a-d3b4-6c0b-1e2f-5a0b7c8d9f35
recon_high_quality = reconstruct(sino_high_quality, geom, output_size, phantom.fov)

# ╔═╡ 1d8a0f6b-e4c5-7d1c-2f3a-6b1c8d9e0a46
md"""
## 5. Protocol Comparison

Comparing image quality across different kVp/mAs protocols:
"""

# ╔═╡ 2e9b1a7c-f5d6-8e2d-3a4b-7c2d9e0f1b57
let
	mid = RECON_SLICES ÷ 2
	fig = Figure(size=(1100, 550))

	# Clinical CT window
	hu_range = (-300, 600)

	recons = [recon_ideal, recon_low_dose, recon_standard, recon_high_quality]
	titles = [
		"Ideal (Reference)",
		"$(PROTOCOL_LOW_DOSE.kVp)kVp / $(PROTOCOL_LOW_DOSE.mAs)mAs\n(Low Dose)",
		"$(PROTOCOL_CHEST.kVp)kVp / $(PROTOCOL_CHEST.mAs)mAs\n(Standard)",
		"$(PROTOCOL_ABDOMEN.kVp)kVp / $(PROTOCOL_ABDOMEN.mAs)mAs\n(High Quality)"
	]

	for (i, (r, t)) in enumerate(zip(recons, titles))
		ax = Axis(fig[1, i], aspect=DataAspect(), title=t, titlesize=11)
		hm = heatmap!(ax, μ_to_HU(r[:, :, mid], μ_water_ref)';
			colormap=:grays, colorrange=hu_range)
		hidedecorations!(ax)
		i == length(recons) && Colorbar(fig[1, 5], hm, label="HU")
	end

	# Zoomed view of center (to show noise differences)
	zoom = 200:312
	for (i, (r, t)) in enumerate(zip(recons, titles))
		ax = Axis(fig[2, i], aspect=DataAspect(), title="Center ROI", titlesize=10)
		hu_slice = μ_to_HU(r[:, :, mid], μ_water_ref)
		heatmap!(ax, hu_slice[zoom, zoom]'; colormap=:grays, colorrange=hu_range)
		hidedecorations!(ax)
	end

	Label(fig[0, :], "GE Revolution Apex Elite - Protocol Comparison ($(N_ANGLES) views)", fontsize=14)

	fig
end

# ╔═╡ 3f0c2b8d-a6e7-9f3e-4b5c-8d3e0f1a2c68
md"""
## 6. Quantitative Analysis by Protocol
"""

# ╔═╡ 4a1d3c9e-b7f8-0a4f-5c6d-9e4f1a2b3d79
let
	mid_z = RECON_SLICES ÷ 2

	# Scale factor from phantom to recon resolution
	scale = PHANTOM_SIZE / RECON_SIZE
	phantom_mid_z = clamp(round(Int, mid_z * (PHANTOM_SLICES / RECON_SLICES)), 1, PHANTOM_SLICES)

	# Find water region in central slice
	water_voxels = []
	for cy in 1:RECON_SIZE
		py = clamp(round(Int, cy * scale), 1, PHANTOM_SIZE)
		for cx in 1:RECON_SIZE
			px = clamp(round(Int, cx * scale), 1, PHANTOM_SIZE)
			if phantom.mask[px, py, phantom_mid_z] == UInt8(REGION_SOLID_WATER)
				push!(water_voxels, (cx, cy))
			end
		end
	end

	function analyze_protocol(recon, name, protocol)
		roi_HU = [μ_to_HU(recon[cx, cy, mid_z], μ_water_ref) for (cx, cy) in water_voxels]
		(
			protocol=name,
			kVp=protocol.kVp,
			mAs=protocol.mAs,
			water_HU=round(mean(roi_HU), digits=1),
			noise_HU=round(std(roi_HU), digits=1)
		)
	end

	protocols = [
		(recon_ideal, "Ideal (Reference)", (kVp=60, mAs=0)),
		(recon_low_dose, PROTOCOL_LOW_DOSE.name, PROTOCOL_LOW_DOSE),
		(recon_standard, PROTOCOL_CHEST.name, PROTOCOL_CHEST),
		(recon_high_quality, PROTOCOL_ABDOMEN.name, PROTOCOL_ABDOMEN),
	]

	[analyze_protocol(r, name, p) for (r, name, p) in protocols]
end

# ╔═╡ 5b2e4d0f-c8a9-1b5a-6d7e-0f5a2b3c4e80
md"""
## 7. Complete Pipeline: `simulate_and_reconstruct()`

For convenience, run simulation + reconstruction in one call with specific protocol:
"""

# ╔═╡ 6c3f5e1a-d9b0-2c6b-7e8f-1a6b3c4d5f91
sino_full, recon_full = simulate_and_reconstruct(phantom, geom, output_size;
	polychromatic=true,
	kVp=PROTOCOL_CHEST.kVp,
	detector=default_detector_model(I0=PROTOCOL_CHEST.I0),
	method=:fdk,
	seed=42
)

# ╔═╡ 7d4a6f2b-e0c1-3d7c-8f9a-2b7c4d5e6a02
let
	mid_z = RECON_SLICES ÷ 2
	fig = Figure(size=(800, 350))

	ax1 = Axis(fig[1, 1], xlabel="Column", ylabel="Angle",
		title="Sinogram ($(N_ANGLES) views × $(N_DETECTOR_COLS) cols)")
	heatmap!(ax1, sino_full[:, N_DETECTOR_ROWS÷2, :]'; colormap=:hot)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(),
		title="Reconstruction $(RECON_SIZE)×$(RECON_SIZE) (HU)")
	hm = heatmap!(ax2, μ_to_HU(recon_full[:, :, mid_z], μ_water_ref)';
		colormap=:grays, colorrange=(-300, 600))
	Colorbar(fig[1, 3], hm, label="HU")
	hidedecorations!(ax2)

	Label(fig[2, :],
		"GE Revolution Apex Elite | $(PROTOCOL_CHEST.kVp) kVp / $(PROTOCOL_CHEST.mAs) mAs | $(N_ANGLES) projections",
		fontsize=11)

	fig
end

# ╔═╡ 8e5b7a3c-f1d2-4e8d-9a0b-3c8d5e6f7b13
md"""
## 8. Material HU Validation

Verify reconstructed HU values for known Gammex phantom materials:
"""

# ╔═╡ 9f6c8b4d-a2e3-5f9e-0b1c-4d9e6f7a8c24
let
	# Use ideal reconstruction for clearest material comparison
	recon = recon_ideal
	mid_z = RECON_SLICES ÷ 2

	scale = PHANTOM_SIZE / RECON_SIZE
	phantom_mid_z = clamp(round(Int, mid_z * (PHANTOM_SLICES / RECON_SLICES)), 1, PHANTOM_SLICES)

	function analyze_material(region_id, name, expected_HU)
		voxels = Float32[]
		for cy in 1:RECON_SIZE
			py = clamp(round(Int, cy * scale), 1, PHANTOM_SIZE)
			for cx in 1:RECON_SIZE
				px = clamp(round(Int, cx * scale), 1, PHANTOM_SIZE)
				if phantom.mask[px, py, phantom_mid_z] == UInt8(region_id)
					push!(voxels, recon[cx, cy, mid_z])
				end
			end
		end

		if isempty(voxels)
			return nothing
		end

		HU_vals = μ_to_HU.(voxels, μ_water_ref)
		(
			material=name,
			n_voxels=length(voxels),
			measured_HU=round(mean(HU_vals), digits=0),
			expected_HU=expected_HU,
			difference=round(mean(HU_vals) - expected_HU, digits=0)
		)
	end

	materials = [
		(REGION_SOLID_WATER, "Solid Water", 0),
		(REGION_BACKGROUND, "Air/Background", -1000),
		(REGION_CA_100, "Calcium 100mg/cc", 375),
		(REGION_CA_200, "Calcium 200mg/cc", 750),
		(REGION_I_5_0, "Iodine 5mg/cc", 175),
		(REGION_I_10_0, "Iodine 10mg/cc", 350),
	]

	filter(!isnothing, [analyze_material(r, n, e) for (r, n, e) in materials])
end

# ╔═╡ 0a7d9c5e-b3f4-6a0f-1c2d-5e0f7a8b9d35
md"""
## Summary

### Scanner Configuration
- **GE Revolution Apex Elite** (FDA 510(k) K213715)
- **Detector**: $(N_DETECTOR_COLS) × $(N_DETECTOR_ROWS) (Gemstone Clarity)
- **SAD/SDD**: $(SAD) cm / $(SDD) cm
- **Projections**: $(N_ANGLES) per rotation

### Simulation Pipeline
```julia
# Create UHR phantom (1024×1024×40)
phantom = create_gammex_472(n_voxels=1024, n_slices=40)

# Create GE Apex geometry
geom = create_aquilion_one(n_angles=984, n_rows=64, n_cols=832,
                           sad=62.6, sdd=109.7)

# Simulate with specific protocol (120 kVp, 200 mAs)
sino = simulate_sinogram(phantom, geom;
    polychromatic=true,
    kVp=120,
    detector=default_detector_model(I0=2e5)  # mAs-proportional
)

# Reconstruct to clinical resolution (512×512×20)
recon = reconstruct(sino, geom, (512, 512, 20), phantom.fov)
```

### Clinical Protocols Demonstrated
| Protocol | kVp | mAs | Application |
|----------|-----|-----|-------------|
| Low-Dose Screening | 100 | 50 | Lung cancer screening |
| Standard Chest | 120 | 200 | Routine chest CT |
| High-Quality Abdomen | 120 | 400 | Contrast-enhanced studies |

---
*Generated with BasisSimulator.jl using GE Revolution Apex Elite configuration*
"""

# ╔═╡ Cell order:
# ╠═8a5d7c3e-b1f2-4a89-9c0d-3e8f5a6b7d12
# ╠═9b6e8d4f-c2a3-5b9a-0d1e-4f9a6b7c8e23
# ╟─0c7f9e5a-d3b4-6c0b-1e2f-5a0b7c8d9f34
# ╟─1d8a0f6b-e4c5-7d1c-2f3a-6b1c8d9e0a45
# ╠═2e9b1a7c-f5d6-8e2d-3a4b-7c2d9e0f1b56
# ╟─3f0c2b8d-a6e7-9f3e-4b5c-8d3e0f1a2c67
# ╠═4a1d3c9e-b7f8-0a4f-5c6d-9e4f1a2b3d78
# ╟─5b2e4d0f-c8a9-1b5a-6d7e-0f5a2b3c4e89
# ╟─0a7d9c5e-b3f4-6a0f-1c2d-5e0f7a8b9d34
# ╟─1b8e0d6f-c4a5-7b1a-2d3e-6f1a8b9c0e45
# ╠═6c3f5e1a-d9b0-2c6b-7e8f-1a6b3c4d5f90
# ╟─2c9f1e7a-d5b6-8c2b-3e4f-7a2b9c0d1f56
# ╠═3d0a2f8b-e6c7-9d3c-4f5a-8b3c0d1e2a67
# ╟─4e1b3a9c-f7d8-0e4d-5a6b-9c4d1e2f3b78
# ╟─5f2c4b0d-a8e9-1f5e-6b7c-0d5e2f3a4c89
# ╟─6a3d5c1e-b9f0-2a6f-7c8d-1e6f3a4b5d90
# ╟─7b4e6d2f-c0a1-3b7a-8d9e-2f7a4b5c6e01
# ╠═8c5f7e3a-d1b2-4c8b-9e0f-3a8b5c6d7f12
# ╟─9d6a8f4b-e2c3-5d9c-0f1a-4b9c6d7e8a23
# ╠═0e7b9a5c-f3d4-6e0d-1a2b-5c0d7e8f9b34
# ╟─1f8c0b6d-a4e5-7f1e-2b3c-6d1e8f9a0c45
# ╠═2a9d1c7e-b5f6-8a2f-3c4d-7e2f9a0b1d56
# ╟─3b0e2d8f-c6a7-9b3a-4d5e-8f3a0b1c2e67
# ╠═4c1f3e9a-d7b8-0c4b-5e6f-9a4b1c2d3f78
# ╟─5d2a4f0b-e8c9-1d5c-6f7a-0b5c2d3e4a89
# ╟─6e3b5a1c-f9d0-2e6d-7a8b-1c6d3e4f5b90
# ╠═7f4c6b2d-a0e1-3f7e-8b9c-2d7e4f5a6c01
# ╠═8a5d7c3e-b1f2-4a8f-9c0d-3e8f5a6b7d13
# ╠═9b6e8d4f-c2a3-5b9a-0d1e-4f9a6b7c8e24
# ╠═0c7f9e5a-d3b4-6c0b-1e2f-5a0b7c8d9f35
# ╟─1d8a0f6b-e4c5-7d1c-2f3a-6b1c8d9e0a46
# ╟─2e9b1a7c-f5d6-8e2d-3a4b-7c2d9e0f1b57
# ╟─3f0c2b8d-a6e7-9f3e-4b5c-8d3e0f1a2c68
# ╠═4a1d3c9e-b7f8-0a4f-5c6d-9e4f1a2b3d79
# ╟─5b2e4d0f-c8a9-1b5a-6d7e-0f5a2b3c4e80
# ╠═6c3f5e1a-d9b0-2c6b-7e8f-1a6b3c4d5f91
# ╟─7d4a6f2b-e0c1-3d7c-8f9a-2b7c4d5e6a02
# ╟─8e5b7a3c-f1d2-4e8d-9a0b-3c8d5e6f7b13
# ╠═9f6c8b4d-a2e3-5f9e-0b1c-4d9e6f7a8c24
# ╟─0a7d9c5e-b3f4-6a0f-1c2d-5e0f7a8b9d35
