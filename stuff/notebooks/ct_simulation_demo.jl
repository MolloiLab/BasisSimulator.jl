### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000001
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 00000001-0000-0000-0000-000000000002
begin
	using BasisSimulator
	using Statistics
	using CairoMakie
	CairoMakie.activate!(type = "svg")
end

# ╔═╡ 00000001-0000-0000-0000-000000000003
md"""
# BasisSimulator.jl - CT Simulation Demo

This notebook demonstrates a complete CT simulation pipeline using **BasisSimulator.jl**, a differentiable 3D cone-beam CT simulator designed for inverse problems and machine learning integration.

## Features Demonstrated

1. **Clean Monochromatic** - Ideal simulation with no physical effects
2. **Polychromatic with Noise** - Realistic beam hardening and quantum noise
3. **Full-Scale Simulation** - All physical effects (scatter, crosstalk, lag, etc.)

## Scan Protocols

We'll compare different tube voltages (kVp) and current-time products (mAs):
- **kVp**: 80, 100, 120, 140 (affects beam hardening and contrast)
- **mAs**: Low (25), Medium (100), High (400) (affects noise level)

---
"""

# ╔═╡ 00000001-0000-0000-0000-000000000004
md"""
## 1. Setup: Phantom and Geometry

We use the **Gammex Model 472** quality assurance phantom, which contains:
- Solid water background
- Calcium inserts (50, 100, 200, 400 mg/cm³)
- Bone-equivalent materials (50%, 100%)
- Adipose and brain tissue equivalents
- Acrylic and polystyrene samples
"""

# ╔═╡ 00000001-0000-0000-0000-000000000005
begin
	# Create phantom (64³ voxels for reasonable computation time)
	phantom = create_gammex_472(n_voxels=64)

	# Create scanner geometry (Canon Aquilion ONE style)
	# 360 angles, 16 detector rows, 256 detector columns
	geom = create_aquilion_one(
		n_angles=360,
		n_rows=16,
		n_cols=256,
		fov_cm=phantom.fov[1]
	)

	# Reference values
	μ_water_60keV = get_reference_μ_water(60.0)  # Water μ at 60 keV
	mid_slice = size(phantom.μ, 3) ÷ 2

	md"""
	**Phantom created:**
	- Size: $(size(phantom.μ))
	- FOV: $(phantom.fov) cm
	- Materials: $(length(unique(phantom.mask))) regions

	**Geometry created:**
	- Angles: $(geom.n_angles)
	- Detector: $(geom.n_rows) × $(geom.n_cols)
	- SAD: $(geom.SAD) cm, SDD: $(geom.SDD) cm
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000006
md"""
### Phantom Visualization
"""

# ╔═╡ 00000001-0000-0000-0000-000000000007
let
	fig = Figure(size=(900, 400))

	# Region labels
	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Region Labels")
	hm1 = heatmap!(ax1, Float64.(phantom.mask[:, :, mid_slice])';
		colormap=:tab20, colorrange=(0, 20))
	hidedecorations!(ax1)

	# Attenuation values (μ)
	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Attenuation (μ, cm⁻¹)")
	hm2 = heatmap!(ax2, phantom.μ[:, :, mid_slice]';
		colormap=:viridis, colorrange=(0, 0.5))
	Colorbar(fig[1, 3], hm2)
	hidedecorations!(ax2)

	# HU values
	phantom_HU = μ_to_HU(phantom.μ, μ_water_60keV)
	ax3 = Axis(fig[1, 4], aspect=DataAspect(), title="Hounsfield Units")
	hm3 = heatmap!(ax3, phantom_HU[:, :, mid_slice]';
		colormap=:grays, colorrange=(-200, 400))
	Colorbar(fig[1, 5], hm3, label="HU")
	hidedecorations!(ax3)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000008
md"""
---
## 2. Clean Monochromatic Simulation

The simplest simulation: monochromatic X-rays (60 keV) with no noise or physical effects.

This represents the **ideal case** - what we'd see with:
- Single energy X-rays
- Perfect detector (100% efficiency, no noise)
- No scatter
- No beam hardening
"""

# ╔═╡ 00000001-0000-0000-0000-000000000009
begin
	# Forward projection (Siddon's ray-driven method)
	sino_mono = forward_project(phantom, geom)

	# FDK reconstruction
	recon_mono = fdk_reconstruct(sino_mono, geom, size(phantom.μ), phantom.fov)
	recon_mono_HU = μ_to_HU(recon_mono, μ_water_60keV)

	md"""
	**Monochromatic simulation complete:**
	- Sinogram size: $(size(sino_mono))
	- Reconstruction size: $(size(recon_mono))
	- HU range: $(round(minimum(recon_mono_HU), digits=1)) to $(round(maximum(recon_mono_HU), digits=1))
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000010
let
	fig = Figure(size=(900, 350))

	# Sinogram (central row)
	ax1 = Axis(fig[1, 1], xlabel="Detector Column", ylabel="Angle",
		title="Sinogram (Central Row)")
	mid_row = size(sino_mono, 2) ÷ 2
	heatmap!(ax1, sino_mono[:, mid_row, :]'; colormap=:hot)

	# Reconstruction
	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Monochromatic Recon")
	hm2 = heatmap!(ax2, recon_mono_HU[:, :, mid_slice]';
		colormap=:grays, colorrange=(-200, 400))
	Colorbar(fig[1, 3], hm2, label="HU")
	hidedecorations!(ax2)

	# Difference from ground truth
	diff = recon_mono_HU[:, :, mid_slice] .- μ_to_HU(phantom.μ[:, :, mid_slice], μ_water_60keV)
	ax3 = Axis(fig[1, 4], aspect=DataAspect(), title="Difference from Ground Truth")
	hm3 = heatmap!(ax3, diff'; colormap=:RdBu, colorrange=(-100, 100))
	Colorbar(fig[1, 5], hm3, label="ΔHU")
	hidedecorations!(ax3)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000011
md"""
---
## 3. Polychromatic Simulation with Different kVp

Real X-ray tubes produce a **spectrum of energies**. Lower energies are preferentially absorbed (beam hardening), causing:
- Cupping artifacts in uniform regions
- Dark bands between dense objects
- Apparent contrast changes

We'll compare **80, 100, 120, and 140 kVp** spectra.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000012
begin
	# Define kVp values to compare
	kVp_values = [80, 100, 120, 140]

	# Create polychromatic projectors for each kVp
	projectors = Dict{Int, Any}()
	sino_poly = Dict{Int, Array}()
	recon_poly = Dict{Int, Array}()
	recon_poly_HU = Dict{Int, Array}()

	for kVp in kVp_values
		# Create projector with 20 energy bins
		projectors[kVp] = create_polychromatic_projector(phantom, geom, kVp; n_bins=20)

		# Forward projection
		sino_poly[kVp] = forward_project_polychromatic(phantom, projectors[kVp])

		# Reconstruction
		recon_poly[kVp] = fdk_reconstruct(sino_poly[kVp], geom, size(phantom.μ), phantom.fov)

		# Convert to HU using effective water μ for this spectrum
		μ_water_eff = get_effective_μ_water(projectors[kVp])
		recon_poly_HU[kVp] = μ_to_HU(recon_poly[kVp], μ_water_eff)
	end

	md"""
	**Polychromatic simulations complete for kVp:** $(kVp_values)
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000013
let
	fig = Figure(size=(1000, 800))

	for (i, kVp) in enumerate(kVp_values)
		row = (i-1) ÷ 2 + 1
		col = (i-1) % 2 + 1

		ax = Axis(fig[row, col], aspect=DataAspect(), title="$(kVp) kVp")
		hm = heatmap!(ax, recon_poly_HU[kVp][:, :, mid_slice]';
			colormap=:grays, colorrange=(-200, 400))
		hidedecorations!(ax)

		if col == 2
			Colorbar(fig[row, 3], hm, label="HU")
		end
	end

	Label(fig[0, :], "Polychromatic Reconstructions (Beam Hardening Effect)", fontsize=16)
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000014
md"""
### Beam Hardening Analysis

Lower kVp shows **more severe cupping** due to stronger beam hardening. Let's visualize this with a profile through the center.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000015
let
	fig = Figure(size=(800, 400))

	# Central profile
	center = size(recon_poly_HU[120], 1) ÷ 2

	ax = Axis(fig[1, 1], xlabel="Position (pixels)", ylabel="HU",
		title="Central Profile - Beam Hardening Comparison")

	colors = [:red, :orange, :blue, :purple]
	for (i, kVp) in enumerate(kVp_values)
		profile = recon_poly_HU[kVp][center, :, mid_slice]
		lines!(ax, profile, label="$(kVp) kVp", color=colors[i], linewidth=2)
	end

	# Ground truth
	gt_profile = μ_to_HU(phantom.μ[center, :, mid_slice], μ_water_60keV)
	lines!(ax, gt_profile, label="Ground Truth", color=:black, linewidth=2, linestyle=:dash)

	axislegend(ax, position=:rt)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000016
md"""
---
## 4. Noise Simulation with Different mAs

**mAs (milliampere-seconds)** controls the number of X-ray photons:
- Higher mAs = more photons = less noise = higher dose
- Lower mAs = fewer photons = more noise = lower dose

We model this via **I₀** (incident photon count per detector pixel).

| mAs Level | I₀ Photons | Clinical Use |
|-----------|------------|--------------|
| Low (25)  | 1×10⁴     | Dose reduction protocols |
| Medium (100) | 5×10⁴  | Standard protocols |
| High (400) | 2×10⁵    | High-quality imaging |
"""

# ╔═╡ 00000001-0000-0000-0000-000000000017
begin
	# Define mAs levels (via I0)
	mAs_configs = [
		(name="Low (25 mAs)", I0=1e4),
		(name="Medium (100 mAs)", I0=5e4),
		(name="High (400 mAs)", I0=2e5)
	]

	# Use 120 kVp polychromatic simulation as base
	sino_base = sino_poly[120]

	# Apply noise for each mAs level
	recon_noisy = Dict{String, Array}()
	recon_noisy_HU = Dict{String, Array}()

	for cfg in mAs_configs
		# Create detector model with appropriate photon count
		detector = default_detector_model(
			blur_fwhm=0.8,          # Slight detector blur
			I0=cfg.I0,              # Photon count (controls noise)
			electronic_noise_std=10.0,
			seed=42
		)

		# Apply noise
		sino_noisy = apply_detector_model(sino_base, detector)

		# Reconstruct
		recon_noisy[cfg.name] = fdk_reconstruct(sino_noisy, geom, size(phantom.μ), phantom.fov)
		recon_noisy_HU[cfg.name] = μ_to_HU(recon_noisy[cfg.name], get_effective_μ_water(projectors[120]))
	end

	md"""
	**Noise simulations complete for mAs levels:** $(length(mAs_configs))
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000018
let
	fig = Figure(size=(1100, 350))

	for (i, cfg) in enumerate(mAs_configs)
		ax = Axis(fig[1, i], aspect=DataAspect(), title=cfg.name)
		hm = heatmap!(ax, recon_noisy_HU[cfg.name][:, :, mid_slice]';
			colormap=:grays, colorrange=(-200, 400))
		hidedecorations!(ax)

		if i == length(mAs_configs)
			Colorbar(fig[1, i+1], hm, label="HU")
		end
	end

	Label(fig[0, :], "Noise Comparison at 120 kVp", fontsize=16)
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000019
md"""
### Noise Analysis

Let's measure the **noise standard deviation** in the water background region.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000020
let
	# Get water region mask
	water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
	water_mask_2d = water_mask[:, :, mid_slice]

	fig = Figure(size=(600, 400))

	ax = Axis(fig[1, 1], xlabel="mAs Level", ylabel="Noise SD (HU)",
		title="Noise vs. Dose (Water Region)",
		xticks=(1:3, [cfg.name for cfg in mAs_configs]))

	noise_values = Float64[]
	for cfg in mAs_configs
		slice_data = recon_noisy_HU[cfg.name][:, :, mid_slice]
		noise_sd = std(slice_data[water_mask_2d])
		push!(noise_values, noise_sd)
	end

	barplot!(ax, 1:3, noise_values, color=:steelblue)

	# Add text labels
	for (i, nv) in enumerate(noise_values)
		text!(ax, i, nv + 2, text="$(round(nv, digits=1))",
			align=(:center, :bottom), fontsize=12)
	end

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000021
md"""
---
## 5. Full-Scale Simulation

Now let's combine **all physical effects** for maximum realism:

1. **Polychromatic X-rays** → Beam hardening
2. **Scatter** → Cupping, reduced contrast
3. **Detector noise** → Quantum noise (Poisson) + electronic noise (Gaussian)
4. **Detector blur** → Reduced spatial resolution
5. **Crosstalk** → Signal bleeding between pixels
6. **Detector lag** → Afterglow from previous views
7. **Bowtie filter** → Dose modulation, beam hardening
8. **Fill factor** → Active detector area

We'll compare a **low-dose** (80 kVp, low mAs) vs **high-quality** (120 kVp, high mAs) protocol.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000022
function full_simulation(phantom, geom, kVp, I0; seed=42)
	"""
	Run complete CT simulation with all physical effects.

	Returns: (sinogram, reconstruction_HU, effective_μ_water)
	"""

	# 1. Polychromatic forward projection
	projector = create_polychromatic_projector(phantom, geom, kVp; n_bins=20)
	sino = forward_project_polychromatic(phantom, projector)

	# 2. Add scatter (XCIST-style convolution model)
	scatter_model = default_scatter_model(scale_factor=0.8)
	sino = add_scatter(sino, scatter_model)

	# 3. Apply bowtie filter (medium body)
	bowtie = bowtie_filter_medium_body()
	sino = apply_bowtie_filter(sino, bowtie, geom)

	# 4. Apply detector effects
	# 4a. Optical crosstalk (light spreading in scintillator)
	opt_xt = optical_crosstalk_typical()
	sino = apply_optical_crosstalk(sino, opt_xt)

	# 4b. Electronic crosstalk
	elec_xt = crosstalk_low()
	sino = apply_crosstalk(sino, elec_xt)

	# 4c. Detector lag (afterglow)
	lag = lag_gadox()
	sino = apply_lag(sino, lag)

	# 4d. Fill factor
	ff = fill_factor_standard()
	sino = apply_fill_factor(sino, ff)

	# 5. Apply noise
	detector = default_detector_model(
		blur_fwhm=0.8,
		I0=I0,
		electronic_noise_std=10.0,
		seed=seed
	)
	sino = apply_detector_model(sino, detector)

	# 6. Reconstruct with soft kernel and Tang weighting
	recon = fdk_reconstruct(sino, geom, size(phantom.μ), phantom.fov;
		kernel=kernel_soft(), tang_order=5)

	# 7. Apply beam hardening correction
	bhc = calibrate_water_bhc_simple(kVp)
	# Note: BHC is typically applied to sinogram before recon
	# Here we just return the reconstruction

	μ_water_eff = get_effective_μ_water(projector)
	recon_HU = μ_to_HU(recon, μ_water_eff)

	return sino, recon_HU, μ_water_eff
end

# ╔═╡ 00000001-0000-0000-0000-000000000023
begin
	# Low-dose protocol: 80 kVp, low mAs
	sino_lowdose, recon_lowdose_HU, _ = full_simulation(phantom, geom, 80, 1e4; seed=42)

	# Standard protocol: 120 kVp, medium mAs
	sino_standard, recon_standard_HU, _ = full_simulation(phantom, geom, 120, 5e4; seed=42)

	# High-quality protocol: 140 kVp, high mAs
	sino_highqual, recon_highqual_HU, _ = full_simulation(phantom, geom, 140, 2e5; seed=42)

	md"""
	**Full simulations complete:**
	- Low-dose: 80 kVp, ~25 mAs equivalent
	- Standard: 120 kVp, ~100 mAs equivalent
	- High-quality: 140 kVp, ~400 mAs equivalent
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000024
let
	fig = Figure(size=(1100, 400))

	protocols = [
		("Low-Dose (80 kVp)", recon_lowdose_HU),
		("Standard (120 kVp)", recon_standard_HU),
		("High-Quality (140 kVp)", recon_highqual_HU)
	]

	for (i, (name, recon)) in enumerate(protocols)
		ax = Axis(fig[1, i], aspect=DataAspect(), title=name)
		hm = heatmap!(ax, recon[:, :, mid_slice]';
			colormap=:grays, colorrange=(-200, 400))
		hidedecorations!(ax)

		if i == length(protocols)
			Colorbar(fig[1, i+1], hm, label="HU")
		end
	end

	Label(fig[0, :], "Full-Scale Simulations with All Physical Effects", fontsize=16)
	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000025
md"""
### Image Quality Metrics

Let's quantify the image quality for each protocol:
"""

# ╔═╡ 00000001-0000-0000-0000-000000000026
let
	water_mask = get_region_mask(phantom, REGION_SOLID_WATER)
	water_mask_2d = water_mask[:, :, mid_slice]

	# Get calcium insert for contrast measurement
	ca_mask = get_region_mask(phantom, REGION_CA_100)
	ca_mask_2d = ca_mask[:, :, mid_slice]

	protocols = [
		("Low-Dose (80 kVp)", recon_lowdose_HU),
		("Standard (120 kVp)", recon_standard_HU),
		("High-Quality (140 kVp)", recon_highqual_HU)
	]

	fig = Figure(size=(900, 350))

	# Noise comparison
	ax1 = Axis(fig[1, 1], xlabel="Protocol", ylabel="Noise SD (HU)",
		title="Image Noise",
		xticks=(1:3, [p[1] for p in protocols]))

	noise_vals = [std(p[2][:, :, mid_slice][water_mask_2d]) for p in protocols]
	barplot!(ax1, 1:3, noise_vals, color=[:red, :blue, :green])

	for (i, nv) in enumerate(noise_vals)
		text!(ax1, i, nv + 1, text="$(round(nv, digits=1))",
			align=(:center, :bottom), fontsize=11)
	end

	# Contrast (Ca100 - Water)
	ax2 = Axis(fig[1, 2], xlabel="Protocol", ylabel="Contrast (HU)",
		title="Calcium-Water Contrast",
		xticks=(1:3, [p[1] for p in protocols]))

	contrast_vals = Float64[]
	for p in protocols
		if sum(ca_mask_2d) > 0
			ca_mean = mean(p[2][:, :, mid_slice][ca_mask_2d])
			water_mean = mean(p[2][:, :, mid_slice][water_mask_2d])
			push!(contrast_vals, ca_mean - water_mean)
		else
			push!(contrast_vals, 0.0)
		end
	end

	barplot!(ax2, 1:3, contrast_vals, color=[:red, :blue, :green])

	for (i, cv) in enumerate(contrast_vals)
		text!(ax2, i, cv + 5, text="$(round(cv, digits=0))",
			align=(:center, :bottom), fontsize=11)
	end

	# CNR (Contrast-to-Noise Ratio)
	ax3 = Axis(fig[1, 3], xlabel="Protocol", ylabel="CNR",
		title="Contrast-to-Noise Ratio",
		xticks=(1:3, [p[1] for p in protocols]))

	cnr_vals = contrast_vals ./ noise_vals
	barplot!(ax3, 1:3, cnr_vals, color=[:red, :blue, :green])

	for (i, cnr) in enumerate(cnr_vals)
		text!(ax3, i, cnr + 0.5, text="$(round(cnr, digits=1))",
			align=(:center, :bottom), fontsize=11)
	end

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000027
md"""
---
## 6. Summary

### Key Observations

1. **Beam Hardening**: Lower kVp (80) shows more cupping artifacts than higher kVp (140)

2. **Noise vs. Dose**:
   - Low mAs → High noise, low dose
   - High mAs → Low noise, high dose
   - Trade-off managed by ALARA principle

3. **Full-Scale Simulation**: Combining all effects produces realistic images matching clinical CT

### Physical Effects Included

| Effect | Module | Impact |
|--------|--------|--------|
| Polychromatic | `Polychromatic.jl` | Beam hardening, cupping |
| Scatter | `Scatter.jl` | Reduced contrast, cupping |
| Quantum noise | `DetectorNoise.jl` | Poisson noise |
| Electronic noise | `DetectorNoise.jl` | Gaussian noise |
| Detector blur | `DetectorNoise.jl` | MTF degradation |
| Crosstalk (optical) | `Crosstalk.jl` | Light spreading |
| Crosstalk (electronic) | `Crosstalk.jl` | Signal coupling |
| Detector lag | `DetectorLag.jl` | Afterglow |
| Bowtie filter | `BowtieFilter.jl` | Dose modulation |
| Fill factor | `FillFactor.jl` | Active area ratio |
| Tang weighting | `FDK.jl` | Cone-beam artifacts |

### Reactant/Enzyme Compatibility

BasisSimulator.jl is designed for differentiability. The forward model can be compiled with Reactant for:
- GPU acceleration
- Automatic differentiation for inverse problems
- Machine learning integration

```julia
using Reactant

# Compile forward projection
@compile forward_project(phantom, geom)

# Use gradients for optimization
∇sino = gradient(loss_fn, sino)
```
"""

# ╔═╡ 00000001-0000-0000-0000-000000000028
md"""
---
## Appendix: API Reference

### Quick Start
```julia
using BasisSimulator

# Create phantom and geometry
phantom = create_gammex_472(n_voxels=64)
geom = create_aquilion_one(n_angles=360)

# Monochromatic
sino = forward_project(phantom, geom)
recon = fdk_reconstruct(sino, geom; n_voxels=64)

# Polychromatic
proj = create_polychromatic_projector(phantom, geom, 120)
sino = forward_project_polychromatic(phantom, proj)

# Add effects
sino = add_scatter(sino, default_scatter_model())
sino = apply_detector_model(sino, default_detector_model(I0=5e4))
```

### Key Functions

| Function | Description |
|----------|-------------|
| `forward_project` | Monochromatic forward projection |
| `forward_project_polychromatic` | Polychromatic (beam hardening) |
| `fdk_reconstruct` | Cone-beam FDK reconstruction |
| `add_scatter` | XCIST-style scatter model |
| `apply_detector_model` | Noise + blur |
| `apply_bowtie_filter` | Bowtie filter attenuation |
| `apply_optical_crosstalk` | Light spreading |
| `apply_lag` | Detector afterglow |
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000001
# ╟─00000001-0000-0000-0000-000000000002
# ╟─00000001-0000-0000-0000-000000000003
# ╟─00000001-0000-0000-0000-000000000004
# ╟─00000001-0000-0000-0000-000000000005
# ╟─00000001-0000-0000-0000-000000000006
# ╟─00000001-0000-0000-0000-000000000007
# ╟─00000001-0000-0000-0000-000000000008
# ╟─00000001-0000-0000-0000-000000000009
# ╟─00000001-0000-0000-0000-000000000010
# ╟─00000001-0000-0000-0000-000000000011
# ╟─00000001-0000-0000-0000-000000000012
# ╟─00000001-0000-0000-0000-000000000013
# ╟─00000001-0000-0000-0000-000000000014
# ╟─00000001-0000-0000-0000-000000000015
# ╟─00000001-0000-0000-0000-000000000016
# ╟─00000001-0000-0000-0000-000000000017
# ╟─00000001-0000-0000-0000-000000000018
# ╟─00000001-0000-0000-0000-000000000019
# ╟─00000001-0000-0000-0000-000000000020
# ╟─00000001-0000-0000-0000-000000000021
# ╟─00000001-0000-0000-0000-000000000022
# ╟─00000001-0000-0000-0000-000000000023
# ╟─00000001-0000-0000-0000-000000000024
# ╟─00000001-0000-0000-0000-000000000025
# ╟─00000001-0000-0000-0000-000000000026
# ╟─00000001-0000-0000-0000-000000000027
# ╟─00000001-0000-0000-0000-000000000028
