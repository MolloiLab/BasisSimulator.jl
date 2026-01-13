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
	using Reactant
	using Statistics
	using CairoMakie
	CairoMakie.activate!(type = "svg")
end

# ╔═╡ 00000001-0000-0000-0000-000000000003
md"""
# Realistic CT Simulation with BasisSimulator.jl

This notebook demonstrates physically realistic CT simulation using:

1. **High-resolution phantom** - Physical objects have atoms at higher resolution than CT voxels
2. **Clinical scanner configuration** - GE Revolution Apex Elite (FDA 510(k): K213715)
3. **Multiple acquisition protocols** - Different kVp/mAs combinations
4. **Full physics pipeline** - Polychromatic X-rays, filters, scatter, noise
5. **Multiple reconstruction methods** - FDK (analytical), SIRT, CGLS (iterative)
6. **Reactant compilation** - XLA compilation for performance

---
"""

# ╔═╡ 00000001-0000-0000-0000-000000000004
md"""
## 1. Configuration

Define the simulation parameters and output resolution.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000005
begin
	# High-resolution phantom (physical resolution)
	# This represents the "true" physical object at atomic resolution
	PHANTOM_SIZE = 512  # 512x512 in-plane (higher than recon for physical realism)
	PHANTOM_SLICES = 32  # z-direction

	# Output reconstruction size (clinical CT resolution)
	RECON_SIZE = 256  # 256x256 in-plane (downsampled for demo speed)
	RECON_SLICES = 16  # z-direction

	# Scanner parameters
	N_ANGLES = 360  # Projection angles (reduced for demo)
	N_DETECTOR_ROWS = 32  # Detector rows
	N_DETECTOR_COLS = 512  # Detector columns

	# Iterative reconstruction parameters
	SIRT_ITERATIONS = 20
	CGLS_ITERATIONS = 10

	md"""
	**Configuration:**
	- Phantom resolution: $(PHANTOM_SIZE) x $(PHANTOM_SIZE) x $(PHANTOM_SLICES)
	- Output resolution: $(RECON_SIZE) x $(RECON_SIZE) x $(RECON_SLICES)
	- Scanner: $(N_ANGLES) angles, $(N_DETECTOR_ROWS) x $(N_DETECTOR_COLS) detector
	- SIRT iterations: $(SIRT_ITERATIONS), CGLS iterations: $(CGLS_ITERATIONS)
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000006
md"""
## 2. High-Resolution Phantom

Create a Gammex 472 phantom at high resolution. This represents the physical object
with material properties at a resolution finer than CT voxels - simulating how real
atoms and materials exist independently of the imaging resolution.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000007
begin
	println("Creating high-resolution phantom...")
	# z_cm controls the number of slices (n_z = n_voxels * z_cm / fov_cm)
	# For 32 slices with 512 voxels and 35cm FOV: z_cm = 32 * 35 / 512 ≈ 2.19 cm
	z_extent = PHANTOM_SLICES * 35.0 / PHANTOM_SIZE
	phantom = create_gammex_472(n_voxels=PHANTOM_SIZE, z_cm=z_extent)

	# Reference μ for HU conversion
	μ_water_ref = get_reference_μ_water(60.0)

	md"""
	**High-resolution phantom created:**
	- Size: $(size(phantom.μ))
	- FOV: $(round.(phantom.fov, digits=2)) cm
	- Voxel size: $(round.(phantom.voxel_size .* 10, digits=3)) mm
	- Materials: $(length(unique(phantom.mask))) distinct regions
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000008
let
	mid_slice = size(phantom.μ, 3) ÷ 2
	fig = Figure(size=(800, 350))

	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Material Regions")
	hm1 = heatmap!(ax1, Float64.(phantom.mask[:, :, mid_slice])'; colormap=:tab20)
	hidedecorations!(ax1)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Attenuation (μ, cm⁻¹)")
	hm2 = heatmap!(ax2, phantom.μ[:, :, mid_slice]'; colormap=:viridis, colorrange=(0, 0.5))
	Colorbar(fig[1, 3], hm2)
	hidedecorations!(ax2)

	phantom_HU = μ_to_HU(phantom.μ[:, :, mid_slice], μ_water_ref)
	ax3 = Axis(fig[1, 4], aspect=DataAspect(), title="Ground Truth (HU)")
	hm3 = heatmap!(ax3, phantom_HU'; colormap=:grays, colorrange=(-200, 500))
	Colorbar(fig[1, 5], hm3, label="HU")
	hidedecorations!(ax3)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000009
md"""
## 3. Scanner Configuration

Use the **GE Revolution Apex Elite** scanner specification from FDA 510(k) K213715.
All geometric parameters are sourced from official documentation.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000010
begin
	# Load clinical scanner specification
	scanner_spec = GERevolutionApexElite()

	# Create geometry from scanner spec
	geom = create_geometry(scanner_spec;
		n_angles=N_ANGLES,
		n_rows=N_DETECTOR_ROWS,
		n_cols=N_DETECTOR_COLS,
		fov_cm=phantom.fov[1]
	)

	md"""
	**GE Revolution Apex Elite (K213715):**
	- Source-Axis Distance (SAD): $(geom.SAD) cm ($(geom.SAD * 10) mm)
	- Source-Detector Distance (SDD): $(geom.SDD) cm ($(geom.SDD * 10) mm)
	- Magnification: $(round(geom.SDD / geom.SAD, digits=3))
	- Detector: $(geom.n_cols) x $(geom.n_rows) elements
	- Angles: $(geom.n_angles) projections
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000011
md"""
## 4. Acquisition Protocols

Define three clinical protocols with different kVp and mAs settings:

| Protocol | kVp | mAs | I₀ (photons) | Use Case |
|----------|-----|-----|--------------|----------|
| Low Dose | 80 | 100 | 1×10⁵ | Pediatric, screening |
| Standard | 120 | 200 | 2×10⁵ | Routine diagnostic |
| High Dose | 140 | 400 | 4×10⁵ | Obese patients, high contrast |

Higher mAs = more photons = lower quantum noise.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000012
begin
	# Protocol definitions
	protocols = [
		(name="Low Dose", kvp=80, mas=100, I0=1e5),
		(name="Standard", kvp=120, mas=200, I0=2e5),
		(name="High Dose", kvp=140, mas=400, I0=4e5),
	]

	md"""
	**Protocols defined:** $(length(protocols)) acquisition settings
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000013
md"""
## 5. Pre-compute Geometry for Compilation

Pre-compute all geometry indices once for efficient Reactant compilation.
This separates the geometry computation from the XLA-traceable operations.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000014
begin
	println("Pre-computing projection geometry...")

	# Output volume configuration
	# Use same z-extent as phantom, just at lower resolution
	n_phantom_slices = size(phantom.μ, 3)
	output_size = (RECON_SIZE, RECON_SIZE, RECON_SLICES)
	output_fov = (phantom.fov[1], phantom.fov[2], phantom.fov[3])  # Same FOV, different resolution
	output_voxel_size = output_fov ./ output_size

	# Pre-compute geometry for forward projection (uses phantom resolution)
	proj_geom = precompute_projection_geometry(
		geom, phantom.fov, phantom.voxel_size, size(phantom.μ)
	)

	# Pre-compute geometry for backprojection (uses output resolution)
	bp_geom = precompute_backprojection_geometry(
		geom, output_size, output_fov
	)

	# Pre-compute geometry for iterative recon (output resolution)
	proj_geom_recon = precompute_projection_geometry(
		geom, output_fov, output_voxel_size, output_size
	)

	md"""
	**Geometry pre-computed:**
	- Forward projection: $(size(phantom.μ)) → sinogram
	- Backprojection: sinogram → $(output_size)
	- Iterative projection: $(output_size) ↔ sinogram
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000015
md"""
## 6. Compile Forward Projection

Use Reactant's `@compile` to JIT-compile the forward projection for maximum performance.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000016
begin
	println("Compiling forward projection...")

	# Convert phantom to Reactant array
	phantom_ra = Reactant.to_rarray(Float32.(phantom.μ))

	# Compile forward projection
	compiled_project = @compile project_volume(phantom_ra, proj_geom)

	# Run compiled forward projection
	sino_mono_ra = compiled_project(phantom_ra, proj_geom)
	sino_mono = Array(sino_mono_ra)

	md"""
	**Forward projection compiled and executed:**
	- Input: $(size(phantom.μ)) phantom volume
	- Output: $(size(sino_mono)) sinogram
	- Max projection value: $(round(maximum(sino_mono), digits=3)) cm⁻¹·cm
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000017
let
	mid_row = N_DETECTOR_ROWS ÷ 2
	fig = Figure(size=(700, 300))

	ax1 = Axis(fig[1, 1], xlabel="Detector Column", ylabel="Angle", title="Sinogram (Row $(mid_row))")
	hm1 = heatmap!(ax1, sino_mono[:, mid_row, :]'; colormap=:hot)

	ax2 = Axis(fig[1, 2], xlabel="Detector Column", ylabel="Detector Row", title="Projection (Angle 1)")
	hm2 = heatmap!(ax2, sino_mono[:, :, 1]'; colormap=:hot)
	Colorbar(fig[1, 3], hm2, label="Line integral")

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000018
md"""
## 7. Physical Effects Pipeline

For each protocol, apply the full physics chain:

```
Phantom → Polychromatic Forward Project → Flat Filter → Bowtie Filter
        → Scatter → Quantum Noise → Realistic Sinogram
```
"""

# ╔═╡ 00000001-0000-0000-0000-000000000019
begin
	# Physics models (shared across protocols)
	flat_filter = flat_filter_al_cu(2.5, 0.1)  # 2.5mm Al + 0.1mm Cu
	bowtie = bowtie_filter_medium_body()
	scatter_model = default_scatter_model(scale_factor=1.0)

	# Store sinograms for each protocol
	sinograms = Dict{String, Array{Float32,3}}()

	for protocol in protocols
		println("Processing $(protocol.name) ($(protocol.kvp) kVp, $(protocol.mas) mAs)...")

		# Create polychromatic projector
		projector = create_polychromatic_projector(
			phantom, geom, protocol.kvp; n_bins=20
		)

		# Forward project with polychromatic spectrum
		sino = forward_project_polychromatic(phantom, projector)

		# Apply flat filter (inherent filtration)
		sino = apply_flat_filter(sino, flat_filter, geom)

		# Apply bowtie filter
		sino = apply_bowtie_filter(sino, bowtie, geom)

		# Add scatter
		sino = add_scatter(sino, scatter_model)

		# Add quantum noise based on mAs (I0)
		detector_model = default_detector_model(
			blur_fwhm=0.0,
			I0=protocol.I0,
			electronic_noise_std=20.0,
			seed=42
		)
		sino = add_quantum_noise(sino, detector_model)
		sino = add_electronic_noise(sino, detector_model)

		# Store result
		sinograms[protocol.name] = Float32.(sino)
	end

	md"""
	**Physical effects applied to all protocols:**
	- Polychromatic X-ray spectrum (20 energy bins)
	- Flat filter: 2.5mm Al + 0.1mm Cu
	- Bowtie filter: Medium body
	- Scatter: Convolution-based model
	- Quantum noise: Poisson distribution (I₀ varies by protocol)
	- Electronic noise: σ = 20
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000020
let
	mid_row = N_DETECTOR_ROWS ÷ 2
	fig = Figure(size=(900, 300))

	for (i, protocol) in enumerate(protocols)
		sino = sinograms[protocol.name]
		ax = Axis(fig[1, i], xlabel="Column", ylabel="Angle",
			title="$(protocol.name)\n$(protocol.kvp) kVp, $(protocol.mas) mAs")
		hm = heatmap!(ax, sino[:, mid_row, :]'; colormap=:hot,
			colorrange=(0, maximum(sino[:, mid_row, :])))
		if i == length(protocols)
			Colorbar(fig[1, length(protocols)+1], hm, label="Projection")
		end
	end

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000021
md"""
## 8. FDK Reconstruction

Analytical filtered back-projection reconstruction. Fast but may have artifacts
from cone-beam geometry and limited angle sampling.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000022
begin
	println("Compiling FDK reconstruction...")

	# Test sinogram for compilation
	test_sino = Float32.(sinograms[protocols[1].name])
	test_sino_ra = Reactant.to_rarray(test_sino)

	# Compile FDK
	compiled_fdk = @compile fdk_reconstruct_xla(test_sino_ra, geom, bp_geom)

	# Reconstruct all protocols with FDK
	recon_fdk = Dict{String, Array{Float32,3}}()

	for protocol in protocols
		sino_ra = Reactant.to_rarray(sinograms[protocol.name])
		recon_ra = compiled_fdk(sino_ra, geom, bp_geom)
		recon_fdk[protocol.name] = Array(recon_ra)
	end

	md"""
	**FDK reconstruction complete for all protocols.**
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000023
let
	mid_slice = RECON_SLICES ÷ 2
	fig = Figure(size=(900, 300))

	for (i, protocol) in enumerate(protocols)
		recon = recon_fdk[protocol.name]
		recon_HU = μ_to_HU(recon[:, :, mid_slice], μ_water_ref)

		ax = Axis(fig[1, i], aspect=DataAspect(),
			title="FDK: $(protocol.name)")
		hm = heatmap!(ax, recon_HU'; colormap=:grays, colorrange=(-200, 500))
		hidedecorations!(ax)

		if i == length(protocols)
			Colorbar(fig[1, length(protocols)+1], hm, label="HU")
		end
	end

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000024
md"""
## 9. SIRT Reconstruction

Simultaneous Iterative Reconstruction Technique. More robust to noise than FDK,
but slower convergence. Uses row-sum and column-sum normalization.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000025
begin
	println("Running SIRT reconstruction ($(SIRT_ITERATIONS) iterations)...")

	recon_sirt = Dict{String, Array{Float32,3}}()

	for protocol in protocols
		println("  SIRT: $(protocol.name)...")
		result = sirt_reconstruct(
			sinograms[protocol.name],
			proj_geom_recon,
			bp_geom;
			n_iterations=SIRT_ITERATIONS,
			relaxation=1.0f0,
			verbose=false
		)
		recon_sirt[protocol.name] = result.volume
	end

	md"""
	**SIRT reconstruction complete for all protocols.**
	- Iterations: $(SIRT_ITERATIONS)
	- Relaxation: 1.0
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000026
let
	mid_slice = RECON_SLICES ÷ 2
	fig = Figure(size=(900, 300))

	for (i, protocol) in enumerate(protocols)
		recon = recon_sirt[protocol.name]
		recon_HU = μ_to_HU(recon[:, :, mid_slice], μ_water_ref)

		ax = Axis(fig[1, i], aspect=DataAspect(),
			title="SIRT: $(protocol.name)")
		hm = heatmap!(ax, recon_HU'; colormap=:grays, colorrange=(-200, 500))
		hidedecorations!(ax)

		if i == length(protocols)
			Colorbar(fig[1, length(protocols)+1], hm, label="HU")
		end
	end

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000027
md"""
## 10. CGLS Reconstruction

Conjugate Gradient Least Squares. Faster convergence than SIRT, but may amplify
noise if run for too many iterations (semi-convergent behavior).
"""

# ╔═╡ 00000001-0000-0000-0000-000000000028
begin
	println("Running CGLS reconstruction ($(CGLS_ITERATIONS) iterations)...")

	recon_cgls = Dict{String, Array{Float32,3}}()

	for protocol in protocols
		println("  CGLS: $(protocol.name)...")
		result = cgls_reconstruct(
			sinograms[protocol.name],
			proj_geom_recon,
			bp_geom;
			n_iterations=CGLS_ITERATIONS,
			verbose=false
		)
		recon_cgls[protocol.name] = result.volume
	end

	md"""
	**CGLS reconstruction complete for all protocols.**
	- Iterations: $(CGLS_ITERATIONS)
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000029
let
	mid_slice = RECON_SLICES ÷ 2
	fig = Figure(size=(900, 300))

	for (i, protocol) in enumerate(protocols)
		recon = recon_cgls[protocol.name]
		recon_HU = μ_to_HU(recon[:, :, mid_slice], μ_water_ref)

		ax = Axis(fig[1, i], aspect=DataAspect(),
			title="CGLS: $(protocol.name)")
		hm = heatmap!(ax, recon_HU'; colormap=:grays, colorrange=(-200, 500))
		hidedecorations!(ax)

		if i == length(protocols)
			Colorbar(fig[1, length(protocols)+1], hm, label="HU")
		end
	end

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000030
md"""
## 11. Comprehensive Comparison

Compare all reconstruction methods across all protocols.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000031
let
	mid_slice = RECON_SLICES ÷ 2
	fig = Figure(size=(1000, 900))

	methods = ["FDK", "SIRT", "CGLS"]
	recons = [recon_fdk, recon_sirt, recon_cgls]

	for (j, method) in enumerate(methods)
		for (i, protocol) in enumerate(protocols)
			recon = recons[j][protocol.name]
			recon_HU = μ_to_HU(recon[:, :, mid_slice], μ_water_ref)

			ax = Axis(fig[j, i], aspect=DataAspect())
			hm = heatmap!(ax, recon_HU'; colormap=:grays, colorrange=(-200, 500))
			hidedecorations!(ax)

			if j == 1
				ax.title = "$(protocol.name)\n$(protocol.kvp) kVp"
			end
			if i == 1
				ax.ylabel = method
			end
		end
	end

	Colorbar(fig[2, length(protocols)+1],
		limits=(-200, 500), colormap=:grays, label="HU")

	Label(fig[0, :], "Reconstruction Comparison: Methods × Protocols", fontsize=18)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000032
md"""
## 12. Quantitative Analysis

Measure HU accuracy and noise in the solid water region for each reconstruction.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000033
begin
	# Get water mask at output resolution
	# We'll use the center region as a proxy for water
	center = RECON_SIZE ÷ 2
	roi_size = RECON_SIZE ÷ 8
	roi_slice = RECON_SLICES ÷ 2

	# Results table
	results = []

	for protocol in protocols
		for (method, recon_dict) in [("FDK", recon_fdk), ("SIRT", recon_sirt), ("CGLS", recon_cgls)]
			recon = recon_dict[protocol.name]

			# Extract center ROI (should be near water)
			roi = recon[center-roi_size:center+roi_size, center-roi_size:center+roi_size, roi_slice]
			roi_HU = μ_to_HU(roi, μ_water_ref)

			mean_HU = mean(roi_HU)
			std_HU = std(roi_HU)

			push!(results, (
				protocol=protocol.name,
				kvp=protocol.kvp,
				mas=protocol.mas,
				method=method,
				mean_HU=round(mean_HU, digits=1),
				noise_HU=round(std_HU, digits=1)
			))
		end
	end

	# Display as markdown table
	md"""
	### Center ROI Statistics (Near Water Region)

	| Protocol | kVp | mAs | Method | Mean HU | Noise (σ) |
	|----------|-----|-----|--------|---------|-----------|
	$(join(["| $(r.protocol) | $(r.kvp) | $(r.mas) | $(r.method) | $(r.mean_HU) | $(r.noise_HU) |" for r in results], "\n"))

	**Expected:** Water should be near 0 HU. Lower noise with higher mAs.
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000034
md"""
## 13. Summary

This notebook demonstrated:

1. **High-resolution phantom creation** - Physical object at finer resolution than CT voxels
2. **Clinical scanner configuration** - GE Revolution Apex Elite with FDA-sourced parameters
3. **Multiple acquisition protocols** - Low dose (80 kVp), standard (120 kVp), high dose (140 kVp)
4. **Full physics simulation** - Polychromatic X-rays, filters, scatter, noise
5. **Three reconstruction methods**:
   - **FDK**: Fast analytical reconstruction
   - **SIRT**: Robust iterative, slower convergence
   - **CGLS**: Fast convergence, may amplify noise
6. **Reactant compilation** - XLA compilation for performance

### Key Observations

- **Noise decreases with mAs**: Higher photon flux reduces quantum noise
- **Beam hardening varies with kVp**: Lower kVp shows more beam hardening artifacts
- **Iterative methods**: Can reduce noise compared to FDK at the cost of computation time
- **CGLS vs SIRT**: CGLS converges faster but may be less stable

---
*Generated with BasisSimulator.jl - A differentiable CT simulator*
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000001
# ╟─00000001-0000-0000-0000-000000000002
# ╟─00000001-0000-0000-0000-000000000003
# ╟─00000001-0000-0000-0000-000000000004
# ╠═00000001-0000-0000-0000-000000000005
# ╟─00000001-0000-0000-0000-000000000006
# ╠═00000001-0000-0000-0000-000000000007
# ╠═00000001-0000-0000-0000-000000000008
# ╟─00000001-0000-0000-0000-000000000009
# ╠═00000001-0000-0000-0000-000000000010
# ╟─00000001-0000-0000-0000-000000000011
# ╠═00000001-0000-0000-0000-000000000012
# ╟─00000001-0000-0000-0000-000000000013
# ╠═00000001-0000-0000-0000-000000000014
# ╟─00000001-0000-0000-0000-000000000015
# ╠═00000001-0000-0000-0000-000000000016
# ╠═00000001-0000-0000-0000-000000000017
# ╟─00000001-0000-0000-0000-000000000018
# ╠═00000001-0000-0000-0000-000000000019
# ╠═00000001-0000-0000-0000-000000000020
# ╟─00000001-0000-0000-0000-000000000021
# ╠═00000001-0000-0000-0000-000000000022
# ╠═00000001-0000-0000-0000-000000000023
# ╟─00000001-0000-0000-0000-000000000024
# ╠═00000001-0000-0000-0000-000000000025
# ╠═00000001-0000-0000-0000-000000000026
# ╟─00000001-0000-0000-0000-000000000027
# ╠═00000001-0000-0000-0000-000000000028
# ╠═00000001-0000-0000-0000-000000000029
# ╟─00000001-0000-0000-0000-000000000030
# ╠═00000001-0000-0000-0000-000000000031
# ╟─00000001-0000-0000-0000-000000000032
# ╠═00000001-0000-0000-0000-000000000033
# ╟─00000001-0000-0000-0000-000000000034
