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
# BasisSimulator.jl - XLA/Reactant CT Simulation

This notebook demonstrates **XLA-compiled CT simulation** using Reactant.

## Key Features

1. **XLA-Only Architecture** - All forward projection and reconstruction use XLA-compiled functions
2. **Reactant Compilation** - Functions are compiled once, then executed fast
3. **Pre-computed Geometry** - Ray indices computed once, reused for all projections

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Pre-compute Geometry (ONCE)                                  │
│   - precompute_forward_projection_geometry()                 │
│   - precompute_backprojection_geometry()                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Convert to Reactant Arrays                                   │
│   - Reactant.to_rarray(Float32.(data))                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Compile (ONCE)                                               │
│   - @compile forward_project_xla_arrays(...)                │
│   - @compile backproject_volume_arrays(...)                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Execute (FAST - reuse compiled function)                    │
│   - compiled_fp(vol_ra, idx_ra, wts_ra)                     │
│   - compiled_bp(sino_ra, bp_idx_ra, ...)                    │
└─────────────────────────────────────────────────────────────┘
```
"""

# ╔═╡ 00000001-0000-0000-0000-000000000004
md"""
## 1. Setup: Phantom and Geometry
"""

# ╔═╡ 00000001-0000-0000-0000-000000000005
begin
	# Create Gammex 472 phantom (default creates 7 z-slices for 64 voxels)
	phantom = create_gammex_472(n_voxels=64)

	# Create scanner geometry (Canon Aquilion ONE style)
	# n_rows=8 provides good cone-beam coverage for this phantom size
	geom = create_aquilion_one(
		n_angles=180,
		n_rows=8,
		n_cols=128,
		fov_cm=phantom.fov[1]
	)

	output_size = size(phantom.μ)
	μ_water_ref = get_reference_μ_water(60.0)
	mid_slice = output_size[3] ÷ 2 + 1  # Central slice (avoid edge effects)

	md"""
	**Phantom:** $(size(phantom.μ)), FOV: $(phantom.fov) cm

	**Geometry:** $(geom.n_angles) angles, $(geom.n_cols)×$(geom.n_rows) detector

	**SAD/SDD:** $(geom.SAD) / $(geom.SDD) cm
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000006
let
	fig = Figure(size=(800, 300))

	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Region Labels")
	heatmap!(ax1, Float64.(phantom.mask[:, :, mid_slice])'; colormap=:tab20)
	hidedecorations!(ax1)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Attenuation μ (cm⁻¹)")
	hm = heatmap!(ax2, phantom.μ[:, :, mid_slice]'; colormap=:viridis)
	Colorbar(fig[1, 3], hm)
	hidedecorations!(ax2)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000007
md"""
---
## 2. XLA Forward Projection with Reactant

### Step 1: Pre-compute Geometry (ONCE)

This computes all ray-volume intersection indices. Memory-intensive but done only once.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000008
begin
	proj_geom = precompute_forward_projection_geometry(geom, size(phantom.μ), phantom.fov)

	mem_fp_gb = (sizeof(proj_geom.linear_indices) + sizeof(proj_geom.sample_weights)) / 1e9

	md"""
	**Forward Projection Geometry:**
	- Samples per ray: $(proj_geom.n_samples)
	- Step size: $(proj_geom.step_size) cm
	- Linear indices shape: $(size(proj_geom.linear_indices))
	- Memory: $(round(mem_fp_gb, digits=3)) GB
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000009
md"""
### Step 2: Convert to Reactant Arrays

All arrays must be converted to `TracedRArray` for XLA compilation.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000010
begin
	vol_ra = Reactant.to_rarray(Float32.(vec(phantom.μ)))
	proj_idx_ra = Reactant.to_rarray(proj_geom.linear_indices)
	proj_wts_ra = Reactant.to_rarray(proj_geom.sample_weights)

	md"""
	**Reactant Arrays Created:**
	- `vol_ra`: $(typeof(vol_ra)), $(size(vol_ra))
	- `proj_idx_ra`: $(typeof(proj_idx_ra)), $(size(proj_idx_ra))
	- `proj_wts_ra`: $(typeof(proj_wts_ra)), $(size(proj_wts_ra))
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000011
md"""
### Step 3: Compile with @compile (ONCE)

This compiles the forward projection function to XLA/MLIR. Takes a few seconds, but only done once.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000012
begin
	compile_start = time()
	compiled_fp = @compile forward_project_xla_arrays(vol_ra, proj_idx_ra, proj_wts_ra)
	compile_time_fp = time() - compile_start

	md"""
	**✓ Forward projection compiled!**

	Compilation time: $(round(compile_time_fp, digits=2)) seconds

	Type: $(typeof(compiled_fp))
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000013
md"""
### Step 4: Execute Compiled Function (FAST)

Now we can run the compiled function multiple times - execution is very fast!
"""

# ╔═╡ 00000001-0000-0000-0000-000000000014
begin
	# Run forward projection
	exec_start = time()
	sino_ra = compiled_fp(vol_ra, proj_idx_ra, proj_wts_ra)
	sino = Array(sino_ra)
	exec_time_fp = time() - exec_start

	md"""
	**✓ Forward projection executed!**

	- Execution time: $(round(exec_time_fp * 1000, digits=2)) ms
	- Sinogram shape: $(size(sino))
	- Value range: [$(round(minimum(sino), digits=4)), $(round(maximum(sino), digits=4))]
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000015
let
	fig = Figure(size=(900, 350))

	mid_row = geom.n_rows ÷ 2
	ax1 = Axis(fig[1, 1], xlabel="Detector Column", ylabel="Angle",
		title="Sinogram (Central Row)")
	heatmap!(ax1, sino[:, mid_row, :]'; colormap=:hot)

	ax2 = Axis(fig[1, 2], xlabel="Detector Column", ylabel="Detector Row",
		title="Sinogram (First Angle)")
	heatmap!(ax2, sino[:, :, 1]'; colormap=:hot)

	ax3 = Axis(fig[1, 3], xlabel="Angle", ylabel="Detector Row",
		title="Sinogram (Central Column)")
	mid_col = geom.n_cols ÷ 2
	hm = heatmap!(ax3, sino[mid_col, :, :]'; colormap=:hot)
	Colorbar(fig[1, 4], hm)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000016
md"""
---
## 3. XLA Backprojection / FDK Reconstruction

### Step 1: Pre-compute Backprojection Geometry
"""

# ╔═╡ 00000001-0000-0000-0000-000000000017
begin
	bp_geom = precompute_backprojection_geometry(geom, output_size, phantom.fov)

	mem_bp_gb = (sizeof(bp_geom.linear_indices) + sizeof(bp_geom.bilinear_weights) +
		sizeof(bp_geom.distance_weights)) / 1e9

	md"""
	**Backprojection Geometry:**
	- Linear indices shape: $(size(bp_geom.linear_indices))
	- Bilinear weights shape: $(size(bp_geom.bilinear_weights))
	- Memory: $(round(mem_bp_gb, digits=3)) GB
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000018
md"""
### Step 2: Pre-weight and Filter Sinogram

FDK requires cosine weighting and ramp filtering before backprojection.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000019
begin
	weighted = preweight_cosine(sino, geom)
	filtered = filter_ramp(weighted, geom)
	sino_flat = Float32.(vec(filtered))

	md"""
	**Sinogram Prepared:**
	- Cosine pre-weighting applied
	- Ramp filtering applied
	- Flattened for backprojection: $(length(sino_flat)) elements
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000020
md"""
### Step 3: Convert to Reactant Arrays and Compile
"""

# ╔═╡ 00000001-0000-0000-0000-000000000021
begin
	sino_flat_ra = Reactant.to_rarray(sino_flat)
	bp_idx_ra = Reactant.to_rarray(bp_geom.linear_indices)
	bp_bilin_ra = Reactant.to_rarray(Float32.(bp_geom.bilinear_weights))
	bp_dist_ra = Reactant.to_rarray(Float32.(bp_geom.distance_weights))

	compile_start_bp = time()
	compiled_bp = @compile backproject_volume_arrays(sino_flat_ra, bp_idx_ra, bp_bilin_ra, bp_dist_ra)
	compile_time_bp = time() - compile_start_bp

	md"""
	**✓ Backprojection compiled!**

	Compilation time: $(round(compile_time_bp, digits=2)) seconds
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000022
md"""
### Step 4: Execute Backprojection
"""

# ╔═╡ 00000001-0000-0000-0000-000000000023
begin
	exec_start_bp = time()
	recon_ra = compiled_bp(sino_flat_ra, bp_idx_ra, bp_bilin_ra, bp_dist_ra)
	recon_flat = Array(recon_ra)
	exec_time_bp = time() - exec_start_bp

	# Reshape and apply FDK scaling
	magnification = geom.SDD / geom.SAD
	fdk_scale = Float32(magnification * 2.26)  # Calibrated scale factor
	recon = reshape(recon_flat .* fdk_scale, output_size)

	md"""
	**✓ Backprojection executed!**

	- Execution time: $(round(exec_time_bp * 1000, digits=2)) ms
	- Reconstruction shape: $(size(recon))
	- Value range: [$(round(minimum(recon), digits=4)), $(round(maximum(recon), digits=4))]
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000024
let
	recon_HU = μ_to_HU(recon, μ_water_ref)
	phantom_HU = μ_to_HU(phantom.μ, μ_water_ref)

	fig = Figure(size=(1000, 350))

	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Ground Truth (HU)")
	hm1 = heatmap!(ax1, phantom_HU[:, :, mid_slice]'; colormap=:grays, colorrange=(-200, 400))
	hidedecorations!(ax1)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="XLA Reconstruction (HU)")
	hm2 = heatmap!(ax2, recon_HU[:, :, mid_slice]'; colormap=:grays, colorrange=(-200, 400))
	Colorbar(fig[1, 3], hm2, label="HU")
	hidedecorations!(ax2)

	diff = recon_HU[:, :, mid_slice] .- phantom_HU[:, :, mid_slice]
	ax3 = Axis(fig[1, 4], aspect=DataAspect(), title="Difference (ΔHU)")
	hm3 = heatmap!(ax3, diff'; colormap=:RdBu, colorrange=(-100, 100))
	Colorbar(fig[1, 5], hm3, label="ΔHU")
	hidedecorations!(ax3)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000025
md"""
---
## 4. Verify HU Values

Check that water region reconstructs to approximately 0 HU.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000026
begin
	recon_HU = μ_to_HU(recon, μ_water_ref)

	# Get water region mask
	water_mask_2d = phantom.mask[:, :, mid_slice] .== UInt8(REGION_SOLID_WATER)
	water_HU_values = recon_HU[:, :, mid_slice][water_mask_2d]

	water_mean = mean(water_HU_values)
	water_std = std(water_HU_values)

	# Get Ca100 region
	ca100_mask_2d = phantom.mask[:, :, mid_slice] .== UInt8(REGION_CA_100)
	ca100_HU_values = recon_HU[:, :, mid_slice][ca100_mask_2d]
	ca100_mean = sum(ca100_mask_2d) > 0 ? mean(ca100_HU_values) : NaN

	md"""
	### HU Verification (Central Slice)

	| Material | Measured HU | Expected HU | Status |
	|----------|-------------|-------------|--------|
	| Water | $(round(water_mean, digits=1)) ± $(round(water_std, digits=1)) | 0 | $(abs(water_mean) < 50 ? "✓ PASS" : "✗ FAIL") |
	| Ca 100 | $(round(ca100_mean, digits=1)) | ~375 | $(200 < ca100_mean < 600 ? "✓ PASS" : "✗ FAIL") |

	**Water HU accuracy:** $(round(abs(water_mean), digits=1)) HU from expected (tolerance: ±50 HU)
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000027
md"""
---
## 5. Reuse Compiled Functions (FAST)

Once compiled, the functions can be reused with different inputs.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000028
begin
	# Create a modified volume (add contrast)
	vol_contrast = phantom.μ .* 1.2  # 20% increase
	vol_contrast_ra = Reactant.to_rarray(Float32.(vec(vol_contrast)))

	# Run forward projection with same compiled function
	reuse_start = time()
	sino_contrast_ra = compiled_fp(vol_contrast_ra, proj_idx_ra, proj_wts_ra)
	sino_contrast = Array(sino_contrast_ra)
	reuse_time = time() - reuse_start

	md"""
	**✓ Reused compiled forward projection!**

	- Execution time: $(round(reuse_time * 1000, digits=2)) ms
	- Same compiled function, new volume data
	- No recompilation needed!
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000029
md"""
---
## 6. High-Level API (Uses XLA Internally)

For convenience, use `simulate_sinogram()` and `reconstruct()` which handle XLA compilation internally.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000030
begin
	# High-level API - ideal monochromatic simulation
	sino_api = simulate_sinogram(phantom, geom;
		polychromatic=false,
		flat_filter=nothing, bowtie_filter=nothing, scatter=nothing,
		detector=nothing, crosstalk=nothing, lag=nothing,
		optical_crosstalk=nothing, fill_factor=nothing, focal_spot=nothing
	)

	recon_api = reconstruct(sino_api, geom, output_size, phantom.fov)

	md"""
	**High-Level API Results:**

	- `simulate_sinogram()`: $(size(sino_api))
	- `reconstruct()`: $(size(recon_api))

	These functions use XLA internally - same performance, simpler interface.
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000031
md"""
---
## 7. With Physical Effects

Add realistic physical effects (noise, scatter, etc.) using the high-level API.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000032
begin
	# Realistic simulation with noise
	sino_noisy = simulate_sinogram(phantom, geom;
		polychromatic=false,
		scatter=DEFAULT_SCATTER_MODEL,
		detector=default_detector_model(I0=1e5, blur_fwhm=0.8),
		seed=42
	)

	recon_noisy = reconstruct(sino_noisy, geom, output_size, phantom.fov)

	md"""
	**Realistic Simulation (with scatter + noise):**

	- Scatter model: `DEFAULT_SCATTER_MODEL`
	- Detector: I₀=1e5 photons, blur FWHM=0.8
	"""
end

# ╔═╡ 00000001-0000-0000-0000-000000000033
let
	recon_api_HU = μ_to_HU(recon_api, μ_water_ref)
	recon_noisy_HU = μ_to_HU(recon_noisy, μ_water_ref)

	fig = Figure(size=(800, 350))

	ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Ideal (No Effects)")
	hm1 = heatmap!(ax1, recon_api_HU[:, :, mid_slice]'; colormap=:grays, colorrange=(-200, 400))
	hidedecorations!(ax1)

	ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="With Scatter + Noise")
	hm2 = heatmap!(ax2, recon_noisy_HU[:, :, mid_slice]'; colormap=:grays, colorrange=(-200, 400))
	Colorbar(fig[1, 3], hm2, label="HU")
	hidedecorations!(ax2)

	fig
end

# ╔═╡ 00000001-0000-0000-0000-000000000034
md"""
---
## 8. Performance Summary

| Operation | Time |
|-----------|------|
| Forward projection geometry | Pre-computed |
| Forward projection compile | $(round(compile_time_fp, digits=2)) s |
| Forward projection execute | $(round(exec_time_fp * 1000, digits=2)) ms |
| Backprojection compile | $(round(compile_time_bp, digits=2)) s |
| Backprojection execute | $(round(exec_time_bp * 1000, digits=2)) ms |
| Reuse compiled (forward) | $(round(reuse_time * 1000, digits=2)) ms |

**Key Insight:** Compilation is expensive (seconds), but execution is fast (milliseconds).
For iterative algorithms or training loops, compile once and reuse!
"""

# ╔═╡ 00000001-0000-0000-0000-000000000035
md"""
---
## 9. Summary: XLA Workflow

### For Direct XLA Control:
```julia
using BasisSimulator
using Reactant

# 1. Pre-compute geometry (ONCE)
proj_geom = precompute_forward_projection_geometry(geom, vol_size, fov)
bp_geom = precompute_backprojection_geometry(geom, output_size, fov)

# 2. Convert to Reactant arrays
vol_ra = Reactant.to_rarray(Float32.(vec(volume)))
idx_ra = Reactant.to_rarray(proj_geom.linear_indices)
wts_ra = Reactant.to_rarray(proj_geom.sample_weights)

# 3. Compile (ONCE)
compiled_fp = @compile forward_project_xla_arrays(vol_ra, idx_ra, wts_ra)

# 4. Execute (FAST - reuse!)
sino_ra = compiled_fp(vol_ra, idx_ra, wts_ra)
sino = Array(sino_ra)
```

### For High-Level API:
```julia
using BasisSimulator

# Forward projection with effects
sino = simulate_sinogram(phantom, geom;
    polychromatic=false,
    scatter=DEFAULT_SCATTER_MODEL,
    detector=default_detector_model(I0=1e5)
)

# Reconstruction
recon = reconstruct(sino, geom, output_size, phantom.fov)
```

Both approaches use XLA internally - choose based on your needs!
"""

# ╔═╡ Cell order:
# ╠═00000001-0000-0000-0000-000000000001
# ╠═00000001-0000-0000-0000-000000000002
# ╟─00000001-0000-0000-0000-000000000003
# ╟─00000001-0000-0000-0000-000000000004
# ╠═00000001-0000-0000-0000-000000000005
# ╟─00000001-0000-0000-0000-000000000006
# ╟─00000001-0000-0000-0000-000000000007
# ╠═00000001-0000-0000-0000-000000000008
# ╟─00000001-0000-0000-0000-000000000009
# ╠═00000001-0000-0000-0000-000000000010
# ╟─00000001-0000-0000-0000-000000000011
# ╠═00000001-0000-0000-0000-000000000012
# ╟─00000001-0000-0000-0000-000000000013
# ╠═00000001-0000-0000-0000-000000000014
# ╟─00000001-0000-0000-0000-000000000015
# ╟─00000001-0000-0000-0000-000000000016
# ╠═00000001-0000-0000-0000-000000000017
# ╟─00000001-0000-0000-0000-000000000018
# ╠═00000001-0000-0000-0000-000000000019
# ╟─00000001-0000-0000-0000-000000000020
# ╠═00000001-0000-0000-0000-000000000021
# ╟─00000001-0000-0000-0000-000000000022
# ╠═00000001-0000-0000-0000-000000000023
# ╟─00000001-0000-0000-0000-000000000024
# ╟─00000001-0000-0000-0000-000000000025
# ╠═00000001-0000-0000-0000-000000000026
# ╟─00000001-0000-0000-0000-000000000027
# ╠═00000001-0000-0000-0000-000000000028
# ╟─00000001-0000-0000-0000-000000000029
# ╠═00000001-0000-0000-0000-000000000030
# ╟─00000001-0000-0000-0000-000000000031
# ╠═00000001-0000-0000-0000-000000000032
# ╟─00000001-0000-0000-0000-000000000033
# ╟─00000001-0000-0000-0000-000000000034
# ╟─00000001-0000-0000-0000-000000000035
